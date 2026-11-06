#include "spectral.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>

#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/MatOp/SparseSymShiftSolve.h>
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/SymEigsSolver.h>
#include <opencv2/core.hpp>

namespace seg {

namespace {

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

const char* to_string(SpectralStatus s) {
    switch (s) {
        case SpectralStatus::Ok: return "ok";
        case SpectralStatus::InvalidK: return "invalid_k";
        case SpectralStatus::ZeroDegreeNode: return "zero_degree_node";
        case SpectralStatus::DisconnectedGraph: return "disconnected_graph";
        case SpectralStatus::EigenNotConverged: return "eigen_not_converged";
        case SpectralStatus::DegenerateEmbedding: return "degenerate_embedding";
    }
    return "unknown";
}

SparseMatrix normalized_affinity(const SparseMatrix& w) {
    const Eigen::VectorXd d = node_degrees(w);
    if ((d.array() <= 0.0).any()) {
        throw std::invalid_argument("normalized_affinity: zero-degree node");
    }
    const Eigen::VectorXd inv_sqrt = d.array().rsqrt();
    SparseMatrix m = inv_sqrt.asDiagonal() * w * inv_sqrt.asDiagonal();
    m.makeCompressed();
    return m;
}

EigenResult leading_eigenpairs(const SparseMatrix& m, int k, int max_iter, double tol, bool shift_invert) {
    const auto n = static_cast<int>(m.rows());
    if (k < 1 || k >= n) throw std::invalid_argument("leading_eigenpairs: need 1 <= k < n");

    EigenResult r;
    const int ncv_tries[2] = {std::min(n, std::max(2 * k + 1, 20)), std::min(n, std::max(4 * k + 1, 60))};
    const int iter_tries[2] = {max_iter, 5 * max_iter};
    auto record = [&r](auto& eigs, int ncv, int attempt) {
        r.ncv = ncv;
        r.iterations = static_cast<int>(eigs.num_iterations());
        r.matvecs = static_cast<int>(eigs.num_operations());
        r.retried = attempt > 0;
        if (eigs.info() == Spectra::CompInfo::Successful) {
            r.converged = true;
            r.values = eigs.eigenvalues();
            r.vectors = eigs.eigenvectors();
        }
    };
    for (int attempt = 0; attempt < 2 && !r.converged; ++attempt) {
        const int ncv = std::max(ncv_tries[attempt], std::min(n, k + 2));
        if (shift_invert) {
            // sigma just above lambda_max = 1 maps eigenvalues near 1 to 1/(lambda - sigma),
            // far apart in magnitude, so a clustered top of the spectrum converges quickly.
            constexpr double kShift = 1e-2;
            try {
                Spectra::SparseSymShiftSolve<double> op(m);
                Spectra::SymEigsShiftSolver<Spectra::SparseSymShiftSolve<double>> eigs(op, k, ncv, 1.0 + kShift);
                eigs.init();
                eigs.compute(Spectra::SortRule::LargestMagn, iter_tries[attempt], tol, Spectra::SortRule::LargestAlge);
                record(eigs, ncv, attempt);
                r.shift_invert = true;
                continue;
            } catch (const std::exception&) {
                // Sparse LU failed; fall back to plain Lanczos.
            }
        }
        Spectra::SparseSymMatProd<double> op(m);
        Spectra::SymEigsSolver<Spectra::SparseSymMatProd<double>> eigs(op, k, ncv);
        eigs.init();
        eigs.compute(Spectra::SortRule::LargestAlge, iter_tries[attempt], tol, Spectra::SortRule::LargestAlge);
        record(eigs, ncv, attempt);
        r.shift_invert = false;
    }
    return r;
}

double orthogonality_error(const Eigen::MatrixXd& u) {
    const Eigen::MatrixXd gram = u.transpose() * u;
    return (gram - Eigen::MatrixXd::Identity(u.cols(), u.cols())).norm();
}

double eigenpair_residual(const SparseMatrix& m, double lambda, const Eigen::VectorXd& v) {
    const Eigen::VectorXd res = m * v - lambda * v;
    return res.norm() / (std::max(1.0, std::abs(lambda)) * v.norm());
}

Eigen::MatrixXd row_normalize(const Eigen::MatrixXd& u, double eps, int* degenerate_rows) {
    Eigen::MatrixXd t = u;
    int degenerate = 0;
    for (Eigen::Index i = 0; i < t.rows(); ++i) {
        const double norm = t.row(i).norm();
        if (norm < eps) {
            t.row(i).setZero();
            ++degenerate;
        } else {
            t.row(i) /= norm;
        }
    }
    if (degenerate_rows != nullptr) *degenerate_rows = degenerate;
    return t;
}

KMeansResult kmeans_rows(const Eigen::MatrixXd& t, int k, const SpectralParams& params) {
    const auto n = static_cast<int>(t.rows());
    const auto dim = static_cast<int>(t.cols());
    cv::Mat data(n, dim, CV_32F);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < dim; ++j) data.at<float>(i, j) = static_cast<float>(t(i, j));
    }
    cv::Mat labels;
    cv::Mat centers;
    cv::theRNG().state = params.seed;  // k-means++ seeding uses the thread-local RNG
    KMeansResult r;
    r.inertia = cv::kmeans(data, k, labels,
                           cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
                                            params.kmeans_max_iter, 1e-6),
                           params.kmeans_attempts, cv::KMEANS_PP_CENTERS, centers);
    r.labels.assign(labels.begin<int>(), labels.end<int>());
    std::vector<int> sizes(static_cast<std::size_t>(k), 0);
    for (int l : r.labels) ++sizes[static_cast<std::size_t>(l)];
    r.nonempty_clusters = static_cast<int>(std::count_if(sizes.begin(), sizes.end(), [](int s) { return s > 0; }));
    return r;
}

SpectralResult spectral_cluster(const SparseMatrix& w, const SpectralParams& params) {
    SpectralResult r;
    const auto n = static_cast<int>(w.rows());
    const auto fail = [&r](SpectralStatus s, std::string msg) {
        r.status = s;
        r.message = std::move(msg);
        return r;
    };
    if (params.k < 2 || params.k >= n) {
        return fail(SpectralStatus::InvalidK,
                    "k=" + std::to_string(params.k) + " invalid for n=" + std::to_string(n) + " nodes (need 2 <= k < n)");
    }
    const Eigen::VectorXd degrees = node_degrees(w);
    const auto zero_deg = (degrees.array() <= 0.0).count();
    if (zero_deg > 0) {
        return fail(SpectralStatus::ZeroDegreeNode, std::to_string(zero_deg) + " node(s) have zero degree");
    }
    // c components give eigenvalue 1 with multiplicity c: Lanczos cannot resolve it and the
    // eigenvectors are only defined up to rotation, so refuse.
    r.components = connected_components(w);
    if (r.components > 1) {
        return fail(SpectralStatus::DisconnectedGraph,
                    "graph has " + std::to_string(r.components) + " connected components");
    }
    const SparseMatrix m = normalized_affinity(w);

    auto t0 = std::chrono::steady_clock::now();
    // One extra pair for the gap l_k - l_{k+1}; Spectra needs nev < n.
    const int nev = params.k + 1 < n ? params.k + 1 : params.k;
    r.eig = leading_eigenpairs(m, nev, params.eig_max_iter, params.eig_tol, params.shift_invert);
    r.eigen_ms = ms_since(t0);
    if (!r.eig.converged) {
        return fail(SpectralStatus::EigenNotConverged,
                    "Spectra did not converge (ncv=" + std::to_string(r.eig.ncv) + ")");
    }

    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < nev; ++i) {
        r.max_residual = std::max(r.max_residual, eigenpair_residual(m, r.eig.values(i), r.eig.vectors.col(i)));
    }
    const Eigen::MatrixXd u = r.eig.vectors.leftCols(params.k);
    r.orthogonality_error = orthogonality_error(u);
    if (nev == params.k + 1) r.spectral_gap = r.eig.values(params.k - 1) - r.eig.values(params.k);
    r.residual_ms = ms_since(t0);

    t0 = std::chrono::steady_clock::now();
    const Eigen::MatrixXd t = row_normalize(u, 1e-12, &r.degenerate_rows);
    if (r.degenerate_rows == n) {
        r.kmeans_ms = ms_since(t0);
        return fail(SpectralStatus::DegenerateEmbedding, "every embedding row has ~zero norm");
    }
    r.kmeans = kmeans_rows(t, params.k, params);
    r.kmeans_ms = ms_since(t0);
    if (r.kmeans.nonempty_clusters < params.k) {
        return fail(SpectralStatus::DegenerateEmbedding,
                    "k-means produced only " + std::to_string(r.kmeans.nonempty_clusters) + " non-empty clusters");
    }
    return r;
}

SparseMatrix induced_subgraph(const SparseMatrix& w, const std::vector<int>& nodes) {
    std::vector<int> local(static_cast<std::size_t>(w.rows()), -1);
    for (std::size_t i = 0; i < nodes.size(); ++i) local[static_cast<std::size_t>(nodes[i])] = static_cast<int>(i);
    std::vector<Eigen::Triplet<double>> t;
    for (std::size_t j = 0; j < nodes.size(); ++j) {
        for (SparseMatrix::InnerIterator it(w, nodes[j]); it; ++it) {
            const int i = local[static_cast<std::size_t>(it.row())];
            if (i >= 0) t.emplace_back(i, static_cast<int>(j), it.value());
        }
    }
    const auto n = static_cast<Eigen::Index>(nodes.size());
    SparseMatrix sub(n, n);
    sub.setFromTriplets(t.begin(), t.end());
    sub.makeCompressed();
    return sub;
}

double ncut_value(const SparseMatrix& w, const std::vector<int>& part) {
    const int parts = part.empty() ? 0 : *std::max_element(part.begin(), part.end()) + 1;
    std::vector<double> cut(static_cast<std::size_t>(parts), 0.0);
    std::vector<double> vol(static_cast<std::size_t>(parts), 0.0);
    for (int c = 0; c < w.outerSize(); ++c) {
        const auto pc = static_cast<std::size_t>(part[static_cast<std::size_t>(c)]);
        for (SparseMatrix::InnerIterator it(w, c); it; ++it) {
            vol[pc] += it.value();
            if (static_cast<std::size_t>(part[static_cast<std::size_t>(it.row())]) != pc) cut[pc] += it.value();
        }
    }
    double ncut = 0.0;
    for (std::size_t i = 0; i < cut.size(); ++i) {
        if (vol[i] <= 0.0) return std::numeric_limits<double>::infinity();
        ncut += cut[i] / vol[i];
    }
    return ncut;
}

RecursiveResult recursive_spectral_cluster(const SparseMatrix& w, const RecursiveParams& params) {
    RecursiveResult r;
    const auto n = static_cast<int>(w.rows());
    if (params.max_segments < 2 || params.max_segments > n) {
        r.status = SpectralStatus::InvalidK;
        r.message = "max segments k=" + std::to_string(params.max_segments) + " invalid for n=" +
                    std::to_string(n) + " nodes (need 2 <= k <= n)";
        return r;
    }
    const auto zero_deg = (node_degrees(w).array() <= 0.0).count();
    if (zero_deg > 0) {
        r.status = SpectralStatus::ZeroDegreeNode;
        r.message = std::to_string(zero_deg) + " node(s) have zero degree";
        return r;
    }
    r.components = connected_components(w);

    SpectralParams sp = params.split;
    sp.k = 2;
    std::vector<std::vector<int>> finals;
    // Largest segment first; ties by first node for determinism.
    using Item = std::pair<std::pair<std::size_t, int>, std::vector<int>>;
    auto cmp = [](const Item& a, const Item& b) {
        if (a.first.first != b.first.first) return a.first.first < b.first.first;
        return a.first.second > b.first.second;
    };
    std::priority_queue<Item, std::vector<Item>, decltype(cmp)> queue(cmp);
    auto push = [&queue](std::vector<int> nodes) {
        const std::size_t size = nodes.size();
        const int first = nodes.front();
        queue.push({{size, first}, std::move(nodes)});
    };
    std::vector<int> all(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) all[static_cast<std::size_t>(i)] = i;
    push(std::move(all));
    bool root_done = false;

    while (!queue.empty() && static_cast<int>(finals.size() + queue.size()) < params.max_segments) {
        std::vector<int> nodes = queue.top().second;
        queue.pop();
        const SparseMatrix sub = induced_subgraph(w, nodes);

        std::vector<int> comp;
        const int nc = connected_components(sub, &comp);
        if (nc > 1 && static_cast<int>(finals.size() + queue.size()) + nc <= params.max_segments) {
            std::vector<std::vector<int>> parts(static_cast<std::size_t>(nc));
            for (std::size_t i = 0; i < nodes.size(); ++i) parts[static_cast<std::size_t>(comp[i])].push_back(nodes[i]);
            ++r.component_splits;
            for (auto& part : parts) push(std::move(part));
            continue;
        }
        if (nc > 1 || static_cast<int>(nodes.size()) < std::max(3, 2 * params.min_segment_nodes)) {
            finals.push_back(std::move(nodes));
            continue;
        }
        if (params.weak_edge > 0.0) {
            SparseMatrix strong = sub.pruned(params.weak_edge, 1.0);
            std::vector<int> wcomp;
            const int wc = connected_components(strong, &wcomp);
            if (wc > 1 && static_cast<int>(finals.size() + queue.size()) + wc <= params.max_segments &&
                ncut_value(sub, wcomp) <= params.ncut_threshold) {
                std::vector<std::vector<int>> parts(static_cast<std::size_t>(wc));
                for (std::size_t i = 0; i < nodes.size(); ++i) parts[static_cast<std::size_t>(wcomp[i])].push_back(nodes[i]);
                ++r.weak_edge_splits;
                for (auto& part : parts) push(std::move(part));
                continue;
            }
        }
        const SpectralResult s = spectral_cluster(sub, sp);
        r.eigen_ms += s.eigen_ms;
        r.kmeans_ms += s.kmeans_ms;
        r.total_iterations += s.eig.iterations;
        r.total_matvecs += s.eig.matvecs;
        r.max_ncv = std::max(r.max_ncv, s.eig.ncv);
        r.any_retried = r.any_retried || s.eig.retried;
        if (!s.ok()) {
            ++r.failed_solves;
            finals.push_back(std::move(nodes));
            continue;
        }
        r.max_residual = std::max(r.max_residual, s.max_residual);
        r.max_orthogonality_error = std::max(r.max_orthogonality_error, s.orthogonality_error);
        if (!std::isnan(s.spectral_gap) && !(s.spectral_gap >= r.min_gap)) r.min_gap = s.spectral_gap;
        const double ncut = ncut_value(sub, s.kmeans.labels);
        if (!root_done) {
            r.root_eigenvalues = s.eig.values;
            r.root_gap = s.spectral_gap;
            r.root_inertia = s.kmeans.inertia;
            r.root_ncut = ncut;
            root_done = true;
        }
        std::vector<int> a, b;
        for (std::size_t i = 0; i < nodes.size(); ++i) (s.kmeans.labels[i] == 0 ? a : b).push_back(nodes[i]);
        if (ncut > params.ncut_threshold || static_cast<int>(std::min(a.size(), b.size())) < params.min_segment_nodes) {
            ++r.splits_rejected;
            finals.push_back(std::move(nodes));
            continue;
        }
        ++r.splits_accepted;
        push(std::move(a));
        push(std::move(b));
    }
    while (!queue.empty()) {
        finals.push_back(queue.top().second);
        queue.pop();
    }

    std::sort(finals.begin(), finals.end(),
              [](const std::vector<int>& x, const std::vector<int>& y) {
                  return *std::min_element(x.begin(), x.end()) < *std::min_element(y.begin(), y.end());
              });
    r.labels.assign(static_cast<std::size_t>(n), -1);
    for (std::size_t s = 0; s < finals.size(); ++s)
        for (int node : finals[s]) r.labels[static_cast<std::size_t>(node)] = static_cast<int>(s);
    r.segments = static_cast<int>(finals.size());
    return r;
}

}  // namespace seg
