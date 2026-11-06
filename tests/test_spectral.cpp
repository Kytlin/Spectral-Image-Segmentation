
#include <cmath>
#include <random>
#include <set>

#include <Eigen/Eigenvalues>
#include <gtest/gtest.h>

#include "spectral.hpp"

namespace {

seg::SparseMatrix from_edges(int n, const std::vector<std::tuple<int, int, double>>& edges) {
    std::vector<Eigen::Triplet<double>> t;
    for (const auto& [a, b, w] : edges) {
        t.emplace_back(a, b, w);
        t.emplace_back(b, a, w);
    }
    seg::SparseMatrix m(n, n);
    m.setFromTriplets(t.begin(), t.end());
    m.makeCompressed();
    return m;
}

// Connected random graph: a ring plus random chords, weights in [0.1, 1].
seg::SparseMatrix random_graph(int n, int chords, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> wdist(0.1, 1.0);
    std::uniform_int_distribution<int> node(0, n - 1);
    std::set<std::pair<int, int>> seen;
    std::vector<std::tuple<int, int, double>> e;
    auto add = [&](int a, int b) {
        if (a == b) return;
        const auto key = std::minmax(a, b);
        if (seen.insert(key).second) e.emplace_back(key.first, key.second, wdist(rng));
    };
    for (int i = 0; i < n; ++i) add(i, (i + 1) % n);
    for (int i = 0; i < chords; ++i) add(node(rng), node(rng));
    return from_edges(n, e);
}

// Two cliques of size m (strong weights) joined by one weak edge.
seg::SparseMatrix two_cliques(int m, double bridge) {
    std::vector<std::tuple<int, int, double>> e;
    for (int c = 0; c < 2; ++c)
        for (int i = 0; i < m; ++i)
            for (int j = i + 1; j < m; ++j) e.emplace_back(c * m + i, c * m + j, 1.0);
    if (bridge > 0) e.emplace_back(m - 1, m, bridge);
    return from_edges(2 * m, e);
}

}  // namespace

TEST(NormalizedAffinity, EntriesAreWOverSqrtDegrees) {
    const seg::SparseMatrix w = random_graph(30, 40, 1);
    const seg::SparseMatrix m = seg::normalized_affinity(w);
    const Eigen::VectorXd d = seg::node_degrees(w);
    ASSERT_EQ(m.rows(), 30);
    EXPECT_EQ(m.nonZeros(), w.nonZeros());
    for (int c = 0; c < w.outerSize(); ++c)
        for (seg::SparseMatrix::InnerIterator it(w, c); it; ++it)
            EXPECT_NEAR(m.coeff(it.row(), it.col()), it.value() / std::sqrt(d(it.row()) * d(it.col())), 1e-15);
    EXPECT_LT((Eigen::MatrixXd(m) - Eigen::MatrixXd(m).transpose()).norm(), 1e-15);
}

TEST(NormalizedAffinity, ZeroDegreeThrows) {
    const seg::SparseMatrix w = from_edges(3, {{0, 1, 1.0}});
    EXPECT_THROW(seg::normalized_affinity(w), std::invalid_argument);
}

TEST(Eigenpairs, SpectraMatchesDenseSolver) {
    for (unsigned seed : {1U, 2U, 3U}) {
        const seg::SparseMatrix m = seg::normalized_affinity(random_graph(80, 120, seed));
        const int k = 6;
        for (bool shift_invert : {true, false}) {
        const seg::EigenResult r = seg::leading_eigenpairs(m, k, 1000, 1e-12, shift_invert);
        ASSERT_TRUE(r.converged);
        ASSERT_EQ(r.values.size(), k);
        ASSERT_EQ(r.vectors.cols(), k);

        const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> dense{Eigen::MatrixXd(m)};
        const Eigen::Index n = m.rows();
        for (int i = 0; i < k; ++i) {
            const Eigen::Index j = n - 1 - i;
            EXPECT_NEAR(r.values(i), dense.eigenvalues()(j), 1e-10);
            // Same eigenvector up to sign (random graphs have simple eigenvalues).
            EXPECT_NEAR(std::abs(r.vectors.col(i).dot(dense.eigenvectors().col(j))), 1.0, 1e-8);
            EXPECT_LT(seg::eigenpair_residual(m, r.values(i), r.vectors.col(i)), 1e-9);
        }
        const Eigen::MatrixXd gram = r.vectors.transpose() * r.vectors;
        EXPECT_LT((gram - Eigen::MatrixXd::Identity(k, k)).norm(), 1e-10);
        // Connected graph: leading eigenvalue of M is exactly 1, eigenvector prop. to D^{1/2} 1.
        EXPECT_NEAR(r.values(0), 1.0, 1e-12);
        }
    }
}

TEST(Eigenpairs, ShiftInvertAndPlainLanczosAgree) {
    const seg::SparseMatrix m = seg::normalized_affinity(random_graph(120, 200, 9));
    const seg::EigenResult a = seg::leading_eigenpairs(m, 5, 1000, 1e-12, true);
    const seg::EigenResult b = seg::leading_eigenpairs(m, 5, 1000, 1e-12, false);
    ASSERT_TRUE(a.converged);
    ASSERT_TRUE(b.converged);
    EXPECT_TRUE(a.shift_invert);
    EXPECT_FALSE(b.shift_invert);
    for (int i = 0; i < 5; ++i) {
        EXPECT_NEAR(a.values(i), b.values(i), 1e-10);
        EXPECT_NEAR(std::abs(a.vectors.col(i).dot(b.vectors.col(i))), 1.0, 1e-8);
        EXPECT_LT(seg::eigenpair_residual(m, a.values(i), a.vectors.col(i)), 1e-9);
    }
}

TEST(Eigenpairs, ResidualIsScaleInvariantAndDetectsWrongPairs) {
    const seg::SparseMatrix m = seg::normalized_affinity(random_graph(40, 50, 4));
    const seg::EigenResult r = seg::leading_eigenpairs(m, 2, 1000, 1e-12);
    ASSERT_TRUE(r.converged);
    const Eigen::VectorXd v = r.vectors.col(1);
    EXPECT_NEAR(seg::eigenpair_residual(m, r.values(1), v), seg::eigenpair_residual(m, r.values(1), 5.0 * v), 1e-14);
    EXPECT_GT(seg::eigenpair_residual(m, r.values(1) + 0.1, v), 0.05);
}

TEST(RowNormalize, UnitRowsAndDegenerateRowsCounted) {
    Eigen::MatrixXd u(3, 2);
    u << 3, 4, 0, 0, -1, 1;
    int degenerate = -1;
    const Eigen::MatrixXd t = seg::row_normalize(u, 1e-12, &degenerate);
    EXPECT_EQ(degenerate, 1);
    EXPECT_NEAR(t.row(0).norm(), 1.0, 1e-15);
    EXPECT_NEAR(t(0, 0), 0.6, 1e-15);
    EXPECT_EQ(t.row(1).norm(), 0.0);
    EXPECT_NEAR(t.row(2).norm(), 1.0, 1e-15);
}

TEST(SpectralCluster, SeparatesWeaklyJoinedCliquesDeterministically) {
    const seg::SparseMatrix w = two_cliques(10, 1e-3);
    seg::SpectralParams p;
    p.k = 2;
    const seg::SpectralResult a = seg::spectral_cluster(w, p);
    ASSERT_TRUE(a.ok()) << a.message;
    EXPECT_EQ(a.components, 1);
    EXPECT_LT(a.max_residual, 1e-8);
    ASSERT_EQ(a.kmeans.labels.size(), 20U);
    for (int i = 0; i < 20; ++i) {
        EXPECT_GE(a.kmeans.labels[i], 0);
        EXPECT_LT(a.kmeans.labels[i], 2);
        EXPECT_EQ(a.kmeans.labels[i], a.kmeans.labels[i < 10 ? 0 : 10]);
    }
    EXPECT_NE(a.kmeans.labels[0], a.kmeans.labels[10]);
    EXPECT_EQ(a.kmeans.nonempty_clusters, 2);
    EXPECT_GE(a.kmeans.inertia, 0.0);

    const seg::SpectralResult b = seg::spectral_cluster(w, p);
    EXPECT_EQ(a.kmeans.labels, b.kmeans.labels);
    EXPECT_EQ(a.eig.values, b.eig.values);
}

TEST(SpectralCluster, ReportsFailuresInsteadOfAdjusting) {
    seg::SpectralParams p;
    const seg::SparseMatrix w = random_graph(20, 20, 5);
    p.k = 1;
    EXPECT_EQ(seg::spectral_cluster(w, p).status, seg::SpectralStatus::InvalidK);
    p.k = 20;
    EXPECT_EQ(seg::spectral_cluster(w, p).status, seg::SpectralStatus::InvalidK);

    p.k = 2;
    const seg::SparseMatrix isolated = from_edges(4, {{0, 1, 1.0}, {1, 2, 1.0}});
    EXPECT_EQ(seg::spectral_cluster(isolated, p).status, seg::SpectralStatus::ZeroDegreeNode);

    const seg::SpectralResult disc = seg::spectral_cluster(two_cliques(5, 0.0), p);
    EXPECT_EQ(disc.status, seg::SpectralStatus::DisconnectedGraph);
    EXPECT_EQ(disc.components, 2);
}

TEST(SpectralCluster, ReportsEigensolverNonconvergence) {
    // Ring: eigenvalues cos(2 pi j / n) cluster near 1, so a 1-iteration budget cannot converge.
    std::vector<std::tuple<int, int, double>> e;
    const int n = 400;
    for (int i = 0; i < n; ++i) e.emplace_back(i, (i + 1) % n, 1.0);
    seg::SpectralParams p;
    p.k = 4;
    p.eig_max_iter = 1;
    p.eig_tol = 1e-14;
    p.shift_invert = false;  // shift-invert converges even in 1 iteration here
    const seg::SpectralResult r = seg::spectral_cluster(from_edges(n, e), p);
    EXPECT_EQ(r.status, seg::SpectralStatus::EigenNotConverged);
    EXPECT_TRUE(r.eig.retried);
    EXPECT_FALSE(r.message.empty());
}

TEST(Ncut, MatchesHandComputedValueAndInducedSubgraph) {
    // Path 0-1-2-3 with weights 1, 0.5, 1; split {0,1} | {2,3}: cut 0.5, vol 2.5 each.
    const seg::SparseMatrix w = from_edges(4, {{0, 1, 1.0}, {1, 2, 0.5}, {2, 3, 1.0}});
    EXPECT_NEAR(seg::ncut_value(w, {0, 0, 1, 1}), 0.5 / 2.5 + 0.5 / 2.5, 1e-15);
    const seg::SparseMatrix sub = seg::induced_subgraph(w, {1, 2, 3});
    ASSERT_EQ(sub.rows(), 3);
    EXPECT_EQ(sub.coeff(0, 1), 0.5);
    EXPECT_EQ(sub.coeff(1, 2), 1.0);
    EXPECT_EQ(sub.coeff(0, 2), 0.0);
    EXPECT_EQ(sub.nonZeros(), 4);
}

TEST(Recursive, SplitsWeakBridgeButNotCliques) {
    seg::RecursiveParams p;
    p.max_segments = 10;
    const seg::RecursiveResult r = seg::recursive_spectral_cluster(two_cliques(10, 1e-3), p);
    ASSERT_TRUE(r.ok()) << r.message;
    EXPECT_EQ(r.segments, 2);  // cap allows 10, but splitting a clique has high Ncut
    EXPECT_GT(r.splits_rejected, 0);
    for (int i = 0; i < 20; ++i) EXPECT_EQ(r.labels[i], r.labels[i < 10 ? 0 : 10]);
    EXPECT_NE(r.labels[0], r.labels[10]);
    EXPECT_LT(r.max_residual, 1e-8);
}

TEST(Recursive, HandlesDisconnectedGraphsAndHonoursCap) {
    seg::RecursiveParams p;
    p.max_segments = 10;
    const seg::RecursiveResult r = seg::recursive_spectral_cluster(two_cliques(5, 0.0), p);
    ASSERT_TRUE(r.ok());
    EXPECT_EQ(r.components, 2);
    EXPECT_EQ(r.segments, 2);
    EXPECT_EQ(r.component_splits, 1);

    // Random graph with a permissive threshold: segment count never exceeds the cap.
    p.max_segments = 5;
    p.ncut_threshold = 10.0;
    const seg::RecursiveResult c = seg::recursive_spectral_cluster(random_graph(100, 150, 11), p);
    ASSERT_TRUE(c.ok());
    EXPECT_LE(c.segments, 5);
    for (int l : c.labels) {
        EXPECT_GE(l, 0);
        EXPECT_LT(l, c.segments);
    }
    const seg::RecursiveResult d = seg::recursive_spectral_cluster(random_graph(100, 150, 11), p);
    EXPECT_EQ(c.labels, d.labels);

    p.max_segments = 1;
    EXPECT_EQ(seg::recursive_spectral_cluster(two_cliques(5, 1e-3), p).status, seg::SpectralStatus::InvalidK);
}


namespace {

// Path graph P_n, unit weights. M = D^-1/2 W D^-1/2 is similar to the random-walk matrix of
// the path, whose eigenvalues are cos(pi j / (n - 1)), j = 0..n-1, all distinct.
seg::SparseMatrix path_graph(int n) {
    std::vector<std::tuple<int, int, double>> e;
    for (int i = 0; i + 1 < n; ++i) e.emplace_back(i, i + 1, 1.0);
    return from_edges(n, e);
}

}  // namespace

TEST(QualityMetrics, PathGraphGapMatchesClosedForm) {
    const double pi = std::acos(-1.0);
    const int n = 10;
    seg::SpectralParams p;
    p.k = 3;
    const seg::SpectralResult r = seg::spectral_cluster(path_graph(n), p);
    ASSERT_TRUE(r.ok()) << r.message;
    ASSERT_EQ(r.eig.values.size(), 4);
    for (int j = 0; j < 4; ++j) EXPECT_NEAR(r.eig.values(j), std::cos(pi * j / (n - 1)), 1e-10);
    EXPECT_NEAR(r.spectral_gap, std::cos(2 * pi / 9) - std::cos(3 * pi / 9), 1e-10);
    EXPECT_LT(r.orthogonality_error, 1e-10);
    EXPECT_LT(r.max_residual, 1e-9);

    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> dense{Eigen::MatrixXd(seg::normalized_affinity(path_graph(n)))};
    EXPECT_NEAR(r.spectral_gap, dense.eigenvalues()(n - 3) - dense.eigenvalues()(n - 4), 1e-10);
}

TEST(QualityMetrics, OrthogonalityErrorDefinition) {
    EXPECT_NEAR(seg::orthogonality_error(Eigen::MatrixXd::Identity(5, 3)), 0.0, 1e-15);
    Eigen::MatrixXd u = Eigen::MatrixXd::Identity(4, 2);
    u(0, 1) = 0.1;  // columns no longer orthogonal: off-diagonals 0.1, (1,1) entry 1.01
    EXPECT_NEAR(seg::orthogonality_error(u), std::sqrt(2 * 0.01 + 0.01 * 0.01), 1e-14);
}

TEST(QualityMetrics, GapUndefinedWhenKPlusOneEqualsN) {
    seg::SpectralParams p;
    p.k = 2;
    const seg::SpectralResult r = seg::spectral_cluster(path_graph(3), p);  // k < n but k + 1 == n
    ASSERT_TRUE(r.ok()) << r.message;
    EXPECT_EQ(r.eig.values.size(), 2);
    EXPECT_TRUE(std::isnan(r.spectral_gap));
    EXPECT_LT(r.orthogonality_error, 1e-10);

    const seg::SpectralResult disc = seg::spectral_cluster(two_cliques(5, 0.0), p);
    EXPECT_EQ(disc.status, seg::SpectralStatus::DisconnectedGraph);
    EXPECT_TRUE(std::isnan(disc.spectral_gap));
    EXPECT_TRUE(std::isnan(disc.orthogonality_error));
}

TEST(QualityMetrics, NcutOfKnownPartitions) {
    // Two cliques K_m joined by one bridge b: the clique split has cut b and
    // vol = m(m-1) + b per side, so Ncut = 2b / (m(m-1) + b).
    const int m = 6;
    const double b = 0.01;
    const seg::SparseMatrix w = two_cliques(m, b);
    std::vector<int> cliques(2 * m, 0);
    for (int i = m; i < 2 * m; ++i) cliques[static_cast<std::size_t>(i)] = 1;
    const double expected = 2 * b / (m * (m - 1) + b);
    EXPECT_NEAR(seg::ncut_value(w, cliques), expected, 1e-15);

    // The spectral 2-way partition finds exactly that minimum.
    seg::SpectralParams p;
    p.k = 2;
    const seg::SpectralResult r = seg::spectral_cluster(w, p);
    ASSERT_TRUE(r.ok());
    EXPECT_NEAR(seg::ncut_value(w, r.kmeans.labels), expected, 1e-15);
    EXPECT_GT(r.spectral_gap, 0.5);  // lambda_2 ~ 1, lambda_3 = -1/(m-1) + O(b): large gap

    // Trivial partition: Ncut 0. Every node alone in K_3: each cut(A_i)/vol(A_i) = 1, sum = 3.
    EXPECT_EQ(seg::ncut_value(w, std::vector<int>(2 * m, 0)), 0.0);
    const seg::SparseMatrix k3 = from_edges(3, {{0, 1, 1.0}, {1, 2, 1.0}, {0, 2, 1.0}});
    EXPECT_NEAR(seg::ncut_value(k3, {0, 1, 2}), 3.0, 1e-15);
}

TEST(QualityMetrics, RecursiveReportsRootGapAndSplitMetrics) {
    seg::RecursiveParams p;
    p.max_segments = 10;
    const seg::SparseMatrix w = two_cliques(10, 1e-3);
    const seg::RecursiveResult r = seg::recursive_spectral_cluster(w, p);
    ASSERT_TRUE(r.ok());
    ASSERT_EQ(r.root_eigenvalues.size(), 3);  // k + 1 with k = 2, on the full graph's M
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> dense{Eigen::MatrixXd(seg::normalized_affinity(w))};
    const Eigen::Index n = w.rows();
    EXPECT_NEAR(r.root_gap, dense.eigenvalues()(n - 2) - dense.eigenvalues()(n - 3), 1e-10);
    EXPECT_LE(r.min_gap, r.root_gap);
    EXPECT_LT(r.max_orthogonality_error, 1e-10);
}
