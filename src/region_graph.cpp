#include "region_graph.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace seg {

namespace {

// Breadth-first flood fill: writes id into out for every pixel 4-connected to seed through
// pixels with the seed's label. queue is caller-owned scratch, used as a FIFO via a head
// index so its storage is contiguous and reused across components.
void bfs_fill_component(const cv::Mat& labels, cv::Mat& out, const cv::Point& seed, int id,
                        std::vector<cv::Point>& queue) {
    const int value = labels.at<int>(seed);
    out.at<int>(seed) = id;
    queue.assign(1, seed);
    for (std::size_t head = 0; head < queue.size(); ++head) {
        const cv::Point p = queue[head];
        const cv::Point nbrs[4] = {{p.x + 1, p.y}, {p.x - 1, p.y}, {p.x, p.y + 1}, {p.x, p.y - 1}};
        for (const cv::Point& q : nbrs) {
            if (q.x < 0 || q.y < 0 || q.x >= labels.cols || q.y >= labels.rows) continue;
            if (out.at<int>(q) != -1 || labels.at<int>(q) != value) continue;
            out.at<int>(q) = id;  // marked on enqueue, so each pixel enters the queue once
            queue.push_back(q);
        }
    }
}

}  // namespace

int relabel_connected(const cv::Mat& labels, cv::Mat& out) {
    if (labels.empty() || labels.type() != CV_32S) {
        throw std::invalid_argument("relabel_connected: expected non-empty CV_32S labels");
    }
    out = cv::Mat(labels.rows, labels.cols, CV_32S, cv::Scalar(-1));
    std::vector<cv::Point> queue;
    int next_id = 0;
    for (int y = 0; y < labels.rows; ++y) {
        for (int x = 0; x < labels.cols; ++x) {
            if (out.at<int>(y, x) == -1) bfs_fill_component(labels, out, cv::Point(x, y), next_id++, queue);
        }
    }
    return next_id;
}

RegionGraph build_region_graph(const cv::Mat& lab, const cv::Mat& labels, const GraphParams& params) {
    if (lab.empty() || lab.type() != CV_32FC3) {
        throw std::invalid_argument("build_region_graph: expected a non-empty CV_32FC3 Lab image");
    }
    if (labels.type() != CV_32S || labels.size() != lab.size()) {
        throw std::invalid_argument("build_region_graph: labels must be CV_32S, same size as image");
    }
    if (!(params.sigma_color > 0.0) || !(params.edge_floor >= 0.0)) {
        throw std::invalid_argument("build_region_graph: need sigma_color > 0 and edge_floor >= 0");
    }
    double min_label = 0.0;
    double max_label = 0.0;
    cv::minMaxLoc(labels, &min_label, &max_label);
    if (min_label < 0.0) throw std::invalid_argument("build_region_graph: negative label");
    const int n = static_cast<int>(max_label) + 1;
    const int rows = lab.rows;
    const int cols = lab.cols;

    RegionGraph g;
    g.labels = labels.clone();
    g.regions.assign(static_cast<std::size_t>(n), RegionFeatures{});
    for (int y = 0; y < rows; ++y) {
        const int* lrow = labels.ptr<int>(y);
        const cv::Vec3f* crow = lab.ptr<cv::Vec3f>(y);
        for (int x = 0; x < cols; ++x) {
            RegionFeatures& r = g.regions[static_cast<std::size_t>(lrow[x])];
            ++r.pixel_count;
            r.centroid_x += x;
            r.centroid_y += y;
            for (int c = 0; c < 3; ++c) r.mean_lab[static_cast<std::size_t>(c)] += crow[x][c];
        }
    }
    for (RegionFeatures& r : g.regions) {
        if (r.pixel_count == 0) {
            throw std::invalid_argument("build_region_graph: labels must be 0..n-1 with no gaps");
        }
        const double inv = 1.0 / r.pixel_count;
        r.centroid_x *= inv;
        r.centroid_y *= inv;
        for (double& v : r.mean_lab) v *= inv;
    }

    // Packed (a, b) keys of differing 4-neighbours; sorting then run-length counting gives
    // deterministic edges and boundary lengths.
    std::vector<std::uint64_t> pairs;
    const auto pack = [](int a, int b) {
        const auto lo = static_cast<std::uint64_t>(std::min(a, b));
        const auto hi = static_cast<std::uint64_t>(std::max(a, b));
        return (lo << 32U) | hi;
    };
    for (int y = 0; y < rows; ++y) {
        const int* lrow = labels.ptr<int>(y);
        const int* next_row = y + 1 < rows ? labels.ptr<int>(y + 1) : nullptr;
        for (int x = 0; x < cols; ++x) {
            if (x + 1 < cols && lrow[x] != lrow[x + 1]) pairs.push_back(pack(lrow[x], lrow[x + 1]));
            if (next_row != nullptr && lrow[x] != next_row[x]) pairs.push_back(pack(lrow[x], next_row[x]));
        }
    }
    std::sort(pairs.begin(), pairs.end());

    const double pixels = static_cast<double>(rows) * cols;
    g.sigma_spatial = params.spatial_scale > 0.0 ? params.spatial_scale * std::sqrt(pixels / n) : 0.0;
    const double inv_color = 1.0 / (2.0 * params.sigma_color * params.sigma_color);
    const double inv_spatial =
        g.sigma_spatial > 0.0 ? 1.0 / (2.0 * g.sigma_spatial * g.sigma_spatial) : 0.0;

    std::vector<Eigen::Triplet<double>> triplets;
    for (std::size_t i = 0; i < pairs.size();) {
        std::size_t j = i;
        while (j < pairs.size() && pairs[j] == pairs[i]) ++j;
        const int a = static_cast<int>(pairs[i] >> 32U);
        const int b = static_cast<int>(pairs[i] & 0xffffffffU);
        const RegionFeatures& ra = g.regions[static_cast<std::size_t>(a)];
        const RegionFeatures& rb = g.regions[static_cast<std::size_t>(b)];
        double color_sq = 0.0;
        for (std::size_t c = 0; c < 3; ++c) {
            const double d = ra.mean_lab[c] - rb.mean_lab[c];
            color_sq += d * d;
        }
        const double dx = ra.centroid_x - rb.centroid_x;
        const double dy = ra.centroid_y - rb.centroid_y;
        const double w = std::exp(-color_sq * inv_color - (dx * dx + dy * dy) * inv_spatial) +
                         params.edge_floor;
        g.edges.push_back({a, b, static_cast<int>(j - i), w});
        triplets.emplace_back(a, b, w);
        triplets.emplace_back(b, a, w);
        i = j;
    }
    g.affinity.resize(n, n);
    g.affinity.setFromTriplets(triplets.begin(), triplets.end());
    g.affinity.makeCompressed();
    return g;
}

Eigen::VectorXd node_degrees(const SparseMatrix& w) {
    Eigen::VectorXd d = Eigen::VectorXd::Zero(w.rows());
    for (int col = 0; col < w.outerSize(); ++col) {
        for (SparseMatrix::InnerIterator it(w, col); it; ++it) d(it.row()) += it.value();
    }
    return d;
}

int connected_components(const SparseMatrix& w, std::vector<int>* component) {
    const auto n = static_cast<int>(w.rows());
    std::vector<int> comp(static_cast<std::size_t>(n), -1);
    std::vector<int> stack;
    int count = 0;
    for (int start = 0; start < n; ++start) {
        if (comp[static_cast<std::size_t>(start)] != -1) continue;
        comp[static_cast<std::size_t>(start)] = count;
        stack.assign(1, start);
        while (!stack.empty()) {
            const int node = stack.back();
            stack.pop_back();
            for (SparseMatrix::InnerIterator it(w, node); it; ++it) {
                const auto next = static_cast<std::size_t>(it.row());
                if (it.value() != 0.0 && comp[next] == -1) {
                    comp[next] = count;
                    stack.push_back(static_cast<int>(next));
                }
            }
        }
        ++count;
    }
    if (component != nullptr) *component = std::move(comp);
    return count;
}

std::size_t estimate_sparse_bytes(const SparseMatrix& m) {
    const auto nnz = static_cast<std::size_t>(m.nonZeros());
    return nnz * (sizeof(double) + sizeof(SparseMatrix::StorageIndex)) +
           (static_cast<std::size_t>(m.outerSize()) + 1) * sizeof(SparseMatrix::StorageIndex);
}

}  // namespace seg
