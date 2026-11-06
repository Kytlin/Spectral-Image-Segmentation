#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include <Eigen/Sparse>
#include <opencv2/core.hpp>

namespace seg {

using SparseMatrix = Eigen::SparseMatrix<double>;

// w_ab = exp(-||lab_a - lab_b||^2 / (2 sigma_color^2)) * exp(-||c_a - c_b||^2 / (2 sigma_s^2)) + edge_floor
struct GraphParams {
    double sigma_color = 10.0;
    // sigma_s = spatial_scale * sqrt(pixels / regions); <= 0 disables the spatial factor.
    double spatial_scale = 2.0;
    // Keeps very dissimilar neighbours from becoming numerically disconnected, which would
    // make eigenvalue 1 of M repeated and unresolvable by Lanczos.
    double edge_floor = 1e-6;
};

struct RegionFeatures {
    int pixel_count = 0;
    double centroid_x = 0.0;
    double centroid_y = 0.0;
    std::array<double, 3> mean_lab{};
};

struct RegionEdge {
    int a = 0;  // a < b
    int b = 0;
    int boundary_length = 0;
    double weight = 0.0;
};

struct RegionGraph {
    cv::Mat labels;
    std::vector<RegionFeatures> regions;
    std::vector<RegionEdge> edges;  // sorted by (a, b)
    double sigma_spatial = 0.0;
    SparseMatrix affinity;          // symmetric, zero diagonal

    int num_nodes() const { return static_cast<int>(regions.size()); }
};

// Gives each 4-connected component of equal labels its own id 0..R-1; returns R.
int relabel_connected(const cv::Mat& labels, cv::Mat& out);

RegionGraph build_region_graph(const cv::Mat& lab, const cv::Mat& labels, const GraphParams& params);

Eigen::VectorXd node_degrees(const SparseMatrix& w);

int connected_components(const SparseMatrix& w, std::vector<int>* component = nullptr);

// Values + indices only; ignores allocator overhead.
std::size_t estimate_sparse_bytes(const SparseMatrix& m);

}  // namespace seg
