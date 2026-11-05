#pragma once

#include <vector>

#include <Eigen/Sparse>

#include "color.hpp"
#include "image.hpp"

namespace seg {

using SparseMatrix = Eigen::SparseMatrix<double>;  // column-major (CSC)

// Parameters of the affinity between two ADJACENT regions a and b:
//   w_ab = exp(-||lab_a - lab_b||^2 / (2 sigma_color^2)) * exp(-||c_a - c_b||^2 / (2 sigma_s^2))
// where lab_* are mean CIELAB colours and c_* are centroids in pixels.
struct GraphParams {
    // Colour scale in Lab units. A mean-colour difference of sigma_color gives a colour
    // factor of exp(-1/2) ~ 0.61; a difference of 3 sigma_color gives ~0.011.
    double sigma_color = 10.0;
    // sigma_s = spatial_scale * sqrt(N / R): a multiple of the typical region spacing, so the
    // setting doesn't depend on image size or superpixel count. <= 0 disables the spatial factor.
    double spatial_scale = 2.0;
};

struct Region {
    int pixel_count = 0;
    double centroid_x = 0.0;
    double centroid_y = 0.0;
    Lab mean_lab{};
};

// One undirected edge (a < b) of the region adjacency graph, with the quantities that
// produced its weight. W(a, b) == W(b, a) == weight.
struct RegionEdge {
    int a = 0;
    int b = 0;
    int boundary_length = 0;  // number of 4-neighbour pixel pairs (p in a, q in b)
    double color_distance = 0.0;
    double spatial_distance = 0.0;
    double weight = 0.0;
};

// A region adjacency graph: node i is region i (a set of pixels).
// Mappings between pixels and nodes are explicit, in both directions:
//   pixel -> node : pixel_to_node[p], p = y * width + x
//   node  -> pixels : node_pixels[node_offsets[i] .. node_offsets[i + 1]), sorted ascending
struct RegionGraph {
    int width = 0;
    int height = 0;
    std::vector<int> pixel_to_node;
    std::vector<int> node_offsets;  // size num_nodes() + 1
    std::vector<int> node_pixels;   // size width * height
    std::vector<Region> regions;    // size num_nodes()
    std::vector<RegionEdge> edges;  // sorted by (a, b)
    double sigma_spatial = 0.0;     // effective sigma_s in pixels (0 if disabled)
    SparseMatrix affinity;          // W: num_nodes x num_nodes, symmetric, zero diagonal

    int num_nodes() const { return static_cast<int>(regions.size()); }
};

// Builds the region adjacency graph and W from a Lab image and a label map.
// Requirements (checked, std::invalid_argument otherwise): labels.size() == width * height,
// labels are exactly 0..R-1 with every value used. Regions need not be connected for this
// function, but SLIC output always is.
// Two regions a != b are joined by an edge iff some pixel of a and some pixel of b are
// 4-neighbours (share a side). Regions touching only at a corner are NOT joined.
RegionGraph build_region_graph(const ImageD& lab, const std::vector<int>& labels,
                               const GraphParams& params);

// d_i = sum_j W(i, j).
Eigen::VectorXd node_degrees(const SparseMatrix& affinity);

// Number of connected components of the graph whose edges are the nonzero entries of W.
int count_connected_components(const SparseMatrix& affinity);

}  // namespace seg
