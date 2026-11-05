#include "region_graph.hpp"

#include <cmath>
#include <map>
#include <stdexcept>
#include <utility>

namespace seg {

RegionGraph build_region_graph(const ImageD& lab, const std::vector<int>& labels,
                               const GraphParams& params) {
    if (lab.channels != 3 || lab.width <= 0 || lab.height <= 0) {
        throw std::invalid_argument("build_region_graph: expected a non-empty Lab image");
    }
    if (!(params.sigma_color > 0.0)) {
        throw std::invalid_argument("build_region_graph: sigma_color must be > 0");
    }
    const int width = lab.width;
    const int height = lab.height;
    const int n = width * height;
    if (static_cast<int>(labels.size()) != n) {
        throw std::invalid_argument("build_region_graph: labels.size() != width * height");
    }

    // Validate labels and count pixels per node.
    int num_nodes = 0;
    for (int label : labels) {
        if (label < 0) {
            throw std::invalid_argument("build_region_graph: negative label");
        }
        num_nodes = std::max(num_nodes, label + 1);
    }
    std::vector<int> counts(static_cast<std::size_t>(num_nodes), 0);
    for (int label : labels) {
        ++counts[static_cast<std::size_t>(label)];
    }
    for (int count : counts) {
        if (count == 0) {
            throw std::invalid_argument("build_region_graph: labels must be 0..R-1 with no gaps");
        }
    }

    RegionGraph graph;
    graph.width = width;
    graph.height = height;
    graph.pixel_to_node = labels;

    // node -> pixels, as a CSR-style list (counting sort, so pixels stay in ascending order).
    graph.node_offsets.assign(static_cast<std::size_t>(num_nodes) + 1, 0);
    for (int i = 0; i < num_nodes; ++i) {
        graph.node_offsets[static_cast<std::size_t>(i) + 1] =
            graph.node_offsets[static_cast<std::size_t>(i)] + counts[static_cast<std::size_t>(i)];
    }
    graph.node_pixels.assign(static_cast<std::size_t>(n), 0);
    std::vector<int> cursor(graph.node_offsets.begin(), graph.node_offsets.end() - 1);
    for (int p = 0; p < n; ++p) {
        const auto node = static_cast<std::size_t>(labels[static_cast<std::size_t>(p)]);
        graph.node_pixels[static_cast<std::size_t>(cursor[node]++)] = p;
    }

    // Per-region statistics: pixel count, centroid, mean Lab colour.
    graph.regions.assign(static_cast<std::size_t>(num_nodes), Region{});
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            Region& r = graph.regions[static_cast<std::size_t>(labels[static_cast<std::size_t>(y * width + x)])];
            ++r.pixel_count;
            r.centroid_x += x;
            r.centroid_y += y;
            for (std::size_t c = 0; c < 3; ++c) {
                r.mean_lab[c] += lab.at(x, y, static_cast<int>(c));
            }
        }
    }
    for (Region& r : graph.regions) {
        const double inv = 1.0 / r.pixel_count;
        r.centroid_x *= inv;
        r.centroid_y *= inv;
        for (double& v : r.mean_lab) {
            v *= inv;
        }
    }

    // Adjacency: visit every 4-neighbour pixel pair once (right and down neighbours) and
    // count, for each pair of different labels, how many such pixel pairs they share.
    std::map<std::pair<int, int>, int> boundary;  // ordered -> deterministic edge order
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int a = labels[static_cast<std::size_t>(y * width + x)];
            if (x + 1 < width) {
                const int b = labels[static_cast<std::size_t>(y * width + x + 1)];
                if (a != b) {
                    ++boundary[{std::min(a, b), std::max(a, b)}];
                }
            }
            if (y + 1 < height) {
                const int b = labels[static_cast<std::size_t>((y + 1) * width + x)];
                if (a != b) {
                    ++boundary[{std::min(a, b), std::max(a, b)}];
                }
            }
        }
    }

    graph.sigma_spatial =
        params.spatial_scale > 0.0 ? params.spatial_scale * std::sqrt(static_cast<double>(n) / num_nodes)
                                   : 0.0;
    const double inv_two_sigma_color_sq = 1.0 / (2.0 * params.sigma_color * params.sigma_color);
    const double inv_two_sigma_spatial_sq =
        graph.sigma_spatial > 0.0 ? 1.0 / (2.0 * graph.sigma_spatial * graph.sigma_spatial) : 0.0;

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(boundary.size() * 2);
    graph.edges.reserve(boundary.size());
    for (const auto& [key, length] : boundary) {
        const Region& ra = graph.regions[static_cast<std::size_t>(key.first)];
        const Region& rb = graph.regions[static_cast<std::size_t>(key.second)];
        double color_sq = 0.0;
        for (std::size_t c = 0; c < 3; ++c) {
            const double d = ra.mean_lab[c] - rb.mean_lab[c];
            color_sq += d * d;
        }
        const double dx = ra.centroid_x - rb.centroid_x;
        const double dy = ra.centroid_y - rb.centroid_y;
        const double spatial_sq = dx * dx + dy * dy;
        const double weight =
            std::exp(-color_sq * inv_two_sigma_color_sq - spatial_sq * inv_two_sigma_spatial_sq);

        graph.edges.push_back({key.first, key.second, length, std::sqrt(color_sq),
                               std::sqrt(spatial_sq), weight});
        triplets.emplace_back(key.first, key.second, weight);
        triplets.emplace_back(key.second, key.first, weight);
    }
    graph.affinity.resize(num_nodes, num_nodes);
    graph.affinity.setFromTriplets(triplets.begin(), triplets.end());
    return graph;
}

Eigen::VectorXd node_degrees(const SparseMatrix& affinity) {
    Eigen::VectorXd degrees = Eigen::VectorXd::Zero(affinity.rows());
    for (int col = 0; col < affinity.outerSize(); ++col) {
        for (SparseMatrix::InnerIterator it(affinity, col); it; ++it) {
            degrees(it.row()) += it.value();
        }
    }
    return degrees;
}

int count_connected_components(const SparseMatrix& affinity) {
    const auto n = static_cast<int>(affinity.rows());
    std::vector<int> component(static_cast<std::size_t>(n), -1);
    std::vector<int> stack;
    int count = 0;
    for (int start = 0; start < n; ++start) {
        if (component[static_cast<std::size_t>(start)] != -1) {
            continue;
        }
        component[static_cast<std::size_t>(start)] = count;
        stack.assign(1, start);
        while (!stack.empty()) {
            const int node = stack.back();
            stack.pop_back();
            // Column `node` of a symmetric W lists node's neighbours.
            for (SparseMatrix::InnerIterator it(affinity, node); it; ++it) {
                const auto next = static_cast<int>(it.row());
                if (it.value() != 0.0 && component[static_cast<std::size_t>(next)] == -1) {
                    component[static_cast<std::size_t>(next)] = count;
                    stack.push_back(next);
                }
            }
        }
        ++count;
    }
    return count;
}

}  // namespace seg
