#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "image.hpp"
#include "region_graph.hpp"

namespace seg {

// Paints pixels whose right or lower neighbour has a different label in `color`.
ImageU8 draw_region_boundaries(const ImageU8& rgb, const std::vector<int>& labels,
                               std::array<std::uint8_t, 3> color);

// Graph overlay at `scale` x the working resolution: the image lightened, region boundaries
// in gray, one straight line per edge between the two region centroids, coloured by weight
// (dark purple = 0, teal = 0.5, yellow = 1), and a dot at every centroid.
ImageU8 draw_region_graph(const ImageU8& rgb, const RegionGraph& graph, int scale);

}  // namespace seg
