// Region graph contract (what the Stage 2 eigensolver will rely on), and the adjacency rule.

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "region_graph.hpp"
#include "slic.hpp"

namespace {

seg::ImageD solid_lab(int width, int height, double l, double a, double b) {
    seg::ImageD lab(width, height, 3);
    for (std::size_t i = 0; i < lab.data.size(); i += 3) {
        lab.data[i] = l;
        lab.data[i + 1] = a;
        lab.data[i + 2] = b;
    }
    return lab;
}

}  // namespace

TEST(RegionGraph, AffinityIsSquareSymmetricNonnegativeFiniteWithPositiveDegrees) {
    // Real SLIC output on a synthetic image: background plus an off-grid rectangle.
    seg::ImageD lab = solid_lab(96, 64, 30, 20, 10);
    for (int y = 13; y < 47; ++y) {
        for (int x = 23; x < 71; ++x) {
            lab.at(x, y, 0) = 70;
            lab.at(x, y, 1) = -40;
        }
    }
    seg::SlicParams params;
    params.num_superpixels = 30;
    const seg::SlicResult s = seg::slic(lab, params);
    const seg::RegionGraph g = seg::build_region_graph(lab, s.labels, seg::GraphParams{});
    const seg::SparseMatrix& w = g.affinity;

    ASSERT_EQ(w.rows(), s.num_labels);  // one node per superpixel
    ASSERT_EQ(w.cols(), s.num_labels);
    const seg::SparseMatrix wt = w.transpose();
    EXPECT_EQ((w - wt).norm(), 0.0) << "W must be exactly symmetric";
    for (int col = 0; col < w.outerSize(); ++col) {
        for (seg::SparseMatrix::InnerIterator it(w, col); it; ++it) {
            EXPECT_NE(it.row(), it.col()) << "no self loops";
            EXPECT_TRUE(std::isfinite(it.value()));
            EXPECT_GT(it.value(), 0.0);
            EXPECT_LE(it.value(), 1.0);
        }
    }
    EXPECT_GT(seg::node_degrees(w).minCoeff(), 0.0) << "every node needs at least one edge";
    EXPECT_EQ(seg::count_connected_components(w), 1);
}

TEST(RegionGraph, EdgeOnlyWhenRegionsShareASide) {
    // Quadrants 0 1 / 2 3. Sides are shared by 0-1, 0-2, 1-3, 2-3.
    // 0-3 and 1-2 touch only at the centre corner, so they must NOT be connected.
    const std::vector<int> labels = {0, 0, 1, 1,
                                     0, 0, 1, 1,
                                     2, 2, 3, 3,
                                     2, 2, 3, 3};
    const seg::RegionGraph g =
        seg::build_region_graph(solid_lab(4, 4, 50, 0, 0), labels, seg::GraphParams{});
    EXPECT_EQ(g.edges.size(), 4u);
    EXPECT_GT(g.affinity.coeff(0, 1), 0.0);
    EXPECT_GT(g.affinity.coeff(0, 2), 0.0);
    EXPECT_GT(g.affinity.coeff(1, 3), 0.0);
    EXPECT_GT(g.affinity.coeff(2, 3), 0.0);
    EXPECT_EQ(g.affinity.coeff(0, 3), 0.0);
    EXPECT_EQ(g.affinity.coeff(1, 2), 0.0);
}

TEST(RegionGraph, PixelAndNodeMappingsAreInverse) {
    const std::vector<int> labels = {0, 0, 1,
                                     2, 1, 1};
    const seg::RegionGraph g =
        seg::build_region_graph(solid_lab(3, 2, 50, 0, 0), labels, seg::GraphParams{});
    EXPECT_EQ(g.pixel_to_node, labels);
    // Node 1 owns pixels 2, 4, 5 (p = y * width + x).
    const std::vector<int> node1(g.node_pixels.begin() + g.node_offsets[1],
                                 g.node_pixels.begin() + g.node_offsets[2]);
    EXPECT_EQ(node1, (std::vector<int>{2, 4, 5}));
    EXPECT_EQ(g.node_offsets.back(), 6);  // every pixel listed exactly once in total
}
