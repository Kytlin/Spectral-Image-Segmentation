
#include <cmath>
#include <set>

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc/slic.hpp>

#include "region_graph.hpp"

namespace {

// 8x8 image split into 4 quadrants: 0 TL, 1 TR, 2 BL, 3 BR; Lab colour per quadrant.
void quadrants(cv::Mat& lab, cv::Mat& labels) {
    lab = cv::Mat(8, 8, CV_32FC3);
    labels = cv::Mat(8, 8, CV_32S);
    const cv::Vec3f colors[4] = {{50, 0, 0}, {60, 0, 0}, {50, 10, 0}, {80, 0, 20}};
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            const int q = (y >= 4 ? 2 : 0) + (x >= 4 ? 1 : 0);
            labels.at<int>(y, x) = q;
            lab.at<cv::Vec3f>(y, x) = colors[q];
        }
    }
}

double max_asymmetry(const seg::SparseMatrix& w) {
    const seg::SparseMatrix wt = w.transpose();
    return (w - wt).norm();
}

}  // namespace

TEST(RelabelConnected, SplitsDisconnectedLabelsAndIsCompact) {
    // Label 7 appears in two separate places; label 3 between them.
    cv::Mat in = (cv::Mat_<int>(2, 5) << 7, 7, 3, 7, 7, 7, 7, 3, 7, 7);
    cv::Mat out;
    const int n = seg::relabel_connected(in, out);
    EXPECT_EQ(n, 3);
    EXPECT_EQ(out.at<int>(0, 0), 0);
    EXPECT_EQ(out.at<int>(0, 2), 1);
    EXPECT_EQ(out.at<int>(0, 4), 2);
    EXPECT_EQ(out.at<int>(1, 1), 0);
    EXPECT_EQ(out.size(), in.size());
}

TEST(RegionGraph, QuadrantsGiveExactAdjacencyWeightsAndMapping) {
    cv::Mat lab, labels;
    quadrants(lab, labels);
    seg::GraphParams p;
    p.sigma_color = 10.0;
    p.spatial_scale = 0.0;
    p.edge_floor = 0.0;
    const seg::RegionGraph g = seg::build_region_graph(lab, labels, p);

    ASSERT_EQ(g.num_nodes(), 4);
    EXPECT_EQ(g.affinity.rows(), 4);
    EXPECT_EQ(g.affinity.cols(), 4);
    EXPECT_TRUE(g.affinity.isCompressed());
    // 4-adjacency: quadrant pairs sharing a side; diagonals (0,3), (1,2) only touch at a corner.
    ASSERT_EQ(g.edges.size(), 4U);
    const std::pair<int, int> expected[4] = {{0, 1}, {0, 2}, {1, 3}, {2, 3}};
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(g.edges[i].a, expected[i].first);
        EXPECT_EQ(g.edges[i].b, expected[i].second);
        EXPECT_EQ(g.edges[i].boundary_length, 4);
    }
    EXPECT_EQ(g.affinity.nonZeros(), 8);
    EXPECT_EQ(g.affinity.coeff(0, 3), 0.0);
    EXPECT_EQ(g.affinity.coeff(1, 2), 0.0);
    // w = exp(-||dlab||^2 / (2 sigma^2)): (0,1) differ by 10 in L -> exp(-0.5).
    EXPECT_NEAR(g.affinity.coeff(0, 1), std::exp(-0.5), 1e-12);
    EXPECT_NEAR(g.affinity.coeff(1, 3), std::exp(-(400.0 + 400.0) / 200.0), 1e-12);

    const Eigen::VectorXd d = seg::node_degrees(g.affinity);
    const Eigen::MatrixXd dense(g.affinity);
    for (int i = 0; i < 4; ++i) EXPECT_NEAR(d(i), dense.row(i).sum(), 1e-14);
    EXPECT_NEAR(d(0), g.affinity.coeff(0, 1) + g.affinity.coeff(0, 2), 1e-14);

    EXPECT_EQ(cv::countNonZero(g.labels != labels), 0);
    for (const auto& r : g.regions) EXPECT_EQ(r.pixel_count, 16);
    EXPECT_DOUBLE_EQ(g.regions[3].centroid_x, 5.5);
    EXPECT_DOUBLE_EQ(g.regions[3].centroid_y, 5.5);
    EXPECT_NEAR(g.regions[3].mean_lab[2], 20.0, 1e-9);
}

TEST(RegionGraph, PropertiesOnRealSlicOutput) {
    cv::Mat bgr(90, 120, CV_8UC3);
    cv::theRNG().state = 7;
    cv::randu(bgr, cv::Scalar::all(0), cv::Scalar::all(255));
    cv::GaussianBlur(bgr, bgr, cv::Size(), 3.0);
    cv::Mat lab;
    bgr.convertTo(lab, CV_32FC3, 1.0 / 255.0);
    cv::cvtColor(lab, lab, cv::COLOR_BGR2Lab);
    auto slic = cv::ximgproc::createSuperpixelSLIC(lab, cv::ximgproc::SLICO, 10);
    slic->iterate(10);
    slic->enforceLabelConnectivity(25);
    cv::Mat raw, labels;
    slic->getLabels(raw);
    const int n = seg::relabel_connected(raw, labels);
    const seg::RegionGraph g = seg::build_region_graph(lab, labels, seg::GraphParams{});

    ASSERT_EQ(g.num_nodes(), n);
    EXPECT_EQ(g.affinity.rows(), n);
    EXPECT_EQ(g.affinity.cols(), n);
    EXPECT_TRUE(g.affinity.isCompressed());
    EXPECT_EQ(g.affinity.nonZeros(), static_cast<long>(2 * g.edges.size()));
    EXPECT_LT(max_asymmetry(g.affinity), 1e-15);
    for (int c = 0; c < g.affinity.outerSize(); ++c) {
        for (seg::SparseMatrix::InnerIterator it(g.affinity, c); it; ++it) {
            EXPECT_TRUE(std::isfinite(it.value()));
            EXPECT_GT(it.value(), 0.0);
            EXPECT_LE(it.value(), 1.0 + seg::GraphParams{}.edge_floor);
            EXPECT_GE(it.value(), seg::GraphParams{}.edge_floor);
            EXPECT_NE(it.row(), it.col());
        }
    }
    EXPECT_TRUE((seg::node_degrees(g.affinity).array() > 0.0).all());
    EXPECT_EQ(seg::connected_components(g.affinity), 1);

    // Every pixel maps to a valid node; node pixel counts add up; each node is 4-connected.
    double lo = 0, hi = 0;
    cv::minMaxLoc(g.labels, &lo, &hi);
    EXPECT_EQ(lo, 0.0);
    EXPECT_EQ(hi, n - 1.0);
    int total = 0;
    for (const auto& r : g.regions) total += r.pixel_count;
    EXPECT_EQ(total, 90 * 120);
    for (int i = 0; i < n; ++i) {
        cv::Mat cc;
        EXPECT_EQ(cv::connectedComponents(g.labels == i, cc, 4), 2) << "node " << i;
    }
    EXPECT_GT(seg::estimate_sparse_bytes(g.affinity), 0U);
}

TEST(RegionGraph, RejectsLabelGaps) {
    cv::Mat lab, labels;
    quadrants(lab, labels);
    labels.setTo(5, labels == 3);  // label 3 unused, 5 present
    EXPECT_THROW(seg::build_region_graph(lab, labels, seg::GraphParams{}), std::invalid_argument);
}
