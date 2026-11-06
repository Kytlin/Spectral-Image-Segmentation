
#include <algorithm>
#include <filesystem>

#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "segmenter.hpp"

namespace {

// Red left half, blue right half, green square in the middle, plus noise.
cv::Mat three_blocks() {
    cv::Mat img(90, 120, CV_8UC3, cv::Scalar(0, 0, 200));
    img(cv::Rect(60, 0, 60, 90)).setTo(cv::Scalar(200, 0, 0));
    img(cv::Rect(40, 25, 40, 40)).setTo(cv::Scalar(0, 200, 0));
    cv::Mat noise(img.size(), CV_8UC3);
    cv::theRNG().state = 3;
    cv::randu(noise, cv::Scalar::all(0), cv::Scalar::all(20));
    return img + noise;
}

void expect_valid(const seg::SegmentationResult& r, int k) {
    ASSERT_TRUE(r.ok) << r.status << ": " << r.message;
    const cv::Size size = r.image.size();
    EXPECT_EQ(r.superpixel_labels.size(), size);
    EXPECT_EQ(r.cluster_map.size(), size);
    EXPECT_EQ(r.region_map.size(), size);
    double lo = 0, hi = 0;
    cv::minMaxLoc(r.cluster_map, &lo, &hi);
    EXPECT_GE(lo, 0.0);
    EXPECT_LE(hi, k - 1.0);
    cv::minMaxLoc(r.region_map, &lo, &hi);
    EXPECT_EQ(lo, 0.0);
    EXPECT_EQ(hi, r.region_count - 1.0);
    EXPECT_GE(r.spectral.segments, 2);
    EXPECT_LE(r.spectral.segments, k);
    EXPECT_GE(r.region_count, r.spectral.segments);
    // Every output region is one 4-connected component with a single cluster label.
    for (int i = 0; i < r.region_count; ++i) {
        const cv::Mat mask = r.region_map == i;
        cv::Mat cc;
        ASSERT_EQ(cv::connectedComponents(mask, cc, 4), 2) << "region " << i;
        double cmin = 0, cmax = 0;
        cv::minMaxLoc(r.cluster_map, &cmin, &cmax, nullptr, nullptr, mask);
        EXPECT_EQ(cmin, cmax);
    }
    EXPECT_LT(r.spectral.max_residual, 1e-6);
    EXPECT_GE(r.spectral.ncut, 0.0);
    EXPECT_LE(r.spectral.ncut, static_cast<double>(r.region_count));
    EXPECT_LT(r.spectral.orthogonality_error, 1e-8);
    EXPECT_EQ(r.spectral.eigenvalues.size(), r.spectral.gap_k + 1);
    EXPECT_GE(r.spectral.spectral_gap, 0.0);
}

}  // namespace

TEST(Segment, SyntheticBlocksValidDeterministicAndColourSeparated) {
    const cv::Mat img = three_blocks();
    seg::SegmentParams p;
    p.superpixels = 100;
    p.k = 3;
    p.method = seg::Method::Flat;
    const seg::SegmentationResult a = seg::segment(img, p);
    expect_valid(a, 3);
    EXPECT_EQ(a.image.size(), img.size());
    EXPECT_EQ(a.graph_nodes, a.superpixels_actual);
    // The three colour blocks fall in three distinct clusters (colour, not semantics).
    const int red = a.cluster_map.at<int>(45, 10);
    const int blue = a.cluster_map.at<int>(45, 110);
    const int green = a.cluster_map.at<int>(45, 60);
    EXPECT_NE(red, blue);
    EXPECT_NE(red, green);
    EXPECT_NE(blue, green);

    const seg::SegmentationResult b = seg::segment(img, p);
    EXPECT_EQ(cv::countNonZero(a.cluster_map != b.cluster_map), 0);
    EXPECT_EQ(cv::countNonZero(a.superpixel_labels != b.superpixel_labels), 0);

    const cv::Mat overlay = seg::render_label_boundaries(a.image, a.region_map, cv::Scalar(0, 0, 255));
    const cv::Mat colors = seg::render_label_colors(a.region_map);
    EXPECT_EQ(overlay.size(), img.size());
    ASSERT_EQ(colors.size(), img.size());
    EXPECT_EQ(colors.type(), CV_8UC3);
    // Interior pixels of different regions get different colours; borders are the border colour.
    std::vector<cv::Vec3b> seen;
    for (int i = 0; i < a.region_count; ++i) {
        cv::Mat eroded;
        cv::erode(a.region_map == i, eroded, cv::Mat());
        cv::findNonZero(eroded, eroded);
        ASSERT_FALSE(eroded.empty()) << "region " << i;
        const cv::Vec3b c = colors.at<cv::Vec3b>(eroded.at<cv::Point>(0));
        for (const cv::Vec3b& other : seen) EXPECT_NE(c, other);
        seen.push_back(c);
    }
    const cv::Mat again = seg::render_label_colors(a.region_map);
    EXPECT_EQ(cv::countNonZero(again.reshape(1) != colors.reshape(1)), 0);
    // Tint = (1 - alpha) * photo + alpha * palette away from borders.
    const cv::Mat tint = seg::render_segment_tint(a.image, a.region_map, 0.35);
    ASSERT_EQ(tint.size(), img.size());
    const cv::Point q(10, 45);  // inside the red block, away from any border
    for (int c = 0; c < 3; ++c) {
        const double expected = 0.65 * a.image.at<cv::Vec3b>(q)[c] + 0.35 * colors.at<cv::Vec3b>(q)[c];
        EXPECT_NEAR(tint.at<cv::Vec3b>(q)[c], expected, 1.0);
    }
}

TEST(Segment, RecursiveStopsOnUniformBlocks) {
    // A generous cap (20) but only three colour blocks: recursion must stop at the colour
    // edges instead of slicing the uniform blocks, and be deterministic.
    const cv::Mat img = three_blocks();
    seg::SegmentParams p;
    p.superpixels = 200;
    p.k = 20;
    const seg::SegmentationResult a = seg::segment(img, p);
    expect_valid(a, 20);
    // Exactly three large segments; at most a couple of slivers made of superpixels that
    // straddle a colour edge (their blended mean colour ties weakly to both sides).
    int large = 0;
    for (int i = 0; i < a.spectral.segments; ++i) large += static_cast<std::size_t>(cv::countNonZero(a.cluster_map == i)) > img.total() / 20 ? 1 : 0;
    EXPECT_EQ(large, 3);
    EXPECT_LE(a.spectral.segments, 5);
    EXPECT_GT(a.spectral.splits_rejected, 0);
    EXPECT_EQ(a.spectral.failed_solves, 0);
    const int red = a.cluster_map.at<int>(45, 10);
    const int blue = a.cluster_map.at<int>(45, 110);
    const int green = a.cluster_map.at<int>(45, 60);
    EXPECT_NE(red, blue);
    EXPECT_NE(red, green);
    EXPECT_NE(blue, green);
    const seg::SegmentationResult b = seg::segment(img, p);
    EXPECT_EQ(cv::countNonZero(a.cluster_map != b.cluster_map), 0);
}

TEST(Segment, InvalidParametersAreReportedNotChanged) {
    const cv::Mat img = three_blocks();
    seg::SegmentParams p;
    p.superpixels = 100;
    p.k = 1;
    seg::SegmentationResult r = seg::segment(img, p);
    EXPECT_FALSE(r.ok);
    EXPECT_EQ(r.status, "invalid_k");
    p.method = seg::Method::Flat;
    r = seg::segment(img, p);
    EXPECT_EQ(r.status, "invalid_k");
    p.method = seg::Method::Recursive;
    p.k = 3;
    p.superpixels = 100000;
    r = seg::segment(img, p);
    EXPECT_FALSE(r.ok);
    EXPECT_EQ(r.status, "invalid_superpixels");
    EXPECT_FALSE(seg::segment(cv::Mat(), p).ok);
}

TEST(Segment, DownscalesToMaxDim) {
    seg::SegmentParams p;
    p.superpixels = 50;
    p.k = 2;
    p.max_dim = 60;
    const seg::SegmentationResult r = seg::segment(three_blocks(), p);
    expect_valid(r, 2);
    EXPECT_EQ(r.image.cols, 60);
    EXPECT_EQ(r.image.rows, 45);
}

TEST(Segment, EveryDataImageEndToEnd) {
    const std::vector<std::filesystem::path> images = seg::list_images(SEG_DATA_DIR);
    if (images.empty()) GTEST_SKIP() << "no images in " << SEG_DATA_DIR;
    seg::SegmentParams p;
    p.superpixels = 400;
    p.k = 10;
    for (const auto& path : images) {
        SCOPED_TRACE(path.string());
        const cv::Mat img = cv::imread(path.string(), cv::IMREAD_COLOR);
        ASSERT_FALSE(img.empty());
        const seg::SegmentationResult r = seg::segment(img, p);
        expect_valid(r, 10);
        // Working image: unchanged if small enough, else shrunk to max_dim keeping aspect ratio.
        if (std::max(img.cols, img.rows) <= p.max_dim) {
            EXPECT_EQ(r.image.size(), img.size());
        } else {
            EXPECT_EQ(std::max(r.image.cols, r.image.rows), p.max_dim);
            EXPECT_NEAR(static_cast<double>(r.image.cols) / r.image.rows, static_cast<double>(img.cols) / img.rows, 0.01);
        }
    }
}
