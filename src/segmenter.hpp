#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "region_graph.hpp"
#include "spectral.hpp"

namespace seg {

enum class Method {
    Recursive,  // k = maximum number of segments
    Flat,       // k = exact number of clusters
};

struct SegmentParams {
    Method method = Method::Recursive;
    int superpixels = 3000;
    int k = 80;
    double ncut_threshold = 0.05;
    int min_segment_superpixels = 2;
    int max_dim = 800;
    double blur_sigma = 1.0;
    int slic_iterations = 10;
    GraphParams graph{5.0};
    std::uint64_t seed = 42;
    int kmeans_attempts = 5;
};

const char* to_string(Method m);

struct StageTimings {
    double preprocess_ms = 0.0;
    double slic_ms = 0.0;
    double graph_ms = 0.0;
    double eigen_ms = 0.0;
    double kmeans_ms = 0.0;
    double total_ms = 0.0;
};

// For Recursive, solver stats are aggregated over all splits; eigenvalues, gap and inertia
// come from the first split, which runs on the full graph.
struct SpectralSummary {
    Eigen::VectorXd eigenvalues;
    bool converged = false;
    int iterations = 0;
    int matvecs = 0;
    int ncv = 0;
    bool retried = false;
    double max_residual = 0.0;
    double inertia = 0.0;
    int components = 0;
    int degenerate_rows = 0;
    int segments = 0;
    int splits_accepted = 0;
    int splits_rejected = 0;
    int failed_solves = 0;
    int gap_k = 0;
    double spectral_gap = std::numeric_limits<double>::quiet_NaN();
    double min_split_gap = std::numeric_limits<double>::quiet_NaN();
    double orthogonality_error = std::numeric_limits<double>::quiet_NaN();
    double ncut = std::numeric_limits<double>::quiet_NaN();  // of the connected regions on the full W
};

struct SegmentationResult {
    bool ok = false;
    std::string status;
    std::string message;
    cv::Mat image;
    cv::Mat superpixel_labels;
    int superpixels_actual = 0;
    int graph_nodes = 0;
    long graph_nnz = 0;
    std::size_t graph_mem_est_bytes = 0;
    SpectralSummary spectral;
    cv::Mat cluster_map;
    cv::Mat region_map;
    int region_count = 0;
    StageTimings timings;
};

SegmentationResult segment(const cv::Mat& bgr, const SegmentParams& params);

std::vector<std::filesystem::path> list_images(const std::filesystem::path& dir);

cv::Mat render_label_boundaries(const cv::Mat& bgr, const cv::Mat& labels, const cv::Scalar& color);
cv::Mat render_label_colors(const cv::Mat& labels, const cv::Scalar& border = cv::Scalar(30, 30, 30));
cv::Mat render_segment_tint(const cv::Mat& bgr, const cv::Mat& labels, double alpha = 0.35,
                            const cv::Scalar& border = cv::Scalar(30, 30, 30));

// The output images shared by the CLI and the HTTP API.
enum class OutputImage { Original, Superpixels, Segmentation, Boundaries };
inline constexpr OutputImage kOutputImages[] = {OutputImage::Original, OutputImage::Superpixels,
                                                OutputImage::Segmentation, OutputImage::Boundaries};
const char* to_string(OutputImage which);
// Empty when unavailable: no superpixels yet, or segmentation/boundaries of a failed result.
cv::Mat render_output(const SegmentationResult& r, OutputImage which);

}  // namespace seg
