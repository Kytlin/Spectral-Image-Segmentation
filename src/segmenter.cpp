#include "segmenter.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc/slic.hpp>

namespace seg {

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

int find_root(std::vector<int>& parent, int x) {
    while (parent[static_cast<std::size_t>(x)] != x) {
        parent[static_cast<std::size_t>(x)] = parent[static_cast<std::size_t>(parent[static_cast<std::size_t>(x)])];
        x = parent[static_cast<std::size_t>(x)];
    }
    return x;
}

// Components of the RAG restricted to edges within one cluster.
std::vector<int> connected_regions(const RegionGraph& g, const std::vector<int>& cluster, int* count) {
    const auto n = static_cast<std::size_t>(g.num_nodes());
    std::vector<int> parent(n);
    std::iota(parent.begin(), parent.end(), 0);
    for (const RegionEdge& e : g.edges) {
        if (cluster[static_cast<std::size_t>(e.a)] != cluster[static_cast<std::size_t>(e.b)]) continue;
        const int ra = find_root(parent, e.a);
        const int rb = find_root(parent, e.b);
        if (ra != rb) parent[static_cast<std::size_t>(std::max(ra, rb))] = std::min(ra, rb);
    }
    std::vector<int> root_id(n, -1);
    std::vector<int> region(n);
    int next = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const auto root = static_cast<std::size_t>(find_root(parent, static_cast<int>(i)));
        if (root_id[root] == -1) root_id[root] = next++;
        region[i] = root_id[root];
    }
    *count = next;
    return region;
}

cv::Mat map_labels(const cv::Mat& superpixels, const std::vector<int>& per_node) {
    cv::Mat out(superpixels.size(), CV_32S);
    for (int y = 0; y < superpixels.rows; ++y) {
        const int* src = superpixels.ptr<int>(y);
        int* dst = out.ptr<int>(y);
        for (int x = 0; x < superpixels.cols; ++x) dst[x] = per_node[static_cast<std::size_t>(src[x])];
    }
    return out;
}

}  // namespace

std::vector<std::filesystem::path> list_images(const std::filesystem::path& dir) {
    std::vector<std::filesystem::path> out;
    std::error_code ec;
    if (!std::filesystem::is_directory(dir, ec)) return out;
    for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" || ext == ".tif" ||
            ext == ".tiff" || ext == ".webp") {
            out.push_back(entry.path());
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

cv::Mat working_image(const cv::Mat& bgr, int max_dim) {
    const int long_side = std::max(bgr.cols, bgr.rows);
    if (max_dim <= 0 || long_side <= max_dim) return bgr.clone();
    const double f = static_cast<double>(max_dim) / long_side;
    cv::Mat out;
    cv::resize(bgr, out, cv::Size(), f, f, cv::INTER_AREA);
    return out;
}

std::string encode_label_png(const cv::Mat& labels) {
    if (labels.empty() || labels.type() != CV_32S) {
        throw std::invalid_argument("encode_label_png: expected non-empty CV_32S labels");
    }
    double lo = 0.0;
    double hi = 0.0;
    cv::minMaxLoc(labels, &lo, &hi);
    if (lo < 0.0 || hi > 65535.0) throw std::invalid_argument("encode_label_png: labels outside [0, 65535]");
    cv::Mat wide;
    labels.convertTo(wide, CV_16U);
    std::vector<uchar> png;
    if (!cv::imencode(".png", wide, png)) throw std::runtime_error("encode_label_png: PNG encoding failed");
    return std::string(png.begin(), png.end());
}

cv::Mat decode_label_png(const std::string& png) {
    const cv::Mat wide = cv::imdecode(
        cv::_InputArray(reinterpret_cast<const uchar*>(png.data()), static_cast<int>(png.size())),
        cv::IMREAD_UNCHANGED);
    if (wide.empty() || wide.type() != CV_16UC1) throw std::runtime_error("decode_label_png: not a 16-bit label PNG");
    cv::Mat labels;
    wide.convertTo(labels, CV_32S);
    return labels;
}

const char* to_string(Method m) { return m == Method::Flat ? "flat" : "recursive"; }

PreparedGraph prepare_graph(const cv::Mat& bgr, const SegmentParams& params) {
    const auto t_total = Clock::now();
    PreparedGraph p;
    if (bgr.empty() || bgr.type() != CV_8UC3) {
        p.status = "invalid_image";
        p.message = "expected a non-empty CV_8UC3 image";
        return p;
    }

    auto t0 = Clock::now();
    p.image = working_image(bgr, params.max_dim);
    cv::Mat smoothed;
    if (params.blur_sigma > 0.0) {
        cv::GaussianBlur(p.image, smoothed, cv::Size(), params.blur_sigma);
    } else {
        smoothed = p.image;
    }
    cv::Mat lab;
    smoothed.convertTo(lab, CV_32FC3, 1.0 / 255.0);
    cv::cvtColor(lab, lab, cv::COLOR_BGR2Lab);
    p.timings.preprocess_ms = ms_since(t0);

    const double pixels = static_cast<double>(p.image.rows) * p.image.cols;
    if (params.superpixels < 2 || params.superpixels > pixels / 4.0) {
        p.status = "invalid_superpixels";
        p.message = "superpixels=" + std::to_string(params.superpixels) + " outside [2, pixels/4]";
        p.timings.total_ms = ms_since(t_total);
        return p;
    }

    t0 = Clock::now();
    const int region_size = std::max(2, static_cast<int>(std::lround(std::sqrt(pixels / params.superpixels))));
    cv::Ptr<cv::ximgproc::SuperpixelSLIC> slic =
        cv::ximgproc::createSuperpixelSLIC(lab, cv::ximgproc::SLICO, region_size);
    slic->iterate(params.slic_iterations);
    slic->enforceLabelConnectivity(25);
    cv::Mat raw_labels;
    slic->getLabels(raw_labels);
    p.superpixels_actual = relabel_connected(raw_labels, p.superpixel_labels);
    p.timings.slic_ms = ms_since(t0);

    t0 = Clock::now();
    p.graph = build_region_graph(lab, p.superpixel_labels, params.graph);
    p.superpixel_labels = p.graph.labels;  // the graph keeps its own copy; share it, don't store two
    p.timings.graph_ms = ms_since(t0);
    p.ok = true;
    p.timings.total_ms = ms_since(t_total);
    return p;
}

// superpixel_labels shares graph.labels' buffer, so the label map is counted once.
std::size_t PreparedGraph::bytes() const {
    return image.total() * image.elemSize() + graph.labels.total() * graph.labels.elemSize() +
           estimate_sparse_bytes(graph.affinity) + graph.edges.size() * sizeof(RegionEdge) +
           graph.regions.size() * sizeof(RegionFeatures);
}

std::string graph_cache_key(const std::string& source_id, const SegmentParams& p) {
    // Hex floats are exact, so equal doubles give equal keys and different ones never collide.
    std::ostringstream key;
    key << std::hexfloat << "graph/v1/" << source_id << "/max_dim=" << p.max_dim << "/blur=" << p.blur_sigma
        << "/superpixels=" << p.superpixels << "/slic_iterations=" << p.slic_iterations
        << "/sigma_color=" << p.graph.sigma_color << "/spatial_scale=" << p.graph.spatial_scale
        << "/edge_floor=" << p.graph.edge_floor;
    return key.str();
}

void charge_graph_timings(SegmentationResult& r, const StageTimings& graph) {
    r.timings.preprocess_ms = graph.preprocess_ms;
    r.timings.slic_ms = graph.slic_ms;
    r.timings.graph_ms = graph.graph_ms;
    r.timings.total_ms += graph.total_ms;
}

SegmentationResult segment(const cv::Mat& bgr, const SegmentParams& params) {
    const PreparedGraph prepared = prepare_graph(bgr, params);
    SegmentationResult r = segment_graph(prepared, params);
    charge_graph_timings(r, prepared.timings);
    return r;
}

SegmentationResult segment_graph(const PreparedGraph& prepared, const SegmentParams& params) {
    const auto t_total = Clock::now();
    SegmentationResult r;
    r.status = prepared.status;
    r.message = prepared.message;
    // Own copies: a result must never alias the (shared, immutable) cached graph's buffers.
    r.image = prepared.image.clone();
    r.superpixel_labels = prepared.superpixel_labels.clone();
    r.superpixels_actual = prepared.superpixels_actual;
    if (!prepared.ok) {
        r.timings.total_ms = ms_since(t_total);
        return r;
    }
    const RegionGraph& g = prepared.graph;
    r.graph_nodes = g.num_nodes();
    r.graph_nnz = static_cast<long>(g.affinity.nonZeros());
    r.graph_mem_est_bytes = 2 * estimate_sparse_bytes(g.affinity) + g.edges.size() * sizeof(RegionEdge);

    SpectralParams sp;
    sp.k = params.k;
    sp.seed = params.seed;
    sp.kmeans_attempts = params.kmeans_attempts;
    std::vector<int> node_cluster;
    SpectralSummary& sum = r.spectral;
    if (params.method == Method::Flat) {
        const SpectralResult s = spectral_cluster(g.affinity, sp);
        r.timings.eigen_ms = s.eigen_ms;
        r.timings.kmeans_ms = s.kmeans_ms;
        r.status = to_string(s.status);
        r.message = s.message;
        sum.eigenvalues = s.eig.values;
        sum.converged = s.eig.converged;
        sum.iterations = s.eig.iterations;
        sum.matvecs = s.eig.matvecs;
        sum.ncv = s.eig.ncv;
        sum.retried = s.eig.retried;
        sum.max_residual = s.max_residual;
        sum.inertia = s.kmeans.inertia;
        sum.components = s.components;
        sum.degenerate_rows = s.degenerate_rows;
        sum.gap_k = params.k;
        sum.spectral_gap = s.spectral_gap;
        sum.orthogonality_error = s.orthogonality_error;
        sum.segments = s.ok() ? params.k : 0;
        if (s.ok()) node_cluster = s.kmeans.labels;
    } else {
        RecursiveParams rp;
        rp.max_segments = params.k;
        rp.ncut_threshold = params.ncut_threshold;
        rp.min_segment_nodes = params.min_segment_superpixels;
        rp.split = sp;
        const RecursiveResult s = recursive_spectral_cluster(g.affinity, rp);
        r.timings.eigen_ms = s.eigen_ms;
        r.timings.kmeans_ms = s.kmeans_ms;
        r.status = to_string(s.status);
        r.message = s.message;
        sum.eigenvalues = s.root_eigenvalues;
        sum.converged = s.ok() && s.failed_solves == 0;
        sum.iterations = s.total_iterations;
        sum.matvecs = s.total_matvecs;
        sum.ncv = s.max_ncv;
        sum.retried = s.any_retried;
        sum.max_residual = s.max_residual;
        sum.inertia = s.root_inertia;
        sum.components = s.components;
        sum.segments = s.segments;
        sum.splits_accepted = s.splits_accepted;
        sum.splits_rejected = s.splits_rejected;
        sum.failed_solves = s.failed_solves;
        sum.gap_k = 2;
        sum.spectral_gap = s.root_gap;
        sum.min_split_gap = s.min_gap;
        if (s.root_eigenvalues.size() > 0) sum.orthogonality_error = s.max_orthogonality_error;
        if (s.ok()) node_cluster = s.labels;
    }
    if (node_cluster.empty()) {
        r.timings.total_ms = ms_since(t_total);
        return r;
    }

    const std::vector<int> region = connected_regions(g, node_cluster, &r.region_count);
    sum.ncut = ncut_value(g.affinity, region);
    r.cluster_map = map_labels(r.superpixel_labels, node_cluster);
    r.region_map = map_labels(r.superpixel_labels, region);
    r.ok = true;
    r.timings.total_ms = ms_since(t_total);
    return r;
}

cv::Mat render_label_boundaries(const cv::Mat& bgr, const cv::Mat& labels, const cv::Scalar& color) {
    cv::Mat out = bgr.clone();
    const cv::Vec3b c(static_cast<uchar>(color[0]), static_cast<uchar>(color[1]), static_cast<uchar>(color[2]));
    for (int y = 0; y < labels.rows; ++y) {
        for (int x = 0; x < labels.cols; ++x) {
            const int l = labels.at<int>(y, x);
            const bool edge = (x + 1 < labels.cols && labels.at<int>(y, x + 1) != l) ||
                              (y + 1 < labels.rows && labels.at<int>(y + 1, x) != l);
            if (edge) out.at<cv::Vec3b>(y, x) = c;
        }
    }
    return out;
}

namespace {

// Golden-angle hue steps keep consecutive ids far apart in colour.
cv::Mat label_palette_image(const cv::Mat& labels) {
    double max_label = 0.0;
    cv::minMaxLoc(labels, nullptr, &max_label);
    const auto n = static_cast<int>(max_label) + 1;
    cv::Mat hsv(1, n, CV_8UC3);  // 8-bit hue range is [0, 180)
    for (int i = 0; i < n; ++i) {
        const double hue = std::fmod(i * 137.507764, 360.0) / 2.0;
        const auto sat = static_cast<uchar>(i % 3 == 0 ? 150 : (i % 3 == 1 ? 120 : 95));
        const auto val = static_cast<uchar>(i % 2 == 0 ? 185 : 150);
        hsv.at<cv::Vec3b>(0, i) = cv::Vec3b(cv::saturate_cast<uchar>(hue), sat, val);
    }
    cv::Mat palette;
    cv::cvtColor(hsv, palette, cv::COLOR_HSV2BGR);
    cv::Mat out(labels.size(), CV_8UC3);
    for (int y = 0; y < labels.rows; ++y) {
        const int* l = labels.ptr<int>(y);
        cv::Vec3b* o = out.ptr<cv::Vec3b>(y);
        for (int x = 0; x < labels.cols; ++x) o[x] = palette.at<cv::Vec3b>(0, l[x]);
    }
    return out;
}

}  // namespace

cv::Mat render_label_colors(const cv::Mat& labels, const cv::Scalar& border) {
    return render_label_boundaries(label_palette_image(labels), labels, border);
}

cv::Mat render_segment_tint(const cv::Mat& bgr, const cv::Mat& labels, double alpha, const cv::Scalar& border) {
    cv::Mat out;
    cv::addWeighted(bgr, 1.0 - alpha, label_palette_image(labels), alpha, 0.0, out);
    return render_label_boundaries(out, labels, border);
}

const char* to_string(OutputImage which) {
    switch (which) {
        case OutputImage::Original: return "original";
        case OutputImage::Superpixels: return "superpixels";
        case OutputImage::Segmentation: return "segmentation";
        case OutputImage::Boundaries: return "boundaries";
    }
    return "unknown";
}

cv::Mat render_output(const SegmentationResult& r, OutputImage which) {
    const cv::Scalar red(0, 0, 255);
    const cv::Scalar yellow(0, 255, 255);
    switch (which) {
        case OutputImage::Original: return r.image;
        case OutputImage::Superpixels:
            return r.superpixel_labels.empty() ? cv::Mat() : render_label_boundaries(r.image, r.superpixel_labels, yellow);
        case OutputImage::Segmentation: return r.ok ? render_segment_tint(r.image, r.region_map) : cv::Mat();
        case OutputImage::Boundaries: return r.ok ? render_label_boundaries(r.image, r.region_map, red) : cv::Mat();
    }
    return cv::Mat();
}

}  // namespace seg
