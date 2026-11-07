#include "api/codec.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

namespace seg::api {

namespace {

using nlohmann::json;

std::int64_t get_int(const json& v, const std::string& field, std::int64_t lo, std::int64_t hi) {
    const std::string range = " in [" + std::to_string(lo) + ", " + std::to_string(hi) + "]";
    if (!v.is_number_integer()) throw ValidationError(field, field + " must be an integer" + range);
    // nlohmann stores non-negative literals as unsigned; values above INT64_MAX would wrap.
    if (v.is_number_unsigned() && v.get<std::uint64_t>() > static_cast<std::uint64_t>(hi)) {
        throw ValidationError(field, field + " must be" + range);
    }
    const auto x = v.get<std::int64_t>();
    if (x < lo || x > hi) throw ValidationError(field, field + " must be" + range);
    return x;
}

// lo_open: whether lo itself is excluded.
double get_double(const json& v, const std::string& field, double lo, double hi, bool lo_open) {
    const std::string range = std::string(" in ") + (lo_open ? "(" : "[") + json(lo).dump() + ", " + json(hi).dump() + "]";
    if (!v.is_number()) throw ValidationError(field, field + " must be a number" + range);
    const double x = v.get<double>();
    if (!std::isfinite(x) || x > hi || x < lo || (lo_open && x == lo)) {
        throw ValidationError(field, field + " must be" + range);
    }
    return x;
}

}  // namespace

SegmentParams parse_segment_params(const json& body) {
    if (!body.is_object()) throw ValidationError("", "request body must be a JSON object");
    if (!body.contains("k")) throw ValidationError("k", "k is required");
    SegmentParams p;
    for (const auto& [key, v] : body.items()) {
        if (key == "k") {
            p.k = static_cast<int>(get_int(v, key, 2, 200));
        } else if (key == "method") {
            if (v == "flat") p.method = Method::Flat;
            else if (v == "recursive") p.method = Method::Recursive;
            else throw ValidationError(key, "method must be \"flat\" or \"recursive\"");
        } else if (key == "superpixels") {
            p.superpixels = static_cast<int>(get_int(v, key, 2, 20000));
        } else if (key == "ncut_threshold") {
            p.ncut_threshold = get_double(v, key, 0.0, 2.0, true);
        } else if (key == "sigma_color") {
            p.graph.sigma_color = get_double(v, key, 0.0, 100.0, true);
        } else if (key == "spatial_scale") {
            p.graph.spatial_scale = get_double(v, key, 0.0, 100.0, false);
        } else if (key == "blur_sigma") {
            p.blur_sigma = get_double(v, key, 0.0, 20.0, false);
        } else if (key == "max_dim") {
            p.max_dim = static_cast<int>(get_int(v, key, 0, 4096));
        } else if (key == "seed") {
            if (!v.is_number_unsigned()) throw ValidationError(key, "seed must be a non-negative integer");
            p.seed = v.get<std::uint64_t>();
        } else {
            throw ValidationError(key, "unknown parameter " + key);
        }
    }
    return p;
}

json params_to_json(const SegmentParams& p) {
    return {
        {"method", to_string(p.method)},
        {"k", p.k},
        {"superpixels", p.superpixels},
        {"ncut_threshold", p.ncut_threshold},
        {"sigma_color", p.graph.sigma_color},
        {"spatial_scale", p.graph.spatial_scale},
        {"edge_floor", p.graph.edge_floor},
        {"blur_sigma", p.blur_sigma},
        {"max_dim", p.max_dim},
        {"seed", p.seed},
        {"slic_iterations", p.slic_iterations},
        {"min_segment_superpixels", p.min_segment_superpixels},
        {"kmeans_attempts", p.kmeans_attempts},
    };
}

// NaN metrics (not applicable to the method) serialize as null.
json metrics_to_json(const SegmentationResult& r) {
    const SpectralSummary& s = r.spectral;
    const std::vector<double> eigenvalues(s.eigenvalues.data(), s.eigenvalues.data() + s.eigenvalues.size());
    return {
        {"status", r.status},
        {"width", r.image.cols},
        {"height", r.image.rows},
        {"superpixels", r.superpixels_actual},
        {"segments", s.segments},
        {"regions", r.region_count},
        {"graph", {{"nodes", r.graph_nodes}, {"nnz", r.graph_nnz}, {"mem_est_bytes", r.graph_mem_est_bytes}}},
        {"spectral",
         {
             {"eigenvalues", eigenvalues},
             {"converged", s.converged},
             {"iterations", s.iterations},
             {"matvecs", s.matvecs},
             {"ncv", s.ncv},
             {"retried", s.retried},
             {"max_residual", s.max_residual},
             {"orthogonality_error", s.orthogonality_error},
             {"gap_k", s.gap_k},
             {"spectral_gap", s.spectral_gap},
             {"min_split_gap", s.min_split_gap},
             {"ncut", s.ncut},
             {"kmeans_inertia", s.inertia},
             {"components", s.components},
             {"degenerate_rows", s.degenerate_rows},
             {"splits_accepted", s.splits_accepted},
             {"splits_rejected", s.splits_rejected},
             {"failed_solves", s.failed_solves},
         }},
        {"compute_ms",
         {
             {"preprocess", r.timings.preprocess_ms},
             {"slic", r.timings.slic_ms},
             {"graph", r.timings.graph_ms},
             {"eigen", r.timings.eigen_ms},
             {"kmeans", r.timings.kmeans_ms},
             {"total", r.timings.total_ms},
         }},
    };
}

json run_to_json(const db::RunRecord& run) {
    return {
        {"run_id", run.id},
        {"created_at", run.created_at},
        {"filename", run.filename},
        {"bytes", run.bytes},
        {"width", run.width},
        {"height", run.height},
        {"result_ids", run.result_ids},
    };
}

json result_to_json(const db::ResultRecord& result) {
    return {
        {"result_id", result.id},
        {"run_id", result.run_id},
        {"created_at", result.created_at},
        {"saved_at", result.saved_at ? json(*result.saved_at) : json(nullptr)},
        {"params", json::parse(result.params_json)},
        {"metrics", json::parse(result.metrics_json)},
    };
}

}  // namespace seg::api
