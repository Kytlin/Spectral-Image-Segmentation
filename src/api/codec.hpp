#pragma once

#include <stdexcept>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

#include "db/database.hpp"
#include "segmenter.hpp"

namespace seg::api {

// A request field that is missing, mistyped, unknown or out of range.
class ValidationError : public std::invalid_argument {
public:
    ValidationError(std::string field, const std::string& message)
        : std::invalid_argument(message), field_(std::move(field)) {}
    const std::string& field() const noexcept { return field_; }

private:
    std::string field_;
};

// Body of POST /runs/{id}/results: "k" is required; omitted fields keep the CLI defaults
// from SegmentParams, so equal inputs give the CLI's result. Throws ValidationError.
SegmentParams parse_segment_params(const nlohmann::json& body);

nlohmann::json params_to_json(const SegmentParams& p);
// The engine's metrics for one segmentation. compute_ms holds that call's own stage
// timings (engine wall clock), never HTTP latency or sums over several results.
nlohmann::json metrics_to_json(const SegmentationResult& r);
nlohmann::json run_to_json(const db::RunRecord& run);
// params and metrics are the stored jsonb, passed through without recomputation.
nlohmann::json result_to_json(const db::ResultRecord& result);

}  // namespace seg::api
