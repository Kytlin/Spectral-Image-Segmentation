#pragma once

#include <stdexcept>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

#include "api/run_store.hpp"
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
nlohmann::json metrics_to_json(const SegmentationResult& r);
nlohmann::json run_to_json(const Run& run);
nlohmann::json result_to_json(const Result& result);

}  // namespace seg::api
