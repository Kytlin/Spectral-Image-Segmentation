#pragma once

#include <cstddef>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>

#include "segmenter.hpp"

namespace seg::api {

// An uploaded image. Pixel buffers are shared with copies handed out by the store and are
// never written after insertion.
struct Run {
    std::string id;
    std::string created_at;  // ISO 8601 UTC
    std::string filename;
    std::size_t bytes = 0;   // encoded upload size
    cv::Mat image;           // decoded BGR, CV_8UC3
    std::vector<std::string> result_ids;  // in creation order
};

// One successful segmentation of a run; immutable once stored.
struct Result {
    std::string id;
    std::string run_id;
    std::string created_at;
    SegmentParams params;
    SegmentationResult output;
};

// Thread-safe in-memory store. One mutex guards all state and is never held while
// segmenting or encoding, so lookups and inserts are short critical sections.
class RunStore {
public:
    RunStore(std::size_t max_runs, std::size_t max_results);

    // nullopt when the store is full.
    std::optional<Run> add_run(const std::string& filename, std::size_t bytes, cv::Mat image);
    std::optional<Run> find_run(const std::string& id) const;

    // nullopt when the store is full; throws std::out_of_range for an unknown run.
    std::optional<Result> add_result(const std::string& run_id, const SegmentParams& params,
                                     SegmentationResult output);
    std::optional<Result> find_result(const std::string& id) const;

    std::size_t run_count() const;
    std::size_t result_count() const;

private:
    std::string new_id_locked();

    const std::size_t max_runs_;
    const std::size_t max_results_;
    mutable std::mutex mutex_;
    std::mt19937_64 rng_;
    std::unordered_map<std::string, Run> runs_;
    std::unordered_map<std::string, Result> results_;
};

}  // namespace seg::api
