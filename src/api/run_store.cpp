#include "api/run_store.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <stdexcept>
#include <utility>

namespace seg::api {

namespace {

std::string now_iso8601() {
    using namespace std::chrono;
    const auto now = system_clock::now();
    const std::time_t t = system_clock::to_time_t(now);
    const auto ms = duration_cast<milliseconds>(now.time_since_epoch()).count() % 1000;
    std::tm tm{};
    gmtime_r(&t, &tm);
    std::array<char, 32> buf{};
    const std::size_t n = std::strftime(buf.data(), buf.size(), "%Y-%m-%dT%H:%M:%S", &tm);
    std::array<char, 8> frac{};
    std::snprintf(frac.data(), frac.size(), ".%03dZ", static_cast<int>(ms));
    return std::string(buf.data(), n) + frac.data();
}

// RFC 4122 version-4 UUID: 122 random bits, version nibble 4, variant bits 10.
std::string uuid_v4(std::mt19937_64& rng) {
    std::array<std::uint8_t, 16> b{};
    const std::uint64_t hi = rng();
    const std::uint64_t lo = rng();
    for (std::size_t i = 0; i < 8; ++i) {
        b[i] = static_cast<std::uint8_t>(hi >> (8 * i));
        b[i + 8] = static_cast<std::uint8_t>(lo >> (8 * i));
    }
    b[6] = static_cast<std::uint8_t>((b[6] & 0x0F) | 0x40);
    b[8] = static_cast<std::uint8_t>((b[8] & 0x3F) | 0x80);
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out;
    out.reserve(36);
    for (std::size_t i = 0; i < b.size(); ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) out += '-';
        out += kHex[b[i] >> 4];
        out += kHex[b[i] & 0x0F];
    }
    return out;
}

}  // namespace

RunStore::RunStore(std::size_t max_runs, std::size_t max_results)
    : max_runs_(max_runs), max_results_(max_results), rng_(std::random_device{}()) {}

std::string RunStore::new_id_locked() {
    // Runs and results share one ID space, so an ID names exactly one object.
    std::string id;
    do {
        id = uuid_v4(rng_);
    } while (runs_.count(id) != 0 || results_.count(id) != 0);
    return id;
}

std::optional<Run> RunStore::add_run(const std::string& filename, std::size_t bytes, cv::Mat image) {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (runs_.size() >= max_runs_) return std::nullopt;
    Run run;
    run.id = new_id_locked();
    run.created_at = now_iso8601();
    run.filename = filename;
    run.bytes = bytes;
    run.image = std::move(image);
    const auto [it, inserted] = runs_.emplace(run.id, std::move(run));
    (void)inserted;
    return it->second;
}

std::optional<Run> RunStore::find_run(const std::string& id) const {
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto it = runs_.find(id);
    if (it == runs_.end()) return std::nullopt;
    return it->second;
}

std::optional<Result> RunStore::add_result(const std::string& run_id, const SegmentParams& params,
                                           SegmentationResult output) {
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto run = runs_.find(run_id);
    if (run == runs_.end()) throw std::out_of_range("unknown run " + run_id);
    if (results_.size() >= max_results_) return std::nullopt;
    Result result;
    result.id = new_id_locked();
    result.run_id = run_id;
    result.created_at = now_iso8601();
    result.params = params;
    result.output = std::move(output);
    run->second.result_ids.push_back(result.id);
    const auto [it, inserted] = results_.emplace(result.id, std::move(result));
    (void)inserted;
    return it->second;
}

std::optional<Result> RunStore::find_result(const std::string& id) const {
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto it = results_.find(id);
    if (it == results_.end()) return std::nullopt;
    return it->second;
}

std::size_t RunStore::run_count() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return runs_.size();
}

std::size_t RunStore::result_count() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return results_.size();
}

}  // namespace seg::api
