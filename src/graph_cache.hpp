#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "segmenter.hpp"

namespace seg {

// How a lookup was served: built by this caller, found ready, or joined another caller's
// build that was still running.
enum class CacheOutcome { Miss, Hit, Shared };
const char* to_string(CacheOutcome o);

struct CacheStats {
    std::size_t entries = 0;
    std::size_t max_entries = 0;
    std::size_t bytes = 0;
    std::size_t capacity_bytes = 0;
    std::uint64_t hits = 0;
    std::uint64_t shared = 0;
    std::uint64_t misses = 0;
    std::uint64_t evictions = 0;
};

// Bounded, thread-safe LRU cache of PreparedGraphs keyed by graph_cache_key(). Holds at most
// max_entries finished graphs; capacity_bytes is a memory safety cap on top of that.
//
// Entries are immutable and handed out as shared_ptr<const>: the cache and every request
// using a graph share ownership, so evicting or clearing never frees a graph that a running
// computation still reads. The mutex only guards the map; builds run outside it, and
// concurrent requests for the same key wait on one shared build (single flight).
class GraphCache {
public:
    struct Lookup {
        std::shared_ptr<const PreparedGraph> graph;
        CacheOutcome outcome = CacheOutcome::Miss;
    };

    // max_entries == 0 or capacity_bytes == 0 disables caching (every lookup builds).
    GraphCache(std::size_t max_entries, std::size_t capacity_bytes);

    // Returns the cached graph for key, or runs build() once and caches it when it succeeded
    // (prepared.ok) and fits. Exceptions from build() propagate to every waiter and nothing
    // is cached.
    Lookup get_or_build(const std::string& key, const std::function<PreparedGraph()>& build);

    // Drops every finished entry; graphs in use stay alive until their users finish, and
    // builds still running complete and are kept.
    void clear();
    CacheStats stats() const;

private:
    using Future = std::shared_future<std::shared_ptr<const PreparedGraph>>;
    struct Entry {
        Future graph;
        std::uint64_t id = 0;         // distinguishes this entry from a later one with the same key
        std::size_t bytes = 0;        // 0 until the build finishes
        std::uint64_t last_used = 0;  // LRU clock value
        bool ready = false;
    };

    void finish_locked(const std::string& key, std::uint64_t id, const PreparedGraph& graph);
    void evict_locked();

    const std::size_t max_entries_;
    const std::size_t capacity_bytes_;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, Entry> entries_;
    std::size_t bytes_ = 0;
    std::size_t ready_ = 0;  // finished entries (in-flight builds are not counted)
    std::uint64_t clock_ = 0;
    std::uint64_t next_id_ = 0;
    CacheStats counters_;
};

}  // namespace seg
