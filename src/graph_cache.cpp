#include "graph_cache.hpp"

#include <exception>
#include <utility>

namespace seg {

const char* to_string(CacheOutcome o) {
    switch (o) {
        case CacheOutcome::Miss: return "miss";
        case CacheOutcome::Hit: return "hit";
        case CacheOutcome::Shared: return "shared";
    }
    return "unknown";
}

GraphCache::GraphCache(std::size_t max_entries, std::size_t capacity_bytes)
    : max_entries_(max_entries), capacity_bytes_(capacity_bytes) {}

GraphCache::Lookup GraphCache::get_or_build(const std::string& key, const std::function<PreparedGraph()>& build) {
    if (max_entries_ == 0 || capacity_bytes_ == 0) {
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            ++counters_.misses;
        }
        return {std::make_shared<const PreparedGraph>(build()), CacheOutcome::Miss};
    }

    std::promise<std::shared_ptr<const PreparedGraph>> promise;
    std::uint64_t id = 0;
    Future existing;
    CacheOutcome outcome = CacheOutcome::Miss;
    {
        // One critical section either joins the existing entry or claims the key, so two
        // callers can never both build the same graph.
        const std::lock_guard<std::mutex> lock(mutex_);
        const auto it = entries_.find(key);
        if (it != entries_.end()) {
            it->second.last_used = ++clock_;
            outcome = it->second.ready ? CacheOutcome::Hit : CacheOutcome::Shared;
            ++(outcome == CacheOutcome::Hit ? counters_.hits : counters_.shared);
            existing = it->second.graph;
        } else {
            ++counters_.misses;
            id = ++next_id_;
            Entry entry;
            entry.graph = promise.get_future().share();
            entry.id = id;
            entry.last_used = ++clock_;
            entries_.emplace(key, std::move(entry));
        }
    }
    // get() may block on another caller's build, so it runs after the lock is released.
    if (existing.valid()) return {existing.get(), outcome};

    std::shared_ptr<const PreparedGraph> graph;
    try {
        graph = std::make_shared<const PreparedGraph>(build());
    } catch (...) {
        {
            // Erase before publishing the failure, so a request arriving after it builds
            // afresh instead of inheriting this (possibly transient) error.
            const std::lock_guard<std::mutex> lock(mutex_);
            const auto it = entries_.find(key);
            if (it != entries_.end() && it->second.id == id) entries_.erase(it);
        }
        promise.set_exception(std::current_exception());
        throw;
    }
    promise.set_value(graph);
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        finish_locked(key, id, *graph);
    }
    return {graph, CacheOutcome::Miss};
}

void GraphCache::finish_locked(const std::string& key, std::uint64_t id, const PreparedGraph& graph) {
    const auto it = entries_.find(key);
    // Cleared or replaced while building: the waiters already have the graph; keep nothing.
    if (it == entries_.end() || it->second.id != id) return;
    const std::size_t bytes = graph.bytes();
    if (!graph.ok || bytes > capacity_bytes_) {
        entries_.erase(it);
        return;
    }
    it->second.ready = true;
    it->second.bytes = bytes;
    it->second.last_used = ++clock_;  // just built: the most recently used, not the oldest
    bytes_ += bytes;
    ++ready_;
    evict_locked();
}

// Least recently used ready entries go first; in-flight builds are never evicted. A linear
// scan is fine for the handful of entries the cache holds.
void GraphCache::evict_locked() {
    while (ready_ > max_entries_ || bytes_ > capacity_bytes_) {
        auto victim = entries_.end();
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            if (it->second.ready && (victim == entries_.end() || it->second.last_used < victim->second.last_used)) {
                victim = it;
            }
        }
        if (victim == entries_.end()) return;
        bytes_ -= victim->second.bytes;
        --ready_;
        entries_.erase(victim);
        ++counters_.evictions;
    }
}

// In-flight builds stay: their graphs cannot be stale (keys hold every input), and dropping
// them would let the next request start a duplicate build of the same graph.
void GraphCache::clear() {
    const std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = entries_.begin(); it != entries_.end();) {
        if (it->second.ready) it = entries_.erase(it);
        else ++it;
    }
    bytes_ = 0;
    ready_ = 0;
}

CacheStats GraphCache::stats() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    CacheStats s = counters_;
    s.entries = ready_;
    s.max_entries = max_entries_;
    s.bytes = bytes_;
    s.capacity_bytes = capacity_bytes_;
    return s;
}

}  // namespace seg
