#include <atomic>
#include <chrono>
#include <future>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>

#include "graph_cache.hpp"
#include "segmenter.hpp"

namespace {

// A ready graph whose footprint is about kb kilobytes (only the image is allocated).
seg::PreparedGraph fake_graph(int kb) {
    seg::PreparedGraph g;
    g.ok = true;
    g.image = cv::Mat(kb, 1024, CV_8UC1, cv::Scalar(7));
    return g;
}

constexpr std::size_t kKB = 1024;

seg::SegmentParams fast_params() {
    seg::SegmentParams p;
    p.method = seg::Method::Flat;
    p.superpixels = 400;
    p.k = 6;
    return p;
}

void expect_same_segmentation(const seg::SegmentationResult& a, const seg::SegmentationResult& b) {
    ASSERT_TRUE(a.ok && b.ok);
    EXPECT_EQ(cv::countNonZero(a.region_map != b.region_map), 0);
    EXPECT_EQ(cv::countNonZero(a.cluster_map != b.cluster_map), 0);
    EXPECT_EQ(a.region_count, b.region_count);
    EXPECT_EQ(a.spectral.segments, b.spectral.segments);
    EXPECT_EQ(a.spectral.ncut, b.spectral.ncut);
    EXPECT_EQ(a.spectral.max_residual, b.spectral.max_residual);
    EXPECT_EQ(a.spectral.eigenvalues, b.spectral.eigenvalues);
}

}  // namespace

TEST(GraphCache, MissThenHitReturnsSameGraph) {
    seg::GraphCache cache(100, 10 * kKB);
    int builds = 0;
    const auto build = [&] { ++builds; return fake_graph(1); };
    const auto first = cache.get_or_build("a", build);
    const auto second = cache.get_or_build("a", build);
    const auto other = cache.get_or_build("b", build);
    EXPECT_EQ(first.outcome, seg::CacheOutcome::Miss);
    EXPECT_EQ(second.outcome, seg::CacheOutcome::Hit);
    EXPECT_EQ(other.outcome, seg::CacheOutcome::Miss);
    EXPECT_EQ(first.graph.get(), second.graph.get());
    EXPECT_EQ(builds, 2);
    const seg::CacheStats s = cache.stats();
    EXPECT_EQ(s.entries, 2u);
    EXPECT_EQ(s.hits, 1u);
    EXPECT_EQ(s.misses, 2u);
}

TEST(GraphCache, ConcurrentRequestsShareOneBuild) {
    seg::GraphCache cache(100, 10 * kKB);
    std::atomic<int> builds{0};
    std::vector<seg::GraphCache::Lookup> out(8);
    std::vector<std::thread> threads;
    for (std::size_t i = 0; i < out.size(); ++i) {
        threads.emplace_back([&, i] {
            out[i] = cache.get_or_build("k", [&] {
                ++builds;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                return fake_graph(1);
            });
        });
    }
    for (auto& t : threads) t.join();
    EXPECT_EQ(builds.load(), 1);
    int misses = 0;
    for (const auto& l : out) {
        EXPECT_EQ(l.graph.get(), out[0].graph.get());
        misses += l.outcome == seg::CacheOutcome::Miss;
    }
    EXPECT_EQ(misses, 1);
}

TEST(GraphCache, EvictsLeastRecentlyUsedWithinByteBudget) {
    seg::GraphCache cache(100, 25 * kKB);  // room for two 10 KB graphs
    const auto a = cache.get_or_build("a", [] { return fake_graph(10); });
    cache.get_or_build("b", [] { return fake_graph(10); });
    cache.get_or_build("a", [] { return fake_graph(10); });  // a is now more recent than b
    cache.get_or_build("c", [] { return fake_graph(10); });  // evicts b
    EXPECT_EQ(cache.get_or_build("a", [] { return fake_graph(10); }).outcome, seg::CacheOutcome::Hit);
    EXPECT_EQ(cache.get_or_build("c", [] { return fake_graph(10); }).outcome, seg::CacheOutcome::Hit);
    EXPECT_EQ(cache.get_or_build("b", [] { return fake_graph(10); }).outcome, seg::CacheOutcome::Miss);
    const seg::CacheStats s = cache.stats();
    EXPECT_LE(s.bytes, s.capacity_bytes);
    EXPECT_EQ(s.evictions, 2u);  // b, then a or c to make room for b again
}

TEST(GraphCache, HoldsAtMostMaxEntriesGraphs) {
    seg::GraphCache cache(2, 1000 * kKB);  // count-bound: bytes never limit here
    cache.get_or_build("a", [] { return fake_graph(1); });
    cache.get_or_build("b", [] { return fake_graph(1); });
    cache.get_or_build("a", [] { return fake_graph(1); });  // b is now least recently used
    cache.get_or_build("c", [] { return fake_graph(1); });  // third graph evicts b
    const seg::CacheStats s = cache.stats();
    EXPECT_EQ(s.entries, 2u);
    EXPECT_EQ(s.max_entries, 2u);
    EXPECT_EQ(s.evictions, 1u);
    EXPECT_EQ(cache.get_or_build("a", [] { return fake_graph(1); }).outcome, seg::CacheOutcome::Hit);
    EXPECT_EQ(cache.get_or_build("c", [] { return fake_graph(1); }).outcome, seg::CacheOutcome::Hit);
    EXPECT_EQ(cache.get_or_build("b", [] { return fake_graph(1); }).outcome, seg::CacheOutcome::Miss);
}

TEST(GraphCache, EvictedGraphStaysValidForItsHolder) {
    seg::GraphCache cache(100, 15 * kKB);
    const std::shared_ptr<const seg::PreparedGraph> held = cache.get_or_build("a", [] { return fake_graph(10); }).graph;
    cache.get_or_build("b", [] { return fake_graph(10); });  // evicts a
    cache.clear();
    EXPECT_EQ(held->image.rows, 10);
    EXPECT_EQ(held->image.at<uchar>(9, 1023), 7);
    EXPECT_EQ(cache.stats().entries, 0u);
}

TEST(GraphCache, FailedOrInvalidBuildsAreNotCached) {
    seg::GraphCache cache(100, 10 * kKB);
    EXPECT_THROW(cache.get_or_build("x", []() -> seg::PreparedGraph { throw std::runtime_error("boom"); }),
                 std::runtime_error);
    EXPECT_EQ(cache.get_or_build("x", [] { return fake_graph(1); }).outcome, seg::CacheOutcome::Miss);

    const auto invalid = [] {
        seg::PreparedGraph g;
        g.status = "invalid_superpixels";
        return g;
    };
    EXPECT_EQ(cache.get_or_build("y", invalid).outcome, seg::CacheOutcome::Miss);
    EXPECT_EQ(cache.get_or_build("y", invalid).outcome, seg::CacheOutcome::Miss);
    EXPECT_EQ(cache.get_or_build("too_big", [] { return fake_graph(20); }).outcome, seg::CacheOutcome::Miss);
    EXPECT_EQ(cache.stats().entries, 1u);  // only x
}

// Blocks a build until the test opens it, so ordering never depends on timing.
struct Gate {
    std::promise<void> open;
    std::shared_future<void> opened = open.get_future().share();
};

TEST(GraphCache, ClearDuringBuildKeepsItSingleFlight) {
    seg::GraphCache cache(100, 10 * kKB);
    Gate gate;
    std::atomic<bool> started{false};
    std::atomic<int> builds{0};
    const auto gated = [&] {
        ++builds;
        started = true;
        gate.opened.wait();
        return fake_graph(1);
    };
    cache.get_or_build("done", [] { return fake_graph(1); });
    std::thread builder([&] { cache.get_or_build("k", gated); });
    while (!started) std::this_thread::yield();
    cache.clear();
    // A request after the clear joins the running build instead of starting a second one.
    seg::CacheOutcome joined = seg::CacheOutcome::Miss;
    std::thread joiner([&] { joined = cache.get_or_build("k", gated).outcome; });
    while (cache.stats().shared == 0) std::this_thread::yield();
    gate.open.set_value();
    builder.join();
    joiner.join();
    EXPECT_EQ(joined, seg::CacheOutcome::Shared);
    EXPECT_EQ(builds.load(), 1);
    EXPECT_EQ(cache.get_or_build("done", [] { return fake_graph(1); }).outcome, seg::CacheOutcome::Miss);
    EXPECT_EQ(cache.get_or_build("k", gated).outcome, seg::CacheOutcome::Hit);
}

TEST(GraphCache, FreshlyBuiltEntryIsNotTheFirstEvicted) {
    seg::GraphCache cache(100, 25 * kKB);  // two 10 KB graphs
    cache.get_or_build("a", [] { return fake_graph(10); });
    cache.get_or_build("b", [] { return fake_graph(10); });
    Gate gate;
    std::atomic<bool> started{false};
    std::thread builder([&] {
        cache.get_or_build("c", [&] {
            started = true;
            gate.opened.wait();
            return fake_graph(10);
        });
    });
    while (!started) std::this_thread::yield();
    cache.get_or_build("a", [] { return fake_graph(10); });  // a and b are used during c's build
    cache.get_or_build("b", [] { return fake_graph(10); });
    gate.open.set_value();
    builder.join();
    EXPECT_EQ(cache.get_or_build("c", [] { return fake_graph(10); }).outcome, seg::CacheOutcome::Hit);
    EXPECT_EQ(cache.get_or_build("b", [] { return fake_graph(10); }).outcome, seg::CacheOutcome::Hit);
    EXPECT_EQ(cache.get_or_build("a", [] { return fake_graph(10); }).outcome, seg::CacheOutcome::Miss);
}

TEST(GraphCache, ZeroCapacityDisablesCaching) {
    seg::GraphCache cache(0, 10 * kKB);
    EXPECT_EQ(cache.get_or_build("a", [] { return fake_graph(1); }).outcome, seg::CacheOutcome::Miss);
    EXPECT_EQ(cache.get_or_build("a", [] { return fake_graph(1); }).outcome, seg::CacheOutcome::Miss);
}

TEST(GraphCacheKey, CoversExactlyTheGraphParameters) {
    const seg::SegmentParams base;
    const std::string key = seg::graph_cache_key("run", base);
    EXPECT_NE(seg::graph_cache_key("other_run", base), key);
    auto changed = [&](auto mutate) {
        seg::SegmentParams p = base;
        mutate(p);
        return seg::graph_cache_key("run", p) != key;
    };
    EXPECT_TRUE(changed([](seg::SegmentParams& p) { p.max_dim = 400; }));
    EXPECT_TRUE(changed([](seg::SegmentParams& p) { p.blur_sigma = 1.5; }));
    EXPECT_TRUE(changed([](seg::SegmentParams& p) { p.superpixels = 3001; }));
    EXPECT_TRUE(changed([](seg::SegmentParams& p) { p.slic_iterations = 11; }));
    EXPECT_TRUE(changed([](seg::SegmentParams& p) { p.graph.sigma_color = 5.000000001; }));
    EXPECT_TRUE(changed([](seg::SegmentParams& p) { p.graph.spatial_scale = 0.0; }));
    EXPECT_TRUE(changed([](seg::SegmentParams& p) { p.graph.edge_floor = 0.0; }));
    EXPECT_FALSE(changed([](seg::SegmentParams& p) { p.k = 7; }));
    EXPECT_FALSE(changed([](seg::SegmentParams& p) { p.method = seg::Method::Flat; }));
    EXPECT_FALSE(changed([](seg::SegmentParams& p) { p.ncut_threshold = 0.2; }));
    EXPECT_FALSE(changed([](seg::SegmentParams& p) { p.seed = 1; }));
    EXPECT_FALSE(changed([](seg::SegmentParams& p) { p.kmeans_attempts = 2; }));
    EXPECT_FALSE(changed([](seg::SegmentParams& p) { p.min_segment_superpixels = 3; }));
}

// The cache is only valid if a reused graph gives exactly what a fresh run gives, for every
// spectral setting that shares the graph.
TEST(PreparedGraph, ReusedGraphMatchesFreshSegmentation) {
    const cv::Mat bgr = cv::imread(std::string(SEG_DATA_DIR) + "/downtown_toronto.jpeg", cv::IMREAD_COLOR);
    ASSERT_FALSE(bgr.empty());
    const seg::SegmentParams base = fast_params();
    const seg::PreparedGraph shared = seg::prepare_graph(bgr, base);
    ASSERT_TRUE(shared.ok);
    for (int k : {4, 6, 8}) {
        for (seg::Method m : {seg::Method::Flat, seg::Method::Recursive}) {
            seg::SegmentParams p = base;
            p.k = k;
            p.method = m;
            SCOPED_TRACE(std::string(seg::to_string(m)) + " k=" + std::to_string(k));
            expect_same_segmentation(seg::segment_graph(shared, p), seg::segment(bgr, p));
        }
    }
}
