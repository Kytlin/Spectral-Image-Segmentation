include(GoogleTest)

# Correctness tests for each spectral clustering stage.
add_executable(segcore_tests test_region_graph.cpp test_spectral.cpp test_segmenter.cpp test_graph_cache.cpp)
target_link_libraries(segcore_tests PRIVATE segcore GTest::gtest_main seg_warnings)
target_compile_definitions(segcore_tests PRIVATE SEG_DATA_DIR="${SEG_DATA_DIR}")
gtest_discover_tests(segcore_tests)
