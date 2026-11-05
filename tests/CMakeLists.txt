include(GoogleTest)

add_executable(toolchain_tests test_toolchain.cpp)
target_link_libraries(toolchain_tests PRIVATE GTest::gtest_main seg_warnings)
target_compile_definitions(toolchain_tests PRIVATE SEG_DATA_DIR="${SEG_DATA_DIR}")
gtest_discover_tests(toolchain_tests)

# Stage 1 contract tests: SLIC output and the region graph.
add_executable(segcore_tests test_slic.cpp test_region_graph.cpp)
target_link_libraries(segcore_tests PRIVATE segcore GTest::gtest_main seg_warnings)
gtest_discover_tests(segcore_tests)
