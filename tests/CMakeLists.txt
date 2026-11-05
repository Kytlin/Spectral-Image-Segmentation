include(GoogleTest)

# Stage 0: checks that the toolchain and dependencies work together. No project code yet.
add_executable(toolchain_tests test_toolchain.cpp)
target_link_libraries(toolchain_tests PRIVATE Spectra::Spectra GTest::gtest_main seg_warnings)
target_compile_definitions(toolchain_tests PRIVATE SEG_SAMPLE_DIR="${SEG_SAMPLE_DIR}")

gtest_discover_tests(toolchain_tests)
