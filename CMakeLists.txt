cmake_minimum_required(VERSION 3.24)
project(spectral_segmentation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Eigen is very slow unoptimized, so default to -O2 with debug symbols.
# Override with -DCMAKE_BUILD_TYPE=Debug (or Release).
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

option(SEG_BUILD_TESTS "Build the unit tests" ON)

# ---- Dependencies -------------------------------------------------------------------------
# Eigen comes from the system (apt: libeigen3-dev; Ubuntu 24.04 ships 3.4.0).
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

# GoogleTest is fetched at configure time, pinned to a release tarball and verified by
# SHA-256, so every machine (and the Docker image later) builds the same code.
include(FetchContent)

# Warning flags shared by every target we own.
add_library(seg_warnings INTERFACE)
target_compile_options(seg_warnings INTERFACE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Wconversion -Wshadow>)

# Sample images shipped with the repo, for tests and the CLI.
set(SEG_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data)

# ---- Core library -------------------------------------------------------------------------
# Image I/O, colour conversion, preprocessing, SLIC and the region graph. Depends on Eigen only.
add_library(segcore STATIC
  src/image_io.cpp src/color.cpp src/preprocess.cpp src/slic.cpp src/region_graph.cpp
  src/visualize.cpp)
target_include_directories(segcore PUBLIC src)
target_include_directories(segcore SYSTEM PRIVATE third_party/stb)
target_link_libraries(segcore PUBLIC Eigen3::Eigen PRIVATE seg_warnings)

add_executable(graph_viz apps/graph_viz.cpp)
target_link_libraries(graph_viz PRIVATE segcore seg_warnings)

# ---- Tests --------------------------------------------------------------------------------
if(SEG_BUILD_TESTS)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    URL_HASH SHA256=7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()
  add_subdirectory(tests)
endif()

