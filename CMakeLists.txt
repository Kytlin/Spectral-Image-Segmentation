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
# Eigen and OpenCV come from the system
# (apt: libeigen3-dev libopencv-dev libopencv-contrib-dev; Ubuntu 24.04 ships 3.4.0 / 4.6.0).
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs ximgproc)

# Spectra and GoogleTest are fetched at configure time, pinned to a release tarball and
# verified by SHA-256, so every machine (and the Docker image later) builds the same code.
include(FetchContent)

# Spectra is header-only; we only need its include directory, not its install/test targets.
FetchContent_Declare(spectra
  URL https://github.com/yixuan/spectra/archive/refs/tags/v1.1.0.tar.gz
  URL_HASH SHA256=d29671e3d1b8036728933cadfddb05668a3cd6133331e91fc4535a9b85bedc79
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_GetProperties(spectra)
if(NOT spectra_POPULATED)
  FetchContent_Populate(spectra)
endif()
add_library(spectra INTERFACE)
add_library(Spectra::Spectra ALIAS spectra)
# SYSTEM: don't report warnings from headers we don't own.
target_include_directories(spectra SYSTEM INTERFACE ${spectra_SOURCE_DIR}/include)
target_link_libraries(spectra INTERFACE Eigen3::Eigen)

# HTTP server and JSON for the API: header-only, same versions as the original repository.
FetchContent_Declare(httplib
  URL https://github.com/yhirose/cpp-httplib/archive/refs/tags/v0.18.3.tar.gz
  URL_HASH SHA256=a0567bcd6c3fe5cef1b329b96245119047f876b49e06cc129a36a7a8dffe173e
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_Declare(nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_GetProperties(httplib)
if(NOT httplib_POPULATED)
  FetchContent_Populate(httplib)
endif()
FetchContent_GetProperties(nlohmann_json)
if(NOT nlohmann_json_POPULATED)
  FetchContent_Populate(nlohmann_json)
endif()
find_package(Threads REQUIRED)
add_library(httplib INTERFACE)
target_include_directories(httplib SYSTEM INTERFACE ${httplib_SOURCE_DIR})
target_link_libraries(httplib INTERFACE Threads::Threads)
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json SYSTEM INTERFACE ${nlohmann_json_SOURCE_DIR}/single_include)

# Warning flags shared by every target we own.
add_library(seg_warnings INTERFACE)
target_compile_options(seg_warnings INTERFACE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Wconversion -Wshadow>)

# Sample images shipped with the repo, for tests and the CLI.
set(SEG_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data)

# ---- Core library -------------------------------------------------------------------------
# Region graph, spectral clustering (Spectra + OpenCV k-means), and the SLIC pipeline.
add_library(segcore STATIC src/region_graph.cpp src/spectral.cpp src/segmenter.cpp)
target_include_directories(segcore PUBLIC src)
target_include_directories(segcore SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(segcore PUBLIC Eigen3::Eigen Spectra::Spectra ${OpenCV_LIBS} PRIVATE seg_warnings)

add_executable(segment apps/segment.cpp)
target_link_libraries(segment PRIVATE segcore seg_warnings)

# ---- HTTP API -----------------------------------------------------------------------------
# In-memory run/result store, JSON codec, and HTTP routes around seg::segment().
add_library(segapi STATIC src/api/run_store.cpp src/api/codec.cpp src/api/routes.cpp)
target_include_directories(segapi PUBLIC src)
target_link_libraries(segapi PUBLIC segcore nlohmann_json httplib PRIVATE seg_warnings)

add_executable(seg_server apps/server.cpp)
target_link_libraries(seg_server PRIVATE segapi seg_warnings)

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

