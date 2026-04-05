cmake_minimum_required(VERSION 3.15...3.30)
project(spatioloji_s_cpp LANGUAGES CXX)

# Only build the C++ extension if the source file exists.
# This allows a pure-Python fallback when the C++ source is absent.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/spatioloji_s/ccc/_cpp/scoring_cpp.cpp")
    find_package(pybind11 CONFIG REQUIRED)
    add_subdirectory(src/spatioloji_s/ccc/_cpp)
else()
    message(STATUS "C++ source not found — building pure-Python wheel")
endif()
