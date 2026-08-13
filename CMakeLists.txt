cmake_minimum_required(VERSION 3.10.0)
project(mlfs VERSION 0.1.0 LANGUAGES C CXX)

find_package(Eigen3 REQUIRED)

add_library(mlfs STATIC
  include/core/model.h
  include/core/regression.h
  src/core/regression.cpp)

include(CTest)
enable_testing()

target_include_directories(mlfs
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(mlfs PUBLIC Eigen3::Eigen)