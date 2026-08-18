cmake_minimum_required(VERSION 3.10.0)
project(mlfs VERSION 0.1.0 LANGUAGES C CXX)

find_package(Eigen3 REQUIRED)
find_package(rapidcsv REQUIRED)

add_library(mlfs STATIC
  include/core/model.h
  include/core/types.h
  include/regression/linear_regression.h
  src/regression/linear_regression.cpp
  examples/main.cpp
  include/core/losses.h
  src/core/losses.cpp
  include/core/utils.h
  include/core/data.h
  include/core/encoders.h
  src/core/encoders.cpp
  src/core/data.cpp)

include(CTest)
enable_testing()

target_include_directories(mlfs
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(mlfs PUBLIC Eigen3::Eigen rapidcsv::rapidcsv)

option(MLFS_BUILD_EXAMPLES "Build mlfs example programs" ON)

if(MLFS_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()