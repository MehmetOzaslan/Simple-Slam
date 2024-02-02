################################################################################
####################### Do not modify this file. ###############################
################################################################################

project(initial_matrix)
cmake_minimum_required(VERSION 3.1)

set (CMAKE_CXX_STANDARD 11)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -fPIC")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")


set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

include_directories(${PROJECT_SOURCE_DIR}/include)

# Include the Eigen directory
include_directories(${PROJECT_SOURCE_DIR}/eigen-3.4.0)
add_subdirectory(src)
add_executable(my_executable main.cpp)
