################################################################################
####################### Do not modify this file. ###############################
################################################################################

project(initial_matrix)
cmake_minimum_required(VERSION 3.1)

set (CMAKE_CXX_STANDARD 11)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -fPIC -DEIGEN_STACK_ALLOCATION_LIMIT=0")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)


add_library(libicp STATIC ${PROJECT_SOURCE_DIR}/src/icp.cpp)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/lib)

add_executable(my_executable main.cpp)

target_link_libraries(my_executable PRIVATE Threads::Threads)
target_include_directories(my_executable PRIVATE ${PROJECT_SOURCE_DIR}/lib/)
