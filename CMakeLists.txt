
cmake_minimum_required(VERSION 3.1.0)

project(test_pipeline LANGUAGES CUDA CXX)

find_package(realsense2 REQUIRED)
FIND_PACKAGE(PkgConfig REQUIRED)
pkg_check_modules(CUDA REQUIRED cuda-10.2)

# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

#set(CMAKE_CUDA_FLAGS "-Wall")
#set(CMAKE_CUDA_FLAGS_DEBUG "-g")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Add the application sources to the target
add_subdirectory(src)

include_directories(${PROJECT_NAME} PUBLIC cuda-samples/Common)
