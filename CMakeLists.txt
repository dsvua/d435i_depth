
cmake_minimum_required(VERSION 3.1.0)

project(test_pipeline LANGUAGES CUDA CXX)

find_package(realsense2 REQUIRED)

# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Add the application sources to the target
add_subdirectory(src)

include_directories(${PROJECT_NAME} PUBLIC cuda-samples/Common)

#file(GLOB_RECURSE sources CONFIGURE_DEPENDS cuda-samples/Common/*.h)
#add_library(cuda_common ${sources})
