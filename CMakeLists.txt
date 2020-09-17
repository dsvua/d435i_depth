
cmake_minimum_required(VERSION 3.1.0)

project(test_pipeline LANGUAGES CXX CUDA)

enable_language(CUDA)

#find_package(CUDAToolkit REQUIRED)
find_package(realsense2 REQUIRED)
find_package(PkgConfig REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package (Sophus REQUIRED NO_MODULE)

#pkg_check_modules(t_CUDA REQUIRED IMPORTED_TARGET cuda-10.2)
#pkg_check_modules(eigen3 REQUIRED IMPORTED_TARGET eigen3)

#set(EIGEN3_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/external/eigen/Eigen)

# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
# disable warnings

#set(CMAKE_CUDA_FLAGS "-Wall")
#set(CMAKE_CUDA_FLAGS_DEBUG "-g")
#set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

#set(CMAKE_CUDA_STANDARD 11)
#set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Add the application sources to the target
add_subdirectory(src)
add_subdirectory(external/eigen)
add_subdirectory(external/Sophus)
add_subdirectory(libs/ICPCUDA)

#target_link_libraries(sophus INTERFACE Eigen3::Eigen)

