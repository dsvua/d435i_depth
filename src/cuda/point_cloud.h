#include <cuda_runtime.h>
#include <Eigen/Core>
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API

#ifndef MY_SLAM_POINT_CLOUD
#define MY_SLAM_POINT_CLOUD

__device__
void project_pixel_to_point_cuda(Eigen::Vector3f * point, const rs2_intrinsics * intrin, const float2 pixel, float depth);

__device__
Eigen::Vector3f project_pixel_to_point_cuda(const Eigen::Vector3f* pixel, const rs2_intrinsics * intrin) ;

__device__ 
Eigen::Vector3f project_point_to_pixel(const Eigen::Vector3f * point, const rs2_intrinsics * intrin);

#endif
