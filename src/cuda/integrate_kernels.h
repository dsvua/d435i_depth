#include "hash_functions.h"
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API

// static float4x4 toCUDA(const float4x4& m);

// static float4x4 toCUDA(const Eigen::Matrix4f& mat);

__global__ void resetHashKernel(HashData hashData);

__global__ void allocKernel(HashData hashData, const float * depth, const struct rs2_intrinsics * dev_intrin);

__global__ void garbageCollectIdentifyKernel(HashData hashData);

__global__ void garbageCollectFreeKernel(HashData hashData);

__global__ void compactifyHashAllInOneKernel(HashData hashData, const struct rs2_intrinsics * dev_intrin);

__global__ void integrateDepthMapKernel(HashData hashData, const float * depth, const struct rs2_intrinsics * dev_intrin);

