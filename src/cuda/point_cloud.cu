#include <cuda_runtime.h>
#include "assert.h"
#include "point_cloud.h"

__device__
void project_pixel_to_point_cuda(Eigen::Vector3f * point, const rs2_intrinsics * intrin, const float2 pixel, float depth) {
    float x = (pixel.x - intrin->ppx) / intrin->fx;
    float y = (pixel.y - intrin->ppy) / intrin->fy;    
    point->x() = depth * x;
    point->y() = depth * y;
    point->z() = depth;
}

__device__
Eigen::Vector3f project_pixel_to_point_cuda(const Eigen::Vector3f* pixel, const rs2_intrinsics * intrin) {
    Eigen::Vector3f point;
    float x = (pixel->x() - intrin->ppx) / intrin->fx;
    float y = (pixel->y() - intrin->ppy) / intrin->fy;    
    point.x() = pixel->z() * x;
    point.y() = pixel->z() * y;
    point.z() = pixel->z();
    return point;
}

__device__ 
Eigen::Vector3f project_point_to_pixel(const Eigen::Vector3f * point, const rs2_intrinsics * intrin) {
    //assert(intrin->model != RS2_DISTORTION_INVERSE_BROWN_CONRADY); // Cannot project to an inverse-distorted image

    float x = point->x() / point->z();
    float y = point->y() / point->z();
    Eigen::Vector3f pixel;
    pixel.x() = x * intrin->fx + intrin->ppx;
    pixel.y() = y * intrin->fy + intrin->ppy;
    pixel.z() = point->z();
    return pixel;
}
