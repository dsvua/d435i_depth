// CUDA runtime
#include <cuda_runtime.h>
#include "assert.h"
#include "deproject_point.h"
#include "CudaPipeline.h"

#ifndef MINF
#define MINF __int_as_float(0xff800000)
#endif

__device__
void deproject_pixel_to_point_cuda(float3 * point, const struct rs2_intrinsics * intrin, const float2 pixel, float depth) {
    assert(intrin->model != RS2_DISTORTION_MODIFIED_BROWN_CONRADY); // Cannot deproject from a forward-distorted image
    assert(intrin->model != RS2_DISTORTION_FTHETA); // Cannot deproject to an ftheta image
    //assert(intrin->model != RS2_DISTORTION_BROWN_CONRADY); // Cannot deproject to an brown conrady model
    float x = (pixel.x - intrin->ppx) / intrin->fx;
    float y = (pixel.y - intrin->ppy) / intrin->fy;    
    if(intrin->model == RS2_DISTORTION_INVERSE_BROWN_CONRADY) {
        float r2  = x*x + y*y;
        float f = 1 + intrin->coeffs[0]*r2 + intrin->coeffs[1]*r2*r2 + intrin->coeffs[4]*r2*r2*r2;
        float ux = x*f + 2*intrin->coeffs[2]*x*y + intrin->coeffs[3]*(r2 + 2*x*x);
        float uy = y*f + 2*intrin->coeffs[3]*x*y + intrin->coeffs[2]*(r2 + 2*y*y);
        x = ux;
        y = uy;
    }
    point.x = depth * x;
    point.y = depth * y;
    point.z = depth;
    
}

__global__
void kernel_deproject_depth_cuda(float3 * points, const rs2_intrinsics* intrin, const uint16_t * depth, float * depth_f, 
    const int minDepth, const int maxDepth) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i >= (*intrin).height * (*intrin).width) {
        return;
    }
    int stride = blockDim.x * gridDim.x;
    int a, b;
    
    for (int j = i; j < (*intrin).height * (*intrin).width; j += stride) {
        b = j / (*intrin).width;
        a = j - b * (*intrin).width;
        const float2 pixel = make_float2(a, b);
        if (depth[j] > maxDepth || depth[j] < minDepth){
            depth_f[j] = MINF;
            deproject_pixel_to_point_cuda(points + j, intrin, pixel, 0);               
        } else {
            depth_f[j] = (float)depth[j] / 1000.0f; // convert mm to meter
            deproject_pixel_to_point_cuda(points + j, intrin, pixel, depth_f[j]);               
        }
    }
}

