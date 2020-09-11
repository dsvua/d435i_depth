// CUDA runtime
#include <cuda_runtime.h>
#include "assert.h"
#include "deproject_point.h"
#include "CudaPipeline.h"


__device__
void deproject_pixel_to_point_cuda(float points[3], const struct rs2_intrinsics * intrin, const float pixel[2], float depth) {
    assert(intrin->model != RS2_DISTORTION_MODIFIED_BROWN_CONRADY); // Cannot deproject from a forward-distorted image
    assert(intrin->model != RS2_DISTORTION_FTHETA); // Cannot deproject to an ftheta image
    //assert(intrin->model != RS2_DISTORTION_BROWN_CONRADY); // Cannot deproject to an brown conrady model
    float x = (pixel[0] - intrin->ppx) / intrin->fx;
    float y = (pixel[1] - intrin->ppy) / intrin->fy;    
    if(intrin->model == RS2_DISTORTION_INVERSE_BROWN_CONRADY)
    {
        float r2  = x*x + y*y;
        float f = 1 + intrin->coeffs[0]*r2 + intrin->coeffs[1]*r2*r2 + intrin->coeffs[4]*r2*r2*r2;
        float ux = x*f + 2*intrin->coeffs[2]*x*y + intrin->coeffs[3]*(r2 + 2*x*x);
        float uy = y*f + 2*intrin->coeffs[3]*x*y + intrin->coeffs[2]*(r2 + 2*y*y);
        x = ux;
        y = uy;
    } 
    points[0] = depth * x;
    points[1] = depth * y;
    points[2] = depth;
    
}


__global__
void kernel_deproject_depth_cuda(float * points, const rs2_intrinsics* intrin, const uint16_t * depth, 
    const int minDepth, const int maxDepth)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i >= (*intrin).height * (*intrin).width) {
        return;
    }
    int stride = blockDim.x * gridDim.x;
    int a, b;
    
    for (int j = i; j < (*intrin).height * (*intrin).width; j += stride) {
        b = j / (*intrin).width;
        a = j - b * (*intrin).width;
        const float pixel[] = { (float)a, (float)b };
        if (depth[j] > maxDepth || depth[j] < minDepth){
            deproject_pixel_to_point_cuda(points + j * 3, intrin, pixel, 0);               
        } else {
            deproject_pixel_to_point_cuda(points + j * 3, intrin, pixel, depth[j]);               
        }
   }
}
