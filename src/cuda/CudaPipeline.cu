
// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include "helper_functions.h"
#include "helper_cuda.h"

#include <iostream>             // for cout
#include "deproject_point.h"
#include "CudaPipeline.h"
#include "cuda_hash_params.h"
#include "hash_functions.h"

int CudaPipeline::init_cuda_device(int argc, char **argv)
{
    checkCudaErrors(cudaMalloc(&dev_intrin, sizeof(rs2_intrinsics)));
    checkCudaErrors(cudaMemcpy(dev_intrin, &_intristics, sizeof(rs2_intrinsics), cudaMemcpyHostToDevice));

    return findCudaDevice(argc, (const char **)argv);
}

void CudaPipeline::process_depth(rs2::depth_frame depth_frame)
{
    // upload
    int count = _intristics.height * _intristics.width;
    int numBlocks = count / RS2_CUDA_THREADS_PER_BLOCK;

    uint16_t *depth_data = (uint16_t *)depth_frame.get_data();
    uint16_t *dev_depth = 0;
    float *dev_points = 0;

    checkCudaErrors(cudaMalloc(&dev_points, count * sizeof(float) * 3));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&dev_depth), count * sizeof(uint16_t)));
    checkCudaErrors(cudaMemcpy(dev_depth, depth_data, count * sizeof(uint16_t), cudaMemcpyHostToDevice));

    std::cout << "image uploaded to gpu" << std::endl;

    // counverting depth image into points cloud and keeping points on gpu device
    kernel_deproject_depth_cuda<<<numBlocks, RS2_CUDA_THREADS_PER_BLOCK>>>(dev_points, dev_intrin, dev_depth,
        minDistance, maxDistance); 
    getLastCudaError("Failed: kernel_deproject_depth_cuda");

    std::cout << "Points cloud is computed" << std::endl;

    if (download_points){
        checkCudaErrors(cudaMemcpy(host_points, dev_points, count * sizeof(float) * 3, cudaMemcpyDeviceToHost));
    };

    std::cout << "Points cloud is downloaded to host" << std::endl;

    // reconstruction begins
    float4x4 transformation = float4x4::identity();

    if (m_numIntegratedFrames > 0){
        // need to find rotation first
    }

    integrate(transformation, dev_depth);

    std::cout << "Image is integrated" << std::endl;

}