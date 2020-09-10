
// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include "helper_functions.h"
#include "helper_cuda.h"

#include <iostream>             // for cout
#include "CudaPipeline.h"


int CudaPipeline::get_cuda_device(int argc, char **argv)
{
    return findCudaDevice(argc, (const char **)argv);
}

void CudaPipeline::process_depth(rs2::depth_frame depth_frame)
{
    // upload
    int count = _intristics.height * _intristics.width;
    int numBlocks = count / RS2_CUDA_THREADS_PER_BLOCK;

    uint16_t *depth_data = (uint16_t *)depth_frame.get_data();
    uint16_t *dev_depth = 0;

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&dev_depth), count * sizeof(uint16_t)));
    checkCudaErrors(cudaMemcpy(dev_depth, depth_data, count * sizeof(uint16_t), cudaMemcpyHostToDevice));

    std::cout << "image uploaded to gpu" << std::endl;



}