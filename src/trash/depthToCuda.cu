// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include "helper_functions.h"
#include "helper_cuda.h"

#include <iostream>             // for cout
#include "depthToCuda.h"
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API

__global__ void kernel_zero_column(uint16_t * depth, int img_h, int img_w)
{
    int x = 360 + threadIdx.x;
    for (int y=0; y<img_h; y++)
    {
        depth[y*img_w+x] = 0;
        // printf(" %d ", y*img_w+x);
    }
}

void upload_depth_to_cuda(int argc, char * argv[], rs2::depth_frame depth, rs2_intrinsics intristics)
{
    // This will pick the best possible CUDA capable device, otherwise
    // override the device ID based on input provided at the command line
    int dev = findCudaDevice(argc, (const char **)argv);

    int count = intristics.height * intristics.width;
    int numBlocks = count / RS2_CUDA_THREADS_PER_BLOCK;

    uint16_t *depth_data = (uint16_t *)depth.get_data();
    uint16_t *dev_depth = 0;
    uint16_t *modified_depth = 0;

    modified_depth = (uint16_t*) malloc (count);

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&dev_depth), count * sizeof(uint16_t)));
    checkCudaErrors(cudaMemcpy(dev_depth, depth_data, count * sizeof(uint16_t), cudaMemcpyHostToDevice));

    //------ this kernel will erase column in depth image. for testing purpose
    kernel_zero_column<<<1,128>>>(dev_depth, intristics.height, intristics.width);
    getLastCudaError("Failed: kernel_zero_column");
    checkCudaErrors(cudaMemcpy(modified_depth, dev_depth, count * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    std::cout << "Original " << depth_data[intristics.width+360] << "Modified " << modified_depth[intristics.width+360] << std::endl;
    //--------------------------


}