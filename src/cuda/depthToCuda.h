#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API

#ifndef RS2_CUDA_THREADS_PER_BLOCK
#define RS2_CUDA_THREADS_PER_BLOCK 32
#endif

void upload_depth_to_cuda(int argc, char * argv[], rs2::depth_frame frame, rs2_intrinsics intristics);
