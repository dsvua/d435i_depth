#include <stdint.h> 

__device__
void deproject_pixel_to_point_cuda(float points[3], const struct rs2_intrinsics * intrin, const float pixel[2], float depth);
__global__
void kernel_deproject_depth_cuda(float * points, const rs2_intrinsics* intrin, const uint16_t * depth, const int minDepth, const int maxDepth);