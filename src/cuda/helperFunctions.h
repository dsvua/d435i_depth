#include <vector> // for 2D vector
#include <cuda_runtime.h>

#ifndef MY_SLAM_HELPERS
#define MY_SLAM_HELPERS

std::vector<float3> convertFloatPointsToVectorPoint(float3 *_points, int count);
void writeToPly(std::vector<float3> points, const char* fileName);

#endif