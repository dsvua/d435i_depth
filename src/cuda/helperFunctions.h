#include <vector> // for 2D vector
#include <cuda_runtime.h>

std::vector<float3> convertFloatPointsToVectorPoint(float3 *_points, int count);
void writeToPly(std::vector<float3> points, const char* fileName);