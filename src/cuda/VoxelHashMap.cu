#include "VoxelHashMap.h"
// #include <cmath>

__device__ __host__
uint VoxelHashMap::computeHashPos(int x, int y, int z) const {
    return abs((x*p0 + y*p1 + z*p2)/voxelHashSize);
}

//merges two voxels (v0 the currently stored voxel, v1 is the input voxel)
__device__ __host__
void VoxelHashMap::combineVoxel(const Voxel &v0, const Voxel& v1, Voxel &out) const {

    out.sdf = (v0.sdf * (float)v0.weight + v1.sdf * (float)v1.weight) / ((float)v0.weight + (float)v1.weight);
    out.weight = min(c_hashParams.m_integrationWeightMax, (unsigned int)v0.weight + (unsigned int)v1.weight);
}
