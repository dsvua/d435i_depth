include "SlamPipeline.h"
// #include <cmath>

SlamPipeline::SlamPipeline(rs2_intrinsics intristics) {
    _h_intristics = intristics;
    checkCudaErrors(cudaMalloc(&_d_voxelHashData, sizeof(VoxelHashMap)));

    checkCudaErrors(cudaMalloc(&_d_intristics, sizeof(rs2_intrinsics)));
    checkCudaErrors(cudaMemcpy(_d_intristics, &_h_intristics, sizeof(rs2_intrinsics), cudaMemcpyHostToDevice));

    *_d_voxelHashData = VoxelHashMap(true, 500000, 1000000000);
    _h_voxelHashData = VoxelHashMap(false, 500000, 1000000000);

    const int HFOV = 90; // horizontal field of view for realsense camera
    const float subpixel = 0.2;
    const int baseline = 50;
    float focalLength_mm = _h_intristics.width * tan(HFOV/2) / 2
    staticPartOfRMS = subpixel / (focalLength_mm * baseline)

}