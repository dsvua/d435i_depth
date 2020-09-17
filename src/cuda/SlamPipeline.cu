#include "SlamPipeline.h"
#include "safe_call.h"
// #include <cmath>

SlamPipeline::SlamPipeline(rs2_intrinsics intristics) {
    const int HFOV = 90; // horizontal field of view for realsense camera
    const float subpixel = 0.2;
    const int baseline = 50;
    float focalLength_mm = intristics.width * tan(HFOV/2) / 2;

    std::cout << "Initializing VoxelHashParameters" << std::endl;
    _h_voxelHashData = new VoxelHashMap();
    _h_voxelHashData->params = new VoxelHashParameters();
    _h_voxelHashData->params->staticPartOfRMS = subpixel / (focalLength_mm * baseline);


    std::cout << "Initializing VoxelHashMap" << std::endl;
    // Allocating memory on device for each pointer element of _d_voxelHashData
    VoxelHashEntry**	    d_tmp_voxelsHash;
    VoxelHashEntry**	    d_tmp_voxelsHashCompactified;
    int*	                d_tmp_deletedVoxelBlocks;
    int*	                d_tmp_mutex;
    VoxelHashParameters*    d_tmp_voxelHashParams;

    // creating hashData
    cudaSafeCall(cudaMalloc((void**)&_d_voxelHashData, sizeof(VoxelHashMap)));
    // creating pointers arrays
    cudaSafeCall(cudaMalloc((void**)&d_tmp_voxelHashParams, sizeof(VoxelHashParameters)));
    cudaSafeCall(cudaMemcpy(d_tmp_voxelHashParams, _h_voxelHashData->params, sizeof(VoxelHashParameters), cudaMemcpyHostToDevice));

    cudaSafeCall(cudaMalloc((void**)&d_tmp_voxelsHash, sizeof(VoxelHashEntry) * _h_voxelHashData->params->voxelHashTotalSize));
    cudaSafeCall(cudaMalloc((void**)&d_tmp_voxelsHashCompactified, sizeof(VoxelHashEntry) * _h_voxelHashData->params->voxelHashTotalSize));
    cudaSafeCall(cudaMalloc((void**)&d_tmp_deletedVoxelBlocks, sizeof(uint) * _h_voxelHashData->params->voxelHashSize));
    cudaSafeCall(cudaMalloc((void**)&d_tmp_mutex, sizeof(int) * _h_voxelHashData->params->voxelHashSize));
    // cudaSafeCall(cudaMemset(d_tmp_mutex, 0, sizeof(uint) * _voxelHashSize));

    // NOTE: Binding pointers with _d_voxelHashData on device
    cudaSafeCall(cudaMemcpy(&(_d_voxelHashData->params), &d_tmp_voxelHashParams, sizeof(_d_voxelHashData->params), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(&(_d_voxelHashData->voxelsHash), &d_tmp_voxelsHash, sizeof(_d_voxelHashData->voxelsHash), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(&(_d_voxelHashData->voxelsHashCompactified), &d_tmp_voxelsHashCompactified, sizeof(_d_voxelHashData->voxelsHashCompactified), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(&(_d_voxelHashData->deletedVoxelBlocks), &d_tmp_deletedVoxelBlocks, sizeof(_d_voxelHashData->deletedVoxelBlocks), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(&(_d_voxelHashData->mutex), &d_tmp_mutex, sizeof(_d_voxelHashData->mutex), cudaMemcpyHostToDevice));
    // copy params


    std::cout << "running _d_voxelHashData.initialize()" << std::endl;
    _d_voxelHashData->initialize();
    std::cout << "Done initializing _d_voxelHashData" << std::endl;

    cudaSafeCall(cudaMalloc(&_d_intristics, sizeof(rs2_intrinsics)));
    cudaSafeCall(cudaMemcpy(_d_intristics, &_h_intristics, sizeof(rs2_intrinsics), cudaMemcpyHostToDevice));
}