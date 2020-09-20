#include "SlamPipeline.h"
#include "point_cloud.h"
#include <cuda_runtime.h>
#include "safe_call.h"
// #include <cmath>

__global__ void compactifyHashKernel(uint16_t * depth, const Eigen::Affine3f* d_currentTransformation,
    VoxelHashMap * _d_voxelHashData, const rs2_intrinsics* _d_intristics) {
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    float depthTruncationDistance = _d_voxelHashData->params->maxIntegrationDistance + _d_voxelHashData->getTruncation(_d_voxelHashData->params->maxIntegrationDistance);
    // point with max depth
    Eigen::Vector3f maxPoint(x, y, depthTruncationDistance);
    maxPoint = project_pixel_to_point_cuda(&maxPoint, _d_intristics);
    Eigen::Vector3f minPoint(x, y, 0); // point on camera
    Eigen::Vector3f deltaPoint;
    Eigen::Vector3f currentBlockPos;
    const Eigen::Affine3f cT = *d_currentTransformation;

    maxPoint = cT * maxPoint;
    minPoint = cT * minPoint;
    deltaPoint = (maxPoint - minPoint) / depthTruncationDistance;
    for (int d = _d_voxelHashData->params->voxelBlockSideSize; d < depthTruncationDistance; d+=_d_voxelHashData->params->voxelBlockSideSize){
        currentBlockPos = _d_voxelHashData->voxelBlockPosFromWorldPos(deltaPoint * d);
        int currHashIdx = _d_voxelHashData->getHashEntryByPosition(currentBlockPos);
        if (currHashIdx == FREE_ENTRY) {
            // voxelBlock/hashEntry was not found - allocate
            VoxelHashEntry tmp_entry = VoxelHashEntry();
            tmp_entry.world_pos = currentBlockPos;
            tmp_entry.voxelBlockIdx = _d_voxelHashData->getFreeVoxelBlock();
            currHashIdx = _d_voxelHashData->insertHashEntry(tmp_entry);
            if (currHashIdx != FAIL_TO_INSERT) {
                int tmp_idx = atomicAdd(&(_d_voxelHashData->voxelsHashCompactifiedCount), 1);
                _d_voxelHashData->voxelsHashCompactified[tmp_idx - 1] = _d_voxelHashData->voxelsHash[currHashIdx];
            }
        } else {
            int tmp_idx = atomicAdd(&(_d_voxelHashData->voxelsHashCompactifiedCount), 1);
            _d_voxelHashData->voxelsHashCompactified[tmp_idx - 1] = _d_voxelHashData->voxelsHash[currHashIdx];
        }
    }
}

__global__ void integrateDepthKernel(const uint16_t* depth,
                                    const Eigen::Affine3f* d_currentTransformationInverted, 
                                    VoxelHashMap* _d_voxelHashData, 
                                    const rs2_intrinsics* _d_intristics) {
    VoxelHashEntry voxelHashEntry= _d_voxelHashData->voxelsHashCompactified[blockIdx.x];
    uint voxelIdx = threadIdx.x;
    Eigen::Vector3f voxelWorldPos = voxelHashEntry.world_pos + _d_voxelHashData->delinearizeVoxelPosition(voxelIdx);
    voxelWorldPos = *d_currentTransformationInverted * voxelWorldPos;
    Eigen::Vector3f reprojectedPoint = project_point_to_pixel(&voxelWorldPos, _d_intristics);
    // float depthTruncationDistance = _d_voxelHashData->params->maxIntegrationDistance + _d_voxelHashData->getTruncation(reprojectedPoint.z());

    if( reprojectedPoint.x() > 0 && reprojectedPoint.x() < _d_intristics->width &&
        reprojectedPoint.y() > 0 && reprojectedPoint.y() < _d_intristics->width
    ){
        uint d = depth[(int)(reprojectedPoint.y() * _d_intristics->width + reprojectedPoint.x())];
        float maxDistance = _d_voxelHashData->params->maxIntegrationDistance + _d_voxelHashData->getTruncation(_d_voxelHashData->params->maxIntegrationDistance);
        float truncation = _d_voxelHashData->getTruncation(d);
        float depthNormalized = (maxDistance - d) / maxDistance;
        float sdf = d - reprojectedPoint.z();
        if (sdf >= 0.0f) {
            sdf = fminf(truncation, sdf);
        } else {
            sdf = fmaxf(-truncation, sdf);
        }
        float weightUpdate = max(_d_voxelHashData->params->integrationWeightMin * 1.5f * (1.0f-depthNormalized), 1.0f);
        Voxel curr_voxel = Voxel();
        Voxel tmp_voxel = Voxel();
        curr_voxel.sdf = sdf;
        curr_voxel.weight = weightUpdate;
        _d_voxelHashData->combineVoxels((_d_voxelHashData->voxels[voxelHashEntry.voxelBlockIdx * _d_voxelHashData->params->voxelBlockCubeSize + voxelIdx]), curr_voxel, tmp_voxel);
        _d_voxelHashData->voxels[voxelHashEntry.voxelBlockIdx * _d_voxelHashData->params->voxelBlockCubeSize + voxelIdx] = tmp_voxel;
    }

}

void SlamPipeline::integrateDepth(rs2::depth_frame depth) {

    _h_currentAffineTransformation.matrix() = _currentTransformation.cast<float>().matrix();
    _h_currentAffineTransformationInverted = _h_currentAffineTransformation.inverse(Eigen::TransformTraits::Affine);
    cudaSafeCall(cudaMemcpy(_d_currentAffineTransformation, &_h_currentAffineTransformation, sizeof(Eigen::Affine3f), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(_d_currentAffineTransformationInverted, &_h_currentAffineTransformationInverted, sizeof(Eigen::Affine3f), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(_d_depth, depth.get_data(), sizeof(uint16_t) * _h_intristics.width * _h_intristics.height, cudaMemcpyHostToDevice));

    // cudaSafeCall(cudaMalloc((void**)&d_currentTransformationInverted, sizeof(Eigen::Affine3f)));

    cudaSafeCall(cudaMemset(&(_d_voxelHashData->voxelsHashCompactifiedCount), 0, sizeof(uint)));

    dim3 block(32, 8);
    dim3 grid(1, 1, 1);  
    grid.x = divUp(_h_intristics.width, block.x);
    grid.y = divUp(_h_intristics.height, block.y);

    compactifyHashKernel<<<grid, block>>>(_d_depth, _d_currentAffineTransformation, _d_voxelHashData, _d_intristics);
    cudaSafeCall(cudaGetLastError());

    int _h_deletedVoxelBlocksLength = 0;
    cudaSafeCall(cudaMemcpy(&_h_deletedVoxelBlocksLength, &(_d_voxelHashData->deletedVoxelBlocksLength), sizeof(int), cudaMemcpyDeviceToHost));

    grid = dim3(_d_voxelHashData->deletedVoxelBlocksLength, 1);
    block = dim3(_h_voxelHashData->params->voxelBlockCubeSize, 1);
    integrateDepthKernel<<<grid, block>>>(_d_depth, _d_currentAffineTransformationInverted, _d_voxelHashData, _d_intristics);
    cudaSafeCall(cudaGetLastError());
}