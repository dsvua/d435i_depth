#include "SlamPipeline.h"
#include "point_cloud.h"
#include <cuda_runtime.h>
#include "safe_call.h"

__global__ void renderKernel(Eigen::Affine3f* d_currentTransformation,
                            PtrStep<float> vmap_prev, int rows,
                            VoxelHashMap* _d_voxelHashData, 
                            const rs2_intrinsics* _d_intristics) {
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    float maxDistance = _d_voxelHashData->params->maxDistanceForICP;
    // point with max depth
    Eigen::Vector3f maxPoint(x, y, _d_voxelHashData->params->maxDistanceForICP);
    maxPoint = project_pixel_to_point_cuda(&maxPoint, _d_intristics);
    Eigen::Vector3f minPoint(x, y, 0); // point on camera
    Eigen::Vector3f direction;
    Eigen::Vector3f currentPos;
    Eigen::Vector3f currentBlockPos;
    Eigen::Vector3f currentVoxelPos;
    int depthTruncationDistance = _d_voxelHashData->getTruncation(maxDistance);

    maxPoint = *d_currentTransformation * maxPoint;
    minPoint = *d_currentTransformation * minPoint;
    direction = (maxPoint - minPoint) / depthTruncationDistance;

    Eigen::ParametrizedLine<float, 3> ray = Eigen::ParametrizedLine<float, 3>::Through(minPoint, maxPoint);

    for(int d = _d_voxelHashData->params->minIntegrationDistance; d<_d_voxelHashData->params->maxDistanceForICP; d+=_d_voxelHashData->params->voxelPhysicalSize) {
        // get voxel
        currentPos += direction * d;
        currentBlockPos = _d_voxelHashData->voxelBlockPosFromWorldPos(currentPos);
        int currHashIdx = _d_voxelHashData->getHashEntryByPosition(currentBlockPos);
        uint currentVoxelIndex = _d_voxelHashData->linearizeVoxelWorldPosition(currentPos - currentBlockPos);
        float sdf = _d_voxelHashData->voxels[_d_voxelHashData->voxelsHash[currHashIdx].voxelBlockIdx * _d_voxelHashData->params->voxelBlockCubeSize].sdf;
        currentVoxelPos = currentBlockPos + _d_voxelHashData->delinearizeVoxelPosition(currentVoxelIndex) * _d_voxelHashData->params->voxelPhysicalSize;
        float distance = ray.distance(currentVoxelPos);
        if ( sdf > 0 && distance < sdf) {
            vmap_prev.ptr(y)[x] = x;
            vmap_prev.ptr(y+rows)[x] = y;
            vmap_prev.ptr(y+rows*2)[x] = d;
            return;
        }
    }
    vmap_prev.ptr(y)[x] = 0;

}

__global__ void reduceVMapKernel(PtrStep<float> dst, PtrStep<float> src, int dst_rows, int src_rows) {
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    dst.ptr(y)[x] = src.ptr(y * 2)[x];
    dst.ptr(y + dst_rows)[x] = src.ptr(y * 2 + src_rows)[x];
    dst.ptr(y + dst_rows * 2)[x] = src.ptr(y * 2 + src_rows * 2)[x];
}

void SlamPipeline::reduceVMap(DeviceArray2D<float> &dst, DeviceArray2D<float> &src) {

    dim3 grid((dst.cols() + _h_voxelHashData->params->threadsPerBlock - 1)/_h_voxelHashData->params->threadsPerBlock, 
              (dst.rows() + _h_voxelHashData->params->threadsPerBlock - 1)/_h_voxelHashData->params->threadsPerBlock);
    dim3 block(_h_voxelHashData->params->threadsPerBlock, _h_voxelHashData->params->threadsPerBlock);

    reduceVMapKernel<<<grid, block>>>(dst, src, dst.rows(), src.rows());
    cudaSafeCall(cudaGetLastError());    

}

void SlamPipeline::renderVoxels() {
    dim3 grid(( _h_intristics.width + _h_voxelHashData->params->threadsPerBlock - 1)/_h_voxelHashData->params->threadsPerBlock, 
              (_h_intristics.height + _h_voxelHashData->params->threadsPerBlock - 1)/_h_voxelHashData->params->threadsPerBlock);
    dim3 block(_h_voxelHashData->params->threadsPerBlock, _h_voxelHashData->params->threadsPerBlock);

    _h_currentAffineTransformation.matrix() = _currentTransformation.cast<float>().matrix();
    _h_currentAffineTransformationInverted = _h_currentAffineTransformation.inverse(Eigen::TransformTraits::Affine);

    cudaSafeCall(cudaMemcpy(_d_currentAffineTransformation, &_h_currentAffineTransformation, sizeof(Eigen::Affine3f), cudaMemcpyHostToDevice));

    renderKernel<<<grid, block>>>(_d_currentAffineTransformation, _icp->vmaps_prev[0], _icp->vmaps_prev[0].rows(), 
                                  _d_voxelHashData, _d_intristics);
    cudaSafeCall(cudaGetLastError());


    createNMap(_icp->vmaps_prev[0], _icp->nmaps_prev[0]);
    for (int i = 1; i < _icp->NUM_PYRS; ++i) {
        reduceVMap(_icp->vmaps_prev[i-1], _icp->vmaps_prev[i]);
        createNMap(_icp->vmaps_prev[i], _icp->nmaps_prev[i]);
      }
    
}