#include "CudaPipeline.h"
#include "cuda_simple_matrix_math.h"
#include "helper_functions.h"
#include "helper_cuda.h"
#include "integrate_kernels.h"

void CudaPipeline::integrate(const float4x4& lastRigidTransform, const float * dev_depth_f) {

    setLastRigidTransform(lastRigidTransform);
		
    //make the rigid transform available on the GPU
    m_hashData.updateParams(m_hashParams);

    // I am not sure we need to reset hash on every integration
    // -----
	// const dim3 gridSize((m_hashParams.m_hashNumBuckets + (T_PER_BLOCK*T_PER_BLOCK) - 1)/(T_PER_BLOCK*T_PER_BLOCK), 1);
	// const dim3 blockSize((T_PER_BLOCK*T_PER_BLOCK), 1);
    // resetHashKernel<<<gridSize, blockSize>>>(hashData);
    // getLastCudaError("Failed: resetHashKernel");


    //allocate all hash blocks which are corresponding to depth map entries
    //this version is faster, but it doesn't guarantee that all blocks are allocated (staggers alloc to the next frame)    
	dim3 gridSize((_intristics.width + T_PER_BLOCK - 1)/T_PER_BLOCK, (_intristics.height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);
	allocKernel<<<gridSize, blockSize>>>(m_hashData, dev_depth_f, dev_intrin);
    getLastCudaError("Failed: allocKernel");

    //generate a linear hash array with only occupied entries
    // -- compactifyHashEntries(depthCameraData);
 
    int threadsPerBlock = COMPACTIFY_HASH_THREADS_PER_BLOCK;
	gridSize = dim3((HASH_BUCKET_SIZE * m_hashParams.m_hashNumBuckets + threadsPerBlock - 1) / threadsPerBlock, 1);
	blockSize = dim3(threadsPerBlock, 1);

	checkCudaErrors(cudaMemset(m_hashData.d_hashCompactifiedCounter, 0, sizeof(int)));
	compactifyHashAllInOneKernel << <gridSize, blockSize >> >(m_hashData, dev_intrin);
    getLastCudaError("Failed: compactifyHashAllInOneKernel");
	unsigned int res = 0;
	checkCudaErrors(cudaMemcpy(&res, m_hashData.d_hashCompactifiedCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost));

    m_hashParams.m_numOccupiedBlocks = res;		//this version uses atomics over prefix sums, which has a much better performance
    m_hashData.updateParams(m_hashParams);	//make sure numOccupiedBlocks is updated on the GPU

    // -- integrateDepthMapCUDA
    //volumetrically integrate the depth data into the depth SDFBlocks
	threadsPerBlock = SDF_BLOCK_SIZE*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE;
	gridSize = dim3(m_hashParams.m_numOccupiedBlocks, 1);
	blockSize = dim3(threadsPerBlock, 1);

	if (m_hashParams.m_numOccupiedBlocks > 0) {	//this guard is important if there is no depth in the current frame (i.e., no blocks were allocated)
		integrateDepthMapKernel << <gridSize, blockSize >> >(m_hashData, dev_depth_f, dev_intrin);
        getLastCudaError("Failed: integrateDepthMapKernel");
    };
    
    // -- garbageCollectIdentifyCUDA
    threadsPerBlock = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE / 2;
	gridSize = dim3(m_hashParams.m_numOccupiedBlocks, 1);
	blockSize = dim3(threadsPerBlock, 1);

	if (m_hashParams.m_numOccupiedBlocks > 0) {
		garbageCollectIdentifyKernel << <gridSize, blockSize >> >(m_hashData);
        getLastCudaError("Failed: garbageCollectIdentifyKernel");
    }
    
    // -- garbageCollectFreeCUDA
	threadsPerBlock = T_PER_BLOCK*T_PER_BLOCK;
	gridSize = dim3((m_hashParams.m_numOccupiedBlocks + threadsPerBlock - 1) / threadsPerBlock, 1);
	blockSize = dim3(threadsPerBlock, 1);
	
	if (m_hashParams.m_numOccupiedBlocks > 0) {
		garbageCollectFreeKernel << <gridSize, blockSize >> >(m_hashData);
        getLastCudaError("Failed: garbageCollectFreeKernel");
	}


    m_numIntegratedFrames++;
}
