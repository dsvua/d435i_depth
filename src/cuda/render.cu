// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include "helper_functions.h"
#include "helper_cuda.h"
#include "helper_math.h"

#include <iostream>             // for cout
#include "deproject_point.h"
#include "CudaPipeline.h"
#include "cuda_hash_params.h"
#include "hash_functions.h"
#include "assert.h"
#include "icp_kernels.h"

#ifndef MINF
#define MINF __int_as_float(0xff800000)
#endif

#ifndef PINF
#define PINF __int_as_float(0x7f800000)
#endif

#ifndef INF
#define INF __int_as_float(0x7f800000)
#endif


__global__ void renderKernel(HashData hashData, RayCastData rayCastData, const struct rs2_intrinsics * dev_intrin) {
	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	const RayCastParams& rayCastParams = c_rayCastParams;

	if (x < rayCastParams.m_width && y < rayCastParams.m_height) {
		rayCastData.d_depth[y*rayCastParams.m_width+x] = MINF;
		rayCastData.d_depth3[y*rayCastParams.m_width+x] = make_float3(MINF,MINF,MINF,MINF);
		rayCastData.d_normals[y*rayCastParams.m_width+x] = make_float3(MINF,MINF,MINF,MINF);

        float3 point;
        // reusing librealsense cuda kernel to calc point cloud
        deproject_pixel_to_point_cuda(&point, dev_intrin, make_float2(float)x, (float)y), 1.0f * (DEPTH_WORLD_MAX - DEPTH_WORLD_MIN) + DEPTH_WORLD_MIN);

		float3 camDir = normalize(point);
		float3 worldCamPos = rayCastParams.m_viewMatrixInverse * make_float3(0.0f, 0.0f, 0.0f);
		float4 w = rayCastParams.m_viewMatrixInverse * make_float4(camDir, 0.0f);
		float3 worldDir = normalize(make_float3(w.x, w.y, w.z));

		//don't use ray interval splatting
		float minInterval = rayCastParams.m_minDepth;
		float maxInterval = rayCastParams.m_maxDepth;

		// shouldn't this return in the case no interval is found?
		if (minInterval == 0 || minInterval == MINF) return;
		if (maxInterval == 0 || maxInterval == MINF) return;

        rayCastData.traverseCoarseGridSimpleSampleAll(hashData, worldCamPos, worldDir, camDir, make_int3(x,y,1), 
                    minInterval, maxInterval, dev_intrin);
	} 
}

void CudaPipeline::render() {
	// rayIntervalSplatting(hashData, hashParams, cameraData, lastRigidTransform);
	if (hashParams.m_numOccupiedBlocks == 0)	return;

    // 6 verts (2 triangles) per block
    assert(m_rayCastParams.m_maxNumVertices > 6*hashParams.m_numOccupiedBlocks);

	m_rayCastParams.m_numOccupiedSDFBlocks = hashParams.m_numOccupiedBlocks;
	m_rayCastParams.m_viewMatrix = lastRigidTransform.getInverse();
	m_rayCastParams.m_viewMatrixInverse = lastRigidTransform;

	m_data.updateParams(m_rayCastParams); // !!! debugging
    // -----

	// renderCS(hashData, m_data, cameraData, m_rayCastParams);
	const dim3 gridSize((rayCastParams.m_width + T_PER_BLOCK - 1)/T_PER_BLOCK, (rayCastParams.m_height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);


	renderKernel<<<gridSize, blockSize>>>(hashData, m_rayCastData, dev_intrin);
    getLastCudaError("Failed: renderKernel");
    computeNormalsDevice<<<gridSize, blockSize>>>(m_rayCastData.d_normals, m_rayCastData.dev_depth3, dev_intrin->width, dev_intrin->height);
    getLastCudaError("Failed: computeNormalsDevice");
}