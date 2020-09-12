// #include "CudaPipeline.h"
#include "helper_math.h"
#include "helper_functions.h"
#include "helper_cuda.h"
#include "cutil_math.h"
#include "deproject_point.h"

#include "integrate_kernels.h"

#ifndef MINF
#define MINF __int_as_float(0xff800000)
#endif

#ifndef PINF
#define PINF __int_as_float(0x7f800000)
#endif

#ifndef INF
#define INF __int_as_float(0x7f800000)
#endif

__shared__ float shared_MinSDF[SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE / 2];
__shared__ uint shared_MaxWeight[SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE / 2];


// static float4x4 toCUDA(const float4x4& m) {
//     return float4x4(m.ptr());
// }

// static float4x4 toCUDA(const Eigen::Matrix4f& mat) {
//     return float4x4(mat.data()).getTranspose();
// }

__global__ void resetHashKernel(HashData hashData) 
{
	const HashParams& hashParams = c_hashParams;
	const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < hashParams.m_hashNumBuckets * HASH_BUCKET_SIZE) {
		hashData.deleteHashEntry(hashData.d_hash[idx]);
		hashData.deleteHashEntry(hashData.d_hashCompactified[idx]);
	}
}

__global__ void allocKernel(HashData hashData, const uint16_t * depth, const struct rs2_intrinsics * dev_intrin) {
	const HashParams& hashParams = c_hashParams;

	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	if (x < dev_intrin->width && y < dev_intrin->height) {

		float d = (float)depth[y*dev_intrin->width + x] / 1000; // convert mm to meter
		
		if (d < hashParams.m_minIntegrationDistance)	return;

		if (d >= hashParams.m_maxIntegrationDistance) return;

		float t = hashData.getTruncation(d);
		float minDepth = min(hashParams.m_maxIntegrationDistance, d-t);
		float maxDepth = min(hashParams.m_maxIntegrationDistance, d+t);
		if (minDepth >= maxDepth) return;

        const float pixel[] = { (float)x, (float)y };
        float point[3];

        // reusing librealsense cuda kernel to calc point cloud
        deproject_pixel_to_point_cuda(point, dev_intrin, pixel, minDepth);
		float3 rayMin = make_float3((float)point[0], (float)point[1], (float)point[2]);
		rayMin = hashParams.m_rigidTransform * rayMin;
        deproject_pixel_to_point_cuda(point, dev_intrin, pixel, minDepth);
		float3 rayMax = make_float3((float)point[0], (float)point[1], (float)point[2]);
		rayMax = hashParams.m_rigidTransform * rayMax;
		
		float3 rayDir = normalize(rayMax - rayMin);
	
		int3 idCurrentVoxel = hashData.worldToSDFBlock(rayMin);
		int3 idEnd = hashData.worldToSDFBlock(rayMax);
		
		float3 step = make_float3(sign(rayDir));
		float3 boundaryPos = hashData.SDFBlockToWorld(idCurrentVoxel+make_int3(clamp(step, 0.0, 1.0f)))-0.5f*hashParams.m_virtualVoxelSize;
		float3 tMax = (boundaryPos-rayMin)/rayDir;
		float3 tDelta = (step*SDF_BLOCK_SIZE*hashParams.m_virtualVoxelSize)/rayDir;
		int3 idBound = make_int3(make_float3(idEnd)+step);

		if (rayDir.x == 0.0f) { tMax.x = PINF; tDelta.x = PINF; }
		if (boundaryPos.x - rayMin.x == 0.0f) { tMax.x = PINF; tDelta.x = PINF; }

		if (rayDir.y == 0.0f) { tMax.y = PINF; tDelta.y = PINF; }
		if (boundaryPos.y - rayMin.y == 0.0f) { tMax.y = PINF; tDelta.y = PINF; }

		if (rayDir.z == 0.0f) { tMax.z = PINF; tDelta.z = PINF; }
		if (boundaryPos.z - rayMin.z == 0.0f) { tMax.z = PINF; tDelta.z = PINF; }


		// unsigned int iter = 0; // iter < g_MaxLoopIterCount
		unsigned int g_MaxLoopIterCount = 1024;
#pragma unroll 1
		// while(iter < g_MaxLoopIterCount) {
        for (int iter = 0; iter < g_MaxLoopIterCount; iter++) {

			//check if it's in the frustum and not checked out
			if (hashData.isSDFBlockInCameraFrustumApprox(idCurrentVoxel, dev_intrin)) {		
				hashData.allocBlock(idCurrentVoxel);
			}

			// Traverse voxel grid
			if(tMax.x < tMax.y && tMax.x < tMax.z)	{
				idCurrentVoxel.x += step.x;
				if(idCurrentVoxel.x == idBound.x) return;
				tMax.x += tDelta.x;
			} else if(tMax.z < tMax.y) {
				idCurrentVoxel.z += step.z;
				if(idCurrentVoxel.z == idBound.z) return;
				tMax.z += tDelta.z;
			} else {
				idCurrentVoxel.y += step.y;
				if(idCurrentVoxel.y == idBound.y) return;
				tMax.y += tDelta.y;
			}

			// iter++;
		}
	}
}

__global__ void garbageCollectIdentifyKernel(HashData hashData) {

	const unsigned int hashIdx = blockIdx.x;
	const HashEntry& entry = hashData.d_hashCompactified[hashIdx];
	
	const unsigned int idx0 = entry.ptr + 2*threadIdx.x+0;
	const unsigned int idx1 = entry.ptr + 2*threadIdx.x+1;

	Voxel v0 = hashData.d_SDFBlocks[idx0];
	Voxel v1 = hashData.d_SDFBlocks[idx1];

	if (v0.weight == 0)	v0.sdf = PINF;
	if (v1.weight == 0)	v1.sdf = PINF;

	shared_MinSDF[threadIdx.x] = min(fabsf(v0.sdf), fabsf(v1.sdf));	//init shared memory
	shared_MaxWeight[threadIdx.x] = max(v0.weight, v1.weight);
		
#pragma unroll 1
	for (uint stride = 2; stride <= blockDim.x; stride <<= 1) {
		__syncthreads();
		if ((threadIdx.x  & (stride-1)) == (stride-1)) {
			shared_MinSDF[threadIdx.x] = min(shared_MinSDF[threadIdx.x-stride/2], shared_MinSDF[threadIdx.x]);
			shared_MaxWeight[threadIdx.x] = max(shared_MaxWeight[threadIdx.x-stride/2], shared_MaxWeight[threadIdx.x]);
		}
	}

	__syncthreads();

	if (threadIdx.x == blockDim.x - 1) {
		float minSDF = shared_MinSDF[threadIdx.x];
		uint maxWeight = shared_MaxWeight[threadIdx.x];

		float t = hashData.getTruncation(DEPTH_WORLD_MAX);

		if (minSDF >= t || maxWeight == 0) {
			hashData.d_hashDecision[hashIdx] = 1;
		} else {
			hashData.d_hashDecision[hashIdx] = 0; 
		}
	}
}

__global__ void garbageCollectFreeKernel(HashData hashData) {

	//const uint hashIdx = blockIdx.x;
	const uint hashIdx = blockIdx.x*blockDim.x + threadIdx.x;


	if (hashIdx < c_hashParams.m_numOccupiedBlocks && hashData.d_hashDecision[hashIdx] != 0) {	//decision to delete the hash entry

		const HashEntry& entry = hashData.d_hashCompactified[hashIdx];
		//if (entry.ptr == FREE_ENTRY) return; //should never happen since we did compactify before

		if (hashData.deleteHashEntryElement(entry.pos)) {	//delete hash entry from hash (and performs heap append)
			const uint linBlockSize = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;

			#pragma unroll 1
			for (uint i = 0; i < linBlockSize; i++) {	//clear sdf block: CHECK TODO another kernel?
				hashData.deleteVoxel(entry.ptr + i);
			}
		}
	}
}

__global__ void compactifyHashAllInOneKernel(HashData hashData, const struct rs2_intrinsics * dev_intrin) {
	const HashParams& hashParams = c_hashParams;
	const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	__shared__ int localCounter;
	if (threadIdx.x == 0) localCounter = 0;
	__syncthreads();

	int addrLocal = -1;
	if (idx < hashParams.m_hashNumBuckets * HASH_BUCKET_SIZE) {
		if (hashData.d_hash[idx].ptr != FREE_ENTRY) {
			if (hashData.isSDFBlockInCameraFrustumApprox(hashData.d_hash[idx].pos, dev_intrin))
			{
				addrLocal = atomicAdd(&localCounter, 1);
			}
		}
	}

	__syncthreads();

	__shared__ int addrGlobal;
	if (threadIdx.x == 0 && localCounter > 0) {
		addrGlobal = atomicAdd(hashData.d_hashCompactifiedCounter, localCounter);
	}
	__syncthreads();

	if (addrLocal != -1) {
		const unsigned int addr = addrGlobal + addrLocal;
		hashData.d_hashCompactified[addr] = hashData.d_hash[idx];
	}
}

__global__ void integrateDepthMapKernel(HashData hashData, const uint16_t * depth, const struct rs2_intrinsics * dev_intrin) {
	const HashParams& hashParams = c_hashParams;

	const HashEntry& entry = hashData.d_hashCompactified[blockIdx.x];

	int3 pi_base = hashData.SDFBlockToVirtualVoxelPos(entry.pos);

	uint i = threadIdx.x;	//inside of an SDF block
	int3 pi = pi_base + make_int3(hashData.delinearizeVoxelIndex(i));
	float3 pf = hashData.virtualVoxelPosToWorld(pi);

    pf = hashParams.m_rigidTransformInverse * pf;
    float2 pImage = make_float2(
        pf.x*dev_intrin->fx/pf.z + dev_intrin->ppx,			
        pf.y*dev_intrin->fy/pf.z + dev_intrin->ppy);
	int2 screenPos = make_int2(pImage + make_float2(0.5f, 0.5f));;


	if (screenPos.x < dev_intrin->width && screenPos.y < dev_intrin->height) {	//on screen

		//float depth = g_InputDepth[screenPos];
		float d = (float)depth[screenPos.x + screenPos.y * dev_intrin->width];

		if (d > hashParams.m_minIntegrationDistance && d < hashParams.m_maxIntegrationDistance) { // valid depth

            float depthZeroOne = (d - DEPTH_WORLD_MIN)/(DEPTH_WORLD_MAX - DEPTH_WORLD_MIN);

            float sdf = d - pf.z;
            float truncation = hashData.getTruncation(d);
            if (sdf > -truncation) // && depthZeroOne >= 0.0f && depthZeroOne <= 1.0f) //check if in truncation range should already be made in depth map computation
            {
                if (sdf >= 0.0f) {
                    sdf = fminf(truncation, sdf);
                } else {
                    sdf = fmaxf(-truncation, sdf);
                }

                float weightUpdate = max(hashParams.m_integrationWeightSample * 1.5f * (1.0f-depthZeroOne), 1.0f);

                Voxel curr;	//construct current voxel
                curr.sdf = sdf;
                curr.weight = weightUpdate;


                uint idx = entry.ptr + i;
                
                Voxel newVoxel;
                hashData.combineVoxel(hashData.d_SDFBlocks[idx], curr, newVoxel);
                hashData.d_SDFBlocks[idx] = newVoxel;
            }
		}
	}
}

