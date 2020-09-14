#ifndef CUDA_RAY_CAST_PARAMS
#define CUDA_RAY_CAST_PARAMS

#include "cuda_simple_matrix_math.h"

extern __constant__ RayCastParams c_rayCastParams;
void updateConstantRayCastParams(const RayCastParams& params);

//has to be aligned to 16 bytes
struct __align__(16) RayCastParams {
	float4x4 m_viewMatrix;
	float4x4 m_viewMatrixInverse;
	float4x4 m_intrinsics;
	float4x4 m_intrinsicsInverse;

	unsigned int m_width;
	unsigned int m_height;

	unsigned int m_numOccupiedSDFBlocks;
	unsigned int m_maxNumVertices;
	int m_splatMinimum;

	float m_minDepth;
	float m_maxDepth;
	float m_rayIncrement;
	float m_thresSampleDist;
	float m_thresDist;
	bool  m_useGradients;

	uint dummy0;
};

struct RayCastSample
{
	float sdf;
	float alpha;
	uint weight;
	//uint3 color;
};

struct RayCastData {

	///////////////
	// Host part //
	///////////////

	float*  d_depth;
	float3* d_depth3;
	float3* d_normals;

	float3* d_vertexBuffer; // ray interval splatting triangles, mapped from directx (memory lives there)

	cudaArray* d_rayIntervalSplatMinArray;
	cudaArray* d_rayIntervalSplatMaxArray;

	__device__ __host__
	RayCastData() {
		d_depth = NULL;
		d_depth3 = NULL;
		d_normals = NULL;

		d_vertexBuffer = NULL;

		d_rayIntervalSplatMinArray = NULL;
		d_rayIntervalSplatMaxArray = NULL;
	}

#ifndef __CUDACC__
	__host__
	void allocate(const RayCastParams& params) {
		checkCudaErrors(cudaMalloc(&d_depth, sizeof(float) * params.m_width * params.m_height));
		checkCudaErrors(cudaMalloc(&d_depth3, sizeof(float3) * params.m_width * params.m_height));
		checkCudaErrors(cudaMalloc(&d_normals, sizeof(float3) * params.m_width * params.m_height));
	}

	__host__
	void updateParams(const RayCastParams& params) {
		updateConstantRayCastParams(params);
	}

	__host__
    void free() {
			checkCudaErrors(cudaFree(d_depth));
			checkCudaErrors(cudaFree(d_depth3));
			checkCudaErrors(cudaFree(d_normals));
	}
#endif

	/////////////////
	// Device part //
	/////////////////
#ifdef __CUDACC__

	__device__
		const RayCastParams& params() const {
			return c_rayCastParams;
	}

	__device__
	float frac(float val) const {
		return (val - floorf(val));
	}
	__device__
	float3 frac(const float3& val) const {
			return make_float3(frac(val.x), frac(val.y), frac(val.z));
	}
	
	__device__
	bool trilinearInterpolationSimpleFastFast(const HashData& hash, const float3& pos, float& dist) const {
		const float oSet = c_hashParams.m_virtualVoxelSize;
		const float3 posDual = pos-make_float3(oSet/2.0f, oSet/2.0f, oSet/2.0f);
		float3 weight = frac(hash.worldToVirtualVoxelPosFloat(pos));

		dist = 0.0f;
		
		return true;
	}


	__device__
	float findIntersectionLinear(float tNear, float tFar, float dNear, float dFar) const
	{
		return tNear+(dNear/(dNear-dFar))*(tFar-tNear);
	}
	
	static const unsigned int nIterationsBisection = 3;
	
	// d0 near, d1 far
	__device__
    bool findIntersectionBisection(const HashData& hash, const float3& worldCamPos, const float3& worldDir, float d0, float r0, float d1, float r1, float& alpha) const
	{
		float a = r0; float aDist = d0;
		float b = r1; float bDist = d1;
		float c = 0.0f;

#pragma unroll 1
		for(uint i = 0; i<nIterationsBisection; i++)
		{
			c = findIntersectionLinear(a, b, aDist, bDist);

			float cDist;
			if(!trilinearInterpolationSimpleFastFast(hash, worldCamPos+c*worldDir, cDist)) return false;

			if(aDist*cDist > 0.0) { a = c; aDist = cDist; }
			else { b = c; bDist = cDist; }
		}

		alpha = c;

		return true;
	}
	
	
	__device__
	float3 gradientForPoint(const HashData& hash, const float3& pos) const
	{
		const float voxelSize = c_hashParams.m_virtualVoxelSize;
		float3 offset = make_float3(voxelSize, voxelSize, voxelSize);

		float distp00; trilinearInterpolationSimpleFastFast(hash, pos-make_float3(0.5f*offset.x, 0.0f, 0.0f), distp00);
		float dist0p0; trilinearInterpolationSimpleFastFast(hash, pos-make_float3(0.0f, 0.5f*offset.y, 0.0f), dist0p0);
		float dist00p; trilinearInterpolationSimpleFastFast(hash, pos-make_float3(0.0f, 0.0f, 0.5f*offset.z), dist00p);

		float dist100; trilinearInterpolationSimpleFastFast(hash, pos+make_float3(0.5f*offset.x, 0.0f, 0.0f), dist100);
		float dist010; trilinearInterpolationSimpleFastFast(hash, pos+make_float3(0.0f, 0.5f*offset.y, 0.0f), dist010);
		float dist001; trilinearInterpolationSimpleFastFast(hash, pos+make_float3(0.0f, 0.0f, 0.5f*offset.z), dist001);

		float3 grad = make_float3((distp00-dist100)/offset.x, (dist0p0-dist010)/offset.y, (dist00p-dist001)/offset.z);

		float l = length(grad);
		if(l == 0.0f) {
			return make_float3(0.0f, 0.0f, 0.0f);
		}

		return -grad/l;
	}

	__device__
	void traverseCoarseGridSimpleSampleAll(const HashData& hash, const float3& worldCamPos, const float3& worldDir, 
            const float3& camDir, const int3& dTid, float minInterval, float maxInterval, 
            const struct rs2_intrinsics * dev_intrin) const
	{
		const RayCastParams& rayCastParams = c_rayCastParams;

		// Last Sample
		RayCastSample lastSample; lastSample.sdf = 0.0f; lastSample.alpha = 0.0f; lastSample.weight = 0; // lastSample.color = int3(0, 0, 0);
		const float depthToRayLength = 1.0f/camDir.z; // scale factor to convert from depth to ray length
		
		float rayCurrent = depthToRayLength * max(rayCastParams.m_minDepth, minInterval);	// Convert depth to raylength
		float rayEnd = depthToRayLength * min(rayCastParams.m_maxDepth, maxInterval);		// Convert depth to raylength
		//float rayCurrent = depthToRayLength * rayCastParams.m_minDepth;	// Convert depth to raylength
		//float rayEnd = depthToRayLength * rayCastParams.m_maxDepth;		// Convert depth to raylength

#pragma unroll 1
		while(rayCurrent < rayEnd)
		{
			float3 currentPosWorld = worldCamPos+rayCurrent*worldDir;
			float dist;

			if(trilinearInterpolationSimpleFastFast(hash, currentPosWorld, dist))
			{
				if(lastSample.weight > 0 && lastSample.sdf > 0.0f && dist < 0.0f) // current sample is always valid here 
				{

					float alpha; // = findIntersectionLinear(lastSample.alpha, rayCurrent, lastSample.sdf, dist);
					bool b = findIntersectionBisection(hash, worldCamPos, worldDir, lastSample.sdf, lastSample.alpha, dist, rayCurrent, alpha);
					
					float3 currentIso = worldCamPos+alpha*worldDir;
					if(b && abs(lastSample.sdf - dist) < rayCastParams.m_thresSampleDist)
					{
						if(abs(dist) < rayCastParams.m_thresDist)
						{
							float depth = alpha / depthToRayLength; // Convert ray length to depth depthToRayLength

							d_depth[dTid.y*rayCastParams.m_width+dTid.x] = depth;
                            float3 point;
                            // reusing librealsense cuda kernel to calc point cloud
                            deproject_pixel_to_point_cuda(&point, dev_intrin, make_float2(dTid.x,dTid.y), depth);
							d_depth3[dTid.y*rayCastParams.m_width+dTid.x] = make_float3(point, 1.0f);

							return;
						}
					}
				}

				lastSample.sdf = dist;
				lastSample.alpha = rayCurrent;
				lastSample.weight = 1;
				rayCurrent += rayCastParams.m_rayIncrement;
			} else {
				lastSample.weight = 0;
				rayCurrent += rayCastParams.m_rayIncrement;
			}

			
		}
		
	}

#endif // __CUDACC__

};

#endif
