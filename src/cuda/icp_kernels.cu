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

#ifndef T_PER_BLOCK
#define T_PER_BLOCK 16
#endif

#ifndef CUDART_PI_F
#define CUDART_PI_F 3.141592654f
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif

#ifndef ARRAY_SIZE
#define ARRAY_SIZE 30
#endif

__global__ void computeNormalsDevice(float3* d_output, float3* d_input, unsigned int width, unsigned int height) {
	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= width || y >= height) return;

	d_output[y*width+x] = make_float3(MINF, MINF, MINF);

	if(x > 0 && x < width-1 && y > 0 && y < height-1)
	{
		const float3 CC = d_input[(y+0)*width+(x+0)];
		const float3 PC = d_input[(y+1)*width+(x+0)];
		const float3 CP = d_input[(y+0)*width+(x+1)];
		const float3 MC = d_input[(y-1)*width+(x+0)];
		const float3 CM = d_input[(y+0)*width+(x-1)];

		if(CC.x != MINF && PC.x != MINF && CP.x != MINF && MC.x != MINF && CM.x != MINF)
		{
			const float3 n = cross(PC-MC, CP-CM);
			const float  l = length(n);

			if(l > 0.0f)
			{
				d_output[y*width+x] = n/-l;
			}
		}
	}
}

inline __device__ float3 bilinearInterpolationFloat3(float x, float y, float3* d_input, unsigned int imageWidth,
                unsigned int imageHeight) {
	const int2 p00 = make_int2(floor(x), floor(y));
	const int2 p01 = p00 + make_int2(0.0f, 1.0f);
	const int2 p10 = p00 + make_int2(1.0f, 0.0f);
	const int2 p11 = p00 + make_int2(1.0f, 1.0f);

	const float alpha = x - p00.x;
	const float beta  = y - p00.y;

	//const float INVALID = 0.0f;
	const float INVALID = MINF;

	float4 s0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f); float w0 = 0.0f;
	if(p00.x < imageWidth && p00.y < imageHeight) { float4 v00 = d_input[p00.y*imageWidth + p00.x]; if(v00.x != INVALID && v00.y != INVALID && v00.z != INVALID) { s0 += (1.0f-alpha)*v00; w0 += (1.0f-alpha); } }
	if(p10.x < imageWidth && p10.y < imageHeight) { float4 v10 = d_input[p10.y*imageWidth + p10.x]; if(v10.x != INVALID && v10.y != INVALID && v10.z != INVALID) { s0 +=		alpha *v10; w0 +=		alpha ; } }

	float4 s1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f); float w1 = 0.0f;
	if(p01.x < imageWidth && p01.y < imageHeight) { float4 v01 = d_input[p01.y*imageWidth + p01.x]; if(v01.x != INVALID && v01.y != INVALID && v01.z != INVALID) { s1 += (1.0f-alpha)*v01; w1 += (1.0f-alpha);} }
	if(p11.x < imageWidth && p11.y < imageHeight) { float4 v11 = d_input[p11.y*imageWidth + p11.x]; if(v11.x != INVALID && v11.y != INVALID && v11.z != INVALID) { s1 +=		alpha *v11; w1 +=		alpha ;} }

	const float4 p0 = s0/w0;
	const float4 p1 = s1/w1;

	float4 ss = make_float4(0.0f, 0.0f, 0.0f, 0.0f); float ww = 0.0f;
	if(w0 > 0.0f) { ss += (1.0f-beta)*p0; ww += (1.0f-beta); }
	if(w1 > 0.0f) { ss +=		beta *p1; ww +=		  beta ; }

	if(ww > 0.0f) return ss/ww;
	else		  return make_float4(MINF, MINF, MINF, MINF);
}

__global__ void resampleFloat3MapDevice(float3* d_output, float3* d_input,
        unsigned int inputWidth, unsigned int inputHeight, unsigned int outputWidth, unsigned int outputHeight) {
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x < outputWidth && y < outputHeight)
	{
		const float scaleWidth  = (float)(inputWidth-1) /(float)(outputWidth-1);
		const float scaleHeight = (float)(inputHeight-1)/(float)(outputHeight-1);

		const unsigned int xInput = (unsigned int)(x*scaleWidth +0.5f);
		const unsigned int yInput = (unsigned int)(y*scaleHeight+0.5f);

		if(xInput < inputWidth && yInput < inputHeight)
		{
            d_output[y*outputWidth+x] = bilinearInterpolationFloat3(x*scaleWidth, y*scaleHeight, d_input, inputWidth,
                                                                        inputHeight);
		}
	}
}

__device__ inline bool isValid(float4 p)
{
	return (p.x != MINF);
}

__device__ inline bool isValidCol(float4 p)
{
	return p.w != 0.0f;
}

__device__ inline void getBestCorrespondence1x1(
	uint2 screenPos, float4 pInput, float4 nInput, float4 cInput, float4& pTarget, float4& nTarget,
	float4* d_Input, float4* d_InputNormals, float4* d_InputColors, 
	float4* d_Target, float4* d_TargetNormals, float4* d_TargetColors, 
	float4* d_Output, float4* d_OutputNormals, unsigned int width, unsigned int height
	)
{
	const unsigned int idx = screenPos.x + screenPos.y*width;
	pTarget = d_Target[idx];
	nTarget = d_TargetNormals[idx];
}

__device__ inline void getBestCorrespondence1x1(
	uint2 screenPos, float4& pTarget, float4& nTarget,
	float4* d_Target, float4* d_TargetNormals, unsigned int width, unsigned int height,
	unsigned int& idx
	)
{
	idx = screenPos.x + screenPos.y*width;
	pTarget = d_Target[idx];
	nTarget = d_TargetNormals[idx];
}


__global__ void projectiveCorrespondencesKernel(	
	float4* d_Input, float4* d_InputNormals, float4* d_InputColors, 
	float4* d_Target, float4* d_TargetNormals, float4* d_TargetColors, 
	float4* d_Output, float4* d_OutputNormals, unsigned int width, unsigned int height,
	float distThres, float normalThres, float levelFactor,
	float4x4 transform, float4x4 intrinsic, const struct rs2_intrinsics * dev_intrin)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= width || y >= height) return;
	d_Output[y*width+x] = make_float4(MINF, MINF, MINF, MINF);
	d_OutputNormals[y*width+x] = make_float4(MINF, MINF, MINF, MINF);

	float4 pInput = d_Input[y*width+x];
	float4 nInput = d_InputNormals[y*width+x];
	float4 cInput = make_float4(0.0f, 0.0f, 0.0f, 0.0f); // = d_InputColors[y*width+x];

	if(isValid(pInput) && isValid(nInput))
	{
		pInput.w = 1.0f; // assert it is a point
		//float4 pTransInput = mul(pInput, transform);
		float4 pTransInput = transform * pInput;

		nInput.w = 0.0f;  // assert it is a vector
		//float4 nTransInput = mul(nInput, transform); // transformation is a rotation M^(-1)^T = M, translation is ignored because it is a vector
		float4 nTransInput = transform * nInput;

		//if(pTransInput.z > FLT_EPSILON) // really necessary
		{
            //int2 screenPos = cameraToKinectScreenInt(make_float3(pTransInput), intrinsic);
            float3 pos = make_float3(pTransInput)
            float2 pImage = make_float2(
                pos.x*d_intrin.fx/pos.z + d_intrin.ppx,			
                pos.y*d_intrin.fy/pos.z + d_intrin.ppy);            
			int2 screenPos = make_int2(pImage + make_float2(0.5f, 0.5f));
			screenPos = make_int2(screenPos.x/levelFactor, screenPos.y/levelFactor);

			if (screenPos.x >= 0 && screenPos.y >= 0 && screenPos.x < width && screenPos.y < height) {
				float4 pTarget, nTarget;
				getBestCorrespondence1x1(make_uint2(screenPos), pTransInput, nTransInput, cInput, pTarget, nTarget,
                    d_Input, d_InputNormals, d_InputColors, d_Target, d_TargetNormals, d_TargetColors, d_Output, 
                    d_OutputNormals, width, height);
				if (isValid(pTarget) && isValid(nTarget)) {
					float d = length(make_float3(pTransInput)-make_float3(pTarget));
					float dNormal = dot(make_float3(nTransInput), make_float3(nTarget));

					if (d <= distThres && dNormal >= normalThres)
					{
						d_Output[y*width+x] = pTarget;

						//nTarget.w = max(0.0, 0.5f*((1.0f-d/distThres)+(1.0f-cameraToKinectProjZ(pTransInput.z)))); // for weighted ICP;
						nTarget.w = max(0.0, 0.5f*((1.0f-d/distThres)+(1.0f-(pTransInput.z - DEPTH_WORLD_MIN)/(DEPTH_WORLD_MAX - DEPTH_WORLD_MIN)))); // for weighted ICP;
						
						d_OutputNormals[y*width+x] = nTarget;
					}
				}
			}
		}
	}
}

__device__
static inline float2 cameraToKinectScreenFloat(const float3& pos)	{
    //return make_float2(pos.x*c_depthCameraParams.fx/pos.z + c_depthCameraParams.mx, c_depthCameraParams.my - pos.y*c_depthCameraParams.fy/pos.z);
    return make_float2(
        pos.x*d_intrin.fx/pos.z + d_intrin.ppx,			
        pos.y*d_intrin.fy/pos.z + d_intrin.ppy);
}

__device__
static inline int2 cameraToKinectScreenInt(const float3& pos)	{
    float2 pImage = make_float2(
        pos.x*d_intrin.fx/pos.z + d_intrin.ppx,			
        pos.y*d_intrin.fy/pos.z + d_intrin.ppy);
    make_int2(pImage + make_float2(0.5f, 0.5f));
}




/////////////////////////////////////////////////////
// Shared Memory
/////////////////////////////////////////////////////

__shared__ float bucket2[ARRAY_SIZE*BLOCK_SIZE];

/////////////////////////////////////////////////////
// Helper Functions
/////////////////////////////////////////////////////

__device__ inline void addToLocalScanElement(uint inpGTid, uint resGTid, volatile float* shared)
{
	#pragma unroll
	for (uint i = 0; i<ARRAY_SIZE; i++)
	{
		shared[ARRAY_SIZE*resGTid+i] += shared[ARRAY_SIZE*inpGTid+i];
	}
}

__device__ inline void CopyToResultScanElement(uint GID, float* output)
{
	#pragma unroll
	for (uint i = 0; i<ARRAY_SIZE; i++)
	{
		output[ARRAY_SIZE*GID+i] = bucket2[0+i];
	}
}

__device__ inline void SetZeroScanElement(uint GTid)
{
	#pragma unroll
	for(uint i = 0; i<ARRAY_SIZE; i++)
	{
		bucket2[GTid*ARRAY_SIZE+i] = 0.0f;
	}
}

/////////////////////////////////////////////////////
// Linearized System Matrix
/////////////////////////////////////////////////////

// Matrix Struct
struct Float1x6
{
	float data[6];
};

// Arguments: q moving point, n normal target
__device__ inline  Float1x6 buildRowSystemMatrixPlane(float3 q, float3 n, float w)
{
	Float1x6 row;
	row.data[0] = n.x*q.y-n.y*q.x;
	row.data[1] = n.z*q.x-n.x*q.z;
	row.data[2] = n.y*q.z-n.z*q.y;

	row.data[3] = -n.x;
	row.data[4] = -n.y;
	row.data[5] = -n.z;

	return row;
}

// Arguments: p target point, q moving point, n normal target
__device__ inline  float buildRowRHSPlane(float3 p, float3 q, float3 n, float w)
{
	return n.x*(q.x-p.x)+n.y*(q.y-p.y)+n.z*(q.z-p.z);
}

// Arguments: p target point, q moving point, n normal target
__device__ inline  void addToLocalSystem(float3 p, float3 q, float3 n, float weight, uint GTid)
{
	const Float1x6 row = buildRowSystemMatrixPlane(q, n, weight);
	const float b = buildRowRHSPlane(p, q, n, weight);
	uint linRowStart = 0;

	#pragma unroll
	for (uint i = 0; i<6; i++) {
		#pragma unroll
		for (uint j = i; j<6; j++) {
			bucket2[ARRAY_SIZE*GTid+linRowStart+j-i] += weight*row.data[i]*row.data[j];
		}

		linRowStart += 6-i;

		bucket2[ARRAY_SIZE*GTid+21+i] += weight*row.data[i]*b;
	}

	const float dN = dot(p-q, n);
	bucket2[ARRAY_SIZE*GTid+27] += weight*dN*dN;		//residual
	bucket2[ARRAY_SIZE*GTid+28] += weight;			//corr weight
	bucket2[ARRAY_SIZE*GTid+29] += 1.0f;				//corr number
}

/////////////////////////////////////////////////////
// Scan
/////////////////////////////////////////////////////

__device__ inline void warpReduce(int GTid) // See Optimizing Parallel Reduction in CUDA by Mark Harris
{
	addToLocalScanElement(GTid + 32, GTid, bucket2);
	addToLocalScanElement(GTid + 16, GTid, bucket2);
	addToLocalScanElement(GTid + 8 , GTid, bucket2);
	addToLocalScanElement(GTid + 4 , GTid, bucket2);
	addToLocalScanElement(GTid + 2 , GTid, bucket2);
	addToLocalScanElement(GTid + 1 , GTid, bucket2);
}

__global__ void scanScanElementsCS(
	unsigned int imageWidth,
	unsigned int imageHeight,
	float* output,
	float4* input,
	float4* target,
	float4* targetNormals,
	float4x4 deltaTransform, unsigned int localWindowSize)
{
	const uint x = blockIdx.x * blockDim.x + threadIdx.x;

	// Set system to zero
	SetZeroScanElement(threadIdx.x);

	//Locally sum small window
	#pragma unroll
	for (uint i = 0; i<localWindowSize; i++)
	{
		const int index1D = localWindowSize*x+i;
		const uint2 index = make_uint2(index1D%imageWidth, index1D/imageWidth);

		if (index.x < imageWidth && index.y < imageHeight)
		{
			if (target[index1D].x != MINF && input[index1D].x != MINF && targetNormals[index1D].x != MINF) {
				const float g_meanStDevInv = 1.0f;
				const float3 g_mean = make_float3(0.0f,0.0f,0.0f);

				const float3 inputT = g_meanStDevInv*((deltaTransform.getTranspose()*make_float3(input[index1D])) - g_mean);
				const float3 targetT = g_meanStDevInv*(make_float3(target[index1D])-g_mean);
				const float weight = targetNormals[index1D].w;

				// Compute Linearized System
				addToLocalSystem(targetT, inputT, make_float3(targetNormals[index1D]), weight, threadIdx.x);
			}
		}
	}

	__syncthreads();

	// Up sweep 2D
	#pragma unroll
	for(unsigned int stride = BLOCK_SIZE/2; stride > 32; stride >>= 1)
	{
		if (threadIdx.x < stride) addToLocalScanElement(threadIdx.x+stride/2, threadIdx.x, bucket2);

		__syncthreads();
	}

	if (threadIdx.x < 32) warpReduce(threadIdx.x);

	// Copy to output texture
	if (threadIdx.x == 0) CopyToResultScanElement(blockIdx.x, output);
}

extern "C" void buildLinearSystem(
	unsigned int imageWidth,
	unsigned int imageHeight,
	float* output,
	float4* input,
	float4* target,
	float4* targetNormals,
	float* deltaTransform, unsigned int localWindowSize, unsigned int blockSizeInt)
{
	const unsigned int numElements = imageWidth*imageHeight;

	dim3 blockSize(blockSizeInt, 1, 1);
	dim3 gridSize((numElements + blockSizeInt*localWindowSize-1) / (blockSizeInt*localWindowSize), 1, 1);

	scanScanElementsCS<<<gridSize, blockSize>>>(imageWidth, imageHeight, output, input, target, targetNormals, float4x4(deltaTransform), localWindowSize);

	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}
