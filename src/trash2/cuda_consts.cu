#include "cuda_hash_params.h"
#include "helper_cuda.h"
#include "cuda_raycast_params.h"

__constant__ HashParams c_hashParams;
__constant__ RayCastParams c_rayCastParams;
// __constant__ DepthCameraParams c_depthCameraParams;

void updateConstantHashParams(const HashParams& params) {

	size_t size;
	checkCudaErrors(cudaGetSymbolSize(&size, c_hashParams));
	checkCudaErrors(cudaMemcpyToSymbol(c_hashParams, &params, size, 0, cudaMemcpyHostToDevice));
	
}

void updateConstantRayCastParams(const RayCastParams& params) {
	
	size_t size;
	checkCudaErrors(cudaGetSymbolSize(&size, c_rayCastParams));
	checkCudaErrors(cudaMemcpyToSymbol(c_rayCastParams, &params, size, 0, cudaMemcpyHostToDevice));
	
}