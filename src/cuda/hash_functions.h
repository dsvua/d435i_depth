#include "cuda_hash_params.h"
#include <cuda_runtime.h>
#include "cuda_declarations.h"
#include "helper_cuda.h"
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API


#ifndef sint
typedef signed int sint;
#endif

#ifndef uint
typedef unsigned int uint;
#endif 

#ifndef slong 
typedef signed long slong;
#endif

#ifndef ulong
typedef unsigned long ulong;
#endif

#ifndef uchar
typedef unsigned char uchar;
#endif

#ifndef schar
typedef signed char schar;
#endif

// #ifndef MINF
// #define MINF __int_as_float(0xff800000)
// #endif

// #ifndef PINF
// #define PINF __int_as_float(0x7f800000)
// #endif

// #ifndef INF
// #define INF __int_as_float(0x7f800000)
// #endif

#ifndef HASH_FUNCTIONS
#define HASH_FUNCTIONS

//status flags for hash entries
static const int LOCK_ENTRY = -1;
static const int FREE_ENTRY = -2;
static const int NO_OFFSET = 0;
 

struct __align__(16) HashEntry 
{
	int3	pos;		//hash position (lower left corner of SDFBlock))
	int		ptr;		//pointer into heap to SDFBlock
	uint	offset;		//offset for collisions

	
	__device__ void operator=(const struct HashEntry& e) {
		((long long*)this)[0] = ((const long long*)&e)[0];
		((long long*)this)[1] = ((const long long*)&e)[1];
		((int*)this)[4] = ((const int*)&e)[4];
	}
};


struct __align__(8) Voxel {
	float	sdf;		//signed distance function
	uchar3	color;		//color 
	uchar	weight;		//accumulated sdf weight

	__device__ void operator=(const struct Voxel& v) {
		((long long*)this)[0] = ((const long long*)&v)[0];
	}

};
extern  __constant__ HashParams c_hashParams;
void updateConstantHashParams(const HashParams& hashParams);

struct HashData {
	uint*		d_heap;						//heap that manages free memory
	uint*		d_heapCounter;				//single element; used as an atomic counter (points to the next free block)
	int*		d_hashDecision;				//
	int*		d_hashDecisionPrefix;		//
	HashEntry*	d_hash;						//hash that stores pointers to sdf blocks
	HashEntry*	d_hashCompactified;			//same as before except that only valid pointers are there
	int*		d_hashCompactifiedCounter;	//atomic counter to add compactified entries atomically 
	Voxel*		d_SDFBlocks;				//sub-blocks that contain 8x8x8 voxels (linearized); are allocated by heap
	int*		d_hashBucketMutex;			//binary flag per hash bucket; used for allocation to atomically lock a bucket
	bool		m_bIsOnGPU;					//the class be be used on both cpu and gpu

	__device__ __host__
	HashData() {
		d_heap = NULL;
		d_heapCounter = NULL;
		d_hash = NULL;
		d_hashDecision = NULL;
		d_hashDecisionPrefix = NULL;
		d_hashCompactified = NULL;
		d_hashCompactifiedCounter = NULL;
		d_SDFBlocks = NULL;
		d_hashBucketMutex = NULL;
		m_bIsOnGPU = false;
	}

    __host__
	void allocate(const HashParams& params, bool dataOnGPU = true) {
        m_bIsOnGPU = dataOnGPU;
        if (m_bIsOnGPU) {
            checkCudaErrors(cudaMalloc(&d_heap, sizeof(unsigned int) * params.m_numSDFBlocks));
            checkCudaErrors(cudaMalloc(&d_heapCounter, sizeof(unsigned int)));
            checkCudaErrors(cudaMalloc(&d_hash, sizeof(HashEntry)* params.m_hashNumBuckets * params.m_hashBucketSize));
            checkCudaErrors(cudaMalloc(&d_hashDecision, sizeof(int)* params.m_hashNumBuckets * params.m_hashBucketSize));
            checkCudaErrors(cudaMalloc(&d_hashDecisionPrefix, sizeof(int)* params.m_hashNumBuckets * params.m_hashBucketSize));
            checkCudaErrors(cudaMalloc(&d_hashCompactified, sizeof(HashEntry)* params.m_hashNumBuckets * params.m_hashBucketSize));
            checkCudaErrors(cudaMalloc(&d_hashCompactifiedCounter, sizeof(int)));
            checkCudaErrors(cudaMalloc(&d_SDFBlocks, sizeof(Voxel) * params.m_numSDFBlocks * params.m_SDFBlockSize*params.m_SDFBlockSize*params.m_SDFBlockSize));
            checkCudaErrors(cudaMalloc(&d_hashBucketMutex, sizeof(int)* params.m_hashNumBuckets));
        } else {
            d_heap = new unsigned int[params.m_numSDFBlocks];
            d_heapCounter = new unsigned int[1];
            d_hash = new HashEntry[params.m_hashNumBuckets * params.m_hashBucketSize];
            d_hashDecision = new int[params.m_hashNumBuckets * params.m_hashBucketSize];
            d_hashDecisionPrefix = new int[params.m_hashNumBuckets * params.m_hashBucketSize];
            d_hashCompactified = new HashEntry[params.m_hashNumBuckets * params.m_hashBucketSize];
            d_hashCompactifiedCounter = new int[1];
            d_SDFBlocks = new Voxel[params.m_numSDFBlocks * params.m_SDFBlockSize*params.m_SDFBlockSize*params.m_SDFBlockSize];
            d_hashBucketMutex = new int[params.m_hashNumBuckets];
        }

        updateParams(params);
    }

	__host__
	void updateParams(const HashParams& params) {
        if (m_bIsOnGPU) {
            updateConstantHashParams(params);
        } 
    }

	__host__
	void free() {
        if (m_bIsOnGPU) {
            checkCudaErrors(cudaFree(d_heap));
            checkCudaErrors(cudaFree(d_heapCounter));
            checkCudaErrors(cudaFree(d_hash));
            checkCudaErrors(cudaFree(d_hashDecision));
            checkCudaErrors(cudaFree(d_hashDecisionPrefix));
            checkCudaErrors(cudaFree(d_hashCompactified));
            checkCudaErrors(cudaFree(d_hashCompactifiedCounter));
            checkCudaErrors(cudaFree(d_SDFBlocks));
            checkCudaErrors(cudaFree(d_hashBucketMutex));
        } else {
            if (d_heap) delete[] d_heap;
            if (d_heapCounter) delete[] d_heapCounter;
            if (d_hash) delete[] d_hash;
            if (d_hashDecision) delete[] d_hashDecision;
            if (d_hashDecisionPrefix) delete[] d_hashDecisionPrefix;
            if (d_hashCompactified) delete[] d_hashCompactified;
            if (d_hashCompactifiedCounter) delete[] d_hashCompactifiedCounter;
            if (d_SDFBlocks) delete[] d_SDFBlocks;
            if (d_hashBucketMutex) delete[] d_hashBucketMutex;
        }

        d_hash = NULL;
        d_heap = NULL;
        d_heapCounter = NULL;
        d_hashDecision = NULL;
        d_hashDecisionPrefix = NULL;
        d_hashCompactified = NULL;
        d_hashCompactifiedCounter = NULL;
        d_SDFBlocks = NULL;
        d_hashBucketMutex = NULL;
    }

	__host__
	HashData copyToCPU() const {
        HashParams params;
        
        HashData hashData;
        hashData.allocate(params, false);	//allocate the data on the CPU
        checkCudaErrors(cudaMemcpy(hashData.d_heap, d_heap, sizeof(unsigned int) * params.m_numSDFBlocks, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(hashData.d_heapCounter, d_heapCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(hashData.d_hash, d_hash, sizeof(HashEntry)* params.m_hashNumBuckets * params.m_hashBucketSize, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(hashData.d_hashDecision, d_hashDecision, sizeof(int)*params.m_hashNumBuckets * params.m_hashBucketSize, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(hashData.d_hashDecisionPrefix, d_hashDecisionPrefix, sizeof(int)*params.m_hashNumBuckets * params.m_hashBucketSize, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(hashData.d_hashCompactified, d_hashCompactified, sizeof(HashEntry)* params.m_hashNumBuckets * params.m_hashBucketSize, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(hashData.d_hashCompactifiedCounter, d_hashCompactifiedCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(hashData.d_SDFBlocks, d_SDFBlocks, sizeof(Voxel) * params.m_numSDFBlocks * params.m_SDFBlockSize*params.m_SDFBlockSize*params.m_SDFBlockSize, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(hashData.d_hashBucketMutex, d_hashBucketMutex, sizeof(int)* params.m_hashNumBuckets, cudaMemcpyDeviceToHost));
        
        return hashData;
    }

	__device__
	const HashParams& params() const;

	//! see teschner et al. (but with correct prime values)
	__device__ 
	uint computeHashPos(const int3& virtualVoxelPos) const;

	//merges two voxels (v0 the currently stored voxel, v1 is the input voxel)
	__device__ 
	void combineVoxel(const Voxel &v0, const Voxel& v1, Voxel &out) const;

	//! returns the truncation of the SDF for a given distance value
	__device__ 
	float getTruncation(float z) const;

	__device__ 
	float3 worldToVirtualVoxelPosFloat(const float3& pos) const;

	__device__ 
	int3 worldToVirtualVoxelPos(const float3& pos) const;

	__device__ 
	int3 virtualVoxelPosToSDFBlock(int3 virtualVoxelPos) const;

	// Computes virtual voxel position of corner sample position
	__device__ 
	int3 SDFBlockToVirtualVoxelPos(const int3& sdfBlock) const;

	__device__ 
	float3 virtualVoxelPosToWorld(const int3& pos) const;

	__device__ 
	float3 SDFBlockToWorld(const int3& sdfBlock) const;

	__device__ 
	int3 worldToSDFBlock(const float3& worldPos) const;

	//! computes the (local) virtual voxel pos of an index; idx in [0;511]
	__device__ 
	uint3 delinearizeVoxelIndex(uint idx) const;

	//! computes the linearized index of a local virtual voxel pos; pos in [0;7]^3
	__device__ 
	uint linearizeVoxelPos(const int3& virtualVoxelPos) const;

	__device__ 
	int virtualVoxelPosToLocalSDFBlockIndex(const int3& virtualVoxelPos) const;

	__device__ 
	int worldToLocalSDFBlockIndex(const float3& world) const;

		//! returns the hash entry for a given worldPos; if there was no hash entry the returned entry will have a ptr with FREE_ENTRY set
	__device__ 
	HashEntry getHashEntry(const float3& worldPos) const;


	__device__ 
    void deleteHashEntry(uint id);

	__device__ 
    void deleteHashEntry(HashEntry& hashEntry);

	__device__ 
    bool voxelExists(const float3& worldPos) const;

	__device__  
	void deleteVoxel(Voxel& v) const;

	__device__ 
    void deleteVoxel(uint id);

	__device__ 
	Voxel getVoxel(const float3& worldPos) const;

	__device__ 
	Voxel getVoxel(const int3& virtualVoxelPos) const;
	
	__device__ 
	void setVoxel(const int3& virtualVoxelPos, Voxel& voxelInput) const;

	//! returns the hash entry for a given sdf block id; if there was no hash entry the returned entry will have a ptr with FREE_ENTRY set
	__device__ 
	HashEntry getHashEntryForSDFBlockPos(const int3& sdfBlock) const;

	//for histogram (no collision traversal)
	__device__ 
	unsigned int getNumHashEntriesPerBucket(unsigned int bucketID);

	//for histogram (collisions traversal only)
	__device__ 
	unsigned int getNumHashLinkedList(unsigned int bucketID);

	__device__
	uint consumeHeap();

	__device__
	void appendHeap(uint ptr);

	//pos in SDF block coordinates
	__device__
	void allocBlock(const int3& pos);

	//!inserts a hash entry without allocating any memory: used by streaming: TODO MATTHIAS check the atomics in this function
	__device__
	bool insertHashEntry(HashEntry entry);

	//! deletes a hash entry position for a given sdfBlock index (returns true uppon successful deletion; otherwise returns false)
	__device__
	bool deleteHashEntryElement(const int3& sdfBlock);

	__device__
	bool isSDFBlockInCameraFrustumApprox(const int3& sdfBlock, const struct rs2_intrinsics * dev_intrin);

};

#endif