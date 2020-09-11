#include "cuda_hash_params.h"
#include <cuda_runtime.h>


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
// #define MINF __int_as_float((int)0xff800000)
// #endif

// #ifndef PINF
// #define PINF __int_as_float((int)0x7f800000)
// #endif

// #ifndef INF
// #define INF __int_as_float((int)0x7f800000)
// #endif

#ifndef HASH_FUNCTIONS
#define HASH_FUNCTIONS

#ifndef SDF_BLOCK_SIZE
#define SDF_BLOCK_SIZE 8
#endif

#ifndef HASH_BUCKET_SIZE
#define HASH_BUCKET_SIZE 10
#endif

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
	void allocate(const HashParams& params, bool dataOnGPU = true);

	__host__
	void updateParams(const HashParams& params);

	__host__
	void free();

	__host__
	HashData copyToCPU() const;

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

};

#endif