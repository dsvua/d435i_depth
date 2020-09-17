#ifndef VOXEL_HASH_MAP

#include <cuda_runtime.h>
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API

enum HashState {
	NO_OFFSET = -1,
	FREE_ENTRY = -2
};

struct Voxel {
	float	sdf;		//signed distance function
	float	weight;		//accumulated sdf weight

};

struct VoxelHashEntry {
	int3 world_pos;
	int voxelBlockIdx;
	int offset;

    __device__ __host__
	VoxelHashEntry() {
		world_pos = {0,0,0};
		voxelBlockIdx = FREE_ENTRY;
		offset = NO_OFFSET;
	}
};


struct VoxelHashParameters {
    int   			p0;
    int   			p1;
    int   			p2;
    float       	staticPartOfRMS; // formula is here: https://dev.intelrealsense.com/docs/tuning-depth-cameras-for-best-performance
	float	    	maxIntegrationDistance; // in millimeters
	float			minIntegrationDistance; // in millimeters
	uint			voxelPhysicalSize; // size of voxel in millimeters
	float			truncationDistance;
	uint	    	integrationWeightMin;
	uint	    	integrationWeightMax;
	uint			voxelBlockSize; // voxelBlock becomes voxelBlockSize x voxelBlockSize x voxelBlockSize voxels size
	uint			voxelsTotalSize;
	uint            numVoxelBlocks;
	uint            voxelHashSize; // how many hash blocks to allocate
	uint			voxelHashBlockSize; // number of hashEntries per hash block
	uint			voxelHashTotalSize; // voxelHashSize * voxelHashBlockSize
	// cuda settings
	int				threadsPerBlock;

    __device__ __host__
	VoxelHashParameters() {
		p0 = 73856093;
		p1 = 19349669;
		p2 = 83492791;
		staticPartOfRMS = 0;
		maxIntegrationDistance = 4000;
		minIntegrationDistance = 300;
		voxelPhysicalSize = 10;
		truncationDistance = voxelPhysicalSize * 5;
		integrationWeightMin = 1;
		integrationWeightMax = 255;
		voxelBlockSize = 8;
		numVoxelBlocks = 500000;
		voxelsTotalSize = numVoxelBlocks * voxelBlockSize * voxelBlockSize * voxelBlockSize;
		voxelHashSize = 10000;
		voxelHashBlockSize = 10;
		voxelHashTotalSize = voxelHashSize * voxelHashBlockSize;
		// cuda settings
		threadsPerBlock = 8;
	}

};

// Stores voxelBlocks in voxelsHash
struct VoxelHashMap {

	VoxelHashEntry*			voxelsHash;
	VoxelHashEntry*			voxelsHashCompactified;	//same as before except that only valid pointers are there
	uint					voxelsHashCompactifiedCount;	//atomic counter to add compactified entries atomically 
	uint*					deletedVoxelBlocks;
	int						deletedVoxelBlocksLength;
	int*					mutex; // should be the same length as voxelsHash
	Voxel*					voxels; // numVoxelBlocks * voxelBlockSize * voxelBlockSize * voxelBlockSize
	VoxelHashParameters* 	params;

    __device__ __host__
	void initialize();

    __device__ __host__
	uint computeHashPos(const int3& voxelPosition);

    __device__ __host__
	int3 getVoxelWorldPosition(uint voxelIndexInBlock, int3 voxelBlockPos);

    __device__ __host__
	uint linearizeVoxelPosition(int3 position);

    __device__ __host__
	int3 delinearizeVoxelPosition(uint idx);

    __device__ __host__
    void combineVoxels(const Voxel &v0, const Voxel& v1, Voxel &out);

    __device__ __host__
	float getTruncation(float z);

    __device__ __host__
	VoxelHashEntry getHashEntryByPosition(int3 worldPosition);

    __device__ __host__
	void deleteHashEntry(uint hashEntryIdx);

    __device__ __host__
	void deleteHashEntry(VoxelHashEntry& voxelHashEntry);

    __device__ __host__
	bool insertHashEntry(VoxelHashEntry voxelHashEntry);

#ifdef __CUDA__ARCH__
    __device__
	void mutex_lock(int idx);

    __device__
	void mutex_unlock(int idx);
#endif

    __device__
	void toDeletedVoxelBlocks(uint voxelBlockIdx);

    __device__
	uint fromDeletedVoxelBlocks();


};

#endif
