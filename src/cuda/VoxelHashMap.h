#ifndef VOXEL_HASH_MAP
#define VOXEL_HASH_MAP

#include <cuda_runtime.h>
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <Eigen/Core>

enum HashState {
	NO_OFFSET = -1,
	FREE_ENTRY = -2,
	FAIL_TO_INSERT = -3
};

struct Voxel {
	float	sdf;		//signed distance function
	float	weight;		//accumulated sdf weight

    __device__ __host__
	Voxel() {
		sdf = 0;
		weight = 0;
	}

};

struct VoxelHashEntry {
	Eigen::Vector3f world_pos;
	int voxelBlockIdx;
	int offset;
	bool compactified;

    __device__ __host__
	VoxelHashEntry() {
		world_pos = Eigen::Vector3f(0,0,0);
		voxelBlockIdx = FREE_ENTRY;
		offset = NO_OFFSET;
		compactified = false;
	}
};


struct VoxelHashParameters {
    int   			p0;
    int   			p1;
    int   			p2;
    float       	staticPartOfRMS; // formula is here: https://dev.intelrealsense.com/docs/tuning-depth-cameras-for-best-performance
	float	    	maxIntegrationDistance; // in meters
	float			minIntegrationDistance; // in meters
	float			maxDistanceForICP;
	uint			voxelPhysicalSize; // size of voxel in millimeters
	float			truncationDistance;
	uint	    	integrationWeightMin;
	uint			voxelBlockCubeSize; // voxelBlock becomes voxelBlockSize x voxelBlockSize x voxelBlockSize voxels size
	uint	    	integrationWeightMax;
	uint			voxelsTotalSize;
	uint			voxelBlockSize;
	uint            numVoxelBlocks;
	uint            voxelHashSize; // how many hash blocks to allocate
	uint			voxelHashBlockSize; // number of hashEntries per hash block
	uint			voxelHashTotalSize; // voxelHashSize * voxelHashBlockSize
	uint			voxelBlockSideSize; // voxelPhysicalSize * voxelBlockSize -- needed for raytracing
	// cuda settings
	int				threadsPerBlock;

    __device__ __host__
	VoxelHashParameters() {
		p0 = 73856093;
		p1 = 19349669;
		p2 = 83492791;
		staticPartOfRMS = 0;
		maxIntegrationDistance = 3000;
		minIntegrationDistance = 100;
		maxDistanceForICP = 6000;
		voxelPhysicalSize = 4;
		truncationDistance = voxelPhysicalSize * 3;
		integrationWeightMin = 1;
		integrationWeightMax = 255;
		voxelBlockSize = 8;
		numVoxelBlocks = 500000;
		voxelBlockCubeSize = voxelBlockSize * voxelBlockSize * voxelBlockSize;
		voxelsTotalSize = numVoxelBlocks * voxelBlockCubeSize;
		voxelHashSize = 10000;
		voxelHashBlockSize = 10;
		voxelHashTotalSize = voxelHashSize * voxelHashBlockSize;
		voxelBlockSideSize = voxelPhysicalSize * voxelBlockSize;
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
	uint computeHashBlockPos(const Eigen::Vector3f voxelBlockWorldPos);

    __device__ __host__
	uint computeHashWorldPos(const Eigen::Vector3f WorldPos);

    __device__ __host__
	Eigen::Vector3f getVoxelWorldPosition(uint voxelIndexInBlock, Eigen::Vector3f voxelBlockWorldPos);

    __device__ __host__
	uint linearizeVoxelWorldPosition(Eigen::Vector3f deltaPos);

    __device__ __host__
	uint linearizeVoxelPosition(Eigen::Vector3f position);

    __device__ __host__
	Eigen::Vector3f delinearizeVoxelPosition(uint idx);

    __device__ __host__
    void combineVoxels(const Voxel &v0, const Voxel& v1, Voxel &out);

    __device__ __host__
	float getTruncation(float z);

    __device__ __host__
	int getHashEntryByWordPosition(Eigen::Vector3f voxelBlockWorldPos);

    __device__ __host__
	int getHashEntryByPosition(Eigen::Vector3f voxelBlockPos);

    __device__ __host__
	Eigen::Vector3f voxelBlockPosFromWorldPos(Eigen::Vector3f worldPosition);

    __device__ __host__
	void deleteHashEntry(uint hashEntryIdx);

    __device__ __host__
	void deleteHashEntry(VoxelHashEntry& voxelHashEntry);

	// inserts hashEntry and returns index where it is inserted
	// or returns FAIL_TO_INSERT if cannot find a free place
    __device__ __host__
	int insertHashEntry(VoxelHashEntry voxelHashEntry);

#ifdef __CUDA__ARCH__
    __device__
	void mutex_lock(int idx);

    __device__
	void mutex_unlock(int idx);
#endif

    __device__
	void deleteVoxelBlocks(uint voxelBlockIdx);

    __device__
	uint getFreeVoxelBlock();


};

#endif
