#ifndef VOXEL_HASH_MAP

#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API

struct HashEntry {
	int		voxelBlockIdx;	//index of SDFBlock in voxelBlocks
};

struct Voxel {
	float	sdf;		//signed distance function
	float	weight;		//accumulated sdf weight

};

struct VoxelBlock {
	int3	world_pos;	    //hash position (lower left corner of SDFBlock))

    // when there more then one block fits into same hashed position - create linked list
	uint	nextVoxelBlock;	//if 0 - last item in a list
	uint	prevVoxelBlock;	// if 0 - first item in a list
};

// Stores voxelBlocks in voxelsHash
struct VoxelHashMap{

	VoxelBlock*	voxelBlocks;			//heap that manages free memory
	uint		voxelUsedBlocksCounter;		//single element; used as an atomic counter (points to the next free block)
    uint*       deletedVoxelBlocks; // list of deleted blocks, new deleted block is added
    uint        deletedVoxelCount;  // at deletedVoxelCount position and deletedVoxelCount +=1
	HashEntry*	voxelsHash;						//hash that stores pointers to sdf blocks
	HashEntry*	voxelsHashCompactified;			//same as before except that only valid pointers are there
	uint		voxelsHashCompactifiedCounter;	//atomic counter to add compactified entries atomically 
	int		    voxelsHashMutex;			//binary flag per hash bucket; used for allocation to atomically lock a bucket
	bool		isOnGPU = false;			//the class be be used on both cpu and gpu
    const int   p0      = 73856093;
    const int   p1      = 19349669;
    const int   p2      = 83492791;
    int   voxelHashSize;
    float       staticPartOfRMS; // formula is here: https://dev.intelrealsense.com/docs/tuning-depth-cameras-for-best-performance

	__device__ __host__
	VoxelHashMap(){
        voxelBlocks = NULL;
        voxelUsedBlocksCounter = 0;
        deletedVoxelBlocks = NULL;
        deletedVoxelCount = 0;
        voxelsHash = NULL;
        voxelsHashCompactified = NULL;
        voxelsHashCompactifiedCounter = 0;
        voxelsHashMutex = 0;
        voxelHashSize = 0;
        staticPartOfRMS = 0;
    }

	__device__ __host__
	VoxelHashMap(const bool _isOnGPU, // create it on GPU or CPU
                const uint _numVoxelBlocks, // total number of voxelBlocks
                const uint _voxelHashSize // hash table size
                ) {
        voxelUsedBlocksCounter = 0;
        deletedVoxelCount = 0;
        voxelsHashCompactifiedCounter = 0;
        voxelHashSize = _voxelHashSize;
        isOnGPU = _isOnGPU;

        // allocate memory on GPU
        if (_isOnGPU) {
            checkCudaErrors(cudaMalloc(&voxelBlocks, sizeof(VoxelBlock) * _numVoxelBlocks));
            checkCudaErrors(cudaMalloc(&deletedVoxelBlocks, sizeof(unsigned int) * _numVoxelBlocks));
            checkCudaErrors(cudaMalloc(&voxelsHash, sizeof(unsigned int) * _voxelHashSize));
            checkCudaErrors(cudaMalloc(&voxelsHashCompactified, sizeof(unsigned int) * _voxelHashSize));
        
        // allocate memory on CPU
        } else {
            voxelBlocks = new unsigned int[_numVoxelBlocks];
            deletedVoxelBlocks = new unsigned int[_numVoxelBlocks];
            voxelsHash = new HashEntry[_voxelHashSize];
            voxelsHashCompactified = new HashEntry[_voxelHashSize];
        }
    }

    ~VoxelHashMap(){
        // deallocate memory on GPU
        if (isOnGPU) {
            checkCudaErrors(cudaFree(voxelBlocks));
            checkCudaErrors(cudaFree(deletedVoxelBlocks));
            checkCudaErrors(cudaFree(voxelsHash));
            checkCudaErrors(cudaFree(voxelsHashCompactified));
        
        // deallocate memory on CPU
        } else {
            if (voxelBlocks) = delete[] voxelBlocks;
            if (deletedVoxelBlocks) = delete[] voxelBlocks;
            if (voxelsHash) = delete[] voxelHashSize;
            if (voxelsHashCompactified) = delete[] voxelHashSize;
        }
    }

    __device__ __host__
	uint computeHashPos(int x, int y, int z) const;

    __device__ __host__
    void combineVoxel(const Voxel &v0, const Voxel& v1) const;
}

#endif
