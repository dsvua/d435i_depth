#include "VoxelHashMap.h"
#include "assert.h"
// #include "safe_call.hpp"
// #include <cmath>



__global__ void initializeVoxelsHash(VoxelHashParameters& params, VoxelHashEntry* voxelsHash, uint* mutex) {
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx < params.voxelHashSize){
        for(int i=0; i<params.voxelHashBlockSize; i++){
            voxelsHash[idx+i] = VoxelHashEntry();
            mutex[idx+i] = 0;
        }
    }
}

__global__ void initializeVoxels(VoxelHashParameters& params, Voxel* voxels, uint* deletedVoxelBlocks) {
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    uint voxelBlockSize = params.voxelBlockSize * params.voxelBlockSize * params.voxelBlockSize;
    if(idx < params.numVoxelBlocks){
        for(int i=0; i<voxelBlockSize; i++){
            voxels[idx+i] = Voxel();
            deletedVoxelBlocks[idx+i] = params.numVoxelBlocks - idx;
        }
    }
}

//------------------------------ VoxelHashMap -------------------------------

__device__ __host__
void VoxelHashMap::initialize() {
#ifdef __CUDA__ARCH__
    // device code
    const dim3 gridSize((params->voxelHashSize + (params->threadsPerBlock*params->threadsPerBlock) - 1)/(params->threadsPerBlock*params->threadsPerBlock), 1);
    const dim3 blockSize((params->threadsPerBlock*params->threadsPerBlock), 1);
    cudaSafeCall(cudaMemset(&(voxelBlocksCompactifiedCounter), 0, sizeof(uint)));
    initializeVoxelsHash<<<gridSize, blockSize>>>(&params, &voxelsHash, &mutex);

    gridSize = dim3((params->numVoxelBlocks + (params->threadsPerBlock*params->threadsPerBlock) - 1)/(params->threadsPerBlock*params->threadsPerBlock), 1);
    blockSize = dim3((params->threadsPerBlock*params->threadsPerBlock), 1);
    cudaSafeCall(cudaMemset(&(deletedVoxelBlocksLength), 0, sizeof(uint)));
    initializeVoxels<<<gridSize, blockSize>>>(&params, &voxels, &deletedVoxelBlocks);

#else
    // host code
#endif
}

__device__ __host__
uint VoxelHashMap::computeHashPos(const int3& voxelPosition) {
    int hash = (voxelPosition.x * params->p0 + voxelPosition.y * params->p1 + voxelPosition.z * params->p2) / params->voxelHashSize;
    // std::abs function have issues with cuda
    if (hash < 0) return (uint)(-hash);
    return (uint)hash;
}

__device__ __host__
int3 VoxelHashMap::getVoxelWorldPosition(uint voxelIndexInBlock, int3 voxelBlockPos) {
    int3 position = delinearizeVoxelPosition(voxelIndexInBlock);
    position.x += voxelBlockPos.x;
    position.y += voxelBlockPos.y;
    position.z += voxelBlockPos.z;
    return position;
}

__device__ __host__
uint VoxelHashMap::linearizeVoxelPosition(int3 position) {
    return  position.z * params->voxelBlockSize * params->voxelBlockSize +
            position.y * params->voxelBlockSize +
            position.x;
}

__device__ __host__
int3 VoxelHashMap::delinearizeVoxelPosition(uint idx) {
    int3 position;
    position.x = idx % params->voxelBlockSize;
    position.y = (idx % (params->voxelBlockSize * params->voxelBlockSize)) / params->voxelBlockSize * params->voxelBlockSize;
    position.z = idx / (params->voxelBlockSize * params->voxelBlockSize);
    return position;
}

//merges two voxels (v0 the currently stored voxel, v1 is the input voxel)
__device__ __host__
void VoxelHashMap::combineVoxels(const Voxel &v0, const Voxel& v1, Voxel &out) {

    out.sdf = (v0.sdf * (float)v0.weight + v1.sdf * (float)v1.weight) / ((float)v0.weight + (float)v1.weight);
    out.weight = min(params->integrationWeightMax, (unsigned int)v0.weight + (unsigned int)v1.weight);
}

//! returns the max truncation distance of the SDF for a given distance value
__device__ __host__
float VoxelHashMap::getTruncation(float z) {
    return params->truncationDistance + params->staticPartOfRMS * z;
}

__device__ __host__
VoxelHashEntry VoxelHashMap::getHashEntryByPosition(int3 worldPosition) {
    uint idx = computeHashPos(worldPosition);

    // search in hash bucket
    for (int i=0; i < params->voxelHashBlockSize; i++) {
        if (voxelsHash[idx+i].world_pos.x == worldPosition.x &&
            voxelsHash[idx+i].world_pos.y == worldPosition.y &&
            voxelsHash[idx+i].world_pos.z == worldPosition.z) {
            // found hashEntry, exit loop and return it
            return voxelsHash[idx+i];
        }
    }

    // search over linked list
    uint tmp_offset = voxelsHash[idx+params->voxelHashBlockSize-1].offset;
    while ((int)tmp_offset != NO_OFFSET) {
        if (voxelsHash[tmp_offset].world_pos.x == worldPosition.x &&
            voxelsHash[tmp_offset].world_pos.y == worldPosition.y &&
            voxelsHash[tmp_offset].world_pos.z == worldPosition.z) {
            // found hashEntry, exit loop and return it
            return voxelsHash[tmp_offset];
        }
        tmp_offset = voxelsHash[tmp_offset].offset;
    }

    // did not find any, return empty VoxelHashEntry
    VoxelHashEntry tmpVoxelHashEntry;

    return tmpVoxelHashEntry;
}

__device__ __host__
void VoxelHashMap::deleteHashEntry(uint hashEntryIdx) {
    deleteHashEntry(voxelsHash[hashEntryIdx]);
}

__device__ __host__
void VoxelHashMap::deleteHashEntry(VoxelHashEntry& voxelHashEntry) {
// I am not sure bucket needs to be locked on deletion
// #ifdef __CUDA__ARCH__
//     mutex_lock(idx);
// #endif

    if (voxelHashEntry.offset == NO_OFFSET) {
        voxelHashEntry.world_pos = make_int3(0,0,0);
        voxelHashEntry.voxelBlockIdx = FREE_ENTRY;
		voxelHashEntry.offset = NO_OFFSET;

    } else {
        voxelHashEntry = voxelsHash[voxelHashEntry.offset];
        voxelsHash[voxelHashEntry.offset].world_pos = make_int3(0,0,0);
        voxelsHash[voxelHashEntry.offset].voxelBlockIdx = FREE_ENTRY;
		voxelsHash[voxelHashEntry.offset].offset = NO_OFFSET;
    }

// #ifdef __CUDA__ARCH__
//     cudaSafeCall(cudaFree(voxelBlock);
//     mutex_unlock(idx);
// #endif
}

__device__ __host__
bool VoxelHashMap::insertHashEntry(VoxelHashEntry voxelHashEntry) {
    uint idx = computeHashPos(voxelHashEntry.world_pos);
#ifdef __CUDA__ARCH__
    mutex_lock(idx);
#endif
    // search in hash bucket
    for (int i=0; i < params->voxelHashBlockSize; i++) {
        if (voxelsHash[idx+i].voxelBlockIdx == FREE_ENTRY) {
            // found hashEntry, exit loop and return true
            voxelsHash[idx+i] = voxelHashEntry;
        #ifdef __CUDA__ARCH__
            mutex_unlock(idx); //unlock before exiting loop
        #endif
            return true;
        }
    } //no free slot found, need to append to offset linked list

    // search over linked list to find last offset item in a list
    uint tmp_offsetEntryIdx = idx+params->voxelHashBlockSize-1;
    while (voxelsHash[tmp_offsetEntryIdx].offset != NO_OFFSET) {
        tmp_offsetEntryIdx = voxelsHash[tmp_offsetEntryIdx].offset;
    }

    // tmp_offsetEntryIdx is an index of last hashEntry, now we can search
    // for empty slot to insert our new hash entry
    uint tmp_newOffsetEntryIdx = (tmp_offsetEntryIdx / params->voxelHashBlockSize + 1) * params->voxelHashBlockSize;
    tmp_newOffsetEntryIdx %= params->voxelHashTotalSize; // rollover if next hash index is outside

    bool tmp_idxIsFound = false;
    int halfVoxelHashBlockSize = params->voxelHashBlockSize / 2;
    int tmp_i = 0;
    while (tmp_idxIsFound) {
        // checking first half of next bucket for empty slots
        for(int i=0; i < halfVoxelHashBlockSize; i++){
            if (voxelsHash[tmp_newOffsetEntryIdx + i].offset == NO_OFFSET) {
            #ifdef __CUDA__ARCH__
                mutex_lock(tmp_newOffsetEntryIdx + i); // lock this bucket as well
            #endif
            
                voxelsHash[tmp_offsetEntryIdx].offset = tmp_newOffsetEntryIdx + i;
                voxelsHash[tmp_newOffsetEntryIdx + i] = voxelHashEntry;

            #ifdef __CUDA__ARCH__
                // unlocking both buckets
                mutex_unlock(tmp_newOffsetEntryIdx + i);
                mutex_unlock(idx); //unlock before exiting loop
            #endif
                return true;
            }
        }
        // checking next bucket
        tmp_newOffsetEntryIdx += params->voxelHashBlockSize;

        // prevent infinite loop if free slot could not be found
        tmp_i++;
        if (tmp_i > params->voxelHashSize) tmp_idxIsFound = true;
    }


#ifdef __CUDA__ARCH__
    mutex_unlock(idx);
#endif
    // could not insert, return false - should never happens
    return false;
}

#ifdef __CUDA__ARCH__
__device__
void VoxelHashMap::mutex_lock(int idx) {
    while(atomicCas(mutex[idx], 0, 1) != 0);
}

__device__
void VoxelHashMap::mutex_unlock(int idx) {
    atomicExch(mutex[idx], 0);
}
#endif

__device__
void VoxelHashMap::toDeletedVoxelBlocks(uint voxelBlockIdx) {
    uint addr = atomicAdd(&deletedVoxelBlocksLength, 1);
    deletedVoxelBlocks[addr+1] = voxelBlockIdx;
}

__device__
uint VoxelHashMap::fromDeletedVoxelBlocks() {
    uint addr = atomicSub(&deletedVoxelBlocksLength, 1);
    return deletedVoxelBlocks[addr];
}



