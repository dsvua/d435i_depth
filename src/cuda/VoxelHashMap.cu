#include "VoxelHashMap.h"
#include "assert.h"
#include "safe_call.h"
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
uint VoxelHashMap::computeHashBlockPos(const Eigen::Vector3f voxelBlockWorldPos) {
    int hash = (voxelBlockWorldPos.x() * params->p0 + voxelBlockWorldPos.y() * params->p1 + voxelBlockWorldPos.z() * params->p2) / params->voxelHashSize;
    // std::abs function have issues with cuda
    if (hash < 0) return (uint)(-hash);
    return (uint)hash;
}

__device__ __host__
uint VoxelHashMap::computeHashWorldPos(const Eigen::Vector3f WorldPos) {
    return computeHashBlockPos(voxelBlockPosFromWorldPos(WorldPos));
}

__device__ __host__
Eigen::Vector3f VoxelHashMap::getVoxelWorldPosition(uint voxelIndexInBlock, Eigen::Vector3f voxelBlockWorldPos) {
    Eigen::Vector3f position = delinearizeVoxelPosition(voxelIndexInBlock);
    position.x() += voxelBlockWorldPos.x();
    position.y() += voxelBlockWorldPos.y();
    position.z() += voxelBlockWorldPos.z();
    return position;
}

__device__ __host__
uint VoxelHashMap::linearizeVoxelWorldPosition(Eigen::Vector3f deltaPos) {
    int x = (int)deltaPos.x();
    int y = (int)deltaPos.y();
    int z = (int)deltaPos.z();
    int d = (int)params->voxelPhysicalSize;

    // if (x<0) x -= d;
    // if (y<0) y -= d;
    // if (z<0) z -= d;

    return linearizeVoxelPosition(Eigen::Vector3f(x/d, y/d, z/d));
}

__device__ __host__
uint VoxelHashMap::linearizeVoxelPosition(Eigen::Vector3f position) {
    return  position.z() * params->voxelBlockSize * params->voxelBlockSize +
            position.y() * params->voxelBlockSize +
            position.x();
}

__device__ __host__
Eigen::Vector3f VoxelHashMap::delinearizeVoxelPosition(uint idx) {
    Eigen::Vector3f position;
    position.x() = idx % params->voxelBlockSize;
    position.y() = (idx % (params->voxelBlockSize * params->voxelBlockSize)) / params->voxelBlockSize * params->voxelBlockSize;
    position.z() = idx / (params->voxelBlockSize * params->voxelBlockSize);
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
    return params->truncationDistance + params->staticPartOfRMS * z * z;
}

__device__ __host__
int VoxelHashMap::getHashEntryByWordPosition(Eigen::Vector3f voxelBlockWorldPos) {
    return getHashEntryByPosition(voxelBlockPosFromWorldPos(voxelBlockWorldPos));
}

    __device__ __host__
int VoxelHashMap::getHashEntryByPosition(Eigen::Vector3f voxelBlockPos) {
    uint idx = computeHashBlockPos(voxelBlockPos);

    // search in hash bucket
    for (int i=0; i < params->voxelHashBlockSize; i++) {
        if (voxelsHash[idx+i].world_pos.x() == voxelBlockPos.x() &&
            voxelsHash[idx+i].world_pos.y() == voxelBlockPos.y() &&
            voxelsHash[idx+i].world_pos.z() == voxelBlockPos.z()) {
            // found hashEntry, exit loop and return it
            // return voxelsHash[idx+i];
            return idx+i;
        }
    }

    // search over linked list
    uint tmp_offset = voxelsHash[idx + params->voxelHashBlockSize-1].offset;
    while ((int)tmp_offset != NO_OFFSET) {
        if (voxelsHash[tmp_offset].world_pos.x() == voxelBlockPos.x() &&
            voxelsHash[tmp_offset].world_pos.y() == voxelBlockPos.y() &&
            voxelsHash[tmp_offset].world_pos.z() == voxelBlockPos.z()) {
            // found hashEntry, exit loop and return it
            // return voxelsHash[tmp_offset];
            return tmp_offset;
        }
        tmp_offset = voxelsHash[tmp_offset].offset;
    }

    // // did not find any, allocate and return empty VoxelHashEntry and voxelBlock
    // VoxelHashEntry tmp_entry = VoxelHashEntry();
    // tmp_entry.world_pos = voxelBlockWorldPos;
    // tmp_entry.voxelBlockIdx = getFreeVoxelBlock();
    // // tmp_entry.world_pos = voxelBlockPosFromWorldPos(worldPosition);

    // return insertHashEntry(tmp_entry);
    return FREE_ENTRY;
}

__device__ __host__
Eigen::Vector3f VoxelHashMap::voxelBlockPosFromWorldPos(Eigen::Vector3f worldPosition) {
    int x = (int)worldPosition.x();
    int y = (int)worldPosition.y();
    int z = (int)worldPosition.z();
    int d = (int)params->voxelBlockSideSize;

    if (x<0) x -= d;
    if (y<0) y -= d;
    if (z<0) z -= d;

    return Eigen::Vector3f(x/d, y/d, z/d);
}

__device__ __host__
void VoxelHashMap::deleteHashEntry(uint hashEntryIdx) {
    deleteHashEntry(voxelsHash[hashEntryIdx]); // need to send address, not value
}

__device__ __host__
void VoxelHashMap::deleteHashEntry(VoxelHashEntry& voxelHashEntry) {
// I am not sure bucket needs to be locked on deletion
// #ifdef __CUDA__ARCH__
//     mutex_lock(idx);
// #endif

    if (voxelHashEntry.offset == NO_OFFSET) {
        voxelHashEntry.world_pos = Eigen::Vector3f(0,0,0);
        voxelHashEntry.voxelBlockIdx = FREE_ENTRY;
		voxelHashEntry.offset = NO_OFFSET;

    } else {
        voxelHashEntry = voxelsHash[voxelHashEntry.offset];
        voxelsHash[voxelHashEntry.offset].world_pos = Eigen::Vector3f(0,0,0);
        voxelsHash[voxelHashEntry.offset].voxelBlockIdx = FREE_ENTRY;
		voxelsHash[voxelHashEntry.offset].offset = NO_OFFSET;
    }

// #ifdef __CUDA__ARCH__
//     cudaSafeCall(cudaFree(voxelBlock);
//     mutex_unlock(idx);
// #endif
}

__device__ __host__
int VoxelHashMap::insertHashEntry(VoxelHashEntry voxelHashEntry) {
    uint idx = computeHashBlockPos(voxelHashEntry.world_pos);
#ifdef __CUDA__ARCH__
    mutex_lock(idx);
#endif
    // check if it is already inserted
    int t_idx = getHashEntryByPosition(voxelHashEntry.world_pos);

    if (t_idx != FREE_ENTRY) {
        // entry already exists return FAIL_TO_INSERT
    #ifdef __CUDA__ARCH__
        mutex_unlock(idx); //unlock before exiting loop
    #endif
        return FAIL_TO_INSERT;
    }
    // search in hash bucket
    for (int i=0; i < params->voxelHashBlockSize; i++) {
        if (voxelsHash[idx+i].voxelBlockIdx == FREE_ENTRY) {
            // found hashEntry, exit loop and return true
            voxelsHash[idx+i] = voxelHashEntry;
        #ifdef __CUDA__ARCH__
            mutex_unlock(idx); //unlock before exiting loop
        #endif
            return idx+i;
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
                return tmp_newOffsetEntryIdx + i;
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
    // could not insert - should never happens
    return FAIL_TO_INSERT;
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
void VoxelHashMap::deleteVoxelBlocks(uint voxelBlockIdx) {
    uint addr = atomicAdd(&deletedVoxelBlocksLength, 1);
    uint t_idx = voxelBlockIdx * params->voxelBlockSize;
    deletedVoxelBlocks[addr+1] = voxelBlockIdx;
    for (int i=0; i<params->voxelBlockSize; i++) voxels[t_idx + i] = Voxel(); // resetting voxels in a block
}

__device__
uint VoxelHashMap::getFreeVoxelBlock() {
    uint addr = atomicSub(&deletedVoxelBlocksLength, 1);
    return deletedVoxelBlocks[addr];
}



