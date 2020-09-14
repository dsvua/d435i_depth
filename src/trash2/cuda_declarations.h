#ifndef RS2_CUDA_THREADS_PER_BLOCK
#define RS2_CUDA_THREADS_PER_BLOCK 32
#endif

#ifndef DEPTH_WORLD_MIN
#define DEPTH_WORLD_MIN 0.1f
#endif
#ifndef DEPTH_WORLD_MAX
#define DEPTH_WORLD_MAX 8.0f
#endif

#ifndef COMPACTIFY_HASH_THREADS_PER_BLOCK
#define COMPACTIFY_HASH_THREADS_PER_BLOCK 256
#endif

#ifndef cudaAssert
#define cudaAssert(condition) if (!(condition)) { printf("ASSERT: %s %s\n", #condition, __FILE__); }
#endif

#ifndef SDF_BLOCK_SIZE
#define SDF_BLOCK_SIZE 8
#endif

#ifndef HASH_BUCKET_SIZE
#define HASH_BUCKET_SIZE 10
#endif

#ifndef T_PER_BLOCK
#define T_PER_BLOCK 16
#endif



