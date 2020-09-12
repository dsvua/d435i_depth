
#include "cuda_declarations.h"
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <stdint.h>
#include "hash_functions.h"
#include "cuda_simple_matrix_math.h"

#ifndef MY_CUDA_PIPELINE

class CudaPipeline {
    public:
        explicit CudaPipeline(int argc, char ** argv, rs2_intrinsics intristics){
            _intristics = intristics;
            _cuda_device = init_cuda_device(argc, argv);
            host_points = (float*) malloc (sizeof(float) * _intristics.height * _intristics.width * 3);

            // TODO - read from file
            m_hashParams.m_rigidTransform.setIdentity();
            m_hashParams.m_rigidTransformInverse.setIdentity();
            m_hashParams.m_hashNumBuckets = 500000;
            m_hashParams.m_hashBucketSize = HASH_BUCKET_SIZE;
            m_hashParams.m_hashMaxCollisionLinkedListSize = 70;
            m_hashParams.m_SDFBlockSize = SDF_BLOCK_SIZE;
            m_hashParams.m_numSDFBlocks = 1000000;
            m_hashParams.m_virtualVoxelSize = 0.004f;
            m_hashParams.m_maxIntegrationDistance = 4.0f;
            m_hashParams.m_minIntegrationDistance = 0.02f;
            m_hashParams.m_truncation = 0.02f;
            m_hashParams.m_truncScale = 0.01f;
            m_hashParams.m_integrationWeightSample = 10;
            m_hashParams.m_integrationWeightMax = 255;
            m_hashParams.m_streamingVoxelExtents = make_float3(1.0f, 1.0f, 1.0f);
            m_hashParams.m_streamingGridDimensions = make_int3(257, 257, 257);
            m_hashParams.m_streamingMinGridPos = make_int3(-128, -128, -128);
            m_hashParams.m_streamingInitialChunkListSize = 2000;

        };

        ~CudaPipeline(){
            m_hashData.free();
        };

        void process_depth(rs2::depth_frame depth_frame);
        void integrate(const float4x4& lastRigidTransform, const uint16_t * dev_depth);

        bool download_points = false;
        float *host_points = 0;
        int maxDistance = 8000;
        int minDistance = 200;
    
    private:
        int _cuda_device = 0;
        rs2_intrinsics* dev_intrin = NULL;
        rs2_intrinsics _intristics;
        HashParams	   m_hashParams;
        HashData	   m_hashData;
        int            m_numIntegratedFrames = 0;

        int init_cuda_device(int argc, char **argv);
        

};


#endif
