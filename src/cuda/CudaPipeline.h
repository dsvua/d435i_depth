
#include "cuda_declarations.h"
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <stdint.h>
#include <vector.h>
#include "hash_functions.h"
#include "cuda_simple_matrix_math.h"
#include "cuda_raycast_params.h"

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
            m_hashParams.m_numSDFBlocks = 500000;
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

            m_hashData.allocate(m_hashParams, true);

            // temporary array to initialize float4x4 later
            float tmp_intr[] = {_intristics.fx,    0,          _intristics.ppx, 0,
                                0,             _intristics.fy, _intristics.ppy, 0,
                                0,                 0,              1,           0,
                                0,                 0,              0,           0};

            m_rayCastParams.m_width = 848;
            m_rayCastParams.m_height = 480;
            m_rayCastParams.m_intrinsics = float4x4(tmp_intr);
            m_rayCastParams.m_intrinsicsInverse = m_rayCastParams.m_intrinsics.getInverse();
            m_rayCastParams.m_minDepth = 0.02f;
            m_rayCastParams.m_maxDepth = 5.0f;
            m_rayCastParams.m_rayIncrement = 0.8f * 0.02f;
            m_rayCastParams.m_thresSampleDist = 50.5f * m_rayCastParams.m_rayIncrement;
            m_rayCastParams.m_thresDist = 50.0f * m_rayCastParams.m_rayIncrement;
            m_rayCastParams.m_useGradients = false;
            m_rayCastParams.m_maxNumVertices = 1000000 * 6;

            m_rayCastData.allocate(m_rayCastParams);
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
        rs2_intrinsics  _intristics;
        HashParams	    m_hashParams;
        HashData	    m_hashData;
        RayCastParams   m_rayCastParams;
        int             m_numIntegratedFrames = 0;
        RayCastData     m_rayCastData;
        
        // should be read from config file
        struct CameraTrackingState {
            unsigned int s_maxLevels = 3;
            std::vector<unsigned int> s_maxOuterIter{8,6,4,4};
            std::vector<unsigned int> s_maxInnerIter{1,1,1,1};
            std::vector<float> s_weightsDepth{1.0f,0.5f,0.5f,0.5f};
            std::vector<float> s_distThres{0.15f,0.15f,0.15f,0.15f};
            std::vector<float> s_normalThres{0.97f,0.97f,0.97f,0.97f};
            std::vector<float> s_angleTransThres{1.0f,1.0f,1.0f,1.0f};
            std::vector<float> s_distTransThres{1.0f,1.0f,1.0f,1.0f};
            std::vector<float> s_residualEarlyOut{0.01f,0.01f,0.01f,0.01f};
        }

        CameraTrackingState cameraTrackingState;

        int init_cuda_device(int argc, char **argv);
        void setLastRigidTransform(const float4x4& lastRigidTransform);
        void render();
        void computeNormals(float4* d_output, float4* d_input, unsigned int width, unsigned int height);
        

};


#endif
