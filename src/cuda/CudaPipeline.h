
#ifndef RS2_CUDA_THREADS_PER_BLOCK
#define RS2_CUDA_THREADS_PER_BLOCK 32
#endif

#ifndef MY_CUDA_PIPELINE

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <stdint.h>
// #include "hash_functions.h"

//void upload_depth_to_cuda(int argc, char * argv[], rs2::depth_frame frame, rs2_intrinsics intristics);

class CudaPipeline {
    public:
        explicit CudaPipeline(int argc, char ** argv, rs2_intrinsics intristics){
            _cuda_device = get_cuda_device(argc, argv);
            _intristics = intristics;
            host_points = (float*) malloc (sizeof(float) * _intristics.height * _intristics.width * 3);
        };
        ~CudaPipeline(){};

        void process_depth(rs2::depth_frame depth_frame);
        void integrate_depth();

        bool download_points = false;
        float *host_points = 0;
        int maxDistance = 8000;
        int minDistance = 200;
    
    private:
        int _cuda_device = 0;
        rs2_intrinsics _intristics;

        int get_cuda_device(int argc, char **argv);
        

};


#endif
