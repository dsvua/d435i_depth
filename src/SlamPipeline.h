#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include "VoxelHashMap.h"

#ifndef MY_SLAM_PIPELINE


class SlamPipeline{
    public:
        SlamPipeline(rs2_intrinsics intristics);
        ~SlamPipeline();

    private:
        VoxelHashMap        _d_voxelHashData;
        VoxelHashMap        _h_voxelHashData;
        rs2_intrinsics      _d_intristics;
        rs2_intrinsics      _h_intristics;
}

#endif
