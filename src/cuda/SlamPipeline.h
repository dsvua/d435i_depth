#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include "VoxelHashMap.h"

#include <sophus/se3.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "ICPOdometry.h"

#ifndef MY_SLAM_PIPELINE
#define MY_SLAM_PIPELINE


class SlamPipeline{
    public:
        SlamPipeline(rs2_intrinsics intristics);
        // ~SlamPipeline();
        void processDepth(rs2::depth_frame depth);

    private:
        VoxelHashMap*   _d_voxelHashData;
        VoxelHashMap*   _h_voxelHashData;

        rs2_intrinsics* _d_intristics;
        rs2_intrinsics  _h_intristics;
        uint16_t*       _d_depth;

        int             _numOfDepthFramesProcessed;

        void allocateHashData();
        void integrateDepth(rs2::depth_frame depth);
        void renderVoxels();
        void reduceVMap(DeviceArray2D<float> &dst, DeviceArray2D<float> &stc);

        Sophus::SE3d     _currentTransformation;
        Sophus::SE3d     _prevTransformation;

        Eigen::Affine3f* _d_currentAffineTransformation;
        Eigen::Affine3f* _d_currentAffineTransformationInverted;
        Eigen::Affine3f  _h_currentAffineTransformation;
        Eigen::Affine3f  _h_currentAffineTransformationInverted;

        //                       int width,        int height,       float cx,       float cy,        float fx,     float fy,  float distThresh
        // ICPOdometry     _icp(_h_intristics.width, _h_intristics.height, _h_intristics.ppx, _h_intristics.ppy, _h_intristics.fx, _h_intristics.fy, _h_voxelHashData->params->maxDistanceForICP);
        ICPOdometry*    _icp;
};

#endif
