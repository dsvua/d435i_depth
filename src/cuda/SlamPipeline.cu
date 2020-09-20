#include "SlamPipeline.h"
#include "safe_call.h"
// #include <cmath>

SlamPipeline::SlamPipeline(rs2_intrinsics intristics) {
    _h_intristics = intristics;
    _numOfDepthFramesProcessed = 0;

    // those values are for RealSense D435i camera
    const int HFOV = 90; // horizontal field of view for realsense camera
    const float subpixel = 0.2;  // 0.2mm
    const float baseline = 50; // 50mm
    float focalLength_mm = intristics.width * tan(HFOV/2) / 2;

    std::cout << "Initializing VoxelHashParameters" << std::endl;
    _h_voxelHashData = new VoxelHashMap();
    _h_voxelHashData->params = new VoxelHashParameters();
    _h_voxelHashData->params->staticPartOfRMS = subpixel / (focalLength_mm * baseline);


    cudaSafeCall(cudaMalloc(&_d_depth, sizeof(uint16_t) * _h_intristics.width * _h_intristics.height));
    cudaSafeCall(cudaMalloc(&_d_currentAffineTransformation, sizeof(Eigen::Affine3f)));
    cudaSafeCall(cudaMalloc(&_d_currentAffineTransformationInverted, sizeof(Eigen::Affine3f)));

    cudaSafeCall(cudaMalloc(&_d_intristics, sizeof(rs2_intrinsics)));
    cudaSafeCall(cudaMemcpy(_d_intristics, &_h_intristics, sizeof(rs2_intrinsics), cudaMemcpyHostToDevice));

    _currentTransformation = Sophus::SE3d();
    _prevTransformation = Sophus::SE3d();

    allocateHashData();

    //                  int width,        int height,       float cx,       float cy,        float fx,     float fy,  float distThresh
    // _icp = ICPOdometry(intristics.width, intristics.height, intristics.ppx, intristics.ppy, intristics.fx, intristics.fy, _h_voxelHashData->params->maxDistanceForICP);
    _icp = new ICPOdometry(intristics.width, intristics.height, intristics.ppx, intristics.ppy, intristics.fx, intristics.fy, _h_voxelHashData->params->maxDistanceForICP);
}

void SlamPipeline::allocateHashData() {

    std::cout << "Initializing device VoxelHashMap" << std::endl;
    // Allocating memory on device for each pointer element of _d_voxelHashData
    VoxelHashEntry**	    d_tmp_voxelsHash;
    VoxelHashEntry**	    d_tmp_voxelsHashCompactified;
    int*	                d_tmp_deletedVoxelBlocks;
    int*	                d_tmp_mutex;
    VoxelHashParameters*    d_tmp_voxelHashParams;

    // creating hashData
    cudaSafeCall(cudaMalloc((void**)&_d_voxelHashData, sizeof(VoxelHashMap)));
    // creating pointers arrays
    cudaSafeCall(cudaMalloc((void**)&d_tmp_voxelHashParams, sizeof(VoxelHashParameters)));
    cudaSafeCall(cudaMemcpy(d_tmp_voxelHashParams, _h_voxelHashData->params, sizeof(VoxelHashParameters), cudaMemcpyHostToDevice));

    cudaSafeCall(cudaMalloc((void**)&d_tmp_voxelsHash, sizeof(VoxelHashEntry) * _h_voxelHashData->params->voxelHashTotalSize));
    cudaSafeCall(cudaMalloc((void**)&d_tmp_voxelsHashCompactified, sizeof(VoxelHashEntry) * _h_voxelHashData->params->voxelHashTotalSize));
    cudaSafeCall(cudaMalloc((void**)&d_tmp_deletedVoxelBlocks, sizeof(uint) * _h_voxelHashData->params->voxelHashSize));
    cudaSafeCall(cudaMalloc((void**)&d_tmp_mutex, sizeof(int) * _h_voxelHashData->params->voxelHashSize));
    // cudaSafeCall(cudaMemset(d_tmp_mutex, 0, sizeof(uint) * _voxelHashSize));

    // NOTE: Binding pointers with _d_voxelHashData on device
    cudaSafeCall(cudaMemcpy(&(_d_voxelHashData->params), &d_tmp_voxelHashParams, sizeof(_d_voxelHashData->params), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(&(_d_voxelHashData->voxelsHash), &d_tmp_voxelsHash, sizeof(_d_voxelHashData->voxelsHash), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(&(_d_voxelHashData->voxelsHashCompactified), &d_tmp_voxelsHashCompactified, sizeof(_d_voxelHashData->voxelsHashCompactified), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(&(_d_voxelHashData->deletedVoxelBlocks), &d_tmp_deletedVoxelBlocks, sizeof(_d_voxelHashData->deletedVoxelBlocks), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(&(_d_voxelHashData->mutex), &d_tmp_mutex, sizeof(_d_voxelHashData->mutex), cudaMemcpyHostToDevice));
    // copy params


    std::cout << "running _d_voxelHashData.initialize()" << std::endl;
    _d_voxelHashData->initialize();
    std::cout << "Done initializing _d_voxelHashData" << std::endl;
}

void SlamPipeline::processDepth(rs2::depth_frame depth) {


    _icp->initICP((unsigned short *)depth.get_data(), _h_voxelHashData->params->maxDistanceForICP); // depth image here

    _prevTransformation = _currentTransformation;
    // _currentTransformation = Sophus::SE3d(); // convert to eigen matrix Eigen::Matrix4f == _currentTransformation.cast<float>().matrix()
    if (_numOfDepthFramesProcessed > 0) {
        //do ICP
        Sophus::SE3d delta_transformation = _prevTransformation.inverse() * _currentTransformation;

        // icp.initICPModel(firstRaw.ptr); // render() will create vmaps and nmaps directly
        renderVoxels(); // create point cloud and normals for previuos position
        int threads = 256;
        int blocks = 96;
        _icp->getIncrementalTransformation(delta_transformation, threads, blocks);
        _currentTransformation = _prevTransformation * delta_transformation;
    } else {
        _icp->initICPModel((unsigned short *)depth.get_data(), _h_voxelHashData->params->maxDistanceForICP); // creates vmaps_prev and nmaps_prev for renderVoxels()
    }

    // since translation is available, integrate new frame into SDF
    integrateDepth(depth);
    _numOfDepthFramesProcessed += 1;

}


