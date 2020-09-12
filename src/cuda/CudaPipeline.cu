
// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include "helper_functions.h"
#include "helper_cuda.h"

#include <iostream>             // for cout
#include "deproject_point.h"
#include "CudaPipeline.h"
#include "cuda_hash_params.h"
#include "hash_functions.h"

int CudaPipeline::init_cuda_device(int argc, char **argv) {
    checkCudaErrors(cudaMalloc(&dev_intrin, sizeof(rs2_intrinsics)));
    checkCudaErrors(cudaMemcpy(dev_intrin, &_intristics, sizeof(rs2_intrinsics), cudaMemcpyHostToDevice));

    return findCudaDevice(argc, (const char **)argv);
}

void CudaPipeline::process_depth(rs2::depth_frame depth_frame) {
    // upload
    int count = _intristics.height * _intristics.width;
    int numBlocks = count / RS2_CUDA_THREADS_PER_BLOCK;

    uint16_t *depth_data = (uint16_t *)depth_frame.get_data();
    uint16_t *dev_depth = 0;
    float *dev_points = 0;

    checkCudaErrors(cudaMalloc(&dev_points, count * sizeof(float) * 3));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&dev_depth), count * sizeof(uint16_t)));
    checkCudaErrors(cudaMemcpy(dev_depth, depth_data, count * sizeof(uint16_t), cudaMemcpyHostToDevice));

    std::cout << "image uploaded to gpu" << std::endl;

    // counverting depth image into points cloud and keeping points on gpu device
    kernel_deproject_depth_cuda<<<numBlocks, RS2_CUDA_THREADS_PER_BLOCK>>>(dev_points, dev_intrin, dev_depth,
        minDistance, maxDistance); 
    getLastCudaError("Failed: kernel_deproject_depth_cuda");

    std::cout << "Points cloud is computed" << std::endl;

    if (download_points){
        checkCudaErrors(cudaMemcpy(host_points, dev_points, count * sizeof(float) * 3, cudaMemcpyDeviceToHost));
    };

    std::cout << "Points cloud is downloaded to host" << std::endl;

    // reconstruction begins
    float4x4 transformation = float4x4::identity();

    if (m_numIntegratedFrames > 0){
        // need to find rotation first

        // g_rayCast->render(g_sceneRep->getHashData(), g_sceneRep->getHashParams(), g_CudaDepthSensor.getDepthCameraData(), renderTransform);
        float4x4 lastTransform = getLastRigidTransform();
        float4x4 deltaTransformEstimate = float4x4::identity();
        applyCT(
            g_CudaDepthSensor.getCameraSpacePositionsFloat4(), g_CudaDepthSensor.getNormalMapFloat4(), g_CudaDepthSensor.getColorMapFilteredFloat4(),
            //g_rayCast->getRayCastData().d_depth4Transformed, g_CudaDepthSensor.getNormalMapNoRefinementFloat4(), g_CudaDepthSensor.getColorMapFilteredFloat4(),
            g_rayCast->getRayCastData().d_depth4, g_rayCast->getRayCastData().d_normals, g_rayCast->getRayCastData().d_colors,
            lastTransform,
            GlobalCameraTrackingState::getInstance().s_maxInnerIter, GlobalCameraTrackingState::getInstance().s_maxOuterIter,
            GlobalCameraTrackingState::getInstance().s_distThres,	 GlobalCameraTrackingState::getInstance().s_normalThres,
            100.0f, 3.0f,
            deltaTransformEstimate,
            GlobalCameraTrackingState::getInstance().s_residualEarlyOut,
            g_RGBDAdapter.getDepthIntrinsics(), g_CudaDepthSensor.getDepthCameraData(), 
            NULL);

    }

    integrate(transformation, dev_depth);

    std::cout << "Image is integrated" << std::endl;

}

void CudaPipeline::setLastRigidTransform(const float4x4& lastRigidTransform) {
    m_hashParams.m_rigidTransform = lastRigidTransform;
    m_hashParams.m_rigidTransformInverse = m_hashParams.m_rigidTransform.getInverse();
}

void CudaPipeline::render() {
	// rayIntervalSplatting(hashData, hashParams, cameraData, lastRigidTransform);
	if (hashParams.m_numOccupiedBlocks == 0)	return;

	if (m_rayCastParams.m_maxNumVertices <= 6*hashParams.m_numOccupiedBlocks) { // 6 verts (2 triangles) per block
		MLIB_EXCEPTION("not enough space for vertex buffer for ray interval splatting");
	}

	m_rayCastParams.m_numOccupiedSDFBlocks = hashParams.m_numOccupiedBlocks;
	m_rayCastParams.m_viewMatrix = MatrixConversion::toCUDA(lastRigidTransform.getInverse());
	m_rayCastParams.m_viewMatrixInverse = MatrixConversion::toCUDA(lastRigidTransform);

	m_data.updateParams(m_rayCastParams); // !!! debugging
    // -----
    m_data.d_rayIntervalSplatMinArray = m_rayIntervalSplatting.mapMinToCuda();
	m_data.d_rayIntervalSplatMaxArray = m_rayIntervalSplatting.mapMaxToCuda();

	renderCS(hashData, m_data, cameraData, m_rayCastParams);

	//convertToCameraSpace(cameraData);
	if (!m_rayCastParams.m_useGradients)
	{
		computeNormals(m_data.d_normals, m_data.d_depth4, m_rayCastParams.m_width, m_rayCastParams.m_height);
	}

	m_rayIntervalSplatting.unmapCuda();

	// Wait for query
	if(GlobalAppState::getInstance().s_timingsDetailledEnabled)
	{
		cutilSafeCall(cudaDeviceSynchronize()); 
		m_timer.stop();
		TimingLog::totalTimeRayCast+=m_timer.getElapsedTimeMS();
		TimingLog::countTimeRayCast++;
	}

}