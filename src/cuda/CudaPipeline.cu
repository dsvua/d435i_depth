
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
#include "assert.h"
#include "icp_kernels.h"
#include "LinearSystemConfidence"

void CudaPipeline::computeNormals(float3* d_output, float3* d_input, unsigned int width, unsigned int height) {
	const dim3 gridSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	computeNormalsDevice<<<gridSize, blockSize>>>(d_output, d_input, width, height);
    getLastCudaError("Failed: computeNormalsDevice");

}

void CudaPipeline::resampleFloat3Map(float3* d_output, unsigned int outputWidth, unsigned int outputHeight,
                float3* d_input, unsigned int inputWidth, unsigned int inputHeight) {
	const dim3 gridSize((outputWidth + T_PER_BLOCK - 1)/T_PER_BLOCK, (outputHeight + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	resampleFloat3MapDevice<<<gridSize, blockSize>>>(d_colorMapResampledFloat4, d_colorMapFloat4, inputWidth, inputHeight, outputWidth, outputHeight);
    getLastCudaError("Failed: resampleFloat3MapDevice");
}


int CudaPipeline::init_cuda_device(int argc, char **argv) {
    // int dev = findCudaDevice(argc, (const char **)argv);
    checkCudaErrors(cudaMalloc(&dev_intrin, sizeof(rs2_intrinsics)));
    checkCudaErrors(cudaMemcpy(dev_intrin, &_intristics, sizeof(rs2_intrinsics), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc(&d_cameraSpaceFloat4, 4 * sizeof(float)*_intristics.width*_intristics.height));
    return dev;
}

void CudaPipeline::process_depth(rs2::depth_frame depth_frame) {
    // upload
    int count = _intristics.height * _intristics.width;
    int numBlocks = count / RS2_CUDA_THREADS_PER_BLOCK;

    uint16_t *depth_data = (uint16_t *)depth_frame.get_data();
    uint16_t *dev_depth = 0;
    float *dev_depth_f = 0;
    float *dev_points = 0;

    checkCudaErrors(cudaMalloc(&dev_points, count * sizeof(float) * 3));
    checkCudaErrors(cudaMalloc(&dev_depth_f, count * sizeof(float)));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&dev_depth), count * sizeof(uint16_t)));
    checkCudaErrors(cudaMemcpy(dev_depth, depth_data, count * sizeof(uint16_t), cudaMemcpyHostToDevice));

    std::cout << "image uploaded to gpu" << std::endl;

    // counverting depth image into points cloud and keeping points on gpu device
    kernel_deproject_depth_cuda<<<numBlocks, RS2_CUDA_THREADS_PER_BLOCK>>>(dev_points, dev_intrin, dev_depth,
        dev_depth_f, minDistance, maxDistance); 
    getLastCudaError("Failed: kernel_deproject_depth_cuda");

    std::cout << "Points cloud is computed" << std::endl;

    if (download_points){
        checkCudaErrors(cudaMemcpy(host_points, dev_points, count * sizeof(float) * 3, cudaMemcpyDeviceToHost));
    };

    std::cout << "Points cloud is downloaded to host" << std::endl;

    // ------------------ reconstruction begins -----------------------------
    float4x4 transformation = float4x4::identity();

    if (m_numIntegratedFrames > 0){
        // need to find rotation first

        render();

        float4x4 lastTransform = getLastRigidTransform();
        float4x4 deltaTransformEstimate = float4x4::identity();
        // applyCT(
        // float4* dInput, float4* dInputNormals, float4* dInputColors, 
        // float4* dModel, float4* dModelNormals, float4* dModelColors, 
        // const mat4f& lastTransform, const std::vector<unsigned int>& maxInnerIter, const std::vector<unsigned int>& maxOuterIter, 
        // const std::vector<float>& distThres, const std::vector<float>& normalThres, float condThres, float angleThres, 
        // const mat4f& deltaTransformEstimate, const std::vector<float>& earlyOutResidual, 
        // const mat4f& intrinsic, const DepthCameraData& depthCameraData,
                
        // g_CudaDepthSensor.getCameraSpacePositionsFloat4(), g_CudaDepthSensor.getNormalMapFloat4(), g_CudaDepthSensor.getColorMapFilteredFloat4(),
        // g_rayCast->getRayCastData().d_depth4, g_rayCast->getRayCastData().d_normals, g_rayCast->getRayCastData().d_colors,
        // lastTransform,
        // GlobalCameraTrackingState::getInstance().s_maxInnerIter, GlobalCameraTrackingState::getInstance().s_maxOuterIter,
        // GlobalCameraTrackingState::getInstance().s_distThres,	 GlobalCameraTrackingState::getInstance().s_normalThres,
        // 100.0f, 3.0f,
        // deltaTransformEstimate,
        // GlobalCameraTrackingState::getInstance().s_residualEarlyOut,
        // g_RGBDAdapter.getDepthIntrinsics(), g_CudaDepthSensor.getDepthCameraData(), 

        // Input
        float3** d_inputPoints;
        float3** d_inputNormals;
        float3** d_modelPoints;
        float3** d_modelNormals;
        // * 3 - x,y,z per point
        checkCudaErrors(cudaMalloc(&d_inputPoints, sizeof(float3) * _intristics.width * 
                _intristics.height * (cameraTrackingState.s_maxLevels + 1)));
        checkCudaErrors(cudaMalloc(&d_inputNormals, sizeof(float3) * _intristics.width * 
                _intristics.height * (cameraTrackingState.s_maxLevels + 1)));
        checkCudaErrors(cudaMalloc(&d_modelPoints, sizeof(float3) * _intristics.width * 
                _intristics.height * (cameraTrackingState.s_maxLevels + 1)));
        checkCudaErrors(cudaMalloc(&d_modelNormals, sizeof(float3) * _intristics.width * 
                _intristics.height * (cameraTrackingState.s_maxLevels + 1)));

        d_inputPoints[0] = dev_points; // index 0 if for original resolution
        d_inputNormals[0] = dInputNormals;
        // need to compute normals for original resolution
        computeNormals(d_inputNormals[0], dev_points, _intristics.width, _intristics.height);

        d_modelPoints[0] = m_rayCastData.d_depth3;
        d_modelNormals[0] = m_rayCastData.d_normals;

        unsigned int* m_imageWidth = new unsigned int[cameraTrackingState.s_maxLevels + 1];
        unsigned int* m_imageHeight = new unsigned int[cameraTrackingState.s_maxLevels + 1];    
        m_imageWidth[0] = _intristics.width;
        m_imageHeight[0] = _intristics.height;

        for (unsigned int i = 0; i < cameraTrackingState.s_maxLevels-1; i++) {
            resampleFloat3Map(d_inputPoints[i+1], m_imageWidth[i+1], m_imageHeight[i+1], d_inputPoints[i], m_imageWidth[i], 
                        m_imageHeight[i]);
            computeNormals(d_inputNormals[i+1], d_inputPoints[i+1], m_imageWidth[i+1], m_imageHeight[i+1]);
        
            resampleFloat3Map(d_modelPoints[i+1], m_imageWidth[i+1], m_imageHeight[i+1], d_modelPoints[i], m_imageWidth[i], 
                        m_imageHeight[i]);
            computeNormals(d_modelNormals[i+1], d_modelPoints[i+1], m_imageWidth[i+1], m_imageHeight[i+1]);
        }

        // Eigen::Matrix4f deltaTransform; deltaTransform = MatToEig(deltaTransformEstimate);
        float4x4 deltaTransform = deltaTransformEstimate;
        for (int level = cameraTrackingState.s_maxLevels-1; level>=0; level--) {	
            // deltaTransform = align(d_inputPoints[level], d_inputNormals[level], d_modelPoints[level], d_modelNormals[level], 
            //    deltaTransform, level, maxInnerIter[level], maxOuterIter[level], distThres[level], normalThres[level], condThres, 
            //    angleThres, earlyOutResidual[level], intrinsic, depthCameraData, errorLog);
            // --- align
            // align(float4* dInput, float4* dInputNormals, float4* dModel, float4* dModelNormals, Eigen::Matrix4f& deltaTransform, 
            //    unsigned int level, unsigned int maxInnerIter, unsigned maxOuterIter, float distThres, float normalThres, 
            //    float condThres, float angleThres, float earlyOut, const mat4f& intrinsic, const DepthCameraData& depthCameraData,
            //    ICPErrorLog* errorLog)
            float lastICPError = -1.0f;
            for(unsigned int i = 0; i<maxOuterIter[level]; i++)
            {
                float3 mean;
                float meanStDev;
                float nValidCorres;
        
                LinearSystemConfidence currConfWiReject;
                LinearSystemConfidence currConfNoReject;
                
                //standard correspondence search and alignment
                float levelFactor = pow(2.0f, (float)level);
                mean = make_float3(0.0f, 0.0f, 0.0f);
                meanStDev = 1.0f;
                const dim3 gridSize((imageWidth + T_PER_BLOCK - 1)/T_PER_BLOCK, (imageHeight + T_PER_BLOCK - 1)/T_PER_BLOCK);
                const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);
            
                projectiveCorrespondencesKernel<<<gridSize, blockSize>>>(
                    d_inputPoints[level], d_inputNormals[level], NULL, d_modelPoints[level], d_modelNormals[level], NULL, 
                    d_correspondence[level], d_correspondenceNormal[level], m_imageWidth[level], m_imageHeight[level],
                    cameraTrackingState.s_distThres[level], cameraTrackingState.s_normalThres[level], levelFactor, 
                    *deltaTransform, *m_rayCastParams.m_intrinsics, dev_intrin);
                getLastCudaError("Failed: projectiveCorrespondencesKernel");
                //----
        
                deltaTransform = computeBestRigidAlignment(d_inputPoints[level], d_inputNormals[level], mean, meanStDev, 
                            nValidCorres, deltaTransform, level, cameraTrackingState.s_maxInnerIter[level], condThres, 
                            angleThres, currConfWiReject);
                
                // computeBestRigidAlignment(float4* dInput, float4* dInputNormals, float3& mean, float meanStDev, 
                //      float nValidCorres, const Eigen::Matrix4f& globalDeltaTransform, unsigned int level, 
                //      unsigned int maxInnerIter, float condThres, float angleThres, LinearSystemConfidence& conf)
                // {
                //     Eigen::Matrix4f deltaTransform = globalDeltaTransform;
                
                //     for (unsigned int i = 0; i < maxInnerIter; i++)
                //     {
                //         conf.reset();
                
                //         Matrix6x7f system;
                
                //         m_CUDABuildLinearSystem->applyBL(dInput, d_correspondence[level], d_correspondenceNormal[level], mean, meanStDev, deltaTransform, m_imageWidth[level], m_imageHeight[level], level, system, conf);
                
                //         Matrix6x6f ATA = system.block(0, 0, 6, 6);
                //         Vector6f ATb = system.block(0, 6, 6, 1);
                
                //         if (ATA.isZero()) {
                //             return m_matrixTrackingLost;
                //         }
                
                //         Eigen::JacobiSVD<Matrix6x6f> SVD(ATA, Eigen::ComputeFullU | Eigen::ComputeFullV);
                //         Vector6f x = SVD.solve(ATb);
                
                //         //computing the matrix condition
                //         Vector6f evs = SVD.singularValues();
                //         conf.matrixCondition = evs[0]/evs[5];
                
                //         Eigen::Matrix4f t = delinearizeTransformation(x, Eigen::Vector3f(mean.x, mean.y, mean.z), meanStDev, level);
                //         if(t(0, 0) == -std::numeric_limits<float>::infinity())
                //         {
                //             conf.trackingLostTresh = true;
                //             return m_matrixTrackingLost;
                //         }
                
                //         deltaTransform = t*deltaTransform;
                //     }
                
                //     return deltaTransform;
                            


                if (std::abs(lastICPError - currConfWiReject.sumRegError) < earlyOutResidual[level]) {
                    //std::cout << lastICPError << " " <<  currConfWiReject.sumRegError << " ICP aboarted because no further convergence... " << i << std::endl;
                    break;
                }
                lastICPError = currConfWiReject.sumRegError;
            }
            // ---
            if(deltaTransform(0, 0) == -std::numeric_limits<float>::infinity()) {
                return EigToMat(m_matrixTrackingLost);
            }
        }
        // return lastTransform*EigToMat(deltaTransform);
        // ----
    }

    integrate(transformation, dev_depth_f);

    std::cout << "Image is integrated" << std::endl;

}

void CudaPipeline::setLastRigidTransform(const float4x4& lastRigidTransform) {
    m_hashParams.m_rigidTransform = lastRigidTransform;
    m_hashParams.m_rigidTransformInverse = m_hashParams.m_rigidTransform.getInverse();
}

