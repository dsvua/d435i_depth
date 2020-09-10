// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2019 Intel Corporation. All Rights Reserved.

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <iostream>             // for cout
#include "cuda/depthToCuda.h"

// Hello RealSense example demonstrates the basics of connecting to a RealSense device
// and taking advantage of depth data
int main(int argc, char * argv[]) try
{

    // Create configuration for pipeline
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_DEPTH);
    // Create a Pipeline - this serves as a top-level API for streaming and processing frames
    rs2::pipeline p;
    rs2::pipeline_profile p_p;
    p_p = p.start(cfg);
    auto stream = p_p.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
    rs2_intrinsics intrinsics = stream.get_intrinsics(); // Calibration data

    // Capture 30 frames to give autoexposure, etc. a chance to settle
    for (auto i = 0; i < 30; ++i) p.wait_for_frames();

    // Get first frame to build initial map
    rs2::frameset frames = p.wait_for_frames();
    rs2::depth_frame depth = frames.get_depth_frame();
    // std::cout << "upload_depth_to_cuda " << RS2_CUDA_THREADS_PER_BLOCK << std::endl;
    upload_depth_to_cuda(argc, argv, depth, intrinsics);

    while (true)
    {
        // Block program until frames arrive
        rs2::frameset frames = p.wait_for_frames();

        // Try to get a frame of a depth image
        rs2::depth_frame depth = frames.get_depth_frame();

        // Get the depth frame's dimensions
        float width = depth.get_width();
        float height = depth.get_height();

        // Query the distance from the camera to the object in the center of the image
        float dist_to_center = depth.get_distance(width / 2, height / 2);

        // Print the distance
        std::cout << "The camera is facing an object " << dist_to_center << " meters away \r";
    }

    return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
