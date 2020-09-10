add_executable(${PROJECT_NAME} main.cpp)

file(GLOB gpu_source_files "cuda/*.cu")
add_library(${PROJECT_NAME}_gpu ${gpu_source_files})

# Link librealsense2 to the target
#target_include_directories(${PROJECT_NAME} PRIVATE ../external/cuda-samples/Common)

target_link_libraries(${PROJECT_NAME} ${realsense2_LIBRARY} ${cuda} ${PROJECT_NAME}_gpu)
target_include_directories(${PROJECT_NAME}_gpu PRIVATE ../external/cuda-samples/Common)