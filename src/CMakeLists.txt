file(GLOB gpu_source_files "cuda/*.cu")
file(GLOB app_source_files "./*.cpp")
add_library(${PROJECT_NAME}_gpu ${gpu_source_files})
set_property(TARGET ${PROJECT_NAME}_gpu PROPERTY CUDA_SEPARABLE_COMPILATION ON)

add_executable(${PROJECT_NAME} ${app_source_files})

# Link librealsense2 to the target
target_include_directories(${PROJECT_NAME} PRIVATE ../external/cuda-samples/Common)
target_link_libraries(${PROJECT_NAME} ${realsense2_LIBRARY} ${PROJECT_NAME}_gpu)

target_include_directories(${PROJECT_NAME}_gpu PRIVATE ../external/cuda-samples/Common)

# testing other options
#set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
#target_link_libraries(${PROJECT_NAME} ${realsense2_LIBRARY} ${CUDART_LIBRARY} ${PROJECT_NAME}_gpu)
#target_compile_options(${PROJECT_NAME} PRIVATE -fpermissive)
#target_compile_options(${PROJECT_NAME}_gpu PRIVATE -forward-unknown-to-host-compiler -fpermissive)
