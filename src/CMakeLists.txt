file(GLOB gpu_source_files "cuda/*.cu")
file(GLOB app_source_files "./*.cpp")
add_library(${PROJECT_NAME}_gpu ${gpu_source_files})
set_property(TARGET ${PROJECT_NAME}_gpu PROPERTY CUDA_SEPARABLE_COMPILATION ON)

add_executable(${PROJECT_NAME} ${app_source_files})
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Link librealsense2 to the target
target_include_directories(${PROJECT_NAME} PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}
            ${realsense2_LIBRARY}
            ${CUDA_LIBRARIES}
            ${PROJECT_NAME}_gpu)

