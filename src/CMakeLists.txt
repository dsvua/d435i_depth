file(GLOB gpu_source_files "cuda/*.cu")
file(GLOB app_source_files "./*.cpp")
add_library(${PROJECT_NAME}_gpu SHARED ${gpu_source_files})
set_property(TARGET ${PROJECT_NAME}_gpu PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET ${PROJECT_NAME}_gpu PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(${PROJECT_NAME}_gpu PRIVATE ${eigen3_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME}_gpu
            ${realsense2_LIBRARY}
            Eigen3::Eigen
            ${PROJECT_NAME}_ICPCUDA)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
add_executable(${PROJECT_NAME} ${app_source_files})
#set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
#set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)

# Link librealsense2 to the target
#target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES} ${eigen3_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME}
            ${realsense2_LIBRARY}
            ${PROJECT_NAME}_ICPCUDA
            ${PROJECT_NAME}_gpu)
