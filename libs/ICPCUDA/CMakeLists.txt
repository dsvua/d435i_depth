file(GLOB gpu_source_files "/Cuda/*.cu" "Cuda/*.cpp" "Cuda/containers/*.cpp")
add_library(${PROJECT_NAME}_ICPCUDA SHARED ICPOdometry.cu ${gpu_source_files})
set_property(TARGET ${PROJECT_NAME}_ICPCUDA PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET ${PROJECT_NAME}_ICPCUDA PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories (${PROJECT_NAME}_ICPCUDA PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES})

target_link_libraries(${PROJECT_NAME}_ICPCUDA
            Sophus::Sophus
            Eigen3::Eigen)
