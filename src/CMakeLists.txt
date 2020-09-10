add_executable(${PROJECT_NAME} main.cpp)

# Link librealsense2 to the target
target_link_libraries(${PROJECT_NAME} ${realsense2_LIBRARY})

