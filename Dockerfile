FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6 \
    glmark2 \
    build-essential \
    gdb \
    cmake \
    libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev \
    xorg-dev libglu1-mesa-dev \
    libeigen3-dev \
    libusb-1.0-0-dev \
    wget vim \
    libgl1-mesa-dev libglew-dev \
    git && \
    rm -rf /var/lib/apt/lists/*

RUN wget -q https://github.com/Kitware/CMake/releases/download/v3.18.2/cmake-3.18.2-Linux-x86_64.sh

RUN chmod +x cmake-3.18.2-Linux-x86_64.sh; ./cmake-3.18.2-Linux-x86_64.sh --skip-license

RUN git clone https://github.com/IntelRealSense/librealsense.git

#RUN cd librealsense; cp config/99-realsense-libusb.rules /etc/udev/rules.d/

RUN cd librealsense; mkdir build

RUN cd librealsense/build; cmake ../ -DFORCE_LIBUVC=true -DCMAKE_BUILD_TYPE=release -DBUILD_EXAMPLES=true -DBUILD_WITH_CUDA=true

RUN cd librealsense/build; make -j`grep -c ^processor /proc/cpuinfo` && make install

# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute
