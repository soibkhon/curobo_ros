# Production Dockerfile - Optimized for using curobo_ros
# This image is smaller and meant for users who want to use curobo_ros without modifying it

FROM nvcr.io/nvidia/pytorch:24.10-py3 AS torch_cuda_base

# Upgrade CUDA toolkit to 12.8 for Blackwell (sm_100) support
RUN apt-get update && apt-get install -y --no-install-recommends wget gnupg2 && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb && \
    apt-get update && apt-get install -y --no-install-recommends cuda-toolkit-12-8 && \
    rm -rf /var/lib/apt/lists/*
ENV PATH=/usr/local/cuda-12.8/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH}
ENV CUDA_HOME=/usr/local/cuda-12.8

# Upgrade PyTorch to 2.6+ for Blackwell (sm_100) arch support
RUN pip3 install --no-cache-dir --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu126

LABEL maintainer="Lucas Carpentier, Guillaume Dupoiron"
LABEL description="Optimized curobo_ros image for production use"

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
ARG ROS_DISTRO=humble
ARG DEBIAN_FRONTEND=noninteractive

# Add GL libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libegl1-mesa-dev \
    libgl1-mesa-dev \
    libgles2-mesa-dev \
    libglvnd-dev \
    pkg-config && \
    rm -rf /var/lib/apt/lists/*

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

# Install essential packages only (minimal set)
RUN apt-get update && apt-get install -y \
    tzdata \
    software-properties-common \
    bash \
    build-essential \
    cmake \
    curl \
    git \
    iputils-ping \
    libssl-dev \
    lsb-core \
    python3-pip \
    sudo \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata

# Install MPI dependencies
RUN apt-get update && apt-get install --reinstall -y \
    hwloc-nox \
    libmpich-dev \
    libmpich12 \
    mpich \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="${PATH}:/opt/hpcx/ompi/bin"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/hpcx/ompi/lib"

ARG TORCH_CUDA_ARCH_LIST="8.0 8.6"
ENV TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"

# Install cuRobo (production version)
ARG CACHE_DATE=2024-07-19
RUN mkdir /pkgs && cd /pkgs && git clone https://github.com/NVlabs/curobo.git
WORKDIR /pkgs/curobo
RUN pip3 install . --no-build-isolation

# Install nvblox for collision checking
ENV PYOPENGL_PLATFORM=egl
RUN echo '{"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}' >> /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN apt-get update && \
    apt-get install -y libbenchmark-dev libgoogle-glog-dev libgtest-dev libsqlite3-dev && \
    cd /usr/src/googletest && cmake . && cmake --build . --target install && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /pkgs
# Install nvblox v0.0.5 (cuRobo-compatible) with CUDA 12.8 patches
RUN git clone https://github.com/nvidia-isaac/nvblox.git && \
    cd nvblox && git checkout v0.0.5

# Fix NVTX v2/v3 header conflicts for CUDA 12.8
RUN ln -sf /usr/include/nvtx3/nvToolsExt.h /usr/local/cuda/targets/x86_64-linux/include/nvToolsExt.h && \
    ln -sf /usr/include/nvtx3/nvToolsExtCuda.h /usr/local/cuda/targets/x86_64-linux/include/nvToolsExtCuda.h && \
    ln -sf /usr/include/nvtx3/nvToolsExtCudaRt.h /usr/local/cuda/targets/x86_64-linux/include/nvToolsExtCudaRt.h

# First cmake pass to download stdgpu (will fail at thrust version)
RUN cd /pkgs/nvblox/nvblox && mkdir -p build && cd build && \
    TORCH_PREFIX="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" && \
    cmake .. -DPRE_CXX11_ABI_LINKABLE=ON -DBUILD_TESTING=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="$(echo $TORCH_CUDA_ARCH_LIST | sed 's/\([0-9]\+\)\.\([0-9]\+\)/\1\2/g; s/ /;/g')" \
        -DCMAKE_PREFIX_PATH="$TORCH_PREFIX" 2>&1 || true

# Patch stdgpu for CUDA 12.8 compatibility
RUN printf 'find_path(THRUST_INCLUDE_DIR HINTS /usr/local/cuda/include NAMES thrust/version.h)\n\
if(THRUST_INCLUDE_DIR)\n\
    set(THRUST_VERSION "2.3.2")\n\
    if(NOT TARGET thrust::thrust)\n\
        add_library(thrust::thrust INTERFACE IMPORTED)\n\
        set_target_properties(thrust::thrust PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${THRUST_INCLUDE_DIR}")\n\
    endif()\n\
endif()\n\
include(FindPackageHandleStandardArgs)\n\
find_package_handle_standard_args(thrust REQUIRED_VARS THRUST_INCLUDE_DIR VERSION_VAR THRUST_VERSION)\n' \
    > /pkgs/nvblox/nvblox/build/_deps/ext_stdgpu-src/cmake/Findthrust.cmake

RUN cd /pkgs/nvblox/nvblox/build/_deps/ext_stdgpu-src/src/stdgpu && \
    sed -i 's/(forward</(std::forward</g' impl/memory_detail.h impl/unordered_base_detail.cuh \
        impl/deque_detail.cuh impl/unordered_set_detail.cuh impl/vector_detail.cuh \
        impl/unordered_map_detail.cuh impl/functional_detail.h functional.h && \
    sed -i 's/ forward</ std::forward</g' impl/memory_detail.h impl/unordered_base_detail.cuh \
        impl/deque_detail.cuh impl/unordered_set_detail.cuh impl/vector_detail.cuh \
        impl/unordered_map_detail.cuh impl/functional_detail.h functional.h && \
    sed -i 's/~forward</~std::forward</g' impl/functional_detail.h functional.h && \
    sed -i 's/std::std::forward/std::forward/g' impl/memory_detail.h impl/unordered_base_detail.cuh \
        impl/deque_detail.cuh impl/unordered_set_detail.cuh impl/vector_detail.cuh \
        impl/unordered_map_detail.cuh impl/functional_detail.h functional.h && \
    sed -i 's/    destroy_at(p);/    stdgpu::destroy_at(p);/' impl/memory_detail.h && \
    sed -i 's/    construct_at(p,/    stdgpu::construct_at(p,/' impl/memory_detail.h && \
    sed -i 's/            construct_at(\&t, _value);/            stdgpu::construct_at(\&t, _value);/' impl/memory_detail.h && \
    sed -i 's/        destroy_at(\&t);/        stdgpu::destroy_at(\&t);/' impl/memory_detail.h

# Second cmake pass + build + install
RUN cd /pkgs/nvblox/nvblox/build && \
    TORCH_PREFIX="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" && \
    cmake .. -DPRE_CXX11_ABI_LINKABLE=ON -DBUILD_TESTING=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="$(echo $TORCH_CUDA_ARCH_LIST | sed 's/\([0-9]\+\)\.\([0-9]\+\)/\1\2/g; s/ /;/g')" \
        -DCMAKE_PREFIX_PATH="$TORCH_PREFIX" && \
    make nvblox_lib nvblox_gpu_hash -j32 && \
    make install 2>&1 || true

RUN apt-get update && apt-get install -y libunwind-dev && rm -rf /var/lib/apt/lists/*

# Remove python3-blinker to avoid conflicts with pip packages
RUN apt remove python3-blinker -y || true

# Install cuRobo-compatible nvblox_torch from NVlabs
RUN cd /pkgs && git clone https://github.com/NVlabs/nvblox_torch.git
RUN sed -i 's|pkg_check_modules(glog REQUIRED libglog)|pkg_check_modules(glog REQUIRED libglog)\nfind_package(glog REQUIRED)\nfind_package(gflags REQUIRED)|' \
    /pkgs/nvblox_torch/src/nvblox_torch/cpp/CMakeLists.txt
RUN cd /pkgs/nvblox_torch && mkdir -p src/nvblox_torch/bin && \
    cd src/nvblox_torch/cpp && mkdir build && cd build && \
    TORCH_PREFIX="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" && \
    cmake -DCMAKE_PREFIX_PATH="$TORCH_PREFIX" -DCMAKE_CUDA_COMPILER=$(which nvcc) .. && \
    make -j32 && \
    cp libpy_nvblox.so ../../bin/ && \
    cd /pkgs/nvblox_torch && pip install -e . --no-build-isolation

# Install essential Python packages
RUN python -m pip install \
    pyrealsense2 \
    transforms3d \
    open3d

# Install ROS 2 Humble (minimal)
RUN apt-get update && apt-get install -y \
    gnupg2 \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

RUN sudo apt update && sudo apt install curl -y && \
    export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}') && \
    curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo $VERSION_CODENAME)_all.deb" && \
    sudo apt install /tmp/ros2-apt-source.deb

# Install minimal ROS packages
RUN apt-get update && apt-get install -y \
    python3-argcomplete \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    ros-humble-nav2-msgs\
    ros-humble-ros-base \
    ros-humble-rviz2 \
    ros-humble-joint-state-publisher \
    ros-humble-robot-state-publisher \
    ros-humble-tf-transformations \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-cyclonedds \
    && rm -rf /var/lib/apt/lists/*

# Initialize rosdep
RUN sudo rosdep init && rosdep update

# Install curobo_ros, curobo_msgs, and curobo_rviz from Git
# Using /home/curobo_ws to discourage users from modifying this workspace
WORKDIR /home/curobo_ws/src
RUN git clone https://github.com/Lab-CORO/curobo_ros.git --recurse-submodules && \
    git clone https://github.com/Lab-CORO/curobo_msgs.git && \
    git clone https://github.com/Lab-CORO/curobo_rviz.git

# Build the packages
WORKDIR /home/curobo_ws
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
    colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release"

# Setup environment - auto-source on every terminal/container startup
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /home/curobo_ws/install/setup.bash" >> ~/.bashrc

# Set Cyclone DDS as RMW
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# Update UCX path
ENV LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:$LD_LIBRARY_PATH

# Set default working directory
WORKDIR /home/ros2_ws

# Production image is ready - user will mount their workspace here
