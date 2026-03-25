# Jetson Thor / aarch64 DEV Dockerfile
# Base: NVIDIA L4T PyTorch (JetPack 6.x container on JetPack 7.x host, Ubuntu 22.04, ROS Humble)
# Target GPU: Jetson Thor Blackwell GB10 → sm_100
#
# Host: Ubuntu 24.04 + JetPack 7.x  (CUDA forward-compat lets r36 containers run on JP7 host)
# Container: Ubuntu 22.04 + ROS Humble  (same as x86 dev image)
#
# NOTE: Verify the exact L4T image tag on NGC before building:
#   https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch
#   Use the latest r36.x.x tag available (JetPack 6.x, Ubuntu 22.04)
#
# Key differences from x86.dockerfile:
#   - No manual CUDA install (provided by JetPack/L4T)
#   - No PyTorch upgrade (L4T ships aarch64-optimized build)
#   - No hpcx (not available on Jetson) → standard openmpi
#   - NVTX symlinks use aarch64-linux paths
#   - libusb installed via apt (no amd64 .deb)
#   - make -j8 instead of -j32 (Jetson has fewer CPU cores)
#   - TORCH_CUDA_ARCH_LIST=10.0 for Jetson Thor Blackwell GB10

FROM nvcr.io/nvidia/l4t-pytorch:r36.4.0-pth2.3-py3 AS jetson_base

ARG TORCH_CUDA_ARCH_LIST="10.0"
ENV TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST

LABEL maintainer="Lucas Carpentier, Guillaume Dupoiron"

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
ARG ROS_DISTRO=humble

# GL libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libegl1-mesa-dev \
    libgl1-mesa-dev \
    libgles2-mesa-dev \
    libglvnd-dev \
    pkg-config && \
    rm -rf /var/lib/apt/lists/*

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

# Timezone + build tools
RUN apt-get update && apt-get install -y \
    tzdata \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/* \
    && ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime \
    && echo "America/New_York" > /etc/timezone \
    && dpkg-reconfigure -f noninteractive tzdata \
    && add-apt-repository -y ppa:git-core/ppa \
    && apt-get update && apt-get install -y \
    apt-utils \
    bash \
    build-essential \
    cmake \
    curl \
    git \
    git-lfs \
    glmark2 \
    iputils-ping \
    libeigen3-dev \
    libssl-dev \
    lsb-core \
    make \
    openssh-client \
    openssh-server \
    python3-ipdb \
    python3-pip \
    python3-tk \
    python3-wstool \
    sudo \
    terminator \
    unattended-upgrades \
    wget \
    && rm -rf /var/lib/apt/lists/*

# MPI: use standard openmpi (no hpcx on Jetson)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenmpi-dev \
    openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"

ARG CACHE_DATE=2024-07-19

RUN pip install "robometrics[evaluator] @ git+https://github.com/fishbotics/robometrics.git"

# cuRobo
RUN mkdir /pkgs && cd /pkgs && git clone https://github.com/NVlabs/curobo.git

WORKDIR /pkgs/curobo
RUN pip3 install .[dev,usd] --no-build-isolation

ENV PYOPENGL_PLATFORM=egl

RUN echo '{"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}' >> /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN apt-get update && \
    apt-get install -y libbenchmark-dev libgoogle-glog-dev libgtest-dev libsqlite3-dev && \
    cd /usr/src/googletest && cmake . && cmake --build . --target install && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /pkgs
# Install nvblox v0.0.5 (cuRobo-compatible)
RUN git clone https://github.com/nvidia-isaac/nvblox.git && \
    cd nvblox && git checkout v0.0.5

# Fix NVTX v2/v3 header conflicts — aarch64 path (not x86_64-linux)
RUN ln -sf /usr/include/nvtx3/nvToolsExt.h /usr/local/cuda/targets/aarch64-linux/include/nvToolsExt.h && \
    ln -sf /usr/include/nvtx3/nvToolsExtCuda.h /usr/local/cuda/targets/aarch64-linux/include/nvToolsExtCuda.h && \
    ln -sf /usr/include/nvtx3/nvToolsExtCudaRt.h /usr/local/cuda/targets/aarch64-linux/include/nvToolsExtCudaRt.h

# First cmake pass to download stdgpu (will fail at thrust version check — expected)
RUN cd /pkgs/nvblox/nvblox && mkdir -p build && cd build && \
    TORCH_PREFIX="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" && \
    cmake .. -DPRE_CXX11_ABI_LINKABLE=ON -DBUILD_TESTING=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="$(echo $TORCH_CUDA_ARCH_LIST | sed 's/\([0-9]\+\)\.\([0-9]\+\)/\1\2/g; s/ /;/g')" \
        -DCMAKE_PREFIX_PATH="$TORCH_PREFIX" 2>&1 || true

# Patch stdgpu for CUDA compatibility
# Fix 1: Findthrust.cmake can't parse newer CUDA thrust version strings
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

# Fix 2: stdgpu forward/construct_at/destroy_at conflict with cuda::std::
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

# Second cmake pass + build + install (j8: Jetson Thor has 12 cores, keep memory safe)
RUN cd /pkgs/nvblox/nvblox/build && \
    TORCH_PREFIX="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" && \
    cmake .. -DPRE_CXX11_ABI_LINKABLE=ON -DBUILD_TESTING=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="$(echo $TORCH_CUDA_ARCH_LIST | sed 's/\([0-9]\+\)\.\([0-9]\+\)/\1\2/g; s/ /;/g')" \
        -DCMAKE_PREFIX_PATH="$TORCH_PREFIX" && \
    make nvblox_lib nvblox_gpu_hash -j8 && \
    make install 2>&1 || true

RUN apt-get update && apt-get install -y libunwind-dev && rm -rf /var/lib/apt/lists/*
RUN apt remove python3-blinker -y || true

# Install cuRobo-compatible nvblox_torch from NVlabs
RUN cd /pkgs && git clone https://github.com/NVlabs/nvblox_torch.git
# Add cmake find_package for glog/gflags before nvblox
RUN sed -i 's|pkg_check_modules(glog REQUIRED libglog)|pkg_check_modules(glog REQUIRED libglog)\nfind_package(glog REQUIRED)\nfind_package(gflags REQUIRED)|' \
    /pkgs/nvblox_torch/src/nvblox_torch/cpp/CMakeLists.txt
RUN cd /pkgs/nvblox_torch && mkdir -p src/nvblox_torch/bin && \
    cd src/nvblox_torch/cpp && mkdir build && cd build && \
    TORCH_PREFIX="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" && \
    cmake -DCMAKE_PREFIX_PATH="$TORCH_PREFIX" -DCMAKE_CUDA_COMPILER=$(which nvcc) .. && \
    make -j8 && \
    cp libpy_nvblox.so ../../bin/ && \
    cd /pkgs/nvblox_torch && pip install -e . --no-build-isolation

# OpenCV
RUN git clone https://github.com/opencv/opencv.git /pkgs/opencv

WORKDIR /pkgs/opencv
RUN mkdir -p build

RUN python -m pip install \
    pyrealsense2 \
    transforms3d

RUN python -m pip install "robometrics[evaluator] @ git+https://github.com/fishbotics/robometrics.git"

# libusb via apt (no amd64 .deb on aarch64)
RUN apt-get update && apt-get install -y libusb-1.0-0 && rm -rf /var/lib/apt/lists/*

##### Installing ROS Humble ######

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    gnupg2 \
    lsb-release \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | apt-key add - \
    && sh -c 'echo "deb http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list'

RUN apt-get update && apt-get install -y \
    python3-rosdep \
    ros-humble-joint-state-publisher \
    ros-humble-joint-state-publisher-gui \
    ros-humble-nav2-msgs \
    ros-humble-moveit \
    ros-humble-realsense2-* \
    ros-humble-librealsense2* \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y software-properties-common && rm -rf /var/lib/apt/lists/*
RUN add-apt-repository universe

RUN apt update && apt install curl -y && \
    export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}') && \
    curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo $VERSION_CODENAME)_all.deb" && \
    apt install /tmp/ros2-apt-source.deb

RUN apt-get update && apt-get install -y \
    python3-argcomplete \
    python3-colcon-common-extensions \
    ros-humble-desktop \
    ros-humble-pcl-ros \
    ros-humble-rviz2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home/ros2_ws/src

# Note: curobo_msgs, curobo_rviz, curobo_ros, and xarm_description will be mounted as volumes in DEV mode

RUN sudo rosdep init && \
    rosdep update

# trajectory_preview
RUN git clone https://github.com/swri-robotics/trajectory_preview.git

RUN apt remove python3-blinker -y

# Open3D system deps
RUN apt-get update && apt-get install --no-install-recommends -y \
    libegl1 \
    libgl1 \
    libgomp1 \
    python3-pip \
    ros-humble-tf-transformations \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir --upgrade open3d
RUN python3 -m pip install --no-cache-dir --force-reinstall --no-deps \
    pandas scikit-learn pyarrow

WORKDIR /home/ros2_ws/src

RUN git clone -b humble https://github.com/Box-Robotics/ros2_numpy.git

# Build workspace
WORKDIR /home/ros2_ws
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
    colcon build"

RUN source /opt/ros/humble/setup.bash && \
    cd /home/ros2_ws && \
    . install/local_setup.bash

WORKDIR /home/ros2_ws

RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /home/ros2_ws/install/setup.bash" >> ~/.bashrc

ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

RUN apt-get update && apt-get install -y ros-humble-rmw-cyclonedds-cpp ros-humble-cyclonedds

ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

COPY branch_switch_entrypoint.sh /home/
