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

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
ARG ROS_DISTRO=humble
# add GL:
RUN apt-get update && apt-get install -y --no-install-recommends \
    libegl1-mesa-dev \
    libgl1-mesa-dev \
    libgles2-mesa-dev \
    libglvnd-dev \
    pkg-config && \
    rm -rf /var/lib/apt/lists/*

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute


# Set timezone info
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
    snapd \
    sudo \
    terminator \
    unattended-upgrades \
    wget \
    && rm -rf /var/lib/apt/lists/*


# push defaults to bashrc:
RUN apt-get update && apt-get install --reinstall -y \
    hwloc-nox \
    libmpich-dev \
    libmpich12 \
    mpich \
    && rm -rf /var/lib/apt/lists/*

# This is required to enable mpi lib access:
ENV PATH="${PATH}:/opt/hpcx/ompi/bin"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/hpcx/ompi/lib"

ARG TORCH_CUDA_ARCH_LIST="6.1 7.0+PTX"
ENV TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"

# Add cache date to avoid using cached layers older than this
ARG CACHE_DATE=2024-07-19

RUN pip install "robometrics[evaluator] @ git+https://github.com/fishbotics/robometrics.git"


# if you want to use a different version of curobo, create folder as docker/pkgs and put your
# version of curobo there. Then uncomment below line and comment the next line that clones from
# github

# COPY pkgs /pkgs
RUN mkdir /pkgs && cd /pkgs && git clone  https://github.com/NVlabs/curobo.git

WORKDIR /pkgs/curobo
RUN pip3 install .[dev,usd] --no-build-isolation

# Optionally install nvblox:

# we require this environment variable to  render images in unit test curobo/tests/nvblox_test.py

ENV PYOPENGL_PLATFORM=egl

# add this file to enable EGL for rendering

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
# Fix 1: Findthrust.cmake can't parse CUDA 12.8 thrust version
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

# Second cmake pass + build + install
RUN cd /pkgs/nvblox/nvblox/build && \
    TORCH_PREFIX="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" && \
    cmake .. -DPRE_CXX11_ABI_LINKABLE=ON -DBUILD_TESTING=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="$(echo $TORCH_CUDA_ARCH_LIST | sed 's/\([0-9]\+\)\.\([0-9]\+\)/\1\2/g; s/ /;/g')" \
        -DCMAKE_PREFIX_PATH="$TORCH_PREFIX" && \
    make nvblox_lib nvblox_gpu_hash -j32 && \
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
    make -j32 && \
    cp libpy_nvblox.so ../../bin/ && \
    cd /pkgs/nvblox_torch && pip install -e . --no-build-isolation

#################################################
# Cloner le dépôt OpenCV et les modules supplémentaires

RUN git clone https://github.com/opencv/opencv.git /pkgs/opencv

WORKDIR /pkgs/opencv
RUN mkdir -p build

RUN python -m pip install \
    pyrealsense2 \
    transforms3d

# install benchmarks:
RUN python -m pip install "robometrics[evaluator] @ git+https://github.com/fishbotics/robometrics.git"

RUN export LD_LIBRARY_PATH="/opt/hpcx/ucx/lib:$LD_LIBRARY_PATH"

ADD http://archive.ubuntu.com/ubuntu/pool/main/libu/libusb-1.0/libusb-1.0-0_1.0.25-1ubuntu2_amd64.deb /tmp/libusb-1.0-0_1.0.25-1ubuntu2_amd64.deb
RUN dpkg -i /tmp/libusb-1.0-0_1.0.25-1ubuntu2_amd64.deb


##### Installing ROS Humble ######

# Définir des arguments pour désactiver les invites interactives pendant l'installation
ARG DEBIAN_FRONTEND=noninteractive

# Ajouter les dépôts ROS 2
RUN apt-get update && apt-get install -y \
    gnupg2 \
    lsb-release \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | apt-key add - \
    && sh -c 'echo "deb http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list'

# Installer des dépendances générales
RUN apt-get update && apt-get install -y \
    python3-rosdep \
    ros-humble-joint-state-publisher \
    ros-humble-joint-state-publisher-gui \
    ros-humble-nav2-msgs \
    ros-humble-moveit \
    ros-humble-realsense2-* \
    ros-humble-librealsense2* \
    && rm -rf /var/lib/apt/lists/*

# Ajouter les sources de ROS 2 Humble
RUN apt-get update && apt-get install -y software-properties-common && rm -rf /var/lib/apt/lists/*
RUN add-apt-repository universe

RUN sudo apt update && sudo apt install curl -y && \ 
    export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}') && \
    curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo $VERSION_CODENAME)_all.deb" && \
    sudo apt install /tmp/ros2-apt-source.deb

# RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
# RUN sh -c 'echo "deb http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list'

# Mettre à jour et installer ROS 2 Humble
RUN apt-get update && apt-get install -y \
    python3-argcomplete \
    python3-colcon-common-extensions \
    ros-humble-desktop \
    ros-humble-pcl-ros \
    ros-humble-rviz2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home/ros2_ws/src

# Currently this package is not use and need a debug
# RUN git clone https://github.com/IntelRealSense/realsense-ros.git -b ros2-master

# Note: curobo_msgs, curobo_rviz, and curobo_ros will be mounted as volumes in DEV mode

RUN sudo rosdep init # "sudo rosdep init --include-eol-distros" && \
    rosdep update # "sudo rosdep update --include-eol-distros" 

# Setup for trajectory_preview
RUN git clone https://github.com/swri-robotics/trajectory_preview.git

# curobo_rviz and curobo_ros will be mounted as volumes in DEV mode

# Add tools for pcd_fuse
RUN apt remove python3-blinker -y

# Install Open3D system dependencies and pip
RUN apt-get update && apt-get install --no-install-recommends -y \
    libegl1 \
    libgl1 \
    libgomp1 \
    python3-pip \
    ros-humble-tf-transformations\
    && rm -rf /var/lib/apt/lists/*

# Install Open3D from the PyPI repositories
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir --upgrade open3d
RUN python3 -m pip install --no-cache-dir --force-reinstall --no-deps \
    pandas scikit-learn pyarrow
# # # Set the workspace directory
WORKDIR /home/ros2_ws/src

# # # Clone the repository directly into the src directory
RUN git clone -b humble https://github.com/Box-Robotics/ros2_numpy.git


# Build workspace
WORKDIR /home/ros2_ws
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
    colcon build"

RUN source /opt/ros/"$ROS_DISTRO"/setup.bash && \
    cd /home/ros2_ws && \
    . install/local_setup.bash

WORKDIR /home/ros2_ws

RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /home/ros2_ws/install/setup.bash" >> ~/.bashrc

# Fix error: "AttributeError: module 'cv2.dnn' has no attribute 'DictValue'"
#RUN sed -i '171d' /usr/local/lib/python3.10/dist-packages/cv2/typing/__init__.py

# update ucx path: https://github.com/openucx/ucc/issues/476
ENV LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:$LD_LIBRARY_PATH
COPY branch_switch_entrypoint.sh /home/

RUN  apt-get update && apt-get install -y ros-humble-rmw-cyclonedds-cpp ros-humble-cyclonedds

ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# Not needed anymore ENTRYPOINT [ "/home/branch_switch_entrypoint.sh" ]

