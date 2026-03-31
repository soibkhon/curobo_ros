#!/bin/bash

##
## Build script for curobo_ros Jetson Docker image (aarch64 / DEV mode)
## Target: Jetson Thor, Ampere GA10B, sm_87
##
## Usage:
##   ./build_docker_jetson.sh
##

set -e

echo "======================================"
echo "  curobo_ros Jetson Docker Build"
echo "======================================"
echo ""
echo "Target: Jetson Orin NX (aarch64, Ampere GA10B, sm_87)"
echo "Base:   nvcr.io/nvidia/l4t-pytorch:r36.4.0-pth2.3-py3 (Ubuntu 22.04 + ROS Humble)"
echo "Mode:   DEV"
echo ""
echo "Warning: This build will take 30-60 minutes and requires ~30 GB disk space."
echo ""
read -p "Press Enter to start, or Ctrl+C to cancel..."

IMAGE_TAG="curobo_ros:jetson-dev"

docker build \
    --build-arg TORCH_CUDA_ARCH_LIST="8.7" \
    -t "$IMAGE_TAG" \
    -f jetson.dockerfile \
    .

echo ""
echo "======================================"
echo "  Build Complete!"
echo "======================================"
echo "Image: $IMAGE_TAG"
echo ""
echo "Next steps:"
echo "  1. Ensure you have run 'vcs import' on the Jetson:"
echo "     cd ~/ros2_ws/src && vcs import < curobo_ros/my.repos"
echo ""
echo "  2. Start the container:"
echo "     bash start_docker_jetson.sh"
echo ""
