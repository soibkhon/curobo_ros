#!/bin/bash

##
## Start script for curobo_ros Jetson DEV container (aarch64)
## Mounts: curobo_ros, curobo_rviz, curobo_msgs, xarm_description
##

set -e

IMAGE_TAG="curobo_ros:jetson-dev"
CONTAINER_NAME="curobo_jetson_dev"

echo "======================================"
echo "  curobo_ros Jetson Container Start"
echo "======================================"
echo ""

# Check image exists
if ! docker image inspect "$IMAGE_TAG" > /dev/null 2>&1; then
    echo "Error: Image '$IMAGE_TAG' not found!"
    echo "Build it first:  bash build_docker_jetson.sh"
    exit 1
fi

# Handle existing container
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container '$CONTAINER_NAME' already exists."
    echo ""
    echo "  1) Start existing container"
    echo "  2) Remove and create new container"
    echo "  3) Cancel"
    echo ""
    read -p "Choice: " container_choice

    case $container_choice in
        1)
            docker start "$CONTAINER_NAME"
            docker exec -it "$CONTAINER_NAME" bash
            exit 0
            ;;
        2)
            docker stop "$CONTAINER_NAME" 2>/dev/null || true
            docker rm "$CONTAINER_NAME"
            ;;
        *)
            echo "Cancelled."
            exit 0
            ;;
    esac
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUROBO_ROS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$(dirname "$CUROBO_ROS_DIR")"

echo "Mounting from: $SRC_DIR"
echo "  - curobo_ros"
echo "  - curobo_rviz"
echo "  - curobo_msgs"
echo "  - xarm_description"
echo ""

# CycloneDDS config
CYCLONEDDS_CONFIG="${SCRIPT_DIR}/cyclonedds.xml"
if [ ! -f "$CYCLONEDDS_CONFIG" ]; then
    echo "Warning: CycloneDDS config not found at $CYCLONEDDS_CONFIG"
    CYCLONEDDS_MOUNT=""
    CYCLONEDDS_ENV=""
else
    CYCLONEDDS_MOUNT="-v ${CYCLONEDDS_CONFIG}:/etc/cyclonedds.xml:ro"
    CYCLONEDDS_ENV="-e CYCLONEDDS_URI=file:///etc/cyclonedds.xml"
fi

read -p "Press Enter to start container, or Ctrl+C to cancel..."

xhost +local:docker 2>/dev/null || echo "Warning: Could not enable X11 forwarding"

docker run --name "$CONTAINER_NAME" -it \
    --privileged \
    --runtime nvidia \
    -e NVIDIA_DISABLE_REQUIRE=1 \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    $CYCLONEDDS_ENV \
    --device=/dev/:/dev/ \
    --network host \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    $CYCLONEDDS_MOUNT \
    -v ${SRC_DIR}/curobo_ros:/home/ros2_ws/src/curobo_ros \
    -v ${SRC_DIR}/curobo_rviz:/home/ros2_ws/src/curobo_rviz \
    -v ${SRC_DIR}/curobo_msgs:/home/ros2_ws/src/curobo_msgs \
    -v ${SRC_DIR}/xarm_description:/home/ros2_ws/src/xarm_description \
    "$IMAGE_TAG"

echo ""
echo "To open additional terminals:"
echo "  docker exec -it $CONTAINER_NAME bash"
echo ""
echo "To stop:"
echo "  docker stop $CONTAINER_NAME"
echo ""
