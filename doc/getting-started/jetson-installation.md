# Jetson Thor Installation Guide

This guide covers running curobo_ros on a **Jetson Thor** in DEV mode.

**Setup:**
- Host: Ubuntu 24.04 + JetPack 7.x
- Container: Ubuntu 24.04 + ROS 2 Jazzy
- GPU: Blackwell GB10 (sm_110, CUDA 13.0)

> **Why ROS Jazzy and not Humble?**
> Jetson Thor requires CUDA 13.0 + Blackwell SM 11.0, which needs GLIBC 2.38 (Ubuntu 24.04).
> ROS Humble only runs on Ubuntu 22.04 (GLIBC 2.35) — these are binary-incompatible.
> ROS Jazzy is the Ubuntu 24.04 equivalent of Humble: same architecture, same package ecosystem.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| **Jetson Thor** with JetPack 7.x | Host OS: Ubuntu 24.04 |
| **~40 GB free disk** | Build takes more space on aarch64 |
| **Docker** with NVIDIA runtime | Comes with JetPack — verify below |
| **NGC account + API key** | Needed to pull the L4T base image |

---

## 1. Verify Docker and NVIDIA Runtime

JetPack includes Docker and the NVIDIA container runtime. Confirm the NVIDIA runtime is set as the default:

```bash
docker info | grep -i runtime
```

If `nvidia` is not listed as the default runtime, add it:

```bash
sudo nano /etc/docker/daemon.json
```

```json
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
```

```bash
sudo systemctl restart docker
```

---

## 2. Log in to NGC (NVIDIA Container Registry)

The L4T base image requires authentication:

```bash
docker login nvcr.io
# Username: $oauthtoken
# Password: <your NGC API key>
```

Get your API key at [ngc.nvidia.com/setup/api-key](https://ngc.nvidia.com/setup/api-key).

---

## 3. Verify the Base Image Tag

Check that the base image is available:

```bash
docker pull nvcr.io/nvidia/pytorch:25.08-py3
```

If this tag does not exist, browse available tags at [NGC pytorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags) and update the `FROM` line in `docker/jetson.dockerfile` accordingly.

---

## 4. Clone the Repository

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

git clone https://github.com/Lab-CORO/curobo_ros.git --recurse-submodules
```

### Import Dependencies

```bash
# Install vcstool if not already installed
sudo apt install python3-vcstool

cd ~/ros2_ws/src
vcs import < curobo_ros/my.repos
```

Your workspace should look like:

```
~/ros2_ws/
└── src/
    ├── curobo_ros/
    ├── curobo_msgs/
    ├── curobo_rviz/
    └── xarm_description/   ← if using xArm7
```

---

## 5. Set Up CycloneDDS (for multi-machine ROS 2)

If you need the Jetson to communicate with another machine over WiFi (e.g. a wheelchair PC), create the CycloneDDS config file:

```bash
nano ~/ros2_ws/src/curobo_ros/docker/cyclonedds.xml
```

```xml
<CycloneDDS>
  <Domain>
    <General>
      <NetworkInterfaceAddress>wlan0</NetworkInterfaceAddress>
    </General>
    <Discovery>
      <Peers>
        <Peer Address="192.168.1.5"/>
      </Peers>
    </Discovery>
    <Tracing>
      <Verbosity>severe</Verbosity>
    </Tracing>
  </Domain>
</CycloneDDS>
```

Replace `wlan0` with your actual WiFi interface (`ip link` to check) and `192.168.1.5` with the IP of your remote machine.

If you are running standalone (no remote machine), skip this step — the start script will warn but continue without it.

---

## 6. Build the Docker Image

**This step takes 30-60+ minutes.** The nvblox compilation is the slowest part.

```bash
cd ~/ros2_ws/src/curobo_ros/docker
bash build_docker_jetson.sh
```

**What happens during the build:**
1. Pulls L4T PyTorch base (Ubuntu 22.04, CUDA, PyTorch for aarch64)
2. Installs build tools and OpenMPI
3. Builds cuRobo from source (compiled for sm_100)
4. Builds nvblox v0.0.5 with CUDA compatibility patches
5. Builds nvblox_torch
6. Installs ROS 2 Humble
7. Builds the ROS workspace

**Result:** image named `curobo_ros:jetson-dev`

---

## 7. Start the Container

```bash
cd ~/ros2_ws/src/curobo_ros/docker
bash start_docker_jetson.sh
```

This mounts the following packages from your host into the container for live editing:

| Host path | Container path |
|---|---|
| `~/ros2_ws/src/curobo_ros` | `/home/ros2_ws/src/curobo_ros` |
| `~/ros2_ws/src/curobo_rviz` | `/home/ros2_ws/src/curobo_rviz` |
| `~/ros2_ws/src/curobo_msgs` | `/home/ros2_ws/src/curobo_msgs` |
| `~/ros2_ws/src/xarm_description` | `/home/ros2_ws/src/xarm_description` |

You are now inside the container.

---

## 8. Build the ROS Workspace (First Run Only)

The mounted packages need to be built inside the container:

```bash
cd /home/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

**Build time:** ~2-5 minutes.

Verify:

```bash
ros2 pkg list | grep curobo
```

Expected output:
```
curobo_msgs
curobo_ros
curobo_rviz
```

---

## 9. Open Additional Terminals

```bash
# On the Jetson host, in a new terminal:
docker exec -it curobo_jetson_dev bash
source /home/ros2_ws/install/setup.bash
```

---

## 10. Launch the System

```bash
ros2 launch curobo_ros unified_planner.launch.py
```

First launch triggers a GPU warmup (~30-60 seconds to compile CUDA kernels).

---

## Restarting the Container

After the first run, you don't need to recreate the container:

```bash
# Start the existing container
docker start curobo_jetson_dev
docker exec -it curobo_jetson_dev bash
```

---

## Troubleshooting

**Build fails pulling L4T image:**
Check the tag exists on NGC and you are logged in (`docker login nvcr.io`).

**`--runtime nvidia` not found:**
Make sure `nvidia-container-runtime` is installed and set as the default runtime (step 1).

**nvblox build OOM / killed:**
Reduce parallel jobs. Edit `jetson.dockerfile` and change `-j8` to `-j4`, then rebuild.

**RViz does not appear:**
Run `xhost +local:docker` on the host before starting the container.

**CycloneDDS errors in logs:**
Normal if no remote peer is present. Set `<Verbosity>severe</Verbosity>` in your CycloneDDS config to suppress warnings.

---

## See Also

- [x86 Installation](installation.md) — standard desktop/laptop setup
- [Docker Workflow Guide](docker-workflow.md) — DEV vs PROD modes
- [Troubleshooting](troubleshooting.md) — common CUDA/Docker issues
