#!/usr/bin/env python3
"""
Persistent GraspGen pipeline server with preloaded SAM3 + GraspGen models.

Eliminates the ~60s model-loading delay on every grasp request.
Start this once before launching the GUI.

Topics:
  Subscribe: /grasp_pipeline/request  (std_msgs/String) — object name to grasp
  Publish:   /grasp_pipeline/status   (std_msgs/String) — progress / result

Status messages:
  "ready"                 — server idle, accepting requests
  "busy: already running" — request rejected (pipeline active)
  "scanning: <obj>"       — capturing frame
  "segmenting: <obj>"     — SAM3 segmentation
  "generating grasps..."  — GraspGen inference
  "success: score=<f>"    — results saved to RESULTS_PATH
  "failed: <reason>"      — pipeline error

Camera topics expected (RealSense D435i via realsense2_camera node):
  /camera/camera/aligned_depth_to_color/image_raw   (uint16, mm)
  /camera/camera/color/image_raw                    (bgr8)
  /camera/camera/aligned_depth_to_color/camera_info

Results are written to:
  ~/contact_graspnet_pytorch/results/grasp_pipeline_result.npz
  ~/contact_graspnet_pytorch/results/graspgen_predictions.npz (visualization)

Usage:
  python3 grasp_pipeline_server.py
"""

import os
import sys
import subprocess
import time
import threading
import numpy as np
import cv2
import torch

# ---------------------------------------------------------------------------
# Paths — add SAM3 and GraspGen to sys.path
# ---------------------------------------------------------------------------
sys.path.insert(0, '/home/wheelchair_pc/sam3')
sys.path.insert(0, '/home/wheelchair_pc/grasp/GraspGen')

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from PIL import Image as PILImage

from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.utils.point_cloud_utils import filter_colliding_grasps
from grasp_gen.robot import get_gripper_info
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from cv_bridge import CvBridge

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GRASPGEN_CONFIG = (
    '/home/wheelchair_pc/grasp/GraspGen/models/checkpoints/'
    'graspgen_robotiq_2f_140.yml'
)
RESULTS_DIR = os.path.expanduser(
    '~/contact_graspnet_pytorch/results'
)
RESULTS_PATH = os.path.join(RESULTS_DIR, 'grasp_pipeline_result.npz')
VIS_PATH = os.path.join(RESULTS_DIR, 'graspgen_predictions.npz')

# Gripper depth correction (Robotiq 2F-140 training vs custom gripper)
ROBOTIQ_DEPTH = 0.195
CUSTOM_GRIPPER_DEPTH = 0.158
DEPTH_CORRECTION = ROBOTIQ_DEPTH - CUSTOM_GRIPPER_DEPTH  # ≈ 0.037 m

# GraspGen parameters
GRASP_THRESHOLD = 0.5
NUM_GRASPS = 200
TOPK_GRASPS = 100
MIN_GRASPS = 20
MAX_TRIES = 4
COLLISION_THRESHOLD = 0.008
MAX_SCENE_POINTS = 8192

# Point cloud depth range (meters)
Z_MIN, Z_MAX = 0.1, 1.5

# SAM3 detection confidence
SAM3_CONFIDENCE = 0.3


class GraspPipelineServer(Node):
    """ROS2 node: preloads models and serves grasp requests."""

    def __init__(self) -> None:
        super().__init__('grasp_pipeline_server')

        # ------------------------------------------------------------------
        # Publishers / subscribers
        # ------------------------------------------------------------------
        self._status_pub = self.create_publisher(String, '/grasp_pipeline/status', 10)
        self.create_subscription(
            String, '/grasp_pipeline/request', self._request_cb, 10
        )

        # Camera subscriptions (persistent, using ApproximateTimeSynchronizer)
        self._bridge = CvBridge()
        self._frame_event = threading.Event()
        self._latest_frame: dict | None = None
        self._capture_active = False

        depth_sub = message_filters.Subscriber(
            self, Image, '/camera/camera/aligned_depth_to_color/image_raw'
        )
        color_sub = message_filters.Subscriber(
            self, Image, '/camera/camera/color/image_raw'
        )
        info_sub = message_filters.Subscriber(
            self, CameraInfo, '/camera/camera/aligned_depth_to_color/camera_info'
        )
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [depth_sub, color_sub, info_sub], queue_size=10, slop=0.1
        )
        self._sync.registerCallback(self._camera_cb)

        self._busy = False

        # ------------------------------------------------------------------
        # Load models (blocks until done)
        # ------------------------------------------------------------------
        self._sam3_model = None
        self._graspgen_sampler = None
        self._graspgen_cfg = None

        self.get_logger().info('Loading SAM3 and GraspGen models...')
        self._publish_status('loading models...')
        self._load_models()
        self._publish_status('ready')
        self.get_logger().info('GraspPipelineServer ready — waiting for requests.')

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_models(self) -> None:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # SAM3
        t0 = time.time()
        sam3_root = os.path.join(os.path.dirname(sam3.__file__), '..')
        bpe_path = os.path.join(sam3_root, 'assets', 'bpe_simple_vocab_16e6.txt.gz')
        self._sam3_model = build_sam3_image_model(bpe_path=bpe_path)
        self.get_logger().info(f'SAM3 loaded in {time.time() - t0:.1f}s')

        # GraspGen
        t0 = time.time()
        self._graspgen_cfg = load_grasp_cfg(GRASPGEN_CONFIG)
        self._graspgen_sampler = GraspGenSampler(self._graspgen_cfg)
        self.get_logger().info(f'GraspGen loaded in {time.time() - t0:.1f}s')

    # ------------------------------------------------------------------
    # Camera callback
    # ------------------------------------------------------------------

    def _camera_cb(self, depth_msg, color_msg, info_msg) -> None:
        """Synchronized camera callback — stores latest frame when capturing."""
        if not self._capture_active:
            return

        depth_image = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        depth_meters = depth_image.astype(np.float64) * 0.001

        color_bgr = self._bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

        K = np.array([
            [info_msg.k[0], 0.0,           info_msg.k[2]],
            [0.0,           info_msg.k[4],  info_msg.k[5]],
            [0.0,           0.0,            1.0],
        ])

        self._latest_frame = {
            'depth': depth_meters, 'K': K, 'rgb': rgb, 'color_bgr': color_bgr,
        }
        self._frame_event.set()

    # ------------------------------------------------------------------
    # ROS callback
    # ------------------------------------------------------------------

    def _request_cb(self, msg: String) -> None:
        if self._busy:
            self._publish_status('busy: pipeline already running')
            return

        object_name = msg.data.strip()
        if not object_name:
            self._publish_status('failed: empty object name')
            return

        self._busy = True
        threading.Thread(
            target=self._run_pipeline, args=(object_name,), daemon=True
        ).start()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _publish_status(self, msg: str) -> None:
        m = String()
        m.data = msg
        self._status_pub.publish(m)
        self.get_logger().info(f'Status: {msg}')

    def _capture_frame(self, timeout_sec: float = 10.0) -> dict | None:
        """Block until a synchronized camera frame arrives."""
        self._latest_frame = None
        self._frame_event.clear()
        self._capture_active = True

        if not self._frame_event.wait(timeout=timeout_sec):
            self._capture_active = False
            return None

        self._capture_active = False
        return self._latest_frame

    @staticmethod
    def _depth_to_point_cloud(depth, K, mask=None):
        """Unproject depth image to 3D point cloud (camera frame)."""
        H, W = depth.shape
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        u, v = np.meshgrid(np.arange(W), np.arange(H))
        z = depth
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        pts = np.stack([x, y, z], axis=-1)  # (H, W, 3)
        valid = (z > Z_MIN) & (z < Z_MAX)
        if mask is not None:
            valid = valid & mask

        return pts[valid]

    @staticmethod
    def _select_grasp(grasps, grasp_conf):
        """Select best forward-facing, horizontally-closing grasp."""
        if len(grasps) == 0:
            return None, -1.0, -1

        candidates = []
        for i in range(len(grasps)):
            approach_dir = grasps[i, :3, 2]   # z-axis = approach
            closing_dir = grasps[i, :3, 0]    # x-axis = closing
            fwd_align = approach_dir[2]
            if fwd_align < 0.5:
                continue
            horiz_closing = abs(closing_dir[0])
            combined = grasp_conf[i] * fwd_align * horiz_closing
            candidates.append((combined, grasp_conf[i], grasps[i], i))

        if not candidates:
            return None, -1.0, -1

        candidates.sort(key=lambda c: c[0], reverse=True)
        _, score, grasp, idx = candidates[0]
        return grasp, score, idx

    # ------------------------------------------------------------------
    # Main pipeline (runs in a daemon thread)
    # ------------------------------------------------------------------

    def _run_pipeline(self, object_name: str) -> None:
        try:
            # 1. Capture frame
            self._publish_status(f'scanning: {object_name}')
            self.get_logger().info('Capturing frame from camera...')
            data = self._capture_frame(timeout_sec=10.0)
            if data is None:
                self._publish_status('failed: no camera frame received (timeout)')
                return

            depth = data['depth']
            K = data['K']
            rgb = data['rgb']
            color_bgr = data['color_bgr']

            valid_depth = depth[depth > 0]
            if len(valid_depth) > 0:
                self.get_logger().info(
                    f'Frame captured: depth [{valid_depth.min():.3f}, '
                    f'{valid_depth.max():.3f}] m'
                )

            # 2. SAM3 segmentation (reuse preloaded model)
            self._publish_status(f'segmenting: {object_name}')
            self.get_logger().info(f'Segmenting "{object_name}" with SAM3...')

            processor = Sam3Processor(self._sam3_model, confidence_threshold=SAM3_CONFIDENCE)
            pil_image = PILImage.fromarray(rgb)
            state = processor.set_image(pil_image)
            processor.reset_all_prompts(state)
            state = processor.set_text_prompt(state=state, prompt=object_name)

            if len(state['scores']) == 0:
                self._publish_status(f'failed: "{object_name}" not detected by SAM3')
                return

            scores = state['scores'].cpu().numpy()
            masks = state['masks'].cpu().numpy()  # (N, 1, H, W) boolean

            best_det_idx = int(np.argmax(scores))
            best_score = float(scores[best_det_idx])
            mask = masks[best_det_idx, 0]  # (H, W)
            self.get_logger().info(f'SAM3 detection score: {best_score:.3f}')

            # 3. Build point clouds
            object_pc = self._depth_to_point_cloud(depth, K, mask=mask)
            scene_pc = self._depth_to_point_cloud(depth, K, mask=~mask)
            full_pc = self._depth_to_point_cloud(depth, K)
            valid_full = (depth > Z_MIN) & (depth < Z_MAX)
            pc_colors = rgb[valid_full]

            if len(object_pc) < 100:
                self._publish_status(
                    f'failed: too few object points ({len(object_pc)} < 100)'
                )
                return

            self.get_logger().info(
                f'Object PC: {len(object_pc)} pts, '
                f'Scene PC: {len(scene_pc)} pts'
            )

            # 4. GraspGen inference (reuse preloaded sampler)
            self._publish_status('generating grasps...')
            self.get_logger().info('Running GraspGen inference...')

            object_pc_tensor = torch.from_numpy(object_pc).cuda().float()
            grasps, grasp_conf = GraspGenSampler.run_inference(
                object_pc_tensor,
                self._graspgen_sampler,
                grasp_threshold=GRASP_THRESHOLD,
                num_grasps=NUM_GRASPS,
                topk_num_grasps=TOPK_GRASPS,
                min_grasps=MIN_GRASPS,
                max_tries=MAX_TRIES,
            )

            if len(grasps) == 0:
                self._publish_status('failed: GraspGen produced no grasps')
                return

            grasps = grasps.cpu().numpy()
            grasp_conf = grasp_conf.cpu().numpy()
            grasps[:, 3, 3] = 1.0

            self.get_logger().info(
                f'Generated {len(grasps)} grasps, '
                f'scores [{grasp_conf.min():.3f}, {grasp_conf.max():.3f}]'
            )

            # Depth correction: shift grasps forward along approach axis
            if abs(DEPTH_CORRECTION) > 0.001:
                for i in range(len(grasps)):
                    grasps[i, :3, 3] += DEPTH_CORRECTION * grasps[i, :3, 2]

            # 5. Collision filtering
            if len(scene_pc) > 0:
                gripper_info = get_gripper_info(self._graspgen_cfg.data.gripper_name)
                scene_ds = (
                    scene_pc[np.random.choice(len(scene_pc), MAX_SCENE_POINTS, replace=False)]
                    if len(scene_pc) > MAX_SCENE_POINTS else scene_pc
                )
                cf_mask = filter_colliding_grasps(
                    scene_pc=scene_ds,
                    grasp_poses=grasps,
                    gripper_collision_mesh=gripper_info.collision_mesh,
                    collision_threshold=COLLISION_THRESHOLD,
                )
                n_before = len(grasps)
                grasps = grasps[cf_mask]
                grasp_conf = grasp_conf[cf_mask]
                self.get_logger().info(
                    f'Collision filter: {len(grasps)}/{n_before} grasps kept'
                )

            # 6. Select best grasp
            best_grasp, best_grasp_score, best_idx = self._select_grasp(grasps, grasp_conf)
            if best_grasp is None:
                self._publish_status('failed: no valid forward-facing grasps')
                return

            pos = best_grasp[:3, 3]
            quat = Rotation.from_matrix(best_grasp[:3, :3]).as_quat()
            self.get_logger().info(
                f'Best grasp: score={best_grasp_score:.4f}, '
                f'pos=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]'
            )

            # 7. Save results
            os.makedirs(RESULTS_DIR, exist_ok=True)
            np.savez(
                RESULTS_PATH,
                best_grasp=best_grasp,
                best_score=best_grasp_score,
                moveit_position=pos,
                moveit_quaternion=quat,
                pc_full=full_pc,
            )

            np.savez(
                VIS_PATH,
                pc_full=full_pc,
                pc_colors=pc_colors,
                object_pc=object_pc,
                grasps=grasps,
                grasp_conf=grasp_conf,
                selected_grasp=best_grasp,
                selected_idx=best_idx,
            )
            self.get_logger().info(f'Results saved to {RESULTS_PATH}')

            # Publish success BEFORE launching visualization so the robot
            # can proceed immediately (visualization runs independently)
            self._publish_status(f'success: score={best_grasp_score:.4f}')

            # Launch visualization non-blocking in a separate process
            try:
                from grasp_pipeline_graspgen import VISUALIZER_CODE as _VIS_CODE
                subprocess.Popen(
                    [sys.executable, '-c', _VIS_CODE, VIS_PATH],
                    env=os.environ.copy(),
                )
                self.get_logger().info('Grasp visualization launched (close window anytime)')
            except Exception as vis_exc:
                self.get_logger().warn(f'Could not launch visualization: {vis_exc}')

        except Exception as exc:
            import traceback
            self.get_logger().error(f'Pipeline error: {exc}\n{traceback.format_exc()}')
            self._publish_status(f'failed: {exc}')
        finally:
            self._busy = False
            # Re-publish ready so the client knows we're free
            self._publish_status('ready')


def main() -> None:
    rclpy.init()
    node = GraspPipelineServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
