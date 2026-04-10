#!/usr/bin/env python3
"""
Integrated RealSense + SAM3 + GraspGen Grasp Pipeline

Captures a scene from RealSense D435i via ROS2 topics, segments a target object
using SAM3, generates 6-DoF grasps using NVIDIA GraspGen (diffusion-based), selects
the optimal grasp, visualizes it with Open3D, and outputs the grasp pose for MoveIt.

Adaptation for custom gripper (width=85mm, depth=178mm):
  - Uses Robotiq 2F-140 pretrained model (closest match: depth=195mm, width=136mm)
  - Applies depth correction: shifts grasps forward by (195-178)=17mm along approach
  - pose_repr=mlp means the diffusion model doesn't use gripper geometry during inference,
    so no retraining is needed

GraspGen grasp convention (same as ContactGraspNet):
  - z-axis = approach direction (toward object)
  - x-axis = closing/contact direction (fingers close along this)
  - y-axis = lateral
  - origin = gripper base/mount point

Usage:
    python grasp_pipeline_graspgen.py --object "bottle"
    python grasp_pipeline_graspgen.py --object "cup" --num_grasps 300
    python grasp_pipeline_graspgen.py --object "bottle" --np_path realsense_capture.npy
"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

# -- SAM3 --
sys.path.insert(0, '/home/wheelchair_pc/sam3')
import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# -- GraspGen --
sys.path.insert(0, '/home/wheelchair_pc/grasp/GraspGen')
from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.utils.point_cloud_utils import filter_colliding_grasps
from grasp_gen.robot import get_gripper_info

# Gripper parameters
ROBOTIQ_DEPTH = 0.195      # Robotiq 2F-140 depth (model was trained with this)
CUSTOM_GRIPPER_DEPTH = 0.158  # Your gripper depth (end-effector to grasp point)
CUSTOM_GRIPPER_WIDTH = 0.085  # Your gripper max opening
DEPTH_CORRECTION = ROBOTIQ_DEPTH - CUSTOM_GRIPPER_DEPTH  # 0.027m forward shift

GRASPGEN_CONFIG = '/home/wheelchair_pc/grasp/GraspGen/models/checkpoints/graspgen_robotiq_2f_140.yml'


def capture_from_ros_topics(timeout_sec=10.0):
    """Capture aligned RGB + depth from ROS2 topics and return data dict.

    Subscribes to:
      /camera/camera/aligned_depth_to_color/image_raw   (uint16, mm)
      /camera/camera/color/image_raw                    (bgr8)
      /camera/camera/aligned_depth_to_color/camera_info (intrinsics)

    Returns dict with keys: depth (float64, meters), K (3x3), rgb (uint8), color_bgr (uint8)
    """
    import rclpy
    from rclpy.node import Node
    import message_filters
    from sensor_msgs.msg import Image, CameraInfo
    from cv_bridge import CvBridge

    print("Initializing ROS2 node to capture from topics...")

    rclpy.init()
    captured = [None]

    class CaptureNode(Node):
        def __init__(self):
            super().__init__('grasp_pipeline_capture')
            self.bridge = CvBridge()
            self.done = False

            depth_sub = message_filters.Subscriber(
                self, Image,
                '/camera/camera/aligned_depth_to_color/image_raw'
            )
            color_sub = message_filters.Subscriber(
                self, Image,
                '/camera/camera/color/image_raw'
            )
            info_sub = message_filters.Subscriber(
                self, CameraInfo,
                '/camera/camera/aligned_depth_to_color/camera_info'
            )

            self.sync = message_filters.ApproximateTimeSynchronizer(
                [depth_sub, color_sub, info_sub],
                queue_size=10,
                slop=0.1
            )
            self.sync.registerCallback(self.callback)

        def callback(self, depth_msg, color_msg, info_msg):
            if self.done:
                return

            # Decode depth: uint16 (mm) -> float64 (meters)
            depth_image = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding='passthrough'
            )
            depth_meters = depth_image.astype(np.float64) * 0.001

            # Decode color
            color_bgr = self.bridge.imgmsg_to_cv2(
                color_msg, desired_encoding='bgr8'
            )
            rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

            # Intrinsics from CameraInfo (K is row-major 3x3)
            K = np.array([
                [info_msg.k[0], 0.0,           info_msg.k[2]],
                [0.0,           info_msg.k[4],  info_msg.k[5]],
                [0.0,           0.0,            1.0          ],
            ])

            captured[0] = {
                'depth': depth_meters,
                'K': K,
                'rgb': rgb,
                'color_bgr': color_bgr,
            }
            self.done = True

    import time
    node = CaptureNode()
    start = time.time()
    while captured[0] is None and (time.time() - start) < timeout_sec:
        rclpy.spin_once(node, timeout_sec=0.1)

    node.destroy_node()
    rclpy.shutdown()

    if captured[0] is None:
        print(f"ERROR: No frames received within {timeout_sec}s timeout.")
        return None

    data = captured[0]
    valid_depth = data['depth'][data['depth'] > 0]
    if len(valid_depth) > 0:
        print(f"Captured frame: {data['color_bgr'].shape}, depth range: "
              f"[{valid_depth.min():.3f}, {valid_depth.max():.3f}] m")
    else:
        print("WARNING: No valid depth data captured!")

    cv2.imwrite('realsense_preview.png', data['color_bgr'])
    print("Saved preview to realsense_preview.png")

    return data


def segment_with_sam3(rgb, object_prompt, confidence_threshold=0.3):
    """
    Segment a target object from an RGB image using SAM3.

    Returns:
        segmap: HxW uint8 array (0=background, 1=target object)
        mask: HxW boolean mask of the target object
        best_score: confidence score of the best detection
    """
    print(f"\nLoading SAM3 model...")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    bpe_path = os.path.join(sam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz")
    model = build_sam3_image_model(bpe_path=bpe_path)

    processor = Sam3Processor(model, confidence_threshold=confidence_threshold)

    pil_image = Image.fromarray(rgb)
    state = processor.set_image(pil_image)

    print(f"Segmenting object: '{object_prompt}' ...")
    processor.reset_all_prompts(state)
    state = processor.set_text_prompt(state=state, prompt=object_prompt)

    if len(state["scores"]) == 0:
        print(f"No '{object_prompt}' detected! Try adjusting the scene or confidence threshold.")
        return None, None, 0.0

    scores = state["scores"].cpu().numpy()
    masks = state["masks"].cpu().numpy()  # (N, 1, H, W) boolean
    boxes = state["boxes"].cpu().numpy()  # (N, 4) xyxy

    best_idx = np.argmax(scores)
    best_score = scores[best_idx]
    best_mask = masks[best_idx, 0]  # (H, W)
    best_box = boxes[best_idx]

    print(f"Found {len(scores)} detection(s). Best score: {best_score:.3f}")
    print(f"  Bounding box (xyxy): [{best_box[0]:.0f}, {best_box[1]:.0f}, "
          f"{best_box[2]:.0f}, {best_box[3]:.0f}]")

    segmap = np.zeros(rgb.shape[:2], dtype=np.uint8)
    segmap[best_mask] = 1

    del model, processor
    torch.cuda.empty_cache()

    return segmap, best_mask, best_score


def depth_to_point_cloud(depth, K, mask=None, z_range=(0.1, 1.5)):
    """Convert depth image to point cloud using camera intrinsics.

    Args:
        depth: HxW depth image in meters
        K: 3x3 camera intrinsic matrix
        mask: HxW boolean mask (only keep points within mask)
        z_range: (min, max) depth range to keep

    Returns:
        points: Nx3 point cloud in camera optical frame
    """
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Create pixel coordinate grid
    u = np.arange(W)
    v = np.arange(H)
    u, v = np.meshgrid(u, v)

    # Unproject to 3D
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.stack([x, y, z], axis=-1)  # (H, W, 3)

    # Apply depth range filter
    valid = (z > z_range[0]) & (z < z_range[1])

    # Apply mask
    if mask is not None:
        valid = valid & mask

    points = points[valid]
    pixel_coords = np.stack([v[valid], u[valid]], axis=-1)  # row, col

    return points, pixel_coords


def run_graspgen(object_pc, scene_pc=None, num_grasps=200, grasp_threshold=0.5,
                 topk_num_grasps=100, min_grasps=20, max_tries=4,
                 collision_check=True, collision_threshold=0.005,
                 max_scene_points=8192):
    """
    Run GraspGen inference on an object point cloud with optional collision filtering.

    Args:
        object_pc: Nx3 numpy array of object points in camera frame
        scene_pc: Mx3 numpy array of scene points (excluding object) for collision checking
        num_grasps: number of grasps to generate per try
        grasp_threshold: discriminator confidence threshold
        topk_num_grasps: max number of top grasps to return
        min_grasps: minimum grasps before retrying
        max_tries: max inference attempts
        collision_check: whether to filter colliding grasps
        collision_threshold: distance threshold for collision (meters)
        max_scene_points: max scene points for collision check (speed optimization)

    Returns:
        grasps: Mx4x4 numpy array of grasp poses (depth-corrected, collision-free)
        grasp_conf: M confidence scores
    """
    print("\nLoading GraspGen model...")
    cfg = load_grasp_cfg(GRASPGEN_CONFIG)
    sampler = GraspGenSampler(cfg)

    print(f"Running GraspGen inference ({num_grasps} grasps, threshold={grasp_threshold})...")
    object_pc_tensor = torch.from_numpy(object_pc).cuda().float()

    grasps, grasp_conf = GraspGenSampler.run_inference(
        object_pc_tensor,
        sampler,
        grasp_threshold=grasp_threshold,
        num_grasps=num_grasps,
        topk_num_grasps=topk_num_grasps,
        min_grasps=min_grasps,
        max_tries=max_tries,
    )

    if len(grasps) == 0:
        print("No grasps generated!")
        return np.array([]), np.array([])

    grasps = grasps.cpu().numpy()
    grasp_conf = grasp_conf.cpu().numpy()

    # Ensure homogeneous coordinate
    grasps[:, 3, 3] = 1

    print(f"Generated {len(grasps)} grasps, scores: [{grasp_conf.min():.3f}, {grasp_conf.max():.3f}]")

    # Apply depth correction: shift each grasp forward along its approach direction
    # The model was trained with depth=0.195m (Robotiq 2F-140), but our gripper has depth=0.178m.
    # This means the gripper base needs to be 0.017m closer to the object.
    if abs(DEPTH_CORRECTION) > 0.001:
        print(f"Applying depth correction: {DEPTH_CORRECTION*1000:.1f}mm forward along approach")
        for i in range(len(grasps)):
            approach_dir = grasps[i, :3, 2]  # z-axis = approach
            grasps[i, :3, 3] += DEPTH_CORRECTION * approach_dir

    # Collision filtering against scene point cloud
    if collision_check and scene_pc is not None and len(scene_pc) > 0:
        print(f"\nRunning collision check against {len(scene_pc)} scene points...")
        gripper_info = get_gripper_info(cfg.data.gripper_name)
        gripper_collision_mesh = gripper_info.collision_mesh

        # Downsample scene for speed
        if len(scene_pc) > max_scene_points:
            indices = np.random.choice(len(scene_pc), max_scene_points, replace=False)
            scene_pc_ds = scene_pc[indices]
            print(f"  Downsampled scene: {len(scene_pc)} -> {len(scene_pc_ds)} points")
        else:
            scene_pc_ds = scene_pc

        collision_free_mask = filter_colliding_grasps(
            scene_pc=scene_pc_ds,
            grasp_poses=grasps,
            gripper_collision_mesh=gripper_collision_mesh,
            collision_threshold=collision_threshold,
        )

        n_before = len(grasps)
        grasps = grasps[collision_free_mask]
        grasp_conf = grasp_conf[collision_free_mask]
        print(f"  Collision-free: {len(grasps)}/{n_before} grasps")

    # Clean up GPU memory
    del sampler
    torch.cuda.empty_cache()

    return grasps, grasp_conf


def select_optimal_grasp(grasps, grasp_conf):
    """
    Select the best grasp for an eye-in-hand setup.

    In camera_color_optical_frame: x=right, y=down, z=forward (depth).

    Requirements:
      - Forward approach only: approach along camera z (the arm extends forward)
      - Gripper flat to ground: closing direction along camera x (horizontal)
      - No 45-degree rotated grasps

    Returns:
        best_grasp, best_score, best_idx
    """
    if len(grasps) == 0:
        return None, -1.0, -1

    candidates = []
    for i in range(len(grasps)):
        score = grasp_conf[i]
        grasp = grasps[i]
        approach_dir = grasp[:3, 2]  # z-axis = approach
        closing_dir = grasp[:3, 0]  # x-axis = closing
        lateral_dir = grasp[:3, 1]  # y-axis = lateral (gripper body direction)

        # Forward approach only: approach must be along camera z
        fwd_align = approach_dir[2]  # positive z = forward
        if fwd_align < 0.5:
            continue

        # Closing direction should be horizontal (along camera x)
        horiz_closing = abs(closing_dir[0])

        combined = score * fwd_align * horiz_closing

        candidates.append((combined, score, horiz_closing, fwd_align,
                          grasp, i, approach_dir.copy(), closing_dir.copy(), lateral_dir.copy()))

    candidates.sort(key=lambda c: c[0], reverse=True)

    # Print top candidates
    print(f"\n  Forward candidates: {len(candidates)} / {len(grasps)} total grasps")
    for rank, (comb, sc, hc, fa, g, idx, app, clo, lat) in enumerate(candidates[:8]):
        print(f"    #{rank}: score={sc:.4f}  fwd={fa:.3f}  "
              f"horiz_close={hc:.3f}  "
              f"approach=[{app[0]:.3f},{app[1]:.3f},{app[2]:.3f}]  "
              f"closing=[{clo[0]:.3f},{clo[1]:.3f},{clo[2]:.3f}]  "
              f"lateral=[{lat[0]:.3f},{lat[1]:.3f},{lat[2]:.3f}]")

    if candidates:
        best_combined, best_score, _, _, best_grasp, best_idx, app, clo, lat = candidates[0]

        print(f"  >>> Selected: score={best_score:.4f}, "
              f"approach=[{app[0]:.3f},{app[1]:.3f},{app[2]:.3f}], "
              f"closing=[{clo[0]:.3f},{clo[1]:.3f},{clo[2]:.3f}]")
        return best_grasp, best_score, best_idx
    else:
        print("  WARNING: No forward grasps found!")
        return None, -1.0, -1


def grasp_to_moveit_pose(grasp_4x4):
    """Convert a 4x4 grasp pose to MoveIt-compatible position + quaternion."""
    position = grasp_4x4[:3, 3]
    rotation_matrix = grasp_4x4[:3, :3]
    quat = Rotation.from_matrix(rotation_matrix).as_quat()  # [x, y, z, w]

    return {
        'position': {'x': float(position[0]),
                     'y': float(position[1]),
                     'z': float(position[2])},
        'orientation': {'x': float(quat[0]),
                        'y': float(quat[1]),
                        'z': float(quat[2]),
                        'w': float(quat[3])},
        'frame_id': 'camera_color_optical_frame',
    }


def print_moveit_pose(pose_dict, grasp_score):
    """Pretty-print the MoveIt-ready grasp pose."""
    p = pose_dict['position']
    o = pose_dict['orientation']

    print("\n" + "=" * 60)
    print("OPTIMAL GRASP POSE (MoveIt-ready, camera frame)")
    print("=" * 60)
    print(f"  Frame:       {pose_dict['frame_id']}")
    print(f"  Score:       {grasp_score:.4f}")
    print(f"  Position:    x={p['x']:.4f}, y={p['y']:.4f}, z={p['z']:.4f}")
    print(f"  Orientation: x={o['x']:.4f}, y={o['y']:.4f}, z={o['z']:.4f}, w={o['w']:.4f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="RealSense + SAM3 + GraspGen grasp pipeline"
    )
    parser.add_argument('--object', type=str, default='bottle',
                        help='Object to segment (SAM3 text prompt)')
    parser.add_argument('--num_grasps', type=int, default=200,
                        help='Number of grasps to generate per attempt')
    parser.add_argument('--grasp_threshold', type=float, default=0.5,
                        help='GraspGen discriminator confidence threshold')
    parser.add_argument('--topk', type=int, default=100,
                        help='Max number of top grasps to keep')
    parser.add_argument('--z_range', type=float, nargs=2, default=[0.1, 1.5],
                        help='Min/max depth range in meters')
    parser.add_argument('--confidence', type=float, default=0.3,
                        help='SAM3 detection confidence threshold')
    parser.add_argument('--np_path', type=str, default=None,
                        help='Load from .npy file instead of live RealSense capture')
    parser.add_argument('--save_capture', action='store_true',
                        help='Save captured data to .npy file')
    parser.add_argument('--no_collision', action='store_true',
                        help='Disable collision checking against scene')
    parser.add_argument('--collision_threshold', type=float, default=0.008,
                        help='Distance threshold for collision detection (meters)')
    parser.add_argument('--timeout', type=float, default=10.0,
                        help='Timeout (seconds) waiting for ROS2 camera topics')

    args = parser.parse_args()

    # ---------------------------------------------------------------
    # 1) Capture or load RGB-D data
    # ---------------------------------------------------------------
    if args.np_path is not None:
        print(f"Loading data from {args.np_path}...")
        data = np.load(args.np_path, allow_pickle=True).item()
        depth = data['depth']
        K = data['K']
        rgb = data['rgb']
    else:
        data = capture_from_ros_topics(timeout_sec=args.timeout)
        if data is None:
            print("No frame captured, exiting.")
            return
        depth = data['depth']
        K = data['K']
        rgb = data['rgb']

        if args.save_capture:
            save_path = 'realsense_capture.npy'
            np.save(save_path, {
                'depth': depth, 'K': K, 'rgb': rgb
            })
            print(f"Saved capture to {save_path}")

    # ---------------------------------------------------------------
    # 2) Segment target object with SAM3
    # ---------------------------------------------------------------
    segmap, mask, seg_score = segment_with_sam3(
        rgb, args.object, confidence_threshold=args.confidence
    )
    if segmap is None:
        return

    print(f"Segmentation mask: {np.sum(mask)} pixels for '{args.object}'")

    # ---------------------------------------------------------------
    # 3) Extract object point cloud from depth + mask
    # ---------------------------------------------------------------
    print("\nExtracting object point cloud...")
    object_pc, pixel_coords = depth_to_point_cloud(
        depth, K, mask=mask, z_range=args.z_range
    )

    if len(object_pc) < 100:
        print(f"Too few points in object cloud ({len(object_pc)}). "
              "Try adjusting z_range or moving the object closer.")
        return

    print(f"Object point cloud: {len(object_pc)} points")
    print(f"  Centroid: [{object_pc.mean(0)[0]:.4f}, {object_pc.mean(0)[1]:.4f}, {object_pc.mean(0)[2]:.4f}]")
    print(f"  Depth range: [{object_pc[:, 2].min():.4f}, {object_pc[:, 2].max():.4f}]")

    # Extract full scene point cloud for visualization
    full_pc, _ = depth_to_point_cloud(depth, K, z_range=args.z_range)

    # Extract scene point cloud EXCLUDING the object (for collision checking)
    inv_mask = ~mask if mask is not None else None
    scene_pc, _ = depth_to_point_cloud(depth, K, mask=inv_mask, z_range=args.z_range)
    print(f"Scene point cloud (excluding object): {len(scene_pc)} points")

    # Get colors for full point cloud
    H, W = depth.shape
    valid_full = (depth > args.z_range[0]) & (depth < args.z_range[1])
    pc_colors = rgb[valid_full]  # RGB colors for full point cloud

    # ---------------------------------------------------------------
    # 4) Run GraspGen (with collision filtering)
    # ---------------------------------------------------------------
    grasps, grasp_conf = run_graspgen(
        object_pc,
        scene_pc=scene_pc,
        num_grasps=args.num_grasps,
        grasp_threshold=args.grasp_threshold,
        topk_num_grasps=args.topk,
        collision_check=not args.no_collision,
        collision_threshold=args.collision_threshold,
    )

    if len(grasps) == 0:
        print("\nNo grasps generated! Try adjusting parameters or moving the object.")
        return

    print(f"\nTotal grasps generated: {len(grasps)}")

    # ---------------------------------------------------------------
    # 5) Select optimal grasp
    # ---------------------------------------------------------------
    best_grasp, best_score, best_idx = select_optimal_grasp(grasps, grasp_conf)

    if best_grasp is None:
        print("No valid forward grasps found.")
        return

    # ---------------------------------------------------------------
    # 6) Convert to MoveIt pose
    # ---------------------------------------------------------------
    moveit_pose = grasp_to_moveit_pose(best_grasp)
    print_moveit_pose(moveit_pose, best_score)

    # Save results (compatible with grasp_execute.py)
    os.makedirs('results', exist_ok=True)
    np.savez('results/grasp_pipeline_result.npz',
             best_grasp=best_grasp,
             best_score=best_score,
             moveit_position=np.array([moveit_pose['position']['x'],
                                       moveit_pose['position']['y'],
                                       moveit_pose['position']['z']]),
             moveit_quaternion=np.array([moveit_pose['orientation']['x'],
                                         moveit_pose['orientation']['y'],
                                         moveit_pose['orientation']['z'],
                                         moveit_pose['orientation']['w']]),
             pc_full=full_pc)
    print("Results saved to results/grasp_pipeline_result.npz")

    # Save predictions for visualization
    vis_pred_path = 'results/graspgen_predictions.npz'
    np.savez(vis_pred_path,
             pc_full=full_pc,
             pc_colors=pc_colors,
             object_pc=object_pc,
             grasps=grasps,
             grasp_conf=grasp_conf,
             selected_grasp=best_grasp,
             selected_idx=best_idx)

    # ---------------------------------------------------------------
    # 7) Visualize
    # ---------------------------------------------------------------
    print("\nLaunching visualization (non-blocking — close window anytime)...")
    import subprocess
    subprocess.Popen(
        [sys.executable, '-c', VISUALIZER_CODE, vis_pred_path],
        env=os.environ.copy()
    )
    print("Pipeline complete. Robot will proceed immediately.")


# Visualization code runs in a separate process (no SAM3/torch conflict with Open3D)
VISUALIZER_CODE = '''
import sys
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def main():
    pred_path = sys.argv[1]
    data = np.load(pred_path, allow_pickle=True)
    pc_full = data["pc_full"]
    pc_colors = data["pc_colors"]
    object_pc = data["object_pc"]
    grasps = data["grasps"]
    grasp_conf = data["grasp_conf"]
    selected_grasp = data["selected_grasp"]
    selected_idx = int(data["selected_idx"])

    # Build point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_full)
    pcd.colors = o3d.utility.Vector3dVector(pc_colors.astype(np.float64) / 255)

    # Object points in green overlay
    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(object_pc)
    obj_pcd.paint_uniform_color([0.0, 1.0, 0.0])

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="GraspGen Visualization")
    vis.add_geometry(pcd)
    vis.add_geometry(obj_pcd)

    # Camera frame axes
    cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vis.add_geometry(cam_frame)

    # Gripper visualization parameters
    gripper_width = 0.085  # Custom gripper width
    gripper_depth = 0.178  # Custom gripper depth

    # Grasp line template: origin -> mid -> left finger -> tip, mid -> right finger -> tip
    # Using a simple line representation
    def make_grasp_lines(g, opening):
        """Create line points for a gripper at grasp pose g."""
        # Control points in gripper frame (z=approach, x=closing)
        pts_local = np.array([
            [0, 0, 0],                          # base
            [0, 0, gripper_depth * 0.5],         # mid-shaft
            [opening/2, 0, gripper_depth * 0.5], # right finger base
            [opening/2, 0, gripper_depth],        # right finger tip
            [-opening/2, 0, gripper_depth * 0.5], # left finger base (back to mid)
            [-opening/2, 0, gripper_depth],        # left finger tip
        ])
        # Transform to world frame
        pts_world = pts_local @ g[:3, :3].T + g[:3, 3]
        return pts_world

    viridis = plt.get_cmap("viridis")

    if len(grasps) > 0:
        max_s, min_s = np.max(grasp_conf), np.min(grasp_conf)
        rng = max_s - min_s if max_s > min_s else 1.0
        grasp_colors = np.array([viridis((s - min_s) / rng)[:3] for s in grasp_conf])

        all_pts = []
        connections = []
        all_colors = []
        index = 0
        N_pts = 6
        # Line connections: 0-1, 1-2, 2-3, 1-4, 4-5
        local_conns = np.array([[0,1],[1,2],[2,3],[1,4],[4,5]])

        for i, (g, conf) in enumerate(zip(grasps, grasp_conf)):
            if i == selected_idx:
                continue  # Draw selected separately in red
            pts = make_grasp_lines(g, gripper_width)
            all_pts.append(pts)
            conn = local_conns + index
            connections.append(conn)
            all_colors.extend([grasp_colors[i]] * len(local_conns))
            index += N_pts

        if all_pts:
            all_pts = np.vstack(all_pts)
            connections = np.vstack(connections).astype(np.int32)
            all_colors = np.array(all_colors, dtype=np.float64)

            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(all_pts)
            line_set.lines = o3d.utility.Vector2iVector(connections)
            line_set.colors = o3d.utility.Vector3dVector(all_colors)
            vis.add_geometry(line_set)

    # Draw the SELECTED grasp in red
    if selected_grasp is not None:
        pts = make_grasp_lines(selected_grasp, gripper_width)

        # Red spheres at control points
        for p in pts:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.004)
            sphere.translate(p)
            sphere.paint_uniform_color([1.0, 0.0, 0.0])
            vis.add_geometry(sphere)

        # Red lines
        local_conns = np.array([[0,1],[1,2],[2,3],[1,4],[4,5]])
        sel_line = o3d.geometry.LineSet()
        sel_line.points = o3d.utility.Vector3dVector(pts)
        sel_line.lines = o3d.utility.Vector2iVector(local_conns.astype(np.int32))
        sel_line.colors = o3d.utility.Vector3dVector(
            np.tile([1.0, 0.0, 0.0], (len(local_conns), 1)))
        vis.add_geometry(sel_line)

        # Coordinate frame at the selected grasp origin
        grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        grasp_frame.transform(selected_grasp)
        vis.add_geometry(grasp_frame)

        pos = selected_grasp[:3, 3]
        app = selected_grasp[:3, 2]
        clo = selected_grasp[:3, 0]
        print(f"Selected grasp (RED): pos=[{pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f}], "
              f"approach=[{app[0]:.3f},{app[1]:.3f},{app[2]:.3f}], "
              f"closing=[{clo[0]:.3f},{clo[1]:.3f},{clo[2]:.3f}]")

    print("Visualizing grasps. Close the window when done.")
    print("RED = selected grasp that will be sent to the robot.")
    print("GREEN points = segmented object.")
    vis.run()
    vis.destroy_window()

main()
'''


if __name__ == '__main__':
    main()
