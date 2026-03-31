"""
Launch file for cuRobo trajectory planning with xArm robots.

Designed to work alongside xarm_fake_moveit or xarm real robot launch.
Does NOT launch robot_state_publisher or joint_state_publisher since
those are provided by the xarm MoveIt launch.

Prerequisites:
    Launch xarm_fake_moveit (or real robot) FIRST, then launch this file.

Usage:
    # With fake robot (launch fake_moveit first on host):
    ros2 launch curobo_ros xarm.launch.py

    # With custom config:
    ros2 launch curobo_ros xarm.launch.py \
        robot_config_file:=/path/to/xarm7_sphere.yaml

    # Without cuRobo RViz (use xarm's own RViz):
    ros2 launch curobo_ros xarm.launch.py gui:=false

    # With cameras:
    ros2 launch curobo_ros xarm.launch.py \
        cameras_config_file:=/path/to/cameras.yaml \
        include_realsense_launch:=true
"""

from launch import LaunchDescription
from launch.actions import LogInfo, OpaqueFunction, DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import yaml


def _get_config_value(config_file_path, keys, default):
    """Extract a nested value from a YAML config file."""
    try:
        with open(config_file_path, 'r') as f:
            config = yaml.safe_load(f)
        value = config
        for key in keys:
            value = value[key]
        return value if value else default
    except Exception:
        return default


def launch_setup(context, *args, **kwargs):
    robot_config_file = LaunchConfiguration('robot_config_file').perform(context)
    urdf_path_arg = LaunchConfiguration('urdf_path').perform(context)

    # Resolve URDF path from config if not provided
    if not urdf_path_arg:
        urdf_path = _get_config_value(
            robot_config_file,
            ['robot_cfg', 'kinematics', 'urdf_path'],
            ''
        )
        if urdf_path:
            print(f"[xarm.launch] URDF path from config: {urdf_path}")
        else:
            print("[xarm.launch] WARNING: No URDF path found in config")
    else:
        urdf_path = urdf_path_arg
        print(f"[xarm.launch] Using provided URDF path: {urdf_path}")

    # Get base_link from config
    base_link = _get_config_value(
        robot_config_file,
        ['robot_cfg', 'kinematics', 'base_link'],
        'link_base'
    )
    print(f"[xarm.launch] Using base_link: {base_link}")

    # Read URDF content for ghost/preview robot
    urdf_content = ""
    if urdf_path:
        try:
            with open(urdf_path, 'r') as f:
                urdf_content = f.read()
        except FileNotFoundError:
            print(f"[xarm.launch] WARNING: URDF file not found: {urdf_path}")

    curobo_ros_launch_dir = os.path.join(
        get_package_share_directory('curobo_ros'), 'launch')

    nodes = []

    # Realsense camera (optional)
    nodes.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(curobo_ros_launch_dir, 'realsense.launch.py')),
            condition=IfCondition(LaunchConfiguration('include_realsense_launch'))
        )
    )

    # cuRobo trajectory planner node
    nodes.append(
        Node(
            package='curobo_ros',
            executable='curobo_trajectory_planner',
            output='screen',
            parameters=[{
                'robot_config_file': LaunchConfiguration('robot_config_file'),
                'cameras_config_file': LaunchConfiguration('cameras_config_file'),
                'base_link': base_link,
                'world_file': LaunchConfiguration('world_file'),
                'robot_type': 'xarm',
                'time_dilation_factor': LaunchConfiguration('time_dilation_factor'),
                'xarm_joint_states_topic': LaunchConfiguration('joint_states_topic'),
                'xarm_action_topic': LaunchConfiguration('action_topic'),
            }]
        )
    )

    # Ghost/preview robot for trajectory visualization
    if urdf_content:
        nodes.extend([
            Node(
                package='joint_state_publisher',
                executable='joint_state_publisher',
                namespace='preview',
                parameters=[{
                    'source_list': ['/trajectory/joint_states'],
                    'base_link': base_link
                }]
            ),

            Node(
                package='robot_state_publisher',
                executable='robot_state_publisher',
                namespace='preview',
                parameters=[{
                    'robot_description': urdf_content,
                    'frame_prefix': 'preview/',
                    'base_link': base_link
                }]
            ),

            Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                namespace='preview',
                arguments=['0', '0', '0', '0', '0', '0', 'world', 'preview/world']
            ),
        ])

    # cuRobo RViz (optional, disabled by default since xarm_fake_moveit has its own)
    nodes.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(curobo_ros_launch_dir, 'rviz_visualization.launch.py')
            ),
            launch_arguments={'base_link': base_link}.items(),
            condition=IfCondition(LaunchConfiguration('gui'))
        )
    )

    nodes.append(LogInfo(msg='cuRobo xArm planner launched'))

    return nodes


def generate_launch_description():
    # Default xarm7 config path
    default_config = os.path.join(
        get_package_share_directory('curobo_ros'),
        'config', 'xarm7_sphere.yaml'
    )
    # Fall back to xarm_description package if available
    try:
        xarm_desc_dir = get_package_share_directory('xarm_description')
        xarm_config = os.path.join(xarm_desc_dir, 'urdf', 'xarm7', 'xarm7_sphere.yaml')
        if os.path.exists(xarm_config):
            default_config = xarm_config
    except Exception:
        pass

    default_world_file = os.path.join(
        get_package_share_directory('curobo_ros'),
        'config', 'floor_world.yml'
    )

    return LaunchDescription([
        # Robot configuration
        DeclareLaunchArgument(
            'robot_config_file',
            default_value=default_config,
            description='Path to the cuRobo robot config YAML (with collision spheres)'
        ),
        DeclareLaunchArgument(
            'urdf_path',
            default_value='',
            description='Path to URDF file (if empty, read from robot_config_file)'
        ),

        # xArm connection
        DeclareLaunchArgument(
            'joint_states_topic',
            default_value='/joint_states',
            description='Topic to read current joint positions from the xArm controller'
        ),
        DeclareLaunchArgument(
            'action_topic',
            default_value='/xarm7_traj_controller/follow_joint_trajectory',
            description='FollowJointTrajectory action server topic'
        ),

        # Optional features
        DeclareLaunchArgument(
            'cameras_config_file',
            default_value='',
            description='Path to cameras configuration YAML'
        ),
        DeclareLaunchArgument(
            'world_file',
            default_value=default_world_file,
            description='Path to world configuration YAML (obstacles)'
        ),
        DeclareLaunchArgument(
            'include_realsense_launch',
            default_value='false',
            description='Launch RealSense camera nodes'
        ),
        DeclareLaunchArgument(
            'gui',
            default_value='false',
            description='Launch cuRobo RViz (disable if using xarm MoveIt RViz)'
        ),

        # Planner parameters
        DeclareLaunchArgument(
            'max_attempts', default_value='4',
            description='Maximum planning attempts'
        ),
        DeclareLaunchArgument(
            'timeout', default_value='5.0',
            description='Planning timeout in seconds'
        ),
        DeclareLaunchArgument(
            'time_dilation_factor', default_value='0.2',
            description='Trajectory time scaling factor'
        ),
        DeclareLaunchArgument(
            'voxel_size', default_value='0.05',
            description='Voxel size for collision checking'
        ),
        DeclareLaunchArgument(
            'collision_activation_distance', default_value='0.025',
            description='Collision activation distance'
        ),

        OpaqueFunction(function=launch_setup)
    ])
