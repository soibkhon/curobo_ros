from curobo_ros.robot.joint_control_strategy import JointCommandStrategy, RobotState
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
import threading
import time
import yaml


class XArmStrategy(JointCommandStrategy):
    '''
    Robot strategy for xArm robots via FollowJointTrajectory action.
    Reads current joint positions from /joint_states and sends trajectories
    to the xarm controller's follow_joint_trajectory action server.
    '''

    def __init__(self, node, dt):
        super().__init__(node, dt)

        self.node = node
        self.dt = dt

        # Read joint names from robot config
        self._curobo_joint_names = self._load_joint_names_from_config(node)

        # Current state from /joint_states subscription
        self.current_joint_positions = [0.0] * len(self._curobo_joint_names)
        self.joint_names = list(self._curobo_joint_names)
        self._joint_state_received = False

        # Trajectory commands (set by set_command from base class)
        self.position_command = []
        self.vel_command = []
        self.accel_command = []

        self.robot_state = RobotState.IDLE
        self.trajectory_progression = 0.0

        # Action server topic (configurable via parameter)
        if not node.has_parameter('xarm_action_topic'):
            node.declare_parameter(
                'xarm_action_topic',
                '/xarm7_traj_controller/follow_joint_trajectory'
            )
        action_topic = node.get_parameter(
            'xarm_action_topic'
        ).get_parameter_value().string_value

        # Joint states topic (configurable via parameter)
        if not node.has_parameter('xarm_joint_states_topic'):
            node.declare_parameter('xarm_joint_states_topic', '/joint_states')
        joint_states_topic = node.get_parameter(
            'xarm_joint_states_topic'
        ).get_parameter_value().string_value

        # Subscribe to joint states from the xarm controller
        self._joint_state_sub = node.create_subscription(
            JointState,
            joint_states_topic,
            self._joint_state_callback,
            10
        )

        # Action client for sending trajectories
        # Use ReentrantCallbackGroup so action client callbacks can fire
        # even while the execute_callback is blocking the executor
        self._action_cb_group = ReentrantCallbackGroup()
        self._action_client = ActionClient(
            node,
            FollowJointTrajectory,
            action_topic,
            callback_group=self._action_cb_group
        )

        # Execution state
        self._goal_handle = None
        self._execution_thread = None
        self._execution_error = False

        node.get_logger().info(
            f"XArm strategy initialized:"
            f" joints={self._curobo_joint_names},"
            f" action={action_topic},"
            f" joint_states={joint_states_topic}"
        )

    def _load_joint_names_from_config(self, node):
        '''Read joint names from robot config YAML.'''
        try:
            from ament_index_python.packages import get_package_share_directory
            import os

            default_config = os.path.join(
                get_package_share_directory('curobo_ros'),
                'curobo_doosan', 'src', 'm1013', 'm1013.yml'
            )

            if not node.has_parameter('robot_config_file'):
                node.declare_parameter('robot_config_file', default_config)
            config_path = node.get_parameter(
                'robot_config_file'
            ).get_parameter_value().string_value

            if not config_path:
                config_path = default_config

            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            joint_names = config['robot_cfg']['kinematics']['cspace']['joint_names']
            node.get_logger().info(
                f"XArm strategy loaded {len(joint_names)} joints from config: {joint_names}"
            )
            return joint_names
        except Exception as e:
            node.get_logger().error(f"Could not load joint names from config: {e}")
            return ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']

    def _joint_state_callback(self, msg):
        '''
        Callback for /joint_states. Extracts only the curobo joint positions
        in the correct order (the xarm may publish joints in arbitrary order).
        '''
        joint_map = dict(zip(msg.name, msg.position))

        positions = []
        for name in self._curobo_joint_names:
            if name in joint_map:
                positions.append(joint_map[name])
            else:
                idx = self._curobo_joint_names.index(name)
                positions.append(self.current_joint_positions[idx])

        self.current_joint_positions = positions
        self._joint_state_received = True

    def _build_trajectory_msg(self):
        '''Build a JointTrajectory message from the stored commands.'''
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = list(self._curobo_joint_names)

        time_from_start = 0.0
        for i, positions in enumerate(self.position_command):
            point = JointTrajectoryPoint()
            point.positions = [float(p) for p in positions]

            if i < len(self.vel_command):
                point.velocities = [float(v) for v in self.vel_command[i]]
            else:
                point.velocities = [0.0] * len(positions)

            if i < len(self.accel_command):
                point.accelerations = [float(a) for a in self.accel_command[i]]

            time_from_start += self.dt
            secs = int(time_from_start)
            nsecs = int((time_from_start - secs) * 1e9)
            point.time_from_start = Duration(sec=secs, nanosec=nsecs)

            trajectory_msg.points.append(point)

        return trajectory_msg

    def send_trajectrory(self):
        '''
        Send the planned trajectory to the xarm controller via FollowJointTrajectory action.
        Runs in a separate thread to keep the ROS executor free for callbacks.
        '''
        self.node.get_logger().info(
            f"XArm: send_trajectrory called, position_command has "
            f"{len(self.position_command)} waypoints"
        )

        if len(self.position_command) == 0:
            self.node.get_logger().warn("XArm: No trajectory to execute")
            self.trajectory_progression = 1.0
            return

        self.robot_state = RobotState.RUNNING
        self.trajectory_progression = 0.0
        self._execution_error = False

        # Run in thread so the ROS executor can process action callbacks
        self._execution_thread = threading.Thread(
            target=self._execute_on_robot,
            daemon=True,
            name="xarm_trajectory_executor"
        )
        self._execution_thread.start()

    def _execute_on_robot(self):
        '''Thread function that sends trajectory and monitors execution.'''
        try:
            self.node.get_logger().info("XArm: Thread started, waiting for action server...")

            if not self._action_client.wait_for_server(timeout_sec=5.0):
                self.node.get_logger().error(
                    "XArm: FollowJointTrajectory action server not available"
                )
                self.robot_state = RobotState.ERROR
                self._execution_error = True
                self.trajectory_progression = 1.0
                return

            self.node.get_logger().info("XArm: Action server found, building trajectory...")
            trajectory_msg = self._build_trajectory_msg()
            total_time = len(self.position_command) * self.dt

            self.node.get_logger().info(
                f"XArm: Sending trajectory with {len(trajectory_msg.points)} points"
                f" ({total_time:.2f}s), joints={trajectory_msg.joint_names}"
            )

            # Log first and last waypoint for debugging
            if len(trajectory_msg.points) > 0:
                first = trajectory_msg.points[0]
                last = trajectory_msg.points[-1]
                self.node.get_logger().info(
                    f"XArm: First waypoint: {[f'{p:.3f}' for p in first.positions]}"
                )
                self.node.get_logger().info(
                    f"XArm: Last waypoint: {[f'{p:.3f}' for p in last.positions]}"
                )

            goal_msg = FollowJointTrajectory.Goal()
            goal_msg.trajectory = trajectory_msg

            # Send goal (async, but we wait for acceptance here)
            self.node.get_logger().info("XArm: Calling send_goal_async...")
            send_future = self._action_client.send_goal_async(
                goal_msg,
                feedback_callback=self._feedback_callback
            )

            # Wait for goal acceptance
            wait_count = 0
            while not send_future.done():
                time.sleep(0.01)
                wait_count += 1
                if wait_count % 100 == 0:
                    self.node.get_logger().info(
                        f"XArm: Still waiting for goal response ({wait_count * 0.01:.1f}s)..."
                    )
                if wait_count > 1000:  # 10 second timeout
                    self.node.get_logger().error("XArm: Timeout waiting for goal response")
                    self.robot_state = RobotState.ERROR
                    self._execution_error = True
                    self.trajectory_progression = 1.0
                    return

            goal_handle = send_future.result()
            if not goal_handle.accepted:
                self.node.get_logger().error("XArm: Trajectory goal REJECTED")
                self.robot_state = RobotState.ERROR
                self._execution_error = True
                self.trajectory_progression = 1.0
                return

            self.node.get_logger().info("XArm: Trajectory goal ACCEPTED, executing...")
            self._goal_handle = goal_handle

            # Wait for result
            result_future = goal_handle.get_result_async()
            wait_count = 0
            while not result_future.done():
                time.sleep(0.05)
                wait_count += 1
                if wait_count % 20 == 0:
                    self.node.get_logger().info(
                        f"XArm: Executing... ({wait_count * 0.05:.1f}s elapsed, "
                        f"progression={self.trajectory_progression:.2f})"
                    )

            result = result_future.result()
            self.node.get_logger().info(f"XArm: Got result, status={result.status}")

            if result.status == 4:  # SUCCEEDED
                self.node.get_logger().info("XArm: Trajectory execution completed successfully")
                self.trajectory_progression = 1.0
                self.robot_state = RobotState.IDLE
            else:
                error_code = result.result.error_code
                self.node.get_logger().error(
                    f"XArm: Trajectory failed (status={result.status},"
                    f" error_code={error_code})"
                )
                self.robot_state = RobotState.ERROR
                self._execution_error = True
                self.trajectory_progression = 1.0

        except Exception as e:
            self.node.get_logger().error(f"XArm: Execution error: {e}")
            import traceback
            self.node.get_logger().error(traceback.format_exc())
            self.robot_state = RobotState.ERROR
            self._execution_error = True
            self.trajectory_progression = 1.0

    def _feedback_callback(self, feedback_msg):
        '''Called with progress feedback from the action server.'''
        feedback = feedback_msg.feedback
        if hasattr(feedback, 'desired') and len(self.position_command) > 0:
            elapsed = (
                feedback.desired.time_from_start.sec
                + feedback.desired.time_from_start.nanosec * 1e-9
            )
            total_time = len(self.position_command) * self.dt
            if total_time > 0:
                self.trajectory_progression = min(elapsed / total_time, 0.99)

    def get_joint_pose(self):
        '''Return current joint positions from /joint_states.'''
        return self.current_joint_positions

    def get_joint_name(self):
        '''Return joint names (in curobo config order).'''
        return self.joint_names

    def get_progression(self):
        '''Return trajectory execution progression (0.0 to 1.0).'''
        return self.trajectory_progression

    def has_error(self):
        '''Return True if the last execution had an error.'''
        return getattr(self, '_execution_error', False)

    def stop_robot(self):
        '''Cancel the current trajectory execution.'''
        self.node.get_logger().info("XArm: Stopping trajectory execution")

        if self._goal_handle is not None:
            try:
                self._goal_handle.cancel_goal_async()
            except Exception as e:
                self.node.get_logger().warn(f"XArm: Could not cancel goal: {e}")

        self.position_command = []
        self.vel_command = []
        self.accel_command = []
        self.trajectory_progression = 0.0
        self.robot_state = RobotState.STOPPED
