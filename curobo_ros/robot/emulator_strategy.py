from curobo_ros.robot.joint_control_strategy import JointCommandStrategy, RobotState
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
import threading
import time
import yaml


class EmulatorStrategy(JointCommandStrategy):
    '''
    Robot emulator strategy for visualization in RViz.
    Publishes JointState messages to simulate robot movement without a real robot.
    This allows testing and visualization of trajectories in RViz.
    '''

    def __init__(self, node, dt):
        super().__init__(node, dt)

        # Publisher for joint states (standard ROS topic for robot visualization)
        self.pub_joint_states = node.create_publisher(
            JointState,
            '/emulator/joint_states',
            10
        )

        # Emulator state
        self.position_command = []
        self.vel_command = []
        self.accel_command = []

        # Read joint names from robot config file (supports any DOF)
        self._initial_positions = None
        joint_names = self._load_joint_names_from_config(node)
        self.joint_names = joint_names
        if self._initial_positions is not None:
            self.current_joint_positions = self._initial_positions
        else:
            self.current_joint_positions = [0.0] * len(joint_names)

        self.command_index = 0
        self.dt = dt
        self.robot_state = RobotState.IDLE
        self.trajectory_progression = 0.0

        self.node = node

        # Thread for trajectory execution simulation
        self.execution_thread = None
        self.stop_execution = threading.Event()

        # Publish initial joint state so RViz robot matches the emulator's start pose
        self._publish_current_state()

        # Keep publishing initial state periodically until a trajectory overrides it
        self._idle_timer = node.create_timer(0.1, self._publish_idle_state)

        node.get_logger().info(
            f"Emulator strategy initialized with {len(self.joint_names)} joints: {self.joint_names}"
        )

    def _load_joint_names_from_config(self, node):
        '''
        Read joint names and retract config from the robot config YAML file.
        Sets initial joint positions to retract_config for a collision-free start.
        Uses the same default config path as ConfigManager for consistency.
        '''
        try:
            from ament_index_python.packages import get_package_share_directory
            import os

            # Use the same default as ConfigManager
            default_config = os.path.join(
                get_package_share_directory('curobo_ros'),
                'curobo_doosan', 'src', 'm1013', 'm1013.yml'
            )

            if not node.has_parameter('robot_config_file'):
                node.declare_parameter('robot_config_file', default_config)
            config_path = node.get_parameter('robot_config_file').get_parameter_value().string_value

            if not config_path:
                config_path = default_config

            node.get_logger().info(f"Emulator reading config from: {config_path}")

            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            cspace = config['robot_cfg']['kinematics']['cspace']
            joint_names = cspace['joint_names']
            retract = cspace.get('retract_config', None)
            if retract and len(retract) == len(joint_names):
                self._initial_positions = [float(v) for v in retract]
            else:
                self._initial_positions = [0.0] * len(joint_names)
            node.get_logger().info(f"Emulator loaded {len(joint_names)} joints: {joint_names}")
            node.get_logger().info(f"Emulator initial positions: {self._initial_positions}")
            return joint_names
        except Exception as e:
            node.get_logger().error(f"Could not load joint names from config: {e}")
            import traceback
            node.get_logger().error(traceback.format_exc())

        # Fallback: generic 6-joint names
        node.get_logger().warn("Emulator using default 6-joint configuration")
        self._initial_positions = None
        return ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']

    def _publish_current_state(self):
        '''Publish current joint positions to /emulator/joint_states.'''
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = list(self.current_joint_positions)
        msg.velocity = [0.0] * len(self.joint_names)
        msg.effort = []
        self.pub_joint_states.publish(msg)

    def _publish_idle_state(self):
        '''Timer callback: publish current state while idle so RViz stays in sync.'''
        if self.robot_state == RobotState.IDLE or self.robot_state == RobotState.STOPPED:
            self._publish_current_state()

    def send_trajectrory(self):
        '''
        Start simulating trajectory execution by publishing joint states progressively.
        '''
        if len(self.position_command) == 0:
            self.node.get_logger().warn("No trajectory to execute")
            self.trajectory_progression = 1.0
            return

        # Stop any previous execution
        self.stop_execution.set()
        if self.execution_thread is not None and self.execution_thread.is_alive():
            self.execution_thread.join(timeout=1.0)

        # Start new execution thread
        self.stop_execution.clear()
        self.robot_state = RobotState.RUNNING
        self.trajectory_progression = 0.0
        self.command_index = 0

        self.execution_thread = threading.Thread(
            target=self._execute_trajectory,
            daemon=True,
            name="emulator_trajectory_executor"
        )
        self.execution_thread.start()

    def _execute_trajectory(self):
        '''
        Thread function that simulates trajectory execution by publishing joint states.
        '''
        try:
            total_points = len(self.position_command)

            while self.command_index < total_points and not self.stop_execution.is_set():
                # Get current command
                positions = self.position_command[self.command_index]
                velocities = self.vel_command[self.command_index] if self.command_index < len(self.vel_command) else [0.0] * len(positions)

                # Update current position
                self.current_joint_positions = positions

                # Create and publish JointState message
                joint_state_msg = JointState()
                joint_state_msg.header.stamp = self.node.get_clock().now().to_msg()
                joint_state_msg.name = self.joint_names
                joint_state_msg.position = positions
                joint_state_msg.velocity = velocities
                joint_state_msg.effort = []

                self.pub_joint_states.publish(joint_state_msg)

                # Update progression
                self.command_index += 1
                self.trajectory_progression = self.command_index / total_points

                # Wait for next timestep
                time.sleep(self.dt)

            # Set robot state to IDLE
            self.robot_state = RobotState.IDLE

            # Trajectory complete
            if not self.stop_execution.is_set():
                self.trajectory_progression = 1.0
                self.robot_state = RobotState.IDLE

        except Exception as e:
            self.node.get_logger().error(f"❌ Emulator execution error: {e}")
            import traceback
            self.node.get_logger().error(traceback.format_exc())
            self.robot_state = RobotState.ERROR

    def get_joint_pose(self):
        '''
        Return the current joint positions of the emulated robot.
        '''
        return self.current_joint_positions

    def get_joint_name(self):
        '''
        Return the joint names of the emulated robot.
        '''
        return self.joint_names

    def wait_for_execution_complete(self, timeout=5.0):
        '''
        Wait for the trajectory execution thread to complete.

        This ensures that the robot's current_joint_positions is fully updated
        before the next planning operation reads it.

        Args:
            timeout: Maximum time to wait in seconds (default: 5.0)

        Returns:
            True if thread completed, False if timeout occurred
        '''
        if self.execution_thread is not None and self.execution_thread.is_alive():
            self.execution_thread.join(timeout=timeout)
            return not self.execution_thread.is_alive()
        return True

    def stop_robot(self):
        '''
        Stop the emulated robot trajectory execution.
        '''
        self.node.get_logger().info("🛑 Emulator: Stopping trajectory execution")

        # Signal thread to stop
        self.stop_execution.set()

        # Clear command buffers
        self.vel_command = []
        self.position_command = []
        self.accel_command = []
        self.command_index = 0
        self.trajectory_progression = 0.0
        self.robot_state = RobotState.STOPPED

        # Publish stopped state (zero velocities)
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = self.node.get_clock().now().to_msg()
        joint_state_msg.name = self.joint_names
        joint_state_msg.position = self.current_joint_positions
        joint_state_msg.velocity = [0.0] * len(self.joint_names)
        joint_state_msg.effort = []
        self.pub_joint_states.publish(joint_state_msg)

    def get_progression(self):
        '''
        Return the trajectory execution progression (0.0 to 1.0).
        '''
        return self.trajectory_progression

    def __del__(self):
        '''
        Cleanup when strategy is destroyed.
        '''
        self.stop_execution.set()
        if self.execution_thread is not None and self.execution_thread.is_alive():
            self.execution_thread.join(timeout=1.0)
