# 13. ROS2 Fundamentals

[← Previous: SLAM](12_SLAM.md) | [Next: ROS2 Navigation Stack →](14_ROS2_Navigation.md)

---

## Learning Objectives

1. Explain the key differences between ROS1 and ROS2 and why the migration occurred
2. Understand ROS2 core concepts: nodes, topics, services, actions, and their communication patterns
3. Define custom message, service, and action interfaces
4. Write ROS2 launch files and manage node parameters
5. Use TF2 for coordinate frame management in multi-frame robotic systems
6. Build and organize ROS2 packages using the colcon build system

---

Throughout the previous lessons, we developed the mathematical foundations of robotics: kinematics, dynamics, control, perception, and SLAM. But building a real robot requires integrating all of these components into a working software system. The control algorithm must receive sensor data, the planner must send trajectories to the controller, the SLAM system must share its map with the navigator — all in real time, reliably, and without tight coupling between components.

This is the problem that **ROS2 (Robot Operating System 2)** solves. ROS2 is not an operating system in the traditional sense — it is a middleware framework that provides the communication infrastructure, tools, and conventions for building modular, reusable robot software. It has become the de facto standard in robotics research and is increasingly adopted in industry. Learning ROS2 is essential for any roboticist who wants to build systems that go beyond simulation.

> **Analogy**: ROS2 nodes are like microservices in a web application — each handles one responsibility (camera driver, SLAM, navigation planner), and they communicate through well-defined interfaces (topics, services, actions). Just as you can replace one microservice without rewriting the entire application, you can swap a SLAM algorithm or a sensor driver in ROS2 without changing the rest of the system. The "ROS2 ecosystem" is like an app store for robotics: thousands of open-source packages ready to plug in.

---

## 1. ROS1 to ROS2: Why the Rewrite?

### 1.1 Limitations of ROS1

ROS1 (2007-2020) was enormously successful but had fundamental architectural limitations:

| Limitation | Impact |
|-----------|--------|
| Single point of failure (rosmaster) | If rosmaster crashes, all communication breaks |
| No real-time support | Cannot guarantee timing for safety-critical control |
| TCP/UDP only | No standardized QoS, no shared memory transport |
| Linux-only (practically) | Cannot run on embedded systems or Windows |
| No built-in security | Open network, no authentication or encryption |
| No lifecycle management | Nodes start/stop in undefined order |

### 1.2 ROS2 Architecture

ROS2 addresses all of these by building on **DDS (Data Distribution Service)**, a mature industrial middleware standard:

```
┌──────────────────────────────────────────┐
│          User Application Code            │
├──────────────────────────────────────────┤
│          ROS2 Client Library (rclpy/rclcpp) │
├──────────────────────────────────────────┤
│          RMW (ROS Middleware Interface)    │
├──────────────────────────────────────────┤
│          DDS Implementation               │
│     (CycloneDDS / FastDDS / Connext)      │
├──────────────────────────────────────────┤
│          UDP / Shared Memory              │
└──────────────────────────────────────────┘
```

Key improvements:
- **No central master**: Nodes discover each other automatically via DDS multicast
- **Quality of Service (QoS)**: Configurable reliability, durability, deadline, and history policies
- **Real-time capable**: With proper DDS configuration and `rclcpp`, sub-millisecond latencies
- **Cross-platform**: Linux, macOS, Windows, and embedded (micro-ROS)
- **Lifecycle nodes**: Standardized state machine for ordered startup/shutdown
- **Security**: DDS-Security provides authentication, encryption, and access control

### 1.3 ROS2 Distributions

ROS2 uses time-based releases (like Ubuntu):

| Distribution | Release | EOL | Ubuntu |
|-------------|---------|-----|--------|
| Humble Hawksbill | May 2022 | May 2027 | 22.04 |
| Iron Irwini | May 2023 | Nov 2024 | 22.04 |
| Jazzy Jalisco | May 2024 | May 2029 | 24.04 |
| Rolling | Continuous | N/A | Latest |

LTS (Long-Term Support) releases are recommended for production.

---

## 2. Core Concepts

### 2.1 Nodes

A **node** is the fundamental unit of computation in ROS2. Each node is a single process that performs a specific function (camera driver, SLAM, controller, etc.).

```python
import rclpy
from rclpy.node import Node


class MinimalNode(Node):
    """A minimal ROS2 node.

    Why inherit from Node? The Node base class provides all ROS2
    communication primitives (publishers, subscribers, services, timers)
    and handles lifecycle, logging, and parameter management.
    """

    def __init__(self):
        super().__init__('minimal_node')
        # get_logger() uses ROS2's logging framework — messages appear
        # in the console and can be captured by logging tools
        self.get_logger().info('Minimal node has started')


def main():
    rclpy.init()
    node = MinimalNode()

    try:
        # spin() blocks and processes callbacks (subscriptions, timers, etc.)
        # Why spin? ROS2 is event-driven — callbacks fire when messages
        # arrive or timers expire. spin() is the event loop that dispatches them.
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### 2.2 Topics (Publish/Subscribe)

**Topics** provide asynchronous, many-to-many communication. A publisher sends messages to a named topic; any number of subscribers can receive them.

```
┌──────────┐     /camera/image     ┌──────────────┐
│  Camera   │ ──────────────────── │  SLAM Node   │
│  Driver   │                      └──────────────┘
│  (pub)    │     /camera/image     ┌──────────────┐
│           │ ──────────────────── │  Detector Node│
└──────────┘                       └──────────────┘
```

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import math


class ObstacleAvoidance(Node):
    """Simple reactive obstacle avoidance using LiDAR.

    Demonstrates ROS2 publish/subscribe pattern:
    - Subscribes to /scan (LiDAR data)
    - Publishes to /cmd_vel (velocity commands)

    Why pub/sub for sensor data? Sensor data flows continuously and
    may be consumed by multiple nodes (SLAM, obstacle avoidance,
    visualization). Pub/sub naturally handles this one-to-many pattern
    without the publisher needing to know who is listening.
    """

    def __init__(self):
        super().__init__('obstacle_avoidance')

        # Subscribe to LiDAR scans
        self.scan_sub = self.create_subscription(
            LaserScan,      # Message type
            '/scan',        # Topic name
            self.scan_callback,  # Callback function
            10              # QoS queue depth
        )

        # Publish velocity commands
        self.cmd_pub = self.create_publisher(
            Twist,          # Message type
            '/cmd_vel',     # Topic name
            10              # QoS queue depth
        )

        self.get_logger().info('Obstacle avoidance node started')

    def scan_callback(self, msg):
        """Process LiDAR scan and compute velocity command.

        Why process in the callback? ROS2 calls this function every time
        a new LaserScan message arrives on /scan. The callback should
        be fast (< message period) to avoid dropping messages.
        """
        # Find minimum distance in the front 60-degree arc
        n_beams = len(msg.ranges)
        front_start = n_beams // 2 - n_beams // 6
        front_end = n_beams // 2 + n_beams // 6
        front_ranges = [r for r in msg.ranges[front_start:front_end]
                        if msg.range_min < r < msg.range_max]

        min_front_dist = min(front_ranges) if front_ranges else float('inf')

        cmd = Twist()

        if min_front_dist < 0.5:
            # Too close — turn in place
            cmd.linear.x = 0.0
            cmd.angular.z = 0.5  # Turn left
        elif min_front_dist < 1.5:
            # Getting close — slow down and turn
            cmd.linear.x = 0.1
            cmd.angular.z = 0.3
        else:
            # Clear ahead — drive forward
            cmd.linear.x = 0.3
            cmd.angular.z = 0.0

        self.cmd_pub.publish(cmd)
```

### 2.3 Services (Request/Response)

**Services** provide synchronous, one-to-one communication. A client sends a request and blocks until the server responds.

```python
from example_interfaces.srv import AddTwoInts


class AddTwoIntsServer(Node):
    """Service server example.

    Why use services instead of topics? Services are for request-response
    interactions where the caller needs a result before proceeding.
    Examples: querying the current map, requesting a path plan,
    triggering a calibration routine. Topics are for streaming data;
    services are for on-demand computation.
    """

    def __init__(self):
        super().__init__('add_server')
        self.srv = self.create_service(
            AddTwoInts,           # Service type
            'add_two_ints',       # Service name
            self.add_callback     # Handler function
        )

    def add_callback(self, request, response):
        """Handle service request."""
        response.sum = request.a + request.b
        self.get_logger().info(f'{request.a} + {request.b} = {response.sum}')
        return response


class AddTwoIntsClient(Node):
    """Service client example.

    Why asynchronous calls? In ROS2, service calls should be
    non-blocking to avoid deadlocks (the node needs to keep processing
    callbacks while waiting for the response). call_async() returns a
    Future that completes when the response arrives.
    """

    def __init__(self):
        super().__init__('add_client')
        self.client = self.create_client(AddTwoInts, 'add_two_ints')

        # Wait for the service to become available
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for service...')

    def send_request(self, a, b):
        request = AddTwoInts.Request()
        request.a = a
        request.b = b
        future = self.client.call_async(request)
        return future
```

### 2.4 Actions (Goal-Based)

**Actions** are for long-running tasks that need feedback and can be canceled. They combine a goal (request), feedback (progress updates), and a result.

```
Client                     Server
  │                           │
  │───── Send Goal ──────────→│
  │←──── Goal Accepted ──────│
  │                           │
  │←──── Feedback ────────────│  (periodic progress updates)
  │←──── Feedback ────────────│
  │←──── Feedback ────────────│
  │                           │
  │←──── Result ──────────────│  (final result)
```

**Use cases**: Navigate to a waypoint, pick up an object, execute a trajectory. These tasks take seconds to minutes and the caller wants progress updates.

```python
from rclpy.action import ActionServer, ActionClient
from nav2_msgs.action import NavigateToPose


class NavigationActionServer(Node):
    """Conceptual navigation action server.

    Why actions instead of services? A navigation command can take
    30+ seconds. With a service, the client would block the entire time
    with no progress information and no ability to cancel. Actions
    provide:
    1. Feedback: "Currently at (3.2, 1.5), 40% complete"
    2. Cancellation: Client can abort if priorities change
    3. Non-blocking: Client can do other work while waiting
    """

    def __init__(self):
        super().__init__('navigation_server')
        self._action_server = ActionServer(
            self,
            NavigateToPose,
            'navigate_to_pose',
            self.execute_callback
        )

    async def execute_callback(self, goal_handle):
        """Execute navigation goal with periodic feedback."""
        self.get_logger().info('Navigating to goal...')

        feedback = NavigateToPose.Feedback()

        # Simulate navigation in steps
        for i in range(10):
            # Check if the goal was canceled
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return NavigateToPose.Result()

            # Publish progress feedback
            feedback.current_pose.pose.position.x = float(i)
            # feedback.distance_remaining = ...
            goal_handle.publish_feedback(feedback)

            # Simulate work
            # await asyncio.sleep(0.5)

        goal_handle.succeed()
        result = NavigateToPose.Result()
        return result
```

---

## 3. Messages and Interfaces

### 3.1 Message Types (.msg)

ROS2 messages define the data structure for topic communication. They are defined in `.msg` files using a simple type-field syntax:

```
# geometry_msgs/msg/Twist.msg
# Represents velocity in free space, decomposed into linear and angular parts
Vector3 linear    # Linear velocity [m/s]
Vector3 angular   # Angular velocity [rad/s]
```

```
# sensor_msgs/msg/LaserScan.msg
std_msgs/Header header
float32 angle_min        # Start angle [rad]
float32 angle_max        # End angle [rad]
float32 angle_increment  # Angular distance between measurements [rad]
float32 time_increment   # Time between measurements [s]
float32 scan_time        # Time between scans [s]
float32 range_min        # Minimum range value [m]
float32 range_max        # Maximum range value [m]
float32[] ranges         # Range data [m]
float32[] intensities    # Intensity data (optional)
```

### 3.2 Custom Messages

```
# my_robot_msgs/msg/RobotState.msg
# Custom message for robot state reporting

std_msgs/Header header
float64[6] joint_positions     # Joint angles [rad]
float64[6] joint_velocities    # Joint velocities [rad/s]
float64[6] joint_torques       # Joint torques [Nm]
geometry_msgs/Pose end_effector_pose  # End-effector pose
bool is_moving                 # Whether the robot is in motion
string current_mode            # Operation mode: "idle", "moving", "error"
```

### 3.3 Service Types (.srv)

Services have a request and a response separated by `---`:

```
# my_robot_msgs/srv/PlanPath.srv
# Request
geometry_msgs/Pose start
geometry_msgs/Pose goal
float32 max_planning_time
---
# Response
nav_msgs/Path path
bool success
string message
```

### 3.4 Action Types (.action)

Actions have a goal, result, and feedback separated by `---`:

```
# my_robot_msgs/action/MoveArm.action
# Goal
float64[] target_joint_positions
float64 max_velocity
---
# Result
bool success
float64 final_error
---
# Feedback
float64[] current_joint_positions
float64 progress_percentage
```

---

## 4. Quality of Service (QoS)

### 4.1 Why QoS Matters

Different data types have different reliability requirements:

| Data Type | QoS Preference | Reason |
|-----------|---------------|--------|
| Camera images | Best effort | Dropping a frame is OK; latency matters |
| LiDAR scans | Best effort | Same reasoning as camera |
| Control commands | Reliable | Missing a command can be dangerous |
| Map data | Reliable, transient local | Subscribers need the latest map, even if they start late |
| TF transforms | Best effort, volatile | High rate, low latency; stale transforms are useless |
| Diagnostics | Reliable | Don't want to miss error reports |

### 4.2 QoS Profiles

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy


# Sensor data: best effort, volatile, keep last 5
sensor_qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=5
)

# Reliable data: reliable delivery, keep last 10
reliable_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10
)

# Latched topic (like map): transient local ensures late subscribers get last message
latched_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1
)
```

### 4.3 QoS Compatibility

Publishers and subscribers must have **compatible** QoS settings to communicate:

| Publisher | Subscriber | Compatible? |
|-----------|-----------|-------------|
| Reliable | Reliable | Yes |
| Best effort | Best effort | Yes |
| Reliable | Best effort | Yes (subscriber gets reliability "for free") |
| Best effort | Reliable | **No** (subscriber requires guarantees publisher cannot provide) |

**Common pitfall**: A subscriber with RELIABLE QoS cannot receive data from a BEST_EFFORT publisher. If you see "no messages received" in ROS2, QoS mismatch is a likely cause.

---

## 5. Launch System

### 5.1 Launch Files

ROS2 launch files (Python) configure and start multiple nodes, set parameters, and define remappings:

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """Launch file for a simple mobile robot.

    Why Python launch files? ROS2 replaced XML launch files (ROS1)
    with Python for full programming flexibility: conditionals, loops,
    environment variable access, and dynamic configuration. XML and
    YAML launch files are also supported for simpler cases.
    """

    # Declare launch arguments (configurable from command line)
    use_sim_arg = DeclareLaunchArgument(
        'use_sim', default_value='true',
        description='Whether to use simulation'
    )

    robot_name_arg = DeclareLaunchArgument(
        'robot_name', default_value='my_robot',
        description='Name of the robot'
    )

    # LiDAR driver node
    lidar_node = Node(
        package='rplidar_ros',
        executable='rplidar_node',
        name='lidar',
        parameters=[{
            'serial_port': '/dev/ttyUSB0',
            'frame_id': 'laser_frame',
            'angle_compensate': True,
        }],
        remappings=[
            ('/scan', '/my_robot/scan'),  # Remap topic name
        ]
    )

    # SLAM node
    slam_node = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim'),
            'resolution': 0.05,
            'max_laser_range': 12.0,
        }]
    )

    # Navigation node
    nav_node = Node(
        package='nav2_bringup',
        executable='bringup_launch.py',
        name='navigation',
    )

    return LaunchDescription([
        use_sim_arg,
        robot_name_arg,
        lidar_node,
        slam_node,
        nav_node,
    ])
```

### 5.2 Parameters

ROS2 parameters allow runtime configuration of nodes:

```python
class ConfigurableNode(Node):
    """Node with declared parameters.

    Why declare parameters? Declaration enforces that all parameters
    are documented with types and defaults. Undeclared parameters are
    rejected by default, catching typos and misconfiguration early.
    """

    def __init__(self):
        super().__init__('configurable_node')

        # Declare parameters with defaults and descriptions
        self.declare_parameter('max_speed', 1.0)
        self.declare_parameter('safety_distance', 0.5)
        self.declare_parameter('control_frequency', 20)

        # Read parameter values
        self.max_speed = self.get_parameter('max_speed').value
        self.safety_dist = self.get_parameter('safety_distance').value
        freq = self.get_parameter('control_frequency').value

        # Create timer at configured frequency
        self.timer = self.create_timer(1.0 / freq, self.control_loop)

        # Register callback for dynamic parameter changes
        self.add_on_set_parameters_callback(self.param_callback)

    def param_callback(self, params):
        """Handle dynamic parameter updates.

        Why support dynamic reconfiguration? During robot operation,
        you may need to adjust speed limits, safety thresholds, or
        algorithm parameters without restarting nodes. This callback
        validates and applies changes in real time.
        """
        from rcl_interfaces.msg import SetParametersResult

        for param in params:
            if param.name == 'max_speed':
                if param.value < 0:
                    return SetParametersResult(successful=False,
                                                reason='Speed must be >= 0')
                self.max_speed = param.value
            elif param.name == 'safety_distance':
                self.safety_dist = param.value

        return SetParametersResult(successful=True)

    def control_loop(self):
        """Periodic control callback using current parameters."""
        self.get_logger().info(
            f'Speed: {self.max_speed}, Safety: {self.safety_dist}')
```

---

## 6. TF2: Transform Management

### 6.1 The Transform Tree

A robot has many coordinate frames: base, each joint, each sensor, the map, the odometry frame. **TF2** maintains a tree of transforms between all frames, enabling any node to query the transform between any two frames at any time.

```
map
 └── odom
      └── base_link
           ├── laser_frame
           ├── camera_frame
           │    └── camera_optical_frame
           ├── imu_frame
           └── arm_base
                ├── shoulder_link
                │    └── elbow_link
                │         └── wrist_link
                │              └── gripper_frame
                └── ...
```

### 6.2 Static vs. Dynamic Transforms

**Static transforms**: Fixed relationships that never change (e.g., camera mounted on the robot at a fixed position). Published once.

**Dynamic transforms**: Relationships that change over time (e.g., joint angles, odometry). Published continuously at high rate.

```python
import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
import math


class TFPublisher(Node):
    """Publishes both static and dynamic transforms.

    Why a transform tree? Without TF2, every node that needs to convert
    coordinates between frames would need to subscribe to all relevant
    sensor and joint data and do the math itself. TF2 centralizes this:
    any node can ask 'where is the camera relative to the map at time T?'
    and get an interpolated answer, even if the camera and map frames
    are connected through multiple intermediate frames.
    """

    def __init__(self):
        super().__init__('tf_publisher')

        # Static: camera is 0.1m above and 0.2m forward of base_link
        self.static_broadcaster = StaticTransformBroadcaster(self)
        self._publish_static_transforms()

        # Dynamic: odometry transform (updated continuously)
        self.dynamic_broadcaster = TransformBroadcaster(self)
        self.timer = self.create_timer(0.02, self._publish_odom_tf)  # 50 Hz

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

    def _publish_static_transforms(self):
        """Publish static transform: base_link → camera_frame.

        Why publish static transforms separately? They only need to
        be sent once (with TRANSIENT_LOCAL QoS), saving bandwidth.
        The TF library caches them permanently.
        """
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = 'camera_frame'
        t.transform.translation.x = 0.2  # 20cm forward
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.1  # 10cm up
        # Identity rotation (camera aligned with base)
        t.transform.rotation.w = 1.0

        self.static_broadcaster.sendTransform(t)

    def _publish_odom_tf(self):
        """Publish dynamic transform: odom → base_link."""
        # Simulate simple forward motion
        self.x += 0.01 * math.cos(self.theta)
        self.y += 0.01 * math.sin(self.theta)
        self.theta += 0.001

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0

        # Convert yaw to quaternion
        t.transform.rotation.z = math.sin(self.theta / 2)
        t.transform.rotation.w = math.cos(self.theta / 2)

        self.dynamic_broadcaster.sendTransform(t)
```

### 6.3 Listening to Transforms

```python
from tf2_ros import Buffer, TransformListener
from rclpy.duration import Duration


class TFListenerNode(Node):
    """Listens to and uses transforms from the TF tree.

    Why buffer transforms? Sensor data arrives with timestamps. To
    correctly transform a LiDAR scan from laser_frame to map, we need
    the transform AT THE TIME the scan was taken, not the current time.
    The TF buffer stores recent transforms and interpolates as needed.
    """

    def __init__(self):
        super().__init__('tf_listener')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def get_robot_in_map(self):
        """Look up the robot's position in the map frame."""
        try:
            # Look up transform from map to base_link at the latest time
            transform = self.tf_buffer.lookup_transform(
                'map',           # Target frame
                'base_link',     # Source frame
                rclpy.time.Time(),  # Latest available
                timeout=Duration(seconds=1.0)
            )
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            self.get_logger().info(f'Robot at ({x:.2f}, {y:.2f}) in map')
            return transform
        except Exception as e:
            self.get_logger().warn(f'TF lookup failed: {e}')
            return None
```

---

## 7. Colcon Build System and Package Structure

### 7.1 Package Structure

A ROS2 Python package follows this structure:

```
my_robot_pkg/
├── package.xml          # Package metadata, dependencies
├── setup.py             # Python package setup
├── setup.cfg            # Entry points configuration
├── my_robot_pkg/
│   ├── __init__.py
│   ├── node_one.py      # Node implementation
│   └── node_two.py
├── launch/
│   └── robot_launch.py  # Launch file
├── config/
│   └── params.yaml      # Parameter files
└── test/
    └── test_node.py     # Unit tests
```

### 7.2 Building and CLI Essentials

```bash
# Build and source
colcon build --symlink-install   # Symlink: Python edits take effect without rebuild
source install/setup.bash

# Run and launch
ros2 run my_robot_pkg node_one
ros2 launch my_robot_pkg robot_launch.py

# Introspection
ros2 node list                   # Active nodes
ros2 topic list -t               # Topics with message types
ros2 topic echo /scan            # Print messages
ros2 topic hz /scan              # Measure publishing rate

# Parameters and services
ros2 param set /my_node max_speed 2.0
ros2 service call /add_two_ints example_interfaces/srv/AddTwoInts "{a: 1, b: 2}"

# TF and recording
ros2 run tf2_tools view_frames   # Visualize transform tree
ros2 bag record -a               # Record all topics
ros2 bag play my_bag/            # Replay recorded data
```

---

## 8. Lifecycle Nodes

### 8.1 Managed Node States

**Lifecycle nodes** provide a standardized state machine for deterministic startup and shutdown:

```
Unconfigured → on_configure() → Inactive → on_activate() → Active
                                    ↑         on_deactivate()  │
                                    └──────────────────────────┘
```

Why lifecycle management? In a complex robot system, nodes must start in a specific order (driver before SLAM, SLAM before navigation). Lifecycle nodes let a supervisor orchestrate this: configure all nodes, then activate them in order. This prevents race conditions where a node tries to use data from a source that hasn't started yet.

```python
from rclpy.lifecycle import Node as LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn


class ManagedSensorNode(LifecycleNode):
    """Lifecycle node for a sensor driver."""

    def __init__(self):
        super().__init__('managed_sensor')

    def on_configure(self, state):
        """Configure: allocate resources, set up connections."""
        self.publisher_ = self.create_publisher(LaserScan, '/scan', 10)
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        """Activate: start publishing data."""
        self.timer = self.create_timer(0.1, self.publish_scan)
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        """Deactivate: stop publishing but keep resources."""
        self.destroy_timer(self.timer)
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        """Cleanup: release all resources."""
        self.destroy_publisher(self.publisher_)
        return TransitionCallbackReturn.SUCCESS
```

---

## Summary

| Concept | Key Idea |
|---------|----------|
| ROS2 vs ROS1 | DDS middleware, no rosmaster, QoS, real-time capable, cross-platform |
| Node | Single-responsibility computation unit; the building block |
| Topic (pub/sub) | Asynchronous, many-to-many streaming data (sensors, commands) |
| Service (req/res) | Synchronous, one-to-one request-response (queries, triggers) |
| Action (goal-based) | Long-running tasks with feedback and cancellation (navigation, manipulation) |
| QoS | Configurable reliability, durability, history; must be compatible between pub and sub |
| Messages/interfaces | Typed data structures (.msg, .srv, .action) for communication |
| Launch files | Python scripts to configure, start, and orchestrate multiple nodes |
| Parameters | Runtime-configurable values with declaration, validation, and dynamic update |
| TF2 | Transform tree for managing coordinate frames; enables frame-agnostic sensor processing |
| Lifecycle nodes | State machine for deterministic startup/shutdown ordering |
| Colcon | Build system for ROS2 workspaces; supports Python and C++ packages |

---

## Exercises

1. **Publisher-subscriber pair**: Create two ROS2 nodes: (a) a publisher that generates simulated odometry data (Odometry message at 10 Hz), and (b) a subscriber that receives it and computes the total distance traveled. Test with `ros2 topic echo` and `ros2 topic hz`.

2. **Service for path planning**: Create a service server that accepts a start and goal position (as Point messages) and returns a list of waypoints (as a Path message). The server should compute a simple straight-line path with 10 intermediate points. Create a client that calls the service and prints the path.

3. **TF tree construction**: Build a TF tree for a mobile robot with: `map → odom → base_link → laser_frame` and `base_link → camera_frame`. Publish `odom → base_link` dynamically (simulated circular motion) and all others as static transforms. Verify with `ros2 run tf2_tools view_frames`.

4. **QoS mismatch debugging**: Create a publisher with BEST_EFFORT QoS and a subscriber with RELIABLE QoS on the same topic. Observe that no messages are received. Fix the QoS to be compatible and verify communication resumes.

5. **Launch file with parameters**: Create a launch file that starts three nodes (a simulated sensor, a processor, and a logger) with configurable parameters loaded from a YAML file. Include a launch argument to switch between simulation and real hardware modes.

---

## Further Reading

- ROS2 Official Documentation: [docs.ros.org](https://docs.ros.org/) (Tutorials, API reference, design documents)
- Macenski, S. et al. "Robot Operating System 2: Design, Architecture, and Uses in the Wild." *Science Robotics*, 2022. (ROS2 architecture paper)
- ROS2 DDS Concepts: [design.ros2.org/articles/ros_on_dds.html](https://design.ros2.org/articles/ros_on_dds.html) (DDS middleware design)
- TF2 Tutorial: [docs.ros.org/en/rolling/Tutorials/Intermediate/Tf2](https://docs.ros.org/en/rolling/Tutorials/Intermediate/Tf2/) (Official TF2 tutorials)

---

[← Previous: SLAM](12_SLAM.md) | [Next: ROS2 Navigation Stack →](14_ROS2_Navigation.md)
