# 13. ROS2 기초

[← 이전: SLAM](12_SLAM.md) | [다음: ROS2 내비게이션 스택 →](14_ROS2_Navigation.md)

---

## 학습 목표

1. ROS1과 ROS2의 핵심 차이점과 마이그레이션이 이루어진 이유를 설명한다
2. ROS2의 핵심 개념인 노드(node), 토픽(topic), 서비스(service), 액션(action)과 그 통신 패턴을 이해한다
3. 사용자 정의 메시지(message), 서비스(service), 액션(action) 인터페이스를 정의한다
4. ROS2 런치(launch) 파일을 작성하고 노드 파라미터를 관리한다
5. 다중 프레임 로봇 시스템에서 TF2를 사용하여 좌표 프레임을 관리한다
6. colcon 빌드 시스템을 사용하여 ROS2 패키지를 빌드하고 구성한다

---

이전 단원들에서 우리는 로봇공학의 수학적 기초인 기구학(kinematics), 동역학(dynamics), 제어(control), 인지(perception), SLAM을 다루었다. 그런데 실제 로봇을 만들려면 이 모든 구성 요소를 하나의 작동하는 소프트웨어 시스템으로 통합해야 한다. 제어 알고리즘은 센서 데이터를 수신해야 하고, 경로 계획기(planner)는 궤적을 컨트롤러에 전달해야 하며, SLAM 시스템은 지도를 내비게이터와 공유해야 한다 — 이 모든 것이 실시간으로, 안정적으로, 구성 요소 간 강한 결합(tight coupling) 없이 이루어져야 한다.

이 문제를 해결하는 것이 바로 **ROS2(Robot Operating System 2)**이다. ROS2는 전통적인 의미의 운영체제가 아니라, 모듈화되고 재사용 가능한 로봇 소프트웨어를 만들기 위한 통신 인프라, 도구, 관례를 제공하는 미들웨어 프레임워크이다. ROS2는 로봇공학 연구의 사실상 표준(de facto standard)이 되었으며 산업 분야에서도 점점 더 많이 채택되고 있다. 시뮬레이션을 넘어 실제 시스템을 구축하려는 로봇 공학자라면 ROS2 학습은 필수이다.

> **비유**: ROS2 노드는 웹 애플리케이션의 마이크로서비스(microservice)와 같다 — 각 노드는 하나의 역할(카메라 드라이버, SLAM, 내비게이션 계획기)을 담당하며, 잘 정의된 인터페이스(토픽, 서비스, 액션)를 통해 통신한다. 전체 애플리케이션을 다시 작성하지 않고 하나의 마이크로서비스를 교체할 수 있듯이, ROS2에서도 나머지 시스템을 변경하지 않고 SLAM 알고리즘이나 센서 드라이버를 교체할 수 있다. "ROS2 생태계"는 로봇공학을 위한 앱스토어와 같다: 수천 개의 오픈소스 패키지를 플러그인처럼 바로 사용할 수 있다.

---

## 1. ROS1에서 ROS2로: 왜 재작성했는가?

### 1.1 ROS1의 한계

ROS1(2007-2020)은 엄청난 성공을 거두었지만 근본적인 아키텍처 한계가 있었다:

| 한계 | 영향 |
|------|------|
| 단일 장애점(rosmaster) | rosmaster가 충돌하면 모든 통신이 끊김 |
| 실시간 지원 없음 | 안전-중요 제어의 타이밍을 보장할 수 없음 |
| TCP/UDP만 지원 | 표준화된 QoS 없음, 공유 메모리 전송 없음 |
| 사실상 Linux 전용 | 임베디드 시스템이나 Windows에서 실행 불가 |
| 내장 보안 없음 | 열린 네트워크, 인증 또는 암호화 없음 |
| 라이프사이클 관리 없음 | 노드가 정의되지 않은 순서로 시작/종료 |

### 1.2 ROS2 아키텍처

ROS2는 성숙한 산업용 미들웨어 표준인 **DDS(Data Distribution Service)**를 기반으로 이 모든 문제를 해결한다:

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

주요 개선 사항:
- **중앙 마스터 없음**: 노드가 DDS 멀티캐스트를 통해 자동으로 서로를 탐색
- **서비스 품질(QoS, Quality of Service)**: 신뢰성(reliability), 지속성(durability), 마감(deadline), 히스토리(history) 정책을 설정 가능
- **실시간 가능**: 적절한 DDS 설정과 `rclcpp`로 밀리초 미만의 지연 시간 달성
- **크로스 플랫폼**: Linux, macOS, Windows, 임베디드(micro-ROS)
- **라이프사이클 노드**: 순서 있는 시작/종료를 위한 표준화된 상태 머신
- **보안**: DDS-Security는 인증, 암호화, 접근 제어를 제공

### 1.3 ROS2 배포판

ROS2는 Ubuntu와 같이 시간 기반 릴리스를 사용한다:

| 배포판 | 릴리스 | EOL | Ubuntu |
|--------|--------|-----|--------|
| Humble Hawksbill | 2022년 5월 | 2027년 5월 | 22.04 |
| Iron Irwini | 2023년 5월 | 2024년 11월 | 22.04 |
| Jazzy Jalisco | 2024년 5월 | 2029년 5월 | 24.04 |
| Rolling | 지속 업데이트 | 해당 없음 | 최신 |

프로덕션 환경에는 LTS(장기 지원) 릴리스를 권장한다.

---

## 2. 핵심 개념

### 2.1 노드(Node)

**노드**는 ROS2에서 연산의 기본 단위이다. 각 노드는 특정 기능(카메라 드라이버, SLAM, 컨트롤러 등)을 수행하는 단일 프로세스이다.

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

### 2.2 토픽(Topic) — 발행/구독(Publish/Subscribe)

**토픽**은 비동기적, 다대다 통신을 제공한다. 발행자(publisher)는 이름 있는 토픽에 메시지를 전송하고, 임의의 수의 구독자(subscriber)가 이를 수신할 수 있다.

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

### 2.3 서비스(Service) — 요청/응답(Request/Response)

**서비스**는 동기적, 일대일 통신을 제공한다. 클라이언트가 요청을 보내고 서버가 응답할 때까지 대기한다.

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

### 2.4 액션(Action) — 목표 기반(Goal-Based)

**액션**은 피드백이 필요하고 취소할 수 있는 장시간 실행 작업을 위한 것이다. 목표(goal, 요청), 피드백(feedback, 진행 상황 업데이트), 결과(result)를 결합한다.

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

**사용 사례**: 웨이포인트까지 이동, 물체 집기, 궤적 실행. 이런 작업은 수 초에서 수 분이 걸리며, 호출자는 진행 상황 업데이트를 원한다.

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

## 3. 메시지와 인터페이스

### 3.1 메시지 타입(.msg)

ROS2 메시지는 토픽 통신에 사용되는 데이터 구조를 정의한다. 단순한 타입-필드 문법을 사용하는 `.msg` 파일로 정의된다:

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

### 3.2 사용자 정의 메시지

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

### 3.3 서비스 타입(.srv)

서비스는 `---`로 구분된 요청(request)과 응답(response)으로 구성된다:

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

### 3.4 액션 타입(.action)

액션은 `---`로 구분된 목표(goal), 결과(result), 피드백(feedback)으로 구성된다:

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

## 4. 서비스 품질(QoS, Quality of Service)

### 4.1 QoS가 중요한 이유

데이터 타입에 따라 신뢰성 요구사항이 다르다:

| 데이터 타입 | QoS 선호도 | 이유 |
|------------|-----------|------|
| 카메라 이미지 | 최선 노력(Best effort) | 프레임 드롭은 허용; 지연 시간이 중요 |
| LiDAR 스캔 | 최선 노력(Best effort) | 카메라와 같은 이유 |
| 제어 명령 | 신뢰성(Reliable) | 명령 누락은 위험할 수 있음 |
| 지도 데이터 | 신뢰성, transient local | 늦게 시작한 구독자도 최신 지도가 필요 |
| TF 변환 | 최선 노력, volatile | 높은 전송률, 낮은 지연; 오래된 변환은 무용 |
| 진단 정보 | 신뢰성(Reliable) | 오류 보고를 놓치지 않기 위함 |

### 4.2 QoS 프로파일

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

### 4.3 QoS 호환성

발행자(publisher)와 구독자(subscriber)는 통신하기 위해 **호환 가능한** QoS 설정을 가져야 한다:

| 발행자 | 구독자 | 호환 여부 |
|--------|--------|---------|
| Reliable | Reliable | 가능 |
| Best effort | Best effort | 가능 |
| Reliable | Best effort | 가능 (구독자가 신뢰성을 "무료로" 얻음) |
| Best effort | Reliable | **불가능** (구독자가 발행자가 제공할 수 없는 보장을 요구함) |

**흔한 함정**: RELIABLE QoS를 가진 구독자는 BEST_EFFORT 발행자로부터 데이터를 수신할 수 없다. ROS2에서 "메시지가 수신되지 않음"이 보이면 QoS 불일치가 원인일 가능성이 높다.

---

## 5. 런치(Launch) 시스템

### 5.1 런치 파일

ROS2 런치 파일(Python)은 여러 노드를 설정하고 시작하며, 파라미터를 설정하고 리매핑(remapping)을 정의한다:

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

### 5.2 파라미터(Parameter)

ROS2 파라미터를 통해 노드를 런타임에 설정할 수 있다:

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

## 6. TF2: 변환 관리

### 6.1 변환 트리(Transform Tree)

로봇에는 많은 좌표 프레임이 있다: 베이스(base), 각 조인트(joint), 각 센서, 지도(map), 오도메트리(odometry) 프레임. **TF2**는 모든 프레임 간의 변환 트리를 유지하여, 어떤 노드든 언제든지 임의의 두 프레임 간의 변환을 조회할 수 있도록 한다.

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

### 6.2 정적 변환과 동적 변환

**정적 변환(Static transforms)**: 절대 변하지 않는 고정 관계(예: 로봇에 고정된 위치에 장착된 카메라). 한 번만 발행한다.

**동적 변환(Dynamic transforms)**: 시간에 따라 변하는 관계(예: 조인트 각도, 오도메트리). 높은 빈도로 지속적으로 발행한다.

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

### 6.3 변환 수신

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

## 7. Colcon 빌드 시스템과 패키지 구조

### 7.1 패키지 구조

ROS2 Python 패키지는 다음 구조를 따른다:

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

### 7.2 빌드 및 CLI 필수 명령어

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

## 8. 라이프사이클 노드(Lifecycle Node)

### 8.1 관리형 노드 상태

**라이프사이클 노드**는 결정론적(deterministic) 시작 및 종료를 위한 표준화된 상태 머신을 제공한다:

```
Unconfigured → on_configure() → Inactive → on_activate() → Active
                                    ↑         on_deactivate()  │
                                    └──────────────────────────┘
```

라이프사이클 관리가 필요한 이유는 무엇인가? 복잡한 로봇 시스템에서 노드는 특정 순서로 시작해야 한다(SLAM 이전에 드라이버, 내비게이션 이전에 SLAM). 라이프사이클 노드는 수퍼바이저(supervisor)가 이를 조율할 수 있도록 한다: 모든 노드를 설정(configure)한 다음 순서대로 활성화(activate). 이를 통해 아직 시작되지 않은 소스의 데이터를 사용하려는 노드에서 발생하는 경쟁 조건(race condition)을 방지한다.

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

## 요약

| 개념 | 핵심 내용 |
|------|----------|
| ROS2 vs ROS1 | DDS 미들웨어, rosmaster 없음, QoS, 실시간 가능, 크로스 플랫폼 |
| 노드(Node) | 단일 책임 연산 단위; 기본 구성 요소 |
| 토픽(Topic) (발행/구독) | 비동기적, 다대다 스트리밍 데이터 (센서, 명령) |
| 서비스(Service) (요청/응답) | 동기적, 일대일 요청-응답 (쿼리, 트리거) |
| 액션(Action) (목표 기반) | 피드백과 취소 기능이 있는 장시간 작업 (내비게이션, 매니퓰레이션) |
| QoS | 설정 가능한 신뢰성, 지속성, 히스토리; 발행자-구독자 간 호환 필수 |
| 메시지/인터페이스 | 통신을 위한 타입이 지정된 데이터 구조 (.msg, .srv, .action) |
| 런치 파일 | 여러 노드를 설정, 시작, 조율하는 Python 스크립트 |
| 파라미터 | 선언, 검증, 동적 업데이트가 가능한 런타임 설정 값 |
| TF2 | 좌표 프레임 관리를 위한 변환 트리; 프레임에 독립적인 센서 처리 가능 |
| 라이프사이클 노드 | 결정론적 시작/종료 순서를 위한 상태 머신 |
| Colcon | ROS2 워크스페이스 빌드 시스템; Python 및 C++ 패키지 지원 |

---

## 연습 문제

1. **발행자-구독자 쌍**: ROS2 노드 두 개를 만들어라: (a) 시뮬레이션된 오도메트리 데이터(10 Hz의 Odometry 메시지)를 생성하는 발행자, (b) 이를 수신하여 총 이동 거리를 계산하는 구독자. `ros2 topic echo`와 `ros2 topic hz`로 테스트하라.

2. **경로 계획 서비스**: 시작 위치와 목표 위치(Point 메시지로)를 받아 웨이포인트 목록(Path 메시지로)을 반환하는 서비스 서버를 만들어라. 서버는 10개의 중간 지점이 있는 단순한 직선 경로를 계산해야 한다. 서비스를 호출하여 경로를 출력하는 클라이언트를 만들어라.

3. **TF 트리 구성**: 다음 프레임을 가진 이동 로봇의 TF 트리를 만들어라: `map → odom → base_link → laser_frame`과 `base_link → camera_frame`. `odom → base_link`를 동적으로 (시뮬레이션된 원형 이동으로) 발행하고, 나머지는 정적 변환으로 발행하라. `ros2 run tf2_tools view_frames`으로 확인하라.

4. **QoS 불일치 디버깅**: 같은 토픽에 BEST_EFFORT QoS를 가진 발행자와 RELIABLE QoS를 가진 구독자를 만들어라. 메시지가 수신되지 않는 것을 관찰하라. QoS를 호환 가능하게 수정하고 통신이 재개되는지 확인하라.

5. **파라미터가 있는 런치 파일**: 세 개의 노드(시뮬레이션 센서, 프로세서, 로거)를 YAML 파일에서 로드한 설정 가능한 파라미터로 시작하는 런치 파일을 만들어라. 시뮬레이션과 실제 하드웨어 모드 간에 전환할 수 있는 런치 인수를 포함하라.

---

## 더 읽을거리

- ROS2 공식 문서: [docs.ros.org](https://docs.ros.org/) (튜토리얼, API 레퍼런스, 설계 문서)
- Macenski, S. et al. "Robot Operating System 2: Design, Architecture, and Uses in the Wild." *Science Robotics*, 2022. (ROS2 아키텍처 논문)
- ROS2 DDS 개념: [design.ros2.org/articles/ros_on_dds.html](https://design.ros2.org/articles/ros_on_dds.html) (DDS 미들웨어 설계)
- TF2 튜토리얼: [docs.ros.org/en/rolling/Tutorials/Intermediate/Tf2](https://docs.ros.org/en/rolling/Tutorials/Intermediate/Tf2/) (공식 TF2 튜토리얼)

---

[← 이전: SLAM](12_SLAM.md) | [다음: ROS2 내비게이션 스택 →](14_ROS2_Navigation.md)
