"""
Exercises for Lesson 13: ROS2 Fundamentals
Topic: Robotics
Solutions to practice problems from the lesson.

Note: These exercises simulate ROS2 concepts in pure Python since
actual ROS2 requires a ROS2 installation. The implementations demonstrate
the core concepts (pub/sub, services, TF, QoS) without the rclpy dependency.
"""

import numpy as np
import time
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from enum import Enum


# ---- Simulated ROS2 primitives ----

class QoSReliability(Enum):
    RELIABLE = "reliable"
    BEST_EFFORT = "best_effort"


@dataclass
class QoSProfile:
    reliability: QoSReliability = QoSReliability.RELIABLE
    depth: int = 10


@dataclass
class Odometry:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    omega: float = 0.0
    timestamp: float = 0.0


@dataclass
class Point:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Path:
    waypoints: List[Point] = field(default_factory=list)


class SimpleTopic:
    """Simulated ROS2 topic with QoS matching."""
    def __init__(self, name: str):
        self.name = name
        self.subscribers: List[tuple] = []  # (callback, qos)
        self.publisher_qos: Optional[QoSProfile] = None

    def register_publisher(self, qos: QoSProfile):
        self.publisher_qos = qos

    def register_subscriber(self, callback: Callable, qos: QoSProfile):
        self.subscribers.append((callback, qos))

    def publish(self, msg):
        if self.publisher_qos is None:
            return
        for callback, sub_qos in self.subscribers:
            # QoS compatibility check
            if (self.publisher_qos.reliability == QoSReliability.BEST_EFFORT and
                    sub_qos.reliability == QoSReliability.RELIABLE):
                continue  # incompatible — subscriber won't receive
            callback(msg)


class TopicBroker:
    """Central broker for simulated topics."""
    def __init__(self):
        self.topics: Dict[str, SimpleTopic] = {}

    def get_topic(self, name: str) -> SimpleTopic:
        if name not in self.topics:
            self.topics[name] = SimpleTopic(name)
        return self.topics[name]


def exercise_1():
    """
    Exercise 1: Publisher-Subscriber Pair
    Publisher sends simulated odometry at 10Hz, subscriber computes total distance.
    """
    broker = TopicBroker()
    topic = broker.get_topic("/odom")
    qos = QoSProfile(reliability=QoSReliability.RELIABLE, depth=10)

    # Publisher node
    odom_data = []
    dt = 0.1  # 10 Hz
    v = 1.0   # 1 m/s forward
    omega = 0.2  # slight turn

    x, y, theta = 0.0, 0.0, 0.0
    for i in range(50):  # 5 seconds of data
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        theta += omega * dt
        odom_data.append(Odometry(x=x, y=y, theta=theta, vx=v, vy=0, omega=omega,
                                   timestamp=i * dt))

    # Subscriber node: compute total distance
    total_distance = 0.0
    prev_odom = None
    received_count = 0

    def odom_callback(msg):
        nonlocal total_distance, prev_odom, received_count
        received_count += 1
        if prev_odom is not None:
            dx = msg.x - prev_odom.x
            dy = msg.y - prev_odom.y
            total_distance += np.sqrt(dx**2 + dy**2)
        prev_odom = msg

    topic.register_publisher(qos)
    topic.register_subscriber(odom_callback, qos)

    # Simulate publishing
    for odom in odom_data:
        topic.publish(odom)

    print("Publisher-Subscriber: Odometry Distance Calculator")
    print(f"  Published {len(odom_data)} messages at 10 Hz")
    print(f"  Received {received_count} messages")
    print(f"  Total distance traveled: {total_distance:.4f} m")
    print(f"  Final position: ({odom_data[-1].x:.4f}, {odom_data[-1].y:.4f})")
    print(f"  Euclidean from start: {np.sqrt(odom_data[-1].x**2 + odom_data[-1].y**2):.4f} m")
    print(f"  (Distance > Euclidean because robot curves)")


def exercise_2():
    """
    Exercise 2: Service for Path Planning
    Service accepts start/goal, returns straight-line path with 10 waypoints.
    """
    def path_planning_service(start: Point, goal: Point, n_points: int = 10) -> Path:
        """Service handler: compute straight-line path."""
        waypoints = []
        for i in range(n_points + 1):
            t = i / n_points
            wp = Point(
                x=start.x + t * (goal.x - start.x),
                y=start.y + t * (goal.y - start.y),
                z=start.z + t * (goal.z - start.z),
            )
            waypoints.append(wp)
        return Path(waypoints=waypoints)

    # Client calls the service
    start = Point(x=0.0, y=0.0, z=0.0)
    goal = Point(x=5.0, y=3.0, z=0.0)

    print("Service: Path Planning")
    print(f"  Request: start=({start.x}, {start.y}), goal=({goal.x}, {goal.y})")

    result = path_planning_service(start, goal, n_points=10)

    print(f"  Response: {len(result.waypoints)} waypoints")
    for i, wp in enumerate(result.waypoints):
        print(f"    [{i:>2}] ({wp.x:6.2f}, {wp.y:6.2f}, {wp.z:6.2f})")

    # Verify path length
    total_len = sum(
        np.sqrt((result.waypoints[i+1].x - result.waypoints[i].x)**2 +
                (result.waypoints[i+1].y - result.waypoints[i].y)**2)
        for i in range(len(result.waypoints) - 1)
    )
    expected_len = np.sqrt((goal.x - start.x)**2 + (goal.y - start.y)**2)
    print(f"  Path length: {total_len:.4f} m (expected: {expected_len:.4f} m)")


def exercise_3():
    """
    Exercise 3: TF Tree Construction
    map -> odom -> base_link -> laser_frame, base_link -> camera_frame.
    """
    def make_transform(x, y, z, theta):
        """Create 4x4 homogeneous transform (rotation about z)."""
        c, s = np.cos(theta), np.sin(theta)
        return np.array([
            [c, -s, 0, x],
            [s, c, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1]
        ])

    # Static transforms
    T_base_to_laser = make_transform(0.2, 0.0, 0.15, 0.0)   # laser 20cm forward, 15cm up
    T_base_to_camera = make_transform(0.15, 0.05, 0.3, 0.0)  # camera slightly offset
    T_map_to_odom = make_transform(0.0, 0.0, 0.0, 0.0)       # initially aligned

    # Dynamic transform: odom -> base_link (circular motion)
    print("TF Tree Construction:")
    print("  Tree: map -> odom -> base_link -> laser_frame")
    print("                               \\-> camera_frame")
    print()

    # Simulate circular motion
    dt = 0.1
    t_values = [0, 1, 2, 3, 4, 5]
    radius = 2.0
    omega = 0.5

    print(f"  {'Time':>5} | {'base_link (x,y)':>20} | {'laser (x,y)':>20} | {'camera (x,y)':>20}")
    print("  " + "-" * 75)

    for t in t_values:
        theta = omega * t
        x = radius * np.sin(theta)
        y = radius * (1 - np.cos(theta))

        T_odom_to_base = make_transform(x, y, 0, theta)

        # Compute all frames in map coordinates
        T_map_base = T_map_to_odom @ T_odom_to_base
        T_map_laser = T_map_base @ T_base_to_laser
        T_map_camera = T_map_base @ T_base_to_camera

        base_pos = T_map_base[:2, 3]
        laser_pos = T_map_laser[:2, 3]
        camera_pos = T_map_camera[:2, 3]

        print(f"  {t:>5.1f} | ({base_pos[0]:>8.4f}, {base_pos[1]:>8.4f}) | "
              f"({laser_pos[0]:>8.4f}, {laser_pos[1]:>8.4f}) | "
              f"({camera_pos[0]:>8.4f}, {camera_pos[1]:>8.4f})")

    print(f"\n  Static transforms remain fixed; dynamic odom->base_link updates.")
    print(f"  All child frames move rigidly with base_link.")


def exercise_4():
    """
    Exercise 4: QoS Mismatch Debugging
    Publisher with BEST_EFFORT, Subscriber with RELIABLE — incompatible.
    """
    broker = TopicBroker()
    topic = broker.get_topic("/sensor_data")

    received_messages = []

    def subscriber_callback(msg):
        received_messages.append(msg)

    # Scenario 1: Mismatched QoS
    pub_qos = QoSProfile(reliability=QoSReliability.BEST_EFFORT)
    sub_qos = QoSProfile(reliability=QoSReliability.RELIABLE)

    topic.register_publisher(pub_qos)
    topic.register_subscriber(subscriber_callback, sub_qos)

    for i in range(10):
        topic.publish(f"message_{i}")

    print("QoS Mismatch Debugging:")
    print(f"  Publisher: BEST_EFFORT, Subscriber: RELIABLE")
    print(f"  Messages published: 10")
    print(f"  Messages received: {len(received_messages)}")
    print(f"  Result: INCOMPATIBLE — RELIABLE subscriber cannot receive from")
    print(f"  BEST_EFFORT publisher. The subscriber demands guaranteed delivery")
    print(f"  but the publisher does not provide it.")

    # Fix: make them compatible
    received_messages.clear()
    topic2 = broker.get_topic("/sensor_data_fixed")
    # Option A: both BEST_EFFORT
    topic2.register_publisher(QoSProfile(reliability=QoSReliability.BEST_EFFORT))
    topic2.register_subscriber(subscriber_callback,
                                QoSProfile(reliability=QoSReliability.BEST_EFFORT))
    for i in range(10):
        topic2.publish(f"message_{i}")

    print(f"\n  Fix: Both BEST_EFFORT")
    print(f"  Messages received: {len(received_messages)}")

    # Option B: both RELIABLE
    received_messages.clear()
    topic3 = broker.get_topic("/sensor_data_reliable")
    topic3.register_publisher(QoSProfile(reliability=QoSReliability.RELIABLE))
    topic3.register_subscriber(subscriber_callback,
                                QoSProfile(reliability=QoSReliability.RELIABLE))
    for i in range(10):
        topic3.publish(f"message_{i}")

    print(f"  Fix: Both RELIABLE")
    print(f"  Messages received: {len(received_messages)}")

    print(f"\n  QoS compatibility rules:")
    print(f"  Pub RELIABLE + Sub RELIABLE = OK")
    print(f"  Pub RELIABLE + Sub BEST_EFFORT = OK")
    print(f"  Pub BEST_EFFORT + Sub BEST_EFFORT = OK")
    print(f"  Pub BEST_EFFORT + Sub RELIABLE = FAIL (mismatch)")


def exercise_5():
    """
    Exercise 5: Launch File with Parameters (simulated)
    Three nodes with YAML-loaded parameters and simulation/real mode.
    """
    # Simulated YAML config
    yaml_config = {
        "sensor_node": {
            "rate_hz": 10,
            "noise_std": 0.1,
            "sensor_type": "lidar",
        },
        "processor_node": {
            "filter_type": "kalman",
            "window_size": 5,
        },
        "logger_node": {
            "log_level": "info",
            "output_file": "/tmp/robot_log.txt",
        }
    }

    class SimulatedNode:
        def __init__(self, name, params, mode="simulation"):
            self.name = name
            self.params = params
            self.mode = mode
            self.running = False

        def start(self):
            self.running = True
            return f"[{self.name}] Started in {self.mode} mode with params: {self.params}"

    # Simulated launch file logic
    def launch(mode="simulation"):
        nodes = []
        for node_name, params in yaml_config.items():
            node = SimulatedNode(node_name, params, mode)
            nodes.append(node)
        return nodes

    print("Launch File with Parameters:")
    print(f"\n  YAML Configuration:")
    for node_name, params in yaml_config.items():
        print(f"    {node_name}:")
        for k, v in params.items():
            print(f"      {k}: {v}")

    # Launch in simulation mode
    print(f"\n  Launching in SIMULATION mode:")
    nodes = launch(mode="simulation")
    for node in nodes:
        msg = node.start()
        print(f"    {msg}")

    # Launch in hardware mode
    print(f"\n  Launching in HARDWARE mode:")
    nodes = launch(mode="hardware")
    for node in nodes:
        msg = node.start()
        print(f"    {msg}")

    print(f"\n  In real ROS2, this would be a Python launch file:")
    print(f"    - LaunchDescription with DeclareLaunchArgument for 'mode'")
    print(f"    - Node() entries for each node with parameters from YAML")
    print(f"    - LaunchConfiguration to read the mode argument")
    print(f"    - Conditional logic (IfCondition) for simulation vs hardware")


if __name__ == "__main__":
    print("=" * 70)
    print("Lesson 13: ROS2 Fundamentals — Exercise Solutions")
    print("=" * 70)

    print("\n--- Exercise 1: Publisher-Subscriber Pair ---")
    exercise_1()

    print("\n--- Exercise 2: Service for Path Planning ---")
    exercise_2()

    print("\n--- Exercise 3: TF Tree Construction ---")
    exercise_3()

    print("\n--- Exercise 4: QoS Mismatch Debugging ---")
    exercise_4()

    print("\n--- Exercise 5: Launch File with Parameters ---")
    exercise_5()

    print("\n" + "=" * 70)
    print("All exercises completed!")
    print("=" * 70)
