"""
ROS2 Communication Patterns: A Pure-Python Simulation
======================================================
Simulate core ROS2 concepts without requiring rclpy or a ROS2 installation.

ROS2 (Robot Operating System 2) is the de facto middleware for robotics
software. It provides:
  - Publish/Subscribe: asynchronous, decoupled data streaming
  - Services: synchronous request/response RPC
  - Actions: long-running tasks with feedback and cancellation
  - Node lifecycle: managed state transitions for reliable startup/shutdown
  - QoS (Quality of Service): configurable reliability and durability

This example builds a simplified but faithful simulation of these patterns
using pure Python (threading + queues), demonstrating how ROS2 organizes
communication in a multi-node robotic system.

Scenario: A mobile robot with:
  - Lidar sensor node (publishes scans)
  - Odometry node (publishes pose estimates)
  - Path planner node (action server for navigation)
  - Safety monitor node (subscribes to scans, provides emergency stop service)
"""

import numpy as np
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# QoS (Quality of Service) profiles
# ---------------------------------------------------------------------------
class ReliabilityPolicy(Enum):
    """How strictly messages must be delivered."""
    BEST_EFFORT = "best_effort"    # May lose messages (fast, for sensors)
    RELIABLE = "reliable"          # Guarantees delivery (for commands)


class DurabilityPolicy(Enum):
    """Whether late subscribers receive past messages."""
    VOLATILE = "volatile"          # Only future messages
    TRANSIENT_LOCAL = "transient_local"  # Late joiners get last N messages


@dataclass
class QoSProfile:
    """Quality of Service configuration for a topic.

    QoS is one of the most important ROS2 concepts. Mismatched QoS between
    publisher and subscriber is a common source of silent failures.
    A RELIABLE publisher can communicate with a BEST_EFFORT subscriber,
    but not vice versa.
    """
    reliability: ReliabilityPolicy = ReliabilityPolicy.RELIABLE
    durability: DurabilityPolicy = DurabilityPolicy.VOLATILE
    depth: int = 10  # queue size


SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    depth=5
)

DEFAULT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    depth=10
)


# ---------------------------------------------------------------------------
# Message types (simplified ROS2 message equivalents)
# ---------------------------------------------------------------------------
@dataclass
class Header:
    """Standard header for all messages (like std_msgs/Header)."""
    stamp: float = 0.0
    frame_id: str = ""


@dataclass
class LaserScan:
    """Simulated sensor_msgs/LaserScan."""
    header: Header = field(default_factory=Header)
    angle_min: float = -np.pi / 2
    angle_max: float = np.pi / 2
    ranges: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class Odometry:
    """Simulated nav_msgs/Odometry."""
    header: Header = field(default_factory=Header)
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    v: float = 0.0
    omega: float = 0.0


@dataclass
class PoseTarget:
    """Navigation goal (like geometry_msgs/PoseStamped)."""
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0


# ---------------------------------------------------------------------------
# Core middleware: simplified DDS-like communication
# ---------------------------------------------------------------------------
class MessageBus:
    """Simulated DDS (Data Distribution Service) middleware.

    In real ROS2, DDS handles discovery, serialization, and transport.
    Our simulation uses in-process queues with QoS enforcement.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._topics: dict[str, list[dict]] = defaultdict(list)
        self._history: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))

    def subscribe(self, topic: str, callback: Callable, qos: QoSProfile):
        """Register a subscription callback for a topic."""
        with self._lock:
            sub_info = {"callback": callback, "qos": qos}
            self._topics[topic].append(sub_info)

            # Deliver historical messages if TRANSIENT_LOCAL
            if qos.durability == DurabilityPolicy.TRANSIENT_LOCAL:
                for msg in self._history[topic]:
                    callback(msg)

    def publish(self, topic: str, message: Any, qos: QoSProfile):
        """Publish a message to a topic.

        QoS matching rules (simplified):
          - RELIABLE pub + RELIABLE sub: always delivered
          - RELIABLE pub + BEST_EFFORT sub: always delivered
          - BEST_EFFORT pub + BEST_EFFORT sub: may drop (simulated)
          - BEST_EFFORT pub + RELIABLE sub: INCOMPATIBLE (skipped)
        """
        with self._lock:
            # Store in history for TRANSIENT_LOCAL durability
            self._history[topic].append(message)

            for sub in self._topics[topic]:
                sub_qos = sub["qos"]

                # QoS compatibility check
                if (qos.reliability == ReliabilityPolicy.BEST_EFFORT
                        and sub_qos.reliability == ReliabilityPolicy.RELIABLE):
                    continue  # incompatible — skip

                # Simulate best-effort drops (10% drop rate)
                if (qos.reliability == ReliabilityPolicy.BEST_EFFORT
                        and sub_qos.reliability == ReliabilityPolicy.BEST_EFFORT):
                    if np.random.random() < 0.1:
                        continue  # simulate message drop

                sub["callback"](message)


# Global message bus (like ROS2 DDS domain)
_bus = MessageBus()


# ---------------------------------------------------------------------------
# Service server/client (synchronous request-response)
# ---------------------------------------------------------------------------
class ServiceServer:
    """Simulated ROS2 service server.

    Services are synchronous: the client sends a request and blocks
    until the server returns a response. Used for quick computations
    like coordinate transforms or parameter queries.
    """

    _registry: dict[str, "ServiceServer"] = {}

    def __init__(self, name: str, handler: Callable):
        self.name = name
        self.handler = handler
        ServiceServer._registry[name] = self

    def handle(self, request):
        return self.handler(request)


class ServiceClient:
    """Simulated ROS2 service client."""

    def __init__(self, name: str):
        self.name = name

    def call(self, request):
        """Send request and wait for response (synchronous)."""
        if self.name not in ServiceServer._registry:
            raise RuntimeError(f"Service '{self.name}' not available")
        server = ServiceServer._registry[self.name]
        return server.handle(request)


# ---------------------------------------------------------------------------
# Action server/client (long-running tasks with feedback)
# ---------------------------------------------------------------------------
class ActionStatus(Enum):
    """Status of an action goal."""
    PENDING = "pending"
    ACTIVE = "active"
    SUCCEEDED = "succeeded"
    CANCELED = "canceled"
    FAILED = "failed"


@dataclass
class ActionGoalHandle:
    """Handle to track and control an active goal."""
    goal: Any
    status: ActionStatus = ActionStatus.PENDING
    result: Any = None
    feedback_log: list = field(default_factory=list)
    _cancel_requested: bool = False

    def cancel(self):
        self._cancel_requested = True

    @property
    def is_cancel_requested(self):
        return self._cancel_requested


class ActionServer:
    """Simulated ROS2 action server.

    Actions are for long-running tasks (e.g., navigate to a point):
      - The client sends a goal
      - The server provides periodic feedback (e.g., distance remaining)
      - The client can cancel the goal at any time
      - The server returns a result when done

    This is the pattern used for navigation, manipulation, etc.
    """

    _registry: dict[str, "ActionServer"] = {}

    def __init__(self, name: str, execute_callback: Callable):
        self.name = name
        self.execute_callback = execute_callback
        ActionServer._registry[name] = self

    def accept_goal(self, goal) -> ActionGoalHandle:
        handle = ActionGoalHandle(goal=goal, status=ActionStatus.ACTIVE)
        return handle

    def execute(self, goal_handle: ActionGoalHandle):
        """Execute the action (runs in a separate thread in real ROS2)."""
        self.execute_callback(goal_handle)


class ActionClient:
    """Simulated ROS2 action client."""

    def __init__(self, name: str):
        self.name = name

    def send_goal(self, goal) -> ActionGoalHandle:
        if self.name not in ActionServer._registry:
            raise RuntimeError(f"Action server '{self.name}' not available")
        server = ActionServer._registry[self.name]
        handle = server.accept_goal(goal)
        server.execute(handle)
        return handle


# ---------------------------------------------------------------------------
# Node lifecycle (managed nodes)
# ---------------------------------------------------------------------------
class LifecycleState(Enum):
    """ROS2 managed node lifecycle states.

    The lifecycle pattern ensures orderly startup and shutdown:
      UNCONFIGURED -> INACTIVE -> ACTIVE -> (FINALIZED or back to INACTIVE)

    This prevents nodes from publishing before they are fully configured,
    which is critical for safety-critical systems.
    """
    UNCONFIGURED = "unconfigured"
    INACTIVE = "inactive"
    ACTIVE = "active"
    FINALIZED = "finalized"


class LifecycleNode:
    """Base class for managed nodes with lifecycle transitions."""

    def __init__(self, name: str):
        self.name = name
        self.state = LifecycleState.UNCONFIGURED
        self._log = []

    def _log_transition(self, from_state, to_state, success):
        entry = f"[{self.name}] {from_state.value} -> {to_state.value}"
        if not success:
            entry += " (FAILED)"
        self._log.append(entry)

    def configure(self) -> bool:
        """Transition: UNCONFIGURED -> INACTIVE."""
        if self.state != LifecycleState.UNCONFIGURED:
            return False
        success = self.on_configure()
        if success:
            self.state = LifecycleState.INACTIVE
        self._log_transition(LifecycleState.UNCONFIGURED,
                             LifecycleState.INACTIVE, success)
        return success

    def activate(self) -> bool:
        """Transition: INACTIVE -> ACTIVE."""
        if self.state != LifecycleState.INACTIVE:
            return False
        success = self.on_activate()
        if success:
            self.state = LifecycleState.ACTIVE
        self._log_transition(LifecycleState.INACTIVE,
                             LifecycleState.ACTIVE, success)
        return success

    def deactivate(self) -> bool:
        """Transition: ACTIVE -> INACTIVE."""
        if self.state != LifecycleState.ACTIVE:
            return False
        success = self.on_deactivate()
        if success:
            self.state = LifecycleState.INACTIVE
        self._log_transition(LifecycleState.ACTIVE,
                             LifecycleState.INACTIVE, success)
        return success

    def shutdown(self) -> bool:
        """Transition: any -> FINALIZED."""
        success = self.on_shutdown()
        prev = self.state
        self.state = LifecycleState.FINALIZED
        self._log_transition(prev, LifecycleState.FINALIZED, success)
        return success

    # Override these in subclasses
    def on_configure(self) -> bool:
        return True

    def on_activate(self) -> bool:
        return True

    def on_deactivate(self) -> bool:
        return True

    def on_shutdown(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Robot nodes
# ---------------------------------------------------------------------------
class LidarNode(LifecycleNode):
    """Simulated lidar sensor node.

    Publishes LaserScan messages at a fixed rate. Uses BEST_EFFORT QoS
    since sensor data is high-frequency and occasional drops are acceptable.
    """

    def __init__(self):
        super().__init__("lidar_node")
        self.scan_count = 0
        self.n_beams = 36
        self.pub_qos = SENSOR_QOS

    def on_configure(self) -> bool:
        print(f"  [{self.name}] Configured: {self.n_beams} beams, "
              f"QoS={self.pub_qos.reliability.value}")
        return True

    def on_activate(self) -> bool:
        print(f"  [{self.name}] Activated: publishing scans")
        return True

    def publish_scan(self, sim_time: float):
        """Generate and publish a simulated laser scan."""
        if self.state != LifecycleState.ACTIVE:
            return

        # Simulate range measurements with noise and some obstacles
        angles = np.linspace(-np.pi / 2, np.pi / 2, self.n_beams)
        ranges = 5.0 * np.ones(self.n_beams)  # default: max range

        # Simulate an obstacle in front
        obstacle_angle = 0.1 * np.sin(sim_time)
        obstacle_dist = 2.0 + 0.5 * np.sin(0.3 * sim_time)
        mask = np.abs(angles - obstacle_angle) < 0.3
        ranges[mask] = obstacle_dist + np.random.normal(0, 0.05, np.sum(mask))

        scan = LaserScan(
            header=Header(stamp=sim_time, frame_id="lidar_frame"),
            ranges=ranges
        )
        _bus.publish("/scan", scan, self.pub_qos)
        self.scan_count += 1


class OdometryNode(LifecycleNode):
    """Simulated odometry node.

    Publishes the robot's estimated pose from wheel encoders.
    Uses RELIABLE QoS since pose estimates are critical for navigation.
    """

    def __init__(self):
        super().__init__("odom_node")
        self.x, self.y, self.theta = 0.0, 0.0, 0.0
        self.pub_qos = DEFAULT_QOS

    def on_configure(self) -> bool:
        print(f"  [{self.name}] Configured: QoS={self.pub_qos.reliability.value}")
        return True

    def on_activate(self) -> bool:
        print(f"  [{self.name}] Activated: publishing odometry")
        return True

    def publish_odom(self, sim_time: float, v: float, omega: float, dt: float):
        """Update pose estimate and publish odometry."""
        if self.state != LifecycleState.ACTIVE:
            return

        # Simple dead reckoning with noise
        self.theta += omega * dt + np.random.normal(0, 0.001)
        self.x += v * np.cos(self.theta) * dt + np.random.normal(0, 0.005)
        self.y += v * np.sin(self.theta) * dt + np.random.normal(0, 0.005)

        odom = Odometry(
            header=Header(stamp=sim_time, frame_id="odom"),
            x=self.x, y=self.y, theta=self.theta,
            v=v, omega=omega
        )
        _bus.publish("/odom", odom, self.pub_qos)


class SafetyMonitorNode(LifecycleNode):
    """Safety monitor: subscribes to scans, provides emergency stop service.

    This node demonstrates two patterns:
      1. Subscription: listens to /scan for obstacle detection
      2. Service server: provides /emergency_stop service

    Safety nodes are critical and use RELIABLE QoS for all communication.
    """

    def __init__(self):
        super().__init__("safety_monitor")
        self.min_distance = float('inf')
        self.obstacle_threshold = 1.0
        self.emergency_active = False
        self.scan_callbacks_received = 0

    def on_configure(self) -> bool:
        # Subscribe to scan topic
        _bus.subscribe("/scan", self._scan_callback, DEFAULT_QOS)

        # Provide emergency stop service
        ServiceServer("/emergency_stop", self._handle_emergency_stop)

        print(f"  [{self.name}] Configured: subscribed to /scan, "
              f"serving /emergency_stop")
        return True

    def on_activate(self) -> bool:
        print(f"  [{self.name}] Activated: monitoring safety")
        return True

    def _scan_callback(self, scan: LaserScan):
        """Process incoming laser scan for obstacle detection."""
        if self.state != LifecycleState.ACTIVE:
            return
        self.scan_callbacks_received += 1
        self.min_distance = float(np.min(scan.ranges))

        if self.min_distance < self.obstacle_threshold:
            self.emergency_active = True

    def _handle_emergency_stop(self, request):
        """Handle emergency stop service call."""
        if request.get("activate"):
            self.emergency_active = True
            return {"success": True, "message": "Emergency stop activated"}
        else:
            self.emergency_active = False
            return {"success": True, "message": "Emergency stop deactivated"}


class NavigationNode(LifecycleNode):
    """Path planner node with action server for navigation goals.

    Demonstrates the action pattern:
      - Accepts navigation goals (target poses)
      - Provides feedback (distance remaining)
      - Supports cancellation
      - Returns result (final pose)
    """

    def __init__(self):
        super().__init__("navigation_node")
        self.current_pose = None
        self.odom_received = 0

    def on_configure(self) -> bool:
        # Subscribe to odometry
        _bus.subscribe("/odom", self._odom_callback, DEFAULT_QOS)

        # Action server for navigation
        ActionServer("/navigate_to_pose", self._execute_navigation)

        print(f"  [{self.name}] Configured: subscribed to /odom, "
              f"serving /navigate_to_pose action")
        return True

    def on_activate(self) -> bool:
        print(f"  [{self.name}] Activated: ready for navigation goals")
        return True

    def _odom_callback(self, odom: Odometry):
        """Update current pose from odometry."""
        self.current_pose = (odom.x, odom.y, odom.theta)
        self.odom_received += 1

    def _execute_navigation(self, goal_handle: ActionGoalHandle):
        """Execute navigation to target pose (simulated).

        In real ROS2, this would compute a path and send velocity commands.
        Here we simulate progress toward the goal with feedback.
        """
        target = goal_handle.goal
        remaining = np.sqrt(
            (target.x - (self.current_pose[0] if self.current_pose else 0)) ** 2
            + (target.y - (self.current_pose[1] if self.current_pose else 0)) ** 2
        )

        # Simulate navigation progress
        n_steps = min(10, max(3, int(remaining / 0.5)))
        for step in range(n_steps):
            if goal_handle.is_cancel_requested:
                goal_handle.status = ActionStatus.CANCELED
                goal_handle.result = {"success": False, "message": "Canceled"}
                return

            progress = (step + 1) / n_steps
            dist_remaining = remaining * (1 - progress)
            feedback = {
                "distance_remaining": dist_remaining,
                "progress_pct": progress * 100
            }
            goal_handle.feedback_log.append(feedback)

        goal_handle.status = ActionStatus.SUCCEEDED
        goal_handle.result = {
            "success": True,
            "final_x": target.x,
            "final_y": target.y,
            "message": f"Reached target ({target.x:.1f}, {target.y:.1f})"
        }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def demo_ros2_concepts():
    """Demonstrate ROS2 communication patterns in a simulated robot system."""
    print("=" * 60)
    print("ROS2 Communication Patterns Simulation")
    print("=" * 60)

    np.random.seed(42)

    # --- 1. Node Lifecycle ---
    print("\n--- 1. Node Lifecycle Management ---")
    print("Nodes transition: UNCONFIGURED -> INACTIVE -> ACTIVE\n")

    lidar = LidarNode()
    odom = OdometryNode()
    safety = SafetyMonitorNode()
    nav = NavigationNode()

    nodes = [lidar, odom, safety, nav]

    # Configure all nodes
    print("Configuring nodes:")
    for node in nodes:
        node.configure()

    # Activate all nodes
    print("\nActivating nodes:")
    for node in nodes:
        node.activate()

    print(f"\nAll nodes active: {all(n.state == LifecycleState.ACTIVE for n in nodes)}")

    # --- 2. Publish/Subscribe ---
    print("\n--- 2. Publish/Subscribe Communication ---")
    print("Running simulated sensor loop (20 steps)...\n")

    dt = 0.1
    v, omega = 0.5, 0.1  # forward velocity and angular velocity

    for step in range(20):
        sim_time = step * dt
        lidar.publish_scan(sim_time)
        odom.publish_odom(sim_time, v, omega, dt)

    print(f"  Lidar published: {lidar.scan_count} scans")
    print(f"  Safety monitor received: {safety.scan_callbacks_received} scan callbacks")
    print(f"  Navigation received: {nav.odom_received} odom callbacks")
    print(f"  Min obstacle distance: {safety.min_distance:.2f} m")
    print(f"  Emergency stop active: {safety.emergency_active}")

    # Note on QoS: BEST_EFFORT lidar -> RELIABLE safety subscription
    # Some messages may be dropped due to QoS incompatibility
    dropped = lidar.scan_count - safety.scan_callbacks_received
    print(f"\n  QoS note: {dropped} scans dropped (BEST_EFFORT pub -> "
          f"RELIABLE sub incompatibility)")

    # --- 3. Service Call ---
    print("\n--- 3. Service Request/Response ---")
    print("Calling /emergency_stop service...\n")

    estop_client = ServiceClient("/emergency_stop")

    response = estop_client.call({"activate": True})
    print(f"  Request:  activate=True")
    print(f"  Response: {response}")
    print(f"  Safety emergency_active: {safety.emergency_active}")

    response = estop_client.call({"activate": False})
    print(f"\n  Request:  activate=False")
    print(f"  Response: {response}")
    print(f"  Safety emergency_active: {safety.emergency_active}")

    # --- 4. Action Server/Client ---
    print("\n--- 4. Action Server/Client (Navigation) ---")

    # Send a navigation goal
    goal = PoseTarget(x=3.0, y=2.0, theta=0.0)
    print(f"  Sending goal: navigate to ({goal.x}, {goal.y})")

    nav_client = ActionClient("/navigate_to_pose")
    goal_handle = nav_client.send_goal(goal)

    print(f"  Status: {goal_handle.status.value}")
    print(f"  Result: {goal_handle.result}")
    print(f"  Feedback steps: {len(goal_handle.feedback_log)}")
    for i, fb in enumerate(goal_handle.feedback_log):
        print(f"    Step {i+1}: {fb['progress_pct']:.0f}% complete, "
              f"distance remaining: {fb['distance_remaining']:.2f}")

    # --- 5. Action Cancellation ---
    print("\n  Testing goal cancellation:")
    goal2 = PoseTarget(x=10.0, y=10.0, theta=0.0)

    # Create a goal handle and cancel it before completion
    server = ActionServer._registry["/navigate_to_pose"]
    handle2 = server.accept_goal(goal2)
    handle2.cancel()  # Cancel before execution
    server.execute(handle2)

    print(f"  Goal: navigate to ({goal2.x}, {goal2.y})")
    print(f"  Status: {handle2.status.value}")
    print(f"  Result: {handle2.result}")

    # --- 6. Lifecycle Shutdown ---
    print("\n--- 5. Lifecycle Shutdown ---")
    for node in nodes:
        node.deactivate()
        node.shutdown()

    print("Lifecycle transitions:")
    for node in nodes:
        for entry in node._log:
            print(f"  {entry}")

    # --- 7. Summary ---
    print("\n--- Summary of ROS2 Patterns ---")
    print(f"  {'Pattern':<25s} {'Example':<35s} {'QoS':<15s}")
    print(f"  {'-'*25} {'-'*35} {'-'*15}")
    print(f"  {'Pub/Sub':<25s} {'Lidar -> Safety Monitor':<35s} {'BEST_EFFORT':<15s}")
    print(f"  {'Pub/Sub':<25s} {'Odometry -> Navigation':<35s} {'RELIABLE':<15s}")
    print(f"  {'Service':<25s} {'Emergency Stop':<35s} {'RELIABLE':<15s}")
    print(f"  {'Action':<25s} {'Navigate to Pose':<35s} {'RELIABLE':<15s}")
    print(f"  {'Lifecycle':<25s} {'All nodes managed':<35s} {'N/A':<15s}")


if __name__ == "__main__":
    demo_ros2_concepts()
