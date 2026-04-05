# Robotics

## Overview

Robotics is the interdisciplinary field that integrates mechanical engineering, electrical engineering, and computer science to design, build, and program machines that can perceive, reason, and act in the physical world. This topic covers the mathematical foundations (rigid body transformations, kinematics, dynamics), motion planning and control, perception and state estimation (sensors, SLAM), and modern software frameworks (ROS2). Whether you're building industrial manipulators, autonomous mobile robots, or collaborative human-robot systems, understanding robotics principles gives you the tools to bridge the gap between computation and physical action.

## Prerequisites

- **Mathematical Methods**: Linear algebra, matrix operations, coordinate frames (Mathematical_Methods L03-L05)
- **Control Theory**: PID control, state-space methods, stability (Control_Theory L01-L10)
- **Programming**: Object-oriented programming, data structures (Programming L05-L06)
- **Python**: NumPy, Matplotlib (Python L01-L08)
- **Computer Vision** (optional): Image processing, feature detection for perception lessons (Computer_Vision L01-L08)

## Learning Path

```
Foundations (L01-L04)
├── L01: Robotics Overview and Classification
├── L02: Rigid Body Transformations
├── L03: Forward Kinematics (DH Parameters)
└── L04: Inverse Kinematics

Dynamics and Control (L05-L08)
├── L05: Velocity Kinematics and the Jacobian
├── L06: Robot Dynamics (Euler-Lagrange)
├── L07: Motion Planning (RRT, PRM)
└── L08: Trajectory Planning and Execution

Perception and Estimation (L09-L12)
├── L09: Robot Control (PID, Computed Torque)
├── L10: Sensors and Perception
├── L11: State Estimation and Filtering
└── L12: SLAM (Simultaneous Localization and Mapping)

Modern Robotics (L13-L16)
├── L13: ROS2 Fundamentals
├── L14: ROS2 Navigation Stack
├── L15: Reinforcement Learning for Robotics
└── L16: Multi-Robot Systems and Swarms
```

## Lesson List

| # | Lesson | Description |
|---|--------|-------------|
| 01 | [Robotics Overview and Classification](01_Robotics_Overview.md) | History, classification (manipulators, mobile, aerial), applications, DOF, workspace |
| 02 | [Rigid Body Transformations](02_Rigid_Body_Transformations.md) | Rotation matrices, homogeneous transforms, Euler angles, quaternions, axis-angle |
| 03 | [Forward Kinematics](03_Forward_Kinematics.md) | DH parameters, kinematic chains, workspace analysis, serial vs parallel |
| 04 | [Inverse Kinematics](04_Inverse_Kinematics.md) | Analytical and numerical IK, Jacobian-based methods, singularities, redundancy |
| 05 | [Velocity Kinematics and the Jacobian](05_Velocity_Kinematics.md) | Manipulator Jacobian, singularity analysis, force/torque mapping, manipulability |
| 06 | [Robot Dynamics](06_Robot_Dynamics.md) | Euler-Lagrange formulation, inertia matrix, Coriolis/gravity terms, Newton-Euler |
| 07 | [Motion Planning](07_Motion_Planning.md) | Configuration space, RRT/RRT*, PRM, potential fields, sampling-based planning |
| 08 | [Trajectory Planning and Execution](08_Trajectory_Planning.md) | Polynomial trajectories, splines, minimum-jerk, time-optimal, task-space planning |
| 09 | [Robot Control](09_Robot_Control.md) | Joint-space PID, computed torque, impedance control, force control, hybrid |
| 10 | [Sensors and Perception](10_Sensors_and_Perception.md) | Encoders, IMU, LiDAR, cameras, depth sensors, sensor fusion fundamentals |
| 11 | [State Estimation and Filtering](11_State_Estimation.md) | Kalman filter, EKF, particle filter, sensor fusion, localization |
| 12 | [SLAM](12_SLAM.md) | Graph-based SLAM, EKF-SLAM, particle filter SLAM, visual SLAM, loop closure |
| 13 | [ROS2 Fundamentals](13_ROS2_Fundamentals.md) | Nodes, topics, services, actions, launch files, parameter server, lifecycle |
| 14 | [ROS2 Navigation Stack](14_ROS2_Navigation.md) | Nav2, costmaps, path planning plugins, recovery behaviors, behavior trees |
| 15 | [Reinforcement Learning for Robotics](15_RL_for_Robotics.md) | Sim-to-real transfer, reward shaping, safe RL, manipulation/locomotion policies |
| 16 | [Multi-Robot Systems and Swarms](16_Multi_Robot_Systems.md) | Task allocation, formation control, consensus algorithms, swarm intelligence |

## Related Topics

| Topic | Connection |
|-------|------------|
| Control_Theory | PID, state-space control, stability analysis for robot controllers |
| Computer_Vision | Perception pipeline, visual SLAM, object detection for grasping |
| Reinforcement_Learning | Policy learning for manipulation and locomotion |
| Mathematical_Methods | Linear algebra, coordinate transforms, differential equations |
| Signal_Processing | Sensor signal processing, filtering |
| Deep_Learning | Neural network-based perception and control policies |

## Example Files

Located in `examples/Robotics/`:

| File | Description |
|------|-------------|
| `01_rigid_transforms.py` | Rotation matrices, homogeneous transforms, quaternion operations |
| `02_forward_kinematics.py` | DH parameter table, FK computation for serial manipulators |
| `03_inverse_kinematics.py` | Analytical and numerical IK solvers, Jacobian pseudo-inverse |
| `04_jacobian.py` | Manipulator Jacobian computation, singularity analysis, manipulability ellipsoid |
| `05_dynamics.py` | Euler-Lagrange dynamics for a 2-link planar arm |
| `06_motion_planning.py` | RRT and RRT* path planning with obstacle avoidance |
| `07_trajectory.py` | Polynomial and minimum-jerk trajectory generation |
| `08_pid_control.py` | Joint-space PID control simulation for a robotic arm |
| `09_kalman_filter.py` | Extended Kalman filter for robot localization |
| `10_slam.py` | Simple EKF-SLAM implementation with landmark observations |
| `11_particle_filter.py` | Monte Carlo localization with particle filter |
| `12_swarm.py` | Reynolds flocking model for multi-robot swarm simulation |
