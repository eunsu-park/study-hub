"""
Forward Kinematics using Denavit-Hartenberg Parameters
=======================================================
Compute end-effector pose from joint angles via DH convention.

Forward kinematics (FK) answers: "Given joint angles, where is the end-effector?"
The Denavit-Hartenberg (DH) convention provides a systematic way to attach
coordinate frames to each link of a serial manipulator and compute the
transformation between consecutive frames using only 4 parameters:
  - alpha (α): twist angle — rotation about x_{i} to align z_{i-1} to z_i
  - a:          link length — distance along x_i between z_{i-1} and z_i
  - d:          link offset — distance along z_{i-1} between x_{i-1} and x_i
  - theta (θ):  joint angle — rotation about z_{i-1} (variable for revolute joints)

The transformation from frame i-1 to frame i is:
    T_i = Rz(θ) @ Tz(d) @ Tx(a) @ Rx(α)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass
from typing import List


@dataclass
class DHParams:
    """Denavit-Hartenberg parameters for one joint.

    We store both the fixed geometric parameters and a flag indicating
    whether the joint is revolute (theta varies) or prismatic (d varies).
    This separation is important because it tells the FK algorithm which
    parameter to replace with the current joint variable.
    """
    alpha: float   # Link twist (rad)
    a: float       # Link length (m)
    d: float       # Link offset (m)
    theta: float   # Joint angle (rad) — nominal value for revolute
    joint_type: str = "revolute"  # "revolute" or "prismatic"


def dh_transform(alpha: float, a: float, d: float, theta: float) -> np.ndarray:
    """Compute the 4x4 homogeneous transform from DH parameters.

    This is the standard DH convention (not modified DH). The matrix is:
        T = Rz(θ) * Tz(d) * Tx(a) * Rx(α)

    Why this specific order? The DH convention decomposes the general
    transform into a sequence of simple rotations and translations along/about
    the z and x axes of consecutive frames. This makes it possible to describe
    any serial chain with just 4 parameters per joint.
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)

    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,   sa,       ca,      d     ],
        [0,   0,        0,       1     ]
    ])


def forward_kinematics(dh_table: List[DHParams],
                        joint_values: np.ndarray) -> tuple:
    """Compute FK for a serial manipulator.

    Args:
        dh_table: List of DH parameters for each joint
        joint_values: Array of joint variables (angles for revolute, offsets for prismatic)

    Returns:
        T_0n: 4x4 homogeneous transform from base to end-effector
        frames: List of all intermediate frame transforms (for visualization)

    The intermediate frames are useful for:
      1. Visualizing the robot configuration
      2. Computing the Jacobian (needed for velocity kinematics)
      3. Collision checking against obstacles
    """
    T = np.eye(4)
    frames = [T.copy()]  # Base frame

    for i, (dh, q) in enumerate(zip(dh_table, joint_values)):
        # Substitute the joint variable into the appropriate DH parameter
        if dh.joint_type == "revolute":
            theta = dh.theta + q  # Add joint angle to offset
            d = dh.d
        else:  # prismatic
            theta = dh.theta
            d = dh.d + q  # Add joint displacement to offset

        T_i = dh_transform(dh.alpha, dh.a, d, theta)
        T = T @ T_i  # Chain the transforms: T_0n = T_01 @ T_12 @ ... @ T_{n-1,n}
        frames.append(T.copy())

    return T, frames


# ---------------------------------------------------------------------------
# Example robots
# ---------------------------------------------------------------------------
def two_link_planar(L1: float = 1.0, L2: float = 0.8) -> List[DHParams]:
    """DH parameters for a 2-link planar manipulator.

    This is the simplest non-trivial serial robot. Both joints rotate
    about the Z-axis (perpendicular to the plane), and there is no
    twist or offset — making it ideal for learning FK concepts.

          joint1 ---- L1 ---- joint2 ---- L2 ---- end-effector
    """
    return [
        DHParams(alpha=0, a=L1, d=0, theta=0),  # Joint 1
        DHParams(alpha=0, a=L2, d=0, theta=0),  # Joint 2
    ]


def three_dof_spatial(L1: float = 0.5, L2: float = 0.8,
                       L3: float = 0.6) -> List[DHParams]:
    """DH parameters for a 3-DOF spatial manipulator (RRR).

    A spatial arm with:
      - Joint 1: rotates about base Z-axis (like a turntable)
      - Joint 2: shoulder joint, tilts the arm up/down
      - Joint 3: elbow joint, extends/folds the forearm

    The alpha=pi/2 for joint 1 transitions from the vertical base Z-axis
    to the horizontal Z-axis of the shoulder frame.
    """
    return [
        DHParams(alpha=np.pi / 2, a=0,  d=L1, theta=0),  # Base rotation
        DHParams(alpha=0,          a=L2, d=0,  theta=0),  # Shoulder
        DHParams(alpha=0,          a=L3, d=0,  theta=0),  # Elbow
    ]


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_2d_manipulator(dh_table: List[DHParams], joint_angles: np.ndarray,
                         ax, color='b', label=''):
    """Draw a planar manipulator in 2D by connecting joint positions."""
    _, frames = forward_kinematics(dh_table, joint_angles)
    positions = np.array([f[:3, 3] for f in frames])

    ax.plot(positions[:, 0], positions[:, 1], '-o', color=color,
            linewidth=3, markersize=8, label=label)
    # Mark end-effector with a star
    ax.plot(positions[-1, 0], positions[-1, 1], '*', color=color,
            markersize=15)


def plot_3d_manipulator(dh_table: List[DHParams], joint_angles: np.ndarray,
                         ax, color='b', label=''):
    """Draw a spatial manipulator in 3D by connecting joint positions."""
    _, frames = forward_kinematics(dh_table, joint_angles)
    positions = np.array([f[:3, 3] for f in frames])

    ax.plot3D(positions[:, 0], positions[:, 1], positions[:, 2],
              '-o', color=color, linewidth=3, markersize=8, label=label)
    ax.plot3D([positions[-1, 0]], [positions[-1, 1]], [positions[-1, 2]],
              '*', color=color, markersize=15)

    # Draw a small coordinate frame at the end-effector to show orientation
    T_ee = frames[-1]
    origin = T_ee[:3, 3]
    length = 0.15
    for i, c in enumerate(['r', 'g', 'b']):
        direction = T_ee[:3, i] * length
        ax.quiver(*origin, *direction, color=c, linewidth=2,
                  arrow_length_ratio=0.15)


def demo_forward_kinematics():
    """Demonstrate FK for planar and spatial manipulators."""
    print("=" * 60)
    print("Forward Kinematics Demo")
    print("=" * 60)

    # --- 2-link planar arm ---
    planar_dh = two_link_planar()
    configs_2d = [
        np.array([0.0, 0.0]),
        np.array([np.pi / 4, -np.pi / 6]),
        np.array([np.pi / 3, np.pi / 3]),
        np.array([-np.pi / 4, np.pi / 2]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Planar arm in different configurations
    ax = axes[0]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, (q, c) in enumerate(zip(configs_2d, colors)):
        T_ee, _ = forward_kinematics(planar_dh, q)
        pos = T_ee[:3, 3]
        label = f"q=[{np.degrees(q[0]):.0f}, {np.degrees(q[1]):.0f}] -> ({pos[0]:.2f}, {pos[1]:.2f})"
        plot_2d_manipulator(planar_dh, q, ax, color=c, label=label)
        print(f"[2-Link] {label}")

    ax.set_xlim([-2.5, 2.5])
    ax.set_ylim([-2.5, 2.5])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("2-Link Planar Manipulator (FK)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    # Workspace trace — sweep joint angles to show reachable region
    ax2 = axes[1]
    theta1_range = np.linspace(-np.pi, np.pi, 60)
    theta2_range = np.linspace(-np.pi, np.pi, 60)
    x_ws, y_ws = [], []
    for t1 in theta1_range:
        for t2 in theta2_range:
            T_ee, _ = forward_kinematics(planar_dh, np.array([t1, t2]))
            x_ws.append(T_ee[0, 3])
            y_ws.append(T_ee[1, 3])
    ax2.scatter(x_ws, y_ws, s=0.5, alpha=0.3, color='steelblue')
    ax2.set_aspect('equal')
    ax2.set_title("Workspace (Reachable Region)")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("02_forward_kinematics_2d.png", dpi=120)
    plt.show()

    # --- 3-DOF spatial arm ---
    print("\n--- 3-DOF Spatial Arm ---")
    spatial_dh = three_dof_spatial()
    configs_3d = [
        np.array([0.0, np.pi / 4, -np.pi / 6]),
        np.array([np.pi / 3, np.pi / 6, -np.pi / 4]),
        np.array([-np.pi / 4, np.pi / 3, np.pi / 6]),
    ]

    fig = plt.figure(figsize=(10, 8))
    ax3d = fig.add_subplot(111, projection='3d')

    for i, (q, c) in enumerate(zip(configs_3d, colors)):
        T_ee, _ = forward_kinematics(spatial_dh, q)
        pos = T_ee[:3, 3]
        label = f"Config {i + 1}: EE=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
        plot_3d_manipulator(spatial_dh, q, ax3d, color=c, label=label)
        print(f"[3-DOF] {label}")

    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.set_title("3-DOF Spatial Manipulator (FK)")
    ax3d.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("02_forward_kinematics_3d.png", dpi=120)
    plt.show()


if __name__ == "__main__":
    demo_forward_kinematics()
