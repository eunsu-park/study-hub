"""
Rigid Body Transformations for Robotics
========================================
Rotation matrices, homogeneous transforms, and quaternion operations.

In robotics, describing the pose (position + orientation) of rigid bodies
is fundamental. We use rotation matrices (SO(3)), homogeneous transformation
matrices (SE(3)), and quaternions to represent orientation. Each representation
has trade-offs:
  - Rotation matrices: intuitive, compose via multiplication, but 9 params for 3 DOF
  - Euler angles: compact (3 params), but suffer from gimbal lock
  - Quaternions: 4 params, no gimbal lock, efficient interpolation (SLERP)

This module implements all three and shows how to compose and visualize them.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ---------------------------------------------------------------------------
# 2D Rotation
# ---------------------------------------------------------------------------
def rot2d(theta: float) -> np.ndarray:
    """2D rotation matrix for angle theta (radians).

    A 2D rotation preserves lengths and angles — it is an element of SO(2).
    The matrix rotates a vector counter-clockwise by theta.
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])


# ---------------------------------------------------------------------------
# 3D Rotation about principal axes
# ---------------------------------------------------------------------------
def rotx(theta: float) -> np.ndarray:
    """Rotation about the X-axis by theta radians.

    Used for roll in aerospace convention. The X-axis remains fixed
    while Y and Z rotate in the YZ-plane.
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0,  0],
                     [0, c, -s],
                     [0, s,  c]])


def roty(theta: float) -> np.ndarray:
    """Rotation about the Y-axis by theta radians.

    Used for pitch. Note the sign pattern differs from rotx/rotz because
    the Y-axis cross-product cycle (z, x) reverses the sign of the
    off-diagonal sine terms relative to the other two.
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])


def rotz(theta: float) -> np.ndarray:
    """Rotation about the Z-axis by theta radians.

    Used for yaw. This is the 3D extension of the 2D rotation matrix —
    the Z component is unchanged.
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])


def euler_to_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert ZYX Euler angles to a rotation matrix.

    Convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    This is the most common convention in robotics (extrinsic XYZ = intrinsic ZYX).
    Gimbal lock occurs when pitch = +/- 90 degrees.
    """
    return rotz(yaw) @ roty(pitch) @ rotx(roll)


# ---------------------------------------------------------------------------
# Quaternion class
# ---------------------------------------------------------------------------
class Quaternion:
    """Unit quaternion for 3D rotation representation.

    A quaternion q = w + xi + yj + zk encodes a rotation of angle theta
    about axis n as: q = [cos(theta/2), sin(theta/2)*n].

    Why quaternions?
      - No gimbal lock (unlike Euler angles)
      - Smooth interpolation via SLERP
      - More compact than rotation matrices (4 vs 9 numbers)
      - Numerically stable: just normalize to stay on the unit sphere
    """

    def __init__(self, w: float = 1.0, x: float = 0.0,
                 y: float = 0.0, z: float = 0.0):
        self.q = np.array([w, x, y, z], dtype=float)
        self.normalize()

    def normalize(self) -> "Quaternion":
        """Project back onto the unit sphere to correct numerical drift.

        Rotation quaternions must have unit norm. After repeated multiplications,
        floating-point errors accumulate, so we periodically re-normalize.
        """
        norm = np.linalg.norm(self.q)
        if norm > 1e-10:
            self.q /= norm
        return self

    @property
    def w(self): return self.q[0]
    @property
    def x(self): return self.q[1]
    @property
    def y(self): return self.q[2]
    @property
    def z(self): return self.q[3]

    def __mul__(self, other: "Quaternion") -> "Quaternion":
        """Hamilton product: compose two rotations.

        Quaternion multiplication is non-commutative, just like rotation
        matrix multiplication. q1 * q2 means 'apply q2 first, then q1'.
        """
        w1, x1, y1, z1 = self.q
        w2, x2, y2, z2 = other.q
        return Quaternion(
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        )

    def conjugate(self) -> "Quaternion":
        """Conjugate = inverse for unit quaternions.

        For unit quaternions, q* = q^{-1}. This inverts the rotation.
        """
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def rotate_point(self, p: np.ndarray) -> np.ndarray:
        """Rotate a 3D point using quaternion: p' = q * p * q*.

        This is equivalent to R @ p where R is the rotation matrix,
        but avoids constructing the full 3x3 matrix.
        """
        p_quat = Quaternion(0.0, p[0], p[1], p[2])
        rotated = self * p_quat * self.conjugate()
        return np.array([rotated.x, rotated.y, rotated.z])

    def to_rotation_matrix(self) -> np.ndarray:
        """Convert to 3x3 rotation matrix.

        Useful when you need matrix form for Jacobian computation or
        when composing with translation in a homogeneous transform.
        """
        w, x, y, z = self.q
        return np.array([
            [1 - 2*(y**2 + z**2),   2*(x*y - w*z),       2*(x*z + w*y)],
            [2*(x*y + w*z),         1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y),         2*(y*z + w*x),       1 - 2*(x**2 + y**2)]
        ])

    @staticmethod
    def from_axis_angle(axis: np.ndarray, angle: float) -> "Quaternion":
        """Create quaternion from axis-angle representation.

        axis: unit vector defining the rotation axis
        angle: rotation angle in radians
        """
        axis = axis / np.linalg.norm(axis)
        half = angle / 2.0
        return Quaternion(np.cos(half), *(np.sin(half) * axis))

    def __repr__(self):
        return f"Quaternion(w={self.w:.4f}, x={self.x:.4f}, y={self.y:.4f}, z={self.z:.4f})"


# ---------------------------------------------------------------------------
# Homogeneous transformation
# ---------------------------------------------------------------------------
def homogeneous(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Create a 4x4 homogeneous transformation matrix from R and t.

    SE(3) combines rotation and translation into a single matrix so that
    composition of poses is just matrix multiplication:
        T_02 = T_01 @ T_12
    This avoids the error-prone process of separately tracking R and t.
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------
def draw_frame(ax, T: np.ndarray, label: str = "", length: float = 1.0):
    """Draw a coordinate frame at the pose given by homogeneous transform T.

    Each axis is drawn as a colored arrow: X=red, Y=green, Z=blue.
    This color convention is standard in robotics visualization (RGB = XYZ).
    """
    origin = T[:3, 3]
    colors = ['r', 'g', 'b']
    axis_labels = ['X', 'Y', 'Z']
    for i in range(3):
        direction = T[:3, i] * length
        ax.quiver(*origin, *direction, color=colors[i], arrow_length_ratio=0.1,
                  linewidth=2)
    if label:
        ax.text(*(origin + 0.1), label, fontsize=10, fontweight='bold')


def demo_transforms():
    """Demonstrate successive rigid transformations in 3D."""
    print("=" * 60)
    print("Rigid Body Transformations Demo")
    print("=" * 60)

    # --- 2D rotation demo ---
    theta = np.pi / 4  # 45 degrees
    R2 = rot2d(theta)
    point_2d = np.array([1.0, 0.0])
    rotated_2d = R2 @ point_2d
    print(f"\n[2D] Rotating {point_2d} by 45 degrees: {rotated_2d}")

    # --- Euler angles demo ---
    roll, pitch, yaw = np.radians(30), np.radians(45), np.radians(60)
    R = euler_to_rotation(roll, pitch, yaw)
    print(f"\n[Euler→R] roll=30, pitch=45, yaw=60 degrees:")
    print(f"  det(R) = {np.linalg.det(R):.6f}  (should be 1.0)")
    print(f"  R @ R^T ≈ I: {np.allclose(R @ R.T, np.eye(3))}")

    # --- Quaternion demo ---
    q1 = Quaternion.from_axis_angle(np.array([0, 0, 1]), np.pi / 2)  # 90 deg about Z
    q2 = Quaternion.from_axis_angle(np.array([1, 0, 0]), np.pi / 2)  # 90 deg about X
    q_composed = q1 * q2  # First q2, then q1
    print(f"\n[Quaternion] q1 (90 about Z): {q1}")
    print(f"[Quaternion] q2 (90 about X): {q2}")
    print(f"[Quaternion] q1*q2:           {q_composed}")

    p = np.array([1.0, 0.0, 0.0])
    p_rot = q1.rotate_point(p)
    print(f"[Quaternion] Rotating {p} by q1: {p_rot}")

    # Verify quaternion ↔ rotation matrix consistency
    R_from_q = q1.to_rotation_matrix()
    p_mat = R_from_q @ p
    print(f"[Matrix]     Rotating {p} by R(q1): {p_mat}")
    print(f"  Results match: {np.allclose(p_rot, p_mat)}")

    # --- Homogeneous transform composition ---
    T1 = homogeneous(rotz(np.pi / 4), np.array([1, 0, 0]))
    T2 = homogeneous(rotx(np.pi / 3), np.array([0, 0.5, 0]))
    T_total = T1 @ T2
    print(f"\n[SE(3)] T1: rotate Z by 45 deg, translate [1,0,0]")
    print(f"[SE(3)] T2: rotate X by 60 deg, translate [0,0.5,0]")
    print(f"[SE(3)] T_total origin: {T_total[:3, 3]}")

    # --- 3D visualization ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    T_world = np.eye(4)
    draw_frame(ax, T_world, "World", length=0.5)
    draw_frame(ax, T1, "T1", length=0.4)
    draw_frame(ax, T_total, "T1*T2", length=0.3)

    ax.set_xlim([-0.5, 2.0])
    ax.set_ylim([-0.5, 1.5])
    ax.set_zlim([-0.5, 1.5])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Successive Rigid Transformations (World → T1 → T1*T2)")
    plt.tight_layout()
    plt.savefig("01_rigid_transforms.png", dpi=120)
    plt.show()


if __name__ == "__main__":
    demo_transforms()
