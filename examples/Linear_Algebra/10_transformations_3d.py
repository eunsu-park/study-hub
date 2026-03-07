"""
3D Transformations

Demonstrates 3D geometric transformations:
- 3D rotation matrices (Euler angles and axis-angle)
- Homogeneous coordinates for combined transformations
- Model-view-projection pipeline
- Camera projection matrix
- Quaternion basics

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def rotation_matrices():
    """Demonstrate rotation matrices around coordinate axes."""
    print("=" * 60)
    print("3D ROTATION MATRICES")
    print("=" * 60)

    def Rx(theta):
        """Rotation around x-axis."""
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[1, 0, 0],
                         [0, c, -s],
                         [0, s, c]])

    def Ry(theta):
        """Rotation around y-axis."""
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])

    def Rz(theta):
        """Rotation around z-axis."""
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s, 0],
                         [s, c, 0],
                         [0, 0, 1]])

    theta = np.pi / 4  # 45 degrees
    print(f"\nRotation angle: {np.degrees(theta)} degrees")

    print(f"\nRx(45):\n{np.round(Rx(theta), 4)}")
    print(f"Ry(45):\n{np.round(Ry(theta), 4)}")
    print(f"Rz(45):\n{np.round(Rz(theta), 4)}")

    # Properties of rotation matrices
    R = Rx(theta)
    print(f"\n--- Rotation Matrix Properties ---")
    print(f"det(R) = {np.linalg.det(R):.6f} (should be 1)")
    print(f"R^T R = I: {np.allclose(R.T @ R, np.eye(3))}")
    print(f"R^{-1} = R^T: {np.allclose(np.linalg.inv(R), R.T)}")

    # Rotation preserves norms
    v = np.array([1, 2, 3])
    print(f"\n||v|| = {np.linalg.norm(v):.4f}")
    print(f"||Rv|| = {np.linalg.norm(R @ v):.4f}")

    # Euler angles: combined rotation (ZYX convention)
    alpha, beta, gamma = np.radians(30), np.radians(45), np.radians(60)
    R_euler = Rz(alpha) @ Ry(beta) @ Rx(gamma)
    print(f"\nEuler angles (Z={30}, Y={45}, X={60}):")
    print(f"Combined rotation:\n{np.round(R_euler, 4)}")
    print(f"det = {np.linalg.det(R_euler):.6f}")

    # Non-commutativity: Rx Ry != Ry Rx
    print(f"\nRx @ Ry == Ry @ Rx? {np.allclose(Rx(theta) @ Ry(theta), Ry(theta) @ Rx(theta))}")

    return Rx, Ry, Rz


def axis_angle_rotation():
    """Rotation about an arbitrary axis (Rodrigues' formula)."""
    print("\n" + "=" * 60)
    print("AXIS-ANGLE ROTATION (RODRIGUES)")
    print("=" * 60)

    def rodrigues(axis, theta):
        """Rodrigues' rotation formula: R = I cos(t) + (1-cos(t)) k k^T + sin(t) K"""
        k = axis / np.linalg.norm(axis)
        K = np.array([[0, -k[2], k[1]],
                       [k[2], 0, -k[0]],
                       [-k[1], k[0], 0]])
        R = np.eye(3) * np.cos(theta) + (1 - np.cos(theta)) * np.outer(k, k) + np.sin(theta) * K
        return R

    # Rotate 90 degrees around [1, 1, 1] / sqrt(3)
    axis = np.array([1, 1, 1])
    theta = np.pi / 2
    R = rodrigues(axis, theta)

    print(f"Axis: {axis / np.linalg.norm(axis)}")
    print(f"Angle: {np.degrees(theta)} degrees")
    print(f"R:\n{np.round(R, 4)}")
    print(f"det(R) = {np.linalg.det(R):.6f}")
    print(f"R^T R = I: {np.allclose(R.T @ R, np.eye(3))}")

    # Apply to standard basis vectors
    for label, v in [('e1', [1,0,0]), ('e2', [0,1,0]), ('e3', [0,0,1])]:
        v = np.array(v, dtype=float)
        print(f"{label} -> {np.round(R @ v, 4)}")


def homogeneous_coordinates():
    """Demonstrate homogeneous coordinates for 3D transformations."""
    print("\n" + "=" * 60)
    print("HOMOGENEOUS COORDINATES")
    print("=" * 60)

    # Translation matrix
    def translation(tx, ty, tz):
        T = np.eye(4)
        T[:3, 3] = [tx, ty, tz]
        return T

    # Rotation around z-axis (4x4)
    def rotation_z_4x4(theta):
        c, s = np.cos(theta), np.sin(theta)
        R = np.eye(4)
        R[0, 0] = c; R[0, 1] = -s
        R[1, 0] = s; R[1, 1] = c
        return R

    # Scaling matrix
    def scaling(sx, sy, sz):
        S = np.eye(4)
        S[0, 0] = sx; S[1, 1] = sy; S[2, 2] = sz
        return S

    T = translation(3, 4, 5)
    R = rotation_z_4x4(np.pi / 4)
    S = scaling(2, 2, 2)

    print("Translation T(3,4,5):")
    print(T)
    print(f"\nRotation Rz(45):")
    print(np.round(R, 4))
    print(f"\nScaling S(2,2,2):")
    print(S)

    # Combined: first scale, then rotate, then translate
    # Read right to left: M = T R S
    M = T @ R @ S
    print(f"\nCombined M = T @ R @ S:")
    print(np.round(M, 4))

    # Apply to a point
    p = np.array([1, 0, 0, 1])  # homogeneous
    p_transformed = M @ p
    print(f"\nPoint [1, 0, 0]:")
    print(f"Transformed: {np.round(p_transformed[:3], 4)}")

    # Inverse transformation
    M_inv = np.linalg.inv(M)
    p_recovered = M_inv @ p_transformed
    print(f"Recovered: {np.round(p_recovered[:3], 10)}")
    print(f"Match: {np.allclose(p[:3], p_recovered[:3])}")


def camera_projection():
    """Demonstrate perspective projection (pinhole camera model)."""
    print("\n" + "=" * 60)
    print("CAMERA PROJECTION")
    print("=" * 60)

    # Intrinsic parameters
    fx, fy = 500, 500  # Focal lengths in pixels
    cx, cy = 320, 240  # Principal point (image center)

    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=float)
    print(f"Camera intrinsic matrix K:\n{K}")

    # Extrinsic: camera at [0,0,5] looking at origin
    # World to camera transform
    R_cam = np.eye(3)  # Camera aligned with world
    t_cam = np.array([0, 0, -5])  # Camera 5 units back on z-axis

    # Project 3D world points to 2D image
    world_points = np.array([
        [0, 0, 0],     # Origin
        [1, 0, 0],     # Along x
        [0, 1, 0],     # Along y
        [1, 1, 0],     # Diagonal
        [0.5, 0.5, 1], # Above center
    ], dtype=float)

    print(f"\n{'World Point':>20}  {'Camera Point':>20}  {'Image Point':>15}")
    print("-" * 60)

    for p_w in world_points:
        # World to camera
        p_c = R_cam @ p_w + t_cam

        # Perspective projection
        p_img_h = K @ p_c
        p_img = p_img_h[:2] / p_img_h[2]

        print(f"{str(np.round(p_w, 2)):>20}  "
              f"{str(np.round(p_c, 2)):>20}  "
              f"{str(np.round(p_img, 1)):>15}")


def quaternion_basics():
    """Demonstrate quaternion representation of rotations."""
    print("\n" + "=" * 60)
    print("QUATERNION BASICS")
    print("=" * 60)

    def quat_from_axis_angle(axis, theta):
        """Create quaternion from axis-angle: q = [cos(t/2), sin(t/2)*axis]."""
        axis = axis / np.linalg.norm(axis)
        return np.array([np.cos(theta / 2),
                         np.sin(theta / 2) * axis[0],
                         np.sin(theta / 2) * axis[1],
                         np.sin(theta / 2) * axis[2]])

    def quat_multiply(q1, q2):
        """Hamilton product of two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def quat_to_matrix(q):
        """Convert unit quaternion to 3x3 rotation matrix."""
        w, x, y, z = q
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
        ])

    # Rotation around z-axis by 90 degrees
    q = quat_from_axis_angle(np.array([0, 0, 1]), np.pi / 2)
    print(f"Quaternion for 90 deg rotation around z: {np.round(q, 4)}")
    print(f"||q|| = {np.linalg.norm(q):.6f} (unit quaternion)")

    R = quat_to_matrix(q)
    print(f"\nCorresponding rotation matrix:\n{np.round(R, 4)}")

    # Compose two rotations via quaternion multiplication
    q1 = quat_from_axis_angle(np.array([0, 0, 1]), np.pi / 4)  # 45 deg z
    q2 = quat_from_axis_angle(np.array([1, 0, 0]), np.pi / 4)  # 45 deg x
    q_combined = quat_multiply(q1, q2)

    R1 = quat_to_matrix(q1)
    R2 = quat_to_matrix(q2)
    R_combined = quat_to_matrix(q_combined)
    R_product = R1 @ R2

    print(f"\nComposed rotation (quat multiply):\n{np.round(R_combined, 4)}")
    print(f"Direct R1 @ R2:\n{np.round(R_product, 4)}")
    print(f"Match: {np.allclose(R_combined, R_product)}")


def visualize_transformations(Rx, Ry, Rz):
    """Visualize 3D transformations on a cube."""
    # Define cube vertices
    cube = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                     [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
    cube -= 0.5  # Center at origin

    # Define edges
    edges = [(0,1), (1,2), (2,3), (3,0),
             (4,5), (5,6), (6,7), (7,4),
             (0,4), (1,5), (2,6), (3,7)]

    fig = plt.figure(figsize=(15, 5))
    titles = ['Original', 'Rotated 45 deg (Z)', 'Rotated 45 deg (X then Z)']
    transforms = [
        np.eye(3),
        Rz(np.pi / 4),
        Rz(np.pi / 4) @ Rx(np.pi / 4),
    ]

    for idx, (title, T) in enumerate(zip(titles, transforms)):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        transformed = (T @ cube.T).T

        for e in edges:
            ax.plot3D(*transformed[list(e)].T, 'b-', linewidth=1.5)

        # Draw axes
        for i, (color, label) in enumerate(zip(['r', 'g', 'b'], ['X', 'Y', 'Z'])):
            axis = np.zeros((2, 3))
            axis[1, i] = 1.2
            ax.plot3D(*axis.T, color=color, linewidth=2, alpha=0.5)
            ax.text(*axis[1], label, color=color, fontsize=10)

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    plt.tight_layout()
    plt.savefig('transformations_3d.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: transformations_3d.png")


if __name__ == "__main__":
    Rx, Ry, Rz = rotation_matrices()
    axis_angle_rotation()
    homogeneous_coordinates()
    camera_projection()
    quaternion_basics()
    visualize_transformations(Rx, Ry, Rz)
    print("\nAll examples completed!")
