"""
Exercises for Lesson 02: Rigid Body Transformations
Topic: Robotics
Solutions to practice problems from the lesson.
"""

import numpy as np


def rot_x(angle):
    """Rotation matrix about x-axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])


def rot_y(angle):
    """Rotation matrix about y-axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def rot_z(angle):
    """Rotation matrix about z-axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


def euler_zyx_to_rotation(yaw, pitch, roll):
    """Convert ZYX Euler angles to rotation matrix."""
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def rotation_to_euler_zyx(R):
    """Extract ZYX Euler angles from rotation matrix."""
    # pitch from R[2,0]
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))

    if np.abs(np.cos(pitch)) > 1e-10:
        yaw = np.arctan2(R[1, 0], R[0, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
    else:
        # Gimbal lock: pitch = +-90 deg
        yaw = np.arctan2(-R[0, 1], R[1, 1])
        roll = 0.0  # arbitrary
    return yaw, pitch, roll


def rotation_to_axis_angle(R):
    """Convert rotation matrix to axis-angle representation."""
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if np.abs(angle) < 1e-10:
        return np.array([0, 0, 1]), 0.0  # identity
    if np.abs(angle - np.pi) < 1e-10:
        # 180-degree rotation: axis from eigenvector
        eigenvalues, eigenvectors = np.linalg.eig(R)
        idx = np.argmin(np.abs(eigenvalues - 1))
        axis = np.real(eigenvectors[:, idx])
        return axis / np.linalg.norm(axis), angle

    axis = np.array([R[2, 1] - R[1, 2],
                     R[0, 2] - R[2, 0],
                     R[1, 0] - R[0, 1]]) / (2 * np.sin(angle))
    return axis / np.linalg.norm(axis), angle


def axis_angle_to_quaternion(axis, angle):
    """Convert axis-angle to quaternion [w, x, y, z]."""
    w = np.cos(angle / 2)
    xyz = np.sin(angle / 2) * axis
    return np.array([w, xyz[0], xyz[1], xyz[2]])


def quaternion_to_rotation(q):
    """Convert quaternion [w, x, y, z] to rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])


def quaternion_multiply(q1, q2):
    """Multiply two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def slerp(q0, q1, t):
    """Spherical linear interpolation between quaternions."""
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1, 1)

    if dot > 0.9995:
        # Nearly parallel — use linear interpolation
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)

    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return s0 * q0 + s1 * q1


def homogeneous(R, t):
    """Create 4x4 homogeneous transformation from R and t."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def exercise_1():
    """
    Exercise 1: Rotation Matrix Properties
    R = Rz(30) * Rx(45): verify properties.
    """
    R = rot_z(np.radians(30)) @ rot_x(np.radians(45))

    print("R = Rz(30°) * Rx(45°):")
    print(R.round(6))

    # Verify R^T R = I
    product = R.T @ R
    print(f"\nR^T * R (should be I):")
    print(product.round(10))
    print(f"Max deviation from I: {np.max(np.abs(product - np.eye(3))):.2e}")

    # Verify det(R) = 1
    det_R = np.linalg.det(R)
    print(f"\ndet(R) = {det_R:.10f} (should be 1.0)")

    # R^-1 = R^T for rotation matrices
    R_inv = R.T
    print(f"\nR^{{-1}} = R^T:")
    print(R_inv.round(6))

    # Verify R * R^-1 = I
    verify = R @ R_inv
    print(f"\nR * R^{{-1}} (should be I):")
    print(verify.round(10))


def exercise_2():
    """
    Exercise 2: Gimbal Lock Investigation
    """
    print("Part 1 & 2: Near-gimbal-lock comparison")
    pitch_near = np.radians(89)

    R_a = euler_zyx_to_rotation(np.radians(10), pitch_near, np.radians(20))
    R_b = euler_zyx_to_rotation(np.radians(15), pitch_near, np.radians(15))

    diff = np.linalg.norm(R_a - R_b, 'fro')
    print(f"  (yaw=10, pitch=89, roll=20) vs (yaw=15, pitch=89, roll=15)")
    print(f"  Frobenius norm of difference: {diff:.6f}")
    print(f"  Despite different yaw/roll, the matrices are very similar")
    print(f"  because pitch ≈ 90° causes yaw and roll to couple.")

    print("\nPart 3: Exact gimbal lock at pitch = 90°")
    pitch_exact = np.radians(90)

    # At pitch=90, yaw+roll determines the same axis rotation.
    # R(yaw, 90, roll) depends only on (yaw + roll)
    # So (yaw=30, roll=0) and (yaw=0, roll=30) give the same R.
    R_1 = euler_zyx_to_rotation(np.radians(30), pitch_exact, np.radians(0))
    R_2 = euler_zyx_to_rotation(np.radians(0), pitch_exact, np.radians(30))

    print(f"  R(yaw=30, pitch=90, roll=0):")
    print(f"  {R_1.round(6)}")
    print(f"\n  R(yaw=0, pitch=90, roll=30):")
    print(f"  {R_2.round(6)}")
    print(f"\n  Difference: {np.linalg.norm(R_1 - R_2, 'fro'):.2e}")
    print(f"  Both give the same rotation matrix because at pitch=90°,")
    print(f"  only (yaw + roll) matters — we lose one degree of freedom.")

    # Try extracting angles back — ambiguous
    y1, p1, r1 = rotation_to_euler_zyx(R_1)
    print(f"\n  Extracted from R1: yaw={np.degrees(y1):.2f}, "
          f"pitch={np.degrees(p1):.2f}, roll={np.degrees(r1):.2f}")
    print(f"  Note: yaw and roll are ambiguous; only their sum is determined.")


def exercise_3():
    """
    Exercise 3: Quaternion Operations
    """
    # (a) 90° about x-axis
    q_x90 = axis_angle_to_quaternion(np.array([1, 0, 0]), np.radians(90))
    # (b) 90° about y-axis
    q_y90 = axis_angle_to_quaternion(np.array([0, 1, 0]), np.radians(90))

    print("Quaternion (a): 90° about x-axis")
    print(f"  q_x = {q_x90.round(6)}")
    print("Quaternion (b): 90° about y-axis")
    print(f"  q_y = {q_y90.round(6)}")

    # Composition: first (a) then (b) => q_b * q_a (right-to-left)
    q_ab = quaternion_multiply(q_y90, q_x90)
    # Composition: first (b) then (a) => q_a * q_b
    q_ba = quaternion_multiply(q_x90, q_y90)

    print(f"\nFirst x then y: q_y * q_x = {q_ab.round(6)}")
    print(f"First y then x: q_x * q_y = {q_ba.round(6)}")
    print(f"Same? {np.allclose(q_ab, q_ba) or np.allclose(q_ab, -q_ba)}")
    print("Rotation composition is NOT commutative (different results).")

    # SLERP between identity and 180° about z
    print("\nSLERP: identity → 180° about z-axis")
    q_identity = np.array([1, 0, 0, 0.0])
    q_z180 = axis_angle_to_quaternion(np.array([0, 0, 1]), np.radians(180))
    print(f"  q_start = {q_identity}")
    print(f"  q_end   = {q_z180.round(6)}")

    t_values = np.linspace(0, 1, 5)
    for t in t_values:
        q_t = slerp(q_identity, q_z180, t)
        # Extract angle
        angle = 2 * np.arccos(np.clip(abs(q_t[0]), 0, 1))
        print(f"  t={t:.2f}: q={q_t.round(4)}, angle={np.degrees(angle):.1f}°")


def exercise_4():
    """
    Exercise 4: Transformation Chains
    Camera on robot end-effector.
    """
    # T_0_ee: end-effector in base frame
    R_ee = rot_z(np.radians(45))
    t_ee = np.array([0.5, 0.3, 0.8])
    T_0_ee = homogeneous(R_ee, t_ee)

    # T_ee_cam: camera in end-effector frame
    R_cam = rot_x(np.radians(180))
    t_cam = np.array([0, 0, 0.1])
    T_ee_cam = homogeneous(R_cam, t_cam)

    # 1. Camera pose in base frame
    T_0_cam = T_0_ee @ T_ee_cam
    print("1. Camera pose in base frame (T_0_cam):")
    print(T_0_cam.round(6))
    print(f"   Camera position: {T_0_cam[:3, 3].round(4)}")

    # 2. Point in camera frame → base frame
    p_cam = np.array([0.2, 0.1, 1.0, 1.0])  # homogeneous
    p_base = T_0_cam @ p_cam
    print(f"\n2. Point p_cam = (0.2, 0.1, 1.0) in base frame:")
    print(f"   p_base = {p_base[:3].round(4)}")

    # 3. New end-effector pose (90° about z, same translation)
    R_ee_new = rot_z(np.radians(90))
    T_0_ee_new = homogeneous(R_ee_new, t_ee)
    T_0_cam_new = T_0_ee_new @ T_ee_cam
    print(f"\n3. New camera pose (after base rotated 90°):")
    print(T_0_cam_new.round(6))
    print(f"   New camera position: {T_0_cam_new[:3, 3].round(4)}")


def exercise_5():
    """
    Exercise 5: Representation Conversion Round-Trip
    Euler → Matrix → Axis-Angle → Quaternion → Matrix → Euler
    """
    yaw_orig = np.radians(25)
    pitch_orig = np.radians(40)
    roll_orig = np.radians(-15)

    print(f"Original Euler angles (ZYX):")
    print(f"  yaw={np.degrees(yaw_orig):.1f}°, pitch={np.degrees(pitch_orig):.1f}°, "
          f"roll={np.degrees(roll_orig):.1f}°")

    # Step 1: Euler → Rotation matrix
    R = euler_zyx_to_rotation(yaw_orig, pitch_orig, roll_orig)
    print(f"\nStep 1 - Rotation matrix:")
    print(R.round(6))

    # Step 2: Rotation matrix → Axis-angle
    axis, angle = rotation_to_axis_angle(R)
    print(f"\nStep 2 - Axis-angle:")
    print(f"  axis = {axis.round(6)}, angle = {np.degrees(angle):.4f}°")

    # Step 3: Axis-angle → Quaternion
    q = axis_angle_to_quaternion(axis, angle)
    print(f"\nStep 3 - Quaternion:")
    print(f"  q = {q.round(6)}")

    # Step 4: Quaternion → Rotation matrix
    R_recovered = quaternion_to_rotation(q)
    print(f"\nStep 4 - Recovered rotation matrix:")
    print(R_recovered.round(6))
    print(f"  Max diff from original R: {np.max(np.abs(R - R_recovered)):.2e}")

    # Step 5: Rotation matrix → Euler angles
    yaw_rec, pitch_rec, roll_rec = rotation_to_euler_zyx(R_recovered)
    print(f"\nStep 5 - Recovered Euler angles:")
    print(f"  yaw={np.degrees(yaw_rec):.4f}°, pitch={np.degrees(pitch_rec):.4f}°, "
          f"roll={np.degrees(roll_rec):.4f}°")

    # Verify round-trip
    errors = [
        abs(np.degrees(yaw_orig) - np.degrees(yaw_rec)),
        abs(np.degrees(pitch_orig) - np.degrees(pitch_rec)),
        abs(np.degrees(roll_orig) - np.degrees(roll_rec)),
    ]
    print(f"\nRound-trip errors (degrees): yaw={errors[0]:.2e}, "
          f"pitch={errors[1]:.2e}, roll={errors[2]:.2e}")
    print(f"All within numerical precision: {all(e < 1e-6 for e in errors)}")


if __name__ == "__main__":
    print("=" * 70)
    print("Lesson 02: Rigid Body Transformations — Exercise Solutions")
    print("=" * 70)

    print("\n--- Exercise 1: Rotation Matrix Properties ---")
    exercise_1()

    print("\n--- Exercise 2: Gimbal Lock Investigation ---")
    exercise_2()

    print("\n--- Exercise 3: Quaternion Operations ---")
    exercise_3()

    print("\n--- Exercise 4: Transformation Chains ---")
    exercise_4()

    print("\n--- Exercise 5: Representation Conversion Round-Trip ---")
    exercise_5()

    print("\n" + "=" * 70)
    print("All exercises completed!")
    print("=" * 70)
