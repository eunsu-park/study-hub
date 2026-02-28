"""
Exercises for Lesson 03: 3D Transformations and Projections
Topic: Computer_Graphics
Solutions to practice problems from the lesson.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ===================================================================
# Helper functions (from the lesson)
# ===================================================================

def normalize(v):
    """Normalize a vector to unit length."""
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


def make_translation(tx, ty, tz):
    return np.array([[1, 0, 0, tx], [0, 1, 0, ty],
                     [0, 0, 1, tz], [0, 0, 0, 1]], dtype=float)


def make_scale(sx, sy, sz):
    return np.array([[sx, 0, 0, 0], [0, sy, 0, 0],
                     [0, 0, sz, 0], [0, 0, 0, 1]], dtype=float)


def make_rotation_x(deg):
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0],
                     [0, s, c, 0], [0, 0, 0, 1]], dtype=float)


def make_rotation_y(deg):
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0],
                     [-s, 0, c, 0], [0, 0, 0, 1]], dtype=float)


def make_rotation_z(deg):
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0],
                     [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)


def look_at(eye, target, up):
    eye, target, up = [np.asarray(v, dtype=float) for v in [eye, target, up]]
    f = normalize(target - eye)
    r = normalize(np.cross(f, up))
    u = np.cross(r, f)
    return np.array([
        [r[0], r[1], r[2], -np.dot(r, eye)],
        [u[0], u[1], u[2], -np.dot(u, eye)],
        [-f[0], -f[1], -f[2], np.dot(f, eye)],
        [0, 0, 0, 1]
    ], dtype=float)


def perspective(fov_deg, aspect, near, far):
    fov = np.radians(fov_deg)
    t = near * np.tan(fov / 2)
    r = t * aspect
    return np.array([
        [near / r, 0, 0, 0],
        [0, near / t, 0, 0],
        [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
        [0, 0, -1, 0]
    ], dtype=float)


def orthographic(left, right, bottom, top, near, far):
    return np.array([
        [2 / (right - left), 0, 0, -(right + left) / (right - left)],
        [0, 2 / (top - bottom), 0, -(top + bottom) / (top - bottom)],
        [0, 0, -2 / (far - near), -(far + near) / (far - near)],
        [0, 0, 0, 1]
    ], dtype=float)


def quaternion_from_axis_angle(axis, angle_deg):
    axis = normalize(np.asarray(axis, dtype=float))
    half_angle = np.radians(angle_deg) / 2
    w = np.cos(half_angle)
    xyz = np.sin(half_angle) * axis
    return np.array([w, xyz[0], xyz[1], xyz[2]])


def quaternion_to_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y), 0],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x), 0],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y), 0],
        [0, 0, 0, 1]
    ], dtype=float)


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ])


def slerp(q1, q2, t):
    dot = np.dot(q1, q2)
    if dot < 0:
        q2 = -q2
        dot = -dot
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    w1 = np.sin((1 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta
    return w1 * q1 + w2 * q2


# Unit cube vertices
CUBE_VERTICES = np.array([
    [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5],
    [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
    [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5],
    [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
], dtype=float)


def exercise_1():
    """
    MVP Construction: Given a camera at (3, 3, 3) looking at the origin with
    up vector (0, 1, 0), construct the full MVP matrix for a unit cube at
    position (1, 0, -2). Transform all 8 vertices to screen coordinates
    (1920x1080 viewport).
    """
    # Model matrix: translate cube to (1, 0, -2)
    M = make_translation(1, 0, -2)

    # View matrix: camera at (3, 3, 3) looking at origin
    V = look_at(eye=[3, 3, 3], target=[0, 0, 0], up=[0, 1, 0])

    # Projection matrix: 60 degree FOV, 16:9 aspect
    P = perspective(fov_deg=60, aspect=1920 / 1080, near=0.1, far=100)

    # Combined MVP
    MVP = P @ V @ M

    print("Model matrix (translate to (1, 0, -2)):")
    print(np.round(M, 4))
    print()
    print("View matrix (camera at (3,3,3) looking at origin):")
    print(np.round(V, 4))
    print()
    print("Projection matrix (60 deg FOV, 16:9):")
    print(np.round(P, 4))
    print()

    # Transform all 8 vertices
    screen_w, screen_h = 1920, 1080
    print(f"{'Vertex':<6} {'Object Space':<22} {'Clip Space':<40} {'NDC':<30} {'Screen':<20}")
    print("-" * 120)

    for i, v in enumerate(CUBE_VERTICES):
        p_obj = np.array([v[0], v[1], v[2], 1.0])
        p_clip = MVP @ p_obj

        w = p_clip[3]
        p_ndc = p_clip[:3] / w

        sx = (p_ndc[0] + 1) * 0.5 * screen_w
        sy = (1 - p_ndc[1]) * 0.5 * screen_h

        print(f"  {i:<4} ({v[0]:5.1f},{v[1]:5.1f},{v[2]:5.1f})  "
              f"({p_clip[0]:7.3f},{p_clip[1]:7.3f},{p_clip[2]:7.3f},{p_clip[3]:7.3f})  "
              f"({p_ndc[0]:7.3f},{p_ndc[1]:7.3f},{p_ndc[2]:7.3f})  "
              f"({sx:7.1f},{sy:7.1f})")


def exercise_2():
    """
    Orthographic vs Perspective: Render the same scene using both projections.
    Describe the visual differences.
    """
    # Scene: several cubes at different depths
    cube_positions = [
        (0, 0, -3), (2, 0, -5), (-2, 0, -7), (0, 2, -4)
    ]

    V = look_at(eye=[0, 3, 5], target=[0, 0, -3], up=[0, 1, 0])

    # Perspective projection
    P_persp = perspective(fov_deg=60, aspect=16 / 9, near=0.1, far=100)

    # Orthographic projection (matching approximate view volume)
    P_ortho = orthographic(-6, 6, -3.375, 3.375, 0.1, 100)

    print("Comparing Orthographic vs Perspective Projection")
    print("=" * 60)

    for proj_name, P in [("Perspective", P_persp), ("Orthographic", P_ortho)]:
        print(f"\n--- {proj_name} Projection ---")
        for pos in cube_positions:
            M = make_translation(*pos)
            MVP = P @ V @ M

            # Transform cube center
            center = MVP @ np.array([0, 0, 0, 1.0])
            ndc = center[:3] / center[3]

            # Transform two opposite corners to measure apparent size
            corner1 = MVP @ np.array([-0.5, -0.5, -0.5, 1.0])
            corner2 = MVP @ np.array([0.5, 0.5, 0.5, 1.0])
            ndc1 = corner1[:2] / corner1[3]
            ndc2 = corner2[:2] / corner2[3]
            apparent_size = np.linalg.norm(ndc2 - ndc1)

            print(f"  Cube at {pos}: NDC center=({ndc[0]:.3f}, {ndc[1]:.3f}), "
                  f"apparent size={apparent_size:.4f}")

    print()
    print("Visual Differences:")
    print("  Perspective: Distant cubes appear smaller (foreshortening)")
    print("  Orthographic: All cubes appear the same size regardless of distance")
    print()
    print("When to prefer orthographic:")
    print("  - CAD/engineering drawings (accurate measurements)")
    print("  - 2D games with isometric view")
    print("  - Shadow maps for directional lights")
    print("  - Technical illustrations")
    print("  - UI elements in 3D applications")


def exercise_3():
    """
    Depth Buffer Values: For a perspective projection with n=0.1, f=100,
    compute the NDC depth for objects at z=-1, z=-10, z=-50, z=-100.
    Plot z_NDC(z) and discuss why this non-linear distribution causes
    precision problems.
    """
    near, far = 0.1, 100.0

    # The perspective depth mapping: z_ndc = -(f+n)/(f-n) + 2*f*n/((f-n)*(-z))
    # Simplified: z_ndc(z) = A + B/z where z is negative (eye space)
    A = -(far + near) / (far - near)
    B = -2 * far * near / (far - near)

    test_depths = [-1, -10, -50, -100]
    print(f"Perspective depth mapping (near={near}, far={far})")
    print(f"A = -(f+n)/(f-n) = {A:.6f}")
    print(f"B = -2fn/(f-n)   = {B:.6f}")
    print()
    print(f"{'Eye-space z':<15} {'NDC depth':<15} {'Depth buffer [0,1]':<20}")
    print("-" * 50)

    for z in test_depths:
        z_ndc = A + B / z
        depth_01 = (z_ndc + 1) * 0.5  # Map [-1,1] to [0,1]
        print(f"  z = {z:>6}      {z_ndc:>10.6f}      {depth_01:>10.6f}")

    print()

    # Plot the depth function
    z_values = np.linspace(-near, -far, 1000)
    z_ndc_values = A + B / z_values
    depth_01_values = (z_ndc_values + 1) * 0.5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(-z_values, z_ndc_values, 'b-', linewidth=2)
    ax1.set_xlabel('Distance from camera (-z)')
    ax1.set_ylabel('NDC depth')
    ax1.set_title('NDC Depth vs Distance')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=-1, color='r', linestyle='--', alpha=0.5, label='NDC min (-1)')
    ax1.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='NDC max (1)')
    ax1.legend()

    ax2.plot(-z_values, depth_01_values, 'b-', linewidth=2)
    ax2.set_xlabel('Distance from camera (-z)')
    ax2.set_ylabel('Depth buffer value [0, 1]')
    ax2.set_title('Depth Buffer Precision Distribution')
    ax2.grid(True, alpha=0.3)

    # Show what fraction of depth range is consumed by near objects
    z_10pct = -10  # 10% of depth range
    db_at_10 = (A + B / z_10pct + 1) * 0.5
    ax2.axvline(x=10, color='r', linestyle='--', alpha=0.5)
    ax2.annotate(f'z=10: {db_at_10:.3f}\n({db_at_10 * 100:.1f}% of buffer used)',
                 xy=(10, db_at_10), fontsize=9, color='red')

    plt.tight_layout()
    plt.savefig('ex03_depth_precision.png', dpi=150, bbox_inches='tight')
    print("Plot saved to ex03_depth_precision.png")
    print()

    # Analyze precision distribution
    ranges = [(near, 1), (1, 10), (10, 50), (50, far)]
    print("Depth buffer usage by distance range:")
    for z_start, z_end in ranges:
        db_start = (A + B / (-z_start) + 1) * 0.5
        db_end = (A + B / (-z_end) + 1) * 0.5
        pct = abs(db_end - db_start) * 100
        print(f"  z=[{z_start:>5.1f}, {z_end:>5.1f}]: uses {pct:.2f}% of depth buffer")

    print()
    print("Problem: Most depth precision is concentrated near the near plane.")
    print("Objects far from the camera share a tiny fraction of the depth range,")
    print("causing z-fighting artifacts between distant objects.")


def exercise_4():
    """
    Gimbal Lock Demonstration: Implement an Euler angle rotation system.
    Show that when pitch = 90 degrees, yaw and roll produce the same rotation.
    Then show the same rotations using quaternions without the lock.
    """
    print("Gimbal Lock Demonstration")
    print("=" * 50)

    # Euler angle rotation: R = Ry(yaw) * Rx(pitch) * Rz(roll)
    pitch = 90  # degrees

    print(f"\nWith pitch = {pitch} degrees:")
    print()

    # Test: vary yaw and roll, see if they produce different results
    test_configs = [
        (30, 90, 0, "Yaw=30, Pitch=90, Roll=0"),
        (0, 90, 30, "Yaw=0, Pitch=90, Roll=30"),
        (15, 90, 15, "Yaw=15, Pitch=90, Roll=15"),
        (45, 90, -15, "Yaw=45, Pitch=90, Roll=-15"),
    ]

    test_point = np.array([1, 0, 0, 1.0])

    print("Euler angle results (R = Ry * Rx * Rz):")
    for yaw, pit, roll, desc in test_configs:
        R = make_rotation_y(yaw) @ make_rotation_x(pit) @ make_rotation_z(roll)
        result = R @ test_point
        print(f"  {desc:<35} -> ({result[0]:7.4f}, {result[1]:7.4f}, {result[2]:7.4f})")

    print()
    print("Observation: Yaw=30,Roll=0 and Yaw=0,Roll=30 produce the SAME result!")
    print("This is gimbal lock: yaw and roll control the same rotation axis")
    print("when pitch = 90 degrees.")
    print()

    # Verify mathematically
    R_a = make_rotation_y(30) @ make_rotation_x(90) @ make_rotation_z(0)
    R_b = make_rotation_y(0) @ make_rotation_x(90) @ make_rotation_z(30)
    print(f"R(yaw=30, pitch=90, roll=0) == R(yaw=0, pitch=90, roll=30)? "
          f"{np.allclose(R_a, R_b)}")
    print()

    # Quaternion solution
    print("Quaternion approach (no gimbal lock):")
    print("-" * 50)

    q_yaw30 = quaternion_from_axis_angle([0, 1, 0], 30)
    q_pitch90 = quaternion_from_axis_angle([1, 0, 0], 90)
    q_roll30 = quaternion_from_axis_angle([0, 0, 1], 30)

    # Compose: yaw * pitch * roll
    q_a = quaternion_multiply(q_yaw30, quaternion_multiply(q_pitch90,
                              quaternion_from_axis_angle([0, 0, 1], 0)))
    q_b = quaternion_multiply(quaternion_from_axis_angle([0, 1, 0], 0),
                              quaternion_multiply(q_pitch90, q_roll30))

    M_qa = quaternion_to_matrix(q_a)
    M_qb = quaternion_to_matrix(q_b)

    result_qa = M_qa @ test_point
    result_qb = M_qb @ test_point

    print(f"  Q(yaw=30) * Q(pitch=90) * Q(roll=0)  -> "
          f"({result_qa[0]:7.4f}, {result_qa[1]:7.4f}, {result_qa[2]:7.4f})")
    print(f"  Q(yaw=0) * Q(pitch=90) * Q(roll=30)  -> "
          f"({result_qb[0]:7.4f}, {result_qb[1]:7.4f}, {result_qb[2]:7.4f})")
    print()

    # Now demonstrate quaternions can represent arbitrary orientations smoothly
    q_target = quaternion_from_axis_angle([1, 1, 0], 90)
    print("Smooth interpolation with SLERP (impossible with Euler at gimbal lock):")
    for t in np.linspace(0, 1, 5):
        q_interp = slerp(quaternion_from_axis_angle([0, 1, 0], 0), q_target, t)
        M = quaternion_to_matrix(q_interp)
        r = M @ test_point
        print(f"  t={t:.2f}: ({r[0]:7.4f}, {r[1]:7.4f}, {r[2]:7.4f})")


def exercise_5():
    """
    SLERP Animation: Using quaternions, implement a smooth rotation from
    "looking forward" to "looking 180 degrees right and 45 degrees up."
    Sample 10 intermediate orientations.
    """
    # Start: looking forward (identity rotation)
    q_start = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion

    # End: rotate 180 degrees around Y (look right), then 45 degrees up (around X)
    q_yaw = quaternion_from_axis_angle([0, 1, 0], 180)
    q_pitch = quaternion_from_axis_angle([1, 0, 0], -45)  # Negative for "up"
    q_end = quaternion_multiply(q_pitch, q_yaw)

    # Normalize
    q_end = q_end / np.linalg.norm(q_end)

    print("SLERP Animation: Forward -> 180 degrees right + 45 degrees up")
    print("=" * 60)
    print()
    print(f"Start quaternion: {np.round(q_start, 4)}")
    print(f"End quaternion:   {np.round(q_end, 4)}")
    print()

    # Sample 10 intermediate orientations
    forward = np.array([0, 0, -1, 0])  # Looking along -Z

    print(f"{'Step':<6} {'t':<6} {'Forward Dir':<30} {'Quaternion':<40}")
    print("-" * 82)

    for i in range(11):
        t = i / 10.0
        q = slerp(q_start, q_end, t)
        q = q / np.linalg.norm(q)  # Ensure unit

        M = quaternion_to_matrix(q)
        fwd = M @ forward
        fwd_dir = fwd[:3]

        print(f"  {i:<4} {t:.1f}   "
              f"({fwd_dir[0]:7.4f}, {fwd_dir[1]:7.4f}, {fwd_dir[2]:7.4f})   "
              f"({q[0]:7.4f}, {q[1]:7.4f}, {q[2]:7.4f}, {q[3]:7.4f})")

    print()
    print("Note: SLERP maintains constant angular velocity throughout the")
    print("interpolation, producing smooth, natural-looking rotation.")

    # Verify constant angular velocity
    print("\nAngular velocity verification (angle between consecutive steps):")
    prev_q = q_start
    for i in range(1, 11):
        t = i / 10.0
        q = slerp(q_start, q_end, t)
        q = q / np.linalg.norm(q)

        dot = abs(np.dot(prev_q, q))
        angle = 2 * np.arccos(min(dot, 1.0))
        print(f"  Step {i-1}->{i}: angular change = {np.degrees(angle):.2f} degrees")
        prev_q = q


def exercise_6():
    """
    Normal Transformation: Create a model matrix with non-uniform scaling
    (2, 1, 0.5) and rotation. Show that applying the model matrix directly
    to a normal vector produces an incorrect result, while the
    inverse-transpose produces the correct perpendicular normal.
    """
    # Model matrix: non-uniform scale + rotation
    S = make_scale(2, 1, 0.5)
    R = make_rotation_y(45)
    M = R @ S

    print("Normal Transformation with Non-Uniform Scaling")
    print("=" * 60)
    print()
    print("Model matrix M = R_y(45) @ S(2, 1, 0.5):")
    print(np.round(M, 4))
    print()

    # Define a surface tangent and normal
    # For a surface lying in the XZ plane:
    tangent = np.array([1, 0, 0, 0.0])    # Tangent along X
    normal = np.array([0, 1, 0, 0.0])     # Normal along Y (perpendicular to tangent)

    # Verify they are perpendicular
    dot_original = np.dot(tangent[:3], normal[:3])
    print(f"Original tangent: {tangent[:3]}")
    print(f"Original normal:  {normal[:3]}")
    print(f"Dot product (should be 0): {dot_original:.6f}")
    print()

    # Transform tangent (tangents transform correctly with M)
    tangent_transformed = M @ tangent
    tangent_transformed_3 = tangent_transformed[:3]

    # INCORRECT: transform normal with M directly
    normal_wrong = M @ normal
    normal_wrong_3 = normal_wrong[:3]
    normal_wrong_3 = normal_wrong_3 / np.linalg.norm(normal_wrong_3)

    dot_wrong = np.dot(tangent_transformed_3, normal_wrong_3)
    print("INCORRECT: Transforming normal with M directly:")
    print(f"  Transformed tangent: {np.round(tangent_transformed_3, 4)}")
    print(f"  Transformed normal:  {np.round(normal_wrong_3, 4)}")
    print(f"  Dot product: {dot_wrong:.6f}")
    print(f"  Perpendicular? {abs(dot_wrong) < 1e-6}")
    print()

    # CORRECT: transform normal with inverse-transpose of M
    M_3x3 = M[:3, :3]
    normal_matrix = np.linalg.inv(M_3x3).T

    normal_correct = normal_matrix @ normal[:3]
    normal_correct = normal_correct / np.linalg.norm(normal_correct)

    dot_correct = np.dot(tangent_transformed_3, normal_correct)
    print("CORRECT: Transforming normal with inverse-transpose:")
    print(f"  Normal matrix (M^-1)^T:")
    print(f"  {np.round(normal_matrix, 4)}")
    print(f"  Transformed tangent: {np.round(tangent_transformed_3, 4)}")
    print(f"  Transformed normal:  {np.round(normal_correct, 4)}")
    print(f"  Dot product: {dot_correct:.6f}")
    print(f"  Perpendicular? {abs(dot_correct) < 1e-6}")
    print()

    # Demonstrate with a second tangent-normal pair
    print("Additional test with a different surface orientation:")
    tangent2 = np.array([0, 0, 1, 0.0])
    normal2 = np.array([0, 1, 0, 0.0])

    t2_xformed = (M @ tangent2)[:3]
    n2_wrong = (M @ normal2)[:3]
    n2_wrong = n2_wrong / np.linalg.norm(n2_wrong)
    n2_correct = normal_matrix @ normal2[:3]
    n2_correct = n2_correct / np.linalg.norm(n2_correct)

    print(f"  Tangent2 transformed: {np.round(t2_xformed, 4)}")
    print(f"  Normal2 (wrong):  dot = {np.dot(t2_xformed, n2_wrong):.6f}")
    print(f"  Normal2 (correct): dot = {np.dot(t2_xformed, n2_correct):.6f}")
    print()

    # Special case: with only rotation (no scaling), M works fine
    print("Special case: rotation only (no non-uniform scaling):")
    M_rot = make_rotation_y(45)
    t_rot = (M_rot @ tangent)[:3]
    n_rot_direct = (M_rot @ normal)[:3]
    print(f"  Dot product with M directly: {np.dot(t_rot, n_rot_direct):.6f}")
    print("  Rotation matrices are orthogonal, so M = (M^-1)^T")
    print("  No inverse-transpose needed for pure rotations!")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: MVP Construction ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Orthographic vs Perspective ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Depth Buffer Values ===")
    print("=" * 60)
    exercise_3()

    print("\n" + "=" * 60)
    print("=== Exercise 4: Gimbal Lock Demonstration ===")
    print("=" * 60)
    exercise_4()

    print("\n" + "=" * 60)
    print("=== Exercise 5: SLERP Animation ===")
    print("=" * 60)
    exercise_5()

    print("\n" + "=" * 60)
    print("=== Exercise 6: Normal Transformation ===")
    print("=" * 60)
    exercise_6()

    print("\nAll exercises completed!")
