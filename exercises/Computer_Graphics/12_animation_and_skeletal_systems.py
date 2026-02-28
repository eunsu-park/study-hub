"""
Exercises for Lesson 12: Animation and Skeletal Systems
Topic: Computer_Graphics
Solutions to practice problems from the lesson.
"""

import numpy as np

matplotlib_available = False
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib_available = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


def quaternion_from_axis_angle(axis, angle_deg):
    """Create quaternion [w, x, y, z] from axis and angle in degrees."""
    angle_rad = np.radians(angle_deg)
    half = angle_rad / 2.0
    axis = normalize(axis)
    return np.array([np.cos(half),
                     axis[0] * np.sin(half),
                     axis[1] * np.sin(half),
                     axis[2] * np.sin(half)])


def slerp(q0, q1, t):
    """Spherical linear interpolation between unit quaternions [w,x,y,z]."""
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        result = (1 - t) * q0 + t * q1
        return result / np.linalg.norm(result)
    omega = np.arccos(np.clip(dot, -1, 1))
    sin_o = np.sin(omega)
    return (np.sin((1 - t) * omega) / sin_o) * q0 + (np.sin(t * omega) / sin_o) * q1


def quaternion_to_angle(q):
    """Extract rotation angle (degrees) from unit quaternion."""
    return 2 * np.degrees(np.arccos(np.clip(abs(q[0]), 0, 1)))


# ---------------------------------------------------------------------------
# Exercise 1 -- Slerp with constant angular velocity verification
# ---------------------------------------------------------------------------

def exercise_1():
    """
    Implement slerp and verify constant angular velocity by measuring the
    angle between successive interpolated quaternions.
    """
    q_start = quaternion_from_axis_angle(np.array([0, 1, 0]), 0)
    q_end = quaternion_from_axis_angle(np.array([0, 1, 0]), 120)

    num_steps = 8
    ts = np.linspace(0, 1, num_steps + 1)
    quats = [slerp(q_start, q_end, t) for t in ts]

    angles = [quaternion_to_angle(q) for q in quats]

    print(f"  Slerp from 0 to 120 degrees in {num_steps} steps:")
    print(f"  {'t':>5s}  {'angle(deg)':>10s}  {'delta':>8s}")
    for i, (t, a) in enumerate(zip(ts, angles)):
        delta = angles[i] - angles[i-1] if i > 0 else 0
        print(f"  {t:5.3f}  {a:10.2f}  {delta:8.2f}")

    # Check that deltas are equal (constant angular velocity)
    deltas = [angles[i] - angles[i-1] for i in range(1, len(angles))]
    max_variation = max(deltas) - min(deltas)
    print(f"  Max variation in angular step: {max_variation:.6f} degrees")
    print(f"  Constant velocity: {'YES' if max_variation < 0.01 else 'NO'}")


# ---------------------------------------------------------------------------
# Exercise 2 -- 2-Bone Analytical IK
# ---------------------------------------------------------------------------

def exercise_2():
    """
    Implement 2-bone analytical IK using law of cosines.
    Visualize both elbow-up and elbow-down solutions.
    """
    L1 = 3.0  # upper arm length
    L2 = 2.0  # forearm length
    target = np.array([3.5, 2.5])

    d = np.linalg.norm(target)
    if d > L1 + L2:
        print("  Target unreachable!")
        return

    # Law of cosines for elbow angle
    cos_theta2 = (d * d - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    cos_theta2 = np.clip(cos_theta2, -1, 1)

    solutions = []
    for sign, label in [(1, "elbow-up"), (-1, "elbow-down")]:
        theta2 = sign * np.arccos(cos_theta2)
        theta1 = np.arctan2(target[1], target[0]) - np.arctan2(
            L2 * np.sin(theta2), L1 + L2 * np.cos(theta2))

        # Compute joint positions
        elbow = np.array([L1 * np.cos(theta1), L1 * np.sin(theta1)])
        end = elbow + np.array([L2 * np.cos(theta1 + theta2),
                                L2 * np.sin(theta1 + theta2)])
        solutions.append((theta1, theta2, elbow, end, label))

    for theta1, theta2, elbow, end, label in solutions:
        err = np.linalg.norm(end - target)
        print(f"  {label}:")
        print(f"    theta1 = {np.degrees(theta1):7.2f} deg")
        print(f"    theta2 = {np.degrees(theta2):7.2f} deg")
        print(f"    elbow  = ({elbow[0]:.3f}, {elbow[1]:.3f})")
        print(f"    end    = ({end[0]:.3f}, {end[1]:.3f})")
        print(f"    error  = {err:.6f}")

    if matplotlib_available:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, (_, _, elbow, end, label) in zip(axes, solutions):
            ax.plot([0, elbow[0], end[0]], [0, elbow[1], end[1]], 'bo-', lw=2)
            ax.plot(*target, 'r*', markersize=15, label='Target')
            ax.set_xlim(-1, 6)
            ax.set_ylim(-2, 5)
            ax.set_aspect('equal')
            ax.set_title(label)
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('ik_2bone.png', dpi=100)
        plt.close()
        print("  Saved ik_2bone.png")


# ---------------------------------------------------------------------------
# Exercise 3 -- FABRIK 3D
# ---------------------------------------------------------------------------

def exercise_3():
    """
    Extend FABRIK to 3D. Create a 5-bone chain and solve for several target
    positions. Visualize convergence over iterations.
    """

    def fabrik_3d(positions, lengths, target, tol=0.001, max_iter=50):
        n = len(positions)
        pos = [np.array(p, dtype=float) for p in positions]
        target = np.array(target, dtype=float)
        root = pos[0].copy()
        total_len = sum(lengths)
        errors = []

        # Check reachability
        if np.linalg.norm(target - root) > total_len:
            direction = normalize(target - root)
            for i in range(1, n):
                pos[i] = pos[i-1] + direction * lengths[i-1]
            errors.append(np.linalg.norm(pos[-1] - target))
            return pos, errors

        for it in range(max_iter):
            err = np.linalg.norm(pos[-1] - target)
            errors.append(err)
            if err < tol:
                break

            # Forward pass
            pos[-1] = target.copy()
            for i in range(n - 2, -1, -1):
                d = pos[i] - pos[i+1]
                dist = np.linalg.norm(d)
                if dist > 1e-10:
                    d /= dist
                pos[i] = pos[i+1] + d * lengths[i]

            # Backward pass
            pos[0] = root.copy()
            for i in range(1, n):
                d = pos[i] - pos[i-1]
                dist = np.linalg.norm(d)
                if dist > 1e-10:
                    d /= dist
                pos[i] = pos[i-1] + d * lengths[i-1]

        return pos, errors

    # 5-bone chain along X axis
    lengths = [2.0, 1.5, 1.0, 0.8, 0.5]
    initial = [np.array([0, 0, 0])]
    for l in lengths:
        initial.append(initial[-1] + np.array([l, 0, 0]))

    targets = [
        np.array([3.0, 3.0, 2.0]),
        np.array([1.0, 4.0, 0.0]),
        np.array([0.0, 0.0, 5.0]),
    ]

    for target in targets:
        result, errors = fabrik_3d(initial, lengths, target)
        final_err = errors[-1] if errors else float('inf')
        print(f"  Target: ({target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f})")
        print(f"    Converged in {len(errors)} iterations, final error: {final_err:.6f}")
        print(f"    End-effector: ({result[-1][0]:.3f}, {result[-1][1]:.3f}, {result[-1][2]:.3f})")


# ---------------------------------------------------------------------------
# Exercise 4 -- FK chain with sinusoidal animation
# ---------------------------------------------------------------------------

def exercise_4():
    """
    Create a 4-bone arm. Animate joint angles using sinusoidal functions at
    different frequencies. Trace the end-effector path.
    """

    class Bone2D:
        def __init__(self, length):
            self.length = length
            self.angle = 0.0  # degrees

    bones = [Bone2D(2.0), Bone2D(1.5), Bone2D(1.0), Bone2D(0.7)]

    def compute_fk(bones):
        """Return list of joint positions and end-effector."""
        positions = [np.array([0.0, 0.0])]
        cumulative_angle = 0.0
        for bone in bones:
            cumulative_angle += bone.angle
            theta = np.radians(cumulative_angle)
            dx = bone.length * np.cos(theta)
            dy = bone.length * np.sin(theta)
            positions.append(positions[-1] + np.array([dx, dy]))
        return positions

    # Animate over 2 seconds at 60 FPS
    dt = 1.0 / 60.0
    num_frames = 120
    end_effector_path = []
    freqs = [0.5, 1.0, 1.5, 2.0]  # Hz for each joint

    for frame in range(num_frames):
        t = frame * dt
        for i, bone in enumerate(bones):
            bone.angle = 30.0 * np.sin(2 * np.pi * freqs[i] * t)

        positions = compute_fk(bones)
        end_effector_path.append(positions[-1].copy())

    path = np.array(end_effector_path)
    print(f"  4-bone chain with sinusoidal animation:")
    print(f"    Bone lengths: {[b.length for b in bones]}")
    print(f"    Joint frequencies: {freqs} Hz")
    print(f"    Frames: {num_frames}")
    print(f"    End-effector range X: [{path[:,0].min():.2f}, {path[:,0].max():.2f}]")
    print(f"    End-effector range Y: [{path[:,1].min():.2f}, {path[:,1].max():.2f}]")

    if matplotlib_available:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(path[:, 0], path[:, 1], 'b-', alpha=0.5, label='End-effector path')
        ax.plot(path[0, 0], path[0, 1], 'go', markersize=10, label='Start')
        ax.plot(path[-1, 0], path[-1, 1], 'r*', markersize=12, label='End')
        ax.set_aspect('equal')
        ax.set_title('FK Chain End-Effector Path (Sinusoidal Animation)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('fk_animation_path.png', dpi=100)
        plt.close()
        print("  Saved fk_animation_path.png")


# ---------------------------------------------------------------------------
# Exercise 5 -- Blend shape face
# ---------------------------------------------------------------------------

def exercise_5():
    """
    Create a simplified face mesh (10-20 vertices). Define 4 morph targets
    (smile, frown, surprise, blink). Animate blend weights over time.
    """

    def apply_blend_shapes(base, targets, weights):
        result = base.copy()
        for target, w in zip(targets, weights):
            if abs(w) > 1e-6:
                result += w * (target - base)
        return result

    # Simplified face: 12 vertices (rough positions of key landmarks)
    # Layout: left eye, right eye, nose, left mouth, right mouth, etc.
    base = np.array([
        [-0.3,  0.5, 0],  # 0: left eye outer
        [-0.1,  0.5, 0],  # 1: left eye inner
        [ 0.1,  0.5, 0],  # 2: right eye inner
        [ 0.3,  0.5, 0],  # 3: right eye outer
        [-0.2,  0.55, 0], # 4: left eyelid
        [ 0.2,  0.55, 0], # 5: right eyelid
        [ 0.0,  0.2, 0],  # 6: nose tip
        [-0.3, -0.1, 0],  # 7: left mouth corner
        [ 0.3, -0.1, 0],  # 8: right mouth corner
        [ 0.0,  0.0, 0],  # 9: upper lip
        [ 0.0, -0.2, 0],  # 10: lower lip
        [ 0.0, -0.5, 0],  # 11: chin
    ], dtype=float)

    # Morph targets
    smile = base.copy()
    smile[7] += [-0.05, 0.1, 0]   # left corner up
    smile[8] += [0.05, 0.1, 0]    # right corner up
    smile[9] += [0, 0.02, 0]      # upper lip slight

    frown = base.copy()
    frown[7] += [0.05, -0.1, 0]   # left corner down
    frown[8] += [-0.05, -0.1, 0]  # right corner down
    frown[10] += [0, -0.05, 0]    # lower lip down

    surprise = base.copy()
    surprise[4] += [0, 0.08, 0]   # eyelids up
    surprise[5] += [0, 0.08, 0]
    surprise[9] += [0, 0.05, 0]   # mouth opens
    surprise[10] += [0, -0.1, 0]
    surprise[11] += [0, -0.05, 0] # chin drops

    blink = base.copy()
    blink[4] += [0, -0.1, 0]      # eyelids close
    blink[5] += [0, -0.1, 0]

    targets = [smile, frown, surprise, blink]
    names = ["smile", "frown", "surprise", "blink"]

    # Animate: smile -> surprise -> blink -> frown
    num_frames = 60
    print(f"  Face mesh: {len(base)} vertices, {len(targets)} morph targets")
    print(f"  Animation: {num_frames} frames")
    print(f"  Frame  Weights (smile, frown, surprise, blink)  Mouth corners Y")

    for frame in range(0, num_frames, 10):
        t = frame / num_frames
        # Animate weights
        w_smile = max(0, np.sin(2 * np.pi * t))
        w_frown = max(0, np.sin(2 * np.pi * t - np.pi))
        w_surprise = max(0, 0.5 * np.sin(4 * np.pi * t))
        w_blink = max(0, np.sin(8 * np.pi * t)) ** 4  # quick blinks

        weights = [w_smile, w_frown, w_surprise, w_blink]
        result = apply_blend_shapes(base, targets, weights)
        mouth_y = (result[7, 1] + result[8, 1]) / 2
        print(f"  {frame:5d}  [{w_smile:.2f}, {w_frown:.2f}, {w_surprise:.2f}, {w_blink:.2f}]"
              f"  {mouth_y:+.3f}")


# ---------------------------------------------------------------------------
# Exercise 6 -- LBS candy-wrapper artifact
# ---------------------------------------------------------------------------

def exercise_6():
    """
    Implement LBS for a cylindrical mesh with two bones. Rotate one bone by
    0, 45, 90, 135, and 180 degrees. Observe the candy-wrapper collapse.
    Then implement dual quaternion skinning and compare.
    """

    def make_cylinder(n_rings=10, n_segments=8, length=4.0, radius=0.5):
        """Create a cylinder along X axis."""
        verts = []
        for ring in range(n_rings + 1):
            x = ring / n_rings * length
            for seg in range(n_segments):
                theta = 2 * np.pi * seg / n_segments
                y = radius * np.cos(theta)
                z = radius * np.sin(theta)
                verts.append([x, y, z])
        return np.array(verts, dtype=float)

    def compute_weights(verts, bone_split=2.0, total_length=4.0):
        """Assign bone weights: bone 0 for left half, bone 1 for right half."""
        weights = np.zeros((len(verts), 2))
        for i, v in enumerate(verts):
            t = v[0] / total_length  # 0 to 1 along the cylinder
            # Smooth blend near the middle
            blend_width = 0.15
            if t < 0.5 - blend_width:
                weights[i] = [1.0, 0.0]
            elif t > 0.5 + blend_width:
                weights[i] = [0.0, 1.0]
            else:
                alpha = (t - (0.5 - blend_width)) / (2 * blend_width)
                weights[i] = [1.0 - alpha, alpha]
        return weights

    def rotation_matrix_x(angle_deg):
        """Rotation around X axis."""
        theta = np.radians(angle_deg)
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[1, 0, 0, 0],
                         [0, c, -s, 0],
                         [0, s, c, 0],
                         [0, 0, 0, 1]])

    def lbs_deform(verts, weights, bone_matrices):
        """Linear Blend Skinning."""
        result = np.zeros_like(verts)
        for i, v in enumerate(verts):
            v4 = np.array([v[0], v[1], v[2], 1.0])
            blended = np.zeros(4)
            for j in range(len(bone_matrices)):
                blended += weights[i, j] * (bone_matrices[j] @ v4)
            result[i] = blended[:3]
        return result

    def dq_from_matrix(M):
        """Convert 4x4 rigid transform to dual quaternion [qr, qd]."""
        R = M[:3, :3]
        t = M[:3, 3]

        # Rotation to quaternion
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2 * np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2 * np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2 * np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        qr = np.array([w, x, y, z])
        qr /= np.linalg.norm(qr)

        # Dual part: qd = 0.5 * t_quat * qr
        t_quat = np.array([0, t[0], t[1], t[2]])
        qd = 0.5 * quat_mul(t_quat, qr)
        return qr, qd

    def quat_mul(a, b):
        """Quaternion multiply [w,x,y,z]."""
        return np.array([
            a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
            a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
            a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
            a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0],
        ])

    def dq_transform_point(qr, qd, point):
        """Transform a point using a dual quaternion."""
        p = np.array([0, point[0], point[1], point[2]])
        qr_conj = np.array([qr[0], -qr[1], -qr[2], -qr[3]])
        # Position = qr * p * qr_conj + 2 * qd * qr_conj
        rotated = quat_mul(quat_mul(qr, p), qr_conj)
        t_part = 2.0 * quat_mul(qd, qr_conj)
        return rotated[1:4] + t_part[1:4]

    def dqs_deform(verts, weights, bone_dqs):
        """Dual Quaternion Skinning."""
        result = np.zeros_like(verts)
        for i, v in enumerate(verts):
            qr_blend = np.zeros(4)
            qd_blend = np.zeros(4)
            for j in range(len(bone_dqs)):
                qr_j, qd_j = bone_dqs[j]
                # Ensure same hemisphere
                if j > 0 and np.dot(qr_j, bone_dqs[0][0]) < 0:
                    qr_j = -qr_j
                    qd_j = -qd_j
                qr_blend += weights[i, j] * qr_j
                qd_blend += weights[i, j] * qd_j
            # Normalize
            norm = np.linalg.norm(qr_blend)
            if norm > 1e-10:
                qr_blend /= norm
                qd_blend /= norm
            result[i] = dq_transform_point(qr_blend, qd_blend, v)
        return result

    verts = make_cylinder()
    weights = compute_weights(verts)
    bone0 = np.eye(4)  # Bone 0: identity (left half stays)

    angles = [0, 45, 90, 135, 180]
    print(f"  Cylinder: {len(verts)} vertices, 2 bones")
    print(f"  {'Angle':>6s}  {'LBS min_r':>10s}  {'DQS min_r':>10s}  {'LBS vol':>10s}  {'DQS vol':>10s}")

    for angle in angles:
        # Bone 1 rotates around X at the midpoint
        T_to_mid = np.eye(4); T_to_mid[0, 3] = -2.0
        T_from_mid = np.eye(4); T_from_mid[0, 3] = 2.0
        bone1 = T_from_mid @ rotation_matrix_x(angle) @ T_to_mid

        bone_matrices = [bone0, bone1]
        lbs_result = lbs_deform(verts, weights, bone_matrices)

        bone_dqs = [dq_from_matrix(bone0), dq_from_matrix(bone1)]
        dqs_result = dqs_deform(verts, weights, bone_dqs)

        # Measure cross-section radius at midpoint as proxy for volume
        mid_lbs = lbs_result[np.abs(verts[:, 0] - 2.0) < 0.25]
        mid_dqs = dqs_result[np.abs(verts[:, 0] - 2.0) < 0.25]

        lbs_r = np.sqrt(mid_lbs[:, 1]**2 + mid_lbs[:, 2]**2).min() if len(mid_lbs) > 0 else 0
        dqs_r = np.sqrt(mid_dqs[:, 1]**2 + mid_dqs[:, 2]**2).min() if len(mid_dqs) > 0 else 0

        # Approximate volume via sum of cross-section areas
        lbs_vol = np.sum(lbs_result[:, 1]**2 + lbs_result[:, 2]**2)
        dqs_vol = np.sum(dqs_result[:, 1]**2 + dqs_result[:, 2]**2)

        print(f"  {angle:6d}  {lbs_r:10.4f}  {dqs_r:10.4f}  {lbs_vol:10.2f}  {dqs_vol:10.2f}")

    print("  LBS min radius collapses toward 0 at 180 deg (candy-wrapper).")
    print("  DQS preserves volume -- min radius stays larger.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Exercise 1: Slerp Constant Angular Velocity ===")
    exercise_1()

    print("\n=== Exercise 2: 2-Bone Analytical IK ===")
    exercise_2()

    print("\n=== Exercise 3: FABRIK 3D ===")
    exercise_3()

    print("\n=== Exercise 4: FK Chain Sinusoidal Animation ===")
    exercise_4()

    print("\n=== Exercise 5: Blend Shape Face ===")
    exercise_5()

    print("\n=== Exercise 6: LBS vs Dual Quaternion Skinning ===")
    exercise_6()

    print("\nAll exercises completed!")
