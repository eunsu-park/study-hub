"""
Exercises for Lesson 04: Inverse Kinematics
Topic: Robotics
Solutions to practice problems from the lesson.
"""

import numpy as np


def fk_2link(q, l1, l2):
    """Forward kinematics for 2-link planar robot. Returns (x, y)."""
    x = l1 * np.cos(q[0]) + l2 * np.cos(q[0] + q[1])
    y = l1 * np.sin(q[0]) + l2 * np.sin(q[0] + q[1])
    return np.array([x, y])


def jacobian_2link(q, l1, l2):
    """Jacobian for 2-link planar robot."""
    s1 = np.sin(q[0])
    c1 = np.cos(q[0])
    s12 = np.sin(q[0] + q[1])
    c12 = np.cos(q[0] + q[1])
    return np.array([
        [-l1 * s1 - l2 * s12, -l2 * s12],
        [l1 * c1 + l2 * c12, l2 * c12]
    ])


def fk_3link(q, l1, l2, l3):
    """Forward kinematics for 3-link planar robot. Returns (x, y)."""
    x = l1 * np.cos(q[0]) + l2 * np.cos(q[0] + q[1]) + l3 * np.cos(q[0] + q[1] + q[2])
    y = l1 * np.sin(q[0]) + l2 * np.sin(q[0] + q[1]) + l3 * np.sin(q[0] + q[1] + q[2])
    return np.array([x, y])


def jacobian_3link(q, l1, l2, l3):
    """Jacobian for 3-link planar robot (2x3)."""
    s1 = np.sin(q[0])
    c1 = np.cos(q[0])
    s12 = np.sin(q[0] + q[1])
    c12 = np.cos(q[0] + q[1])
    s123 = np.sin(q[0] + q[1] + q[2])
    c123 = np.cos(q[0] + q[1] + q[2])
    return np.array([
        [-l1*s1 - l2*s12 - l3*s123, -l2*s12 - l3*s123, -l3*s123],
        [l1*c1 + l2*c12 + l3*c123, l2*c12 + l3*c123, l3*c123]
    ])


def exercise_1():
    """
    Exercise 1: 2-Link Analytical IK
    l1=1.0, l2=0.7.
    """
    l1, l2 = 1.0, 0.7

    def analytical_ik(x, y, l1, l2):
        """Analytical IK for 2-link planar robot. Returns list of solutions."""
        r_sq = x**2 + y**2
        r = np.sqrt(r_sq)
        r_max = l1 + l2
        r_min = abs(l1 - l2)

        if r > r_max + 1e-10:
            return []  # unreachable
        if r < r_min - 1e-10:
            return []  # unreachable

        # Clamp for numerical safety
        cos_q2 = np.clip((r_sq - l1**2 - l2**2) / (2 * l1 * l2), -1, 1)
        q2_pos = np.arccos(cos_q2)
        q2_neg = -q2_pos

        solutions = []
        for q2 in [q2_pos, q2_neg]:
            k1 = l1 + l2 * np.cos(q2)
            k2 = l2 * np.sin(q2)
            q1 = np.arctan2(y, x) - np.arctan2(k2, k1)
            solutions.append(np.array([q1, q2]))

        return solutions

    # Case 1: Target (1.2, 0.5)
    targets = [(1.2, 0.5), (0.3, 0.0), (2.0, 0.0)]

    for tx, ty in targets:
        r = np.sqrt(tx**2 + ty**2)
        print(f"\nTarget ({tx}, {ty}), distance = {r:.4f}")
        print(f"  Reach range: [{abs(l1 - l2):.1f}, {l1 + l2:.1f}]")

        solutions = analytical_ik(tx, ty, l1, l2)
        if not solutions:
            if r > l1 + l2:
                print(f"  UNREACHABLE: target beyond maximum reach ({l1 + l2:.1f})")
            else:
                print(f"  UNREACHABLE: target inside minimum reach ({abs(l1 - l2):.1f})")
        else:
            for i, sol in enumerate(solutions):
                pos = fk_2link(sol, l1, l2)
                err = np.linalg.norm(pos - np.array([tx, ty]))
                config = "elbow-up" if sol[1] >= 0 else "elbow-down"
                print(f"  Solution {i + 1} ({config}): "
                      f"q1={np.degrees(sol[0]):.2f}°, q2={np.degrees(sol[1]):.2f}°, "
                      f"error={err:.2e}")

            if abs(r - (l1 + l2)) < 0.01 or abs(r - abs(l1 - l2)) < 0.01:
                print(f"  Special case: target is at workspace boundary "
                      f"— solutions coincide (singular).")


def exercise_2():
    """
    Exercise 2: Numerical IK Comparison
    Newton-Raphson, pseudo-inverse, and DLS for 2-link robot.
    """
    l1, l2 = 1.0, 0.7
    target = np.array([0.5, 1.0])
    q0 = np.array([0.0, 0.0])
    tol = 1e-8
    max_iter = 100

    def newton_raphson(q0, target):
        q = q0.copy()
        for i in range(max_iter):
            e = target - fk_2link(q, l1, l2)
            if np.linalg.norm(e) < tol:
                return q, i + 1, True
            J = jacobian_2link(q, l1, l2)
            dq = np.linalg.solve(J, e)
            q = q + dq
        return q, max_iter, False

    def pseudo_inverse(q0, target, alpha=1.0):
        q = q0.copy()
        for i in range(max_iter):
            e = target - fk_2link(q, l1, l2)
            if np.linalg.norm(e) < tol:
                return q, i + 1, True
            J = jacobian_2link(q, l1, l2)
            J_pinv = np.linalg.pinv(J)
            dq = alpha * J_pinv @ e
            q = q + dq
        return q, max_iter, False

    def dls(q0, target, lam=0.1):
        q = q0.copy()
        for i in range(max_iter):
            e = target - fk_2link(q, l1, l2)
            if np.linalg.norm(e) < tol:
                return q, i + 1, True
            J = jacobian_2link(q, l1, l2)
            # DLS: dq = J^T (J J^T + lambda^2 I)^{-1} e
            JJT = J @ J.T
            dq = J.T @ np.linalg.solve(JJT + lam**2 * np.eye(2), e)
            q = q + dq
        return q, max_iter, False

    print(f"Target: {target}, Initial: q0 = (0°, 0°)")
    print(f"Tolerance: {tol}")

    # Part 1: Normal target
    for name, solver in [("Newton-Raphson", lambda: newton_raphson(q0, target)),
                         ("Pseudo-inverse", lambda: pseudo_inverse(q0, target)),
                         ("DLS (λ=0.1)", lambda: dls(q0, target, 0.1))]:
        q, iters, converged = solver()
        pos = fk_2link(q, l1, l2)
        err = np.linalg.norm(pos - target)
        print(f"\n  {name}:")
        print(f"    Iterations: {iters}, Converged: {converged}")
        print(f"    q = ({np.degrees(q[0]):.4f}°, {np.degrees(q[1]):.4f}°)")
        print(f"    Pos error: {err:.2e}")

    # Part 2: Near workspace boundary
    print(f"\n--- Near workspace boundary (r = {l1 + l2 - 0.01:.2f}) ---")
    r_boundary = l1 + l2 - 0.01
    target_boundary = np.array([r_boundary, 0.0])
    for name, solver in [("Newton-Raphson", lambda: newton_raphson(q0, target_boundary)),
                         ("Pseudo-inverse", lambda: pseudo_inverse(q0, target_boundary)),
                         ("DLS (λ=0.1)", lambda: dls(q0, target_boundary, 0.1))]:
        q, iters, converged = solver()
        err = np.linalg.norm(fk_2link(q, l1, l2) - target_boundary)
        print(f"  {name}: iters={iters}, converged={converged}, err={err:.2e}")

    # Part 3: Starting from singular configuration
    print(f"\n--- From singular config q0=(0,0) to (0.5, 1.0) ---")
    print(f"  At q=(0,0), arm is fully extended, det(J)={np.linalg.det(jacobian_2link(q0, l1, l2)):.6f}")
    print(f"  Newton-Raphson may fail (singular Jacobian), DLS handles it.")


def exercise_3():
    """
    Exercise 3: Singularity Analysis for 2-link robot.
    """
    l1, l2 = 1.0, 0.7

    print("Jacobian for 2-link robot:")
    print("  J = [[-l1*s1 - l2*s12, -l2*s12],")
    print("       [ l1*c1 + l2*c12,  l2*c12]]")
    print()
    print("det(J) = l1*l2*sin(q2)")
    print("  Singular when sin(q2) = 0 => q2 = 0° or q2 = 180°")
    print()

    # q2 = 0: fully extended
    q_singular = np.array([np.radians(45), 0])
    J = jacobian_2link(q_singular, l1, l2)
    det_J = np.linalg.det(J)
    U, S, Vt = np.linalg.svd(J)

    print(f"At q = (45°, 0°) [fully extended]:")
    print(f"  det(J) = {det_J:.6f}")
    print(f"  Singular values: {S.round(6)}")
    print(f"  The lost direction (right singular vector for smallest S):")
    print(f"    v = {Vt[-1].round(6)}")
    print(f"  The impossible end-effector direction (left singular vector):")
    print(f"    u = {U[:, -1].round(6)}")

    # EE direction at this config
    pos = fk_2link(q_singular, l1, l2)
    direction_to_ee = pos / np.linalg.norm(pos)
    print(f"\n  End-effector position: {pos.round(4)}")
    print(f"  Direction from origin: {direction_to_ee.round(4)}")
    print(f"  The impossible motion is along the radial direction (toward/away from base)")
    print(f"  because the arm cannot extend further or retract without bending q2.")

    # Manipulability measure
    print(f"\nManipulability w(q2) = |l1*l2*sin(q2)|")
    q2_range = np.linspace(0, np.pi, 7)
    for q2 in q2_range:
        w = abs(l1 * l2 * np.sin(q2))
        print(f"  q2 = {np.degrees(q2):6.1f}°: w = {w:.4f}")
    print(f"\n  Maximum at q2 = 90° (w = {l1 * l2:.4f})")
    print(f"  Minimum at q2 = 0° and 180° (w = 0, singular)")


def exercise_4():
    """
    Exercise 4: 6-DOF IK Framework (Wrist Center Computation)
    """
    d6 = 0.056  # wrist-to-EE offset

    # Desired EE pose
    R_d = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ]).astype(float)
    p_d = np.array([0.5, 0.3, 0.8])
    print("Desired end-effector pose:")
    print(f"  Position: {p_d}")
    print(f"  Orientation (R):")
    print(f"  {R_d}")

    # Wrist center = p_d - d6 * R_d * [0,0,1]^T = p_d - d6 * z_6
    z_6 = R_d[:, 2]  # approach direction
    p_wc = p_d - d6 * z_6
    print(f"\nWrist center computation:")
    print(f"  z_6 (approach vector) = {z_6}")
    print(f"  p_wc = p_d - d6 * z_6 = {p_d} - {d6} * {z_6}")
    print(f"  p_wc = {p_wc.round(6)}")

    # Verify independence from joints 4-6
    print(f"\nVerification: Wrist center depends only on joints 1-3.")
    print(f"  By construction, joints 4,5,6 rotate about the wrist center point.")
    print(f"  Changing q4, q5, q6 changes the orientation of the end-effector")
    print(f"  but not the wrist center position. This is why spherical wrist")
    print(f"  robots allow decoupled IK: first solve for q1-q3 from p_wc,")
    print(f"  then solve for q4-q6 from the desired orientation.")

    # Solve theta1 geometrically
    theta1 = np.arctan2(p_wc[1], p_wc[0])
    print(f"\n  theta1 from wrist center: arctan2(y_wc, x_wc) = "
          f"arctan2({p_wc[1]:.4f}, {p_wc[0]:.4f}) = {np.degrees(theta1):.2f}°")


def exercise_5():
    """
    Exercise 5: Redundancy — 3-link planar robot reaching 2D target.
    """
    l1, l2, l3 = 0.5, 0.5, 0.3
    target = np.array([0.5, 0.8])
    q0 = np.radians(np.array([20.0, 40.0, 60.0]))
    q_center = np.radians(np.array([0.0, 0.0, 0.0]))  # joint centering reference

    tol = 1e-6
    max_iter = 200
    alpha = 0.5  # step size

    print("3-link planar robot (redundant): reaching 2D target")
    print(f"  Link lengths: {l1}, {l2}, {l3}")
    print(f"  Target: {target}")
    print(f"  Initial q: ({np.degrees(q0[0]):.1f}°, {np.degrees(q0[1]):.1f}°, "
          f"{np.degrees(q0[2]):.1f}°)")

    # Without null-space optimization
    q = q0.copy()
    for i in range(max_iter):
        pos = fk_3link(q, l1, l2, l3)
        e = target - pos
        if np.linalg.norm(e) < tol:
            break
        J = jacobian_3link(q, l1, l2, l3)
        J_pinv = np.linalg.pinv(J)
        dq = alpha * J_pinv @ e
        q = q + dq
    q_no_null = q.copy()

    print(f"\nWithout null-space optimization:")
    print(f"  Final q: ({np.degrees(q[0]):.2f}°, {np.degrees(q[1]):.2f}°, "
          f"{np.degrees(q[2]):.2f}°)")
    print(f"  Position error: {np.linalg.norm(fk_3link(q, l1, l2, l3) - target):.2e}")

    # With null-space joint centering
    q = q0.copy()
    k_null = 0.5  # null-space gain
    for i in range(max_iter):
        pos = fk_3link(q, l1, l2, l3)
        e = target - pos
        if np.linalg.norm(e) < tol:
            break
        J = jacobian_3link(q, l1, l2, l3)
        J_pinv = np.linalg.pinv(J)

        # Primary task
        dq_primary = alpha * J_pinv @ e

        # Null-space projection: (I - J_pinv J) * q0_centering
        N = np.eye(3) - J_pinv @ J
        q_centering = k_null * (q_center - q)
        dq_null = N @ q_centering

        q = q + dq_primary + dq_null
    q_with_null = q.copy()

    print(f"\nWith null-space joint centering:")
    print(f"  Final q: ({np.degrees(q[0]):.2f}°, {np.degrees(q[1]):.2f}°, "
          f"{np.degrees(q[2]):.2f}°)")
    print(f"  Position error: {np.linalg.norm(fk_3link(q, l1, l2, l3) - target):.2e}")

    # Compare joint deviation from center
    dev_no_null = np.linalg.norm(np.degrees(q_no_null - q_center))
    dev_with_null = np.linalg.norm(np.degrees(q_with_null - q_center))
    print(f"\nJoint deviation from center (degrees, L2 norm):")
    print(f"  Without null-space: {dev_no_null:.2f}°")
    print(f"  With null-space:    {dev_with_null:.2f}°")
    print(f"\n  The null-space optimization uses the 1 DOF of redundancy to")
    print(f"  push joints toward their center positions while still reaching the target.")
    print(f"  This is the self-motion manifold: infinite configurations reach the")
    print(f"  same target, and we use the extra freedom for secondary objectives.")


if __name__ == "__main__":
    print("=" * 70)
    print("Lesson 04: Inverse Kinematics — Exercise Solutions")
    print("=" * 70)

    print("\n--- Exercise 1: 2-Link Analytical IK ---")
    exercise_1()

    print("\n--- Exercise 2: Numerical IK Comparison ---")
    exercise_2()

    print("\n--- Exercise 3: Singularity Analysis ---")
    exercise_3()

    print("\n--- Exercise 4: 6-DOF IK Framework ---")
    exercise_4()

    print("\n--- Exercise 5: Redundancy ---")
    exercise_5()

    print("\n" + "=" * 70)
    print("All exercises completed!")
    print("=" * 70)
