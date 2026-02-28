"""
Exercises for Lesson 05: Velocity Kinematics and the Jacobian
Topic: Robotics
Solutions to practice problems from the lesson.
"""

import numpy as np


def jacobian_2link(q, l1, l2):
    """Analytical Jacobian for 2-link planar robot (2x2)."""
    s1, c1 = np.sin(q[0]), np.cos(q[0])
    s12, c12 = np.sin(q[0] + q[1]), np.cos(q[0] + q[1])
    return np.array([
        [-l1*s1 - l2*s12, -l2*s12],
        [ l1*c1 + l2*c12,  l2*c12]
    ])


def jacobian_3link(q, l1, l2, l3):
    """Analytical Jacobian for 3-link planar robot (2x3)."""
    s1, c1 = np.sin(q[0]), np.cos(q[0])
    s12, c12 = np.sin(q[0] + q[1]), np.cos(q[0] + q[1])
    s123, c123 = np.sin(q[0] + q[1] + q[2]), np.cos(q[0] + q[1] + q[2])
    return np.array([
        [-l1*s1 - l2*s12 - l3*s123, -l2*s12 - l3*s123, -l3*s123],
        [ l1*c1 + l2*c12 + l3*c123,  l2*c12 + l3*c123,  l3*c123]
    ])


def fk_2link(q, l1, l2):
    """Forward kinematics for 2-link planar robot."""
    x = l1 * np.cos(q[0]) + l2 * np.cos(q[0] + q[1])
    y = l1 * np.sin(q[0]) + l2 * np.sin(q[0] + q[1])
    return np.array([x, y])


def fk_3link(q, l1, l2, l3):
    """Forward kinematics for 3-link planar robot."""
    x = (l1 * np.cos(q[0]) + l2 * np.cos(q[0] + q[1])
         + l3 * np.cos(q[0] + q[1] + q[2]))
    y = (l1 * np.sin(q[0]) + l2 * np.sin(q[0] + q[1])
         + l3 * np.sin(q[0] + q[1] + q[2]))
    return np.array([x, y])


def numerical_jacobian(fk_func, q, delta=1e-7):
    """Compute Jacobian via finite differences."""
    n = len(q)
    f0 = fk_func(q)
    m = len(f0)
    J = np.zeros((m, n))
    for j in range(n):
        q_plus = q.copy()
        q_plus[j] += delta
        J[:, j] = (fk_func(q_plus) - f0) / delta
    return J


def exercise_1():
    """
    Exercise 1: Jacobian Derivation for 3-link planar robot.
    """
    l1, l2, l3 = 0.5, 0.5, 0.5
    q = np.radians([30.0, 45.0, -60.0])

    print(f"3-link planar robot: l1=l2=l3={l1}")
    print(f"q = (30°, 45°, -60°)")

    # Analytical Jacobian
    J_analytical = jacobian_3link(q, l1, l2, l3)
    print(f"\nAnalytical Jacobian (2x3):")
    print(J_analytical.round(6))

    # Finite difference verification
    fk_func = lambda q: fk_3link(q, l1, l2, l3)
    J_numerical = numerical_jacobian(fk_func, q)
    print(f"\nNumerical Jacobian (finite differences, delta=1e-7):")
    print(J_numerical.round(6))

    print(f"\nMax difference: {np.max(np.abs(J_analytical - J_numerical)):.2e}")

    # Null space
    U, S, Vt = np.linalg.svd(J_analytical)
    # Null space = last row of Vt (since J is 2x3, null space is 1-dimensional)
    null_vec = Vt[-1]
    print(f"\nSingular values: {S.round(6)}")
    print(f"Null space vector: {null_vec.round(6)}")
    print(f"  Verification J*v_null = {(J_analytical @ null_vec).round(10)}")
    print(f"\n  Physical meaning: The null space motion corresponds to the")
    print(f"  self-motion of the redundant robot — all three joints move")
    print(f"  simultaneously such that the end-effector stays stationary.")
    print(f"  This is the internal reconfiguration DOF.")


def exercise_2():
    """
    Exercise 2: Singularity Analysis for 2-link planar robot.
    """
    l1, l2 = 1.0, 0.8

    print(f"2-link robot: l1={l1}, l2={l2}")
    print(f"det(J) = l1*l2*sin(q2)")
    print(f"Singular when q2 = 0° (fully extended) or q2 = 180° (fully folded)")

    q_singular = np.array([np.radians(45), 0.0])
    J = jacobian_2link(q_singular, l1, l2)
    U, S, Vt = np.linalg.svd(J)

    print(f"\nAt q = (45°, 0°):")
    print(f"  J = {J.round(6)}")
    print(f"  Singular values: {S.round(6)}")
    print(f"  Lost direction (left sing. vec): u2 = {U[:, -1].round(6)}")

    # Attempt to move in lost direction using pseudo-inverse
    lost_dir = U[:, -1]
    J_pinv = np.linalg.pinv(J)
    dq = J_pinv @ lost_dir
    print(f"\n  Pseudo-inverse for lost direction:")
    print(f"    Required dq = {dq.round(4)}")
    print(f"    Joint velocity magnitude: {np.linalg.norm(dq):.4f}")
    print(f"    This is very large — near-singular, pseudo-inverse produces huge velocities.")

    # DLS with different lambda values
    print(f"\n  DLS comparison:")
    for lam in [0.01, 0.1, 0.5, 1.0]:
        JJT = J @ J.T
        dq_dls = J.T @ np.linalg.solve(JJT + lam**2 * np.eye(2), lost_dir)
        actual_v = J @ dq_dls
        print(f"    λ={lam:.2f}: |dq|={np.linalg.norm(dq_dls):.4f}, "
              f"actual v={actual_v.round(4)}, "
              f"tracking error={np.linalg.norm(lost_dir - actual_v):.4f}")
    print(f"\n  DLS trades off tracking accuracy for bounded joint velocities.")
    print(f"  Larger λ → smaller joint velocities but worse tracking.")


def exercise_3():
    """
    Exercise 3: Force Mapping
    2-link arm, l1=l2=0.5, q=(0°, 90°), 20N in +x direction.
    """
    l1, l2 = 0.5, 0.5
    q = np.radians([0.0, 90.0])

    J = jacobian_2link(q, l1, l2)
    F_desired = np.array([20.0, 0.0])  # 20 N in +x

    # tau = J^T * F
    tau = J.T @ F_desired

    print(f"Configuration: q = (0°, 90°)")
    print(f"Jacobian:")
    print(f"  {J.round(6)}")
    print(f"\n1. Required torques for 20 N in +x:")
    print(f"   tau = J^T * F = {tau.round(4)}")
    print(f"   tau_1 = {tau[0]:.4f} N*m")
    print(f"   tau_2 = {tau[1]:.4f} N*m")

    # Check if tau_1 < 15 N*m
    tau_max = 15.0
    print(f"\n2. Can joint 1 handle it? |tau_1| = {abs(tau[0]):.4f} N*m vs max = {tau_max} N*m")
    print(f"   {'YES' if abs(tau[0]) <= tau_max else 'NO'}")

    # Maximum force in +x direction
    # tau = J^T * F, F = [Fx, 0]
    # tau = J^T * [Fx, 0] = Fx * J^T[:,0]
    # Subject to |tau_i| <= tau_max
    # Fx_max = min(tau_max / |J^T[i,0]|) for each joint
    JT_col = J.T[:, 0]  # First column of J^T (maps Fx to torques)
    Fx_max = min(tau_max / abs(JT_col[i]) for i in range(2) if abs(JT_col[i]) > 1e-10)
    print(f"\n3. Maximum Fx with |tau_i| <= {tau_max} N*m:")
    print(f"   J^T column for Fx: {JT_col.round(4)}")
    print(f"   Max Fx = {Fx_max:.4f} N")


def exercise_4():
    """
    Exercise 4: Manipulability Ellipsoid
    2-link robot, l1=1.0, l2=0.5, various theta2 values.
    """
    l1, l2 = 1.0, 0.5
    theta1 = 0.0

    print(f"2-link robot: l1={l1}, l2={l2}, theta1=0°")
    print(f"\n{'theta2':>8} | {'sigma1':>8} {'sigma2':>8} | "
          f"{'w=s1*s2':>10} | {'isotropy':>10}")
    print("-" * 60)

    best_isotropy = 0
    best_theta2 = 0

    for t2_deg in [30, 60, 90, 120, 150]:
        q = np.array([theta1, np.radians(t2_deg)])
        J = jacobian_2link(q, l1, l2)
        U, S, Vt = np.linalg.svd(J)

        w = S[0] * S[1]  # manipulability measure
        isotropy = S[1] / S[0] if S[0] > 1e-10 else 0  # 1 = isotropic

        if isotropy > best_isotropy:
            best_isotropy = isotropy
            best_theta2 = t2_deg

        print(f"{t2_deg:>7}° | {S[0]:>8.4f} {S[1]:>8.4f} | "
              f"{w:>10.6f} | {isotropy:>10.4f}")

    print(f"\nMost isotropic at theta2 = {best_theta2}° (isotropy = {best_isotropy:.4f})")

    # Does optimal theta2 depend on theta1?
    print(f"\nDoes isotropy depend on theta1?")
    for t1_deg in [0, 45, 90]:
        q = np.array([np.radians(t1_deg), np.radians(best_theta2)])
        J = jacobian_2link(q, l1, l2)
        _, S, _ = np.linalg.svd(J)
        iso = S[1] / S[0]
        print(f"  theta1={t1_deg:>3}°, theta2={best_theta2}°: isotropy = {iso:.6f}")
    print(f"  Isotropy is independent of theta1 because the Jacobian's")
    print(f"  singular values depend only on the relative angle theta2.")
    print(f"  Changing theta1 rotates the ellipsoid but doesn't change its shape.")


def exercise_5():
    """
    Exercise 5: Numerical Jacobian Verification for PUMA 560-like robot.
    """
    # Simplified PUMA FK using DH
    d1, d4, d6 = 0.67, 0.433, 0.056
    a2, a3 = 0.432, -0.02

    def dh_transform(theta, d, a, alpha):
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([
            [ct, -st*ca,  st*sa, a*ct],
            [st,  ct*ca, -ct*sa, a*st],
            [0,   sa,     ca,    d   ],
            [0,   0,      0,     1   ]
        ])

    def puma_fk_pos(q):
        """Returns 3D position of PUMA end-effector."""
        T = np.eye(4)
        T = T @ dh_transform(q[0], d1, 0, -np.pi/2)
        T = T @ dh_transform(q[1], 0, a2, 0)
        T = T @ dh_transform(q[2], 0, a3, -np.pi/2)
        T = T @ dh_transform(q[3], d4, 0, np.pi/2)
        T = T @ dh_transform(q[4], 0, 0, -np.pi/2)
        T = T @ dh_transform(q[5], d6, 0, 0)
        return T[:3, 3]

    print("Numerical Jacobian for PUMA 560-like robot (position only, 3x6)")
    print("Testing step sizes and random configurations:\n")

    # Test different step sizes
    q_test = np.array([0.3, -0.5, 0.8, 0.2, -0.4, 0.1])
    best_delta = None
    best_err = float('inf')

    print(f"Config: q = {np.degrees(q_test).round(1)}°")
    print(f"\n{'delta':>12} | {'max element diff':>18} | {'Frobenius diff':>15}")
    print("-" * 55)

    # Reference: very small delta
    J_ref = numerical_jacobian(puma_fk_pos, q_test, delta=1e-8)

    for exp in range(-4, -11, -1):
        delta = 10.0 ** exp
        J_num = numerical_jacobian(puma_fk_pos, q_test, delta=delta)
        max_diff = np.max(np.abs(J_num - J_ref))
        frob_diff = np.linalg.norm(J_num - J_ref, 'fro')
        print(f"  {delta:>10.0e} | {max_diff:>18.2e} | {frob_diff:>15.2e}")
        if frob_diff < best_err:
            best_err = frob_diff
            best_delta = delta

    print(f"\n  Best delta (vs reference at 1e-8): {best_delta:.0e}")
    print(f"\n  Why accuracy degrades for very small delta:")
    print(f"  Finite differences compute (f(x+h) - f(x))/h. When h is very small,")
    print(f"  f(x+h) and f(x) differ by only a few ULPs (units in the last place),")
    print(f"  and the subtraction suffers catastrophic cancellation. The rounding error")
    print(f"  is O(eps/h) while truncation error is O(h). The optimal h balances both")
    print(f"  at approximately h = sqrt(eps) ≈ 1.5e-8 for float64.")

    # Test at random configs
    np.random.seed(42)
    print(f"\nComparison at 5 random configurations (delta=1e-7):")
    for i in range(5):
        q = np.random.uniform(-np.pi, np.pi, 6)
        J_num = numerical_jacobian(puma_fk_pos, q, delta=1e-7)
        print(f"  Config {i + 1}: J shape = {J_num.shape}, "
              f"||J||_F = {np.linalg.norm(J_num, 'fro'):.4f}, "
              f"cond(J) = {np.linalg.cond(J_num):.2f}")


if __name__ == "__main__":
    print("=" * 70)
    print("Lesson 05: Velocity Kinematics — Exercise Solutions")
    print("=" * 70)

    print("\n--- Exercise 1: Jacobian Derivation ---")
    exercise_1()

    print("\n--- Exercise 2: Singularity Analysis ---")
    exercise_2()

    print("\n--- Exercise 3: Force Mapping ---")
    exercise_3()

    print("\n--- Exercise 4: Manipulability Ellipsoid ---")
    exercise_4()

    print("\n--- Exercise 5: Numerical Jacobian Verification ---")
    exercise_5()

    print("\n" + "=" * 70)
    print("All exercises completed!")
    print("=" * 70)
