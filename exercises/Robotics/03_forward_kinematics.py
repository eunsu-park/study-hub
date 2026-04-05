"""
Exercises for Lesson 03: Forward Kinematics
Topic: Robotics
Solutions to practice problems from the lesson.
"""

import numpy as np


def dh_transform(theta, d, a, alpha):
    """Compute the 4x4 DH transformation matrix for one joint."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,   sa,       ca,      d     ],
        [0,   0,        0,       1     ]
    ])


def forward_kinematics(dh_params, joint_values):
    """
    Compute FK from DH parameters and joint values.
    dh_params: list of (d, a, alpha, joint_type) per joint
    joint_values: list of joint values (theta for R, d for P)
    Returns 4x4 homogeneous transformation matrix.
    """
    T = np.eye(4)
    for i, (d, a, alpha, jtype) in enumerate(dh_params):
        if jtype == 'R':
            theta = joint_values[i]
            d_val = d
        else:  # Prismatic
            theta = 0.0
            d_val = joint_values[i]
        T = T @ dh_transform(theta, d_val, a, alpha)
    return T


def exercise_1():
    """
    Exercise 1: DH Parameter Assignment
    3-DOF planar robot: 3 revolute joints, l1=0.5, l2=0.4, l3=0.3.
    """
    l1, l2, l3 = 0.5, 0.4, 0.3

    print("DH Parameter Table (3-DOF Planar Robot):")
    print(f"{'Joint':>6} | {'theta':>10} | {'d':>6} | {'a':>6} | {'alpha':>8}")
    print("-" * 50)
    print(f"{'1':>6} | {'theta_1':>10} | {'0':>6} | {l1:>6} | {'0':>8}")
    print(f"{'2':>6} | {'theta_2':>10} | {'0':>6} | {l2:>6} | {'0':>8}")
    print(f"{'3':>6} | {'theta_3':>10} | {'0':>6} | {l3:>6} | {'0':>8}")

    # FK: x = l1*c1 + l2*c12 + l3*c123
    #     y = l1*s1 + l2*s12 + l3*s123
    #     phi = theta1 + theta2 + theta3

    dh_params = [
        (0, l1, 0, 'R'),
        (0, l2, 0, 'R'),
        (0, l3, 0, 'R'),
    ]

    # Numerical verification
    t1 = np.radians(30)
    t2 = np.radians(45)
    t3 = np.radians(-30)
    q = [t1, t2, t3]

    T = forward_kinematics(dh_params, q)

    # Analytical
    c1 = np.cos(t1)
    s1 = np.sin(t1)
    c12 = np.cos(t1 + t2)
    s12 = np.sin(t1 + t2)
    c123 = np.cos(t1 + t2 + t3)
    s123 = np.sin(t1 + t2 + t3)

    x_analytical = l1 * c1 + l2 * c12 + l3 * c123
    y_analytical = l1 * s1 + l2 * s12 + l3 * s123
    phi = t1 + t2 + t3

    print(f"\nVerification at theta = (30°, 45°, -30°):")
    print(f"  Analytical: x={x_analytical:.6f}, y={y_analytical:.6f}, "
          f"phi={np.degrees(phi):.1f}°")
    print(f"  DH FK:      x={T[0, 3]:.6f}, y={T[1, 3]:.6f}")
    print(f"  Match: {np.allclose([x_analytical, y_analytical], [T[0, 3], T[1, 3]])}")


def exercise_2():
    """
    Exercise 2: SCARA Robot FK
    """
    # SCARA DH parameters
    # Joint | theta    | d    | a   | alpha | Type
    # 1     | theta1   | 0.5  | 0.4 | 0     | R
    # 2     | theta2   | 0    | 0.3 | pi    | R
    # 3     | 0        | d3   | 0   | 0     | P
    # 4     | theta4   | 0    | 0   | 0     | R

    dh_params = [
        (0.5, 0.4, 0, 'R'),
        (0, 0.3, np.pi, 'R'),
        (0, 0, 0, 'P'),
        (0, 0, 0, 'R'),
    ]

    t1 = np.radians(45)
    t2 = np.radians(-30)
    d3 = 0.15
    t4 = np.radians(60)

    # For SCARA: joint 3 is prismatic (d3 is the variable)
    # Custom FK for SCARA
    T = np.eye(4)
    # Joint 1: R
    T = T @ dh_transform(t1, 0.5, 0.4, 0)
    # Joint 2: R
    T = T @ dh_transform(t2, 0, 0.3, np.pi)
    # Joint 3: P (theta=0, d=d3)
    T = T @ dh_transform(0, d3, 0, 0)
    # Joint 4: R
    T = T @ dh_transform(t4, 0, 0, 0)

    print("SCARA Robot FK")
    print(f"  Joint values: theta1=45°, theta2=-30°, d3=0.15m, theta4=60°")
    print(f"\nEnd-effector pose (T):")
    print(T.round(6))
    print(f"\nPosition: ({T[0, 3]:.4f}, {T[1, 3]:.4f}, {T[2, 3]:.4f})")

    # SCARA workspace is a planar annulus (top view)
    # R_inner = |a1 - a2|, R_outer = a1 + a2
    a1, a2 = 0.4, 0.3
    print(f"\nWorkspace (top-down view):")
    print(f"  Annular ring with:")
    print(f"  Inner radius = |a1 - a2| = {abs(a1 - a2):.1f} m")
    print(f"  Outer radius = a1 + a2 = {a1 + a2:.1f} m")
    print(f"  Vertical range determined by d3 limits")
    print(f"  Shape: cylindrical annulus (flat top/bottom, ring cross-section)")


def exercise_3():
    """
    Exercise 3: 6-DOF Verification (PUMA 560-like)
    """
    # PUMA 560-like DH parameters (simplified)
    d1, d4, d6 = 0.67, 0.433, 0.056
    a2, a3 = 0.432, -0.02

    def puma_fk(q):
        """FK for PUMA 560-like robot."""
        T = np.eye(4)
        # Joint 1: theta=q[0], d=d1, a=0, alpha=-pi/2
        T = T @ dh_transform(q[0], d1, 0, -np.pi/2)
        # Joint 2: theta=q[1], d=0, a=a2, alpha=0
        T = T @ dh_transform(q[1], 0, a2, 0)
        # Joint 3: theta=q[2], d=0, a=a3, alpha=-pi/2 (approximate)
        T = T @ dh_transform(q[2], 0, a3, -np.pi/2)
        # Joint 4: theta=q[3], d=d4, a=0, alpha=pi/2
        T = T @ dh_transform(q[3], d4, 0, np.pi/2)
        # Joint 5: theta=q[4], d=0, a=0, alpha=-pi/2
        T = T @ dh_transform(q[4], 0, 0, -np.pi/2)
        # Joint 6: theta=q[5], d=d6, a=0, alpha=0
        T = T @ dh_transform(q[5], d6, 0, 0)
        return T

    # Case 1: q = [0, -90°, 0, 0, 0, 0] (arm pointing up)
    q1 = [0, np.radians(-90), 0, 0, 0, 0]
    T1 = puma_fk(q1)
    print("Case 1: q = [0, -90°, 0, 0, 0, 0]")
    print(f"  End-effector position: ({T1[0, 3]:.4f}, {T1[1, 3]:.4f}, {T1[2, 3]:.4f})")
    print(f"  Geometric sense: With q2=-90°, the upper arm points up.")
    print(f"  The EE should be approximately above the base.")
    print(f"  x ≈ small (near zero), z ≈ high")

    # Case 2: q = [90°, 0, 0, 0, 0, 0] (base rotated 90°)
    q2 = [np.radians(90), 0, 0, 0, 0, 0]
    T2 = puma_fk(q2)
    print(f"\nCase 2: q = [90°, 0, 0, 0, 0, 0]")
    print(f"  End-effector position: ({T2[0, 3]:.4f}, {T2[1, 3]:.4f}, {T2[2, 3]:.4f})")

    # Compare with q = [0, 0, 0, 0, 0, 0]
    q0 = [0, 0, 0, 0, 0, 0]
    T0 = puma_fk(q0)
    print(f"  Reference (q=0): ({T0[0, 3]:.4f}, {T0[1, 3]:.4f}, {T0[2, 3]:.4f})")
    print(f"  Geometric sense: Rotating base 90° should swap x↔y coordinates")
    print(f"  of the home position.")


def exercise_4():
    """
    Exercise 4: Workspace Visualization
    2-link planar robot, l1=1.0, l2=0.8.
    """
    l1, l2 = 1.0, 0.8
    n = 200

    # Full workspace
    t1_full = np.linspace(-np.pi, np.pi, n)
    t2_full = np.linspace(-np.pi, np.pi, n)

    x_full, y_full = [], []
    for t1 in t1_full:
        for t2 in t2_full:
            x = l1 * np.cos(t1) + l2 * np.cos(t1 + t2)
            y = l1 * np.sin(t1) + l2 * np.sin(t1 + t2)
            x_full.append(x)
            y_full.append(y)
    x_full = np.array(x_full)
    y_full = np.array(y_full)

    # Restricted workspace (theta2 in [-90°, 90°])
    t2_restricted = np.linspace(-np.pi / 2, np.pi / 2, n)
    x_rest, y_rest = [], []
    for t1 in t1_full:
        for t2 in t2_restricted:
            x = l1 * np.cos(t1) + l2 * np.cos(t1 + t2)
            y = l1 * np.sin(t1) + l2 * np.sin(t1 + t2)
            x_rest.append(x)
            y_rest.append(y)
    x_rest = np.array(x_rest)
    y_rest = np.array(y_rest)

    # Area calculations
    r_outer = l1 + l2
    r_inner_full = abs(l1 - l2)
    area_full = np.pi * (r_outer**2 - r_inner_full**2)

    # For restricted theta2, the inner radius changes
    # Min reach with theta2 restricted to [-90, 90]:
    # r_min = sqrt(l1^2 + l2^2 - 2*l1*l2*cos(90)) = sqrt(l1^2 + l2^2)
    # Actually min reach is at theta2 = +-90: r = sqrt(l1^2 + l2^2)
    r_inner_rest = np.sqrt(l1**2 + l2**2 - 2 * l1 * l2 * np.cos(np.pi / 2))
    # Outer radius same: theta2=0 gives l1+l2
    area_rest = np.pi * (r_outer**2 - r_inner_rest**2)

    print("Workspace Analysis: 2-link planar robot (l1=1.0, l2=0.8)")
    print(f"\n1. Full workspace (both joints: ±180°):")
    print(f"   Outer radius = l1+l2 = {r_outer:.1f} m")
    print(f"   Inner radius = |l1-l2| = {r_inner_full:.1f} m")
    print(f"   Area = π(R²-r²) = {area_full:.4f} m²")
    print(f"   Sampled {len(x_full)} points")
    r_full = np.sqrt(x_full**2 + y_full**2)
    print(f"   Reach range: [{r_full.min():.4f}, {r_full.max():.4f}] m")

    print(f"\n2. Restricted workspace (theta2: ±90°):")
    print(f"   Outer radius = {r_outer:.1f} m (still at theta2=0)")
    print(f"   Inner radius = sqrt(l1²+l2²) ≈ {r_inner_rest:.4f} m (at theta2=±90°)")
    print(f"   Approximate area = {area_rest:.4f} m²")
    r_rest = np.sqrt(x_rest**2 + y_rest**2)
    print(f"   Reach range: [{r_rest.min():.4f}, {r_rest.max():.4f}] m")
    print(f"   Area ratio (restricted/full): {area_rest / area_full:.3f}")


def exercise_5():
    """
    Exercise 5: Mixed Joint Robot (R-P-R)
    Design a 3-DOF Revolute-Prismatic-Revolute robot.
    """
    # DH parameters for R-P-R robot
    # Joint 1: Revolute — base rotation about z
    # Joint 2: Prismatic — extend along the rotated direction
    # Joint 3: Revolute — wrist rotation
    a1 = 0.3  # base offset
    d2_min, d2_max = 0.2, 1.0  # prismatic range
    a3 = 0.2  # wrist link

    print("R-P-R Robot Design")
    print(f"\nDH Parameter Table:")
    print(f"{'Joint':>6} | {'theta':>10} | {'d':>10} | {'a':>6} | {'alpha':>8} | {'Type':>5}")
    print("-" * 60)
    print(f"{'1':>6} | {'theta_1':>10} | {'0':>10} | {a1:>6} | {'0':>8} | {'R':>5}")
    print(f"{'2':>6} | {'0':>10} | {'d_2':>10} | {'0':>6} | {'0':>8} | {'P':>5}")
    print(f"{'3':>6} | {'theta_3':>10} | {'0':>10} | {a3:>6} | {'0':>8} | {'R':>5}")

    # Test configurations
    configs = [
        ("Home", [0, 0.5, 0]),
        ("Extended", [0, 1.0, 0]),
        ("Rotated", [np.radians(90), 0.6, np.radians(45)]),
    ]

    for name, q in configs:
        T = np.eye(4)
        T = T @ dh_transform(q[0], 0, a1, 0)   # Joint 1 (R)
        T = T @ dh_transform(0, q[1], 0, 0)     # Joint 2 (P)
        T = T @ dh_transform(q[2], 0, a3, 0)    # Joint 3 (R)

        pos = T[:3, 3]
        print(f"\n  Config '{name}': theta1={np.degrees(q[0]):.0f}°, "
              f"d2={q[1]:.2f}m, theta3={np.degrees(q[2]):.0f}°")
        print(f"  End-effector: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")

    # Workspace comparison
    print(f"\nWorkspace comparison:")
    print(f"  R-P-R: annular region where the radial extent is controlled by")
    print(f"  the prismatic joint. Range: [{a1 + d2_min + a3:.2f}, "
          f"{a1 + d2_max + a3:.2f}] m")
    print(f"  Unlike the all-revolute 3R planar robot whose workspace boundary")
    print(f"  is determined by link lengths, the R-P-R workspace boundary is")
    print(f"  determined by the prismatic joint stroke.")
    print(f"  The R-P-R workspace is a full annulus with adjustable inner radius.")


if __name__ == "__main__":
    print("=" * 70)
    print("Lesson 03: Forward Kinematics — Exercise Solutions")
    print("=" * 70)

    print("\n--- Exercise 1: DH Parameter Assignment ---")
    exercise_1()

    print("\n--- Exercise 2: SCARA Robot ---")
    exercise_2()

    print("\n--- Exercise 3: 6-DOF Verification ---")
    exercise_3()

    print("\n--- Exercise 4: Workspace Visualization ---")
    exercise_4()

    print("\n--- Exercise 5: Mixed Joint Robot ---")
    exercise_5()

    print("\n" + "=" * 70)
    print("All exercises completed!")
    print("=" * 70)
