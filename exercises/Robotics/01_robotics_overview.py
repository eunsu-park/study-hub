"""
Exercises for Lesson 01: Robotics Overview
Topic: Robotics
Solutions to practice problems from the lesson.
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: Robot Classification
    Classify each robot by type, sub-type, and estimate DOF.
    """
    robots = [
        {
            "description": "A car-painting robot in a factory",
            "type": "Manipulator (Industrial)",
            "sub_type": "Articulated serial arm (e.g., 6-axis paint robot)",
            "dof": 6,
            "explanation": (
                "Factory painting robots are typically 6-DOF articulated arms "
                "mounted on a fixed base. 6 DOF allows full position and orientation "
                "control of the spray nozzle. Some have a 7th axis (linear rail) "
                "for extended reach along the car body."
            ),
        },
        {
            "description": "A self-driving delivery truck",
            "type": "Mobile Robot (Ground)",
            "sub_type": "Ackermann-steered wheeled robot",
            "dof": 3,
            "explanation": (
                "A truck moves in the plane (x, y, heading) giving 3 DOF for the "
                "body. The Ackermann steering has 2 control inputs (throttle and "
                "steering angle), making it a nonholonomic system — it cannot "
                "move sideways instantaneously."
            ),
        },
        {
            "description": "A hexacopter for aerial photography",
            "type": "Aerial Robot (UAV)",
            "sub_type": "Multi-rotor (hexacopter)",
            "dof": 6,
            "explanation": (
                "A hexacopter has 6 DOF (x, y, z, roll, pitch, yaw) in free "
                "flight. With 6 rotors and 6 DOF, it is fully actuated (unlike "
                "a quadcopter which is underactuated with only 4 inputs for 6 DOF). "
                "This gives it redundancy — it can maintain control even if one "
                "motor fails."
            ),
        },
        {
            "description": "An underwater vehicle inspecting oil pipelines",
            "type": "Underwater Robot (AUV/ROV)",
            "sub_type": "Remotely Operated Vehicle (ROV) or Autonomous Underwater Vehicle (AUV)",
            "dof": 6,
            "explanation": (
                "Underwater vehicles have 6 DOF like aerial robots (x, y, z, "
                "roll, pitch, yaw). They use thrusters in various configurations. "
                "Inspection ROVs typically have a manipulator arm (adding 4-6 more "
                "DOF) for interacting with structures."
            ),
        },
        {
            "description": "A robotic hand for prosthetics",
            "type": "Manipulator (Dexterous Hand)",
            "sub_type": "Anthropomorphic robotic hand / prosthetic end-effector",
            "dof": "15-20+",
            "explanation": (
                "A robotic prosthetic hand typically has 3-4 DOF per finger "
                "(MCP flexion/extension, MCP abduction/adduction, PIP, DIP) "
                "across 5 fingers, plus thumb opposition. In practice, many "
                "prosthetics use underactuated designs with fewer motors "
                "(e.g., 6-8 actuators) and mechanical coupling."
            ),
        },
    ]

    for i, robot in enumerate(robots, 1):
        print(f"\nRobot {i}: {robot['description']}")
        print(f"  Type:      {robot['type']}")
        print(f"  Sub-type:  {robot['sub_type']}")
        print(f"  DOF:       {robot['dof']}")
        print(f"  Notes:     {robot['explanation']}")


def exercise_2():
    """
    Exercise 2: Grubler's Formula
    5 links (including ground), 5 revolute joints, 1 prismatic joint.
    Planar mechanism: lambda = 3.
    """
    # Grubler's formula for planar mechanisms:
    # DOF = lambda * (N - 1) - sum_i (lambda - f_i)
    # where lambda = 3 (planar), N = number of links,
    # f_i = DOF of joint i (revolute = 1, prismatic = 1 for planar)

    lam = 3        # planar mechanism
    N = 5          # number of links (including ground)
    n_revolute = 5
    n_prismatic = 1
    J = n_revolute + n_prismatic  # total joints

    # Each revolute and prismatic joint has 1 DOF,
    # so each removes (lambda - 1) = 2 constraints
    dof = lam * (N - 1) - sum([(lam - 1)] * J)

    print(f"Grubler's formula: DOF = lambda*(N-1) - sum(lambda - f_i)")
    print(f"  lambda = {lam} (planar)")
    print(f"  N = {N} links (including ground)")
    print(f"  J = {J} joints ({n_revolute} revolute + {n_prismatic} prismatic)")
    print(f"  Each joint has f_i = 1 DOF, removing {lam - 1} constraints")
    print(f"  DOF = {lam}*({N}-1) - {J}*({lam}-1)")
    print(f"  DOF = {lam * (N - 1)} - {J * (lam - 1)}")
    print(f"  DOF = {dof}")
    print()
    if dof > 0:
        print(f"  Result: DOF = {dof} => This is a MECHANISM (can move)")
    elif dof == 0:
        print(f"  Result: DOF = {dof} => This is a STRUCTURE (rigid, no motion)")
    else:
        print(f"  Result: DOF = {dof} => This is an OVER-CONSTRAINED structure")


def exercise_3():
    """
    Exercise 3: Motor Sizing
    2-link planar robot, l1=0.4m, l2=0.3m, 2kg payload.
    """
    l1 = 0.4   # m
    l2 = 0.3   # m
    m = 2.0    # kg (payload)
    g = 9.81   # m/s^2
    safety_factor = 2.0
    omega_max = np.pi / 2  # rad/s

    # 1. Max gravitational torque at joint 1 (both links horizontal)
    # Torque = m * g * (l1 + l2) — full moment arm when both links are horizontal
    tau1_max = m * g * (l1 + l2)

    # 2. Max gravitational torque at joint 2 (link 2 horizontal)
    # Torque = m * g * l2
    tau2_max = m * g * l2

    # 3. Motor power: P = tau * omega (with safety factor)
    P1 = tau1_max * omega_max * safety_factor
    P2 = tau2_max * omega_max * safety_factor

    print("2-Link Planar Robot Motor Sizing")
    print(f"  Link lengths: l1={l1} m, l2={l2} m")
    print(f"  Payload: {m} kg")
    print(f"  Max joint velocity: {np.degrees(omega_max):.1f} deg/s")
    print(f"  Safety factor: {safety_factor}")
    print()
    print("1. Max gravitational torque at joint 1 (both links horizontal):")
    print(f"   tau1 = m*g*(l1+l2) = {m}*{g}*{l1+l2}")
    print(f"   tau1 = {tau1_max:.2f} N*m")
    print()
    print("2. Max gravitational torque at joint 2 (link 2 horizontal):")
    print(f"   tau2 = m*g*l2 = {m}*{g}*{l2}")
    print(f"   tau2 = {tau2_max:.2f} N*m")
    print()
    print("3. Motor power requirements (with safety factor 2):")
    print(f"   P1 = tau1 * omega_max * SF = {tau1_max:.2f} * {omega_max:.4f} * {safety_factor}")
    print(f"   P1 = {P1:.2f} W")
    print(f"   P2 = tau2 * omega_max * SF = {tau2_max:.2f} * {omega_max:.4f} * {safety_factor}")
    print(f"   P2 = {P2:.2f} W")


def exercise_4():
    """
    Exercise 4: Workspace Estimation
    2-DOF planar robot: l1=1.0m, l2=0.5m, full rotation.
    """
    l1 = 1.0
    l2 = 0.5

    # 1. Full rotation workspace — annular region
    r_outer = l1 + l2   # both links fully extended
    r_inner = abs(l1 - l2)  # link 2 folded back

    print("1. Workspace with full 360-degree rotation at both joints:")
    print(f"   Outer radius: l1 + l2 = {r_outer:.1f} m")
    print(f"   Inner radius: |l1 - l2| = {r_inner:.1f} m")
    print(f"   Shape: annular ring (donut)")
    print(f"   Area = pi * (R_outer^2 - R_inner^2)")
    area_full = np.pi * (r_outer**2 - r_inner**2)
    print(f"   Area = {area_full:.4f} m^2")

    # 2. Restricted joints: theta1 in [-90, 90], theta2 in [-180, 180]
    # theta2 in [-180, 180] is equivalent to full rotation for theta2
    # theta1 in [-90, 90] limits to a half annulus (right half-plane)
    print()
    print("2. Restricted workspace: theta1 in [-90, 90], theta2 in [-180, 180]:")
    print("   theta2 covers full range, so radial reach is unchanged.")
    print("   theta1 restricted to [-90, 90] => workspace covers right half-plane.")

    # Numerical workspace computation
    n_samples = 500
    theta1_full = np.linspace(-np.pi, np.pi, n_samples)
    theta2_range = np.linspace(-np.pi, np.pi, n_samples)

    theta1_restricted = np.linspace(-np.pi / 2, np.pi / 2, n_samples)

    # Full workspace points
    x_full, y_full = [], []
    for t1 in theta1_full:
        for t2 in theta2_range:
            x = l1 * np.cos(t1) + l2 * np.cos(t1 + t2)
            y = l1 * np.sin(t1) + l2 * np.sin(t1 + t2)
            x_full.append(x)
            y_full.append(y)

    # Restricted workspace points
    x_rest, y_rest = [], []
    for t1 in theta1_restricted:
        for t2 in theta2_range:
            x = l1 * np.cos(t1) + l2 * np.cos(t1 + t2)
            y = l1 * np.sin(t1) + l2 * np.sin(t1 + t2)
            x_rest.append(x)
            y_rest.append(y)

    # Approximate area via convex hull or bounding box
    # The restricted workspace is roughly half the full annulus
    area_restricted_approx = area_full / 2
    print(f"   Approximate area (half annulus): {area_restricted_approx:.4f} m^2")
    print(f"   The workspace is approximately the right half of the annulus,")
    print(f"   but with curved boundaries due to the combined joint effects.")
    print(f"   Numerically sampled {len(x_rest)} workspace points.")


def exercise_5():
    """
    Exercise 5: Research
    Summary of a specific robot in a chosen application domain.
    This exercise is research-based; we provide a sample answer for a surgical robot.
    """
    report = {
        "domain": "Medical / Surgical Robotics",
        "robot_name": "da Vinci Xi Surgical System",
        "manufacturer": "Intuitive Surgical",
        "classification": {
            "type": "Manipulator (Teleoperated)",
            "sub_type": "Multi-arm surgical system with 4 articulated arms",
            "dof": "7 DOF per arm (6 for positioning + 1 EndoWrist articulation)",
        },
        "sensors": [
            "Stereoscopic 3D HD camera (endoscope) for surgeon visualization",
            "Joint encoders for precise arm positioning",
            "Force/torque sensors for trocar interaction detection",
            "Instrument tip position tracking",
        ],
        "actuators": [
            "Electric motors at each joint (cable-driven for compact design)",
            "EndoWrist instruments with 7 DOF for dexterous manipulation",
        ],
        "application": (
            "Minimally invasive surgery (prostatectomy, cardiac surgery, "
            "gynecological procedures). The robot provides: (1) tremor filtering "
            "for sub-millimeter precision, (2) motion scaling (1:5 ratio reduces "
            "surgeon hand motion), (3) wristed instruments with more dexterity "
            "than the human hand in confined spaces, (4) 3D magnified vision."
        ),
        "why_robot": (
            "Human surgeons have hand tremor (~100 micron amplitude), limited "
            "dexterity through small incisions, and fatigue over long procedures. "
            "The robot eliminates tremor, scales motion for precision, provides "
            "7-DOF wristed instruments through 8mm ports, and the surgeon works "
            "seated at an ergonomic console."
        ),
        "limitations": [
            "No haptic (force) feedback to the surgeon — reliance on visual cues",
            "Very high cost (~$2M per system + $3K/procedure in instrument costs)",
            "Large footprint in the operating room",
            "Requires specialized training for the surgical team",
            "Communication latency limits telesurgery over long distances",
        ],
    }

    print("=== Robot Research Report ===")
    print(f"Domain: {report['domain']}")
    print(f"Robot: {report['robot_name']}")
    print(f"Manufacturer: {report['manufacturer']}")
    print(f"\nClassification:")
    for k, v in report['classification'].items():
        print(f"  {k}: {v}")
    print(f"\nKey Sensors:")
    for s in report['sensors']:
        print(f"  - {s}")
    print(f"\nKey Actuators:")
    for a in report['actuators']:
        print(f"  - {a}")
    print(f"\nPrimary Application:\n  {report['application']}")
    print(f"\nWhy Robot > Human:\n  {report['why_robot']}")
    print(f"\nCurrent Limitations:")
    for lim in report['limitations']:
        print(f"  - {lim}")


if __name__ == "__main__":
    print("=" * 70)
    print("Lesson 01: Robotics Overview — Exercise Solutions")
    print("=" * 70)

    print("\n--- Exercise 1: Robot Classification ---")
    exercise_1()

    print("\n--- Exercise 2: Grubler's Formula ---")
    exercise_2()

    print("\n--- Exercise 3: Motor Sizing ---")
    exercise_3()

    print("\n--- Exercise 4: Workspace Estimation ---")
    exercise_4()

    print("\n--- Exercise 5: Research ---")
    exercise_5()

    print("\n" + "=" * 70)
    print("All exercises completed!")
    print("=" * 70)
