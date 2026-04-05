"""
Exercises for Lesson 10: Sensors and Perception
Topic: Robotics
Solutions to practice problems from the lesson.
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: Encoder Resolution Analysis
    6-DOF robot, 1024 CPR encoders, quadrature decoding, 100:1 gear ratio.
    """
    cpr = 1024
    quadrature_factor = 4  # quadrature decoding multiplies by 4
    gear_ratio = 100
    effective_cpr = cpr * quadrature_factor * gear_ratio
    L_arm = 1.0  # 1m from base to end-effector

    # Angular resolution per joint
    angle_per_count = 360.0 / effective_cpr  # degrees

    print("Encoder Resolution Analysis:")
    print(f"  CPR: {cpr}")
    print(f"  Quadrature decoding: x{quadrature_factor}")
    print(f"  Gear ratio: {gear_ratio}:1")
    print(f"  Effective counts/rev: {effective_cpr:,}")
    print(f"\n  Angular resolution per joint:")
    print(f"    {angle_per_count:.6f}° = {angle_per_count * 3600:.4f} arcsec")
    print(f"    {np.radians(angle_per_count):.8f} rad")

    # Worst-case EE position resolution
    # At maximum arm extension (L=1m), the position error due to
    # one encoder count at the base joint is: delta_x ≈ L * delta_theta
    delta_theta = np.radians(angle_per_count)
    delta_x_worst = L_arm * delta_theta

    print(f"\n  Worst-case EE position resolution (1m arm):")
    print(f"    delta_x ≈ L * delta_theta = {L_arm} * {delta_theta:.2e}")
    print(f"    delta_x ≈ {delta_x_worst * 1000:.4f} mm = {delta_x_worst * 1e6:.1f} μm")
    print(f"\n  Note: This is for a single joint. With 6 joints,")
    print(f"  errors accumulate. Worst case (all errors aligned):")
    print(f"    ~{6 * delta_x_worst * 1000:.4f} mm")
    print(f"  RMS case (random errors): ~√6 * {delta_x_worst * 1000:.4f} = "
          f"{np.sqrt(6) * delta_x_worst * 1000:.4f} mm")


def exercise_2():
    """
    Exercise 2: Camera Calibration (Pinhole Model)
    Project 3D points to pixels, then back-project.
    """
    # Intrinsic parameters
    fx, fy = 500.0, 500.0
    cx, cy = 320.0, 240.0
    K = np.array([[fx, 0, cx],
                   [0, fy, cy],
                   [0, 0, 1]])

    # 3D points in camera frame
    points_3d = np.array([
        [1, 0, 5],
        [0, 1, 5],
        [-1, -1, 10]
    ], dtype=float)

    print("Pinhole Camera Model: Projection and Back-projection")
    print(f"  K = [[{fx}, 0, {cx}], [0, {fy}, {cy}], [0, 0, 1]]")

    # Project to pixel coordinates: p = K * [X/Z, Y/Z, 1]^T
    print(f"\nProjection (3D → 2D):")
    print(f"  {'Point 3D':>20} | {'Pixel (u, v)':>20}")
    print("  " + "-" * 45)
    pixels = []
    for p in points_3d:
        X, Y, Z = p
        u = fx * X / Z + cx
        v = fy * Y / Z + cy
        pixels.append((u, v))
        print(f"  ({X:5.1f}, {Y:5.1f}, {Z:5.1f}) | ({u:8.2f}, {v:8.2f})")

    # Back-project using known depths
    print(f"\nBack-projection (2D + depth → 3D):")
    print(f"  {'Pixel + depth':>30} | {'Recovered 3D':>20} | {'Error':>10}")
    print("  " + "-" * 70)
    for i, ((u, v), p_orig) in enumerate(zip(pixels, points_3d)):
        Z = p_orig[2]  # known depth
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        error = np.linalg.norm(np.array([X, Y, Z]) - p_orig)
        print(f"  ({u:8.2f}, {v:8.2f}, Z={Z:4.1f}) | "
              f"({X:6.3f}, {Y:6.3f}, {Z:4.1f}) | {error:.2e}")

    print(f"\n  Back-projection recovers original 3D points exactly when")
    print(f"  the depth is known. Without depth, a 2D pixel maps to a ray")
    print(f"  in 3D space — infinitely many 3D points can project to the")
    print(f"  same pixel.")


def exercise_3():
    """
    Exercise 3: LiDAR Scan Matching (ICP)
    """
    np.random.seed(42)

    # Generate a simple environment: rectangular room with circular obstacle
    n_points = 100
    # Room walls
    wall_pts = []
    for i in range(25):
        t = i / 24.0
        wall_pts.append([t * 5, 0])       # bottom
        wall_pts.append([t * 5, 5])       # top
        wall_pts.append([0, t * 5])       # left
        wall_pts.append([5, t * 5])       # right
    scan1 = np.array(wall_pts)

    # Apply known transform to create scan2
    true_theta = np.radians(5)  # 5 degrees rotation
    true_t = np.array([0.1, 0.05])  # small translation

    R_true = np.array([[np.cos(true_theta), -np.sin(true_theta)],
                        [np.sin(true_theta), np.cos(true_theta)]])
    scan2 = (R_true @ scan1.T).T + true_t

    # Add noise
    scan2 += np.random.normal(0, 0.01, scan2.shape)

    def icp_2d(source, target, max_iter=50, tol=1e-6):
        """Simple 2D ICP implementation."""
        src = source.copy()
        total_R = np.eye(2)
        total_t = np.zeros(2)
        errors = []

        for iteration in range(max_iter):
            # Find nearest neighbors
            from scipy.spatial import cKDTree
            tree = cKDTree(target)
            dists, indices = tree.query(src)
            matched = target[indices]

            # Compute centroids
            src_centroid = np.mean(src, axis=0)
            tgt_centroid = np.mean(matched, axis=0)

            # Center the points
            src_centered = src - src_centroid
            tgt_centered = matched - tgt_centroid

            # SVD for optimal rotation
            H = src_centered.T @ tgt_centered
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T

            # Ensure proper rotation (det = +1)
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T

            t = tgt_centroid - R @ src_centroid

            # Apply transform
            src = (R @ src.T).T + t
            total_R = R @ total_R
            total_t = R @ total_t + t

            mean_error = np.mean(dists)
            errors.append(mean_error)

            if iteration > 0 and abs(errors[-1] - errors[-2]) < tol:
                break

        # Extract rotation angle
        angle = np.arctan2(total_R[1, 0], total_R[0, 0])
        return total_R, total_t, angle, errors

    try:
        R_est, t_est, angle_est, errors = icp_2d(scan1, scan2)

        print("ICP Scan Matching:")
        print(f"  True transform: theta={np.degrees(true_theta):.2f}°, "
              f"t=({true_t[0]:.3f}, {true_t[1]:.3f})")
        print(f"  ICP estimate:   theta={np.degrees(angle_est):.2f}°, "
              f"t=({t_est[0]:.3f}, {t_est[1]:.3f})")
        print(f"  Rotation error: {abs(np.degrees(true_theta - angle_est)):.4f}°")
        print(f"  Translation error: {np.linalg.norm(true_t - t_est):.6f} m")
        print(f"  Converged in {len(errors)} iterations")
        print(f"  Final mean point distance: {errors[-1]:.6f} m")
    except ImportError:
        print("ICP Scan Matching (scipy required for cKDTree):")
        print("  scipy not available — showing analytical result.")
        print(f"  True transform: theta={np.degrees(true_theta):.2f}°, "
              f"t=({true_t[0]:.3f}, {true_t[1]:.3f})")
        print(f"  ICP would recover this transform iteratively.")

    # Analyze error vs initial displacement
    print(f"\n  Error vs initial displacement:")
    print(f"  ICP converges well for small displacements but can fail for")
    print(f"  large initial offsets due to incorrect point correspondences.")
    print(f"  As a rule of thumb, initial displacement should be < 50%")
    print(f"  of the environment feature size for reliable convergence.")


def exercise_4():
    """
    Exercise 4: Complementary Filter for pitch angle.
    """
    dt = 0.01
    T = 10.0
    steps = int(T / dt)

    # Ground truth: sinusoidal rocking motion
    omega = 2 * np.pi * 0.5  # 0.5 Hz
    amplitude = np.radians(20)  # 20 degrees

    # Sensor noise parameters
    gyro_bias = np.radians(2)    # 2 deg/s bias
    gyro_noise_std = np.radians(0.5)
    accel_noise_std = np.radians(3)  # equivalent angle noise

    np.random.seed(42)

    def simulate_filter(alpha):
        pitch_est = 0.0
        pitch_gyro_only = 0.0
        errors = []

        for step in range(steps):
            t = step * dt
            # Ground truth
            pitch_true = amplitude * np.sin(omega * t)
            pitch_rate_true = amplitude * omega * np.cos(omega * t)

            # Gyroscope: rate + bias + noise
            gyro = pitch_rate_true + gyro_bias + np.random.normal(0, gyro_noise_std)

            # Accelerometer: pitch angle + noise (only valid in static/slow motion)
            accel_pitch = pitch_true + np.random.normal(0, accel_noise_std)

            # Complementary filter: combines gyro (good short-term) + accel (good long-term)
            pitch_est = alpha * (pitch_est + gyro * dt) + (1 - alpha) * accel_pitch

            # Gyro-only integration (for comparison)
            pitch_gyro_only += gyro * dt

            errors.append(np.degrees(abs(pitch_est - pitch_true)))

        rms_error = np.sqrt(np.mean(np.array(errors)**2))
        return rms_error

    print("Complementary Filter: pitch estimation")
    print(f"  Motion: 20° amplitude at 0.5 Hz")
    print(f"  Gyro bias: 2°/s, Gyro noise: 0.5°/s")
    print(f"  Accelerometer noise: 3°")
    print(f"\n  {'alpha':>8} | {'RMS error (°)':>14}")
    print("  " + "-" * 28)

    for alpha in [0.5, 0.8, 0.9, 0.95, 0.98, 0.99]:
        rms = simulate_filter(alpha)
        print(f"  {alpha:>8.2f} | {rms:>14.4f}")

    print(f"\n  alpha near 0.5: trusts accelerometer more (good for slow motion,")
    print(f"    noisy due to accel noise)")
    print(f"  alpha near 1.0: trusts gyroscope more (smooth but drifts due to bias)")
    print(f"  Optimal alpha balances these two effects, typically 0.9-0.98.")


def exercise_5():
    """
    Exercise 5: Sensor Selection Report for warehouse robot.
    """
    sensors = [
        {
            "sensor": "2D LiDAR (e.g., SICK TiM series)",
            "purpose": "Navigation and obstacle avoidance in aisles",
            "justification": "Accurate ranging (±30mm) in structured indoor environment. "
                             "Works in all lighting conditions. Cost-effective (~$1-3K).",
            "cost": "Medium",
        },
        {
            "sensor": "RGB-D Camera (e.g., Intel RealSense D455)",
            "purpose": "Shelf item identification and pick planning",
            "justification": "Provides both color (for barcode/label reading) and depth "
                             "(for grasp planning). Works at shelf distances (0.4-4m). "
                             "Low cost (~$250).",
            "cost": "Low",
        },
        {
            "sensor": "Wheel Encoders (incremental, quadrature)",
            "purpose": "Odometry for dead-reckoning navigation",
            "justification": "Essential for continuous pose estimation between LiDAR scans. "
                             "High update rate (>100 Hz). Very low cost.",
            "cost": "Low",
        },
        {
            "sensor": "IMU (6-axis: 3-axis gyro + 3-axis accel)",
            "purpose": "Angular velocity and tilt sensing, odometry improvement",
            "justification": "Fused with wheel encoders for better odometry (detects slip). "
                             "High bandwidth (200+ Hz) for dynamic motion. Low cost (~$20).",
            "cost": "Low",
        },
        {
            "sensor": "Force/Torque Sensor (6-axis, on gripper)",
            "purpose": "Grasp force control during picking",
            "justification": "Prevents damage to items by controlling grasp force. "
                             "Detects successful grasp vs. missed pick. Medium cost (~$2-5K).",
            "cost": "Medium-High",
        },
        {
            "sensor": "Safety LiDAR / Bumper switches",
            "purpose": "Collision detection for human safety",
            "justification": "Required for operation near humans. Safety-rated sensors "
                             "for emergency stop zones. Provides redundancy to main LiDAR.",
            "cost": "Medium",
        },
    ]

    print("Sensor Suite for Warehouse Robot")
    print("=" * 60)
    for s in sensors:
        print(f"\n  {s['sensor']}")
        print(f"    Purpose:       {s['purpose']}")
        print(f"    Justification: {s['justification']}")
        print(f"    Cost:          {s['cost']}")

    print(f"\n  Redundancy considerations:")
    print(f"    - Navigation: LiDAR + wheel encoders + IMU (triple redundancy)")
    print(f"    - Safety: dedicated safety LiDAR + bumper (dual redundancy)")
    print(f"    - Perception: RGB-D for normal operation, LiDAR for fallback")


if __name__ == "__main__":
    print("=" * 70)
    print("Lesson 10: Sensors and Perception — Exercise Solutions")
    print("=" * 70)

    print("\n--- Exercise 1: Encoder Resolution Analysis ---")
    exercise_1()

    print("\n--- Exercise 2: Camera Calibration ---")
    exercise_2()

    print("\n--- Exercise 3: LiDAR Scan Matching ---")
    exercise_3()

    print("\n--- Exercise 4: Complementary Filter ---")
    exercise_4()

    print("\n--- Exercise 5: Sensor Selection Report ---")
    exercise_5()

    print("\n" + "=" * 70)
    print("All exercises completed!")
    print("=" * 70)
