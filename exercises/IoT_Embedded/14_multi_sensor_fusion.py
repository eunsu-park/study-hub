"""
Exercises for Lesson 14: Multi-Sensor Fusion
Topic: IoT_Embedded

Solutions to practice problems covering weighted average fusion,
1D Kalman filter tracing, complementary filter tuning,
sensor fusion architecture design, and EKF bias estimation.
"""

import math
import random


def exercise_1():
    """
    Weighted average fusion of 3 temperature sensors.
    """
    print("=== Exercise 1: Weighted Average Fusion ===\n")

    sensors = [
        {"name": "A", "value": 22.1, "variance": 0.5},
        {"name": "B", "value": 22.8, "variance": 0.2},
        {"name": "C", "value": 22.4, "variance": 0.3},
    ]

    # Inverse-variance weighting
    total_precision = sum(1.0 / s["variance"] for s in sensors)
    weights = [1.0 / (s["variance"] * total_precision) for s in sensors]

    print("  Sensor data:")
    for s, w in zip(sensors, weights):
        print(f"    Sensor {s['name']}: {s['value']}°C, "
              f"σ² = {s['variance']}, weight = {w:.3f}")

    fused_value = sum(s["value"] * w for s, w in zip(sensors, weights))
    fused_variance = 1.0 / total_precision

    print(f"\n  Calculation:")
    print(f"    Total precision = Σ(1/σ²) = {' + '.join(f'1/{s['variance']}' for s in sensors)}")
    print(f"                    = {total_precision:.3f}")
    print(f"\n    Fused value = Σ(wᵢ × zᵢ)")
    for s, w in zip(sensors, weights):
        print(f"                + {w:.3f} × {s['value']} = {w * s['value']:.4f}")
    print(f"                = {fused_value:.4f}°C")
    print(f"\n    Fused variance = 1 / {total_precision:.3f} = {fused_variance:.4f}")
    print(f"    Fused std dev  = {math.sqrt(fused_variance):.4f}°C")
    print(f"\n  Result: {fused_value:.3f}°C ± {math.sqrt(fused_variance):.3f}°C")
    print(f"  Best single sensor variance: {min(s['variance'] for s in sensors)}")
    print(f"  Fused variance: {fused_variance:.4f}")
    print(f"  Improvement: {min(s['variance'] for s in sensors)/fused_variance:.1f}x better")
    print()


def exercise_2():
    """
    1D Kalman filter trace for 5 steps.
    """
    print("=== Exercise 2: 1D Kalman Filter Trace ===\n")

    # Parameters
    v = 1.0     # m/s velocity
    dt = 1.0    # 1 second
    Q = 0.1     # process noise
    R = 1.0     # measurement noise

    # Initial conditions
    x = 0.0     # state estimate
    P = 1.0     # error covariance
    A = 1.0     # state transition (position persists)
    B = dt      # control input (velocity × dt)
    H = 1.0     # measurement matrix

    measurements = [1.2, 2.5, 2.8, 4.1, 5.3]

    print(f"  Model: x_k = x_{{k-1}} + v·Δt,  v = {v} m/s, Δt = {dt}s")
    print(f"  Q = {Q}, R = {R}, x₀ = {x}, P₀ = {P}")
    print()

    print(f"  {'Step':<5} {'Predict x⁻':<12} {'Predict P⁻':<12} "
          f"{'z':<8} {'K':<8} {'x̂':<10} {'P':<10}")
    print("  " + "-" * 60)

    for k, z in enumerate(measurements, 1):
        # Predict
        x_pred = A * x + B * v
        P_pred = A * P * A + Q

        # Update
        K = P_pred * H / (H * P_pred * H + R)
        x = x_pred + K * (z - H * x_pred)
        P = (1 - K * H) * P_pred

        print(f"  {k:<5} {x_pred:<12.3f} {P_pred:<12.3f} "
              f"{z:<8.1f} {K:<8.3f} {x:<10.3f} {P:<10.3f}")

    print(f"\n  Final state estimate: x̂ = {x:.3f} m")
    print(f"  Final uncertainty:   P = {P:.3f}")
    print(f"  Expected true position: {v * dt * len(measurements):.1f} m")
    print(f"  Kalman gain converged to K ≈ {K:.3f}")
    print(f"  This means the filter trusts the measurement about "
          f"{K*100:.0f}% vs the prediction {(1-K)*100:.0f}%")
    print()


def exercise_3():
    """
    Complementary filter with synthetic IMU data.
    """
    print("=== Exercise 3: Complementary Filter Tuning ===\n")

    random.seed(42)

    alpha = 0.98
    dt = 0.01  # 100 Hz
    duration = 5.0
    n_steps = int(duration / dt)

    cf_angle = 0.0
    gyro_only = 0.0
    gyro_noise_std = math.radians(0.5)
    accel_noise_std = math.radians(3.0)
    gyro_bias = math.radians(0.3)  # Small bias

    true_list = []
    accel_list = []
    gyro_list = []
    fused_list = []

    for i in range(n_steps):
        t = i * dt

        # True angle: 30° oscillation at 0.5 Hz
        true_angle = math.radians(30) * math.sin(2 * math.pi * 0.5 * t)
        true_rate = math.radians(30) * 2 * math.pi * 0.5 * math.cos(2 * math.pi * 0.5 * t)

        # Sensors
        gyro_rate = true_rate + random.gauss(0, gyro_noise_std) + gyro_bias
        accel_angle = true_angle + random.gauss(0, accel_noise_std)

        # Gyro integration (no correction)
        gyro_only += gyro_rate * dt

        # Complementary filter
        cf_angle = alpha * (cf_angle + gyro_rate * dt) + (1 - alpha) * accel_angle

        true_list.append(math.degrees(true_angle))
        accel_list.append(math.degrees(accel_angle))
        gyro_list.append(math.degrees(gyro_only))
        fused_list.append(math.degrees(cf_angle))

    # Print samples
    print(f"  α = {alpha}, dt = {dt}s, duration = {duration}s")
    print(f"  Gyro noise: {math.degrees(gyro_noise_std):.1f}°/s, "
          f"Accel noise: {math.degrees(accel_noise_std):.1f}°")
    print(f"  Gyro bias: {math.degrees(gyro_bias):.1f}°/s")
    print()

    print(f"  {'Time':<6} {'True':<8} {'Accel':<8} {'Gyro':<8} {'Fused':<8}")
    print("  " + "-" * 38)
    for i in range(0, n_steps, int(0.5 / dt)):
        print(f"  {i*dt:<6.1f} {true_list[i]:<8.1f} {accel_list[i]:<8.1f} "
              f"{gyro_list[i]:<8.1f} {fused_list[i]:<8.1f}")

    # RMSE
    def rmse(est, true_vals):
        return math.sqrt(sum((e - t)**2 for e, t in zip(est, true_vals)) / len(true_vals))

    r_accel = rmse(accel_list, true_list)
    r_gyro = rmse(gyro_list, true_list)
    r_fused = rmse(fused_list, true_list)

    print(f"\n  RMSE:")
    print(f"    Accelerometer: {r_accel:.2f}°")
    print(f"    Gyroscope:     {r_gyro:.2f}° (drifts due to bias)")
    print(f"    Fused (α={alpha}): {r_fused:.2f}°")
    print(f"\n  Complementary filter eliminates gyro drift while")
    print(f"  smoothing accelerometer noise. Best of both worlds.")
    print()


def exercise_4():
    """
    Sensor fusion architecture design for a delivery robot.
    """
    print("=== Exercise 4: Delivery Robot Sensor Fusion Architecture ===\n")

    print("  1) Sensor List:")
    sensors = [
        ("IMU (6-axis)",     "200 Hz", "Orientation, acceleration"),
        ("Wheel encoders",   "100 Hz", "Velocity, distance"),
        ("LiDAR",            "20 Hz",  "2D/3D environment map"),
        ("Camera (stereo)",  "30 Hz",  "Visual features, obstacles"),
        ("GPS",              "10 Hz",  "Global position (outdoor)"),
        ("Barometer",        "50 Hz",  "Floor detection (elevator)"),
        ("WiFi/BLE beacons", "1 Hz",   "Indoor positioning"),
        ("Ultrasonic",       "40 Hz",  "Close-range obstacles"),
    ]
    print(f"    {'Sensor':<22} {'Rate':<10} {'Measures'}")
    print("    " + "-" * 55)
    for name, rate, measures in sensors:
        print(f"    {name:<22} {rate:<10} {measures}")

    print(f"\n  2) Architecture: Hierarchical (3 levels)")
    print(f"    Centralized fusion would require too much bandwidth")
    print(f"    from LiDAR and cameras (100+ Mbps). Hierarchical")
    print(f"    reduces data flow and enables modular failure handling.")

    print(f"\n  3) Data flow diagram:")
    print()
    print("    Level 1: Sensor-level preprocessing")
    print("    ┌─────────────┐  ┌───────────────┐  ┌────────────┐")
    print("    │ IMU + Wheel  │  │ LiDAR + Camera│  │GPS + WiFi  │")
    print("    │ EKF→Odometry │  │ SLAM          │  │ WA→Position│")
    print("    └──────┬───────┘  └──────┬────────┘  └─────┬──────┘")
    print("           │                 │                  │")
    print("    Level 2: Group fusion")
    print("    ┌──────┴─────────────────┴──────────────────┴──────┐")
    print("    │        EKF: Pose estimation (x, y, θ, v)         │")
    print("    │        Fuses odometry + SLAM + global position    │")
    print("    └──────────────────────┬────────────────────────────┘")
    print("                          │")
    print("    Level 3: Global state")
    print("    ┌──────────────────────┴────────────────────────────┐")
    print("    │    Navigation + Obstacle avoidance + Path planning │")
    print("    │    Full robot state at 50 Hz                       │")
    print("    └───────────────────────────────────────────────────┘")

    print(f"\n  4) Filter selection at each level:")
    filters = [
        ("IMU+Encoder → Odometry",   "EKF",     "Nonlinear motion model"),
        ("LiDAR+Camera → SLAM",      "Particle Filter", "Multi-modal map matching"),
        ("GPS+WiFi → Position",      "Weighted Avg",    "Independent measurements"),
        ("Odometry+SLAM+GPS → Pose", "EKF",     "Nonlinear state transitions"),
    ]
    print(f"    {'Fusion Point':<30} {'Filter':<18} {'Reason'}")
    print("    " + "-" * 65)
    for point, filt, reason in filters:
        print(f"    {point:<30} {filt:<18} {reason}")
    print()


def exercise_5():
    """
    EKF bias estimation: estimate angle + gyroscope bias simultaneously.
    """
    print("=== Exercise 5: EKF Gyroscope Bias Estimation ===\n")

    random.seed(7)

    # State: [angle, gyro_bias]
    # True values
    true_bias = math.radians(0.5)  # 0.5°/s

    dt = 0.01
    duration = 30.0
    n_steps = int(duration / dt)

    # State estimate: x = [angle, bias]
    x_angle = 0.0
    x_bias = 0.0

    # Covariance
    P = [[1.0, 0.0],
         [0.0, 0.1]]

    Q = [[0.001, 0.0],
         [0.0, 0.00001]]  # Bias changes very slowly

    R_accel = math.radians(3.0) ** 2  # Accelerometer noise variance

    angle_errors = []
    bias_estimates = []

    for i in range(n_steps):
        t = i * dt

        # True angle: slow oscillation
        true_angle = math.radians(20) * math.sin(2 * math.pi * 0.1 * t)
        true_rate = math.radians(20) * 2 * math.pi * 0.1 * math.cos(2 * math.pi * 0.1 * t)

        # Gyroscope measurement (with true bias + noise)
        gyro_meas = true_rate + true_bias + random.gauss(0, math.radians(0.3))

        # Accelerometer angle measurement (noisy, no bias)
        accel_angle = true_angle + random.gauss(0, math.radians(3.0))

        # --- EKF Predict ---
        # State transition: angle += (gyro - bias) * dt, bias stays
        x_angle_pred = x_angle + (gyro_meas - x_bias) * dt
        x_bias_pred = x_bias

        # Jacobian F = [[1, -dt], [0, 1]]
        F = [[1, -dt], [0, 1]]

        # P_pred = F @ P @ F^T + Q
        P_pred = [[0, 0], [0, 0]]
        for r in range(2):
            for c in range(2):
                for k in range(2):
                    for j in range(2):
                        P_pred[r][c] += F[r][k] * P[k][j] * F[c][j]
                P_pred[r][c] += Q[r][c]

        # --- EKF Update (accelerometer provides angle) ---
        # H = [1, 0] (we measure angle directly)
        # Innovation
        y = accel_angle - x_angle_pred

        # S = H @ P @ H^T + R = P[0][0] + R
        S = P_pred[0][0] + R_accel

        # K = P @ H^T / S
        K = [P_pred[0][0] / S, P_pred[1][0] / S]

        # Update state
        x_angle = x_angle_pred + K[0] * y
        x_bias = x_bias_pred + K[1] * y

        # Update covariance
        P_new = [[0, 0], [0, 0]]
        for r in range(2):
            for c in range(2):
                P_new[r][c] = P_pred[r][c] - K[r] * P_pred[0][c]
        P = P_new

        angle_errors.append(abs(math.degrees(x_angle - true_angle)))
        bias_estimates.append(math.degrees(x_bias))

    # Print convergence
    print(f"  True gyro bias: {math.degrees(true_bias):.2f}°/s")
    print(f"  Initial bias estimate: 0.00°/s")
    print()
    print(f"  {'Time(s)':<8} {'Bias Est(°/s)':<15} {'Bias Err(°/s)':<15} {'Angle Err(°)'}")
    print("  " + "-" * 48)

    checkpoints = [0.5, 1, 2, 5, 10, 15, 20, 25, 30]
    for t_check in checkpoints:
        idx = min(int(t_check / dt), n_steps - 1)
        b_est = bias_estimates[idx]
        b_err = abs(b_est - math.degrees(true_bias))
        a_err = angle_errors[idx]
        print(f"  {t_check:<8.1f} {b_est:<15.4f} {b_err:<15.4f} {a_err:.4f}")

    final_bias = bias_estimates[-1]
    final_err = abs(final_bias - math.degrees(true_bias))
    print(f"\n  Final bias estimate: {final_bias:.4f}°/s")
    print(f"  True bias:           {math.degrees(true_bias):.4f}°/s")
    print(f"  Estimation error:    {final_err:.4f}°/s")
    print(f"  Converged: {'YES' if final_err < 0.05 else 'NO'} "
          f"(threshold: 0.05°/s)")
    print()


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
