"""
Multi-Sensor Fusion Examples

Demonstrates key sensor fusion algorithms:
1. Weighted average fusion (multi-sensor temperature)
2. 1D Kalman filter (position tracking)
3. Complementary filter (IMU pitch estimation)
4. Kalman filter vs raw sensor comparison

All examples use simulated sensor data with configurable noise.
"""

import math
import random


# ========================================================================
# 1. Weighted Average Fusion
# ========================================================================

def weighted_average_fusion(measurements: list[dict]) -> dict:
    """Fuse multiple measurements using inverse-variance weighting.

    Args:
        measurements: List of {"value": float, "variance": float}

    Returns:
        {"value": fused_value, "variance": fused_variance}
    """
    total_precision = sum(1.0 / m["variance"] for m in measurements)
    fused_value = sum(m["value"] / m["variance"] for m in measurements) / total_precision
    fused_variance = 1.0 / total_precision
    return {"value": fused_value, "variance": fused_variance}


def demo_weighted_fusion():
    """Demo: fusing 3 temperature sensors."""
    print("=" * 60)
    print("1. Weighted Average Fusion (Temperature)")
    print("=" * 60)

    true_temp = 22.5
    sensors = [
        {"name": "Sensor A", "value": 22.1, "variance": 0.5},
        {"name": "Sensor B", "value": 22.8, "variance": 0.2},
        {"name": "Sensor C", "value": 22.4, "variance": 0.3},
    ]

    print(f"\n  True temperature: {true_temp}°C\n")
    for s in sensors:
        weight = (1 / s["variance"]) / sum(1 / x["variance"] for x in sensors)
        error = abs(s["value"] - true_temp)
        print(f"  {s['name']}: {s['value']}°C  "
              f"(σ²={s['variance']:.1f}, weight={weight:.2f}, "
              f"error={error:.1f}°C)")

    result = weighted_average_fusion(sensors)
    error = abs(result["value"] - true_temp)
    print(f"\n  Fused:    {result['value']:.3f}°C  "
          f"(σ²={result['variance']:.4f}, error={error:.3f}°C)")
    print(f"  Improvement: variance reduced from "
          f"{min(s['variance'] for s in sensors):.1f} to {result['variance']:.4f} "
          f"({min(s['variance'] for s in sensors)/result['variance']:.1f}x better)")


# ========================================================================
# 2. 1D Kalman Filter
# ========================================================================

class KalmanFilter1D:
    """1D Kalman filter for scalar state estimation."""

    def __init__(self, A=1.0, B=0.0, H=1.0, Q=0.01, R=0.5, x0=0.0, P0=1.0):
        self.A = A    # State transition
        self.B = B    # Control input
        self.H = H    # Measurement matrix
        self.Q = Q    # Process noise variance
        self.R = R    # Measurement noise variance
        self.x = x0   # State estimate
        self.P = P0   # Error covariance

    def predict(self, u=0.0):
        """Predict next state."""
        self.x = self.A * self.x + self.B * u
        self.P = self.A * self.P * self.A + self.Q

    def update(self, z):
        """Update with measurement."""
        K = self.P * self.H / (self.H * self.P * self.H + self.R)
        innovation = z - self.H * self.x
        self.x = self.x + K * innovation
        self.P = (1 - K * self.H) * self.P
        return K


def demo_kalman_1d():
    """Demo: tracking position with noisy measurements."""
    print("\n" + "=" * 60)
    print("2. 1D Kalman Filter (Position Tracking)")
    print("=" * 60)

    random.seed(42)

    # True motion: constant velocity v=1 m/s
    dt = 1.0
    velocity = 1.0
    process_noise_std = 0.3
    measurement_noise_std = 2.0

    kf = KalmanFilter1D(
        A=1.0,    # x_k = x_{k-1} (position persists)
        B=dt,     # + v * dt
        H=1.0,    # We measure position directly
        Q=process_noise_std**2,
        R=measurement_noise_std**2,
        x0=0.0,
        P0=1.0,
    )

    n_steps = 15
    true_positions = []
    measurements = []
    estimates = []
    kalman_gains = []

    true_pos = 0.0

    print(f"\n  {'Step':<5} {'True':<8} {'Measured':<10} {'Estimate':<10} "
          f"{'K gain':<8} {'Error(raw)':<11} {'Error(KF)':<10}")
    print("  " + "-" * 62)

    for k in range(n_steps):
        # True position with small process noise
        true_pos += velocity * dt + random.gauss(0, process_noise_std)
        true_positions.append(true_pos)

        # Noisy measurement
        z = true_pos + random.gauss(0, measurement_noise_std)
        measurements.append(z)

        # Kalman filter
        kf.predict(u=velocity)
        K = kf.update(z)
        estimates.append(kf.x)
        kalman_gains.append(K)

        err_raw = abs(z - true_pos)
        err_kf = abs(kf.x - true_pos)

        print(f"  {k:<5} {true_pos:<8.2f} {z:<10.2f} {kf.x:<10.2f} "
              f"{K:<8.3f} {err_raw:<11.2f} {err_kf:<10.2f}")

    # Summary statistics
    raw_errors = [abs(m - t) for m, t in zip(measurements, true_positions)]
    kf_errors = [abs(e - t) for e, t in zip(estimates, true_positions)]
    print(f"\n  Average error (raw measurements): {sum(raw_errors)/len(raw_errors):.2f} m")
    print(f"  Average error (Kalman filter):    {sum(kf_errors)/len(kf_errors):.2f} m")
    print(f"  Error reduction: {(1 - sum(kf_errors)/sum(raw_errors))*100:.0f}%")
    print(f"  Final Kalman gain: {kalman_gains[-1]:.3f} "
          f"(converged: {'yes' if abs(kalman_gains[-1] - kalman_gains[-2]) < 0.01 else 'no'})")


# ========================================================================
# 3. Complementary Filter
# ========================================================================

class ComplementaryFilter:
    """Complementary filter for IMU pitch estimation."""

    def __init__(self, alpha=0.96):
        self.alpha = alpha
        self.angle = 0.0

    def update(self, gyro_rate, accel_angle, dt):
        """
        Fuse gyroscope rate with accelerometer angle.

        gyro_rate: Angular rate from gyroscope (rad/s)
        accel_angle: Angle computed from accelerometer (rad)
        dt: Time step (s)
        """
        # High-pass: trust gyro for fast changes (integration)
        # Low-pass: trust accel for static orientation
        self.angle = self.alpha * (self.angle + gyro_rate * dt) + \
                     (1 - self.alpha) * accel_angle
        return self.angle


def demo_complementary_filter():
    """Demo: fusing gyro and accel for pitch estimation."""
    print("\n" + "=" * 60)
    print("3. Complementary Filter (IMU Pitch)")
    print("=" * 60)

    random.seed(123)

    # Simulate an oscillating pitch motion
    dt = 0.01  # 100 Hz
    duration = 10.0  # 10 seconds
    n_steps = int(duration / dt)

    cf = ComplementaryFilter(alpha=0.98)
    gyro_only_angle = 0.0
    gyro_bias = 0.005  # 0.29°/s drift (typical MEMS gyro)

    true_angles = []
    accel_angles = []
    gyro_angles = []
    fused_angles = []

    for i in range(n_steps):
        t = i * dt

        # True pitch: 30° oscillation at 0.5 Hz
        true_angle = math.radians(30) * math.sin(2 * math.pi * 0.5 * t)
        true_rate = math.radians(30) * 2 * math.pi * 0.5 * \
                    math.cos(2 * math.pi * 0.5 * t)

        # Gyroscope: accurate rate + small noise + constant bias
        gyro_rate = true_rate + random.gauss(0, math.radians(0.5)) + gyro_bias
        gyro_only_angle += gyro_rate * dt

        # Accelerometer: noisy angle measurement
        accel_angle = true_angle + random.gauss(0, math.radians(3.0))

        # Complementary filter
        fused = cf.update(gyro_rate, accel_angle, dt)

        true_angles.append(math.degrees(true_angle))
        accel_angles.append(math.degrees(accel_angle))
        gyro_angles.append(math.degrees(gyro_only_angle))
        fused_angles.append(math.degrees(fused))

    # Print samples every 1 second
    print(f"\n  {'Time(s)':<8} {'True(°)':<10} {'Accel(°)':<10} "
          f"{'Gyro(°)':<10} {'Fused(°)':<10}")
    print("  " + "-" * 48)
    for i in range(0, n_steps, int(1.0 / dt)):
        print(f"  {i*dt:<8.1f} {true_angles[i]:<10.1f} {accel_angles[i]:<10.1f} "
              f"{gyro_angles[i]:<10.1f} {fused_angles[i]:<10.1f}")

    # Error analysis at end
    def rmse(estimated, true_vals):
        return math.sqrt(sum((e - t)**2 for e, t in zip(estimated, true_vals)) / len(true_vals))

    accel_rmse = rmse(accel_angles, true_angles)
    gyro_rmse = rmse(gyro_angles, true_angles)
    fused_rmse = rmse(fused_angles, true_angles)

    print(f"\n  RMSE Analysis:")
    print(f"    Accelerometer only: {accel_rmse:.2f}° (noisy but no drift)")
    print(f"    Gyroscope only:     {gyro_rmse:.2f}° (smooth but drifts)")
    print(f"    Complementary:      {fused_rmse:.2f}° (best of both)")
    print(f"    Improvement over accel: {(1-fused_rmse/accel_rmse)*100:.0f}%")
    print(f"    Improvement over gyro:  {(1-fused_rmse/gyro_rmse)*100:.0f}%")

    # Show gyro drift
    print(f"\n  Gyroscope drift after {duration}s: "
          f"{gyro_angles[-1] - true_angles[-1]:.1f}° "
          f"(bias = {math.degrees(gyro_bias):.2f}°/s)")


# ========================================================================
# 4. Multi-Sensor Kalman Filter
# ========================================================================

def demo_multisensor_kalman():
    """Demo: fusing GPS (slow, accurate) + IMU (fast, drifty) for position."""
    print("\n" + "=" * 60)
    print("4. Multi-Rate Sensor Fusion (GPS + IMU)")
    print("=" * 60)

    random.seed(99)

    # Scenario: vehicle moving at ~10 m/s
    # IMU provides acceleration at 100 Hz (noisy)
    # GPS provides position at 1 Hz (accurate but slow)
    dt_imu = 0.01    # 100 Hz
    dt_gps = 1.0     # 1 Hz
    duration = 5.0

    kf = KalmanFilter1D(
        A=1.0, B=dt_imu,
        H=1.0,
        Q=0.5,    # IMU integration noise
        R=3.0,    # GPS measurement noise
        x0=0.0, P0=10.0,
    )

    true_pos = 0.0
    true_vel = 10.0  # m/s
    imu_only_pos = 0.0
    imu_bias = 0.05  # Small acceleration bias

    n_steps = int(duration / dt_imu)

    print(f"\n  IMU: 100 Hz, GPS: 1 Hz, Duration: {duration}s\n")
    print(f"  {'Time(s)':<8} {'True(m)':<10} {'IMU-only(m)':<12} "
          f"{'KF(m)':<10} {'GPS?':<6} {'KF err(m)':<10}")
    print("  " + "-" * 56)

    for i in range(n_steps):
        t = i * dt_imu

        # True motion (constant velocity + slight curve)
        true_vel_actual = true_vel + 0.5 * math.sin(t)
        true_pos += true_vel_actual * dt_imu

        # IMU: noisy acceleration + bias
        imu_accel = true_vel_actual + random.gauss(0, 0.5) + imu_bias
        imu_only_pos += imu_accel * dt_imu

        # Kalman predict (every IMU step)
        kf.predict(u=imu_accel)

        # GPS update (every 1 second)
        is_gps = (i > 0 and i % int(dt_gps / dt_imu) == 0)
        if is_gps:
            gps_reading = true_pos + random.gauss(0, 3.0)  # GPS noise ±3m
            kf.update(gps_reading)

        # Print every 0.5s
        if i % int(0.5 / dt_imu) == 0:
            kf_err = abs(kf.x - true_pos)
            print(f"  {t:<8.1f} {true_pos:<10.1f} {imu_only_pos:<12.1f} "
                  f"{kf.x:<10.1f} {'YES' if is_gps else '':<6} {kf_err:<10.2f}")

    imu_err = abs(imu_only_pos - true_pos)
    kf_err = abs(kf.x - true_pos)
    print(f"\n  Final errors:")
    print(f"    IMU integration only: {imu_err:.1f} m (accumulated drift)")
    print(f"    Kalman filter (IMU+GPS): {kf_err:.1f} m")
    print(f"    GPS corrections prevent IMU drift from accumulating")


# ========================================================================
# Main
# ========================================================================

if __name__ == "__main__":
    demo_weighted_fusion()
    demo_kalman_1d()
    demo_complementary_filter()
    demo_multisensor_kalman()
