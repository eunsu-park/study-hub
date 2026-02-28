# 14. Multi-Sensor Fusion

**Previous**: [Zigbee and Z-Wave](./13_Zigbee_ZWave.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why combining multiple sensors produces more accurate and reliable results than any single sensor
2. Implement a Kalman filter for fusing noisy sensor measurements with a dynamic model
3. Apply a complementary filter to combine accelerometer and gyroscope data for orientation estimation
4. Design a sensor fusion architecture for an IMU (Inertial Measurement Unit) using an Extended Kalman Filter
5. Compare centralized, decentralized, and distributed fusion architectures for multi-sensor IoT systems

---

A single thermometer drifts; a single accelerometer accumulates error; a single GPS receiver loses signal indoors. But combine a thermometer with humidity and pressure sensors, or fuse an accelerometer with a gyroscope and magnetometer, and the result is far more accurate than any individual reading. This is sensor fusion -- the art of making many imperfect measurements produce one reliable answer.

---

## Table of Contents

1. [Why Sensor Fusion?](#1-why-sensor-fusion)
2. [Fusion Architectures](#2-fusion-architectures)
3. [Weighted Average Fusion](#3-weighted-average-fusion)
4. [Kalman Filter](#4-kalman-filter)
5. [Complementary Filter](#5-complementary-filter)
6. [Extended Kalman Filter for IMU](#6-extended-kalman-filter-for-imu)
7. [Practical Applications](#7-practical-applications)
8. [Practice Problems](#8-practice-problems)

---

## 1. Why Sensor Fusion?

### 1.1 Limitations of Individual Sensors

| Sensor | Strength | Weakness |
|--------|----------|----------|
| Accelerometer | Accurate static orientation (gravity direction) | Noisy, vibration-sensitive, no yaw |
| Gyroscope | Smooth rotation rate | Drift accumulates over time |
| Magnetometer | Absolute heading (compass) | Susceptible to magnetic interference |
| GPS | Absolute global position | No indoor coverage, low update rate |
| Barometer | Altitude changes | Slow, weather-dependent drift |
| Camera | Rich spatial information | Computationally expensive, lighting-dependent |

### 1.2 Fusion Benefits

```
Individual sensors:          Fused result:

Accelerometer  ──┐
  (noisy but     │
   no drift)     ├──► Sensor Fusion ──► Accurate, smooth,
                 │    Algorithm         drift-free estimate
Gyroscope     ───┤
  (smooth but    │
   drifts)       │
                 │
Magnetometer  ───┘
  (absolute but
   interfered)
```

| Benefit | Description |
|---------|-------------|
| **Accuracy** | Combining sensors reduces overall error |
| **Robustness** | If one sensor fails, others compensate |
| **Completeness** | Different sensors measure different things (position + orientation) |
| **Temporal coverage** | Fast sensors fill gaps between slow sensor updates |

---

## 2. Fusion Architectures

### 2.1 Centralized Fusion

All raw sensor data sent to a single processing node:

```
Sensor A ─── raw data ──┐
Sensor B ─── raw data ──┼──► Central Processor ──► Fused Output
Sensor C ─── raw data ──┘
```

**Pros**: Optimal accuracy (all data available). **Cons**: High bandwidth, single point of failure.

### 2.2 Decentralized Fusion

Each sensor preprocesses locally, only sends estimates:

```
Sensor A ──► Local Filter A ──┐
Sensor B ──► Local Filter B ──┼──► Fusion Node ──► Fused Output
Sensor C ──► Local Filter C ──┘
```

**Pros**: Lower bandwidth, fault-tolerant. **Cons**: May lose correlations between sensors.

### 2.3 Hierarchical Fusion

Multi-level processing:

```
Level 1: Raw sensors → per-sensor preprocessing
Level 2: Group fusion (e.g., all IMU sensors together)
Level 3: Global fusion (IMU + GPS + vision)
```

This is common in autonomous vehicles: IMU fusion runs at 200 Hz locally, then fuses with GPS (10 Hz) and LiDAR (20 Hz) at higher levels.

---

## 3. Weighted Average Fusion

The simplest fusion method: weight each sensor inversely by its noise variance.

### 3.1 Two-Sensor Case

Given sensors with measurements z₁ and z₂ and variances σ₁² and σ₂²:

$$\hat{x} = \frac{\sigma_2^2}{\sigma_1^2 + \sigma_2^2} z_1 + \frac{\sigma_1^2}{\sigma_1^2 + \sigma_2^2} z_2$$

The fused variance is always smaller than either individual variance:

$$\sigma_{fused}^2 = \frac{\sigma_1^2 \cdot \sigma_2^2}{\sigma_1^2 + \sigma_2^2} < \min(\sigma_1^2, \sigma_2^2)$$

### 3.2 Example: Temperature Fusion

```python
import numpy as np

# Two temperature sensors measuring the same room
sensor_a = {"mean": 22.3, "variance": 0.5}   # Less accurate
sensor_b = {"mean": 22.8, "variance": 0.1}   # More accurate

# Optimal weights (inverse variance weighting)
w_a = sensor_b["variance"] / (sensor_a["variance"] + sensor_b["variance"])
w_b = sensor_a["variance"] / (sensor_a["variance"] + sensor_b["variance"])

fused_temp = w_a * sensor_a["mean"] + w_b * sensor_b["mean"]
fused_var  = (sensor_a["variance"] * sensor_b["variance"]) / \
             (sensor_a["variance"] + sensor_b["variance"])

print(f"Sensor A: {sensor_a['mean']}°C ± {sensor_a['variance']:.1f}")
print(f"Sensor B: {sensor_b['mean']}°C ± {sensor_b['variance']:.1f}")
print(f"Weights:  A={w_a:.2f}, B={w_b:.2f}")
print(f"Fused:    {fused_temp:.2f}°C ± {fused_var:.3f}")
# More accurate sensor (B) gets higher weight
```

---

## 4. Kalman Filter

The Kalman filter is the most important algorithm in sensor fusion. It optimally combines a prediction from a dynamic model with a noisy measurement.

### 4.1 The Core Idea

```
Time k-1                        Time k

 x̂[k-1] ──► Predict ──► x̂⁻[k]  ──┐
             (model)               ├──► Update ──► x̂[k]
                                   │   (correct)
             Sensor ──► z[k]  ────┘
             reading
```

1. **Predict**: Use the physics/motion model to estimate the next state
2. **Update**: Correct the prediction using the actual sensor reading
3. **Repeat**: The result becomes the input for the next prediction

### 4.2 Equations

**State model**: $x_k = A \cdot x_{k-1} + B \cdot u_k + w_k$ (process noise w ~ N(0, Q))

**Measurement model**: $z_k = H \cdot x_k + v_k$ (measurement noise v ~ N(0, R))

**Predict step**:

$$\hat{x}_k^- = A \cdot \hat{x}_{k-1} + B \cdot u_k$$
$$P_k^- = A \cdot P_{k-1} \cdot A^T + Q$$

**Update step**:

$$K_k = P_k^- \cdot H^T \cdot (H \cdot P_k^- \cdot H^T + R)^{-1}$$
$$\hat{x}_k = \hat{x}_k^- + K_k \cdot (z_k - H \cdot \hat{x}_k^-)$$
$$P_k = (I - K_k \cdot H) \cdot P_k^-$$

Where:
- $K_k$ is the **Kalman gain** (how much to trust the measurement vs the prediction)
- $P_k$ is the **error covariance** (uncertainty in the state estimate)
- When R is large (noisy sensor): K is small → trust the prediction more
- When Q is large (uncertain model): K is large → trust the measurement more

### 4.3 1D Example: Tracking Temperature

```python
import numpy as np

class KalmanFilter1D:
    """1D Kalman filter for scalar measurements."""

    def __init__(self, A=1.0, H=1.0, Q=0.01, R=0.5, x0=20.0, P0=1.0):
        self.A = A    # State transition
        self.H = H    # Measurement matrix
        self.Q = Q    # Process noise variance
        self.R = R    # Measurement noise variance
        self.x = x0   # State estimate
        self.P = P0   # Error covariance

    def predict(self):
        self.x = self.A * self.x
        self.P = self.A * self.P * self.A + self.Q

    def update(self, z):
        K = self.P * self.H / (self.H * self.P * self.H + self.R)
        self.x = self.x + K * (z - self.H * self.x)
        self.P = (1 - K * self.H) * self.P
        return self.x, K
```

### 4.4 Multi-Dimensional Kalman Filter

For tracking position and velocity:

```python
import numpy as np

class KalmanFilterND:
    """N-dimensional Kalman filter using numpy matrices."""

    def __init__(self, A, H, Q, R, x0, P0):
        self.A = np.array(A)  # State transition matrix
        self.H = np.array(H)  # Measurement matrix
        self.Q = np.array(Q)  # Process noise covariance
        self.R = np.array(R)  # Measurement noise covariance
        self.x = np.array(x0) # Initial state
        self.P = np.array(P0) # Initial covariance

    def predict(self, u=None, B=None):
        self.x = self.A @ self.x
        if u is not None and B is not None:
            self.x += np.array(B) @ np.array(u)
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, z):
        z = np.array(z)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        I = np.eye(len(self.x))
        self.P = (I - K @ self.H) @ self.P
        return self.x.copy(), K
```

---

## 5. Complementary Filter

A complementary filter is a simpler alternative to the Kalman filter, commonly used for IMU orientation estimation.

### 5.1 The Idea

- **Accelerometer**: Accurate at low frequencies (static orientation), noisy at high frequencies (vibrations)
- **Gyroscope**: Accurate at high frequencies (fast rotations), drifts at low frequencies

Combine them with complementary frequency weighting:

$$\theta_{fused} = \alpha \cdot (\theta_{prev} + \omega_{gyro} \cdot \Delta t) + (1 - \alpha) \cdot \theta_{accel}$$

Where:
- $\alpha$ = 0.95-0.98 (trust gyroscope for short-term, accelerometer for long-term)
- $\omega_{gyro}$ = gyroscope angular rate
- $\theta_{accel}$ = angle computed from accelerometer (using atan2)
- $\Delta t$ = time step

### 5.2 Why It Works

```
Frequency response:

                  Accelerometer    Gyroscope       Fused
                  (low-pass)       (high-pass)     (full band)

High freq:        ████             ████████████    ████████████
                  (noisy)          (accurate)      (from gyro)

Low freq:         ████████████     ████            ████████████
                  (accurate)       (drifts)        (from accel)
```

The complementary filter is a first-order approximation of the Kalman filter but requires no matrix operations, making it ideal for microcontrollers.

### 5.3 Implementation

```python
class ComplementaryFilter:
    """Simple complementary filter for pitch/roll from IMU."""

    def __init__(self, alpha=0.96):
        self.alpha = alpha
        self.pitch = 0.0
        self.roll = 0.0

    def update(self, accel, gyro, dt):
        """
        accel: (ax, ay, az) in m/s² (gravity direction)
        gyro: (gx, gy, gz) in rad/s
        dt: time step in seconds
        """
        import math
        ax, ay, az = accel
        gx, gy, gz = gyro

        # Angle from accelerometer (reliable for static orientation)
        accel_pitch = math.atan2(ay, math.sqrt(ax**2 + az**2))
        accel_roll  = math.atan2(-ax, az)

        # Integrate gyroscope (good for fast changes)
        self.pitch = self.alpha * (self.pitch + gy * dt) + \
                     (1 - self.alpha) * accel_pitch
        self.roll  = self.alpha * (self.roll + gx * dt) + \
                     (1 - self.alpha) * accel_roll

        return self.pitch, self.roll
```

---

## 6. Extended Kalman Filter for IMU

When the system is nonlinear (like 3D orientation), the standard Kalman filter does not work directly. The Extended Kalman Filter (EKF) linearizes around the current estimate.

### 6.1 Nonlinearity in IMU Fusion

Orientation in 3D involves rotation matrices or quaternions -- both are nonlinear. The measurement model (converting gravity vector to orientation angles) uses `atan2`, which is also nonlinear.

### 6.2 EKF Modifications

The EKF replaces the constant matrices A and H with Jacobians evaluated at the current state:

- **Prediction**: Use the nonlinear model $f(x)$, but compute the Jacobian $F = \partial f / \partial x$ for covariance propagation
- **Update**: Use the nonlinear measurement model $h(x)$, but compute $H = \partial h / \partial x$ for the Kalman gain

$$\hat{x}_k^- = f(\hat{x}_{k-1}, u_k)$$
$$P_k^- = F_k \cdot P_{k-1} \cdot F_k^T + Q$$
$$K_k = P_k^- \cdot H_k^T \cdot (H_k \cdot P_k^- \cdot H_k^T + R)^{-1}$$
$$\hat{x}_k = \hat{x}_k^- + K_k \cdot (z_k - h(\hat{x}_k^-))$$

### 6.3 9-DOF IMU Fusion

A 9-DOF (Degree of Freedom) IMU contains:
- 3-axis accelerometer (measures gravity + acceleration)
- 3-axis gyroscope (measures angular velocity)
- 3-axis magnetometer (measures Earth's magnetic field)

The EKF state vector for orientation estimation:

$$x = [q_0, q_1, q_2, q_3, b_{gx}, b_{gy}, b_{gz}]^T$$

Where $q_0..q_3$ is a unit quaternion (orientation) and $b_{gx}, b_{gy}, b_{gz}$ are gyroscope bias estimates. The filter simultaneously estimates orientation and learns the gyroscope drift.

### 6.4 Sensor Fusion Pipeline

```
┌────────────┐   ┌────────────┐   ┌────────────┐
│ Gyroscope  │   │ Accel.     │   │ Magneto.   │
│ 200-1000Hz │   │ 100-400Hz  │   │ 50-100Hz   │
└──────┬─────┘   └──────┬─────┘   └──────┬─────┘
       │                │                │
       ▼                ▼                ▼
  ┌─────────┐     ┌──────────┐    ┌──────────┐
  │ Bias    │     │ Gravity  │    │ Tilt     │
  │ removal │     │ normali- │    │ compen-  │
  │         │     │ zation   │    │ sation   │
  └────┬────┘     └─────┬────┘    └─────┬────┘
       │                │               │
       └────────┬───────┴───────┬───────┘
                │               │
                ▼               ▼
          ┌──────────┐   ┌──────────┐
          │ EKF      │   │ EKF      │
          │ Predict  │──►│ Update   │
          └──────────┘   └────┬─────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ Quaternion → RPY │
                    │ (Roll/Pitch/Yaw) │
                    └──────────────────┘
```

---

## 7. Practical Applications

### 7.1 Drone Stabilization

Drones fuse IMU (200 Hz) + barometer (50 Hz) + GPS (10 Hz) + optical flow (30 Hz):

| Sensor | What It Measures | Fused Into |
|--------|-----------------|------------|
| IMU (accel + gyro) | Attitude (roll/pitch/yaw), angular rates | Orientation + angular velocity |
| Barometer | Altitude (relative) | Height above ground |
| GPS | Lat/lon/altitude (absolute) | Global position |
| Optical flow camera | Ground-relative velocity | Horizontal velocity (indoor) |
| Ultrasonic/LiDAR | Distance to ground | Precision landing height |

### 7.2 Indoor Navigation

Indoor environments lack GPS. Sensor fusion enables positioning through:
- **Pedestrian Dead Reckoning (PDR)**: IMU tracks steps and heading
- **WiFi fingerprinting**: Signal strength maps to position
- **BLE beacons**: Proximity-based location
- **Barometer**: Floor detection in multi-story buildings

The fusion algorithm (usually particle filter or EKF) combines all sources.

### 7.3 Smart Home Environmental Monitoring

Fusing multiple environmental sensors:

```
Temperature (3 sensors, different rooms) → Room-level averages
Humidity + Temperature → Dew point, mold risk
CO2 + Occupancy + HVAC state → Ventilation control
Light + Motion + Time → Presence detection + automation
```

---

## 8. Practice Problems

### Problem 1: Weighted Average Fusion

Three temperature sensors measure the same location:
- Sensor A: 22.1°C, variance = 0.5
- Sensor B: 22.8°C, variance = 0.2
- Sensor C: 22.4°C, variance = 0.3

Calculate the optimal fused temperature and its variance using inverse-variance weighting.

### Problem 2: 1D Kalman Filter

A robot moves along a line. Its position is predicted by a constant-velocity model: $x_k = x_{k-1} + v \cdot \Delta t$, with v = 1 m/s and Δt = 1 s. Process noise Q = 0.1, measurement noise R = 1.0.

Starting from x₀ = 0, P₀ = 1.0, trace 5 steps of the Kalman filter with these measurements: z = [1.2, 2.5, 2.8, 4.1, 5.3]. Show the state estimate and Kalman gain at each step.

### Problem 3: Complementary Filter Tuning

Implement a complementary filter with α = 0.98 for combining accelerometer and gyroscope data. Generate synthetic data:
- True angle: θ(t) = 30° × sin(2π × 0.5 × t) (oscillating)
- Gyroscope: dθ/dt + N(0, 0.1°/s) (low noise)
- Accelerometer angle: θ(t) + N(0, 3°) (high noise)

Plot: raw accelerometer angle, integrated gyroscope angle, and complementary filter output. Show that the filter combines the best of both.

### Problem 4: Sensor Fusion Architecture

Design a sensor fusion system for a delivery robot operating in an office building:
1. List all sensors needed and their update rates
2. Choose a fusion architecture (centralized vs hierarchical)
3. Draw the data flow diagram
4. Specify which filter type (KF, EKF, particle filter) to use at each fusion point

### Problem 5: EKF Bias Estimation

A gyroscope has an unknown constant bias of 0.5°/s. Design an EKF that estimates both the angle and the gyroscope bias simultaneously. State vector: x = [θ, b_gyro]. Show that after sufficient time, the bias estimate converges to the true value.

---

*End of Lesson 14*
