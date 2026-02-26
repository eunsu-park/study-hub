# 10. Sensors and Perception

[← Previous: Robot Control](09_Robot_Control.md) | [Next: State Estimation and Filtering →](11_State_Estimation.md)

---

## Learning Objectives

1. Classify robot sensors into proprioceptive and exteroceptive categories and describe their roles
2. Understand encoder types (incremental, absolute) and inertial measurement units (IMUs)
3. Explain camera models (pinhole, distortion) and camera calibration procedures
4. Describe LiDAR operating principles and point cloud processing fundamentals
5. Apply basic sensor fusion concepts to combine multiple sensor modalities
6. Select appropriate sensors for different robotic applications

---

A robot that cannot perceive its environment is blind — no matter how sophisticated its control algorithms, it will collide, drop objects, and fail at any task requiring awareness of the world. Perception is the process of converting raw sensor signals into useful information about the robot's state and its surroundings. It answers two fundamental questions: *Where am I?* (proprioception and localization) and *What is around me?* (exteroception and environment modeling).

This lesson surveys the sensors used in modern robotics and introduces the mathematical models needed to interpret their data. We will see that no single sensor is perfect — each has strengths and weaknesses. The art of robotic perception lies in combining multiple sensors to compensate for individual limitations, a theme we will develop further in the next lesson on state estimation and filtering.

> **Analogy**: A robot's sensors are like different musical instruments in an orchestra — each captures a different aspect of the performance (melody, rhythm, harmony), but fusion creates the full symphony. A camera captures rich visual detail but struggles in darkness. LiDAR provides precise depth but misses color and texture. An IMU knows how the robot is rotating but drifts over time. Combining them produces a complete, robust perception of the world.

---

## 1. Proprioceptive Sensors

**Proprioceptive sensors** measure the robot's own internal state — joint positions, velocities, forces, and body orientation. These are the robot's sense of its own body.

### 1.1 Encoders

Encoders are the most fundamental robot sensor. They measure the rotation (or linear displacement) of a joint or motor shaft.

#### Incremental Encoders

An incremental encoder produces pulses as the shaft rotates. A quadrature encoder uses two channels (A and B) offset by 90 degrees to determine both speed and direction:

```
Channel A: ___╱‾‾‾╲___╱‾‾‾╲___╱‾‾‾╲___
Channel B: ╱‾‾‾╲___╱‾‾‾╲___╱‾‾‾╲___╱‾‾
                ↑ Count up (A leads B: clockwise)
```

**Resolution**: Given $N$ counts per revolution (CPR), the angular resolution is:

$$\Delta\theta = \frac{2\pi}{N} \text{ rad}$$

With quadrature decoding (counting all edges of both channels), the effective resolution is $4\times$ the base CPR. A 1024 CPR encoder gives $4 \times 1024 = 4096$ counts per revolution, or approximately $0.088°$ per count.

**Velocity estimation** from encoder counts:

$$\hat{\omega} = \frac{\Delta \text{count}}{N_{cpr}} \cdot \frac{2\pi}{\Delta t}$$

```python
class IncrementalEncoder:
    """Simulates an incremental quadrature encoder.

    Why quadrature? Two offset channels let us determine rotation
    direction (which channel leads) and multiply resolution by 4x
    (counting all rising and falling edges on both channels).
    """

    def __init__(self, cpr=1024, gear_ratio=1.0):
        self.cpr = cpr
        self.gear_ratio = gear_ratio
        self.effective_cpr = 4 * cpr  # Quadrature decoding
        self.count = 0
        self.prev_count = 0

    def update(self, motor_angle_rad):
        """Convert motor angle to encoder counts.

        Why track motor angle, not joint angle? In geared robots,
        the encoder is typically on the motor side, where resolution
        is multiplied by the gear ratio. A 100:1 gear ratio with a
        1024 CPR encoder gives 409,600 effective counts per joint revolution.
        """
        motor_counts = motor_angle_rad / (2 * np.pi) * self.effective_cpr
        self.prev_count = self.count
        self.count = int(round(motor_counts))

    def get_joint_angle(self):
        """Convert encoder counts to joint angle."""
        motor_angle = self.count / self.effective_cpr * 2 * np.pi
        return motor_angle / self.gear_ratio

    def get_velocity(self, dt):
        """Estimate joint velocity from count difference.

        Why is velocity estimation noisy? At low speeds, very few counts
        change per sample period, causing quantization noise. Solutions:
        1. Use a longer time window (reduces noise but adds lag)
        2. Use a Kalman filter (optimal trade-off)
        3. Use a dedicated tachometer
        """
        delta_count = self.count - self.prev_count
        motor_vel = delta_count / self.effective_cpr * 2 * np.pi / dt
        return motor_vel / self.gear_ratio
```

#### Absolute Encoders

Unlike incremental encoders, **absolute encoders** report the exact shaft position at all times, even after power cycling. They use a coded disk pattern (Gray code or binary) to produce a unique code for each angular position.

| Feature | Incremental | Absolute |
|---------|-------------|----------|
| Output | Pulse train | Position code |
| Power-up | Must home/reference | Knows position immediately |
| Resolution | Higher (simpler disks) | Lower (complex code tracks) |
| Cost | Lower | Higher |
| Use case | Most joints | Joints that cannot home safely |

### 1.2 Inertial Measurement Unit (IMU)

An IMU combines:
- **Accelerometer** (3-axis): Measures linear acceleration + gravity
- **Gyroscope** (3-axis): Measures angular velocity
- **Magnetometer** (3-axis, optional): Measures magnetic field direction (compass)

A 6-axis IMU (accelerometer + gyroscope) is standard; 9-axis adds the magnetometer.

#### IMU Measurement Model

**Gyroscope** output:

$$\boldsymbol{\omega}_m = \boldsymbol{\omega}_{true} + \mathbf{b}_g + \mathbf{n}_g$$

where $\mathbf{b}_g$ is a slowly drifting bias and $\mathbf{n}_g$ is white noise.

**Accelerometer** output:

$$\mathbf{a}_m = R^T(\mathbf{a}_{true} - \mathbf{g}) + \mathbf{b}_a + \mathbf{n}_a$$

where $R$ is the rotation from world to body frame, $\mathbf{g}$ is gravity, $\mathbf{b}_a$ is bias, and $\mathbf{n}_a$ is noise.

**Key challenge — drift**: Integrating gyroscope angular velocity to get orientation accumulates bias error over time. A gyroscope with 0.01 deg/s bias drifts 36 degrees per hour. This is why IMUs are always fused with other sensors (GPS, vision, encoders).

```python
import numpy as np

class IMUSimulator:
    """Simulates a 6-axis IMU with realistic noise and bias.

    Why model bias separately from noise? Noise is zero-mean and
    can be averaged out. Bias is a slowly varying offset that causes
    unbounded drift when integrated. Estimating and compensating bias
    is a primary goal of IMU fusion algorithms (Kalman filter, etc.).
    """

    def __init__(self, gyro_noise_std=0.01, gyro_bias_std=0.001,
                 accel_noise_std=0.05, accel_bias_std=0.005):
        self.gyro_noise_std = gyro_noise_std     # rad/s
        self.gyro_bias_std = gyro_bias_std       # rad/s (bias random walk)
        self.accel_noise_std = accel_noise_std   # m/s^2
        self.accel_bias_std = accel_bias_std     # m/s^2

        # Initial biases
        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)

        self.gravity = np.array([0, 0, -9.81])

    def measure(self, true_angular_vel, true_accel, rotation_matrix, dt):
        """Generate noisy IMU measurements.

        Why does the accelerometer measure gravity even when stationary?
        An accelerometer measures specific force: the difference between
        true acceleration and gravitational acceleration. A stationary
        accelerometer on a table reads +9.81 m/s^2 upward because it
        measures the table's support force, not 'zero acceleration'.
        """
        # Bias random walk (bias drifts slowly over time)
        self.gyro_bias += np.random.normal(0, self.gyro_bias_std, 3) * np.sqrt(dt)
        self.accel_bias += np.random.normal(0, self.accel_bias_std, 3) * np.sqrt(dt)

        # Gyroscope: true angular velocity + bias + noise
        gyro_meas = (true_angular_vel + self.gyro_bias
                     + np.random.normal(0, self.gyro_noise_std, 3))

        # Accelerometer: rotated (true_accel - gravity) + bias + noise
        accel_meas = (rotation_matrix.T @ (true_accel - self.gravity)
                      + self.accel_bias
                      + np.random.normal(0, self.accel_noise_std, 3))

        return gyro_meas, accel_meas
```

### 1.3 Joint Torque Sensors

Torque sensors measure the force/torque at a joint, typically using strain gauges on a compliant element in the drivetrain. They are essential for:

- **Force control**: Measuring contact forces (via the robot's dynamics model)
- **Collision detection**: Sudden torque spikes indicate unexpected contact
- **Gravity compensation**: Improving force estimation accuracy

**Series Elastic Actuators (SEAs)** embed a known-compliance spring between the motor and the joint. The spring deflection, measured by encoders on both sides, gives a torque measurement:

$$\tau_{joint} = k_{spring} \cdot (\theta_{motor}/N - \theta_{joint})$$

where $k_{spring}$ is the spring stiffness and $N$ is the gear ratio.

---

## 2. Exteroceptive Sensors

**Exteroceptive sensors** measure the external environment — distances to objects, visual appearance, and spatial structure.

### 2.1 Cameras

Cameras are the richest source of information for robots, providing dense 2D images that contain color, texture, shape, and motion cues.

#### Types of Camera Systems

| Type | Output | Depth? | Cost | Use Case |
|------|--------|--------|------|----------|
| Monocular | 2D image | No (requires structure-from-motion) | $ | General vision, SLAM |
| Stereo | 2D image pair | Yes (triangulation) | $$ | Obstacle detection, 3D mapping |
| RGB-D | 2D image + depth map | Yes (structured light or ToF) | $$ | Indoor robotics, manipulation |
| Event camera | Asynchronous events | No | $$$ | High-speed motion, low latency |

#### Resolution, Field of View, and Frame Rate Trade-offs

Cameras force trade-offs:
- **Higher resolution** → more detail, but more data to process and slower frame rate
- **Wider field of view** → see more, but more distortion and lower angular resolution
- **Higher frame rate** → better for fast motion, but more data and potentially more noise (shorter exposure)

### 2.2 LiDAR

**LiDAR** (Light Detection And Ranging) measures distances by timing laser pulses:

$$d = \frac{c \cdot \Delta t}{2}$$

where $c$ is the speed of light and $\Delta t$ is the round-trip time.

#### 2D vs. 3D LiDAR

**2D LiDAR** (single scanning plane):
- Measures distances in one plane (e.g., horizontal)
- Output: array of $(r, \theta)$ measurements
- Typical: 360 degree scan, 10-40 Hz, 0.25-1 degree angular resolution
- Range: 10-30 m, accuracy: 1-3 cm

**3D LiDAR** (multiple scanning planes or rotating array):
- Measures distances in 3D (point cloud)
- Typical: 16-128 channels, 10-20 Hz, millions of points per second
- Range: 100-200 m, accuracy: 1-3 cm

```python
class LiDAR2DSimulator:
    """Simulates a 2D scanning LiDAR.

    Why is LiDAR preferred over cameras for navigation?
    LiDAR provides direct, accurate range measurements unaffected
    by lighting conditions. Cameras give rich appearance information
    but require complex algorithms to extract depth and struggle
    in low light or high dynamic range scenes.
    """

    def __init__(self, n_beams=360, max_range=30.0, noise_std=0.02,
                 fov_deg=360.0):
        self.n_beams = n_beams
        self.max_range = max_range
        self.noise_std = noise_std  # meters
        self.fov = np.deg2rad(fov_deg)

        # Beam angles evenly distributed across FOV
        self.angles = np.linspace(-self.fov/2, self.fov/2,
                                   n_beams, endpoint=False)

    def scan(self, robot_pose, obstacles):
        """Generate a LiDAR scan given robot pose and obstacles.

        Each beam is cast from the robot and the closest intersection
        with any obstacle is returned.

        Why model individual beams? Real LiDAR beams can be partially
        occluded, miss thin objects, or produce multiple returns from
        semi-transparent surfaces. Beam-level modeling captures these effects.
        """
        x, y, theta = robot_pose
        ranges = np.full(self.n_beams, self.max_range)

        for i, angle in enumerate(self.angles):
            beam_angle = theta + angle
            dx = np.cos(beam_angle)
            dy = np.sin(beam_angle)

            # Ray-cast against each obstacle (simplified: circle obstacles)
            for obs_x, obs_y, obs_r in obstacles:
                # Solve |robot + t*direction - obstacle_center|^2 = r^2
                ox = obs_x - x
                oy = obs_y - y
                a = dx*dx + dy*dy
                b = -2*(ox*dx + oy*dy)
                c = ox*ox + oy*oy - obs_r*obs_r
                disc = b*b - 4*a*c

                if disc >= 0:
                    t = (-b - np.sqrt(disc)) / (2*a)
                    if 0 < t < ranges[i]:
                        ranges[i] = t

        # Add measurement noise
        ranges += np.random.normal(0, self.noise_std, self.n_beams)
        ranges = np.clip(ranges, 0, self.max_range)

        return ranges

    def to_cartesian(self, ranges, robot_pose=None):
        """Convert polar scan to Cartesian points.

        Why convert to Cartesian? Many algorithms (ICP, occupancy grid
        mapping, clustering) work with Cartesian coordinates. The
        conversion is straightforward but must account for the robot's
        pose if we want points in the global frame.
        """
        x_local = ranges * np.cos(self.angles)
        y_local = ranges * np.sin(self.angles)

        if robot_pose is not None:
            rx, ry, rtheta = robot_pose
            cos_t = np.cos(rtheta)
            sin_t = np.sin(rtheta)
            x_global = cos_t * x_local - sin_t * y_local + rx
            y_global = sin_t * x_local + cos_t * y_local + ry
            return x_global, y_global

        return x_local, y_local
```

### 2.3 Other Exteroceptive Sensors

**Ultrasonic sensors**: Measure distance using sound waves (40 kHz typical). Cheap and simple but have wide beam angle (poor angular resolution), limited range (2-5 m), and susceptibility to specular reflections on smooth surfaces.

**Infrared (IR) proximity sensors**: Measure short-range distance (10-80 cm) using IR light. Fast and cheap. Used for obstacle detection, cliff detection (prevent robot from falling off edges).

**Force/Torque sensors (wrist-mounted)**: Measure 6-axis forces and torques at the end-effector. Critical for manipulation tasks involving contact (assembly, polishing, human handover).

---

## 3. Camera Models and Calibration

### 3.1 The Pinhole Camera Model

The **pinhole model** describes how 3D points project onto a 2D image:

$$\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \frac{1}{Z_c} \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix}$$

where:
- $(X_c, Y_c, Z_c)$ is the 3D point in camera coordinates
- $(u, v)$ is the pixel coordinate in the image
- $f_x, f_y$ are the focal lengths in pixels
- $(c_x, c_y)$ is the principal point (usually near image center)

The **intrinsic matrix** $K$:

$$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

The **full projection** from world to pixel coordinates includes the **extrinsic** transform:

$$\mathbf{p}_{pixel} \sim K [R | \mathbf{t}] \mathbf{P}_{world}$$

where $[R | \mathbf{t}]$ is the 3x4 camera pose matrix (rotation and translation).

```python
import numpy as np

class PinholeCamera:
    """Pinhole camera model for projection and back-projection.

    Why model the camera mathematically? Every vision algorithm that
    interprets pixel coordinates (feature matching, stereo, visual SLAM)
    needs to map between 2D pixels and 3D rays/points. The intrinsic
    matrix K encapsulates the camera's optical properties.
    """

    def __init__(self, fx, fy, cx, cy, width, height):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
        self.K = np.array([[fx, 0, cx],
                            [0, fy, cy],
                            [0,  0,  1]], dtype=float)

    def project(self, points_3d):
        """Project 3D points (Nx3) to 2D pixels (Nx2).

        Why check Z > 0? Points behind the camera would project to
        valid pixel coordinates but are physically invisible. This is
        a common source of bugs in visual SLAM systems.
        """
        X, Y, Z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]

        # Only project points in front of the camera
        valid = Z > 0.01
        u = np.full_like(X, np.nan)
        v = np.full_like(Y, np.nan)

        u[valid] = self.fx * X[valid] / Z[valid] + self.cx
        v[valid] = self.fy * Y[valid] / Z[valid] + self.cy

        return np.stack([u, v], axis=-1), valid

    def back_project(self, pixels, depth):
        """Back-project 2D pixels + depth to 3D points.

        Why is this called 'back-projection'? It reverses the projection:
        given a pixel (u,v) and its depth Z, we recover the 3D point.
        This is how RGB-D cameras create 3D point clouds from depth images.
        """
        u, v = pixels[:, 0], pixels[:, 1]

        X = (u - self.cx) * depth / self.fx
        Y = (v - self.cy) * depth / self.fy
        Z = depth

        return np.stack([X, Y, Z], axis=-1)
```

### 3.2 Lens Distortion

Real lenses introduce distortion. The two main types are:

**Radial distortion** (barrel/pincushion):

$$x_d = x(1 + k_1 r^2 + k_2 r^4 + k_3 r^6)$$
$$y_d = y(1 + k_1 r^2 + k_2 r^4 + k_3 r^6)$$

where $r^2 = x^2 + y^2$ is the squared distance from the optical center, and $k_1, k_2, k_3$ are radial distortion coefficients.

**Tangential distortion** (due to imperfect lens alignment):

$$x_d = x + 2p_1 xy + p_2(r^2 + 2x^2)$$
$$y_d = y + p_1(r^2 + 2y^2) + 2p_2 xy$$

The five distortion coefficients $[k_1, k_2, p_1, p_2, k_3]$ are estimated during camera calibration.

### 3.3 Camera Calibration

**Camera calibration** estimates the intrinsic parameters $(f_x, f_y, c_x, c_y, k_1, \ldots)$ from images of a known calibration pattern (usually a checkerboard).

**Procedure**:
1. Print a checkerboard pattern with known square size
2. Capture 15-25 images of the checkerboard from different angles and distances
3. Detect checkerboard corners in each image (sub-pixel accuracy)
4. Solve for intrinsic and extrinsic parameters using Zhang's method (homography-based initialization + nonlinear refinement)

```python
def calibrate_camera_overview():
    """Overview of camera calibration using OpenCV.

    Why calibrate? Without calibration, pixel coordinates don't correspond
    to real-world angles. Measurements from uncalibrated cameras have
    systematic errors that compound in 3D reconstruction and SLAM.
    Calibration also estimates distortion coefficients needed to
    'undistort' images for straight-line preservation.
    """
    # Pseudocode — requires OpenCV (cv2)
    # import cv2

    # 1. Define the calibration pattern
    pattern_size = (9, 6)  # Inner corners
    square_size = 0.025    # 25mm squares

    # 2. Generate 3D points of the pattern (Z=0, planar pattern)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0],
                            0:pattern_size[1]].T.reshape(-1, 2) * square_size

    # 3. For each calibration image:
    #    - Find checkerboard corners: cv2.findChessboardCorners()
    #    - Refine to sub-pixel: cv2.cornerSubPix()
    #    - Store object points and image points

    # 4. Calibrate: cv2.calibrateCamera(obj_points, img_points, image_size)
    #    Returns: camera_matrix (K), dist_coeffs, rotation_vecs, translation_vecs

    # 5. Undistort images: cv2.undistort(image, K, dist_coeffs)
    # 6. Evaluate: compute reprojection error (should be < 0.5 pixels)
    pass
```

### 3.4 Stereo Vision

**Stereo cameras** use two cameras separated by a known **baseline** $b$ to estimate depth through triangulation:

$$Z = \frac{f \cdot b}{d}$$

where $d = u_L - u_R$ is the **disparity** (pixel difference between left and right images for the same 3D point).

Key steps in stereo vision:
1. **Rectification**: Transform images so that corresponding points lie on the same horizontal line
2. **Stereo matching**: Find correspondences between left and right images (block matching, semi-global matching)
3. **Triangulation**: Compute depth from disparity using the formula above

**Depth accuracy** degrades with distance: $\Delta Z \propto Z^2 / (fb)$. This means stereo is accurate at close range but noisy at long range.

---

## 4. Point Cloud Processing

LiDAR and depth cameras produce **point clouds** — sets of 3D points $\{(x_i, y_i, z_i)\}_{i=1}^N$.

### 4.1 Filtering and Preprocessing

```python
def preprocess_point_cloud(points, voxel_size=0.05, z_min=-0.5, z_max=3.0):
    """Basic point cloud preprocessing.

    Why voxel downsample? Raw LiDAR scans contain hundreds of thousands
    of points. Downstream algorithms (registration, feature extraction)
    run much faster on a reduced set without significant information loss.
    Voxel grid downsampling replaces all points within each voxel with
    their centroid.
    """
    # 1. Remove outliers based on Z range (e.g., remove ground and ceiling)
    z_mask = (points[:, 2] > z_min) & (points[:, 2] < z_max)
    filtered = points[z_mask]

    # 2. Voxel grid downsampling
    # Assign each point to a voxel
    voxel_indices = np.floor(filtered / voxel_size).astype(int)

    # Unique voxels and their point indices
    _, unique_idx, inverse_idx = np.unique(
        voxel_indices, axis=0, return_index=True, return_inverse=True
    )

    # Compute centroid for each voxel
    downsampled = np.zeros((len(unique_idx), 3))
    for i in range(len(unique_idx)):
        mask = inverse_idx == i
        downsampled[i] = filtered[mask].mean(axis=0)

    return downsampled
```

### 4.2 Point Cloud Registration (ICP)

**Iterative Closest Point (ICP)** aligns two point clouds by iteratively finding correspondences and minimizing the alignment error:

$$T^* = \arg\min_T \sum_{i=1}^N \| T \mathbf{p}_i - \mathbf{q}_{c(i)} \|^2$$

where $T$ is a rigid transform, $\mathbf{p}_i$ are source points, and $\mathbf{q}_{c(i)}$ are the closest points in the target cloud.

```python
def icp_step(source, target, max_dist=1.0):
    """One iteration of the ICP algorithm.

    Why iterate? The optimal transform depends on point correspondences,
    but correspondences depend on the transform. ICP alternates between
    finding correspondences (nearest neighbors) and solving for the
    transform. It converges to a local minimum, so good initialization
    matters (e.g., from odometry or IMU).
    """
    from scipy.spatial import KDTree

    # Step 1: Find nearest neighbors (correspondences)
    tree = KDTree(target)
    distances, indices = tree.query(source)

    # Filter out bad correspondences
    valid = distances < max_dist
    src_matched = source[valid]
    tgt_matched = target[indices[valid]]

    # Step 2: Compute optimal rigid transform (SVD-based)
    src_centroid = src_matched.mean(axis=0)
    tgt_centroid = tgt_matched.mean(axis=0)

    src_centered = src_matched - src_centroid
    tgt_centered = tgt_matched - tgt_centroid

    # Cross-covariance matrix
    H = src_centered.T @ tgt_centered

    # SVD to find optimal rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure proper rotation (det = +1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = tgt_centroid - R @ src_centroid

    # Apply transform
    source_transformed = (R @ source.T).T + t

    mean_error = np.mean(distances[valid])
    return source_transformed, R, t, mean_error
```

### 4.3 Ground Plane Estimation

For mobile robots, separating the ground plane from obstacles is a critical preprocessing step:

$$ax + by + cz + d = 0$$

**RANSAC** (Random Sample Consensus) is the standard approach:

1. Randomly sample 3 points
2. Fit a plane to these points
3. Count how many other points lie within a threshold distance of the plane (inliers)
4. Repeat and keep the plane with the most inliers

```python
def ransac_ground_plane(points, n_iterations=100, threshold=0.05):
    """Estimate ground plane using RANSAC.

    Why RANSAC instead of least squares? Least squares fits to ALL
    points including obstacles, walls, and noise. RANSAC is robust
    to outliers: it finds the plane that has the most support (inliers)
    even if the majority of points are not on the ground.
    """
    best_inliers = 0
    best_plane = None

    for _ in range(n_iterations):
        # Sample 3 random points
        idx = np.random.choice(len(points), 3, replace=False)
        p1, p2, p3 = points[idx]

        # Compute plane normal
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal /= norm

        d = -np.dot(normal, p1)

        # Count inliers
        distances = np.abs(points @ normal + d)
        inliers = np.sum(distances < threshold)

        if inliers > best_inliers:
            best_inliers = inliers
            best_plane = (normal, d)

    # Separate ground and obstacle points
    if best_plane is not None:
        normal, d = best_plane
        distances = np.abs(points @ normal + d)
        ground_mask = distances < threshold
        return best_plane, ground_mask

    return None, np.zeros(len(points), dtype=bool)
```

---

## 5. Sensor Fusion Fundamentals

### 5.1 Why Fuse Sensors?

No sensor is perfect:

| Sensor | Strengths | Weaknesses |
|--------|-----------|------------|
| Encoder | Precise, high rate, cheap | Local only (no environment) |
| IMU | High rate (100-1000 Hz), measures rotation | Drift over time |
| Camera | Rich information, texture | Lighting dependent, no direct depth |
| LiDAR | Accurate depth, lighting invariant | Sparse, expensive, no texture |
| GPS | Global position, no drift | Low rate, poor indoors, 2-5m accuracy |

**Sensor fusion** combines multiple sensors to achieve:

1. **Complementary fusion**: Different sensors measure different things (IMU for rotation + encoder for joint position)
2. **Redundant fusion**: Multiple sensors measure the same thing for improved accuracy (GPS + visual odometry for position)
3. **Cooperative fusion**: Sensors work together in ways neither could alone (stereo camera pair for depth)

### 5.2 Simple Complementary Filter

For combining a gyroscope (accurate short-term, drifts long-term) with an accelerometer (noisy short-term, accurate long-term for gravity direction):

$$\hat{\theta} = \alpha(\hat{\theta}_{prev} + \omega_{gyro} \cdot dt) + (1 - \alpha) \cdot \theta_{accel}$$

where $\alpha \in [0.95, 0.99]$ gives high weight to the gyroscope for short-term changes and low weight to the accelerometer for long-term correction.

```python
class ComplementaryFilter:
    """Simple complementary filter for IMU orientation.

    Why not just integrate the gyroscope? Gyroscope bias causes
    unbounded drift. Why not just use the accelerometer? It's noisy
    and affected by robot acceleration. The complementary filter
    is essentially a 'trust' blend: trust the gyro for fast changes
    and the accelerometer for the long-term average (gravity direction).
    """

    def __init__(self, alpha=0.98):
        self.alpha = alpha  # High = trust gyro more
        self.angle = 0.0

    def update(self, gyro_rate, accel_angle, dt):
        """Update angle estimate.

        Why does this work? In the frequency domain, the complementary
        filter applies a high-pass filter to the gyroscope (keeping fast
        dynamics) and a low-pass filter to the accelerometer (keeping
        steady-state). Together they cover the full frequency range.
        """
        # Gyroscope: integrate for short-term angle change
        gyro_estimate = self.angle + gyro_rate * dt

        # Blend with accelerometer-derived angle
        self.angle = self.alpha * gyro_estimate + (1 - self.alpha) * accel_angle

        return self.angle
```

### 5.3 From Complementary Filter to Kalman Filter

The complementary filter is an intuitive starting point, but it has limitations:

- Fixed blend ratio $\alpha$ does not adapt to changing noise conditions
- Cannot handle multiple sensors with different update rates
- No formal uncertainty quantification

The **Kalman filter** (covered in the next lesson) provides the optimal solution to the sensor fusion problem under Gaussian noise assumptions. It can be viewed as a generalization of the complementary filter where the blend ratio is computed optimally from the sensor noise models.

### 5.4 Sensor Selection for Common Platforms

| Platform | Typical Sensors | Rationale |
|----------|----------------|-----------|
| Industrial arm | Encoders, F/T sensor | High precision joints, contact tasks |
| Mobile robot (indoor) | LiDAR, IMU, wheel encoders | Navigation without GPS |
| Mobile robot (outdoor) | GPS, IMU, LiDAR, camera | Global + local positioning |
| Drone | IMU, barometer, GPS, camera | 6-DOF state estimation |
| Humanoid | IMU, joint encoders, F/T (feet), camera | Balance + perception |
| Self-driving car | LiDAR, cameras (×8+), radar, IMU, GPS | 360° perception, redundancy for safety |

---

## 6. Depth Sensing Technologies

### 6.1 Structured Light

Projects a known pattern (dots, lines, or speckle) onto the scene. A camera observes how the pattern deforms on surfaces, and triangulation computes depth. **Example**: Intel RealSense D400 series.

**Pros**: Dense depth maps, works indoors
**Cons**: Fails in sunlight (IR pattern washed out), limited range (0.3-10 m)

### 6.2 Time-of-Flight (ToF)

Emits modulated light and measures phase shift to compute distance:

$$d = \frac{c \cdot \Delta\phi}{4\pi f_{mod}}$$

where $f_{mod}$ is the modulation frequency and $\Delta\phi$ is the phase difference.

**Pros**: Works at any lighting, fast, compact
**Cons**: Lower resolution, multi-path interference, limited range

### 6.3 Comparison

| Technology | Range | Resolution | Outdoor? | Cost |
|------------|-------|------------|----------|------|
| Structured light | 0.3-10 m | High (640×480+) | No | $$ |
| ToF camera | 0.3-5 m | Low (320×240) | Partial | $$ |
| Stereo camera | 0.5-20 m | High | Yes | $$ |
| LiDAR (3D) | 1-200 m | Sparse (point cloud) | Yes | $$$$ |

---

## Summary

| Concept | Key Idea |
|---------|----------|
| Proprioceptive sensors | Measure robot's internal state (encoders, IMU, torque sensors) |
| Incremental encoder | Pulse counting with quadrature for direction; must home on startup |
| Absolute encoder | Unique code per position; knows position after power cycle |
| IMU | Accelerometer + gyroscope; high rate but drifts without fusion |
| Pinhole camera model | 3D-to-2D projection via intrinsic matrix $K$ |
| Lens distortion | Radial and tangential distortion; corrected via calibration |
| LiDAR | Direct range measurement via laser time-of-flight; accurate, lighting invariant |
| ICP | Iterative alignment of point clouds; nearest-neighbor + SVD per iteration |
| RANSAC | Robust model fitting; immune to outliers |
| Sensor fusion | Combine sensors to compensate for individual weaknesses |
| Complementary filter | Frequency-domain blend of gyro (high-pass) and accelerometer (low-pass) |

---

## Exercises

1. **Encoder resolution analysis**: A 6-DOF robot arm has 1024 CPR incremental encoders with quadrature decoding and 100:1 gear ratios. Calculate the angular resolution at each joint in degrees. If the end-effector is 1 m from the base, estimate the worst-case position resolution at the end-effector.

2. **Camera calibration**: Using the pinhole camera model with $f_x = f_y = 500$ pixels and $c_x = 320$, $c_y = 240$, project a set of 3D points $[(1,0,5), (0,1,5), (-1,-1,10)]$ to pixel coordinates. Then back-project them using their known depths and verify you recover the original 3D points.

3. **LiDAR scan matching**: Generate two 2D LiDAR scans of a simple environment (e.g., rectangular room with circular obstacles). Add a known rigid transform (rotation + translation) to create the second scan. Implement ICP to recover the transform and measure the error as a function of the initial displacement.

4. **Complementary filter**: Simulate an IMU measuring the pitch angle of a robot that executes a sinusoidal rocking motion. Generate noisy gyroscope and accelerometer signals. Implement the complementary filter and compare the result with the ground truth for different values of $\alpha$.

5. **Sensor selection report**: For a warehouse robot that must navigate aisles, pick items from shelves, and deliver them to packing stations, propose a sensor suite. Justify each sensor choice considering cost, environment conditions (indoor, structured), required accuracy, and redundancy.

---

## Further Reading

- Corke, P. *Robotics, Vision and Control*, 3rd ed. Springer, 2023. Part III: Vision. (Comprehensive robotics + vision textbook)
- Hartley, R. and Zisserman, A. *Multiple View Geometry in Computer Vision*, 2nd ed. Cambridge, 2004. (Definitive reference for camera geometry)
- Zhang, Z. "A Flexible New Technique for Camera Calibration." *IEEE TPAMI*, 2000. (The standard calibration method)
- Besl, P. and McKay, N. "A Method for Registration of 3-D Shapes." *IEEE TPAMI*, 1992. (Original ICP paper)
- Fischler, M. and Bolles, R. "Random Sample Consensus." *Communications of the ACM*, 1981. (RANSAC)

---

[← Previous: Robot Control](09_Robot_Control.md) | [Next: State Estimation and Filtering →](11_State_Estimation.md)
