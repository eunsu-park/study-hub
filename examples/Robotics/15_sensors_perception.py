"""
Sensors and Perception Pipeline
================================
Simulate common robot sensor models and demonstrate basic sensor fusion.

Robotics perception combines data from multiple sensors to build a coherent
understanding of the environment. This example simulates three key sensors:

  1. **Encoders** (proprioceptive): Measure wheel rotations to estimate
     odometry. Cheap and fast, but drift over time due to slip and
     integration error.

  2. **LiDAR** (exteroceptive): Emits laser beams in a fan pattern and
     measures the distance to obstacles via time-of-flight. Provides
     accurate range data but no color/texture.

  3. **Camera** (exteroceptive): Projects 3D landmarks onto a 2D image
     plane using the pinhole model. Rich information, but depth is lost
     without stereo or depth sensors.

We then demonstrate a simple **sensor fusion** approach: complementary
weighting of encoder odometry and LiDAR-based position corrections,
showing how fusion reduces drift compared to odometry alone.

Coordinate frames:
  - World frame: fixed, origin at (0, 0)
  - Robot frame: moves with the robot, x-forward, y-left
  - Sensor frames: attached to the robot body
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Encoder simulation
# ---------------------------------------------------------------------------
class EncoderOdometry:
    """Simulate incremental encoder odometry for a differential-drive robot.

    Quadrature encoders count ticks as wheels rotate. Given the wheel radius
    and counts-per-revolution, we convert ticks to distance traveled.

    Odometry integrates these incremental motions to estimate the robot pose.
    The fundamental problem: small errors in each step accumulate over time,
    causing unbounded drift — especially in heading, which then corrupts
    the position estimate.

    Parameters:
        wheel_radius: Wheel radius in meters.
        wheel_base: Distance between left and right wheels (meters).
        ticks_per_rev: Encoder counts per revolution (CPR x 4 for quadrature).
        noise_std_ticks: Standard deviation of tick count noise.
    """

    def __init__(self, wheel_radius: float = 0.05, wheel_base: float = 0.3,
                 ticks_per_rev: int = 4096, noise_std_ticks: float = 2.0):
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base
        self.meters_per_tick = (2 * np.pi * wheel_radius) / ticks_per_rev
        self.noise_std = noise_std_ticks

        # Estimated pose from odometry
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

    def update(self, true_left_ticks: float, true_right_ticks: float):
        """Process encoder ticks (with simulated noise) to update pose.

        Differential-drive kinematics:
            d_left  = left_ticks  * meters_per_tick
            d_right = right_ticks * meters_per_tick
            d_center = (d_left + d_right) / 2
            d_theta  = (d_right - d_left) / wheel_base
        """
        # Add quantization and slip noise
        left = true_left_ticks + np.random.normal(0, self.noise_std)
        right = true_right_ticks + np.random.normal(0, self.noise_std)

        d_left = left * self.meters_per_tick
        d_right = right * self.meters_per_tick

        d_center = (d_left + d_right) / 2.0
        d_theta = (d_right - d_left) / self.wheel_base

        # Update pose using midpoint integration
        self.x += d_center * np.cos(self.theta + d_theta / 2.0)
        self.y += d_center * np.sin(self.theta + d_theta / 2.0)
        self.theta += d_theta

        return np.array([self.x, self.y, self.theta])


# ---------------------------------------------------------------------------
# LiDAR simulation
# ---------------------------------------------------------------------------
class LiDAR:
    """Simulate a 2D LiDAR scanner.

    A LiDAR sensor emits laser beams at regular angular intervals and
    measures the distance to the nearest obstacle for each beam. In 2D
    (planar LiDAR, e.g., RPLIDAR, Hokuyo), the output is a vector of
    ranges indexed by angle.

    We model circular obstacles and compute ray-circle intersection to
    determine range readings. Noise is added to simulate real sensor
    imprecision (typically 1-3 cm for commercial LiDARs).

    Parameters:
        n_beams: Number of laser beams.
        max_range: Maximum detection range (meters).
        fov: Field of view in radians (default: full 360 degrees).
        noise_std: Range measurement noise standard deviation (meters).
    """

    def __init__(self, n_beams: int = 180, max_range: float = 10.0,
                 fov: float = 2 * np.pi, noise_std: float = 0.02):
        self.n_beams = n_beams
        self.max_range = max_range
        self.angles = np.linspace(-fov / 2, fov / 2, n_beams, endpoint=False)
        self.noise_std = noise_std

    def scan(self, robot_x: float, robot_y: float, robot_theta: float,
             obstacles: np.ndarray) -> np.ndarray:
        """Generate a LiDAR scan from the robot's current pose.

        Args:
            robot_x, robot_y, robot_theta: Robot pose in world frame.
            obstacles: Array of shape (N, 3) — each row is [cx, cy, radius].

        Returns:
            ranges: Array of shape (n_beams,) with measured distances.
        """
        ranges = np.full(self.n_beams, self.max_range)

        for i, angle in enumerate(self.angles):
            beam_angle = robot_theta + angle

            # Ray direction
            dx = np.cos(beam_angle)
            dy = np.sin(beam_angle)

            for obs in obstacles:
                cx, cy, r = obs
                # Vector from robot to obstacle center
                ox = cx - robot_x
                oy = cy - robot_y

                # Solve quadratic for ray-circle intersection
                # |P + t*D - C|^2 = r^2
                a = 1.0  # dx^2 + dy^2
                b = 2.0 * (dx * (-ox) + dy * (-oy))
                c = ox**2 + oy**2 - r**2

                discriminant = b**2 - 4 * a * c
                if discriminant < 0:
                    continue

                sqrt_disc = np.sqrt(discriminant)
                t1 = (-b - sqrt_disc) / (2 * a)
                t2 = (-b + sqrt_disc) / (2 * a)

                # Take nearest positive intersection
                t = t1 if t1 > 0 else t2
                if t > 0:
                    ranges[i] = min(ranges[i], t)

        # Add Gaussian noise
        ranges += np.random.normal(0, self.noise_std, self.n_beams)
        ranges = np.clip(ranges, 0, self.max_range)

        return ranges


# ---------------------------------------------------------------------------
# Pinhole camera model
# ---------------------------------------------------------------------------
class PinholeCamera:
    """Simulate a pinhole camera projecting 3D landmarks onto a 2D image.

    The pinhole model is the simplest camera model:
        u = fx * (X / Z) + cx
        v = fy * (Y / Z) + cy

    where (X, Y, Z) are coordinates in the camera frame, (fx, fy) are focal
    lengths in pixels, and (cx, cy) is the principal point.

    This example projects known 3D landmarks and returns their pixel
    coordinates with simulated noise, demonstrating how depth information
    is lost in monocular projection.

    Parameters:
        fx, fy: Focal lengths (pixels).
        cx, cy: Principal point (pixels).
        img_width, img_height: Image dimensions.
        noise_std: Pixel noise standard deviation.
    """

    def __init__(self, fx: float = 500.0, fy: float = 500.0,
                 cx: float = 320.0, cy: float = 240.0,
                 img_width: int = 640, img_height: int = 480,
                 noise_std: float = 1.0):
        self.K = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0,  0,  1]])
        self.img_width = img_width
        self.img_height = img_height
        self.noise_std = noise_std

    def project(self, landmarks_world: np.ndarray,
                robot_x: float, robot_y: float,
                robot_theta: float) -> list:
        """Project 3D world landmarks to 2D pixel coordinates.

        Transforms landmarks from world frame to camera frame (assuming
        camera is at robot pose, facing forward), then applies projection.

        Returns:
            List of (u, v, landmark_index) for landmarks visible in the image.
        """
        # Robot-to-world rotation (2D, camera looks along robot x-axis)
        cos_t = np.cos(robot_theta)
        sin_t = np.sin(robot_theta)

        projections = []
        for idx, lm in enumerate(landmarks_world):
            # Transform to robot frame
            dx = lm[0] - robot_x
            dy = lm[1] - robot_y

            # In camera frame: x_cam = forward, y_cam = left, z_cam = up
            # For pinhole: X = -y_cam, Y = -z_cam, Z = x_cam
            x_robot = cos_t * dx + sin_t * dy
            y_robot = -sin_t * dx + cos_t * dy

            # Only project landmarks in front of the camera
            if x_robot < 0.5:
                continue

            # Pinhole projection (2D world, so z_cam = 0 for ground landmarks)
            u = self.K[0, 0] * (-y_robot / x_robot) + self.K[0, 2]
            v = self.K[1, 2]  # On the horizontal center line (2D world)

            # Add noise
            u += np.random.normal(0, self.noise_std)
            v += np.random.normal(0, self.noise_std)

            # Check if within image bounds
            if 0 <= u < self.img_width and 0 <= v < self.img_height:
                projections.append((u, v, idx))

        return projections


# ---------------------------------------------------------------------------
# Simple sensor fusion: odometry + LiDAR position correction
# ---------------------------------------------------------------------------
def fuse_odometry_lidar(odom_pose: np.ndarray, lidar_correction: np.ndarray,
                        alpha: float = 0.3) -> np.ndarray:
    """Complementary filter: blend odometry with LiDAR-based correction.

    A full sensor fusion system would use an EKF or particle filter (see
    examples 09 and 11). Here we demonstrate the principle with a simple
    weighted average — the "complementary filter" approach often used as
    a first step before implementing more sophisticated methods.

    fused = (1 - alpha) * odometry + alpha * lidar_correction

    alpha near 0: trust odometry more (smooth but drifts)
    alpha near 1: trust LiDAR more (noisy but no drift)
    """
    fused = (1 - alpha) * odom_pose[:2] + alpha * lidar_correction[:2]
    # Use odometry heading (LiDAR heading estimation is complex)
    return np.array([fused[0], fused[1], odom_pose[2]])


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def demo_sensors_perception():
    """Simulate sensors on a mobile robot and demonstrate fusion benefits."""
    print("=" * 60)
    print("Sensors and Perception Demo")
    print("=" * 60)

    np.random.seed(42)
    dt = 0.1
    n_steps = 200

    # Environment: circular obstacles
    obstacles = np.array([
        [3.0, 2.0, 0.5],
        [5.0, -1.0, 0.8],
        [7.0, 3.0, 0.6],
        [2.0, -2.0, 0.4],
        [8.0, 0.0, 0.7],
    ])

    # Known landmarks (for camera)
    landmarks = obstacles[:, :2]

    # Sensors
    encoder = EncoderOdometry()
    lidar = LiDAR(n_beams=90, max_range=8.0)
    camera = PinholeCamera()

    # True robot state
    x_true, y_true, theta_true = 0.0, 0.0, 0.0

    # Storage
    true_path = []
    odom_path = []
    fused_path = []

    ticks_per_meter = 1.0 / encoder.meters_per_tick

    for step in range(n_steps):
        t = step * dt

        # Control: gentle curve
        v = 0.5
        omega = 0.15 * np.sin(0.1 * t)

        # True motion
        x_true += v * np.cos(theta_true) * dt
        y_true += v * np.sin(theta_true) * dt
        theta_true += omega * dt

        # Convert to encoder ticks
        d_center = v * dt
        d_theta = omega * dt
        d_left = (d_center - 0.5 * encoder.wheel_base * d_theta)
        d_right = (d_center + 0.5 * encoder.wheel_base * d_theta)
        left_ticks = d_left * ticks_per_meter
        right_ticks = d_right * ticks_per_meter

        odom_pose = encoder.update(left_ticks, right_ticks)

        # LiDAR scan (we use it to generate a "correction" —
        # in practice this would come from scan matching or AMCL)
        _ranges = lidar.scan(x_true, y_true, theta_true, obstacles)

        # Simulate a LiDAR-based position estimate (true + noise, no drift)
        lidar_est = np.array([
            x_true + np.random.normal(0, 0.05),
            y_true + np.random.normal(0, 0.05),
            theta_true + np.random.normal(0, 0.01),
        ])

        fused_pose = fuse_odometry_lidar(odom_pose, lidar_est, alpha=0.4)

        true_path.append([x_true, y_true])
        odom_path.append([odom_pose[0], odom_pose[1]])
        fused_path.append([fused_pose[0], fused_pose[1]])

    true_path = np.array(true_path)
    odom_path = np.array(odom_path)
    fused_path = np.array(fused_path)

    # --- Camera projection snapshot at final pose ---
    projections = camera.project(landmarks, x_true, y_true, theta_true)

    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Trajectory comparison
    ax1 = axes[0]
    ax1.plot(true_path[:, 0], true_path[:, 1], 'b-', lw=2, label='True path')
    ax1.plot(odom_path[:, 0], odom_path[:, 1], 'r--', lw=1.5,
             label='Encoder odometry', alpha=0.7)
    ax1.plot(fused_path[:, 0], fused_path[:, 1], 'g-.', lw=1.5,
             label='Fused (odom+LiDAR)', alpha=0.8)
    for obs in obstacles:
        circle = plt.Circle((obs[0], obs[1]), obs[2],
                             color='gray', alpha=0.4)
        ax1.add_patch(circle)
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_title("Trajectory: Odometry vs Fusion")
    ax1.legend(fontsize=8)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # 2. LiDAR scan visualization (final pose)
    ax2 = axes[1]
    final_ranges = lidar.scan(x_true, y_true, theta_true, obstacles)
    beam_x = x_true + final_ranges * np.cos(theta_true + lidar.angles)
    beam_y = y_true + final_ranges * np.sin(theta_true + lidar.angles)
    ax2.scatter(beam_x, beam_y, s=2, c='red', label='LiDAR hits')
    ax2.plot(x_true, y_true, 'bo', markersize=8, label='Robot')
    for obs in obstacles:
        circle = plt.Circle((obs[0], obs[1]), obs[2],
                             color='gray', alpha=0.3)
        ax2.add_patch(circle)
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_title("LiDAR Point Cloud (Final Pose)")
    ax2.legend(fontsize=8)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # 3. Drift error over time
    ax3 = axes[2]
    odom_err = np.sqrt(np.sum((odom_path - true_path)**2, axis=1))
    fused_err = np.sqrt(np.sum((fused_path - true_path)**2, axis=1))
    t_arr = np.arange(n_steps) * dt
    ax3.plot(t_arr, odom_err, 'r-', label='Odometry error')
    ax3.plot(t_arr, fused_err, 'g-', label='Fused error')
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Position Error (m)")
    ax3.set_title("Drift: Odometry vs Fusion")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.suptitle("Robot Sensors and Perception", fontsize=14)
    plt.tight_layout()
    plt.savefig("15_sensors_perception.png", dpi=120)
    plt.show()

    # Summary
    print(f"\nFinal odometry drift: {odom_err[-1]:.3f} m")
    print(f"Final fused error:   {fused_err[-1]:.3f} m")
    print(f"Drift reduction:     {(1 - fused_err[-1]/odom_err[-1])*100:.1f}%")
    print(f"Camera projections at final pose: {len(projections)} landmarks visible")


if __name__ == "__main__":
    demo_sensors_perception()
