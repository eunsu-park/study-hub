"""
EKF-SLAM: Simultaneous Localization and Mapping
=================================================
Estimate robot pose AND landmark positions simultaneously.

In localization (09_kalman_filter.py), we assumed landmark positions were known.
In SLAM, the robot must build a map of landmarks while simultaneously localizing
itself within that map — a chicken-and-egg problem.

EKF-SLAM maintains a joint state vector:
    x = [x_robot, y_robot, theta, lm1_x, lm1_y, lm2_x, lm2_y, ...]

The state grows as new landmarks are discovered (state augmentation).
Correlations between robot pose and landmarks are crucial: when we
re-observe a landmark, the correlation allows us to correct both the
robot pose AND the landmark position.

This is a simplified but complete EKF-SLAM implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


# ---------------------------------------------------------------------------
# State management helpers
# ---------------------------------------------------------------------------
def robot_state(x):
    """Extract robot pose from the full SLAM state vector."""
    return x[:3]


def landmark_state(x, idx):
    """Extract the (x, y) of landmark idx from the state vector.

    Landmark i is stored at indices [3 + 2*i, 3 + 2*i + 1].
    """
    start = 3 + 2 * idx
    return x[start:start + 2]


def n_landmarks(x):
    """Count the number of landmarks currently in the state."""
    return (len(x) - 3) // 2


# ---------------------------------------------------------------------------
# EKF-SLAM
# ---------------------------------------------------------------------------
class EKFSLAM:
    """Extended Kalman Filter SLAM.

    The key insight of EKF-SLAM: the covariance matrix captures correlations
    between the robot and ALL landmarks. When we observe landmark j:
      - We correct the robot pose (as in localization)
      - We also correct ALL other landmark positions through their
        correlations with the robot pose

    This "loop closure" effect is what makes SLAM work: revisiting known
    landmarks reduces uncertainty across the entire map.

    Limitations:
      - O(n^2) per update where n = number of landmarks (quadratic scaling)
      - Assumes known data association (which measurement corresponds to which landmark)
      - Gaussian approximation may fail for ambiguous environments
    """

    def __init__(self, Q: np.ndarray, R: np.ndarray):
        """
        Args:
            Q: Process noise covariance (3x3) for robot motion
            R: Measurement noise covariance (2x2) for range-bearing
        """
        self.Q = Q
        self.R = R

        # Initialize with robot state only
        self.x = np.zeros(3)
        self.P = np.diag([0.01, 0.01, 0.01])  # Small initial uncertainty

        # Data association: track which landmarks have been observed
        self.landmark_ids = {}  # Maps external ID → internal index

    def predict(self, u: np.ndarray, dt: float):
        """Prediction step: propagate robot pose, keep landmarks unchanged.

        Only the robot state changes during prediction. The Jacobian F_x
        is identity except for the 3x3 robot block. This means landmarks
        don't move (they are static), but their correlations with the robot
        change because the robot moves.
        """
        n = len(self.x)
        theta = self.x[2]
        v, omega = u

        # Update robot state
        self.x[0] += v * np.cos(theta) * dt
        self.x[1] += v * np.sin(theta) * dt
        self.x[2] += omega * dt
        self.x[2] = (self.x[2] + np.pi) % (2 * np.pi) - np.pi

        # Jacobian of motion model (only robot block is non-trivial)
        F = np.eye(n)
        F[0, 2] = -v * np.sin(theta) * dt
        F[1, 2] = v * np.cos(theta) * dt

        # Process noise only affects robot state
        Q_full = np.zeros((n, n))
        Q_full[:3, :3] = self.Q

        self.P = F @ self.P @ F.T + Q_full

    def update(self, z: np.ndarray, landmark_id: int):
        """Update step: correct state using a range-bearing measurement.

        If the landmark has been seen before, we do a standard EKF update.
        If it is new, we first augment the state (add the landmark) and
        then update.

        Args:
            z: Measurement [range, bearing]
            landmark_id: External identifier for the landmark
        """
        if landmark_id not in self.landmark_ids:
            # New landmark: augment the state
            self._add_landmark(z, landmark_id)
            return

        # Existing landmark: standard EKF update
        lm_idx = self.landmark_ids[landmark_id]
        lm_pos = landmark_state(self.x, lm_idx)

        # Expected measurement
        dx = lm_pos[0] - self.x[0]
        dy = lm_pos[1] - self.x[1]
        r_pred = np.sqrt(dx**2 + dy**2)
        b_pred = np.arctan2(dy, dx) - self.x[2]
        z_pred = np.array([r_pred, b_pred])

        # Innovation
        innovation = z - z_pred
        innovation[1] = (innovation[1] + np.pi) % (2 * np.pi) - np.pi

        # Measurement Jacobian H: [2 x n]
        n = len(self.x)
        H = np.zeros((2, n))
        r_sq = dx**2 + dy**2
        r = np.sqrt(r_sq)

        # Derivatives w.r.t. robot state [x, y, theta]
        H[0, 0] = -dx / r
        H[0, 1] = -dy / r
        H[1, 0] = dy / r_sq
        H[1, 1] = -dx / r_sq
        H[1, 2] = -1

        # Derivatives w.r.t. landmark state [lm_x, lm_y]
        lm_start = 3 + 2 * lm_idx
        H[0, lm_start] = dx / r
        H[0, lm_start + 1] = dy / r
        H[1, lm_start] = -dy / r_sq
        H[1, lm_start + 1] = dx / r_sq

        # Kalman gain
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        # State and covariance correction
        self.x = self.x + K @ innovation
        self.x[2] = (self.x[2] + np.pi) % (2 * np.pi) - np.pi

        I_KH = np.eye(n) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

    def _add_landmark(self, z: np.ndarray, landmark_id: int):
        """State augmentation: add a new landmark to the state vector.

        When we observe a landmark for the first time, we initialize its
        position from the measurement and the current robot pose. The new
        landmark gets large initial uncertainty (we have only one observation).

        The augmented covariance includes cross-correlations between the
        new landmark and the existing state — these are zero initially but
        will grow as we make more observations.
        """
        r, bearing = z
        theta = self.x[2]

        # Initialize landmark position from measurement
        lm_x = self.x[0] + r * np.cos(theta + bearing)
        lm_y = self.x[1] + r * np.sin(theta + bearing)

        # Assign internal index
        idx = n_landmarks(self.x)
        self.landmark_ids[landmark_id] = idx

        # Augment state
        self.x = np.append(self.x, [lm_x, lm_y])

        # Augment covariance
        n_old = len(self.P)
        n_new = n_old + 2

        # Jacobian of landmark initialization w.r.t. robot state
        G_r = np.array([
            [1, 0, -r * np.sin(theta + bearing)],
            [0, 1,  r * np.cos(theta + bearing)]
        ])

        P_new = np.zeros((n_new, n_new))
        P_new[:n_old, :n_old] = self.P

        # Cross-correlations between new landmark and existing state
        P_new[n_old:, :3] = G_r @ self.P[:3, :3]
        P_new[:3, n_old:] = P_new[n_old:, :3].T

        # New landmark's own covariance (large initial uncertainty)
        P_new[n_old:, n_old:] = G_r @ self.P[:3, :3] @ G_r.T + 10 * self.R

        self.P = P_new


# ---------------------------------------------------------------------------
# Data association: nearest neighbor
# ---------------------------------------------------------------------------
def nearest_neighbor_association(slam: EKFSLAM, z: np.ndarray,
                                   threshold: float = 5.0) -> int:
    """Simple nearest-neighbor data association.

    For each measurement, find the existing landmark whose predicted
    measurement is closest. If no landmark is close enough (Mahalanobis
    distance > threshold), it is treated as a new landmark.

    In real SLAM systems, data association is often the hardest part.
    More robust methods include JCBB (Joint Compatibility Branch & Bound)
    and maximum-likelihood association.
    """
    if n_landmarks(slam.x) == 0:
        return -1  # No landmarks yet → new

    best_id = -1
    best_dist = threshold

    robot = robot_state(slam.x)

    for ext_id, idx in slam.landmark_ids.items():
        lm = landmark_state(slam.x, idx)
        dx = lm[0] - robot[0]
        dy = lm[1] - robot[1]
        r_pred = np.sqrt(dx**2 + dy**2)
        b_pred = np.arctan2(dy, dx) - robot[2]

        z_pred = np.array([r_pred, b_pred])
        residual = z - z_pred
        residual[1] = (residual[1] + np.pi) % (2 * np.pi) - np.pi

        # Euclidean distance in measurement space (simplified)
        dist = np.linalg.norm(residual)
        if dist < best_dist:
            best_dist = dist
            best_id = ext_id

    return best_id


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_slam_state(ax, slam: EKFSLAM, true_landmarks: np.ndarray,
                     true_robot: np.ndarray):
    """Visualize the SLAM estimate vs ground truth."""
    robot = robot_state(slam.x)

    # Draw estimated landmarks with uncertainty ellipses
    for ext_id, idx in slam.landmark_ids.items():
        lm = landmark_state(slam.x, idx)
        start = 3 + 2 * idx
        lm_cov = slam.P[start:start+2, start:start+2]

        evals, evecs = np.linalg.eigh(lm_cov)
        evals = np.maximum(evals, 0)
        angle = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))
        w = 2 * 2 * np.sqrt(evals[0])
        h = 2 * 2 * np.sqrt(evals[1])

        ellipse = Ellipse(xy=lm, width=w, height=h, angle=angle,
                           fill=False, edgecolor='blue', linewidth=1, alpha=0.6)
        ax.add_patch(ellipse)
        ax.plot(lm[0], lm[1], 'b+', markersize=10, markeredgewidth=2)

    # True landmarks
    ax.plot(true_landmarks[:, 0], true_landmarks[:, 1], 'k^',
            markersize=10, label='True landmarks')

    # True robot
    ax.plot(true_robot[0], true_robot[1], 'go', markersize=8)

    # Estimated robot
    ax.plot(robot[0], robot[1], 'rs', markersize=8)


def demo_ekf_slam():
    """Run EKF-SLAM with a robot navigating among landmarks."""
    print("=" * 60)
    print("EKF-SLAM Demo")
    print("=" * 60)

    np.random.seed(42)

    # Environment: landmarks scattered in a 20x20 area
    true_landmarks = np.array([
        [5, 5], [15, 5], [10, 10], [5, 15], [15, 15],
        [0, 10], [20, 10], [10, 0], [10, 20], [8, 7],
    ])
    n_lm = len(true_landmarks)

    # Noise parameters
    Q = np.diag([0.05**2, 0.05**2, 0.01**2])
    R = np.diag([0.3**2, 0.05**2])

    slam = EKFSLAM(Q, R)

    # Simulation
    dt = 0.1
    n_steps = 500
    max_range = 8.0

    # True robot state
    x_true = np.array([10.0, 10.0, 0.0])
    slam.x[:3] = x_true.copy() + np.random.normal(0, 0.1, 3)

    true_path = [x_true.copy()]
    est_path = [slam.x[:3].copy()]
    next_landmark_id = 0

    for step in range(n_steps):
        t = step * dt

        # Control: circular trajectory
        v = 1.5
        omega = 0.3
        u = np.array([v, omega])

        # True motion with noise
        noise_v = v + np.random.normal(0, 0.2)
        noise_omega = omega + np.random.normal(0, 0.05)
        x_true = x_true + np.array([
            noise_v * np.cos(x_true[2]) * dt,
            noise_v * np.sin(x_true[2]) * dt,
            noise_omega * dt
        ])
        x_true[2] = (x_true[2] + np.pi) % (2 * np.pi) - np.pi

        # EKF predict
        slam.predict(u, dt)

        # Generate measurements to visible landmarks
        for lm_id in range(n_lm):
            dx = true_landmarks[lm_id, 0] - x_true[0]
            dy = true_landmarks[lm_id, 1] - x_true[1]
            r_true = np.sqrt(dx**2 + dy**2)

            if r_true > max_range:
                continue

            b_true = np.arctan2(dy, dx) - x_true[2]
            z = np.array([
                r_true + np.random.normal(0, 0.3),
                b_true + np.random.normal(0, 0.05)
            ])

            # Data association
            assoc_id = nearest_neighbor_association(slam, z, threshold=3.0)

            if assoc_id == -1:
                # New landmark
                slam.update(z, lm_id)
            else:
                slam.update(z, assoc_id)

        true_path.append(x_true.copy())
        est_path.append(slam.x[:3].copy())

    true_path = np.array(true_path)
    est_path = np.array(est_path)

    # --- Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Map and trajectory
    ax = axes[0]
    ax.plot(true_path[:, 0], true_path[:, 1], 'b-', linewidth=1, alpha=0.6,
            label='True path')
    ax.plot(est_path[:, 0], est_path[:, 1], 'r--', linewidth=1, alpha=0.6,
            label='Estimated path')
    plot_slam_state(ax, slam, true_landmarks, x_true)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("EKF-SLAM: Map and Trajectory")
    ax.legend(fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Landmark estimation errors
    ax2 = axes[1]
    lm_errors = []
    lm_labels = []
    for ext_id, idx in slam.landmark_ids.items():
        est_lm = landmark_state(slam.x, idx)
        true_lm = true_landmarks[ext_id]
        err = np.linalg.norm(est_lm - true_lm)
        lm_errors.append(err)
        lm_labels.append(f"LM {ext_id}")

    if lm_errors:
        bars = ax2.bar(range(len(lm_errors)), lm_errors, color='steelblue')
        ax2.set_xticks(range(len(lm_errors)))
        ax2.set_xticklabels(lm_labels, rotation=45, fontsize=8)
        ax2.set_ylabel("Position Error (m)")
        ax2.set_title("Landmark Estimation Errors")
        ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle("EKF-SLAM: Simultaneous Localization and Mapping", fontsize=14)
    plt.tight_layout()
    plt.savefig("10_slam.png", dpi=120)
    plt.show()

    # Print summary
    print(f"\nLandmarks discovered: {n_landmarks(slam.x)} / {n_lm}")
    print(f"State dimension: {len(slam.x)}")
    if lm_errors:
        print(f"Mean landmark error: {np.mean(lm_errors):.3f} m")
        print(f"Max landmark error: {np.max(lm_errors):.3f} m")
    pos_err = np.linalg.norm(slam.x[:2] - x_true[:2])
    print(f"Final robot position error: {pos_err:.3f} m")


if __name__ == "__main__":
    demo_ekf_slam()
