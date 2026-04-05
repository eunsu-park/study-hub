"""
Extended Kalman Filter (EKF) for Robot Localization
====================================================
Estimate a mobile robot's pose using noisy odometry and landmark observations.

The Kalman filter is the optimal state estimator for linear Gaussian systems.
Real robots have nonlinear dynamics (e.g., the bicycle model), so we use the
Extended Kalman Filter (EKF), which linearizes around the current estimate.

State: x = [x, y, theta]^T  (robot position and heading)
Control: u = [v, omega]^T    (linear and angular velocity)
Measurements: z_i = [range, bearing]^T to known landmarks

The EKF alternates between:
  1. Predict: propagate state using motion model + increase uncertainty
  2. Update: correct state using measurements + decrease uncertainty

The uncertainty is represented by a covariance matrix P, visualized as
a 2D confidence ellipse.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


# ---------------------------------------------------------------------------
# Motion model
# ---------------------------------------------------------------------------
def motion_model(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """Nonlinear motion model for a differential-drive robot.

    x_new = f(x, u) where:
        x_{t+1} = x_t + v * cos(theta) * dt
        y_{t+1} = y_t + v * sin(theta) * dt
        theta_{t+1} = theta_t + omega * dt

    This is the constant-velocity model — the simplest useful model for
    wheeled robots. More sophisticated models include the bicycle model
    (for cars) or Ackermann steering.
    """
    theta = x[2]
    v, omega = u

    return x + np.array([
        v * np.cos(theta) * dt,
        v * np.sin(theta) * dt,
        omega * dt
    ])


def motion_jacobian(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """Jacobian of the motion model with respect to state x.

    F = ∂f/∂x

    Linearization is the key approximation in EKF. We compute the Jacobian
    at the current state estimate, which is exact at that point but introduces
    error elsewhere. This is why EKF can diverge for highly nonlinear systems.
    """
    theta = x[2]
    v = u[0]

    return np.array([
        [1, 0, -v * np.sin(theta) * dt],
        [0, 1,  v * np.cos(theta) * dt],
        [0, 0,  1]
    ])


# ---------------------------------------------------------------------------
# Measurement model
# ---------------------------------------------------------------------------
def measurement_model(x: np.ndarray, landmark: np.ndarray) -> np.ndarray:
    """Nonlinear measurement model: range and bearing to a landmark.

    z = h(x, landmark) where:
        range = sqrt((lx - x)^2 + (ly - y)^2)
        bearing = atan2(ly - y, lx - x) - theta

    Range-bearing sensors are common in robotics (LiDAR, sonar, cameras).
    The bearing is measured relative to the robot's heading.
    """
    dx = landmark[0] - x[0]
    dy = landmark[1] - x[1]

    r = np.sqrt(dx**2 + dy**2)
    bearing = np.arctan2(dy, dx) - x[2]
    # Normalize bearing to [-pi, pi]
    bearing = (bearing + np.pi) % (2 * np.pi) - np.pi

    return np.array([r, bearing])


def measurement_jacobian(x: np.ndarray, landmark: np.ndarray) -> np.ndarray:
    """Jacobian of measurement model with respect to state x.

    H = ∂h/∂x

    This matrix tells us how sensitive each measurement component is to
    changes in the robot state. Landmarks far away have small Jacobians
    (less informative), while nearby landmarks have large Jacobians.
    """
    dx = landmark[0] - x[0]
    dy = landmark[1] - x[1]
    r_sq = dx**2 + dy**2
    r = np.sqrt(r_sq)

    return np.array([
        [-dx / r,     -dy / r,      0],
        [dy / r_sq,  -dx / r_sq,   -1]
    ])


# ---------------------------------------------------------------------------
# EKF class
# ---------------------------------------------------------------------------
class EKF:
    """Extended Kalman Filter for robot localization.

    The EKF maintains a Gaussian belief over the robot state:
        bel(x) = N(x_hat, P)
    where x_hat is the mean (best estimate) and P is the covariance (uncertainty).

    Why EKF for localization?
      - Computationally efficient (matrix operations, not sampling)
      - Provides uncertainty estimates (covariance → confidence ellipses)
      - Well-understood theory with decades of practical experience
      - Good for approximately linear/Gaussian systems

    Limitations:
      - Cannot handle multimodal distributions (use particle filter instead)
      - Linearization can cause divergence for highly nonlinear systems
      - Assumes Gaussian noise (may not hold for real sensors)
    """

    def __init__(self, x0: np.ndarray, P0: np.ndarray,
                 Q: np.ndarray, R: np.ndarray):
        """
        Args:
            x0: Initial state estimate [x, y, theta]
            P0: Initial covariance matrix (3x3)
            Q: Process noise covariance (3x3) — how much we distrust the motion model
            R: Measurement noise covariance (2x2) — how much we distrust sensors
        """
        self.x = x0.copy()
        self.P = P0.copy()
        self.Q = Q
        self.R = R

    def predict(self, u: np.ndarray, dt: float):
        """EKF prediction step.

        1. Propagate state through nonlinear motion model
        2. Propagate covariance through linearized model

        The covariance grows during prediction (uncertainty increases when
        we move without measuring). This captures our decreasing confidence
        in the state estimate over time without corrections.
        """
        F = motion_jacobian(self.x, u, dt)
        self.x = motion_model(self.x, u, dt)
        # Wrap theta to [-pi, pi]
        self.x[2] = (self.x[2] + np.pi) % (2 * np.pi) - np.pi
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z: np.ndarray, landmark: np.ndarray):
        """EKF update (correction) step.

        1. Compute expected measurement from current estimate
        2. Compute innovation (measurement residual)
        3. Compute Kalman gain
        4. Correct state and covariance

        The Kalman gain K determines how much we trust the measurement vs
        the prediction. When sensor noise R is small, K is large (trust sensor).
        When prediction uncertainty P is small, K is small (trust model).
        """
        # Expected measurement
        z_pred = measurement_model(self.x, landmark)
        # Innovation (residual)
        innovation = z - z_pred
        # Normalize bearing innovation
        innovation[1] = (innovation[1] + np.pi) % (2 * np.pi) - np.pi

        # Measurement Jacobian
        H = measurement_jacobian(self.x, landmark)

        # Innovation covariance
        S = H @ self.P @ H.T + self.R

        # Kalman gain: K = P H^T S^{-1}
        # This is the optimal weighting between prediction and measurement
        K = self.P @ H.T @ np.linalg.inv(S)

        # State correction
        self.x = self.x + K @ innovation
        self.x[2] = (self.x[2] + np.pi) % (2 * np.pi) - np.pi

        # Covariance correction (Joseph form for numerical stability)
        I_KH = np.eye(3) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------
def plot_covariance_ellipse(ax, mean, cov, n_std=2.0, **kwargs):
    """Draw a 2D confidence ellipse from a 2x2 covariance matrix.

    The ellipse represents the n_std-sigma confidence region.
    2-sigma covers ~95% of the probability mass for a Gaussian.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(cov[:2, :2])
    # Ensure positive eigenvalues (numerical safety)
    eigenvalues = np.maximum(eigenvalues, 0)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width = 2 * n_std * np.sqrt(eigenvalues[0])
    height = 2 * n_std * np.sqrt(eigenvalues[1])

    ellipse = Ellipse(xy=mean[:2], width=width, height=height,
                       angle=angle, **kwargs)
    ax.add_patch(ellipse)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def demo_ekf_localization():
    """Run an EKF localization simulation with a mobile robot."""
    print("=" * 60)
    print("Extended Kalman Filter Localization Demo")
    print("=" * 60)

    # Simulation parameters
    dt = 0.1
    n_steps = 300
    np.random.seed(42)

    # Known landmark positions (the robot observes these)
    landmarks = np.array([
        [5.0, 10.0],
        [10.0, 5.0],
        [15.0, 15.0],
        [0.0, 15.0],
        [10.0, 0.0],
    ])

    # Noise parameters
    # Process noise: uncertainty in our motion model
    sigma_v = 0.3      # Linear velocity noise (m/s)
    sigma_omega = 0.1   # Angular velocity noise (rad/s)
    Q = np.diag([0.1**2, 0.1**2, 0.02**2])

    # Measurement noise: sensor imprecision
    sigma_range = 0.5    # Range noise (m)
    sigma_bearing = 0.1  # Bearing noise (rad)
    R = np.diag([sigma_range**2, sigma_bearing**2])

    # Maximum sensor range
    max_range = 10.0

    # True initial state
    x_true = np.array([0.0, 0.0, 0.0])

    # EKF initial state (slightly off from true state to simulate uncertainty)
    x_est = np.array([0.5, -0.3, 0.05])
    P0 = np.diag([1.0, 1.0, 0.1])
    ekf = EKF(x_est, P0, Q, R)

    # Storage for plotting
    true_path = [x_true.copy()]
    est_path = [ekf.x.copy()]
    cov_history = [ekf.P.copy()]

    # Simulate robot moving in a figure-8 pattern
    for step in range(n_steps):
        t = step * dt

        # Control input: figure-8 trajectory
        v = 1.0  # Constant forward velocity
        omega = 0.5 * np.sin(0.2 * t)  # Varying angular velocity

        u = np.array([v, omega])

        # --- True robot motion (with noise) ---
        v_noisy = v + np.random.normal(0, sigma_v)
        omega_noisy = omega + np.random.normal(0, sigma_omega)
        u_noisy = np.array([v_noisy, omega_noisy])
        x_true = motion_model(x_true, u_noisy, dt)

        # --- EKF predict (using commanded control, not noisy) ---
        ekf.predict(u, dt)

        # --- Generate and process measurements ---
        for lm in landmarks:
            z_true = measurement_model(x_true, lm)

            # Only observe landmarks within sensor range
            if z_true[0] > max_range:
                continue

            # Add sensor noise
            z_noisy = z_true + np.array([
                np.random.normal(0, sigma_range),
                np.random.normal(0, sigma_bearing)
            ])

            # EKF update
            ekf.update(z_noisy, lm)

        # Store results
        true_path.append(x_true.copy())
        est_path.append(ekf.x.copy())
        cov_history.append(ekf.P.copy())

    true_path = np.array(true_path)
    est_path = np.array(est_path)

    # --- Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Main trajectory plot
    ax = axes[0]
    ax.plot(true_path[:, 0], true_path[:, 1], 'b-', linewidth=1.5,
            label='True path', alpha=0.8)
    ax.plot(est_path[:, 0], est_path[:, 1], 'r--', linewidth=1.5,
            label='EKF estimate', alpha=0.8)

    # Draw covariance ellipses every 20 steps
    for i in range(0, len(cov_history), 20):
        plot_covariance_ellipse(ax, est_path[i], cov_history[i],
                                 n_std=2.0, fill=False,
                                 edgecolor='red', linewidth=0.8, alpha=0.5)

    # Draw landmarks
    ax.plot(landmarks[:, 0], landmarks[:, 1], 'k^', markersize=12,
            label='Landmarks')

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("EKF Robot Localization")
    ax.legend(fontsize=9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Error over time
    ax2 = axes[1]
    position_error = np.sqrt((true_path[:, 0] - est_path[:, 0])**2
                              + (true_path[:, 1] - est_path[:, 1])**2)
    heading_error = np.abs(true_path[:, 2] - est_path[:, 2])
    heading_error = np.minimum(heading_error, 2 * np.pi - heading_error)

    t_arr = np.arange(len(position_error)) * dt
    ax2.plot(t_arr, position_error, 'b-', label='Position error (m)')
    ax2.plot(t_arr, np.degrees(heading_error), 'r-', label='Heading error (deg)')

    # Plot 2-sigma bound from covariance
    sigma_pos = [np.sqrt(cov_history[i][0, 0] + cov_history[i][1, 1])
                  for i in range(len(cov_history))]
    ax2.plot(t_arr, [2 * s for s in sigma_pos], 'b--', alpha=0.5,
             label='2-sigma position bound')

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Error")
    ax2.set_title("Estimation Error Over Time")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Extended Kalman Filter for Mobile Robot Localization", fontsize=14)
    plt.tight_layout()
    plt.savefig("09_kalman_filter.png", dpi=120)
    plt.show()

    # Print summary
    print(f"\nFinal position error: {position_error[-1]:.3f} m")
    print(f"Final heading error: {np.degrees(heading_error[-1]):.3f} deg")
    print(f"Average position error: {np.mean(position_error):.3f} m")


if __name__ == "__main__":
    demo_ekf_localization()
