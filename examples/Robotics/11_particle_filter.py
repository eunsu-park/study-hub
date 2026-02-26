"""
Particle Filter (Monte Carlo Localization) for Robot Localization
=================================================================
Estimate robot pose using a set of weighted particles.

Unlike the EKF (which maintains a single Gaussian), a particle filter
represents the belief distribution using a set of samples (particles).
Each particle is a hypothesis about the robot's state, and its weight
reflects how well it agrees with the measurements.

Why particle filters?
  - Can represent multimodal distributions (e.g., the robot could be in
    two hallways that look identical)
  - Handle highly nonlinear models without linearization
  - Simple to implement and understand

The particle filter cycle:
  1. Predict: move each particle according to the motion model + noise
  2. Weight: assign each particle a weight based on measurement likelihood
  3. Resample: replace unlikely particles with copies of likely ones

This is also known as Monte Carlo Localization (MCL) when applied to
robot localization with a known map.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


# ---------------------------------------------------------------------------
# Particle filter
# ---------------------------------------------------------------------------
class ParticleFilter:
    """Particle filter for 2D mobile robot localization.

    State: [x, y, theta] for each particle
    Each particle represents one hypothesis of where the robot might be.
    The collection of particles approximates the posterior distribution
    p(x_t | z_{1:t}, u_{1:t}).
    """

    def __init__(self, n_particles: int, x_range: tuple, y_range: tuple,
                 landmarks: np.ndarray):
        """Initialize particles uniformly across the space.

        Starting with a uniform distribution represents "global uncertainty" —
        the robot has no idea where it is. As measurements arrive, particles
        near the true pose get high weights and survive resampling, while
        others die off. This is how the filter converges.
        """
        self.n_particles = n_particles
        self.landmarks = landmarks

        # Initialize particles uniformly (global localization)
        self.particles = np.zeros((n_particles, 3))
        self.particles[:, 0] = np.random.uniform(x_range[0], x_range[1], n_particles)
        self.particles[:, 1] = np.random.uniform(y_range[0], y_range[1], n_particles)
        self.particles[:, 2] = np.random.uniform(-np.pi, np.pi, n_particles)

        # Equal initial weights (no preference for any particle)
        self.weights = np.ones(n_particles) / n_particles

    def predict(self, u: np.ndarray, dt: float,
                sigma_v: float = 0.3, sigma_omega: float = 0.1):
        """Propagate each particle through the motion model with added noise.

        We add noise to each particle INDEPENDENTLY. This is crucial:
        the noise "spreads out" the particles to cover the range of possible
        true states. Without noise, all particles would follow the same
        trajectory and the filter would be overconfident.

        The noise magnitude should match the true process noise. Too little
        noise → particle depletion (all particles cluster and miss the true state).
        Too much noise → slow convergence.
        """
        v, omega = u
        for i in range(self.n_particles):
            # Sample noisy control for this particle
            v_noisy = v + np.random.normal(0, sigma_v)
            omega_noisy = omega + np.random.normal(0, sigma_omega)

            theta = self.particles[i, 2]
            self.particles[i, 0] += v_noisy * np.cos(theta) * dt
            self.particles[i, 1] += v_noisy * np.sin(theta) * dt
            self.particles[i, 2] += omega_noisy * dt
            # Normalize theta
            self.particles[i, 2] = (self.particles[i, 2] + np.pi) % (2 * np.pi) - np.pi

    def update(self, measurements: list, sigma_r: float = 0.5,
               sigma_b: float = 0.15):
        """Update particle weights based on range-bearing measurements.

        For each particle, we compute the likelihood of the actual
        measurements given that particle's state. Particles that predict
        measurements close to the actual ones get high weights.

        We multiply likelihoods across all measurements (assuming independence).
        This is where the magic happens: even if a single measurement is
        ambiguous, combining multiple measurements rapidly narrows down
        the possible states.

        Args:
            measurements: List of (range, bearing, landmark_index) tuples
            sigma_r: Range measurement noise standard deviation
            sigma_b: Bearing measurement noise standard deviation
        """
        for i in range(self.n_particles):
            log_weight = 0.0

            for r_meas, b_meas, lm_idx in measurements:
                lm = self.landmarks[lm_idx]
                dx = lm[0] - self.particles[i, 0]
                dy = lm[1] - self.particles[i, 1]
                r_pred = np.sqrt(dx**2 + dy**2)
                b_pred = np.arctan2(dy, dx) - self.particles[i, 2]

                # Range likelihood (Gaussian)
                r_diff = r_meas - r_pred
                log_weight += -0.5 * (r_diff / sigma_r) ** 2

                # Bearing likelihood (Gaussian, wrapped)
                b_diff = b_meas - b_pred
                b_diff = (b_diff + np.pi) % (2 * np.pi) - np.pi
                log_weight += -0.5 * (b_diff / sigma_b) ** 2

            # Use log-weights for numerical stability, then exponentiate
            self.weights[i] = np.exp(log_weight)

        # Normalize weights to sum to 1
        total = np.sum(self.weights)
        if total > 1e-300:
            self.weights /= total
        else:
            # All weights collapsed → uniform (reset)
            self.weights = np.ones(self.n_particles) / self.n_particles

    def resample(self):
        """Low-variance resampling (systematic resampling).

        Standard multinomial resampling randomly picks particles proportional
        to their weights, but this has high variance. Low-variance resampling
        uses a single random number and evenly-spaced pointers, which:
          - Guarantees particles with weight >= 1/N survive at least once
          - Produces less variance in the number of copies
          - Is more computationally efficient (O(N) vs O(N log N))

        This is the standard resampling method in modern particle filters.
        """
        N = self.n_particles
        new_particles = np.zeros_like(self.particles)

        # Cumulative sum of weights
        cumsum = np.cumsum(self.weights)

        # Single random starting point + uniform spacing
        r = np.random.uniform(0, 1.0 / N)
        idx = 0

        for i in range(N):
            u = r + i / N
            while u > cumsum[idx]:
                idx += 1
            new_particles[i] = self.particles[idx]

        self.particles = new_particles
        self.weights = np.ones(N) / N  # Reset weights after resampling

    def estimate(self) -> np.ndarray:
        """Compute the weighted mean as the state estimate.

        For orientation (theta), we use circular mean to handle wrapping.
        Simple averaging would give wrong results near +/- pi.
        """
        x_mean = np.average(self.particles[:, 0], weights=self.weights)
        y_mean = np.average(self.particles[:, 1], weights=self.weights)

        # Circular mean for angle
        sin_mean = np.average(np.sin(self.particles[:, 2]), weights=self.weights)
        cos_mean = np.average(np.cos(self.particles[:, 2]), weights=self.weights)
        theta_mean = np.arctan2(sin_mean, cos_mean)

        return np.array([x_mean, y_mean, theta_mean])

    def effective_particles(self) -> float:
        """Compute the effective sample size (ESS).

        ESS = 1 / sum(w_i^2)

        ESS ranges from 1 (one particle has all the weight — degenerate)
        to N (all particles have equal weight — fully diverse).
        When ESS drops below N/2, resampling is typically triggered.
        """
        return 1.0 / np.sum(self.weights ** 2)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def demo_particle_filter():
    """Demonstrate Monte Carlo Localization with a particle filter."""
    print("=" * 60)
    print("Particle Filter (Monte Carlo Localization) Demo")
    print("=" * 60)

    np.random.seed(42)

    # Environment
    world_size = (0, 20)
    landmarks = np.array([
        [5, 5], [15, 5], [10, 10], [5, 15], [15, 15],
        [0, 10], [20, 10], [10, 0], [10, 20],
    ])

    # Particle filter initialization
    n_particles = 500
    pf = ParticleFilter(n_particles, world_size, world_size, landmarks)

    # Simulation
    dt = 0.1
    n_steps = 300
    max_range = 8.0

    # Noise parameters
    sigma_r = 0.5
    sigma_b = 0.15

    # True robot state — start at center, facing right
    x_true = np.array([10.0, 10.0, 0.0])

    true_path = [x_true.copy()]
    est_path = [pf.estimate()]
    ess_history = [pf.effective_particles()]

    # Snapshot steps for visualization
    snapshot_steps = [0, 20, 50, 150]
    snapshots = {}

    for step in range(n_steps):
        t = step * dt

        # Control: figure-8 trajectory
        v = 1.2
        omega = 0.5 * np.sin(0.15 * t)
        u = np.array([v, omega])

        # True motion with noise
        v_true = v + np.random.normal(0, 0.2)
        omega_true = omega + np.random.normal(0, 0.05)
        x_true += np.array([
            v_true * np.cos(x_true[2]) * dt,
            v_true * np.sin(x_true[2]) * dt,
            omega_true * dt
        ])
        x_true[2] = (x_true[2] + np.pi) % (2 * np.pi) - np.pi

        # Predict
        pf.predict(u, dt, sigma_v=0.3, sigma_omega=0.1)

        # Generate measurements
        measurements = []
        for lm_idx, lm in enumerate(landmarks):
            dx = lm[0] - x_true[0]
            dy = lm[1] - x_true[1]
            r = np.sqrt(dx**2 + dy**2)
            if r <= max_range:
                b = np.arctan2(dy, dx) - x_true[2]
                r_noisy = r + np.random.normal(0, sigma_r)
                b_noisy = b + np.random.normal(0, sigma_b)
                measurements.append((r_noisy, b_noisy, lm_idx))

        # Update weights
        if measurements:
            pf.update(measurements, sigma_r, sigma_b)

        # Resample when ESS drops too low
        ess = pf.effective_particles()
        if ess < n_particles / 2:
            pf.resample()

        # Save snapshots
        if step in snapshot_steps:
            snapshots[step] = pf.particles.copy()

        # Record
        true_path.append(x_true.copy())
        est_path.append(pf.estimate())
        ess_history.append(pf.effective_particles())

    true_path = np.array(true_path)
    est_path = np.array(est_path)

    # --- Visualization ---
    # 1. Particle convergence snapshots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, step in zip(axes.flat, snapshot_steps):
        particles = snapshots[step]
        ax.scatter(particles[:, 0], particles[:, 1], s=1, alpha=0.5,
                   color='blue', label='Particles')
        ax.plot(landmarks[:, 0], landmarks[:, 1], 'k^', markersize=10,
                label='Landmarks')

        # True position at this step
        true_x = true_path[step + 1]
        ax.plot(true_x[0], true_x[1], 'r*', markersize=15, label='True pose')

        ax.set_xlim(world_size)
        ax.set_ylim(world_size)
        ax.set_aspect('equal')
        ax.set_title(f"Step {step}: {n_particles} particles")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    plt.suptitle("Particle Filter Convergence", fontsize=14)
    plt.tight_layout()
    plt.savefig("11_particle_filter_convergence.png", dpi=120)
    plt.show()

    # 2. Trajectory and error
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    ax.plot(true_path[:, 0], true_path[:, 1], 'b-', linewidth=1.5,
            label='True path', alpha=0.8)
    ax.plot(est_path[:, 0], est_path[:, 1], 'r--', linewidth=1.5,
            label='PF estimate', alpha=0.8)
    ax.plot(landmarks[:, 0], landmarks[:, 1], 'k^', markersize=10,
            label='Landmarks')

    # Show final particle cloud
    ax.scatter(pf.particles[:, 0], pf.particles[:, 1], s=1, alpha=0.3,
               color='green', label='Final particles')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Particle Filter Localization")
    ax.legend(fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Error and ESS
    ax2 = axes[1]
    pos_error = np.sqrt((true_path[:, 0] - est_path[:, 0])**2
                         + (true_path[:, 1] - est_path[:, 1])**2)
    t_arr = np.arange(len(pos_error)) * dt

    ax2.plot(t_arr, pos_error, 'b-', label='Position error (m)')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(t_arr, ess_history, 'r-', alpha=0.5, label='ESS')
    ax2_twin.axhline(y=n_particles / 2, color='r', linestyle='--', alpha=0.3,
                      label='Resample threshold')
    ax2_twin.set_ylabel("Effective Sample Size", color='red')

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Position Error (m)")
    ax2.set_title("Estimation Error and Effective Sample Size")
    ax2.legend(loc='upper left', fontsize=8)
    ax2_twin.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Monte Carlo Localization (Particle Filter)", fontsize=14)
    plt.tight_layout()
    plt.savefig("11_particle_filter_trajectory.png", dpi=120)
    plt.show()

    # Summary
    print(f"\nFinal position error: {pos_error[-1]:.3f} m")
    print(f"Average position error (after step 50): {np.mean(pos_error[50:]):.3f} m")
    print(f"Final ESS: {ess_history[-1]:.0f} / {n_particles}")


if __name__ == "__main__":
    demo_particle_filter()
