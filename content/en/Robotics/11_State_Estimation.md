# 11. State Estimation and Filtering

[← Previous: Sensors and Perception](10_Sensors_and_Perception.md) | [Next: SLAM →](12_SLAM.md)

---

## Learning Objectives

1. Formulate the state estimation problem as a probabilistic inference task
2. Derive the Kalman filter predict-update cycle and implement it for linear systems
3. Extend the Kalman filter to nonlinear systems using the Extended Kalman Filter (EKF)
4. Understand the Unscented Kalman Filter (UKF) as a derivative-free alternative to EKF
5. Implement a particle filter for non-Gaussian, highly nonlinear estimation problems
6. Compare KF, EKF, UKF, and particle filter trade-offs for robot localization

---

In the previous lesson, we surveyed the sensors that give a robot information about itself and its environment. But raw sensor data is noisy, incomplete, and arrives at different rates from different sources. A single encoder reading tells us where a joint was a millisecond ago, but what we really need is the best estimate of where the robot is *right now*, combining everything we know — the physics of how the robot moves, the latest sensor readings, and our uncertainty about both.

State estimation is the mathematical framework for answering this question optimally. It is the backbone of virtually every autonomous system: self-driving cars, drones, mobile robots, and spacecraft all rely on state estimators to maintain an accurate picture of their state in real time. Without state estimation, sensor noise and model uncertainty would make autonomous behavior impossible.

> **Analogy**: A Kalman filter is like a wise advisor — it combines your prediction (experience) with new measurements (evidence) to give the best estimate. Imagine you are navigating a foggy road. Your internal model says "I should be near the bridge by now" (prediction). Then you glimpse a sign through the fog (measurement). The advisor weighs both: "Your prediction was pretty reliable, but the sign is fairly clear too — you are probably 30 meters from the bridge." If the fog is thick (noisy measurement), the advisor trusts your prediction more. If you have been guessing wildly (uncertain model), the advisor trusts the sign more.

---

## 1. The State Estimation Problem

### 1.1 Problem Formulation

We want to estimate a **state vector** $\mathbf{x}_k$ at time $k$ from:

1. **Process model** (how the state evolves):
$$\mathbf{x}_k = f(\mathbf{x}_{k-1}, \mathbf{u}_{k-1}) + \mathbf{w}_{k-1}$$

2. **Measurement model** (how sensors observe the state):
$$\mathbf{z}_k = h(\mathbf{x}_k) + \mathbf{v}_k$$

where:
- $\mathbf{u}_{k-1}$ is the control input
- $\mathbf{w}_{k-1} \sim \mathcal{N}(\mathbf{0}, Q)$ is process noise
- $\mathbf{v}_k \sim \mathcal{N}(\mathbf{0}, R)$ is measurement noise
- $Q$ is the process noise covariance
- $R$ is the measurement noise covariance

### 1.2 Bayesian Estimation Framework

State estimation is fundamentally a problem of **Bayesian inference**. We maintain a probability distribution over the state, called the **belief**:

$$\text{bel}(\mathbf{x}_k) = p(\mathbf{x}_k | \mathbf{z}_{1:k}, \mathbf{u}_{0:k-1})$$

The update follows Bayes' rule in two steps:

**Prediction** (time update):
$$\overline{\text{bel}}(\mathbf{x}_k) = \int p(\mathbf{x}_k | \mathbf{x}_{k-1}, \mathbf{u}_{k-1}) \text{bel}(\mathbf{x}_{k-1}) \, d\mathbf{x}_{k-1}$$

**Correction** (measurement update):
$$\text{bel}(\mathbf{x}_k) = \eta \, p(\mathbf{z}_k | \mathbf{x}_k) \, \overline{\text{bel}}(\mathbf{x}_k)$$

where $\eta$ is a normalizing constant.

Different filters arise from different representations of the belief:

| Filter | Belief Representation | Linearity | Modality |
|--------|----------------------|-----------|----------|
| Kalman Filter | Gaussian ($\mu, \Sigma$) | Linear | Unimodal |
| EKF | Gaussian ($\mu, \Sigma$) | Nonlinear (linearized) | Unimodal |
| UKF | Gaussian ($\mu, \Sigma$) | Nonlinear (sigma points) | Unimodal |
| Particle Filter | Weighted samples | Any | Multimodal |

---

## 2. The Kalman Filter

### 2.1 Linear System Assumption

The Kalman filter assumes **linear** models with **Gaussian** noise:

$$\mathbf{x}_k = A \mathbf{x}_{k-1} + B \mathbf{u}_{k-1} + \mathbf{w}_{k-1}$$
$$\mathbf{z}_k = H \mathbf{x}_k + \mathbf{v}_k$$

where $A$ is the state transition matrix, $B$ is the control input matrix, and $H$ is the observation matrix.

Under these assumptions, the belief remains Gaussian at all times, fully characterized by its mean $\hat{\mathbf{x}}_k$ and covariance $P_k$.

### 2.2 The Predict-Update Cycle

**Prediction step** (propagate state and uncertainty forward):

$$\hat{\mathbf{x}}_k^- = A \hat{\mathbf{x}}_{k-1} + B \mathbf{u}_{k-1}$$
$$P_k^- = A P_{k-1} A^T + Q$$

**Update step** (incorporate measurement):

$$K_k = P_k^- H^T (H P_k^- H^T + R)^{-1}$$
$$\hat{\mathbf{x}}_k = \hat{\mathbf{x}}_k^- + K_k (\mathbf{z}_k - H \hat{\mathbf{x}}_k^-)$$
$$P_k = (I - K_k H) P_k^-$$

where $K_k$ is the **Kalman gain** — the optimal weighting between prediction and measurement.

### 2.3 Understanding the Kalman Gain

The Kalman gain $K_k$ has a beautiful interpretation:

- When $R$ is small (accurate sensor): $K_k \to H^{-1}$ — trust the measurement
- When $P_k^-$ is small (accurate prediction): $K_k \to 0$ — trust the prediction
- When $R$ is large (noisy sensor): $K_k \to 0$ — ignore the measurement
- When $P_k^-$ is large (uncertain prediction): $K_k \to H^{-1}$ — rely on the measurement

The **innovation** (or residual) $\boldsymbol{\nu}_k = \mathbf{z}_k - H \hat{\mathbf{x}}_k^-$ represents the "surprise" — how much the measurement differs from what we predicted. The Kalman gain scales this surprise by the relative trust in the measurement vs. the prediction.

### 2.4 Implementation

```python
import numpy as np

class KalmanFilter:
    """Linear Kalman filter for state estimation.

    Why is the Kalman filter optimal? Under the assumptions of linear
    dynamics, Gaussian noise, and known noise statistics (Q, R), the
    Kalman filter produces the minimum-variance unbiased estimate.
    No other linear estimator can do better.
    """

    def __init__(self, A, B, H, Q, R, x0, P0):
        self.A = np.array(A, dtype=float)  # State transition
        self.B = np.array(B, dtype=float)  # Control input
        self.H = np.array(H, dtype=float)  # Observation
        self.Q = np.array(Q, dtype=float)  # Process noise covariance
        self.R = np.array(R, dtype=float)  # Measurement noise covariance
        self.x = np.array(x0, dtype=float) # State estimate
        self.P = np.array(P0, dtype=float) # Estimate covariance

    def predict(self, u):
        """Prediction step: propagate state and uncertainty.

        Why does covariance grow? The process noise Q adds uncertainty
        at each step. Even if we had a perfect estimate at time k-1,
        we become less certain about x_k because we can't predict the
        process noise perfectly.
        """
        self.x = self.A @ self.x + self.B @ u
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x.copy()

    def update(self, z):
        """Update step: incorporate measurement to refine estimate.

        Why does covariance shrink? A measurement provides new information,
        reducing our uncertainty. The amount of shrinkage depends on
        the Kalman gain: more informative measurements (low R) cause
        bigger reductions in P.
        """
        # Innovation (measurement residual)
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain — optimal weighting between prediction and measurement
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state estimate
        self.x = self.x + K @ y

        # Update covariance (Joseph form for numerical stability)
        I_KH = np.eye(len(self.x)) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        return self.x.copy()
```

### 2.5 Example: 1D Robot Position Tracking

```python
def kf_1d_robot_example():
    """Track a 1D robot with position and velocity state.

    State: [position, velocity]
    Control: acceleration command
    Measurement: position (from GPS or landmark)

    This example demonstrates the core Kalman filter behavior:
    - Covariance grows during prediction (between measurements)
    - Covariance shrinks at each measurement update
    - The filter smoothly interpolates between the motion model and sensors
    """
    dt = 0.1  # 10 Hz

    # State transition: constant acceleration model
    A = np.array([[1, dt],
                   [0,  1]])

    B = np.array([[0.5*dt**2],
                   [dt]])

    # Observation: measure position only
    H = np.array([[1, 0]])

    # Noise covariances
    Q = np.array([[0.01, 0],
                   [0, 0.1]])     # Process noise: uncertain acceleration
    R = np.array([[1.0]])          # Measurement noise: GPS ~1m accuracy

    # Initial state: at origin, stationary, very uncertain
    x0 = np.array([0.0, 0.0])
    P0 = np.array([[10.0, 0],
                    [0, 10.0]])

    kf = KalmanFilter(A, B, H, Q, R, x0, P0)

    # Simulate
    true_pos = 0.0
    true_vel = 0.0
    np.random.seed(42)

    positions_true = []
    positions_meas = []
    positions_est = []
    covariances = []

    for i in range(100):
        u = np.array([0.5])  # Constant acceleration command

        # True dynamics
        true_vel += u[0] * dt + np.random.normal(0, 0.1)
        true_pos += true_vel * dt + np.random.normal(0, 0.01)

        # Prediction
        kf.predict(u)

        # Measurement (every step, with noise)
        z = np.array([true_pos + np.random.normal(0, 1.0)])

        # Update
        kf.update(z)

        positions_true.append(true_pos)
        positions_meas.append(z[0])
        positions_est.append(kf.x[0])
        covariances.append(kf.P[0, 0])

    return positions_true, positions_meas, positions_est, covariances
```

---

## 3. Extended Kalman Filter (EKF)

### 3.1 Handling Nonlinearity

Most robot systems are nonlinear. For example, a differential-drive robot has the process model:

$$\begin{bmatrix} x_{k+1} \\ y_{k+1} \\ \theta_{k+1} \end{bmatrix} = \begin{bmatrix} x_k + v_k \cos\theta_k \cdot \Delta t \\ y_k + v_k \sin\theta_k \cdot \Delta t \\ \theta_k + \omega_k \cdot \Delta t \end{bmatrix}$$

This is clearly nonlinear in $\theta$. The standard Kalman filter requires linear $A$ and $H$ matrices, so we cannot apply it directly.

The **Extended Kalman Filter (EKF)** linearizes the nonlinear functions $f$ and $h$ around the current estimate using their Jacobians:

$$F_k = \frac{\partial f}{\partial \mathbf{x}} \bigg|_{\hat{\mathbf{x}}_{k-1}, \mathbf{u}_{k-1}}$$
$$H_k = \frac{\partial h}{\partial \mathbf{x}} \bigg|_{\hat{\mathbf{x}}_k^-}$$

### 3.2 EKF Algorithm

**Prediction**:
$$\hat{\mathbf{x}}_k^- = f(\hat{\mathbf{x}}_{k-1}, \mathbf{u}_{k-1})$$
$$P_k^- = F_k P_{k-1} F_k^T + Q$$

**Update**:
$$K_k = P_k^- H_k^T (H_k P_k^- H_k^T + R)^{-1}$$
$$\hat{\mathbf{x}}_k = \hat{\mathbf{x}}_k^- + K_k (\mathbf{z}_k - h(\hat{\mathbf{x}}_k^-))$$
$$P_k = (I - K_k H_k) P_k^-$$

Note: The nonlinear functions $f$ and $h$ are used for state propagation and expected measurement computation, but the linearized versions $F_k$ and $H_k$ are used for covariance propagation.

### 3.3 Implementation for Mobile Robot Localization

```python
class EKFLocalization:
    """Extended Kalman Filter for differential-drive robot localization.

    State: [x, y, theta] — robot pose in 2D
    Control: [v, omega] — linear and angular velocity
    Measurements: range-bearing to known landmarks

    Why EKF instead of KF? The robot's motion model involves sin/cos
    of the heading angle theta, making it nonlinear. The EKF handles
    this by linearizing around the current estimate at each step.
    """

    def __init__(self, x0, P0, Q, R):
        self.x = np.array(x0, dtype=float)  # [x, y, theta]
        self.P = np.array(P0, dtype=float)   # 3x3 covariance
        self.Q = np.array(Q, dtype=float)    # Process noise
        self.R = np.array(R, dtype=float)    # Measurement noise (range, bearing)

    def predict(self, u, dt):
        """Predict using differential-drive motion model.

        Why compute the Jacobian analytically? Numerical Jacobians
        (finite differences) introduce approximation error and are slower.
        For simple models like differential drive, the analytical Jacobian
        is straightforward and exact.
        """
        v, omega = u
        theta = self.x[2]

        # Nonlinear state propagation
        if abs(omega) > 1e-6:
            # Arc motion
            self.x[0] += -v/omega * np.sin(theta) + v/omega * np.sin(theta + omega*dt)
            self.x[1] += v/omega * np.cos(theta) - v/omega * np.cos(theta + omega*dt)
            self.x[2] += omega * dt
        else:
            # Straight-line motion (avoid division by zero)
            self.x[0] += v * np.cos(theta) * dt
            self.x[1] += v * np.sin(theta) * dt

        # Normalize angle to [-pi, pi]
        self.x[2] = (self.x[2] + np.pi) % (2*np.pi) - np.pi

        # Jacobian of motion model w.r.t. state
        if abs(omega) > 1e-6:
            F = np.array([
                [1, 0, -v/omega*np.cos(theta) + v/omega*np.cos(theta+omega*dt)],
                [0, 1, -v/omega*np.sin(theta) + v/omega*np.sin(theta+omega*dt)],
                [0, 0, 1]
            ])
        else:
            F = np.array([
                [1, 0, -v * np.sin(theta) * dt],
                [0, 1,  v * np.cos(theta) * dt],
                [0, 0, 1]
            ])

        # Update covariance
        self.P = F @ self.P @ F.T + self.Q

        return self.x.copy()

    def update(self, z, landmark_pos):
        """Update with range-bearing measurement to a known landmark.

        z = [range, bearing] measurement
        landmark_pos = [lx, ly] known landmark position

        Why range-bearing measurements? They are what LiDAR and camera-based
        detectors naturally provide. Range = distance to landmark,
        bearing = angle to landmark relative to the robot's heading.
        """
        lx, ly = landmark_pos
        dx = lx - self.x[0]
        dy = ly - self.x[1]
        q = dx**2 + dy**2
        r = np.sqrt(q)

        # Expected measurement
        z_hat = np.array([
            r,
            np.arctan2(dy, dx) - self.x[2]
        ])
        # Normalize bearing
        z_hat[1] = (z_hat[1] + np.pi) % (2*np.pi) - np.pi

        # Jacobian of measurement model
        H = np.array([
            [-dx/r, -dy/r, 0],
            [dy/q, -dx/q, -1]
        ])

        # Innovation
        innovation = z - z_hat
        innovation[1] = (innovation[1] + np.pi) % (2*np.pi) - np.pi

        # Kalman gain
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update
        self.x = self.x + K @ innovation
        self.x[2] = (self.x[2] + np.pi) % (2*np.pi) - np.pi
        self.P = (np.eye(3) - K @ H) @ self.P

        return self.x.copy()
```

### 3.4 EKF Limitations

The EKF has well-known limitations:

1. **First-order approximation**: The Jacobian linearization is accurate only when the nonlinearity is mild near the current estimate. Highly nonlinear systems (e.g., bearing-only measurements) can cause the EKF to diverge.
2. **Gaussian assumption**: The EKF cannot represent multimodal distributions (e.g., "the robot could be at location A or location B").
3. **Jacobian computation**: Requires analytical or numerical Jacobians, which can be error-prone for complex models.

---

## 4. Unscented Kalman Filter (UKF)

### 4.1 Sigma Points Instead of Jacobians

The **Unscented Kalman Filter** avoids linearization entirely. Instead of computing Jacobians, it propagates a set of carefully chosen **sigma points** through the nonlinear function and recovers the mean and covariance from the transformed points.

The **Unscented Transform** for an $n$-dimensional state:

1. Choose $2n + 1$ sigma points:
$$\boldsymbol{\chi}_0 = \hat{\mathbf{x}}$$
$$\boldsymbol{\chi}_i = \hat{\mathbf{x}} + \left(\sqrt{(n + \lambda) P}\right)_i, \quad i = 1, \ldots, n$$
$$\boldsymbol{\chi}_{n+i} = \hat{\mathbf{x}} - \left(\sqrt{(n + \lambda) P}\right)_i, \quad i = 1, \ldots, n$$

where $\lambda = \alpha^2(n + \kappa) - n$ is a scaling parameter and $\left(\sqrt{(n+\lambda)P}\right)_i$ is the $i$-th column of the matrix square root.

2. Propagate each sigma point through the nonlinear function:
$$\boldsymbol{\gamma}_i = f(\boldsymbol{\chi}_i)$$

3. Recover the mean and covariance from the transformed points:
$$\hat{\mathbf{x}}^- = \sum_{i=0}^{2n} w_i^{(m)} \boldsymbol{\gamma}_i$$
$$P^- = \sum_{i=0}^{2n} w_i^{(c)} (\boldsymbol{\gamma}_i - \hat{\mathbf{x}}^-)(\boldsymbol{\gamma}_i - \hat{\mathbf{x}}^-)^T + Q$$

### 4.2 Why UKF Can Be Better Than EKF

The UKF captures the mean and covariance accurately to **second order** (or higher for Gaussian distributions), while the EKF captures them only to **first order**. This means:

- Better handling of strong nonlinearities
- No need to compute Jacobians (easier implementation, no analytical derivatives)
- Often similar computational cost to EKF (for moderate state dimensions)

```python
class UKFBasic:
    """Basic Unscented Kalman Filter.

    Why use sigma points instead of Jacobians? The EKF approximates
    the nonlinear function as linear (first-order Taylor expansion).
    The UKF approximates the probability distribution by sampling it
    at carefully chosen points, then passing those points through the
    exact nonlinear function. This captures higher-order statistical
    moments and avoids the need for Jacobian computation.
    """

    def __init__(self, n_states, f, h, Q, R, alpha=1e-3, beta=2, kappa=0):
        self.n = n_states
        self.f = f  # Process model function
        self.h = h  # Measurement model function
        self.Q = np.array(Q, dtype=float)
        self.R = np.array(R, dtype=float)

        # UKF parameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lam = alpha**2 * (n_states + kappa) - n_states

        # Weights for mean and covariance
        self.Wm = np.zeros(2 * n_states + 1)
        self.Wc = np.zeros(2 * n_states + 1)
        self.Wm[0] = self.lam / (n_states + self.lam)
        self.Wc[0] = self.lam / (n_states + self.lam) + (1 - alpha**2 + beta)
        for i in range(1, 2 * n_states + 1):
            self.Wm[i] = 1.0 / (2 * (n_states + self.lam))
            self.Wc[i] = 1.0 / (2 * (n_states + self.lam))

    def sigma_points(self, x, P):
        """Generate sigma points around the mean.

        Why 2n+1 points? We need to capture the mean (1 point) plus
        spread in each dimension (2 points per dimension: +/- from mean).
        """
        n = self.n
        sigma_pts = np.zeros((2*n+1, n))
        sigma_pts[0] = x

        sqrt_P = np.linalg.cholesky((n + self.lam) * P)

        for i in range(n):
            sigma_pts[i+1] = x + sqrt_P[i]
            sigma_pts[n+i+1] = x - sqrt_P[i]

        return sigma_pts

    def predict(self, x, P, u, dt):
        """UKF prediction step."""
        # Generate sigma points
        chi = self.sigma_points(x, P)

        # Propagate through process model
        chi_pred = np.array([self.f(chi[i], u, dt) for i in range(2*self.n+1)])

        # Recover mean and covariance
        x_pred = np.sum(self.Wm[:, None] * chi_pred, axis=0)
        P_pred = self.Q.copy()
        for i in range(2*self.n+1):
            diff = chi_pred[i] - x_pred
            P_pred += self.Wc[i] * np.outer(diff, diff)

        return x_pred, P_pred, chi_pred

    def update(self, x_pred, P_pred, chi_pred, z):
        """UKF update step."""
        # Transform sigma points through measurement model
        z_pred = np.array([self.h(chi_pred[i]) for i in range(2*self.n+1)])

        # Mean predicted measurement
        z_mean = np.sum(self.Wm[:, None] * z_pred, axis=0)

        # Innovation covariance
        Pzz = self.R.copy()
        Pxz = np.zeros((self.n, len(z)))
        for i in range(2*self.n+1):
            dz = z_pred[i] - z_mean
            dx = chi_pred[i] - x_pred
            Pzz += self.Wc[i] * np.outer(dz, dz)
            Pxz += self.Wc[i] * np.outer(dx, dz)

        # Kalman gain
        K = Pxz @ np.linalg.inv(Pzz)

        # Update
        x_upd = x_pred + K @ (z - z_mean)
        P_upd = P_pred - K @ Pzz @ K.T

        return x_upd, P_upd
```

---

## 5. Particle Filter

### 5.1 Monte Carlo Localization

For highly nonlinear systems with non-Gaussian noise or multimodal distributions, the **particle filter** (Sequential Monte Carlo) represents the belief using a set of weighted samples (particles):

$$\text{bel}(\mathbf{x}_k) \approx \sum_{i=1}^N w_k^{(i)} \delta(\mathbf{x}_k - \mathbf{x}_k^{(i)})$$

where $\mathbf{x}_k^{(i)}$ is the $i$-th particle and $w_k^{(i)}$ is its weight.

### 5.2 Algorithm

1. **Prediction**: Propagate each particle through the process model with noise:
$$\mathbf{x}_k^{(i)} \sim p(\mathbf{x}_k | \mathbf{x}_{k-1}^{(i)}, \mathbf{u}_{k-1})$$

2. **Update**: Weight each particle by how well it explains the measurement:
$$w_k^{(i)} = p(\mathbf{z}_k | \mathbf{x}_k^{(i)})$$

3. **Normalize** weights: $w_k^{(i)} \leftarrow w_k^{(i)} / \sum_j w_k^{(j)}$

4. **Resample**: Replace low-weight particles with copies of high-weight particles to avoid particle degeneracy.

### 5.3 Implementation

```python
class ParticleFilter:
    """Particle filter for robot localization (Monte Carlo Localization).

    Why particles instead of Gaussians? Particles can represent ANY
    distribution: multimodal (robot could be in two places), skewed,
    heavy-tailed, or wrapped (angles). This generality comes at the
    cost of computational expense — we need hundreds to thousands of
    particles for good estimates.
    """

    def __init__(self, n_particles, state_dim, motion_model, measurement_model,
                 process_noise_std, initial_state=None, initial_spread=None):
        self.N = n_particles
        self.dim = state_dim
        self.motion_model = motion_model
        self.measurement_model = measurement_model
        self.process_noise_std = np.array(process_noise_std)

        # Initialize particles
        if initial_state is not None and initial_spread is not None:
            self.particles = initial_state + initial_spread * np.random.randn(n_particles, state_dim)
        else:
            # Uniform initialization over workspace (global localization)
            self.particles = np.random.uniform(-10, 10, (n_particles, state_dim))

        self.weights = np.ones(n_particles) / n_particles

    def predict(self, u, dt):
        """Propagate particles through motion model with noise.

        Why add noise to each particle? The noise represents our uncertainty
        in the motion model. Without noise, all particles starting at the
        same location would stay together forever, failing to explore the
        state space and making the filter overconfident.
        """
        for i in range(self.N):
            noise = self.process_noise_std * np.random.randn(self.dim)
            self.particles[i] = self.motion_model(self.particles[i], u, dt) + noise

    def update(self, z, landmarks):
        """Update particle weights based on measurement likelihood.

        Why is this step crucial? The prediction step spreads particles
        based on the motion model, but doesn't use sensor information.
        The update step assigns high weights to particles that are
        consistent with the observed measurements and low weights to
        those that are not. Over time, particles accumulate at the
        true robot location.
        """
        for i in range(self.N):
            self.weights[i] = self.measurement_model(self.particles[i], z, landmarks)

        # Normalize weights
        total = np.sum(self.weights)
        if total > 1e-300:
            self.weights /= total
        else:
            # All weights are essentially zero — reinitialize
            self.weights = np.ones(self.N) / self.N

    def resample(self):
        """Systematic resampling to combat particle degeneracy.

        Why resample? After many updates, most particles end up with
        near-zero weight — they are in locations inconsistent with
        observations. Resampling replaces these 'dead' particles with
        copies of high-weight particles, concentrating the computational
        effort where it matters.

        Why systematic resampling? It's O(N), has lower variance than
        multinomial resampling, and maintains better diversity.
        """
        N = self.N
        positions = (np.arange(N) + np.random.uniform()) / N
        cumulative_sum = np.cumsum(self.weights)

        indices = np.zeros(N, dtype=int)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indices[i] = j
                i += 1
            else:
                j += 1

        self.particles = self.particles[indices]
        self.weights = np.ones(N) / N

    def estimate(self):
        """Compute weighted mean and covariance.

        Why not just take the highest-weight particle? The weighted mean
        is a better estimate because it averages over the posterior
        distribution. The highest-weight particle is just one sample
        and may be noisy.
        """
        mean = np.average(self.particles, weights=self.weights, axis=0)
        diff = self.particles - mean
        cov = np.average(diff[:, :, None] * diff[:, None, :],
                         weights=self.weights, axis=0)
        return mean, cov

    def effective_particles(self):
        """Compute effective sample size.

        Why monitor this? When N_eff << N, most particles have negligible
        weight — the filter is degenerating. Resampling when N_eff < N/2
        is a common heuristic.
        """
        return 1.0 / np.sum(self.weights**2)
```

### 5.4 Complete Localization Example

```python
def particle_filter_localization_demo():
    """Demonstrate particle filter localization with known landmarks.

    Scenario: A differential-drive robot moves through a 2D environment
    with known landmarks. It measures range and bearing to nearby
    landmarks. The particle filter tracks its pose.
    """
    # Known landmark positions
    landmarks = np.array([
        [5.0, 5.0], [5.0, -5.0], [-5.0, 5.0], [-5.0, -5.0],
        [0.0, 8.0], [8.0, 0.0]
    ])

    def motion_model(state, u, dt):
        """Differential-drive motion model."""
        x, y, theta = state
        v, omega = u
        if abs(omega) > 1e-6:
            x += -v/omega*np.sin(theta) + v/omega*np.sin(theta+omega*dt)
            y += v/omega*np.cos(theta) - v/omega*np.cos(theta+omega*dt)
        else:
            x += v*np.cos(theta)*dt
            y += v*np.sin(theta)*dt
        theta += omega*dt
        return np.array([x, y, theta])

    def measurement_likelihood(state, z, landmarks):
        """Compute likelihood of measurements given state.

        Uses a Gaussian sensor model: the probability of observing
        measurement z given that the robot is at 'state'.
        """
        x, y, theta = state
        likelihood = 1.0
        range_std = 0.5
        bearing_std = 0.1

        for j, (lx, ly) in enumerate(landmarks):
            dx = lx - x
            dy = ly - y
            expected_range = np.sqrt(dx**2 + dy**2)
            expected_bearing = np.arctan2(dy, dx) - theta

            if j < len(z):
                z_range, z_bearing = z[j]
                range_err = z_range - expected_range
                bearing_err = z_bearing - expected_bearing
                bearing_err = (bearing_err + np.pi) % (2*np.pi) - np.pi

                likelihood *= np.exp(-0.5 * (range_err/range_std)**2)
                likelihood *= np.exp(-0.5 * (bearing_err/bearing_std)**2)

        return max(likelihood, 1e-300)

    # Create particle filter
    pf = ParticleFilter(
        n_particles=500,
        state_dim=3,
        motion_model=motion_model,
        measurement_model=measurement_likelihood,
        process_noise_std=[0.1, 0.1, 0.02],
        initial_state=np.array([0.0, 0.0, 0.0]),
        initial_spread=np.array([2.0, 2.0, 0.5])
    )

    return pf  # Ready for step-by-step simulation
```

---

## 6. Filter Comparison

### 6.1 Trade-offs

| Filter | Computation | Accuracy (Linear) | Accuracy (Nonlinear) | Multimodal | Implementation |
|--------|-------------|-------------------|---------------------|------------|----------------|
| KF | $O(n^3)$ | Optimal | N/A (linear only) | No | Simple |
| EKF | $O(n^3)$ | Near-optimal | Good (mild nonlinearity) | No | Moderate (Jacobians) |
| UKF | $O(n^3)$ | Near-optimal | Better than EKF | No | Moderate |
| PF | $O(Nn^2)$ | Converges to optimal | Excellent | Yes | Simple concept, tuning |

where $n$ is the state dimension and $N$ is the number of particles.

### 6.2 When to Use Which

```
Is the system linear?
├── Yes → Kalman Filter (optimal, simplest)
└── No → Is the distribution unimodal?
    ├── Yes → Is the nonlinearity mild?
    │   ├── Yes → EKF (widely used, fast)
    │   └── No → UKF (better accuracy, no Jacobians)
    └── No (multimodal) → Particle Filter (only option)
```

### 6.3 Practical Considerations

**EKF is the workhorse** of robotics: most SLAM systems, drone state estimators, and industrial localization systems use EKF or its variants. It is fast, well-understood, and sufficient for most applications.

**Particle filters** are essential for **global localization** (the "kidnapped robot" problem, where the robot does not know its initial position) because they can maintain multiple hypotheses simultaneously.

**UKF** is gaining popularity because it avoids error-prone Jacobian derivations. Modern robotics libraries (e.g., `filterpy`, `ukfm`) make UKF implementation straightforward.

**Sensor fusion** frameworks often run multiple filters or combine them:
- EKF for fast, continuous state estimation (IMU + odometry at 100+ Hz)
- Particle filter for global localization and recovery from tracking failure
- Factor graph optimization for offline smoothing (covered in the SLAM lesson)

---

## 7. Multi-Rate Sensor Fusion

### 7.1 The Asynchronous Measurement Problem

In practice, sensors arrive at different rates:

| Sensor | Rate | Latency |
|--------|------|---------|
| IMU | 200-1000 Hz | < 1 ms |
| Wheel encoders | 50-100 Hz | < 5 ms |
| LiDAR | 10-20 Hz | 50-100 ms |
| Camera | 15-60 Hz | 30-100 ms |
| GPS | 1-10 Hz | 100-500 ms |

The solution is to run **prediction** at the highest sensor rate (IMU) and **update** whenever a new measurement arrives from any sensor:

```python
def multi_rate_fusion_loop(ekf, imu_queue, lidar_queue, gps_queue):
    """Multi-rate sensor fusion using EKF.

    Why predict at IMU rate? The IMU runs at 200+ Hz and provides
    continuous motion information. Between slower measurements (LiDAR
    at 10 Hz, GPS at 1 Hz), we use IMU-driven prediction to maintain
    a smooth, high-rate state estimate. When a slower sensor arrives,
    we incorporate it via an update step.
    """
    while True:
        # Always process IMU first (highest rate)
        if imu_queue.has_data():
            imu_data = imu_queue.get()
            ekf.predict(imu_data.accel, imu_data.gyro, imu_data.dt)

        # Process LiDAR when available
        if lidar_queue.has_data():
            lidar_data = lidar_queue.get()
            ekf.update_lidar(lidar_data.ranges, lidar_data.angles)

        # Process GPS when available
        if gps_queue.has_data():
            gps_data = gps_queue.get()
            ekf.update_gps(gps_data.lat, gps_data.lon)

        # Output current state estimate at fixed rate
        yield ekf.get_state()
```

---

## Summary

| Concept | Key Idea |
|---------|----------|
| State estimation | Infer hidden state from noisy observations using probabilistic models |
| Bayes filter | Recursive predict-update framework; different filters = different belief representations |
| Kalman filter | Optimal for linear Gaussian systems; tracks mean and covariance |
| Kalman gain | Optimal blend between prediction and measurement based on their uncertainties |
| EKF | First-order linearization of nonlinear models; most widely used in robotics |
| UKF | Sigma-point propagation avoids Jacobians; better for strong nonlinearities |
| Particle filter | Weighted sample representation; handles any distribution and nonlinearity |
| Resampling | Combats particle degeneracy by duplicating high-weight particles |
| Multi-rate fusion | Predict at highest sensor rate; update asynchronously per sensor |

---

## Exercises

1. **Kalman filter implementation**: Implement the 1D Kalman filter for tracking a robot moving with constant velocity. The state is $[x, \dot{x}]$. Generate a ground-truth trajectory with random accelerations, simulate noisy position measurements (GPS-like, 1 Hz, $\sigma = 2$ m), and run the KF. Plot the true trajectory, measurements, and KF estimate with $\pm 2\sigma$ confidence bounds.

2. **EKF for differential-drive robot**: Implement the EKF localization class above. Simulate a robot driving in a circle in an environment with 6 known landmarks. Generate range-bearing measurements with noise ($\sigma_r = 0.5$ m, $\sigma_\phi = 5°$). Plot the true path, odometry-only path (dead reckoning), and EKF-estimated path. How does the number of visible landmarks affect accuracy?

3. **Particle filter global localization**: Implement the particle filter and demonstrate global localization: start with particles uniformly distributed over a 20m x 20m environment. The robot makes a series of observations of 3 landmarks. Show how the particle cloud converges from a uniform distribution to a tight cluster around the true position.

4. **Filter comparison**: For the same nonlinear localization problem, run EKF, UKF, and particle filter side by side. Compare: (a) estimation error over time, (b) computational time per step, (c) behavior when initialized far from the true state. Under what conditions does each filter fail?

5. **Multi-rate fusion**: Implement a multi-rate estimator that fuses: IMU at 100 Hz (acceleration + angular velocity), wheel odometry at 50 Hz, and simulated GPS at 1 Hz. Compare the estimate quality when GPS is available vs. when GPS is lost for 30 seconds. How quickly does the estimate degrade without GPS?

---

## Further Reading

- Thrun, S., Burgard, W., and Fox, D. *Probabilistic Robotics*. MIT Press, 2005. (Definitive robotics estimation reference)
- Kalman, R. E. "A New Approach to Linear Filtering and Prediction Problems." *ASME Journal of Basic Engineering*, 1960. (Original Kalman filter paper)
- Julier, S. and Uhlmann, J. "Unscented Filtering and Nonlinear Estimation." *Proceedings of the IEEE*, 2004. (UKF tutorial)
- Doucet, A. et al. *Sequential Monte Carlo Methods in Practice*. Springer, 2001. (Particle filtering reference)
- Sola, J. "Quaternion Kinematics for the Error-State Kalman Filter." arXiv:1711.02508, 2017. (Essential for IMU-based estimation)

---

[← Previous: Sensors and Perception](10_Sensors_and_Perception.md) | [Next: SLAM →](12_SLAM.md)
