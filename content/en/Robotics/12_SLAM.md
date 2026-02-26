# 12. SLAM (Simultaneous Localization and Mapping)

[← Previous: State Estimation and Filtering](11_State_Estimation.md) | [Next: ROS2 Fundamentals →](13_ROS2_Fundamentals.md)

---

## Learning Objectives

1. Formulate the SLAM problem as joint estimation of robot pose and map
2. Implement EKF-SLAM with state augmentation and landmark management
3. Understand graph-based SLAM and its optimization formulation
4. Describe particle filter SLAM (FastSLAM) and its advantages
5. Compare visual SLAM approaches: feature-based (ORB-SLAM) vs. direct methods
6. Explain loop closure detection and its critical role in map consistency

---

In the previous lesson, we learned how to estimate a robot's state given known sensor models and a known map of landmarks. But what if the robot does not have a map? This is the reality for most autonomous systems exploring new environments — a delivery robot in an unfamiliar building, a Mars rover on unexplored terrain, or a drone surveying a disaster site. The robot must simultaneously build a map of the environment while localizing itself within that map.

This **chicken-and-egg problem** is what makes SLAM so challenging and fascinating: accurate localization requires a good map, but building a good map requires accurate localization. SLAM has been called "the holy grail of mobile robotics" and remains one of the most active research areas in the field. Over the past three decades, practical SLAM solutions have gone from theoretical curiosities to production-ready systems powering self-driving cars, robot vacuums, and augmented reality headsets.

> **Analogy**: SLAM is like being dropped in a foreign city without a map — you simultaneously sketch the map and figure out where you are on it. At first, everything is uncertain. But as you walk around, you recognize landmarks ("that fountain again!"), and each recognition refines both your position and your map. If you walk in a big loop and recognize where you started, you can suddenly correct all the small errors accumulated along the way — this is loop closure, and it is the key to globally consistent maps.

---

## 1. The SLAM Problem

### 1.1 Formal Definition

Given:
- Control inputs: $\mathbf{u}_{1:T} = \{\mathbf{u}_1, \mathbf{u}_2, \ldots, \mathbf{u}_T\}$
- Observations: $\mathbf{z}_{1:T} = \{\mathbf{z}_1, \mathbf{z}_2, \ldots, \mathbf{z}_T\}$

Estimate:
- Robot trajectory: $\mathbf{x}_{0:T} = \{\mathbf{x}_0, \mathbf{x}_1, \ldots, \mathbf{x}_T\}$
- Map: $\mathbf{m} = \{\mathbf{m}_1, \mathbf{m}_2, \ldots, \mathbf{m}_N\}$ (landmark positions or occupancy grid)

The full SLAM posterior:

$$p(\mathbf{x}_{0:T}, \mathbf{m} | \mathbf{z}_{1:T}, \mathbf{u}_{1:T})$$

Two variants:

- **Full SLAM**: Estimate the entire trajectory $\mathbf{x}_{0:T}$ and map $\mathbf{m}$ — offline, batch optimization
- **Online SLAM**: Estimate only the current pose $\mathbf{x}_T$ and map $\mathbf{m}$ — real-time, recursive

### 1.2 Why SLAM Is Hard

1. **High dimensionality**: The state includes both the robot pose (3-6 DOF) and all landmark positions ($2N$ or $3N$ dimensions). With thousands of landmarks, the state space is enormous.
2. **Data association**: The robot must determine which observation corresponds to which landmark. Wrong associations lead to catastrophic map corruption.
3. **Loop closure**: Small odometry errors accumulate into large drift. When the robot revisits a location, it must recognize the place and correct the accumulated error across the entire trajectory.

### 1.3 Map Representations

| Representation | Description | Pros | Cons |
|---------------|-------------|------|------|
| Landmark map | Set of point landmarks $\{(x_i, y_i)\}$ | Compact, easy to match | Sparse, limited to distinctive features |
| Occupancy grid | Probability grid of obstacles | Dense, works with LiDAR | Memory-intensive, resolution-limited |
| Point cloud | Dense 3D points | Rich geometry | Very large, hard to update |
| Mesh/surface | Triangulated surfaces | Good for visualization | Complex to construct and update |
| Topological | Graph of places and connections | Compact, human-readable | Loses metric precision |

---

## 2. EKF-SLAM

### 2.1 State Augmentation

EKF-SLAM maintains a joint state vector containing both the robot pose and all landmark positions:

$$\mathbf{y} = \begin{bmatrix} \mathbf{x}_r \\ \mathbf{m}_1 \\ \mathbf{m}_2 \\ \vdots \\ \mathbf{m}_N \end{bmatrix} \in \mathbb{R}^{3 + 2N}$$

The covariance matrix has a crucial structure:

$$P = \begin{bmatrix} P_{rr} & P_{rm_1} & P_{rm_2} & \cdots \\ P_{m_1 r} & P_{m_1 m_1} & P_{m_1 m_2} & \cdots \\ P_{m_2 r} & P_{m_2 m_1} & P_{m_2 m_2} & \cdots \\ \vdots & \vdots & \vdots & \ddots \end{bmatrix}$$

The off-diagonal blocks $P_{m_i m_j}$ capture **correlations between landmarks** — these correlations are the key insight of SLAM. When the robot observes one landmark, the update propagates through the correlations to improve estimates of other landmarks.

### 2.2 Prediction

Only the robot pose changes during motion:

$$\mathbf{x}_r^- = f(\mathbf{x}_r, \mathbf{u})$$

The state transition Jacobian affects only the robot-related portions of the covariance:

$$F = \begin{bmatrix} F_r & 0 \\ 0 & I_{2N} \end{bmatrix}$$

$$P^- = F P F^T + \begin{bmatrix} Q & 0 \\ 0 & 0 \end{bmatrix}$$

Landmarks don't move, so their state and covariance rows/columns are unchanged during prediction. Only the robot rows and the robot-landmark cross-covariances are updated.

### 2.3 Update and Landmark Initialization

When the robot observes landmark $j$:

1. If landmark $j$ is **new** (not in the state): augment the state vector and covariance with the new landmark
2. If landmark $j$ is **known**: perform a standard EKF update

```python
import numpy as np

class EKFSLAM:
    """EKF-SLAM with state augmentation for landmark-based mapping.

    Why track landmark-landmark correlations? When the robot observes
    landmark A, it creates a correlation between A and the robot's position.
    If the robot then observes landmark B, a correlation between A and B
    is created through the shared robot state. When A is later re-observed,
    the information propagates to improve B's estimate even though B was
    not directly observed — this is the essence of SLAM.
    """

    def __init__(self, x0, P0, Q, R):
        self.state = np.array(x0, dtype=float)  # [x, y, theta]
        self.P = np.array(P0, dtype=float)       # 3x3 initially
        self.Q = np.array(Q, dtype=float)        # Process noise (3x3)
        self.R = np.array(R, dtype=float)        # Measurement noise (2x2)
        self.n_landmarks = 0
        self.landmark_ids = {}  # Map from landmark ID to state index

    def predict(self, u, dt):
        """Predict robot motion (landmarks are stationary)."""
        v, omega = u
        theta = self.state[2]

        # Robot motion model
        if abs(omega) > 1e-6:
            dx = -v/omega*np.sin(theta) + v/omega*np.sin(theta+omega*dt)
            dy = v/omega*np.cos(theta) - v/omega*np.cos(theta+omega*dt)
            dtheta = omega * dt
        else:
            dx = v*np.cos(theta)*dt
            dy = v*np.sin(theta)*dt
            dtheta = 0.0

        self.state[0] += dx
        self.state[1] += dy
        self.state[2] += dtheta
        self.state[2] = (self.state[2] + np.pi) % (2*np.pi) - np.pi

        # Jacobian of motion model (only affects robot pose)
        F_r = np.eye(3)
        if abs(omega) > 1e-6:
            F_r[0, 2] = -v/omega*np.cos(theta) + v/omega*np.cos(theta+omega*dt)
            F_r[1, 2] = -v/omega*np.sin(theta) + v/omega*np.sin(theta+omega*dt)
        else:
            F_r[0, 2] = -v*np.sin(theta)*dt
            F_r[1, 2] = v*np.cos(theta)*dt

        # Build full Jacobian
        n = len(self.state)
        F = np.eye(n)
        F[:3, :3] = F_r

        # Update covariance — only robot rows/cols get process noise
        self.P = F @ self.P @ F.T
        self.P[:3, :3] += self.Q

    def update(self, landmark_id, z):
        """Update with range-bearing measurement to a landmark.

        z = [range, bearing]

        Why augment state for new landmarks instead of maintaining a
        separate list? By including landmarks in the state vector, the
        EKF automatically computes and maintains the correlations between
        all landmarks and the robot. This correlation structure is what
        makes SLAM work — it allows information from one landmark
        observation to improve estimates of other landmarks.
        """
        if landmark_id not in self.landmark_ids:
            # New landmark — initialize it
            self._add_landmark(landmark_id, z)
            return

        # Known landmark — perform EKF update
        idx = self.landmark_ids[landmark_id]
        lx = self.state[idx]
        ly = self.state[idx + 1]
        rx, ry, rtheta = self.state[0], self.state[1], self.state[2]

        dx = lx - rx
        dy = ly - ry
        q = dx**2 + dy**2
        r = np.sqrt(q)

        # Expected measurement
        z_hat = np.array([r, np.arctan2(dy, dx) - rtheta])
        z_hat[1] = (z_hat[1] + np.pi) % (2*np.pi) - np.pi

        # Jacobian w.r.t. full state
        n = len(self.state)
        H = np.zeros((2, n))
        # w.r.t. robot pose
        H[0, 0] = -dx/r;  H[0, 1] = -dy/r;  H[0, 2] = 0
        H[1, 0] = dy/q;   H[1, 1] = -dx/q;  H[1, 2] = -1
        # w.r.t. landmark position
        H[0, idx] = dx/r;   H[0, idx+1] = dy/r
        H[1, idx] = -dy/q;  H[1, idx+1] = dx/q

        # Innovation
        innovation = z - z_hat
        innovation[1] = (innovation[1] + np.pi) % (2*np.pi) - np.pi

        # Standard EKF update
        S = H @ self.P @ H.T + self.R
        # Kalman gain K optimally weights prediction vs observation:
        # K = P*H^T * (H*P*H^T + R)^{-1} — when prediction uncertainty (P) is
        # large relative to measurement noise (R), K is large and the update
        # trusts the observation more. When P is small (confident prediction),
        # K is small and the observation has little effect. Adding innovation
        # directly (K=I) would ignore these uncertainties entirely.
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ innovation
        self.state[2] = (self.state[2] + np.pi) % (2*np.pi) - np.pi
        self.P = (np.eye(n) - K @ H) @ self.P

    def _add_landmark(self, landmark_id, z):
        """Add a new landmark to the state vector.

        Why initialize with the current measurement? The first observation
        gives us a range-bearing estimate of the landmark's position
        relative to the robot. We convert this to global coordinates
        and add it to the state. The initial uncertainty is large
        (based on measurement noise) but shrinks with future observations.
        """
        r, bearing = z
        rx, ry, rtheta = self.state[0], self.state[1], self.state[2]

        # Convert range-bearing to global position
        lx = rx + r * np.cos(rtheta + bearing)
        ly = ry + r * np.sin(rtheta + bearing)

        # Augment state
        self.state = np.append(self.state, [lx, ly])
        idx = len(self.state) - 2
        self.landmark_ids[landmark_id] = idx

        # Augment covariance
        n_old = len(self.P)
        P_new = np.zeros((n_old + 2, n_old + 2))
        P_new[:n_old, :n_old] = self.P

        # Jacobian of landmark initialization w.r.t. robot state
        G_r = np.array([
            [1, 0, -r*np.sin(rtheta + bearing)],
            [0, 1,  r*np.cos(rtheta + bearing)]
        ])
        # Jacobian w.r.t. measurement
        G_z = np.array([
            [np.cos(rtheta + bearing), -r*np.sin(rtheta + bearing)],
            [np.sin(rtheta + bearing),  r*np.cos(rtheta + bearing)]
        ])

        # New landmark covariance
        P_new[n_old:, n_old:] = G_r @ self.P[:3, :3] @ G_r.T + G_z @ self.R @ G_z.T
        # Cross-covariance with existing state
        P_new[n_old:, :n_old] = G_r @ self.P[:3, :]
        P_new[:n_old, n_old:] = P_new[n_old:, :n_old].T

        self.P = P_new
        self.n_landmarks += 1
```

### 2.4 EKF-SLAM Limitations

- **Quadratic complexity**: The covariance matrix is $(3+2N) \times (3+2N)$, making each update $O(N^2)$. With thousands of landmarks, this becomes prohibitive.
- **Linearization errors**: As with any EKF, the first-order approximation can lead to inconsistency, especially after long trajectories.
- **Data association**: Correctly matching observations to landmarks is critical and difficult, especially in environments with repetitive features. Mahalanobis distance ($d_M = \sqrt{(\mathbf{z} - \hat{\mathbf{z}})^T S^{-1} (\mathbf{z} - \hat{\mathbf{z}})}$) is preferred over Euclidean distance for matching because it accounts for the covariance structure of the innovation: a 1-meter range error matters more when the range sensor has 0.1 m noise than when it has 10 m noise. Euclidean distance treats all dimensions equally and ignores uncertainty, leading to incorrect associations when sensor noise is anisotropic.

---

## 3. Graph-Based SLAM

### 3.1 SLAM as Graph Optimization

Graph-based SLAM reformulates the problem as a **factor graph** optimization:

- **Nodes**: Robot poses $\mathbf{x}_0, \mathbf{x}_1, \ldots, \mathbf{x}_T$ and landmark positions $\mathbf{m}_1, \ldots, \mathbf{m}_N$
- **Edges**: Constraints from odometry (between consecutive poses) and observations (between poses and landmarks)

```
x0 ──odometry── x1 ──odometry── x2 ──odometry── x3
 \              / \               |               /
  \landmark   /   \landmark      |landmark      /
   \        /      \             |             / loop closure
    m1              m2           m3          x0
```

Each edge encodes a measurement with its uncertainty. The optimization finds the node positions that best satisfy all constraints simultaneously.

### 3.2 Pose Graph Optimization

In **pose graph SLAM**, landmarks are eliminated and only robot poses are nodes. Edges come from:

- **Odometry**: Relative motion between consecutive poses
- **Loop closures**: Relative pose constraints between non-consecutive poses (detected when the robot revisits a location)

The optimization minimizes:

$$\mathbf{x}^* = \arg\min_\mathbf{x} \sum_{(i,j) \in \text{edges}} \| h(\mathbf{x}_i, \mathbf{x}_j) - \mathbf{z}_{ij} \|^2_{\Omega_{ij}}$$

where $\mathbf{z}_{ij}$ is the measured relative transformation, $h(\mathbf{x}_i, \mathbf{x}_j)$ is the expected relative transformation, and $\Omega_{ij}$ is the information matrix (inverse covariance).

### 3.3 Gauss-Newton Optimization

The standard solver for pose graph SLAM is **Gauss-Newton** or **Levenberg-Marquardt**:

$$\Delta\mathbf{x}^* = -(J^T \Omega J)^{-1} J^T \Omega \mathbf{e}$$

where $J$ is the Jacobian of the error function and $\mathbf{e}$ is the error vector.

The matrix $H = J^T \Omega J$ (the Hessian approximation) is **sparse** because each edge only involves two nodes. This sparsity allows efficient solving using sparse linear algebra (Cholesky decomposition).

```python
class PoseGraphSLAM:
    """Simplified 2D pose graph SLAM optimizer.

    Why graph-based instead of filter-based? Graph-based SLAM solves
    the full SLAM posterior by optimizing all poses jointly. This gives
    better results than filters (which are causal and cannot revise
    past estimates). The key insight: when a loop closure is detected,
    the optimization redistributes the accumulated error across the
    entire trajectory, producing a globally consistent map.
    """

    def __init__(self):
        self.poses = []      # List of [x, y, theta]
        self.edges = []      # List of (i, j, z_ij, omega_ij)

    def add_pose(self, pose):
        """Add a new pose node."""
        self.poses.append(np.array(pose, dtype=float))
        return len(self.poses) - 1

    def add_edge(self, i, j, z_ij, info_matrix):
        """Add a constraint between poses i and j.

        z_ij: relative transformation from i to j [dx, dy, dtheta]
        info_matrix: 3x3 information matrix (inverse covariance)

        Why an information matrix? Inverse covariance weights the
        residuals: highly certain measurements (low covariance → high
        information) contribute more to the cost function. This is
        equivalent to Mahalanobis distance minimization.
        """
        self.edges.append((i, j, np.array(z_ij), np.array(info_matrix)))

    def compute_error(self, poses_flat):
        """Compute total error for current pose configuration."""
        n = len(self.poses)
        total_error = 0.0

        for i, j, z_ij, omega in self.edges:
            xi = poses_flat[3*i:3*i+3]
            xj = poses_flat[3*j:3*j+3]

            # Relative transformation from i to j
            dx = xj[0] - xi[0]
            dy = xj[1] - xi[1]
            dtheta = xj[2] - xi[2]

            # Transform to frame of pose i
            c = np.cos(xi[2])
            s = np.sin(xi[2])
            e = np.array([
                c*dx + s*dy - z_ij[0],
                -s*dx + c*dy - z_ij[1],
                dtheta - z_ij[2]
            ])
            e[2] = (e[2] + np.pi) % (2*np.pi) - np.pi

            total_error += e.T @ omega @ e

        return total_error

    def optimize(self, n_iterations=20):
        """Run Gauss-Newton optimization on the pose graph.

        Why fix the first pose? The pose graph defines only relative
        constraints. Without an anchor, the entire graph could translate
        and rotate without changing the error. Fixing x0 removes this
        gauge freedom and makes the system well-conditioned.
        """
        n = len(self.poses)
        x = np.concatenate(self.poses)

        for iteration in range(n_iterations):
            # Build linear system H * dx = b
            dim = 3 * n
            H = np.zeros((dim, dim))
            b = np.zeros(dim)

            for idx_i, idx_j, z_ij, omega in self.edges:
                xi = x[3*idx_i:3*idx_i+3]
                xj = x[3*idx_j:3*idx_j+3]

                # Error
                dx_val = xj[0] - xi[0]
                dy_val = xj[1] - xi[1]
                c = np.cos(xi[2])
                s = np.sin(xi[2])

                e = np.array([
                    c*dx_val + s*dy_val - z_ij[0],
                    -s*dx_val + c*dy_val - z_ij[1],
                    (xj[2] - xi[2] - z_ij[2] + np.pi) % (2*np.pi) - np.pi
                ])

                # Jacobian w.r.t. xi
                A = np.array([
                    [-c, -s, -s*dx_val + c*dy_val],
                    [s, -c, -c*dx_val - s*dy_val],
                    [0, 0, -1]
                ])
                # Jacobian w.r.t. xj
                B = np.array([
                    [c, s, 0],
                    [-s, c, 0],
                    [0, 0, 1]
                ])

                # Accumulate into H and b
                H[3*idx_i:3*idx_i+3, 3*idx_i:3*idx_i+3] += A.T @ omega @ A
                H[3*idx_i:3*idx_i+3, 3*idx_j:3*idx_j+3] += A.T @ omega @ B
                H[3*idx_j:3*idx_j+3, 3*idx_i:3*idx_i+3] += B.T @ omega @ A
                H[3*idx_j:3*idx_j+3, 3*idx_j:3*idx_j+3] += B.T @ omega @ B
                b[3*idx_i:3*idx_i+3] += A.T @ omega @ e
                b[3*idx_j:3*idx_j+3] += B.T @ omega @ e

            # Fix first pose (anchor)
            H[:3, :3] += np.eye(3) * 1e6

            # Solve
            delta_x = np.linalg.solve(H, -b)
            x += delta_x

            if np.linalg.norm(delta_x) < 1e-6:
                break

        # Update poses
        for i in range(n):
            self.poses[i] = x[3*i:3*i+3]

        return self.poses
```

### 3.4 Advantages of Graph-Based SLAM

- **Global optimization**: Distributes error across the entire trajectory when loop closures are detected
- **Sparsity**: The Hessian matrix is sparse, enabling efficient solvers ($O(n)$ for sparse Cholesky in favorable cases)
- **Modularity**: The frontend (data association, loop closure detection) is decoupled from the backend (optimization)
- **Flexible**: Can handle any combination of measurement types (odometry, GPS, landmarks, relative poses)

---

## 4. Particle Filter SLAM (FastSLAM)

### 4.1 Key Insight: Conditional Independence

FastSLAM exploits the fact that **given the robot trajectory, landmark estimates are independent**. This means we can factorize:

$$p(\mathbf{x}_{0:T}, \mathbf{m} | \mathbf{z}, \mathbf{u}) = p(\mathbf{x}_{0:T} | \mathbf{z}, \mathbf{u}) \prod_{j=1}^{N} p(\mathbf{m}_j | \mathbf{x}_{0:T}, \mathbf{z}, \mathbf{u})$$

**FastSLAM uses**:
- A **particle filter** for the robot trajectory $\mathbf{x}_{0:T}$
- An **independent EKF** for each landmark $\mathbf{m}_j$ within each particle

### 4.2 Complexity

- Each particle maintains $N$ small (2x2) EKFs instead of one large $(3+2N) \times (3+2N)$ EKF
- Update cost per particle: $O(N)$ instead of $O(N^2)$
- Total cost: $O(MN)$ where $M$ is the number of particles

In practice, FastSLAM requires far fewer particles than a pure particle filter for the joint state because the landmarks are estimated analytically (via EKFs).

```python
class FastSLAMParticle:
    """A single particle in FastSLAM.

    Each particle represents one hypothesis for the robot trajectory.
    Attached to each particle is a set of independent EKFs — one per
    landmark. Because landmarks are conditionally independent given
    the trajectory, each EKF is just 2x2 (for 2D landmarks), making
    FastSLAM much more scalable than EKF-SLAM.
    """

    def __init__(self, pose, n_max_landmarks=100):
        self.pose = np.array(pose, dtype=float)  # [x, y, theta]
        self.weight = 1.0
        self.landmarks = {}  # landmark_id -> (mean, covariance)

    def predict(self, u, dt, noise_std):
        """Propagate this particle's pose with noise."""
        v, omega = u
        theta = self.pose[2]
        noise = noise_std * np.random.randn(3)

        if abs(omega) > 1e-6:
            self.pose[0] += -v/omega*np.sin(theta) + v/omega*np.sin(theta+omega*dt)
            self.pose[1] += v/omega*np.cos(theta) - v/omega*np.cos(theta+omega*dt)
            self.pose[2] += omega * dt
        else:
            self.pose[0] += v*np.cos(theta)*dt
            self.pose[1] += v*np.sin(theta)*dt

        self.pose += noise
        self.pose[2] = (self.pose[2] + np.pi) % (2*np.pi) - np.pi

    def update_landmark(self, landmark_id, z, R):
        """Update or initialize a landmark EKF for this particle.

        Why is this O(1) per landmark? Each landmark has its own
        tiny 2x2 EKF, independent of all other landmarks. The update
        involves a 2x2 matrix inversion — constant time. Compare with
        EKF-SLAM where every update is O(N^2).
        """
        r, bearing = z
        rx, ry, rtheta = self.pose

        if landmark_id not in self.landmarks:
            # Initialize new landmark
            lx = rx + r * np.cos(rtheta + bearing)
            ly = ry + r * np.sin(rtheta + bearing)
            # Initial covariance from measurement uncertainty
            G_z = np.array([
                [np.cos(rtheta+bearing), -r*np.sin(rtheta+bearing)],
                [np.sin(rtheta+bearing),  r*np.cos(rtheta+bearing)]
            ])
            P_init = G_z @ R @ G_z.T
            self.landmarks[landmark_id] = (np.array([lx, ly]), P_init)
            self.weight *= 1.0  # No information gain for new landmarks
            return

        # Existing landmark — EKF update
        mu, sigma = self.landmarks[landmark_id]
        dx = mu[0] - rx
        dy = mu[1] - ry
        q = dx**2 + dy**2
        r_exp = np.sqrt(q)

        z_hat = np.array([r_exp, np.arctan2(dy, dx) - rtheta])
        z_hat[1] = (z_hat[1] + np.pi) % (2*np.pi) - np.pi

        H = np.array([
            [dx/r_exp, dy/r_exp],
            [-dy/q, dx/q]
        ])

        S = H @ sigma @ H.T + R
        K = sigma @ H.T @ np.linalg.inv(S)

        innovation = z - z_hat
        innovation[1] = (innovation[1] + np.pi) % (2*np.pi) - np.pi

        mu_new = mu + K @ innovation
        sigma_new = (np.eye(2) - K @ H) @ sigma

        self.landmarks[landmark_id] = (mu_new, sigma_new)

        # Update particle weight based on measurement likelihood.
        # Why resampling is needed (done externally after all updates):
        # Without resampling, particle weights diverge over time — a few
        # particles accumulate nearly all the weight while the rest become
        # negligible. This "particle degeneracy" wastes computation on
        # irrelevant hypotheses. Resampling duplicates high-weight particles
        # and discards low-weight ones, keeping the particle set focused on
        # plausible trajectories.
        self.weight *= np.exp(-0.5 * innovation.T @ np.linalg.inv(S) @ innovation)
        self.weight /= np.sqrt(np.linalg.det(2 * np.pi * S))
```

---

## 5. Visual SLAM

### 5.1 Feature-Based Visual SLAM

**Feature-based methods** extract and track visual features (corners, blobs) across frames:

1. **Feature detection**: Find distinctive points (ORB, SIFT, SURF)
2. **Feature description**: Compute a descriptor vector for each feature
3. **Feature matching**: Match features between frames using descriptor similarity
4. **Pose estimation**: Compute camera motion from matched features (essential matrix, PnP)
5. **Triangulation**: Estimate 3D positions of matched features
6. **Bundle adjustment**: Jointly optimize camera poses and 3D points

**ORB-SLAM** (and its successor ORB-SLAM3) is the most well-known feature-based visual SLAM system:

```
Input Image → Feature Extraction (ORB) → Tracking (match to local map)
     ↓                                       ↓
 Relocalization                        Local Mapping
     ↑                                       ↓
Loop Closing ←── Place Recognition ←── Keyframe Selection
     ↓
 Pose Graph Optimization
```

Key components:
- **Tracking thread**: Localizes each new frame against the local map
- **Local mapping thread**: Triangulates new map points, performs local bundle adjustment
- **Loop closing thread**: Detects loop closures using a bag-of-words place recognition system, performs global optimization

### 5.2 Direct Visual SLAM

**Direct methods** (e.g., LSD-SLAM, DSO) use pixel intensities directly rather than extracting features:

$$E_{photo} = \sum_{\mathbf{p} \in \mathcal{P}} \| I_{ref}(\mathbf{p}) - I_{cur}(\pi(T \cdot \pi^{-1}(\mathbf{p}, d_\mathbf{p}))) \|^2$$

where the photometric error measures how well the reference image $I_{ref}$ matches the current image $I_{cur}$ under the estimated camera motion $T$ and depth map $d$.

**Feature-based vs. Direct**:

| Aspect | Feature-Based (ORB-SLAM) | Direct (LSD-SLAM, DSO) |
|--------|-------------------------|----------------------|
| Input | Feature points | Raw pixel intensities |
| Map density | Sparse (feature points) | Semi-dense or dense |
| Texture requirement | Needs corners/edges | Needs intensity gradients |
| Speed | Fast feature extraction | Slower (per-pixel optimization) |
| Robustness | Handles moderate blur | Sensitive to photometric changes |
| Initialization | Needs sufficient parallax | Needs texture |

### 5.3 Modern Visual SLAM Systems

**ORB-SLAM3** (2021):
- Supports monocular, stereo, RGB-D, and IMU
- Multi-map system: handles tracking failures by creating new maps and merging later
- Visual-inertial fusion for robust state estimation

**RTAB-Map** (Real-Time Appearance-Based Mapping):
- Works with LiDAR, stereo, and RGB-D
- Memory management: maintains short-term and long-term memory to handle large environments
- Graph-based optimization with loop closure

```python
def visual_slam_pipeline_overview():
    """Conceptual overview of a feature-based visual SLAM pipeline.

    This is not a complete implementation — full visual SLAM systems
    are thousands of lines of optimized code. This shows the key steps
    to illustrate the concepts.
    """
    # Step 1: Feature detection and description
    # keypoints, descriptors = orb.detectAndCompute(image, None)

    # Step 2: Feature matching with previous frame
    # matches = matcher.knnMatch(desc_prev, desc_curr, k=2)
    # Apply Lowe's ratio test to filter bad matches

    # Step 3: Estimate relative pose (essential matrix + decomposition)
    # E, mask = cv2.findEssentialMat(pts1, pts2, K)
    # _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)

    # Step 4: Triangulate new 3D points
    # points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    # points_3d = points_4d[:3] / points_4d[3]

    # Step 5: Add to map, create keyframe if needed
    # Check parallax, number of new points, tracking quality

    # Step 6: Local bundle adjustment (optimize recent keyframes + points)
    # Minimize reprojection error using Levenberg-Marquardt

    # Step 7: Loop closure detection
    # Use bag-of-words to find similar keyframes
    # Verify with geometric consistency check
    # If confirmed, add loop closure edge and optimize pose graph
    pass
```

---

## 6. Loop Closure

### 6.1 Why Loop Closure Matters

Without loop closure, odometry drift accumulates without bound. After traveling 100 meters, a robot might have 1-5 meters of positional error. When the robot returns to a previously visited location, detecting this **revisit** allows correcting the entire trajectory.

### 6.2 Place Recognition

Loop closure detection is fundamentally a **place recognition** problem: given the current sensor observation, have I seen this place before?

**Visual place recognition** approaches:

1. **Bag of Visual Words (BoVW)**: Cluster feature descriptors into a visual vocabulary. Represent each image as a histogram of visual words. Compare histograms using TF-IDF similarity.

2. **Neural network embeddings**: Use CNNs to compute a global descriptor for each image. Compare descriptors using cosine similarity. Modern methods (NetVLAD, SuperGlue) achieve impressive performance.

3. **Point cloud descriptors**: For LiDAR, compute global descriptors of point clouds (Scan Context, PointNetVLAD).

### 6.3 Loop Closure Verification

A false loop closure (incorrectly matching two different places) is catastrophic — it corrupts the map irreversibly. Therefore, rigorous verification is essential:

1. **Geometric verification**: After a candidate match is found, verify that the geometric relationship between features is consistent (compute the fundamental matrix, check inlier ratio).
2. **Temporal consistency**: Require multiple consecutive loop closure candidates before accepting.
3. **Robust optimization**: Use robust cost functions (Huber, Cauchy) in the backend optimizer to downweight potentially wrong loop closures.

```python
def loop_closure_detection(current_descriptor, keyframe_descriptors,
                          threshold=0.8, min_interval=20):
    """Simple loop closure detection by descriptor matching.

    Why require a minimum interval? Recent keyframes are expected to
    be similar (the robot hasn't moved far). True loop closures occur
    when the robot returns to a much earlier location. Requiring a
    minimum keyframe interval avoids trivially matching nearby frames.
    """
    current_idx = len(keyframe_descriptors) - 1
    best_score = -1
    best_match = -1

    for i in range(len(keyframe_descriptors) - min_interval):
        # Cosine similarity between descriptors
        score = np.dot(current_descriptor, keyframe_descriptors[i])
        score /= (np.linalg.norm(current_descriptor) *
                  np.linalg.norm(keyframe_descriptors[i]) + 1e-10)

        if score > best_score:
            best_score = score
            best_match = i

    if best_score > threshold:
        return best_match, best_score  # Loop closure candidate
    return None, best_score
```

---

## 7. SLAM Frontends and Backends

### 7.1 Architecture Pattern

Modern SLAM systems follow a **frontend-backend** architecture:

```
┌─────────────────────────────────────────────────┐
│  FRONTEND (Perception)                           │
│  - Feature extraction / scan matching            │
│  - Data association (which landmark is which?)    │
│  - Loop closure detection                        │
│  - Odometry estimation                           │
├─────────────────────────────────────────────────┤
│  BACKEND (Optimization)                          │
│  - Factor graph construction                     │
│  - Nonlinear optimization (Gauss-Newton, LM)     │
│  - Sparse linear algebra (Cholesky, QR)          │
│  - Marginal covariance computation               │
└─────────────────────────────────────────────────┘
```

**Popular backends**:
- **g2o**: General graph optimization (C++)
- **GTSAM**: Factor graph library with incremental solvers (iSAM2)
- **Ceres Solver**: General nonlinear least squares (Google)

### 7.2 Comparison of SLAM Approaches

| Method | Complexity | Map Type | Strengths | Weaknesses |
|--------|-----------|----------|-----------|------------|
| EKF-SLAM | $O(N^2)$ | Landmarks | Simple, well-understood | Scales poorly, linearization errors |
| Graph-SLAM | $O(N)$ amortized | Any | Globally optimal, scalable | Batch (not fully online) |
| FastSLAM | $O(M \log N)$ | Landmarks | Handles data association uncertainty | Particle degeneracy |
| ORB-SLAM3 | Real-time | Sparse 3D | Robust, multi-sensor | Needs texture |
| LiDAR-SLAM | Real-time | Point cloud | Accurate, lighting invariant | Expensive sensor |

---

## Summary

| Concept | Key Idea |
|---------|----------|
| SLAM problem | Simultaneously estimate robot pose and build environment map |
| EKF-SLAM | Joint Gaussian over robot pose and landmarks; $O(N^2)$ |
| Landmark correlations | Observing one landmark improves estimates of others through shared uncertainty |
| Graph-based SLAM | Formulate as nonlinear optimization over a pose graph; sparse, scalable |
| Loop closure | Detecting revisited places; enables global error correction |
| FastSLAM | Particle filter for trajectory + EKF per landmark; exploits conditional independence |
| Visual SLAM | Camera-based SLAM; feature-based (ORB-SLAM) or direct (LSD-SLAM) |
| Frontend-backend | Decouple perception (feature extraction, matching) from optimization (graph solving) |

---

## Exercises

1. **EKF-SLAM simulation**: Implement the EKF-SLAM class above. Place 10 landmarks in a 20x20 m environment. Simulate a robot driving in a square path, observing landmarks within 8 m range. Plot the estimated and true landmark positions over time. How does the uncertainty ellipse of each landmark change as the robot revisits it?

2. **Pose graph optimization**: Create a pose graph for a robot that drives in a loop (e.g., a square trajectory). Add odometry edges with small Gaussian noise. Then add a loop closure edge connecting the last pose to the first. Run the optimizer and compare the trajectory before and after optimization. How does the loop closure edge redistribute the accumulated drift?

3. **Data association challenge**: Modify the EKF-SLAM to use nearest-neighbor data association (match each observation to the closest existing landmark). Create a scenario where two landmarks are close together. Show how incorrect data association can corrupt the map. Then implement a simple Mahalanobis distance gating strategy and demonstrate improved robustness.

4. **Visual feature matching**: Using Python (OpenCV), detect ORB features in two images taken from slightly different viewpoints. Match features using brute-force matching with Lowe's ratio test. Estimate the essential matrix and recover the relative camera pose. Triangulate matched points to create a sparse 3D reconstruction.

5. **Loop closure impact**: Create a pose graph SLAM scenario where a robot traverses a long corridor (50 poses) and returns to the start. First run optimization without any loop closure. Then add a loop closure edge and re-optimize. Quantify the improvement in terms of (a) final position error and (b) total trajectory error. How does the quality of the loop closure measurement affect the result?

---

## Further Reading

- Thrun, S., Burgard, W., and Fox, D. *Probabilistic Robotics*. MIT Press, 2005. Chapters 10-13. (Foundational SLAM reference)
- Cadena, C. et al. "Past, Present, and Future of Simultaneous Localization and Mapping." *IEEE Transactions on Robotics*, 2016. (Comprehensive SLAM survey)
- Grisetti, G. et al. "A Tutorial on Graph-Based SLAM." *IEEE Intelligent Transportation Systems Magazine*, 2010. (Excellent graph SLAM tutorial)
- Campos, C. et al. "ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial, and Multi-Map SLAM." *IEEE Transactions on Robotics*, 2021.
- Montemerlo, M. et al. "FastSLAM: A Factored Solution to the Simultaneous Localization and Mapping Problem." *AAAI*, 2002.

---

[← Previous: State Estimation and Filtering](11_State_Estimation.md) | [Next: ROS2 Fundamentals →](13_ROS2_Fundamentals.md)
