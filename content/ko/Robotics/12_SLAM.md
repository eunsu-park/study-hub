# 12. 동시적 위치 추정 및 지도 작성(SLAM, Simultaneous Localization and Mapping)

[← 이전: 상태 추정과 필터링](11_State_Estimation.md) | [다음: ROS2 기초 →](13_ROS2_Fundamentals.md)

---

## 학습 목표

1. SLAM 문제를 로봇 자세(pose)와 지도의 결합 추정(joint estimation)으로 공식화하기
2. 상태 확장(state augmentation) 및 랜드마크 관리를 활용한 EKF-SLAM 구현하기
3. 그래프 기반 SLAM(graph-based SLAM)과 그 최적화 공식 이해하기
4. 파티클 필터 SLAM인 FastSLAM과 그 장점 설명하기
5. 시각적 SLAM(visual SLAM) 접근법 비교하기: 특징 기반(feature-based, ORB-SLAM) vs. 직접 방식(direct methods)
6. 루프 폐쇄 감지(loop closure detection)와 지도 일관성에서의 핵심 역할 설명하기

---

이전 레슨에서는 알려진 센서 모델과 랜드마크 지도가 주어진 상황에서 로봇의 상태를 추정하는 방법을 배웠습니다. 그런데 로봇이 지도를 갖고 있지 않다면 어떨까요? 이것이 새로운 환경을 탐색하는 대부분의 자율 시스템이 마주하는 현실입니다. 낯선 건물 안의 배달 로봇, 미지의 지형을 탐사하는 화성 탐사차, 재난 현장을 촬영하는 드론 등이 이에 해당합니다. 로봇은 환경의 지도를 구축하는 동시에 그 지도 안에서 자신의 위치를 파악해야 합니다.

이것이 바로 SLAM을 그토록 도전적이고 매력적으로 만드는 **닭이 먼저냐 달걀이 먼저냐의 문제**입니다. 정확한 위치 추정에는 좋은 지도가 필요하고, 좋은 지도를 만들려면 정확한 위치 추정이 필요합니다. SLAM은 "모바일 로봇공학의 성배(holy grail)"라고 불리며, 이 분야에서 가장 활발히 연구되는 주제 중 하나입니다. 지난 30년에 걸쳐 실용적인 SLAM 솔루션은 이론적 호기심에서 벗어나 자율주행 자동차, 로봇 청소기, 증강현실 헤드셋을 구동하는 생산 준비 완료 시스템으로 발전했습니다.

> **비유**: SLAM은 지도 없이 낯선 도시에 던져지는 것과 같습니다. 지도를 그리는 동시에 그 지도 위에서 자신의 위치를 파악해야 합니다. 처음에는 모든 것이 불확실합니다. 하지만 걸어 다니다 보면 랜드마크("저 분수 또 나왔네!")를 알아보게 되고, 매번 인식할 때마다 자신의 위치와 지도 모두가 정교해집니다. 큰 루프를 돌아 출발점을 알아보는 순간, 그동안 쌓인 작은 오차들을 한꺼번에 수정할 수 있습니다. 이것이 루프 폐쇄(loop closure)이며, 전역적으로 일관된 지도를 만드는 핵심입니다.

---

## 1. SLAM 문제

### 1.1 형식적 정의(Formal Definition)

주어진 것:
- 제어 입력(control inputs): $\mathbf{u}_{1:T} = \{\mathbf{u}_1, \mathbf{u}_2, \ldots, \mathbf{u}_T\}$
- 관측(observations): $\mathbf{z}_{1:T} = \{\mathbf{z}_1, \mathbf{z}_2, \ldots, \mathbf{z}_T\}$

추정해야 할 것:
- 로봇 궤적(robot trajectory): $\mathbf{x}_{0:T} = \{\mathbf{x}_0, \mathbf{x}_1, \ldots, \mathbf{x}_T\}$
- 지도(map): $\mathbf{m} = \{\mathbf{m}_1, \mathbf{m}_2, \ldots, \mathbf{m}_N\}$ (랜드마크 위치 또는 점유 격자)

SLAM의 완전한 사후 분포(full SLAM posterior):

$$p(\mathbf{x}_{0:T}, \mathbf{m} | \mathbf{z}_{1:T}, \mathbf{u}_{1:T})$$

두 가지 변형:

- **전체 SLAM(Full SLAM)**: 전체 궤적 $\mathbf{x}_{0:T}$와 지도 $\mathbf{m}$을 추정 — 오프라인, 배치 최적화
- **온라인 SLAM(Online SLAM)**: 현재 자세 $\mathbf{x}_T$와 지도 $\mathbf{m}$만 추정 — 실시간, 재귀적

### 1.2 SLAM이 어려운 이유

1. **높은 차원성(High dimensionality)**: 상태에는 로봇 자세(3~6 자유도)와 모든 랜드마크 위치($2N$ 또는 $3N$ 차원)가 포함됩니다. 수천 개의 랜드마크가 있을 경우 상태 공간이 매우 거대해집니다.
2. **데이터 연관(Data association)**: 로봇은 어느 관측이 어느 랜드마크에 해당하는지 파악해야 합니다. 잘못된 연관은 지도를 치명적으로 손상시킵니다.
3. **루프 폐쇄(Loop closure)**: 작은 오도메트리(odometry) 오차가 큰 드리프트로 누적됩니다. 로봇이 이전에 방문한 위치를 재방문할 때, 그 장소를 인식하고 전체 궤적에 걸쳐 누적된 오차를 수정해야 합니다.

### 1.3 지도 표현(Map Representations)

| 표현 방식 | 설명 | 장점 | 단점 |
|----------|------|------|------|
| 랜드마크 지도(Landmark map) | 점 랜드마크의 집합 $\{(x_i, y_i)\}$ | 간결하고 매칭이 쉬움 | 희소하고 뚜렷한 특징에 한정 |
| 점유 격자(Occupancy grid) | 장애물의 확률 격자 | 밀집하고 LiDAR에 적합 | 메모리 소모가 크고 해상도 한계 존재 |
| 점군(Point cloud) | 밀집 3D 점들 | 풍부한 기하 정보 | 매우 크고 갱신이 어려움 |
| 메시/표면(Mesh/surface) | 삼각형 분할 표면 | 시각화에 유리 | 구성 및 갱신이 복잡 |
| 위상학적(Topological) | 장소와 연결의 그래프 | 간결하고 사람이 읽기 쉬움 | 미터법 정밀도 손실 |

---

## 2. EKF-SLAM

### 2.1 상태 확장(State Augmentation)

EKF-SLAM은 로봇 자세와 모든 랜드마크 위치를 모두 포함하는 결합 상태 벡터를 유지합니다:

$$\mathbf{y} = \begin{bmatrix} \mathbf{x}_r \\ \mathbf{m}_1 \\ \mathbf{m}_2 \\ \vdots \\ \mathbf{m}_N \end{bmatrix} \in \mathbb{R}^{3 + 2N}$$

공분산 행렬은 중요한 구조를 가집니다:

$$P = \begin{bmatrix} P_{rr} & P_{rm_1} & P_{rm_2} & \cdots \\ P_{m_1 r} & P_{m_1 m_1} & P_{m_1 m_2} & \cdots \\ P_{m_2 r} & P_{m_2 m_1} & P_{m_2 m_2} & \cdots \\ \vdots & \vdots & \vdots & \ddots \end{bmatrix}$$

비대각 블록 $P_{m_i m_j}$는 **랜드마크 간 상관관계(correlations between landmarks)**를 포착합니다. 이 상관관계가 SLAM의 핵심 통찰입니다. 로봇이 하나의 랜드마크를 관측하면, 갱신이 상관관계를 통해 전파되어 다른 랜드마크의 추정값도 개선됩니다.

### 2.2 예측(Prediction)

움직임 중에는 로봇 자세만 변합니다:

$$\mathbf{x}_r^- = f(\mathbf{x}_r, \mathbf{u})$$

상태 전이 야코비안(state transition Jacobian)은 공분산의 로봇 관련 부분에만 영향을 미칩니다:

$$F = \begin{bmatrix} F_r & 0 \\ 0 & I_{2N} \end{bmatrix}$$

$$P^- = F P F^T + \begin{bmatrix} Q & 0 \\ 0 & 0 \end{bmatrix}$$

랜드마크는 움직이지 않으므로, 예측 단계에서 랜드마크의 상태와 공분산 행/열은 변하지 않습니다. 로봇의 행과 로봇-랜드마크 간 교차 공분산(cross-covariance)만 갱신됩니다.

### 2.3 갱신 및 랜드마크 초기화(Update and Landmark Initialization)

로봇이 랜드마크 $j$를 관측할 때:

1. 랜드마크 $j$가 **새로운** 것(상태에 없음): 상태 벡터와 공분산에 새 랜드마크를 추가하여 확장
2. 랜드마크 $j$가 **이미 알려진** 것: 표준 EKF 갱신 수행

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
        # 칼만 이득 K는 예측과 관측을 최적 가중한다:
        # K = P*H^T * (H*P*H^T + R)^{-1} — 예측 불확실성(P)이 측정 노이즈(R)에
        # 비해 클 때 K가 커지고 관측을 더 신뢰한다. P가 작으면(예측 확신) K가
        # 작아지고 관측의 영향이 미미하다. 혁신(innovation)을 직접 더하면(K=I)
        # 이러한 불확실성을 완전히 무시하게 된다.
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

### 2.4 EKF-SLAM의 한계

- **이차 복잡도(Quadratic complexity)**: 공분산 행렬의 크기가 $(3+2N) \times (3+2N)$이므로, 각 갱신에 $O(N^2)$의 비용이 듭니다. 수천 개의 랜드마크가 있으면 이는 감당하기 어려워집니다.
- **선형화 오차(Linearization errors)**: 모든 EKF와 마찬가지로 1차 근사가 비일관성(inconsistency)을 초래할 수 있으며, 특히 긴 궤적 이후에 두드러집니다.
- **데이터 연관(Data association)**: 관측을 랜드마크에 올바르게 매칭하는 것이 중요하고 어렵습니다. 특히 반복적인 특징이 많은 환경에서 더욱 그렇습니다. 매칭에는 유클리드 거리(Euclidean distance)보다 마할라노비스 거리(Mahalanobis distance, $d_M = \sqrt{(\mathbf{z} - \hat{\mathbf{z}})^T S^{-1} (\mathbf{z} - \hat{\mathbf{z}})}$)가 선호되는데, 이는 혁신(innovation)의 공분산 구조를 고려하기 때문입니다. 예를 들어, 거리 센서 노이즈가 0.1 m일 때의 1 m 오차는 10 m 노이즈일 때보다 훨씬 중요합니다. 유클리드 거리는 모든 차원을 동등하게 취급하고 불확실성을 무시하므로, 센서 노이즈가 비등방적(anisotropic)일 때 잘못된 연관을 초래합니다.

---

## 3. 그래프 기반 SLAM(Graph-Based SLAM)

### 3.1 그래프 최적화로서의 SLAM

그래프 기반 SLAM은 문제를 **인수 그래프(factor graph)** 최적화로 재구성합니다:

- **노드(Nodes)**: 로봇 자세 $\mathbf{x}_0, \mathbf{x}_1, \ldots, \mathbf{x}_T$ 및 랜드마크 위치 $\mathbf{m}_1, \ldots, \mathbf{m}_N$
- **엣지(Edges)**: 오도메트리로부터의 제약(연속 자세 간) 및 관측으로부터의 제약(자세와 랜드마크 간)

```
x0 ──odometry── x1 ──odometry── x2 ──odometry── x3
 \              / \               |               /
  \landmark   /   \landmark      |landmark      /
   \        /      \             |             / loop closure
    m1              m2           m3          x0
```

각 엣지는 불확실성과 함께 측정값을 인코딩합니다. 최적화는 모든 제약을 동시에 가장 잘 만족하는 노드 위치를 찾습니다.

### 3.2 자세 그래프 최적화(Pose Graph Optimization)

**자세 그래프 SLAM(pose graph SLAM)**에서는 랜드마크를 제거하고 로봇 자세만 노드로 사용합니다. 엣지는 다음으로부터 생성됩니다:

- **오도메트리(Odometry)**: 연속 자세 간의 상대 움직임
- **루프 폐쇄(Loop closures)**: 비연속 자세 간의 상대 자세 제약 (로봇이 이전 위치를 재방문할 때 감지)

최적화는 다음을 최소화합니다:

$$\mathbf{x}^* = \arg\min_\mathbf{x} \sum_{(i,j) \in \text{edges}} \| h(\mathbf{x}_i, \mathbf{x}_j) - \mathbf{z}_{ij} \|^2_{\Omega_{ij}}$$

여기서 $\mathbf{z}_{ij}$는 측정된 상대 변환, $h(\mathbf{x}_i, \mathbf{x}_j)$는 예상 상대 변환, $\Omega_{ij}$는 정보 행렬(information matrix, 공분산의 역행렬)입니다.

### 3.3 가우스-뉴턴 최적화(Gauss-Newton Optimization)

자세 그래프 SLAM의 표준 솔버는 **가우스-뉴턴(Gauss-Newton)** 또는 **레벤버그-마르콰르트(Levenberg-Marquardt)**입니다:

$$\Delta\mathbf{x}^* = -(J^T \Omega J)^{-1} J^T \Omega \mathbf{e}$$

여기서 $J$는 오차 함수의 야코비안이고, $\mathbf{e}$는 오차 벡터입니다.

행렬 $H = J^T \Omega J$ (헤시안 근사)는 각 엣지가 두 개의 노드만 관련되므로 **희소(sparse)**합니다. 이 희소성 덕분에 희소 선형 대수(희소 촐레스키 분해)를 사용한 효율적인 풀이가 가능합니다.

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

### 3.4 그래프 기반 SLAM의 장점

- **전역 최적화(Global optimization)**: 루프 폐쇄가 감지되면 전체 궤적에 걸쳐 오차를 분산시킴
- **희소성(Sparsity)**: 헤시안 행렬이 희소하므로 효율적인 솔버 사용 가능 (유리한 경우 희소 촐레스키에서 $O(n)$)
- **모듈성(Modularity)**: 프론트엔드(데이터 연관, 루프 폐쇄 감지)와 백엔드(최적화)가 분리됨
- **유연성(Flexible)**: 모든 종류의 측정 유형(오도메트리, GPS, 랜드마크, 상대 자세)을 처리 가능

---

## 4. 파티클 필터 SLAM — FastSLAM

### 4.1 핵심 통찰: 조건부 독립성(Conditional Independence)

FastSLAM은 **로봇 궤적이 주어지면 랜드마크 추정이 독립적**이라는 사실을 활용합니다. 이를 통해 다음과 같이 인수분해할 수 있습니다:

$$p(\mathbf{x}_{0:T}, \mathbf{m} | \mathbf{z}, \mathbf{u}) = p(\mathbf{x}_{0:T} | \mathbf{z}, \mathbf{u}) \prod_{j=1}^{N} p(\mathbf{m}_j | \mathbf{x}_{0:T}, \mathbf{z}, \mathbf{u})$$

**FastSLAM은 다음을 사용합니다**:
- 로봇 궤적 $\mathbf{x}_{0:T}$에 대한 **파티클 필터(particle filter)**
- 각 파티클 내에서 각 랜드마크 $\mathbf{m}_j$에 대한 **독립 EKF(independent EKF)**

### 4.2 복잡도(Complexity)

- 각 파티클은 하나의 큰 $(3+2N) \times (3+2N)$ EKF 대신 $N$개의 작은 (2x2) EKF를 유지
- 파티클당 갱신 비용: $O(N^2)$ 대신 $O(N)$
- 전체 비용: $O(MN)$, 여기서 $M$은 파티클 수

실제로 FastSLAM은 랜드마크를 EKF를 통해 분석적으로 추정하므로, 결합 상태에 대한 순수 파티클 필터보다 훨씬 적은 파티클이 필요합니다.

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

        # 측정 우도(measurement likelihood)를 기반으로 파티클 가중치 갱신.
        # 리샘플링(resampling)이 필요한 이유 (모든 갱신 후 외부에서 수행):
        # 리샘플링 없이는 파티클 가중치가 시간이 지남에 따라 발산하여 소수의
        # 파티클만 거의 모든 가중치를 축적하고 나머지는 무시할 수 있게 된다.
        # 이 "파티클 퇴화(particle degeneracy)"는 무관한 가설에 계산을 낭비한다.
        # 리샘플링은 높은 가중치 파티클을 복제하고 낮은 가중치 파티클을 제거하여,
        # 파티클 집합을 그럴듯한 궤적에 집중시킨다.
        self.weight *= np.exp(-0.5 * innovation.T @ np.linalg.inv(S) @ innovation)
        self.weight /= np.sqrt(np.linalg.det(2 * np.pi * S))
```

---

## 5. 시각적 SLAM(Visual SLAM)

### 5.1 특징 기반 시각적 SLAM(Feature-Based Visual SLAM)

**특징 기반 방법(feature-based methods)**은 프레임 간에 시각적 특징(모서리, 점)을 추출하고 추적합니다:

1. **특징 감지(Feature detection)**: 뚜렷한 점을 찾음 (ORB, SIFT, SURF)
2. **특징 기술(Feature description)**: 각 특징에 대한 기술자(descriptor) 벡터 계산
3. **특징 매칭(Feature matching)**: 기술자 유사도를 사용하여 프레임 간 특징 매칭
4. **자세 추정(Pose estimation)**: 매칭된 특징으로부터 카메라 움직임 계산 (필수 행렬(essential matrix), PnP)
5. **삼각측량(Triangulation)**: 매칭된 특징의 3D 위치 추정
6. **번들 조정(Bundle adjustment)**: 카메라 자세와 3D 점을 공동 최적화

**ORB-SLAM**(및 후속작 ORB-SLAM3)은 가장 잘 알려진 특징 기반 시각적 SLAM 시스템입니다:

```
Input Image → Feature Extraction (ORB) → Tracking (match to local map)
     ↓                                       ↓
 Relocalization                        Local Mapping
     ↑                                       ↓
Loop Closing ←── Place Recognition ←── Keyframe Selection
     ↓
 Pose Graph Optimization
```

핵심 구성요소:
- **추적 스레드(Tracking thread)**: 각 새 프레임을 로컬 지도에 대해 위치 추정
- **로컬 매핑 스레드(Local mapping thread)**: 새로운 지도 점을 삼각측량하고 로컬 번들 조정 수행
- **루프 폐쇄 스레드(Loop closing thread)**: 시각 단어 사전(bag-of-words) 장소 인식 시스템으로 루프 폐쇄를 감지하고 전역 최적화 수행

### 5.2 직접 방식 시각적 SLAM(Direct Visual SLAM)

**직접 방식(direct methods)** (예: LSD-SLAM, DSO)은 특징을 추출하는 대신 픽셀 밝기값을 직접 사용합니다:

$$E_{photo} = \sum_{\mathbf{p} \in \mathcal{P}} \| I_{ref}(\mathbf{p}) - I_{cur}(\pi(T \cdot \pi^{-1}(\mathbf{p}, d_\mathbf{p}))) \|^2$$

여기서 광도 오차(photometric error)는 추정된 카메라 움직임 $T$와 깊이 지도(depth map) $d$ 아래에서 참조 영상 $I_{ref}$와 현재 영상 $I_{cur}$가 얼마나 잘 일치하는지 측정합니다.

**특징 기반 vs. 직접 방식**:

| 측면 | 특징 기반(ORB-SLAM) | 직접 방식(LSD-SLAM, DSO) |
|------|---------------------|--------------------------|
| 입력 | 특징 점 | 원시 픽셀 밝기 |
| 지도 밀도 | 희소(특징 점) | 준밀집 또는 밀집 |
| 텍스처 요구 | 모서리/엣지 필요 | 밝기 기울기(intensity gradient) 필요 |
| 속도 | 빠른 특징 추출 | 느림(픽셀별 최적화) |
| 강건성 | 중간 정도의 블러 허용 | 광도 변화에 민감 |
| 초기화 | 충분한 시차(parallax) 필요 | 텍스처 필요 |

### 5.3 현대적인 시각적 SLAM 시스템

**ORB-SLAM3** (2021):
- 단안(monocular), 스테레오(stereo), RGB-D, IMU 지원
- 다중 지도 시스템: 추적 실패 시 새 지도를 생성하고 나중에 병합
- 시각-관성 융합(visual-inertial fusion)으로 강건한 상태 추정

**RTAB-Map** (Real-Time Appearance-Based Mapping, 실시간 외관 기반 매핑):
- LiDAR, 스테레오, RGB-D 지원
- 메모리 관리: 대규모 환경을 처리하기 위해 단기 및 장기 메모리 유지
- 루프 폐쇄를 포함한 그래프 기반 최적화

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

## 6. 루프 폐쇄(Loop Closure)

### 6.1 루프 폐쇄가 중요한 이유

루프 폐쇄가 없으면 오도메트리 드리프트(drift)가 무한정 누적됩니다. 100미터를 이동한 후 로봇의 위치 오차가 1~5미터에 달할 수 있습니다. 로봇이 이전에 방문한 위치로 돌아올 때, 이 **재방문**을 감지하면 전체 궤적을 수정할 수 있습니다.

### 6.2 장소 인식(Place Recognition)

루프 폐쇄 감지는 근본적으로 **장소 인식(place recognition)** 문제입니다. 현재 센서 관측을 기반으로, 이 장소를 전에 본 적이 있는가?

**시각적 장소 인식(visual place recognition)** 접근법:

1. **시각 단어 사전(Bag of Visual Words, BoVW)**: 특징 기술자를 시각 어휘(visual vocabulary)로 클러스터링합니다. 각 이미지를 시각 단어의 히스토그램으로 표현합니다. TF-IDF 유사도를 사용하여 히스토그램을 비교합니다.

2. **신경망 임베딩(Neural network embeddings)**: CNN을 사용하여 각 이미지의 전역 기술자(global descriptor)를 계산합니다. 코사인 유사도로 기술자를 비교합니다. 현대 방법(NetVLAD, SuperGlue)은 인상적인 성능을 달성합니다.

3. **점군 기술자(Point cloud descriptors)**: LiDAR의 경우, 점군의 전역 기술자를 계산합니다 (Scan Context, PointNetVLAD).

### 6.3 루프 폐쇄 검증(Loop Closure Verification)

잘못된 루프 폐쇄(서로 다른 두 장소를 잘못 매칭)는 치명적입니다. 지도를 돌이킬 수 없이 손상시킵니다. 따라서 엄격한 검증이 필수적입니다:

1. **기하학적 검증(Geometric verification)**: 후보 매칭을 찾은 후, 특징 간 기하학적 관계가 일관성이 있는지 검증합니다 (기본 행렬(fundamental matrix) 계산, 인라이어(inlier) 비율 확인).
2. **시간적 일관성(Temporal consistency)**: 수락하기 전에 여러 개의 연속적인 루프 폐쇄 후보가 필요합니다.
3. **강건한 최적화(Robust optimization)**: 백엔드 최적화기에서 강건한 비용 함수(Huber, Cauchy)를 사용하여 잠재적으로 잘못된 루프 폐쇄의 가중치를 낮춥니다.

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

## 7. SLAM 프론트엔드와 백엔드(SLAM Frontends and Backends)

### 7.1 아키텍처 패턴(Architecture Pattern)

현대 SLAM 시스템은 **프론트엔드-백엔드(frontend-backend)** 아키텍처를 따릅니다:

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

**인기 있는 백엔드**:
- **g2o**: 일반 그래프 최적화 (C++)
- **GTSAM**: 증분 솔버를 포함한 인수 그래프 라이브러리 (iSAM2)
- **Ceres Solver**: 일반 비선형 최소제곱법 (Google)

### 7.2 SLAM 접근법 비교(Comparison of SLAM Approaches)

| 방법 | 복잡도 | 지도 유형 | 장점 | 단점 |
|------|--------|----------|------|------|
| EKF-SLAM | $O(N^2)$ | 랜드마크 | 단순하고 잘 이해됨 | 확장성 나쁨, 선형화 오차 |
| Graph-SLAM | 분할상환 $O(N)$ | 모든 유형 | 전역 최적, 확장 가능 | 배치식(완전 온라인 아님) |
| FastSLAM | $O(M \log N)$ | 랜드마크 | 데이터 연관 불확실성 처리 | 파티클 퇴화 |
| ORB-SLAM3 | 실시간 | 희소 3D | 강건하고 다중 센서 지원 | 텍스처 필요 |
| LiDAR-SLAM | 실시간 | 점군 | 정확하고 조명 불변 | 고가의 센서 |

---

## 요약

| 개념 | 핵심 아이디어 |
|------|--------------|
| SLAM 문제 | 로봇 자세를 추정하고 환경 지도를 동시에 구축 |
| EKF-SLAM | 로봇 자세와 랜드마크에 대한 결합 가우시안; $O(N^2)$ |
| 랜드마크 상관관계 | 하나의 랜드마크 관측이 공유 불확실성을 통해 다른 랜드마크 추정값 개선 |
| 그래프 기반 SLAM | 자세 그래프에 대한 비선형 최적화로 공식화; 희소하고 확장 가능 |
| 루프 폐쇄 | 재방문 장소 감지; 전역 오차 수정 가능 |
| FastSLAM | 궤적에 파티클 필터 + 랜드마크당 EKF; 조건부 독립성 활용 |
| 시각적 SLAM | 카메라 기반 SLAM; 특징 기반(ORB-SLAM) 또는 직접 방식(LSD-SLAM) |
| 프론트엔드-백엔드 | 인식(특징 추출, 매칭)과 최적화(그래프 풀이) 분리 |

---

## 연습문제

1. **EKF-SLAM 시뮬레이션**: 위의 EKF-SLAM 클래스를 구현하세요. 20x20 m 환경에 10개의 랜드마크를 배치하세요. 로봇이 정사각형 경로로 주행하면서 8 m 범위 내의 랜드마크를 관측하도록 시뮬레이션하세요. 시간에 따른 추정 및 실제 랜드마크 위치를 그래프로 표시하세요. 로봇이 랜드마크를 재방문할수록 각 랜드마크의 불확실성 타원이 어떻게 변하는지 확인하세요.

2. **자세 그래프 최적화**: 루프(예: 정사각형 궤적)를 도는 로봇의 자세 그래프를 만드세요. 작은 가우시안 노이즈가 있는 오도메트리 엣지를 추가하세요. 그런 다음 마지막 자세와 첫 번째 자세를 연결하는 루프 폐쇄 엣지를 추가하세요. 최적화기를 실행하고 최적화 전후의 궤적을 비교하세요. 루프 폐쇄 엣지가 누적된 드리프트를 어떻게 재분배하는지 확인하세요.

3. **데이터 연관 도전**: EKF-SLAM을 수정하여 최근접 이웃 데이터 연관(각 관측을 가장 가까운 기존 랜드마크에 매칭)을 사용하세요. 두 랜드마크가 서로 가까운 시나리오를 만드세요. 잘못된 데이터 연관이 어떻게 지도를 손상시킬 수 있는지 보여주세요. 그런 다음 간단한 마할라노비스 거리(Mahalanobis distance) 게이팅 전략을 구현하고 강건성이 향상됨을 확인하세요.

4. **시각적 특징 매칭**: Python(OpenCV)을 사용하여 약간 다른 시점에서 찍은 두 이미지에서 ORB 특징을 감지하세요. Lowe의 비율 테스트를 적용한 브루트-포스 매칭으로 특징을 매칭하세요. 필수 행렬(essential matrix)을 추정하고 상대 카메라 자세를 복원하세요. 매칭된 점을 삼각측량하여 희소 3D 재구성을 만드세요.

5. **루프 폐쇄 효과**: 로봇이 긴 복도를 이동하다(50개 자세) 시작점으로 돌아오는 자세 그래프 SLAM 시나리오를 만드세요. 먼저 루프 폐쇄 없이 최적화를 실행하세요. 그런 다음 루프 폐쇄 엣지를 추가하고 재최적화하세요. (a) 최종 위치 오차와 (b) 전체 궤적 오차 측면에서 개선 정도를 정량화하세요. 루프 폐쇄 측정의 품질이 결과에 어떻게 영향을 미치는지 확인하세요.

---

## 참고 문헌

- Thrun, S., Burgard, W., and Fox, D. *Probabilistic Robotics*. MIT Press, 2005. Chapters 10-13. (SLAM의 기초 참고문헌)
- Cadena, C. et al. "Past, Present, and Future of Simultaneous Localization and Mapping." *IEEE Transactions on Robotics*, 2016. (포괄적인 SLAM 개요)
- Grisetti, G. et al. "A Tutorial on Graph-Based SLAM." *IEEE Intelligent Transportation Systems Magazine*, 2010. (훌륭한 그래프 SLAM 튜토리얼)
- Campos, C. et al. "ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial, and Multi-Map SLAM." *IEEE Transactions on Robotics*, 2021.
- Montemerlo, M. et al. "FastSLAM: A Factored Solution to the Simultaneous Localization and Mapping Problem." *AAAI*, 2002.

---

[← 이전: 상태 추정과 필터링](11_State_Estimation.md) | [다음: ROS2 기초 →](13_ROS2_Fundamentals.md)
