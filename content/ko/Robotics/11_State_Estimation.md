# 11. 상태 추정과 필터링(State Estimation and Filtering)

[← 이전: 센서와 인식](10_Sensors_and_Perception.md) | [다음: SLAM →](12_SLAM.md)

---

## 학습 목표

1. 상태 추정(state estimation) 문제를 확률적 추론(probabilistic inference) 과제로 정식화한다
2. 칼만 필터(Kalman Filter)의 예측-갱신(predict-update) 사이클을 유도하고 선형 시스템에 구현한다
3. 확장 칼만 필터(Extended Kalman Filter, EKF)를 이용해 비선형 시스템으로 확장한다
4. 무향 칼만 필터(Unscented Kalman Filter, UKF)를 EKF의 미분 불필요 대안으로 이해한다
5. 비가우시안(non-Gaussian)·고비선형 추정 문제에 파티클 필터(particle filter)를 구현한다
6. 로봇 위치 추정을 위한 KF, EKF, UKF, 파티클 필터의 절충점을 비교한다

---

이전 레슨에서 로봇이 자신과 환경에 대한 정보를 얻는 센서를 살펴봤다. 그러나 원시 센서 데이터는 잡음이 있고 불완전하며, 서로 다른 소스에서 서로 다른 주기로 도착한다. 하나의 엔코더 판독값은 1밀리초 전에 관절이 어디 있었는지만 알려주지만, 우리에게 실제로 필요한 것은 로봇이 *지금 이 순간* 어디 있는지에 대한 최선의 추정값이다 — 로봇이 움직이는 물리 법칙, 최신 센서 판독값, 그리고 그 둘에 대한 불확실성을 모두 결합해서.

상태 추정(state estimation)은 이 질문에 최적으로 답하기 위한 수학적 프레임워크다. 사실상 모든 자율 시스템의 핵심이다. 자율 주행 자동차, 드론, 이동 로봇, 우주선 모두 상태 추정기에 의존해 실시간으로 정확한 상태 인식을 유지한다. 상태 추정 없이는 센서 잡음과 모델 불확실성으로 인해 자율 동작이 불가능할 것이다.

> **비유**: 칼만 필터는 현명한 조언자 같다 — 당신의 예측(경험)과 새로운 측정값(증거)을 결합해 최선의 추정치를 제공한다. 안개 낀 도로를 운전하는 상황을 상상해보자. 내부 모델은 "나는 지금쯤 다리 근처에 있을 것"이라고 말한다(예측). 그런데 안개 사이로 표지판을 잠깐 발견한다(측정). 조언자는 둘을 저울질한다. "당신의 예측은 꽤 신뢰할 만하고, 표지판도 꽤 선명하네 — 아마 다리에서 30미터 앞에 있을 거야." 안개가 짙을수록(잡음 많은 측정) 조언자는 예측을 더 신뢰한다. 추측이 불안정했다면(불확실한 모델) 조언자는 표지판을 더 신뢰한다.

---

## 1. 상태 추정 문제

### 1.1 문제 정식화

시간 $k$에서 **상태 벡터(state vector)** $\mathbf{x}_k$를 추정하려 한다. 주어진 것:

1. **과정 모델(process model)** (상태가 어떻게 변화하는가):
$$\mathbf{x}_k = f(\mathbf{x}_{k-1}, \mathbf{u}_{k-1}) + \mathbf{w}_{k-1}$$

2. **측정 모델(measurement model)** (센서가 상태를 어떻게 관측하는가):
$$\mathbf{z}_k = h(\mathbf{x}_k) + \mathbf{v}_k$$

여기서:
- $\mathbf{u}_{k-1}$은 제어 입력(control input)
- $\mathbf{w}_{k-1} \sim \mathcal{N}(\mathbf{0}, Q)$는 과정 잡음(process noise)
- $\mathbf{v}_k \sim \mathcal{N}(\mathbf{0}, R)$은 측정 잡음(measurement noise)
- $Q$는 과정 잡음 공분산(process noise covariance)
- $R$은 측정 잡음 공분산(measurement noise covariance)

### 1.2 베이즈 추정 프레임워크

상태 추정은 근본적으로 **베이즈 추론(Bayesian inference)** 문제다. 상태에 대한 확률 분포인 **믿음(belief)**을 유지한다:

$$\text{bel}(\mathbf{x}_k) = p(\mathbf{x}_k | \mathbf{z}_{1:k}, \mathbf{u}_{0:k-1})$$

갱신은 두 단계로 베이즈 규칙을 따른다:

**예측(prediction)** (시간 갱신):
$$\overline{\text{bel}}(\mathbf{x}_k) = \int p(\mathbf{x}_k | \mathbf{x}_{k-1}, \mathbf{u}_{k-1}) \text{bel}(\mathbf{x}_{k-1}) \, d\mathbf{x}_{k-1}$$

**보정(correction)** (측정 갱신):
$$\text{bel}(\mathbf{x}_k) = \eta \, p(\mathbf{z}_k | \mathbf{x}_k) \, \overline{\text{bel}}(\mathbf{x}_k)$$

여기서 $\eta$는 정규화 상수다.

믿음을 다르게 표현하면 서로 다른 필터가 만들어진다:

| 필터 | 믿음 표현 | 선형성 | 분포 형태 |
|------|-----------|--------|----------|
| 칼만 필터(Kalman Filter) | 가우시안 ($\mu, \Sigma$) | 선형 | 단봉형(Unimodal) |
| EKF | 가우시안 ($\mu, \Sigma$) | 비선형(선형화) | 단봉형 |
| UKF | 가우시안 ($\mu, \Sigma$) | 비선형(시그마 포인트) | 단봉형 |
| 파티클 필터(Particle Filter) | 가중 샘플 | 임의 | 다봉형(Multimodal) |

---

## 2. 칼만 필터(Kalman Filter)

### 2.1 선형 시스템 가정

칼만 필터는 **가우시안(Gaussian)** 잡음을 갖는 **선형(linear)** 모델을 가정한다:

$$\mathbf{x}_k = A \mathbf{x}_{k-1} + B \mathbf{u}_{k-1} + \mathbf{w}_{k-1}$$
$$\mathbf{z}_k = H \mathbf{x}_k + \mathbf{v}_k$$

여기서 $A$는 상태 전이 행렬(state transition matrix), $B$는 제어 입력 행렬(control input matrix), $H$는 관측 행렬(observation matrix)이다.

이 가정 하에서 믿음은 항상 가우시안을 유지하며, 평균 $\hat{\mathbf{x}}_k$와 공분산 $P_k$로 완전히 특성화된다.

### 2.2 예측-갱신 사이클

**예측 단계** (상태와 불확실성을 앞으로 전파):

$$\hat{\mathbf{x}}_k^- = A \hat{\mathbf{x}}_{k-1} + B \mathbf{u}_{k-1}$$
$$P_k^- = A P_{k-1} A^T + Q$$

**갱신 단계** (측정값 반영):

$$K_k = P_k^- H^T (H P_k^- H^T + R)^{-1}$$
$$\hat{\mathbf{x}}_k = \hat{\mathbf{x}}_k^- + K_k (\mathbf{z}_k - H \hat{\mathbf{x}}_k^-)$$
$$P_k = (I - K_k H) P_k^-$$

여기서 $K_k$는 **칼만 이득(Kalman gain)** — 예측과 측정 사이의 최적 가중치다.

### 2.3 칼만 이득 이해하기

칼만 이득 $K_k$는 아름다운 해석을 갖는다:

- $R$이 작을 때(정확한 센서): $K_k \to H^{-1}$ — 측정값을 신뢰
- $P_k^-$가 작을 때(정확한 예측): $K_k \to 0$ — 예측을 신뢰
- $R$이 클 때(잡음 많은 센서): $K_k \to 0$ — 측정값을 무시
- $P_k^-$가 클 때(불확실한 예측): $K_k \to H^{-1}$ — 측정값에 의존

**이노베이션(innovation)** (또는 잔차(residual)) $\boldsymbol{\nu}_k = \mathbf{z}_k - H \hat{\mathbf{x}}_k^-$는 "놀라움"을 나타낸다 — 측정값이 예측과 얼마나 다른지. 칼만 이득은 측정값과 예측에 대한 상대적 신뢰도에 따라 이 놀라움을 스케일링한다.

### 2.4 구현

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

### 2.5 예제: 1D 로봇 위치 추적

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

## 3. 확장 칼만 필터(Extended Kalman Filter, EKF)

### 3.1 비선형성 처리

대부분의 로봇 시스템은 비선형이다. 예를 들어, 차동 구동 로봇(differential-drive robot)의 과정 모델은 다음과 같다:

$$\begin{bmatrix} x_{k+1} \\ y_{k+1} \\ \theta_{k+1} \end{bmatrix} = \begin{bmatrix} x_k + v_k \cos\theta_k \cdot \Delta t \\ y_k + v_k \sin\theta_k \cdot \Delta t \\ \theta_k + \omega_k \cdot \Delta t \end{bmatrix}$$

이는 $\theta$에 대해 명백히 비선형이다. 표준 칼만 필터는 선형 $A$와 $H$ 행렬을 요구하므로 직접 적용할 수 없다.

**확장 칼만 필터(EKF)**는 현재 추정값 주변에서 야코비안(Jacobian)을 이용해 비선형 함수 $f$와 $h$를 선형화한다:

$$F_k = \frac{\partial f}{\partial \mathbf{x}} \bigg|_{\hat{\mathbf{x}}_{k-1}, \mathbf{u}_{k-1}}$$
$$H_k = \frac{\partial h}{\partial \mathbf{x}} \bigg|_{\hat{\mathbf{x}}_k^-}$$

### 3.2 EKF 알고리즘

**예측**:
$$\hat{\mathbf{x}}_k^- = f(\hat{\mathbf{x}}_{k-1}, \mathbf{u}_{k-1})$$
$$P_k^- = F_k P_{k-1} F_k^T + Q$$

**갱신**:
$$K_k = P_k^- H_k^T (H_k P_k^- H_k^T + R)^{-1}$$
$$\hat{\mathbf{x}}_k = \hat{\mathbf{x}}_k^- + K_k (\mathbf{z}_k - h(\hat{\mathbf{x}}_k^-))$$
$$P_k = (I - K_k H_k) P_k^-$$

참고: 비선형 함수 $f$와 $h$는 상태 전파와 기대 측정값 계산에 사용되지만, 공분산 전파에는 선형화된 $F_k$와 $H_k$가 사용된다.

### 3.3 이동 로봇 위치 추정 구현

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

### 3.4 EKF의 한계

EKF는 잘 알려진 한계를 가진다:

1. **1차 근사**: 야코비안 선형화는 현재 추정값 근처에서 비선형성이 완만할 때만 정확하다. 매우 비선형인 시스템(예: 방위각 전용 측정)에서는 EKF가 발산할 수 있다.
2. **가우시안 가정**: EKF는 다봉형(multimodal) 분포를 표현할 수 없다(예: "로봇이 위치 A 또는 위치 B에 있을 수 있다").
3. **야코비안 계산**: 복잡한 모델에서 오류가 발생하기 쉬운 해석적 또는 수치적 야코비안이 필요하다.

---

## 4. 무향 칼만 필터(Unscented Kalman Filter, UKF)

### 4.1 야코비안 대신 시그마 포인트

**무향 칼만 필터(UKF)**는 선형화를 완전히 피한다. 야코비안을 계산하는 대신, 신중하게 선택된 **시그마 포인트(sigma points)** 집합을 비선형 함수에 통과시키고 변환된 포인트들에서 평균과 공분산을 복원한다.

$n$차원 상태에 대한 **무향 변환(Unscented Transform)**:

1. $2n + 1$개의 시그마 포인트 선택:
$$\boldsymbol{\chi}_0 = \hat{\mathbf{x}}$$
$$\boldsymbol{\chi}_i = \hat{\mathbf{x}} + \left(\sqrt{(n + \lambda) P}\right)_i, \quad i = 1, \ldots, n$$
$$\boldsymbol{\chi}_{n+i} = \hat{\mathbf{x}} - \left(\sqrt{(n + \lambda) P}\right)_i, \quad i = 1, \ldots, n$$

여기서 $\lambda = \alpha^2(n + \kappa) - n$은 스케일링 파라미터이고, $\left(\sqrt{(n+\lambda)P}\right)_i$는 행렬 제곱근의 $i$번째 열이다.

2. 각 시그마 포인트를 비선형 함수에 통과:
$$\boldsymbol{\gamma}_i = f(\boldsymbol{\chi}_i)$$

3. 변환된 포인트들에서 평균과 공분산 복원:
$$\hat{\mathbf{x}}^- = \sum_{i=0}^{2n} w_i^{(m)} \boldsymbol{\gamma}_i$$
$$P^- = \sum_{i=0}^{2n} w_i^{(c)} (\boldsymbol{\gamma}_i - \hat{\mathbf{x}}^-)(\boldsymbol{\gamma}_i - \hat{\mathbf{x}}^-)^T + Q$$

### 4.2 UKF가 EKF보다 나은 이유

UKF는 평균과 공분산을 **2차(second order)** 이상(가우시안 분포의 경우 그 이상)까지 정확히 포착하는 반면, EKF는 **1차(first order)**까지만 포착한다. 이 차이의 의미:

- 강한 비선형성에 대한 더 나은 처리
- 야코비안 계산 불필요(더 쉬운 구현, 해석적 미분 불필요)
- 보통 EKF와 유사한 계산 비용(중간 정도의 상태 차원에서)

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

## 5. 파티클 필터(Particle Filter)

### 5.1 몬테카를로 위치 추정

비가우시안 잡음이나 다봉형 분포를 갖는 고비선형 시스템에서, **파티클 필터(Sequential Monte Carlo)**는 가중 샘플(파티클) 집합으로 믿음을 표현한다:

$$\text{bel}(\mathbf{x}_k) \approx \sum_{i=1}^N w_k^{(i)} \delta(\mathbf{x}_k - \mathbf{x}_k^{(i)})$$

여기서 $\mathbf{x}_k^{(i)}$는 $i$번째 파티클이고 $w_k^{(i)}$는 그 가중치다.

### 5.2 알고리즘

1. **예측**: 과정 모델에 잡음을 추가하여 각 파티클을 전파:
$$\mathbf{x}_k^{(i)} \sim p(\mathbf{x}_k | \mathbf{x}_{k-1}^{(i)}, \mathbf{u}_{k-1})$$

2. **갱신**: 각 파티클이 측정값을 얼마나 잘 설명하는지로 가중치 부여:
$$w_k^{(i)} = p(\mathbf{z}_k | \mathbf{x}_k^{(i)})$$

3. **정규화**: $w_k^{(i)} \leftarrow w_k^{(i)} / \sum_j w_k^{(j)}$

4. **리샘플링(resampling)**: 파티클 퇴화(particle degeneracy)를 방지하기 위해 낮은 가중치의 파티클을 높은 가중치 파티클의 복사본으로 대체.

### 5.3 구현

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

### 5.4 완전한 위치 추정 예제

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

## 6. 필터 비교

### 6.1 절충점

| 필터 | 계산량 | 정확도(선형) | 정확도(비선형) | 다봉형 지원 | 구현 난이도 |
|------|--------|------------|--------------|------------|------------|
| KF | $O(n^3)$ | 최적 | 해당 없음(선형만) | 불가 | 단순 |
| EKF | $O(n^3)$ | 근사 최적 | 양호(완만한 비선형) | 불가 | 보통(야코비안 필요) |
| UKF | $O(n^3)$ | 근사 최적 | EKF보다 우수 | 불가 | 보통 |
| PF | $O(Nn^2)$ | 최적으로 수렴 | 우수 | 가능 | 개념은 단순, 튜닝 필요 |

여기서 $n$은 상태 차원, $N$은 파티클 수다.

### 6.2 어떤 필터를 선택할까

```
시스템이 선형인가?
├── 예 → 칼만 필터(최적, 가장 단순)
└── 아니오 → 분포가 단봉형인가?
    ├── 예 → 비선형성이 완만한가?
    │   ├── 예 → EKF(광범위하게 사용, 빠름)
    │   └── 아니오 → UKF(더 높은 정확도, 야코비안 불필요)
    └── 아니오(다봉형) → 파티클 필터(유일한 선택)
```

### 6.3 실용적 고려사항

**EKF는 로보틱스의 핵심 도구**다: 대부분의 SLAM 시스템, 드론 상태 추정기, 산업용 위치 추정 시스템이 EKF 또는 그 변형을 사용한다. 빠르고, 잘 이해되어 있으며, 대부분의 응용에 충분하다.

**파티클 필터**는 **전역 위치 추정(global localization)** ("납치된 로봇(kidnapped robot)" 문제 — 로봇이 초기 위치를 모르는 경우)에 필수적인데, 여러 가설을 동시에 유지할 수 있기 때문이다.

**UKF**는 오류가 발생하기 쉬운 야코비안 유도를 피할 수 있어 점점 인기를 얻고 있다. 현대 로보틱스 라이브러리(예: `filterpy`, `ukfm`)는 UKF 구현을 간단하게 만든다.

**센서 융합(sensor fusion)** 프레임워크는 종종 여러 필터를 실행하거나 조합한다:
- 빠른 연속 상태 추정을 위한 EKF (IMU + 오도메트리(odometry), 100+ Hz)
- 전역 위치 추정 및 추적 실패 복구를 위한 파티클 필터
- 오프라인 평활화(smoothing)를 위한 인자 그래프(factor graph) 최적화 (SLAM 레슨에서 다룸)

---

## 7. 다중 주기 센서 융합(Multi-Rate Sensor Fusion)

### 7.1 비동기 측정 문제

실제로 센서는 서로 다른 주기로 도착한다:

| 센서 | 주기 | 지연 |
|------|------|------|
| IMU | 200-1000 Hz | < 1 ms |
| 휠 엔코더 | 50-100 Hz | < 5 ms |
| LiDAR | 10-20 Hz | 50-100 ms |
| 카메라 | 15-60 Hz | 30-100 ms |
| GPS | 1-10 Hz | 100-500 ms |

해결책은 가장 높은 센서 주기(IMU)에서 **예측**을 실행하고, 어느 센서에서든 새 측정값이 도착할 때마다 **갱신**을 수행하는 것이다:

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

## 요약

| 개념 | 핵심 아이디어 |
|------|-------------|
| 상태 추정(state estimation) | 확률 모델을 이용해 잡음 있는 관측으로부터 숨겨진 상태를 추론 |
| 베이즈 필터(Bayes filter) | 재귀적 예측-갱신 프레임워크; 다양한 믿음 표현 = 다양한 필터 |
| 칼만 필터(Kalman filter) | 선형 가우시안 시스템에 최적; 평균과 공분산 추적 |
| 칼만 이득(Kalman gain) | 불확실성에 기반한 예측과 측정 사이의 최적 혼합 |
| EKF | 비선형 모델의 1차 선형화; 로보틱스에서 가장 널리 사용 |
| UKF | 시그마 포인트 전파로 야코비안 회피; 강한 비선형성에 더 적합 |
| 파티클 필터(particle filter) | 가중 샘플 표현; 임의 분포와 비선형성 처리 가능 |
| 리샘플링(resampling) | 높은 가중치 파티클을 복제하여 파티클 퇴화 방지 |
| 다중 주기 융합(multi-rate fusion) | 가장 높은 센서 주기로 예측; 각 센서마다 비동기 갱신 |

---

## 연습 문제

1. **칼만 필터 구현**: 등속 운동 로봇을 추적하는 1D 칼만 필터를 구현하라. 상태는 $[x, \dot{x}]$이다. 무작위 가속도로 실제 궤적을 생성하고, 잡음 있는 위치 측정(GPS 유사, 1 Hz, $\sigma = 2$ m)을 시뮬레이션하고, KF를 실행하라. 실제 궤적, 측정값, KF 추정값을 $\pm 2\sigma$ 신뢰 구간과 함께 플롯하라.

2. **차동 구동 로봇을 위한 EKF**: 위의 EKF 위치 추정 클래스를 구현하라. 6개의 알려진 랜드마크가 있는 환경에서 원을 그리며 이동하는 로봇을 시뮬레이션하라. 잡음이 있는 거리-방위각 측정을 생성하라($\sigma_r = 0.5$ m, $\sigma_\phi = 5°$). 실제 경로, 오도메트리 단독 경로(데드 레코닝(dead reckoning)), EKF 추정 경로를 플롯하라. 가시 랜드마크 수가 정확도에 어떤 영향을 미치는가?

3. **파티클 필터 전역 위치 추정**: 파티클 필터를 구현하고 전역 위치 추정을 시연하라: 파티클을 20m × 20m 환경에 균일하게 분포시켜 시작하라. 로봇이 3개의 랜드마크를 연속으로 관측한다. 파티클 구름이 균일 분포에서 실제 위치 주변의 집중된 군집으로 수렴하는 과정을 보여라.

4. **필터 비교**: 동일한 비선형 위치 추정 문제에 대해 EKF, UKF, 파티클 필터를 나란히 실행하라. 비교: (a) 시간에 따른 추정 오차, (b) 단계당 계산 시간, (c) 실제 상태에서 멀리 초기화된 경우의 동작. 각 필터는 어떤 조건에서 실패하는가?

5. **다중 주기 융합**: 다음을 융합하는 다중 주기 추정기를 구현하라: 100 Hz의 IMU(가속도 + 각속도), 50 Hz의 휠 오도메트리, 1 Hz의 시뮬레이션 GPS. GPS가 있을 때와 30초 동안 GPS가 없을 때의 추정 품질을 비교하라. GPS 없이 추정이 얼마나 빨리 열화되는가?

---

## 추가 자료

- Thrun, S., Burgard, W., and Fox, D. *Probabilistic Robotics*. MIT Press, 2005. (로보틱스 추정의 결정판 참고서)
- Kalman, R. E. "A New Approach to Linear Filtering and Prediction Problems." *ASME Journal of Basic Engineering*, 1960. (원본 칼만 필터 논문)
- Julier, S. and Uhlmann, J. "Unscented Filtering and Nonlinear Estimation." *Proceedings of the IEEE*, 2004. (UKF 튜토리얼)
- Doucet, A. et al. *Sequential Monte Carlo Methods in Practice*. Springer, 2001. (파티클 필터링 참고서)
- Sola, J. "Quaternion Kinematics for the Error-State Kalman Filter." arXiv:1711.02508, 2017. (IMU 기반 추정에 필수)

---

[← 이전: 센서와 인식](10_Sensors_and_Perception.md) | [다음: SLAM →](12_SLAM.md)
