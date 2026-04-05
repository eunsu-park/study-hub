# 14. 다중 센서 융합(Multi-Sensor Fusion)

**이전**: [Zigbee와 Z-Wave](./13_Zigbee_ZWave.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 여러 센서를 결합했을 때 단일 센서보다 더 정확하고 신뢰할 수 있는 결과가 나오는 이유를 설명할 수 있다
2. 동적 모델과 잡음이 있는 센서 측정값을 융합하기 위한 칼만 필터(Kalman Filter)를 구현할 수 있다
3. 자세 추정을 위해 가속도계와 자이로스코프 데이터를 결합하는 보완 필터(Complementary Filter)를 적용할 수 있다
4. 확장 칼만 필터(Extended Kalman Filter)를 사용하여 IMU(Inertial Measurement Unit, 관성 측정 장치)를 위한 센서 융합 아키텍처를 설계할 수 있다
5. 다중 센서 IoT 시스템에서 중앙 집중식(Centralized), 분산식(Decentralized), 계층적(Distributed) 융합 아키텍처를 비교할 수 있다

---

온도계 하나는 드리프트(drift)가 생기고, 가속도계 하나는 오차가 누적되며, GPS 수신기 하나는 실내에서 신호를 잃는다. 하지만 온도계에 습도 센서와 기압 센서를 결합하거나, 가속도계에 자이로스코프와 자력계를 융합하면 결과는 어떤 단일 측정값보다 훨씬 정확해진다. 이것이 센서 융합(Sensor Fusion)이다 -- 불완전한 여러 측정값으로 하나의 신뢰할 수 있는 답을 만들어내는 기술이다.

---

## 목차

1. [왜 센서 융합인가?](#1-왜-센서-융합인가)
2. [융합 아키텍처](#2-융합-아키텍처)
3. [가중 평균 융합](#3-가중-평균-융합)
4. [칼만 필터](#4-칼만-필터)
5. [보완 필터](#5-보완-필터)
6. [IMU를 위한 확장 칼만 필터](#6-imu를-위한-확장-칼만-필터)
7. [실용적인 응용 사례](#7-실용적인-응용-사례)
8. [연습 문제](#8-연습-문제)

---

## 1. 왜 센서 융합인가?

### 1.1 개별 센서의 한계

| 센서 | 강점 | 약점 |
|------|------|------|
| 가속도계(Accelerometer) | 정적 자세 정확도(중력 방향) | 잡음이 많고 진동에 민감, 요(Yaw) 측정 불가 |
| 자이로스코프(Gyroscope) | 부드러운 회전 속도 | 시간이 지나면 드리프트 누적 |
| 자력계(Magnetometer) | 절대 방위각(나침반) | 자기 간섭에 취약 |
| GPS | 절대 글로벌 위치 | 실내 미지원, 낮은 업데이트 주기 |
| 기압계(Barometer) | 고도 변화 | 느리고 날씨에 따라 드리프트 발생 |
| 카메라(Camera) | 풍부한 공간 정보 | 연산 비용이 높고 조명 조건에 의존 |

### 1.2 융합의 장점

```
개별 센서:                   융합 결과:

가속도계       ──┐
  (잡음은 많지만  │
   드리프트 없음) ├──► 센서 융합 ──► 정확하고 부드러운,
                 │    알고리즘      드리프트 없는 추정값
자이로스코프  ───┤
  (부드럽지만    │
   드리프트 발생) │
                 │
자력계        ───┘
  (절대적이지만
   간섭 발생)
```

| 장점 | 설명 |
|------|------|
| **정확도(Accuracy)** | 센서를 결합하면 전체 오차가 감소 |
| **견고성(Robustness)** | 하나의 센서가 실패해도 다른 센서가 보완 |
| **완전성(Completeness)** | 서로 다른 센서가 서로 다른 것을 측정(위치 + 자세) |
| **시간적 커버리지(Temporal coverage)** | 빠른 센서가 느린 센서 업데이트 사이의 공백을 채움 |

---

## 2. 융합 아키텍처

### 2.1 중앙 집중식 융합(Centralized Fusion)

모든 원시 센서 데이터를 단일 처리 노드로 전송:

```
센서 A ─── 원시 데이터 ──┐
센서 B ─── 원시 데이터 ──┼──► 중앙 처리기 ──► 융합 출력
센서 C ─── 원시 데이터 ──┘
```

**장점**: 최적 정확도(모든 데이터 사용 가능). **단점**: 높은 대역폭, 단일 장애점(Single Point of Failure).

### 2.2 분산식 융합(Decentralized Fusion)

각 센서가 로컬에서 전처리하고 추정값만 전송:

```
센서 A ──► 로컬 필터 A ──┐
센서 B ──► 로컬 필터 B ──┼──► 융합 노드 ──► 융합 출력
센서 C ──► 로컬 필터 C ──┘
```

**장점**: 낮은 대역폭, 장애 허용(Fault-tolerant). **단점**: 센서 간 상관관계가 손실될 수 있음.

### 2.3 계층적 융합(Hierarchical Fusion)

다단계 처리:

```
Level 1: 원시 센서 → 센서별 전처리
Level 2: 그룹 융합 (예: 모든 IMU 센서를 함께)
Level 3: 전역 융합 (IMU + GPS + 비전)
```

이는 자율주행차에서 일반적인 방식이다: IMU 융합은 로컬에서 200 Hz로 실행되고, 이후 더 상위 레벨에서 GPS(10 Hz) 및 LiDAR(20 Hz)와 융합된다.

---

## 3. 가중 평균 융합(Weighted Average Fusion)

가장 단순한 융합 방법으로, 각 센서에 잡음 분산의 역수에 비례한 가중치를 부여한다.

### 3.1 두 센서의 경우

측정값 z₁, z₂와 분산 σ₁², σ₂²를 가진 두 센서가 있을 때:

$$\hat{x} = \frac{\sigma_2^2}{\sigma_1^2 + \sigma_2^2} z_1 + \frac{\sigma_1^2}{\sigma_1^2 + \sigma_2^2} z_2$$

융합된 분산은 항상 각 개별 분산보다 작다:

$$\sigma_{fused}^2 = \frac{\sigma_1^2 \cdot \sigma_2^2}{\sigma_1^2 + \sigma_2^2} < \min(\sigma_1^2, \sigma_2^2)$$

### 3.2 예제: 온도 융합

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

## 4. 칼만 필터(Kalman Filter)

칼만 필터는 센서 융합에서 가장 중요한 알고리즘이다. 동적 모델의 예측과 잡음이 있는 측정값을 최적으로 결합한다.

### 4.1 핵심 아이디어

```
시간 k-1                        시간 k

 x̂[k-1] ──► 예측 ──► x̂⁻[k]  ──┐
             (모델)              ├──► 업데이트 ──► x̂[k]
                                 │   (보정)
             센서  ──► z[k]  ────┘
             읽기
```

1. **예측(Predict)**: 물리/운동 모델을 사용하여 다음 상태를 추정
2. **업데이트(Update)**: 실제 센서 측정값으로 예측을 보정
3. **반복(Repeat)**: 결과가 다음 예측의 입력이 됨

### 4.2 수식

**상태 모델**: $x_k = A \cdot x_{k-1} + B \cdot u_k + w_k$ (프로세스 잡음 w ~ N(0, Q))

**측정 모델**: $z_k = H \cdot x_k + v_k$ (측정 잡음 v ~ N(0, R))

**예측 단계**:

$$\hat{x}_k^- = A \cdot \hat{x}_{k-1} + B \cdot u_k$$
$$P_k^- = A \cdot P_{k-1} \cdot A^T + Q$$

**업데이트 단계**:

$$K_k = P_k^- \cdot H^T \cdot (H \cdot P_k^- \cdot H^T + R)^{-1}$$
$$\hat{x}_k = \hat{x}_k^- + K_k \cdot (z_k - H \cdot \hat{x}_k^-)$$
$$P_k = (I - K_k \cdot H) \cdot P_k^-$$

여기서:
- $K_k$는 **칼만 이득(Kalman Gain)** (측정값과 예측 중 얼마나 신뢰할지)
- $P_k$는 **오차 공분산(Error Covariance)** (상태 추정의 불확실성)
- R이 클 때(잡음 많은 센서): K가 작아짐 → 예측을 더 신뢰
- Q가 클 때(불확실한 모델): K가 커짐 → 측정값을 더 신뢰

### 4.3 1D 예제: 온도 추적

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

### 4.4 다차원 칼만 필터(Multi-Dimensional Kalman Filter)

위치와 속도를 함께 추적하는 경우:

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

## 5. 보완 필터(Complementary Filter)

보완 필터는 칼만 필터의 더 간단한 대안으로, IMU 자세 추정에 일반적으로 사용된다.

### 5.1 아이디어

- **가속도계**: 저주파(정적 자세)에서 정확, 고주파(진동)에서 잡음 발생
- **자이로스코프**: 고주파(빠른 회전)에서 정확, 저주파에서 드리프트 발생

상보적인 주파수 가중치로 둘을 결합:

$$\theta_{fused} = \alpha \cdot (\theta_{prev} + \omega_{gyro} \cdot \Delta t) + (1 - \alpha) \cdot \theta_{accel}$$

여기서:
- $\alpha$ = 0.95~0.98 (단기에는 자이로스코프, 장기에는 가속도계를 신뢰)
- $\omega_{gyro}$ = 자이로스코프 각속도
- $\theta_{accel}$ = 가속도계로 계산된 각도 (atan2 사용)
- $\Delta t$ = 시간 간격

### 5.2 왜 효과적인가

```
주파수 응답:

                  가속도계         자이로스코프     융합 결과
                  (저역 통과)      (고역 통과)     (전대역)

고주파:           ████             ████████████    ████████████
                  (잡음)           (정확)          (자이로에서)

저주파:           ████████████     ████            ████████████
                  (정확)           (드리프트)      (가속도계에서)
```

보완 필터는 칼만 필터의 1차 근사(First-Order Approximation)이지만 행렬 연산이 필요 없어 마이크로컨트롤러에 이상적이다.

### 5.3 구현

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

## 6. IMU를 위한 확장 칼만 필터(Extended Kalman Filter)

시스템이 3D 자세처럼 비선형인 경우, 표준 칼만 필터를 그대로 적용할 수 없다. 확장 칼만 필터(EKF)는 현재 추정값 주변에서 선형화(Linearization)를 수행한다.

### 6.1 IMU 융합에서의 비선형성

3D에서의 자세는 회전 행렬이나 쿼터니언(Quaternion)을 사용하는데, 둘 다 비선형이다. 측정 모델(중력 벡터를 자세 각도로 변환하는 것)도 `atan2`를 사용하므로 비선형이다.

### 6.2 EKF의 수정 사항

EKF는 상수 행렬 A와 H를 현재 상태에서 평가된 야코비안(Jacobian)으로 대체한다:

- **예측**: 비선형 모델 $f(x)$를 사용하되, 공분산 전파를 위해 야코비안 $F = \partial f / \partial x$를 계산
- **업데이트**: 비선형 측정 모델 $h(x)$를 사용하되, 칼만 이득을 위해 $H = \partial h / \partial x$를 계산

$$\hat{x}_k^- = f(\hat{x}_{k-1}, u_k)$$
$$P_k^- = F_k \cdot P_{k-1} \cdot F_k^T + Q$$
$$K_k = P_k^- \cdot H_k^T \cdot (H_k \cdot P_k^- \cdot H_k^T + R)^{-1}$$
$$\hat{x}_k = \hat{x}_k^- + K_k \cdot (z_k - h(\hat{x}_k^-))$$

### 6.3 9-DOF IMU 융합

9-DOF(자유도, Degree of Freedom) IMU는 다음을 포함한다:
- 3축 가속도계(중력 + 가속도 측정)
- 3축 자이로스코프(각속도 측정)
- 3축 자력계(지구 자기장 측정)

자세 추정을 위한 EKF 상태 벡터:

$$x = [q_0, q_1, q_2, q_3, b_{gx}, b_{gy}, b_{gz}]^T$$

여기서 $q_0..q_3$는 단위 쿼터니언(자세), $b_{gx}, b_{gy}, b_{gz}$는 자이로스코프 바이어스(Bias) 추정값이다. 필터는 자세를 추정하는 동시에 자이로스코프 드리프트를 학습한다.

### 6.4 센서 융합 파이프라인

```
┌────────────┐   ┌────────────┐   ┌────────────┐
│ 자이로스코프│   │ 가속도계   │   │ 자력계     │
│ 200-1000Hz │   │ 100-400Hz  │   │ 50-100Hz   │
└──────┬─────┘   └──────┬─────┘   └──────┬─────┘
       │                │                │
       ▼                ▼                ▼
  ┌─────────┐     ┌──────────┐    ┌──────────┐
  │ 바이어스 │     │ 중력     │    │ 기울기   │
  │ 제거    │     │ 정규화   │    │ 보상     │
  │         │     │          │    │          │
  └────┬────┘     └─────┬────┘    └─────┬────┘
       │                │               │
       └────────┬───────┴───────┬───────┘
                │               │
                ▼               ▼
          ┌──────────┐   ┌──────────┐
          │ EKF      │   │ EKF      │
          │ 예측     │──►│ 업데이트 │
          └──────────┘   └────┬─────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ 쿼터니언 → RPY   │
                    │ (롤/피치/요)     │
                    └──────────────────┘
```

---

## 7. 실용적인 응용 사례

### 7.1 드론 안정화

드론은 IMU(200 Hz) + 기압계(50 Hz) + GPS(10 Hz) + 광학 흐름(30 Hz)을 융합한다:

| 센서 | 측정 대상 | 융합 결과 |
|------|-----------|-----------|
| IMU(가속도계 + 자이로스코프) | 자세(롤/피치/요), 각속도 | 자세 + 각속도 |
| 기압계 | 고도(상대적) | 지면으로부터의 높이 |
| GPS | 위도/경도/고도(절대적) | 글로벌 위치 |
| 광학 흐름 카메라 | 지면 대비 속도 | 수평 속도(실내) |
| 초음파/LiDAR | 지면까지의 거리 | 정밀 착륙 고도 |

### 7.2 실내 내비게이션

실내 환경에서는 GPS를 사용할 수 없다. 센서 융합은 다음을 통해 위치 추정을 가능하게 한다:
- **보행자 추측 항법(PDR, Pedestrian Dead Reckoning)**: IMU가 걸음 수와 방향을 추적
- **WiFi 핑거프린팅(WiFi Fingerprinting)**: 신호 강도로 위치를 매핑
- **BLE 비콘(BLE Beacon)**: 근접 기반 위치 추정
- **기압계**: 다층 건물에서 층(Floor) 감지

융합 알고리즘(주로 파티클 필터(Particle Filter) 또는 EKF)이 모든 소스를 결합한다.

### 7.3 스마트 홈 환경 모니터링

여러 환경 센서를 융합:

```
온도 (3개 센서, 서로 다른 방) → 방 단위 평균
습도 + 온도 → 이슬점, 곰팡이 위험도
CO2 + 점유 상태 + HVAC 상태 → 환기 제어
조도 + 동작 + 시간 → 재실 감지 + 자동화
```

---

## 8. 연습 문제

### 문제 1: 가중 평균 융합

세 개의 온도 센서가 같은 위치를 측정한다:
- 센서 A: 22.1°C, 분산 = 0.5
- 센서 B: 22.8°C, 분산 = 0.2
- 센서 C: 22.4°C, 분산 = 0.3

역분산 가중(Inverse-Variance Weighting)을 사용하여 최적의 융합 온도와 그 분산을 계산하라.

### 문제 2: 1D 칼만 필터

로봇이 직선 위를 이동한다. 위치는 등속 모델 $x_k = x_{k-1} + v \cdot \Delta t$로 예측되며, v = 1 m/s, Δt = 1 s이다. 프로세스 잡음 Q = 0.1, 측정 잡음 R = 1.0이다.

x₀ = 0, P₀ = 1.0에서 시작하여 z = [1.2, 2.5, 2.8, 4.1, 5.3]의 측정값으로 5단계의 칼만 필터를 수행하라. 각 단계에서 상태 추정값과 칼만 이득을 제시하라.

### 문제 3: 보완 필터 튜닝

α = 0.98의 보완 필터를 구현하여 가속도계와 자이로스코프 데이터를 결합한다. 합성 데이터를 생성하라:
- 실제 각도: θ(t) = 30° × sin(2π × 0.5 × t) (진동)
- 자이로스코프: dθ/dt + N(0, 0.1°/s) (낮은 잡음)
- 가속도계 각도: θ(t) + N(0, 3°) (높은 잡음)

원시 가속도계 각도, 적분된 자이로스코프 각도, 보완 필터 출력을 플롯하라. 필터가 두 센서의 장점을 결합하는 것을 보여라.

### 문제 4: 센서 융합 아키텍처

사무 건물 내에서 운용되는 배달 로봇을 위한 센서 융합 시스템을 설계하라:
1. 필요한 모든 센서와 업데이트 주기를 나열하라
2. 융합 아키텍처(중앙 집중식 vs 계층적)를 선택하라
3. 데이터 흐름 다이어그램을 그려라
4. 각 융합 지점에서 사용할 필터 유형(KF, EKF, 파티클 필터)을 명시하라

### 문제 5: EKF 바이어스 추정

자이로스코프에 알 수 없는 상수 바이어스 0.5°/s가 있다. 각도와 자이로스코프 바이어스를 동시에 추정하는 EKF를 설계하라. 상태 벡터: x = [θ, b_gyro]. 충분한 시간이 지나면 바이어스 추정값이 실제 값으로 수렴하는 것을 보여라.

---

*레슨 14 끝*
