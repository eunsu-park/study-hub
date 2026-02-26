# 10. 센서와 인지(Sensors and Perception)

[← 이전: 로봇 제어](09_Robot_Control.md) | [다음: 상태 추정과 필터링 →](11_State_Estimation.md)

---

## 학습 목표

1. 로봇 센서를 고유 수용성 센서(proprioceptive)와 외부 수용성 센서(exteroceptive)로 분류하고 각각의 역할을 설명한다
2. 인코더 종류(증분식, 절대식)와 관성 측정 장치(IMU)를 이해한다
3. 카메라 모델(핀홀, 왜곡)과 카메라 캘리브레이션 절차를 설명한다
4. LiDAR 동작 원리와 포인트 클라우드 처리 기초를 설명한다
5. 여러 센서 모달리티를 결합하는 기본적인 센서 융합 개념을 적용한다
6. 다양한 로봇 응용에 맞는 적절한 센서를 선택한다

---

환경을 인지할 수 없는 로봇은 눈이 먼 것과 같다 — 제어 알고리즘이 아무리 정교해도 충돌하고, 물체를 떨어뜨리고, 세계에 대한 인식이 필요한 모든 작업에서 실패한다. 인지(Perception)는 원시 센서 신호를 로봇의 상태와 주변 환경에 대한 유용한 정보로 변환하는 과정이다. 인지는 두 가지 근본적인 질문에 답한다: *나는 어디에 있는가?* (고유 수용성과 위치 추정) 그리고 *내 주변에 무엇이 있는가?* (외부 수용성과 환경 모델링).

이 레슨에서는 현대 로보틱스에서 사용되는 센서들을 살펴보고, 그 데이터를 해석하는 데 필요한 수학적 모델을 소개한다. 어떤 단일 센서도 완벽하지 않으며 — 각각 장점과 단점이 있다. 로봇 인지의 핵심은 여러 센서를 결합하여 개별 한계를 보완하는 것인데, 이는 다음 레슨인 상태 추정과 필터링에서 더 깊이 다룰 주제이다.

> **비유**: 로봇의 센서는 오케스트라의 악기들과 같다 — 각각은 공연의 다른 측면(멜로디, 리듬, 화음)을 포착하지만, 융합을 통해 완전한 교향곡이 만들어진다. 카메라는 풍부한 시각 정보를 포착하지만 어둠 속에서는 어려움을 겪는다. LiDAR는 정밀한 깊이를 제공하지만 색상과 질감을 놓친다. IMU는 로봇이 회전하는 방식을 알지만 시간이 지나면 드리프트가 발생한다. 이들을 결합하면 세계에 대한 완전하고 강건한 인지가 가능해진다.

---

## 1. 고유 수용성 센서(Proprioceptive Sensors)

**고유 수용성 센서(Proprioceptive sensors)**는 로봇 자체의 내부 상태 — 관절 위치, 속도, 힘, 그리고 본체 방향 — 를 측정한다. 이는 로봇이 자신의 몸을 감지하는 감각이다.

### 1.1 인코더(Encoders)

인코더는 가장 기본적인 로봇 센서이다. 관절이나 모터 축의 회전(또는 직선 변위)을 측정한다.

#### 증분식 인코더(Incremental Encoders)

증분식 인코더는 축이 회전할 때 펄스를 생성한다. 직교 인코더(quadrature encoder)는 90도 위상차가 있는 두 채널(A와 B)을 사용하여 속도와 방향 모두를 결정한다:

```
Channel A: ___╱‾‾‾╲___╱‾‾‾╲___╱‾‾‾╲___
Channel B: ╱‾‾‾╲___╱‾‾‾╲___╱‾‾‾╲___╱‾‾
                ↑ Count up (A leads B: clockwise)
```

**해상도(Resolution)**: 1회전당 $N$ 카운트(CPR)가 주어지면, 각도 해상도는:

$$\Delta\theta = \frac{2\pi}{N} \text{ rad}$$

직교 디코딩(두 채널의 모든 엣지를 카운트)을 사용하면 유효 해상도는 기본 CPR의 $4\times$이다. 1024 CPR 인코더는 1회전당 $4 \times 1024 = 4096$ 카운트, 즉 카운트당 약 $0.088°$의 해상도를 제공한다.

인코더 카운트로부터 **속도 추정(velocity estimation)**:

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

#### 절대식 인코더(Absolute Encoders)

증분식 인코더와 달리, **절대식 인코더(absolute encoders)**는 전원을 껐다가 켜도 항상 정확한 축 위치를 보고한다. 각도 위치마다 고유한 코드를 생성하기 위해 코드화된 디스크 패턴(그레이 코드 또는 이진 코드)을 사용한다.

| 특징 | 증분식(Incremental) | 절대식(Absolute) |
|---------|-------------|----------|
| 출력 | 펄스 트레인 | 위치 코드 |
| 전원 인가 시 | 원점 복귀/기준 설정 필요 | 즉시 위치 파악 |
| 해상도 | 높음 (단순한 디스크) | 낮음 (복잡한 코드 트랙) |
| 비용 | 낮음 | 높음 |
| 사용 사례 | 대부분의 관절 | 안전하게 원점 복귀할 수 없는 관절 |

### 1.2 관성 측정 장치(Inertial Measurement Unit, IMU)

IMU는 다음을 결합한다:
- **가속도계(Accelerometer)** (3축): 선형 가속도 + 중력 측정
- **자이로스코프(Gyroscope)** (3축): 각속도 측정
- **자력계(Magnetometer)** (3축, 선택): 자기장 방향(나침반) 측정

6축 IMU(가속도계 + 자이로스코프)가 표준이며, 9축은 자력계를 추가한다.

#### IMU 측정 모델(IMU Measurement Model)

**자이로스코프** 출력:

$$\boldsymbol{\omega}_m = \boldsymbol{\omega}_{true} + \mathbf{b}_g + \mathbf{n}_g$$

여기서 $\mathbf{b}_g$는 천천히 드리프트하는 바이어스이고 $\mathbf{n}_g$는 백색 잡음이다.

**가속도계** 출력:

$$\mathbf{a}_m = R^T(\mathbf{a}_{true} - \mathbf{g}) + \mathbf{b}_a + \mathbf{n}_a$$

여기서 $R$은 세계 프레임에서 바디 프레임으로의 회전, $\mathbf{g}$는 중력, $\mathbf{b}_a$는 바이어스, $\mathbf{n}_a$는 잡음이다.

**핵심 과제 — 드리프트(drift)**: 자이로스코프 각속도를 적분하여 방향을 얻으면 시간에 따라 바이어스 오차가 누적된다. 0.01 deg/s 바이어스를 가진 자이로스코프는 시간당 36도 드리프트한다. 이것이 IMU를 항상 다른 센서(GPS, 비전, 인코더)와 융합하는 이유이다.

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

### 1.3 관절 토크 센서(Joint Torque Sensors)

토크 센서는 관절의 힘/토크를 측정하며, 일반적으로 구동 계통의 순응적(compliant) 요소에 부착된 스트레인 게이지를 사용한다. 다음 용도에 필수적이다:

- **힘 제어(Force control)**: 접촉 힘 측정 (로봇의 동역학 모델을 통해)
- **충돌 감지(Collision detection)**: 갑작스러운 토크 급증은 예기치 않은 접촉을 나타냄
- **중력 보상(Gravity compensation)**: 힘 추정 정확도 향상

**직렬 탄성 액추에이터(Series Elastic Actuators, SEAs)**는 모터와 관절 사이에 알려진 순응성을 가진 스프링을 내장한다. 양쪽의 인코더로 측정된 스프링 변형이 토크 측정값을 제공한다:

$$\tau_{joint} = k_{spring} \cdot (\theta_{motor}/N - \theta_{joint})$$

여기서 $k_{spring}$은 스프링 강성이고 $N$은 기어비이다.

---

## 2. 외부 수용성 센서(Exteroceptive Sensors)

**외부 수용성 센서(Exteroceptive sensors)**는 외부 환경 — 물체까지의 거리, 시각적 외관, 공간적 구조 — 를 측정한다.

### 2.1 카메라(Cameras)

카메라는 색상, 질감, 형태, 움직임 단서를 포함하는 밀도 높은 2D 이미지를 제공하여 로봇에게 가장 풍부한 정보 원천이다.

#### 카메라 시스템 유형

| 유형 | 출력 | 깊이? | 비용 | 사용 사례 |
|------|--------|--------|------|----------|
| 단안(Monocular) | 2D 이미지 | 없음 (구조로부터 운동 필요) | $ | 일반 비전, SLAM |
| 스테레오(Stereo) | 2D 이미지 쌍 | 있음 (삼각측량) | $$ | 장애물 감지, 3D 매핑 |
| RGB-D | 2D 이미지 + 깊이 맵 | 있음 (구조광 또는 ToF) | $$ | 실내 로보틱스, 조작 |
| 이벤트 카메라(Event camera) | 비동기 이벤트 | 없음 | $$$ | 고속 움직임, 낮은 지연 |

#### 해상도, 시야각, 프레임 레이트 트레이드오프

카메라는 트레이드오프를 강요한다:
- **높은 해상도** → 더 많은 세부 정보, 하지만 처리할 데이터가 많고 프레임 레이트가 느림
- **넓은 시야각(Field of View)** → 더 많이 볼 수 있지만, 더 많은 왜곡과 낮은 각도 해상도
- **높은 프레임 레이트** → 빠른 움직임에 유리하지만, 더 많은 데이터와 잠재적으로 더 많은 잡음 (짧은 노출 시간)

### 2.2 LiDAR

**LiDAR**(Light Detection And Ranging, 광 감지 및 거리 측정)는 레이저 펄스의 시간을 측정하여 거리를 측정한다:

$$d = \frac{c \cdot \Delta t}{2}$$

여기서 $c$는 빛의 속도이고 $\Delta t$는 왕복 시간이다.

#### 2D vs. 3D LiDAR

**2D LiDAR** (단일 스캔 평면):
- 한 평면에서 거리 측정 (예: 수평)
- 출력: $(r, \theta)$ 측정값 배열
- 일반적: 360도 스캔, 10-40 Hz, 0.25-1도 각도 해상도
- 범위: 10-30 m, 정확도: 1-3 cm

**3D LiDAR** (다중 스캔 평면 또는 회전 배열):
- 3D로 거리 측정 (포인트 클라우드)
- 일반적: 16-128 채널, 10-20 Hz, 초당 수백만 포인트
- 범위: 100-200 m, 정확도: 1-3 cm

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

### 2.3 기타 외부 수용성 센서

**초음파 센서(Ultrasonic sensors)**: 음파(일반적으로 40 kHz)를 사용하여 거리를 측정한다. 저렴하고 단순하지만 빔 각도가 넓어 (각도 해상도가 낮음), 범위가 제한적이며 (2-5 m), 매끄러운 표면의 거울 반사에 취약하다.

**적외선(IR) 근접 센서**: IR 빛을 사용하여 단거리 (10-80 cm) 거리를 측정한다. 빠르고 저렴하다. 장애물 감지, 낙하 감지 (로봇이 가장자리에서 떨어지는 것 방지)에 사용된다.

**힘/토크 센서 (손목 장착형, Force/Torque sensors)**: 엔드 이펙터에서 6축 힘과 토크를 측정한다. 접촉이 포함된 조작 작업 (조립, 연마, 사람에게 물체 전달)에 필수적이다.

---

## 3. 카메라 모델과 캘리브레이션(Camera Models and Calibration)

### 3.1 핀홀 카메라 모델(The Pinhole Camera Model)

**핀홀 모델(pinhole model)**은 3D 점이 2D 이미지에 투영되는 방식을 설명한다:

$$\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \frac{1}{Z_c} \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix}$$

여기서:
- $(X_c, Y_c, Z_c)$는 카메라 좌표계의 3D 점
- $(u, v)$는 이미지의 픽셀 좌표
- $f_x, f_y$는 픽셀 단위의 초점 거리
- $(c_x, c_y)$는 주점(principal point) (보통 이미지 중심 근처)

**내부 행렬(intrinsic matrix)** $K$:

$$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

세계 좌표에서 픽셀 좌표로의 **전체 투영**은 **외부(extrinsic)** 변환을 포함한다:

$$\mathbf{p}_{pixel} \sim K [R | \mathbf{t}] \mathbf{P}_{world}$$

여기서 $[R | \mathbf{t}]$는 3x4 카메라 포즈 행렬 (회전 및 이동)이다.

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

### 3.2 렌즈 왜곡(Lens Distortion)

실제 렌즈는 왜곡을 유발한다. 두 가지 주요 유형은:

**방사 왜곡(Radial distortion)** (배럴/핀쿠션):

$$x_d = x(1 + k_1 r^2 + k_2 r^4 + k_3 r^6)$$
$$y_d = y(1 + k_1 r^2 + k_2 r^4 + k_3 r^6)$$

여기서 $r^2 = x^2 + y^2$는 광학 중심으로부터의 거리 제곱이고, $k_1, k_2, k_3$는 방사 왜곡 계수이다.

**접선 왜곡(Tangential distortion)** (불완전한 렌즈 정렬로 인한):

$$x_d = x + 2p_1 xy + p_2(r^2 + 2x^2)$$
$$y_d = y + p_1(r^2 + 2y^2) + 2p_2 xy$$

5개의 왜곡 계수 $[k_1, k_2, p_1, p_2, k_3]$는 카메라 캘리브레이션 중에 추정된다.

### 3.3 카메라 캘리브레이션(Camera Calibration)

**카메라 캘리브레이션(Camera calibration)**은 알려진 캘리브레이션 패턴(보통 체크보드)의 이미지로부터 내부 파라미터 $(f_x, f_y, c_x, c_y, k_1, \ldots)$를 추정한다.

**절차**:
1. 알려진 정사각형 크기의 체크보드 패턴 출력
2. 다양한 각도와 거리에서 체크보드의 15-25장 이미지 촬영
3. 각 이미지에서 체크보드 코너 검출 (서브 픽셀 정확도)
4. 장의 방법(Zhang's method, 단응사상 기반 초기화 + 비선형 정밀화)을 사용하여 내부 및 외부 파라미터 추정

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

### 3.4 스테레오 비전(Stereo Vision)

**스테레오 카메라(Stereo cameras)**는 알려진 **기선(baseline)** $b$만큼 떨어진 두 카메라를 사용하여 삼각측량으로 깊이를 추정한다:

$$Z = \frac{f \cdot b}{d}$$

여기서 $d = u_L - u_R$은 **시차(disparity)** (같은 3D 점에 대한 왼쪽과 오른쪽 이미지의 픽셀 차이)이다.

스테레오 비전의 주요 단계:
1. **정류(Rectification)**: 대응 점들이 같은 수평선 위에 놓이도록 이미지 변환
2. **스테레오 매칭(Stereo matching)**: 왼쪽과 오른쪽 이미지 간의 대응 관계 찾기 (블록 매칭, 세미 글로벌 매칭)
3. **삼각측량(Triangulation)**: 위 공식을 사용하여 시차로부터 깊이 계산

**깊이 정확도**는 거리에 따라 저하된다: $\Delta Z \propto Z^2 / (fb)$. 이는 스테레오가 근거리에서는 정확하지만 원거리에서는 잡음이 많다는 것을 의미한다.

---

## 4. 포인트 클라우드 처리(Point Cloud Processing)

LiDAR와 깊이 카메라는 **포인트 클라우드(point clouds)** — 3D 점의 집합 $\{(x_i, y_i, z_i)\}_{i=1}^N$ — 를 생성한다.

### 4.1 필터링과 전처리(Filtering and Preprocessing)

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

### 4.2 포인트 클라우드 정합(Point Cloud Registration, ICP)

**반복 최근접점(Iterative Closest Point, ICP)**은 대응 관계를 반복적으로 찾고 정렬 오차를 최소화하여 두 포인트 클라우드를 정렬한다:

$$T^* = \arg\min_T \sum_{i=1}^N \| T \mathbf{p}_i - \mathbf{q}_{c(i)} \|^2$$

여기서 $T$는 강체 변환, $\mathbf{p}_i$는 소스 점들, $\mathbf{q}_{c(i)}$는 타겟 클라우드에서 가장 가까운 점들이다.

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

### 4.3 지면 평면 추정(Ground Plane Estimation)

이동 로봇의 경우, 지면 평면을 장애물로부터 분리하는 것은 핵심적인 전처리 단계이다:

$$ax + by + cz + d = 0$$

**RANSAC**(Random Sample Consensus, 무작위 샘플 합의)이 표준 접근법이다:

1. 3개의 점을 무작위로 샘플링
2. 이 점들에 평면 피팅
3. 얼마나 많은 다른 점들이 평면으로부터 임계값 거리 이내에 있는지 카운트 (내점, inliers)
4. 반복하여 가장 많은 내점을 가진 평면 유지

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

## 5. 센서 융합 기초(Sensor Fusion Fundamentals)

### 5.1 센서를 융합하는 이유(Why Fuse Sensors?)

어떤 센서도 완벽하지 않다:

| 센서 | 강점 | 약점 |
|--------|-----------|------------|
| 인코더(Encoder) | 정밀, 높은 레이트, 저렴 | 국소적 (환경 없음) |
| IMU | 높은 레이트 (100-1000 Hz), 회전 측정 | 시간에 따른 드리프트 |
| 카메라(Camera) | 풍부한 정보, 질감 | 조명 의존적, 직접적인 깊이 없음 |
| LiDAR | 정확한 깊이, 조명 불변 | 희소, 비쌈, 질감 없음 |
| GPS | 전역 위치, 드리프트 없음 | 낮은 레이트, 실내 불가, 2-5m 정확도 |

**센서 융합(Sensor fusion)**은 여러 센서를 결합하여 다음을 달성한다:

1. **상보적 융합(Complementary fusion)**: 서로 다른 센서가 서로 다른 것을 측정 (IMU로 회전 + 인코더로 관절 위치)
2. **중복 융합(Redundant fusion)**: 여러 센서가 동일한 것을 측정하여 향상된 정확도 (위치를 위한 GPS + 시각 오도메트리)
3. **협력 융합(Cooperative fusion)**: 센서들이 단독으로는 불가능한 방식으로 함께 작동 (깊이를 위한 스테레오 카메라 쌍)

### 5.2 단순 상보 필터(Simple Complementary Filter)

자이로스코프 (단기 정확, 장기 드리프트)와 가속도계 (단기 잡음, 중력 방향에 대해 장기 정확)를 결합하기 위해:

$$\hat{\theta} = \alpha(\hat{\theta}_{prev} + \omega_{gyro} \cdot dt) + (1 - \alpha) \cdot \theta_{accel}$$

여기서 $\alpha \in [0.95, 0.99]$는 단기 변화에 자이로스코프에 높은 가중치를 주고 장기 보정에 가속도계에 낮은 가중치를 준다.

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

### 5.3 상보 필터에서 칼만 필터로(From Complementary Filter to Kalman Filter)

상보 필터는 직관적인 출발점이지만 한계가 있다:

- 고정된 혼합 비율 $\alpha$는 변화하는 잡음 조건에 적응하지 못함
- 서로 다른 업데이트 레이트를 가진 여러 센서를 처리할 수 없음
- 공식적인 불확실성 정량화 없음

**칼만 필터(Kalman filter)** (다음 레슨에서 다룸)는 가우시안 잡음 가정 하에 센서 융합 문제에 대한 최적의 해를 제공한다. 이는 혼합 비율이 센서 잡음 모델로부터 최적으로 계산되는 상보 필터의 일반화로 볼 수 있다.

### 5.4 일반 플랫폼을 위한 센서 선택(Sensor Selection for Common Platforms)

| 플랫폼 | 일반적인 센서 | 근거 |
|----------|----------------|-----------|
| 산업용 팔 | 인코더, F/T 센서 | 고정밀 관절, 접촉 작업 |
| 이동 로봇 (실내) | LiDAR, IMU, 휠 인코더 | GPS 없이 내비게이션 |
| 이동 로봇 (실외) | GPS, IMU, LiDAR, 카메라 | 전역 + 국소 위치 결정 |
| 드론 | IMU, 기압계, GPS, 카메라 | 6-DOF 상태 추정 |
| 휴머노이드 | IMU, 관절 인코더, F/T (발), 카메라 | 균형 + 인지 |
| 자율 주행 차 | LiDAR, 카메라 (×8+), 레이더, IMU, GPS | 360° 인지, 안전을 위한 중복성 |

---

## 6. 깊이 감지 기술(Depth Sensing Technologies)

### 6.1 구조광(Structured Light)

알려진 패턴 (점, 선 또는 스펙클)을 장면에 투영한다. 카메라는 패턴이 표면에서 어떻게 변형되는지 관찰하고, 삼각측량으로 깊이를 계산한다. **예시**: Intel RealSense D400 시리즈.

**장점**: 밀도 높은 깊이 맵, 실내에서 작동
**단점**: 햇빛에서 실패 (IR 패턴이 씻겨 나감), 제한된 범위 (0.3-10 m)

### 6.2 비행 시간(Time-of-Flight, ToF)

변조된 빛을 방출하고 위상 이동을 측정하여 거리를 계산한다:

$$d = \frac{c \cdot \Delta\phi}{4\pi f_{mod}}$$

여기서 $f_{mod}$는 변조 주파수이고 $\Delta\phi$는 위상 차이이다.

**장점**: 어떤 조명에서도 작동, 빠름, 소형
**단점**: 낮은 해상도, 다중 경로 간섭, 제한된 범위

### 6.3 비교(Comparison)

| 기술 | 범위 | 해상도 | 실외? | 비용 |
|------------|-------|------------|----------|------|
| 구조광(Structured light) | 0.3-10 m | 높음 (640×480+) | 아니오 | $$ |
| ToF 카메라 | 0.3-5 m | 낮음 (320×240) | 부분적 | $$ |
| 스테레오 카메라(Stereo camera) | 0.5-20 m | 높음 | 예 | $$ |
| LiDAR (3D) | 1-200 m | 희소 (포인트 클라우드) | 예 | $$$$ |

---

## 요약

| 개념 | 핵심 아이디어 |
|---------|----------|
| 고유 수용성 센서(Proprioceptive sensors) | 로봇의 내부 상태 측정 (인코더, IMU, 토크 센서) |
| 증분식 인코더(Incremental encoder) | 방향을 위한 직교 펄스 카운팅; 시작 시 원점 복귀 필요 |
| 절대식 인코더(Absolute encoder) | 위치마다 고유한 코드; 전원 사이클 후 위치 파악 |
| IMU | 가속도계 + 자이로스코프; 높은 레이트이지만 융합 없이는 드리프트 |
| 핀홀 카메라 모델(Pinhole camera model) | 내부 행렬 $K$를 통한 3D에서 2D로의 투영 |
| 렌즈 왜곡(Lens distortion) | 방사 및 접선 왜곡; 캘리브레이션을 통해 보정 |
| LiDAR | 레이저 비행 시간을 통한 직접 거리 측정; 정확, 조명 불변 |
| ICP | 포인트 클라우드의 반복적 정렬; 반복당 최근접점 + SVD |
| RANSAC | 강건한 모델 피팅; 이상치에 면역 |
| 센서 융합(Sensor fusion) | 개별 약점을 보완하기 위한 센서 결합 |
| 상보 필터(Complementary filter) | 자이로 (고역 통과)와 가속도계 (저역 통과)의 주파수 도메인 혼합 |

---

## 연습문제

1. **인코더 해상도 분석**: 6-DOF 로봇 팔에 직교 디코딩이 있는 1024 CPR 증분식 인코더와 100:1 기어비가 있다. 각 관절의 각도 해상도를 도(degree) 단위로 계산하라. 엔드 이펙터가 베이스에서 1 m 떨어져 있다면, 엔드 이펙터에서 최악의 경우 위치 해상도를 추정하라.

2. **카메라 캘리브레이션**: $f_x = f_y = 500$ 픽셀, $c_x = 320$, $c_y = 240$인 핀홀 카메라 모델을 사용하여 3D 점 집합 $[(1,0,5), (0,1,5), (-1,-1,10)]$을 픽셀 좌표로 투영하라. 그런 다음 알려진 깊이를 사용하여 역투영하고 원래의 3D 점을 복구하는지 확인하라.

3. **LiDAR 스캔 매칭**: 단순한 환경 (예: 원형 장애물이 있는 직사각형 방)의 두 2D LiDAR 스캔을 생성하라. 알려진 강체 변환 (회전 + 이동)을 적용하여 두 번째 스캔을 만들어라. ICP를 구현하여 변환을 복구하고 초기 변위의 함수로 오차를 측정하라.

4. **상보 필터**: 정현파 흔들림 운동을 실행하는 로봇의 피치 각도를 측정하는 IMU를 시뮬레이션하라. 잡음이 있는 자이로스코프와 가속도계 신호를 생성하라. 상보 필터를 구현하고 다양한 $\alpha$ 값에 대해 결과를 실제값과 비교하라.

5. **센서 선택 보고서**: 통로를 탐색하고, 선반에서 물품을 집어 포장 스테이션으로 배달해야 하는 창고 로봇을 위해 센서 조합을 제안하라. 비용, 환경 조건 (실내, 구조화됨), 필요한 정확도, 중복성을 고려하여 각 센서 선택을 정당화하라.

---

## 참고 문헌

- Corke, P. *Robotics, Vision and Control*, 3rd ed. Springer, 2023. Part III: Vision. (종합적인 로보틱스 + 비전 교과서)
- Hartley, R. and Zisserman, A. *Multiple View Geometry in Computer Vision*, 2nd ed. Cambridge, 2004. (카메라 기하학의 결정적 참고 문헌)
- Zhang, Z. "A Flexible New Technique for Camera Calibration." *IEEE TPAMI*, 2000. (표준 캘리브레이션 방법)
- Besl, P. and McKay, N. "A Method for Registration of 3-D Shapes." *IEEE TPAMI*, 1992. (원본 ICP 논문)
- Fischler, M. and Bolles, R. "Random Sample Consensus." *Communications of the ACM*, 1981. (RANSAC)

---

[← 이전: 로봇 제어](09_Robot_Control.md) | [다음: 상태 추정과 필터링 →](11_State_Estimation.md)
