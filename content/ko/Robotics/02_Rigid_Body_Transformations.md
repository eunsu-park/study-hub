# 강체 변환(Rigid Body Transformations)

[← 이전: 로봇공학 개요](01_Robotics_Overview.md) | [다음: 순운동학 →](03_Forward_Kinematics.md)

## 학습 목표

1. 2D 및 3D 회전에 대한 회전 행렬(rotation matrix)을 구성하고 SO(3) 군의 성질을 이해한다
2. 오일러 각도(Euler angles)를 사용해 방향을 기술하고, 짐벌 잠금(gimbal lock) 문제를 파악하며, 그것이 유용성을 제한하는 이유를 설명한다
3. 축-각도(axis-angle) 표현을 적용하고 로드리게스 공식(Rodrigues' formula)을 통해 회전을 유도한다
4. 쿼터니언(quaternion)을 정의하고, 쿼터니언 곱셈을 수행하며, 쿼터니언과 회전 행렬 간의 변환을 수행한다
5. SE(3)의 동차 변환 행렬(homogeneous transformation matrix)을 구성해 회전과 이동의 결합을 표현한다
6. 복수의 변환을 합성해 중간 좌표계를 거쳐 한 좌표계의 자세를 다른 좌표계에 대해 기술한다

---

## 왜 이것이 중요한가

모든 로봇 시스템은 겉보기에 단순해 보이는 질문에 답해야 한다: "이것은 어디에 있고, 어느 방향을 향하고 있는가?" 관절이든, 끝단 효과기든, 카메라든, 장애물이든 간에, 우리는 그것의 **위치(position)**와 **방향(orientation)** — 통칭해서 **자세(pose)**라 함 — 을 기술하는 정밀한 수학 언어가 필요하다.

강체 변환(Rigid body transformations)이 그 언어를 제공한다. 이것은 모든 기구학(레슨 3-5), 동역학(레슨 6), 계획(레슨 7-8)이 구축되는 토대다. 회전 행렬은 그리퍼가 어떻게 방향을 잡고 있는지 알려준다. 동차 변환 행렬은 로봇 베이스에 대한 카메라의 위치를 알려준다. 이 표현들을 올바르게 이해하고 그 미묘함을 파악하는 것은 선택 사항이 아니다. 로봇 수학의 입장권이다.

> **비유**: 쿼터니언(Quaternion)은 4차원의 나침반 방위 같다 — 오일러 각도를 괴롭히는 짐벌 잠금의 "혼란"을 피한다. 마치 자기 나침반이 방향을 "왼쪽 회전"과 "오른쪽 회전"만으로 기술하려는 혼란을 피하는 것처럼.

---

## 좌표계 및 표기법

로봇공학에서 우리는 **좌표계(coordinate frames)** (오른손 직교 좌표계)를 물체에 부착한다: 세계, 로봇 베이스, 각 링크, 끝단 효과기, 센서, 장애물.

다음과 같은 표기법을 사용한다:
- $\{A\}$, $\{B\}$: 좌표계
- ${}^{A}\mathbf{p}$: 좌표계 $\{A\}$에서 표현된 점
- ${}^{A}_{B}\mathbf{R}$: 좌표계 $\{B\}$에서 좌표계 $\{A\}$로 벡터를 변환하는 회전 행렬
- ${}^{A}_{B}\mathbf{T}$: 좌표계 $\{B\}$에서 좌표계 $\{A\}$로의 동차 변환

관례: **위 첨자는 벡터가 표현된 좌표계, 아래 첨자는 기술되는 좌표계다.**

---

## 2D 회전

### 2D 회전 행렬

각도 $\theta$ (반시계 방향)의 2D 회전은 다음으로 표현된다:

$$R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

이 행렬은 평면에서 벡터 $\mathbf{v}$를 회전시킨다:

$$\mathbf{v}' = R(\theta) \mathbf{v}$$

```python
import numpy as np

def rot2d(theta):
    """2D rotation matrix.

    Why a matrix? Because rotation is a linear transformation,
    and matrices are the natural representation of linear maps.
    Composing two rotations is just matrix multiplication.
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])

# Rotate a point (1, 0) by 90 degrees
theta = np.pi / 2
p = np.array([1.0, 0.0])
p_rotated = rot2d(theta) @ p
print(f"Original: {p}")
print(f"Rotated by 90 deg: {p_rotated}")  # [0, 1] (approximately)

# Key property: composition
# Rotating by 30 deg then by 60 deg = rotating by 90 deg
R_30 = rot2d(np.radians(30))
R_60 = rot2d(np.radians(60))
R_90 = rot2d(np.radians(90))
print(f"R(30)*R(60) == R(90): {np.allclose(R_30 @ R_60, R_90)}")  # True
```

### 2D 회전 행렬의 성질

1. **직교(Orthogonal)**: $R^T R = I$ (열들이 정규 직교(orthonormal))
2. **행렬식 1**: $\det(R) = 1$ (방향 보존 — 반사 없음)
3. **역행렬 = 전치행렬**: $R^{-1} = R^T$ (계산적으로 효율적!)
4. **합성**: $R(\alpha) R(\beta) = R(\alpha + \beta)$

---

## 3D 회전과 SO(3)

### 기본 회전 행렬

세 주축을 중심으로 한 회전:

**$x$축을 중심으로 각도 $\alpha$ 회전**:

$$R_x(\alpha) = \begin{bmatrix} 1 & 0 & 0 \\ 0 & \cos\alpha & -\sin\alpha \\ 0 & \sin\alpha & \cos\alpha \end{bmatrix}$$

**$y$축을 중심으로 각도 $\beta$ 회전**:

$$R_y(\beta) = \begin{bmatrix} \cos\beta & 0 & \sin\beta \\ 0 & 1 & 0 \\ -\sin\beta & 0 & \cos\beta \end{bmatrix}$$

**$z$축을 중심으로 각도 $\gamma$ 회전**:

$$R_z(\gamma) = \begin{bmatrix} \cos\gamma & -\sin\gamma & 0 \\ \sin\gamma & \cos\gamma & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

```python
def rotx(alpha):
    """Rotation about x-axis."""
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]])

def roty(beta):
    """Rotation about y-axis.

    Why is the sign pattern different? The y-axis rotation has +sin
    in the (0,2) position and -sin in the (2,0) position — this is
    because the cyclic order x→y→z requires this sign convention
    to maintain a right-handed coordinate system.
    """
    c, s = np.cos(beta), np.sin(beta)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])

def rotz(gamma):
    """Rotation about z-axis."""
    c, s = np.cos(gamma), np.sin(gamma)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])
```

### 특수 직교군 SO(3)

모든 3D 회전 행렬의 집합은 **SO(3)** — 3차원 특수 직교군(Special Orthogonal group) — 이라는 수학적 군(group)을 형성한다.

**SO(3)의 성질**:
- **닫힘(Closure)**: $R_1, R_2 \in SO(3)$이면 $R_1 R_2 \in SO(3)$
- **항등원(Identity)**: $I \in SO(3)$
- **역원(Inverse)**: $R^{-1} = R^T \in SO(3)$
- **결합법칙(Associativity)**: $(R_1 R_2) R_3 = R_1 (R_2 R_3)$
- **교환법칙 불성립**: $R_1 R_2 \neq R_2 R_1$ (일반적으로)

비교환성은 매우 중요하다 — 회전의 순서가 중요하다:

```python
# Demonstration: rotation order matters!
# Rotate 90 deg about x, then 90 deg about z
R1 = rotx(np.pi/2) @ rotz(np.pi/2)
# Rotate 90 deg about z, then 90 deg about x
R2 = rotz(np.pi/2) @ rotx(np.pi/2)

print("Rx(90)*Rz(90):")
print(np.round(R1, 3))
print("\nRz(90)*Rx(90):")
print(np.round(R2, 3))
print(f"\nAre they equal? {np.allclose(R1, R2)}")  # False!
```

> **직관**: 앞에 책을 들어 보라. 수직 축을 중심으로 90도 돌려라(문을 여는 것처럼), 그 다음 당신을 향한 축을 중심으로 90도 기울여라. 이제 처음으로 돌아가 먼저 기울이고 그 다음 돌려라. 최종 방향이 다르다!

### 회전 행렬 해석

회전 행렬 $R$은 두 가지 동등한 방법으로 해석할 수 있다:

1. **열 해석**: ${}^{A}_{B}R$의 열들은 좌표계 $\{A\}$에서 표현된 좌표계 $\{B\}$ 축의 단위 벡터들이다:

$${}^{A}_{B}R = \begin{bmatrix} {}^{A}\hat{x}_B & {}^{A}\hat{y}_B & {}^{A}\hat{z}_B \end{bmatrix}$$

2. **변환 해석**: $R$은 좌표를 좌표계 $\{B\}$에서 좌표계 $\{A\}$로 변환한다:

$${}^{A}\mathbf{p} = {}^{A}_{B}R \cdot {}^{B}\mathbf{p}$$

---

## 오일러 각도(Euler Angles)

### 정의

오일러 각도는 회전을 좌표 축에 대한 세 번의 기본 회전 시퀀스로 매개변수화한다. 가능한 관례는 12가지이다 ($\{x, y, z\}$에서 인접하지 않은 축 3개를 선택).

로봇공학에서 가장 일반적인 두 가지:

| 관례 | 시퀀스 | 일반 명칭 | 일반적 사용 |
|-----------|----------|-------------|-------------|
| ZYX | $R_z(\psi) R_y(\theta) R_x(\phi)$ | 요-피치-롤(Yaw-Pitch-Roll) | 항공우주, 이동 로봇 |
| ZXZ | $R_z(\alpha) R_x(\beta) R_z(\gamma)$ | 고유 오일러(Proper Euler) | 손목 방향 |

### ZYX (요-피치-롤, Yaw-Pitch-Roll)

$$R_{ZYX}(\psi, \theta, \phi) = R_z(\psi) \cdot R_y(\theta) \cdot R_x(\phi)$$

여기서:
- $\psi$ = 요(yaw, $z$축 회전)
- $\theta$ = 피치(pitch, $y'$축 회전)
- $\phi$ = 롤(roll, $x''$축 회전)

```python
def euler_zyx_to_rotation(yaw, pitch, roll):
    """Convert ZYX Euler angles to rotation matrix.

    Why ZYX? In aerospace and mobile robotics, we think of orientation
    as: first face a compass heading (yaw), then tilt up/down (pitch),
    then tilt sideways (roll). This matches our physical intuition.
    """
    return rotz(yaw) @ roty(pitch) @ rotx(roll)

def rotation_to_euler_zyx(R):
    """Extract ZYX Euler angles from rotation matrix.

    Why the atan2 function? Unlike atan, atan2 returns the correct
    quadrant for the angle, handling the full [-pi, pi] range.

    WARNING: This fails at pitch = +/- 90 degrees (gimbal lock).
    """
    # Check for gimbal lock
    if abs(R[2, 0]) >= 1.0 - 1e-10:
        # Gimbal lock: pitch = +/- 90 degrees
        yaw = np.arctan2(-R[0, 1], R[0, 2])  # or any consistent choice
        pitch = -np.arcsin(np.clip(R[2, 0], -1, 1))
        roll = 0.0  # arbitrary — one DOF is lost
        print("WARNING: Gimbal lock detected!")
    else:
        pitch = -np.arcsin(R[2, 0])
        yaw = np.arctan2(R[1, 0], R[0, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])

    return yaw, pitch, roll

# Example: yaw=30 deg, pitch=45 deg, roll=60 deg
yaw, pitch, roll = np.radians(30), np.radians(45), np.radians(60)
R = euler_zyx_to_rotation(yaw, pitch, roll)
yaw_r, pitch_r, roll_r = rotation_to_euler_zyx(R)
print(f"Original: yaw={np.degrees(yaw):.1f}, pitch={np.degrees(pitch):.1f}, "
      f"roll={np.degrees(roll):.1f}")
print(f"Recovered: yaw={np.degrees(yaw_r):.1f}, pitch={np.degrees(pitch_r):.1f}, "
      f"roll={np.degrees(roll_r):.1f}")
```

### 짐벌 잠금(Gimbal Lock) 문제

**짐벌 잠금(Gimbal lock)**은 두 회전 축이 정렬되어 자유도 하나가 손실될 때 발생한다. ZYX 오일러 각도에서는 피치 $\theta = \pm 90°$일 때 발생한다.

$\theta = 90°$일 때:

$$R = R_z(\psi) R_y(90°) R_x(\phi) = \begin{bmatrix} 0 & \sin(\phi-\psi) & \cos(\phi-\psi) \\ 0 & -\cos(\phi-\psi) & \sin(\phi-\psi) \\ -1 & 0 & 0 \end{bmatrix}$$

주목할 점: *차이* $\phi - \psi$만 나타난다 — $\phi$와 $\psi$를 개별적으로 결정할 수 없다. 회전 자유도 하나가 사라진다.

```python
# Gimbal lock demonstration
# At pitch = 90 degrees, changing yaw and roll have the same effect

R1 = euler_zyx_to_rotation(
    yaw=np.radians(30), pitch=np.radians(90), roll=np.radians(0))
R2 = euler_zyx_to_rotation(
    yaw=np.radians(0), pitch=np.radians(90), roll=np.radians(-30))

# These produce the SAME rotation matrix!
print("Are R1 and R2 equal?", np.allclose(R1, R2))  # True
# This means yaw=30,roll=0 and yaw=0,roll=-30 are indistinguishable
# at pitch=90 — we've lost one DOF.
```

> **실제 결과**: 아폴로 11호 우주선은 유도 시스템에 오일러 각도를 사용했다. 엔지니어들은 우주선이 짐벌 잠금 방향에 가까워지지 않도록 해야 했으며, 이는 임무 계획에 제약을 추가했다. 현대 시스템은 이를 완전히 피하기 위해 쿼터니언을 사용한다.

---

## 축-각도 표현(Axis-Angle Representation)

### 오일러의 회전 정리(Euler's Rotation Theorem)

3D의 모든 회전은 고정 축 $\hat{\omega}$ (단위 벡터)를 중심으로 각도 $\theta$만큼의 단일 회전으로 기술할 수 있다. 이것이 **오일러의 회전 정리**이다.

축-각도 표현은: $(\hat{\omega}, \theta)$, 여기서 $\hat{\omega} \in \mathbb{R}^3$, $\|\hat{\omega}\| = 1$, $\theta \in [0, \pi]$.

또는 단일 벡터 $\boldsymbol{\omega} = \theta \hat{\omega}$ (회전 벡터)로 결합할 수 있으며, 방향이 축을 인코딩하고 크기가 각도를 인코딩한다.

### 로드리게스의 회전 공식(Rodrigues' Rotation Formula)

축 $\hat{\omega}$와 각도 $\theta$가 주어졌을 때, 회전 행렬은:

$$R = I + \sin\theta \, [\hat{\omega}]_\times + (1 - \cos\theta) \, [\hat{\omega}]_\times^2$$

여기서 $[\hat{\omega}]_\times$는 $\hat{\omega}$의 **반대칭 행렬(skew-symmetric matrix)**이다:

$$[\hat{\omega}]_\times = \begin{bmatrix} 0 & -\omega_3 & \omega_2 \\ \omega_3 & 0 & -\omega_1 \\ -\omega_2 & \omega_1 & 0 \end{bmatrix}$$

반대칭 행렬은 벡터곱(cross product)을 구현한다: $[\hat{\omega}]_\times \mathbf{v} = \hat{\omega} \times \mathbf{v}$.

```python
def skew(w):
    """Create skew-symmetric matrix from 3D vector.

    Why skew-symmetric? It encodes the cross product as matrix multiplication.
    This is the bridge between the Lie algebra so(3) and the Lie group SO(3),
    which becomes important in advanced robotics (screw theory, exponential maps).
    """
    return np.array([[    0, -w[2],  w[1]],
                     [ w[2],     0, -w[0]],
                     [-w[1],  w[0],     0]])

def rodrigues(omega_hat, theta):
    """Rodrigues' rotation formula: axis-angle to rotation matrix.

    Why this formula? It provides a direct, singularity-free way to
    compute a rotation matrix from a physically intuitive representation
    (axis + angle). No gimbal lock possible.
    """
    K = skew(omega_hat)
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R

def rotation_to_axis_angle(R):
    """Extract axis-angle from rotation matrix.

    Uses the fact that:
    - theta = arccos((trace(R) - 1) / 2)
    - omega is the eigenvector of R with eigenvalue 1
    """
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))

    if abs(theta) < 1e-10:
        # No rotation — axis is undefined
        return np.array([0, 0, 1]), 0.0

    if abs(theta - np.pi) < 1e-10:
        # theta = pi: use eigenvector method
        # R + I has rank 1; any non-zero column is proportional to omega
        M = R + np.eye(3)
        for col in range(3):
            if np.linalg.norm(M[:, col]) > 1e-10:
                omega_hat = M[:, col] / np.linalg.norm(M[:, col])
                return omega_hat, theta

    # General case
    omega_hat = np.array([R[2,1] - R[1,2],
                          R[0,2] - R[2,0],
                          R[1,0] - R[0,1]]) / (2 * np.sin(theta))
    return omega_hat, theta

# Example: 120 degrees about the [1,1,1]/sqrt(3) axis
omega_hat = np.array([1, 1, 1]) / np.sqrt(3)
theta = np.radians(120)
R = rodrigues(omega_hat, theta)
print("Rotation matrix:")
print(np.round(R, 4))

# Verify: recover axis-angle
omega_recovered, theta_recovered = rotation_to_axis_angle(R)
print(f"\nRecovered axis: {np.round(omega_recovered, 4)}")
print(f"Recovered angle: {np.degrees(theta_recovered):.1f} degrees")
```

### 지수 사상(Exponential Map)

로드리게스 공식은 실제로 반대칭 행렬의 **행렬 지수(matrix exponential)**이다:

$$R = e^{[\hat{\omega}]_\times \theta} = I + \sin\theta \, [\hat{\omega}]_\times + (1 - \cos\theta) \, [\hat{\omega}]_\times^2$$

반대칭 행렬(**리 대수(Lie algebra)** $\mathfrak{so}(3)$)과 회전 행렬(**리 군(Lie group)** SO(3)) 사이의 이 연결은 현대 로봇공학, 특히 스크류 이론(screw theory)과 다양체에서의 최적화에 근본적으로 중요하다.

```python
from scipy.linalg import expm

# The matrix exponential gives the same result as Rodrigues' formula
omega_hat = np.array([0, 0, 1])  # z-axis
theta = np.radians(45)
K = skew(omega_hat)

R_rodrigues = rodrigues(omega_hat, theta)
R_expm = expm(K * theta)  # matrix exponential

print(f"Rodrigues == expm? {np.allclose(R_rodrigues, R_expm)}")  # True
```

---

## 쿼터니언(Quaternions)

### 동기

오일러 각도는 짐벌 잠금으로 인해 문제가 있다. 회전 행렬은 3 DOF에 대해 9개의 숫자(6개의 제약 조건 포함)를 사용한다. 쿼터니언은 4개의 숫자(1개의 제약 조건)를 사용 — 더 간결하고, 특이점이 없으며, 보간에 수치적으로 안정적이다.

### 정의

**단위 쿼터니언(unit quaternion)** $\mathbf{q}$은 단위 노름(unit norm)을 갖는 4차원 벡터이다:

$$\mathbf{q} = q_w + q_x \mathbf{i} + q_y \mathbf{j} + q_z \mathbf{k} = (q_w, \mathbf{q}_v)$$

여기서:
- $q_w$는 **스칼라(scalar)** (실수) 부분
- $\mathbf{q}_v = (q_x, q_y, q_z)$는 **벡터(vector)** (허수) 부분
- $\|\mathbf{q}\| = \sqrt{q_w^2 + q_x^2 + q_y^2 + q_z^2} = 1$

허수 단위는 해밀턴의 방정식(Hamilton's equations)을 만족한다:

$$\mathbf{i}^2 = \mathbf{j}^2 = \mathbf{k}^2 = \mathbf{ijk} = -1$$

### 축-각도로부터 쿼터니언

회전 축 $\hat{\omega}$와 각도 $\theta$가 주어졌을 때:

$$\mathbf{q} = \left(\cos\frac{\theta}{2}, \, \sin\frac{\theta}{2} \, \hat{\omega}\right)$$

반각(half-angle)에 주목하라! 단위 쿼터니언이 SO(3)를 이중 피복(double-covers)하기 때문이다: $\mathbf{q}$와 $-\mathbf{q}$는 동일한 회전을 나타낸다.

```python
class Quaternion:
    """Unit quaternion for 3D rotation.

    Why a class? Quaternion arithmetic (especially multiplication) has
    specific rules that differ from regular vector operations. Encapsulating
    these in a class prevents errors and makes code readable.

    Convention: q = (w, x, y, z) where w is the scalar part.
    """
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z
        self._normalize()

    def _normalize(self):
        """Ensure unit norm. Numerical drift can violate this."""
        norm = np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        if norm > 1e-10:
            self.w /= norm
            self.x /= norm
            self.y /= norm
            self.z /= norm

    @classmethod
    def from_axis_angle(cls, axis, angle):
        """Create quaternion from axis-angle representation.

        Why half-angle? Because quaternion rotation uses q * p * q_conj,
        which applies the rotation twice (once from each side), so each
        side contributes half the angle.
        """
        axis = np.array(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)  # ensure unit vector
        half = angle / 2
        w = np.cos(half)
        x, y, z = np.sin(half) * axis
        return cls(w, x, y, z)

    def conjugate(self):
        """Quaternion conjugate — negates the vector part.

        For unit quaternions, the conjugate equals the inverse.
        This represents the reverse rotation.
        """
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def __mul__(self, other):
        """Hamilton product (quaternion multiplication).

        This is the core operation: composing two rotations.
        The formula follows directly from i^2 = j^2 = k^2 = ijk = -1.
        """
        w1, x1, y1, z1 = self.w, self.x, self.y, self.z
        w2, x2, y2, z2 = other.w, other.x, other.y, other.z

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        return Quaternion(w, x, y, z)

    def rotate_vector(self, v):
        """Rotate a 3D vector using this quaternion.

        The rotation formula: v' = q * v_quat * q_conj
        where v_quat = (0, v_x, v_y, v_z) — a "pure" quaternion.
        """
        v_quat = Quaternion(0, v[0], v[1], v[2])
        result = self * v_quat * self.conjugate()
        return np.array([result.x, result.y, result.z])

    def to_rotation_matrix(self):
        """Convert to 3x3 rotation matrix.

        Why provide this? Many algorithms (FK, Jacobian) work with
        rotation matrices. Quaternions are better for storage, interpolation,
        and composition; matrices are better for transforming many points.
        """
        w, x, y, z = self.w, self.x, self.y, self.z
        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z),     2*(x*z + w*y)],
            [2*(x*y + w*z),     1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x**2 + y**2)]
        ])

    def __repr__(self):
        return f"Q({self.w:.4f}, {self.x:.4f}, {self.y:.4f}, {self.z:.4f})"


# Example: 90 degrees about z-axis
q = Quaternion.from_axis_angle([0, 0, 1], np.pi/2)
print(f"Quaternion: {q}")

# Rotate (1, 0, 0) — should give (0, 1, 0)
v = np.array([1.0, 0.0, 0.0])
v_rotated = q.rotate_vector(v)
print(f"Rotated vector: {np.round(v_rotated, 4)}")  # [0, 1, 0]

# Compose two 90-degree rotations about z — should give 180 degrees
q2 = q * q
print(f"Double rotation: {q2}")
v_double = q2.rotate_vector(v)
print(f"After 180 deg: {np.round(v_double, 4)}")  # [-1, 0, 0]

# Verify: quaternion → matrix matches Rodrigues
R_quat = q.to_rotation_matrix()
R_rodrigues = rodrigues(np.array([0, 0, 1]), np.pi/2)
print(f"Matrix match: {np.allclose(R_quat, R_rodrigues)}")  # True
```

### 쿼터니언 보간 (SLERP)

쿼터니언의 가장 큰 장점 중 하나는 부드러운 보간이다. **SLERP**(구면 선형 보간, Spherical Linear Interpolation)은 두 방향 사이에서 등속 회전을 생성한다:

$$\text{slerp}(\mathbf{q}_0, \mathbf{q}_1, t) = \frac{\sin((1-t)\Omega)}{\sin\Omega} \mathbf{q}_0 + \frac{\sin(t\Omega)}{\sin\Omega} \mathbf{q}_1$$

여기서 $\Omega = \arccos(\mathbf{q}_0 \cdot \mathbf{q}_1)$, $t \in [0, 1]$.

```python
def slerp(q0, q1, t):
    """Spherical linear interpolation between two quaternions.

    Why SLERP instead of linear interpolation? Linear interpolation of
    rotation matrices or Euler angles produces non-uniform angular velocity
    and can pass through non-rotation states. SLERP follows the shortest
    path on the unit sphere at constant angular velocity.
    """
    # Ensure shortest path (q and -q are the same rotation)
    dot = q0.w*q1.w + q0.x*q1.x + q0.y*q1.y + q0.z*q1.z
    if dot < 0:
        q1 = Quaternion(-q1.w, -q1.x, -q1.y, -q1.z)
        dot = -dot

    dot = np.clip(dot, -1, 1)

    if dot > 0.9995:
        # Very close — use linear interpolation to avoid numerical issues
        w = q0.w + t * (q1.w - q0.w)
        x = q0.x + t * (q1.x - q0.x)
        y = q0.y + t * (q1.y - q0.y)
        z = q0.z + t * (q1.z - q0.z)
        return Quaternion(w, x, y, z)

    omega = np.arccos(dot)
    sin_omega = np.sin(omega)

    s0 = np.sin((1 - t) * omega) / sin_omega
    s1 = np.sin(t * omega) / sin_omega

    w = s0 * q0.w + s1 * q1.w
    x = s0 * q0.x + s1 * q1.x
    y = s0 * q0.y + s1 * q1.y
    z = s0 * q0.z + s1 * q1.z

    return Quaternion(w, x, y, z)

# Interpolate from identity (no rotation) to 90 deg about z
q_start = Quaternion(1, 0, 0, 0)  # identity
q_end = Quaternion.from_axis_angle([0, 0, 1], np.pi/2)

for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
    q_t = slerp(q_start, q_end, t)
    v = q_t.rotate_vector(np.array([1, 0, 0]))
    angle = np.degrees(np.arctan2(v[1], v[0]))
    print(f"t={t:.2f}: angle = {angle:.1f} deg")
```

### 회전 표현 비교

| 표현 | 매개변수 수 | 특이점 없음? | 합성 | 보간 | 사용 사례 |
|---------------|------------|-------------------|-------------|---------------|----------|
| 회전 행렬 | 9 (제약 6) | 예 | 행렬 곱 | 불량 (SO(3) 위에 없음) | FK, 야코비안 |
| 오일러 각도 | 3 | 아니오 (짐벌 잠금) | 복잡한 공식 | 불량 | 사람 인터페이스 |
| 축-각도 | 4 (제약 1) | $\theta=0$ 근처에서 특이점 | 로드리게스 | 보통 | 시각화 |
| 쿼터니언 | 4 (제약 1) | 예 | 해밀턴 곱 | SLERP (탁월) | 저장, 보간 |

---

## 동차 변환 — SE(3)

### 회전과 이동의 결합

3D에서 강체 변환은 회전 $R$과 이동 $\mathbf{d}$로 구성된다:

$${}^{A}\mathbf{p} = {}^{A}_{B}R \cdot {}^{B}\mathbf{p} + {}^{A}\mathbf{d}_{B}$$

이를 **4x4 동차 변환 행렬(homogeneous transformation matrix)**로 간결하게 인코딩한다:

$${}^{A}_{B}T = \begin{bmatrix} {}^{A}_{B}R & {}^{A}\mathbf{d}_B \\ \mathbf{0}^T & 1 \end{bmatrix} = \begin{bmatrix} r_{11} & r_{12} & r_{13} & d_x \\ r_{21} & r_{22} & r_{23} & d_y \\ r_{31} & r_{32} & r_{33} & d_z \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

**동차 좌표(homogeneous coordinates)**를 사용한다: 3D 점 $(x, y, z)$는 $(x, y, z, 1)^T$가 된다.

$$\begin{bmatrix} {}^{A}\mathbf{p} \\ 1 \end{bmatrix} = {}^{A}_{B}T \begin{bmatrix} {}^{B}\mathbf{p} \\ 1 \end{bmatrix}$$

```python
def make_transform(R, d):
    """Create 4x4 homogeneous transformation matrix.

    Why 4x4? Because combining rotation and translation into a single
    matrix allows us to compose multiple transformations by simple
    matrix multiplication — the same operation for rotation alone.
    This is the key insight of homogeneous coordinates.
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = d
    return T

def transform_point(T, p):
    """Transform a 3D point using a homogeneous transformation."""
    p_h = np.array([p[0], p[1], p[2], 1.0])  # homogeneous coordinates
    p_transformed = T @ p_h
    return p_transformed[:3]  # back to 3D

def inverse_transform(T):
    """Compute the inverse of a homogeneous transformation.

    Why not just np.linalg.inv? Because we can exploit the structure:
    T_inv = [[R^T, -R^T * d], [0, 1]]
    This is faster and more numerically stable.
    """
    R = T[:3, :3]
    d = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ d
    return T_inv

# Example: frame B is rotated 90 deg about z and translated (1, 0, 0.5)
R = rotz(np.pi/2)
d = np.array([1.0, 0.0, 0.5])
T_AB = make_transform(R, d)

print("T_AB:")
print(np.round(T_AB, 4))

# Transform a point (1, 0, 0) in frame B to frame A
p_B = np.array([1.0, 0.0, 0.0])
p_A = transform_point(T_AB, p_B)
print(f"\nPoint in B: {p_B}")
print(f"Point in A: {np.round(p_A, 4)}")  # [1, 1, 0.5]

# Verify inverse
T_BA = inverse_transform(T_AB)
p_B_recovered = transform_point(T_BA, p_A)
print(f"Recovered in B: {np.round(p_B_recovered, 4)}")  # [1, 0, 0]
```

### SE(3) 군

모든 동차 변환 행렬의 집합은 **SE(3)** — 3차원 특수 유클리드 군(Special Euclidean group) — 을 형성한다. 이는 3D 공간에서의 모든 강체 운동(회전 + 이동)을 나타낸다.

**성질**:
- **닫힘**: $T_1 \cdot T_2 \in SE(3)$
- **항등원**: $I_{4\times4} \in SE(3)$
- **역원**: $T^{-1} \in SE(3)$ (위에서 계산한 바와 같이)
- **교환법칙 불성립**: 변환의 순서가 중요함

---

## 변환의 합성

### 연쇄 법칙(Chain Rule)

좌표계 $\{C\}$의 $\{B\}$에 대한 자세와 $\{B\}$의 $\{A\}$에 대한 자세를 알고 있다면:

$${}^{A}_{C}T = {}^{A}_{B}T \cdot {}^{B}_{C}T$$

이것이 **연쇄 법칙(chain rule)**이며 순운동학(레슨 3)의 수학적 기초이다.

```python
# Example: a robotic arm with base frame {0}, elbow frame {1}, and tool frame {2}

# Link 1: rotate 45 deg about z, translate 1m along new x
T_01 = make_transform(rotz(np.radians(45)), np.array([1.0, 0.0, 0.0]))

# Link 2: rotate 30 deg about z, translate 0.8m along new x
T_12 = make_transform(rotz(np.radians(30)), np.array([0.8, 0.0, 0.0]))

# Compose: tool frame in base frame
T_02 = T_01 @ T_12

print("T_01 (base to elbow):")
print(np.round(T_01, 4))
print("\nT_12 (elbow to tool):")
print(np.round(T_12, 4))
print("\nT_02 (base to tool) = T_01 * T_12:")
print(np.round(T_02, 4))

# Where is the tool origin in the base frame?
tool_position = T_02[:3, 3]
print(f"\nTool position in base frame: {np.round(tool_position, 4)}")

# What orientation does the tool have?
total_angle = np.degrees(np.arctan2(T_02[1, 0], T_02[0, 0]))
print(f"Tool orientation (z-rotation): {total_angle:.1f} degrees")  # 75 deg
```

### 고정 좌표계 vs 몸체 좌표계 회전

중요한 미묘함: 행렬 곱셈의 순서는 회전이 **고정(세계) 축**에 대한 것인지 **몸체(현재) 축**에 대한 것인지에 따라 달라진다.

- **몸체 좌표계(Body frame)** (뒤에 곱하기, post-multiply): $T = T_1 \cdot T_2 \cdot T_3$ — 각 회전이 *현재* 좌표계의 축에 대한 것
- **고정 좌표계(Fixed frame)** (앞에 곱하기, pre-multiply): $T = T_3 \cdot T_2 \cdot T_1$ — 각 회전이 *고정된 세계* 좌표계의 축에 대한 것

DH 관례(레슨 3)에서는 몸체 좌표계 (뒤에 곱하기) 관례를 사용한다.

```python
# Body frame vs fixed frame rotation
# Rotate 90 deg about z, then 90 deg about y (body frame)
T_body = rotz(np.pi/2) @ roty(np.pi/2)

# The same final orientation using fixed frame:
# must reverse the order
T_fixed = roty(np.pi/2) @ rotz(np.pi/2)

# These are different!
print("Body frame (Rz then Ry_body):")
print(np.round(T_body, 3))
print("\nFixed frame (Ry_fixed then Rz_fixed):")
print(np.round(T_fixed, 3))
print(f"\nSame? {np.allclose(T_body, T_fixed)}")  # False in general
```

---

## 실용적 고려 사항

### 회전 행렬의 수치적 문제

회전 행렬을 반복적으로 곱하면 수치 표류(numerical drift)가 발생한다 — 결과가 더 이상 정확히 직교하지 않을 수 있다. SVD를 사용해 주기적으로 **재직교화(re-orthogonalize)**하라:

```python
def reorthogonalize(R):
    """Re-orthogonalize a rotation matrix using SVD.

    Why SVD? It finds the closest orthogonal matrix in the Frobenius norm
    sense. This is crucial in real-time robot control where thousands of
    matrix multiplications accumulate floating-point errors.
    """
    U, _, Vt = np.linalg.svd(R)
    R_clean = U @ Vt
    # Ensure det = +1 (not a reflection)
    if np.linalg.det(R_clean) < 0:
        U[:, -1] *= -1
        R_clean = U @ Vt
    return R_clean

# Simulate numerical drift
R = rotz(0.01)  # tiny rotation
R_accumulated = np.eye(3)
for _ in range(100000):
    R_accumulated = R_accumulated @ R

print(f"det(R) after 100k multiplications: {np.linalg.det(R_accumulated):.10f}")
print(f"R^T R == I? {np.allclose(R_accumulated.T @ R_accumulated, np.eye(3))}")

R_clean = reorthogonalize(R_accumulated)
print(f"\nAfter re-orthogonalization:")
print(f"det(R) = {np.linalg.det(R_clean):.10f}")
print(f"R^T R == I? {np.allclose(R_clean.T @ R_clean, np.eye(3))}")
```

### 어떤 표현을 언제 사용할 것인가

| 시나리오 | 최적 표현 | 이유 |
|----------|-------------------|--------|
| 순운동학 | 회전 행렬 / SE(3) | 직접 연쇄 곱셈 |
| 방향 저장 | 쿼터니언 | 간결(4개), 특이점 없음 |
| 방향 보간 | 쿼터니언 (SLERP) | 부드럽고 등속 |
| 사람이 읽기 쉬운 표시 | 오일러 각도 | 직관적 (요/피치/롤) |
| SO(3)에서 최적화 | 축-각도 / 리 대수 | 최소 매개변수화 |
| 실시간 제어 루프 | 쿼터니언 | 빠른 합성, 쉬운 재정규화 |

---

## 요약

- **회전 행렬**은 행렬식 +1을 갖는 3x3 직교 행렬로, SO(3) 군을 형성한다
- **오일러 각도**는 3개의 매개변수를 사용하지만 특정 구성에서 **짐벌 잠금**으로 인해 문제가 있다
- **축-각도** (로드리게스 공식)는 물리적 축과 각도로부터 특이점 없는 회전을 제공한다
- **쿼터니언**은 4개의 매개변수를 사용하고, 특이점이 없으며, SLERP를 통한 부드러운 보간이 가능하다
- **동차 변환** (SE(3))은 회전과 이동을 단일 4x4 행렬로 결합한다
- 변환의 **합성**은 행렬 곱셈으로 이루어지며, 이것이 순운동학의 기초를 형성한다

---

## 연습 문제

### 연습 문제 1: 회전 행렬 성질

$R = R_z(30°) \cdot R_x(45°)$에 대해:
1. $R$을 수치적으로 계산하라
2. $R^T R = I$이고 $\det(R) = 1$임을 검증하라
3. $R^{-1}$은 무엇인가? $R \cdot R^{-1}$을 계산하여 검증하라

### 연습 문제 2: 짐벌 잠금 탐구

1. ZYX 오일러 각도를 받아 회전 행렬을 반환하는 함수를 작성하라
2. 피치를 89도로 설정하고 (요=10, 피치=89, 롤=20)과 (요=15, 피치=89, 롤=15)에 대한 행렬을 계산하라. 결과가 얼마나 다른가?
3. 이제 피치를 정확히 90도로 설정하라. 동일한 회전 행렬을 만드는 두 개의 다른 (요, 롤) 쌍을 찾아 추출된 요와 롤이 모호함을 보여라

### 연습 문제 3: 쿼터니언 연산

1. 다음에 대한 쿼터니언을 생성하라: (a) x축을 중심으로 90도, (b) y축을 중심으로 90도
2. 합성하라: 먼저 (a) 다음 (b), 먼저 (b) 다음 (a). 결과가 같은가?
3. SLERP를 사용해 항등 쿼터니언과 z축을 중심으로 180도 회전 사이를 5개의 등간격 $t$ 값으로 보간하라

### 연습 문제 4: 변환 연쇄

카메라가 로봇 끝단 효과기에 장착되어 있다. 다음이 주어졌을 때:
- ${}^{0}_{ee}T$: 베이스 좌표계에서 끝단 효과기 (z축 기준 45도 회전, 이동 $(0.5, 0.3, 0.8)$)
- ${}^{ee}_{cam}T$: 끝단 효과기 좌표계에서 카메라 (x축 기준 180도 회전, 이동 $(0, 0, 0.1)$)

1. 베이스 좌표계에서 카메라의 자세 ${}^{0}_{cam}T$를 계산하라
2. 카메라 좌표에서 점 $\mathbf{p}_{cam} = (0.2, 0.1, 1.0)$이 감지되었다. 베이스 좌표계에서의 위치를 구하라
3. 로봇이 이동해 ${}^{0}_{ee}T$가 변경된다면 (새 회전: z축 기준 90도, 동일한 이동), 카메라 자세를 다시 계산하라

### 연습 문제 5: 표현 변환 왕복 검증

1. 오일러 각도로 시작: 요 = 25도, 피치 = 40도, 롤 = -15도
2. 회전 행렬로 변환
3. 회전 행렬을 축-각도로 변환
4. 축-각도를 쿼터니언으로 변환
5. 쿼터니언을 회전 행렬로 다시 변환
6. 회전 행렬을 오일러 각도로 다시 변환
7. 원래 각도를 (수치 정밀도 범위 내에서) 복원했는지 검증하라

---

[← 이전: 로봇공학 개요](01_Robotics_Overview.md) | [다음: 순운동학 →](03_Forward_Kinematics.md)
