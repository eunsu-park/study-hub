# 순운동학(Forward Kinematics)

[← 이전: 강체 변환](02_Rigid_Body_Transformations.md) | [다음: 역운동학 →](04_Inverse_Kinematics.md)

## 학습 목표

1. 운동 체인(kinematic chain)을 설명하고, 회전 관절(revolute joint)과 직선 관절(prismatic joint)을 구분할 수 있다
2. 데나비트-하텐베르크(Denavit-Hartenberg, DH) 규약을 적용하여 각 관절에 좌표 프레임을 체계적으로 할당하고, 4개의 DH 매개변수를 추출할 수 있다
3. DH 매개변수로부터 개별 변환 행렬을 구성하고, 이를 곱하여 순운동학을 계산할 수 있다
4. 2-링크 평면 로봇, 3-자유도 공간 로봇, 6-자유도 매니퓰레이터 등 주요 로봇 구성에 대한 순운동학을 계산할 수 있다
5. 관절 구성을 탐색하여 매니퓰레이터의 작업 공간(workspace)을 분석할 수 있다
6. 파이썬으로 범용 FK 솔버를 구현할 수 있다

---

## 왜 중요한가

순운동학(Forward Kinematics)은 로보틱스에서 가장 근본적인 질문에 답한다: **"관절을 이 각도로 설정하면 말단 작동체(end-effector)는 어디에 위치하는가?"** 로봇이 움직일 때마다 컨트롤러는 실제 도구 위치를 추적하기 위해 FK를 초당 수천 번 계산한다. FK가 없으면 모터들이 맹목적으로 회전할 뿐이지만, FK가 있으면 로봇은 자신의 기하학적 형태를 파악할 수 있다.

FK는 이후 모든 내용의 기초이기도 하다. 역운동학(Lesson 4)은 FK를 역산한다. 야코비안(Jacobian, Lesson 5)은 FK의 도함수다. 동역학(Dynamics, Lesson 6)에서는 링크의 위치와 속도를 계산하기 위해 FK가 필요하다. 공식만이 아니라 FK를 유도하는 체계적인 절차를 깊이 이해하는 것은 모든 로보틱스 실무자에게 필수적이다.

> **비유**: DH 매개변수는 로봇 관절을 위한 GPS 좌표와 같다. 4개의 숫자가 각 관절의 위치를 이전 관절에 상대적으로 고유하게 정의한다. GPS가 (위도, 경도, 고도, 방위각)으로 지구상의 위치를 특정하듯이, DH는 $(d, \theta, a, \alpha)$로 운동 체인에서 하나의 관절 프레임을 다음 프레임에 상대적으로 특정한다.

---

## 운동 체인(Kinematic Chains)

### 링크와 관절

**운동 체인(kinematic chain)**은 **관절(joint)**로 연결된 일련의 강체(**링크(link)**)다. 로보틱스에서 두 가지 기본 관절 유형은 다음과 같다:

| 관절 유형 | 운동 | 자유도(DOF) | 변수 | 기호 |
|-----------|--------|-----|----------|--------|
| **회전 관절(Revolute, R)** | 축을 중심으로 회전 | 1 | 각도 $\theta$ | ![revolute](revolute) |
| **직선 관절(Prismatic, P)** | 축을 따라 이동 | 1 | 변위 $d$ | ![prismatic](prismatic) |

다른 관절 유형은 이 두 가지의 조합으로 분해할 수 있다:
- **유니버설 관절(Universal joint, U)** = 교차하는 수직 축을 가진 2개의 회전 관절 (2 DOF)
- **구형 관절(Spherical joint, S)** = 같은 점에서 만나는 축을 가진 3개의 회전 관절 (3 DOF)
- **원통형 관절(Cylindrical joint, C)** = 같은 축에서의 회전 관절 1개 + 직선 관절 1개 (2 DOF)

### 개방 체인과 폐쇄 체인

- **개방 체인(open chain, serial)**: 각 링크가 최대 두 개의 다른 링크에 연결된다. 베이스는 고정되고 말단 작동체는 자유롭다. (예: 산업용 로봇 팔)
- **폐쇄 체인(closed chain, parallel)**: 하나 이상의 링크가 세 개 이상의 다른 링크에 연결되어 루프를 형성한다. (예: 스튜어트 플랫폼)

이 레슨은 개방 체인에 집중하며, 이 경우 FK는 행렬 연쇄 곱으로 직접적으로 계산된다.

### 번호 매기기 규약

$n$-자유도 직렬 매니퓰레이터에서:
- **링크**: $0$ (베이스/지면)에서 $n$ (말단 작동체 링크)까지 번호를 매긴다
- **관절**: $1$에서 $n$까지 번호를 매기며, 관절 $i$는 링크 $i-1$과 링크 $i$를 연결한다
- **프레임**: 프레임 $\{i\}$는 링크 $i$에 부착된다

```
{0} ──[Joint 1]── {1} ──[Joint 2]── {2} ── ... ──[Joint n]── {n}
base                                                        end-effector
```

---

## 데나비트-하텐베르크(Denavit-Hartenberg, DH) 규약

### 동기

$n$-자유도 로봇에서 FK는 각 관절에 대해 프레임 $\{i-1\}$에서 프레임 $\{i\}$로의 변환을 기술해야 한다. 체계적인 규약 없이는 모든 로봇마다 맞춤 유도가 필요하다. DH 규약은 **관절당 정확히 4개의 매개변수**로 이를 표준화한다.

### 4개의 DH 매개변수

링크 $i-1$과 링크 $i$를 연결하는 관절 $i$에 대해:

| 매개변수 | 기호 | 설명 |
|-----------|--------|-------------|
| **링크 길이(Link length)** | $a_i$ | $x_i$를 따라 $z_{i-1}$에서 $z_i$까지의 거리 |
| **링크 비틀림(Link twist)** | $\alpha_i$ | $x_i$를 중심으로 $z_{i-1}$에서 $z_i$까지의 각도 |
| **링크 오프셋(Link offset)** | $d_i$ | $z_{i-1}$을 따라 $x_{i-1}$에서 $x_i$까지의 거리 |
| **관절 각도(Joint angle)** | $\theta_i$ | $z_{i-1}$을 중심으로 $x_{i-1}$에서 $x_i$까지의 각도 |

**회전 관절**의 경우, $\theta_i$가 관절 변수(변하는 값)다.
**직선 관절**의 경우, $d_i$가 관절 변수다.

### DH 프레임 할당 규칙

1. **$z_i$ 축**: 관절 $i+1$의 축 방향으로 정렬
2. **$x_i$ 축**: $z_{i-1}$에서 $z_i$까지의 공통 법선 방향 (즉, $x_i = z_{i-1} \times z_i / \|z_{i-1} \times z_i\|$)
3. **$y_i$ 축**: 오른손 좌표계 완성 ($y_i = z_i \times x_i$)
4. **프레임 $\{i\}$의 원점**: $z_i$와 $x_i$의 교점

**특수한 경우**:
- $z_{i-1}$과 $z_i$가 평행한 경우: $x_i$ 방향이 모호하므로 단순함을 위해 선택 (보통 원점을 잇는 선 방향)
- $z_{i-1}$과 $z_i$가 교차하는 경우: $x_i$는 두 축에 수직 (외적)

### 단계별 DH 절차

1. 모든 관절에 1부터 $n$까지 번호를 매긴다
2. 각 관절 축을 따라 $z_i$ 축을 할당한다
3. 공통 법선 규칙에 따라 $x_i$ 축을 할당한다
4. $x_i$와 $z_i$의 교점에 프레임 원점을 위치시킨다
5. 각 관절에 대해 4개의 DH 매개변수를 읽어낸다
6. 각 관절에 대한 변환 행렬을 구성한다
7. 모든 행렬을 곱한다

### DH 변환 행렬

프레임 $\{i-1\}$에서 프레임 $\{i\}$로의 변환은 다음과 같다:

$${}^{i-1}_{i}T = Rot_z(\theta_i) \cdot Trans_z(d_i) \cdot Trans_x(a_i) \cdot Rot_x(\alpha_i)$$

행렬 형태로:

$${}^{i-1}_{i}T = \begin{bmatrix} \cos\theta_i & -\sin\theta_i \cos\alpha_i & \sin\theta_i \sin\alpha_i & a_i \cos\theta_i \\ \sin\theta_i & \cos\theta_i \cos\alpha_i & -\cos\theta_i \sin\alpha_i & a_i \sin\theta_i \\ 0 & \sin\alpha_i & \cos\alpha_i & d_i \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

```python
import numpy as np

def dh_transform(theta, d, a, alpha):
    """Compute the 4x4 homogeneous transformation from DH parameters.

    Why this specific order (Rz, Tz, Tx, Rx)? It follows from the DH
    convention: first rotate about z_{i-1} by theta, then translate along
    z_{i-1} by d, then translate along x_i by a, then rotate about x_i
    by alpha. This specific decomposition ensures that exactly 4 parameters
    suffice for any joint configuration.

    Parameters:
        theta: joint angle (revolute variable) [rad]
        d: link offset (prismatic variable) [m]
        a: link length [m]
        alpha: link twist [rad]

    Returns:
        4x4 homogeneous transformation matrix
    """
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [ 0,     sa,     ca,    d],
        [ 0,      0,      0,    1]
    ])
```

---

## 예제 1: 2-링크 평면 로봇

가장 단순한 비자명적(non-trivial) 매니퓰레이터: 평면에 있는 두 개의 회전 관절.

```
        Joint 2
          o───────── End-effector
         /  Link 2
        /   (length l2)
Joint 1
  o────────o
  |  Link 1
  |  (length l1)
  |
 ===  Base (fixed)
```

### DH 매개변수

| 관절 $i$ | $\theta_i$ | $d_i$ | $a_i$ | $\alpha_i$ |
|-----------|-----------|-------|-------|------------|
| 1 | $\theta_1$ (변수) | 0 | $l_1$ | 0 |
| 2 | $\theta_2$ (변수) | 0 | $l_2$ | 0 |

모든 $\alpha_i = 0$ (모든 관절 축이 평면 밖으로 나란히 향함), 모든 $d_i = 0$ (평면 — $z$ 방향 오프셋 없음).

### 순운동학

$${}^{0}_{2}T = {}^{0}_{1}T \cdot {}^{1}_{2}T$$

말단 작동체의 위치:

$$x = l_1 \cos\theta_1 + l_2 \cos(\theta_1 + \theta_2)$$
$$y = l_1 \sin\theta_1 + l_2 \sin(\theta_1 + \theta_2)$$
$$\phi = \theta_1 + \theta_2 \quad \text{(말단 작동체 방위각)}$$

```python
def fk_2link_planar(theta1, theta2, l1, l2):
    """Forward kinematics for a 2-link planar robot.

    Why derive the closed-form? For simple robots, the closed-form is
    faster than matrix multiplication and gives physical insight.
    But for complex robots, we always use the general DH approach.
    """
    x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)
    phi = theta1 + theta2  # orientation angle

    return x, y, phi

def fk_2link_planar_dh(theta1, theta2, l1, l2):
    """Same FK using the general DH method (for verification).

    Why verify both ways? In robotics, cross-checking is essential.
    A bug in FK propagates to every downstream computation.
    """
    T01 = dh_transform(theta1, 0, l1, 0)
    T12 = dh_transform(theta2, 0, l2, 0)
    T02 = T01 @ T12
    x, y = T02[0, 3], T02[1, 3]
    phi = np.arctan2(T02[1, 0], T02[0, 0])
    return x, y, phi

# Test with specific joint angles
l1, l2 = 1.0, 0.8  # link lengths in meters
theta1 = np.radians(30)
theta2 = np.radians(45)

x1, y1, phi1 = fk_2link_planar(theta1, theta2, l1, l2)
x2, y2, phi2 = fk_2link_planar_dh(theta1, theta2, l1, l2)

print(f"Closed-form: x={x1:.4f}, y={y1:.4f}, phi={np.degrees(phi1):.1f} deg")
print(f"DH method:   x={x2:.4f}, y={y2:.4f}, phi={np.degrees(phi2):.1f} deg")
print(f"Match: {np.allclose([x1,y1,phi1], [x2,y2,phi2])}")
```

---

## 예제 2: 3-자유도 공간 로봇 (RPR)

더 흥미로운 예제: $\alpha$ 값이 0이 아닌 회전-직선-회전(Revolute-Prismatic-Revolute) 구성.

### DH 매개변수

| 관절 $i$ | $\theta_i$ | $d_i$ | $a_i$ | $\alpha_i$ |
|-----------|-----------|-------|-------|------------|
| 1 | $\theta_1$ (변수) | $d_1$ | 0 | $-90°$ |
| 2 | $-90°$ | $d_2$ (변수) | 0 | $90°$ |
| 3 | $\theta_3$ (변수) | 0 | 0 | 0 |

```python
def fk_3dof_rpr(theta1, d2, theta3, d1=0.5):
    """FK for a 3-DOF RPR spatial manipulator.

    Why this configuration? RPR (Revolute-Prismatic-Revolute) is common
    in cylindrical robots. The first revolute provides base rotation,
    the prismatic joint provides vertical reach, and the second revolute
    provides wrist rotation.
    """
    T01 = dh_transform(theta1, d1, 0, -np.pi/2)
    T12 = dh_transform(-np.pi/2, d2, 0, np.pi/2)
    T23 = dh_transform(theta3, 0, 0, 0)

    T03 = T01 @ T12 @ T23

    position = T03[:3, 3]
    orientation = T03[:3, :3]

    return T03, position, orientation

# Example configuration
T, pos, ori = fk_3dof_rpr(
    theta1=np.radians(30),
    d2=0.4,
    theta3=np.radians(60)
)
print(f"End-effector position: {np.round(pos, 4)}")
print(f"End-effector orientation:\n{np.round(ori, 4)}")
```

---

## 예제 3: 6-자유도 관절형 로봇 (PUMA 유사형)

표준 산업용 로봇: 특정 기하학적 배열을 가진 6개의 회전 관절. 이는 고전적인 PUMA 560 구성과 유사하다.

### DH 매개변수 (PUMA 560 유사형)

| 관절 | $\theta_i$ | $d_i$ | $a_i$ | $\alpha_i$ |
|-------|-----------|-------|-------|------------|
| 1 | $\theta_1$ | 0 | 0 | $-90°$ |
| 2 | $\theta_2$ | 0 | $a_2$ | 0 |
| 3 | $\theta_3$ | $d_3$ | $a_3$ | $-90°$ |
| 4 | $\theta_4$ | $d_4$ | 0 | $90°$ |
| 5 | $\theta_5$ | 0 | 0 | $-90°$ |
| 6 | $\theta_6$ | 0 | 0 | 0 |

관절 4, 5, 6은 **구형 손목(spherical wrist)**을 형성한다 — 이들의 축이 하나의 공통 점에서 교차한다. 이는 위치와 방위를 **분리(decouple)**하여 역운동학을 다룰 수 있게 만드는 매우 중요한 설계 선택이다 (Lesson 4).

```python
class SerialRobot:
    """General-purpose serial robot with DH parameterization.

    Why a class? Because we'll reuse this for FK, IK, Jacobian, and dynamics.
    Encapsulating the DH parameters and FK computation in one place avoids
    duplicated code and ensures consistency.
    """
    def __init__(self, name, dh_params):
        """
        dh_params: list of dicts, each with keys:
            'theta': nominal angle (for revolute: this is the offset, variable added at runtime)
            'd': link offset
            'a': link length
            'alpha': link twist
            'type': 'revolute' or 'prismatic'
        """
        self.name = name
        self.dh_params = dh_params
        self.n_joints = len(dh_params)

    def fk(self, q):
        """Compute forward kinematics for joint values q.

        Parameters:
            q: array of joint values (angles for revolute, displacements for prismatic)

        Returns:
            T: 4x4 homogeneous transformation (base to end-effector)
            transforms: list of intermediate transforms [{0}_T_{1}, {0}_T_{2}, ...]

        Why return intermediate transforms? They're needed for:
        - Jacobian computation (Lesson 5)
        - Dynamics computation (Lesson 6)
        - Visualization
        - Collision checking
        """
        assert len(q) == self.n_joints, \
            f"Expected {self.n_joints} joint values, got {len(q)}"

        T = np.eye(4)
        transforms = []

        for i, (params, qi) in enumerate(zip(self.dh_params, q)):
            if params['type'] == 'revolute':
                theta = params['theta'] + qi  # qi is the joint angle
                d = params['d']
            else:  # prismatic
                theta = params['theta']
                d = params['d'] + qi  # qi is the displacement

            Ti = dh_transform(theta, d, params['a'], params['alpha'])
            T = T @ Ti
            transforms.append(T.copy())

        return T, transforms

    def joint_positions(self, q):
        """Get 3D positions of all joints (for visualization).

        Returns list of (x, y, z) positions from base to end-effector.
        """
        _, transforms = self.fk(q)
        positions = [np.array([0, 0, 0])]  # base position
        for T in transforms:
            positions.append(T[:3, 3])
        return positions


# Define PUMA 560-like robot
puma_dh = [
    {'theta': 0, 'd': 0,    'a': 0,    'alpha': -np.pi/2, 'type': 'revolute'},
    {'theta': 0, 'd': 0,    'a': 0.4318, 'alpha': 0,       'type': 'revolute'},
    {'theta': 0, 'd': 0.15, 'a': 0.0203, 'alpha': -np.pi/2, 'type': 'revolute'},
    {'theta': 0, 'd': 0.4318, 'a': 0, 'alpha': np.pi/2,    'type': 'revolute'},
    {'theta': 0, 'd': 0,    'a': 0,    'alpha': -np.pi/2, 'type': 'revolute'},
    {'theta': 0, 'd': 0,    'a': 0,    'alpha': 0,        'type': 'revolute'},
]

puma = SerialRobot("PUMA 560", puma_dh)

# Home configuration (all zeros)
q_home = np.zeros(6)
T_home, _ = puma.fk(q_home)
print(f"PUMA 560 home position: {np.round(T_home[:3, 3], 4)}")

# Some non-trivial configuration
q_test = np.radians([30, -45, 60, 0, 30, 0])
T_test, _ = puma.fk(q_test)
print(f"PUMA 560 test position: {np.round(T_test[:3, 3], 4)}")
print(f"Test orientation:\n{np.round(T_test[:3, :3], 4)}")
```

---

## 작업 공간 분석

로봇의 **작업 공간(workspace)**은 말단 작동체가 도달할 수 있는 모든 위치의 집합이다. 관절 구성을 탐색하여 이를 시각화할 수 있다.

### 작업 공간의 종류

- **도달 가능 작업 공간(Reachable workspace)**: 말단 작동체 끝이 *하나 이상의* 방위로 도달할 수 있는 모든 점
- **유연 작업 공간(Dexterous workspace)**: *모든* 방위로 도달 가능한 모든 점 (도달 가능 작업 공간의 부분집합)
- **단면 작업 공간(Cross-section workspace)**: 3D 작업 공간의 2D 단면 (시각화에 유용)

### 작업 공간 계산

```python
def compute_workspace_2d(robot, n_samples=50):
    """Sample the workspace of a planar robot by grid search.

    Why grid search? For 2-DOF robots, uniform grid sampling gives
    a clear picture. For higher DOF, random sampling is more efficient
    because the grid grows exponentially.
    """
    positions = []

    # Create grid of joint values
    for i in range(robot.n_joints):
        pass  # We'll create ranges below

    # For a 2-DOF planar robot
    if robot.n_joints == 2:
        for q1 in np.linspace(-np.pi, np.pi, n_samples):
            for q2 in np.linspace(-np.pi, np.pi, n_samples):
                T, _ = robot.fk(np.array([q1, q2]))
                positions.append(T[:3, 3])

    return np.array(positions)

def compute_workspace_random(robot, joint_limits, n_samples=10000):
    """Sample workspace by random joint configurations.

    Why random sampling? For robots with > 3 DOF, grid sampling is
    impractical (e.g., 50 samples per joint x 6 joints = 15.6 billion
    configurations). Random sampling provides a reasonable approximation.
    """
    positions = []

    for _ in range(n_samples):
        q = np.array([
            np.random.uniform(lo, hi)
            for lo, hi in joint_limits
        ])
        T, _ = robot.fk(q)
        positions.append(T[:3, 3])

    return np.array(positions)

# Define 2-link planar robot using our SerialRobot class
planar_2link_dh = [
    {'theta': 0, 'd': 0, 'a': 1.0, 'alpha': 0, 'type': 'revolute'},
    {'theta': 0, 'd': 0, 'a': 0.8, 'alpha': 0, 'type': 'revolute'},
]
planar_2link = SerialRobot("2-Link Planar", planar_2link_dh)

# Compute workspace boundaries analytically
l1, l2 = 1.0, 0.8
r_max = l1 + l2      # fully extended
r_min = abs(l1 - l2)  # fully folded
print(f"Workspace: annular ring with r_min={r_min}, r_max={r_max}")
print(f"(When both joints have full rotation range)")

# Numerical verification
positions = compute_workspace_random(
    planar_2link,
    joint_limits=[(-np.pi, np.pi), (-np.pi, np.pi)],
    n_samples=5000
)
distances = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
print(f"Sampled distance range: [{distances.min():.3f}, {distances.max():.3f}]")
```

### 작업 공간에 영향을 미치는 요소

여러 요소가 작업 공간에 영향을 준다:

1. **링크 길이**: 링크가 길수록 작업 공간이 넓어진다
2. **관절 한계(Joint limits)**: 범위가 줄어들면 작업 공간이 작아진다
3. **관절 유형**: 직선 관절은 작업 공간을 선형으로 확장하고, 회전 관절은 곡선 경계를 만든다
4. **자유도 수**: 자유도가 많을수록 유연 작업 공간이 일반적으로 커진다
5. **구성**: 직렬 로봇은 크기 대비 넓은 작업 공간을 가지며, 병렬 로봇은 상대적으로 작업 공간이 좁다

---

## 수정 DH 규약(Modified DH Convention)

실제로 널리 사용되는 DH 규약은 **두 가지**가 있다:

| 측면 | 표준(Classic) DH | 수정(Craig) DH |
|--------|----------------------|---------------------|
| 프레임 배치 | 프레임 $\{i\}$는 링크 $i$의 끝에 위치 | 프레임 $\{i\}$는 링크 $i$의 시작에 위치 |
| 변환 순서 | $Rot_z \cdot Trans_z \cdot Trans_x \cdot Rot_x$ | $Rot_x \cdot Trans_x \cdot Rot_z \cdot Trans_z$ |
| 참고 문헌 | Denavit & Hartenberg (1955) | Craig 교재 |

수정 DH 규약(Craig)은 순서를 반대로 한다:

$${}^{i-1}_{i}T = Rot_x(\alpha_{i-1}) \cdot Trans_x(a_{i-1}) \cdot Rot_z(\theta_i) \cdot Trans_z(d_i)$$

```python
def dh_transform_modified(alpha_prev, a_prev, theta, d):
    """Modified DH transformation (Craig convention).

    Why two conventions? Historical reasons. Both give the same final
    result for FK if parameters are assigned consistently. However,
    the modified convention places frame {i} at the proximal end of
    link i, which some find more intuitive for dynamics computation.

    IMPORTANT: Never mix conventions! Check which one a paper or
    textbook uses before copying DH parameters.
    """
    ca = np.cos(alpha_prev)
    sa = np.sin(alpha_prev)
    ct = np.cos(theta)
    st = np.sin(theta)

    return np.array([
        [     ct,      -st,   0,    a_prev],
        [st * ca,  ct * ca, -sa, -sa * d  ],
        [st * sa,  ct * sa,  ca,  ca * d  ],
        [      0,        0,   0,        1 ]
    ])
```

> **경고**: 두 규약은 동일한 로봇에 대해 서로 다른 매개변수 할당을 사용한다. 출처가 다른 DH 매개변수를 사용할 때는 반드시 규약을 확인한 후 혼용하지 않도록 한다.

---

## 순운동학 검증

로보틱스에서 검증은 매우 중요하다 — FK의 버그는 이후의 모든 계산으로 전파된다.

### 검증 전략

```python
def verify_fk(robot, test_cases):
    """Verify FK against known configurations.

    Why formal verification? Because FK bugs can cause real robots to
    crash into obstacles or themselves. These test cases are like unit
    tests for your robot model.
    """
    all_passed = True

    for name, q, expected_pos, tol in test_cases:
        T, _ = robot.fk(q)
        actual_pos = T[:3, 3]
        error = np.linalg.norm(actual_pos - expected_pos)

        if error > tol:
            print(f"FAIL: {name} - error={error:.6f} > tol={tol}")
            print(f"  Expected: {expected_pos}")
            print(f"  Actual:   {np.round(actual_pos, 6)}")
            all_passed = False
        else:
            print(f"PASS: {name} - error={error:.6f}")

    return all_passed

# Test cases for 2-link planar robot
test_cases = [
    # (name, q, expected_position, tolerance)
    ("Home (both zero)", np.array([0, 0]),
     np.array([1.8, 0, 0]), 1e-10),

    ("Joint 1 = 90 deg, Joint 2 = 0", np.array([np.pi/2, 0]),
     np.array([0, 1.8, 0]), 1e-10),

    ("Both 90 deg", np.array([np.pi/2, np.pi/2]),
     np.array([-0.8, 1.0, 0]), 1e-10),

    ("Fully folded back", np.array([0, np.pi]),
     np.array([0.2, 0, 0]), 1e-10),
]

verify_fk(planar_2link, test_cases)
```

### 일관성 확인

1. **홈 위치**: $q = 0$일 때 결과가 "영점 구성(zero configuration)" 기하와 일치해야 한다
2. **단일 관절 운동**: 한 관절씩 움직이면 예측 가능한 운동이 나타나야 한다
3. **대칭성**: 대칭 로봇의 경우, 대칭적인 관절 구성은 대칭적인 포즈를 만들어야 한다
4. **작업 공간 경계**: 완전 펼침과 완전 접음은 각각 $l_1 + l_2$와 $|l_1 - l_2|$와 일치해야 한다
5. **교차 검증**: 알려진 FK 구현체(예: Python용 Robotics Toolbox)와 비교한다

---

## 범용 FK 솔버

지금까지 배운 내용을 종합하여 완전하고 재사용 가능한 FK 구현을 제시한다:

```python
class RobotFK:
    """Complete forward kinematics solver with DH parameterization.

    This class supports both standard and modified DH conventions,
    arbitrary combinations of revolute and prismatic joints,
    and provides intermediate frame transforms for downstream use.
    """
    def __init__(self, dh_params, convention='standard'):
        """
        dh_params: list of dicts with keys:
            For standard DH: 'theta', 'd', 'a', 'alpha', 'type'
            For modified DH: 'alpha_prev', 'a_prev', 'theta', 'd', 'type'
        convention: 'standard' or 'modified'
        """
        self.dh_params = dh_params
        self.convention = convention
        self.n = len(dh_params)

    def compute(self, q):
        """Compute FK for given joint configuration.

        Returns:
            T_0n: base-to-end-effector transform
            T_list: list of base-to-frame-i transforms (i=1..n)
            T_individual: list of frame-to-frame transforms (i-1 to i)
        """
        T_0i = np.eye(4)
        T_list = []
        T_individual = []

        for i in range(self.n):
            p = self.dh_params[i]

            if self.convention == 'standard':
                theta = p['theta'] + (q[i] if p['type'] == 'revolute' else 0)
                d = p['d'] + (q[i] if p['type'] == 'prismatic' else 0)
                Ti = dh_transform(theta, d, p['a'], p['alpha'])
            else:
                theta = p['theta'] + (q[i] if p['type'] == 'revolute' else 0)
                d = p['d'] + (q[i] if p['type'] == 'prismatic' else 0)
                Ti = dh_transform_modified(p['alpha_prev'], p['a_prev'], theta, d)

            T_individual.append(Ti)
            T_0i = T_0i @ Ti
            T_list.append(T_0i.copy())

        return T_0i, T_list, T_individual

    def end_effector_pose(self, q):
        """Convenience method: returns position and rotation matrix."""
        T, _, _ = self.compute(q)
        return T[:3, 3], T[:3, :3]

    def joint_origins(self, q):
        """Get all joint origin positions in the base frame."""
        _, T_list, _ = self.compute(q)
        origins = [np.zeros(3)]  # base
        for T in T_list:
            origins.append(T[:3, 3])
        return origins

    def print_joint_frames(self, q):
        """Print all intermediate frame poses (useful for debugging)."""
        _, T_list, _ = self.compute(q)
        for i, T in enumerate(T_list):
            pos = T[:3, 3]
            # Extract ZYX Euler angles for readability
            R = T[:3, :3]
            beta = -np.arcsin(np.clip(R[2, 0], -1, 1))
            alpha = np.arctan2(R[1, 0], R[0, 0])
            gamma = np.arctan2(R[2, 1], R[2, 2])
            print(f"Frame {i+1}: pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}), "
                  f"rpy=({np.degrees(gamma):.1f}, {np.degrees(beta):.1f}, "
                  f"{np.degrees(alpha):.1f}) deg")


# Usage example: SCARA robot (RRP + R for wrist)
scara_dh = [
    {'theta': 0, 'd': 0.4, 'a': 0.35, 'alpha': 0, 'type': 'revolute'},
    {'theta': 0, 'd': 0,   'a': 0.25, 'alpha': np.pi, 'type': 'revolute'},
    {'theta': 0, 'd': 0,   'a': 0,    'alpha': 0, 'type': 'prismatic'},
    {'theta': 0, 'd': 0,   'a': 0,    'alpha': 0, 'type': 'revolute'},
]

scara = RobotFK(scara_dh, convention='standard')

q_scara = np.array([np.radians(30), np.radians(-45), 0.1, np.radians(90)])
T_scara, _, _ = scara.compute(q_scara)
print("\nSCARA FK result:")
scara.print_joint_frames(q_scara)
print(f"\nEnd-effector:\n{np.round(T_scara, 4)}")
```

---

## 흔한 실수

### 1. 규약 혼용

표준 DH와 수정 DH 매개변수를 혼용하는 것이 가장 흔한 FK 오류다. 어떤 출처에서 DH 표를 사용하기 전에 항상 규약을 확인한다.

### 2. 각도 단위

DH 표에는 각도가 도(degree) 단위로 표기될 수 있지만, 삼각함수는 라디안(radian)을 필요로 한다. 항상 변환한다.

### 3. 프레임 할당의 모호성

연속된 $z$ 축이 평행할 경우 $x$ 축 방향이 임의적이다. 다른 선택을 하면 DH 매개변수가 달라지지만 FK 결과는 동일하다 (매개변수가 서로를 보상함).

### 4. 도구 프레임(Tool Frame)

DH 규약은 마지막 관절 프레임까지의 변환을 제공하며, 도구 끝이 아니다. 마지막에 **도구 변환(tool transform)** $T_{tool}$을 추가해야 한다:

$$T_{0,tool} = T_{0,n} \cdot T_{n,tool}$$

```python
# Adding a tool frame
# The tool tip may be offset from the last joint frame
T_tool = np.eye(4)
T_tool[2, 3] = 0.1  # tool extends 10 cm along z of last frame

T_0n, _, _ = scara.compute(q_scara)
T_0_tool = T_0n @ T_tool
print(f"Last joint position:  {np.round(T_0n[:3, 3], 4)}")
print(f"Tool tip position:    {np.round(T_0_tool[:3, 3], 4)}")
```

---

## 요약

- **운동 체인(kinematic chain)**은 회전(rotation) 또는 직선 이동(translation) 관절로 연결된 링크들로 구성된다
- **DH 규약**은 관절당 4개의 매개변수 $(\theta, d, a, \alpha)$를 사용하여 좌표 프레임을 할당한다
- 회전 관절에서는 $\theta$가 변수이고, 직선 관절에서는 $d$가 변수다
- **순운동학(Forward kinematics)**은 모든 관절 변환 행렬의 곱이다: $T_{0n} = T_{01} \cdot T_{12} \cdots T_{(n-1)n}$
- **작업 공간(workspace)**은 링크 길이, 관절 유형, 관절 한계에 의해 결정되는 말단 작동체의 모든 도달 가능 위치의 집합이다
- 표준(standard)과 수정(modified) 두 가지 DH 규약이 존재하며 — **절대 혼용하지 않는다**
- 실제 시스템에서 사용하기 전에 알려진 테스트 구성으로 반드시 FK를 **검증**한다

---

## 연습 문제

### 연습 1: DH 매개변수 할당

링크 길이가 $l_1 = 0.5$ m, $l_2 = 0.4$ m, $l_3 = 0.3$ m인 3-자유도 평면 로봇 (3개의 회전 관절, 모두 같은 평면에 있음):
1. 로봇을 그리고 DH 프레임을 할당한다
2. DH 매개변수 표를 작성한다
3. FK를 기호적으로 계산한다 (말단 작동체 $x$, $y$, $\phi$)
4. $\theta_1 = 30°$, $\theta_2 = 45°$, $\theta_3 = -30°$에 대해 수치적으로 검증한다

### 연습 2: SCARA 로봇

SCARA 로봇의 DH 매개변수가 다음과 같다:

| 관절 | $\theta$ | $d$ | $a$ | $\alpha$ | 유형 |
|-------|---------|-----|-----|---------|------|
| 1 | $\theta_1$ | 0.5 | 0.4 | 0 | R |
| 2 | $\theta_2$ | 0 | 0.3 | $\pi$ | R |
| 3 | 0 | $d_3$ | 0 | 0 | P |
| 4 | $\theta_4$ | 0 | 0 | 0 | R |

1. 이 로봇의 FK를 구현한다
2. $\theta_1 = 45°$, $\theta_2 = -30°$, $d_3 = 0.15$ m, $\theta_4 = 60°$에 대한 말단 작동체 포즈를 계산한다
3. 작업 공간의 형태는 어떤가? 위에서 바라본 스케치를 그린다

### 연습 3: 6-자유도 검증

레슨에서 정의한 PUMA 560 유사형 로봇을 사용하여:
1. $q = [0, -90°, 0, 0, 0, 0]$ (팔이 수직으로 위를 향하는 자세)에 대한 FK를 계산한다
2. $q = [90°, 0, 0, 0, 0, 0]$ (베이스가 90도 회전한 자세)에 대한 FK를 계산한다
3. 각 경우에 대해 결과가 기하학적으로 타당한지 검증한다

### 연습 4: 작업 공간 시각화

1. 2-링크 평면 로봇 ($l_1 = 1.0$, $l_2 = 0.8$)에 대해 관절 각도 격자를 사용하여 작업 공간을 샘플링한다
2. 샘플링된 점들을 플롯하여 고리형(annular) 작업 공간을 시각화한다
3. 관절 2를 $[-90°, 90°]$로 제한한다. 새로운 작업 공간을 플롯한다
4. 두 작업 공간의 면적을 수치적으로 계산한다

### 연습 5: 혼합 관절 로봇

다음 구조를 가진 3-자유도 로봇을 설계한다: 회전-직선-회전(Revolute-Prismatic-Revolute).
1. 유용한 작업 공간을 만드는 DH 매개변수를 선택한다 ("유용함"의 기준은 스스로 정한다)
2. FK를 구현하고 최소 3개의 테스트 구성으로 검증한다
3. 연습 1의 전체 회전 관절 3-자유도 평면 로봇과 작업 공간 형태를 비교한다

---

[← 이전: 강체 변환](02_Rigid_Body_Transformations.md) | [다음: 역운동학 →](04_Inverse_Kinematics.md)
