# 속도 기구학과 야코비안(Velocity Kinematics and the Jacobian)

[← 이전: 역기구학](04_Inverse_Kinematics.md) | [다음: 로봇 동역학 →](06_Robot_Dynamics.md)

## 학습 목표

1. 강체의 각속도와 선속도를 정의하고 서로 다른 기준 좌표계에서 표현한다
2. 매니퓰레이터 야코비안(Jacobian)을 유도하고, 이것이 조인트 공간과 작업 공간 사이의 기본 속도 매핑임을 이해한다
3. 기하 야코비안(geometric Jacobian)과 해석적 야코비안(analytical Jacobian)을 구분하고 각각을 언제 사용할지 안다
4. 야코비안의 특이값(singular value)을 통해 특이점(singularity)을 분석하고 그 물리적 의미를 파악한다
5. 힘/토크 매핑(정역학 쌍대성(statics duality))을 적용하여 끝단 작용기의 힘과 조인트 토크를 연결한다
6. 조작 가능도 타원체(manipulability ellipsoid)와 민첩성 척도(dexterity measure)를 계산하고 해석하여 로봇 형태 구성(configuration)을 평가한다

---

## 왜 중요한가

야코비안(Jacobian)은 로보틱스에서 가장 중요한 수학적 대상이라 해도 과언이 아니다. 야코비안은 어디에나 등장한다. 속도 제어(조인트 속도를 끝단 작용기 속도로 변환), 힘 제어(끝단 작용기 힘을 조인트 토크로 변환), 특이점 분석(로봇이 민첩성을 잃는 위치 판단), 궤적 추종(조인트 가속도 계산), 그리고 동역학(운동 방정식 구성)에 이르기까지 야코비안이 핵심 역할을 한다.

야코비안을 이해하면 기구학 퍼즐만 풀 수 있는 수준을 넘어, 제어기를 설계하고 작업 공간 품질을 분석하며 어떤 형태 구성에서 로봇이 무엇을 할 수 있고 무엇을 할 수 없는지를 추론할 수 있게 된다. 순기구학(forward kinematics)이 "어디에"를 알려준다면, 야코비안은 "얼마나 빠르게, 어느 방향으로"를 알려준다.

> **비유**: 야코비안은 기어비(gear ratio)와 같다. 기어비가 엔진 RPM을 바퀴 속도로 변환하듯(그리고 그 비율이 기어 선택에 따라 달라지듯), 야코비안은 조인트 속도를 끝단 작용기 속도로 변환한다(그리고 그 "비율"은 로봇의 형태 구성에 따라 달라진다).

---

## 각속도와 선속도

### 선속도(Linear Velocity)

공간에서 움직이는 점 $\mathbf{p}(t)$의 선속도는 단순히 다음과 같다.

$$\mathbf{v} = \dot{\mathbf{p}} = \frac{d\mathbf{p}}{dt}$$

### 각속도(Angular Velocity)

회전하는 강체는 회전 축과 회전 속도를 모두 표현하는 **각속도 벡터** $\boldsymbol{\omega}$를 갖는다. 회전 행렬 $R(t)$에 대해:

$$\dot{R}(t) = [\boldsymbol{\omega}]_\times R(t) \quad \text{(물체 좌표계)}$$
$$\dot{R}(t) = [\boldsymbol{\omega}_s]_\times R(t) \quad \text{여기서 } \boldsymbol{\omega}_s = R \boldsymbol{\omega} \text{ (공간 좌표계)}$$

여기서 $[\boldsymbol{\omega}]_\times$는 $\boldsymbol{\omega}$의 왜대칭 행렬(skew-symmetric matrix)이다.

**핵심 성질**: 각속도는 벡터로 더해진다(유한 회전과 달리 교환 법칙이 성립하지 않는 것과 대조적이다).

$$\boldsymbol{\omega}_{total} = \boldsymbol{\omega}_1 + \boldsymbol{\omega}_2$$

### 강체 위의 점의 속도

원점 $O$, 각속도 $\boldsymbol{\omega}$, 선속도 $\mathbf{v}_O$를 가진 강체 위의 점 $P$에 대해:

$$\mathbf{v}_P = \mathbf{v}_O + \boldsymbol{\omega} \times \mathbf{r}_{OP}$$

여기서 $\mathbf{r}_{OP}$는 $O$에 대한 $P$의 위치이다.

```python
import numpy as np

def skew(w):
    """왜대칭 행렬(외적 행렬).

    Why a matrix for cross product? Because it lets us express
    omega x r as a matrix-vector product [omega]_x * r, which
    integrates cleanly into the Jacobian formulation.
    """
    return np.array([[    0, -w[2],  w[1]],
                     [ w[2],     0, -w[0]],
                     [-w[1],  w[0],     0]])

def velocity_of_point(v_origin, omega, r_OP):
    """강체 위의 점 P의 속도.

    v_P = v_O + omega x r_OP

    This is the fundamental kinematic equation for rigid bodies.
    Every column of the Jacobian is essentially this equation
    applied to one joint's contribution.
    """
    return v_origin + np.cross(omega, r_OP)

# 예제: z축을 중심으로 10 rad/s로 회전하는 바퀴
omega = np.array([0, 0, 10])  # rad/s about z
r = np.array([0.3, 0, 0])     # point on the rim, 0.3m from center
v_point = velocity_of_point(np.zeros(3), omega, r)
print(f"Rim velocity: {v_point} m/s")  # [0, 3, 0] — tangential velocity
print(f"|v| = {np.linalg.norm(v_point):.1f} m/s")  # omega * r = 3 m/s
```

---

## 매니퓰레이터 야코비안

### 정의

**매니퓰레이터 야코비안** $J(\mathbf{q})$는 조인트 속도를 끝단 작용기 속도로 매핑하는 행렬이다.

$$\begin{bmatrix} \mathbf{v} \\ \boldsymbol{\omega} \end{bmatrix} = J(\mathbf{q}) \, \dot{\mathbf{q}}$$

여기서:
- $\mathbf{v} \in \mathbb{R}^3$은 끝단 작용기의 선속도
- $\boldsymbol{\omega} \in \mathbb{R}^3$은 끝단 작용기의 각속도
- $\dot{\mathbf{q}} \in \mathbb{R}^n$은 조인트 속도 벡터
- $J(\mathbf{q}) \in \mathbb{R}^{6 \times n}$ (6자유도 전체 작업 공간의 경우)

### 야코비안 열: 각 조인트의 기여

$J$의 $i$번째 열은 $i$번째 조인트에서 단위 속도가 발생할 때(나머지 조인트는 정지) 끝단 작용기에 생기는 속도를 나타낸다. $n$자유도 직렬 매니퓰레이터의 경우:

$$J = \begin{bmatrix} J_{v_1} & J_{v_2} & \cdots & J_{v_n} \\ J_{\omega_1} & J_{\omega_2} & \cdots & J_{\omega_n} \end{bmatrix}$$

**회전(revolute)** 조인트 $i$의 경우:
$$J_i = \begin{bmatrix} J_{v_i} \\ J_{\omega_i} \end{bmatrix} = \begin{bmatrix} \hat{z}_{i-1} \times (\mathbf{p}_n - \mathbf{p}_{i-1}) \\ \hat{z}_{i-1} \end{bmatrix}$$

**직동(prismatic)** 조인트 $i$의 경우:
$$J_i = \begin{bmatrix} J_{v_i} \\ J_{\omega_i} \end{bmatrix} = \begin{bmatrix} \hat{z}_{i-1} \\ \mathbf{0} \end{bmatrix}$$

여기서:
- $\hat{z}_{i-1}$은 기저 좌표계(base frame)로 표현된 조인트 축(좌표계 $\{i-1\}$의 z축)
- $\mathbf{p}_n$은 기저 좌표계에서 끝단 작용기의 위치
- $\mathbf{p}_{i-1}$은 기저 좌표계에서 좌표계 $\{i-1\}$의 원점

```python
def compute_geometric_jacobian(joint_types, z_axes, origins, p_end):
    """6xN 기하 야코비안을 계산한다.

    Why 'geometric'? Because this Jacobian uses the geometric relationship
    between joint axes and end-effector position. It directly gives
    linear + angular velocity of the end-effector in the base frame.

    This is the most commonly used Jacobian in robotics.

    Parameters:
        joint_types: list of 'revolute' or 'prismatic'
        z_axes: list of joint axis directions in base frame (z_{i-1})
        origins: list of joint origin positions in base frame (p_{i-1})
        p_end: end-effector position in base frame (p_n)

    Returns:
        J: 6xN Jacobian matrix
    """
    n = len(joint_types)
    J = np.zeros((6, n))

    for i in range(n):
        z = z_axes[i]    # joint axis direction
        p = origins[i]   # joint origin position

        if joint_types[i] == 'revolute':
            # Linear: z x (p_end - p_joint)
            J[:3, i] = np.cross(z, p_end - p)
            # Angular: z
            J[3:, i] = z
        else:  # prismatic
            # Linear: z (translation along joint axis)
            J[:3, i] = z
            # Angular: 0 (translation doesn't create rotation)
            J[3:, i] = 0

    return J
```

### DH 파라미터 / 순기구학으로부터 야코비안 계산

실제로는 순기구학 변환으로부터 야코비안을 계산한다.

```python
class RobotJacobian:
    """DH 파라미터와 순기구학으로부터 기하 야코비안을 계산한다.

    Why combine with FK? Because the Jacobian depends on the current
    configuration — we need FK to know where each joint axis is
    in the base frame. This class reuses the FK computation.
    """
    def __init__(self, dh_params):
        self.dh_params = dh_params
        self.n = len(dh_params)

    def dh_transform(self, theta, d, a, alpha):
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([
            [ct, -st*ca,  st*sa, a*ct],
            [st,  ct*ca, -ct*sa, a*st],
            [ 0,     sa,     ca,    d],
            [ 0,      0,      0,    1]
        ])

    def fk(self, q):
        """모든 중간 변환을 반환하는 순기구학."""
        T = np.eye(4)
        transforms = [T.copy()]  # T_00 = I (base frame)

        for i in range(self.n):
            p = self.dh_params[i]
            if p['type'] == 'revolute':
                theta = p['theta'] + q[i]
                d = p['d']
            else:
                theta = p['theta']
                d = p['d'] + q[i]

            Ti = self.dh_transform(theta, d, p['a'], p['alpha'])
            T = T @ Ti
            transforms.append(T.copy())

        return T, transforms

    def jacobian(self, q):
        """형태 구성 q에서 6xN 기하 야코비안을 계산한다.

        The Jacobian changes with configuration — it must be recomputed
        at each time step in a control loop. For a 6-DOF robot at 1 kHz
        control rate, that's 1000 Jacobian computations per second.
        """
        T_end, transforms = self.fk(q)
        p_end = T_end[:3, 3]  # end-effector position

        J = np.zeros((6, self.n))

        for i in range(self.n):
            T_i = transforms[i]  # T_{0,i} — frame i-1 in base
            z_i = T_i[:3, 2]     # z-axis of frame i-1
            p_i = T_i[:3, 3]     # origin of frame i-1

            if self.dh_params[i]['type'] == 'revolute':
                J[:3, i] = np.cross(z_i, p_end - p_i)
                J[3:, i] = z_i
            else:
                J[:3, i] = z_i
                J[3:, i] = 0

        return J

    def end_effector_velocity(self, q, q_dot):
        """조인트 속도로부터 끝단 작용기 속도를 계산한다."""
        J = self.jacobian(q)
        v_ee = J @ q_dot
        return v_ee[:3], v_ee[3:]  # (linear, angular)


# 예제: 2링크 평면 로봇 야코비안
planar_2link_dh = [
    {'theta': 0, 'd': 0, 'a': 1.0, 'alpha': 0, 'type': 'revolute'},
    {'theta': 0, 'd': 0, 'a': 0.8, 'alpha': 0, 'type': 'revolute'},
]

robot = RobotJacobian(planar_2link_dh)

q = np.radians([30, 45])
J = robot.jacobian(q)
print("Jacobian at q=[30, 45] deg:")
print(np.round(J, 4))

# 검증: 조인트 1이 1 rad/s일 때 선속도와 각속도 모두 발생해야 함
q_dot = np.array([1.0, 0.0])
v_lin, v_ang = robot.end_effector_velocity(q, q_dot)
print(f"\nJoint 1 only (1 rad/s):")
print(f"  Linear velocity:  {np.round(v_lin, 4)} m/s")
print(f"  Angular velocity: {np.round(v_ang, 4)} rad/s")
```

### 2링크 평면 로봇의 해석적 야코비안

2링크 평면 로봇의 경우, 순기구학 방정식을 미분하여 야코비안을 해석적으로 유도할 수 있다.

$$x = l_1 \cos\theta_1 + l_2 \cos(\theta_1 + \theta_2)$$
$$y = l_1 \sin\theta_1 + l_2 \sin(\theta_1 + \theta_2)$$

야코비안(2D이므로 위치 부분만):

$$J = \begin{bmatrix} \frac{\partial x}{\partial \theta_1} & \frac{\partial x}{\partial \theta_2} \\ \frac{\partial y}{\partial \theta_1} & \frac{\partial y}{\partial \theta_2} \end{bmatrix} = \begin{bmatrix} -l_1 s_1 - l_2 s_{12} & -l_2 s_{12} \\ l_1 c_1 + l_2 c_{12} & l_2 c_{12} \end{bmatrix}$$

여기서 $s_1 = \sin\theta_1$, $c_1 = \cos\theta_1$, $s_{12} = \sin(\theta_1 + \theta_2)$, $c_{12} = \cos(\theta_1 + \theta_2)$.

```python
def jacobian_2link_analytical(theta1, theta2, l1, l2):
    """2링크 평면 로봇의 해석적 야코비안 (2x2 위치 야코비안).

    Why derive analytically? For simple robots, the analytical form
    reveals the structure directly. For instance, we can immediately see
    that det(J) = l1*l2*sin(theta2), telling us singularities occur
    at theta2 = 0 or pi — the arm is fully extended or folded.
    """
    s1 = np.sin(theta1)
    c1 = np.cos(theta1)
    s12 = np.sin(theta1 + theta2)
    c12 = np.cos(theta1 + theta2)

    J = np.array([
        [-l1*s1 - l2*s12, -l2*s12],
        [ l1*c1 + l2*c12,  l2*c12]
    ])
    return J

# 수치 야코비안과 교차 검증
def numerical_jacobian(fk_func, q, delta=1e-7):
    """유한 차분법(finite differences)을 이용한 수치 야코비안.

    Why also compute numerically? Cross-checking analytical against
    numerical Jacobian catches derivation errors. This is a standard
    verification technique in robotics.
    """
    x0 = fk_func(q)
    n = len(q)
    m = len(x0)
    J = np.zeros((m, n))

    for i in range(n):
        q_plus = q.copy()
        q_plus[i] += delta
        q_minus = q.copy()
        q_minus[i] -= delta
        J[:, i] = (fk_func(q_plus) - fk_func(q_minus)) / (2 * delta)

    return J

# 수치 야코비안을 위한 순기구학 함수
l1, l2 = 1.0, 0.8
def fk_2link(q):
    return np.array([
        l1 * np.cos(q[0]) + l2 * np.cos(q[0] + q[1]),
        l1 * np.sin(q[0]) + l2 * np.sin(q[0] + q[1])
    ])

q = np.radians([30, 45])
J_analytical = jacobian_2link_analytical(q[0], q[1], l1, l2)
J_numerical = numerical_jacobian(fk_2link, q)

print("Analytical Jacobian:")
print(np.round(J_analytical, 6))
print("\nNumerical Jacobian:")
print(np.round(J_numerical, 6))
print(f"\nMatch: {np.allclose(J_analytical, J_numerical, atol=1e-5)}")
```

---

## 기하 야코비안 vs 해석적 야코비안

두 종류의 야코비안이 존재하며, 그 차이는 중요하다.

### 기하 야코비안 $J_g$

조인트 속도를 **공간 트위스트(spatial twist)**(선속도 + 각속도)로 매핑한다.

$$\begin{bmatrix} \mathbf{v} \\ \boldsymbol{\omega} \end{bmatrix} = J_g(\mathbf{q}) \, \dot{\mathbf{q}}$$

각속도 $\boldsymbol{\omega}$는 물리적으로 잘 정의된 양으로, 순간 회전 축과 속도를 나타낸다.

### 해석적 야코비안 $J_a$

조인트 속도를 **포즈 매개변수화(pose parameterization)의 시간 미분**(예: 위치 + 오일러 각도)으로 매핑한다.

$$\dot{\mathbf{x}} = J_a(\mathbf{q}) \, \dot{\mathbf{q}}$$

여기서 $\mathbf{x} = (x, y, z, \phi, \theta, \psi)^T$이며 오일러 각도를 포함한다.

### 관계

선속도 부분은 동일하다. 각속도 부분이 다르다.

$$\boldsymbol{\omega} = B(\boldsymbol{\phi}) \, \dot{\boldsymbol{\phi}}$$

여기서 $B$는 오일러 각도 규약과 현재 방향에 따라 달라지는 행렬이다. 따라서:

$$J_a = \begin{bmatrix} I & 0 \\ 0 & B^{-1} \end{bmatrix} J_g$$

```python
def euler_zyx_rate_matrix(phi, theta, psi):
    """오일러 각도 변화율을 각속도로 변환하는 행렬 B.

    omega = B * [phi_dot, theta_dot, psi_dot]^T

    Why does this matrix exist? Because Euler angle rates are NOT
    the same as angular velocity components. The angular velocity
    is a vector in 3D; Euler angle rates are derivatives of three
    successive rotation angles about different (moving) axes.

    WARNING: B is singular when theta = +/- 90 deg (gimbal lock!).
    This is another reason to prefer the geometric Jacobian.
    """
    cp = np.cos(phi)
    sp = np.sin(phi)
    ct = np.cos(theta)
    st = np.sin(theta)

    B = np.array([
        [1, 0,     -st    ],
        [0, cp,  sp * ct  ],
        [0, -sp, cp * ct  ]
    ])
    return B

# 시연: 오일러 각도 변화율 vs 각속도
phi, theta, psi = np.radians([10, 30, 20])
B = euler_zyx_rate_matrix(phi, theta, psi)
print("Euler rate matrix B:")
print(np.round(B, 4))
print(f"det(B) = {np.linalg.det(B):.4f}")  # Non-zero: no gimbal lock here

# 짐벌 락(gimbal lock) 상태 (theta = 90 deg):
B_singular = euler_zyx_rate_matrix(0, np.pi/2, 0)
print(f"\nAt theta=90 deg, det(B) = {np.linalg.det(B_singular):.6f}")  # ~0
```

### 어떤 것을 사용할까?

| 야코비안 | 사용 사례 |
|----------|----------|
| 기하 야코비안 $J_g$ | 속도/힘 제어, 조작 가능도 분석, 특이점 분석 |
| 해석적 야코비안 $J_a$ | 오일러 각도 오차를 사용하는 작업 공간 궤적 추종, 최적화 |

**경험 법칙**: 오일러 각도 변화율이 명시적으로 필요한 경우가 아니면 기하 야코비안을 사용한다.

---

## 특이점 분석

### SVD를 통한 특이점 재조명

야코비안의 **특이값 분해(Singular Value Decomposition, SVD)**는 로봇 거동에 대한 가장 깊은 통찰을 제공한다.

$$J = U \Sigma V^T$$

여기서:
- $U \in \mathbb{R}^{m \times m}$: 열 벡터는 **작업 공간의 특이 방향(singular direction)**
- $\Sigma \in \mathbb{R}^{m \times n}$: 특이값 $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$의 대각 행렬
- $V \in \mathbb{R}^{n \times n}$: 열 벡터는 **조인트 공간의 특이 방향**

**물리적 해석**:
- $\sigma_i$: $i$번째 특이 방향에서의 "이득(gain)" (조인트 움직임이 작업 공간 움직임을 얼마나 만들어내는가)
- 특이점은 어떤 $i$에 대해 $\sigma_i = 0$일 때 발생한다
- 특이점 근방: $\sigma_i \approx 0$ (작은 작업 공간 움직임을 위해 큰 조인트 움직임이 필요)

```python
def singularity_analysis(J, verbose=True):
    """SVD를 통한 완전한 특이점 분석.

    Why SVD? Because it decomposes the Jacobian into independent
    'channels' of motion. Each singular value is the gain of one
    channel. When a singular value goes to zero, that channel of
    end-effector motion shuts down — the robot can't move in that
    direction no matter how fast the joints spin.
    """
    U, sigma, Vt = np.linalg.svd(J)
    rank = np.sum(sigma > 1e-10)

    # 조건수: 최대 특이값과 최소 특이값의 비율
    # 매핑이 얼마나 '왜곡'되어 있는지를 측정
    if sigma[-1] > 1e-10:
        condition = sigma[0] / sigma[-1]
    else:
        condition = np.inf

    # 조작 가능도: 특이값의 곱 (Yoshikawa 척도)
    manipulability = np.prod(sigma[:rank])

    if verbose:
        print(f"Singular values: {np.round(sigma, 6)}")
        print(f"Rank: {rank} / {min(J.shape)}")
        print(f"Condition number: {condition:.1f}")
        print(f"Manipulability (Yoshikawa): {manipulability:.6f}")

        if rank < min(J.shape):
            print("\n*** SINGULAR CONFIGURATION ***")
            # 손실된 방향 식별
            for i in range(rank, len(sigma)):
                lost_dir = U[:, i]
                print(f"Lost task-space direction: {np.round(lost_dir, 4)}")

    return sigma, rank, condition, manipulability


# 예제: 다양한 형태 구성에서의 2링크 로봇
l1, l2 = 1.0, 0.8

# 일반 형태 구성
print("=== q = [30, 45] deg (regular) ===")
J_regular = jacobian_2link_analytical(np.radians(30), np.radians(45), l1, l2)
singularity_analysis(J_regular)

# 특이점 근방
print("\n=== q = [30, 5] deg (near-singular) ===")
J_near = jacobian_2link_analytical(np.radians(30), np.radians(5), l1, l2)
singularity_analysis(J_near)

# 특이점 (완전히 펼쳐진 상태)
print("\n=== q = [30, 0] deg (SINGULAR) ===")
J_singular = jacobian_2link_analytical(np.radians(30), np.radians(0), l1, l2)
singularity_analysis(J_singular)
```

### 정방 야코비안에 대한 행렬식 분석

비여유(non-redundant) 로봇(정방 야코비안)의 경우, 행렬식이 빠른 특이점 테스트를 제공한다.

2링크 평면 로봇의 경우:

$$\det(J) = l_1 l_2 \sin\theta_2$$

이것은 $\theta_2 = 0$ (팔이 펴진 상태) 또는 $\theta_2 = \pi$ (팔이 접힌 상태)일 때 0이 된다.

```python
def plot_manipulability_landscape(l1, l2, n_points=100):
    """조인트 공간 전체에 걸쳐 조작 가능도를 계산한다.

    Why map the entire landscape? Because it reveals which regions of
    joint space are well-conditioned (good for control) and which are
    dangerous (near singularity). Motion planners use this information
    to prefer paths through well-conditioned regions.
    """
    theta1_range = np.linspace(-np.pi, np.pi, n_points)
    theta2_range = np.linspace(-np.pi, np.pi, n_points)

    manipulability = np.zeros((n_points, n_points))

    for i, t1 in enumerate(theta1_range):
        for j, t2 in enumerate(theta2_range):
            J = jacobian_2link_analytical(t1, t2, l1, l2)
            manipulability[j, i] = abs(np.linalg.det(J))

    # 통계 보고
    print(f"Manipulability range: [{manipulability.min():.4f}, "
          f"{manipulability.max():.4f}]")
    print(f"Max manipulability at theta2 = +/- 90 deg "
          f"(= {l1*l2:.4f})")
    print(f"Zero manipulability at theta2 = 0 or 180 deg")

    return manipulability, theta1_range, theta2_range

w, _, _ = plot_manipulability_landscape(1.0, 0.8)
```

---

## 힘/토크 매핑 (정역학 쌍대성)

### 가상 일의 원리

로보틱스에서 가장 우아한 결과 중 하나: 야코비안 전치(transpose)가 끝단 작용기 힘을 조인트 토크로 매핑한다.

끝단 작용기가 환경에 렌치(wrench, 힘 + 토크) $\mathbf{F} = (\mathbf{f}, \boldsymbol{\tau}_{ee})^T$를 가하면, 이에 해당하는 조인트 토크는:

$$\boldsymbol{\tau} = J^T(\mathbf{q}) \, \mathbf{F}$$

이것이 **정역학 쌍대성(statics duality)**이다. 속도를 앞방향으로 매핑하는 야코비안이 힘은 역방향으로 매핑한다.

**가상 일을 통한 유도**: 가상 변위 $\delta \mathbf{q}$와 $\delta \mathbf{x} = J \delta \mathbf{q}$에 대해:

$$\delta W = \mathbf{F}^T \delta \mathbf{x} = \mathbf{F}^T J \delta \mathbf{q} = \boldsymbol{\tau}^T \delta \mathbf{q}$$

이것이 모든 $\delta \mathbf{q}$에 대해 성립하므로: $\boldsymbol{\tau} = J^T \mathbf{F}$.

```python
def compute_joint_torques(J, F_ee):
    """끝단 작용기 렌치를 조인트 토크로 매핑한다.

    Why J transpose (not J inverse)? Because the force mapping goes
    in the opposite direction from velocity. Velocity: joints → EE
    uses J. Force: EE → joints uses J^T. This is the statics duality,
    derived from energy conservation (virtual work).

    This is fundamental to force control: if you want the end-effector
    to exert a specific force, compute the required joint torques using J^T.
    """
    return J.T @ F_ee

# 예제: 2링크 팔이 10N의 힘으로 아래를 누르는 경우
l1, l2 = 1.0, 0.8
q = np.radians([45, 30])

# 2D 위치 야코비안
J = jacobian_2link_analytical(q[0], q[1], l1, l2)

# 끝단 작용기 힘: 10N 하향
F_ee = np.array([0, -10])  # Fx=0, Fy=-10 N

# 필요한 조인트 토크
tau = compute_joint_torques(J, F_ee)
print(f"Configuration: q = [{np.degrees(q[0]):.0f}, {np.degrees(q[1]):.0f}] deg")
print(f"End-effector force: F = {F_ee} N")
print(f"Required joint torques: tau = {np.round(tau, 3)} N*m")

# 물리적 검증: 이 형태 구성에서 모멘트 암(moment arm)은 얼마인가?
# 조인트 1은 전체 힘을 조인트 1에서 끝단 작용기까지의
# 수평 거리(모멘트 암)에서 지지해야 한다
x_ee, y_ee = l1*np.cos(q[0]) + l2*np.cos(q[0]+q[1]), \
             l1*np.sin(q[0]) + l2*np.sin(q[0]+q[1])
print(f"\nEnd-effector position: ({x_ee:.4f}, {y_ee:.4f})")
print(f"Moment arm for joint 1: x_ee = {x_ee:.4f} m")
print(f"Expected tau_1: {x_ee * 10:.4f} N*m")  # Should match tau[0]
```

### 힘 타원체(Force Ellipsoid)

힘 타원체는 제한된 조인트 토크($\|\boldsymbol{\tau}\| \leq 1$)로 달성 가능한 끝단 작용기 힘을 보여준다.

$$\mathbf{F}^T (J J^T)^{-1} \mathbf{F} \leq 1$$

힘 타원체의 축은 SVD의 $U$ 열 벡터 방향과 일치하고, 그 길이는 특이값 $\sigma_i$이다.

> **핵심 통찰**: 속도 타원체와 힘 타원체는 **서로 역수** 관계이다. 로봇이 빠르게 움직일 수 있는 방향(큰 $\sigma$)은 힘을 거의 발휘할 수 없는 방향이고, 그 반대도 마찬가지이다. 이것은 지렛대의 로봇 버전이다. 긴 지렛대 = 빠른 끝 속도이지만 작은 힘, 짧은 지렛대 = 느린 끝 속도이지만 큰 힘.

---

## 조작 가능도 타원체

### 정의

**조작 가능도 타원체(manipulability ellipsoid)**는 단위 조인트 속도($\|\dot{\mathbf{q}}\| \leq 1$)로 달성 가능한 끝단 작용기 속도를 시각화한다.

$$\mathbf{v}^T (J J^T)^{-1} \mathbf{v} \leq 1$$

타원체의 주축(principal axes)은 $J$의 왼쪽 특이 벡터(left singular vectors)이고, 반축(semi-axis) 길이는 특이값이다.

```python
def compute_manipulability_ellipse(J):
    """2D 조작 가능도 타원 파라미터를 계산한다.

    Why an ellipse (not a circle)? Because the Jacobian maps a unit
    sphere in joint space to an ellipsoid in task space. The shape of
    this ellipsoid tells us:
    - Long axis: direction of easy/fast motion
    - Short axis: direction of difficult/slow motion
    - Ratio (condition number): isotropy of the mapping
    - Area (proportional to manipulability): overall dexterity
    """
    U, sigma, Vt = np.linalg.svd(J)

    # 타원 파라미터
    semi_axes = sigma  # lengths of semi-axes
    directions = U[:, :len(sigma)]  # directions of semi-axes
    angle = np.arctan2(U[1, 0], U[0, 0])  # angle of major axis

    return semi_axes, directions, angle

# 다양한 형태 구성에서 시각화
l1, l2 = 1.0, 0.8
configs = {
    'Regular (30, 90)': np.radians([30, 90]),
    'Near-singular (30, 10)': np.radians([30, 10]),
    'Symmetric (45, -90)': np.radians([45, -90]),
}

for name, q in configs.items():
    J = jacobian_2link_analytical(q[0], q[1], l1, l2)
    semi_axes, directions, angle = compute_manipulability_ellipse(J)
    w = np.prod(semi_axes)
    cond = semi_axes[0] / max(semi_axes[1], 1e-10)

    print(f"\n{name}:")
    print(f"  Semi-axes: {np.round(semi_axes, 4)}")
    print(f"  Manipulability: {w:.4f}")
    print(f"  Condition number: {cond:.1f}")
    print(f"  Ellipse angle: {np.degrees(angle):.1f} deg")
```

### 민첩성 척도(Dexterity Measures)

여러 스칼라 척도들이 조작 가능도를 정량화한다.

| 척도 | 수식 | 해석 |
|---------|---------|----------------|
| **요시카와 조작 가능도(Yoshikawa manipulability)** | $w = \sqrt{\det(JJ^T)} = \prod \sigma_i$ | 조작 가능도 타원체의 부피 (특이점에서 0) |
| **조건수(Condition number)** | $\kappa = \sigma_{max} / \sigma_{min}$ | 등방성 ($\kappa = 1$이 이상적; 특이점에서 $\kappa = \infty$) |
| **최소 특이값** | $\sigma_{min}$ | 최악의 경우 속도 이득 |
| **등방성 지수(Isotropy index)** | $1/\kappa$ | 정규화된 등방성 ($1$ = 등방성, $0$ = 특이점) |

```python
def dexterity_measures(J):
    """모든 표준 민첩성 척도를 계산한다.

    Why multiple measures? Because no single number captures all
    aspects of dexterity. Yoshikawa's measure can be high even if
    the ellipsoid is very elongated (good in one direction, bad in
    another). The condition number captures isotropy but not overall
    magnitude. Use them together for a complete picture.
    """
    U, sigma, Vt = np.linalg.svd(J)

    measures = {
        'yoshikawa': np.prod(sigma),
        'condition_number': sigma[0] / max(sigma[-1], 1e-15),
        'min_singular_value': sigma[-1],
        'isotropy': sigma[-1] / max(sigma[0], 1e-15),
    }

    return measures

# 등방성에 최적인 형태 구성 (2링크 평면 로봇)
# 조건수가 최소화(등방성 최대화)되는 것은 조작 가능도 타원이
# 원에 가장 가까울 때이다.
# 2링크 로봇에서는 theta2 = +/- 90도일 때 발생하며,
# 링크 길이 비율에도 영향을 받는다.

best_config = None
best_isotropy = 0

for t1 in np.linspace(-np.pi, np.pi, 100):
    for t2 in np.linspace(-np.pi, np.pi, 100):
        J = jacobian_2link_analytical(t1, t2, l1, l2)
        m = dexterity_measures(J)
        if m['isotropy'] > best_isotropy:
            best_isotropy = m['isotropy']
            best_config = (t1, t2)

print(f"\nBest isotropy: {best_isotropy:.4f} at "
      f"q = ({np.degrees(best_config[0]):.1f}, {np.degrees(best_config[1]):.1f}) deg")
```

---

## 6자유도 로봇의 야코비안

완전한 6자유도 매니퓰레이터의 야코비안은 6x6이다. 다음은 완전한 구현이다.

```python
def jacobian_6dof(dh_params, q):
    """6자유도 직렬 매니퓰레이터의 완전한 6x6 야코비안.

    This function extracts joint axes and origins from FK,
    then applies the geometric Jacobian formula for each joint.

    For real-time control at 1 kHz, this computation takes
    roughly 10-50 microseconds on modern hardware — fast enough.
    """
    n = len(dh_params)
    assert n == 6, "This function is for 6-DOF robots"

    # 모든 좌표계 변환을 얻기 위한 순기구학 계산
    T = np.eye(4)
    transforms = [T.copy()]

    for i in range(n):
        p = dh_params[i]
        ct = np.cos(p['theta'] + (q[i] if p['type'] == 'revolute' else 0))
        st = np.sin(p['theta'] + (q[i] if p['type'] == 'revolute' else 0))
        d = p['d'] + (q[i] if p['type'] == 'prismatic' else 0)
        ca = np.cos(p['alpha'])
        sa = np.sin(p['alpha'])
        a = p['a']

        Ti = np.array([
            [ct, -st*ca,  st*sa, a*ct],
            [st,  ct*ca, -ct*sa, a*st],
            [ 0,     sa,     ca,    d],
            [ 0,      0,      0,    1]
        ])
        T = T @ Ti
        transforms.append(T.copy())

    p_end = transforms[-1][:3, 3]
    J = np.zeros((6, n))

    for i in range(n):
        z_i = transforms[i][:3, 2]
        p_i = transforms[i][:3, 3]

        if dh_params[i]['type'] == 'revolute':
            J[:3, i] = np.cross(z_i, p_end - p_i)
            J[3:, i] = z_i
        else:
            J[:3, i] = z_i
            J[3:, i] = 0

    return J

# PUMA 560 유사 DH 파라미터
puma_dh = [
    {'theta': 0, 'd': 0,    'a': 0,    'alpha': -np.pi/2, 'type': 'revolute'},
    {'theta': 0, 'd': 0,    'a': 0.4318, 'alpha': 0,       'type': 'revolute'},
    {'theta': 0, 'd': 0.15, 'a': 0.0203, 'alpha': -np.pi/2, 'type': 'revolute'},
    {'theta': 0, 'd': 0.4318, 'a': 0, 'alpha': np.pi/2,    'type': 'revolute'},
    {'theta': 0, 'd': 0,    'a': 0,    'alpha': -np.pi/2, 'type': 'revolute'},
    {'theta': 0, 'd': 0,    'a': 0,    'alpha': 0,        'type': 'revolute'},
]

q = np.radians([30, -45, 60, 0, 30, 0])
J_puma = jacobian_6dof(puma_dh, q)

print("PUMA 560 Jacobian (6x6):")
print(np.round(J_puma, 4))

print("\nSingularity analysis:")
sigma, rank, cond, manip = singularity_analysis(J_puma)
```

---

## 요약

- **야코비안** $J(\mathbf{q})$는 기본 속도 매핑이다: $\dot{\mathbf{x}} = J \dot{\mathbf{q}}$
- **회전 조인트**의 경우 각 야코비안 열은 $\hat{z} \times (\mathbf{p}_{ee} - \mathbf{p}_{joint})$ (선속도)와 $\hat{z}$ (각속도)를 포함한다
- **직동 조인트**의 경우 열은 $\hat{z}$ (선속도)와 $\mathbf{0}$ (각속도)이다
- **기하 야코비안**은 공간 트위스트를 제공하고, **해석적 야코비안**은 오일러 각도 변화율을 제공한다
- **특이점** ($\det(J) = 0$)은 자유도 손실을 나타낸다. 로봇이 특정 방향으로 이동할 수 없는 상태이다
- **정역학 쌍대성**: $\boldsymbol{\tau} = J^T \mathbf{F}$는 끝단 작용기 힘을 조인트 토크로 매핑한다
- **조작 가능도 타원체**는 각 형태 구성에서 속도/힘 능력을 시각화한다
- 야코비안의 **SVD**는 로봇 거동에 대한 가장 완전한 분석을 제공한다

---

## 연습 문제

### 연습 1: 야코비안 유도

링크 길이 $l_1$, $l_2$, $l_3$인 3링크 평면 로봇에 대해:
1. 2x3 위치 야코비안을 해석적으로 유도하라
2. $l_1 = l_2 = l_3 = 0.5$ m일 때 $q = (30°, 45°, -60°)$에서 야코비안을 계산하라
3. 유한 차분법으로 결과를 검증하라
4. 이 형태 구성에서 영공간(null space)은 무엇인가? 어떤 물리적 움직임을 나타내는가?

### 연습 2: 특이점 분석

2링크 평면 로봇 ($l_1 = 1.0$, $l_2 = 0.8$)에 대해:
1. $\det(J) = 0$으로부터 모든 특이 형태 구성을 해석적으로 구하라
2. $q = (45°, 0°)$에서 SVD를 계산하고 손실된 운동 방향을 파악하라
3. 유사역행렬(pseudo-inverse)을 사용하여 손실된 방향으로 끝단 작용기를 이동시키려 할 때 조인트 속도에 어떤 일이 발생하는가?
4. DLS(Damped Least Squares, 감쇠 최소 제곱)를 다양한 $\lambda$ 값으로 반복하고 비교하라

### 연습 3: 힘 매핑

$q = (0°, 90°)$의 2링크 평면 팔 ($l_1 = l_2 = 0.5$ m)이 끝단 작용기에서 오른쪽(+x 방향)으로 20 N의 힘을 발휘해야 한다.
1. $\tau = J^T F$를 사용하여 필요한 조인트 토크를 계산하라
2. 조인트 1의 최대 토크가 15 N*m라면, 이 힘을 달성할 수 있는가?
3. 이 형태 구성에서 로봇이 +x 방향으로 발휘할 수 있는 최대 힘은 얼마인가 ($|\tau_i| \leq 15$ N*m를 가정)?

### 연습 4: 조작 가능도 타원체

2링크 로봇 ($l_1 = 1.0$, $l_2 = 0.5$)에 대해:
1. $\theta_1 = 0$으로 고정하고 $\theta_2 = 30°$, $60°$, $90°$, $120°$, $150°$에서 조작 가능도 타원을 계산하고 비교하라
2. 어떤 $\theta_2$에서 타원이 가장 원에 가까운가 (최고 등방성)?
3. 최적 등방성을 위한 $\theta_2$가 $\theta_1$에 의존하는가? 그 이유는?

### 연습 5: 수치 야코비안 검증

PUMA 560 유사 로봇에 대한 일반적인 수치 야코비안(유한 차분법 사용)을 구현하라:
1. 5개의 무작위 형태 구성에서 기하 야코비안과 비교하라
2. 어떤 스텝 크기 $\delta$가 최고의 정확도를 주는가? ($10^{-4}$에서 $10^{-10}$ 시도)
3. 매우 작은 $\delta$에서 정확도가 저하되는 이유는? (힌트: 부동소수점 연산)

---

[← 이전: 역기구학](04_Inverse_Kinematics.md) | [다음: 로봇 동역학 →](06_Robot_Dynamics.md)
