# 로봇 동역학(Robot Dynamics)

[← 이전: 속도 기구학](05_Velocity_Kinematics.md) | [다음: 모션 플래닝 →](07_Motion_Planning.md)

## 학습 목표

1. 동역학이 로봇 제어에서 왜 중요한지 설명한다 — 기구학을 넘어서는 핵심 단계
2. 운동 에너지, 위치 에너지, 라그랑지안(Lagrangian)을 이용하여 오일러-라그랑주(Euler-Lagrange) 공식으로 운동 방정식을 유도한다
3. 표준 매니퓰레이터 동역학 방정식의 각 항을 파악하고 해석한다: $M(\mathbf{q})\ddot{\mathbf{q}} + C(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q}) = \boldsymbol{\tau}$
4. 관성 행렬(inertia matrix)의 성질을 서술하고 코리올리(Coriolis) 항과 원심력 항의 물리적 기원을 설명한다
5. 중력 보상(gravity compensation)과 계산 토크 제어(computed torque control)를 구현한다
6. 오일러-라그랑주 공식과 뉴턴-오일러(Newton-Euler) 공식을 비교하고 각각의 적합한 사용 시점을 이해한다

---

## 왜 중요한가

기구학은 로봇이 *어디로* 갈 수 있는지를 알려준다. 동역학은 그곳에 도달하기 위해 *얼마나 세게 밀어야* 하는지를 알려준다. 동역학 없이는 공간을 통과하는 아름다운 경로를 계획할 수 있지만, 그것을 따라가기 위해 어떤 모터 토크가 필요한지는 전혀 알 수 없다. 그 결과는 목표를 지나치고, 진동하고, 자체 무게로 무너지는 로봇이 될 것이다.

동역학은 기하학과 제어 사이의 다리이다. 로봇 팔이 조인트 1을 빠르게 흔들면 원심력 효과가 조인트 2를 가속시킨다(결합 효과). 팔이 수평으로 펼쳐지면 중력이 모터가 끊임없이 저항해야 하는 토크를 만들어낸다. 팔이 빠르게 움직이면 속도에 의존하는 힘(코리올리 효과)이 예기치 않은 힘을 만들어 끝단 작용기를 경로에서 벗어나게 한다. 이러한 효과들을 이해하고 정확하게 계산하는 것이 고성능 모션 제어의 핵심이다.

> **비유**: 동역학은 단순히 "어디로 조향할지"가 아니라 "가속 페달을 얼마나 밟아야 하는지"를 이해하는 것과 같다. 기구학만 사용하는 제어기는 지도는 있지만 자동차의 무게, 도로 경사, 바람 저항 감각이 없는 운전자와 같다. 방향은 알지만, 스로틀을 얼마나 적용해야 할지는 전혀 모른다.

---

## 기구학에서 동역학으로

### 기구학이 알려줄 수 없는 것

기구학이 제공하는 것:
- **순기구학(FK)**: 조인트 각도 $\to$ 끝단 작용기 포즈
- **역기구학(IK)**: 원하는 포즈 $\to$ 조인트 각도
- **야코비안**: 조인트 속도 $\to$ 끝단 작용기 속도

하지만 다음에는 답할 수 없다: **원하는 조인트 가속도를 달성하려면 모터에 어떤 토크를 가해야 하는가?**

이것은 동역학, 즉 뉴턴의 법칙에 의해 지배되는 힘/토크와 운동 사이의 관계가 필요하다.

### 매니퓰레이터 동역학 방정식

매니퓰레이터 운동 방정식의 표준 형태:

$$M(\mathbf{q}) \ddot{\mathbf{q}} + C(\mathbf{q}, \dot{\mathbf{q}}) \dot{\mathbf{q}} + \mathbf{g}(\mathbf{q}) = \boldsymbol{\tau}$$

여기서:
| 항 | 기호 | 차원 | 물리적 의미 |
|------|--------|-----------|------------------|
| 관성 행렬 | $M(\mathbf{q})$ | $n \times n$ | 질량과 회전 관성 (형태 구성 의존) |
| 코리올리/원심력 | $C(\mathbf{q}, \dot{\mathbf{q}})$ | $n \times n$ | 속도 의존 힘 (조인트 간 결합) |
| 중력 | $\mathbf{g}(\mathbf{q})$ | $n \times 1$ | 각 조인트의 중력 토크 |
| 조인트 토크 | $\boldsymbol{\tau}$ | $n \times 1$ | 가해진 모터 토크 (제어 입력) |

---

## 오일러-라그랑주 공식

### 라그랑지안

라그랑지안은 운동 에너지와 위치 에너지의 차로 정의된다.

$$\mathcal{L}(\mathbf{q}, \dot{\mathbf{q}}) = K(\mathbf{q}, \dot{\mathbf{q}}) - P(\mathbf{q})$$

여기서 $K$는 매니퓰레이터의 총 운동 에너지이고, $P$는 총 위치 에너지이다.

### 오일러-라그랑주 방정식

$i$번째 조인트의 운동 방정식:

$$\frac{d}{dt}\frac{\partial \mathcal{L}}{\partial \dot{q}_i} - \frac{\partial \mathcal{L}}{\partial q_i} = \tau_i, \quad i = 1, \ldots, n$$

이것은 최소 작용의 원리(principle of least action)로부터 유도된다. 로봇의 궤적은 라그랑지안의 시간 적분을 최소화한다. 이 프레임워크는 구속력(constraint force)을 계산할 필요 없이 어떤 좌표계(조인트 각도, 작업 공간 좌표 등)에서도 작동하기 때문에 매우 강력하다.

```python
import numpy as np
from numpy import sin, cos

# 오일러-라그랑주 절차를 설명하기 위한
# 단일 진자(1자유도)의 기호 유도

def single_pendulum_dynamics(theta, theta_dot, m, l, g=9.81):
    """단순 진자의 동역학 (1자유도 로봇 팔).

    Why start with a pendulum? It's the simplest possible robot arm
    (1 link, 1 joint). The derivation illustrates the complete
    Euler-Lagrange procedure without algebraic complexity.

    Step-by-step:
    1. Kinetic energy: K = (1/2) * I * theta_dot^2
       where I = m*l^2 (point mass at end of massless rod)
    2. Potential energy: P = m*g*l*(1 - cos(theta))
       (zero at the bottom, theta=0 = hanging down)
    3. Lagrangian: L = K - P
    4. Euler-Lagrange: d/dt(dL/d_theta_dot) - dL/d_theta = tau

    Result: m*l^2 * theta_ddot + m*g*l*sin(theta) = tau
    """
    I = m * l**2        # moment of inertia
    M = I               # 1x1 "inertia matrix"
    g_term = m * g * l * sin(theta)  # gravity torque

    return M, g_term
    # The equation of motion: M * theta_ddot + g_term = tau
    # Or: theta_ddot = (tau - g_term) / M

# 예제: 진자 파라미터
m, l = 2.0, 0.5  # 2 kg mass, 0.5 m rod
theta = np.radians(45)
theta_dot = 0.0

M, g_term = single_pendulum_dynamics(theta, theta_dot, m, l)
print(f"Inertia: {M:.4f} kg*m^2")
print(f"Gravity torque at 45 deg: {g_term:.4f} N*m")
print(f"To hold position (tau = g_term): {g_term:.4f} N*m")
```

---

## 매니퓰레이터의 운동 에너지와 위치 에너지

### 운동 에너지

질량 $m_i$, 질량 중심 속도 $\mathbf{v}_{c_i}$, 각속도 $\boldsymbol{\omega}_i$, 관성 텐서(inertia tensor) $I_{c_i}$ (질량 중심에 대해)를 갖는 링크 $i$의 운동 에너지:

$$K_i = \frac{1}{2} m_i \mathbf{v}_{c_i}^T \mathbf{v}_{c_i} + \frac{1}{2} \boldsymbol{\omega}_i^T I_{c_i} \boldsymbol{\omega}_i$$

총 운동 에너지:

$$K = \sum_{i=1}^{n} K_i$$

야코비안을 사용하면 $\mathbf{v}_{c_i}$와 $\boldsymbol{\omega}_i$를 조인트 속도로 표현할 수 있다.

$$\mathbf{v}_{c_i} = J_{v_i}(\mathbf{q}) \dot{\mathbf{q}}, \quad \boldsymbol{\omega}_i = J_{\omega_i}(\mathbf{q}) \dot{\mathbf{q}}$$

여기서 $J_{v_i}$와 $J_{\omega_i}$는 링크 $i$의 질량 중심 선속도와 각속도에 대한 야코비안이다.

총 운동 에너지는 다음이 된다.

$$K = \frac{1}{2} \dot{\mathbf{q}}^T M(\mathbf{q}) \dot{\mathbf{q}}$$

여기서 **관성 행렬**은:

$$M(\mathbf{q}) = \sum_{i=1}^{n} \left[ m_i J_{v_i}^T J_{v_i} + J_{\omega_i}^T I_{c_i} J_{\omega_i} \right]$$

### 위치 에너지

$$P = \sum_{i=1}^{n} m_i \mathbf{g}^T \mathbf{p}_{c_i}(\mathbf{q})$$

여기서 $\mathbf{g} = (0, 0, -g)^T$는 중력 벡터이고, $\mathbf{p}_{c_i}$는 링크 $i$의 질량 중심 위치이다.

```python
def compute_link_energy(m, v_cm, omega, I_cm, p_cm, g_vec):
    """단일 링크의 운동 에너지와 위치 에너지를 계산한다.

    Why separate per-link? Because each link has different mass,
    inertia, and position. The total energy is the sum over all links.
    This modular approach scales to any number of links.
    """
    # 운동 에너지: 병진 + 회전
    K_trans = 0.5 * m * np.dot(v_cm, v_cm)
    K_rot = 0.5 * np.dot(omega, I_cm @ omega)
    K = K_trans + K_rot

    # 위치 에너지 (중력)
    P = -m * np.dot(g_vec, p_cm)  # P = -m*g*h (with g pointing down)

    return K, P
```

---

## 2링크 평면 팔: 완전한 유도

로봇 동역학에서 가장 중요한 실습 예제이다. 모든 항을 명시적으로 유도한다.

### 설정

- 링크 1: 길이 $l_1$, 질량 $m_1$, 조인트 1에서 거리 $l_{c_1}$에 질량 중심, 관성 $I_1$
- 링크 2: 길이 $l_2$, 질량 $m_2$, 조인트 2에서 거리 $l_{c_2}$에 질량 중심, 관성 $I_2$
- 중력 $g$는 $-y$ 방향으로 작용한다

### 질량 중심 위치

$$x_{c_1} = l_{c_1} \cos\theta_1, \quad y_{c_1} = l_{c_1} \sin\theta_1$$

$$x_{c_2} = l_1 \cos\theta_1 + l_{c_2} \cos(\theta_1 + \theta_2)$$
$$y_{c_2} = l_1 \sin\theta_1 + l_{c_2} \sin(\theta_1 + \theta_2)$$

### 질량 중심 속도

$$\dot{x}_{c_1} = -l_{c_1} \sin\theta_1 \, \dot{\theta}_1$$
$$\dot{y}_{c_1} = l_{c_1} \cos\theta_1 \, \dot{\theta}_1$$

$$\dot{x}_{c_2} = -l_1 \sin\theta_1 \, \dot{\theta}_1 - l_{c_2} \sin(\theta_1 + \theta_2)(\dot{\theta}_1 + \dot{\theta}_2)$$
$$\dot{y}_{c_2} = l_1 \cos\theta_1 \, \dot{\theta}_1 + l_{c_2} \cos(\theta_1 + \theta_2)(\dot{\theta}_1 + \dot{\theta}_2)$$

### 운동 에너지

$$K = \frac{1}{2}(m_1 l_{c_1}^2 + I_1)\dot{\theta}_1^2 + \frac{1}{2}m_2\left[l_1^2 \dot{\theta}_1^2 + l_{c_2}^2(\dot{\theta}_1 + \dot{\theta}_2)^2 + 2 l_1 l_{c_2} \cos\theta_2 \, \dot{\theta}_1(\dot{\theta}_1 + \dot{\theta}_2)\right] + \frac{1}{2}I_2(\dot{\theta}_1 + \dot{\theta}_2)^2$$

이것은 다음과 같이 쓸 수 있다.

$$K = \frac{1}{2} \dot{\mathbf{q}}^T M(\mathbf{q}) \dot{\mathbf{q}}$$

### 관성 행렬

$$M(\mathbf{q}) = \begin{bmatrix} M_{11} & M_{12} \\ M_{21} & M_{22} \end{bmatrix}$$

여기서:
$$M_{11} = m_1 l_{c_1}^2 + m_2(l_1^2 + l_{c_2}^2 + 2l_1 l_{c_2}\cos\theta_2) + I_1 + I_2$$
$$M_{12} = M_{21} = m_2(l_{c_2}^2 + l_1 l_{c_2}\cos\theta_2) + I_2$$
$$M_{22} = m_2 l_{c_2}^2 + I_2$$

### 코리올리와 원심력 항

크리스토펠 기호(Christoffel symbols) $c_{ijk} = \frac{1}{2}\left(\frac{\partial M_{kj}}{\partial q_i} + \frac{\partial M_{ki}}{\partial q_j} - \frac{\partial M_{ij}}{\partial q_k}\right)$를 사용하여:

$$C(\mathbf{q}, \dot{\mathbf{q}}) = \begin{bmatrix} -m_2 l_1 l_{c_2} \sin\theta_2 \, \dot{\theta}_2 & -m_2 l_1 l_{c_2} \sin\theta_2 (\dot{\theta}_1 + \dot{\theta}_2) \\ m_2 l_1 l_{c_2} \sin\theta_2 \, \dot{\theta}_1 & 0 \end{bmatrix}$$

$h = m_2 l_1 l_{c_2} \sin\theta_2$라 하면:

$$C \dot{\mathbf{q}} = \begin{bmatrix} -h \dot{\theta}_2 \dot{\theta}_1 - h(\dot{\theta}_1 + \dot{\theta}_2)\dot{\theta}_2 \\ h \dot{\theta}_1^2 \end{bmatrix}$$

첫 번째 행에는 코리올리 항($-2h\dot{\theta}_1\dot{\theta}_2$)과 원심력 항($-h\dot{\theta}_2^2$)이 포함된다. 두 번째 행은 조인트 1이 조인트 2에 미치는 원심력 항이다.

### 중력 항

$$\mathbf{g}(\mathbf{q}) = \begin{bmatrix} (m_1 l_{c_1} + m_2 l_1)g\cos\theta_1 + m_2 l_{c_2} g\cos(\theta_1 + \theta_2) \\ m_2 l_{c_2} g\cos(\theta_1 + \theta_2) \end{bmatrix}$$

```python
class TwoLinkDynamics:
    """2링크 평면 로봇 팔의 완전한 동역학.

    This is the workhorse example for understanding manipulator dynamics.
    Every concept — inertia, Coriolis, gravity — is visible and verifiable.
    """
    def __init__(self, m1, m2, l1, l2, lc1, lc2, I1, I2, g=9.81):
        self.m1 = m1    # link 1 mass
        self.m2 = m2    # link 2 mass
        self.l1 = l1    # link 1 length
        self.l2 = l2    # link 2 length
        self.lc1 = lc1  # link 1 center of mass distance from joint 1
        self.lc2 = lc2  # link 2 center of mass distance from joint 2
        self.I1 = I1    # link 1 rotational inertia about its CM
        self.I2 = I2    # link 2 rotational inertia about its CM
        self.g = g

    def inertia_matrix(self, q):
        """2x2 관성 행렬 M(q)를 계산한다.

        Why is M configuration-dependent? Because the effective inertia
        'seen' by each joint motor depends on how the links are arranged.
        When link 2 is extended, joint 1 sees more inertia (longer moment arm).
        When link 2 is folded, joint 1 sees less inertia.
        """
        t2 = q[1]
        c2 = cos(t2)

        a = self.m1*self.lc1**2 + self.m2*(self.l1**2 + self.lc2**2 + \
            2*self.l1*self.lc2*c2) + self.I1 + self.I2
        b = self.m2*(self.lc2**2 + self.l1*self.lc2*c2) + self.I2
        d = self.m2*self.lc2**2 + self.I2

        M = np.array([[a, b],
                       [b, d]])
        return M

    def coriolis_matrix(self, q, q_dot):
        """2x2 코리올리/원심력 행렬 C(q, q_dot)를 계산한다.

        Why do Coriolis terms exist? When joint 1 rotates, link 2's
        center of mass follows a curved path. This curved motion creates
        centripetal acceleration, which appears as a 'phantom force' in
        the joint-space equations. It's not a real external force — it's
        an artifact of using rotating (non-inertial) reference frames.
        """
        t2 = q[1]
        t1_dot, t2_dot = q_dot

        h = self.m2 * self.l1 * self.lc2 * sin(t2)

        C = np.array([[-h*t2_dot,    -h*(t1_dot + t2_dot)],
                       [ h*t1_dot,    0                    ]])
        return C

    def gravity_vector(self, q):
        """2x1 중력 벡터 g(q)를 계산한다.

        Why does gravity depend on q? Because the gravitational torque
        on each joint depends on the horizontal distance of each link's
        center of mass from that joint. As the arm moves, these distances
        change. At theta=0 (horizontal), gravity torque is maximum;
        at theta=90 (vertical), it's zero.
        """
        t1 = q[0]
        t12 = q[0] + q[1]
        g = self.g

        g1 = (self.m1*self.lc1 + self.m2*self.l1)*g*cos(t1) + \
              self.m2*self.lc2*g*cos(t12)
        g2 = self.m2*self.lc2*g*cos(t12)

        return np.array([g1, g2])

    def forward_dynamics(self, q, q_dot, tau):
        """주어진 토크에서 조인트 가속도를 계산한다.

        tau = M*q_ddot + C*q_dot + g
        =>  q_ddot = M^{-1} * (tau - C*q_dot - g)

        This is the 'forward dynamics' problem: given forces, find motion.
        Used in simulation.
        """
        M = self.inertia_matrix(q)
        C = self.coriolis_matrix(q, q_dot)
        g = self.gravity_vector(q)

        q_ddot = np.linalg.solve(M, tau - C @ q_dot - g)
        return q_ddot

    def inverse_dynamics(self, q, q_dot, q_ddot):
        """원하는 운동에 필요한 토크를 계산한다.

        tau = M*q_ddot + C*q_dot + g

        This is the 'inverse dynamics' problem: given desired motion,
        find required forces. Used in computed torque control.
        """
        M = self.inertia_matrix(q)
        C = self.coriolis_matrix(q, q_dot)
        g = self.gravity_vector(q)

        tau = M @ q_ddot + C @ q_dot + g
        return tau


# 2링크 팔 생성
arm = TwoLinkDynamics(
    m1=5.0, m2=3.0,       # masses in kg
    l1=0.5, l2=0.4,       # link lengths in m
    lc1=0.25, lc2=0.2,    # center of mass distances
    I1=0.1, I2=0.05       # rotational inertias in kg*m^2
)

# 형태 구성: 두 링크 모두 45도
q = np.radians([45, 30])
q_dot = np.array([0.5, -0.3])  # some joint velocities

print("=== 2-Link Arm Dynamics ===")
print(f"Configuration: q = [{np.degrees(q[0]):.0f}, {np.degrees(q[1]):.0f}] deg")
print(f"Velocities: q_dot = {q_dot}")

M = arm.inertia_matrix(q)
C = arm.coriolis_matrix(q, q_dot)
g = arm.gravity_vector(q)

print(f"\nInertia matrix M:")
print(np.round(M, 4))
print(f"\nCoriolis matrix C:")
print(np.round(C, 4))
print(f"\nGravity vector g: {np.round(g, 4)}")

# 정지 상태를 유지하는 데 필요한 토크 (q_ddot = 0, q_dot = 0)?
tau_static = arm.inverse_dynamics(q, np.zeros(2), np.zeros(2))
print(f"\nTorques to hold position: {np.round(tau_static, 4)} N*m")
print(f"(These equal the gravity vector — only gravity acts when stationary)")
```

---

## 관성 행렬의 성질

관성 행렬 $M(\mathbf{q})$는 몇 가지 중요한 성질을 갖는다.

### 1. 대칭성

$M(\mathbf{q}) = M(\mathbf{q})^T$는 항상 성립한다. 이것은 운동 에너지의 이차 형식(quadratic form)으로서의 정의로부터 따른다.

### 2. 양정치성(Positive Definite)

$\dot{\mathbf{q}}^T M(\mathbf{q}) \dot{\mathbf{q}} > 0$이 모든 $\dot{\mathbf{q}} \neq 0$에 대해 성립한다. 이것은 로봇이 정지해 있지 않으면 운동 에너지가 항상 양수임을 의미한다.

### 3. 형태 구성 의존성

$M$은 로봇이 움직임에 따라 변한다. 일부 형태 구성에서는 특정 조인트가 다른 형태 구성보다 더 많은 관성을 경험한다.

### 4. 유계성(Bounded)

$\lambda_{min}(M) \leq \frac{\dot{\mathbf{q}}^T M \dot{\mathbf{q}}}{\|\dot{\mathbf{q}}\|^2} \leq \lambda_{max}(M)$

$M$의 고유값(eigenvalue)은 유계인 조인트 형태 구성에 대해 위아래로 유계이다. 이것은 제어 설계에 중요하다.

```python
def verify_inertia_properties(arm, n_samples=100):
    """관성 행렬의 핵심 성질을 검증한다.

    Why verify? Because analytical derivations can have sign errors
    or missing terms. Verifying properties catches these bugs —
    a non-positive-definite M means something is wrong in the derivation.
    """
    print("Verifying inertia matrix properties:")

    for _ in range(n_samples):
        q = np.random.uniform(-np.pi, np.pi, 2)
        M = arm.inertia_matrix(q)

        # 대칭성
        assert np.allclose(M, M.T), f"M not symmetric at q={q}"

        # 양정치성 (모든 고유값 > 0)
        eigvals = np.linalg.eigvalsh(M)
        assert np.all(eigvals > 0), f"M not positive definite at q={q}, eigs={eigvals}"

    print(f"  Symmetry: PASS ({n_samples} random configs)")
    print(f"  Positive definiteness: PASS ({n_samples} random configs)")

    # 형태 구성에 따른 M의 변화 표시
    print("\nInertia variation with theta2:")
    for t2_deg in [0, 30, 60, 90, 120, 150, 180]:
        q = np.array([0, np.radians(t2_deg)])
        M = arm.inertia_matrix(q)
        print(f"  theta2={t2_deg:>3d} deg: M11={M[0,0]:.4f}, "
              f"M12={M[0,1]:.4f}, M22={M[1,1]:.4f}")

verify_inertia_properties(arm)
```

---

## 코리올리와 원심력 효과

### 물리적 기원

**원심력 효과**: 조인트 $i$가 회전할 때 링크 $i+1$ (및 그 이후)에 대한 원운동으로 인한 "바깥쪽" 유사 힘. $\dot{q}_i^2$에 비례한다.

**코리올리 효과**: 두 조인트 속도 사이의 교차 결합. 조인트 $i$와 $j$가 모두 움직이면 다른 조인트에 $\dot{q}_i \dot{q}_j$에 비례하는 힘이 발생한다.

### 왜대칭성 성질

제어 이론에서 사용되는 근본적인 성질:

$$\dot{M}(\mathbf{q}) - 2C(\mathbf{q}, \dot{\mathbf{q}})$$

은 왜대칭(skew-symmetric)이다. 이것은 다음을 의미한다.

$$\dot{\mathbf{q}}^T [\dot{M} - 2C] \dot{\mathbf{q}} = 0$$

이 성질은 많은 로봇 제어기(예: 계산 토크 제어, 적응 제어)의 안정성을 증명하는 데 필수적이다.

```python
def verify_skew_symmetry(arm, q, q_dot, delta=1e-7):
    """왜대칭성 성질 검증: M_dot - 2C는 왜대칭이다.

    Why is this important? Because it implies that the 'power' of the
    Coriolis/centrifugal forces is zero: these forces do no work.
    They only redirect energy between joints, never create or destroy it.
    This is the robotic equivalent of the fact that the Coriolis force
    in rotating frames does no work (it's always perpendicular to velocity).

    This property is used in Lyapunov stability proofs for controllers.
    """
    # 유한 차분법을 통한 수치적 M_dot
    M = arm.inertia_matrix(q)
    q_shifted = q + delta * q_dot
    M_shifted = arm.inertia_matrix(q_shifted)
    M_dot = (M_shifted - M) / delta

    C = arm.coriolis_matrix(q, q_dot)

    S = M_dot - 2 * C

    # 왜대칭성 확인: S + S^T는 0이어야 함
    is_skew = np.allclose(S + S.T, 0, atol=1e-4)
    print(f"M_dot - 2C:")
    print(np.round(S, 6))
    print(f"Skew-symmetric? {is_skew}")

    # 검증: q_dot^T * S * q_dot = 0
    power = q_dot @ S @ q_dot
    print(f"q_dot^T * S * q_dot = {power:.2e} (should be ~0)")

verify_skew_symmetry(arm, np.radians([45, 30]), np.array([1.0, -0.5]))
```

---

## 중력 보상

### 가장 단순한 동역학 제어기

많은 응용에서 로봇은 위치를 유지하거나 느린 궤적을 따라야 한다. 이런 경우 지배적인 동역학 효과는 중력이다. **중력 보상(gravity compensation)**은 순수 기구학 제어에 비해 가장 단순하면서도 가장 큰 개선이다.

$$\boldsymbol{\tau} = \mathbf{g}(\mathbf{q}) + K_p (\mathbf{q}_d - \mathbf{q}) + K_d (\dot{\mathbf{q}}_d - \dot{\mathbf{q}})$$

이것은 중력 순방향 보상(feedforward)이 포함된 PD 제어기이다. 중력 보상 없이는 PD 제어기가 중력에 대항하기 위한 큰 토크를 생성해야 하므로 추종을 위한 여유가 줄어든다.

```python
def gravity_compensation_controller(arm, q, q_dot, q_desired, q_dot_desired,
                                    Kp, Kd):
    """중력 보상이 포함된 PD 제어기.

    Why add gravity compensation? Consider holding the arm horizontal.
    Without it, the PD controller must generate a constant error-correcting
    torque equal to the gravitational torque. This means the arm hangs
    below the desired position (steady-state error for P control) or
    requires very high gains (which cause oscillation).

    Gravity compensation eliminates this: the feedforward term handles
    gravity exactly, and the PD terms only need to handle dynamics.
    """
    g = arm.gravity_vector(q)

    # PD 제어
    tau_pd = Kp @ (q_desired - q) + Kd @ (q_dot_desired - q_dot)

    # 총 토크 = 중력 보상 + PD
    tau = g + tau_pd

    return tau

# 중력 보상 유무에 따른 위치 유지 시뮬레이션
def simulate(arm, controller, q0, q_desired, dt=0.001, duration=2.0):
    """단순한 오일러 적분 시뮬레이션.

    Why simulate? Because it reveals the actual behavior of the
    controller, including transient response, steady-state error,
    and stability issues that analysis alone might miss.
    """
    n_steps = int(duration / dt)
    q = q0.copy()
    q_dot = np.zeros(2)
    trajectory = [q.copy()]

    for _ in range(n_steps):
        tau = controller(q, q_dot, q_desired)
        q_ddot = arm.forward_dynamics(q, q_dot, tau)

        # 오일러 적분
        q_dot = q_dot + q_ddot * dt
        q = q + q_dot * dt

        trajectory.append(q.copy())

    return np.array(trajectory)

# 중력 보상 없는 제어기
Kp = np.diag([50, 30])
Kd = np.diag([10, 5])
q_desired = np.radians([45, 30])

def ctrl_pd_only(q, q_dot, q_des):
    return Kp @ (q_des - q) + Kd @ (0 - q_dot)

def ctrl_with_grav(q, q_dot, q_des):
    return arm.gravity_vector(q) + Kp @ (q_des - q) + Kd @ (0 - q_dot)

# 두 경우 시뮬레이션
traj_pd = simulate(arm, ctrl_pd_only, np.zeros(2), q_desired)
traj_grav = simulate(arm, ctrl_with_grav, np.zeros(2), q_desired)

# 최종 오차 비교
error_pd = np.degrees(np.abs(traj_pd[-1] - q_desired))
error_grav = np.degrees(np.abs(traj_grav[-1] - q_desired))

print("Steady-state error (degrees):")
print(f"  PD only:       joint1={error_pd[0]:.2f}, joint2={error_pd[1]:.2f}")
print(f"  PD + gravity:  joint1={error_grav[0]:.2f}, joint2={error_grav[1]:.2f}")
```

---

## 계산 토크 제어

### 완전한 동역학 보상

**계산 토크(computed torque)** 제어(**역동역학 제어(inverse dynamics control)** 또는 **피드백 선형화(feedback linearization)**라고도 함)는 완전한 동역학 모델을 사용하여 비선형성을 상쇄한다.

$$\boldsymbol{\tau} = M(\mathbf{q}) \mathbf{a} + C(\mathbf{q}, \dot{\mathbf{q}}) \dot{\mathbf{q}} + \mathbf{g}(\mathbf{q})$$

여기서 $\mathbf{a}$는 "가상 가속도(virtual acceleration)" 명령이다.

$$\mathbf{a} = \ddot{\mathbf{q}}_d + K_d (\dot{\mathbf{q}}_d - \dot{\mathbf{q}}) + K_p (\mathbf{q}_d - \mathbf{q})$$

동역학 방정식에 대입하면:

$$M \ddot{\mathbf{q}} + C \dot{\mathbf{q}} + \mathbf{g} = M \mathbf{a} + C \dot{\mathbf{q}} + \mathbf{g}$$

$$\ddot{\mathbf{q}} = \mathbf{a} = \ddot{\mathbf{q}}_d + K_d (\dot{\mathbf{q}}_d - \dot{\mathbf{q}}) + K_p (\mathbf{q}_d - \mathbf{q})$$

결과적인 오차 동역학은 선형이다: $\ddot{\mathbf{e}} + K_d \dot{\mathbf{e}} + K_p \mathbf{e} = 0$

적절한 이득 선택($K_p$, $K_d$으로 특성 다항식을 안정화)으로 추종 오차가 지수적으로 0으로 수렴한다.

```python
def computed_torque_controller(arm, q, q_dot, q_desired, q_dot_desired,
                               q_ddot_desired, Kp, Kd):
    """계산 토크 (역동역학) 제어기.

    Why is this the 'gold standard'? Because it exactly linearizes the
    system. After cancellation, the closed-loop behaves like a linear
    spring-damper system, regardless of the robot's nonlinear dynamics.

    The catch: it requires a perfect dynamics model. Model errors
    (inaccurate masses, unmodeled friction) degrade performance.
    In practice, robust or adaptive variants are used.
    """
    # 오차 계산
    e = q_desired - q
    e_dot = q_dot_desired - q_dot

    # 가상 가속도 명령
    a = q_ddot_desired + Kd @ e_dot + Kp @ e

    # 역동역학: 가속도 'a'를 위한 필요 토크 계산
    M = arm.inertia_matrix(q)
    C = arm.coriolis_matrix(q, q_dot)
    g = arm.gravity_vector(q)

    tau = M @ a + C @ q_dot + g
    return tau

# 계산 토크 제어 시뮬레이션
Kp = np.diag([100, 100])   # 동역학이 상쇄되므로 더 높은 이득 가능
Kd = np.diag([20, 20])

def ctrl_computed_torque(q, q_dot, q_des):
    return computed_torque_controller(
        arm, q, q_dot, q_des, np.zeros(2), np.zeros(2), Kp, Kd)

traj_ct = simulate(arm, ctrl_computed_torque, np.zeros(2), q_desired, duration=1.0)
error_ct = np.degrees(np.abs(traj_ct[-1] - q_desired))
print(f"\nComputed torque error: joint1={error_ct[0]:.4f} deg, "
      f"joint2={error_ct[1]:.4f} deg")
```

---

## 뉴턴-오일러 공식

### 개요

뉴턴-오일러 접근법은 **두 번의 재귀적 알고리즘**으로 동역학을 계산한다.

1. **순방향 패스(base to tip)**: 링크 속도와 가속도를 반복적으로 계산
2. **역방향 패스(tip to base)**: 뉴턴과 오일러의 방정식을 사용하여 각 조인트의 힘과 토크를 계산

### 오일러-라그랑주와의 비교

| 측면 | 오일러-라그랑주 | 뉴턴-오일러 |
|--------|---------------|--------------|
| 공식화 | 에너지 기반 | 힘/토크 균형 |
| 계산 | 기호적 (해석적) | 재귀적 수치적 |
| 복잡도 | $O(n^3)$ 또는 $O(n^4)$ | $O(n)$ — **선형!** |
| 통찰 | 구조 드러냄 (M, C, g) | 효율적인 계산 |
| 사용 | 분석, 제어 설계 | 실시간 시뮬레이션, 순방향 보상 |

```python
def newton_euler_2link(arm, q, q_dot, q_ddot):
    """2링크 평면 로봇의 뉴턴-오일러 재귀적 동역학.

    Why Newton-Euler alongside Euler-Lagrange? Because for real-time
    control of robots with 6+ joints, the recursive O(n) algorithm is
    essential. Euler-Lagrange gives beautiful analytical expressions but
    its computational cost grows quickly with the number of joints.

    For a 6-DOF robot:
    - Euler-Lagrange: ~66,000 multiplications
    - Newton-Euler: ~852 multiplications
    That's nearly 80x faster!
    """
    m1, m2 = arm.m1, arm.m2
    l1, l2 = arm.l1, arm.l2
    lc1, lc2 = arm.lc1, arm.lc2
    I1, I2 = arm.I1, arm.I2
    g = arm.g
    t1, t2 = q
    t1d, t2d = q_dot
    t1dd, t2dd = q_ddot

    # === 순방향 패스: 속도와 가속도 계산 ===

    # 링크 1 각속도와 각가속도
    w1 = t1d
    alpha1 = t1dd

    # 링크 1 질량 중심 선가속도 (중력 포함)
    # a_c1 = d^2/dt^2 (lc1 * [cos(t1), sin(t1)])
    # 기저를 유사 가속도로 취급: a0 = [0, g, 0]
    ac1_x = -lc1 * sin(t1) * t1dd - lc1 * cos(t1) * t1d**2
    ac1_y = lc1 * cos(t1) * t1dd - lc1 * sin(t1) * t1d**2 + g

    # 링크 2 각속도와 각가속도
    w2 = t1d + t2d
    alpha2 = t1dd + t2dd

    # 조인트 2 위치 가속도
    a2_x = -l1 * sin(t1) * t1dd - l1 * cos(t1) * t1d**2
    a2_y = l1 * cos(t1) * t1dd - l1 * sin(t1) * t1d**2 + g

    # 링크 2 질량 중심 가속도
    t12 = t1 + t2
    w12 = t1d + t2d
    ac2_x = a2_x - lc2 * sin(t12) * (t1dd + t2dd) - lc2 * cos(t12) * w12**2
    ac2_y = a2_y + lc2 * cos(t12) * (t1dd + t2dd) - lc2 * sin(t12) * w12**2

    # === 역방향 패스: 힘과 토크 계산 ===

    # 링크 2 힘 (뉴턴: F = m*a)
    f2_x = m2 * ac2_x
    f2_y = m2 * ac2_y

    # 링크 2 토크 (오일러: tau = I*alpha + r x F)
    tau2 = I2 * alpha2 + lc2 * (cos(t12) * f2_y - sin(t12) * f2_x)
    # 평면 케이스 단순화: tau2는 조인트 2에 대한 모멘트를 포함

    # 링크 1 힘 (링크 2의 반력 포함)
    f1_x = m1 * ac1_x + f2_x
    f1_y = m1 * ac1_y + f2_y

    # 링크 1 토크
    tau1 = I1 * alpha1 + lc1 * (cos(t1) * f1_y - sin(t1) * f1_x) + tau2 + \
           l1 * (cos(t1) * f2_y - sin(t1) * f2_x)

    return np.array([tau1, tau2])

# 오일러-라그랑주와 뉴턴-오일러 비교
q = np.radians([45, 30])
q_dot = np.array([1.0, -0.5])
q_ddot = np.array([0.5, 0.3])

tau_el = arm.inverse_dynamics(q, q_dot, q_ddot)
tau_ne = newton_euler_2link(arm, q, q_dot, q_ddot)

print("Euler-Lagrange torques:", np.round(tau_el, 4))
print("Newton-Euler torques:  ", np.round(tau_ne, 4))
print(f"Match: {np.allclose(tau_el, tau_ne, atol=1e-3)}")
```

---

## 시뮬레이션: 자유 낙하와 중력 효과

```python
def simulate_free_fall(arm, q0, dt=0.001, duration=2.0):
    """팔이 중력 하에서 자유 낙하하는 것을 시뮬레이션한다 (가해진 토크 없음).

    Why simulate free fall? It's the ultimate test of the dynamics model.
    The arm should swing like a double pendulum — chaotic, energy-conserving
    (if no friction), and physically realistic.
    """
    n_steps = int(duration / dt)
    q = q0.copy()
    q_dot = np.zeros(2)
    history = {'t': [], 'q': [], 'q_dot': [], 'energy': []}

    for step in range(n_steps):
        t = step * dt

        # 가해진 토크 없음 — 중력만
        q_ddot = arm.forward_dynamics(q, q_dot, np.zeros(2))

        # 에너지 보존 확인을 위한 에너지 계산
        M = arm.inertia_matrix(q)
        K = 0.5 * q_dot @ M @ q_dot  # kinetic energy
        # 위치 에너지 (평면 케이스 근사)
        P = (arm.m1 * arm.lc1 * sin(q[0]) + \
             arm.m2 * (arm.l1 * sin(q[0]) + arm.lc2 * sin(q[0] + q[1]))) * arm.g
        E = K + P

        history['t'].append(t)
        history['q'].append(q.copy())
        history['q_dot'].append(q_dot.copy())
        history['energy'].append(E)

        # 오일러 적분 (더 높은 정확도를 위해 RK4 사용)
        q_dot = q_dot + q_ddot * dt
        q = q + q_dot * dt

    return history

# 수평 위치에서 팔 낙하
history = simulate_free_fall(arm, np.radians([90, 0]))

# 에너지 보존 확인 (보존 시스템에서는 일정해야 함)
energies = np.array(history['energy'])
energy_drift = (energies[-1] - energies[0]) / abs(energies[0])
print(f"Energy drift over 2s: {energy_drift*100:.2f}%")
print(f"  (Non-zero drift is due to Euler integration — use RK4 for better results)")

# 궤적 표시
qs = np.array(history['q'])
print(f"\nFinal configuration: q = {np.degrees(qs[-1]).round(1)} deg")
print(f"Final velocity: q_dot = {np.array(history['q_dot'][-1]).round(2)} rad/s")
```

---

## 요약

- **로봇 동역학**은 조인트 토크와 운동을 연결한다: $M(\mathbf{q})\ddot{\mathbf{q}} + C(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q}) = \boldsymbol{\tau}$
- **오일러-라그랑주** 공식은 에너지(운동 에너지 - 위치 에너지)로부터 동역학을 유도하여 우아한 해석적 표현을 제공한다
- **관성 행렬** $M$은 대칭, 양정치, 형태 구성 의존적이다
- **코리올리/원심력** 항은 속도 의존 결합 힘을 나타내며, $\dot{M}$과의 왜대칭성 성질을 만족한다
- **중력 보상** ($\tau = g(q) + \text{PD}$)은 가장 단순하면서도 가장 큰 효과를 가진 동역학 제어기이다
- **계산 토크 제어**는 완전한 모델을 사용하여 시스템을 선형화하고 선형 제어 설계를 가능하게 한다
- **뉴턴-오일러** 재귀적 알고리즘은 $O(n)$ 시간에 역동역학을 계산하여 많은 조인트를 가진 실시간 응용에 필수적이다

---

## 연습 문제

### 연습 1: 단일 링크 동역학

질량 $m = 2$ kg, 길이 $l = 0.5$ m, 질량 중심이 $l_c = 0.25$ m에 위치하고, 관성 $I = 0.02$ kg m$^2$인 단일 링크 팔 (진자)에 대해:
1. 오일러-라그랑주를 사용하여 운동 방정식을 유도하라
2. $\theta = 0°$, $45°$, $90°$에서 중력 토크를 계산하라
3. 팔을 $\theta = 45°$에서 유지하는 데 필요한 토크는 얼마인가?
4. 오일러 적분을 사용하여 $\theta = 90°$에서 자유 낙하를 시뮬레이션하라

### 연습 2: 관성 행렬 분석

이 레슨의 2링크 팔에 대해:
1. $\theta_1 = 0$으로 고정하고 $\theta_2 = 0°$, $90°$, $180°$에서 $M(q)$를 계산하라
2. 각 형태 구성에서 고유값을 계산하라. 어떻게 변화하는가?
3. $M_{11}$이 최대가 되는 $\theta_2$는 어디인가? 최소는? 물리적으로 설명하라
4. 100개의 무작위 형태 구성에서 양정치성을 검증하라

### 연습 3: 코리올리 효과

2링크 팔을 이용하여:
1. $q = (45°, 60°)$이고 $\dot{q} = (2, 0)$ (조인트 1만 움직임)일 때 조인트 2에 대한 코리올리 토크를 계산하라
2. 조인트 1이 움직이면 조인트 2에 토크가 생기는 이유를 물리적으로 설명하라
3. 3가지 다른 형태 구성에서 왜대칭성 성질을 수치적으로 검증하라

### 연습 4: 제어기 비교

$q = (0, 0)$에서 $q_d = (45°, 30°)$로의 계단 입력을 추종하는 2링크 팔에 대해 세 가지 제어기를 구현하고 비교하라:
1. PD 제어만 (모델 보상 없음)
2. PD + 중력 보상
3. 계산 토크 제어
각 경우에 추종 오차와 조인트 토크를 플로팅하라. 정상 상태 오차와 과도 응답을 파악하라.

### 연습 5: 뉴턴-오일러 구현

1. 일반적인 $n$자유도 직렬 매니퓰레이터를 위한 뉴턴-오일러 재귀적 알고리즘을 구현하라
2. 5개의 무작위 형태 구성에서 2링크 팔의 오일러-라그랑주 결과와 검증하라
3. 역동역학 계산에 대해 두 방법의 속도를 측정하라. 속도가 어떻게 비교되는가?

---

[← 이전: 속도 기구학](05_Velocity_Kinematics.md) | [다음: 모션 플래닝 →](07_Motion_Planning.md)
