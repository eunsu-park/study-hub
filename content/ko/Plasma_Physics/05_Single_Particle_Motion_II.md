# 5. 단일 입자 운동 II

## 학습 목표

- 자기화된 플라즈마에서 임의의 힘에 대한 일반 드리프트 속도 공식 유도
- 불균일한 자기장에서 grad-B 드리프트와 그 물리적 기원 이해
- 곡선 자기장 기하학에서 곡률 드리프트와 그 역할 분석
- 시간 변동 전기장에 의한 편극 드리프트 계산 및 질량 의존성
- 모든 드리프트 유형의 전하 및 질량 의존성 비교 및 대조
- Python을 사용하여 공간적으로 변화하는 자기장에서 입자 궤도 시뮬레이션

## 1. 일반 드리프트 속도 공식

### 1.1 기본 원리로부터의 유도

Lesson 4에서 우리는 $\mathbf{E}\times\mathbf{B}$ 드리프트를 공부했습니다. 이제 임의의 힘으로 일반화합니다. 추가 힘 $\mathbf{F}$를 받는 균일한 자기장 $\mathbf{B}$에서의 입자를 고려합니다.

운동 방정식은:

$$
m\frac{d\mathbf{v}}{dt} = q(\mathbf{E} + \mathbf{v}\times\mathbf{B}) + \mathbf{F}
$$

속도를 다음과 같이 분해합니다:
- 자기력선 주위의 자이로운동
- $\mathbf{B}$에 수직인 드리프트
- $\mathbf{B}$에 평행한 운동

천천히 변하는 힘(자이로주파수 $\omega_c$에 비해 느림)에 대해, 자이로주기에 걸쳐 평균할 수 있습니다. 수직 드리프트 속도는:

$$
\mathbf{v}_D = \frac{\mathbf{F}\times\mathbf{B}}{qB^2}
$$

**핵심 통찰**: $\mathbf{B}$에 수직인 모든 힘 $\mathbf{F}$는 $\mathbf{F}\times\mathbf{B}$ 방향으로 드리프트를 일으킵니다.

### 1.2 물리적 해석

드리프트는 힘 $\mathbf{F}$가 자이로궤도의 다른 지점에서 입자의 속도를 다르게 수정하기 때문에 발생합니다:

```
     위에서 본 자이로궤도 (B가 위를 가리킴)

     F → (오른쪽으로의 힘)

        v 증가
           ↑
     ←-----o-----→  드리프트 방향: ⊙ (지면 안으로)
           ↓         F × B가 지면 안을 가리킴
        v 감소

     v가 큰 곳에서 반경이 더 큼
     → B와 F 모두에 수직인 알짜 변위
```

드리프트 방향은:
- $\mathbf{F}$가 전하 독립적(예: 중력)이면 전하 부호에 독립적
- $\mathbf{F}$가 전하에 의존(예: 전기력)하면 반대 전하에 대해 반대

### 1.3 E×B 드리프트와의 비교

전기력 $\mathbf{F} = q\mathbf{E}$에 대해:

$$
\mathbf{v}_E = \frac{q\mathbf{E}\times\mathbf{B}}{qB^2} = \frac{\mathbf{E}\times\mathbf{B}}{B^2}
$$

이것은 예상대로 전하와 질량에 독립적인 $\mathbf{E}\times\mathbf{B}$ 드리프트를 복원합니다.

## 2. Grad-B 드리프트

### 2.1 물리적 기원

$|\mathbf{B}|$가 공간적으로 변하는 불균일한 자기장에서, 자이로하는 입자는 궤도 동안 다른 자기장 세기를 경험합니다. Larmor 반경이 $B$에 의존하므로:

$$
r_L = \frac{mv_\perp}{|q|B}
$$

반경은 $B$가 약한 곳에서 더 큽니다. 이 비대칭성이 알짜 드리프트를 생성합니다.

### 2.2 유도 중심 전개를 사용한 유도

x-방향으로 변하는 자기장을 고려합니다: $\mathbf{B} = B(x)\hat{z}$. 유도 중심 위치 $x_0$ 주위로 전개:

$$
B(x) \approx B_0 + \left(\frac{\partial B}{\partial x}\right)_0 (x - x_0)
$$

자이로운동 중 입자 위치는:

$$
x = x_0 + r_L\cos(\omega_c t)
$$

공간 변화로부터의 유효 힘은:

$$
F_x = -\mu\frac{\partial B}{\partial x}
$$

여기서 자기 모멘트 $\mu = \frac{mv_\perp^2}{2B}$는 첫 번째 단열 불변량입니다 (천천히 변하는 장에 대해 보존됨).

자이로주기에 걸쳐 평균하고 일반 드리프트 공식을 사용:

$$
\mathbf{v}_{\nabla B} = \frac{\mathbf{F}_{\nabla B}\times\mathbf{B}}{qB^2} = \frac{-\mu\nabla B\times\mathbf{B}}{qB^2}
$$

$\mu = \frac{mv_\perp^2}{2B}$이므로:

$$
\boxed{\mathbf{v}_{\nabla B} = \pm\frac{mv_\perp^2}{2qB^3}(\mathbf{B}\times\nabla B)}
$$

부호는 전하에 의존합니다: 이온과 전자는 **반대** 방향으로 드리프트합니다.

### 2.3 대체 형태

$\nabla B = (\nabla B)$를 $\mathbf{B}$에 수직인 방향으로 사용:

$$
\mathbf{v}_{\nabla B} = \pm\frac{v_\perp^2}{2\omega_c}\frac{\mathbf{B}\times\nabla B}{B^2}
$$

여기서 $\omega_c = |q|B/m$는 자이로주파수입니다.

### 2.4 Grad-B 드리프트로부터의 전류

이온과 전자가 반대 방향으로 드리프트하므로, grad-B 드리프트는 **반자성 전류**를 생성합니다:

$$
\mathbf{J}_{\nabla B} = n(q_i\mathbf{v}_{\nabla B,i} + q_e\mathbf{v}_{\nabla B,e})
$$

$q_i = e$, $q_e = -e$이고 $T_i = T_e = T$를 가정한 플라즈마에 대해:

$$
\mathbf{J}_{\nabla B} = -\frac{2nkT}{B^2}\mathbf{B}\times\nabla B
$$

이 전류는 기울기를 반대하는 자기장을 만들어, "반자성"이라고 합니다.

### 2.5 수치 예제

**예제**: 지구 자기권의 양성자.
- $B_0 = 10^{-5}$ T, $\nabla B \sim 10^{-7}$ T/m (쌍극자 장 기울기)
- $v_\perp = 10^5$ m/s, $m = 1.67\times 10^{-27}$ kg, $q = 1.6\times 10^{-19}$ C

$$
v_{\nabla B} = \frac{mv_\perp^2}{2qB^2}\frac{\nabla B}{B} \approx \frac{(1.67\times 10^{-27})(10^5)^2}{2(1.6\times 10^{-19})(10^{-5})^2} \cdot 10^{-2} \approx 5.2\times 10^3 \text{ m/s}
$$

자이로운동 속도보다는 훨씬 작지만 긴 시간 규모에서 중요합니다.

## 3. 곡률 드리프트

### 3.1 곡선 자기력선

자기력선이 곡선일 때, 자기력선을 따라 움직이는 입자는 유도 중심 좌표계에서 원심력을 경험합니다. 곡률 반경 $R_c$는 다음으로 정의됩니다:

$$
\frac{\mathbf{B}}{B}\cdot\nabla\frac{\mathbf{B}}{B} = -\frac{\mathbf{R}_c}{R_c^2}
$$

여기서 $\mathbf{R}_c$는 곡률 중심을 가리킵니다.

### 3.2 원심력

평행 속도 $v_\parallel$로 곡선 자기력선을 따르는 입자는 다음을 느낍니다:

$$
\mathbf{F}_c = \frac{mv_\parallel^2}{R_c}\hat{R}_c = \frac{mv_\parallel^2}{R_c^2}\mathbf{R}_c
$$

일반 드리프트 공식 사용:

$$
\boxed{\mathbf{v}_R = \frac{mv_\parallel^2}{qB^2}\frac{\mathbf{R}_c\times\mathbf{B}}{R_c^2}}
$$

이것이 **곡률 드리프트**입니다.

### 3.3 결합된 Grad-B와 곡률 드리프트

대부분의 현실적인 기하학(예: 토카막, 쌍극자 장)에서, 곡률과 자기장 기울기가 함께 발생합니다. 변하는 크기를 가진 곡선 장에 대해:

$$
\nabla B \approx \frac{B}{R_c}
$$

총 드리프트는:

$$
\mathbf{v}_{gc} = \mathbf{v}_{\nabla B} + \mathbf{v}_R = \frac{m}{qB^2}\left(\frac{v_\perp^2}{2} + v_\parallel^2\right)\frac{\mathbf{B}\times\nabla B}{B}
$$

이것은 총 운동 에너지를 사용하여 쓸 수도 있습니다:

$$
\mathbf{v}_{gc} = \frac{m}{qB^3}\left(\frac{v_\perp^2}{2} + v_\parallel^2\right)(\mathbf{B}\times\nabla B)
$$

### 3.4 환형 장 기하학

토카막에서, 환형 장 $B_\phi \propto 1/R$는 곡률과 기울기 모두를 만듭니다:

```
    토카막 단면 (폴로이달 평면)

         약한 장              강한 장
         (큰 R)                (작은 R)
            |                         |
            |                         |
         ───┼────────────o────────────┼───
            |        플라즈마          |
            |         중심            |
            |                         |

    Grad-B 드리프트: ∇B가 안쪽을 가리킴 (주축 쪽)
    이온 (+)의 경우: 아래쪽으로 드리프트 (폴로이달 평면에서)
    전자 (−)의 경우: 위쪽으로 드리프트

    → 전하 분리 → 수직 전기장
    → E×B 드리프트 바깥쪽 → 입자 손실

    (자기 전단과 환형 회전으로 해결)
```

## 4. 편극 드리프트

### 4.1 시간 변동 전기장

전기장이 시간에 따라 변할 때, $\mathbf{E}\times\mathbf{B}$ 드리프트 속도가 변하고, 입자가 가속해야 합니다. 이것이 추가 드리프트를 만듭니다.

다음으로부터 시작:

$$
m\frac{d\mathbf{v}}{dt} = q(\mathbf{E} + \mathbf{v}\times\mathbf{B})
$$

$\mathbf{v} = \mathbf{v}_E + \mathbf{v}_\perp + \mathbf{v}_\parallel$로 분해, 여기서 $\mathbf{v}_E = \mathbf{E}\times\mathbf{B}/B^2$는 $\mathbf{E}\times\mathbf{B}$ 드리프트입니다.

$\mathbf{v}_E$의 시간 미분:

$$
\frac{d\mathbf{v}_E}{dt} = \frac{1}{B^2}\frac{d\mathbf{E}}{dt}\times\mathbf{B}
$$

수직 속도에 대한 운동 방정식은:

$$
m\frac{d\mathbf{v}_\perp}{dt} = q\mathbf{v}_\perp\times\mathbf{B} - m\frac{d\mathbf{v}_E}{dt}
$$

마지막 항이 추가 힘처럼 작용합니다. 자이로주기에 걸쳐 평균하고 일반 드리프트 공식 적용:

$$
\boxed{\mathbf{v}_P = \frac{m}{qB^2}\frac{d\mathbf{E}_\perp}{dt}}
$$

이것이 **편극 드리프트**입니다.

### 4.2 질량 의존성

$\mathbf{v}_P \propto m/q$임을 주목하세요:
- 이온(무거움): 중요한 드리프트
- 전자(가벼움): 무시할 만한 드리프트 ($m_e/m_i \sim 1/1836$ 배)

이것은 **편극 전류**를 만듭니다:

$$
\mathbf{J}_P = \sum_s n_s q_s \mathbf{v}_{P,s} \approx n_i q_i \mathbf{v}_{P,i} = \frac{n_i m_i}{B^2}\frac{d\mathbf{E}_\perp}{dt}
$$

전자 기여는 무시할 만합니다.

### 4.3 물리적 해석

$\mathbf{E}$가 변할 때, $\mathbf{E}\times\mathbf{B}$ 드리프트가 변합니다. 이온은 무거워서 즉시 반응할 수 없고 새로운 드리프트 속도를 "초과" 하거나 "뒤쳐집니다". 이 과도 운동이 편극 드리프트입니다.

```
    E가 갑자기 증가

    이전:  vE (작음)
             ───→

    이후:   vE (큼)
             ─────────→

    이온이 뒤쳐짐: ──→  (처음에 새 vE보다 작음)
    전자는 즉시 반응: ─────────→

    → 과도 기간 동안 알짜 전류
```

### 4.4 파동에서의 편극 드리프트

주파수 $\omega$인 플라즈마 파동에서, $\mathbf{E} = \mathbf{E}_0 e^{-i\omega t}$이면:

$$
\frac{d\mathbf{E}}{dt} = -i\omega\mathbf{E}
$$

편극 드리프트는:

$$
\mathbf{v}_P = -i\frac{m\omega}{qB^2}\mathbf{E}_\perp
$$

이것은 자기화된 플라즈마에서 전자기파에 대한 유전 텐서를 유도하는 데 중요합니다.

## 5. 중력 드리프트

### 5.1 중력장에서의 드리프트

중력 $\mathbf{F}_g = m\mathbf{g}$에 대해, 드리프트는:

$$
\boxed{\mathbf{v}_g = \frac{m}{q}\frac{\mathbf{g}\times\mathbf{B}}{B^2}}
$$

이 드리프트는:
- 전하 부호에 의존 (이온과 전자가 반대 방향으로 드리프트)
- 질량에 비례 (더 무거운 입자가 더 빨리 드리프트)
- 입자 에너지에 독립적

### 5.2 중력 드리프트로부터의 전류

이온과 전자가 반대 방향으로 드리프트하므로:

$$
\mathbf{J}_g = n(q_i\mathbf{v}_{g,i} + q_e\mathbf{v}_{g,e}) = n(m_i - m_e)\frac{\mathbf{g}\times\mathbf{B}}{B^2} \approx n m_i\frac{\mathbf{g}\times\mathbf{B}}{B^2}
$$

실험실 플라즈마에서, 중력은 보통 무시할 만합니다. 하지만 천체물리학 플라즈마(예: 태양 대기)에서, 중력 드리프트가 중요할 수 있습니다.

### 5.3 수치 예제

**예제**: 태양 코로나의 양성자.
- $B = 10^{-3}$ T (강한 활동 영역), $g = 274$ m/s² (태양 표면 중력)
- $m = 1.67\times 10^{-27}$ kg, $q = 1.6\times 10^{-19}$ C

$$
v_g = \frac{mg}{qB} = \frac{(1.67\times 10^{-27})(274)}{(1.6\times 10^{-19})(10^{-3})} \approx 2.9 \text{ m/s}
$$

작지만 태양 시간 규모(몇 시간에서 며칠)에 걸쳐 무시할 수 없습니다.

## 6. 모든 드리프트 요약

### 6.1 종합 드리프트 표

| 드리프트 유형 | 공식 | 전하 의존성 | 질량 의존성 | 물리적 기원 |
|------------|---------|-------------------|-----------------|-----------------|
| **일반** | $\mathbf{F}\times\mathbf{B}/(qB^2)$ | $\mathbf{F}$에 의존 | $\mathbf{F}$에 의존 | 임의의 힘 |
| **E×B** | $\mathbf{E}\times\mathbf{B}/B^2$ | 독립적 | 독립적 | 전기력 |
| **Grad-B** | $\pm\frac{mv_\perp^2}{2qB^3}(\mathbf{B}\times\nabla B)$ | 반대 부호 | $\propto m$ | 불균일한 $\|\mathbf{B}\|$ |
| **곡률** | $\frac{mv_\parallel^2}{qB^2R_c^2}(\mathbf{R}_c\times\mathbf{B})$ | 반대 부호 | $\propto m$ | 곡선 자기력선 |
| **결합 GC** | $\frac{m}{qB^2}(\frac{v_\perp^2}{2}+v_\parallel^2)\frac{\mathbf{B}\times\nabla B}{B}$ | 반대 부호 | $\propto m$ | Grad-B + 곡률 |
| **편극** | $\frac{m}{qB^2}\frac{d\mathbf{E}_\perp}{dt}$ | 반대 부호 | $\propto m$ | 시간 변동 E |
| **중력** | $\frac{m}{q}\frac{\mathbf{g}\times\mathbf{B}}{B^2}$ | 반대 부호 | $\propto m$ | 중력 |

### 6.2 주요 관찰

1. **전하 독립적 드리프트**: $\mathbf{E}\times\mathbf{B}$ 드리프트만 (전류 없음)
2. **전하 의존적 드리프트**: 다른 모든 것 (전류 생성)
3. **질량 독립적 드리프트**: $\mathbf{E}\times\mathbf{B}$만
4. **질량 비례 드리프트**: Grad-B, 곡률, 편극, 중력

이러한 드리프트의 조합이 다음을 결정합니다:
- 자기 트랩에서의 입자 구속
- 교차장 수송
- 플라즈마에서의 전류 생성
- 안정성 특성

### 6.3 드리프트 속도의 순서

일반적으로 자기화된 플라즈마에서:

$$
v_\parallel \sim v_{th} \gg v_E \sim \frac{E}{B}v_{th} \gg v_{\nabla B} \sim \frac{r_L}{L}v_{th} \gg v_P
$$

여기서 $L$은 기울기 척도 길이입니다. 계층은 다음에 의존합니다:
- $E/B$ 비율
- 기울기 척도 $L/r_L$
- 시간 변화율 $\omega/\omega_c$

## 7. Python 구현

### 7.1 Grad-B 드리프트 시뮬레이션

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Physical constants
m_p = 1.67e-27  # proton mass (kg)
m_e = 9.11e-31  # electron mass (kg)
q_p = 1.6e-19   # proton charge (C)
q_e = -1.6e-19  # electron charge (C)

def magnetic_field_gradient(x, y, z):
    """
    Non-uniform magnetic field: B = B0(1 + alpha*x)*z_hat
    Creates a gradient in x-direction
    """
    B0 = 1e-3  # Tesla
    alpha = 0.1  # gradient parameter (1/m)

    Bx = 0
    By = 0
    Bz = B0 * (1 + alpha * x)

    return np.array([Bx, By, Bz])

def equations_of_motion_gradb(t, state, q, m):
    """
    Equations of motion in non-uniform B field
    state = [x, y, z, vx, vy, vz]
    """
    x, y, z, vx, vy, vz = state

    # Magnetic field at current position
    B = magnetic_field_gradient(x, y, z)

    # Lorentz force: F = q(v × B)
    v = np.array([vx, vy, vz])
    F = q * np.cross(v, B)

    # Acceleration
    ax, ay, az = F / m

    return np.array([vx, vy, vz, ax, ay, az])

def rk4_step(f, t, y, dt, q, m):
    """4th-order Runge-Kutta step"""
    k1 = f(t, y, q, m)
    k2 = f(t + dt/2, y + dt*k1/2, q, m)
    k3 = f(t + dt/2, y + dt*k2/2, q, m)
    k4 = f(t + dt, y + dt*k3, q, m)
    return y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

def simulate_gradb_drift(particle_type='proton', v_perp=1e5, v_para=1e4,
                         duration=1e-3, dt=1e-8):
    """
    Simulate particle motion in grad-B field
    """
    # Particle properties
    if particle_type == 'proton':
        q, m = q_p, m_p
    else:
        q, m = q_e, m_e

    # Initial conditions
    x0, y0, z0 = 0.0, 0.0, 0.0
    vx0, vy0, vz0 = v_perp, 0.0, v_para
    state = np.array([x0, y0, z0, vx0, vy0, vz0])

    # Time array
    num_steps = int(duration / dt)
    times = np.linspace(0, duration, num_steps)

    # Storage
    trajectory = np.zeros((num_steps, 6))
    trajectory[0] = state

    # Integration
    for i in range(1, num_steps):
        state = rk4_step(equations_of_motion_gradb, times[i-1], state, dt, q, m)
        trajectory[i] = state

    return times, trajectory

# Simulate proton and electron
print("Simulating grad-B drift...")
t_p, traj_p = simulate_gradb_drift('proton', v_perp=5e4, v_para=1e4,
                                    duration=5e-4, dt=1e-8)
t_e, traj_e = simulate_gradb_drift('electron', v_perp=5e6, v_para=1e6,
                                    duration=5e-7, dt=1e-11)

# Plotting
fig = plt.figure(figsize=(15, 5))

# 3D trajectory - Proton
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot(traj_p[:, 0], traj_p[:, 1], traj_p[:, 2], 'b-', linewidth=0.5)
ax1.scatter([traj_p[0, 0]], [traj_p[0, 1]], [traj_p[0, 2]],
            color='green', s=50, label='Start')
ax1.scatter([traj_p[-1, 0]], [traj_p[-1, 1]], [traj_p[-1, 2]],
            color='red', s=50, label='End')
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
ax1.set_zlabel('z (m)')
ax1.set_title('Proton Trajectory (Grad-B Drift)')
ax1.legend()
ax1.grid(True)

# 3D trajectory - Electron
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot(traj_e[:, 0], traj_e[:, 1], traj_e[:, 2], 'r-', linewidth=0.5)
ax2.scatter([traj_e[0, 0]], [traj_e[0, 1]], [traj_e[0, 2]],
            color='green', s=50, label='Start')
ax2.scatter([traj_e[-1, 0]], [traj_e[-1, 1]], [traj_e[-1, 2]],
            color='red', s=50, label='End')
ax2.set_xlabel('x (m)')
ax2.set_ylabel('y (m)')
ax2.set_zlabel('z (m)')
ax2.set_title('Electron Trajectory (Grad-B Drift)')
ax2.legend()
ax2.grid(True)

# XY projection showing drift
ax3 = fig.add_subplot(133)
ax3.plot(traj_p[:, 0], traj_p[:, 1], 'b-', linewidth=1, label='Proton')
ax3.plot(traj_e[:, 0]*1e3, traj_e[:, 1]*1e3, 'r-', linewidth=1, label='Electron (scaled 1000x)')
ax3.set_xlabel('x (m)')
ax3.set_ylabel('y (m)')
ax3.set_title('XY Projection: Opposite Drift Directions')
ax3.legend()
ax3.grid(True)
ax3.axis('equal')

plt.tight_layout()
plt.savefig('gradb_drift_simulation.png', dpi=150)
print("Saved: gradb_drift_simulation.png")

# Calculate drift velocity
y_drift_p = traj_p[-1, 1] - traj_p[0, 1]
y_drift_e = traj_e[-1, 1] - traj_e[0, 1]
v_drift_p = y_drift_p / t_p[-1]
v_drift_e = y_drift_e / t_e[-1]

print(f"\nProton drift velocity: {v_drift_p:.2e} m/s (y-direction)")
print(f"Electron drift velocity: {v_drift_e:.2e} m/s (y-direction)")
print(f"Opposite directions: {np.sign(v_drift_p) != np.sign(v_drift_e)}")
```

### 7.2 쌍극자 장에서의 곡률 드리프트

```python
def magnetic_field_dipole(r, theta, phi, M=1e15):
    """
    Dipole magnetic field in spherical coordinates
    B_r = (2M/r^3) cos(theta)
    B_theta = (M/r^3) sin(theta)
    B_phi = 0
    M: magnetic moment (A·m^2)
    """
    r_safe = max(r, 0.1)  # avoid singularity

    B_r = (2 * M / r_safe**3) * np.cos(theta)
    B_theta = (M / r_safe**3) * np.sin(theta)
    B_phi = 0.0

    return np.array([B_r, B_theta, B_phi])

def spherical_to_cartesian_vector(v_sph, theta, phi):
    """Convert vector from spherical to Cartesian coordinates"""
    v_r, v_theta, v_phi = v_sph

    # Transformation matrix
    v_x = v_r * np.sin(theta) * np.cos(phi) + v_theta * np.cos(theta) * np.cos(phi) - v_phi * np.sin(phi)
    v_y = v_r * np.sin(theta) * np.sin(phi) + v_theta * np.cos(theta) * np.sin(phi) + v_phi * np.cos(phi)
    v_z = v_r * np.cos(theta) - v_theta * np.sin(theta)

    return np.array([v_x, v_y, v_z])

def cartesian_to_spherical(x, y, z):
    """Convert Cartesian to spherical coordinates"""
    r = np.sqrt(x**2 + y**2 + z**2)
    r = max(r, 1e-10)
    theta = np.arccos(np.clip(z / r, -1, 1))
    phi = np.arctan2(y, x)
    return r, theta, phi

def equations_of_motion_dipole(t, state, q, m, M):
    """
    Equations of motion in dipole field (Cartesian coordinates)
    """
    x, y, z, vx, vy, vz = state

    # Convert position to spherical
    r, theta, phi = cartesian_to_spherical(x, y, z)

    # Magnetic field in spherical coordinates
    B_sph = magnetic_field_dipole(r, theta, phi, M)

    # Convert B to Cartesian
    B = spherical_to_cartesian_vector(B_sph, theta, phi)

    # Lorentz force
    v = np.array([vx, vy, vz])
    F = q * np.cross(v, B)

    # Acceleration
    ax, ay, az = F / m

    return np.array([vx, vy, vz, ax, ay, az])

def simulate_dipole_drift(particle_type='proton', r0=1e6, theta0=np.pi/4,
                         v_perp=1e5, v_para=5e4, duration=10.0, dt=1e-3):
    """
    Simulate particle in dipole field (e.g., Earth's magnetosphere)
    """
    # Particle properties
    if particle_type == 'proton':
        q, m = q_p, m_p
    else:
        q, m = q_e, m_e

    # Earth's magnetic moment
    M = 8e15  # A·m^2 (approximate)

    # Initial position (spherical)
    phi0 = 0.0
    x0 = r0 * np.sin(theta0) * np.cos(phi0)
    y0 = r0 * np.sin(theta0) * np.sin(phi0)
    z0 = r0 * np.cos(theta0)

    # Initial velocity (perpendicular and parallel to B)
    B_sph = magnetic_field_dipole(r0, theta0, phi0, M)
    B_cart = spherical_to_cartesian_vector(B_sph, theta0, phi0)
    B_mag = np.linalg.norm(B_cart)
    b_hat = B_cart / B_mag

    # Perpendicular direction
    perp1 = np.array([1, 0, 0]) - np.dot([1, 0, 0], b_hat) * b_hat
    perp1 /= np.linalg.norm(perp1)

    v0 = v_para * b_hat + v_perp * perp1
    vx0, vy0, vz0 = v0

    state = np.array([x0, y0, z0, vx0, vy0, vz0])

    # Time array
    num_steps = int(duration / dt)
    times = np.linspace(0, duration, num_steps)

    # Storage
    trajectory = np.zeros((num_steps, 6))
    trajectory[0] = state

    # Integration
    for i in range(1, num_steps):
        state = rk4_step(lambda t, s, q, m: equations_of_motion_dipole(t, s, q, m, M),
                        times[i-1], state, dt, q, m)
        trajectory[i] = state

    return times, trajectory

# Simulate proton in dipole field
print("\nSimulating curvature drift in dipole field...")
t_dip, traj_dip = simulate_dipole_drift('proton', r0=2e6, theta0=np.pi/3,
                                        v_perp=5e4, v_para=3e4,
                                        duration=100.0, dt=0.01)

# Plot
fig = plt.figure(figsize=(12, 10))

# 3D trajectory
ax1 = fig.add_subplot(221, projection='3d')
ax1.plot(traj_dip[:, 0]/1e6, traj_dip[:, 1]/1e6, traj_dip[:, 2]/1e6,
         'b-', linewidth=0.8)
ax1.scatter([0], [0], [0], color='cyan', s=200, marker='o', label='Earth')
ax1.set_xlabel('x (Mm)')
ax1.set_ylabel('y (Mm)')
ax1.set_zlabel('z (Mm)')
ax1.set_title('Proton Trajectory in Dipole Field')
ax1.legend()
ax1.grid(True)

# XY projection
ax2 = fig.add_subplot(222)
ax2.plot(traj_dip[:, 0]/1e6, traj_dip[:, 1]/1e6, 'b-', linewidth=0.8)
ax2.scatter([0], [0], color='cyan', s=200, marker='o', label='Earth')
ax2.set_xlabel('x (Mm)')
ax2.set_ylabel('y (Mm)')
ax2.set_title('XY Projection')
ax2.axis('equal')
ax2.grid(True)
ax2.legend()

# XZ projection
ax3 = fig.add_subplot(223)
ax3.plot(traj_dip[:, 0]/1e6, traj_dip[:, 2]/1e6, 'b-', linewidth=0.8)
ax3.scatter([0], [0], color='cyan', s=200, marker='o', label='Earth')
ax3.set_xlabel('x (Mm)')
ax3.set_ylabel('z (Mm)')
ax3.set_title('XZ Projection (Meridional Plane)')
ax3.axis('equal')
ax3.grid(True)
ax3.legend()

# Radial distance vs time
r_traj = np.sqrt(traj_dip[:, 0]**2 + traj_dip[:, 1]**2 + traj_dip[:, 2]**2)
ax4 = fig.add_subplot(224)
ax4.plot(t_dip, r_traj/1e6, 'b-', linewidth=1)
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Radial Distance (Mm)')
ax4.set_title('Radial Distance vs Time')
ax4.grid(True)

plt.tight_layout()
plt.savefig('curvature_drift_dipole.png', dpi=150)
print("Saved: curvature_drift_dipole.png")
```

### 7.3 모든 드리프트 비교

```python
def compute_drift_velocities(B=1e-3, E=1e-2, grad_B=1e-5, R_c=1.0,
                            dE_dt=1.0, g=9.8, v_perp=1e5, v_para=1e5,
                            particle='proton'):
    """
    Compute magnitudes of all drift velocities for comparison
    """
    if particle == 'proton':
        q, m = q_p, m_p
    else:
        q, m = q_e, m_e

    # E×B drift
    v_ExB = E / B

    # Grad-B drift
    v_gradB = (m * v_perp**2) / (2 * abs(q) * B**2) * (grad_B / B)

    # Curvature drift
    v_curv = (m * v_para**2) / (abs(q) * B**2 * R_c)

    # Combined grad-B + curvature
    v_gc = (m / (abs(q) * B**2)) * (v_perp**2/2 + v_para**2) * (grad_B / B)

    # Polarization drift
    v_pol = (m / (abs(q) * B**2)) * dE_dt

    # Gravitational drift
    v_grav = (m * g) / (abs(q) * B)

    return {
        'E×B': v_ExB,
        'Grad-B': v_gradB,
        'Curvature': v_curv,
        'GC (combined)': v_gc,
        'Polarization': v_pol,
        'Gravitational': v_grav,
        'Parallel': v_para,
        'Perpendicular': v_perp
    }

# Compute for typical tokamak parameters
print("\n=== Drift Velocity Comparison ===")
print("\nTokamak Parameters:")
print("B = 2 T, E = 1 kV/m, ∇B/B = 0.1 m⁻¹, Rc = 3 m")
print("v_perp = v_para = 1e5 m/s")

drifts_p = compute_drift_velocities(B=2.0, E=1e3, grad_B=0.2, R_c=3.0,
                                    dE_dt=1e4, g=9.8,
                                    v_perp=1e5, v_para=1e5,
                                    particle='proton')

drifts_e = compute_drift_velocities(B=2.0, E=1e3, grad_B=0.2, R_c=3.0,
                                    dE_dt=1e4, g=9.8,
                                    v_perp=1e5, v_para=1e5,
                                    particle='electron')

print("\nProton drifts:")
for name, value in drifts_p.items():
    if name not in ['Parallel', 'Perpendicular']:
        print(f"  {name:20s}: {value:12.3e} m/s")

print("\nElectron drifts:")
for name, value in drifts_e.items():
    if name not in ['Parallel', 'Perpendicular']:
        print(f"  {name:20s}: {value:12.3e} m/s")

# Visualization
drift_names = ['E×B', 'Grad-B', 'Curv.', 'GC', 'Polar.', 'Grav.']
drift_values_p = [drifts_p['E×B'], drifts_p['Grad-B'], drifts_p['Curvature'],
                  drifts_p['GC (combined)'], drifts_p['Polarization'],
                  drifts_p['Gravitational']]
drift_values_e = [drifts_e['E×B'], drifts_e['Grad-B'], drifts_e['Curvature'],
                  drifts_e['GC (combined)'], drifts_e['Polarization'],
                  drifts_e['Gravitational']]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Proton drifts
x_pos = np.arange(len(drift_names))
bars1 = ax1.bar(x_pos, drift_values_p, color='blue', alpha=0.7)
ax1.set_yscale('log')
ax1.set_ylabel('Drift Velocity (m/s)', fontsize=12)
ax1.set_xlabel('Drift Type', fontsize=12)
ax1.set_title('Proton Drift Velocities (Tokamak)', fontsize=14, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(drift_names, rotation=45, ha='right')
ax1.grid(True, alpha=0.3, which='both')
ax1.axhline(y=drifts_p['Parallel'], color='red', linestyle='--',
            linewidth=2, label=f"v_parallel = {drifts_p['Parallel']:.1e} m/s")
ax1.legend()

# Add values on bars
for i, (bar, val) in enumerate(zip(bars1, drift_values_p)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1e}', ha='center', va='bottom', fontsize=8, rotation=0)

# Electron drifts
bars2 = ax2.bar(x_pos, drift_values_e, color='red', alpha=0.7)
ax2.set_yscale('log')
ax2.set_ylabel('Drift Velocity (m/s)', fontsize=12)
ax2.set_xlabel('Drift Type', fontsize=12)
ax2.set_title('Electron Drift Velocities (Tokamak)', fontsize=14, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(drift_names, rotation=45, ha='right')
ax2.grid(True, alpha=0.3, which='both')
ax2.axhline(y=drifts_e['Parallel'], color='blue', linestyle='--',
            linewidth=2, label=f"v_parallel = {drifts_e['Parallel']:.1e} m/s")
ax2.legend()

# Add values on bars
for i, (bar, val) in enumerate(zip(bars2, drift_values_e)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1e}', ha='center', va='bottom', fontsize=8, rotation=0)

plt.tight_layout()
plt.savefig('drift_comparison.png', dpi=150)
print("\nSaved: drift_comparison.png")
```

### 7.4 드리프트 전류 시각화

```python
def calculate_drift_currents(n=1e19, B=2.0, T=1e3, grad_B=0.2,
                             E_field=1e3, dE_dt=1e4, g=9.8):
    """
    Calculate current densities from different drifts
    n: plasma density (m^-3)
    T: temperature (eV)
    B: magnetic field (T)
    """
    # Convert temperature to SI
    T_J = T * q_p  # Joules

    # Thermal velocity
    v_th_p = np.sqrt(2 * T_J / m_p)
    v_th_e = np.sqrt(2 * T_J / m_e)

    # E×B drift (no current)
    J_ExB = 0.0

    # Grad-B current (diamagnetic)
    J_gradB = n * 2 * T_J / B**2 * grad_B

    # Polarization current (ions only)
    J_pol = n * m_p / B**2 * dE_dt

    # Gravitational current
    J_grav = n * m_p * g / B

    return {
        'E×B': J_ExB,
        'Grad-B': J_gradB,
        'Polarization': J_pol,
        'Gravitational': J_grav
    }

# Calculate currents
print("\n=== Drift Current Densities ===")
print("Plasma: n = 1e19 m^-3, T = 1 keV, B = 2 T")

currents = calculate_drift_currents(n=1e19, B=2.0, T=1e3, grad_B=0.2,
                                    E_field=1e3, dE_dt=1e4, g=9.8)

print("\nCurrent densities:")
for name, value in currents.items():
    print(f"  J_{name:20s}: {value:12.3e} A/m²")

# Visualization
fig, ax = plt.subplots(figsize=(10, 6))

current_names = list(currents.keys())
current_values = list(currents.values())
colors = ['gray', 'blue', 'orange', 'green']

bars = ax.bar(current_names, current_values, color=colors, alpha=0.7, edgecolor='black')

ax.set_ylabel('Current Density (A/m²)', fontsize=14, fontweight='bold')
ax.set_xlabel('Drift Type', fontsize=14, fontweight='bold')
ax.set_title('Current Densities from Different Drifts', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add values on bars
for bar, val in zip(bars, current_values):
    height = bar.get_height()
    if val != 0:
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.2e}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    else:
        ax.text(bar.get_x() + bar.get_width()/2., 0.1*max(current_values),
               'No current', ha='center', va='bottom', fontsize=10,
               fontweight='bold', style='italic')

plt.tight_layout()
plt.savefig('drift_currents.png', dpi=150)
print("\nSaved: drift_currents.png")
```

## 요약

이 레슨에서, 우리는 자기화된 플라즈마에서 다양한 드리프트 운동을 탐구했습니다:

1. **일반 드리프트 공식**: $\mathbf{v}_D = \mathbf{F}\times\mathbf{B}/(qB^2)$가 $\mathbf{B}$에 수직인 모든 힘에 적용됩니다.

2. **Grad-B 드리프트**: 불균일한 자기장 세기는 입자가 $\mathbf{B}$와 $\nabla B$ 모두에 수직으로 드리프트하게 합니다. 방향은 전하 부호에 의존 → 전류 생성.

3. **곡률 드리프트**: 곡선 자기력선에서의 원심력이 드리프트를 일으킵니다. grad-B 드리프트와 결합하여 총 유도 중심 드리프트를 제공합니다.

4. **편극 드리프트**: 시간 변동 전기장은 질량 의존적 드리프트를 일으킵니다. 이온에게 중요하고, 전자에게는 무시할 만함 → 편극 전류.

5. **중력 드리프트**: 중력(또는 모든 질량 비례 힘)은 전하 의존적 드리프트를 일으킵니다.

6. **드리프트 전류**: 대부분의 드리프트($\mathbf{E}\times\mathbf{B}$ 제외)는 이온과 전자가 다르게 드리프트하기 때문에 전류를 생성합니다.

7. **플라즈마 구속**: 이러한 드리프트를 이해하는 것은 다음에 중요합니다:
   - 자기 핵융합 (토카막, 스텔러레이터)
   - 우주 플라즈마 (자기권, 태양풍)
   - 천체물리학 플라즈마

시간 규모의 계층 — 자이로운동 ($\omega_c^{-1}$), 바운스 ($\omega_b^{-1}$), 드리프트 ($\omega_d^{-1}$) — 는 체계적인 섭동 이론과 유도 중심 근사를 허용합니다.

## 연습 문제

### 문제 1: 거울에서의 Grad-B 드리프트

에너지 $W = 10$ keV이고 (중앙면에서) 피치각 $\alpha = 60°$인 양성자가 $B_{min} = 0.5$ T이고 $B_{max} = 2$ T인 자기 거울에 구속되어 있습니다.

(a) 자기 모멘트 $\mu$를 계산하고 보존됨을 확인하세요.

(b) 기울기 척도 길이가 $L = \frac{B}{|\nabla B|} = 1$ m일 때 중앙면에서 grad-B 드리프트 속도를 계산하세요.

(c) 드리프트 궤도 둘레와 한 번의 완전한 드리프트 궤도에 대한 시간을 추정하세요.

(d) 드리프트 주기를 바운스 주기와 비교하세요.

**힌트**: $v_\perp = v\sin\alpha$, $v_\parallel = v\cos\alpha$, 그리고 거울 힘 $F_\parallel = -\mu\nabla_\parallel B$를 사용하세요.

---

### 문제 2: 토카막 수직 드리프트

주반경 $R_0 = 2$ m인 토카막에서, 환형 장은 $B_\phi(R) = B_0 R_0/R$이고 여기서 $B_0 = 3$ T입니다. 외부 중앙면($R = R_0 + a$, 여기서 $a = 0.5$ m는 소반경)에서 $v_\perp = 1\times 10^5$ m/s이고 $v_\parallel = 5\times 10^5$ m/s인 중수소 핵을 고려하세요.

(a) grad-B 드리프트 속도(수직 방향)를 계산하세요.

(b) 곡률 드리프트 속도를 계산하세요.

(c) 총 유도 중심 드리프트를 계산하고 개별 기여와 비교하세요.

(d) 환형 회전이 없는 간단한 모델에서, 이 수직 드리프트는 전하 분리와 수직 전기장을 일으킵니다. 이 장이 생성하는 $\mathbf{E}\times\mathbf{B}$ 드리프트(방사상 방향)를 추정하세요. 이것이 입자 손실 메커니즘인 이유는?

**힌트**: 환형 장에 대해, $\nabla B \approx -\frac{B}{R}\hat{R}$이고 $R_c \approx R_0$입니다.

---

### 문제 3: 파동에서의 편극 전류

플라즈마의 이온 음향파는 자기장 $\mathbf{B} = B_0\hat{z}$에서 전기장 $\mathbf{E} = E_0 \cos(kx - \omega t)\hat{y}$를 가집니다. 파동 주파수는 $\omega = 10^5$ rad/s, $E_0 = 100$ V/m, $B_0 = 1$ T이고, 이온 밀도는 $n_i = 10^{18}$ m$^{-3}$입니다.

(a) $\mathbf{E}\times\mathbf{B}$ 드리프트 속도(시간 의존적)를 계산하세요.

(b) 이온(중수소 핵)에 대한 편극 드리프트 속도를 계산하세요.

(c) 편극 전류 밀도 $\mathbf{J}_P$를 계산하세요.

(d) $\mathbf{v}_E$와 $\mathbf{v}_P$의 크기를 비교하세요. 편극 드리프트가 중요한 조건은?

**힌트**: $\frac{d\mathbf{E}}{dt} = -\frac{\partial \mathbf{E}}{\partial t} - \mathbf{v}\cdot\nabla\mathbf{E}$. 느린 드리프트에 대해, $\frac{d\mathbf{E}}{dt} \approx -\frac{\partial \mathbf{E}}{\partial t}$로 근사하세요.

---

### 문제 4: 손실 원뿔과 Grad-B 드리프트

자기 거울 장치에서, 작은 피치각을 가진 입자는 손실 원뿔을 통해 탈출합니다. Grad-B 드리프트는 입자가 축 주위로 방위각 방향으로 드리프트하게 합니다.

(a) 원형 대칭(축대칭)을 가진 거울에 대해, grad-B 드리프트가 입자 손실을 일으키지 않는 이유를 설명하세요.

(b) $\mathbf{B}$가 국소 최소값(사중극 장)을 가진 "minimum-B" 구성을 고려하세요. grad-B 드리프트 방향이 간단한 거울에 비해 반전됨을 보이세요.

(c) 실제 거울에서, 비대칭성이 대칭을 깹니다. 장이 작은 비축대칭 성분 $\delta B/B \sim 0.01$를 가지면, 이 비대칭으로 인해 입자가 손실 원뿔로 드리프트하는 시간을 추정하세요.

**힌트**: minimum-B에서, $\nabla B$는 최소값으로부터 바깥쪽을 가리키며, 거울 최대값과 반대입니다.

---

### 문제 5: 태양 홍염에서의 중력 침강

태양 홍염은 자기장에 의해 뜨거운 코로나에 매달린 차갑고 밀도 높은 플라즈마 구조입니다. 홍염에서 양성자의 중력 드리프트를 추정하세요.

매개변수:
- 태양 표면 중력: $g = 274$ m/s²
- 자기장: $B = 5\times 10^{-3}$ T (전형적인 홍염 장)
- 장이 수평이고 중력이 수직(태양 표면을 향함)이라고 가정

(a) 양성자와 전자에 대한 중력 드리프트 속도를 계산하세요.

(b) 홍염 밀도가 $n = 10^{16}$ m$^{-3}$일 때 전류 밀도를 추정하세요.

(c) 이 전류는 자기장(Ampère의 법칙)을 만듭니다. 홍염 두께가 $L = 10^7$ m일 때 이 유도 장의 크기를 추정하세요.

(d) 이 자체 생성 장이 중력에 대해 홍염을 지지할 수 있는지 논의하세요 (자기 압력 $B^2/(2\mu_0)$를 중력 압력 $\rho g L$과 비교).

**힌트**: 중력 전류는 $\mathbf{J}_g \approx n m_p \mathbf{g}\times\mathbf{B}/B^2$입니다. Ampère의 법칙 $\nabla\times\mathbf{B} = \mu_0\mathbf{J}$를 사용하여 $\delta B \sim \mu_0 J L$을 추정하세요.

---

## 내비게이션

- **이전**: [단일 입자 운동 I](./04_Single_Particle_Motion_I.md)
- **다음**: [자기 거울과 단열 불변량](./06_Magnetic_Mirrors_Adiabatic_Invariants.md)
