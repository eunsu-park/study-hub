# 맥스웰 방정식 — 미분 형태

[← 이전: 06. 전자기 유도](06_Electromagnetic_Induction.md) | [다음: 08. 맥스웰 방정식 — 적분 형태 →](08_Maxwells_Equations_Integral.md)

---

## 학습 목표

1. 변위 전류(displacement current)의 개념과 맥스웰이 앙페르 법칙에 이 항을 추가한 이유를 설명한다
2. 네 개의 맥스웰 방정식을 미분 형태로 쓰고, 각각의 물리적 의미를 설명한다
3. 맥스웰 방정식으로부터 전자기파 방정식을 유도한다
4. $\epsilon_0$과 $\mu_0$으로부터 빛의 속도를 계산한다
5. 시간에 따라 변하는 경우에 대한 스칼라 퍼텐셜 $\phi$와 벡터 퍼텐셜 $\mathbf{A}$를 소개한다
6. 로렌츠 게이지(Lorenz gauge)와 파동 전파에서 그 장점을 설명한다
7. 유한 차분법(finite differences)으로 파동 방정식을 수치적으로 시연한다

---

이 레슨은 고전 전자기학의 정점이다. 앞선 레슨들에서 발전시킨 네 가지 법칙 — 가우스 법칙, 자기 단극자 없음 조건, 패러데이 법칙, 앙페르 법칙 — 을 하나의 통일된 체계인 맥스웰 방정식으로 집대성한다. 그런데 여기에 반전이 있다. 정자기학에서 서술된 앙페르 법칙은 전하 보존과 모순된다. 맥스웰의 천재성은 이 모순을 발견하고 **변위 전류**를 추가하여 해결한 것이다. 수정된 방정식은 놀라운 결과를 예측한다: 바로 빛의 속도로 전파하는 전자기파이다. 빛 자체가 전자기파인 것이다. 이 단 하나의 추론으로 광학, 전기학, 자기학이 하나의 이론으로 통합된다.

---

## 앙페르 법칙의 모순

정자기학에서의 앙페르 법칙은 다음과 같다:

$$\nabla \times \mathbf{B} = \mu_0 \mathbf{J}$$

양변에 발산을 취하면:

$$\nabla \cdot (\nabla \times \mathbf{B}) = \mu_0 \nabla \cdot \mathbf{J}$$

좌변은 항등적으로 영이다(회전의 발산은 항상 0이다). 따라서 다음이 요구된다:

$$\nabla \cdot \mathbf{J} = 0$$

그런데 연속 방정식(전하 보존)은 이렇게 말한다:

$$\nabla \cdot \mathbf{J} = -\frac{\partial \rho}{\partial t}$$

정자기학($\partial \rho/\partial t = 0$)에서는 문제없다. 하지만 시간에 따라 변하는 장에서는 일반적으로 $\nabla \cdot \mathbf{J} \neq 0$이다. 가장 전형적인 예는 충전 중인 축전기(capacitor)이다: 전류가 한 판으로 흘러 들어가고 다른 판에서 나오지만, 두 판 사이에는 전류가 흐르지 않는다. 그럼에도 판 사이에 자기장 B가 존재해야 한다.

**앙페르 법칙이 틀렸다!** (시간에 따라 변하는 장의 경우에.)

---

## 변위 전류

맥스웰의 해결책: 앙페르 법칙에 일관성을 회복하는 항을 추가한다. 가우스 법칙으로부터:

$$\nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0} \implies \frac{\partial \rho}{\partial t} = \epsilon_0 \frac{\partial}{\partial t}(\nabla \cdot \mathbf{E}) = \epsilon_0 \nabla \cdot \frac{\partial \mathbf{E}}{\partial t}$$

따라서 연속 방정식은:

$$\nabla \cdot \mathbf{J} + \epsilon_0 \nabla \cdot \frac{\partial \mathbf{E}}{\partial t} = 0 \implies \nabla \cdot \left(\mathbf{J} + \epsilon_0 \frac{\partial \mathbf{E}}{\partial t}\right) = 0$$

수정된 앙페르 법칙:

$$\boxed{\nabla \times \mathbf{B} = \mu_0 \mathbf{J} + \mu_0\epsilon_0 \frac{\partial \mathbf{E}}{\partial t}}$$

여기서 $\mathbf{J}_d = \epsilon_0 \frac{\partial \mathbf{E}}{\partial t}$를 **변위 전류 밀도(displacement current density)**라고 한다. 이것은 실제 전하의 흐름이 아니라, 실제 전류와 마찬가지로 자기장을 생성하는 전기장의 시간 변화이다.

> **비유**: 이어달리기를 생각해보자. 주자들(실제 전류 $\mathbf{J}$)이 구간마다 바통을 넘긴다. 인계 구간(충전 중인 축전기의 간격)에는 주자가 없지만, 바통(전자기적 효과)은 여전히 변하는 전기장을 통해 전달된다. 변위 전류는 공백 없이 이어달리기를 계속시키는 이 "유령 주자"이다.

```python
import numpy as np
import matplotlib.pyplot as plt

# Displacement current in a charging parallel-plate capacitor
# Why this example: it's the canonical case that motivated Maxwell's correction

epsilon_0 = 8.854e-12
mu_0 = 4 * np.pi * 1e-7

# Capacitor parameters
A_plate = 0.01       # plate area (100 cm²)
d = 0.002            # plate separation (2 mm)
C = epsilon_0 * A_plate / d   # capacitance
R = 1000             # charging resistance (Ω)
V0 = 10.0            # source voltage (V)
tau = R * C           # RC time constant

t = np.linspace(0, 5 * tau, 1000)

# Charging current
I_real = (V0 / R) * np.exp(-t / tau)

# Electric field between plates: E = V_cap/(d) = (V₀/d)(1-e^(-t/τ))
E = (V0 / d) * (1 - np.exp(-t / tau))

# Displacement current density: J_d = ε₀ ∂E/∂t
# Why ε₀ ∂E/∂t: this is Maxwell's displacement current
dE_dt = (V0 / d) * (1 / tau) * np.exp(-t / tau)
J_d = epsilon_0 * dE_dt

# Total displacement current through the capacitor gap = J_d × A
I_displacement = J_d * A_plate

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Real current in the wire
axes[0].plot(t / tau, I_real * 1e6, 'b-', linewidth=2)
axes[0].set_xlabel('t / τ')
axes[0].set_ylabel('I (μA)')
axes[0].set_title('Real Current in Wire')
axes[0].grid(True, alpha=0.3)

# Displacement current in the gap
axes[1].plot(t / tau, I_displacement * 1e6, 'r-', linewidth=2)
axes[1].set_xlabel('t / τ')
axes[1].set_ylabel('$I_d$ (μA)')
axes[1].set_title('Displacement Current in Gap')
axes[1].grid(True, alpha=0.3)

# Compare: they should be identical!
axes[2].plot(t / tau, I_real * 1e6, 'b-', linewidth=2, label='Real current $I$')
axes[2].plot(t / tau, I_displacement * 1e6, 'r--', linewidth=2, label='Displacement current $I_d$')
axes[2].set_xlabel('t / τ')
axes[2].set_ylabel('Current (μA)')
axes[2].set_title('$I_{real} = I_{displacement}$ (Continuity!)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.suptitle(f'Displacement Current in Charging Capacitor (τ = {tau*1e9:.1f} ns)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('displacement_current.png', dpi=150)
plt.show()

print(f"RC time constant: τ = {tau*1e9:.2f} ns")
print(f"Max real current: I₀ = {V0/R*1e6:.1f} μA")
print(f"Max displacement current: I_d = {epsilon_0 * V0/(d*tau) * A_plate * 1e6:.1f} μA")
```

---

## 맥스웰 방정식 — 완전한 형태

진공에서의 맥스웰 방정식 4개(미분 형태):

$$\boxed{
\begin{aligned}
(i) \quad & \nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0} && \text{(가우스 법칙)} \\[8pt]
(ii) \quad & \nabla \cdot \mathbf{B} = 0 && \text{(자기 단극자 없음)} \\[8pt]
(iii) \quad & \nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t} && \text{(패러데이 법칙)} \\[8pt]
(iv) \quad & \nabla \times \mathbf{B} = \mu_0 \mathbf{J} + \mu_0\epsilon_0\frac{\partial \mathbf{E}}{\partial t} && \text{(앙페르-맥스웰 법칙)}
\end{aligned}
}$$

### 물리적 해석

| 방정식 | 의미 |
|---|---|
| (i) 가우스 법칙 | 전기력선은 전하에서 시작하거나 끝난다 |
| (ii) 자기 단극자 없음 | 자기력선은 항상 닫힌 곡선을 이룬다 |
| (iii) 패러데이 법칙 | 자기장의 변화가 순환하는 전기장을 만든다 |
| (iv) 앙페르-맥스웰 법칙 | 전류와 전기장의 변화가 순환하는 자기장을 만든다 |

### 아름다운 대칭성

방정식 (iii)과 (iv)는 놀라운 상호성을 드러낸다:
- $\mathbf{B}$의 변화가 $\mathbf{E}$를 만든다 (패러데이)
- $\mathbf{E}$의 변화가 $\mathbf{B}$를 만든다 (변위 전류)

이 상호 생성이 전자기파가 전파하는 메커니즘이다: 진동하는 $\mathbf{E}$ 장이 진동하는 $\mathbf{B}$ 장을 만들고, 그 $\mathbf{B}$ 장이 다시 진동하는 $\mathbf{E}$ 장을 만들어, 빛의 속도로 진공을 통해 서로를 부트스트랩한다.

### 물질 내에서

선형(linear), 등방성(isotropic), 균질(homogeneous) 매질에서:

$$\nabla \cdot \mathbf{D} = \rho_f, \quad \nabla \cdot \mathbf{B} = 0, \quad \nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}, \quad \nabla \times \mathbf{H} = \mathbf{J}_f + \frac{\partial \mathbf{D}}{\partial t}$$

여기서 $\mathbf{D} = \epsilon\mathbf{E}$, $\mathbf{H} = \mathbf{B}/\mu$이다.

---

## 파동 방정식 유도

전하와 전류가 없는 진공($\rho = 0$, $\mathbf{J} = 0$)에서 맥스웰 방정식은:

$$\nabla \cdot \mathbf{E} = 0, \quad \nabla \cdot \mathbf{B} = 0$$
$$\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}, \quad \nabla \times \mathbf{B} = \mu_0\epsilon_0\frac{\partial \mathbf{E}}{\partial t}$$

패러데이 법칙의 양변에 회전을 취한다:

$$\nabla \times (\nabla \times \mathbf{E}) = -\frac{\partial}{\partial t}(\nabla \times \mathbf{B}) = -\mu_0\epsilon_0\frac{\partial^2 \mathbf{E}}{\partial t^2}$$

벡터 항등식 $\nabla \times (\nabla \times \mathbf{E}) = \nabla(\nabla \cdot \mathbf{E}) - \nabla^2\mathbf{E}$와 $\nabla \cdot \mathbf{E} = 0$을 이용하면:

$$\boxed{\nabla^2\mathbf{E} = \mu_0\epsilon_0\frac{\partial^2\mathbf{E}}{\partial t^2}}$$

마찬가지로 $\mathbf{B}$에 대해서도:

$$\boxed{\nabla^2\mathbf{B} = \mu_0\epsilon_0\frac{\partial^2\mathbf{B}}{\partial t^2}}$$

이것이 바로 **파동 방정식**이다! 표준 파동 방정식 $\nabla^2 f = \frac{1}{v^2}\frac{\partial^2 f}{\partial t^2}$와 비교하면:

$$v = \frac{1}{\sqrt{\mu_0\epsilon_0}} = \frac{1}{\sqrt{(4\pi\times10^{-7})(8.854\times10^{-12})}} = 2.998 \times 10^8 \text{ m/s}$$

이것이 **빛의 속도**이다! 맥스웰은 1864년에 이렇게 썼다: "이 속도가 빛의 속도와 매우 가까우므로, 빛 자체가 전자기 교란임을 결론 내릴 강력한 근거가 있는 것으로 보인다."

```python
import numpy as np

# Calculate the speed of light from ε₀ and μ₀
# Why from first principles: this was Maxwell's greatest triumph

mu_0 = 4 * np.pi * 1e-7     # permeability of free space (T·m/A)
epsilon_0 = 8.854187817e-12  # permittivity of free space (F/m)

c_calculated = 1 / np.sqrt(mu_0 * epsilon_0)
c_measured = 299_792_458     # exact (by definition since 1983), m/s

print("Speed of Light from Maxwell's Equations")
print("=" * 50)
print(f"μ₀ = {mu_0:.10e} T·m/A")
print(f"ε₀ = {epsilon_0:.10e} F/m")
print(f"")
print(f"c = 1/√(μ₀ε₀) = {c_calculated:.6f} m/s")
print(f"c (measured)    = {c_measured} m/s")
print(f"Relative error  = {abs(c_calculated - c_measured)/c_measured:.2e}")
print(f"")
print(f"This agreement was the smoking gun that light is an EM wave!")
```

---

## 시간 의존 장의 게이지 퍼텐셜

시간 의존 경우에도 $\nabla \cdot \mathbf{B} = 0$이 성립하므로:

$$\mathbf{B} = \nabla \times \mathbf{A}$$

이제 패러데이 법칙은 $\nabla \times \mathbf{E} = -\partial\mathbf{B}/\partial t = -\nabla \times (\partial\mathbf{A}/\partial t)$를 주므로:

$$\nabla \times \left(\mathbf{E} + \frac{\partial \mathbf{A}}{\partial t}\right) = 0$$

회전이 0인 벡터장은 어떤 스칼라 함수의 기울기(gradient)이므로:

$$\mathbf{E} + \frac{\partial \mathbf{A}}{\partial t} = -\nabla V$$

따라서:

$$\boxed{\mathbf{E} = -\nabla V - \frac{\partial \mathbf{A}}{\partial t}}$$

$$\boxed{\mathbf{B} = \nabla \times \mathbf{A}}$$

정전기학($\partial\mathbf{A}/\partial t = 0$)에서는 $\mathbf{E} = -\nabla V$로 환원된다.

### 게이지 자유도(Gauge Freedom) — 시간 의존 경우

게이지 변환은 다음과 같이 일반화된다:

$$\mathbf{A} \to \mathbf{A}' = \mathbf{A} + \nabla\lambda, \qquad V \to V' = V - \frac{\partial\lambda}{\partial t}$$

$\mathbf{E}$와 $\mathbf{B}$ 모두 이 변환 아래에서 불변이다.

---

## 로렌츠 게이지(Lorenz Gauge)

퍼텐셜을 맥스웰 방정식에 대입하면 $V$와 $\mathbf{A}$에 대한 연립 방정식이 얻어진다. **로렌츠 게이지**는 이를 분리(decouple)한다:

$$\boxed{\nabla \cdot \mathbf{A} + \mu_0\epsilon_0 \frac{\partial V}{\partial t} = 0} \qquad \text{(로렌츠 게이지 조건)}$$

이 선택 하에서 $V$와 $\mathbf{A}$에 대한 방정식은 네 개의 독립적인 파동 방정식이 된다:

$$\nabla^2 V - \mu_0\epsilon_0 \frac{\partial^2 V}{\partial t^2} = -\frac{\rho}{\epsilon_0}$$

$$\nabla^2 \mathbf{A} - \mu_0\epsilon_0 \frac{\partial^2 \mathbf{A}}{\partial t^2} = -\mu_0 \mathbf{J}$$

달랑베르 연산자(d'Alembertian operator) $\Box^2 \equiv \nabla^2 - \mu_0\epsilon_0\frac{\partial^2}{\partial t^2}$를 이용하면:

$$\Box^2 V = -\rho/\epsilon_0, \qquad \Box^2 \mathbf{A} = -\mu_0\mathbf{J}$$

아름다운 결과이다: 각 퍼텐셜이 자신의 소스만으로 구동되는 파동 방정식을 만족한다.

> **비유**: 로렌츠 게이지는 운동 방정식을 분리하는 좌표를 선택하는 것과 같다. 고전 역학에서 정규 모드(normal modes)가 결합된 진동자를 분리하듯, 로렌츠 게이지는 전자기 퍼텐셜에 대해 같은 역할을 한다 — 각 퍼텐셜은 자신의 소스에 의해서만 구동되며 독립적으로 진화한다.

---

## 수치 시연: 1차원 파동 방정식

유한 차분법(finite differences)으로 1차원 전자기 파동 방정식을 풀 수 있다:

$$\frac{\partial^2 E}{\partial t^2} = c^2 \frac{\partial^2 E}{\partial x^2}$$

```python
import numpy as np
import matplotlib.pyplot as plt

# Solve the 1D electromagnetic wave equation using FDTD
# Why FDTD: it directly solves Maxwell's equations on a grid

# Domain
L = 10.0              # domain length (in arbitrary units where c = 1)
c = 1.0               # speed of light (normalized)
N = 500               # spatial grid points
dx = L / N
dt = 0.5 * dx / c     # CFL condition: dt < dx/c for stability
# Why CFL: the wave must not travel more than one cell per time step

x = np.linspace(0, L, N)
T_total = 8.0         # total simulation time
N_t = int(T_total / dt)

# Initialize E field: Gaussian pulse
# Why Gaussian: it's a localized disturbance that clearly shows wave propagation
sigma = 0.3
x_center = L / 2
E = np.exp(-((x - x_center) / sigma)**2)
E_prev = E.copy()     # E at previous time step (for leapfrog)

# Store snapshots for visualization
snapshots = []
snapshot_times = [0, 1, 2, 3, 4, 5, 6, 7]
snap_idx = 0

for n in range(N_t):
    t_now = n * dt

    # Save snapshots
    if snap_idx < len(snapshot_times) and t_now >= snapshot_times[snap_idx]:
        snapshots.append((t_now, E.copy()))
        snap_idx += 1

    # Finite difference update: E(t+dt) = 2E(t) - E(t-dt) + (c*dt/dx)² * [E(x+dx) - 2E(x) + E(x-dx)]
    # Why leapfrog: second-order accurate in both space and time
    r = (c * dt / dx)**2
    E_next = np.zeros_like(E)
    E_next[1:-1] = 2*E[1:-1] - E_prev[1:-1] + r * (E[2:] - 2*E[1:-1] + E[:-2])

    # Absorbing boundary conditions (first-order Mur)
    # Why absorbing: we want waves to leave the domain without reflection
    E_next[0] = E[1] + (c*dt - dx)/(c*dt + dx) * (E_next[1] - E[0])
    E_next[-1] = E[-2] + (c*dt - dx)/(c*dt + dx) * (E_next[-2] - E[-1])

    E_prev = E.copy()
    E = E_next.copy()

fig, axes = plt.subplots(4, 2, figsize=(14, 14))
axes_flat = axes.flatten()

for i, (t_snap, E_snap) in enumerate(snapshots):
    if i < len(axes_flat):
        axes_flat[i].plot(x, E_snap, 'b-', linewidth=1.5)
        axes_flat[i].set_ylim(-1.2, 1.2)
        axes_flat[i].set_xlabel('x')
        axes_flat[i].set_ylabel('E')
        axes_flat[i].set_title(f't = {t_snap:.1f}')
        axes_flat[i].grid(True, alpha=0.3)
        axes_flat[i].axhline(y=0, color='gray', linestyle='-', alpha=0.3)

plt.suptitle('1D EM Wave Propagation (FDTD)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('wave_equation_1d.png', dpi=150)
plt.show()
```

---

## 지연 퍼텐셜(Retarded Potentials)

로렌츠 게이지에서 퍼텐셜의 파동 방정식에 대한 해는 **지연 퍼텐셜**이다:

$$V(\mathbf{r}, t) = \frac{1}{4\pi\epsilon_0}\int \frac{\rho(\mathbf{r}', t_r)}{|\mathbf{r}-\mathbf{r}'|}\,d\tau'$$

$$\mathbf{A}(\mathbf{r}, t) = \frac{\mu_0}{4\pi}\int \frac{\mathbf{J}(\mathbf{r}', t_r)}{|\mathbf{r}-\mathbf{r}'|}\,d\tau'$$

여기서 $t_r = t - |\mathbf{r}-\mathbf{r}'|/c$는 **지연 시간(retarded time)**이다 — 속도 $c$로 이동하는 신호가 소스점 $\mathbf{r}'$을 출발하여 시각 $t$에 장점 $\mathbf{r}$에 도달하려면 언제 출발해야 하는가를 나타낸다.

지연 퍼텐셜은 **인과율(causality)**을 구현한다: $(\mathbf{r}, t)$에서의 장은 소스가 현재 무엇을 하고 있는지가 아니라, 과거 어느 시점에 무엇을 했는지에 달려 있다. 전자기 효과는 빛의 속도로 전파되며 — 즉각적으로 작용하지 않는다.

참고: $t_a = t + |\mathbf{r}-\mathbf{r}'|/c$를 사용하는 "앞선(advanced)" 퍼텐셜도 존재한다(미래의 소스가 현재의 장에 영향을 준다). 수학적으로는 유효하지만 인과율을 위반하므로 통상 버린다. 이들의 존재는 맥스웰 방정식의 시간 역전 대칭성을 반영한다.

```python
import numpy as np
import matplotlib.pyplot as plt

# Retarded potential: visualize the causal structure
# Why light cones: they show which source events can influence a given field point

c = 1.0  # normalized speed of light

fig, ax = plt.subplots(figsize=(8, 8))

# Draw the light cone at the observation point (x=0, t=5)
t_obs = 5.0
x_obs = 0.0

# Past light cone: events that can influence (x_obs, t_obs)
t_past = np.linspace(0, t_obs, 100)
# Why ±c(t_obs - t): the light cone boundary in 1+1D spacetime
x_left = x_obs - c * (t_obs - t_past)
x_right = x_obs + c * (t_obs - t_past)

ax.fill_betweenx(t_past, x_left, x_right, alpha=0.15, color='blue',
                  label='Past light cone (causal region)')
ax.plot(x_left, t_past, 'b-', linewidth=1.5)
ax.plot(x_right, t_past, 'b-', linewidth=1.5)
ax.plot(x_obs, t_obs, 'ro', markersize=10, zorder=5, label='Observation event')

# Source events
sources = [(-2, 2), (1, 3), (3, 1), (-4, 4.5)]
for sx, st in sources:
    # Check if inside past light cone
    inside = abs(sx - x_obs) <= c * (t_obs - st) and st < t_obs
    color = 'green' if inside else 'red'
    marker = 's' if inside else 'x'
    label = 'Can influence' if inside else 'Cannot influence'
    ax.plot(sx, st, marker, color=color, markersize=12, zorder=5)
    t_r = t_obs - abs(sx - x_obs) / c
    if inside:
        ax.annotate(f't_r = {t_r:.1f}', (sx, st), textcoords="offset points",
                   xytext=(10, 5), fontsize=9)

ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_title('Causal Structure: Retarded Potentials Use Past Light Cone')
ax.set_xlim(-6, 6)
ax.set_ylim(0, 6)
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('retarded_potentials.png', dpi=150)
plt.show()
```

---

## 전자기 쌍대성(Electromagnetic Duality)

전하가 없는 경우($\rho = 0$, $\mathbf{J} = 0$), 맥스웰 방정식은 $\mathbf{E}$와 $\mathbf{B}$ 사이의 놀라운 **쌍대성**을 보인다.

$\mathbf{E} \to c\mathbf{B}$, $c\mathbf{B} \to -\mathbf{E}$로 치환하면 소스가 없는 맥스웰 방정식이 자기 자신으로 변환된다. 만약 자기 단극자(magnetic monopole)가 존재한다면 이 대칭성은 완전해지며, 완전히 대칭적인 형태를 갖는다:

$$\nabla \cdot \mathbf{E} = \rho_e/\epsilon_0, \quad \nabla \cdot \mathbf{B} = \mu_0\rho_m$$
$$\nabla \times \mathbf{E} = -\mu_0\mathbf{J}_m - \frac{\partial\mathbf{B}}{\partial t}, \quad \nabla \times \mathbf{B} = \mu_0\mathbf{J}_e + \mu_0\epsilon_0\frac{\partial\mathbf{E}}{\partial t}$$

자기 단극자는 아직 관측되지 않았지만, 이 쌍대성은 강력한 이론적 도구이며 현대 물리학의 다양한 분야에서 여러 형태로 등장한다.

---

## 차원 분석과 단위

맥스웰 방정식의 단위를 확인해보는 것은 유익하다. SI 단위계에서:

| 물리량 | 기호 | SI 단위 |
|---|---|---|
| 전기장(Electric field) | $\mathbf{E}$ | V/m = kg$\cdot$m/(A$\cdot$s$^3$) |
| 자기장(Magnetic field) | $\mathbf{B}$ | T = kg/(A$\cdot$s$^2$) |
| 전하 밀도(Charge density) | $\rho$ | C/m$^3$ = A$\cdot$s/m$^3$ |
| 전류 밀도(Current density) | $\mathbf{J}$ | A/m$^2$ |
| 유전율(Permittivity) | $\epsilon_0$ | F/m = A$^2\cdot$s$^4$/(kg$\cdot$m$^3$) |
| 투자율(Permeability) | $\mu_0$ | H/m = kg$\cdot$m/(A$^2\cdot$s$^2$) |

**$\nabla \times \mathbf{B} = \mu_0\epsilon_0\partial\mathbf{E}/\partial t$ 단위 검증**:

- 좌변: $[\nabla \times \mathbf{B}] = \text{T/m}$
- 우변: $[\mu_0\epsilon_0][\mathbf{E}]/[\text{s}] = \text{s}^2/\text{m}^2 \cdot \text{V}/(\text{m}\cdot\text{s}) = \text{T/m}$ ✓

$\mu_0\epsilon_0 = 1/c^2$의 단위는 s$^2$/m$^2$으로, 속도의 제곱의 역수이다.

---

## 역사적 배경

맥스웰은 1865년 "전자기장의 동역학적 이론(A Dynamical Theory of the Electromagnetic Field)"에서 자신의 방정식을 발표했다. 오늘날 우리가 알고 있는 네 개의 방정식 형태는 올리버 헤비사이드(Oliver Heaviside)가 1880년대에 맥스웰의 원래 스무 개 방정식으로부터 — 헤비사이드 자신이 발전시키는 데 기여한 — 벡터 미적분 표기법을 사용하여 간추린 것이다. 하인리히 헤르츠(Heinrich Hertz)는 맥스웰의 예측 이후 20년이 지난 1887년에 전자기파를 실험적으로 확인했다.

---

## 요약

| 개념 | 핵심 방정식 |
|---|---|
| 변위 전류(Displacement current) | $\mathbf{J}_d = \epsilon_0 \partial\mathbf{E}/\partial t$ |
| 가우스 법칙(Gauss's law) | $\nabla \cdot \mathbf{E} = \rho/\epsilon_0$ |
| 자기 단극자 없음(No monopoles) | $\nabla \cdot \mathbf{B} = 0$ |
| 패러데이 법칙(Faraday's law) | $\nabla \times \mathbf{E} = -\partial\mathbf{B}/\partial t$ |
| 앙페르-맥스웰(Ampere-Maxwell) | $\nabla \times \mathbf{B} = \mu_0\mathbf{J} + \mu_0\epsilon_0\partial\mathbf{E}/\partial t$ |
| 파동 방정식(Wave equation) | $\nabla^2\mathbf{E} = \mu_0\epsilon_0\,\partial^2\mathbf{E}/\partial t^2$ |
| 빛의 속도(Speed of light) | $c = 1/\sqrt{\mu_0\epsilon_0}$ |
| 퍼텐셜(Potentials) | $\mathbf{E} = -\nabla V - \partial\mathbf{A}/\partial t$, $\mathbf{B} = \nabla\times\mathbf{A}$ |
| 로렌츠 게이지(Lorenz gauge) | $\nabla\cdot\mathbf{A} + \mu_0\epsilon_0\,\partial V/\partial t = 0$ |
| 달랑베르 연산자(d'Alembertian) | $\Box^2 V = -\rho/\epsilon_0$, $\Box^2\mathbf{A} = -\mu_0\mathbf{J}$ |

---

## 연습 문제

### 연습 1: 변위 전류의 크기
반지름 $R = 5$ cm, 간격 $d = 2$ mm인 원형 평행판 축전기를 전류 $I = 0.5$ A로 충전하고 있다. 두 판 사이의 변위 전류 밀도와 판의 가장자리에서의 자기장을 계산하라. 실제 전류로부터 얻어지는 자기장과 비교하라.

### 연습 2: 파동 방정식 유도
맥스웰 방정식으로부터 출발하여 $\mathbf{B}$에 대한 파동 방정식을 유도하라(이 레슨에서 $\mathbf{E}$에 대해 진행한 유도와 병행하여). $\mathbf{E}$와 $\mathbf{B}$ 모두 같은 속도로 전파함을 확인하라.

### 연습 3: 2차원 파동 시뮬레이션
1차원 FDTD 시뮬레이션을 2차원으로 확장하라. 점 형태의 초기 교란에서 시작하여 원형 파동이 퍼져나가는 것을 관찰하라. 파동 속도가 $c$와 일치함을 확인하라.

### 연습 4: 로렌츠 게이지 검증
원점에 놓인 시간에 따라 변하는 점전하 $q(t) = q_0 \sin(\omega t)$에 대한 지연 퍼텐셜을 써라. 이 퍼텐셜이 로렌츠 게이지 조건을 만족함을 확인하라.

### 연습 5: 게이지 변환
등속도로 운동하는 전하에 대한 로렌츠 게이지 퍼텐셜에서 시작하라. $\lambda = f(x - vt)$인 게이지 변환을 적용하고, 장 $\mathbf{E}$와 $\mathbf{B}$가 불변임을 확인하라.

---

[← 이전: 06. 전자기 유도](06_Electromagnetic_Induction.md) | [다음: 08. 맥스웰 방정식 — 적분 형태 →](08_Maxwells_Equations_Integral.md)
