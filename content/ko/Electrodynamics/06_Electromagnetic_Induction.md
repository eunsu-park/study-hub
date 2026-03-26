# 전자기 유도

[← 이전: 05. 자기 벡터 퍼텐셜](05_Magnetic_Vector_Potential.md) | [다음: 07. 맥스웰 방정식 — 미분 형태 →](07_Maxwells_Equations_Differential.md)

---

## 학습 목표

1. 패러데이 전자기 유도 법칙(Faraday's law of electromagnetic induction)을 적분 형태와 미분 형태로 서술한다
2. 렌츠의 법칙(Lenz's law)과 에너지 보존과의 연결을 설명한다
3. 자기장 속에서 운동하는 도체의 운동 기전력(motional EMF)을 계산한다
4. 자기 인덕턴스(self-inductance)와 상호 인덕턴스(mutual inductance)를 정의하고 표준 기하학에서 계산한다
5. 자기장에 저장된 에너지와 자기 에너지 밀도를 유도한다
6. 인덕턴스를 이용해 RL 회로의 과도 응답(transient)을 분석한다
7. 파이썬으로 전자기 유도 현상을 수치적으로 시뮬레이션한다

---

지금까지 우리는 전기장과 자기장을 완전히 별개의 현상으로 다루었다 — 전기장은 전하에서, 자기장은 전류에서 생기며, 둘 사이에는 아무런 상호작용이 없다고 보았다. 1831년, 마이클 패러데이(Michael Faraday)는 간단하지만 혁명적인 실험으로 이 분리를 무너뜨렸다: 변하는 자기장이 전기장을 만들어낸다는 것을 보인 것이다. 이 전자기 유도의 발견은 전기와 자기를 잇는 다리이며, 궁극적으로 맥스웰 방정식과 전자기파 예측으로 이어진다. 또한 현대 세계의 모든 발전기, 변압기, 무선 충전기의 작동 원리이기도 하다.

---

## 패러데이 법칙

### 실험

패러데이는 회로를 통과하는 자기 선속(magnetic flux)이 시간에 따라 변할 때마다 회로에 기전력(EMF, electromotive force)이 유도된다는 것을 관찰했다. 변화가 빠를수록 기전력이 크다.

### 적분 형태

$$\boxed{\mathcal{E} = -\frac{d\Phi_B}{dt}}$$

여기서 회로 $C$로 경계 지어진 곡면 $S$를 통과하는 자기 선속은:

$$\Phi_B = \int_S \mathbf{B} \cdot d\mathbf{a}$$

회로 주위의 기전력은:

$$\mathcal{E} = \oint_C \mathbf{E} \cdot d\mathbf{l}$$

따라서 패러데이 법칙을 완전한 적분 형태로 쓰면:

$$\oint_C \mathbf{E} \cdot d\mathbf{l} = -\frac{d}{dt}\int_S \mathbf{B} \cdot d\mathbf{a}$$

### 미분 형태

왼쪽에 스토크스 정리(Stokes' theorem)를 적용하면:

$$\int_S (\nabla \times \mathbf{E}) \cdot d\mathbf{a} = -\int_S \frac{\partial \mathbf{B}}{\partial t} \cdot d\mathbf{a}$$

이것이 임의의 곡면 $S$에 대해 성립하므로:

$$\boxed{\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}}$$

이는 정전기학으로부터의 심오한 도약이다: 정전기학에서는 $\nabla \times \mathbf{E} = 0$ (보존장)이었다. 이제 시간에 따라 변하는 $\mathbf{B}$가 있으면, 전기장에 컬이 생긴다 — 더 이상 보존장이 아니다. 전기장이 닫힌 루프를 따라 전하를 밀 수 있게 된다.

> **유추**: 물을 저으면(자기장을 변화시키면) 비로소 나타나는 소용돌이를 상상해 보라. 정전기학에서 전기적 "물"은 높은 곳에서 낮은 곳으로 곧장 흐른다 (높은 $V$에서 낮은 $V$로). 전자기 유도에서는 변하는 자기장이 젓는 숟가락처럼 작용하여, 시작도 끝도 없이 순환하는 전기장의 소용돌이를 만들어낸다.

---

## 렌츠의 법칙

패러데이 법칙의 음의 부호를 **렌츠의 법칙(Lenz's law)**이라 한다:

> 유도 기전력은 그것을 만들어내는 선속 변화에 반대한다.

이는 **에너지 보존**의 결과이다. 유도 전류가 선속 변화에 반대하지 않고 오히려 강화한다면, 계는 폭주하게 된다 — 점점 커지는 전류와 장을 만들어내며 에너지 보존을 위반할 것이다.

**실용적 규칙**: 루프를 통과하는 선속이 증가하면, 유도 전류는 그 증가에 반대하는 자기장을 만드는 방향으로 흐른다 (원래의 선속을 유지하려 한다). 선속이 감소하면, 전류는 이를 유지하는 방향으로 흐른다.

```python
import numpy as np
import matplotlib.pyplot as plt

# Faraday's law: EMF induced by a time-varying magnetic field through a loop
# Why simulation: seeing the relationship between dΦ/dt and EMF builds intuition

# A circular loop of radius R is in a time-varying uniform B field
R_loop = 0.1        # loop radius (10 cm)
A_loop = np.pi * R_loop**2   # loop area

t = np.linspace(0, 2, 1000)  # time in seconds

# Case 1: Linearly increasing B
B_linear = 0.5 * t    # B in Tesla, increasing at 0.5 T/s
Phi_linear = B_linear * A_loop
EMF_linear = -np.gradient(Phi_linear, t)   # EMF = -dΦ/dt

# Case 2: Sinusoidal B
omega = 2 * np.pi * 2    # 2 Hz
B_sin = 0.5 * np.sin(omega * t)
Phi_sin = B_sin * A_loop
EMF_sin = -np.gradient(Phi_sin, t)

# Case 3: Exponentially decaying B
tau = 0.5       # decay time constant
B_exp = 0.5 * np.exp(-t / tau)
Phi_exp = B_exp * A_loop
EMF_exp = -np.gradient(Phi_exp, t)

fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# Why paired plots: seeing B(t) alongside EMF(t) makes Faraday's law visual
for idx, (B, EMF, Phi, label) in enumerate([
    (B_linear, EMF_linear, Phi_linear, 'Linear B(t) = 0.5t'),
    (B_sin, EMF_sin, Phi_sin, 'Sinusoidal B(t) = 0.5sin(ωt)'),
    (B_exp, EMF_exp, Phi_exp, 'Exponential B(t) = 0.5e^{-t/τ}')
]):
    axes[idx, 0].plot(t, B, 'b-', linewidth=2, label='B(t)')
    axes[idx, 0].plot(t, Phi * 100, 'g--', linewidth=1.5, label='Φ(t) ×100')
    axes[idx, 0].set_ylabel('B (T) / Φ×100 (Wb)')
    axes[idx, 0].set_title(f'{label}')
    axes[idx, 0].legend()
    axes[idx, 0].grid(True, alpha=0.3)

    axes[idx, 1].plot(t, EMF * 1e3, 'r-', linewidth=2)
    axes[idx, 1].set_ylabel('EMF (mV)')
    axes[idx, 1].set_title(f'EMF = -dΦ/dt')
    axes[idx, 1].grid(True, alpha=0.3)
    axes[idx, 1].axhline(y=0, color='gray', linestyle='-', alpha=0.3)

for ax in axes[-1, :]:
    ax.set_xlabel('Time (s)')

plt.suptitle("Faraday's Law: EMF = -dΦ/dt", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('faraday_law.png', dpi=150)
plt.show()
```

---

## 운동 기전력

도체가 자기장 속에서 움직일 때, 도체 내부의 자유 전하는 자기력 $\mathbf{F} = q\mathbf{v} \times \mathbf{B}$를 받는다. 이 힘이 전류를 구동하여 기전력을 만들어낸다.

### 미끄러지는 막대

길이 $l$인 도체 막대가 속도 $v$로 두 평행 레일 위를 미끄러지며, 균일한 자기장 $\mathbf{B} = B\hat{z}$가 레일 면에 수직으로 걸려 있다:

$$\mathcal{E} = Blv$$

이는 패러데이 법칙과 일치한다: 막대가 움직이면서 회로의 넓이가 $dA/dt = lv$로 증가하므로, $d\Phi/dt = BlV$이고 $|\mathcal{E}| = Blv$이다.

### 일반적인 운동 기전력

속도 $\mathbf{v}$로 움직이는 임의의 도체 루프에 대해:

$$\mathcal{E}_{\text{motional}} = \oint (\mathbf{v} \times \mathbf{B}) \cdot d\mathbf{l}$$

### 패러데이 법칙이 두 경우를 통합한다

패러데이 법칙 $\mathcal{E} = -d\Phi_B/dt$는 보편적으로 적용된다:
1. **변압기 기전력(Transformer EMF)**: $\mathbf{B}$가 변하고 루프는 정지 $\to$ $\nabla \times \mathbf{E} = -\partial\mathbf{B}/\partial t$
2. **운동 기전력(Motional EMF)**: 루프가 움직이고 $\mathbf{B}$는 정적 $\to$ $\mathbf{v} \times \mathbf{B}$에 의한 힘
3. **둘 다**: 두 효과 모두 기여

```python
import numpy as np
import matplotlib.pyplot as plt

# Motional EMF: conducting bar sliding on rails in a uniform B field
# Why simulate: watching the current build up and the bar decelerate is instructive

B = 0.5          # magnetic field (T)
l = 0.2          # bar length (m)
R = 1.0          # circuit resistance (Ω)
m = 0.01         # bar mass (kg)
v0 = 5.0         # initial velocity (m/s)

# Time integration
# Why fine dt: the exponential decay requires good resolution
dt = 1e-4
t_max = 0.1
t = np.arange(0, t_max, dt)
N = len(t)

v = np.zeros(N)
x = np.zeros(N)
v[0] = v0

for i in range(N - 1):
    # EMF from motion
    emf = B * l * v[i]

    # Current in the circuit: I = EMF/R
    I = emf / R

    # Force on the bar: F = -BIl (Lenz's law — opposes motion)
    # Why negative: the induced current creates a force opposing the velocity
    F = -B * I * l

    # Update velocity and position
    v[i+1] = v[i] + (F / m) * dt
    x[i+1] = x[i] + v[i+1] * dt

# Analytic solution: v(t) = v₀ exp(-B²l²t/(mR))
# Why exponential: the braking force is proportional to velocity
tau_decay = m * R / (B * l)**2
v_analytic = v0 * np.exp(-t / tau_decay)
emf_values = B * l * v
I_values = emf_values / R

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(t * 1000, v, 'b-', linewidth=2, label='Numerical')
axes[0, 0].plot(t * 1000, v_analytic, 'r--', linewidth=2, label='Analytic')
axes[0, 0].set_xlabel('Time (ms)')
axes[0, 0].set_ylabel('Velocity (m/s)')
axes[0, 0].set_title('Bar Velocity')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(t * 1000, emf_values * 1000, 'g-', linewidth=2)
axes[0, 1].set_xlabel('Time (ms)')
axes[0, 1].set_ylabel('EMF (mV)')
axes[0, 1].set_title('Induced EMF = Blv')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(t * 1000, I_values * 1000, 'r-', linewidth=2)
axes[1, 0].set_xlabel('Time (ms)')
axes[1, 0].set_ylabel('Current (mA)')
axes[1, 0].set_title('Induced Current = EMF/R')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(t * 1000, x * 100, 'm-', linewidth=2)
axes[1, 1].set_xlabel('Time (ms)')
axes[1, 1].set_ylabel('Position (cm)')
axes[1, 1].set_title('Bar Position')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle(f'Motional EMF: Sliding Bar (τ = {tau_decay*1000:.2f} ms)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('motional_emf.png', dpi=150)
plt.show()

print(f"Decay time constant: τ = {tau_decay*1000:.2f} ms")
print(f"Initial EMF: {B*l*v0*1000:.1f} mV")
print(f"At t = τ, v = {v0/np.e:.3f} m/s (v₀/e)")
```

---

## 자기 인덕턴스

회로에 전류가 흐르면, 회로 자체를 통과하는 자기 선속이 생긴다. 전류가 변하면 이 선속도 변하여, 변화에 반대하는 기전력을 유도한다. 이 현상을 **자기 유도(self-induction)**라 한다.

**자기 인덕턴스(self-inductance)** $L$은 다음과 같이 정의된다:

$$\Phi_B = LI$$

역기전력(back-EMF)은:

$$\mathcal{E} = -L\frac{dI}{dt}$$

인덕턴스의 단위는 **헨리(henry, H)**이다: 1 H = 1 V$\cdot$s/A = 1 Wb/A.

### 표준 기하학의 인덕턴스

**솔레노이드** (길이 $l$, $N$회 감김, 단면적 $A$):
$$L = \mu_0 \frac{N^2}{l} A = \mu_0 n^2 l A$$

**토로이드(Toroid)** ($N$회 감김, 안쪽 반지름 $a$, 바깥쪽 반지름 $b$, 높이 $h$):
$$L = \frac{\mu_0 N^2 h}{2\pi} \ln\frac{b}{a}$$

**동축 케이블(Coaxial cable)** (안쪽 반지름 $a$, 바깥쪽 반지름 $b$, 길이 $l$):
$$L = \frac{\mu_0 l}{2\pi} \ln\frac{b}{a}$$

> **유추**: 인덕턴스는 관성(inertia)의 전자기적 유사체이다. 질량이 큰 물체가 속도 변화에 저항하듯 ($F = ma = m \, dv/dt$), 인덕터는 전류 변화에 저항한다 ($\mathcal{E} = -L \, dI/dt$). $L$이 큰 코일은 "무거운" 것이다 — 전류를 빠르게 바꾸려면 큰 기전력이 필요하다.

---

## 상호 인덕턴스

두 회로가 가까이 있으면, 한 회로의 전류가 다른 회로를 통과하는 선속을 만들어낸다. **상호 인덕턴스(mutual inductance)** $M$은 회로 1의 전류에 의한 회로 2를 통과하는 선속을 연결한다:

$$\Phi_{21} = M_{21} I_1$$

**노이만 공식(Neumann formula)** (두 루프의 경우):
$$M = \frac{\mu_0}{4\pi} \oint_{C_1} \oint_{C_2} \frac{d\mathbf{l}_1 \cdot d\mathbf{l}_2}{|\mathbf{r}_1 - \mathbf{r}_2|}$$

**핵심 성질**: $M_{12} = M_{21} \equiv M$ — 상호 인덕턴스는 대칭이다! 증명은 두 루프에 대해 명백히 대칭인 노이만 공식을 이용한다.

### 변압기

변압기(transformer)는 상호 인덕턴스를 활용한다. 공통 코어에 감긴 두 코일:

$$\frac{V_2}{V_1} = \frac{N_2}{N_1}$$

이 전압비는 이상적인 변압기(완전 결합, 저항 없음)에서 정확히 성립한다.

```python
import numpy as np
import matplotlib.pyplot as plt

# Mutual inductance: two coaxial loops
# Why Neumann formula: it gives M directly from the geometry

mu_0 = 4 * np.pi * 1e-7

def mutual_inductance_coaxial_loops(R1, R2, d, N_segments=1000):
    """
    Compute mutual inductance between two coaxial circular loops.

    Parameters:
        R1, R2: radii of the two loops
        d: axial separation between loop centers
        N_segments: number of segments per loop

    Returns:
        M: mutual inductance (H)
    """
    # Discretize both loops
    phi1 = np.linspace(0, 2*np.pi, N_segments, endpoint=False)
    phi2 = np.linspace(0, 2*np.pi, N_segments, endpoint=False)
    dphi = 2 * np.pi / N_segments

    # Neumann formula: M = (μ₀/4π) ∮∮ dl₁·dl₂ / |r₁-r₂|
    # Why double integral: each segment of loop 1 interacts with each of loop 2
    M = 0.0
    for p1 in phi1:
        # Position and direction of current element on loop 1 (at z=0)
        x1 = R1 * np.cos(p1)
        y1 = R1 * np.sin(p1)
        z1 = 0.0
        dl1_x = -R1 * np.sin(p1) * dphi
        dl1_y = R1 * np.cos(p1) * dphi

        # All segments of loop 2 (at z=d)
        x2 = R2 * np.cos(phi2)
        y2 = R2 * np.sin(phi2)
        z2 = d

        dl2_x = -R2 * np.sin(phi2) * dphi
        dl2_y = R2 * np.cos(phi2) * dphi

        # Separation
        dx = x1 - x2
        dy = y1 - y2
        dz = z1 - z2
        r = np.sqrt(dx**2 + dy**2 + dz**2)

        # dl₁ · dl₂
        dot = dl1_x * dl2_x + dl1_y * dl2_y

        M += np.sum(dot / r)

    M *= mu_0 / (4 * np.pi)
    return M

# Compute M as a function of separation
R1 = R2 = 0.1   # both loops have radius 10 cm
separations = np.linspace(0.01, 0.5, 50)
M_values = np.array([mutual_inductance_coaxial_loops(R1, R2, d, N_segments=500)
                      for d in separations])

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(separations * 100, M_values * 1e6, 'b-', linewidth=2)
ax.set_xlabel('Separation d (cm)')
ax.set_ylabel('M (μH)')
ax.set_title(f'Mutual Inductance of Two Coaxial Loops (R = {R1*100:.0f} cm)')
ax.grid(True, alpha=0.3)
ax.set_yscale('log')
plt.tight_layout()
plt.savefig('mutual_inductance.png', dpi=150)
plt.show()

print(f"M at d = R:     {mutual_inductance_coaxial_loops(R1, R2, R1, 500)*1e6:.4f} μH")
print(f"M at d = 0.01m: {M_values[0]*1e6:.4f} μH")
```

---

## 자기장에 저장된 에너지

전류 $I$가 흐르는 인덕터에 저장된 에너지:

$$W = \frac{1}{2}LI^2$$

더 일반적으로, 자기장에 저장된 에너지는:

$$\boxed{W = \frac{1}{2\mu_0}\int_{\text{all space}} |\mathbf{B}|^2 \, d\tau}$$

**자기 에너지 밀도(magnetic energy density)**는:

$$u_B = \frac{B^2}{2\mu_0} \quad [\text{J/m}^3]$$

전기 에너지 밀도 $u_E = \frac{\epsilon_0}{2}E^2$와 비교하면, 전자기장의 총 에너지 밀도는:

$$u = \frac{1}{2}\left(\epsilon_0 E^2 + \frac{B^2}{\mu_0}\right)$$

### 예: 솔레노이드의 에너지

솔레노이드 내부 ($B = \mu_0 nI$, 부피 $= Al$):

$$W = \frac{B^2}{2\mu_0}(Al) = \frac{(\mu_0 nI)^2}{2\mu_0}Al = \frac{1}{2}\mu_0 n^2 Al \cdot I^2 = \frac{1}{2}LI^2$$

일치한다!

---

## RL 회로

전압원 $V_0$와 직렬로 연결된 저항 $R$과 인덕터 $L$로 구성된 RL 회로:

$$V_0 = IR + L\frac{dI}{dt}$$

### 충전 ($t=0$에 스위치 닫힘):

$$I(t) = \frac{V_0}{R}\left(1 - e^{-t/\tau}\right), \qquad \tau = \frac{L}{R}$$

시간 상수 $\tau = L/R$이 전류가 얼마나 빠르게 상승하는지를 결정한다. $L$이 클수록 ("관성"이 클수록) 상승이 느리다.

### 방전 (전원 제거):

$$I(t) = I_0 \, e^{-t/\tau}$$

```python
import numpy as np
import matplotlib.pyplot as plt

# RL circuit transients: charging and discharging
# Why both analytic and numerical: the numerical approach generalizes to nonlinear circuits

R = 10.0        # resistance (Ω)
L = 0.1         # inductance (H)
V0 = 5.0        # source voltage (V)
tau = L / R      # time constant

t = np.linspace(0, 5 * tau, 1000)
dt = t[1] - t[0]

# Analytic solutions
I_charge_analytic = (V0 / R) * (1 - np.exp(-t / tau))
I_discharge_analytic = (V0 / R) * np.exp(-t / tau)

# Numerical solution using Euler method
I_charge_numerical = np.zeros_like(t)
I_discharge_numerical = np.zeros_like(t)
I_discharge_numerical[0] = V0 / R

for i in range(len(t) - 1):
    # Charging: V₀ = IR + L dI/dt → dI/dt = (V₀ - IR)/L
    dI_dt_c = (V0 - I_charge_numerical[i] * R) / L
    I_charge_numerical[i+1] = I_charge_numerical[i] + dI_dt_c * dt

    # Discharging: 0 = IR + L dI/dt → dI/dt = -IR/L
    dI_dt_d = -I_discharge_numerical[i] * R / L
    I_discharge_numerical[i+1] = I_discharge_numerical[i] + dI_dt_d * dt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Charging
axes[0].plot(t / tau, I_charge_analytic * 1000, 'b-', linewidth=2, label='Analytic')
axes[0].plot(t / tau, I_charge_numerical * 1000, 'r--', linewidth=2, label='Numerical')
axes[0].axhline(y=V0/R * 1000, color='gray', linestyle=':', label=f'I_max = {V0/R*1000:.0f} mA')
axes[0].axvline(x=1, color='green', linestyle='--', alpha=0.5, label=f'τ = {tau*1000:.1f} ms')
axes[0].set_xlabel('t / τ')
axes[0].set_ylabel('I (mA)')
axes[0].set_title('RL Charging')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Discharging
axes[1].plot(t / tau, I_discharge_analytic * 1000, 'b-', linewidth=2, label='Analytic')
axes[1].plot(t / tau, I_discharge_numerical * 1000, 'r--', linewidth=2, label='Numerical')
axes[1].axvline(x=1, color='green', linestyle='--', alpha=0.5, label=f'τ = {tau*1000:.1f} ms')
axes[1].set_xlabel('t / τ')
axes[1].set_ylabel('I (mA)')
axes[1].set_title('RL Discharging')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle(f'RL Circuit (R = {R} Ω, L = {L*1000:.0f} mH, τ = {tau*1000:.1f} ms)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('rl_circuit.png', dpi=150)
plt.show()
```

---

## 요약

| 개념 | 핵심 방정식 |
|---|---|
| 패러데이 법칙 (적분형) | $\mathcal{E} = -d\Phi_B/dt$ |
| 패러데이 법칙 (미분형) | $\nabla \times \mathbf{E} = -\partial\mathbf{B}/\partial t$ |
| 렌츠의 법칙 | 유도 기전력은 변화에 반대한다 |
| 운동 기전력 | $\mathcal{E} = \oint (\mathbf{v}\times\mathbf{B})\cdot d\mathbf{l}$ |
| 자기 인덕턴스 | $\mathcal{E} = -L\,dI/dt$ |
| 솔레노이드 인덕턴스 | $L = \mu_0 n^2 l A$ |
| 상호 인덕턴스 | $\Phi_{21} = MI_1$ |
| 노이만 공식 | $M = \frac{\mu_0}{4\pi}\oint\oint \frac{d\mathbf{l}_1\cdot d\mathbf{l}_2}{|\mathbf{r}_1-\mathbf{r}_2|}$ |
| 자기 에너지 | $W = \frac{1}{2}LI^2 = \frac{1}{2\mu_0}\int B^2\,d\tau$ |
| 자기 에너지 밀도 | $u_B = B^2/(2\mu_0)$ |
| RL 시간 상수 | $\tau = L/R$ |

---

## 연습 문제

### 연습 문제 1: 교류 발전기
넓이 $A$인 직사각형 루프가 균일한 자기장 $\mathbf{B}$ 속에서 각진동수 $\omega$로 회전한다. $N$회 감긴 경우, 유도 기전력이 $\mathcal{E} = NBA\omega\sin(\omega t)$임을 보여라. 현실적인 값을 사용하여 기전력과 순시 전력을 도시하라.

### 연습 문제 2: 맴돌이 전류 제동
도전성 원판이 자기장 속에서 회전한다. 맴돌이 전류(eddy current)에 의한 제동 토크를 모델링하고 감속을 시뮬레이션하라. 제동 토크는 각속도에 어떻게 의존하는가?

### 연습 문제 3: 결합된 RL 회로
자기 인덕턴스 $L_1$, $L_2$와 상호 인덕턴스 $M$을 가진 두 코일이 직렬로 연결되어 있다. 운동 방정식을 유도하고 유효 인덕턴스를 계산하라. 같은 방향으로 연결된 경우($L_{\text{eff}} = L_1 + L_2 + 2M$)와 반대 방향으로 연결된 경우($L_{\text{eff}} = L_1 + L_2 - 2M$)를 비교하라.

### 연습 문제 4: 유도에서의 에너지 보존
미끄러지는 막대 문제에서 에너지 보존을 수치적으로 검증하라: 막대가 잃은 운동 에너지가 저항에서 소비된 총 에너지($\int I^2 R \, dt$)와 같음을 보여라.

### 연습 문제 5: 토로이드의 인덕턴스
$N = 500$회 감기고, 안쪽 반지름 $a = 10$ cm, 바깥쪽 반지름 $b = 15$ cm, 높이 $h = 3$ cm인 토로이드 코일의 인덕턴스를 해석적으로 계산하라. 그런 다음 $N$개의 원형 전류 루프로 모델링하여 노이만 공식으로 인덕턴스를 수치적으로 계산하라. 두 결과를 비교하라.

---

[← 이전: 05. 자기 벡터 퍼텐셜](05_Magnetic_Vector_Potential.md) | [다음: 07. 맥스웰 방정식 — 미분 형태 →](07_Maxwells_Equations_Differential.md)
