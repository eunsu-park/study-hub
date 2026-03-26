# 13. 2계 상미분방정식(Second-Order Ordinary Differential Equations)

## 학습 목표

- 특성방정식(characteristic equation)을 사용하여 상수계수 2계 제차 ODE를 푼다
- 세 가지 경우를 모두 다룬다: 서로 다른 실근, 중근, 켤레 복소근
- 미정계수법(undetermined coefficients)과 매개변수 변환법(variation of parameters)으로 특수해를 구한다
- 기계적 진동을 모델링한다: 자유, 감쇠, 강제 진동, 그리고 공명(resonance)
- 스프링-질량 시스템을 수치적으로 시뮬레이션하고 Python으로 공명 현상을 시각화한다

---

## 1. 상수계수 제차 방정식(Homogeneous Equations with Constant Coefficients)

### 1.1 일반형

**상수계수 2계 선형 ODE**는:

$$ay'' + by' + cy = g(x)$$

여기서 $a, b, c$는 상수이고 $g(x)$는 주어진 함수이다.

$g(x) = 0$이면 방정식은 **제차(homogeneous)**이다:

$$ay'' + by' + cy = 0$$

### 1.2 특성방정식

핵심 통찰은 $y = e^{rx}$를 추측하는 것이다. 대입하면:

$$ar^2 e^{rx} + bre^{rx} + ce^{rx} = 0$$

$e^{rx} \neq 0$이므로, 나누면 **특성방정식(characteristic equation)**을 얻는다:

$$ar^2 + br + c = 0$$

이것은 $r$에 관한 이차방정식이다. 그 근이 해의 형태를 완전히 결정한다.

**비유:** 특성방정식은 미분방정식 문제를 대수 문제로 변환한다. 미분방정식을 푸는 대신 이차방정식을 풀면 된다 -- 훨씬 간단한 작업이다.

### 1.3 세 가지 경우

**경우 1: 서로 다른 두 실근 $r_1 \neq r_2$**

$$y = C_1 e^{r_1 x} + C_2 e^{r_2 x}$$

**예제:** $y'' - 5y' + 6y = 0$

특성방정식: $r^2 - 5r + 6 = (r-2)(r-3) = 0$, 따라서 $r = 2, 3$.

$$y = C_1 e^{2x} + C_2 e^{3x}$$

**경우 2: 중근 $r_1 = r_2 = r$**

독립적인 해 하나 $e^{rx}$만 얻는다. 두 번째는 $x$를 곱하여 찾는다:

$$y = C_1 e^{rx} + C_2\, x\, e^{rx} = (C_1 + C_2 x)e^{rx}$$

**예제:** $y'' - 4y' + 4y = 0$

특성방정식: $r^2 - 4r + 4 = (r-2)^2 = 0$, 따라서 $r = 2$ (중근).

$$y = (C_1 + C_2 x)e^{2x}$$

**왜 $xe^{rx}$인가?** 특성 다항식이 중근을 가질 때, 표준 지수함수는 하나의 해만 준다. **차수 축소법(reduction of order)**을 통해 $xe^{rx}$가 누락된 독립적인 해임을 알 수 있다.

**경우 3: 켤레 복소근 $r = \alpha \pm i\beta$**

오일러 공식 $e^{i\beta x} = \cos\beta x + i\sin\beta x$를 사용하면:

$$y = e^{\alpha x}(C_1\cos\beta x + C_2\sin\beta x)$$

**예제:** $y'' + 2y' + 5y = 0$

특성방정식: $r^2 + 2r + 5 = 0$, 따라서 $r = -1 \pm 2i$.

$$y = e^{-x}(C_1\cos 2x + C_2\sin 2x)$$

이것은 **지수적 감소** ($e^{-x}$)를 동반한 **진동** ($\cos$과 $\sin$)을 나타낸다 -- 정확히 감쇠 진동의 모습이다.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Visualize all three cases ---
x = np.linspace(0, 4, 300)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Case 1: distinct real roots (r=2, r=-1)
# y'' - y' - 2y = 0, y(0)=1, y'(0)=0
# y = (1/3)e^(2x) + (2/3)e^(-x)
y1 = (1/3) * np.exp(2*x) + (2/3) * np.exp(-x)
axes[0].plot(x, y1, 'b-', linewidth=2)
axes[0].set_title('Case 1: Distinct Real Roots\n$r = 2, -1$')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_ylim(-1, 10)
axes[0].grid(True)

# Case 2: repeated root (r=2)
# y'' - 4y' + 4y = 0, y(0)=1, y'(0)=0
# y = (1 - 2x)e^(2x)
x2 = np.linspace(0, 2, 300)
y2 = (1 - 2*x2) * np.exp(2*x2)
axes[1].plot(x2, y2, 'r-', linewidth=2)
axes[1].axhline(y=0, color='gray', linewidth=0.5)
axes[1].set_title('Case 2: Repeated Root\n$r = 2$ (double)')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].grid(True)

# Case 3: complex roots (r = -1 ± 2i)
# y'' + 2y' + 5y = 0, y(0)=1, y'(0)=0
# y = e^(-x)(cos(2x) + (1/2)sin(2x))
y3 = np.exp(-x) * (np.cos(2*x) + 0.5*np.sin(2*x))
envelope = np.exp(-x) * np.sqrt(1 + 0.25)
axes[2].plot(x, y3, 'g-', linewidth=2, label='Solution')
axes[2].plot(x, envelope, 'r--', alpha=0.5, label='Envelope $e^{-x}\\sqrt{5/4}$')
axes[2].plot(x, -envelope, 'r--', alpha=0.5)
axes[2].axhline(y=0, color='gray', linewidth=0.5)
axes[2].set_title('Case 3: Complex Roots\n$r = -1 \\pm 2i$')
axes[2].set_xlabel('x')
axes[2].set_ylabel('y')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()
```

---

## 2. 비제차 방정식(Nonhomogeneous Equations)

### 2.1 일반해의 구조

$ay'' + by' + cy = g(x)$에 대해, 일반해는:

$$y = y_h + y_p$$

- $y_h$: **제차(보조) 해** -- $ay'' + by' + cy = 0$을 만족
- $y_p$: **특수해** -- 전체 방정식의 어떤 하나의 해

**비유:** 일반해를 찾는 것은 직선 위의 모든 점을 찾는 것과 같다. 직선 위의 하나의 특정 점($y_p$)과 직선의 방향($y_h$, 해의 "자유도"를 포착)이 필요하다.

### 2.2 미정계수법(Method of Undetermined Coefficients)

이 방법은 $g(x)$가 다항식, 지수함수, 사인, 코사인, 또는 이들의 조합일 때 작동한다.

**아이디어:** $g(x)$와 같은 형태이지만 미지의 계수를 가진 해를 추측한다. ODE에 대입하고 계수를 구한다.

| $g(x)$ | $y_p$에 대한 추측 |
|---------|-----------------|
| $P_n(x)$ ($n$차 다항식) | $A_n x^n + A_{n-1}x^{n-1} + \cdots + A_0$ |
| $e^{\alpha x}$ | $Ae^{\alpha x}$ |
| $\cos\beta x$ 또는 $\sin\beta x$ | $A\cos\beta x + B\sin\beta x$ |
| $e^{\alpha x}\cos\beta x$ | $e^{\alpha x}(A\cos\beta x + B\sin\beta x)$ |

**수정 규칙:** 추측이 $y_h$의 항과 중복되면, $x$를 곱한다 (두 항 모두 일치하면 $x^2$를 곱한다).

**예제:** $y'' + 3y' + 2y = 4e^{-x}$

$y_h = C_1 e^{-x} + C_2 e^{-2x}$ (근 $r = -1, -2$)

초기 추측 $y_p = Ae^{-x}$는 $C_1 e^{-x}$와 중복된다. 수정: $y_p = Axe^{-x}$.

대입하면: $y_p'' + 3y_p' + 2y_p = Ae^{-x}(-1 + 3(-1) + 2\cdot 0) \ldots$ (세심한 대수 계산 후) $A = -4$를 준다.

$$y_p = -4xe^{-x}$$

### 2.3 매개변수 변환법(Variation of Parameters)

특수한 형태뿐만 아니라 **임의의** $g(x)$에 대해 작동하는 더 일반적인 방법.

제차해 $y_1, y_2$가 주어졌을 때, $y_p = u_1 y_1 + u_2 y_2$를 구한다:

$$u_1' = -\frac{y_2 g(x)}{aW}, \quad u_2' = \frac{y_1 g(x)}{aW}$$

여기서 $W = y_1 y_2' - y_2 y_1'$는 **론스키안(Wronskian)**이다.

**어떤 방법을 언제 사용하는가:**
- **미정계수법:** $g(x)$가 적절한 형태(다항식, 지수, 삼각함수)일 때 빠르고 쉽다
- **매개변수 변환법:** 항상 작동하지만, 론스키안 계산과 적분이 필요하다 -- 미정계수법이 적용되지 않을 때 사용

```python
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Function, dsolve, Eq, exp, cos, sin

x = symbols('x')
y = Function('y')

# --- Undetermined coefficients example ---
# y'' + 3y' + 2y = 10*cos(x)
ode = Eq(y(x).diff(x, 2) + 3*y(x).diff(x) + 2*y(x), 10*cos(x))
sol = dsolve(ode, y(x))
print(f"General solution: {sol}")

# With initial conditions y(0) = 0, y'(0) = 0
sol_ivp = dsolve(ode, y(x), ics={y(0): 0, y(x).diff(x).subs(x, 0): 0})
print(f"IVP solution:     {sol_ivp}")

# --- Variation of parameters example ---
# y'' + y = sec(x)  (cannot use undetermined coefficients!)
ode2 = Eq(y(x).diff(x, 2) + y(x), 1/cos(x))
sol2 = dsolve(ode2, y(x))
print(f"\nVariation of parameters: {sol2}")
```

---

## 3. 기계적 진동(Mechanical Vibrations)

2계 ODE의 가장 중요한 물리적 응용은 **질량-스프링 시스템**이다.

### 3.1 운동 방정식

질량 $m$에 스프링 (상수 $k$), 감쇠($\gamma$), 외력 $F(t)$이 작용할 때:

$$m\ddot{x} + \gamma\dot{x} + kx = F(t)$$

정의:
- $\omega_0 = \sqrt{k/m}$: **고유 진동수(natural frequency)** (시스템이 자연스럽게 진동하려는 주파수)
- $\beta = \gamma/(2m)$: **감쇠 계수**

방정식은 다음이 된다:

$$\ddot{x} + 2\beta\dot{x} + \omega_0^2 x = \frac{F(t)}{m}$$

### 3.2 자유 비감쇠 진동(Free Undamped Oscillation) ($\beta = 0$, $F = 0$)

$$\ddot{x} + \omega_0^2 x = 0 \implies x(t) = A\cos(\omega_0 t + \phi)$$

이것은 **단순 조화 운동(simple harmonic motion)**이다 -- 고유 진동수에서의 정현파 진동이 영원히 지속된다. 진폭 $A$와 위상 $\phi$는 초기 조건에 의해 결정된다.

### 3.3 자유 감쇠 진동(Free Damped Oscillation) ($\beta > 0$, $F = 0$)

$$\ddot{x} + 2\beta\dot{x} + \omega_0^2 x = 0$$

특성근: $r = -\beta \pm \sqrt{\beta^2 - \omega_0^2}$

| 영역 | 조건 | 거동 |
|--------|-----------|----------|
| **부족감쇠(Underdamped)** | $\beta < \omega_0$ | 감소하는 진폭으로 진동: $x = Ae^{-\beta t}\cos(\omega_d t + \phi)$ |
| **임계감쇠(Critically damped)** | $\beta = \omega_0$ | 진동 없이 가장 빠르게 평형으로 복귀: $x = (C_1 + C_2 t)e^{-\beta t}$ |
| **과감쇠(Overdamped)** | $\beta > \omega_0$ | 진동 없이 느리게 복귀: $x = C_1 e^{r_1 t} + C_2 e^{r_2 t}$ |

여기서 $\omega_d = \sqrt{\omega_0^2 - \beta^2}$는 **감쇠 진동수(damped frequency)** ($\omega_0$보다 약간 낮다)이다.

**실세계 예시:**
- **부족감쇠:** 자동차가 움푹한 곳에 부딪힌 경우 (안정되기 전에 몇 번 튀어오른다)
- **임계감쇠:** 문닫힘 장치 (쿵 닫히지 않고 빠르게 복귀한다)
- **과감쇠:** 두꺼운 기름 속의 무거운 문 (매우 느리게 복귀한다)

### 3.4 강제 진동과 공명(Forced Oscillation and Resonance) ($F(t) = F_0\cos\omega t$)

$$\ddot{x} + 2\beta\dot{x} + \omega_0^2 x = \frac{F_0}{m}\cos\omega t$$

정상 상태 응답은 구동력과 같은 주파수를 가진다:

$$x_p(t) = A(\omega)\cos(\omega t - \delta)$$

여기서 **진폭**과 **위상 지연**은:

$$A(\omega) = \frac{F_0/m}{\sqrt{(\omega_0^2 - \omega^2)^2 + 4\beta^2\omega^2}}$$

$$\delta = \arctan\frac{2\beta\omega}{\omega_0^2 - \omega^2}$$

**공명(resonance)**은 $\omega = \omega_0$ 근처(더 정확히는 $\omega_{\text{res}} = \sqrt{\omega_0^2 - 2\beta^2}$)에서 발생하며, 진폭이 최대가 된다.

**Q인자(quality factor)**는 공명의 날카로움을 측정한다:

$$Q = \frac{\omega_0}{2\beta}$$

Q가 높을수록 공명 피크가 더 날카롭고 높으며 에너지 소산이 느려진다.

**공명 재해:** 타코마 해협 다리 붕괴(1940), 소리로 와인 잔이 깨지는 현상, 잘못 설계된 구조물의 기계적 파괴 등이 모두 공명과 관련된다.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Spring-mass system: three damping regimes ---
omega0 = 5.0  # natural frequency
x0, v0 = 1.0, 0.0  # initial displacement 1, initial velocity 0

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top-left: Three damping regimes
t_eval = np.linspace(0, 6, 600)
for label, beta, color in [('Underdamped ($\\beta=1$)', 1.0, 'blue'),
                             ('Critically damped ($\\beta=5$)', 5.0, 'green'),
                             ('Overdamped ($\\beta=8$)', 8.0, 'red')]:
    def osc(t, y, b=beta):
        return [y[1], -2*b*y[1] - omega0**2 * y[0]]
    sol = solve_ivp(osc, (0, 6), [x0, v0], t_eval=t_eval)
    axes[0, 0].plot(sol.t, sol.y[0], color=color, linewidth=2, label=label)

axes[0, 0].axhline(y=0, color='gray', linewidth=0.5)
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Displacement x(t)')
axes[0, 0].set_title('Free Damped Oscillation')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Top-right: Resonance curves (amplitude vs driving frequency)
omega = np.linspace(0.1, 10, 500)
F0_m = 1.0

for beta in [0.2, 0.5, 1.0, 2.0]:
    A = F0_m / np.sqrt((omega0**2 - omega**2)**2 + (2*beta*omega)**2)
    Q = omega0 / (2*beta)
    axes[0, 1].plot(omega, A, linewidth=2, label=f'$\\beta={beta}$, Q={Q:.1f}')

axes[0, 1].axvline(x=omega0, color='gray', linestyle=':', alpha=0.5)
axes[0, 1].set_xlabel('Driving frequency $\\omega$ (rad/s)')
axes[0, 1].set_ylabel('Amplitude $A(\\omega)$')
axes[0, 1].set_title('Resonance Curves')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Bottom-left: Forced oscillation (transient + steady-state)
beta_f = 0.3
omega_drive = 4.8  # near resonance

def forced_osc(t, y):
    """Forced damped oscillator."""
    return [y[1],
            F0_m * np.cos(omega_drive * t) - 2*beta_f*y[1] - omega0**2*y[0]]

t_long = np.linspace(0, 30, 2000)
sol_f = solve_ivp(forced_osc, (0, 30), [0, 0], t_eval=t_long)

axes[1, 0].plot(sol_f.t, sol_f.y[0], 'b-', linewidth=1)
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Displacement x(t)')
axes[1, 0].set_title(f'Forced Oscillation ($\\omega_d={omega_drive}$, near resonance)')
axes[1, 0].grid(True)

# Bottom-right: Beat phenomenon (undamped, omega close to omega0)
omega_beat = 4.5
def beat_osc(t, y):
    """Undamped forced oscillator -- produces beats."""
    return [y[1], F0_m * np.cos(omega_beat * t) - omega0**2 * y[0]]

sol_beat = solve_ivp(beat_osc, (0, 40), [0, 0],
                      t_eval=np.linspace(0, 40, 3000))

axes[1, 1].plot(sol_beat.t, sol_beat.y[0], 'purple', linewidth=1)
# Beat envelope
delta_omega = omega0 - omega_beat
envelope = F0_m / (omega0**2 - omega_beat**2) * 2
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('Displacement x(t)')
axes[1, 1].set_title(f'Beat Phenomenon ($\\omega_d={omega_beat}$, $\\omega_0={omega0}$)')
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()
```

---

## 4. 위상 지연과 에너지(Phase Lag and Energy)

### 4.1 위상 응답

위상 지연 $\delta$는 응답이 구동력에 비해 얼마나 뒤처지는지를 알려준다:

- 낮은 주파수에서 ($\omega \ll \omega_0$): $\delta \approx 0$ -- 시스템이 힘과 동위상(in phase)으로 따라간다
- 공명에서 ($\omega = \omega_0$): $\delta = \pi/2$ -- 응답이 1/4 주기 뒤처진다
- 높은 주파수에서 ($\omega \gg \omega_0$): $\delta \approx \pi$ -- 응답이 힘과 거의 반대 위상이다

### 4.2 감쇠 진동에서의 에너지

감쇠 진동자의 전체 역학적 에너지는 시간이 지남에 따라 감소한다:

$$E(t) = \frac{1}{2}m\dot{x}^2 + \frac{1}{2}kx^2$$

$$\frac{dE}{dt} = -\gamma\dot{x}^2 \le 0$$

에너지 손실률은 감쇠력 곱하기 속도와 같다 -- 열로 소산되는 일률이다.

부족감쇠 진동자에서: $E(t) \approx E_0\, e^{-2\beta t}$

**감쇠 시간**(에너지가 $1/e$로 떨어지는 시간)은 $\tau = 1/(2\beta)$이고, Q인자는 $Q = \omega_0\tau/2 = \omega_0/(2\beta)$이다.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Energy decay in underdamped oscillation ---
omega0 = 5.0
beta = 0.3
m = 1.0
k = omega0**2 * m

def damped_osc(t, y):
    return [y[1], -2*beta*y[1] - omega0**2 * y[0]]

t_eval = np.linspace(0, 20, 2000)
sol = solve_ivp(damped_osc, (0, 20), [1.0, 0.0], t_eval=t_eval)
x_vals = sol.y[0]
v_vals = sol.y[1]

# Compute kinetic and potential energy
KE = 0.5 * m * v_vals**2
PE = 0.5 * k * x_vals**2
E_total = KE + PE

# Theoretical envelope
E_envelope = 0.5 * k * np.exp(-2 * beta * t_eval)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax1.plot(t_eval, x_vals, 'b-', linewidth=1.5)
ax1.plot(t_eval, np.exp(-beta * t_eval), 'r--', alpha=0.6, label='Envelope')
ax1.plot(t_eval, -np.exp(-beta * t_eval), 'r--', alpha=0.6)
ax1.set_ylabel('Displacement x(t)')
ax1.set_title('Underdamped Oscillation and Energy Decay')
ax1.legend()
ax1.grid(True)

ax2.plot(t_eval, KE, 'r-', alpha=0.7, linewidth=1, label='Kinetic Energy')
ax2.plot(t_eval, PE, 'b-', alpha=0.7, linewidth=1, label='Potential Energy')
ax2.plot(t_eval, E_total, 'k-', linewidth=2, label='Total Energy')
ax2.plot(t_eval, E_envelope, 'g--', linewidth=2, label='$E_0 e^{-2\\beta t}$')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Energy (J)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
```

---

## 5. 요약 표

| 유형 | 방정식 | 특성근 | 해의 형태 |
|------|----------|---------------------|---------------|
| 서로 다른 실근 | $ay'' + by' + cy = 0$ | $r_1 \neq r_2$ (실수) | $C_1 e^{r_1 x} + C_2 e^{r_2 x}$ |
| 중근 | $ay'' + by' + cy = 0$ | $r_1 = r_2 = r$ | $(C_1 + C_2 x)e^{rx}$ |
| 복소근 | $ay'' + by' + cy = 0$ | $\alpha \pm i\beta$ | $e^{\alpha x}(C_1\cos\beta x + C_2\sin\beta x)$ |
| 비제차 | $ay'' + by' + cy = g(x)$ | -- | $y_h + y_p$ |

---

## 6. 교차 참조

- **Mathematical Methods 레슨 10**은 고계 ODE, 연립 ODE, 멱급수법(프로베니우스)을 다룬다.
- **레슨 12 (1계 ODE)**는 존재-유일성과 1계 기법을 포함하여 이 레슨의 기초를 제공한다.
- **레슨 14 (연립 ODE)**는 2계 ODE를 시스템으로 변환하고 고유값 방법과 위상 평면으로 분석하는 방법을 보여준다.
- **전자기학(Electrodynamics) 레슨 08**은 RLC 회로(스프링-질량 시스템의 전기적 유사체)에 2계 ODE 이론을 적용한다.

---

## 연습 문제

**1.** IVP를 풀어라: $y'' + 6y' + 8y = 0$, $y(0) = 2$, $y'(0) = -1$. 근을 분류하고 해를 스케치하라.

**2.** $y'' + 4y' + 4y = 3e^{-2x}$의 일반해를 구하라. (힌트: 추측 $Ae^{-2x}$는 두 제차해 모두와 중복된다 -- $Ax^2 e^{-2x}$가 필요하다.)

**3.** 매개변수 변환법을 사용하여 $y'' + y = \tan x$를 풀어라.

**4.** 스프링-질량 시스템이 $m = 2\,\text{kg}$, $k = 50\,\text{N/m}$, $\gamma = 4\,\text{kg/s}$를 가진다.
   - (a) 시스템이 부족감쇠, 임계감쇠, 과감쇠 중 어느 것인가?
   - (b) $x(0) = 0.1\,\text{m}$이고 $\dot{x}(0) = 0$이면, $x(t)$를 구하라.
   - (c) `solve_ivp`를 사용하여 $x(t)$를 그리고 해석적 해를 겹쳐 그려라.
   - (d) 진폭이 초기 변위의 1%로 떨어지는 데 얼마나 걸리는가?

**5.** 강제 진동자 $\ddot{x} + 0.4\dot{x} + 25x = 5\cos\omega t$에 대해:
   - (a) 고유 진동수 $\omega_0$와 Q인자를 구하라.
   - (b) 정상 상태 진폭이 최대화되는 구동 주파수 $\omega$는?
   - (c) 진폭 $A(\omega)$와 위상 $\delta(\omega)$를 $\omega$의 함수로 그려라.
   - (d) 공명에서와 $\omega = 1$ (공명에서 먼)에서 `solve_ivp`로 시스템을 시뮬레이션하라. 정상 상태 진폭을 비교하라.

---

## 참고 자료

- **William E. Boyce & Richard C. DiPrima**, *Elementary Differential Equations*, 11th Edition, Chapters 3-4
- **Erwin Kreyszig**, *Advanced Engineering Mathematics*, 10th Edition, Chapter 2
- **Mary L. Boas**, *Mathematical Methods in the Physical Sciences*, 3rd Edition, Chapter 8
- **A.P. French**, *Vibrations and Waves*, MIT Introductory Physics Series
- **3Blue1Brown**, "Differential Equations" series

---

[이전: 1계 상미분방정식](./12_First_Order_ODE.md) | [다음: 상미분방정식의 연립](./14_Systems_of_ODE.md)
