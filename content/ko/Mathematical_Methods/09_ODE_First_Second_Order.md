# 09. 상미분방정식: 1차와 2차 (Ordinary Differential Equations: First and Second Order)

## 학습 목표

- 1차 ODE의 주요 풀이 기법(분리형, 적분인자, 완전미분, 치환법)을 익힌다
- 2차 상수계수 ODE의 제차·비제차 해법을 이해한다
- 감쇠 조화 진동자와 RLC 회로를 ODE로 모델링하고 해석한다
- 해의 존재·유일성 정리와 론스키안의 의미를 파악한다
- SymPy와 SciPy를 사용하여 해석해를 검증하고 수치해를 구한다

---

## 1. 1차 ODE

1차 상미분방정식의 일반적 형태는 다음과 같다:

$$\frac{dy}{dx} = f(x, y)$$

초기 조건 $y(x_0) = y_0$가 주어지면 **초기값 문제(Initial Value Problem, IVP)**가 된다.

### 1.1 분리형 (Separable)

$f(x,y)$가 $g(x) \cdot h(y)$ 꼴로 분리되면 **분리형 방정식**이다:

$$\frac{dy}{dx} = g(x)\,h(y) \quad\Longrightarrow\quad \frac{dy}{h(y)} = g(x)\,dx$$

양변을 적분하면 해를 얻는다.

**예제: 인구 성장 모델 (로지스틱 방정식)**

$$\frac{dP}{dt} = rP\!\left(1 - \frac{P}{K}\right)$$

여기서 $r$은 고유 성장률, $K$는 환경 수용력(carrying capacity)이다.

부분분수 분해를 통해 분리하면:

$$\frac{dP}{P(1 - P/K)} = r\,dt \quad\Longrightarrow\quad \frac{1}{P} + \frac{1/K}{1 - P/K}\,dP = r\,dt$$

적분하면 해는:

$$P(t) = \frac{K}{1 + \left(\frac{K}{P_0} - 1\right)e^{-rt}}$$

```python
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Function, dsolve, Eq, exp

# SymPy로 로지스틱 방정식 풀기
t = symbols('t')
P = Function('P')
r_val, K_val, P0_val = 0.5, 100, 10

ode = Eq(P(t).diff(t), r_val * P(t) * (1 - P(t) / K_val))
sol = dsolve(ode, P(t), ics={P(0): P0_val})
print("해석해:", sol)

# 수치 검증 (SciPy)
from scipy.integrate import solve_ivp

def logistic(t, y):
    return r_val * y[0] * (1 - y[0] / K_val)

t_span = (0, 20)
t_eval = np.linspace(0, 20, 200)
result = solve_ivp(logistic, t_span, [P0_val], t_eval=t_eval)

plt.figure(figsize=(8, 5))
plt.plot(result.t, result.y[0], 'b-', label='수치해 (solve_ivp)')
plt.axhline(y=K_val, color='r', linestyle='--', label=f'환경 수용력 K={K_val}')
plt.xlabel('시간 t')
plt.ylabel('개체수 P(t)')
plt.title('로지스틱 성장 모델')
plt.legend()
plt.grid(True)
plt.show()
```

### 1.2 선형 1차 ODE와 적분인자 (Integrating Factor)

**표준형(standard form)**:

$$\frac{dy}{dx} + P(x)\,y = Q(x)$$

**적분인자(integrating factor)** $\mu(x)$를 다음과 같이 정의한다:

$$\mu(x) = e^{\int P(x)\,dx}$$

양변에 $\mu(x)$를 곱하면 좌변이 완전미분이 된다:

$$\frac{d}{dx}\bigl[\mu(x)\,y\bigr] = \mu(x)\,Q(x)$$

따라서 일반해는:

$$y = \frac{1}{\mu(x)}\left[\int \mu(x)\,Q(x)\,dx + C\right]$$

**예제: 뉴턴의 냉각 법칙**

$$\frac{dT}{dt} = -k(T - T_{\text{env}})$$

표준형으로 고치면 $\frac{dT}{dt} + kT = kT_{\text{env}}$이므로 $\mu = e^{kt}$이다.

$$T(t) = T_{\text{env}} + (T_0 - T_{\text{env}})\,e^{-kt}$$

```python
from sympy import symbols, Function, dsolve, Eq, exp

t, k, T_env, T0 = symbols('t k T_env T_0', positive=True)
T = Function('T')

ode = Eq(T(t).diff(t), -k * (T(t) - T_env))
sol = dsolve(ode, T(t), ics={T(0): T0})
print("뉴턴 냉각 법칙 해:", sol)
# T(t) = T_env + (T_0 - T_env)*exp(-k*t)
```

### 1.3 완전미분방정식 (Exact Equations)

다음 형태의 방정식:

$$M(x,y)\,dx + N(x,y)\,dy = 0$$

이 **완전(exact)**하려면 다음 조건이 필요충분하다:

$$\frac{\partial M}{\partial y} = \frac{\partial N}{\partial x}$$

이때 포텐셜 함수 $F(x,y)$가 존재하여:

$$\frac{\partial F}{\partial x} = M, \quad \frac{\partial F}{\partial y} = N$$

해는 $F(x,y) = C$ (음함수 형태)이다.

**풀이 절차:**
1. $\frac{\partial M}{\partial y} = \frac{\partial N}{\partial x}$ 확인
2. $F = \int M\,dx + g(y)$ 계산 ($g(y)$는 $y$만의 함수)
3. $\frac{\partial F}{\partial y} = N$으로부터 $g'(y)$ 결정
4. $F(x,y) = C$ 기술

**예제:** $(2xy + 3)\,dx + (x^2 + 4y)\,dy = 0$

- $M = 2xy + 3$, $N = x^2 + 4y$
- $\frac{\partial M}{\partial y} = 2x = \frac{\partial N}{\partial x}$ → 완전
- $F = \int (2xy + 3)\,dx = x^2 y + 3x + g(y)$
- $\frac{\partial F}{\partial y} = x^2 + g'(y) = x^2 + 4y$ → $g'(y) = 4y$ → $g(y) = 2y^2$
- **해:** $x^2 y + 3x + 2y^2 = C$

### 1.4 치환법 (Substitution Methods)

분리형이나 선형이 아닌 방정식도 적절한 치환으로 풀 수 있다.

**동차 방정식 (Homogeneous equation):**

$\frac{dy}{dx} = f\!\left(\frac{y}{x}\right)$ 꼴이면 $v = y/x$ 치환 ($y = vx$)을 사용한다.

$$x\frac{dv}{dx} + v = f(v) \quad\Longrightarrow\quad \frac{dv}{f(v) - v} = \frac{dx}{x}$$

**베르누이 방정식 (Bernoulli equation):**

$$\frac{dy}{dx} + P(x)\,y = Q(x)\,y^n \quad (n \neq 0, 1)$$

$w = y^{1-n}$ 치환으로 선형 1차 ODE로 변환된다:

$$\frac{dw}{dx} + (1-n)P(x)\,w = (1-n)Q(x)$$

```python
from sympy import symbols, Function, dsolve, Eq

x = symbols('x')
y = Function('y')

# 베르누이 방정식: y' + y/x = x*y^2
bernoulli_ode = Eq(y(x).diff(x) + y(x)/x, x * y(x)**2)
sol = dsolve(bernoulli_ode, y(x))
print("베르누이 방정식 해:", sol)
```

---

## 2. 2차 상수계수 ODE

일반적인 2차 상수계수 선형 ODE:

$$a\,y'' + b\,y' + c\,y = f(x)$$

여기서 $a, b, c$는 상수이고, $f(x) = 0$이면 **제차(homogeneous)**, $f(x) \neq 0$이면 **비제차(nonhomogeneous)**이다.

### 2.1 제차 방정식 (Homogeneous)과 특성방정식

$$a\,y'' + b\,y' + c\,y = 0$$

$y = e^{rx}$를 대입하면 **특성방정식(characteristic equation)**을 얻는다:

$$ar^2 + br + c = 0$$

근의 공식: $r = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$

판별식 $D = b^2 - 4ac$에 따라 세 가지 경우가 발생한다.

### 2.2 세 가지 경우: 서로 다른 실근, 중근, 복소근

**경우 1: $D > 0$ — 서로 다른 두 실근 $r_1, r_2$**

$$y = C_1 e^{r_1 x} + C_2 e^{r_2 x}$$

**경우 2: $D = 0$ — 중근 $r_1 = r_2 = r$**

$$y = (C_1 + C_2 x)\,e^{rx}$$

($xe^{rx}$가 두 번째 독립해 — 축퇴(degenerate) 방지)

**경우 3: $D < 0$ — 복소근 $r = \alpha \pm i\beta$**

$$y = e^{\alpha x}\bigl(C_1 \cos\beta x + C_2 \sin\beta x\bigr)$$

오일러 공식 $e^{i\theta} = \cos\theta + i\sin\theta$를 활용한 실수 형태이다.

```python
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Function, dsolve, Eq, cos, sin, exp

x = symbols('x')
y = Function('y')

# 경우 1: y'' - 5y' + 6y = 0 → r = 2, 3
sol1 = dsolve(Eq(y(x).diff(x, 2) - 5*y(x).diff(x) + 6*y(x), 0), y(x))
print("서로 다른 실근:", sol1)

# 경우 2: y'' - 4y' + 4y = 0 → r = 2 (중근)
sol2 = dsolve(Eq(y(x).diff(x, 2) - 4*y(x).diff(x) + 4*y(x), 0), y(x))
print("중근:", sol2)

# 경우 3: y'' + 2y' + 5y = 0 → r = -1 ± 2i
sol3 = dsolve(Eq(y(x).diff(x, 2) + 2*y(x).diff(x) + 5*y(x), 0), y(x))
print("복소근:", sol3)
```

### 2.3 비제차 방정식: 미정계수법 (Undetermined Coefficients)

$$a\,y'' + b\,y' + c\,y = f(x)$$

**일반해** = 제차해 $y_h$ + 특수해 $y_p$

미정계수법은 $f(x)$가 다항식, 지수함수, 삼각함수 또는 이들의 조합일 때 적용한다.

| $f(x)$의 형태 | $y_p$의 추정 형태 |
|---|---|
| $P_n(x)$ (n차 다항식) | $A_n x^n + A_{n-1}x^{n-1} + \cdots + A_0$ |
| $e^{\alpha x}$ | $A e^{\alpha x}$ |
| $\cos\beta x$ 또는 $\sin\beta x$ | $A\cos\beta x + B\sin\beta x$ |
| $e^{\alpha x}\cos\beta x$ | $e^{\alpha x}(A\cos\beta x + B\sin\beta x)$ |

**주의:** 추정한 $y_p$가 제차해 $y_h$에 포함되면 $x$를 곱하여 수정한다 (중복 규칙).

**예제:** $y'' + 4y = 3\sin 2x$

제차해: $y_h = C_1\cos 2x + C_2\sin 2x$ (특성근 $r = \pm 2i$)

$\sin 2x$가 $y_h$에 포함되므로 $y_p = x(A\cos 2x + B\sin 2x)$로 추정한다.

대입 후 계수 비교:

$$y_p = -\frac{3}{4}x\cos 2x$$

```python
from sympy import symbols, Function, dsolve, Eq, sin, cos

x = symbols('x')
y = Function('y')

# y'' + 4y = 3*sin(2x)
ode = Eq(y(x).diff(x, 2) + 4*y(x), 3*sin(2*x))
sol = dsolve(ode, y(x))
print("미정계수법 해:", sol)
```

### 2.4 비제차 방정식: 매개변수 변환법 (Variation of Parameters)

미정계수법이 적용 불가능한 $f(x)$에 대해 사용하는 **일반적 방법**이다.

제차해 $y_1(x), y_2(x)$를 알고 있을 때, 특수해를 다음과 같이 구한다:

$$y_p = u_1(x)\,y_1(x) + u_2(x)\,y_2(x)$$

여기서 $u_1', u_2'$는 연립방정식으로 결정된다:

$$u_1' y_1 + u_2' y_2 = 0$$
$$u_1' y_1' + u_2' y_2' = \frac{f(x)}{a}$$

**론스키안** $W = y_1 y_2' - y_2 y_1'$을 사용하면:

$$u_1' = -\frac{y_2 f(x)}{aW}, \quad u_2' = \frac{y_1 f(x)}{aW}$$

**예제:** $y'' + y = \sec x$

- 제차해: $y_1 = \cos x$, $y_2 = \sin x$
- $W = \cos x \cdot \cos x - \sin x \cdot (-\sin x) = 1$
- $u_1' = -\sin x \cdot \sec x = -\tan x$ → $u_1 = \ln|\cos x|$
- $u_2' = \cos x \cdot \sec x = 1$ → $u_2 = x$
- $y_p = \cos x \ln|\cos x| + x\sin x$

```python
from sympy import symbols, Function, dsolve, Eq, sec

x = symbols('x')
y = Function('y')

# y'' + y = sec(x)
ode = Eq(y(x).diff(x, 2) + y(x), sec(x))
sol = dsolve(ode, y(x), hint='variation_of_parameters')
print("매개변수 변환법 해:", sol)
```

---

## 3. 감쇠 조화 진동자

물리학에서 가장 중요한 2차 ODE 응용이다.

질량 $m$, 감쇠 계수 $\gamma$, 스프링 상수 $k$인 계:

$$m\ddot{x} + \gamma\dot{x} + kx = F(t)$$

$\omega_0 = \sqrt{k/m}$ (고유 진동수), $\beta = \gamma/(2m)$ (감쇠 상수)로 정의하면:

$$\ddot{x} + 2\beta\dot{x} + \omega_0^2 x = \frac{F(t)}{m}$$

### 3.1 자유 진동 (Free Oscillation)

외력이 없는 경우 ($F(t) = 0$):

$$\ddot{x} + 2\beta\dot{x} + \omega_0^2 x = 0$$

특성방정식: $r^2 + 2\beta r + \omega_0^2 = 0$

$$r = -\beta \pm \sqrt{\beta^2 - \omega_0^2}$$

### 3.2 과감쇠, 임계감쇠, 부족감쇠

판별식 $\beta^2 - \omega_0^2$의 부호에 따라 세 가지 진동 양상이 나타난다:

**1) 부족감쇠 (Underdamped): $\beta < \omega_0$**

$$x(t) = A e^{-\beta t}\cos(\omega_d t + \phi)$$

여기서 감쇠 진동수 $\omega_d = \sqrt{\omega_0^2 - \beta^2}$

진동하면서 지수적으로 감소한다. 대부분의 물리계가 이 경우에 해당한다.

**2) 임계감쇠 (Critically Damped): $\beta = \omega_0$**

$$x(t) = (C_1 + C_2 t)\,e^{-\beta t}$$

진동 없이 가장 빠르게 평형으로 복귀한다. 도어 댐퍼 등에 활용된다.

**3) 과감쇠 (Overdamped): $\beta > \omega_0$**

$$x(t) = C_1 e^{r_1 t} + C_2 e^{r_2 t}$$

두 실근 모두 음수이므로 천천히 평형에 접근한다.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

omega0 = 5.0   # 고유 진동수
x0, v0 = 1.0, 0.0  # 초기 조건: x(0)=1, v(0)=0

fig, ax = plt.subplots(figsize=(10, 6))
t_eval = np.linspace(0, 5, 500)

for label, beta in [('부족감쇠 (β=1)', 1.0),
                     ('임계감쇠 (β=5)', 5.0),
                     ('과감쇠 (β=8)', 8.0)]:
    def damped_osc(t, y, b=beta):
        return [y[1], -2*b*y[1] - omega0**2 * y[0]]

    sol = solve_ivp(damped_osc, (0, 5), [x0, v0], t_eval=t_eval)
    ax.plot(sol.t, sol.y[0], label=label)

ax.set_xlabel('시간 t (s)')
ax.set_ylabel('변위 x(t)')
ax.set_title('감쇠 조화 진동자: 세 가지 감쇠 영역')
ax.legend()
ax.grid(True)
ax.axhline(y=0, color='k', linewidth=0.5)
plt.tight_layout()
plt.show()
```

### 3.3 강제 진동과 공명 (Resonance)

외력 $F(t) = F_0 \cos\omega t$가 가해지는 경우:

$$\ddot{x} + 2\beta\dot{x} + \omega_0^2 x = \frac{F_0}{m}\cos\omega t$$

정상 상태(steady-state) 특수해:

$$x_p(t) = A(\omega)\cos(\omega t - \delta)$$

여기서 진폭과 위상:

$$A(\omega) = \frac{F_0/m}{\sqrt{(\omega_0^2 - \omega^2)^2 + 4\beta^2\omega^2}}$$

$$\tan\delta = \frac{2\beta\omega}{\omega_0^2 - \omega^2}$$

**공명 조건:** 진폭 $A(\omega)$가 최대가 되는 구동 진동수:

$$\omega_{\text{res}} = \sqrt{\omega_0^2 - 2\beta^2}$$

$\beta \to 0$이면 $\omega_{\text{res}} \to \omega_0$ (비감쇠 공명). 감쇠가 없으면 진폭이 무한대로 발산한다.

**Q-팩터 (Quality Factor):**

$$Q = \frac{\omega_0}{2\beta}$$

$Q$가 클수록 공명 피크가 날카롭고, 에너지 손실이 적다.

```python
import numpy as np
import matplotlib.pyplot as plt

omega0 = 10.0
F0_over_m = 1.0
omega = np.linspace(0.1, 20, 500)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for beta in [0.2, 0.5, 1.0, 2.0]:
    A = F0_over_m / np.sqrt((omega0**2 - omega**2)**2 + (2*beta*omega)**2)
    delta = np.arctan2(2*beta*omega, omega0**2 - omega**2)
    Q = omega0 / (2*beta)
    ax1.plot(omega, A, label=f'β={beta}, Q={Q:.1f}')
    ax2.plot(omega, np.degrees(delta), label=f'β={beta}')

ax1.set_xlabel('구동 진동수 ω')
ax1.set_ylabel('진폭 A(ω)')
ax1.set_title('강제 진동: 공명 곡선')
ax1.legend()
ax1.grid(True)

ax2.set_xlabel('구동 진동수 ω')
ax2.set_ylabel('위상차 δ (°)')
ax2.set_title('강제 진동: 위상 응답')
ax2.legend()
ax2.grid(True)
plt.tight_layout()
plt.show()
```

---

## 4. 전기 회로 해석

### 4.1 RLC 회로 방정식

직렬 RLC 회로에 키르히호프 전압 법칙(KVL)을 적용하면:

$$L\frac{dI}{dt} + RI + \frac{Q}{C} = V(t)$$

$I = dQ/dt$를 사용하면 전하 $Q$에 대한 2차 ODE가 된다:

$$L\frac{d^2Q}{dt^2} + R\frac{dQ}{dt} + \frac{1}{C}Q = V(t)$$

**감쇠 진동자와의 대응 관계:**

| 역학계 | 전기 회로 |
|--------|----------|
| 질량 $m$ | 인덕턴스 $L$ |
| 감쇠 계수 $\gamma$ | 저항 $R$ |
| 스프링 상수 $k$ | $1/C$ |
| 변위 $x$ | 전하 $Q$ |
| 속도 $\dot{x}$ | 전류 $I$ |
| 외력 $F(t)$ | 전원 전압 $V(t)$ |

고유 진동수: $\omega_0 = 1/\sqrt{LC}$, 감쇠 상수: $\beta = R/(2L)$

### 4.2 과도 응답과 정상 상태 응답

전체 응답은 두 부분으로 나뉜다:

- **과도 응답 (Transient response):** 제차해 $Q_h(t)$ — 시간이 지남에 따라 감쇠하여 사라진다
- **정상 상태 응답 (Steady-state response):** 특수해 $Q_p(t)$ — 구동 주파수로 지속 진동한다

$V(t) = V_0 \cos\omega t$인 경우, 정상 상태 전류:

$$I_{\text{ss}}(t) = \frac{V_0}{Z}\cos(\omega t - \phi)$$

여기서 $Z$는 임피던스, $\phi$는 위상각이다.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# RLC 회로 파라미터
L = 0.5      # 인덕턴스 (H)
R = 10.0     # 저항 (Ω)
C = 100e-6   # 커패시턴스 (F)
V0 = 12.0    # 전원 진폭 (V)
omega_drive = 100.0  # 구동 각진동수 (rad/s)

omega0 = 1 / np.sqrt(L * C)
beta = R / (2 * L)
print(f"고유 진동수: ω₀ = {omega0:.1f} rad/s")
print(f"감쇠 상수: β = {beta:.1f} s⁻¹")

# Q'' + (R/L)Q' + (1/LC)Q = V(t)/L
def rlc_circuit(t, y):
    Q, I = y
    dQ = I
    dI = (V0 * np.cos(omega_drive * t) - R * I - Q / C) / L
    return [dQ, dI]

t_span = (0, 0.5)
t_eval = np.linspace(0, 0.5, 2000)
sol = solve_ivp(rlc_circuit, t_span, [0.0, 0.0], t_eval=t_eval,
                method='RK45', max_step=1e-4)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(sol.t * 1000, sol.y[0] * 1e6, 'b-')
ax1.set_ylabel('전하 Q (μC)')
ax1.set_title('직렬 RLC 회로 응답')
ax1.grid(True)

ax2.plot(sol.t * 1000, sol.y[1] * 1000, 'r-')
ax2.set_xlabel('시간 (ms)')
ax2.set_ylabel('전류 I (mA)')
ax2.grid(True)
plt.tight_layout()
plt.show()
```

### 4.3 임피던스와 복소수 풀이

교류 회로에서는 복소 임피던스(complex impedance)를 사용하면 ODE를 대수 방정식으로 변환할 수 있다.

$$V(t) = V_0 e^{i\omega t} \quad\text{로 놓으면}$$

각 소자의 복소 임피던스:
- 저항: $Z_R = R$
- 인덕터: $Z_L = i\omega L$
- 커패시터: $Z_C = \frac{1}{i\omega C} = -\frac{i}{\omega C}$

직렬 합성 임피던스:

$$Z = R + i\!\left(\omega L - \frac{1}{\omega C}\right)$$

임피던스의 크기와 위상:

$$|Z| = \sqrt{R^2 + \left(\omega L - \frac{1}{\omega C}\right)^2}$$

$$\phi = \arctan\frac{\omega L - 1/(\omega C)}{R}$$

정상 상태 전류 진폭: $I_0 = V_0 / |Z|$

**공명 조건:** $\omega L = 1/(\omega C)$, 즉 $\omega = \omega_0 = 1/\sqrt{LC}$일 때 $|Z| = R$로 최소, 전류 최대.

```python
import numpy as np
import matplotlib.pyplot as plt

L, R, C = 0.5, 10.0, 100e-6
V0 = 12.0
omega = np.linspace(10, 500, 1000)
omega0 = 1 / np.sqrt(L * C)

# 복소 임피던스
Z = R + 1j * (omega * L - 1 / (omega * C))
I_amp = V0 / np.abs(Z)
phase = np.angle(Z, deg=True)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(omega, I_amp, 'b-')
ax1.axvline(x=omega0, color='r', linestyle='--', label=f'공명 ω₀={omega0:.1f}')
ax1.set_ylabel('전류 진폭 I₀ (A)')
ax1.set_title('RLC 회로 주파수 응답')
ax1.legend()
ax1.grid(True)

ax2.plot(omega, phase, 'g-')
ax2.axvline(x=omega0, color='r', linestyle='--')
ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.set_xlabel('각진동수 ω (rad/s)')
ax2.set_ylabel('위상각 φ (°)')
ax2.grid(True)
plt.tight_layout()
plt.show()
```

---

## 5. 해의 존재와 유일성

### 5.1 피카르-린델뢰프 정리

**정리 (Picard-Lindelöf Theorem):**

초기값 문제 $\frac{dy}{dx} = f(x, y)$, $y(x_0) = y_0$에서, $f(x,y)$와 $\frac{\partial f}{\partial y}$가 $(x_0, y_0)$를 포함하는 직사각형 영역에서 연속이면, $x_0$ 근방에서 해가 **존재하고 유일**하다.

핵심은 $f$의 **립시츠 조건(Lipschitz condition)**이다:

$$|f(x, y_1) - f(x, y_2)| \leq L|y_1 - y_2|$$

$\frac{\partial f}{\partial y}$가 유계(bounded)이면 립시츠 조건이 자동 충족된다.

**반례:** $\frac{dy}{dx} = y^{1/3}$, $y(0) = 0$

$f(x,y) = y^{1/3}$은 $y = 0$에서 $\frac{\partial f}{\partial y} = \frac{1}{3}y^{-2/3} \to \infty$이므로 립시츠 조건이 깨진다. 실제로 $y = 0$과 $y = \left(\frac{2x}{3}\right)^{3/2}$ 모두 해이다 (유일성 실패).

**피카르 반복법 (Picard Iteration):**

정리의 증명에 사용되는 반복법은 수치적으로도 유용하다:

$$y_{n+1}(x) = y_0 + \int_{x_0}^{x} f(t, y_n(t))\,dt$$

```python
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Rational, Piecewise, integrate

# 피카르 반복: y' = x + y, y(0) = 1
# 정확한 해: y = 2e^x - x - 1

x_sym = symbols('x')
y_exact = 2 * np.e**np.linspace(0, 2, 200) - np.linspace(0, 2, 200) - 1

# 수치적 피카르 반복
x_vals = np.linspace(0, 2, 200)

def picard_iteration(f, x0, y0, x_vals, n_iter=6):
    """피카르 반복법으로 ODE를 근사 해석한다."""
    from scipy.integrate import cumulative_trapezoid
    y_n = np.full_like(x_vals, y0, dtype=float)
    results = [y_n.copy()]

    for _ in range(n_iter):
        integrand = f(x_vals, y_n)
        integral = cumulative_trapezoid(integrand, x_vals, initial=0)
        y_n = y0 + integral
        results.append(y_n.copy())
    return results

f = lambda x, y: x + y
iterations = picard_iteration(f, 0, 1, x_vals, n_iter=6)

plt.figure(figsize=(10, 6))
for i, y_approx in enumerate(iterations[1:], 1):
    if i in [1, 2, 3, 6]:
        plt.plot(x_vals, y_approx, '--', label=f'피카르 반복 {i}회')
plt.plot(x_vals, y_exact, 'k-', linewidth=2, label='정확한 해')
plt.xlabel('x')
plt.ylabel('y')
plt.title('피카르 반복법의 수렴')
plt.legend()
plt.grid(True)
plt.show()
```

### 5.2 론스키안 (Wronskian)과 선형 독립

2차 ODE $y'' + P(x)y' + Q(x)y = 0$의 두 해 $y_1, y_2$에 대해 **론스키안**은:

$$W(y_1, y_2) = \begin{vmatrix} y_1 & y_2 \\ y_1' & y_2' \end{vmatrix} = y_1 y_2' - y_2 y_1'$$

**핵심 정리:**

1. $y_1, y_2$가 선형 독립 ⟺ $W \neq 0$ (ODE의 해인 경우)
2. **아벨의 정리 (Abel's Theorem):** $W(x) = W(x_0)\,\exp\!\left(-\int_{x_0}^{x} P(s)\,ds\right)$
3. $W$는 항등적으로 0이거나 절대 0이 아니다 (ODE의 해에 대해)

**예제:** $y_1 = e^x$, $y_2 = e^{2x}$

$$W = e^x \cdot 2e^{2x} - e^{2x} \cdot e^x = 2e^{3x} - e^{3x} = e^{3x} \neq 0$$

따라서 선형 독립이고, 일반해 $y = C_1 e^x + C_2 e^{2x}$의 기저(basis)가 된다.

**$n$차로의 일반화:**

$n$개의 함수 $y_1, \ldots, y_n$에 대해:

$$W = \begin{vmatrix}
y_1 & y_2 & \cdots & y_n \\
y_1' & y_2' & \cdots & y_n' \\
\vdots & \vdots & \ddots & \vdots \\
y_1^{(n-1)} & y_2^{(n-1)} & \cdots & y_n^{(n-1)}
\end{vmatrix}$$

```python
from sympy import symbols, exp, Matrix, simplify

x = symbols('x')

# 론스키안 계산
y1 = exp(x)
y2 = exp(2*x)

W_matrix = Matrix([
    [y1, y2],
    [y1.diff(x), y2.diff(x)]
])
W = simplify(W_matrix.det())
print(f"W(e^x, e^(2x)) = {W}")  # e^(3x)

# 3개 함수의 론스키안
y3 = exp(3*x)
W3 = Matrix([
    [y1, y2, y3],
    [y1.diff(x), y2.diff(x), y3.diff(x)],
    [y1.diff(x, 2), y2.diff(x, 2), y3.diff(x, 2)]
])
print(f"W(e^x, e^(2x), e^(3x)) = {simplify(W3.det())}")  # 2*e^(6x)
```

---

## 연습 문제

### 기본 문제

**1.** 다음 분리형 ODE를 풀어라: $\frac{dy}{dx} = \frac{x^2}{1 + y^2}$, $y(0) = 0$

**2.** 적분인자를 사용하여 풀어라: $\frac{dy}{dx} + 2xy = x$

**3.** 다음이 완전미분방정식인지 확인하고, 완전하면 풀어라:
$(3x^2 y + y^3)\,dx + (x^3 + 3xy^2)\,dy = 0$

**4.** 특성방정식을 사용하여 풀어라: $y'' - 6y' + 9y = 0$, $y(0) = 2$, $y'(0) = 5$

**5.** 미정계수법으로 풀어라: $y'' + 3y' + 2y = 4e^{-x}$

### 응용 문제

**6.** 질량 $m = 0.5\,\text{kg}$, 스프링 상수 $k = 8\,\text{N/m}$, 감쇠 계수 $\gamma = 2\,\text{kg/s}$인 감쇠 진동자가 $x(0) = 0.1\,\text{m}$, $\dot{x}(0) = 0$에서 출발한다.
   - (a) 감쇠 유형(과감쇠/임계/부족감쇠)을 판별하라
   - (b) 해석적 해 $x(t)$를 구하라
   - (c) Python으로 수치해를 구하여 비교하라

**7.** 직렬 RLC 회로에서 $L = 0.1\,\text{H}$, $R = 20\,\Omega$, $C = 50\,\mu\text{F}$이고 $V(t) = 10\cos(100t)\,\text{V}$일 때:
   - (a) 고유 진동수 $\omega_0$와 감쇠 상수 $\beta$를 구하라
   - (b) 정상 상태 전류의 진폭과 위상을 구하라
   - (c) 공명이 되려면 $\omega$를 얼마로 해야 하는가?

**8.** 매개변수 변환법을 사용하여 풀어라: $y'' + 4y = \frac{1}{\sin 2x}$

### 심화 문제

**9.** 론스키안을 사용하여 $\{1, x, x^2\}$가 선형 독립임을 보여라.

**10.** 피카르 반복법을 적용하여 $y' = y$, $y(0) = 1$의 처음 5번 반복을 수행하고, $e^x$의 테일러 급수와 비교하라.

```python
# 문제 6 풀이 코드 (뼈대)
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

m, gamma_val, k = 0.5, 2.0, 8.0
omega0 = np.sqrt(k / m)
beta_val = gamma_val / (2 * m)
print(f"ω₀ = {omega0:.2f}, β = {beta_val:.2f}")
print(f"β² - ω₀² = {beta_val**2 - omega0**2:.2f}")
# β=2, ω₀=4 → β < ω₀ → 부족감쇠

def damped_system(t, y):
    return [y[1], -(gamma_val/m)*y[1] - (k/m)*y[0]]

t_span = (0, 5)
t_eval = np.linspace(0, 5, 500)
sol = solve_ivp(damped_system, t_span, [0.1, 0.0], t_eval=t_eval)

# 해석해: x(t) = A*exp(-β*t)*cos(ωd*t + φ)
omega_d = np.sqrt(omega0**2 - beta_val**2)
A = 0.1 / np.cos(np.arctan(beta_val / omega_d))  # 초기 조건으로부터
phi = np.arctan(beta_val / omega_d)
x_analytic = A * np.exp(-beta_val * t_eval) * np.cos(omega_d * t_eval + phi)

plt.figure(figsize=(10, 5))
plt.plot(sol.t, sol.y[0], 'b-', label='수치해')
plt.plot(t_eval, x_analytic, 'r--', label='해석해')
plt.xlabel('시간 (s)')
plt.ylabel('변위 x(t) (m)')
plt.title('부족감쇠 조화 진동자')
plt.legend()
plt.grid(True)
plt.show()
```

---

## 참고 자료

- **Mary L. Boas**, *Mathematical Methods in the Physical Sciences*, 3rd Edition, Chapter 8
- **George B. Arfken**, *Mathematical Methods for Physicists*, Chapter 9
- **Erwin Kreyszig**, *Advanced Engineering Mathematics*, Chapters 1-3
- **SymPy ODE 문서**: https://docs.sympy.org/latest/modules/solvers/ode.html
- **SciPy solve_ivp 문서**: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
- **3Blue1Brown**: "Differential Equations" 시리즈 (시각적 직관)

---

## 다음 레슨

[10. 고차 ODE와 연립계 (Higher-Order ODE Systems)](10_Higher_Order_ODE_Systems.md)에서는 고차 상미분방정식, 연립 ODE, 행렬 방법, 위상 평면 분석을 다룹니다.
