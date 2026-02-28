# ODE를 위한 라플라스 변환(Laplace Transform for ODE)

## 학습 목표

- 라플라스 변환(Laplace transform)을 정의하고 적분 정의로부터 기본 함수의 변환을 계산한다
- 핵심 성질(선형성, 이동, 미분, 합성곱)을 적용하여 변환 계산을 간소화한다
- 부분분수 분해를 사용하여 역 라플라스 변환(inverse Laplace transform)을 수행한다
- ODE를 s-영역의 대수 방정식으로 변환하여 초기값 문제를 푼다
- 헤비사이드 계단 함수(Heaviside step function)와 디랙 델타 함수(Dirac delta function)를 사용하여 불연속 외력을 모델링한다

## 선수 과목

이 레슨을 공부하기 전에 다음에 익숙해야 한다:
- 2계 선형 ODE와 초기값 문제 (레슨 10-12)
- 미분방정식의 연립 (레슨 14)
- 미적분학의 이상적분(improper integrals)

## 동기: 왜 라플라스 변환인가?

미정계수법이나 매개변수 변환법으로 상수계수 ODE를 매끄러운 외력 함수에 대해 풀 수 있다. 그런데 외력 함수가 **불연속**이면 어떻게 할까 -- 시간 $t = 3$에서 스위치가 켜지거나, $t = 0$에서 충격적인 망치 타격이 있는 경우? 일반해를 먼저 구한 다음 상수를 맞추는 대신, 하나의 체계적인 단계로 해를 원하면 어떻게 할까?

**라플라스 변환**은 시간 영역의 미분방정식을 주파수 영역($s$-영역)의 **대수 방정식**으로 변환한다. 대수를 풀고 다시 역변환한다. 로그를 사용하여 곱셈을 덧셈으로 바꾸는 것과 같다: 문제를 더 쉬운 것으로 바꾸고, 풀고, 다시 변환한다.

```
  Time Domain              s-Domain
  ──────────              ────────
  Differential Eq.  ──L──>  Algebraic Eq.
       │                        │
       │  (hard)                │  (easy)
       ▼                        ▼
  Solution y(t)   <──L⁻¹──  Solution Y(s)
```

## 라플라스 변환의 정의

$t \geq 0$에서 정의된 함수 $f(t)$가 주어졌을 때, 그 **라플라스 변환**은:

$$\mathcal{L}\{f(t)\} = F(s) = \int_0^{\infty} e^{-st} f(t) \, dt$$

여기서:
- $f(t)$: **시간 영역**의 원래 함수 (입력 신호, 외력 함수 등)
- $F(s)$: **$s$-영역**(주파수 영역이라고도 함)의 변환된 함수
- $s$: 복소 변수, $s = \sigma + j\omega$, 적분이 수렴하도록 선택
- $e^{-st}$: 함수에 지수 감소로 "가중치"를 부여하는 **핵(kernel)**

적분은 $\text{Re}(s)$가 $f(t)$의 성장을 충분히 이길 수 있을 때 수렴한다. 수렴을 위한 $\text{Re}(s)$의 최솟값을 **수렴 좌표(abscissa of convergence)**라 한다.

### 정의로부터 변환 계산

**예제 1: $f(t) = 1$ (상수 함수)**

$$\mathcal{L}\{1\} = \int_0^{\infty} e^{-st} \cdot 1 \, dt = \left[-\frac{1}{s} e^{-st}\right]_0^{\infty} = 0 - \left(-\frac{1}{s}\right) = \frac{1}{s}, \quad s > 0$$

**예제 2: $f(t) = e^{at}$ (지수함수)**

$$\mathcal{L}\{e^{at}\} = \int_0^{\infty} e^{-st} e^{at} \, dt = \int_0^{\infty} e^{-(s-a)t} \, dt = \frac{1}{s - a}, \quad s > a$$

**예제 3: $f(t) = t^n$ (멱함수)**

반복 부분적분(또는 감마 함수 인식)에 의해:

$$\mathcal{L}\{t^n\} = \frac{n!}{s^{n+1}}, \quad s > 0, \quad n = 0, 1, 2, \ldots$$

## 주요 라플라스 쌍 표

이 표는 필수 참조표이다. 모든 항목은 정의로부터 유도할 수 있지만, 암기하거나(또는 가까이 두면) 문제 풀이가 크게 빨라진다.

| $f(t)$ | $F(s) = \mathcal{L}\{f(t)\}$ | 수렴 영역 |
|---------|-------------------------------|----------------------|
| $1$ | $\dfrac{1}{s}$ | $s > 0$ |
| $t^n$ | $\dfrac{n!}{s^{n+1}}$ | $s > 0$ |
| $e^{at}$ | $\dfrac{1}{s-a}$ | $s > a$ |
| $\sin(\omega t)$ | $\dfrac{\omega}{s^2 + \omega^2}$ | $s > 0$ |
| $\cos(\omega t)$ | $\dfrac{s}{s^2 + \omega^2}$ | $s > 0$ |
| $e^{at}\sin(\omega t)$ | $\dfrac{\omega}{(s-a)^2 + \omega^2}$ | $s > a$ |
| $e^{at}\cos(\omega t)$ | $\dfrac{s-a}{(s-a)^2 + \omega^2}$ | $s > a$ |
| $t \cdot e^{at}$ | $\dfrac{1}{(s-a)^2}$ | $s > a$ |
| $u(t-a)$ (헤비사이드) | $\dfrac{e^{-as}}{s}$ | $s > 0$ |
| $\delta(t-a)$ (디랙) | $e^{-as}$ | 모든 $s$ |

## 핵심 성질

### 1. 선형성(Linearity)

$$\mathcal{L}\{af(t) + bg(t)\} = aF(s) + bG(s)$$

이것은 적분의 선형성에서 직접 물려받는다. ODE의 각 항을 개별적으로 변환할 수 있다는 것을 의미한다.

### 2. 제1이동 정리(s-이동)(First Shifting Theorem)

$$\mathcal{L}\{e^{at}f(t)\} = F(s - a)$$

시간 영역에서 $e^{at}$를 곱하면 $s$-영역에서 $F(s)$가 $a$만큼 이동한다. $\mathcal{L}\{e^{at}\cos(\omega t)\}$에서 $s$가 $(s-a)$로 대체되는 이유이다.

### 3. 제2이동 정리(t-이동)(Second Shifting Theorem)

$$\mathcal{L}\{u(t-a)f(t-a)\} = e^{-as}F(s)$$

여기서 $u(t-a)$는 헤비사이드 계단 함수이다. $a$초의 시간 지연은 변환에 $e^{-as}$를 곱한다.

### 4. 미분 성질 (ODE 풀이의 핵심)

$$\mathcal{L}\{f'(t)\} = sF(s) - f(0)$$

$$\mathcal{L}\{f''(t)\} = s^2 F(s) - sf(0) - f'(0)$$

더 일반적으로:

$$\mathcal{L}\{f^{(n)}(t)\} = s^n F(s) - s^{n-1}f(0) - s^{n-2}f'(0) - \cdots - f^{(n-1)}(0)$$

이것이 **핵심 성질**이다: 도함수가 $s$의 곱셈이 되고, 초기 조건이 자동으로 나타난다. 라플라스 변환이 초기값 문제에 최적화된 이유이다.

### 5. 적분 성질

$$\mathcal{L}\left\{\int_0^t f(\tau) \, d\tau\right\} = \frac{F(s)}{s}$$

시간에서의 적분은 주파수 영역에서 $s$로 나누는 것에 대응한다.

### 6. 합성곱 정리(Convolution Theorem)

$$\mathcal{L}\{(f * g)(t)\} = F(s) \cdot G(s)$$

여기서 합성곱은 $(f * g)(t) = \int_0^t f(\tau)g(t - \tau) \, d\tau$이다. $s$-영역에서의 곱셈은 시간에서의 합성곱에 대응한다. 곱의 역변환에 유용하다.

## 역 라플라스 변환(Inverse Laplace Transform)

역 라플라스 변환은 $F(s)$로부터 $f(t)$를 복원한다:

$$f(t) = \mathcal{L}^{-1}\{F(s)\}$$

실제로 복소 적분 공식을 거의 사용하지 않는다. 대신:

1. **표 참조**: $F(s)$를 알려진 쌍과 매칭한다
2. **부분분수 분해**: 유리 $F(s)$를 더 간단한 항으로 분해한다
3. **완전제곱식**: 분모의 기약 이차식에 대해 사용한다

### 부분분수법

$F(s) = \frac{P(s)}{Q(s)}$ ($\deg(P) < \deg(Q)$)가 주어졌을 때:

**서로 다른 실근**: $Q(s) = (s - r_1)(s - r_2) \cdots (s - r_n)$이면:

$$F(s) = \frac{A_1}{s - r_1} + \frac{A_2}{s - r_2} + \cdots + \frac{A_n}{s - r_n}$$

각 항은 $A_k e^{r_k t}$로 역변환된다.

**중근**: $Q(s)$가 $(s - r)^m$을 가지면:

$$\frac{A_1}{s - r} + \frac{A_2}{(s - r)^2} + \cdots + \frac{A_m}{(s - r)^m}$$

**복소근**: 기약 $s^2 + bs + c$에 대해, 완전제곱식을 만들고 이동된 사인/코사인 형태와 매칭한다.

### 풀이 예제: 역변환

$\mathcal{L}^{-1}\left\{\dfrac{5s + 3}{s^2 + 4s + 13}\right\}$를 구하라.

**1단계**: 분모를 완전제곱식으로:

$$s^2 + 4s + 13 = (s + 2)^2 + 9 = (s + 2)^2 + 3^2$$

**2단계**: 분자를 이동된 형태에 맞게 다시 쓰기:

$$\frac{5s + 3}{(s+2)^2 + 9} = \frac{5(s+2) - 7}{(s+2)^2 + 9} = 5\cdot\frac{s+2}{(s+2)^2 + 9} - \frac{7}{3}\cdot\frac{3}{(s+2)^2 + 9}$$

**3단계**: 표를 사용하여 역변환:

$$f(t) = 5e^{-2t}\cos(3t) - \frac{7}{3}e^{-2t}\sin(3t)$$

## 라플라스 변환으로 IVP 풀기

### 체계적 절차

1. ODE 양변의 라플라스 변환을 취한다
2. 초기 조건을 대입한다 (미분 성질에서 자동으로 나타남)
3. $Y(s)$에 대해 대수적으로 풀다
4. 부분분수와 표 참조를 사용하여 $y(t) = \mathcal{L}^{-1}\{Y(s)\}$를 구한다

### 풀이 예제: 2계 IVP

풀기: $y'' + 5y' + 6y = 2e^{-t}$, $y(0) = 1$, $y'(0) = 0$.

**1단계**: 양변을 변환.

$$[s^2 Y - sy(0) - y'(0)] + 5[sY - y(0)] + 6Y = \frac{2}{s + 1}$$

**2단계**: $y(0) = 1$, $y'(0) = 0$을 대입:

$$s^2 Y - s + 5sY - 5 + 6Y = \frac{2}{s+1}$$

$$(s^2 + 5s + 6)Y = \frac{2}{s+1} + s + 5$$

**3단계**: 인수분해하고 $Y$에 대해 풀기:

$$(s+2)(s+3)Y = \frac{2 + (s+5)(s+1)}{s+1} = \frac{s^2 + 6s + 7}{s+1}$$

$$Y(s) = \frac{s^2 + 6s + 7}{(s+1)(s+2)(s+3)}$$

**4단계**: 부분분수:

$$\frac{s^2 + 6s + 7}{(s+1)(s+2)(s+3)} = \frac{A}{s+1} + \frac{B}{s+2} + \frac{C}{s+3}$$

$s = -1$을 넣으면: $A = \frac{1 - 6 + 7}{(1)(2)} = 1$

$s = -2$를 넣으면: $B = \frac{4 - 12 + 7}{(-2+1)(-2+3)} = \frac{-1}{(-1)(1)} = 1$

$s = -3$을 넣으면: $C = \frac{9 - 18 + 7}{(-3+1)(-3+2)} = \frac{-2}{(-2)(-1)} = \frac{-2}{2} = -1$

$$Y(s) = \frac{1}{s+1} + \frac{1}{s+2} - \frac{1}{s+3}$$

**5단계**: 역변환:

$$y(t) = e^{-t} + e^{-2t} - e^{-3t}$$

**검증**: $t = 0$에서: $y(0) = 1 + 1 - 1 = 1$ (정확). $y'(0) = -1 - 2 + 3 = 0$ 계산 (정확).

## 헤비사이드 계단 함수(Heaviside Step Function)

**헤비사이드 계단 함수**(또는 단위 계단 함수)는 갑작스러운 스위칭을 모델링한다:

$$u(t - a) = \begin{cases} 0, & t < a \\ 1, & t \geq a \end{cases}$$

시간 $t = a$에서 "켜진다". 임의의 구분적 함수(piecewise function)는 계단 함수를 사용하여 쓸 수 있다:

$$f(t) = \begin{cases} 0, & t < 1 \\ 3, & 1 \leq t < 4 \\ 0, & t \geq 4 \end{cases} = 3[u(t-1) - u(t-4)]$$

**라플라스 변환**: $\mathcal{L}\{u(t-a)\} = \dfrac{e^{-as}}{s}$

이동된 함수에 대해: $\mathcal{L}\{u(t-a)f(t-a)\} = e^{-as}F(s)$ (제2이동 정리).

## 디랙 델타 함수(Dirac Delta Function)

**디랙 델타 함수(Dirac delta function)** $\delta(t - a)$는 시간 $t = a$에서의 순간적 충격을 모델링한다:

- $\delta(t - a) = 0$ ($t \neq a$일 때)
- $\int_{-\infty}^{\infty} \delta(t - a) f(t) \, dt = f(a)$ (체거름 성질, sifting property)

전체 넓이가 1인 매우 높고 매우 좁은 펄스의 극한으로 생각하라. 망치 타격, 갑작스러운 전압 스파이크, 또는 점 하중을 모델링한다.

**라플라스 변환**: $\mathcal{L}\{\delta(t - a)\} = e^{-as}$

$a = 0$일 때: $\mathcal{L}\{\delta(t)\} = 1$. 이것은 시스템의 충격 응답이 전달 함수를 직접 역변환하여 얻어진다는 것을 의미한다.

## 전달 함수 (소개)(Transfer Functions)

영 초기 조건을 가진 선형 상수계수 ODE에 대해:

$$a_n y^{(n)} + \cdots + a_1 y' + a_0 y = f(t)$$

**전달 함수(transfer function)**는:

$$H(s) = \frac{Y(s)}{F(s)} = \frac{1}{a_n s^n + \cdots + a_1 s + a_0}$$

$H(s)$는 입력과 독립적으로 시스템을 특성화한다. 임의의 입력에 대한 출력은 $Y(s) = H(s) F(s)$이고, 시간 영역에서는 합성곱 $y(t) = (h * f)(t)$이다. 여기서 $h(t) = \mathcal{L}^{-1}\{H(s)\}$는 **충격 응답(impulse response)**이다.

이 개념은 제어이론(Control_Theory 토픽 참조)과 신호 처리(Signal_Processing 토픽 참조)의 핵심이다.

## Python 구현

```python
"""
Laplace Transform for solving ODE using SymPy.

This script demonstrates:
1. Computing Laplace transforms symbolically
2. Solving an IVP via the Laplace method
3. Visualizing the solution and step-function forcing
"""

import numpy as np
import matplotlib.pyplot as plt
from sympy import (
    symbols, Function, laplace_transform, inverse_laplace_transform,
    exp, sin, cos, Heaviside, DiracDelta, apart, oo,
    dsolve, Eq, pprint, simplify
)

# Define symbolic variables
t, s = symbols('t s', positive=True)
# positive=True helps SymPy assume t > 0 for Laplace integrals

# ── 1. Basic Laplace Transforms ──────────────────────────────
print("=== Basic Laplace Transforms ===\n")

functions = [1, t, t**2, exp(-3*t), sin(2*t), cos(2*t)]
for f in functions:
    # laplace_transform returns (transform, convergence_region, condition)
    F, region, cond = laplace_transform(f, t, s)
    print(f"L{{{f}}} = {F},  (converges for {cond})")

# ── 2. Properties demonstration ──────────────────────────────
print("\n=== Shifting Property ===")
# L{e^{at} f(t)} = F(s-a)
# L{e^{-2t} sin(3t)} should be 3/((s+2)^2 + 9)
f_shifted = exp(-2*t) * sin(3*t)
F_shifted, _, _ = laplace_transform(f_shifted, t, s)
print(f"L{{e^(-2t) sin(3t)}} = {F_shifted}")

# ── 3. Solving an IVP with Laplace Transform ────────────────
print("\n=== Solving IVP: y'' + 5y' + 6y = 2e^(-t), y(0)=1, y'(0)=0 ===\n")

# Method: Manual algebraic approach
# After transforming: Y(s) = (s^2 + 6s + 7) / ((s+1)(s+2)(s+3))
Y_s = (s**2 + 6*s + 7) / ((s + 1) * (s + 2) * (s + 3))

# Partial fraction decomposition — apart() does the heavy lifting
Y_partial = apart(Y_s, s)
print(f"Y(s) = {Y_s}")
print(f"Partial fractions: Y(s) = {Y_partial}")

# Inverse Laplace transform to get y(t)
y_solution = inverse_laplace_transform(Y_s, s, t)
print(f"y(t) = {simplify(y_solution)}")

# ── 4. Verify with dsolve (direct ODE solver) ───────────────
print("\n=== Verification with dsolve ===")
y = Function('y')
t_sym = symbols('t')
ode = Eq(y(t_sym).diff(t_sym, 2) + 5*y(t_sym).diff(t_sym) + 6*y(t_sym),
         2*exp(-t_sym))
# dsolve finds the general solution; we apply initial conditions
sol = dsolve(ode, y(t_sym), ics={y(0): 1, y(t_sym).diff(t_sym).subs(t_sym, 0): 0})
print(f"dsolve result: {sol}")

# ── 5. Visualization ─────────────────────────────────────────
t_vals = np.linspace(0, 5, 300)

# Our Laplace-derived solution: y(t) = e^{-t} + e^{-2t} - e^{-3t}
y_vals = np.exp(-t_vals) + np.exp(-2*t_vals) - np.exp(-3*t_vals)

# Forcing function: 2e^{-t}
f_vals = 2 * np.exp(-t_vals)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot the solution
axes[0].plot(t_vals, y_vals, 'b-', linewidth=2, label=r'$y(t) = e^{-t} + e^{-2t} - e^{-3t}$')
axes[0].axhline(y=0, color='gray', linewidth=0.5)
axes[0].set_xlabel('t')
axes[0].set_ylabel('y(t)')
axes[0].set_title("Solution of y'' + 5y' + 6y = 2e^{-t}")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot with Heaviside forcing
# Solve y'' + 4y = u(t-2), y(0)=0, y'(0)=0
# The step function turns on a constant force at t=2
t_vals2 = np.linspace(0, 10, 500)
# Analytical solution using Laplace:
#   Y(s) = e^{-2s} / (s(s^2+4))
#   y(t) = u(t-2) * [1/4 - (1/4)cos(2(t-2))]
y_step = np.where(t_vals2 >= 2,
                  0.25 * (1 - np.cos(2*(t_vals2 - 2))),
                  0.0)
# The forcing function
f_step = np.where(t_vals2 >= 2, 1.0, 0.0)

axes[1].plot(t_vals2, y_step, 'b-', linewidth=2, label=r'$y(t)$')
axes[1].plot(t_vals2, f_step, 'r--', linewidth=1.5, alpha=0.6, label=r'$u(t-2)$ (forcing)')
axes[1].axhline(y=0, color='gray', linewidth=0.5)
axes[1].axvline(x=2, color='gray', linewidth=0.5, linestyle=':')
axes[1].set_xlabel('t')
axes[1].set_ylabel('y(t)')
axes[1].set_title("Response to Step Function: y'' + 4y = u(t-2)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('laplace_transform_solutions.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved to laplace_transform_solutions.png")
```

## 요약

라플라스 변환은 초기값 문제를 풀기 위한 강력한 대수적 도구이다:

| 개념 | 핵심 아이디어 |
|---------|----------|
| 정의 | $F(s) = \int_0^\infty e^{-st} f(t) \, dt$ |
| 미분 | 도함수가 $s$의 거듭제곱이 됨; 초기 조건이 자동으로 나타남 |
| 역변환 | 부분분수 + 표 참조 |
| 계단 함수 | 갑작스러운 스위치를 모델링; $e^{-as}$ 이동 사용 |
| 델타 함수 | 충격을 모델링; $\mathcal{L}\{\delta(t)\} = 1$ |
| 전달 함수 | $H(s) = Y(s)/F(s)$로 시스템을 특성화 |

브로미치 적분(Bromwich integral)과 복소 해석에서의 응용을 포함한 라플라스 변환의 더 심화된 다룸은 [Mathematical Methods - 라플라스 변환](../Mathematical_Methods/15_Laplace_Transform.md)을 참조하라.

## 연습 문제

1. **기본 변환**: 성질(적분 정의가 아닌)을 사용하여 $\mathcal{L}\{t^3 e^{-2t}\}$와 $\mathcal{L}\{e^{3t}\cos(4t)\}$를 계산하라. SymPy로 답을 검증하라.

2. **역변환**: 부분분수를 사용하여 $\mathcal{L}^{-1}\left\{\dfrac{3s + 7}{(s+1)(s^2 + 4)}\right\}$를 구하라. 해의 과도 부분과 진동 부분을 식별하라.

3. **IVP 풀기**: 라플라스 변환을 사용하여 $y'' + 4y' + 4y = e^{-2t}$, $y(0) = 0$, $y'(0) = 1$을 풀어라. 분모에 중근이 있다는 점에 주목하라 -- 이것이 부분분수에 어떤 영향을 미치는가?

4. **계단 함수**: 스프링-질량 시스템이 $y'' + 9y = 5u(t - \pi)$, $y(0) = 0$, $y'(0) = 0$을 만족한다. 응답 $y(t)$를 구하라. 해를 스케치하고 $t = \pi$에서 물리적으로 무슨 일이 일어나는지 설명하라.

5. **충격 응답**: 시스템 $y'' + 2y' + 5y = \delta(t)$, $y(0^-) = 0$, $y'(0^-) = 0$에 대해, 충격 응답 $h(t)$를 구하라. 그런 다음 합성곱을 사용하여 (ODE를 다시 풀지 않고) $f(t) = e^{-t}$에 대한 응답을 구하라.

---

*이전: [상미분방정식의 연립](./14_Systems_of_ODE.md) | 다음: [멱급수 해법](./16_Power_Series_Solutions.md)*
