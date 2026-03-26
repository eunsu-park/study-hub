# 도함수의 기초

## 학습 목표

- 도함수(derivative)를 차분 몫(difference quotient)의 극한으로 **정의**하고 이를 순간 변화율로 해석할 수 있다
- 미분 규칙(거듭제곱, 곱, 몫, 연쇄)을 **적용**하여 합성 함수의 도함수를 계산할 수 있다
- $y$에 대해 풀리지 않은 방정식에 대해 음함수 미분(implicit differentiation)을 **수행**할 수 있다
- 고계 도함수를 **계산**하고 그 물리적 의미(속도, 가속도, 저크)를 이해할 수 있다
- Python으로 수치적 미분과 기호적 미분을 모두 **구현**하고 정확도를 비교할 수 있다

## 소개

미적분학이 변화의 수학이라면, 도함수(derivative)는 그 가장 기본적인 도구이다. 이렇게 생각해 보자: 자동차의 속도계는 전체 여정의 평균 속도를 계산하지 않는다 -- 바로 *지금*, 이 정확한 순간에 얼마나 빠르게 달리고 있는지를 알려준다. 도함수는 이 "순간 변화율"이라는 개념을 형식화한다.

역사적으로 뉴턴(Newton)과 라이프니츠(Leibniz)는 17세기 후반에 독립적으로 도함수를 개발했다. 뉴턴은 이를 "유율(fluxions)" (흐르는 양의 변화율)로 생각했고, 라이프니츠는 오늘날에도 사용하는 $\frac{dy}{dx}$ 표기법을 개발했다. 둘 다 같은 실용적 문제를 풀려고 했다: 행성은 어떻게 움직이며, 곡선은 어떻게 행동하는가?

## 극한으로서의 도함수

### 차분 몫

함수 $f(x)$가 주어졌을 때, $x = a$와 $x = a + h$ 사이의 **평균 변화율**은:

$$\frac{f(a + h) - f(a)}{h}$$

이것은 점 $(a, f(a))$와 $(a+h, f(a+h))$를 연결하는 **할선(secant line)** 의 기울기이다.

### 도함수의 정의

$x = a$에서 $f$의 **도함수**는 $h \to 0$일 때 차분 몫의 극한이다:

$$f'(a) = \lim_{h \to 0} \frac{f(a + h) - f(a)}{h}$$

- $f'(a)$: $a$에서 $f$의 도함수 (뉴턴의 프라임 표기법)
- $\frac{df}{dx}\bigg|_{x=a}$: 라이프니츠 표기법으로 같은 것
- $h$: 0으로 줄어드는 작은 증분
- 극한은 *평균* 변화율을 *순간* 변화율로 변환한다

**기하학적 해석:** 도함수 $f'(a)$는 점 $(a, f(a))$에서 $f$의 그래프에 대한 **접선(tangent line)** 의 기울기이다.

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_derivative(f, a, h_values=[1.0, 0.5, 0.1], x_range=(-1, 4)):
    """
    Show how secant lines approach the tangent line as h -> 0.

    This visualization makes the limit definition concrete: each secant
    line connects two points on the curve, and as h shrinks, the secant
    rotates to become the tangent.
    """
    x = np.linspace(*x_range, 500)
    y = f(x)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(x, y, 'b-', linewidth=2, label='$f(x) = x^2$')

    colors = ['red', 'orange', 'green']
    for h, color in zip(h_values, colors):
        # Slope of secant line through (a, f(a)) and (a+h, f(a+h))
        slope = (f(a + h) - f(a)) / h
        # Equation of secant line: y - f(a) = slope * (x - a)
        y_secant = f(a) + slope * (x - a)
        ax.plot(x, y_secant, '--', color=color, linewidth=1.5,
                label=f'Secant (h={h}), slope={slope:.2f}')
        ax.plot([a, a + h], [f(a), f(a + h)], 'o', color=color, markersize=6)

    # True tangent line (derivative of x^2 at x=a is 2a)
    true_slope = 2 * a
    y_tangent = f(a) + true_slope * (x - a)
    ax.plot(x, y_tangent, 'k-', linewidth=2.5,
            label=f'Tangent, slope={true_slope:.2f}')
    ax.plot(a, f(a), 'ko', markersize=8, zorder=5)

    ax.set_xlim(*x_range)
    ax.set_ylim(-2, 12)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$f(x)$')
    ax.set_title(f'Secant lines approaching tangent at $x = {a}$')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('derivative_visualization.png', dpi=150)
    plt.show()

f = lambda x: x**2
visualize_derivative(f, a=1.5, h_values=[1.5, 0.8, 0.3, 0.05])
```

## 기본 미분 규칙

매번 극한 정의에서 도함수를 계산하는 것은 번거로울 것이다. 다행히도, 소수의 규칙이 우리가 만나는 대부분의 함수를 다룬다.

### 거듭제곱 법칙 (Power Rule)

$$\frac{d}{dx} x^n = n x^{n-1}$$

여기서 $n$은 임의의 실수이다. 이 단일 규칙이 다항식, 근($x^{1/2}$), 역수($x^{-1}$)를 처리한다.

**예시:**
- $\frac{d}{dx} x^3 = 3x^2$
- $\frac{d}{dx} \sqrt{x} = \frac{d}{dx} x^{1/2} = \frac{1}{2} x^{-1/2} = \frac{1}{2\sqrt{x}}$
- $\frac{d}{dx} \frac{1}{x^2} = \frac{d}{dx} x^{-2} = -2x^{-3} = \frac{-2}{x^3}$

### 합과 상수배 법칙

$$\frac{d}{dx} [f(x) + g(x)] = f'(x) + g'(x) \qquad \frac{d}{dx} [c \cdot f(x)] = c \cdot f'(x)$$

미분은 **선형(linear)** 이다: 항별로 미분할 수 있고 상수를 밖으로 꺼낼 수 있다.

### 곱 법칙 (Product Rule)

두 함수가 곱해졌을 때, 그 도함수는 단순히 도함수의 곱이 아니다:

$$\frac{d}{dx} [f(x) \cdot g(x)] = f'(x) \cdot g(x) + f(x) \cdot g'(x)$$

**기억법:** "첫째 곱하기 둘째의 도함수, 더하기 둘째 곱하기 첫째의 도함수."

**예시:** $\frac{d}{dx} [x^2 \sin x] = 2x \sin x + x^2 \cos x$

### 몫 법칙 (Quotient Rule)

$$\frac{d}{dx} \left[\frac{f(x)}{g(x)}\right] = \frac{f'(x) g(x) - f(x) g'(x)}{[g(x)]^2}$$

**기억법:** "아래 곱하기 위의 미분 빼기 위 곱하기 아래의 미분, 아래의 제곱으로 나눈다."

**예시:**
$$\frac{d}{dx} \left[\frac{x^2}{x+1}\right] = \frac{2x(x+1) - x^2(1)}{(x+1)^2} = \frac{x^2 + 2x}{(x+1)^2}$$

### 연쇄 법칙 (Chain Rule)

연쇄 법칙은 미적분학에서 가장 중요한 규칙이라 할 수 있으며, 특히 기계 학습에서 중요하다 (역전파(backpropagation)는 재귀적으로 적용된 연쇄 법칙이다).

$y = f(g(x))$이면:

$$\frac{dy}{dx} = f'(g(x)) \cdot g'(x)$$

라이프니츠 표기법으로, $y = f(u)$이고 $u = g(x)$이면:

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

**직관:** $u$가 $x$보다 3배 빠르게 변하고, $y$가 $u$보다 5배 빠르게 변하면, $y$는 $x$보다 $15 = 3 \times 5$배 빠르게 변한다. 변화율은 합성의 "사슬"을 통해 곱해진다.

**예시:**
- $\frac{d}{dx} (3x + 1)^5 = 5(3x+1)^4 \cdot 3 = 15(3x+1)^4$
- $\frac{d}{dx} \sin(x^2) = \cos(x^2) \cdot 2x$
- $\frac{d}{dx} e^{-x^2} = e^{-x^2} \cdot (-2x)$

### 기본 도함수 표

| 함수 $f(x)$ | 도함수 $f'(x)$ |
|------------------|---------------------|
| $x^n$ | $nx^{n-1}$ |
| $e^x$ | $e^x$ |
| $\ln x$ | $1/x$ |
| $\sin x$ | $\cos x$ |
| $\cos x$ | $-\sin x$ |
| $\tan x$ | $\sec^2 x$ |
| $a^x$ | $a^x \ln a$ |
| $\arcsin x$ | $1/\sqrt{1-x^2}$ |
| $\arctan x$ | $1/(1+x^2)$ |

## 음함수 미분

모든 관계가 $y = f(x)$로 주어지는 것은 아니다. 예를 들어, 원의 방정식은:

$$x^2 + y^2 = 25$$

여기서 $y$는 $x$의 함수로 명시적으로 쓰여 있지 않다. $\frac{dy}{dx}$를 구하기 위해, $y$를 $x$의 음함수로 취급하면서 양변을 $x$에 대해 미분한다:

$$\frac{d}{dx}(x^2) + \frac{d}{dx}(y^2) = \frac{d}{dx}(25)$$

$$2x + 2y \frac{dy}{dx} = 0$$

$$\frac{dy}{dx} = -\frac{x}{y}$$

이것은 원 위의 임의의 점 $(x, y)$에서의 기울기를 알려준다. $(3, 4)$에서: 기울기 $= -3/4$. $(0, 5)$에서: 기울기 $= 0$ (원의 꼭대기, 예상대로).

```python
import sympy as sp

# Implicit differentiation with SymPy
x, y = sp.symbols('x y')

# Circle: x^2 + y^2 = 25
# SymPy's idiff handles implicit differentiation
circle_eq = x**2 + y**2 - 25

# Method 1: Using idiff (implicit differentiation)
dydx = sp.idiff(circle_eq, y, x)
print(f"dy/dx for circle: {dydx}")
# Output: -x/y

# Method 2: Manual approach -- differentiate and solve
# Treat y as a function of x: y(x)
y_func = sp.Function('y')(x)
eq_implicit = x**2 + y_func**2 - 25

# Differentiate with respect to x
diff_eq = sp.diff(eq_implicit, x)
print(f"After differentiating: {diff_eq}")

# Solve for dy/dx
dydx_manual = sp.solve(diff_eq, sp.diff(y_func, x))[0]
print(f"dy/dx (manual): {dydx_manual}")
```

## 고계 도함수

도함수의 도함수는 **2계 도함수(second derivative)** 로, $f''(x)$ 또는 $\frac{d^2y}{dx^2}$로 쓴다:

$$f''(x) = \frac{d}{dx}\left[f'(x)\right]$$

**위치 $s(t)$에 대한 물리적 해석:**
- $s'(t) = v(t)$: 속도(velocity) (위치 변화의 비율)
- $s''(t) = v'(t) = a(t)$: 가속도(acceleration) (속도 변화의 비율)
- $s'''(t) = a'(t) = j(t)$: 저크(jerk) (가속도 변화의 비율 -- 엘리베이터에서 느끼는 것)

```python
import sympy as sp

x = sp.Symbol('x')

# Compute derivatives of increasing order
f = sp.sin(x) * sp.exp(-x)
print(f"f(x) = {f}")
print(f"f'(x) = {sp.diff(f, x)}")
print(f"f''(x) = {sp.simplify(sp.diff(f, x, 2))}")
print(f"f'''(x) = {sp.simplify(sp.diff(f, x, 3))}")
print(f"f''''(x) = {sp.simplify(sp.diff(f, x, 4))}")

# Note: the nth derivative of sin(x)*exp(-x) has a beautiful pattern
# related to complex exponentials, which we explore in later lessons
```

## 수치적 미분 대 기호적 미분

실제로 도함수를 계산하는 두 가지 접근법을 사용한다:

### 전방 차분 (Forward Difference, 수치적)

$$f'(x) \approx \frac{f(x + h) - f(x)}{h}$$

오차: $O(h)$ -- 정확도가 작은 $h$에 대해 선형적으로 향상된다.

### 중심 차분 (Central Difference, 수치적)

$$f'(x) \approx \frac{f(x + h) - f(x - h)}{2h}$$

오차: $O(h^2)$ -- 같은 $h$에서 훨씬 더 정확하다.

### 기호적 (정확)

SymPy와 같은 라이브러리가 대수적 표현을 조작하여 정확한 도함수를 구한다.

```python
import numpy as np
import sympy as sp

# Compare numerical and symbolic approaches for f(x) = sin(x) at x = 1
def f_numeric(x):
    return np.sin(x)

x_val = 1.0
exact = np.cos(1.0)  # True derivative of sin(x) is cos(x)

# Numerical derivatives with decreasing h
print(f"{'h':<12} {'Forward':>14} {'Central':>14} {'Fwd Error':>12} {'Ctr Error':>12}")
print("-" * 66)
for h in [0.1, 0.01, 0.001, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]:
    fwd = (f_numeric(x_val + h) - f_numeric(x_val)) / h
    ctr = (f_numeric(x_val + h) - f_numeric(x_val - h)) / (2 * h)
    print(f"{h:<12.0e} {fwd:>14.10f} {ctr:>14.10f} "
          f"{abs(fwd - exact):>12.2e} {abs(ctr - exact):>12.2e}")

# Note: Very small h causes floating-point cancellation errors.
# For h=1e-12, the central difference becomes LESS accurate.
# This demonstrates the tension between approximation error and
# machine precision -- a key theme in numerical computing.

# Symbolic (exact) approach
x = sp.Symbol('x')
f_sym = sp.sin(x)
df_sym = sp.diff(f_sym, x)  # cos(x) -- exact
print(f"\nSymbolic derivative: {df_sym}")
print(f"Evaluated at x=1: {float(df_sym.subs(x, 1)):.15f}")
print(f"Exact cos(1):      {exact:.15f}")
```

## 요약

- **도함수(derivative)** $f'(a)$는 $\lim_{h \to 0} \frac{f(a+h) - f(a)}{h}$로 정의된다 -- 순간 변화율
- **미분 규칙** (거듭제곱, 곱, 몫, 연쇄)은 매번 극한을 계산할 필요를 없앤다
- **연쇄 법칙(chain rule)** 은 특히 중요하다: 합성 함수를 미분하는 방법을 알려주며, 신경망(neural network)에서 역전파(backpropagation)의 수학적 기초이다
- **음함수 미분(implicit differentiation)** 은 $y$가 분리되지 않은 방정식을 처리한다
- **고계 도함수(higher-order derivatives)** 는 가속도, 곡률, 더 높은 차수의 변화율을 포착한다
- **수치적 미분** (전방/중심 차분)은 함수값으로부터 도함수를 근사하지만, 매우 작은 $h$에서는 부동소수점 오차가 정확도를 제한한다
- **기호적 미분** (SymPy)은 정확한 결과를 제공하지만 대수적 표현이 필요하다

## 연습 문제

### 문제 1: 미분 규칙 적용

각 함수의 도함수를 계산하라:

(a) $f(x) = 3x^4 - 2x^3 + 7x - 9$

(b) $g(x) = x^2 e^x \sin x$ (곱 법칙을 두 번 사용하라)

(c) $h(x) = \frac{\ln x}{x^2 + 1}$

(d) $k(x) = \cos(\sqrt{x^2 + 1})$ (연쇄 법칙을 주의 깊게 사용하라 -- 세 개의 중첩된 함수가 있다)

### 문제 2: 음함수 미분

타원 $\frac{x^2}{9} + \frac{y^2}{4} = 1$에 대해 $\frac{dy}{dx}$를 구하라. 접선이 수평인 점은 어디인가? 수직인 점은?

### 문제 3: 수치적 정확도 조사

$f(x) = e^x$의 $x = 0$에서의 도함수를 다음을 사용하여 계산하는 Python 스크립트를 작성하라:
- $k = 1, 2, \ldots, 16$에 대해 $h = 10^{-k}$인 전방 차분
- 같은 $h$ 값에 대한 중심 차분

절대 오차 대 $h$를 로그-로그 스케일로 그려라. $h$가 매우 작아지면 왜 오차가 감소했다가 다시 증가하는지 설명하라.

### 문제 4: 제일 원리에서 도함수

극한 정의만을 사용하여 (규칙은 사용하지 않고) $\frac{d}{dx}[\sin x] = \cos x$임을 증명하라.

(힌트: 각도 덧셈 공식 $\sin(a+b) = \sin a \cos b + \cos a \sin b$와 극한 $\lim_{h \to 0} \frac{\sin h}{h} = 1$ 및 $\lim_{h \to 0} \frac{\cos h - 1}{h} = 0$이 필요하다.)

### 문제 5: 기계 학습에서의 연쇄 법칙

간단한 신경망에서, 손실 함수는 $L = (y - \hat{y})^2$이며 여기서 $\hat{y} = \sigma(wx + b)$이고 $\sigma(z) = \frac{1}{1+e^{-z}}$는 시그모이드 함수(sigmoid function)이다.

(a) 연쇄 법칙을 사용하여 $\frac{\partial L}{\partial w}$를 계산하라.

(b) 기호적 표현을 정의하고 미분하여 SymPy로 답을 검증하라.

## 참고 자료

- Stewart, *Calculus: Early Transcendentals*, Ch. 3 (Differentiation Rules)
- [3Blue1Brown: Derivative formulas through geometry](https://www.youtube.com/watch?v=S0_qX4VJhMQ)
- [MIT OCW 18.01: Derivatives](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/pages/1.-differentiation/)

---

[이전: 극한과 연속](./01_Limits_and_Continuity.md) | [다음: 도함수의 응용](./03_Applications_of_Derivatives.md)
