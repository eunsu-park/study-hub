# 극한과 연속

## 학습 목표

- 엡실론-델타(epsilon-delta) 공식화를 사용하여 함수의 극한(limit)을 **정의**하고 그 기하학적 의미를 설명할 수 있다
- 대수적 기법, 조임 정리(Squeeze Theorem), 로피탈의 법칙(L'Hopital's rule, 미리보기)을 사용하여 극한을 **계산**할 수 있다
- 불연속(discontinuity)의 유형(제거 가능, 점프, 무한)을 **분류**하고 함수가 연속인 곳을 결정할 수 있다
- 중간값 정리(Intermediate Value Theorem)를 **적용**하여 해의 존재성을 증명할 수 있다
- Python으로 수치적 극한 추정과 엡실론-델타 시각화를 **구현**할 수 있다

## 소개

터널 입구를 향해 차를 운전하는 상황을 상상해 보자. 터널에 도달하기 *이전* 의 모든 순간에서 당신의 위치를 설명할 수 있으며, 도착했을 때 정확히 어디에 있을지 예측할 수 있다 -- 어떤 이유로 실제로 들어가지 않더라도. 이것이 극한(limit)의 본질이다: 극한은 입력이 특정 점에 접근할 때 함수가 *다가가는* 값을 설명하며, 함수가 실제로 그 점에서 정의되어 있는지 여부와는 무관하다.

극한은 미적분학의 기초 개념이다. 모든 도함수(derivative)는 극한으로 정의된다. 모든 적분(integral)은 극한으로 정의된다. 극한을 엄밀하게 이해하는 것이 미적분학을 단순한 대수적 조작과 구별짓는 것이다.

## 극한의 직관적 개념

다음 함수를 생각해 보자:

$$f(x) = \frac{x^2 - 1}{x - 1}$$

$x = 1$에서 이 함수는 정의되지 않는다(0으로 나누기). 하지만 우리는 물을 수 있다: $x$가 1에 점점 더 가까워질 때 $f(x)$는 어떤 값에 접근하는가?

분자를 인수분해하면: $f(x) = \frac{(x-1)(x+1)}{x-1} = x + 1$ ($x \neq 1$일 때).

$x \to 1$이면, $f(x) \to 2$이다. 다음과 같이 쓴다:

$$\lim_{x \to 1} \frac{x^2 - 1}{x - 1} = 2$$

```python
import numpy as np
import matplotlib.pyplot as plt

# Numerically approach x = 1 from both sides
x_left = 1 - np.array([0.1, 0.01, 0.001, 0.0001, 0.00001])
x_right = 1 + np.array([0.1, 0.01, 0.001, 0.0001, 0.00001])

# f(x) = (x^2 - 1) / (x - 1) -- undefined at x=1, but approaches 2
f = lambda x: (x**2 - 1) / (x - 1)

print("Approaching from the left:")
for x in x_left:
    print(f"  f({x:.5f}) = {f(x):.10f}")

print("\nApproaching from the right:")
for x in x_right:
    print(f"  f({x:.5f}) = {f(x):.10f}")
# Both sides converge to 2.0, confirming lim_{x->1} f(x) = 2
```

## 엡실론-델타 정의

"$x$가 $a$에 접근할 때 $f(x)$가 $L$에 접근한다"는 비형식적 문장은 엡실론-델타 정의에 의해 엄밀하게 만들어진다:

$$\lim_{x \to a} f(x) = L$$

이것은 다음을 의미한다: 모든 $\varepsilon > 0$에 대해, 다음을 만족하는 $\delta > 0$이 존재한다

$$0 < |x - a| < \delta \implies |f(x) - L| < \varepsilon$$

**기호의 의미:**
- $\varepsilon$ (엡실론): 허용 오차 -- $f(x)$가 $L$에 얼마나 가까워야 하는지
- $\delta$ (델타): 대응하는 제한 -- $x$가 $a$에 얼마나 가까워야 하는지
- $0 < |x - a|$: $x = a$ 자체에서는 절대 계산하지 않는다; $a$ *근처*만 살펴본다

**기하학적 해석:** $y = L$을 중심으로 폭 $2\varepsilon$인 수평 띠가 주어지면, $x$가 수직 띠 내에 있을 때($x = a$ 자체는 제외) $f$의 그래프가 수평 띠 안에 머무르도록 하는 $x = a$를 중심으로 한 폭 $2\delta$인 수직 띠를 찾을 수 있다.

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_epsilon_delta(f, a, L, epsilon, delta, x_range=(0, 3)):
    """
    Visualize the epsilon-delta definition of a limit.

    The shaded regions show the epsilon-band (horizontal, around L)
    and the delta-band (vertical, around a). If the function's graph
    stays inside the epsilon-band whenever x is in the delta-band,
    the limit condition is satisfied for these particular values.
    """
    x = np.linspace(*x_range, 1000)
    # Remove the point x=a to show it's excluded from consideration
    x = x[np.abs(x - a) > 0.001]
    y = f(x)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the function
    ax.plot(x, y, 'b-', linewidth=2, label='$f(x)$')

    # Epsilon band (horizontal)
    ax.axhspan(L - epsilon, L + epsilon, alpha=0.2, color='green',
               label=f'$\\varepsilon$-band: $({L-epsilon:.1f}, {L+epsilon:.1f})$')
    ax.axhline(y=L, color='green', linestyle='--', alpha=0.5)

    # Delta band (vertical)
    ax.axvspan(a - delta, a + delta, alpha=0.2, color='red',
               label=f'$\\delta$-band: $({a-delta:.2f}, {a+delta:.2f})$')
    ax.axvline(x=a, color='red', linestyle='--', alpha=0.5)

    # Mark the limit point (open circle since f may not be defined there)
    ax.plot(a, L, 'o', color='green', markersize=10, markerfacecolor='white',
            markeredgewidth=2, zorder=5)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$f(x)$')
    ax.set_title(f'$\\varepsilon$-$\\delta$ visualization: '
                 f'$\\varepsilon={epsilon}$, $\\delta={delta}$')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('epsilon_delta.png', dpi=150)
    plt.show()

# Example: lim_{x->1} (x^2 - 1)/(x - 1) = 2
f = lambda x: (x**2 - 1) / (x - 1)
visualize_epsilon_delta(f, a=1, L=2, epsilon=0.5, delta=0.4, x_range=(0, 3))
```

## 극한 법칙

$\lim_{x \to a} f(x) = L$이고 $\lim_{x \to a} g(x) = M$이면:

| 법칙 | 설명 |
|-----|-----------|
| **합(Sum)** | $\lim_{x \to a} [f(x) + g(x)] = L + M$ |
| **차(Difference)** | $\lim_{x \to a} [f(x) - g(x)] = L - M$ |
| **곱(Product)** | $\lim_{x \to a} [f(x) \cdot g(x)] = L \cdot M$ |
| **몫(Quotient)** | $\lim_{x \to a} \frac{f(x)}{g(x)} = \frac{L}{M}$, 단 $M \neq 0$ |
| **거듭제곱(Power)** | $\lim_{x \to a} [f(x)]^n = L^n$ |
| **상수(Constant)** | $\lim_{x \to a} c = c$ |

이 법칙들은 복잡한 극한을 더 간단한 조각들로 분해할 수 있게 해준다. 예를 들어:

$$\lim_{x \to 2} (3x^2 + 5x - 1) = 3(4) + 5(2) - 1 = 21$$

## 한쪽 극한

때때로 함수는 왼쪽과 오른쪽에서 다르게 행동한다:

$$\lim_{x \to a^-} f(x) \quad \text{(좌극한: 아래쪽에서 접근)}$$
$$\lim_{x \to a^+} f(x) \quad \text{(우극한: 위쪽에서 접근)}$$

**핵심 정리:** $\lim_{x \to a} f(x) = L$이 존재할 필요충분조건은 양쪽 한쪽 극한이 모두 존재하고 같은 것이다:

$$\lim_{x \to a^-} f(x) = \lim_{x \to a^+} f(x) = L$$

**예시:** 헤비사이드 계단 함수(Heaviside step function) $H(x) = \begin{cases} 0 & x < 0 \\ 1 & x \geq 0 \end{cases}$

여기서 $\lim_{x \to 0^-} H(x) = 0$이고 $\lim_{x \to 0^+} H(x) = 1$이다. 이들이 다르므로, $\lim_{x \to 0} H(x)$는 존재하지 않는다.

## 조임 정리

$x = a$ 근처에서 (아마도 $a$ 자체는 제외) $g(x) \leq f(x) \leq h(x)$이고,

$$\lim_{x \to a} g(x) = \lim_{x \to a} h(x) = L$$

이면 $\lim_{x \to a} f(x) = L$이다.

**고전적 응용:** $\lim_{x \to 0} x \sin(1/x) = 0$을 증명하라.

$-1 \leq \sin(1/x) \leq 1$이므로, $-|x| \leq x\sin(1/x) \leq |x|$이다.

$\lim_{x \to 0} (-|x|) = 0$이고 $\lim_{x \to 0} |x| = 0$이므로, 조임 정리에 의해:

$$\lim_{x \to 0} x \sin(1/x) = 0$$

```python
import numpy as np
import matplotlib.pyplot as plt

# Visualize the Squeeze Theorem for x*sin(1/x)
x = np.linspace(-0.5, 0.5, 10000)
x = x[x != 0]  # Remove x=0 to avoid division by zero

y = x * np.sin(1/x)
y_upper = np.abs(x)   # Upper bound: |x|
y_lower = -np.abs(x)  # Lower bound: -|x|

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y, 'b-', linewidth=0.5, label='$x \\sin(1/x)$')
ax.plot(x, y_upper, 'r--', linewidth=1.5, label='$|x|$ (upper bound)')
ax.plot(x, y_lower, 'g--', linewidth=1.5, label='$-|x|$ (lower bound)')
ax.plot(0, 0, 'ko', markersize=8, zorder=5, label='Limit = 0')

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_title('Squeeze Theorem: $-|x| \\leq x\\sin(1/x) \\leq |x|$')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('squeeze_theorem.png', dpi=150)
plt.show()
```

## 무한대에서의 극한

$x$가 한없이 커질 때 어떤 일이 일어나는지도 고려한다:

$$\lim_{x \to \infty} f(x) = L$$

이것은 $x$를 충분히 크게 취하면 $f(x)$를 $L$에 임의로 가깝게 만들 수 있음을 의미한다.

**유리함수에 유용한 기법:** 분자와 분모를 분모의 최고 차수의 $x$로 나눈다.

$$\lim_{x \to \infty} \frac{3x^2 + 2x}{5x^2 - 1} = \lim_{x \to \infty} \frac{3 + 2/x}{5 - 1/x^2} = \frac{3}{5}$$

**유리함수 $\frac{p(x)}{q(x)}$에 대한 경험 법칙:**
- $\deg(p) < \deg(q)$이면: 극한은 0
- $\deg(p) = \deg(q)$이면: 극한은 최고차 계수의 비
- $\deg(p) > \deg(q)$이면: 극한은 $\pm\infty$ (유한한 값으로 존재하지 않음)

## 연속

함수 $f$가 $x = a$에서 **연속(continuous)** 이려면 세 가지 조건이 성립해야 한다:

1. $f(a)$가 정의되어 있다
2. $\lim_{x \to a} f(x)$가 존재한다
3. $\lim_{x \to a} f(x) = f(a)$

일상적인 표현으로: 펜을 떼지 않고 그래프를 그릴 수 있다.

### 불연속의 유형

| 유형 | 설명 | 예시 |
|------|-------------|---------|
| **제거 가능(Removable)** | 극한은 존재하지만 $f(a)$가 없거나 다르다 | $f(x) = \frac{x^2-1}{x-1}$, $x=1$에서 |
| **점프(Jump)** | 한쪽 극한이 존재하지만 다르다 | 헤비사이드 함수, $x=0$에서 |
| **무한(Infinite)** | 함수가 $\pm\infty$로 발산한다 | $f(x) = 1/x$, $x=0$에서 |
| **진동(Oscillatory)** | 진동으로 인해 극한이 없다 | $f(x) = \sin(1/x)$, $x=0$에서 |

```python
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Demonstrate different types of discontinuity
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Removable discontinuity: (x^2 - 1)/(x - 1) at x = 1
x1 = np.linspace(-1, 3, 1000)
x1 = x1[np.abs(x1 - 1) > 0.01]
y1 = (x1**2 - 1) / (x1 - 1)
axes[0, 0].plot(x1, y1, 'b-', linewidth=2)
axes[0, 0].plot(1, 2, 'o', color='blue', markersize=10,
                markerfacecolor='white', markeredgewidth=2)  # Open circle at hole
axes[0, 0].set_title('Removable: $(x^2-1)/(x-1)$ at $x=1$')
axes[0, 0].grid(True, alpha=0.3)

# 2. Jump discontinuity: Heaviside function at x = 0
x2 = np.linspace(-2, 2, 1000)
y2 = np.heaviside(x2, 0.5)
axes[0, 1].plot(x2[x2 < 0], y2[x2 < 0], 'b-', linewidth=2)
axes[0, 1].plot(x2[x2 > 0], y2[x2 > 0], 'b-', linewidth=2)
axes[0, 1].plot(0, 0, 'o', color='blue', markersize=10,
                markerfacecolor='white', markeredgewidth=2)
axes[0, 1].plot(0, 1, 'o', color='blue', markersize=10, markeredgewidth=2)
axes[0, 1].set_title('Jump: Heaviside at $x=0$')
axes[0, 1].grid(True, alpha=0.3)

# 3. Infinite discontinuity: 1/x at x = 0
x3 = np.linspace(-2, 2, 1000)
x3 = x3[np.abs(x3) > 0.05]
y3 = 1 / x3
axes[1, 0].plot(x3, y3, 'b-', linewidth=2)
axes[1, 0].set_ylim(-10, 10)
axes[1, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
axes[1, 0].set_title('Infinite: $1/x$ at $x=0$')
axes[1, 0].grid(True, alpha=0.3)

# 4. Oscillatory discontinuity: sin(1/x) at x = 0
x4 = np.linspace(-0.5, 0.5, 100000)
x4 = x4[np.abs(x4) > 0.001]
y4 = np.sin(1 / x4)
axes[1, 1].plot(x4, y4, 'b-', linewidth=0.3)
axes[1, 1].set_title('Oscillatory: $\\sin(1/x)$ at $x=0$')
axes[1, 1].grid(True, alpha=0.3)

for ax in axes.flat:
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

plt.tight_layout()
plt.savefig('discontinuity_types.png', dpi=150)
plt.show()
```

## 중간값 정리 (IVT)

**정리:** $f$가 $[a, b]$에서 연속이고 $N$이 $f(a)$와 $f(b)$ 사이의 임의의 수이면, $f(c) = N$인 $c \in (a, b)$가 적어도 하나 존재한다.

**직관:** 연속 함수는 값을 "건너뛸" 수 없다. 해수면 아래에서 시작하여 해수면 위에서 끝나면, 어딘가에서 반드시 해수면을 통과해야 한다.

**실용적 응용:** IVT는 근의 존재를 보장한다. $f(a) < 0$이고 $f(b) > 0$($f$가 연속)이면, 어떤 $c \in (a, b)$에 대해 $f(c) = 0$이다.

```python
import numpy as np

def bisection_method(f, a, b, tol=1e-10, max_iter=100):
    """
    Find a root of f in [a, b] using the bisection method.

    The IVT guarantees a root exists when f(a) and f(b) have
    opposite signs. Bisection repeatedly halves the interval,
    choosing the half where the sign change persists.
    """
    if f(a) * f(b) >= 0:
        raise ValueError("f(a) and f(b) must have opposite signs")

    for i in range(max_iter):
        c = (a + b) / 2
        if abs(f(c)) < tol or (b - a) / 2 < tol:
            return c, i + 1  # Return root and iteration count
        if f(a) * f(c) < 0:
            b = c  # Root is in left half
        else:
            a = c  # Root is in right half

    return c, max_iter

# Example: find sqrt(2) as a root of f(x) = x^2 - 2
f = lambda x: x**2 - 2
root, iterations = bisection_method(f, 1, 2)
print(f"Root found: {root:.15f}")
print(f"Actual sqrt(2): {np.sqrt(2):.15f}")
print(f"Iterations: {iterations}")
print(f"Error: {abs(root - np.sqrt(2)):.2e}")
```

## SymPy를 이용한 기호적 극한

Python의 SymPy 라이브러리는 극한을 기호적으로 계산하여 정확한 답을 제공할 수 있다:

```python
import sympy as sp

x = sp.Symbol('x')

# Basic limit
expr1 = (x**2 - 1) / (x - 1)
print(f"lim (x^2-1)/(x-1) as x->1: {sp.limit(expr1, x, 1)}")
# Output: 2

# Limit at infinity
expr2 = (3*x**2 + 2*x) / (5*x**2 - 1)
print(f"lim (3x^2+2x)/(5x^2-1) as x->inf: {sp.limit(expr2, x, sp.oo)}")
# Output: 3/5

# One-sided limits
expr3 = 1 / x
print(f"lim 1/x as x->0+: {sp.limit(expr3, x, 0, '+')}")  # oo
print(f"lim 1/x as x->0-: {sp.limit(expr3, x, 0, '-')}")  # -oo

# The famous limit: sin(x)/x as x -> 0
expr4 = sp.sin(x) / x
print(f"lim sin(x)/x as x->0: {sp.limit(expr4, x, 0)}")
# Output: 1

# Squeeze theorem example
expr5 = x * sp.sin(1/x)
print(f"lim x*sin(1/x) as x->0: {sp.limit(expr5, x, 0)}")
# Output: 0
```

## 요약

- **극한(limit)** 은 함수가 접근하는 값을 설명하며, 반드시 함수가 취하는 값은 아니다
- **엡실론-델타 정의**는 "접근한다"를 엄밀하게 만든다: 임의의 허용 오차 $\varepsilon$에 대해 근방 $\delta$를 찾을 수 있다
- **극한 법칙**은 복잡한 극한을 더 간단한 부분들로 분해할 수 있게 해준다
- **한쪽 극한**이 일치해야 양쪽 극한이 존재한다
- **조임 정리(Squeeze Theorem)** 는 진동하는 함수를 상하한으로 묶어 처리한다
- **연속(continuity)** 은 극한이 함수값과 같다는 것을 의미한다 -- 간격, 점프, 구멍이 없다
- **중간값 정리(Intermediate Value Theorem)** 는 연속 함수가 값을 건너뛸 수 없음을 보장하여, 근 찾기 알고리즘을 가능하게 한다

## 연습 문제

### 문제 1: 대수적 극한 계산

다음 극한을 대수적으로 계산하라 (풀이 과정을 보일 것):

(a) $\lim_{x \to 3} \frac{x^2 - 9}{x - 3}$

(b) $\lim_{x \to 0} \frac{\sqrt{1 + x} - 1}{x}$ (힌트: 분자를 유리화하라)

(c) $\lim_{x \to \infty} \frac{2x^3 - x + 5}{4x^3 + 3x^2}$

### 문제 2: 엡실론-델타 증명

엡실론-델타 정의를 사용하여 $\lim_{x \to 2} (3x + 1) = 7$임을 증명하라.

(힌트: $|f(x) - L| < \varepsilon$에서 시작하여 $\varepsilon$에 대한 $\delta$를 역방향으로 구하라.)

### 문제 3: 불연속 분류

각 함수에 대해, 주어진 점에서의 불연속 유형을 결정하라:

(a) $f(x) = \frac{\sin x}{x}$, $x = 0$에서

(b) $f(x) = \lfloor x \rfloor$ (바닥 함수), $x = 2$에서

(c) $f(x) = \frac{1}{(x-1)^2}$, $x = 1$에서

### 문제 4: IVT 응용

방정식 $x^5 - 3x + 1 = 0$이 구간 $[0, 1]$에서 적어도 하나의 근을 가짐을 보여라. 그런 다음 이분법(위의 코드를 수정)을 사용하여 이 근을 소수점 8자리까지 구하라.

### 문제 5: 조임 정리

조임 정리를 사용하여 $\lim_{x \to 0} x^2 \cos(1/x^2)$를 계산하라. 그런 다음 Python 코드를 작성하여 $[-0.5, 0.5]$에서 함수와 그 경계 함수를 시각화하라.

## 참고 자료

- Stewart, *Calculus: Early Transcendentals*, Ch. 2 (Limits and Derivatives)
- [3Blue1Brown: Limits](https://www.youtube.com/watch?v=kfF40MiS7zA)
- [Paul's Online Notes: Limits](https://tutorial.math.lamar.edu/Classes/CalcI/Limits.aspx)

---

[이전: 과정 개요](./00_Overview.md) | [다음: 도함수의 기초](./02_Derivatives_Fundamentals.md)
