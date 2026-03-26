# 적분의 기초

## 학습 목표

- 리만 합(Riemann sum, 왼쪽, 오른쪽, 중점, 사다리꼴)을 사용하여 정적분을 **근사**하고 더 미세한 분할이 어떻게 정확도를 향상시키는지 설명할 수 있다
- 미적분학의 기본 정리(Fundamental Theorem of Calculus)의 두 부분을 **진술**하고 왜 미분과 적분을 연결하는지 설명할 수 있다
- 다항식, 지수, 삼각, 기본 유리 함수에 대한 역도함수(antiderivative)를 **계산**할 수 있다
- FTC를 사용하여 정적분을 **계산**하고 결과를 부호 면적(signed area)으로 해석할 수 있다
- Python으로 수치 적분 방법(리만 합, 사다리꼴 법칙, 심프슨 법칙, `scipy.integrate.quad`)을 **비교**할 수 있다

## 소개

도함수가 "얼마나 빠르게 변하는가?"에 답한다면, 적분은 "얼마나 누적되었는가?"에 답한다. 자동차의 속도계는 속도(위치의 도함수)를 알려주고, 주행 거리계는 총 거리(속도의 적분)를 알려준다. 우량계는 총 강우량(강우 속도의 적분)을 측정한다. 은행 잔액은 시간에 따른 입출금의 적분이다.

적분은 미분의 역(inverse)이지만, 그보다 훨씬 더 많은 것이다: 면적, 부피, 평균, 확률, 일(work)과 에너지 같은 물리량을 계산하기 위해 무한히 많은 무한소 조각들을 합산하는 방법이다.

## 리만 합: 직관 쌓기

핵심 아이디어는 곡선 아래의 면적을 얇은 직사각형으로 잘라 근사하는 것이다.

$[a, b]$에서 함수 $f(x)$가 주어지면, 구간을 너비 $\Delta x = \frac{b - a}{n}$인 $n$개의 동일한 부분 구간으로 나눈다.

### 리만 합의 유형

| 방법 | 표본점 | 공식 |
|--------|-------------|---------|
| **왼쪽** | 왼쪽 끝점 $x_i = a + i \cdot \Delta x$ | $L_n = \sum_{i=0}^{n-1} f(x_i) \Delta x$ |
| **오른쪽** | 오른쪽 끝점 $x_{i+1} = a + (i+1) \cdot \Delta x$ | $R_n = \sum_{i=1}^{n} f(x_i) \Delta x$ |
| **중점** | 중점 $\bar{x}_i = a + (i + \frac{1}{2}) \Delta x$ | $M_n = \sum_{i=0}^{n-1} f(\bar{x}_i) \Delta x$ |
| **사다리꼴** | 왼쪽과 오른쪽의 평균 | $T_n = \frac{L_n + R_n}{2}$ |

$n \to \infty$ (직사각형이 더 얇아짐)이면, 네 가지 방법 모두 같은 값, 즉 **정적분(definite integral)** 에 수렴한다.

```python
import numpy as np
import matplotlib.pyplot as plt

def riemann_sum_visualization(f, a, b, n=10, method='left'):
    """
    Visualize a Riemann sum approximation.

    The rectangles show how we approximate the curved area with
    flat-topped boxes. More rectangles = better approximation.
    """
    x = np.linspace(a, b, 1000)
    dx = (b - a) / n

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the actual function
    ax.plot(x, f(x), 'b-', linewidth=2, label='$f(x)$', zorder=3)

    total = 0
    for i in range(n):
        xi = a + i * dx
        if method == 'left':
            height = f(xi)
        elif method == 'right':
            height = f(xi + dx)
        elif method == 'midpoint':
            height = f(xi + dx/2)

        total += height * dx

        # Draw rectangle
        rect = plt.Rectangle((xi, 0), dx, height,
                              facecolor='skyblue', edgecolor='navy',
                              alpha=0.5, linewidth=1)
        ax.add_patch(rect)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$f(x)$')
    ax.set_title(f'{method.capitalize()} Riemann Sum (n={n}): '
                 f'$\\sum \\approx {total:.4f}$')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'riemann_{method}_{n}.png', dpi=150)
    plt.show()
    return total

# Example: integral of x^2 from 0 to 1 (exact answer: 1/3)
f = lambda x: x**2
for n in [5, 10, 50]:
    approx = riemann_sum_visualization(f, 0, 1, n, method='midpoint')
    print(f"  n={n}: approximation = {approx:.6f}, error = {abs(approx - 1/3):.6f}")
```

### 리만 합의 수렴

```python
import numpy as np

def compare_riemann_methods(f, a, b, exact, n_values):
    """
    Compare how quickly different Riemann sum methods converge.

    This demonstrates that midpoint and trapezoidal converge faster
    (O(1/n^2)) than left and right (O(1/n)). Simpson's rule, which
    combines them, converges even faster (O(1/n^4)).
    """
    print(f"{'n':>8} {'Left':>12} {'Right':>12} {'Midpoint':>12} "
          f"{'Trapezoid':>12} {'Simpson':>12}")
    print("-" * 72)

    for n in n_values:
        dx = (b - a) / n
        x_left = np.linspace(a, b - dx, n)
        x_right = np.linspace(a + dx, b, n)
        x_mid = np.linspace(a + dx/2, b - dx/2, n)

        L = np.sum(f(x_left)) * dx
        R = np.sum(f(x_right)) * dx
        M = np.sum(f(x_mid)) * dx
        T = (L + R) / 2
        S = (2*M + T) / 3  # Simpson's rule combines midpoint and trapezoidal

        print(f"{n:>8d} {abs(L-exact):>12.2e} {abs(R-exact):>12.2e} "
              f"{abs(M-exact):>12.2e} {abs(T-exact):>12.2e} {abs(S-exact):>12.2e}")

# Integral of sin(x) from 0 to pi, exact answer = 2
f = lambda x: np.sin(x)
compare_riemann_methods(f, 0, np.pi, exact=2.0,
                        n_values=[10, 50, 100, 500, 1000, 5000])
```

## 정적분

$n \to \infty$이면, 리만 합은 **정적분(definite integral)** 이 된다:

$$\int_a^b f(x) \, dx = \lim_{n \to \infty} \sum_{i=1}^{n} f(x_i^*) \Delta x$$

**표기법 설명:**
- $\int$ : "합(sum)"을 뜻하는 늘어난 S (라이프니츠의 표기법)
- $a, b$: 적분의 하한과 상한
- $f(x)$: 피적분함수(integrand) (적분되는 함수)
- $dx$: 무한소의 작은 너비 ($\Delta x$의 극한)

**해석:** 정적분은 곡선과 $x$축 사이의 **부호 면적(signed area)** 을 제공한다:
- $x$축 위의 면적은 **양수**로 계산
- $x$축 아래의 면적은 **음수**로 계산

### 정적분의 성질

| 성질 | 설명 |
|----------|-----------|
| **선형성(Linearity)** | $\int_a^b [cf(x) + dg(x)] \, dx = c\int_a^b f(x)\,dx + d\int_a^b g(x)\,dx$ |
| **가산성(Additivity)** | $\int_a^b f(x)\,dx + \int_b^c f(x)\,dx = \int_a^c f(x)\,dx$ |
| **역순(Reversal)** | $\int_a^b f(x)\,dx = -\int_b^a f(x)\,dx$ |
| **비교(Comparison)** | $[a,b]$에서 $f(x) \geq g(x)$이면, $\int_a^b f\,dx \geq \int_a^b g\,dx$ |
| **영 너비(Zero width)** | $\int_a^a f(x)\,dx = 0$ |

## 역도함수와 부정적분

$f(x)$의 **역도함수(antiderivative)** 는 $F'(x) = f(x)$인 임의의 함수 $F(x)$이다.

**부정적분(indefinite integral)** 은 모든 역도함수를 모은다:

$$\int f(x) \, dx = F(x) + C$$

여기서 $C$는 **적분 상수(constant of integration)** 이다 -- 임의의 상수가 미분 시 사라진다는 것을 상기시킨다.

### 기본 역도함수 표

| $f(x)$ | $\int f(x)\,dx$ |
|---------|-----------------|
| $x^n$ ($n \neq -1$) | $\frac{x^{n+1}}{n+1} + C$ |
| $x^{-1} = 1/x$ | $\ln|x| + C$ |
| $e^x$ | $e^x + C$ |
| $\sin x$ | $-\cos x + C$ |
| $\cos x$ | $\sin x + C$ |
| $\sec^2 x$ | $\tan x + C$ |
| $\frac{1}{1+x^2}$ | $\arctan x + C$ |
| $\frac{1}{\sqrt{1-x^2}}$ | $\arcsin x + C$ |

```python
import sympy as sp

x = sp.Symbol('x')

# SymPy computes antiderivatives (indefinite integrals)
expressions = [x**3, sp.sin(x), sp.exp(x), 1/(1 + x**2), 1/x]

for expr in expressions:
    antideriv = sp.integrate(expr, x)
    # Verify by differentiating the result
    check = sp.diff(antideriv, x)
    print(f"integral of {expr} = {antideriv}")
    print(f"  Verification: d/dx[{antideriv}] = {sp.simplify(check)}\n")
```

## 미적분학의 기본 정리

이것은 수학 전체에서 가장 중요한 정리 중 하나이다. 미분과 적분이 역연산(inverse operation)임을 밝혀준다.

### 제1부 (적분의 미분)

$f$가 $[a, b]$에서 연속이고 다음을 정의하면:

$$F(x) = \int_a^x f(t) \, dt$$

$F'(x) = f(x)$이다.

**말로 표현하면:** 누적 함수의 도함수는 원래 함수이다. $f(t)$가 속도(예를 들어, 탱크에 시간당 $f(t)$ 리터의 물이 흘러들어가는 속도)라면, $F(x) = \int_a^x f(t)\,dt$는 시간 $a$부터 시간 $x$까지 누적된 총 물의 양이며, 시간 $x$에서의 누적 속도는 정확히 $f(x)$이다.

### 제2부 (정적분의 계산)

$f$가 $[a, b]$에서 연속이고 $F$가 $f$의 임의의 역도함수(즉, $F' = f$)이면:

$$\int_a^b f(x) \, dx = F(b) - F(a)$$

**이것은 혁명적이다:** 리만 합의 극한(무한 분할이 필요한)을 계산하는 대신, 단순히 역도함수를 찾아 끝점에서 계산하면 된다. 무한히 많은 무한소 조각들의 합이 단순한 뺄셈으로 환원된다.

**표기법:** $F(x) \Big|_a^b = F(b) - F(a)$로 쓴다.

**예시:**

$$\int_0^{\pi} \sin x \, dx = [-\cos x]_0^{\pi} = -\cos(\pi) - (-\cos(0)) = -(-1) + 1 = 2$$

이것은 사인 곡선의 한 아치 아래의 총 면적이다.

```python
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Visualize FTC Part 1: the accumulation function
x_sym = sp.Symbol('x')
t_sym = sp.Symbol('t')

# f(t) = sin(t), accumulation F(x) = integral from 0 to x of sin(t) dt
f_expr = sp.sin(t_sym)
F_expr = sp.integrate(f_expr, (t_sym, 0, x_sym))  # = 1 - cos(x)
F_prime = sp.diff(F_expr, x_sym)  # Should equal sin(x)

print(f"f(t) = {f_expr}")
print(f"F(x) = integral_0^x sin(t) dt = {F_expr}")
print(f"F'(x) = {F_prime}")
print(f"FTC Part 1 verified: F'(x) = f(x) = sin(x)")

# Visualize both f and F
x_vals = np.linspace(0, 4*np.pi, 500)
f_vals = np.sin(x_vals)
F_vals = 1 - np.cos(x_vals)  # The accumulation function

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Original function
ax1.plot(x_vals, f_vals, 'b-', linewidth=2)
ax1.fill_between(x_vals, f_vals, 0, where=(f_vals >= 0),
                  alpha=0.3, color='green', label='Positive area')
ax1.fill_between(x_vals, f_vals, 0, where=(f_vals < 0),
                  alpha=0.3, color='red', label='Negative area')
ax1.set_ylabel('$f(t) = \\sin(t)$')
ax1.set_title('Integrand and Accumulation Function (FTC Part 1)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Accumulation function
ax2.plot(x_vals, F_vals, 'r-', linewidth=2,
         label='$F(x) = \\int_0^x \\sin(t)\\,dt = 1 - \\cos(x)$')
ax2.set_xlabel('$x$')
ax2.set_ylabel('$F(x)$')
ax2.legend()
ax2.grid(True, alpha=0.3)
# Notice: F increases when f > 0, decreases when f < 0, and
# has extrema where f crosses zero. This is FTC Part 1 in action.

plt.tight_layout()
plt.savefig('ftc_accumulation.png', dpi=150)
plt.show()
```

## SciPy를 이용한 수치 적분

닫힌 형태의 역도함수가 없는 함수에 대해, 수치 적분은 필수적이다.

```python
import numpy as np
from scipy import integrate

# Example: integral of e^(-x^2) from 0 to infinity
# This is related to the Gaussian integral: sqrt(pi)/2

# scipy.integrate.quad: adaptive quadrature (gold standard)
result, error = integrate.quad(lambda x: np.exp(-x**2), 0, np.inf)
print(f"integral of exp(-x^2) from 0 to inf:")
print(f"  Numerical: {result:.15f}")
print(f"  Exact:     {np.sqrt(np.pi)/2:.15f}")
print(f"  Estimated error: {error:.2e}")

# Compare methods on a simpler integral: integral of sin(x) from 0 to pi
f = lambda x: np.sin(x)
exact = 2.0

# Trapezoidal rule (numpy)
for n in [10, 100, 1000]:
    x = np.linspace(0, np.pi, n+1)
    trap = np.trapz(f(x), x)
    print(f"\n  Trapezoidal (n={n:>4d}): {trap:.12f}, error = {abs(trap-exact):.2e}")

# Simpson's rule (scipy)
for n in [10, 100, 1000]:
    x = np.linspace(0, np.pi, n+1)
    simp = integrate.simpson(f(x), x=x)
    print(f"  Simpson's  (n={n:>4d}): {simp:.12f}, error = {abs(simp-exact):.2e}")

# Adaptive quadrature (scipy) -- usually the best choice
quad_result, quad_error = integrate.quad(f, 0, np.pi)
print(f"\n  Adaptive quad:       {quad_result:.15f}, est. error = {quad_error:.2e}")
```

## 요약

- **리만 합(Riemann sums)** 은 정의역을 직사각형으로 분할하여 적분을 근사한다; 더 미세한 분할은 더 나은 근사를 제공한다
- **정적분(definite integral)** $\int_a^b f(x)\,dx$은 리만 합의 극한이며 부호 면적을 나타낸다
- **역도함수(antiderivatives)** 는 도함수의 역이다: $F'(x) = f(x) \implies \int f(x)\,dx = F(x) + C$
- **FTC 제1부**: 누적 함수 $\int_a^x f(t)\,dt$의 도함수는 $f(x)$이다
- **FTC 제2부**: $\int_a^b f(x)\,dx = F(b) - F(a)$ -- 계산이 무한 합산을 대체한다
- 닫힌 형태의 역도함수가 없는 함수에 대해, **수치적 방법** (사다리꼴, 심프슨, 적응 구적법)이 정확한 근사를 제공한다

## 연습 문제

### 문제 1: 리만 합 계산

$n = 4$개의 부분 구간으로 $\int_0^2 x^3 \, dx$에 대한 왼쪽, 오른쪽, 중점 리만 합을 계산하라. 각각을 정확한 값과 비교하라. 어떤 방법이 가장 가까운가?

### 문제 2: FTC 응용

미적분학의 기본 정리를 사용하여 다음을 계산하라:

(a) $\int_1^4 (3\sqrt{x} - 1/x) \, dx$

(b) $\int_0^{\pi/4} \sec^2 \theta \, d\theta$

(c) $\frac{d}{dx} \int_0^{x^2} \sin(t^2) \, dt$ (힌트: FTC 제1부와 연쇄 법칙을 적용)

### 문제 3: 부호 면적 해석

$\int_0^{2\pi} \sin x \, dx$를 계산하라. 사인 함수가 분명히 면적을 가지고 있는데도 결과가 0인 이유를 설명하라. 그런 다음 *부호 없는 총 면적* $\int_0^{2\pi} |\sin x| \, dx$를 계산하라.

### 문제 4: 수치 적분 비교

$\int_0^1 e^{-x^2} dx$를 다음을 사용하여 근사하는 Python 코드를 작성하라:
- $n = 1000$인 왼쪽 리만 합
- $n = 1000$인 사다리꼴 법칙
- $n = 1000$인 심프슨 법칙
- `scipy.integrate.quad`

네 가지 결과를 모두 비교하라. 어떤 방법이 가장 적은 함수 계산으로 가장 정확한 답을 제공하는가?

### 문제 5: FTC 증명

FTC 제1부가 $\frac{d}{dx}\int_a^x f(t)\,dt = f(x)$라 할 때, FTC 제2부를 증명하라: $F' = f$일 때 $\int_a^b f(x)\,dx = F(b) - F(a)$.

(힌트: $G(x) = \int_a^x f(t)\,dt$를 정의하라. 제1부에 의해, $G'(x) = f(x) = F'(x)$이다. $G(x) - F(x)$에 대해 무엇을 결론지을 수 있는가?)

## 참고 자료

- Stewart, *Calculus: Early Transcendentals*, Ch. 5 (Integrals)
- [3Blue1Brown: Integration and the Fundamental Theorem](https://www.youtube.com/watch?v=rfG8ce4nNh0)
- [Paul's Online Notes: Integrals](https://tutorial.math.lamar.edu/Classes/CalcI/IntegralsIntro.aspx)

---

[이전: 도함수의 응용](./03_Applications_of_Derivatives.md) | [다음: 적분 기법](./05_Integration_Techniques.md)
