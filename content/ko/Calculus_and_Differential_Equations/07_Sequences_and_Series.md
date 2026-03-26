# 수열과 급수

## 학습 목표

- 극한 법칙, 조임 정리(Squeeze Theorem), 단조 수렴 정리(Monotone Convergence Theorem)를 사용하여 수열(sequence)의 수렴(convergence) 또는 발산(divergence)을 **판정**할 수 있다
- 수렴 판정법 (비교, 비율, 근, 적분, 교대급수)을 **적용**하여 무한 급수(infinite series)의 수렴 여부를 판정할 수 있다
- 멱급수(power series)의 수렴 반지름(radius of convergence)과 수렴 구간을 **구할** 수 있다
- 일반적인 함수에 대한 테일러 급수(Taylor series)와 매클로린 급수(Maclaurin series)를 **유도**하고 근사 오차를 한정할 수 있다
- Python으로 테일러 다항식 근사를 **구현**하고 수렴 행동을 시각화할 수 있다

## 소개

무한히 많은 수를 더해서 유한한 결과를 얻을 수 있을까? 놀랍게도, 답은 종종 "그렇다"이다. 제논의 역설(Zeno's paradox)을 생각해 보자: 방을 가로지르려면 먼저 방의 절반을 건너야 하고, 그 다음 남은 것의 절반을, 그 다음 또 절반을 건너야 한다. 거리들은 급수를 이룬다:

$$\frac{1}{2} + \frac{1}{4} + \frac{1}{8} + \frac{1}{16} + \cdots = 1$$

이 무한 합은 정확히 1이다. 수열과 급수의 이론은 이를 엄밀하게 만들고, 무한 합이 언제 수렴하는지, 얼마나 빨리 수렴하는지, 그리고 이를 사용하여 함수를 어떻게 표현하는지를 판정하는 도구를 제공한다.

급수는 단순한 이론적 호기심이 아니다. 테일러 급수는 컴퓨터가 내부적으로 $\sin$, $\cos$, $e^x$, $\ln$을 계산하는 방법의 근간이다. 멱급수(power series)는 닫힌 형태의 해가 불가능한 미분방정식을 푼다. 푸리에 급수(Fourier series)는 신호를 주파수로 분해한다. 이 레슨은 이 모든 것의 기초를 쌓는다.

> **상호 참조:** [물리수학](../Mathematical_Methods/00_Overview.md) 토픽 (레슨 01)은 물리학자의 관점에서 무한 급수와 수렴을 다루며, 점근 해석(asymptotic analysis)과 고급 합산 기법을 강조한다.

## 수열

**수열(sequence)** 은 순서가 있는 수의 목록이다: $a_1, a_2, a_3, \ldots$ 또는 동등하게 함수 $a : \mathbb{N} \to \mathbb{R}$.

### 수열의 수렴

수열 $\{a_n\}$이 극한 $L$에 **수렴(converges)** 한다는 것은 모든 $\varepsilon > 0$에 대해 다음을 만족하는 정수 $N$이 존재한다는 것이다:

$$n > N \implies |a_n - L| < \varepsilon$$

$\lim_{n \to \infty} a_n = L$로 쓴다. 그러한 $L$이 존재하지 않으면, 수열은 **발산(diverges)** 한다.

### 유용한 수열 극한

| 수열 $a_n$ | 극한 | 조건 |
|-----------------|-------|-----------|
| $\frac{1}{n^p}$ | $0$ | $p > 0$ |
| $r^n$ | $0$ | $\|r\| < 1$ |
| $n^{1/n}$ | $1$ | -- |
| $\left(1 + \frac{1}{n}\right)^n$ | $e$ | $e$의 정의 |
| $\frac{n!}{n^n}$ | $0$ | 스털링 근사(Stirling's approximation) |
| $\frac{\ln n}{n}$ | $0$ | 로그는 선형보다 느리게 성장 |

### 단조 수렴 정리

수열이 **단조(monotone)** (항상 증가 또는 항상 감소)이고 **유계(bounded)** 이면, 수렴한다.

이것이 강력한 이유는 극한을 찾지 않고도 수렴을 보장하기 때문이다. 예를 들어, 수열 $a_n = \left(1 + \frac{1}{n}\right)^n$은 증가하고 3으로 위에서 유계이므로 수렴한다. 그 극한은 $e$로 밝혀진다.

```python
import numpy as np
import matplotlib.pyplot as plt

# Visualize convergence of several sequences
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
n = np.arange(1, 51)

# (1 + 1/n)^n -> e
a1 = (1 + 1/n)**n
axes[0, 0].stem(n, a1, linefmt='b-', markerfmt='bo', basefmt='k-')
axes[0, 0].axhline(y=np.e, color='red', linestyle='--', label=f'$e \\approx {np.e:.4f}$')
axes[0, 0].set_title('$(1 + 1/n)^n \\to e$')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# n^(1/n) -> 1
a2 = n**(1/n)
axes[0, 1].stem(n, a2, linefmt='b-', markerfmt='bo', basefmt='k-')
axes[0, 1].axhline(y=1, color='red', linestyle='--', label='Limit = 1')
axes[0, 1].set_title('$n^{1/n} \\to 1$')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# (-1)^n / n -> 0 (oscillating but converging)
a3 = (-1)**n / n
axes[1, 0].stem(n, a3, linefmt='b-', markerfmt='bo', basefmt='k-')
axes[1, 0].axhline(y=0, color='red', linestyle='--', label='Limit = 0')
axes[1, 0].set_title('$(-1)^n / n \\to 0$')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# (-1)^n diverges (oscillates without settling)
a4 = (-1.0)**n
axes[1, 1].stem(n, a4, linefmt='b-', markerfmt='bo', basefmt='k-')
axes[1, 1].set_title('$(-1)^n$ diverges (oscillates)')
axes[1, 1].grid(True, alpha=0.3)

for ax in axes.flat:
    ax.set_xlabel('$n$')

plt.tight_layout()
plt.savefig('sequence_convergence.png', dpi=150)
plt.show()
```

## 무한 급수

**무한 급수(infinite series)** 는 수열의 항들의 합이다:

$$\sum_{n=1}^{\infty} a_n = a_1 + a_2 + a_3 + \cdots$$

**부분합(partial sums)** 은 $S_N = \sum_{n=1}^{N} a_n$이다. 부분합의 수열 $\{S_N\}$이 수렴하면 급수가 수렴한다:

$$\sum_{n=1}^{\infty} a_n = \lim_{N \to \infty} S_N$$

### 등비급수 (Geometric Series)

가장 기본적인 급수:

$$\sum_{n=0}^{\infty} r^n = \frac{1}{1 - r}, \quad |r| < 1$$

**유도:** $S_N = 1 + r + r^2 + \cdots + r^N = \frac{1 - r^{N+1}}{1 - r}$. $N \to \infty$이면, $|r| < 1$일 때 $r^{N+1} \to 0$.

더 일반적으로: $|r| < 1$이면 $\sum_{n=0}^{\infty} ar^n = \frac{a}{1-r}$.

### 조화급수 (Harmonic Series)

$$\sum_{n=1}^{\infty} \frac{1}{n} = 1 + \frac{1}{2} + \frac{1}{3} + \frac{1}{4} + \cdots = \infty$$

$a_n \to 0$임에도 불구하고, 이 급수는 발산한다! 항들이 너무 느리게 줄어든다. 이것은 중요한 경고이다: $a_n \to 0$은 수렴의 **필요** 조건이지만 **충분** 조건은 아니다.

### 발산 판정법 (n번째 항 판정법)

$\lim_{n \to \infty} a_n \neq 0$이면, $\sum a_n$은 발산한다.

**대우:** $\sum a_n$이 수렴하면, $a_n \to 0$이다. 그러나 다시, $a_n \to 0$이 수렴을 보장하지 않는다 (조화급수가 반례).

```python
import numpy as np
import matplotlib.pyplot as plt

# Compare partial sums: geometric (converges) vs harmonic (diverges)
N = 100
n = np.arange(1, N + 1)

# Geometric series: sum of (1/2)^n
geometric_partial = np.cumsum(0.5**n)

# Harmonic series: sum of 1/n
harmonic_partial = np.cumsum(1.0 / n)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(n, geometric_partial, 'b-', linewidth=2)
ax1.axhline(y=1.0, color='red', linestyle='--',
            label='Limit = 1')
ax1.set_xlabel('$N$ (number of terms)')
ax1.set_ylabel('Partial sum $S_N$')
ax1.set_title('Geometric Series $\\sum (1/2)^n$: Converges to 1')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(n, harmonic_partial, 'r-', linewidth=2)
ax2.set_xlabel('$N$ (number of terms)')
ax2.set_ylabel('Partial sum $S_N$')
ax2.set_title('Harmonic Series $\\sum 1/n$: Diverges (slowly!)')
ax2.grid(True, alpha=0.3)
# Note: after 100 terms, the harmonic series is only about 5.2.
# It takes about e^(10^6) terms to reach a partial sum of 10^6.
# This is the slowest possible divergence.

plt.tight_layout()
plt.savefig('series_convergence_comparison.png', dpi=150)
plt.show()
```

## 수렴 판정법

### 비교 판정법 (Comparison Test)

모든 $n$에 대해 $0 \leq a_n \leq b_n$이면:
- $\sum b_n$이 수렴하면, $\sum a_n$도 수렴한다 (유한한 것에 의해 유계)
- $\sum a_n$이 발산하면, $\sum b_n$도 발산한다 (무한한 것보다 큼)

**예시:** $\frac{1}{n^2+1} < \frac{1}{n^2}$이고 $\sum \frac{1}{n^2}$가 수렴($p = 2 > 1$인 $p$-급수)하므로 $\sum \frac{1}{n^2 + 1}$은 수렴한다.

### 극한 비교 판정법 (Limit Comparison Test)

$a_n, b_n > 0$이고 $\lim_{n \to \infty} \frac{a_n}{b_n} = L$ ($0 < L < \infty$)이면, $\sum a_n$과 $\sum b_n$은 동시에 수렴하거나 동시에 발산한다.

**직관:** $a_n/b_n \to L$이면, 큰 $n$에 대해 $a_n \approx L \cdot b_n$이므로 급수는 같은 방식으로 행동한다.

### 비율 판정법 (Ratio Test)

$$L = \lim_{n \to \infty} \left|\frac{a_{n+1}}{a_n}\right|$$

- $L < 1$: 급수가 (절대적으로) 수렴
- $L > 1$: 급수가 발산
- $L = 1$: 판정 불가

**최적 대상:** 팩토리얼, 지수, 또는 곱을 포함하는 급수. 비율 판정법은 수렴 검사의 주력 도구이다.

### 근 판정법 (Root Test)

$$L = \lim_{n \to \infty} \sqrt[n]{|a_n|}$$

비율 판정법과 같은 결론. $a_n$이 $(...)^n$ 형태일 때 유용하다.

### 적분 판정법 (Integral Test)

$f(x)$가 $[1, \infty)$에서 양수, 연속, 감소하고 $f(n) = a_n$이면:

$$\sum_{n=1}^{\infty} a_n \text{ 수렴} \iff \int_1^{\infty} f(x) \, dx \text{ 수렴}$$

**응용:** $p$-급수 $\sum \frac{1}{n^p}$는 $p > 1$일 때만 수렴한다 (이상 적분의 $p$-판정법과 일치).

### 교대급수 판정법 (Alternating Series Test)

$\{b_n\}$이 양수이고 감소하며 $b_n \to 0$이면, 교대급수(alternating series):

$$\sum_{n=1}^{\infty} (-1)^{n+1} b_n = b_1 - b_2 + b_3 - b_4 + \cdots$$

는 수렴한다. 또한, $N$항 이후의 오차는 $|R_N| \leq b_{N+1}$로 한정된다.

**예시:** 교대 조화급수 $\sum \frac{(-1)^{n+1}}{n} = 1 - \frac{1}{2} + \frac{1}{3} - \frac{1}{4} + \cdots = \ln 2$.

```python
import numpy as np
import sympy as sp

n = sp.Symbol('n', positive=True, integer=True)

# Demonstrate the ratio test on several series
print("=== Ratio Test Examples ===\n")

# 1. sum of n! / n^n
a_n = sp.factorial(n) / n**n
ratio = sp.simplify(a_n.subs(n, n+1) / a_n)
L = sp.limit(ratio, n, sp.oo)
print(f"Series: sum n!/n^n")
print(f"  Ratio: a_(n+1)/a_n simplified -> limit = {L}")
print(f"  L = {float(L):.4f} < 1, so series CONVERGES\n")

# 2. sum of 2^n / n!
a_n = 2**n / sp.factorial(n)
ratio = sp.simplify(a_n.subs(n, n+1) / a_n)
L = sp.limit(ratio, n, sp.oo)
print(f"Series: sum 2^n/n!")
print(f"  Ratio limit = {L}")
print(f"  L = 0 < 1, so series CONVERGES (sum = e^2 - 1)\n")

# 3. sum of n^2 / 2^n
a_n = n**2 / 2**n
ratio = sp.simplify(a_n.subs(n, n+1) / a_n)
L = sp.limit(ratio, n, sp.oo)
print(f"Series: sum n^2/2^n")
print(f"  Ratio limit = {L}")
print(f"  L = 1/2 < 1, so series CONVERGES\n")

# Alternating harmonic series partial sums -> ln(2)
N_values = np.arange(1, 201)
partial_sums = np.cumsum([(-1)**(k+1) / k for k in N_values])

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(N_values, partial_sums, 'b-', linewidth=1, alpha=0.7)
ax.axhline(y=np.log(2), color='red', linestyle='--',
           label=f'$\\ln 2 \\approx {np.log(2):.6f}$')
ax.set_xlabel('Number of terms $N$')
ax.set_ylabel('Partial sum $S_N$')
ax.set_title('Alternating Harmonic Series: $\\sum (-1)^{n+1}/n = \\ln 2$')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('alternating_harmonic.png', dpi=150)
plt.show()
```

### 수렴 판정법 결정 가이드

```
a_n -> 0인가?
  아니오 --> 급수 발산 (발산 판정법)
  예    --> 수렴을 보장하지 않음; 추가 판정법 적용:

등비급수 (a*r^n)인가?
  예 --> |r| < 1일 때만 수렴

p-급수 (1/n^p)인가?
  예 --> p > 1일 때만 수렴

팩토리얼이나 지수를 포함하는가?
  예 --> 비율 판정법 시도

a_n이 (...)^n 형태인가?
  예 --> 근 판정법 시도

알려진 수렴/발산 급수로 a_n을 한정할 수 있는가?
  예 --> 비교 판정법 또는 극한 비교

감소하는 항의 교대급수인가?
  예 --> 교대급수 판정법

f(n) = a_n인 f(x)를 적분할 수 있는가?
  예 --> 적분 판정법

명확하게 작동하는 것이 없는가?
  --> a_n을 다시 쓰거나, 부분분수, 또는 직접 부분합 계산 시도
```

## 멱급수

$a$를 중심으로 한 **멱급수(power series)** 는:

$$\sum_{n=0}^{\infty} c_n (x - a)^n = c_0 + c_1(x-a) + c_2(x-a)^2 + \cdots$$

이것은 "무한 다항식으로서의 함수"이다. 일부 $x$ 값에서 수렴하고 다른 값에서 발산한다.

### 수렴 반지름

모든 멱급수는 **수렴 반지름(radius of convergence)** $R$을 가져서:
- $|x - a| < R$이면 급수가 절대 수렴
- $|x - a| > R$이면 급수가 발산
- $|x - a| = R$ (경계)에서는 수렴을 별도로 확인해야 함

반지름은 비율 판정법으로 구할 수 있다:

$$R = \lim_{n \to \infty} \left|\frac{c_n}{c_{n+1}}\right| \quad \text{또는 동등하게} \quad \frac{1}{R} = \lim_{n \to \infty} \left|\frac{c_{n+1}}{c_n}\right|$$

**예시:** $\sum \frac{x^n}{n!}$의 경우: $\frac{1}{R} = \lim \frac{n!}{(n+1)!} = \lim \frac{1}{n+1} = 0$이므로 $R = \infty$. 이 급수는 모든 $x$에서 수렴한다 ($e^x$를 나타낸다).

**예시:** $\sum n! \, x^n$의 경우: $\frac{1}{R} = \lim \frac{(n+1)!}{n!} = \lim (n+1) = \infty$이므로 $R = 0$. 이 급수는 $x = 0$에서만 수렴한다.

## 테일러 급수와 매클로린 급수

### 정의

$a$를 중심으로 한 $f(x)$의 **테일러 급수(Taylor series)** 는:

$$f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!} (x - a)^n$$

$a = 0$일 때, 이를 **매클로린 급수(Maclaurin series)** 라 한다.

**핵심 통찰:** 테일러 급수는 $n$번째 부분합(테일러 다항식 $T_n$)이 $x = a$에서 $f$와 그 처음 $n$개의 도함수와 일치하도록 구성된다. 각 추가 항은 더 미세한 국소적 행동을 포착한다.

### 중요한 매클로린 급수

이것들은 암기해야 한다 -- 수학과 과학 전반에 걸쳐 나타난다:

| 함수 | 매클로린 급수 | 반지름 $R$ |
|----------|------------------|------------|
| $e^x$ | $\sum_{n=0}^{\infty} \frac{x^n}{n!} = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \cdots$ | $\infty$ |
| $\sin x$ | $\sum_{n=0}^{\infty} \frac{(-1)^n x^{2n+1}}{(2n+1)!} = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \cdots$ | $\infty$ |
| $\cos x$ | $\sum_{n=0}^{\infty} \frac{(-1)^n x^{2n}}{(2n)!} = 1 - \frac{x^2}{2!} + \frac{x^4}{4!} - \cdots$ | $\infty$ |
| $\frac{1}{1-x}$ | $\sum_{n=0}^{\infty} x^n = 1 + x + x^2 + x^3 + \cdots$ | $1$ |
| $\ln(1+x)$ | $\sum_{n=1}^{\infty} \frac{(-1)^{n+1} x^n}{n} = x - \frac{x^2}{2} + \frac{x^3}{3} - \cdots$ | $1$ |
| $(1+x)^\alpha$ | $\sum_{n=0}^{\infty} \binom{\alpha}{n} x^n$ (이항 급수) | $1$ |
| $\arctan x$ | $\sum_{n=0}^{\infty} \frac{(-1)^n x^{2n+1}}{2n+1} = x - \frac{x^3}{3} + \frac{x^5}{5} - \cdots$ | $1$ |

### 테일러 나머지 정리

$n$차 테일러 다항식의 오차는:

$$R_n(x) = \frac{f^{(n+1)}(c)}{(n+1)!} (x - a)^{n+1}$$

$a$와 $x$ 사이의 어떤 $c$에 대해 성립한다. 이를 **라그랑주 나머지(Lagrange remainder)** 라 한다. 이것은 근사 오차의 상한을 제공한다:

$$|R_n(x)| \leq \frac{M}{(n+1)!} |x - a|^{n+1}$$

여기서 $M = \max |f^{(n+1)}(t)|$ ($t$는 $a$와 $x$ 사이).

```python
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

def taylor_polynomial(f, x, a, n):
    """
    Compute the nth-degree Taylor polynomial of f centered at a.

    This builds the polynomial term by term, each term adding
    more accuracy near x = a. The factorial in the denominator
    ensures the derivatives match.
    """
    poly = 0
    for k in range(n + 1):
        coeff = sp.diff(f, x, k).subs(x, a) / sp.factorial(k)
        poly += coeff * (x - a)**k
    return poly

x = sp.Symbol('x')
f = sp.sin(x)

# Compute Taylor polynomials of sin(x) at a=0 for various degrees
print("Taylor polynomials of sin(x) centered at 0:")
for degree in [1, 3, 5, 7, 9]:
    T = taylor_polynomial(f, x, 0, degree)
    print(f"  T_{degree}(x) = {T}")

# Visualization: Taylor polynomials converging to sin(x)
x_vals = np.linspace(-2*np.pi, 2*np.pi, 500)

fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(x_vals, np.sin(x_vals), 'k-', linewidth=3, label='$\\sin(x)$')

colors = ['red', 'orange', 'green', 'blue', 'purple']
degrees = [1, 3, 5, 7, 9]

for degree, color in zip(degrees, colors):
    T = taylor_polynomial(f, x, 0, degree)
    T_func = sp.lambdify(x, T, 'numpy')
    y_taylor = T_func(x_vals)
    # Clip extreme values for visualization (Taylor polynomials diverge far from center)
    y_taylor = np.clip(y_taylor, -3, 3)
    ax.plot(x_vals, y_taylor, '--', color=color, linewidth=1.5,
            label=f'$T_{{{degree}}}(x)$')

ax.set_ylim(-3, 3)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_title('Taylor Polynomials of $\\sin(x)$ at $a = 0$')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.axvline(x=0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig('taylor_sin.png', dpi=150)
plt.show()
```

### 컴퓨터는 어떻게 sin(x)을 계산하는가

최신 CPU와 수학 라이브러리는 테일러 유사 다항식 근사 (종종 체비셰프(Chebyshev) 또는 미니맥스(minimax) 다항식)를 사용하여 삼각 함수를 계산한다. 아이디어는:

1. 대칭을 이용하여 인수를 작은 구간 (예: $[0, \pi/4]$)으로 **축소**
2. 7-13차 다항식을 사용하여 **근사** (배정밀도에 충분)
3. 다항식은 최대 오차를 최소화하도록 사전 계산됨

```python
import numpy as np

def sin_taylor(x, n_terms=10):
    """
    Approximate sin(x) using its Maclaurin series.

    This demonstrates the principle behind how computers evaluate
    trigonometric functions, though real implementations use
    optimized polynomial approximations (minimax/Chebyshev).
    """
    # First, reduce x to [-pi, pi] using periodicity
    x = x % (2 * np.pi)
    if x > np.pi:
        x -= 2 * np.pi

    result = 0.0
    for k in range(n_terms):
        # Each term: (-1)^k * x^(2k+1) / (2k+1)!
        term = ((-1)**k * x**(2*k + 1)) / np.math.factorial(2*k + 1)
        result += term
    return result

# Test accuracy
test_values = [0.1, 0.5, 1.0, np.pi/4, np.pi/2, np.pi, 3.0, 10.0]
print(f"{'x':>8} {'Taylor (10 terms)':>20} {'np.sin(x)':>15} {'Error':>12}")
print("-" * 58)
for x in test_values:
    approx = sin_taylor(x, n_terms=10)
    exact = np.sin(x)
    print(f"{x:>8.4f} {approx:>20.15f} {exact:>15.15f} {abs(approx-exact):>12.2e}")
```

### e를 계산하기 위한 테일러 급수

```python
import numpy as np

def compute_e(n_terms=20):
    """
    Compute e using its Maclaurin series: e = sum_{n=0}^{inf} 1/n!

    The factorial in the denominator makes this series converge
    extremely fast. With just 20 terms, we get 18 digits of accuracy.
    """
    e_approx = 0.0
    for n in range(n_terms):
        term = 1.0 / np.math.factorial(n)
        e_approx += term
        if n < 12:
            print(f"  n={n:>2d}: term = 1/{n}! = {term:.15f}, "
                  f"partial sum = {e_approx:.15f}")
    return e_approx

print("Computing e via Taylor series:")
result = compute_e(20)
print(f"\nResult:  {result:.18f}")
print(f"np.e:    {np.e:.18f}")
print(f"Error:   {abs(result - np.e):.2e}")
```

## 요약

- **수열(sequences)** 은 항들이 유한한 극한에 접근하면 수렴한다; 단조 수렴 정리는 유계 단조 수열의 수렴을 보장한다
- **급수(series)** 는 수열 항들의 합이다; 수렴은 부분합이 유한한 극한에 접근하는 것을 의미한다
- **발산 판정법**은 빠른 첫 번째 검사이다: $a_n \not\to 0$이면 급수는 발산한다
- **수렴 판정법** (비교, 비율, 근, 적분, 교대급수)은 각각 이상적인 사용 경우가 있다 -- 비율 판정법이 가장 다재다능하다
- **멱급수** $\sum c_n(x-a)^n$은 $a$를 중심으로 반지름 $R$인 디스크 안에서 수렴한다
- **테일러 급수**는 매끄러운 함수를 무한 다항식으로 표현한다; 테일러 나머지가 근사 오차를 한정한다
- 암기할 핵심 급수: $e^x$, $\sin x$, $\cos x$, $\frac{1}{1-x}$, $\ln(1+x)$
- 컴퓨터는 초월 함수를 계산하기 위해 다항식 근사 (테일러 급수의 후손)를 사용한다

## 연습 문제

### 문제 1: 수열의 수렴

각 수열이 수렴하는지 발산하는지 판정하라. 수렴하면 극한을 구하라.

(a) $a_n = \frac{n^2 + 3n}{2n^2 - 1}$

(b) $a_n = \frac{(-1)^n n}{n + 1}$

(c) $a_n = \left(1 + \frac{3}{n}\right)^n$

### 문제 2: 급수 수렴 판정법

각 급수가 수렴하는지 발산하는지 판정하고, 사용한 판정법을 명시하라.

(a) $\sum_{n=1}^{\infty} \frac{n^2}{3^n}$ (비율 판정법)

(b) $\sum_{n=2}^{\infty} \frac{1}{n \ln n}$ (적분 판정법)

(c) $\sum_{n=1}^{\infty} \frac{(-1)^n}{\sqrt{n}}$ (교대급수 판정법)

(d) $\sum_{n=1}^{\infty} \frac{n!}{n^n}$ (비율 판정법)

### 문제 3: 수렴 반지름

각 멱급수의 수렴 반지름과 수렴 구간을 구하라:

(a) $\sum_{n=0}^{\infty} \frac{(x-3)^n}{n \cdot 2^n}$

(b) $\sum_{n=0}^{\infty} \frac{n! \, x^n}{n^n}$

### 문제 4: 테일러 급수 유도

(a) 등비급수 $\frac{1}{1-u} = \sum u^n$에 $-x^2$를 대입하여 $f(x) = \frac{1}{1+x^2}$의 매클로린 급수를 유도하라.

(b) 결과를 항별로 적분하여 $\arctan x$의 급수를 얻어라.

(c) $\arctan(1) = \pi/4$를 사용하여 라이프니츠 공식을 유도하라: $\frac{\pi}{4} = 1 - \frac{1}{3} + \frac{1}{5} - \frac{1}{7} + \cdots$

(d) 이 급수를 사용하여 $\pi$를 계산하는 Python 코드를 작성하라. 6자리 정확도를 위해 몇 개의 항이 필요한가?

### 문제 5: 테일러 다항식 오차 한정

테일러 나머지 정리를 사용하여 $e^x$의 매클로린 다항식 $T_n(x)$가 $e^{0.5}$를 $10^{-8}$ 미만의 오차로 근사하는 데 필요한 차수 $n$을 결정하라.

그런 다음 계산적으로 검증하라: 증가하는 $n$에 대해 $T_n(0.5)$을 계산하고 $e^{0.5}$의 정확한 값과 비교하라.

## 참고 자료

- Stewart, *Calculus: Early Transcendentals*, Ch. 11 (Infinite Sequences and Series)
- [3Blue1Brown: Taylor Series](https://www.youtube.com/watch?v=3d6DsjIBzJ4)
- [Paul's Online Notes: Series and Sequences](https://tutorial.math.lamar.edu/Classes/CalcII/SeriesIntro.aspx)
- 참고: [물리수학 L01](../Mathematical_Methods/01_Infinite_Series.md) - 고급 수렴 주제

---

[이전: 적분의 응용](./06_Applications_of_Integration.md) | [다음: 매개변수 곡선과 극좌표](./08_Parametric_and_Polar.md)
