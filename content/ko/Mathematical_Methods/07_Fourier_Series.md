# 07. 푸리에 급수 (Fourier Series)

## 학습 목표

- **주기 함수(periodic function)**를 삼각 함수의 급수로 분해하는 **푸리에 급수**의 개념과 수학적 기반(직교성)을 이해한다
- **푸리에 계수** $a_0, a_n, b_n$을 계산하고, 구형파·톱니파 등 대표적인 파형에 적용할 수 있다
- **디리클레 조건**, **점별·균등 수렴**, **깁스 현상(Gibbs phenomenon)**을 이해하고, 수렴 거동을 분석할 수 있다
- **반구간 전개(half-range expansion)**와 **파르세발 정리(Parseval's theorem)**를 활용하여 물리 문제를 풀 수 있다
- 진동하는 현, 열전도, 전자기파 등 **물리학 응용**에서 푸리에 급수가 핵심 도구로 쓰이는 방식을 이해한다

---

## 1. 푸리에 급수의 기본 개념

### 1.1 주기 함수와 직교성

**주기 함수(periodic function)**는 양의 상수 $T > 0$에 대해 다음을 만족하는 함수이다:

$$f(x + T) = f(x) \quad \text{for all } x$$

여기서 $T$를 **주기(period)**라 한다. 가장 작은 양의 주기를 **기본 주기(fundamental period)**라 부른다.

대표적인 주기 함수로는 $\sin x$ (주기 $2\pi$), $\cos x$ (주기 $2\pi$), $\tan x$ (주기 $\pi$) 등이 있다.

**직교성(orthogonality)**은 푸리에 급수의 핵심 개념이다. 두 함수 $f(x)$와 $g(x)$가 구간 $[a, b]$에서 **직교(orthogonal)**한다는 것은:

$$\int_a^b f(x) g(x) \, dx = 0$$

이를 만족하는 것을 뜻한다. 이것은 벡터의 내적 $\mathbf{a} \cdot \mathbf{b} = 0$인 직교 조건을 함수 공간으로 확장한 것이다.

### 1.2 삼각 함수의 직교성

구간 $[-\pi, \pi]$ (또는 임의의 한 주기 구간)에서 삼각 함수 집합 $\{1, \cos x, \sin x, \cos 2x, \sin 2x, \ldots\}$는 서로 직교한다:

$$\int_{-\pi}^{\pi} \cos(mx) \cos(nx) \, dx = \begin{cases} 0 & m \neq n \\ \pi & m = n \neq 0 \\ 2\pi & m = n = 0 \end{cases}$$

$$\int_{-\pi}^{\pi} \sin(mx) \sin(nx) \, dx = \begin{cases} 0 & m \neq n \\ \pi & m = n \neq 0 \end{cases}$$

$$\int_{-\pi}^{\pi} \cos(mx) \sin(nx) \, dx = 0 \quad \text{(모든 } m, n \text{에 대해)}$$

이 직교 관계는 $\cos(mx)$와 $\sin(nx)$가 서로 다른 "방향"을 가리키는 기저 벡터(basis vector)와 같다는 것을 의미한다.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- 삼각 함수의 직교성 수치적 검증 ---
x = np.linspace(-np.pi, np.pi, 10000)
dx = x[1] - x[0]

# cos(mx) * cos(nx) 적분
print("=== cos(mx) * cos(nx) 직교성 검증 ===")
for m in range(0, 4):
    for n in range(0, 4):
        integral = np.trapz(np.cos(m * x) * np.cos(n * x), x)
        if abs(integral) > 1e-10:
            print(f"  m={m}, n={n}: {integral:.4f}")

# cos(mx) * sin(nx) 적분 (항상 0)
print("\n=== cos(mx) * sin(nx) 직교성 검증 ===")
for m in range(1, 4):
    for n in range(1, 4):
        integral = np.trapz(np.cos(m * x) * np.sin(n * x), x)
        print(f"  m={m}, n={n}: {integral:.6f}")  # 모두 ≈ 0

# 출력:
# === cos(mx) * cos(nx) 직교성 검증 ===
#   m=0, n=0: 6.2832  (= 2*pi)
#   m=1, n=1: 3.1416  (= pi)
#   m=2, n=2: 3.1416  (= pi)
#   m=3, n=3: 3.1416  (= pi)
# === cos(mx) * sin(nx) 직교성 검증 ===
#   m=1, n=1: 0.000000
#   ...
```

### 1.3 푸리에 급수의 정의

주기 $2\pi$인 함수 $f(x)$의 **푸리에 급수(Fourier series)**는:

$$f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty} \left[ a_n \cos(nx) + b_n \sin(nx) \right]$$

여기서 $a_0/2$는 $f(x)$의 평균값에 해당한다. 계수 $a_n$, $b_n$은 직교성을 이용하여 결정한다.

일반적으로 주기 $2L$인 함수에 대해서는:

$$f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty} \left[ a_n \cos\!\left(\frac{n\pi x}{L}\right) + b_n \sin\!\left(\frac{n\pi x}{L}\right) \right]$$

---

## 2. 푸리에 계수 계산

### 2.1 $a_0$, $a_n$, $b_n$ 공식

직교성 관계를 이용하면, 주기 $2\pi$인 함수 $f(x)$의 푸리에 계수는:

$$a_0 = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x) \, dx$$

$$a_n = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x) \cos(nx) \, dx \quad (n = 1, 2, 3, \ldots)$$

$$b_n = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x) \sin(nx) \, dx \quad (n = 1, 2, 3, \ldots)$$

**유도 과정**: 양변에 $\cos(mx)$를 곱하고 $[-\pi, \pi]$에서 적분하면, 직교성에 의해 $n = m$인 항만 남아:

$$\int_{-\pi}^{\pi} f(x) \cos(mx) \, dx = a_m \cdot \pi \quad \Rightarrow \quad a_m = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x) \cos(mx) \, dx$$

$b_n$도 같은 방법으로 $\sin(mx)$를 곱하여 유도한다.

주기 $2L$인 함수의 경우:

$$a_0 = \frac{1}{L} \int_{-L}^{L} f(x) \, dx, \quad a_n = \frac{1}{L} \int_{-L}^{L} f(x) \cos\!\left(\frac{n\pi x}{L}\right) dx, \quad b_n = \frac{1}{L} \int_{-L}^{L} f(x) \sin\!\left(\frac{n\pi x}{L}\right) dx$$

### 2.2 계산 예제: 구형파 (Square Wave)

주기 $2\pi$인 **구형파(square wave)**를 다음과 같이 정의하자:

$$f(x) = \begin{cases} 1 & 0 < x < \pi \\ -1 & -\pi < x < 0 \end{cases}$$

$f(x)$는 **기함수(odd function)**이므로 $a_0 = 0$, $a_n = 0$이다 (기함수 $\times$ 우함수 = 기함수이므로 대칭 구간에서 적분이 0).

$b_n$을 계산하면:

$$b_n = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x) \sin(nx) \, dx = \frac{2}{\pi} \int_0^{\pi} \sin(nx) \, dx = \frac{2}{\pi} \left[-\frac{\cos(nx)}{n}\right]_0^{\pi}$$

$$= \frac{2}{n\pi} (1 - \cos(n\pi)) = \begin{cases} \frac{4}{n\pi} & n \text{ 홀수} \\ 0 & n \text{ 짝수} \end{cases}$$

따라서 구형파의 푸리에 급수는:

$$f(x) = \frac{4}{\pi} \left( \sin x + \frac{\sin 3x}{3} + \frac{\sin 5x}{5} + \cdots \right) = \frac{4}{\pi} \sum_{k=0}^{\infty} \frac{\sin\left[(2k+1)x\right]}{2k+1}$$

```python
import numpy as np
import matplotlib.pyplot as plt

# --- 구형파의 푸리에 급수 수렴 시각화 ---
x = np.linspace(-2 * np.pi, 2 * np.pi, 2000)

# 원래 구형파 (주기 2*pi)
square_wave = np.sign(np.sin(x))

# 푸리에 급수 부분합 (N항까지)
def fourier_square(x, N):
    """구형파의 푸리에 급수 부분합 (N개의 사인 항)"""
    result = np.zeros_like(x)
    for k in range(N):
        n = 2 * k + 1  # 홀수만
        result += np.sin(n * x) / n
    return (4 / np.pi) * result

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
terms = [1, 3, 10, 50]

for ax, N in zip(axes.flat, terms):
    ax.plot(x, square_wave, 'k--', alpha=0.3, linewidth=1, label='원래 함수')
    ax.plot(x, fourier_square(x, N), 'b-', linewidth=1.5, label=f'N={N}')
    ax.set_title(f'구형파 푸리에 급수: {N}항')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_ylim(-1.5, 1.5)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fourier_square_wave.png', dpi=150)
plt.show()
```

### 2.3 계산 예제: 톱니파 (Sawtooth Wave)

주기 $2\pi$인 **톱니파(sawtooth wave)**:

$$f(x) = x \quad (-\pi < x < \pi)$$

이것 역시 기함수이므로 $a_0 = 0$, $a_n = 0$이다.

$$b_n = \frac{1}{\pi} \int_{-\pi}^{\pi} x \sin(nx) \, dx = \frac{2}{\pi} \int_0^{\pi} x \sin(nx) \, dx$$

부분적분을 수행하면:

$$= \frac{2}{\pi} \left[-\frac{x \cos(nx)}{n} + \frac{\sin(nx)}{n^2}\right]_0^{\pi} = \frac{2}{\pi} \cdot \left(-\frac{\pi \cos(n\pi)}{n}\right) = -\frac{2 \cos(n\pi)}{n} = \frac{2(-1)^{n+1}}{n}$$

따라서:

$$f(x) = 2 \sum_{n=1}^{\infty} \frac{(-1)^{n+1}}{n} \sin(nx) = 2\left(\sin x - \frac{\sin 2x}{2} + \frac{\sin 3x}{3} - \cdots\right)$$

```python
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, sin, cos, pi, integrate, simplify

# --- SymPy를 이용한 해석적 계수 계산 ---
x_sym, n_sym = symbols('x n', real=True)

# 톱니파의 b_n 계수
b_n_expr = (1 / pi) * integrate(x_sym * sin(n_sym * x_sym), (x_sym, -pi, pi))
print(f"b_n = {simplify(b_n_expr)}")
# 출력: b_n = -2*cos(pi*n)/n = 2*(-1)^(n+1)/n

# --- 톱니파 푸리에 급수 시각화 ---
x = np.linspace(-2 * np.pi, 2 * np.pi, 2000)

# 원래 톱니파 (주기 2*pi)
sawtooth = x - 2 * np.pi * np.round(x / (2 * np.pi))

def fourier_sawtooth(x, N):
    """톱니파의 푸리에 급수 부분합"""
    result = np.zeros_like(x)
    for n in range(1, N + 1):
        result += ((-1) ** (n + 1) / n) * np.sin(n * x)
    return 2 * result

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x, sawtooth, 'k--', alpha=0.3, linewidth=1, label='원래 함수')
for N, color in [(3, 'blue'), (10, 'green'), (50, 'red')]:
    ax.plot(x, fourier_sawtooth(x, N), color=color, linewidth=1.2, label=f'N={N}')

ax.set_title('톱니파의 푸리에 급수')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fourier_sawtooth_wave.png', dpi=150)
plt.show()
```

---

## 3. 수렴과 깁스 현상

### 3.1 디리클레 조건 (Dirichlet Conditions)

함수 $f(x)$의 푸리에 급수가 수렴하기 위한 **충분조건**을 **디리클레 조건(Dirichlet conditions)**이라 한다:

1. $f(x)$는 한 주기 내에서 **유한 개의 불연속점**만 갖는다
2. $f(x)$는 한 주기 내에서 **유한 개의 극값(maxima and minima)**만 갖는다
3. $\int_{-\pi}^{\pi} |f(x)| \, dx < \infty$ (**절대적분가능**, absolutely integrable)

이 조건을 만족하면:
- **연속인 점** $x$에서: 푸리에 급수는 $f(x)$로 수렴
- **불연속인 점** $x_0$에서: 푸리에 급수는 좌극한과 우극한의 **산술평균**으로 수렴

$$S(x_0) = \frac{f(x_0^+) + f(x_0^-)}{2}$$

> 물리과학에서 다루는 대부분의 함수는 디리클레 조건을 만족한다.

### 3.2 점별 수렴과 균등 수렴

**점별 수렴(pointwise convergence)**: 각 점 $x$에서 개별적으로 $S_N(x) \to f(x)$

$$\forall x, \forall \varepsilon > 0, \exists N_0(x, \varepsilon): N > N_0 \Rightarrow |S_N(x) - f(x)| < \varepsilon$$

여기서 $N_0$가 $x$에 따라 달라질 수 있다.

**균등 수렴(uniform convergence)**: 모든 점에서 동시에, $x$에 무관하게 수렴

$$\forall \varepsilon > 0, \exists N_0(\varepsilon): N > N_0 \Rightarrow |S_N(x) - f(x)| < \varepsilon \quad \forall x$$

불연속점이 있는 함수의 푸리에 급수는 **점별 수렴**하지만 **균등 수렴**하지 않는다 (깁스 현상 때문).
연속이고 구간별 매끄러운(piecewise smooth) 함수의 푸리에 급수는 **균등 수렴**한다.

### 3.3 깁스 현상 (Gibbs Phenomenon)

불연속점 근처에서 푸리에 급수의 부분합은 원래 함수값을 약 **9%** 초과하여 진동(overshoot)하며, 이 초과분은 항의 수를 아무리 늘려도 사라지지 않는다. 이를 **깁스 현상(Gibbs phenomenon)**이라 한다.

정확한 초과 비율:

$$\text{overshoot} \approx \frac{2}{\pi} \int_0^{\pi} \frac{\sin t}{t} \, dt - 1 \approx 0.0895 \approx 8.95\%$$

이것은 불연속의 크기(점프 크기)에 대한 비율이다. 구형파의 경우, 점프 크기가 2이므로 $S_N$의 최대 overshoot은 약 $1 + 2 \times 0.0895 = 1.179$이다.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# --- 깁스 현상 시각화 ---
x = np.linspace(-0.5, 0.5, 5000)

def fourier_square_detail(x, N):
    """구형파의 부분합 (불연속점 x=0 근처 확대)"""
    result = np.zeros_like(x)
    for k in range(N):
        n = 2 * k + 1
        result += np.sin(n * x) / n
    return (4 / np.pi) * result

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (좌) 다양한 N에 대한 불연속점 근처 거동
for N in [10, 50, 200, 1000]:
    axes[0].plot(x, fourier_square_detail(x, N), linewidth=1, label=f'N={N}')
axes[0].axhline(y=1, color='k', linestyle='--', alpha=0.3)
axes[0].axhline(y=-1, color='k', linestyle='--', alpha=0.3)
axes[0].set_title('깁스 현상: 불연속점(x=0) 근처')
axes[0].set_xlabel('x')
axes[0].set_ylabel('$S_N(x)$')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# (우) 이론적 overshoot 계산
# Si(pi) = integral_0^pi sin(t)/t dt
Si_pi, _ = quad(lambda t: np.sin(t) / t, 0, np.pi)
overshoot = 2 * Si_pi / np.pi - 1
print(f"이론적 overshoot: {overshoot:.4f} ({overshoot*100:.2f}%)")

# N에 따른 최대값 변화
N_values = np.arange(5, 505, 5)
max_values = []
x_fine = np.linspace(0.0001, 0.5, 10000)
for N in N_values:
    S_N = fourier_square_detail(x_fine, N)
    max_values.append(np.max(S_N))

axes[1].plot(N_values, max_values, 'b-', linewidth=1)
axes[1].axhline(y=1 + overshoot * 2, color='r', linestyle='--',
                label=f'이론값: {1 + overshoot*2:.4f}')
axes[1].axhline(y=1, color='k', linestyle='--', alpha=0.3)
axes[1].set_title('항의 수(N)에 따른 최대값')
axes[1].set_xlabel('N (항의 수)')
axes[1].set_ylabel('$\\max S_N(x)$')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gibbs_phenomenon.png', dpi=150)
plt.show()
```

---

## 4. 반구간 전개와 대칭성

### 4.1 우함수와 코사인 급수

$f(x)$가 **우함수(even function)**, 즉 $f(-x) = f(x)$이면:

- $f(x) \sin(nx)$는 기함수이므로 $b_n = 0$
- 푸리에 급수는 **코사인 급수(cosine series)**만 포함:

$$f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty} a_n \cos(nx)$$

$$a_n = \frac{2}{\pi} \int_0^{\pi} f(x) \cos(nx) \, dx$$

**예제**: $f(x) = |x|$ (주기 $2\pi$)

$$a_0 = \frac{2}{\pi} \int_0^{\pi} x \, dx = \pi$$

$$a_n = \frac{2}{\pi} \int_0^{\pi} x \cos(nx) \, dx = \frac{2}{n^2\pi} (\cos(n\pi) - 1) = \begin{cases} -\frac{4}{n^2\pi} & n \text{ 홀수} \\ 0 & n \text{ 짝수} \end{cases}$$

$$|x| = \frac{\pi}{2} - \frac{4}{\pi}\left(\cos x + \frac{\cos 3x}{9} + \frac{\cos 5x}{25} + \cdots\right)$$

### 4.2 기함수와 사인 급수

$f(x)$가 **기함수(odd function)**, 즉 $f(-x) = -f(x)$이면:

- $f(x) \cos(nx)$는 기함수이므로 $a_n = 0$ (모든 $n \ge 0$)
- 푸리에 급수는 **사인 급수(sine series)**만 포함:

$$f(x) = \sum_{n=1}^{\infty} b_n \sin(nx)$$

$$b_n = \frac{2}{\pi} \int_0^{\pi} f(x) \sin(nx) \, dx$$

앞에서 다룬 구형파와 톱니파가 바로 기함수의 사인 급수 예제이다.

### 4.3 반구간 전개 (Half-Range Expansion)

구간 $[0, L]$에서만 정의된 함수 $f(x)$를 전체 구간 $[-L, L]$로 확장하는 방법은 두 가지이다:

1. **우함수 확장 (even extension)**: $f(-x) = f(x)$로 확장 → **코사인 급수**
2. **기함수 확장 (odd extension)**: $f(-x) = -f(x)$로 확장 → **사인 급수**

물리 문제에서는 경계 조건에 따라 적절한 확장을 선택한다:
- **디리클레 경계조건** ($f(0) = f(L) = 0$): 사인 급수 사용
- **노이만 경계조건** ($f'(0) = f'(L) = 0$): 코사인 급수 사용

```python
import numpy as np
import matplotlib.pyplot as plt

# --- 반구간 전개 시각화 ---
# f(x) = x(pi - x), 0 < x < pi 에서 정의

x_half = np.linspace(0, np.pi, 500)
f_half = x_half * (np.pi - x_half)

# 우함수 확장 (코사인 급수)
def cosine_expansion(x, N):
    """f(x) = x(pi - x)의 코사인 급수"""
    # a_0 = (2/pi) * integral_0^pi x(pi-x) dx = pi^2/3
    a0 = np.pi**2 / 3
    result = a0 / 2 * np.ones_like(x)
    for n in range(1, N + 1):
        # a_n 계산: a_n = (2/pi) * integral_0^pi x(pi-x)*cos(nx) dx
        if n % 2 == 0:
            a_n = -4 / (n**2)
        else:
            a_n = -4 / (n**2)
        # 정확한 계산: a_n = -4/(n^2) (모든 n에 대해 -2(1 + (-1)^n)/(n^2))
        a_n = -2 * (1 + (-1)**n) / (n**2)
        result += a_n * np.cos(n * x)
    return result

# 기함수 확장 (사인 급수)
def sine_expansion(x, N):
    """f(x) = x(pi - x)의 사인 급수"""
    result = np.zeros_like(x)
    for n in range(1, N + 1):
        # b_n = (2/pi) * integral_0^pi x(pi-x)*sin(nx) dx
        if n % 2 == 1:
            b_n = 8 / (n**3 * np.pi)
        else:
            b_n = 0
        result += b_n * np.sin(n * x)
    return result

x_full = np.linspace(-np.pi, np.pi, 1000)
N = 20

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 우함수 확장
f_even = np.abs(x_full) * (np.pi - np.abs(x_full))
axes[0].plot(x_full, f_even, 'k--', alpha=0.3, label='우함수 확장 원본')
axes[0].plot(x_full, cosine_expansion(x_full, N), 'b-', label=f'코사인 급수 (N={N})')
axes[0].set_title('우함수 확장 → 코사인 급수')
axes[0].set_xlabel('x')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 기함수 확장
f_odd = np.where(x_full >= 0,
                 x_full * (np.pi - x_full),
                 -(-x_full) * (np.pi - (-x_full)))
axes[1].plot(x_full, f_odd, 'k--', alpha=0.3, label='기함수 확장 원본')
axes[1].plot(x_full, sine_expansion(x_full, N), 'r-', label=f'사인 급수 (N={N})')
axes[1].set_title('기함수 확장 → 사인 급수')
axes[1].set_xlabel('x')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('half_range_expansion.png', dpi=150)
plt.show()
```

---

## 5. 파르세발 정리

### 5.1 에너지와 파르세발 등식

**파르세발 정리(Parseval's theorem)**는 함수의 "에너지"가 시간/공간 영역과 주파수 영역에서 동일하다는 것을 표현한다:

$$\frac{1}{\pi} \int_{-\pi}^{\pi} |f(x)|^2 \, dx = \frac{a_0^2}{2} + \sum_{n=1}^{\infty} (a_n^2 + b_n^2)$$

물리적 해석:
- **좌변**: 한 주기에 걸친 함수의 "평균 제곱"(에너지에 비례)
- **우변**: 각 주파수 성분의 에너지(진폭의 제곱)의 합
- 전체 에너지는 각 주파수 성분의 에너지를 모두 합한 것과 같다

이것은 피타고라스 정리의 무한차원 확장이다: 벡터의 길이 제곱이 각 성분 제곱의 합인 것처럼, 함수의 "노름(norm) 제곱"은 각 푸리에 성분의 제곱합이다.

### 5.2 응용: 급수의 합 계산

파르세발 정리를 이용하면 특정 급수의 합을 계산할 수 있다.

**예제 1**: 구형파에 대한 파르세발 정리 적용

구형파 $f(x) = \pm 1$에 대해:
- 좌변: $\frac{1}{\pi} \int_{-\pi}^{\pi} 1 \, dx = 2$
- 우변: $\sum_{k=0}^{\infty} b_{2k+1}^2 = \sum_{k=0}^{\infty} \left(\frac{4}{(2k+1)\pi}\right)^2 = \frac{16}{\pi^2} \sum_{k=0}^{\infty} \frac{1}{(2k+1)^2}$

따라서:

$$2 = \frac{16}{\pi^2} \sum_{k=0}^{\infty} \frac{1}{(2k+1)^2} \quad \Rightarrow \quad \sum_{k=0}^{\infty} \frac{1}{(2k+1)^2} = \frac{\pi^2}{8}$$

이로부터 유명한 결과를 얻을 수 있다:

$$\sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{1}{1^2} + \frac{1}{2^2} + \frac{1}{3^2} + \cdots = \frac{\pi^2}{6}$$

(홀수 항의 합 $\pi^2/8$에 짝수 항의 합 $= \frac{1}{4}\sum 1/n^2$을 더하면 유도된다.)

**예제 2**: 톱니파 $f(x) = x$에 파르세발 정리 적용

$$\frac{1}{\pi}\int_{-\pi}^{\pi} x^2 \, dx = \frac{2\pi^2}{3} = \sum_{n=1}^{\infty} \frac{4}{n^2} \quad \Rightarrow \quad \sum_{n=1}^{\infty}\frac{1}{n^2} = \frac{\pi^2}{6}$$

```python
import numpy as np

# --- 파르세발 정리 수치 검증 ---
x = np.linspace(-np.pi, np.pi, 100000)
dx = x[1] - x[0]

# 구형파
f_square = np.sign(np.sin(x + 1e-15))
lhs = np.trapz(f_square**2, x) / np.pi
print(f"구형파 |f|^2 적분 / pi = {lhs:.6f}")

# 우변: 푸리에 계수의 제곱합
rhs = 0
for k in range(10000):
    n = 2 * k + 1
    b_n = 4 / (n * np.pi)
    rhs += b_n**2
print(f"sum b_n^2 = {rhs:.6f}")
print(f"오차: {abs(lhs - rhs):.2e}")

# 급수의 합 계산
series_sum = sum(1 / (2*k+1)**2 for k in range(100000))
print(f"\nsum 1/(2k+1)^2 = {series_sum:.10f}")
print(f"pi^2/8         = {np.pi**2/8:.10f}")

series_all = sum(1 / n**2 for n in range(1, 100001))
print(f"\nsum 1/n^2      = {series_all:.10f}")
print(f"pi^2/6         = {np.pi**2/6:.10f}")

# 출력:
# 구형파 |f|^2 적분 / pi = 2.000000
# sum b_n^2 = 1.999999
# 오차: 5.07e-05
# sum 1/(2k+1)^2 = 1.2337005501
# pi^2/8         = 1.2337005501
# sum 1/n^2      = 1.6449240669
# pi^2/6         = 1.6449340668
```

---

## 6. 복소 푸리에 급수

### 6.1 복소 지수 형태

오일러 공식 $e^{inx} = \cos(nx) + i\sin(nx)$를 이용하면, 푸리에 급수를 더 간결한 **복소 지수(complex exponential)** 형태로 쓸 수 있다:

$$f(x) = \sum_{n=-\infty}^{\infty} c_n \, e^{inx}$$

여기서 복소 푸리에 계수 $c_n$은:

$$c_n = \frac{1}{2\pi} \int_{-\pi}^{\pi} f(x) \, e^{-inx} \, dx$$

실수 계수와의 관계:

$$c_0 = \frac{a_0}{2}, \quad c_n = \frac{a_n - ib_n}{2}, \quad c_{-n} = \frac{a_n + ib_n}{2} = c_n^* \quad (n > 0)$$

$f(x)$가 실수 함수이면 $c_{-n} = c_n^*$ (켤레복소수)이 성립한다.

복소 형태의 장점:
- 수학적으로 더 간결하고 대칭적
- 양과 음의 주파수를 동시에 표현
- 연속 푸리에 변환(Fourier transform)으로의 자연스러운 확장

### 6.2 주파수 스펙트럼

**주파수 스펙트럼(frequency spectrum)**은 각 주파수 성분의 크기를 나타내는 그래프이다:

- **진폭 스펙트럼(amplitude spectrum)**: $|c_n|$ vs. $n$
- **위상 스펙트럼(phase spectrum)**: $\arg(c_n)$ vs. $n$
- **파워 스펙트럼(power spectrum)**: $|c_n|^2$ vs. $n$

파르세발 정리의 복소 형태:

$$\frac{1}{2\pi} \int_{-\pi}^{\pi} |f(x)|^2 \, dx = \sum_{n=-\infty}^{\infty} |c_n|^2$$

```python
import numpy as np
import matplotlib.pyplot as plt

# --- 복소 푸리에 계수와 주파수 스펙트럼 ---
# 구형파, 톱니파, 삼각파의 스펙트럼 비교

x = np.linspace(-np.pi, np.pi, 10000, endpoint=False)
dx = x[1] - x[0]

# 파형 정의
square = np.sign(np.sin(x + 1e-15))
sawtooth = x / np.pi  # 정규화
triangle = 1 - 2 * np.abs(x) / np.pi

# 복소 푸리에 계수 수치 계산
def compute_cn(f, x, n_max):
    """복소 푸리에 계수 c_n을 수치적으로 계산"""
    cn = {}
    for n in range(-n_max, n_max + 1):
        integrand = f * np.exp(-1j * n * x)
        cn[n] = np.trapz(integrand, x) / (2 * np.pi)
    return cn

n_max = 20
cn_square = compute_cn(square, x, n_max)
cn_sawtooth = compute_cn(sawtooth, x, n_max)
cn_triangle = compute_cn(triangle, x, n_max)

# 스펙트럼 시각화
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
waveforms = [('구형파', cn_square), ('톱니파', cn_sawtooth), ('삼각파', cn_triangle)]

for j, (name, cn) in enumerate(waveforms):
    ns = sorted(cn.keys())
    amplitudes = [np.abs(cn[n]) for n in ns]
    phases = [np.angle(cn[n]) for n in ns]

    # 진폭 스펙트럼
    axes[0, j].stem(ns, amplitudes, linefmt='b-', markerfmt='bo', basefmt='k-')
    axes[0, j].set_title(f'{name} - 진폭 스펙트럼')
    axes[0, j].set_xlabel('n (주파수)')
    axes[0, j].set_ylabel('$|c_n|$')
    axes[0, j].grid(True, alpha=0.3)

    # 위상 스펙트럼
    # 진폭이 매우 작은 성분의 위상은 의미 없으므로 필터링
    significant = [n for n in ns if np.abs(cn[n]) > 1e-10]
    sig_phases = [np.angle(cn[n]) for n in significant]
    axes[1, j].stem(significant, sig_phases, linefmt='r-', markerfmt='ro', basefmt='k-')
    axes[1, j].set_title(f'{name} - 위상 스펙트럼')
    axes[1, j].set_xlabel('n (주파수)')
    axes[1, j].set_ylabel('$\\arg(c_n)$')
    axes[1, j].set_ylim(-np.pi - 0.3, np.pi + 0.3)
    axes[1, j].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('frequency_spectra.png', dpi=150)
plt.show()
```

---

## 7. 물리학 응용

### 7.1 진동하는 현 (Standing Waves)

양 끝이 고정된 길이 $L$의 현(string)의 진동은 **파동 방정식(wave equation)**으로 기술된다:

$$\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}$$

경계 조건 $u(0, t) = u(L, t) = 0$과 초기 조건 $u(x, 0) = f(x)$, $\frac{\partial u}{\partial t}(x, 0) = 0$이 주어지면, 변수분리법(separation of variables)에 의해:

$$u(x, t) = \sum_{n=1}^{\infty} b_n \sin\!\left(\frac{n\pi x}{L}\right) \cos\!\left(\frac{n\pi ct}{L}\right)$$

여기서 $b_n$은 초기 변위 $f(x)$의 **사인 급수(반구간 전개)** 계수이다:

$$b_n = \frac{2}{L} \int_0^L f(x) \sin\!\left(\frac{n\pi x}{L}\right) dx$$

물리적 의미:
- $n = 1$: **기본 진동수(fundamental frequency)** $f_1 = c/(2L)$
- $n = 2, 3, \ldots$: **고조파(harmonics)** $f_n = n f_1$
- 각 $b_n$의 크기가 해당 고조파의 기여도를 결정

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 진동하는 현: 초기 변위의 푸리에 분해 ---
L = 1.0  # 현의 길이
c = 1.0  # 파동 속도
N = 20   # 고조파 수

# 초기 변위: 삼각형 (중앙을 잡아당긴 형태)
def initial_displacement(x, L):
    """삼각형 초기 변위: 중앙에서 최대"""
    return np.where(x <= L/2, 2*x/L, 2*(L-x)/L)

# 사인 급수 계수 계산
x_int = np.linspace(0, L, 10000)
f_init = initial_displacement(x_int, L)

b_n = np.zeros(N + 1)
for n in range(1, N + 1):
    integrand = f_init * np.sin(n * np.pi * x_int / L)
    b_n[n] = 2 / L * np.trapz(integrand, x_int)

print("고조파 계수 b_n:")
for n in range(1, 8):
    print(f"  b_{n} = {b_n[n]:+.6f}")

# 시간 t에서의 변위
def string_displacement(x, t, b_n, L, c):
    u = np.zeros_like(x)
    for n in range(1, len(b_n)):
        u += b_n[n] * np.sin(n * np.pi * x / L) * np.cos(n * np.pi * c * t / L)
    return u

# 여러 시점에서의 현의 모양 시각화
x = np.linspace(0, L, 500)
times = [0, 0.1, 0.25, 0.4, 0.5]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for t in times:
    u = string_displacement(x, t, b_n, L, c)
    axes[0].plot(x, u, label=f't = {t:.2f}')

axes[0].set_title('진동하는 현의 시간 변화')
axes[0].set_xlabel('x')
axes[0].set_ylabel('u(x, t)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 고조파 스펙트럼
axes[1].bar(range(1, N + 1), np.abs(b_n[1:]), color='steelblue', alpha=0.7)
axes[1].set_title('고조파 계수 $|b_n|$')
axes[1].set_xlabel('고조파 번호 n')
axes[1].set_ylabel('$|b_n|$')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('vibrating_string.png', dpi=150)
plt.show()
```

### 7.2 열전도 문제의 초기 조건

길이 $L$인 1차원 막대의 **열 방정식(heat equation)**:

$$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$$

여기서 $\alpha$는 **열확산율(thermal diffusivity)**이다.

경계 조건 $u(0, t) = u(L, t) = 0$ (양 끝 온도 고정)과 초기 조건 $u(x, 0) = f(x)$에 대한 해:

$$u(x, t) = \sum_{n=1}^{\infty} b_n \sin\!\left(\frac{n\pi x}{L}\right) \exp\!\left(-\frac{n^2 \pi^2 \alpha t}{L^2}\right)$$

파동 방정식과 달리, 열 방정식에서는 고조파 성분이 $n^2$에 비례하는 속도로 **지수적 감쇠(exponential decay)**한다. 따라서:
- 고주파 성분은 빠르게 소멸
- 시간이 지나면 기본 모드($n = 1$)만 남음
- 이것이 열이 "퍼지는" 현상의 수학적 표현

```python
import numpy as np
import matplotlib.pyplot as plt

# --- 열전도 문제의 푸리에 급수 해 ---
L = 1.0
alpha = 0.01  # 열확산율

# 초기 온도 분포: 중앙에 집중된 열원
def initial_temp(x, L):
    """가우시안 형태의 초기 온도"""
    return np.exp(-50 * (x - L/2)**2)

# 사인 급수 계수
x_int = np.linspace(0, L, 10000)
f_init = initial_temp(x_int, L)
N = 50

b_n = np.zeros(N + 1)
for n in range(1, N + 1):
    integrand = f_init * np.sin(n * np.pi * x_int / L)
    b_n[n] = 2 / L * np.trapz(integrand, x_int)

# 열전도 해
def heat_solution(x, t, b_n, L, alpha):
    u = np.zeros_like(x)
    for n in range(1, len(b_n)):
        decay = np.exp(-n**2 * np.pi**2 * alpha * t / L**2)
        u += b_n[n] * np.sin(n * np.pi * x / L) * decay
    return u

# 시각화
x = np.linspace(0, L, 500)
times = [0, 0.5, 1.0, 2.0, 5.0, 10.0]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for t in times:
    u = heat_solution(x, t, b_n, L, alpha)
    axes[0].plot(x, u, label=f't = {t:.1f}')

axes[0].set_title('열전도: 온도 분포의 시간 변화')
axes[0].set_xlabel('x')
axes[0].set_ylabel('u(x, t)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 각 모드의 감쇠
t_range = np.linspace(0, 10, 200)
for n in [1, 2, 3, 5, 10]:
    decay = np.abs(b_n[n]) * np.exp(-n**2 * np.pi**2 * alpha * t_range / L**2)
    axes[1].plot(t_range, decay, label=f'n={n} (|b_{n}|={np.abs(b_n[n]):.3f})')

axes[1].set_title('고조파 모드의 지수적 감쇠')
axes[1].set_xlabel('시간 t')
axes[1].set_ylabel('$|b_n| e^{-n^2 \\pi^2 \\alpha t / L^2}$')
axes[1].legend()
axes[1].set_yscale('log')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('heat_conduction_fourier.png', dpi=150)
plt.show()
```

### 7.3 전자기파의 주파수 분석

주기적인 전자기 신호의 주파수 성분 분석은 푸리에 급수의 중요한 응용이다.

**AM 변조 신호(Amplitude Modulated signal)**:

$$s(t) = [1 + m \cos(\omega_m t)] \cos(\omega_c t)$$

여기서 $\omega_c$는 **반송파 주파수(carrier frequency)**, $\omega_m$은 **변조 주파수(modulation frequency)**, $m$은 **변조 지수(modulation index)**이다.

삼각함수 곱을 전개하면:

$$s(t) = \cos(\omega_c t) + \frac{m}{2}\cos[(\omega_c + \omega_m)t] + \frac{m}{2}\cos[(\omega_c - \omega_m)t]$$

이처럼 시간 영역의 곱은 주파수 영역에서 **합과 차**의 주파수 성분으로 나타난다.

**비주기적 펄스 열(pulse train)**의 푸리에 급수:

주기 $T$, 폭 $\tau$인 구형 펄스 열:

$$f(t) = \frac{\tau}{T} + \frac{2}{T}\sum_{n=1}^{\infty} \frac{\sin(n\pi\tau/T)}{n\pi/T} \cos\!\left(\frac{2\pi n t}{T}\right)$$

여기서 $\text{sinc}$ 함수 모양의 포락선(envelope)이 주파수 스펙트럼을 결정한다.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- 전자기파 주파수 분석 ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (1) AM 변조 신호
t = np.linspace(0, 1, 10000)
fc = 50   # 반송파 주파수 (Hz)
fm = 5    # 변조 주파수 (Hz)
m = 0.7   # 변조 지수

signal = (1 + m * np.cos(2 * np.pi * fm * t)) * np.cos(2 * np.pi * fc * t)

axes[0, 0].plot(t[:2000], signal[:2000], 'b-', linewidth=0.5)
axes[0, 0].set_title('AM 변조 신호 (시간 영역)')
axes[0, 0].set_xlabel('시간 (s)')
axes[0, 0].set_ylabel('s(t)')
axes[0, 0].grid(True, alpha=0.3)

# FFT를 이용한 스펙트럼 분석
N_fft = len(t)
freq = np.fft.fftfreq(N_fft, d=t[1] - t[0])
S = np.fft.fft(signal) / N_fft

# 양의 주파수만 표시
pos_mask = freq > 0
axes[0, 1].plot(freq[pos_mask], 2 * np.abs(S[pos_mask]), 'r-', linewidth=1)
axes[0, 1].set_title('AM 변조 신호 (주파수 스펙트럼)')
axes[0, 1].set_xlabel('주파수 (Hz)')
axes[0, 1].set_ylabel('진폭')
axes[0, 1].set_xlim(0, 100)
axes[0, 1].grid(True, alpha=0.3)
# 반송파(fc), 상측파대(fc+fm), 하측파대(fc-fm) 표시
for f_label, label in [(fc, '$f_c$'), (fc+fm, '$f_c+f_m$'), (fc-fm, '$f_c-f_m$')]:
    axes[0, 1].axvline(x=f_label, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].text(f_label, 0.45, label, ha='center', fontsize=9)

# (2) 구형 펄스 열의 스펙트럼
T_pulse = 0.1    # 주기 (s)
tau = 0.02        # 펄스 폭 (s)
t2 = np.linspace(0, 0.5, 10000)

# 구형 펄스 열 생성
pulse_train = np.zeros_like(t2)
for k in range(int(0.5 / T_pulse) + 1):
    center = k * T_pulse
    pulse_train[(t2 >= center) & (t2 < center + tau)] = 1.0

axes[1, 0].plot(t2, pulse_train, 'b-', linewidth=1)
axes[1, 0].set_title(f'구형 펄스 열 (T={T_pulse}s, τ={tau}s)')
axes[1, 0].set_xlabel('시간 (s)')
axes[1, 0].set_ylabel('f(t)')
axes[1, 0].set_ylim(-0.2, 1.4)
axes[1, 0].grid(True, alpha=0.3)

# 구형 펄스 열의 FFT
N_fft2 = len(t2)
freq2 = np.fft.fftfreq(N_fft2, d=t2[1] - t2[0])
S2 = np.fft.fft(pulse_train) / N_fft2

pos_mask2 = freq2 > 0
axes[1, 1].stem(freq2[pos_mask2][:50], 2 * np.abs(S2[pos_mask2][:50]),
                linefmt='r-', markerfmt='ro', basefmt='k-')
axes[1, 1].set_title('구형 펄스 열의 주파수 스펙트럼')
axes[1, 1].set_xlabel('주파수 (Hz)')
axes[1, 1].set_ylabel('진폭')
axes[1, 1].grid(True, alpha=0.3)

# sinc 포락선 표시
f_env = np.linspace(0.1, freq2[pos_mask2][49], 500)
envelope = tau / T_pulse * np.abs(np.sinc(f_env * tau))
axes[1, 1].plot(f_env, 2 * envelope, 'b--', alpha=0.5, linewidth=2, label='sinc 포락선')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('em_wave_fourier.png', dpi=150)
plt.show()
```

---

## 연습 문제

### 문제 1: 푸리에 계수 계산

주기 $2\pi$인 함수 $f(x) = x^2$ ($-\pi < x < \pi$)의 푸리에 급수를 구하시오.

**힌트**: $f(x) = x^2$는 우함수이므로 코사인 급수만 남는다. 부분적분을 두 번 수행하라.

**기대 결과**: $x^2 = \frac{\pi^2}{3} + 4\sum_{n=1}^{\infty} \frac{(-1)^n}{n^2}\cos(nx)$

### 문제 2: 파르세발 정리 응용

문제 1의 결과에 파르세발 정리를 적용하여 $\sum_{n=1}^{\infty} \frac{1}{n^4}$의 값을 구하시오.

**힌트**: $\frac{1}{\pi}\int_{-\pi}^{\pi} x^4 \, dx = \frac{a_0^2}{2} + \sum a_n^2$에서 출발하라.

**기대 결과**: $\sum_{n=1}^{\infty} \frac{1}{n^4} = \frac{\pi^4}{90}$

### 문제 3: 반구간 전개

구간 $[0, \pi]$에서 $f(x) = \cos x$를 **사인 급수**로 전개하시오.

**힌트**: 기함수 확장 후 $b_n$을 계산한다. $n = 1$일 때와 $n \neq 1$일 때를 구분하라.

### 문제 4: 깁스 현상 분석

톱니파 $f(x) = x$ ($-\pi < x < \pi$)의 푸리에 급수에서, $N$항 부분합의 $x = \pi$ 근처 overshoot이 $\text{Si}(\pi) \approx 1.8519$와 관련됨을 보이시오.

**힌트**: $S_N(x)$를 적분 형태로 쓰고, $x \to \pi^-$에서의 극한을 분석하라.

### 문제 5: 물리 응용 — 진동하는 현

길이 $L = 1$ m인 현이 $f(x) = A\sin(\pi x/L)\sin(3\pi x/L)$의 초기 변위를 갖고 정지 상태에서 놓아진다.

(a) 초기 변위를 사인 급수로 전개하시오. (곱을 차로 변환하는 삼각함수 공식 활용)

(b) $u(x, t)$를 구하고, 이 진동이 두 개의 정상파(standing wave)의 합임을 보이시오.

### 문제 6: 열전도

길이 $L$인 막대의 초기 온도가 $u(x, 0) = u_0$(상수)이고, $t > 0$에서 양 끝이 $0$°C로 유지된다.

(a) $u_0$의 사인 급수 전개를 구하시오.

(b) $t \to \infty$에서 온도가 0에 접근하는 속도를 분석하시오. (기본 모드의 감쇠 시간 상수 $\tau_1 = L^2 / (\pi^2 \alpha)$)

```python
# --- 연습 문제 풀이 보조 코드 ---
import numpy as np
from sympy import symbols, cos, sin, pi, integrate, simplify, Rational

x, n = symbols('x n', real=True, positive=True)

# 문제 1: f(x) = x^2의 푸리에 계수
print("=== 문제 1: f(x) = x^2 ===")
a0 = (1 / pi) * integrate(x**2, (x, -pi, pi))
print(f"a_0 = {a0}")

a_n = (1 / pi) * integrate(x**2 * cos(n * x), (x, -pi, pi))
a_n_simplified = simplify(a_n)
print(f"a_n = {a_n_simplified}")

# 문제 2: 파르세발 정리 적용
print("\n=== 문제 2: 파르세발 정리 ===")
lhs = (1 / pi) * integrate(x**4, (x, -pi, pi))
print(f"(1/pi) * integral x^4 dx = {lhs}")

# a_0^2/2 + sum a_n^2 = 2*pi^4/5
# pi^4/9/2 + 16 * sum 1/n^4 = 2*pi^4/5
# => sum 1/n^4 = (2*pi^4/5 - pi^4/18) / 16 = pi^4/90
print(f"sum 1/n^4 = pi^4/90 = {float(pi**4/90):.10f}")
print(f"수치 검증: {sum(1/k**4 for k in range(1, 100001)):.10f}")
```

---

## 참고 자료

### 교재
1. **Boas, M. L.** (2005). *Mathematical Methods in the Physical Sciences*, 3rd ed., Chapter 7. Wiley.
   - 본 레슨의 주요 참고서
2. **Arfken, G. B., Weber, H. J., & Harris, F. E.** (2012). *Mathematical Methods for Physicists*, 7th ed., Chapter 19. Academic Press.
   - 보다 엄밀한 수학적 처리
3. **Kreyszig, E.** (2011). *Advanced Engineering Mathematics*, 10th ed., Chapters 11-12. Wiley.
   - 공학적 관점의 푸리에 해석

### 온라인 자료
1. **3Blue1Brown**: *But what is a Fourier series?* — 직관적인 시각화
2. **MIT OCW 18.03**: Fourier Series 강의
3. **Paul's Online Math Notes**: Fourier Series 참고
4. **Wolfram MathWorld**: Fourier Series 정의 및 공식

### 핵심 공식 요약

| 공식 | 수식 |
|------|------|
| 푸리에 급수 | $f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty}[a_n\cos(nx) + b_n\sin(nx)]$ |
| $a_n$ | $\frac{1}{\pi}\int_{-\pi}^{\pi} f(x)\cos(nx)\,dx$ |
| $b_n$ | $\frac{1}{\pi}\int_{-\pi}^{\pi} f(x)\sin(nx)\,dx$ |
| 복소 형태 | $f(x) = \sum_{n=-\infty}^{\infty} c_n e^{inx}$ |
| 파르세발 | $\frac{1}{\pi}\int_{-\pi}^{\pi}|f|^2\,dx = \frac{a_0^2}{2} + \sum(a_n^2 + b_n^2)$ |

---

## 다음 레슨

[08. 푸리에 변환 (Fourier Transforms)](08_Fourier_Transforms.md)에서는 주기 함수에 국한된 푸리에 급수를 **비주기 함수**로 확장하는 **푸리에 변환(Fourier transform)**을 다룹니다. 연속 푸리에 변환, 이산 푸리에 변환(DFT), 고속 푸리에 변환(FFT), 그리고 컨볼루션 정리를 학습합니다.
