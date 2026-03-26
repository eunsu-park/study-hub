# 12. 스투름-리우빌 이론 (Sturm-Liouville Theory)

## 학습 목표

- **스투름-리우빌(Sturm-Liouville) 문제**의 표준 형태를 이해하고, 일반적인 2차 ODE를 자기수반 형태로 변환할 수 있다
- 고유값의 **실수성**, 고유함수의 **직교성**, **완비성** 등 S-L 정리의 핵심 결과를 증명하고 활용할 수 있다
- **일반 푸리에 급수(Generalized Fourier Series)**로 임의의 함수를 고유함수 전개할 수 있다
- 삼각함수, 베셀 함수, 르장드르 다항식이 각각 어떤 S-L 문제의 고유함수인지 파악할 수 있다
- **가중 내적(weighted inner product)**과 **그람-슈미트 직교화**를 수행할 수 있다
- 열방정식, 진동 문제, 양자역학 등 **물리학 응용**에서 S-L 이론의 역할을 이해한다

---

## 1. 스투름-리우빌 문제

### 1.1 자기수반 형태 (Self-Adjoint Form)

**스투름-리우빌 방정식**은 다음과 같은 2차 상미분방정식의 표준 형태이다:

$$\frac{d}{dx}\left[p(x)\frac{dy}{dx}\right] + q(x)y + \lambda w(x) y = 0$$

여기서:
- $p(x) > 0$: 계수 함수 (구간의 끝점에서 0이 될 수 있음)
- $q(x)$: 포텐셜 항
- $w(x) > 0$: **가중 함수(weight function)** 또는 **밀도 함수**
- $\lambda$: **고유값(eigenvalue)** 파라미터

연산자 표기법으로 쓰면:

$$\mathcal{L}[y] = -\lambda w(x) y, \quad \mathcal{L} = \frac{d}{dx}\left[p(x)\frac{d}{dx}\right] + q(x)$$

$\mathcal{L}$을 **스투름-리우빌 연산자**라 부르며, 적절한 경계 조건 아래서 **자기수반(self-adjoint)** 성질을 갖는다.

**일반 2차 ODE를 S-L 형태로 변환**: $a(x)y'' + b(x)y' + c(x)y + \lambda d(x)y = 0$에 **적분 인자** $\mu(x) = \frac{1}{a}\exp\left(\int \frac{b}{a}dx\right)$를 곱하면 $p = \mu a$, $q = \mu c$, $w = \mu d$이다.

```python
import sympy as sp

x = sp.Symbol('x')
# 예: 에르미트 방정식 y'' - 2xy' + 2ny = 0을 S-L 형태로 변환
a_x, b_x = sp.Integer(1), -2 * x

integrating_factor = sp.exp(sp.integrate(b_x / a_x, x))
p_x = integrating_factor * a_x
print(f"p(x) = {p_x}")   # exp(-x^2)
print(f"w(x) = {integrating_factor}")  # exp(-x^2)
# S-L 형태: d/dx[exp(-x^2) y'] + 2n exp(-x^2) y = 0
```

### 1.2 경계 조건

S-L 문제는 구간 $[a, b]$에서 정의되며, **경계 조건(boundary conditions)**이 함께 주어진다.

**정칙 S-L 문제 (Regular)**: $p, q, w$ 연속, $p > 0$, $w > 0$ on $[a,b]$, 분리된 경계 조건:

$$\alpha_1 y(a) + \alpha_2 y'(a) = 0, \quad \beta_1 y(b) + \beta_2 y'(b) = 0$$

**특이 S-L 문제 (Singular)**: $p(x)$가 끝점에서 0이거나 구간이 무한한 경우. 해의 **유계성** 또는 **제곱적분 가능성**이 경계 조건을 대체한다.

| 유형 | 조건 | 물리적 예 |
|---|---|---|
| 디리클레 (Dirichlet) | $y(a) = 0, \; y(b) = 0$ | 양 끝 고정된 현 |
| 노이만 (Neumann) | $y'(a) = 0, \; y'(b) = 0$ | 양 끝 자유로운 현 |
| 로빈 (Robin) | $\alpha y + \beta y' = 0$ | 대류 열전달 |
| 주기적 (Periodic) | $y(a) = y(b), \; y'(a) = y'(b)$ | 원형 경계 |

### 1.3 고유값과 고유함수

S-L 방정식을 경계 조건과 함께 풀면, 특정 $\lambda$ 값에서만 비자명해가 존재한다:

$$\mathcal{L}[y_n] = -\lambda_n w(x) y_n, \quad n = 0, 1, 2, \ldots$$

**예제**: $y'' + \lambda y = 0$, $y(0) = 0$, $y(L) = 0$ ($p=1, q=0, w=1$)

고유값: $\lambda_n = (n\pi/L)^2$, 고유함수: $y_n(x) = \sin(n\pi x/L)$, $n = 1, 2, 3, \ldots$

```python
import numpy as np
import matplotlib.pyplot as plt

L = np.pi
x = np.linspace(0, L, 500)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for n, ax in zip(range(1, 5), axes.flat):
    y_n = np.sin(n * x)
    ax.plot(x, y_n, 'b-', linewidth=2)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.set_title(f'$y_{n}(x) = \\sin({n}x)$, $\\lambda_{n} = {n**2}$')
    ax.grid(True, alpha=0.3)

plt.suptitle('디리클레 경계 조건의 고유함수들', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

---

## 2. 스투름-리우빌 정리

### 2.1 고유값의 실수성과 고유함수의 직교성

**정리 1 (실수성)**: 정칙 S-L 문제의 모든 고유값은 **실수**이다.

**증명 개요**: $y$의 방정식에 $\bar{y}$를, $\bar{y}$의 방정식에 $y$를 곱하고 빼면:

$$(\lambda - \bar{\lambda})\int_a^b w|y|^2 dx = [p(\bar{y}y' - y\bar{y}')]_a^b = 0$$

$w > 0$, $|y|^2 > 0$이므로 $\lambda = \bar{\lambda}$. $\blacksquare$

**정리 2 (직교성)**: $\lambda_m \neq \lambda_n$이면 고유함수는 가중 함수에 대해 직교한다:

$$\int_a^b w(x) y_m(x) y_n(x) \, dx = 0$$

**증명**: 같은 방법으로 $(\lambda_m - \lambda_n)\int_a^b w y_m y_n dx = 0$이고, $\lambda_m \neq \lambda_n$이므로 적분 = 0. $\blacksquare$

```python
import numpy as np

L = np.pi
x = np.linspace(0, L, 10000)

print("=== 직교성 검증: int_0^pi sin(mx) sin(nx) dx ===")
for m in range(1, 5):
    for n in range(1, 5):
        inner = np.trapz(np.sin(m*x) * np.sin(n*x), x)
        status = "= pi/2" if m == n else "= 0"
        print(f"  <y_{m}, y_{n}> = {inner:.6f}  ({status})")
```

### 2.2 고유함수의 완비성 (Completeness)

**정리 3**: 정칙 S-L 문제의 고유함수 $\{y_n\}$은 $L^2_w[a,b]$에서 **완비**하다. 임의의 구분적 매끄러운 함수 $f(x)$에 대해:

$$f(x) = \sum_{n=1}^{\infty} c_n y_n(x), \quad \lim_{N\to\infty} \int_a^b w \left| f - \sum_{n=1}^{N} c_n y_n \right|^2 dx = 0$$

**파르세발 등식**: $\int_a^b w |f|^2 dx = \sum_{n=1}^{\infty} |c_n|^2 \|y_n\|^2$

### 2.3 일반 푸리에 급수 (Generalized Fourier Series)

임의의 함수를 고유함수 $\{y_n\}$으로 전개하는 것을 **일반 푸리에 급수**라 한다:

$$f(x) = \sum_{n=1}^{\infty} c_n y_n(x), \quad c_n = \frac{\int_a^b w(x) f(x) y_n(x) dx}{\int_a^b w(x) y_n^2(x) dx}$$

이것은 벡터 정사영 $c_n = \frac{\mathbf{f} \cdot \mathbf{e}_n}{\mathbf{e}_n \cdot \mathbf{e}_n}$의 함수 공간 일반화이다.

```python
import numpy as np
import matplotlib.pyplot as plt

L = np.pi
x = np.linspace(0, L, 1000)
f = lambda t: t  # f(x) = x

def compute_cn(n):
    return 2.0 / L * np.trapz(f(x) * np.sin(n * x), x)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for N, ax in zip([1, 3, 10, 50], axes.flat):
    approx = sum(compute_cn(n) * np.sin(n*x) for n in range(1, N+1))
    ax.plot(x, f(x), 'k--', linewidth=1.5, label='$f(x) = x$')
    ax.plot(x, approx, 'b-', linewidth=2, label=f'$N = {N}$')
    ax.set_title(f'일반 푸리에 급수 ($N = {N}$)')
    ax.legend(); ax.grid(True, alpha=0.3)

plt.suptitle('$f(x) = x$의 고유함수 전개', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

### 2.4 레일리 몫과 변분 원리 (Rayleigh Quotient)

S-L 고유값을 **변분적**으로 특성화할 수 있다. 경계 조건을 만족하는 함수 $y$에 대한 **레일리 몫(Rayleigh quotient)**:

$$R[y] = \frac{-[p y y']_a^b + \int_a^b \left[p(y')^2 - q y^2\right] dx}{\int_a^b w y^2 dx}$$

디리클레 경계 조건 $y(a) = y(b) = 0$에서 경계 항이 사라지므로:

$$R[y] = \frac{\int_a^b \left[p(y')^2 - q y^2\right] dx}{\int_a^b w y^2 dx}$$

**핵심 성질**:
- **최소 원리**: $R[y] \geq \lambda_1$ (최소 고유값). 등호는 $y = y_1$일 때 성립
- **정확성**: $R[y_n] = \lambda_n$ (고유함수에서 정확한 고유값)
- **상한**: 임의의 시행 함수로 $\lambda_1$의 상한(upper bound)을 얻을 수 있다

이 성질은 **리츠 방법(Ritz method)**의 기초이다: 시행 함수의 매개변수를 조절하여 $R$을 최소화하면 고유값의 좋은 근사를 얻는다.

```python
import numpy as np

L = np.pi
x = np.linspace(0, L, 1000)

# y'' + λy = 0, y(0) = y(π) = 0  →  p=1, q=0, w=1
# 정확한 λ₁ = 1

# 시행 함수 1: φ₁(x) = x(π - x)
phi1 = x * (L - x)
phi1_prime = L - 2*x
R1 = np.trapz(phi1_prime**2, x) / np.trapz(phi1**2, x)

# 시행 함수 2: φ₂(x) = sin(πx/L) (정확한 고유함수)
phi2 = np.sin(np.pi * x / L)
phi2_prime = (np.pi / L) * np.cos(np.pi * x / L)
R2 = np.trapz(phi2_prime**2, x) / np.trapz(phi2**2, x)

print(f"시행 함수 1: x(π-x)")
print(f"  R₁ = {R1:.6f}  (정확값 1.0, 오차 {abs(R1-1)*100:.2f}%)")
print(f"시행 함수 2: sin(πx/π) (정확)")
print(f"  R₂ = {R2:.6f}  (정확값 1.0)")
print(f"\nR ≥ λ₁ 확인: {R1:.6f} ≥ {1.0} ✓")
```

---

## 3. 주요 예제

### 3.1 삼각함수 (푸리에 급수 재해석)

일반적인 **푸리에 급수**는 S-L 문제 $y'' + \lambda y = 0$에 주기적 경계 조건을 부여한 것이다:

| 고유값 | 고유함수 | 정규화 상수 |
|---|---|---|
| $\lambda_0 = 0$ | $y_0 = 1$ | $\|y_0\|^2 = 2L$ |
| $\lambda_n = (n\pi/L)^2$ | $\cos(n\pi x/L), \; \sin(n\pi x/L)$ | $\|y_n\|^2 = L$ |

따라서 $[-L, L]$에서의 푸리에 급수 $f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty}[a_n \cos\frac{n\pi x}{L} + b_n \sin\frac{n\pi x}{L}]$는 정확히 S-L 고유함수에 의한 일반 푸리에 급수이다.

### 3.2 베셀 함수의 S-L 형태

**베셀 방정식** $x^2 y'' + xy' + (x^2 - \nu^2) y = 0$을 S-L 형태로 쓰면:

$$\frac{d}{dx}\left[x\frac{dy}{dx}\right] - \frac{\nu^2}{x}y + \lambda x \, y = 0$$

$p(x) = x$, $q(x) = -\nu^2/x$, $w(x) = x$. $p(0) = 0$이므로 **특이 S-L 문제**이다.

경계 조건: $|y(0)| < \infty$, $y(a) = 0$. 고유값 $\lambda_{n\nu} = (\alpha_{n\nu}/a)^2$ ($\alpha_{n\nu}$는 $J_\nu$의 영점).

**직교성**: $\int_0^a x J_\nu(\frac{\alpha_{m\nu}}{a}x) J_\nu(\frac{\alpha_{n\nu}}{a}x) dx = 0$ ($m \neq n$)

```python
from scipy.special import jn_zeros, jv
import numpy as np
import matplotlib.pyplot as plt

nu, a = 0, 1.0
zeros = jn_zeros(nu, 5)
x = np.linspace(0, a, 500)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for n in range(5):
    ax1.plot(x, jv(nu, zeros[n]*x), linewidth=2,
             label=f'$J_0(\\alpha_{{{n+1}}} x)$, $\\alpha={zeros[n]:.2f}$')
ax1.axhline(0, color='k', linewidth=0.5)
ax1.set_title('베셀 함수 고유함수'); ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

# 직교성 행렬
ortho = np.array([[np.trapz(x * jv(nu, zeros[m]*x) * jv(nu, zeros[n]*x), x)
                    for n in range(5)] for m in range(5)])
ax2.imshow(ortho, cmap='RdBu_r', vmin=-0.1, vmax=0.3)
ax2.set_title('직교성 행렬 (가중 $w=x$)'); plt.colorbar(ax2.images[0], ax=ax2)
plt.tight_layout(); plt.show()
```

### 3.3 르장드르 다항식의 S-L 형태

**르장드르 방정식** $(1-x^2)y'' - 2xy' + l(l+1)y = 0$의 S-L 형태:

$$\frac{d}{dx}\left[(1-x^2)\frac{dy}{dx}\right] + l(l+1)y = 0$$

$p(x) = 1-x^2$, $q = 0$, $w = 1$, $\lambda = l(l+1)$. $p(\pm 1) = 0$이므로 특이 S-L 문제.

유계 조건을 요구하면 $l = 0, 1, 2, \ldots$일 때만 해가 존재: **르장드르 다항식** $P_l(x)$.

**직교성**: $\int_{-1}^{1} P_m(x) P_n(x) dx = \frac{2}{2n+1} \delta_{mn}$

```python
from scipy.special import legendre
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1, 1, 500)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for l in range(6):
    ax1.plot(x, legendre(l)(x), linewidth=2, label=f'$P_{l}(x)$')
ax1.set_title('르장드르 다항식'); ax1.legend(); ax1.grid(True, alpha=0.3)

N = 6
ortho = np.array([[np.trapz(legendre(m)(x)*legendre(n)(x), x)
                    for n in range(N)] for m in range(N)])
ax2.imshow(ortho, cmap='RdBu_r', vmin=-0.2, vmax=1.0)
ax2.set_title('직교성 행렬 $\\langle P_m, P_n \\rangle$')
plt.tight_layout(); plt.show()
```

### 3.4 특이 스투름-리우빌 문제 상세

**특이 S-L 문제**는 정칙 조건이 위반되는 경우이며, 물리에서 오히려 더 자주 등장한다. 세 가지 원인:

1. **끝점에서 $p(x) = 0$**: 르장드르, 베셀 등
2. **구간이 무한**: 에르미트 $(-\infty, \infty)$, 라게르 $[0, \infty)$
3. **$q(x)$ 또는 $w(x)$의 발산**: 체비셰프의 $w = (1-x^2)^{-1/2}$

경계 조건은 해의 **유계성** 또는 **제곱적분 가능성** $\int_a^b w |y|^2 dx < \infty$으로 대체된다.

| 방정식 | 구간 | 특이 원인 | 경계 조건 |
|---|---|---|---|
| 베셀 | $[0, a]$ | $p(0) = 0$ | $|y(0)| < \infty$, $y(a) = 0$ |
| 르장드르 | $[-1, 1]$ | $p(\pm 1) = 0$ | $|y(\pm 1)| < \infty$ |
| 에르미트 | $(-\infty, \infty)$ | 무한 구간 | $y \to 0$ as $|x| \to \infty$ |
| 라게르 | $[0, \infty)$ | 반무한 | $|y(0)| < \infty$, $y \to 0$ as $x \to \infty$ |
| 체비셰프 | $[-1, 1]$ | $w(\pm 1) = \infty$ | $|y(\pm 1)| < \infty$ |

**핵심 결과 (특이 S-L 정리)**: 적절한 경계 조건 하에서, 특이 S-L 문제도 정칙 경우의 모든 핵심 정리(실수 고유값, 직교성, 완비성)가 성립한다.

> **증명의 핵심**: 경계 항 $[p(\bar{y}_m y_n' - y_n \bar{y}_m')]_a^b$이 특이 경계 조건에 의해 자동으로 0이 된다. 예를 들어 르장드르의 경우 $p(\pm 1) = 1 - (\pm 1)^2 = 0$이므로 경계 항이 소멸한다.

```python
import numpy as np
from scipy.special import eval_hermite
import matplotlib.pyplot as plt

# 에르미트 함수: 무한 구간 특이 S-L
# d/dx[e^{-x²} y'] + 2n e^{-x²} y = 0
x = np.linspace(-5, 5, 500)
w = np.exp(-x**2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for n in range(5):
    Hn = eval_hermite(n, x)
    ax1.plot(x, Hn * np.sqrt(w), linewidth=1.5,
             label=f'$H_{n}(x) e^{{-x^2/2}}$')

ax1.set_title('에르미트 함수 (가중된 고유함수)')
ax1.set_xlabel('x'); ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# 제곱적분 가능성: ∫w|H_n|² dx < ∞
for n in range(4):
    Hn = eval_hermite(n, x)
    integrand = w * Hn**2
    ax2.fill_between(x, 0, integrand, alpha=0.3, label=f'$w H_{n}^2$')
    area = np.trapz(integrand, x)
    print(f"||H_{n}||²_w = {area:.4f}  (이론: {np.sqrt(np.pi) * 2**n * np.math.factorial(n):.4f})")

ax2.set_title('$w(x)|H_n(x)|^2$ (제곱적분 가능)')
ax2.set_xlabel('x'); ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
```

---

## 4. 가중 함수와 내적

### 4.1 가중 내적 (Weighted Inner Product)

**가중 내적**: $\langle f, g \rangle_w = \int_a^b w(x) f(x) g(x) dx$, **노름**: $\|f\|_w = \sqrt{\langle f, f \rangle_w}$

| S-L 문제 | 구간 | 가중 함수 $w(x)$ | 고유함수 |
|---|---|---|---|
| 삼각함수 (푸리에) | $[-L, L]$ | $1$ | $\cos, \sin$ |
| 르장드르 | $[-1, 1]$ | $1$ | $P_l(x)$ |
| 체비셰프 | $[-1, 1]$ | $(1-x^2)^{-1/2}$ | $T_n(x)$ |
| 에르미트 | $(-\infty, \infty)$ | $e^{-x^2}$ | $H_n(x)$ |
| 라게르 | $[0, \infty)$ | $e^{-x}$ | $L_n(x)$ |
| 베셀 | $[0, a]$ | $x$ | $J_\nu(\alpha_{n\nu}x/a)$ |

```python
import numpy as np
from scipy.special import eval_hermite
import math

# 에르미트 다항식의 가중 직교성 검증
x = np.linspace(-5, 5, 5000)
w = np.exp(-x**2)

print("=== 에르미트 다항식 가중 직교성 ===")
for m in range(5):
    for n in range(m, 5):
        inner = np.trapz(w * eval_hermite(m, x) * eval_hermite(n, x), x)
        if m == n:
            theory = math.sqrt(math.pi) * 2**n * math.factorial(n)
            print(f"  <H_{m}, H_{n}>_w = {inner:.4f}  (이론: {theory:.4f})")
        elif abs(inner) > 0.01:
            print(f"  <H_{m}, H_{n}>_w = {inner:.4f}  (should be ~0)")
```

### 4.2 그람-슈미트 직교화 (Gram-Schmidt)

일차독립 함수 $\{f_0, f_1, \ldots\}$로부터 가중 직교 함수 $\{\phi_n\}$을 구성:

$$\phi_0 = f_0, \quad \phi_n = f_n - \sum_{k=0}^{n-1} \frac{\langle f_n, \phi_k \rangle_w}{\langle \phi_k, \phi_k \rangle_w} \phi_k$$

**예**: $\{1, x, x^2, \ldots\}$에 $w=1$, 구간 $[-1,1]$ 적용 $\rightarrow$ **르장드르 다항식**.

```python
import numpy as np
import matplotlib.pyplot as plt

def gram_schmidt(basis_funcs, x, w):
    ortho = []
    for f_n in basis_funcs:
        phi = f_n(x).copy()
        for phi_k in ortho:
            coeff = np.trapz(w * f_n(x) * phi_k, x) / np.trapz(w * phi_k**2, x)
            phi -= coeff * phi_k
        ortho.append(phi)
    return ortho

x = np.linspace(-1, 1, 2000)
monomials = [lambda x, k=k: x**k for k in range(5)]
ortho_polys = gram_schmidt(monomials, x, np.ones_like(x))

# 직교성 검증
N = len(ortho_polys)
mat = np.array([[np.trapz(ortho_polys[i]*ortho_polys[j], x)
                 for j in range(N)] for i in range(N)])

plt.figure(figsize=(6, 5))
plt.imshow(np.abs(mat), cmap='viridis', norm=plt.matplotlib.colors.LogNorm(1e-15, 2))
plt.title('그람-슈미트 직교성 행렬 $|\\langle\\phi_m, \\phi_n\\rangle|$')
plt.colorbar(); plt.show()
```

### 4.3 스투름 비교 정리 (Sturm Comparison Theorem)

**스투름 비교 정리**는 서로 다른 포텐셜 아래서의 해의 **진동 성질**을 비교하는 도구이다.

**정리**: 같은 $p(x)$를 가진 두 방정식:

$$[p(x)u']' + q_1(x) u = 0, \quad [p(x)v']' + q_2(x) v = 0$$

에서 $q_1(x) \geq q_2(x)$ ($q_1 \not\equiv q_2$)이면, $v$의 연속하는 두 영점 사이에 $u$의 영점이 적어도 하나 존재한다.

**직관**: "복원력"($q$)이 강할수록 해가 더 빨리 진동한다.

**물리적 응용**:
- **양자역학**: 에너지 $E$ 높은 상태 → 유효 $q(x)$ 큰 → 파동함수 영점(node) 증가
- **일반 결과**: $n$번째 고유함수는 정확히 $n-1$개의 영점을 가짐 (진동 정리)

```python
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# y'' + q(x) y = 0  두 가지 경우 비교
def ode(t, Y, q_func):
    y, yp = Y
    return [yp, -q_func(t) * y]

x_span = (0, 4*np.pi)
x_eval = np.linspace(*x_span, 2000)
y0 = [0, 1]

# q₁ = 4 (빠른 진동), q₂ = 1 (느린 진동)
sol1 = solve_ivp(ode, x_span, y0, args=(lambda x: 4.0,),
                 t_eval=x_eval, rtol=1e-10)
sol2 = solve_ivp(ode, x_span, y0, args=(lambda x: 1.0,),
                 t_eval=x_eval, rtol=1e-10)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(sol1.t, sol1.y[0], 'b-', linewidth=1.5, label='$q_1 = 4$: $\\sin(2x)$')
ax.plot(sol2.t, sol2.y[0], 'r-', linewidth=1.5, label='$q_2 = 1$: $\\sin(x)$')
ax.axhline(0, color='k', linewidth=0.5)
ax.set_xlabel('x'); ax.set_ylabel('y(x)')
ax.set_title('스투름 비교 정리: $q_1 > q_2$이면 해가 더 빠르게 진동')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# 영점 개수 비교 (0, 4π 구간)
zeros1 = np.where(np.diff(np.sign(sol1.y[0])))[0]
zeros2 = np.where(np.diff(np.sign(sol2.y[0])))[0]
print(f"q₁=4 해의 영점 수: {len(zeros1)}  (이론: {int(4*np.pi/(np.pi/2))})")
print(f"q₂=1 해의 영점 수: {len(zeros2)}  (이론: {int(4*np.pi/np.pi)})")
```

---

## 5. 물리학 응용

### 5.1 열방정식의 고유함수 전개

길이 $L$인 막대의 **열방정식**: $\frac{\partial u}{\partial t} = \alpha^2 \frac{\partial^2 u}{\partial x^2}$

경계: $u(0,t) = u(L,t) = 0$, 초기: $u(x,0) = f(x)$

변수 분리 $u = X(x)T(t)$ $\rightarrow$ S-L 문제 $X'' + \lambda X = 0$:

$$u(x, t) = \sum_{n=1}^{\infty} b_n \sin\frac{n\pi x}{L} e^{-\alpha^2 (n\pi/L)^2 t}, \quad b_n = \frac{2}{L}\int_0^L f(x) \sin\frac{n\pi x}{L} dx$$

```python
import numpy as np
import matplotlib.pyplot as plt

L, alpha2, N = 1.0, 0.01, 50
x = np.linspace(0, L, 200)
f = lambda t: np.sin(np.pi*t/L) + 0.5*np.sin(3*np.pi*t/L)

def bn(n):
    return 2/L * np.trapz(f(x) * np.sin(n*np.pi*x/L), x)

def u(x_val, t):
    return sum(bn(n)*np.sin(n*np.pi*x_val/L)*np.exp(-alpha2*(n*np.pi/L)**2*t)
               for n in range(1, N+1))

fig, ax = plt.subplots(figsize=(10, 6))
for t_val, c in zip([0, 0.5, 2, 5, 10, 30], plt.cm.coolwarm(np.linspace(0,1,6))):
    ax.plot(x, u(x, t_val), color=c, linewidth=2, label=f'$t={t_val}$')

ax.set_xlabel('$x$'); ax.set_ylabel('$u(x,t)$')
ax.set_title('열방정식의 고유함수 전개 해', fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
```

### 5.2 진동 문제의 정규 모드

양 끝 고정된 현의 **파동방정식**: $\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}$

같은 S-L 문제에서 시간 부분이 다르다: $T_n(t) = A_n\cos(\omega_n t) + B_n\sin(\omega_n t)$

$\omega_n = cn\pi/L$은 $n$번째 **고유 진동수**, $X_n T_n$이 **정규 모드(normal mode)**.

```python
import numpy as np
import matplotlib.pyplot as plt

L, c = 1.0, 1.0
x = np.linspace(0, L, 300)

fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
for mode_n, ax in zip([1, 2, 3], axes):
    omega = c * mode_n * np.pi / L
    X_n = np.sin(mode_n * np.pi * x / L)
    for t in [0, 0.1, 0.25, 0.4]:
        ax.plot(x, X_n*np.cos(omega*t), linewidth=1.5, label=f't={t}')
    ax.set_ylabel(f'Mode {mode_n}')
    ax.set_title(f'$\\omega_{mode_n} = {omega:.3f}$ rad/s')
    ax.legend(fontsize=8, ncol=4); ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('$x$')
plt.suptitle('현의 정규 모드', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()
```

### 5.3 양자역학의 에너지 고유상태

**시간 독립 슈뢰딩거 방정식**: $-\frac{\hbar^2}{2m}\psi'' + V(x)\psi = E\psi$

S-L 형태: $p = \hbar^2/(2m)$, $q = -V(x)$, $w = 1$, $\lambda = E$

**무한 사각 우물** ($V=0$ for $0<x<L$):
- 에너지: $E_n = \frac{n^2\pi^2\hbar^2}{2mL^2}$
- 파동함수: $\psi_n(x) = \sqrt{2/L}\sin(n\pi x/L)$

**S-L의 물리적 의미**:
- 실수 고유값 $\rightarrow$ 관측량(에너지)은 실수
- 직교성 $\rightarrow$ 서로 다른 에너지 상태는 직교
- 완비성 $\rightarrow$ 임의의 상태 = 에너지 고유상태의 중첩

```python
import numpy as np
import matplotlib.pyplot as plt

L, hbar, m = 1.0, 1.0, 1.0
x = np.linspace(0, L, 500)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for n in range(1, 6):
    E_n = (n*np.pi*hbar)**2 / (2*m*L**2)
    psi = np.sqrt(2/L) * np.sin(n*np.pi*x/L)
    # 파동함수 (에너지 기준선 위에 표시)
    ax1.axhline(E_n, color='gray', linewidth=0.5, linestyle='--')
    ax1.plot(x, E_n + 3*psi, linewidth=2, label=f'$n={n}, E={E_n:.1f}$')
    ax1.fill_between(x, E_n, E_n + 3*psi, alpha=0.1)
    # 확률 밀도
    ax2.plot(x, psi**2, linewidth=2, label=f'$|\\psi_{n}|^2$')

ax1.set_title('에너지 고유상태'); ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
ax2.set_title('확률 밀도'); ax2.legend(); ax2.grid(True, alpha=0.3)
plt.suptitle('양자역학: S-L 이론의 응용', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()

# 직교 정규성 검증
print("\n=== 양자 고유상태 직교 정규성 ===")
for i in range(1, 4):
    for j in range(i, 4):
        psi_i = np.sqrt(2/L)*np.sin(i*np.pi*x/L)
        psi_j = np.sqrt(2/L)*np.sin(j*np.pi*x/L)
        print(f"  <psi_{i}|psi_{j}> = {np.trapz(psi_i*psi_j, x):.6f}")
```

**조화 진동자**: $V = \frac{1}{2}m\omega^2 x^2$ 경우, 에르미트 방정식으로 환원되며 고유함수는 에르미트 함수, $E_n = \hbar\omega(n+1/2)$.

---

## 연습 문제

### 기본 문제

**문제 1.** 다음을 S-L 표준 형태로 변환하고 $p, q, w$를 구하라.
(a) 체비셰프: $(1-x^2)y'' - xy' + n^2 y = 0$
(b) 라게르: $xy'' + (1-x)y' + ny = 0$

**문제 2.** $y'' + \lambda y = 0$, $y'(0) = 0$, $y'(L) = 0$ (노이만 경계 조건)의 고유값과 고유함수를 구하라.

**문제 3.** $f(x) = \begin{cases} 1 & 0 < x < \pi/2 \\ 0 & \pi/2 < x < \pi \end{cases}$를 $[0, \pi]$에서 $\sin(nx)$ 급수로 전개하라.

### 심화 문제

**문제 4.** $\{1, x, x^2\}$에 $w(x) = (1-x^2)^{-1/2}$, 구간 $[-1,1]$로 그람-슈미트를 수행하여 체비셰프 다항식 $T_0, T_1, T_2$를 유도하라.

**문제 5.** 초기 온도 $f(x) = 4x(1-x)$, $L=1$, $\alpha^2=0.01$일 때 열방정식 해를 구하고 $t=0, 0.1, 1, 10$에서 온도를 그래프로 그려라.

**문제 6 (양자역학).** 무한 사각 우물에서 $\Psi(x,0) = Ax(L-x)$일 때:
(a) 정규화 상수 $A$, (b) 전개 계수 $c_n$, (c) $\langle E \rangle = \sum_n |c_n|^2 E_n$.

**문제 7 (레일리 몫).** $y'' + \lambda y = 0$, $y(0) = y(L) = 0$ 문제에서 시행 함수 $\phi(x) = x^2(L-x)^2$의 레일리 몫을 해석적으로 계산하고, 정확한 $\lambda_1 = (\pi/L)^2$와 비교하라. (힌트: $\int_0^L [x^2(L-x)^2]^2 dx$와 $\int_0^L [\phi']^2 dx$를 각각 계산)

**문제 8 (스투름 비교).** 슈뢰딩거 방정식 $\psi'' + [E - V(x)]\psi = 0$에서 유효 포텐셜 $q(x) = E - V(x)$로 해석할 때, $V(x) = x^2$ (조화 진동자)의 $E_3$ 상태 파동함수가 가지는 영점의 수를 스투름 비교 정리로 예측하라.

---

## 참고 자료

- **Boas, M. L.** *Mathematical Methods in the Physical Sciences*, 3rd Ed., Ch. 12
- **Arfken, Weber, Harris.** *Mathematical Methods for Physicists*, 7th Ed., Ch. 10
- **Riley, Hobson, Bence.** *Mathematical Methods for Physics and Engineering*, Ch. 17
- **Zettl, A.** *Sturm-Liouville Theory*, AMS (2005)

---

## 다음 레슨

- [13. 편미분방정식 (Partial Differential Equations)](13_Partial_Differential_Equations.md) - S-L 이론이 변수 분리법의 핵심 도구로 활용되는 PDE 풀이