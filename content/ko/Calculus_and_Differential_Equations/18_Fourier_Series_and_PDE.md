# 푸리에 급수와 PDE(Fourier Series and PDE)

## 학습 목표

- 주기 함수의 푸리에 계수를 계산하고 이를 직교 기저 함수에 대한 사영으로 해석할 수 있다
- 경계 조건에 따라 완전 푸리에 급수, 코사인 급수, 사인 급수 중 어느 것을 사용할지 결정할 수 있다
- 변수 분리법과 푸리에 급수를 결합하여 유한 구간에서의 열 방정식을 풀 수 있다
- 푸리에 방법과 달랑베르 공식을 모두 사용하여 파동 방정식을 풀 수 있다
- 파이썬으로 푸리에 급수 계산을 구현하고 PDE 해를 애니메이션으로 만들 수 있다

## 선수 과목

이 레슨을 학습하기 전에 다음 내용을 숙지해야 합니다:
- PDE 입문, 분류, 경계 조건 (레슨 17)
- 부분적분을 포함한 적분 기법 (레슨 1-4)
- 기본적인 삼각함수 항등식과 직교성

## 핵심 아이디어: 단순한 조각들로 분해하기

유한 영역에서 PDE를 풀기 위한 전략은 세 단계로 구성됩니다:

1. **변수 분리**(Separation of variables): $u(x,t) = X(x)T(t)$로 가정하고 PDE를 두 개의 ODE로 분리
2. **공간 ODE 풀기**: 경계 조건이 특정 고유값과 고유함수를 강제
3. **중첩**(Superpose): 푸리에 급수를 사용하여 초기 조건을 고유함수의 합으로 맞추기

이것이 작동하는 이유는 PDE가 선형이기 때문입니다: 해의 합은 다시 해가 됩니다. 백색광을 스펙트럼 색상(푸리에 모드)으로 분해하고, 각 색상이 어떻게 변화하는지를 개별적으로 풀고, 다시 합치는 것과 같다고 생각하면 됩니다.

## 푸리에 급수: 수학적 기초

### 주기 함수와 직교성

주기 $2L$인 함수 $f(x)$는 다음과 같이 표현할 수 있습니다:

$$f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty} \left[a_n \cos\left(\frac{n\pi x}{L}\right) + b_n \sin\left(\frac{n\pi x}{L}\right)\right]$$

이것을 가능하게 하는 핵심 성질은 $[-L, L]$에서의 **직교성**(orthogonality)입니다:

$$\int_{-L}^{L} \cos\left(\frac{m\pi x}{L}\right) \cos\left(\frac{n\pi x}{L}\right) dx = \begin{cases} 0 & m \neq n \\ L & m = n \neq 0 \\ 2L & m = n = 0 \end{cases}$$

$$\int_{-L}^{L} \sin\left(\frac{m\pi x}{L}\right) \sin\left(\frac{n\pi x}{L}\right) dx = \begin{cases} 0 & m \neq n \\ L & m = n \end{cases}$$

$$\int_{-L}^{L} \cos\left(\frac{m\pi x}{L}\right) \sin\left(\frac{n\pi x}{L}\right) dx = 0 \quad \text{모든 } m, n \text{에 대해}$$

이것을 $\mathbb{R}^3$에서의 벡터 분해와 같다고 생각하세요: $\hat{e}_1$ 방향의 벡터 성분을 구하려면 $\hat{e}_1$과 내적을 합니다. 여기서는 "내적"이 적분이고, "기저 벡터"가 사인과 코사인입니다.

### 푸리에 계수

계수는 $f$를 각 기저 함수에 "사영"하여 계산합니다:

$$a_0 = \frac{1}{L} \int_{-L}^{L} f(x) \, dx$$

$$a_n = \frac{1}{L} \int_{-L}^{L} f(x) \cos\left(\frac{n\pi x}{L}\right) dx, \quad n \geq 1$$

$$b_n = \frac{1}{L} \int_{-L}^{L} f(x) \sin\left(\frac{n\pi x}{L}\right) dx, \quad n \geq 1$$

급수에서 $a_0$ 대신 $a_0/2$를 사용하는 것은 $a_0$의 공식을 일반적인 $a_n$ 공식과 일관되게 만들기 위한 관례입니다.

### 수렴: 디리클레 조건

$f$가 구간별 매끄러운 함수(각 주기에서 유한 개의 불연속점과 꺾임점을 가짐)이면 푸리에 급수는 연속점에서 $f(x)$로 수렴합니다. 불연속점에서는 좌극한과 우극한의 평균으로 수렴합니다:

$$\frac{f(x^-) + f(x^+)}{2}$$

불연속점 근처에서 부분합은 **깁스 현상**(Gibbs phenomenon)을 보입니다: 더 많은 항을 추가해도 줄어들지 않는 약 9%의 과대 추정(overshoot)이 나타납니다(과대 추정 영역은 줄어듭니다).

### 우함수와 기함수: 반범위 전개

$f(x)$가 $[0, L]$에서만 정의된 경우(유한 구간에서의 PDE에서 전형적), 다음 중 하나를 얻기 위해 확장할 수 있습니다:

**코사인 급수** (우함수 확장, $b_n = 0$):

$$f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty} a_n \cos\left(\frac{n\pi x}{L}\right), \quad a_n = \frac{2}{L}\int_0^L f(x) \cos\left(\frac{n\pi x}{L}\right) dx$$

경계 조건이 도함수가 0인 경우(노이만): $u_x(0,t) = u_x(L,t) = 0$일 때 적절합니다.

**사인 급수** (기함수 확장, $a_n = 0$):

$$f(x) = \sum_{n=1}^{\infty} b_n \sin\left(\frac{n\pi x}{L}\right), \quad b_n = \frac{2}{L}\int_0^L f(x) \sin\left(\frac{n\pi x}{L}\right) dx$$

경계 조건이 값이 0인 경우(디리클레): $u(0,t) = u(L,t) = 0$일 때 적절합니다.

### 풀이 예제: 구형파

$[0, \pi]$에서 $f(x) = 1$의 푸리에 사인 급수를 구합니다.

$$b_n = \frac{2}{\pi}\int_0^{\pi} 1 \cdot \sin(nx) \, dx = \frac{2}{\pi}\left[-\frac{\cos(nx)}{n}\right]_0^{\pi} = \frac{2}{n\pi}(1 - \cos(n\pi)) = \frac{2}{n\pi}(1 - (-1)^n)$$

따라서 짝수 $n$에서 $b_n = 0$이고 홀수 $n$에서 $b_n = \frac{4}{n\pi}$입니다:

$$f(x) = \frac{4}{\pi}\left(\sin x + \frac{\sin 3x}{3} + \frac{\sin 5x}{5} + \cdots\right)$$

계수의 느린 $1/n$ 감쇠는 구형파의 불연속을 반영합니다. 매끄러운 함수는 빠르게 감쇠하는 계수를 가집니다.

## 열 방정식 풀기

### 문제 설정

$$u_t = \alpha u_{xx}, \quad 0 < x < L, \quad t > 0$$
$$u(0, t) = 0, \quad u(L, t) = 0 \quad \text{(디리클레 경계 조건)}$$
$$u(x, 0) = f(x) \quad \text{(초기 조건)}$$

### 단계 1: 변수 분리

$u(x, t) = X(x)T(t)$로 가정합니다. 대입하면:

$$X(x)T'(t) = \alpha X''(x)T(t)$$

$\alpha X T$로 나누면:

$$\frac{T'(t)}{\alpha T(t)} = \frac{X''(x)}{X(x)}$$

왼쪽은 $t$에만 의존하고 오른쪽은 $x$에만 의존합니다. 모든 $x$와 $t$에 대해 같으므로 둘 다 상수, 예컨대 $-\lambda$와 같아야 합니다:

$$\frac{X''}{X} = -\lambda \quad \Longrightarrow \quad X'' + \lambda X = 0$$

$$\frac{T'}{\alpha T} = -\lambda \quad \Longrightarrow \quad T' + \alpha\lambda T = 0$$

### 단계 2: 공간 문제 풀기 (고유값 문제)

경계 조건 $X(0) = 0$과 $X(L) = 0$에서:

$\lambda > 0$인 경우 ($\lambda = \mu^2$으로 쓰면): $X = A\cos(\mu x) + B\sin(\mu x)$.

$X(0) = 0$에서 $A = 0$. $X(L) = 0$에서 $B\sin(\mu L) = 0$.

비자명 해($B \neq 0$)의 경우: $\sin(\mu L) = 0$이므로 $\mu_n = \frac{n\pi}{L}$.

**고유값**(Eigenvalue): $\lambda_n = \left(\frac{n\pi}{L}\right)^2$, $n = 1, 2, 3, \ldots$

**고유함수**(Eigenfunction): $X_n(x) = \sin\left(\frac{n\pi x}{L}\right)$

### 단계 3: 시간 문제 풀기

각 $n$에 대해: $T_n' + \alpha\lambda_n T_n = 0$이므로:

$$T_n(t) = e^{-\alpha \lambda_n t} = e^{-\alpha n^2 \pi^2 t / L^2}$$

이것은 지수적 감쇠입니다. 고차 모드(더 큰 $n$)는 $\lambda_n \propto n^2$이므로 **더 빠르게** 감쇠합니다. 이것이 열 방정식이 날카로운 특징을 빠르게 평활화하는 이유입니다: 초기 온도의 고주파 성분이 먼저 사라집니다.

### 단계 4: 중첩하여 초기 조건 맞추기

일반해는:

$$u(x, t) = \sum_{n=1}^{\infty} B_n \sin\left(\frac{n\pi x}{L}\right) e^{-\alpha n^2 \pi^2 t / L^2}$$

$t = 0$에서: $u(x, 0) = \sum_{n=1}^{\infty} B_n \sin\left(\frac{n\pi x}{L}\right) = f(x)$.

$B_n$은 $f(x)$의 푸리에 사인 계수입니다:

$$B_n = \frac{2}{L}\int_0^L f(x) \sin\left(\frac{n\pi x}{L}\right) dx$$

**완전한 해**: 임의의 점 $(x, t)$에서의 온도는 초기 온도의 푸리에 분해에 의해 결정되며, 각 모드는 지수적으로 감쇠합니다.

## 파동 방정식 풀기

### 푸리에 방법

$$u_{tt} = c^2 u_{xx}, \quad u(0,t) = u(L,t) = 0, \quad u(x,0) = f(x), \quad u_t(x,0) = g(x)$$

변수 분리는 동일한 고유함수 $\sin(n\pi x / L)$를 주지만, 시간 방정식은 이제 다음과 같습니다:

$$T_n'' + c^2 \lambda_n T_n = 0 \quad \Longrightarrow \quad T_n(t) = A_n \cos(\omega_n t) + B_n \sin(\omega_n t)$$

여기서 $\omega_n = cn\pi/L$은 줄의 **고유 진동수**(natural frequency)입니다. 열 방정식과 달리 이 모드들은 감쇠하지 않고 진동합니다 -- 파동 방정식은 에너지를 보존합니다.

$$u(x,t) = \sum_{n=1}^{\infty} \left[A_n \cos(\omega_n t) + B_n \sin(\omega_n t)\right] \sin\left(\frac{n\pi x}{L}\right)$$

계수는 초기 조건에 의해 결정됩니다:

$$A_n = \frac{2}{L}\int_0^L f(x)\sin\left(\frac{n\pi x}{L}\right) dx, \quad B_n = \frac{2}{\omega_n L}\int_0^L g(x)\sin\left(\frac{n\pi x}{L}\right) dx$$

### 달랑베르 해(D'Alembert's Solution)

무한 영역($-\infty < x < \infty$)에서의 파동 방정식에 대해:

$$u(x,t) = \frac{1}{2}[f(x - ct) + f(x + ct)] + \frac{1}{2c}\int_{x-ct}^{x+ct} g(\xi) \, d\xi$$

이 우아한 공식은 물리를 드러냅니다: 초기 변위는 속도 $c$로 반대 방향으로 이동하는 두 파동으로 분리됩니다. $g$를 포함하는 항은 초기 속도를 설명합니다.

## 직사각형에서의 라플라스 방정식 풀기

$$u_{xx} + u_{yy} = 0, \quad 0 < x < a, \quad 0 < y < b$$

경계 조건: $u(x, 0) = 0$, $u(x, b) = f(x)$, $u(0, y) = 0$, $u(a, y) = 0$.

분리 $u = X(x)Y(y)$를 하면:

$$\frac{X''}{X} = -\frac{Y''}{Y} = -\lambda$$

동차 경계 조건을 가진 $x$-문제는 $X_n = \sin(n\pi x / a)$, $\lambda_n = (n\pi/a)^2$을 줍니다.

$y$-문제: $Y'' - \lambda_n Y = 0$이고 $Y(0) = 0$이면 $Y_n(y) = \sinh(n\pi y / a)$.

$$u(x, y) = \sum_{n=1}^{\infty} C_n \sin\left(\frac{n\pi x}{a}\right) \sinh\left(\frac{n\pi y}{a}\right)$$

$u(x, b) = f(x)$를 맞추면:

$$C_n = \frac{2}{a \sinh(n\pi b/a)} \int_0^a f(x) \sin\left(\frac{n\pi x}{a}\right) dx$$

## 파이썬 구현

```python
"""
Fourier Series and PDE Solutions.

This script demonstrates:
1. Computing and visualizing Fourier coefficients
2. Animated heat equation solution via Fourier series
3. Wave equation with D'Alembert and Fourier solutions
4. Laplace equation on a rectangle
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


# ── 1. Fourier Series Approximation ─────────────────────────
def fourier_sine_coefficients(f, L, N):
    """
    Compute the first N Fourier sine coefficients of f on [0, L].

    b_n = (2/L) * integral_0^L f(x) sin(n*pi*x/L) dx

    We use numerical integration (trapezoidal rule) for generality.
    """
    x = np.linspace(0, L, 1000)
    dx = x[1] - x[0]
    coeffs = []
    for n in range(1, N + 1):
        integrand = f(x) * np.sin(n * np.pi * x / L)
        # Trapezoidal integration — simple but effective for smooth integrands
        b_n = (2.0 / L) * np.trapz(integrand, x)
        coeffs.append(b_n)
    return coeffs


def fourier_sine_series(coeffs, L, x, t=None, alpha=None, mode='static'):
    """
    Evaluate the Fourier sine series at points x.

    Modes:
        'static': sum of b_n * sin(n*pi*x/L)
        'heat':   sum of b_n * sin(n*pi*x/L) * exp(-alpha*(n*pi/L)^2 * t)
        'wave':   sum of b_n * sin(n*pi*x/L) * cos(c*n*pi*t/L)
    """
    result = np.zeros_like(x)
    for n, b_n in enumerate(coeffs, start=1):
        spatial = np.sin(n * np.pi * x / L)
        if mode == 'static':
            result += b_n * spatial
        elif mode == 'heat':
            decay = np.exp(-alpha * (n * np.pi / L)**2 * t)
            result += b_n * spatial * decay
        elif mode == 'wave':
            # alpha parameter reused as wave speed c here
            oscillation = np.cos(alpha * n * np.pi * t / L)
            result += b_n * spatial * oscillation
    return result


# Example: Fourier series of a triangle wave
L = 1.0
f_triangle = lambda x: np.where(x <= L/2, 2*x/L, 2*(L-x)/L)

x_plot = np.linspace(0, L, 500)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot convergence of Fourier series
for N in [1, 3, 5, 15, 50]:
    coeffs = fourier_sine_coefficients(f_triangle, L, N)
    y_approx = fourier_sine_series(coeffs, L, x_plot)
    axes[0, 0].plot(x_plot, y_approx, label=f'N={N}', alpha=0.8)

axes[0, 0].plot(x_plot, f_triangle(x_plot), 'k--', linewidth=2, label='Exact')
axes[0, 0].set_title('Fourier Sine Series Convergence (Triangle Wave)')
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('f(x)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Coefficient magnitudes — shows how quickly coefficients decay
N_show = 20
coeffs_20 = fourier_sine_coefficients(f_triangle, L, N_show)
axes[0, 1].stem(range(1, N_show + 1), np.abs(coeffs_20), basefmt='k-')
axes[0, 1].set_title('|b_n| vs n (Triangle Wave)')
axes[0, 1].set_xlabel('n')
axes[0, 1].set_ylabel('|b_n|')
axes[0, 1].grid(True, alpha=0.3)

# ── 2. Heat Equation via Fourier Series ──────────────────────
alpha_heat = 0.01  # thermal diffusivity
N_modes = 50
coeffs_heat = fourier_sine_coefficients(f_triangle, L, N_modes)

times = [0, 0.5, 2.0, 5.0, 15.0]
colors_heat = plt.cm.hot(np.linspace(0.1, 0.9, len(times)))

for i, t in enumerate(times):
    u = fourier_sine_series(coeffs_heat, L, x_plot, t=t,
                            alpha=alpha_heat, mode='heat')
    axes[1, 0].plot(x_plot, u, color=colors_heat[i], linewidth=2,
                    label=f't = {t}')

axes[1, 0].set_title('Heat Equation: Fourier Series Solution')
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('u(x, t)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# ── 3. Wave Equation via Fourier Series ──────────────────────
c_wave = 1.0  # wave speed
N_modes_wave = 30
# Initial condition: plucked string (triangle)
coeffs_wave = fourier_sine_coefficients(f_triangle, L, N_modes_wave)

times_wave = [0, 0.1, 0.25, 0.5, 0.75, 1.0]
colors_wave = plt.cm.viridis(np.linspace(0, 1, len(times_wave)))

for i, t in enumerate(times_wave):
    u = fourier_sine_series(coeffs_wave, L, x_plot, t=t,
                            alpha=c_wave, mode='wave')
    axes[1, 1].plot(x_plot, u, color=colors_wave[i], linewidth=1.5,
                    label=f't = {t:.2f}')

axes[1, 1].set_title('Wave Equation: Plucked String')
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('u(x, t)')
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fourier_series_pde.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved to fourier_series_pde.png")

# ── 4. Laplace Equation on a Rectangle ──────────────────────
print("\n=== Laplace Equation on [0,a] x [0,b] ===")
a, b_rect = 2.0, 1.0  # rectangle dimensions
N_laplace = 30

# Boundary condition at y = b: f(x) = sin(pi*x/a)
# This is already the n=1 Fourier mode, so only C_1 is nonzero
# C_1 = 1 / sinh(pi * b / a)
x_rect = np.linspace(0, a, 100)
y_rect = np.linspace(0, b_rect, 50)
X_rect, Y_rect = np.meshgrid(x_rect, y_rect)

# For general f(x), compute Fourier coefficients
f_boundary = lambda x: np.sin(np.pi * x / a) + 0.5 * np.sin(3 * np.pi * x / a)

U = np.zeros_like(X_rect)
for n in range(1, N_laplace + 1):
    # Numerical integration for C_n
    integrand = f_boundary(x_rect) * np.sin(n * np.pi * x_rect / a)
    coeff = (2.0 / a) * np.trapz(integrand, x_rect)
    C_n = coeff / np.sinh(n * np.pi * b_rect / a)

    # Add this mode's contribution
    U += C_n * np.sin(n * np.pi * X_rect / a) * np.sinh(n * np.pi * Y_rect / a)

fig2, ax2 = plt.subplots(figsize=(10, 5))
contour = ax2.contourf(X_rect, Y_rect, U, levels=30, cmap='RdBu_r')
plt.colorbar(contour, ax=ax2, label='u(x, y)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Laplace Equation: Steady-State Temperature on Rectangle')
ax2.set_aspect('equal')

plt.tight_layout()
plt.savefig('laplace_rectangle.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved to laplace_rectangle.png")
```

## 요약

| 방법 | 방정식 | 해의 형태 |
|------|--------|----------|
| 푸리에 + 열 | $u_t = \alpha u_{xx}$ | $\sum B_n \sin(n\pi x/L) e^{-\alpha(n\pi/L)^2 t}$ (지수적 감쇠) |
| 푸리에 + 파동 | $u_{tt} = c^2 u_{xx}$ | $\sum [A_n \cos + B_n \sin](\omega_n t) \sin(n\pi x/L)$ (진동) |
| 달랑베르 | $u_{tt} = c^2 u_{xx}$ | $\frac{1}{2}[f(x-ct) + f(x+ct)]$ (진행파) |
| 푸리에 + 라플라스 | $u_{xx} + u_{yy} = 0$ | $\sum C_n \sin(n\pi x/a) \sinh(n\pi y/a)$ |

사인 급수와 코사인 급수의 선택은 경계 조건에 의해 결정됩니다: 디리클레(값이 0)는 사인, 노이만(도함수가 0)은 코사인을 줍니다. 열 방정식과 파동 방정식의 물리적 차이는 시간 인자에 나타납니다: 지수적 감쇠 대 진동.

무한 영역에서의 푸리에 변환과 파르세발 정리를 포함한 더 깊은 푸리에 해석에 대해서는 [물리수학 - 푸리에 급수](../Mathematical_Methods/07_Fourier_Series.md)와 [푸리에 변환](../Mathematical_Methods/08_Fourier_Transform.md)을 참조하세요.

## 연습 문제

1. **푸리에 계수**: $[0, 1]$에서 $f(x) = x(1-x)$의 푸리에 사인 급수를 계산하세요. 짝수 $n$에서 $b_n = 0$임을 보이고 홀수 $n$에 대한 닫힌 형태를 구하세요. $b_n$의 감쇠율이 구형파 예제와 어떻게 비교되나요?

2. **열 방정식**: $[0, \pi]$에서 $u(0,t) = u(\pi,t) = 0$이고 $u(x,0) = 100$인 $u_t = u_{xx}$를 풀으세요. 푸리에 급수 해를 구하고 최대 온도가 $50°$로 떨어지는 데 걸리는 시간을 결정하세요. (추정을 위해 첫 번째 항만 유지하세요.)

3. **파동 방정식**: 길이 $L = 0.65$ m인 기타 줄이 $x = L/4$에서 초기 변위 $f(x) = \begin{cases} 4x/L & 0 \leq x \leq L/4 \\ 4(L-x)/(3L) & L/4 < x \leq L \end{cases}$로 뽑힌 후 정지 상태에서 놓아집니다. $c = 300$ m/s이면 푸리에 급수 해의 처음 네 개의 0이 아닌 항을 구하세요. 대응하는 주파수는 Hz로 얼마인가요?

4. **라플라스 방정식**: $[0, 1] \times [0, 1]$에서 $u(x,0) = 0$, $u(x,1) = \sin(\pi x)$, $u(0,y) = u(1,y) = 0$인 $u_{xx} + u_{yy} = 0$을 풀으세요. 중심점 $(1/2, 1/2)$에서의 해가 평균값 성질을 만족하는지 확인하세요: $u(1/2, 1/2)$를 그 점 중심의 작은 원 위에서의 $u$의 평균과 비교하세요.

5. **깁스 현상**: $[0, \pi]$에서 $f(x) = 1$의 부분 푸리에 사인 급수를 $N = 10, 50, 200$에 대해 그리는 파이썬 스크립트를 작성하세요. $x = 0$과 $x = \pi$ 근처에서의 과대 추정을 측정하세요. $N$에 관계없이 과대 추정이 약 $9\%$로 수렴하는지 확인하세요.

---

*이전: [편미분방정식 입문](./17_Introduction_to_PDE.md) | 다음: [미분방정식의 수치 해법](./19_Numerical_Methods_for_DE.md)*
