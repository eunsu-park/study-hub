# ODE의 멱급수 해법(Power Series Solutions of ODE)

## 학습 목표

- ODE의 특이점을 정상점(ordinary point), 정칙 특이점(regular singular point), 비정칙 특이점(irregular singular point)으로 분류한다
- 정상점 근방에서 계수에 대한 점화식(recurrence relation)을 구하여 멱급수 해를 구성한다
- 정칙 특이점 근방에서 급수 해를 구하기 위해 프로베니우스 방법(Frobenius method)을 적용한다
- 베셀 방정식(Bessel's equation)과 르장드르 방정식(Legendre's equation)이 어떻게 발생하는지 인식하고 급수 해를 특수 함수와 연결한다
- Python을 사용하여 급수 해를 수치적으로 검증하고 알려진 특수 함수와 비교한다

## 선수 과목

이 레슨을 공부하기 전에 다음에 익숙해야 한다:
- 멱급수, 수렴 반지름, 테일러 급수 (레슨 3-4)
- 2계 선형 ODE (레슨 10-12)
- 대안적 접근법으로서의 라플라스 변환법 (레슨 15)

## 동기: 표준 방법이 실패할 때

상수계수 ODE에서는 특성방정식을 사용하여 해를 찾았다. 특정 비상수 계수 방정식에서는 라플라스 변환이나 매개변수 변환법이 작동한다. 그러나 물리학과 공학의 많은 중요한 방정식들은 이 모든 방법에 저항하는 **변수 계수(variable coefficients)**를 가진다:

$$x^2 y'' + xy' + (x^2 - n^2)y = 0 \quad \text{(베셀 방정식)}$$

$$(1 - x^2)y'' - 2xy' + \ell(\ell+1)y = 0 \quad \text{(르장드르 방정식)}$$

$$xy'' + (1-x)y' + ny = 0 \quad \text{(라게르 방정식)}$$

이 방정식들은 곳곳에 나타난다: 원형 막의 진동(드럼), 중력 퍼텐셜, 양자 역학. **멱급수법**은 $y = \sum a_n x^n$으로 가정하고 계수를 결정하여 해를 구한다.

## 정상점과 특이점(Ordinary and Singular Points)

2계 선형 ODE의 표준형을 고려하자:

$$y'' + P(x)y' + Q(x)y = 0$$

**정상점(Ordinary point)**: $P(x)$와 $Q(x)$ 모두 $x_0$에서 해석적(수렴하는 테일러 급수를 가진)이면 $x = x_0$은 정상점이다.

**특이점(Singular point)**: $P(x)$ 또는 $Q(x)$가 $x_0$에서 해석적이 아니면 $x = x_0$은 특이점이다.

$a_2(x)y'' + a_1(x)y' + a_0(x)y = 0$ 형태의 방정식을 $a_2(x)$로 나누어 표준형을 얻는다. 특이점은 $a_2(x_0) = 0$인 곳에서 발생한다.

### 정칙 특이점 vs 비정칙 특이점

특이점 $x_0$이 **정칙(regular)**인 것은:

$$(x - x_0)P(x) \quad \text{와} \quad (x - x_0)^2 Q(x)$$

가 모두 $x_0$에서 해석적일 때이다. 그렇지 않으면 **비정칙(irregular)** 특이점이다.

**이것이 왜 중요한가**: 정칙 특이점에서 프로베니우스 방법은 적어도 하나의 급수 해를 보장한다. 비정칙 특이점에서는 그러한 보장이 없다.

**예제**: $x^2 y'' + xy' + (x^2 - n^2)y = 0$ (베셀 방정식)의 특이점을 분류하라.

표준형: $y'' + \frac{1}{x}y' + \frac{x^2 - n^2}{x^2}y = 0$

따라서 $P(x) = 1/x$이고 $Q(x) = (x^2 - n^2)/x^2$이다. 특이점은 $x = 0$이다.

확인: $xP(x) = 1$ (해석적)이고 $x^2 Q(x) = x^2 - n^2$ (해석적)이다. 따라서 $x = 0$은 **정칙 특이점**이다.

## 정상점 근방의 멱급수 해

### 방법

$x_0$이 정상점이면, 다음 형태의 해를 가정한다:

$$y = \sum_{n=0}^{\infty} a_n (x - x_0)^n$$

수렴 반지름은 적어도 $x_0$에서 복소 평면에서 가장 가까운 특이점까지의 거리이다.

### 단계별 절차

1. $y = \sum_{n=0}^{\infty} a_n x^n$으로 가정 (간단히 $x_0 = 0$으로 취한다)
2. $y' = \sum_{n=1}^{\infty} n a_n x^{n-1}$과 $y'' = \sum_{n=2}^{\infty} n(n-1) a_n x^{n-2}$를 계산
3. ODE에 대입
4. 모든 합이 같은 $x$의 거듭제곱에서 시작하도록 지수를 이동
5. 각 $x^k$의 계수를 모아 0으로 놓기
6. 결과 **점화식(recurrence relation)**을 $a_n$에 대해 풀기

### 풀이 예제: 에어리 방정식(Airy's Equation)

$y'' - xy = 0$을 $x_0 = 0$ 근방에서 풀어라.

$x = 0$은 정상점이다 (이 방정식에는 특이점이 없다).

**1-2단계**: $y = \sum_{n=0}^{\infty} a_n x^n$으로 놓으면, $y'' = \sum_{n=2}^{\infty} n(n-1)a_n x^{n-2}$.

**3단계**: 대입:

$$\sum_{n=2}^{\infty} n(n-1)a_n x^{n-2} - x \sum_{n=0}^{\infty} a_n x^n = 0$$

$$\sum_{n=2}^{\infty} n(n-1)a_n x^{n-2} - \sum_{n=0}^{\infty} a_n x^{n+1} = 0$$

**4단계**: 지수 이동. 첫 번째 합에서 $m = n - 2$ ($n = m + 2$):

$$\sum_{m=0}^{\infty} (m+2)(m+1)a_{m+2} x^{m} - \sum_{n=0}^{\infty} a_n x^{n+1} = 0$$

두 번째 합에서 $m = n + 1$ ($n = m - 1$):

$$\sum_{m=0}^{\infty} (m+2)(m+1)a_{m+2} x^{m} - \sum_{m=1}^{\infty} a_{m-1} x^{m} = 0$$

**5단계**: 첫 번째 합의 $m = 0$ 항은 $2 \cdot 1 \cdot a_2 = 0$을 주므로, $a_2 = 0$.

$m \geq 1$에 대해: $(m+2)(m+1)a_{m+2} - a_{m-1} = 0$.

**6단계**: 점화식:

$$a_{m+2} = \frac{a_{m-1}}{(m+2)(m+1)}, \quad m \geq 1$$

이것은 $a_{m+2}$를 $a_{m-1}$에 연결하며, 3씩 건너뛴다. 따라서 세 개의 독립적인 체인을 얻는다:

- $a_0$에서 시작하는 체인: $a_0, a_3, a_6, a_9, \ldots$
- $a_1$에서 시작하는 체인: $a_1, a_4, a_7, a_{10}, \ldots$
- $a_2 = 0$에서 시작하는 체인: $a_2 = a_5 = a_8 = \cdots = 0$

계산:
- $a_3 = \frac{a_0}{3 \cdot 2}$, $a_6 = \frac{a_3}{6 \cdot 5} = \frac{a_0}{6 \cdot 5 \cdot 3 \cdot 2}$
- $a_4 = \frac{a_1}{4 \cdot 3}$, $a_7 = \frac{a_4}{7 \cdot 6} = \frac{a_1}{7 \cdot 6 \cdot 4 \cdot 3}$

일반해는 $y = a_0 y_1(x) + a_1 y_2(x)$이며 $y_1$과 $y_2$는 두 개의 선형 독립인 에어리 함수이다.

## 프로베니우스 방법(정칙 특이점)(Frobenius Method)

$x_0$이 **정칙 특이**점이면, 멱급수법이 실패할 수 있다 (첫 번째 계수가 0일 수 있다). **프로베니우스 방법**은 다음을 가정하여 접근법을 일반화한다:

$$y = \sum_{n=0}^{\infty} a_n (x - x_0)^{n+r}$$

여기서 $r$은 결정해야 할 미지의 지수이다. 핵심 차이: 지수 $r$이 정수일 필요가 없다.

### 결정 방정식(The Indicial Equation)

프로베니우스 급수를 ODE에 대입하고 $x$의 가장 낮은 거듭제곱을 모으면 $r$에 관한 이차방정식인 **결정 방정식(indicial equation)**을 얻는다:

$$r(r - 1) + p_0 r + q_0 = 0$$

여기서 $p_0 = \lim_{x \to x_0} (x - x_0)P(x)$이고 $q_0 = \lim_{x \to x_0} (x - x_0)^2 Q(x)$이다.

두 근 $r_1 \geq r_2$가 해의 형태를 결정한다:

| 경우 | 조건 | 해 |
|------|-----------|-----------|
| 서로 다르고 차이가 비정수 | $r_1 - r_2 \notin \mathbb{Z}$ | 두 프로베니우스 급수 |
| 같은 근 | $r_1 = r_2$ | 하나의 프로베니우스 급수 + 로그 해 |
| 차이가 정수 | $r_1 - r_2 \in \mathbb{Z}^+$ | $r_1$으로 첫 번째 해; 두 번째는 로그가 필요할 수 있음 |

### 풀이 예제: 0차 베셀 방정식

$$x^2 y'' + xy' + x^2 y = 0$$

$x^2$으로 나누면: $y'' + \frac{1}{x}y' + y = 0$. 여기서 $x = 0$은 정칙 특이점이다.

$p_0 = \lim_{x \to 0} x \cdot (1/x) = 1$, $q_0 = \lim_{x \to 0} x^2 \cdot 1 = 0$.

**결정 방정식**: $r(r-1) + r + 0 = r^2 = 0$, 따라서 $r = 0$ (중근).

이것은 하나의 해가 $r = 0$인 프로베니우스 급수(실제로는 일반 멱급수)이고, 두 번째 해는 로그 항을 포함한다는 것을 의미한다. 첫 번째 해는 유명한 **베셀 함수(Bessel function)** $J_0(x)$이다:

$$J_0(x) = \sum_{m=0}^{\infty} \frac{(-1)^m}{(m!)^2} \left(\frac{x}{2}\right)^{2m} = 1 - \frac{x^2}{4} + \frac{x^4}{64} - \cdots$$

## 특수 함수 소개(Introduction to Special Functions)

수학과 물리학에서 가장 중요한 함수의 상당수가 특정 ODE의 급수 해로 발생한다.

### 베셀 함수(Bessel Functions)

$\nu$차 베셀 방정식:

$$x^2 y'' + xy' + (x^2 - \nu^2)y = 0$$

프로베니우스 방법은 **제1종 베셀 함수** $J_\nu(x)$를 준다. 이 함수들은:
- 원형 막(드럼 모드)의 진동을 기술한다
- 파동 방정식의 원통 좌표 해에 나타난다
- 원형 개구를 통한 회절 패턴을 모델링한다

### 르장드르 다항식(Legendre Polynomials)

르장드르 방정식:

$$(1 - x^2)y'' - 2xy' + \ell(\ell+1)y = 0$$

$\ell$이 음이 아닌 정수이면, 하나의 해가 **다항식** $P_\ell(x)$이다:
- $P_0(x) = 1$, $P_1(x) = x$, $P_2(x) = \frac{1}{2}(3x^2 - 1)$

이 다항식들은:
- 구면 좌표에서 중력 및 전기 퍼텐셜을 기술한다
- 수소 원자 파동함수의 각도 부분을 형성한다 (구면 조화함수와 함께)
- 수치 적분에서 가우스-르장드르 구적법(Gauss-Legendre quadrature)의 기반이다

이러한 특수 함수의 포괄적 다룸은 [Mathematical Methods - 특수 함수](../Mathematical_Methods/11_Special_Functions.md)를 참조하라.

## Python 구현

```python
"""
Power Series Solutions of ODE.

This script demonstrates:
1. Computing series coefficients via recurrence relations (Airy equation)
2. Frobenius method for Bessel's equation
3. Comparing series approximations with scipy.special functions
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import airy, jv  # Airy functions and Bessel J_v
from scipy.integrate import solve_ivp


# ── 1. Airy Equation: y'' - xy = 0 ──────────────────────────
def airy_series_coefficients(N, a0=1.0, a1=0.0):
    """
    Compute the first N coefficients of the power series solution
    to y'' - xy = 0.

    The recurrence is: a_{m+2} = a_{m-1} / ((m+2)(m+1)) for m >= 1
    with a_2 = 0 always.

    Parameters:
        N: number of coefficients to compute
        a0, a1: initial coefficients (free parameters)

    Returns:
        array of coefficients [a_0, a_1, ..., a_{N-1}]
    """
    a = np.zeros(N)
    a[0] = a0
    if N > 1:
        a[1] = a1
    # a[2] = 0 is already set by np.zeros

    # Apply recurrence: a_{m+2} = a_{m-1} / ((m+2)(m+1))
    for m in range(1, N - 2):
        a[m + 2] = a[m - 1] / ((m + 2) * (m + 1))

    return a


def evaluate_power_series(coeffs, x_vals):
    """Evaluate y = sum(a_n * x^n) at given x values."""
    result = np.zeros_like(x_vals)
    for n, a_n in enumerate(coeffs):
        result += a_n * x_vals**n
    return result


# Compare our series with scipy's Airy functions
N_terms = 30  # Number of series terms (more terms = better approximation)
x_range = np.linspace(-10, 5, 500)

# Ai(x) corresponds to a specific linear combination of a0 and a1
# scipy returns (Ai, Ai', Bi, Bi')
Ai, Ai_prime, Bi, Bi_prime = airy(x_range)

# Our series approximation for y_1 (a0=1, a1=0) and y_2 (a0=0, a1=1)
coeffs_y1 = airy_series_coefficients(N_terms, a0=1.0, a1=0.0)
coeffs_y2 = airy_series_coefficients(N_terms, a0=0.0, a1=1.0)

y1_series = evaluate_power_series(coeffs_y1, x_range)
y2_series = evaluate_power_series(coeffs_y2, x_range)

# Ai(x) is a specific combination: Ai(x) = c1*y1 + c2*y2
# At x=0: Ai(0) = 1/(3^{2/3} Gamma(2/3)) and Ai'(0) = -1/(3^{1/3} Gamma(1/3))
from scipy.special import gamma
c1 = 1.0 / (3**(2/3) * gamma(2/3))  # coefficient for y1
c2 = -1.0 / (3**(1/3) * gamma(1/3))  # coefficient for y2
Ai_series = c1 * y1_series + c2 * y2_series

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Airy function comparison
axes[0].plot(x_range, Ai, 'b-', linewidth=2, label='Ai(x) [scipy]')
axes[0].plot(x_range, Ai_series, 'r--', linewidth=1.5,
             label=f'Series ({N_terms} terms)')
axes[0].set_xlim(-10, 5)
axes[0].set_ylim(-0.6, 0.6)
axes[0].set_xlabel('x')
axes[0].set_ylabel('Ai(x)')
axes[0].set_title("Airy Function: Series vs scipy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# ── 2. Bessel Function J_0(x) ────────────────────────────────
def bessel_J0_series(x, N_terms=20):
    """
    Compute J_0(x) from its power series:
    J_0(x) = sum_{m=0}^{N} (-1)^m / (m!)^2 * (x/2)^{2m}

    This series comes from the Frobenius method applied to
    Bessel's equation with nu=0 and indicial root r=0.
    """
    result = np.zeros_like(x, dtype=float)
    for m in range(N_terms):
        # Each term: (-1)^m * (x/2)^{2m} / (m!)^2
        term = ((-1)**m / (np.math.factorial(m))**2) * (x / 2)**(2*m)
        result += term
    return result


x_bessel = np.linspace(0, 20, 500)
J0_exact = jv(0, x_bessel)  # scipy's exact Bessel function

# Compare different numbers of series terms
for n_terms in [5, 10, 20]:
    J0_approx = bessel_J0_series(x_bessel, N_terms=n_terms)
    axes[1].plot(x_bessel, J0_approx, '--',
                 label=f'Series ({n_terms} terms)', alpha=0.7)

axes[1].plot(x_bessel, J0_exact, 'k-', linewidth=2, label='J_0(x) [scipy]')
axes[1].set_xlabel('x')
axes[1].set_ylabel('J_0(x)')
axes[1].set_title("Bessel Function J_0: Series Convergence")
axes[1].set_ylim(-0.5, 1.1)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('power_series_solutions.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved to power_series_solutions.png")

# ── 3. Legendre Polynomials ──────────────────────────────────
print("\n=== Legendre Polynomials (from series solution) ===")
from numpy.polynomial.legendre import Legendre

x_leg = np.linspace(-1, 1, 200)
fig2, ax2 = plt.subplots(figsize=(8, 5))

for ell in range(5):
    # Legendre class uses coefficients in the Legendre basis
    # We use a simpler construction: scipy provides legendre via special
    from scipy.special import legendre
    P_ell = legendre(ell)
    ax2.plot(x_leg, P_ell(x_leg), linewidth=2, label=f'$P_{ell}(x)$')

ax2.set_xlabel('x')
ax2.set_ylabel('P_l(x)')
ax2.set_title('Legendre Polynomials P_0 through P_4')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='gray', linewidth=0.5)

plt.tight_layout()
plt.savefig('legendre_polynomials.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved to legendre_polynomials.png")
```

## 요약

| 개념 | 핵심 아이디어 |
|---------|----------|
| 정상점 | $P(x)$, $Q(x)$가 해석적; 표준 멱급수가 작동 |
| 정칙 특이점 | $(x-x_0)P$, $(x-x_0)^2 Q$가 해석적; 프로베니우스 방법이 작동 |
| 결정 방정식 | 프로베니우스 급수에서 지수 $r$을 결정 |
| 점화식 | $a_{n+k}$를 이전 계수에 연결하는 대수적 공식 |
| 특수 함수 | 베셀, 르장드르, 라게르, 에르미트 -- 모두 급수 해에서 발생 |

멱급수법은 단지 추상적인 기법이 아니다. 이것은 사인과 코사인만큼 응용 수학에 근본적인 많은 함수들의 역사적 기원이다. `scipy.special.jv(0, x)`를 호출할 때, ODE에 급수를 넣고 기계적으로 돌려서 처음 발견된 함수를 평가하고 있는 것이다.

## 연습 문제

1. **정상점 분류**: 방정식 $(1 + x^2)y'' + 2xy' + 4y = 0$에 대해, 모든 특이점을 식별하고 정칙 또는 비정칙으로 분류하라. $x_0 = 0$을 중심으로 한 급수 해의 보장된 수렴 반지름은 얼마인가?

2. **급수 해**: $y'' + xy' + y = 0$, $y(0) = 1$, $y'(0) = 0$의 멱급수 해에서 처음 6개의 0이 아닌 항을 구하라. 점화식을 명시적으로 써라.

3. **프로베니우스 방법**: 오일러 방정식 $x^2 y'' + 3xy' + y = 0$에 프로베니우스 방법을 적용하라. 결정 방정식과 두 해를 구하라. (힌트: 이 방정식은 $x^r$ 형태의 정확한 해도 가진다. 프로베니우스 급수가 정확한 답으로 환원됨을 검증하라.)

4. **베셀 함수 계산**: 급수 표현을 사용하여 $J_1(x)$를 계산하는 Python 함수를 작성하라. 구간 $[0, 15]$에서 `scipy.special.jv(1, x)`와 비교하라. $x = 10$에서 8자리 정확도를 위해 몇 항이 필요한가?

5. **르장드르 방정식**: $\ell = 3$인 르장드르 방정식에서 출발하여 급수 해를 유도하고 종결됨(다항식이 됨)을 보여라. 결과 다항식이 $P_3(1) = 1$이 되도록 정규화하면 $P_3(x) = \frac{1}{2}(5x^3 - 3x)$의 스칼라 배임을 검증하라.

---

*이전: [ODE를 위한 라플라스 변환](./15_Laplace_Transform_for_ODE.md) | 다음: [편미분방정식 입문](./17_Introduction_to_PDE.md)*
