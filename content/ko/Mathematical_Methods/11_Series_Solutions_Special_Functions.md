# 11. 급수해와 특수함수 (Series Solutions and Special Functions)

## 학습 목표

- **정상점(ordinary point)**과 **특이점(singular point)**을 구별하고, 정상점 주위에서 **멱급수 해법**을 적용할 수 있다
- **프로베니우스 방법(Frobenius method)**으로 **정칙 특이점** 주위의 급수해를 구하고, **지표 방정식(indicial equation)**의 세 가지 경우를 처리할 수 있다
- **베셀 함수(Bessel functions)** $J_n(x)$, $Y_n(x)$의 성질, 직교성, 점화식을 이해하고 원통 좌표계 문제에 적용할 수 있다
- **르장드르 다항식(Legendre polynomials)** $P_l(x)$의 로드리게스 공식, 생성 함수, 직교성을 이해하고 구면 좌표계 문제에 적용할 수 있다
- **에르미트 함수**와 **라게르 함수**가 양자역학의 조화 진동자 및 수소 원자 문제에서 어떻게 등장하는지 이해한다
- **감마 함수**와 **베타 함수**의 정의, 성질, 상호 관계를 알고 적분 계산에 활용할 수 있다

---

## 1. 급수해법 (Power Series Method)

### 1.1 정상점과 특이점

2차 선형 ODE의 표준형을 고려하자:

$$y'' + P(x)y' + Q(x)y = 0$$

점 $x = x_0$에서의 분류:

| 분류 | 조건 | 예시 |
|---|---|---|
| **정상점 (ordinary point)** | $P(x)$, $Q(x)$가 $x_0$에서 해석적 | $y'' + y = 0$의 $x_0 = 0$ |
| **정칙 특이점 (regular singular point)** | $(x-x_0)P(x)$, $(x-x_0)^2 Q(x)$가 해석적 | 베셀 방정식의 $x_0 = 0$ |
| **비정칙 특이점 (irregular singular point)** | 위 어느 것도 아님 | $x^3 y'' + y = 0$의 $x_0 = 0$ |

정상점에서는 **멱급수 해법**, 정칙 특이점에서는 **프로베니우스 방법**을 사용한다.

```python
import sympy as sp

# --- 특이점 분류 ---
x = sp.Symbol('x')

# 예제: 베셀 방정식 x^2 y'' + x y' + (x^2 - n^2) y = 0
# 표준형으로 변환: y'' + (1/x) y' + (1 - n^2/x^2) y = 0
n = sp.Symbol('n', positive=True)
P_bessel = 1 / x
Q_bessel = 1 - n**2 / x**2

# x = 0에서의 분류
xP = sp.simplify(x * P_bessel)   # x * (1/x) = 1 (해석적)
x2Q = sp.simplify(x**2 * Q_bessel)  # x^2 * (1 - n^2/x^2) = x^2 - n^2 (해석적)

print(f"베셀 방정식 (x=0):")
print(f"  P(x) = {P_bessel}  →  x=0에서 특이")
print(f"  x·P(x) = {xP}  →  해석적")
print(f"  x²·Q(x) = {x2Q}  →  해석적")
print(f"  결론: x=0은 정칙 특이점")

# 예제: 에어리 방정식 y'' - x y = 0
P_airy = 0
Q_airy = -x

print(f"\n에어리 방정식 (x=0):")
print(f"  P(x) = {P_airy}, Q(x) = {Q_airy}")
print(f"  둘 다 x=0에서 해석적 → x=0은 정상점")
```

### 1.2 정상점 주위의 급수해

정상점 $x_0 = 0$ 주위에서 해를 멱급수로 가정한다:

$$y = \sum_{n=0}^{\infty} a_n x^n$$

이를 ODE에 대입하여 **점화 관계(recurrence relation)**를 유도하고, 계수 $a_n$을 순차적으로 결정한다.

**예제: 에어리 방정식** $y'' - xy = 0$

$$\sum_{n=2}^{\infty} n(n-1) a_n x^{n-2} - \sum_{n=0}^{\infty} a_n x^{n+1} = 0$$

인덱스를 맞추면 ($m = n-2$ 치환):

$$\sum_{m=0}^{\infty} (m+2)(m+1) a_{m+2} x^m - \sum_{m=1}^{\infty} a_{m-1} x^m = 0$$

$x^0$ 항: $2 \cdot 1 \cdot a_2 = 0 \;\Rightarrow\; a_2 = 0$

$x^m$ ($m \geq 1$) 항: $(m+2)(m+1) a_{m+2} = a_{m-1}$

$$\boxed{a_{m+2} = \frac{a_{m-1}}{(m+2)(m+1)}, \quad m \geq 1}$$

$a_0$과 $a_1$은 자유 매개변수(초기조건)이므로 두 개의 일차독립인 해를 얻는다.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import airy

# --- 에어리 방정식의 급수해 ---
def airy_series(x_val, a0, a1, N_terms=30):
    """에어리 방정식 y'' - xy = 0의 멱급수 해"""
    a = np.zeros(N_terms)
    a[0] = a0
    a[1] = a1
    a[2] = 0  # a_2 = 0 항상

    # 점화 관계: a[m+2] = a[m-1] / ((m+2)*(m+1)), m >= 1
    for m in range(1, N_terms - 2):
        a[m + 2] = a[m - 1] / ((m + 2) * (m + 1))

    result = sum(a[k] * x_val**k for k in range(N_terms))
    return result

# SciPy의 에어리 함수와 비교
x = np.linspace(-15, 5, 500)
Ai, Aip, Bi, Bip = airy(x)

# 급수해 (수렴 범위 내에서)
x_series = np.linspace(-8, 5, 300)
y_series_Ai = np.array([airy_series(xi, 1, 0, N_terms=60) for xi in x_series])
y_series_Bi = np.array([airy_series(xi, 0, 1, N_terms=60) for xi in x_series])

# 정규화 상수 (SciPy 에어리 함수와 맞추기)
c_Ai = airy(0)[0]  # Ai(0)
c_Bi = airy(0)[2]  # Bi(0)
c_Ai_prime = airy(0)[1]  # Ai'(0)
c_Bi_prime = airy(0)[3]  # Bi'(0)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Ai(x): 제1종 에어리 함수
axes[0].plot(x, Ai, 'b-', linewidth=2, label='Ai(x) (SciPy)')
y_approx = c_Ai * np.array([airy_series(xi, 1, 0, 60) for xi in x_series]) \
         + c_Ai_prime * np.array([airy_series(xi, 0, 1, 60) for xi in x_series])
axes[0].plot(x_series, y_approx, 'r--', linewidth=1.5, label='급수해 (60항)')
axes[0].set_xlim(-15, 5)
axes[0].set_ylim(-0.6, 0.8)
axes[0].set_xlabel('x')
axes[0].set_ylabel('Ai(x)')
axes[0].set_title('에어리 함수 Ai(x)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Bi(x): 제2종 에어리 함수
axes[1].plot(x, Bi, 'b-', linewidth=2, label='Bi(x) (SciPy)')
axes[1].set_xlim(-15, 5)
axes[1].set_ylim(-0.6, 1.5)
axes[1].set_xlabel('x')
axes[1].set_ylabel('Bi(x)')
axes[1].set_title('에어리 함수 Bi(x)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('에어리 방정식 $y\'\' - xy = 0$의 해', fontsize=14)
plt.tight_layout()
plt.savefig('airy_functions.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 2. 프로베니우스 방법 (Frobenius Method)

### 2.1 정칙 특이점

정칙 특이점 $x_0 = 0$ 주위에서는 일반적인 멱급수가 수렴하지 않을 수 있다. 대신 **프로베니우스 급수**를 가정한다:

$$y = x^s \sum_{n=0}^{\infty} a_n x^n = \sum_{n=0}^{\infty} a_n x^{n+s}, \quad a_0 \neq 0$$

여기서 $s$는 미지의 **지표(index)**이며, ODE에 대입하여 결정한다.

### 2.2 지표 방정식 (Indicial Equation)

ODE를 $x^2 y'' + x b(x) y' + c(x) y = 0$ 형태로 쓰고 ($b(x) = xP(x)$, $c(x) = x^2 Q(x)$), 프로베니우스 급수를 대입하면 가장 낮은 차수($x^s$)의 계수에서 **지표 방정식**이 나온다:

$$\boxed{s(s-1) + b_0 s + c_0 = 0}$$

여기서 $b_0 = b(0)$, $c_0 = c(0)$이다. 이 이차방정식의 두 근 $s_1, s_2$ ($s_1 \geq s_2$)가 급수해의 형태를 결정한다.

### 2.3 세 가지 경우

지표 방정식의 두 근 $s_1, s_2$의 차이 $s_1 - s_2$에 따라 세 가지 경우가 생긴다:

| 경우 | 조건 | 해의 형태 |
|---|---|---|
| **경우 1** | $s_1 - s_2 \notin \mathbb{Z}$ (비정수) | $y_1 = x^{s_1}\sum a_n x^n$, $y_2 = x^{s_2}\sum b_n x^n$ |
| **경우 2** | $s_1 = s_2$ (중근) | $y_1 = x^s \sum a_n x^n$, $y_2 = y_1 \ln x + x^s \sum b_n x^n$ |
| **경우 3** | $s_1 - s_2 \in \mathbb{Z}^+$ (양의 정수) | $y_2$에 $\ln x$ 항이 포함될 수도 있고 아닐 수도 있음 |

**예제: 베셀 방정식** $x^2 y'' + x y' + (x^2 - \nu^2)y = 0$

$b(x) = 1$, $c(x) = x^2 - \nu^2$이므로 $b_0 = 1$, $c_0 = -\nu^2$.

지표 방정식: $s(s-1) + s - \nu^2 = s^2 - \nu^2 = 0$

$$s_1 = \nu, \quad s_2 = -\nu$$

```python
import sympy as sp

# --- 프로베니우스 방법: 베셀 방정식의 지표 방정식 ---
s, nu = sp.symbols('s nu', real=True)

# 지표 방정식: s(s-1) + b0*s + c0 = 0
b0 = 1        # b(0) = 1
c0 = -nu**2   # c(0) = -nu^2

indicial_eq = s*(s - 1) + b0*s + c0
print(f"지표 방정식: {sp.expand(indicial_eq)} = 0")
roots = sp.solve(indicial_eq, s)
print(f"지표근: s1 = {roots[1]}, s2 = {roots[0]}")

# nu = 0: 중근 (경우 2) → 제2종 베셀 함수에 ln 항 포함
# nu = 1/2: 비정수 차이 (경우 1) → 두 독립 급수해
# nu = 1: 정수 차이 (경우 3) → 주의 필요
for nu_val in [0, sp.Rational(1, 2), 1, 2]:
    diff = 2 * nu_val  # s1 - s2 = 2*nu
    case = "경우 2 (중근)" if diff == 0 else \
           ("경우 1 (비정수)" if not diff.is_integer else "경우 3 (정수 차이)")
    print(f"  nu = {nu_val}: s1-s2 = {diff} → {case}")
```

---

## 3. 베셀 함수 (Bessel Functions)

### 3.1 베셀 방정식

**베셀 방정식**은 원통 좌표계에서 라플라스 방정식이나 헬름홀츠 방정식을 변수분리하면 자연스럽게 등장한다:

$$x^2 y'' + x y' + (x^2 - \nu^2) y = 0$$

여기서 $\nu \geq 0$은 **차수(order)**이다. 이 방정식은 물리학에서 가장 자주 만나는 2차 ODE 중 하나이다.

### 3.2 제1종, 제2종 베셀 함수

**제1종 베셀 함수** $J_\nu(x)$는 프로베니우스 방법으로 얻는 $s = \nu$인 해이다:

$$J_\nu(x) = \sum_{k=0}^{\infty} \frac{(-1)^k}{k! \, \Gamma(k + \nu + 1)} \left(\frac{x}{2}\right)^{2k + \nu}$$

**제2종 베셀 함수** $Y_\nu(x)$ (노이만 함수)는 $J_\nu$와 일차독립인 제2의 해이다:

$$Y_\nu(x) = \frac{J_\nu(x) \cos(\nu\pi) - J_{-\nu}(x)}{\sin(\nu\pi)}$$

$\nu$가 정수인 경우 이 공식은 $0/0$ 형태가 되므로 극한으로 정의한다.

**주요 성질**:
- $J_\nu(x)$: $x = 0$에서 유한 ($\nu \geq 0$일 때)
- $Y_\nu(x)$: $x \to 0^+$에서 $-\infty$로 발산
- 점근 거동 ($x \to \infty$): $J_\nu(x) \sim \sqrt{\frac{2}{\pi x}} \cos\left(x - \frac{\nu\pi}{2} - \frac{\pi}{4}\right)$

**점화식 (recurrence relations)**:

$$J_{\nu-1}(x) + J_{\nu+1}(x) = \frac{2\nu}{x} J_\nu(x)$$

$$J_{\nu-1}(x) - J_{\nu+1}(x) = 2 J_\nu'(x)$$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, yv, jn_zeros

# --- 베셀 함수 시각화 ---
x = np.linspace(0.01, 20, 500)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 제1종 베셀 함수
for nu in [0, 1, 2, 3]:
    axes[0].plot(x, jv(nu, x), linewidth=1.5, label=f'$J_{{{nu}}}(x)$')
axes[0].axhline(y=0, color='k', linewidth=0.5)
axes[0].set_xlim(0, 20)
axes[0].set_ylim(-0.5, 1.1)
axes[0].set_xlabel('x')
axes[0].set_ylabel('$J_\\nu(x)$')
axes[0].set_title('제1종 베셀 함수 $J_\\nu(x)$')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 제2종 베셀 함수
for nu in [0, 1, 2]:
    axes[1].plot(x, yv(nu, x), linewidth=1.5, label=f'$Y_{{{nu}}}(x)$')
axes[1].axhline(y=0, color='k', linewidth=0.5)
axes[1].set_xlim(0, 20)
axes[1].set_ylim(-1.5, 0.8)
axes[1].set_xlabel('x')
axes[1].set_ylabel('$Y_\\nu(x)$')
axes[1].set_title('제2종 베셀 함수 $Y_\\nu(x)$')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bessel_functions.png', dpi=150, bbox_inches='tight')
plt.show()

# --- 점화식 검증 ---
nu_test = 2
x_test = np.linspace(1, 10, 100)

# J_{nu-1} + J_{nu+1} = (2*nu/x) * J_nu
lhs = jv(nu_test - 1, x_test) + jv(nu_test + 1, x_test)
rhs = (2 * nu_test / x_test) * jv(nu_test, x_test)
print(f"점화식 검증 (nu={nu_test}):")
print(f"  max|LHS - RHS| = {np.max(np.abs(lhs - rhs)):.2e}")
```

### 3.3 직교성과 급수 전개

$J_\nu$의 영점(zeros)을 $\alpha_{\nu,1} < \alpha_{\nu,2} < \cdots$라 하면, 구간 $[0, a]$에서 다음 **직교성**이 성립한다:

$$\int_0^a x \, J_\nu\!\left(\frac{\alpha_{\nu,m}}{a} x\right) J_\nu\!\left(\frac{\alpha_{\nu,n}}{a} x\right) dx = \begin{cases} 0 & m \neq n \\ \frac{a^2}{2} [J_{\nu+1}(\alpha_{\nu,n})]^2 & m = n \end{cases}$$

이를 이용하면 구간 $[0, a]$에서 정의된 함수 $f(r)$을 **푸리에-베셀 급수(Fourier-Bessel series)**로 전개할 수 있다:

$$f(r) = \sum_{n=1}^{\infty} c_n J_\nu\!\left(\frac{\alpha_{\nu,n}}{a} r\right)$$

$$c_n = \frac{2}{a^2 [J_{\nu+1}(\alpha_{\nu,n})]^2} \int_0^a r \, f(r) \, J_\nu\!\left(\frac{\alpha_{\nu,n}}{a} r\right) dr$$

```python
import numpy as np
from scipy.special import jv, jn_zeros
from scipy.integrate import quad
import matplotlib.pyplot as plt

# --- 푸리에-베셀 급수 예제: f(r) = 1을 J_0 급수로 전개 ---
a = 1.0  # 구간 [0, a]
nu = 0
N_terms = 20

# J_0의 영점
zeros = jn_zeros(nu, N_terms)

def compute_bessel_coeff(n_idx, func, a, nu):
    """푸리에-베셀 계수 c_n 계산"""
    alpha_n = zeros[n_idx]
    norm = (a**2 / 2) * jv(nu + 1, alpha_n)**2

    integrand = lambda r: r * func(r) * jv(nu, alpha_n * r / a)
    integral, _ = quad(integrand, 0, a)

    return integral / norm

# f(r) = 1
f = lambda r: 1.0
coeffs = [compute_bessel_coeff(n, f, a, nu) for n in range(N_terms)]

# 급수 재구성
r = np.linspace(0, a, 200)
f_approx = {5: None, 10: None, 20: None}

for N in f_approx:
    f_approx[N] = sum(coeffs[n] * jv(nu, zeros[n] * r / a) for n in range(N))

plt.figure(figsize=(10, 5))
plt.axhline(y=1.0, color='k', linewidth=2, label='$f(r) = 1$')
for N, color in zip([5, 10, 20], ['red', 'blue', 'green']):
    plt.plot(r, f_approx[N], color=color, linewidth=1.5,
             label=f'베셀 급수 ({N}항)')
plt.xlabel('r')
plt.ylabel('f(r)')
plt.title('푸리에-베셀 급수 전개: $f(r) = 1$ on $[0, 1]$')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fourier_bessel_series.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 3.4 물리학 응용 (원형 막 진동, 원통 문제)

**원형 막의 진동**: 반지름 $a$, 경계가 고정된 원형 막의 진동은 2차원 파동 방정식을 극좌표로 풀면 구해진다:

$$u(r, \theta, t) = \sum_{m=0}^{\infty} \sum_{n=1}^{\infty} J_m\!\left(\frac{\alpha_{m,n}}{a} r\right) \left(A_{mn} \cos m\theta + B_{mn} \sin m\theta\right) \cos(\omega_{mn} t)$$

여기서 $\omega_{mn} = c \, \alpha_{m,n}/a$이고, $c$는 파동 속도이다. 경계 조건 $u(a, \theta, t) = 0$이 $J_m(\alpha_{m,n}) = 0$을 요구한다.

```python
import numpy as np
from scipy.special import jv, jn_zeros
import matplotlib.pyplot as plt

# --- 원형 막의 정상 진동 모드 시각화 ---
fig, axes = plt.subplots(2, 3, figsize=(15, 10),
                         subplot_kw={'projection': 'polar'})

modes = [(0, 1), (0, 2), (0, 3),
         (1, 1), (1, 2), (2, 1)]

for idx, (m, n) in enumerate(modes):
    row, col = idx // 3, idx % 3
    ax = axes[row, col]

    alpha_mn = jn_zeros(m, n)[-1]  # n번째 영점

    r = np.linspace(0, 1, 100)
    theta = np.linspace(0, 2 * np.pi, 200)
    R, Theta = np.meshgrid(r, theta)

    Z = jv(m, alpha_mn * R) * np.cos(m * Theta)

    c = ax.pcolormesh(Theta, R, Z, cmap='RdBu_r', shading='auto')
    ax.set_title(f'모드 ($m$={m}, $n$={n})\n'
                 f'$\\alpha_{{{m},{n}}}$ = {alpha_mn:.3f}',
                 fontsize=11, pad=12)
    ax.set_rticks([])

plt.suptitle('원형 막 진동 모드 — $J_m(\\alpha_{m,n} r/a) \\cos(m\\theta)$',
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('circular_membrane_modes.png', dpi=150, bbox_inches='tight')
plt.show()

# --- 정상 모드의 진동수 비율 ---
print("원형 막 정상 모드 진동수 비율 (f_mn / f_01):")
alpha_01 = jn_zeros(0, 1)[0]
for m in range(3):
    for n in range(1, 4):
        alpha_mn = jn_zeros(m, n)[-1]
        ratio = alpha_mn / alpha_01
        print(f"  (m={m}, n={n}): alpha = {alpha_mn:.4f}, "
              f"f/f_01 = {ratio:.4f}")
```

**물리적 의미**: 원형 막의 진동 주파수 비는 비조화적(inharmonic)이다 — 이것이 드럼이 피아노처럼 명확한 음정을 내지 못하는 이유이다.

### 3.5 변형 베셀 함수 (Modified Bessel Functions)

**변형 베셀 방정식**은 원통 좌표 문제에서 $z$-방향으로 지수적 거동이 나타날 때 등장한다:

$$x^2 y'' + x y' - (x^2 + \nu^2) y = 0$$

이것은 베셀 방정식에서 $x \to ix$로 치환한 것이며, 해는 **변형 베셀 함수**이다:

- **제1종 변형 베셀 함수** $I_\nu(x) = i^{-\nu} J_\nu(ix)$: $x \to \infty$에서 **지수적 증가** ($I_\nu(x) \sim \frac{e^x}{\sqrt{2\pi x}}$)
- **제2종 변형 베셀 함수** $K_\nu(x)$: $x \to \infty$에서 **지수적 감소** ($K_\nu(x) \sim \sqrt{\frac{\pi}{2x}} e^{-x}$)

급수 표현:

$$I_\nu(x) = \sum_{k=0}^{\infty} \frac{1}{k! \, \Gamma(k+\nu+1)} \left(\frac{x}{2}\right)^{2k+\nu}$$

(베셀 $J_\nu$와 비교: $(-1)^k$가 없으므로 모든 항이 양수 → 진동 없이 단조 증가)

**물리 응용**:
- 원통 도파관의 감쇠 모드
- 원형 단면 열전도 (정상 상태)
- 유카와 퍼텐셜 $\propto K_0(mr)/r$
- 변형 베셀 방정식으로 귀결되는 확산 문제

```python
import numpy as np
from scipy.special import iv, kv, jv
import matplotlib.pyplot as plt

x = np.linspace(0.01, 5, 300)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# I_nu(x): 제1종 변형 베셀
for nu in [0, 1, 2]:
    axes[0].plot(x, iv(nu, x), linewidth=2, label=f'$I_{{{nu}}}(x)$')
axes[0].set_ylim(0, 10)
axes[0].set_xlabel('x'); axes[0].set_ylabel('$I_\\nu(x)$')
axes[0].set_title('제1종 변형 베셀 함수 $I_\\nu(x)$')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# K_nu(x): 제2종 변형 베셀
for nu in [0, 1, 2]:
    axes[1].plot(x, kv(nu, x), linewidth=2, label=f'$K_{{{nu}}}(x)$')
axes[1].set_ylim(0, 5)
axes[1].set_xlabel('x'); axes[1].set_ylabel('$K_\\nu(x)$')
axes[1].set_title('제2종 변형 베셀 함수 $K_\\nu(x)$')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('modified_bessel.png', dpi=150, bbox_inches='tight')
plt.show()

# J_nu vs I_nu 비교: 진동 vs 단조증가
fig2, ax = plt.subplots(figsize=(10, 5))
ax.plot(x, jv(0, x), 'b-', linewidth=2, label='$J_0(x)$ (진동)')
ax.plot(x, iv(0, x), 'r-', linewidth=2, label='$I_0(x)$ (단조증가)')
ax.plot(x, kv(0, x), 'g-', linewidth=2, label='$K_0(x)$ (단조감소)')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_ylim(-0.5, 5)
ax.set_title('베셀 $J_0$ vs 변형 베셀 $I_0$, $K_0$')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
```

---

## 4. 르장드르 다항식 (Legendre Polynomials)

### 4.1 르장드르 방정식

**르장드르 방정식**은 구면 좌표계에서 라플라스 방정식을 변수분리하면 등장한다:

$$(1 - x^2) y'' - 2x y' + l(l+1) y = 0$$

여기서 $x = \cos\theta$이고, $l$은 비음의 정수이다. $x = \pm 1$은 정칙 특이점이다.

$l$이 비음의 정수일 때, 하나의 급수해가 **다항식**으로 끊어지며, 이것이 **르장드르 다항식** $P_l(x)$이다.

### 4.2 로드리게스 공식과 생성 함수

**로드리게스 공식 (Rodrigues' formula)**:

$$P_l(x) = \frac{1}{2^l l!} \frac{d^l}{dx^l} (x^2 - 1)^l$$

처음 몇 개의 르장드르 다항식:

| $l$ | $P_l(x)$ |
|---|---|
| 0 | $1$ |
| 1 | $x$ |
| 2 | $\frac{1}{2}(3x^2 - 1)$ |
| 3 | $\frac{1}{2}(5x^3 - 3x)$ |
| 4 | $\frac{1}{8}(35x^4 - 30x^2 + 3)$ |

**생성 함수 (generating function)**:

$$\frac{1}{\sqrt{1 - 2xt + t^2}} = \sum_{l=0}^{\infty} P_l(x) t^l, \quad |t| < 1$$

이 생성 함수의 물리적 의미: $\frac{1}{|\mathbf{r} - \mathbf{r}'|}$를 전개할 때 나타나며, **다중극 전개(multipole expansion)**의 핵심이다.

```python
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.special import legendre

# --- 르장드르 다항식: SymPy 로드리게스 공식 ---
x = sp.Symbol('x')

def rodrigues(l):
    """로드리게스 공식으로 P_l(x) 계산"""
    return sp.simplify(
        sp.diff((x**2 - 1)**l, x, l) / (2**l * sp.factorial(l))
    )

print("로드리게스 공식으로 구한 르장드르 다항식:")
for l in range(6):
    Pl = rodrigues(l)
    print(f"  P_{l}(x) = {Pl}")

# --- 시각화 ---
x_vals = np.linspace(-1, 1, 500)

plt.figure(figsize=(10, 6))
for l in range(6):
    Pl = legendre(l)
    plt.plot(x_vals, Pl(x_vals), linewidth=1.5, label=f'$P_{{{l}}}(x)$')

plt.xlabel('$x$')
plt.ylabel('$P_l(x)$')
plt.title('르장드르 다항식 $P_l(x)$, $l = 0, 1, \\ldots, 5$')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.tight_layout()
plt.savefig('legendre_polynomials.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 4.3 직교성과 급수 전개

르장드르 다항식은 구간 $[-1, 1]$에서 **직교**한다:

$$\int_{-1}^{1} P_m(x) P_n(x) \, dx = \frac{2}{2n+1} \delta_{mn}$$

따라서 구간 $[-1, 1]$에서 정의된 함수 $f(x)$를 **르장드르 급수**로 전개할 수 있다:

$$f(x) = \sum_{l=0}^{\infty} c_l P_l(x), \quad c_l = \frac{2l+1}{2} \int_{-1}^{1} f(x) P_l(x) \, dx$$

```python
import numpy as np
from scipy.special import legendre
from scipy.integrate import quad
import matplotlib.pyplot as plt

# --- 르장드르 급수 전개: f(x) = |x| ---
def legendre_coeff(l, func):
    """르장드르 급수 계수 c_l 계산"""
    Pl = legendre(l)
    integrand = lambda x: func(x) * Pl(x)
    integral, _ = quad(integrand, -1, 1)
    return (2*l + 1) / 2 * integral

f = lambda x: np.abs(x)
N_max = 20
coeffs = [legendre_coeff(l, f) for l in range(N_max)]

# |x|는 우함수이므로 홀수 l 계수는 0
print("르장드르 급수 계수 (f(x) = |x|):")
for l in range(8):
    print(f"  c_{l} = {coeffs[l]:.6f}", "(= 0 이론적)" if l % 2 == 1 else "")

# 급수 재구성
x = np.linspace(-1, 1, 500)

plt.figure(figsize=(10, 6))
plt.plot(x, np.abs(x), 'k-', linewidth=2.5, label='$f(x) = |x|$')

for N, color in [(3, 'red'), (7, 'blue'), (15, 'green')]:
    f_approx = sum(coeffs[l] * legendre(l)(x) for l in range(N))
    plt.plot(x, f_approx, color=color, linewidth=1.5,
             label=f'르장드르 급수 ({N}항)')

plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.title('르장드르 급수 전개: $f(x) = |x|$')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('legendre_series.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 4.4 물리학 응용 (다중극 전개, 구 대칭 문제)

**다중극 전개 (Multipole Expansion)**: 전하 분포에 의한 원거리 전위를 르장드르 다항식으로 전개한다.

점전하 $q$가 원점에서 거리 $d$만큼 떨어진 곳에 있을 때, 관측점 $r > d$에서의 전위:

$$\Phi(r, \theta) = \frac{q}{4\pi\epsilon_0} \frac{1}{|\mathbf{r} - \mathbf{d}|} = \frac{q}{4\pi\epsilon_0} \sum_{l=0}^{\infty} \frac{d^l}{r^{l+1}} P_l(\cos\theta)$$

- $l = 0$: **단극자(monopole)** — 총 전하
- $l = 1$: **쌍극자(dipole)** — $\propto 1/r^2$
- $l = 2$: **사중극자(quadrupole)** — $\propto 1/r^3$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre

# --- 다중극 전개 시각화 ---
# 점전하 q가 z축 위 d에 위치
q = 1.0
d = 0.5  # 전하 위치 (원점에서 z방향)

# 극좌표 그리드 (r > d 영역)
r = np.linspace(0.8, 3.0, 100)
theta = np.linspace(0, np.pi, 100)
R, Theta = np.meshgrid(r, theta)

# 정확한 전위
X = R * np.sin(Theta)
Z = R * np.cos(Theta)
dist = np.sqrt(X**2 + (Z - d)**2)
Phi_exact = q / dist

# 다중극 근사
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
titles = ['단극자 (l=0)', '단극자+쌍극자 (l=0,1)', '다중극 (l=0,...,4)']
L_max_list = [0, 1, 4]

for idx, L_max in enumerate(L_max_list):
    Phi_approx = np.zeros_like(R)
    for l in range(L_max + 1):
        Pl = legendre(l)
        Phi_approx += q * (d**l / R**(l + 1)) * Pl(np.cos(Theta))

    # r-theta 평면에서 등고선 (x-z 단면)
    ax = axes[idx]
    levels = np.linspace(0.2, 2.0, 15)
    c1 = ax.contour(X, Z, Phi_exact, levels=levels,
                    colors='gray', linewidths=0.5, linestyles='--')
    c2 = ax.contour(X, Z, Phi_approx, levels=levels,
                    colors='blue', linewidths=1.0)
    ax.plot(0, d, 'ro', markersize=8, label='점전하')
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_title(titles[idx])
    ax.set_aspect('equal')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

plt.suptitle('다중극 전개: 정확해(회색 점선) vs 근사(파란색 실선)', fontsize=13)
plt.tight_layout()
plt.savefig('multipole_expansion.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 4.5 결합 르장드르 함수와 구면조화함수

#### 결합 르장드르 함수 (Associated Legendre Functions)

$\theta$-의존성뿐 아니라 $\phi$-의존성까지 포함하는 3차원 구면 좌표 문제에서는 **결합 르장드르 방정식**이 등장한다:

$$(1 - x^2) y'' - 2x y' + \left[l(l+1) - \frac{m^2}{1-x^2}\right] y = 0$$

여기서 $x = \cos\theta$, $l = 0, 1, 2, \ldots$, $m = -l, \ldots, l$이다.

해는 **결합 르장드르 함수** $P_l^m(x)$이며, 로드리게스 공식의 확장으로 주어진다:

$$P_l^m(x) = (-1)^m (1-x^2)^{m/2} \frac{d^m}{dx^m} P_l(x), \quad m \geq 0$$

음의 $m$에 대해서는:

$$P_l^{-m}(x) = (-1)^m \frac{(l-m)!}{(l+m)!} P_l^m(x)$$

처음 몇 개의 결합 르장드르 함수:

| $P_l^m(\cos\theta)$ | 명시적 형태 |
|---|---|
| $P_0^0$ | $1$ |
| $P_1^0$ | $\cos\theta$ |
| $P_1^1$ | $-\sin\theta$ |
| $P_2^0$ | $\frac{1}{2}(3\cos^2\theta - 1)$ |
| $P_2^1$ | $-3\sin\theta\cos\theta$ |
| $P_2^2$ | $3\sin^2\theta$ |

**직교성**:

$$\int_{-1}^{1} P_l^m(x) P_{l'}^m(x) dx = \frac{2}{2l+1} \frac{(l+m)!}{(l-m)!} \delta_{ll'}$$

```python
import numpy as np
from scipy.special import lpmv
import matplotlib.pyplot as plt

# 결합 르장드르 함수 시각화
theta = np.linspace(0, np.pi, 200)
x = np.cos(theta)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
funcs = [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)]

for idx, (l, m) in enumerate(funcs):
    ax = axes[idx // 3, idx % 3]
    Plm = lpmv(m, l, x)
    ax.plot(theta * 180 / np.pi, Plm, 'b-', linewidth=2)
    ax.set_title(f'$P_{{{l}}}^{{{m}}}(\\cos\\theta)$')
    ax.set_xlabel('$\\theta$ (도)')
    ax.grid(True, alpha=0.3)

plt.suptitle('결합 르장드르 함수', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()

# 직교성 검증 (같은 m, 다른 l)
print("=== 직교성: ∫P_l^m P_l'^m dx (m=1) ===")
x_fine = np.linspace(-1, 1, 5000)
for l1 in range(1, 5):
    for l2 in range(l1, 5):
        inner = np.trapz(lpmv(1, l1, x_fine) * lpmv(1, l2, x_fine), x_fine)
        print(f"  <P_{l1}^1, P_{l2}^1> = {inner:.6f}")
```

#### 구면조화함수 (Spherical Harmonics)

$P_l^m$과 $e^{im\phi}$를 결합하면 **구면조화함수** $Y_l^m(\theta, \phi)$를 얻는다:

$$Y_l^m(\theta, \phi) = \sqrt{\frac{2l+1}{4\pi} \frac{(l-m)!}{(l+m)!}} \, P_l^m(\cos\theta) \, e^{im\phi}$$

이것은 구면 위의 **라플라시안의 고유함수**이다:

$$\nabla^2_{S^2} Y_l^m = -l(l+1) Y_l^m$$

**핵심 성질**:
- **직교정규성**: $\int_0^{2\pi}\int_0^{\pi} Y_l^m(\theta,\phi)^* Y_{l'}^{m'}(\theta,\phi) \sin\theta \, d\theta \, d\phi = \delta_{ll'}\delta_{mm'}$
- **완비성**: 구면 위의 임의의 함수 $f(\theta, \phi) = \sum_{l=0}^{\infty} \sum_{m=-l}^{l} c_{lm} Y_l^m(\theta, \phi)$
- **대칭 관계**: $Y_l^{-m} = (-1)^m (Y_l^m)^*$
- **덧셈 정리**: $P_l(\cos\gamma) = \frac{4\pi}{2l+1} \sum_{m=-l}^{l} Y_l^{m*}(\theta', \phi') Y_l^m(\theta, \phi)$

**물리 응용**:
- 수소 원자 파동함수의 각도 부분: $\psi_{nlm} = R_{nl}(r) Y_l^m(\theta, \phi)$
- 다중극 전개: 전위, 중력 퍼텐셜의 구면 전개
- 지구물리학: 지자기장, 중력장의 구면조화 분석

```python
import numpy as np
from scipy.special import sph_harm
import matplotlib.pyplot as plt

# 구면조화함수 |Y_l^m|² 시각화
theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2*np.pi, 100)
Theta, Phi = np.meshgrid(theta, phi)

fig, axes = plt.subplots(2, 3, figsize=(15, 10),
                         subplot_kw={'projection': '3d'})

harmonics = [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)]

for idx, (l, m) in enumerate(harmonics):
    ax = axes[idx // 3, idx % 3]

    # scipy sph_harm: sph_harm(m, l, phi, theta)
    Y = sph_harm(m, l, Phi, Theta)
    r = np.abs(Y)

    X = r * np.sin(Theta) * np.cos(Phi)
    Y_coord = r * np.sin(Theta) * np.sin(Phi)
    Z = r * np.cos(Theta)

    # 위상에 따른 색상
    fcolors = Y.real
    fmax = np.max(np.abs(fcolors))
    if fmax > 0:
        fcolors = fcolors / fmax

    ax.plot_surface(X, Y_coord, Z, facecolors=plt.cm.RdBu_r((fcolors + 1)/2),
                    rstride=2, cstride=2, alpha=0.8)
    ax.set_title(f'$Y_{{{l}}}^{{{m}}}(\\theta, \\phi)$')
    ax.set_xlim(-0.5, 0.5); ax.set_ylim(-0.5, 0.5); ax.set_zlim(-0.5, 0.5)

plt.suptitle('구면조화함수 (실수 부분에 따른 색상)', fontsize=14, y=1.02)
plt.tight_layout(); plt.show()

# 직교정규성 검증
print("=== 구면조화함수 직교정규성 ===")
dtheta = theta[1] - theta[0]
dphi = phi[1] - phi[0]
for l1, m1 in [(0,0), (1,0), (1,1), (2,0)]:
    for l2, m2 in [(0,0), (1,0), (1,1), (2,0)]:
        Y1 = sph_harm(m1, l1, Phi, Theta)
        Y2 = sph_harm(m2, l2, Phi, Theta)
        integrand = np.conj(Y1) * Y2 * np.sin(Theta)
        inner = np.trapz(np.trapz(integrand, phi, axis=0), theta)
        if abs(inner) > 0.01 or (l1==l2 and m1==m2):
            print(f"  <Y_{l1}^{m1}|Y_{l2}^{m2}> = {inner.real:.4f}")
```

---

## 5. 에르미트, 라게르 함수

### 5.1 에르미트 함수와 양자 조화 진동자

**에르미트 방정식**:

$$y'' - 2xy' + 2ny = 0, \quad n = 0, 1, 2, \ldots$$

다항식 해가 **에르미트 다항식** $H_n(x)$이다:

$$H_n(x) = (-1)^n e^{x^2} \frac{d^n}{dx^n} e^{-x^2}$$

처음 몇 개: $H_0 = 1$, $H_1 = 2x$, $H_2 = 4x^2 - 2$, $H_3 = 8x^3 - 12x$

**양자 조화 진동자**의 슈뢰딩거 방정식:

$$-\frac{\hbar^2}{2m}\frac{d^2\psi}{dx^2} + \frac{1}{2}m\omega^2 x^2 \psi = E\psi$$

무차원 변수 $\xi = \sqrt{m\omega/\hbar} \, x$를 도입하면 해가 된다:

$$\psi_n(\xi) = \frac{1}{\sqrt{2^n n! \sqrt{\pi}}} H_n(\xi) e^{-\xi^2/2}, \quad E_n = \hbar\omega\left(n + \frac{1}{2}\right)$$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite
from math import factorial, sqrt, pi

# --- 양자 조화 진동자 파동함수 ---
def psi_n(xi, n):
    """n번째 조화 진동자 파동함수"""
    Hn = hermite(n)
    norm = 1.0 / sqrt(2**n * factorial(n) * sqrt(pi))
    return norm * Hn(xi) * np.exp(-xi**2 / 2)

xi = np.linspace(-6, 6, 500)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 파동함수
for n in range(5):
    axes[0].plot(xi, psi_n(xi, n) + n, linewidth=1.5,
                 label=f'$\\psi_{{{n}}}$ ($E_{n} = {n+0.5}\\hbar\\omega$)')
    axes[0].axhline(y=n, color='gray', linewidth=0.3, linestyle='--')

axes[0].set_xlabel('$\\xi = \\sqrt{m\\omega/\\hbar} \\, x$')
axes[0].set_ylabel('$\\psi_n(\\xi)$ + offset')
axes[0].set_title('양자 조화 진동자 파동함수')
axes[0].legend(loc='upper right', fontsize=9)
axes[0].grid(True, alpha=0.2)

# 확률밀도
for n in range(5):
    prob = psi_n(xi, n)**2
    axes[1].fill_between(xi, n, prob + n, alpha=0.3)
    axes[1].plot(xi, prob + n, linewidth=1.5, label=f'$|\\psi_{{{n}}}|^2$')
    axes[1].axhline(y=n, color='gray', linewidth=0.3, linestyle='--')

# 고전적 확률밀도 (n=4, 비교용)
n_class = 4
A = np.sqrt(2 * n_class + 1)
xi_class = np.linspace(-A + 0.01, A - 0.01, 300)
prob_class = 1.0 / (np.pi * np.sqrt(A**2 - xi_class**2))
axes[1].plot(xi_class, prob_class + n_class, 'k--', linewidth=1.5,
             label='고전적 ($n=4$)')

axes[1].set_xlabel('$\\xi$')
axes[1].set_ylabel('$|\\psi_n|^2$ + offset')
axes[1].set_title('확률밀도 (양자 vs 고전 대응)')
axes[1].legend(loc='upper right', fontsize=9)
axes[1].grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('quantum_harmonic_oscillator.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 5.2 라게르 함수와 수소 원자

**연관 라게르 방정식**:

$$x y'' + (k + 1 - x) y' + n y = 0$$

해가 **연관 라게르 다항식** $L_n^k(x)$이다.

**수소 원자**의 방사 파동함수:

$$R_{nl}(r) = N_{nl} \left(\frac{2r}{na_0}\right)^l e^{-r/(na_0)} L_{n-l-1}^{2l+1}\!\left(\frac{2r}{na_0}\right)$$

여기서 $n$은 주양자수, $l$은 궤도 양자수, $a_0$는 보어 반지름이다. 에너지 준위:

$$E_n = -\frac{13.6 \text{ eV}}{n^2}$$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import assoc_laguerre, factorial
from math import sqrt

# --- 수소 원자 방사 파동함수 ---
a0 = 1.0  # 보어 반지름 단위

def R_nl(r, n, l):
    """수소 원자 방사 파동함수 R_nl(r)"""
    rho = 2 * r / (n * a0)
    norm = sqrt((2 / (n * a0))**3 * factorial(n - l - 1)
                / (2 * n * factorial(n + l)))
    return norm * rho**l * np.exp(-rho / 2) * assoc_laguerre(rho, n - l - 1, 2*l + 1)

r = np.linspace(0, 25, 500)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 방사 파동함수 R_nl(r)
states = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)]
for n, l in states:
    axes[0].plot(r, R_nl(r, n, l), linewidth=1.5,
                 label=f'$R_{{{n}{l}}}$ ($n$={n}, $l$={l})')

axes[0].set_xlabel('$r / a_0$')
axes[0].set_ylabel('$R_{nl}(r)$')
axes[0].set_title('수소 원자 방사 파동함수')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=0, color='k', linewidth=0.5)

# 방사 확률밀도 r^2 |R_nl|^2
for n, l in states:
    prob = r**2 * R_nl(r, n, l)**2
    axes[1].plot(r, prob, linewidth=1.5,
                 label=f'$r^2|R_{{{n}{l}}}|^2$')

axes[1].set_xlabel('$r / a_0$')
axes[1].set_ylabel('$r^2 |R_{nl}|^2$')
axes[1].set_title('방사 확률밀도')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hydrogen_radial.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 6. 감마 함수와 베타 함수

### 6.1 감마 함수의 정의와 성질

**감마 함수**는 계승(factorial)을 실수(및 복소수)로 확장한 것이다:

$$\Gamma(z) = \int_0^{\infty} t^{z-1} e^{-t} \, dt, \quad \text{Re}(z) > 0$$

**핵심 성질**:

| 성질 | 공식 |
|---|---|
| 점화식 | $\Gamma(z+1) = z \Gamma(z)$ |
| 계승과의 관계 | $\Gamma(n+1) = n!$ ($n$이 비음의 정수) |
| 반정수값 | $\Gamma(1/2) = \sqrt{\pi}$ |
| 반사 공식 | $\Gamma(z)\Gamma(1-z) = \frac{\pi}{\sin(\pi z)}$ |
| 곱셈 공식 (르장드르) | $\Gamma(z)\Gamma(z+\frac{1}{2}) = \frac{\sqrt{\pi}}{2^{2z-1}}\Gamma(2z)$ |
| 스털링 근사 | $\Gamma(z+1) \sim \sqrt{2\pi z} \left(\frac{z}{e}\right)^z$ ($z \to \infty$) |

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# --- 감마 함수 시각화 ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 실수축에서의 감마 함수
x = np.linspace(-4.5, 5, 1000)
y = np.array([gamma(xi) if abs(xi - round(xi)) > 0.01 or xi > 0
              else np.nan for xi in x])

axes[0].plot(x, y, 'b-', linewidth=1.5)
# 양의 정수에서의 값 (계승)
for n in range(1, 6):
    axes[0].plot(n, gamma(n), 'ro', markersize=6)
    axes[0].annotate(f'{n-1}! = {int(gamma(n))}',
                     xy=(n, gamma(n)), xytext=(n+0.2, gamma(n)+2),
                     fontsize=9)

axes[0].set_xlim(-4.5, 5.5)
axes[0].set_ylim(-10, 25)
axes[0].axhline(y=0, color='k', linewidth=0.5)
axes[0].axvline(x=0, color='k', linewidth=0.5)
axes[0].set_xlabel('$x$')
axes[0].set_ylabel('$\\Gamma(x)$')
axes[0].set_title('감마 함수 $\\Gamma(x)$')
axes[0].grid(True, alpha=0.3)

# 스털링 근사 비교
n_vals = np.arange(1, 15)
exact = np.array([gamma(n + 1) for n in n_vals])
stirling = np.sqrt(2 * np.pi * n_vals) * (n_vals / np.e)**n_vals
rel_error = np.abs(stirling - exact) / exact

axes[1].semilogy(n_vals, rel_error, 'bo-', linewidth=1.5, markersize=6)
axes[1].set_xlabel('$n$')
axes[1].set_ylabel('상대 오차')
axes[1].set_title('스털링 근사 $n! \\approx \\sqrt{2\\pi n}(n/e)^n$의 정확도')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gamma_function.png', dpi=150, bbox_inches='tight')
plt.show()

# 반사 공식 검증
print("감마 함수 반사 공식 검증: Gamma(z) * Gamma(1-z) = pi/sin(pi*z)")
for z in [0.25, 0.5, 0.75, 1.3]:
    lhs = gamma(z) * gamma(1 - z)
    rhs = np.pi / np.sin(np.pi * z)
    print(f"  z = {z}: LHS = {lhs:.8f}, RHS = {rhs:.8f}, "
          f"차이 = {abs(lhs-rhs):.2e}")
```

### 6.2 베타 함수

**베타 함수**는 다음과 같이 정의된다:

$$B(p, q) = \int_0^1 t^{p-1}(1-t)^{q-1} \, dt, \quad p, q > 0$$

감마 함수와의 관계:

$$\boxed{B(p, q) = \frac{\Gamma(p)\Gamma(q)}{\Gamma(p+q)}}$$

이 공식은 매우 유용하며, 복잡한 적분을 감마 함수 값으로 바로 계산할 수 있게 해준다.

**응용 예시**: $\int_0^{\pi/2} \sin^m\theta \cos^n\theta \, d\theta = \frac{1}{2} B\!\left(\frac{m+1}{2}, \frac{n+1}{2}\right)$

```python
import numpy as np
from scipy.special import gamma, beta

# --- 베타 함수를 이용한 적분 계산 ---
print("베타 함수를 이용한 적분 계산:")
print("=" * 55)

# 예제 1: int_0^1 x^3 (1-x)^4 dx = B(4, 5) = 3!*4!/8!
result = beta(4, 5)
exact = 6 * 24 / 40320
print(f"\n∫₀¹ x³(1-x)⁴ dx = B(4,5) = {result:.8f}")
print(f"  = 3!·4!/8! = {exact:.8f}")

# 예제 2: int_0^{pi/2} sin^5(theta) d(theta)
# = (1/2) B(3, 1/2) = (1/2) * Gamma(3)*Gamma(1/2) / Gamma(7/2)
m = 5
integral = 0.5 * beta((m + 1)/2, 0.5)
print(f"\n∫₀^(π/2) sin⁵θ dθ = (1/2)B(3, 1/2) = {integral:.8f}")
print(f"  이론값 = 8/15 = {8/15:.8f}")

# 예제 3: 가우시안 적분의 일반화
# int_0^inf x^(2n) e^(-x^2) dx = Gamma(n + 1/2) / 2
print(f"\n가우시안 적분의 일반화:")
for n in range(5):
    val = gamma(n + 0.5) / 2
    print(f"  ∫₀^∞ x^{2*n} e^(-x²) dx = Γ({n}+1/2)/2 = {val:.6f}")

# 스털링 공식: Gamma(1/2) = sqrt(pi) 확인
print(f"\nΓ(1/2) = {gamma(0.5):.10f}")
print(f"√π     = {np.sqrt(np.pi):.10f}")
```

---

## 연습 문제

### 기본 문제

**문제 1**: 다음 ODE의 정상점과 특이점을 분류하시오.
- (a) $(1-x^2)y'' - 2xy' + 6y = 0$ (르장드르 방정식, $l=2$)
- (b) $xy'' + (1-x)y' + ny = 0$ (라게르 방정식)
- (c) $y'' + \frac{1}{x}y' + \left(1 - \frac{4}{x^2}\right)y = 0$ (베셀 방정식, $\nu=2$)

<details>
<summary>풀이 힌트</summary>

(a) 표준형: $P(x) = \frac{-2x}{1-x^2}$, $Q(x) = \frac{6}{1-x^2}$

- $x = 0$: 정상점 ($P$, $Q$ 해석적)
- $x = \pm 1$: 특이점 ($P$, $Q$ 발산)
  - $(x-1)P(x) = \frac{-2x}{-(1+x)} \to 1$ (해석적), $(x-1)^2 Q(x) \to 0$ (해석적) → 정칙 특이점

(b) 표준형: $P(x) = (1-x)/x$, $Q(x) = n/x$

- $x = 0$: $xP(x) = 1-x$ (해석적), $x^2 Q(x) = nx$ (해석적) → 정칙 특이점

(c) $x = 0$: $xP(x) = 1$ (해석적), $x^2 Q(x) = x^2 - 4$ (해석적) → 정칙 특이점

</details>

**문제 2**: 에어리 방정식 $y'' - xy = 0$의 멱급수 해에서 처음 6개의 비영 항을 구하시오 ($a_0 = 1$, $a_1 = 0$인 경우와 $a_0 = 0$, $a_1 = 1$인 경우).

<details>
<summary>풀이 힌트</summary>

점화 관계: $a_{m+2} = \frac{a_{m-1}}{(m+2)(m+1)}$, $a_2 = 0$

**경우 1** ($a_0 = 1$, $a_1 = 0$): $a_3 = \frac{1}{6}$, $a_6 = \frac{1}{180}$, $a_9 = \frac{1}{12960}$, ...

$$y_1 = 1 + \frac{x^3}{6} + \frac{x^6}{180} + \frac{x^9}{12960} + \cdots$$

**경우 2** ($a_0 = 0$, $a_1 = 1$): $a_4 = \frac{1}{12}$, $a_7 = \frac{1}{504}$, $a_{10} = \frac{1}{45360}$, ...

$$y_2 = x + \frac{x^4}{12} + \frac{x^7}{504} + \frac{x^{10}}{45360} + \cdots$$

```python
import sympy as sp
x = sp.Symbol('x')
y = sp.Function('y')
sol = sp.dsolve(y(x).diff(x, 2) - x*y(x), y(x), hint='power_series', n=12)
print(sol)
```

</details>

### 응용 문제

**문제 3** (베셀 함수): 반지름 $a = 1$인 원형 막에서 축대칭($m = 0$) 모드의 처음 5개 고유 진동수 비율 $\omega_n / \omega_1$을 구하시오.

<details>
<summary>풀이 힌트</summary>

축대칭 모드의 진동수: $\omega_n = c \, \alpha_{0,n} / a$

여기서 $\alpha_{0,n}$은 $J_0(x) = 0$의 $n$번째 양의 근이다.

```python
from scipy.special import jn_zeros

zeros_J0 = jn_zeros(0, 5)
print("J_0(x)의 영점:", zeros_J0)
print("진동수 비율:")
for n in range(5):
    print(f"  omega_{n+1}/omega_1 = {zeros_J0[n]/zeros_J0[0]:.4f}")
```

</details>

**문제 4** (르장드르 다항식): 전하 $+q$가 $(0, 0, d)$에, $-q$가 $(0, 0, -d)$에 위치한 전기 쌍극자의 전위를 $r \gg d$ 영역에서 르장드르 다항식을 이용해 전개하고, 쌍극자 항($l = 1$)이 지배적임을 보이시오.

<details>
<summary>풀이 힌트</summary>

$$\Phi = \frac{q}{4\pi\epsilon_0}\left(\frac{1}{|\mathbf{r} - d\hat{z}|} - \frac{1}{|\mathbf{r} + d\hat{z}|}\right)$$

각각을 르장드르 급수로 전개:

$$\Phi = \frac{q}{4\pi\epsilon_0} \sum_{l=0}^{\infty} \frac{d^l}{r^{l+1}} P_l(\cos\theta) [1 - (-1)^l]$$

$l$이 짝수이면 $[1 - 1] = 0$이므로 홀수 $l$만 남는다. $l = 1$ (쌍극자) 항:

$$\Phi_{\text{dipole}} = \frac{q \cdot 2d}{4\pi\epsilon_0} \frac{\cos\theta}{r^2} = \frac{p \cos\theta}{4\pi\epsilon_0 r^2}$$

여기서 $p = 2qd$는 쌍극자 모멘트이다.

</details>

**문제 5** (감마/베타 함수): 다음 적분을 감마 함수와 베타 함수를 이용해 계산하시오.

- (a) $\int_0^{\infty} x^4 e^{-x^2} dx$
- (b) $\int_0^1 x^2 \sqrt{1 - x^2} \, dx$
- (c) $n$차원 초구의 체적 $V_n(R) = \frac{\pi^{n/2}}{\Gamma(n/2 + 1)} R^n$에서 $V_1, V_2, V_3, V_4$를 구하시오.

<details>
<summary>풀이 힌트</summary>

(a) $t = x^2$로 치환하면: $\frac{1}{2}\int_0^{\infty} t^{3/2} e^{-t} dt = \frac{1}{2}\Gamma(5/2) = \frac{1}{2} \cdot \frac{3}{2} \cdot \frac{1}{2} \cdot \sqrt{\pi} = \frac{3\sqrt{\pi}}{8}$

(b) $x = \sin\theta$로 치환하면: $\int_0^{\pi/2} \sin^2\theta \cos^2\theta \, d\theta = \frac{1}{2}B(3/2, 3/2) = \frac{\pi}{16}$

(c)
```python
from scipy.special import gamma
import numpy as np

for n in range(1, 5):
    V = np.pi**(n/2) / gamma(n/2 + 1)
    print(f"V_{n}(R=1) = pi^({n}/2) / Gamma({n}/2 + 1) = {V:.6f}")
# V_1 = 2 (선분 길이), V_2 = pi (원), V_3 = 4pi/3 (구), V_4 = pi^2/2
```

</details>

---

## 참고 자료

### 교재
1. **Boas, M. L.** (2005). *Mathematical Methods in the Physical Sciences*, 3rd ed., Chapters 11-12. Wiley.
   - 본 레슨의 주요 참고서
2. **Arfken, G. B., Weber, H. J., & Harris, F. E.** (2012). *Mathematical Methods for Physicists*, 7th ed., Chapters 7-11. Academic Press.
   - 대학원 수준의 특수함수 이론
3. **Watson, G. N.** (1995). *A Treatise on the Theory of Bessel Functions*, 2nd ed. Cambridge University Press.
   - 베셀 함수에 대한 고전적 참고서
4. **Griffiths, D. J.** (2018). *Introduction to Quantum Mechanics*, 3rd ed. Cambridge University Press.
   - 에르미트/라게르 함수의 양자역학 응용

### 온라인 자료
1. **NIST Digital Library of Mathematical Functions**: https://dlmf.nist.gov/
   - 특수함수의 공식, 성질, 수치 데이터의 표준 참고
2. **MIT OCW 18.03**: Differential Equations — Power Series Solutions
3. **3Blue1Brown**: 특수함수의 직관적 이해

### 핵심 라이브러리 문서
1. **SciPy `scipy.special`**: https://docs.scipy.org/doc/scipy/reference/special.html
   - `jv`, `yv` (베셀), `legendre` (르장드르), `hermite` (에르미트), `gamma`, `beta` 등
2. **SymPy Special Functions**: https://docs.sympy.org/latest/modules/functions/special.html

---

## 다음 레슨

[12. 스투름-리우빌 이론 (Sturm-Liouville Theory)](12_Sturm_Liouville_Theory.md)에서는 이 레슨에서 다룬 베셀 함수, 르장드르 다항식, 에르미트 함수 등이 모두 **스투름-리우빌 고유값 문제**의 특수한 경우임을 보이고, **직교 함수계의 완비성(completeness)**과 **일반 푸리에 급수**의 수학적 기반을 다룹니다.
