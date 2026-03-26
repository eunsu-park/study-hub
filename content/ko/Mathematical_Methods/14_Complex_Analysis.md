# 14. 복소해석 (Complex Analysis)

## 학습 목표

- 복소함수의 미분 가능성과 코시-리만 조건을 이해한다
- 코시 적분 정리와 적분 공식을 활용하여 복소 적분을 계산한다
- 테일러 급수와 로랑 급수를 통해 복소함수를 급수로 전개한다
- 유수 정리를 이용하여 실수 적분 문제를 효율적으로 풀 수 있다
- 등각사상의 개념을 이해하고 물리학 문제에 적용한다

> **물리학에서의 중요성**: 복소해석은 양자역학의 전파함수, 전기역학의 전위론, 유체역학의 흐름 함수, 신호처리의 주파수 분석 등 물리학 전반에 걸쳐 핵심 도구이다. 특히 유수 정리를 통한 적분 계산은 이론물리학에서 가장 빈번하게 활용되는 기법 중 하나이다.

---

## 1. 해석함수 (Analytic Functions)

### 1.1 복소 미분과 코시-리만 조건

복소함수 $f(z) = u(x, y) + iv(x, y)$가 점 $z_0$에서 **미분 가능**하려면, 극한

$$f'(z_0) = \lim_{\Delta z \to 0} \frac{f(z_0 + \Delta z) - f(z_0)}{\Delta z}$$

이 $\Delta z$가 어떤 방향에서 접근하든 같은 값으로 수렴해야 한다. 이로부터 **코시-리만 방정식**이 도출된다:

$$\frac{\partial u}{\partial x} = \frac{\partial v}{\partial y}, \quad \frac{\partial u}{\partial y} = -\frac{\partial v}{\partial x}$$

한 영역의 모든 점에서 미분 가능한 함수를 그 영역에서 **해석적(analytic)**이라 한다.

**극좌표 형태** ($z = re^{i\theta}$):

$$\frac{\partial u}{\partial r} = \frac{1}{r}\frac{\partial v}{\partial \theta}, \quad \frac{1}{r}\frac{\partial u}{\partial \theta} = -\frac{\partial v}{\partial r}$$

```python
import numpy as np
import matplotlib.pyplot as plt

def check_cauchy_riemann(u_func, v_func, x, y, h=1e-7):
    """코시-리만 조건을 수치적으로 검증"""
    du_dx = (u_func(x + h, y) - u_func(x - h, y)) / (2 * h)
    du_dy = (u_func(x, y + h) - u_func(x, y - h)) / (2 * h)
    dv_dx = (v_func(x + h, y) - v_func(x - h, y)) / (2 * h)
    dv_dy = (v_func(x, y + h) - v_func(x, y - h)) / (2 * h)

    cond1 = np.abs(du_dx - dv_dy)   # ∂u/∂x = ∂v/∂y
    cond2 = np.abs(du_dy + dv_dx)   # ∂u/∂y = -∂v/∂x

    print(f"점 ({x}, {y}): |∂u/∂x - ∂v/∂y| = {cond1:.2e}, "
          f"|∂u/∂y + ∂v/∂x| = {cond2:.2e}")
    return cond1 < 1e-5 and cond2 < 1e-5

# f(z) = z² = (x² - y²) + i(2xy)
u = lambda x, y: x**2 - y**2
v = lambda x, y: 2 * x * y

print("=== f(z) = z² 코시-리만 검증 ===")
for pt in [(1, 1), (2, -1), (0.5, 3)]:
    analytic = check_cauchy_riemann(u, v, *pt)
    print(f"  해석적: {analytic}")
```

### 1.2 조화함수 (Harmonic Functions)

해석함수의 실수부 $u$와 허수부 $v$는 각각 **라플라스 방정식**을 만족한다:

$$\nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = 0, \quad \nabla^2 v = 0$$

이러한 함수를 **조화함수(harmonic function)**라 하며, $u$와 $v$는 서로의 **조화 켤레**이다.

**물리적 의미**: 2차원 정전기학에서 전위 $\phi(x,y)$는 라플라스 방정식을 만족하므로 조화함수이다. 해석함수의 실수부를 전위로, 허수부를 전기력선 함수로 해석할 수 있다.

```python
import sympy as sp

x, y = sp.symbols('x y', real=True)

def is_harmonic(expr):
    """라플라시안이 0인지 확인"""
    lap = sp.simplify(sp.diff(expr, x, 2) + sp.diff(expr, y, 2))
    return lap, lap == 0

# f(z) = z³의 실수부/허수부
u_expr = x**3 - 3*x*y**2      # Re(z³)
v_expr = 3*x**2*y - y**3      # Im(z³)

for name, expr in [("Re(z³)", u_expr), ("Im(z³)", v_expr)]:
    lap, harmonic = is_harmonic(expr)
    print(f"{name} = {expr}: ∇² = {lap}, 조화함수: {harmonic}")
```

### 1.3 해석 함수의 예

물리학에서 자주 등장하는 해석함수들:

| 함수 | 실수부 $u$ | 허수부 $v$ | 물리적 응용 |
|------|-----------|-----------|------------|
| $e^z$ | $e^x \cos y$ | $e^x \sin y$ | 파동, 감쇠 |
| $\ln z$ | $\ln r$ | $\theta$ | 선전하 전위 |
| $z^n$ | $r^n \cos n\theta$ | $r^n \sin n\theta$ | 다극자 전개 |
| $1/z$ | $x/(x^2+y^2)$ | $-y/(x^2+y^2)$ | 점전하, 소스/싱크 |

```python
from matplotlib.colors import hsv_to_rgb

def domain_coloring(f, xlim=(-2, 2), ylim=(-2, 2), N=500):
    """복소함수의 도메인 컬러링 시각화 (위상→색상, 크기→명도)"""
    xv = np.linspace(*xlim, N)
    yv = np.linspace(*ylim, N)
    X, Y = np.meshgrid(xv, yv)
    Z = X + 1j * Y

    with np.errstate(divide='ignore', invalid='ignore'):
        W = f(Z)

    H = (np.angle(W) + np.pi) / (2 * np.pi)
    V = 1 - 1 / (1 + np.abs(W)**0.3)
    HSV = np.stack([H, np.ones_like(H), V], axis=-1)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(hsv_to_rgb(HSV), extent=[*xlim, *ylim], origin='lower')
    ax.set_xlabel('Re(z)'); ax.set_ylabel('Im(z)')
    plt.tight_layout(); plt.show()

# domain_coloring(lambda z: z**3)       # z³의 도메인 컬러링
# domain_coloring(lambda z: np.exp(z))  # e^z의 도메인 컬러링
```

---

## 2. 복소 적분

### 2.1 경로 적분 (Contour Integrals)

경로 $C$를 따라 복소함수를 적분하는 것을 **경로 적분**이라 한다:

$$\oint_C f(z)\, dz = \int_a^b f(z(t))\, z'(t)\, dt$$

**핵심 결과**: $|z - z_0| = r$ 위의 반시계 방향 적분에 대해:

$$\oint \frac{dz}{(z - z_0)^n} = \begin{cases} 2\pi i & n = 1 \\ 0 & n \neq 1 \end{cases}$$

```python
def contour_integral_circle(f, z0=0, r=1, N=10000):
    """원형 경로 |z-z0|=r 위에서의 경로 적분 (수치 계산)"""
    t = np.linspace(0, 2*np.pi, N, endpoint=False)
    dt = 2*np.pi / N
    z = z0 + r * np.exp(1j * t)
    dz_dt = 1j * r * np.exp(1j * t)
    return np.sum(f(z) * dz_dt) * dt

# ∮ 1/z dz = 2πi
I1 = contour_integral_circle(lambda z: 1/z)
print(f"∮ 1/z dz = {I1:.6f}  (이론값: {2*np.pi*1j:.6f})")

# ∮ 1/z² dz = 0
I2 = contour_integral_circle(lambda z: 1/z**2)
print(f"∮ 1/z² dz = {I2:.6f}  (이론값: 0)")

# ∮ e^z/z dz = 2πi (코시 공식: f(0)=e⁰=1)
I3 = contour_integral_circle(lambda z: np.exp(z)/z)
print(f"∮ e^z/z dz = {I3:.6f}  (이론값: {2*np.pi*1j:.6f})")
```

### 2.2 코시 적분 정리 (Cauchy's Theorem)

$f(z)$가 단순 연결 영역 $D$ 내에서 해석적이면, $D$ 안의 임의의 폐곡선 $C$에 대해:

$$\oint_C f(z)\, dz = 0$$

**물리적 의미**: 보존력의 순환이 0인 것과 동치이다. 해석함수의 적분은 경로에 무관하다.

### 2.3 코시 적분 공식 (Cauchy's Formula)

$f(z)$가 $C$ 내부에서 해석적이고 $z_0$가 내부 점이면:

$$f(z_0) = \frac{1}{2\pi i} \oint_C \frac{f(z)}{z - z_0}\, dz$$

**일반화** (n차 도함수):

$$f^{(n)}(z_0) = \frac{n!}{2\pi i} \oint_C \frac{f(z)}{(z - z_0)^{n+1}}\, dz$$

이 공식은 해석함수가 무한번 미분 가능하며, 경계값이 내부값을 결정함을 의미한다.

```python
from math import factorial

def cauchy_derivative(f, z0, n=0, r=1, N=10000):
    """코시 적분 공식으로 f^(n)(z0) 계산"""
    t = np.linspace(0, 2*np.pi, N, endpoint=False)
    dt = 2*np.pi / N
    z = z0 + r * np.exp(1j * t)
    dz_dt = 1j * r * np.exp(1j * t)
    integrand = f(z) / (z - z0)**(n + 1) * dz_dt
    return np.sum(integrand) * dt * factorial(n) / (2*np.pi*1j)

# 검증: f(z) = sin(z)
f = lambda z: np.sin(z)
print("코시 공식 검증: f(z) = sin(z)")
print(f"  f(0)   = {cauchy_derivative(f, 0, 0):.6f}  (정확값: 0)")
print(f"  f'(0)  = {cauchy_derivative(f, 0, 1):.6f}  (정확값: 1)")
print(f"  f''(0) = {cauchy_derivative(f, 0, 2):.6f}  (정확값: 0)")
print(f"  f'''(0)= {cauchy_derivative(f, 0, 3):.6f}  (정확값: -1)")
```

---

## 3. 급수 전개

### 3.1 테일러 급수

$f(z)$가 $z_0$ 중심 반지름 $R$인 원 내에서 해석적이면:

$$f(z) = \sum_{n=0}^{\infty} a_n (z - z_0)^n, \quad a_n = \frac{f^{(n)}(z_0)}{n!}$$

수렴 반지름 $R$은 $z_0$에서 가장 가까운 특이점까지의 거리이다.

**예**: $f(z) = 1/(1+z^2)$의 $z=0$ 주위 테일러 급수 수렴 반지름은 $R=1$이다. $z = \pm i$에 극점이 있기 때문이다. 실수축에서는 특이성이 없지만, 복소평면의 특이점이 수렴 반지름을 결정한다.

### 3.2 로랑 급수 (Laurent Series)

$f(z)$가 고리형 영역 $r < |z - z_0| < R$에서 해석적이면:

$$f(z) = \sum_{n=-\infty}^{\infty} a_n (z - z_0)^n$$

$n < 0$인 항들을 **주부(principal part)**, 특히 $a_{-1}$을 **유수(residue)**라 한다.

```python
z = sp.Symbol('z')

# 로랑 급수: e^z / z³ (z=0 주위)
f1 = sp.exp(z) / z**3
print("e^z/z³ 의 로랑 급수:")
print(f"  {sp.series(f1, z, 0, n=5)}")
print(f"  유수 (z⁻¹ 계수) = 1/2\n")

# 로랑 급수: 1/(z(z-1)) (z=0 주위)
f2 = 1 / (z * (z - 1))
print("1/(z(z-1)) 의 로랑 급수 (z=0 주위):")
print(f"  {sp.series(f2, z, 0, n=4)}")

# z=1 주위
w = sp.Symbol('w')
print("\n1/(z(z-1)) 의 로랑 급수 (z=1 주위, w=z-1):")
print(f"  {sp.series(f2.subs(z, w+1), w, 0, n=4)}")
```

### 3.3 특이점의 분류 (제거가능, 극점, 본질적)

| 종류 | 주부의 항 수 | 예시 | $\lim_{z \to z_0} f(z)$ |
|------|-------------|------|------------------------|
| **제거가능 특이점** | 0개 | $\sin z / z$ at $z=0$ | 유한값 |
| **$m$차 극점** | $m$개 | $1/z^m$ at $z=0$ | $\infty$ |
| **본질적 특이점** | 무한개 | $e^{1/z}$ at $z=0$ | 존재 안 함 |

**카소라티-바이어슈트라스 정리**: 본질적 특이점 근방에서 함수는 거의 모든 복소수 값을 취한다.

```python
# 특이점 분류 확인
cases = [
    ("sin(z)/z (z=0, 제거가능)", sp.sin(z)/z, z, 0),
    ("1/(z-1)³ (z=1, 3차 극점)", 1/(z-1)**3, z, 1),
    ("exp(1/z) (z=0, 본질적)", sp.exp(1/z), z, 0),
]

for name, expr, var, pt in cases:
    if pt != 0:
        w = sp.Symbol('w')
        s = sp.series(expr.subs(var, w + pt), w, 0, n=5)
    else:
        s = sp.series(expr, var, 0, n=5)
    print(f"{name}:\n  {s}\n")
```

---

## 4. 유수 정리 (Residue Theorem)

### 4.1 유수의 정의와 계산

점 $z_0$에서 $f(z)$의 **유수**는 로랑 급수의 $a_{-1}$ 계수이다:

$$\text{Res}_{z=z_0} f(z) = \frac{1}{2\pi i} \oint_C f(z)\, dz$$

**유수 계산법**:

1. **단순 극점**: $\text{Res}_{z=z_0} f = \lim_{z \to z_0} (z - z_0) f(z)$
2. **$m$차 극점**: $\text{Res}_{z=z_0} f = \frac{1}{(m-1)!} \lim_{z \to z_0} \frac{d^{m-1}}{dz^{m-1}} [(z - z_0)^m f(z)]$
3. **$p/q$ 형태** (단순 극점): $\text{Res}_{z=z_0} \frac{p}{q} = \frac{p(z_0)}{q'(z_0)}$

```python
z = sp.Symbol('z')

examples = [
    ("1/(z²+1)", 1/(z**2+1), sp.I),
    ("1/(z²+1)", 1/(z**2+1), -sp.I),
    ("e^z/z²", sp.exp(z)/z**2, 0),
    ("z/(z²-3z+2)", z/(z**2-3*z+2), 1),
    ("z/(z²-3z+2)", z/(z**2-3*z+2), 2),
]

print("=== 유수 계산 ===")
for name, expr, z0 in examples:
    print(f"Res[{name}, z={z0}] = {sp.residue(expr, z, z0)}")
```

### 4.2 유수 정리

$f(z)$가 폐곡선 $C$ 내부에서 유한 개의 특이점 $z_1, \ldots, z_n$을 제외하고 해석적이면:

$$\oint_C f(z)\, dz = 2\pi i \sum_{k=1}^{n} \text{Res}_{z=z_k} f(z)$$

### 4.3 조르당 보조정리 (Jordan's Lemma)

유수 정리로 실수 적분을 계산할 때, 무한 반원 경로의 기여가 0인지 확인해야 한다. **조르당 보조정리**는 이를 보장한다.

**정리**: $f(z) \to 0$ uniformly as $|z| \to \infty$ (상반면)이면, $a > 0$에 대해:

$$\lim_{R \to \infty} \int_{C_R} f(z) e^{iaz}\, dz = 0$$

여기서 $C_R$은 상반면의 반지름 $R$인 반원이다. 핵심은 $e^{iaz} = e^{ia(x+iy)} = e^{iax}e^{-ay}$이므로, 상반면($y > 0$)에서 $e^{-ay}$가 지수적으로 감소한다는 것이다.

> **주의**: $a < 0$이면 하반면 반원을 사용해야 하며, 이 경우 경로의 방향이 시계방향이므로 유수에 $(-2\pi i)$가 곱해진다.

### 4.4 실수 적분 계산의 4가지 유형

유수 정리의 가장 중요한 응용은 실수 적분의 계산이다. 적분의 형태에 따라 체계적으로 분류한다.

#### 유형 1: 삼각함수 유리식 — $\int_0^{2\pi} R(\cos\theta, \sin\theta)\, d\theta$

$z = e^{i\theta}$ 치환: $\cos\theta = (z + z^{-1})/2$, $\sin\theta = (z - z^{-1})/(2i)$, $d\theta = dz/(iz)$.

적분이 단위원 $|z| = 1$ 위의 경로 적분으로 변환된다. 단위원 **내부**의 극점에 대한 유수만 계산한다.

```python
from scipy.integrate import quad
import sympy as sp
import numpy as np

z = sp.Symbol('z')

# --- 유형 1: ∫₀²π dθ/(2 + cosθ) ---
print("=== 유형 1: ∫₀²π dθ/(2 + cosθ) ===")
integrand1 = 2 / (sp.I * (z**2 + 4*z + 1))
inner_pole = -2 + sp.sqrt(3)  # |z| < 1인 극점
res1 = sp.residue(integrand1, z, inner_pole)
result1 = sp.simplify(2 * sp.pi * sp.I * res1)
print(f"유수 정리: {result1} = {float(result1):.6f}")
num1, _ = quad(lambda t: 1/(2 + np.cos(t)), 0, 2*np.pi)
print(f"수치 검증: {num1:.6f}\n")
```

#### 유형 2: 유리함수 — $\int_{-\infty}^{\infty} \frac{P(x)}{Q(x)}\, dx$

조건: $\deg(Q) \geq \deg(P) + 2$ (적분이 수렴), $Q(x) \neq 0$ on 실수축.

상반면 반원 경로를 사용. $R \to \infty$에서 반원 위의 적분이 0이 되므로 ($f(z) \to 0$ sufficiently fast):

$$\int_{-\infty}^{\infty} \frac{P(x)}{Q(x)}\, dx = 2\pi i \sum_{\text{Im}(z_k) > 0} \text{Res}_{z=z_k} \frac{P(z)}{Q(z)}$$

```python
# --- 유형 2: ∫₋∞^∞ dx/(x²+1)² ---
print("=== 유형 2: ∫₋∞^∞ dx/(x²+1)² ===")
res2 = sp.residue(1/(z**2+1)**2, z, sp.I)
result2 = sp.simplify(2 * sp.pi * sp.I * res2)
print(f"유수 정리: {result2} = {float(result2):.6f}")
num2, _ = quad(lambda x: 1/(x**2+1)**2, -100, 100)
print(f"수치 검증: {num2:.6f}\n")
```

#### 유형 3: 푸리에형 적분 — $\int_{-\infty}^{\infty} f(x) e^{iax}\, dx$ ($a > 0$)

조르당 보조정리를 적용. $f(z) \to 0$ as $|z| \to \infty$이면 (1차 충분) 상반면 반원의 기여가 0:

$$\int_{-\infty}^{\infty} f(x) e^{iax}\, dx = 2\pi i \sum_{\text{Im}(z_k) > 0} \text{Res}_{z=z_k} f(z) e^{iaz}$$

$\cos(ax)$나 $\sin(ax)$를 포함하는 적분은 $e^{iax}$를 사용한 후 실수부/허수부를 취한다.

```python
# --- 유형 3: ∫₋∞^∞ cos(x)/(x²+1) dx = π/e ---
print("=== 유형 3: ∫₋∞^∞ cos(x)/(x²+1) dx ===")
f3 = sp.exp(sp.I*z) / (z**2 + 1)
res3 = sp.residue(f3, z, sp.I)
result3 = sp.simplify(2 * sp.pi * sp.I * res3)
print(f"유수 정리: Re({result3}) = π/e = {float(sp.pi/sp.E):.6f}")
num3, _ = quad(lambda x: np.cos(x)/(x**2+1), -100, 100)
print(f"수치 검증: {num3:.6f}\n")
```

#### 유형 4: 분지절단 적분 — $\int_0^{\infty} x^{a-1} f(x)\, dx$ ($0 < a < 1$)

피적분함수에 $x^a$ ($a$가 정수가 아닌 경우)가 포함되면 **분지절단(branch cut)**이 필요하다. 열쇠구멍(keyhole) 경로를 사용한다.

**대표 예제**: $\int_0^{\infty} \frac{x^{a-1}}{1+x}\, dx = \frac{\pi}{\sin(\pi a)}$ ($0 < a < 1$)

**풀이 전략**:
1. $f(z) = z^{a-1}/(1+z)$에서 양의 실수축을 분지절단으로 선택
2. 열쇠구멍 경로: 분지절단 위쪽 → 큰 원 → 분지절단 아래쪽 → 작은 원
3. 분지절단 아래쪽에서 $z^{a-1} = |z|^{a-1} e^{2\pi i(a-1)}$
4. $z = -1$에서의 유수: $e^{i\pi(a-1)} = -e^{i\pi a}$

$$\int_0^{\infty} \frac{x^{a-1}}{1+x} dx - e^{2\pi i(a-1)} \int_0^{\infty} \frac{x^{a-1}}{1+x} dx = 2\pi i \cdot (-e^{i\pi a})$$

$$(1 - e^{2\pi i(a-1)}) I = -2\pi i e^{i\pi a} \implies I = \frac{\pi}{\sin(\pi a)}$$

```python
# --- 유형 4: ∫₀^∞ x^{a-1}/(1+x) dx = π/sin(πa) ---
print("=== 유형 4: ∫₀^∞ x^{a-1}/(1+x) dx ===")
for a in [0.25, 0.5, 0.75]:
    theory = np.pi / np.sin(np.pi * a)
    numerical, _ = quad(lambda x: x**(a-1)/(1+x), 0, np.inf)
    print(f"  a = {a}: π/sin(πa) = {theory:.6f}, 수치 = {numerical:.6f}")
```

---

## 5. 등각사상 (Conformal Mapping)

### 5.1 등각사상의 정의와 성질

$f'(z_0) \neq 0$인 해석함수 $w = f(z)$는 $z_0$ 근방에서 **등각(conformal)**이다:

- **각도 보존**: 두 곡선의 교차 각도가 사상 후에도 유지
- **라플라스 방정식 불변**: 조화함수가 사상 후에도 조화함수
- **경계조건 보존**: 물리적 경계 조건이 사상 후에도 유효

### 5.2 뫼비우스 변환

$$w = \frac{az + b}{cz + d}, \quad ad - bc \neq 0$$

**성질**: 원과 직선을 원과 직선으로 사상하며, 세 점으로 유일하게 결정된다.

```python
def mobius_transform(z, a, b, c, d):
    """뫼비우스 변환 w = (az+b)/(cz+d)"""
    return (a*z + b) / (c*z + d)

def plot_mobius(a, b, c, d, title="뫼비우스 변환"):
    """격자선의 뫼비우스 변환 시각화"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    t = np.linspace(-2, 2, 300)

    for x0 in np.linspace(-2, 2, 9):
        zv = x0 + 1j*t; wv = mobius_transform(zv, a, b, c, d)
        axes[0].plot(zv.real, zv.imag, 'b-', alpha=0.4, lw=0.7)
        m = np.abs(wv) < 10
        axes[1].plot(wv.real[m], wv.imag[m], 'b-', alpha=0.4, lw=0.7)

    for y0 in np.linspace(-2, 2, 9):
        zv = t + 1j*y0; wv = mobius_transform(zv, a, b, c, d)
        axes[0].plot(zv.real, zv.imag, 'r-', alpha=0.4, lw=0.7)
        m = np.abs(wv) < 10
        axes[1].plot(wv.real[m], wv.imag[m], 'r-', alpha=0.4, lw=0.7)

    for ax, lab in zip(axes, ['z-평면', 'w-평면']):
        ax.set_xlim(-4,4); ax.set_ylim(-4,4)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3); ax.set_title(lab)
    fig.suptitle(f'{title}: w=({a}z+{b})/({c}z+{d})')
    plt.tight_layout(); plt.show()

# 케일리 변환 (상반면→단위원): w = (z-i)/(z+i)
# plot_mobius(1, -1j, 1, 1j, "케일리 변환")
```

### 5.3 물리학 응용 (유체 역학, 전기장)

**복소 포텐셜**: 2차원 비압축성 비회전 흐름에서:

$$W(z) = \phi(x, y) + i\psi(x, y), \quad \frac{dW}{dz} = v_x - iv_y$$

$\phi$는 속도 포텐셜, $\psi$는 유선 함수이다.

| 흐름 | $W(z)$ | 물리적 의미 |
|------|--------|------------|
| 균일 흐름 | $Uz$ | 속도 $U$ |
| 소스/싱크 | $(Q/2\pi)\ln z$ | 강도 $Q$ |
| 와류 | $(-i\Gamma/2\pi)\ln z$ | 순환 $\Gamma$ |
| 이중극 | $\mu/z$ | 쌍극자 |
| 원주 주위 | $U(z + a^2/z)$ | 반지름 $a$ |

```python
def plot_flow(W_func, xlim=(-3,3), ylim=(-3,3), N=400, title="유선도"):
    """복소 포텐셜로부터 유선도와 등포텐셜선 시각화"""
    xv = np.linspace(*xlim, N); yv = np.linspace(*ylim, N)
    X, Y = np.meshgrid(xv, yv); Z = X + 1j*Y

    with np.errstate(divide='ignore', invalid='ignore'):
        W = W_func(Z)
    phi, psi = np.real(W), np.imag(W)
    mask = np.abs(W) > 50; phi[mask] = np.nan; psi[mask] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].contour(X, Y, psi, levels=30, colors='blue', linewidths=0.8)
    axes[0].set_title(f'{title} - 유선 (ψ=const)')
    axes[1].contour(X, Y, phi, levels=30, colors='red', linewidths=0.8)
    axes[1].set_title(f'{title} - 등포텐셜선 (φ=const)')
    for ax in axes:
        ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('y')
    plt.tight_layout(); plt.show()

# 순환 있는 원주 주위 흐름
U, a, Gamma = 1.0, 1.0, 2*np.pi
W_cyl = lambda z: U*(z + a**2/z) - 1j*Gamma/(2*np.pi)*np.log(z)
# plot_flow(W_cyl, title="순환 있는 원주 주위 흐름")

# 전기 쌍극자 (+q at z=d, -q at z=-d)
W_dipole = lambda z: -1/(2*np.pi) * (np.log(z-0.5) - np.log(z+0.5))
# plot_flow(W_dipole, title="전기 쌍극자 전위")
```

**주코프스키 변환과 양력**: $w = z + c^2/z$는 원을 에어포일로 사상한다. 쿠타-주코프스키 정리에 의한 단위 길이당 양력:

$$L = \rho U \Gamma$$

이 결과는 유수 정리로부터 우아하게 도출된다.

### 5.4 슈바르츠-크리스토펠 사상 (Schwarz-Christoffel Mapping)

**문제**: 상반면을 다각형 영역으로 사상하는 등각사상을 구하라.

**슈바르츠-크리스토펠 공식**: 상반면의 실수축 위의 점 $x_1, x_2, \ldots, x_n$이 다각형의 꼭짓점 $w_1, w_2, \ldots, w_n$으로 사상되고, 각 꼭짓점의 내각이 $\alpha_k \pi$이면:

$$\frac{dw}{dz} = A \prod_{k=1}^{n} (z - x_k)^{\alpha_k - 1}$$

여기서 $A$는 복소 상수이다.

**예**: 상반면 → 직사각형 사상은 타원 적분으로 표현되며, 정전기학에서 평행판 축전기의 가장자리 효과(fringing field) 계산에 사용된다.

---

## 6. 해석적 연속 (Analytic Continuation)

### 6.1 기본 개념

함수 $f(z)$가 영역 $D_1$에서 정의되어 있을 때, 더 큰 영역 $D_2 \supset D_1$에서 해석적인 함수 $g(z)$가 $D_1$에서 $f$와 일치하면, $g$를 $f$의 **해석적 연속**이라 한다.

**유일성**: 해석적 연속이 존재하면 유일하다 (항등 정리에 의해).

### 6.2 물리학 응용

**감마 함수**: 원래 $\Gamma(z) = \int_0^{\infty} t^{z-1} e^{-t} dt$는 $\text{Re}(z) > 0$에서만 정의되지만, 점화식 $\Gamma(z) = \Gamma(z+1)/z$를 이용하면 음의 정수를 제외한 전체 복소 평면으로 해석적 연속된다.

**리만 제타 함수**: $\zeta(s) = \sum_{n=1}^{\infty} n^{-s}$는 $\text{Re}(s) > 1$에서 수렴하지만, 해석적 연속을 통해 $s = 1$을 제외한 전체 복소 평면으로 확장된다. 이는 물리학에서 **제타 함수 정규화**(카시미르 효과 등)에 사용된다.

```python
# 감마 함수의 해석적 연속 시각화
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

x = np.linspace(-4.5, 5, 2000)
y = np.array([gamma(xi) if abs(xi - round(xi)) > 0.02 or xi > 0.5
              else np.nan for xi in x])

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=1.5)
plt.ylim(-10, 10)
for n in range(0, -5, -1):
    plt.axvline(x=n, color='red', linewidth=0.5, linestyle='--', alpha=0.5)

plt.xlabel('z')
plt.ylabel('Γ(z)')
plt.title('감마 함수의 해석적 연속 (음의 정수에서 극점)')
plt.grid(True, alpha=0.3)
plt.axhline(0, color='k', linewidth=0.5)
plt.tight_layout()
plt.show()
```

---

## 연습 문제

### 기본 문제

1. 다음 함수가 해석적인지 코시-리만 조건으로 판별하라:
   - (a) $f(z) = z^3$
   - (b) $f(z) = |z|^2$
   - (c) $f(z) = \bar{z}$
   - (d) $f(z) = e^{-z}\sin z$

2. $u(x, y) = x^3 - 3xy^2 + 2x$의 조화 켤레 $v(x, y)$를 구하라.

3. 다음 적분을 계산하라 ($C$: 원점 중심 반지름 2):
   - (a) $\oint_C e^z/z^2\, dz$
   - (b) $\oint_C \cos z/z^3\, dz$
   - (c) $\oint_C z^2/((z-1)(z+2))\, dz$

### 중급 문제

4. 유수 정리로 계산하라:
   - (a) $\displaystyle\int_0^{2\pi} \frac{d\theta}{5 + 4\cos\theta}$
   - (b) $\displaystyle\int_0^{\infty} \frac{x^2}{(x^2+1)(x^2+4)}\, dx$
   - (c) $\displaystyle\int_0^{\infty} \frac{\cos 3x}{x^2 + 1}\, dx$

5. $f(z) = z/((z-1)^2(z+2))$의 모든 특이점에서 유수를 구하고, $|z|=3$ 위의 적분값을 구하라.

### 심화 문제

6. **유체역학**: 소스($z=a$)와 싱크($z=-a$)가 있고 $x$축이 벽일 때, 영상법으로 상반면 유선 함수를 구하라.

7. **양자역학**: 그린 함수 $G(E) = (E - H + i\epsilon)^{-1}$로부터 상태밀도 $\rho(E) = -\text{Im}\, G(E)/\pi$를 유수 정리로 유도하라.

8. **풀이** (문제 4a):

```python
z = sp.Symbol('z')
# ∫₀²π dθ/(5+4cosθ) → ∮ 1/(i(2z²+5z+2)) dz
integrand = 1 / (sp.I * (2*z**2 + 5*z + 2))
poles = sp.solve(2*z**2 + 5*z + 2, z)
print(f"극점: {poles}")  # z=-1/2 (내부), z=-2 (외부)
res = sp.residue(integrand, z, sp.Rational(-1, 2))
print(f"적분값: {sp.simplify(2*sp.pi*sp.I*res)}")  # 2π/3
```

---

## 참고 자료

### 교재
- **Boas, M. L.** *Mathematical Methods in the Physical Sciences*, 3rd ed., Ch. 14
- **Arfken, Weber** *Mathematical Methods for Physicists*, Ch. 6-7
- **Churchill, Brown** *Complex Variables and Applications*

### 보충 자료
- **Needham, T.** *Visual Complex Analysis* - 기하학적 직관
- **Ablowitz, Fokas** *Complex Variables: Introduction and Applications* - 물리학 응용

### 핵심 공식 요약

| 공식 | 조건 |
|------|------|
| 코시-리만: $u_x = v_y$, $u_y = -v_x$ | 미분 가능 |
| 코시 정리: $\oint_C f\, dz = 0$ | 단순 연결 내 해석적 |
| 코시 공식: $f(z_0) = \frac{1}{2\pi i}\oint \frac{f}{z-z_0} dz$ | $f$ 해석적, $z_0$ 내부 |
| 유수 정리: $\oint f\, dz = 2\pi i \sum \text{Res}$ | 유한 특이점 |

---

## 다음 레슨

[15. 라플라스 변환 (Laplace Transform)](15_Laplace_Transform.md)에서는 라플라스 변환의 정의와 성질, 역변환, 미분방정식에의 응용을 다룹니다.
