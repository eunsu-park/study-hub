# 10. 다중 적분

## 학습 목표

- 반복 적분을 사용하여 직사각형 및 일반 영역에서의 이중 적분을 계산할 수 있다
- 푸비니 정리(Fubini's theorem)를 적용하여 적분 순서를 변경할 수 있다
- 이중 및 삼중 적분을 극좌표(polar), 원통좌표(cylindrical), 구면좌표(spherical)로 변환할 수 있다
- 야코비안(Jacobian)을 사용하여 다중 적분에서 일반적인 변수 변환을 수행할 수 있다
- 다중 적분을 이용하여 물리량(질량, 질량 중심, 관성 모멘트)을 계산할 수 있다

---

## 1. 직사각형에서의 이중 적분

### 1.1 정의

직사각형 $R = [a, b] \times [c, d]$에서 $f(x, y)$의 **이중 적분(double integral)**은 리만 합(Riemann sum)의 극한으로 정의된다:

$$\iint_R f(x, y)\, dA = \lim_{m,n \to \infty} \sum_{i=1}^{m}\sum_{j=1}^{n} f(x_i^*, y_j^*)\,\Delta A$$

여기서 $\Delta A = \Delta x\, \Delta y$는 각 작은 부분 직사각형의 넓이이다.

**기하학적 해석:** $f(x, y) \ge 0$일 때, 이중 적분은 곡면 $z = f(x, y)$ 아래이고 영역 $R$ 위에 있는 **부피**를 나타낸다.

**비유:** 단일 적분 $\int_a^b f(x)\,dx$는 가느다란 수직 띠의 넓이를 더하는 것이다. 이중 적분은 작은 수직 기둥의 부피를 더하는 것으로, 이를 2차원 영역 전체에 걸쳐 쌓아 올리는 것이다.

### 1.2 반복 적분

**푸비니 정리(Fubini's Theorem)** (직사각형의 경우): $f$가 $R = [a, b] \times [c, d]$에서 연속이면:

$$\iint_R f(x, y)\, dA = \int_a^b \left[\int_c^d f(x, y)\, dy\right] dx = \int_c^d \left[\int_a^b f(x, y)\, dx\right] dy$$

두 반복 적분은 같은 결과를 준다. 이는 2차원 적분 문제를 두 번의 연속적인 1차원 적분으로 변환한다.

**예제:**

$$\iint_R xy^2\, dA, \quad R = [0, 2] \times [1, 3]$$

$$= \int_0^2 \left[\int_1^3 xy^2\, dy\right] dx = \int_0^2 x\left[\frac{y^3}{3}\right]_1^3 dx = \int_0^2 x \cdot \frac{26}{3}\, dx = \frac{26}{3}\left[\frac{x^2}{2}\right]_0^2 = \frac{52}{3}$$

---

## 2. 일반 영역에서의 이중 적분

### 2.1 제1형과 제2형 영역

모든 영역이 직사각형인 것은 아니다. 편리한 두 가지 유형을 구분한다:

**제1형(Type I)** (곡선 $y = g_1(x)$과 $y = g_2(x)$로 둘러싸인 영역):

$$\iint_D f(x, y)\, dA = \int_a^b \int_{g_1(x)}^{g_2(x)} f(x, y)\, dy\, dx$$

**제2형(Type II)** (곡선 $x = h_1(y)$과 $x = h_2(y)$로 둘러싸인 영역):

$$\iint_D f(x, y)\, dA = \int_c^d \int_{h_1(y)}^{h_2(y)} f(x, y)\, dx\, dy$$

### 2.2 적분 순서 변경

때때로 한 가지 적분 순서는 닫힌 형태로 계산하기 어렵거나 불가능한 적분으로 이어지는 반면, 다른 순서는 간단한 경우가 있다.

**전략:** 영역 $D$를 스케치하고, 경계를 파악한 후, 다른 순서에 맞게 다시 표현한다.

**예제:** $\int_0^1 \int_x^1 e^{y^2}\, dy\, dx$를 계산하라.

내부 적분 $\int_x^1 e^{y^2}\,dy$는 초등 역도함수(elementary antiderivative)가 없다. 적분 순서를 바꾸자:

- 영역: $0 \le x \le y$, $0 \le y \le 1$ ($y = x$ 선 아래, $x$축 위의 삼각형)
- 순서 변경: $\int_0^1 \int_0^y e^{y^2}\, dx\, dy = \int_0^1 y\,e^{y^2}\, dy = \frac{1}{2}(e - 1)$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad

# --- Double integral over a triangular region ---
# Region: 0 <= x <= y, 0 <= y <= 1

# Numerical computation using scipy
result, error = dblquad(
    lambda x, y: np.exp(y**2),  # integrand f(x, y)
    0, 1,                        # y limits: 0 to 1
    lambda y: 0,                 # x lower limit: 0
    lambda y: y                  # x upper limit: y
)

print(f"Numerical result: {result:.8f}")
print(f"Exact value (e-1)/2: {(np.e - 1) / 2:.8f}")

# Visualize the region
fig, ax = plt.subplots(figsize=(6, 6))
# Fill the triangular region
triangle = plt.Polygon([(0, 0), (0, 1), (1, 1)], alpha=0.3, color='blue')
ax.add_patch(triangle)
ax.plot([0, 1], [0, 1], 'b-', linewidth=2, label='$y = x$')
ax.plot([0, 0], [0, 1], 'b-', linewidth=2)
ax.plot([0, 1], [1, 1], 'b-', linewidth=2)

# Annotate
ax.annotate('Type I: $x \\leq y \\leq 1$', xy=(0.15, 0.6), fontsize=12)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Region for Changing Order of Integration')
ax.set_aspect('equal')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

---

## 3. 극좌표에서의 이중 적분

영역 $D$가 **원형 대칭(circular symmetry)**을 가질 때, 극좌표(polar coordinates)는 적분을 극적으로 단순화한다.

### 3.1 변환

$x = r\cos\theta$, $y = r\sin\theta$를 대입하면:

$$\iint_D f(x, y)\, dA = \int_\alpha^\beta \int_{r_1(\theta)}^{r_2(\theta)} f(r\cos\theta,\, r\sin\theta)\, r\, dr\, d\theta$$

**추가 인수(factor) $r$**이 핵심이다. 이는 좌표 변환의 **야코비안(Jacobian)**에서 나온다:

$$dA = dx\,dy = r\,dr\,d\theta$$

**왜 $r$ 인수가 필요한가?** 극좌표에서의 작은 "상자"는 직사각형이 아니라 곡선 쐐기(curved wedge)이다. 그 넓이는 대략 $(dr)(r\,d\theta) = r\,dr\,d\theta$이다. 원점에서 멀수록 호의 길이가 더 크다.

**예제:** $D$가 원판 $x^2 + y^2 \le 4$일 때 $\iint_D e^{-(x^2 + y^2)}\, dA$를 계산하라.

극좌표에서: $x^2 + y^2 = r^2$이고, $D$는 $0 \le r \le 2$, $0 \le \theta \le 2\pi$가 된다:

$$\int_0^{2\pi}\int_0^2 e^{-r^2}\, r\, dr\, d\theta = 2\pi \int_0^2 r\,e^{-r^2}\, dr = 2\pi\left[-\frac{1}{2}e^{-r^2}\right]_0^2 = \pi(1 - e^{-4})$$

이 기법은 **가우스 적분(Gaussian integral)** $\int_{-\infty}^{\infty} e^{-x^2}\,dx = \sqrt{\pi}$를 계산하는 데 핵심적이다.

---

## 4. 삼중 적분

### 4.1 직교 좌표

3차원 공간의 영역 $E$에서 $f(x, y, z)$의 삼중 적분(triple integral):

$$\iiint_E f(x, y, z)\, dV$$

은 세 단계의 중첩을 가진 반복 적분으로 계산된다. 순서는 $E$가 어떻게 기술되느냐에 따라 달라진다.

**예제:** $x = 0$, $y = 0$, $z = 0$, $x + y + z = 1$로 둘러싸인 사면체에서 $f = xyz$를 적분하라:

$$\int_0^1 \int_0^{1-x} \int_0^{1-x-y} xyz\, dz\, dy\, dx$$

### 4.2 원통좌표

**원통좌표(cylindrical coordinates)** $(r, \theta, z)$는 극좌표에 수직축을 추가한 것이다:

$$x = r\cos\theta, \quad y = r\sin\theta, \quad z = z$$

$$dV = r\, dr\, d\theta\, dz$$

**적합한 경우:** 원형 단면을 가진 영역(원기둥, 원뿔 등).

### 4.3 구면좌표

**구면좌표(spherical coordinates)** $(\rho, \theta, \phi)$:

- $\rho$: 원점으로부터의 거리
- $\theta$: 방위각(azimuthal angle, 극좌표의 $\theta$와 동일)
- $\phi$: 극각(polar angle, 양의 $z$축에서 측정)

$$x = \rho\sin\phi\cos\theta, \quad y = \rho\sin\phi\sin\theta, \quad z = \rho\cos\phi$$

$$dV = \rho^2 \sin\phi\, d\rho\, d\phi\, d\theta$$

부피 요소 $\rho^2\sin\phi$는 구면좌표에서 작은 "상자"의 왜곡을 보정한다.

**적합한 경우:** 구 대칭(spherical symmetry)을 가진 영역(구, 구각 등).

**예제:** 반지름 $R$인 구의 부피:

$$V = \int_0^{2\pi}\int_0^{\pi}\int_0^R \rho^2\sin\phi\, d\rho\, d\phi\, d\theta = 2\pi \cdot 2 \cdot \frac{R^3}{3} = \frac{4}{3}\pi R^3$$

```python
import numpy as np
from scipy.integrate import tplquad, dblquad
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Triple integral: volume of intersection of sphere and cylinder ---
# Sphere: x^2 + y^2 + z^2 <= 4 (radius 2)
# Cylinder: x^2 + y^2 <= 1 (radius 1)
# We compute the volume inside both using cylindrical coordinates.

# In cylindrical: 0 <= r <= 1, 0 <= theta <= 2pi,
# -sqrt(4 - r^2) <= z <= sqrt(4 - r^2)

def integrand_cyl(z, r, theta):
    """Volume element in cylindrical: r * dz * dr * dtheta."""
    return r

volume, error = tplquad(
    integrand_cyl,
    0, 2 * np.pi,              # theta limits
    lambda theta: 0,            # r lower
    lambda theta: 1.0,          # r upper
    lambda theta, r: -np.sqrt(4 - r**2),  # z lower
    lambda theta, r: np.sqrt(4 - r**2)    # z upper
)

print(f"Volume of sphere-cylinder intersection: {volume:.6f}")
# Analytical: 2*pi * integral_0^1 of 2*sqrt(4-r^2) * r dr
# = 2*pi * [-2/3 * (4-r^2)^(3/2)]_0^1 = 2*pi * (16/3 - 2*sqrt(3)*3/3)
analytical = 2 * np.pi * (2/3) * (4**(3/2) - 3**(3/2))
print(f"Analytical value:                       {analytical:.6f}")

# --- Spherical coordinates: mass of a hemisphere with density rho(x,y,z) = z ---
# Region: x^2+y^2+z^2 <= R^2, z >= 0
# In spherical: rho*cos(phi) * rho^2*sin(phi)

R = 2.0

def integrand_sph(rho, phi, theta):
    """Density z = rho*cos(phi) times volume element rho^2*sin(phi)."""
    return rho * np.cos(phi) * rho**2 * np.sin(phi)

mass, error = tplquad(
    integrand_sph,
    0, 2 * np.pi,                 # theta
    lambda theta: 0,               # phi lower
    lambda theta: np.pi / 2,       # phi upper (hemisphere z >= 0)
    lambda theta, phi: 0,          # rho lower
    lambda theta, phi: R           # rho upper
)

print(f"\nMass of hemisphere (density=z, R={R}): {mass:.6f}")
print(f"Analytical (pi*R^4/4):                 {np.pi * R**4 / 4:.6f}")
```

---

## 5. 야코비안과 변수 변환

### 5.1 일반 공식

변환 $(x, y) = T(u, v)$, 즉 $x = x(u, v)$이고 $y = y(u, v)$에 대해:

$$\iint_R f(x, y)\, dx\, dy = \iint_S f(x(u,v),\, y(u,v))\, |J|\, du\, dv$$

여기서 $J$는 **야코비안 행렬식(Jacobian determinant)**이다:

$$J = \frac{\partial(x, y)}{\partial(u, v)} = \begin{vmatrix} \frac{\partial x}{\partial u} & \frac{\partial x}{\partial v} \\ \frac{\partial y}{\partial u} & \frac{\partial y}{\partial v} \end{vmatrix}$$

**야코비안은 변환에 의해 넓이가 얼마나 왜곡되는지를 측정한다.** $|J| = 2$이면, $(u, v)$ 공간에서의 단위 정사각형은 $(x, y)$ 공간에서 넓이 2인 영역으로 사상(mapping)된다.

### 5.2 특수 경우로서의 극좌표

극좌표 $x = r\cos\theta$, $y = r\sin\theta$에 대해:

$$J = \begin{vmatrix} \cos\theta & -r\sin\theta \\ \sin\theta & r\cos\theta \end{vmatrix} = r\cos^2\theta + r\sin^2\theta = r$$

이는 $dA = |J|\, dr\, d\theta = r\, dr\, d\theta$를 확인해 준다.

### 5.3 삼중 적분의 야코비안

변환 $(x, y, z) = T(u, v, w)$에 대해:

$$J = \frac{\partial(x, y, z)}{\partial(u, v, w)} = \begin{vmatrix} x_u & x_v & x_w \\ y_u & y_v & y_w \\ z_u & z_v & z_w \end{vmatrix}$$

**구면좌표:** $J = \rho^2\sin\phi$ (위에서 사용한 것과 같다).

**원통좌표:** $J = r$ (극좌표와 같은 인수).

```python
import numpy as np
from sympy import symbols, cos, sin, Matrix, simplify

# --- Verify Jacobians symbolically ---
r, theta, phi, rho = symbols('r theta phi rho', positive=True)

# Polar coordinates
x_pol = r * cos(theta)
y_pol = r * sin(theta)
J_polar = Matrix([
    [x_pol.diff(r), x_pol.diff(theta)],
    [y_pol.diff(r), y_pol.diff(theta)]
]).det()
print(f"Jacobian (polar): {simplify(J_polar)}")  # r

# Spherical coordinates
x_sph = rho * sin(phi) * cos(theta)
y_sph = rho * sin(phi) * sin(theta)
z_sph = rho * cos(phi)
J_sph = Matrix([
    [x_sph.diff(rho), x_sph.diff(phi), x_sph.diff(theta)],
    [y_sph.diff(rho), y_sph.diff(phi), y_sph.diff(theta)],
    [z_sph.diff(rho), z_sph.diff(phi), z_sph.diff(theta)]
]).det()
print(f"Jacobian (spherical): {simplify(J_sph)}")  # rho^2*sin(phi)

# Cylindrical coordinates
z_var = symbols('z')
x_cyl = r * cos(theta)
y_cyl = r * sin(theta)
z_cyl = z_var
J_cyl = Matrix([
    [x_cyl.diff(r), x_cyl.diff(theta), x_cyl.diff(z_var)],
    [y_cyl.diff(r), y_cyl.diff(theta), y_cyl.diff(z_var)],
    [z_cyl.diff(r), z_cyl.diff(theta), z_cyl.diff(z_var)]
]).det()
print(f"Jacobian (cylindrical): {simplify(J_cyl)}")  # r
```

---

## 6. 응용

### 6.1 질량과 질량 중심

밀도 $\rho(x, y)$를 가진 영역 $D$를 차지하는 라미나(lamina, 얇은 판)에 대해:

| 물리량 | 공식 |
|----------|---------|
| 질량(Mass) | $m = \iint_D \rho(x,y)\, dA$ |
| 질량 중심(Center of mass) $\bar{x}$ | $\bar{x} = \frac{1}{m}\iint_D x\,\rho(x,y)\, dA$ |
| 질량 중심(Center of mass) $\bar{y}$ | $\bar{y} = \frac{1}{m}\iint_D y\,\rho(x,y)\, dA$ |

### 6.2 관성 모멘트

| 물리량 | 공식 |
|----------|---------|
| $I_x$ ($x$축에 대한) | $I_x = \iint_D y^2 \rho(x,y)\, dA$ |
| $I_y$ ($y$축에 대한) | $I_y = \iint_D x^2 \rho(x,y)\, dA$ |
| $I_0$ (원점에 대한) | $I_0 = \iint_D (x^2 + y^2) \rho(x,y)\, dA = I_x + I_y$ |

이 공식들은 삼중 적분을 이용하여 3차원으로 확장된다.

```python
import numpy as np
from scipy.integrate import dblquad

# --- Center of mass of a semicircular lamina ---
# Region: x^2 + y^2 <= R^2, y >= 0
# Density: rho(x, y) = 1 (uniform)

R = 2.0

# Mass
mass, _ = dblquad(
    lambda y, x: 1.0,
    -R, R,
    lambda x: 0,
    lambda x: np.sqrt(max(R**2 - x**2, 0))
)

# First moment about x-axis: integral of y * rho dA
My, _ = dblquad(
    lambda y, x: y,
    -R, R,
    lambda x: 0,
    lambda x: np.sqrt(max(R**2 - x**2, 0))
)

y_bar = My / mass

print(f"Mass of semicircular lamina: {mass:.6f}")
print(f"Analytical mass (pi*R^2/2):  {np.pi * R**2 / 2:.6f}")
print(f"Center of mass y_bar:        {y_bar:.6f}")
print(f"Analytical (4R / 3pi):       {4 * R / (3 * np.pi):.6f}")
print(f"By symmetry, x_bar = 0")
```

---

## 7. 교차 참조

- **수리물리학(Mathematical Methods) 레슨 06**에서 곡선좌표계(curvilinear coordinates)인 원통좌표, 구면좌표, 일반 직교좌표를 스케일 인자(scale factor)와 미분 연산자를 포함하여 더 깊이 다룬다.
- **레슨 09 (다변수 함수)**에서 이 레슨의 변수 변환의 기반이 되는 편미분과 그래디언트(gradient)를 소개했다.
- **레슨 11 (벡터 미적분학)**에서는 적분을 선적분(line integral)과 면적분(surface integral)으로 확장한다.

---

## 연습 문제

**1.** $R = [0, 1] \times [0, 2]$에서 $\iint_R (x^2 + y)\, dA$를 계산하라. 두 가지 적분 순서로 계산하여 결과가 같은지 확인하라.

**2.** 적분 순서를 변경하여 $\int_0^1 \int_{\sqrt{y}}^1 \sin(x^2)\, dx\, dy$를 계산하라.
   (힌트: 먼저 영역을 스케치하라.)

**3.** 극좌표를 사용하여 $\iint_D (x^2 + y^2)^{3/2}\, dA$를 계산하라. 여기서 $D$는 고리 영역(annulus) $1 \le x^2 + y^2 \le 4$이다.

**4.** 포물면(paraboloid) $z = 4 - x^2 - y^2$의 위쪽과 평면 $z = 0$의 아래쪽으로 둘러싸인 입체의 부피를 구하라.
   - (a) 직교 좌표를 사용하여 적분을 세우고 계산하라.
   - (b) 극좌표를 사용하여 적분을 세우고 계산하라 (훨씬 쉬울 것이다).

**5.** 밀도가 $\rho(x, y, z) = z$인 반구(hemisphere) $x^2 + y^2 + z^2 \le R^2$, $z \ge 0$의 질량과 질량 중심을 구하라. 구면좌표를 사용하라. `scipy.integrate.tplquad`로 수치적으로 검증하라.

---

## 참고 자료

- **James Stewart**, *Calculus: Early Transcendentals*, 9th Edition, Chapters 15.1-15.9
- **Jerrold E. Marsden & Anthony Tromba**, *Vector Calculus*, 6th Edition, Chapter 5
- **SciPy Integration Documentation**: https://docs.scipy.org/doc/scipy/reference/integrate.html
- **Khan Academy**, "Double and Triple Integrals"

---

[이전: 다변수 함수](./09_Multivariable_Functions.md) | [다음: 벡터 미적분학](./11_Vector_Calculus.md)
