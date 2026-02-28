# 11. 벡터 미적분(Vector Calculus)

## 학습 목표

- 벡터장을 서술하고 발산(divergence)과 회전(curl)을 계산한다
- 곡선을 따라 스칼라 및 벡터장의 선적분(line integral)을 계산한다
- 벡터장이 보존적(conservative)인지 판별하고 퍼텐셜 함수(potential function)를 구한다
- 그린 정리(Green's theorem), 스토크스 정리(Stokes' theorem), 발산 정리(Divergence theorem)를 서술하고 적용한다
- Python을 사용하여 벡터 미적분의 기본 정리를 계산적으로 검증한다

---

## 1. 벡터장(Vector Fields)

**벡터장(vector field)**은 공간의 각 점에 벡터를 할당한다.

**2차원:** $\mathbf{F}(x, y) = P(x, y)\,\hat{\mathbf{i}} + Q(x, y)\,\hat{\mathbf{j}}$

**3차원:** $\mathbf{F}(x, y, z) = P\,\hat{\mathbf{i}} + Q\,\hat{\mathbf{j}} + R\,\hat{\mathbf{k}}$

**물리학에서의 예시:**
- **중력장:** $\mathbf{F} = -\frac{GMm}{r^3}\mathbf{r}$ (질량 쪽을 향한다)
- **속도장:** 유체의 각 점에서 $\mathbf{v}(x, y, z)$는 해당 위치의 속도를 나타낸다
- **전기장:** $\mathbf{E} = -\nabla V$ (전위의 기울기)

### 1.1 발산과 회전

**발산(divergence)**은 한 점에서 벡터장의 순수한 "유출량"을 측정한다:

$$\text{div}\,\mathbf{F} = \nabla \cdot \mathbf{F} = \frac{\partial P}{\partial x} + \frac{\partial Q}{\partial y} + \frac{\partial R}{\partial z}$$

**비유:** 벡터장을 유체 흐름이라고 상상해 보자. 양의 발산은 유체가 생성되고 있다는 것을 의미하고(수도꼭지와 같은 소스), 음의 발산은 유체가 흡수되고 있다는 것을 의미한다(배수구와 같은 싱크).

**회전(curl)**은 회전하는 경향을 측정한다:

$$\text{curl}\,\mathbf{F} = \nabla \times \mathbf{F} = \begin{vmatrix} \hat{\mathbf{i}} & \hat{\mathbf{j}} & \hat{\mathbf{k}} \\ \frac{\partial}{\partial x} & \frac{\partial}{\partial y} & \frac{\partial}{\partial z} \\ P & Q & R \end{vmatrix}$$

**비유:** 흐름 속에 작은 물레방아를 놓아보자. 회전은 회전의 축과 속도를 알려준다. 회전이 모든 곳에서 0이면, 흐름에는 "소용돌이"가 없다.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Vector field visualization ---
x = np.linspace(-2, 2, 15)
y = np.linspace(-2, 2, 15)
X, Y = np.meshgrid(x, y)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 1. Source field: F = (x, y) -- divergence = 2, curl = 0
axes[0].quiver(X, Y, X, Y, color='blue', alpha=0.7)
axes[0].set_title('Source: $\\mathbf{F} = (x, y)$\ndiv = 2, curl = 0')
axes[0].set_aspect('equal')
axes[0].grid(True, alpha=0.3)

# 2. Rotational field: F = (-y, x) -- divergence = 0, curl = 2k
axes[1].quiver(X, Y, -Y, X, color='red', alpha=0.7)
axes[1].set_title('Rotation: $\\mathbf{F} = (-y, x)$\ndiv = 0, curl = 2')
axes[1].set_aspect('equal')
axes[1].grid(True, alpha=0.3)

# 3. Saddle field: F = (x, -y) -- divergence = 0, curl = 0
axes[2].quiver(X, Y, X, -Y, color='green', alpha=0.7)
axes[2].set_title('Saddle: $\\mathbf{F} = (x, -y)$\ndiv = 0, curl = 0')
axes[2].set_aspect('equal')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 2. 선적분(Line Integrals)

### 2.1 스칼라 선적분

$\mathbf{r}(t) = (x(t), y(t))$ ($a \le t \le b$)로 매개변수화된 곡선 $C$를 따른 **스칼라 함수의 선적분** $f$:

$$\int_C f\, ds = \int_a^b f(\mathbf{r}(t))\,\|\mathbf{r}'(t)\|\, dt$$

여기서 $ds = \|\mathbf{r}'(t)\|\, dt$는 호 길이 요소이다.

**물리적 의미:** $f(x, y)$가 $C$ 모양의 와이어의 선밀도라면, $\int_C f\,ds$는 와이어의 전체 **질량**을 나타낸다.

### 2.2 벡터 선적분(일)

곡선 $C$를 따른 **벡터장** $\mathbf{F}$의 선적분:

$$\int_C \mathbf{F} \cdot d\mathbf{r} = \int_a^b \mathbf{F}(\mathbf{r}(t)) \cdot \mathbf{r}'(t)\, dt$$

**물리적 의미:** 이것은 입자가 $C$를 따라 이동할 때 힘 $\mathbf{F}$에 의해 수행된 **일(work)**이다. 내적 $\mathbf{F} \cdot d\mathbf{r}$는 운동 방향의 힘 성분을 추출한다.

성분 형태로:

$$\int_C \mathbf{F} \cdot d\mathbf{r} = \int_C P\,dx + Q\,dy$$

**예제:** $(0, 0)$에서 $(1, 1)$까지 포물선 $y = x^2$를 따라 $\mathbf{F} = (y, x)$가 한 일을 구하라.

매개변수화: $x = t$, $y = t^2$, $0 \le t \le 1$. 그러면 $dx = dt$, $dy = 2t\,dt$.

$$W = \int_0^1 (t^2\,dt + t\cdot 2t\,dt) = \int_0^1 3t^2\,dt = 1$$

```python
import numpy as np
from scipy.integrate import quad

# --- Line integral computation ---
# F = (y, x) along y = x^2 from (0,0) to (1,1)

def work_integrand(t):
    """Integrand for F dot r'(t) where r(t) = (t, t^2)."""
    x, y = t, t**2
    dx_dt, dy_dt = 1.0, 2 * t
    P, Q = y, x   # F = (y, x)
    return P * dx_dt + Q * dy_dt  # F dot r'

work, _ = quad(work_integrand, 0, 1)
print(f"Work along parabola: {work:.6f}")  # Should be 1.0

# Compare: work along straight line y = x from (0,0) to (1,1)
def work_line(t):
    """Integrand along the straight line r(t) = (t, t)."""
    x, y = t, t
    P, Q = y, x
    return P * 1.0 + Q * 1.0  # dx/dt = dy/dt = 1

work2, _ = quad(work_line, 0, 1)
print(f"Work along straight line: {work2:.6f}")  # Also 1.0!
# Both give 1.0 -- this field is conservative (spoiler for Section 3)
```

---

## 3. 보존장과 퍼텐셜 함수(Conservative Fields and Potential Functions)

### 3.1 정의

벡터장 $\mathbf{F}$가 **보존적(conservative)**이라 함은 스칼라 함수 $\varphi$ (**퍼텐셜 함수**)가 존재하여 다음을 만족하는 것이다:

$$\mathbf{F} = \nabla\varphi$$

동치적으로, $P = \partial\varphi/\partial x$, $Q = \partial\varphi/\partial y$.

### 3.2 경로 독립성(Path Independence)

**선적분의 기본 정리(Fundamental Theorem for Line Integrals)**에 따르면: $\mathbf{F} = \nabla\varphi$이면,

$$\int_C \mathbf{F} \cdot d\mathbf{r} = \varphi(\mathbf{r}(b)) - \varphi(\mathbf{r}(a))$$

적분은 경로가 아닌 **양 끝점**에만 의존한다 -- 미적분학의 기본 정리와 마찬가지이다.

**결과:** 임의의 **폐곡선** $C$ (루프)에 대해, $\oint_C \mathbf{F} \cdot d\mathbf{r} = 0$.

### 3.3 보존성 판정

2차원에서, $\mathbf{F} = (P, Q)$가 단순 연결 영역에서 보존적일 필요충분조건은:

$$\frac{\partial P}{\partial y} = \frac{\partial Q}{\partial x}$$

3차원에서, $\mathbf{F} = (P, Q, R)$가 보존적일 필요충분조건은 $\nabla \times \mathbf{F} = \mathbf{0}$이다.

### 3.4 퍼텐셜 함수 구하기

$\mathbf{F} = (2xy + z, x^2, x)$가 주어졌을 때:

1. $P$를 $x$에 대해 적분: $\varphi = x^2 y + xz + g(y, z)$
2. $y$에 대해 미분: $\varphi_y = x^2 + g_y = Q = x^2$, 따라서 $g_y = 0$
3. $z$에 대해 미분: $\varphi_z = x + g_z = R = x$, 따라서 $g_z = 0$
4. 따라서 $\varphi = x^2 y + xz + C$

```python
from sympy import symbols, diff, integrate, simplify

x, y, z = symbols('x y z')

# --- Test conservativeness and find potential function ---
P = 2*x*y + z
Q = x**2
R = x

# Check curl = 0
curl_x = diff(R, y) - diff(Q, z)  # dR/dy - dQ/dz
curl_y = diff(P, z) - diff(R, x)  # dP/dz - dR/dx
curl_z = diff(Q, x) - diff(P, y)  # dQ/dx - dP/dy

print(f"curl F = ({curl_x}, {curl_y}, {curl_z})")
# Should be (0, 0, 0) for conservative field

# Find potential: integrate P w.r.t. x
phi = integrate(P, x)  # x^2*y + x*z
print(f"After integrating P w.r.t. x: phi = {phi} + g(y,z)")

# Check: d(phi)/dy should equal Q
phi_y = diff(phi, y)
g_y = simplify(Q - phi_y)
print(f"g_y = Q - phi_y = {g_y}")  # 0

# Check: d(phi)/dz should equal R
phi_z = diff(phi, z)
g_z = simplify(R - phi_z)
print(f"g_z = R - phi_z = {g_z}")  # 0

print(f"\nPotential function: phi = {phi}")
```

---

## 4. 그린 정리(Green's Theorem)

**그린 정리(Green's theorem)**는 폐곡선을 따른 **선적분**과 둘러싸인 영역에 대한 **이중적분**을 연결한다.

### 4.1 정리

$C$가 영역 $D$를 둘러싸는 단순 폐곡선(반시계 방향)이고 $P, Q$가 연속 편도함수를 가진다면:

$$\oint_C P\,dx + Q\,dy = \iint_D \left(\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}\right) dA$$

**좌변:** $(P, Q)$가 루프를 따라 한 일.
**우변:** "미시적 순환"(2D 회전)의 영역 적분.

**직관:** 그린 정리는 경계를 따른 전체 순환이 내부의 모든 작은 회전의 합과 같다고 말한다. 이것은 스토크스 정리의 2차원 특수한 경우이다.

### 4.2 응용

**넓이 공식:** $P = -y/2$, $Q = x/2$로 설정하면:

$$A = \frac{1}{2}\oint_C x\,dy - y\,dx$$

이것이 플래니미터가 넓이를 측정하는 방식이며 다각형 넓이의 **신발끈 공식(shoelace formula)**이 작동하는 원리이다.

```python
import numpy as np
from scipy.integrate import dblquad, quad

# --- Green's theorem verification ---
# F = (-y^2, x^2), C = unit circle (counterclockwise)
# Line integral: integral_C -y^2 dx + x^2 dy

# Parameterize: x = cos(t), y = sin(t), 0 <= t <= 2pi
def line_integrand(t):
    """Integrand for the line integral around the unit circle."""
    x, y = np.cos(t), np.sin(t)
    dx_dt, dy_dt = -np.sin(t), np.cos(t)
    P = -y**2
    Q = x**2
    return P * dx_dt + Q * dy_dt

line_result, _ = quad(line_integrand, 0, 2 * np.pi)
print(f"Line integral:   {line_result:.6f}")

# Double integral: integral_D (dQ/dx - dP/dy) dA
# dQ/dx = 2x, dP/dy = -2y
# Integrand = 2x + 2y = 2(x + y)
# Over the unit disk

area_result, _ = dblquad(
    lambda y, x: 2 * (x + y),
    -1, 1,
    lambda x: -np.sqrt(1 - x**2),
    lambda x: np.sqrt(1 - x**2)
)
print(f"Double integral: {area_result:.6f}")
print(f"Green's theorem verified: {np.isclose(line_result, area_result)}")
```

---

## 5. 면적분(Surface Integrals)

### 5.1 매개변수 곡면

곡면 $S$는 $(u, v) \in D$에 대해 $\mathbf{r}(u, v) = (x(u,v), y(u,v), z(u,v))$로 매개변수화할 수 있다.

**면적 요소**는:

$$dS = \left\|\frac{\partial\mathbf{r}}{\partial u} \times \frac{\partial\mathbf{r}}{\partial v}\right\| du\, dv$$

외적 $\mathbf{r}_u \times \mathbf{r}_v$는 법선 벡터를 주고, 그 크기는 매개변수화의 "늘림 인자"를 제공한다.

### 5.2 플럭스 적분(Flux Integrals)

외향 법선 $\hat{\mathbf{n}}$을 가진 곡면 $S$를 통과하는 벡터장 $\mathbf{F}$의 **플럭스(flux)**:

$$\iint_S \mathbf{F} \cdot d\mathbf{S} = \iint_S \mathbf{F} \cdot \hat{\mathbf{n}}\, dS = \iint_D \mathbf{F} \cdot (\mathbf{r}_u \times \mathbf{r}_v)\, du\, dv$$

**물리적 의미:** 플럭스는 장의 곡면 통과 "유량"을 측정한다. $\mathbf{F}$가 속도장이면, 플럭스는 단위 시간당 $S$를 통과하는 유체의 부피이다.

---

## 6. 스토크스 정리(Stokes' Theorem)

**스토크스 정리(Stokes' theorem)**는 그린 정리를 3차원 곡면으로 일반화한다.

### 6.1 정리

$S$가 단순 폐곡선 $C$로 둘러싸인 방향이 있는 곡면이면(호환되는 방향):

$$\oint_C \mathbf{F} \cdot d\mathbf{r} = \iint_S (\nabla \times \mathbf{F}) \cdot d\mathbf{S}$$

**좌변:** 경계 곡선을 따른 $\mathbf{F}$의 순환.
**우변:** 곡면을 통과하는 회전의 플럭스.

**직관:** 가장자리를 따른 전체 순환은 곡면 전체에 걸친 모든 미시적 회전의 합과 같다 -- 곡면이 3차원에서 휘어져 있더라도.

**특수한 경우:** $S$가 ($xy$-평면의) 평면일 때, 스토크스 정리는 그린 정리로 환원된다.

```python
import numpy as np
from scipy.integrate import dblquad, quad

# --- Stokes' theorem verification ---
# F = (z, x, y)
# Surface S: portion of z = 1 - x^2 - y^2 above z = 0
# Boundary C: the circle x^2 + y^2 = 1, z = 0

# 1. Line integral around C: x = cos(t), y = sin(t), z = 0
def stokes_line(t):
    x, y, z = np.cos(t), np.sin(t), 0.0
    dx, dy, dz = -np.sin(t), np.cos(t), 0.0
    # F = (z, x, y) = (0, cos(t), sin(t))
    return z * dx + x * dy + y * dz

line_val, _ = quad(stokes_line, 0, 2 * np.pi)
print(f"Line integral (circulation): {line_val:.6f}")

# 2. Surface integral of curl(F) dot dS
# curl(F) = (dR/dy - dQ/dz, dP/dz - dR/dx, dQ/dx - dP/dy)
#          = (1 - 1, 1 - 0, 1 - 0) = (0, 1, 1)
# Wait, let's recalculate:
# P=z, Q=x, R=y
# curl_x = dR/dy - dQ/dz = 1 - 0 = 1
# curl_y = dP/dz - dR/dx = 1 - 0 = 1
# curl_z = dQ/dx - dP/dy = 1 - 0 = 1
# curl(F) = (1, 1, 1)

# Surface z = 1 - x^2 - y^2 above z=0, parameterized by (x, y)
# Normal: (-dz/dx, -dz/dy, 1) = (2x, 2y, 1) (outward/upward)
# curl(F) dot n = 1*(2x) + 1*(2y) + 1*(1) = 2x + 2y + 1

def surface_integrand(y, x):
    """curl(F) dot (outward normal) for z = 1 - x^2 - y^2."""
    return 2 * x + 2 * y + 1

surface_val, _ = dblquad(
    surface_integrand,
    -1, 1,
    lambda x: -np.sqrt(max(1 - x**2, 0)),
    lambda x: np.sqrt(max(1 - x**2, 0))
)
print(f"Surface integral (curl flux): {surface_val:.6f}")
print(f"Stokes' theorem verified: {np.isclose(line_val, surface_val)}")
# Both should equal pi
print(f"Analytical value: pi = {np.pi:.6f}")
```

---

## 7. 발산 정리(가우스 정리)(The Divergence Theorem)

### 7.1 정리

$E$가 폐곡면 $S$ (외향 법선)로 둘러싸인 고체 영역이면:

$$\oiint_S \mathbf{F} \cdot d\mathbf{S} = \iiint_E \nabla \cdot \mathbf{F}\, dV$$

**좌변:** 곡면을 통과하는 $\mathbf{F}$의 전체 플럭스.
**우변:** 체적 내부의 전체 발산(소스 세기).

**직관:** 닫힌 상자 안에 소스가 있으면, 벽을 통한 순유출량은 내부의 전체 소스 세기와 같다. 이것은 미적분학 기본 정리의 3차원 유사체이다.

### 7.2 응용

**정전기학의 가우스 법칙:**

$$\oiint_S \mathbf{E} \cdot d\mathbf{S} = \frac{Q_{\text{enc}}}{\varepsilon_0}$$

이것은 곡면을 통과하는 전기 플럭스를 둘러싸인 전하와 연결한다 -- $\nabla \cdot \mathbf{E} = \rho / \varepsilon_0$에 발산 정리를 적용한 직접적인 결과이다.

### 7.3 전체 그림

세 가지 주요 정리는 하나의 계층 구조를 이룬다:

| 정리 | 차원 | 관계 |
|---------|-----------|---------|
| 미적분학의 기본 정리 | 1D | $\int_a^b f'(x)\,dx = f(b) - f(a)$ |
| 그린 / 스토크스 | 2D/3D | 경계 적분 = 내부 회전 적분 |
| 발산 정리 | 3D | 경계 플럭스 = 내부 발산 적분 |

세 가지 모두 **일반화된 스토크스 정리**: $\int_{\partial\Omega} \omega = \int_\Omega d\omega$의 사례이다. 패턴은 항상: **영역에 걸친 도함수의 적분은 경계에 걸친 원래 함수의 적분과 같다**.

```python
import numpy as np
from scipy.integrate import tplquad, dblquad

# --- Divergence theorem verification ---
# F = (x^2, y^2, z^2)
# Region E: unit sphere x^2 + y^2 + z^2 <= 1
# div(F) = 2x + 2y + 2z

# Volume integral of div(F) over the unit sphere (spherical coords)
def div_integrand(rho, phi, theta):
    """div(F) * rho^2 * sin(phi) in spherical coordinates."""
    x = rho * np.sin(phi) * np.cos(theta)
    y = rho * np.sin(phi) * np.sin(theta)
    z = rho * np.cos(phi)
    div_F = 2 * x + 2 * y + 2 * z
    return div_F * rho**2 * np.sin(phi)

vol_integral, _ = tplquad(
    div_integrand,
    0, 2 * np.pi,                  # theta
    lambda t: 0, lambda t: np.pi,  # phi
    lambda t, p: 0, lambda t, p: 1 # rho
)
print(f"Volume integral of div(F): {vol_integral:.6f}")

# By symmetry, integral of 2x, 2y, 2z over the sphere are each 0
# (odd functions over symmetric domain)
# So the divergence theorem gives flux = 0
print(f"Expected (by symmetry): 0.000000")
print(f"Verified: {np.isclose(vol_integral, 0, atol=1e-10)}")

# Let's try F = (x, y, z) instead: div(F) = 3
# Volume of unit sphere = 4*pi/3
# Volume integral = 3 * 4*pi/3 = 4*pi
def div_integrand_2(rho, phi, theta):
    """div(F) = 3, times volume element."""
    return 3 * rho**2 * np.sin(phi)

vol2, _ = tplquad(
    div_integrand_2,
    0, 2 * np.pi,
    lambda t: 0, lambda t: np.pi,
    lambda t, p: 0, lambda t, p: 1
)
print(f"\nFor F=(x,y,z): volume integral of div(F) = {vol2:.6f}")
print(f"Analytical (4*pi): {4 * np.pi:.6f}")
```

---

## 8. 교차 참조

- **Mathematical Methods 레슨 05**는 다양한 좌표계에서의 나블라 연산자, 헬름홀츠 분해(Helmholtz decomposition), 적분 정리의 증명을 포함한 벡터 해석의 포괄적인 다룸을 제공한다.
- **전자기학(Electrodynamics) 레슨 01-06**은 맥스웰 방정식에 그린, 스토크스, 발산 정리를 광범위하게 적용한다.
- **레슨 10 (중적분)**은 이 레슨 전체에서 사용되는 이중 및 삼중 적분 기법을 다룬다.

---

## 연습 문제

**1.** $\mathbf{F} = (x^2 y, xy^2)$일 때, 꼭짓점이 $(0,0)$, $(1,0)$, $(0,1)$인 삼각형을 따른 선적분과 대응하는 이중적분을 모두 계산하여 그린 정리를 검증하라.

**2.** $\mathbf{F} = (2xy + z^2, x^2 + 2yz, 2xz + y^2)$가 보존적인지 판별하라. 보존적이라면, 퍼텐셜 함수 $\varphi$를 구하고 $(0,0,0)$에서 $(1,2,3)$까지 $\int_C \mathbf{F} \cdot d\mathbf{r}$을 계산하라.

**3.** 발산 정리를 사용하여 $\mathbf{F} = (x^3, y^3, z^3)$이고 $S$가 구 $x^2 + y^2 + z^2 = 4$일 때 $\oiint_S \mathbf{F} \cdot d\mathbf{S}$를 구하라.

**4.** 스토크스 정리를 사용하여 $\mathbf{F} = (y^2, z^2, x^2)$이고 $C$가 $(1,0,0)$, $(0,1,0)$, $(0,0,1)$로 이루어진 삼각형(위에서 볼 때 반시계 방향)일 때 $\oint_C \mathbf{F} \cdot d\mathbf{r}$을 구하라.

**5.** $\mathbf{F} = \mathbf{r}/\|\mathbf{r}\|^3$의 플럭스가 원점을 포함하는 임의의 폐곡면을 통해 $4\pi$임을 보여라. (힌트: 이것은 점전하에 대한 가우스 법칙이다.) 곡면이 원점을 포함하지 않으면 어떻게 되는가?

---

## 참고 자료

- **James Stewart**, *Calculus: Early Transcendentals*, 9th Edition, Chapter 16
- **Jerrold E. Marsden & Anthony Tromba**, *Vector Calculus*, 6th Edition, Chapters 7-8
- **H.M. Schey**, *Div, Grad, Curl, and All That*, 4th Edition (탁월한 직관적 설명)
- **3Blue1Brown**, "Divergence and Curl" (시각적 직관)

---

[이전: 중적분](./10_Multiple_Integrals.md) | [다음: 1계 상미분방정식](./12_First_Order_ODE.md)
