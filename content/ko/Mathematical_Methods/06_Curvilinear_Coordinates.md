# 06. 곡선좌표계와 다중적분 (Curvilinear Coordinates and Multiple Integrals)

## 학습 목표

- **다중적분**의 정의와 계산법을 이해하고, 적분 순서 변경 기법을 익힌다
- **야코비안(Jacobian)**을 이용한 좌표 변환에서의 적분 변수 치환을 수행할 수 있다
- **원통 좌표계(cylindrical)**와 **구면 좌표계(spherical)**의 좌표 변환, 체적/면적 요소, 미분 연산자를 유도하고 활용한다
- **일반 곡선좌표계**에서 스케일 인자(scale factor)와 미분 연산자의 일반적 표현을 이해한다
- 물리학 문제(관성 모멘트, 전기장, 중력장)에 적절한 좌표계를 선택하여 적용할 수 있다

---

## 1. 이중적분과 삼중적분

### 1.1 이중적분의 정의와 계산

이중적분(double integral)은 2차원 영역 $R$ 위에서 함수 $f(x, y)$를 적분하는 것이다:

$$\iint_R f(x, y) \, dA = \lim_{n \to \infty} \sum_{i=1}^{n} f(x_i, y_i) \, \Delta A_i$$

직교 좌표에서 면적 요소는 $dA = dx \, dy$이므로, 반복적분(iterated integral)으로 계산한다:

$$\iint_R f(x, y) \, dA = \int_a^b \left[ \int_{g_1(x)}^{g_2(x)} f(x, y) \, dy \right] dx$$

여기서 $a \le x \le b$이고, 각 $x$에 대해 $g_1(x) \le y \le g_2(x)$이다.

**예제**: 삼각형 영역 $0 \le x \le 1$, $0 \le y \le x$ 위에서 $f(x, y) = x + y$의 이중적분

$$\int_0^1 \int_0^x (x + y) \, dy \, dx = \int_0^1 \left[ xy + \frac{y^2}{2} \right]_0^x dx = \int_0^1 \frac{3x^2}{2} \, dx = \frac{1}{2}$$

```python
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# --- 이중적분 수치 계산 ---
# f(x, y) = x + y, 영역: 0 <= x <= 1, 0 <= y <= x
f = lambda y, x: x + y
result, error = integrate.dblquad(f, 0, 1, lambda x: 0, lambda x: x)
print(f"이중적분 결과: {result:.6f} (오차: {error:.2e})")
# 출력: 이중적분 결과: 0.500000 (오차: 5.55e-15)

# --- 적분 영역 시각화 ---
fig, ax = plt.subplots(1, 1, figsize=(5, 4))
triangle = Polygon([[0, 0], [1, 0], [1, 1]], alpha=0.3, color='steelblue')
ax.add_patch(triangle)
ax.set_xlim(-0.1, 1.3)
ax.set_ylim(-0.1, 1.3)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('적분 영역: 0 ≤ y ≤ x, 0 ≤ x ≤ 1')
ax.set_aspect('equal')
ax.plot([0, 1], [0, 1], 'r-', linewidth=2, label='y = x')
ax.legend()
plt.tight_layout()
plt.savefig('double_integral_region.png', dpi=150)
plt.show()
```

### 1.2 삼중적분

삼중적분(triple integral)은 3차원 영역 $V$ 위에서 함수를 적분한다:

$$\iiint_V f(x, y, z) \, dV = \int_a^b \int_{g_1(x)}^{g_2(x)} \int_{h_1(x,y)}^{h_2(x,y)} f(x, y, z) \, dz \, dy \, dx$$

직교 좌표에서 체적 요소는 $dV = dx \, dy \, dz$이다.

**예제**: 단위 구 $x^2 + y^2 + z^2 \le 1$ 내부에서 $f = 1$의 삼중적분 (구의 부피)

```python
# 단위 구의 부피: 삼중적분 (직교 좌표)
def sphere_volume_cartesian():
    f = lambda z, y, x: 1.0
    x_lo, x_hi = -1, 1
    y_lo = lambda x: -np.sqrt(1 - x**2)
    y_hi = lambda x:  np.sqrt(1 - x**2)
    z_lo = lambda x, y: -np.sqrt(max(0, 1 - x**2 - y**2))
    z_hi = lambda x, y:  np.sqrt(max(0, 1 - x**2 - y**2))

    result, error = integrate.tplquad(f, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)
    return result

V = sphere_volume_cartesian()
print(f"단위 구의 부피 (수치적분): {V:.6f}")
print(f"해석적 결과 (4π/3):       {4*np.pi/3:.6f}")
```

### 1.3 적분 순서 변경

이중적분에서 적분 순서를 바꾸면 계산이 크게 간단해지는 경우가 많다. 핵심은 **적분 영역의 경계를 새로운 순서에 맞게 다시 기술**하는 것이다.

**예제**: $\int_0^1 \int_y^1 e^{x^2} \, dx \, dy$

$x$ 방향 적분이 먼저인데, $e^{x^2}$의 부정적분은 초등함수로 표현 불가. 순서를 변경하면:

- 원래 영역: $0 \le y \le 1$, $y \le x \le 1$ (삼각형: $0 \le y \le x \le 1$)
- 변경 후: $0 \le x \le 1$, $0 \le y \le x$

$$\int_0^1 \int_0^x e^{x^2} \, dy \, dx = \int_0^1 x \, e^{x^2} \, dx = \frac{1}{2}(e - 1)$$

```python
import sympy as sp

x, y = sp.symbols('x y', positive=True)

# 순서 변경 후 적분
inner = sp.integrate(sp.exp(x**2), (y, 0, x))   # ∫₀ˣ e^{x²} dy = x·e^{x²}
result = sp.integrate(inner, (x, 0, 1))           # ∫₀¹ x·e^{x²} dx
print(f"적분 순서 변경 후 결과: {result}")
# 출력: -1/2 + E/2  즉, (e-1)/2
print(f"수치값: {float(result):.6f}")
```

---

## 2. 좌표 변환과 야코비안

### 2.1 야코비안 (Jacobian) 행렬식

좌표 변환 $(x, y) \to (u, v)$에서, $x = x(u, v)$, $y = y(u, v)$일 때, **야코비안(Jacobian)**은:

$$J = \frac{\partial(x, y)}{\partial(u, v)} = \begin{vmatrix} \dfrac{\partial x}{\partial u} & \dfrac{\partial x}{\partial v} \\[8pt] \dfrac{\partial y}{\partial u} & \dfrac{\partial y}{\partial v} \end{vmatrix}$$

야코비안은 좌표 변환에 의한 **면적(또는 체적)의 확대/축소 비율**을 나타낸다.

3차원에서는:

$$J = \frac{\partial(x, y, z)}{\partial(u, v, w)} = \begin{vmatrix} \dfrac{\partial x}{\partial u} & \dfrac{\partial x}{\partial v} & \dfrac{\partial x}{\partial w} \\[6pt] \dfrac{\partial y}{\partial u} & \dfrac{\partial y}{\partial v} & \dfrac{\partial y}{\partial w} \\[6pt] \dfrac{\partial z}{\partial u} & \dfrac{\partial z}{\partial v} & \dfrac{\partial z}{\partial w} \end{vmatrix}$$

### 2.2 일반적인 좌표 변환 공식

좌표 변환 $(u, v) \to (x, y)$에 의한 이중적분의 변환 공식:

$$\iint_R f(x, y) \, dx \, dy = \iint_{R'} f(x(u,v), y(u,v)) \, |J| \, du \, dv$$

여기서 $|J|$는 야코비안의 **절댓값**이다.

### 2.3 변수 치환을 이용한 적분

```python
import sympy as sp

u, v = sp.symbols('u v', positive=True)

# 예: 극좌표 변환 x = r cos(θ), y = r sin(θ)
r, theta = sp.symbols('r theta', positive=True)
x_expr = r * sp.cos(theta)
y_expr = r * sp.sin(theta)

# 야코비안 계산
J = sp.Matrix([
    [sp.diff(x_expr, r), sp.diff(x_expr, theta)],
    [sp.diff(y_expr, r), sp.diff(y_expr, theta)]
])
jacobian_det = J.det().simplify()
print(f"극좌표 야코비안: J = {jacobian_det}")
# 출력: 극좌표 야코비안: J = r

# 야코비안 행렬 출력
print(f"\n야코비안 행렬:")
sp.pprint(J)

# --- 타원 좌표 변환 예제 ---
# x = a·u·cos(v), y = b·u·sin(v)
a, b = sp.symbols('a b', positive=True)
x_ellip = a * u * sp.cos(v)
y_ellip = b * u * sp.sin(v)

J_ellip = sp.Matrix([
    [sp.diff(x_ellip, u), sp.diff(x_ellip, v)],
    [sp.diff(y_ellip, u), sp.diff(y_ellip, v)]
])
det_ellip = J_ellip.det().simplify()
print(f"\n타원 좌표 야코비안: J = {det_ellip}")
# 출력: 타원 좌표 야코비안: J = a*b*u
```

---

## 3. 원통 좌표계 (Cylindrical Coordinates)

### 3.1 좌표 정의와 변환

원통 좌표(cylindrical coordinates) $(\rho, \phi, z)$는 2차원 극좌표에 $z$축을 추가한 것이다:

$$x = \rho \cos\phi, \quad y = \rho \sin\phi, \quad z = z$$

역변환:

$$\rho = \sqrt{x^2 + y^2}, \quad \phi = \arctan\left(\frac{y}{x}\right), \quad z = z$$

범위: $\rho \ge 0$, $0 \le \phi < 2\pi$, $-\infty < z < \infty$

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def cylindrical_to_cartesian(rho, phi, z):
    """원통 좌표 → 직교 좌표 변환"""
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y, z

def cartesian_to_cylindrical(x, y, z):
    """직교 좌표 → 원통 좌표 변환"""
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi, z

# --- 원통 좌표계 시각화 ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# ρ = const 곡면 (원통)
phi_grid = np.linspace(0, 2*np.pi, 50)
z_grid = np.linspace(-2, 2, 20)
PHI, Z = np.meshgrid(phi_grid, z_grid)
for rho_val in [0.5, 1.0, 1.5]:
    X = rho_val * np.cos(PHI)
    Y = rho_val * np.sin(PHI)
    ax.plot_surface(X, Y, Z, alpha=0.15, color='blue')

# φ = const 곡면 (반평면)
rho_grid = np.linspace(0, 2, 20)
RHO, Z2 = np.meshgrid(rho_grid, z_grid)
for phi_val in [0, np.pi/3, 2*np.pi/3, np.pi]:
    X2 = RHO * np.cos(phi_val)
    Y2 = RHO * np.sin(phi_val)
    ax.plot_surface(X2, Y2, Z2, alpha=0.1, color='red')

# z = const 곡면 (수평면)
RHO3, PHI3 = np.meshgrid(rho_grid, phi_grid)
X3 = RHO3 * np.cos(PHI3)
Y3 = RHO3 * np.sin(PHI3)
for z_val in [-1, 0, 1]:
    Z3 = np.full_like(X3, z_val)
    ax.plot_surface(X3, Y3, Z3, alpha=0.1, color='green')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('원통 좌표계 등위면: ρ=const(파랑), φ=const(빨강), z=const(초록)')
plt.tight_layout()
plt.savefig('cylindrical_coords.png', dpi=150)
plt.show()
```

### 3.2 체적 요소와 면적 요소

원통 좌표에서의 야코비안:

$$J = \frac{\partial(x, y, z)}{\partial(\rho, \phi, z)} = \begin{vmatrix} \cos\phi & -\rho\sin\phi & 0 \\ \sin\phi & \rho\cos\phi & 0 \\ 0 & 0 & 1 \end{vmatrix} = \rho$$

따라서 **체적 요소**는:

$$dV = \rho \, d\rho \, d\phi \, dz$$

**면적 요소**:
- $\rho = \text{const}$ 면 (원통 옆면): $dA = \rho \, d\phi \, dz$
- $z = \text{const}$ 면 (수평면): $dA = \rho \, d\rho \, d\phi$
- $\phi = \text{const}$ 면 (반평면): $dA = d\rho \, dz$

**예제**: 반지름 $R$, 높이 $H$인 원통의 부피

$$V = \int_0^H \int_0^{2\pi} \int_0^R \rho \, d\rho \, d\phi \, dz = \pi R^2 H$$

```python
import sympy as sp

rho, phi, z = sp.symbols('rho phi z', positive=True)
R, H = sp.symbols('R H', positive=True)

# 원통의 부피
V = sp.integrate(rho, (rho, 0, R), (phi, 0, 2*sp.pi), (z, 0, H))
print(f"원통의 부피: V = {V}")
# 출력: V = π·R²·H

# 야코비안 검증
x_cyl = rho * sp.cos(phi)
y_cyl = rho * sp.sin(phi)
z_cyl = z

J_cyl = sp.Matrix([
    [sp.diff(x_cyl, rho), sp.diff(x_cyl, phi), sp.diff(x_cyl, z)],
    [sp.diff(y_cyl, rho), sp.diff(y_cyl, phi), sp.diff(y_cyl, z)],
    [sp.diff(z_cyl, rho), sp.diff(z_cyl, phi), sp.diff(z_cyl, z)]
])
print(f"야코비안 det = {J_cyl.det().simplify()}")
# 출력: 야코비안 det = rho
```

### 3.3 기울기, 발산, 회전의 원통 좌표 표현

원통 좌표에서 단위 벡터 $\hat{\boldsymbol{\rho}}$, $\hat{\boldsymbol{\phi}}$, $\hat{\mathbf{z}}$를 사용한다.

**기울기 (Gradient)**:

$$\nabla f = \frac{\partial f}{\partial \rho}\hat{\boldsymbol{\rho}} + \frac{1}{\rho}\frac{\partial f}{\partial \phi}\hat{\boldsymbol{\phi}} + \frac{\partial f}{\partial z}\hat{\mathbf{z}}$$

**발산 (Divergence)**:

$$\nabla \cdot \mathbf{F} = \frac{1}{\rho}\frac{\partial}{\partial \rho}(\rho F_\rho) + \frac{1}{\rho}\frac{\partial F_\phi}{\partial \phi} + \frac{\partial F_z}{\partial z}$$

**회전 (Curl)**:

$$\nabla \times \mathbf{F} = \left(\frac{1}{\rho}\frac{\partial F_z}{\partial \phi} - \frac{\partial F_\phi}{\partial z}\right)\hat{\boldsymbol{\rho}} + \left(\frac{\partial F_\rho}{\partial z} - \frac{\partial F_z}{\partial \rho}\right)\hat{\boldsymbol{\phi}} + \frac{1}{\rho}\left(\frac{\partial}{\partial \rho}(\rho F_\phi) - \frac{\partial F_\rho}{\partial \phi}\right)\hat{\mathbf{z}}$$

**라플라시안 (Laplacian)**:

$$\nabla^2 f = \frac{1}{\rho}\frac{\partial}{\partial \rho}\left(\rho \frac{\partial f}{\partial \rho}\right) + \frac{1}{\rho^2}\frac{\partial^2 f}{\partial \phi^2} + \frac{\partial^2 f}{\partial z^2}$$

### 3.4 응용 예제

**예제**: 무한히 긴 직선 도선(전류 $I$)의 자기장

앙페르 법칙에 의해 $\mathbf{B} = \frac{\mu_0 I}{2\pi \rho}\hat{\boldsymbol{\phi}}$이다. 발산과 회전을 검증한다.

```python
import sympy as sp
from sympy.vector import CoordSys3D

# SymPy 벡터 시스템은 직교 좌표 기반이므로, 직접 원통 좌표 미분 연산 구현
rho, phi, z = sp.symbols('rho phi z', positive=True)
mu_0, I = sp.symbols('mu_0 I', positive=True)

# B = (μ₀I / 2πρ) φ̂  →  B_rho = 0, B_phi = μ₀I/(2πρ), B_z = 0
B_rho = 0
B_phi = mu_0 * I / (2 * sp.pi * rho)
B_z = 0

# 발산 (원통 좌표)
div_B = (1/rho) * sp.diff(rho * B_rho, rho) + \
        (1/rho) * sp.diff(B_phi, phi) + \
        sp.diff(B_z, z)
print(f"∇·B = {sp.simplify(div_B)}")
# 출력: ∇·B = 0  (맥스웰 방정식 ∇·B = 0 성립)

# 회전 (원통 좌표) - z 성분만 계산 (나머지는 0)
curl_B_z = (1/rho) * (sp.diff(rho * B_phi, rho) - sp.diff(B_rho, phi))
print(f"(∇×B)_z = {sp.simplify(curl_B_z)}")
# ρ ≠ 0에서 0 (도선 외부에서 전류 없음)
```

---

## 4. 구면 좌표계 (Spherical Coordinates)

### 4.1 좌표 정의와 변환

구면 좌표(spherical coordinates) $(r, \theta, \phi)$:

$$x = r \sin\theta \cos\phi, \quad y = r \sin\theta \sin\phi, \quad z = r \cos\theta$$

역변환:

$$r = \sqrt{x^2 + y^2 + z^2}, \quad \theta = \arccos\left(\frac{z}{r}\right), \quad \phi = \arctan\left(\frac{y}{x}\right)$$

범위: $r \ge 0$, $0 \le \theta \le \pi$ (극각, polar angle), $0 \le \phi < 2\pi$ (방위각, azimuthal angle)

> **주의**: 물리학에서는 $\theta$를 극각(polar angle), $\phi$를 방위각(azimuthal angle)으로 쓰는 관례를 따른다. 수학에서는 반대 관례를 사용하기도 한다.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def spherical_to_cartesian(r, theta, phi):
    """구면 좌표 → 직교 좌표 변환"""
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def cartesian_to_spherical(x, y, z):
    """직교 좌표 → 구면 좌표 변환"""
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / np.where(r > 0, r, 1))
    phi = np.arctan2(y, x)
    return r, theta, phi

# --- 구면 좌표계 등위면 시각화 ---
fig = plt.figure(figsize=(12, 5))

# (a) r = const (구면)
ax1 = fig.add_subplot(131, projection='3d')
theta_g = np.linspace(0, np.pi, 30)
phi_g = np.linspace(0, 2*np.pi, 30)
THETA, PHI = np.meshgrid(theta_g, phi_g)
for r_val in [0.5, 1.0, 1.5]:
    X = r_val * np.sin(THETA) * np.cos(PHI)
    Y = r_val * np.sin(THETA) * np.sin(PHI)
    Z = r_val * np.cos(THETA)
    ax1.plot_surface(X, Y, Z, alpha=0.2, color='blue')
ax1.set_title('r = const (구면)')

# (b) θ = const (원뿔)
ax2 = fig.add_subplot(132, projection='3d')
r_g = np.linspace(0, 2, 20)
R_grid, PHI2 = np.meshgrid(r_g, phi_g)
for theta_val in [np.pi/6, np.pi/3, np.pi/2]:
    X2 = R_grid * np.sin(theta_val) * np.cos(PHI2)
    Y2 = R_grid * np.sin(theta_val) * np.sin(PHI2)
    Z2 = R_grid * np.cos(theta_val)
    ax2.plot_surface(X2, Y2, Z2, alpha=0.2, color='red')
ax2.set_title('θ = const (원뿔)')

# (c) φ = const (반평면)
ax3 = fig.add_subplot(133, projection='3d')
R3, THETA3 = np.meshgrid(r_g, theta_g)
for phi_val in [0, np.pi/3, 2*np.pi/3, np.pi]:
    X3 = R3 * np.sin(THETA3) * np.cos(phi_val)
    Y3 = R3 * np.sin(THETA3) * np.sin(phi_val)
    Z3 = R3 * np.cos(THETA3)
    ax3.plot_surface(X3, Y3, Z3, alpha=0.2, color='green')
ax3.set_title('φ = const (반평면)')

for ax in [ax1, ax2, ax3]:
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

plt.suptitle('구면 좌표계 등위면', fontsize=14)
plt.tight_layout()
plt.savefig('spherical_coords.png', dpi=150)
plt.show()
```

### 4.2 체적 요소와 면적 요소

구면 좌표의 야코비안:

$$J = \frac{\partial(x, y, z)}{\partial(r, \theta, \phi)} = r^2 \sin\theta$$

> **유도**: $3 \times 3$ 야코비안 행렬의 행렬식을 직접 계산하거나, 기하학적으로 미소 체적 요소 $dr \cdot (r \, d\theta) \cdot (r\sin\theta \, d\phi)$로 이해할 수 있다.

따라서 **체적 요소**:

$$dV = r^2 \sin\theta \, dr \, d\theta \, d\phi$$

**면적 요소**:
- $r = \text{const}$ 면 (구면): $dA = r^2 \sin\theta \, d\theta \, d\phi$
- $\theta = \text{const}$ 면 (원뿔면): $dA = r \sin\theta \, dr \, d\phi$
- $\phi = \text{const}$ 면 (반평면): $dA = r \, dr \, d\theta$

**예제**: 구의 부피와 표면적

```python
import sympy as sp

r, theta, phi = sp.symbols('r theta phi', positive=True)
R = sp.Symbol('R', positive=True)

# 야코비안 검증
x_sph = r * sp.sin(theta) * sp.cos(phi)
y_sph = r * sp.sin(theta) * sp.sin(phi)
z_sph = r * sp.cos(theta)

J_sph = sp.Matrix([
    [sp.diff(x_sph, r), sp.diff(x_sph, theta), sp.diff(x_sph, phi)],
    [sp.diff(y_sph, r), sp.diff(y_sph, theta), sp.diff(y_sph, phi)],
    [sp.diff(z_sph, r), sp.diff(z_sph, theta), sp.diff(z_sph, phi)]
])
det_J = sp.trigsimp(J_sph.det())
print(f"구면 좌표 야코비안: {det_J}")
# 출력: r**2*sin(theta)

# 구의 부피
V = sp.integrate(r**2 * sp.sin(theta), (r, 0, R), (theta, 0, sp.pi), (phi, 0, 2*sp.pi))
print(f"구의 부피: V = {V}")
# 출력: 4*pi*R**3/3

# 구의 표면적 (r = R 고정)
S = sp.integrate(R**2 * sp.sin(theta), (theta, 0, sp.pi), (phi, 0, 2*sp.pi))
print(f"구의 표면적: S = {S}")
# 출력: 4*pi*R**2
```

### 4.3 기울기, 발산, 회전의 구면 좌표 표현

**기울기 (Gradient)**:

$$\nabla f = \frac{\partial f}{\partial r}\hat{\mathbf{r}} + \frac{1}{r}\frac{\partial f}{\partial \theta}\hat{\boldsymbol{\theta}} + \frac{1}{r\sin\theta}\frac{\partial f}{\partial \phi}\hat{\boldsymbol{\phi}}$$

**발산 (Divergence)**:

$$\nabla \cdot \mathbf{F} = \frac{1}{r^2}\frac{\partial}{\partial r}(r^2 F_r) + \frac{1}{r\sin\theta}\frac{\partial}{\partial \theta}(\sin\theta \, F_\theta) + \frac{1}{r\sin\theta}\frac{\partial F_\phi}{\partial \phi}$$

**회전 (Curl)**:

$$\nabla \times \mathbf{F} = \frac{1}{r\sin\theta}\left[\frac{\partial}{\partial \theta}(\sin\theta \, F_\phi) - \frac{\partial F_\theta}{\partial \phi}\right]\hat{\mathbf{r}} + \frac{1}{r}\left[\frac{1}{\sin\theta}\frac{\partial F_r}{\partial \phi} - \frac{\partial}{\partial r}(r F_\phi)\right]\hat{\boldsymbol{\theta}} + \frac{1}{r}\left[\frac{\partial}{\partial r}(r F_\theta) - \frac{\partial F_r}{\partial \theta}\right]\hat{\boldsymbol{\phi}}$$

**라플라시안 (Laplacian)**:

$$\nabla^2 f = \frac{1}{r^2}\frac{\partial}{\partial r}\left(r^2 \frac{\partial f}{\partial r}\right) + \frac{1}{r^2 \sin\theta}\frac{\partial}{\partial \theta}\left(\sin\theta \frac{\partial f}{\partial \theta}\right) + \frac{1}{r^2 \sin^2\theta}\frac{\partial^2 f}{\partial \phi^2}$$

### 4.4 응용 예제

**예제**: 쿨롱 포텐셜 $\Phi = \frac{q}{4\pi\epsilon_0 r}$의 라플라시안

```python
import sympy as sp

r, theta, phi = sp.symbols('r theta phi', positive=True)
q, eps0 = sp.symbols('q epsilon_0', positive=True)

# 쿨롱 포텐셜
Phi = q / (4 * sp.pi * eps0 * r)

# 구면 좌표 라플라시안 계산 (r > 0 영역)
laplacian_Phi = (1/r**2) * sp.diff(r**2 * sp.diff(Phi, r), r) + \
                (1/(r**2 * sp.sin(theta))) * sp.diff(sp.sin(theta) * sp.diff(Phi, theta), theta) + \
                (1/(r**2 * sp.sin(theta)**2)) * sp.diff(Phi, phi, 2)

result = sp.simplify(laplacian_Phi)
print(f"∇²Φ = {result}  (r > 0에서)")
# 출력: 0  (원점 제외 영역에서 라플라스 방정식 만족)
# 원점에서는 ∇²(1/r) = -4πδ(r) (디랙 델타 함수)

# --- 구면 좌표에서 발산 정리 검증 ---
# E = -∇Φ = (q/4πε₀r²) r̂
E_r = q / (4 * sp.pi * eps0 * r**2)

# 발산 (구 대칭이므로 r 성분만)
div_E = (1/r**2) * sp.diff(r**2 * E_r, r)
print(f"∇·E = {sp.simplify(div_E)}  (r > 0)")
# 출력: 0 (전하가 없는 영역)
```

**예제**: 구면 좌표에서의 입체각(solid angle) 적분

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 입체각 요소 dΩ = sin(θ) dθ dφ
# 전체 구의 입체각: ∫∫ sin(θ) dθ dφ = 4π 스테라디안

# 원뿔(θ ≤ α)이 차지하는 입체각
alpha_values = np.linspace(0, np.pi, 100)
solid_angles = 2 * np.pi * (1 - np.cos(alpha_values))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# (a) 입체각 vs 반꼭지각
axes[0].plot(np.degrees(alpha_values), solid_angles, 'b-', linewidth=2)
axes[0].axhline(y=4*np.pi, color='r', linestyle='--', label='전체 구 = 4π sr')
axes[0].axhline(y=2*np.pi, color='g', linestyle='--', label='반구 = 2π sr')
axes[0].set_xlabel('반꼭지각 α (도)')
axes[0].set_ylabel('입체각 Ω (sr)')
axes[0].set_title('원뿔의 입체각')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# (b) 구면 위 면적 요소 시각화
ax2 = fig.add_subplot(122, projection='3d')
theta_g = np.linspace(0, np.pi, 40)
phi_g = np.linspace(0, 2*np.pi, 40)
THETA, PHI = np.meshgrid(theta_g, phi_g)
X = np.sin(THETA) * np.cos(PHI)
Y = np.sin(THETA) * np.sin(PHI)
Z = np.cos(THETA)

# sin(θ)에 비례하는 색으로 면적 요소 밀도 표현
colors = plt.cm.viridis(np.sin(THETA) / np.sin(THETA).max())
ax2.plot_surface(X, Y, Z, facecolors=colors, alpha=0.7)
ax2.set_title('면적 요소 밀도 ∝ sinθ\n(적도 부근이 밝음)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')

plt.tight_layout()
plt.savefig('solid_angle.png', dpi=150)
plt.show()
```

---

## 5. 일반 곡선좌표계

### 5.1 스케일 인자 (Scale Factors)

일반 곡선좌표 $(q_1, q_2, q_3)$와 직교 좌표 $(x, y, z)$ 사이의 변환이 $\mathbf{r} = \mathbf{r}(q_1, q_2, q_3)$로 주어질 때, **스케일 인자(scale factor)** $h_i$는:

$$h_i = \left|\frac{\partial \mathbf{r}}{\partial q_i}\right|$$

스케일 인자는 좌표 $q_i$가 단위만큼 변할 때 실제 거리가 얼마나 변하는지를 나타낸다.

**미소 변위**:

$$d\mathbf{r} = h_1 \, dq_1 \, \hat{\mathbf{e}}_1 + h_2 \, dq_2 \, \hat{\mathbf{e}}_2 + h_3 \, dq_3 \, \hat{\mathbf{e}}_3$$

**체적 요소**:

$$dV = h_1 h_2 h_3 \, dq_1 \, dq_2 \, dq_3$$

| 좌표계 | $(q_1, q_2, q_3)$ | $(h_1, h_2, h_3)$ |
|--------|-------------------|-------------------|
| 직교 | $(x, y, z)$ | $(1, 1, 1)$ |
| 원통 | $(\rho, \phi, z)$ | $(1, \rho, 1)$ |
| 구면 | $(r, \theta, \phi)$ | $(1, r, r\sin\theta)$ |

```python
import sympy as sp

# --- 스케일 인자 계산 함수 ---
def compute_scale_factors(coords, transform):
    """
    곡선좌표계의 스케일 인자를 계산한다.

    Parameters:
        coords: 곡선좌표 변수 리스트 [q1, q2, q3]
        transform: 직교좌표 [x(q), y(q), z(q)]

    Returns:
        스케일 인자 [h1, h2, h3]
    """
    r = sp.Matrix(transform)
    scale_factors = []
    for q in coords:
        dr_dq = r.diff(q)
        h = sp.sqrt(dr_dq.dot(dr_dq)).simplify()
        # trigsimp로 삼각함수 정리
        h = sp.trigsimp(h)
        scale_factors.append(h)
    return scale_factors

# 원통 좌표
rho, phi, z = sp.symbols('rho phi z', positive=True)
h_cyl = compute_scale_factors(
    [rho, phi, z],
    [rho * sp.cos(phi), rho * sp.sin(phi), z]
)
print(f"원통 좌표 스케일 인자: h_ρ={h_cyl[0]}, h_φ={h_cyl[1]}, h_z={h_cyl[2]}")
# 출력: h_ρ=1, h_φ=rho, h_z=1

# 구면 좌표
r, theta = sp.symbols('r theta', positive=True)
h_sph = compute_scale_factors(
    [r, theta, phi],
    [r * sp.sin(theta) * sp.cos(phi),
     r * sp.sin(theta) * sp.sin(phi),
     r * sp.cos(theta)]
)
print(f"구면 좌표 스케일 인자: h_r={h_sph[0]}, h_θ={h_sph[1]}, h_φ={h_sph[2]}")
# 출력: h_r=1, h_θ=r, h_φ=r*sin(theta)

# 포물선 좌표 (parabolic cylindrical): x = (u²-v²)/2, y = uv, z = z
u, v = sp.symbols('u v', positive=True)
h_parab = compute_scale_factors(
    [u, v, z],
    [(u**2 - v**2)/2, u*v, z]
)
print(f"포물선 원통 좌표 스케일 인자: h_u={h_parab[0]}, h_v={h_parab[1]}, h_z={h_parab[2]}")
# 출력: h_u=sqrt(u²+v²), h_v=sqrt(u²+v²), h_z=1
```

### 5.2 일반적인 미분 연산자

직교 곡선좌표계(orthogonal curvilinear coordinates)에서 미분 연산자의 일반 표현:

**기울기**:

$$\nabla f = \frac{1}{h_1}\frac{\partial f}{\partial q_1}\hat{\mathbf{e}}_1 + \frac{1}{h_2}\frac{\partial f}{\partial q_2}\hat{\mathbf{e}}_2 + \frac{1}{h_3}\frac{\partial f}{\partial q_3}\hat{\mathbf{e}}_3$$

**발산**:

$$\nabla \cdot \mathbf{F} = \frac{1}{h_1 h_2 h_3}\left[\frac{\partial}{\partial q_1}(h_2 h_3 F_1) + \frac{\partial}{\partial q_2}(h_1 h_3 F_2) + \frac{\partial}{\partial q_3}(h_1 h_2 F_3)\right]$$

**회전**:

$$\nabla \times \mathbf{F} = \frac{1}{h_1 h_2 h_3}\begin{vmatrix} h_1\hat{\mathbf{e}}_1 & h_2\hat{\mathbf{e}}_2 & h_3\hat{\mathbf{e}}_3 \\[4pt] \dfrac{\partial}{\partial q_1} & \dfrac{\partial}{\partial q_2} & \dfrac{\partial}{\partial q_3} \\[4pt] h_1 F_1 & h_2 F_2 & h_3 F_3 \end{vmatrix}$$

**라플라시안**:

$$\nabla^2 f = \frac{1}{h_1 h_2 h_3}\left[\frac{\partial}{\partial q_1}\left(\frac{h_2 h_3}{h_1}\frac{\partial f}{\partial q_1}\right) + \frac{\partial}{\partial q_2}\left(\frac{h_1 h_3}{h_2}\frac{\partial f}{\partial q_2}\right) + \frac{\partial}{\partial q_3}\left(\frac{h_1 h_2}{h_3}\frac{\partial f}{\partial q_3}\right)\right]$$

```python
import sympy as sp

def laplacian_curvilinear(f, coords, scale_factors):
    """
    일반 직교 곡선좌표계에서 라플라시안을 계산한다.

    Parameters:
        f: 스칼라 함수
        coords: [q1, q2, q3]
        scale_factors: [h1, h2, h3]
    """
    q1, q2, q3 = coords
    h1, h2, h3 = scale_factors

    term1 = sp.diff((h2*h3/h1) * sp.diff(f, q1), q1)
    term2 = sp.diff((h1*h3/h2) * sp.diff(f, q2), q2)
    term3 = sp.diff((h1*h2/h3) * sp.diff(f, q3), q3)

    return sp.simplify((term1 + term2 + term3) / (h1 * h2 * h3))

# 구면 좌표에서 라플라시안 검증: f = 1/r
r, theta, phi = sp.symbols('r theta phi', positive=True)
f = 1 / r

lap_f = laplacian_curvilinear(
    f,
    [r, theta, phi],
    [1, r, r * sp.sin(theta)]
)
print(f"∇²(1/r) = {lap_f}  (r > 0에서)")
# 출력: 0 (원점 제외)

# 구면 좌표에서 라플라시안 검증: f = r² cos(θ) = rz → ∇²f = 0 (조화함수)
f2 = r**2 * sp.cos(theta)
lap_f2 = laplacian_curvilinear(
    f2,
    [r, theta, phi],
    [1, r, r * sp.sin(theta)]
)
print(f"∇²(r²cosθ) = {lap_f2}")
# ∇²(r²cosθ) = 0 이어야 한다 (단, 사실 이것은 조화함수가 아님)
# 올바른 조화함수: r cos(θ) = z
f3 = r * sp.cos(theta)
lap_f3 = laplacian_curvilinear(
    f3,
    [r, theta, phi],
    [1, r, r * sp.sin(theta)]
)
print(f"∇²(r·cosθ) = {lap_f3}")
# 출력: 0 (z = r·cosθ 는 조화함수)
```

### 5.3 직교 곡선좌표계

직교 곡선좌표계(orthogonal curvilinear coordinates)의 조건: 좌표 곡면들이 서로 직교하는 것이다. 수학적으로:

$$\frac{\partial \mathbf{r}}{\partial q_i} \cdot \frac{\partial \mathbf{r}}{\partial q_j} = 0 \quad (i \ne j)$$

이 조건이 만족되면 계량 텐서(metric tensor)가 대각 행렬이 되어 미분 연산자가 크게 간단해진다.

**주요 직교 좌표계 목록**:

| 좌표계 | 변수 | 스케일 인자 | 주요 용도 |
|--------|------|------------|----------|
| 직교 (Cartesian) | $(x,y,z)$ | $(1,1,1)$ | 일반적 |
| 원통 (Cylindrical) | $(\rho,\phi,z)$ | $(1,\rho,1)$ | 축대칭 문제 |
| 구면 (Spherical) | $(r,\theta,\phi)$ | $(1,r,r\sin\theta)$ | 구대칭 문제 |
| 타원 원통 (Elliptic cyl.) | $(u,v,z)$ | $(\cdot,\cdot,1)$ | 타원 경계 |
| 포물선 원통 (Parabolic cyl.) | $(u,v,z)$ | $(\sqrt{u^2+v^2},\sqrt{u^2+v^2},1)$ | 포물면 경계 |
| 장구 좌표 (Prolate sph.) | $(\xi,\eta,\phi)$ | 복잡 | 이심률 있는 문제 |

---

## 6. 물리학 응용

### 6.1 관성 모멘트 계산

물체의 관성 모멘트(moment of inertia)는 회전축에 대한 질량 분포를 특성짓는 양이다:

$$I = \iiint_V \rho(\mathbf{r}) \, d^2 \, dV$$

여기서 $d$는 회전축까지의 거리, $\rho(\mathbf{r})$는 질량 밀도이다.

**예제 1**: 균일 밀도 $\rho_0$인 반지름 $R$의 속이 찬 구의 관성 모멘트 ($z$축 기준)

구면 좌표에서 $z$축까지의 거리: $d = r\sin\theta$

$$I_z = \rho_0 \int_0^{2\pi} \int_0^{\pi} \int_0^R (r\sin\theta)^2 \cdot r^2 \sin\theta \, dr \, d\theta \, d\phi$$

```python
import sympy as sp

r, theta, phi = sp.symbols('r theta phi', positive=True)
R, rho0, M = sp.symbols('R rho_0 M', positive=True)

# z축 기준 관성 모멘트
integrand = rho0 * (r * sp.sin(theta))**2 * r**2 * sp.sin(theta)
I_z = sp.integrate(integrand, (r, 0, R), (theta, 0, sp.pi), (phi, 0, 2*sp.pi))
I_z_simplified = sp.simplify(I_z)
print(f"I_z = {I_z_simplified}")

# 총 질량 M = (4/3)πR³ρ₀ 로 치환
M_expr = sp.Rational(4, 3) * sp.pi * R**3 * rho0
I_z_M = I_z_simplified.subs(rho0, M / (sp.Rational(4, 3) * sp.pi * R**3))
I_z_final = sp.simplify(I_z_M)
print(f"I_z = {I_z_final}")
# 출력: 2*M*R²/5

print(f"\n--- 다양한 도형의 관성 모멘트 ---")

# 원통 (반지름 R, 높이 H, z축 기준)
rho_cyl, z_cyl = sp.symbols('rho z', positive=True)
H = sp.Symbol('H', positive=True)
I_cylinder = sp.integrate(
    rho0 * rho_cyl**2 * rho_cyl,  # d² * dV/dρdφdz 에서 d=ρ
    (rho_cyl, 0, R), (phi, 0, 2*sp.pi), (z_cyl, 0, H)
)
M_cyl = sp.pi * R**2 * H * rho0
I_cyl_M = sp.simplify(I_cylinder.subs(rho0, M / (sp.pi * R**2 * H)))
print(f"원통 (z축): I = {I_cyl_M}")
# 출력: M*R²/2

# 속이 빈 구껍질 (반지름 R, z축 기준)
# 면적분: I = ∫ (R sinθ)² σ R² sinθ dθ dφ
sigma = sp.Symbol('sigma', positive=True)
I_shell = sp.integrate(
    sigma * (R * sp.sin(theta))**2 * R**2 * sp.sin(theta),
    (theta, 0, sp.pi), (phi, 0, 2*sp.pi)
)
M_shell = 4 * sp.pi * R**2 * sigma
I_shell_M = sp.simplify(I_shell.subs(sigma, M / (4 * sp.pi * R**2)))
print(f"구껍질 (z축): I = {I_shell_M}")
# 출력: 2*M*R²/3
```

### 6.2 구 대칭 전하 분포의 전기장

가우스 법칙: $\oint \mathbf{E} \cdot d\mathbf{A} = \frac{Q_{\text{enc}}}{\epsilon_0}$

구 대칭 전하 분포에서는 구면 좌표가 자연스러운 선택이다.

**예제**: 균일 전하 밀도 $\rho_e$를 가진 반지름 $R$인 구의 전기장

- $r > R$: $E(r) = \frac{Q}{4\pi\epsilon_0 r^2}$ (점전하와 동일)
- $r < R$: $E(r) = \frac{\rho_e \, r}{3\epsilon_0} = \frac{Q r}{4\pi\epsilon_0 R^3}$

```python
import numpy as np
import matplotlib.pyplot as plt

# 균일 전하 구의 전기장
Q = 1.0        # 총 전하 (임의 단위)
R = 1.0        # 구의 반지름
eps0 = 1.0     # 진공 유전율 (단위계 단순화)

r = np.linspace(0.01, 3.0, 500)

# 전기장 크기
E = np.where(
    r < R,
    Q * r / (4 * np.pi * eps0 * R**3),      # 내부: E ∝ r
    Q / (4 * np.pi * eps0 * r**2)            # 외부: E ∝ 1/r²
)

# 전위
V_inside = Q / (8 * np.pi * eps0 * R) * (3 - r**2 / R**2)
V_outside = Q / (4 * np.pi * eps0 * r)
V = np.where(r < R, V_inside, V_outside)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# (a) 전기장
axes[0].plot(r/R, E * (4*np.pi*eps0*R**2/Q), 'b-', linewidth=2)
axes[0].axvline(x=1, color='gray', linestyle='--', alpha=0.5, label='r = R')
axes[0].set_xlabel('r / R')
axes[0].set_ylabel('E × (4πε₀R²/Q)')
axes[0].set_title('균일 전하 구의 전기장')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].annotate('E ∝ r', xy=(0.5, 0.3), fontsize=12, color='blue')
axes[0].annotate('E ∝ 1/r²', xy=(1.8, 0.25), fontsize=12, color='blue')

# (b) 전위
axes[1].plot(r/R, V * (4*np.pi*eps0*R/Q), 'r-', linewidth=2)
axes[1].axvline(x=1, color='gray', linestyle='--', alpha=0.5, label='r = R')
axes[1].set_xlabel('r / R')
axes[1].set_ylabel('V × (4πε₀R/Q)')
axes[1].set_title('균일 전하 구의 전위')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('uniform_sphere_field.png', dpi=150)
plt.show()

# --- 가우스 법칙 수치 검증 ---
from scipy import integrate

rho_e = 3 * Q / (4 * np.pi * R**3)  # 균일 전하 밀도

# 반지름 r_test인 구면을 통한 전기 선속
r_test_values = [0.3, 0.7, 1.0, 1.5, 2.0]

print("가우스 법칙 검증:")
print(f"{'r/R':>6s}  {'Q_enc':>10s}  {'Φ = Q_enc/ε₀':>14s}  {'E·4πr²':>10s}")
print("-" * 48)

for r_test in r_test_values:
    if r_test <= R:
        Q_enc = rho_e * (4/3) * np.pi * r_test**3
    else:
        Q_enc = Q
    flux = Q_enc / eps0
    E_at_r = Q_enc / (4 * np.pi * eps0 * r_test**2)
    E_times_area = E_at_r * 4 * np.pi * r_test**2
    print(f"{r_test/R:6.2f}  {Q_enc:10.4f}  {flux:14.4f}  {E_times_area:10.4f}")
```

### 6.3 구면 조화 함수 미리보기

라플라스 방정식 $\nabla^2 f = 0$을 구면 좌표에서 변수분리하면, 각도 부분의 해가 **구면 조화 함수(spherical harmonics)** $Y_l^m(\theta, \phi)$이다:

$$Y_l^m(\theta, \phi) = N_{lm} \, P_l^m(\cos\theta) \, e^{im\phi}$$

여기서 $P_l^m$은 결합 르장드르 함수(associated Legendre function), $N_{lm}$은 정규화 상수이다.

구면 조화 함수는 양자역학(수소 원자 궤도함수), 전자기학(다극 전개), 지구물리학(중력장 모델링) 등에서 핵심적으로 사용된다.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm
from mpl_toolkits.mplot3d import Axes3D

# --- 구면 조화 함수 시각화 ---
theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2*np.pi, 100)
THETA, PHI = np.meshgrid(theta, phi)

fig, axes = plt.subplots(2, 3, figsize=(15, 10),
                          subplot_kw={'projection': '3d'})

harmonics = [
    (0, 0, '$Y_0^0$'),
    (1, 0, '$Y_1^0$'),
    (1, 1, '$Y_1^1$ (real)'),
    (2, 0, '$Y_2^0$'),
    (2, 1, '$Y_2^1$ (real)'),
    (2, 2, '$Y_2^2$ (real)'),
]

for idx, (l, m, title) in enumerate(harmonics):
    ax = axes[idx // 3][idx % 3]

    # scipy의 sph_harm은 (m, l, φ, θ) 순서에 주의
    Y = sph_harm(m, l, PHI, THETA)

    # 실수 구면 조화 함수
    if m > 0:
        Y_real = np.real(Y) * np.sqrt(2) * (-1)**m
    elif m < 0:
        Y_real = np.imag(Y) * np.sqrt(2) * (-1)**m
    else:
        Y_real = np.real(Y)

    # |Y|를 반지름으로, 부호를 색으로 표현
    R = np.abs(Y_real)
    X = R * np.sin(THETA) * np.cos(PHI)
    Y_coord = R * np.sin(THETA) * np.sin(PHI)
    Z = R * np.cos(THETA)

    colors = np.where(Y_real >= 0, 'steelblue', 'coral')
    # 양수/음수 영역을 다른 색으로 표시
    norm = plt.Normalize(vmin=-np.max(np.abs(Y_real)), vmax=np.max(np.abs(Y_real)))
    facecolors = plt.cm.RdBu(norm(Y_real))

    ax.plot_surface(X, Y_coord, Z, facecolors=facecolors, alpha=0.8)
    ax.set_title(title, fontsize=14)
    ax.set_box_aspect([1, 1, 1])
    # 축 레이블 숨기기
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

plt.suptitle('구면 조화 함수 $Y_l^m(\\theta, \\phi)$\n(파랑: 양, 빨강: 음)', fontsize=16)
plt.tight_layout()
plt.savefig('spherical_harmonics.png', dpi=150)
plt.show()
```

> **참고**: 구면 조화 함수의 자세한 이론은 [11. 급수해와 특수함수](11_Series_Solutions_Special_Functions.md)에서 르장드르 함수와 함께 다룬다.

---

## 연습 문제

### 문제 1: 야코비안 계산

다음 좌표 변환의 야코비안을 구하라:

(a) $x = u^2 - v^2$, $y = 2uv$ (포물선 좌표)

(b) $x = e^u \cos v$, $y = e^u \sin v$ (로그 극좌표)

**힌트**: 야코비안 행렬의 행렬식을 계산한다.

### 문제 2: 적분 순서 변경

다음 적분의 순서를 변경하여 계산하라:

$$\int_0^4 \int_{\sqrt{y}}^{2} \frac{1}{x^3 + 1} \, dx \, dy$$

**힌트**: 적분 영역은 $0 \le \sqrt{y} \le x \le 2$이므로 $0 \le y \le x^2$, $0 \le x \le 2$로 변환한다.

### 문제 3: 원통 좌표 적분

원통 좌표를 이용하여 다음을 계산하라:

(a) 반지름 $R$, 높이 $H$인 원뿔 $z = H(1 - \rho/R)$의 부피

(b) 위 원뿔이 균일 밀도 $\rho_0$를 가질 때, 꼭짓점을 지나는 $z$축 기준의 관성 모멘트

### 문제 4: 구면 좌표 응용

반지름 $R$인 구의 상반부($z > 0$)에서 균일 밀도 $\rho_0$인 물체의 질량 중심 $\bar{z}$를 구하라.

$$\bar{z} = \frac{\iiint z \, \rho_0 \, dV}{\iiint \rho_0 \, dV}$$

**답**: $\bar{z} = 3R/8$

### 문제 5: 일반 곡선좌표계

토로이드 좌표(toroidal coordinates) $(\tau, \sigma, \phi)$의 변환식이 다음과 같다:

$$x = \frac{a \sinh\tau \cos\phi}{\cosh\tau - \cos\sigma}, \quad y = \frac{a \sinh\tau \sin\phi}{\cosh\tau - \cos\sigma}, \quad z = \frac{a \sin\sigma}{\cosh\tau - \cos\sigma}$$

이 좌표계의 스케일 인자 $h_\tau$, $h_\sigma$, $h_\phi$를 구하라 (SymPy 사용 가능).

### 문제 6: 물리 응용 종합

(a) 반지름 $R$인 구 표면에 면전하 밀도 $\sigma(\theta) = \sigma_0 \cos\theta$가 분포되어 있을 때, 총 전하량을 구하라.

(b) 이 전하 분포에 의한 구 중심에서의 전위를 구하라.

**힌트**: (a)에서 $\int_0^{\pi} \cos\theta \sin\theta \, d\theta = 0$임을 이용하라. 이는 쌍극자(dipole) 분포이다.

---

## 참고 자료

### 교재
1. **Boas, M. L.** (2005). *Mathematical Methods in the Physical Sciences*, 3rd ed., Chapter 5. Wiley.
2. **Arfken, G. B., Weber, H. J., & Harris, F. E.** (2012). *Mathematical Methods for Physicists*, 7th ed., Chapters 2-3. Academic Press.
3. **Griffiths, D. J.** (2017). *Introduction to Electrodynamics*, 4th ed., Chapter 1 (벡터 해석 및 좌표계). Cambridge University Press.

### 핵심 공식 요약

| 항목 | 원통 좌표 $(\rho, \phi, z)$ | 구면 좌표 $(r, \theta, \phi)$ |
|------|---------------------------|-------------------------------|
| 스케일 인자 | $(1, \rho, 1)$ | $(1, r, r\sin\theta)$ |
| 체적 요소 | $\rho \, d\rho \, d\phi \, dz$ | $r^2\sin\theta \, dr \, d\theta \, d\phi$ |
| $\nabla f$ (r성분) | $\partial f/\partial\rho$ | $\partial f/\partial r$ |
| $\nabla \cdot \mathbf{F}$ | $\frac{1}{\rho}\partial_\rho(\rho F_\rho) + \cdots$ | $\frac{1}{r^2}\partial_r(r^2 F_r) + \cdots$ |
| 야코비안 | $\rho$ | $r^2\sin\theta$ |

### 온라인 자료
1. **MIT OCW 18.02**: Multivariable Calculus (이중/삼중적분, 좌표변환)
2. **Paul's Online Math Notes**: Cylindrical and Spherical Coordinates
3. **Wolfram MathWorld**: Curvilinear Coordinates

---

## 다음 레슨

[07. 푸리에 급수 (Fourier Series)](07_Fourier_Series.md)에서는 주기 함수를 삼각함수의 급수로 전개하는 푸리에 급수를 다룬다. 좌표계에서 변수분리법을 적용할 때 나타나는 고유함수 전개의 기초가 되는 핵심 도구이다.
