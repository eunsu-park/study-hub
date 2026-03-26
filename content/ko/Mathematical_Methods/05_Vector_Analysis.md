# 05. 벡터 해석 (Vector Analysis)

## 학습 목표
- 기울기(gradient), 발산(divergence), 회전(curl) 연산자의 물리적 의미를 이해하고 계산할 수 있다
- 선적분과 면적분을 수행하고, 보존장(conservative field)의 판별 조건을 설명할 수 있다
- 그린 정리, 스토크스 정리, 가우스 발산 정리를 서술하고 적용할 수 있다
- 맥스웰 방정식을 적분 형태와 미분 형태로 상호 변환할 수 있다
- Python(SymPy, Matplotlib)을 이용하여 벡터장을 시각화하고, 선적분/면적분을 수치적으로 계산할 수 있다

---

## 1. 벡터 미분 연산자

벡터 미분 연산자는 스칼라장(scalar field)과 벡터장(vector field)의 공간적 변화를 기술하는 핵심 도구이다. 3차원 직교 좌표계에서 나블라(nabla) 연산자는 다음과 같이 정의된다:

$$
\nabla = \hat{x}\frac{\partial}{\partial x} + \hat{y}\frac{\partial}{\partial y} + \hat{z}\frac{\partial}{\partial z}
$$

### 1.1 기울기 (Gradient, nabla f)

스칼라장 $f(x, y, z)$의 **기울기(gradient)**는 $f$가 가장 빠르게 증가하는 방향과 그 변화율을 나타내는 벡터장이다.

$$
\nabla f = \frac{\partial f}{\partial x}\hat{x} + \frac{\partial f}{\partial y}\hat{y} + \frac{\partial f}{\partial z}\hat{z}
$$

**물리적 의미:**
- 방향: $f$가 가장 빠르게 증가하는 방향
- 크기: 그 방향으로의 변화율 (directional derivative의 최댓값)
- 등위면(level surface)에 항상 수직

**예시:** 온도 분포 $T(x, y) = x^2 + y^2$ (2D 열원)

```python
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.vector import CoordSys3D

# === SymPy를 이용한 해석적 gradient 계산 ===
N = CoordSys3D('N')
x, y, z = sp.symbols('x y z')

# 스칼라장 정의
f = x**2 + y**2

# gradient 계산
grad_f = sp.diff(f, x)*N.i + sp.diff(f, y)*N.j
print(f"f = {f}")
print(f"∇f = {grad_f}")  # 2*x*N.i + 2*y*N.j

# === Matplotlib를 이용한 시각화 ===
X, Y = np.meshgrid(np.linspace(-3, 3, 30), np.linspace(-3, 3, 30))
T = X**2 + Y**2  # 온도 분포

# gradient 성분
dTdx = 2 * X
dTdy = 2 * Y

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 등고선 + gradient 벡터
ax = axes[0]
contour = ax.contourf(X, Y, T, levels=20, cmap='hot')
ax.quiver(X[::3, ::3], Y[::3, ::3], dTdx[::3, ::3], dTdy[::3, ::3],
          color='cyan', alpha=0.8)
plt.colorbar(contour, ax=ax, label='T(x,y)')
ax.set_title('온도 분포와 gradient 벡터')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')

# gradient 크기
ax = axes[1]
grad_mag = np.sqrt(dTdx**2 + dTdy**2)
im = ax.pcolormesh(X, Y, grad_mag, cmap='viridis', shading='auto')
plt.colorbar(im, ax=ax, label='|∇T|')
ax.set_title('gradient 크기 (변화율)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('gradient_visualization.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 1.2 발산 (Divergence, nabla . F)

벡터장 $\mathbf{F} = F_x\hat{x} + F_y\hat{y} + F_z\hat{z}$의 **발산(divergence)**은 각 점에서 벡터장이 "퍼져나가는 정도"를 나타내는 스칼라량이다.

$$
\nabla \cdot \mathbf{F} = \frac{\partial F_x}{\partial x} + \frac{\partial F_y}{\partial y} + \frac{\partial F_z}{\partial z}
$$

**물리적 의미:**
- $\nabla \cdot \mathbf{F} > 0$: 해당 점은 **원천(source)** — 벡터장이 퍼져나감
- $\nabla \cdot \mathbf{F} < 0$: 해당 점은 **흡수원(sink)** — 벡터장이 모여듦
- $\nabla \cdot \mathbf{F} = 0$: **비압축(solenoidal)** — 생성도 소멸도 없음

```python
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

x, y = sp.symbols('x y')

# 원천이 있는 벡터장: F = (x, y) — 원점에서 퍼져나감
Fx_expr = x
Fy_expr = y
div_F = sp.diff(Fx_expr, x) + sp.diff(Fy_expr, y)
print(f"F = ({Fx_expr})x̂ + ({Fy_expr})ŷ")
print(f"∇·F = {div_F}")  # 2 (항상 양수 → 모든 점이 source)

# 비압축 벡터장: G = (-y, x) — 회전만 하는 장
Gx_expr = -y
Gy_expr = x
div_G = sp.diff(Gx_expr, x) + sp.diff(Gy_expr, y)
print(f"\nG = ({Gx_expr})x̂ + ({Gy_expr})ŷ")
print(f"∇·G = {div_G}")  # 0 (비압축)

# 시각화
X, Y = np.meshgrid(np.linspace(-2, 2, 15), np.linspace(-2, 2, 15))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# F = (x, y): divergence > 0 (source field)
ax = axes[0]
ax.quiver(X, Y, X, Y, color='red', alpha=0.7)
ax.set_title(f'F = (x, y),  ∇·F = {div_F} (원천장)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# G = (-y, x): divergence = 0 (solenoidal)
ax = axes[1]
ax.quiver(X, Y, -Y, X, color='blue', alpha=0.7)
ax.set_title(f'G = (-y, x),  ∇·G = {div_G} (비압축장)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('divergence_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 1.3 회전 (Curl, nabla x F)

벡터장 $\mathbf{F}$의 **회전(curl)**은 각 점에서 벡터장이 "회전하는 정도와 회전축"을 나타내는 벡터장이다.

$$
\nabla \times \mathbf{F} = \begin{vmatrix} \hat{x} & \hat{y} & \hat{z} \\ \frac{\partial}{\partial x} & \frac{\partial}{\partial y} & \frac{\partial}{\partial z} \\ F_x & F_y & F_z \end{vmatrix}
$$

전개하면:

$$
\nabla \times \mathbf{F} = \left(\frac{\partial F_z}{\partial y} - \frac{\partial F_y}{\partial z}\right)\hat{x} + \left(\frac{\partial F_x}{\partial z} - \frac{\partial F_z}{\partial x}\right)\hat{y} + \left(\frac{\partial F_y}{\partial x} - \frac{\partial F_x}{\partial y}\right)\hat{z}
$$

**물리적 의미:**
- 방향: 오른손 법칙에 따른 회전축
- 크기: 회전의 세기 (단위 면적당 순환)
- $\nabla \times \mathbf{F} = \mathbf{0}$이면 **비회전장(irrotational field)** — 보존장의 필요충분조건 (단순연결 영역에서)

```python
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

x, y, z = sp.symbols('x y z')

# 3D 벡터장: F = (-y, x, 0) — z축 둘레 회전
Fx, Fy, Fz = -y, x, sp.Integer(0)

# curl 계산
curl_x = sp.diff(Fz, y) - sp.diff(Fy, z)
curl_y = sp.diff(Fx, z) - sp.diff(Fz, x)
curl_z = sp.diff(Fy, x) - sp.diff(Fx, y)

print(f"F = ({Fx})x̂ + ({Fy})ŷ + ({Fz})ẑ")
print(f"∇×F = ({curl_x})x̂ + ({curl_y})ŷ + ({curl_z})ẑ")
# 결과: (0)x̂ + (0)ŷ + (2)ẑ → z 방향으로 균일한 회전

# 2D streamplot으로 시각화
X, Y = np.meshgrid(np.linspace(-3, 3, 30), np.linspace(-3, 3, 30))
U = -Y  # Fx = -y
V = X   # Fy = x
speed = np.sqrt(U**2 + V**2)

fig, ax = plt.subplots(figsize=(8, 8))
strm = ax.streamplot(X, Y, U, V, color=speed, cmap='coolwarm',
                      density=1.5, linewidth=1.5, arrowsize=1.5)
plt.colorbar(strm.lines, ax=ax, label='|F|')
ax.set_title('F = (-y, x): ∇×F = 2ẑ (균일한 회전장)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('curl_streamplot.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 1.4 라플라시안 (nabla^2)

**스칼라 라플라시안**은 기울기의 발산으로 정의된다:

$$
\nabla^2 f = \nabla \cdot (\nabla f) = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2} + \frac{\partial^2 f}{\partial z^2}
$$

**벡터 라플라시안**은 각 성분에 스칼라 라플라시안을 적용한다:

$$
\nabla^2 \mathbf{F} = (\nabla^2 F_x)\hat{x} + (\nabla^2 F_y)\hat{y} + (\nabla^2 F_z)\hat{z}
$$

**물리적 의미:**
- 한 점에서의 값과 그 주변 평균값의 차이를 나타냄
- $\nabla^2 f > 0$: 주변 평균이 현재 값보다 큼 (극소 경향)
- $\nabla^2 f = 0$: **조화함수(harmonic function)** — 라플라스 방정식의 해

```python
import sympy as sp

x, y, z = sp.symbols('x y z')

# 조화함수인지 확인: f = 1/r (r = sqrt(x^2 + y^2 + z^2))
r = sp.sqrt(x**2 + y**2 + z**2)
f = 1 / r

laplacian_f = sp.diff(f, x, 2) + sp.diff(f, y, 2) + sp.diff(f, z, 2)
laplacian_f_simplified = sp.simplify(laplacian_f)
print(f"f = 1/r")
print(f"∇²f = {laplacian_f_simplified}")  # 0 (r ≠ 0에서 조화함수)

# 비조화함수 예시: g = x^2 + y^2
g = x**2 + y**2
laplacian_g = sp.diff(g, x, 2) + sp.diff(g, y, 2)
print(f"\ng = {g}")
print(f"∇²g = {laplacian_g}")  # 4 (비조화)
```

### 1.5 벡터 항등식

벡터 해석에서 자주 사용되는 중요한 항등식들:

```
┌─────────────────────────────────────────────────────────────────┐
│                    핵심 벡터 항등식                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. ∇×(∇f) = 0          gradient의 curl은 항상 0               │
│     → 보존장은 항상 비회전                                       │
│                                                                 │
│  2. ∇·(∇×F) = 0         curl의 divergence는 항상 0             │
│     → 자기장은 항상 비발산 (∇·B = 0)                            │
│                                                                 │
│  3. ∇×(∇×F) = ∇(∇·F) - ∇²F                                   │
│     → curl of curl 분해 (전자기파 방정식 유도에 사용)             │
│                                                                 │
│  4. ∇·(fF) = f(∇·F) + F·(∇f)         곱의 발산               │
│  5. ∇×(fF) = f(∇×F) + (∇f)×F         곱의 회전               │
│  6. ∇(F·G) = (F·∇)G + (G·∇)F + F×(∇×G) + G×(∇×F)           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

```python
import sympy as sp
from sympy.vector import CoordSys3D, curl, divergence, gradient

N = CoordSys3D('N')
x, y, z = N.x, N.y, N.z

# 항등식 1 검증: curl(grad(f)) = 0
f = x**2 * y + y**2 * z + z**2 * x
grad_f = gradient(f, N)
curl_grad_f = curl(grad_f, N)
print(f"f = {f}")
print(f"∇f = {grad_f}")
print(f"∇×(∇f) = {curl_grad_f}")  # 0

# 항등식 2 검증: div(curl(F)) = 0
F = (x*y*z)*N.i + (x**2 - z)*N.j + (y*z**2)*N.k
curl_F = curl(F, N)
div_curl_F = divergence(curl_F, N)
print(f"\nF = {F}")
print(f"∇×F = {curl_F}")
print(f"∇·(∇×F) = {sp.simplify(div_curl_F)}")  # 0
```

---

## 2. 선적분 (Line Integrals)

선적분은 곡선을 따라 스칼라장이나 벡터장을 적분하는 것으로, 일(work), 순환(circulation), 경로 길이 등의 물리량을 계산한다.

### 2.1 스칼라장의 선적분

스칼라장 $f$를 곡선 $C: \mathbf{r}(t) = (x(t), y(t), z(t))$, $a \leq t \leq b$ 를 따라 적분:

$$
\int_C f \, ds = \int_a^b f(\mathbf{r}(t)) \left|\frac{d\mathbf{r}}{dt}\right| dt
$$

여기서 $ds = |\mathbf{r}'(t)| \, dt$는 호 길이 요소(arc length element)이다.

**응용:** 밀도가 변하는 곡선 철사의 질량, 곡선의 길이

```python
import numpy as np
import sympy as sp

t = sp.Symbol('t')

# 예제: 나선 경로 r(t) = (cos t, sin t, t), 0 <= t <= 2pi 위에서
# f = x^2 + y^2 + z^2 의 선적분
x_t = sp.cos(t)
y_t = sp.sin(t)
z_t = t

f = x_t**2 + y_t**2 + z_t**2  # cos²t + sin²t + t² = 1 + t²

# dr/dt 계산
dx = sp.diff(x_t, t)
dy = sp.diff(y_t, t)
dz = sp.diff(z_t, t)
ds_dt = sp.sqrt(dx**2 + dy**2 + dz**2)
ds_dt_simplified = sp.simplify(ds_dt)
print(f"|dr/dt| = {ds_dt_simplified}")  # sqrt(2)

# 선적분 계산
integrand = f * ds_dt_simplified
result = sp.integrate(integrand, (t, 0, 2*sp.pi))
print(f"∫_C f ds = {sp.simplify(result)}")
print(f"수치값 = {float(result):.4f}")
```

### 2.2 벡터장의 선적분 (일)

벡터장 $\mathbf{F}$를 곡선 $C$를 따라 적분하면 **일(work)**을 얻는다:

$$
W = \int_C \mathbf{F} \cdot d\mathbf{r} = \int_a^b \mathbf{F}(\mathbf{r}(t)) \cdot \mathbf{r}'(t) \, dt
$$

성분으로 전개하면:

$$
W = \int_C F_x \, dx + F_y \, dy + F_z \, dz
$$

```python
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

t = sp.Symbol('t')

# 벡터장 F = (y, -x) 에서 원형 경로를 따른 일(work) 계산
# 경로: r(t) = (cos t, sin t), 0 <= t <= 2pi

# 경로 매개변수화
x_t = sp.cos(t)
y_t = sp.sin(t)

# 벡터장 성분 (경로 위)
Fx = y_t    # F_x = y = sin t
Fy = -x_t   # F_y = -x = -cos t

# dr/dt
dx_dt = sp.diff(x_t, t)  # -sin t
dy_dt = sp.diff(y_t, t)  #  cos t

# F · dr/dt
integrand = Fx * dx_dt + Fy * dy_dt
integrand_simplified = sp.simplify(integrand)
print(f"F·dr/dt = {integrand_simplified}")  # -1

# 일(work) 계산
W = sp.integrate(integrand, (t, 0, 2*sp.pi))
print(f"W = ∮ F·dr = {W}")  # -2*pi (음수: 장이 경로와 반대 방향)

# 시각화: 벡터장과 경로
theta = np.linspace(0, 2*np.pi, 100)
X, Y = np.meshgrid(np.linspace(-1.5, 1.5, 12), np.linspace(-1.5, 1.5, 12))

fig, ax = plt.subplots(figsize=(8, 8))
ax.quiver(X, Y, Y, -X, color='steelblue', alpha=0.6, label='F = (y, -x)')
ax.plot(np.cos(theta), np.sin(theta), 'r-', linewidth=2, label='경로 C')
ax.annotate('', xy=(0.7, 0.7), xytext=(0.71, 0.69),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax.set_title(f'∮ F·dr = {W} (시계 방향 순환)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('line_integral_work.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 2.3 보존장과 퍼텐셜 함수

벡터장 $\mathbf{F}$가 **보존장(conservative field)**이면, 선적분의 값은 경로에 무관하고 양 끝점에만 의존한다.

$$
\mathbf{F} = \nabla \phi \quad \Longleftrightarrow \quad \int_A^B \mathbf{F} \cdot d\mathbf{r} = \phi(B) - \phi(A)
$$

**보존장의 동치 조건 (단순연결 영역):**

```
┌─────────────────────────────────────────────────────────────────┐
│  다음 조건들은 모두 동치이다 (단순연결 영역에서):                 │
├─────────────────────────────────────────────────────────────────┤
│  (1) F = ∇φ 인 퍼텐셜 함수 φ가 존재                            │
│  (2) ∮_C F·dr = 0  (임의의 닫힌 경로에 대해)                    │
│  (3) ∫_A^B F·dr 은 경로에 무관                                  │
│  (4) ∇×F = 0                                                   │
│  (5) F_x dx + F_y dy + F_z dz 가 완전미분                      │
└─────────────────────────────────────────────────────────────────┘
```

```python
import sympy as sp
from sympy.vector import CoordSys3D, curl

N = CoordSys3D('N')
x, y, z = N.x, N.y, N.z

# === 보존장 판별 예제 ===
# F1 = (2xy + z)x̂ + (x² + 2yz)ŷ + (x + y²)ẑ
F1 = (2*x*y + z)*N.i + (x**2 + 2*y*z)*N.j + (x + y**2)*N.k
curl_F1 = curl(F1, N)
print(f"F1 = {F1}")
print(f"∇×F1 = {curl_F1}")  # 0 → 보존장!

# 퍼텐셜 함수 구하기: ∂φ/∂x = 2xy + z
phi_x = sp.Symbol('phi')
phi = sp.integrate(2*x*y + z, x)  # x²y + xz + g(y,z)
print(f"\n∫ (2xy+z)dx = {phi} + g(y,z)")

# g(y,z) 결정: ∂φ/∂y = x² + ∂g/∂y = x² + 2yz → ∂g/∂y = 2yz
g = sp.integrate(2*y*z, y)  # y²z + h(z)
print(f"∫ 2yz dy = {g} + h(z)")

# h(z) 결정: ∂φ/∂z = x + y² + h'(z) = x + y² → h'(z) = 0 → h = C
phi_total = x**2 * y + x*z + y**2 * z
print(f"\nφ(x,y,z) = {phi_total}")

# 검증: ∇φ = F1?
from sympy.vector import gradient
grad_phi = gradient(phi_total, N)
print(f"∇φ = {grad_phi}")
print(f"F1 = ∇φ? {sp.simplify(grad_phi - F1) == N.zero}")

# === 비보존장 예제 ===
# F2 = (y)x̂ + (x + z)ŷ + (y + 1)ẑ — curl ≠ 0 확인
F2 = y*N.i + (x + z)*N.j + (y + 1)*N.k
curl_F2 = curl(F2, N)
print(f"\nF2 = {F2}")
print(f"∇×F2 = {curl_F2}")  # 비보존장 여부 확인
```

---

## 3. 면적분 (Surface Integrals)

### 3.1 면적 요소와 법선 벡터

곡면 $S$가 매개변수 $(u, v)$로 표현될 때: $\mathbf{r}(u, v) = (x(u,v),\, y(u,v),\, z(u,v))$

**면적 요소(surface element):**

$$
d\mathbf{S} = \left(\frac{\partial \mathbf{r}}{\partial u} \times \frac{\partial \mathbf{r}}{\partial v}\right) du \, dv = \hat{n} \, dA
$$

여기서 $\hat{n}$은 단위 법선 벡터, $dA = |d\mathbf{S}|$는 면적 요소의 크기이다.

**$z = g(x, y)$로 주어진 곡면의 경우:**

$$
d\mathbf{S} = \left(-\frac{\partial g}{\partial x}\hat{x} - \frac{\partial g}{\partial y}\hat{y} + \hat{z}\right) dx \, dy
$$

### 3.2 스칼라장의 면적분

$$
\iint_S f \, dA = \iint_D f(\mathbf{r}(u,v)) \left|\frac{\partial \mathbf{r}}{\partial u} \times \frac{\partial \mathbf{r}}{\partial v}\right| du \, dv
$$

**응용:** 곡면의 넓이 ($f = 1$), 곡면 위 물리량의 총합

```python
import numpy as np
import sympy as sp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# 구면 r = 1 의 면적 계산 (구면좌표)
theta, phi = sp.symbols('theta phi')

# 구면 매개변수화: r(θ, φ) = (sinθ cosφ, sinθ sinφ, cosθ)
r_theta = sp.Matrix([sp.cos(phi)*sp.cos(theta),  # ∂r/∂θ
                      sp.sin(phi)*sp.cos(theta),
                      -sp.sin(theta)])
r_phi = sp.Matrix([-sp.sin(phi)*sp.sin(theta),    # ∂r/∂φ
                    sp.cos(phi)*sp.sin(theta),
                    0])

# 외적: ∂r/∂θ × ∂r/∂φ
cross = r_theta.cross(r_phi)
dA = sp.simplify(cross.norm())
print(f"|∂r/∂θ × ∂r/∂φ| = {dA}")  # sin(theta) (θ ∈ [0, π]에서 양수)

# 면적 적분
area = sp.integrate(sp.sin(theta), (phi, 0, 2*sp.pi), (theta, 0, sp.pi))
print(f"구의 면적 = {area}")  # 4*pi

# 3D 시각화
u = np.linspace(0, np.pi, 40)
v = np.linspace(0, 2*np.pi, 40)
U, V = np.meshgrid(u, v)

X = np.sin(U) * np.cos(V)
Y = np.sin(U) * np.sin(V)
Z = np.cos(U)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, alpha=0.6, cmap='viridis')

# 법선 벡터 표시 (일부 점에서)
step = 8
for i in range(0, len(u), step):
    for j in range(0, len(v), step):
        px, py, pz = X[j, i], Y[j, i], Z[j, i]
        ax.quiver(px, py, pz, px*0.3, py*0.3, pz*0.3,
                  color='red', arrow_length_ratio=0.3)

ax.set_title('단위 구면과 법선 벡터')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.tight_layout()
plt.savefig('sphere_normal_vectors.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 3.3 벡터장의 면적분 (플럭스)

벡터장 $\mathbf{F}$가 곡면 $S$를 통과하는 **플럭스(flux)**:

$$
\Phi = \iint_S \mathbf{F} \cdot d\mathbf{S} = \iint_S \mathbf{F} \cdot \hat{n} \, dA
$$

**물리적 의미:**
- 전기 플럭스: 전기장이 곡면을 관통하는 양 (가우스 법칙)
- 질량 플럭스: 유체가 곡면을 통과하는 질량 유량

```python
import sympy as sp

x, y, z = sp.symbols('x y z')
u, v = sp.symbols('u v')

# 예제: F = (x, y, z)가 구면 r = R을 통과하는 플럭스
R = sp.Symbol('R', positive=True)
theta, phi = sp.symbols('theta phi')

# 구면 위에서 r̂ = (sinθ cosφ, sinθ sinφ, cosθ)
# 구면 위에서 F = R*(sinθ cosφ, sinθ sinφ, cosθ) = R*r̂
# F·n̂ = F·r̂ = R (구면에서 법선은 r̂ 방향)

# dA = R² sinθ dθ dφ
integrand = R * R**2 * sp.sin(theta)
flux = sp.integrate(integrand, (phi, 0, 2*sp.pi), (theta, 0, sp.pi))
print(f"F = (x, y, z) 가 구면 r={R}을 관통하는 플럭스:")
print(f"Φ = ∬ F·dS = {flux}")  # 4*pi*R^3

# 발산 정리로 검증: ∬ F·dS = ∭ (∇·F) dV
div_F = 3  # ∇·(x,y,z) = 1 + 1 + 1 = 3
volume = sp.Rational(4, 3) * sp.pi * R**3
flux_divergence = div_F * volume
print(f"\n발산 정리 검증: ∭ (∇·F)dV = 3 × (4/3)πR³ = {flux_divergence}")
print(f"일치 여부: {sp.simplify(flux - flux_divergence) == 0}")  # True
```

---

## 4. 적분 정리

벡터 해석의 세 대적분 정리는 미분 연산(gradient, curl, divergence)과 적분(선적분, 면적분, 체적적분)을 연결하는 근본적인 결과이다.

```
┌─────────────────────────────────────────────────────────────────┐
│          벡터 해석의 세 대적분 정리 — 차원별 정리                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  차원   정리          미분 연산     적분                         │
│  ───────────────────────────────────────────────────            │
│  2D     그린          ∂/∂x, ∂/∂y   면적분 ↔ 선적분             │
│  3D-S   스토크스      ∇×            면적분 ↔ 선적분             │
│  3D-V   가우스        ∇·            체적적분 ↔ 면적분           │
│                                                                 │
│  공통 패턴: ∫∫(미분 연산) = ∮(경계에서의 적분)                  │
│            "내부의 미분 = 경계의 적분"                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.1 그린 정리 (Green's Theorem)

2차원 평면에서, 단순 닫힌 곡선 $C$와 그것이 둘러싸는 영역 $D$에 대해:

$$
\oint_C (P \, dx + Q \, dy) = \iint_D \left(\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}\right) dA
$$

**물리적 의미:** 벡터장의 **순환(circulation)**을 영역 내부의 $z$-성분 회전(curl)의 합으로 환산.

```python
import numpy as np
import sympy as sp

x, y, t = sp.symbols('x y t')

# 예제: P = -y², Q = x² 에 대해 그린 정리 검증
# 영역: 단위 원 x² + y² ≤ 1
P = -y**2
Q = x**2

# 좌변: 선적분 (단위 원 경로)
x_t = sp.cos(t)
y_t = sp.sin(t)
dx_dt = sp.diff(x_t, t)
dy_dt = sp.diff(y_t, t)

P_on_C = P.subs([(x, x_t), (y, y_t)])
Q_on_C = Q.subs([(x, x_t), (y, y_t)])

line_integral = sp.integrate(P_on_C * dx_dt + Q_on_C * dy_dt, (t, 0, 2*sp.pi))
print(f"선적분 ∮(P dx + Q dy) = {line_integral}")

# 우변: 면적분 (극좌표)
r, theta = sp.symbols('r theta')
dQ_dx = sp.diff(Q, x)  # 2x
dP_dy = sp.diff(P, y)  # -2y
integrand = dQ_dx - dP_dy  # 2x + 2y

# 극좌표 변환
integrand_polar = integrand.subs([(x, r*sp.cos(theta)), (y, r*sp.sin(theta))])
area_integral = sp.integrate(integrand_polar * r, (r, 0, 1), (theta, 0, 2*sp.pi))
print(f"면적분 ∬(∂Q/∂x - ∂P/∂y)dA = {area_integral}")

print(f"\n그린 정리 성립: {sp.simplify(line_integral - area_integral) == 0}")
```

**그린 정리의 특수 형태 — 면적 공식:**

$$
A = \frac{1}{2} \oint_C (x \, dy - y \, dx)
$$

이 공식은 측량, 컴퓨터 그래픽스에서 다각형 넓이를 계산할 때 사용된다.

### 4.2 스토크스 정리 (Stokes' Theorem)

3차원에서, 곡면 $S$와 그 경계 곡선 $C = \partial S$에 대해:

$$
\oint_C \mathbf{F} \cdot d\mathbf{r} = \iint_S (\nabla \times \mathbf{F}) \cdot d\mathbf{S}
$$

**해석:** 벡터장의 경계 순환(circulation) = 곡면 위 curl의 플럭스

```python
import numpy as np
import sympy as sp

x, y, z, t = sp.symbols('x y z t')

# 예제: F = (y, -x, z²) 에 대해 스토크스 정리 검증
# 곡면 S: z = 1 - x² - y² (z ≥ 0인 포물면)
# 경계 C: z = 0에서 x² + y² = 1 (단위 원)

# --- curl(F) 계산 ---
Fx, Fy, Fz = y, -x, z**2

curl_x = sp.diff(Fz, y) - sp.diff(Fy, z)  # 0 - 0 = 0
curl_y = sp.diff(Fx, z) - sp.diff(Fz, x)  # 0 - 0 = 0
curl_z = sp.diff(Fy, x) - sp.diff(Fx, y)  # -1 - 1 = -2
print(f"∇×F = ({curl_x}, {curl_y}, {curl_z})")

# --- 좌변: 선적분 ∮_C F·dr ---
# C: r(t) = (cos t, sin t, 0), 0 ≤ t ≤ 2π (반시계)
x_t, y_t, z_t = sp.cos(t), sp.sin(t), sp.Integer(0)
dx_dt = sp.diff(x_t, t)
dy_dt = sp.diff(y_t, t)
dz_dt = sp.diff(z_t, t)

Fx_C = Fy_expr = y_t   # Fx = y = sin t
Fy_C = -x_t             # Fy = -x = -cos t
Fz_C = z_t**2           # Fz = z² = 0

line_int = sp.integrate(
    Fx_C * dx_dt + Fy_C * dy_dt + Fz_C * dz_dt,
    (t, 0, 2*sp.pi)
)
print(f"\n좌변 (선적분): ∮ F·dr = {line_int}")

# --- 우변: 면적분 ∬_S (∇×F)·dS ---
# 곡면 z = 1 - x² - y², dS = (-∂z/∂x, -∂z/∂y, 1) dx dy = (2x, 2y, 1) dx dy
# (∇×F)·dS = (0, 0, -2)·(2x, 2y, 1) dx dy = -2 dx dy

r_sym, theta_sym = sp.symbols('r_s theta_s')
surface_int = sp.integrate(
    -2 * r_sym,  # -2 × r (야코비안)
    (r_sym, 0, 1),
    (theta_sym, 0, 2*sp.pi)
)
print(f"우변 (면적분): ∬ (∇×F)·dS = {surface_int}")
print(f"스토크스 정리 성립: {line_int == surface_int}")
```

### 4.3 가우스 발산 정리 (Divergence Theorem)

닫힌 곡면 $S$가 둘러싸는 체적 $V$에 대해:

$$
\oiint_S \mathbf{F} \cdot d\mathbf{S} = \iiint_V (\nabla \cdot \mathbf{F}) \, dV
$$

**해석:** 벡터장이 닫힌 곡면을 통과하는 총 플럭스 = 체적 내부의 발산 총합

```python
import numpy as np
import sympy as sp

x, y, z = sp.symbols('x y z')

# 예제: F = (x³, y³, z³), 닫힌 곡면 = 단위 구 x²+y²+z² = 1

# ∇·F = 3x² + 3y² + 3z² = 3r²
div_F = sp.diff(x**3, x) + sp.diff(y**3, y) + sp.diff(z**3, z)
print(f"∇·F = {div_F}")  # 3x² + 3y² + 3z²

# 체적 적분 (구면좌표)
r, theta, phi = sp.symbols('r theta phi')
div_F_spherical = 3 * r**2  # 3(x² + y² + z²) = 3r²
jacobian = r**2 * sp.sin(theta)

volume_int = sp.integrate(
    div_F_spherical * jacobian,
    (r, 0, 1),
    (theta, 0, sp.pi),
    (phi, 0, 2*sp.pi)
)
print(f"∭ (∇·F) dV = {volume_int}")  # 12π/5

# 직접 면적분으로 검증
# 구면 위에서 r̂ = (x, y, z) (단위구이므로 |r| = 1)
# F·r̂ = x⁴ + y⁴ + z⁴ (구면 위에서 x² + y² + z² = 1)
# 구면좌표: x = sinθ cosφ, y = sinθ sinφ, z = cosθ

F_dot_n = (sp.sin(theta)*sp.cos(phi))**4 + \
          (sp.sin(theta)*sp.sin(phi))**4 + \
          sp.cos(theta)**4

surface_int = sp.integrate(
    F_dot_n * sp.sin(theta),  # dA = sinθ dθ dφ
    (theta, 0, sp.pi),
    (phi, 0, 2*sp.pi)
)
surface_int_simplified = sp.simplify(surface_int)
print(f"∬ F·dS = {surface_int_simplified}")
print(f"일치: {sp.simplify(volume_int - surface_int_simplified) == 0}")
```

### 4.4 세 정리의 관계

세 적분 정리는 모두 **일반화된 스토크스 정리(generalized Stokes' theorem)**의 특수한 경우이다:

$$
\int_{\partial \Omega} \omega = \int_{\Omega} d\omega
$$

```
┌─────────────────────────────────────────────────────────────────┐
│          세 정리의 통일적 관점                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  미적분학의 기본정리 (1D):                                       │
│    ∫_a^b f'(x) dx = f(b) - f(a)                                │
│    "미분의 적분 = 경계값의 차이"                                 │
│                                                                 │
│  그린 정리 (2D):                                                │
│    ∬_D (∂Q/∂x - ∂P/∂y) dA = ∮_{∂D} (P dx + Q dy)             │
│    "curl의 면적분 = 경계의 선적분"                               │
│                                                                 │
│  스토크스 정리 (3D, 곡면↔경계선):                                │
│    ∬_S (∇×F)·dS = ∮_{∂S} F·dr                                 │
│    "curl의 면적분 = 경계의 선적분"                               │
│                                                                 │
│  가우스 정리 (3D, 체적↔경계면):                                  │
│    ∭_V (∇·F) dV = ∬_{∂V} F·dS                                 │
│    "divergence의 체적적분 = 경계의 면적분"                       │
│                                                                 │
│  패턴: "내부의 미분 = 경계에서의 값"                             │
│        차원 n의 적분 ↔ 차원 (n-1)의 경계 적분                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. 물리학 응용

### 5.1 전기장과 가우스 법칙

**가우스 법칙 (적분 형태):**

$$
\oiint_S \mathbf{E} \cdot d\mathbf{S} = \frac{Q_{\text{enc}}}{\epsilon_0}
$$

**가우스 법칙 (미분 형태):** 가우스 발산 정리를 적용하면:

$$
\nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0}
$$

전하밀도 $\rho$가 있는 곳에서 전기장의 발산이 0이 아니다.

```python
import numpy as np
import matplotlib.pyplot as plt

# 점전하의 전기장 시각화 (2D 단면)
# E = q/(4πε₀) × r̂/r² (쿨롱 법칙)

q = 1.0  # 전하량 (임의 단위)
eps0 = 1.0  # ε₀ (단위계 편의상)

X, Y = np.meshgrid(np.linspace(-3, 3, 20), np.linspace(-3, 3, 20))
R = np.sqrt(X**2 + Y**2)
R = np.where(R < 0.3, 0.3, R)  # 특이점 방지

# 전기장 성분
k = q / (4 * np.pi * eps0)
Ex = k * X / R**3
Ey = k * Y / R**3
E_mag = np.sqrt(Ex**2 + Ey**2)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 양전하 (+q)
ax = axes[0]
ax.streamplot(X, Y, Ex, Ey, color=np.log(E_mag + 1), cmap='Reds',
              density=2, linewidth=1.2)
ax.plot(0, 0, 'ro', markersize=15, label='+q')
circle1 = plt.Circle((0, 0), 1.0, fill=False, color='gray', linestyle='--', label='가우스 면 r=1')
circle2 = plt.Circle((0, 0), 2.0, fill=False, color='gray', linestyle=':', label='가우스 면 r=2')
ax.add_patch(circle1)
ax.add_patch(circle2)
ax.set_title('양전하의 전기장 (발산 > 0)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
ax.legend(loc='upper left', fontsize=9)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)

# 쌍극자 (dipole): +q at (1,0), -q at (-1,0)
ax = axes[1]
d = 1.0
R1 = np.sqrt((X - d)**2 + Y**2)
R2 = np.sqrt((X + d)**2 + Y**2)
R1 = np.where(R1 < 0.3, 0.3, R1)
R2 = np.where(R2 < 0.3, 0.3, R2)

Ex_dip = k * (X - d) / R1**3 - k * (X + d) / R2**3
Ey_dip = k * Y / R1**3 - k * Y / R2**3
E_dip_mag = np.sqrt(Ex_dip**2 + Ey_dip**2)

ax.streamplot(X, Y, Ex_dip, Ey_dip, color=np.log(E_dip_mag + 1),
              cmap='coolwarm', density=2, linewidth=1.2)
ax.plot(d, 0, 'ro', markersize=12, label='+q')
ax.plot(-d, 0, 'bo', markersize=12, label='-q')
ax.set_title('전기 쌍극자 (dipole)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
ax.legend(fontsize=9)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)

plt.tight_layout()
plt.savefig('electric_field_gauss.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 5.2 자기장과 앙페르 법칙

**앙페르 법칙 (적분 형태):**

$$
\oint_C \mathbf{B} \cdot d\mathbf{r} = \mu_0 I_{\text{enc}}
$$

**앙페르 법칙 (미분 형태):** 스토크스 정리를 적용하면:

$$
\nabla \times \mathbf{B} = \mu_0 \mathbf{J}
$$

여기서 $\mathbf{J}$는 전류 밀도(current density)이다.

**물리적 의미:**
- 자기장의 curl은 전류 밀도에 비례
- 자기장선은 전류 주위를 감싸는 닫힌 루프 (오른손 법칙)
- $\nabla \cdot \mathbf{B} = 0$ — 자기 단극자(magnetic monopole)는 존재하지 않음

```python
import numpy as np
import matplotlib.pyplot as plt

# 무한 직선 전류에 의한 자기장
# B = μ₀I/(2πr) × φ̂ (원통좌표)

mu0 = 1.0  # μ₀ (임의 단위)
I = 1.0    # 전류 (z 방향)

X, Y = np.meshgrid(np.linspace(-3, 3, 20), np.linspace(-3, 3, 20))
R = np.sqrt(X**2 + Y**2)
R = np.where(R < 0.3, 0.3, R)

# B = μ₀I/(2πr) × φ̂, 여기서 φ̂ = (-y/r, x/r, 0)
B_coeff = mu0 * I / (2 * np.pi * R)
Bx = B_coeff * (-Y / R)
By = B_coeff * (X / R)

fig, ax = plt.subplots(figsize=(8, 8))
B_mag = np.sqrt(Bx**2 + By**2)
strm = ax.streamplot(X, Y, Bx, By, color=np.log(B_mag + 0.01),
                      cmap='plasma', density=2, linewidth=1.5)
plt.colorbar(strm.lines, ax=ax, label='log|B|')

# 전류 위치 (원점, z 방향)
ax.plot(0, 0, 'g^', markersize=15, label='I (z 방향, 지면에서 나옴)')

# 앙페르 루프 표시
for r in [1.0, 2.0]:
    circle = plt.Circle((0, 0), r, fill=False, color='lime',
                         linestyle='--', linewidth=2)
    ax.add_patch(circle)

ax.set_title('직선 전류 주위의 자기장 (앙페르 법칙)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
ax.legend(loc='upper left')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
plt.tight_layout()
plt.savefig('magnetic_field_ampere.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 5.3 유체역학의 연속 방정식

질량 보존을 표현하는 **연속 방정식(continuity equation)**:

$$
\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{v}) = 0
$$

여기서 $\rho$는 유체 밀도, $\mathbf{v}$는 유속(velocity field)이다.

**유도 과정:**
1. 가우스 발산 정리로부터 닫힌 면을 통한 질량 유출율 = $\oiint_S \rho \mathbf{v} \cdot d\mathbf{S}$
2. 체적 내 질량 변화율 = $-\frac{\partial}{\partial t}\iiint_V \rho \, dV$
3. 발산 정리 적용: $\iiint_V \left[\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{v})\right] dV = 0$
4. 임의의 체적에 대해 성립하므로 피적분함수 = 0

**비압축 유체** ($\rho$ = 상수)의 경우:

$$
\nabla \cdot \mathbf{v} = 0
$$

```python
import numpy as np
import matplotlib.pyplot as plt

# 2D 비압축 유체 흐름 예시
# 흐름 함수(stream function) ψ를 이용: vx = ∂ψ/∂y, vy = -∂ψ/∂x
# → 자동으로 ∇·v = 0 만족

X, Y = np.meshgrid(np.linspace(-3, 3, 25), np.linspace(-3, 3, 25))

# 예제 1: 균일 흐름 + 원통 주위 흐름 (퍼텐셜 흐름)
# ψ = U*y*(1 - a²/r²), U = 자유류 속도, a = 원통 반지름
U_inf = 1.0
a = 1.0
R_sq = X**2 + Y**2
R_sq = np.where(R_sq < a**2, a**2, R_sq)  # 원통 내부 마스킹

Vx = U_inf * (1 - a**2 * (X**2 - Y**2) / R_sq**2)
Vy = -U_inf * 2 * a**2 * X * Y / R_sq**2

# 원통 내부 속도 = 0
mask = (X**2 + Y**2) < a**2
Vx[mask] = 0
Vy[mask] = 0

fig, ax = plt.subplots(figsize=(10, 8))
speed = np.sqrt(Vx**2 + Vy**2)
strm = ax.streamplot(X, Y, Vx, Vy, color=speed, cmap='RdYlBu_r',
                      density=2, linewidth=1.2)
plt.colorbar(strm.lines, ax=ax, label='|v| (속력)')

# 원통 표시
circle = plt.Circle((0, 0), a, color='gray', alpha=0.5)
ax.add_patch(circle)

# 발산 계산 (수치적)
dVx_dx = np.gradient(Vx, X[0], axis=1)
dVy_dy = np.gradient(Vy, Y[:, 0], axis=0)
div_v = dVx_dx + dVy_dy
max_div = np.max(np.abs(div_v[~mask]))
ax.set_title(f'원통 주위 비압축 유체 흐름  (max|∇·v| ≈ {max_div:.2e})')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('fluid_flow_cylinder.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 5.4 맥스웰 방정식의 적분 형태와 미분 형태

맥스웰 방정식은 전자기 현상을 완전히 기술하는 4개의 방정식이다. 벡터 해석의 적분 정리(가우스, 스토크스)를 통해 적분 형태와 미분 형태를 상호 변환할 수 있다.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    맥스웰 방정식 (Maxwell's Equations)                    │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  법칙             미분 형태              적분 형태              변환 정리 │
│  ─────────────────────────────────────────────────────────────────────── │
│                                                                          │
│  가우스 (전기)    ∇·E = ρ/ε₀            ∮ E·dS = Q/ε₀         가우스   │
│                                                                          │
│  가우스 (자기)    ∇·B = 0               ∮ B·dS = 0             가우스   │
│                                                                          │
│  패러데이         ∇×E = -∂B/∂t          ∮ E·dr = -dΦ_B/dt     스토크스 │
│                                                                          │
│  앙페르-맥스웰    ∇×B = μ₀J + μ₀ε₀∂E/∂t                      스토크스 │
│                                   ∮ B·dr = μ₀I + μ₀ε₀ dΦ_E/dt          │
│                                                                          │
│  ─────────────────────────────────────────────────────────────────────── │
│                                                                          │
│  물리적 의미:                                                             │
│  • 가우스(전기): 전하가 전기장의 원천                                      │
│  • 가우스(자기): 자기 단극자는 존재하지 않음                               │
│  • 패러데이: 변하는 자기장이 전기장을 유도                                 │
│  • 앙페르-맥스웰: 전류와 변하는 전기장이 자기장을 유도                     │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

**적분 형태에서 미분 형태로의 변환 (가우스 법칙 예시):**

$$
\oiint_S \mathbf{E} \cdot d\mathbf{S} = \frac{Q_{\text{enc}}}{\epsilon_0}
= \frac{1}{\epsilon_0}\iiint_V \rho \, dV
$$

가우스 발산 정리를 좌변에 적용:

$$
\iiint_V (\nabla \cdot \mathbf{E}) \, dV = \frac{1}{\epsilon_0}\iiint_V \rho \, dV
$$

임의의 체적 $V$에 대해 성립하므로:

$$
\nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0}
$$

```python
import numpy as np
import matplotlib.pyplot as plt

# 맥스웰 방정식의 시각적 정리: 전기장과 자기장의 관계
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# === (1) 가우스 법칙 (전기): ∇·E = ρ/ε₀ ===
ax = axes[0, 0]
X, Y = np.meshgrid(np.linspace(-2, 2, 15), np.linspace(-2, 2, 15))
R = np.sqrt(X**2 + Y**2)
R = np.where(R < 0.3, 0.3, R)
Ex = X / R**3
Ey = Y / R**3
ax.quiver(X, Y, Ex, Ey, color='red', alpha=0.6)
circle = plt.Circle((0, 0), 0.2, color='red', alpha=0.8)
ax.add_patch(circle)
ax.set_title('(1) 가우스 법칙: ∇·E = ρ/ε₀\n전하 → 발산하는 E')
ax.set_aspect('equal')
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.grid(True, alpha=0.2)

# === (2) 가우스 법칙 (자기): ∇·B = 0 ===
ax = axes[0, 1]
# 자기 쌍극자 (단극자 없음)
d = 0.5
R1 = np.sqrt((X - 0)**2 + (Y - d)**2)
R2 = np.sqrt((X - 0)**2 + (Y + d)**2)
R1 = np.where(R1 < 0.3, 0.3, R1)
R2 = np.where(R2 < 0.3, 0.3, R2)
Bx = X / R1**3 - X / R2**3
By = (Y - d) / R1**3 - (Y + d) / R2**3
speed = np.sqrt(Bx**2 + By**2)
ax.streamplot(X, Y, Bx, By, color=np.log(speed + 0.1), cmap='Blues',
              density=2, linewidth=1)
ax.set_title('(2) 가우스 법칙: ∇·B = 0\n자기장선은 닫힌 루프')
ax.set_aspect('equal')
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.grid(True, alpha=0.2)

# === (3) 패러데이 법칙: ∇×E = -∂B/∂t ===
ax = axes[1, 0]
# 변하는 자기장 → 유도 전기장 (원형)
R_circ = np.sqrt(X**2 + Y**2)
R_circ = np.where(R_circ < 0.2, 0.2, R_circ)
Ex_ind = -Y / R_circ**2
Ey_ind = X / R_circ**2
ax.streamplot(X, Y, Ex_ind, Ey_ind, color='orange', density=1.5, linewidth=1.5)
ax.annotate('dB/dt\n(z 방향)', xy=(0, 0), fontsize=12, ha='center',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
ax.set_title('(3) 패러데이: ∇×E = -∂B/∂t\n변하는 B → 유도 E')
ax.set_aspect('equal')
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.grid(True, alpha=0.2)

# === (4) 앙페르-맥스웰: ∇×B = μ₀J + μ₀ε₀ ∂E/∂t ===
ax = axes[1, 1]
R_wire = np.sqrt(X**2 + Y**2)
R_wire = np.where(R_wire < 0.2, 0.2, R_wire)
Bx_wire = -Y / R_wire**2
By_wire = X / R_wire**2
ax.streamplot(X, Y, Bx_wire, By_wire, color='purple', density=1.5, linewidth=1.5)
ax.annotate('I or ∂E/∂t\n(z 방향)', xy=(0, 0), fontsize=12, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax.set_title('(4) 앙페르-맥스웰: ∇×B = μ₀J + μ₀ε₀∂E/∂t\n전류/변하는 E → B')
ax.set_aspect('equal')
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.grid(True, alpha=0.2)

plt.suptitle('맥스웰 방정식의 4가지 법칙', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('maxwell_equations.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 연습 문제

### 문제 1: 기울기와 방향도함수

스칼라장 $f(x, y, z) = x^2 y + y^2 z + z^2 x$ 에 대해:
1. $\nabla f$를 구하라.
2. 점 $(1, 1, 1)$에서 $\hat{u} = \frac{1}{\sqrt{3}}(1, 1, 1)$ 방향의 방향도함수(directional derivative)를 구하라.
3. 점 $(1, 1, 1)$에서 $f$가 가장 빠르게 증가하는 방향과 그 변화율을 구하라.

### 문제 2: 발산과 회전 판별

다음 벡터장에 대해 발산과 회전을 계산하고, 보존장인지 판별하라:

(a) $\mathbf{F} = (yz, xz, xy)$

(b) $\mathbf{G} = (x^2 - y, y^2 + x, z)$

보존장이면 퍼텐셜 함수를 구하라.

### 문제 3: 선적분

벡터장 $\mathbf{F} = (2xy + z^2)\hat{x} + x^2\hat{y} + 2xz\hat{z}$ 에 대해:
1. $\mathbf{F}$가 보존장임을 보이고 퍼텐셜 함수 $\phi$를 구하라.
2. $(0, 0, 0)$에서 $(1, 2, 3)$까지의 선적분을 (a) 퍼텐셜 함수를 이용하여, (b) 직선 경로 $\mathbf{r}(t) = (t, 2t, 3t)$를 따라 직접 계산하여 결과가 같음을 확인하라.

### 문제 4: 가우스 발산 정리 검증

$\mathbf{F} = (x^2, y^2, z^2)$에 대해, 단위 정육면체 $[0,1]^3$에서 가우스 발산 정리를 검증하라.
1. $\nabla \cdot \mathbf{F}$를 구하고 체적적분을 계산하라.
2. 6개 면에서의 면적분을 각각 계산하여 합산하라.
3. 두 결과가 일치함을 확인하라.

### 문제 5: 스토크스 정리와 물리 응용

전류 밀도 $\mathbf{J} = J_0 \hat{z}$ (균일)가 반지름 $a$인 원통 도선에 흐를 때:
1. 앙페르 법칙(적분 형태)을 이용하여 $r < a$와 $r > a$ 영역에서의 자기장 $\mathbf{B}$를 구하라.
2. $r < a$에서 $\nabla \times \mathbf{B} = \mu_0 \mathbf{J}$가 성립함을 직접 확인하라 (원통좌표 curl 공식 사용).

### 문제 6: 맥스웰 방정식 변환

패러데이 법칙의 적분 형태:

$$
\oint_C \mathbf{E} \cdot d\mathbf{r} = -\frac{d}{dt}\iint_S \mathbf{B} \cdot d\mathbf{S}
$$

에서 스토크스 정리를 이용하여 미분 형태 $\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}$를 유도하라. 유도 과정을 단계별로 서술하라.

---

## 참고 자료

### 교재
1. **Boas, M. L.** (2005). *Mathematical Methods in the Physical Sciences*, 3rd ed., Chapter 6. Wiley.
2. **Griffiths, D. J.** (2017). *Introduction to Electrodynamics*, 4th ed. Cambridge University Press.
   - 벡터 해석의 전자기학 응용에 대한 최고의 참고서
3. **Arfken, G. B., Weber, H. J., & Harris, F. E.** (2012). *Mathematical Methods for Physicists*, 7th ed., Chapters 1-3. Academic Press.
4. **Schey, H. M.** (2005). *Div, Grad, Curl, and All That*, 4th ed. W.W. Norton.
   - 벡터 해석의 직관적 입문서

### 온라인 자료
1. **3Blue1Brown** — [Divergence and Curl](https://www.youtube.com/watch?v=rB83DpBJQsE): 발산과 회전의 시각적 이해
2. **MIT OCW 18.02** — Multivariable Calculus: 벡터 해석 강의
3. **Paul's Online Math Notes** — Calculus III: 연습 문제 풍부

### Python 도구
- `sympy.vector`: 기호적 벡터 미적분 (`gradient`, `divergence`, `curl`)
- `matplotlib.pyplot.quiver`: 2D 벡터장 화살표 시각화
- `matplotlib.pyplot.streamplot`: 유선(streamline) 시각화
- `mpl_toolkits.mplot3d`: 3D 곡면 및 벡터 시각화

---

## 다음 레슨

- **이전**: [04. 편미분 (Partial Differentiation)](04_Partial_Differentiation.md) — 다변수 미적분, 연쇄법칙, 테일러 급수, 라그랑주 승수법
- **다음**: [06. 곡선좌표계 (Curvilinear Coordinates)](06_Curvilinear_Coordinates.md) — 원통/구면 좌표, 야코비안, 좌표 변환
