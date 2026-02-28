# 09. 다변수 함수

## 학습 목표

- 여러 변수의 함수를 정의하고 등고선(level curve)과 등위면(level surface)을 기하학적으로 해석할 수 있다
- 편도함수(partial derivative)와 방향도함수(directional derivative)를 계산하고 기울기 벡터(gradient vector)의 역할을 설명할 수 있다
- 합성 함수에 다변수 연쇄 법칙(chain rule)을 적용할 수 있다
- 곡면에 대한 접평면(tangent plane)과 선형 근사(linear approximation)를 구성할 수 있다
- 이변수 함수에 대한 2계 도함수 판정법을 사용하여 임계점을 분류할 수 있다

---

## 1. 여러 변수의 함수

함수 $f: \mathbb{R}^n \to \mathbb{R}$은 $n$차원 공간의 각 점에 하나의 실수를 할당한다.

**두 변수:** $z = f(x, y)$는 평면의 점 $(x, y)$를 높이 $z$로 대응시킨다. 그래프는 3D에서 **곡면(surface)** 이다.

**비유:** $f(x, y)$를 지형도(topographic map)라 생각하라. 입력 $(x, y)$는 당신의 GPS 위치이고; 출력 $z$는 당신의 고도이다. 지도 자체는 **등고선(level curves, contour lines)** 을 보여준다 -- 상수 $c$에 대해 $f(x, y) = c$인 곡선.

**예시:**

$$f(x, y) = x^2 + y^2$$

- 그래프는 위로 열린 **포물면(paraboloid)** 이다.
- 등고선: $x^2 + y^2 = c$는 반지름 $\sqrt{c}$인 원이다.
- 촘촘한 등고선은 곡면이 가파르다는 것을 의미한다 (산에서 촘촘한 등고선처럼).

**세 변수:** $w = f(x, y, z)$는 직접 그릴 수 없지만 (4D가 필요), **등위면(level surfaces)** $f(x, y, z) = c$를 시각화할 수 있다.

**예시:** 온도장 $T(x, y, z) = 100 - x^2 - y^2 - z^2$는 동심 구인 등위면(등온면)을 가진다.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Level curves (contour plot) ---
x = np.linspace(-3, 3, 200)
y = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(x, y)

# Saddle function: f(x,y) = x^2 - y^2
Z = X**2 - Y**2

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Contour plot (level curves)
cs = axes[0].contour(X, Y, Z, levels=15, cmap='RdBu_r')
axes[0].clabel(cs, inline=True, fontsize=8)
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title('Level Curves of $f(x,y) = x^2 - y^2$')
axes[0].set_aspect('equal')

# Filled contour for better visualization
cf = axes[1].contourf(X, Y, Z, levels=20, cmap='RdBu_r')
plt.colorbar(cf, ax=axes[1], label='f(x,y)')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_title('Filled Contour Plot')
axes[1].set_aspect('equal')

plt.tight_layout()
plt.show()

# --- 3D surface plot ---
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 5))

# Paraboloid
ax1 = fig.add_subplot(121, projection='3d')
Z1 = X**2 + Y**2
ax1.plot_surface(X, Y, Z1, cmap='viridis', alpha=0.8)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_title('Paraboloid: $z = x^2 + y^2$')

# Saddle surface
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, Z, cmap='RdBu_r', alpha=0.8)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
ax2.set_title('Saddle: $z = x^2 - y^2$')

plt.tight_layout()
plt.show()
```

---

## 2. 편도함수

### 2.1 정의와 기하학적 의미

$f(x, y)$의 $x$에 대한 **편도함수(partial derivative)** 는:

$$\frac{\partial f}{\partial x} = f_x = \lim_{h \to 0} \frac{f(x + h, y) - f(x, y)}{h}$$

**핵심 아이디어:** 다른 모든 변수를 상수로 고정하고 한 번에 하나의 변수에 대해 미분한다.

**기하학적 해석:** $f_x(a, b)$는 점 $(a, b)$에서 $x$-방향으로의 곡면 $z = f(x, y)$의 기울기이다. 곡면을 평면 $y = b$로 자르면, 결과 곡선의 기울기가 $f_x$이다.

**예시:** $f(x, y) = x^2 y + \sin(xy)$인 경우:

$$f_x = 2xy + y\cos(xy) \quad \text{($y$를 상수로 취급)}$$

$$f_y = x^2 + x\cos(xy) \quad \text{($x$를 상수로 취급)}$$

### 2.2 고계 편도함수

2계 편도함수:

$$f_{xx} = \frac{\partial^2 f}{\partial x^2}, \quad f_{yy} = \frac{\partial^2 f}{\partial y^2}, \quad f_{xy} = \frac{\partial^2 f}{\partial y\,\partial x}, \quad f_{yx} = \frac{\partial^2 f}{\partial x\,\partial y}$$

**클레로의 정리(Clairaut's Theorem):** $f_{xy}$와 $f_{yx}$가 모두 연속이면, $f_{xy} = f_{yx}$이다. 이는 미분 순서가 상관없다는 것을 의미한다 -- 매우 유용한 사실이다.

---

## 3. 방향도함수와 기울기

### 3.1 방향도함수

편도함수는 좌표축 방향으로의 변화율을 제공한다. **임의의 방향**에서는 어떨까?

$(a, b)$에서 단위 벡터 $\hat{\mathbf{u}} = (u_1, u_2)$ 방향으로의 $f$의 **방향도함수(directional derivative)** 는:

$$D_{\hat{\mathbf{u}}} f = \lim_{h \to 0} \frac{f(a + hu_1,\, b + hu_2) - f(a, b)}{h}$$

$f$가 미분 가능하면, 이것은 아름답게 단순화된다:

$$D_{\hat{\mathbf{u}}} f = f_x \, u_1 + f_y \, u_2 = \nabla f \cdot \hat{\mathbf{u}}$$

### 3.2 기울기 벡터

$f(x, y)$의 **기울기(gradient)** 는 편도함수의 벡터이다:

$$\nabla f = \left(\frac{\partial f}{\partial x},\, \frac{\partial f}{\partial y}\right) = f_x \,\hat{\mathbf{i}} + f_y \,\hat{\mathbf{j}}$$

**기울기의 세 가지 근본 성질:**

1. **최급상승 방향(direction of steepest ascent):** $\nabla f$는 $f$가 가장 빠르게 증가하는 방향을 가리킨다.
2. **크기 = 최대 변화율:** $\|\nabla f\|$는 최대 방향도함수와 같다.
3. **등고선에 수직:** $\nabla f$는 그 점을 지나는 등고선 $f(x,y) = c$에 항상 직교한다.

**비유:** 산 비탈에 서 있다고 상상하라. 기울기 벡터는 "이것이 가장 가파른 오르막 방향이고, 이것이 그 가파른 정도"라고 말해준다. 물은 $-\nabla f$ 방향(최급하강)으로 흐른다 -- 이것이 정확히 기계 학습에서 경사 하강법(gradient descent)의 원리이다.

**예시:** $f(x, y) = x^2 + 4y^2$, 점 $(1, 1)$에서:

$$\nabla f = (2x, 8y) \big|_{(1,1)} = (2, 8)$$

- 최급상승 방향: $(2, 8) / \|(2,8)\| = (1/\sqrt{17},\, 4/\sqrt{17})$
- 최대 증가율: $\|(2, 8)\| = \sqrt{68} = 2\sqrt{17}$

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Gradient field visualization ---
x = np.linspace(-2, 2, 20)
y = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x, y)

# Function: f(x,y) = x^2 + 4y^2 (elliptic paraboloid)
Z = X**2 + 4 * Y**2

# Gradient components
dfdx = 2 * X      # partial f / partial x
dfdy = 8 * Y      # partial f / partial y

fig, ax = plt.subplots(figsize=(8, 8))

# Draw level curves in the background
x_fine = np.linspace(-2, 2, 200)
y_fine = np.linspace(-2, 2, 200)
Xf, Yf = np.meshgrid(x_fine, y_fine)
Zf = Xf**2 + 4 * Yf**2
cs = ax.contour(Xf, Yf, Zf, levels=10, cmap='Blues', alpha=0.6)
ax.clabel(cs, inline=True, fontsize=8)

# Draw gradient vectors (arrows point in direction of steepest ascent)
ax.quiver(X, Y, dfdx, dfdy, color='red', alpha=0.7,
          scale=60, width=0.004)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Gradient Field of $f(x,y) = x^2 + 4y^2$\n'
             'Arrows = $\\nabla f$ (steepest ascent), perpendicular to contours')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 4. 다변수 함수의 연쇄 법칙

### 4.1 단일 매개변수

$z = f(x, y)$에서 $x = x(t)$이고 $y = y(t)$이면:

$$\frac{dz}{dt} = \frac{\partial f}{\partial x}\frac{dx}{dt} + \frac{\partial f}{\partial y}\frac{dy}{dt}$$

**직관:** $z$의 총 변화율은 $x$-경로와 $y$-경로 양쪽의 기여를 가진다. 각 기여는 민감도 ($\partial f/\partial x$ 또는 $\partial f/\partial y$)에 해당 변수의 변화율을 곱한 것이다.

### 4.2 두 매개변수

$z = f(x, y)$에서 $x = x(s, t)$이고 $y = y(s, t)$이면:

$$\frac{\partial z}{\partial s} = \frac{\partial f}{\partial x}\frac{\partial x}{\partial s} + \frac{\partial f}{\partial y}\frac{\partial y}{\partial s}$$

$$\frac{\partial z}{\partial t} = \frac{\partial f}{\partial x}\frac{\partial x}{\partial t} + \frac{\partial f}{\partial y}\frac{\partial y}{\partial t}$$

유용한 기억법은 **수형도(tree diagram)** 이다: $z$에서 $x$와 $y$로 분기하고, 각각에서 $s$와 $t$로 분기하는 나무를 그린다. 각 가지를 따라 곱하고 모든 경로에 대해 합한다.

**예시:** $z = x^2 y$, $x = s\cos t$, $y = s\sin t$이면:

$$\frac{\partial z}{\partial s} = 2xy \cdot \cos t + x^2 \cdot \sin t$$

### 4.3 음함수 미분

$F(x, y) = 0$이 $y$를 $x$의 함수로 암묵적으로 정의하면:

$$\frac{dy}{dx} = -\frac{F_x}{F_y} \quad \text{(단, } F_y \neq 0\text{)}$$

이것은 $F(x, y(x)) = 0$을 연쇄 법칙으로 미분하면 나온다.

---

## 5. 접평면과 선형 근사

### 5.1 접평면

점 $(a, b, f(a,b))$에서 $z = f(x, y)$의 **접평면(tangent plane)** 은:

$$z - f(a,b) = f_x(a,b)(x - a) + f_y(a,b)(y - b)$$

이것은 접선 $y - f(a) = f'(a)(x - a)$의 2차원 대응물이다.

### 5.2 선형 근사

점 $(a, b)$ 근처에서 $f$를 접평면으로 근사할 수 있다:

$$f(x, y) \approx f(a, b) + f_x(a,b)(x - a) + f_y(a,b)(y - b)$$

동등하게, **전미분(total differential)** 은:

$$df = f_x\,dx + f_y\,dy$$

**응용:** $z = \sqrt{x^2 + y^2}$ (원점으로부터의 거리)일 때, $(3, 4)$에서의 접평면을 사용하여 $(3.02, 3.97)$에서 $z$를 근사하라:

$$f(3, 4) = 5, \quad f_x = \frac{x}{\sqrt{x^2+y^2}} = \frac{3}{5}, \quad f_y = \frac{4}{5}$$

$$f(3.02, 3.97) \approx 5 + \frac{3}{5}(0.02) + \frac{4}{5}(-0.03) = 5 + 0.012 - 0.024 = 4.988$$

정확한 값: $\sqrt{3.02^2 + 3.97^2} = \sqrt{24.8813} \approx 4.98812$ -- 선형 근사가 매우 가깝다.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Tangent plane visualization ---
def f(x, y):
    return np.sin(x) * np.cos(y)

def tangent_plane(x, y, a, b):
    """Tangent plane to f at point (a, b)."""
    f0 = f(a, b)
    fx = np.cos(a) * np.cos(b)   # partial f / partial x
    fy = -np.sin(a) * np.sin(b)  # partial f / partial y
    return f0 + fx * (x - a) + fy * (y - b)

a, b = 1.0, 0.5  # point of tangency

x = np.linspace(-1, 3, 100)
y = np.linspace(-1.5, 2.5, 100)
X, Y = np.meshgrid(x, y)

Z_surface = f(X, Y)
Z_tangent = tangent_plane(X, Y, a, b)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot surface with some transparency
ax.plot_surface(X, Y, Z_surface, cmap='viridis', alpha=0.6)

# Plot tangent plane (clip to a small region for clarity)
mask = (np.abs(X - a) < 1.2) & (np.abs(Y - b) < 1.2)
Z_plane_clipped = np.where(mask, Z_tangent, np.nan)
ax.plot_surface(X, Y, Z_plane_clipped, color='red', alpha=0.4)

# Mark the point of tangency
ax.scatter([a], [b], [f(a, b)], color='black', s=80, zorder=5)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title(f'Surface $z = \\sin(x)\\cos(y)$ and Tangent Plane at ({a}, {b})')
plt.tight_layout()
plt.show()
```

---

## 6. 임계점과 2계 도함수 판정법

### 6.1 임계점 찾기

$f(x, y)$의 **임계점(critical point)** 은 두 편도함수가 모두 0인 곳이다:

$$f_x(a, b) = 0 \quad \text{그리고} \quad f_y(a, b) = 0$$

임계점에서 접평면은 수평이다.

### 6.2 2계 도함수 판정법

임계점 $(a, b)$를 분류하기 위해, **판별식(discriminant)** (헤시안 행렬식(Hessian determinant))을 계산한다:

$$D = f_{xx}(a,b)\,f_{yy}(a,b) - [f_{xy}(a,b)]^2$$

| 조건 | 분류 |
|-----------|----------------|
| $D > 0$이고 $f_{xx} > 0$ | 극소(local minimum) |
| $D > 0$이고 $f_{xx} < 0$ | 극대(local maximum) |
| $D < 0$ | 안장점(saddle point) |
| $D = 0$ | 판정 불가(inconclusive, 추가 분석 필요) |

**왜 이것이 작동하는가?** 헤시안 행렬(Hessian matrix) $H = \begin{pmatrix} f_{xx} & f_{xy} \\ f_{xy} & f_{yy} \end{pmatrix}$는 모든 방향에서 $f$의 곡률을 포착한다. $H$의 두 고유값이 모두 양수이면(양의 정부호(positive definite)) 극소; 모두 음수이면 극대; 부호가 다르면 안장점이다.

**예시:** $f(x, y) = x^3 - 3xy + y^3$

$$f_x = 3x^2 - 3y = 0 \implies y = x^2$$

$$f_y = -3x + 3y^2 = 0 \implies x = y^2$$

대입하면: $x = (x^2)^2 = x^4$이므로 $x(x^3 - 1) = 0$이고, $x = 0$ 또는 $x = 1$.

- $(0, 0)$에서: $D = (0)(0) - (-3)^2 = -9 < 0$ -- **안장점**
- $(1, 1)$에서: $D = (6)(6) - (-3)^2 = 27 > 0$이고 $f_{xx} = 6 > 0$ -- **극소**

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy import symbols, diff, solve, Matrix

# --- Symbolic critical point analysis ---
x, y = symbols('x y', real=True)
f = x**3 - 3*x*y + y**3

fx = diff(f, x)
fy = diff(f, y)
print(f"f_x = {fx}")
print(f"f_y = {fy}")

# Find critical points
critical_pts = solve([fx, fy], [x, y])
print(f"Critical points: {critical_pts}")

# Second derivative test
fxx = diff(f, x, 2)
fyy = diff(f, y, 2)
fxy = diff(f, x, y)

for pt in critical_pts:
    D_val = fxx.subs([(x, pt[0]), (y, pt[1])]) * \
            fyy.subs([(x, pt[0]), (y, pt[1])]) - \
            fxy.subs([(x, pt[0]), (y, pt[1])])**2
    fxx_val = fxx.subs([(x, pt[0]), (y, pt[1])])
    print(f"\nAt {pt}: D = {D_val}, f_xx = {fxx_val}")
    if D_val > 0 and fxx_val > 0:
        print("  -> Local minimum")
    elif D_val > 0 and fxx_val < 0:
        print("  -> Local maximum")
    elif D_val < 0:
        print("  -> Saddle point")
    else:
        print("  -> Inconclusive")
```

---

## 7. 상호 참조

- **물리수학 레슨 04**는 여기서 다룬 비제약 최적화를 확장하는 **라그랑주 승수법(Lagrange multipliers)** 을 다룬다.
- **물리수학 레슨 05**는 다양한 좌표계에서의 div, grad, curl을 포함한 벡터 해석(vector analysis)의 심화 내용을 제공한다.
- **AI를 위한 수학 레슨 08**은 이 레슨의 기울기 개념을 직접 적용하는 경사 하강법(gradient descent) 최적화를 논의한다.

---

## 연습 문제

**1.** $f(x, y) = \ln(x^2 + y^2)$에 대해:
   - (a) $\nabla f$를 구하고 방사 방향을 가리킴을 보여라.
   - (b) $(1, 1)$에서 $(3, 4)$ 방향의 방향도함수를 계산하라.
   - (c) $\nabla f$가 정의되지 않는 점은 어디인가?

**2.** $w = xy + yz + zx$이고 $x = t$, $y = t^2$, $z = t^3$이다. 연쇄 법칙을 사용하여 $dw/dt$를 구하고, 먼저 대입한 후 직접 미분하여 검증하라.

**3.** $f(x, y) = 2x^3 + 6xy^2 - 3y^3 - 150x$의 모든 임계점을 찾고 분류하라.

**4.** 이상 기체 법칙(ideal gas law) $PV = nRT$는 $P$를 $V$와 $T$의 함수로 암묵적으로 정의한다.
   - (a) 음함수 미분을 사용하여 $\partial P/\partial V$와 $\partial P/\partial T$를 구하라.
   - (b) $\frac{\partial P}{\partial V}\frac{\partial V}{\partial T}\frac{\partial T}{\partial P} = -1$ (**순환 관계(cyclic relation)**)을 검증하라.

**5.** $(4, 0)$에서 $f(x, y) = \sqrt{x}\,e^y$의 선형 근사를 사용하여 $f(4.1, -0.05)$를 추정하라. 정확한 값과 비교하라.

---

## 참고 자료

- **James Stewart**, *Calculus: Early Transcendentals*, 9th Edition, Chapters 14.1-14.7
- **Jerrold E. Marsden & Anthony Tromba**, *Vector Calculus*, 6th Edition, Chapter 2
- **George B. Thomas**, *Thomas' Calculus*, Chapters 14-15
- **Khan Academy**, "Multivariable Calculus" (인터랙티브 시각화)

---

[이전: 매개변수 곡선과 극좌표](./08_Parametric_and_Polar.md) | [다음: 중적분](./10_Multiple_Integrals.md)
