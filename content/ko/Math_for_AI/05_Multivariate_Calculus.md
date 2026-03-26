# 05. 다변수 미적분 (Multivariate Calculus)

## 학습 목표

- 편미분의 기하학적 의미를 이해하고 계산할 수 있다
- 그래디언트 벡터와 등고선의 관계를 시각화하고 해석할 수 있다
- 방향 도함수를 계산하고 그래디언트와의 관계를 이해한다
- 다변수 체인 룰을 적용하여 복잡한 합성 함수를 미분할 수 있다
- 테일러 전개를 이용한 함수 근사와 뉴턴 방법의 원리를 이해한다
- 머신러닝의 손실 지형을 시각화하고 최적화 문제와 연결한다

---

## 1. 편미분 (Partial Derivatives)

### 1.1 편미분의 정의

다변수 함수 $f(x_1, x_2, \ldots, x_n)$에서 $x_i$에 대한 편미분은 다른 변수를 상수로 고정하고 $x_i$만 변화시킬 때의 변화율입니다:

$$\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \ldots, x_i + h, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}$$

### 1.2 편미분 계산 예제

**예제 1**: $f(x, y) = x^2 + 3xy + y^2$

$$\frac{\partial f}{\partial x} = 2x + 3y$$

$$\frac{\partial f}{\partial y} = 3x + 2y$$

**예제 2**: $f(x, y) = e^{x^2 + y^2}$

$$\frac{\partial f}{\partial x} = 2x e^{x^2 + y^2}$$

$$\frac{\partial f}{\partial y} = 2y e^{x^2 + y^2}$$

### 1.3 SymPy를 이용한 심볼릭 미분

```python
import numpy as np
import sympy as sp
from sympy import symbols, diff, exp, sin, cos, simplify

# 심볼 정의
x, y = symbols('x y')

# 함수 정의
f = x**2 + 3*x*y + y**2

# 편미분
df_dx = diff(f, x)
df_dy = diff(f, y)

print("함수:", f)
print(f"∂f/∂x = {df_dx}")
print(f"∂f/∂y = {df_dy}")

# 특정 점에서 계산
point = {x: 1, y: 2}
print(f"\n점 (1, 2)에서:")
print(f"  f(1,2) = {f.subs(point)}")
print(f"  ∂f/∂x(1,2) = {df_dx.subs(point)}")
print(f"  ∂f/∂y(1,2) = {df_dy.subs(point)}")

# 복잡한 함수
g = exp(x**2 + y**2) * sin(x*y)
dg_dx = simplify(diff(g, x))
dg_dy = simplify(diff(g, y))

print(f"\n함수: g(x,y) = {g}")
print(f"∂g/∂x = {dg_dx}")
print(f"∂g/∂y = {dg_dy}")
```

### 1.4 편미분의 기하학적 의미

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 함수 정의: f(x, y) = x^2 + y^2
def f(x, y):
    return x**2 + y**2

# 격자 생성
x_vals = np.linspace(-3, 3, 100)
y_vals = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = f(X, Y)

# 특정 점
x0, y0 = 1.0, 1.5
z0 = f(x0, y0)

fig = plt.figure(figsize=(16, 6))

# 3D 표면
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
ax1.scatter([x0], [y0], [z0], c='red', s=100, label=f'점 ({x0}, {y0})')

# x 방향 접선 (y 고정)
x_line = np.linspace(x0-1, x0+1, 50)
y_line = np.full_like(x_line, y0)
z_line = f(x_line, y_line)
ax1.plot(x_line, y_line, z_line, 'r-', linewidth=3, label='y 고정 (∂f/∂x)')

# y 방향 접선 (x 고정)
x_line2 = np.full(50, x0)
y_line2 = np.linspace(y0-1, y0+1, 50)
z_line2 = f(x_line2, y_line2)
ax1.plot(x_line2, y_line2, z_line2, 'b-', linewidth=3, label='x 고정 (∂f/∂y)')

ax1.set_title('편미분의 기하학적 의미', fontsize=12)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x,y)')
ax1.legend()

# x 고정 단면
ax2 = fig.add_subplot(132)
y_slice = np.linspace(-3, 3, 100)
z_slice = f(x0, y_slice)
ax2.plot(y_slice, z_slice, 'b-', linewidth=2)
ax2.plot([y0], [z0], 'ro', markersize=10)
# 접선
slope_y = 2*y0  # ∂f/∂y = 2y
tangent_y = z0 + slope_y * (y_slice - y0)
ax2.plot(y_slice, tangent_y, 'r--', linewidth=2, label=f'접선 (기울기={slope_y:.1f})')
ax2.set_title(f'x={x0} 고정 단면', fontsize=12)
ax2.set_xlabel('y')
ax2.set_ylabel('f(x,y)')
ax2.grid(True, alpha=0.3)
ax2.legend()

# y 고정 단면
ax3 = fig.add_subplot(133)
x_slice = np.linspace(-3, 3, 100)
z_slice = f(x_slice, y0)
ax3.plot(x_slice, z_slice, 'r-', linewidth=2)
ax3.plot([x0], [z0], 'ro', markersize=10)
# 접선
slope_x = 2*x0  # ∂f/∂x = 2x
tangent_x = z0 + slope_x * (x_slice - x0)
ax3.plot(x_slice, tangent_x, 'b--', linewidth=2, label=f'접선 (기울기={slope_x:.1f})')
ax3.set_title(f'y={y0} 고정 단면', fontsize=12)
ax3.set_xlabel('x')
ax3.set_ylabel('f(x,y)')
ax3.grid(True, alpha=0.3)
ax3.legend()

plt.tight_layout()
plt.savefig('partial_derivatives_geometry.png', dpi=150, bbox_inches='tight')
plt.close()

print("편미분 기하학 시각화 저장: partial_derivatives_geometry.png")
```

### 1.5 고차 편미분

슈바르츠 정리: $f$가 클래스 $C^2$이면 편미분 순서 무관

$$\frac{\partial^2 f}{\partial x \partial y} = \frac{\partial^2 f}{\partial y \partial x}$$

```python
# 슈바르츠 정리 검증
x, y = symbols('x y')
f = x**3 * y**2 + x * y**3

# 혼합 편미분
f_xy = diff(diff(f, x), y)
f_yx = diff(diff(f, y), x)

print("함수:", f)
print(f"∂²f/∂x∂y = {f_xy}")
print(f"∂²f/∂y∂x = {f_yx}")
print(f"동일한가? {simplify(f_xy - f_yx) == 0}")
```

## 2. 그래디언트 (Gradient)

### 2.1 그래디언트 벡터

그래디언트는 모든 편미분을 모은 벡터입니다:

$$\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

그래디언트는 다음을 가리킵니다:
1. **방향**: 함수가 가장 빠르게 증가하는 방향
2. **크기**: 그 방향으로의 변화율

### 2.2 등고선과 그래디언트

그래디언트는 등고선 (level set)에 수직입니다.

```python
# 등고선과 그래디언트 시각화
def f(x, y):
    """함수: f(x, y) = x^2 + 2y^2"""
    return x**2 + 2*y**2

def grad_f(x, y):
    """그래디언트: ∇f = [2x, 4y]"""
    return np.array([2*x, 4*y])

# 격자
x = np.linspace(-3, 3, 300)
y = np.linspace(-3, 3, 300)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# 그래디언트 계산 지점
n_points = 15
x_grad = np.linspace(-2.5, 2.5, n_points)
y_grad = np.linspace(-2.5, 2.5, n_points)
X_grad, Y_grad = np.meshgrid(x_grad, y_grad)

# 각 점에서 그래디언트
U = 2 * X_grad
V = 4 * Y_grad

# 그래디언트 크기로 정규화 (화살표 길이 조정)
norm = np.sqrt(U**2 + V**2)
U_norm = U / (norm + 1e-8) * 0.3
V_norm = V / (norm + 1e-8) * 0.3

plt.figure(figsize=(10, 8))
# 등고선
contours = plt.contour(X, Y, Z, levels=15, cmap='viridis', alpha=0.6)
plt.clabel(contours, inline=True, fontsize=8)
# 그래디언트 벡터
plt.quiver(X_grad, Y_grad, U_norm, V_norm, norm, cmap='Reds',
           scale=1, scale_units='xy', width=0.004)
plt.colorbar(label='그래디언트 크기')
plt.title('등고선과 그래디언트 (그래디언트 ⊥ 등고선)', fontsize=14)
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('gradient_contour.png', dpi=150)
plt.close()

print("등고선-그래디언트 시각화 저장: gradient_contour.png")
```

### 2.3 그래디언트 하강법 시각화

```python
# 그래디언트 하강법 경로
def gradient_descent_2d(grad_f, x0, learning_rate=0.1, n_steps=50):
    """2D 그래디언트 하강법"""
    path = [x0]
    x = x0.copy()

    for _ in range(n_steps):
        grad = grad_f(x[0], x[1])
        x = x - learning_rate * grad
        path.append(x.copy())

    return np.array(path)

# 초기점
x0 = np.array([2.5, 2.0])

# 경로 계산
path = gradient_descent_2d(grad_f, x0, learning_rate=0.15, n_steps=30)

# 시각화
plt.figure(figsize=(10, 8))
plt.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
plt.plot(path[:, 0], path[:, 1], 'ro-', linewidth=2, markersize=6,
         label='그래디언트 하강 경로')
plt.plot(path[0, 0], path[0, 1], 'go', markersize=12, label='시작점')
plt.plot(path[-1, 0], path[-1, 1], 'r*', markersize=20, label='종료점')
plt.plot(0, 0, 'b*', markersize=20, label='최솟값')
plt.colorbar(label='f(x, y)')
plt.title('그래디언트 하강법 경로', fontsize=14)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.savefig('gradient_descent_path.png', dpi=150)
plt.close()

print("그래디언트 하강법 경로 저장: gradient_descent_path.png")
print(f"시작: {path[0]}, 종료: {path[-1]}, 최솟값: [0, 0]")
print(f"최종 함수 값: {f(path[-1, 0], path[-1, 1]):.6f}")
```

### 2.4 그래디언트의 크기와 학습률

```python
# 다양한 학습률로 실험
learning_rates = [0.05, 0.15, 0.3, 0.5]

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for idx, lr in enumerate(learning_rates):
    path = gradient_descent_2d(grad_f, x0, learning_rate=lr, n_steps=30)

    axes[idx].contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
    axes[idx].plot(path[:, 0], path[:, 1], 'ro-', linewidth=2, markersize=4)
    axes[idx].plot(path[0, 0], path[0, 1], 'go', markersize=12)
    axes[idx].plot(path[-1, 0], path[-1, 1], 'r*', markersize=15)
    axes[idx].plot(0, 0, 'b*', markersize=15)
    axes[idx].set_title(f'학습률 = {lr}\n최종 값: {f(path[-1,0], path[-1,1]):.4f}',
                        fontsize=11)
    axes[idx].set_xlabel('x')
    axes[idx].set_ylabel('y')
    axes[idx].grid(True, alpha=0.3)
    axes[idx].axis('equal')

plt.tight_layout()
plt.savefig('gradient_descent_learning_rates.png', dpi=150)
plt.close()

print("학습률 비교 시각화 저장: gradient_descent_learning_rates.png")
```

## 3. 방향 도함수 (Directional Derivative)

### 3.1 방향 도함수의 정의

단위 벡터 $\mathbf{u}$ 방향으로의 방향 도함수:

$$D_\mathbf{u} f(\mathbf{x}) = \lim_{h \to 0} \frac{f(\mathbf{x} + h\mathbf{u}) - f(\mathbf{x})}{h}$$

그래디언트와의 관계:

$$D_\mathbf{u} f = \nabla f \cdot \mathbf{u}$$

### 3.2 최대 변화 방향

방향 도함수는 $\mathbf{u}$가 $\nabla f$ 방향일 때 최대:

$$\max_{\|\mathbf{u}\|=1} D_\mathbf{u} f = \|\nabla f\|$$

```python
# 방향 도함수 시각화
point = np.array([1.5, 1.0])
grad_at_point = grad_f(point[0], point[1])
grad_norm = np.linalg.norm(grad_at_point)

# 다양한 방향
n_directions = 36
angles = np.linspace(0, 2*np.pi, n_directions)
directions = np.array([np.cos(angles), np.sin(angles)]).T

# 각 방향으로 방향 도함수 계산
directional_derivatives = []
for u in directions:
    Du_f = np.dot(grad_at_point, u)
    directional_derivatives.append(Du_f)

directional_derivatives = np.array(directional_derivatives)

# 극좌표 플롯
fig = plt.figure(figsize=(14, 6))

# 극좌표 플롯
ax1 = fig.add_subplot(121, projection='polar')
ax1.plot(angles, directional_derivatives, 'b-', linewidth=2)
ax1.fill(angles, directional_derivatives, alpha=0.3)
ax1.set_title('방향 도함수 크기\n(각도별)', fontsize=12)
ax1.grid(True)

# 2D 플롯
ax2 = fig.add_subplot(122)
ax2.contour(X, Y, Z, levels=15, cmap='viridis', alpha=0.6)
ax2.plot(point[0], point[1], 'ro', markersize=12, label='평가 점')

# 그래디언트 방향 (최대)
grad_unit = grad_at_point / grad_norm
ax2.arrow(point[0], point[1], grad_unit[0]*0.5, grad_unit[1]*0.5,
          head_width=0.15, head_length=0.1, fc='red', ec='red',
          linewidth=2, label=f'그래디언트 (최대: {grad_norm:.2f})')

# 몇 가지 다른 방향
sample_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
colors = ['blue', 'green', 'orange', 'purple']
for angle, color in zip(sample_angles, colors):
    u = np.array([np.cos(angle), np.sin(angle)])
    Du = np.dot(grad_at_point, u)
    ax2.arrow(point[0], point[1], u[0]*0.5, u[1]*0.5,
              head_width=0.1, head_length=0.08, fc=color, ec=color,
              alpha=0.6, linewidth=1.5)

ax2.set_title('다양한 방향의 도함수', fontsize=12)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.axis('equal')

plt.tight_layout()
plt.savefig('directional_derivative.png', dpi=150)
plt.close()

print("방향 도함수 시각화 저장: directional_derivative.png")
print(f"점 {point}에서 그래디언트: {grad_at_point}")
print(f"그래디언트 크기 (최대 방향 도함수): {grad_norm:.4f}")
```

## 4. 다변수 체인 룰 (Multivariate Chain Rule)

### 4.1 체인 룰의 형태

$z = f(x, y)$이고 $x = x(t)$, $y = y(t)$일 때:

$$\frac{dz}{dt} = \frac{\partial f}{\partial x} \frac{dx}{dt} + \frac{\partial f}{\partial y} \frac{dy}{dt}$$

일반적으로:

$$\frac{dz}{dt} = \nabla f \cdot \frac{d\mathbf{x}}{dt}$$

### 4.2 체인 룰 예제

```python
# 체인 룰: z = f(x, y) = x^2 + y^2, x(t) = cos(t), y(t) = sin(t)
t = symbols('t')
x_t = sp.cos(t)
y_t = sp.sin(t)

# 함수
z = x_t**2 + y_t**2

# 방법 1: 직접 미분
dz_dt_direct = diff(z, t)
print("z(t) =", simplify(z))
print(f"dz/dt (직접) = {simplify(dz_dt_direct)}")

# 방법 2: 체인 룰
x_sym, y_sym = symbols('x y')
f = x_sym**2 + y_sym**2
df_dx = diff(f, x_sym)
df_dy = diff(f, y_sym)
dx_dt = diff(x_t, t)
dy_dt = diff(y_t, t)

dz_dt_chain = df_dx.subs(x_sym, x_t).subs(y_sym, y_t) * dx_dt + \
              df_dy.subs(x_sym, x_t).subs(y_sym, y_t) * dy_dt

print(f"dz/dt (체인 룰) = {simplify(dz_dt_chain)}")
print(f"동일한가? {simplify(dz_dt_direct - dz_dt_chain) == 0}")
```

### 4.3 역전파와 체인 룰

신경망의 역전파는 다변수 체인 룰의 반복 적용입니다.

```python
import torch
import torch.nn as nn

# 간단한 계산 그래프
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# z = f(x, y) = x^2 + xy + y^2
z = x**2 + x*y + y**2

# 자동 미분
z.backward()

print("계산 그래프: z = x² + xy + y²")
print(f"x = {x.item()}, y = {y.item()}")
print(f"z = {z.item()}")
print(f"\n자동 미분:")
print(f"∂z/∂x = {x.grad.item()}")
print(f"∂z/∂y = {y.grad.item()}")

# 수동 계산
x_val, y_val = 2.0, 3.0
dz_dx_manual = 2*x_val + y_val  # ∂z/∂x = 2x + y
dz_dy_manual = x_val + 2*y_val  # ∂z/∂y = x + 2y

print(f"\n수동 계산:")
print(f"∂z/∂x = 2x + y = {dz_dx_manual}")
print(f"∂z/∂y = x + 2y = {dz_dy_manual}")
```

## 5. 테일러 전개 (Taylor Expansion)

### 5.1 1차 테일러 전개 (선형 근사)

점 $\mathbf{x}_0$ 근처에서:

$$f(\mathbf{x}_0 + \boldsymbol{\delta}) \approx f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^T \boldsymbol{\delta}$$

### 5.2 2차 테일러 전개 (이차 근사)

헤시안 $H$를 포함:

$$f(\mathbf{x}_0 + \boldsymbol{\delta}) \approx f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^T \boldsymbol{\delta} + \frac{1}{2} \boldsymbol{\delta}^T H(\mathbf{x}_0) \boldsymbol{\delta}$$

### 5.3 테일러 전개 시각화

```python
# 테일러 전개 근사
def f_example(x, y):
    return np.exp(-(x**2 + y**2)) * np.sin(x) * np.cos(y)

def grad_f_example(x, y):
    """수치 그래디언트"""
    h = 1e-5
    df_dx = (f_example(x+h, y) - f_example(x-h, y)) / (2*h)
    df_dy = (f_example(x, y+h) - f_example(x, y-h)) / (2*h)
    return np.array([df_dx, df_dy])

def hessian_f_example(x, y):
    """수치 헤시안"""
    h = 1e-5
    H = np.zeros((2, 2))
    grad_base = grad_f_example(x, y)

    # H[0,0] = ∂²f/∂x²
    grad_x_plus = grad_f_example(x+h, y)
    H[0, 0] = (grad_x_plus[0] - grad_base[0]) / h

    # H[1,1] = ∂²f/∂y²
    grad_y_plus = grad_f_example(x, y+h)
    H[1, 1] = (grad_y_plus[1] - grad_base[1]) / h

    # H[0,1] = H[1,0] = ∂²f/∂x∂y
    H[0, 1] = (grad_x_plus[1] - grad_base[1]) / h
    H[1, 0] = H[0, 1]

    return H

# 전개 중심점
x0, y0 = 0.5, 0.5
f0 = f_example(x0, y0)
grad0 = grad_f_example(x0, y0)
H0 = hessian_f_example(x0, y0)

# 격자
x_range = np.linspace(-0.5, 1.5, 100)
y_range = np.linspace(-0.5, 1.5, 100)
X_grid, Y_grid = np.meshgrid(x_range, y_range)

# 원 함수
Z_true = f_example(X_grid, Y_grid)

# 1차 근사
delta_x = X_grid - x0
delta_y = Y_grid - y0
Z_linear = f0 + grad0[0]*delta_x + grad0[1]*delta_y

# 2차 근사
Z_quadratic = np.zeros_like(Z_true)
for i in range(len(y_range)):
    for j in range(len(x_range)):
        delta = np.array([delta_x[i, j], delta_y[i, j]])
        Z_quadratic[i, j] = f0 + grad0 @ delta + 0.5 * delta @ H0 @ delta

# 시각화
fig = plt.figure(figsize=(18, 5))

titles = ['원 함수', '1차 테일러 근사 (선형)', '2차 테일러 근사 (이차)']
Z_list = [Z_true, Z_linear, Z_quadratic]

for idx, (title, Z) in enumerate(zip(titles, Z_list)):
    ax = fig.add_subplot(1, 3, idx+1, projection='3d')
    ax.plot_surface(X_grid, Y_grid, Z, cmap='viridis', alpha=0.8)
    ax.scatter([x0], [y0], [f0], c='red', s=100, label='전개 중심')
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x,y)')
    ax.legend()

plt.tight_layout()
plt.savefig('taylor_expansion.png', dpi=150)
plt.close()

# 오차 계산
error_linear = np.abs(Z_true - Z_linear).max()
error_quad = np.abs(Z_true - Z_quadratic).max()

print("테일러 전개 시각화 저장: taylor_expansion.png")
print(f"1차 근사 최대 오차: {error_linear:.6f}")
print(f"2차 근사 최대 오차: {error_quad:.6f}")
```

### 5.4 뉴턴 방법의 수학적 기초

뉴턴 방법은 2차 테일러 전개를 이용한 최적화입니다:

$$\mathbf{x}_{k+1} = \mathbf{x}_k - H^{-1}(\mathbf{x}_k) \nabla f(\mathbf{x}_k)$$

```python
def newton_method_2d(f, grad_f, hess_f, x0, tol=1e-6, max_iter=20):
    """2D 뉴턴 방법"""
    path = [x0]
    x = x0.copy()

    for i in range(max_iter):
        grad = grad_f(x[0], x[1])
        hess = hess_f(x[0], x[1])

        # 뉴턴 스텝
        try:
            delta = np.linalg.solve(hess, -grad)
        except np.linalg.LinAlgError:
            print(f"특이 헤시안 at iteration {i}")
            break

        x = x + delta
        path.append(x.copy())

        if np.linalg.norm(delta) < tol:
            print(f"수렴 at iteration {i}")
            break

    return np.array(path)

# 테스트 함수: f(x, y) = (x-1)^2 + 2(y-2)^2
def f_newton(x, y):
    return (x - 1)**2 + 2*(y - 2)**2

def grad_f_newton(x, y):
    return np.array([2*(x - 1), 4*(y - 2)])

def hess_f_newton(x, y):
    return np.array([[2, 0], [0, 4]])

# 뉴턴 방법 vs 그래디언트 하강
x0_newton = np.array([0.0, 0.0])
path_newton = newton_method_2d(f_newton, grad_f_newton, hess_f_newton, x0_newton)
path_gd = gradient_descent_2d(grad_f_newton, x0_newton, learning_rate=0.2, n_steps=50)

# 시각화
x_plt = np.linspace(-0.5, 2.5, 200)
y_plt = np.linspace(-0.5, 3.5, 200)
X_plt, Y_plt = np.meshgrid(x_plt, y_plt)
Z_plt = f_newton(X_plt, Y_plt)

plt.figure(figsize=(10, 8))
plt.contour(X_plt, Y_plt, Z_plt, levels=25, cmap='viridis', alpha=0.6)
plt.plot(path_gd[:, 0], path_gd[:, 1], 'ro-', linewidth=2, markersize=5,
         label=f'그래디언트 하강 ({len(path_gd)} 단계)')
plt.plot(path_newton[:, 0], path_newton[:, 1], 'bo-', linewidth=2, markersize=7,
         label=f'뉴턴 방법 ({len(path_newton)} 단계)')
plt.plot(1, 2, 'g*', markersize=20, label='최솟값')
plt.colorbar(label='f(x, y)')
plt.title('뉴턴 방법 vs 그래디언트 하강법', fontsize=14)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('newton_vs_gradient_descent.png', dpi=150)
plt.close()

print("뉴턴 vs GD 시각화 저장: newton_vs_gradient_descent.png")
print(f"뉴턴 방법 단계: {len(path_newton)}, 최종: {path_newton[-1]}")
print(f"GD 단계: {len(path_gd)}, 최종: {path_gd[-1]}")
```

## 6. 손실 지형 시각화 (Loss Landscape)

### 6.1 손실 함수의 지형

```python
# 비등방 손실 함수 (조건수가 큰 경우)
def loss_anisotropic(x, y):
    """길쭉한 골짜기 형태의 손실 함수"""
    return 0.5 * x**2 + 10 * y**2

def grad_loss_anisotropic(x, y):
    return np.array([x, 20*y])

# 시각화
x_loss = np.linspace(-10, 10, 300)
y_loss = np.linspace(-3, 3, 300)
X_loss, Y_loss = np.meshgrid(x_loss, y_loss)
Z_loss = loss_anisotropic(X_loss, Y_loss)

fig = plt.figure(figsize=(16, 6))

# 3D 표면
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X_loss, Y_loss, Z_loss, cmap='viridis', alpha=0.8,
                 vmin=0, vmax=50)
ax1.set_title('3D 손실 지형\n(높은 조건수)', fontsize=12)
ax1.set_xlabel('$w_1$')
ax1.set_ylabel('$w_2$')
ax1.set_zlabel('Loss')
ax1.view_init(elev=25, azim=45)

# 등고선
ax2 = fig.add_subplot(132)
contours = ax2.contour(X_loss, Y_loss, Z_loss, levels=30, cmap='viridis')
ax2.clabel(contours, inline=True, fontsize=8)
ax2.set_title('등고선 (길쭉한 골짜기)', fontsize=12)
ax2.set_xlabel('$w_1$')
ax2.set_ylabel('$w_2$')
ax2.axis('equal')
ax2.grid(True, alpha=0.3)

# 그래디언트 하강 경로
ax3 = fig.add_subplot(133)
ax3.contour(X_loss, Y_loss, Z_loss, levels=30, cmap='viridis', alpha=0.6)

x0_loss = np.array([8.0, 2.5])
path_slow = gradient_descent_2d(grad_loss_anisotropic, x0_loss,
                                 learning_rate=0.05, n_steps=100)
ax3.plot(path_slow[:, 0], path_slow[:, 1], 'r.-', linewidth=1.5,
         markersize=3, label='느린 수렴')
ax3.plot(x0_loss[0], x0_loss[1], 'go', markersize=10, label='시작')
ax3.plot(0, 0, 'b*', markersize=15, label='최솟값')
ax3.set_title('그래디언트 하강 경로\n(지그재그)', fontsize=12)
ax3.set_xlabel('$w_1$')
ax3.set_ylabel('$w_2$')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.axis('equal')

plt.tight_layout()
plt.savefig('loss_landscape_anisotropic.png', dpi=150)
plt.close()

print("비등방 손실 지형 시각화 저장: loss_landscape_anisotropic.png")
```

### 6.2 안장점 (Saddle Point)

```python
# 안장점 함수: f(x, y) = x^2 - y^2
def saddle_function(x, y):
    return x**2 - y**2

def grad_saddle(x, y):
    return np.array([2*x, -2*y])

# 시각화
x_saddle = np.linspace(-3, 3, 200)
y_saddle = np.linspace(-3, 3, 200)
X_saddle, Y_saddle = np.meshgrid(x_saddle, y_saddle)
Z_saddle = saddle_function(X_saddle, Y_saddle)

fig = plt.figure(figsize=(16, 6))

# 3D 표면
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X_saddle, Y_saddle, Z_saddle, cmap='coolwarm', alpha=0.8)
ax1.scatter([0], [0], [0], c='red', s=100, label='안장점')
ax1.set_title('안장점 함수: $f(x,y) = x^2 - y^2$', fontsize=12)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f')
ax1.legend()

# 등고선
ax2 = fig.add_subplot(132)
contours = ax2.contour(X_saddle, Y_saddle, Z_saddle, levels=30, cmap='coolwarm')
ax2.clabel(contours, inline=True, fontsize=8)
ax2.plot(0, 0, 'ro', markersize=10, label='안장점 (0, 0)')
ax2.set_title('등고선 (쌍곡선 형태)', fontsize=12)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.legend()
ax2.axis('equal')
ax2.grid(True, alpha=0.3)

# 다양한 시작점에서 GD
ax3 = fig.add_subplot(133)
ax3.contour(X_saddle, Y_saddle, Z_saddle, levels=30, cmap='coolwarm', alpha=0.6)

start_points = [np.array([2, 0.1]), np.array([0.1, 2]),
                np.array([-2, -0.1]), np.array([-0.1, -2])]
colors = ['red', 'blue', 'green', 'orange']

for start, color in zip(start_points, colors):
    path = gradient_descent_2d(grad_saddle, start, learning_rate=0.1, n_steps=30)
    ax3.plot(path[:, 0], path[:, 1], 'o-', color=color, linewidth=1.5,
             markersize=4, alpha=0.7)

ax3.plot(0, 0, 'r*', markersize=20, label='안장점')
ax3.set_title('안장점 근처 GD 경로\n(불안정)', fontsize=12)
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.axis('equal')

plt.tight_layout()
plt.savefig('saddle_point.png', dpi=150)
plt.close()

print("안장점 시각화 저장: saddle_point.png")
```

### 6.3 조건수 (Condition Number)와 수렴 속도

조건수는 헤시안의 최대/최소 고유값 비율:

$$\kappa(H) = \frac{\lambda_{\max}(H)}{\lambda_{\min}(H)}$$

조건수가 클수록 최적화가 어렵습니다.

```python
# 조건수 비교
def well_conditioned(x, y):
    """조건수 = 1"""
    return x**2 + y**2

def ill_conditioned(x, y):
    """조건수 = 100"""
    return x**2 + 100*y**2

def grad_well(x, y):
    return np.array([2*x, 2*y])

def grad_ill(x, y):
    return np.array([2*x, 200*y])

# 헤시안과 조건수
H_well = np.array([[2, 0], [0, 2]])
H_ill = np.array([[2, 0], [0, 200]])

eigs_well = np.linalg.eigvals(H_well)
eigs_ill = np.linalg.eigvals(H_ill)

cond_well = np.max(eigs_well) / np.min(eigs_well)
cond_ill = np.max(eigs_ill) / np.min(eigs_ill)

print(f"Well-conditioned 조건수: {cond_well:.2f}")
print(f"Ill-conditioned 조건수:  {cond_ill:.2f}")

# GD 경로 비교
x0_cond = np.array([3.0, 3.0])
path_well = gradient_descent_2d(grad_well, x0_cond, learning_rate=0.2, n_steps=30)
path_ill = gradient_descent_2d(grad_ill, x0_cond, learning_rate=0.005, n_steps=100)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Well-conditioned
x_grid = np.linspace(-4, 4, 200)
y_grid = np.linspace(-4, 4, 200)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
Z_well = well_conditioned(X_grid, Y_grid)

axes[0].contour(X_grid, Y_grid, Z_well, levels=20, cmap='viridis', alpha=0.6)
axes[0].plot(path_well[:, 0], path_well[:, 1], 'ro-', linewidth=2, markersize=4)
axes[0].set_title(f'Well-conditioned (κ={cond_well:.1f})\n{len(path_well)} 단계',
                  fontsize=12)
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].axis('equal')
axes[0].grid(True, alpha=0.3)

# Ill-conditioned
Z_ill = ill_conditioned(X_grid, Y_grid)
axes[1].contour(X_grid, Y_grid, Z_ill, levels=30, cmap='viridis', alpha=0.6)
axes[1].plot(path_ill[:, 0], path_ill[:, 1], 'ro-', linewidth=1.5, markersize=3)
axes[1].set_title(f'Ill-conditioned (κ={cond_ill:.1f})\n{len(path_ill)} 단계',
                  fontsize=12)
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].axis('equal')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('condition_number.png', dpi=150)
plt.close()

print("조건수 비교 시각화 저장: condition_number.png")
```

## 7. ML 응용

### 7.1 볼록성 분석

볼록 함수는 헤시안이 양반정치 (positive semi-definite)입니다.

```python
# MSE 손실의 볼록성 확인
from sklearn.datasets import make_regression

X_reg, y_reg = make_regression(n_samples=100, n_features=2, noise=10, random_state=42)

def mse_loss(w):
    """MSE 손실: L(w) = (1/2n) ||y - Xw||^2"""
    pred = X_reg @ w
    return 0.5 * np.mean((y_reg - pred)**2)

def grad_mse(w):
    """그래디언트: ∇L = -(1/n) X^T (y - Xw)"""
    residual = y_reg - X_reg @ w
    return -X_reg.T @ residual / len(y_reg)

def hess_mse(w):
    """헤시안: H = (1/n) X^T X"""
    return X_reg.T @ X_reg / len(y_reg)

# 헤시안의 고유값 (볼록성 확인)
w_sample = np.random.randn(2)
H = hess_mse(w_sample)
eigenvalues = np.linalg.eigvalsh(H)

print("MSE 손실 함수 볼록성 분석:")
print(f"헤시안의 고유값: {eigenvalues}")
print(f"모두 양수? {np.all(eigenvalues >= -1e-10)}")
print("→ MSE는 볼록 함수 (전역 최솟값 보장)")
```

### 7.2 그래디언트 기반 최적화 비교

```python
# 다양한 최적화 알고리즘 비교
from scipy.optimize import minimize

# 시작점
w0 = np.array([2.0, -1.5])

# 최적화 알고리즘
methods = ['BFGS', 'CG', 'Newton-CG', 'L-BFGS-B']
results = {}

for method in methods:
    if method == 'Newton-CG':
        result = minimize(mse_loss, w0, method=method, jac=grad_mse,
                          hess=hess_mse, options={'disp': False})
    else:
        result = minimize(mse_loss, w0, method=method, jac=grad_mse,
                          options={'disp': False})

    results[method] = result

    print(f"\n{method}:")
    print(f"  최적 w: {result.x}")
    print(f"  최종 손실: {result.fun:.6f}")
    print(f"  반복 횟수: {result.nit}")
    print(f"  함수 평가: {result.nfev}")

# 해석해와 비교
w_analytic = np.linalg.lstsq(X_reg, y_reg, rcond=None)[0]
print(f"\n해석해 (Normal Equation): {w_analytic}")
```

## 연습 문제

### 문제 1: 임계점 분류
함수 $f(x, y) = x^3 - 3xy^2$에 대해:
1. 그래디언트를 계산하고 임계점 (critical point)을 찾으시오
2. 헤시안을 계산하고 각 임계점의 성질 (극값/안장점)을 판정하시오
3. 3D로 시각화하여 검증하시오

### 문제 2: 제약 최적화
등고선과 제약 조건 $x^2 + y^2 = 1$을 시각화하고, 라그랑주 승수법을 사용하여 다음을 최적화하시오:

$$\min_{x,y} f(x, y) = x^2 + 2y^2 \quad \text{s.t.} \quad x^2 + y^2 = 1$$

해석해와 수치해를 비교하시오.

### 문제 3: 모멘텀 경사 하강법
모멘텀을 추가한 경사 하강법을 구현하시오:

$$\mathbf{v}_{t+1} = \beta \mathbf{v}_t - \alpha \nabla f(\mathbf{x}_t)$$
$$\mathbf{x}_{t+1} = \mathbf{x}_t + \mathbf{v}_{t+1}$$

조건수가 큰 함수에서 일반 GD와 성능을 비교하시오.

### 문제 4: 선형 회귀 정규 방정식
테일러 전개를 이용하여 MSE 손실의 정규 방정식을 유도하시오:

$$\mathbf{w}^* = (X^T X)^{-1} X^T \mathbf{y}$$

힌트: $\nabla L = 0$을 풀고, 헤시안이 양정치임을 확인하시오.

### 문제 5: Adam 옵티마이저 구현
Adam 옵티마이저의 업데이트 규칙을 구현하고, 비등방 손실 함수에서 일반 GD, Momentum, Adam을 비교하시오. 수렴 속도와 경로를 시각화하시오.

## 참고 자료

### 온라인 자료
- [Multivariable Calculus - MIT OpenCourseWare](https://ocw.mit.edu/courses/mathematics/18-02sc-multivariable-calculus-fall-2010/)
- [Visual Calculus](https://visualcalculus.com/) - 대화형 시각화
- [Gradient Descent Visualization](https://distill.pub/2017/momentum/) - Distill.pub

### 교재
- Stewart, *Calculus: Early Transcendentals*, Chapters 14-15
- Strang, *Calculus*, Volume 3 (Multivariable)
- Boyd & Vandenberghe, *Convex Optimization*, Appendix A

### 논문 및 튜토리얼
- Ruder, *An Overview of Gradient Descent Optimization Algorithms* (2016)
- Goodfellow et al., *Deep Learning*, Chapter 4 (Numerical Computation)
- Nocedal & Wright, *Numerical Optimization*, Chapter 2 (Fundamentals)
