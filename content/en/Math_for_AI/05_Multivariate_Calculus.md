# 05. Multivariate Calculus

## Learning Objectives

- Understand and compute the geometric meaning of partial derivatives
- Visualize and interpret the relationship between gradient vectors and contour lines
- Calculate directional derivatives and understand their relationship to gradients
- Apply the multivariate chain rule to differentiate complex composite functions
- Understand the principles of function approximation and Newton's method using Taylor expansion
- Visualize loss landscapes in machine learning and connect them to optimization problems

---

## 1. Partial Derivatives

### 1.1 Definition of Partial Derivative

For a multivariate function $f(x_1, x_2, \ldots, x_n)$, the partial derivative with respect to $x_i$ is the rate of change when only $x_i$ varies while other variables remain constant:

$$\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \ldots, x_i + h, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}$$

### 1.2 Examples of Partial Derivative Computation

**Example 1**: $f(x, y) = x^2 + 3xy + y^2$

$$\frac{\partial f}{\partial x} = 2x + 3y$$

$$\frac{\partial f}{\partial y} = 3x + 2y$$

**Example 2**: $f(x, y) = e^{x^2 + y^2}$

$$\frac{\partial f}{\partial x} = 2x e^{x^2 + y^2}$$

$$\frac{\partial f}{\partial y} = 2y e^{x^2 + y^2}$$

### 1.3 Symbolic Differentiation with SymPy

```python
import numpy as np
import sympy as sp
from sympy import symbols, diff, exp, sin, cos, simplify

# Define symbols
x, y = symbols('x y')

# Define function
f = x**2 + 3*x*y + y**2

# Partial derivatives
df_dx = diff(f, x)
df_dy = diff(f, y)

print("Function:", f)
print(f"∂f/∂x = {df_dx}")
print(f"∂f/∂y = {df_dy}")

# Evaluate at a specific point
point = {x: 1, y: 2}
print(f"\nAt point (1, 2):")
print(f"  f(1,2) = {f.subs(point)}")
print(f"  ∂f/∂x(1,2) = {df_dx.subs(point)}")
print(f"  ∂f/∂y(1,2) = {df_dy.subs(point)}")

# More complex function
g = exp(x**2 + y**2) * sin(x*y)
dg_dx = simplify(diff(g, x))
dg_dy = simplify(diff(g, y))

print(f"\nFunction: g(x,y) = {g}")
print(f"∂g/∂x = {dg_dx}")
print(f"∂g/∂y = {dg_dy}")
```

### 1.4 Geometric Meaning of Partial Derivatives

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define function: f(x, y) = x^2 + y^2
def f(x, y):
    return x**2 + y**2

# Create grid
x_vals = np.linspace(-3, 3, 100)
y_vals = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = f(X, Y)

# Specific point
x0, y0 = 1.0, 1.5
z0 = f(x0, y0)

fig = plt.figure(figsize=(16, 6))

# 3D surface
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
ax1.scatter([x0], [y0], [z0], c='red', s=100, label=f'Point ({x0}, {y0})')

# Tangent in x direction (y fixed)
x_line = np.linspace(x0-1, x0+1, 50)
y_line = np.full_like(x_line, y0)
z_line = f(x_line, y_line)
ax1.plot(x_line, y_line, z_line, 'r-', linewidth=3, label='y fixed (∂f/∂x)')

# Tangent in y direction (x fixed)
x_line2 = np.full(50, x0)
y_line2 = np.linspace(y0-1, y0+1, 50)
z_line2 = f(x_line2, y_line2)
ax1.plot(x_line2, y_line2, z_line2, 'b-', linewidth=3, label='x fixed (∂f/∂y)')

ax1.set_title('Geometric Meaning of Partial Derivatives', fontsize=12)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x,y)')
ax1.legend()

# Cross section with x fixed
ax2 = fig.add_subplot(132)
y_slice = np.linspace(-3, 3, 100)
z_slice = f(x0, y_slice)
ax2.plot(y_slice, z_slice, 'b-', linewidth=2)
ax2.plot([y0], [z0], 'ro', markersize=10)
# Tangent line
slope_y = 2*y0  # ∂f/∂y = 2y
tangent_y = z0 + slope_y * (y_slice - y0)
ax2.plot(y_slice, tangent_y, 'r--', linewidth=2, label=f'Tangent (slope={slope_y:.1f})')
ax2.set_title(f'Cross section with x={x0} fixed', fontsize=12)
ax2.set_xlabel('y')
ax2.set_ylabel('f(x,y)')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Cross section with y fixed
ax3 = fig.add_subplot(133)
x_slice = np.linspace(-3, 3, 100)
z_slice = f(x_slice, y0)
ax3.plot(x_slice, z_slice, 'r-', linewidth=2)
ax3.plot([x0], [z0], 'ro', markersize=10)
# Tangent line
slope_x = 2*x0  # ∂f/∂x = 2x
tangent_x = z0 + slope_x * (x_slice - x0)
ax3.plot(x_slice, tangent_x, 'b--', linewidth=2, label=f'Tangent (slope={slope_x:.1f})')
ax3.set_title(f'Cross section with y={y0} fixed', fontsize=12)
ax3.set_xlabel('x')
ax3.set_ylabel('f(x,y)')
ax3.grid(True, alpha=0.3)
ax3.legend()

plt.tight_layout()
plt.savefig('partial_derivatives_geometry.png', dpi=150, bbox_inches='tight')
plt.close()

print("Partial derivative geometry visualization saved: partial_derivatives_geometry.png")
```

### 1.5 Higher-Order Partial Derivatives

Schwarz's theorem: If $f$ is $C^2$ class, the order of partial derivatives doesn't matter

$$\frac{\partial^2 f}{\partial x \partial y} = \frac{\partial^2 f}{\partial y \partial x}$$

```python
# Verify Schwarz's theorem
x, y = symbols('x y')
f = x**3 * y**2 + x * y**3

# Mixed partial derivatives
f_xy = diff(diff(f, x), y)
f_yx = diff(diff(f, y), x)

print("Function:", f)
print(f"∂²f/∂x∂y = {f_xy}")
print(f"∂²f/∂y∂x = {f_yx}")
print(f"Are they equal? {simplify(f_xy - f_yx) == 0}")
```

## 2. Gradient

### 2.1 Gradient Vector

The gradient is a vector containing all partial derivatives:

$$\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

The gradient indicates:
1. **Direction**: The direction of steepest increase of the function
2. **Magnitude**: The rate of change in that direction

### 2.2 Contour Lines and Gradient

The gradient is perpendicular to contour lines (level sets).

```python
# Visualize contour lines and gradient
def f(x, y):
    """Function: f(x, y) = x^2 + 2y^2"""
    return x**2 + 2*y**2

def grad_f(x, y):
    """Gradient: ∇f = [2x, 4y]"""
    return np.array([2*x, 4*y])

# Grid
x = np.linspace(-3, 3, 300)
y = np.linspace(-3, 3, 300)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Points to compute gradient at
n_points = 15
x_grad = np.linspace(-2.5, 2.5, n_points)
y_grad = np.linspace(-2.5, 2.5, n_points)
X_grad, Y_grad = np.meshgrid(x_grad, y_grad)

# Gradient at each point
U = 2 * X_grad
V = 4 * Y_grad

# Normalize by gradient magnitude (adjust arrow length)
norm = np.sqrt(U**2 + V**2)
U_norm = U / (norm + 1e-8) * 0.3
V_norm = V / (norm + 1e-8) * 0.3

plt.figure(figsize=(10, 8))
# Contour lines
contours = plt.contour(X, Y, Z, levels=15, cmap='viridis', alpha=0.6)
plt.clabel(contours, inline=True, fontsize=8)
# Gradient vectors
plt.quiver(X_grad, Y_grad, U_norm, V_norm, norm, cmap='Reds',
           scale=1, scale_units='xy', width=0.004)
plt.colorbar(label='Gradient magnitude')
plt.title('Contour Lines and Gradient (Gradient ⊥ Contours)', fontsize=14)
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('gradient_contour.png', dpi=150)
plt.close()

print("Contour-gradient visualization saved: gradient_contour.png")
```

### 2.3 Gradient Descent Visualization

```python
# Gradient descent path
def gradient_descent_2d(grad_f, x0, learning_rate=0.1, n_steps=50):
    """2D gradient descent"""
    path = [x0]
    x = x0.copy()

    for _ in range(n_steps):
        grad = grad_f(x[0], x[1])
        x = x - learning_rate * grad
        path.append(x.copy())

    return np.array(path)

# Initial point
x0 = np.array([2.5, 2.0])

# Compute path
path = gradient_descent_2d(grad_f, x0, learning_rate=0.15, n_steps=30)

# Visualization
plt.figure(figsize=(10, 8))
plt.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
plt.plot(path[:, 0], path[:, 1], 'ro-', linewidth=2, markersize=6,
         label='Gradient descent path')
plt.plot(path[0, 0], path[0, 1], 'go', markersize=12, label='Start point')
plt.plot(path[-1, 0], path[-1, 1], 'r*', markersize=20, label='End point')
plt.plot(0, 0, 'b*', markersize=20, label='Minimum')
plt.colorbar(label='f(x, y)')
plt.title('Gradient Descent Path', fontsize=14)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.savefig('gradient_descent_path.png', dpi=150)
plt.close()

print("Gradient descent path saved: gradient_descent_path.png")
print(f"Start: {path[0]}, End: {path[-1]}, Minimum: [0, 0]")
print(f"Final function value: {f(path[-1, 0], path[-1, 1]):.6f}")
```

### 2.4 Gradient Magnitude and Learning Rate

```python
# Experiment with various learning rates
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
    axes[idx].set_title(f'Learning rate = {lr}\nFinal value: {f(path[-1,0], path[-1,1]):.4f}',
                        fontsize=11)
    axes[idx].set_xlabel('x')
    axes[idx].set_ylabel('y')
    axes[idx].grid(True, alpha=0.3)
    axes[idx].axis('equal')

plt.tight_layout()
plt.savefig('gradient_descent_learning_rates.png', dpi=150)
plt.close()

print("Learning rate comparison visualization saved: gradient_descent_learning_rates.png")
```

## 3. Directional Derivative

### 3.1 Definition of Directional Derivative

The directional derivative in the direction of unit vector $\mathbf{u}$:

$$D_\mathbf{u} f(\mathbf{x}) = \lim_{h \to 0} \frac{f(\mathbf{x} + h\mathbf{u}) - f(\mathbf{x})}{h}$$

Relationship to gradient:

$$D_\mathbf{u} f = \nabla f \cdot \mathbf{u}$$

### 3.2 Direction of Maximum Change

The directional derivative is maximum when $\mathbf{u}$ points in the direction of $\nabla f$:

$$\max_{\|\mathbf{u}\|=1} D_\mathbf{u} f = \|\nabla f\|$$

```python
# Directional derivative visualization
point = np.array([1.5, 1.0])
grad_at_point = grad_f(point[0], point[1])
grad_norm = np.linalg.norm(grad_at_point)

# Various directions
n_directions = 36
angles = np.linspace(0, 2*np.pi, n_directions)
directions = np.array([np.cos(angles), np.sin(angles)]).T

# Compute directional derivative in each direction
directional_derivatives = []
for u in directions:
    Du_f = np.dot(grad_at_point, u)
    directional_derivatives.append(Du_f)

directional_derivatives = np.array(directional_derivatives)

# Polar plot
fig = plt.figure(figsize=(14, 6))

# Polar plot
ax1 = fig.add_subplot(121, projection='polar')
ax1.plot(angles, directional_derivatives, 'b-', linewidth=2)
ax1.fill(angles, directional_derivatives, alpha=0.3)
ax1.set_title('Directional Derivative Magnitude\n(by angle)', fontsize=12)
ax1.grid(True)

# 2D plot
ax2 = fig.add_subplot(122)
ax2.contour(X, Y, Z, levels=15, cmap='viridis', alpha=0.6)
ax2.plot(point[0], point[1], 'ro', markersize=12, label='Evaluation point')

# Gradient direction (maximum)
grad_unit = grad_at_point / grad_norm
ax2.arrow(point[0], point[1], grad_unit[0]*0.5, grad_unit[1]*0.5,
          head_width=0.15, head_length=0.1, fc='red', ec='red',
          linewidth=2, label=f'Gradient (max: {grad_norm:.2f})')

# Several other directions
sample_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
colors = ['blue', 'green', 'orange', 'purple']
for angle, color in zip(sample_angles, colors):
    u = np.array([np.cos(angle), np.sin(angle)])
    Du = np.dot(grad_at_point, u)
    ax2.arrow(point[0], point[1], u[0]*0.5, u[1]*0.5,
              head_width=0.1, head_length=0.08, fc=color, ec=color,
              alpha=0.6, linewidth=1.5)

ax2.set_title('Derivatives in Various Directions', fontsize=12)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.axis('equal')

plt.tight_layout()
plt.savefig('directional_derivative.png', dpi=150)
plt.close()

print("Directional derivative visualization saved: directional_derivative.png")
print(f"Gradient at point {point}: {grad_at_point}")
print(f"Gradient magnitude (max directional derivative): {grad_norm:.4f}")
```

## 4. Multivariate Chain Rule

### 4.1 Forms of Chain Rule

When $z = f(x, y)$ and $x = x(t)$, $y = y(t)$:

$$\frac{dz}{dt} = \frac{\partial f}{\partial x} \frac{dx}{dt} + \frac{\partial f}{\partial y} \frac{dy}{dt}$$

In general:

$$\frac{dz}{dt} = \nabla f \cdot \frac{d\mathbf{x}}{dt}$$

### 4.2 Chain Rule Example

```python
# Chain rule: z = f(x, y) = x^2 + y^2, x(t) = cos(t), y(t) = sin(t)
t = symbols('t')
x_t = sp.cos(t)
y_t = sp.sin(t)

# Function
z = x_t**2 + y_t**2

# Method 1: Direct differentiation
dz_dt_direct = diff(z, t)
print("z(t) =", simplify(z))
print(f"dz/dt (direct) = {simplify(dz_dt_direct)}")

# Method 2: Chain rule
x_sym, y_sym = symbols('x y')
f = x_sym**2 + y_sym**2
df_dx = diff(f, x_sym)
df_dy = diff(f, y_sym)
dx_dt = diff(x_t, t)
dy_dt = diff(y_t, t)

dz_dt_chain = df_dx.subs(x_sym, x_t).subs(y_sym, y_t) * dx_dt + \
              df_dy.subs(x_sym, x_t).subs(y_sym, y_t) * dy_dt

print(f"dz/dt (chain rule) = {simplify(dz_dt_chain)}")
print(f"Are they equal? {simplify(dz_dt_direct - dz_dt_chain) == 0}")
```

### 4.3 Backpropagation and Chain Rule

Backpropagation in neural networks is the repeated application of the multivariate chain rule.

```python
import torch
import torch.nn as nn

# Simple computation graph
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# z = f(x, y) = x^2 + xy + y^2
z = x**2 + x*y + y**2

# Automatic differentiation
z.backward()

print("Computation graph: z = x² + xy + y²")
print(f"x = {x.item()}, y = {y.item()}")
print(f"z = {z.item()}")
print(f"\nAutomatic differentiation:")
print(f"∂z/∂x = {x.grad.item()}")
print(f"∂z/∂y = {y.grad.item()}")

# Manual computation
x_val, y_val = 2.0, 3.0
dz_dx_manual = 2*x_val + y_val  # ∂z/∂x = 2x + y
dz_dy_manual = x_val + 2*y_val  # ∂z/∂y = x + 2y

print(f"\nManual computation:")
print(f"∂z/∂x = 2x + y = {dz_dx_manual}")
print(f"∂z/∂y = x + 2y = {dz_dy_manual}")
```

## 5. Taylor Expansion

### 5.1 First-Order Taylor Expansion (Linear Approximation)

Near point $\mathbf{x}_0$:

$$f(\mathbf{x}_0 + \boldsymbol{\delta}) \approx f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^T \boldsymbol{\delta}$$

### 5.2 Second-Order Taylor Expansion (Quadratic Approximation)

Including Hessian $H$:

$$f(\mathbf{x}_0 + \boldsymbol{\delta}) \approx f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^T \boldsymbol{\delta} + \frac{1}{2} \boldsymbol{\delta}^T H(\mathbf{x}_0) \boldsymbol{\delta}$$

### 5.3 Taylor Expansion Visualization

```python
# Taylor expansion approximation
def f_example(x, y):
    return np.exp(-(x**2 + y**2)) * np.sin(x) * np.cos(y)

def grad_f_example(x, y):
    """Numerical gradient"""
    h = 1e-5
    df_dx = (f_example(x+h, y) - f_example(x-h, y)) / (2*h)
    df_dy = (f_example(x, y+h) - f_example(x, y-h)) / (2*h)
    return np.array([df_dx, df_dy])

def hessian_f_example(x, y):
    """Numerical Hessian"""
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

# Expansion center point
x0, y0 = 0.5, 0.5
f0 = f_example(x0, y0)
grad0 = grad_f_example(x0, y0)
H0 = hessian_f_example(x0, y0)

# Grid
x_range = np.linspace(-0.5, 1.5, 100)
y_range = np.linspace(-0.5, 1.5, 100)
X_grid, Y_grid = np.meshgrid(x_range, y_range)

# Original function
Z_true = f_example(X_grid, Y_grid)

# 1st-order approximation
delta_x = X_grid - x0
delta_y = Y_grid - y0
Z_linear = f0 + grad0[0]*delta_x + grad0[1]*delta_y

# 2nd-order approximation
Z_quadratic = np.zeros_like(Z_true)
for i in range(len(y_range)):
    for j in range(len(x_range)):
        delta = np.array([delta_x[i, j], delta_y[i, j]])
        Z_quadratic[i, j] = f0 + grad0 @ delta + 0.5 * delta @ H0 @ delta

# Visualization
fig = plt.figure(figsize=(18, 5))

titles = ['Original Function', '1st-order Taylor Approximation (Linear)', '2nd-order Taylor Approximation (Quadratic)']
Z_list = [Z_true, Z_linear, Z_quadratic]

for idx, (title, Z) in enumerate(zip(titles, Z_list)):
    ax = fig.add_subplot(1, 3, idx+1, projection='3d')
    ax.plot_surface(X_grid, Y_grid, Z, cmap='viridis', alpha=0.8)
    ax.scatter([x0], [y0], [f0], c='red', s=100, label='Expansion center')
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x,y)')
    ax.legend()

plt.tight_layout()
plt.savefig('taylor_expansion.png', dpi=150)
plt.close()

# Compute errors
error_linear = np.abs(Z_true - Z_linear).max()
error_quad = np.abs(Z_true - Z_quadratic).max()

print("Taylor expansion visualization saved: taylor_expansion.png")
print(f"1st-order approximation max error: {error_linear:.6f}")
print(f"2nd-order approximation max error: {error_quad:.6f}")
```

### 5.4 Mathematical Foundation of Newton's Method

Newton's method is optimization using second-order Taylor expansion:

$$\mathbf{x}_{k+1} = \mathbf{x}_k - H^{-1}(\mathbf{x}_k) \nabla f(\mathbf{x}_k)$$

```python
def newton_method_2d(f, grad_f, hess_f, x0, tol=1e-6, max_iter=20):
    """2D Newton's method"""
    path = [x0]
    x = x0.copy()

    for i in range(max_iter):
        grad = grad_f(x[0], x[1])
        hess = hess_f(x[0], x[1])

        # Newton step
        try:
            delta = np.linalg.solve(hess, -grad)
        except np.linalg.LinAlgError:
            print(f"Singular Hessian at iteration {i}")
            break

        x = x + delta
        path.append(x.copy())

        if np.linalg.norm(delta) < tol:
            print(f"Converged at iteration {i}")
            break

    return np.array(path)

# Test function: f(x, y) = (x-1)^2 + 2(y-2)^2
def f_newton(x, y):
    return (x - 1)**2 + 2*(y - 2)**2

def grad_f_newton(x, y):
    return np.array([2*(x - 1), 4*(y - 2)])

def hess_f_newton(x, y):
    return np.array([[2, 0], [0, 4]])

# Newton's method vs gradient descent
x0_newton = np.array([0.0, 0.0])
path_newton = newton_method_2d(f_newton, grad_f_newton, hess_f_newton, x0_newton)
path_gd = gradient_descent_2d(grad_f_newton, x0_newton, learning_rate=0.2, n_steps=50)

# Visualization
x_plt = np.linspace(-0.5, 2.5, 200)
y_plt = np.linspace(-0.5, 3.5, 200)
X_plt, Y_plt = np.meshgrid(x_plt, y_plt)
Z_plt = f_newton(X_plt, Y_plt)

plt.figure(figsize=(10, 8))
plt.contour(X_plt, Y_plt, Z_plt, levels=25, cmap='viridis', alpha=0.6)
plt.plot(path_gd[:, 0], path_gd[:, 1], 'ro-', linewidth=2, markersize=5,
         label=f'Gradient descent ({len(path_gd)} steps)')
plt.plot(path_newton[:, 0], path_newton[:, 1], 'bo-', linewidth=2, markersize=7,
         label=f"Newton's method ({len(path_newton)} steps)")
plt.plot(1, 2, 'g*', markersize=20, label='Minimum')
plt.colorbar(label='f(x, y)')
plt.title("Newton's Method vs Gradient Descent", fontsize=14)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('newton_vs_gradient_descent.png', dpi=150)
plt.close()

print("Newton vs GD visualization saved: newton_vs_gradient_descent.png")
print(f"Newton's method steps: {len(path_newton)}, final: {path_newton[-1]}")
print(f"GD steps: {len(path_gd)}, final: {path_gd[-1]}")
```

## 6. Loss Landscape Visualization

### 6.1 Loss Function Landscape

```python
# Anisotropic loss function (large condition number)
def loss_anisotropic(x, y):
    """Loss function shaped like an elongated valley"""
    return 0.5 * x**2 + 10 * y**2

def grad_loss_anisotropic(x, y):
    return np.array([x, 20*y])

# Visualization
x_loss = np.linspace(-10, 10, 300)
y_loss = np.linspace(-3, 3, 300)
X_loss, Y_loss = np.meshgrid(x_loss, y_loss)
Z_loss = loss_anisotropic(X_loss, Y_loss)

fig = plt.figure(figsize=(16, 6))

# 3D surface
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X_loss, Y_loss, Z_loss, cmap='viridis', alpha=0.8,
                 vmin=0, vmax=50)
ax1.set_title('3D Loss Landscape\n(High condition number)', fontsize=12)
ax1.set_xlabel('$w_1$')
ax1.set_ylabel('$w_2$')
ax1.set_zlabel('Loss')
ax1.view_init(elev=25, azim=45)

# Contour lines
ax2 = fig.add_subplot(132)
contours = ax2.contour(X_loss, Y_loss, Z_loss, levels=30, cmap='viridis')
ax2.clabel(contours, inline=True, fontsize=8)
ax2.set_title('Contour Lines (Elongated Valley)', fontsize=12)
ax2.set_xlabel('$w_1$')
ax2.set_ylabel('$w_2$')
ax2.axis('equal')
ax2.grid(True, alpha=0.3)

# Gradient descent path
ax3 = fig.add_subplot(133)
ax3.contour(X_loss, Y_loss, Z_loss, levels=30, cmap='viridis', alpha=0.6)

x0_loss = np.array([8.0, 2.5])
path_slow = gradient_descent_2d(grad_loss_anisotropic, x0_loss,
                                 learning_rate=0.05, n_steps=100)
ax3.plot(path_slow[:, 0], path_slow[:, 1], 'r.-', linewidth=1.5,
         markersize=3, label='Slow convergence')
ax3.plot(x0_loss[0], x0_loss[1], 'go', markersize=10, label='Start')
ax3.plot(0, 0, 'b*', markersize=15, label='Minimum')
ax3.set_title('Gradient Descent Path\n(Zigzag)', fontsize=12)
ax3.set_xlabel('$w_1$')
ax3.set_ylabel('$w_2$')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.axis('equal')

plt.tight_layout()
plt.savefig('loss_landscape_anisotropic.png', dpi=150)
plt.close()

print("Anisotropic loss landscape visualization saved: loss_landscape_anisotropic.png")
```

### 6.2 Saddle Point

```python
# Saddle point function: f(x, y) = x^2 - y^2
def saddle_function(x, y):
    return x**2 - y**2

def grad_saddle(x, y):
    return np.array([2*x, -2*y])

# Visualization
x_saddle = np.linspace(-3, 3, 200)
y_saddle = np.linspace(-3, 3, 200)
X_saddle, Y_saddle = np.meshgrid(x_saddle, y_saddle)
Z_saddle = saddle_function(X_saddle, Y_saddle)

fig = plt.figure(figsize=(16, 6))

# 3D surface
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X_saddle, Y_saddle, Z_saddle, cmap='coolwarm', alpha=0.8)
ax1.scatter([0], [0], [0], c='red', s=100, label='Saddle point')
ax1.set_title('Saddle Point Function: $f(x,y) = x^2 - y^2$', fontsize=12)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f')
ax1.legend()

# Contour lines
ax2 = fig.add_subplot(132)
contours = ax2.contour(X_saddle, Y_saddle, Z_saddle, levels=30, cmap='coolwarm')
ax2.clabel(contours, inline=True, fontsize=8)
ax2.plot(0, 0, 'ro', markersize=10, label='Saddle point (0, 0)')
ax2.set_title('Contour Lines (Hyperbolic Shape)', fontsize=12)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.legend()
ax2.axis('equal')
ax2.grid(True, alpha=0.3)

# GD from various starting points
ax3 = fig.add_subplot(133)
ax3.contour(X_saddle, Y_saddle, Z_saddle, levels=30, cmap='coolwarm', alpha=0.6)

start_points = [np.array([2, 0.1]), np.array([0.1, 2]),
                np.array([-2, -0.1]), np.array([-0.1, -2])]
colors = ['red', 'blue', 'green', 'orange']

for start, color in zip(start_points, colors):
    path = gradient_descent_2d(grad_saddle, start, learning_rate=0.1, n_steps=30)
    ax3.plot(path[:, 0], path[:, 1], 'o-', color=color, linewidth=1.5,
             markersize=4, alpha=0.7)

ax3.plot(0, 0, 'r*', markersize=20, label='Saddle point')
ax3.set_title('GD Paths Near Saddle Point\n(Unstable)', fontsize=12)
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.axis('equal')

plt.tight_layout()
plt.savefig('saddle_point.png', dpi=150)
plt.close()

print("Saddle point visualization saved: saddle_point.png")
```

### 6.3 Condition Number and Convergence Rate

The condition number is the ratio of maximum to minimum eigenvalues of the Hessian:

$$\kappa(H) = \frac{\lambda_{\max}(H)}{\lambda_{\min}(H)}$$

Higher condition number makes optimization more difficult.

```python
# Condition number comparison
def well_conditioned(x, y):
    """Condition number = 1"""
    return x**2 + y**2

def ill_conditioned(x, y):
    """Condition number = 100"""
    return x**2 + 100*y**2

def grad_well(x, y):
    return np.array([2*x, 2*y])

def grad_ill(x, y):
    return np.array([2*x, 200*y])

# Hessian and condition number
H_well = np.array([[2, 0], [0, 2]])
H_ill = np.array([[2, 0], [0, 200]])

eigs_well = np.linalg.eigvals(H_well)
eigs_ill = np.linalg.eigvals(H_ill)

cond_well = np.max(eigs_well) / np.min(eigs_well)
cond_ill = np.max(eigs_ill) / np.min(eigs_ill)

print(f"Well-conditioned condition number: {cond_well:.2f}")
print(f"Ill-conditioned condition number:  {cond_ill:.2f}")

# GD path comparison
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
axes[0].set_title(f'Well-conditioned (κ={cond_well:.1f})\n{len(path_well)} steps',
                  fontsize=12)
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].axis('equal')
axes[0].grid(True, alpha=0.3)

# Ill-conditioned
Z_ill = ill_conditioned(X_grid, Y_grid)
axes[1].contour(X_grid, Y_grid, Z_ill, levels=30, cmap='viridis', alpha=0.6)
axes[1].plot(path_ill[:, 0], path_ill[:, 1], 'ro-', linewidth=1.5, markersize=3)
axes[1].set_title(f'Ill-conditioned (κ={cond_ill:.1f})\n{len(path_ill)} steps',
                  fontsize=12)
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].axis('equal')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('condition_number.png', dpi=150)
plt.close()

print("Condition number comparison visualization saved: condition_number.png")
```

## 7. ML Applications

### 7.1 Convexity Analysis

A convex function has a positive semi-definite Hessian.

```python
# Check convexity of MSE loss
from sklearn.datasets import make_regression

X_reg, y_reg = make_regression(n_samples=100, n_features=2, noise=10, random_state=42)

def mse_loss(w):
    """MSE loss: L(w) = (1/2n) ||y - Xw||^2"""
    pred = X_reg @ w
    return 0.5 * np.mean((y_reg - pred)**2)

def grad_mse(w):
    """Gradient: ∇L = -(1/n) X^T (y - Xw)"""
    residual = y_reg - X_reg @ w
    return -X_reg.T @ residual / len(y_reg)

def hess_mse(w):
    """Hessian: H = (1/n) X^T X"""
    return X_reg.T @ X_reg / len(y_reg)

# Eigenvalues of Hessian (convexity check)
w_sample = np.random.randn(2)
H = hess_mse(w_sample)
eigenvalues = np.linalg.eigvalsh(H)

print("MSE loss function convexity analysis:")
print(f"Eigenvalues of Hessian: {eigenvalues}")
print(f"All positive? {np.all(eigenvalues >= -1e-10)}")
print("→ MSE is a convex function (global minimum guaranteed)")
```

### 7.2 Gradient-Based Optimization Comparison

```python
# Compare various optimization algorithms
from scipy.optimize import minimize

# Starting point
w0 = np.array([2.0, -1.5])

# Optimization algorithms
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
    print(f"  Optimal w: {result.x}")
    print(f"  Final loss: {result.fun:.6f}")
    print(f"  Iterations: {result.nit}")
    print(f"  Function evaluations: {result.nfev}")

# Compare with analytical solution
w_analytic = np.linalg.lstsq(X_reg, y_reg, rcond=None)[0]
print(f"\nAnalytical solution (Normal Equation): {w_analytic}")
```

## Practice Problems

### Problem 1: Critical Point Classification
For the function $f(x, y) = x^3 - 3xy^2$:
1. Compute the gradient and find critical points
2. Compute the Hessian and determine the nature of each critical point (extremum/saddle point)
3. Visualize in 3D to verify

### Problem 2: Constrained Optimization
Visualize contour lines and the constraint $x^2 + y^2 = 1$, and use Lagrange multipliers to optimize:

$$\min_{x,y} f(x, y) = x^2 + 2y^2 \quad \text{s.t.} \quad x^2 + y^2 = 1$$

Compare analytical and numerical solutions.

### Problem 3: Momentum Gradient Descent
Implement gradient descent with momentum:

$$\mathbf{v}_{t+1} = \beta \mathbf{v}_t - \alpha \nabla f(\mathbf{x}_t)$$
$$\mathbf{x}_{t+1} = \mathbf{x}_t + \mathbf{v}_{t+1}$$

Compare performance with regular GD on a function with high condition number.

### Problem 4: Normal Equation for Linear Regression
Derive the normal equation for MSE loss using Taylor expansion:

$$\mathbf{w}^* = (X^T X)^{-1} X^T \mathbf{y}$$

Hint: Solve $\nabla L = 0$ and verify that the Hessian is positive definite.

### Problem 5: Adam Optimizer Implementation
Implement the Adam optimizer update rules and compare regular GD, Momentum, and Adam on an anisotropic loss function. Visualize convergence rate and paths.

## References

### Online Resources
- [Multivariable Calculus - MIT OpenCourseWare](https://ocw.mit.edu/courses/mathematics/18-02sc-multivariable-calculus-fall-2010/)
- [Visual Calculus](https://visualcalculus.com/) - Interactive visualization
- [Gradient Descent Visualization](https://distill.pub/2017/momentum/) - Distill.pub

### Textbooks
- Stewart, *Calculus: Early Transcendentals*, Chapters 14-15
- Strang, *Calculus*, Volume 3 (Multivariable)
- Boyd & Vandenberghe, *Convex Optimization*, Appendix A

### Papers and Tutorials
- Ruder, *An Overview of Gradient Descent Optimization Algorithms* (2016)
- Goodfellow et al., *Deep Learning*, Chapter 4 (Numerical Computation)
- Nocedal & Wright, *Numerical Optimization*, Chapter 2 (Fundamentals)
