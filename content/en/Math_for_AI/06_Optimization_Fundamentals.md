# 06. Optimization Fundamentals

## Learning Objectives

- Understand the mathematical formulation of optimization problems and types of constraints
- Learn the definitions and properties of convex sets and convex functions, and understand how convexity affects optimization
- Express necessary and sufficient conditions for extrema using gradients and Hessians, and apply them
- Solve equality-constrained optimization problems using the Lagrange multiplier method
- Understand KKT conditions and apply them to inequality-constrained optimization problems
- Learn how optimization theory is utilized in machine learning through practical examples

---

## 1. Formulation of Optimization Problems

### 1.1 General Form of Optimization Problems

Optimization is the process of finding variable values that minimize (or maximize) an objective function under given constraints.

**Standard form:**

$$
\begin{align}
\min_{x \in \mathbb{R}^n} \quad & f(x) \\
\text{subject to} \quad & h_i(x) = 0, \quad i = 1, \ldots, m \\
& g_j(x) \leq 0, \quad j = 1, \ldots, p
\end{align}
$$

- $f(x)$: objective function
- $h_i(x) = 0$: equality constraints
- $g_j(x) \leq 0$: inequality constraints

### 1.2 Feasible Region

The feasible region $\mathcal{F}$ is the set of all points that satisfy all constraints:

$$
\mathcal{F} = \{x \in \mathbb{R}^n : h_i(x) = 0, \; g_j(x) \leq 0, \; \forall i, j\}
$$

**Example: 2D Constrained Optimization**

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.optimize import minimize

# Objective function: f(x,y) = (x-1)^2 + (y-2)^2
def objective(X):
    x, y = X
    return (x - 1)**2 + (y - 2)**2

# Equality constraint: h(x,y) = x + y - 1 = 0
def equality_constraint(X):
    x, y = X
    return x + y - 1

# Inequality constraint: g(x,y) = x^2 + y^2 - 4 <= 0 (inside the circle)
def inequality_constraint(X):
    x, y = X
    return x**2 + y**2 - 4

# Visualization
fig, ax = plt.subplots(figsize=(10, 8))

# Objective function contour
x = np.linspace(-2.5, 3, 300)
y = np.linspace(-2.5, 3, 300)
X, Y = np.meshgrid(x, y)
Z = (X - 1)**2 + (Y - 2)**2
contour = ax.contour(X, Y, Z, levels=20, alpha=0.6, cmap='viridis')
ax.clabel(contour, inline=True, fontsize=8)

# Equality constraint (line)
x_line = np.linspace(-2.5, 3, 100)
y_line = 1 - x_line
ax.plot(x_line, y_line, 'r-', linewidth=2, label='$x + y = 1$ (equality constraint)')

# Inequality constraint (circle)
circle = Circle((0, 0), 2, fill=False, edgecolor='blue', linewidth=2,
                label='$x^2 + y^2 \leq 4$ (inequality constraint)')
ax.add_patch(circle)

# Highlight feasible region
theta = np.linspace(0, 2*np.pi, 100)
x_circle = 2 * np.cos(theta)
y_circle = 2 * np.sin(theta)
mask = (x_circle + y_circle >= 0.9) & (x_circle + y_circle <= 1.1)
ax.fill_between(x_circle[mask], -2.5, y_circle[mask], alpha=0.2, color='green')

ax.set_xlim(-2.5, 3)
ax.set_ylim(-2.5, 3)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('Constrained Optimization: Feasible Region and Objective Function', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)
plt.tight_layout()
plt.savefig('optimization_feasible_region.png', dpi=150)
plt.show()

print("Feasible region: the portion of the line-circle intersection satisfying all constraints")
```

### 1.3 Classification of Optimization Problems

- **Unconstrained Optimization**: $\min f(x)$
- **Constrained Optimization**: includes equality/inequality constraints
- **Linear Programming**: $f$, $h$, $g$ are all linear
- **Convex Optimization**: $f$, $g$ convex, $h$ affine
- **Nonlinear Optimization**: general nonlinear functions

## 2. Convex Sets and Convex Functions

### 2.1 Convex Set

A set $C \subseteq \mathbb{R}^n$ is convex if the line segment connecting any two points in the set is entirely contained within the set:

$$
x, y \in C, \; \lambda \in [0, 1] \implies \lambda x + (1-\lambda)y \in C
$$

**Example: Convex vs Non-convex Sets**

```python
import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# 1. Circle (convex)
theta = np.linspace(0, 2*np.pi, 100)
x1 = np.cos(theta)
y1 = np.sin(theta)
axes[0].fill(x1, y1, alpha=0.3, color='green')
axes[0].plot([0.5, -0.5], [0.3, -0.7], 'ro-', linewidth=2, markersize=8)
axes[0].set_title('Circle (Convex)', fontsize=12)
axes[0].set_aspect('equal')
axes[0].grid(True, alpha=0.3)

# 2. Half-plane (convex)
x2 = np.array([-2, 2, 2, -2])
y2 = np.array([0, 0, 2, 2])
axes[1].fill(x2, y2, alpha=0.3, color='green')
axes[1].plot([0.5, -0.5], [0.5, 1.5], 'ro-', linewidth=2, markersize=8)
axes[1].set_title('Half-plane $y \geq 0$ (Convex)', fontsize=12)
axes[1].set_xlim(-2, 2)
axes[1].set_ylim(-2, 2)
axes[1].grid(True, alpha=0.3)

# 3. Polygon (convex)
x3 = np.array([0, 1, 0.5, -0.5, -1])
y3 = np.array([1, 0, -1, -1, 0])
axes[2].fill(x3, y3, alpha=0.3, color='green')
axes[2].plot([0.5, -0.5], [0.3, -0.3], 'ro-', linewidth=2, markersize=8)
axes[2].set_title('Convex Polygon (Convex)', fontsize=12)
axes[2].set_aspect('equal')
axes[2].grid(True, alpha=0.3)

# 4. Crescent (non-convex)
theta = np.linspace(0, 2*np.pi, 100)
x4_outer = 1.5 * np.cos(theta)
y4_outer = 1.5 * np.sin(theta)
x4_inner = 1.0 * np.cos(theta) + 0.5
y4_inner = 1.0 * np.sin(theta)
axes[3].fill(x4_outer, y4_outer, alpha=0.3, color='red')
axes[3].fill(x4_inner, y4_inner, alpha=1.0, color='white')
axes[3].plot([-0.5, 1.0], [0.5, -0.5], 'bo--', linewidth=2, markersize=8)
axes[3].set_title('Crescent (Non-convex)', fontsize=12)
axes[3].set_aspect('equal')
axes[3].grid(True, alpha=0.3)

# 5. Star shape (non-convex)
theta = np.linspace(0, 2*np.pi, 11)
r = np.where(np.arange(11) % 2 == 0, 1.5, 0.7)
x5 = r * np.cos(theta)
y5 = r * np.sin(theta)
axes[4].fill(x5, y5, alpha=0.3, color='red')
axes[4].plot([1.0, -1.0], [0.0, 0.0], 'bo--', linewidth=2, markersize=8)
axes[4].set_title('Star (Non-convex)', fontsize=12)
axes[4].set_aspect('equal')
axes[4].grid(True, alpha=0.3)

# 6. Ring (non-convex)
theta = np.linspace(0, 2*np.pi, 100)
x6_outer = 1.5 * np.cos(theta)
y6_outer = 1.5 * np.sin(theta)
x6_inner = 0.7 * np.cos(theta)
y6_inner = 0.7 * np.sin(theta)
axes[5].fill(x6_outer, y6_outer, alpha=0.3, color='red')
axes[5].fill(x6_inner, y6_inner, alpha=1.0, color='white')
axes[5].plot([-1.0, 1.0], [0.0, 0.0], 'bo--', linewidth=2, markersize=8)
axes[5].set_title('Ring (Non-convex)', fontsize=12)
axes[5].set_aspect('equal')
axes[5].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('convex_vs_nonconvex_sets.png', dpi=150)
plt.show()

print("Convex sets (green): line segment between any two points stays inside the set")
print("Non-convex sets (red): line segment goes outside the set")
```

### 2.2 Convex Function

A function $f: \mathbb{R}^n \to \mathbb{R}$ is convex if its domain is a convex set and it satisfies:

$$
f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y), \quad \forall x, y \in \text{dom}(f), \; \lambda \in [0,1]
$$

**Geometric interpretation**: The line segment connecting any two points on the function graph always lies above the graph.

**Strictly Convex**: inequality becomes $<$

**Strongly Convex**: $\exists m > 0$ such that

$$
f(x) - \frac{m}{2}\|x\|^2 \text{ is convex}
$$

### 2.3 Criteria for Convexity

**First-order condition (if differentiable):**

$$
f(y) \geq f(x) + \nabla f(x)^T (y - x), \quad \forall x, y
$$

**Second-order condition (if twice differentiable):**

$$
\nabla^2 f(x) \succeq 0 \quad \text{(Hessian is positive semidefinite)}
$$

Strongly convex: $\nabla^2 f(x) \succ 0$ (Hessian is positive definite)

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Convex function examples
def convex_func(x, y):
    """Convex: f(x,y) = x^2 + y^2"""
    return x**2 + y**2

def nonconvex_func(x, y):
    """Non-convex: f(x,y) = x^2 - y^2 (saddle point)"""
    return x**2 - y**2

def visualize_function(func, title, ax):
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x,y)')
    ax.set_title(title)

fig = plt.figure(figsize=(15, 6))

# Convex function
ax1 = fig.add_subplot(121, projection='3d')
visualize_function(convex_func, 'Convex Function: $f(x,y) = x^2 + y^2$', ax1)

# Non-convex function
ax2 = fig.add_subplot(122, projection='3d')
visualize_function(nonconvex_func, 'Non-convex Function: $f(x,y) = x^2 - y^2$', ax2)

plt.tight_layout()
plt.savefig('convex_vs_nonconvex_functions.png', dpi=150)
plt.show()

# Check Hessian
def check_convexity(x, y, name):
    if name == "convex":
        # f(x,y) = x^2 + y^2
        # ∇²f = [[2, 0], [0, 2]]
        hessian = np.array([[2, 0], [0, 2]])
    else:
        # f(x,y) = x^2 - y^2
        # ∇²f = [[2, 0], [0, -2]]
        hessian = np.array([[2, 0], [0, -2]])

    eigenvalues = np.linalg.eigvals(hessian)
    print(f"\nHessian eigenvalues of {name} function: {eigenvalues}")

    if np.all(eigenvalues >= 0):
        print("  → All eigenvalues ≥ 0: convex function")
    elif np.all(eigenvalues > 0):
        print("  → All eigenvalues > 0: strongly convex function")
    else:
        print("  → Negative eigenvalue exists: non-convex function")

check_convexity(0, 0, "convex")
check_convexity(0, 0, "nonconvex")
```

### 2.4 Important Properties of Convex Functions

**Theorem 1**: A local minimum of a convex function is a global minimum.

**Theorem 2**: A strongly convex function has a unique global minimum.

**Theorem 3**: The sum, positive scalar multiple, and maximum of convex functions are convex.

**Example: Convex Functions in Machine Learning**

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 300)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Quadratic function
axes[0, 0].plot(x, x**2, linewidth=2)
axes[0, 0].set_title('$f(x) = x^2$ (Ridge Regularization)', fontsize=12)
axes[0, 0].grid(True, alpha=0.3)

# 2. Absolute value (L1 norm)
axes[0, 1].plot(x, np.abs(x), linewidth=2, color='orange')
axes[0, 1].set_title('$f(x) = |x|$ (Lasso Regularization)', fontsize=12)
axes[0, 1].grid(True, alpha=0.3)

# 3. Exponential function
axes[0, 2].plot(x, np.exp(x), linewidth=2, color='green')
axes[0, 2].set_title('$f(x) = e^x$', fontsize=12)
axes[0, 2].set_ylim(0, 10)
axes[0, 2].grid(True, alpha=0.3)

# 4. Logistic loss
axes[1, 0].plot(x, np.log(1 + np.exp(x)), linewidth=2, color='red')
axes[1, 0].set_title('$f(x) = \log(1 + e^x)$ (Logistic Loss)', fontsize=12)
axes[1, 0].grid(True, alpha=0.3)

# 5. Huber loss
delta = 1.0
huber = np.where(np.abs(x) <= delta, 0.5 * x**2, delta * (np.abs(x) - 0.5 * delta))
axes[1, 1].plot(x, huber, linewidth=2, color='purple')
axes[1, 1].set_title('Huber Loss ($\delta=1$)', fontsize=12)
axes[1, 1].grid(True, alpha=0.3)

# 6. -log (negative log, convex)
x_pos = np.linspace(0.01, 3, 300)
axes[1, 2].plot(x_pos, -np.log(x_pos), linewidth=2, color='brown')
axes[1, 2].set_title('$f(x) = -\log(x)$ (x > 0)', fontsize=12)
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ml_convex_functions.png', dpi=150)
plt.show()

print("Commonly used convex functions in machine learning:")
print("- Quadratic: Ridge regularization, MSE loss")
print("- L1 norm: Lasso regularization")
print("- Logistic loss: logistic regression")
print("- Huber loss: robust regression")
```

## 3. Necessary and Sufficient Conditions for Extrema

### 3.1 First-Order Necessary Condition

For unconstrained optimization, if $x^*$ is a local minimum:

$$
\nabla f(x^*) = 0
$$

Such a point is called a **stationary point** or **critical point**.

### 3.2 Second-Order Sufficient Condition

When $x^*$ is a stationary point:
- **Local minimum**: $\nabla^2 f(x^*) \succ 0$ (Hessian is positive definite)
- **Local maximum**: $\nabla^2 f(x^*) \prec 0$ (Hessian is negative definite)
- **Saddle Point**: Hessian is indefinite (has both positive and negative eigenvalues)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

def rosenbrock(X):
    """Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2"""
    x, y = X
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_gradient(X):
    x, y = X
    df_dx = -2*(1 - x) - 400*x*(y - x**2)
    df_dy = 200*(y - x**2)
    return np.array([df_dx, df_dy])

def rosenbrock_hessian(X):
    x, y = X
    d2f_dx2 = 2 - 400*y + 1200*x**2
    d2f_dxdy = -400*x
    d2f_dy2 = 200
    return np.array([[d2f_dx2, d2f_dxdy],
                     [d2f_dxdy, d2f_dy2]])

# Find minimum
x0 = np.array([0.0, 0.0])
result = minimize(rosenbrock, x0, method='BFGS', jac=rosenbrock_gradient)
x_min = result.x

print("Rosenbrock function optimization")
print(f"Minimum location: x* = {x_min}")
print(f"Minimum value: f(x*) = {rosenbrock(x_min):.6f}")
print(f"\nGradient: ∇f(x*) = {rosenbrock_gradient(x_min)}")

hessian = rosenbrock_hessian(x_min)
eigenvalues = np.linalg.eigvals(hessian)
print(f"\nHessian:\n{hessian}")
print(f"Eigenvalues: {eigenvalues}")
print(f"  → All eigenvalues > 0: local minimum confirmed")

# Visualization
fig = plt.figure(figsize=(16, 6))

# 3D surface
ax1 = fig.add_subplot(131, projection='3d')
x = np.linspace(-2, 2, 100)
y = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x, y)
Z = (1 - X)**2 + 100*(Y - X**2)**2
ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, edgecolor='none')
ax1.scatter([x_min[0]], [x_min[1]], [rosenbrock(x_min)],
            color='red', s=100, label='Minimum')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x,y)')
ax1.set_title('Rosenbrock Function', fontsize=12)
ax1.legend()

# Contour plot
ax2 = fig.add_subplot(132)
levels = np.logspace(-1, 3, 20)
contour = ax2.contour(X, Y, Z, levels=levels, cmap='viridis')
ax2.clabel(contour, inline=True, fontsize=8)
ax2.plot(x_min[0], x_min[1], 'r*', markersize=20, label='Minimum (1, 1)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Contour Plot', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Saddle point example: f(x,y) = x^2 - y^2
ax3 = fig.add_subplot(133, projection='3d')
Z_saddle = X**2 - Y**2
ax3.plot_surface(X, Y, Z_saddle, cmap='coolwarm', alpha=0.8, edgecolor='none')
ax3.scatter([0], [0], [0], color='red', s=100, label='Saddle point (0, 0)')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('f(x,y)')
ax3.set_title('Saddle Point: $f(x,y) = x^2 - y^2$', fontsize=12)
ax3.legend()

plt.tight_layout()
plt.savefig('critical_points_analysis.png', dpi=150)
plt.show()

# Hessian at saddle point
print("\n\nSaddle point example: f(x,y) = x^2 - y^2")
saddle_hessian = np.array([[2, 0], [0, -2]])
saddle_eigenvalues = np.linalg.eigvals(saddle_hessian)
print(f"Hessian at origin (0, 0):\n{saddle_hessian}")
print(f"Eigenvalues: {saddle_eigenvalues}")
print(f"  → Mixed positive and negative eigenvalues: saddle point")
```

## 4. Lagrange Multipliers

### 4.1 Equality-Constrained Optimization

Consider the following problem:

$$
\begin{align}
\min_{x} \quad & f(x) \\
\text{subject to} \quad & h(x) = 0
\end{align}
$$

**Lagrangian:**

$$
\mathcal{L}(x, \lambda) = f(x) + \lambda^T h(x)
$$

where $\lambda$ is the **Lagrange multiplier**.

**Necessary condition for optimality:**

$$
\begin{align}
\nabla_x \mathcal{L} &= \nabla f(x^*) + \lambda^* \nabla h(x^*) = 0 \\
\nabla_\lambda \mathcal{L} &= h(x^*) = 0
\end{align}
$$

**Geometric interpretation**: At the optimal point, $\nabla f$ and $\nabla h$ are parallel (same or opposite direction).

### 4.2 Example: Equality-Constrained Optimization

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Objective function: f(x,y) = x^2 + y^2
def objective(X):
    return X[0]**2 + X[1]**2

# Equality constraint: h(x,y) = x + 2y - 1 = 0
def constraint_eq(X):
    return X[0] + 2*X[1] - 1

# Lagrangian: L(x,y,λ) = x^2 + y^2 + λ(x + 2y - 1)
def lagrangian(X, lam):
    return objective(X) + lam * constraint_eq(X)

# Optimize with scipy
constraints = {'type': 'eq', 'fun': constraint_eq}
x0 = np.array([0.0, 0.0])
result = minimize(objective, x0, method='SLSQP', constraints=constraints)

x_opt = result.x
print("Lagrange multiplier method example")
print(f"Optimal solution: x* = {x_opt}")
print(f"Objective function value: f(x*) = {objective(x_opt):.6f}")
print(f"Constraint check: h(x*) = {constraint_eq(x_opt):.6e}")

# Analytical solution (by hand)
# ∇L = 0: 2x + λ = 0, 2y + 2λ = 0, x + 2y - 1 = 0
# → x = -λ/2, y = -λ, x + 2y = 1 → -λ/2 - 2λ = 1 → λ = -2/5
# → x = 1/5, y = 2/5
x_analytical = np.array([1/5, 2/5])
lambda_analytical = -2/5

print(f"\nAnalytical solution: x* = {x_analytical}")
print(f"Lagrange multiplier: λ* = {lambda_analytical}")

# Visualization
fig, ax = plt.subplots(figsize=(10, 8))

# Objective function contour
x = np.linspace(-1, 2, 300)
y = np.linspace(-1, 2, 300)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2
contour = ax.contour(X, Y, Z, levels=15, alpha=0.6, cmap='viridis')
ax.clabel(contour, inline=True, fontsize=8)

# Constraint (line)
x_line = np.linspace(-1, 2, 100)
y_line = (1 - x_line) / 2
ax.plot(x_line, y_line, 'r-', linewidth=3, label='Constraint: $x + 2y = 1$')

# Optimal point
ax.plot(x_opt[0], x_opt[1], 'r*', markersize=20, label=f'Optimal point ({x_opt[0]:.2f}, {x_opt[1]:.2f})')

# Gradient vectors
grad_f = 2 * x_opt  # ∇f = [2x, 2y]
grad_h = np.array([1, 2])  # ∇h = [1, 2]
ax.quiver(x_opt[0], x_opt[1], grad_f[0], grad_f[1],
          angles='xy', scale_units='xy', scale=5, color='blue', width=0.006,
          label='$\\nabla f$ (objective function gradient)')
ax.quiver(x_opt[0], x_opt[1], grad_h[0], grad_h[1],
          angles='xy', scale_units='xy', scale=5, color='green', width=0.006,
          label='$\\nabla h$ (constraint gradient)')

ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-0.5, 1.5)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('Lagrange Multipliers: $\\nabla f$ and $\\nabla h$ are Parallel', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('lagrange_multipliers.png', dpi=150)
plt.show()

print(f"\nAt optimal point: ∇f = {grad_f}, ∇h = {grad_h}")
print(f"Verify ∇f = λ∇h: λ = {grad_f[0] / grad_h[0]:.3f} (x component)")
```

## 5. KKT Conditions (Karush-Kuhn-Tucker Conditions)

### 5.1 Inequality-Constrained Optimization

Consider the general optimization problem:

$$
\begin{align}
\min_{x} \quad & f(x) \\
\text{subject to} \quad & h_i(x) = 0, \quad i = 1, \ldots, m \\
& g_j(x) \leq 0, \quad j = 1, \ldots, p
\end{align}
$$

**Generalized Lagrangian:**

$$
\mathcal{L}(x, \lambda, \mu) = f(x) + \sum_{i=1}^m \lambda_i h_i(x) + \sum_{j=1}^p \mu_j g_j(x)
$$

### 5.2 KKT Conditions

If $x^*$ is optimal and certain regularity conditions are satisfied, there exist $\lambda^*, \mu^*$ satisfying the KKT conditions:

1. **Stationarity:**
   $$\nabla_x \mathcal{L}(x^*, \lambda^*, \mu^*) = 0$$

2. **Primal Feasibility:**
   $$h_i(x^*) = 0, \quad g_j(x^*) \leq 0$$

3. **Dual Feasibility:**
   $$\mu_j^* \geq 0$$

4. **Complementary Slackness:**
   $$\mu_j^* g_j(x^*) = 0, \quad \forall j$$

**Meaning of complementary slackness**: For each inequality constraint
- $g_j(x^*) < 0$ (constraint inactive) → $\mu_j^* = 0$
- $\mu_j^* > 0$ (Lagrange multiplier positive) → $g_j(x^*) = 0$ (constraint active)

### 5.3 Example: Applying KKT Conditions

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Objective function: f(x,y) = (x-2)^2 + (y-1)^2
def objective(X):
    x, y = X
    return (x - 2)**2 + (y - 1)**2

def objective_grad(X):
    x, y = X
    return np.array([2*(x - 2), 2*(y - 1)])

# Inequality constraint: g1(x,y) = x + y - 1 <= 0
def constraint_ineq1(X):
    return X[0] + X[1] - 1

# Inequality constraint: g2(x,y) = -x <= 0 (i.e., x >= 0)
def constraint_ineq2(X):
    return -X[0]

# Inequality constraint: g3(x,y) = -y <= 0 (i.e., y >= 0)
def constraint_ineq3(X):
    return -X[1]

# Optimize with scipy
constraints = [
    {'type': 'ineq', 'fun': lambda X: -constraint_ineq1(X)},  # scipy uses g(x) >= 0 format
    {'type': 'ineq', 'fun': lambda X: -constraint_ineq2(X)},
    {'type': 'ineq', 'fun': lambda X: -constraint_ineq3(X)}
]

x0 = np.array([0.5, 0.5])
result = minimize(objective, x0, method='SLSQP', constraints=constraints,
                  options={'disp': True})

x_opt = result.x
print("KKT conditions example")
print(f"Optimal solution: x* = {x_opt}")
print(f"Objective function value: f(x*) = {objective(x_opt):.6f}")

# Check constraints
g1 = constraint_ineq1(x_opt)
g2 = constraint_ineq2(x_opt)
g3 = constraint_ineq3(x_opt)
print(f"\nConstraint values:")
print(f"  g1(x*) = x + y - 1 = {g1:.6f} {'(active)' if abs(g1) < 1e-6 else '(inactive)'}")
print(f"  g2(x*) = -x = {g2:.6f} {'(active)' if abs(g2) < 1e-6 else '(inactive)'}")
print(f"  g3(x*) = -y = {g3:.6f} {'(active)' if abs(g3) < 1e-6 else '(inactive)'}")

# Visualization
fig, ax = plt.subplots(figsize=(10, 8))

# Objective function contour
x = np.linspace(-0.5, 2.5, 300)
y = np.linspace(-0.5, 2.5, 300)
X, Y = np.meshgrid(x, y)
Z = (X - 2)**2 + (Y - 1)**2
contour = ax.contour(X, Y, Z, levels=15, alpha=0.6, cmap='viridis')
ax.clabel(contour, inline=True, fontsize=8)

# Feasible region (triangle)
feasible_x = [0, 1, 0, 0]
feasible_y = [0, 0, 1, 0]
ax.fill(feasible_x, feasible_y, alpha=0.2, color='green', label='Feasible region')

# Constraint boundaries
ax.plot([0, 1], [1, 0], 'r-', linewidth=2, label='$x + y = 1$')
ax.axvline(x=0, color='blue', linewidth=2, linestyle='--', label='$x = 0$')
ax.axhline(y=0, color='purple', linewidth=2, linestyle='--', label='$y = 0$')

# Unconstrained minimum
ax.plot(2, 1, 'go', markersize=12, label='Unconstrained minimum (2, 1)')

# KKT optimal point
ax.plot(x_opt[0], x_opt[1], 'r*', markersize=20,
        label=f'KKT optimal point ({x_opt[0]:.2f}, {x_opt[1]:.2f})')

ax.set_xlim(-0.5, 2.5)
ax.set_ylim(-0.5, 2.0)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('KKT Conditions: Inequality-Constrained Optimization', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('kkt_conditions.png', dpi=150)
plt.show()

# Manual KKT verification (when active constraint is g1)
print("\n\nManual KKT condition verification:")
print("Active constraint: g1(x,y) = x + y - 1 = 0")
print("∇f(x*) + μ1∇g1(x*) = 0")
print(f"∇f(x*) = {objective_grad(x_opt)}")
print("∇g1 = [1, 1]")
mu1 = -objective_grad(x_opt)[0]  # or -objective_grad(x_opt)[1]
print(f"→ μ1 = {mu1:.4f}")
print(f"Complementary slackness: μ1 * g1(x*) = {mu1 * g1:.6e} ≈ 0 ✓")
```

## 6. Optimization Applications in Machine Learning

### 6.1 SVM Dual Problem

**Primal Problem:**

$$
\begin{align}
\min_{w, b, \xi} \quad & \frac{1}{2}\|w\|^2 + C\sum_{i=1}^n \xi_i \\
\text{subject to} \quad & y_i(w^T x_i + b) \geq 1 - \xi_i \\
& \xi_i \geq 0
\end{align}
$$

Constructing the Lagrangian and applying KKT conditions yields the **Dual Problem**:

$$
\begin{align}
\max_{\alpha} \quad & \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i,j}\alpha_i \alpha_j y_i y_j x_i^T x_j \\
\text{subject to} \quad & 0 \leq \alpha_i \leq C \\
& \sum_{i=1}^n \alpha_i y_i = 0
\end{align}
$$

Advantages of dual problem:
- Kernel trick applicable
- Dimensionality depends on number of data points (favorable for high-dimensional data)

```python
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Generate data
X, y = datasets.make_classification(n_samples=100, n_features=2, n_informative=2,
                                     n_redundant=0, n_clusters_per_class=1,
                                     random_state=42)
y = 2*y - 1  # {0,1} → {-1,1}

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train SVM
svm = SVC(kernel='linear', C=1.0)
svm.fit(X, y)

# Visualize decision boundary
fig, ax = plt.subplots(figsize=(10, 8))

# Data points
ax.scatter(X[y==1, 0], X[y==1, 1], c='blue', marker='o', s=100,
           edgecolors='k', label='Class +1')
ax.scatter(X[y==-1, 0], X[y==-1, 1], c='red', marker='s', s=100,
           edgecolors='k', label='Class -1')

# Decision boundary
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(xlim[0], xlim[1], 300)
yy = np.linspace(ylim[0], ylim[1], 300)
XX, YY = np.meshgrid(xx, yy)
Z = svm.decision_function(np.c_[XX.ravel(), YY.ravel()])
Z = Z.reshape(XX.shape)

# Decision boundary and margin
ax.contour(XX, YY, Z, levels=[-1, 0, 1], colors=['red', 'black', 'blue'],
           linestyles=['--', '-', '--'], linewidths=[2, 3, 2])

# Highlight support vectors
support_vectors = svm.support_vectors_
ax.scatter(support_vectors[:, 0], support_vectors[:, 1], s=300,
           linewidths=2, facecolors='none', edgecolors='green',
           label='Support Vectors')

ax.set_xlabel('Feature 1', fontsize=12)
ax.set_ylabel('Feature 2', fontsize=12)
ax.set_title('SVM: Dual Problem Derived via KKT Conditions', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('svm_kkt_dual.png', dpi=150)
plt.show()

print(f"Number of support vectors: {len(support_vectors)}")
print(f"Dual coefficients (α): {svm.dual_coef_}")
print("\nKKT complementary slackness:")
print("  α = 0: outside margin (inactive constraint)")
print("  0 < α < C: on margin (support vector)")
print("  α = C: inside margin (possible misclassification)")
```

### 6.2 Regularization = Constrained Optimization

**Two Formulations of Ridge Regression:**

**Regularization form:**
$$\min_w \|Xw - y\|^2 + \lambda\|w\|^2$$

**Constrained form:**
$$
\begin{align}
\min_w \quad & \|Xw - y\|^2 \\
\text{subject to} \quad & \|w\|^2 \leq t
\end{align}
$$

These two problems are equivalent with appropriate correspondence between $\lambda$ and $t$ (Lagrangian duality).

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Generate data
np.random.seed(42)
n_samples, n_features = 100, 50
X = np.random.randn(n_samples, n_features)
true_w = np.random.randn(n_features)
true_w[20:] = 0  # sparsity
y = X @ true_w + 0.1 * np.random.randn(n_samples)

# Ridge regression (multiple λ values)
lambdas = np.logspace(-3, 3, 100)
coefs = []

for lam in lambdas:
    ridge = Ridge(alpha=lam, fit_intercept=False)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)

coefs = np.array(coefs)

# Visualization: Regularization Path
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: coefficient path
for i in range(min(10, n_features)):
    axes[0].plot(np.log10(lambdas), coefs[:, i], linewidth=2)
axes[0].set_xlabel('$\log_{10}(\lambda)$', fontsize=12)
axes[0].set_ylabel('Coefficient value', fontsize=12)
axes[0].set_title('Ridge Regularization Path', fontsize=14)
axes[0].grid(True, alpha=0.3)
axes[0].axvline(x=0, color='r', linestyle='--', label='$\lambda=1$')
axes[0].legend()

# Right: L2 norm
l2_norms = np.linalg.norm(coefs, axis=1)
axes[1].plot(np.log10(lambdas), l2_norms, linewidth=3, color='purple')
axes[1].set_xlabel('$\log_{10}(\lambda)$', fontsize=12)
axes[1].set_ylabel('$\|w\|_2$', fontsize=12)
axes[1].set_title('L2 Norm of Coefficients vs Regularization Strength', fontsize=14)
axes[1].grid(True, alpha=0.3)
axes[1].axvline(x=0, color='r', linestyle='--', label='$\lambda=1$')
axes[1].legend()

plt.tight_layout()
plt.savefig('ridge_regularization_path.png', dpi=150)
plt.show()

print("Relationship between regularization and constrained optimization:")
print(f"  λ ↑ → constraint t ↓ → ||w|| ↓")
print(f"  λ=0: no constraint (OLS)")
print(f"  λ→∞: w → 0")
```

### 6.3 Convex vs Non-convex Optimization

**Convex Optimization:**
- Linear/logistic regression, SVM, Lasso
- Global minimum guaranteed
- Efficient algorithms exist

**Non-convex Optimization:**
- Neural networks (MLP, CNN, Transformer)
- Local minima and saddle points exist
- Heuristics: SGD + momentum, adaptive learning rates
- Over-parameterization → many good local minima

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Simple neural network: non-convex loss landscape
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.tensor([1.0]))
        self.w2 = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        return self.w1 * x + self.w2 * x**2

# Loss function (simulating a non-convex landscape)
def loss_landscape(w1, w2):
    """Artificially non-convex loss landscape"""
    return (w1**2 + w2**2) * (1 + 0.3*np.sin(5*w1) * np.sin(5*w2))

# Visualization
w1_vals = np.linspace(-2, 2, 200)
w2_vals = np.linspace(-2, 2, 200)
W1, W2 = np.meshgrid(w1_vals, w2_vals)
Loss = loss_landscape(W1, W2)

fig = plt.figure(figsize=(16, 6))

# 3D surface
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(W1, W2, Loss, cmap='viridis', alpha=0.8, edgecolor='none')
ax1.set_xlabel('$w_1$', fontsize=12)
ax1.set_ylabel('$w_2$', fontsize=12)
ax1.set_zlabel('Loss', fontsize=12)
ax1.set_title('Neural Network Loss Landscape (Non-convex)', fontsize=14)

# Contour
ax2 = fig.add_subplot(122)
contour = ax2.contourf(W1, W2, Loss, levels=30, cmap='viridis')
plt.colorbar(contour, ax=ax2)
ax2.set_xlabel('$w_1$', fontsize=12)
ax2.set_ylabel('$w_2$', fontsize=12)
ax2.set_title('Loss Contour: Multiple Local Minima', fontsize=14)

# Approximate local minima locations
local_minima = [(-1.2, -1.2), (0, 0), (1.2, 1.2)]
for x, y in local_minima:
    ax2.plot(x, y, 'r*', markersize=15)

plt.tight_layout()
plt.savefig('nonconvex_loss_landscape.png', dpi=150)
plt.show()

print("Characteristics of neural network optimization:")
print("  - Non-convex loss landscape")
print("  - Multiple local minima and saddle points")
print("  - SGD + momentum to escape saddle points")
print("  - Over-parameterization: many good local minima")
```

## 7. Implementing Optimization Algorithms

### 7.1 Using scipy.optimize

```python
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
import numpy as np

# Example: Portfolio Optimization
# Objective: minimize risk (variance)
# Constraints: achieve target return, weights sum to 1

np.random.seed(42)
n_assets = 5
returns = np.random.randn(n_assets) * 0.1 + 0.05  # expected returns
cov_matrix = np.random.randn(n_assets, n_assets)
cov_matrix = cov_matrix @ cov_matrix.T  # positive definite covariance

def portfolio_variance(weights):
    """Portfolio variance (objective function)"""
    return weights @ cov_matrix @ weights

def portfolio_return(weights):
    """Portfolio expected return"""
    return weights @ returns

# Constraints
target_return = 0.06
constraints = [
    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # sum = 1
    {'type': 'eq', 'fun': lambda w: portfolio_return(w) - target_return}  # return target
]

# Bounds: 0 <= w_i <= 1 (no short selling)
bounds = [(0, 1) for _ in range(n_assets)]

# Initial weights
w0 = np.ones(n_assets) / n_assets

# Optimize
result = minimize(portfolio_variance, w0, method='SLSQP',
                  bounds=bounds, constraints=constraints)

optimal_weights = result.x

print("Portfolio Optimization (KKT Conditions)")
print(f"Optimal weights: {optimal_weights}")
print(f"Portfolio variance: {portfolio_variance(optimal_weights):.6f}")
print(f"Portfolio return: {portfolio_return(optimal_weights):.6f}")
print(f"Target return: {target_return}")

# Visualization
fig, ax = plt.subplots(figsize=(10, 6))
assets = [f'Asset {i+1}' for i in range(n_assets)]
bars = ax.bar(assets, optimal_weights, color='skyblue', edgecolor='black', linewidth=1.5)

# Show values
for bar, weight in zip(bars, optimal_weights):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{weight:.3f}', ha='center', va='bottom', fontsize=11)

ax.set_ylabel('Weight', fontsize=12)
ax.set_title(f'Optimal Portfolio (Target Return {target_return:.1%})', fontsize=14)
ax.set_ylim(0, max(optimal_weights) * 1.2)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('portfolio_optimization.png', dpi=150)
plt.show()
```

## Practice Problems

1. **Convexity Determination**: Determine whether the following functions are convex, and prove using the Hessian.
   - (a) $f(x, y) = e^x + e^y$
   - (b) $f(x, y) = x^2 - xy + y^2$
   - (c) $f(x, y) = \log(e^x + e^y)$

2. **Lagrange Multipliers**: Solve the following optimization problem using Lagrange multipliers.
   $$
   \begin{align}
   \min_{x, y} \quad & x^2 + 2y^2 \\
   \text{subject to} \quad & x + y = 1
   \end{align}
   $$
   Find the analytical solution and verify with Python.

3. **KKT Conditions Application**: Write the KKT conditions for the following problem and find the optimal solution.
   $$
   \begin{align}
   \min_{x, y} \quad & (x-3)^2 + (y-2)^2 \\
   \text{subject to} \quad & x + 2y \leq 4 \\
   & x \geq 0, \; y \geq 0
   \end{align}
   $$
   Analyze which constraints are active.

4. **SVM Dual Problem**: Generate a linearly separable 2D dataset and solve the SVM dual problem directly. Use the `cvxopt` library to solve the quadratic programming (QP) problem and identify support vectors.

5. **Non-convex Optimization**: Run gradient descent on the Rosenbrock function from multiple initial points and compare the local minima they converge to. Analyze how convergence speed changes when momentum is added.

## References

- Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.
  - The bible of convex optimization, freely available online
- Nocedal, J., & Wright, S. (2006). *Numerical Optimization*. Springer.
  - Theory and implementation of numerical optimization algorithms
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
  - Chapter 4: Numerical Computation (optimization fundamentals)
- Bottou, L., Curtis, F. E., & Nocedal, J. (2018). "Optimization Methods for Large-Scale Machine Learning". *SIAM Review*.
  - Modern perspective on ML optimization
- SciPy Optimize Documentation: https://docs.scipy.org/doc/scipy/reference/optimize.html
- CVX Forum: https://ask.cvxr.com/
  - Convex optimization Q&A
