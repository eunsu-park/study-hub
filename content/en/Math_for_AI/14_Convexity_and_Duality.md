# 14. Convexity and Duality

## Learning Objectives

- Understand advanced properties of convex sets and convex functions
- Grasp the principles of Lagrangian duality and weak/strong duality
- Fully understand the derivation of SVM dual problem and its connection to the kernel trick
- Master the concept of Fenchel conjugate and its applications
- Understand proximal operators and proximal gradient methods applied to LASSO
- Learn practical applications of convex optimization and duality in machine learning

---

## 1. Review and Deep Dive into Convex Optimization

### 1.1 Convex Sets

A set $C$ is **convex** if:
$$
\forall x, y \in C, \forall \theta \in [0, 1]: \quad \theta x + (1-\theta) y \in C
$$

**Examples**:
- Hyperplane: $\{x : a^T x = b\}$
- Halfspace: $\{x : a^T x \leq b\}$
- Norm ball: $\{x : \|x\| \leq r\}$
- Positive semidefinite matrices: $\mathbb{S}_+^n$

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Convex set vs non-convex set visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Convex set (polygon)
convex_points = np.array([[0, 0], [2, 0], [2.5, 1], [1.5, 2], [0.5, 1.5]])
axes[0].add_patch(Polygon(convex_points, fill=True, alpha=0.3, color='blue'))
axes[0].plot(convex_points[:, 0], convex_points[:, 1], 'bo-')

# Connect two points
p1, p2 = convex_points[0], convex_points[3]
axes[0].plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=2, label='Line segment')
axes[0].scatter(*p1, color='red', s=100, zorder=5)
axes[0].scatter(*p2, color='red', s=100, zorder=5)
axes[0].set_title('Convex set: line segment between two points stays inside')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_aspect('equal')

# Non-convex set (crescent shape)
theta = np.linspace(0, 2*np.pi, 100)
outer = np.column_stack([np.cos(theta), np.sin(theta)])
inner = np.column_stack([0.5*np.cos(theta) + 0.3, 0.5*np.sin(theta)])
crescent = np.vstack([outer[:50], inner[50:0:-1]])
axes[1].add_patch(Polygon(crescent, fill=True, alpha=0.3, color='orange'))

# Show convexity violation
p1, p2 = crescent[10], crescent[-10]
axes[1].plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=2, label='Line segment (exits set)')
axes[1].scatter(*p1, color='red', s=100, zorder=5)
axes[1].scatter(*p2, color='red', s=100, zorder=5)
axes[1].set_title('Non-convex set: line segment exits the set')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_aspect('equal')

plt.tight_layout()
plt.show()
```

### 1.2 Convex Functions

A function $f: \mathbb{R}^n \to \mathbb{R}$ is **convex** if:
$$
f(\theta x + (1-\theta) y) \leq \theta f(x) + (1-\theta) f(y), \quad \forall x, y, \forall \theta \in [0, 1]
$$

**First-order condition** (if differentiable):
$$
f(y) \geq f(x) + \nabla f(x)^T (y - x)
$$
(function always lies above its tangent)

**Second-order condition** (if twice differentiable):
$$
\nabla^2 f(x) \succeq 0 \quad \text{(Hessian is positive semidefinite)}
$$

```python
# Convex function vs non-convex function
x = np.linspace(-3, 3, 200)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Convex function: f(x) = x^2
axes[0, 0].plot(x, x**2, 'b-', linewidth=2, label='$f(x) = x^2$')
x0 = 1.0
axes[0, 0].plot(x, x0**2 + 2*x0*(x - x0), 'r--', label=f'Tangent (x={x0})')
axes[0, 0].scatter([x0], [x0**2], color='red', s=100, zorder=5)
axes[0, 0].set_title('Convex function: function lies above tangent')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Non-convex function: f(x) = x^3
axes[0, 1].plot(x, x**3, 'b-', linewidth=2, label='$f(x) = x^3$')
x0 = -1.0
axes[0, 1].plot(x, x0**3 + 3*x0**2*(x - x0), 'r--', label=f'Tangent (x={x0})')
axes[0, 1].scatter([x0], [x0**3], color='red', s=100, zorder=5)
axes[0, 1].set_title('Non-convex function: function dips below tangent')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Strongly convex function: f(x) = x^2 + 2x + 1
axes[1, 0].plot(x, x**2 + 2*x + 1, 'b-', linewidth=2, label='$f(x) = x^2 + 2x + 1$')
minimum = -1  # f'(x) = 2x + 2 = 0
axes[1, 0].scatter([minimum], [minimum**2 + 2*minimum + 1], color='green', s=100,
                   zorder=5, label='Global minimum')
axes[1, 0].set_title('Strongly convex function: unique global minimum')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Hessian check
def hessian_demo():
    """Hessian of a bivariate function"""
    from mpl_toolkits.mplot3d import Axes3D

    x1 = np.linspace(-2, 2, 50)
    x2 = np.linspace(-2, 2, 50)
    X1, X2 = np.meshgrid(x1, x2)

    # Convex: f(x1, x2) = x1^2 + x2^2
    Z = X1**2 + X2**2

    ax = fig.add_subplot(2, 2, 4, projection='3d')
    ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)
    ax.set_title('Bivariate convex function: $f(x_1, x_2) = x_1^2 + x_2^2$')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$f(x_1, x_2)$')

hessian_demo()
plt.tight_layout()
plt.show()

# Check positive semidefiniteness of Hessian
def check_convexity(f, grad, hessian, x):
    """Check convexity via second-order condition"""
    H = hessian(x)
    eigenvalues = np.linalg.eigvals(H)
    print(f"At point {x}:")
    print(f"  Hessian eigenvalues: {eigenvalues}")
    print(f"  All >= 0? {np.all(eigenvalues >= -1e-10)}")
    return np.all(eigenvalues >= -1e-10)

# Example: f(x) = x^T A x (A is positive definite)
A = np.array([[2, 0], [0, 3]])
hessian = lambda x: 2 * A
x_test = np.array([1.0, 1.0])
is_convex = check_convexity(None, None, hessian, x_test)
print(f"\nFunction x^T A x is convex: {is_convex}")
```

### 1.3 Epigraph Perspective

The **epigraph** of function $f$:
$$
\text{epi}(f) = \{(x, t) : f(x) \leq t\}
$$

**Theorem**: $f$ is convex $\Leftrightarrow$ $\text{epi}(f)$ is a convex set

This perspective is useful for geometric understanding of convex functions.

### 1.4 Importance of Convex Optimization

**Key property**: Local minimum = Global minimum

```python
from scipy.optimize import minimize

# Convex function: same minimum regardless of starting point
def convex_func(x):
    return x[0]**2 + x[1]**2 + 2*x[0] + 3*x[1]

starting_points = [
    np.array([10.0, 10.0]),
    np.array([-5.0, 5.0]),
    np.array([0.0, -8.0])
]

print("Convex function optimization:")
for i, x0 in enumerate(starting_points):
    result = minimize(convex_func, x0, method='BFGS')
    print(f"Starting point {i+1} {x0}: minimum at {result.x}, value {result.fun:.4f}")

# Non-convex function: different results depending on starting point
def nonconvex_func(x):
    return np.sin(x[0]) + np.cos(x[1]) + 0.1*(x[0]**2 + x[1]**2)

print("\nNon-convex function optimization:")
for i, x0 in enumerate(starting_points):
    result = minimize(nonconvex_func, x0, method='BFGS')
    print(f"Starting point {i+1} {x0}: minimum at {result.x}, value {result.fun:.4f}")
```

---

## 2. Lagrangian Duality

### 2.1 Primal Problem

**General optimization problem**:
$$
\begin{align}
\min_{x} \quad & f(x) \\
\text{s.t.} \quad & g_i(x) \leq 0, \quad i = 1, \ldots, m \\
& h_j(x) = 0, \quad j = 1, \ldots, p
\end{align}
$$

- $f$: objective function
- $g_i$: inequality constraints
- $h_j$: equality constraints

### 2.2 Lagrangian

**Lagrangian function**:
$$
\mathcal{L}(x, \lambda, \nu) = f(x) + \sum_{i=1}^{m} \lambda_i g_i(x) + \sum_{j=1}^{p} \nu_j h_j(x)
$$

- $\lambda_i \geq 0$: Lagrange multipliers for inequality constraints
- $\nu_j$: Lagrange multipliers for equality constraints

```python
# Simple example: min x^2 + y^2 s.t. x + y = 1
def primal_objective(x):
    return x[0]**2 + x[1]**2

def equality_constraint(x):
    return x[0] + x[1] - 1

def lagrangian(x, nu):
    """Lagrangian: L(x, ν) = x^2 + y^2 + ν(x + y - 1)"""
    return primal_objective(x) + nu * equality_constraint(x)

# Visualization
x_range = np.linspace(-1, 2, 100)
y_range = np.linspace(-1, 2, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = X**2 + Y**2

plt.figure(figsize=(10, 8))
plt.contour(X, Y, Z, levels=20, alpha=0.6)
plt.plot(x_range, 1 - x_range, 'r-', linewidth=3, label='Constraint: x + y = 1')

# Optimal solution (from Lagrange conditions)
# ∇_x L = 2x + ν = 0, ∇_y L = 2y + ν = 0, x + y = 1
# => x = y = 1/2
optimal_x = np.array([0.5, 0.5])
plt.scatter(*optimal_x, color='green', s=200, zorder=5, label=f'Optimal solution {optimal_x}')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Constrained optimization: Lagrange multiplier method')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()

print(f"Optimal solution: x = {optimal_x}")
print(f"Optimal value: f(x*) = {primal_objective(optimal_x):.4f}")
```

### 2.3 Dual Function

**Dual function**:
$$
g(\lambda, \nu) = \inf_{x} \mathcal{L}(x, \lambda, \nu)
$$

**Key property**: $g(\lambda, \nu)$ is always **concave** (even if the primal problem is not convex!)

### 2.4 Weak Duality

**Theorem**: For any $\lambda \geq 0, \nu$
$$
g(\lambda, \nu) \leq p^*
$$
where $p^*$ is the optimal value of the primal problem.

**Proof**:
$$
g(\lambda, \nu) = \inf_{x} \mathcal{L}(x, \lambda, \nu) \leq \mathcal{L}(x^*, \lambda, \nu) = f(x^*) + \sum \lambda_i \underbrace{g_i(x^*)}_{\leq 0} + \sum \nu_j \underbrace{h_j(x^*)}_{=0} \leq f(x^*) = p^*
$$

### 2.5 Strong Duality

**Dual problem**:
$$
\max_{\lambda \geq 0, \nu} \quad g(\lambda, \nu)
$$

**Strong duality**: $p^* = d^*$ (optimal values are equal)

**Slater's condition** (sufficient condition):
- Primal problem is convex
- There exists feasible $x$ such that all $g_i(x) < 0$ (strict inequality)

```python
# Strong duality example: Quadratic programming (QP)
from scipy.optimize import minimize
import cvxpy as cp

# Primal problem: min 1/2 x^T Q x + c^T x, s.t. Ax <= b
Q = np.array([[2, 0], [0, 2]])
c = np.array([1, 1])
A = np.array([[1, 1], [-1, 0], [0, -1]])
b = np.array([1, 0, 0])

# Solve primal problem with CVXPY
x_primal = cp.Variable(2)
objective_primal = cp.Minimize(0.5 * cp.quad_form(x_primal, Q) + c @ x_primal)
constraints_primal = [A @ x_primal <= b]
prob_primal = cp.Problem(objective_primal, constraints_primal)
p_star = prob_primal.solve()

print(f"Primal problem:")
print(f"  Optimal value: p* = {p_star:.4f}")
print(f"  Optimal solution: x* = {x_primal.value}")

# Dual problem (derivation omitted, theoretical)
# max -1/2 λ^T (A Q^{-1} A^T) λ - b^T λ - 1/2 c^T Q^{-1} c, s.t. λ >= 0, A^T λ = -c
lambda_dual = cp.Variable(3)
Q_inv = np.linalg.inv(Q)
objective_dual = cp.Maximize(
    -0.5 * cp.quad_form(lambda_dual, A @ Q_inv @ A.T) - b @ lambda_dual - 0.5 * c @ Q_inv @ c
)
constraints_dual = [lambda_dual >= 0, A.T @ lambda_dual == -c]
prob_dual = cp.Problem(objective_dual, constraints_dual)
d_star = prob_dual.solve()

print(f"\nDual problem:")
print(f"  Optimal value: d* = {d_star:.4f}")
print(f"  Optimal multipliers: λ* = {lambda_dual.value}")

print(f"\nStrong duality check: p* - d* = {p_star - d_star:.6f} ≈ 0")
```

---

## 3. SVM Dual Problem

### 3.1 Primal Problem

**Hard-margin SVM**:
$$
\begin{align}
\min_{w, b} \quad & \frac{1}{2} \|w\|^2 \\
\text{s.t.} \quad & y_i (w^T x_i + b) \geq 1, \quad i = 1, \ldots, n
\end{align}
$$

Goal: Maximize margin $\frac{2}{\|w\|}$ $\Leftrightarrow$ Minimize $\|w\|^2$

### 3.2 Constructing the Lagrangian

$$
\mathcal{L}(w, b, \alpha) = \frac{1}{2} \|w\|^2 - \sum_{i=1}^{n} \alpha_i [y_i(w^T x_i + b) - 1]
$$

where $\alpha_i \geq 0$ are Lagrange multipliers.

### 3.3 KKT Conditions

**1. Stationarity**:
$$
\nabla_w \mathcal{L} = w - \sum_{i=1}^{n} \alpha_i y_i x_i = 0 \quad \Rightarrow \quad w = \sum_{i=1}^{n} \alpha_i y_i x_i
$$

$$
\frac{\partial \mathcal{L}}{\partial b} = -\sum_{i=1}^{n} \alpha_i y_i = 0 \quad \Rightarrow \quad \sum_{i=1}^{n} \alpha_i y_i = 0
$$

**2. Complementary Slackness**:
$$
\alpha_i [y_i(w^T x_i + b) - 1] = 0
$$

**3. Primal/Dual Feasibility**: $y_i(w^T x_i + b) \geq 1$, $\alpha_i \geq 0$

### 3.4 Deriving the Dual Problem

Substituting $w = \sum \alpha_i y_i x_i$ into the Lagrangian:

$$
\begin{align}
\mathcal{L}(w, b, \alpha) &= \frac{1}{2} \left\|\sum_{i} \alpha_i y_i x_i\right\|^2 - \sum_{i,j} \alpha_i \alpha_j y_i y_j (x_i^T x_j) - b\sum_i \alpha_i y_i + \sum_i \alpha_i \\
&= -\frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j (x_i^T x_j) + \sum_i \alpha_i
\end{align}
$$

**Dual problem**:
$$
\begin{align}
\max_{\alpha} \quad & \sum_{i=1}^{n} \alpha_i - \frac{1}{2} \sum_{i,j=1}^{n} \alpha_i \alpha_j y_i y_j (x_i^T x_j) \\
\text{s.t.} \quad & \sum_{i=1}^{n} \alpha_i y_i = 0 \\
& \alpha_i \geq 0, \quad i = 1, \ldots, n
\end{align}
$$

```python
from sklearn.svm import SVC
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# Simple 2D data
np.random.seed(42)
X, y = make_classification(n_samples=100, n_features=2, n_informative=2,
                           n_redundant=0, n_clusters_per_class=1, flip_y=0.1,
                           class_sep=1.5, random_state=42)
y = 2*y - 1  # {0, 1} -> {-1, 1}

# Train SVM (linear kernel)
svm = SVC(kernel='linear', C=1000)  # Large C approximates hard margin
svm.fit(X, y)

# Results
w = svm.coef_[0]
b = svm.intercept_[0]
support_vectors = svm.support_vectors_
alphas = svm.dual_coef_[0]  # α_i * y_i

print("SVM results:")
print(f"  Weight w: {w}")
print(f"  Intercept b: {b}")
print(f"  Number of support vectors: {len(support_vectors)}")
print(f"  Dual coefficients (α_i * y_i): {alphas[:5]}...")  # First 5 only

# Visualization
plt.figure(figsize=(10, 8))
plt.scatter(X[y==1, 0], X[y==1, 1], c='blue', marker='o', label='Class +1')
plt.scatter(X[y==-1, 0], X[y==-1, 1], c='red', marker='s', label='Class -1')
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=200, linewidth=2,
            facecolors='none', edgecolors='green', label='Support vectors')

# Decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x_plot = np.linspace(x_min, x_max, 100)
y_plot = -(w[0] * x_plot + b) / w[1]
plt.plot(x_plot, y_plot, 'k-', linewidth=2, label='Decision boundary')

# Margin
margin = 1 / np.linalg.norm(w)
y_plot_margin_up = y_plot + margin * np.sqrt(1 + (w[0]/w[1])**2)
y_plot_margin_down = y_plot - margin * np.sqrt(1 + (w[0]/w[1])**2)
plt.plot(x_plot, y_plot_margin_up, 'k--', linewidth=1, alpha=0.5)
plt.plot(x_plot, y_plot_margin_down, 'k--', linewidth=1, alpha=0.5)
plt.fill_between(x_plot, y_plot_margin_down, y_plot_margin_up, alpha=0.1, color='gray')

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title(f'SVM: margin = {2*margin:.3f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 3.5 Natural Emergence of the Kernel Trick

In the dual problem, data appears **only through inner products $x_i^T x_j$**:

$$
\max_{\alpha} \quad \sum_{i} \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j \underbrace{(x_i^T x_j)}_{k(x_i, x_j)}
$$

**Kernel trick**: $k(x_i, x_j) = \phi(x_i)^T \phi(x_j)$ (high-dimensional feature space)

**Example**: Polynomial kernel $k(x, x') = (x^T x' + 1)^2$

```python
# Nonlinear SVM (RBF kernel)
from sklearn.datasets import make_moons

X_nonlinear, y_nonlinear = make_moons(n_samples=200, noise=0.15, random_state=42)
y_nonlinear = 2*y_nonlinear - 1

svm_rbf = SVC(kernel='rbf', gamma=2, C=1.0)
svm_rbf.fit(X_nonlinear, y_nonlinear)

# Decision boundary visualization
h = 0.02
x_min, x_max = X_nonlinear[:, 0].min() - 0.5, X_nonlinear[:, 0].max() + 0.5
y_min, y_max = X_nonlinear[:, 1].min() - 0.5, X_nonlinear[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = svm_rbf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.6)
plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
plt.scatter(X_nonlinear[y_nonlinear==1, 0], X_nonlinear[y_nonlinear==1, 1],
            c='blue', marker='o', edgecolors='k', label='Class +1')
plt.scatter(X_nonlinear[y_nonlinear==-1, 0], X_nonlinear[y_nonlinear==-1, 1],
            c='red', marker='s', edgecolors='k', label='Class -1')
plt.scatter(svm_rbf.support_vectors_[:, 0], svm_rbf.support_vectors_[:, 1],
            s=200, linewidth=2, facecolors='none', edgecolors='green', label='Support vectors')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Nonlinear SVM (RBF kernel)')
plt.legend()
plt.show()

print(f"Number of support vectors: {len(svm_rbf.support_vectors_)}")
```

---

## 4. Fenchel Conjugate

### 4.1 Definition

The **Fenchel conjugate** of function $f: \mathbb{R}^n \to \mathbb{R}$:
$$
f^*(y) = \sup_{x} \left( y^T x - f(x) \right)
$$

**Intuition**: $y$ is the slope, $f^*(y)$ is the y-intercept of the tangent line with slope $y$

### 4.2 Geometric Interpretation

```python
# Fenchel conjugate visualization
def fenchel_conjugate_demo():
    x = np.linspace(-3, 3, 200)
    f_x = x**2  # f(x) = x^2

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Original function
    axes[0].plot(x, f_x, 'b-', linewidth=2, label='$f(x) = x^2$')

    # Tangent lines with various slopes
    slopes = [-2, 0, 2, 4]
    colors = ['red', 'green', 'orange', 'purple']

    for y, color in zip(slopes, colors):
        # f*(y) = sup_x (yx - x^2)
        # Maximize: d/dx (yx - x^2) = y - 2x = 0 => x* = y/2
        x_star = y / 2
        f_star_y = y * x_star - x_star**2  # = y^2/4

        # Tangent line: yx - f*(y)
        tangent = y * x - f_star_y

        axes[0].plot(x, tangent, color=color, linestyle='--', alpha=0.7,
                     label=f'y={y}: $f^*({y})={f_star_y:.2f}$')
        axes[0].scatter([x_star], [f_x[np.abs(x - x_star).argmin()]],
                        color=color, s=100, zorder=5)

    axes[0].set_xlabel('x')
    axes[0].set_ylabel('f(x)')
    axes[0].set_title('Fenchel conjugate: tangent line perspective')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-2, 10)

    # Conjugate function
    y_range = np.linspace(-4, 4, 100)
    f_star = y_range**2 / 4  # f*(y) = y^2/4 for f(x) = x^2

    axes[1].plot(y_range, f_star, 'b-', linewidth=2, label='$f^*(y) = y^2/4$')
    axes[1].scatter(slopes, [s**2/4 for s in slopes], color=colors, s=100, zorder=5)
    axes[1].set_xlabel('y (slope)')
    axes[1].set_ylabel('$f^*(y)$')
    axes[1].set_title('Fenchel conjugate function')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

fenchel_conjugate_demo()
```

### 4.3 Important Examples

**1. Conjugate of norm**: $f(x) = \frac{1}{2}\|x\|_2^2$
$$
f^*(y) = \frac{1}{2}\|y\|_2^2
$$

**2. Conjugate of indicator function**: $f(x) = I_C(x)$ (indicator function of set $C$)
$$
f^*(y) = \sup_{x \in C} y^T x \quad \text{(support function)}
$$

**3. Conjugate of log-sum-exp**: $f(x) = \log(\sum_i e^{x_i})$
$$
f^*(y) = \sum_i y_i \log y_i \quad \text{(negative entropy)}
$$

```python
# Fenchel conjugate computation examples
def conjugate_squared_norm(y):
    """Conjugate of f(x) = 1/2 ||x||^2"""
    return 0.5 * np.linalg.norm(y)**2

def conjugate_indicator_ball(y, radius=1.0):
    """Conjugate of f(x) = I_{||x|| <= r} (support function)"""
    return radius * np.linalg.norm(y)

# Verification: f**(x) = f(x) (biconjugate theorem)
x_test = np.array([1.0, 2.0])
f_x = 0.5 * np.linalg.norm(x_test)**2

# Numerically compute f**
from scipy.optimize import minimize_scalar

def double_conjugate(x):
    """Biconjugate f**(x) = sup_y (x^T y - f*(y))"""
    # Simplified to 1D (illustrative)
    def negative_objective(y_norm):
        y = y_norm * x / np.linalg.norm(x)  # Fix direction
        return -(np.dot(x, y) - conjugate_squared_norm(y))

    result = minimize_scalar(negative_objective, bounds=(0, 10), method='bounded')
    return -result.fun

f_double_star_x = double_conjugate(x_test)
print(f"Biconjugate theorem verification:")
print(f"  f(x) = {f_x:.4f}")
print(f"  f**(x) = {f_double_star_x:.4f}")
print(f"  Matches: {np.isclose(f_x, f_double_star_x)}")
```

---

## 5. Proximal Operator

### 5.1 Definition

The **proximal operator** of function $f$:
$$
\text{prox}_f(v) = \arg\min_{x} \left( f(x) + \frac{1}{2}\|x - v\|^2 \right)
$$

**Intuition**: Find $x$ that is close to $v$ while minimizing $f$

### 5.2 Proximal Operator of L1 Norm (Soft Thresholding)

For $f(x) = \lambda \|x\|_1$:
$$
[\text{prox}_{\lambda \|\cdot\|_1}(v)]_i = \text{sign}(v_i) \max(|v_i| - \lambda, 0)
$$

This is called **soft thresholding**.

```python
def soft_threshold(v, lambda_param):
    """L1 proximal operator (soft thresholding)"""
    return np.sign(v) * np.maximum(np.abs(v) - lambda_param, 0)

# Visualization
v = np.linspace(-3, 3, 200)
lambda_values = [0.5, 1.0, 1.5]

plt.figure(figsize=(10, 6))
plt.plot(v, v, 'k--', label='Identity function (λ=0)', alpha=0.5)

for lam in lambda_values:
    prox_v = soft_threshold(v, lam)
    plt.plot(v, prox_v, linewidth=2, label=f'λ={lam}')

plt.xlabel('v')
plt.ylabel('prox(v)')
plt.title('L1 proximal operator (soft thresholding)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.show()
```

### 5.3 Proximal Gradient Descent

**Problem**: $\min f(x) + g(x)$, where $f$ is smooth and $g$ is non-smooth (e.g., L1)

**Algorithm**:
$$
x_{k+1} = \text{prox}_{\alpha g}(x_k - \alpha \nabla f(x_k))
$$

```python
# LASSO problem: min 1/2 ||Ax - b||^2 + λ||x||_1
def lasso_proximal_gradient(A, b, lambda_param, max_iter=1000, alpha=0.01):
    """Solve LASSO via proximal gradient descent"""
    n = A.shape[1]
    x = np.zeros(n)
    losses = []

    for k in range(max_iter):
        # Gradient of smooth part: ∇(1/2 ||Ax - b||^2) = A^T(Ax - b)
        residual = A @ x - b
        grad_f = A.T @ residual

        # Gradient descent step
        x_temp = x - alpha * grad_f

        # Proximal operation (L1)
        x = soft_threshold(x_temp, alpha * lambda_param)

        # Compute loss
        loss = 0.5 * np.sum(residual**2) + lambda_param * np.sum(np.abs(x))
        losses.append(loss)

        if k > 0 and abs(losses[-1] - losses[-2]) < 1e-6:
            break

    return x, losses

# Test: sparse regression
np.random.seed(42)
n, p = 100, 50
A = np.random.randn(n, p)
x_true = np.zeros(p)
x_true[:10] = np.random.randn(10)  # Only 10 are nonzero
b = A @ x_true + 0.1 * np.random.randn(n)

lambda_param = 0.1
x_lasso, losses = lasso_proximal_gradient(A, b, lambda_param, max_iter=1000, alpha=0.001)

# Results
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss function
axes[0].plot(losses)
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Loss')
axes[0].set_title('LASSO proximal gradient descent: convergence')
axes[0].set_yscale('log')
axes[0].grid(True, alpha=0.3)

# Coefficient comparison
axes[1].stem(x_true, linefmt='b-', markerfmt='bo', basefmt=' ', label='True coefficients')
axes[1].stem(x_lasso, linefmt='r-', markerfmt='rs', basefmt=' ', label='Estimated coefficients (LASSO)')
axes[1].set_xlabel('Coefficient index')
axes[1].set_ylabel('Value')
axes[1].set_title(f'LASSO result (λ={lambda_param})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Number of nonzero coefficients:")
print(f"  True: {np.sum(x_true != 0)}")
print(f"  Estimated: {np.sum(np.abs(x_lasso) > 1e-3)}")
```

---

## 6. Machine Learning Applications

### 6.1 Dual Interpretation of Logistic Regression

Logistic regression:
$$
\min_{w} \sum_{i=1}^{n} \log(1 + e^{-y_i w^T x_i}) + \frac{\lambda}{2} \|w\|^2
$$

The dual problem connects to **entropy maximization**.

### 6.2 ADMM (Alternating Direction Method of Multipliers)

**Problem**: $\min f(x) + g(z)$ s.t. $Ax + Bz = c$

**ADMM algorithm**:
1. $x_{k+1} = \arg\min_x \mathcal{L}_\rho(x, z_k, \lambda_k)$
2. $z_{k+1} = \arg\min_z \mathcal{L}_\rho(x_{k+1}, z, \lambda_k)$
3. $\lambda_{k+1} = \lambda_k + \rho(Ax_{k+1} + Bz_{k+1} - c)$

where $\mathcal{L}_\rho$ is the augmented Lagrangian.

```python
# Solve LASSO with ADMM
def lasso_admm(A, b, lambda_param, rho=1.0, max_iter=100):
    """ADMM for LASSO: min 1/2||Ax-b||^2 + λ||z||_1, s.t. x = z"""
    n, p = A.shape
    x = np.zeros(p)
    z = np.zeros(p)
    u = np.zeros(p)  # Scaled dual variable

    # Precompute
    AtA = A.T @ A
    Atb = A.T @ b
    I = np.eye(p)

    losses = []

    for k in range(max_iter):
        # x update: (A^T A + ρI)x = A^T b + ρ(z - u)
        x = np.linalg.solve(AtA + rho * I, Atb + rho * (z - u))

        # z update: soft thresholding
        z_old = z.copy()
        z = soft_threshold(x + u, lambda_param / rho)

        # u update (dual variable)
        u = u + x - z

        # Loss
        loss = 0.5 * np.linalg.norm(A @ x - b)**2 + lambda_param * np.linalg.norm(z, 1)
        losses.append(loss)

        # Convergence check
        if np.linalg.norm(z - z_old) < 1e-4:
            break

    return x, z, losses

# Comparison: Proximal Gradient vs ADMM
x_admm, z_admm, losses_admm = lasso_admm(A, b, lambda_param, rho=1.0, max_iter=100)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(losses, label='Proximal gradient descent')
plt.plot(losses_admm, label='ADMM')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Convergence speed comparison')
plt.legend()
plt.yscale('log')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.stem(x_lasso, linefmt='b-', markerfmt='bo', basefmt=' ', label='Proximal GD')
plt.stem(z_admm, linefmt='r-', markerfmt='rs', basefmt=' ', label='ADMM')
plt.xlabel('Coefficient index')
plt.ylabel('Value')
plt.title('Final coefficient comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Coefficient difference (L2): {np.linalg.norm(x_lasso - z_admm):.6f}")
```

### 6.3 Real-World Applications

- **SVM**: Efficient solution for large-scale classification via dual problem
- **L1 regularization**: Proximal operators in LASSO, Elastic Net
- **Distributed optimization**: ADMM for large-scale parallel data processing
- **Deep learning**: Fenchel duality used in GANs, f-divergence

---

## Practice Problems

### Problem 1: Proving Convexity
Verify whether the following functions are convex:
(a) $f(x) = e^x$
(b) $f(x) = -\log(x)$ (x > 0)
(c) $f(x, y) = x^2 + xy + y^2$

Prove each using the second-order condition (Hessian).

### Problem 2: Lagrangian Duality
Derive the Lagrangian and dual function for the following problem:
$$
\min x^2 + y^2 \quad \text{s.t.} \quad x + y \geq 1, \quad x, y \geq 0
$$

Apply KKT conditions to find the optimal solution.

### Problem 3: SVM Implementation
Implement linear SVM from scratch (cvxpy allowed).
(a) Solve both primal and dual problems
(b) Verify that both solutions satisfy strong duality
(c) Identify and visualize support vectors

### Problem 4: Fenchel Conjugate
Derive the Fenchel conjugate for the following functions:
(a) $f(x) = \frac{1}{2}x^2 + x$ (1D)
(b) $f(x) = |x|$ (L1 norm)
(c) $f(x) = \max(0, x)$ (ReLU)

### Problem 5: ADMM Application
Solve a distributed linear regression problem using ADMM:
$$
\min \sum_{i=1}^{N} \frac{1}{2}\|A_i x - b_i\|^2
$$

Assume data is distributed across $N$ nodes.

---

## References

### Books
- **Convex Optimization** (Boyd & Vandenberghe, 2004) - Standard textbook
- **Proximal Algorithms** (Parikh & Boyd, 2014) - Comprehensive treatment of proximal methods
- **Online Convex Optimization** (Hazan, 2016) - Connection to online learning

### Papers
- Vapnik (1995), "The Nature of Statistical Learning Theory" - SVM foundations
- Boyd et al. (2011), "Distributed Optimization and Statistical Learning via ADMM"
- Rockafellar (1970), "Convex Analysis" - Theory of Fenchel conjugate

### Online Resources
- [Stanford EE364a: Convex Optimization I](https://web.stanford.edu/class/ee364a/)
- [Boyd's ADMM page](https://web.stanford.edu/~boyd/papers/admm_distr_stats.html)
- [CVXPY Tutorial](https://www.cvxpy.org/tutorial/index.html)

### Libraries
- **CVXPY**: Python convex optimization
- **scikit-learn**: SVM, LASSO implementations
- **ProximalOperators.jl**: Julia proximal operator library
