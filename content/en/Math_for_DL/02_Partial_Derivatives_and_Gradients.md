# Lesson 2: Partial Derivatives and Gradients

## Learning Objectives

- Compute partial derivatives of multivariable functions analytically and numerically
- Construct the gradient vector and interpret it geometrically as the direction of steepest ascent
- Calculate directional derivatives and understand their relationship to the gradient
- Visualize gradient fields and level curves of loss landscapes
- Apply the gradient to simple optimization problems (gradient descent on a quadratic)
- Verify analytical gradients using finite difference methods
- Understand why gradient descent works by connecting gradients to first-order Taylor approximations

---

## 1. From Single-Variable to Multivariable

In single-variable calculus, the derivative $f'(x)$ tells us the rate of change of $f$ as we move along the number line. In deep learning, our loss function depends on millions of parameters simultaneously. We need to generalize differentiation to functions of many variables.

### 1.1 Multivariable Functions

A function $f: \mathbb{R}^n \to \mathbb{R}$ takes a vector $\mathbf{x} = (x_1, x_2, \ldots, x_n)$ and returns a scalar. In deep learning:

- $\mathbf{x}$ represents all model parameters (weights and biases flattened into one vector)
- $f(\mathbf{x}) = L(\mathbf{x})$ is the loss function

**Example**: A simple 2D loss surface:

$$f(x_1, x_2) = x_1^2 + 3x_2^2 - 2x_1 x_2 + x_1 - 4x_2 + 5$$

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x1, x2):
    return x1**2 + 3*x2**2 - 2*x1*x2 + x1 - 4*x2 + 5

# Visualize as a surface
x1 = np.linspace(-3, 3, 200)
x2 = np.linspace(-2, 4, 200)
X1, X2 = np.meshgrid(x1, x2)
Z = f(X1, X2)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Contour plot (level curves)
cs = axes[0].contour(X1, X2, Z, levels=20)
axes[0].clabel(cs, inline=True, fontsize=8)
axes[0].set_xlabel('$x_1$')
axes[0].set_ylabel('$x_2$')
axes[0].set_title('Level curves of $f(x_1, x_2)$')

# Surface plot
ax3d = fig.add_subplot(122, projection='3d')
ax3d.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.8)
ax3d.set_xlabel('$x_1$')
ax3d.set_ylabel('$x_2$')
ax3d.set_zlabel('$f$')
ax3d.set_title('Loss surface')
plt.tight_layout()
plt.show()
```

---

## 2. Partial Derivatives

### 2.1 Definition

The **partial derivative** of $f$ with respect to $x_i$ measures how $f$ changes when we vary $x_i$ alone, holding all other variables fixed:

$$\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \ldots, x_i + h, \ldots, x_n) - f(x_1, \ldots, x_n)}{h}$$

**Intuition**: Imagine standing on a mountain (the loss surface). The partial derivative $\frac{\partial f}{\partial x_1}$ tells you how steep the slope is if you walk only in the $x_1$ direction.

### 2.2 Computing Partial Derivatives

For $f(x_1, x_2) = x_1^2 + 3x_2^2 - 2x_1 x_2 + x_1 - 4x_2 + 5$:

$$\frac{\partial f}{\partial x_1} = 2x_1 - 2x_2 + 1$$

$$\frac{\partial f}{\partial x_2} = 6x_2 - 2x_1 - 4$$

To compute $\frac{\partial f}{\partial x_1}$, we treat $x_2$ as a constant:
- $x_1^2 \to 2x_1$
- $3x_2^2 \to 0$ (constant)
- $-2x_1 x_2 \to -2x_2$ (treat $x_2$ as a constant coefficient)
- $x_1 \to 1$
- $-4x_2 \to 0$
- $5 \to 0$

### 2.3 Numerical Partial Derivatives

In practice, we verify analytical gradients using **finite differences**:

**Forward difference**: $\frac{\partial f}{\partial x_i} \approx \frac{f(\mathbf{x} + h\mathbf{e}_i) - f(\mathbf{x})}{h}$ -- accuracy $O(h)$

**Central difference**: $\frac{\partial f}{\partial x_i} \approx \frac{f(\mathbf{x} + h\mathbf{e}_i) - f(\mathbf{x} - h\mathbf{e}_i)}{2h}$ -- accuracy $O(h^2)$

where $\mathbf{e}_i$ is the $i$-th standard basis vector.

```python
def f_vec(x):
    """f as a function of a vector."""
    return x[0]**2 + 3*x[1]**2 - 2*x[0]*x[1] + x[0] - 4*x[1] + 5

def grad_f_analytical(x):
    """Analytical gradient."""
    return np.array([
        2*x[0] - 2*x[1] + 1,
        6*x[1] - 2*x[0] - 4
    ])

def grad_f_numerical(x, h=1e-5):
    """Central difference gradient."""
    n = len(x)
    grad = np.zeros(n)
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        grad[i] = (f_vec(x + h * e_i) - f_vec(x - h * e_i)) / (2 * h)
    return grad

# Compare at a test point
x_test = np.array([1.0, 2.0])
g_ana = grad_f_analytical(x_test)
g_num = grad_f_numerical(x_test)

print(f"Analytical gradient: {g_ana}")
print(f"Numerical gradient:  {g_num}")
print(f"Max difference: {np.max(np.abs(g_ana - g_num)):.2e}")
```

---

## 3. The Gradient Vector

### 3.1 Definition

The **gradient** of $f: \mathbb{R}^n \to \mathbb{R}$ at point $\mathbf{x}$ is the vector of all partial derivatives:

$$\nabla f(\mathbf{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

### 3.2 Geometric Interpretation

The gradient has two crucial geometric properties:

1. **Direction**: $\nabla f(\mathbf{x})$ points in the direction of **steepest ascent** of $f$ at $\mathbf{x}$
2. **Magnitude**: $\|\nabla f(\mathbf{x})\|$ equals the rate of change in that steepest direction
3. **Perpendicularity**: $\nabla f(\mathbf{x})$ is perpendicular to the level curve $f(\mathbf{x}) = c$ at point $\mathbf{x}$

**Consequence**: To minimize $f$, we should move in the direction $-\nabla f(\mathbf{x})$.

```python
# Visualize gradient field on level curves
x1 = np.linspace(-3, 3, 200)
x2 = np.linspace(-2, 4, 200)
X1, X2 = np.meshgrid(x1, x2)
Z = f(X1, X2)

# Gradient components
dfdx1 = 2*X1 - 2*X2 + 1
dfdx2 = 6*X2 - 2*X1 - 4

fig, ax = plt.subplots(figsize=(8, 6))
ax.contour(X1, X2, Z, levels=20, alpha=0.6)

# Sample gradient arrows (subsample for clarity)
stride = 25
ax.quiver(X1[::stride, ::stride], X2[::stride, ::stride],
          -dfdx1[::stride, ::stride], -dfdx2[::stride, ::stride],
          color='red', alpha=0.7, scale=50)

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('Negative gradient field (descent directions)')
plt.show()
```

### 3.3 Why the Gradient Points Uphill: First-Order Taylor Approximation

The connection between gradients and optimization comes from the Taylor expansion:

$$f(\mathbf{x} + \boldsymbol{\delta}) \approx f(\mathbf{x}) + \nabla f(\mathbf{x})^\top \boldsymbol{\delta}$$

The change in $f$ is approximately $\nabla f(\mathbf{x})^\top \boldsymbol{\delta}$. For a unit step $\|\boldsymbol{\delta}\| = 1$, this is maximized when $\boldsymbol{\delta}$ is parallel to $\nabla f(\mathbf{x})$ (by Cauchy-Schwarz), confirming that the gradient is the steepest ascent direction.

To **decrease** $f$ most rapidly, choose $\boldsymbol{\delta} = -\eta \nabla f(\mathbf{x})$ for some small $\eta > 0$.

---

## 4. Directional Derivatives

### 4.1 Definition

The **directional derivative** of $f$ at $\mathbf{x}$ in the direction of unit vector $\mathbf{u}$ is:

$$D_\mathbf{u} f(\mathbf{x}) = \lim_{h \to 0} \frac{f(\mathbf{x} + h\mathbf{u}) - f(\mathbf{x})}{h} = \nabla f(\mathbf{x})^\top \mathbf{u}$$

This measures the rate of change of $f$ when we walk in direction $\mathbf{u}$.

### 4.2 Key Properties

- Maximum directional derivative: along $\mathbf{u} = \frac{\nabla f}{\|\nabla f\|}$, giving $D_\mathbf{u} f = \|\nabla f\|$
- Minimum directional derivative: along $\mathbf{u} = -\frac{\nabla f}{\|\nabla f\|}$, giving $D_\mathbf{u} f = -\|\nabla f\|$
- Zero directional derivative: along any direction perpendicular to $\nabla f$

```python
# Directional derivative computation
x = np.array([1.0, 2.0])
grad = grad_f_analytical(x)

# Various directions
directions = {
    'gradient direction': grad / np.linalg.norm(grad),
    'negative gradient': -grad / np.linalg.norm(grad),
    'perpendicular': np.array([-grad[1], grad[0]]) / np.linalg.norm(grad),
    'x1 axis': np.array([1.0, 0.0]),
    'x2 axis': np.array([0.0, 1.0]),
}

print(f"Gradient at x = {x}: {grad}")
print(f"Gradient norm: {np.linalg.norm(grad):.4f}")
print()
for name, u in directions.items():
    dd = grad @ u
    print(f"  Direction '{name}': u = {u}, D_u f = {dd:.4f}")
```

---

## 5. Gradient Descent

### 5.1 The Algorithm

Gradient descent iteratively updates parameters to minimize a function:

$$\mathbf{x}^{(t+1)} = \mathbf{x}^{(t)} - \eta \nabla f(\mathbf{x}^{(t)})$$

where $\eta > 0$ is the **learning rate**.

### 5.2 Gradient Descent on a Quadratic

For a quadratic $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top \mathbf{A} \mathbf{x} - \mathbf{b}^\top \mathbf{x} + c$ with symmetric positive definite $\mathbf{A}$:

$$\nabla f(\mathbf{x}) = \mathbf{A}\mathbf{x} - \mathbf{b}$$

The minimum is at $\mathbf{x}^* = \mathbf{A}^{-1}\mathbf{b}$.

**Convergence rate** depends on the **condition number** $\kappa = \lambda_{\max}(\mathbf{A}) / \lambda_{\min}(\mathbf{A})$:
- $\kappa \approx 1$: rapid convergence (nearly circular level curves)
- $\kappa \gg 1$: slow convergence (elongated elliptical level curves)

```python
# Gradient descent on a 2D quadratic
A = np.array([[4.0, 1.0],
              [1.0, 2.0]])  # Symmetric positive definite
b_vec = np.array([1.0, 3.0])

# Optimal solution
x_star = np.linalg.solve(A, b_vec)
print(f"Optimal solution: {x_star}")

# Condition number
eigenvalues = np.linalg.eigvalsh(A)
kappa = eigenvalues[-1] / eigenvalues[0]
print(f"Condition number: {kappa:.2f}")

# Gradient descent
eta = 0.15  # Learning rate
x = np.array([3.0, -1.0])  # Starting point
trajectory = [x.copy()]

for t in range(50):
    grad = A @ x - b_vec
    x = x - eta * grad
    trajectory.append(x.copy())

trajectory = np.array(trajectory)

# Visualize trajectory on contour plot
x1 = np.linspace(-2, 4, 200)
x2 = np.linspace(-2, 4, 200)
X1, X2 = np.meshgrid(x1, x2)
Z = 0.5 * (A[0,0]*X1**2 + 2*A[0,1]*X1*X2 + A[1,1]*X2**2) - b_vec[0]*X1 - b_vec[1]*X2

fig, ax = plt.subplots(figsize=(8, 6))
ax.contour(X1, X2, Z, levels=30, alpha=0.6)
ax.plot(trajectory[:, 0], trajectory[:, 1], 'ro-', markersize=3, linewidth=1)
ax.plot(x_star[0], x_star[1], 'g*', markersize=15, label='Optimum')
ax.plot(trajectory[0, 0], trajectory[0, 1], 'bs', markersize=10, label='Start')
ax.legend()
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title(f'Gradient descent (lr={eta}, κ={kappa:.1f})')
plt.show()
```

### 5.3 Learning Rate Selection

The learning rate must satisfy $\eta < \frac{2}{\lambda_{\max}(\mathbf{A})}$ for convergence on a quadratic. The optimal fixed learning rate is:

$$\eta^* = \frac{2}{\lambda_{\max} + \lambda_{\min}}$$

```python
# Effect of learning rate
lrs = [0.05, 0.15, 0.3, 0.45]

fig, axes = plt.subplots(1, len(lrs), figsize=(20, 4))
for ax, eta in zip(axes, lrs):
    x = np.array([3.0, -1.0])
    traj = [x.copy()]
    for _ in range(30):
        grad = A @ x - b_vec
        x = x - eta * grad
        traj.append(x.copy())
        if np.linalg.norm(x) > 1e6:
            break
    traj = np.array(traj)
    ax.contour(X1, X2, Z, levels=30, alpha=0.4)
    ax.plot(traj[:, 0], traj[:, 1], 'ro-', markersize=2)
    ax.plot(x_star[0], x_star[1], 'g*', markersize=12)
    ax.set_title(f'η = {eta}')
    ax.set_xlim(-3, 5)
    ax.set_ylim(-3, 5)
plt.tight_layout()
plt.show()
```

---

## 6. Gradients of Common DL Functions

### 6.1 Sigmoid

$$\sigma(x) = \frac{1}{1 + e^{-x}}, \quad \sigma'(x) = \sigma(x)(1 - \sigma(x))$$

### 6.2 ReLU

$$\text{ReLU}(x) = \max(0, x), \quad \text{ReLU}'(x) = \begin{cases} 1 & x > 0 \\ 0 & x < 0 \\ \text{undefined} & x = 0 \end{cases}$$

In practice, the subgradient at $x = 0$ is set to 0 (or sometimes 1).

### 6.3 Softplus (Smooth ReLU)

$$\text{softplus}(x) = \ln(1 + e^x), \quad \text{softplus}'(x) = \sigma(x)$$

### 6.4 MSE Loss

$$L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2, \quad \frac{\partial L}{\partial \hat{y}_i} = \frac{2}{n}(\hat{y}_i - y_i)$$

```python
# Visualize activation functions and their gradients
x = np.linspace(-5, 5, 500)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Sigmoid
sig = 1 / (1 + np.exp(-x))
axes[0, 0].plot(x, sig)
axes[0, 0].set_title('Sigmoid')
axes[1, 0].plot(x, sig * (1 - sig))
axes[1, 0].set_title("Sigmoid' (gradient)")

# ReLU
relu = np.maximum(0, x)
axes[0, 1].plot(x, relu)
axes[0, 1].set_title('ReLU')
axes[1, 1].plot(x, (x > 0).astype(float))
axes[1, 1].set_title("ReLU' (gradient)")

# Softplus
softplus = np.log1p(np.exp(x))
axes[0, 2].plot(x, softplus)
axes[0, 2].set_title('Softplus')
axes[1, 2].plot(x, sig)
axes[1, 2].set_title("Softplus' = Sigmoid")

for ax in axes.flat:
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
plt.tight_layout()
plt.show()
```

---

## 7. The Gradient in High Dimensions

### 7.1 Curse of Dimensionality for Gradient Computation

For a function with $n$ parameters, computing the full gradient requires $n$ partial derivatives. A finite-difference gradient needs $2n$ function evaluations (central differences). For modern neural networks with $n \sim 10^9$, this is prohibitive.

**This is why backpropagation (reverse-mode automatic differentiation) is so important** -- it computes the full gradient in time proportional to a single forward pass, regardless of $n$. We will derive this in Lesson 03.

### 7.2 Gradient Norms During Training

Monitoring $\|\nabla L\|$ during training reveals important dynamics:

- **Gradient vanishing**: $\|\nabla L\| \to 0$ prematurely, training stalls
- **Gradient exploding**: $\|\nabla L\| \to \infty$, training diverges
- **Healthy training**: $\|\nabla L\|$ decreases gradually as we approach a minimum

```python
# Simulate gradient norm during training
np.random.seed(42)
n_steps = 200

# Healthy training on a quadratic
A = np.eye(10) * 2
b_vec = np.random.randn(10)
x = np.random.randn(10) * 5

grad_norms = []
losses = []
eta = 0.1

for t in range(n_steps):
    grad = A @ x - b_vec
    loss = 0.5 * x @ A @ x - b_vec @ x
    grad_norms.append(np.linalg.norm(grad))
    losses.append(loss)
    x = x - eta * grad

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].semilogy(losses)
axes[0].set_xlabel('Step')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss curve')
axes[1].semilogy(grad_norms)
axes[1].set_xlabel('Step')
axes[1].set_ylabel('||∇L||')
axes[1].set_title('Gradient norm')
plt.tight_layout()
plt.show()
```

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| Partial derivative | Rate of change of $f$ when varying one variable while holding others fixed |
| Gradient | Vector of all partial derivatives; points in steepest ascent direction |
| Directional derivative | $D_\mathbf{u} f = \nabla f \cdot \mathbf{u}$; projects gradient onto direction $\mathbf{u}$ |
| Gradient descent | $\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f$; convergence depends on condition number |
| Finite differences | Central differences have $O(h^2)$ accuracy; use for gradient checking |
| Gradient monitoring | Track $\|\nabla L\|$ to detect vanishing/exploding gradients |

---

## Exercises

1. Compute the gradient of $f(x_1, x_2, x_3) = x_1 x_2 + x_2 x_3^2 - \ln(x_1)$ analytically and verify numerically.
2. Implement gradient descent to minimize the Rosenbrock function $f(x, y) = (1 - x)^2 + 100(y - x^2)^2$.
3. Experiment with different learning rates on the Rosenbrock function and plot convergence curves.
4. Compute the directional derivative of $f(x, y) = e^{xy} + \sin(x + y)$ at $(0, \pi)$ in the direction $(\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}})$.
5. Implement a gradient checker that compares analytical and numerical gradients with a relative error metric.

---

**Next**: [03. Chain Rule and Computation Graphs](03_Chain_Rule_and_Computation_Graphs.md)
