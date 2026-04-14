# Lesson 5: Optimization Theory

## Learning Objectives

- Define convex sets, convex functions, and their relevance to optimization
- Prove that every local minimum of a convex function is a global minimum
- Characterize saddle points and their prevalence in high-dimensional optimization
- State convergence conditions for gradient descent on smooth and strongly convex functions
- Analyze the convergence of SGD with decreasing learning rates
- Understand momentum methods from the perspective of damped oscillation
- Compare first-order optimizers (SGD, momentum, Adam) on pathological loss surfaces
- Recognize the role of learning rate schedules and warm-up in training stability

---

## 1. Convexity

### 1.1 Convex Sets

A set $S \subseteq \mathbb{R}^n$ is **convex** if for any two points $\mathbf{x}, \mathbf{y} \in S$ and any $\lambda \in [0, 1]$:

$$\lambda \mathbf{x} + (1 - \lambda) \mathbf{y} \in S$$

In other words, the line segment connecting any two points in $S$ lies entirely within $S$.

### 1.2 Convex Functions

A function $f: \mathbb{R}^n \to \mathbb{R}$ is **convex** if for all $\mathbf{x}, \mathbf{y}$ and $\lambda \in [0, 1]$:

$$f(\lambda \mathbf{x} + (1 - \lambda)\mathbf{y}) \leq \lambda f(\mathbf{x}) + (1 - \lambda) f(\mathbf{y})$$

**Geometric meaning**: The function lies below (or on) the chord between any two points.

**Equivalent conditions** (for twice-differentiable $f$):
1. $\nabla^2 f(\mathbf{x}) \succeq 0$ for all $\mathbf{x}$ (Hessian is positive semi-definite)
2. $f(\mathbf{y}) \geq f(\mathbf{x}) + \nabla f(\mathbf{x})^\top (\mathbf{y} - \mathbf{x})$ (function lies above every tangent hyperplane)

### 1.3 Strong Convexity

$f$ is **$\mu$-strongly convex** if $f(\mathbf{x}) - \frac{\mu}{2}\|\mathbf{x}\|^2$ is convex, or equivalently:

$$\nabla^2 f(\mathbf{x}) \succeq \mu \mathbf{I} \quad \forall \mathbf{x}$$

Strong convexity guarantees a unique global minimum and faster convergence.

```python
import numpy as np
import matplotlib.pyplot as plt

# Convex vs non-convex functions in 1D
x = np.linspace(-3, 3, 500)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Convex: quadratic
axes[0].plot(x, x**2, 'b-', linewidth=2)
axes[0].set_title('Convex: $f(x) = x^2$')

# Convex but not strongly: absolute value
axes[1].plot(x, np.abs(x), 'b-', linewidth=2)
axes[1].set_title('Convex (not strongly): $f(x) = |x|$')

# Non-convex: typical DL loss
f_nc = np.sin(3*x) + 0.5*x**2 - x
axes[2].plot(x, f_nc, 'r-', linewidth=2)
axes[2].set_title('Non-convex (typical DL)')

for ax in axes:
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### 1.4 The Key Theorem

> **Theorem**: If $f$ is convex, then every local minimum is a global minimum. If $f$ is strictly convex, the global minimum is unique.

**Proof sketch**: Suppose $\mathbf{x}^*$ is a local minimum but not a global minimum. Then there exists $\mathbf{y}$ with $f(\mathbf{y}) < f(\mathbf{x}^*)$. By convexity, for any $\lambda \in (0, 1)$:

$$f(\lambda \mathbf{y} + (1-\lambda)\mathbf{x}^*) \leq \lambda f(\mathbf{y}) + (1-\lambda) f(\mathbf{x}^*) < f(\mathbf{x}^*)$$

This means points arbitrarily close to $\mathbf{x}^*$ (small $\lambda$) have lower function values, contradicting that $\mathbf{x}^*$ is a local minimum.

### 1.5 DL Loss Functions Are NOT Convex

Neural network loss functions are highly non-convex. Why do we study convexity then?
1. **Local structure**: Near a local minimum, the loss may be approximately convex
2. **Subproblems**: Some DL components are convex (e.g., the softmax cross-entropy in the last layer)
3. **Analysis tools**: Convergence proofs for SGD often assume (local) convexity
4. **Intuition**: Convex optimization provides the mental framework for understanding non-convex optimization

---

## 2. Critical Points and Saddle Points

### 2.1 First-Order Necessary Condition

At a local minimum $\mathbf{x}^*$: $\nabla f(\mathbf{x}^*) = \mathbf{0}$.

Points where $\nabla f = \mathbf{0}$ are called **critical points** (or stationary points).

### 2.2 Second-Order Conditions

At a critical point $\mathbf{x}^*$:
- **Local minimum**: $\mathbf{H}(\mathbf{x}^*) \succ 0$ (positive definite)
- **Local maximum**: $\mathbf{H}(\mathbf{x}^*) \prec 0$ (negative definite)
- **Saddle point**: $\mathbf{H}(\mathbf{x}^*)$ has both positive and negative eigenvalues

### 2.3 The Saddle Point Problem

As discussed in Lesson 04, saddle points dominate in high dimensions. The **index** of a saddle point is the number of negative eigenvalues of $\mathbf{H}$.

**Empirical observation** (Dauphin et al., 2014): In deep networks, critical points with low loss tend to have few negative eigenvalues (low index), while critical points with high loss tend to have many. This suggests that as optimization progresses, saddle points become less problematic.

```python
# Visualize different critical points
from mpl_toolkits.mplot3d import Axes3D

x = np.linspace(-2, 2, 200)
y = np.linspace(-2, 2, 200)
X, Y = np.meshgrid(x, y)

fig, axes = plt.subplots(1, 3, figsize=(15, 4), subplot_kw={'projection': '3d'})

# Local minimum
Z1 = X**2 + Y**2
axes[0].plot_surface(X, Y, Z1, cmap='Blues', alpha=0.8)
axes[0].set_title('Minimum: $x^2 + y^2$')

# Saddle point
Z2 = X**2 - Y**2
axes[1].plot_surface(X, Y, Z2, cmap='RdBu', alpha=0.8)
axes[1].set_title('Saddle: $x^2 - y^2$')

# Local max
Z3 = -(X**2 + Y**2)
axes[2].plot_surface(X, Y, Z3, cmap='Reds', alpha=0.8)
axes[2].set_title('Maximum: $-(x^2 + y^2)$')

plt.tight_layout()
plt.show()
```

---

## 3. Gradient Descent Convergence

### 3.1 Smoothness Condition

$f$ is **$L$-smooth** if its gradient is $L$-Lipschitz continuous:

$$\|\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})\| \leq L \|\mathbf{x} - \mathbf{y}\| \quad \forall \mathbf{x}, \mathbf{y}$$

Equivalently, $\nabla^2 f(\mathbf{x}) \preceq L \mathbf{I}$ for all $\mathbf{x}$ (all eigenvalues of Hessian are at most $L$).

### 3.2 Convergence for Smooth Convex Functions

**Theorem**: For $f$ convex and $L$-smooth, gradient descent with $\eta = 1/L$ satisfies:

$$f(\mathbf{x}^{(T)}) - f(\mathbf{x}^*) \leq \frac{L \|\mathbf{x}^{(0)} - \mathbf{x}^*\|^2}{2T}$$

This is a $O(1/T)$ convergence rate -- sublinear.

### 3.3 Convergence for Strongly Convex Functions

**Theorem**: For $f$ $\mu$-strongly convex and $L$-smooth, gradient descent with $\eta = 2/(\mu + L)$ satisfies:

$$\|\mathbf{x}^{(T)} - \mathbf{x}^*\|^2 \leq \left(\frac{\kappa - 1}{\kappa + 1}\right)^{2T} \|\mathbf{x}^{(0)} - \mathbf{x}^*\|^2$$

where $\kappa = L/\mu$ is the condition number. This is **linear convergence** (exponential decrease in error), but the rate degrades with $\kappa$.

```python
# Demonstrate convergence rates on quadratics with different condition numbers
np.random.seed(42)

def gradient_descent_quadratic(A, b, x0, lr, n_steps):
    x = x0.copy()
    x_star = np.linalg.solve(A, b)
    errors = []
    for _ in range(n_steps):
        errors.append(np.linalg.norm(x - x_star))
        grad = A @ x - b
        x = x - lr * grad
    return errors

n = 10
n_steps = 200

fig, ax = plt.subplots(figsize=(8, 5))
for kappa in [2, 10, 100]:
    # Create matrix with prescribed condition number
    eigvals = np.linspace(1, kappa, n)
    Q, _ = np.linalg.qr(np.random.randn(n, n))
    A = Q @ np.diag(eigvals) @ Q.T
    b = np.random.randn(n)
    x0 = np.zeros(n)
    lr = 2 / (eigvals[0] + eigvals[-1])

    errors = gradient_descent_quadratic(A, b, x0, lr, n_steps)
    ax.semilogy(errors, label=f'κ = {kappa}')

ax.set_xlabel('Iteration')
ax.set_ylabel('$\|x - x^*\|$')
ax.set_title('GD convergence vs. condition number')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

---

## 4. Stochastic Gradient Descent (SGD)

### 4.1 The Stochastic Gradient

In DL, the loss is an expectation over data:

$$L(\boldsymbol{\theta}) = \mathbb{E}_{(\mathbf{x}, y) \sim \mathcal{D}} [\ell(\boldsymbol{\theta}; \mathbf{x}, y)]$$

Computing the full gradient requires the entire dataset. SGD uses a **mini-batch** estimate:

$$\hat{\nabla} L(\boldsymbol{\theta}) = \frac{1}{B} \sum_{i=1}^{B} \nabla \ell(\boldsymbol{\theta}; \mathbf{x}_i, y_i)$$

This is an **unbiased** estimator: $\mathbb{E}[\hat{\nabla} L] = \nabla L$.

### 4.2 SGD Convergence

For convex functions with bounded gradient variance $\sigma^2$:

$$\mathbb{E}[f(\bar{\mathbf{x}}^{(T)})] - f(\mathbf{x}^*) = O\left(\frac{\sigma}{\sqrt{T}}\right) \quad \text{with } \eta_t = \frac{c}{\sqrt{t}}$$

Key insight: **the noise in SGD prevents exact convergence** with a fixed learning rate. To converge, we need decreasing learning rates satisfying:

$$\sum_{t=1}^{\infty} \eta_t = \infty \quad \text{and} \quad \sum_{t=1}^{\infty} \eta_t^2 < \infty$$

### 4.3 Mini-Batch Size and Variance

The variance of the mini-batch gradient estimate scales as:

$$\text{Var}(\hat{\nabla} L) = \frac{\sigma^2}{B}$$

Larger batch size $B$ reduces variance but increases computation per step. The **critical batch size** is the point where increasing $B$ no longer improves wall-clock time.

---

## 5. Momentum Methods

### 5.1 Classical Momentum (Polyak)

Momentum adds a "velocity" term that accumulates past gradients:

$$\mathbf{v}^{(t+1)} = \beta \mathbf{v}^{(t)} + \nabla f(\mathbf{x}^{(t)})$$
$$\mathbf{x}^{(t+1)} = \mathbf{x}^{(t)} - \eta \mathbf{v}^{(t+1)}$$

**Physical analogy**: A heavy ball rolling down the loss surface with friction coefficient $1 - \beta$.

**Effect**: Momentum accelerates convergence along consistent gradient directions and dampens oscillations across steep directions.

### 5.2 Nesterov Accelerated Gradient (NAG)

NAG evaluates the gradient at a "look-ahead" position:

$$\mathbf{v}^{(t+1)} = \beta \mathbf{v}^{(t)} + \nabla f(\mathbf{x}^{(t)} - \eta \beta \mathbf{v}^{(t)})$$
$$\mathbf{x}^{(t+1)} = \mathbf{x}^{(t)} - \eta \mathbf{v}^{(t+1)}$$

NAG achieves the **optimal convergence rate** for first-order methods on smooth convex functions:

$$f(\mathbf{x}^{(T)}) - f(\mathbf{x}^*) = O\left(\frac{1}{T^2}\right) \quad \text{vs. } O\left(\frac{1}{T}\right) \text{ for vanilla GD}$$

### 5.3 Momentum Visualization

```python
def optimize_2d(optimizer_fn, f, grad_f, x0, n_steps):
    """Run an optimizer and return the trajectory."""
    x = np.array(x0, dtype=float)
    state = {}
    trajectory = [x.copy()]
    for t in range(n_steps):
        g = grad_f(x)
        x, state = optimizer_fn(x, g, t, state)
        trajectory.append(x.copy())
    return np.array(trajectory)

# Rosenbrock-like function
def f_rosen(x):
    return (1 - x[0])**2 + 10*(x[1] - x[0]**2)**2

def grad_rosen(x):
    dx = -2*(1 - x[0]) - 40*x[0]*(x[1] - x[0]**2)
    dy = 20*(x[1] - x[0]**2)
    return np.array([dx, dy])

# SGD
def sgd(x, g, t, state, lr=0.001):
    return x - lr * g, state

# SGD + Momentum
def sgd_momentum(x, g, t, state, lr=0.001, beta=0.9):
    v = state.get('v', np.zeros_like(x))
    v = beta * v + g
    state['v'] = v
    return x - lr * v, state

x0 = [-1.0, 1.0]
n_steps = 500

traj_sgd = optimize_2d(sgd, f_rosen, grad_rosen, x0, n_steps)
traj_mom = optimize_2d(sgd_momentum, f_rosen, grad_rosen, x0, n_steps)

# Plot
x1 = np.linspace(-2, 2, 200)
x2 = np.linspace(-1, 3, 200)
X1, X2 = np.meshgrid(x1, x2)
Z = (1 - X1)**2 + 10*(X2 - X1**2)**2

fig, ax = plt.subplots(figsize=(8, 6))
ax.contour(X1, X2, Z, levels=np.logspace(-1, 3, 20), alpha=0.5)
ax.plot(traj_sgd[:, 0], traj_sgd[:, 1], 'r.-', markersize=1, linewidth=0.5, label='SGD')
ax.plot(traj_mom[:, 0], traj_mom[:, 1], 'b.-', markersize=1, linewidth=0.5, label='Momentum')
ax.plot(1, 1, 'g*', markersize=15, label='Optimum')
ax.legend()
ax.set_title('SGD vs. Momentum on Rosenbrock')
plt.show()
```

---

## 6. Adaptive Learning Rates

### 6.1 AdaGrad

Accumulates squared gradients to scale learning rates per parameter:

$$G_{ii}^{(t)} = \sum_{\tau=0}^{t} g_{i,\tau}^2$$
$$\theta_i^{(t+1)} = \theta_i^{(t)} - \frac{\eta}{\sqrt{G_{ii}^{(t)}} + \epsilon} g_{i,t}$$

**Problem**: $G_{ii}$ grows monotonically, causing the learning rate to decay to zero.

### 6.2 Adam

Combines momentum with adaptive learning rates:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{(first moment estimate)}$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{(second moment estimate)}$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \quad \text{(bias correction)}$$
$$\theta_t = \theta_{t-1} - \frac{\eta \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Default hyperparameters: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.

### 6.3 Why Bias Correction Matters

At $t = 1$: $m_1 = (1 - \beta_1) g_1$, which is biased toward zero. The correction $\hat{m}_1 = m_1 / (1 - \beta_1) = g_1$ removes this bias. Without correction, early updates are too small.

```python
# Adam implementation
def adam(x, g, t, state, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
    m = state.get('m', np.zeros_like(x))
    v = state.get('v', np.zeros_like(x))

    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g**2

    m_hat = m / (1 - beta1**(t + 1))
    v_hat = v / (1 - beta2**(t + 1))

    x_new = x - lr * m_hat / (np.sqrt(v_hat) + eps)
    state['m'] = m
    state['v'] = v
    return x_new, state

traj_adam = optimize_2d(adam, f_rosen, grad_rosen, x0, n_steps)

fig, ax = plt.subplots(figsize=(8, 6))
ax.contour(X1, X2, Z, levels=np.logspace(-1, 3, 20), alpha=0.5)
ax.plot(traj_sgd[:, 0], traj_sgd[:, 1], 'r.-', markersize=1, linewidth=0.5, label='SGD')
ax.plot(traj_mom[:, 0], traj_mom[:, 1], 'b.-', markersize=1, linewidth=0.5, label='Momentum')
ax.plot(traj_adam[:, 0], traj_adam[:, 1], 'g.-', markersize=1, linewidth=0.5, label='Adam')
ax.plot(1, 1, 'k*', markersize=15)
ax.legend()
ax.set_title('Optimizer comparison on Rosenbrock')
plt.show()
```

---

## 7. Learning Rate Schedules

### 7.1 Common Schedules

| Schedule | Formula | Use Case |
|----------|---------|----------|
| Step decay | $\eta_t = \eta_0 \cdot \gamma^{\lfloor t/s \rfloor}$ | Classic CV training |
| Cosine annealing | $\eta_t = \frac{\eta_0}{2}(1 + \cos(\pi t / T))$ | Modern DL training |
| Linear warmup | $\eta_t = \eta_0 \cdot \min(1, t / t_w)$ | Transformer training |
| 1/sqrt decay | $\eta_t = \eta_0 / \sqrt{t}$ | Theoretically motivated |

### 7.2 Warmup

Large learning rates at the start of training can cause divergence because:
1. The initial parameters are far from any basin of attraction
2. The gradient variance is high on random parameters
3. Adaptive optimizers have noisy moment estimates early on

**Warmup** gradually increases the learning rate from a small value, allowing the optimizer to stabilize.

```python
# Learning rate schedules
T = 1000
t = np.arange(T)
eta0 = 0.01

schedules = {
    'Constant': np.full(T, eta0),
    'Step decay': eta0 * 0.1 ** (t // 300),
    'Cosine': eta0 / 2 * (1 + np.cos(np.pi * t / T)),
    'Warmup + Cosine': np.where(t < 100, eta0 * t / 100,
                                 eta0 / 2 * (1 + np.cos(np.pi * (t - 100) / (T - 100)))),
}

fig, ax = plt.subplots(figsize=(10, 4))
for name, lr in schedules.items():
    ax.plot(t, lr, label=name, linewidth=2)
ax.set_xlabel('Step')
ax.set_ylabel('Learning rate')
ax.set_title('Learning rate schedules')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| Convexity | Every local minimum is global; DL losses are NOT convex |
| Smoothness | $L$-smooth means bounded Hessian; sets max learning rate $\eta \leq 1/L$ |
| GD convergence | $O(1/T)$ for convex, $O((\frac{\kappa-1}{\kappa+1})^T)$ for strongly convex |
| SGD | Unbiased gradient estimate; needs decreasing $\eta$ for convergence |
| Momentum | Accelerates along consistent directions; $O(1/T^2)$ for Nesterov |
| Adam | Adaptive per-parameter learning rates from second moment estimates |
| Schedules | Warmup + cosine annealing is the modern default |

---

## Exercises

1. Prove that $f(x) = \log(1 + e^x)$ (softplus) is convex by showing its second derivative is non-negative.
2. Implement gradient descent, momentum, and Adam on the Styblinski-Tang function and compare convergence.
3. Derive the optimal learning rate for gradient descent on $f(x) = \frac{1}{2}x^\top A x - b^\top x$.
4. Implement cosine annealing with warm restarts (SGDR) and plot the learning rate schedule.
5. Empirically verify that SGD with constant learning rate oscillates around the minimum by running it on a noisy quadratic.

---

**Next**: [06. Probability Distributions for DL](06_Probability_Distributions_for_DL.md)
