# 07. Gradient Descent Theory

## Learning Objectives

- Understand the basic principle and update rule of gradient descent and implement it
- Theoretically analyze convergence rates for convex and strongly convex functions
- Learn the principle of stochastic gradient descent (SGD) and the role of mini-batches
- Understand the operating principle of momentum and Nesterov accelerated gradient with physical intuition
- Learn the derivation process of adaptive learning rate methods like Adam and RMSProp
- Understand and apply practical considerations in neural network optimization

---

## 1. Gradient Descent Basics

### 1.1 Basic Principle

Gradient descent (GD) is a first-order optimization algorithm that iteratively moves in the opposite direction of the gradient to minimize a function.

**Update rule:**

$$
x_{t+1} = x_t - \eta \nabla f(x_t)
$$

- $x_t$: parameter at time $t$
- $\eta$: learning rate (step size)
- $\nabla f(x_t)$: gradient at $x_t$

**Intuition:**
- Gradient $\nabla f(x)$ is the direction of fastest increase
- Negative gradient $-\nabla f(x)$ is the direction of fastest decrease (steepest descent)
- Learning rate $\eta$ controls the step size

### 1.2 First-Order Taylor Approximation

Gradient descent is based on first-order Taylor approximation:

$$
f(x + \Delta x) \approx f(x) + \nabla f(x)^T \Delta x
$$

Choosing $\Delta x = -\eta \nabla f(x)$:

$$
f(x - \eta \nabla f(x)) \approx f(x) - \eta \|\nabla f(x)\|^2
$$

If $\eta$ is sufficiently small, the function value decreases.

### 1.3 Implementation and Visualization

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Objective function: f(x,y) = (x-1)^2 + 2(y-2)^2
def objective(x, y):
    return (x - 1)**2 + 2*(y - 2)**2

def gradient(x, y):
    df_dx = 2*(x - 1)
    df_dy = 4*(y - 2)
    return np.array([df_dx, df_dy])

# Gradient descent
def gradient_descent(x0, learning_rate, n_iterations):
    """Basic gradient descent"""
    trajectory = [x0]
    x = x0.copy()

    for _ in range(n_iterations):
        grad = gradient(x[0], x[1])
        x = x - learning_rate * grad
        trajectory.append(x.copy())

    return np.array(trajectory)

# Initial point
x0 = np.array([-2.0, -1.0])

# Experiment with multiple learning rates
learning_rates = [0.1, 0.3, 0.5, 0.9]
n_iterations = 50

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

# Contour grid
x_vals = np.linspace(-3, 3, 300)
y_vals = np.linspace(-2, 4, 300)
X, Y = np.meshgrid(x_vals, y_vals)
Z = objective(X, Y)

for idx, lr in enumerate(learning_rates):
    ax = axes[idx]

    # Contour
    contour = ax.contour(X, Y, Z, levels=20, alpha=0.6, cmap='viridis')
    ax.clabel(contour, inline=True, fontsize=8)

    # Gradient descent trajectory
    trajectory = gradient_descent(x0, lr, n_iterations)
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'ro-', linewidth=2,
            markersize=6, label='GD trajectory')
    ax.plot(x0[0], x0[1], 'g*', markersize=20, label='Start point')
    ax.plot(1, 2, 'r*', markersize=20, label='Minimum')

    # Show gradient vectors (first few)
    for i in range(0, min(5, len(trajectory)-1), 1):
        x_curr = trajectory[i]
        grad = gradient(x_curr[0], x_curr[1])
        ax.quiver(x_curr[0], x_curr[1], -grad[0], -grad[1],
                  angles='xy', scale_units='xy', scale=5,
                  color='blue', width=0.005, alpha=0.7)

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(f'Learning rate η = {lr} ({len(trajectory)-1} iterations)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-2, 4)

    # Convergence info
    final_x = trajectory[-1]
    final_loss = objective(final_x[0], final_x[1])
    distance_to_optimum = np.linalg.norm(final_x - np.array([1, 2]))
    ax.text(0.05, 0.95, f'Final loss: {final_loss:.4f}\nDistance to minimum: {distance_to_optimum:.4f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('gradient_descent_learning_rates.png', dpi=150)
plt.show()

print("Effect of learning rate:")
print("  - η too small: slow convergence")
print("  - η appropriate: fast convergence")
print("  - η too large: divergence or oscillation")
```

### 1.4 Choosing the Learning Rate

**Learning rate too small:**
- Very slow convergence
- Many iterations required

**Learning rate too large:**
- Oscillation around minimum
- Possible divergence

**Appropriate learning rate:**
- Theoretical upper bound: $\eta \leq \frac{1}{L}$ (L: Lipschitz constant)
- Practice: grid search or learning rate schedule

```python
# Convergence curves for different learning rates
fig, ax = plt.subplots(figsize=(12, 6))

for lr in learning_rates:
    trajectory = gradient_descent(x0, lr, n_iterations)
    losses = [objective(x[0], x[1]) for x in trajectory]
    ax.plot(losses, linewidth=2, label=f'η = {lr}')

ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Loss Reduction by Learning Rate', fontsize=14)
ax.set_yscale('log')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('learning_rate_convergence.png', dpi=150)
plt.show()
```

## 2. Convergence Analysis

### 2.1 Lipschitz Continuous Gradient

A function $f$ has Lipschitz continuous gradient if there exists a constant $L > 0$ satisfying:

$$
\|\nabla f(x) - \nabla f(y)\| \leq L\|x - y\|, \quad \forall x, y
$$

This is equivalent to $\nabla^2 f(x) \preceq LI$ (Hessian bounded by $LI$).

### 2.2 Convergence for Convex Functions

**Theorem (Convex Function):**
If $f$ is convex and the gradient is $L$-Lipschitz continuous, with learning rate $\eta = \frac{1}{L}$:

$$
f(x_t) - f(x^*) \leq \frac{L\|x_0 - x^*\|^2}{2t}
$$

That is, **sublinear convergence**: $O(1/t)$

### 2.3 Convergence for Strongly Convex Functions

A function $f$ is $m$-strongly convex if:

$$
f(y) \geq f(x) + \nabla f(x)^T(y-x) + \frac{m}{2}\|y-x\|^2
$$

or equivalently $\nabla^2 f(x) \succeq mI$.

**Theorem (Strongly Convex Function):**
If $f$ is $m$-strongly convex and the gradient is $L$-Lipschitz continuous, with learning rate $\eta = \frac{1}{L}$:

$$
\|x_t - x^*\|^2 \leq \left(1 - \frac{m}{L}\right)^t \|x_0 - x^*\|^2
$$

That is, **linear convergence**: $O(\rho^t)$, where $\rho = 1 - \frac{m}{L} < 1$

**Condition Number:**
$$\kappa = \frac{L}{m}$$

The larger the condition number (ill-conditioned), the slower the convergence.

### 2.4 Convergence Rate Simulation

```python
import numpy as np
import matplotlib.pyplot as plt

# Quadratic form: f(x) = 0.5 * x^T A x
# Strongly convex: eigenvalues(A) > 0

def create_quadratic(m, L, dim=10):
    """Create strongly convex quadratic function (condition number κ = L/m)"""
    # Uniformly distributed eigenvalues between m and L
    eigenvalues = np.linspace(m, L, dim)
    # Random orthogonal matrix
    Q, _ = np.linalg.qr(np.random.randn(dim, dim))
    # A = Q Λ Q^T
    A = Q @ np.diag(eigenvalues) @ Q.T
    return A

def quadratic_objective(x, A):
    return 0.5 * x @ A @ x

def quadratic_gradient(x, A):
    return A @ x

def gd_quadratic(A, x0, learning_rate, n_iterations):
    """Gradient descent on quadratic function"""
    x = x0.copy()
    trajectory = [np.linalg.norm(x)**2]  # ||x - x*||^2, x* = 0

    for _ in range(n_iterations):
        grad = quadratic_gradient(x, A)
        x = x - learning_rate * grad
        trajectory.append(np.linalg.norm(x)**2)

    return trajectory

# Experiment setup
dim = 10
n_iterations = 100

# Experiment with multiple condition numbers
condition_numbers = [1, 10, 100, 1000]
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for kappa in condition_numbers:
    m = 1.0
    L = kappa * m
    A = create_quadratic(m, L, dim)

    # Initial point
    x0 = np.random.randn(dim)

    # Gradient descent
    learning_rate = 1 / L
    trajectory = gd_quadratic(A, x0, learning_rate, n_iterations)

    # Linear scale
    axes[0].plot(trajectory, linewidth=2, label=f'κ = {kappa}')

    # Log scale
    axes[1].semilogy(trajectory, linewidth=2, label=f'κ = {kappa}')

    # Theoretical convergence rate
    rho = 1 - m/L
    theoretical = [trajectory[0] * (rho ** t) for t in range(n_iterations + 1)]
    axes[1].semilogy(theoretical, '--', linewidth=1, alpha=0.6,
                     label=f'Theory (κ={kappa})')

axes[0].set_xlabel('Iteration', fontsize=12)
axes[0].set_ylabel('$\|x_t - x^*\|^2$', fontsize=12)
axes[0].set_title('Convergence Curve (Linear Scale)', fontsize=14)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('Iteration', fontsize=12)
axes[1].set_ylabel('$\|x_t - x^*\|^2$ (log scale)', fontsize=12)
axes[1].set_title('Convergence Curve (Log Scale): Linear Convergence', fontsize=14)
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('convergence_analysis_condition_number.png', dpi=150)
plt.show()

print("Effect of condition number:")
print("  κ = 1: perfectly conditioned (equal curvature in all directions)")
print("  κ >> 1: ill-conditioned (some directions very flat)")
print("  Convergence rate: O((1 - 1/κ)^t)")
```

## 3. Stochastic Gradient Descent (SGD)

### 3.1 Batch vs Mini-batch

**Batch Gradient Descent:**
Compute gradient using entire dataset:
$$x_{t+1} = x_t - \eta \nabla f(x_t) = x_t - \eta \frac{1}{n}\sum_{i=1}^n \nabla f_i(x_t)$$

**Stochastic Gradient Descent (SGD):**
Estimate gradient using one randomly selected sample:
$$x_{t+1} = x_t - \eta \nabla f_{i_t}(x_t)$$

**Mini-batch SGD:**
Estimate gradient using mini-batch of $B$ samples:
$$x_{t+1} = x_t - \eta \frac{1}{|B|}\sum_{i \in B} \nabla f_i(x_t)$$

### 3.2 Advantages and Disadvantages of SGD

**Advantages:**
- **Computational efficiency**: No need for entire dataset per iteration
- **Memory efficiency**: Suitable for large datasets
- **Regularization effect**: Noise helps escape sharp minima
- **Online learning**: Works even with streaming data

**Disadvantages:**
- **Noise**: Unstable gradient estimation
- **Learning rate tuning**: More sensitive than batch GD
- **Convergence speed**: Theoretically slower (but faster in practice)

### 3.3 Effect of Mini-batch Size

```python
import numpy as np
import matplotlib.pyplot as plt

# Synthetic data: linear regression
np.random.seed(42)
n_samples = 1000
n_features = 20

X = np.random.randn(n_samples, n_features)
true_w = np.random.randn(n_features)
y = X @ true_w + 0.1 * np.random.randn(n_samples)

def mse_loss(w, X, y):
    """MSE loss"""
    return 0.5 * np.mean((X @ w - y) ** 2)

def mse_gradient(w, X, y):
    """MSE gradient"""
    return X.T @ (X @ w - y) / len(y)

def sgd_minibatch(X, y, batch_size, learning_rate, n_epochs):
    """Mini-batch SGD"""
    n_samples = len(X)
    w = np.zeros(X.shape[1])
    losses = []

    for epoch in range(n_epochs):
        # Shuffle data
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        # Split into mini-batches
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            # Compute gradient and update
            grad = mse_gradient(w, X_batch, y_batch)
            w = w - learning_rate * grad

        # Record full loss per epoch
        loss = mse_loss(w, X, y)
        losses.append(loss)

    return w, losses

# Experiment with multiple batch sizes
batch_sizes = [1, 10, 50, 200, 1000]
n_epochs = 50
learning_rate = 0.01

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for batch_size in batch_sizes:
    w_final, losses = sgd_minibatch(X, y, batch_size, learning_rate, n_epochs)

    # Loss curves
    axes[0].plot(losses, linewidth=2, label=f'Batch size = {batch_size}')
    axes[1].semilogy(losses, linewidth=2, label=f'Batch size = {batch_size}')

axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('MSE Loss', fontsize=12)
axes[0].set_title('Convergence by Batch Size', fontsize=14)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('MSE Loss (log scale)', fontsize=12)
axes[1].set_title('Convergence by Batch Size (Log Scale)', fontsize=14)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sgd_batch_size_effect.png', dpi=150)
plt.show()

print("Effect of batch size:")
print("  Small batch: noisy and unstable, regularization effect, memory efficient")
print("  Large batch: stable gradient, faster convergence, parallelizable computation")
print("  In practice: powers of 2 like 32, 64, 128, 256 (GPU optimized)")
```

### 3.4 SGD Variance and Learning Rate

The SGD gradient has the same expectation as the true gradient, but with variance:

$$
\mathbb{E}[\nabla f_i(x)] = \nabla f(x), \quad \text{Var}[\nabla f_i(x)] = \sigma^2
$$

High variance slows convergence and causes instability. Solutions:
- **Learning rate decay**: $\eta_t = \frac{\eta_0}{\sqrt{t}}$
- **Increase mini-batch**: variance $\propto 1/|B|$
- **Adaptive methods**: Adam, RMSProp

## 4. Momentum-based Methods

### 4.1 Momentum SGD

Basic momentum (Heavy Ball):

$$
\begin{align}
v_t &= \beta v_{t-1} + \nabla f(x_t) \\
x_{t+1} &= x_t - \eta v_t
\end{align}
$$

where $\beta \in [0, 1)$ is the momentum coefficient (typically 0.9).

**Physical intuition:**
- Like a ball rolling down a hill, maintains inertia from previous direction
- Accelerates in consistent directions, dampens oscillations
- Helps escape local minima and saddle points

### 4.2 Nesterov Accelerated Gradient (NAG)

**Nesterov Accelerated Gradient:**

$$
\begin{align}
v_t &= \beta v_{t-1} + \nabla f(x_t - \beta v_{t-1}) \\
x_{t+1} &= x_t - \eta v_t
\end{align}
$$

**Key idea:**
- "Look ahead" first ($x_t - \beta v_{t-1}$) and compute gradient there
- Smarter look-ahead than momentum
- Theoretically better convergence rate

```python
import numpy as np
import matplotlib.pyplot as plt

# Rosenbrock function
def rosenbrock(x, y):
    return (1 - x)**2 + 100*(y - x**2)**2

def rosenbrock_gradient(x, y):
    df_dx = -2*(1 - x) - 400*x*(y - x**2)
    df_dy = 200*(y - x**2)
    return np.array([df_dx, df_dy])

# Basic GD
def gd(x0, learning_rate, n_iterations):
    trajectory = [x0]
    x = x0.copy()

    for _ in range(n_iterations):
        grad = rosenbrock_gradient(x[0], x[1])
        x = x - learning_rate * grad
        trajectory.append(x.copy())

    return np.array(trajectory)

# Momentum GD
def momentum_gd(x0, learning_rate, beta, n_iterations):
    trajectory = [x0]
    x = x0.copy()
    v = np.zeros_like(x)

    for _ in range(n_iterations):
        grad = rosenbrock_gradient(x[0], x[1])
        v = beta * v + grad
        x = x - learning_rate * v
        trajectory.append(x.copy())

    return np.array(trajectory)

# Nesterov GD
def nesterov_gd(x0, learning_rate, beta, n_iterations):
    trajectory = [x0]
    x = x0.copy()
    v = np.zeros_like(x)

    for _ in range(n_iterations):
        # Look-ahead
        x_lookahead = x - beta * v
        grad = rosenbrock_gradient(x_lookahead[0], x_lookahead[1])
        v = beta * v + grad
        x = x - learning_rate * v
        trajectory.append(x.copy())

    return np.array(trajectory)

# Visualization
x0 = np.array([-1.0, 0.5])
learning_rate = 0.001
beta = 0.9
n_iterations = 200

trajectories = {
    'GD': gd(x0, learning_rate, n_iterations),
    'Momentum': momentum_gd(x0, learning_rate, beta, n_iterations),
    'Nesterov': nesterov_gd(x0, learning_rate, beta, n_iterations)
}

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Contour
x_vals = np.linspace(-1.5, 1.5, 300)
y_vals = np.linspace(-0.5, 1.5, 300)
X, Y = np.meshgrid(x_vals, y_vals)
Z = rosenbrock(X, Y)

# Left: trajectory comparison
ax = axes[0]
levels = np.logspace(-1, 3, 20)
contour = ax.contour(X, Y, Z, levels=levels, alpha=0.4, cmap='gray')

colors = {'GD': 'blue', 'Momentum': 'red', 'Nesterov': 'green'}
for name, traj in trajectories.items():
    ax.plot(traj[:, 0], traj[:, 1], '-', linewidth=2, color=colors[name],
            label=name, alpha=0.7)
    ax.plot(traj[0, 0], traj[0, 1], 'o', markersize=10, color=colors[name])

ax.plot(1, 1, 'k*', markersize=20, label='Minimum (1, 1)')
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('Rosenbrock Function: Momentum vs Nesterov', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Right: loss curves
ax = axes[1]
for name, traj in trajectories.items():
    losses = [rosenbrock(x[0], x[1]) for x in traj]
    ax.semilogy(losses, linewidth=2, color=colors[name], label=name)

ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Loss (log scale)', fontsize=12)
ax.set_title('Loss Reduction Comparison', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('momentum_vs_nesterov.png', dpi=150)
plt.show()

print("Effect of momentum:")
print("  - Accelerates in consistent direction")
print("  - Dampens oscillations (especially for ill-conditioned problems)")
print("  - Passes through saddle points faster")
print("\nAdvantages of Nesterov:")
print("  - Smarter updates via look-ahead")
print("  - Theoretically better convergence rate")
```

## 5. Adaptive Learning Rate Methods

### 5.1 AdaGrad

**Adaptive Gradient Algorithm:**

$$
\begin{align}
G_t &= G_{t-1} + \nabla f(x_t) \odot \nabla f(x_t) \quad \text{(cumulative square)} \\
x_{t+1} &= x_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot \nabla f(x_t)
\end{align}
$$

- $\odot$: element-wise multiplication (Hadamard product)
- $\epsilon$: numerical stability ($10^{-8}$)

**Characteristics:**
- Frequently updated parameters: learning rate decreases
- Rarely updated parameters: learning rate maintained
- **Problem**: $G_t$ keeps growing → learning rate becomes too small

### 5.2 RMSProp

**Root Mean Square Propagation:**

$$
\begin{align}
G_t &= \beta G_{t-1} + (1-\beta) \nabla f(x_t) \odot \nabla f(x_t) \quad \text{(exponential moving average)} \\
x_{t+1} &= x_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot \nabla f(x_t)
\end{align}
$$

**AdaGrad improvement:**
- Exponential moving average (EMA) instead of accumulation
- Decay of old gradient influence
- Solves the too-small learning rate problem

### 5.3 Adam

**Adaptive Moment Estimation:**

$$
\begin{align}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) \nabla f(x_t) \quad \text{(1st moment, mean)} \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) \nabla f(x_t) \odot \nabla f(x_t) \quad \text{(2nd moment, variance)} \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \quad \text{(bias correction)} \\
x_{t+1} &= x_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \odot \hat{m}_t
\end{align}
$$

**Hyperparameters:**
- $\beta_1 = 0.9$ (first moment decay)
- $\beta_2 = 0.999$ (second moment decay)
- $\eta = 0.001$ (learning rate)
- $\epsilon = 10^{-8}$

**Characteristics:**
- Momentum + adaptive learning rate
- Bias correction: removes bias in moment estimates at early stages
- Works well for most deep learning problems

### 5.4 Implementation and Comparison

```python
import numpy as np
import matplotlib.pyplot as plt

# Optimization algorithm implementations
class Optimizer:
    def __init__(self, learning_rate):
        self.lr = learning_rate

class SGD(Optimizer):
    def update(self, x, grad):
        return x - self.lr * grad

class Momentum(Optimizer):
    def __init__(self, learning_rate, beta=0.9):
        super().__init__(learning_rate)
        self.beta = beta
        self.v = None

    def update(self, x, grad):
        if self.v is None:
            self.v = np.zeros_like(x)
        self.v = self.beta * self.v + grad
        return x - self.lr * self.v

class AdaGrad(Optimizer):
    def __init__(self, learning_rate, epsilon=1e-8):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.G = None

    def update(self, x, grad):
        if self.G is None:
            self.G = np.zeros_like(x)
        self.G += grad ** 2
        return x - self.lr * grad / (np.sqrt(self.G) + self.epsilon)

class RMSProp(Optimizer):
    def __init__(self, learning_rate, beta=0.9, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta = beta
        self.epsilon = epsilon
        self.G = None

    def update(self, x, grad):
        if self.G is None:
            self.G = np.zeros_like(x)
        self.G = self.beta * self.G + (1 - self.beta) * grad ** 2
        return x - self.lr * grad / (np.sqrt(self.G) + self.epsilon)

class Adam(Optimizer):
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, x, grad):
        if self.m is None:
            self.m = np.zeros_like(x)
            self.v = np.zeros_like(x)

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2

        # Bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        return x - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

# Test function: Beale function
def beale(x, y):
    return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2

def beale_gradient(x, y):
    df_dx = 2*(1.5 - x + x*y)*(-1 + y) + 2*(2.25 - x + x*y**2)*(-1 + y**2) + \
            2*(2.625 - x + x*y**3)*(-1 + y**3)
    df_dy = 2*(1.5 - x + x*y)*x + 2*(2.25 - x + x*y**2)*(2*x*y) + \
            2*(2.625 - x + x*y**3)*(3*x*y**2)
    return np.array([df_dx, df_dy])

# Run optimization
x0 = np.array([3.0, 3.0])
n_iterations = 500

optimizers = {
    'SGD': SGD(0.001),
    'Momentum': Momentum(0.001, beta=0.9),
    'AdaGrad': AdaGrad(0.5),
    'RMSProp': RMSProp(0.01, beta=0.9),
    'Adam': Adam(0.01, beta1=0.9, beta2=0.999)
}

trajectories = {}
for name, opt in optimizers.items():
    x = x0.copy()
    traj = [x.copy()]

    for _ in range(n_iterations):
        grad = beale_gradient(x[0], x[1])
        x = opt.update(x, grad)
        traj.append(x.copy())

    trajectories[name] = np.array(traj)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Contour
x_vals = np.linspace(-1, 4, 300)
y_vals = np.linspace(-1, 4, 300)
X, Y = np.meshgrid(x_vals, y_vals)
Z = beale(X, Y)

# Left: trajectories
ax = axes[0]
levels = np.logspace(0, 4, 20)
contour = ax.contour(X, Y, Z, levels=levels, alpha=0.3, cmap='gray')

colors = {'SGD': 'blue', 'Momentum': 'red', 'AdaGrad': 'green',
          'RMSProp': 'purple', 'Adam': 'orange'}

for name, traj in trajectories.items():
    ax.plot(traj[:, 0], traj[:, 1], '-', linewidth=2, color=colors[name],
            label=name, alpha=0.7)

ax.plot(3, 0.5, 'k*', markersize=20, label='Minimum')
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('Beale Function: Optimizer Comparison', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(-1, 4)
ax.set_ylim(-1, 4)

# Right: loss curves
ax = axes[1]
for name, traj in trajectories.items():
    losses = [beale(x[0], x[1]) for x in traj]
    ax.semilogy(losses, linewidth=2, color=colors[name], label=name)

ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Loss (log scale)', fontsize=12)
ax.set_title('Loss Reduction Comparison', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('optimizer_comparison.png', dpi=150)
plt.show()

print("Optimizer characteristics:")
print("  SGD: simple, slow")
print("  Momentum: acceleration, oscillation damping")
print("  AdaGrad: good for sparse data, learning rate decay problem")
print("  RMSProp: improved AdaGrad, stable")
print("  Adam: best in most situations, de facto standard")
```

### 5.5 Adam Bias Correction

In early stages, $m_t$ and $v_t$ are biased toward zero due to initialization. Bias correction fixes this.

$$
\mathbb{E}[m_t] = \mathbb{E}\left[\sum_{i=1}^t \beta_1^{t-i}(1-\beta_1)g_i\right] = \mathbb{E}[g_t](1 - \beta_1^t)
$$

Therefore, $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$ is an unbiased estimator.

```python
# Visualize bias correction effect
fig, ax = plt.subplots(figsize=(12, 6))

beta1, beta2 = 0.9, 0.999
t_vals = np.arange(1, 101)

correction1 = 1 / (1 - beta1 ** t_vals)
correction2 = 1 / (1 - beta2 ** t_vals)

ax.plot(t_vals, correction1, linewidth=2, label=f'1st moment correction (β₁={beta1})')
ax.plot(t_vals, correction2, linewidth=2, label=f'2nd moment correction (β₂={beta2})')
ax.axhline(y=1, color='black', linestyle='--', linewidth=1, label='No correction')

ax.set_xlabel('Iteration t', fontsize=12)
ax.set_ylabel('Correction factor', fontsize=12)
ax.set_title('Adam Bias Correction Factors', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('adam_bias_correction.png', dpi=150)
plt.show()

print("Importance of bias correction:")
print("  - Moments are biased toward 0 in early steps")
print("  - Without correction, initial learning rate is excessively small")
print("  - Correction effect disappears after tens of iterations")
```

## 6. Learning Rate Schedules

### 6.1 Major Scheduling Strategies

**Step Decay:**
$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t/k \rfloor}$$

**Exponential Decay:**
$$\eta_t = \eta_0 \cdot e^{-\lambda t}$$

**Cosine Annealing:**
$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)$$

**1-Cycle Policy:**
- Early: learning rate increase (warm-up)
- Middle: maintain maximum learning rate
- Late: learning rate decrease (annealing)

```python
import numpy as np
import matplotlib.pyplot as plt

def step_decay(t, eta0=0.1, gamma=0.5, k=50):
    return eta0 * (gamma ** (t // k))

def exponential_decay(t, eta0=0.1, lam=0.01):
    return eta0 * np.exp(-lam * t)

def cosine_annealing(t, eta_min=0.0, eta_max=0.1, T=100):
    return eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * t / T))

def one_cycle(t, eta_min=0.001, eta_max=0.1, T=100, warmup_frac=0.3):
    if t < warmup_frac * T:
        # Warm-up
        return eta_min + (eta_max - eta_min) * t / (warmup_frac * T)
    else:
        # Annealing
        progress = (t - warmup_frac * T) / ((1 - warmup_frac) * T)
        return eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * progress))

# Visualization
t_vals = np.arange(0, 200)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

schedules = [
    ('Step Decay', step_decay, axes[0, 0]),
    ('Exponential Decay', exponential_decay, axes[0, 1]),
    ('Cosine Annealing', cosine_annealing, axes[1, 0]),
    ('1-Cycle Policy', one_cycle, axes[1, 1])
]

for name, func, ax in schedules:
    if name == 'Cosine Annealing':
        lr_vals = [func(t, T=200) for t in t_vals]
    elif name == '1-Cycle Policy':
        lr_vals = [func(t, T=200) for t in t_vals]
    else:
        lr_vals = [func(t) for t in t_vals]

    ax.plot(t_vals, lr_vals, linewidth=2, color='blue')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title(name, fontsize=14)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('learning_rate_schedules.png', dpi=150)
plt.show()

print("Role of learning rate schedules:")
print("  - Early: explore good regions quickly with large learning rate")
print("  - Late: fine-tune with small learning rate")
print("  - Warm-up: prevent early instability")
print("  - Cosine/1-Cycle: smooth decay, popular for Transformers etc.")
```

## 7. Practical Considerations in Neural Network Optimization

### 7.1 Geometry of Loss Landscapes

Neural network loss functions are:
- **Non-convex**: multiple local minima
- **High-dimensional**: millions to billions of parameters
- **Saddle Points**: minimum in some directions, maximum in others
- **Plateaus**: gradients nearly zero

### 7.2 Sharp Minima vs Flat Minima

**Sharp Minima:**
- Narrow, sharp minimum
- Poor generalization on test data
- Often occurs with large batch sizes

**Flat Minima:**
- Wide, flat minimum
- Good generalization on test data
- Preferred with small batch sizes + SGD noise

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Loss landscape simulation
def sharp_minimum(x, y):
    """Sharp minimum"""
    return x**2 + y**2 + 0.01 * np.random.randn()

def flat_minimum(x, y):
    """Flat minimum"""
    return 0.1 * x**2 + 0.1 * y**2 + 0.01 * np.random.randn()

fig = plt.figure(figsize=(16, 6))

x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)

# Sharp minimum
ax1 = fig.add_subplot(121, projection='3d')
Z_sharp = X**2 + Y**2
ax1.plot_surface(X, Y, Z_sharp, cmap='Reds', alpha=0.8, edgecolor='none')
ax1.set_title('Sharp Minimum (Poor generalization)', fontsize=14)
ax1.set_xlabel('w₁')
ax1.set_ylabel('w₂')
ax1.set_zlabel('Loss')

# Flat minimum
ax2 = fig.add_subplot(122, projection='3d')
Z_flat = 0.1 * X**2 + 0.1 * Y**2
ax2.plot_surface(X, Y, Z_flat, cmap='Blues', alpha=0.8, edgecolor='none')
ax2.set_title('Flat Minimum (Good generalization)', fontsize=14)
ax2.set_xlabel('w₁')
ax2.set_ylabel('w₂')
ax2.set_zlabel('Loss')

plt.tight_layout()
plt.savefig('sharp_vs_flat_minima.png', dpi=150)
plt.show()

print("Sharp vs Flat Minima:")
print("  Sharp: loss changes greatly with small parameter changes → overfitting")
print("  Flat: loss insensitive to parameter changes → generalization")
print("  SGD noise acts as implicit regularization favoring flat minima")
```

### 7.3 Gradient Clipping

Prevents gradient explosion in RNN/Transformer:

**Norm-based Clipping:**
$$
\tilde{g} = \begin{cases}
g & \text{if } \|g\| \leq \theta \\
\theta \frac{g}{\|g\|} & \text{otherwise}
\end{cases}
$$

**Value-based Clipping:**
$$
\tilde{g}_i = \max(-\theta, \min(\theta, g_i))
$$

```python
import torch

def gradient_clipping_demo():
    # Simple RNN simulation
    torch.manual_seed(42)
    hidden_size = 10
    W = torch.randn(hidden_size, hidden_size, requires_grad=True) * 2  # large weights

    # Forward pass (multiple time steps)
    h = torch.randn(hidden_size)
    for _ in range(20):
        h = torch.tanh(W @ h)

    loss = h.sum()
    loss.backward()

    grad_norm = W.grad.norm().item()
    print(f"Gradient norm before clipping: {grad_norm:.4f}")

    # Gradient clipping
    max_norm = 1.0
    torch.nn.utils.clip_grad_norm_([W], max_norm)

    clipped_grad_norm = W.grad.norm().item()
    print(f"Gradient norm after clipping: {clipped_grad_norm:.4f}")

gradient_clipping_demo()

print("\nGradient clipping:")
print("  - Essential for RNN/Transformer")
print("  - Typically max_norm = 1.0 or 5.0")
print("  - Greatly improves training stability")
```

### 7.4 Practical Optimization Recipe

```python
# PyTorch-style optimization configuration example
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        return self.layers(x)

model = SimpleNN()

# 1. Optimizer choice: Adam (in most cases)
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))

# 2. Learning rate scheduler: Cosine Annealing
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# 3. Training loop (mock)
for epoch in range(10):
    # Forward & Backward
    # loss.backward()

    # Gradient clipping (RNN/Transformer)
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()

    # Update learning rate
    scheduler.step()

    print(f"Epoch {epoch+1}, LR: {scheduler.get_last_lr()[0]:.6f}")

print("\nPractical optimization tips:")
print("  1. Start with Adam as default (lr=1e-3)")
print("  2. Apply Cosine Annealing or Step Decay")
print("  3. Use Warm-up (Transformer etc.)")
print("  4. Gradient clipping (RNN/Transformer)")
print("  5. Batch size: 32-256 (depending on GPU memory)")
print("  6. Weight Decay (L2 regularization): 1e-4 ~ 1e-5")
```

## Practice Problems

1. **Convergence Rate Analysis**: For $f(x) = \frac{1}{2}x^T A x$ (quadratic form), simulate and compare convergence rates of gradient descent when condition number $\kappa = 10$ vs $\kappa = 100$. Verify if it matches the theoretical rate $O((1 - 1/\kappa)^t)$.

2. **SGD vs Batch GD**: Train a simple 2-layer neural network on MNIST, comparing batch GD, mini-batch SGD (batch sizes 32, 128, 512), and full SGD (batch size 1). Compare convergence speed, final test accuracy, and computation time.

3. **Optimizer Implementation**: Implement the Adam optimizer from scratch using NumPy and visualize the difference with and without bias correction. Test on Rosenbrock or Beale function.

4. **Momentum vs Nesterov**: Compare convergence rates of momentum SGD and Nesterov accelerated gradient on an ill-conditioned quadratic function ($\kappa = 100$). Analyze situations where Nesterov is superior.

5. **Learning Rate Schedules**: Train a simple CNN on CIFAR-10 and compare: (1) fixed learning rate, (2) Step Decay, (3) Cosine Annealing, (4) 1-Cycle Policy. Report learning curves and final test accuracy for each method.

## References

- Ruder, S. (2016). "An Overview of Gradient Descent Optimization Algorithms". arXiv:1609.04747
  - Comprehensive review of all major optimizers
- Bottou, L., Curtis, F. E., & Nocedal, J. (2018). "Optimization Methods for Large-Scale Machine Learning". *SIAM Review*, 60(2), 223-311
  - Theory and practice of large-scale ML optimization
- Kingma, D. P., & Ba, J. (2015). "Adam: A Method for Stochastic Optimization". ICLR
  - Original Adam paper
- Sutskever, I., et al. (2013). "On the Importance of Initialization and Momentum in Deep Learning". ICML
  - Importance of momentum
- Keskar, N. S., et al. (2017). "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima". ICLR
  - Sharp vs Flat Minima
- Smith, L. N. (2017). "Cyclical Learning Rates for Training Neural Networks". WACV
  - 1-Cycle Policy
- PyTorch Optimization Documentation: https://pytorch.org/docs/stable/optim.html
- Stanford CS231n Lecture Notes: http://cs231n.github.io/neural-networks-3/
