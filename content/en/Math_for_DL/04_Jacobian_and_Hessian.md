# Lesson 4: Jacobian and Hessian

## Learning Objectives

- Define and compute the Jacobian matrix of a vector-valued function
- Understand the Jacobian as a linear approximation of a nonlinear mapping
- Define the Hessian matrix and interpret it as the curvature of a scalar function
- Compute second-order Taylor expansions using the Hessian
- Relate the Hessian eigenvalues to the local geometry of the loss surface
- Understand Newton's method and its connection to the Hessian
- Recognize why second-order methods are impractical for large-scale DL and what approximations exist
- Relate the Hessian to the Fisher information matrix

---

## 1. The Jacobian Matrix

### 1.1 Definition

For a function $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$ with components $f_1, \ldots, f_m$, the **Jacobian** is the $m \times n$ matrix of all first-order partial derivatives:

$$\mathbf{J} = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n} \end{bmatrix}$$

Row $i$ is the gradient of $f_i$; column $j$ tells how all outputs change when $x_j$ changes.

### 1.2 The Jacobian as a Linear Approximation

The Jacobian gives the best linear approximation to $\mathbf{f}$ near $\mathbf{x}_0$:

$$\mathbf{f}(\mathbf{x}_0 + \boldsymbol{\delta}) \approx \mathbf{f}(\mathbf{x}_0) + \mathbf{J}(\mathbf{x}_0) \boldsymbol{\delta}$$

This is the multivariate first-order Taylor expansion.

### 1.3 Examples

**Example 1**: Polar to Cartesian coordinates $\mathbf{f}(r, \theta) = (r\cos\theta, r\sin\theta)$

$$\mathbf{J} = \begin{bmatrix} \cos\theta & -r\sin\theta \\ \sin\theta & r\cos\theta \end{bmatrix}$$

The determinant $|\det(\mathbf{J})| = r$ is the familiar Jacobian determinant from change of variables in integration.

**Example 2**: A linear layer $\mathbf{f}(\mathbf{x}) = \mathbf{W}\mathbf{x} + \mathbf{b}$

$$\mathbf{J} = \mathbf{W}$$

The Jacobian of a linear function is the weight matrix itself.

**Example 3**: Element-wise ReLU $\mathbf{f}(\mathbf{x}) = \text{ReLU}(\mathbf{x})$

$$\mathbf{J} = \text{diag}(\mathbf{1}[x_1 > 0], \ldots, \mathbf{1}[x_n > 0])$$

A diagonal matrix with 0s and 1s. The Jacobian of an element-wise function is always diagonal.

```python
import numpy as np

def compute_jacobian_numerical(f, x, eps=1e-5):
    """Compute the Jacobian of f: R^n -> R^m by central differences."""
    x = np.asarray(x, dtype=float)
    f0 = np.asarray(f(x))
    n = len(x)
    m = len(f0)
    J = np.zeros((m, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = eps
        J[:, j] = (f(x + e_j) - f(x - e_j)) / (2 * eps)
    return J

# Example: softmax
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

x = np.array([1.0, 2.0, 3.0])
J_num = compute_jacobian_numerical(softmax, x)

# Analytical Jacobian of softmax
s = softmax(x)
J_ana = np.diag(s) - np.outer(s, s)

print("Softmax Jacobian (numerical):")
print(J_num.round(4))
print("\nSoftmax Jacobian (analytical):")
print(J_ana.round(4))
print(f"\nMax error: {np.max(np.abs(J_num - J_ana)):.2e}")
```

---

## 2. Jacobians of Important DL Functions

### 2.1 Softmax Jacobian

For $\mathbf{s} = \text{softmax}(\mathbf{z})$ where $s_i = \frac{e^{z_i}}{\sum_k e^{z_k}}$:

$$\frac{\partial s_i}{\partial z_j} = s_i(\delta_{ij} - s_j)$$

In matrix form:

$$\mathbf{J}_{\text{softmax}} = \text{diag}(\mathbf{s}) - \mathbf{s}\mathbf{s}^\top$$

**Properties**:
- Each row sums to zero (since $\sum_i s_i = 1$ is constant)
- The Jacobian is symmetric
- Rank is $n - 1$ (the constraint $\sum s_i = 1$ removes one degree of freedom)

### 2.2 Batch Normalization Jacobian

For batch norm $\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$ where $\mu = \frac{1}{n}\sum x_i$ and $\sigma^2 = \frac{1}{n}\sum(x_i - \mu)^2$:

$$\frac{\partial \hat{x}_i}{\partial x_j} = \frac{1}{\sqrt{\sigma^2 + \epsilon}} \left(\delta_{ij} - \frac{1}{n} - \frac{\hat{x}_i \hat{x}_j}{n}\right)$$

This couples all elements in the batch, which is why batch norm introduces dependencies between samples.

### 2.3 Layer Normalization Jacobian

Similar structure to batch norm, but normalization is over the feature dimension instead of the batch dimension. The Jacobian has the same form but with $n$ being the feature dimension.

---

## 3. The Hessian Matrix

### 3.1 Definition

For a scalar function $f: \mathbb{R}^n \to \mathbb{R}$, the **Hessian** is the $n \times n$ matrix of second-order partial derivatives:

$$\mathbf{H} = \nabla^2 f = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\ \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2} \end{bmatrix}$$

By Schwarz's theorem (for twice continuously differentiable functions): $\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial^2 f}{\partial x_j \partial x_i}$, so $\mathbf{H}$ is **symmetric**.

### 3.2 The Hessian as Curvature

The Hessian encodes the **curvature** of $f$. The second-order Taylor expansion around $\mathbf{x}_0$:

$$f(\mathbf{x}_0 + \boldsymbol{\delta}) \approx f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^\top \boldsymbol{\delta} + \frac{1}{2} \boldsymbol{\delta}^\top \mathbf{H}(\mathbf{x}_0) \boldsymbol{\delta}$$

The term $\frac{1}{2}\boldsymbol{\delta}^\top \mathbf{H} \boldsymbol{\delta}$ is a quadratic form that describes how the function curves in each direction.

### 3.3 Eigenvalue Interpretation

Since $\mathbf{H}$ is symmetric, it has real eigenvalues $\lambda_1, \ldots, \lambda_n$ and orthogonal eigenvectors $\mathbf{v}_1, \ldots, \mathbf{v}_n$.

- $\lambda_i > 0$: $f$ curves **upward** along direction $\mathbf{v}_i$ (bowl-shaped)
- $\lambda_i < 0$: $f$ curves **downward** along direction $\mathbf{v}_i$ (ridge-shaped)
- $\lambda_i = 0$: $f$ is **flat** along direction $\mathbf{v}_i$ (no curvature)

| Hessian Eigenvalues | Critical Point Type |
|--------------------|--------------------|
| All $\lambda_i > 0$ | Local minimum |
| All $\lambda_i < 0$ | Local maximum |
| Mixed signs | **Saddle point** |
| Some $\lambda_i = 0$ | Degenerate (inconclusive) |

```python
import numpy as np
import matplotlib.pyplot as plt

def compute_hessian_numerical(f, x, eps=1e-5):
    """Compute the Hessian by finite differences of the gradient."""
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            e_i = np.zeros(n); e_i[i] = eps
            e_j = np.zeros(n); e_j[j] = eps
            H[i, j] = (f(x + e_i + e_j) - f(x + e_i - e_j)
                       - f(x - e_i + e_j) + f(x - e_i - e_j)) / (4 * eps**2)
    return 0.5 * (H + H.T)  # Enforce symmetry

# Example: f(x, y) = x^3 - 3xy^2 (monkey saddle)
def monkey_saddle(x):
    return x[0]**3 - 3 * x[0] * x[1]**2

x0 = np.array([0.0, 0.0])
H = compute_hessian_numerical(monkey_saddle, x0)
eigvals, eigvecs = np.linalg.eigh(H)

print(f"Hessian at origin:\n{H.round(4)}")
print(f"Eigenvalues: {eigvals.round(4)}")
print(f"Critical point type: {'saddle' if np.any(eigvals < 0) and np.any(eigvals > 0) else 'degenerate'}")
```

---

## 4. Saddle Points in Deep Learning

### 4.1 Why Saddle Points Dominate

In high-dimensional optimization (e.g., neural network training), saddle points are far more common than local minima. For a critical point to be a local minimum, **all** $n$ eigenvalues must be positive. In high dimensions, the probability of this happening by chance is exponentially small.

**Rough argument**: If each eigenvalue is independently positive with probability $p = 0.5$, then:

$$P(\text{local minimum}) = p^n = 0.5^n$$

For $n = 10^6$ parameters, this is essentially zero. Saddle points (mixed eigenvalue signs) are overwhelmingly more likely.

### 4.2 Saddle Point Visualization

```python
# Saddle point: f(x, y) = x^2 - y^2
def saddle(x):
    return x[0]**2 - x[1]**2

x = np.linspace(-2, 2, 200)
y = np.linspace(-2, 2, 200)
X, Y = np.meshgrid(x, y)
Z = X**2 - Y**2

fig = plt.figure(figsize=(14, 5))

# 3D surface
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='RdBu', alpha=0.8)
ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('f')
ax1.set_title('Saddle point: $f = x^2 - y^2$')

# Contour with gradient
ax2 = fig.add_subplot(122)
ax2.contour(X, Y, Z, levels=20)
ax2.plot(0, 0, 'ko', markersize=8, label='Saddle point')
ax2.set_xlabel('x'); ax2.set_ylabel('y')
ax2.set_title('Contours and saddle point')
ax2.legend()
plt.tight_layout()
plt.show()
```

### 4.3 Hessian Spectrum of Real Networks

Research has shown that the Hessian of neural network loss functions has a characteristic spectrum:
- A **bulk** of eigenvalues near zero (many flat directions)
- A few **outlier** eigenvalues that are large and positive
- Typically very few (if any) significantly negative eigenvalues near convergence

This means the loss surface is like a high-dimensional valley with a few steep walls and many flat directions.

---

## 5. Newton's Method

### 5.1 Idea

Newton's method uses the Hessian to take "curvature-aware" steps:

$$\mathbf{x}^{(t+1)} = \mathbf{x}^{(t)} - \mathbf{H}^{-1} \nabla f(\mathbf{x}^{(t)})$$

**Intuition**: Instead of following the steepest descent direction, Newton's method jumps directly to the minimum of the local quadratic approximation.

### 5.2 Why It Works for Quadratics

For $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top \mathbf{A}\mathbf{x} - \mathbf{b}^\top \mathbf{x}$:
- $\nabla f = \mathbf{A}\mathbf{x} - \mathbf{b}$
- $\mathbf{H} = \mathbf{A}$
- Newton step: $\mathbf{x}^{(1)} = \mathbf{x}^{(0)} - \mathbf{A}^{-1}(\mathbf{A}\mathbf{x}^{(0)} - \mathbf{b}) = \mathbf{A}^{-1}\mathbf{b} = \mathbf{x}^*$

Newton's method converges in **one step** on quadratics, regardless of the condition number.

### 5.3 Why Newton's Method Is Impractical for DL

| Issue | Details |
|-------|---------|
| Memory | $\mathbf{H} \in \mathbb{R}^{n \times n}$; for $n = 10^6$, storing $\mathbf{H}$ needs $10^{12}$ floats ($\sim 4$ TB) |
| Computation | Computing $\mathbf{H}^{-1}\mathbf{g}$ costs $O(n^3)$ |
| Non-convexity | $\mathbf{H}$ may be indefinite; Newton step could ascend |
| Stochasticity | Mini-batch gradients are noisy; Hessian estimation is even noisier |

```python
# Compare gradient descent vs. Newton's method on a 2D quadratic
A = np.array([[10.0, 3.0],
              [3.0, 2.0]])
b_vec = np.array([1.0, 2.0])
x_star = np.linalg.solve(A, b_vec)

def f_quad(x):
    return 0.5 * x @ A @ x - b_vec @ x

# Gradient descent
x_gd = np.array([5.0, -3.0])
traj_gd = [x_gd.copy()]
for _ in range(50):
    grad = A @ x_gd - b_vec
    x_gd = x_gd - 0.05 * grad
    traj_gd.append(x_gd.copy())
traj_gd = np.array(traj_gd)

# Newton's method
x_nt = np.array([5.0, -3.0])
traj_nt = [x_nt.copy()]
for _ in range(5):
    grad = A @ x_nt - b_vec
    x_nt = x_nt - np.linalg.solve(A, grad)
    traj_nt.append(x_nt.copy())
traj_nt = np.array(traj_nt)

# Plot
x1 = np.linspace(-2, 6, 200)
x2 = np.linspace(-5, 3, 200)
X1, X2 = np.meshgrid(x1, x2)
Z = 0.5 * (A[0,0]*X1**2 + 2*A[0,1]*X1*X2 + A[1,1]*X2**2) - b_vec[0]*X1 - b_vec[1]*X2

fig, ax = plt.subplots(figsize=(8, 6))
ax.contour(X1, X2, Z, levels=30, alpha=0.5)
ax.plot(traj_gd[:, 0], traj_gd[:, 1], 'ro-', markersize=3, label=f'GD ({len(traj_gd)-1} steps)')
ax.plot(traj_nt[:, 0], traj_nt[:, 1], 'bs-', markersize=6, label=f'Newton ({len(traj_nt)-1} steps)')
ax.plot(x_star[0], x_star[1], 'g*', markersize=15, label='Optimum')
ax.legend()
ax.set_title('Gradient Descent vs. Newton\'s Method')
plt.show()
```

---

## 6. Hessian Approximations in Practice

### 6.1 Diagonal Hessian Approximation

The simplest approximation: keep only diagonal entries.

$$\mathbf{H} \approx \text{diag}(H_{11}, H_{22}, \ldots, H_{nn})$$

Used in **AdaGrad/RMSProp/Adam**: the accumulated squared gradients approximate diagonal Hessian entries.

### 6.2 Gauss-Newton Approximation

For a loss $L = \frac{1}{2}\|\mathbf{r}(\boldsymbol{\theta})\|^2$ where $\mathbf{r}$ is the residual vector:

$$\mathbf{H} = \mathbf{J}_\mathbf{r}^\top \mathbf{J}_\mathbf{r} + \sum_i r_i \nabla^2 r_i$$

The **Gauss-Newton** approximation drops the second term:

$$\mathbf{H} \approx \mathbf{J}_\mathbf{r}^\top \mathbf{J}_\mathbf{r}$$

This is always positive semi-definite, avoiding the saddle-point problem.

### 6.3 Fisher Information Matrix

For probabilistic models with parameters $\boldsymbol{\theta}$ and log-likelihood $\ell(\boldsymbol{\theta}; \mathbf{x})$:

$$\mathbf{F} = \mathbb{E}\left[\nabla \ell \, \nabla \ell^\top\right]$$

The Fisher information matrix equals the expected Hessian of the negative log-likelihood (for the true data distribution):

$$\mathbf{F} = -\mathbb{E}\left[\nabla^2 \ell\right]$$

**In natural gradient descent**, the update is $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta \mathbf{F}^{-1} \nabla L$, which accounts for the geometry of the probability distribution space.

### 6.4 Hessian-Vector Products

Even without forming $\mathbf{H}$ explicitly, we can compute $\mathbf{H}\mathbf{v}$ for any vector $\mathbf{v}$ using the identity:

$$\mathbf{H}\mathbf{v} = \lim_{\epsilon \to 0} \frac{\nabla f(\mathbf{x} + \epsilon \mathbf{v}) - \nabla f(\mathbf{x})}{\epsilon}$$

This requires only two gradient evaluations and costs $O(n)$ -- no need to form the $O(n^2)$ Hessian.

```python
# Hessian-vector product via finite differences of gradient
def hvp_finite_diff(grad_f, x, v, eps=1e-4):
    """Compute H @ v using finite differences of the gradient."""
    return (grad_f(x + eps * v) - grad_f(x - eps * v)) / (2 * eps)

# Example
def f_example(x):
    return x[0]**3 + x[1]**3 + x[0]*x[1]

def grad_f_example(x):
    return np.array([3*x[0]**2 + x[1], 3*x[1]**2 + x[0]])

x0 = np.array([1.0, 2.0])
v = np.array([1.0, 0.0])

# Analytical Hessian
H_analytical = np.array([[6*x0[0], 1.0],
                          [1.0, 6*x0[1]]])
Hv_analytical = H_analytical @ v

# Finite difference HVP
Hv_numerical = hvp_finite_diff(grad_f_example, x0, v)

print(f"H @ v (analytical): {Hv_analytical}")
print(f"H @ v (numerical):  {Hv_numerical}")
print(f"Error: {np.linalg.norm(Hv_analytical - Hv_numerical):.2e}")
```

---

## 7. Hessian and Loss Surface Geometry

### 7.1 Condition Number of the Hessian

The condition number $\kappa(\mathbf{H}) = \lambda_{\max} / \lambda_{\min}$ determines the eccentricity of the loss surface:

- $\kappa \approx 1$: isotropic (all directions are equally curved)
- $\kappa \gg 1$: anisotropic (some directions are much more curved than others)

High condition numbers cause gradient descent to zigzag, requiring many iterations.

### 7.2 Connecting to Adaptive Optimizers

Adam and similar optimizers implicitly adapt to the curvature:

$$\theta_i \leftarrow \theta_i - \frac{\eta}{\sqrt{v_i} + \epsilon} \cdot m_i$$

where $v_i \approx \mathbb{E}[g_i^2]$ estimates the diagonal Hessian. Parameters with large curvature (large $v_i$) receive smaller updates, effectively preconditioning the optimization.

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| Jacobian | $m \times n$ matrix of first derivatives; best linear approximation of a nonlinear mapping |
| Softmax Jacobian | $\text{diag}(\mathbf{s}) - \mathbf{s}\mathbf{s}^\top$; rank $n-1$, symmetric |
| Hessian | $n \times n$ matrix of second derivatives; encodes curvature |
| Eigenvalue interpretation | Positive = bowl, negative = ridge, mixed = saddle point |
| Newton's method | Uses $\mathbf{H}^{-1} \nabla f$; quadratic convergence but $O(n^2)$ memory |
| HVP | $\mathbf{H}\mathbf{v}$ via finite differences: $O(n)$ cost, no need to store $\mathbf{H}$ |
| Practical approximations | Diagonal (Adam), Gauss-Newton, Fisher information |

---

## Exercises

1. Compute the Jacobian of the function $\mathbf{f}(x, y) = (x^2 y, \sin(xy), e^x + y)$ analytically and verify numerically.
2. Compute the Hessian of $f(x, y) = x^4 + y^4 - 2x^2 y^2$ and classify all critical points.
3. Implement Newton's method on the Rosenbrock function and compare convergence with gradient descent.
4. Compute the softmax Jacobian at $\mathbf{z} = (1, 2, 3)$ and verify that each row sums to zero.
5. Implement a Hessian-vector product function and use it to find the largest eigenvalue via power iteration.

---

**Next**: [05. Optimization Theory](05_Optimization_Theory.md)
