# Lesson 11: Positive Definite Matrices

[Previous: Lesson 10](./10_Matrix_Norms_and_Condition_Numbers.md) | [Overview](./00_Overview.md) | [Next: Lesson 12](./12_Sparse_Matrices.md)

---

## Learning Objectives

- Define positive definite, positive semidefinite, negative definite, and indefinite matrices
- Apply multiple tests for positive definiteness: eigenvalue test, Cholesky test, leading principal minors, and pivots
- Understand the connection between quadratic forms and positive definite matrices
- Explain the role of the Hessian matrix in optimization and convexity
- Recognize applications in covariance matrices, kernel methods, and semidefinite programming
- Implement positive definiteness checks and Cholesky decomposition in Python

---

## 1. Definition and Quadratic Forms

### 1.1 Quadratic Forms

A **quadratic form** is a scalar-valued function of a vector $\mathbf{x} \in \mathbb{R}^n$ defined by a symmetric matrix $A \in \mathbb{R}^{n \times n}$:

$$Q(\mathbf{x}) = \mathbf{x}^T A \mathbf{x} = \sum_{i=1}^{n} \sum_{j=1}^{n} a_{ij} x_i x_j$$

For example, with a $2 \times 2$ symmetric matrix:

$$Q(x_1, x_2) = \begin{bmatrix} x_1 & x_2 \end{bmatrix} \begin{bmatrix} a & b \\ b & c \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = a x_1^2 + 2b x_1 x_2 + c x_2^2$$

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define a symmetric matrix
A = np.array([[3, 1],
              [1, 2]])

# Evaluate quadratic form at a point
x = np.array([1, 2])
Q = x.T @ A @ x
print(f"Q(x) = x^T A x = {Q}")

# Visualize the quadratic form surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

x1 = np.linspace(-3, 3, 100)
x2 = np.linspace(-3, 3, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = A[0, 0] * X1**2 + 2 * A[0, 1] * X1 * X2 + A[1, 1] * X2**2

ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('Q(x)')
ax.set_title('Quadratic Form: x^T A x (Positive Definite)')
plt.tight_layout()
plt.show()
```

### 1.2 Classification of Symmetric Matrices

A symmetric matrix $A$ is classified based on its quadratic form:

| Classification | Condition | Eigenvalues |
|---|---|---|
| **Positive definite** (PD) | $\mathbf{x}^T A \mathbf{x} > 0$ for all $\mathbf{x} \neq \mathbf{0}$ | All $\lambda_i > 0$ |
| **Positive semidefinite** (PSD) | $\mathbf{x}^T A \mathbf{x} \geq 0$ for all $\mathbf{x}$ | All $\lambda_i \geq 0$ |
| **Negative definite** (ND) | $\mathbf{x}^T A \mathbf{x} < 0$ for all $\mathbf{x} \neq \mathbf{0}$ | All $\lambda_i < 0$ |
| **Negative semidefinite** (NSD) | $\mathbf{x}^T A \mathbf{x} \leq 0$ for all $\mathbf{x}$ | All $\lambda_i \leq 0$ |
| **Indefinite** | $Q$ takes both positive and negative values | Mixed signs |

We write $A \succ 0$ for positive definite and $A \succeq 0$ for positive semidefinite. This notation defines a partial ordering on symmetric matrices called the **Loewner order**.

```python
# Classify different matrices
matrices = {
    "Positive definite": np.array([[2, -1], [-1, 2]]),
    "Positive semidefinite": np.array([[1, 1], [1, 1]]),
    "Negative definite": np.array([[-3, 1], [1, -4]]),
    "Indefinite": np.array([[1, 0], [0, -1]]),
}

for name, M in matrices.items():
    eigenvalues = np.linalg.eigvalsh(M)
    print(f"{name}:")
    print(f"  Matrix:\n{M}")
    print(f"  Eigenvalues: {eigenvalues}")

    # Test with random vectors
    n_tests = 10000
    results = []
    for _ in range(n_tests):
        x = np.random.randn(2)
        results.append(x.T @ M @ x)
    results = np.array(results)
    print(f"  Min Q(x): {results.min():.4f}, Max Q(x): {results.max():.4f}")
    print()
```

---

## 2. Tests for Positive Definiteness

There are several equivalent tests to determine whether a symmetric matrix is positive definite. Each one has different computational advantages.

### 2.1 Eigenvalue Test

A symmetric matrix $A$ is positive definite if and only if **all eigenvalues are strictly positive**.

```python
def is_positive_definite_eigenvalue(A, tol=1e-10):
    """Check positive definiteness via eigenvalues."""
    if not np.allclose(A, A.T):
        raise ValueError("Matrix must be symmetric")
    eigenvalues = np.linalg.eigvalsh(A)
    return np.all(eigenvalues > tol), eigenvalues

# Example
A = np.array([[4, 2, 1],
              [2, 5, 3],
              [1, 3, 6]])

is_pd, eigenvalues = is_positive_definite_eigenvalue(A)
print(f"Eigenvalues: {eigenvalues}")
print(f"Is positive definite? {is_pd}")
```

### 2.2 Cholesky Decomposition Test

A symmetric matrix $A$ is positive definite if and only if it has a **Cholesky decomposition** $A = LL^T$, where $L$ is a lower triangular matrix with strictly positive diagonal entries.

The Cholesky decomposition is essentially Gaussian elimination that exploits symmetry. It requires roughly half the computation of a general LU decomposition.

```python
def is_positive_definite_cholesky(A):
    """Check positive definiteness via Cholesky decomposition."""
    try:
        L = np.linalg.cholesky(A)
        return True, L
    except np.linalg.LinAlgError:
        return False, None

# Positive definite matrix
A_pd = np.array([[4, 2, 1],
                 [2, 5, 3],
                 [1, 3, 6]])

is_pd, L = is_positive_definite_cholesky(A_pd)
print(f"Is PD (Cholesky)? {is_pd}")
if L is not None:
    print(f"L:\n{L}")
    print(f"Reconstruction error: {np.linalg.norm(A_pd - L @ L.T):.2e}")

# Not positive definite
A_indef = np.array([[1, 2],
                    [2, 1]])
is_pd, _ = is_positive_definite_cholesky(A_indef)
print(f"\nIndefinite matrix is PD? {is_pd}")
```

### 2.3 Leading Principal Minors (Sylvester's Criterion)

A symmetric matrix $A$ is positive definite if and only if **all leading principal minors are positive**. The $k$-th leading principal minor is the determinant of the upper-left $k \times k$ submatrix.

$$\Delta_1 = a_{11} > 0, \quad \Delta_2 = \det\begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix} > 0, \quad \ldots, \quad \Delta_n = \det(A) > 0$$

```python
def is_positive_definite_minors(A, tol=1e-10):
    """Check positive definiteness via leading principal minors (Sylvester)."""
    n = A.shape[0]
    minors = []
    for k in range(1, n + 1):
        submatrix = A[:k, :k]
        det_k = np.linalg.det(submatrix)
        minors.append(det_k)
        if det_k <= tol:
            return False, minors
    return True, minors

A = np.array([[4, 2, 1],
              [2, 5, 3],
              [1, 3, 6]])

is_pd, minors = is_positive_definite_minors(A)
print(f"Leading principal minors: {minors}")
print(f"Is positive definite? {is_pd}")

# Compare: indefinite matrix
A_indef = np.array([[1, 3],
                    [3, 1]])
is_pd, minors = is_positive_definite_minors(A_indef)
print(f"\nIndefinite matrix minors: {minors}")
print(f"Is positive definite? {is_pd}")
```

### 2.4 Pivot Test

During Gaussian elimination (without row exchanges), a symmetric matrix is positive definite if and only if **all pivots are strictly positive**.

```python
def is_positive_definite_pivots(A, tol=1e-10):
    """Check positive definiteness via Gaussian elimination pivots."""
    n = A.shape[0]
    U = A.astype(float).copy()
    pivots = []

    for i in range(n):
        if abs(U[i, i]) <= tol:
            return False, pivots
        pivots.append(U[i, i])
        for j in range(i + 1, n):
            factor = U[j, i] / U[i, i]
            U[j, i:] -= factor * U[i, i:]

    all_positive = all(p > tol for p in pivots)
    return all_positive, pivots

A = np.array([[4, 2, 1],
              [2, 5, 3],
              [1, 3, 6]])

is_pd, pivots = is_positive_definite_pivots(A)
print(f"Pivots: {pivots}")
print(f"Is positive definite? {is_pd}")
```

### 2.5 Comparison of Tests

| Test | Complexity | Pros | Cons |
|---|---|---|---|
| Eigenvalue | $O(n^3)$ | Gives all eigenvalues | Full eigendecomposition needed |
| Cholesky | $O(n^3/3)$ | Fastest; gives useful decomposition | Only yes/no answer |
| Sylvester's Criterion | $O(n^4)$ | Theoretical elegance | Numerically unstable for large $n$ |
| Pivot | $O(n^3/3)$ | Tied to Gaussian elimination | Requires careful implementation |

In practice, the **Cholesky test** is the standard method. If the decomposition succeeds, the matrix is positive definite. The Cholesky factor $L$ is itself useful for solving linear systems and sampling from multivariate Gaussians.

---

## 3. Properties of Positive Definite Matrices

### 3.1 Algebraic Properties

Positive definite matrices satisfy many useful closure properties:

```python
import numpy as np
from scipy.linalg import sqrtm

# Start with a positive definite matrix
A = np.array([[4, 1],
              [1, 3]])

eigenvalues_A = np.linalg.eigvalsh(A)
print(f"A eigenvalues: {eigenvalues_A} (all positive)")

# Property 1: Sum of PD matrices is PD
B = np.array([[2, 0.5],
              [0.5, 2]])
C = A + B
print(f"\nA + B eigenvalues: {np.linalg.eigvalsh(C)} (all positive)")

# Property 2: Scalar multiple (positive scalar) preserves PD
print(f"3A eigenvalues: {np.linalg.eigvalsh(3 * A)} (all positive)")

# Property 3: Inverse of PD is PD
A_inv = np.linalg.inv(A)
print(f"A^(-1) eigenvalues: {np.linalg.eigvalsh(A_inv)} (all positive)")

# Property 4: Diagonal entries are positive
print(f"A diagonal: {np.diag(A)} (all positive)")

# Property 5: det(A) > 0 for PD
print(f"det(A) = {np.linalg.det(A):.4f} (positive)")

# Property 6: Congruence transformation preserves PD
# If A is PD and P is invertible, then P^T A P is PD
P = np.array([[1, 2],
              [0, 1]])
P_A_P = P.T @ A @ P
print(f"\nP^T A P eigenvalues: {np.linalg.eigvalsh(P_A_P)} (all positive)")

# Property 7: A^(1/2) exists and is PD (matrix square root)
A_sqrt = sqrtm(A).real
print(f"\nsqrt(A):\n{A_sqrt}")
print(f"sqrt(A) @ sqrt(A) = A? {np.allclose(A_sqrt @ A_sqrt, A)}")
print(f"sqrt(A) eigenvalues: {np.linalg.eigvalsh(A_sqrt)} (all positive)")
```

### 3.2 Geometric Interpretation: Ellipsoids

A positive definite matrix $A$ defines an ellipsoid:

$$\mathcal{E} = \{ \mathbf{x} \in \mathbb{R}^n : \mathbf{x}^T A \mathbf{x} \leq 1 \}$$

The eigenvectors of $A$ give the principal axes of the ellipsoid, and the eigenvalues determine the lengths of those axes (the semi-axis length along eigenvector $\mathbf{v}_i$ is $1/\sqrt{\lambda_i}$).

```python
# Visualize ellipsoids for different PD matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

matrices = [
    np.array([[2, 0], [0, 2]]),    # Circle (identity-like)
    np.array([[4, 1], [1, 1]]),    # Tilted ellipse
    np.array([[5, 3], [3, 5]]),    # Tilted ellipse (different orientation)
]
titles = ["A = 2I (circle)", "Tilted ellipse", "Rotated ellipse"]

theta = np.linspace(0, 2 * np.pi, 200)
circle = np.array([np.cos(theta), np.sin(theta)])

for ax, A, title in zip(axes, matrices, titles):
    eigenvalues, eigenvectors = np.linalg.eigh(A)

    # The ellipsoid x^T A x = 1 is the image of the unit circle under A^(-1/2)
    A_inv_sqrt = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    ellipse = A_inv_sqrt @ circle

    ax.plot(ellipse[0], ellipse[1], 'b-', linewidth=2)

    # Draw principal axes
    for i in range(2):
        v = eigenvectors[:, i] / np.sqrt(eigenvalues[i])
        ax.arrow(0, 0, v[0], v[1], head_width=0.05, head_length=0.03,
                 fc=f'C{i+1}', ec=f'C{i+1}', linewidth=2,
                 label=f'lambda={eigenvalues[i]:.2f}')
        ax.arrow(0, 0, -v[0], -v[1], head_width=0.05, head_length=0.03,
                 fc=f'C{i+1}', ec=f'C{i+1}', linewidth=2)

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.legend()

plt.tight_layout()
plt.show()
```

---

## 4. The Hessian and Optimization

### 4.1 The Hessian Matrix

For a scalar-valued function $f: \mathbb{R}^n \to \mathbb{R}$, the Hessian matrix $H$ contains all second-order partial derivatives:

$$H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$$

The Hessian is always symmetric (assuming continuous second derivatives, by Schwarz's theorem). At a critical point (where $\nabla f = \mathbf{0}$):

- $H \succ 0$ (positive definite) $\Rightarrow$ **local minimum**
- $H \prec 0$ (negative definite) $\Rightarrow$ **local maximum**
- $H$ indefinite $\Rightarrow$ **saddle point**

```python
from scipy.optimize import minimize

# Example: f(x, y) = 3x^2 + 2xy + 2y^2 - 6x - 4y
# Gradient: [6x + 2y - 6, 2x + 4y - 4]
# Hessian: [[6, 2], [2, 4]]

def f(x):
    return 3*x[0]**2 + 2*x[0]*x[1] + 2*x[1]**2 - 6*x[0] - 4*x[1]

def grad_f(x):
    return np.array([6*x[0] + 2*x[1] - 6, 2*x[0] + 4*x[1] - 4])

# Hessian is constant for a quadratic function
H = np.array([[6, 2],
              [2, 4]])

eigenvalues_H = np.linalg.eigvalsh(H)
print(f"Hessian eigenvalues: {eigenvalues_H}")
print(f"Hessian is PD: {np.all(eigenvalues_H > 0)}")
print("=> Critical point is a local minimum")

# Find the critical point
result = minimize(f, x0=[0, 0], jac=grad_f)
print(f"\nMinimum at: {result.x}")
print(f"Minimum value: {result.fun:.4f}")

# Verify: solve gradient = 0
x_star = np.linalg.solve(H, np.array([6, 4]))
print(f"Analytic solution: {x_star}")
```

### 4.2 Convexity and Positive Semidefiniteness

A twice-differentiable function $f$ is **convex** if and only if its Hessian $H(x) \succeq 0$ for all $x$. Convexity is the cornerstone of optimization: every local minimum of a convex function is a global minimum.

```python
# Visualize convex vs non-convex functions
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

x = np.linspace(-3, 3, 200)
y = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(x, y)

# Convex function: f(x,y) = x^2 + y^2 (Hessian = 2I, PD)
Z_convex = X**2 + Y**2
axes[0].contour(X, Y, Z_convex, levels=20, cmap='viridis')
axes[0].set_title('Convex: f = x^2 + y^2\nHessian = 2I (PD)')
axes[0].set_aspect('equal')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')

# Non-convex (saddle): f(x,y) = x^2 - y^2 (Hessian indefinite)
Z_saddle = X**2 - Y**2
axes[1].contour(X, Y, Z_saddle, levels=20, cmap='RdBu_r')
axes[1].set_title('Saddle: f = x^2 - y^2\nHessian indefinite')
axes[1].set_aspect('equal')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')

plt.tight_layout()
plt.show()
```

### 4.3 Newton's Method and the Hessian

Newton's method for optimization uses the Hessian to compute the update direction:

$$\mathbf{x}_{k+1} = \mathbf{x}_k - H^{-1} \nabla f(\mathbf{x}_k)$$

This works well when $H$ is positive definite. When it is not, the Newton direction may point uphill, and modifications (Levenberg-Marquardt, trust region) are needed.

```python
def newtons_method(f, grad_f, hess_f, x0, tol=1e-8, max_iter=100):
    """Newton's method for unconstrained optimization."""
    x = np.array(x0, dtype=float)
    history = [x.copy()]

    for i in range(max_iter):
        g = grad_f(x)
        H = hess_f(x)

        if np.linalg.norm(g) < tol:
            print(f"Converged in {i} iterations")
            break

        # Newton step: solve H @ dx = -g
        try:
            dx = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            print("Hessian is singular; cannot proceed")
            break

        x = x + dx
        history.append(x.copy())

    return x, np.array(history)

# Rosenbrock-like function (modified for demonstration)
def f_opt(x):
    return 5 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def grad_f_opt(x):
    dx0 = -20 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    dx1 = 10 * (x[1] - x[0]**2)
    return np.array([dx0, dx1])

def hess_f_opt(x):
    h00 = -20 * (x[1] - 3 * x[0]**2) + 2
    h01 = -20 * x[0]
    h11 = 10
    return np.array([[h00, h01],
                     [h01, h11]])

x_min, history = newtons_method(f_opt, grad_f_opt, hess_f_opt, x0=[-1, 1])
print(f"Minimum at: {x_min}")
print(f"Hessian at minimum:\n{hess_f_opt(x_min)}")
print(f"Hessian eigenvalues: {np.linalg.eigvalsh(hess_f_opt(x_min))}")
```

---

## 5. Positive Definite Matrices in Practice

### 5.1 Covariance Matrices

Every covariance matrix is positive semidefinite. If the data has no perfect linear dependencies, the covariance matrix is positive definite.

$$\text{Cov}(X) = E[(X - \mu)(X - \mu)^T]$$

This is PSD because for any vector $\mathbf{a}$:

$$\mathbf{a}^T \text{Cov}(X) \mathbf{a} = E[(\mathbf{a}^T(X - \mu))^2] \geq 0$$

```python
# Generate multivariate data and verify covariance is PSD
np.random.seed(42)
n_samples = 500
mean = np.array([1, 2, 3])
cov_true = np.array([[2, 1, 0.5],
                     [1, 3, 1],
                     [0.5, 1, 2]])

# Check the true covariance is PD
L = np.linalg.cholesky(cov_true)
print(f"True covariance is PD (Cholesky exists)")

# Generate samples using Cholesky
Z = np.random.randn(n_samples, 3)
X = mean + (L @ Z.T).T

# Empirical covariance
cov_empirical = np.cov(X.T)
print(f"\nTrue covariance:\n{cov_true}")
print(f"\nEmpirical covariance:\n{np.round(cov_empirical, 3)}")

# Verify PD
eigenvalues = np.linalg.eigvalsh(cov_empirical)
print(f"\nEmpirical covariance eigenvalues: {eigenvalues}")
print(f"Is PD? {np.all(eigenvalues > 0)}")
```

### 5.2 Kernel Matrices (Gram Matrices)

A kernel function $k(x, y)$ is valid if and only if the kernel matrix (Gram matrix) is positive semidefinite for any set of input points. This is **Mercer's condition**.

$$K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j) \quad \Rightarrow \quad K \succeq 0$$

```python
from scipy.spatial.distance import cdist

# RBF (Gaussian) kernel
def rbf_kernel(X, gamma=1.0):
    """Compute the RBF kernel matrix."""
    sq_dists = cdist(X, X, 'sqeuclidean')
    return np.exp(-gamma * sq_dists)

# Generate data points
np.random.seed(42)
X = np.random.randn(50, 2)

# Compute kernel matrix
K = rbf_kernel(X, gamma=0.5)

# Verify PSD
eigenvalues = np.linalg.eigvalsh(K)
print(f"Kernel matrix shape: {K.shape}")
print(f"Min eigenvalue: {eigenvalues.min():.6e}")
print(f"Max eigenvalue: {eigenvalues.max():.6e}")
print(f"All eigenvalues >= 0? {np.all(eigenvalues >= -1e-10)}")

# Visualize eigenvalue spectrum
plt.figure(figsize=(8, 4))
plt.plot(eigenvalues, 'b.-')
plt.xlabel('Index')
plt.ylabel('Eigenvalue')
plt.title('Eigenvalue Spectrum of RBF Kernel Matrix')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### 5.3 Regularization to Ensure Positive Definiteness

In practice, matrices that should be PSD may lose this property due to numerical errors. Adding a small diagonal perturbation (ridge) restores positive definiteness:

$$A_{\text{reg}} = A + \epsilon I$$

If $A$ has smallest eigenvalue $\lambda_{\min}$, then $A_{\text{reg}}$ has smallest eigenvalue $\lambda_{\min} + \epsilon$.

```python
def make_positive_definite(A, epsilon=1e-6):
    """Add ridge to ensure positive definiteness."""
    eigenvalues = np.linalg.eigvalsh(A)
    if np.all(eigenvalues > 0):
        return A
    # Shift eigenvalues so minimum is epsilon
    shift = max(0, -eigenvalues.min()) + epsilon
    return A + shift * np.eye(A.shape[0])

# Near-singular covariance (e.g., highly correlated features)
A_near_singular = np.array([[1.0, 0.9999],
                            [0.9999, 1.0]])

eigenvalues = np.linalg.eigvalsh(A_near_singular)
print(f"Original eigenvalues: {eigenvalues}")
print(f"Condition number: {eigenvalues.max() / eigenvalues.min():.0f}")

# Regularize
A_reg = make_positive_definite(A_near_singular, epsilon=1e-4)
eigenvalues_reg = np.linalg.eigvalsh(A_reg)
print(f"\nRegularized eigenvalues: {eigenvalues_reg}")
print(f"Condition number: {eigenvalues_reg.max() / eigenvalues_reg.min():.0f}")
```

---

## 6. Semidefinite Programming (SDP)

### 6.1 What is SDP?

Semidefinite programming is a generalization of linear programming where the constraint involves a matrix being positive semidefinite:

$$\min_X \text{tr}(CX) \quad \text{subject to} \quad \text{tr}(A_i X) = b_i, \quad X \succeq 0$$

SDP is a powerful framework that encompasses linear programming, quadratic programming, and second-order cone programming as special cases. It has applications in control theory, combinatorial optimization, and quantum information.

### 6.2 SDP Example with CVXPY

```python
# pip install cvxpy
import cvxpy as cp

# Simple SDP: find the nearest correlation matrix
# Given a matrix that is almost but not quite a valid correlation matrix,
# find the nearest PSD matrix with unit diagonal

# Corrupted correlation matrix (not PSD)
C = np.array([[1.0,  0.9,  0.7],
              [0.9,  1.0, -0.4],
              [0.7, -0.4,  1.0]])

eigenvalues_C = np.linalg.eigvalsh(C)
print(f"Original eigenvalues: {eigenvalues_C}")
print(f"Is PSD? {np.all(eigenvalues_C >= 0)}")

# SDP formulation: minimize ||X - C||_F subject to X PSD, diag(X) = 1
X = cp.Variable((3, 3), symmetric=True)
objective = cp.Minimize(cp.norm(X - C, 'fro'))
constraints = [X >> 0]  # X is PSD
for i in range(3):
    constraints.append(X[i, i] == 1)

prob = cp.Problem(objective, constraints)
prob.solve()

X_opt = X.value
print(f"\nNearest correlation matrix:\n{np.round(X_opt, 4)}")
print(f"Eigenvalues: {np.linalg.eigvalsh(X_opt)}")
print(f"Frobenius distance: {np.linalg.norm(X_opt - C):.4f}")
```

### 6.3 MAX-CUT Relaxation

The classic MAX-CUT problem in combinatorics can be approximated using an SDP relaxation (Goemans-Williamson algorithm, which achieves a 0.878 approximation ratio):

```python
# MAX-CUT SDP relaxation for a small graph
# Graph with 4 nodes
W = np.array([[0, 1, 0, 1],
              [1, 0, 1, 0],
              [0, 1, 0, 1],
              [1, 0, 1, 0]])  # Cycle graph C4

n = W.shape[0]
L = np.diag(W.sum(axis=1)) - W  # Laplacian matrix

# SDP relaxation: max (1/4) tr(LX) s.t. diag(X)=1, X PSD
X = cp.Variable((n, n), symmetric=True)
objective = cp.Maximize(0.25 * cp.trace(L @ X))
constraints = [X >> 0]
for i in range(n):
    constraints.append(X[i, i] == 1)

prob = cp.Problem(objective, constraints)
prob.solve()

print(f"SDP optimal value (upper bound on MAX-CUT): {prob.value:.4f}")
print(f"SDP solution X:\n{np.round(X.value, 3)}")

# Round the SDP solution to get an actual cut
# Use random hyperplane rounding
from scipy.linalg import sqrtm
V = sqrtm(X.value).real
r = np.random.randn(n)
cut = np.sign(V @ r)
cut_value = 0.25 * sum(W[i, j] * (1 - cut[i] * cut[j])
                        for i in range(n) for j in range(i+1, n))
print(f"Rounded cut: {cut}")
print(f"Cut value: {cut_value:.0f}")
```

---

## 7. Schur Complement and Block Positive Definiteness

### 7.1 Schur Complement

For a block matrix:

$$M = \begin{bmatrix} A & B \\ B^T & C \end{bmatrix}$$

where $A$ is invertible, the **Schur complement** of $A$ in $M$ is:

$$S = C - B^T A^{-1} B$$

**Key result**: $M \succ 0$ if and only if $A \succ 0$ and $S \succ 0$.

```python
# Schur complement example
A = np.array([[4, 1],
              [1, 3]])
B = np.array([[1, 0],
              [0, 1]])
C = np.array([[5, 1],
              [1, 4]])

# Form the block matrix
M = np.block([[A, B],
              [B.T, C]])

print(f"Block matrix M:\n{M}")
print(f"M eigenvalues: {np.linalg.eigvalsh(M)}")
print(f"M is PD? {np.all(np.linalg.eigvalsh(M) > 0)}")

# Check via Schur complement
A_inv = np.linalg.inv(A)
S = C - B.T @ A_inv @ B  # Schur complement

print(f"\nA eigenvalues: {np.linalg.eigvalsh(A)}")
print(f"A is PD? {np.all(np.linalg.eigvalsh(A) > 0)}")

print(f"\nSchur complement S = C - B^T A^(-1) B:")
print(f"{S}")
print(f"S eigenvalues: {np.linalg.eigvalsh(S)}")
print(f"S is PD? {np.all(np.linalg.eigvalsh(S) > 0)}")
print(f"\nConclusion: M is PD (both A and S are PD)")
```

---

## Exercises

### Exercise 1: Testing Positive Definiteness

For each of the following matrices, determine whether it is positive definite, positive semidefinite, negative definite, or indefinite. Use at least two different tests for each matrix.

$$A = \begin{bmatrix} 5 & 2 \\ 2 & 1 \end{bmatrix}, \quad
B = \begin{bmatrix} 1 & -1 & 0 \\ -1 & 3 & -1 \\ 0 & -1 & 2 \end{bmatrix}, \quad
C = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 5 & 7 \\ 3 & 7 & 10 \end{bmatrix}$$

### Exercise 2: Cholesky Decomposition by Hand

Compute the Cholesky decomposition of the following matrix by hand (derive each entry of $L$ step by step), then verify using `np.linalg.cholesky`:

$$A = \begin{bmatrix} 4 & 2 & -2 \\ 2 & 10 & 2 \\ -2 & 2 & 5 \end{bmatrix}$$

### Exercise 3: Hessian Analysis

For the function $f(x, y) = x^3 + y^3 - 3xy$, compute the Hessian at the critical points and classify each as a local minimum, local maximum, or saddle point.

### Exercise 4: Nearest Positive Definite Matrix

Write a function that takes any symmetric matrix and returns the nearest positive semidefinite matrix (in the Frobenius norm). Hint: compute the eigendecomposition and set negative eigenvalues to zero.

### Exercise 5: Covariance and Sampling

Given a target covariance matrix $\Sigma = \begin{bmatrix} 3 & 1 & 0.5 \\ 1 & 2 & 0.8 \\ 0.5 & 0.8 & 1.5 \end{bmatrix}$:

1. Verify $\Sigma$ is positive definite
2. Compute its Cholesky decomposition
3. Generate 10,000 samples from $\mathcal{N}(\mathbf{0}, \Sigma)$ using the Cholesky factor
4. Compare the empirical covariance with $\Sigma$

---

[Previous: Lesson 10](./10_Matrix_Norms_and_Condition_Numbers.md) | [Overview](./00_Overview.md) | [Next: Lesson 12](./12_Sparse_Matrices.md)

**License**: CC BY-NC 4.0
