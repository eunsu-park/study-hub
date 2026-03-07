# Lesson 6: Eigenvalues and Eigenvectors

## Learning Objectives

- Define eigenvalues and eigenvectors and understand their geometric meaning
- Compute eigenvalues by finding roots of the characteristic polynomial
- Diagonalize a matrix and understand when diagonalization is possible
- State and apply the spectral theorem for symmetric matrices
- Implement the power method for finding the dominant eigenvalue
- Recognize applications of eigenvalues in stability analysis, Markov chains, and PCA

---

## 1. Definition and Geometric Meaning

### 1.1 What Are Eigenvalues and Eigenvectors?

For a square matrix $A \in \mathbb{R}^{n \times n}$, a non-zero vector $\mathbf{v}$ is an **eigenvector** of $A$ if:

$$A\mathbf{v} = \lambda\mathbf{v}$$

The scalar $\lambda$ is the corresponding **eigenvalue**.

**Geometric meaning**: An eigenvector is a direction that the transformation $A$ merely stretches (or flips), without rotating. The eigenvalue tells us the stretching factor.

- $|\lambda| > 1$: stretching
- $|\lambda| < 1$: compression
- $\lambda < 0$: direction reversal
- $\lambda = 0$: collapse (eigenvector mapped to zero)

```python
import numpy as np
import matplotlib.pyplot as plt

A = np.array([[2, 1],
              [1, 2]], dtype=float)

eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors (columns):\n{eigenvectors}")

# Verify: A @ v = lambda * v
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    lhs = A @ v
    rhs = lam * v
    print(f"\nlambda_{i} = {lam:.4f}")
    print(f"  A @ v = {lhs}")
    print(f"  lambda * v = {rhs}")
    print(f"  Match? {np.allclose(lhs, rhs)}")
```

### 1.2 Visualizing Eigenvectors

```python
fig, ax = plt.subplots(figsize=(8, 8))

# Draw transformed and original vectors along many directions
theta_vals = np.linspace(0, 2 * np.pi, 60)
for t in theta_vals:
    v = np.array([np.cos(t), np.sin(t)])
    Av = A @ v
    ax.arrow(0, 0, v[0], v[1], head_width=0.03, color='blue', alpha=0.2)
    ax.arrow(0, 0, Av[0], Av[1], head_width=0.03, color='red', alpha=0.2)

# Highlight eigenvectors
for i in range(2):
    v = eigenvectors[:, i]
    Av = A @ v
    ax.arrow(0, 0, v[0], v[1], head_width=0.05, color='blue', linewidth=2,
             label=f'v{i+1}' if i == 0 else f'v{i+1}')
    ax.arrow(0, 0, Av[0], Av[1], head_width=0.05, color='red', linewidth=2,
             label=f'Av{i+1} (lambda={eigenvalues[i]:.1f})' if i == 0 else f'Av{i+1}')

ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
ax.legend()
ax.set_title('Blue = original, Red = transformed\nEigenvectors stay on their line')
plt.show()
```

---

## 2. Characteristic Polynomial

### 2.1 Derivation

Rearranging $A\mathbf{v} = \lambda\mathbf{v}$:

$$(A - \lambda I)\mathbf{v} = \mathbf{0}$$

For a non-zero solution $\mathbf{v}$ to exist, the matrix $(A - \lambda I)$ must be singular:

$$\det(A - \lambda I) = 0$$

This is the **characteristic equation**. The left-hand side is a polynomial of degree $n$ in $\lambda$, called the **characteristic polynomial**:

$$p(\lambda) = \det(A - \lambda I) = (-1)^n \lambda^n + \cdots$$

### 2.2 Example: 2x2 Matrix

For $A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$:

$$p(\lambda) = \lambda^2 - (a+d)\lambda + (ad - bc) = \lambda^2 - \mathrm{tr}(A)\lambda + \det(A)$$

```python
import sympy

# Symbolic computation of characteristic polynomial
A_sym = sympy.Matrix([[2, 1],
                       [1, 2]])

lam = sympy.Symbol('lambda')
char_poly = (A_sym - lam * sympy.eye(2)).det()
print(f"Characteristic polynomial: {sympy.expand(char_poly)}")

# Find eigenvalues symbolically
eigenvals = sympy.solve(char_poly, lam)
print(f"Eigenvalues: {eigenvals}")
```

### 2.3 Properties

For an $n \times n$ matrix with eigenvalues $\lambda_1, \ldots, \lambda_n$ (counted with multiplicity):

| Property | Formula |
|----------|---------|
| Sum of eigenvalues | $\lambda_1 + \cdots + \lambda_n = \mathrm{tr}(A)$ |
| Product of eigenvalues | $\lambda_1 \cdots \lambda_n = \det(A)$ |
| Eigenvalues of $A^k$ | $\lambda_1^k, \ldots, \lambda_n^k$ |
| Eigenvalues of $A^{-1}$ | $1/\lambda_1, \ldots, 1/\lambda_n$ |
| Eigenvalues of $A + cI$ | $\lambda_1 + c, \ldots, \lambda_n + c$ |

```python
A = np.array([[3, 1, 0],
              [0, 2, 1],
              [0, 0, 1]], dtype=float)

eigenvalues = np.linalg.eigvals(A)
print(f"Eigenvalues: {eigenvalues}")
print(f"Sum = {np.sum(eigenvalues):.4f}, tr(A) = {np.trace(A):.4f}")
print(f"Product = {np.prod(eigenvalues):.4f}, det(A) = {np.linalg.det(A):.4f}")

# Eigenvalues of A^2
eig_A2 = np.linalg.eigvals(A @ A)
print(f"\nEigenvalues of A^2: {np.sort(eig_A2)}")
print(f"lambda^2:           {np.sort(eigenvalues**2)}")
```

---

## 3. Eigendecomposition (Diagonalization)

### 3.1 When Is Diagonalization Possible?

A matrix $A$ is **diagonalizable** if there exists an invertible matrix $P$ and a diagonal matrix $\Lambda$ such that:

$$A = P \Lambda P^{-1}$$

where the columns of $P$ are eigenvectors and $\Lambda = \mathrm{diag}(\lambda_1, \ldots, \lambda_n)$.

**Sufficient conditions**:
- $A$ has $n$ distinct eigenvalues (always diagonalizable)
- $A$ is symmetric (always diagonalizable, even with repeated eigenvalues)

**Not diagonalizable**: when the geometric multiplicity of some eigenvalue is less than its algebraic multiplicity (e.g., defective matrices).

### 3.2 Computing the Eigendecomposition

```python
A = np.array([[4, 1],
              [2, 3]], dtype=float)

eigenvalues, P = np.linalg.eig(A)
Lambda = np.diag(eigenvalues)

print(f"P (eigenvectors):\n{P}")
print(f"Lambda (eigenvalues):\n{Lambda}")

# Verify: A = P Lambda P^{-1}
A_reconstructed = P @ Lambda @ np.linalg.inv(P)
print(f"\nP @ Lambda @ P^(-1):\n{A_reconstructed}")
print(f"Matches A? {np.allclose(A, A_reconstructed)}")
```

### 3.3 Powers of Diagonalizable Matrices

If $A = P\Lambda P^{-1}$, then:

$$A^k = P \Lambda^k P^{-1}$$

This makes computing high powers of matrices efficient.

```python
# Compute A^10 efficiently via diagonalization
k = 10

# Direct computation
A_power_direct = np.linalg.matrix_power(A, k)

# Via eigendecomposition
A_power_eig = P @ np.diag(eigenvalues**k) @ np.linalg.inv(P)

print(f"A^{k} (direct):\n{np.round(A_power_direct, 4)}")
print(f"A^{k} (eigen):\n{np.round(A_power_eig, 4)}")
print(f"Match? {np.allclose(A_power_direct, A_power_eig)}")
```

---

## 4. Spectral Theorem

### 4.1 Statement

For a **real symmetric** matrix $A = A^T$:

1. All eigenvalues are **real**
2. Eigenvectors corresponding to distinct eigenvalues are **orthogonal**
3. $A$ can be diagonalized by an **orthogonal** matrix $Q$ (i.e., $Q^{-1} = Q^T$):

$$A = Q \Lambda Q^T$$

This is called the **spectral decomposition** (or eigendecomposition of a symmetric matrix).

### 4.2 Spectral Decomposition as a Sum of Outer Products

$$A = \sum_{i=1}^n \lambda_i \mathbf{q}_i \mathbf{q}_i^T$$

Each $\mathbf{q}_i \mathbf{q}_i^T$ is a rank-1 projection matrix onto the eigenspace.

```python
# Symmetric matrix
A = np.array([[4, 2, 0],
              [2, 5, 3],
              [0, 3, 6]], dtype=float)

print(f"Symmetric? {np.allclose(A, A.T)}")

eigenvalues, Q = np.linalg.eigh(A)  # eigh for symmetric matrices
print(f"Eigenvalues: {eigenvalues}")
print(f"All real? {np.all(np.isreal(eigenvalues))}")

# Verify orthogonality
print(f"Q^T @ Q:\n{np.round(Q.T @ Q, 10)}")

# Verify spectral decomposition
A_reconstructed = Q @ np.diag(eigenvalues) @ Q.T
print(f"\nQ Lambda Q^T:\n{np.round(A_reconstructed, 10)}")
print(f"Matches A? {np.allclose(A, A_reconstructed)}")

# Outer product sum
A_outer = sum(eigenvalues[i] * np.outer(Q[:, i], Q[:, i]) for i in range(3))
print(f"Outer product sum matches? {np.allclose(A, A_outer)}")
```

### 4.3 `eig` vs `eigh`

| Function | Use for | Eigenvalues | Eigenvectors |
|----------|---------|-------------|--------------|
| `np.linalg.eig` | General matrices | May be complex | May not be orthogonal |
| `np.linalg.eigh` | Symmetric/Hermitian | Always real, sorted | Always orthonormal |

Always use `eigh` for symmetric matrices -- it is faster and more numerically stable.

---

## 5. Power Method

### 5.1 Algorithm

The **power method** is an iterative algorithm that finds the **dominant eigenvalue** (largest in absolute value) and its eigenvector.

Given an initial vector $\mathbf{v}_0$, iterate:

$$\mathbf{v}_{k+1} = \frac{A \mathbf{v}_k}{\|A \mathbf{v}_k\|}$$

The vectors converge to the eigenvector corresponding to the dominant eigenvalue.

### 5.2 Implementation

```python
def power_method(A, num_iterations=100, tol=1e-10):
    """Find the dominant eigenvalue and eigenvector using the power method."""
    n = A.shape[0]
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)

    eigenvalue_estimates = []

    for i in range(num_iterations):
        Av = A @ v
        eigenvalue = np.dot(v, Av)  # Rayleigh quotient
        eigenvalue_estimates.append(eigenvalue)

        v_new = Av / np.linalg.norm(Av)

        if np.linalg.norm(v_new - v) < tol:
            print(f"Converged after {i+1} iterations")
            break

        v = v_new

    return eigenvalue, v, eigenvalue_estimates

# Test
A = np.array([[4, 1, 0],
              [1, 3, 1],
              [0, 1, 2]], dtype=float)

lam, v, history = power_method(A)
print(f"Dominant eigenvalue: {lam:.6f}")
print(f"Eigenvector: {v}")

# Compare with NumPy
eigenvalues_np = np.linalg.eigvals(A)
print(f"\nAll eigenvalues (NumPy): {np.sort(np.abs(eigenvalues_np))[::-1]}")

# Plot convergence
plt.figure(figsize=(8, 4))
plt.plot(history, 'b.-')
plt.axhline(y=max(np.abs(eigenvalues_np)), color='r', linestyle='--', label='True dominant eigenvalue')
plt.xlabel('Iteration')
plt.ylabel('Eigenvalue estimate')
plt.title('Power Method Convergence')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 5.3 Inverse Power Method

To find the **smallest** eigenvalue, apply the power method to $A^{-1}$:

```python
def inverse_power_method(A, num_iterations=100, tol=1e-10):
    """Find the smallest eigenvalue using the inverse power method."""
    n = A.shape[0]
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)

    lu_piv = linalg.lu_factor(A)  # Factor once for efficiency

    for i in range(num_iterations):
        from scipy import linalg as sp_linalg
        w = sp_linalg.lu_solve(lu_piv, v)
        eigenvalue = np.dot(v, w)
        v_new = w / np.linalg.norm(w)

        if np.linalg.norm(v_new - v) < tol:
            print(f"Converged after {i+1} iterations")
            break
        v = v_new

    return 1.0 / eigenvalue, v

from scipy import linalg

lam_min, v_min = inverse_power_method(A)
print(f"Smallest eigenvalue: {lam_min:.6f}")
print(f"All eigenvalues: {np.sort(eigenvalues_np)}")
```

---

## 6. Applications

### 6.1 Stability Analysis

A discrete-time system $\mathbf{x}_{k+1} = A\mathbf{x}_k$ is **stable** if all eigenvalues satisfy $|\lambda_i| < 1$.

```python
# Stable system
A_stable = np.array([[0.5, 0.1],
                      [0.1, 0.5]])
eigs_stable = np.linalg.eigvals(A_stable)
print(f"Stable system eigenvalues: {eigs_stable}")
print(f"All |lambda| < 1? {np.all(np.abs(eigs_stable) < 1)}")

# Simulate
x = np.array([1.0, 0.5])
trajectory = [x.copy()]
for _ in range(50):
    x = A_stable @ x
    trajectory.append(x.copy())
trajectory = np.array(trajectory)

plt.figure(figsize=(8, 4))
plt.plot(trajectory[:, 0], label='x1')
plt.plot(trajectory[:, 1], label='x2')
plt.xlabel('Time step'); plt.ylabel('Value')
plt.title('Stable System (eigenvalues inside unit circle)')
plt.legend(); plt.grid(True, alpha=0.3)
plt.show()
```

### 6.2 Markov Chains

A **Markov chain** with transition matrix $P$ has a **stationary distribution** $\boldsymbol{\pi}$ satisfying $P^T \boldsymbol{\pi} = \boldsymbol{\pi}$. This is the eigenvector of $P^T$ with eigenvalue $\lambda = 1$.

```python
# Weather Markov chain: Sunny, Cloudy, Rainy
P = np.array([[0.7, 0.2, 0.1],   # Sunny -> ...
              [0.3, 0.4, 0.3],   # Cloudy -> ...
              [0.2, 0.3, 0.5]])  # Rainy -> ...

# Find stationary distribution
eigenvalues, eigenvectors = np.linalg.eig(P.T)
print(f"Eigenvalues of P^T: {eigenvalues}")

# Eigenvector with eigenvalue 1
idx = np.argmin(np.abs(eigenvalues - 1.0))
pi = np.abs(eigenvectors[:, idx])
pi = pi / np.sum(pi)  # normalize

print(f"Stationary distribution: {pi}")
print(f"Sunny: {pi[0]:.3f}, Cloudy: {pi[1]:.3f}, Rainy: {pi[2]:.3f}")

# Verify: P^T pi = pi
print(f"P^T @ pi = {P.T @ pi}")
```

### 6.3 Google PageRank

PageRank is fundamentally an eigenvalue problem: the ranking vector is the dominant eigenvector of the modified web link matrix.

```python
# Simplified PageRank example
# 4 web pages with link structure
# Page 0 links to 1, 2
# Page 1 links to 2
# Page 2 links to 0
# Page 3 links to 0, 1, 2

n_pages = 4
L = np.zeros((n_pages, n_pages))
L[1, 0] = 1; L[2, 0] = 1     # page 0 links to 1, 2
L[2, 1] = 1                    # page 1 links to 2
L[0, 2] = 1                    # page 2 links to 0
L[0, 3] = 1; L[1, 3] = 1; L[2, 3] = 1  # page 3 links to 0, 1, 2

# Column-normalize (each column sums to 1)
col_sums = L.sum(axis=0)
col_sums[col_sums == 0] = 1  # avoid division by zero
M = L / col_sums

# Damping factor
d = 0.85
G = d * M + (1 - d) / n_pages * np.ones((n_pages, n_pages))

# Find dominant eigenvector
eigenvalues, eigenvectors = np.linalg.eig(G)
idx = np.argmax(np.abs(eigenvalues))
rank = np.abs(eigenvectors[:, idx])
rank = rank / np.sum(rank)

print(f"PageRank: {rank}")
for i, r in enumerate(rank):
    print(f"  Page {i}: {r:.4f}")
```

---

## 7. Complex Eigenvalues

Non-symmetric real matrices can have **complex eigenvalues**, which always come in conjugate pairs: $\lambda = a \pm bi$.

Complex eigenvalues correspond to **rotational** behavior in the transformation.

```python
# Rotation matrix has complex eigenvalues
theta = np.pi / 4
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

eigs = np.linalg.eigvals(R)
print(f"Rotation eigenvalues: {eigs}")
print(f"Magnitude: {np.abs(eigs)}")  # both = 1

# Non-symmetric example with complex eigenvalues
A = np.array([[0, -1],
              [1,  0]], dtype=float)  # 90-degree rotation
eigs = np.linalg.eigvals(A)
print(f"\n90-degree rotation eigenvalues: {eigs}")
```

---

## 8. Summary

| Concept | Description |
|---------|-------------|
| Eigenvalue equation | $A\mathbf{v} = \lambda\mathbf{v}$ |
| Characteristic polynomial | $\det(A - \lambda I) = 0$ |
| Diagonalization | $A = P\Lambda P^{-1}$ |
| Spectral theorem | Symmetric $A = Q\Lambda Q^T$ with orthogonal $Q$ |
| Power method | Iterative dominant eigenvalue finder |
| Trace = sum of eigenvalues | $\mathrm{tr}(A) = \sum \lambda_i$ |
| Determinant = product | $\det(A) = \prod \lambda_i$ |
| NumPy | `np.linalg.eig(A)` or `np.linalg.eigh(A)` |

---

## Exercises

### Exercise 1: Manual Eigenvalue Computation

Find the eigenvalues and eigenvectors of $A = \begin{bmatrix} 3 & 1 \\ 0 & 2 \end{bmatrix}$ by hand (characteristic polynomial). Verify with NumPy.

### Exercise 2: Diagonalization

Diagonalize $A = \begin{bmatrix} 5 & 4 \\ 1 & 2 \end{bmatrix}$:
1. Find eigenvalues and eigenvectors.
2. Construct $P$ and $\Lambda$.
3. Verify $A = P\Lambda P^{-1}$.
4. Use the decomposition to compute $A^{100}$.

### Exercise 3: Spectral Theorem

For the symmetric matrix $A = \begin{bmatrix} 2 & 1 & 0 \\ 1 & 3 & 1 \\ 0 & 1 & 2 \end{bmatrix}$:
1. Find its spectral decomposition $A = Q\Lambda Q^T$.
2. Verify that $Q$ is orthogonal.
3. Express $A$ as a sum of rank-1 outer products.

### Exercise 4: Power Method

Implement the power method with shift: to find the eigenvalue closest to a target $\sigma$, apply the inverse power method to $(A - \sigma I)$. Test it on a $4 \times 4$ matrix.

### Exercise 5: Markov Chain

A frog jumps between 3 lily pads with transition probabilities $P = \begin{bmatrix} 0.5 & 0.25 & 0.25 \\ 0.25 & 0.5 & 0.25 \\ 0.25 & 0.25 & 0.5 \end{bmatrix}$. Find the stationary distribution. Simulate 10,000 steps and compare the empirical frequencies with the theoretical distribution.

---

[<< Previous: Lesson 5 - Linear Transformations](05_Linear_Transformations.md) | [Overview](00_Overview.md) | [Next: Lesson 7 - Singular Value Decomposition >>](07_Singular_Value_Decomposition.md)

**License**: CC BY-NC 4.0
