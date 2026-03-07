# Lesson 10: Matrix Decompositions

## Learning Objectives

- Compute and apply the Cholesky decomposition for symmetric positive definite matrices
- Understand and compute the LDL^T decomposition as a variant of Cholesky
- Describe the Schur decomposition and its relationship to eigenvalues
- Explain the polar decomposition and its geometric meaning
- Compare decompositions by cost, stability, applicability, and use cases

---

## 1. Overview of Decompositions

Matrix decompositions (or factorizations) are the workhorses of numerical linear algebra. Each decomposition reveals a different structural property and is suited to particular tasks.

| Decomposition | Form | Requirements | Primary Use |
|--------------|------|-------------|-------------|
| LU | $PA = LU$ | Square (general) | Solving linear systems |
| QR | $A = QR$ | Full column rank | Least squares, eigenvalues |
| Eigendecomposition | $A = P\Lambda P^{-1}$ | Square, diagonalizable | Spectral analysis |
| SVD | $A = U\Sigma V^T$ | Any matrix | Rank, pseudoinverse, compression |
| Cholesky | $A = LL^T$ | Symmetric positive definite | Fast SPD solves, sampling |
| LDL^T | $A = LDL^T$ | Symmetric | SPD check, indefinite systems |
| Schur | $A = QTQ^T$ | Square | Eigenvalues, matrix functions |
| Polar | $A = UP$ | Any matrix | Rotation extraction |

---

## 2. Cholesky Decomposition

### 2.1 Definition

For a **symmetric positive definite** (SPD) matrix $A$, the **Cholesky decomposition** is:

$$A = LL^T$$

where $L$ is a **lower triangular** matrix with **positive diagonal entries**. This is unique.

### 2.2 Why Cholesky?

- **Twice as fast** as LU decomposition (about $\frac{1}{3}n^3$ vs $\frac{2}{3}n^3$ flops)
- **Numerically stable** without pivoting (for SPD matrices)
- **SPD test**: $A$ is SPD if and only if the Cholesky decomposition succeeds

### 2.3 Algorithm

```python
import numpy as np
from scipy import linalg

def cholesky_manual(A):
    """Compute the Cholesky decomposition A = L L^T."""
    n = A.shape[0]
    L = np.zeros_like(A, dtype=float)

    for j in range(n):
        # Diagonal entry
        val = A[j, j] - np.sum(L[j, :j]**2)
        if val <= 0:
            raise ValueError("Matrix is not positive definite")
        L[j, j] = np.sqrt(val)

        # Below-diagonal entries
        for i in range(j + 1, n):
            L[i, j] = (A[i, j] - np.sum(L[i, :j] * L[j, :j])) / L[j, j]

    return L

# Create an SPD matrix
np.random.seed(42)
B = np.random.randn(4, 4)
A = B.T @ B + 0.1 * np.eye(4)  # guarantee SPD

print(f"A is symmetric? {np.allclose(A, A.T)}")
print(f"Eigenvalues: {np.linalg.eigvalsh(A)}")  # all positive

# Manual implementation
L_manual = cholesky_manual(A)
print(f"\nL (manual):\n{np.round(L_manual, 4)}")
print(f"L L^T == A? {np.allclose(L_manual @ L_manual.T, A)}")

# NumPy implementation
L_numpy = np.linalg.cholesky(A)
print(f"\nL (numpy):\n{np.round(L_numpy, 4)}")
print(f"Match? {np.allclose(L_manual, L_numpy)}")
```

### 2.4 Solving Systems with Cholesky

For $A\mathbf{x} = \mathbf{b}$ where $A$ is SPD:

1. Compute $A = LL^T$
2. Solve $L\mathbf{y} = \mathbf{b}$ (forward substitution)
3. Solve $L^T\mathbf{x} = \mathbf{y}$ (back substitution)

```python
b = np.array([1, 2, 3, 4], dtype=float)

# Solve using Cholesky
L = np.linalg.cholesky(A)
y = linalg.solve_triangular(L, b, lower=True)
x = linalg.solve_triangular(L.T, y, lower=False)

print(f"Solution: {x}")
print(f"Verification: Ax = {A @ x}")
print(f"Match b? {np.allclose(A @ x, b)}")

# Compare with direct solve
x_direct = np.linalg.solve(A, b)
print(f"Direct solve: {x_direct}")
print(f"Match? {np.allclose(x, x_direct)}")
```

### 2.5 Application: Sampling from Multivariate Normal

To sample $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$:

1. Compute $L$ such that $\Sigma = LL^T$
2. Sample $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, I)$
3. Set $\mathbf{x} = \boldsymbol{\mu} + L\mathbf{z}$

```python
import matplotlib.pyplot as plt

# Desired distribution parameters
mu = np.array([2, 3])
Sigma = np.array([[2, 1.5],
                   [1.5, 3]])

# Cholesky decomposition
L = np.linalg.cholesky(Sigma)

# Sample
n_samples = 1000
Z = np.random.randn(n_samples, 2)
X = Z @ L.T + mu  # equivalent to mu + L @ z for each sample

print(f"Sample mean: {X.mean(axis=0)} (expected: {mu})")
print(f"Sample covariance:\n{np.round(np.cov(X.T), 3)} (expected:\n{Sigma})")

plt.figure(figsize=(7, 7))
plt.scatter(X[:, 0], X[:, 1], alpha=0.3, s=5)
plt.plot(mu[0], mu[1], 'r+', markersize=15, markeredgewidth=3, label='True mean')
plt.xlabel('x1'); plt.ylabel('x2')
plt.title('Samples from Multivariate Normal via Cholesky')
plt.legend(); plt.grid(True, alpha=0.3); plt.axis('equal')
plt.show()
```

---

## 3. LDL^T Decomposition

### 3.1 Definition

For a **symmetric** matrix $A$ (not necessarily positive definite):

$$A = LDL^T$$

where $L$ is unit lower triangular (ones on the diagonal) and $D$ is diagonal.

### 3.2 Relationship to Cholesky

If $A$ is SPD, then $D$ has all positive entries and:

$$A = LDL^T = L\sqrt{D}\sqrt{D}L^T = (L\sqrt{D})(L\sqrt{D})^T$$

So the Cholesky factor is $\tilde{L} = L\sqrt{D}$.

### 3.3 Advantages

- Works for **indefinite** symmetric matrices (unlike Cholesky)
- Avoids square roots (unlike Cholesky)
- Can detect whether $A$ is positive definite by checking the signs of $D$

```python
def ldlt_decomposition(A):
    """Compute A = L D L^T for symmetric A."""
    n = A.shape[0]
    L = np.eye(n)
    D = np.zeros(n)

    for j in range(n):
        D[j] = A[j, j] - sum(L[j, k]**2 * D[k] for k in range(j))

        for i in range(j + 1, n):
            L[i, j] = (A[i, j] - sum(L[i, k] * L[j, k] * D[k] for k in range(j))) / D[j]

    return L, D

# SPD matrix
A_spd = np.array([[4, 2, 1],
                   [2, 5, 3],
                   [1, 3, 6]], dtype=float)

L, D = ldlt_decomposition(A_spd)
print(f"L:\n{np.round(L, 4)}")
print(f"D: {D}")
print(f"L D L^T:\n{np.round(L @ np.diag(D) @ L.T, 10)}")
print(f"Matches A? {np.allclose(L @ np.diag(D) @ L.T, A_spd)}")
print(f"All D > 0 (SPD)? {np.all(D > 0)}")

# Indefinite symmetric matrix
A_indef = np.array([[ 2,  1, -1],
                     [ 1, -3,  2],
                     [-1,  2,  1]], dtype=float)
A_indef = (A_indef + A_indef.T) / 2  # ensure symmetric

L_i, D_i = ldlt_decomposition(A_indef)
print(f"\nIndefinite matrix D: {D_i}")
print(f"Positive definite? {np.all(D_i > 0)}")
print(f"Reconstruction matches? {np.allclose(L_i @ np.diag(D_i) @ L_i.T, A_indef)}")
```

### 3.4 SciPy's LDL^T

```python
# SciPy provides LDL^T with pivoting
from scipy.linalg import ldl

L_sp, D_sp, perm = ldl(A_spd)
print(f"SciPy L:\n{np.round(L_sp, 4)}")
print(f"SciPy D:\n{np.round(D_sp, 4)}")

# Reconstruction (perm is a permutation)
print(f"Matches A? {np.allclose(L_sp @ D_sp @ L_sp.T, A_spd)}")
```

---

## 4. Schur Decomposition

### 4.1 Definition

Every square matrix $A \in \mathbb{R}^{n \times n}$ has a **Schur decomposition**:

$$A = QTQ^T \quad \text{(real Schur form)}$$

where $Q$ is orthogonal and $T$ is **quasi-upper triangular** (upper triangular with possible $2 \times 2$ diagonal blocks for complex eigenvalue pairs).

If we allow complex numbers:

$$A = QTQ^* \quad \text{(complex Schur form)}$$

where $T$ is upper triangular and the diagonal entries are the eigenvalues of $A$.

### 4.2 Why Schur?

- **Always exists** for any square matrix (unlike diagonalization)
- The eigenvalues appear on the diagonal of $T$
- Numerically stable (uses orthogonal transformations)
- Foundation for computing eigenvalues, matrix functions, and matrix equations

```python
from scipy.linalg import schur

A = np.array([[1, 2, 3],
              [0, 4, 5],
              [0, 0, 6]], dtype=float)

# Real Schur decomposition
T, Q = schur(A, output='real')
print(f"T (quasi-upper triangular):\n{np.round(T, 4)}")
print(f"Q (orthogonal):\n{np.round(Q, 4)}")
print(f"Q^T Q = I? {np.allclose(Q.T @ Q, np.eye(3))}")
print(f"Q T Q^T = A? {np.allclose(Q @ T @ Q.T, A)}")

# Eigenvalues on the diagonal
print(f"Diagonal of T: {np.diag(T)}")
print(f"Eigenvalues:    {np.linalg.eigvals(A)}")

# Complex Schur for a matrix with complex eigenvalues
B = np.array([[0, -1],
              [1,  0]], dtype=float)  # 90-degree rotation

T_c, Q_c = schur(B, output='complex')
print(f"\nComplex Schur T:\n{np.round(T_c, 4)}")
print(f"Eigenvalues: {np.diag(T_c)}")
```

### 4.3 Schur for Matrix Functions

The Schur decomposition is used to compute matrix functions (exponential, logarithm, square root):

$$f(A) = Q \, f(T) \, Q^T$$

Since $T$ is (quasi-)triangular, computing $f(T)$ is much simpler than computing $f(A)$ directly.

```python
from scipy.linalg import expm, sqrtm

A = np.array([[1, 0.5],
              [0.5, 2]], dtype=float)

# Matrix exponential via Schur
T, Q = schur(A)
# For a diagonal T, exp(T) is just exp of diagonal entries
# For triangular T, use Parlett's algorithm (what scipy does internally)
exp_A = expm(A)
print(f"exp(A) =\n{np.round(exp_A, 4)}")

# Matrix square root
sqrt_A = sqrtm(A)
print(f"\nsqrt(A) =\n{np.round(sqrt_A, 4)}")
print(f"sqrt(A)^2 = A? {np.allclose(sqrt_A @ sqrt_A, A)}")
```

---

## 5. Polar Decomposition

### 5.1 Definition

Any matrix $A \in \mathbb{R}^{m \times n}$ (with $m \ge n$) can be written as:

$$A = UP$$

where:
- $U \in \mathbb{R}^{m \times n}$ has orthonormal columns ($U^TU = I_n$)
- $P \in \mathbb{R}^{n \times n}$ is symmetric positive semi-definite

### 5.2 Geometric Interpretation

The polar decomposition separates a linear transformation into:
1. A **stretch** (by $P$) -- symmetric, changes shape but not orientation
2. A **rotation/reflection** (by $U$) -- orthogonal, preserves shape

This is analogous to polar coordinates $z = re^{i\theta}$ separating magnitude from phase.

### 5.3 Computing via SVD

Given $A = U_s \Sigma V_s^T$:

$$U = U_s V_s^T, \quad P = V_s \Sigma V_s^T$$

```python
A = np.array([[2, 1],
              [1, 3]], dtype=float)

# Compute via SVD
U_svd, s, Vt_svd = np.linalg.svd(A)

U_polar = U_svd @ Vt_svd
P_polar = Vt_svd.T @ np.diag(s) @ Vt_svd

print(f"U (orthogonal part):\n{np.round(U_polar, 4)}")
print(f"P (SPD part):\n{np.round(P_polar, 4)}")

# Verify
print(f"\nU @ P = A? {np.allclose(U_polar @ P_polar, A)}")
print(f"U^T U = I? {np.allclose(U_polar.T @ U_polar, np.eye(2))}")
print(f"P = P^T? {np.allclose(P_polar, P_polar.T)}")
print(f"P eigenvalues: {np.linalg.eigvalsh(P_polar)}")  # all >= 0

# SciPy polar decomposition
from scipy.linalg import polar
U_sp, P_sp = polar(A)
print(f"\nSciPy U:\n{np.round(U_sp, 4)}")
print(f"SciPy P:\n{np.round(P_sp, 4)}")
```

### 5.4 Visualizing the Polar Decomposition

```python
# Show how a transformation decomposes into stretch + rotation
theta = np.linspace(0, 2 * np.pi, 200)
circle = np.array([np.cos(theta), np.sin(theta)])

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Original unit circle
axes[0].plot(circle[0], circle[1], 'b-', linewidth=2)
axes[0].set_title('Unit circle')

# After P (stretch)
stretched = P_polar @ circle
axes[1].plot(stretched[0], stretched[1], 'g-', linewidth=2)
axes[1].set_title('After P (stretch)')

# After U rotation of stretched
final = U_polar @ stretched
axes[2].plot(final[0], final[1], 'r-', linewidth=2)
axes[2].set_title('After U (rotation) = A @ circle')

# Direct A @ circle (should match)
direct = A @ circle
axes[3].plot(direct[0], direct[1], 'r--', linewidth=2)
axes[3].set_title('A @ circle (direct)')

for ax in axes:
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.5); ax.axvline(0, color='k', linewidth=0.5)

plt.tight_layout()
plt.show()
```

---

## 6. Comparison of Decompositions

### 6.1 When to Use What

| Task | Best Decomposition | Why |
|------|-------------------|-----|
| Solve $A\mathbf{x} = \mathbf{b}$ (general) | LU | Standard approach |
| Solve $A\mathbf{x} = \mathbf{b}$ (SPD) | Cholesky | 2x faster, always stable |
| Least squares | QR or SVD | Numerically stable |
| Eigenvalues | Schur then extract | Numerically robust |
| Low-rank approximation | SVD | Optimal (Eckart-Young) |
| Detect positive definiteness | Cholesky or LDL^T | Fails if not SPD |
| Sampling (MVN) | Cholesky | $\Sigma = LL^T$, then $x = \mu + Lz$ |
| Extract rotation | Polar | $A = UP$ separates rotation from stretch |
| Matrix functions | Schur | $f(A) = Qf(T)Q^T$ |
| Pseudoinverse | SVD | $A^+ = V\Sigma^+ U^T$ |

### 6.2 Computational Cost

| Decomposition | Cost (flops) | Requirement |
|--------------|-------------|-------------|
| LU | $\frac{2}{3}n^3$ | Square |
| Cholesky | $\frac{1}{3}n^3$ | SPD |
| QR (Householder) | $\frac{2}{3}n^3$ | Any |
| SVD | $\sim 4n^3$ (full) | Any |
| Eigendecomposition | $\sim 9n^3$ | Square |
| Schur | $\sim 9n^3$ | Square |

```python
import time

sizes = [100, 200, 500, 1000]
results = {name: [] for name in ['LU', 'Cholesky', 'QR', 'SVD', 'Eig']}

for n in sizes:
    np.random.seed(42)
    B = np.random.randn(n, n)
    A = B.T @ B + np.eye(n)  # SPD

    # LU
    t0 = time.perf_counter()
    linalg.lu(A)
    results['LU'].append(time.perf_counter() - t0)

    # Cholesky
    t0 = time.perf_counter()
    np.linalg.cholesky(A)
    results['Cholesky'].append(time.perf_counter() - t0)

    # QR
    t0 = time.perf_counter()
    np.linalg.qr(A)
    results['QR'].append(time.perf_counter() - t0)

    # SVD
    t0 = time.perf_counter()
    np.linalg.svd(A)
    results['SVD'].append(time.perf_counter() - t0)

    # Eigendecomposition
    t0 = time.perf_counter()
    np.linalg.eigh(A)
    results['Eig'].append(time.perf_counter() - t0)

# Plot
plt.figure(figsize=(10, 6))
for name, times in results.items():
    plt.plot(sizes, times, 'o-', label=name, linewidth=2)
plt.xlabel('Matrix size n')
plt.ylabel('Time (seconds)')
plt.title('Decomposition Timing Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.show()
```

---

## 7. Stability and Numerical Considerations

### 7.1 Condition Number and Decomposition Choice

```python
# Create matrices with different condition numbers
cond_numbers = [1, 10, 1e4, 1e8, 1e12]

print(f"{'Condition':>12} {'LU error':>12} {'Cholesky err':>14} {'QR error':>12} {'SVD error':>12}")
print("-" * 64)

for kappa in cond_numbers:
    n = 50
    np.random.seed(42)

    # Create SPD matrix with desired condition number
    U, _ = np.linalg.qr(np.random.randn(n, n))
    s = np.logspace(0, -np.log10(kappa), n)
    A = U @ np.diag(s) @ U.T

    x_true = np.random.randn(n)
    b = A @ x_true

    # Solve with different methods
    x_lu = linalg.solve(A, b)
    L_chol = np.linalg.cholesky(A)
    y = linalg.solve_triangular(L_chol, b, lower=True)
    x_chol = linalg.solve_triangular(L_chol.T, y, lower=False)
    Q, R = np.linalg.qr(A)
    x_qr = linalg.solve_triangular(R, Q.T @ b)
    x_svd = np.linalg.lstsq(A, b, rcond=None)[0]

    err_lu = np.linalg.norm(x_lu - x_true) / np.linalg.norm(x_true)
    err_chol = np.linalg.norm(x_chol - x_true) / np.linalg.norm(x_true)
    err_qr = np.linalg.norm(x_qr - x_true) / np.linalg.norm(x_true)
    err_svd = np.linalg.norm(x_svd - x_true) / np.linalg.norm(x_true)

    print(f"{kappa:12.0e} {err_lu:12.2e} {err_chol:14.2e} {err_qr:12.2e} {err_svd:12.2e}")
```

### 7.2 Best Practices

1. **Know your matrix**: Use the structure (symmetry, positive definiteness, sparsity) to choose the right decomposition
2. **Avoid forming $A^TA$** when possible: use QR or SVD for least squares instead of normal equations
3. **Factor once, solve many times**: LU, Cholesky, and QR can be reused for multiple right-hand sides
4. **Check condition number** before solving: if $\kappa(A) > 10^{12}$, results may be unreliable

---

## 8. Application: Solving a Generalized Eigenvalue Problem

The generalized eigenvalue problem $A\mathbf{x} = \lambda B\mathbf{x}$ with SPD $B$ can be reduced to a standard problem using Cholesky:

1. Compute $B = LL^T$
2. Form $C = L^{-1} A L^{-T}$
3. Solve the standard eigenvalue problem $C\mathbf{y} = \lambda\mathbf{y}$
4. Recover $\mathbf{x} = L^{-T}\mathbf{y}$

```python
# Generalized eigenvalue problem
A = np.array([[5, 2], [2, 3]], dtype=float)
B = np.array([[2, 1], [1, 2]], dtype=float)  # SPD

# Via Cholesky reduction
L = np.linalg.cholesky(B)
L_inv = np.linalg.inv(L)
C = L_inv @ A @ L_inv.T

eigvals_gen, Y = np.linalg.eigh(C)
X = linalg.solve_triangular(L.T, Y, lower=False)

print(f"Generalized eigenvalues: {eigvals_gen}")

# Verify: A x = lambda B x
for i in range(2):
    lhs = A @ X[:, i]
    rhs = eigvals_gen[i] * B @ X[:, i]
    print(f"lambda={eigvals_gen[i]:.4f}: Ax={lhs}, lambda*Bx={rhs}, match={np.allclose(lhs, rhs)}")

# Compare with scipy
from scipy.linalg import eigh
eigvals_sp, eigvecs_sp = eigh(A, B)
print(f"\nSciPy generalized eigenvalues: {eigvals_sp}")
```

---

## 9. Decision Flowchart

To choose a decomposition, follow this logic:

```
Is A square?
  |
  |-- YES --> Is A symmetric?
  |             |
  |             |-- YES --> Is A positive definite?
  |             |             |
  |             |             |-- YES --> Cholesky (fastest)
  |             |             |-- NO  --> LDL^T or eigendecomposition
  |             |
  |             |-- NO  --> Need eigenvalues?
  |                           |
  |                           |-- YES --> Schur (most stable)
  |                           |-- NO  --> LU (for solving Ax = b)
  |
  |-- NO  --> Need least squares?
                |
                |-- YES --> QR (stable) or SVD (most robust)
                |-- NO  --> SVD (for rank, pseudoinverse, etc.)
```

---

## 10. Summary

| Decomposition | Form | Key Property | NumPy / SciPy |
|--------------|------|-------------|---------------|
| Cholesky | $A = LL^T$ | SPD, fast, unique | `np.linalg.cholesky` |
| LDL^T | $A = LDL^T$ | Symmetric, no sqrt | `scipy.linalg.ldl` |
| Schur | $A = QTQ^T$ | Always exists, eigenvalues on diag | `scipy.linalg.schur` |
| Polar | $A = UP$ | Rotation + stretch | `scipy.linalg.polar` |
| LU | $PA = LU$ | General solve | `scipy.linalg.lu` |
| QR | $A = QR$ | Least squares | `np.linalg.qr` |
| SVD | $A = U\Sigma V^T$ | Universal | `np.linalg.svd` |
| Eigen | $A = P\Lambda P^{-1}$ | Spectral analysis | `np.linalg.eig`, `eigh` |

---

## Exercises

### Exercise 1: Cholesky

1. Show that the Cholesky decomposition of a diagonal matrix with positive entries is just the matrix of square roots of those entries.
2. Compute the Cholesky decomposition of $A = \begin{bmatrix} 4 & 2 \\ 2 & 3 \end{bmatrix}$ by hand. Verify with NumPy.
3. Attempt the Cholesky decomposition of $\begin{bmatrix} 1 & 2 \\ 2 & 1 \end{bmatrix}$. What happens and why?

### Exercise 2: LDL^T for Definiteness Testing

Generate 100 random $5 \times 5$ symmetric matrices. For each, compute the LDL^T decomposition and classify the matrix as positive definite, positive semi-definite, indefinite, or negative definite based on the signs of the diagonal entries of $D$. Verify your classification using eigenvalues.

### Exercise 3: Schur Decomposition

For $A = \begin{bmatrix} 0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 2 \end{bmatrix}$:
1. Compute the real Schur decomposition.
2. Compute the complex Schur decomposition.
3. Identify the eigenvalues from the diagonal of $T$.

### Exercise 4: Polar Decomposition

For $A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$:
1. Compute the polar decomposition $A = UP$.
2. Verify that $U$ is orthogonal and $P$ is SPD.
3. Visualize how the unit circle transforms under $P$ alone, $U$ alone, and $A = UP$.

### Exercise 5: Decomposition Race

For an SPD matrix of size $n = 500$, measure the time to solve $A\mathbf{x} = \mathbf{b}$ using:
1. `np.linalg.solve` (LU internally)
2. Cholesky factorization + two triangular solves
3. QR decomposition
4. SVD + pseudoinverse

Compare the wall-clock times. Which is fastest? Solve the same system 100 times (with different $\mathbf{b}$) and measure the total time for each method. How does pre-factoring help?

---

[<< Previous: Lesson 9 - Orthogonality and Projections](09_Orthogonality_and_Projections.md) | [Overview](00_Overview.md) | [Next: Lesson 11 - Quadratic Forms and Definiteness >>](11_Quadratic_Forms_and_Definiteness.md)

**License**: CC BY-NC 4.0
