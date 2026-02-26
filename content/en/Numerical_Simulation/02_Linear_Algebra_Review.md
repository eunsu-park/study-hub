# Linear Algebra Review

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement fundamental matrix operations (multiplication, transpose, inverse) using NumPy and explain their computational properties.
2. Solve systems of linear equations using direct methods (LU decomposition, Gaussian elimination) and iterative methods with SciPy.
3. Compute eigenvalues and eigenvectors and apply them to physical simulation problems.
4. Apply matrix decompositions (LU, QR, SVD, Cholesky) and explain when each is appropriate.
5. Analyze the condition number of a matrix and its effect on the numerical stability of linear system solutions.

---

## Overview

Linear algebra plays a crucial role in numerical simulation. We will learn how to implement matrix operations, solving systems of linear equations, eigenvalue problems, and matrix decomposition using NumPy/SciPy.

**Why This Lesson Matters:** Almost every numerical method ultimately reduces to solving a linear system $Ax = b$. Finite difference discretization of PDEs produces sparse tridiagonal systems. Implicit time-stepping requires solving a system at each step. Eigenvalue analysis reveals stability properties and natural frequencies. The choice of linear algebra algorithm -- direct vs. iterative, dense vs. sparse -- often determines whether a simulation takes seconds or hours.

---

## 1. Basic Matrix Operations

### 1.1 Matrix Creation and Operations

```python
import numpy as np
from scipy import linalg

# Matrix creation
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 10]])

B = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])

# Basic operations
print("Matrix A:")
print(A)
print(f"\nTranspose: A.T =\n{A.T}")
print(f"\nMatrix multiplication: A @ B =\n{A @ B}")
print(f"\nElement-wise multiplication: A * B =\n{A * B}")
print(f"\nInverse: A⁻¹ =\n{np.linalg.inv(A)}")
print(f"\nDeterminant: det(A) = {np.linalg.det(A):.4f}")
print(f"\nTrace: tr(A) = {np.trace(A)}")
```

### 1.2 Special Matrix Creation

```python
n = 4

# Identity matrix
I = np.eye(n)

# Zero matrix
Z = np.zeros((n, n))

# Diagonal matrix
D = np.diag([1, 2, 3, 4])

# Tridiagonal matrix
diag_main = 2 * np.ones(n)
diag_off = -1 * np.ones(n - 1)
T = np.diag(diag_main) + np.diag(diag_off, k=1) + np.diag(diag_off, k=-1)

print("Tridiagonal matrix:")
print(T)

# Random matrix
np.random.seed(42)
R = np.random.randn(3, 3)
print(f"\nRandom matrix:\n{R}")
```

### 1.3 Matrix Norms

```python
A = np.array([[1, 2], [3, 4]])

# Various norms
print("Matrix norms:")
print(f"  1-norm (max column sum): {np.linalg.norm(A, 1)}")
print(f"  2-norm (spectral): {np.linalg.norm(A, 2)}")
print(f"  ∞-norm (max row sum): {np.linalg.norm(A, np.inf)}")
print(f"  Frobenius norm: {np.linalg.norm(A, 'fro')}")

# Vector norms
v = np.array([3, 4])
print(f"\nVector norms:")
print(f"  L1: {np.linalg.norm(v, 1)}")
print(f"  L2: {np.linalg.norm(v, 2)}")
print(f"  L∞: {np.linalg.norm(v, np.inf)}")
```

---

## 2. Solving Linear Systems

### 2.1 Direct Solution

```python
# Solve Ax = b
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])

# numpy.linalg.solve (recommended)
x = np.linalg.solve(A, b)
print(f"Solution: x = {x}")
print(f"Verification: Ax = {A @ x}")

# Using inverse (not recommended - slow and unstable)
x_inv = np.linalg.inv(A) @ b
print(f"Inverse method: x = {x_inv}")
```

### 2.2 Overdetermined/Underdetermined Systems

```python
# Overdetermined system (equations > unknowns) - least squares solution
A_over = np.array([[1, 1], [1, 2], [1, 3]])
b_over = np.array([1, 2, 2])

# Least squares solution
x_lstsq, residuals, rank, s = np.linalg.lstsq(A_over, b_over, rcond=None)
print(f"Least squares solution: {x_lstsq}")
print(f"Residuals: {residuals}")

# Underdetermined system (equations < unknowns) - minimum norm solution
A_under = np.array([[1, 2, 3]])
b_under = np.array([6])

# Using pseudoinverse
x_min_norm = np.linalg.pinv(A_under) @ b_under
print(f"\nMinimum norm solution: {x_min_norm}")
print(f"Norm: {np.linalg.norm(x_min_norm):.4f}")
```

---

## 3. Eigenvalues and Eigenvectors

Eigenvalues appear throughout numerical simulation: the stability of an ODE system depends on the eigenvalues of its Jacobian, the natural frequencies of a vibrating structure are eigenvalues of the mass-stiffness system, and the convergence rate of iterative solvers depends on the eigenvalue spectrum of the iteration matrix.

### 3.1 Eigenvalue Decomposition

```python
A = np.array([[4, -2],
              [1,  1]])

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalue decomposition:")
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")

# Verification: A @ v = λ @ v
for i in range(len(eigenvalues)):
    lam = eigenvalues[i]
    v = eigenvectors[:, i]
    print(f"\nλ_{i} = {lam:.4f}")
    print(f"  A @ v = {A @ v}")
    print(f"  λ * v = {lam * v}")
```

### 3.2 Eigenvalues of Symmetric Matrices

```python
# Symmetric matrix - guarantees real eigenvalues
S = np.array([[2, 1, 0],
              [1, 3, 1],
              [0, 1, 2]])

# eigh is optimized for symmetric matrices (faster and more stable)
eigenvalues, eigenvectors = np.linalg.eigh(S)

print("Symmetric matrix eigenvalue decomposition:")
print(f"Eigenvalues: {eigenvalues}")
print(f"\nEigenvector orthogonality verification:")
print(f"V^T @ V =\n{eigenvectors.T @ eigenvectors}")  # Identity matrix
```

### 3.3 Power Method

```python
def power_method(A, max_iter=100, tol=1e-10):
    """Calculate largest eigenvalue and eigenvector using power method"""
    n = A.shape[0]
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)

    for _ in range(max_iter):
        v_new = A @ v
        eigenvalue = np.dot(v, v_new)  # Rayleigh quotient
        v_new = v_new / np.linalg.norm(v_new)

        if np.linalg.norm(v_new - v) < tol:
            break
        v = v_new

    return eigenvalue, v_new

A = np.array([[4, 1], [2, 3]])
lam, v = power_method(A)
print(f"Power method result:")
print(f"  Largest eigenvalue: {lam:.6f}")
print(f"  Eigenvector: {v}")

# Compare with numpy result
lam_np, v_np = np.linalg.eig(A)
print(f"\nnumpy result:")
print(f"  Eigenvalues: {lam_np}")
```

---

## 4. Matrix Decomposition

Matrix decompositions are the "toolbox" of numerical linear algebra. Each decomposition exposes different structure in the matrix and is suited to different tasks: LU for general linear systems, Cholesky for symmetric positive definite systems (2x faster), QR for least-squares and eigenvalue problems, and SVD for rank analysis and data compression.

### 4.1 LU Decomposition

```python
from scipy.linalg import lu, lu_factor, lu_solve

A = np.array([[2, 1, 1],
              [4, 3, 3],
              [8, 7, 9]], dtype=float)

# LU decomposition
P, L, U = lu(A)

print("LU decomposition:")
print(f"P (permutation matrix):\n{P}")
print(f"L (lower triangular):\n{L}")
print(f"U (upper triangular):\n{U}")
print(f"\nVerification: P @ L @ U =\n{P @ L @ U}")

# Using for solving linear systems
b = np.array([4, 10, 24])
lu_piv = lu_factor(A)
x = lu_solve(lu_piv, b)
print(f"\nSolution: {x}")
```

### 4.2 Cholesky Decomposition

```python
# Only possible for positive definite symmetric matrices
A_spd = np.array([[4, 2, 2],
                  [2, 5, 1],
                  [2, 1, 6]], dtype=float)

# Cholesky decomposition: A = L @ L.T
L = np.linalg.cholesky(A_spd)

print("Cholesky decomposition:")
print(f"L:\n{L}")
print(f"\nVerification: L @ L.T =\n{L @ L.T}")

# Solving linear systems (2x faster than LU)
b = np.array([8, 8, 9])
# Solve L @ y = b, then L.T @ x = y
y = linalg.solve_triangular(L, b, lower=True)
x = linalg.solve_triangular(L.T, y)
print(f"\nSolution: {x}")
```

### 4.3 QR Decomposition

```python
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 10],
              [10, 11, 12]], dtype=float)

# QR decomposition
Q, R = np.linalg.qr(A)

print("QR decomposition:")
print(f"Q (orthogonal matrix):\n{Q}")
print(f"\nR (upper triangular):\n{R}")
print(f"\nVerification: Q @ R =\n{Q @ R}")
print(f"\nOrthogonality of Q: Q.T @ Q =\n{Q.T @ Q}")

# Using for least squares problem
b = np.array([1, 2, 3, 4])
x = linalg.solve_triangular(R, Q.T @ b)
print(f"\nLeast squares solution: {x}")
```

### 4.4 SVD (Singular Value Decomposition)

```python
A = np.array([[1, 2],
              [3, 4],
              [5, 6]])

# SVD: A = U @ Σ @ V.T
U, s, Vt = np.linalg.svd(A)

print("SVD decomposition:")
print(f"U (m×m orthogonal):\n{U}")
print(f"\nSingular values: {s}")
print(f"\nV.T (n×n orthogonal):\n{Vt}")

# Construct Σ matrix
Sigma = np.zeros_like(A, dtype=float)
np.fill_diagonal(Sigma, s)

print(f"\nVerification: U @ Σ @ V.T =\n{U @ Sigma @ Vt}")

# Condition number calculation
cond = s[0] / s[-1]
print(f"\nCondition number: {cond:.4f}")
```

---

## 5. Sparse Matrices

**Why sparse matrices matter:** PDE discretization on an $N$-point grid produces an $N \times N$ matrix, but each row has only a few non-zero entries (e.g., 3 for 1D tridiagonal, 5 for 2D five-point stencil). Storing and operating on the full dense matrix wastes memory ($O(N^2)$) and time ($O(N^3)$ for solving). Sparse matrix formats store only the non-zeros, reducing memory to $O(N)$ and enabling specialized solvers that exploit the sparsity pattern.

### 5.1 Sparse Matrix Formats

```python
from scipy import sparse

# COO format (coordinate format)
row = [0, 1, 2, 2]
col = [0, 1, 0, 2]
data = [1, 2, 3, 4]
A_coo = sparse.coo_matrix((data, (row, col)), shape=(3, 3))

print("COO format:")
print(A_coo)
print(f"\nDense form:\n{A_coo.toarray()}")

# CSR format (compressed sparse row, efficient for matrix-vector multiplication)
A_csr = A_coo.tocsr()
print(f"\nCSR format: {A_csr}")

# CSC format (compressed sparse column, efficient for column slicing)
A_csc = A_coo.tocsc()
```

### 5.2 Sparse Matrix Operations

```python
# Create large sparse matrix
n = 1000
diagonals = [np.ones(n-1), -2*np.ones(n), np.ones(n-1)]
offsets = [-1, 0, 1]
A_sparse = sparse.diags(diagonals, offsets, format='csr')

print(f"Sparse matrix size: {A_sparse.shape}")
print(f"Number of non-zero elements: {A_sparse.nnz}")
print(f"Sparsity: {1 - A_sparse.nnz / (n*n):.4%}")

# Sparse linear system
b = np.random.randn(n)
x = sparse.linalg.spsolve(A_sparse, b)
print(f"\nNorm of sparse linear system solution: {np.linalg.norm(x):.4f}")
```

### 5.3 Iterative Solvers

```python
from scipy.sparse.linalg import cg, gmres, bicgstab

# Create symmetric positive definite matrix
n = 100
A = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(n, n), format='csr')
A = A.T @ A + 0.1 * sparse.eye(n)  # Make positive definite
b = np.random.randn(n)

# CG (Conjugate Gradient) - optimal for symmetric positive definite
x_cg, info_cg = cg(A, b, tol=1e-10)
print(f"CG: convergence status = {info_cg}, residual = {np.linalg.norm(A @ x_cg - b):.2e}")

# GMRES - general matrices
x_gmres, info_gmres = gmres(A, b, tol=1e-10)
print(f"GMRES: convergence status = {info_gmres}, residual = {np.linalg.norm(A @ x_gmres - b):.2e}")
```

---

## 6. Application Examples

### 6.1 Discretization of Heat Conduction Problem

```python
def heat_equation_matrix(n, alpha, dx, dt):
    """Implicit discretization matrix for 1D heat equation"""
    r = alpha * dt / dx**2

    # Create tridiagonal matrix
    main_diag = (1 + 2*r) * np.ones(n)
    off_diag = -r * np.ones(n - 1)

    A = np.diag(main_diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
    return A

n = 10
A = heat_equation_matrix(n, alpha=0.1, dx=0.1, dt=0.01)
print("Heat equation discretization matrix:")
print(A[:5, :5])  # Print only part
```

### 6.2 Image Compression (SVD)

```python
def svd_compression(image, k):
    """Compress image using SVD (using top k singular values)"""
    U, s, Vt = np.linalg.svd(image, full_matrices=False)

    # Use only top k components
    compressed = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

    # Calculate compression ratio
    original_size = image.shape[0] * image.shape[1]
    compressed_size = k * (image.shape[0] + image.shape[1] + 1)
    compression_ratio = compressed_size / original_size

    return compressed, compression_ratio

# Example (random image)
image = np.random.randn(100, 100)
for k in [5, 10, 20, 50]:
    comp, ratio = svd_compression(image, k)
    error = np.linalg.norm(image - comp, 'fro') / np.linalg.norm(image, 'fro')
    print(f"k={k:2d}: compression ratio={ratio:.2%}, relative error={error:.4f}")
```

---

## Practice Problems

### Problem 1
Find the eigenvalues of a 3x3 tridiagonal matrix and compare with the power method result.

```python
# Solution
T = np.array([[2, -1, 0],
              [-1, 2, -1],
              [0, -1, 2]], dtype=float)

# numpy
eig_np, _ = np.linalg.eig(T)
print(f"numpy eigenvalues: {sorted(eig_np)}")

# Power method
lam, v = power_method(T)
print(f"Power method largest eigenvalue: {lam:.6f}")
```

### Problem 2
Solve a 100x100 sparse matrix linear system using direct and iterative methods and compare execution times.

```python
import time

n = 1000
A = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(n, n), format='csr')
A = A + 3 * sparse.eye(n)  # Make diagonally dominant
b = np.random.randn(n)

# Direct method
start = time.time()
x_direct = sparse.linalg.spsolve(A, b)
time_direct = time.time() - start

# Iterative method (CG)
A_spd = A.T @ A
b_spd = A.T @ b
start = time.time()
x_cg, _ = cg(A_spd, b_spd, tol=1e-10)
time_cg = time.time() - start

print(f"Direct method: {time_direct:.4f}s")
print(f"CG: {time_cg:.4f}s")
```

---

## Summary

| Decomposition Method | Purpose | Requirements |
|----------|------|------|
| LU | Solving linear systems | Square matrix |
| Cholesky | Solving linear systems (fast) | Symmetric positive definite |
| QR | Least squares, eigenvalues | Any matrix |
| SVD | Compression, pseudoinverse | Any matrix |

| Solver | Matrix Type | Characteristics |
|------|----------|------|
| spsolve | Sparse | Direct method |
| CG | Symmetric positive definite | Iterative method |
| GMRES | General | Iterative method |
| BiCGSTAB | Non-symmetric | Iterative method |

## Exercises

### Exercise 1: LU vs. Cholesky Solve Comparison

Solve the linear system `Ax = b` where `A` is the 4×4 symmetric positive definite matrix below using (a) LU decomposition and (b) Cholesky decomposition. Verify that both solutions agree and compare the number of arithmetic operations conceptually.

```
A = [[10, 2, 1, 0],
     [ 2, 8, 2, 1],
     [ 1, 2, 9, 3],
     [ 0, 1, 3, 7]]
b = [1, 2, 3, 4]
```

<details>
<summary>Show Answer</summary>

```python
import numpy as np
from scipy.linalg import lu_factor, lu_solve, cholesky, solve_triangular

A = np.array([[10, 2, 1, 0],
              [ 2, 8, 2, 1],
              [ 1, 2, 9, 3],
              [ 0, 1, 3, 7]], dtype=float)
b = np.array([1, 2, 3, 4], dtype=float)

# (a) LU decomposition
lu_piv = lu_factor(A)
x_lu   = lu_solve(lu_piv, b)
print(f"LU solution:       {x_lu}")
print(f"LU residual:       {np.linalg.norm(A @ x_lu - b):.2e}")

# (b) Cholesky decomposition (A must be symmetric positive definite)
L = cholesky(A, lower=True)          # A = L @ L.T
y = solve_triangular(L, b, lower=True)   # forward substitution
x_chol = solve_triangular(L.T, y)        # back substitution
print(f"\nCholesky solution: {x_chol}")
print(f"Cholesky residual: {np.linalg.norm(A @ x_chol - b):.2e}")

# Verify agreement
print(f"\nMax difference between solutions: {np.max(np.abs(x_lu - x_chol)):.2e}")
```

Cholesky requires roughly **half** the operations of LU for SPD matrices (≈ n³/6 vs. n³/3 flops) because it exploits symmetry. Both give the same solution up to floating-point rounding.

</details>

### Exercise 2: Power Method and Convergence Rate

Apply the power method to find the largest eigenvalue of the matrix `B = [[3, 1], [1, 3]]`. Run 20 iterations starting from `v₀ = [1, 0]`, record the Rayleigh quotient at each step, and plot convergence. Verify that the convergence rate is `|λ₂/λ₁|` where `λ₁ = 4` and `λ₂ = 2`.

<details>
<summary>Show Answer</summary>

```python
import numpy as np
import matplotlib.pyplot as plt

B = np.array([[3, 1], [1, 3]], dtype=float)
# Eigenvalues: λ₁ = 4 (eigenvector [1,1]/√2), λ₂ = 2 (eigenvector [1,-1]/√2)
# Convergence ratio = |λ₂/λ₁| = 2/4 = 0.5

v = np.array([1.0, 0.0])
v = v / np.linalg.norm(v)

rayleigh_quotients = []
true_lambda = 4.0

for k in range(20):
    w = B @ v
    lam = v @ w                       # Rayleigh quotient
    rayleigh_quotients.append(lam)
    v = w / np.linalg.norm(w)

errors = [abs(r - true_lambda) for r in rayleigh_quotients]

plt.figure(figsize=(9, 4))
plt.semilogy(errors, 'bo-', label='|λ_approx - λ₁|')

# Theoretical convergence: error ~ (|λ₂/λ₁|)^(2k) = 0.5^(2k) = 0.25^k
k_vals = np.arange(20)
plt.semilogy(k_vals, errors[0] * (0.5**2)**k_vals, 'r--',
             label='Theoretical (|λ₂/λ₁|)²ᵏ')
plt.xlabel('Iteration k')
plt.ylabel('Error in eigenvalue')
plt.title('Power Method Convergence')
plt.legend()
plt.grid(True)
plt.show()

print(f"Final eigenvalue estimate: {rayleigh_quotients[-1]:.10f}")
print(f"True value:                {true_lambda:.10f}")
print(f"Convergence ratio |λ₂/λ₁| = {2/4:.2f}")
```

The Rayleigh quotient converges at rate `(|λ₂/λ₁|)²` per iteration because the eigenvector error shrinks as `|λ₂/λ₁|^k`, and the Rayleigh quotient error is second-order in the eigenvector error.

</details>

### Exercise 3: SVD Low-Rank Approximation Error

For a randomly generated 20×10 matrix `M` (seed 42), compute the SVD and construct rank-k approximations for k = 1, 2, 3, 5, 8. For each rank, compute the Frobenius norm error `||M - M_k||_F` and show it equals the square root of the sum of squared singular values from index k+1 onward (Eckart-Young theorem).

<details>
<summary>Show Answer</summary>

```python
import numpy as np

rng = np.random.default_rng(42)
M = rng.standard_normal((20, 10))

U, s, Vt = np.linalg.svd(M, full_matrices=False)

print(f"Singular values: {s.round(3)}\n")
print(f"{'Rank k':>7}  {'||M - M_k||_F':>16}  {'sqrt(Σσᵢ² i>k)':>18}  {'Match?':>8}")
print("-" * 58)

for k in [1, 2, 3, 5, 8]:
    # Rank-k approximation
    M_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

    # Frobenius norm of error
    error_frob = np.linalg.norm(M - M_k, 'fro')

    # Eckart-Young: should equal sqrt(sum of s[k:]^2)
    eckart_young = np.sqrt(np.sum(s[k:]**2))

    match = np.isclose(error_frob, eckart_young)
    print(f"{k:>7}  {error_frob:>16.6f}  {eckart_young:>18.6f}  {str(match):>8}")
```

The Eckart-Young theorem states that the best rank-k approximation in the Frobenius norm (and spectral norm) is given by truncated SVD, with error `||M - M_k||_F = sqrt(σ²_{k+1} + ... + σ²_r)`.

</details>

### Exercise 4: Iterative Solver Comparison on a 1D Poisson System

Build the tridiagonal system arising from 1D finite differences on `-u''(x) = f(x)` with n = 200 interior points and `f(x) = π² sin(πx)` (exact solution: `u(x) = sin(πx)`). Solve with (a) `numpy.linalg.solve`, (b) `scipy.sparse.linalg.spsolve`, and (c) Conjugate Gradient (CG). Record wall-clock time and residual for each.

<details>
<summary>Show Answer</summary>

```python
import numpy as np
import time
from scipy import sparse
from scipy.sparse.linalg import spsolve, cg

n   = 200
h   = 1.0 / (n + 1)
x   = np.linspace(h, 1 - h, n)

# Right-hand side
f = np.pi**2 * np.sin(np.pi * x)

# Tridiagonal matrix: (1/h²)(2u_i - u_{i-1} - u_{i+1}) = f_i
diags  = [np.ones(n-1), -2*np.ones(n), np.ones(n-1)]
A_sp   = sparse.diags(diags, [-1, 0, 1], format='csr') * (-1 / h**2)
A_np   = A_sp.toarray()
b      = f.copy()

# (a) NumPy dense solver
t0 = time.perf_counter()
x_np = np.linalg.solve(A_np, b)
t_np = time.perf_counter() - t0

# (b) SciPy sparse direct solver
t0 = time.perf_counter()
x_sp = spsolve(A_sp, b)
t_sp = time.perf_counter() - t0

# (c) Conjugate Gradient
t0 = time.perf_counter()
x_cg, info = cg(A_sp, b, tol=1e-10)
t_cg = time.perf_counter() - t0

u_exact = np.sin(np.pi * x)

for name, sol, t in [("NumPy dense", x_np, t_np),
                     ("Sparse direct", x_sp, t_sp),
                     ("CG iterative", x_cg, t_cg)]:
    res   = np.linalg.norm(A_sp @ sol - b)
    err   = np.linalg.norm(sol - u_exact, np.inf)
    print(f"{name:>14}: time={t*1e3:6.2f} ms  residual={res:.2e}  max_err={err:.2e}")
```

The sparse direct solver is typically 10-100× faster than the dense solver for large tridiagonal systems because it exploits sparsity. CG is competitive for very large systems but requires more iterations for high accuracy.

</details>
