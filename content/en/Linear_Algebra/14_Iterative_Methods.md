# Lesson 14: Iterative Methods

[Previous: Lesson 13](./13_Numerical_Linear_Algebra.md) | [Overview](./00_Overview.md) | [Next: Lesson 15](./15_Tensors_and_Multilinear_Algebra.md)

---

## Learning Objectives

- Understand when iterative methods are preferred over direct methods for solving linear systems
- Implement and analyze classical iterative methods: Jacobi, Gauss-Seidel, and SOR
- Derive and implement the Conjugate Gradient (CG) method for symmetric positive definite systems
- Understand Krylov subspace methods: GMRES and BiCGSTAB for non-symmetric systems
- Apply preconditioning techniques to accelerate convergence
- Choose the right solver for a given problem based on matrix properties

---

## 1. Direct vs Iterative Methods

### 1.1 When to Use Iterative Methods

**Direct methods** (LU, Cholesky, QR) compute the exact solution (up to rounding errors) in a fixed number of operations. **Iterative methods** generate a sequence of approximations $x^{(0)}, x^{(1)}, x^{(2)}, \ldots$ that converge to the solution.

| Criterion | Direct | Iterative |
|---|---|---|
| Matrix size | Small to moderate ($n < 10^4$) | Large ($n > 10^4$) |
| Sparsity | Dense or moderate | Sparse (preserves sparsity) |
| Required accuracy | Machine precision | Adjustable tolerance |
| Multiple right-hand sides | Efficient (factor once) | Each RHS from scratch |
| Memory | $O(n^2)$ or more (fill-in) | $O(\text{nnz})$ + few vectors |

```python
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, cg, gmres, bicgstab
import time
import matplotlib.pyplot as plt

# Compare direct vs iterative for increasing matrix sizes
def build_poisson_2d(n):
    """Build 2D Poisson system (n^2 x n^2 SPD matrix)."""
    T = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(n, n))
    I = sparse.eye(n)
    A = -(sparse.kron(I, T) + sparse.kron(T, I))
    return A.tocsr()

sizes = [20, 40, 60, 80, 100]
times_direct = []
times_iterative = []

for n in sizes:
    A = build_poisson_2d(n)
    b = np.random.randn(n**2)

    # Direct solve
    start = time.time()
    x_direct = spsolve(A.tocsc(), b)
    times_direct.append(time.time() - start)

    # Iterative solve (CG)
    start = time.time()
    x_iter, info = cg(A, b, tol=1e-10)
    times_iterative.append(time.time() - start)

    print(f"n={n:3d} ({n**2:5d} unknowns): direct={times_direct[-1]:.4f}s, "
          f"CG={times_iterative[-1]:.4f}s, "
          f"||x_d - x_i||/||x_d|| = {np.linalg.norm(x_direct - x_iter) / np.linalg.norm(x_direct):.2e}")
```

---

## 2. Classical Iterative Methods

### 2.1 Splitting Framework

Classical iterative methods are based on splitting the matrix $A = M - N$, leading to the iteration:

$$M x^{(k+1)} = N x^{(k)} + b \quad \Rightarrow \quad x^{(k+1)} = M^{-1} N x^{(k)} + M^{-1} b$$

The iteration converges if and only if the **spectral radius** $\rho(M^{-1}N) < 1$.

### 2.2 Jacobi Method

Split $A = D - (L + U)$, where $D$ is the diagonal, $L$ is the strict lower triangle, and $U$ is the strict upper triangle:

$$x_i^{(k+1)} = \frac{1}{a_{ii}} \left( b_i - \sum_{j \neq i} a_{ij} x_j^{(k)} \right)$$

```python
def jacobi(A, b, x0=None, tol=1e-8, max_iter=1000):
    """Jacobi iterative method."""
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy()
    D_inv = 1.0 / A.diagonal()
    residuals = []

    for k in range(max_iter):
        r = b - A @ x
        residual = np.linalg.norm(r)
        residuals.append(residual)

        if residual < tol * np.linalg.norm(b):
            print(f"Jacobi converged in {k+1} iterations")
            return x, residuals

        # x_new = D^{-1} (b - (L+U) x)
        x_new = D_inv * (b - A @ x + A.diagonal() * x)
        x = x_new

    print(f"Jacobi did not converge in {max_iter} iterations")
    return x, residuals

# Test on a diagonally dominant system
n = 50
A = build_poisson_2d(n)
b = np.ones(n**2)

x_jacobi, res_jacobi = jacobi(A, b, tol=1e-8, max_iter=5000)
```

### 2.3 Gauss-Seidel Method

Instead of using all old values, Gauss-Seidel uses updated values as soon as they are available:

$$x_i^{(k+1)} = \frac{1}{a_{ii}} \left( b_i - \sum_{j < i} a_{ij} x_j^{(k+1)} - \sum_{j > i} a_{ij} x_j^{(k)} \right)$$

```python
def gauss_seidel(A, b, x0=None, tol=1e-8, max_iter=1000):
    """Gauss-Seidel iterative method (works with dense or sparse matrices)."""
    A_dense = A.toarray() if sparse.issparse(A) else A
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy()
    residuals = []

    for k in range(max_iter):
        r = b - (A @ x if sparse.issparse(A) else A_dense @ x)
        residual = np.linalg.norm(r)
        residuals.append(residual)

        if residual < tol * np.linalg.norm(b):
            print(f"Gauss-Seidel converged in {k+1} iterations")
            return x, residuals

        for i in range(n):
            sigma = A_dense[i, :] @ x - A_dense[i, i] * x[i]
            x[i] = (b[i] - sigma) / A_dense[i, i]

    print(f"Gauss-Seidel did not converge in {max_iter} iterations")
    return x, residuals

# Test (smaller system for GS since it is serial)
n_small = 15
A_small = build_poisson_2d(n_small)
b_small = np.ones(n_small**2)

x_gs, res_gs = gauss_seidel(A_small, b_small, tol=1e-8, max_iter=5000)
```

### 2.4 Successive Over-Relaxation (SOR)

SOR introduces a relaxation parameter $\omega$ to accelerate Gauss-Seidel:

$$x_i^{(k+1)} = (1 - \omega) x_i^{(k)} + \frac{\omega}{a_{ii}} \left( b_i - \sum_{j < i} a_{ij} x_j^{(k+1)} - \sum_{j > i} a_{ij} x_j^{(k)} \right)$$

For $\omega = 1$, SOR reduces to Gauss-Seidel. The optimal $\omega$ lies in $(1, 2)$ and depends on the spectral radius of the Jacobi iteration matrix.

```python
def sor(A, b, omega=1.5, x0=None, tol=1e-8, max_iter=1000):
    """Successive Over-Relaxation."""
    A_dense = A.toarray() if sparse.issparse(A) else A
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy()
    residuals = []

    for k in range(max_iter):
        r = b - (A @ x if sparse.issparse(A) else A_dense @ x)
        residual = np.linalg.norm(r)
        residuals.append(residual)

        if residual < tol * np.linalg.norm(b):
            print(f"SOR (omega={omega}) converged in {k+1} iterations")
            return x, residuals

        for i in range(n):
            sigma = A_dense[i, :] @ x - A_dense[i, i] * x[i]
            x_gs = (b[i] - sigma) / A_dense[i, i]
            x[i] = (1 - omega) * x[i] + omega * x_gs

    print(f"SOR did not converge in {max_iter} iterations")
    return x, residuals

# Compare convergence for different omega values
omegas = [0.8, 1.0, 1.2, 1.5, 1.8]
results = {}
for omega in omegas:
    x_sor, res_sor = sor(A_small, b_small, omega=omega, tol=1e-8, max_iter=3000)
    results[omega] = res_sor

# Plot convergence
plt.figure(figsize=(10, 6))
for omega, res in results.items():
    label = f'omega={omega}' + (' (GS)' if omega == 1.0 else '')
    plt.semilogy(res[:500], label=label)
plt.xlabel('Iteration')
plt.ylabel('Residual norm')
plt.title('SOR Convergence for Different omega')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 3. Conjugate Gradient Method

### 3.1 CG Derivation

The Conjugate Gradient (CG) method is the optimal Krylov subspace method for **symmetric positive definite** (SPD) systems. It minimizes the $A$-norm of the error over the Krylov subspace:

$$\mathcal{K}_k(A, r_0) = \text{span}\{r_0, Ar_0, A^2 r_0, \ldots, A^{k-1} r_0\}$$

The key insight: CG generates a sequence of **conjugate directions** $p_0, p_1, \ldots$ satisfying $p_i^T A p_j = 0$ for $i \neq j$.

### 3.2 Algorithm

```
Given: SPD matrix A, right-hand side b, initial guess x_0
r_0 = b - A x_0
p_0 = r_0

For k = 0, 1, 2, ...
    alpha_k = (r_k^T r_k) / (p_k^T A p_k)
    x_{k+1} = x_k + alpha_k * p_k
    r_{k+1} = r_k - alpha_k * A p_k
    beta_k = (r_{k+1}^T r_{k+1}) / (r_k^T r_k)
    p_{k+1} = r_{k+1} + beta_k * p_k

    If ||r_{k+1}|| < tol: stop
```

```python
def conjugate_gradient(A, b, x0=None, tol=1e-10, max_iter=None):
    """Conjugate Gradient method for SPD matrices."""
    n = len(b)
    if max_iter is None:
        max_iter = 2 * n
    x = np.zeros(n) if x0 is None else x0.copy()

    r = b - A @ x
    p = r.copy()
    rs_old = r @ r
    residuals = [np.sqrt(rs_old)]

    for k in range(max_iter):
        Ap = A @ p
        alpha = rs_old / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = r @ r
        residuals.append(np.sqrt(rs_new))

        if np.sqrt(rs_new) < tol * np.linalg.norm(b):
            print(f"CG converged in {k+1} iterations")
            return x, residuals

        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    print(f"CG did not converge in {max_iter} iterations")
    return x, residuals

# Test on 2D Poisson system
n = 30
A = build_poisson_2d(n)
b = np.ones(n**2)

x_cg, res_cg = conjugate_gradient(A, b, tol=1e-10)

# Compare with scipy
x_scipy, info = cg(A, b, tol=1e-10)
print(f"||x_cg - x_scipy|| = {np.linalg.norm(x_cg - x_scipy):.2e}")
```

### 3.3 Convergence Analysis

CG converges in at most $n$ iterations (in exact arithmetic). In practice, the convergence rate depends on the **condition number** $\kappa(A) = \lambda_{\max} / \lambda_{\min}$:

$$\|e_k\|_A \leq 2 \left( \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1} \right)^k \|e_0\|_A$$

A smaller condition number leads to faster convergence.

```python
# Demonstrate CG convergence vs condition number
fig, ax = plt.subplots(figsize=(10, 6))

for n in [10, 20, 40]:
    A = build_poisson_2d(n)
    b = np.ones(n**2)
    cond = np.linalg.cond(A.toarray()) if n <= 20 else None

    x, residuals = conjugate_gradient(A, b, tol=1e-12)

    label = f'n={n} (N={n**2})'
    if cond is not None:
        label += f', kappa={cond:.0f}'
    ax.semilogy(np.array(residuals) / residuals[0], label=label)

ax.set_xlabel('Iteration')
ax.set_ylabel('Relative residual')
ax.set_title('CG Convergence for 2D Poisson Problems')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 4. GMRES (Generalized Minimum Residual)

### 4.1 Overview

GMRES works for **any non-singular matrix** (not just SPD). It finds the solution in the Krylov subspace that minimizes the 2-norm of the residual:

$$x_k = \arg\min_{x \in x_0 + \mathcal{K}_k} \|b - Ax\|_2$$

GMRES builds an orthonormal basis for the Krylov subspace using the **Arnoldi process** and solves a small least-squares problem at each step.

```python
def gmres_simple(A, b, x0=None, tol=1e-10, max_iter=None):
    """Simplified GMRES (full, no restarts)."""
    n = len(b)
    if max_iter is None:
        max_iter = min(n, 200)

    x = np.zeros(n) if x0 is None else x0.copy()
    r0 = b - A @ x
    beta = np.linalg.norm(r0)
    residuals = [beta]

    if beta < tol:
        return x, residuals

    # Arnoldi vectors
    V = np.zeros((n, max_iter + 1))
    H = np.zeros((max_iter + 1, max_iter))
    V[:, 0] = r0 / beta

    for j in range(max_iter):
        # Arnoldi step
        w = A @ V[:, j]
        for i in range(j + 1):
            H[i, j] = w @ V[:, i]
            w = w - H[i, j] * V[:, i]
        H[j + 1, j] = np.linalg.norm(w)

        if H[j + 1, j] < 1e-14:
            # Lucky breakdown: exact solution found
            break
        V[:, j + 1] = w / H[j + 1, j]

        # Solve least squares: min ||beta*e1 - H_j*y||
        e1 = np.zeros(j + 2)
        e1[0] = beta
        y, _, _, _ = np.linalg.lstsq(H[:j + 2, :j + 1], e1, rcond=None)
        x_k = x + V[:, :j + 1] @ y
        res = np.linalg.norm(b - A @ x_k)
        residuals.append(res)

        if res < tol * np.linalg.norm(b):
            print(f"GMRES converged in {j+1} iterations")
            return x_k, residuals

    print(f"GMRES did not converge in {max_iter} iterations")
    return x_k, residuals

# Test on non-symmetric system
np.random.seed(42)
n = 100
A_nonsym = sparse.random(n, n, density=0.1, format='csr', random_state=42)
A_nonsym = A_nonsym + 5 * sparse.eye(n)  # Make diagonally dominant
b = np.random.randn(n)

x_gmres, res_gmres = gmres_simple(A_nonsym.toarray(), b, tol=1e-10)

# Compare with scipy
x_scipy, info = gmres(A_nonsym, b, atol=1e-10)
print(f"||x_gmres - x_scipy|| = {np.linalg.norm(x_gmres - x_scipy):.2e}")
```

### 4.2 Restarted GMRES

Full GMRES stores all Arnoldi vectors, requiring $O(n \cdot k)$ memory for $k$ iterations. **Restarted GMRES(m)** restarts after $m$ iterations, using the current solution as the new initial guess.

```python
from scipy.sparse.linalg import gmres as scipy_gmres

# GMRES with different restart values
n = 50
A = build_poisson_2d(n)
# Make it non-symmetric by adding a convection term
A_conv = A + 0.5 * sparse.diags([np.ones(n**2 - 1)], [1], shape=(n**2, n**2))
b = np.ones(n**2)

for restart in [10, 20, 50, 100]:
    residuals = []
    def callback(rk):
        residuals.append(rk)

    x, info = scipy_gmres(A_conv, b, restart=restart, atol=1e-10,
                           callback=callback, callback_type='pr_norm')
    print(f"GMRES({restart:3d}): {len(residuals):4d} iterations, "
          f"info={info}, residual={np.linalg.norm(A_conv @ x - b):.2e}")
```

---

## 5. BiCGSTAB

### 5.1 Overview

BiCGSTAB (Bi-Conjugate Gradient Stabilized) is designed for **non-symmetric systems**. It combines the BiCG method with stabilization to avoid the erratic convergence of plain BiCG.

**Advantages over GMRES**:
- Fixed storage: only a few vectors regardless of iteration count
- No need for restarts

**Disadvantages**:
- Can stagnate or break down
- No optimality guarantee (unlike GMRES)

```python
# BiCGSTAB example
n = 50
A_conv = build_poisson_2d(n) + 0.5 * sparse.diags(
    [np.ones(n**2 - 1)], [1], shape=(n**2, n**2))
b = np.ones(n**2)

# Collect residual history
res_history_bicgstab = []
def callback_bicgstab(x):
    res_history_bicgstab.append(np.linalg.norm(A_conv @ x - b))

x_bicg, info = bicgstab(A_conv, b, tol=1e-10, callback=callback_bicgstab)
print(f"BiCGSTAB: {len(res_history_bicgstab)} iterations, info={info}")
print(f"Residual: {np.linalg.norm(A_conv @ x_bicg - b):.2e}")
```

### 5.2 Solver Comparison

```python
# Compare CG, GMRES, BiCGSTAB on an SPD system
n = 40
A_spd = build_poisson_2d(n)
b = np.ones(n**2)

solvers = {}

# CG (only for SPD)
res_cg = []
def cb_cg(x):
    res_cg.append(np.linalg.norm(A_spd @ x - b))
x_cg, _ = cg(A_spd, b, tol=1e-12, callback=cb_cg)
solvers['CG'] = res_cg

# GMRES
res_gmres = []
def cb_gmres(rk):
    res_gmres.append(rk)
x_gm, _ = scipy_gmres(A_spd, b, atol=1e-12, callback=cb_gmres,
                        callback_type='pr_norm')
solvers['GMRES'] = res_gmres

# BiCGSTAB
res_bicg = []
def cb_bicg(x):
    res_bicg.append(np.linalg.norm(A_spd @ x - b))
x_bi, _ = bicgstab(A_spd, b, tol=1e-12, callback=cb_bicg)
solvers['BiCGSTAB'] = res_bicg

# Plot comparison
plt.figure(figsize=(10, 6))
for name, res in solvers.items():
    plt.semilogy(res, label=f'{name} ({len(res)} iters)')
plt.xlabel('Iteration')
plt.ylabel('Residual norm')
plt.title('Solver Comparison on 2D Poisson (SPD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 6. Preconditioning

### 6.1 Why Precondition?

Preconditioning transforms the system $Ax = b$ into an equivalent system with a smaller condition number. Instead of solving $Ax = b$, we solve:

$$M^{-1}Ax = M^{-1}b \quad \text{(left preconditioning)}$$

A good preconditioner $M$ satisfies:
1. $M \approx A$ (so $M^{-1}A \approx I$, giving $\kappa(M^{-1}A) \approx 1$)
2. $Mz = r$ is cheap to solve

### 6.2 Jacobi (Diagonal) Preconditioner

The simplest preconditioner: $M = \text{diag}(A)$.

```python
from scipy.sparse.linalg import LinearOperator

def jacobi_preconditioner(A):
    """Diagonal (Jacobi) preconditioner."""
    d = A.diagonal()
    d_inv = 1.0 / d
    n = A.shape[0]
    return LinearOperator((n, n), matvec=lambda x: d_inv * x)

# Unpreconditioned CG
n = 50
A = build_poisson_2d(n)
b = np.ones(n**2)

res_unprecond = []
def cb1(x):
    res_unprecond.append(np.linalg.norm(A @ x - b))
cg(A, b, tol=1e-10, callback=cb1)

# Jacobi preconditioned CG
M_jacobi = jacobi_preconditioner(A)
res_jacobi = []
def cb2(x):
    res_jacobi.append(np.linalg.norm(A @ x - b))
cg(A, b, tol=1e-10, M=M_jacobi, callback=cb2)

print(f"Unpreconditioned CG: {len(res_unprecond)} iterations")
print(f"Jacobi PCG:          {len(res_jacobi)} iterations")
```

### 6.3 Incomplete LU (ILU) Preconditioner

ILU computes an approximate LU factorization by keeping only nonzero entries that match the sparsity pattern of $A$ (ILU(0)), or allowing limited fill-in (ILUT).

```python
from scipy.sparse.linalg import spilu

# ILU preconditioner
n = 50
A = build_poisson_2d(n)
A_csc = A.tocsc()
b = np.ones(n**2)

# Compute ILU(0)
ilu = spilu(A_csc, drop_tol=0.0)
M_ilu = LinearOperator(A.shape, matvec=ilu.solve)

# Compare convergence
res_none, res_jac, res_ilu = [], [], []

def cb_none(x):
    res_none.append(np.linalg.norm(A @ x - b))
def cb_jac(x):
    res_jac.append(np.linalg.norm(A @ x - b))
def cb_ilu(x):
    res_ilu.append(np.linalg.norm(A @ x - b))

cg(A, b, tol=1e-10, callback=cb_none)
cg(A, b, tol=1e-10, M=jacobi_preconditioner(A), callback=cb_jac)
cg(A, b, tol=1e-10, M=M_ilu, callback=cb_ilu)

print(f"No preconditioner: {len(res_none)} iterations")
print(f"Jacobi:            {len(res_jac)} iterations")
print(f"ILU(0):            {len(res_ilu)} iterations")

# Plot
plt.figure(figsize=(10, 6))
plt.semilogy(res_none, label=f'None ({len(res_none)} iters)')
plt.semilogy(res_jac, label=f'Jacobi ({len(res_jac)} iters)')
plt.semilogy(res_ilu, label=f'ILU(0) ({len(res_ilu)} iters)')
plt.xlabel('Iteration')
plt.ylabel('Residual norm')
plt.title('Effect of Preconditioning on CG Convergence')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### 6.4 SSOR Preconditioner

Symmetric Successive Over-Relaxation combines forward and backward Gauss-Seidel sweeps with a relaxation parameter $\omega$:

$$M_{\text{SSOR}} = \frac{1}{\omega(2-\omega)} (D + \omega L) D^{-1} (D + \omega U)$$

```python
def ssor_preconditioner(A, omega=1.0):
    """SSOR preconditioner (dense implementation for clarity)."""
    A_dense = A.toarray() if sparse.issparse(A) else A
    n = A_dense.shape[0]
    D = np.diag(np.diag(A_dense))
    L = np.tril(A_dense, -1)
    U = np.triu(A_dense, 1)

    # M = (D + omega*L) D^{-1} (D + omega*U) / (omega*(2-omega))
    D_inv = np.diag(1.0 / np.diag(A_dense))
    M1 = D + omega * L
    M2 = D + omega * U
    scale = omega * (2 - omega)

    def solve(r):
        # Forward solve: (D + omega*L) y = r
        y = np.linalg.solve(M1, r)
        # Scale: z = D y
        z = D @ y
        # Backward solve: (D + omega*U) x = z
        x = np.linalg.solve(M2, z)
        return x / scale

    return LinearOperator((n, n), matvec=solve)
```

---

## 7. Convergence Criteria

### 7.1 Stopping Criteria

Iterative methods need well-chosen stopping criteria to balance accuracy and computational cost:

```python
def cg_with_monitoring(A, b, tol=1e-10, max_iter=1000):
    """CG with detailed convergence monitoring."""
    n = len(b)
    x = np.zeros(n)
    r = b.copy()
    p = r.copy()
    rs_old = r @ r
    b_norm = np.linalg.norm(b)

    history = {
        'residual': [],
        'relative_residual': [],
        'a_norm_error': [],
    }

    # Compute exact solution for error tracking
    x_exact = spsolve(A.tocsc(), b)

    for k in range(max_iter):
        Ap = A @ p
        alpha = rs_old / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = r @ r

        # Monitor different quantities
        res_norm = np.sqrt(rs_new)
        error = x - x_exact
        a_norm_error = np.sqrt(error @ (A @ error))

        history['residual'].append(res_norm)
        history['relative_residual'].append(res_norm / b_norm)
        history['a_norm_error'].append(a_norm_error)

        # Stopping criterion: relative residual
        if res_norm / b_norm < tol:
            print(f"Converged in {k+1} iterations")
            break

        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    return x, history

n = 30
A = build_poisson_2d(n)
b = np.random.randn(n**2)

x, history = cg_with_monitoring(A, b, tol=1e-12)

# Plot different convergence measures
fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(history['residual'], label='||r_k||')
ax.semilogy(history['relative_residual'], label='||r_k||/||b||')
ax.semilogy(history['a_norm_error'], label='||e_k||_A')
ax.set_xlabel('Iteration')
ax.set_ylabel('Value')
ax.set_title('CG Convergence Measures')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 8. Choosing the Right Solver

### 8.1 Decision Flowchart

```
Is the matrix symmetric?
├── Yes: Is it positive definite?
│   ├── Yes → CG (+ preconditioner)
│   └── No → MINRES
└── No: Is it square?
    ├── Yes → GMRES or BiCGSTAB
    │   ├── Need optimality? → GMRES
    │   └── Need fixed memory? → BiCGSTAB
    └── No → LSQR or LSMR (least squares)
```

### 8.2 Summary Table

| Solver | Matrix type | Memory | Convergence | Notes |
|---|---|---|---|---|
| CG | SPD | $O(n)$ | Monotone | Best for SPD |
| MINRES | Symmetric indefinite | $O(n)$ | Monotone | For saddle point problems |
| GMRES | General | $O(nk)$ | Optimal | Restart for large $k$ |
| BiCGSTAB | General | $O(n)$ | Erratic | No restart needed |
| LSQR | Rectangular | $O(n)$ | Monotone | For least squares |

```python
# Demonstrate solver selection
from scipy.sparse.linalg import minres, lsqr

# Case 1: SPD matrix -> CG
A_spd = build_poisson_2d(20)
b = np.ones(400)
x_cg, info = cg(A_spd, b, tol=1e-10)
print(f"SPD system with CG: info={info}")

# Case 2: Non-symmetric -> GMRES
A_nonsym = A_spd + 0.3 * sparse.random(400, 400, density=0.01, format='csr')
x_gm, info = scipy_gmres(A_nonsym, b, atol=1e-10)
print(f"Non-symmetric with GMRES: info={info}")

# Case 3: Rectangular (overdetermined) -> LSQR
m, n_cols = 500, 100
A_rect = sparse.random(m, n_cols, density=0.05, format='csr')
b_rect = np.random.randn(m)
result = lsqr(A_rect, b_rect)
x_lsqr = result[0]
print(f"Least squares with LSQR: ||Ax-b|| = {np.linalg.norm(A_rect @ x_lsqr - b_rect):.4f}")
```

---

## Exercises

### Exercise 1: Jacobi Convergence

Implement the Jacobi method for the system $Ax = b$ where $A$ is the $100 \times 100$ tridiagonal matrix with 4 on the diagonal and -1 on the sub- and super-diagonals. Plot the convergence (residual vs. iteration). Compute the spectral radius of the Jacobi iteration matrix and relate it to the observed convergence rate.

### Exercise 2: CG Implementation

Implement CG from scratch and verify it against `scipy.sparse.linalg.cg` on a 2D Poisson system of size $50 \times 50$. Track the A-norm of the error $\|e_k\|_A = \sqrt{(x_k - x^*)^T A (x_k - x^*)}$ and verify that it decreases monotonically.

### Exercise 3: Preconditioner Comparison

For a 2D Poisson system of size $80 \times 80$, compare the following preconditioners with CG:

1. No preconditioner
2. Jacobi (diagonal scaling)
3. ILU(0) (incomplete LU with zero fill-in)

Report the number of iterations and total time for each. Which is the most effective?

### Exercise 4: GMRES vs BiCGSTAB

Create a non-symmetric convection-diffusion system by adding a convection term to the 2D Poisson system. Compare GMRES(20), GMRES(50), and BiCGSTAB in terms of iterations, time, and final residual.

### Exercise 5: Optimal SOR Parameter

For the $n \times n$ tridiagonal matrix with 2 on the diagonal and -1 on the off-diagonals, the optimal SOR parameter is known analytically: $\omega^* = 2 / (1 + \sin(\pi / (n+1)))$. Verify this experimentally for $n = 50$ by running SOR with $\omega$ values from 1.0 to 1.99 and plotting iterations to convergence vs. $\omega$.

---

[Previous: Lesson 13](./13_Numerical_Linear_Algebra.md) | [Overview](./00_Overview.md) | [Next: Lesson 15](./15_Tensors_and_Multilinear_Algebra.md)

**License**: CC BY-NC 4.0
