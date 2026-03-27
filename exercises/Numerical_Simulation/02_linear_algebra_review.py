"""
Exercises for Lesson 02: Linear Algebra Review

Topics: LU/Cholesky decomposition, power method, SVD low-rank approximation,
        iterative solvers for sparse systems.
"""

import numpy as np
import time
from scipy import sparse
from scipy.linalg import lu_factor, lu_solve, cholesky, solve_triangular
from scipy.sparse.linalg import spsolve, cg

# ---------------------------------------------------------------------------
# Exercise 1: LU vs. Cholesky Solve Comparison
# ---------------------------------------------------------------------------
# Solve Ax = b where A is 4x4 SPD using (a) LU decomposition and
# (b) Cholesky decomposition. Verify both solutions agree.
#   A = [[10,2,1,0],[2,8,2,1],[1,2,9,3],[0,1,3,7]]
#   b = [1, 2, 3, 4]
# ---------------------------------------------------------------------------

def exercise_1():
    """LU vs. Cholesky solve comparison."""
    A = np.array([[10, 2, 1, 0],
                  [ 2, 8, 2, 1],
                  [ 1, 2, 9, 3],
                  [ 0, 1, 3, 7]], dtype=float)
    b = np.array([1, 2, 3, 4], dtype=float)

    # (a) LU decomposition
    lu_piv = lu_factor(A)
    x_lu = lu_solve(lu_piv, b)
    print(f"LU solution:       {x_lu}")
    print(f"LU residual:       {np.linalg.norm(A @ x_lu - b):.2e}")

    # (b) Cholesky decomposition (A must be symmetric positive definite)
    L = cholesky(A, lower=True)
    y = solve_triangular(L, b, lower=True)
    x_chol = solve_triangular(L.T, y)
    print(f"\nCholesky solution: {x_chol}")
    print(f"Cholesky residual: {np.linalg.norm(A @ x_chol - b):.2e}")

    # Verify agreement
    print(f"\nMax difference between solutions: {np.max(np.abs(x_lu - x_chol)):.2e}")


# ---------------------------------------------------------------------------
# Exercise 2: Power Method and Convergence Rate
# ---------------------------------------------------------------------------
# Apply the power method to B = [[3,1],[1,3]] for 20 iterations starting
# from v0 = [1, 0]. Record the Rayleigh quotient at each step and verify
# convergence rate is |lambda2/lambda1| = 2/4 = 0.5.
# ---------------------------------------------------------------------------

def exercise_2():
    """Power method convergence rate verification."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    B = np.array([[3, 1], [1, 3]], dtype=float)
    v = np.array([1.0, 0.0])
    v = v / np.linalg.norm(v)

    rayleigh_quotients = []
    true_lambda = 4.0

    for k in range(20):
        w = B @ v
        lam = v @ w
        rayleigh_quotients.append(lam)
        v = w / np.linalg.norm(w)

    errors = [abs(r - true_lambda) for r in rayleigh_quotients]

    plt.figure(figsize=(9, 4))
    plt.semilogy(errors, 'bo-', label='|lambda_approx - lambda_1|')
    k_vals = np.arange(20)
    plt.semilogy(k_vals, errors[0] * (0.5**2)**k_vals, 'r--',
                 label='Theoretical (|lambda_2/lambda_1|)^(2k)')
    plt.xlabel('Iteration k')
    plt.ylabel('Error in eigenvalue')
    plt.title('Power Method Convergence')
    plt.legend()
    plt.grid(True)
    plt.savefig('ex02_power_method.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved to ex02_power_method.png")

    print(f"Final eigenvalue estimate: {rayleigh_quotients[-1]:.10f}")
    print(f"True value:                {true_lambda:.10f}")
    print(f"Convergence ratio |lambda_2/lambda_1| = {2/4:.2f}")


# ---------------------------------------------------------------------------
# Exercise 3: SVD Low-Rank Approximation Error
# ---------------------------------------------------------------------------
# For a random 20x10 matrix M (seed 42), compute rank-k approximations
# for k = 1, 2, 3, 5, 8 and verify the Eckart-Young theorem.
# ---------------------------------------------------------------------------

def exercise_3():
    """SVD low-rank approximation and Eckart-Young theorem."""
    rng = np.random.default_rng(42)
    M = rng.standard_normal((20, 10))

    U, s, Vt = np.linalg.svd(M, full_matrices=False)

    print(f"Singular values: {s.round(3)}\n")
    print(f"{'Rank k':>7}  {'||M - M_k||_F':>16}  {'sqrt(sum s_i^2 i>k)':>20}  {'Match?':>8}")
    print("-" * 58)

    for k in [1, 2, 3, 5, 8]:
        M_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
        error_frob = np.linalg.norm(M - M_k, 'fro')
        eckart_young = np.sqrt(np.sum(s[k:]**2))
        match = np.isclose(error_frob, eckart_young)
        print(f"{k:>7}  {error_frob:>16.6f}  {eckart_young:>20.6f}  {str(match):>8}")


# ---------------------------------------------------------------------------
# Exercise 4: Iterative Solver Comparison on a 1D Poisson System
# ---------------------------------------------------------------------------
# Build the tridiagonal system from 1D finite differences on -u''(x) = f(x)
# with n = 200 interior points and f(x) = pi^2 sin(pi x). Solve with
# (a) numpy dense, (b) scipy sparse direct, (c) CG. Compare timing/residuals.
# ---------------------------------------------------------------------------

def exercise_4():
    """Iterative solver comparison on 1D Poisson system."""
    n = 200
    h = 1.0 / (n + 1)
    x = np.linspace(h, 1 - h, n)

    f = np.pi**2 * np.sin(np.pi * x)

    diags_vals = [np.ones(n - 1), -2 * np.ones(n), np.ones(n - 1)]
    A_sp = sparse.diags(diags_vals, [-1, 0, 1], format='csr') * (-1 / h**2)
    A_np = A_sp.toarray()
    b = f.copy()

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
        res = np.linalg.norm(A_sp @ sol - b)
        err = np.linalg.norm(sol - u_exact, np.inf)
        print(f"{name:>14}: time={t*1e3:6.2f} ms  residual={res:.2e}  max_err={err:.2e}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 1: LU vs. Cholesky Solve Comparison")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("Exercise 2: Power Method and Convergence Rate")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("Exercise 3: SVD Low-Rank Approximation Error")
    print("=" * 60)
    exercise_3()

    print("\n" + "=" * 60)
    print("Exercise 4: Iterative Solver Comparison on 1D Poisson")
    print("=" * 60)
    exercise_4()
