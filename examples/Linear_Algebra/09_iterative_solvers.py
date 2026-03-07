"""
Iterative Solvers

Demonstrates iterative methods for large linear systems:
- Conjugate Gradient (CG) for SPD systems
- GMRES for general nonsymmetric systems
- Jacobi and Gauss-Seidel iteration
- Preconditioning with incomplete Cholesky
- Convergence monitoring and comparison

Dependencies: numpy, scipy, matplotlib
"""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg
import matplotlib.pyplot as plt
import time


def jacobi_iteration():
    """Implement Jacobi iterative method."""
    print("=" * 60)
    print("JACOBI ITERATION")
    print("=" * 60)

    # Create diagonally dominant system
    A = np.array([[4, -1, 0, 0],
                  [-1, 4, -1, 0],
                  [0, -1, 4, -1],
                  [0, 0, -1, 4]], dtype=float)
    b = np.array([15, 10, 10, 15], dtype=float)
    x_exact = np.linalg.solve(A, b)

    print(f"A:\n{A}")
    print(f"b: {b}")
    print(f"Exact solution: {np.round(x_exact, 4)}")

    # Jacobi: x_{k+1} = D^{-1} (b - (L + U) x_k)
    n = len(b)
    D = np.diag(np.diag(A))
    D_inv = np.diag(1.0 / np.diag(A))
    LU = A - D

    x = np.zeros(n)
    residuals = []

    print(f"\n{'Iter':>4}  {'||r||':>12}  {'||x - x*||':>12}")
    print("-" * 35)

    for k in range(30):
        x = D_inv @ (b - LU @ x)
        r = np.linalg.norm(b - A @ x)
        e = np.linalg.norm(x - x_exact)
        residuals.append(r)
        if k < 10 or k % 5 == 0:
            print(f"{k+1:4d}  {r:12.2e}  {e:12.2e}")
        if r < 1e-10:
            break

    print(f"Converged in {len(residuals)} iterations")
    return residuals


def gauss_seidel():
    """Implement Gauss-Seidel iterative method."""
    print("\n" + "=" * 60)
    print("GAUSS-SEIDEL ITERATION")
    print("=" * 60)

    A = np.array([[4, -1, 0, 0],
                  [-1, 4, -1, 0],
                  [0, -1, 4, -1],
                  [0, 0, -1, 4]], dtype=float)
    b = np.array([15, 10, 10, 15], dtype=float)
    x_exact = np.linalg.solve(A, b)

    n = len(b)
    x = np.zeros(n)
    residuals = []

    print(f"{'Iter':>4}  {'||r||':>12}  {'||x - x*||':>12}")
    print("-" * 35)

    for k in range(30):
        for i in range(n):
            sigma = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], x[i+1:])
            x[i] = (b[i] - sigma) / A[i, i]
        r = np.linalg.norm(b - A @ x)
        e = np.linalg.norm(x - x_exact)
        residuals.append(r)
        if k < 10 or k % 5 == 0:
            print(f"{k+1:4d}  {r:12.2e}  {e:12.2e}")
        if r < 1e-10:
            break

    print(f"Converged in {len(residuals)} iterations")
    print(f"(Gauss-Seidel converges faster than Jacobi for this system)")
    return residuals


def conjugate_gradient():
    """Implement Conjugate Gradient method for SPD systems."""
    print("\n" + "=" * 60)
    print("CONJUGATE GRADIENT")
    print("=" * 60)

    # Create SPD system
    n = 100
    A = sparse.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n), format='csr', dtype=float)
    x_true = np.ones(n)
    b = A @ x_true

    print(f"System size: {n}x{n}")
    print(f"Condition number: {np.linalg.cond(A.toarray()):.2f}")

    # Manual CG implementation
    x = np.zeros(n)
    r = b - A @ x
    p = r.copy()
    residuals_cg = [np.linalg.norm(r)]

    for k in range(n):
        Ap = A @ p
        alpha = np.dot(r, r) / np.dot(p, Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        residuals_cg.append(np.linalg.norm(r_new))

        if np.linalg.norm(r_new) < 1e-10:
            break

        beta = np.dot(r_new, r_new) / np.dot(r, r)
        p = r_new + beta * p
        r = r_new

    print(f"CG converged in {len(residuals_cg) - 1} iterations")
    print(f"||x - x_true|| = {np.linalg.norm(x - x_true):.2e}")

    # SciPy CG comparison
    residuals_scipy = []

    def callback(xk):
        residuals_scipy.append(np.linalg.norm(b - A @ xk))

    x_scipy, info = splinalg.cg(A, b, tol=1e-10, callback=callback)
    print(f"SciPy CG iterations: {len(residuals_scipy)}")
    print(f"Solutions match: {np.allclose(x, x_scipy, atol=1e-8)}")

    return residuals_cg


def gmres_solver():
    """Demonstrate GMRES for nonsymmetric systems."""
    print("\n" + "=" * 60)
    print("GMRES (GENERAL NONSYMMETRIC SYSTEMS)")
    print("=" * 60)

    # Create nonsymmetric system
    n = 100
    np.random.seed(42)
    # Convection-diffusion type matrix (nonsymmetric)
    A = sparse.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n), format='csr', dtype=float)
    # Add convection term (nonsymmetric part)
    A += 0.1 * sparse.diags([1, -1], [-1, 1], shape=(n, n), format='csr')

    b = np.ones(n)

    print(f"System size: {n}x{n}")
    print(f"Symmetric: {np.allclose(A.toarray(), A.toarray().T)}")

    residuals_gmres = []

    def callback(rk):
        residuals_gmres.append(rk)

    x_gmres, info = splinalg.gmres(A, b, tol=1e-10, callback=callback,
                                    callback_type='pr_norm')
    print(f"GMRES converged: {info == 0}")
    print(f"Iterations: {len(residuals_gmres)}")
    print(f"||Ax - b|| = {np.linalg.norm(A @ x_gmres - b):.2e}")

    return residuals_gmres


def preconditioning():
    """Demonstrate the effect of preconditioning."""
    print("\n" + "=" * 60)
    print("PRECONDITIONING")
    print("=" * 60)

    n = 200
    # Ill-conditioned SPD system
    diags = [2 + 0.01 * np.arange(n)]
    A = sparse.diags([-1, diags[0], -1], [-1, 0, 1], shape=(n, n), format='csr')
    b = np.ones(n)

    print(f"System size: {n}x{n}")
    cond = np.linalg.cond(A.toarray())
    print(f"Condition number: {cond:.2f}")

    # Unpreconditioned CG
    residuals_no_prec = []

    def cb1(xk):
        residuals_no_prec.append(np.linalg.norm(b - A @ xk))

    x1, info1 = splinalg.cg(A, b, tol=1e-8, maxiter=500, callback=cb1)
    print(f"\nUnpreconditioned CG: {len(residuals_no_prec)} iterations")

    # Jacobi preconditioner M = diag(A)
    M_jacobi = sparse.diags(1.0 / A.diagonal())

    residuals_jacobi = []

    def cb2(xk):
        residuals_jacobi.append(np.linalg.norm(b - A @ xk))

    x2, info2 = splinalg.cg(A, b, tol=1e-8, maxiter=500, M=M_jacobi, callback=cb2)
    print(f"Jacobi-preconditioned CG: {len(residuals_jacobi)} iterations")

    # ILU preconditioner (incomplete factorization)
    A_csc = A.tocsc()
    ilu = splinalg.spilu(A_csc)
    M_ilu = splinalg.LinearOperator(A.shape, matvec=ilu.solve)

    residuals_ilu = []

    def cb3(xk):
        residuals_ilu.append(np.linalg.norm(b - A @ xk))

    x3, info3 = splinalg.cg(A, b, tol=1e-8, maxiter=500, M=M_ilu, callback=cb3)
    print(f"ILU-preconditioned CG: {len(residuals_ilu)} iterations")

    print(f"\nAll solutions match: "
          f"{np.allclose(x1, x2, atol=1e-6) and np.allclose(x2, x3, atol=1e-6)}")

    return residuals_no_prec, residuals_jacobi, residuals_ilu


def visualize_convergence(res_jacobi, res_gs, res_cg, res_no_prec, res_jac_prec, res_ilu_prec):
    """Create convergence comparison plots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Classical methods comparison
    ax = axes[0]
    ax.semilogy(range(1, len(res_jacobi) + 1), res_jacobi, 'b-o', markersize=3, label='Jacobi')
    ax.semilogy(range(1, len(res_gs) + 1), res_gs, 'r-s', markersize=3, label='Gauss-Seidel')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('||r||')
    ax.set_title('Classical Iterative Methods')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Preconditioning comparison
    ax = axes[1]
    ax.semilogy(range(1, len(res_no_prec) + 1), res_no_prec, 'b-', label='No precond.')
    ax.semilogy(range(1, len(res_jac_prec) + 1), res_jac_prec, 'r-', label='Jacobi precond.')
    ax.semilogy(range(1, len(res_ilu_prec) + 1), res_ilu_prec, 'g-', label='ILU precond.')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('||r||')
    ax.set_title('Effect of Preconditioning on CG')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('iterative_convergence.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: iterative_convergence.png")


if __name__ == "__main__":
    res_jacobi = jacobi_iteration()
    res_gs = gauss_seidel()
    res_cg = conjugate_gradient()
    res_gmres = gmres_solver()
    res_no_prec, res_jac_prec, res_ilu_prec = preconditioning()
    visualize_convergence(res_jacobi, res_gs, res_cg, res_no_prec, res_jac_prec, res_ilu_prec)
    print("\nAll examples completed!")
