#!/bin/bash
# Exercises for Lesson 19: Iterative Methods
# Topic: Linear_Algebra
# Solutions to practice problems from the lesson.

# === Exercise 1: Jacobi vs Gauss-Seidel ===
# Problem: Compare Jacobi and Gauss-Seidel convergence for the system
# A = [[4,-1,0],[-1,4,-1],[0,-1,4]], b = [15,10,10].
exercise_1() {
    echo "=== Exercise 1: Jacobi vs Gauss-Seidel ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

A = np.array([[4, -1, 0],
              [-1, 4, -1],
              [0, -1, 4]], dtype=float)
b = np.array([15, 10, 10], dtype=float)
x_exact = np.linalg.solve(A, b)
n = len(b)

# Jacobi iteration
x_j = np.zeros(n)
D_inv = np.diag(1.0 / np.diag(A))
LU = A - np.diag(np.diag(A))

print("Jacobi iteration:")
for k in range(10):
    x_j = D_inv @ (b - LU @ x_j)
    err = np.linalg.norm(x_j - x_exact)
    if k < 5 or k == 9:
        print(f"  Iter {k+1}: x = {np.round(x_j, 4)}, error = {err:.2e}")

# Gauss-Seidel iteration
x_gs = np.zeros(n)
print("\nGauss-Seidel iteration:")
for k in range(10):
    for i in range(n):
        sigma = np.dot(A[i, :i], x_gs[:i]) + np.dot(A[i, i+1:], x_gs[i+1:])
        x_gs[i] = (b[i] - sigma) / A[i, i]
    err = np.linalg.norm(x_gs - x_exact)
    if k < 5 or k == 9:
        print(f"  Iter {k+1}: x = {np.round(x_gs, 4)}, error = {err:.2e}")

print(f"\nExact solution: {np.round(x_exact, 4)}")
print("Gauss-Seidel converges faster for this system.")
SOLUTION
}

# === Exercise 2: Conjugate Gradient ===
# Problem: Implement CG from scratch to solve a 50x50 SPD system and
# count iterations to convergence.
exercise_2() {
    echo "=== Exercise 2: Conjugate Gradient ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

np.random.seed(42)
n = 50
# Create SPD matrix
M = np.random.randn(n, n)
A = M.T @ M + n * np.eye(n)
x_true = np.random.randn(n)
b = A @ x_true

# CG algorithm
x = np.zeros(n)
r = b - A @ x
p = r.copy()
r_norm_sq = np.dot(r, r)

print(f"System size: {n}x{n}")
print(f"Condition number: {np.linalg.cond(A):.2f}")
print(f"\n{'Iter':>4}  {'||r||':>12}")

for k in range(n):
    Ap = A @ p
    alpha = r_norm_sq / np.dot(p, Ap)
    x = x + alpha * p
    r = r - alpha * Ap

    r_norm_sq_new = np.dot(r, r)
    r_norm = np.sqrt(r_norm_sq_new)

    if k < 5 or k % 10 == 9 or r_norm < 1e-10:
        print(f"{k+1:4d}  {r_norm:12.2e}")

    if r_norm < 1e-10:
        print(f"\nConverged in {k+1} iterations")
        break

    beta = r_norm_sq_new / r_norm_sq
    p = r + beta * p
    r_norm_sq = r_norm_sq_new

error = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)
print(f"Relative error: {error:.2e}")
SOLUTION
}

# === Exercise 3: GMRES for Nonsymmetric System ===
# Problem: Solve a nonsymmetric 100x100 system using scipy GMRES
# with and without restart.
exercise_3() {
    echo "=== Exercise 3: GMRES for Nonsymmetric System ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import gmres

np.random.seed(42)
n = 100

# Nonsymmetric matrix: tridiagonal + asymmetric perturbation
A = sparse.diags([-1, 3, -1], [-1, 0, 1], shape=(n, n), format='csr', dtype=float)
A += 0.5 * sparse.diags([1, -1], [-1, 1], shape=(n, n), format='csr')

b = np.ones(n)

# Full GMRES (no restart)
iters_full = [0]
def count_full(rk):
    iters_full[0] += 1

x_full, info_full = gmres(A, b, tol=1e-10, callback=count_full,
                           callback_type='pr_norm')

# Restarted GMRES(20)
iters_restart = [0]
def count_restart(rk):
    iters_restart[0] += 1

x_restart, info_restart = gmres(A, b, tol=1e-10, restart=20,
                                 callback=count_restart,
                                 callback_type='pr_norm')

print(f"System: {n}x{n} nonsymmetric")
print(f"Symmetric: {np.allclose(A.toarray(), A.toarray().T)}")
print(f"\nFull GMRES:")
print(f"  Iterations: {iters_full[0]}")
print(f"  ||Ax-b||: {np.linalg.norm(A @ x_full - b):.2e}")

print(f"\nGMRES(20) (restarted):")
print(f"  Iterations: {iters_restart[0]}")
print(f"  ||Ax-b||: {np.linalg.norm(A @ x_restart - b):.2e}")

print(f"\nSolutions match: {np.allclose(x_full, x_restart, atol=1e-8)}")
SOLUTION
}

# === Exercise 4: Convergence Rate ===
# Problem: Show that CG convergence depends on condition number by
# solving systems with different condition numbers.
exercise_4() {
    echo "=== Exercise 4: Convergence Rate ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np
from scipy.sparse.linalg import cg

n = 100

print(f"{'Condition':>10}  {'CG iterations':>14}  {'Error':>12}")
print("-" * 40)

for kappa in [10, 100, 1000, 10000]:
    # Create SPD matrix with specified condition number
    np.random.seed(42)
    Q = np.linalg.qr(np.random.randn(n, n))[0]
    # Eigenvalues from 1 to kappa
    evals = np.linspace(1, kappa, n)
    A = Q @ np.diag(evals) @ Q.T

    x_true = np.ones(n)
    b = A @ x_true

    # Count CG iterations
    iterations = [0]
    def callback(xk):
        iterations[0] += 1

    x_cg, info = cg(A, b, tol=1e-8, callback=callback)
    error = np.linalg.norm(x_cg - x_true) / np.linalg.norm(x_true)

    print(f"{kappa:10.0f}  {iterations[0]:14d}  {error:12.2e}")

print(f"\nCG iterations roughly proportional to sqrt(kappa).")
print(f"Preconditioning reduces effective condition number.")
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 19: Iterative Methods"
echo "====================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
