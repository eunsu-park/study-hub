#!/bin/bash
# Exercises for Lesson 10: Matrix Decompositions
# Topic: Linear_Algebra
# Solutions to practice problems from the lesson.

# === Exercise 1: Cholesky Decomposition ===
# Problem: Compute the Cholesky decomposition of the SPD matrix
# A = [[4, 2, 2], [2, 5, 1], [2, 1, 6]] and solve Ax = [1, 2, 3].
exercise_1() {
    echo "=== Exercise 1: Cholesky Decomposition ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np
from scipy.linalg import cho_factor, cho_solve

A = np.array([[4, 2, 2],
              [2, 5, 1],
              [2, 1, 6]], dtype=float)
b = np.array([1, 2, 3], dtype=float)

# Verify A is SPD
eigenvalues = np.linalg.eigvalsh(A)
print(f"A:\n{A}")
print(f"Eigenvalues: {np.round(eigenvalues, 4)}")
print(f"SPD: {np.all(eigenvalues > 0)}")

# Cholesky: A = L L^T
L = np.linalg.cholesky(A)
print(f"\nL (lower triangular):\n{np.round(L, 4)}")
print(f"L @ L^T:\n{np.round(L @ L.T, 10)}")
print(f"A == L L^T: {np.allclose(A, L @ L.T)}")

# Solve: L y = b (forward), L^T x = y (back)
from scipy.linalg import solve_triangular
y = solve_triangular(L, b, lower=True)
x = solve_triangular(L.T, y, lower=False)
print(f"\nSolution: {np.round(x, 4)}")
print(f"Verification: {np.round(A @ x, 10)}")
SOLUTION
}

# === Exercise 2: LDL^T Decomposition ===
# Problem: Compute LDL^T decomposition of a symmetric matrix.
exercise_2() {
    echo "=== Exercise 2: LDL^T Decomposition ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np
from scipy.linalg import ldl

A = np.array([[4, 2, -2],
              [2, 10, 4],
              [-2, 4, 5]], dtype=float)

L_ldl, D_arr, perm = ldl(A)
print(f"A:\n{A}")
print(f"\nL:\n{np.round(L_ldl, 4)}")
print(f"\nD (diagonal):\n{np.round(D_arr, 4)}")

# Verify A = L D L^T
A_recon = L_ldl @ D_arr @ L_ldl.T
print(f"\nL @ D @ L^T:\n{np.round(A_recon, 10)}")
print(f"Match: {np.allclose(A, A_recon)}")

# Advantage: works for indefinite symmetric matrices
# D entries tell definiteness: all positive -> PD
D_diag = np.diag(D_arr)
print(f"\nD diagonal entries: {D_diag}")
print(f"Positive definite: {np.all(D_diag > 0)}")
SOLUTION
}

# === Exercise 3: Schur Decomposition ===
# Problem: Compute the Schur decomposition A = Q T Q^H for a non-symmetric
# matrix and verify.
exercise_3() {
    echo "=== Exercise 3: Schur Decomposition ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np
from scipy.linalg import schur

A = np.array([[1, 2, 3],
              [0, 4, 5],
              [0, 0, 6]], dtype=float)

T, Q = schur(A)
print(f"A:\n{A}")
print(f"\nSchur form T (upper triangular):\n{np.round(T, 4)}")
print(f"\nUnitary Q:\n{np.round(Q, 4)}")

# Verify A = Q T Q^T
A_recon = Q @ T @ Q.T
print(f"\nQ @ T @ Q^T:\n{np.round(A_recon, 10)}")
print(f"Match: {np.allclose(A, A_recon)}")
print(f"Q is orthogonal: {np.allclose(Q.T @ Q, np.eye(3))}")

# Eigenvalues appear on diagonal of T
print(f"\nDiagonal of T: {np.diag(T)}")
print(f"Eigenvalues: {np.linalg.eigvals(A)}")
SOLUTION
}

# === Exercise 4: Comparing Decompositions ===
# Problem: For a given SPD matrix, compare LU, Cholesky, and
# eigendecomposition for solving Ax = b in terms of accuracy.
exercise_4() {
    echo "=== Exercise 4: Comparing Decompositions ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np
from scipy.linalg import lu_factor, lu_solve, cho_factor, cho_solve

# Create SPD matrix
np.random.seed(42)
M = np.random.randn(5, 5)
A = M.T @ M + np.eye(5)  # Guaranteed SPD
b = np.array([1, 2, 3, 4, 5], dtype=float)

print(f"Condition number: {np.linalg.cond(A):.4f}")

# Method 1: LU decomposition
lu, piv = lu_factor(A)
x_lu = lu_solve((lu, piv), b)

# Method 2: Cholesky
c, low = cho_factor(A)
x_chol = cho_solve((c, low), b)

# Method 3: Eigendecomposition
eigenvalues, Q = np.linalg.eigh(A)
x_eig = Q @ (Q.T @ b / eigenvalues)

# Method 4: Direct solve
x_direct = np.linalg.solve(A, b)

print(f"\nResidual norms ||Ax - b||:")
for name, x in [("LU", x_lu), ("Cholesky", x_chol),
                ("Eigen", x_eig), ("Direct", x_direct)]:
    r = np.linalg.norm(A @ x - b)
    print(f"  {name:10s}: {r:.2e}")

print(f"\nAll solutions agree: "
      f"{np.allclose(x_lu, x_chol) and np.allclose(x_chol, x_eig)}")
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 10: Matrix Decompositions"
echo "========================================================"
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
