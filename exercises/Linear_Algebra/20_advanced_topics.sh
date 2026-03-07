#!/bin/bash
# Exercises for Lesson 20: Advanced Decompositions and Applications
# Topic: Linear_Algebra
# Solutions to practice problems from the lesson.

# === Exercise 1: Jordan Normal Form ===
# Problem: Find the Jordan normal form of a matrix with repeated
# eigenvalues: A = [[5,4,2,1],[0,1,-1,-1],[-1,-1,3,0],[1,1,-1,2]].
exercise_1() {
    echo "=== Exercise 1: Jordan Normal Form ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np
from scipy.linalg import jordan_normal_form as jnf

# Simple example with a defective matrix
# A has eigenvalue 2 with algebraic multiplicity 2 but geometric mult 1
A = np.array([[2, 1],
              [0, 2]], dtype=float)

print(f"A:\n{A}")
eigenvalues = np.linalg.eigvals(A)
print(f"Eigenvalues: {eigenvalues}")

# A is already in Jordan form!
# Jordan block for eigenvalue 2, size 2x2:
# [[2, 1], [0, 2]]
print(f"\nA is already a Jordan block.")
print(f"Eigenvalue 2 with algebraic multiplicity 2, geometric multiplicity 1.")

# Verify: (A - 2I) is nilpotent
N = A - 2 * np.eye(2)
print(f"\nA - 2I:\n{N}")
print(f"(A-2I)^2:\n{N @ N}")
print(f"Nilpotent: {np.allclose(N @ N, 0)}")

# Larger example
B = np.array([[3, 1, 0],
              [0, 3, 1],
              [0, 0, 3]], dtype=float)
print(f"\nB (3x3 Jordan block):\n{B}")
print(f"Eigenvalues: {np.linalg.eigvals(B)}")
print(f"Rank(B-3I) = {np.linalg.matrix_rank(B - 3*np.eye(3))}")
print(f"Geometric multiplicity = {3 - np.linalg.matrix_rank(B - 3*np.eye(3))}")
SOLUTION
}

# === Exercise 2: Matrix Logarithm ===
# Problem: Compute the matrix logarithm of a rotation matrix and verify
# that exponentiating gives back the original.
exercise_2() {
    echo "=== Exercise 2: Matrix Logarithm ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np
from scipy.linalg import logm, expm

# Rotation matrix (30 degrees)
theta = np.pi / 6
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

print(f"R (rotation by {np.degrees(theta)} deg):\n{np.round(R, 4)}")

# Matrix logarithm
log_R = logm(R)
print(f"\nlog(R):\n{np.round(log_R, 4)}")

# log(R) should be skew-symmetric: [[0, -theta], [theta, 0]]
print(f"\nExpected: [[0, {-theta:.4f}], [{theta:.4f}, 0]]")
print(f"Skew-symmetric: {np.allclose(log_R, -log_R.T)}")

# Verify exp(log(R)) = R
R_recovered = expm(log_R)
print(f"\nexp(log(R)):\n{np.round(R_recovered, 4)}")
print(f"Match: {np.allclose(R, R_recovered)}")

# For rotation: log(R(theta)) = theta * [[0,-1],[1,0]]
J = np.array([[0, -1], [1, 0]], dtype=float)
print(f"\ntheta * J:\n{np.round(theta * J, 4)}")
print(f"Match: {np.allclose(log_R, theta * J)}")
SOLUTION
}

# === Exercise 3: Kronecker Product Properties ===
# Problem: Verify key Kronecker product identities and solve a
# Kronecker-structured linear system.
exercise_3() {
    echo "=== Exercise 3: Kronecker Product Properties ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

A = np.array([[1, 2], [3, 4]], dtype=float)
B = np.array([[5, 6], [7, 8]], dtype=float)

# vec(AXB^T) = (B kron A) vec(X)
# This is the key identity for Kronecker products in linear algebra.
X = np.array([[1, 0], [0, 1]], dtype=float)

# Left side: vec(AXB^T)
AXBt = A @ X @ B.T
vec_AXBt = AXBt.flatten(order='F')  # Column-major vectorization

# Right side: (B kron A) vec(X)
BkronA = np.kron(B, A)
vec_X = X.flatten(order='F')
result = BkronA @ vec_X

print(f"A:\n{A}")
print(f"B:\n{B}")
print(f"X:\n{X}")
print(f"\nvec(AXB^T) = {vec_AXBt}")
print(f"(B kron A) vec(X) = {result}")
print(f"Match: {np.allclose(vec_AXBt, result)}")

# Solve Kronecker system: (B kron A) x = b
b = np.array([1, 2, 3, 4], dtype=float)
x_kron = np.linalg.solve(BkronA, b)
print(f"\n(B kron A) x = b")
print(f"x = {np.round(x_kron, 4)}")

# Efficient solve: reshape x to matrix, solve A X_mat B^T = b_mat
b_mat = b.reshape(2, 2, order='F')
# X_mat = A^{-1} b_mat (B^T)^{-1}
X_mat = np.linalg.solve(A, b_mat) @ np.linalg.inv(B.T)
x_efficient = X_mat.flatten(order='F')
print(f"Efficient solve: {np.round(x_efficient, 4)}")
print(f"Match: {np.allclose(x_kron, x_efficient)}")
SOLUTION
}

# === Exercise 4: Generalized Eigenvalue Problem ===
# Problem: Solve the generalized eigenvalue problem Ax = lambda Bx
# where A and B are symmetric and B is positive definite.
exercise_4() {
    echo "=== Exercise 4: Generalized Eigenvalue Problem ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np
from scipy.linalg import eigh

# Stiffness and mass matrices (common in structural engineering)
A = np.array([[6, -2, 0],
              [-2, 4, -2],
              [0, -2, 3]], dtype=float)  # Stiffness

B = np.array([[2, 0, 0],
              [0, 1, 0],
              [0, 0, 1]], dtype=float)  # Mass (SPD)

print(f"A (stiffness):\n{A}")
print(f"B (mass):\n{B}")

# Solve Ax = lambda Bx
eigenvalues, eigenvectors = eigh(A, B)
print(f"\nGeneralized eigenvalues: {np.round(eigenvalues, 4)}")
print(f"Eigenvectors:\n{np.round(eigenvectors, 4)}")

# Verify: A v_i = lambda_i B v_i
for i in range(3):
    lam = eigenvalues[i]
    v = eigenvectors[:, i]
    lhs = A @ v
    rhs = lam * B @ v
    print(f"\nlambda_{i+1} = {lam:.4f}")
    print(f"  Av = {np.round(lhs, 4)}")
    print(f"  lambda Bv = {np.round(rhs, 4)}")
    print(f"  Match: {np.allclose(lhs, rhs)}")

# B-orthogonality: V^T B V = I
print(f"\nV^T B V:\n{np.round(eigenvectors.T @ B @ eigenvectors, 10)}")

# Physical interpretation: eigenvalues are natural frequencies squared
print(f"\nNatural frequencies: {np.round(np.sqrt(eigenvalues), 4)}")
SOLUTION
}

# === Exercise 5: Polar Decomposition ===
# Problem: Compute the polar decomposition A = UP where U is unitary
# and P is positive semidefinite.
exercise_5() {
    echo "=== Exercise 5: Polar Decomposition ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np
from scipy.linalg import polar

A = np.array([[1, 2],
              [3, 4]], dtype=float)

# Polar decomposition: A = U P
U, P = polar(A)

print(f"A:\n{A}")
print(f"\nUnitary factor U:\n{np.round(U, 4)}")
print(f"Positive semidefinite factor P:\n{np.round(P, 4)}")
print(f"\nU @ P:\n{np.round(U @ P, 10)}")
print(f"Match: {np.allclose(A, U @ P)}")

# U properties
print(f"\nU^T U = I: {np.allclose(U.T @ U, np.eye(2))}")
print(f"det(U) = {np.linalg.det(U):.4f}")

# P properties
evals_P = np.linalg.eigvalsh(P)
print(f"\nP eigenvalues: {np.round(evals_P, 4)}")
print(f"P is PSD: {np.all(evals_P >= -1e-10)}")
print(f"P is symmetric: {np.allclose(P, P.T)}")

# P = sqrt(A^T A)
ATA_sqrt = np.linalg.cholesky(A.T @ A)  # Not exactly sqrt
# Better: use eigendecomposition
evals, Q = np.linalg.eigh(A.T @ A)
P_manual = Q @ np.diag(np.sqrt(evals)) @ Q.T
print(f"\nP via sqrt(A^T A):\n{np.round(P_manual, 4)}")
print(f"Match: {np.allclose(P, P_manual)}")
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 20: Advanced Decompositions and Applications"
echo "============================================================================"
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
echo ""
exercise_5
