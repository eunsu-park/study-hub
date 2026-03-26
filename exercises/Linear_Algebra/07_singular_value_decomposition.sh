#!/bin/bash
# Exercises for Lesson 07: Singular Value Decomposition
# Topic: Linear_Algebra
# Solutions to practice problems from the lesson.

# === Exercise 1: SVD Computation ===
# Problem: Compute the SVD of A = [[3, 2, 2], [2, 3, -2]] and verify
# A = U Sigma V^T.
exercise_1() {
    echo "=== Exercise 1: SVD Computation ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

A = np.array([[3, 2, 2],
              [2, 3, -2]], dtype=float)

U, sigma, Vt = np.linalg.svd(A, full_matrices=True)
print(f"A:\n{A}")
print(f"\nU (2x2):\n{np.round(U, 4)}")
print(f"\nSingular values: {np.round(sigma, 4)}")
print(f"\nV^T (3x3):\n{np.round(Vt, 4)}")

# Reconstruct
Sigma = np.zeros_like(A)
np.fill_diagonal(Sigma, sigma)
A_recon = U @ Sigma @ Vt
print(f"\nReconstruction U @ Sigma @ V^T:\n{np.round(A_recon, 10)}")
print(f"Match: {np.allclose(A, A_recon)}")

# Verify U and V are orthogonal
print(f"\nU^T U = I: {np.allclose(U.T @ U, np.eye(2))}")
print(f"V^T V = I: {np.allclose(Vt @ Vt.T, np.eye(3))}")
SOLUTION
}

# === Exercise 2: Low-Rank Approximation ===
# Problem: Create a rank-1 approximation of A = [[1,2],[3,4],[5,6]]
# and compute the approximation error.
exercise_2() {
    echo "=== Exercise 2: Low-Rank Approximation ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

A = np.array([[1, 2],
              [3, 4],
              [5, 6]], dtype=float)

U, sigma, Vt = np.linalg.svd(A, full_matrices=False)

# Rank-1 approximation
A1 = sigma[0] * np.outer(U[:, 0], Vt[0, :])
print(f"A:\n{A}")
print(f"Singular values: {np.round(sigma, 4)}")
print(f"\nRank-1 approximation:\n{np.round(A1, 4)}")

# Error
error_frob = np.linalg.norm(A - A1, 'fro')
total_frob = np.linalg.norm(A, 'fro')
print(f"\n||A - A_1||_F = {error_frob:.4f}")
print(f"||A||_F = {total_frob:.4f}")
print(f"Relative error: {error_frob / total_frob:.4f}")

# Eckart-Young: error = sqrt(sum of remaining sigma^2)
theoretical_error = np.sqrt(np.sum(sigma[1:] ** 2))
print(f"\nEckart-Young error = sigma_2 = {sigma[1]:.4f}")
print(f"Actual error: {error_frob:.4f}")
print(f"Match: {np.isclose(error_frob, theoretical_error)}")
SOLUTION
}

# === Exercise 3: Pseudoinverse via SVD ===
# Problem: Compute the pseudoinverse of A = [[1,0],[0,0],[0,1]] using SVD
# and verify the Moore-Penrose conditions.
exercise_3() {
    echo "=== Exercise 3: Pseudoinverse via SVD ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

A = np.array([[1, 0],
              [0, 0],
              [0, 1]], dtype=float)

U, sigma, Vt = np.linalg.svd(A, full_matrices=False)

# Pseudoinverse: A^+ = V Sigma^+ U^T
sigma_pinv = np.array([1/s if s > 1e-10 else 0 for s in sigma])
A_pinv = Vt.T @ np.diag(sigma_pinv) @ U.T

print(f"A:\n{A}")
print(f"Singular values: {sigma}")
print(f"\nA^+ (via SVD):\n{np.round(A_pinv, 4)}")
print(f"A^+ (numpy):\n{np.round(np.linalg.pinv(A), 4)}")

# Moore-Penrose conditions
print(f"\nMoore-Penrose conditions:")
print(f"1. A A+ A == A:   {np.allclose(A @ A_pinv @ A, A)}")
print(f"2. A+ A A+ == A+: {np.allclose(A_pinv @ A @ A_pinv, A_pinv)}")
print(f"3. (A A+)^T = A A+: {np.allclose((A @ A_pinv).T, A @ A_pinv)}")
print(f"4. (A+ A)^T = A+ A: {np.allclose((A_pinv @ A).T, A_pinv @ A)}")
SOLUTION
}

# === Exercise 4: SVD and Matrix Properties ===
# Problem: Using SVD, compute rank, Frobenius norm, 2-norm, and condition
# number of A = [[1,2,3],[4,5,6],[7,8,9]].
exercise_4() {
    echo "=== Exercise 4: SVD and Matrix Properties ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]], dtype=float)

sigma = np.linalg.svd(A, compute_uv=False)
print(f"A:\n{A}")
print(f"Singular values: {np.round(sigma, 6)}")

# Rank = number of nonzero singular values
rank = np.sum(sigma > 1e-10)
print(f"\nRank: {rank}")

# Frobenius norm
frob = np.sqrt(np.sum(sigma ** 2))
print(f"||A||_F = sqrt(sum sigma_i^2) = {frob:.4f}")
print(f"numpy: {np.linalg.norm(A, 'fro'):.4f}")

# 2-norm = largest singular value
norm2 = sigma[0]
print(f"||A||_2 = sigma_1 = {norm2:.4f}")
print(f"numpy: {np.linalg.norm(A, 2):.4f}")

# Condition number = sigma_max / sigma_min (nonzero)
nonzero_sigma = sigma[sigma > 1e-10]
cond = nonzero_sigma[0] / nonzero_sigma[-1]
print(f"cond(A) = sigma_1/sigma_r = {cond:.4f}")
print(f"numpy: {np.linalg.cond(A):.4f}")
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 07: Singular Value Decomposition"
echo "==============================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
