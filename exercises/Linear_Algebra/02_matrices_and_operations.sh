#!/bin/bash
# Exercises for Lesson 02: Matrices and Operations
# Topic: Linear_Algebra
# Solutions to practice problems from the lesson.

# === Exercise 1: Matrix Multiplication Dimensions ===
# Problem: Given A (3x4), B (4x2), C (2x5), compute the shape of ABC
# and verify with NumPy.
exercise_1() {
    echo "=== Exercise 1: Matrix Multiplication Dimensions ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

A = np.random.randn(3, 4)
B = np.random.randn(4, 2)
C = np.random.randn(2, 5)

# ABC: (3x4)(4x2)(2x5) = (3x5)
ABC = A @ B @ C
print(f"A: {A.shape}")
print(f"B: {B.shape}")
print(f"C: {C.shape}")
print(f"ABC: {ABC.shape}")

# Associativity: (AB)C == A(BC)
ABC_left = (A @ B) @ C
ABC_right = A @ (B @ C)
print(f"\n(AB)C == A(BC): {np.allclose(ABC_left, ABC_right)}")
SOLUTION
}

# === Exercise 2: Inverse and System Solving ===
# Problem: Find the inverse of A = [[2,1],[5,3]] and use it to solve
# Ax = [7, 17]^T.
exercise_2() {
    echo "=== Exercise 2: Inverse and System Solving ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

A = np.array([[2, 1],
              [5, 3]])
b = np.array([7, 17])

# Manual inverse of 2x2: A^{-1} = (1/det) * [[d,-b],[-c,a]]
det = A[0,0]*A[1,1] - A[0,1]*A[1,0]
A_inv_manual = (1/det) * np.array([[A[1,1], -A[0,1]],
                                    [-A[1,0], A[0,0]]])

A_inv = np.linalg.inv(A)
print(f"A:\n{A}")
print(f"det(A) = {det}")
print(f"\nA^(-1) (manual):\n{A_inv_manual}")
print(f"A^(-1) (numpy):\n{A_inv}")

# Solve via inverse: x = A^{-1} b
x = A_inv @ b
print(f"\nx = A^(-1) @ b = {x}")
print(f"Verification: A @ x = {A @ x}")
print(f"Match b: {np.allclose(A @ x, b)}")
SOLUTION
}

# === Exercise 3: Determinant Properties ===
# Problem: Verify det(AB) = det(A)*det(B) and det(A^T) = det(A) for
# two random 3x3 matrices.
exercise_3() {
    echo "=== Exercise 3: Determinant Properties ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

np.random.seed(42)
A = np.random.randint(-5, 6, (3, 3)).astype(float)
B = np.random.randint(-5, 6, (3, 3)).astype(float)

print(f"A:\n{A}")
print(f"B:\n{B}")

det_A = np.linalg.det(A)
det_B = np.linalg.det(B)
det_AB = np.linalg.det(A @ B)
det_AT = np.linalg.det(A.T)

print(f"\ndet(A) = {det_A:.4f}")
print(f"det(B) = {det_B:.4f}")
print(f"det(AB) = {det_AB:.4f}")
print(f"det(A)*det(B) = {det_A * det_B:.4f}")
print(f"det(AB) == det(A)*det(B): {np.isclose(det_AB, det_A * det_B)}")

print(f"\ndet(A^T) = {det_AT:.4f}")
print(f"det(A^T) == det(A): {np.isclose(det_AT, det_A)}")

# Also: det(kA) = k^n * det(A) for n x n matrix
k = 3
print(f"\ndet({k}*A) = {np.linalg.det(k * A):.4f}")
print(f"{k}^3 * det(A) = {k**3 * det_A:.4f}")
print(f"Match: {np.isclose(np.linalg.det(k * A), k**3 * det_A)}")
SOLUTION
}

# === Exercise 4: Special Matrices ===
# Problem: Construct a 3x3 orthogonal matrix from Gram-Schmidt on
# v1=[1,1,0], v2=[1,0,1], v3=[0,1,1]. Verify Q^T Q = I.
exercise_4() {
    echo "=== Exercise 4: Special Matrices ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

v1 = np.array([1, 1, 0], dtype=float)
v2 = np.array([1, 0, 1], dtype=float)
v3 = np.array([0, 1, 1], dtype=float)

# Gram-Schmidt
u1 = v1 / np.linalg.norm(v1)

u2 = v2 - np.dot(v2, u1) * u1
u2 = u2 / np.linalg.norm(u2)

u3 = v3 - np.dot(v3, u1) * u1 - np.dot(v3, u2) * u2
u3 = u3 / np.linalg.norm(u3)

Q = np.column_stack([u1, u2, u3])
print(f"Orthogonal matrix Q:\n{np.round(Q, 4)}")
print(f"\nQ^T @ Q:\n{np.round(Q.T @ Q, 10)}")
print(f"Q^T Q == I: {np.allclose(Q.T @ Q, np.eye(3))}")
print(f"det(Q) = {np.linalg.det(Q):.4f}")

# Verify: orthogonal matrix preserves norms
x = np.array([1, 2, 3])
print(f"\n||x|| = {np.linalg.norm(x):.4f}")
print(f"||Qx|| = {np.linalg.norm(Q @ x):.4f}")
SOLUTION
}

# === Exercise 5: Trace and Rank ===
# Problem: For A = [[1,2,3],[4,5,6],[5,7,9]], compute rank, trace, and
# explain why A is singular.
exercise_5() {
    echo "=== Exercise 5: Trace and Rank ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [5, 7, 9]])

print(f"A:\n{A}")
print(f"Rank: {np.linalg.matrix_rank(A)}")
print(f"Trace: {np.trace(A)}")
print(f"Determinant: {np.linalg.det(A):.6f}")
print(f"Singular: {np.isclose(np.linalg.det(A), 0)}")

# Row 3 = Row 1 + Row 2
print(f"\nRow1 + Row2 = {A[0] + A[1]}")
print(f"Row3        = {A[2]}")
print(f"Row3 == Row1 + Row2: {np.allclose(A[2], A[0] + A[1])}")
print("A is singular because its rows are linearly dependent.")
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 02: Matrices and Operations"
echo "=========================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
echo ""
exercise_5
