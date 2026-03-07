#!/bin/bash
# Exercises for Lesson 03: Systems of Linear Equations
# Topic: Linear_Algebra
# Solutions to practice problems from the lesson.

# === Exercise 1: Gaussian Elimination ===
# Problem: Solve the system using Gaussian elimination:
# 2x + y - z = 8, -3x - y + 2z = -11, -2x + y + 2z = -3
exercise_1() {
    echo "=== Exercise 1: Gaussian Elimination ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

A = np.array([[2, 1, -1],
              [-3, -1, 2],
              [-2, 1, 2]], dtype=float)
b = np.array([8, -11, -3], dtype=float)

# Augmented matrix
aug = np.column_stack([A, b])
print(f"Augmented [A|b]:\n{aug}")

# Row reduction (manual steps)
# R2 = R2 + (3/2)R1
aug[1] += (3/2) * aug[0]
# R3 = R3 + R1
aug[2] += aug[0]
print(f"\nAfter forward elimination step 1:\n{aug}")

# R3 = R3 - 4*R2
aug[2] -= 4 * aug[1]
print(f"After forward elimination step 2:\n{aug}")

# Back substitution
z = aug[2, 3] / aug[2, 2]
y = (aug[1, 3] - aug[1, 2] * z) / aug[1, 1]
x = (aug[0, 3] - aug[0, 1] * y - aug[0, 2] * z) / aug[0, 0]

print(f"\nSolution: x={x}, y={y}, z={z}")

# Verify with np.linalg.solve
x_np = np.linalg.solve(A, b)
print(f"numpy solve: {x_np}")
print(f"Match: {np.allclose([x, y, z], x_np)}")
SOLUTION
}

# === Exercise 2: LU Decomposition ===
# Problem: Compute LU decomposition of A = [[2,1,1],[4,3,3],[8,7,9]]
# and use it to solve Ax = [4, 10, 24]^T.
exercise_2() {
    echo "=== Exercise 2: LU Decomposition ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np
from scipy.linalg import lu, lu_factor, lu_solve

A = np.array([[2, 1, 1],
              [4, 3, 3],
              [8, 7, 9]], dtype=float)
b = np.array([4, 10, 24], dtype=float)

# Compute PA = LU
P, L, U = lu(A)
print(f"P:\n{P}")
print(f"L:\n{L}")
print(f"U:\n{U}")
print(f"P @ L @ U == A: {np.allclose(P @ L @ U, A)}")

# Solve using LU
lu_factors, piv = lu_factor(A)
x = lu_solve((lu_factors, piv), b)
print(f"\nSolution: {x}")
print(f"Verification Ax: {A @ x}")

# Solve a second system with same A
b2 = np.array([1, 0, 0], dtype=float)
x2 = lu_solve((lu_factors, piv), b2)
print(f"\nSecond solution (b=[1,0,0]): {x2}")
print(f"Verification: {A @ x2}")
SOLUTION
}

# === Exercise 3: Existence and Uniqueness ===
# Problem: Classify the following systems as having a unique solution,
# infinitely many solutions, or no solution.
exercise_3() {
    echo "=== Exercise 3: Existence and Uniqueness ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

# System 1: Unique solution (full rank)
A1 = np.array([[1, 2], [3, 4]])
b1 = np.array([5, 6])
rank_A1 = np.linalg.matrix_rank(A1)
rank_Ab1 = np.linalg.matrix_rank(np.column_stack([A1, b1]))
print("System 1: [[1,2],[3,4]] x = [5,6]")
print(f"  rank(A)={rank_A1}, rank([A|b])={rank_Ab1}, n={A1.shape[1]}")
print(f"  Result: Unique solution (rank(A) = rank([A|b]) = n)")
print(f"  x = {np.linalg.solve(A1, b1)}")

# System 2: No solution (inconsistent)
A2 = np.array([[1, 1], [1, 1]])
b2 = np.array([2, 3])
rank_A2 = np.linalg.matrix_rank(A2)
rank_Ab2 = np.linalg.matrix_rank(np.column_stack([A2, b2]))
print(f"\nSystem 2: [[1,1],[1,1]] x = [2,3]")
print(f"  rank(A)={rank_A2}, rank([A|b])={rank_Ab2}")
print(f"  Result: No solution (rank(A) < rank([A|b]))")

# System 3: Infinitely many solutions (underdetermined)
A3 = np.array([[1, 2, 3], [4, 5, 6]])
b3 = np.array([6, 15])
rank_A3 = np.linalg.matrix_rank(A3)
rank_Ab3 = np.linalg.matrix_rank(np.column_stack([A3, b3]))
print(f"\nSystem 3: [[1,2,3],[4,5,6]] x = [6,15]")
print(f"  rank(A)={rank_A3}, rank([A|b])={rank_Ab3}, n={A3.shape[1]}")
print(f"  Result: Infinitely many solutions (rank(A) = rank([A|b]) < n)")
print(f"  Minimum norm solution: {np.round(np.linalg.pinv(A3) @ b3, 4)}")
SOLUTION
}

# === Exercise 4: Overdetermined Least Squares ===
# Problem: Fit a line y = mx + c to points (0,1), (1,2), (2,4), (3,5)
# using least squares.
exercise_4() {
    echo "=== Exercise 4: Overdetermined Least Squares ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

x = np.array([0, 1, 2, 3], dtype=float)
y = np.array([1, 2, 4, 5], dtype=float)

# Design matrix: [x, 1]
A = np.column_stack([x, np.ones_like(x)])
print(f"Design matrix A:\n{A}")
print(f"y: {y}")

# Normal equations: (A^T A) c = A^T y
ATA = A.T @ A
ATy = A.T @ y
coeffs = np.linalg.solve(ATA, ATy)
m, c = coeffs

print(f"\nNormal equations:")
print(f"A^T A:\n{ATA}")
print(f"A^T y: {ATy}")
print(f"\nBest fit: y = {m:.4f}x + {c:.4f}")

# Verify with lstsq
coeffs_lstsq = np.linalg.lstsq(A, y, rcond=None)[0]
print(f"lstsq:   y = {coeffs_lstsq[0]:.4f}x + {coeffs_lstsq[1]:.4f}")

# Residual
residual = y - A @ coeffs
print(f"\nResiduals: {residual}")
print(f"Sum of squared residuals: {np.sum(residual**2):.4f}")
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 03: Systems of Linear Equations"
echo "=============================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
