#!/bin/bash
# Exercises for Lesson 05: Linear Transformations
# Topic: Linear_Algebra
# Solutions to practice problems from the lesson.

# === Exercise 1: Transformation Matrix ===
# Problem: Find the standard matrix for the linear transformation
# T(x,y) = (2x+y, x-3y, 4y).
exercise_1() {
    echo "=== Exercise 1: Transformation Matrix ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

# T(x,y) = (2x+y, x-3y, 4y)
# T(e1) = T(1,0) = (2, 1, 0)
# T(e2) = T(0,1) = (1, -3, 4)
# Standard matrix: columns are T(e1), T(e2)
A = np.array([[2, 1],
              [1, -3],
              [0, 4]])

print(f"Standard matrix A (3x2):\n{A}")

# Test with specific vector
v = np.array([3, 2])
Tv = A @ v
print(f"\nT({v}) = {Tv}")
print(f"Manual: ({2*3+2}, {3-3*2}, {4*2}) = ({2*3+2}, {3-6}, {8})")
SOLUTION
}

# === Exercise 2: Kernel and Image ===
# Problem: Find the kernel and image of A = [[1,2,3],[2,4,6]].
exercise_2() {
    echo "=== Exercise 2: Kernel and Image ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np
from scipy.linalg import null_space

A = np.array([[1, 2, 3],
              [2, 4, 6]], dtype=float)

print(f"A:\n{A}")
print(f"Rank: {np.linalg.matrix_rank(A)}")

# Kernel (null space): Ax = 0
ker = null_space(A)
print(f"\nKernel (null space) basis:\n{np.round(ker, 4)}")
print(f"dim(ker) = {ker.shape[1]}")

# Verify kernel vectors
for i in range(ker.shape[1]):
    print(f"A @ ker[:,{i}] = {np.round(A @ ker[:, i], 10)}")

# Image (column space)
# rank = 1, so image = span{[1,2]}
print(f"\nImage: span of column 1 = span{{[1, 2]}}")
print(f"dim(image) = {np.linalg.matrix_rank(A)}")

# Rank-Nullity Theorem: dim(ker) + dim(image) = n
n = A.shape[1]
print(f"\nRank-Nullity: dim(ker) + dim(image) = {ker.shape[1]} + {np.linalg.matrix_rank(A)} = {ker.shape[1] + np.linalg.matrix_rank(A)}")
print(f"n = {n}")
print(f"Verified: {ker.shape[1] + np.linalg.matrix_rank(A) == n}")
SOLUTION
}

# === Exercise 3: Composition of Transformations ===
# Problem: Compose a 90-degree rotation with a reflection across y-axis
# in R^2. Find the combined matrix and describe the result.
exercise_3() {
    echo "=== Exercise 3: Composition of Transformations ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

# 90-degree counter-clockwise rotation
theta = np.pi / 2
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

# Reflection across y-axis
F = np.array([[-1, 0],
              [0, 1]])

print(f"Rotation R (90 deg):\n{np.round(R, 4)}")
print(f"\nReflection F (y-axis):\n{F}")

# Compose: first rotate, then reflect -> T = F @ R
T = F @ R
print(f"\nT = F @ R (rotate then reflect):\n{np.round(T, 4)}")

# Apply to standard basis vectors
e1 = np.array([1, 0])
e2 = np.array([0, 1])
print(f"\ne1 -> {np.round(T @ e1, 4)}")
print(f"e2 -> {np.round(T @ e2, 4)}")

# This is a reflection across x = y line!
print(f"\ndet(T) = {np.linalg.det(T):.4f} (negative = includes reflection)")

# Compare: reflect then rotate -> T2 = R @ F
T2 = R @ F
print(f"\nT2 = R @ F (reflect then rotate):\n{np.round(T2, 4)}")
print(f"T == T2? {np.allclose(T, T2)} (order matters!)")
SOLUTION
}

# === Exercise 4: Rank-Nullity Theorem ===
# Problem: For A (5x3) with rank 2, determine dim(ker(A)) and dim(image(A)).
exercise_4() {
    echo "=== Exercise 4: Rank-Nullity Theorem ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np
from scipy.linalg import null_space

# Create a 5x3 matrix with rank 2
np.random.seed(42)
# Rank-2: third column is linear combination of first two
A = np.random.randn(5, 2) @ np.random.randn(2, 3)
# Ensure rank is exactly 2
print(f"A (5x3):\n{np.round(A, 4)}")
print(f"Rank: {np.linalg.matrix_rank(A)}")

# Rank-Nullity: dim(ker) + dim(image) = n (number of columns)
n = A.shape[1]
rank = np.linalg.matrix_rank(A)
nullity = n - rank

print(f"\nn (columns) = {n}")
print(f"dim(image) = rank = {rank}")
print(f"dim(ker) = nullity = n - rank = {nullity}")
print(f"Rank-Nullity: {rank} + {nullity} = {n}")

# Compute and verify
ker = null_space(A)
print(f"\nKernel basis ({ker.shape[1]} vector(s)):\n{np.round(ker, 4)}")
print(f"A @ ker = {np.round(A @ ker, 10).flatten()} (should be ~0)")
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 05: Linear Transformations"
echo "========================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
