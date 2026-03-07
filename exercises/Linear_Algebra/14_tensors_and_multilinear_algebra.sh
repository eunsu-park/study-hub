#!/bin/bash
# Exercises for Lesson 14: Tensors and Multilinear Algebra
# Topic: Linear_Algebra
# Solutions to practice problems from the lesson.

# === Exercise 1: Einsum Operations ===
# Problem: Implement the following operations using np.einsum:
# (a) trace, (b) matrix multiply, (c) batch outer product, (d) tensor contraction.
exercise_1() {
    echo "=== Exercise 1: Einsum Operations ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
v = np.array([1, 2])

# (a) Trace
tr = np.einsum('ii', A)
print(f"Trace: einsum('ii', A) = {tr}, np.trace = {np.trace(A)}")

# (b) Matrix multiply
C = np.einsum('ij,jk->ik', A, B)
print(f"A @ B: einsum = \n{C}")
print(f"Match: {np.allclose(C, A @ B)}")

# (c) Batch outer product
u = np.random.randn(3, 4)
w = np.random.randn(3, 5)
outer = np.einsum('bi,bj->bij', u, w)
print(f"\nBatch outer: {u.shape} x {w.shape} -> {outer.shape}")

# (d) Tensor contraction
T = np.random.randn(2, 3, 4)
v3 = np.random.randn(4)
result = np.einsum('ijk,k->ij', T, v3)
print(f"Tensor contraction: {T.shape} x {v3.shape} -> {result.shape}")
SOLUTION
}

# === Exercise 2: Kronecker Product ===
# Problem: Compute the Kronecker product of A = [[1,2],[3,4]] and I_2,
# then verify its eigenvalues.
exercise_2() {
    echo "=== Exercise 2: Kronecker Product ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

A = np.array([[1, 2], [3, 4]])
I = np.eye(2)

K = np.kron(A, I)
print(f"A:\n{A}")
print(f"A kron I_2:\n{K}")

# Eigenvalues of A kron I = eigenvalues of A, each repeated
eig_A = np.linalg.eigvals(A)
eig_K = np.linalg.eigvals(K)
print(f"\nEigenvalues of A: {np.round(np.sort(eig_A), 4)}")
print(f"Eigenvalues of A kron I: {np.round(np.sort(eig_K), 4)}")
print(f"Each eigenvalue of A repeated 2 times")

# Property: (A kron I)(I kron B) = A kron B
B = np.array([[5, 6], [7, 8]])
lhs = np.kron(A, I) @ np.kron(I, B)
rhs = np.kron(A, B)
print(f"\n(A kron I)(I kron B) == A kron B: {np.allclose(lhs, rhs)}")
SOLUTION
}

# === Exercise 3: Tensor Reshaping ===
# Problem: Reshape a (2,3,4) tensor into a matrix by unfolding along
# each mode and verify the dimensions.
exercise_3() {
    echo "=== Exercise 3: Tensor Reshaping ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

T = np.arange(24).reshape(2, 3, 4)
print(f"Tensor shape: {T.shape}")
print(f"T:\n{T}\n")

# Mode-0 unfolding: (2, 3*4) = (2, 12)
mode0 = T.reshape(2, -1)
print(f"Mode-0 unfolding: {mode0.shape}")
print(f"{mode0}\n")

# Mode-1 unfolding: (3, 2*4) = (3, 8)
mode1 = T.transpose(1, 0, 2).reshape(3, -1)
print(f"Mode-1 unfolding: {mode1.shape}")
print(f"{mode1}\n")

# Mode-2 unfolding: (4, 2*3) = (4, 6)
mode2 = T.transpose(2, 0, 1).reshape(4, -1)
print(f"Mode-2 unfolding: {mode2.shape}")
print(f"{mode2}")

# Verify: can reconstruct tensor from each unfolding
T_recon0 = mode0.reshape(2, 3, 4)
T_recon1 = mode1.reshape(3, 2, 4).transpose(1, 0, 2)
T_recon2 = mode2.reshape(4, 2, 3).transpose(1, 2, 0)
print(f"\nReconstruction from mode-0: {np.allclose(T, T_recon0)}")
print(f"Reconstruction from mode-1: {np.allclose(T, T_recon1)}")
print(f"Reconstruction from mode-2: {np.allclose(T, T_recon2)}")
SOLUTION
}

# === Exercise 4: Broadcasting ===
# Problem: Use broadcasting to subtract column means from a matrix
# and row means from a matrix, explaining the shapes.
exercise_4() {
    echo "=== Exercise 4: Broadcasting ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]], dtype=float)

# Column means (mean of each column)
col_means = np.mean(X, axis=0)  # shape (3,)
X_centered_cols = X - col_means  # (3,3) - (3,) broadcasts along rows

# Row means (mean of each row)
row_means = np.mean(X, axis=1, keepdims=True)  # shape (3,1)
X_centered_rows = X - row_means  # (3,3) - (3,1) broadcasts along cols

print(f"X:\n{X}")
print(f"\nColumn means: {col_means} shape={col_means.shape}")
print(f"Column-centered:\n{X_centered_cols}")
print(f"Verify: column means are 0: {np.allclose(X_centered_cols.mean(axis=0), 0)}")

print(f"\nRow means: {row_means.flatten()} shape={row_means.shape}")
print(f"Row-centered:\n{X_centered_rows}")
print(f"Verify: row means are 0: {np.allclose(X_centered_rows.mean(axis=1), 0)}")
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 14: Tensors and Multilinear Algebra"
echo "=================================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
