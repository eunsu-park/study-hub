#!/bin/bash
# Exercises for Lesson 09: Orthogonality and Projections
# Topic: Linear_Algebra
# Solutions to practice problems from the lesson.

# === Exercise 1: Gram-Schmidt Process ===
# Problem: Apply Gram-Schmidt to {[1,1,1], [0,1,1], [0,0,1]} to obtain
# an orthonormal basis.
exercise_1() {
    echo "=== Exercise 1: Gram-Schmidt Process ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

v1 = np.array([1, 1, 1], dtype=float)
v2 = np.array([0, 1, 1], dtype=float)
v3 = np.array([0, 0, 1], dtype=float)

# Step 1: Normalize v1
u1 = v1 / np.linalg.norm(v1)

# Step 2: Orthogonalize v2 against u1, then normalize
w2 = v2 - np.dot(v2, u1) * u1
u2 = w2 / np.linalg.norm(w2)

# Step 3: Orthogonalize v3 against u1 and u2, then normalize
w3 = v3 - np.dot(v3, u1) * u1 - np.dot(v3, u2) * u2
u3 = w3 / np.linalg.norm(w3)

print(f"u1 = {np.round(u1, 4)}")
print(f"u2 = {np.round(u2, 4)}")
print(f"u3 = {np.round(u3, 4)}")

# Verify orthonormality
Q = np.column_stack([u1, u2, u3])
print(f"\nQ^T Q:\n{np.round(Q.T @ Q, 10)}")
print(f"Orthonormal: {np.allclose(Q.T @ Q, np.eye(3))}")

# Compare with np.linalg.qr
A = np.column_stack([v1, v2, v3])
Q_qr, R_qr = np.linalg.qr(A)
print(f"\nnp.linalg.qr Q:\n{np.round(Q_qr, 4)}")
SOLUTION
}

# === Exercise 2: QR Decomposition ===
# Problem: Compute the QR decomposition of A = [[1,1,0],[1,0,1],[0,1,1]]
# and solve Ax = [2, 3, 4] using QR.
exercise_2() {
    echo "=== Exercise 2: QR Decomposition ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

A = np.array([[1, 1, 0],
              [1, 0, 1],
              [0, 1, 1]], dtype=float)
b = np.array([2, 3, 4], dtype=float)

Q, R = np.linalg.qr(A)
print(f"Q:\n{np.round(Q, 4)}")
print(f"R:\n{np.round(R, 4)}")

# Solve via QR: Ax = b => QRx = b => Rx = Q^T b
Qtb = Q.T @ b
x = np.linalg.solve(R, Qtb)

print(f"\nQ^T b = {np.round(Qtb, 4)}")
print(f"Solution x = {np.round(x, 4)}")
print(f"Verification Ax = {np.round(A @ x, 10)}")
print(f"Match: {np.allclose(A @ x, b)}")
SOLUTION
}

# === Exercise 3: Orthogonal Projection ===
# Problem: Project b = [1, 2, 3, 4] onto the column space of
# A = [[1, 0], [1, 1], [0, 1], [0, 0]].
exercise_3() {
    echo "=== Exercise 3: Orthogonal Projection ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

A = np.array([[1, 0],
              [1, 1],
              [0, 1],
              [0, 0]], dtype=float)
b = np.array([1, 2, 3, 4], dtype=float)

# Projection matrix: P = A (A^T A)^{-1} A^T
ATA = A.T @ A
P = A @ np.linalg.inv(ATA) @ A.T

proj = P @ b
residual = b - proj

print(f"A:\n{A}")
print(f"b: {b}")
print(f"\nProjection P @ b: {np.round(proj, 4)}")
print(f"Residual b - Pb: {np.round(residual, 4)}")

# Verify residual is orthogonal to column space
print(f"\nA^T @ residual = {np.round(A.T @ residual, 10)} (should be 0)")

# Projection matrix properties
print(f"\nP^2 == P (idempotent): {np.allclose(P @ P, P)}")
print(f"P^T == P (symmetric): {np.allclose(P.T, P)}")
SOLUTION
}

# === Exercise 4: Least Squares via QR ===
# Problem: Fit y = a*sin(x) + b*cos(x) to noisy data using QR least squares.
exercise_4() {
    echo "=== Exercise 4: Least Squares via QR ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

np.random.seed(42)
x = np.linspace(0, 2*np.pi, 50)
y_true = 3 * np.sin(x) + 2 * np.cos(x)
y = y_true + np.random.randn(50) * 0.5

# Design matrix [sin(x), cos(x)]
A = np.column_stack([np.sin(x), np.cos(x)])

# QR least squares
Q, R = np.linalg.qr(A)
coeffs = np.linalg.solve(R, Q.T @ y)

print(f"True coefficients: a=3.0, b=2.0")
print(f"Estimated: a={coeffs[0]:.4f}, b={coeffs[1]:.4f}")

# Residual
residual = y - A @ coeffs
print(f"\nResidual norm: {np.linalg.norm(residual):.4f}")
print(f"Residual orthogonal to col(A): {np.allclose(A.T @ residual, 0, atol=1e-10)}")
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 09: Orthogonality and Projections"
echo "================================================================"
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
