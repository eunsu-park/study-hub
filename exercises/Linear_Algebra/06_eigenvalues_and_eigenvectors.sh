#!/bin/bash
# Exercises for Lesson 06: Eigenvalues and Eigenvectors
# Topic: Linear_Algebra
# Solutions to practice problems from the lesson.

# === Exercise 1: Eigenvalue Computation ===
# Problem: Find eigenvalues and eigenvectors of A = [[2, 1], [1, 2]]
# by solving the characteristic equation det(A - lambda I) = 0.
exercise_1() {
    echo "=== Exercise 1: Eigenvalue Computation ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

A = np.array([[2, 1],
              [1, 2]])

# Characteristic polynomial: det(A - lambda I) = 0
# (2-l)(2-l) - 1 = l^2 - 4l + 3 = (l-1)(l-3) = 0
# lambda = 1, 3
print("Characteristic polynomial: lambda^2 - 4*lambda + 3 = 0")
coeffs = [1, -np.trace(A), np.linalg.det(A)]
roots = np.roots(coeffs)
print(f"Roots: {roots}")

# NumPy eigendecomposition
eigenvalues, eigenvectors = np.linalg.eigh(A)
print(f"\nEigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")

# Verify Av = lambda v
for i in range(2):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    print(f"\nlambda={lam}: Av = {A @ v}, lambda*v = {lam * v}")
    print(f"Match: {np.allclose(A @ v, lam * v)}")
SOLUTION
}

# === Exercise 2: Diagonalization ===
# Problem: Diagonalize A = [[5, 4], [1, 2]] and compute A^10.
exercise_2() {
    echo "=== Exercise 2: Diagonalization ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

A = np.array([[5, 4],
              [1, 2]])

eigenvalues, P = np.linalg.eig(A)
D = np.diag(eigenvalues)
P_inv = np.linalg.inv(P)

print(f"A:\n{A}")
print(f"Eigenvalues: {eigenvalues}")
print(f"P:\n{np.round(P, 4)}")
print(f"D:\n{np.round(D, 4)}")
print(f"A == P D P^(-1): {np.allclose(A, P @ D @ P_inv)}")

# A^10 = P D^10 P^{-1}
n = 10
D_n = np.diag(eigenvalues ** n)
A_n = P @ D_n @ P_inv

print(f"\nA^{n} via diagonalization:\n{np.round(A_n, 2)}")
print(f"A^{n} via matrix_power:\n{np.round(np.linalg.matrix_power(A, n), 2)}")
print(f"Match: {np.allclose(A_n, np.linalg.matrix_power(A, n))}")
SOLUTION
}

# === Exercise 3: Spectral Theorem ===
# Problem: Verify the spectral theorem for the symmetric matrix
# A = [[3, 1, 1], [1, 3, 1], [1, 1, 3]].
exercise_3() {
    echo "=== Exercise 3: Spectral Theorem ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

A = np.array([[3, 1, 1],
              [1, 3, 1],
              [1, 1, 3]])

eigenvalues, Q = np.linalg.eigh(A)
Lambda = np.diag(eigenvalues)

print(f"Symmetric A:\n{A}")
print(f"Eigenvalues: {eigenvalues}")
print(f"Q (orthogonal eigenvectors):\n{np.round(Q, 4)}")

# Spectral theorem: A = Q Lambda Q^T
A_reconstructed = Q @ Lambda @ Q.T
print(f"\nQ Lambda Q^T:\n{np.round(A_reconstructed, 10)}")
print(f"A == Q Lambda Q^T: {np.allclose(A, A_reconstructed)}")

# Q is orthogonal
print(f"\nQ^T Q = I: {np.allclose(Q.T @ Q, np.eye(3))}")

# Eigenvalues are real (guaranteed for symmetric)
print(f"All eigenvalues real: {np.all(np.isreal(eigenvalues))}")

# Spectral decomposition: A = sum lambda_i * q_i * q_i^T
A_spectral = sum(eigenvalues[i] * np.outer(Q[:, i], Q[:, i]) for i in range(3))
print(f"Spectral sum matches: {np.allclose(A, A_spectral)}")
SOLUTION
}

# === Exercise 4: Power Iteration ===
# Problem: Implement power iteration to find the dominant eigenvalue of
# A = [[4, 1, 0], [1, 3, 1], [0, 1, 2]].
exercise_4() {
    echo "=== Exercise 4: Power Iteration ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

A = np.array([[4, 1, 0],
              [1, 3, 1],
              [0, 1, 2]], dtype=float)

np.random.seed(42)
v = np.random.randn(3)
v = v / np.linalg.norm(v)

# True eigenvalues for comparison
true_evals = np.linalg.eigvalsh(A)
dominant = np.max(true_evals)
print(f"True eigenvalues: {np.round(true_evals, 6)}")
print(f"Dominant: {dominant:.6f}\n")

for k in range(20):
    w = A @ v
    lam = np.dot(w, v)  # Rayleigh quotient
    v = w / np.linalg.norm(w)
    if k < 5 or k % 5 == 4:
        print(f"Iter {k+1:2d}: lambda = {lam:.8f}, error = {abs(lam - dominant):.2e}")

print(f"\nConverged eigenvalue: {lam:.8f}")
print(f"Eigenvector: {np.round(v, 4)}")
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 06: Eigenvalues and Eigenvectors"
echo "==============================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
