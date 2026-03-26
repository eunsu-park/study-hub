#!/bin/bash
# Exercises for Lesson 11: Quadratic Forms and Definiteness
# Topic: Linear_Algebra
# Solutions to practice problems from the lesson.

# === Exercise 1: Classifying Definiteness ===
# Problem: Classify the following matrices as positive definite, positive
# semidefinite, negative definite, or indefinite using eigenvalues.
exercise_1() {
    echo "=== Exercise 1: Classifying Definiteness ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

matrices = {
    "A": np.array([[2, -1], [-1, 2]]),
    "B": np.array([[1, 2], [2, 1]]),
    "C": np.array([[-3, 0], [0, -5]]),
    "D": np.array([[1, 0], [0, 0]]),
}

for name, M in matrices.items():
    evals = np.linalg.eigvalsh(M)
    if np.all(evals > 0):
        classification = "Positive definite"
    elif np.all(evals >= 0):
        classification = "Positive semidefinite"
    elif np.all(evals < 0):
        classification = "Negative definite"
    elif np.all(evals <= 0):
        classification = "Negative semidefinite"
    else:
        classification = "Indefinite"

    print(f"{name}: eigenvalues = {np.round(evals, 4)} -> {classification}")
SOLUTION
}

# === Exercise 2: Sylvester's Criterion ===
# Problem: Use leading principal minors to check if
# A = [[2,1,0],[1,3,1],[0,1,4]] is positive definite.
exercise_2() {
    echo "=== Exercise 2: Sylvester's Criterion ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

A = np.array([[2, 1, 0],
              [1, 3, 1],
              [0, 1, 4]])

# Leading principal minors
d1 = A[0, 0]
d2 = np.linalg.det(A[:2, :2])
d3 = np.linalg.det(A)

print(f"A:\n{A}")
print(f"\nLeading principal minors:")
print(f"  d1 = {d1}")
print(f"  d2 = {d2:.4f}")
print(f"  d3 = {d3:.4f}")
print(f"\nAll positive: {d1 > 0 and d2 > 0 and d3 > 0}")
print(f"=> A is positive definite")

# Verify with eigenvalues
evals = np.linalg.eigvalsh(A)
print(f"\nEigenvalues: {np.round(evals, 4)}")
print(f"All positive: {np.all(evals > 0)}")
SOLUTION
}

# === Exercise 3: Quadratic Form Evaluation ===
# Problem: For Q(x) = x^T A x with A = [[3,1],[1,2]], evaluate Q at
# several points and verify the sign matches the definiteness.
exercise_3() {
    echo "=== Exercise 3: Quadratic Form Evaluation ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

A = np.array([[3, 1],
              [1, 2]])

# Q(x) = 3x1^2 + 2x1*x2 + 2x2^2
evals = np.linalg.eigvalsh(A)
print(f"A:\n{A}")
print(f"Eigenvalues: {evals}")
print(f"Positive definite: {np.all(evals > 0)}")
print(f"\n=> Q(x) > 0 for all nonzero x\n")

test_points = [
    np.array([1, 0]),
    np.array([0, 1]),
    np.array([1, 1]),
    np.array([-2, 3]),
    np.array([0.1, -0.5]),
]

for x in test_points:
    Q = x @ A @ x
    print(f"Q({x}) = {Q:.4f} > 0: {Q > 0}")

# At eigenvectors, Q = lambda * ||v||^2
_, evecs = np.linalg.eigh(A)
for i in range(2):
    v = evecs[:, i]
    Q = v @ A @ v
    print(f"\nQ(eigenvec_{i+1}) = {Q:.4f} = lambda_{i+1} * ||v||^2 = {evals[i]:.4f}")
SOLUTION
}

# === Exercise 4: Optimization Connection ===
# Problem: Show that x = A^{-1} b minimizes f(x) = (1/2) x^T A x - b^T x
# when A is positive definite.
exercise_4() {
    echo "=== Exercise 4: Optimization Connection ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

A = np.array([[4, 1],
              [1, 3]], dtype=float)
b = np.array([1, 2], dtype=float)

# Minimum at x* = A^{-1} b (gradient = Ax - b = 0)
x_star = np.linalg.solve(A, b)
f_star = 0.5 * x_star @ A @ x_star - b @ x_star

print(f"A (SPD):\n{A}")
print(f"b: {b}")
print(f"x* = A^(-1) b = {np.round(x_star, 4)}")
print(f"f(x*) = {f_star:.4f}")

# Verify gradient is zero at x*
grad = A @ x_star - b
print(f"\nGradient at x*: {np.round(grad, 10)} (should be 0)")

# Compare f at other points
np.random.seed(42)
for _ in range(5):
    x = np.random.randn(2)
    f_x = 0.5 * x @ A @ x - b @ x
    print(f"f({np.round(x, 2)}) = {f_x:.4f} >= {f_star:.4f}: {f_x >= f_star - 1e-10}")
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 11: Quadratic Forms and Definiteness"
echo "==================================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
