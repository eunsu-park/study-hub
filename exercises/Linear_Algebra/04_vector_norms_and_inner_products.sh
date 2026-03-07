#!/bin/bash
# Exercises for Lesson 04: Vector Norms and Inner Products
# Topic: Linear_Algebra
# Solutions to practice problems from the lesson.

# === Exercise 1: Norm Computation ===
# Problem: For v = [3, -4, 5, -2, 1], compute L1, L2, L3, and L-inf norms.
exercise_1() {
    echo "=== Exercise 1: Norm Computation ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

v = np.array([3, -4, 5, -2, 1])
print(f"v = {v}")
print(f"L1 norm:   {np.linalg.norm(v, 1)}")
print(f"L2 norm:   {np.linalg.norm(v, 2):.4f}")
print(f"L3 norm:   {np.linalg.norm(v, 3):.4f}")
print(f"L-inf norm:{np.linalg.norm(v, np.inf)}")

# Manual verification
print(f"\nManual L1:   |3|+|-4|+|5|+|-2|+|1| = {np.sum(np.abs(v))}")
print(f"Manual L2:   sqrt(9+16+25+4+1) = {np.sqrt(np.sum(v**2)):.4f}")
print(f"Manual L-inf: max(|v_i|) = {np.max(np.abs(v))}")
SOLUTION
}

# === Exercise 2: Cauchy-Schwarz Inequality ===
# Problem: Verify |<u,v>| <= ||u|| * ||v|| for u=[1,2,3] and v=[4,-1,2].
exercise_2() {
    echo "=== Exercise 2: Cauchy-Schwarz Inequality ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

u = np.array([1, 2, 3])
v = np.array([4, -1, 2])

inner = np.abs(np.dot(u, v))
norm_product = np.linalg.norm(u) * np.linalg.norm(v)

print(f"u = {u}, v = {v}")
print(f"|<u,v>| = |{np.dot(u, v)}| = {inner}")
print(f"||u|| * ||v|| = {np.linalg.norm(u):.4f} * {np.linalg.norm(v):.4f} = {norm_product:.4f}")
print(f"|<u,v>| <= ||u||*||v||: {inner <= norm_product + 1e-10}")

# Equality holds iff u and v are parallel
# cos(theta) = <u,v> / (||u|| ||v||)
cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
print(f"\ncos(theta) = {cos_theta:.4f}")
print(f"Angle: {np.degrees(np.arccos(cos_theta)):.2f} degrees")
print(f"Parallel (equality): {np.isclose(abs(cos_theta), 1)}")
SOLUTION
}

# === Exercise 3: Orthogonal Projection ===
# Problem: Project u = [3, 4, 5] onto the plane spanned by
# v1 = [1, 0, 0] and v2 = [0, 1, 0].
exercise_3() {
    echo "=== Exercise 3: Orthogonal Projection ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

u = np.array([3, 4, 5])
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])

# v1 and v2 are already orthonormal
# Projection = <u,v1>v1 + <u,v2>v2
proj = np.dot(u, v1) * v1 + np.dot(u, v2) * v2
print(f"u = {u}")
print(f"Projection onto xy-plane: {proj}")

# Residual (orthogonal component)
residual = u - proj
print(f"Residual: {residual}")
print(f"Residual orthogonal to v1: {np.isclose(np.dot(residual, v1), 0)}")
print(f"Residual orthogonal to v2: {np.isclose(np.dot(residual, v2), 0)}")

# Using projection matrix P = V(V^T V)^{-1} V^T
V = np.column_stack([v1, v2])
P = V @ np.linalg.inv(V.T @ V) @ V.T
proj_matrix = P @ u
print(f"\nProjection matrix P:\n{P}")
print(f"P @ u = {proj_matrix}")
print(f"P^2 == P: {np.allclose(P @ P, P)}")
SOLUTION
}

# === Exercise 4: Frobenius Norm ===
# Problem: Compute the Frobenius norm of A = [[1,2,3],[4,5,6]] and verify
# it equals sqrt(trace(A^T A)).
exercise_4() {
    echo "=== Exercise 4: Frobenius Norm ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]])

frob = np.linalg.norm(A, 'fro')
trace_method = np.sqrt(np.trace(A.T @ A))
element_method = np.sqrt(np.sum(A ** 2))

print(f"A:\n{A}")
print(f"||A||_F (numpy): {frob:.4f}")
print(f"sqrt(tr(A^T A)): {trace_method:.4f}")
print(f"sqrt(sum a_ij^2): {element_method:.4f}")
print(f"All equal: {np.isclose(frob, trace_method) and np.isclose(frob, element_method)}")

# Also equals sqrt(sum of singular values squared)
sigma = np.linalg.svd(A, compute_uv=False)
svd_method = np.sqrt(np.sum(sigma ** 2))
print(f"\nsqrt(sum sigma_i^2): {svd_method:.4f}")
print(f"Match: {np.isclose(frob, svd_method)}")
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 04: Vector Norms and Inner Products"
echo "=================================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
