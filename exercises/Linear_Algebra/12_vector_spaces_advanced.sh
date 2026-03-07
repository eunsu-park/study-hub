#!/bin/bash
# Exercises for Lesson 12: Advanced Vector Spaces
# Topic: Linear_Algebra
# Solutions to practice problems from the lesson.

# === Exercise 1: Direct Sum ===
# Problem: Show that R^3 = V + W where V = span{[1,0,0], [0,1,0]} and
# W = span{[0,0,1]} is a direct sum.
exercise_1() {
    echo "=== Exercise 1: Direct Sum ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

# V = span of first two standard basis vectors (xy-plane)
# W = span of third standard basis vector (z-axis)
# V + W is direct sum iff V intersection W = {0}

# Any vector in R^3 can be uniquely decomposed as v + w
# where v is in V and w is in W
x = np.array([3, 5, 7])
v_part = np.array([x[0], x[1], 0])  # projection onto V
w_part = np.array([0, 0, x[2]])      # projection onto W

print(f"x = {x}")
print(f"v-component (in V): {v_part}")
print(f"w-component (in W): {w_part}")
print(f"v + w = {v_part + w_part}")
print(f"Decomposition unique: True")

# Verify dim(V) + dim(W) = dim(R^3)
dim_V, dim_W = 2, 1
print(f"\ndim(V) = {dim_V}, dim(W) = {dim_W}")
print(f"dim(V) + dim(W) = {dim_V + dim_W} = dim(R^3)")

# Check intersection is trivial
B_total = np.column_stack([[1,0,0], [0,1,0], [0,0,1]])
print(f"Combined basis rank: {np.linalg.matrix_rank(B_total)} = 3")
print(f"=> Direct sum R^3 = V (+) W")
SOLUTION
}

# === Exercise 2: Dual Space ===
# Problem: For V = R^3 with standard basis, construct the dual basis
# and evaluate dual vectors on a given vector.
exercise_2() {
    echo "=== Exercise 2: Dual Space ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

# Standard basis for R^3
e1, e2, e3 = np.eye(3)

# Dual basis: f_i(e_j) = delta_{ij}
# For standard basis, dual vectors are just the standard basis itself
# f_i(v) = v_i (the i-th coordinate function)

v = np.array([3, -1, 4])
print(f"v = {v}")
print(f"f1(v) = {np.dot(e1, v)} (first coordinate)")
print(f"f2(v) = {np.dot(e2, v)} (second coordinate)")
print(f"f3(v) = {np.dot(e3, v)} (third coordinate)")

# Non-standard basis example
b1 = np.array([1, 1, 0])
b2 = np.array([1, 0, 1])
b3 = np.array([0, 1, 1])
B = np.column_stack([b1, b2, b3])

# Dual basis: rows of B^{-1}
B_inv = np.linalg.inv(B)
print(f"\nBasis B:\n{B}")
print(f"Dual basis (rows of B^(-1)):\n{np.round(B_inv, 4)}")

# Verify: dual_i(b_j) = delta_{ij}
for i in range(3):
    for j in range(3):
        val = np.dot(B_inv[i], B[:, j])
        print(f"  f{i+1}(b{j+1}) = {val:.0f}", end="")
    print()
SOLUTION
}

# === Exercise 3: Function Spaces ===
# Problem: Show that {1, x, x^2} forms a basis for the space of
# polynomials of degree <= 2 by expressing p(x) = 3x^2 - 2x + 5.
exercise_3() {
    echo "=== Exercise 3: Function Spaces ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

# Polynomials of degree <= 2: P_2 = span{1, x, x^2}
# p(x) = 5*1 + (-2)*x + 3*x^2
# Coordinates in basis {1, x, x^2}: [5, -2, 3]

print("Basis: {1, x, x^2}")
print("p(x) = 3x^2 - 2x + 5")
print("Coordinates: [5, -2, 3]")

# Evaluate polynomial using NumPy
coeffs = [3, -2, 5]  # highest degree first for np.polyval
x_vals = np.array([0, 1, 2, -1])
p_vals = np.polyval(coeffs, x_vals)

print(f"\nEvaluation:")
for x, p in zip(x_vals, p_vals):
    manual = 3*x**2 - 2*x + 5
    print(f"  p({x}) = {p} (manual: {manual})")

# Inner product on P_2: <p, q> = integral_0^1 p(x)q(x) dx
# Gram matrix for basis {1, x, x^2}:
# G[i,j] = integral_0^1 x^i * x^j dx = 1/(i+j+1)
G = np.array([[1/(i+j+1) for j in range(3)] for i in range(3)])
print(f"\nGram matrix (inner products):\n{np.round(G, 4)}")
print(f"Positive definite: {np.all(np.linalg.eigvalsh(G) > 0)}")
SOLUTION
}

# === Exercise 4: Change of Basis ===
# Problem: Given bases B = {[1,1],[0,1]} and C = {[1,0],[1,1]},
# find the change-of-basis matrix from B to C.
exercise_4() {
    echo "=== Exercise 4: Change of Basis ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

# Basis B
B = np.array([[1, 0],
              [1, 1]])

# Basis C
C = np.array([[1, 1],
              [0, 1]])

# Change of basis from B to C: P_{C<-B} = C^{-1} @ B
P_CB = np.linalg.inv(C) @ B
print(f"B:\n{B}")
print(f"C:\n{C}")
print(f"\nChange of basis P (B -> C):\n{P_CB}")

# Test: vector v with B-coordinates [2, 3]
v_B = np.array([2, 3])
v_standard = B @ v_B  # Convert to standard basis
v_C = P_CB @ v_B       # Convert B-coords to C-coords

print(f"\nv in B-coords: {v_B}")
print(f"v in standard: {v_standard}")
print(f"v in C-coords: {v_C}")

# Verify: C @ v_C should give same standard vector
v_check = C @ v_C
print(f"C @ v_C = {v_check}")
print(f"Match: {np.allclose(v_standard, v_check)}")
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 12: Advanced Vector Spaces"
echo "========================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
