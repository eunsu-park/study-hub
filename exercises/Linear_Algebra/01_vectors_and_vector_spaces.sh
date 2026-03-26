#!/bin/bash
# Exercises for Lesson 01: Vectors and Vector Spaces
# Topic: Linear_Algebra
# Solutions to practice problems from the lesson.

# === Exercise 1: Linear Independence ===
# Problem: Determine whether the vectors v1=[1,2,3], v2=[4,5,6], v3=[7,8,9]
# are linearly independent in R^3.
exercise_1() {
    echo "=== Exercise 1: Linear Independence ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
v3 = np.array([7, 8, 9])

# Stack as columns and check rank
A = np.column_stack([v1, v2, v3])
rank = np.linalg.matrix_rank(A)
det = np.linalg.det(A)

print(f"Matrix A:\n{A}")
print(f"Rank: {rank}")
print(f"Determinant: {det:.4f}")
print(f"Linearly independent: {rank == 3}")
# Result: rank=2, det=0. Vectors are linearly DEPENDENT.
# v3 = 2*v2 - v1, so v3 is in span(v1, v2).

# Verify the dependency
print(f"\n2*v2 - v1 = {2*v2 - v1}")
print(f"v3         = {v3}")
print(f"Match: {np.allclose(2*v2 - v1, v3)}")
SOLUTION
}

# === Exercise 2: Basis and Coordinates ===
# Problem: Given basis B = {b1=[1,0,1], b2=[0,1,1], b3=[1,1,0]},
# express v = [3, 5, 4] in basis B coordinates.
exercise_2() {
    echo "=== Exercise 2: Basis and Coordinates ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

b1 = np.array([1, 0, 1])
b2 = np.array([0, 1, 1])
b3 = np.array([1, 1, 0])
v = np.array([3, 5, 4])

# B matrix: basis vectors as columns
B = np.column_stack([b1, b2, b3])

# Solve B @ coords = v
coords = np.linalg.solve(B, v)
print(f"Basis matrix B:\n{B}")
print(f"v = {v}")
print(f"Coordinates in B: {coords}")
print(f"v = {coords[0]}*b1 + {coords[1]}*b2 + {coords[2]}*b3")

# Verify
v_check = coords[0]*b1 + coords[1]*b2 + coords[2]*b3
print(f"Reconstruction: {v_check}")
print(f"Correct: {np.allclose(v, v_check)}")
SOLUTION
}

# === Exercise 3: Subspace Verification ===
# Problem: Show that W = {(x, y, z) in R^3 : x + y + z = 0} is a subspace.
# Find a basis for W and its dimension.
exercise_3() {
    echo "=== Exercise 3: Subspace Verification ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

# W = {(x,y,z) : x + y + z = 0} => z = -x - y
# Parametric form: (x, y, -x-y) = x*(1,0,-1) + y*(0,1,-1)
# So W = span{(1,0,-1), (0,1,-1)}

w1 = np.array([1, 0, -1])
w2 = np.array([0, 1, -1])

# Verify they are in W (sum of components = 0)
print(f"w1 = {w1}, sum = {np.sum(w1)}")
print(f"w2 = {w2}, sum = {np.sum(w2)}")

# Check linear independence
W = np.column_stack([w1, w2])
print(f"\nRank of [w1, w2]: {np.linalg.matrix_rank(W)}")
print(f"Basis for W: {{w1, w2}}")
print(f"dim(W) = 2")

# Verify subspace properties with random elements
u = 3*w1 + 2*w2  # = (3, 2, -5)
v = -1*w1 + 4*w2  # = (-1, 4, -3)
print(f"\nu = {u}, sum = {np.sum(u)} (in W)")
print(f"v = {v}, sum = {np.sum(v)} (in W)")
print(f"u + v = {u+v}, sum = {np.sum(u+v)} (closure under addition)")
print(f"5*u = {5*u}, sum = {np.sum(5*u)} (closure under scalar mult)")
SOLUTION
}

# === Exercise 4: Span and Dimension ===
# Problem: Find the dimension of span{[1,2,0,1], [2,4,0,2], [0,1,1,0], [1,3,1,1]}
exercise_4() {
    echo "=== Exercise 4: Span and Dimension ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

v1 = np.array([1, 2, 0, 1])
v2 = np.array([2, 4, 0, 2])
v3 = np.array([0, 1, 1, 0])
v4 = np.array([1, 3, 1, 1])

A = np.column_stack([v1, v2, v3, v4])
rank = np.linalg.matrix_rank(A)

print(f"Vectors as columns:\n{A}")
print(f"Rank = {rank}")
print(f"dim(span) = {rank}")

# Note: v2 = 2*v1, and v4 = v1 + v3
# So only v1 and v3 are needed as basis
print(f"\nv2 == 2*v1: {np.allclose(v2, 2*v1)}")
print(f"v4 == v1 + v3: {np.allclose(v4, v1 + v3)}")
print(f"Basis: {{v1, v3}}")
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 01: Vectors and Vector Spaces"
echo "============================================================"
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
