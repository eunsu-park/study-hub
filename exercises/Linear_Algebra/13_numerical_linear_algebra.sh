#!/bin/bash
# Exercises for Lesson 13: Numerical Linear Algebra
# Topic: Linear_Algebra
# Solutions to practice problems from the lesson.

# === Exercise 1: Floating-Point Sensitivity ===
# Problem: Demonstrate how floating-point arithmetic affects the result
# of summing 1/3 ten million times vs computing 10000000/3.
exercise_1() {
    echo "=== Exercise 1: Floating-Point Sensitivity ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

# Naive summation
n = 10_000_000
naive_sum = sum(1/3 for _ in range(100))  # smaller example
exact = 100 / 3
error = abs(naive_sum - exact)

print(f"Sum of 1/3 (100 times): {naive_sum}")
print(f"Exact 100/3: {exact}")
print(f"Error: {error:.2e}")

# Machine epsilon
eps = np.finfo(np.float64).eps
print(f"\nMachine epsilon (float64): {eps:.2e}")
print(f"1.0 + eps == 1.0: {1.0 + eps == 1.0}")
print(f"1.0 + eps/2 == 1.0: {1.0 + eps/2 == 1.0}")

# Catastrophic cancellation
a = 1e15 + 1.0
b = 1e15
print(f"\n(1e15 + 1.0) - 1e15 = {a - b}")
print(f"Expected: 1.0")
# In this case exact, but shows precision limits
SOLUTION
}

# === Exercise 2: Condition Number Analysis ===
# Problem: Compare the solution accuracy for well-conditioned and
# ill-conditioned systems of the same size.
exercise_2() {
    echo "=== Exercise 2: Condition Number Analysis ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

np.random.seed(42)

# Well-conditioned system
A_good = np.eye(5) + 0.1 * np.random.randn(5, 5)
A_good = A_good @ A_good.T  # Make symmetric positive definite

# Ill-conditioned system (Hilbert matrix)
A_bad = np.array([[1.0/(i+j+1) for j in range(5)] for i in range(5)])

x_true = np.ones(5)

for name, A in [("Well-conditioned", A_good), ("Ill-conditioned", A_bad)]:
    b = A @ x_true
    x_computed = np.linalg.solve(A, b)
    error = np.linalg.norm(x_computed - x_true) / np.linalg.norm(x_true)
    cond = np.linalg.cond(A)

    print(f"{name}:")
    print(f"  cond(A) = {cond:.2e}")
    print(f"  Relative error = {error:.2e}")
    print(f"  Lost ~{int(np.log10(cond))} digits of accuracy")
    print()
SOLUTION
}

# === Exercise 3: Sparse vs Dense Solve ===
# Problem: Create a 1000x1000 tridiagonal system and compare solve times
# for sparse vs dense solvers.
exercise_3() {
    echo "=== Exercise 3: Sparse vs Dense Solve ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import time

n = 1000

# Tridiagonal: -1 on sub/super diag, 2 on main diagonal
A_sparse = sparse.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n), format='csc')
A_dense = A_sparse.toarray()
b = np.ones(n)

# Sparse solve
start = time.perf_counter()
x_sparse = spsolve(A_sparse, b)
t_sparse = (time.perf_counter() - start) * 1000

# Dense solve
start = time.perf_counter()
x_dense = np.linalg.solve(A_dense, b)
t_dense = (time.perf_counter() - start) * 1000

print(f"System size: {n}x{n}")
print(f"Non-zeros: {A_sparse.nnz} out of {n*n} ({A_sparse.nnz/n**2*100:.2f}%)")
print(f"\nSparse solve: {t_sparse:.2f} ms")
print(f"Dense solve:  {t_dense:.2f} ms")
print(f"Speedup: {t_dense/t_sparse:.1f}x")
print(f"\nMemory:")
print(f"  Dense: {n*n*8/1024:.1f} KB")
print(f"  Sparse: ~{A_sparse.nnz*12/1024:.1f} KB")
print(f"Solutions match: {np.allclose(x_sparse, x_dense)}")
SOLUTION
}

# === Exercise 4: Iterative vs Direct ===
# Problem: Compare conjugate gradient (iterative) with direct solve
# for a large SPD system.
exercise_4() {
    echo "=== Exercise 4: Iterative vs Direct ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg, spsolve
import time

n = 2000
A = sparse.diags([-1, 4, -1], [-1, 0, 1], shape=(n, n), format='csr', dtype=float)
b = np.random.randn(n)

# Direct solve
start = time.perf_counter()
x_direct = spsolve(A.tocsc(), b)
t_direct = (time.perf_counter() - start) * 1000

# CG solve
start = time.perf_counter()
x_cg, info = cg(A, b, tol=1e-10)
t_cg = (time.perf_counter() - start) * 1000

print(f"System size: {n}x{n}")
print(f"Direct solve: {t_direct:.2f} ms")
print(f"CG solve:     {t_cg:.2f} ms (converged: {info == 0})")
print(f"||x_direct - x_cg||: {np.linalg.norm(x_direct - x_cg):.2e}")
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 13: Numerical Linear Algebra"
echo "==========================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
