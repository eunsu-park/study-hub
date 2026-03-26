#!/bin/bash
# Exercises for Lesson 18: Matrix Functions and Exponentials
# Topic: Linear_Algebra
# Solutions to practice problems from the lesson.

# === Exercise 1: Matrix Exponential ===
# Problem: Compute e^A for A = [[0, -pi/2], [pi/2, 0]] and interpret
# the result geometrically.
exercise_1() {
    echo "=== Exercise 1: Matrix Exponential ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np
from scipy.linalg import expm

A = np.array([[0, -np.pi/2],
              [np.pi/2, 0]])

# Matrix exponential
eA = expm(A)
print(f"A:\n{np.round(A, 4)}")
print(f"\ne^A:\n{np.round(eA, 4)}")

# A is skew-symmetric -> e^A is a rotation matrix
# A = theta * [[0, -1], [1, 0]] with theta = pi/2
# e^A = [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]
theta = np.pi / 2
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
print(f"\nExpected (rotation by {np.degrees(theta)} deg):\n{np.round(R, 4)}")
print(f"Match: {np.allclose(eA, R)}")

# Verify properties
print(f"\ndet(e^A) = {np.linalg.det(eA):.4f} (should be 1)")
print(f"(e^A)^T e^A = I: {np.allclose(eA.T @ eA, np.eye(2))}")
SOLUTION
}

# === Exercise 2: Matrix Power Series ===
# Problem: Compute e^A by truncating the power series at different orders
# and compare convergence.
exercise_2() {
    echo "=== Exercise 2: Matrix Power Series ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np
from scipy.linalg import expm

A = np.array([[1, 1],
              [0, 1]], dtype=float)

# e^A = sum_{k=0}^{inf} A^k / k!
exact = expm(A)
print(f"A:\n{A}")
print(f"Exact e^A:\n{np.round(exact, 6)}")

print(f"\n{'Terms':>6}  {'Max error':>12}")
print("-" * 22)

for n_terms in [1, 2, 3, 5, 10, 15, 20]:
    approx = np.zeros_like(A)
    A_power = np.eye(2)
    factorial = 1
    for k in range(n_terms):
        if k > 0:
            factorial *= k
        approx += A_power / factorial
        A_power = A_power @ A

    error = np.max(np.abs(approx - exact))
    print(f"{n_terms:6d}  {error:12.2e}")
SOLUTION
}

# === Exercise 3: Solving ODEs with Matrix Exponential ===
# Problem: Solve dx/dt = Ax where A = [[-1, 2], [0, -3]] with
# initial condition x(0) = [1, 1]. Find x(1).
exercise_3() {
    echo "=== Exercise 3: Solving ODEs with Matrix Exponential ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp

A = np.array([[-1, 2],
              [0, -3]], dtype=float)
x0 = np.array([1, 1], dtype=float)

# Analytical solution: x(t) = e^{At} x(0)
t = 1.0
x_exact = expm(A * t) @ x0
print(f"A:\n{A}")
print(f"x(0) = {x0}")
print(f"\nx({t}) = e^(A*{t}) @ x(0) = {np.round(x_exact, 6)}")

# Verify with numerical ODE solver
def ode_rhs(t, x):
    return A @ x

sol = solve_ivp(ode_rhs, [0, t], x0, method='RK45', rtol=1e-10)
x_numerical = sol.y[:, -1]
print(f"Numerical (RK45): {np.round(x_numerical, 6)}")
print(f"Match: {np.allclose(x_exact, x_numerical, atol=1e-6)}")

# Eigenvalue solution
evals, evecs = np.linalg.eig(A)
print(f"\nEigenvalues: {evals}")
print(f"System is stable (all eigenvalues negative real parts)")

# Solution at multiple times
print(f"\n{'t':>5}  {'x1':>10}  {'x2':>10}")
for t_val in [0, 0.25, 0.5, 0.75, 1.0, 2.0]:
    x_t = expm(A * t_val) @ x0
    print(f"{t_val:5.2f}  {x_t[0]:10.4f}  {x_t[1]:10.4f}")
SOLUTION
}

# === Exercise 4: Cayley-Hamilton Theorem ===
# Problem: Verify the Cayley-Hamilton theorem for A = [[1,2],[3,4]]:
# A satisfies its own characteristic equation.
exercise_4() {
    echo "=== Exercise 4: Cayley-Hamilton Theorem ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

A = np.array([[1, 2],
              [3, 4]])

# Characteristic polynomial: det(A - lambda I) = lambda^2 - 5*lambda - 2
# p(lambda) = lambda^2 - tr(A)*lambda + det(A)
tr_A = np.trace(A)
det_A = np.linalg.det(A)
print(f"A:\n{A}")
print(f"tr(A) = {tr_A}")
print(f"det(A) = {det_A}")
print(f"Characteristic polynomial: lambda^2 - {tr_A}*lambda + {det_A}")

# Cayley-Hamilton: p(A) = A^2 - tr(A)*A + det(A)*I = 0
A2 = A @ A
p_A = A2 - tr_A * A + det_A * np.eye(2)
print(f"\np(A) = A^2 - {tr_A}*A + {det_A}*I:")
print(np.round(p_A, 10))
print(f"p(A) == 0: {np.allclose(p_A, 0)}")

# Use Cayley-Hamilton to express A^{-1}:
# A^2 - 5A - 2I = 0 => A^{-1} = (A - 5I) / (-2)
A_inv_CH = (A - tr_A * np.eye(2)) / (-det_A)
A_inv = np.linalg.inv(A)
print(f"\nA^(-1) via Cayley-Hamilton:\n{np.round(A_inv_CH, 4)}")
print(f"A^(-1) via numpy:\n{np.round(A_inv, 4)}")
print(f"Match: {np.allclose(A_inv_CH, A_inv)}")
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 18: Matrix Functions and Exponentials"
echo "===================================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
