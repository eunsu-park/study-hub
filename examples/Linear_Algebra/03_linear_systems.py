"""
Solving Linear Systems

Demonstrates methods for solving Ax = b:
- Direct solution with np.linalg.solve
- LU decomposition with scipy.linalg.lu
- Gaussian elimination step-by-step
- Overdetermined systems with least squares
- Underdetermined systems with minimum-norm solution
- Condition number and numerical stability

Dependencies: numpy, scipy
"""

import numpy as np
from scipy import linalg


def direct_solve():
    """Solve Ax = b using np.linalg.solve."""
    print("=" * 60)
    print("DIRECT SOLUTION: np.linalg.solve")
    print("=" * 60)

    # Simple 3x3 system
    A = np.array([[2, 1, -1],
                  [-3, -1, 2],
                  [-2, 1, 2]])
    b = np.array([8, -11, -3])

    x = np.linalg.solve(A, b)
    print(f"\nA:\n{A}")
    print(f"b: {b}")
    print(f"x: {x}")
    print(f"Verification Ax: {A @ x}")
    print(f"||Ax - b|| = {np.linalg.norm(A @ x - b):.2e}")

    # Multiple right-hand sides
    print("\n--- Multiple Right-Hand Sides ---")
    B = np.array([[8, 1],
                  [-11, 5],
                  [-3, -2]])
    X = np.linalg.solve(A, B)
    print(f"B (two RHS columns):\n{B}")
    print(f"X (solutions):\n{X}")
    print(f"Verification AX:\n{A @ X}")


def lu_decomposition():
    """Demonstrate LU decomposition for solving linear systems."""
    print("\n" + "=" * 60)
    print("LU DECOMPOSITION")
    print("=" * 60)

    A = np.array([[2, 1, -1],
                  [-3, -1, 2],
                  [-2, 1, 2]], dtype=float)

    # Compute PA = LU
    P, L, U = linalg.lu(A)
    print(f"\nA:\n{A}")
    print(f"\nPermutation P:\n{P}")
    print(f"\nLower triangular L:\n{np.round(L, 4)}")
    print(f"\nUpper triangular U:\n{np.round(U, 4)}")
    print(f"\nP @ L @ U:\n{np.round(P @ L @ U, 10)}")
    print(f"P @ L @ U == A: {np.allclose(P @ L @ U, A)}")

    # Solve using LU factorization
    b = np.array([8, -11, -3], dtype=float)

    # Step 1: Solve Ly = P^T b (forward substitution)
    y = linalg.solve_triangular(L, P.T @ b, lower=True)
    print(f"\nForward substitution Ly = P^T b:")
    print(f"y = {y}")

    # Step 2: Solve Ux = y (back substitution)
    x = linalg.solve_triangular(U, y, lower=False)
    print(f"\nBack substitution Ux = y:")
    print(f"x = {x}")
    print(f"Verification: Ax = {A @ x}")

    # LU factorization object for repeated solves
    print("\n--- scipy.linalg.lu_factor for efficiency ---")
    lu, piv = linalg.lu_factor(A)
    x1 = linalg.lu_solve((lu, piv), b)
    b2 = np.array([1, 0, 0], dtype=float)
    x2 = linalg.lu_solve((lu, piv), b2)
    print(f"Solution for b1: {x1}")
    print(f"Solution for b2: {x2}")


def gaussian_elimination():
    """Step-by-step Gaussian elimination."""
    print("\n" + "=" * 60)
    print("GAUSSIAN ELIMINATION (STEP BY STEP)")
    print("=" * 60)

    # Augmented matrix [A | b]
    A = np.array([[2, 1, -1, 8],
                  [-3, -1, 2, -11],
                  [-2, 1, 2, -3]], dtype=float)

    print(f"\nAugmented matrix [A|b]:\n{A}")
    n = A.shape[0]

    # Forward elimination
    for col in range(n):
        # Partial pivoting
        max_row = col + np.argmax(np.abs(A[col:, col]))
        if max_row != col:
            A[[col, max_row]] = A[[max_row, col]]
            print(f"\nSwap rows {col} and {max_row}")

        pivot = A[col, col]
        print(f"\nPivot on element ({col},{col}) = {pivot:.4f}")

        # Eliminate below pivot
        for row in range(col + 1, n):
            factor = A[row, col] / pivot
            A[row] -= factor * A[col]
            print(f"  R{row} = R{row} - ({factor:.4f}) * R{col}")

        print(f"After step {col + 1}:\n{np.round(A, 4)}")

    # Back substitution
    print("\n--- Back Substitution ---")
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (A[i, -1] - np.dot(A[i, i+1:n], x[i+1:])) / A[i, i]
        print(f"x[{i}] = {x[i]:.4f}")

    print(f"\nSolution: x = {x}")


def overdetermined_systems():
    """Solve overdetermined systems (more equations than unknowns)."""
    print("\n" + "=" * 60)
    print("OVERDETERMINED SYSTEMS (LEAST SQUARES)")
    print("=" * 60)

    # Line fitting: y = ax + b through noisy data
    np.random.seed(42)
    x_data = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    y_data = 2 * x_data + 1 + np.random.randn(6) * 0.5

    # Set up Ax = b where A = [x, 1], x = [a, b]
    A = np.column_stack([x_data, np.ones_like(x_data)])
    b = y_data

    print(f"Data points: x = {x_data}, y = {np.round(y_data, 3)}")
    print(f"\nDesign matrix A:\n{A}")

    # Method 1: Normal equations
    x_normal = np.linalg.solve(A.T @ A, A.T @ b)
    print(f"\nNormal equation solution: a = {x_normal[0]:.4f}, b = {x_normal[1]:.4f}")

    # Method 2: np.linalg.lstsq
    x_lstsq, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    print(f"lstsq solution: a = {x_lstsq[0]:.4f}, b = {x_lstsq[1]:.4f}")

    # Residual analysis
    residual = b - A @ x_lstsq
    print(f"\nResidual vector: {np.round(residual, 4)}")
    print(f"||residual|| = {np.linalg.norm(residual):.4f}")
    print(f"Residuals sum of squares: {np.sum(residual**2):.4f}")

    # Verify residual is orthogonal to column space
    print(f"\nA^T @ residual = {np.round(A.T @ residual, 10)} (should be ~0)")


def underdetermined_systems():
    """Solve underdetermined systems (fewer equations than unknowns)."""
    print("\n" + "=" * 60)
    print("UNDERDETERMINED SYSTEMS (MINIMUM NORM)")
    print("=" * 60)

    # 2 equations, 4 unknowns -> infinitely many solutions
    A = np.array([[1, 2, 1, 0],
                  [0, 1, 1, 1]], dtype=float)
    b = np.array([4, 3], dtype=float)

    print(f"A (2x4):\n{A}")
    print(f"b: {b}")
    print(f"System has {A.shape[1] - np.linalg.matrix_rank(A)} free variables")

    # Minimum norm solution via pseudoinverse
    x_min_norm = np.linalg.pinv(A) @ b
    print(f"\nMinimum norm solution: {np.round(x_min_norm, 4)}")
    print(f"||x|| = {np.linalg.norm(x_min_norm):.4f}")
    print(f"Verification Ax: {np.round(A @ x_min_norm, 10)}")

    # Alternative solution (not minimum norm)
    # Set x3 = 0, x4 = 0 and solve 2x2 system
    x_alt = np.zeros(4)
    A_sub = A[:, :2]
    x_alt[:2] = np.linalg.solve(A_sub, b)
    print(f"\nAlternative solution (x3=x4=0): {x_alt}")
    print(f"||x_alt|| = {np.linalg.norm(x_alt):.4f}")
    print(f"Verification: Ax_alt = {A @ x_alt}")

    print(f"\nMinimum norm solution has smaller norm: "
          f"{np.linalg.norm(x_min_norm) < np.linalg.norm(x_alt)}")


def condition_number():
    """Demonstrate condition number and numerical stability."""
    print("\n" + "=" * 60)
    print("CONDITION NUMBER AND STABILITY")
    print("=" * 60)

    # Well-conditioned system
    A_good = np.array([[1, 0],
                       [0, 1]], dtype=float)
    print(f"Well-conditioned A:\n{A_good}")
    print(f"Condition number: {np.linalg.cond(A_good):.2f}")

    # Ill-conditioned system (Hilbert matrix)
    print("\n--- Hilbert Matrix (ill-conditioned) ---")
    for n in [3, 5, 8, 10]:
        H = np.array([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)])
        cond = np.linalg.cond(H)
        print(f"H_{n}: condition number = {cond:.2e}")

    # Effect on solution accuracy
    print("\n--- Effect on Solution Accuracy ---")
    n = 8
    H = np.array([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)])
    x_true = np.ones(n)
    b = H @ x_true

    x_computed = np.linalg.solve(H, b)
    error = np.linalg.norm(x_computed - x_true) / np.linalg.norm(x_true)
    print(f"H_{n} condition number: {np.linalg.cond(H):.2e}")
    print(f"Relative error in x: {error:.2e}")
    print(f"Lost ~{int(np.log10(np.linalg.cond(H)))} digits of accuracy")

    # Perturbed system
    print("\n--- Sensitivity to Perturbation ---")
    A = np.array([[1, 1],
                  [1, 1.0001]], dtype=float)
    b1 = np.array([2, 2.0001])
    b2 = np.array([2, 2.0002])

    x1 = np.linalg.solve(A, b1)
    x2 = np.linalg.solve(A, b2)
    print(f"cond(A) = {np.linalg.cond(A):.2f}")
    print(f"b perturbation: {np.linalg.norm(b2 - b1) / np.linalg.norm(b1):.6f}")
    print(f"x perturbation: {np.linalg.norm(x2 - x1) / np.linalg.norm(x1):.6f}")
    print(f"Amplification factor: {(np.linalg.norm(x2 - x1) / np.linalg.norm(x1)) / (np.linalg.norm(b2 - b1) / np.linalg.norm(b1)):.2f}")


if __name__ == "__main__":
    direct_solve()
    lu_decomposition()
    gaussian_elimination()
    overdetermined_systems()
    underdetermined_systems()
    condition_number()
    print("\nAll examples completed!")
