"""
Exercises for Lesson 01: Vectors and Matrices
Topic: Math_for_AI

Solutions to practice problems from the lesson.
"""

import numpy as np


# === Exercise 1: Vector Spaces and Basis ===
# Problem: Determine whether the following vectors form a basis for R^3.
# If they do, express vector v = [1, 2, 3]^T in this basis.
# b1 = [1, 0, 1]^T, b2 = [0, 1, 1]^T, b3 = [1, 1, 0]^T

def exercise_1():
    """Check if vectors form a basis and express v in that basis."""
    b1 = np.array([1, 0, 1])
    b2 = np.array([0, 1, 1])
    b3 = np.array([1, 1, 0])
    v = np.array([1, 2, 3])

    # Form matrix B with basis vectors as columns
    B = np.column_stack([b1, b2, b3])

    # Check if they form a basis: matrix must be invertible (rank = 3)
    rank = np.linalg.matrix_rank(B)
    det = np.linalg.det(B)
    print(f"Basis vectors as columns:\n{B}")
    print(f"Rank: {rank} (need 3 for basis)")
    print(f"Determinant: {det:.4f} (non-zero means invertible)")
    print(f"Forms a basis for R^3: {rank == 3}")

    if rank == 3:
        # Solve B @ coords = v for coordinates in the new basis
        coords = np.linalg.solve(B, v)
        print(f"\nVector v = {v}")
        print(f"Coordinates in new basis: {coords}")
        print(f"v = {coords[0]:.4f}*b1 + {coords[1]:.4f}*b2 + {coords[2]:.4f}*b3")

        # Verify reconstruction
        v_reconstructed = coords[0] * b1 + coords[1] * b2 + coords[2] * b3
        print(f"Reconstruction: {v_reconstructed}")
        print(f"Correct: {np.allclose(v, v_reconstructed)}")


# === Exercise 2: Linear Transformation Visualization ===
# Problem: Find and visualize the final transformation matrix when applying:
# 1. 45 degree rotation
# 2. Scale by 2 in x direction
# 3. Reflection about x-axis

def exercise_2():
    """Compose rotation, scaling, and reflection transformations."""
    # 1. 45 degree rotation
    theta = np.pi / 4
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    print("Rotation (45 deg):")
    print(R)

    # 2. Scale by 2 in x direction
    S = np.array([
        [2, 0],
        [0, 1]
    ])
    print("\nScaling (2x in x):")
    print(S)

    # 3. Reflection about x-axis
    F = np.array([
        [1,  0],
        [0, -1]
    ])
    print("\nReflection (x-axis):")
    print(F)

    # Combined: apply in order R -> S -> F
    # Final = F @ S @ R (rightmost applied first)
    T = F @ S @ R
    print(f"\nCombined transformation T = F @ S @ R:")
    print(T)

    # Verify by applying to unit vectors
    e1 = np.array([1, 0])
    e2 = np.array([0, 1])
    print(f"\ne1 -> {T @ e1}")
    print(f"e2 -> {T @ e2}")

    # Apply to unit square vertices
    square = np.array([[0, 1, 1, 0],
                       [0, 0, 1, 1]])
    transformed = T @ square
    print(f"\nOriginal square corners:\n{square.T}")
    print(f"Transformed corners:\n{transformed.T}")


# === Exercise 3: Projection Matrix ===
# Problem: Find the projection matrix P onto the xy-plane (z=0 plane).
# P is 3x3 such that Pv projects v onto the xy-plane.

def exercise_3():
    """Find and verify the projection matrix onto the xy-plane."""
    # Projection onto xy-plane simply zeroes out the z-component
    P = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ])
    print("Projection matrix onto xy-plane:")
    print(P)

    # Verify properties
    # 1. P^2 = P (idempotent)
    print(f"\nP^2 = P (idempotent): {np.allclose(P @ P, P)}")

    # 2. P^T = P (symmetric)
    print(f"P^T = P (symmetric): {np.allclose(P.T, P)}")

    # 3. Test with several vectors
    test_vectors = [
        np.array([1, 2, 3]),
        np.array([0, 0, 5]),
        np.array([4, -1, 0]),
    ]

    for v in test_vectors:
        proj = P @ v
        print(f"\nv = {v}")
        print(f"Pv = {proj} (z-component zeroed out)")
        # Verify the projected vector is on the xy-plane
        print(f"On xy-plane (z=0): {proj[2] == 0}")


# === Exercise 4: Least Squares Problem ===
# Problem: Find the least squares solution to the overdetermined system:
# [[1,1],[1,2],[1,3],[1,4]] @ [x1,x2]^T = [2,3,5,6]^T

def exercise_4():
    """Solve an overdetermined system using least squares."""
    A = np.array([
        [1, 1],
        [1, 2],
        [1, 3],
        [1, 4]
    ])
    b = np.array([2, 3, 5, 6])

    # Method 1: Normal equation x = (A^T A)^{-1} A^T b
    ATA = A.T @ A
    ATb = A.T @ b
    x_normal = np.linalg.solve(ATA, ATb)
    print("Normal equation solution:", x_normal)

    # Method 2: np.linalg.lstsq
    x_lstsq, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    print("lstsq solution:", x_lstsq)

    # Method 3: QR decomposition
    Q, R = np.linalg.qr(A)
    x_qr = np.linalg.solve(R[:2, :], (Q.T @ b)[:2])
    print("QR solution:", x_qr)

    # Verify all methods agree
    print(f"\nAll methods agree: {np.allclose(x_normal, x_lstsq) and np.allclose(x_lstsq, x_qr)}")

    # Compute residual
    residual = b - A @ x_lstsq
    print(f"\nResidual vector: {residual}")
    print(f"Residual norm: {np.linalg.norm(residual):.6f}")

    # Interpretation: y = x1 + x2 * t (linear fit)
    print(f"\nLinear fit: y = {x_lstsq[0]:.4f} + {x_lstsq[1]:.4f} * t")


# === Exercise 5: ML Application ===
# Problem: Linear regression with 100 samples, 5 features.
# 1. Specify dimensions of X and w
# 2. Dimension of y_hat = Xw
# 3. Verify dimensions in normal equation w = (X^T X)^{-1} X^T y

def exercise_5():
    """Verify dimensions in linear regression operations."""
    n_samples = 100
    n_features = 5

    # Create synthetic data
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    true_w = np.array([2.0, -1.0, 0.5, 3.0, -0.5])
    y = X @ true_w + np.random.randn(n_samples) * 0.1

    print("=== Dimension Analysis ===")
    print(f"1. Data matrix X: {X.shape} ({n_samples} x {n_features})")
    print(f"   Weight vector w: ({n_features},) or ({n_features}, 1)")
    print()

    # 2. Predicted values
    y_hat = X @ true_w
    print(f"2. Predicted y_hat = Xw: {y_hat.shape} ({n_samples},)")
    print()

    # 3. Normal equation: w = (X^T X)^{-1} X^T y
    XTX = X.T @ X
    print(f"3. Normal equation dimension verification:")
    print(f"   X^T: {X.T.shape} ({n_features} x {n_samples})")
    print(f"   X^T X: {XTX.shape} ({n_features} x {n_features})")
    print(f"   (X^T X)^{{-1}}: {np.linalg.inv(XTX).shape} ({n_features} x {n_features})")

    XTy = X.T @ y
    print(f"   X^T y: {XTy.shape} ({n_features},)")

    w_hat = np.linalg.inv(XTX) @ XTy
    print(f"   w = (X^T X)^{{-1}} X^T y: {w_hat.shape} ({n_features},)")
    print()

    # Verify solution
    print("=== Solution Verification ===")
    print(f"True weights:      {true_w}")
    print(f"Estimated weights: {np.round(w_hat, 4)}")
    print(f"Max error: {np.max(np.abs(true_w - w_hat)):.6f}")


if __name__ == "__main__":
    print("=== Exercise 1: Vector Spaces and Basis ===")
    exercise_1()
    print("\n=== Exercise 2: Linear Transformation Visualization ===")
    exercise_2()
    print("\n=== Exercise 3: Projection Matrix ===")
    exercise_3()
    print("\n=== Exercise 4: Least Squares Problem ===")
    exercise_4()
    print("\n=== Exercise 5: ML Application ===")
    exercise_5()
    print("\nAll exercises completed!")
