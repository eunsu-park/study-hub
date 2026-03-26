"""
QR Decomposition and Least Squares

Demonstrates QR decomposition and its applications:
- QR decomposition with np.linalg.qr
- Classical and Modified Gram-Schmidt process
- Householder reflections
- Solving least squares via QR: Rx = Q^T b
- Polynomial curve fitting

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


def qr_basics():
    """Compute and verify QR decomposition."""
    print("=" * 60)
    print("QR DECOMPOSITION BASICS")
    print("=" * 60)

    A = np.array([[1, 1, 0],
                  [1, 0, 1],
                  [0, 1, 1]], dtype=float)

    Q, R = np.linalg.qr(A)
    print(f"\nA:\n{A}")
    print(f"\nQ (orthogonal):\n{np.round(Q, 4)}")
    print(f"\nR (upper triangular):\n{np.round(R, 4)}")

    # Verify A = QR
    print(f"\nQ @ R:\n{np.round(Q @ R, 10)}")
    print(f"A == QR: {np.allclose(A, Q @ R)}")

    # Q is orthogonal: Q^T Q = I
    print(f"\nQ^T @ Q:\n{np.round(Q.T @ Q, 10)}")
    print(f"Q is orthogonal: {np.allclose(Q.T @ Q, np.eye(3))}")

    # R is upper triangular
    print(f"R is upper triangular: {np.allclose(R, np.triu(R))}")

    # Reduced QR for non-square matrices
    print("\n--- Reduced QR (non-square) ---")
    A_tall = np.array([[1, 2],
                       [3, 4],
                       [5, 6],
                       [7, 8]], dtype=float)
    Q_full, R_full = np.linalg.qr(A_tall, mode='complete')
    Q_reduced, R_reduced = np.linalg.qr(A_tall, mode='reduced')
    print(f"A: {A_tall.shape}")
    print(f"Full QR: Q={Q_full.shape}, R={R_full.shape}")
    print(f"Reduced QR: Q={Q_reduced.shape}, R={R_reduced.shape}")
    print(f"Both reconstruct A: {np.allclose(Q_full @ R_full, A_tall) and np.allclose(Q_reduced @ R_reduced, A_tall)}")


def gram_schmidt():
    """Implement Classical and Modified Gram-Schmidt."""
    print("\n" + "=" * 60)
    print("GRAM-SCHMIDT ORTHOGONALIZATION")
    print("=" * 60)

    A = np.array([[1, 1, 0],
                  [1, 0, 1],
                  [0, 1, 1]], dtype=float)
    m, n = A.shape

    # Classical Gram-Schmidt
    print("--- Classical Gram-Schmidt ---")
    Q_cgs = np.zeros((m, n))
    R_cgs = np.zeros((n, n))

    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            R_cgs[i, j] = np.dot(Q_cgs[:, i], A[:, j])
            v -= R_cgs[i, j] * Q_cgs[:, i]
        R_cgs[j, j] = np.linalg.norm(v)
        Q_cgs[:, j] = v / R_cgs[j, j]

    print(f"Q (CGS):\n{np.round(Q_cgs, 4)}")
    print(f"R (CGS):\n{np.round(R_cgs, 4)}")
    print(f"A == QR: {np.allclose(A, Q_cgs @ R_cgs)}")
    print(f"Q^TQ - I:\n{np.round(Q_cgs.T @ Q_cgs - np.eye(n), 10)}")

    # Modified Gram-Schmidt (more numerically stable)
    print("\n--- Modified Gram-Schmidt ---")
    V = A.copy()
    Q_mgs = np.zeros((m, n))
    R_mgs = np.zeros((n, n))

    for i in range(n):
        R_mgs[i, i] = np.linalg.norm(V[:, i])
        Q_mgs[:, i] = V[:, i] / R_mgs[i, i]
        for j in range(i + 1, n):
            R_mgs[i, j] = np.dot(Q_mgs[:, i], V[:, j])
            V[:, j] -= R_mgs[i, j] * Q_mgs[:, i]

    print(f"Q (MGS):\n{np.round(Q_mgs, 4)}")
    print(f"R (MGS):\n{np.round(R_mgs, 4)}")
    print(f"A == QR: {np.allclose(A, Q_mgs @ R_mgs)}")

    # Numerical stability comparison with ill-conditioned matrix
    print("\n--- Stability Comparison ---")
    np.random.seed(42)
    eps = 1e-8
    B = np.array([[1, 1, 1],
                  [eps, 0, 0],
                  [0, eps, 0],
                  [0, 0, eps]], dtype=float)

    # CGS
    Q_c = np.zeros((4, 3))
    R_c = np.zeros((3, 3))
    for j in range(3):
        v = B[:, j].copy()
        for i in range(j):
            R_c[i, j] = np.dot(Q_c[:, i], B[:, j])
            v -= R_c[i, j] * Q_c[:, i]
        R_c[j, j] = np.linalg.norm(v)
        Q_c[:, j] = v / R_c[j, j]

    # MGS
    V2 = B.copy()
    Q_m = np.zeros((4, 3))
    R_m = np.zeros((3, 3))
    for i in range(3):
        R_m[i, i] = np.linalg.norm(V2[:, i])
        Q_m[:, i] = V2[:, i] / R_m[i, i]
        for j in range(i + 1, 3):
            R_m[i, j] = np.dot(Q_m[:, i], V2[:, j])
            V2[:, j] -= R_m[i, j] * Q_m[:, i]

    orth_error_cgs = np.linalg.norm(Q_c.T @ Q_c - np.eye(3))
    orth_error_mgs = np.linalg.norm(Q_m.T @ Q_m - np.eye(3))
    print(f"Orthogonality error (CGS): {orth_error_cgs:.2e}")
    print(f"Orthogonality error (MGS): {orth_error_mgs:.2e}")
    print(f"MGS is more stable: {orth_error_mgs < orth_error_cgs}")


def householder_qr():
    """Demonstrate Householder reflection-based QR."""
    print("\n" + "=" * 60)
    print("HOUSEHOLDER QR")
    print("=" * 60)

    A = np.array([[1, 1, 0],
                  [1, 0, 1],
                  [0, 1, 1]], dtype=float)
    m, n = A.shape

    R = A.copy()
    Q = np.eye(m)

    for j in range(min(m, n)):
        # Compute Householder vector
        x = R[j:, j]
        e1 = np.zeros_like(x)
        e1[0] = np.linalg.norm(x) * (-1 if x[0] < 0 else 1)
        v = x + e1
        v = v / np.linalg.norm(v)

        # Apply Householder reflection H = I - 2 v v^T
        R[j:, j:] -= 2 * np.outer(v, v @ R[j:, j:])
        Q[:, j:] -= 2 * np.outer(Q[:, j:] @ v, v)

    print(f"A:\n{A}")
    print(f"\nQ (Householder):\n{np.round(Q, 4)}")
    print(f"\nR (Householder):\n{np.round(R, 4)}")
    print(f"\nQ @ R:\n{np.round(Q @ R, 10)}")
    print(f"A == QR: {np.allclose(A, Q @ R)}")
    print(f"Q orthogonal: {np.allclose(Q.T @ Q, np.eye(m))}")


def least_squares_via_qr():
    """Solve least squares using QR decomposition."""
    print("\n" + "=" * 60)
    print("LEAST SQUARES VIA QR")
    print("=" * 60)

    # Generate noisy data
    np.random.seed(42)
    x = np.linspace(0, 5, 30)
    y_true = 2 * x ** 2 - 3 * x + 1
    y = y_true + np.random.randn(30) * 2

    # Fit polynomial of degree 2: y = a0 + a1*x + a2*x^2
    A = np.column_stack([np.ones_like(x), x, x**2])
    print(f"Design matrix A: {A.shape}")

    # QR decomposition approach: A = QR, then Rx = Q^T y
    Q, R = np.linalg.qr(A)
    coeffs_qr = np.linalg.solve(R, Q.T @ y)

    # Compare with normal equations
    coeffs_normal = np.linalg.solve(A.T @ A, A.T @ y)

    # Compare with lstsq
    coeffs_lstsq = np.linalg.lstsq(A, y, rcond=None)[0]

    print(f"\nCoefficients (QR):     a0={coeffs_qr[0]:.4f}, a1={coeffs_qr[1]:.4f}, a2={coeffs_qr[2]:.4f}")
    print(f"Coefficients (Normal): a0={coeffs_normal[0]:.4f}, a1={coeffs_normal[1]:.4f}, a2={coeffs_normal[2]:.4f}")
    print(f"Coefficients (lstsq):  a0={coeffs_lstsq[0]:.4f}, a1={coeffs_lstsq[1]:.4f}, a2={coeffs_lstsq[2]:.4f}")
    print(f"True values:           a0=1.0000, a1=-3.0000, a2=2.0000")
    print(f"\nAll methods agree: {np.allclose(coeffs_qr, coeffs_normal) and np.allclose(coeffs_qr, coeffs_lstsq)}")

    # Residual
    residual = y - A @ coeffs_qr
    print(f"Residual norm: {np.linalg.norm(residual):.4f}")

    # Visualize fit
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, alpha=0.5, label='Noisy data', s=20)
    x_fine = np.linspace(0, 5, 200)
    y_fit = coeffs_qr[0] + coeffs_qr[1] * x_fine + coeffs_qr[2] * x_fine ** 2
    y_exact = 2 * x_fine ** 2 - 3 * x_fine + 1
    ax.plot(x_fine, y_fit, 'r-', linewidth=2, label='QR least squares fit')
    ax.plot(x_fine, y_exact, 'g--', linewidth=1.5, label='True function')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Polynomial Regression via QR-Based Least Squares')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('least_squares_fit.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: least_squares_fit.png")


if __name__ == "__main__":
    qr_basics()
    gram_schmidt()
    householder_qr()
    least_squares_via_qr()
    print("\nAll examples completed!")
