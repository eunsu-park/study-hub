"""
Singular Value Decomposition Applications

Demonstrates SVD concepts:
- Full and reduced SVD computation
- Image compression via low-rank approximation
- Pseudoinverse computation via SVD
- Eckart-Young theorem (best rank-k approximation)
- Numerical rank determination

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


def svd_basics():
    """Compute and verify SVD."""
    print("=" * 60)
    print("SVD BASICS")
    print("=" * 60)

    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9],
                  [10, 11, 12]], dtype=float)

    U, sigma, Vt = np.linalg.svd(A, full_matrices=True)
    print(f"\nA (4x3):\n{A}")
    print(f"\nU (4x4):\n{np.round(U, 4)}")
    print(f"\nSingular values: {np.round(sigma, 4)}")
    print(f"\nV^T (3x3):\n{np.round(Vt, 4)}")

    # Reconstruct A = U @ Sigma @ V^T
    Sigma = np.zeros_like(A)
    np.fill_diagonal(Sigma, sigma)
    A_reconstructed = U @ Sigma @ Vt
    print(f"\nU @ Sigma @ V^T:\n{np.round(A_reconstructed, 10)}")
    print(f"Reconstruction matches: {np.allclose(A, A_reconstructed)}")

    # Reduced SVD
    U_r, sigma_r, Vt_r = np.linalg.svd(A, full_matrices=False)
    print(f"\n--- Reduced SVD ---")
    print(f"U_r shape: {U_r.shape} (vs full: {U.shape})")
    print(f"Sigma_r shape: {sigma_r.shape}")
    print(f"Vt_r shape: {Vt_r.shape} (vs full: {Vt.shape})")

    A_reduced = U_r @ np.diag(sigma_r) @ Vt_r
    print(f"Reduced SVD reconstruction matches: {np.allclose(A, A_reduced)}")


def svd_properties():
    """Demonstrate key SVD properties."""
    print("\n" + "=" * 60)
    print("SVD PROPERTIES")
    print("=" * 60)

    A = np.array([[3, 2, 2],
                  [2, 3, -2]], dtype=float)

    U, sigma, Vt = np.linalg.svd(A)
    print(f"\nA:\n{A}")
    print(f"Singular values: {np.round(sigma, 4)}")

    # Rank = number of nonzero singular values
    rank = np.sum(sigma > 1e-10)
    print(f"\nRank from SVD: {rank}")
    print(f"np.linalg.matrix_rank: {np.linalg.matrix_rank(A)}")

    # Frobenius norm = sqrt(sum of sigma^2)
    frob = np.linalg.norm(A, 'fro')
    frob_svd = np.sqrt(np.sum(sigma ** 2))
    print(f"\n||A||_F = {frob:.4f}")
    print(f"sqrt(sum(sigma^2)) = {frob_svd:.4f}")

    # 2-norm = largest singular value
    norm_2 = np.linalg.norm(A, 2)
    print(f"\n||A||_2 = {norm_2:.4f}")
    print(f"sigma_1 = {sigma[0]:.4f}")

    # Relationship to eigenvalues of A^T A and A A^T
    ATA = A.T @ A
    AAT = A @ A.T
    eig_ATA = np.sort(np.linalg.eigvalsh(ATA))[::-1]
    eig_AAT = np.sort(np.linalg.eigvalsh(AAT))[::-1]
    print(f"\nEigenvalues of A^T A: {np.round(eig_ATA, 4)}")
    print(f"sigma^2: {np.round(sigma**2, 4)}")
    print(f"Match: {np.allclose(eig_ATA[:len(sigma)], sigma**2)}")


def low_rank_approximation():
    """Demonstrate low-rank approximation (Eckart-Young theorem)."""
    print("\n" + "=" * 60)
    print("LOW-RANK APPROXIMATION")
    print("=" * 60)

    # Create a matrix with known singular values
    np.random.seed(42)
    m, n = 20, 15
    # Build A with specific singular value profile
    U_true = np.linalg.qr(np.random.randn(m, m))[0]
    V_true = np.linalg.qr(np.random.randn(n, n))[0]
    sigma_true = np.array([100, 50, 20, 5, 1] + [0.1] * 10)
    Sigma = np.zeros((m, n))
    np.fill_diagonal(Sigma, sigma_true)
    A = U_true @ Sigma @ V_true.T

    U, sigma, Vt = np.linalg.svd(A, full_matrices=False)

    print(f"\nMatrix A: {A.shape}")
    print(f"Singular values: {np.round(sigma[:8], 4)} ...")
    print(f"||A||_F = {np.linalg.norm(A, 'fro'):.4f}")

    # Rank-k approximations
    print(f"\n{'k':>3}  {'||A-A_k||_F':>14}  {'Relative error':>14}  {'Compression':>12}")
    print("-" * 50)

    original_size = m * n
    for k in [1, 2, 3, 4, 5, 10]:
        A_k = U[:, :k] @ np.diag(sigma[:k]) @ Vt[:k, :]
        error = np.linalg.norm(A - A_k, 'fro')
        rel_error = error / np.linalg.norm(A, 'fro')
        compressed_size = k * (m + n + 1)
        ratio = compressed_size / original_size
        print(f"{k:3d}  {error:14.4f}  {rel_error:14.6f}  {ratio:11.2%}")

    # Eckart-Young theorem: sum of remaining sigma^2
    k = 3
    theoretical_error = np.sqrt(np.sum(sigma[k:] ** 2))
    actual_error = np.linalg.norm(A - U[:, :k] @ np.diag(sigma[:k]) @ Vt[:k, :], 'fro')
    print(f"\nEckart-Young: ||A - A_{k}||_F = sqrt(sum sigma_{{i>k}}^2)")
    print(f"Theoretical: {theoretical_error:.4f}")
    print(f"Actual: {actual_error:.4f}")
    print(f"Match: {np.isclose(theoretical_error, actual_error)}")


def image_compression():
    """Demonstrate image compression via SVD."""
    print("\n" + "=" * 60)
    print("IMAGE COMPRESSION VIA SVD")
    print("=" * 60)

    # Create a synthetic grayscale image
    np.random.seed(42)
    m, n = 100, 80
    x = np.linspace(0, 2 * np.pi, n)
    y = np.linspace(0, 2 * np.pi, m)
    X, Y = np.meshgrid(x, y)
    image = np.sin(X) * np.cos(Y) + 0.5 * np.sin(2 * X + Y) + np.random.randn(m, n) * 0.1

    U, sigma, Vt = np.linalg.svd(image, full_matrices=False)
    print(f"Image size: {image.shape}")
    print(f"Original storage: {m * n} values")
    print(f"Singular values (first 10): {np.round(sigma[:10], 2)}")

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    ranks = [1, 5, 10, 20, 40, min(m, n)]

    for ax, k in zip(axes.flat, ranks):
        compressed = U[:, :k] @ np.diag(sigma[:k]) @ Vt[:k, :]
        storage = k * (m + n + 1)
        ratio = storage / (m * n) * 100
        error = np.linalg.norm(image - compressed, 'fro') / np.linalg.norm(image, 'fro')

        ax.imshow(compressed, cmap='viridis', aspect='auto')
        if k == min(m, n):
            ax.set_title(f'Full rank (k={k})\n100% storage')
        else:
            ax.set_title(f'Rank {k}\n{ratio:.1f}% storage, err={error:.4f}')
        ax.axis('off')

    plt.suptitle('SVD Image Compression at Various Ranks', fontsize=14)
    plt.tight_layout()
    plt.savefig('svd_compression.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: svd_compression.png")


def pseudoinverse_via_svd():
    """Compute pseudoinverse using SVD."""
    print("\n" + "=" * 60)
    print("PSEUDOINVERSE VIA SVD")
    print("=" * 60)

    A = np.array([[1, 2],
                  [3, 4],
                  [5, 6]], dtype=float)

    U, sigma, Vt = np.linalg.svd(A, full_matrices=False)

    # A^+ = V Sigma^+ U^T
    sigma_pinv = np.diag(1.0 / sigma)
    A_pinv_svd = Vt.T @ sigma_pinv @ U.T

    A_pinv_numpy = np.linalg.pinv(A)

    print(f"\nA (3x2):\n{A}")
    print(f"\nA^+ via SVD:\n{np.round(A_pinv_svd, 6)}")
    print(f"A^+ via np.linalg.pinv:\n{np.round(A_pinv_numpy, 6)}")
    print(f"Match: {np.allclose(A_pinv_svd, A_pinv_numpy)}")

    # Moore-Penrose conditions
    print(f"\n--- Moore-Penrose Conditions ---")
    print(f"1. A @ A+ @ A == A: {np.allclose(A @ A_pinv_svd @ A, A)}")
    print(f"2. A+ @ A @ A+ == A+: {np.allclose(A_pinv_svd @ A @ A_pinv_svd, A_pinv_svd)}")
    print(f"3. (A @ A+)^T == A @ A+: {np.allclose((A @ A_pinv_svd).T, A @ A_pinv_svd)}")
    print(f"4. (A+ @ A)^T == A+ @ A: {np.allclose((A_pinv_svd @ A).T, A_pinv_svd @ A)}")

    # Least squares via pseudoinverse
    b = np.array([1, 2, 3], dtype=float)
    x_pinv = A_pinv_svd @ b
    x_lstsq = np.linalg.lstsq(A, b, rcond=None)[0]
    print(f"\nLeast squares solution:")
    print(f"  via pinv: {np.round(x_pinv, 6)}")
    print(f"  via lstsq: {np.round(x_lstsq, 6)}")
    print(f"  Match: {np.allclose(x_pinv, x_lstsq)}")


def numerical_rank():
    """Determine numerical rank using SVD."""
    print("\n" + "=" * 60)
    print("NUMERICAL RANK DETERMINATION")
    print("=" * 60)

    # Matrix with exact rank 2, but noise makes all SVs nonzero
    np.random.seed(42)
    u1 = np.array([1, 2, 3, 4], dtype=float)
    u2 = np.array([5, 6, 7, 8], dtype=float)
    v1 = np.array([1, 0, 1], dtype=float)
    v2 = np.array([0, 1, 1], dtype=float)

    A_exact = np.outer(u1, v1) + np.outer(u2, v2)
    noise = np.random.randn(4, 3) * 1e-10
    A_noisy = A_exact + noise

    sigma_exact = np.linalg.svd(A_exact, compute_uv=False)
    sigma_noisy = np.linalg.svd(A_noisy, compute_uv=False)

    print(f"\nExact rank-2 matrix (4x3):")
    print(f"Singular values: {sigma_exact}")
    print(f"matrix_rank: {np.linalg.matrix_rank(A_exact)}")

    print(f"\nWith noise (1e-10):")
    print(f"Singular values: {sigma_noisy}")
    print(f"matrix_rank: {np.linalg.matrix_rank(A_noisy)}")

    # Custom rank determination with tolerance
    for tol in [1e-8, 1e-12, 1e-14]:
        rank = np.sum(sigma_noisy > tol)
        print(f"Rank with tol={tol:.0e}: {rank}")


if __name__ == "__main__":
    svd_basics()
    svd_properties()
    low_rank_approximation()
    image_compression()
    pseudoinverse_via_svd()
    numerical_rank()
    print("\nAll examples completed!")
