"""
Matrix Decompositions

Demonstrates SVD, low-rank approximation, and DL applications:
- SVD computation and reconstruction
- Truncated SVD for compression
- LoRA simulation
- Spectral normalization via power iteration
- PCA via SVD

Dependencies: numpy, matplotlib
"""

import numpy as np


def svd_basics():
    """SVD computation and properties."""
    print("=" * 60)
    print("SVD BASICS")
    print("=" * 60)
    A = np.array([[1, 2, 3], [4, 5, 6]])
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    recon = U @ np.diag(s) @ Vt
    print(f"A shape: {A.shape}")
    print(f"Singular values: {s.round(4)}")
    print(f"Reconstruction error: {np.linalg.norm(A - recon):.2e}")

    # Verify: singular values = sqrt(eigenvalues of A^T A)
    eigvals = np.sort(np.linalg.eigvalsh(A.T @ A))[::-1]
    print(f"sqrt(eigvals of A^TA): {np.sqrt(eigvals[:2]).round(4)}")


def low_rank_approximation():
    """Truncated SVD for matrix compression."""
    print("\n" + "=" * 60)
    print("LOW-RANK APPROXIMATION")
    print("=" * 60)
    np.random.seed(42)
    m, n = 100, 80
    # Matrix with decaying singular values
    U0, _ = np.linalg.qr(np.random.randn(m, m))
    V0, _ = np.linalg.qr(np.random.randn(n, n))
    true_s = np.exp(-np.arange(min(m,n)) * 0.15)
    S = np.zeros((m, n))
    for i in range(min(m,n)):
        S[i,i] = true_s[i]
    A = U0 @ S @ V0.T

    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    norm_A = np.linalg.norm(A, 'fro')

    for k in [1, 5, 10, 20, 50]:
        Ak = U[:,:k] @ np.diag(s[:k]) @ Vt[:k,:]
        err = np.linalg.norm(A - Ak, 'fro') / norm_A
        compression = (m*n) / (k*(m+n+1))
        print(f"  rank-{k:2d}: error = {err:.4f}, compression = {compression:.1f}x")


def lora_simulation():
    """Simulate LoRA fine-tuning."""
    print("\n" + "=" * 60)
    print("LoRA SIMULATION")
    print("=" * 60)
    d = 512
    r = 8
    W0 = np.random.randn(d, d) * 0.01  # frozen
    B = np.zeros((d, r))
    A = np.random.randn(r, d) * 0.01
    W = W0 + B @ A
    print(f"Full params: {d*d:,}")
    print(f"LoRA params: {2*d*r:,}")
    print(f"Reduction: {d*d/(2*d*r):.0f}x")


def spectral_normalization():
    """Spectral norm via power iteration."""
    print("\n" + "=" * 60)
    print("SPECTRAL NORMALIZATION")
    print("=" * 60)
    W = np.random.randn(100, 80)
    u = np.random.randn(100); u /= np.linalg.norm(u)
    for _ in range(20):
        v = W.T @ u; v /= np.linalg.norm(v)
        u = W @ v; u /= np.linalg.norm(u)
    sigma_pi = u @ W @ v
    sigma_svd = np.linalg.svd(W, compute_uv=False)[0]
    print(f"Power iteration: {sigma_pi:.6f}")
    print(f"SVD:             {sigma_svd:.6f}")
    W_norm = W / sigma_svd
    print(f"After normalization: sigma_1 = {np.linalg.svd(W_norm, compute_uv=False)[0]:.6f}")


def pca_via_svd():
    """PCA using SVD."""
    print("\n" + "=" * 60)
    print("PCA VIA SVD")
    print("=" * 60)
    np.random.seed(42)
    N = 500
    cov = np.array([[2, 1.5], [1.5, 1.5]])
    data = np.random.multivariate_normal([0, 0], cov, N)
    data_c = data - data.mean(axis=0)
    U, s, Vt = np.linalg.svd(data_c, full_matrices=False)
    var_explained = s**2 / (N-1)
    print(f"Variance explained: {var_explained.round(3)}")
    print(f"Ratio: {(var_explained / var_explained.sum()).round(3)}")
    print(f"PC1 direction: {Vt[0].round(4)}")


if __name__ == "__main__":
    svd_basics()
    low_rank_approximation()
    lora_simulation()
    spectral_normalization()
    pca_via_svd()
