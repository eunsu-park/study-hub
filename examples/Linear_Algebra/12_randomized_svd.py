"""
Randomized SVD

Demonstrates randomized algorithms for large-scale SVD:
- Randomized SVD algorithm implementation
- Power iteration for improved approximation
- Comparison with full SVD: accuracy and timing
- Application to large-scale matrix approximation
- Choosing oversampling parameter and power iterations

Dependencies: numpy, scipy
"""

import numpy as np
import time


def randomized_svd(A, k, p=10, q=1):
    """
    Randomized SVD algorithm.

    Parameters
    ----------
    A : ndarray, shape (m, n)
        Input matrix.
    k : int
        Target rank.
    p : int
        Oversampling parameter (default 10).
    q : int
        Number of power iterations for accuracy (default 1).

    Returns
    -------
    U : ndarray, shape (m, k)
    sigma : ndarray, shape (k,)
    Vt : ndarray, shape (k, n)
    """
    m, n = A.shape
    l = k + p  # Total sketch dimension

    # Step 1: Random projection
    # Draw random Gaussian matrix
    Omega = np.random.randn(n, l)

    # Step 2: Form sketch Y = A @ Omega
    Y = A @ Omega

    # Step 3: Power iteration for better approximation
    # (helps when singular values decay slowly)
    for _ in range(q):
        Y = A @ (A.T @ Y)

    # Step 4: Orthogonalize columns of Y
    Q, _ = np.linalg.qr(Y)

    # Step 5: Project A onto low-dimensional space
    B = Q.T @ A  # B is (l x n), much smaller than A

    # Step 6: Compute SVD of small matrix B
    U_B, sigma, Vt = np.linalg.svd(B, full_matrices=False)

    # Step 7: Recover left singular vectors of A
    U = Q @ U_B

    # Truncate to rank k
    return U[:, :k], sigma[:k], Vt[:k, :]


def basic_demo():
    """Basic demonstration of randomized SVD."""
    print("=" * 60)
    print("RANDOMIZED SVD - BASIC DEMO")
    print("=" * 60)

    np.random.seed(42)
    m, n = 100, 80
    k_true = 5

    # Create low-rank matrix with noise
    U_true = np.linalg.qr(np.random.randn(m, k_true))[0]
    V_true = np.linalg.qr(np.random.randn(n, k_true))[0]
    sigma_true = np.array([100, 50, 20, 10, 5])
    A = U_true @ np.diag(sigma_true) @ V_true.T + np.random.randn(m, n) * 0.1

    print(f"\nMatrix A: {A.shape}")
    print(f"True rank: {k_true}")

    # Full SVD
    U_full, sigma_full, Vt_full = np.linalg.svd(A, full_matrices=False)
    print(f"\nFull SVD singular values (first 10): {np.round(sigma_full[:10], 4)}")

    # Randomized SVD
    k = 5
    U_rand, sigma_rand, Vt_rand = randomized_svd(A, k=k, p=10, q=2)
    print(f"\nRandomized SVD (k={k}):")
    print(f"Singular values: {np.round(sigma_rand, 4)}")

    # Reconstruction error
    A_full_k = U_full[:, :k] @ np.diag(sigma_full[:k]) @ Vt_full[:k, :]
    A_rand_k = U_rand @ np.diag(sigma_rand) @ Vt_rand

    err_full = np.linalg.norm(A - A_full_k, 'fro')
    err_rand = np.linalg.norm(A - A_rand_k, 'fro')

    print(f"\nRank-{k} approximation error (Frobenius):")
    print(f"  Full SVD:       {err_full:.6f}")
    print(f"  Randomized SVD: {err_rand:.6f}")
    print(f"  Relative gap:   {abs(err_rand - err_full) / err_full:.6f}")


def power_iteration_effect():
    """Show how power iterations improve approximation quality."""
    print("\n" + "=" * 60)
    print("EFFECT OF POWER ITERATIONS")
    print("=" * 60)

    np.random.seed(42)
    m, n = 200, 150
    # Matrix with slowly decaying singular values
    sigma_true = 1.0 / np.arange(1, min(m, n) + 1)
    U_true = np.linalg.qr(np.random.randn(m, m))[0]
    V_true = np.linalg.qr(np.random.randn(n, n))[0]
    Sigma = np.zeros((m, n))
    np.fill_diagonal(Sigma, sigma_true)
    A = U_true @ Sigma @ V_true.T

    k = 10
    sigma_exact = np.linalg.svd(A, compute_uv=False)[:k]

    print(f"Matrix: {A.shape}")
    print(f"Target rank: {k}")
    print(f"Singular value decay: 1/i (slow)")
    print(f"\nExact top-{k} singular values: {np.round(sigma_exact, 6)}")

    print(f"\n{'q':>3}  {'Max SV error':>14}  {'Frobenius error':>16}")
    print("-" * 38)

    for q in [0, 1, 2, 3, 5]:
        np.random.seed(42)
        _, sigma_rand, _ = randomized_svd(A, k=k, p=10, q=q)
        sv_error = np.max(np.abs(sigma_exact - sigma_rand))
        A_approx = _  # Not used directly
        # Recompute full approximation
        U_r, s_r, Vt_r = randomized_svd(A, k=k, p=10, q=q)
        frob_error = np.linalg.norm(A - U_r @ np.diag(s_r) @ Vt_r, 'fro')
        print(f"{q:3d}  {sv_error:14.2e}  {frob_error:16.6f}")


def oversampling_analysis():
    """Analyze the effect of oversampling parameter p."""
    print("\n" + "=" * 60)
    print("OVERSAMPLING PARAMETER ANALYSIS")
    print("=" * 60)

    np.random.seed(42)
    m, n = 200, 150
    k_true = 10
    U_true = np.linalg.qr(np.random.randn(m, k_true))[0]
    V_true = np.linalg.qr(np.random.randn(n, k_true))[0]
    sigma_true = np.logspace(2, 0, k_true)
    A = U_true @ np.diag(sigma_true) @ V_true.T + np.random.randn(m, n) * 0.01

    k = 10
    exact_error = np.linalg.norm(
        A - np.linalg.svd(A, full_matrices=False)[0][:, :k] @
        np.diag(np.linalg.svd(A, full_matrices=False)[1][:k]) @
        np.linalg.svd(A, full_matrices=False)[2][:k, :],
        'fro'
    )

    print(f"Matrix: {A.shape}, true rank: {k_true}")
    print(f"Target rank k = {k}")
    print(f"Exact rank-{k} error: {exact_error:.6f}")

    print(f"\n{'p':>3}  {'Error':>12}  {'Relative excess':>16}")
    print("-" * 36)

    for p in [0, 2, 5, 10, 20, 50]:
        errors = []
        for trial in range(10):
            np.random.seed(trial)
            U_r, s_r, Vt_r = randomized_svd(A, k=k, p=p, q=2)
            err = np.linalg.norm(A - U_r @ np.diag(s_r) @ Vt_r, 'fro')
            errors.append(err)
        mean_err = np.mean(errors)
        excess = (mean_err - exact_error) / exact_error
        print(f"{p:3d}  {mean_err:12.6f}  {excess:16.6f}")


def timing_benchmark():
    """Compare timing of randomized vs full SVD."""
    print("\n" + "=" * 60)
    print("TIMING BENCHMARK")
    print("=" * 60)

    sizes = [(500, 400), (1000, 800), (2000, 1500)]
    k = 20

    print(f"\nTarget rank k = {k}")
    print(f"\n{'Size':>12}  {'Full SVD (ms)':>14}  {'Rand SVD (ms)':>14}  {'Speedup':>8}  {'Rel. error':>12}")
    print("-" * 68)

    for m, n in sizes:
        np.random.seed(42)
        # Low-rank + noise matrix
        A = np.random.randn(m, k) @ np.random.randn(k, n) + np.random.randn(m, n) * 0.01

        # Full SVD
        start = time.perf_counter()
        U_f, s_f, Vt_f = np.linalg.svd(A, full_matrices=False)
        t_full = (time.perf_counter() - start) * 1000

        A_full_k = U_f[:, :k] @ np.diag(s_f[:k]) @ Vt_f[:k, :]

        # Randomized SVD
        start = time.perf_counter()
        U_r, s_r, Vt_r = randomized_svd(A, k=k, p=10, q=2)
        t_rand = (time.perf_counter() - start) * 1000

        A_rand_k = U_r @ np.diag(s_r) @ Vt_r

        rel_error = np.linalg.norm(A_full_k - A_rand_k, 'fro') / np.linalg.norm(A_full_k, 'fro')
        speedup = t_full / t_rand

        print(f"{m}x{n:>5}  {t_full:14.1f}  {t_rand:14.1f}  {speedup:7.1f}x  {rel_error:12.2e}")


def sklearn_comparison():
    """Compare with sklearn's TruncatedSVD (also uses randomized methods)."""
    print("\n" + "=" * 60)
    print("SKLEARN COMPARISON")
    print("=" * 60)

    try:
        from sklearn.utils.extmath import randomized_svd as sklearn_rsvd

        np.random.seed(42)
        m, n = 500, 400
        k = 20
        A = np.random.randn(m, k) @ np.random.randn(k, n) + np.random.randn(m, n) * 0.01

        # Our implementation
        start = time.perf_counter()
        U1, s1, Vt1 = randomized_svd(A, k=k, p=10, q=2)
        t1 = (time.perf_counter() - start) * 1000

        # sklearn implementation
        start = time.perf_counter()
        U2, s2, Vt2 = sklearn_rsvd(A, n_components=k, n_oversamples=10, n_iter=2)
        t2 = (time.perf_counter() - start) * 1000

        # Full SVD for reference
        U_f, s_f, Vt_f = np.linalg.svd(A, full_matrices=False)

        err1 = np.linalg.norm(A - U1 @ np.diag(s1) @ Vt1, 'fro')
        err2 = np.linalg.norm(A - U2 @ np.diag(s2) @ Vt2, 'fro')
        err_exact = np.linalg.norm(A - U_f[:, :k] @ np.diag(s_f[:k]) @ Vt_f[:k, :], 'fro')

        print(f"Matrix: {A.shape}, k={k}")
        print(f"\n{'Method':>20}  {'Time (ms)':>10}  {'Error':>12}")
        print("-" * 48)
        print(f"{'Exact rank-k':>20}  {'N/A':>10}  {err_exact:12.6f}")
        print(f"{'Our randomized':>20}  {t1:10.2f}  {err1:12.6f}")
        print(f"{'sklearn randomized':>20}  {t2:10.2f}  {err2:12.6f}")

    except ImportError:
        print("sklearn not available, skipping comparison")


if __name__ == "__main__":
    basic_demo()
    power_iteration_effect()
    oversampling_analysis()
    timing_benchmark()
    sklearn_comparison()
    print("\nAll examples completed!")
