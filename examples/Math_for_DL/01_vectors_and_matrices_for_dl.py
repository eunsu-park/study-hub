"""
Vectors and Matrices for Deep Learning

Demonstrates tensor operations fundamental to DL:
- Einstein summation convention with np.einsum
- Batched linear transformations
- Matrix calculus: gradient of linear layer
- Special matrices (orthogonal, diagonal)
- Norm computations (Frobenius, spectral)

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


def einsum_examples():
    """Demonstrate Einstein summation convention with np.einsum."""
    print("=" * 60)
    print("EINSTEIN SUMMATION EXAMPLES")
    print("=" * 60)

    A = np.random.randn(4, 3)
    B = np.random.randn(3, 5)
    x = np.random.randn(3)

    # Matrix-vector product
    y1 = np.einsum('ij,j->i', A, x)
    y2 = A @ x
    print(f"\nMatrix-vector: match = {np.allclose(y1, y2)}")

    # Matrix-matrix product
    C1 = np.einsum('ik,kj->ij', A, B)
    C2 = A @ B
    print(f"Matrix-matrix: match = {np.allclose(C1, C2)}")

    # Trace
    M = np.random.randn(4, 4)
    tr1 = np.einsum('ii->', M)
    tr2 = np.trace(M)
    print(f"Trace: match = {np.isclose(tr1, tr2)}")

    # Outer product
    u = np.array([1, 2, 3])
    v = np.array([4, 5])
    outer1 = np.einsum('i,j->ij', u, v)
    outer2 = np.outer(u, v)
    print(f"Outer product: match = {np.allclose(outer1, outer2)}")

    # Batch matrix multiply (attention-like)
    B_size, T, d = 4, 10, 64
    Q = np.random.randn(B_size, T, d)
    K = np.random.randn(B_size, T, d)
    scores = np.einsum('btd,bsd->bts', Q, K)
    print(f"\nBatched attention scores shape: {scores.shape}")

    # Trace of A @ B without forming the product
    A2 = np.random.randn(5, 5)
    B2 = np.random.randn(5, 5)
    tr_ab = np.einsum('ij,ji->', A2, B2)
    tr_ab_check = np.trace(A2 @ B2)
    print(f"Trace(AB) via einsum: {tr_ab:.4f}, direct: {tr_ab_check:.4f}")


def batched_linear_layer():
    """Demonstrate batched linear transformation."""
    print("\n" + "=" * 60)
    print("BATCHED LINEAR LAYER")
    print("=" * 60)

    B, n_in, n_out = 32, 784, 256
    X = np.random.randn(B, n_in)
    W = np.random.randn(n_out, n_in) * 0.01
    b = np.zeros(n_out)

    # Standard batched forward
    Y = X @ W.T + b
    print(f"Input: {X.shape}, Weight: {W.shape}, Output: {Y.shape}")

    # Using einsum
    Y_einsum = np.einsum('bi,oi->bo', X, W) + b
    print(f"einsum match: {np.allclose(Y, Y_einsum)}")


def linear_layer_gradient():
    """Derive and verify linear layer gradients."""
    print("\n" + "=" * 60)
    print("LINEAR LAYER GRADIENT VERIFICATION")
    print("=" * 60)

    n_in, n_out = 4, 3
    np.random.seed(42)
    W = np.random.randn(n_out, n_in)
    x = np.random.randn(n_in)
    b = np.random.randn(n_out)

    # Forward
    y = W @ x + b
    dL_dy = np.random.randn(n_out)

    # Analytical gradients
    dL_dW = np.outer(dL_dy, x)
    dL_dx = W.T @ dL_dy
    dL_db = dL_dy

    # Numerical verification
    eps = 1e-5
    dL_dW_num = np.zeros_like(W)
    for i in range(n_out):
        for j in range(n_in):
            W_p = W.copy(); W_p[i, j] += eps
            W_m = W.copy(); W_m[i, j] -= eps
            dL_dW_num[i, j] = dL_dy @ ((W_p @ x + b) - (W_m @ x + b)) / (2 * eps)

    print(f"dL/dW max error: {np.max(np.abs(dL_dW - dL_dW_num)):.2e}")
    print(f"dL/dW shape: {dL_dW.shape} (same as W: {W.shape})")


def norm_computations():
    """Demonstrate various norms used in DL."""
    print("\n" + "=" * 60)
    print("NORMS IN DEEP LEARNING")
    print("=" * 60)

    W = np.random.randn(100, 80)

    # Frobenius norm
    frob = np.linalg.norm(W, 'fro')
    frob_manual = np.sqrt(np.sum(W**2))
    print(f"Frobenius norm: {frob:.4f} (manual: {frob_manual:.4f})")

    # Spectral norm (largest singular value)
    spectral = np.linalg.norm(W, 2)
    svd_vals = np.linalg.svd(W, compute_uv=False)
    print(f"Spectral norm: {spectral:.4f} (max sv: {svd_vals[0]:.4f})")

    # L1, L2 vector norms
    x = np.array([3, -4, 0, 5, -2])
    print(f"\nVector x = {x}")
    print(f"  L1 norm: {np.linalg.norm(x, 1)}")
    print(f"  L2 norm: {np.linalg.norm(x, 2):.4f}")
    print(f"  Linf norm: {np.linalg.norm(x, np.inf)}")


def orthogonal_preservation():
    """Show orthogonal matrices preserve norms."""
    print("\n" + "=" * 60)
    print("ORTHOGONAL MATRIX NORM PRESERVATION")
    print("=" * 60)

    n = 10
    Q, _ = np.linalg.qr(np.random.randn(n, n))

    for trial in range(5):
        x = np.random.randn(n)
        Qx = Q @ x
        print(f"  ||x|| = {np.linalg.norm(x):.6f}, ||Qx|| = {np.linalg.norm(Qx):.6f}, "
              f"match = {np.isclose(np.linalg.norm(x), np.linalg.norm(Qx))}")


def visualize_two_layer_forward():
    """Trace math through a 2-layer network."""
    print("\n" + "=" * 60)
    print("TWO-LAYER NETWORK FORWARD PASS")
    print("=" * 60)

    np.random.seed(42)
    n_in, n_hidden, n_out = 4, 8, 1
    W1 = np.random.randn(n_hidden, n_in) * np.sqrt(2.0 / n_in)
    b1 = np.zeros(n_hidden)
    w2 = np.random.randn(n_hidden) * np.sqrt(2.0 / n_hidden)
    b2 = 0.0
    x = np.random.randn(n_in)

    z1 = W1 @ x + b1
    h = np.maximum(z1, 0)  # ReLU
    y_hat = w2 @ h + b2

    print(f"x:     {x.shape} -> z1:    {z1.shape}")
    print(f"z1:    {z1.shape} -> h:     {h.shape} (ReLU)")
    print(f"h:     {h.shape} -> y_hat: scalar = {y_hat:.4f}")
    print(f"Active neurons: {np.sum(z1 > 0)}/{n_hidden}")


if __name__ == "__main__":
    einsum_examples()
    batched_linear_layer()
    linear_layer_gradient()
    norm_computations()
    orthogonal_preservation()
    visualize_two_layer_forward()
