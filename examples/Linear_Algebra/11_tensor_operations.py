"""
Tensor Operations

Demonstrates tensor and multilinear algebra:
- np.einsum for general tensor contractions
- Batch matrix multiplication
- Outer product and tensor product
- Kronecker product and its properties
- Reshaping tensors for neural network operations

Dependencies: numpy
"""

import numpy as np
import time


def einsum_basics():
    """Demonstrate Einstein summation notation with np.einsum."""
    print("=" * 60)
    print("EINSUM BASICS")
    print("=" * 60)

    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[5, 6],
                  [7, 8]])
    v = np.array([1, 2])

    # Trace: sum of diagonal
    tr = np.einsum('ii', A)
    print(f"\nA:\n{A}")
    print(f"Trace: einsum('ii', A) = {tr}")
    print(f"np.trace: {np.trace(A)}")

    # Diagonal extraction
    diag = np.einsum('ii->i', A)
    print(f"\nDiagonal: einsum('ii->i', A) = {diag}")

    # Matrix-vector product
    mv = np.einsum('ij,j->i', A, v)
    print(f"\nA @ v: einsum('ij,j->i', A, v) = {mv}")
    print(f"np.dot: {A @ v}")

    # Matrix multiplication
    mm = np.einsum('ij,jk->ik', A, B)
    print(f"\nA @ B: einsum('ij,jk->ik', A, B):\n{mm}")
    print(f"Match: {np.allclose(mm, A @ B)}")

    # Outer product
    outer = np.einsum('i,j->ij', v, v)
    print(f"\nOuter product: einsum('i,j->ij', v, v):\n{outer}")

    # Element-wise multiplication
    hadamard = np.einsum('ij,ij->ij', A, B)
    print(f"\nHadamard: einsum('ij,ij->ij', A, B):\n{hadamard}")
    print(f"Match: {np.allclose(hadamard, A * B)}")

    # Sum all elements
    total = np.einsum('ij->', A)
    print(f"\nSum: einsum('ij->', A) = {total}")
    print(f"np.sum: {np.sum(A)}")

    # Frobenius inner product: tr(A^T B)
    frob = np.einsum('ij,ij->', A, B)
    print(f"\nFrobenius inner product: einsum('ij,ij->', A, B) = {frob}")
    print(f"np.trace(A.T @ B) = {np.trace(A.T @ B)}")


def batch_operations():
    """Demonstrate batch matrix operations with einsum."""
    print("\n" + "=" * 60)
    print("BATCH MATRIX OPERATIONS")
    print("=" * 60)

    np.random.seed(42)
    batch_size = 4
    A_batch = np.random.randn(batch_size, 3, 4)
    B_batch = np.random.randn(batch_size, 4, 2)

    # Batch matrix multiplication
    C_batch = np.einsum('bij,bjk->bik', A_batch, B_batch)
    print(f"A batch: {A_batch.shape}")
    print(f"B batch: {B_batch.shape}")
    print(f"C = A @ B batch: {C_batch.shape}")

    # Verify against loop
    C_loop = np.array([A_batch[i] @ B_batch[i] for i in range(batch_size)])
    print(f"Match loop: {np.allclose(C_batch, C_loop)}")

    # Also achievable with np.matmul (@ operator)
    C_matmul = A_batch @ B_batch
    print(f"Match matmul: {np.allclose(C_batch, C_matmul)}")

    # Batch trace
    M_batch = np.random.randn(batch_size, 3, 3)
    traces = np.einsum('bii->b', M_batch)
    traces_loop = np.array([np.trace(M_batch[i]) for i in range(batch_size)])
    print(f"\nBatch traces: {np.round(traces, 4)}")
    print(f"Match: {np.allclose(traces, traces_loop)}")

    # Batch outer product
    u_batch = np.random.randn(batch_size, 3)
    v_batch = np.random.randn(batch_size, 4)
    outer_batch = np.einsum('bi,bj->bij', u_batch, v_batch)
    print(f"\nBatch outer product: {outer_batch.shape}")


def tensor_contraction():
    """Demonstrate tensor contraction operations."""
    print("\n" + "=" * 60)
    print("TENSOR CONTRACTION")
    print("=" * 60)

    # 3rd order tensor
    T = np.random.randn(3, 4, 5)
    print(f"Tensor T shape: {T.shape}")

    # Contract first two indices (partial trace)
    result = np.einsum('iij->j', T[:3, :3, :])
    print(f"\nPartial trace (contract dims 0,1): shape {result.shape}")

    # Contract with vectors
    u = np.random.randn(3)
    v = np.random.randn(4)
    w = np.random.randn(5)

    # T(u, v, w) = sum_ijk T_ijk u_i v_j w_k
    scalar = np.einsum('ijk,i,j,k->', T, u, v, w)
    print(f"Full contraction T(u,v,w): {scalar:.4f}")

    # Mode-1 product: T x_1 u
    mode1 = np.einsum('ijk,i->jk', T, u)
    print(f"Mode-1 product shape: {mode1.shape}")

    # Mode-2 product: T x_2 v
    mode2 = np.einsum('ijk,j->ik', T, v)
    print(f"Mode-2 product shape: {mode2.shape}")

    # Mode-3 product: T x_3 w
    mode3 = np.einsum('ijk,k->ij', T, w)
    print(f"Mode-3 product shape: {mode3.shape}")

    # Tensor dot product
    T1 = np.random.randn(3, 4, 5)
    T2 = np.random.randn(3, 4, 5)
    tdot = np.einsum('ijk,ijk->', T1, T2)
    print(f"\nTensor inner product: {tdot:.4f}")
    print(f"np.sum(T1 * T2): {np.sum(T1 * T2):.4f}")


def kronecker_product():
    """Demonstrate Kronecker product and its properties."""
    print("\n" + "=" * 60)
    print("KRONECKER PRODUCT")
    print("=" * 60)

    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[0, 5],
                  [6, 7]])

    # Kronecker product
    K = np.kron(A, B)
    print(f"A:\n{A}")
    print(f"B:\n{B}")
    print(f"\nA kron B:\n{K}")
    print(f"Shape: ({A.shape[0]}*{B.shape[0]}, {A.shape[1]}*{B.shape[1]}) = {K.shape}")

    # Properties
    print(f"\n--- Kronecker Product Properties ---")

    # (A kron B)(C kron D) = (AC) kron (BD)
    C = np.array([[1, 0], [0, 1]])
    D = np.array([[2, 1], [1, 2]])
    lhs = np.kron(A, B) @ np.kron(C, D)
    rhs = np.kron(A @ C, B @ D)
    print(f"(A kron B)(C kron D) == (AC) kron (BD): {np.allclose(lhs, rhs)}")

    # det(A kron B) = det(A)^n * det(B)^m
    m, n = A.shape[0], B.shape[0]
    det_kron = np.linalg.det(K)
    det_formula = np.linalg.det(A)**n * np.linalg.det(B)**m
    print(f"det(A kron B) = {det_kron:.4f}")
    print(f"det(A)^n * det(B)^m = {det_formula:.4f}")
    print(f"Match: {np.isclose(det_kron, det_formula)}")

    # tr(A kron B) = tr(A) * tr(B)
    print(f"tr(A kron B) = {np.trace(K)}")
    print(f"tr(A) * tr(B) = {np.trace(A) * np.trace(B)}")

    # Eigenvalues of A kron B = all products of eigenvalues
    eig_A = np.linalg.eigvals(A)
    eig_B = np.linalg.eigvals(B)
    eig_K = np.sort(np.linalg.eigvals(K))
    eig_products = np.sort(np.array([a * b for a in eig_A for b in eig_B]))
    print(f"\nEigenvalues of A kron B: {np.round(np.sort(eig_K.real), 4)}")
    print(f"Products of eigenvalues: {np.round(np.sort(eig_products.real), 4)}")


def neural_network_reshaping():
    """Demonstrate tensor reshaping for neural network operations."""
    print("\n" + "=" * 60)
    print("TENSOR RESHAPING FOR NEURAL NETWORKS")
    print("=" * 60)

    # Simulated batch of images: (batch, height, width, channels)
    batch_size, H, W, C = 8, 4, 4, 3
    images = np.random.randn(batch_size, H, W, C)
    print(f"Input images: {images.shape} (B, H, W, C)")

    # Flatten spatial dimensions for fully-connected layer
    flat = images.reshape(batch_size, -1)
    print(f"Flattened: {flat.shape} (B, H*W*C)")

    # Channel-first format (for PyTorch): (B, C, H, W)
    images_chw = np.transpose(images, (0, 3, 1, 2))
    print(f"Channel-first: {images_chw.shape} (B, C, H, W)")

    # Reshape for multi-head attention
    # (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
    batch, seq_len, d_model = 4, 10, 64
    num_heads, d_k = 8, d_model // 8
    X = np.random.randn(batch, seq_len, d_model)
    print(f"\nAttention input: {X.shape}")

    X_heads = X.reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    print(f"Multi-head reshape: {X_heads.shape} (B, H, S, d_k)")

    # Compute attention for all heads using einsum
    Q = K_mat = V = X_heads
    scores = np.einsum('bhsd,bhtd->bhst', Q, K_mat) / np.sqrt(d_k)
    print(f"Attention scores: {scores.shape}")

    # Concatenate heads back
    output = X_heads.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)
    print(f"Concatenated output: {output.shape}")


def einsum_performance():
    """Compare einsum with explicit operations."""
    print("\n" + "=" * 60)
    print("EINSUM PERFORMANCE COMPARISON")
    print("=" * 60)

    np.random.seed(42)
    n = 200
    A = np.random.randn(n, n)
    B = np.random.randn(n, n)
    v = np.random.randn(n)

    operations = [
        ("Trace", lambda: np.einsum('ii', A), lambda: np.trace(A)),
        ("Matrix multiply", lambda: np.einsum('ij,jk->ik', A, B), lambda: A @ B),
        ("Frobenius norm^2", lambda: np.einsum('ij,ij->', A, A), lambda: np.sum(A * A)),
        ("Outer product", lambda: np.einsum('i,j->ij', v, v), lambda: np.outer(v, v)),
    ]

    print(f"\n{'Operation':>20}  {'einsum (ms)':>12}  {'explicit (ms)':>14}  {'Match':>6}")
    print("-" * 60)

    for name, einsum_op, explicit_op in operations:
        # Time einsum
        start = time.perf_counter()
        for _ in range(100):
            r1 = einsum_op()
        t_einsum = (time.perf_counter() - start) / 100 * 1000

        # Time explicit
        start = time.perf_counter()
        for _ in range(100):
            r2 = explicit_op()
        t_explicit = (time.perf_counter() - start) / 100 * 1000

        match = np.allclose(r1, r2)
        print(f"{name:>20}  {t_einsum:12.4f}  {t_explicit:14.4f}  {match!s:>6}")


if __name__ == "__main__":
    einsum_basics()
    batch_operations()
    tensor_contraction()
    kronecker_product()
    neural_network_reshaping()
    einsum_performance()
    print("\nAll examples completed!")
