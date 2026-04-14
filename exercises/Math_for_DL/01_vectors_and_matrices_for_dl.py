"""
Exercises for Lesson 01: Vectors and Matrices for DL
Topic: Math_for_DL

Complete the TODO sections. Run with: python 01_vectors_and_matrices_for_dl.py
"""

import numpy as np


def exercise_1_trace_via_einsum():
    """Compute tr(AB) using np.einsum WITHOUT forming the product matrix.

    Given A (n x n) and B (n x n), compute trace(A @ B) using einsum
    with only two input tensors and no intermediate matrix.
    """
    A = np.array([[1, 2], [3, 4]], dtype=float)
    B = np.array([[5, 6], [7, 8]], dtype=float)

    # TODO: Compute trace(A @ B) using np.einsum in one call
    trace_ab = None  # Replace with np.einsum(...)

    return trace_ab


def exercise_2_bias_gradient():
    """Derive dL/db for linear layer y = Wx + b.

    Given dL/dy (upstream gradient), return dL/db.
    Hint: since y_i = sum_j W_ij x_j + b_i, what is dy_i/db_j?
    """
    n_out = 4
    dL_dy = np.array([0.5, -0.3, 0.8, -0.1])

    # TODO: Compute dL/db (should have shape (n_out,))
    dL_db = None  # Replace with the correct expression

    return dL_db


def exercise_3_orthogonal_norm_preservation():
    """Verify ||Qx||_2 = ||x||_2 for random orthogonal Q.

    Create a random orthogonal matrix Q (n x n) and verify that
    it preserves the L2 norm for multiple random vectors x.
    """
    n = 10
    np.random.seed(42)

    # TODO: Create a random orthogonal matrix Q
    Q = None  # Hint: use np.linalg.qr on a random matrix

    # TODO: Verify norm preservation for 5 random vectors
    results = []  # List of (norm_x, norm_Qx) tuples
    for _ in range(5):
        x = np.random.randn(n)
        # TODO: compute ||x|| and ||Qx||
        norm_x = None
        norm_Qx = None
        results.append((norm_x, norm_Qx))

    return Q, results


def exercise_4_batched_linear_einsum():
    """Implement batched linear layer Y = XW^T + b using einsum.

    Given X (B x n_in), W (n_out x n_in), b (n_out),
    compute Y (B x n_out) using np.einsum for the matmul part.
    """
    B, n_in, n_out = 16, 5, 3
    np.random.seed(42)
    X = np.random.randn(B, n_in)
    W = np.random.randn(n_out, n_in)
    b = np.random.randn(n_out)

    # TODO: Compute Y = X @ W.T + b using einsum for the matrix multiply
    Y = None

    return Y


def exercise_5_gradient_frobenius_norm():
    """Compute the Frobenius norm of dL/dW for a linear layer.

    Given x, y=Wx+b, and dL/dy, compute dL/dW and its Frobenius norm.
    """
    np.random.seed(42)
    n_in, n_out = 10, 5
    W = np.random.randn(n_out, n_in)
    x = np.random.randn(n_in)
    b = np.random.randn(n_out)
    dL_dy = np.random.randn(n_out)

    # TODO: Compute dL/dW
    dL_dW = None

    # TODO: Compute Frobenius norm of dL/dW
    grad_norm = None

    return dL_dW, grad_norm


if __name__ == "__main__":
    print("Exercise 1: Trace via einsum")
    result = exercise_1_trace_via_einsum()
    expected = np.trace(np.array([[1,2],[3,4]]) @ np.array([[5,6],[7,8]]))
    if result is not None:
        print(f"  Result: {result}, Expected: {expected}, Pass: {np.isclose(result, expected)}")
    else:
        print("  Not implemented yet")

    print("\nExercise 2: Bias gradient")
    result = exercise_2_bias_gradient()
    if result is not None:
        print(f"  dL/db = {result}")
        print(f"  Pass: {np.allclose(result, [0.5, -0.3, 0.8, -0.1])}")
    else:
        print("  Not implemented yet")

    print("\nExercise 3: Orthogonal norm preservation")
    Q, results = exercise_3_orthogonal_norm_preservation()
    if Q is not None and results[0][0] is not None:
        all_preserved = all(np.isclose(a, b) for a, b in results)
        print(f"  All norms preserved: {all_preserved}")
    else:
        print("  Not implemented yet")

    print("\nExercise 4: Batched linear via einsum")
    Y = exercise_4_batched_linear_einsum()
    if Y is not None:
        np.random.seed(42)
        X = np.random.randn(16, 5)
        W = np.random.randn(3, 5)
        b = np.random.randn(3)
        Y_expected = X @ W.T + b
        print(f"  Shape: {Y.shape}, Pass: {np.allclose(Y, Y_expected)}")
    else:
        print("  Not implemented yet")

    print("\nExercise 5: Gradient Frobenius norm")
    dW, gn = exercise_5_gradient_frobenius_norm()
    if dW is not None:
        print(f"  dL/dW shape: {dW.shape}, ||dL/dW||_F = {gn:.4f}")
    else:
        print("  Not implemented yet")
