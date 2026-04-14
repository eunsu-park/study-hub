"""
Exercises for Lesson 09: Matrix Decompositions
Topic: Math_for_DL

Complete the TODO sections.
"""

import numpy as np


def exercise_1_power_iteration():
    """Find largest singular value and vectors using power iteration.

    Given matrix A, iterate: v = A^T u / ||A^T u||, u = A v / ||A v||
    sigma = u^T A v
    """
    np.random.seed(42)
    A = np.random.randn(10, 8)

    # TODO: Implement power iteration for largest singular value
    n_iters = 50
    u = np.random.randn(A.shape[0])
    u = u / np.linalg.norm(u)

    sigma = None  # Replace with computed largest singular value

    # Verify
    sigma_true = np.linalg.svd(A, compute_uv=False)[0]
    return sigma, sigma_true


def exercise_2_truncated_svd_compression():
    """Compress a matrix using truncated SVD and measure error vs rank k."""
    np.random.seed(42)
    m, n = 50, 40
    A = np.random.randn(m, 5) @ np.random.randn(5, n) + np.random.randn(m, n) * 0.1

    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    norm_A = np.linalg.norm(A, 'fro')

    # TODO: Compute relative error for each rank k
    results = {}
    for k in [1, 2, 5, 10, 20]:
        # TODO: Reconstruct A_k and compute relative Frobenius error
        error = None  # Replace
        results[k] = error

    return results


def exercise_3_lora_forward():
    """Implement LoRA forward pass: y = (W0 + B@A) @ x.

    Verify that the output matches the effective weight matrix.
    """
    np.random.seed(42)
    d = 64
    r = 4

    W0 = np.random.randn(d, d) * 0.01
    B = np.random.randn(d, r) * 0.01
    A = np.random.randn(r, d) * 0.01
    x = np.random.randn(d)

    # TODO: Compute output using LoRA
    y_lora = None  # Replace: W0 @ x + B @ (A @ x)

    # TODO: Compute output using effective weight
    W_eff = W0 + B @ A
    y_full = W_eff @ x

    if y_lora is not None:
        return np.allclose(y_lora, y_full)
    return None


if __name__ == "__main__":
    print("Exercise 1: Power iteration")
    sigma, sigma_true = exercise_1_power_iteration()
    if sigma is not None:
        print(f"  Power iteration: {sigma:.6f}")
        print(f"  True (SVD):      {sigma_true:.6f}")
        print(f"  Pass: {abs(sigma - sigma_true) < 1e-4}")
    else:
        print("  Not implemented yet")

    print("\nExercise 2: Truncated SVD compression")
    results = exercise_2_truncated_svd_compression()
    if any(v is not None for v in results.values()):
        for k, err in results.items():
            print(f"  rank-{k:2d}: error = {err:.4f}" if err is not None else f"  rank-{k}: not done")
    else:
        print("  Not implemented yet")

    print("\nExercise 3: LoRA forward pass")
    match = exercise_3_lora_forward()
    if match is not None:
        print(f"  LoRA matches full: {match}")
    else:
        print("  Not implemented yet")
