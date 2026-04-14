"""
Exercises for Lesson 11: Attention and Softmax Math
Topic: Math_for_DL

Complete the TODO sections.
"""

import numpy as np


def exercise_1_causal_attention():
    """Implement causal (autoregressive) attention.

    Verify that position i only attends to positions <= i.
    """
    T, d = 6, 8
    np.random.seed(42)
    Q = np.random.randn(T, d)
    K = np.random.randn(T, d)
    V = np.random.randn(T, d)

    # TODO: Implement causal attention
    # 1. Compute scores = Q @ K^T / sqrt(d)
    # 2. Create causal mask (lower triangular)
    # 3. Set masked positions to -1e9
    # 4. Apply softmax
    # 5. Multiply by V
    output = None
    weights = None

    if weights is not None:
        # Verify upper triangle is zero
        upper_zero = np.allclose(np.triu(weights, 1), 0, atol=1e-6)
        return output, weights, upper_zero
    return None, None, None


def exercise_2_scaling_entropy():
    """Plot attention entropy vs d_k to show why scaling is necessary.

    Without scaling, entropy decreases as d_k increases (attention becomes peaky).
    """
    np.random.seed(42)
    d_k_values = [4, 8, 16, 32, 64, 128, 256, 512]

    entropies_unscaled = []
    entropies_scaled = []

    for d_k in d_k_values:
        q = np.random.randn(d_k)
        K = np.random.randn(20, d_k)

        # TODO: Compute entropy of attention weights (unscaled and scaled)
        # Unscaled: softmax(K @ q)
        # Scaled: softmax(K @ q / sqrt(d_k))
        H_unscaled = None
        H_scaled = None

        entropies_unscaled.append(H_unscaled)
        entropies_scaled.append(H_scaled)

    return d_k_values, entropies_unscaled, entropies_scaled


def exercise_3_multi_head_attention():
    """Implement multi-head attention from scratch.

    Verify input/output shapes at each step.
    """
    T, d_model, n_heads = 8, 32, 4
    d_head = d_model // n_heads
    np.random.seed(42)

    X = np.random.randn(T, d_model) * 0.1
    W_Q = np.random.randn(d_model, d_model) * 0.02
    W_K = np.random.randn(d_model, d_model) * 0.02
    W_V = np.random.randn(d_model, d_model) * 0.02
    W_O = np.random.randn(d_model, d_model) * 0.02

    # TODO: Implement multi-head attention
    # 1. Project: Q = X @ W_Q, K = X @ W_K, V = X @ W_V
    # 2. Reshape to (n_heads, T, d_head)
    # 3. Compute attention per head
    # 4. Concatenate heads
    # 5. Project with W_O
    output = None

    if output is not None:
        return output.shape == (T, d_model)
    return None


if __name__ == "__main__":
    print("Exercise 1: Causal attention")
    out, w, upper_zero = exercise_1_causal_attention()
    if out is not None:
        print(f"  Output shape: {out.shape}")
        print(f"  Upper triangle zero: {upper_zero}")
    else:
        print("  Not implemented yet")

    print("\nExercise 2: Scaling entropy")
    dks, eu, es = exercise_2_scaling_entropy()
    if eu[0] is not None:
        print(f"  d_k=4:   H_unscaled={eu[0]:.3f}, H_scaled={es[0]:.3f}")
        print(f"  d_k=256: H_unscaled={eu[-2]:.3f}, H_scaled={es[-2]:.3f}")
        print(f"  Scaling preserves entropy: {abs(es[0] - es[-2]) < abs(eu[0] - eu[-2])}")
    else:
        print("  Not implemented yet")

    print("\nExercise 3: Multi-head attention")
    correct = exercise_3_multi_head_attention()
    if correct is not None:
        print(f"  Output shape correct: {correct}")
    else:
        print("  Not implemented yet")
