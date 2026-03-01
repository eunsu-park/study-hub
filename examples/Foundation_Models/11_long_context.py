#!/usr/bin/env python3
"""
Foundation Models - Long Context Techniques
============================================

Implements key techniques for extending Transformer context windows:
1. Sliding Window Attention — O(n * w) instead of O(n^2)
2. ALiBi (Attention with Linear Biases) — position-aware without PE
3. Flash Attention simulation — memory-efficient exact attention
4. NTK-aware RoPE scaling — extend context at inference time

Why long context matters:
    Standard attention is O(n^2) in sequence length. A 128K-token context
    would need 128K^2 = 16B attention scores per layer per head.
    These techniques make long-context LLMs practical.

Requires: numpy, matplotlib (no PyTorch dependency)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def standard_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standard scaled dot-product attention: O(n^2) time and memory.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) @ V

    This is the baseline we want to improve upon.
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)  # (n, n) — this is the bottleneck
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ V, weights


def sliding_window_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                             window_size: int) -> np.ndarray:
    """
    Sliding Window Attention (used in Mistral, Longformer).

    Each token only attends to the nearest `window_size` tokens.
    Complexity: O(n * w) instead of O(n^2).

    Why this works:
        Most attention weight concentrates on nearby tokens anyway.
        Information from far-away tokens propagates through multiple
        layers (layer L attends to L * w tokens effectively).

    Args:
        Q, K, V: Query, Key, Value matrices of shape (seq_len, d_model)
        window_size: Number of tokens to attend to on each side

    Returns:
        Output of shape (seq_len, d_model)
    """
    seq_len, d_k = Q.shape
    output = np.zeros_like(V)

    for i in range(seq_len):
        # Why: Causal mask + window. Token i attends to
        # [max(0, i - window_size), i] (inclusive).
        start = max(0, i - window_size)
        end = i + 1  # Causal: can't see future tokens

        # Compute attention only within the window
        q = Q[i:i + 1]              # (1, d_k)
        k = K[start:end]            # (w, d_k) where w <= window_size + 1
        v = V[start:end]            # (w, d_k)

        scores = q @ k.T / np.sqrt(d_k)
        weights = np.exp(scores - scores.max())
        weights /= weights.sum()

        output[i] = weights @ v

    return output


def alibi_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                    num_heads: int = 1, head_idx: int = 0) -> np.ndarray:
    """
    ALiBi: Attention with Linear Biases (Press et al., 2022).

    Instead of position embeddings, ALiBi adds a linear bias to
    attention scores: score(q_i, k_j) -= m * |i - j|

    Why ALiBi is elegant:
        - No learned position parameters at all
        - Each head gets a different slope m, creating multi-scale attention
        - Extrapolates perfectly to longer sequences (just extend the bias)
        - Used in BLOOM, MPT, and as an alternative to RoPE

    Args:
        Q, K, V: Shape (seq_len, d_model)
        num_heads: Total number of attention heads
        head_idx: Index of current head (determines slope)

    Returns:
        Output of shape (seq_len, d_model)
    """
    seq_len, d_k = Q.shape

    # Compute standard attention scores
    scores = Q @ K.T / np.sqrt(d_k)  # (seq_len, seq_len)

    # Why: Slopes form a geometric sequence. Head 0 has the steepest slope
    # (most local attention), last head has the gentlest (most global).
    # This gives the model both local and long-range attention patterns.
    ratio = 2 ** (-8.0 / num_heads)
    m = ratio ** (head_idx + 1)

    # Build distance matrix: bias[i][j] = -m * |i - j|
    positions = np.arange(seq_len)
    distance = positions[:, None] - positions[None, :]  # (seq_len, seq_len)

    # Why: Causal mask — only attend to past. Future positions get -inf.
    causal_mask = np.triu(np.ones((seq_len, seq_len)) * (-1e9), k=1)

    # Apply linear bias (penalizes distant tokens)
    alibi_bias = -m * np.abs(distance)
    scores = scores + alibi_bias + causal_mask

    # Softmax
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)

    return weights @ V


def flash_attention_tiled(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                          block_size: int = 16) -> np.ndarray:
    """
    Simplified Flash Attention (Dao et al., 2022).

    Key insight: Compute attention in blocks to avoid materializing the
    full N x N attention matrix in memory.

    Standard attention:  O(N^2) memory  (stores full attention matrix)
    Flash attention:     O(N) memory    (processes in tiles)

    Why this matters:
        For N=128K, the attention matrix alone takes 128K^2 * 4 bytes = 64 GB.
        Flash attention never creates this matrix, instead computing softmax
        incrementally using the "online softmax" trick.

    This is a simplified NumPy simulation — real Flash Attention uses
    GPU SRAM tiling and fused CUDA kernels.
    """
    seq_len, d_k = Q.shape
    output = np.zeros_like(V)

    # Why: We maintain running statistics for numerically stable softmax.
    # m = running max of scores (for numerical stability)
    # l = running sum of exp(scores - m) (softmax denominator)
    m_running = np.full(seq_len, -np.inf)  # Running max per query
    l_running = np.zeros(seq_len)          # Running sum per query

    num_blocks = (seq_len + block_size - 1) // block_size

    for j_block in range(num_blocks):
        j_start = j_block * block_size
        j_end = min(j_start + block_size, seq_len)

        K_block = K[j_start:j_end]  # (block, d_k)
        V_block = V[j_start:j_end]  # (block, d_k)

        # Compute scores for this block
        scores = Q @ K_block.T / np.sqrt(d_k)  # (seq_len, block)

        # Apply causal mask within this block
        for i in range(seq_len):
            for j in range(j_end - j_start):
                if j_start + j > i:
                    scores[i, j] = -np.inf

        # Online softmax update (the key algorithmic insight)
        # Why: Standard softmax requires two passes (find max, then compute).
        # Online softmax combines them, enabling single-pass streaming.
        block_max = scores.max(axis=-1)  # (seq_len,)
        new_max = np.maximum(m_running, block_max)

        # Rescale previous running sums with the new max
        # Why: When max increases, previous exp values were too large
        # and need to be scaled down by exp(old_max - new_max).
        scale_old = np.exp(m_running - new_max)
        scale_new = np.exp(block_max - new_max)

        exp_scores = np.exp(scores - block_max[:, None]) * scale_new[:, None]
        l_new = scale_old * l_running + exp_scores.sum(axis=-1)

        # Update output: rescale old output and add new contribution
        for i in range(seq_len):
            if l_new[i] > 0:
                output[i] = (scale_old[i] * l_running[i] * output[i]
                             + exp_scores[i:i + 1] @ V_block) / l_new[i]

        m_running = new_max
        l_running = l_new

    return output


def ntk_rope_scaling(positions: np.ndarray, dim: int,
                     base: float = 10000.0, scale_factor: float = 4.0
                     ) -> np.ndarray:
    """
    NTK-aware RoPE scaling for context extension.

    Instead of linearly interpolating positions (which degrades quality),
    NTK scaling adjusts the frequency base to effectively spread the
    same number of "wavelengths" across a longer sequence.

    Why NTK scaling works:
        Linear interpolation: positions /= scale (loses resolution)
        NTK scaling: base *= scale^(d/(d-2)) (preserves resolution)

        High frequencies (local patterns) are preserved while low
        frequencies (long-range patterns) are stretched.

    Args:
        positions: Position indices
        dim: Embedding dimension
        base: Original theta base
        scale_factor: Context extension factor (e.g., 4x)

    Returns:
        Rotation angles for each (position, frequency) pair
    """
    # Why: The exponent d/(d-2) comes from neural tangent kernel theory.
    # It ensures the feature map's effective dimensionality is preserved.
    scaled_base = base * scale_factor ** (dim / (dim - 2))

    freqs = 1.0 / (scaled_base ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    angles = np.outer(positions, freqs)
    return angles


def visualize_techniques():
    """Compare attention patterns from different techniques."""
    np.random.seed(42)
    seq_len = 64
    d_model = 32

    Q = np.random.randn(seq_len, d_model) * 0.1
    K = np.random.randn(seq_len, d_model) * 0.1
    V = np.random.randn(seq_len, d_model) * 0.1

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 1. Standard attention
    _, std_weights = standard_attention(Q, K, V)
    # Apply causal mask for visualization
    causal = np.tril(np.ones((seq_len, seq_len)))
    std_weights *= causal
    std_weights /= std_weights.sum(axis=-1, keepdims=True).clip(1e-10)
    axes[0, 0].imshow(std_weights, cmap='viridis', aspect='auto')
    axes[0, 0].set_title("Standard Attention\n(O(n^2) memory)")

    # 2. Sliding window
    window_sizes = [8, 16]
    for idx, w in enumerate(window_sizes):
        # Build window mask for visualization
        mask = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            start = max(0, i - w)
            mask[i, start:i + 1] = 1
        sparse_weights = std_weights * mask
        sparse_weights /= sparse_weights.sum(axis=-1, keepdims=True).clip(1e-10)
        axes[0, idx + 1].imshow(sparse_weights, cmap='viridis', aspect='auto')
        axes[0, idx + 1].set_title(f"Sliding Window (w={w})\nO(n*w) memory")

    # 3. ALiBi biases for different heads
    for idx, head in enumerate([0, 3, 7]):
        num_heads = 8
        ratio = 2 ** (-8.0 / num_heads)
        m = ratio ** (head + 1)
        positions = np.arange(seq_len)
        distance = np.abs(positions[:, None] - positions[None, :])
        bias = -m * distance
        # Apply causal mask
        bias += np.triu(np.ones((seq_len, seq_len)) * (-1e9), k=1)
        axes[1, idx].imshow(bias, cmap='RdBu_r', aspect='auto',
                            vmin=-5, vmax=0)
        axes[1, idx].set_title(f"ALiBi Bias (head={head}, m={m:.4f})")
        axes[1, idx].set_xlabel("Key position")
        if idx == 0:
            axes[1, idx].set_ylabel("Query position")

    plt.suptitle("Long Context Attention Techniques", fontsize=14)
    plt.tight_layout()
    plt.savefig("long_context_techniques.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: long_context_techniques.png")


if __name__ == "__main__":
    np.random.seed(42)
    seq_len = 64
    d_model = 32

    Q = np.random.randn(seq_len, d_model) * 0.1
    K = np.random.randn(seq_len, d_model) * 0.1
    V = np.random.randn(seq_len, d_model) * 0.1

    print("=" * 60)
    print("Long Context Techniques Demonstration")
    print("=" * 60)
    print(f"\nSequence length: {seq_len}, Model dim: {d_model}")

    # 1. Standard attention (baseline)
    print("\n--- 1. Standard Attention ---")
    out_std, _ = standard_attention(Q, K, V)
    print(f"Output shape: {out_std.shape}")
    print(f"Memory: O(n^2) = O({seq_len}^2) = {seq_len ** 2} scores")

    # 2. Sliding window
    print("\n--- 2. Sliding Window Attention ---")
    for w in [8, 16, 32]:
        out_sw = sliding_window_attention(Q, K, V, window_size=w)
        error = np.mean((out_std - out_sw) ** 2)
        print(f"  Window={w:2d}: MSE vs full = {error:.6f}, "
              f"Memory: O(n*w) = O({seq_len}*{w}) = {seq_len * w} scores")

    # 3. ALiBi
    print("\n--- 3. ALiBi (Attention with Linear Biases) ---")
    num_heads = 8
    for h in range(num_heads):
        out_alibi = alibi_attention(Q, K, V, num_heads=num_heads, head_idx=h)
        ratio = 2 ** (-8.0 / num_heads)
        m = ratio ** (h + 1)
        print(f"  Head {h}: slope m = {m:.6f}")

    # 4. Flash attention (tiled)
    print("\n--- 4. Flash Attention (Tiled) ---")
    for bs in [8, 16, 32]:
        out_flash = flash_attention_tiled(Q, K, V, block_size=bs)
        error = np.mean((out_std - out_flash) ** 2)
        print(f"  Block size={bs:2d}: MSE vs standard = {error:.2e} "
              f"(should be ~0, exact same result)")

    # 5. NTK-aware RoPE scaling
    print("\n--- 5. NTK-aware RoPE Scaling ---")
    dim = 64
    positions = np.arange(256)
    for scale in [1.0, 2.0, 4.0, 8.0]:
        angles = ntk_rope_scaling(positions, dim, scale_factor=scale)
        max_angle = angles[-1, 0]  # Highest frequency at last position
        print(f"  Scale={scale:.0f}x: max rotation angle (freq 0) = {max_angle:.2f} rad")

    # Visualize
    print("\nGenerating visualization...")
    visualize_techniques()
