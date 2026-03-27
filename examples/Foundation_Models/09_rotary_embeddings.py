#!/usr/bin/env python3
"""
Foundation Models - Rotary Position Embeddings (RoPE)
=====================================================

Implements Rotary Position Embedding from scratch, the position encoding
used in LLaMA, Mistral, Qwen, and most modern LLMs.

Key Idea:
    Instead of adding position vectors to token embeddings, RoPE encodes
    position by rotating query/key vectors in 2D subspaces. Two tokens
    at positions m and n interact via a rotation of angle (m - n) * theta,
    making attention naturally depend on *relative* distance.

Why RoPE over sinusoidal?
    - Relative position is encoded implicitly (no need for explicit bias)
    - Decays gracefully with distance (far tokens attend less)
    - Compatible with KV-cache (no need to recompute for cached keys)
    - Extrapolates better to unseen sequence lengths

Requires: numpy, matplotlib (no PyTorch dependency)
"""

import numpy as np
import matplotlib.pyplot as plt


def build_rotation_matrix(dim: int, position: int, theta_base: float = 10000.0):
    """
    Build the RoPE rotation matrix for a single position.

    For each pair of dimensions (2i, 2i+1), the rotation angle is:
        angle_i = position * theta_i
        theta_i = 1 / (base^(2i/d))

    Why theta_base=10000?
        Lower frequencies (large i) use small theta, so they rotate slowly
        and encode long-range position. Higher frequencies encode fine-grained
        local position. 10000 gives a good frequency spread.

    Args:
        dim: Embedding dimension (must be even)
        position: Token position in the sequence
        theta_base: Base for frequency computation

    Returns:
        (dim, dim) rotation matrix
    """
    assert dim % 2 == 0, "Dimension must be even for RoPE"

    # Why: Each pair of dimensions gets its own frequency, creating a
    # multi-scale position encoding (like Fourier features).
    freqs = 1.0 / (theta_base ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    angles = position * freqs  # Shape: (dim/2,)

    # Build block-diagonal rotation matrix
    # Why: Block-diagonal structure means each 2D subspace rotates
    # independently. This preserves the dot product structure.
    R = np.eye(dim)
    for i in range(dim // 2):
        cos_a = np.cos(angles[i])
        sin_a = np.sin(angles[i])
        # 2x2 rotation block for dimensions (2i, 2i+1)
        R[2 * i, 2 * i] = cos_a
        R[2 * i, 2 * i + 1] = -sin_a
        R[2 * i + 1, 2 * i] = sin_a
        R[2 * i + 1, 2 * i + 1] = cos_a

    return R


def apply_rope_efficient(x: np.ndarray, positions: np.ndarray,
                         theta_base: float = 10000.0):
    """
    Apply RoPE to a batch of vectors efficiently (without explicit matrix).

    This is the standard implementation used in practice (e.g., LLaMA).
    Instead of matrix multiplication, we reshape into pairs and apply
    rotation formula directly.

    Why this is faster:
        Matrix multiply: O(d^2) per position
        Pair-wise rotation: O(d) per position

    Args:
        x: Input tensor of shape (seq_len, dim)
        positions: Position indices of shape (seq_len,)
        theta_base: Frequency base

    Returns:
        Rotated tensor of same shape
    """
    seq_len, dim = x.shape
    assert dim % 2 == 0

    # Compute rotation angles for all positions and frequency pairs
    freqs = 1.0 / (theta_base ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    # Why: outer product gives all (position, frequency) combinations at once
    angles = np.outer(positions, freqs)  # (seq_len, dim/2)

    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)

    # Split into even/odd dimension pairs
    x_even = x[:, 0::2]  # Dimensions 0, 2, 4, ...
    x_odd = x[:, 1::2]   # Dimensions 1, 3, 5, ...

    # Apply 2D rotation: [cos -sin; sin cos] @ [x_even; x_odd]
    # Why: This avoids constructing the full rotation matrix entirely.
    out_even = x_even * cos_angles - x_odd * sin_angles
    out_odd = x_even * sin_angles + x_odd * cos_angles

    # Interleave back
    out = np.empty_like(x)
    out[:, 0::2] = out_even
    out[:, 1::2] = out_odd

    return out


def sinusoidal_position_encoding(seq_len: int, dim: int):
    """
    Classic sinusoidal position encoding (Vaswani et al., 2017) for comparison.

    Unlike RoPE, this is ADDED to embeddings rather than applied as rotation.
    """
    positions = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, dim, 2) * (-np.log(10000.0) / dim))
    pe = np.zeros((seq_len, dim))
    pe[:, 0::2] = np.sin(positions * div_term)
    pe[:, 1::2] = np.cos(positions * div_term)
    return pe


def demonstrate_relative_attention():
    """
    Show that RoPE encodes relative position in the dot product.

    Key property: q_m^T @ k_n = f(x_m, x_n, m - n)
    The attention score depends only on the relative distance (m - n),
    not on absolute positions m and n individually.
    """
    dim = 8
    np.random.seed(42)

    # Create two random embedding vectors
    x_q = np.random.randn(dim)
    x_k = np.random.randn(dim)

    print("=== RoPE Relative Position Property ===\n")
    print("If q is at position m and k at position n,")
    print("then q_rope^T @ k_rope depends only on (m - n).\n")

    # Test: different absolute positions, same relative distance
    for m, n in [(5, 3), (10, 8), (100, 98)]:
        R_m = build_rotation_matrix(dim, m)
        R_n = build_rotation_matrix(dim, n)
        q_rotated = R_m @ x_q
        k_rotated = R_n @ x_k
        score = np.dot(q_rotated, k_rotated)
        print(f"  positions ({m:3d}, {n:3d}), distance = {m - n}: score = {score:.6f}")

    # Why: All scores should be identical because m - n = 2 in every case.
    # This proves RoPE encodes relative position, not absolute.
    print("\n  All scores are equal because the relative distance is the same!")


def visualize_rope():
    """
    Visualize RoPE frequency patterns and compare with sinusoidal PE.
    """
    dim = 64
    seq_len = 128

    # Generate RoPE patterns
    positions = np.arange(seq_len)
    freqs = 1.0 / (10000.0 ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    angle_matrix = np.outer(positions, freqs)  # (seq_len, dim/2)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: RoPE rotation angles across positions and dimensions
    im1 = axes[0, 0].imshow(angle_matrix[:, :16].T, aspect='auto', cmap='coolwarm')
    axes[0, 0].set_xlabel("Position")
    axes[0, 0].set_ylabel("Frequency index")
    axes[0, 0].set_title("RoPE Rotation Angles (first 16 freq pairs)")
    plt.colorbar(im1, ax=axes[0, 0])

    # Plot 2: Frequency spectrum
    axes[0, 1].semilogy(freqs, 'b-o', markersize=3)
    axes[0, 1].set_xlabel("Frequency index i")
    axes[0, 1].set_ylabel("theta_i = 1 / base^(2i/d)")
    axes[0, 1].set_title("RoPE Frequency Spectrum")
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Attention decay with distance
    # Why: This shows how RoPE naturally creates a distance-dependent
    # attention bias, even without explicit position bias terms.
    q = np.random.randn(1, dim)
    k = np.random.randn(1, dim)
    distances = np.arange(0, 64)
    scores = []
    for d in distances:
        q_rot = apply_rope_efficient(q, np.array([0]))
        k_rot = apply_rope_efficient(k, np.array([d]))
        scores.append(np.dot(q_rot[0], k_rot[0]))

    axes[1, 0].plot(distances, scores, 'g-', linewidth=1.5)
    axes[1, 0].set_xlabel("Relative distance |m - n|")
    axes[1, 0].set_ylabel("Dot product (unnormalized score)")
    axes[1, 0].set_title("Attention Score vs. Relative Distance")
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Compare RoPE rotation vs sinusoidal addition
    sinusoidal = sinusoidal_position_encoding(seq_len, dim)
    axes[1, 1].plot(sinusoidal[:, 0], sinusoidal[:, 1], 'b.', alpha=0.4,
                    label='Sinusoidal PE (dims 0,1)', markersize=2)
    # Show RoPE trajectory by rotating a fixed vector
    v = np.array([1.0, 0.0])
    rope_traj = []
    for pos in range(seq_len):
        angle = pos * freqs[0]
        rope_traj.append([np.cos(angle) * v[0] - np.sin(angle) * v[1],
                          np.sin(angle) * v[0] + np.cos(angle) * v[1]])
    rope_traj = np.array(rope_traj)
    axes[1, 1].plot(rope_traj[:, 0], rope_traj[:, 1], 'r.', alpha=0.4,
                    label='RoPE rotation (freq 0)', markersize=2)
    axes[1, 1].set_title("Sinusoidal (additive) vs RoPE (rotation)")
    axes[1, 1].legend()
    axes[1, 1].set_aspect('equal')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Rotary Position Embeddings (RoPE) Analysis", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("rope_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: rope_analysis.png")


if __name__ == "__main__":
    # 1. Demonstrate the relative position property
    demonstrate_relative_attention()

    print("\n" + "=" * 60)

    # 2. Benchmark: matrix vs efficient implementation
    dim = 64
    seq_len = 32
    np.random.seed(0)
    x = np.random.randn(seq_len, dim)
    positions = np.arange(seq_len)

    # Matrix version (for validation)
    x_matrix = np.zeros_like(x)
    for i in range(seq_len):
        R = build_rotation_matrix(dim, i)
        x_matrix[i] = R @ x[i]

    # Efficient version
    x_efficient = apply_rope_efficient(x, positions)

    max_diff = np.max(np.abs(x_matrix - x_efficient))
    print(f"\nMax difference between matrix and efficient: {max_diff:.2e}")
    print("(Should be ~1e-15, confirming they are mathematically equivalent)")

    # 3. Visualize
    print("\nGenerating visualization...")
    visualize_rope()
