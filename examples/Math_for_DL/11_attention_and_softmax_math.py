"""
Attention and Softmax Math

Demonstrates the mathematics of attention mechanisms:
- Scaled dot-product attention
- Why scaling by sqrt(d_k) matters
- Temperature scaling effects
- Multi-head attention
- Causal masking
- Attention gradient computation

Dependencies: numpy, matplotlib
"""

import numpy as np


def scaled_attention(Q, K, V, mask=None):
    """Scaled dot-product attention."""
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    s_max = scores.max(axis=-1, keepdims=True)
    e = np.exp(scores - s_max)
    weights = e / e.sum(axis=-1, keepdims=True)
    return weights @ V, weights


def attention_demo():
    """Basic attention demonstration."""
    print("=" * 60)
    print("SCALED DOT-PRODUCT ATTENTION")
    print("=" * 60)
    np.random.seed(42)
    T, d = 6, 8
    Q = np.random.randn(T, d)
    K = np.random.randn(T, d)
    V = np.random.randn(T, d)
    out, w = scaled_attention(Q, K, V)
    print(f"Q, K, V: ({T}, {d})")
    print(f"Attention weights: {w.shape}, row sums: {w.sum(axis=-1).round(4)}")
    print(f"Output: {out.shape}")


def scaling_importance():
    """Show why sqrt(d_k) scaling matters."""
    print("\n" + "=" * 60)
    print("WHY SCALE BY sqrt(d_k)?")
    print("=" * 60)
    np.random.seed(42)
    for d_k in [4, 16, 64, 256]:
        q = np.random.randn(d_k)
        K = np.random.randn(50, d_k)
        scores_unscaled = K @ q
        scores_scaled = scores_unscaled / np.sqrt(d_k)
        # Softmax entropy
        def softmax_entropy(z):
            e = np.exp(z - np.max(z))
            s = e / e.sum()
            return -np.sum(s * np.log(s + 1e-10))
        H_us = softmax_entropy(scores_unscaled)
        H_sc = softmax_entropy(scores_scaled)
        print(f"  d_k={d_k:3d}: std_unscaled={scores_unscaled.std():.2f}, "
              f"entropy_unscaled={H_us:.2f}, entropy_scaled={H_sc:.2f}")


def temperature_demo():
    """Temperature scaling effects."""
    print("\n" + "=" * 60)
    print("TEMPERATURE SCALING")
    print("=" * 60)
    z = np.array([2.0, 1.0, 0.5, -1.0, -2.0])
    for tau in [0.1, 0.5, 1.0, 2.0, 5.0]:
        e = np.exp(z/tau - np.max(z/tau))
        s = e / e.sum()
        H = -np.sum(s * np.log(s + 1e-10))
        print(f"  tau={tau:4.1f}: max_p={s.max():.4f}, entropy={H:.4f}")


def causal_attention():
    """Causal (autoregressive) masking."""
    print("\n" + "=" * 60)
    print("CAUSAL ATTENTION")
    print("=" * 60)
    T, d = 5, 4
    np.random.seed(42)
    Q = np.random.randn(T, d)
    K = np.random.randn(T, d)
    V = np.random.randn(T, d)
    mask = np.tril(np.ones((T, T), dtype=bool))
    _, w = scaled_attention(Q, K, V, mask)
    print("Causal attention weights (lower triangular):")
    print(w.round(3))
    print(f"Upper triangle all zeros: {np.allclose(np.triu(w, 1), 0, atol=1e-6)}")


def multi_head_demo():
    """Multi-head attention."""
    print("\n" + "=" * 60)
    print("MULTI-HEAD ATTENTION")
    print("=" * 60)
    T, d_model, n_heads = 8, 64, 8
    d_head = d_model // n_heads
    np.random.seed(42)
    X = np.random.randn(T, d_model) * 0.1
    W_Q = np.random.randn(d_model, d_model) * 0.02
    W_K = np.random.randn(d_model, d_model) * 0.02
    W_V = np.random.randn(d_model, d_model) * 0.02
    W_O = np.random.randn(d_model, d_model) * 0.02

    Q = (X @ W_Q).reshape(T, n_heads, d_head).transpose(1, 0, 2)
    K = (X @ W_K).reshape(T, n_heads, d_head).transpose(1, 0, 2)
    V = (X @ W_V).reshape(T, n_heads, d_head).transpose(1, 0, 2)

    scores = np.einsum('htd,hsd->hts', Q, K) / np.sqrt(d_head)
    sm = scores.max(axis=-1, keepdims=True)
    attn = np.exp(scores - sm)
    attn = attn / attn.sum(axis=-1, keepdims=True)
    out = np.einsum('hts,hsd->htd', attn, V)
    out = out.transpose(1, 0, 2).reshape(T, d_model) @ W_O

    print(f"Input: {X.shape}")
    print(f"Per-head Q/K/V: ({n_heads}, {T}, {d_head})")
    print(f"Attention per head: {attn.shape}")
    print(f"Output: {out.shape}")


if __name__ == "__main__":
    attention_demo()
    scaling_importance()
    temperature_demo()
    causal_attention()
    multi_head_demo()
