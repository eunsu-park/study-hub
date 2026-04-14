"""
14. From Fundamentals to Deep Learning - Exercises
====================================================
Lesson 14: From Fundamentals to Deep Learning

Exercises cover:
  1. Implement 1D convolution
  2. Implement vanilla RNN cell
  3. Implement scaled dot-product attention
"""

import numpy as np


# ============================================================
# Exercise 1: 1D Convolution
# Implement 1D convolution and compare with FC layer.
# ============================================================
def exercise_1_conv1d():
    """Implement 1D convolution."""
    print("=" * 60)
    print("Exercise 1: 1D Convolution")
    print("=" * 60)

    def conv1d(x, kernel, stride=1):
        # TODO: Implement 1D convolution (valid mode)
        # x: input signal, shape (n,)
        # kernel: filter, shape (k,)
        # Returns: output, shape ((n - k) // stride + 1,)
        raise NotImplementedError("Implement conv1d")

    # Test
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    kernel = np.array([1, 0, -1], dtype=float)  # edge detector
    # output = conv1d(x, kernel)
    # Expected: [1-3, 2-4, 3-5, 4-6, 5-7, 6-8] = [-2, -2, -2, -2, -2, -2]
    # print(f"  Output: {output}")


# ============================================================
# Exercise 2: Vanilla RNN Cell
# Implement a simple RNN cell and process a sequence.
# ============================================================
def exercise_2_rnn_cell():
    """Implement vanilla RNN cell."""
    print("\n" + "=" * 60)
    print("Exercise 2: Vanilla RNN Cell")
    print("=" * 60)

    def rnn_step(x_t, h_prev, W_xh, W_hh, b_h):
        # TODO: Implement one RNN step
        # h_t = tanh(W_hh @ h_prev + W_xh @ x_t + b_h)
        raise NotImplementedError("Implement RNN step")

    # Test: process a 5-step sequence
    np.random.seed(42)
    input_size, hidden_size = 3, 4
    W_xh = np.random.randn(hidden_size, input_size) * 0.1
    W_hh = np.random.randn(hidden_size, hidden_size) * 0.1
    b_h = np.zeros((hidden_size, 1))

    h = np.zeros((hidden_size, 1))
    for t in range(5):
        x_t = np.random.randn(input_size, 1)
        # h = rnn_step(x_t, h, W_xh, W_hh, b_h)
        # print(f"  t={t}: h = {h.ravel()}")


# ============================================================
# Exercise 3: Scaled Dot-Product Attention
# Implement attention mechanism from scratch.
# ============================================================
def exercise_3_attention():
    """Implement scaled dot-product attention."""
    print("\n" + "=" * 60)
    print("Exercise 3: Scaled Dot-Product Attention")
    print("=" * 60)

    def attention(Q, K, V):
        # TODO: Implement Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
        # Q: (seq_len, d_k)
        # K: (seq_len, d_k)
        # V: (seq_len, d_v)
        # Returns: (seq_len, d_v), attention_weights (seq_len, seq_len)
        raise NotImplementedError("Implement attention")

    # Test
    np.random.seed(42)
    seq_len, d_k = 4, 8
    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_k)

    # output, weights = attention(Q, K, V)
    # print(f"  Output shape: {output.shape}")
    # print(f"  Weights sum per row: {weights.sum(axis=1)}")  # should all be 1.0


if __name__ == "__main__":
    exercise_1_conv1d()
    exercise_2_rnn_cell()
    exercise_3_attention()
