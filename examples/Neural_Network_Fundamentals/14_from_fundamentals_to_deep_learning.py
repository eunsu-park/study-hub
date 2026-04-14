"""
14. From Fundamentals to Deep Learning
========================================
Implements simplified versions of CNN convolution, RNN cell,
and self-attention to preview deep learning architectures.

Key Concepts:
  - 1D convolution operation
  - Vanilla RNN cell
  - Scaled dot-product attention
"""

import numpy as np

np.random.seed(42)


# ============================================================
# 1. 1D Convolution
# ============================================================
print("=" * 60)
print("1. 1D Convolution (CNN Preview)")
print("=" * 60)


def conv1d(x, kernel):
    """1D convolution (valid mode).

    Args:
        x: Input signal, shape (n,)
        kernel: Filter weights, shape (k,)
    Returns:
        Output, shape (n - k + 1,)
    """
    n, k = len(x), len(kernel)
    output = np.zeros(n - k + 1)
    for i in range(n - k + 1):
        output[i] = np.dot(x[i:i + k], kernel)
    return output


# Edge detection kernel
x_signal = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0], dtype=float)
edge_kernel = np.array([-1, 0, 1], dtype=float)
edges = conv1d(x_signal, edge_kernel)

print(f"\nInput signal:    {x_signal}")
print(f"Edge kernel:     {edge_kernel}")
print(f"Conv output:     {edges}")
print("Edges detected at transitions (nonzero values)")


# Compare parameters: conv vs fully connected
n_input = 100
fc_params = n_input * 10  # 10 output neurons
conv_params = 3 * 10  # 3-wide kernel, 10 filters
print(f"\nFor input size {n_input}, 10 outputs:")
print(f"  FC parameters:   {fc_params}")
print(f"  Conv parameters: {conv_params} ({fc_params // conv_params}x fewer!)")


# ============================================================
# 2. Vanilla RNN Cell
# ============================================================
print("\n" + "=" * 60)
print("2. Vanilla RNN Cell (RNN Preview)")
print("=" * 60)


class RNNCell:
    """Simple RNN cell: h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t + b)."""

    def __init__(self, input_size, hidden_size):
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.1
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.1
        self.b_h = np.zeros((hidden_size, 1))

    def forward(self, x_t, h_prev):
        """Process one time step."""
        h_t = np.tanh(self.W_hh @ h_prev + self.W_xh @ x_t + self.b_h)
        return h_t


# Process a 5-step sequence
rnn = RNNCell(input_size=3, hidden_size=4)
h = np.zeros((4, 1))

print(f"\nProcessing 5-step sequence (input_dim=3, hidden_dim=4):")
for t in range(5):
    x_t = np.random.randn(3, 1)
    h = rnn.forward(x_t, h)
    print(f"  t={t}: h = [{', '.join(f'{v:.3f}' for v in h.ravel())}]")

print("\nNotice: hidden state evolves at each step, carrying information forward.")


# ============================================================
# 3. Scaled Dot-Product Attention
# ============================================================
print("\n" + "=" * 60)
print("3. Scaled Dot-Product Attention (Transformer Preview)")
print("=" * 60)


def scaled_dot_product_attention(Q, K, V):
    """
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    Args:
        Q: Queries, shape (seq_len, d_k)
        K: Keys, shape (seq_len, d_k)
        V: Values, shape (seq_len, d_v)
    Returns:
        output: shape (seq_len, d_v)
        weights: attention weights, shape (seq_len, seq_len)
    """
    d_k = Q.shape[1]
    scores = Q @ K.T / np.sqrt(d_k)
    # Softmax
    scores_exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    weights = scores_exp / np.sum(scores_exp, axis=1, keepdims=True)
    output = weights @ V
    return output, weights


# 4 tokens, dimension 8
seq_len, d_model = 4, 8
X = np.random.randn(seq_len, d_model)

# Linear projections
W_Q = np.random.randn(d_model, d_model) * 0.1
W_K = np.random.randn(d_model, d_model) * 0.1
W_V = np.random.randn(d_model, d_model) * 0.1

Q = X @ W_Q
K = X @ W_K
V = X @ W_V

output, attn_weights = scaled_dot_product_attention(Q, K, V)

print(f"\nInput: {seq_len} tokens, d_model={d_model}")
print(f"Attention weights (how much each token attends to others):")
for i in range(seq_len):
    weights_str = ', '.join(f'{w:.3f}' for w in attn_weights[i])
    print(f"  Token {i}: [{weights_str}]")
print(f"Output shape: {output.shape}")
print(f"Weights sum per row: {attn_weights.sum(axis=1)} (all 1.0)")


# ============================================================
# 4. Architecture Comparison
# ============================================================
print("\n" + "=" * 60)
print("4. Architecture Selection Guide")
print("=" * 60)

guide = [
    ("Images", "CNN", "Local features, translation invariance"),
    ("Short sequences", "LSTM/GRU", "Temporal dependencies"),
    ("Long sequences", "Transformer", "Parallel, long-range attention"),
    ("Tabular data", "MLP", "Simple, effective baseline"),
    ("Graphs", "GNN", "Node/edge relationships"),
]

print(f"\n{'Data Type':<18} {'Architecture':<14} {'Why'}")
print("-" * 65)
for data, arch, why in guide:
    print(f"{data:<18} {arch:<14} {why}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
