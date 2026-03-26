"""
Exercises for Lesson 13: Linear Algebra for Deep Learning
Topic: Math_for_AI

Solutions to practice problems from the lesson.
"""

import numpy as np


# === Exercise 1: Tensor Manipulation ===
# Problem: (a) Transform (32,3,64,64) to (32,64,64,3)
# (b) Compute spatial mean per image: (32,3)
# (c) Per-channel normalization

def exercise_1():
    """Tensor manipulation operations for image data."""
    np.random.seed(42)

    # Simulated image batch: (batch, channels, height, width)
    batch = np.random.randn(32, 3, 64, 64).astype(np.float32)
    print(f"Original shape: {batch.shape}  (N, C, H, W)")

    # (a) NCHW -> NHWC
    batch_nhwc = np.transpose(batch, (0, 2, 3, 1))
    print(f"(a) After transpose to NHWC: {batch_nhwc.shape}")

    # Verify data integrity
    assert np.allclose(batch[0, 1, 2, 3], batch_nhwc[0, 2, 3, 1])
    print(f"    Data integrity check: passed")

    # (b) Spatial mean for each image -> (32, 3)
    # Average over H and W dimensions (axes 2, 3)
    spatial_mean = np.mean(batch, axis=(2, 3))
    print(f"\n(b) Spatial mean shape: {spatial_mean.shape}")
    print(f"    First image channel means: {np.round(spatial_mean[0], 4)}")

    # (c) Per-channel std and normalization
    # Compute std per channel across spatial dimensions
    channel_std = np.std(batch, axis=(2, 3), keepdims=True)
    channel_mean = np.mean(batch, axis=(2, 3), keepdims=True)
    print(f"\n(c) Channel std shape (with keepdims): {channel_std.shape}")

    # Normalize: (x - mean) / std
    batch_normalized = (batch - channel_mean) / (channel_std + 1e-5)
    print(f"    Normalized shape: {batch_normalized.shape}")

    # Verify normalization
    norm_mean = np.mean(batch_normalized, axis=(2, 3))
    norm_std = np.std(batch_normalized, axis=(2, 3))
    print(f"    After normalization, channel means (sample 0): {np.round(norm_mean[0], 6)}")
    print(f"    After normalization, channel stds  (sample 0): {np.round(norm_std[0], 4)}")


# === Exercise 2: einsum Mastery ===
# Problem: Implement various tensor operations using einsum.

def exercise_2():
    """Advanced einsum operations."""
    np.random.seed(42)

    # (a) Diagonal sum of batch matrices: (batch, n, n) -> (batch,)
    batch_size = 4
    n = 5
    batch_matrices = np.random.randn(batch_size, n, n)

    # Using einsum: trace of each matrix
    traces_einsum = np.einsum('bii->b', batch_matrices)
    traces_loop = np.array([np.trace(batch_matrices[i]) for i in range(batch_size)])

    print("(a) Batch trace: (batch, n, n) -> (batch,)")
    print(f"    einsum: {np.round(traces_einsum, 4)}")
    print(f"    loop:   {np.round(traces_loop, 4)}")
    print(f"    match: {np.allclose(traces_einsum, traces_loop)}")

    # (b) Multi-head attention Q @ K^T: (batch, heads, seq, d) x (batch, heads, seq, d) -> (batch, heads, seq, seq)
    batch_size = 2
    n_heads = 4
    seq_len = 8
    d_k = 16

    Q = np.random.randn(batch_size, n_heads, seq_len, d_k)
    K = np.random.randn(batch_size, n_heads, seq_len, d_k)

    # Q @ K^T using einsum
    attn_scores_einsum = np.einsum('bhsd,bhtd->bhst', Q, K)
    attn_scores_matmul = Q @ K.transpose(0, 1, 3, 2)

    print(f"\n(b) Multi-head attention scores: Q @ K^T")
    print(f"    Q shape: {Q.shape}, K shape: {K.shape}")
    print(f"    Scores shape: {attn_scores_einsum.shape}")
    print(f"    match: {np.allclose(attn_scores_einsum, attn_scores_matmul)}")

    # (c) 4D tensor contraction: (a,b,c,d) x (c,d,e,f) -> (a,b,e,f)
    a, b, c, d, e, f = 2, 3, 4, 5, 6, 7
    T1 = np.random.randn(a, b, c, d)
    T2 = np.random.randn(c, d, e, f)

    result_einsum = np.einsum('abcd,cdef->abef', T1, T2)

    # Manual: contract over c and d
    result_manual = np.tensordot(T1, T2, axes=([2, 3], [0, 1]))

    print(f"\n(c) 4D tensor contraction")
    print(f"    T1: {T1.shape}, T2: {T2.shape}")
    print(f"    Result: {result_einsum.shape}")
    print(f"    match: {np.allclose(result_einsum, result_manual)}")


# === Exercise 3: Numerical Stability ===
# Problem: Implement stable log-softmax using log-sum-exp trick.

def exercise_3():
    """Numerically stable log-softmax implementation."""

    def log_softmax_naive(x):
        """Naive implementation (numerically unstable for large values)."""
        exp_x = np.exp(x)
        return np.log(exp_x / np.sum(exp_x))

    def log_softmax_stable(x):
        """Stable implementation using log-sum-exp trick."""
        # log(exp(x_i) / sum(exp(x_j)))
        # = x_i - log(sum(exp(x_j)))
        # = x_i - (max(x) + log(sum(exp(x_j - max(x)))))
        c = np.max(x)
        logsumexp = c + np.log(np.sum(np.exp(x - c)))
        return x - logsumexp

    # (a) Test with normal values
    x_normal = np.array([1.0, 2.0, 3.0])
    print("(a) Normal values:", x_normal)
    print(f"    Naive:  {log_softmax_naive(x_normal)}")
    print(f"    Stable: {log_softmax_stable(x_normal)}")
    print(f"    Match:  {np.allclose(log_softmax_naive(x_normal), log_softmax_stable(x_normal))}")

    # (b) Test with very large values
    x_large = np.array([1000.0, 2000.0, 3000.0])
    print(f"\n(b) Large values: {x_large}")

    try:
        result_naive = log_softmax_naive(x_large)
        print(f"    Naive:  {result_naive}")
    except (RuntimeWarning, FloatingPointError):
        print("    Naive:  OVERFLOW ERROR")

    # Suppress runtime warnings for naive computation
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        result_naive = log_softmax_naive(x_large)
        has_nan = np.any(np.isnan(result_naive)) or np.any(np.isinf(result_naive))

    result_stable = log_softmax_stable(x_large)
    print(f"    Naive result has NaN/Inf: {has_nan}")
    print(f"    Stable: {result_stable}")
    print(f"    Stable result valid: {not np.any(np.isnan(result_stable))}")

    # (c) Verify: log_softmax values should sum to 0 in exp (i.e., softmax sums to 1)
    print(f"\n(c) Verification:")
    print(f"    exp(log_softmax) sums to: {np.sum(np.exp(result_stable)):.10f}")
    print(f"    Expected: 1.0")


# === Exercise 4: Initialization Experiments ===
# Problem: Compare Xavier, He, and random(0.01) initialization.

def exercise_4():
    """Compare weight initialization strategies through a 10-layer network."""
    np.random.seed(42)
    n_layers = 10
    hidden_dim = 256
    batch_size = 64

    def relu(x):
        return np.maximum(0, x)

    def forward_pass(weights, x):
        """Forward pass through layers, return activation variances."""
        variances = [np.var(x)]
        for W in weights:
            x = x @ W
            x = relu(x)
            variances.append(np.var(x))
        return variances

    # Input
    x = np.random.randn(batch_size, hidden_dim)

    # Initialize with different strategies
    strategies = {}

    # Xavier initialization: std = sqrt(2 / (fan_in + fan_out))
    xavier_weights = [np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + hidden_dim))
                      for _ in range(n_layers)]
    strategies['Xavier'] = xavier_weights

    # He initialization: std = sqrt(2 / fan_in) -- designed for ReLU
    he_weights = [np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
                  for _ in range(n_layers)]
    strategies['He'] = he_weights

    # Small random: std = 0.01
    small_weights = [np.random.randn(hidden_dim, hidden_dim) * 0.01
                     for _ in range(n_layers)]
    strategies['Random(0.01)'] = small_weights

    print("Activation variance through 10-layer ReLU network")
    print(f"Hidden dim: {hidden_dim}, Batch size: {batch_size}")
    print()

    print(f"{'Layer':>6}", end="")
    for name in strategies:
        print(f"  {name:>14}", end="")
    print()
    print("-" * 52)

    all_variances = {}
    for name, weights in strategies.items():
        all_variances[name] = forward_pass(weights, x.copy())

    for layer in range(n_layers + 1):
        print(f"{layer:>6}", end="")
        for name in strategies:
            var = all_variances[name][layer]
            if var < 1e-10:
                print(f"  {'~0':>14}", end="")
            else:
                print(f"  {var:>14.6f}", end="")
        print()

    print()
    print("Analysis:")
    print("  He initialization maintains variance well with ReLU (designed for it)")
    print("  Xavier works but variance may shrink slightly with ReLU")
    print("  Random(0.01) causes variance to collapse (vanishing activations)")


# === Exercise 5: Batch Normalization Implementation ===
# Problem: Implement 1D batch normalization from scratch.

def exercise_5():
    """Batch normalization forward and backward pass."""
    np.random.seed(42)
    eps = 1e-5
    momentum = 0.1

    class BatchNorm1D:
        def __init__(self, num_features):
            self.gamma = np.ones(num_features)
            self.beta = np.zeros(num_features)
            self.running_mean = np.zeros(num_features)
            self.running_var = np.ones(num_features)
            self.eps = eps

        def forward(self, x, training=True):
            """Forward pass. x shape: (batch_size, num_features)"""
            if training:
                self.batch_mean = np.mean(x, axis=0)
                self.batch_var = np.var(x, axis=0)

                # Update running statistics
                self.running_mean = (1 - momentum) * self.running_mean + momentum * self.batch_mean
                self.running_var = (1 - momentum) * self.running_var + momentum * self.batch_var

                self.x_centered = x - self.batch_mean
                self.std = np.sqrt(self.batch_var + self.eps)
                self.x_norm = self.x_centered / self.std
            else:
                x_centered = x - self.running_mean
                std = np.sqrt(self.running_var + self.eps)
                self.x_norm = x_centered / std

            self.out = self.gamma * self.x_norm + self.beta
            return self.out

        def backward(self, dout):
            """Backward pass."""
            N = dout.shape[0]

            dgamma = np.sum(dout * self.x_norm, axis=0)
            dbeta = np.sum(dout, axis=0)

            dx_norm = dout * self.gamma
            dx = (1.0 / (N * self.std)) * (
                N * dx_norm
                - np.sum(dx_norm, axis=0)
                - self.x_norm * np.sum(dx_norm * self.x_norm, axis=0)
            )

            return dx, dgamma, dbeta

    # Test
    batch_size = 32
    features = 10
    x = np.random.randn(batch_size, features) * 3 + 5  # non-normalized input

    bn = BatchNorm1D(features)

    # (a) Forward pass (training mode)
    out_train = bn.forward(x, training=True)
    print("(a) Forward pass (training):")
    print(f"    Input  mean: {np.round(np.mean(x, axis=0)[:3], 4)}")
    print(f"    Input  std:  {np.round(np.std(x, axis=0)[:3], 4)}")
    print(f"    Output mean: {np.round(np.mean(out_train, axis=0)[:3], 6)}")
    print(f"    Output std:  {np.round(np.std(out_train, axis=0)[:3], 4)}")

    # (b) Backward pass verification
    dout = np.random.randn(batch_size, features)
    dx, dgamma, dbeta = bn.backward(dout)

    # Numerical gradient check
    eps_num = 1e-5
    dx_numerical = np.zeros_like(x)
    for i in range(min(3, batch_size)):
        for j in range(min(3, features)):
            x_plus = x.copy()
            x_plus[i, j] += eps_num
            x_minus = x.copy()
            x_minus[i, j] -= eps_num

            out_plus = bn.forward(x_plus, training=True)
            loss_plus = np.sum(out_plus * dout)
            out_minus = bn.forward(x_minus, training=True)
            loss_minus = np.sum(out_minus * dout)

            dx_numerical[i, j] = (loss_plus - loss_minus) / (2 * eps_num)

    # Re-run forward to restore state
    bn.forward(x, training=True)
    dx, dgamma, dbeta = bn.backward(dout)

    print(f"\n(b) Gradient check (first 3x3 elements):")
    print(f"    Analytical dx:\n{np.round(dx[:3, :3], 6)}")
    print(f"    Numerical dx:\n{np.round(dx_numerical[:3, :3], 6)}")
    max_diff = np.max(np.abs(dx[:3, :3] - dx_numerical[:3, :3]))
    print(f"    Max difference: {max_diff:.2e}")

    # (c) Training vs inference mode
    # Run several forward passes to build up running stats
    for _ in range(100):
        x_batch = np.random.randn(batch_size, features) * 3 + 5
        bn.forward(x_batch, training=True)

    x_test = np.random.randn(batch_size, features) * 3 + 5
    out_train = bn.forward(x_test, training=True)
    out_eval = bn.forward(x_test, training=False)

    print(f"\n(c) Training vs inference mode:")
    print(f"    Training output mean: {np.round(np.mean(out_train, axis=0)[:3], 6)}")
    print(f"    Inference output mean: {np.round(np.mean(out_eval, axis=0)[:3], 6)}")
    print(f"    Difference (due to running vs batch stats): "
          f"{np.max(np.abs(np.mean(out_train - out_eval, axis=0))):.4f}")


if __name__ == "__main__":
    print("=== Exercise 1: Tensor Manipulation ===")
    exercise_1()
    print("\n=== Exercise 2: einsum Mastery ===")
    exercise_2()
    print("\n=== Exercise 3: Numerical Stability ===")
    exercise_3()
    print("\n=== Exercise 4: Initialization Experiments ===")
    exercise_4()
    print("\n=== Exercise 5: Batch Normalization ===")
    exercise_5()
    print("\nAll exercises completed!")
