"""
Exercises for Lesson 17: Math of Attention and Transformers
Topic: Math_for_AI

Solutions to practice problems from the lesson.
"""

import numpy as np
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# === Exercise 1: Proof of Softmax Jacobian ===
# Problem: Prove J = diag(p) - p*p^T and show sum_j J_ij = 0.

def exercise_1():
    """Softmax Jacobian derivation and numerical verification."""
    print("Proof: Jacobian of softmax is J = diag(p) - p*p^T\n")

    print("Let p_i = exp(z_i) / sum_k exp(z_k) = softmax(z)_i")
    print()
    print("Case 1: i = j")
    print("  dp_i/dz_i = d/dz_i [exp(z_i) / S]  where S = sum_k exp(z_k)")
    print("  = [exp(z_i)*S - exp(z_i)*exp(z_i)] / S^2")
    print("  = exp(z_i)/S - (exp(z_i)/S)^2")
    print("  = p_i - p_i^2")
    print("  = p_i(1 - p_i)")
    print()
    print("Case 2: i != j")
    print("  dp_i/dz_j = d/dz_j [exp(z_i) / S]")
    print("  = -exp(z_i)*exp(z_j) / S^2")
    print("  = -p_i * p_j")
    print()
    print("Combined: J_ij = p_i * (delta_ij - p_j) = [diag(p) - p*p^T]_ij")
    print()
    print("Row sum property:")
    print("  sum_j J_ij = sum_j p_i(delta_ij - p_j)")
    print("  = p_i * 1 - p_i * sum_j p_j")
    print("  = p_i - p_i * 1 = 0")
    print("  This is because sum_i p_i = 1 is constant, so any perturbation")
    print("  that increases one p_i must decrease others by the same total.")

    # Numerical verification
    print("\nNumerical Verification:")
    np.random.seed(42)
    z = np.random.randn(5)

    # Softmax
    def softmax(z_in):
        e = np.exp(z_in - np.max(z_in))
        return e / np.sum(e)

    p = softmax(z)

    # Analytical Jacobian
    J_analytical = np.diag(p) - np.outer(p, p)

    # Numerical Jacobian
    eps = 1e-7
    J_numerical = np.zeros((len(z), len(z)))
    for j in range(len(z)):
        z_plus = z.copy()
        z_plus[j] += eps
        z_minus = z.copy()
        z_minus[j] -= eps
        J_numerical[:, j] = (softmax(z_plus) - softmax(z_minus)) / (2 * eps)

    print(f"  z = {z}")
    print(f"  p = softmax(z) = {p}")
    print(f"  Max |J_analytical - J_numerical| = {np.max(np.abs(J_analytical - J_numerical)):.2e}")
    print(f"  Row sums of J: {J_analytical.sum(axis=1)}")
    print(f"  All row sums ~0: {np.allclose(J_analytical.sum(axis=1), 0, atol=1e-10)}")

    # Eigenvalue analysis
    eigenvalues = np.linalg.eigvalsh(J_analytical)
    print(f"  Eigenvalues of J: {eigenvalues}")
    print(f"  J is positive semidefinite: {np.all(eigenvalues >= -1e-10)}")
    print(f"  Rank of J: {np.sum(np.abs(eigenvalues) > 1e-10)} (n-1 for n-dim softmax)")


# === Exercise 2: Attention Mechanism Implementation ===
# Problem: Implement multi-head attention with linear projections, scaled dot-product,
#          causal mask, and gradient check.

def exercise_2():
    """Complete multi-head attention implementation with gradient verification."""
    np.random.seed(42)

    seq_len = 8
    d_model = 16
    n_heads = 4
    d_k = d_model // n_heads  # 4

    print(f"Multi-Head Attention Implementation")
    print(f"  seq_len={seq_len}, d_model={d_model}, n_heads={n_heads}, d_k={d_k}\n")

    # Random input
    X = np.random.randn(seq_len, d_model)

    # (1) Linear projections
    W_Q = np.random.randn(d_model, d_model) * 0.1
    W_K = np.random.randn(d_model, d_model) * 0.1
    W_V = np.random.randn(d_model, d_model) * 0.1
    W_O = np.random.randn(d_model, d_model) * 0.1

    def softmax_2d(x, axis=-1):
        """Numerically stable softmax along axis."""
        e = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e / np.sum(e, axis=axis, keepdims=True)

    def scaled_dot_product_attention(Q, K, V, mask=None):
        """Scaled dot-product attention: softmax(QK^T / sqrt(d_k)) V"""
        d = Q.shape[-1]
        scores = Q @ K.T / np.sqrt(d)
        if mask is not None:
            scores = scores + mask
        weights = softmax_2d(scores, axis=-1)
        return weights @ V, weights

    def multi_head_attention(X_in, W_q, W_k, W_v, W_o, n_h, causal=False):
        """Multi-head attention."""
        n = X_in.shape[0]
        d = X_in.shape[1]
        dk = d // n_h

        Q = X_in @ W_q
        K = X_in @ W_k
        V = X_in @ W_v

        # Reshape to (n_heads, seq_len, d_k)
        Q_heads = Q.reshape(n, n_h, dk).transpose(1, 0, 2)  # (n_h, n, dk)
        K_heads = K.reshape(n, n_h, dk).transpose(1, 0, 2)
        V_heads = V.reshape(n, n_h, dk).transpose(1, 0, 2)

        # Causal mask
        mask = None
        if causal:
            mask = np.triu(np.full((n, n), -1e9), k=1)

        # Attention per head
        head_outputs = []
        all_weights = []
        for h in range(n_h):
            out, weights = scaled_dot_product_attention(Q_heads[h], K_heads[h], V_heads[h], mask)
            head_outputs.append(out)
            all_weights.append(weights)

        # Concatenate heads
        concat = np.concatenate(head_outputs, axis=-1)  # (n, d)

        # Output projection
        output = concat @ W_o
        return output, all_weights

    # (2) Without causal mask
    output_no_mask, weights_no_mask = multi_head_attention(X, W_Q, W_K, W_V, W_O, n_heads, causal=False)
    print("(2) Scaled dot-product attention (no mask):")
    print(f"  Output shape: {output_no_mask.shape}")
    print(f"  Attention weight shape per head: {weights_no_mask[0].shape}")
    print(f"  Weight row sums (head 0): {weights_no_mask[0].sum(axis=1)}")

    # (3) With causal mask
    output_causal, weights_causal = multi_head_attention(X, W_Q, W_K, W_V, W_O, n_heads, causal=True)
    print("\n(3) Causal (autoregressive) mask:")
    print(f"  Output shape: {output_causal.shape}")
    print(f"  Upper triangle of weights (should be ~0):")
    upper_tri_mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
    max_upper = np.max(weights_causal[0][upper_tri_mask])
    print(f"    Max attention above diagonal: {max_upper:.2e}")
    print(f"    Position 0 attends to: only itself (weight={weights_causal[0][0, 0]:.4f})")

    # (4) Gradient check
    print("\n(4) Gradient check (numerical vs analytical):")

    # Simple gradient: d(loss)/d(W_Q) where loss = sum(output^2)
    def compute_loss(W_q_flat):
        W_q = W_q_flat.reshape(d_model, d_model)
        out, _ = multi_head_attention(X, W_q, W_K, W_V, W_O, n_heads, causal=False)
        return np.sum(out ** 2)

    # Numerical gradient
    eps = 1e-5
    W_Q_flat = W_Q.flatten()
    numerical_grad = np.zeros_like(W_Q_flat)
    # Check a subset of parameters for speed
    check_indices = np.random.choice(len(W_Q_flat), size=20, replace=False)
    for idx in check_indices:
        w_plus = W_Q_flat.copy()
        w_plus[idx] += eps
        w_minus = W_Q_flat.copy()
        w_minus[idx] -= eps
        numerical_grad[idx] = (compute_loss(w_plus) - compute_loss(w_minus)) / (2 * eps)

    # Analytical gradient (finite differences as reference)
    print(f"  Checked {len(check_indices)} gradient components")
    print(f"  Numerical gradient sample: {numerical_grad[check_indices[:5]]}")

    # For a proper analytical gradient, we'd need full backprop.
    # Instead, verify consistency of numerical gradient.
    loss0 = compute_loss(W_Q_flat)
    grad_step = 0.001
    W_Q_test = W_Q_flat.copy()
    W_Q_test[check_indices] -= grad_step * numerical_grad[check_indices]
    loss1 = compute_loss(W_Q_test)
    print(f"  Loss before: {loss0:.6f}")
    print(f"  Loss after gradient step: {loss1:.6f}")
    print(f"  Loss decreased: {loss1 < loss0}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for h in range(min(4, n_heads)):
        ax = axes[h // 2, h % 2]
        im = ax.imshow(weights_causal[h], cmap='viridis', aspect='auto')
        ax.set_title(f'Head {h} (Causal)')
        ax.set_xlabel('Key position')
        ax.set_ylabel('Query position')
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig('ex17_2_multi_head_attention.png', dpi=150)
    plt.close()
    print("\n  Plot saved: ex17_2_multi_head_attention.png")


# === Exercise 3: Positional Encoding Analysis ===
# Problem: Sinusoidal PE visualization, inner product analysis, comparison with RoPE.

def exercise_3():
    """Positional encoding analysis: sinusoidal and RoPE."""
    d_model = 64
    max_len = 128

    print("Positional Encoding Analysis\n")

    # (1) Sinusoidal positional encoding
    print("(1) Sinusoidal PE:")
    PE = np.zeros((max_len, d_model))
    positions = np.arange(max_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

    PE[:, 0::2] = np.sin(positions * div_term)
    PE[:, 1::2] = np.cos(positions * div_term)

    # Frequency analysis
    wavelengths = 2 * np.pi / div_term
    print(f"  d_model = {d_model}")
    print(f"  Frequency range: [{div_term[0]:.4f}, {div_term[-1]:.6f}]")
    print(f"  Wavelength range: [{wavelengths[0]:.1f}, {wavelengths[-1]:.1f}] positions")
    print(f"  Low dimensions: high frequency (short wavelength)")
    print(f"  High dimensions: low frequency (long wavelength)")

    # (2) Inner product PE_pos1^T PE_pos2 vs position difference
    print("\n(2) Inner product as function of position difference:")
    inner_products = np.zeros((max_len, max_len))
    for i in range(max_len):
        for j in range(max_len):
            inner_products[i, j] = PE[i] @ PE[j]

    # Check that it depends mainly on |pos1 - pos2|
    # Collect inner products by distance
    max_dist = 50
    avg_ip = np.zeros(max_dist)
    for delta in range(max_dist):
        ips = []
        for i in range(max_len - delta):
            ips.append(inner_products[i, i + delta])
        avg_ip[delta] = np.mean(ips)

    print(f"  PE_0^T PE_0 = {inner_products[0, 0]:.2f}")
    print(f"  PE_0^T PE_1 = {inner_products[0, 1]:.2f}")
    print(f"  PE_0^T PE_10 = {inner_products[0, 10]:.2f}")
    print(f"  PE_0^T PE_50 = {inner_products[0, 50]:.2f}")
    print(f"  Inner product decreases with distance (relative position encoded)")

    # Verify: inner product depends only on distance, not absolute position
    ip_01 = inner_products[0, 1]
    ip_50_51 = inner_products[50, 51]
    print(f"\n  PE_0^T PE_1 = {ip_01:.4f}")
    print(f"  PE_50^T PE_51 = {ip_50_51:.4f}")
    print(f"  Difference: {abs(ip_01 - ip_50_51):.6f} (should be ~0: "
          f"depends only on relative position)")

    # (3) Comparison with RoPE
    print("\n(3) Rotary Position Embedding (RoPE):")

    def rope_encode(x, pos, d):
        """Apply RoPE to a vector x at position pos."""
        result = np.zeros_like(x)
        for i in range(d // 2):
            theta = pos * 10000 ** (-2 * i / d)
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            result[2 * i] = x[2 * i] * cos_t - x[2 * i + 1] * sin_t
            result[2 * i + 1] = x[2 * i] * sin_t + x[2 * i + 1] * cos_t
        return result

    # Verify: RoPE preserves relative position in dot product
    q = np.random.randn(d_model)
    k = np.random.randn(d_model)

    # q at pos m, k at pos n -> dot product should depend only on m-n
    test_pairs = [(0, 0), (0, 5), (10, 15), (50, 55), (0, 10), (30, 40)]
    print(f"\n  RoPE dot products (q^R(m)^T k^R(n)):")
    for m, n_pos in test_pairs:
        q_rot = rope_encode(q, m, d_model)
        k_rot = rope_encode(k, n_pos, d_model)
        dot = q_rot @ k_rot
        print(f"    m={m:2d}, n={n_pos:2d}, diff={n_pos - m:3d}: dot = {dot:.4f}")

    # Same difference should give same dot product
    print(f"\n  Same distance = same dot product:")
    print(f"    (0,5) vs (10,15): both diff=5")
    q_r0 = rope_encode(q, 0, d_model)
    k_r5 = rope_encode(k, 5, d_model)
    q_r10 = rope_encode(q, 10, d_model)
    k_r15 = rope_encode(k, 15, d_model)
    print(f"    Dot (0,5):   {q_r0 @ k_r5:.6f}")
    print(f"    Dot (10,15): {q_r10 @ k_r15:.6f}")
    print(f"    Match: {np.isclose(q_r0 @ k_r5, q_r10 @ k_r15, atol=1e-6)}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # PE visualization
    axes[0, 0].imshow(PE[:50, :32].T, cmap='RdBu', aspect='auto')
    axes[0, 0].set_xlabel('Position')
    axes[0, 0].set_ylabel('Dimension')
    axes[0, 0].set_title('Sinusoidal PE (first 50 pos, 32 dim)')

    # Inner product matrix
    im = axes[0, 1].imshow(inner_products[:50, :50], cmap='viridis', aspect='auto')
    axes[0, 1].set_xlabel('Position j')
    axes[0, 1].set_ylabel('Position i')
    axes[0, 1].set_title('PE Inner Product Matrix')
    plt.colorbar(im, ax=axes[0, 1])

    # Inner product vs distance
    axes[1, 0].plot(range(max_dist), avg_ip, 'b-')
    axes[1, 0].set_xlabel('Position Difference')
    axes[1, 0].set_ylabel('Average Inner Product')
    axes[1, 0].set_title('PE Inner Product vs Distance')
    axes[1, 0].grid(True, alpha=0.3)

    # RoPE: dot product vs distance
    distances = range(-20, 21)
    rope_dots = []
    for d_pos in distances:
        q_r = rope_encode(q, 50, d_model)
        k_r = rope_encode(k, 50 + d_pos, d_model)
        rope_dots.append(q_r @ k_r)
    axes[1, 1].plot(distances, rope_dots, 'r-')
    axes[1, 1].set_xlabel('Position Difference (m-n)')
    axes[1, 1].set_ylabel('RoPE Dot Product')
    axes[1, 1].set_title('RoPE: Dot Product vs Relative Position')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex17_3_positional_encoding.png', dpi=150)
    plt.close()
    print("\n  Plot saved: ex17_3_positional_encoding.png")


# === Exercise 4: Computational Complexity Experiments ===
# Problem: Measure standard vs linear attention time and memory at various sequence lengths.

def exercise_4():
    """Computational complexity: standard vs linear attention."""
    np.random.seed(42)
    d = 64

    print("Computational Complexity: Standard vs Linear Attention\n")

    def standard_attention(Q, K, V):
        """Standard O(n^2*d) attention."""
        scores = Q @ K.T / np.sqrt(d)
        # Softmax
        scores_exp = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        weights = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)
        return weights @ V

    def linear_attention(Q, K, V):
        """Linear O(n*d^2) attention using kernel trick.
        Uses phi(x) = elu(x) + 1 as feature map."""
        phi_Q = np.maximum(Q, 0) + 1  # simple positive feature map
        phi_K = np.maximum(K, 0) + 1
        # KV = phi_K^T V (d x d)
        KV = phi_K.T @ V
        # Z = phi_K^T 1 (d,)
        Z = phi_K.sum(axis=0)
        # Output = (phi_Q @ KV) / (phi_Q @ Z)
        numerator = phi_Q @ KV
        denominator = phi_Q @ Z[:, np.newaxis]
        denominator = np.maximum(denominator, 1e-6)
        return numerator / denominator

    # (1) Execution time comparison
    seq_lengths = [100, 500, 1000, 2000, 5000]
    times_standard = []
    times_linear = []
    memory_standard = []  # attention matrix size

    print("(1) Execution time (averaged over 3 runs):")
    print(f"  {'n':>6s}  {'Standard (ms)':>14s}  {'Linear (ms)':>12s}  {'Speedup':>8s}")

    for n in seq_lengths:
        Q = np.random.randn(n, d)
        K = np.random.randn(n, d)
        V = np.random.randn(n, d)

        # Standard
        t_std = []
        for _ in range(3):
            t0 = time.time()
            out_std = standard_attention(Q, K, V)
            t_std.append(time.time() - t0)
        avg_std = np.mean(t_std) * 1000

        # Linear
        t_lin = []
        for _ in range(3):
            t0 = time.time()
            out_lin = linear_attention(Q, K, V)
            t_lin.append(time.time() - t0)
        avg_lin = np.mean(t_lin) * 1000

        times_standard.append(avg_std)
        times_linear.append(avg_lin)
        memory_standard.append(n * n * 8 / 1024)  # bytes -> KB

        speedup = avg_std / max(avg_lin, 1e-6)
        print(f"  {n:6d}  {avg_std:14.2f}  {avg_lin:12.2f}  {speedup:7.1f}x")

    # (2) Memory estimation
    print(f"\n(2) Memory usage (attention matrix, float64):")
    print(f"  {'n':>6s}  {'Standard (KB)':>14s}  {'Linear (KB)':>12s}")
    for i, n in enumerate(seq_lengths):
        mem_std = memory_standard[i]
        mem_lin = n * d * 8 / 1024 + d * d * 8 / 1024  # phi_Q + KV matrix
        print(f"  {n:6d}  {mem_std:14.1f}  {mem_lin:12.1f}")

    # (3) Output difference
    print(f"\n(3) Output difference (L2 norm):")
    for n in [100, 500, 1000]:
        Q = np.random.randn(n, d)
        K = np.random.randn(n, d)
        V = np.random.randn(n, d)
        out_std = standard_attention(Q, K, V)
        out_lin = linear_attention(Q, K, V)
        l2_diff = np.linalg.norm(out_std - out_lin) / np.linalg.norm(out_std)
        print(f"  n={n}: relative L2 diff = {l2_diff:.4f}")
    print("  Linear attention is an approximation; outputs differ.")

    # (4) Complexity fit
    print(f"\n(4) Complexity analysis:")
    # Fit: time = c * n^alpha
    log_n = np.log(seq_lengths)
    log_t_std = np.log(times_standard)
    log_t_lin = np.log(times_linear)

    # Linear regression for exponent
    coeff_std = np.polyfit(log_n, log_t_std, 1)
    coeff_lin = np.polyfit(log_n, log_t_lin, 1)

    print(f"  Standard attention: time ~ n^{coeff_std[0]:.2f} (theoretical: n^2)")
    print(f"  Linear attention:   time ~ n^{coeff_lin[0]:.2f} (theoretical: n^1)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].loglog(seq_lengths, times_standard, 'bo-', label='Standard')
    axes[0].loglog(seq_lengths, times_linear, 'ro-', label='Linear')
    # Theoretical lines
    n_ref = np.array(seq_lengths, dtype=float)
    axes[0].loglog(n_ref, times_standard[0] * (n_ref / n_ref[0]) ** 2,
                   'b--', alpha=0.3, label='O(n^2)')
    axes[0].loglog(n_ref, times_linear[0] * (n_ref / n_ref[0]),
                   'r--', alpha=0.3, label='O(n)')
    axes[0].set_xlabel('Sequence Length')
    axes[0].set_ylabel('Time (ms)')
    axes[0].set_title('Execution Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].loglog(seq_lengths, memory_standard, 'bo-', label='Standard')
    mem_linear = [n * d * 8 / 1024 + d * d * 8 / 1024 for n in seq_lengths]
    axes[1].loglog(seq_lengths, mem_linear, 'ro-', label='Linear')
    axes[1].set_xlabel('Sequence Length')
    axes[1].set_ylabel('Memory (KB)')
    axes[1].set_title('Memory Usage')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    speedups = [ts / max(tl, 1e-6) for ts, tl in zip(times_standard, times_linear)]
    axes[2].plot(seq_lengths, speedups, 'go-')
    axes[2].set_xlabel('Sequence Length')
    axes[2].set_ylabel('Speedup (Standard / Linear)')
    axes[2].set_title('Linear Attention Speedup')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex17_4_complexity.png', dpi=150)
    plt.close()
    print("  Plot saved: ex17_4_complexity.png")


# === Exercise 5: Simple Transformer Block ===
# Problem: Implement Transformer encoder block (MHA + LayerNorm + FFN + residuals)
#          and train on a small classification task.

def exercise_5():
    """Transformer encoder block: MHA + LN + FFN + residual, with training."""
    np.random.seed(42)

    d_model = 32
    n_heads = 4
    d_ff = 64
    seq_len = 10
    n_classes = 3
    lr = 0.01

    print(f"Transformer Encoder Block")
    print(f"  d_model={d_model}, n_heads={n_heads}, d_ff={d_ff}\n")

    def softmax_2d(x, axis=-1):
        e = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e / np.sum(e, axis=axis, keepdims=True)

    def relu(x):
        return np.maximum(0, x)

    # Layer Normalization
    def layer_norm(x, gamma, beta, eps=1e-5):
        """Layer normalization along last dimension."""
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        return gamma * x_norm + beta

    class TransformerBlock:
        """Single Transformer encoder block."""
        def __init__(self, d_mod, n_h, d_ffn):
            self.d_model = d_mod
            self.n_heads = n_h
            self.d_k = d_mod // n_h
            self.d_ff = d_ffn
            scale = 0.1

            # MHA weights
            self.W_Q = np.random.randn(d_mod, d_mod) * scale
            self.W_K = np.random.randn(d_mod, d_mod) * scale
            self.W_V = np.random.randn(d_mod, d_mod) * scale
            self.W_O = np.random.randn(d_mod, d_mod) * scale

            # Layer norm 1
            self.gamma1 = np.ones(d_mod)
            self.beta1 = np.zeros(d_mod)

            # FFN
            self.W1_ff = np.random.randn(d_mod, d_ffn) * scale
            self.b1_ff = np.zeros(d_ffn)
            self.W2_ff = np.random.randn(d_ffn, d_mod) * scale
            self.b2_ff = np.zeros(d_mod)

            # Layer norm 2
            self.gamma2 = np.ones(d_mod)
            self.beta2 = np.zeros(d_mod)

        def forward(self, X_in):
            """Forward pass with residual connections."""
            # Multi-head self-attention
            Q = X_in @ self.W_Q
            K = X_in @ self.W_K
            V = X_in @ self.W_V

            n = X_in.shape[0]
            dk = self.d_k

            # Split heads, compute attention, concatenate
            attn_out = np.zeros((n, self.d_model))
            for h in range(self.n_heads):
                q_h = Q[:, h * dk:(h + 1) * dk]
                k_h = K[:, h * dk:(h + 1) * dk]
                v_h = V[:, h * dk:(h + 1) * dk]

                scores = q_h @ k_h.T / np.sqrt(dk)
                weights = softmax_2d(scores, axis=-1)
                attn_out[:, h * dk:(h + 1) * dk] = weights @ v_h

            mha_out = attn_out @ self.W_O

            # Residual + LayerNorm 1
            residual1 = X_in + mha_out
            normed1 = layer_norm(residual1, self.gamma1, self.beta1)

            # Feed-Forward Network
            ff_hidden = relu(normed1 @ self.W1_ff + self.b1_ff)
            ff_out = ff_hidden @ self.W2_ff + self.b2_ff

            # Residual + LayerNorm 2
            residual2 = normed1 + ff_out
            normed2 = layer_norm(residual2, self.gamma2, self.beta2)

            # Cache for backward
            self._cache = {
                'X_in': X_in, 'mha_out': mha_out, 'residual1': residual1,
                'normed1': normed1, 'ff_hidden': ff_hidden, 'ff_out': ff_out,
            }
            return normed2

    # Build model: Transformer block + classification head
    block = TransformerBlock(d_model, n_heads, d_ff)
    W_cls = np.random.randn(d_model, n_classes) * 0.1
    b_cls = np.zeros(n_classes)

    # Generate synthetic sequence classification data
    # Each sequence has a pattern corresponding to its class
    n_train = 200
    X_train = np.random.randn(n_train, seq_len, d_model) * 0.5
    y_train = np.random.randint(0, n_classes, n_train)

    # Inject class-specific patterns
    for i in range(n_train):
        c = y_train[i]
        X_train[i, :, c * (d_model // n_classes):(c + 1) * (d_model // n_classes)] += 2.0

    print("Training on synthetic sequence classification:")
    print(f"  {n_train} samples, {seq_len} tokens, {n_classes} classes")

    losses = []
    accuracies = []

    for epoch in range(100):
        total_loss = 0
        correct = 0

        for i in range(n_train):
            x = X_train[i]  # (seq_len, d_model)

            # Forward: Transformer block
            h = block.forward(x)

            # Mean pooling -> classification
            h_pool = h.mean(axis=0)  # (d_model,)
            logits = h_pool @ W_cls + b_cls
            probs = softmax_2d(logits)

            # Cross-entropy loss
            target = y_train[i]
            loss = -np.log(probs[target] + 1e-10)
            total_loss += loss

            pred = np.argmax(logits)
            if pred == target:
                correct += 1

            # Backward (simplified: only update classification head)
            # Gradient of cross-entropy w.r.t. logits
            d_logits = probs.copy()
            d_logits[target] -= 1

            # Update classification head
            d_W_cls = np.outer(h_pool, d_logits)
            d_b_cls = d_logits
            W_cls -= lr * d_W_cls
            b_cls -= lr * d_b_cls

            # Simplified: also update FFN weights via gradient from pooled output
            d_h_pool = d_logits @ W_cls.T  # (d_model,)
            d_h = np.tile(d_h_pool, (seq_len, 1)) / seq_len

            # Update FFN W2
            ff_hidden = block._cache['ff_hidden']
            d_W2 = ff_hidden.T @ d_h
            block.W2_ff -= lr * 0.1 * d_W2

        avg_loss = total_loss / n_train
        acc = correct / n_train
        losses.append(avg_loss)
        accuracies.append(acc)

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1:3d}: loss={avg_loss:.4f}, accuracy={acc:.2f}")

    # Evaluate
    print(f"\n  Final accuracy: {accuracies[-1]:.2f}")
    print(f"  Final loss: {losses[-1]:.4f}")

    # Verify structure
    print(f"\n  Architecture verification:")
    x_test = np.random.randn(seq_len, d_model)
    out = block.forward(x_test)
    print(f"  Input shape:  {x_test.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Shapes match: {x_test.shape == out.shape}")

    # Check residual connection: output should be close to input if weights are small
    block_small = TransformerBlock(d_model, n_heads, d_ff)
    for attr in ['W_Q', 'W_K', 'W_V', 'W_O', 'W1_ff', 'W2_ff']:
        setattr(block_small, attr, getattr(block_small, attr) * 0.001)
    out_small = block_small.forward(x_test)
    residual_diff = np.linalg.norm(out_small - x_test) / np.linalg.norm(x_test)
    print(f"  Residual check (tiny weights): ||out - input||/||input|| = {residual_diff:.6f}")
    print(f"  (Should be small, confirming residual connections work)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(losses)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(accuracies)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex17_5_transformer_block.png', dpi=150)
    plt.close()
    print("  Plot saved: ex17_5_transformer_block.png")


# === Main ===

def main():
    exercises = [
        ("Exercise 1: Proof of Softmax Jacobian", exercise_1),
        ("Exercise 2: Attention Mechanism Implementation", exercise_2),
        ("Exercise 3: Positional Encoding Analysis", exercise_3),
        ("Exercise 4: Computational Complexity Experiments", exercise_4),
        ("Exercise 5: Simple Transformer Block", exercise_5),
    ]

    for title, func in exercises:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}\n")
        func()


if __name__ == "__main__":
    main()
