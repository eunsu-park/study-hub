"""
Exercises for Lesson 13: Model Quantization
Topic: LLM_and_NLP

Solutions to practice problems from the lesson.
"""

import numpy as np


# === Exercise 1: Quantization Memory Calculation ===
# Problem: Calculate memory requirements for a transformer model across
# different precision formats (FP32, FP16/BF16, INT8, INT4).

def exercise_1():
    """Quantization memory calculation for a transformer model."""
    print("=" * 60)
    print("Exercise 1: Quantization Memory Calculation")
    print("=" * 60)

    def calculate_model_memory(
        num_layers: int,
        hidden_size: int,
        ffn_size: int,
        bytes_per_param: float,
    ) -> dict:
        """Calculate model memory requirements."""
        # Attention: Q, K, V, O projections each of shape (hidden, hidden)
        attention_params = 4 * hidden_size * hidden_size
        attention_mb = attention_params * bytes_per_param / (1024 ** 2)

        # FFN: up-projection (hidden->ffn), down-projection (ffn->hidden)
        # Problem specifies: 2 matrices of shape (4096, 16384) + 1 of (16384, 4096)
        ffn_params = 2 * (hidden_size * ffn_size) + (ffn_size * hidden_size)
        ffn_mb = ffn_params * bytes_per_param / (1024 ** 2)

        total_params = num_layers * (attention_params + ffn_params)
        total_gb = total_params * bytes_per_param / (1024 ** 3)

        return {
            "attention_params": attention_params,
            "attention_mb": attention_mb,
            "ffn_params": ffn_params,
            "ffn_mb": ffn_mb,
            "total_params": total_params,
            "total_gb": total_gb,
        }

    # Model: 32 layers, hidden=4096, ffn=16384
    NUM_LAYERS = 32
    HIDDEN = 4096
    FFN = 16384

    precisions = {
        "FP32":       4.0,
        "FP16/BF16":  2.0,
        "INT8":       1.0,
        "INT4":       0.5,
    }

    print(f"\nModel: {NUM_LAYERS} layers, hidden={HIDDEN}, ffn={FFN}")
    print(f"  Attention per layer: 4 x ({HIDDEN} x {HIDDEN}) = {4 * HIDDEN * HIDDEN:,} params")
    print(f"  FFN per layer: 3 x ({HIDDEN} x {FFN}) = {3 * HIDDEN * FFN:,} params")
    print()

    print(f"{'Precision':<12} {'Bytes/param':<13} {'Attn (MB)':<12} {'FFN (MB)':<11} {'Total (GB)'}")
    print("-" * 60)

    for name, bpp in precisions.items():
        r = calculate_model_memory(NUM_LAYERS, HIDDEN, FFN, bpp)
        print(
            f"{name:<12} {bpp:<13.1f} {r['attention_mb']:<12.1f} "
            f"{r['ffn_mb']:<11.1f} {r['total_gb']:.2f}"
        )

    print("\n  Key insight: INT4 reduces memory by 8x vs FP32.")
    print("  For a real LLaMA-2-7B (~7B params), INT4 = ~3.5GB vs 28GB FP32.")
    print("  That's the difference between a consumer GPU and a datacenter GPU.")


# === Exercise 2: Symmetric vs Asymmetric Quantization ===
# Problem: Apply both symmetric and asymmetric INT8 quantization to
# a non-centered weight tensor, compare quantization errors.

def exercise_2():
    """Symmetric vs asymmetric INT8 quantization comparison."""
    print("\n" + "=" * 60)
    print("Exercise 2: Symmetric vs Asymmetric Quantization")
    print("=" * 60)

    weights = np.array(
        [0.01, 0.05, 0.12, 0.23, 0.45, 0.67, 0.89, 1.20, 1.45, 1.80],
        dtype=np.float32,
    )

    # --- Symmetric INT8 quantization ---
    def quantize_symmetric(tensor, bits=8):
        qmin = -(2 ** (bits - 1))       # -128
        qmax = 2 ** (bits - 1) - 1      # 127

        abs_max = np.abs(tensor).max()
        scale = abs_max / qmax           # scale = 1.80 / 127

        quantized = np.round(tensor / scale).clip(qmin, qmax).astype(np.int8)
        return quantized, scale

    def dequantize_sym(q, scale):
        return q.astype(np.float32) * scale

    # --- Asymmetric INT8 quantization ---
    def quantize_asymmetric(tensor, bits=8):
        qmin = 0
        qmax = 2 ** bits - 1            # 255

        min_val = tensor.min()
        max_val = tensor.max()
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = int(round(-min_val / scale))

        quantized = np.round(tensor / scale + zero_point).clip(qmin, qmax).astype(np.uint8)
        return quantized, scale, zero_point

    def dequantize_asym(q, scale, zp):
        return (q.astype(np.float32) - zp) * scale

    # Apply both methods
    q_sym, s_sym = quantize_symmetric(weights)
    rec_sym = dequantize_sym(q_sym, s_sym)
    error_sym = np.abs(weights - rec_sym)

    q_asym, s_asym, zp_asym = quantize_asymmetric(weights)
    rec_asym = dequantize_asym(q_asym, s_asym, zp_asym)
    error_asym = np.abs(weights - rec_asym)

    print(f"\nOriginal weights: {weights}")

    print(f"\nSymmetric quantization:")
    print(f"  Scale: {s_sym:.6f}")
    print(f"  Quantized (INT8): {q_sym}")
    print(f"  Reconstructed:    {np.round(rec_sym, 4)}")
    print(f"  Mean error: {error_sym.mean():.6f}")
    print(f"  Max error:  {error_sym.max():.6f}")

    print(f"\nAsymmetric quantization:")
    print(f"  Scale: {s_asym:.6f}, Zero point: {zp_asym}")
    print(f"  Quantized (UINT8): {q_asym}")
    print(f"  Reconstructed:     {np.round(rec_asym, 4)}")
    print(f"  Mean error: {error_asym.mean():.6f}")
    print(f"  Max error:  {error_asym.max():.6f}")

    granularity_ratio = s_sym / s_asym
    print(f"\nGranularity improvement: {granularity_ratio:.2f}x finer with asymmetric")

    print(f"\n  Why asymmetric wins for non-centered distributions:")
    print(f"    Symmetric scale = {s_sym:.5f} (half the 256 range wasted on negatives)")
    print(f"    Asymmetric scale = {s_asym:.5f} (full 256 range covers 0.01-1.80)")
    print(f"    Smaller scale -> finer granularity -> less quantization error")

    # Verify asymmetric is better for this distribution
    assert error_asym.mean() < error_sym.mean(), "Asymmetric should have lower error"
    print("\n  Assertion passed: asymmetric error < symmetric error for this data.")


# === Exercise 3: NF4 vs INT4 Intuition ===
# Problem: Compare NF4 (non-uniform) vs INT4 (uniform) quantization levels
# for normally distributed weights.

def exercise_3():
    """NF4 vs INT4 quantization for normally distributed weights."""
    print("\n" + "=" * 60)
    print("Exercise 3: NF4 vs INT4 Intuition")
    print("=" * 60)

    # NF4 uses quantiles of N(0,1) for non-uniform levels.
    # We use manual normal quantile approximation to avoid scipy dependency.

    def norm_ppf_approx(p):
        """
        Approximate inverse normal CDF (percent-point function).
        Uses Beasley-Springer-Moro algorithm for reasonable accuracy.
        """
        if p <= 0:
            return -6.0
        if p >= 1:
            return 6.0
        if p == 0.5:
            return 0.0

        # Rational approximation
        if p < 0.5:
            t = np.sqrt(-2.0 * np.log(p))
        else:
            t = np.sqrt(-2.0 * np.log(1.0 - p))

        # Coefficients for rational approximation
        c0 = 2.515517
        c1 = 0.802853
        c2 = 0.010328
        d1 = 1.432788
        d2 = 0.189269
        d3 = 0.001308

        result = t - (c0 + c1 * t + c2 * t**2) / (1 + d1 * t + d2 * t**2 + d3 * t**3)

        if p < 0.5:
            return -result
        return result

    num_levels = 16

    # INT4 uses 16 UNIFORM quantization levels
    int4_levels = np.linspace(-1, 1, num_levels)  # Uniform spacing in [-1, 1]

    # NF4 uses 16 NON-UNIFORM levels based on quantiles of N(0,1)
    # Each level covers an equal probability mass
    prob_centers = np.linspace(1 / (2 * num_levels), 1 - 1 / (2 * num_levels), num_levels)
    nf4_levels_raw = np.array([norm_ppf_approx(p) for p in prob_centers])
    # Scale to [-1, 1] for fair comparison
    nf4_levels = nf4_levels_raw / np.abs(nf4_levels_raw).max()

    # Simulate normally distributed weights
    np.random.seed(42)
    weights = np.random.normal(0, 0.3, size=10000)
    weights = np.clip(weights, -1, 1)

    def quantize_to_levels(w, levels):
        """Map each weight to the nearest quantization level."""
        levels_sorted = np.sort(levels)
        indices = np.abs(w[:, None] - levels_sorted[None, :]).argmin(axis=1)
        return levels_sorted[indices]

    q_int4 = quantize_to_levels(weights, int4_levels)
    q_nf4 = quantize_to_levels(weights, nf4_levels)

    error_int4 = np.abs(weights - q_int4)
    error_nf4 = np.abs(weights - q_nf4)

    print(f"\nINT4 quantization (uniform levels):")
    print(f"  Levels: {np.round(int4_levels, 3)}")
    print(f"  Mean absolute error: {error_int4.mean():.6f}")
    print(f"  Max absolute error:  {error_int4.max():.6f}")

    print(f"\nNF4 quantization (normal-quantile levels):")
    print(f"  Levels: {np.round(nf4_levels, 3)}")
    print(f"  Mean absolute error: {error_nf4.mean():.6f}")
    print(f"  Max absolute error:  {error_nf4.max():.6f}")

    improvement = (error_int4.mean() - error_nf4.mean()) / error_int4.mean() * 100
    print(f"\nNF4 reduces mean error by {improvement:.1f}% vs INT4")

    # Show level density near zero
    print(f"\nLevel density analysis (spacing between adjacent levels):")
    int4_spacings = np.diff(np.sort(int4_levels))
    nf4_spacings = np.diff(np.sort(nf4_levels))

    # Near-zero region: middle 4 spacings
    mid = num_levels // 2
    print(f"  INT4 center spacings: {np.round(int4_spacings[mid-2:mid+2], 4)}")
    print(f"  NF4 center spacings:  {np.round(nf4_spacings[mid-2:mid+2], 4)}")
    print(f"  NF4 tail spacings:    {np.round(nf4_spacings[:2], 4)}, {np.round(nf4_spacings[-2:], 4)}")

    print(f"\n  Key insight: NF4 packs more levels near 0 (where ~68% of")
    print(f"  normally distributed weights cluster), using wider spacing")
    print(f"  in the tails. INT4 wastes resolution on rarely-populated tails.")
    print(f"  This is why NF4 is the recommended quantization type in bitsandbytes.")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
