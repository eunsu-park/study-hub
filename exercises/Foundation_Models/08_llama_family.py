"""
Exercises for Lesson 08: LLaMA Family
Topic: Foundation_Models

Solutions to practice problems from the lesson.
"""

import math


# === Exercise 1: RoPE vs Absolute Position Encoding ===
# Problem: Compare RoPE and absolute position encoding on 4 dimensions.

def exercise_1():
    """Solution: RoPE vs Absolute PE comparison"""
    comparison = {
        "Extrapolation beyond training length": {
            "Absolute PE": (
                "Poor -- positions beyond trained maximum have no learned "
                "embedding, causing degraded performance."
            ),
            "RoPE": (
                "Better -- rotation in complex plane generalizes more "
                "gracefully; can be extended with YaRN or LongRoPE scaling."
            ),
        },
        "How position is encoded": {
            "Absolute PE": (
                "Adds a learned/fixed vector to each token embedding "
                "(position info in embedding space)."
            ),
            "RoPE": (
                "Multiplies Q and K by rotation matrix based on position "
                "(position info in attention dot-product relationship)."
            ),
        },
        "Relative distance representation": {
            "Absolute PE": (
                "Indirect -- relative distance computed from absolute positions, "
                "model must learn this relationship."
            ),
            "RoPE": (
                "Direct -- Q_m . K_n depends only on relative position (m-n), "
                "not absolute positions."
            ),
        },
        "Memory cost": {
            "Absolute PE": "Stores learned matrix of size (max_len x dim).",
            "RoPE": "No additional parameters -- computed on-the-fly.",
        },
    }

    for dim, vals in comparison.items():
        print(f"  {dim}:")
        print(f"    Absolute PE: {vals['Absolute PE']}")
        print(f"    RoPE:        {vals['RoPE']}")
        print()


# === Exercise 2: GQA Memory Savings Calculation ===
# Problem: Calculate KV cache memory for GQA vs MHA.

def exercise_2():
    """Solution: GQA memory savings"""
    # LLaMA 2 70B specs
    num_query_heads = 64
    num_kv_heads_gqa = 8
    num_kv_heads_mha = 64
    head_dim = 128
    fp16_bytes = 2
    batch_size = 32
    seq_len = 4096
    num_layers = 80

    # Part 1: KV cache with GQA per token per layer
    kv_per_token_gqa = 2 * num_kv_heads_gqa * head_dim * fp16_bytes
    print("  Part 1: KV cache with GQA (8 KV heads)")
    print(f"    Per token per layer = 2 * {num_kv_heads_gqa} * {head_dim} * {fp16_bytes} = {kv_per_token_gqa:,} bytes = {kv_per_token_gqa/1024:.0f} KB")
    print()

    # Part 2: KV cache with MHA per token per layer
    kv_per_token_mha = 2 * num_kv_heads_mha * head_dim * fp16_bytes
    ratio = kv_per_token_mha / kv_per_token_gqa
    print("  Part 2: KV cache with MHA (64 KV heads)")
    print(f"    Per token per layer = 2 * {num_kv_heads_mha} * {head_dim} * {fp16_bytes} = {kv_per_token_mha:,} bytes = {kv_per_token_mha/1024:.0f} KB")
    print(f"    MHA requires {ratio:.0f}x more KV cache than GQA")
    print()

    # Part 3: Total savings
    gqa_total = kv_per_token_gqa * batch_size * seq_len * num_layers
    mha_total = kv_per_token_mha * batch_size * seq_len * num_layers
    savings = mha_total - gqa_total

    print("  Part 3: Total for batch=32, seq_len=4096, 80 layers")
    print(f"    GQA total: {gqa_total / (1024**3):.2f} GB")
    print(f"    MHA total: {mha_total / (1024**3):.2f} GB")
    print(f"    Savings: ~{savings / (1024**3):.2f} GB")
    print(f"    GQA enables serving 70B at this batch/sequence size on far fewer GPUs.")


# === Exercise 3: SwiGLU vs GELU Analysis ===
# Problem: Analyze gating mechanism and dimension adjustment.

def exercise_3():
    """Solution: SwiGLU analysis"""
    print("  SwiGLU(x, W, V, b, c) = Swish(xW + b) * (xV + c)")
    print()

    print("  1. What gating accomplishes:")
    print("     The element-wise multiplication creates CONDITIONAL COMPUTATION:")
    print("     the gate Swish(xW) acts as a learnable filter that can 'turn off'")
    print("     specific dimensions of the value projection (xV). This allows")
    print("     input-dependent feature selection, unlike standard activations.")
    print()

    print("  2. Why gating benefits FFN 'key-value memories':")
    print("     FFN hidden units correspond to 'key' patterns and 'values' to write.")
    print("     The gate dynamically suppresses irrelevant memories based on input.")
    print("     For diverse contexts (code, reasoning, facts), selective activation")
    print("     of relevant knowledge is more expressive than uniform application.")
    print()

    d_model = 4096
    # Standard FFN: 2 matrices of (d x 4d) -> total = 8d^2
    standard_params = 2 * d_model * (4 * d_model)
    # SwiGLU: 3 matrices of (d x h) -> total = 3*d*h
    # To match: 3*d*h = 8*d^2 -> h = 8d/3
    h = int(8 * d_model / 3)
    # Round to multiple of 256
    h_rounded = ((h + 255) // 256) * 256
    swiglu_params = 3 * d_model * h_rounded

    print("  3. Dimension adjustment:")
    print(f"     Standard FFN: 2 matrices of ({d_model} x {4*d_model}) = {standard_params:,} params")
    print(f"     SwiGLU FFN: 3 matrices of ({d_model} x h)")
    print(f"     To match: 3*d*h = 8*d^2 -> h = 8d/3 = {8*d_model/3:.0f}")
    print(f"     Rounded to multiple of 256: h = {h_rounded}")
    print(f"     SwiGLU total: {swiglu_params:,} params")
    print(f"     Ratio: {swiglu_params/standard_params:.3f} (approximately parameter-neutral)")


# === Exercise 4: LLaMA Version Selection ===
# Problem: Recommend LLaMA version and size for 4 use cases.

def exercise_4():
    """Solution: LLaMA version selection"""
    cases = [
        {
            "case": "Raspberry Pi 5, 8GB RAM, smart home assistant",
            "model": "LLaMA 3.2 1B",
            "reason": (
                "1B model designed for on-device/edge inference. "
                "With 4-bit quantization: ~0.5GB footprint. "
                "LLaMA 3.2 explicitly designed for edge deployment."
            ),
        },
        {
            "case": "Single A100 80GB, medical QA fine-tuning",
            "model": "LLaMA 3 8B with QLoRA",
            "reason": (
                "8B fits in fp16 (16GB), leaving room for fine-tuning overhead. "
                "QLoRA (4-bit + LoRA) reduces to ~6GB. "
                "Better base performance for domain adaptation."
            ),
        },
        {
            "case": "4x H100 server, general-purpose chat",
            "model": "LLaMA 3.1 70B",
            "reason": (
                "70B in bf16 needs ~140GB, fits across 4xH100 with tensor "
                "parallelism. GPT-4-competitive for general chat. "
                "Strong instruction-following and tool use."
            ),
        },
        {
            "case": "50K-token legal document processing",
            "model": "LLaMA 3.1 8B or 70B",
            "reason": (
                "LLaMA 3.1 natively supports 128K context window, essential "
                "for 50K-token documents. LLaMA 3 (without .1) only supports "
                "8K natively. Choose 8B for cost, 70B for accuracy."
            ),
        },
    ]

    for c in cases:
        print(f"  Use case: {c['case']}")
        print(f"    Recommended: {c['model']}")
        print(f"    Reason: {c['reason']}")
        print()


if __name__ == "__main__":
    print("=== Exercise 1: RoPE vs Absolute Position Encoding ===")
    exercise_1()
    print("\n=== Exercise 2: GQA Memory Savings Calculation ===")
    exercise_2()
    print("\n=== Exercise 3: SwiGLU vs GELU Analysis ===")
    exercise_3()
    print("\n=== Exercise 4: LLaMA Version Selection ===")
    exercise_4()
    print("\nAll exercises completed!")
