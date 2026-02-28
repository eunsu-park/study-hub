"""
Exercises for Lesson 22: Inference Optimization
Topic: Foundation_Models

Solutions to practice problems from the lesson.
"""


# === Exercise 1: KV Cache Memory Calculation ===
# Problem: Calculate KV cache memory for LLaMA-7B serving.

def exercise_1():
    """Solution: KV cache memory calculation"""
    num_layers = 32
    num_heads = 32
    head_dim = 128
    seq_len = 4096
    dtype_size = 2  # FP16
    batch_size = 16
    gpu_memory_gb = 40

    # KV cache per sequence
    kv_per_seq = 2 * num_layers * num_heads * head_dim * seq_len * dtype_size
    kv_per_seq_gb = kv_per_seq / (1024**3)

    print(f"  KV cache per sequence:")
    print(f"    2 * {num_layers} * {num_heads} * {head_dim} * {seq_len} * {dtype_size}")
    print(f"    = {kv_per_seq:,} bytes = {kv_per_seq_gb:.1f} GB")
    print()

    # For batch size 16
    kv_batch = kv_per_seq_gb * batch_size
    print(f"  For batch={batch_size}: {kv_per_seq_gb:.1f} GB * {batch_size} = {kv_batch:.0f} GB")
    print()

    # Model weights
    model_gb = 14  # 7B * 2 bytes
    total = model_gb + kv_batch
    print(f"  Model weights (FP16): ~{model_gb} GB")
    print(f"  Total with batch={batch_size}: {model_gb} + {kv_batch:.0f} = {total:.0f} GB")
    print(f"  {'EXCEEDS' if total > gpu_memory_gb else 'fits in'} {gpu_memory_gb} GB limit")
    print()

    # Maximum batch size
    available = gpu_memory_gb - model_gb
    max_batch = int(available / kv_per_seq_gb)
    print(f"  Maximum batch size within {gpu_memory_gb} GB:")
    print(f"    Available for KV: {gpu_memory_gb} - {model_gb} = {available} GB")
    print(f"    Max batch = {available} / {kv_per_seq_gb:.1f} = {max_batch} sequences")
    print()
    print("  PagedAttention solves this: allocates blocks on demand,")
    print("  so average utilization is much higher.")


# === Exercise 2: Speculative Decoding Acceptance Rate ===
# Problem: Analyze speedup under different acceptance rates.

def exercise_2():
    """Solution: Speculative decoding analysis"""
    k = 5  # speculative tokens

    scenarios = [
        {
            "name": "A) High acceptance (alpha ~= 95%)",
            "alpha": 0.95,
            "analysis": (
                "Expected tokens per target pass ~= 5 * 0.95 = 4.75. "
                "Theoretical speedup ~4.75x. In practice: ~2-3x (draft overhead). "
                "Best scenario: draft and target are similar architectures."
            ),
        },
        {
            "name": "B) Low acceptance (alpha ~= 30%)",
            "alpha": 0.30,
            "analysis": (
                "Expected tokens per target pass ~= 5 * 0.30 = 1.5. "
                "Target still runs on full k-token sequence. "
                "Net speedup ~1.5x minus draft overhead -> barely faster. "
                "May be SLOWER than no speculation!"
            ),
        },
        {
            "name": "C) Text type analysis",
            "alpha": None,
            "analysis": (
                "Common phrases ('the quick brown fox'): draft accurate -> "
                "high alpha -> good speedup. "
                "Rare technical jargon ('CYP3A4 enzyme inhibits'): draft "
                "guesses generic words while target predicts specific terms -> "
                "low alpha -> poor speedup."
            ),
        },
    ]

    for s in scenarios:
        print(f"  {s['name']}:")
        if s['alpha']:
            expected = k * s['alpha']
            print(f"    Expected tokens/pass = {k} * {s['alpha']} = {expected:.2f}")
        print(f"    {s['analysis']}")
        print()

    print("  Key insight: Speculative decoding works best when output is")
    print("  'predictable' by the draft model. For domain-specific text,")
    print("  choose a draft model specializing in the same domain.")


# === Exercise 3: Quantization Method Selection ===
# Problem: Select quantization method for 3 deployment scenarios.

def exercise_3():
    """Solution: Quantization method selection"""
    scenarios = [
        {
            "scenario": "A) Customer chatbot, 1000 req/day, A100 80GB",
            "method": "FP16",
            "reason": (
                "13B * 2 bytes = 26GB -- easily fits in 80GB. "
                "With only 1000 req/day (~0.7/min), throughput not bottleneck. "
                "Full precision maximizes quality. No reason to sacrifice quality."
            ),
        },
        {
            "scenario": "B) Edge deployment, CPU only, 16GB RAM",
            "method": "GGUF Q4_K_M",
            "reason": (
                "CPU inference requires llama.cpp + GGUF. "
                "FP16 = 26GB doesn't fit. GGUF Q4 = ~6.5GB fits easily. "
                "Q4_K_M gives best quality/size trade-off for CPU."
            ),
        },
        {
            "scenario": "C) High-throughput API, 100K req/day, 2x A100",
            "method": "AWQ 4-bit",
            "reason": (
                "100K req/day = ~70/min, throughput critical. "
                "AWQ provides ~2x speedup over FP16. "
                "~6.5GB per GPU allows batch=~60 with TP. "
                "Preserves 98-99% of FP16 quality."
            ),
        },
    ]

    for s in scenarios:
        print(f"  {s['scenario']}")
        print(f"    Method: {s['method']}")
        print(f"    Reason: {s['reason']}")
        print()

    print("  Key principle: bottleneck determines quantization choice.")
    print("    Quality -> FP16. Edge CPU -> GGUF. Throughput -> AWQ/GPTQ.")


# === Exercise 4: Continuous vs Static Batching ===
# Problem: Explain why continuous batching improves throughput.

def exercise_4():
    """Solution: Continuous vs static batching"""
    # Concrete example
    requests = {"A": 500, "B": 20, "C": 200}
    batch_size = 3

    print("  Example: 3 concurrent requests")
    for name, tokens in requests.items():
        print(f"    Request {name}: expects {tokens} tokens")
    print()

    print("  Static batching behavior:")
    print("    Step 0-20:    [A, B, C] -- all 3 generating")
    print("    Step 21:      B finishes. GPU runs [A, _, C] -- 1/3 wasted")
    print("    Step 21-200:  [A, _, C] -- only 2/3 utilization")
    print("    Step 201:     C finishes. GPU runs [A, _, _] -- 2/3 wasted")
    print("    Step 201-500: [A, _, _] -- only 1/3 utilization")
    print()

    print("  Continuous batching behavior:")
    print("    Step 0-20:    [A, B, C] -- all 3 generating")
    print("    Step 21:      B finishes -> immediately add Request D")
    print("    Step 21-200:  [A, C, D] -- batch stays full!")
    print("    Step 201:     C finishes -> immediately add Request E")
    print("    Step 201+:    [A, D, E] -- batch stays full!")
    print()

    # Throughput comparison
    # Static: weighted average batch size over 500 steps
    static_weighted = (3 * 20 + 2 * 180 + 1 * 300) / 500
    continuous_avg = 3.0
    improvement = continuous_avg / static_weighted

    print("  Throughput comparison:")
    print(f"    Static avg batch size: (3*20 + 2*180 + 1*300) / 500 = {static_weighted:.2f}")
    print(f"    Continuous avg batch size: ~{continuous_avg:.0f} (maintained with new requests)")
    print(f"    Throughput improvement: {continuous_avg:.0f}/{static_weighted:.2f} = ~{improvement:.1f}x")
    print()
    print("  Real-world gains are often 3-5x because production workloads have")
    print("  much higher variance in output lengths.")


if __name__ == "__main__":
    print("=== Exercise 1: KV Cache Memory ===")
    exercise_1()
    print("\n=== Exercise 2: Speculative Decoding ===")
    exercise_2()
    print("\n=== Exercise 3: Quantization Selection ===")
    exercise_3()
    print("\n=== Exercise 4: Continuous vs Static Batching ===")
    exercise_4()
    print("\nAll exercises completed!")
