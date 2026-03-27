"""
Exercises for Lesson 06: Pretraining Infrastructure
Topic: Foundation_Models

Solutions to practice problems from the lesson.
"""

import math


# === Exercise 1: Memory Budget Estimation ===
# Problem: Estimate GPU memory for a 7B model with AdamW optimizer.

def exercise_1():
    """Solution: Memory budget estimation"""
    params = 7e9
    batch_size = 4
    seq_len = 2048
    hidden_dim = 4096
    num_layers = 32

    # Model parameters (fp16)
    model_mem_gb = params * 2 / (1024**3)
    print(f"  1. Model parameters (fp16): {params/1e9:.0f}B * 2 bytes = {model_mem_gb:.0f} GB")

    # Optimizer states (AdamW: fp32 copy + 1st moment + 2nd moment)
    opt_fp32_copy = params * 4 / (1024**3)
    opt_momentum = params * 4 / (1024**3)
    opt_variance = params * 4 / (1024**3)
    opt_total = opt_fp32_copy + opt_momentum + opt_variance

    print(f"  2. Optimizer states (AdamW in fp32):")
    print(f"     fp32 parameter master copy: {params/1e9:.0f}B * 4 bytes = {opt_fp32_copy:.0f} GB")
    print(f"     1st moment (momentum):      {params/1e9:.0f}B * 4 bytes = {opt_momentum:.0f} GB")
    print(f"     2nd moment (variance):      {params/1e9:.0f}B * 4 bytes = {opt_variance:.0f} GB")
    print(f"     Total optimizer states: {opt_total:.0f} GB")

    # Gradients (fp32)
    grad_mem = params * 4 / (1024**3)
    print(f"  3. Gradients (fp32): {params/1e9:.0f}B * 4 bytes = {grad_mem:.0f} GB")

    total = model_mem_gb + opt_total + grad_mem
    print(f"\n  Total (excluding activations): {model_mem_gb:.0f} + {opt_total:.0f} + {grad_mem:.0f} = ~{total:.0f} GB")

    # Activations estimate
    act_mem = batch_size * seq_len * hidden_dim * num_layers * 2 / (1024**3)
    print(f"\n  Activations estimate: {batch_size}*{seq_len}*{hidden_dim}*{num_layers}*2 = ~{act_mem:.0f} GB")
    print(f"\n  Which dominates: Optimizer states ({opt_total:.0f} GB) = {opt_total/model_mem_gb:.0f}x model params")
    print(f"  This is why ZeRO Stage 1 targets optimizer state sharding first.")


# === Exercise 2: Parallelism Strategy Selection ===
# Problem: Recommend parallelism strategy for 3 scenarios.

def exercise_2():
    """Solution: Parallelism strategy selection"""
    scenarios = [
        {
            "name": "A: 7B model, 8x A100 80GB, single node, fits on 1 GPU",
            "recommendation": "Data Parallelism (DDP or FSDP)",
            "justification": (
                "Model fits on one GPU -> no need to split. "
                "Replicate model, split batches across 8 GPUs. "
                "FSDP (ZeRO-3) can shard optimizer states for larger batch sizes. "
                "Intra-node NVLink makes all-reduce very fast."
            ),
        },
        {
            "name": "B: 70B model, 64x A100 80GB across 8 nodes, doesn't fit on 1 GPU",
            "recommendation": "3D Parallelism (TP=4 + PP=2 + DP=8)",
            "justification": (
                "TP=4: Split each layer across 4 GPUs within a node (fast NVLink). "
                "PP=2: Split layers across 2 pipeline stages across nodes. "
                "DP=8: Replicate TP*PP model across 8 copies. "
                "Total: 4*2*8 = 64 GPUs."
            ),
        },
        {
            "name": "C: 7B model, 8K context, attention bottleneck",
            "recommendation": "Sequence Parallelism (SP) + FlashAttention",
            "justification": (
                "8K context -> O(8K^2) = 64M ops per attention layer. "
                "SP splits sequence dimension across GPUs. "
                "FlashAttention reduces memory from O(n^2) to O(n) via tiling. "
                "Can be combined."
            ),
        },
    ]

    for s in scenarios:
        print(f"  Scenario {s['name']}")
        print(f"    Recommendation: {s['recommendation']}")
        print(f"    Justification: {s['justification']}")
        print()


# === Exercise 3: Gradient Clipping Analysis ===
# Problem: Explain gradient norm clipping mathematically and its effects.

def exercise_3():
    """Solution: Gradient clipping analysis"""
    import numpy as np

    print("  1. What gradient norm clipping does:")
    print("     Computes global L2 norm: global_norm = sqrt(sum(g_i^2))")
    print("     If global_norm > max_norm:")
    print("       g_i <- g_i * (max_norm / global_norm)")
    print("     Preserves gradient direction, bounds magnitude.")
    print()

    # Demonstrate with example gradients
    np.random.seed(42)
    gradients = [np.random.randn(10) * 3 for _ in range(5)]

    global_norm = math.sqrt(sum(np.sum(g**2) for g in gradients))
    print(f"  Example: 5 parameter groups, global_norm = {global_norm:.2f}")

    max_norm = 1.0
    if global_norm > max_norm:
        scale = max_norm / global_norm
        clipped = [g * scale for g in gradients]
        clipped_norm = math.sqrt(sum(np.sum(g**2) for g in clipped))
        print(f"  After clipping (max_norm={max_norm}): norm = {clipped_norm:.2f}")
    print()

    print("  2. Why it matters for large model training:")
    print("     - Gradients compound multiplicatively through layers (exploding gradients)")
    print("     - Early training: initialization produces large gradient magnitudes")
    print("     - A single large step can destroy learned representations")
    print("     - Loss spikes are often caused by such large updates")
    print()

    print("  3. max_norm too small vs too large:")
    print("     max_norm=0.001 (too small): Nearly every update scaled down severely.")
    print("       Training extremely slow -- equivalent to tiny learning rate.")
    print("     max_norm=1000 (too large): Clipping never activates.")
    print("       No protection against gradient explosions during instabilities.")
    print("     max_norm=1.0: Common default for LLMs, activates rarely (instabilities only).")


# === Exercise 4: ZeRO Stage Comparison ===
# Problem: Complete the ZeRO stages comparison table.

def exercise_4():
    """Solution: ZeRO stage comparison"""
    stages = [
        {
            "stage": "ZeRO-1",
            "sharded": "Optimizer states (momentum + variance)",
            "saving": "~4x reduction in optimizer state memory",
            "communication": "Minimal: only all-reduce after parameter update (same as DDP)",
        },
        {
            "stage": "ZeRO-2",
            "sharded": "Optimizer states + gradients",
            "saving": "~8x reduction in optimizer + gradient memory",
            "communication": "Moderate: reduce-scatter gradients (slightly more efficient than DDP)",
        },
        {
            "stage": "ZeRO-3",
            "sharded": "Optimizer states + gradients + model parameters",
            "saving": "~Nx reduction (linear with number of GPUs)",
            "communication": (
                "Significant: all-gather parameters before each forward/backward pass"
            ),
        },
    ]

    print(f"  {'Stage':<10} | {'What is sharded':<40} | {'Memory saving':<35} | {'Communication'}")
    print("  " + "-" * 130)
    for s in stages:
        print(f"  {s['stage']:<10} | {s['sharded']:<40} | {s['saving']:<35} | {s['communication']}")

    print()
    print("  Key insight: ZeRO-3 achieves the best memory efficiency but at")
    print("  2x more communication ops per step compared to standard DDP.")
    print("  When models fit in memory, ZeRO-1/2 is preferred for lower latency.")


if __name__ == "__main__":
    print("=== Exercise 1: Memory Budget Estimation ===")
    exercise_1()
    print("\n=== Exercise 2: Parallelism Strategy Selection ===")
    exercise_2()
    print("\n=== Exercise 3: Gradient Clipping Analysis ===")
    exercise_3()
    print("\n=== Exercise 4: ZeRO Stage Comparison ===")
    exercise_4()
    print("\nAll exercises completed!")
