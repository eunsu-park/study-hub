"""
Exercises for Lesson 10: Long Context Models
Topic: Foundation_Models

Solutions to practice problems from the lesson.
"""

import math


# === Exercise 1: Attention Complexity Analysis ===
# Problem: Calculate memory for full attention and compare with FlashAttention.

def exercise_1():
    """Solution: Attention complexity analysis"""
    n = 8192
    d_model = 4096
    num_heads = 32
    head_dim = 128
    fp16_bytes = 2

    # Part 1: Full attention weight matrix memory
    attn_bytes = num_heads * n * n * fp16_bytes
    attn_gb = attn_bytes / (1024**3)
    print("  Part 1: Full attention weight matrix (n=8192)")
    print(f"    {num_heads} heads * {n} * {n} * {fp16_bytes} bytes = {attn_bytes:,} bytes")
    print(f"    = {attn_gb:.1f} GB")
    print()

    # Part 2: Peak memory comparison
    standard_total_gb = attn_gb * 2.5  # softmax, dropout overhead
    flash_bytes = n * head_dim * num_heads * fp16_bytes
    flash_gb = flash_bytes / (1024**3)
    ratio = standard_total_gb / flash_gb

    print("  Part 2: Peak memory comparison")
    print(f"    Standard attention (full n*n matrix): ~{standard_total_gb:.0f} GB")
    print(f"    FlashAttention (O(n*d) in SRAM): ~{flash_gb*1000:.0f} MB")
    print(f"    FlashAttention uses ~{ratio:.0f}x less memory for n={n}")
    print()

    # Part 3: When does attention matrix exceed 40GB?
    target_bytes = 40 * (1024**3)
    # num_heads * n^2 * fp16_bytes = target_bytes
    n_squared = target_bytes / (num_heads * fp16_bytes)
    n_threshold = math.sqrt(n_squared)

    print("  Part 3: When does attention matrix exceed 40GB?")
    print(f"    {num_heads} * n^2 * {fp16_bytes} = 40 * 10^9")
    print(f"    n^2 = {n_squared:,.0f}")
    print(f"    n = {n_threshold:,.0f} tokens")
    print(f"    At ~{n_threshold/1000:.0f}K tokens, attention alone fills an A100.")


# === Exercise 2: "Lost in the Middle" Phenomenon ===
# Problem: Explain and propose mitigations.

def exercise_2():
    """Solution: Lost in the Middle analysis"""
    print("  1. Hypothesis for 'lost in the middle':")
    print("     During training, relevant info for predictions tends to appear")
    print("     at the beginning (introductions, premises) and end (conclusions).")
    print("     The middle contains supporting detail that is less frequently")
    print("     the direct 'answer' target during next-token prediction.")
    print("     Attention mechanisms implicitly learn to weight boundary positions.")
    print()

    print("  2. RAG mitigation strategies:")
    print("     a) Position-aware ordering: Place most relevant docs at positions")
    print("        1 and 10 (beginning/end), fill middle with less relevant docs.")
    print("     b) Sandwich prompt structure: question -> context -> question again.")
    print("        Or use multiple shorter contexts with aggregation.")
    print()

    print("  3. Impact on agent system design:")
    print("     Tool outputs accumulate, pushing early important results to middle.")
    print("     The agent may over-rely on most recent outputs (recency bias).")
    print("     Mitigation: Use a running 'scratchpad' refreshed at top of context")
    print("     each turn, rather than pure context accumulation.")


# === Exercise 3: Position Encoding Extension ===
# Problem: Compare direct inference, position interpolation, and YaRN.

def exercise_3():
    """Solution: Position encoding extension comparison"""
    approaches = [
        {
            "name": "Direct inference at 32K",
            "description": "Just run the model at longer lengths",
            "tradeoff": (
                "Catastrophic -- RoPE values at positions >4K are completely "
                "out-of-distribution. Attention patterns collapse. Performance "
                "degrades severely beyond training length."
            ),
        },
        {
            "name": "Position Interpolation",
            "description": "Scale positions down by s=32K/4K=8",
            "tradeoff": (
                "Works but degrades quality for short sequences: positions "
                "1-512 compressed into 0-64, making it harder to distinguish "
                "nearby tokens. Requires some fine-tuning (100-1000 steps). "
                "Better than direct inference, simpler than YaRN."
            ),
        },
        {
            "name": "YaRN",
            "description": "NTK-aware interpolation with attention temperature",
            "tradeoff": (
                "Best quality: applies different scaling to different frequency "
                "components of RoPE. Attention temperature prevents distribution "
                "from becoming too peaked. Minor fine-tuning recommended but "
                "often works zero-shot. Preferred method in practice."
            ),
        },
    ]

    for a in approaches:
        print(f"  {a['name']}:")
        print(f"    Description: {a['description']}")
        print(f"    Trade-off: {a['tradeoff']}")
        print()

    print("  Recommendation: Use YaRN for best quality extension;")
    print("  Position Interpolation for simplicity. Never use direct inference")
    print("  beyond ~1.5-2x training length.")


# === Exercise 4: FlashAttention Algorithm Understanding ===
# Problem: Explain why naive attention is memory-problematic, the online
# softmax trick, and why FlashAttention is faster despite more arithmetic.

def exercise_4():
    """Solution: FlashAttention algorithm understanding"""
    print("  1. Why naive attention is memory-problematic:")
    print("     Materializes intermediate tensors:")
    print("       S = QK^T: shape (n x n) -- full attention score matrix")
    print("       P = softmax(S): shape (n x n) -- normalized weights")
    print("       O = PV: shape (n x d) -- output")
    print("     For n=32K fp16: QK^T alone = 32K*32K*2 = ~2GB per head")
    print("     Times 32 heads = 64GB -- far exceeds GPU memory.")
    print()

    print("  2. Online softmax trick:")
    print("     Standard softmax needs two passes (find max, compute exp).")
    print("     FlashAttention uses incremental update rule:")
    print("       For each new tile of K,V processed:")
    print("       m_new = max(m_old, max(S_new))")
    print("       l_new = exp(m_old - m_new) * l_old + sum(exp(S_new - m_new))")
    print("       O_new = (exp(m_old-m_new)*l_old*O_old + exp(S_new-m_new)*V_new) / l_new")
    print("     This computes EXACT softmax without storing full n*n matrix.")
    print()

    print("  3. Why FlashAttention is faster despite more arithmetic:")
    print("     Standard attention is MEMORY-BANDWIDTH bound, not compute bound.")
    print("     Bottleneck is reading/writing large intermediate matrices to HBM.")
    print("     FlashAttention keeps everything in SRAM (on-chip cache):")
    print("       SRAM bandwidth: ~19 TB/s  vs  HBM: ~2 TB/s")
    print("     By computing tiles entirely in SRAM, total data movement to HBM")
    print("     is reduced by O(n) factor. This yields 2-4x wall-clock speedup")
    print("     despite 10-15% more arithmetic operations.")


if __name__ == "__main__":
    print("=== Exercise 1: Attention Complexity Analysis ===")
    exercise_1()
    print("\n=== Exercise 2: Lost in the Middle ===")
    exercise_2()
    print("\n=== Exercise 3: Position Encoding Extension ===")
    exercise_3()
    print("\n=== Exercise 4: FlashAttention Algorithm ===")
    exercise_4()
    print("\nAll exercises completed!")
