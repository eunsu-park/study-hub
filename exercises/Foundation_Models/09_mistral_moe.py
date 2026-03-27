"""
Exercises for Lesson 09: Mistral & Mixture of Experts
Topic: Foundation_Models

Solutions to practice problems from the lesson.
"""

import math


# === Exercise 1: MoE Effective Parameter Analysis ===
# Problem: Analyze Mixtral 8x7B active parameters and efficiency ratio.

def exercise_1():
    """Solution: MoE effective parameter analysis"""
    total_params = 46.7e9
    active_params = 12.9e9
    num_experts = 8
    top_k = 2

    # Part 1
    print("  Part 1: Why only 12.9B active parameters")
    print(f"    Mixtral has {num_experts} experts per MoE layer, activates {top_k} per token.")
    print(f"    Non-selected {num_experts - top_k} experts don't execute at all.")
    print(f"    Shared attention params + {top_k}/{num_experts} of MoE FFN params = ~{active_params/1e9:.1f}B")
    print()

    # Part 2
    efficiency = active_params / total_params
    print("  Part 2: Computation efficiency ratio")
    print(f"    Ratio = {active_params/1e9:.1f}B / {total_params/1e9:.1f}B = {efficiency:.1%}")
    print(f"    Each forward pass costs ~{efficiency:.1%} of a dense {total_params/1e9:.1f}B model.")
    print(f"    Pay for inference like a ~{active_params/1e9:.0f}B model,")
    print(f"    benefit from {total_params/1e9:.1f}B parameter capacity.")
    print()

    # Part 3: K=4
    # Roughly double the expert computation
    active_k4 = active_params * 2  # approximate
    efficiency_k4 = active_k4 / total_params
    print("  Part 3: Effect of K=4 (instead of K=2)")
    print(f"    Active params roughly double: ~{active_k4/1e9:.1f}B per forward pass")
    print(f"    Efficiency ratio: {active_k4/1e9:.1f} / {total_params/1e9:.1f} = {efficiency_k4:.1%}")
    print(f"    Quality typically improves, but loses efficiency advantage.")


# === Exercise 2: Sliding Window Attention Memory Analysis ===
# Problem: Compare standard MHA vs SWA KV cache memory.

def exercise_2():
    """Solution: Sliding window attention memory analysis"""
    W = 4096  # window size
    n = 32768  # sequence length
    num_heads = 32
    head_dim = 128
    fp16 = 2
    num_layers = 32

    # Part 1
    print("  Part 1: SWA memory complexity")
    print(f"    SWA stores only last W={W} tokens in KV cache.")
    print(f"    Memory complexity: O(W * d) = O(d), constant w.r.t. sequence length.")
    print()

    # Part 2
    mha_bytes = n * 2 * num_heads * head_dim * fp16 * num_layers
    swa_bytes = W * 2 * num_heads * head_dim * fp16 * num_layers

    print(f"  Part 2: KV cache comparison for n={n} tokens")
    print(f"    Standard MHA (stores all n tokens):")
    print(f"      {n} * 2(K+V) * {num_heads} * {head_dim} * {fp16} * {num_layers}")
    print(f"      = {mha_bytes:,} bytes = {mha_bytes/(1024**3):.0f} GB")
    print()
    print(f"    SWA (stores only last W={W} tokens):")
    print(f"      {W} * 2 * {num_heads} * {head_dim} * {fp16} * {num_layers}")
    print(f"      = {swa_bytes:,} bytes = {swa_bytes/(1024**3):.0f} GB")
    print()
    savings = mha_bytes - swa_bytes
    ratio = mha_bytes / swa_bytes
    print(f"    SWA saves ~{savings/(1024**3):.0f} GB -- an {ratio:.0f}x reduction")
    print()

    # Part 3
    effective_receptive = W * num_layers
    print("  Part 3: Information loss and compensation")
    print(f"    When a token falls outside window, direct attention is lost.")
    print(f"    Multi-layer receptive field expansion compensates:")
    print(f"      Layer 1: attends to last W tokens")
    print(f"      Layer 2: effectively sees W^2 tokens of history")
    print(f"      After k layers: receptive field ~= W * k")
    print(f"      For W={W}, {num_layers} layers: ~{effective_receptive:,} tokens")


# === Exercise 3: Load Balancing Loss ===
# Problem: Explain what happens without load balancing loss.

def exercise_3():
    """Solution: Load balancing loss analysis"""
    print("  Without load balancing loss, router collapse proceeds as follows:")
    print()

    steps = [
        ("Step 1: Initial asymmetry",
         "During initialization, router has random weights. Some experts "
         "receive slightly higher routing probabilities by chance."),
        ("Step 2: Rich-get-richer dynamics",
         "Preferred experts receive more tokens -> improve more -> "
         "produce better outputs -> router prefers them more strongly."),
        ("Step 3: Expert collapse",
         "After sufficient training, 1-2 experts receive nearly all tokens. "
         "Other 6-7 experts receive almost no gradient signal and stagnate."),
        ("Step 4: Complete collapse",
         "Model effectively becomes a dense model with 1-2 FFN matrices. "
         "Total params still 8x larger but only 1-2 are useful."),
    ]

    for title, desc in steps:
        print(f"  {title}:")
        print(f"    {desc}")
        print()

    print("  What load balancing loss does:")
    print("    Penalizes when f_e (fraction of tokens to expert e) deviates")
    print("    from uniform (1/E). Creates gradient pushing router toward")
    print("    even distribution. Combined with noisy top-k (exploration noise),")
    print("    prevents early convergence to degenerate solution.")

    # Simulate simple router collapse
    import random
    random.seed(42)

    num_experts = 8
    num_tokens = 1000

    # Without load balancing: simulate rich-get-richer
    print()
    print("  Simulation: expert utilization over training rounds")
    print(f"    {'Round':<10}", end="")
    for e in range(num_experts):
        print(f"  E{e}", end="")
    print()

    scores = [1.0] * num_experts
    for round_num in range(6):
        # Softmax to get probabilities
        total = sum(math.exp(s) for s in scores)
        probs = [math.exp(s) / total for s in scores]

        # Route tokens
        counts = [0] * num_experts
        for _ in range(num_tokens):
            r = random.random()
            cumsum = 0.0
            for e, p in enumerate(probs):
                cumsum += p
                if r < cumsum:
                    counts[e] += 1
                    break

        # Update scores: more tokens -> better -> higher score
        for e in range(num_experts):
            scores[e] += counts[e] / num_tokens * 0.5

        print(f"    Round {round_num:<4}", end="")
        for c in counts:
            pct = c / num_tokens * 100
            print(f"  {pct:3.0f}%", end="")
        print()


# === Exercise 4: MoE vs Dense Trade-off ===
# Problem: Choose between dense 13B and sparse MoE 46.7B for 4 scenarios.

def exercise_4():
    """Solution: MoE vs Dense trade-off decision"""
    scenarios = [
        {
            "scenario": "Memory-constrained edge server (32GB RAM)",
            "choice": "Dense 13B",
            "reason": (
                "MoE requires loading all 46.7B params (~93GB in fp16), "
                "far exceeding 32GB. Dense 13B needs ~26GB in fp16."
            ),
        },
        {
            "scenario": "Research lab, fine-tuning on small datasets",
            "choice": "Dense 13B",
            "reason": (
                "MoE models are harder to fine-tune: router must learn "
                "expert assignments, small datasets don't provide enough "
                "signal, instability is common. Dense fine-tunes predictably."
            ),
        },
        {
            "scenario": "Maximum knowledge breadth across diverse topics",
            "choice": "MoE 46.7B",
            "reason": (
                "Different experts specialize in different domains. "
                "46.7B total params = far more capacity for diverse knowledge. "
                "For general-purpose APIs, breadth advantage is significant."
            ),
        },
        {
            "scenario": "Exactly reproducible outputs",
            "choice": "Prefer Dense 13B",
            "reason": (
                "MoE adds non-determinism from discrete top-K routing. "
                "Minor floating-point differences can flip which expert is "
                "selected. Dense only has floating-point precision as "
                "non-determinism source."
            ),
        },
    ]

    for s in scenarios:
        print(f"  Scenario: {s['scenario']}")
        print(f"    Choice: {s['choice']}")
        print(f"    Reason: {s['reason']}")
        print()


if __name__ == "__main__":
    print("=== Exercise 1: MoE Effective Parameter Analysis ===")
    exercise_1()
    print("\n=== Exercise 2: Sliding Window Attention Memory Analysis ===")
    exercise_2()
    print("\n=== Exercise 3: Load Balancing Loss ===")
    exercise_3()
    print("\n=== Exercise 4: MoE vs Dense Trade-off ===")
    exercise_4()
    print("\nAll exercises completed!")
