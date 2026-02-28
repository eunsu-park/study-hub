"""
Exercises for Lesson 02: Scaling Laws
Topic: Foundation_Models

Solutions to practice problems from the lesson.
"""

import math


# === Exercise 1: Chinchilla Optimal Allocation ===
# Problem: Using Chinchilla rule (D ~ 20*N) and C ~ 6*N*D:
# 1. Find Chinchilla-optimal N and D for C = 6e23 FLOPs
# 2. Is GPT-3 (175B params, 300B tokens) over- or under-trained?

def exercise_1():
    """Solution: Chinchilla optimal allocation"""
    # Part 1: Optimal allocation for C = 6e23
    C = 6e23
    # D = 20*N => C = 6*N*20*N = 120*N^2
    # N^2 = C / 120
    N_squared = C / 120
    N_optimal = math.sqrt(N_squared)
    D_optimal = 20 * N_optimal

    print("  Part 1: Chinchilla-optimal allocation for C = 6e23 FLOPs")
    print(f"    C = 6*N*D = 6*N*20*N = 120*N^2")
    print(f"    N^2 = C / 120 = {C} / 120 = {N_squared:.2e}")
    print(f"    N = sqrt({N_squared:.2e}) = {N_optimal:.2e} = ~{N_optimal/1e9:.1f}B parameters")
    print(f"    D = 20 * {N_optimal/1e9:.1f}B = {D_optimal/1e9:.0f}B tokens")
    print()

    # Part 2: GPT-3 assessment
    gpt3_N = 175e9
    gpt3_D = 300e9
    D_star = 20 * gpt3_N  # Chinchilla-optimal D for 175B

    ratio = gpt3_D / D_star
    under_factor = D_star / gpt3_D

    print("  Part 2: GPT-3 assessment")
    print(f"    Chinchilla-optimal for 175B params: D* = 20 * 175B = {D_star/1e12:.1f}T tokens")
    print(f"    GPT-3 actual: D = 300B tokens")
    print(f"    Ratio: 300B / {D_star/1e9:.0f}B = {ratio:.3f}")
    print(f"    GPT-3 trained on only {ratio*100:.1f}% of optimal token count")
    print(f"    GPT-3 is severely UNDER-TRAINED by a factor of ~{under_factor:.1f}x")


# === Exercise 2: Training Cost Estimation ===
# Problem: Estimate training cost for a 13B model on 260B tokens
# using A100 GPUs at $2/hour.

def exercise_2():
    """Solution: Training cost estimation"""
    N = 13e9       # 13B parameters
    D = 260e9      # 260B tokens (= 20 * 13B, Chinchilla-optimal)
    gpu_cost_per_hour = 2.0
    gpu_flops_per_sec = 300e12  # A100 effective TFLOPS

    # Step 1: Total FLOPs
    total_flops = 6 * N * D
    print(f"  Step 1: Total FLOPs = 6 * {N:.0e} * {D:.0e} = {total_flops:.3e}")

    # Step 2: GPU throughput
    print(f"  Step 2: A100 effective throughput = {gpu_flops_per_sec:.0e} FLOPS")

    # Step 3: GPU-seconds
    gpu_seconds = total_flops / gpu_flops_per_sec
    print(f"  Step 3: GPU-seconds = {total_flops:.3e} / {gpu_flops_per_sec:.0e} = {gpu_seconds:.2e}")

    # Step 4: GPU-hours
    gpu_hours = gpu_seconds / 3600
    print(f"  Step 4: GPU-hours = {gpu_seconds:.2e} / 3600 = {gpu_hours:,.0f}")

    # Step 5: Cost
    cost = gpu_hours * gpu_cost_per_hour
    print(f"  Step 5: Cost = {gpu_hours:,.0f} * ${gpu_cost_per_hour} = ${cost:,.0f}")

    # With parallelism
    num_gpus = 8
    parallel_cost = cost  # Same total cost, just faster
    wall_clock_hours = gpu_hours / num_gpus
    print(f"\n  With {num_gpus} GPUs: ~${parallel_cost:,.0f} total, "
          f"{wall_clock_hours:,.0f} wall-clock hours")


# === Exercise 3: Over-training vs Compute-optimal Trade-off ===
# Problem: Explain why deployment orgs might over-train smaller models.

def exercise_3():
    """Solution: Over-training trade-off analysis"""
    tradeoffs = {
        "Training efficiency": {
            "Chinchilla-optimal": "Maximum",
            "Over-training": "Diminishing returns on data",
        },
        "Inference cost": {
            "Chinchilla-optimal": "Higher (larger N)",
            "Over-training": "Lower (smaller N)",
        },
        "Deployment flexibility": {
            "Chinchilla-optimal": "Less portable",
            "Over-training": "More portable / edge-friendly",
        },
        "Data requirement": {
            "Chinchilla-optimal": "Moderate",
            "Over-training": "Very large (may exhaust quality data)",
        },
    }

    print("  Why over-train a smaller model:")
    print("  - Inference cost dominates in production (millions of requests/day)")
    print("  - Train once, infer trillions of times")
    print("  - Over-training improves small model quality until it matches larger models")
    print()

    print(f"  {'Factor':<25} | {'Chinchilla-optimal':<30} | {'Over-training'}")
    print("  " + "-" * 85)
    for factor, vals in tradeoffs.items():
        print(f"  {factor:<25} | {vals['Chinchilla-optimal']:<30} | {vals['Over-training']}")

    print()
    print("  Key insight: 'optimal' depends on whether compute is measured at")
    print("  training time or total deployment lifetime.")


# === Exercise 4: Power Law Interpretation ===
# Problem: L(N, D) = E + A/N^alpha + B/D^beta, alpha~0.34, beta~0.28
# 1. Factor by which A/N^alpha decreases when doubling N
# 2. Which has stronger marginal effect: doubling N or D?

def exercise_4():
    """Solution: Power law interpretation"""
    alpha = 0.34
    beta = 0.28

    # Part 1: Effect of doubling N
    factor_N = 1 / (2 ** alpha)
    reduction_N = 1 - factor_N

    print("  Part 1: Effect of doubling N")
    print(f"    A / (2N)^alpha = A / (2^alpha * N^alpha) = (A/N^alpha) * (1/2^alpha)")
    print(f"    Reduction factor = 1 / 2^{alpha} = 1 / {2**alpha:.3f} = {factor_N:.3f}")
    print(f"    Doubling N reduces the parameter component by ~{reduction_N*100:.0f}%")
    print()

    # Part 2: Comparing doubling N vs doubling D
    factor_D = 1 / (2 ** beta)
    reduction_D = 1 - factor_D

    print("  Part 2: Doubling N vs doubling D")
    print(f"    Doubling N: component multiplied by 2^(-{alpha}) = {factor_N:.3f}")
    print(f"    Doubling D: component multiplied by 2^(-{beta}) = {factor_D:.3f}")
    print()
    print(f"    Doubling N has stronger effect ({factor_N:.3f} < {factor_D:.3f})")
    print(f"    because alpha = {alpha} > beta = {beta}")
    print()
    print("    However, the absolute impact also depends on the current")
    print("    ratio of A/N^alpha vs B/D^beta. When under-trained,")
    print("    adding data has larger absolute impact despite smaller exponent.")


# === Exercise 5: Inference Scaling Analysis ===
# Problem: Describe 3 techniques for inference-time scaling.

def exercise_5():
    """Solution: Inference-time scaling techniques"""
    techniques = [
        {
            "name": "Chain-of-Thought (CoT)",
            "mechanism": (
                "Generate explicit reasoning steps before the final answer, "
                "using token generation budget for intermediate computation"
            ),
            "best_tasks": "Multi-step math, logical deduction, word problems",
        },
        {
            "name": "Self-Consistency",
            "mechanism": (
                "Sample multiple independent reasoning paths (high temperature), "
                "then take the majority vote on the final answer"
            ),
            "best_tasks": (
                "Arithmetic, factual QA, tasks with a single correct "
                "answer where individual paths may err"
            ),
        },
        {
            "name": "Tree of Thoughts (ToT) / Beam Search",
            "mechanism": (
                "Explore a branching tree of intermediate reasoning states, "
                "evaluate each node, and prune unpromising branches"
            ),
            "best_tasks": (
                "Planning tasks, puzzles, code generation with verification, "
                "multi-hop reasoning"
            ),
        },
    ]

    for t in techniques:
        print(f"  {t['name']}:")
        print(f"    Mechanism: {t['mechanism']}")
        print(f"    Best tasks: {t['best_tasks']}")
        print()

    print("  Why inference scaling is powerful:")
    print("  - No additional training required")
    print("  - Can be applied retroactively to deployed models")
    print("  - Effective for tasks with correct reasoning chains in training data")
    print("  Limitation: Increases latency and cost per query proportionally")


if __name__ == "__main__":
    print("=== Exercise 1: Chinchilla Optimal Allocation ===")
    exercise_1()
    print("\n=== Exercise 2: Training Cost Estimation ===")
    exercise_2()
    print("\n=== Exercise 3: Over-training vs Compute-optimal ===")
    exercise_3()
    print("\n=== Exercise 4: Power Law Interpretation ===")
    exercise_4()
    print("\n=== Exercise 5: Inference Scaling Analysis ===")
    exercise_5()
    print("\nAll exercises completed!")
