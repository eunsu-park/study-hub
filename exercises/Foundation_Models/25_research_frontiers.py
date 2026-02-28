"""
Exercises for Lesson 25: Research Frontiers
Topic: Foundation_Models

Solutions to practice problems from the lesson.
"""


# === Exercise 1: Test-Time Compute Scaling Trade-offs ===
# Problem: Compare training-time vs test-time compute scaling for 3 scenarios.

def exercise_1():
    """Solution: Test-time compute scaling trade-offs"""
    scenarios = [
        {
            "scenario": "A) Answering 1M customer support queries per day",
            "training_time": (
                "Larger model handles more query types with fixed latency. "
                "Per-query cost is fixed and predictable."
            ),
            "test_time": (
                "Higher quality on difficult queries, but adds latency "
                "and cost per query (10-50x compute multiplier)."
            ),
            "better": "Training-time scaling",
            "reason": (
                "High volume (1M/day) and predictable query types. "
                "Each query needs fast, consistent responses. "
                "Test-time compute at 10-50x per query is cost-prohibitive "
                "at this scale. A larger/better-trained model gives "
                "consistent quality at fixed marginal cost."
            ),
        },
        {
            "scenario": "B) Solving a novel mathematics competition problem",
            "training_time": (
                "Better general mathematical knowledge baked into weights. "
                "Fixed inference cost but may miss novel solution paths."
            ),
            "test_time": (
                "Generate multiple solution paths, verify each, "
                "backtrack on errors. Mimics how humans solve hard problems."
            ),
            "better": "Test-time compute scaling",
            "reason": (
                "Novel competition problems require exploring multiple "
                "approaches before finding the right one. A human "
                "mathematician doesn't solve IMO problems in 2 seconds -- "
                "they think for hours. Volume is low (1 problem), so "
                "high per-problem cost is acceptable. Spend 10-100x more "
                "tokens on reasoning chains."
            ),
        },
        {
            "scenario": "C) Real-time coding autocomplete (< 50ms latency)",
            "training_time": (
                "Model must know common code patterns at inference time. "
                "Well-trained small model (< 1B params) can meet latency."
            ),
            "test_time": (
                "More thinking means more forward passes means more latency. "
                "Violates the hard 50ms requirement."
            ),
            "better": "Training-time scaling",
            "reason": (
                "Hard latency constraint of 50ms makes test-time compute "
                "infeasible (requires many additional forward passes). "
                "Use a well-trained small model with high accuracy on common "
                "code patterns from training. Speculative decoding can help "
                "speed, but the thinking budget must be near-zero."
            ),
        },
    ]

    print(f"  {'Scenario':<55} {'Better Choice'}")
    print("  " + "-" * 75)
    for s in scenarios:
        print(f"  {s['scenario']:<55} {s['better']}")
    print()

    for s in scenarios:
        print(f"  {s['scenario']}:")
        print(f"    Training-time: {s['training_time']}")
        print(f"    Test-time:     {s['test_time']}")
        print(f"    Better choice: {s['better']}")
        print(f"    Reason: {s['reason']}")
        print()

    print("  Key insight: Test-time compute is most valuable for problems that are")
    print("  RARE, DIFFICULT, and ALLOW SPENDING MORE TIME on individual instances.")
    print("  Training-time scaling is better for HIGH-VOLUME, LATENCY-SENSITIVE,")
    print("  or COST-CONSTRAINED applications.")


# === Exercise 2: Synthetic Data Model Collapse Risk ===
# Problem: Explain model collapse and describe 2 prevention strategies.

def exercise_2():
    """Solution: Synthetic data model collapse analysis"""
    print("  Model collapse mechanism:")
    print()
    print("  Round 1 (human data):   Distribution covers {common, uncommon, rare}")
    print("  Round 2 (v1 synthetic): Drops some 'rare' patterns (low prob -> rarely sampled)")
    print("  Round 3 (v2 synthetic): Further shrinks distribution, amplifies v1's biases")
    print("  Round N:                Distribution collapses to repetitive, high-prob outputs")
    print()

    print("  At each round, the model generates synthetic data from its own distribution.")
    print("  Each generation step introduces subtle biases and loses tail-distribution")
    print("  examples. Training on this narrowed distribution amplifies the bias further.")
    print()

    print("  Symptoms of collapse:")
    print("    - Reduced vocabulary diversity")
    print("    - Repetitive stylistic patterns")
    print("    - Loss of factual edge cases")
    print("    - Overconfident on uncertain questions")
    print("      (model v1's confident wrong answers propagate)")
    print()

    # Prevention Strategy 1: Human data replay
    print("  Prevention Strategy 1: Human Data Replay")
    print("  -" * 25)
    print()

    import random
    random.seed(42)

    # Simulate the strategy
    human_data = [f"human_example_{i}" for i in range(100)]
    synthetic_data = [f"synthetic_example_{i}" for i in range(900)]

    replay_ratio = 0.10
    human_sample_size = int(len(synthetic_data) * replay_ratio)
    human_sample = random.sample(human_data, human_sample_size)
    combined = synthetic_data + human_sample

    print(f"    Synthetic data:      {len(synthetic_data)} examples")
    print(f"    Human replay ratio:  {replay_ratio:.0%}")
    print(f"    Human samples added: {human_sample_size}")
    print(f"    Combined training:   {len(combined)} examples")
    print()
    print("    In every batch: 10% human data + 90% synthetic.")
    print("    Human data anchors the distribution to the real world.")
    print("    Prevents drift toward narrow synthetic distribution.")
    print()

    # Prevention Strategy 2: Distribution divergence monitoring
    print("  Prevention Strategy 2: Distribution Divergence Monitoring")
    print("  -" * 25)
    print()

    # Simulate diversity measurement
    def measure_diversity(tokens):
        """Vocabulary diversity = unique tokens / total tokens."""
        return len(set(tokens)) / len(tokens) if tokens else 0

    # Simulated model outputs across generations
    # Each generation loses some diversity
    gen_0_tokens = list(range(1000))  # Human data: 1000 unique tokens
    random.shuffle(gen_0_tokens)

    gen_1_tokens = random.choices(gen_0_tokens[:800], k=1000)  # v1: fewer unique
    gen_2_tokens = random.choices(gen_0_tokens[:500], k=1000)  # v2: even fewer
    gen_3_tokens = random.choices(gen_0_tokens[:200], k=1000)  # v3: collapsing

    generations = [
        ("Human (baseline)", gen_0_tokens),
        ("Model v1", gen_1_tokens),
        ("Model v2", gen_2_tokens),
        ("Model v3", gen_3_tokens),
    ]

    print("    Vocabulary diversity across synthetic generations:")
    prev_diversity = None
    for name, tokens in generations:
        diversity = measure_diversity(tokens)
        drop = ""
        if prev_diversity is not None:
            pct_change = (diversity - prev_diversity) / prev_diversity * 100
            drop = f"  ({pct_change:+.1f}%)"
            if pct_change < -10:
                drop += " ** COLLAPSE WARNING **"
        print(f"      {name:<20} diversity = {diversity:.3f}{drop}")
        prev_diversity = diversity

    print()
    print("    Detection rule: Alert if diversity drops >10% between generations.")
    print("    Also monitor perplexity on held-out human text.")
    print("    Increasing perplexity = model drifting from human distribution.")


# === Exercise 3: Multi-Agent Debate vs Single Model ===
# Problem: Evaluate when debate helps vs hurts for 3 task types.

def exercise_3():
    """Solution: Multi-agent debate analysis"""
    tasks = [
        {
            "task": "A) Code correctness with subtle off-by-one bug",
            "debate_helps": True,
            "analysis": (
                "Different agents independently review the code and may notice "
                "the off-by-one error from different angles:\n"
                "      Agent 1: 'Loop uses < but should use <= because...'\n"
                "      Agent 2: 'Initialization at i=1 misses first element...'\n"
                "      Agent 3: 'When array length is 0, this crashes...'\n"
                "    The adversarial structure forces each agent to look for flaws. "
                "A single model may confidently say 'looks correct' (sycophancy "
                "toward implicit assumption that code is correct). Multiple agents "
                "with independent review are more likely to surface the bug."
            ),
        },
        {
            "task": "B) 'What is the capital of France?'",
            "debate_helps": False,
            "analysis": (
                "'Paris' is factual with no ambiguity. If agents debate:\n"
                "      Agent 1: 'Paris'\n"
                "      Agent 2: 'Paris'\n"
                "      Agent 3: 'Paris, but historically Versailles under Louis XIV...'\n"
                "    Debate adds latency, cost, and potential for spurious "
                "'interesting' qualifications that confuse the final answer. "
                "For factual, closed-domain questions with deterministic answers, "
                "a single model with high confidence is better."
            ),
        },
        {
            "task": "C) 'Should a company prioritize profit or employee well-being?'",
            "debate_helps": True,
            "analysis": (
                "Genuinely contested question with multiple valid perspectives. "
                "Debate surfaces:\n"
                "      - Economic efficiency (profit enables reinvestment)\n"
                "      - Social contract (employees are stakeholders)\n"
                "      - Empirical evidence (high satisfaction -> outperformance)\n"
                "    RISK: If agents are assigned fixed 'sides', they may generate "
                "persuasive but one-sided arguments that amplify polarization "
                "rather than reaching nuanced synthesis. Better to have agents "
                "generate multiple independent perspectives without fixed assignments."
            ),
        },
    ]

    for t in tasks:
        verdict = "HELPS" if t["debate_helps"] else "HURTS (or wastes resources)"
        print(f"  {t['task']}")
        print(f"    Debate {verdict}")
        print(f"    Analysis: {t['analysis']}")
        print()

    print("  General rule for when debate helps:")
    print("    HELPS:  Reasoning problems (cross-checkable paths),")
    print("            adversarial tasks (bug finding, security review),")
    print("            value-laden questions (multiple perspectives add nuance)")
    print()
    print("    HURTS:  Factual lookups, simple computations,")
    print("            consensus tasks where all models agree anyway")
    print()
    print("    Debate adds cost. Only use when the diversity of perspectives")
    print("    genuinely improves the answer beyond what a single model provides.")


# === Exercise 4: World Models for Physical Reasoning ===
# Problem: Explain a fundamental limitation of video-based world models.

def exercise_4():
    """Solution: World model limitations analysis"""
    print("  Fundamental limitation: Lack of causal grounding")
    print("  (correlation vs. causation)")
    print()

    print("  What video world models learn:")
    print("    Statistical correlations in pixel space:")
    print("    'When I see this frame, the next frame typically looks like this.'")
    print()
    print("  What they do NOT learn:")
    print("    Causal physical laws (Newton's laws, conservation of energy,")
    print("    fluid dynamics equations) that GENERATE those correlations.")
    print()

    print("  Concrete example:")
    print("    A model trained on videos of balls rolling down ramps learns:")
    print("    'ball at top -> ball rolling -> ball at bottom'")
    print("    by memorizing the visual trajectory pattern,")
    print("    NOT by modeling force = mass * acceleration.")
    print()

    print("  Failure modes:")
    failures = [
        (
            "Counterfactual reasoning",
            "'If the ball was twice as heavy, how would it roll?' -- "
            "model can't answer because it never learned mass-acceleration "
            "relationships, only visual patterns."
        ),
        (
            "Novel physics configurations",
            "A ramp shape never seen in training may generate physically "
            "impossible trajectories (ball passing through the ramp surface)."
        ),
        (
            "Long-horizon extrapolation",
            "Even if short-term prediction is accurate, extrapolating 10+ "
            "seconds ahead accumulates errors because small deviations from "
            "learned distribution compound without physical law constraints."
        ),
    ]

    for name, desc in failures:
        print(f"    {name}:")
        print(f"      {desc}")
        print()

    print("  Training data needed to address this:")
    print()
    data_types = [
        (
            "1. Labeled physics simulations",
            "Videos from game engines paired with ground-truth physical "
            "quantities (velocity, force, energy at each frame). Model "
            "learns to predict these quantities, not just pixel values."
        ),
        (
            "2. Interventional (counterfactual) data",
            "Same physical setup with different object masses, friction, etc. "
            "E.g., Ball A (1kg) and Ball B (2kg) on same ramp. Teaches "
            "the model actual causal structure rather than single-trajectory "
            "pattern matching."
        ),
        (
            "3. Long-horizon physics benchmarks",
            "Evaluation datasets testing 30+ second extrapolations, forcing "
            "models to learn stable physical representations rather than "
            "short-term pattern matching."
        ),
    ]

    for name, desc in data_types:
        print(f"    {name}:")
        print(f"      {desc}")
        print()

    print("  Key insight: Visual world models are excellent for PREDICTIVE tasks")
    print("  (what will happen next?) but insufficient for CAUSAL tasks")
    print("  (why did it happen, and what if it were different?).")


if __name__ == "__main__":
    print("=== Exercise 1: Test-Time Compute Scaling ===")
    exercise_1()
    print("\n=== Exercise 2: Synthetic Data Model Collapse ===")
    exercise_2()
    print("\n=== Exercise 3: Multi-Agent Debate vs Single Model ===")
    exercise_3()
    print("\n=== Exercise 4: World Models for Physical Reasoning ===")
    exercise_4()
    print("\nAll exercises completed!")
