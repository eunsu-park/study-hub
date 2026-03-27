"""
Exercises for Lesson 24: API and Evaluation
Topic: Foundation_Models

Solutions to practice problems from the lesson.
"""


# === Exercise 1: API Cost Calculation and Model Selection ===
# Problem: Calculate daily cost for document summarization service.

def exercise_1():
    """Solution: API cost calculation and routing"""
    requests_per_day = 10_000
    input_tokens = 2_000
    output_tokens = 300

    # GPT-4o pricing
    gpt4o_input_price = 5.00 / 1_000_000
    gpt4o_output_price = 15.00 / 1_000_000

    # Claude 3 Haiku pricing
    haiku_input_price = 0.25 / 1_000_000
    haiku_output_price = 1.25 / 1_000_000

    # Part A: Daily costs
    gpt4o_input_cost = requests_per_day * input_tokens * gpt4o_input_price
    gpt4o_output_cost = requests_per_day * output_tokens * gpt4o_output_price
    gpt4o_total = gpt4o_input_cost + gpt4o_output_cost

    haiku_input_cost = requests_per_day * input_tokens * haiku_input_price
    haiku_output_cost = requests_per_day * output_tokens * haiku_output_price
    haiku_total = haiku_input_cost + haiku_output_cost

    print("  A) Daily cost at 10,000 requests:")
    print(f"    GPT-4o:  input ${gpt4o_input_cost:.2f} + output ${gpt4o_output_cost:.2f} = ${gpt4o_total:.2f}/day")
    print(f"    Haiku:   input ${haiku_input_cost:.2f} + output ${haiku_output_cost:.2f} = ${haiku_total:.2f}/day")
    print(f"    Savings: ${gpt4o_total - haiku_total:.2f}/day")
    print()

    # Part B: Volume threshold for >$100/day savings
    gpt4o_per_req = input_tokens * gpt4o_input_price + output_tokens * gpt4o_output_price
    haiku_per_req = input_tokens * haiku_input_price + output_tokens * haiku_output_price
    savings_per_req = gpt4o_per_req - haiku_per_req
    min_requests = 100.0 / savings_per_req

    print(f"  B) Volume threshold for >$100/day savings:")
    print(f"    Per-request cost: GPT-4o ${gpt4o_per_req:.4f}, Haiku ${haiku_per_req:.6f}")
    print(f"    Savings/request: ${savings_per_req:.6f}")
    print(f"    Minimum requests: ${100:.0f} / ${savings_per_req:.6f} = {min_requests:,.0f} requests/day")
    print()

    # Part C: Routing strategy
    complex_pct = 0.20
    complex_count = int(requests_per_day * complex_pct)
    simple_count = requests_per_day - complex_count

    # Sonnet pricing for complex
    sonnet_input_price = 3.00 / 1_000_000
    sonnet_output_price = 15.00 / 1_000_000

    complex_cost = complex_count * (
        input_tokens * sonnet_input_price + output_tokens * sonnet_output_price
    )
    simple_cost = simple_count * (
        input_tokens * haiku_input_price + output_tokens * haiku_output_price
    )
    routed_total = complex_cost + simple_cost

    print("  C) Routing strategy (20% complex, 80% simple):")
    print(f"    {complex_count:,} complex -> Sonnet: ${complex_cost:.2f}/day")
    print(f"    {simple_count:,} simple -> Haiku: ${simple_cost:.2f}/day")
    print(f"    Total with routing: ${routed_total:.2f}/day")
    print(f"    vs all-GPT-4o: ${gpt4o_total:.2f}/day ({(1-routed_total/gpt4o_total)*100:.0f}% reduction)")


# === Exercise 2: LLM-as-a-Judge Bias Analysis ===
# Problem: Identify 3 biases in LLM judge evaluations.

def exercise_2():
    """Solution: LLM-as-a-Judge bias analysis"""
    biases = [
        {
            "name": "Position bias (order bias)",
            "description": (
                "LLM judge favors response that appears first (or last) "
                "regardless of quality."
            ),
            "example": (
                "Same judge scores A=7,B=5 in one order, but A=5,B=7 reversed."
            ),
            "mitigation": (
                "Evaluate both orderings (A vs B AND B vs A) and average. "
                "Treat inconsistent results as ties (positional debiasing)."
            ),
        },
        {
            "name": "Verbosity bias (length preference)",
            "description": (
                "LLM judges rate longer, more detailed responses as higher "
                "quality, even when concise answer is more appropriate."
            ),
            "example": (
                "500-word answer scored higher than 50-word answer "
                "even when 50-word is more accurate."
            ),
            "mitigation": (
                "Add scoring criteria penalizing unnecessary length. "
                "Provide reference answer length as calibration anchor."
            ),
        },
        {
            "name": "Self-enhancement bias",
            "description": (
                "If GPT-4 judges GPT-4 vs Claude, it may systematically "
                "favor its own model family's style."
            ),
            "example": (
                "GPT-4 judge scores GPT-4o +0.3 points higher than "
                "equally good Claude responses on average."
            ),
            "mitigation": (
                "Use multiple judges from different model families and "
                "ensemble scores. Include at least one judge from a different "
                "provider than the evaluated model."
            ),
        },
    ]

    for b in biases:
        print(f"  Bias: {b['name']}")
        print(f"    Description: {b['description']}")
        print(f"    Example: {b['example']}")
        print(f"    Mitigation: {b['mitigation']}")
        print()


# === Exercise 3: Benchmark Selection ===
# Problem: Choose benchmarks for 3 product deployments.

def exercise_3():
    """Solution: Benchmark selection for specific use cases"""
    deployments = [
        {
            "deployment": "A) AI coding assistant for Python developers",
            "benchmark": "HumanEval",
            "guarantees": (
                "Generates syntactically correct, functionally complete "
                "Python functions for common algorithmic tasks"
            ),
            "not_guaranteed": (
                "Domain-specific code (finance, medical APIs), debugging, "
                "code review quality, long functions (100+ lines)"
            ),
        },
        {
            "deployment": "B) Customer service chatbot for multi-turn support",
            "benchmark": "MT-Bench",
            "guarantees": (
                "Multi-turn dialogue coherence, context maintenance, "
                "handling follow-up questions"
            ),
            "not_guaranteed": (
                "Company policy adherence, customer satisfaction, "
                "domain-specific vocabulary, ambiguous real-world requests"
            ),
        },
        {
            "deployment": "C) Financial analysis assistant",
            "benchmark": "GSM8K + MATH",
            "guarantees": (
                "Accurate arithmetic, algebraic manipulation, "
                "structured mathematical reasoning, word problems"
            ),
            "not_guaranteed": (
                "Domain-specific financial knowledge (CAPM, Black-Scholes), "
                "regulatory compliance, real financial data formats, "
                "detecting data inconsistencies"
            ),
        },
    ]

    for d in deployments:
        print(f"  {d['deployment']}")
        print(f"    Best benchmark: {d['benchmark']}")
        print(f"    Guarantees: {d['guarantees']}")
        print(f"    Does NOT guarantee: {d['not_guaranteed']}")
        print()

    print("  Key insight: Every benchmark evaluates a PROXY for real-world capability.")
    print("  Always supplement benchmarks with domain-specific evaluation sets.")


# === Exercise 4: Evaluation Pipeline Design ===
# Problem: Design evaluation pipeline for legal document Q&A system.

def exercise_4():
    """Solution: Legal Q&A evaluation pipeline design"""
    print("  Evaluation dimensions and metrics:")
    print()

    dimensions = [
        ("Factual accuracy", "LLM-as-Judge with references", "1-5 score", ">= 4.0 avg"),
        ("Citation correctness", "Rule-based: check cited clause exists", "Precision/Recall", "Precision >= 90%"),
        ("Completeness", "LLM-as-Judge: all parts addressed?", "Binary + score", ">= 80% complete"),
        ("Safety", "Keyword + LLM classifier for definitive advice", "Binary flag", "0% definitive advice"),
        ("Fluency", "Automated perplexity + grammar", "Score 1-5", ">= 4.0"),
    ]

    print(f"  {'Dimension':<22} {'Method':<40} {'Metric':<17} {'Threshold'}")
    print("  " + "-" * 100)
    for dim, method, metric, threshold in dimensions:
        print(f"  {dim:<22} {method:<40} {metric:<17} {threshold}")

    print()
    print("  Offline (development):")
    print("    - Golden test set: 200 human-curated legal Q&A pairs")
    print("    - Run after every model update")
    print("    - Gate: all 5 dimensions must pass thresholds")
    print()
    print("  Online (production):")
    print("    - Sample 1% of live queries for human review")
    print("    - Track: user satisfaction, escalation rate, refusal rate")
    print("    - Alert if accuracy drops >5% below baseline")
    print()
    print("  Critical safety rule:")
    print("    Any answer construed as legal advice MUST include disclaimer:")
    print("    'This is not legal advice. Consult a licensed attorney.'")


if __name__ == "__main__":
    print("=== Exercise 1: API Cost Calculation ===")
    exercise_1()
    print("\n=== Exercise 2: LLM-as-a-Judge Bias ===")
    exercise_2()
    print("\n=== Exercise 3: Benchmark Selection ===")
    exercise_3()
    print("\n=== Exercise 4: Evaluation Pipeline Design ===")
    exercise_4()
    print("\nAll exercises completed!")
