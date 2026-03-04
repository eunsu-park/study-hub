"""
Exercises for Lesson 19: Models, Pricing, and Optimization
Topic: Claude_Ecosystem

Solutions to practice problems from the lesson.
"""

from dataclasses import dataclass
from typing import Any


# === Exercise 1: Cost Calculator ===
# Problem: Calculate API costs across models with support for
#   standard, prompt caching, and batch API pricing.

@dataclass
class ModelPricing:
    """Pricing for a Claude model (per million tokens)."""
    name: str
    input_per_mtok: float
    output_per_mtok: float
    cache_write_per_mtok: float    # prompt caching: write
    cache_read_per_mtok: float     # prompt caching: read
    batch_input_per_mtok: float    # batch API: 50% discount
    batch_output_per_mtok: float


PRICING = [
    ModelPricing("Claude Opus 4",
                 15.0, 75.0, 18.75, 1.50, 7.50, 37.50),
    ModelPricing("Claude Sonnet 4",
                 3.0, 15.0, 3.75, 0.30, 1.50, 7.50),
    ModelPricing("Claude Haiku",
                 0.25, 1.25, 0.30, 0.025, 0.125, 0.625),
]


def calculate_cost(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    mode: str = "standard",
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> dict[str, Any]:
    """Calculate API cost for a given usage pattern.

    Modes: "standard", "cached", "batch"
    """
    pricing = next((p for p in PRICING if p.name == model_name), None)
    if not pricing:
        return {"error": f"Unknown model: {model_name}"}

    if mode == "batch":
        input_cost = (input_tokens / 1e6) * pricing.batch_input_per_mtok
        output_cost = (output_tokens / 1e6) * pricing.batch_output_per_mtok
        cache_cost = 0.0
    elif mode == "cached":
        fresh_input = max(0, input_tokens - cache_read_tokens)
        input_cost = (fresh_input / 1e6) * pricing.input_per_mtok
        output_cost = (output_tokens / 1e6) * pricing.output_per_mtok
        cache_cost = (
            (cache_write_tokens / 1e6) * pricing.cache_write_per_mtok
            + (cache_read_tokens / 1e6) * pricing.cache_read_per_mtok
        )
    else:
        input_cost = (input_tokens / 1e6) * pricing.input_per_mtok
        output_cost = (output_tokens / 1e6) * pricing.output_per_mtok
        cache_cost = 0.0

    total = input_cost + output_cost + cache_cost
    return {
        "model": model_name,
        "mode": mode,
        "input_cost": round(input_cost, 4),
        "output_cost": round(output_cost, 4),
        "cache_cost": round(cache_cost, 4),
        "total_usd": round(total, 4),
    }


def exercise_1():
    """Compare costs across models and modes."""
    scenarios = [
        ("Standard", "standard", 10_000, 2_000, 0, 0),
        ("Cached", "cached", 10_000, 2_000, 8_000, 8_000),
        ("Batch", "batch", 10_000, 2_000, 0, 0),
    ]
    for model in PRICING:
        print(f"  {model.name}:")
        for label, mode, inp, out, cr, cw in scenarios:
            result = calculate_cost(model.name, inp, out, mode, cr, cw)
            print(f"    {label:>8}: ${result['total_usd']:.4f}")
        print()


# === Exercise 2: Model Selection Optimizer ===
# Problem: Given task requirements, recommend the most cost-effective
#   model that meets quality thresholds.

@dataclass
class TaskRequirements:
    """Requirements for a task."""
    description: str
    min_quality: str       # "high", "medium", "low"
    max_latency: str       # "fast", "moderate", "slow"
    budget_per_call: float  # max USD per API call
    estimated_input: int
    estimated_output: int


QUALITY_MAP = {"high": ["Claude Opus 4"],
               "medium": ["Claude Opus 4", "Claude Sonnet 4"],
               "low": ["Claude Opus 4", "Claude Sonnet 4", "Claude Haiku"]}

LATENCY_MAP = {"fast": ["Claude Haiku", "Claude Sonnet 4"],
               "moderate": ["Claude Haiku", "Claude Sonnet 4", "Claude Opus 4"],
               "slow": ["Claude Haiku", "Claude Sonnet 4", "Claude Opus 4"]}


def recommend_model(req: TaskRequirements) -> dict[str, Any]:
    """Recommend the best model for given requirements."""
    quality_models = set(QUALITY_MAP.get(req.min_quality, []))
    latency_models = set(LATENCY_MAP.get(req.max_latency, []))
    candidates = quality_models & latency_models

    if not candidates:
        return {"recommendation": None, "reason": "No model meets all constraints"}

    # Score by cost-effectiveness
    results: list[dict[str, Any]] = []
    for model_name in candidates:
        cost = calculate_cost(model_name, req.estimated_input,
                              req.estimated_output)
        if cost["total_usd"] <= req.budget_per_call:
            results.append({"model": model_name, "cost": cost["total_usd"]})

    if not results:
        return {"recommendation": None, "reason": "All models exceed budget"}

    best = min(results, key=lambda r: r["cost"])
    return {
        "recommendation": best["model"],
        "cost_per_call": best["cost"],
        "alternatives": [r for r in results if r["model"] != best["model"]],
    }


def exercise_2():
    """Demonstrate model selection optimization."""
    tasks = [
        TaskRequirements("Code review", "high", "moderate", 0.10, 5000, 1000),
        TaskRequirements("Log classification", "low", "fast", 0.01, 500, 50),
        TaskRequirements("Architecture design", "high", "slow", 1.00, 20000, 5000),
        TaskRequirements("Chat assistant", "medium", "fast", 0.05, 2000, 500),
    ]
    for task in tasks:
        result = recommend_model(task)
        rec = result.get("recommendation", "None")
        cost = result.get("cost_per_call", "N/A")
        print(f"  {task.description}: {rec} "
              f"(${cost}/call, budget: ${task.budget_per_call})")


# === Exercise 3: Prompt Caching Strategy Analyzer ===
# Problem: Determine when prompt caching provides cost savings
#   based on usage patterns.

def analyze_caching_benefit(
    model_name: str,
    system_prompt_tokens: int,
    calls_per_hour: int,
    output_tokens_per_call: int,
    cache_ttl_minutes: int = 5,
) -> dict[str, Any]:
    """Analyze whether prompt caching saves money."""
    calls_per_cache = max(1, int(calls_per_hour * (cache_ttl_minutes / 60)))

    # Without caching: full input every call
    no_cache = calculate_cost(
        model_name, system_prompt_tokens, output_tokens_per_call)
    no_cache_hourly = no_cache["total_usd"] * calls_per_hour

    # With caching: 1 cache write + (N-1) cache reads per TTL window
    windows_per_hour = max(1, 60 // cache_ttl_minutes)
    write_cost = calculate_cost(
        model_name, 0, output_tokens_per_call, "cached",
        cache_write_tokens=system_prompt_tokens)
    read_cost = calculate_cost(
        model_name, system_prompt_tokens, output_tokens_per_call, "cached",
        cache_read_tokens=system_prompt_tokens)

    cached_hourly = (
        write_cost["total_usd"] * windows_per_hour
        + read_cost["total_usd"] * (calls_per_hour - windows_per_hour)
    )
    savings = no_cache_hourly - cached_hourly
    savings_pct = (savings / max(no_cache_hourly, 0.0001)) * 100

    return {
        "model": model_name,
        "hourly_without_cache": round(no_cache_hourly, 4),
        "hourly_with_cache": round(cached_hourly, 4),
        "hourly_savings": round(savings, 4),
        "savings_pct": round(savings_pct, 1),
        "recommendation": "Use caching" if savings > 0 else "Skip caching",
    }


def exercise_3():
    """Analyze prompt caching benefits across scenarios."""
    scenarios = [
        ("High-volume chatbot", "Claude Sonnet 4", 5000, 100, 500),
        ("Occasional analysis", "Claude Opus 4", 10000, 5, 2000),
        ("Batch processing", "Claude Haiku", 1000, 200, 100),
    ]
    for label, model, sys_tokens, calls, output in scenarios:
        result = analyze_caching_benefit(model, sys_tokens, calls, output)
        print(f"  {label} ({model}):")
        print(f"    Without cache: ${result['hourly_without_cache']:.4f}/hr")
        print(f"    With cache:    ${result['hourly_with_cache']:.4f}/hr")
        print(f"    Savings:       ${result['hourly_savings']:.4f}/hr "
              f"({result['savings_pct']}%)")
        print(f"    → {result['recommendation']}\n")


if __name__ == "__main__":
    print("=== Exercise 1: Cost Calculator ===")
    exercise_1()

    print("=== Exercise 2: Model Selection ===")
    exercise_2()

    print("\n=== Exercise 3: Caching Analysis ===")
    exercise_3()

    print("All exercises completed!")
