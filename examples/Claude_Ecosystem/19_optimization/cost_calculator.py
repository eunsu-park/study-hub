"""
Claude API: Cost Calculator and Optimization

Demonstrates cost estimation across models with prompt caching
and batch API discount analysis.

Requirements:
    No external dependencies (pure calculation)
"""

from __future__ import annotations

from dataclasses import dataclass


# --- Model Pricing Data ---

@dataclass
class Pricing:
    """Claude model pricing (USD per million tokens)."""
    name: str
    input: float
    output: float
    cache_write: float
    cache_read: float
    batch_input: float    # 50% discount
    batch_output: float   # 50% discount


MODELS = [
    Pricing("Opus 4", 15.0, 75.0, 18.75, 1.50, 7.50, 37.50),
    Pricing("Sonnet 4", 3.0, 15.0, 3.75, 0.30, 1.50, 7.50),
    Pricing("Haiku", 0.25, 1.25, 0.30, 0.025, 0.125, 0.625),
]


# --- Cost Calculation ---

def calculate(
    model: Pricing,
    input_tokens: int,
    output_tokens: int,
    *,
    cached_tokens: int = 0,
    batch: bool = False,
) -> float:
    """Calculate cost in USD."""
    if batch:
        return (
            (input_tokens / 1e6) * model.batch_input
            + (output_tokens / 1e6) * model.batch_output
        )

    if cached_tokens > 0:
        fresh = max(0, input_tokens - cached_tokens)
        return (
            (fresh / 1e6) * model.input
            + (cached_tokens / 1e6) * model.cache_read
            + (output_tokens / 1e6) * model.output
        )

    return (
        (input_tokens / 1e6) * model.input
        + (output_tokens / 1e6) * model.output
    )


# --- Main ---

if __name__ == "__main__":
    print("Claude Cost Calculator")
    print("=" * 60)

    # Scenario: 10K input, 2K output tokens per call
    inp, out = 10_000, 2_000

    print(f"\nSingle call ({inp:,} input, {out:,} output tokens):")
    print(f"{'Model':<12} {'Standard':>10} {'Cached*':>10} {'Batch':>10}")
    print("-" * 45)
    for m in MODELS:
        std = calculate(m, inp, out)
        cached = calculate(m, inp, out, cached_tokens=8_000)
        batch = calculate(m, inp, out, batch=True)
        print(f"{m.name:<12} ${std:>9.5f} ${cached:>9.5f} ${batch:>9.5f}")
    print("* Cached: 8K of 10K input tokens from cache")

    # Monthly projection: 1000 calls/day
    daily_calls = 1000
    monthly_calls = daily_calls * 30
    print(f"\nMonthly cost ({monthly_calls:,} calls):")
    print(f"{'Model':<12} {'Standard':>12} {'Cached':>12} {'Savings':>10}")
    print("-" * 50)
    for m in MODELS:
        std_monthly = calculate(m, inp, out) * monthly_calls
        cached_monthly = calculate(m, inp, out, cached_tokens=8_000) * monthly_calls
        savings_pct = (1 - cached_monthly / std_monthly) * 100
        print(f"{m.name:<12} ${std_monthly:>11.2f} ${cached_monthly:>11.2f} "
              f"{savings_pct:>9.1f}%")
