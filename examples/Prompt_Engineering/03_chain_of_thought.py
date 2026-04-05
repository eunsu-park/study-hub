# 03_chain_of_thought.py — Chain-of-thought prompting techniques
#
# Run: python 03_chain_of_thought.py

"""
Demonstrates three chain-of-thought (CoT) strategies:
  1. Zero-shot CoT   — "Let's think step by step"
  2. Manual CoT       — Hand-crafted reasoning traces in the prompt
  3. Self-consistency — Multiple CoT samples + majority vote
"""

import os
import re
from collections import Counter

import anthropic

client: anthropic.Anthropic


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def call_claude(
    prompt: str,
    system: str = "",
    temperature: float = 0.0,
) -> str:
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


MATH_PROBLEM = (
    "A store sells notebooks for $4 each. If you buy 5 or more, you get "
    "a 20% discount on the entire purchase. Sarah buys 7 notebooks. "
    "How much does she pay?"
)


# ---------------------------------------------------------------------------
# 1. Zero-Shot CoT
# ---------------------------------------------------------------------------

def demo_zero_shot_cot():
    """Append the magic phrase to trigger step-by-step reasoning."""

    # Without CoT
    direct_prompt = f"{MATH_PROBLEM}\n\nGive only the final dollar amount."

    # With zero-shot CoT
    cot_prompt = (
        f"{MATH_PROBLEM}\n\n"
        "Let's think step by step, then state the final answer on the "
        "last line as: ANSWER: $<amount>"
    )

    print("=" * 60)
    print("SECTION 1 — Zero-Shot Chain of Thought")
    print("=" * 60)

    print("\n[Direct answer]")
    print(call_claude(direct_prompt))

    print("\n[Zero-shot CoT]")
    print(call_claude(cot_prompt))


# ---------------------------------------------------------------------------
# 2. Manual CoT (with worked examples)
# ---------------------------------------------------------------------------

def demo_manual_cot():
    """Provide an explicit reasoning trace as a few-shot example."""

    manual_example = (
        "Example:\n"
        "Q: A shop sells pens for $2. Buy 3+ and get 10% off. Tom buys 4.\n"
        "A: Step 1 — Base cost: 4 x $2 = $8.\n"
        "   Step 2 — Tom buys 4 >= 3, so the discount applies.\n"
        "   Step 3 — Discount: 10% of $8 = $0.80.\n"
        "   Step 4 — Final price: $8 - $0.80 = $7.20.\n"
        "   ANSWER: $7.20\n"
    )

    prompt = (
        f"{manual_example}\n"
        f"Now solve:\nQ: {MATH_PROBLEM}\nA:"
    )

    print("\n" + "=" * 60)
    print("SECTION 2 — Manual CoT (Worked Example)")
    print("=" * 60)
    print(call_claude(prompt))


# ---------------------------------------------------------------------------
# 3. Self-Consistency (majority vote over multiple CoT paths)
# ---------------------------------------------------------------------------

def extract_dollar_amount(text: str) -> str | None:
    """Pull the last dollar figure from a CoT response."""
    matches = re.findall(r"\$[\d]+(?:\.[\d]{1,2})?", text)
    return matches[-1] if matches else None


def demo_self_consistency(n_samples: int = 5):
    """Sample several CoT responses at higher temperature, then vote."""

    cot_prompt = (
        f"{MATH_PROBLEM}\n\n"
        "Think step by step, then state the final answer as ANSWER: $<amount>."
    )

    print("\n" + "=" * 60)
    print(f"SECTION 3 — Self-Consistency ({n_samples} samples)")
    print("=" * 60)

    answers: list[str] = []
    for i in range(n_samples):
        response = call_claude(cot_prompt, temperature=0.8)
        amount = extract_dollar_amount(response)
        answers.append(amount or "PARSE_ERROR")
        print(f"  Sample {i+1}: {amount}")

    # Majority vote
    vote = Counter(answers).most_common(1)[0]
    print(f"\n  Majority answer: {vote[0]}  (appeared {vote[1]}/{n_samples} times)")


# ---------------------------------------------------------------------------
# 4. CoT for Logical Reasoning
# ---------------------------------------------------------------------------

def demo_logic_cot():
    """Show CoT applied to a deductive logic puzzle."""

    puzzle = (
        "Five people (A, B, C, D, E) sit in a row.\n"
        "- B sits immediately to the right of A.\n"
        "- C does not sit next to D.\n"
        "- E sits at one of the two ends.\n"
        "- D sits in the middle (position 3).\n"
        "List all valid seating arrangements."
    )

    prompt = (
        f"{puzzle}\n\n"
        "Solve step by step. First list constraints, then enumerate "
        "possibilities, eliminate invalid ones, and show the result."
    )

    print("\n" + "=" * 60)
    print("SECTION 4 — CoT for Logical Reasoning")
    print("=" * 60)
    print(call_claude(prompt))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: Set the ANTHROPIC_API_KEY environment variable first.")
        raise SystemExit(1)

    client = anthropic.Anthropic()

    try:
        demo_zero_shot_cot()
        demo_manual_cot()
        demo_self_consistency()
        demo_logic_cot()
    except anthropic.APIError as exc:
        print(f"\nAPI error: {exc}")
