# 01_prompt_fundamentals.py — Prompt anatomy and temperature effects
#
# Run: python 01_prompt_fundamentals.py

"""
Demonstrates the four pillars of prompt construction:
  1. Role   — who the model should act as
  2. Task   — what it should do
  3. Context — background information
  4. Format  — desired output structure

Also shows how temperature influences creativity vs determinism.
"""

import os
import anthropic


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def call_claude(prompt: str, system: str = "", temperature: float = 0.0) -> str:
    """Send a single-turn request to Claude and return the text response."""
    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


# ---------------------------------------------------------------------------
# 1. Prompt Anatomy — Role / Task / Context / Format
# ---------------------------------------------------------------------------

def demo_prompt_anatomy():
    """Build a prompt piece by piece to show each component's effect."""

    # Minimal prompt (task only)
    task_only = "Explain recursion."

    # Adding role
    role = "You are an experienced computer science tutor."

    # Adding context
    context = (
        "The student is a first-year undergraduate who has only used "
        "Python for-loops so far."
    )

    # Adding format constraint
    format_spec = (
        "Reply in exactly three bullet points: analogy, definition, "
        "and a one-line Python example."
    )

    full_prompt = f"{context}\n\n{format_spec}\n\n{task_only}"

    print("=" * 60)
    print("SECTION 1 — Prompt Anatomy")
    print("=" * 60)

    # Task only
    print("\n[Task only]")
    print(call_claude(task_only))

    # Full prompt with role + context + format
    print("\n[Role + Context + Format + Task]")
    print(call_claude(full_prompt, system=role))


# ---------------------------------------------------------------------------
# 2. Temperature Effects
# ---------------------------------------------------------------------------

def demo_temperature():
    """Run the same creative prompt at different temperatures."""

    system = "You are a creative writer."
    prompt = "Write a one-sentence opening line for a sci-fi short story."

    temperatures = [0.0, 0.5, 1.0]

    print("\n" + "=" * 60)
    print("SECTION 2 — Temperature Effects")
    print("=" * 60)

    for temp in temperatures:
        print(f"\n--- temperature={temp} ---")
        # Run twice to show variance (or lack thereof)
        for run in range(1, 3):
            result = call_claude(prompt, system=system, temperature=temp)
            print(f"  Run {run}: {result.strip()}")


# ---------------------------------------------------------------------------
# 3. Specificity Comparison
# ---------------------------------------------------------------------------

def demo_specificity():
    """Show how vague vs specific instructions change output quality."""

    vague = "Tell me about sorting."
    specific = (
        "Compare quicksort and mergesort. For each, state the average "
        "and worst-case time complexity, whether it is stable, and when "
        "you would prefer one over the other. Use a markdown table."
    )

    print("\n" + "=" * 60)
    print("SECTION 3 — Specificity Comparison")
    print("=" * 60)

    print("\n[Vague prompt]")
    print(call_claude(vague))

    print("\n[Specific prompt]")
    print(call_claude(specific))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: Set the ANTHROPIC_API_KEY environment variable first.")
        raise SystemExit(1)

    try:
        demo_prompt_anatomy()
        demo_temperature()
        demo_specificity()
    except anthropic.APIError as exc:
        print(f"\nAPI error: {exc}")
        print("Make sure your ANTHROPIC_API_KEY is valid and has quota.")
