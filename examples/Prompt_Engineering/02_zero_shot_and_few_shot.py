# 02_zero_shot_and_few_shot.py — Zero-shot vs few-shot classification
#
# Run: python 02_zero_shot_and_few_shot.py

"""
Demonstrates:
  1. Zero-shot classification — the model classifies without examples
  2. Few-shot classification  — examples guide the model's behavior
  3. Dynamic example selection — pick the most relevant examples at runtime
"""

import os
from typing import List, Tuple

import anthropic

client: anthropic.Anthropic  # initialized in main


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def call_claude(prompt: str, system: str = "") -> str:
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        temperature=0.0,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


# ---------------------------------------------------------------------------
# 1. Zero-Shot Classification
# ---------------------------------------------------------------------------

CATEGORIES = ["Bug Report", "Feature Request", "General Question", "Praise"]

def zero_shot_classify(text: str) -> str:
    """Classify a support ticket with no examples provided."""
    prompt = (
        f"Classify the following support ticket into exactly one category.\n"
        f"Categories: {', '.join(CATEGORIES)}\n\n"
        f"Ticket: \"{text}\"\n\n"
        f"Reply with ONLY the category name."
    )
    return call_claude(prompt)


def demo_zero_shot():
    tickets = [
        "The app crashes every time I open the settings page.",
        "It would be awesome if you added dark mode!",
        "How do I reset my password?",
        "Love this product — best purchase I've made this year!",
    ]

    print("=" * 60)
    print("SECTION 1 — Zero-Shot Classification")
    print("=" * 60)
    for ticket in tickets:
        label = zero_shot_classify(ticket)
        print(f"  [{label}] {ticket}")


# ---------------------------------------------------------------------------
# 2. Few-Shot Classification
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES = [
    ("My screen goes blank after login.",                  "Bug Report"),
    ("Can you add calendar integration?",                  "Feature Request"),
    ("What file formats do you support?",                  "General Question"),
    ("The new update is fantastic, keep it up!",           "Praise"),
    ("Export to PDF always produces an empty file.",        "Bug Report"),
    ("Please support multiple languages in the UI.",       "Feature Request"),
]


def few_shot_classify(text: str) -> str:
    """Classify a support ticket using hand-picked examples."""
    examples_block = "\n".join(
        f'  Ticket: "{t}"\n  Category: {c}' for t, c in FEW_SHOT_EXAMPLES
    )
    prompt = (
        f"Classify the support ticket into one category.\n"
        f"Categories: {', '.join(CATEGORIES)}\n\n"
        f"Examples:\n{examples_block}\n\n"
        f'Now classify:\n  Ticket: "{text}"\n  Category:'
    )
    return call_claude(prompt)


def demo_few_shot():
    ambiguous_tickets = [
        "I wish the export didn't break — also, could you add .docx support?",
        "Is there a way to undo? I accidentally deleted my project.",
        "Just wanted to say thanks for fixing the sync issue so quickly.",
    ]

    print("\n" + "=" * 60)
    print("SECTION 2 — Few-Shot Classification")
    print("=" * 60)
    for ticket in ambiguous_tickets:
        label = few_shot_classify(ticket)
        print(f"  [{label}] {ticket}")


# ---------------------------------------------------------------------------
# 3. Dynamic Example Selection
# ---------------------------------------------------------------------------

def keyword_overlap(a: str, b: str) -> int:
    """Simple word-overlap similarity for demonstration."""
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    return len(set_a & set_b)


def select_examples(
    query: str,
    pool: List[Tuple[str, str]],
    k: int = 3,
) -> List[Tuple[str, str]]:
    """Return the k most relevant examples from the pool."""
    scored = [(keyword_overlap(query, ex[0]), ex) for ex in pool]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [ex for _, ex in scored[:k]]


def dynamic_few_shot_classify(text: str) -> str:
    """Pick the best examples dynamically, then classify."""
    chosen = select_examples(text, FEW_SHOT_EXAMPLES, k=3)
    examples_block = "\n".join(
        f'  Ticket: "{t}"\n  Category: {c}' for t, c in chosen
    )
    prompt = (
        f"Classify the support ticket into one category.\n"
        f"Categories: {', '.join(CATEGORIES)}\n\n"
        f"Examples:\n{examples_block}\n\n"
        f'Now classify:\n  Ticket: "{text}"\n  Category:'
    )
    return call_claude(prompt)


def demo_dynamic_selection():
    text = "Exporting to PDF gives a blank page on my machine."

    print("\n" + "=" * 60)
    print("SECTION 3 — Dynamic Example Selection")
    print("=" * 60)

    chosen = select_examples(text, FEW_SHOT_EXAMPLES, k=3)
    print("  Selected examples for the query:")
    for t, c in chosen:
        print(f"    [{c}] {t}")

    label = dynamic_few_shot_classify(text)
    print(f"\n  Prediction: [{label}] {text}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: Set the ANTHROPIC_API_KEY environment variable first.")
        raise SystemExit(1)

    client = anthropic.Anthropic()

    try:
        demo_zero_shot()
        demo_few_shot()
        demo_dynamic_selection()
    except anthropic.APIError as exc:
        print(f"\nAPI error: {exc}")
