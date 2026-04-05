# 04_advanced_reasoning.py — Tree of Thoughts, Self-Refine, meta-prompting
#
# Run: python 04_advanced_reasoning.py

"""
Demonstrates three advanced reasoning frameworks:
  1. Tree of Thoughts (ToT)  — explore multiple reasoning branches
  2. Self-Refine             — iterative critique and revision loop
  3. Meta-Prompting          — use the LLM to generate its own prompt
"""

import os
import textwrap

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


# ---------------------------------------------------------------------------
# 1. Tree of Thoughts (ToT)
# ---------------------------------------------------------------------------

def demo_tree_of_thoughts():
    """Simulate ToT by generating multiple candidate approaches, evaluating
    each, then selecting the best path to continue."""

    problem = (
        "Design an algorithm that finds the shortest path in a weighted "
        "graph where edge weights can change dynamically during traversal."
    )

    # Step 1 — Generate candidate approaches
    gen_prompt = (
        f"Problem: {problem}\n\n"
        "Propose exactly 3 distinct algorithmic approaches. "
        "For each, write a one-paragraph description."
    )
    candidates = call_claude(gen_prompt, temperature=0.7)

    # Step 2 — Evaluate each approach
    eval_prompt = (
        f"Problem: {problem}\n\n"
        f"Candidate approaches:\n{candidates}\n\n"
        "Evaluate each approach on correctness, complexity, and "
        "practicality. Score each 1-10 and select the best one."
    )
    evaluation = call_claude(eval_prompt)

    # Step 3 — Expand the chosen approach
    expand_prompt = (
        f"Based on this evaluation:\n{evaluation}\n\n"
        "Expand the winning approach into detailed pseudocode with "
        "time complexity analysis."
    )
    solution = call_claude(expand_prompt)

    print("=" * 60)
    print("SECTION 1 — Tree of Thoughts")
    print("=" * 60)
    print("\n[Candidates]\n", textwrap.indent(candidates, "  "))
    print("\n[Evaluation]\n", textwrap.indent(evaluation, "  "))
    print("\n[Final Solution]\n", textwrap.indent(solution, "  "))


# ---------------------------------------------------------------------------
# 2. Self-Refine (iterative critique → revision)
# ---------------------------------------------------------------------------

def demo_self_refine(max_rounds: int = 2):
    """Generate an initial answer, critique it, then revise — repeat."""

    task = (
        "Write a Python function `flatten(nested)` that flattens an "
        "arbitrarily nested list. Handle edge cases."
    )

    print("\n" + "=" * 60)
    print(f"SECTION 2 — Self-Refine ({max_rounds} rounds)")
    print("=" * 60)

    # Initial generation
    draft = call_claude(f"{task}\n\nProvide the implementation.")
    print(f"\n[Draft 0]\n{textwrap.indent(draft, '  ')}")

    for i in range(1, max_rounds + 1):
        # Critique
        critique_prompt = (
            f"Here is a Python function:\n{draft}\n\n"
            "List specific issues: bugs, missing edge cases, style "
            "problems, performance concerns. Be thorough."
        )
        critique = call_claude(critique_prompt)
        print(f"\n[Critique {i}]\n{textwrap.indent(critique, '  ')}")

        # Revise
        revise_prompt = (
            f"Original code:\n{draft}\n\n"
            f"Critique:\n{critique}\n\n"
            "Produce a revised version that addresses every issue."
        )
        draft = call_claude(revise_prompt)
        print(f"\n[Revision {i}]\n{textwrap.indent(draft, '  ')}")


# ---------------------------------------------------------------------------
# 3. Meta-Prompting (LLM writes its own prompt)
# ---------------------------------------------------------------------------

def demo_meta_prompting():
    """Ask Claude to craft the optimal prompt for a task, then execute it."""

    goal = (
        "I need to extract structured event data (title, date, location, "
        "description) from informal email text."
    )

    # Step 1 — Generate the prompt
    meta_prompt = (
        f"Goal: {goal}\n\n"
        "You are a prompt engineering expert. Write the best possible "
        "system prompt and user-prompt template for this task. "
        "Include format instructions and few-shot examples."
    )
    generated_prompt = call_claude(meta_prompt, temperature=0.3)

    # Step 2 — Use the generated prompt on a sample email
    sample_email = (
        "Hey! Just a heads up — the team offsite is happening on March 15 "
        "at the Riverside Conference Center. It'll be a full day of "
        "workshops on Q2 planning. Don't forget to RSVP!"
    )

    execution_prompt = (
        f"Using the following prompting strategy:\n{generated_prompt}\n\n"
        f"Now apply it to extract event data from this email:\n\"{sample_email}\""
    )
    result = call_claude(execution_prompt)

    print("\n" + "=" * 60)
    print("SECTION 3 — Meta-Prompting")
    print("=" * 60)
    print("\n[Generated Prompt Template]")
    print(textwrap.indent(generated_prompt, "  "))
    print("\n[Extraction Result]")
    print(textwrap.indent(result, "  "))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: Set the ANTHROPIC_API_KEY environment variable first.")
        raise SystemExit(1)

    client = anthropic.Anthropic()

    try:
        demo_tree_of_thoughts()
        demo_self_refine()
        demo_meta_prompting()
    except anthropic.APIError as exc:
        print(f"\nAPI error: {exc}")
