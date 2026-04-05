# Exercise: Lesson 11 — Prompt Optimization
# Complete the TODO items below.
#
# Run: python 11_prompt_optimization.py

from __future__ import annotations

import time
from dataclasses import dataclass, field


# === Exercise 1: Prompt Variant Comparison ===
# Define multiple prompt variants for the same task and compare them
# using a simple scoring rubric.

@dataclass
class PromptVariant:
    name: str
    system: str
    user_template: str
    # Populated after evaluation
    scores: list[float] = field(default_factory=list)

    @property
    def avg_score(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0


def create_variants() -> list[PromptVariant]:
    """Create at least 3 prompt variants for a summarization task.

    Hint: Vary instruction specificity, output format, and constraints.
    Example axes of variation:
      - Concise vs. detailed system prompts
      - Bullet-point vs. paragraph output
      - With vs. without word-count constraint
    """
    # TODO: Create and return a list of 3+ PromptVariant instances
    # Each should have a different system prompt and/or user_template
    pass


def exercise_1():
    """Verify that variants are properly defined."""
    variants = create_variants()
    assert variants is not None and len(variants) >= 3, "Need at least 3 variants"
    for v in variants:
        assert v.name, "Each variant needs a name"
        assert v.system, "Each variant needs a system prompt"
        assert v.user_template, "Each variant needs a user template"
        print(f"  Variant '{v.name}': system={len(v.system)} chars")
    print(f"  Total variants: {len(variants)}")


# === Exercise 2: Simulated A/B Evaluation ===
# Score each variant against a test set and pick the winner.

TEST_CASES = [
    {"input": "Explain machine learning in one sentence.", "ideal_length": 20},
    {"input": "What is Python used for?", "ideal_length": 25},
    {"input": "Describe Docker briefly.", "ideal_length": 15},
]


def score_response(response: str, ideal_word_count: int) -> float:
    """Score a mock response on a 0-1 scale.

    Hint: Combine two sub-scores:
      1. Length score: 1 - min(1, |word_count - ideal| / ideal)
      2. Structure score: 0.5 if response ends with '.', else 0.0
    Return the average of the sub-scores.
    """
    # TODO: Compute word count of the response
    # TODO: Compute length score (closeness to ideal_word_count)
    # TODO: Compute structure score (proper ending punctuation)
    # TODO: Return average of the two sub-scores
    pass


def evaluate_variants(variants: list[PromptVariant]) -> PromptVariant:
    """Evaluate all variants against TEST_CASES and return the best one.

    Hint: For each variant and test case, generate a mock response
    (you can use the user_template formatted with the input) and score it.
    Store scores in variant.scores and return the variant with the
    highest avg_score.
    """
    # TODO: Loop over variants and test cases
    # TODO: Generate a mock response (e.g., use the template text itself)
    # TODO: Score each response and append to variant.scores
    # TODO: Return the variant with the highest avg_score
    pass


def exercise_2():
    """Run evaluation and verify a winner is selected."""
    variants = create_variants()
    winner = evaluate_variants(variants)
    assert winner is not None, "Must return a winning variant"
    assert winner.avg_score > 0, "Winner must have a positive score"
    for v in variants:
        print(f"  {v.name}: avg_score={v.avg_score:.3f}")
    print(f"  Winner: {winner.name} ({winner.avg_score:.3f})")


# === Exercise 3: Iterative Prompt Refinement Loop ===
# Implement a loop that refines a prompt based on feedback.

@dataclass
class RefinementStep:
    iteration: int
    prompt: str
    score: float
    feedback: str


def iterative_refinement(
    initial_prompt: str,
    target_score: float = 0.8,
    max_iterations: int = 5,
) -> list[RefinementStep]:
    """Simulate iterative prompt refinement.

    Hint: In each iteration:
      1. Score the current prompt (use len-based heuristic or mock scorer)
      2. If score >= target_score, stop
      3. Generate feedback (e.g., 'add more specificity')
      4. Apply a simple transformation to the prompt
      5. Record each step as a RefinementStep
    """
    # TODO: Initialize history list
    # TODO: Loop up to max_iterations
    # TODO: Score current prompt, generate feedback, refine
    # TODO: Return the history
    pass


def exercise_3():
    """Verify refinement loop runs and improves."""
    history = iterative_refinement("Summarize the text.")
    assert history is not None and len(history) >= 1, "Must have at least 1 step"
    assert all(isinstance(s, RefinementStep) for s in history)
    for step in history:
        print(f"  Iteration {step.iteration}: score={step.score:.3f} "
              f"feedback='{step.feedback}'")
    print(f"  Total iterations: {len(history)}")


# === Exercise 4: Token Cost Estimator ===
# Estimate prompt cost to compare verbose vs. concise variants.

def estimate_tokens(text: str) -> int:
    """Estimate token count using a simple heuristic.

    Hint: A reasonable approximation is len(text) / 4 for English text.
    """
    # TODO: Return an integer estimate of token count
    pass


def compare_cost(variants: list[PromptVariant]) -> dict[str, dict]:
    """Compare estimated token costs across variants.

    Hint: For each variant, estimate tokens for system + user_template.
    Return a dict mapping variant name to {'tokens': N, 'relative_cost': float}
    where relative_cost = tokens / min_tokens.
    """
    # TODO: Estimate tokens for each variant
    # TODO: Find the minimum token count
    # TODO: Compute relative cost for each variant
    pass


def exercise_4():
    """Verify cost comparison works."""
    variants = create_variants()
    costs = compare_cost(variants)
    assert costs is not None and len(costs) >= 3, "Must compare all variants"
    for name, info in costs.items():
        print(f"  {name}: ~{info['tokens']} tokens "
              f"(relative: {info['relative_cost']:.2f}x)")


if __name__ == "__main__":
    print("=== Exercise 1: Prompt Variants ===")
    exercise_1()

    print("=== Exercise 2: A/B Evaluation ===")
    exercise_2()

    print("=== Exercise 3: Iterative Refinement ===")
    exercise_3()

    print("=== Exercise 4: Token Cost Estimator ===")
    exercise_4()

    print("\nAll exercises completed!")
