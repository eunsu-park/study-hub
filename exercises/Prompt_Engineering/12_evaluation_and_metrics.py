# Exercise: Lesson 12 — Evaluation and Metrics
# Complete the TODO items below.
#
# Run: python 12_evaluation_and_metrics.py

from __future__ import annotations

from dataclasses import dataclass, field


# === Exercise 1: Build an Evaluation Dataset ===
# Create a structured evaluation dataset with inputs, expected outputs,
# and metadata for systematic prompt testing.

@dataclass
class EvalExample:
    input_text: str
    expected_output: str
    category: str
    difficulty: str  # "easy", "medium", "hard"


def build_eval_dataset() -> list[EvalExample]:
    """Create an evaluation dataset with at least 6 examples.

    Hint: Include a mix of categories (factual, reasoning, creative)
    and difficulty levels. Each example needs a clear expected output
    that can be compared against model output.
    """
    # TODO: Create and return a list of 6+ EvalExample instances
    # Cover at least 2 categories and 2 difficulty levels
    pass


def exercise_1():
    """Verify the evaluation dataset is well-formed."""
    dataset = build_eval_dataset()
    assert dataset is not None and len(dataset) >= 6, "Need at least 6 examples"
    categories = {ex.category for ex in dataset}
    difficulties = {ex.difficulty for ex in dataset}
    assert len(categories) >= 2, "Need at least 2 categories"
    assert len(difficulties) >= 2, "Need at least 2 difficulty levels"
    for ex in dataset:
        assert ex.input_text and ex.expected_output, "All fields must be non-empty"
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Categories: {categories}")
    print(f"  Difficulties: {difficulties}")


# === Exercise 2: Exact and Fuzzy Match Metrics ===
# Implement basic evaluation metrics.

def exact_match(predicted: str, expected: str) -> bool:
    """Return True if predicted matches expected (case-insensitive, stripped).

    Hint: Strip whitespace and compare lowercased strings.
    """
    # TODO: Implement case-insensitive exact match
    pass


def fuzzy_match(predicted: str, expected: str) -> float:
    """Return a 0-1 similarity score based on word overlap (Jaccard index).

    Hint: Tokenize both strings to lowercase word sets.
    Jaccard = |A intersection B| / |A union B|
    Handle the edge case where both sets are empty (return 1.0).
    """
    # TODO: Tokenize to lowercase word sets
    # TODO: Compute and return Jaccard similarity
    pass


def exercise_2():
    """Verify match metrics work correctly."""
    assert exact_match("Hello World", "hello world") is True
    assert exact_match("Hello", "World") is False
    assert exact_match("  foo  ", "foo") is True

    score = fuzzy_match("the quick brown fox", "the fast brown fox")
    assert 0.0 <= score <= 1.0, "Score must be in [0, 1]"
    assert score > 0.5, "Should have high overlap"

    score_zero = fuzzy_match("apple", "orange")
    assert score_zero == 0.0, "No overlap should yield 0"

    print(f"  fuzzy_match('the quick brown fox', 'the fast brown fox') = {score:.3f}")
    print(f"  fuzzy_match('apple', 'orange') = {score_zero:.3f}")
    print("  All metric assertions passed")


# === Exercise 3: LLM-as-Judge Prompt Builder ===
# Build a prompt that uses an LLM to judge response quality.

def build_judge_prompt(
    question: str,
    expected: str,
    actual: str,
    criteria: list[str] | None = None,
) -> str:
    """Build a prompt for LLM-as-judge evaluation.

    The judge should rate the actual response on a 1-5 scale and provide
    a brief explanation.

    Hint: Include the question, reference answer, candidate answer, and
    evaluation criteria. Ask the judge to respond with:
    {"score": N, "explanation": "..."}

    Default criteria: ["accuracy", "completeness", "clarity"]
    """
    # TODO: Set default criteria if None
    # TODO: Build the judge prompt with all required sections
    # TODO: Specify the JSON output format
    pass


def exercise_3():
    """Verify the judge prompt has all required components."""
    prompt = build_judge_prompt(
        question="What is Python?",
        expected="A high-level programming language.",
        actual="Python is a snake.",
        criteria=["accuracy", "relevance"],
    )
    assert prompt is not None, "Must return a string"
    assert "Python" in prompt, "Must include the question content"
    assert "score" in prompt.lower(), "Must request a score"
    assert "accuracy" in prompt, "Must include evaluation criteria"
    print(f"  Judge prompt length: {len(prompt)} chars")
    print("  Judge prompt contains all required sections")


# === Exercise 4: Batch Evaluation Runner ===
# Run evaluation across a full dataset and compute aggregate metrics.

@dataclass
class EvalResult:
    example: EvalExample
    predicted: str
    exact: bool
    fuzzy: float


def run_evaluation(
    dataset: list[EvalExample],
    predict_fn=None,
) -> dict:
    """Run evaluation on the full dataset and return aggregate metrics.

    Args:
        dataset: List of evaluation examples.
        predict_fn: A callable(input_text) -> str. If None, use a mock
                    that returns the first 5 words of expected_output.

    Returns a dict with:
        - "results": list of EvalResult
        - "exact_match_rate": float (0-1)
        - "avg_fuzzy_score": float (0-1)
        - "by_category": dict[category, {"exact_rate": float, "fuzzy_avg": float}]

    Hint: Use exact_match() and fuzzy_match() from Exercise 2.
    """
    # TODO: Define default predict_fn if None
    # TODO: Evaluate each example and collect EvalResult instances
    # TODO: Compute overall exact match rate and average fuzzy score
    # TODO: Group results by category and compute per-category metrics
    pass


def exercise_4():
    """Verify the batch evaluator works end-to-end."""
    dataset = build_eval_dataset()
    report = run_evaluation(dataset)
    assert report is not None, "Must return a report dict"
    assert "exact_match_rate" in report, "Must include exact_match_rate"
    assert "avg_fuzzy_score" in report, "Must include avg_fuzzy_score"
    assert "by_category" in report, "Must include per-category breakdown"
    print(f"  Exact match rate: {report['exact_match_rate']:.2%}")
    print(f"  Avg fuzzy score:  {report['avg_fuzzy_score']:.3f}")
    for cat, metrics in report["by_category"].items():
        print(f"  [{cat}] exact={metrics['exact_rate']:.2%} "
              f"fuzzy={metrics['fuzzy_avg']:.3f}")


if __name__ == "__main__":
    print("=== Exercise 1: Evaluation Dataset ===")
    exercise_1()

    print("=== Exercise 2: Match Metrics ===")
    exercise_2()

    print("=== Exercise 3: LLM-as-Judge ===")
    exercise_3()

    print("=== Exercise 4: Batch Evaluation ===")
    exercise_4()

    print("\nAll exercises completed!")
