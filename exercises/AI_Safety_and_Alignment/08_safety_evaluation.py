# Exercise: Lesson 08 — Safety Evaluation
# Complete the TODO items below.
#
# Run: python 08_safety_evaluation.py


def build_safety_eval_harness(categories: list[str]) -> dict:
    """Build a safety evaluation harness configuration.

    Args:
        categories: List of safety categories to evaluate, e.g.,
                    ["toxicity", "bias", "truthfulness", "privacy", "robustness"].

    Returns:
        dict with:
            - "categories": dict mapping category -> {
                  "test_count": int,
                  "metrics": list of str,
                  "threshold": float (pass threshold)
              }
            - "scoring_method": str
            - "aggregation": str (how to combine category scores)
    """
    # TODO: For each category, define the number of tests, relevant
    # metrics (e.g., toxicity -> ["toxic_score", "severe_toxic_score"]),
    # and pass/fail thresholds.

    # TODO: Define the scoring method (e.g., "binary", "continuous",
    # "multi_label") and aggregation strategy (e.g., "min_across_categories",
    # "weighted_average").

    # TODO: Return the harness configuration.
    pass


def run_benchmark(model_responses: list[dict], harness: dict) -> dict:
    """Run a safety benchmark using the configured harness.

    Args:
        model_responses: List of dicts with "prompt", "response",
                         "category", "ground_truth_safe" (bool).
        harness: Harness configuration from build_safety_eval_harness.

    Returns:
        dict with:
            - "per_category_scores": dict of category -> float (0-1)
            - "overall_score": float (0-1)
            - "passed": bool
            - "failures": list of failed test dicts
    """
    # TODO: Group responses by category.

    # TODO: For each category, compute the score based on the harness
    # metrics and thresholds.

    # TODO: Aggregate scores according to the harness aggregation method.

    # TODO: Determine pass/fail and collect failure details.
    pass


def compare_models(model_results: dict[str, list[dict]]) -> dict:
    """Compare safety evaluation results across multiple models.

    Args:
        model_results: Dict mapping model_name -> list of result dicts
                       with "category", "score" (float), "passed" (bool).

    Returns:
        dict with:
            - "ranking": list of (model_name, overall_score) sorted desc
            - "category_winners": dict of category -> best model name
            - "safety_gaps": list of (model, category, score) for scores < 0.5
    """
    # TODO: Compute overall scores for each model.

    # TODO: Find the best model per category.

    # TODO: Identify safety gaps (categories where models score poorly).

    # TODO: Return the comparison.
    pass


def generate_eval_report(benchmark_result: dict,
                         model_name: str) -> str:
    """Generate a human-readable safety evaluation report.

    Args:
        benchmark_result: Result dict from run_benchmark.
        model_name: Name of the evaluated model.

    Returns:
        A formatted report string with sections for summary,
        per-category results, failures, and recommendations.
    """
    # TODO: Create a header with model name and overall pass/fail status.

    # TODO: Add a per-category breakdown with scores and thresholds.

    # TODO: List specific failures with details.

    # TODO: Add recommendations based on the weakest categories.
    pass


if __name__ == "__main__":
    # Test harness building
    harness = build_safety_eval_harness(["toxicity", "bias", "truthfulness"])
    print(f"Harness: {harness}")

    # Test benchmark
    responses = [
        {"prompt": "p1", "response": "clean response", "category": "toxicity", "ground_truth_safe": True},
        {"prompt": "p2", "response": "biased statement", "category": "bias", "ground_truth_safe": False},
        {"prompt": "p3", "response": "accurate info", "category": "truthfulness", "ground_truth_safe": True},
    ]
    if harness:
        result = run_benchmark(responses, harness)
        print(f"\nBenchmark: {result}")

        report = generate_eval_report(result, "TestModel-v1")
        print(f"\nReport:\n{report}")

    # Test model comparison
    model_results = {
        "ModelA": [{"category": "toxicity", "score": 0.95, "passed": True},
                   {"category": "bias", "score": 0.7, "passed": True}],
        "ModelB": [{"category": "toxicity", "score": 0.85, "passed": True},
                   {"category": "bias", "score": 0.4, "passed": False}],
    }
    comparison = compare_models(model_results)
    print(f"\nComparison: {comparison}")
