"""
Exercises for Lesson 16: Evaluation Metrics
Topic: LLM_and_NLP

Solutions to practice problems from the lesson.
"""

import numpy as np
from math import comb
from typing import Dict, List, Tuple
from collections import Counter
from enum import Enum


# ============================================================
# Shared utilities: BLEU score implementation (no nltk needed)
# ============================================================

def compute_ngrams(tokens: List[str], n: int) -> Counter:
    """Compute n-gram frequency counts."""
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu_score(reference: str, candidate: str, max_n: int = 4,
               weights: Tuple[float, ...] = None) -> float:
    """
    Compute BLEU score with smoothing (method 1: add epsilon).
    Simplified implementation without nltk dependency.
    """
    if weights is None:
        weights = tuple(1.0 / max_n for _ in range(max_n))

    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()

    # Brevity penalty
    bp = min(1.0, np.exp(1 - len(ref_tokens) / max(len(cand_tokens), 1)))

    # Modified precision for each n-gram order
    log_precisions = []
    epsilon = 1e-10  # Smoothing

    for n in range(1, max_n + 1):
        ref_ngrams = compute_ngrams(ref_tokens, n)
        cand_ngrams = compute_ngrams(cand_tokens, n)

        # Clipped counts
        clipped = 0
        total = 0
        for ngram, count in cand_ngrams.items():
            clipped += min(count, ref_ngrams.get(ngram, 0))
            total += count

        precision = (clipped + epsilon) / (total + epsilon) if total > 0 else epsilon
        log_precisions.append(weights[n - 1] * np.log(precision))

    return bp * np.exp(sum(log_precisions))


# === Exercise 1: BLEU Score Limitations ===
# Problem: Compute BLEU-1 and BLEU-4 for three candidate-reference pairs.
# Explain why BLEU fails to capture quality differences.

def exercise_1():
    """BLEU score limitations analysis."""
    print("=" * 60)
    print("Exercise 1: BLEU Score Limitations")
    print("=" * 60)

    reference = "The patient was given a high dose of aspirin to reduce fever."

    candidates = {
        "a_exact": "The patient was given a high dose of aspirin to reduce fever.",
        "b_paraphrase": (
            "The sick person received a large amount of aspirin medication "
            "for temperature reduction."
        ),
        "c_incoherent": "High aspirin dose fever patient given reduce.",
    }

    print(f"\nReference: {reference}")
    print()

    print(f"{'Candidate':<16} {'BLEU-1':<10} {'BLEU-4':<10} {'Quality'}")
    print("-" * 55)

    for name, cand in candidates.items():
        bleu1 = bleu_score(reference, cand, max_n=1, weights=(1.0,))
        bleu4 = bleu_score(reference, cand, max_n=4, weights=(0.25, 0.25, 0.25, 0.25))

        # Quality label
        if name == "a_exact":
            quality = "Exact match"
        elif name == "b_paraphrase":
            quality = "Good paraphrase"
        else:
            quality = "Incoherent"

        print(f"{name:<16} {bleu1:<10.4f} {bleu4:<10.4f} {quality}")

    print("\nAnalysis:")
    print("-" * 55)
    print("  Candidate A (exact): BLEU-1 = ~1.0, BLEU-4 = ~1.0")
    print("    As expected for an exact match.")
    print()
    print("  Candidate B (paraphrase): Low BLEU-1 despite being a PERFECT paraphrase!")
    print("    BLEU counts n-gram overlaps literally:")
    print("      'sick person' != 'patient'")
    print("      'large amount' != 'high dose'")
    print("      'temperature reduction' != 'reduce fever'")
    print("    BLEU has no concept of synonymy or semantic similarity.")
    print()
    print("  Candidate C (incoherent): Moderate BLEU-1 despite being USELESS!")
    print("    Several individual words match ('aspirin', 'dose', 'fever', etc.)")
    print("    BLEU-1 has no notion of word ORDER or grammaticality.")
    print("    Only at bigram level (BLEU-2+) does incoherence get penalized.")

    print("\n  Better metrics for this scenario:")
    print("    - BERTScore: contextual embeddings capture semantic similarity")
    print("    - COMET: neural metric trained on human quality judgments")
    print("    - LLM-as-Judge: holistic evaluation of fluency + accuracy + completeness")
    print("    - BLEU is still useful when: lexical overlap genuinely matters")
    print("      (machine translation, technical documentation)")


# === Exercise 2: pass@k Calculation and Interpretation ===
# Problem: Calculate pass@1, pass@3, pass@10 for code generation results.

def exercise_2():
    """pass@k calculation and interpretation for code generation."""
    print("\n" + "=" * 60)
    print("Exercise 2: pass@k Calculation and Interpretation")
    print("=" * 60)

    def pass_at_k(n: int, c: int, k: int) -> float:
        """
        Unbiased estimator for pass@k.
        n = samples generated, c = correct samples, k = budget.

        P(at least 1 correct in k samples) = 1 - C(n-c, k) / C(n, k)
        """
        if n - c < k:
            return 1.0
        return 1.0 - comb(n - c, k) / comb(n, k)

    # Evaluation results: (problem_name, n_samples, n_correct)
    results = [
        ("fibonacci",      10, 10),  # Very easy - all correct
        ("binary_search",  10,  7),  # Moderate - most correct
        ("merge_sort",     10,  3),  # Hard - few correct
        ("regex_parser",   10,  1),  # Very hard - barely one correct
        ("graph_coloring", 10,  0),  # Failed - none correct
    ]

    print(f"\n{'Problem':<20} {'n':<5} {'c':<5} {'pass@1':<10} {'pass@3':<10} {'pass@10'}")
    print("-" * 60)

    totals = {1: 0.0, 3: 0.0, 10: 0.0}
    for name, n, c in results:
        p1 = pass_at_k(n, c, 1)
        p3 = pass_at_k(n, c, 3)
        p10 = pass_at_k(n, c, 10)
        print(f"{name:<20} {n:<5} {c:<5} {p1:<10.4f} {p3:<10.4f} {p10:.4f}")
        totals[1] += p1
        totals[3] += p3
        totals[10] += p10

    num_problems = len(results)
    print("-" * 60)
    print(
        f"{'Average':<30} "
        f"{totals[1]/num_problems:<10.4f} "
        f"{totals[3]/num_problems:<10.4f} "
        f"{totals[10]/num_problems:.4f}"
    )

    print("\nInterpretation:")
    print("-" * 50)

    interpretations = {
        "pass@1": (
            "Probability a single generated solution is correct. "
            "Used when deploying the model to generate one answer. "
            f"Average: {totals[1]/num_problems:.2f} - mediocre for production use."
        ),
        "pass@3": (
            "Probability at least one of 3 candidates is correct. "
            "Used when you can verify 3 solutions (e.g., run tests). "
            f"Average: {totals[3]/num_problems:.2f} - improved with selection."
        ),
        "pass@10": (
            "Upper bound with 10 samples. Measures peak capability vs consistency. "
            f"Average: {totals[10]/num_problems:.2f} - model knows answers, just inconsistently."
        ),
    }

    for metric, desc in interpretations.items():
        print(f"\n  {metric}: {desc}")

    print("\n\nGap Analysis (pass@1 vs pass@10):")
    print("-" * 50)
    for name, n, c in results:
        p1 = pass_at_k(n, c, 1)
        p10 = pass_at_k(n, c, 10)
        gap = p10 - p1

        if gap < 0.01:
            insight = "Consistent - model reliably knows this"
        elif p10 == 0:
            insight = "Lacks capability entirely - no sampling helps"
        else:
            insight = f"Unreliable (gap={gap:.2f}) - use best-of-N with verifier"

        print(f"  {name:<20} pass@1={p1:.2f}  pass@10={p10:.2f}  {insight}")


# === Exercise 3: LLM-as-Judge Bias Mitigation ===
# Problem: Implement a debiased LLM judge using position-swapped
# double evaluation. Simulated (no API calls).

def exercise_3():
    """LLM-as-Judge debiased evaluation (simulated)."""
    print("\n" + "=" * 60)
    print("Exercise 3: LLM-as-Judge Bias Mitigation")
    print("=" * 60)

    class JudgeResult(Enum):
        A_WINS = "A"
        B_WINS = "B"
        TIE = "Tie"

    def simulated_judge(
        response_first: str, response_second: str,
        position_bias: float = 0.15,
        true_quality_diff: float = 0.0,
    ) -> str:
        """
        Simulate an LLM judge with configurable position bias.

        Args:
            response_first: Response shown first (position A)
            response_second: Response shown second (position B)
            position_bias: Probability of preferring first position (0.0-1.0 extra)
            true_quality_diff: True quality difference (positive = first is better)

        Returns:
            "First", "Second", or "Tie"
        """
        # Base probability of choosing first: 0.5 + quality_diff + position_bias
        p_first = 0.5 + true_quality_diff + position_bias

        # Add some noise
        p_first += np.random.normal(0, 0.05)
        p_first = np.clip(p_first, 0.05, 0.95)

        roll = np.random.random()
        if roll < p_first - 0.1:
            return "First"
        elif roll > p_first + 0.1:
            return "Second"
        else:
            return "Tie"

    def single_comparison(
        question: str, resp_first: str, resp_second: str,
        position_bias: float = 0.15,
        true_quality_diff: float = 0.0,
    ) -> JudgeResult:
        """Single comparison (simulated)."""
        result = simulated_judge(
            resp_first, resp_second,
            position_bias=position_bias,
            true_quality_diff=true_quality_diff,
        )
        if result == "First":
            return JudgeResult.A_WINS
        elif result == "Second":
            return JudgeResult.B_WINS
        return JudgeResult.TIE

    def debiased_judge(
        question: str, response_a: str, response_b: str,
        position_bias: float = 0.15,
        true_quality_diff: float = 0.0,
    ) -> Dict:
        """
        Debiased LLM judge using position-swapped double evaluation.

        Run 1: A first, B second -> get judgment
        Run 2: B first, A second -> get judgment (swapped)

        If both agree -> confident result
        If they disagree -> Tie (position bias detected)
        """
        # Run 1: A shown first
        result_1 = single_comparison(
            question, response_a, response_b,
            position_bias=position_bias,
            true_quality_diff=true_quality_diff,
        )

        # Run 2: B shown first (swapped order)
        result_2_raw = single_comparison(
            question, response_b, response_a,
            position_bias=position_bias,
            true_quality_diff=-true_quality_diff,  # Flip quality perspective
        )

        # Normalize result_2: "First won" in run 2 means B won (since B was first)
        if result_2_raw == JudgeResult.A_WINS:  # "First" won = B was first = B wins
            result_2_normalized = JudgeResult.B_WINS
        elif result_2_raw == JudgeResult.B_WINS:  # "Second" won = A was second = A wins
            result_2_normalized = JudgeResult.A_WINS
        else:
            result_2_normalized = JudgeResult.TIE

        # Aggregate
        if result_1 == result_2_normalized:
            final = result_1
            confidence = "high"
        elif result_1 == JudgeResult.TIE or result_2_normalized == JudgeResult.TIE:
            final = result_1 if result_2_normalized == JudgeResult.TIE else result_2_normalized
            confidence = "medium"
        else:
            final = JudgeResult.TIE  # Disagreement = position bias detected
            confidence = "low (position bias detected)"

        return {
            "final_result": final.value,
            "run_1_result": result_1.value,
            "run_2_result": result_2_normalized.value,
            "confidence": confidence,
            "position_bias_detected": result_1 != result_2_normalized,
        }

    # Statistical analysis: run many trials to show debiasing effect
    np.random.seed(42)
    n_trials = 1000

    question = "What is recursion in programming?"
    response_a = "Recursion is when a function calls itself."
    response_b = (
        "Recursion is a programming technique where a function invokes itself "
        "to solve smaller subproblems, with a base case to stop the recursion."
    )

    # true_quality_diff = -0.2 means B is genuinely better
    true_quality_diff = -0.2
    position_bias = 0.15

    print(f"\n  Response A: {response_a}")
    print(f"  Response B: {response_b[:80]}...")
    print(f"  True quality: B is better (diff = {true_quality_diff})")
    print(f"  Position bias: {position_bias} (favors whichever is shown first)")

    # Biased evaluation (single comparison, A always first)
    biased_results = {"A": 0, "B": 0, "Tie": 0}
    for _ in range(n_trials):
        r = single_comparison(
            question, response_a, response_b,
            position_bias=position_bias,
            true_quality_diff=true_quality_diff,
        )
        biased_results[r.value] += 1

    # Debiased evaluation
    debiased_results = {"A": 0, "B": 0, "Tie": 0}
    bias_detected_count = 0
    for _ in range(n_trials):
        r = debiased_judge(
            question, response_a, response_b,
            position_bias=position_bias,
            true_quality_diff=true_quality_diff,
        )
        debiased_results[r["final_result"]] += 1
        if r["position_bias_detected"]:
            bias_detected_count += 1

    print(f"\n  Results over {n_trials} trials:")
    print("-" * 55)
    print(f"  {'Method':<20} {'A wins':<10} {'B wins':<10} {'Tie':<10}")
    print("  " + "-" * 40)
    print(
        f"  {'Biased (single)':<20} "
        f"{biased_results['A']:>5} ({biased_results['A']/n_trials*100:.1f}%)  "
        f"{biased_results['B']:>5} ({biased_results['B']/n_trials*100:.1f}%)  "
        f"{biased_results['Tie']:>5}"
    )
    print(
        f"  {'Debiased (double)':<20} "
        f"{debiased_results['A']:>5} ({debiased_results['A']/n_trials*100:.1f}%)  "
        f"{debiased_results['B']:>5} ({debiased_results['B']/n_trials*100:.1f}%)  "
        f"{debiased_results['Tie']:>5}"
    )
    print(f"\n  Position bias detected in {bias_detected_count}/{n_trials} trials "
          f"({bias_detected_count/n_trials*100:.1f}%)")

    # The debiased method should show B winning more often (ground truth)
    biased_b_rate = biased_results["B"] / n_trials
    debiased_b_rate = debiased_results["B"] / n_trials
    print(f"\n  B win rate: biased={biased_b_rate:.2%} vs debiased={debiased_b_rate:.2%}")
    print(f"  Debiasing {'improved' if debiased_b_rate > biased_b_rate else 'maintained'} "
          f"detection of the better response.")

    print("\n  Additional bias mitigation strategies:")
    print("    - Multiple independent judges with majority vote")
    print("    - Present responses without labels (just 'First'/'Second')")
    print("    - Use scoring (1-5) instead of pairwise preference")
    print("    - Shuffle response order across test set and check consistency")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
