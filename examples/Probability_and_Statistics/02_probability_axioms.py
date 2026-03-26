"""
Probability Axioms and Rules

Demonstrates:
1. Sample Space and Event Generator
2. Conditional Probability Calculator
3. Bayes' Theorem Solver
4. Independence Checker

Theory:
- Kolmogorov Axioms: P(Ω)=1, P(A)≥0, P(∪Ai)=ΣP(Ai) for disjoint Ai
- Bayes: P(A|B) = P(B|A)P(A) / P(B)
- Independence: P(A∩B) = P(A)·P(B)

Adapted from Probability and Statistics Lesson 02.
"""

from itertools import product
from collections import Counter
import random


# ─────────────────────────────────────────────────
# 1. SAMPLE SPACE AND EVENTS
# ─────────────────────────────────────────────────

def build_sample_space(*experiments: list) -> list[tuple]:
    """Build sample space as Cartesian product of experiments."""
    return list(product(*experiments))


def event_probability(sample_space: list, event_condition) -> float:
    """P(A) = |A| / |Ω| for equally likely outcomes."""
    favorable = [s for s in sample_space if event_condition(s)]
    return len(favorable) / len(sample_space)


def demo_sample_space():
    print("=" * 60)
    print("  Sample Spaces and Events")
    print("=" * 60)

    # Two dice
    die = [1, 2, 3, 4, 5, 6]
    omega = build_sample_space(die, die)
    print(f"\n  Two dice: |Ω| = {len(omega)}")

    # Events
    events = {
        "Sum = 7": lambda s: s[0] + s[1] == 7,
        "Sum ≥ 10": lambda s: s[0] + s[1] >= 10,
        "Doubles": lambda s: s[0] == s[1],
        "At least one 6": lambda s: 6 in s,
    }

    for name, cond in events.items():
        p = event_probability(omega, cond)
        count = sum(1 for s in omega if cond(s))
        print(f"  P({name}) = {count}/36 = {p:.4f}")

    # Sum distribution
    print(f"\n  Sum distribution:")
    sums = Counter(s[0] + s[1] for s in omega)
    for total in sorted(sums):
        bar = "█" * sums[total]
        print(f"  {total:>3}: {sums[total]:>2}/36  {bar}")


# ─────────────────────────────────────────────────
# 2. CONDITIONAL PROBABILITY
# ─────────────────────────────────────────────────

def conditional_probability(sample_space: list,
                            event_a, event_b) -> float:
    """P(A|B) = P(A∩B) / P(B)."""
    b_outcomes = [s for s in sample_space if event_b(s)]
    if not b_outcomes:
        return 0.0
    ab_outcomes = [s for s in b_outcomes if event_a(s)]
    return len(ab_outcomes) / len(b_outcomes)


def demo_conditional():
    print("\n" + "=" * 60)
    print("  Conditional Probability")
    print("=" * 60)

    die = [1, 2, 3, 4, 5, 6]
    omega = build_sample_space(die, die)

    # P(Sum = 8 | first die is 3)
    p = conditional_probability(
        omega,
        event_a=lambda s: s[0] + s[1] == 8,
        event_b=lambda s: s[0] == 3,
    )
    print(f"\n  P(Sum=8 | Die1=3) = {p:.4f}")
    print(f"  (Only outcome: (3,5), given 6 possibilities for die2)")

    # P(Die1=6 | Sum≥10)
    p2 = conditional_probability(
        omega,
        event_a=lambda s: s[0] == 6,
        event_b=lambda s: s[0] + s[1] >= 10,
    )
    print(f"  P(Die1=6 | Sum≥10) = {p2:.4f}")

    # Card drawing — without replacement
    print(f"\n  Card example (no replacement):")
    print(f"  P(2nd King | 1st King) = 3/51 = {3/51:.4f}")
    print(f"  P(2nd King | 1st not King) = 4/51 = {4/51:.4f}")


# ─────────────────────────────────────────────────
# 3. BAYES' THEOREM
# ─────────────────────────────────────────────────

def bayes_theorem(prior: float, likelihood: float,
                  evidence: float) -> float:
    """P(H|E) = P(E|H) · P(H) / P(E)."""
    return (likelihood * prior) / evidence


def total_probability(priors: list[float],
                      likelihoods: list[float]) -> float:
    """P(E) = Σ P(E|Hi) · P(Hi)."""
    return sum(p * l for p, l in zip(priors, likelihoods))


def demo_bayes():
    print("\n" + "=" * 60)
    print("  Bayes' Theorem")
    print("=" * 60)

    # Medical test example
    print("\n  Medical Test Example:")
    prevalence = 0.001        # P(Disease)
    sensitivity = 0.99        # P(+|Disease)
    specificity = 0.95        # P(-|No Disease)
    false_positive = 1 - specificity  # P(+|No Disease)

    p_positive = total_probability(
        priors=[prevalence, 1 - prevalence],
        likelihoods=[sensitivity, false_positive],
    )

    p_disease_given_pos = bayes_theorem(
        prior=prevalence,
        likelihood=sensitivity,
        evidence=p_positive,
    )

    print(f"  Prevalence:   P(D) = {prevalence:.4f}")
    print(f"  Sensitivity:  P(+|D) = {sensitivity:.4f}")
    print(f"  Specificity:  P(-|~D) = {specificity:.4f}")
    print(f"  P(+) = {p_positive:.6f}")
    print(f"  P(D|+) = {p_disease_given_pos:.4f}")
    print(f"  → Only {p_disease_given_pos:.1%} of positive tests are true positives!")

    # Sequential updating
    print(f"\n  Sequential updating (two positive tests):")
    new_prior = p_disease_given_pos
    p_pos_2 = total_probability(
        [new_prior, 1 - new_prior],
        [sensitivity, false_positive],
    )
    p_after_2 = bayes_theorem(new_prior, sensitivity, p_pos_2)
    print(f"  After 1st test: P(D|+) = {new_prior:.4f}")
    print(f"  After 2nd test: P(D|++) = {p_after_2:.4f}")


# ─────────────────────────────────────────────────
# 4. INDEPENDENCE CHECKER
# ─────────────────────────────────────────────────

def check_independence(sample_space: list, event_a, event_b,
                       name_a: str = "A", name_b: str = "B") -> dict:
    """Check if P(A∩B) = P(A)·P(B)."""
    n = len(sample_space)
    a = sum(1 for s in sample_space if event_a(s))
    b = sum(1 for s in sample_space if event_b(s))
    ab = sum(1 for s in sample_space if event_a(s) and event_b(s))

    p_a = a / n
    p_b = b / n
    p_ab = ab / n
    product = p_a * p_b
    independent = abs(p_ab - product) < 1e-10

    return {
        "P(A)": p_a, "P(B)": p_b,
        "P(A∩B)": p_ab, "P(A)·P(B)": product,
        "independent": independent,
    }


def demo_independence():
    print("\n" + "=" * 60)
    print("  Independence Checker")
    print("=" * 60)

    die = [1, 2, 3, 4, 5, 6]
    omega = build_sample_space(die, die)

    tests = [
        ("Die1 even", "Die2 even",
         lambda s: s[0] % 2 == 0, lambda s: s[1] % 2 == 0),
        ("Die1 even", "Sum even",
         lambda s: s[0] % 2 == 0, lambda s: (s[0]+s[1]) % 2 == 0),
        ("Die1 ≤ 3", "Sum ≤ 6",
         lambda s: s[0] <= 3, lambda s: s[0]+s[1] <= 6),
    ]

    for name_a, name_b, ea, eb in tests:
        result = check_independence(omega, ea, eb, name_a, name_b)
        status = "Independent" if result["independent"] else "Dependent"
        print(f"\n  {name_a} vs {name_b}: {status}")
        print(f"    P(A)={result['P(A)']:.4f}, P(B)={result['P(B)']:.4f}")
        print(f"    P(A∩B)={result['P(A∩B)']:.4f}, P(A)·P(B)={result['P(A)·P(B)']:.4f}")

    # Monte Carlo independence test
    print(f"\n  Monte Carlo independence test (coin flips):")
    random.seed(42)
    n = 100_000
    flips = [(random.choice([0, 1]), random.choice([0, 1])) for _ in range(n)]
    p_h1 = sum(1 for f in flips if f[0] == 1) / n
    p_h2 = sum(1 for f in flips if f[1] == 1) / n
    p_both = sum(1 for f in flips if f[0] == 1 and f[1] == 1) / n
    print(f"    P(H1)={p_h1:.4f}, P(H2)={p_h2:.4f}")
    print(f"    P(H1∩H2)={p_both:.4f} ≈ P(H1)·P(H2)={p_h1*p_h2:.4f}")


if __name__ == "__main__":
    demo_sample_space()
    demo_conditional()
    demo_bayes()
    demo_independence()
