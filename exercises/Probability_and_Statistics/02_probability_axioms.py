"""
Probability and Statistics — Probability Axioms
Exercises covering the addition rule, complement rule, Bayes' theorem,
and independence of events.
"""
import random
from typing import Set, Tuple


# === Exercise 1: Verify the Addition Rule for Dice Events ===
def exercise_1() -> None:
    """Verify P(A ∪ B) = P(A) + P(B) - P(A ∩ B) for two events defined
    on a fair six-sided die.  A = {even outcome}, B = {outcome >= 4}."""
    print("=== Exercise 1: Verify P(A∪B) = P(A) + P(B) - P(A∩B) ===")

    sample_space: Set[int] = {1, 2, 3, 4, 5, 6}
    n = len(sample_space)

    A: Set[int] = {2, 4, 6}         # even
    B: Set[int] = {4, 5, 6}         # >= 4

    A_union_B = A | B
    A_inter_B = A & B

    P_A = len(A) / n
    P_B = len(B) / n
    P_A_union_B = len(A_union_B) / n
    P_A_inter_B = len(A_inter_B) / n

    rhs = P_A + P_B - P_A_inter_B

    print(f"  Sample space: {sorted(sample_space)}")
    print(f"  A (even):      {sorted(A)}   => P(A) = {P_A:.4f}")
    print(f"  B (>= 4):      {sorted(B)}   => P(B) = {P_B:.4f}")
    print(f"  A ∩ B:         {sorted(A_inter_B)}      => P(A∩B) = {P_A_inter_B:.4f}")
    print(f"  A ∪ B:         {sorted(A_union_B)}")
    print(f"  P(A∪B)         = {P_A_union_B:.4f}")
    print(f"  P(A)+P(B)-P(A∩B) = {P_A:.4f} + {P_B:.4f} - {P_A_inter_B:.4f} = {rhs:.4f}")

    assert abs(P_A_union_B - rhs) < 1e-12, "Addition rule failed!"
    print("  Addition rule verified.\n")


# === Exercise 2: Complement Rule — At Least One 6 in 4 Rolls ===
def exercise_2() -> None:
    """Compute P(at least one 6 in 4 independent rolls) using the complement
    rule and verify by simulation."""
    print("=== Exercise 2: P(at least one 6 in 4 rolls) ===")

    n_rolls = 4
    p_no_six_single = 5 / 6

    # Complement rule: 1 - P(no six in any roll)
    p_no_six_all = p_no_six_single ** n_rolls
    p_at_least_one = 1 - p_no_six_all

    print(f"  P(no 6 on single roll) = 5/6 = {p_no_six_single:.6f}")
    print(f"  P(no 6 in {n_rolls} rolls)   = (5/6)^{n_rolls} = {p_no_six_all:.6f}")
    print(f"  P(at least one 6)     = 1 - {p_no_six_all:.6f} = {p_at_least_one:.6f}")

    # Simulation
    random.seed(42)
    n_simulations = 100_000
    successes = 0
    for _ in range(n_simulations):
        rolls = [random.randint(1, 6) for _ in range(n_rolls)]
        if 6 in rolls:
            successes += 1
    p_simulated = successes / n_simulations

    print(f"  Simulation ({n_simulations:,} trials): {p_simulated:.6f}")
    print(f"  Difference: {abs(p_at_least_one - p_simulated):.6f}\n")


# === Exercise 3: Bayes' Theorem — Disease Testing ===
def exercise_3() -> None:
    """Apply Bayes' theorem to compute the probability that a person actually
    has a disease given a positive test result.
    Prevalence = 2%, Sensitivity = 95%, Specificity = 90%."""
    print("=== Exercise 3: Bayes' Theorem — Disease Testing ===")

    prevalence = 0.02       # P(D)
    sensitivity = 0.95      # P(+|D)  — true positive rate
    specificity = 0.90      # P(-|~D) — true negative rate

    p_disease = prevalence
    p_no_disease = 1 - prevalence
    p_pos_given_disease = sensitivity
    p_pos_given_no_disease = 1 - specificity  # false positive rate

    # Total probability of positive test
    p_pos = (p_pos_given_disease * p_disease
             + p_pos_given_no_disease * p_no_disease)

    # Bayes' theorem: P(D|+) = P(+|D) * P(D) / P(+)
    p_disease_given_pos = (p_pos_given_disease * p_disease) / p_pos

    print(f"  Prevalence P(D)          = {prevalence}")
    print(f"  Sensitivity P(+|D)       = {sensitivity}")
    print(f"  Specificity P(-|~D)      = {specificity}")
    print(f"  False positive P(+|~D)   = {p_pos_given_no_disease}")
    print(f"  P(+) = P(+|D)P(D) + P(+|~D)P(~D)")
    print(f"       = {p_pos_given_disease}*{p_disease} + "
          f"{p_pos_given_no_disease}*{p_no_disease}")
    print(f"       = {p_pos:.4f}")
    print(f"  P(D|+) = P(+|D)*P(D) / P(+)")
    print(f"         = {p_pos_given_disease}*{p_disease} / {p_pos:.4f}")
    print(f"         = {p_disease_given_pos:.4f}")
    print(f"  Despite the positive test, only {p_disease_given_pos*100:.1f}% "
          f"chance of having the disease.\n")


# === Exercise 4: Check Independence for Card Events ===
def exercise_4() -> None:
    """Check whether two events on a standard 52-card deck are independent.
    Event A: card is a Heart.  Event B: card is a face card (J, Q, K).
    Two events are independent iff P(A∩B) = P(A) * P(B)."""
    print("=== Exercise 4: Independence Check on a Deck of Cards ===")

    total = 52

    # Build deck as (rank, suit) tuples
    suits = ["Hearts", "Diamonds", "Clubs", "Spades"]
    ranks = list(range(2, 11)) + ["J", "Q", "K", "A"]
    deck: list[Tuple[object, str]] = [(r, s) for s in suits for r in ranks]

    assert len(deck) == total

    # Event A: Hearts
    A = {card for card in deck if card[1] == "Hearts"}
    # Event B: Face card (J, Q, K)
    B = {card for card in deck if card[0] in ("J", "Q", "K")}
    A_inter_B = A & B

    P_A = len(A) / total
    P_B = len(B) / total
    P_A_inter_B = len(A_inter_B) / total
    product = P_A * P_B

    print(f"  |Deck| = {total}")
    print(f"  A (Hearts):     {len(A)} cards => P(A) = {P_A:.6f}")
    print(f"  B (Face cards): {len(B)} cards => P(B) = {P_B:.6f}")
    print(f"  A ∩ B (Heart face cards): {len(A_inter_B)} cards "
          f"=> P(A∩B) = {P_A_inter_B:.6f}")
    print(f"  P(A) * P(B) = {P_A:.6f} * {P_B:.6f} = {product:.6f}")

    independent = abs(P_A_inter_B - product) < 1e-12
    print(f"  P(A∩B) {'==' if independent else '!='} P(A)*P(B) => "
          f"Events are {'independent' if independent else 'NOT independent'}.")
    print()


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
