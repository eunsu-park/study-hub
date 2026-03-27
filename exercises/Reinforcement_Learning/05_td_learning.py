"""
Exercises for Lesson 05: Temporal Difference Learning
Topic: Reinforcement_Learning
Solutions to practice problems from the lesson.
"""

import numpy as np
from collections import defaultdict


def exercise_1():
    """
    TD Error: What is the TD error when V(s)=5, R=1, gamma=0.9, V(s')=6?

    delta = R + gamma * V(s') - V(s)
    """
    V_s = 5
    R = 1
    gamma = 0.9
    V_s_prime = 6

    td_target = R + gamma * V_s_prime
    td_error = td_target - V_s

    print(f"Given:")
    print(f"  V(s) = {V_s}")
    print(f"  R = {R}")
    print(f"  gamma = {gamma}")
    print(f"  V(s') = {V_s_prime}")
    print()

    print(f"TD Target = R + gamma * V(s')")
    print(f"         = {R} + {gamma} * {V_s_prime}")
    print(f"         = {R} + {gamma * V_s_prime}")
    print(f"         = {td_target}")
    print()

    print(f"TD Error (delta) = TD Target - V(s)")
    print(f"                 = {td_target} - {V_s}")
    print(f"                 = {td_error}")
    print()

    print(f"Interpretation:")
    if td_error > 0:
        print(f"  delta = {td_error} > 0: The outcome was BETTER than expected.")
        print(f"  V(s) should be INCREASED.")
    elif td_error < 0:
        print(f"  delta = {td_error} < 0: The outcome was WORSE than expected.")
        print(f"  V(s) should be DECREASED.")
    else:
        print(f"  delta = 0: The outcome matched expectations exactly.")

    # Show the update
    alpha = 0.1
    V_s_new = V_s + alpha * td_error
    print(f"\n  With alpha={alpha}:")
    print(f"  V(s) <- V(s) + alpha * delta")
    print(f"  V(s) <- {V_s} + {alpha} * {td_error}")
    print(f"  V(s) <- {V_s_new}")


def exercise_2():
    """
    n-step: Write the return formula for n=2.

    G_t^(2) = R_{t+1} + gamma * R_{t+2} + gamma^2 * V(S_{t+2})
    """
    print("2-step Return Formula:")
    print("=" * 50)
    print()
    print("General n-step return:")
    print("  G_t^(n) = R_{t+1} + gamma*R_{t+2} + ... + gamma^{n-1}*R_{t+n}")
    print("            + gamma^n * V(S_{t+n})")
    print()
    print("For n = 2:")
    print("  G_t^(2) = R_{t+1} + gamma * R_{t+2} + gamma^2 * V(S_{t+2})")
    print()
    print("This combines 2 actual rewards with a bootstrapped estimate")
    print("of the remaining value from state S_{t+2}.")
    print()

    # Numerical example
    R_t1 = 3.0   # Reward at t+1
    R_t2 = -1.0  # Reward at t+2
    V_st2 = 10.0 # Value of state at t+2
    gamma = 0.9

    G_2step = R_t1 + gamma * R_t2 + gamma**2 * V_st2
    print(f"Numerical example:")
    print(f"  R_{{t+1}} = {R_t1}, R_{{t+2}} = {R_t2}, V(S_{{t+2}}) = {V_st2}, gamma = {gamma}")
    print()
    print(f"  G_t^(2) = {R_t1} + {gamma} * {R_t2} + {gamma}^2 * {V_st2}")
    print(f"          = {R_t1} + {gamma * R_t2} + {gamma**2 * V_st2}")
    print(f"          = {G_2step}")
    print()

    # Compare with 1-step and MC (full return)
    print("Comparison of n-step returns (same example):")
    V_st1 = 8.0  # Hypothetical value of state at t+1

    G_1step = R_t1 + gamma * V_st1
    print(f"  1-step (TD(0)): G_t^(1) = {R_t1} + {gamma}*{V_st1} = {G_1step}")
    print(f"  2-step:         G_t^(2) = {G_2step}")

    R_t3 = 2.0
    G_mc = R_t1 + gamma * R_t2 + gamma**2 * R_t3  # Simplified MC (3-step episode)
    print(f"  MC (3 steps):   G_t^MC  = {R_t1} + {gamma}*{R_t2} + {gamma**2}*{R_t3} = {G_mc}")
    print()
    print("Note: As n increases, we use more real rewards (less bias)")
    print("but the estimate has more variance due to more random variables.")


def exercise_3():
    """
    TD(lambda): What is the weight ratio of 1-step and 2-step returns
    when lambda=0.5?

    The lambda-return weights n-step returns by (1-lambda) * lambda^{n-1}
    """
    lambd = 0.5

    print(f"TD(lambda) with lambda = {lambd}:")
    print("=" * 50)
    print()
    print("The lambda-return is a weighted average of all n-step returns:")
    print("  G_t^lambda = (1-lambda) * sum_{n=1}^{inf} lambda^{n-1} * G_t^{(n)}")
    print()
    print("Weight for n-step return:")
    print("  w(n) = (1-lambda) * lambda^{n-1}")
    print()

    # Calculate weights
    w_1step = (1 - lambd) * lambd**(1-1)
    w_2step = (1 - lambd) * lambd**(2-1)

    print(f"For lambda = {lambd}:")
    print(f"  Weight of 1-step return: w(1) = (1-{lambd}) * {lambd}^0 = {w_1step}")
    print(f"  Weight of 2-step return: w(2) = (1-{lambd}) * {lambd}^1 = {w_2step}")
    print()
    print(f"  Ratio w(1)/w(2) = {w_1step}/{w_2step} = {w_1step/w_2step}")
    print()

    # Show all weights sum to 1
    print("All weights (first 10 terms):")
    total_weight = 0
    for n in range(1, 11):
        w = (1 - lambd) * lambd**(n-1)
        total_weight += w
        print(f"  w({n:2d}) = {w:.6f}  (cumulative: {total_weight:.6f})")

    print(f"\n  Sum of infinite weights = (1-lambda) * sum lambda^{n-1}")
    print(f"                         = (1-lambda) / (1-lambda) = 1.0")
    print()

    # Compare different lambda values
    print("Weight ratios for different lambda values:")
    print(f"{'lambda':>8} {'w(1)':>8} {'w(2)':>8} {'w(1)/w(2)':>10}")
    for l in [0.0, 0.3, 0.5, 0.8, 0.9, 0.95]:
        w1 = (1 - l) * l**(0)
        w2 = (1 - l) * l**(1)
        ratio = w1 / w2 if w2 > 0 else float('inf')
        print(f"{l:>8.2f} {w1:>8.4f} {w2:>8.4f} {ratio:>10.2f}")

    print()
    print("Insight: lambda=0 puts ALL weight on 1-step (pure TD(0)).")
    print("As lambda -> 1, weights become more uniform (approaching MC).")


def exercise_4():
    """
    Eligibility Trace: What is the value of the accumulating trace if state s
    is visited 2 times consecutively?

    With accumulating trace: E(s) <- E(s) + 1 on each visit
    Between visits: E(s) <- gamma * lambda * E(s)
    """
    gamma = 0.9
    lambd = 0.8

    print("Accumulating Eligibility Trace for State s Visited Twice:")
    print("=" * 60)
    print(f"  gamma = {gamma}, lambda = {lambd}")
    print()
    print("Accumulating trace update rule:")
    print("  At each timestep: E(s) <- gamma * lambda * E(s)")
    print("  When s is visited: E(s) <- E(s) + 1")
    print()

    # Scenario: s is visited at t=0, then again at t=1 (consecutive)
    print("Scenario: s visited at t=0 and t=1 (consecutively)")
    print()

    E_s = 0.0
    print(f"  Before t=0: E(s) = {E_s:.4f}")

    # Visit at t=0
    E_s += 1  # Visit increment
    print(f"  After visit at t=0: E(s) = {E_s:.4f}")

    # Decay between t=0 and t=1
    E_s *= gamma * lambd
    print(f"  After decay (gamma*lambda): E(s) = {gamma}*{lambd}*1 = {E_s:.4f}")

    # Visit at t=1
    E_s += 1  # Visit increment
    print(f"  After visit at t=1: E(s) = {E_s:.4f}")
    print()
    print(f"Answer: E(s) = gamma*lambda + 1 = {gamma}*{lambd} + 1 = {gamma * lambd + 1:.4f}")
    print()

    # Extended example: trace decay over time
    print("Extended example: trace evolution over 10 timesteps")
    print("(s visited at t=0, t=1, and t=5)")
    print("-" * 50)

    E_s = 0.0
    visit_times = {0, 1, 5}

    for t in range(10):
        # Decay first (except at t=0)
        if t > 0:
            E_s *= gamma * lambd

        # Visit?
        visited = t in visit_times
        if visited:
            E_s += 1

        status = " <-- visited" if visited else ""
        print(f"  t={t}: E(s) = {E_s:.6f}{status}")

    print()
    print("Key observations:")
    print("  1. Each visit adds 1 to the trace (accumulating).")
    print("  2. The trace decays by gamma*lambda at each step.")
    print("  3. Recent visits contribute more than old visits.")
    print("  4. The trace captures both recency and frequency of visits.")

    # Compare with replacing trace
    print()
    print("Comparison with Replacing Trace:")
    E_acc = 0.0
    E_rep = 0.0

    for t in range(10):
        if t > 0:
            E_acc *= gamma * lambd
            E_rep *= gamma * lambd

        if t in visit_times:
            E_acc += 1       # Accumulating
            E_rep = 1        # Replacing

        if t in visit_times or t in {2, 6}:  # Show key timesteps
            print(f"  t={t}: Accumulating={E_acc:.4f}, Replacing={E_rep:.4f}")


if __name__ == "__main__":
    print("=== Exercise 1: TD Error ===")
    exercise_1()

    print("\n=== Exercise 2: n-step Return ===")
    exercise_2()

    print("\n=== Exercise 3: TD(lambda) Weights ===")
    exercise_3()

    print("\n=== Exercise 4: Eligibility Trace ===")
    exercise_4()

    print("\nAll exercises completed!")
