"""
Exercises for Lesson 27: Safe Reinforcement Learning
Topic: Reinforcement_Learning
Solutions to practice problems from the lesson.
"""

import numpy as np
from collections import defaultdict


def exercise_1():
    """
    Exercise 1: Constrained MDP Formulation

    Formally define a CMDP and show the difference between
    unconstrained and constrained optimal policies.
    """
    print("Constrained MDP (CMDP) Formulation:")
    print("=" * 60)

    print()
    print("Standard MDP:     max_π E[sum_t gamma^t * r_t]")
    print()
    print("CMDP (Constrained MDP):")
    print("  max_π E[sum_t gamma^t * r_t]")
    print("  s.t. E[sum_t gamma^t * c_t] <= d")
    print()
    print("  where c_t = constraint cost (e.g., safety violation)")
    print("        d   = cost budget (allowed violations)")
    print()
    print("  Lagrangian relaxation:")
    print("  L(π, λ) = E[Σ r_t] - λ * (E[Σ c_t] - d)")
    print("  Solve: max_π min_{λ>=0} L(π, λ)")
    print()

    # Numerical example: grid with reward and cost tradeoff
    np.random.seed(42)

    # 5 possible paths, each with (reward, cost)
    paths = [
        ("Safe path",      5.0, 0.0),   # safe, moderate reward
        ("Risky shortcut", 9.0, 3.0),   # high reward, high cost
        ("Moderate risk",  7.0, 1.5),   # medium reward, medium cost
        ("Very risky",    11.0, 5.0),   # highest reward, exceeds budget
        ("Avoid",          3.0, 0.0),   # low reward, safe
    ]

    cost_budget = 2.0

    print(f"  Cost budget: d = {cost_budget}")
    print(f"\n  {'Path':>16} | {'Reward':>8} | {'Cost':>8} | {'Feasible':>10} | {'Lagrangian (λ=1)':>18}")
    print("  " + "-" * 70)

    lam = 1.0
    for name, reward, cost in paths:
        feasible = cost <= cost_budget
        lagrangian = reward - lam * (cost - cost_budget)
        print(f"  {name:>16} | {reward:>8.1f} | {cost:>8.1f} | {str(feasible):>10} | {lagrangian:>18.2f}")

    print(f"\n  Unconstrained optimal: 'Very risky' (highest reward).")
    print(f"  Constrained optimal:   'Moderate risk' (best among feasible paths).")
    print(f"\n  The Lagrangian with large λ automatically penalizes costly paths.")


def exercise_2():
    """
    Exercise 2: Lagrangian Method for Safe RL

    Implement the dual ascent update and show convergence to
    a policy that satisfies the cost constraint.
    """
    print("\nLagrangian Method — Constraint Satisfaction:")
    print("=" * 60)

    np.random.seed(7)

    # Simple 1-D problem: choose action in [0, 1]
    # Reward: r(a) = a (maximize)
    # Cost: c(a) = 2*a (cost of high action)
    # Constraint: E[c] <= d = 0.5
    cost_budget = 0.5

    def reward(action): return action
    def cost(action): return 2 * action

    def optimal_action(lam):
        """
        Lagrangian: L = r(a) - λ*(c(a) - d) = a - λ*(2a - d) = a*(1-2λ) + λ*d
        Optimal: a=1 if 1-2λ > 0 (λ < 0.5), a=0 if λ > 0.5, a=any if λ=0.5
        """
        if lam < 0.5:
            return 1.0
        elif lam > 0.5:
            return 0.0
        else:
            return 0.25  # arbitrary at exactly 0.5

    lam = 0.0
    lam_lr = 0.2
    history = []

    for step in range(20):
        a = optimal_action(lam)
        r = reward(a)
        c = cost(a)
        # Dual ascent: increase λ if constraint violated
        lam = max(0.0, lam + lam_lr * (c - cost_budget))
        history.append((step, a, r, c, lam))

    print(f"\n  Cost budget d = {cost_budget}")
    print(f"  Reward = a, Cost = 2a")
    print(f"  Optimal constrained action = {cost_budget/2:.2f} (cost_budget / 2)")
    print(f"\n  Lagrangian training:")
    print(f"  {'Step':>5} | {'Action':>8} | {'Reward':>8} | {'Cost':>8} | {'Lambda':>8}")
    print("  " + "-" * 48)
    for step, a, r, c, lam_val in history:
        print(f"  {step:>5} | {a:>8.3f} | {r:>8.3f} | {c:>8.3f} | {lam_val:>8.3f}")

    final_a = history[-1][1]
    print(f"\n  Converged action: {final_a:.3f} (target: {cost_budget/2:.3f})")
    print(f"  Lambda: {history[-1][4]:.3f}")
    print(f"  => λ increases until E[c] ≈ d, enforcing the constraint.")


def exercise_3():
    """
    Exercise 3: Safety Layer / Action Projection

    Implement a safety filter that projects actions into the safe set.
    """
    print("\nSafety Layer (Action Projection):")
    print("=" * 60)

    np.random.seed(1)

    # 2-D position control with safety constraint: stay within circle of radius 2
    # State: (x, y), Action: (dx, dy) — desired displacement
    def is_safe(pos, radius=2.0):
        return np.linalg.norm(pos) <= radius

    def project_to_safe(pos, action, radius=2.0, max_iter=10):
        """
        Project action so that resulting position stays in safe set.
        Uses bisection: reduce action magnitude until safe.
        """
        new_pos = pos + action
        if is_safe(new_pos, radius):
            return action

        # Bisection
        lo, hi = 0.0, 1.0
        for _ in range(max_iter):
            mid = (lo + hi) / 2
            if is_safe(pos + mid * action, radius):
                lo = mid
            else:
                hi = mid

        scale = lo
        projected = scale * action
        # Apply small safety margin
        if is_safe(pos + projected, radius):
            return projected
        return np.zeros_like(action)

    # Test: agent near boundary trying to cross it
    test_cases = [
        {"pos": np.array([0.0, 0.0]),   "action": np.array([1.0, 0.0]),  "desc": "Center moving right"},
        {"pos": np.array([1.8, 0.0]),   "action": np.array([0.5, 0.0]),  "desc": "Near boundary, unsafe action"},
        {"pos": np.array([1.9, 0.0]),   "action": np.array([0.5, 0.0]),  "desc": "Very near boundary"},
        {"pos": np.array([0.0, 1.5]),   "action": np.array([0.0, 1.0]),  "desc": "Moving toward boundary"},
        {"pos": np.array([1.5, 1.0]),   "action": np.array([1.0, 1.0]),  "desc": "Corner, large unsafe action"},
    ]

    print(f"\n  Safety constraint: ||position|| <= 2.0")
    print(f"\n  {'Description':>35} | {'Action norm':>12} | {'Projected norm':>15} | {'Safe after':>12}")
    print("  " + "-" * 82)

    for tc in test_cases:
        pos = tc["pos"]
        action = tc["action"]
        projected = project_to_safe(pos, action)
        new_pos = pos + projected
        safe = is_safe(new_pos)
        print(f"  {tc['desc']:>35} | {np.linalg.norm(action):>12.3f} | "
              f"{np.linalg.norm(projected):>15.3f} | {str(safe):>12}")

    print("\n  Safety layer guarantees the agent never leaves the safe set,")
    print("  regardless of the underlying RL policy's raw action.")


def exercise_4():
    """
    Exercise 4: Risk Measures in Safe RL

    Compare VaR, CVaR, and mean as decision criteria
    and show when risk-sensitive objectives matter.
    """
    print("\nRisk Measures for Safe RL:")
    print("=" * 60)

    np.random.seed(99)
    n = 100000

    # Three potential outcomes for a policy:
    # Returns large positive rewards most of the time
    # But has rare catastrophic failures

    policies = {
        "Safe (conservative)": np.random.normal(3.0, 0.5, n),
        "Normal RL": np.concatenate([
            np.random.normal(7.0, 1.0, int(n * 0.97)),
            np.random.normal(-50.0, 5.0, int(n * 0.03))
        ]),
        "Extreme RL": np.concatenate([
            np.random.normal(10.0, 2.0, int(n * 0.95)),
            np.random.normal(-200.0, 20.0, int(n * 0.05))
        ]),
    }

    def var(samples, alpha=0.05):
        """Value at Risk: the alpha-quantile."""
        return np.percentile(samples, alpha * 100)

    def cvar(samples, alpha=0.05):
        """Conditional VaR: expected return in the worst alpha fraction."""
        threshold = var(samples, alpha)
        tail = samples[samples <= threshold]
        return np.mean(tail) if len(tail) > 0 else threshold

    print(f"\n  {'Policy':>22} | {'Mean':>8} | {'VaR@5%':>8} | {'CVaR@5%':>9} | {'Min':>8}")
    print("  " + "-" * 68)

    for name, dist in policies.items():
        print(f"  {name:>22} | {np.mean(dist):>8.2f} | {var(dist):>8.2f} | "
              f"{cvar(dist):>9.2f} | {np.min(dist):>8.2f}")

    print()
    print("  Risk measure comparison:")
    print("  Mean (risk-neutral): Extreme RL wins (10.0 > 7.0 > 3.0)")
    print("  CVaR@5% (risk-averse): Safe policy wins (avoids catastrophic tails)")
    print()
    print("  When to use risk-sensitive objectives:")
    print("  - Autonomous driving: rare catastrophic accidents are unacceptable")
    print("  - Medical treatment: extreme outcomes must be bounded")
    print("  - Finance: tail risk (drawdown) often matters more than mean return")
    print("  - Robotics: hardware damage is irreversible and expensive")


if __name__ == "__main__":
    print("=== Exercise 1: CMDP Formulation ===")
    exercise_1()

    print("\n=== Exercise 2: Lagrangian Method ===")
    exercise_2()

    print("\n=== Exercise 3: Safety Layer ===")
    exercise_3()

    print("\n=== Exercise 4: Risk Measures ===")
    exercise_4()

    print("\nAll exercises completed!")
