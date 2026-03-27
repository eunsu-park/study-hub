"""
Exercises for Lesson 18: Distributional Reinforcement Learning
Topic: Reinforcement_Learning
Solutions to practice problems from the lesson.
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: Return Distribution vs Expected Return

    Given two state-action pairs with the same expected return,
    explain why distributional RL can distinguish them.
    """
    print("Return Distribution vs Expected Return:")
    print("=" * 60)

    np.random.seed(42)
    n = 50000

    # Z_A: Gaussian, mean=5, std=0.5 (safe, predictable)
    Z_A = np.random.normal(5.0, 0.5, n)
    # Z_B: bimodal (0 or 10 with equal prob), mean=5 (risky)
    Z_B = np.random.choice([0.0, 10.0], size=n)

    print(f"\n  Z_A ~ N(5, 0.5):         E[Z_A] = {np.mean(Z_A):.3f}")
    print(f"  Z_B ~ Bernoulli(0, 10):  E[Z_B] = {np.mean(Z_B):.3f}")
    print(f"\n  Both have the same expected return (~5).")
    print(f"\n  But distributional RL captures more information:")

    for label, Z in [("Z_A", Z_A), ("Z_B", Z_B)]:
        std = np.std(Z)
        cvar5 = np.mean(Z[Z <= np.percentile(Z, 5)])
        cvar95 = np.mean(Z[Z >= np.percentile(Z, 95)])
        print(f"\n  {label}:")
        print(f"    Std = {std:.4f}")
        print(f"    CVaR@5%  (worst 5%) = {cvar5:.4f}")
        print(f"    CVaR@95% (best  5%) = {cvar95:.4f}")

    print("\n  A risk-averse agent should pick Z_A (much smaller downside).")
    print("  Scalar Q-learning CANNOT make this distinction.")


def exercise_2():
    """
    Exercise 2: Categorical Distribution Projection

    Project the shifted distribution r + gamma * Z_target onto
    the fixed support [V_min, V_max] with N atoms.
    """
    print("\nCategorical Projection (C51 Style):")
    print("=" * 60)

    V_min, V_max, N = -10, 10, 11
    support = np.linspace(V_min, V_max, N)
    delta_z = (V_max - V_min) / (N - 1)

    # Target distribution: uniform over atoms (each atom has prob 1/N)
    probs_target = np.ones(N) / N
    print(f"\n  Support: {support}")
    print(f"  Target probs: uniform (each = {1/N:.3f})")

    # Projection for reward=1.0, gamma=0.9
    reward, gamma = 1.0, 0.9
    projected_probs = np.zeros(N)

    for j in range(N):
        # Shift atom j of the target
        tz = np.clip(reward + gamma * support[j], V_min, V_max)
        b = (tz - V_min) / delta_z
        l = int(np.floor(b))
        u = int(np.ceil(b))
        if l == u:
            projected_probs[l] += probs_target[j]
        else:
            projected_probs[l] += probs_target[j] * (u - b)
            projected_probs[u] += probs_target[j] * (b - l)

    print(f"\n  After Bellman projection (r={reward}, gamma={gamma}):")
    print(f"  Projected probs: {projected_probs.round(4)}")

    # Sanity checks
    assert abs(projected_probs.sum() - 1.0) < 1e-8, "Probabilities must sum to 1"
    mean_target = np.dot(probs_target, support)
    mean_projected = np.dot(projected_probs, support)
    expected_mean = reward + gamma * mean_target

    print(f"\n  Sanity checks:")
    print(f"    Sum of projected probs: {projected_probs.sum():.6f} (expected 1.0)")
    print(f"    Mean of target dist:    {mean_target:.4f}")
    print(f"    Projected dist mean:    {mean_projected:.4f}")
    print(f"    Expected (r+g*mean):    {expected_mean:.4f}")
    print(f"    Mean preserved: {abs(mean_projected - expected_mean) < 0.5}")


def exercise_3():
    """
    Exercise 3: Quantile Regression Loss Properties

    Show that minimizing quantile regression loss gives
    the correct quantile estimate.
    """
    print("\nQuantile Regression Loss:")
    print("=" * 60)

    np.random.seed(0)
    # Distribution: 70% chance of -1, 30% chance of +5
    samples = np.random.choice([-1.0, 5.0], size=20000, p=[0.7, 0.3])

    # True quantiles
    true_q10 = np.quantile(samples, 0.1)
    true_q50 = np.quantile(samples, 0.5)
    true_q90 = np.quantile(samples, 0.9)

    print(f"\n  Distribution: 70% -> -1, 30% -> +5")
    print(f"  True 10th percentile: {true_q10:.3f}")
    print(f"  True 50th percentile: {true_q50:.3f}")
    print(f"  True 90th percentile: {true_q90:.3f}")

    def qr_loss(tau, prediction, targets):
        """Pinball (quantile regression) loss."""
        errors = targets - prediction
        return np.mean(np.where(errors >= 0, tau * errors, (tau - 1) * errors))

    def find_quantile(tau, n_iter=2000, lr=0.05):
        """Gradient descent to minimize quantile loss."""
        theta = 0.0
        for _ in range(n_iter):
            errors = samples - theta
            grad = np.mean(np.where(errors >= 0, -tau, -(tau - 1)))
            theta -= lr * grad
        return theta

    print(f"\n  Learned quantiles via gradient descent:")
    for tau in [0.1, 0.5, 0.9]:
        learned = find_quantile(tau)
        true_q = np.quantile(samples, tau)
        print(f"  tau={tau}: learned={learned:.4f}, true={true_q:.4f}, "
              f"error={abs(learned-true_q):.4f}")


def exercise_4():
    """
    Exercise 4: Risk-Sensitive Action Selection with CVaR

    Compare risk-neutral (mean) vs risk-averse (CVaR) action selection.
    """
    print("\nRisk-Sensitive Action Selection:")
    print("=" * 60)

    np.random.seed(99)
    n = 20000

    # Three actions with different return distributions
    dist_A = np.random.normal(4.0, 0.3, n)          # safe, moderate
    dist_B = np.random.normal(6.0, 2.0, n)           # high mean, moderate risk
    dist_C = np.where(np.random.random(n) < 0.9,     # usually good, sometimes disaster
                      np.random.normal(7.0, 1.0, n),
                      np.random.normal(-30.0, 3.0, n))

    def metrics(dist, name):
        mean = np.mean(dist)
        std = np.std(dist)
        cvar5 = np.mean(dist[dist <= np.percentile(dist, 5)])
        cvar10 = np.mean(dist[dist <= np.percentile(dist, 10)])
        print(f"  Action {name}:")
        print(f"    Mean = {mean:.3f}, Std = {std:.3f}")
        print(f"    CVaR@5% = {cvar5:.3f}, CVaR@10% = {cvar10:.3f}")

    for label, dist in [("A (safe)", dist_A),
                         ("B (moderate risk)", dist_B),
                         ("C (high risk)", dist_C)]:
        metrics(dist, label)
        print()

    print("  Risk-neutral ranking (by mean):  C > B > A")
    print("  Risk-averse ranking (by CVaR@5%): A > B >> C")
    print("  => Distributional RL enables adaptive risk preferences.")


if __name__ == "__main__":
    print("=== Exercise 1: Distribution vs Expected Return ===")
    exercise_1()

    print("\n=== Exercise 2: Categorical Projection ===")
    exercise_2()

    print("\n=== Exercise 3: Quantile Regression ===")
    exercise_3()

    print("\n=== Exercise 4: Risk-Sensitive Selection ===")
    exercise_4()

    print("\nAll exercises completed!")
