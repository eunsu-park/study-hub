"""
Distributional Reinforcement Learning — Example Code

Demonstrates:
  1. Return distribution tracking (categorical / histogram)
  2. C51-style categorical projection (toy environment)
  3. Quantile Regression loss vs MSE
  4. Risk-sensitive action selection (CVaR)

No PyTorch/gym required — uses NumPy only.
"""

import numpy as np


# ============================================================
# 1. Why Distributional RL? — Two Machines
# ============================================================

def demonstrate_same_mean_different_dist():
    """
    Two options have the same expected value but different risk.
    Traditional Q-learning cannot distinguish them.
    """
    print("=" * 60)
    print("1. Same Expected Value, Different Distributions")
    print("=" * 60)

    rng = np.random.RandomState(42)
    n_samples = 10000

    # Machine A: always pays 5
    samples_A = np.full(n_samples, 5.0)

    # Machine B: pays 0 or 10 with equal probability
    samples_B = rng.choice([0.0, 10.0], size=n_samples)

    for name, samples in [("A (constant $5)", samples_A),
                           ("B ($0 or $10)", samples_B)]:
        mean = np.mean(samples)
        std = np.std(samples)
        p5 = np.percentile(samples, 5)    # 5th percentile (worst 5%)
        p95 = np.percentile(samples, 95)  # 95th percentile (best 5%)
        print(f"\n  Machine {name}:")
        print(f"    E[R] = {mean:.2f}  (same!)")
        print(f"    Std  = {std:.2f}")
        print(f"    5th  = {p5:.2f}  95th = {p95:.2f}")

    print("\n  => Risk-averse agent prefers A, risk-seeking may prefer B.")
    print("     Traditional Q-learning (scalar) cannot tell the difference.")


# ============================================================
# 2. Categorical Return Distribution (C51-style)
# ============================================================

class CategoricalDist:
    """
    Represents a return distribution as a categorical distribution
    over a fixed support [V_min, V_max] with N_atoms atoms.

    This mimics the C51 approach without neural networks.
    """

    def __init__(self, v_min=-10, v_max=10, n_atoms=51):
        self.v_min = v_min
        self.v_max = v_max
        self.n_atoms = n_atoms
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        self.support = np.linspace(v_min, v_max, n_atoms)
        # Start with uniform distribution
        self.probs = np.ones(n_atoms) / n_atoms

    def mean(self):
        return np.dot(self.probs, self.support)

    def cvar(self, alpha=0.05):
        """
        Conditional Value at Risk at level alpha.
        Average return over the worst alpha fraction of outcomes.
        Risk-averse criterion: minimize CVaR.
        """
        cumprob = np.cumsum(self.probs)
        cutoff_idx = np.searchsorted(cumprob, alpha)
        # Weighted average of atoms below threshold
        tail_probs = self.probs[:cutoff_idx + 1].copy()
        tail_probs[-1] = alpha - (cumprob[cutoff_idx - 1] if cutoff_idx > 0 else 0)
        tail_probs = np.clip(tail_probs, 0, None)
        total = tail_probs.sum()
        if total < 1e-10:
            return self.support[0]
        return np.dot(tail_probs / total, self.support[:cutoff_idx + 1])

    def project(self, reward, gamma, target_dist):
        """
        Bellman projection: Z' = r + gamma * Z_target
        Projects target distribution onto current support.
        """
        new_probs = np.zeros(self.n_atoms)
        for j, (p, z) in enumerate(zip(target_dist.probs, target_dist.support)):
            # Shift atom
            tz = np.clip(reward + gamma * z, self.v_min, self.v_max)
            # Find bounding atoms
            b = (tz - self.v_min) / self.delta_z
            l = int(np.floor(b))
            u = int(np.ceil(b))
            # Distribute probability
            if l == u:
                new_probs[l] += p
            else:
                new_probs[l] += p * (u - b)
                new_probs[u] += p * (b - l)
        self.probs = new_probs


def demonstrate_categorical_projection():
    """Show how C51 projects the Bellman distribution onto the support."""
    print("\n" + "=" * 60)
    print("2. Categorical Distribution Projection (C51)")
    print("=" * 60)

    target = CategoricalDist(v_min=-10, v_max=10, n_atoms=11)
    # Set target to a bimodal distribution: mass at -5 and +5
    target.probs = np.zeros(11)
    target.probs[2] = 0.5   # atom at -6
    target.probs[8] = 0.5   # atom at +6
    target.probs /= target.probs.sum()

    print(f"\n  Target distribution mean: {target.mean():.2f}")
    print(f"  Support: {target.support}")
    print(f"  Probs: {target.probs.round(3)}")

    # Project: Z' = 1.0 + 0.9 * Z_target
    predicted = CategoricalDist(v_min=-10, v_max=10, n_atoms=11)
    reward = 1.0
    gamma = 0.9
    predicted.project(reward, gamma, target)

    print(f"\n  After Bellman projection (r={reward}, gamma={gamma}):")
    print(f"  Projected mean:  {predicted.mean():.2f}")
    print(f"  Expected mean:   {reward + gamma * target.mean():.2f}")
    print(f"  Projected probs: {predicted.probs.round(3)}")


# ============================================================
# 3. Quantile Regression Loss
# ============================================================

def quantile_huber_loss(quantile_tau, error, kappa=1.0):
    """
    Quantile Huber loss as used in QR-DQN.

    Args:
        quantile_tau: Target quantile in (0, 1)
        error: (predicted_quantile - target)
        kappa: Huber loss threshold
    """
    abs_error = np.abs(error)
    huber = np.where(abs_error <= kappa,
                     0.5 * error**2,
                     kappa * (abs_error - 0.5 * kappa))
    indicator = (error < 0).astype(float)
    return np.abs(quantile_tau - indicator) * huber


def demonstrate_quantile_loss():
    """Compare quantile regression vs MSE for learning quantiles."""
    print("\n" + "=" * 60)
    print("3. Quantile Regression vs MSE")
    print("=" * 60)

    rng = np.random.RandomState(0)
    # Asymmetric distribution: 80% chance of -1, 20% chance of +10
    samples = rng.choice([-1.0, 10.0], size=5000, p=[0.8, 0.2])

    # True quantiles
    q10 = np.quantile(samples, 0.10)
    q50 = np.quantile(samples, 0.50)
    q90 = np.quantile(samples, 0.90)
    mean = np.mean(samples)

    print(f"\n  Distribution: 80% chance -1, 20% chance +10")
    print(f"  Mean (MSE target):          {mean:.3f}")
    print(f"  True 10th percentile:       {q10:.3f}")
    print(f"  True 50th percentile:       {q50:.3f}")
    print(f"  True 90th percentile:       {q90:.3f}")

    # Gradient descent to find quantile estimates
    for tau in [0.1, 0.5, 0.9]:
        estimate = 0.0  # initial estimate
        lr = 0.05
        for _ in range(2000):
            errors = samples - estimate
            grad = np.mean(np.where(errors >= 0, tau, tau - 1.0))
            estimate += lr * grad

        true_q = np.quantile(samples, tau)
        print(f"\n  tau={tau}: learned={estimate:.3f}, true={true_q:.3f}")

    print("\n  => Quantile regression recovers full return distribution,")
    print("     not just the mean — crucial for risk-sensitive decisions.")


# ============================================================
# 4. Risk-Sensitive Action Selection
# ============================================================

def risk_sensitive_selection():
    """
    Compare risk-neutral (mean) vs risk-averse (CVaR) action selection.
    """
    print("\n" + "=" * 60)
    print("4. Risk-Sensitive Action Selection")
    print("=" * 60)

    rng = np.random.RandomState(99)
    n_samples = 10000

    # Action distributions
    # A: safe — Gaussian centered at 5 with small variance
    dist_A = rng.normal(5.0, 0.5, n_samples)
    # B: risky — bimodal: often 8 but sometimes -20
    dist_B = np.where(rng.random(n_samples) < 0.9,
                      rng.normal(8.0, 1.0, n_samples),
                      rng.normal(-20.0, 2.0, n_samples))

    for name, dist in [("A (safe)", dist_A), ("B (risky)", dist_B)]:
        mean = np.mean(dist)
        cvar5 = np.mean(dist[dist <= np.percentile(dist, 5)])
        cvar10 = np.mean(dist[dist <= np.percentile(dist, 10)])
        print(f"\n  Action {name}:")
        print(f"    Mean (risk-neutral criterion):   {mean:.2f}")
        print(f"    CVaR@5%  (risk-averse criterion): {cvar5:.2f}")
        print(f"    CVaR@10% (risk-averse criterion): {cvar10:.2f}")

    print("\n  Risk-neutral agent picks B (higher mean).")
    print("  Risk-averse agent picks A (much better CVaR — avoids catastrophe).")
    print("  Distributional RL enables this choice; scalar Q-learning cannot.")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    demonstrate_same_mean_different_dist()
    demonstrate_categorical_projection()
    demonstrate_quantile_loss()
    risk_sensitive_selection()

    print("\n" + "=" * 60)
    print("Distributional RL examples complete!")
    print("=" * 60)
