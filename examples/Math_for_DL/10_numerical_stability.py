"""
Numerical Stability

Demonstrates stable implementations of common DL operations:
- Log-sum-exp trick
- Stable softmax and log-softmax
- Stable sigmoid and binary cross-entropy
- Catastrophic cancellation examples
- Dynamic loss scaling simulation

Dependencies: numpy
"""

import numpy as np


def log_sum_exp_demo():
    """Demonstrate the log-sum-exp trick."""
    print("=" * 60)
    print("LOG-SUM-EXP TRICK")
    print("=" * 60)

    def lse_naive(z):
        return np.log(np.sum(np.exp(z)))

    def lse_stable(z):
        c = np.max(z)
        return c + np.log(np.sum(np.exp(z - c)))

    cases = [
        ('Normal', np.array([1.0, 2.0, 3.0])),
        ('Large', np.array([1000.0, 1001.0, 1002.0])),
        ('Small', np.array([-1000.0, -999.0, -998.0])),
    ]
    for name, z in cases:
        naive = lse_naive(z)
        stable = lse_stable(z)
        print(f"  {name:6s}: naive={naive:12.4f}, stable={stable:12.4f}")


def stable_softmax_demo():
    """Stable softmax and log-softmax."""
    print("\n" + "=" * 60)
    print("STABLE SOFTMAX")
    print("=" * 60)

    z = np.array([1000.0, 1000.5, 999.0])

    # Naive
    try:
        e = np.exp(z)
        naive = e / e.sum()
    except:
        naive = np.array([np.nan, np.nan, np.nan])

    # Stable
    e = np.exp(z - np.max(z))
    stable = e / e.sum()

    print(f"Naive:  {naive}")
    print(f"Stable: {stable.round(4)}")


def stable_bce():
    """Numerically stable binary cross-entropy from logits."""
    print("\n" + "=" * 60)
    print("STABLE BINARY CROSS-ENTROPY")
    print("=" * 60)

    def bce_naive(z, y):
        p = 1 / (1 + np.exp(-z))
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    def bce_stable(z, y):
        return np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))

    z_vals = np.array([-100, -10, 0, 10, 100], dtype=np.float64)
    y_vals = np.array([0, 0, 1, 1, 1], dtype=np.float64)

    print("Naive: ", bce_naive(z_vals, y_vals))
    print("Stable:", bce_stable(z_vals, y_vals))


def catastrophic_cancellation():
    """Demonstrate catastrophic cancellation."""
    print("\n" + "=" * 60)
    print("CATASTROPHIC CANCELLATION")
    print("=" * 60)

    # (1 + eps) - 1
    eps = 1e-16
    result = (1.0 + eps) - 1.0
    print(f"(1 + 1e-16) - 1 = {result} (should be 1e-16)")

    # Variance: naive vs two-pass
    np.random.seed(42)
    data = np.random.randn(10000) * 0.001 + 1e6
    var_naive = np.mean(data**2) - np.mean(data)**2
    var_stable = np.var(data)
    print(f"Variance (naive):  {var_naive:.10e}")
    print(f"Variance (stable): {var_stable:.10e}")

    # Safe alternatives
    x_small = 1e-15
    print(f"\nlog(1+x): np.log(1+{x_small}) = {np.log(1+x_small)}, np.log1p = {np.log1p(x_small)}")
    print(f"exp(x)-1: np.exp({x_small})-1 = {np.exp(x_small)-1}, np.expm1 = {np.expm1(x_small)}")


def dynamic_loss_scaling():
    """Simulate dynamic loss scaling for mixed precision."""
    print("\n" + "=" * 60)
    print("DYNAMIC LOSS SCALING")
    print("=" * 60)

    S = 2**15
    scale_factor = 2
    n_ok = 0
    window = 5  # shortened for demo

    np.random.seed(42)
    for step in range(15):
        grad = np.random.randn() * 1e-5
        scaled = grad * S
        if np.isinf(np.float16(scaled)):
            S /= scale_factor
            n_ok = 0
            print(f"  Step {step:2d}: OVERFLOW, S -> {S:.0f}")
        else:
            n_ok += 1
            if n_ok >= window:
                S *= scale_factor
                n_ok = 0
                print(f"  Step {step:2d}: Scale UP, S -> {S:.0f}")
            else:
                print(f"  Step {step:2d}: OK, S = {S:.0f}")


if __name__ == "__main__":
    log_sum_exp_demo()
    stable_softmax_demo()
    stable_bce()
    catastrophic_cancellation()
    dynamic_loss_scaling()
