"""
Exercises for Lesson 08: Information Theory
Topic: Math_for_DL

Complete the TODO sections.
"""

import numpy as np


def exercise_1_uniform_max_entropy():
    """Prove uniform maximizes entropy using Lagrange multipliers.

    Compute entropy for K=5 outcomes at various distributions
    and verify uniform gives the maximum.
    """
    K = 5

    # TODO: Compute entropy for several distributions and verify uniform is max
    distributions = {
        'uniform': np.ones(K) / K,
        'peaked': np.array([0.8, 0.05, 0.05, 0.05, 0.05]),
        'moderate': np.array([0.3, 0.25, 0.2, 0.15, 0.1]),
    }

    results = {}
    for name, p in distributions.items():
        # TODO: Compute entropy H(p) = -sum(p * log(p))
        H = None  # Replace
        results[name] = H

    return results


def exercise_2_bernoulli_kl():
    """Compute D_KL(Bernoulli(p) || Bernoulli(q)) for p=0.7 over a range of q."""
    p = 0.7
    q_values = np.linspace(0.01, 0.99, 100)

    # TODO: Compute KL for each q
    # D_KL = p*log(p/q) + (1-p)*log((1-p)/(1-q))
    kl_values = None  # Replace with array of length 100

    return q_values, kl_values


def exercise_3_mutual_information_gaussian():
    """Compute MI between jointly Gaussian variables with correlation rho.

    I(X;Y) = -0.5 * log(1 - rho^2)
    """
    rho_values = np.linspace(-0.99, 0.99, 100)

    # TODO: Compute MI for each rho
    mi_values = None  # Replace

    return rho_values, mi_values


if __name__ == "__main__":
    print("Exercise 1: Uniform max entropy")
    results = exercise_1_uniform_max_entropy()
    if results.get('uniform') is not None:
        for name, H in results.items():
            print(f"  {name:10s}: H = {H:.4f}")
        print(f"  Uniform is max: {results['uniform'] >= max(v for v in results.values())}")
    else:
        print("  Not implemented yet")

    print("\nExercise 2: Bernoulli KL")
    qs, kls = exercise_2_bernoulli_kl()
    if kls is not None:
        min_idx = np.argmin(kls)
        print(f"  KL minimized at q = {qs[min_idx]:.3f} (should be ~0.7)")
        print(f"  Min KL = {kls[min_idx]:.6f} (should be ~0)")
    else:
        print("  Not implemented yet")

    print("\nExercise 3: MI of jointly Gaussian")
    rhos, mis = exercise_3_mutual_information_gaussian()
    if mis is not None:
        print(f"  MI at rho=0: {mis[50]:.4f} (should be ~0)")
        print(f"  MI at rho=0.9: {mis[np.argmin(np.abs(rhos-0.9))]:.4f}")
    else:
        print("  Not implemented yet")
