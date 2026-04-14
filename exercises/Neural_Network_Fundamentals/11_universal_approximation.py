"""
11. Universal Approximation - Exercises
=========================================
Lesson 11: Universal Approximation Theorem

Exercises cover:
  1. Bump-function approximation of x^2
  2. Width vs approximation quality
"""

import numpy as np


# ============================================================
# Exercise 1: Approximate x^2 with Bumps
# Use bump functions to approximate f(x) = x^2 on [0, 1].
# ============================================================
def exercise_1_approximate_x_squared():
    """Approximate x^2 using bump functions."""
    print("=" * 60)
    print("Exercise 1: Approximate x^2 with Bump Functions")
    print("=" * 60)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    # TODO: Create N bumps to approximate x^2 on [0, 1]
    # For each of N regions:
    #   center = (i + 0.5) / N
    #   width = 1 / N
    #   height = center^2  (sample x^2 at center)
    #   bump_i(x) = height * (sigmoid(S*(x-left)) - sigmoid(S*(x-right)))
    # Sum all bumps and measure max error
    # Try N = 5, 10, 20, 50

    x = np.linspace(0, 1, 1000)
    target = x ** 2

    for N in [5, 10, 20, 50]:
        # approx = ...
        # error = np.max(np.abs(target - approx))
        # print(f"  N={N:3d}: max error = {error:.6f}")
        pass

    raise NotImplementedError("Approximate x^2 with bumps")


# ============================================================
# Exercise 2: Width vs Error
# Train networks with different widths and measure error on sin(x).
# ============================================================
def exercise_2_width_vs_error():
    """Compare approximation quality for different widths."""
    print("\n" + "=" * 60)
    print("Exercise 2: Width vs Approximation Quality")
    print("=" * 60)

    # TODO: For N_neurons in [5, 10, 25, 50, 100]:
    # 1. Build network [1, N, 1] with ReLU
    # 2. Train on sin(x) for x in [-pi, pi]
    # 3. Report final MSE
    # Show that error decreases with more neurons
    raise NotImplementedError("Compare width vs approximation quality")


if __name__ == "__main__":
    exercise_1_approximate_x_squared()
    exercise_2_width_vs_error()
