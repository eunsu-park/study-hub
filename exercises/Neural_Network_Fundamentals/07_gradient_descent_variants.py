"""
07. Gradient Descent Variants - Exercises
==========================================
Lesson 07: Gradient Descent Variants

Exercises cover:
  1. Implement SGD, Momentum, Adam and compare on a quadratic
  2. Implement cosine annealing with warm restarts
"""

import numpy as np


# ============================================================
# Exercise 1: Optimizer Comparison on Quadratic
# Compare SGD, Momentum, Adam on f(x,y) = x^2 + 10*y^2.
# ============================================================
def exercise_1_optimizer_comparison():
    """Compare optimizers on an elongated quadratic."""
    print("=" * 60)
    print("Exercise 1: Optimizer Comparison")
    print("=" * 60)

    def f(params):
        x, y = params
        return x ** 2 + 10 * y ** 2

    def grad_f(params):
        x, y = params
        return np.array([2 * x, 20 * y])

    x0 = np.array([5.0, 5.0])

    # TODO: Implement SGD, SGD+Momentum, and Adam
    # Run each for 100 steps
    # Report final loss for each
    # Adam should converge fastest
    raise NotImplementedError("Implement and compare optimizers")


# ============================================================
# Exercise 2: Cosine Annealing with Warm Restarts
# Implement SGDR (cosine annealing with periodic restarts).
# ============================================================
def exercise_2_cosine_warm_restart():
    """Implement cosine annealing with warm restarts."""
    print("\n" + "=" * 60)
    print("Exercise 2: Cosine Annealing with Warm Restarts")
    print("=" * 60)

    def cosine_warm_restart(epoch, lr_max=0.1, lr_min=1e-6, T_0=10, T_mult=2):
        # TODO: Implement SGDR schedule
        # T_0 = initial cycle length
        # T_mult = cycle length multiplier after each restart
        # Within each cycle, use cosine annealing from lr_max to lr_min
        # After each cycle, restart at lr_max with cycle length *= T_mult
        raise NotImplementedError("Implement SGDR")

    # Test: print LR for epochs 0-50
    for e in range(51):
        lr = cosine_warm_restart(e)
        if e % 5 == 0:
            print(f"  Epoch {e:3d}: lr = {lr:.6f}")


if __name__ == "__main__":
    exercise_1_optimizer_comparison()
    exercise_2_cosine_warm_restart()
