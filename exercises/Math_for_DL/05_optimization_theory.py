"""
Exercises for Lesson 05: Optimization Theory
Topic: Math_for_DL

Complete the TODO sections.
"""

import numpy as np


def exercise_1_softplus_convexity():
    """Prove softplus f(x) = log(1 + e^x) is convex by showing f''(x) >= 0.

    Compute f''(x) analytically and verify it is non-negative for x in [-10, 10].
    """
    x = np.linspace(-10, 10, 1000)

    # TODO: Compute f''(x) analytically
    # f'(x) = sigmoid(x) = e^x / (1 + e^x)
    # f''(x) = sigmoid(x) * (1 - sigmoid(x))
    second_deriv = None  # Replace

    if second_deriv is not None:
        return np.all(second_deriv >= -1e-10)
    return None


def exercise_2_optimal_lr():
    """Derive and verify optimal LR for GD on f(x) = 0.5 * x^T A x - b^T x.

    Optimal LR = 2 / (lambda_max + lambda_min).
    Compare convergence with suboptimal LRs.
    """
    np.random.seed(42)
    n = 5
    eigvals = np.array([1.0, 2.0, 5.0, 10.0, 20.0])
    Q, _ = np.linalg.qr(np.random.randn(n, n))
    A = Q @ np.diag(eigvals) @ Q.T
    b = np.random.randn(n)
    x_star = np.linalg.solve(A, b)

    # TODO: Compute optimal learning rate
    lr_optimal = None  # Replace: 2 / (lambda_max + lambda_min)

    # TODO: Run GD with optimal LR for 100 steps, return final error
    x = np.zeros(n)
    for _ in range(100):
        # TODO: GD step
        pass

    final_error = np.linalg.norm(x - x_star)
    return lr_optimal, final_error


def exercise_3_cosine_annealing_warm_restarts():
    """Implement SGDR (cosine annealing with warm restarts).

    Parameters: eta_max=0.01, T_0=200 (initial period), T_mult=2
    Return the learning rate array for 1000 steps.
    """
    eta_max = 0.01
    eta_min = 0.0
    T_0 = 200
    T_mult = 2
    total_steps = 1000

    # TODO: Implement SGDR schedule
    # At each restart, the period doubles: T_0, T_0*T_mult, T_0*T_mult^2, ...
    lrs = np.zeros(total_steps)

    return lrs


if __name__ == "__main__":
    print("Exercise 1: Softplus convexity")
    result = exercise_1_softplus_convexity()
    if result is not None:
        print(f"  f''(x) >= 0 everywhere: {result}")
    else:
        print("  Not implemented yet")

    print("\nExercise 2: Optimal learning rate")
    lr, err = exercise_2_optimal_lr()
    if lr is not None:
        print(f"  Optimal LR: {lr:.6f}")
        print(f"  Final error: {err:.2e}")
    else:
        print("  Not implemented yet")

    print("\nExercise 3: SGDR schedule")
    lrs = exercise_3_cosine_annealing_warm_restarts()
    if np.any(lrs > 0):
        print(f"  LR at step 0: {lrs[0]:.5f}")
        print(f"  LR at step 100: {lrs[100]:.5f}")
        print(f"  LR at step 200: {lrs[200]:.5f} (should restart)")
    else:
        print("  Not implemented yet")
