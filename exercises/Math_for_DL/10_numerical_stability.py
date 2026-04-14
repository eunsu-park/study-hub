"""
Exercises for Lesson 10: Numerical Stability
Topic: Math_for_DL

Complete the TODO sections.
"""

import numpy as np


def exercise_1_stable_log_sigmoid():
    """Implement numerically stable log(sigmoid(z)) for z in [-1000, 1000].

    log(sigma(z)) = z - log(1 + e^z) for z < 0
                  = -log(1 + e^{-z}) for z >= 0
    Or equivalently: -softplus(-z) where softplus(x) = log(1 + e^x)
    """
    z_values = np.array([-1000, -100, -1, 0, 1, 100, 1000], dtype=np.float64)

    # TODO: Implement stable log-sigmoid
    def log_sigmoid_stable(z):
        # Replace with stable implementation
        return None

    results = log_sigmoid_stable(z_values)
    return results


def exercise_2_two_pass_variance():
    """Show naive variance fails and implement two-pass algorithm.

    Data: random values with large mean (1e8) and small variance (1e-4).
    """
    np.random.seed(42)
    data = np.random.randn(10000) * 0.01 + 1e8

    # Naive: E[X^2] - (E[X])^2
    var_naive = np.mean(data**2) - np.mean(data)**2

    # TODO: Implement two-pass algorithm
    # Pass 1: compute mean
    # Pass 2: compute mean of (x - mean)^2
    var_two_pass = None  # Replace

    var_true = np.var(data)
    return var_naive, var_two_pass, var_true


def exercise_3_stable_softmax_ce():
    """Implement numerically stable softmax CE loss and gradient from logits.

    Verify gradient with finite differences.
    """
    z = np.array([100.0, 200.0, 300.0])
    y = np.zeros(3); y[1] = 1.0

    # TODO: Stable softmax CE loss
    # L = log_sum_exp(z) - z[true_class]
    loss = None  # Replace

    # TODO: Stable gradient = softmax(z) - y
    grad = None  # Replace

    # TODO: Finite difference check
    grad_num = np.zeros(3)
    eps = 1e-5
    # for j in range(3): ...

    return loss, grad, grad_num


def exercise_4_dynamic_loss_scaling():
    """Implement dynamic loss scaling.

    Start S=2^15, halve on overflow, double every 200 clean steps.
    Simulate 1000 steps with random gradients.
    Return the final scale factor.
    """
    S = 2**15
    n_ok = 0
    window = 200

    np.random.seed(42)
    for step in range(1000):
        grad = np.random.randn() * np.random.choice([1e-5, 1e-3, 1e-1, 10.0])
        scaled = grad * S

        # TODO: Check overflow, adjust S
        pass

    return S


if __name__ == "__main__":
    print("Exercise 1: Stable log-sigmoid")
    results = exercise_1_stable_log_sigmoid()
    if results is not None:
        print(f"  Results: {results}")
        # log(sigma(-1000)) should be close to -1000
        # log(sigma(1000)) should be close to 0
    else:
        print("  Not implemented yet")

    print("\nExercise 2: Two-pass variance")
    vn, vtp, vt = exercise_2_two_pass_variance()
    if vtp is not None:
        print(f"  Naive:    {vn:.10e}")
        print(f"  Two-pass: {vtp:.10e}")
        print(f"  True:     {vt:.10e}")
        print(f"  Two-pass correct: {abs(vtp - vt) / vt < 1e-6}")
    else:
        print("  Not implemented yet")

    print("\nExercise 3: Stable softmax CE")
    loss, grad, grad_num = exercise_3_stable_softmax_ce()
    if loss is not None:
        print(f"  Loss: {loss:.4f}")
        if grad is not None and np.any(grad_num != 0):
            print(f"  Grad error: {np.max(np.abs(grad - grad_num)):.2e}")
    else:
        print("  Not implemented yet")

    print("\nExercise 4: Dynamic loss scaling")
    S = exercise_4_dynamic_loss_scaling()
    print(f"  Final scale: {S}")
