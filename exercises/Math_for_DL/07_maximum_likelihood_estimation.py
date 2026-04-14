"""
Exercises for Lesson 07: Maximum Likelihood Estimation
Topic: Math_for_DL

Complete the TODO sections.
"""

import numpy as np


def exercise_1_poisson_mle():
    """Derive and verify that Poisson MLE is the sample mean.

    Generate Poisson data and show MLE(lambda) = mean(data).
    """
    np.random.seed(42)
    true_lambda = 3.5
    data = np.random.poisson(true_lambda, 500)

    # TODO: Compute MLE of lambda
    # For Poisson: log L = sum(k_i * log(lambda) - lambda - log(k_i!))
    # d/dlambda: sum(k_i/lambda - 1) = 0 => lambda_MLE = mean(k_i)
    lambda_mle = None  # Replace

    return lambda_mle, true_lambda


def exercise_2_softmax_ce_implementation():
    """Implement softmax CE loss and gradient from logits. Verify with finite differences."""
    K = 4
    np.random.seed(42)
    z = np.random.randn(K)
    y = np.zeros(K); y[1] = 1.0  # true class = 1

    # TODO: Compute loss (stable softmax CE)
    loss = None  # Replace

    # TODO: Compute gradient (should be softmax(z) - y)
    grad = None  # Replace

    # TODO: Numerical gradient check
    grad_num = np.zeros(K)
    eps = 1e-5
    # for j in range(K): ...

    return loss, grad, grad_num


def exercise_3_logistic_regression_l2():
    """Logistic regression with L2 regularization.

    Train on 2D data, compare decision boundaries with and without regularization.
    Return weights for lambda=0 and lambda=0.1.
    """
    np.random.seed(42)
    N = 100
    X_pos = np.random.randn(N//2, 2) + [1, 1]
    X_neg = np.random.randn(N//2, 2) + [-1, -1]
    X = np.vstack([X_pos, X_neg])
    y = np.hstack([np.ones(N//2), np.zeros(N//2)])

    def train(X, y, lam=0.0, lr=0.1, epochs=200):
        # TODO: Train logistic regression with L2 regularization
        # Loss = -mean(y*log(p) + (1-y)*log(1-p)) + (lam/2)*||w||^2
        w = np.zeros(2)
        b = 0.0
        # for epoch in range(epochs): ...
        return w, b

    w_noreg, b_noreg = train(X, y, lam=0.0)
    w_reg, b_reg = train(X, y, lam=0.1)

    return w_noreg, w_reg


if __name__ == "__main__":
    print("Exercise 1: Poisson MLE")
    mle, true_val = exercise_1_poisson_mle()
    if mle is not None:
        print(f"  MLE: {mle:.3f}, True: {true_val}")
    else:
        print("  Not implemented yet")

    print("\nExercise 2: Softmax CE implementation")
    loss, grad, grad_num = exercise_2_softmax_ce_implementation()
    if loss is not None:
        print(f"  Loss: {loss:.4f}")
        if grad is not None and np.any(grad_num != 0):
            print(f"  Grad error: {np.max(np.abs(grad - grad_num)):.2e}")
    else:
        print("  Not implemented yet")

    print("\nExercise 3: Logistic regression with L2")
    w0, w1 = exercise_3_logistic_regression_l2()
    if w0 is not None and np.any(w0 != 0):
        print(f"  No reg weights: {w0.round(3)}, norm: {np.linalg.norm(w0):.3f}")
        print(f"  L2 reg weights: {w1.round(3)}, norm: {np.linalg.norm(w1):.3f}")
    else:
        print("  Not implemented yet")
