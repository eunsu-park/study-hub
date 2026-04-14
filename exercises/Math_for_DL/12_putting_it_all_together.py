"""
Exercises for Lesson 12: Putting It All Together
Topic: Math_for_DL

Complete the TODO sections.
"""

import numpy as np


def exercise_1_spiral_classifier():
    """Train a 2-layer classifier on the spiral dataset.

    Target: >95% accuracy with Adam optimizer.
    """
    np.random.seed(42)
    N, K = 200, 3
    X = np.zeros((N * K, 2))
    y = np.zeros(N * K, dtype=int)
    for k in range(K):
        ix = range(N*k, N*(k+1))
        r = np.linspace(0, 1, N)
        t = np.linspace(k*4, (k+1)*4, N) + np.random.randn(N)*0.2
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = k
    Y = np.zeros((N*K, K))
    Y[np.arange(N*K), y] = 1

    # TODO: Initialize weights (He init)
    n_h = 100
    W1 = None  # Replace
    b1 = None
    W2 = None
    b2 = None

    # TODO: Train with Adam for 500 epochs
    # Include: stable softmax, CE loss, L2 reg, Adam

    # TODO: Compute final accuracy
    accuracy = None

    return accuracy


def exercise_2_batch_norm_backward():
    """Add batch normalization and derive backward pass.

    BN: hat_x = (x - mu) / sqrt(var + eps)
    y = gamma * hat_x + beta
    """
    np.random.seed(42)
    N, D = 32, 10
    x = np.random.randn(N, D)
    gamma = np.ones(D)
    beta = np.zeros(D)
    eps = 1e-5

    # TODO: Forward pass
    mu = None  # Replace: mean over batch
    var = None  # Replace: variance over batch
    x_hat = None  # Replace: normalized
    y = None  # Replace: gamma * x_hat + beta

    # TODO: Backward pass (given dy)
    dy = np.random.randn(N, D)
    dgamma = None
    dbeta = None
    dx = None  # This is the tricky part

    return dgamma, dbeta, dx


def exercise_3_gradient_clipping():
    """Implement gradient clipping by global norm.

    If ||g|| > max_norm, scale g = g * max_norm / ||g||
    Show it prevents divergence at high learning rates.
    """
    def clip_grad_norm(grads, max_norm):
        """Clip list of gradient arrays by global norm.

        Args:
            grads: list of numpy arrays
            max_norm: float

        Returns:
            clipped_grads: list of clipped arrays
            global_norm: original global norm
        """
        # TODO: Compute global norm across all gradient arrays
        global_norm = None  # Replace: sqrt(sum of squared Frobenius norms)

        # TODO: Scale if necessary
        clipped = None  # Replace

        return clipped, global_norm

    # Test
    grads = [np.random.randn(3, 4) * 10, np.random.randn(4,) * 5]
    clipped, gn = clip_grad_norm(grads, max_norm=5.0)

    if clipped is not None:
        clipped_norm = np.sqrt(sum(np.sum(g**2) for g in clipped))
        return gn, clipped_norm
    return None, None


if __name__ == "__main__":
    print("Exercise 1: Spiral classifier")
    acc = exercise_1_spiral_classifier()
    if acc is not None:
        print(f"  Accuracy: {acc:.3f}, Pass: {acc > 0.95}")
    else:
        print("  Not implemented yet")

    print("\nExercise 2: Batch norm backward")
    dg, db, dx = exercise_2_batch_norm_backward()
    if dg is not None:
        print(f"  dgamma shape: {dg.shape}")
        print(f"  dbeta shape: {db.shape}")
        if dx is not None:
            print(f"  dx shape: {dx.shape}")
    else:
        print("  Not implemented yet")

    print("\nExercise 3: Gradient clipping")
    orig, clipped = exercise_3_gradient_clipping()
    if orig is not None:
        print(f"  Original norm: {orig:.4f}")
        print(f"  Clipped norm: {clipped:.4f}")
        print(f"  Pass: {clipped <= 5.0 + 1e-6}")
    else:
        print("  Not implemented yet")
