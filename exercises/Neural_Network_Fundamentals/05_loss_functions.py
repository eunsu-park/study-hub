"""
05. Loss Functions - Exercises
================================
Lesson 05: Loss Functions

Exercises cover:
  1. Prove sigmoid + BCE gradient simplification
  2. Implement Huber loss
  3. Implement CCE for a batch
"""

import numpy as np


# ============================================================
# Exercise 1: Sigmoid + BCE Gradient
# Prove numerically that dL/dz = sigmoid(z) - y when using BCE.
# ============================================================
def exercise_1_bce_gradient():
    """Verify sigmoid + BCE gradient = sigmoid(z) - y."""
    print("=" * 60)
    print("Exercise 1: BCE + Sigmoid Gradient Simplification")
    print("=" * 60)

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    # TODO: For several (z, y) pairs:
    # 1. Compute BCE loss: L = -[y*log(sigmoid(z)) + (1-y)*log(1-sigmoid(z))]
    # 2. Compute numerical gradient: dL/dz ≈ (L(z+eps) - L(z-eps)) / (2*eps)
    # 3. Compute analytical gradient: dL/dz = sigmoid(z) - y
    # 4. Verify they match
    raise NotImplementedError("Verify BCE gradient simplification")


# ============================================================
# Exercise 2: Huber Loss
# Implement Huber loss and compare with MSE on data with outliers.
# ============================================================
def exercise_2_huber_loss():
    """Implement Huber loss."""
    print("\n" + "=" * 60)
    print("Exercise 2: Huber Loss")
    print("=" * 60)

    def huber_loss(y_pred, y_true, delta=1.0):
        # TODO: Implement Huber loss
        # L = 0.5*(y-ŷ)^2           if |y-ŷ| <= delta
        # L = delta*|y-ŷ| - 0.5*delta^2  otherwise
        raise NotImplementedError("Implement Huber loss")

    # Test: compare MSE and Huber on data with outlier
    y_true = np.array([1.0, 2.0, 3.0, 100.0])  # 100 is outlier
    y_pred = np.array([1.1, 2.2, 2.8, 3.0])

    mse = 0.5 * np.mean((y_pred - y_true) ** 2)
    huber = huber_loss(y_pred, y_true, delta=1.0)
    print(f"  MSE:   {mse:.4f}")
    print(f"  Huber: {huber:.4f}")
    print("  Huber should be much less sensitive to the outlier")


# ============================================================
# Exercise 3: Batch CCE
# Implement CCE for a batch of 32 samples with 5 classes.
# ============================================================
def exercise_3_batch_cce():
    """Implement categorical cross-entropy for a batch."""
    print("\n" + "=" * 60)
    print("Exercise 3: Batch Categorical Cross-Entropy")
    print("=" * 60)

    np.random.seed(42)

    def softmax(z):
        e = np.exp(z - np.max(z, axis=0, keepdims=True))
        return e / np.sum(e, axis=0, keepdims=True)

    def batch_cce(y_pred, y_true, eps=1e-15):
        # TODO: Implement CCE for a batch
        # y_pred: shape (K, N) softmax probabilities
        # y_true: shape (K, N) one-hot labels
        # Return: scalar average loss
        raise NotImplementedError("Implement batch CCE")

    # Create test data: 5 classes, 32 samples
    K, N = 5, 32
    logits = np.random.randn(K, N)
    y_pred = softmax(logits)

    # Random one-hot labels
    labels = np.random.randint(0, K, N)
    y_true = np.zeros((K, N))
    y_true[labels, np.arange(N)] = 1

    loss = batch_cce(y_pred, y_true)
    print(f"  Batch CCE (5 classes, 32 samples): {loss:.6f}")


if __name__ == "__main__":
    exercise_1_bce_gradient()
    exercise_2_huber_loss()
    exercise_3_batch_cce()
