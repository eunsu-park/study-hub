"""
06. Backpropagation - Exercises
=================================
Lesson 06: Backpropagation

Exercises cover:
  1. Backpropagation for 3-layer MLP with tanh
  2. Gradient checking implementation
"""

import numpy as np


# ============================================================
# Exercise 1: 3-Layer Backprop with Tanh
# Derive and implement backprop for a 3-layer MLP using tanh.
# ============================================================
def exercise_1_tanh_backprop():
    """Implement backprop for [3, 5, 4, 2] network with tanh."""
    print("=" * 60)
    print("Exercise 1: Backprop with Tanh Activation")
    print("=" * 60)

    np.random.seed(42)

    def softmax(z):
        e = np.exp(z - np.max(z, axis=0, keepdims=True))
        return e / np.sum(e, axis=0, keepdims=True)

    def cce_loss(y_pred, y_true):
        m = y_true.shape[1]
        return -np.sum(y_true * np.log(y_pred + 1e-15)) / m

    # TODO: Initialize network [3, 5, 4, 2]
    # TODO: Implement forward pass using tanh for hidden layers, softmax for output
    # TODO: Implement backward pass
    #   - Output: dz = a - y
    #   - Hidden: dz = (W_next.T @ dz_next) * (1 - a^2)  [tanh derivative]
    # TODO: Verify with gradient checking

    X = np.random.randn(3, 10)
    Y = np.zeros((2, 10))
    Y[0, :5] = 1
    Y[1, 5:] = 1

    raise NotImplementedError("Implement 3-layer backprop with tanh")


# ============================================================
# Exercise 2: Gradient Checking
# Implement gradient checking and verify backward pass.
# ============================================================
def exercise_2_gradient_checking():
    """Implement numerical gradient checking."""
    print("\n" + "=" * 60)
    print("Exercise 2: Gradient Checking")
    print("=" * 60)

    # TODO: For a small network [2, 3, 2]:
    # 1. Run forward + backward to get analytical gradients
    # 2. For each parameter, compute numerical gradient:
    #    dL/dw ≈ (L(w+eps) - L(w-eps)) / (2*eps)
    # 3. Compare and report relative error
    # 4. All relative errors should be < 1e-5

    raise NotImplementedError("Implement gradient checking")


if __name__ == "__main__":
    exercise_1_tanh_backprop()
    exercise_2_gradient_checking()
