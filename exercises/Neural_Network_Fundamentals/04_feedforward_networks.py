"""
04. Feedforward Networks - Exercises
======================================
Lesson 04: Feedforward Networks

Exercises cover:
  1. Parameter count calculation
  2. Forward pass implementation for 3-layer MLP
"""

import numpy as np


# ============================================================
# Exercise 1: Parameter Count
# Compute the parameter count for [784, 256, 128, 10].
# ============================================================
def exercise_1_param_count():
    """Calculate total parameters for a given architecture."""
    print("=" * 60)
    print("Exercise 1: Parameter Count")
    print("=" * 60)

    layer_sizes = [784, 256, 128, 10]

    # TODO: Compute the number of parameters (weights + biases) for each layer
    # and the total. Show your work.
    # Formula: Layer l has n_l * n_{l-1} weights + n_l biases
    raise NotImplementedError("Compute parameter count")


# ============================================================
# Exercise 2: Forward Pass
# Implement forward pass for a 3-layer MLP and verify shapes.
# ============================================================
def exercise_2_forward_pass():
    """Implement forward pass and verify output shapes."""
    print("\n" + "=" * 60)
    print("Exercise 2: Forward Pass Implementation")
    print("=" * 60)

    np.random.seed(42)
    layer_sizes = [4, 8, 6, 3]

    # TODO: Initialize weights (He init) and biases for each layer
    # TODO: Implement forward pass with ReLU for hidden layers, Softmax for output
    # TODO: Verify that output shape is (3, batch_size) and probabilities sum to 1

    # Test with single sample
    x = np.random.randn(4, 1)
    # y_pred = forward(x, params)
    # assert y_pred.shape == (3, 1)
    # assert np.allclose(y_pred.sum(), 1.0)

    # Test with batch
    X = np.random.randn(4, 32)
    # y_pred_batch = forward(X, params)
    # assert y_pred_batch.shape == (3, 32)
    raise NotImplementedError("Implement forward pass")


if __name__ == "__main__":
    exercise_1_param_count()
    exercise_2_forward_pass()
