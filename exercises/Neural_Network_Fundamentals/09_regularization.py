"""
09. Regularization - Exercises
================================
Lesson 09: Regularization

Exercises cover:
  1. L2 regularization effect on overfitting
  2. Dropout expected value verification
  3. Early stopping implementation
"""

import numpy as np


# ============================================================
# Exercise 1: L2 Regularization Effect
# Train an MLP with and without L2 and compare overfitting.
# ============================================================
def exercise_1_l2_effect():
    """Compare training with and without L2 regularization."""
    print("=" * 60)
    print("Exercise 1: L2 Regularization Effect")
    print("=" * 60)

    # TODO: Generate noisy 2D data (e.g., noisy spiral with few points)
    # TODO: Train a large MLP (e.g., [2, 128, 128, 3]) with and without L2
    # TODO: Report train/val accuracy for both
    # Without L2: train acc should be high, val acc should be lower (overfit)
    # With L2: train acc may be lower, but val acc should be higher
    raise NotImplementedError("Compare L2 regularization effect")


# ============================================================
# Exercise 2: Dropout Expected Value
# Verify that inverted dropout preserves expected activations.
# ============================================================
def exercise_2_dropout_expected_value():
    """Verify dropout expected value matches."""
    print("\n" + "=" * 60)
    print("Exercise 2: Dropout Expected Value")
    print("=" * 60)

    # TODO: For keep_prob in [0.3, 0.5, 0.7, 0.9]:
    # 1. Create input a = ones(100, 1)
    # 2. Apply inverted dropout 10000 times
    # 3. Compute mean of all outputs
    # 4. Verify mean ≈ 1.0 (expected value preserved)
    raise NotImplementedError("Verify dropout expected value")


# ============================================================
# Exercise 3: Early Stopping
# Implement early stopping with patience and restore best params.
# ============================================================
def exercise_3_early_stopping():
    """Implement early stopping with parameter restoration."""
    print("\n" + "=" * 60)
    print("Exercise 3: Early Stopping")
    print("=" * 60)

    # TODO: Implement EarlyStopping class with:
    # - patience: number of epochs to wait for improvement
    # - best_loss: track the best validation loss
    # - best_params: store a copy of the best parameters
    # - check(val_loss, params): returns True if should stop

    # Simulate: val_losses = [1.0, 0.9, 0.85, 0.84, 0.84, 0.85, 0.87, 0.9]
    # With patience=3, should stop at epoch 7
    raise NotImplementedError("Implement early stopping")


if __name__ == "__main__":
    exercise_1_l2_effect()
    exercise_2_dropout_expected_value()
    exercise_3_early_stopping()
