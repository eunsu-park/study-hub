"""
13. Building MLP from Scratch - Exercises
===========================================
Lesson 13: Building MLP from Scratch

Exercises cover:
  1. Add L2 regularization to the MLP
  2. Add learning rate scheduling
"""

import numpy as np


# ============================================================
# Exercise 1: Add L2 Regularization
# Extend the MLP class to support L2 weight decay.
# ============================================================
def exercise_1_l2_regularization():
    """Add L2 regularization to MLP."""
    print("=" * 60)
    print("Exercise 1: MLP with L2 Regularization")
    print("=" * 60)

    # TODO: Modify the MLP backward pass to include L2 gradient
    # dW_total = dW_data + lambda * W
    # Modify the loss computation to include L2 penalty:
    # total_loss = data_loss + (lambda/2) * sum(||W||^2)

    # Test: Train on spiral data with lambda=0 and lambda=0.01
    # Report train/val accuracy for both
    raise NotImplementedError("Add L2 to MLP")


# ============================================================
# Exercise 2: Learning Rate Scheduling
# Add cosine annealing to the training loop.
# ============================================================
def exercise_2_lr_scheduling():
    """Implement cosine annealing in training loop."""
    print("\n" + "=" * 60)
    print("Exercise 2: Cosine Annealing LR Schedule")
    print("=" * 60)

    # TODO: Implement cosine annealing:
    # lr(t) = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(pi*t/T))
    # Integrate into the Adam optimizer by adjusting self.lr each epoch
    # Train on spiral data and compare with fixed LR
    raise NotImplementedError("Implement LR scheduling")


if __name__ == "__main__":
    exercise_1_l2_regularization()
    exercise_2_lr_scheduling()
