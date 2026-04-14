"""
Loss Functions and Optimizers - Exercises
=========================================
Lesson 06: Loss Functions and Optimizers

Exercises:
  1. Implement a custom focal loss
  2. Train a model with AdamW and cosine scheduler
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def exercise_1_focal_loss(logits, targets, gamma=2.0):
    """Implement focal loss for class-imbalanced classification.

    Focal loss: FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    Args:
        logits: tensor of shape [B, C] (raw logits)
        targets: tensor of shape [B] (class indices)
        gamma: focusing parameter

    Returns:
        scalar tensor: mean focal loss

    TODO:
      - Compute cross entropy per sample (reduction='none')
      - Compute p_t = exp(-ce)
      - Apply focal weighting: (1 - p_t)^gamma * ce
      - Return the mean
    """
    # TODO: implement
    raise NotImplementedError


def exercise_2_train_with_scheduler(model, X, y, epochs=50):
    """Train a model using AdamW optimizer with CosineAnnealingLR.

    Args:
        model: nn.Module
        X: input tensor [N, D]
        y: target tensor [N]
        epochs: number of epochs

    Returns:
        list: learning rate at each epoch

    TODO:
      - Create AdamW optimizer (lr=0.01, weight_decay=0.01)
      - Create CosineAnnealingLR scheduler (T_max=epochs)
      - Train for `epochs` steps, recording LR at each epoch
      - Return list of LRs
    """
    # TODO: implement
    raise NotImplementedError


if __name__ == "__main__":
    print("Exercise 1: Focal Loss")
    print("-" * 40)
    try:
        logits = torch.randn(8, 5)
        targets = torch.randint(0, 5, (8,))
        fl = exercise_1_focal_loss(logits, targets, gamma=2.0)
        ce = F.cross_entropy(logits, targets)
        print(f"Focal loss: {fl.item():.4f}")
        print(f"CE loss:    {ce.item():.4f}")
        assert fl.item() <= ce.item(), "Focal loss should be <= CE loss"
        assert fl.requires_grad, "Should be differentiable"
        print("PASSED")
    except NotImplementedError:
        print("NOT IMPLEMENTED")

    print("\nExercise 2: Train with Scheduler")
    print("-" * 40)
    try:
        model = nn.Linear(10, 3)
        X = torch.randn(100, 10)
        y = torch.randint(0, 3, (100,))
        lrs = exercise_2_train_with_scheduler(model, X, y, epochs=20)
        print(f"LR at epoch 0: {lrs[0]:.6f}")
        print(f"LR at epoch 10: {lrs[10]:.6f}")
        print(f"LR at epoch 19: {lrs[19]:.6f}")
        assert lrs[0] > lrs[-1], "LR should decrease with cosine schedule"
        assert len(lrs) == 20
        print("PASSED")
    except NotImplementedError:
        print("NOT IMPLEMENTED")
