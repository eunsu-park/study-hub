"""
Training Loop - Exercises
=========================
Lesson 08: Training Loop

Exercises:
  1. Implement a validation function
  2. Implement early stopping
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@torch.no_grad()
def exercise_1_validate(model, loader, loss_fn, device='cpu'):
    """Compute validation loss and accuracy.

    Args:
        model: nn.Module
        loader: DataLoader
        loss_fn: loss function
        device: torch device

    Returns:
        tuple: (avg_loss, accuracy)

    TODO:
      - Set model to eval mode
      - Iterate over loader, compute loss and accuracy
      - Return average loss and accuracy (as floats)
      - Remember: no gradient computation needed
    """
    # TODO: implement
    raise NotImplementedError


class EarlyStopping:
    """Stop training when validation loss stops improving.

    TODO:
      - Track the best loss seen so far
      - Count consecutive epochs without improvement
      - Set self.should_stop = True when counter >= patience
    """

    def __init__(self, patience=5):
        self.patience = patience
        # TODO: initialize other attributes
        raise NotImplementedError

    def __call__(self, val_loss):
        # TODO: implement
        raise NotImplementedError


if __name__ == "__main__":
    print("Exercise 1: Validate")
    print("-" * 40)
    try:
        torch.manual_seed(42)
        model = nn.Linear(10, 3)
        X = torch.randn(100, 10)
        y = torch.randint(0, 3, (100,))
        loader = DataLoader(TensorDataset(X, y), batch_size=32)
        loss_fn = nn.CrossEntropyLoss()

        val_loss, val_acc = exercise_1_validate(model, loader, loss_fn)
        print(f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")
        assert isinstance(val_loss, float)
        assert 0.0 <= val_acc <= 1.0
        print("PASSED")
    except NotImplementedError:
        print("NOT IMPLEMENTED")

    print("\nExercise 2: Early Stopping")
    print("-" * 40)
    try:
        es = EarlyStopping(patience=3)
        losses = [1.0, 0.9, 0.95, 0.96, 0.97]
        for i, loss in enumerate(losses):
            es(loss)
            if es.should_stop:
                print(f"Stopped at step {i} (loss={loss})")
                break
        assert es.should_stop, "Should have stopped"
        print("PASSED")
    except NotImplementedError:
        print("NOT IMPLEMENTED")
