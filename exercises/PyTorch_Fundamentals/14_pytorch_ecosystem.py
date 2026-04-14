"""
PyTorch Ecosystem - Exercises
=============================
Lesson 14: PyTorch Ecosystem

Exercises:
  1. Create a model summary utility
  2. Build a Lightning-style training module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def exercise_1_model_summary(model):
    """Create a model summary showing layer names, types, and param counts.

    Args:
        model: nn.Module

    Returns:
        dict with keys:
          'total_params': int
          'trainable_params': int
          'layers': list of dicts with 'name', 'type', 'params'

    TODO:
      - Iterate over model.named_modules() (skip root)
      - For each module, count parameters (recurse=False)
      - Only include modules that have parameters
    """
    # TODO: implement
    raise NotImplementedError


class LitClassifier(nn.Module):
    """A Lightning-style classifier module (pure PyTorch).

    TODO:
      - __init__: create a 2-layer MLP (input_dim -> hidden_dim -> num_classes)
      - forward: return logits
      - training_step(batch): compute and return loss
      - configure_optimizers: return an Adam optimizer
    """

    def __init__(self, input_dim, hidden_dim, num_classes, lr=1e-3):
        super().__init__()
        # TODO: implement
        raise NotImplementedError

    def forward(self, x):
        # TODO: implement
        raise NotImplementedError

    def training_step(self, batch):
        """Compute loss for a batch.

        Args:
            batch: tuple of (X, y)

        Returns:
            dict: {'loss': tensor, 'acc': tensor}
        """
        # TODO: implement
        raise NotImplementedError

    def configure_optimizers(self):
        """Return an Adam optimizer.

        Returns:
            torch.optim.Optimizer
        """
        # TODO: implement
        raise NotImplementedError


if __name__ == "__main__":
    print("Exercise 1: Model Summary")
    print("-" * 40)
    try:
        model = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(), nn.BatchNorm1d(256),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )
        summary = exercise_1_model_summary(model)
        print(f"Total params: {summary['total_params']:,}")
        print(f"Trainable: {summary['trainable_params']:,}")
        for layer in summary['layers']:
            print(f"  {layer['name']}: {layer['type']} ({layer['params']:,})")
        assert summary['total_params'] > 0
        assert len(summary['layers']) > 0
        print("PASSED")
    except NotImplementedError:
        print("NOT IMPLEMENTED")

    print("\nExercise 2: Lightning-Style Classifier")
    print("-" * 40)
    try:
        clf = LitClassifier(20, 64, 5)
        batch = (torch.randn(16, 20), torch.randint(0, 5, (16,)))

        # Test forward
        logits = clf(batch[0])
        assert logits.shape == (16, 5)

        # Test training step
        result = clf.training_step(batch)
        assert 'loss' in result and 'acc' in result
        print(f"Loss: {result['loss'].item():.4f}")
        print(f"Acc: {result['acc'].item():.2%}")

        # Test optimizer
        opt = clf.configure_optimizers()
        assert isinstance(opt, torch.optim.Optimizer)
        print(f"Optimizer: {type(opt).__name__}")
        print("PASSED")
    except NotImplementedError:
        print("NOT IMPLEMENTED")
