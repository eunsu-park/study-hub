"""
Model Saving and Loading - Exercises
====================================
Lesson 09: Model Saving and Loading

Exercises:
  1. Save and load a model checkpoint
  2. Partial loading for transfer learning
"""

import torch
import torch.nn as nn
import tempfile
import os


def exercise_1_checkpoint(model, optimizer, epoch, val_loss):
    """Save a training checkpoint and reload it.

    Args:
        model: nn.Module
        optimizer: torch.optim.Optimizer
        epoch: int
        val_loss: float

    Returns:
        dict: loaded checkpoint with keys
              'epoch', 'model_state_dict', 'optimizer_state_dict', 'val_loss'

    TODO:
      - Create a checkpoint dict with the 4 keys above
      - Save to a temp file
      - Load and return the checkpoint
    """
    # TODO: implement
    raise NotImplementedError


def exercise_2_partial_load(old_state_dict, new_model):
    """Load only matching parameters from old_state_dict into new_model.

    Args:
        old_state_dict: OrderedDict from a different model
        new_model: nn.Module with potentially different architecture

    Returns:
        tuple: (n_loaded, n_total) - number of loaded vs total parameters

    TODO:
      - Get the new model's state dict
      - Find keys that exist in both and have matching shapes
      - Update the new state dict with matched parameters
      - Load the updated state dict into new_model
    """
    # TODO: implement
    raise NotImplementedError


if __name__ == "__main__":
    print("Exercise 1: Checkpoint Save/Load")
    print("-" * 40)
    try:
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())
        ckpt = exercise_1_checkpoint(model, optimizer, epoch=10, val_loss=0.42)
        assert ckpt['epoch'] == 10
        assert ckpt['val_loss'] == 0.42
        assert 'model_state_dict' in ckpt
        assert 'optimizer_state_dict' in ckpt
        print(f"Epoch: {ckpt['epoch']}, Val loss: {ckpt['val_loss']}")
        print("PASSED")
    except NotImplementedError:
        print("NOT IMPLEMENTED")

    print("\nExercise 2: Partial Loading")
    print("-" * 40)
    try:
        old_model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(),
                                   nn.Linear(20, 5))
        old_state = old_model.state_dict()

        new_model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(),
                                   nn.Linear(20, 10))
        n_loaded, n_total = exercise_2_partial_load(old_state, new_model)
        print(f"Loaded {n_loaded}/{n_total} parameter tensors")
        assert n_loaded == 2  # first layer weight and bias match
        assert n_total == 4   # 4 total parameter tensors
        print("PASSED")
    except NotImplementedError:
        print("NOT IMPLEMENTED")
