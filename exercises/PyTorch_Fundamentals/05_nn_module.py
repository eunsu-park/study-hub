"""
nn.Module - Exercises
=====================
Lesson 05: nn.Module

Exercises:
  1. Build a 3-layer MLP with dropout
  2. Count trainable parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def exercise_1_build_mlp(input_dim, hidden_dims, output_dim, dropout=0.3):
    """Build a multi-layer perceptron with ReLU and dropout.

    Architecture: input -> [Linear -> ReLU -> Dropout] x len(hidden_dims) -> Linear -> output

    Args:
        input_dim: int, input feature dimension
        hidden_dims: list of int, hidden layer dimensions
        output_dim: int, output dimension
        dropout: float, dropout probability

    Returns:
        nn.Module: the MLP model

    TODO:
      - Create an nn.Sequential model
      - For each hidden dim: add Linear, ReLU, Dropout
      - Add final Linear layer (no activation)
    """
    # TODO: implement
    raise NotImplementedError


def exercise_2_count_params(model):
    """Count total and trainable parameters in a model.

    Args:
        model: nn.Module

    Returns:
        tuple: (total_params, trainable_params)

    TODO:
      - Iterate over model.parameters()
      - Count total and trainable (requires_grad=True) parameters
    """
    # TODO: implement
    raise NotImplementedError


if __name__ == "__main__":
    print("Exercise 1: Build MLP")
    print("-" * 40)
    try:
        model = exercise_1_build_mlp(784, [256, 128], 10)
        x = torch.randn(4, 784)
        model.eval()
        output = model(x)
        assert output.shape == (4, 10), f"Expected (4,10), got {output.shape}"
        print(f"Output shape: {output.shape}")
        print(f"Model:\n{model}")
        print("PASSED")
    except NotImplementedError:
        print("NOT IMPLEMENTED")

    print("\nExercise 2: Count Parameters")
    print("-" * 40)
    try:
        model = nn.Sequential(
            nn.Linear(100, 50),  # 100*50 + 50 = 5050
            nn.ReLU(),
            nn.Linear(50, 10),   # 50*10 + 10 = 510
        )
        total, trainable = exercise_2_count_params(model)
        expected = 5050 + 510
        print(f"Total: {total}, Trainable: {trainable}")
        assert total == expected, f"Expected {expected}, got {total}"
        assert trainable == expected
        print("PASSED")
    except NotImplementedError:
        print("NOT IMPLEMENTED")
