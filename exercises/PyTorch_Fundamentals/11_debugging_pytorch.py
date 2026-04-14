"""
Debugging PyTorch - Exercises
=============================
Lesson 11: Debugging PyTorch

Exercises:
  1. Fix a buggy model (shape errors)
  2. Implement a gradient health checker
"""

import torch
import torch.nn as nn


def exercise_1_fix_model():
    """Fix the shape errors in this model so it runs correctly.

    The model should:
      - Take input of shape [batch, 1, 28, 28] (like MNIST)
      - Output shape [batch, 10]

    TODO: Fix the bugs in BuggyModel below.
    Return the fixed model.

    Hints:
      - Check the flatten dimension
      - Check the Linear input dimension
    """

    class BuggyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)   # [B,16,28,28]
            self.pool = nn.MaxPool2d(2)                     # [B,16,14,14]
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)   # [B,32,14,14]
            # After pool: [B,32,7,7]
            self.fc = nn.Linear(32 * 7 * 7, 10)            # BUG? Check this

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)  # flatten
            x = self.fc(x)
            return x

    # TODO: Fix and return the model
    # The model above may actually be correct -- verify by running it!
    # If it's wrong, create a corrected version.
    raise NotImplementedError


def exercise_2_gradient_health(model):
    """Check gradient health after a forward-backward pass.

    Args:
        model: nn.Module (with gradients computed)

    Returns:
        dict: mapping parameter name to status string:
              'ok', 'vanishing', 'exploding', 'none', or 'nan'

    TODO:
      - For each named parameter:
        - If grad is None -> 'none'
        - If grad has NaN -> 'nan'
        - If grad norm < 1e-7 -> 'vanishing'
        - If grad norm > 1000 -> 'exploding'
        - Otherwise -> 'ok'
    """
    # TODO: implement
    raise NotImplementedError


if __name__ == "__main__":
    print("Exercise 1: Fix Model")
    print("-" * 40)
    try:
        model = exercise_1_fix_model()
        x = torch.randn(2, 1, 28, 28)
        output = model(x)
        assert output.shape == (2, 10), f"Expected (2,10), got {output.shape}"
        print(f"Output shape: {output.shape}")
        print("PASSED")
    except NotImplementedError:
        print("NOT IMPLEMENTED")

    print("\nExercise 2: Gradient Health")
    print("-" * 40)
    try:
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()

        health = exercise_2_gradient_health(model)
        for name, status in health.items():
            print(f"  {name}: {status}")

        assert all(v == 'ok' for v in health.values())
        print("PASSED")
    except NotImplementedError:
        print("NOT IMPLEMENTED")
