"""
Introduction to PyTorch - Exercises
====================================
Lesson 01: Introduction to PyTorch

Exercises:
  1. Create tensors and compute statistics
  2. Implement linear regression from scratch
"""

import torch


def exercise_1_tensor_basics():
    """Create a tensor of integers 1-10 and compute sum, mean, std.

    TODO:
      - Create a float tensor containing values 1.0 through 10.0
      - Compute and return (tensor, sum, mean, std) as a tuple
    """
    # TODO: implement
    raise NotImplementedError("Implement exercise_1_tensor_basics")


def exercise_2_linear_regression(X, y, lr=0.01, epochs=200):
    """Train a linear regression model: y = w*x + b.

    Args:
        X: input tensor of shape [N, 1]
        y: target tensor of shape [N, 1]
        lr: learning rate
        epochs: number of training epochs

    Returns:
        tuple: (w, b, final_loss) where w and b are learned parameters

    TODO:
      - Initialize w and b with requires_grad=True
      - Implement the training loop (forward, loss, backward, update)
      - Return the learned parameters and final loss
    """
    # TODO: implement
    raise NotImplementedError("Implement exercise_2_linear_regression")


if __name__ == "__main__":
    print("Exercise 1: Tensor Basics")
    print("-" * 40)
    try:
        t, s, m, sd = exercise_1_tensor_basics()
        assert t.shape == (10,), f"Expected shape (10,), got {t.shape}"
        assert abs(s.item() - 55.0) < 1e-5, f"Expected sum=55, got {s}"
        assert abs(m.item() - 5.5) < 1e-5, f"Expected mean=5.5, got {m}"
        print(f"Tensor: {t}")
        print(f"Sum: {s.item()}, Mean: {m.item()}, Std: {sd.item():.4f}")
        print("PASSED")
    except NotImplementedError:
        print("NOT IMPLEMENTED")

    print("\nExercise 2: Linear Regression")
    print("-" * 40)
    try:
        torch.manual_seed(42)
        X = torch.rand(100, 1) * 10
        y = 3 * X + 2 + torch.randn(100, 1) * 0.5

        w, b, loss = exercise_2_linear_regression(X, y)
        print(f"Learned: w={w.item():.4f}, b={b.item():.4f}")
        print(f"Expected: w~3.0, b~2.0")
        print(f"Final loss: {loss:.4f}")
        assert abs(w.item() - 3.0) < 0.5, f"w should be ~3.0, got {w.item()}"
        assert abs(b.item() - 2.0) < 1.0, f"b should be ~2.0, got {b.item()}"
        print("PASSED")
    except NotImplementedError:
        print("NOT IMPLEMENTED")
