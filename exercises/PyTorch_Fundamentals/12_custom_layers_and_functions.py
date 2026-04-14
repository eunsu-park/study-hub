"""
Custom Layers and Functions - Exercises
=======================================
Lesson 12: Custom Layers and Functions

Exercises:
  1. Implement a custom Leaky ReLU via autograd.Function
  2. Implement a custom nn.Module with learnable temperature
"""

import torch
import torch.nn as nn
from torch.autograd import Function, gradcheck


class LeakyReLUFunction(Function):
    """Custom Leaky ReLU: f(x) = x if x > 0, else alpha * x.

    TODO:
      - forward: save input and alpha for backward, apply leaky relu
      - backward: gradient is 1 if x > 0, else alpha
    """

    @staticmethod
    def forward(ctx, input, alpha=0.01):
        # TODO: implement
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        # TODO: implement
        # Return gradients for input and None for alpha
        raise NotImplementedError


def exercise_1_leaky_relu():
    """Test the custom LeakyReLU function.

    Returns:
        tuple: (output, gradient) for input [-2, -1, 0, 1, 2]
    """
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    y = LeakyReLUFunction.apply(x, 0.1)
    y.sum().backward()
    return y.data.clone(), x.grad.clone()


class TemperatureSoftmax(nn.Module):
    """Softmax with a learnable temperature parameter.

    softmax(x / temperature)

    Args:
        init_temp: initial temperature value (default 1.0)

    TODO:
      - Register temperature as nn.Parameter
      - In forward, divide input by temperature and apply softmax
    """

    def __init__(self, init_temp=1.0):
        super().__init__()
        # TODO: implement
        raise NotImplementedError

    def forward(self, x):
        # TODO: implement
        raise NotImplementedError


if __name__ == "__main__":
    print("Exercise 1: Custom Leaky ReLU")
    print("-" * 40)
    try:
        output, grad = exercise_1_leaky_relu()
        expected_out = torch.tensor([-0.2, -0.1, 0.0, 1.0, 2.0])
        expected_grad = torch.tensor([0.1, 0.1, 0.1, 1.0, 1.0])
        print(f"Output:   {output}")
        print(f"Expected: {expected_out}")
        print(f"Gradient:   {grad}")
        print(f"Expected:   {expected_grad}")
        assert torch.allclose(output, expected_out, atol=1e-5)
        assert torch.allclose(grad, expected_grad, atol=1e-5)

        # Gradcheck
        x = torch.randn(5, dtype=torch.double, requires_grad=True)
        passed = gradcheck(LeakyReLUFunction.apply, (x, 0.1))
        print(f"Gradcheck: {passed}")
        print("PASSED")
    except NotImplementedError:
        print("NOT IMPLEMENTED")

    print("\nExercise 2: Temperature Softmax")
    print("-" * 40)
    try:
        tsm = TemperatureSoftmax(init_temp=2.0)
        x = torch.tensor([[1.0, 2.0, 3.0]])
        probs = tsm(x)
        print(f"Probs (temp=2): {probs}")
        assert probs.shape == (1, 3)
        assert abs(probs.sum().item() - 1.0) < 1e-5
        # Higher temperature -> more uniform distribution
        assert probs.max() < 0.5, "High temp should make distribution flatter"
        print(f"Temperature is learnable: "
              f"{tsm.temperature.requires_grad}")
        print("PASSED")
    except NotImplementedError:
        print("NOT IMPLEMENTED")
