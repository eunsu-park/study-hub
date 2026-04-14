"""
Custom Layers and Functions - Examples
======================================
Lesson 12: Custom Layers and Functions

Demonstrates:
  1. Custom autograd.Function (MyReLU)
  2. Multi-input custom function
  3. Gradient checking with gradcheck
  4. Straight-through estimator
  5. Custom nn.Module with parameters
"""

import torch
import torch.nn as nn
from torch.autograd import Function, gradcheck


class MyReLU(Function):
    """Custom ReLU implementation with explicit backward."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad = grad_output.clone()
        grad[input < 0] = 0
        return grad


class STE(Function):
    """Straight-Through Estimator for quantization."""

    @staticmethod
    def forward(ctx, input):
        return input.round()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def example_1_custom_relu():
    """Custom ReLU via autograd.Function."""
    print("=" * 60)
    print("Example 1: Custom ReLU")
    print("=" * 60)

    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    y = MyReLU.apply(x)
    y.sum().backward()

    print(f"Input:    {x.data}")
    print(f"Output:   {y.data}")
    print(f"Gradient: {x.grad}")

    # Compare with built-in
    x2 = x.detach().clone().requires_grad_(True)
    y2 = torch.relu(x2)
    y2.sum().backward()
    print(f"Match built-in: {torch.equal(x.grad, x2.grad)}")


def example_2_gradcheck():
    """Verify custom gradients with finite differences."""
    print("\n" + "=" * 60)
    print("Example 2: Gradient Checking")
    print("=" * 60)

    x = torch.randn(5, dtype=torch.double, requires_grad=True)
    passed = gradcheck(MyReLU.apply, (x,), eps=1e-6, atol=1e-4)
    print(f"MyReLU gradcheck passed: {passed}")


def example_3_ste():
    """Straight-through estimator for quantization."""
    print("\n" + "=" * 60)
    print("Example 3: Straight-Through Estimator")
    print("=" * 60)

    x = torch.tensor([0.3, 0.7, 1.2, 1.8, 2.5], requires_grad=True)
    y = STE.apply(x)
    y.sum().backward()

    print(f"Input (float): {x.data}")
    print(f"Output (quantized): {y.data}")
    print(f"Gradient (straight-through): {x.grad}")


def example_4_custom_module():
    """Custom nn.Module with learnable parameters."""
    print("\n" + "=" * 60)
    print("Example 4: Custom Module")
    print("=" * 60)

    class ScaleShift(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(dim))
            self.shift = nn.Parameter(torch.zeros(dim))

        def forward(self, x):
            return self.scale * x + self.shift

        def extra_repr(self):
            return f"dim={self.scale.shape[0]}"

    layer = ScaleShift(5)
    print(f"Module: {layer}")
    print(f"Parameters: {sum(p.numel() for p in layer.parameters())}")

    x = torch.randn(3, 5)
    y = layer(x)
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")


def example_5_in_model():
    """Use custom function inside a model."""
    print("\n" + "=" * 60)
    print("Example 5: Custom Function in Model")
    print("=" * 60)

    class CustomActivation(nn.Module):
        def forward(self, x):
            return MyReLU.apply(x)

    model = nn.Sequential(
        nn.Linear(10, 20),
        CustomActivation(),
        nn.Linear(20, 5),
    )

    x = torch.randn(4, 10)
    output = model(x)
    loss = output.sum()
    loss.backward()

    print(f"Model output shape: {output.shape}")
    print(f"All gradients computed: "
          f"{all(p.grad is not None for p in model.parameters())}")


if __name__ == "__main__":
    example_1_custom_relu()
    example_2_gradcheck()
    example_3_ste()
    example_4_custom_module()
    example_5_in_model()
