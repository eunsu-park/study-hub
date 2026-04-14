"""
Autograd - Examples
===================
Lesson 04: Autograd

Demonstrates:
  1. Computational graph and backward pass
  2. Gradient accumulation and zeroing
  3. torch.no_grad() and detach()
  4. Higher-order gradients
  5. Gradient hooks
"""

import torch


def example_1_basic_autograd():
    """Basic gradient computation with backward()."""
    print("=" * 60)
    print("Example 1: Basic Autograd")
    print("=" * 60)

    x = torch.tensor(2.0, requires_grad=True)
    y = x ** 3 + 2 * x ** 2 + x  # y = x^3 + 2x^2 + x
    y.backward()
    print(f"x = {x.item()}")
    print(f"y = x^3 + 2x^2 + x = {y.item()}")
    print(f"dy/dx = 3x^2 + 4x + 1 = {x.grad.item()}")
    print(f"Expected: {3 * 4 + 4 * 2 + 1}")


def example_2_gradient_accumulation():
    """Demonstrate gradient accumulation behavior."""
    print("\n" + "=" * 60)
    print("Example 2: Gradient Accumulation")
    print("=" * 60)

    x = torch.tensor(3.0, requires_grad=True)

    y1 = x ** 2
    y1.backward()
    print(f"After y1 = x^2: grad = {x.grad.item()} (expected 6)")

    y2 = x ** 3
    y2.backward()
    print(f"After y2 = x^3 (accumulated): grad = {x.grad.item()} "
          f"(expected 6 + 27 = 33)")

    x.grad.zero_()
    y3 = x ** 3
    y3.backward()
    print(f"After zeroing + y3 = x^3: grad = {x.grad.item()} (expected 27)")


def example_3_no_grad():
    """Using torch.no_grad() and detach()."""
    print("\n" + "=" * 60)
    print("Example 3: no_grad and detach")
    print("=" * 60)

    x = torch.tensor(2.0, requires_grad=True)

    with torch.no_grad():
        y = x * 3
        print(f"Inside no_grad: requires_grad={y.requires_grad}")

    z = x * 3
    z_det = z.detach()
    print(f"z.requires_grad = {z.requires_grad}")
    print(f"z.detach().requires_grad = {z_det.requires_grad}")


def example_4_vector_gradients():
    """Gradients with vector outputs (Jacobian-vector product)."""
    print("\n" + "=" * 60)
    print("Example 4: Vector Gradients")
    print("=" * 60)

    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x ** 2  # [1, 4, 9]

    # Must provide gradient vector for non-scalar output
    y.backward(torch.ones_like(y))
    print(f"x = {x.data}")
    print(f"y = x^2 = {(x ** 2).data}")
    print(f"dy/dx = 2x = {x.grad}")


def example_5_higher_order():
    """Higher-order gradients (second derivative)."""
    print("\n" + "=" * 60)
    print("Example 5: Higher-Order Gradients")
    print("=" * 60)

    x = torch.tensor(2.0, requires_grad=True)
    y = x ** 4  # y = x^4

    # First derivative: dy/dx = 4x^3
    grad1 = torch.autograd.grad(y, x, create_graph=True)[0]
    print(f"dy/dx at x=2: {grad1.item()} (expected {4 * 8})")

    # Second derivative: d2y/dx2 = 12x^2
    grad2 = torch.autograd.grad(grad1, x, create_graph=True)[0]
    print(f"d2y/dx2 at x=2: {grad2.item()} (expected {12 * 4})")

    # Third derivative: d3y/dx3 = 24x
    grad3 = torch.autograd.grad(grad2, x)[0]
    print(f"d3y/dx3 at x=2: {grad3.item()} (expected {24 * 2})")


def example_6_hooks():
    """Register gradient hooks for inspection."""
    print("\n" + "=" * 60)
    print("Example 6: Gradient Hooks")
    print("=" * 60)

    gradients = {}

    def save_grad(name):
        def hook(grad):
            gradients[name] = grad.clone()
        return hook

    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    x.register_hook(save_grad('x'))

    y = x ** 2
    z = y.sum()
    z.backward()

    print(f"x.grad = {x.grad}")
    print(f"Captured via hook: {gradients['x']}")
    print(f"Match: {torch.equal(x.grad, gradients['x'])}")


if __name__ == "__main__":
    example_1_basic_autograd()
    example_2_gradient_accumulation()
    example_3_no_grad()
    example_4_vector_gradients()
    example_5_higher_order()
    example_6_hooks()
