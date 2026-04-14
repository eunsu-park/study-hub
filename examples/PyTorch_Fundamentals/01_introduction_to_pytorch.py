"""
Introduction to PyTorch - Examples
==================================
Lesson 01: Introduction to PyTorch

Demonstrates:
  1. PyTorch installation verification
  2. Basic tensor creation and arithmetic
  3. NumPy interoperability
  4. Linear regression from scratch (Hello World)
"""

import torch
import numpy as np


def example_1_verify_installation():
    """Verify PyTorch installation and print system info."""
    print("=" * 60)
    print("Example 1: Installation Verification")
    print("=" * 60)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available:  {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version:    {torch.version.cuda}")
        print(f"GPU device:      {torch.cuda.get_device_name(0)}")

    x = torch.tensor([1.0, 2.0, 3.0])
    print(f"\nTest tensor: {x}")
    print(f"Sum: {x.sum().item()}")
    print(f"Device: {x.device}")


def example_2_basic_tensors():
    """Create tensors and perform basic arithmetic."""
    print("\n" + "=" * 60)
    print("Example 2: Basic Tensor Creation and Arithmetic")
    print("=" * 60)

    # Creation
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])

    print(f"a = {a}")
    print(f"b = {b}")
    print(f"a + b = {a + b}")
    print(f"a * b = {a * b}")
    print(f"a ** 2 = {a ** 2}")
    print(f"dot(a, b) = {torch.dot(a, b)}")

    # Factory functions
    zeros = torch.zeros(2, 3)
    ones = torch.ones(2, 3)
    rand = torch.rand(2, 3)

    print(f"\nzeros(2,3):\n{zeros}")
    print(f"ones(2,3):\n{ones}")
    print(f"rand(2,3):\n{rand}")

    # Reductions
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    print(f"\nx = {x}")
    print(f"sum={x.sum():.1f}, mean={x.mean():.1f}, "
          f"max={x.max():.1f}, min={x.min():.1f}")


def example_3_numpy_interop():
    """Demonstrate NumPy-PyTorch interoperability."""
    print("\n" + "=" * 60)
    print("Example 3: NumPy Interoperability")
    print("=" * 60)

    # NumPy -> PyTorch (shared memory)
    np_arr = np.array([1.0, 2.0, 3.0])
    t_shared = torch.from_numpy(np_arr)
    print(f"NumPy array: {np_arr}")
    print(f"Shared tensor: {t_shared}")

    np_arr[0] = 99.0
    print(f"\nAfter modifying NumPy array[0] = 99:")
    print(f"NumPy array: {np_arr}")
    print(f"Shared tensor: {t_shared}  (changed!)")

    # PyTorch -> NumPy
    t = torch.tensor([10.0, 20.0, 30.0])
    np_from_t = t.numpy()
    print(f"\nTensor -> NumPy: {np_from_t}")

    # Independent copy
    t_copy = torch.tensor(np_arr)
    np_arr[0] = -1.0
    print(f"\nIndependent copy after modifying original:")
    print(f"Copy: {t_copy}  (not affected)")


def example_4_linear_regression():
    """Linear regression from scratch using PyTorch."""
    print("\n" + "=" * 60)
    print("Example 4: Linear Regression (Hello World)")
    print("=" * 60)

    torch.manual_seed(42)

    # Generate data: y = 2x + 1 + noise
    X = torch.rand(100, 1) * 10
    y = 2 * X + 1 + torch.randn(100, 1) * 0.5

    # Initialize parameters
    w = torch.randn(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    learning_rate = 0.01

    for epoch in range(100):
        # Forward
        y_pred = X * w + b
        loss = ((y_pred - y) ** 2).mean()

        # Backward
        loss.backward()

        # Update
        with torch.no_grad():
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad

        w.grad.zero_()
        b.grad.zero_()

        if (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | "
                  f"w: {w.item():.4f} | b: {b.item():.4f}")

    print(f"\nLearned: y = {w.item():.2f}x + {b.item():.2f}")
    print(f"True:    y = 2.00x + 1.00")


if __name__ == "__main__":
    example_1_verify_installation()
    example_2_basic_tensors()
    example_3_numpy_interop()
    example_4_linear_regression()
