"""
Exercises for Lesson 01: Tensors and Autograd
Topic: Deep_Learning

Solutions to practice problems from the lesson.
"""

import numpy as np
import torch
import torch.nn as nn


# === Exercise 1: Tensor Creation and Comparison ===
# Problem: Create equivalent arrays in NumPy and PyTorch, then compare
# their properties (shape, dtype, device).

def exercise_1():
    """Create tensors in NumPy and PyTorch and compare properties."""
    # NumPy arrays
    np_arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np_zeros = np.zeros((3, 4))
    np_rand = np.random.randn(3, 4)

    # PyTorch tensors
    pt_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    pt_zeros = torch.zeros(3, 4)
    pt_rand = torch.randn(3, 4)

    print("  NumPy array: shape={}, dtype={}".format(np_arr.shape, np_arr.dtype))
    print("  PyTorch tensor: shape={}, dtype={}, device={}".format(
        pt_tensor.shape, pt_tensor.dtype, pt_tensor.device))

    # Conversion: NumPy <-> PyTorch
    converted = torch.from_numpy(np_rand)
    back = converted.numpy()

    print("  NumPy->PyTorch shape: {}".format(converted.shape))
    print("  PyTorch->NumPy shape: {}".format(back.shape))

    # Verify same values after round-trip
    assert np.allclose(np_rand, back), "Values changed during conversion!"
    print("  Round-trip conversion: OK (values identical)")


# === Exercise 2: Autograd — Polynomial Derivative ===
# Problem: Use PyTorch autograd to compute the derivative of
# f(x) = x^3 + 2x^2 - 5x + 3 at x=2. Compare with the analytical
# result f'(x) = 3x^2 + 4x - 5.

def exercise_2():
    """Compute polynomial derivative with autograd and verify analytically."""
    x_val = 2.0

    # Analytical derivative: f'(x) = 3x^2 + 4x - 5
    df_analytical = 3 * x_val**2 + 4 * x_val - 5
    print("  Analytical f'(2) = {}".format(df_analytical))  # 15.0

    # PyTorch autograd
    x = torch.tensor(x_val, requires_grad=True)
    y = x**3 + 2 * x**2 - 5 * x + 3
    y.backward()

    print("  Autograd f'(2) = {}".format(x.grad.item()))
    print("  Match: {}".format(abs(df_analytical - x.grad.item()) < 1e-5))


# === Exercise 3: Computational Graph — Multi-variable Function ===
# Problem: For z = (x + y) * (x - y), compute dz/dx and dz/dy at x=3, y=1.
# Analytically: z = x^2 - y^2, so dz/dx = 2x = 6, dz/dy = -2y = -2.

def exercise_3():
    """Multi-variable autograd: z = (x+y)*(x-y)."""
    x = torch.tensor(3.0, requires_grad=True)
    y = torch.tensor(1.0, requires_grad=True)

    z = (x + y) * (x - y)  # z = x^2 - y^2
    z.backward()

    print("  z = (x+y)*(x-y) at x=3, y=1: z={:.1f}".format(z.item()))
    print("  dz/dx = {} (expected 6.0)".format(x.grad.item()))
    print("  dz/dy = {} (expected -2.0)".format(y.grad.item()))
    print("  dz/dx correct: {}".format(abs(x.grad.item() - 6.0) < 1e-5))
    print("  dz/dy correct: {}".format(abs(y.grad.item() - (-2.0)) < 1e-5))


# === Exercise 4: no_grad and detach ===
# Problem: Demonstrate the effect of torch.no_grad() on memory and speed,
# and show how detach() separates a tensor from the computational graph.

def exercise_4():
    """Demonstrate no_grad() and detach() behavior."""
    x = torch.randn(100, 100, requires_grad=True)
    W = torch.randn(100, 100, requires_grad=True)

    # With gradient tracking (training mode)
    y = x @ W
    print("  y.requires_grad (with grad): {}".format(y.requires_grad))
    print("  y.grad_fn: {}".format(y.grad_fn))

    # Without gradient tracking (inference mode)
    with torch.no_grad():
        y_no_grad = x @ W
    print("  y.requires_grad (no_grad): {}".format(y_no_grad.requires_grad))
    print("  y.grad_fn (no_grad): {}".format(y_no_grad.grad_fn))

    # detach() — same data, no graph link
    y_detached = y.detach()
    print("  y_detached.requires_grad: {}".format(y_detached.requires_grad))
    print("  Data pointer same: {}".format(y.data_ptr() == y_detached.data_ptr()))


# === Exercise 5: Gradient Accumulation and zero_grad ===
# Problem: Show that PyTorch accumulates gradients across multiple backward()
# calls and explain why optimizer.zero_grad() is required each step.

def exercise_5():
    """Demonstrate gradient accumulation and the need for zero_grad."""
    x = torch.tensor(2.0, requires_grad=True)

    # First backward call
    y = x ** 2
    y.backward()
    print("  After 1st backward: x.grad = {}".format(x.grad.item()))  # 4.0

    # Second backward call — gradient ACCUMULATES
    y = x ** 2
    y.backward()
    print("  After 2nd backward (no zero_grad): x.grad = {}".format(x.grad.item()))  # 8.0

    # Reset gradients (simulates optimizer.zero_grad())
    x.grad.zero_()
    y = x ** 2
    y.backward()
    print("  After zero_grad + backward: x.grad = {}".format(x.grad.item()))  # 4.0
    print("  Conclusion: always call zero_grad() before each training step!")


# === Exercise 6: GPU Transfer ===
# Problem: Move tensors to GPU (if available) and verify device placement.
# Fall back to CPU gracefully if CUDA is unavailable.

def exercise_6():
    """Move tensors to GPU if available and verify device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("  Using device: {}".format(device))

    x = torch.randn(4, 4)
    x_device = x.to(device)
    print("  x device: {}".format(x_device.device))

    W = torch.randn(4, 4, device=device)
    y = x_device @ W
    print("  x @ W device: {}".format(y.device))
    print("  Computation on {}: OK".format(device))

    # Bring result back to CPU for numpy conversion
    y_cpu = y.cpu().detach().numpy()
    print("  CPU numpy result shape: {}".format(y_cpu.shape))


if __name__ == "__main__":
    print("=== Exercise 1: Tensor Creation and Comparison ===")
    exercise_1()
    print("\n=== Exercise 2: Polynomial Derivative with Autograd ===")
    exercise_2()
    print("\n=== Exercise 3: Multi-variable Computational Graph ===")
    exercise_3()
    print("\n=== Exercise 4: no_grad and detach ===")
    exercise_4()
    print("\n=== Exercise 5: Gradient Accumulation and zero_grad ===")
    exercise_5()
    print("\n=== Exercise 6: GPU Transfer ===")
    exercise_6()
    print("\nAll exercises completed!")
