"""
Debugging PyTorch - Examples
============================
Lesson 11: Debugging PyTorch

Demonstrates:
  1. Shape debugging with print statements
  2. Gradient checking utility
  3. Forward hooks for activation inspection
  4. Anomaly detection
  5. Simple timing utility
"""

import torch
import torch.nn as nn


def example_1_shape_debugging():
    """Systematic shape debugging."""
    print("=" * 60)
    print("Example 1: Shape Debugging")
    print("=" * 60)

    class DebugMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 256)
            self.fc2 = nn.Linear(256, 10)

        def forward(self, x):
            print(f"  Input:     {x.shape}")
            x = torch.relu(self.fc1(x))
            print(f"  After fc1: {x.shape}")
            x = self.fc2(x)
            print(f"  Output:    {x.shape}")
            return x

    model = DebugMLP()
    print("Forward pass shapes:")
    output = model(torch.randn(4, 784))


def example_2_gradient_check():
    """Check gradients for common issues."""
    print("\n" + "=" * 60)
    print("Example 2: Gradient Checking")
    print("=" * 60)

    model = nn.Sequential(
        nn.Linear(10, 20), nn.ReLU(),
        nn.Linear(20, 20), nn.ReLU(),
        nn.Linear(20, 5),
    )

    x = torch.randn(4, 10)
    loss = model(x).sum()
    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            norm = param.grad.norm().item()
            has_nan = torch.isnan(param.grad).any().item()
            status = "NaN!" if has_nan else ("vanishing" if norm < 1e-7 else "OK")
            print(f"  {name}: grad_norm={norm:.6f} [{status}]")
        else:
            print(f"  {name}: grad=None")


def example_3_forward_hooks():
    """Use forward hooks to inspect activations."""
    print("\n" + "=" * 60)
    print("Example 3: Forward Hooks")
    print("=" * 60)

    model = nn.Sequential(
        nn.Linear(784, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 10),
    )

    activations = {}

    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = {
                'shape': output.shape,
                'mean': output.mean().item(),
                'std': output.std().item(),
                'min': output.min().item(),
                'max': output.max().item(),
            }
        return hook

    handles = []
    for i, layer in enumerate(model):
        handles.append(layer.register_forward_hook(hook_fn(f"layer_{i}")))

    _ = model(torch.randn(8, 784))

    print("Activation statistics:")
    for name, stats in activations.items():
        print(f"  {name}: shape={stats['shape']}, "
              f"mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
              f"range=[{stats['min']:.4f}, {stats['max']:.4f}]")

    for h in handles:
        h.remove()


def example_4_anomaly_detection():
    """Use detect_anomaly to find NaN sources."""
    print("\n" + "=" * 60)
    print("Example 4: Anomaly Detection")
    print("=" * 60)

    print("Running normal computation (no anomaly):")
    x = torch.randn(3, requires_grad=True)
    y = x ** 2
    y.sum().backward()
    print(f"  grad = {x.grad}")

    print("\nWith detect_anomaly (catches problematic ops):")
    try:
        with torch.autograd.detect_anomaly():
            x = torch.tensor([0.0], requires_grad=True)
            y = torch.log(x)  # log(0) = -inf
            z = y * 2
            z.backward()
    except RuntimeError as e:
        print(f"  Caught: {str(e)[:80]}...")


def example_5_timer():
    """Simple timing utility for profiling."""
    print("\n" + "=" * 60)
    print("Example 5: Timing Utility")
    print("=" * 60)

    import time

    class Timer:
        def __init__(self, name=""):
            self.name = name

        def __enter__(self):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.start = time.time()
            return self

        def __exit__(self, *args):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.elapsed = time.time() - self.start
            print(f"  {self.name}: {self.elapsed*1000:.2f} ms")

    model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
    x = torch.randn(64, 784)

    with Timer("Forward"):
        out = model(x)

    with Timer("Backward"):
        out.sum().backward()


if __name__ == "__main__":
    example_1_shape_debugging()
    example_2_gradient_check()
    example_3_forward_hooks()
    example_4_anomaly_detection()
    example_5_timer()
