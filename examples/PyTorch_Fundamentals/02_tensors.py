"""
Tensors - Examples
==================
Lesson 02: Tensors

Demonstrates:
  1. Tensor creation methods (from data, factory functions, like-tensors)
  2. Tensor attributes (shape, dtype, device)
  3. View vs copy behavior
  4. Reshaping operations
  5. Memory layout and contiguity
"""

import torch
import numpy as np


def example_1_creation():
    """Various tensor creation methods."""
    print("=" * 60)
    print("Example 1: Tensor Creation")
    print("=" * 60)

    # From data
    t1 = torch.tensor([1, 2, 3])
    t2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    print(f"From list: {t1}, dtype={t1.dtype}")
    print(f"From nested list: shape={t2.shape}, dtype={t2.dtype}")

    # Factory functions
    print(f"\nzeros(2,3):\n{torch.zeros(2, 3)}")
    print(f"rand(2,3):\n{torch.rand(2, 3)}")
    print(f"arange(0,10,2): {torch.arange(0, 10, 2)}")
    print(f"linspace(0,1,5): {torch.linspace(0, 1, 5)}")
    print(f"eye(3):\n{torch.eye(3)}")

    # like-tensors
    x = torch.randn(2, 3)
    print(f"\nzeros_like(x): {torch.zeros_like(x).shape}")


def example_2_attributes():
    """Tensor attributes: shape, dtype, device."""
    print("\n" + "=" * 60)
    print("Example 2: Tensor Attributes")
    print("=" * 60)

    t = torch.randn(2, 3, 4)
    print(f"Shape: {t.shape}")
    print(f"ndim: {t.ndim}")
    print(f"numel: {t.numel()}")
    print(f"dtype: {t.dtype}")
    print(f"device: {t.device}")

    # dtype casting
    x = torch.tensor([1, 2, 3])
    print(f"\nint64: {x.dtype}")
    print(f"float(): {x.float().dtype}")
    print(f"half(): {x.half().dtype}")
    print(f"bool(): {x.bool()}")


def example_3_view_vs_copy():
    """View (shared memory) vs copy (independent memory)."""
    print("\n" + "=" * 60)
    print("Example 3: View vs Copy")
    print("=" * 60)

    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    # View: shared memory
    y = x.view(2, 3)
    y[0, 0] = 99.0
    print(f"After view mutation: x[0] = {x[0]} (shared!)")

    # Clone: independent copy
    x = torch.tensor([1.0, 2.0, 3.0])
    z = x.clone()
    z[0] = 99.0
    print(f"After clone mutation: x[0] = {x[0]} (independent)")

    # Check memory sharing
    a = torch.randn(6)
    b = a.view(2, 3)
    c = a.clone()
    print(f"\nview shares memory: {a.data_ptr() == b.data_ptr()}")
    print(f"clone shares memory: {a.data_ptr() == c.data_ptr()}")


def example_4_reshaping():
    """Reshaping operations: view, reshape, squeeze, unsqueeze, permute."""
    print("\n" + "=" * 60)
    print("Example 4: Reshaping Operations")
    print("=" * 60)

    x = torch.arange(12)
    print(f"Original: {x.shape}")

    print(f"view(3,4): {x.view(3, 4).shape}")
    print(f"view(3,-1): {x.view(3, -1).shape}")

    # squeeze and unsqueeze
    y = torch.randn(1, 3, 1, 4)
    print(f"\nOriginal: {y.shape}")
    print(f"squeeze(): {y.squeeze().shape}")
    print(f"squeeze(0): {y.squeeze(0).shape}")

    z = torch.randn(3, 4)
    print(f"\nOriginal: {z.shape}")
    print(f"unsqueeze(0): {z.unsqueeze(0).shape}")
    print(f"unsqueeze(-1): {z.unsqueeze(-1).shape}")

    # permute
    t = torch.randn(2, 3, 4)
    print(f"\nOriginal: {t.shape}")
    print(f"permute(2,0,1): {t.permute(2, 0, 1).shape}")

    # flatten
    print(f"flatten(): {t.flatten().shape}")
    print(f"flatten(1): {t.flatten(1).shape}")


def example_5_contiguity():
    """Memory contiguity and strides."""
    print("\n" + "=" * 60)
    print("Example 5: Contiguity and Strides")
    print("=" * 60)

    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(f"Original strides: {x.stride()}, contiguous: {x.is_contiguous()}")

    y = x.T
    print(f"Transposed strides: {y.stride()}, contiguous: {y.is_contiguous()}")

    y_c = y.contiguous()
    print(f"After .contiguous(): strides={y_c.stride()}, "
          f"contiguous={y_c.is_contiguous()}")


if __name__ == "__main__":
    example_1_creation()
    example_2_attributes()
    example_3_view_vs_copy()
    example_4_reshaping()
    example_5_contiguity()
