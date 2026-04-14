"""
Tensor Operations - Examples
============================
Lesson 03: Tensor Operations

Demonstrates:
  1. Indexing and slicing
  2. Element-wise operations and reductions
  3. Broadcasting
  4. Matrix operations
  5. Concatenation and stacking
"""

import torch


def example_1_indexing():
    """Indexing, slicing, boolean and fancy indexing."""
    print("=" * 60)
    print("Example 1: Indexing and Slicing")
    print("=" * 60)

    x = torch.arange(20).view(4, 5)
    print(f"x:\n{x}")
    print(f"x[1, 2] = {x[1, 2]}")
    print(f"x[1:3]:\n{x[1:3]}")
    print(f"x[:, ::2]:\n{x[:, ::2]}")

    # Boolean indexing
    vals = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0])
    print(f"\nvals > 0: {vals[vals > 0]}")

    # Fancy indexing
    idx = torch.tensor([0, 2, 4])
    print(f"vals[idx]: {vals[idx]}")


def example_2_operations():
    """Element-wise operations and reductions."""
    print("\n" + "=" * 60)
    print("Example 2: Operations and Reductions")
    print("=" * 60)

    x = torch.tensor([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]])

    print(f"sum(): {x.sum()}")
    print(f"sum(dim=0): {x.sum(dim=0)}")
    print(f"sum(dim=1): {x.sum(dim=1)}")
    print(f"mean(dim=1, keepdim=True):\n{x.mean(dim=1, keepdim=True)}")

    print(f"\nargmax(dim=1): {x.argmax(dim=1)}")
    vals, idx = x.max(dim=1)
    print(f"max(dim=1): values={vals}, indices={idx}")


def example_3_broadcasting():
    """Broadcasting rules and common patterns."""
    print("\n" + "=" * 60)
    print("Example 3: Broadcasting")
    print("=" * 60)

    x = torch.randn(3, 4)
    row_mean = x.mean(dim=1, keepdim=True)
    row_std = x.std(dim=1, keepdim=True)
    x_norm = (x - row_mean) / row_std

    print(f"x shape: {x.shape}")
    print(f"row_mean shape: {row_mean.shape}")
    print(f"x_norm row means: {x_norm.mean(dim=1)}")
    print(f"x_norm row stds: {x_norm.std(dim=1)}")

    # Outer product via broadcasting
    a = torch.tensor([1, 2, 3]).unsqueeze(1).float()
    b = torch.tensor([10, 20]).float()
    print(f"\nOuter product:\n{a * b}")


def example_4_matrix_ops():
    """Matrix multiplication and linear algebra."""
    print("\n" + "=" * 60)
    print("Example 4: Matrix Operations")
    print("=" * 60)

    A = torch.randn(2, 3)
    B = torch.randn(3, 4)
    C = A @ B
    print(f"A @ B: {A.shape} @ {B.shape} = {C.shape}")

    # Batch matmul
    bA = torch.randn(5, 2, 3)
    bB = torch.randn(5, 3, 4)
    bC = torch.bmm(bA, bB)
    print(f"bmm: {bA.shape} x {bB.shape} = {bC.shape}")

    # Dot product
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    print(f"\ndot(a, b) = {torch.dot(a, b)}")

    # Einsum
    C2 = torch.einsum('ij,jk->ik', A, B)
    print(f"einsum matmul matches @: {torch.allclose(C, C2)}")


def example_5_cat_stack():
    """Concatenation and stacking."""
    print("\n" + "=" * 60)
    print("Example 5: Concatenation and Stacking")
    print("=" * 60)

    a = torch.randn(2, 3)
    b = torch.randn(2, 3)

    cat0 = torch.cat([a, b], dim=0)
    cat1 = torch.cat([a, b], dim=1)
    stacked = torch.stack([a, b], dim=0)

    print(f"a shape: {a.shape}")
    print(f"cat(dim=0): {cat0.shape}")
    print(f"cat(dim=1): {cat1.shape}")
    print(f"stack(dim=0): {stacked.shape}")

    # Split
    parts = torch.split(cat0, 2, dim=0)
    print(f"\nsplit into {len(parts)} parts: {[p.shape for p in parts]}")


if __name__ == "__main__":
    example_1_indexing()
    example_2_operations()
    example_3_broadcasting()
    example_4_matrix_ops()
    example_5_cat_stack()
