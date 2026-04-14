"""
Tensor Operations - Exercises
=============================
Lesson 03: Tensor Operations

Exercises:
  1. Row-wise normalization using broadcasting
  2. Batch matrix multiplication with einsum
"""

import torch


def exercise_1_normalize_rows(x):
    """Normalize each row to zero mean and unit variance.

    Args:
        x: tensor of shape [N, D]

    Returns:
        Normalized tensor of shape [N, D] where each row has
        mean~0 and std~1.

    TODO:
      - Compute row-wise mean and std (use keepdim=True)
      - Normalize: (x - mean) / (std + 1e-8)
    """
    # TODO: implement
    raise NotImplementedError


def exercise_2_batch_matmul_einsum(A, B):
    """Compute batch matrix multiplication using torch.einsum.

    Args:
        A: tensor of shape [B, M, K]
        B: tensor of shape [B, K, N]

    Returns:
        Result tensor of shape [B, M, N]

    TODO:
      - Use torch.einsum with the pattern 'bij,bjk->bik'
    """
    # TODO: implement
    raise NotImplementedError


if __name__ == "__main__":
    print("Exercise 1: Row Normalization")
    print("-" * 40)
    try:
        torch.manual_seed(42)
        x = torch.randn(5, 10) * 5 + 3
        result = exercise_1_normalize_rows(x)
        row_means = result.mean(dim=1)
        row_stds = result.std(dim=1)
        print(f"Row means (should be ~0): {row_means}")
        print(f"Row stds (should be ~1): {row_stds}")
        assert result.shape == x.shape
        assert row_means.abs().max() < 0.1
        assert (row_stds - 1.0).abs().max() < 0.2
        print("PASSED")
    except NotImplementedError:
        print("NOT IMPLEMENTED")

    print("\nExercise 2: Batch MatMul with Einsum")
    print("-" * 40)
    try:
        A = torch.randn(4, 3, 5)
        B = torch.randn(4, 5, 2)
        result = exercise_2_batch_matmul_einsum(A, B)
        expected = torch.bmm(A, B)
        assert result.shape == (4, 3, 2)
        assert torch.allclose(result, expected, atol=1e-5)
        print(f"Output shape: {result.shape}")
        print(f"Matches torch.bmm: {torch.allclose(result, expected)}")
        print("PASSED")
    except NotImplementedError:
        print("NOT IMPLEMENTED")
