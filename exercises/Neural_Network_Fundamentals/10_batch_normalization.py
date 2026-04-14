"""
10. Batch Normalization - Exercises
=====================================
Lesson 10: Batch Normalization

Exercises cover:
  1. Implement BN and verify activation statistics
  2. Implement Layer Normalization
"""

import numpy as np


# ============================================================
# Exercise 1: Batch Normalization
# Implement BN and verify mean≈0, std≈1 after normalization.
# ============================================================
def exercise_1_batch_norm():
    """Implement batch normalization."""
    print("=" * 60)
    print("Exercise 1: Batch Normalization Implementation")
    print("=" * 60)

    # TODO: Implement BatchNorm class with:
    # - forward(z, training): normalize, scale, shift
    # - Running statistics update during training
    # - Use running statistics during inference

    # Test:
    np.random.seed(42)
    z = np.random.randn(8, 64) * 5 + 3  # non-zero mean, non-unit variance

    # bn = BatchNorm(8)
    # z_bn = bn.forward(z, training=True)
    # Verify: mean of z_bn per feature ≈ 0, std ≈ 1
    raise NotImplementedError("Implement batch normalization")


# ============================================================
# Exercise 2: Layer Normalization
# Implement Layer Norm and compare with Batch Norm.
# ============================================================
def exercise_2_layer_norm():
    """Implement layer normalization."""
    print("\n" + "=" * 60)
    print("Exercise 2: Layer Normalization")
    print("=" * 60)

    # TODO: Implement LayerNorm class
    # - Normalize across features (axis=0), not across batch
    # - Should work with batch size = 1
    # - Compare output statistics with BatchNorm
    raise NotImplementedError("Implement layer normalization")


if __name__ == "__main__":
    exercise_1_batch_norm()
    exercise_2_layer_norm()
