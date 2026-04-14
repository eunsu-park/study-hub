"""
Tensors - Exercises
===================
Lesson 02: Tensors

Exercises:
  1. Tensor creation and dtype conversion
  2. View vs clone behavior
"""

import torch


def exercise_1_create_and_convert():
    """Create tensors with specific properties.

    TODO:
      - Create a 3x4 tensor of ones with dtype float32
      - Create a 3x4 tensor of random integers in [0, 100)
      - Convert the integer tensor to float32
      - Return (ones_tensor, int_tensor, float_tensor)
    """
    # TODO: implement
    raise NotImplementedError


def exercise_2_view_vs_clone():
    """Demonstrate understanding of view vs clone.

    Given x = torch.arange(12, dtype=torch.float32):
      - Create y as a VIEW of x reshaped to (3, 4)
      - Create z as an independent COPY of y
      - Set y[0, 0] = 99.0
      - Return (x, y, z)

    After the mutation:
      - x[0] should be 99.0 (shared memory with y)
      - z[0, 0] should be 0.0 (independent copy, unaffected)
    """
    # TODO: implement
    raise NotImplementedError


if __name__ == "__main__":
    print("Exercise 1: Create and Convert")
    print("-" * 40)
    try:
        ones, ints, floats = exercise_1_create_and_convert()
        assert ones.shape == (3, 4) and ones.dtype == torch.float32
        assert ints.shape == (3, 4) and ints.dtype == torch.int64
        assert floats.shape == (3, 4) and floats.dtype == torch.float32
        print(f"Ones dtype: {ones.dtype}, Ints dtype: {ints.dtype}, "
              f"Floats dtype: {floats.dtype}")
        print("PASSED")
    except NotImplementedError:
        print("NOT IMPLEMENTED")

    print("\nExercise 2: View vs Clone")
    print("-" * 40)
    try:
        x, y, z = exercise_2_view_vs_clone()
        assert x[0].item() == 99.0, f"x[0] should be 99.0, got {x[0]}"
        assert y[0, 0].item() == 99.0, f"y[0,0] should be 99.0"
        assert z[0, 0].item() == 0.0, f"z[0,0] should be 0.0 (copy)"
        print(f"x[0] = {x[0].item()} (shared with y)")
        print(f"z[0,0] = {z[0, 0].item()} (independent copy)")
        print("PASSED")
    except NotImplementedError:
        print("NOT IMPLEMENTED")
