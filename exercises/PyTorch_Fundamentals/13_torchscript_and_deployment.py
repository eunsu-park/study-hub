"""
TorchScript and Deployment - Exercises
======================================
Lesson 13: TorchScript and Deployment

Exercises:
  1. Trace a model and verify outputs match
  2. Script a model with control flow
"""

import torch
import torch.nn as nn
import tempfile
import os


def exercise_1_trace_model():
    """Trace a simple MLP and verify the traced version matches.

    TODO:
      - Create an MLP: Linear(20,64) -> ReLU -> Linear(64,10)
      - Set to eval mode
      - Trace with example input shape [1, 20]
      - Verify outputs match for a test input
      - Save the traced model and report file size in KB

    Returns:
        tuple: (traced_model, file_size_kb)
    """
    # TODO: implement
    raise NotImplementedError


def exercise_2_script_model():
    """Script a model that uses data-dependent control flow.

    Create and script a model where:
      - If input sum > 0: return relu(linear(x))
      - Else: return sigmoid(linear(x))

    TODO:
      - Define a model class with the above logic
      - Script it with torch.jit.script
      - Verify both branches work

    Returns:
        tuple: (scripted_model, pos_output, neg_output)
        where pos_output = scripted(ones(1,10))
        and neg_output = scripted(-ones(1,10))
    """
    # TODO: implement
    raise NotImplementedError


if __name__ == "__main__":
    print("Exercise 1: Trace Model")
    print("-" * 40)
    try:
        traced, size_kb = exercise_1_trace_model()
        x = torch.randn(5, 20)
        output = traced(x)
        assert output.shape == (5, 10)
        print(f"Output shape: {output.shape}")
        print(f"File size: {size_kb:.1f} KB")
        print("PASSED")
    except NotImplementedError:
        print("NOT IMPLEMENTED")

    print("\nExercise 2: Script Model")
    print("-" * 40)
    try:
        scripted, pos_out, neg_out = exercise_2_script_model()
        print(f"Positive branch output min: {pos_out.min():.4f} (>= 0 for relu)")
        print(f"Negative branch output range: [{neg_out.min():.4f}, "
              f"{neg_out.max():.4f}] (in (0,1) for sigmoid)")
        assert pos_out.min() >= 0, "ReLU output should be >= 0"
        assert neg_out.min() > 0 and neg_out.max() < 1, \
            "Sigmoid output should be in (0, 1)"
        print("PASSED")
    except NotImplementedError:
        print("NOT IMPLEMENTED")
