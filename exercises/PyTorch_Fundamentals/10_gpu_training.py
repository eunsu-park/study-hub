"""
GPU Training - Exercises
========================
Lesson 10: GPU Training

Exercises:
  1. Write device-agnostic training code
  2. Benchmark CPU vs GPU (or measure CPU-only)
"""

import torch
import torch.nn as nn
import time


def exercise_1_device_agnostic_forward(model, x):
    """Move model and data to the best available device and run forward.

    Args:
        model: nn.Module (on CPU)
        x: input tensor (on CPU)

    Returns:
        tuple: (output_tensor, device_name_str)

    TODO:
      - Detect the best device (cuda if available, else cpu)
      - Move model and x to that device
      - Run forward pass
      - Return (output, device_name) where device_name is 'cuda' or 'cpu'
    """
    # TODO: implement
    raise NotImplementedError


def exercise_2_benchmark_matmul(size=1024, n_iters=50):
    """Benchmark matrix multiplication on CPU (and GPU if available).

    Args:
        size: matrix dimension (size x size)
        n_iters: number of iterations

    Returns:
        dict: {'cpu_ms': float, 'gpu_ms': float or None}

    TODO:
      - Create two random matrices of given size
      - Time n_iters matrix multiplications on CPU
      - If GPU available, do the same on GPU (with synchronization)
      - Return times in milliseconds
    """
    # TODO: implement
    raise NotImplementedError


if __name__ == "__main__":
    print("Exercise 1: Device-Agnostic Forward")
    print("-" * 40)
    try:
        model = nn.Linear(10, 5)
        x = torch.randn(4, 10)
        output, dev = exercise_1_device_agnostic_forward(model, x)
        print(f"Device: {dev}")
        print(f"Output shape: {output.shape}")
        assert output.shape == (4, 5)
        print("PASSED")
    except NotImplementedError:
        print("NOT IMPLEMENTED")

    print("\nExercise 2: Matmul Benchmark")
    print("-" * 40)
    try:
        result = exercise_2_benchmark_matmul(512, 20)
        print(f"CPU: {result['cpu_ms']:.2f} ms")
        if result['gpu_ms'] is not None:
            print(f"GPU: {result['gpu_ms']:.2f} ms")
            print(f"Speedup: {result['cpu_ms']/result['gpu_ms']:.1f}x")
        else:
            print("GPU: not available")
        print("PASSED")
    except NotImplementedError:
        print("NOT IMPLEMENTED")
