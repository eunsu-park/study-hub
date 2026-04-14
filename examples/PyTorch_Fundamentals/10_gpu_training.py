"""
GPU Training - Examples
=======================
Lesson 10: GPU Training

Demonstrates:
  1. Device-agnostic code
  2. GPU memory monitoring
  3. CPU vs GPU speed comparison
  4. Mixed precision training pattern (CPU-safe demo)
"""

import torch
import torch.nn as nn
import time


def example_1_device_agnostic():
    """Write device-agnostic code."""
    print("=" * 60)
    print("Example 1: Device-Agnostic Code")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
    model = model.to(device)

    x = torch.randn(32, 784, device=device)
    output = model(x)
    print(f"Input device: {x.device}")
    print(f"Output device: {output.device}")
    print(f"Output shape: {output.shape}")


def example_2_memory():
    """Monitor GPU memory usage."""
    print("\n" + "=" * 60)
    print("Example 2: GPU Memory Monitoring")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("No GPU available. Showing CPU-only info.")
        x = torch.randn(1000, 1000)
        print(f"Tensor size: {x.nelement() * x.element_size() / 1024:.1f} KB")
        return

    torch.cuda.reset_peak_memory_stats()
    print(f"Initial: {torch.cuda.memory_allocated()/1024**2:.1f} MB")

    x = torch.randn(1000, 1000, device='cuda')
    print(f"After 1000x1000 tensor: "
          f"{torch.cuda.memory_allocated()/1024**2:.1f} MB")

    del x
    torch.cuda.empty_cache()
    print(f"After del + empty_cache: "
          f"{torch.cuda.memory_allocated()/1024**2:.1f} MB")
    print(f"Peak: {torch.cuda.max_memory_allocated()/1024**2:.1f} MB")


def example_3_speed_comparison():
    """Compare CPU vs GPU speed for matrix multiplication."""
    print("\n" + "=" * 60)
    print("Example 3: Speed Comparison")
    print("=" * 60)

    size = 2048
    n_iters = 20

    # CPU benchmark
    A_cpu = torch.randn(size, size)
    B_cpu = torch.randn(size, size)
    for _ in range(3):  # warmup
        _ = A_cpu @ B_cpu

    start = time.time()
    for _ in range(n_iters):
        _ = A_cpu @ B_cpu
    cpu_time = (time.time() - start) / n_iters
    print(f"CPU ({size}x{size} matmul): {cpu_time*1000:.1f} ms")

    if torch.cuda.is_available():
        A_gpu = A_cpu.cuda()
        B_gpu = B_cpu.cuda()
        for _ in range(10):
            _ = A_gpu @ B_gpu
        torch.cuda.synchronize()

        start = time.time()
        for _ in range(n_iters):
            _ = A_gpu @ B_gpu
        torch.cuda.synchronize()
        gpu_time = (time.time() - start) / n_iters
        print(f"GPU ({size}x{size} matmul): {gpu_time*1000:.1f} ms")
        print(f"Speedup: {cpu_time/gpu_time:.1f}x")
    else:
        print("No GPU available for comparison.")


def example_4_reproducibility():
    """Set seeds for reproducibility."""
    print("\n" + "=" * 60)
    print("Example 4: Reproducibility")
    print("=" * 60)

    def set_seed(seed=42):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    set_seed(42)
    a = torch.randn(5)
    set_seed(42)
    b = torch.randn(5)
    print(f"Same seed produces same values: {torch.equal(a, b)}")


if __name__ == "__main__":
    example_1_device_agnostic()
    example_2_memory()
    example_3_speed_comparison()
    example_4_reproducibility()
