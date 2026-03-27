"""
Example: GPU Acceleration for Numerical Simulation
Topic: Numerical_Simulation – Lesson 23

Demonstrates GPU acceleration concepts using pure NumPy
(CuPy is a drop-in replacement — same API, GPU-backed).

  1. CPU vs GPU computation patterns
  2. Memory transfer overhead analysis
  3. GPU-style parallel heat equation solver
  4. Scaling analysis: when GPU wins

Run: python 23_gpu_acceleration.py
Requirements: numpy only (CuPy code shown in comments for GPU execution)
"""

import numpy as np
import time


# ============================================================
# Demo 1: Vectorized Operations — The GPU Advantage
# ============================================================
def demo_vectorization():
    """Show why GPUs excel at embarrassingly parallel operations.

    GPUs have thousands of cores optimized for the same operation
    on different data (SIMD/SIMT). Vectorized NumPy operations
    map directly to this paradigm.
    """
    print("=" * 60)
    print("Demo 1: Vectorized vs Loop-Based Computation")
    print("=" * 60)

    sizes = [1000, 10_000, 100_000, 1_000_000]

    print(f"\n  Element-wise operation: c = sin(a) + cos(b) * a")
    print(f"\n  {'N':>10} | {'Loop (ms)':>10} | {'Vectorized (ms)':>16} | {'Speedup':>8}")
    print(f"  {'-'*52}")

    for n in sizes:
        a = np.random.randn(n)
        b = np.random.randn(n)

        # Loop-based (simulates single-core scalar execution)
        t0 = time.perf_counter()
        c_loop = np.empty(n)
        for i in range(n):
            c_loop[i] = np.sin(a[i]) + np.cos(b[i]) * a[i]
        t_loop = (time.perf_counter() - t0) * 1000

        # Vectorized (simulates GPU-style parallel execution)
        t0 = time.perf_counter()
        c_vec = np.sin(a) + np.cos(b) * a
        t_vec = (time.perf_counter() - t0) * 1000

        speedup = t_loop / t_vec if t_vec > 0 else float("inf")
        print(f"  {n:>10,} | {t_loop:>10.2f} | {t_vec:>16.3f} | {speedup:>7.1f}x")

    print(f"\n  GPU acceleration works on the same principle:")
    print(f"  thousands of threads execute the same operation in parallel.")
    print(f"  CuPy: just replace np → cp for GPU execution.")

    # Show CuPy equivalent (commented — runs only with CuPy installed)
    print(f"\n  CuPy equivalent:")
    print(f"    import cupy as cp")
    print(f"    a_gpu = cp.array(a)     # CPU → GPU transfer")
    print(f"    b_gpu = cp.array(b)")
    print(f"    c_gpu = cp.sin(a_gpu) + cp.cos(b_gpu) * a_gpu  # GPU compute")
    print(f"    c_cpu = c_gpu.get()     # GPU → CPU transfer")
    print()


# ============================================================
# Demo 2: Memory Transfer Overhead
# ============================================================
def demo_memory_transfer():
    """Analyze the cost of CPU↔GPU data transfer.

    GPU computation is fast, but moving data between CPU and GPU
    memory is expensive. This overhead can negate GPU benefits
    for small problems.
    """
    print("=" * 60)
    print("Demo 2: Memory Transfer Overhead Analysis")
    print("=" * 60)

    # Simulate PCIe bandwidth: ~12 GB/s (PCIe 3.0 x16)
    pcie_bandwidth = 12e9  # bytes/second

    # GPU compute throughput: ~10 TFLOPS (single precision)
    gpu_flops = 10e12

    print(f"\n  Assumed: PCIe 3.0 x16 (~12 GB/s), GPU ~10 TFLOPS")
    print(f"\n  {'Array Size':>12} | {'Transfer (ms)':>14} | {'Compute (ms)':>13} | "
          f"{'Ratio':>6} | {'GPU Worth It?'}")
    print(f"  {'-'*70}")

    for n in [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]:
        data_bytes = n * 4  # float32 = 4 bytes
        transfer_time = 2 * data_bytes / pcie_bandwidth  # round trip

        # Assume 10 FLOPs per element (typical stencil operation)
        flops_total = n * 10
        compute_time = flops_total / gpu_flops

        ratio = transfer_time / compute_time if compute_time > 0 else float("inf")
        worth_it = "YES" if ratio < 1.0 else "NO — transfer dominates"

        print(f"  {n:>12,} | {transfer_time*1000:>14.4f} | "
              f"{compute_time*1000:>13.6f} | {ratio:>6.1f} | {worth_it}")

    print(f"\n  Key insight: GPU is only beneficial when compute >> transfer.")
    print(f"  Strategies to improve compute/transfer ratio:")
    print(f"    1. Keep data on GPU across multiple kernel launches")
    print(f"    2. Use pinned (page-locked) memory for faster transfers")
    print(f"    3. Overlap transfer and compute with CUDA streams")
    print(f"    4. Increase arithmetic intensity (more FLOPs per byte)")
    print()


# ============================================================
# Demo 3: GPU-Accelerated Heat Equation (2D)
# ============================================================
def demo_heat_equation():
    """Solve 2D heat equation using GPU-style vectorized stencil.

    PDE: ∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)
    Method: FTCS (Forward Time, Centered Space)

    The 5-point stencil is perfectly suited for GPU parallelism:
    each grid point can be updated independently.
    """
    print("=" * 60)
    print("Demo 3: GPU-Style 2D Heat Equation Solver")
    print("=" * 60)

    grid_sizes = [64, 128, 256, 512]
    alpha = 0.01
    n_steps = 100

    print(f"\n  PDE: ∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²), α={alpha}")
    print(f"  Method: FTCS, {n_steps} time steps")
    print(f"\n  {'Grid':>8} | {'Loop (ms)':>10} | {'Vectorized (ms)':>16} | "
          f"{'Speedup':>8} | {'Max diff'}")
    print(f"  {'-'*62}")

    for N in grid_sizes:
        dx = 1.0 / (N - 1)
        dt = 0.2 * dx ** 2 / alpha  # CFL condition

        # Initial condition: hot center
        u0 = np.zeros((N, N))
        cx, cy = N // 2, N // 2
        u0[cx - N // 8:cx + N // 8, cy - N // 8:cy + N // 8] = 100.0

        # --- Loop-based (CPU single-core style) ---
        # Only run for small grids to avoid timeout
        if N <= 128:
            u_loop = u0.copy()
            t0 = time.perf_counter()
            for _ in range(n_steps):
                u_new = u_loop.copy()
                for i in range(1, N - 1):
                    for j in range(1, N - 1):
                        laplacian = (u_loop[i + 1, j] + u_loop[i - 1, j]
                                     + u_loop[i, j + 1] + u_loop[i, j - 1]
                                     - 4 * u_loop[i, j]) / dx ** 2
                        u_new[i, j] = u_loop[i, j] + alpha * dt * laplacian
                u_loop = u_new
            t_loop = (time.perf_counter() - t0) * 1000
        else:
            t_loop = float("nan")

        # --- Vectorized (GPU-style parallel) ---
        u_vec = u0.copy()
        t0 = time.perf_counter()
        for _ in range(n_steps):
            # Stencil operation: all interior points updated simultaneously
            # This is exactly what a CUDA kernel would do
            laplacian = (u_vec[2:, 1:-1] + u_vec[:-2, 1:-1]
                         + u_vec[1:-1, 2:] + u_vec[1:-1, :-2]
                         - 4 * u_vec[1:-1, 1:-1]) / dx ** 2
            u_vec[1:-1, 1:-1] += alpha * dt * laplacian
        t_vec = (time.perf_counter() - t0) * 1000

        # Verify results match (for small grids)
        if N <= 128:
            max_diff = np.max(np.abs(u_loop - u_vec))
            speedup = t_loop / t_vec
            print(f"  {N:>4}x{N:<3} | {t_loop:>10.1f} | {t_vec:>16.3f} | "
                  f"{speedup:>7.1f}x | {max_diff:.2e}")
        else:
            print(f"  {N:>4}x{N:<3} | {'(skip)':>10} | {t_vec:>16.3f} | "
                  f"{'—':>8} | {'—'}")

    # Show final state statistics
    print(f"\n  Final state (512x512): min={u_vec.min():.2f}, "
          f"max={u_vec.max():.2f}, mean={u_vec.mean():.2f}")

    print(f"\n  GPU stencil: each thread computes one grid point.")
    print(f"  For 512×512 = 262,144 points, a GPU launches 262K threads.")
    print(f"  With shared memory tiling, bandwidth is further reduced.")
    print()


# ============================================================
# Demo 4: Scaling Analysis — When to Use GPU
# ============================================================
def demo_scaling():
    """Analyze how problem size affects CPU vs GPU performance.

    Small problems: CPU wins (transfer overhead dominates).
    Large problems: GPU wins (massive parallelism amortizes transfer).
    """
    print("=" * 60)
    print("Demo 4: CPU vs GPU Scaling Analysis")
    print("=" * 60)

    # Model: total_time = transfer_time + compute_time
    # CPU: compute = N * flops_per_element / cpu_flops
    # GPU: compute = N * flops_per_element / gpu_flops + 2 * N * 4 / pcie_bw

    cpu_gflops = 50     # single-core GFLOPS
    gpu_gflops = 10_000  # GPU GFLOPS
    pcie_gbps = 12       # GB/s
    flops_per_elem = 10  # typical stencil

    print(f"\n  CPU: {cpu_gflops} GFLOPS, GPU: {gpu_gflops} GFLOPS, "
          f"PCIe: {pcie_gbps} GB/s")
    print(f"\n  {'Problem Size':>14} | {'CPU (ms)':>9} | {'GPU (ms)':>9} | "
          f"{'Speedup':>8} | {'Winner'}")
    print(f"  {'-'*58}")

    for n in [100, 1_000, 10_000, 100_000, 1_000_000,
              10_000_000, 100_000_000]:
        cpu_time = n * flops_per_elem / (cpu_gflops * 1e9) * 1000  # ms

        transfer = 2 * n * 4 / (pcie_gbps * 1e9) * 1000  # ms
        gpu_compute = n * flops_per_elem / (gpu_gflops * 1e9) * 1000  # ms
        gpu_time = transfer + gpu_compute

        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        winner = "GPU" if speedup > 1 else "CPU"

        print(f"  {n:>14,} | {cpu_time:>9.4f} | {gpu_time:>9.4f} | "
              f"{speedup:>7.1f}x | {winner}")

    # Find crossover point analytically
    # cpu_time = gpu_time  →  N * F / Fc = 2*N*4/B + N*F/Fg
    # N * F * (1/Fc - 1/Fg) = 2*N*4/B
    # F * (1/Fc - 1/Fg) = 8/B → always true for all N
    # Actually: N cancels, so crossover depends on flops_per_elem
    # For low arithmetic intensity, GPU may never win

    print(f"\n  Crossover depends on arithmetic intensity (FLOPs/byte):")
    print(f"    Low intensity (1 FLOP/byte): GPU needs very large N")
    print(f"    High intensity (100 FLOPs/byte): GPU wins even for small N")
    print(f"\n  Rule of thumb: GPU shines when:")
    print(f"    - Problem size > 100K elements")
    print(f"    - Data stays on GPU across multiple operations")
    print(f"    - Operations are embarrassingly parallel (stencils, matmul)")
    print()


if __name__ == "__main__":
    demo_vectorization()
    demo_memory_transfer()
    demo_heat_equation()
    demo_scaling()
