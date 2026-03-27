"""
Exercises for Lesson 23: GPU Acceleration
Topic: Numerical_Simulation

Solutions to practice problems covering CUDA concepts,
memory transfer analysis, GPU stencil operations,
performance modeling, and CuPy-style parallelism.
"""

import numpy as np
import time


# ============================================================
# Exercise 1: CUDA Thread Hierarchy Simulation
# ============================================================
def exercise_1():
    """
    Simulate CUDA's thread hierarchy: grid → blocks → threads.
    Map a 2D problem to CUDA thread indices.
    """
    print("=== Exercise 1: CUDA Thread Hierarchy ===\n")

    def simulate_cuda_launch(grid_dim, block_dim, problem_size):
        """Simulate how CUDA maps threads to a 2D problem.

        Each thread computes: global_id = blockIdx * blockDim + threadIdx
        """
        Gx, Gy = grid_dim
        Bx, By = block_dim
        Nx, Ny = problem_size

        total_threads = Gx * Gy * Bx * By
        active_threads = 0
        wasted_threads = 0

        # Simulate thread assignment
        for bx in range(Gx):
            for by in range(Gy):
                for tx in range(Bx):
                    for ty in range(By):
                        gx = bx * Bx + tx
                        gy = by * By + ty
                        if gx < Nx and gy < Ny:
                            active_threads += 1
                        else:
                            wasted_threads += 1

        return {
            "total_threads": total_threads,
            "active": active_threads,
            "wasted": wasted_threads,
            "occupancy": active_threads / total_threads * 100,
        }

    # Test different configurations for a 100×100 problem
    problem = (100, 100)
    configs = [
        ((4, 4), (32, 32)),    # 128×128 threads
        ((7, 7), (16, 16)),    # 112×112 threads
        ((13, 13), (8, 8)),    # 104×104 threads
        ((100, 100), (1, 1)),  # 1 thread per block (bad!)
    ]

    print(f"  Problem size: {problem[0]}×{problem[1]} = "
          f"{problem[0]*problem[1]} elements\n")
    print(f"  {'Grid':>10} | {'Block':>10} | {'Total':>8} | "
          f"{'Active':>8} | {'Wasted':>8} | {'Occupancy':>10}")
    print(f"  {'-'*62}")

    for grid, block in configs:
        result = simulate_cuda_launch(grid, block, problem)
        print(f"  {str(grid):>10} | {str(block):>10} | "
              f"{result['total_threads']:>8} | {result['active']:>8} | "
              f"{result['wasted']:>8} | {result['occupancy']:>9.1f}%")

    print(f"\n  Best practice: block_dim = (16,16) or (32,32)")
    print(f"  Choose grid_dim = ceil(N/block_dim) to cover all elements")
    print(f"  Wasted threads occur at boundary blocks — check bounds in kernel")
    print()


# ============================================================
# Exercise 2: Memory Transfer Optimization
# ============================================================
def exercise_2():
    """
    Analyze strategies to minimize CPU↔GPU memory transfer overhead.
    Compare: naive transfer vs pinned memory vs compute overlap.
    """
    print("=== Exercise 2: Memory Transfer Optimization ===\n")

    # Parameters
    pcie_bw = 12e9         # PCIe 3.0 x16: 12 GB/s
    pinned_bw = 13e9       # Pinned memory: ~10% faster
    gpu_bw = 900e9         # GPU HBM2: 900 GB/s
    gpu_flops = 10e12      # 10 TFLOPS

    data_sizes_mb = [1, 10, 100, 1000, 10000]

    print(f"  PCIe: {pcie_bw/1e9:.0f} GB/s, Pinned: {pinned_bw/1e9:.0f} GB/s, "
          f"GPU BW: {gpu_bw/1e9:.0f} GB/s")
    print(f"\n  Strategy comparison for iterative solver (100 iterations):\n")

    n_iter = 100
    flops_per_byte = 10  # arithmetic intensity

    print(f"  {'Data (MB)':>10} | {'Naive (ms)':>11} | {'Pinned (ms)':>12} | "
          f"{'Keep-on-GPU':>12} | {'Best Speedup':>13}")
    print(f"  {'-'*64}")

    for size_mb in data_sizes_mb:
        data_bytes = size_mb * 1e6

        # Strategy 1: Naive — transfer each iteration
        transfer_naive = 2 * data_bytes / pcie_bw  # upload + download
        compute = data_bytes * flops_per_byte / gpu_flops
        naive_total = n_iter * (transfer_naive + compute)

        # Strategy 2: Pinned memory
        transfer_pinned = 2 * data_bytes / pinned_bw
        pinned_total = n_iter * (transfer_pinned + compute)

        # Strategy 3: Keep on GPU (transfer only at start/end)
        keep_gpu_total = (2 * data_bytes / pcie_bw  # initial + final transfer
                          + n_iter * compute)

        speedup = naive_total / keep_gpu_total if keep_gpu_total > 0 else 0

        print(f"  {size_mb:>10} | {naive_total*1000:>11.2f} | "
              f"{pinned_total*1000:>12.2f} | {keep_gpu_total*1000:>12.2f} | "
              f"{speedup:>12.1f}x")

    print(f"\n  Key optimization: minimize transfers!")
    print(f"    1. Keep data on GPU across iterations")
    print(f"    2. Use pinned memory for unavoidable transfers")
    print(f"    3. Overlap transfer with compute (CUDA streams)")
    print(f"    4. Batch small transfers into large ones")
    print()


# ============================================================
# Exercise 3: GPU Stencil Operation — Jacobi Iteration
# ============================================================
def exercise_3():
    """
    Implement Jacobi iteration for Laplace equation using
    GPU-style vectorized stencil operations.
    Compare convergence with different grid sizes.
    """
    print("=== Exercise 3: Jacobi Iteration (GPU-style) ===\n")

    def jacobi_solve(N, max_iter=1000, tol=1e-6):
        """Solve Laplace equation ∇²u = 0 with Jacobi iteration.

        BCs: u=1 on top, u=0 on other sides.
        The 5-point stencil is ideal for GPU parallelism.
        """
        u = np.zeros((N, N))
        u[0, :] = 1.0  # top boundary

        for iteration in range(max_iter):
            # Vectorized stencil — equivalent to a CUDA kernel
            u_new = u.copy()
            u_new[1:-1, 1:-1] = 0.25 * (
                u[2:, 1:-1] + u[:-2, 1:-1]
                + u[1:-1, 2:] + u[1:-1, :-2]
            )
            # Enforce BCs
            u_new[0, :] = 1.0

            # Check convergence
            diff = np.max(np.abs(u_new - u))
            u = u_new

            if diff < tol:
                return u, iteration + 1, diff

        return u, max_iter, diff

    print(f"  Laplace eq: ∇²u = 0, BCs: top=1, others=0")
    print(f"  Method: Jacobi iteration, tol=1e-6\n")

    print(f"  {'Grid':>8} | {'Iterations':>11} | {'Time (ms)':>10} | "
          f"{'Center u':>9} | {'Residual':>10}")
    print(f"  {'-'*56}")

    for N in [16, 32, 64, 128, 256]:
        t0 = time.perf_counter()
        u, iters, residual = jacobi_solve(N, max_iter=5000)
        t_ms = (time.perf_counter() - t0) * 1000

        center = u[N // 2, N // 2]
        print(f"  {N:>4}x{N:<3} | {iters:>11} | {t_ms:>10.1f} | "
              f"{center:>9.4f} | {residual:>10.2e}")

    print(f"\n  Larger grids need more iterations (convergence slows with h)")
    print(f"  GPU advantage: each grid point updated independently → parallel")
    print(f"  On GPU: 256×256 = 65K threads, ~50-100x faster than CPU loop")
    print()


# ============================================================
# Exercise 4: Roofline Model Analysis
# ============================================================
def exercise_4():
    """
    Apply the roofline performance model to determine whether
    a kernel is compute-bound or memory-bound.
    """
    print("=== Exercise 4: Roofline Model Analysis ===\n")

    # GPU specs (typical mid-range)
    peak_flops = 10e12      # 10 TFLOPS (FP32)
    peak_bw = 900e9         # 900 GB/s (HBM2)
    ridge_point = peak_flops / peak_bw  # FLOPs/byte

    print(f"  GPU: {peak_flops/1e12:.0f} TFLOPS, {peak_bw/1e9:.0f} GB/s BW")
    print(f"  Ridge point: {ridge_point:.1f} FLOPs/byte\n")

    kernels = [
        {"name": "Vector Add", "flops_per_elem": 1, "bytes_per_elem": 12,
         "desc": "c[i] = a[i] + b[i]"},
        {"name": "Dot Product", "flops_per_elem": 2, "bytes_per_elem": 8,
         "desc": "sum += a[i] * b[i]"},
        {"name": "5pt Stencil", "flops_per_elem": 5, "bytes_per_elem": 20,
         "desc": "Laplacian: 4 neighbors + center"},
        {"name": "MatMul (N=1K)", "flops_per_elem": 2000, "bytes_per_elem": 8,
         "desc": "C[i,j] += A[i,k]*B[k,j], N=1000"},
        {"name": "FFT (N=1M)", "flops_per_elem": 100, "bytes_per_elem": 16,
         "desc": "5*N*log2(N) FLOPs"},
    ]

    print(f"  {'Kernel':>15} | {'AI':>6} | {'Bound':>10} | {'Peak GFLOPS':>12} | "
          f"{'% Peak':>7}")
    print(f"  {'-'*58}")

    for k in kernels:
        ai = k["flops_per_elem"] / k["bytes_per_elem"]  # arithmetic intensity

        if ai < ridge_point:
            bound = "Memory"
            achievable = ai * peak_bw  # limited by bandwidth
        else:
            bound = "Compute"
            achievable = peak_flops  # limited by compute

        pct_peak = achievable / peak_flops * 100

        print(f"  {k['name']:>15} | {ai:>5.1f} | {bound:>10} | "
              f"{achievable/1e9:>12.0f} | {pct_peak:>6.1f}%")

    print(f"\n  Most PDE stencils are memory-bound (low arithmetic intensity).")
    print(f"  Strategies to improve:")
    print(f"    - Shared memory tiling: reduce global memory reads")
    print(f"    - Operator fusion: combine multiple stencils in one kernel")
    print(f"    - Mixed precision: FP16 halves memory traffic")
    print()


# ============================================================
# Exercise 5: Multi-GPU Decomposition
# ============================================================
def exercise_5():
    """
    Simulate domain decomposition for multi-GPU computation.
    Analyze halo exchange overhead.
    """
    print("=== Exercise 5: Multi-GPU Domain Decomposition ===\n")

    def decompose_domain(Nx, Ny, n_gpus):
        """Split a 2D domain into strips for multi-GPU.

        Each GPU gets Nx × (Ny/n_gpus) with halo layers.
        """
        strip_height = Ny // n_gpus
        remainder = Ny % n_gpus

        gpus = []
        y_start = 0
        for g in range(n_gpus):
            h = strip_height + (1 if g < remainder else 0)
            # Halo: 1 row from each neighbor
            halo_top = 1 if g > 0 else 0
            halo_bottom = 1 if g < n_gpus - 1 else 0
            total_rows = h + halo_top + halo_bottom

            gpus.append({
                "gpu": g,
                "rows": h,
                "halo_rows": total_rows,
                "y_range": (y_start, y_start + h),
                "compute_cells": Nx * h,
                "halo_cells": Nx * (halo_top + halo_bottom),
            })
            y_start += h

        return gpus

    # Problem: 2D heat equation on 1024×1024 grid
    Nx, Ny = 1024, 1024
    gpu_bw = 900e9  # bytes/sec (GPU memory)
    nvlink_bw = 300e9  # bytes/sec (inter-GPU)
    gpu_flops = 10e12

    print(f"  Domain: {Nx}×{Ny}, 5-point stencil (10 FLOPs/point)")
    print(f"  GPU: {gpu_flops/1e12:.0f} TFLOPS, NVLink: {nvlink_bw/1e9:.0f} GB/s\n")

    print(f"  {'GPUs':>5} | {'Rows/GPU':>9} | {'Compute':>10} | "
          f"{'Halo':>10} | {'Halo%':>6} | {'Speedup':>8} | {'Efficiency':>10}")
    print(f"  {'-'*68}")

    for n_gpus in [1, 2, 4, 8, 16]:
        gpus = decompose_domain(Nx, Ny, n_gpus)

        total_compute = sum(g["compute_cells"] for g in gpus)
        total_halo = sum(g["halo_cells"] for g in gpus)

        # Time model
        compute_per_gpu = max(g["compute_cells"] for g in gpus)
        compute_time = compute_per_gpu * 10 / gpu_flops  # seconds

        halo_per_gpu = Nx * 2 * 4  # 2 rows × 4 bytes (float32)
        halo_time = halo_per_gpu / nvlink_bw if n_gpus > 1 else 0

        total_time = compute_time + halo_time
        single_gpu_time = Nx * Ny * 10 / gpu_flops

        speedup = single_gpu_time / total_time if total_time > 0 else 1
        efficiency = speedup / n_gpus * 100

        halo_pct = total_halo / (total_compute + total_halo) * 100

        print(f"  {n_gpus:>5} | {gpus[0]['rows']:>9} | {total_compute:>10,} | "
              f"{total_halo:>10,} | {halo_pct:>5.1f}% | {speedup:>7.1f}x | "
              f"{efficiency:>9.1f}%")

    print(f"\n  Halo exchange overhead grows with GPU count.")
    print(f"  Efficiency drops because:")
    print(f"    1. Communication latency (synchronization barrier)")
    print(f"    2. Halo cells = redundant computation at boundaries")
    print(f"    3. Load imbalance if domain doesn't divide evenly")
    print(f"  Strategy: overlap halo exchange with interior computation.")
    print()


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
