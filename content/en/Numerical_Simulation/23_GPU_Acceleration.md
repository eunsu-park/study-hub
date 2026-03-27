[Previous: Finite Element Method](./22_Finite_Element_Method.md) | [Next: Physics-Informed Neural Networks](./24_PINN.md)

---

# 23. GPU Acceleration for Numerical Simulation

> **Prerequisites**: Familiarity with NumPy array operations and basic PDE solving (Lessons 6-10).

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the CUDA programming model and GPU architecture for scientific computing
2. Use CuPy as a drop-in GPU-accelerated replacement for NumPy
3. Accelerate PDE solvers (finite difference, spectral methods) on GPU
4. Benchmark CPU vs GPU performance and identify when GPU acceleration helps
5. Handle GPU memory management and data transfer optimization

---

## Table of Contents

1. [Why GPUs for Simulation?](#1-why-gpus-for-simulation)
2. [CUDA Programming Model](#2-cuda-programming-model)
3. [CuPy: GPU-Accelerated NumPy](#3-cupy-gpu-accelerated-numpy)
4. [GPU-Accelerated PDE Solvers](#4-gpu-accelerated-pde-solvers)
5. [Performance Optimization](#5-performance-optimization)
6. [Benchmarking CPU vs GPU](#6-benchmarking-cpu-vs-gpu)
7. [Exercises](#7-exercises)

---

## 1. Why GPUs for Simulation?

### 1.1 CPU vs GPU Architecture

```
CPU (few powerful cores):          GPU (thousands of simple cores):

  ┌────────────────────┐           ┌────────────────────────────────┐
  │  Core 1 │  Core 2  │           │ ████████████████████████████   │
  │ (fast)  │ (fast)   │           │ ████████████████████████████   │
  ├─────────┼──────────┤           │ ████████████████████████████   │
  │  Core 3 │  Core 4  │           │ ████████████████████████████   │
  │ (fast)  │ (fast)   │           │     Thousands of cores         │
  ├─────────┴──────────┤           │     (individually slower)      │
  │    Large Cache     │           ├────────────────────────────────┤
  │    Branch Pred.    │           │      High-bandwidth memory     │
  │    Out-of-order    │           │      (HBM: 1-3 TB/s)          │
  └────────────────────┘           └────────────────────────────────┘

  Best for: sequential, complex      Best for: parallel, simple
  logic with branches                 operations on large arrays
```

### 1.2 When GPU Acceleration Helps

| Good for GPU | Bad for GPU |
|-------------|-------------|
| Large array operations (element-wise) | Small arrays (< 10,000 elements) |
| Matrix multiplications | Heavy branching logic |
| Stencil operations (PDE finite diff.) | Sequential algorithms |
| FFT on large grids | Irregular memory access |
| Monte Carlo (embarrassingly parallel) | I/O-bound computations |

Rule of thumb: GPU wins when N > 10,000 and the operation is data-parallel.

### 1.3 Typical Speedups

```
Operation                    Array Size    CPU Time    GPU Time    Speedup
─────────────────────────────────────────────────────────────────────────
Matrix multiply              1000×1000     50 ms       2 ms        25×
Element-wise operations      10M           120 ms      3 ms        40×
2D FFT                       4096×4096     200 ms      8 ms        25×
Finite difference stencil    1000×1000     80 ms       1.5 ms      53×
Matrix multiply              100×100       0.1 ms      0.5 ms      0.2× ✗
  (too small — transfer overhead dominates)
```

---

## 2. CUDA Programming Model

### 2.1 Key Concepts

```
CUDA Hierarchy:

  Grid ─────────────────────────────────────
  │  Block (0,0)    Block (1,0)    Block (2,0)
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │  │ Thread   │  │ Thread   │  │ Thread   │
  │  │ (0,0)    │  │ (0,0)    │  │ (0,0)    │
  │  │ Thread   │  │ Thread   │  │ Thread   │
  │  │ (1,0)    │  │ (1,0)    │  │ (1,0)    │
  │  │ ...      │  │ ...      │  │ ...      │
  │  └──────────┘  └──────────┘  └──────────┘
  │
  │  Block (0,1)    Block (1,1)    Block (2,1)
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │  │ ...      │  │ ...      │  │ ...      │
  │  └──────────┘  └──────────┘  └──────────┘
  ───────────────────────────────────────────

  Grid: collection of blocks (launched as one kernel)
  Block: group of threads (shared memory, synchronization)
  Thread: single execution unit
  Warp: 32 threads executed in lockstep (SIMT)
```

### 2.2 Memory Hierarchy

```
GPU Memory Hierarchy:

  ┌─────────────────────────────────┐
  │        Host (CPU) Memory        │  ← System RAM (slow to access)
  └──────────────┬──────────────────┘
                 │ PCIe / NVLink
  ┌──────────────┴──────────────────┐
  │       Global Memory (HBM)       │  ← Main GPU memory (large, ~GB)
  ├─────────────────────────────────┤
  │       Shared Memory (SRAM)      │  ← Per-block cache (~48-164 KB)
  ├─────────────────────────────────┤
  │       Registers                 │  ← Per-thread (fastest)
  └─────────────────────────────────┘

  Key principle: minimize data transfer between CPU and GPU.
  Once data is on GPU, keep it there for as many operations as possible.
```

---

## 3. CuPy: GPU-Accelerated NumPy

### 3.1 CuPy Basics

CuPy provides a NumPy-compatible API that runs on GPU:

```python
import numpy as np

# NumPy (CPU)
a_cpu = np.random.randn(10000, 10000)
b_cpu = np.random.randn(10000, 10000)
c_cpu = a_cpu @ b_cpu  # CPU matrix multiply

# CuPy (GPU) — same API!
import cupy as cp

a_gpu = cp.random.randn(10000, 10000)
b_gpu = cp.random.randn(10000, 10000)
c_gpu = a_gpu @ b_gpu  # GPU matrix multiply (much faster)

# Transfer between CPU and GPU
a_gpu = cp.asarray(a_cpu)        # CPU → GPU
c_cpu = cp.asnumpy(c_gpu)        # GPU → CPU
c_cpu = c_gpu.get()              # same as above
```

### 3.2 When CuPy Beats NumPy

```python
# Example: element-wise operations on large arrays
import cupy as cp
import numpy as np
import time

def benchmark(lib, N=10_000_000):
    """Compare NumPy vs CuPy for common operations."""
    x = lib.random.randn(N)
    y = lib.random.randn(N)

    if lib == cp:
        cp.cuda.Stream.null.synchronize()  # ensure GPU ready

    start = time.perf_counter()

    # Chain of element-wise operations (GPU excels here)
    z = lib.sin(x) * lib.cos(y)
    z = z + lib.exp(-x**2)
    z = lib.sqrt(lib.abs(z))
    result = z.sum()

    if lib == cp:
        cp.cuda.Stream.null.synchronize()  # wait for GPU

    elapsed = time.perf_counter() - start
    return elapsed

# Typical results:
# NumPy:  ~120 ms
# CuPy:   ~3 ms (40x speedup)
```

### 3.3 Custom CUDA Kernels in CuPy

```python
import cupy as cp

# Define a custom CUDA kernel for a specific stencil operation
laplacian_kernel = cp.ElementwiseKernel(
    'raw float64 u, int32 nx, int32 ny, float64 dx2, float64 dy2',
    'float64 lap',
    '''
    int i = _ind.get()[0];
    int ix = i / ny;
    int iy = i % ny;

    if (ix > 0 && ix < nx-1 && iy > 0 && iy < ny-1) {
        lap = (u[i+ny] - 2*u[i] + u[i-ny]) / dx2 +
              (u[i+1] - 2*u[i] + u[i-1]) / dy2;
    } else {
        lap = 0.0;
    }
    ''',
    'laplacian_2d'
)
```

---

## 4. GPU-Accelerated PDE Solvers

### 4.1 Heat Equation (Finite Difference)

```python
import numpy as np


def heat_equation_cpu(nx, ny, nt, dt, alpha=0.01):
    """2D heat equation solver on CPU using NumPy.

    du/dt = alpha * (d²u/dx² + d²u/dy²)
    Explicit Euler time stepping.
    """
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)

    u = np.zeros((nx, ny))
    # Initial condition: hot spot in center
    u[nx//4:3*nx//4, ny//4:3*ny//4] = 1.0

    rx = alpha * dt / dx**2
    ry = alpha * dt / dy**2

    for _ in range(nt):
        # Laplacian via array slicing (vectorized)
        laplacian = (
            rx * (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) +
            ry * (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2])
        )
        u[1:-1, 1:-1] += laplacian

    return u


def heat_equation_gpu(nx, ny, nt, dt, alpha=0.01):
    """Same solver on GPU using CuPy — just replace np with cp.

    The API is identical; CuPy handles GPU memory and kernels.
    """
    # In practice: import cupy as cp and replace np→cp
    # dx = 1.0 / (nx - 1)
    # dy = 1.0 / (ny - 1)
    # u = cp.zeros((nx, ny))
    # ... (same code with cp instead of np)
    pass  # See example file for full implementation
```

### 4.2 Spectral Method on GPU

```python
def spectral_diffusion_gpu(N, nt, dt, nu=0.01):
    """1D diffusion using spectral method on GPU.

    FFT-based spectral methods benefit enormously from GPU:
    CuPy's FFT uses cuFFT (NVIDIA's optimized FFT library).

    du/dt = nu * d²u/dx²
    In Fourier space: du_hat/dt = -nu * k² * u_hat
    """
    # In practice with CuPy:
    # x = cp.linspace(0, 2*cp.pi, N, endpoint=False)
    # u = cp.sin(x) + 0.5 * cp.sin(3*x)
    # k = cp.fft.fftfreq(N, d=2*cp.pi/N) * 2 * cp.pi
    #
    # for _ in range(nt):
    #     u_hat = cp.fft.fft(u)
    #     u_hat *= cp.exp(-nu * k**2 * dt)
    #     u = cp.fft.ifft(u_hat).real

    # Typical speedup: 20-50x for N > 4096
    pass
```

### 4.3 Particle Simulation

```python
def nbody_step_cpu(positions, velocities, masses, dt, G=1.0):
    """N-body gravitational simulation step (CPU).

    O(N²) pairwise force computation — perfect for GPU
    since each particle's force is independent.
    """
    N = len(positions)
    forces = np.zeros_like(positions)

    for i in range(N):
        for j in range(N):
            if i != j:
                r = positions[j] - positions[i]
                dist = np.linalg.norm(r) + 1e-10
                forces[i] += G * masses[j] * r / dist**3

    velocities += forces * dt
    positions += velocities * dt
    return positions, velocities
```

The GPU version replaces the double loop with a vectorized pairwise computation:

```python
def nbody_step_gpu_vectorized(positions, velocities, masses, dt, G=1.0):
    """Vectorized N-body on GPU (no explicit loops).

    Uses broadcasting: compute all pairwise distances at once.
    Memory: O(N²), but much faster than CPU loops for N < 50,000.
    """
    # dx[i,j] = positions[j] - positions[i] (N×N×3 tensor)
    # dist[i,j] = ||dx[i,j]||
    # forces[i] = sum_j G * m[j] * dx[i,j] / dist[i,j]^3
    #
    # All computed with CuPy broadcasting — no Python loops
    pass
```

---

## 5. Performance Optimization

### 5.1 Minimizing Data Transfer

```python
# BAD: Transfer every iteration
for step in range(1000):
    u_cpu = np.array(...)
    u_gpu = cp.asarray(u_cpu)     # CPU → GPU transfer
    result_gpu = compute(u_gpu)
    result_cpu = result_gpu.get()  # GPU → CPU transfer

# GOOD: Transfer once, compute everything on GPU
u_gpu = cp.asarray(u_cpu)  # one-time transfer
for step in range(1000):
    u_gpu = compute(u_gpu)  # stays on GPU
result_cpu = u_gpu.get()    # one-time transfer back
```

### 5.2 Memory Management

```python
# GPU memory is limited — monitor usage
# mempool = cp.get_default_memory_pool()
# print(f"GPU memory used: {mempool.used_bytes() / 1e9:.2f} GB")
# print(f"GPU memory total: {mempool.total_bytes() / 1e9:.2f} GB")
#
# Free unused memory:
# mempool.free_all_blocks()

# For large simulations: use memory-mapped arrays
# or process data in chunks that fit in GPU memory
```

### 5.3 Kernel Fusion

```python
# Separate operations = separate kernel launches
# y = cp.sin(x)
# z = cp.exp(y)
# w = z + 1.0

# Fused = one kernel launch (faster)
# Use cp.fuse for automatic fusion:

# @cp.fuse()
# def fused_op(x):
#     return cp.exp(cp.sin(x)) + 1.0
#
# w = fused_op(x)  # single kernel launch
```

---

## 6. Benchmarking CPU vs GPU

### 6.1 Fair Benchmarking

```python
def benchmark_fair(func_cpu, func_gpu, *args, warmup=5, trials=10):
    """Fair CPU vs GPU benchmark.

    Key: GPU has warmup time (JIT compilation, memory allocation).
    Always discard first few runs and synchronize properly.
    """
    # Warmup (discard)
    for _ in range(warmup):
        func_gpu(*args)
        # cp.cuda.Stream.null.synchronize()

    # Timed runs
    cpu_times = []
    for _ in range(trials):
        start = time.perf_counter()
        func_cpu(*args)
        cpu_times.append(time.perf_counter() - start)

    gpu_times = []
    for _ in range(trials):
        # cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        func_gpu(*args)
        # cp.cuda.Stream.null.synchronize()
        gpu_times.append(time.perf_counter() - start)

    return np.median(cpu_times), np.median(gpu_times)
```

### 6.2 Crossover Point

```
Performance vs Array Size:

Time │
     │  CPU
     │  ╲
     │   ╲          GPU overhead
     │    ╲        (transfer, launch)
     │     ╲           │
     │      ╲──────────┘
     │       ╲
     │  GPU   ╲─────────── GPU wins here
     │         ╲
     │          ╲
     └──────────────────────────── Array Size
              ↑
         Crossover point
         (~10,000 elements typically)
```

---

## 7. Exercises

### Exercise 1: CuPy vs NumPy Benchmark

Compare CuPy and NumPy for these operations:
1. Matrix multiplication (N = 100, 1000, 5000, 10000)
2. Element-wise operations chain (sin, cos, exp) on arrays of size 10K to 10M
3. 2D FFT on grids from 64×64 to 4096×4096
4. Find the crossover point where GPU becomes faster for each operation
5. Plot speedup vs array size

### Exercise 2: GPU Heat Equation

Implement the 2D heat equation on both CPU and GPU:
1. Grid sizes: 100×100, 500×500, 1000×1000, 2000×2000
2. Run 1000 time steps
3. Verify that CPU and GPU produce the same result (within floating point tolerance)
4. Benchmark and plot speedup vs grid size
5. Estimate GPU memory usage for each grid size

### Exercise 3: Spectral Solver on GPU

Implement a 1D advection equation using spectral methods:
1. du/dt + c * du/dx = 0 with c = 1
2. N = 256, 1024, 4096, 16384 grid points
3. Use FFT for spatial derivatives
4. Compare CPU (NumPy FFT) vs GPU (CuPy cuFFT) performance
5. Verify solution against analytical solution

### Exercise 4: N-Body Simulation

GPU-accelerate an N-body gravitational simulation:
1. N = 100, 500, 1000, 5000, 10000 particles
2. CPU: use NumPy vectorized pairwise computation
3. GPU: use CuPy broadcasting for pairwise distances
4. Run 100 time steps, benchmark both
5. Find the N where GPU becomes 10× faster than CPU

### Exercise 5: Memory-Optimized Large Simulation

Handle a simulation too large for GPU memory:
1. Create a 10000×10000 grid heat equation (800 MB in float64)
2. If GPU has < 1 GB free, split into 4 tiles with halo exchange
3. Process each tile on GPU, exchange boundaries on CPU
4. Compare with full-GPU approach (if memory allows)
5. Discuss the trade-off between tiling overhead and GPU acceleration

---

*End of Lesson 23*
