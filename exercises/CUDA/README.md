# CUDA Exercises

Hands-on exercises for the CUDA course (L01–L38). Each block maps to a lesson group; exercises are self-contained `.cu` files you compile with `nvcc`.

## Hardware Requirements

| Block | Minimum Compute Capability | Notes |
|-------|---------------------------|-------|
| Block 1 – Fundamentals | sm_60 | Any Pascal or newer GPU |
| Block 2 – Performance | sm_70 | Volta+ for warp-level intrinsics; Ampere preferred for `ncu` |
| Block 3 – Algorithms | sm_60 | Any Pascal or newer |
| Block 4 – Simulation | sm_60 | cuRAND + cuFFT must be installed |
| Block 5 – Ecosystem | sm_80 | WMMA requires sm_70+; cuBLAS/Thrust standard |
| Block 6 – Production Kernels | sm_80 | `dp4a` requires sm_61+; FlashAttention tile needs sm_80 |
| Block 7 – Capstone | sm_80 (multi-GPU for 7.1) | NCCL requires matching driver; two GPUs for Ex 7.1 |

## Compilation Template

```bash
nvcc -O2 -arch=sm_80 -o exercise exercise.cu
# Link libraries as needed:
nvcc -O2 -arch=sm_80 -o exercise exercise.cu -lcublas -lcufft -lcurand
```

## Exercise Index

### Block 1 — Fundamentals (`block1_fundamentals.md`) — L01–L07

| ID | Title | Difficulty | Key Concept |
|----|-------|-----------|-------------|
| 1.1 | Vector Scale | Beginner | cudaMalloc, kernel launch, cudaMemcpy |
| 1.2 | 2D Matrix Addition | Beginner | 2D grid/block indexing, non-square matrices |
| 1.3 | Global Memory Bandwidth Benchmark | Intermediate | Streaming copy, GB/s measurement |
| 1.4 | Shared Memory Transpose | Intermediate | Bank conflict avoidance, padding |
| 1.5 | Warp Shuffle Reduction | Intermediate | `__shfl_down_sync`, warp lanes |

### Block 2 — Performance (`block2_performance.md`) — L08–L13

| ID | Title | Difficulty | Key Concept |
|----|-------|-----------|-------------|
| 2.1 | Stride Access Penalty | Beginner | Memory coalescing, `ncu` metrics |
| 2.2 | Occupancy Tuning | Intermediate | `cudaOccupancyMaxPotentialBlockSize` |
| 2.3 | Roofline Chart | Intermediate | Arithmetic intensity, bandwidth bound vs compute bound |
| 2.4 | Async Overlap with Streams | Intermediate | `cudaMemcpyAsync`, multi-stream pipelining |
| 2.5 | CUDA Graph Launch | Advanced | `cudaGraphCapture`, launch overhead |

### Block 3 — Algorithms (`block3_algorithms.md`) — L14–L19

| ID | Title | Difficulty | Key Concept |
|----|-------|-----------|-------------|
| 3.1 | Multi-block Reduction | Intermediate | Two-pass reduction, global atomics |
| 3.2 | Blelloch Parallel Scan | Advanced | Work-efficient scan, up/down sweep |
| 3.3 | Bitonic Sort | Advanced | Bitonic merge network, power-of-2 |
| 3.4 | 2D Laplacian Stencil | Advanced | Shared memory halos, ghost cells |
| 3.5 | Privatized Histogram | Intermediate | Shared memory privatization, atomicAdd |

### Block 4 — Simulation (`block4_simulation.md`) — L20–L27

| ID | Title | Difficulty | Key Concept |
|----|-------|-----------|-------------|
| 4.1 | N-body Shared Tile | Intermediate | Tile-based force computation |
| 4.2 | Monte Carlo π | Beginner | cuRAND, statistical verification |
| 4.3 | cuFFT Low-pass Filter | Intermediate | Forward + inverse FFT, spectral zeroing |
| 4.4 | 2D Heat Equation PDE | Intermediate | Iterative time stepping, convergence check |
| 4.5 | Gaussian Blur | Intermediate | 2D stencil, boundary handling |

### Block 5 — Ecosystem (`block5_ecosystem.md`) — L28–L31

| ID | Title | Difficulty | Key Concept |
|----|-------|-----------|-------------|
| 5.1 | Thrust Sort | Beginner | `thrust::device_vector`, `thrust::sort` |
| 5.2 | cuBLAS SGEMM | Intermediate | Column-major layout, alpha/beta |
| 5.3 | WMMA Half-precision GEMM | Advanced | `nvcuda::wmma`, tensor core fragment |
| 5.4 | Cooperative Groups Grid Sync | Advanced | `cooperative_groups::grid_group` |

### Block 6 — Production Kernels (`block6_kernels.md`) — L32–L36

| ID | Title | Difficulty | Key Concept |
|----|-------|-----------|-------------|
| 6.1 | Tiled GEMM v1→v2 | Advanced | Shared memory tiles, register blocking |
| 6.2 | Online Softmax | Advanced | Single-pass numerically stable softmax |
| 6.3 | FlashAttention Tile | Expert | Running max/sum statistics, SRAM tile |
| 6.4 | INT8 GEMV with dp4a | Advanced | `__dp4a`, quantization, dequantization |
| 6.5 | Fused Bias + ReLU | Intermediate | Kernel fusion, elementwise ops |

### Block 7 — Capstone (`block7_capstone.md`) — L37–L38

| ID | Title | Difficulty | Key Concept |
|----|-------|-----------|-------------|
| 7.1 | NCCL AllReduce | Advanced | Multi-GPU collective, ncclAllReduce |
| 7.2 | LBM D2Q9 + Obstacle | Expert | Bounce-back BC, streamline visualization |
| 7.3 | GPT-2-style Token Generation | Expert | Embed → Transformer block → Unembed |

## Grading / Verification Philosophy

Every exercise includes:
1. **Correctness check** — compare GPU result to CPU reference or known value.
2. **Performance target** — expected GB/s, GFLOP/s, or speedup ratio.
3. **Hints** — which CUDA API calls and memory space to use.

Passing = correct output AND performance within 20% of the stated target on a mid-range GPU (e.g., RTX 3080 / A100).
