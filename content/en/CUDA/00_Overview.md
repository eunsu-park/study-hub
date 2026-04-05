# CUDA Programming — Study Guide

## Introduction

This folder provides a **comprehensive, general-purpose CUDA curriculum** — from GPU architecture fundamentals to high-performance custom kernels, scientific simulation, and multi-GPU distributed computing. The emphasis is on understanding the GPU as a parallel computing platform, not just as a deep learning accelerator.

**Why general-purpose CUDA?**
CUDA was developed for scientific computing long before the deep learning era. N-body simulations, molecular dynamics, computational fluid dynamics, Monte Carlo methods, and FFT-based signal processing all run natively on GPUs. Understanding these applications builds the parallel algorithm intuition that transfers directly to any domain — including DL kernels, which are ultimately just well-optimized parallel algorithms.

The curriculum follows this progression:

```
Hardware → Programming Model → Performance Engineering
       → Parallel Algorithms → Scientific Applications
       → Library Ecosystem → Custom HPC Kernels → Multi-GPU
```

## Target Audience

- Systems programmers (**C_Advanced**, **CPP_Advanced**) who want to program GPUs
- Scientists and engineers who need to accelerate simulations (physics, chemistry, finance, signal processing)
- ML engineers who want to understand what runs under PyTorch/TensorFlow
- Anyone who has "hit the wall" with CPU parallelism and needs GPU-scale throughput

## Prerequisites

| Topic | Required Level |
|-------|---------------|
| **C_Advanced** | Proficient — pointers, memory, compilation |
| **CPP_Advanced** | Familiar — classes, templates, operator overloading |
| **Computer_Architecture** | Familiar — cache hierarchy, SIMD, memory bandwidth, Amdahl's Law |
| Linear_Algebra | Recommended — for GEMM lessons (Block 6) |
| Numerical_Simulation | Recommended — for PDE/simulation lessons (Block 4) |
| Deep_Learning | Recommended — for DL kernel lessons (Block 6) |

## Learning Roadmap

```
┌─────────────────────────┐
│  Block 1: GPU Arch      │  L01–L07
│  & CUDA Fundamentals    │  SIMT, thread hierarchy, memory model, atomics
└──────────┬──────────────┘
           │
┌──────────▼──────────────┐
│  Block 2: Performance   │  L08–L13
│  Engineering            │  Coalescing, occupancy, roofline, profiling, streams
└──────────┬──────────────┘
           │
┌──────────▼──────────────┐
│  Block 3: Parallel      │  L14–L19
│  Algorithms             │  Reduction, scan, sort, stencil, histogram, sparse
└─────┬────────────┬───────┘
      │            │
┌─────▼──────┐  ┌─▼───────────────────────────┐
│  Block 4:  │  │  Block 5: CUDA C++ Ecosystem │  L28–L31
│  Scientific│  │  Thrust, CUB, cuBLAS, Tensor │
│  Computing │  │  Cores, Cooperative Groups   │
│  L20–L27   │  └──────────────┬───────────────┘
└─────┬──────┘                 │
      └──────────┬─────────────┘
                 │
      ┌──────────▼──────────────┐
      │  Block 6: HPC Kernels   │  L32–L36
      │  GEMM, Softmax, LN,     │
      │  FlashAttention, Quant  │
      └──────────┬──────────────┘
                 │
      ┌──────────▼──────────────┐
      │  Block 7: Multi-GPU     │  L37–L38
      │  & Capstone             │
      └─────────────────────────┘
```

## File List

| Lesson | Filename | Difficulty | Description |
|--------|----------|------------|-------------|
| **Block 1: GPU Architecture & CUDA Fundamentals** |
| L01 | `01_GPU_Architecture_Overview.md` | ⭐⭐ | SIMT vs SIMD, SM structure, warp execution, peak FLOPS/bandwidth |
| L02 | `02_CUDA_Programming_Model.md` | ⭐⭐ | `<<<grid, block>>>` syntax, thread hierarchy, first kernel |
| L03 | `03_Thread_Indexing_and_Grids.md` | ⭐⭐⭐ | 1D/2D/3D grids, global index calculation, boundary handling |
| L04 | `04_CUDA_Memory_Model.md` | ⭐⭐⭐ | Global/shared/registers/L1-L2/constant/texture, bandwidth benchmark |
| L05 | `05_Shared_Memory_and_Tiling.md` | ⭐⭐⭐ | Tiled matmul, `__syncthreads()`, bank conflicts, ncu profiling |
| L06 | `06_Warp_Execution_and_Divergence.md` | ⭐⭐⭐ | Warp divergence, predicated execution, `__shfl_sync`, warp reduction |
| L07 | `07_Atomic_Operations.md` | ⭐⭐⭐ | `atomicAdd/CAS/Exch`, histogram kernel, collision cost measurement |
| **Block 2: Performance Engineering** |
| L08 | `08_Memory_Coalescing.md` | ⭐⭐⭐ | 128-byte transactions, AoS vs SoA, stride penalty, ncu metrics |
| L09 | `09_Occupancy_and_Launch_Config.md` | ⭐⭐⭐⭐ | Register pressure, shared memory limits, `__launch_bounds__` |
| L10 | `10_Roofline_Model.md` | ⭐⭐⭐⭐ | Arithmetic intensity, compute vs memory bound, roofline chart |
| L11 | `11_Profiling_with_NCU_NSYS.md` | ⭐⭐⭐⭐ | Nsight Compute kernel metrics, Nsight Systems timeline analysis |
| L12 | `12_Streams_and_Async.md` | ⭐⭐⭐⭐ | CUDA streams, `cudaMemcpyAsync`, double-buffering pipeline |
| L13 | `13_CUDA_Graphs.md` | ⭐⭐⭐⭐ | Graph capture, stream capture, CPU launch overhead reduction |
| **Block 3: Parallel Algorithms** |
| L14 | `14_Parallel_Reduction.md` | ⭐⭐⭐ | Tree reduction, warp shuffle, multi-stage, CUB device reduce |
| L15 | `15_Parallel_Scan_Prefix_Sum.md` | ⭐⭐⭐⭐ | Hillis-Steele, Blelloch work-efficient scan, stream compaction |
| L16 | `16_Parallel_Sort.md` | ⭐⭐⭐⭐ | Bitonic sort, radix sort (CUB), merge sort, thrust::sort |
| L17 | `17_Stencil_Computations.md` | ⭐⭐⭐ | 1D/2D/3D stencil, heat equation, halo cells, periodic BC |
| L18 | `18_Histogram_and_Binning.md` | ⭐⭐⭐ | Atomic histogram, privatized (shared mem), 2D binning |
| L19 | `19_Sparse_Matrix_Ops.md` | ⭐⭐⭐⭐ | COO/CSR/CSC formats, SpMV (cuSPARSE), SpGEMM |
| **Block 4: Scientific Computing & Simulation** |
| L20 | `20_N_Body_Simulation.md` | ⭐⭐⭐⭐ | Gravitational N-body, Barnes-Hut concept, shared mem tile |
| L21 | `21_Monte_Carlo_Methods.md` | ⭐⭐⭐ | cuRAND, π estimation, Black-Scholes option pricing, variance reduction |
| L22 | `22_FFT_on_GPU.md` | ⭐⭐⭐⭐ | cuFFT API, 1D/2D/3D, batch FFT, convolution via FFT |
| L23 | `23_PDE_Solvers_Heat_Equation.md` | ⭐⭐⭐ | 2D heat equation (explicit FD), stencil kernel, stability condition |
| L24 | `24_Fluid_Dynamics_LBM.md` | ⭐⭐⭐⭐ | Lattice Boltzmann D2Q9, collision + streaming kernels, visualization |
| L25 | `25_Molecular_Dynamics.md` | ⭐⭐⭐⭐ | Lennard-Jones potential, velocity Verlet, neighbor list, energy conservation |
| L26 | `26_Image_Processing_GPU.md` | ⭐⭐⭐ | Gaussian blur, Sobel, bilateral filter, histogram equalization |
| L27 | `27_Random_Number_and_Stochastic.md` | ⭐⭐⭐⭐ | Quasi-random sequences, parallel MCMC, Metropolis-Hastings |
| **Block 5: CUDA C++ Ecosystem** |
| L28 | `28_Thrust_and_CUB.md` | ⭐⭐⭐ | Thrust STL-equivalent, CUB block/device primitives |
| L29 | `29_cuBLAS_and_cuSPARSE.md` | ⭐⭐⭐⭐ | `cublasSgemm`, batched GEMM, `cublasGemmEx` (Tensor Core path) |
| L30 | `30_Mixed_Precision_and_Tensor_Cores.md` | ⭐⭐⭐⭐⭐ | FP16/BF16/FP8, WMMA API, loss scaling, Tensor Core FLOPS measurement |
| L31 | `31_Cooperative_Groups.md` | ⭐⭐⭐⭐ | `cooperative_groups`, grid-level sync, coalesced groups |
| **Block 6: High-Performance Custom Kernels** |
| L32 | `32_GEMM_from_Scratch.md` | ⭐⭐⭐⭐⭐ | Naive→shared→register tiling→float4 vectorized; 80%+ cuBLAS target |
| L33 | `33_Softmax_and_LayerNorm_Kernels.md` | ⭐⭐⭐⭐ | Online softmax, fused LayerNorm/RMSNorm (warp shuffle), backward |
| L34 | `34_FlashAttention_Kernel.md` | ⭐⭐⭐⭐⭐ | FlashAttention-2: Q/K/V tiling, online softmax, causal mask |
| L35 | `35_Quantized_Kernels_INT8.md` | ⭐⭐⭐⭐ | INT8 GEMM, `dp4a`, dequant-fused epilogue, INT4 weight-only |
| L36 | `36_Fused_Kernel_Patterns.md` | ⭐⭐⭐⭐ | Bias+activation+dropout fusion, CUTLASS epilogue fusion |
| **Block 7: Multi-GPU & Capstone** |
| L37 | `37_Multi_GPU_and_NCCL.md` | ⭐⭐⭐⭐⭐ | NVLink, P2P transfer, NCCL AllReduce, tensor parallelism |
| L38 | `38_Capstone_CUDA_Application.md` | ⭐⭐⭐⭐⭐ | Choice: (A) fluid simulation + visualization / (B) LLM inference engine |

**Total: 38 lessons**

## What Makes This Course Different

Most CUDA courses focus exclusively on deep learning kernels. This course covers the **full GPU computing ecosystem**:

| Domain | Lessons | Key Techniques |
|--------|---------|---------------|
| Physics | L20 (N-body), L25 (MD) | Particle systems, potential energy, neighbor lists |
| Fluid Dynamics | L24 (LBM) | Lattice Boltzmann, collision operators |
| Numerical Methods | L23 (PDE), L22 (FFT) | Finite difference, spectral methods |
| Statistics | L21, L27 | Monte Carlo, MCMC, quasi-random |
| Image Processing | L26 | Convolution, edge detection, bilateral filter |
| Linear Algebra | L32 (GEMM), L19 (sparse) | Dense and sparse matrix operations |
| Deep Learning | L33–L36 | FlashAttention, quantization, fused ops |

The parallel algorithm foundations in Block 3 (reduction, scan, sort, stencil) are the building blocks for **all** of the above domains.

## Difficulty Curve

```
Block 1 │▓▓░░░░░│  Moderate — CUDA syntax is new but approachable
Block 2 │▓▓▓░░░░│  Upper-moderate — hardware internals reasoning required
Block 3 │▓▓▓░░░░│  Upper-moderate — parallel thinking shift needed (Wall #1)
Block 4 │▓▓░░░░░│  Moderate–upper — varies by domain; LBM/MD need physics background
Block 5 │▓▓░░░░░│  Moderate — library API learning
Block 6 │▓▓▓▓▓▓▓│  Expert — custom GEMM and FlashAttention (Wall #2)
Block 7 │▓▓▓▓▓░░│  Advanced — distributed + integration challenge
```

**Peak Difficulty Lessons**: L30 (Tensor Core WMMA), L32 (GEMM from scratch), L34 (FlashAttention kernel), L37 (NCCL + tensor parallelism)

## Key Milestones

| After | You Can |
|-------|---------|
| L05 | Write a tiled matmul in CUDA faster than NumPy; understand shared memory |
| L10 | Profile any kernel with `ncu`/`nsys` and explain the top bottleneck |
| L14 | Implement parallel reduction — the foundational GPU primitive |
| L19 | Implement any parallel algorithm using reduction/scan/sort/stencil |
| L24 | Run a 2D fluid simulation (LBM) on GPU with real-time visualization |
| L27 | Implement GPU-accelerated Monte Carlo and MCMC sampling |
| L32 | Build a custom SGEMM achieving 80%+ cuBLAS performance |
| L34 | Implement FlashAttention-2 from scratch; validated against PyTorch |
| L38 | Build a complete CUDA application from scratch (simulation or inference) |

## Environment Setup

### NVIDIA GPU Required

```bash
# Check GPU and CUDA version
nvidia-smi
nvcc --version

# Install CUDA Toolkit (if not present)
# Ubuntu: https://developer.nvidia.com/cuda-downloads
# macOS: Use Metal instead (code examples include Metal alternatives)

# Verify: run the first example
cd study-hub/examples/CUDA/02_CUDA_Programming_Model/
nvcc -O2 -o vector_add vector_add.cu
./vector_add
```

### Profiling Tools

```bash
# Nsight Compute (kernel-level profiling)
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
    --metrics l1tex__t_bytes.sum \
    ./my_kernel

# Nsight Systems (system-level timeline)
nsys profile --stats=true ./my_application
```

### Build System

```makefile
# Typical Makefile for CUDA examples
NVCC    = nvcc
NVFLAGS = -O3 -arch=sm_80 --use_fast_math -Xcompiler -Wall
LIBS    = -lcublas -lcurand -lcufft

my_kernel: my_kernel.cu
	$(NVCC) $(NVFLAGS) -o $@ $^ $(LIBS)
```

> **GPU Architecture Flags**: `-arch=sm_80` for A100, `-arch=sm_89` for RTX 4090, `-arch=sm_90` for H100. Use `-arch=sm_75` for RTX 2080 / T4.

## Related Topics

- **[DL_Scratch_C](../DL_Scratch_C/00_Overview.md)**: The C/C++ DL implementation course — many of the kernels here (attention, GEMM) GPU-accelerate what was built there
- **[Deep_Learning](../Deep_Learning/00_Overview.md)**: PyTorch-based DL — understand what PyTorch calls under the hood
- **[Computer_Architecture](../Computer_Architecture/00_Overview.md)**: CPU architecture prerequisite — cache, SIMD, memory hierarchy
- **[Numerical_Simulation](../Numerical_Simulation/00_Overview.md)**: CPU-based numerical methods that Block 4 GPU-accelerates

## Study Tips

1. **Profile before optimizing**: Use `ncu` on your very first kernel. You will be surprised by what the bottleneck actually is.
2. **Think in warps, not threads**: Every branch or memory access affects 32 threads at once. Mental model: warp = one hardware execution unit.
3. **Measure, don't guess**: GPU performance is counterintuitive. A "slow" kernel often stalls on memory, not compute. The roofline model (L10) gives you the answer.
4. **Start with Thrust**: For prototyping, use Thrust/CUB. Write custom kernels only when benchmarking proves Thrust is a bottleneck.
5. **Domain knowledge accelerates**: The LBM lesson (L24) is easier if you understand fluids; N-body (L20) is easier if you know Newtonian mechanics. Don't skip the physics context.

## Learning Outcomes

After completing this course, you will be able to:

- ✅ Write CUDA C kernels with correct thread indexing and memory access patterns
- ✅ Profile kernels with Nsight Compute and identify top bottlenecks
- ✅ Implement parallel reduction, scan, sort, and stencil operations from scratch
- ✅ Build GPU-accelerated scientific simulations (N-body, LBM, PDE solvers, Monte Carlo)
- ✅ Use cuBLAS, cuFFT, cuRAND, cuSPARSE, and Thrust productively
- ✅ Write a high-performance SGEMM achieving 80%+ of cuBLAS performance
- ✅ Implement FlashAttention-2 in CUDA from the algorithm paper
- ✅ Synchronize gradients across GPUs using NCCL AllReduce
- ✅ Apply GPU acceleration to any compute-intensive problem in your domain

---

Start with `01_GPU_Architecture_Overview.md` to build the hardware mental model, then `02_CUDA_Programming_Model.md` to write your first kernel.
