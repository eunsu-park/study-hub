# 01. GPU Architecture Overview

**Next**: [CUDA Programming Model](./02_CUDA_Programming_Model.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the fundamental design difference between CPUs and GPUs
2. Describe the GPU execution model: SMs, warps, and SIMT
3. Calculate theoretical peak FLOPS and memory bandwidth for a given GPU
4. Understand the occupancy model: why more threads can hide latency
5. Choose the right GPU for a given workload type

---

## 1. CPU vs GPU: A Design Philosophy Contrast

CPUs and GPUs solve fundamentally different problems:

| Aspect | CPU | GPU |
|--------|-----|-----|
| **Core count** | 4–128 | 1,000–18,000+ |
| **Clock speed** | 3–5 GHz | 1–2 GHz |
| **Cache per core** | Large (MB-range per core) | Small (shared across SMs) |
| **Optimization goal** | Minimize **latency** for a single thread | Maximize **throughput** for thousands of threads |
| **Branch prediction** | Aggressive (>15 stage pipeline) | Simple (divergence kills throughput) |
| **SIMD width** | 8–16 FP32 (AVX-512) | 32 FP32 per warp, 128 warps/SM |
| **Memory model** | Coherent, uniform latency | High-bandwidth, high-latency DRAM |

**The key insight**: A GPU trades single-thread performance for **massive parallel throughput**. One warp of 32 threads executes the same instruction simultaneously. While one warp waits for memory, another warp runs — this is **latency hiding through parallelism**.

---

## 2. GPU Hardware Hierarchy

### Streaming Multiprocessors (SMs)

The SM is the GPU's fundamental compute unit, analogous to a CPU core but much more specialized:

```
┌─────────────────────────────────────────────────────┐
│                  Streaming Multiprocessor (SM)       │
│                                                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐  │
│  │  Warp 0  │ │  Warp 1  │ │  Warp 2  │ │  ...   │  │
│  │ 32 thds  │ │ 32 thds  │ │ 32 thds  │ │        │  │
│  └──────────┘ └──────────┘ └──────────┘ └────────┘  │
│                    │                                  │
│  ┌─────────────────▼──────────────────────────────┐  │
│  │              Warp Scheduler                     │  │
│  │    (selects ready warps each cycle)             │  │
│  └─────────────────────────────────────────────────┘  │
│                    │                                  │
│  ┌─────────────────▼──────────────────────────────┐  │
│  │  INT32/FP32 ALUs  │  FP64 ALUs  │  Tensor Cores │  │
│  └─────────────────────────────────────────────────┘  │
│                                                       │
│  ┌─────────────────────────────────────────────────┐  │
│  │      Shared Memory / L1 Cache (48–228 KB)       │  │
│  └─────────────────────────────────────────────────┘  │
│                                                       │
│  ┌─────────────────────────────────────────────────┐  │
│  │   Register File (64K × 32-bit registers)        │  │
│  └─────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

**NVIDIA A100 GPU:**
- 108 SMs
- 64 FP32 CUDA cores per SM → **6,912 CUDA cores** total
- 32 warps per SM → 3,456 warps simultaneously active
- 40–80 GB HBM2e memory, **2 TB/s** bandwidth
- **312 TFLOPS** FP16 Tensor Core, **77 TFLOPS** FP32

### Memory Hierarchy

```
Registers     ←→  L1 / Shared    ←→  L2 Cache    ←→  HBM / GDDR
~0 cycles          1–5 cycles       50–100 cycles    400–700 cycles
256 KB/SM          48–228 KB/SM     40–80 MB         40–80 GB total
~19 TB/s           ~19 TB/s         ~5 TB/s          ~2 TB/s (A100)
```

---

## 3. The Warp: GPU Execution Unit

A **warp** is a group of **32 threads** that execute the same instruction simultaneously (SIMT — Single Instruction, Multiple Threads).

### SIMT vs SIMD

```
SIMD (CPU, AVX2):   one instruction, 8 floats, same data vector
SIMT (GPU, warp):   one instruction, 32 threads, each with OWN data
```

In SIMT, each thread in a warp has:
- Its own **registers** (private)
- Its own **program counter** (but all warps in a block share the same PC when not diverged)
- Access to **shared memory** (block-level) and **global memory** (device-level)

### Warp Scheduling and Latency Hiding

An SM holds many warps resident simultaneously. When one warp stalls on memory (400+ cycle latency), the warp scheduler **immediately switches** to another ready warp:

```
Cycle:  0   400   800   1200
Warp 0: LOAD  ....  COMPUTE  STORE
Warp 1:      LOAD  ....  COMPUTE
Warp 2:           LOAD  ....  COMPUTE
Warp 3:                LOAD  ....
                                ↑
                     No idle cycles — latency hidden
```

This is the core reason GPUs tolerate high memory latency: **occupancy** (the ratio of active warps to the maximum possible) determines how well latency is hidden.

---

## 4. Occupancy

**Occupancy** = active warps per SM / maximum warps per SM

Maximum warps per SM (A100): 64

Occupancy is limited by:
1. **Register usage**: Each thread uses registers; too many → fewer threads fit → lower occupancy
2. **Shared memory**: Each block uses shared memory; too much → fewer blocks → fewer warps
3. **Block size**: Must be a multiple of 32; very small blocks waste potential occupancy

```
Example: kernel uses 32 registers/thread, block size = 128 (4 warps)
  Register limit: 256KB / (32 regs × 4 bytes × 32 threads) = 64 warps ✓
  Max blocks/SM = 64 warps / 4 warps per block = 16 blocks
  Occupancy = 64/64 = 100%

If kernel uses 64 registers/thread:
  Register limit: 256KB / (64 × 4 × 32) = 32 warps
  Occupancy = 32/64 = 50%
```

> **Higher occupancy is usually better** but not always — sometimes a kernel with fewer threads per SM but more registers per thread achieves better throughput.

---

## 5. GPU Memory Bandwidth: The Real Bottleneck

For most kernels, **memory bandwidth is the limiting factor**, not compute.

```
A100 theoretical:
  Peak FP32:        19.5 TFLOPS
  Memory bandwidth: 2.0 TB/s

Bandwidth-to-compute ratio: 2.0e12 / 19.5e12 = 0.103 bytes per FLOP
Arithmetic intensity needed to be compute-bound: > 9.8 FLOPs/byte

Vector add (y = a*x + b):
  FLOPs = 2N (one mul, one add)
  Bytes = 3N × 4 (read x, read y, write y)
  AI = 2N / 12N = 0.17 FLOPs/byte → MEMORY BOUND (×58 below threshold)

Matrix multiply (C = A × B, NxN):
  FLOPs = 2N^3
  Bytes = 3N^2 × 4 (read A, B; write C)  [ignoring reuse]
  AI = 2N^3 / 12N^2 = N/6
  For N=1024: AI = 170 FLOPs/byte → COMPUTE BOUND ✓
```

This is the **roofline model** — we will use it in L10 to analyze real kernels.

---

## 6. GPU Generations Comparison

| GPU | Arch | SMs | FP32 TFLOPS | BW (TB/s) | Tensor (FP16) |
|-----|------|-----|-------------|-----------|---------------|
| RTX 3090 | Ampere | 82 | 35.6 | 0.94 | 285 TFLOPS |
| A100 (80GB) | Ampere | 108 | 19.5 | 2.0 | 312 TFLOPS |
| RTX 4090 | Ada | 128 | 82.6 | 1.01 | 660 TFLOPS |
| H100 SXM | Hopper | 132 | 67.0 | 3.35 | 1979 TFLOPS (FP8) |
| L40S | Ada | 142 | 91.6 | 0.86 | 733 TFLOPS |

**Key observations**:
- H100 has **67% more memory bandwidth** than A100 — crucial for memory-bound kernels
- RTX 4090 has 4× the FP32 TFLOPS of A100 but only half the memory bandwidth — better for compute-bound but worse for memory-bound
- Tensor Cores (FP16/BF16) are 4–16× faster than CUDA cores — DL training uses them heavily

---

## 7. Calculating Theoretical Performance

### For a kernel's machine balance:

```python
# Example: A100 GPU
compute_tflops = 19.5    # FP32 TFLOPS
bandwidth_tbps = 2.0     # TB/s

# Machine ridge point (bytes/FLOP where compute = memory)
ridge = compute_tflops / bandwidth_tbps  # = 9.75 FLOPs/byte

# A kernel with AI = 5 FLOP/byte is memory-bound:
#   Peak achievable throughput = 5 * 2.0 TB/s = 10 TFLOPS  (not 19.5)
```

### Runtime estimation:

```python
N = 1024  # Matrix dimension (N×N matmul)
flops = 2 * N**3        # = 2.1 billion FLOPs
bytes = 3 * N**2 * 4    # = 12.6 MB (reading A, B, writing C)
ai    = flops / bytes   # = 167 FLOPs/byte (strongly compute-bound)

# Compute bound:
time_compute = flops / (19.5e12)   # = 0.11 ms
# Memory bound:
time_memory  = bytes / (2.0e12)    # = 0.006 ms

actual_time = max(time_compute, time_memory)  # = 0.11 ms (compute-bound)
```

---

## 8. Hands-On: GPU Info Query

```c
// gpu_info.cu
#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int device_count;
    cudaGetDeviceCount(&device_count);
    printf("CUDA devices found: %d\n", device_count);

    for (int i = 0; i < device_count; i++) {
        struct cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        printf("\nDevice %d: %s\n", i, prop.name);
        printf("  Compute capability:    %d.%d\n",
               prop.major, prop.minor);
        printf("  SMs:                   %d\n",
               prop.multiProcessorCount);
        printf("  Warp size:             %d threads\n",
               prop.warpSize);
        printf("  Max threads per SM:    %d\n",
               prop.maxThreadsPerMultiProcessor);
        printf("  Max warps per SM:      %d\n",
               prop.maxThreadsPerMultiProcessor / prop.warpSize);
        printf("  Global memory:         %.1f GB\n",
               (float)prop.totalGlobalMem / 1e9);
        printf("  Memory bandwidth:      %.1f GB/s\n",
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6);
        printf("  L2 cache:              %d MB\n",
               prop.l2CacheSize / (1024*1024));
        printf("  Shared memory per SM:  %zu KB\n",
               prop.sharedMemPerMultiprocessor / 1024);
        printf("  Registers per SM:      %d\n",
               prop.regsPerMultiprocessor);
    }
    return 0;
}
```

```bash
nvcc -o gpu_info gpu_info.cu && ./gpu_info
```

---

## Key Takeaways

- GPUs optimize for **throughput**, not single-thread latency: thousands of simple cores vs. dozens of complex ones
- A **warp** (32 threads) is the atomic execution unit — all 32 threads execute the same instruction
- **Occupancy** = active warps / max warps; high occupancy hides memory latency through thread switching
- **Memory bandwidth** (not compute) is the bottleneck for most kernels — understand your kernel's arithmetic intensity
- The roofline model predicts whether a kernel is memory-bound or compute-bound and sets performance expectations

---

**Next**: [02. CUDA Programming Model](./02_CUDA_Programming_Model.md) — Write your first CUDA kernel, understand the `<<<grid, block>>>` launch syntax, and learn how threads map to hardware.
