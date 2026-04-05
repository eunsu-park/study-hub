# 09. Occupancy and Launch Configuration

**Previous**: [Memory Coalescing](./08_Memory_Coalescing.md) | **Next**: [Roofline Model](./10_Roofline_Model.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Calculate occupancy from register count, shared memory, and block size
2. Identify which resource is the occupancy-limiting constraint for a given kernel
3. Apply `__launch_bounds__` to control register allocation
4. Use the CUDA Occupancy Calculator API and Nsight Compute
5. Understand when higher occupancy does NOT improve performance

---

## 1. Occupancy Recap

**Occupancy** = active warps per SM / maximum warps per SM

On Ampere (A100/RTX 3090): maximum 64 warps per SM (2048 threads).

Three resources limit occupancy:
1. **Registers** (64K per SM)
2. **Shared memory** (48–228 KB per SM, configurable)
3. **Block size** (must allocate whole blocks)

The **minimum** of the three limits applies.

---

## 2. Register-Limited Occupancy

The formula: `max_threads = 64K registers / registers_per_thread`

```
registers/thread  →  max_threads/SM  →  max_warps/SM  →  occupancy
      16                4096                128               100%  (exceeds SM max of 64 warps)
      32                2048                 64               100%  (exactly at SM max)
      48                1365                 42                66%
      64                1024                 32                50%
      96                 682                 21                33%
     128                 512                 16                25%
     256                 256                  8                12%
```

**How to check register usage**:

```bash
# Method 1: compiler verbose output
nvcc -Xptxas -v mykernel.cu
# Output: used 32 registers, ...

# Method 2: Nsight Compute
ncu --metrics launch__registers_per_thread ./my_kernel
```

---

## 3. Shared-Memory-Limited Occupancy

Each block consumes shared memory. Fewer blocks → fewer warps → lower occupancy.

```
A100: 228 KB shared memory per SM (when fully configured)

If block uses 48 KB shared memory and block has 256 threads (8 warps):
  max blocks/SM = floor(228 KB / 48 KB) = 4 blocks
  active warps  = 4 × 8 = 32 warps
  occupancy     = 32/64 = 50%

If block uses 8 KB shared memory:
  max blocks/SM = floor(228 KB / 8 KB) = 28, capped at SM block limit (32)
  active warps  = min(28, 32) × 8 = 224 warps → capped at 64 warps max
  occupancy     = 64/64 = 100%
```

**Shared memory vs L1 ratio** is configurable on Ampere:

```c
// Set a specific shared memory size for a kernel
cudaFuncSetAttribute(myKernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    96 * 1024);  // 96 KB for this kernel

// System-wide setting
cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);  // 8-byte banks
```

---

## 4. Block-Size-Limited Occupancy

The SM must allocate whole blocks. If blocks are too large, occupancy suffers:

```
SM max: 64 warps (2048 threads), max 32 blocks

Block size = 1024 (32 warps):
  max blocks = min(2048/1024, 32) = 2 blocks
  active warps = 2 × 32 = 64 → 100% occupancy ✓

Block size = 1536 (48 warps):
  max blocks = floor(2048/1536) = 1 block
  active warps = 1 × 48 = 48 → 75% occupancy ✗
  (1536 is not a factor of 2048 → 1 block wastes 512 thread slots)

Block size = 96 (3 warps, a multiple of 32 but not a power of two):
  max blocks = min(2048/96, 32) = 21 blocks
  active warps = 21 × 3 = 63 → 98% occupancy
  (96 = 3×32 is a valid warp-aligned size; near-perfect occupancy here,
   but the large block count adds scheduling overhead)
```

**Rule of thumb**: block sizes of 128, 256, or 512 are universally safe. Avoid 192, 320, 384, 448, 640, 768, 896, 1536 — these often leave warp slots unused.

---

## 5. The Occupancy Calculator API

```c
#include <cuda_runtime.h>

// Ask CUDA for the optimal block size
int minGridSize, optimalBlockSize;
cudaOccupancyMaxPotentialBlockSize(
    &minGridSize,       // minimum grid size for full occupancy
    &optimalBlockSize,  // block size that maximizes occupancy
    myKernel,           // kernel function pointer
    0,                  // dynamic shared memory per block (0 = none)
    0                   // max block size constraint (0 = no limit)
);

// Compute actual occupancy for a given block size
int activeWarps, maxWarps;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &activeWarps,   // returns: active blocks per SM (multiply by warps/block)
    myKernel,
    blockSize,
    sharedMemBytes
);
cudaDeviceGetAttribute(&maxWarps, cudaDevAttrMaxThreadsPerMultiProcessor, 0);
float occupancy = (float)(activeWarps * blockSize / 32) / (maxWarps / 32);
printf("Occupancy: %.1f%%\n", occupancy * 100);
```

---

## 6. `__launch_bounds__`: Compiler Register Hints

Tell the compiler the maximum block size this kernel will ever be launched with, so it can limit register allocation:

```c
// Guarantee: this kernel is only launched with blocks of ≤ 256 threads
// and at least 2 blocks per SM (minimum occupancy hint)
__global__ __launch_bounds__(256, 2)
void myKernel(float *data, int n) {
    // ...
}
```

Effect:
- Compiler may use fewer registers (to hit the occupancy target)
- Possibly increases register spilling (trade-off)
- Use when the compiler over-allocates registers and you know the launch configuration

```c
// More aggressive: force low register count for high occupancy
__global__ __launch_bounds__(128, 4)  // 128 threads max, ≥4 blocks per SM
void high_occupancy_kernel(float *data) {
    // Compiler will try to fit in 64K/(128*4) = 128 registers max per thread
    // This may cause spilling if the kernel is complex
}
```

---

## 7. Occupancy vs Performance: The Non-Linear Relationship

**Higher occupancy does not always mean higher throughput.** This is one of the most important subtleties in GPU optimization.

### Case 1: Latency-bound kernel (high occupancy helps)

A kernel that stalls frequently on global memory:
```
- 50% occupancy: some stalls can't be hidden → GPU partially idle
- 100% occupancy: every stall is covered by a ready warp → full utilization
```

### Case 2: Compute-bound kernel (occupancy doesn't matter much)

A kernel spending all time on FP arithmetic with no stalls:
```
- 25% occupancy but warps never stall: FP units always busy → same throughput
- Adding more warps doesn't help when there's no latency to hide
```

### Case 3: Register-heavy kernel (lower occupancy may be faster)

More registers per thread = more accumulators = fewer memory accesses:
```
- L32's GEMM kernel with 128 registers/thread (25% occupancy) outperforms
  the same kernel with 32 registers/thread (100% occupancy) because
  the high-register version computes more per memory byte loaded
```

---

## 8. Profiling Occupancy with Nsight Compute

```bash
ncu --metrics \
    sm__warps_active.avg.pct_of_peak_sustained_active,\
    sm__active_cycles_avg,\
    sm__warps_eligible.avg,\
    launch__occupancy_limit_registers,\
    launch__occupancy_limit_shared_mem,\
    launch__occupancy_limit_warps \
    ./my_kernel
```

Key metrics:
- `sm__warps_active.avg.pct_of_peak_sustained_active`: actual achieved occupancy
- `launch__occupancy_limit_registers`: 1 if registers are the binding constraint
- `launch__occupancy_limit_shared_mem`: 1 if shared memory is the binding constraint
- `sm__warps_eligible.avg`: how many warps are ready to schedule on average

If `warps_eligible` is consistently low, you have a real occupancy problem. If it's high but throughput is still low, the bottleneck is something else (compute or memory bandwidth).

---

## 9. Worked Example: Optimizing a Reduction Kernel

```c
// Version 1: naive — no shared memory optimization, 32 registers/thread
__global__ void reduce_v1(float *in, float *out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // ... naive reduction
}
// Occupancy: 100% (register count allows it)
// Problem: no latency hiding beyond natural warp switching

// Version 2: warp shuffle, 20 registers/thread
__global__ __launch_bounds__(256)
void reduce_v2(float *in, float *out, int n) {
    float val = (blockIdx.x * blockDim.x + threadIdx.x < n) ?
                in[blockIdx.x * blockDim.x + threadIdx.x] : 0.0f;
    val = warp_reduce_sum(val);
    if (threadIdx.x % 32 == 0) out[blockIdx.x * 32 + threadIdx.x / 32] = val;
}
// Occupancy: 100%
// Performance: 3× faster than v1 (warp shuffle + coalesced access)
```

---

## Key Takeaways

- Occupancy = min(register limit, shared memory limit, block size limit)
- Check which constraint limits you: `launch__occupancy_limit_*` metrics in ncu
- Use `cudaOccupancyMaxPotentialBlockSize` to find the optimal block size automatically
- `__launch_bounds__` guides the compiler to reduce register allocation — may introduce spilling
- **High occupancy helps latency-hiding; it doesn't help compute-bound kernels**
- Target 50%+ occupancy for most kernels; lower is acceptable if the kernel is compute-bound with high register counts

---

**Next**: [10. Roofline Model](./10_Roofline_Model.md) — Build the roofline chart for your GPU, calculate arithmetic intensity, and determine whether any kernel is memory- or compute-bound.
