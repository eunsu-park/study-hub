# Lesson 9 — Occupancy and Launch Configuration (per-lesson exercise)

Prerequisites: L03 (thread indexing), L04 (memory model).

Compile: `nvcc -O3 -arch=sm_80 ex.cu -o ex`

**Occupancy** is the fraction of an SM's maximum thread capacity that your kernel actually uses at runtime. Higher occupancy lets the SM hide memory latency by switching between warps; lower occupancy can leave half the SM idle.

Three resources cap occupancy:

1. **Registers per thread** (each SM has e.g. 65536 registers — divide by `threads × registers_per_thread`)
2. **Shared memory per block** (e.g. 48 KB or 96 KB depending on the GPU)
3. **Threads per block** (max 1024)

---

## Exercise 9.1 — Querying Occupancy

**Difficulty**: ★

### Problem

Use `cudaOccupancyMaxActiveBlocksPerMultiprocessor` to query, for a given kernel and block size, how many concurrent blocks an SM can host.

### Starter

```cuda
#include <cstdio>
#include <cuda_runtime.h>

__global__ void light_kernel(float *out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = 0.0f;
}

int main(void) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    for (int block_size : {64, 128, 256, 512, 1024}) {
        int active_blocks = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks, light_kernel, block_size, /* shmem */ 0);
        int active_warps = active_blocks * (block_size / 32);
        int max_warps    = prop.maxThreadsPerMultiProcessor / 32;
        printf("block=%4d  active_blocks=%d  active_warps=%d/%d  occ=%.0f%%\n",
               block_size, active_blocks,
               active_warps, max_warps,
               100.0 * active_warps / max_warps);
    }
    return 0;
}
```

You will typically see the highest occupancy at `block=128` or `256`, dropping at the extremes (small blocks waste resources on per-block overhead; very large blocks force a smaller `active_blocks` count).

---

## Exercise 9.2 — Register Pressure vs. Occupancy

**Difficulty**: ★★

### Problem

Write a kernel that uses many registers — declare a local array of, say, 64 floats:

```cuda
__global__ void heavy_register(float *out, int N) {
    float buf[64];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    /* fill buf, do some math */
    for (int k = 0; k < 64; k++) buf[k] = i * 0.001f * (float)k;
    float sum = 0;
    for (int k = 0; k < 64; k++) sum += buf[k] * buf[(k + 1) % 64];
    if (i < N) out[i] = sum;
}
```

Compile with `nvcc -O3 --ptxas-options=-v` and read the per-thread register count from the output. Then re-query occupancy with `cudaOccupancyMaxActiveBlocksPerMultiprocessor`. You should see occupancy collapse to 25% or less.

Now try `__launch_bounds__(256, 4)` (block size hint + minimum blocks per SM) on the kernel and recompile. The compiler will spill some registers to local memory to keep the launch bound viable. Time the new kernel — sometimes occupancy gains beat the spill cost, sometimes not.

---

## Exercise 9.3 — Shared Memory vs. Occupancy

**Difficulty**: ★★

Allocate, say, 32 KB of shared memory per block. With 96 KB of shared memory per SM, you can fit 3 blocks at once. With 48 KB per SM, only 1. Demonstrate by querying `cudaOccupancyMaxActiveBlocksPerMultiprocessor` with varying `dynamic_shmem` argument.

The lesson: shared memory is a finite budget. Doubling your tile size halves the number of concurrent blocks per SM, so the bandwidth gain may not materialize.

---

## Exercise 9.4 — `cudaOccupancyMaxPotentialBlockSize` — Bonus

**Difficulty**: ★

CUDA can pick a block size that maximizes theoretical occupancy for you:

```cuda
int min_grid_size, block_size;
cudaOccupancyMaxPotentialBlockSize(
    &min_grid_size, &block_size, my_kernel, 0, 0);
```

Use this on each of your previous kernels and compare its choice to your hand-picked block size. The auto-picker is rarely a perfect choice (it does not know your access patterns), but it is a great sanity check — usually within 10% of optimal.
