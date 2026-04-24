# Lesson 7 — Atomic Operations (per-lesson exercise)

Prerequisites: L04 (memory model), L06 (warp execution).

Compile: `nvcc -O3 -arch=sm_80 ex.cu -o ex`

Atomics serialize updates to a memory location across threads. They are essential for any reduction-like pattern where multiple threads contribute to the same output, but they are slow when contention is high — every thread writing to the same address gets serialized 32-way at the warp level.

---

## Exercise 7.1 — Histogram with Global Atomics

**Difficulty**: ★★

### Problem

Compute a 256-bin histogram from `N = 16 * 1024 * 1024` 8-bit values. The naive kernel uses one global atomic per thread:

```cuda
__global__ void histogram_global_atomic(const uint8_t *input, int *bins, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) atomicAdd(&bins[input[idx]], 1);
}
```

Time this. With uniformly distributed input, every bin sees roughly 64k contending threads — modest serialization, but real.

---

## Exercise 7.2 — Shared-Memory Atomics

**Difficulty**: ★★★

### Problem

Each block keeps its own private histogram in shared memory; only the per-block totals are atomically added to global memory at the end:

```cuda
__global__ void histogram_shared(const uint8_t *input, int *bins, int N) {
    __shared__ int local[256];
    int tid = threadIdx.x;

    if (tid < 256) local[tid] = 0;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) atomicAdd(&local[input[idx]], 1);
    __syncthreads();

    if (tid < 256) atomicAdd(&bins[tid], local[tid]);
}
```

Shared-memory atomics are roughly 10× faster than global atomics on most GPUs, AND they avoid the all-blocks-contending-for-one-address bottleneck.

Time this against 7.1 — expect a 5–20× speedup on uniformly distributed input. The factor is even larger when the input is skewed (one very popular bin).

---

## Exercise 7.3 — Lock-Free Counter Pattern — Bonus

**Difficulty**: ★★★

Implement a parallel job-claim queue: a single global counter `next_job` that threads atomically increment to claim work. Each thread runs:

```cuda
__device__ int next_job;     // global counter

__global__ void worker(...) {
    while (true) {
        int job = atomicAdd(&next_job, 1);
        if (job >= total_jobs) return;
        process(job);
    }
}
```

This is the standard "work-stealing without explicit queues" pattern. Compare it against pre-partitioned work (each block gets a fixed range). Pre-partitioning wins when work-per-job is uniform; the atomic counter wins when work-per-job varies wildly (e.g., ray tracing) because it provides automatic load balancing.
