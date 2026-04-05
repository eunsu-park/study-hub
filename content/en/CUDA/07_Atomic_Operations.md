# 07. Atomic Operations

**Previous**: [Warp Execution and Divergence](./06_Warp_Execution_and_Divergence.md) | **Next**: [Memory Coalescing](./08_Memory_Coalescing.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why regular read-modify-write is unsafe in parallel code
2. Use `atomicAdd`, `atomicCAS`, `atomicExch`, and related intrinsics
3. Implement a correct parallel histogram using atomic operations
4. Measure and understand atomic throughput vs contention cost
5. Apply privatization to dramatically reduce atomic contention

---

## 1. The Race Condition Problem

Without atomics, concurrent writes to the same address produce wrong results:

```c
// Naive parallel counter — WRONG
__global__ void count_positives(const float *data, int *count, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && data[i] > 0.0f) {
        (*count)++;  // Race condition! Read-modify-write is not atomic
    }
}

// What happens:
// Thread 0 reads count = 5
// Thread 1 reads count = 5  (before thread 0 writes)
// Thread 0 writes count = 6
// Thread 1 writes count = 6  (overwrites thread 0's result!)
// Lost update: two positive values found, but count only increased by 1
```

`atomicAdd` solves this:

```c
__global__ void count_positives_atomic(const float *data, int *count, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && data[i] > 0.0f) {
        atomicAdd(count, 1);  // atomic read-modify-write — always correct
    }
}
```

---

## 2. Atomic Operations Reference

All atomics return the **old value** of the target before the operation:

```c
// Integer atomics
int old = atomicAdd(int *addr, int val);      // *addr += val
int old = atomicSub(int *addr, int val);      // *addr -= val
int old = atomicMax(int *addr, int val);      // *addr = max(*addr, val)
int old = atomicMin(int *addr, int val);      // *addr = min(*addr, val)
int old = atomicAnd(int *addr, int val);      // *addr &= val
int old = atomicOr (int *addr, int val);      // *addr |= val
int old = atomicXor(int *addr, int val);      // *addr ^= val
int old = atomicExch(int *addr, int val);     // *addr = val (exchange)
int old = atomicInc(unsigned *addr, unsigned wrap);  // *addr = ((*addr >= wrap) ? 0 : *addr + 1)

// Float atomics (CUDA 2.0+)
float old = atomicAdd(float *addr, float val);  // *addr += val (FP32)
// Also available: atomicAdd for double, half (Volta+)

// Compare-and-swap: the foundation of lock-free programming
int old = atomicCAS(int *addr, int compare, int val);
// Semantics: if (*addr == compare) { *addr = val; } return old *addr;
```

---

## 3. Compare-and-Swap (CAS): The Universal Primitive

`atomicCAS` can implement any atomic operation:

```c
// Atomic floating-point max (not natively available before SM 8.0)
__device__ void atomicMaxFloat(float *addr, float val) {
    int *addr_as_int = (int *)addr;
    int old = *addr_as_int;
    int expected;
    do {
        expected = old;
        float current = __int_as_float(old);
        if (val <= current) return;  // no update needed
        int new_val = __float_as_int(val);
        old = atomicCAS(addr_as_int, expected, new_val);
    } while (old != expected);  // retry if another thread changed addr
}
```

The CAS loop pattern:
1. Read the current value
2. Compute the desired new value
3. CAS: only update if nobody else changed it between step 1 and 3
4. If CAS fails (somebody changed it), retry with the latest value

This is the basis for all lock-free data structures.

---

## 4. Histogram Kernel

Histograms are a classic atomic use case: count occurrences of values in bins.

### Version 1: Naive global atomics

```c
__global__ void histogram_naive(const unsigned char *data, int *hist,
                                int n, int nbins) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int bin = (int)(data[i] * nbins / 256);
        atomicAdd(&hist[bin], 1);  // all threads contend on ~256 bins
    }
}
```

**Problem**: High contention — many threads write to the same bin simultaneously, serializing the atomics.

---

### Version 2: Privatized histogram (shared memory + atomic merge)

Each block builds its own private histogram in shared memory, then merges into global:

```c
__global__ void histogram_privatized(const unsigned char *data, int *hist,
                                     int n, int nbins) {
    extern __shared__ int local_hist[];  // nbins ints, dynamically allocated

    // Initialize shared histogram
    for (int i = threadIdx.x; i < nbins; i += blockDim.x)
        local_hist[i] = 0;
    __syncthreads();

    // Accumulate into shared memory (low contention — block-local)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int bin = (int)(data[i] * nbins / 256);
        atomicAdd(&local_hist[bin], 1);  // much lower contention
    }
    __syncthreads();

    // Merge local histogram into global (one merge per block per bin)
    for (int i = threadIdx.x; i < nbins; i += blockDim.x)
        atomicAdd(&hist[i], local_hist[i]);
}

// Launch:
int sharedBytes = nbins * sizeof(int);
histogram_privatized<<<grid, block, sharedBytes>>>(d_data, d_hist, n, nbins);
```

**Contention reduction**: instead of N total atomic operations on global memory, we have (N block-local atomics) + (gridSize × nbins global merges). For large N, this is drastically fewer global atomic conflicts.

---

## 5. Throughput vs Contention

**No contention** (all threads write to different addresses):

```c
// Each thread writes to its own location — no contention
atomicAdd(&hist[threadIdx.x], 1);  // 256 distinct addresses

// Throughput: ~1 atomic/cycle per SM → peak throughput
```

**High contention** (all threads write to the same address):

```c
// All 1024 threads in a block write to hist[0]
atomicAdd(&hist[0], 1);

// The SM serializes all 1024 writes → 1024 cycles for this operation
// Throughput: ~1024× slower than zero-contention case
```

**Benchmark**:

```c
// Measure: how does atomic throughput scale with contention?
// Setup: N=10M elements, vary number of distinct bins

// 1 bin   (max contention): ~ 45,000 μs
// 4 bins:                   ~ 11,500 μs
// 64 bins:                  ~  1,200 μs
// 1024 bins (low):          ~    320 μs
// No atomics:               ~     80 μs

// Lesson: even 64 bins with privatization beats 1024 bins without
```

---

## 6. Warp-Aggregated Atomics

Within a warp, aggregate atomicAdd before hitting global memory:

```c
__device__ void warp_aggregated_add(int *addr, int val) {
    unsigned mask = __activemask();                    // which threads are active
    int leader    = __ffs(mask) - 1;                   // lowest active lane

    // Count how many threads want to add the same value
    // (use for histogram bins where multiple threads map to same bin)
    unsigned match = __match_any_sync(mask, (unsigned long long)addr);
    int count      = __popc(match);                    // threads hitting same bin

    // Only the leader does the actual atomic
    if ((mask & match) == match) {  // I am the leader of this group
        int group_val = val * count;
        atomicAdd(addr, group_val);
    }
}
```

This reduces atomic operations by up to 32× when many threads hit the same bin — crucial for non-uniform distributions.

---

## 7. Atomic Scope (CUDA 9+)

Atomics can be scoped to the SM, GPU, or system (NVLink/PCIE):

```c
#include <cuda/atomic>  // C++ atomics header (libcu++)

// Block-scoped atomic (fastest — only visible within the block)
cuda::atomic<int, cuda::thread_scope_block> block_counter;

// Device-scoped (default — visible to all threads on this GPU)
cuda::atomic<int, cuda::thread_scope_device> global_counter;

// System-scoped (visible to CPU and all GPUs — slowest)
cuda::atomic<int, cuda::thread_scope_system> system_counter;
```

Use the narrowest scope that is semantically correct.

---

## 8. Complete Example: Parallel Word Frequency Counter

```c
#define ALPHA_SIZE 26

__global__ void letter_frequency(const char *text, int *freq, int n) {
    __shared__ int local_freq[ALPHA_SIZE];

    // Initialize
    if (threadIdx.x < ALPHA_SIZE) local_freq[threadIdx.x] = 0;
    __syncthreads();

    // Count (stride loop for arbitrary N)
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
             i < n;
             i += gridDim.x * blockDim.x) {
        char c = text[i];
        if (c >= 'a' && c <= 'z')
            atomicAdd(&local_freq[c - 'a'], 1);
        else if (c >= 'A' && c <= 'Z')
            atomicAdd(&local_freq[c - 'A'], 1);
    }
    __syncthreads();

    // Merge
    if (threadIdx.x < ALPHA_SIZE)
        atomicAdd(&freq[threadIdx.x], local_freq[threadIdx.x]);
}
```

---

## Key Takeaways

- `atomicAdd` / `atomicCAS` guarantee correct read-modify-write in parallel without explicit locking
- **Contention is the enemy**: many threads writing to the same address serializes — use privatization
- **Privatized histogram**: each block builds a local histogram in shared memory, then merges; reduces global atomic operations from N to N_blocks × N_bins
- `atomicCAS` is the universal building block — can implement any custom atomic operation
- Warp-aggregated atomics can reduce contention 32× by combining same-bin writes before hitting global memory

---

**Next**: [08. Memory Coalescing](./08_Memory_Coalescing.md) — Learn how 128-byte transaction granularity makes strided access expensive, compare AoS vs SoA layouts, and measure stride penalties with Nsight Compute.
