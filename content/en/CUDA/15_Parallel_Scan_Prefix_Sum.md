# 15. Parallel Scan / Prefix Sum

**Previous**: [Parallel Reduction](./14_Parallel_Reduction.md) | **Next**: [Parallel Sort](./16_Parallel_Sort.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Distinguish inclusive from exclusive prefix sums and derive one from the other
2. Implement Hillis-Steele inclusive scan (O(n log n) work, O(log n) depth)
3. Implement Blelloch work-efficient exclusive scan (O(n) work, O(log n) depth)
4. Apply scan to stream compaction (filtering non-zero elements)
5. Use CUB `DeviceScan` for production-quality scan operations

---

## 1. What Is a Prefix Sum?

**Inclusive scan**: output[i] = input[0] + input[1] + ... + input[i]
**Exclusive scan**: output[i] = input[0] + input[1] + ... + input[i-1]  (output[0] = 0)

```
Input:            [3,  1,  4,  1,  5,  9,  2,  6]
Inclusive scan:   [3,  4,  8,  9, 14, 23, 25, 31]
Exclusive scan:   [0,  3,  4,  8,  9, 14, 23, 25]
```

**Relationship**: exclusive[i] = inclusive[i-1] (with exclusive[0] = identity).

Prefix sum is the foundation of:
- Stream compaction (select elements satisfying a predicate)
- Radix sort (offset computation per digit bucket)
- Segmented operations (variable-length parallel workloads)
- Load balancing (distribute unequal work chunks)

---

## 2. Hillis-Steele Inclusive Scan (O(n log n) Work)

The Hillis-Steele algorithm achieves minimum depth (O(log n) steps) at the cost of extra work (O(n log n) total additions):

```
Step 1 (stride=1): [3, 1+3, 4+1, 1+4, 5+1, 9+5, 2+9, 6+2]
                 = [3,   4,   5,   5,   6,  14,  11,   8]
Step 2 (stride=2): [3, 4, 5+3, 5+4, 6+5, 14+5, 11+6, 8+14]
                 = [3, 4,   8,   9,  11,   19,   17,   22]
Step 3 (stride=4): [3, 4, 8, 9, 11+3, 19+4, 17+8, 22+9]
                 = [3, 4, 8, 9,   14,   23,   25,   31]  ✓
```

```c
// Hillis-Steele inclusive scan (single block, n <= blockDim.x)
__global__ void scan_hillis_steele(const float *g_in, float *g_out, int n) {
    extern __shared__ float temp[];  // double-buffered shared mem

    int tid = threadIdx.x;
    int pout = 0, pin = 1;          // ping-pong buffer indices

    // Load input
    temp[pout * n + tid] = (tid < n) ? g_in[tid] : 0.0f;
    __syncthreads();

    for (int stride = 1; stride < n; stride <<= 1) {
        pout = 1 - pout;  // swap buffers
        pin  = 1 - pout;

        if (tid >= stride)
            temp[pout * n + tid] = temp[pin * n + tid] + temp[pin * n + tid - stride];
        else
            temp[pout * n + tid] = temp[pin * n + tid];

        __syncthreads();
    }

    if (tid < n) g_out[tid] = temp[pout * n + tid];
}
```

**When to use Hillis-Steele**: when depth (latency) matters more than total work — e.g., inside a single warp where all 32 lanes execute in lockstep.

---

## 3. Blelloch Work-Efficient Exclusive Scan (O(n) Work)

Blelloch scan performs only O(n) total additions, matching sequential complexity, at depth O(log n). It uses two phases:

**Phase 1 — Up-sweep (reduce)**: build a reduction tree from leaves to root.
**Phase 2 — Down-sweep**: traverse from root to leaves, distributing partial sums.

```c
// Blelloch exclusive scan — single block, n must be power of 2
__global__ void scan_blelloch(float *g_data, float *g_out, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    // Load into shared memory
    sdata[tid]     = (2 * tid     < n) ? g_data[2 * tid]     : 0.0f;
    sdata[tid + 1] = (2 * tid + 1 < n) ? g_data[2 * tid + 1] : 0.0f;
    // (Each thread handles 2 elements; launch n/2 threads)

    // Phase 1: Up-sweep (reduce)
    int offset = 1;
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            sdata[bi] += sdata[ai];
        }
        offset <<= 1;
    }

    // Set root to identity (0 for sum)
    if (tid == 0) sdata[n - 1] = 0.0f;

    // Phase 2: Down-sweep
    for (int d = 1; d < n; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            float tmp   = sdata[ai];
            sdata[ai]   = sdata[bi];
            sdata[bi]  += tmp;
        }
    }
    __syncthreads();

    // Write output
    if (2 * tid     < n) g_out[2 * tid]     = sdata[tid];
    if (2 * tid + 1 < n) g_out[2 * tid + 1] = sdata[tid + 1];
}
```

**Complexity comparison:**

```
Algorithm        Work          Depth (steps)   Extra memory
--------------------------------------------------------------
Sequential       O(n)          O(n)            O(1)
Hillis-Steele    O(n log n)    O(log n)        O(n) double buffer
Blelloch         O(n)          O(log n)        O(n) shared mem
```

Blelloch is preferred when n is large enough that work efficiency matters.

---

## 4. Warp-Level Inclusive Scan with Shuffle

For n ≤ 32, use shuffle for the lowest-latency scan possible:

```c
// Inclusive warp scan using shuffle up
__device__ float warp_scan_inclusive(float val) {
    for (int offset = 1; offset < 32; offset <<= 1) {
        float y = __shfl_up_sync(0xffffffff, val, offset);
        if ((threadIdx.x & 31) >= offset) val += y;
    }
    return val;
}

// Block-level inclusive scan: warp scan -> combine warp sums -> add prefix
__global__ void scan_block_shuffle(const float *g_in, float *g_out, int n) {
    extern __shared__ float warp_sums[];  // one float per warp

    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    float val = (tid < n) ? g_in[tid] : 0.0f;

    // Step 1: inclusive scan within each warp
    val = warp_scan_inclusive(val);

    // Step 2: store each warp's total (lane 31's value)
    if (lane == 31) warp_sums[wid] = val;
    __syncthreads();

    // Step 3: scan the warp totals (only first warp does this)
    if (wid == 0) {
        float ws = (lane < (blockDim.x / 32)) ? warp_sums[lane] : 0.0f;
        ws = warp_scan_inclusive(ws);
        warp_sums[lane] = ws;
    }
    __syncthreads();

    // Step 4: add the prefix from earlier warps
    float prefix = (wid > 0) ? warp_sums[wid - 1] : 0.0f;
    val += prefix;

    if (tid < n) g_out[tid] = val;
}
```

---

## 5. Multi-Block Scan (Large Arrays)

Scanning arrays larger than one block requires communicating partial sums across blocks. The standard approach is a **three-kernel** strategy:

```c
// Kernel 1: scan each block independently, write block totals
__global__ void scan_blocks(const float *in, float *out, float *block_sums, int n);

// Kernel 2: scan the block_sums array (small — one element per block)
__global__ void scan_block_sums(float *block_sums, int num_blocks);

// Kernel 3: add the scanned block prefix to each element
__global__ void add_block_prefix(float *out, const float *block_sums, int n);

// Host orchestration
void scan_large(const float *d_in, float *d_out, int n) {
    const int BLOCK = 256;
    int num_blocks = (n + BLOCK - 1) / BLOCK;

    float *d_block_sums;
    cudaMalloc(&d_block_sums, num_blocks * sizeof(float));

    scan_blocks<<<num_blocks, BLOCK, BLOCK * sizeof(float)>>>(
        d_in, d_out, d_block_sums, n);

    // Recursively scan block sums (small array, single block)
    scan_block_sums<<<1, num_blocks, num_blocks * sizeof(float)>>>(
        d_block_sums, num_blocks);

    add_block_prefix<<<num_blocks, BLOCK>>>(d_out, d_block_sums, n);

    cudaFree(d_block_sums);
}
```

Modern alternative: **look-back scan** (used by CUB) — blocks proceed without a global synchronization barrier by using atomic flags to check when the previous block's prefix is available.

---

## 6. Stream Compaction with Scan

Stream compaction selects elements satisfying a predicate and packs them into a contiguous output array — a core building block of GPU renderers, collision detection, and graph BFS:

```c
// Example: compact non-zero elements from d_in into d_out
// Returns the number of selected elements.

__global__ void mark_nonzero(const float *d_in, int *d_flags, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d_flags[i] = (d_in[i] != 0.0f) ? 1 : 0;
}

__global__ void scatter(const float *d_in, const int *d_flags,
                        const int *d_scan,  // exclusive scan of d_flags
                        float *d_out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && d_flags[i]) {
        d_out[d_scan[i]] = d_in[i];  // scatter to scanned position
    }
}

int stream_compact(const float *d_in, float *d_out, int n) {
    const int BLOCK = 256;
    int *d_flags, *d_scan;
    cudaMalloc(&d_flags, n * sizeof(int));
    cudaMalloc(&d_scan,  n * sizeof(int));

    // Step 1: mark
    mark_nonzero<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(d_in, d_flags, n);

    // Step 2: exclusive scan of flags to get output positions
    // (use CUB DeviceScan::ExclusiveSum in practice)
    exclusive_scan(d_flags, d_scan, n);

    // Step 3: scatter selected elements
    scatter<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(d_in, d_flags, d_scan, d_out, n);

    // Number of output elements = scan[n-1] + flags[n-1]
    int last_flag, last_scan;
    cudaMemcpy(&last_flag, d_flags + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_scan, d_scan  + n - 1, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_flags); cudaFree(d_scan);
    return last_flag + last_scan;
}
```

---

## 7. Using CUB DeviceScan

```c
#include <cub/cub.cuh>

void cub_exclusive_scan(const float *d_in, float *d_out, int n) {
    void   *d_temp = nullptr;
    size_t  temp_bytes = 0;

    // Query temp storage size
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_in, d_out, n);

    // Allocate
    cudaMalloc(&d_temp, temp_bytes);

    // Run
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_in, d_out, n);
    cudaDeviceSynchronize();

    cudaFree(d_temp);
}

// Inclusive scan
void cub_inclusive_scan(const int *d_in, int *d_out, int n) {
    void *d_temp = nullptr; size_t temp_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp, temp_bytes, d_in, d_out, n);
    cudaMalloc(&d_temp, temp_bytes);
    cub::DeviceScan::InclusiveSum(d_temp, temp_bytes, d_in, d_out, n);
    cudaFree(d_temp);
}

// Segmented scan: independent scan within segments defined by flags
// cub::DeviceScan::ExclusiveSumByKey(d_keys, d_vals, d_out, n)
```

CUB's DeviceScan uses the **decoupled look-back** algorithm: blocks proceed as soon as their predecessor's prefix is available via atomic flags, rather than waiting for a global synchronization barrier. This achieves near-peak memory bandwidth.

---

## Key Takeaways

- **Inclusive scan**: output[i] includes input[i]; **exclusive scan**: output[i] excludes input[i] (output[0] = 0)
- Hillis-Steele uses O(n log n) work for O(log n) depth — good for small warp-level scans
- **Blelloch** achieves O(n) work at O(log n) depth via up-sweep/down-sweep — preferred for large n
- Warp shuffle (`__shfl_up_sync`) gives the fastest warp-level scan with no shared memory
- **Stream compaction** = mark + exclusive scan + scatter — 3-step pipeline enabling data-dependent output sizes
- **CUB `DeviceScan`** uses decoupled look-back for near-peak bandwidth; use it in production

---

**Next**: [16. Parallel Sort](./16_Parallel_Sort.md) — Implement bitonic sort, radix sort, and thrust::sort; understand when to use each algorithm on the GPU.
