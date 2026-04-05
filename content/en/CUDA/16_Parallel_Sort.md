# 16. Parallel Sort

**Previous**: [Parallel Scan Prefix Sum](./15_Parallel_Scan_Prefix_Sum.md) | **Next**: [Stencil Computations](./17_Stencil_Computations.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement bitonic sort for power-of-2 input sizes on the GPU
2. Explain how radix sort uses scan to achieve O(n) sorting passes
3. Use CUB `DeviceRadixSort` and `thrust::sort` for production sorting
4. Choose the right sorting algorithm given data type, size, and access pattern
5. Measure sorting throughput in elements per second and compare to theoretical limits

---

## 1. Why Sorting Is Hard on GPUs

Sorting is inherently **data dependent** — the sequence of memory accesses and comparisons depends on the data values. This conflicts with the GPU's SIMD execution model where all threads in a warp execute the same instruction.

Good GPU sort algorithms exploit two properties:
1. **Oblivious comparator networks** (bitonic, odd-even merge): the comparison sequence is fixed regardless of data values — no divergence
2. **Digit decomposition** (radix sort): reduce sorting to a sequence of counting + prefix-sum passes, each of which is trivially parallel

---

## 2. Bitonic Sort

Bitonic sort is a **sorting network** — a fixed sequence of compare-and-swap operations that correctly sorts any input. Because the sequence is data-independent, every thread follows the same instruction path (no divergence).

**Bitonic sequence**: a sequence that first increases then decreases (or vice versa).

```
N=8 example: compare-and-swap pairs at each step
Pass 1 (k=2): [0↔1, 2↔3, 4↔5, 6↔7]  (form 4 bitonic pairs)
Pass 2 (k=4): [0↔3, 1↔2] then [4↔7, 5↔6]
Pass 3 (k=8): [0↔7, 1↔6, 2↔5, 3↔4] (merge into sorted sequence)
```

```c
// Bitonic sort — sorts array of N elements (N must be power of 2)
__device__ void compare_and_swap(float *a, float *b, bool ascending) {
    if (ascending ? (*a > *b) : (*a < *b)) {
        float tmp = *a; *a = *b; *b = tmp;
    }
}

__global__ void bitonic_sort_step(float *data, int j, int k) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ij = i ^ j;

    if (ij > i) {
        // Determine sort direction based on which bitonic sequence we're merging
        bool ascending = ((i & k) == 0);
        compare_and_swap(&data[i], &data[ij], ascending);
    }
}

// Host: launch log2(N)*(log2(N)+1)/2 kernel passes
void bitonic_sort(float *d_data, int n) {
    // n must be power of 2
    const int BLOCK = 256;
    int grid = n / BLOCK;

    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonic_sort_step<<<grid, BLOCK>>>(d_data, j, k);
            cudaDeviceSynchronize();
        }
    }
}
```

**Shared memory optimization** for small subarrays (fits in a block):

```c
// Sort within a block using shared memory — avoids global memory round trips
// for the inner passes where stride < blockDim.x
__global__ void bitonic_sort_shared(float *data, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (gid < n) ? data[gid] : FLT_MAX;
    __syncthreads();

    int bsize = blockDim.x;
    for (int k = 2; k <= bsize; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ij = tid ^ j;
            if (ij > tid) {
                bool asc = ((tid & k) == 0);
                if (asc ? (sdata[tid] > sdata[ij]) : (sdata[tid] < sdata[ij])) {
                    float tmp = sdata[tid]; sdata[tid] = sdata[ij]; sdata[ij] = tmp;
                }
            }
            __syncthreads();
        }
    }

    if (gid < n) data[gid] = sdata[tid];
}
```

**Complexity**: O(n log²n) comparisons, O(log²n) parallel steps. Practical for n ≤ 1M on GPU.

---

## 3. Radix Sort

Radix sort processes b bits at a time (typically 4-bit passes = 16 buckets). Each pass:
1. **Count** how many elements fall in each bucket
2. **Exclusive scan** the counts to get bucket offsets
3. **Scatter** elements to their new positions

This is O(n * 32/b) total work — for 4-bit passes on 32-bit integers: 8 passes.

```c
// 1-bit radix sort pass (for clarity; production uses 4-bit)
__global__ void radix_1bit_pass(const uint32_t *in, uint32_t *out,
                                int *zeros_count, int n, int bit) {
    // Phase 1: determine which elements have a 0 or 1 at 'bit'
    extern __shared__ int sdata[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int val = (tid < n) ? ((in[tid] >> bit) & 1) : 1;

    sdata[threadIdx.x] = (val == 0) ? 1 : 0;  // flags for 0-bucket
    __syncthreads();

    // Blelloch scan within block to get local positions
    // (simplified — production CUB handles this correctly)
    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        int x = (threadIdx.x >= stride) ? sdata[threadIdx.x - stride] : 0;
        __syncthreads();
        sdata[threadIdx.x] += x;
        __syncthreads();
    }

    // Block-level scatter (actual multi-block coordination omitted for brevity)
    if (tid < n) {
        // position computed from block scan + inter-block prefix
        // out[position] = in[tid];
    }

    // Report number of zeros in this block (for inter-block scan)
    if (threadIdx.x == blockDim.x - 1)
        atomicAdd(zeros_count, sdata[threadIdx.x]);
}
```

For production, implement a full 4-bit radix sort using CUB's primitives:

```c
#include <cub/cub.cuh>

void radix_sort_example(uint32_t *d_keys, int n) {
    // CUB DeviceRadixSort — highly optimized 4-bit radix sort
    cub::DoubleBuffer<uint32_t> d_keys_buf(d_keys, d_tmp_keys);

    void *d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_keys_buf, n);
    cudaMalloc(&d_temp, temp_bytes);
    cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_keys_buf, n);

    // If result is in alternate buffer, copy back
    if (d_keys_buf.Current() != d_keys)
        cudaMemcpy(d_keys, d_keys_buf.Current(), n * sizeof(uint32_t),
                   cudaMemcpyDeviceToDevice);

    cudaFree(d_temp);
}

// Sort key-value pairs (e.g., float keys with int indices)
void radix_sort_pairs(float *d_keys, int *d_vals, int n) {
    cub::DoubleBuffer<float> d_keys_buf(d_keys, d_tmp_keys);
    cub::DoubleBuffer<int>   d_vals_buf(d_vals, d_tmp_vals);

    void *d_temp = nullptr; size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes, d_keys_buf, d_vals_buf, n);
    cudaMalloc(&d_temp, temp_bytes);
    cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes, d_keys_buf, d_vals_buf, n);
    cudaFree(d_temp);
}
```

---

## 4. Merge Sort on GPU

GPU merge sort works in two phases:
1. **Local sort**: each block sorts its subarray (using bitonic or bitonic+shared)
2. **Global merge**: repeatedly merge sorted runs (stride doubling)

```c
// Merge two sorted halves using two pointers (sequential per thread)
__global__ void merge_step(const float *in, float *out, int width, int n) {
    // Each thread block handles one merged segment
    int seg_start = blockIdx.x * (2 * width);
    int mid       = min(seg_start + width, n);
    int seg_end   = min(seg_start + 2 * width, n);

    // Simple serial merge within block (improved version uses parallel merge)
    int l = seg_start, r = mid, out_i = seg_start;
    while (l < mid && r < seg_end) {
        if (in[l] <= in[r]) out[out_i++] = in[l++];
        else                 out[out_i++] = in[r++];
    }
    while (l < mid)     out[out_i++] = in[l++];
    while (r < seg_end) out[out_i++] = in[r++];
}

void merge_sort(float *d_data, int n) {
    float *d_tmp;
    cudaMalloc(&d_tmp, n * sizeof(float));

    // Phase 1: local sort each block of BLOCK elements
    const int BLOCK = 1024;
    bitonic_sort_shared<<<(n + BLOCK - 1) / BLOCK, BLOCK,
                          BLOCK * sizeof(float)>>>(d_data, n);

    // Phase 2: merge passes
    float *src = d_data, *dst = d_tmp;
    for (int width = BLOCK; width < n; width <<= 1) {
        int num_segs = (n + 2 * width - 1) / (2 * width);
        merge_step<<<num_segs, 1>>>(src, dst, width, n);
        cudaDeviceSynchronize();
        float *swap = src; src = dst; dst = swap;
    }

    if (src != d_data) cudaMemcpy(d_data, src, n * sizeof(float),
                                  cudaMemcpyDeviceToDevice);
    cudaFree(d_tmp);
}
```

GPU merge sort is useful when comparison functions are expensive (e.g., sorting strings or composite keys), since merge sort needs only O(n log n) comparisons vs radix sort's fixed digit-decomposition passes.

---

## 5. Thrust Sort

Thrust provides the highest-level interface — one line of code:

```c
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

void thrust_sort_example() {
    // Convenience: thrust::device_vector manages GPU memory
    thrust::device_vector<float> d_vec = {3.f, 1.f, 4.f, 1.f, 5.f, 9.f};
    thrust::sort(d_vec.begin(), d_vec.end());       // ascending
    thrust::sort(d_vec.begin(), d_vec.end(),
                 thrust::greater<float>());          // descending

    // Sort with custom comparator
    thrust::sort(thrust::device, d_vec.begin(), d_vec.end(),
                 [] __device__ (float a, float b) { return fabsf(a) < fabsf(b); });

    // Sort key-value pairs (argsort equivalent)
    thrust::device_vector<int> d_idx(d_vec.size());
    thrust::sequence(d_idx.begin(), d_idx.end());   // 0, 1, 2, ...
    thrust::sort_by_key(d_vec.begin(), d_vec.end(), d_idx.begin());
    // d_idx now holds original indices in sorted order

    // Interop with raw CUDA pointers
    float *d_raw;  int n = 1 << 20;
    cudaMalloc(&d_raw, n * sizeof(float));
    thrust::sort(thrust::device, d_raw, d_raw + n);
    cudaFree(d_raw);
}
```

Thrust's `sort` internally uses radix sort for primitive types (int, float, double) and merge sort for complex comparators.

---

## 6. Algorithm Comparison and Selection Guide

```
Algorithm         Complexity       Stable?  Best Use Case
-----------------------------------------------------------------
Bitonic sort      O(n log²n)       No       n < 1M, fixed HW,
                                            needs oblivious network
Radix sort (CUB)  O(n * k/b)       Yes      Integers, floats,
                  (k=key bits,              large n (>1M)
                   b=bits/pass)
Merge sort        O(n log n)       Yes      Complex comparators,
                                            linked structures
Thrust::sort      O(n log n)       Depends  Fastest to write;
                  (auto-selects)            uses radix for POD types

Performance (N=128M int32, RTX 3090, single pass end-to-end):
  CUB DeviceRadixSort:  ~4 GB/s throughput
  Thrust::sort (int):   ~3.8 GB/s (uses CUB internally)
  Bitonic:              ~1.2 GB/s (O(log²n) overhead)
```

**Selection rules:**
- **Integers or floats, large n**: use `cub::DeviceRadixSort` or `thrust::sort`
- **Need stable sort**: use `cub::DeviceRadixSort` (stable) or `thrust::stable_sort`
- **Complex custom comparator**: use `thrust::sort` with a device lambda
- **Small n (< 2048) or single block**: bitonic sort in shared memory
- **Variable-length keys or strings**: merge sort with per-element comparison kernel

---

## Key Takeaways

- **Bitonic sort** is data-oblivious (no divergence) but O(n log²n) — best for small n or oblivious network requirements
- **Radix sort** is O(n) per pass (8 passes for 32-bit keys) — fastest for large arrays of integers or floats
- Each radix pass is: count per bucket → exclusive scan → scatter; scan is the bottleneck
- **Thrust::sort** automatically selects radix sort for primitive types; single line of code is correct and fast
- **CUB DeviceRadixSort** offers the finest control (key-value pairs, partial bit range, in-place double buffer)
- Sort performance is memory-bandwidth bound; the best GPU sorts achieve ~4 elements per byte of memory bandwidth

---

**Next**: [17. Stencil Computations](./17_Stencil_Computations.md) — Implement 1D/2D/3D stencil kernels with shared memory tiling, halo cells, and time-stepping loops for the heat equation.
