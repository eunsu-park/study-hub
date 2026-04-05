# 18. Histogram and Binning

**Previous**: [Stencil Computations](./17_Stencil_Computations.md) | **Next**: [Sparse Matrix Ops](./19_Sparse_Matrix_Ops.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement a global-atomic histogram and understand its serialization bottleneck
2. Use shared memory privatization to reduce atomic contention by a factor of ~(block size / bins)
3. Optimize for 256-bin histograms using 8-bit accumulators and overflow handling
4. Compute 2D histograms for joint probability estimation
5. Use CUB `DeviceHistogram` and choose between atomic-based vs sort-based approaches

---

## 1. Why Histograms Are Challenging on GPU

A histogram tallies how many input values fall into each bin. The conceptual update `bins[bucket(x)]++` is a **read-modify-write** on an address determined by data — creating write conflicts when multiple threads map to the same bin.

```
Input: [2, 5, 2, 7, 2, 5, 0, 2]  (8 elements, 8 bins)
Hist:  [1, 0, 4, 0, 0, 2, 0, 1]

Problem: if threads 0,2,4,6 all try to increment bin[2] simultaneously,
three increments will be lost without atomics.
```

The challenge is achieving high parallelism while resolving write conflicts correctly.

---

## 2. Baseline: Global Atomic Histogram

```c
// Simplest correct implementation — one atomic per thread
__global__ void histogram_global_atomic(
    const int *data, int *hist, int n, int num_bins)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int bin = data[i];           // assumes data already in [0, num_bins)
        atomicAdd(&hist[bin], 1);    // global atomic
    }
}

// Host
void run_histogram_global(const int *d_data, int *d_hist, int n, int B) {
    cudaMemset(d_hist, 0, B * sizeof(int));
    const int BLOCK = 256;
    histogram_global_atomic<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(
        d_data, d_hist, n, B);
}
```

**Performance problem**: with B=256 bins and N=128M elements, each bin receives on average N/B = 500K increments. At peak, ~4096 threads simultaneously issue `atomicAdd` to 256 addresses — severe serialization in global L2 cache. Throughput: ~50M elements/second vs 10B/s memory bandwidth (200× slower than theoretical).

---

## 3. Shared Memory Privatization

Each block maintains a **private copy** of the histogram in shared memory. After processing the block's data, the private histogram is merged into the global histogram with a single atomic per bin per block:

```c
// Shared memory privatized histogram
__global__ void histogram_smem(
    const int *data, int *hist, int n, int num_bins)
{
    extern __shared__ int local_hist[];  // size = num_bins * sizeof(int)

    int tid = threadIdx.x;

    // Initialize local histogram to zero
    for (int b = tid; b < num_bins; b += blockDim.x)
        local_hist[b] = 0;
    __syncthreads();

    // Each thread processes multiple elements (grid-stride loop)
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += stride)
        atomicAdd(&local_hist[data[i]], 1);  // shared mem atomic (fast)
    __syncthreads();

    // Merge local histogram into global histogram
    for (int b = tid; b < num_bins; b += blockDim.x)
        atomicAdd(&hist[b], local_hist[b]);  // one global atomic per bin per block
}

void run_histogram_smem(const int *d_data, int *d_hist, int n, int B) {
    cudaMemset(d_hist, 0, B * sizeof(int));
    const int BLOCK = 256;
    // Limit grid to keep shared mem reuse high (each block processes many elements)
    int grid = min((n + BLOCK - 1) / BLOCK, 1024);
    histogram_smem<<<grid, BLOCK, B * sizeof(int)>>>(d_data, d_hist, n, B);
}
```

**Shared memory atomics** are roughly 10–30× faster than global memory atomics (they operate within the SM's L1 cache). Each shared atomic conflict resolves in ~4 cycles vs ~100+ cycles for global.

**Memory constraint**: a 256-bin histogram uses 256 × 4 = 1024 bytes of shared memory — trivial. A 4096-bin histogram uses 16 KB — still fits within 48 KB shared. Beyond ~8192 bins, shared memory privatization becomes impractical.

---

## 4. 256-Bin Optimization with 8-Bit Accumulators

For 256 bins, use `uint8_t` per bin (4× smaller) and flush to global when any bin reaches 255:

```c
// 8-bit privatized histogram with overflow check
__global__ void histogram_256_u8(
    const uint8_t *data, int *hist, int n)
{
    // 256 bins × 1 byte = 256 bytes of shared mem
    __shared__ uint8_t local8[256];
    int tid = threadIdx.x;

    // Zero the 256 bytes using 64-thread stores (4 bytes each)
    if (tid < 64) ((uint32_t*)local8)[tid] = 0;
    __syncthreads();

    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += stride) {
        int bin = data[i];
        // Increment; if overflow, flush bin to global
        if (++local8[bin] == 0) {          // wrapped around (was 255 → 0)
            atomicAdd(&hist[bin], 256);    // recover the 256 that wrapped
        }
    }
    __syncthreads();

    // Flush remaining counts
    if (tid < 256) atomicAdd(&hist[tid], local8[tid]);
}
```

This halves shared memory usage and can improve warp-level throughput when multiple threads hit the same bin (the 8-bit word is cheaper to atomically update than a 32-bit word in some GPU architectures).

---

## 5. 2D Histogram (Joint Distribution)

A 2D histogram counts pairs (x[i], y[i]) → bin (bx, by), useful for co-occurrence matrices, joint probability estimation, and color histogram descriptors:

```c
// 2D histogram: Bx bins × By bins grid
__global__ void histogram_2d(
    const float *x_data, const float *y_data, int *hist,
    int n, int Bx, int By,
    float x_min, float x_max, float y_min, float y_max)
{
    extern __shared__ int local_hist[];  // Bx * By ints

    int tid = threadIdx.x;
    int total_bins = Bx * By;

    for (int b = tid; b < total_bins; b += blockDim.x) local_hist[b] = 0;
    __syncthreads();

    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += stride) {
        float x = x_data[i], y = y_data[i];
        if (x < x_min || x >= x_max || y < y_min || y >= y_max) continue;

        int bx = (int)((x - x_min) / (x_max - x_min) * Bx);
        int by = (int)((y - y_min) / (y_max - y_min) * By);
        bx = min(bx, Bx - 1);
        by = min(by, By - 1);

        atomicAdd(&local_hist[by * Bx + bx], 1);
    }
    __syncthreads();

    for (int b = tid; b < total_bins; b += blockDim.x)
        atomicAdd(&hist[b], local_hist[b]);
}
```

For large 2D histograms (e.g., 1024×1024 = 4M bins), privatization no longer fits in shared memory. Switch to the **sort-based approach**: sort all (bx, by) pairs by their linearized bin index, then use run-length encoding to count.

---

## 6. Using CUB DeviceHistogram

CUB provides optimized histogram implementations for common use cases:

```c
#include <cub/cub.cuh>

// Single-channel histogram (e.g., grayscale image)
void cub_histogram_single(const uint8_t *d_samples, int *d_hist,
                           int n_samples) {
    const int NUM_BINS = 256;
    int lower = 0, upper = 256;  // sample range [lower, upper)

    void *d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceHistogram::HistogramEven(
        d_temp, temp_bytes,
        d_samples, d_hist, NUM_BINS + 1,  // +1: CUB counts bin edges
        lower, upper, n_samples);

    cudaMalloc(&d_temp, temp_bytes);
    cub::DeviceHistogram::HistogramEven(
        d_temp, temp_bytes,
        d_samples, d_hist, NUM_BINS + 1,
        lower, upper, n_samples);

    cudaFree(d_temp);
}

// Multi-channel histogram (e.g., RGB image — 3 channels, 256 bins each)
void cub_histogram_multi_channel(const uint8_t *d_image,   // interleaved RGB
                                  int *d_hist_r, int *d_hist_g, int *d_hist_b,
                                  int n_pixels) {
    const int NUM_CHANNELS = 3;
    const int NUM_ACTIVE   = 3;   // all channels active
    const int NUM_BINS     = 256;

    int* d_hists[3] = {d_hist_r, d_hist_g, d_hist_b};
    int  levels[3]  = {NUM_BINS + 1, NUM_BINS + 1, NUM_BINS + 1};
    int  lower[3]   = {0, 0, 0};
    int  upper[3]   = {256, 256, 256};

    void *d_temp = nullptr; size_t temp_bytes = 0;
    cub::DeviceHistogram::MultiHistogramEven<NUM_CHANNELS, NUM_ACTIVE>(
        d_temp, temp_bytes,
        d_image, d_hists, levels, lower, upper, n_pixels);

    cudaMalloc(&d_temp, temp_bytes);
    cub::DeviceHistogram::MultiHistogramEven<NUM_CHANNELS, NUM_ACTIVE>(
        d_temp, temp_bytes,
        d_image, d_hists, levels, lower, upper, n_pixels);

    cudaFree(d_temp);
}
```

---

## 7. Atomic vs Sort-Based Histogram

```
Approach          When to Use                         Throughput
-----------------------------------------------------------------------
Global atomic     Quick prototype, B is large,         ~50M elem/s
                  irregular distribution               (contention limited)

Shared atomic     B ≤ 8192, uniform/moderate dist,    ~500M elem/s
(privatized)      best general-purpose choice

CUB Histogram     B is power-of-2, 256–4096 bins,     ~2B elem/s
                  production code

Sort-based        Very large B (>16K bins), need       ~1B elem/s
                  exact bucket positions,              (sort dominated)
                  subsequent per-bucket processing
```

**Sort-based pipeline**: sort keys → count consecutive equal keys (run-length encoding). This naturally produces the output positions alongside counts, enabling subsequent per-bucket work.

---

## Key Takeaways

- **Global atomics** are correct but slow due to L2 contention — avoid for production histograms
- **Shared memory privatization** reduces global atomic traffic from N to `grid × B` atomics, typically 1000× fewer global atomics
- 256-bin histogram fits comfortably in shared memory (1 KB); >8192 bins exceeds typical shared memory budget
- **CUB DeviceHistogram** handles edge cases (out-of-range samples, non-power-of-2 bins, multi-channel) and achieves near-bandwidth-limited throughput
- **2D histograms** with large B×B grids should use the sort-based approach rather than shared memory atomics
- Choose atomic-based for small B, sort-based for large B or when per-bucket downstream work is needed

---

**Next**: [19. Sparse Matrix Ops](./19_Sparse_Matrix_Ops.md) — Represent sparse matrices in COO, CSR, and CSC formats and implement efficient SpMV with cuSPARSE.
