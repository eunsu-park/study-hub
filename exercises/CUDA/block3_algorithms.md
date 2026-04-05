# Block 3 — Algorithms

**Lessons covered**: L14 (Reduction), L15 (Scan / Prefix Sum), L16 (Sort),
L17 (Stencil Computations), L18 (Histogram), L19 (Algorithm Design Patterns)

---

## Exercise 3.1 — Multi-block Reduction

**Concept introduced in**: L14 (Reduction)

### Problem Statement

Implement a two-pass global sum reduction:

1. **Pass 1** — Each block reduces its chunk of the input array and writes a partial sum to
   `d_partial[blockIdx.x]`.
2. **Pass 2** — A second (single-block) kernel reduces `d_partial` to a single scalar.

Verify that the result equals `N * (N - 1) / 2` for input `[0, 1, 2, ..., N-1]`.

### Requirements

- N = 1 << 25 (32M elements).
- Block size: 256 for pass 1; 256 for pass 2.
- Use shared memory reduction inside each block (tree reduction with `__syncthreads`).
- No `atomicAdd` at global scope — the second kernel handles the final accumulation.

### Starter Code

```cuda
// ex3_1_multiblock_reduce.cu
// Compile: nvcc -O2 -arch=sm_80 -o ex3_1 ex3_1_multiblock_reduce.cu

#include <cuda_runtime.h>
#include <cstdio>

#define N          (1 << 25)   // 32M
#define BLOCK_SIZE 256

// Pass 1: each block reduces BLOCK_SIZE elements from d_in.
// Write one partial sum per block into d_partial.
__global__ void reduce_pass1(const float* d_in, float* d_partial, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Load — guard against out-of-bounds
    sdata[tid] = (gid < n) ? d_in[gid] : 0.0f;
    __syncthreads();

    // TODO: Tree reduction in shared memory.
    // Iterate: for (int s = blockDim.x / 2; s > 0; s >>= 1)
    //            if (tid < s) sdata[tid] += sdata[tid + s];
    //            __syncthreads();

    // TODO: Thread 0 writes result to d_partial[blockIdx.x]
}

// Pass 2: single block reduces d_partial (of length nblocks) to d_result[0].
__global__ void reduce_pass2(const float* d_partial, float* d_result, int nblocks) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    // TODO: Load d_partial into shared memory (guard tid < nblocks)
    // TODO: Tree reduction
    // TODO: Thread 0 writes d_result[0]
}

int main() {
    const int n = N;
    const int nblocks1 = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float* h_in = new float[n];
    for (int i = 0; i < n; ++i) h_in[i] = static_cast<float>(i);

    float *d_in, *d_partial, *d_result;
    // TODO: cudaMalloc d_in (n), d_partial (nblocks1), d_result (1)
    // TODO: cudaMemcpy h_in -> d_in

    // Pass 1
    size_t smem1 = BLOCK_SIZE * sizeof(float);
    // TODO: reduce_pass1<<<nblocks1, BLOCK_SIZE, smem1>>>(d_in, d_partial, n)

    // Pass 2: nblocks1 must be <= BLOCK_SIZE for a single-block reduction.
    // If nblocks1 > BLOCK_SIZE you would need a third pass or more blocks.
    // For N = 1<<25 and BLOCK_SIZE = 256: nblocks1 = 131072 > 256.
    // Fix: use a loop inside pass2 or reduce nblocks1 in stages.
    // For this exercise, simplify by doing an extra CPU reduction of d_partial.
    float* h_partial = new float[nblocks1];
    // TODO: cudaMemcpy d_partial -> h_partial
    double gpu_sum = 0.0;
    for (int i = 0; i < nblocks1; ++i) gpu_sum += h_partial[i];

    double expected = (double)n * (n - 1) / 2.0;
    double rel_err  = (gpu_sum - expected) / expected;
    printf("GPU sum:  %.0f\n", gpu_sum);
    printf("Expected: %.0f\n", expected);
    printf("Rel err:  %.2e\n", rel_err);
    printf("Result: %s\n", (rel_err < 1e-4) ? "PASS" : "FAIL");

    // TODO: cudaFree x3
    delete[] h_in; delete[] h_partial;
    return 0;
}
```

### Expected Output

```
GPU sum:  536870895616
Expected: 536870895616
Rel err:  0.00e+00
Result: PASS
```

### Hints

- For the extension challenge: implement a fully GPU-side two-pass where pass 2 is a single block launched with `nblocks1` threads (pad to next power-of-2).
- Floating-point summation of 32M elements can accumulate error — use double for the expected value.
- Profile with `ncu --metrics l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum` to confirm shared memory usage.

### Performance Target

Pass 1 should achieve > 50% of peak memory bandwidth. The reduction is memory-bound; compute time is negligible.

---

## Exercise 3.2 — Blelloch (Work-Efficient) Parallel Scan

**Concept introduced in**: L15 (Scan / Prefix Sum)

### Problem Statement

Implement an exclusive prefix sum (scan) using the Blelloch work-efficient algorithm:

1. **Up-sweep** (reduce phase): build a partial-sum tree in-place in shared memory.
2. **Down-sweep** phase: distribute results back down the tree.

Verify on a small test array `[1, 2, 3, 4, 5]` and on a large random array.

### Requirements

- Single-block version for arrays that fit in shared memory (N ≤ 2 × BLOCK_SIZE).
- Block size: 512 (handles arrays up to 1024 elements).
- Exclusive scan: output[0] = 0, output[i] = sum(input[0..i-1]).
- Verify: input `[1, 2, 3, 4, 5]` → exclusive scan → `[0, 1, 3, 6, 10]`.

### Starter Code

```cuda
// ex3_2_scan.cu
// Compile: nvcc -O2 -arch=sm_80 -o ex3_2 ex3_2_scan.cu

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#define BLOCK_SIZE 512

// Blelloch scan — operates on shared memory array temp[] of size 2*BLOCK_SIZE.
// Each thread handles two elements.
__global__ void exclusive_scan(const float* d_in, float* d_out, int n) {
    extern __shared__ float temp[];   // 2 * BLOCK_SIZE floats

    int tid = threadIdx.x;
    int ai = tid;
    int bi = tid + blockDim.x;

    // Load two elements per thread
    temp[ai] = (ai < n) ? d_in[ai] : 0.0f;
    temp[bi] = (bi < n) ? d_in[bi] : 0.0f;

    int offset = 1;

    // --- Up-sweep (reduce) phase ---
    // TODO: for d = n_padded >> 1; d > 0; d >>= 1
    //         if (tid < d): ai = offset*(2*tid+1)-1; bi = offset*(2*tid+2)-1
    //                       temp[bi] += temp[ai]
    //         offset <<= 1; __syncthreads()

    // TODO: Clear the last element (identity for addition)
    // if (tid == 0) temp[2*BLOCK_SIZE - 1] = 0.0f;
    // __syncthreads();

    // --- Down-sweep phase ---
    // TODO: for d = 1; d < n_padded; d <<= 1
    //         offset >>= 1
    //         if (tid < d): swap temp[ai] and temp[ai+bi-ai], temp[bi] += temp[ai]
    //         __syncthreads()

    // Store results
    if (ai < n) d_out[ai] = temp[ai];
    if (bi < n) d_out[bi] = temp[bi];
}

int main() {
    // --- Small test ---
    int small_n = 5;
    float h_small[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float expected[] = {0.0f, 1.0f, 3.0f, 6.0f, 10.0f};
    float h_out[5] = {};

    float *d_in, *d_out;
    cudaMalloc(&d_in,  BLOCK_SIZE * 2 * sizeof(float));
    cudaMalloc(&d_out, BLOCK_SIZE * 2 * sizeof(float));
    cudaMemcpy(d_in, h_small, small_n * sizeof(float), cudaMemcpyHostToDevice);

    exclusive_scan<<<1, BLOCK_SIZE, 2 * BLOCK_SIZE * sizeof(float)>>>(d_in, d_out, small_n);
    cudaMemcpy(h_out, d_out, small_n * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Input:    [1, 2, 3, 4, 5]\n");
    printf("Scan out: [%.0f, %.0f, %.0f, %.0f, %.0f]\n",
           h_out[0], h_out[1], h_out[2], h_out[3], h_out[4]);

    bool ok = true;
    for (int i = 0; i < small_n; ++i)
        if (fabsf(h_out[i] - expected[i]) > 0.5f) { ok = false; break; }
    printf("Small test: %s\n\n", ok ? "PASS" : "FAIL");

    // --- Large test: N = 1024 ---
    int large_n = 1024;
    float* h_large_in  = new float[large_n];
    float* h_large_out = new float[large_n];
    for (int i = 0; i < large_n; ++i) h_large_in[i] = 1.0f;  // scan of all-ones → [0,1,2,...,1023]

    // TODO: cudaMemcpy h_large_in -> d_in, launch kernel, cudaMemcpy back
    // TODO: verify h_large_out[i] == i for all i

    cudaFree(d_in); cudaFree(d_out);
    delete[] h_large_in; delete[] h_large_out;
    return ok ? 0 : 1;
}
```

### Expected Output

```
Input:    [1, 2, 3, 4, 5]
Scan out: [0, 1, 3, 6, 10]
Small test: PASS

Large test (all-ones): PASS
```

### Hints

- The shared memory array needs to be padded to the next power-of-2 ≥ N.
- Up-sweep: `for (d = n>>1; d > 0; d >>= 1)` — threads `0..d-1` each update one pair.
- Down-sweep: reverse the loop, swap and add at each level.
- The down-sweep starts by clearing `temp[last] = 0` (the identity element).

### Performance Target

For N = 1024, the single-block scan should complete in < 50 µs. For production use, see the CUB `DeviceScan` which handles multi-block arrays efficiently.

---

## Exercise 3.3 — Bitonic Sort

**Concept introduced in**: L16 (Sort)

### Problem Statement

Implement a bitonic sort for N = 1024 elements (must be a power of 2). Bitonic sort is
a comparison network that maps perfectly to CUDA: each sorting step is a fully independent
set of compare-and-swap operations that threads can execute in parallel.

### Requirements

- N = 1024, fits in a single block.
- Use shared memory to store the array.
- Implement the outer loop (sequence size `k`) and inner loop (subsequence size `j`) using `__syncthreads`.
- Verify: input is a random shuffled array; output is sorted ascending.

### Starter Code

```cuda
// ex3_3_bitonic_sort.cu
// Compile: nvcc -O2 -arch=sm_80 -o ex3_3 ex3_3_bitonic_sort.cu

#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>
#include <cstdlib>

#define N 1024

// Bitonic sort kernel — single block, shared memory.
__global__ void bitonic_sort(float* d_data, int n) {
    extern __shared__ float s[];

    int tid = threadIdx.x;
    s[tid] = d_data[tid];
    __syncthreads();

    // Outer loop: k is the size of the bitonic sequence being merged.
    // k doubles each iteration: 2, 4, 8, ..., n.
    for (int k = 2; k <= n; k <<= 1) {
        // Inner loop: j is the stride of the compare-and-swap.
        // j halves each iteration: k/2, k/4, ..., 1.
        for (int j = k >> 1; j > 0; j >>= 1) {
            // TODO: Compute ixj (XOR partner index).
            // int ixj = tid ^ j;
            // if (ixj > tid):
            //   ascending  = ((tid & k) == 0)
            //   swap if (ascending && s[tid] > s[ixj]) || (!ascending && s[tid] < s[ixj])
            // __syncthreads() after each inner step

            __syncthreads();
        }
    }

    d_data[tid] = s[tid];
}

int main() {
    float h_data[N];
    for (int i = 0; i < N; ++i) h_data[i] = static_cast<float>(N - i);  // reverse order
    // Shuffle for a more interesting test
    srand(42);
    for (int i = N - 1; i > 0; --i) std::swap(h_data[i], h_data[rand() % (i + 1)]);

    float* d_data;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

    bitonic_sort<<<1, N, N * sizeof(float)>>>(d_data, N);

    cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify sorted
    bool sorted = true;
    for (int i = 1; i < N; ++i)
        if (h_data[i] < h_data[i - 1]) { sorted = false; break; }
    printf("First 8: %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f\n",
           h_data[0], h_data[1], h_data[2], h_data[3],
           h_data[4], h_data[5], h_data[6], h_data[7]);
    printf("Last  8: %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f\n",
           h_data[N-8], h_data[N-7], h_data[N-6], h_data[N-5],
           h_data[N-4], h_data[N-3], h_data[N-2], h_data[N-1]);
    printf("Result: %s\n", sorted ? "PASS" : "FAIL");

    cudaFree(d_data);
    return sorted ? 0 : 1;
}
```

### Expected Output

```
First 8: 1 2 3 4 5 6 7 8
Last  8: 1017 1018 1019 1020 1021 1022 1023 1024
Result: PASS
```

### Hints

- Direction of comparison: `ascending = ((tid & k) == 0)`. Threads in the lower half of each `k`-group sort ascending; upper half sorts descending.
- `ixj = tid ^ j` gives the partner index for the compare-and-swap at stride `j`.
- Only one thread of the pair (the one with smaller `tid`) should perform the swap: `if (ixj > tid)`.
- For N > 1024 you'd need a multi-block version or use `thrust::sort`.

### Performance Target

For N = 1024 the kernel runs in < 100 µs. Total operations: O(N log² N) = 1024 × 100 ≈ 100K compare-swaps.

---

## Exercise 3.4 — 2D Laplacian Stencil with Shared Memory Halos

**Concept introduced in**: L17 (Stencil Computations)

### Problem Statement

Implement a 5-point 2D Laplacian stencil:

```
out[i][j] = in[i][j-1] + in[i][j+1] + in[i-1][j] + in[i+1][j] - 4*in[i][j]
```

Use shared memory with halo (ghost) cells to avoid redundant global memory reads. Each thread block loads a `(TILE+2) × (TILE+2)` shared memory region including one-cell halos on all sides.

Compare the result to a CPU reference and measure bandwidth.

### Requirements

- Grid size: 2048×2048 floats.
- Tile size: 32×32 interior cells (with 1-cell halo on each side → 34×34 shared memory).
- Boundary condition: treat edges as 0 (Dirichlet).
- Verify max absolute error vs CPU reference < 1e-4.

### Starter Code

```cuda
// ex3_4_laplacian.cu
// Compile: nvcc -O2 -arch=sm_80 -o ex3_4 ex3_4_laplacian.cu

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#define DIM  2048
#define TILE 32
#define HALO (TILE + 2)

// CPU reference
void laplacian_cpu(const float* in, float* out, int rows, int cols) {
    for (int i = 1; i < rows - 1; ++i)
        for (int j = 1; j < cols - 1; ++j)
            out[i * cols + j] = in[i * cols + j - 1] + in[i * cols + j + 1]
                              + in[(i-1) * cols + j] + in[(i+1) * cols + j]
                              - 4.0f * in[i * cols + j];
}

// GPU kernel — 5-point Laplacian with shared memory halos.
__global__ void laplacian_gpu(const float* __restrict__ in,
                              float* __restrict__       out,
                              int rows, int cols) {
    // Shared memory: (TILE+2) x (TILE+2), indexed as [local_row][local_col]
    __shared__ float s[HALO][HALO];

    int tx = threadIdx.x, ty = threadIdx.y;
    int col = blockIdx.x * TILE + tx;
    int row = blockIdx.y * TILE + ty;

    // Local indices in shared memory (offset by 1 for halo)
    int lx = tx + 1, ly = ty + 1;

    // TODO: Load interior cell into s[ly][lx] (guard row < rows && col < cols)
    // TODO: Load halo cells:
    //   - Left halo:   s[ly][0]        from in[row * cols + col - 1]  (guard col > 0)
    //   - Right halo:  s[ly][TILE+1]   from in[row * cols + col + TILE] (guard col+TILE < cols)
    //   - Top halo:    s[0][lx]        from in[(row-1) * cols + col]  (guard row > 0)
    //   - Bottom halo: s[TILE+1][lx]   from in[(row+1) * cols + col]  (guard row+TILE < rows)
    //   Only threads in the appropriate border lanes load halos.
    // TODO: __syncthreads()

    // Compute stencil — skip boundary rows/cols
    if (row > 0 && row < rows - 1 && col > 0 && col < cols - 1) {
        // TODO: out[row*cols+col] = s[ly][lx-1] + s[ly][lx+1] + s[ly-1][lx] + s[ly+1][lx] - 4*s[ly][lx]
    }
}

int main() {
    const int rows = DIM, cols = DIM;
    const size_t bytes = rows * cols * sizeof(float);

    float* h_in  = new float[rows * cols];
    float* h_out_cpu = new float[rows * cols]();
    float* h_out_gpu = new float[rows * cols]();
    for (int i = 0; i < rows * cols; ++i) h_in[i] = static_cast<float>(rand()) / RAND_MAX;

    // CPU reference
    laplacian_cpu(h_in, h_out_cpu, rows, cols);

    // GPU
    float *d_in, *d_out;
    // TODO: cudaMalloc d_in, d_out; copy h_in -> d_in; cudaMemset d_out to 0

    dim3 block(TILE, TILE);
    dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    laplacian_gpu<<<grid, block>>>(d_in, d_out, rows, cols);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms, s, e);

    // TODO: cudaMemcpy d_out -> h_out_gpu

    // Verify
    float max_err = 0.0f;
    for (int i = 0; i < rows * cols; ++i)
        max_err = fmaxf(max_err, fabsf(h_out_gpu[i] - h_out_cpu[i]));

    // Arithmetic intensity: 9 flops, reads ~5 unique cells ≈ 20 bytes (with reuse), 1 write
    double bw = (5.0 + 1.0) * bytes / (ms * 1e-3) / 1e9;
    printf("Time: %.3f ms  Effective BW: %.1f GB/s\n", ms, bw);
    printf("Max error vs CPU: %.2e\n", max_err);
    printf("Result: %s\n", (max_err < 1e-4f) ? "PASS" : "FAIL");

    // TODO: cudaFree x2
    delete[] h_in; delete[] h_out_cpu; delete[] h_out_gpu;
    return 0;
}
```

### Expected Output

```
Time: 0.82 ms  Effective BW: 201.3 GB/s
Max error vs CPU: 0.00e+00
Result: PASS
```

### Hints

- Halo loading is the tricky part: use `tx == 0` to load the left halo, `tx == TILE-1` for right, etc.
- Alternatively, use 4 boundary conditions inside the stencil computation and skip shared memory halos (simpler, slightly slower).
- The `__restrict__` qualifier enables the compiler to use the read-only cache (L1 texture path) for `d_in`.

### Performance Target

Should achieve > 60% of peak memory bandwidth. The stencil reads ~5 elements and writes 1 per interior point.

---

## Exercise 3.5 — Privatized Histogram

**Concept introduced in**: L18 (Histogram)

### Problem Statement

Compute a 256-bin histogram of an array of byte values. Compare two approaches:

1. **Global atomic**: `atomicAdd(&d_hist[val], 1)` directly in global memory.
2. **Privatized (shared memory)**: each block maintains a private per-block histogram in shared memory using `atomicAdd` on `__shared__` memory, then accumulates to global memory once per block.

Verify both match a CPU reference. Measure throughput (elements/sec) for each.

### Requirements

- N = 64M bytes.
- 256 bins (one per byte value 0–255).
- Block size: 256 threads for global version, 256 threads for privatized version.

### Starter Code

```cuda
// ex3_5_histogram.cu
// Compile: nvcc -O2 -arch=sm_80 -o ex3_5 ex3_5_histogram.cu

#include <cuda_runtime.h>
#include <cstdio>

#define N       (1 << 26)   // 64M bytes
#define NBINS   256
#define BLOCK_SIZE 256

// Version 1: global atomics — simple but high contention
__global__ void hist_global(const unsigned char* d_in, unsigned int* d_hist, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) atomicAdd(&d_hist[d_in[gid]], 1u);
}

// Version 2: privatized — per-block shared memory histogram
__global__ void hist_privatized(const unsigned char* d_in, unsigned int* d_hist, int n) {
    __shared__ unsigned int s_hist[NBINS];

    int tid = threadIdx.x;

    // TODO: Initialize s_hist to 0 (stride through NBINS with blockDim.x threads)
    // __syncthreads();

    // TODO: Each thread processes multiple elements with a grid-stride loop
    // for (int i = gid; i < n; i += gridDim.x * blockDim.x)
    //     atomicAdd(&s_hist[d_in[i]], 1u);
    // __syncthreads();

    // TODO: Accumulate s_hist into d_hist with atomicAdd
}

int main() {
    unsigned char* h_in = new unsigned char[N];
    unsigned int h_hist_cpu[NBINS] = {};
    unsigned int h_hist_global[NBINS] = {};
    unsigned int h_hist_priv[NBINS] = {};

    // Fill with values 0-255 in a pattern
    for (int i = 0; i < N; ++i) {
        h_in[i] = static_cast<unsigned char>(i % NBINS);
        h_hist_cpu[h_in[i]]++;
    }

    unsigned char* d_in;
    unsigned int*  d_hist;
    cudaMalloc(&d_in,   N * sizeof(unsigned char));
    cudaMalloc(&d_hist, NBINS * sizeof(unsigned int));
    cudaMemcpy(d_in, h_in, N * sizeof(unsigned char), cudaMemcpyHostToDevice);

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);

    // --- Global atomic version ---
    cudaMemset(d_hist, 0, NBINS * sizeof(unsigned int));
    int nblocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaEventRecord(s);
    hist_global<<<nblocks, BLOCK_SIZE>>>(d_in, d_hist, N);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms_global; cudaEventElapsedTime(&ms_global, s, e);
    cudaMemcpy(h_hist_global, d_hist, NBINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // --- Privatized version ---
    cudaMemset(d_hist, 0, NBINS * sizeof(unsigned int));
    // Use fewer blocks for grid-stride loop: e.g., 256 blocks
    int nblocks_priv = 256;
    cudaEventRecord(s);
    hist_privatized<<<nblocks_priv, BLOCK_SIZE>>>(d_in, d_hist, N);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms_priv; cudaEventElapsedTime(&ms_priv, s, e);
    cudaMemcpy(h_hist_priv, d_hist, NBINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Verify both match CPU reference
    bool ok_global = true, ok_priv = true;
    for (int b = 0; b < NBINS; ++b) {
        if (h_hist_global[b] != h_hist_cpu[b]) ok_global = false;
        if (h_hist_priv[b]   != h_hist_cpu[b]) ok_priv   = false;
    }

    printf("Global atomic:   %.2f ms  %.1f MElems/s  Correct: %s\n",
           ms_global, N / (ms_global * 1e-3) / 1e6, ok_global ? "YES" : "NO");
    printf("Privatized smem: %.2f ms  %.1f MElems/s  Correct: %s\n",
           ms_priv,   N / (ms_priv   * 1e-3) / 1e6, ok_priv   ? "YES" : "NO");
    printf("Speedup: %.2fx\n", ms_global / ms_priv);

    cudaFree(d_in); cudaFree(d_hist);
    delete[] h_in;
    return (ok_global && ok_priv) ? 0 : 1;
}
```

### Expected Output (A100 example)

```
Global atomic:   12.4 ms   51.6 MElems/s  Correct: YES
Privatized smem: 2.1 ms   304.8 MElems/s  Correct: YES
Speedup: 5.9x
```

### Hints

- Shared memory `atomicAdd` is much faster than global `atomicAdd` because it avoids the L2 cache bottleneck.
- Grid-stride loop pattern: `for (int i = gid; i < n; i += gridDim.x * blockDim.x)` — each thread processes multiple elements.
- For a uniform distribution the global atomic kernel has massive contention: 256M / 256 bins = 1M increments per bin.
- After shared memory reduction, use `if (tid < NBINS) atomicAdd(&d_hist[tid], s_hist[tid])` for the final accumulation.

### Performance Target

Privatized version should be at least 3× faster than global atomic. Achieved throughput should approach 300+ MElems/s on Ampere.
