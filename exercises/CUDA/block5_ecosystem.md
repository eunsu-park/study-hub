# Block 5 — Ecosystem

**Lessons covered**: L28 (Thrust), L29 (cuBLAS), L30 (Tensor Cores / WMMA),
L31 (Cooperative Groups & Grid-level Sync)

---

## Exercise 5.1 — Thrust Sort

**Concept introduced in**: L28 (Thrust)

### Problem Statement

Sort 1 million floats on the GPU using `thrust::sort` on a `thrust::device_vector`. Measure
and compare elapsed time against `std::sort` on a CPU vector of the same size.

### Requirements

- N = 1,000,000 floats, filled with random values.
- Use `thrust::device_vector<float>` and `thrust::sort`.
- Time the GPU sort with CUDA events.
- Time the CPU sort with `std::chrono`.
- Print both times and the speedup.
- Verify the GPU output is sorted (check adjacent elements).

### Starter Code

```cuda
// ex5_1_thrust_sort.cu
// Compile: nvcc -O2 -arch=sm_80 -o ex5_1 ex5_1_thrust_sort.cu

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/generate.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#define N 1000000

int main() {
    // Fill host vector with random floats
    thrust::host_vector<float> h_vec(N);
    std::generate(h_vec.begin(), h_vec.end(), []() {
        return static_cast<float>(rand()) / RAND_MAX;
    });

    // --- CPU sort ---
    thrust::host_vector<float> h_ref = h_vec;  // copy for CPU sort
    auto t0 = std::chrono::high_resolution_clock::now();
    std::sort(h_ref.begin(), h_ref.end());
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // --- GPU Thrust sort ---
    // TODO: Create thrust::device_vector<float> d_vec from h_vec
    // TODO: Warm-up: thrust::sort(d_vec.begin(), d_vec.end())
    // TODO: Re-fill d_vec with original h_vec data (thrust::copy)

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);

    // TODO: cudaEventRecord(s)
    // TODO: thrust::sort(d_vec.begin(), d_vec.end())
    // TODO: cudaEventRecord(e); cudaEventSynchronize(e)
    float gpu_ms = 0.0f;
    // TODO: cudaEventElapsedTime(&gpu_ms, s, e)

    // Copy back and verify
    // TODO: thrust::host_vector<float> h_result = d_vec  (triggers D2H copy)
    bool sorted = true;
    // TODO: for (int i = 1; i < N; ++i) if (h_result[i] < h_result[i-1]) { sorted = false; break; }

    printf("N = %d elements\n", N);
    printf("CPU std::sort:    %.2f ms\n", cpu_ms);
    printf("GPU thrust::sort: %.2f ms\n", gpu_ms);
    printf("Speedup: %.2fx\n", cpu_ms / gpu_ms);
    printf("GPU sorted correctly: %s\n", sorted ? "YES" : "NO");

    cudaEventDestroy(s); cudaEventDestroy(e);
    return sorted ? 0 : 1;
}
```

### Expected Output (RTX 3080 example)

```
N = 1000000 elements
CPU std::sort:    98.4 ms
GPU thrust::sort: 4.2 ms
Speedup: 23.4x
```

### Hints

- `thrust::device_vector` handles all `cudaMalloc` / `cudaFree` internally.
- The first GPU sort call may be slower due to JIT compilation; always warm up before timing.
- `thrust::sort` uses a radix sort implementation internally — O(kN) where k is the key width in bits.
- For descending order: `thrust::sort(d.begin(), d.end(), thrust::greater<float>())`.

### Performance Target

GPU should be at least 10× faster than single-threaded `std::sort` for N = 1M.

---

## Exercise 5.2 — cuBLAS SGEMM

**Concept introduced in**: L29 (cuBLAS)

### Problem Statement

Use `cublasSgemm` to compute `C = alpha * A * B + beta * C` for 1024×1024 float matrices.
Verify the result matches a naive CPU matrix multiply (or a reference computed via Thrust).
Measure achieved GFLOP/s and compare to your GPU's theoretical peak.

### Requirements

- M = N = K = 1024.
- `alpha = 1.0f`, `beta = 0.0f`.
- cuBLAS uses **column-major** layout — handle the row-major ↔ column-major conversion.
- Run 10 iterations for timing; print average GFLOP/s.
- Verify max absolute error vs CPU < 1e-3 (float32 matmul accumulates ~N * eps ≈ 1e-4 error).

### Starter Code

```cuda
// ex5_2_cublas_sgemm.cu
// Compile: nvcc -O2 -arch=sm_80 -o ex5_2 ex5_2_cublas_sgemm.cu -lcublas

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>

#define M_DIM 1024
#define N_DIM 1024
#define K_DIM 1024
#define NRUNS 10

// Naive CPU matmul for verification (slow, only for small test)
void cpu_matmul(const float* A, const float* B, float* C, int m, int n, int k) {
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int l = 0; l < k; ++l) sum += A[i * k + l] * B[l * n + j];
            C[i * n + j] = sum;
        }
}

int main() {
    int m = M_DIM, n = N_DIM, k = K_DIM;
    size_t sz_A = m * k * sizeof(float);
    size_t sz_B = k * n * sizeof(float);
    size_t sz_C = m * n * sizeof(float);

    float* h_A = new float[m * k];
    float* h_B = new float[k * n];
    float* h_C = new float[m * n]();
    float* h_C_ref = new float[m * n]();

    for (int i = 0; i < m * k; ++i) h_A[i] = (float)rand() / RAND_MAX - 0.5f;
    for (int i = 0; i < k * n; ++i) h_B[i] = (float)rand() / RAND_MAX - 0.5f;

    float *d_A, *d_B, *d_C;
    // TODO: cudaMalloc d_A, d_B, d_C
    // TODO: cudaMemcpy h_A -> d_A, h_B -> d_B

    // Create cuBLAS handle
    cublasHandle_t handle;
    // TODO: cublasCreate(&handle)

    float alpha = 1.0f, beta = 0.0f;

    // cuBLAS is column-major. Trick: compute C^T = B^T * A^T (swap A<->B and use their transposes).
    // Equivalently: call cublasSgemm with CUBLAS_OP_N, CUBLAS_OP_N but swap A and B:
    //   cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
    //               &alpha, d_B, n, d_A, k, &beta, d_C, n)
    // This gives row-major C = A * B directly.

    // Warm-up
    // TODO: call cublasSgemm once

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int r = 0; r < NRUNS; ++r) {
        // TODO: cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
        //                   &alpha, d_B, n, d_A, k, &beta, d_C, n)
    }
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms, s, e);
    float avg_ms = ms / NRUNS;

    double flops   = 2.0 * m * n * k;
    double gflops  = flops / (avg_ms * 1e-3) / 1e9;
    printf("cublasSgemm: %.2f ms  %.1f GFLOP/s\n", avg_ms, gflops);

    // Verify (on a smaller submatrix to avoid CPU matmul being too slow)
    // TODO: cudaMemcpy d_C -> h_C
    int v = 64;  // verify top-left 64x64 submatrix only
    cpu_matmul(h_A, h_B, h_C_ref, v, v, k);
    float max_err = 0.0f;
    for (int i = 0; i < v * v; ++i) max_err = fmaxf(max_err, fabsf(h_C[i] - h_C_ref[i]));
    printf("Max error (64x64 sub): %.4e\n", max_err);
    printf("Result: %s\n", (max_err < 1e-1f) ? "PASS" : "FAIL");
    // Note: large k accumulates floating point error; threshold is generous

    // TODO: cublasDestroy, cudaFree x3
    delete[] h_A; delete[] h_B; delete[] h_C; delete[] h_C_ref;
    cudaEventDestroy(s); cudaEventDestroy(e);
    return 0;
}
```

### Expected Output (A100 example)

```
cublasSgemm: 0.38 ms  5703.2 GFLOP/s
Max error (64x64 sub): 3.8147e-06
Result: PASS
```

### Hints

- cuBLAS assumes column-major (Fortran) layout. The "swap A and B" trick works because `(AB)^T = B^T A^T` and transposing a row-major matrix gives a column-major one.
- `CUBLAS_OP_N` = no transpose, `CUBLAS_OP_T` = transpose.
- Leading dimension parameters (`lda`, `ldb`, `ldc`) are the number of rows in the column-major matrix; for a row-major M×K matrix treated as column-major K×M, `lda = K`.

### Performance Target

cuBLAS SGEMM on A100 should approach 15–20 TFLOP/s (FP32). On an RTX 3080 expect 5–10 TFLOP/s.

---

## Exercise 5.3 — WMMA Half-precision GEMM

**Concept introduced in**: L30 (Tensor Cores / WMMA)

### Problem Statement

Use the WMMA (Warp Matrix Multiply-Accumulate) API to compute a 16×16×16 half-precision
matrix multiplication: `C_frag = A_frag * B_frag` where A is 16×16 FP16, B is 16×16 FP16,
and C is 16×16 FP32.

Verify the result against `cublasSgemm` on the same inputs (after converting FP16 → FP32).

### Requirements

- Exactly one warp (32 threads) handles one 16×16×16 tile.
- Load fragments from device memory; execute `wmma::mma_sync`; store the result.
- Verify max absolute error vs cuBLAS < 0.1 (FP16 has ~3 decimal digits of precision).

### Starter Code

```cuda
// ex5_3_wmma_gemm.cu
// Compile: nvcc -O2 -arch=sm_80 -o ex5_3 ex5_3_wmma_gemm.cu -lcublas

#include <cuda_runtime.h>
#include <mma.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cmath>

using namespace nvcuda;

#define WARP_SIZE 16  // WMMA tile dimension

// Convert FP32 to FP16 helper kernel
__global__ void fp32_to_fp16(const float* src, __half* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

// WMMA kernel: computes C = A * B for 16x16x16 tiles.
// Launch with exactly 1 warp (32 threads) in 1 block.
__global__ void wmma_gemm(const __half* A, const __half* B, float* C, int m, int n, int k) {
    // Declare WMMA fragments
    // TODO: wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    // TODO: wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
    // TODO: wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // Initialize accumulator to zero
    // TODO: wmma::fill_fragment(c_frag, 0.0f);

    // Load A and B fragments from global memory
    // TODO: wmma::load_matrix_sync(a_frag, A, 16);  // leading dim = 16 (row-major)
    // TODO: wmma::load_matrix_sync(b_frag, B, 16);  // leading dim = 16 (col-major)

    // Matrix multiply
    // TODO: wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Store result
    // TODO: wmma::store_matrix_sync(C, c_frag, 16, wmma::mem_row_major);
}

int main() {
    const int dim = 16;
    const int sz  = dim * dim;

    float h_A_fp32[sz], h_B_fp32[sz], h_C_wmma[sz], h_C_ref[sz];
    for (int i = 0; i < sz; ++i) {
        h_A_fp32[i] = (float)(rand() % 10) / 10.0f;
        h_B_fp32[i] = (float)(rand() % 10) / 10.0f;
    }

    // Allocate device buffers
    float  *d_A_fp32, *d_B_fp32, *d_C_wmma, *d_C_ref;
    __half *d_A_fp16, *d_B_fp16;
    // TODO: cudaMalloc all 6 buffers
    // TODO: cudaMemcpy h_A_fp32 -> d_A_fp32, h_B_fp32 -> d_B_fp32

    // Convert FP32 → FP16
    // TODO: launch fp32_to_fp16 kernel for A and B

    // Run WMMA kernel (1 block, 32 threads = 1 warp)
    wmma_gemm<<<1, 32>>>(d_A_fp16, d_B_fp16, d_C_wmma, dim, dim, dim);
    cudaMemcpy(h_C_wmma, d_C_wmma, sz * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify against CPU reference
    for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j) {
            float sum = 0.0f;
            for (int l = 0; l < dim; ++l)
                sum += h_A_fp32[i * dim + l] * h_B_fp32[l * dim + j];
            h_C_ref[i * dim + j] = sum;
        }

    float max_err = 0.0f;
    for (int i = 0; i < sz; ++i) max_err = fmaxf(max_err, fabsf(h_C_wmma[i] - h_C_ref[i]));
    printf("Max error vs CPU FP32: %.4f\n", max_err);
    printf("Result: %s\n", (max_err < 0.1f) ? "PASS" : "FAIL");

    // TODO: cudaFree all
    return (max_err < 0.1f) ? 0 : 1;
}
```

### Expected Output

```
Max error vs CPU FP32: 0.0000
Result: PASS
```

### Hints

| Fragment type | Declaration |
|--------------|-------------|
| A (row-major) | `wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>` |
| B (col-major) | `wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major>` |
| C (accumulator) | `wmma::fragment<wmma::accumulator, 16, 16, 16, float>` |

- WMMA requires the matrix data to be in device memory; you cannot use host pointers.
- `wmma::load_matrix_sync` and `wmma::store_matrix_sync` are warp-collective — all 32 threads in the warp call them together.
- Requires compute capability ≥ sm_70.

### Performance Target

For a 16×16×16 tile, this is a proof-of-concept; throughput measurement is only meaningful for tiled matmul over large matrices. The focus here is correctness.

---

## Exercise 5.4 — Grid-level Reduction with Cooperative Groups

**Concept introduced in**: L31 (Cooperative Groups & Grid-level Sync)

### Problem Statement

Implement a single-pass global sum reduction using `cooperative_groups::grid_group::sync()`
to synchronize all blocks within a single kernel launch — eliminating the need for a
two-pass approach (Exercise 3.1).

Each block reduces its chunk, writes the partial sum to a global array, then all blocks
synchronize via `grid.sync()`. Block 0 then reduces the partial sums.

### Requirements

- N = 1 << 24 (16M elements).
- Use `cooperative_groups::this_grid()` and `grid.sync()`.
- Must be launched with `cudaLaunchCooperativeKernel`.
- Verify result equals N * (N-1) / 2.

### Starter Code

```cuda
// ex5_4_coop_reduce.cu
// Compile: nvcc -O2 -arch=sm_80 -o ex5_4 ex5_4_coop_reduce.cu

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>

namespace cg = cooperative_groups;

#define N          (1 << 24)
#define BLOCK_SIZE 256

__global__ void coop_reduce(const float* d_in, float* d_partial, float* d_result, int n) {
    cg::grid_group grid = cg::this_grid();

    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Phase 1: each block reduces its chunk
    sdata[tid] = (gid < n) ? d_in[gid] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    // TODO: if (tid == 0) d_partial[blockIdx.x] = sdata[0];

    // Phase 2: sync across all blocks in the grid
    // TODO: grid.sync();

    // Phase 3: block 0 reduces d_partial into d_result[0]
    if (blockIdx.x == 0) {
        // TODO: load d_partial into sdata (stride if nblocks > BLOCK_SIZE)
        // TODO: reduce sdata
        // TODO: if (tid == 0) d_result[0] = sdata[0];
    }
}

int main() {
    const int n = N;
    const int nblocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float* h_in = new float[n];
    for (int i = 0; i < n; ++i) h_in[i] = static_cast<float>(i);

    float *d_in, *d_partial, *d_result;
    cudaMalloc(&d_in,      n       * sizeof(float));
    cudaMalloc(&d_partial, nblocks * sizeof(float));
    cudaMalloc(&d_result,  1       * sizeof(float));
    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_partial, 0, nblocks * sizeof(float));
    cudaMemset(d_result,  0, sizeof(float));

    // Check that cooperative launch is supported
    int supports_coop;
    cudaDeviceGetAttribute(&supports_coop,
                           cudaDevAttrCooperativeLaunch,
                           0 /*device 0*/);
    if (!supports_coop) {
        printf("Cooperative launch not supported on this device.\n");
        return 1;
    }

    // Launch via cudaLaunchCooperativeKernel
    void* args[] = {&d_in, &d_partial, &d_result, &n};
    size_t smem = BLOCK_SIZE * sizeof(float);
    // TODO: cudaLaunchCooperativeKernel((void*)coop_reduce, nblocks, BLOCK_SIZE, args, smem)

    float h_result;
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    double expected = (double)n * (n - 1) / 2.0;
    double rel_err  = fabs(h_result - expected) / expected;
    printf("GPU sum:  %.0f\n", (double)h_result);
    printf("Expected: %.0f\n", expected);
    printf("Rel err:  %.2e\n", rel_err);
    printf("Result: %s\n", (rel_err < 1e-4) ? "PASS" : "FAIL");

    cudaFree(d_in); cudaFree(d_partial); cudaFree(d_result);
    delete[] h_in;
    return (rel_err < 1e-4) ? 0 : 1;
}
```

### Expected Output

```
GPU sum:  140737479966720
Expected: 140737479966720
Rel err:  0.00e+00
Result: PASS
```

### Hints

- `cudaLaunchCooperativeKernel` takes: kernel function pointer, grid dim, block dim, args array, shared mem, stream.
- `grid.sync()` is a device-side collective barrier — all thread blocks in the grid must reach it before any proceeds.
- The number of blocks that can participate is limited by the GPU's occupancy at the given register/smem usage. Query with `cudaOccupancyMaxActiveBlocksPerMultiprocessor`.
- Cooperative kernels require CUDA 9.0+ and a Volta+ GPU.

### Performance Target

Single-pass cooperative reduction should be within 5% of the two-pass version (Ex 3.1). The main benefit is code simplicity, not raw performance.
