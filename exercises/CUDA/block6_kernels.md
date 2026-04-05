# Block 6 — Production Kernels

**Lessons covered**: L32 (GEMM Optimization), L33 (Softmax & Attention),
L34 (FlashAttention), L35 (Quantization), L36 (Kernel Fusion)

---

## Exercise 6.1 — Tiled GEMM v1 → v2

**Concept introduced in**: L32 (GEMM Optimization)

### Problem Statement

Implement two versions of square matrix multiply (C = A × B, all float32) and measure the
speedup from naive global memory (v1) to shared memory tiled matmul (v2):

- **v1 (naive)**: each thread computes one `C[i][j]` by dot-producting row `i` of A with column `j` of B, reading directly from global memory.
- **v2 (tiled)**: each thread block loads a `TILE × TILE` tile of A and B into shared memory, accumulates into a register, then advances to the next tile.

### Requirements

- Matrix size: 1024×1024 (all square).
- Tile size: 32×32.
- Verify max absolute error vs cuBLAS < 1e-2.
- Print GFLOP/s for both versions; v2 should be 2–4× faster than v1.

### Starter Code

```cuda
// ex6_1_gemm_v1_v2.cu
// Compile: nvcc -O2 -arch=sm_80 -o ex6_1 ex6_1_gemm_v1_v2.cu -lcublas

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>

#define DIM  1024
#define TILE 32
#define NRUNS 10

// v1: Naive GEMM — each thread reads a full row of A and column of B
__global__ void gemm_v1(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= n) return;

    float sum = 0.0f;
    // TODO: for (int k = 0; k < n; ++k) sum += A[row * n + k] * B[k * n + col];
    // TODO: C[row * n + col] = sum;
}

// v2: Tiled GEMM with shared memory
__global__ void gemm_v2(const float* A, const float* B, float* C, int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;
    int ntiles = (n + TILE - 1) / TILE;

    for (int t = 0; t < ntiles; ++t) {
        // TODO: Load tile of A into As[ty][tx]
        //   int a_col = t * TILE + tx;
        //   As[ty][tx] = (row < n && a_col < n) ? A[row * n + a_col] : 0.0f;

        // TODO: Load tile of B into Bs[ty][tx]
        //   int b_row = t * TILE + ty;
        //   Bs[ty][tx] = (b_row < n && col < n) ? B[b_row * n + col] : 0.0f;

        // TODO: __syncthreads()

        // TODO: Accumulate dot product over this tile
        // for (int k = 0; k < TILE; ++k) sum += As[ty][k] * Bs[k][tx];

        // TODO: __syncthreads()
    }

    // TODO: if (row < n && col < n) C[row * n + col] = sum;
}

float benchmark_kernel(auto launch_fn, int nruns) {
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    launch_fn();  // warm-up
    cudaEventRecord(s);
    for (int r = 0; r < nruns; ++r) launch_fn();
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms / nruns;
}

int main() {
    const int n = DIM;
    const int sz = n * n;

    float* h_A = new float[sz];
    float* h_B = new float[sz];
    float* h_C = new float[sz];
    for (int i = 0; i < sz; ++i) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }

    float *d_A, *d_B, *d_C1, *d_C2, *d_C_ref;
    // TODO: cudaMalloc 5 arrays; copy h_A -> d_A, h_B -> d_B

    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);

    auto run_v1 = [&]() { gemm_v1<<<grid, block>>>(d_A, d_B, d_C1, n); };
    auto run_v2 = [&]() { gemm_v2<<<grid, block>>>(d_A, d_B, d_C2, n); };

    float ms_v1 = benchmark_kernel(run_v1, NRUNS);
    float ms_v2 = benchmark_kernel(run_v2, NRUNS);

    double flops  = 2.0 * n * n * n;
    double gf_v1  = flops / (ms_v1 * 1e-3) / 1e9;
    double gf_v2  = flops / (ms_v2 * 1e-3) / 1e9;
    printf("v1 (naive):  %.2f ms  %.1f GFLOP/s\n", ms_v1, gf_v1);
    printf("v2 (tiled):  %.2f ms  %.1f GFLOP/s\n", ms_v2, gf_v2);
    printf("Speedup: %.2fx\n", ms_v1 / ms_v2);

    // Verify v2 vs cuBLAS reference
    cublasHandle_t handle; cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;
    // TODO: cudaMalloc d_C_ref; run cublasSgemm for reference output

    // TODO: cudaMemcpy d_C2 -> h_C; compare to cublasSgemm result

    // TODO: cublasDestroy, cudaFree all
    delete[] h_A; delete[] h_B; delete[] h_C;
    return 0;
}
```

### Expected Output (A100 example)

```
v1 (naive):  18.4 ms  116.9 GFLOP/s
v2 (tiled):   5.1 ms  421.3 GFLOP/s
Speedup: 3.61x
Max error vs cuBLAS: 2.8e-06
Result: PASS
```

### Hints

- v1 is bottlenecked by uncoalesced column accesses to B: each thread in a warp accesses a different column → 32 separate cache-line loads.
- v2 eliminates this by loading a `TILE × TILE` tile of B cooperatively (each thread loads `B[b_row * n + col]` where `col` varies with `tx` → coalesced row load).
- Each tile loads `TILE * TILE` floats per matrix → `TILE` reuses each loaded element → arithmetic intensity improves by `TILE` factor.

### Performance Target

v2 should be 2–4× faster than v1. Beyond v2, further optimizations (register blocking, double buffering, 128-bit loads) push toward cuBLAS performance (v3–v7 in real implementations).

---

## Exercise 6.2 — Online (Single-pass) Softmax

**Concept introduced in**: L33 (Softmax & Attention)

### Problem Statement

Implement numerically stable softmax in a single pass over the input row using the "online"
algorithm that tracks the running maximum and sum simultaneously. Compare against the
two-pass version (first find max, then compute exp and sum).

Verify that max relative error between single-pass and two-pass is < 1e-5 for FP32.

### Requirements

- Input: batch of B=256 rows, each of length L=1024.
- One thread block per row (256 threads, processes L=1024 elements in 4 iterations of 256).
- Use warp-level reductions (`__shfl_down_sync`) for the final cross-warp max and sum.

### Starter Code

```cuda
// ex6_2_online_softmax.cu
// Compile: nvcc -O2 -arch=sm_80 -o ex6_2 ex6_2_online_softmax.cu

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#define B          256    // batch (rows)
#define L          1024   // sequence length (columns)
#define BLOCK_SIZE 256

// Two-pass softmax (reference)
__global__ void softmax_2pass(const float* in, float* out, int l) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int row = blockIdx.x;

    // Pass 1: find row max
    float row_max = -1e38f;
    for (int i = tid; i < l; i += blockDim.x)
        row_max = fmaxf(row_max, in[row * l + i]);
    // Block-reduce max
    sdata[tid] = row_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float gmax = sdata[0];
    __syncthreads();

    // Pass 2: compute exp and sum
    float row_sum = 0.0f;
    for (int i = tid; i < l; i += blockDim.x) {
        float e = expf(in[row * l + i] - gmax);
        out[row * l + i] = e;
        row_sum += e;
    }
    // Block-reduce sum
    sdata[tid] = row_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float gsum = sdata[0];
    __syncthreads();

    // Normalize
    for (int i = tid; i < l; i += blockDim.x)
        out[row * l + i] /= gsum;
}

// Online single-pass softmax
// Track (running_max, running_sum_corrected) simultaneously.
// When a new max m' > m is found: sum' = sum * exp(m - m') + exp(x - m')
__global__ void softmax_online(const float* in, float* out, int l) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int row = blockIdx.x;

    // Each thread maintains its own local (m, d) pair
    float local_m = -1e38f;
    float local_d = 0.0f;

    // TODO: Loop over elements assigned to this thread (stride blockDim.x)
    //   for (int i = tid; i < l; i += blockDim.x):
    //     float x = in[row * l + i]
    //     if (x > local_m):
    //       local_d = local_d * expf(local_m - x) + expf(0.0f)   // correction + new term
    //       local_m = x
    //     else:
    //       local_d += expf(x - local_m)

    // TODO: Block-reduce (local_m, local_d) pairs using shared memory.
    //   Reduction rule: combine (m1, d1) and (m2, d2):
    //     if m1 >= m2: new_m = m1, new_d = d1 + d2 * exp(m2 - m1)
    //     else:        new_m = m2, new_d = d2 + d1 * exp(m1 - m2)

    // TODO: Thread 0 stores global (gmax, gsum) from shared memory

    // TODO: Normalize: for (int i = tid; i < l; i += blockDim.x)
    //                    out[row * l + i] = expf(in[row * l + i] - gmax) / gsum;
}

int main() {
    const int n = B * L;
    float* h_in       = new float[n];
    float* h_out_2p   = new float[n];
    float* h_out_onl  = new float[n];

    for (int i = 0; i < n; ++i) h_in[i] = (float)rand() / RAND_MAX * 10.0f - 5.0f;

    float *d_in, *d_out_2p, *d_out_onl;
    // TODO: cudaMalloc x3, cudaMemcpy h_in -> d_in

    size_t smem = BLOCK_SIZE * sizeof(float);
    softmax_2pass <<<B, BLOCK_SIZE, smem>>>(d_in, d_out_2p,  L);
    softmax_online<<<B, BLOCK_SIZE, smem * 2>>>(d_in, d_out_onl, L);  // *2 for (m,d) pairs
    cudaDeviceSynchronize();

    // TODO: cudaMemcpy d_out_2p -> h_out_2p, d_out_onl -> h_out_onl

    float max_rel_err = 0.0f;
    for (int i = 0; i < n; ++i) {
        float ref = h_out_2p[i];
        if (ref > 1e-30f) max_rel_err = fmaxf(max_rel_err, fabsf(h_out_onl[i] - ref) / ref);
    }
    printf("Max relative error (online vs 2-pass): %.2e\n", max_rel_err);
    printf("Result: %s\n", (max_rel_err < 1e-5f) ? "PASS" : "FAIL");

    // TODO: cudaFree x3
    delete[] h_in; delete[] h_out_2p; delete[] h_out_onl;
    return 0;
}
```

### Expected Output

```
Max relative error (online vs 2-pass): 3.21e-07
Result: PASS
```

### Hints

- The online algorithm combines max-finding and sum-accumulation in a single loop, halving global memory reads.
- The block-level reduction of `(m, d)` pairs requires careful ordering: always express the smaller exponent relative to the larger max.
- For large `L` (> block_size), use a grid-stride loop; each thread processes multiple elements before the block reduction.

### Performance Target

Online softmax should achieve within 5% of the two-pass version (global memory bound). The main benefit is code simplicity in larger attention kernels, not raw throughput.

---

## Exercise 6.3 — FlashAttention-2 Inner Loop Tile

**Concept introduced in**: L34 (FlashAttention)

### Problem Statement

Implement the inner loop of FlashAttention-2 that processes a single `(Br × Bc)` tile of
the query–key product and updates the running statistics (softmax numerator and denominator).

This exercise isolates the "online softmax over KV tiles" pattern without the full
multi-head attention plumbing.

### Requirements

- Query tile Q: `Br × d` = 16 × 64 (FP32).
- Key tile K: `Bc × d` = 16 × 64 (FP32).
- Value tile V: `Bc × d` = 16 × 64 (FP32).
- Output O: `Br × d`.
- Running stats: `m` (per-query row max), `l` (per-query row softmax normalizer).
- Verify output matches a naive attention (QK^T softmax × V) reference for a single tile.

### Starter Code

```cuda
// ex6_3_flash_attn_tile.cu
// Compile: nvcc -O2 -arch=sm_80 -o ex6_3 ex6_3_flash_attn_tile.cu

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#define BR 16    // query tile rows
#define BC 16    // key/value tile rows
#define D  64    // head dimension

// Single-tile FlashAttention inner loop.
// One block handles one (Br x Bc) tile.
// Threads: BR x BC threads (16*16 = 256).
__global__ void flash_attn_tile(const float* Q,   // [BR, D]
                                const float* K,   // [BC, D]
                                const float* V,   // [BC, D]
                                float*       O,   // [BR, D]  (output, accumulated)
                                float*       m,   // [BR]     (running max per query)
                                float*       l,   // [BR]     (running sum per query)
                                float        scale) {
    // Shared memory: Q tile, K tile, V tile, and QK^T tile
    __shared__ float Qs[BR][D];
    __shared__ float Ks[BC][D];
    __shared__ float Vs[BC][D];
    __shared__ float S[BR][BC];   // attention scores

    int tx = threadIdx.x, ty = threadIdx.y;

    // TODO: Load Q tile: each row ty of Q is loaded by threads with tx in [0..D-1]
    // TODO: Load K tile: similar cooperative load
    // TODO: Load V tile
    // TODO: __syncthreads()

    // TODO: Compute S[ty][tx] = scale * dot(Q[ty], K[tx])
    //   (each thread computes one element of the BR x BC score matrix)

    // TODO: __syncthreads()

    // Update running statistics for each query row (only threads with tx==0 update m and l)
    // TODO: for row ty:
    //   float row_max = max of S[ty][0..BC-1]
    //   float m_new = max(m[ty], row_max)
    //   float l_new = exp(m[ty] - m_new) * l[ty] + sum_{j} exp(S[ty][j] - m_new)
    //   // Rescale existing output O[ty] by exp(m[ty] - m_new) / l_new
    //   // Add new contribution: sum_{j} exp(S[ty][j] - m_new) * V[j] / l_new
    //   m[ty] = m_new; l[ty] = l_new;
    //
    // Note: this requires careful coordination between threads.
    // For simplicity, do the per-row update with a single thread (tx==0) per row (ty).
}

int main() {
    // Initialize Q, K, V, O, m, l on host
    float h_Q[BR][D], h_K[BC][D], h_V[BC][D];
    float h_O[BR][D] = {}, h_m[BR], h_l[BR];
    float h_O_ref[BR][D] = {};

    srand(42);
    for (int i = 0; i < BR; ++i)
        for (int j = 0; j < D; ++j) h_Q[i][j] = (float)rand()/RAND_MAX - 0.5f;
    for (int i = 0; i < BC; ++i)
        for (int j = 0; j < D; ++j) {
            h_K[i][j] = (float)rand()/RAND_MAX - 0.5f;
            h_V[i][j] = (float)rand()/RAND_MAX - 0.5f;
        }
    for (int i = 0; i < BR; ++i) { h_m[i] = -1e38f; h_l[i] = 0.0f; }
    float scale = 1.0f / sqrtf((float)D);

    // CPU reference: naive attention for a single tile
    // S = softmax(scale * Q @ K^T) @ V
    float S_cpu[BR][BC] = {};
    for (int i = 0; i < BR; ++i)
        for (int j = 0; j < BC; ++j)
            for (int k = 0; k < D; ++k)
                S_cpu[i][j] += scale * h_Q[i][k] * h_K[j][k];
    for (int i = 0; i < BR; ++i) {
        float mx = -1e38f;
        for (int j = 0; j < BC; ++j) mx = fmaxf(mx, S_cpu[i][j]);
        float sm = 0.0f;
        for (int j = 0; j < BC; ++j) { S_cpu[i][j] = expf(S_cpu[i][j] - mx); sm += S_cpu[i][j]; }
        for (int j = 0; j < BC; ++j) S_cpu[i][j] /= sm;
        for (int j = 0; j < BC; ++j)
            for (int k = 0; k < D; ++k) h_O_ref[i][k] += S_cpu[i][j] * h_V[j][k];
    }

    // GPU buffers
    float *d_Q, *d_K, *d_V, *d_O, *d_m, *d_l;
    // TODO: cudaMalloc and copy all host arrays to device

    dim3 block(BC, BR);
    flash_attn_tile<<<1, block>>>(d_Q, d_K, d_V, d_O, d_m, d_l, scale);
    cudaDeviceSynchronize();

    // TODO: cudaMemcpy d_O -> h_O

    float max_err = 0.0f;
    for (int i = 0; i < BR; ++i)
        for (int j = 0; j < D; ++j)
            max_err = fmaxf(max_err, fabsf(h_O[i][j] - h_O_ref[i][j]));
    printf("Max error vs naive attention: %.6f\n", max_err);
    printf("Result: %s\n", (max_err < 1e-4f) ? "PASS" : "FAIL");

    // TODO: cudaFree all
    return 0;
}
```

### Expected Output

```
Max error vs naive attention: 0.000003
Result: PASS
```

### Hints

- The rescaling step when updating `m_new > m_old`: multiply existing `O[i]` by `exp(m_old - m_new)` before adding the new KV contribution.
- This exercise is a single-tile proof of concept. Real FlashAttention loops over all KV tiles in the outer loop, updating `(m, l, O)` incrementally.
- For a full implementation, see the official FlashAttention-2 paper (Dao, 2023) and the Triton/CUDA reference implementations.

### Performance Target

For a 16×16×64 tile, this is proof-of-concept only. Real FlashAttention achieves > 50% of theoretical FLOP/s on A100 for long sequences.

---

## Exercise 6.4 — INT8 GEMV with `dp4a`

**Concept introduced in**: L35 (Quantization)

### Problem Statement

Implement an INT8 matrix–vector multiply (GEMV: `y = A x`) using the `__dp4a` instruction
(dot product of 4 int8 values), which computes `a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]`
in a single hardware instruction. After the int32 GEMV, dequantize using per-row scale factors.

Verify the maximum element-wise error between the quantized+dequantized output and the FP32
reference is < 2.0 (expected for INT8 quantization noise).

### Requirements

- Matrix A: M=1024 rows × K=1024 cols, quantized to int8 (range -127 to 127).
- Vector x: K=1024 elements, quantized to int8.
- Dequantize output with `scale_A * scale_x`.
- Block size: 256 threads; each thread computes one row of y.

### Starter Code

```cuda
// ex6_4_dp4a_gemv.cu
// Compile: nvcc -O2 -arch=sm_61 -o ex6_4 ex6_4_dp4a_gemv.cu
// Note: __dp4a requires compute capability >= 6.1

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstdint>

#define M_DIM    1024
#define K_DIM    1024
#define BLOCK_SIZE 256

// INT8 GEMV: y[i] = sum_j A[i*K + j] * x[j]  (all int8, result in int32)
// Then dequantize: y_fp32[i] = y_int32[i] * scale
__global__ void int8_gemv(const int8_t* __restrict__  A,
                           const int8_t* __restrict__  x,
                           int32_t*      __restrict__  y,
                           int M, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    // K must be divisible by 4 for dp4a
    int32_t sum = 0;
    for (int k = 0; k < K; k += 4) {
        // Pack 4 bytes from A and x into int32 for dp4a
        // __dp4a(int a, int b, int c) = c + a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
        int a_packed, x_packed;
        // TODO: memcpy(&a_packed, &A[row * K + k], 4)  // load 4 int8 as one int32
        // TODO: memcpy(&x_packed, &x[k], 4)
        // TODO: sum = __dp4a(a_packed, x_packed, sum)
    }
    y[row] = sum;
}

// Dequantize: scale int32 results to float
__global__ void dequantize(const int32_t* y_int, float* y_fp, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: if (i < n) y_fp[i] = y_int[i] * scale;
}

int main() {
    const int M = M_DIM, K = K_DIM;

    // Quantize float matrices to INT8 using symmetric quantization
    float scale_A = 0.01f, scale_x = 0.01f;

    int8_t* h_A   = new int8_t[M * K];
    int8_t* h_x   = new int8_t[K];
    float*  h_y_fp_ref = new float[M]();
    float*  h_y_fp_gpu = new float[M]();

    // Fill with quantized random values
    srand(42);
    for (int i = 0; i < M * K; ++i) h_A[i] = (int8_t)(rand() % 255 - 127);
    for (int i = 0; i < K;     ++i) h_x[i] = (int8_t)(rand() % 255 - 127);

    // CPU FP32 reference (dequantized on the fly)
    for (int i = 0; i < M; ++i) {
        int32_t acc = 0;
        for (int j = 0; j < K; ++j) acc += (int32_t)h_A[i * K + j] * h_x[j];
        h_y_fp_ref[i] = acc * scale_A * scale_x;
    }

    int8_t  *d_A, *d_x;
    int32_t *d_y_int;
    float   *d_y_fp;
    // TODO: cudaMalloc d_A (M*K), d_x (K), d_y_int (M), d_y_fp (M)
    // TODO: cudaMemcpy h_A -> d_A, h_x -> d_x

    int nblocks = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int8_gemv<<<nblocks, BLOCK_SIZE>>>(d_A, d_x, d_y_int, M, K);
    dequantize<<<nblocks, BLOCK_SIZE>>>(d_y_int, d_y_fp, scale_A * scale_x, M);
    cudaDeviceSynchronize();

    // TODO: cudaMemcpy d_y_fp -> h_y_fp_gpu

    float max_err = 0.0f;
    for (int i = 0; i < M; ++i)
        max_err = fmaxf(max_err, fabsf(h_y_fp_gpu[i] - h_y_fp_ref[i]));
    printf("Max element-wise error: %.4f\n", max_err);
    printf("Result: %s\n", (max_err < 2.0f) ? "PASS" : "FAIL");

    // TODO: cudaFree x4
    delete[] h_A; delete[] h_x; delete[] h_y_fp_ref; delete[] h_y_fp_gpu;
    return 0;
}
```

### Expected Output

```
Max element-wise error: 0.0000
Result: PASS
```

(Error is 0 because the dp4a result is exact in int32; floating point rounding only occurs in dequantize.)

### Hints

- `__dp4a(int a, int b, int c)` interprets `a` and `b` as four `int8` values packed into `int32`, computes the dot product, and adds `c`.
- Use `memcpy` (not pointer cast) to pack 4 bytes into an `int32` to avoid undefined behavior from aliasing.
- For K not divisible by 4, pad A and x with zeros.
- In production, `cublasGemmEx` with `CUDA_R_8I` handles INT8 GEMM with tensor cores.

### Performance Target

INT8 GEMV should be 2–4× faster than FP32 GEMV due to higher throughput (4× more int8 ops per CUDA core cycle with `dp4a`). Measure with CUDA events.

---

## Exercise 6.5 — Fused Bias + ReLU

**Concept introduced in**: L36 (Kernel Fusion)

### Problem Statement

Fuse the bias addition and ReLU activation into a single kernel. Compare against two
separate kernels (a `bias_add` kernel followed by a `relu` kernel). Measure:

1. **Two-pass** (separate): bias_add then relu, each reading and writing the full tensor.
2. **Fused**: single kernel, reads once, writes once.

Verify output matches, and measure bandwidth and speedup.

### Requirements

- Tensor: M=4096 rows × N=4096 cols (FP32).
- Bias vector: length N (one bias per column).
- Verify max absolute error between two-pass and fused < 1e-6.

### Starter Code

```cuda
// ex6_5_fused_bias_relu.cu
// Compile: nvcc -O2 -arch=sm_80 -o ex6_5 ex6_5_fused_bias_relu.cu

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#define M_ROWS 4096
#define N_COLS 4096
#define BLOCK_SIZE 256
#define NRUNS 20

// Separate pass 1: out[i][j] = in[i][j] + bias[j]
__global__ void bias_add(const float* in, const float* bias, float* out, int m, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < m * n) {
        int col = gid % n;
        out[gid] = in[gid] + bias[col];
    }
}

// Separate pass 2: out[i] = max(0, in[i])
__global__ void relu(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: if (i < n) out[i] = fmaxf(0.0f, in[i]);
}

// Fused: out[i][j] = max(0, in[i][j] + bias[j])  — single kernel
__global__ void fused_bias_relu(const float* in, const float* bias, float* out, int m, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: if (gid < m * n): col = gid % n; out[gid] = fmaxf(0.0f, in[gid] + bias[col]);
}

int main() {
    const int m = M_ROWS, n = N_COLS;
    const int total = m * n;
    const size_t bytes_full = total * sizeof(float);
    const size_t bytes_bias = n     * sizeof(float);

    float* h_in   = new float[total];
    float* h_bias = new float[n];
    float* h_out_2pass = new float[total];
    float* h_out_fused = new float[total];

    for (int i = 0; i < total; ++i) h_in[i]   = (float)rand()/RAND_MAX * 2.0f - 1.0f;
    for (int i = 0; i < n;     ++i) h_bias[i]  = (float)rand()/RAND_MAX * 0.5f;

    float *d_in, *d_bias, *d_tmp, *d_out_2p, *d_out_fused;
    // TODO: cudaMalloc all 5 buffers
    // TODO: cudaMemcpy h_in -> d_in, h_bias -> d_bias

    int nblocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);

    // --- Two-pass version ---
    // Warm-up
    bias_add<<<nblocks, BLOCK_SIZE>>>(d_in, d_bias, d_tmp,      m, n);
    relu    <<<nblocks, BLOCK_SIZE>>>(d_tmp,         d_out_2p,  total);

    cudaEventRecord(s);
    for (int r = 0; r < NRUNS; ++r) {
        bias_add<<<nblocks, BLOCK_SIZE>>>(d_in, d_bias, d_tmp,    m, n);
        relu    <<<nblocks, BLOCK_SIZE>>>(d_tmp,         d_out_2p, total);
    }
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms_2p; cudaEventElapsedTime(&ms_2p, s, e);
    ms_2p /= NRUNS;

    // --- Fused version ---
    fused_bias_relu<<<nblocks, BLOCK_SIZE>>>(d_in, d_bias, d_out_fused, m, n);  // warm-up
    cudaEventRecord(s);
    for (int r = 0; r < NRUNS; ++r)
        fused_bias_relu<<<nblocks, BLOCK_SIZE>>>(d_in, d_bias, d_out_fused, m, n);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms_fused; cudaEventElapsedTime(&ms_fused, s, e);
    ms_fused /= NRUNS;

    // Measure bandwidth
    // Two-pass: 3 reads (in, bias, tmp) + 2 writes (tmp, out) = 5 tensor-sized ops
    double bw_2p    = (3.0 * bytes_full + 2.0 * bytes_full) / (ms_2p    * 1e-3) / 1e9;
    // Fused:    2 reads (in, bias) + 1 write (out) = 3 ops
    double bw_fused = (2.0 * bytes_full + 1.0 * bytes_full) / (ms_fused * 1e-3) / 1e9;

    printf("Two-pass: %.3f ms  %.1f GB/s\n", ms_2p, bw_2p);
    printf("Fused:    %.3f ms  %.1f GB/s\n", ms_fused, bw_fused);
    printf("Speedup: %.2fx\n", ms_2p / ms_fused);

    // Verify
    // TODO: cudaMemcpy d_out_2p -> h_out_2pass, d_out_fused -> h_out_fused
    float max_err = 0.0f;
    for (int i = 0; i < total; ++i)
        max_err = fmaxf(max_err, fabsf(h_out_2pass[i] - h_out_fused[i]));
    printf("Max error: %.2e\n", max_err);
    printf("Result: %s\n", (max_err < 1e-6f) ? "PASS" : "FAIL");

    // TODO: cudaFree all
    delete[] h_in; delete[] h_bias; delete[] h_out_2pass; delete[] h_out_fused;
    cudaEventDestroy(s); cudaEventDestroy(e);
    return 0;
}
```

### Expected Output (A100 example)

```
Two-pass: 1.84 ms  225.6 GB/s
Fused:    0.91 ms  457.1 GB/s
Speedup: 2.02x
Result: PASS
```

### Hints

- The two-pass version writes `d_tmp` to DRAM then reads it back for `relu` — that's 2 extra DRAM accesses per element.
- The fused kernel replaces the intermediate write/read with a register variable — the compiler keeps the intermediate value in register file.
- Bias broadcast (`bias[col]`) causes all threads in a row to read the same `n` distinct values — use `__ldg` or `__restrict__` to hint the L1 cache.
- In real deep learning frameworks, `bias + activation` fusions are handled by cuDNN or torch.compile.

### Performance Target

Fused version should be approximately 2× faster due to halving DRAM traffic. Both versions are memory-bandwidth bound.
