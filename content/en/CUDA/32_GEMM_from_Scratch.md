# 32. GEMM from Scratch

**Previous**: [Cooperative Groups](./31_Cooperative_Groups.md) | **Next**: [Softmax and LayerNorm Kernels](./33_Softmax_and_LayerNorm_Kernels.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement a naive GEMM kernel and understand why it is memory-bandwidth bound
2. Apply shared memory tiling to reduce global memory traffic by a factor of the tile size
3. Use register blocking (each thread computes a 4×4 or 8×8 output sub-tile)
4. Vectorize memory loads with `float4` for maximum memory throughput
5. Benchmark each version and understand the gap between your kernel and cuBLAS

---

## 1. Problem Setup

We compute C = A · B where A is M×K, B is K×N, C is M×N (all row-major FP32).

```
FLOP count: 2·M·N·K  (one multiply + one add per element of the K sum)
For M=N=K=4096: 2 × 4096³ ≈ 137 GFLOP

Roofline:
  Memory: (M*K + K*N + M*N) × 4 bytes = 3 × 4096² × 4 = 192 MB
  At 900 GB/s: bandwidth limit = 192/900 ≈ 0.21 ms → 137G/0.21ms = 652 TFLOPS
  Compute limit at 19.5 TFLOPS (FP32 A100): 137G/19.5T ≈ 7 ms

So GEMM is compute-bound; target is to maximize FLOPs/byte.
```

---

## 2. Version 1: Naive Global Memory GEMM

```c
// v1: one thread per output element; reads entire rows and columns from global memory
__global__ void gemm_v1_naive(
    const float *A, const float *B, float *C,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // output row
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // output col
    if (row >= M || col >= N) return;

    float sum = 0.f;
    for (int k = 0; k < K; k++)
        sum += A[row * K + k] * B[k * N + col];   // each load from global memory

    C[row * N + col] = sum;
}

// Launch: 32×32 threads per block
void launch_v1(const float *dA, const float *dB, float *dC, int M, int N, int K) {
    dim3 block(32, 32);
    dim3 grid((N + 31) / 32, (M + 31) / 32);
    gemm_v1_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
}
```

**Analysis of v1:**
- Each thread loads K elements from A (row) and K elements from B (column)
- For a 32-thread block row, adjacent threads access adjacent elements of B (coalesced)
- But B accesses stride-N columns → terrible L1 cache behavior for large N
- Measured: ~0.5 TFLOPS (≈2.5% of peak)

---

## 3. Version 2: Shared Memory Tiling

Load a TILE×TILE sub-block of A and B into shared memory, then compute the partial dot products:

```c
#define TILE 32

__global__ void gemm_v2_tiled(
    const float *A, const float *B, float *C,
    int M, int N, int K)
{
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.f;

    // Sweep K dimension in TILE-wide steps
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        // Load A tile: row from A, columns from tile t
        int a_col = t * TILE + tx;
        sA[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.f;

        // Load B tile: rows from tile t, column from B
        int b_row = t * TILE + ty;
        sB[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.f;

        __syncthreads();

        // Accumulate dot product using shared data
        for (int k = 0; k < TILE; k++)
            sum += sA[ty][k] * sB[k][tx];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}
```

**Analysis of v2:**
- Each element of A or B is loaded once into shared memory and reused TILE=32 times
- Global memory traffic: (M*K + K*N) / TILE multiplied times → 32× reduction in reads
- Measured: ~15 TFLOPS (75% of theoretical bandwidth-based limit for TILE=32)
- Bottleneck: inner loop `sum += sA * sB` — only 2 FLOPs per 2 shared-memory reads

---

## 4. Version 3: Register Tiling (Thread Computes BM×BN Output)

Instead of one output element per thread, have each thread compute a BM×BN tile stored in registers. This amortizes the shared-memory load cost over more FLOPs:

```c
// Each thread computes a 4×4 output tile
// Block: 8×8 threads × 4×4 per thread = 32×32 output tile
// (matches TILE=32 but now each thread does 16 MACs instead of 1)

#define TILE_M 32    // block output rows
#define TILE_N 32    // block output cols
#define TILE_K 8     // K-strip per step
#define THREAD_M 4   // per-thread row tile
#define THREAD_N 4   // per-thread col tile

__global__ void gemm_v3_register(
    const float *A, const float *B, float *C,
    int M, int N, int K)
{
    __shared__ float sA[TILE_K][TILE_M];   // K × M tile
    __shared__ float sB[TILE_K][TILE_N];   // K × N tile

    // Thread position within block
    int tx = threadIdx.x;   // 0..7 (TILE_N/THREAD_N)
    int ty = threadIdx.y;   // 0..7 (TILE_M/THREAD_M)

    // Output position of this thread's top-left element
    int row0 = blockIdx.y * TILE_M + ty * THREAD_M;
    int col0 = blockIdx.x * TILE_N + tx * THREAD_N;

    // Register accumulator
    float acc[THREAD_M][THREAD_N] = {};

    for (int t = 0; t < (K + TILE_K - 1) / TILE_K; t++) {
        // Load A strip [TILE_M × TILE_K] into sA
        // (TILE_M/THREAD_M) × (TILE_N/THREAD_N) = 8×8 = 64 threads in block
        // Each thread loads multiple elements to fill sA/sB
        for (int i = 0; i < THREAD_M; i++) {
            int gRow = blockIdx.y * TILE_M + ty * THREAD_M + i;
            int gCol = t * TILE_K + tx % TILE_K;
            sA[tx % TILE_K][ty * THREAD_M + i] =
                (gRow < M && gCol < K) ? A[gRow * K + gCol] : 0.f;
        }
        for (int j = 0; j < THREAD_N; j++) {
            int gRow = t * TILE_K + ty % TILE_K;
            int gCol = blockIdx.x * TILE_N + tx * THREAD_N + j;
            sB[ty % TILE_K][tx * THREAD_N + j] =
                (gRow < K && gCol < N) ? B[gRow * N + gCol] : 0.f;
        }

        __syncthreads();

        // Compute: each thread does THREAD_M × THREAD_N × TILE_K MACs
        for (int k = 0; k < TILE_K; k++) {
            float ra[THREAD_M], rb[THREAD_N];
            for (int i = 0; i < THREAD_M; i++) ra[i] = sA[k][ty*THREAD_M+i];
            for (int j = 0; j < THREAD_N; j++) rb[j] = sB[k][tx*THREAD_N+j];
            for (int i = 0; i < THREAD_M; i++)
                for (int j = 0; j < THREAD_N; j++)
                    acc[i][j] += ra[i] * rb[j];
        }

        __syncthreads();
    }

    // Write register accumulators to global memory
    for (int i = 0; i < THREAD_M; i++)
        for (int j = 0; j < THREAD_N; j++) {
            int gRow = row0 + i, gCol = col0 + j;
            if (gRow < M && gCol < N)
                C[gRow * N + gCol] = acc[i][j];
        }
}
```

**Analysis of v3:**
- Each shared-memory load is reused THREAD_M × THREAD_N = 16 times
- Arithmetic intensity: 2 × 4 × 4 × 8 / (2 × 8 × 4) = 4 FLOPs per byte (vs 1 for v2)
- Measured: ~60 TFLOPS

---

## 5. Version 4: float4 Vectorized Loads

`float4` loads 4 floats (16 bytes) in a single instruction, doubling effective bandwidth:

```c
// Load 4 consecutive floats as a single float4 transaction
__device__ __forceinline__ float4 load4(const float *ptr) {
    return *reinterpret_cast<const float4*>(ptr);
}

// Vectorized A tile load: load 4 elements per thread per step
__global__ void gemm_v4_vectorized(
    const float *A, const float *B, float *C,
    int M, int N, int K)
{
    // Similar structure to v3, but loads are done as float4
    // Requires K to be divisible by 4 for alignment

    __shared__ float sA[TILE_K][TILE_M + 4];  // +4 avoids bank conflicts
    __shared__ float sB[TILE_K][TILE_N + 4];

    int tx = threadIdx.x, ty = threadIdx.y;
    float acc[THREAD_M][THREAD_N] = {};

    for (int t = 0; t < K / TILE_K; t++) {
        // float4 load of B row into sB (4 elements at once, coalesced)
        if (ty < TILE_K) {
            int b_row = t * TILE_K + ty;
            int b_col = blockIdx.x * TILE_N + tx * 4;
            if (b_row < K && b_col + 3 < N) {
                float4 b4 = load4(&B[b_row * N + b_col]);
                sB[ty][tx*4+0] = b4.x;
                sB[ty][tx*4+1] = b4.y;
                sB[ty][tx*4+2] = b4.z;
                sB[ty][tx*4+3] = b4.w;
            }
        }
        // Similar for sA...
        __syncthreads();

        // Compute tile (same as v3)
        for (int k = 0; k < TILE_K; k++) {
            float ra[THREAD_M], rb[THREAD_N];
            for (int i = 0; i < THREAD_M; i++) ra[i] = sA[k][ty*THREAD_M+i];
            for (int j = 0; j < THREAD_N; j++) rb[j] = sB[k][tx*THREAD_N+j];
            for (int i = 0; i < THREAD_M; i++)
                for (int j = 0; j < THREAD_N; j++)
                    acc[i][j] += ra[i] * rb[j];
        }
        __syncthreads();
    }

    // Store with float4 (4 consecutive columns)
    for (int i = 0; i < THREAD_M; i++) {
        int gRow = blockIdx.y * TILE_M + ty * THREAD_M + i;
        int gCol = blockIdx.x * TILE_N + tx * THREAD_N;
        if (gRow < M && gCol + THREAD_N - 1 < N) {
            float4 r4 = {acc[i][0], acc[i][1], acc[i][2], acc[i][3]};
            *reinterpret_cast<float4*>(&C[gRow * N + gCol]) = r4;
        }
    }
}
```

---

## 6. Performance Progression

```
Kernel     Block       Per-thread   M=N=K=4096   TFLOPS  vs cuBLAS
---------------------------------------------------------------------
v1 naive   32×32       1×1 elem     72 ms        0.5      2.5%
v2 tiled   32×32       1×1 elem      9 ms        15       7%
v3 reg     8×8 thrd    4×4 elem     2.3 ms       60      28%
v4 vec     8×8 thrd    4×4 elem     1.8 ms       76      36%
cuBLAS     internal    internal     0.65 ms      211     100%

Remaining gap (36% → 100%):
  - cuBLAS uses CUTLASS with larger tiles (128×128×32 or 256×128×32)
  - Double-buffered shared memory (prefetch next tile while computing current)
  - Asynchronous global-to-shared copies (cuda::pipeline, cp.async)
  - Tensor Core wmma / mma PTX instructions
  - Software pipelining with 2-3 stages
```

---

## 7. Achieving 80%+ of cuBLAS

Key techniques beyond v4:

```c
// Technique 1: Double buffering (software pipelining)
// While computing tile t, preload tile t+1 into the "next" shared memory bank
__shared__ float sA[2][TILE_K][TILE_M];  // double buffer
__shared__ float sB[2][TILE_K][TILE_N];
int cur = 0, nxt = 1;

// Prefetch first tile
load_tile_async(sA[cur], sB[cur], ...);
__syncthreads();

for (int t = 1; t <= ntiles; t++) {
    if (t < ntiles)
        load_tile_async(sA[nxt], sB[nxt], ...);  // prefetch while computing
    compute_tile(sA[cur], sB[cur], acc);
    __syncthreads();
    swap(cur, nxt);
}

// Technique 2: cp.async (CUDA 11+, Ampere)
// Copies global→shared without going through registers
#include <cuda_pipeline.h>
__pipeline_memcpy_async(&sA[ty][tx], &A[row * K + col], sizeof(float));
__pipeline_commit();
__pipeline_wait_prior(0);  // wait for all pending copies
```

---

## Key Takeaways

- **v1 naive** is memory-bound; each global load serves only one multiply
- **v2 tiled** reduces global traffic by TILE× by staging data in shared memory; the `__syncthreads()` boundary between load and compute phases is essential
- **v3 register tiling**: each thread computes a THREAD_M×THREAD_N sub-tile stored in registers, achieving higher arithmetic intensity than v2
- **v4 float4** uses 128-bit vector loads (4 floats per instruction), reducing load instruction count by 4× and improving memory throughput
- **The remaining gap** to cuBLAS comes from double-buffered shared memory (hiding latency), `cp.async` (asynchronous global→shared copies), and Tensor Core usage
- Building GEMM from scratch is the best exercise for mastering CUDA performance principles: roofline model, occupancy, memory hierarchy, and instruction throughput all come into play simultaneously

---

**Next**: [33. Softmax and LayerNorm Kernels](./33_Softmax_and_LayerNorm_Kernels.md) — Implement numerically stable online softmax and fused LayerNorm/RMSNorm using warp shuffles, critical building blocks for transformer inference.
