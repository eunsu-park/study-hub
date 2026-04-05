# 30. Mixed Precision and Tensor Cores

**Previous**: [cuBLAS and cuSPARSE](./29_cuBLAS_and_cuSPARSE.md) | **Next**: [Cooperative Groups](./31_Cooperative_Groups.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the numerical properties of FP32, FP16, BF16, and FP8 and their trade-offs
2. Use the WMMA API (`nvcuda::wmma`) to write CUDA kernels that directly use Tensor Cores
3. Implement a 16×16×16 WMMA matrix multiply-accumulate operation
4. Apply loss scaling to prevent FP16 gradient underflow during training
5. Measure Tensor Core FLOPS vs CUDA core FLOPS and understand when each is dominant

---

## 1. Floating-Point Formats

```
Format     Bits   Exponent  Mantissa  Dynamic range      Notes
----------------------------------------------------------------------
FP64       64      11        52        ±10^±308           CPU default
FP32       32       8        23        ±10^±38            GPU default
FP16       16       5        10        ±65504             IEEE 754 half
BF16       16       8         7        ±10^±38            "Brain Float" (Google TPU)
FP8 E4M3   8        4         3        ±448               CUDA 12+ (Hopper)
FP8 E5M2   8        5         2        ±57344             looser range, for gradients
TF32       19      (subset of FP32)   ±10^±38            A100 internal Tensor Core format

Key trade-offs:
  FP16: good precision, limited range → needs loss scaling for gradients
  BF16: same range as FP32, lower precision → drop-in replacement for FP32 range
  FP8:  2× FP16 throughput on Hopper; requires careful quantization
```

---

## 2. FP16 Data Types in CUDA

```c
#include <cuda_fp16.h>

// half: 16-bit float type
__global__ void half_demo() {
    half a = __float2half(3.14f);    // FP32 → FP16
    half b = __float2half(2.71f);
    half c = __hadd(a, b);           // FP16 add
    half d = __hmul(a, b);           // FP16 multiply
    float f = __half2float(c);       // FP16 → FP32

    // half2: two FP16 values packed in 32 bits (SIMD 2× throughput)
    half2 v = __float22half2_rn(make_float2(1.f, 2.f));
    half2 w = __float22half2_rn(make_float2(3.f, 4.f));
    half2 r = __hadd2(v, w);        // packed 2× FP16 add
}

// BF16 requires cuda_bf16.h (CUDA 11.0+)
#include <cuda_bf16.h>
__global__ void bf16_demo() {
    __nv_bfloat16 a = __float2bfloat16(3.14f);
    __nv_bfloat16 b = __float2bfloat16(2.71f);
    __nv_bfloat16 c = __hadd(a, b);  // same intrinsics as FP16
    float f = __bfloat162float(c);
}
```

---

## 3. Tensor Core Overview

Tensor Cores are specialized matrix-multiply-accumulate (MMA) units:

```
A100 Tensor Core performance (per SM, per clock):
  FP16 TC:   256 FLOPs
  BF16 TC:   256 FLOPs
  TF32 TC:   128 FLOPs
  FP64 TC:    64 FLOPs
  INT8 TC:   512 OPs

vs CUDA Core:
  FP32 CUDA:   2 FLOPs (1 FMA)
  FP16 CUDA:   2 FLOPs

So Tensor Cores are 128-256× more efficient per unit per clock.

WMMA fragment sizes (must match Tensor Core hardware):
  16×16×16  (FP16/BF16 accumulate FP32 or FP16)
  8×16×16   (alternative for some GPUs)
  32×8×16   (alternative)

Alignment requirement: matrices must be 16-element aligned for coalesced loads.
```

---

## 4. WMMA API

The WMMA (Warp Matrix Multiply Accumulate) API exposes Tensor Cores at the warp level. One warp (32 threads) cooperatively holds a 16×16 matrix fragment:

```c
#include <mma.h>
using namespace nvcuda;

// Fragment types for 16×16×16 FP16 → FP32 WMMA
// Each warp cooperatively stores one fragment (distributed across 32 threads)
using frag_a   = wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>;
using frag_b   = wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major>;
using frag_acc = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

// WMMA GEMM: C[16×16] += A[16×16] * B[16×16]
// One warp per output tile
__global__ void wmma_gemm_16x16(
    const half *A, const half *B, float *C,
    int M, int N, int K)
{
    // Warp position in output
    int warp_row = (blockIdx.y * blockDim.y + threadIdx.y) / 32 * 16;
    int warp_col = (blockIdx.x * blockDim.x + threadIdx.x) / 32 * 16;

    if (warp_row >= M || warp_col >= N) return;

    frag_a   a_frag;
    frag_b   b_frag;
    frag_acc c_frag;

    // Initialize accumulator to zero
    wmma::fill_fragment(c_frag, 0.f);

    // Loop over K in 16-wide tiles
    for (int k = 0; k < K; k += 16) {
        // Load A tile: pointer to row warp_row, col k
        wmma::load_matrix_sync(a_frag, A + warp_row * K + k, K);

        // Load B tile: pointer to row k, col warp_col (B in col-major)
        wmma::load_matrix_sync(b_frag, B + k * N + warp_col, N);

        // Matrix multiply-accumulate: c_frag += a_frag * b_frag
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store result
    wmma::store_matrix_sync(C + warp_row * N + warp_col, c_frag, N,
                             wmma::mem_row_major);
}
```

---

## 5. WMMA with Shared Memory (Production Pattern)

Raw global-memory WMMA is bandwidth-limited. Real implementations tile into shared memory:

```c
// 128×128 block tile, processed as 8×8 warp tiles of 16×16 each
// This is the structure used by cuBLAS internally

#define BM 128   // block M tile
#define BN 128   // block N tile
#define BK 16    // block K tile (Tensor Core inner dim)

__global__ void wmma_tiled(const half *A, const half *B, float *C,
                            int M, int N, int K) {
    __shared__ half sA[BM][BK];
    __shared__ half sB[BK][BN];

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    // Warp grid within block: 4×4 arrangement (16 warps per block)
    int warp_row = warp_id / 4;  // 0..3 (each handles 32 rows)
    int warp_col = warp_id % 4;  // 0..3 (each handles 32 cols)

    frag_acc c_frag[2][2];  // 2×2 16×16 accumulators per warp
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            wmma::fill_fragment(c_frag[i][j], 0.f);

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    for (int k = 0; k < K; k += BK) {
        // Cooperative load of A[BM×BK] and B[BK×BN] into shared memory
        // (each thread loads several elements; details omitted for clarity)
        load_tile_to_shared(A, sA, block_row, k, M, K);
        load_tile_to_shared_B(B, sB, k, block_col, K, N);
        __syncthreads();

        // Each warp computes 2×2 WMMA tiles
        for (int wi = 0; wi < 2; wi++) {
            for (int wj = 0; wj < 2; wj++) {
                frag_a a_frag; frag_b b_frag;
                int row_off = (warp_row * 2 + wi) * 16;
                int col_off = (warp_col * 2 + wj) * 16;
                wmma::load_matrix_sync(a_frag, &sA[row_off][0], BK);
                wmma::load_matrix_sync(b_frag, &sB[0][col_off], BN);
                wmma::mma_sync(c_frag[wi][wj], a_frag, b_frag, c_frag[wi][wj]);
            }
        }
        __syncthreads();
    }

    // Store accumulators
    for (int wi = 0; wi < 2; wi++) {
        for (int wj = 0; wj < 2; wj++) {
            int row = block_row + (warp_row*2+wi)*16;
            int col = block_col + (warp_col*2+wj)*16;
            if (row < M && col < N)
                wmma::store_matrix_sync(C + row*N + col, c_frag[wi][wj], N,
                                        wmma::mem_row_major);
        }
    }
}
```

---

## 6. Loss Scaling for FP16 Training

FP16 gradients can underflow to zero for values < ~6×10^-5 (FP16 minimum normal). Loss scaling multiplies the loss by a large constant before backprop, then divides the gradients:

```c
// Dynamic loss scaling (PyTorch AMP approach)
float loss_scale = 65536.f;   // initial scale factor
int growth_interval = 2000;   // steps between scale increases
int n_skipped = 0;

for (int step = 0; step < total_steps; step++) {
    // Forward pass (FP32 accumulation in mixed-precision)
    // Backward pass with scaled loss:
    //   scaled_loss = loss * loss_scale
    //   gradients  *= loss_scale

    // Check for inf/nan in gradients
    bool has_inf = check_gradients_for_inf(d_grads, param_count);

    if (has_inf) {
        loss_scale /= 2.f;   // reduce scale
        n_skipped++;
        printf("Step %d: skip (overflow), scale → %.0f\n", step, loss_scale);
        continue;  // skip parameter update
    }

    // Unscale gradients
    scale_gradients<<<GRID, BLOCK>>>(d_grads, 1.f / loss_scale, param_count);

    // Optimizer step...

    // Periodically increase scale
    if ((step + 1) % growth_interval == 0 && n_skipped == 0) {
        loss_scale = fminf(loss_scale * 2.f, 65536.f);
    }
    n_skipped = 0;
}

// Inf/NaN check kernel
__global__ void check_inf_kernel(const float *grad, int *flag, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && !isfinite(grad[i]))
        atomicExch(flag, 1);   // signal overflow
}
```

---

## 7. Measuring Tensor Core FLOPS

```c
// Profile with nvcc -arch=sm_80, then use ncu to compare:
// ncu --metrics sm__ops_warps_eligible.avg,
//               sm__inst_executed_pipe_tensor.avg
//               ./my_gemm

// Quick host-side measurement
void measure_flops(int M, int N, int K, int iters) {
    // allocate d_A(M×K), d_B(K×N), d_C(M×N) as half/float...

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i = 0; i < iters; i++)
        wmma_gemm_16x16<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    double flops = 2.0 * M * N * K * iters;  // 2 FLOPs per FMA
    double tflops = flops / (ms * 1e9);
    printf("WMMA: %.1f TFLOPS (%.2f ms for %d iterations)\n", tflops, ms, iters);

    // Compare: CUDA core FP32 GEMM
    // A100: 312 TFLOPS (TC FP16), 19.5 TFLOPS (CUDA FP32)
    // Ratio ≈ 16× for FP16 Tensor Core vs FP32 CUDA cores
}
```

---

## Key Takeaways

- **FP16** has 5-bit exponent (range ±65504) and 10-bit mantissa; **BF16** has 8-bit exponent (same range as FP32) and 7-bit mantissa — BF16 is preferred for training stability
- **WMMA API**: one warp (32 threads) cooperatively holds a 16×16 fragment; `load_matrix_sync` → `mma_sync` → `store_matrix_sync` is the full pattern
- **Fragment layout** across the 32 warp threads is hardware-defined and opaque; do not rely on specific thread-to-element mapping
- **Tiled WMMA** loads A and B sub-tiles into shared memory before calling `mma_sync`; this is critical to reach near-peak Tensor Core throughput
- **Loss scaling** multiplies the loss by a large constant (e.g., 2^16) to shift small gradients above FP16's minimum; dynamic loss scaling automatically adjusts the factor
- On A100: FP16 Tensor Cores deliver ~312 TFLOPS vs ~19.5 TFLOPS for FP32 CUDA cores — a 16× theoretical gap, with ~8-12× measured in practice

---

**Next**: [31. Cooperative Groups](./31_Cooperative_Groups.md) — Use CUDA Cooperative Groups to write flexible thread coordination code that works at warp, block, and grid scope without hardcoded `__syncthreads()`.
