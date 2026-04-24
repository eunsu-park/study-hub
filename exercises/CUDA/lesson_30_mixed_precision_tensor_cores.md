# Lesson 30 — Mixed Precision and Tensor Cores (per-lesson exercise)

Prerequisites: L04 (memory model), L09 (occupancy), familiarity with `half`/`__half` types.

Compile: `nvcc -O3 -arch=sm_80 ex.cu -o ex`

Tensor Cores are specialized multiply-accumulate units on Volta (V100) and later NVIDIA GPUs. They compute a 16×16×16 matrix-multiply-accumulate in a single instruction cycle — up to 8× the throughput of FP32 on FP16 inputs.

Using them requires either:
- The `wmma::` C++ API (flexible, verbose)
- A library that calls tensor cores internally (cuBLAS, cuDNN, CUTLASS)

This exercise uses the `wmma::` API to expose the mechanics.

---

## Exercise 30.1 — Single Warp MMA

**Difficulty**: ★★★

### Problem

A single warp (32 threads) performs a 16×16×16 fragment matrix-multiply-accumulate:

$$D = A \cdot B + C$$

Where $A$ is 16×16 FP16, $B$ is 16×16 FP16, and $C, D$ are 16×16 FP32 (mixed precision).

### Starter

```cuda
#include <cstdio>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

__global__ void wmma_single_fragment(const __half *A, const __half *B, float *C) {
    // Each warp handles one 16x16 output tile
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float>                 c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    wmma::load_matrix_sync(a_frag, A, 16);   // leading dimension = 16
    wmma::load_matrix_sync(b_frag, B, 16);

    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    wmma::store_matrix_sync(C, c_frag, 16, wmma::mem_row_major);
}

int main(void) {
    __half *dA, *dB;
    float *dC;
    __half *hA = new __half[16*16];
    __half *hB = new __half[16*16];
    float  *hC = new float [16*16];
    for (int i = 0; i < 16 * 16; i++) { hA[i] = __float2half(1.0f); hB[i] = __float2half(1.0f); }

    cudaMalloc(&dA, 16*16*sizeof(__half));
    cudaMalloc(&dB, 16*16*sizeof(__half));
    cudaMalloc(&dC, 16*16*sizeof(float));
    cudaMemcpy(dA, hA, 16*16*sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, 16*16*sizeof(__half), cudaMemcpyHostToDevice);

    wmma_single_fragment<<<1, 32>>>(dA, dB, dC);   // one warp only
    cudaMemcpy(hC, dC, 16*16*sizeof(float), cudaMemcpyDeviceToHost);

    // Every C[i][j] should equal 16 (16 multiplications of 1*1 summed)
    printf("C[0][0] = %.1f (expected 16.0)\n", hC[0]);

    delete[] hA; delete[] hB; delete[] hC;
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
```

---

## Exercise 30.2 — Tiled GEMM Using Tensor Cores

**Difficulty**: ★★★★

Extend 30.1 to a full GEMM for $M \times K$ times $K \times N$ matrices where $M, K, N$ are multiples of 16. Launch a 2D grid of warps; each warp accumulates one 16×16 output tile by iterating over $K/16$ fragments along the reduction dimension.

Compare against cuBLAS `cublasGemmEx` with `CUDA_R_16F` inputs and `CUDA_R_32F` accumulator. cuBLAS will be faster (hand-written WMMA rarely matches it), but your version should hit within 2× on a square 2048×2048 problem — the gap highlights how much extra tuning cuBLAS does (register tiling, async copies, split-k).

---

## Exercise 30.3 — BF16 vs FP16 — Bonus

**Difficulty**: ★★

Repeat 30.1 using `__nv_bfloat16` inputs instead of `__half`. BF16 has the same 8-bit exponent as FP32, avoiding the FP16 overflow issues during training. Verify the output of a matrix with entries around 1e3 — FP16 overflows, BF16 does not.
