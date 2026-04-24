# Lesson 29 — cuBLAS and cuSPARSE (per-lesson exercise)

Prerequisites: L32 (GEMM from scratch), basic linear algebra.

Compile: `nvcc -O3 -arch=sm_80 ex.cu -lcublas -lcusparse -o ex`

cuBLAS is NVIDIA's hand-tuned linear-algebra library. It implements BLAS routines (GEMM, GEMV, dot products) that hit 80–95% of theoretical peak on every recent NVIDIA GPU. cuSPARSE does the same for sparse matrices.

For dense LA, **always start with cuBLAS**. Hand-rolled kernels rarely match it; they should only exist when a library does not cover the case (custom dtypes, fused operations).

---

## Exercise 29.1 — cuBLAS Sgemm

**Difficulty**: ★

### Problem

Multiply two `4096 × 4096` `float` matrices. Time against your tiled hand-roll from CUDA L32.

```cuda
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

int main(void) {
    const int N = 4096;
    size_t bytes = N * N * sizeof(float);

    float *dA, *dB, *dC;
    cudaMalloc(&dA, bytes); cudaMalloc(&dB, bytes); cudaMalloc(&dC, bytes);
    /* fill A and B with some values; omitted for brevity */

    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaEvent_t s0, s1;
    cudaEventCreate(&s0); cudaEventCreate(&s1);

    const float alpha = 1.0f, beta = 0.0f;
    cudaEventRecord(s0);
    /* C = alpha * A * B + beta * C, column-major; transpose flags adjust if you store row-major */
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N,
                &alpha, dA, N, dB, N,
                &beta,  dC, N);
    cudaEventRecord(s1);
    cudaEventSynchronize(s1);

    float ms = 0; cudaEventElapsedTime(&ms, s0, s1);
    double gflops = (2.0 * N * N * N) / (ms * 1e6);
    printf("cuBLAS Sgemm %dx%d: %.2f ms  %.1f GFLOPS\n", N, N, ms, gflops);

    cublasDestroy(handle);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
```

On an A100, expect ~17 TFLOPS for fp32. Your hand-rolled version probably reaches 5–10 TFLOPS — cuBLAS is ahead by a factor that grows with matrix size.

---

## Exercise 29.2 — Mixed-Precision via cublasGemmEx

**Difficulty**: ★★

For fp16 input with fp32 accumulator (the standard for ML training), use `cublasGemmEx`:

```cuda
cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
             N, N, N,
             &alpha,
             dA, CUDA_R_16F, N,
             dB, CUDA_R_16F, N,
             &beta,
             dC, CUDA_R_32F, N,
             CUBLAS_COMPUTE_32F,                 /* accumulator */
             CUBLAS_GEMM_DEFAULT_TENSOR_OP);     /* use tensor cores */
```

On Volta+ GPUs this hits 4–8× the throughput of the fp32 version. Verify against an fp32 reference: the relative error should be small (<1e-3) because the accumulator stays in fp32.

---

## Exercise 29.3 — cuSPARSE SpMV

**Difficulty**: ★★★

Sparse matrix-vector multiplication ($y = A x$, $A$ sparse) is the workhorse of GNNs and finite-element solvers. cuSPARSE handles the indexing complexity:

```cuda
#include <cusparse.h>

cusparseHandle_t sp;
cusparseCreate(&sp);

cusparseSpMatDescr_t matA;
cusparseCreateCsr(&matA, M, N, nnz,
                  d_csr_offsets, d_csr_columns, d_csr_values,
                  CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                  CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

cusparseDnVecDescr_t vecX, vecY;
cusparseCreateDnVec(&vecX, N, d_x, CUDA_R_32F);
cusparseCreateDnVec(&vecY, M, d_y, CUDA_R_32F);

float alpha = 1, beta = 0;
size_t bufsz; void *buf;
cusparseSpMV_bufferSize(sp, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha, matA, vecX, &beta, vecY,
                        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufsz);
cudaMalloc(&buf, bufsz);
cusparseSpMV(sp, CUSPARSE_OPERATION_NON_TRANSPOSE,
             &alpha, matA, vecX, &beta, vecY,
             CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buf);
```

Build a CSR matrix for an arbitrary 5-point stencil on a 1024×1024 grid (5 nonzeros per row, ~5 million total). Time SpMV vs. dense GEMV — sparse should be 100–500× faster because it avoids the zero entries entirely.

---

## Exercise 29.4 — Choosing a Library — Bonus

**Difficulty**: ★

A short table of "when to reach for which":

| Operation | Library | Function |
|-----------|---------|----------|
| Dense GEMM, GEMV | cuBLAS | `Sgemm`, `GemmEx` |
| Sparse SpMV, SpMM | cuSPARSE | `SpMV`, `SpMM` |
| FFT | cuFFT | `cufftExecC2C` |
| Random numbers | cuRAND | `curand_uniform` |
| Sort, scan, reduce | Thrust / CUB | `thrust::sort`, `cub::DeviceReduce` |
| Convolution, attention | cuDNN | `cudnnConvolutionForward` |
| LP / QP optimization | cuOPT | (newer; check availability) |

For each entry above, write one sentence describing a scenario where that library is the right pick. The point is to recognize that NVIDIA's libraries cover ~90% of common GPU primitives — your job is usually to compose them, not to re-implement.
