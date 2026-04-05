# 29. cuBLAS and cuSPARSE

**Previous**: [Thrust and CUB](./28_Thrust_and_CUB.md) | **Next**: [Mixed Precision and Tensor Cores](./30_Mixed_Precision_and_Tensor_Cores.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Initialize a cuBLAS handle and call `cublasSgemm` for dense matrix multiplication
2. Understand cuBLAS's column-major convention and correctly map row-major C arrays
3. Perform batched GEMM using `cublasGemmBatchedEx` for many small matrices
4. Use `cublasGemmEx` with Tensor Core acceleration via `CUDA_R_16F`
5. Format sparse matrices in CSR and use cuSPARSE for sparse matrix-vector multiplication (SpMV)

---

## 1. cuBLAS Handle and Setup

Every cuBLAS function requires a `cublasHandle_t` that encapsulates the CUDA context, stream, and workspace:

```c
#include <cublas_v2.h>

cublasHandle_t handle;
cublasCreate(&handle);

// Associate with a non-default stream (optional)
cudaStream_t stream;
cudaStreamCreate(&stream);
cublasSetStream(handle, stream);

// Always destroy when done
// cublasDestroy(handle);
```

**Error checking macro:**

```c
#define CUBLAS_CHECK(call) do {                                 \
    cublasStatus_t status = call;                               \
    if (status != CUBLAS_STATUS_SUCCESS) {                      \
        fprintf(stderr, "cuBLAS error %d at %s:%d\n",          \
                status, __FILE__, __LINE__);                    \
        exit(1);                                                \
    }                                                           \
} while(0)
```

---

## 2. cuBLAS Column-Major Convention

cuBLAS follows **Fortran (column-major)** convention. For row-major C arrays, you must **transpose the arguments or swap M/N**:

```
cuBLAS computes: C = α·op(A)·op(B) + β·C

For row-major matrices A(M×K), B(K×N), C(M×N):
  Pass them as column-major B^T(N×K), A^T(K×M), C^T(N×M)
  → cublasSgemm computes C^T = α·B^T·A^T + β·C^T
  → which equals (C = α·A·B + β·C)^T ← correct!

Trick: swap arguments A↔B, swap M↔N:
  cublasSgemm(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    N, M, K,           // ← M and N are swapped
    &alpha,
    d_B, N,            // ← B comes first, leading dim = N
    d_A, K,            // ← A comes second, leading dim = K
    &beta,
    d_C, N);           // ← output leading dim = N
```

```c
// Complete example: C = A * B, A is M×K, B is K×N, C is M×N (all row-major)
void sgemm_rowmajor(cublasHandle_t handle,
                    const float *d_A, const float *d_B, float *d_C,
                    int M, int N, int K) {
    float alpha = 1.0f, beta = 0.0f;

    // Row-major trick: swap A↔B and M↔N
    CUBLAS_CHECK(cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N,      // rows of op(B) = N
        M,      // cols of op(A) = M
        K,      // inner dimension
        &alpha,
        d_B, N, // B: leading dim = N (each row has N elements in row-major)
        d_A, K, // A: leading dim = K
        &beta,
        d_C, N  // C: leading dim = N
    ));
}
```

---

## 3. Performance: cuBLAS vs Naive GEMM

```
Matrix size 4096×4096, FP32, A100 GPU:

Kernel              Time     TFLOPS
-----------------------------------
Our L32 kernel v2   1.8 ms    75.6
Our L32 kernel v3   1.1 ms   124.8   (register tiling)
cublasSgemm         0.65 ms  211.5   (uses Tensor Cores internally)
Theoretical peak    0.42 ms  312     (A100 FP32 dense)

For FP16 (Tensor Cores):
cublasHgemm         0.17 ms  ~600 TFLOPS (A100 has 312 TF FP16 TC)
```

---

## 4. Batched GEMM

Batched GEMM runs many independent matrix multiplications in one call — essential for neural network layers operating on mini-batches:

```c
// Batched GEMM: compute C_i = A_i * B_i for i = 0..batch_size-1
// All matrices same shape: A(M×K), B(K×N), C(M×N)
void batched_gemm(cublasHandle_t handle,
                  int M, int N, int K, int batch_size) {
    float alpha = 1.f, beta = 0.f;

    // Method 1: cublasGemmBatchedEx — array of pointers
    // Each d_Aarray[i] points to a different M×K matrix

    float **d_Aarray, **d_Barray, **d_Carray;
    cudaMalloc(&d_Aarray, batch_size * sizeof(float*));
    cudaMalloc(&d_Barray, batch_size * sizeof(float*));
    cudaMalloc(&d_Carray, batch_size * sizeof(float*));

    // Fill pointer arrays on host, then copy to device
    float **h_Aarray = (float**)malloc(batch_size * sizeof(float*));
    // ... allocate each matrix and fill h_Aarray[i] ...
    cudaMemcpy(d_Aarray, h_Aarray, batch_size*sizeof(float*), cudaMemcpyHostToDevice);

    CUBLAS_CHECK(cublasGemmBatchedEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        (const void**)d_Barray, CUDA_R_32F, N,
        (const void**)d_Aarray, CUDA_R_32F, K,
        &beta,
        (void**)d_Carray,       CUDA_R_32F, N,
        batch_size,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT));

    // Method 2: cublasGemmStridedBatchedEx — contiguous strides (more efficient)
    // Assumes A[i] = base_A + i * stride_A, etc.
    long long stride_A = M * K, stride_B = K * N, stride_C = M * N;
    float *d_A, *d_B, *d_C;
    // ... allocate batch_size * M*K floats for d_A, etc. ...

    CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, CUDA_R_32F, N, stride_B,
        d_A, CUDA_R_32F, K, stride_A,
        &beta,
        d_C, CUDA_R_32F, N, stride_C,
        batch_size,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT));
}
```

---

## 5. Tensor Core GEMM (FP16)

Tensor Cores compute 16×16×16 FP16 matrix multiply-accumulate in a single instruction. Enable via `CUDA_R_16F` and `CUBLAS_COMPUTE_32F_FAST_16F`:

```c
#include <cuda_fp16.h>

void gemm_tensor_cores(cublasHandle_t handle,
                       const half *d_A, const half *d_B, float *d_C,
                       int M, int N, int K) {
    float alpha = 1.f, beta = 0.f;

    // cublasGemmEx: explicit type specification
    CUBLAS_CHECK(cublasGemmEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, CUDA_R_16F, N,    // B in FP16
        d_A, CUDA_R_16F, K,    // A in FP16
        &beta,
        d_C, CUDA_R_32F, N,    // C accumulated in FP32
        CUBLAS_COMPUTE_32F_FAST_16F,   // use Tensor Cores
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
}

// Convert FP32 arrays to FP16 before calling
__global__ void f32_to_f16(const float *in, half *out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = __float2half(in[i]);
}
```

---

## 6. cuSPARSE: CSR Format

Sparse matrices are stored compactly in Compressed Sparse Row (CSR) format:

```
Dense 4×4 matrix:
  0  3  0  0
  2  0  0  5
  0  1  4  0
  0  0  0  6

CSR representation:
  values   = [3, 2, 5, 1, 4, 6]          (non-zeros in row-major order)
  col_idx  = [1, 0, 3, 1, 2, 3]          (column index of each non-zero)
  row_ptr  = [0, 1, 3, 5, 6]             (start of each row in values; length = nrows+1)

nnz = 6 (number of non-zeros)
Compression ratio: 6/16 = 37.5% of dense storage
```

---

## 7. cuSPARSE SpMV

Sparse matrix-vector multiplication y = A·x using CSR format:

```c
#include <cusparse.h>

void spmv_csr(
    int nrows, int ncols, int nnz,
    const int *d_row_ptr, const int *d_col_idx, const float *d_values,
    const float *d_x, float *d_y)
{
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    // Create matrix descriptor
    cusparseSpMatDescr_t matA;
    cusparseCreateCsr(
        &matA,
        nrows, ncols, nnz,
        (void*)d_row_ptr, (void*)d_col_idx, (void*)d_values,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    // Create vector descriptors
    cusparseDnVecDescr_t vecX, vecY;
    cusparseCreateDnVec(&vecX, ncols, (void*)d_x, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, nrows, (void*)d_y, CUDA_R_32F);

    float alpha = 1.f, beta = 0.f;

    // Query buffer size
    void   *d_buf = nullptr;
    size_t  buf_bytes = 0;
    cusparseSpMV_bufferSize(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &buf_bytes);
    cudaMalloc(&d_buf, buf_bytes);

    // Execute SpMV
    cusparseSpMV(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_buf);

    // Cleanup
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cudaFree(d_buf);
    cusparseDestroy(handle);
}
```

---

## 8. Performance: Dense vs Sparse

```
SpMV is memory-bandwidth bound; the benefit of sparsity depends on nnz/N ratio.

Matrix: N=100,000 rows, K=100,000 cols
Case A: dense A (10B elements) → GEMV at 900 GB/s: ~88 ms
Case B: sparse A (1% density, 100M nnz) → SpMV: ~2 ms
  → 44× faster when 99% zeros

Real-world deep learning (BERT attention, sparsity 50%):
  Dense GEMM:       0.8 ms
  Sparse (CSR):     1.2 ms  ← often SLOWER for moderate sparsity!
  Reason: CSR has poor memory access patterns; needs >90% sparsity for speedup

Structured sparsity (2:4 format — 2 non-zeros per 4 elements):
  NVIDIA Ampere cuSPARSE structured:  ~1.5× faster than dense at exactly 50% sparse
  Requires specific pattern; used in Ampere sparse Tensor Cores
```

---

## 9. cuSPARSE SpMM (Sparse × Dense Matrix)

```c
// y = A * B where A is sparse CSR (nrows×k), B is dense (k×ncols)
void spmm_csr(cusparseHandle_t handle,
              cusparseSpMatDescr_t matA,
              const float *d_B, float *d_C,
              int nrows, int ncols_B, int k) {
    cusparseDnMatDescr_t matB, matC;
    cusparseCreateDnMat(&matB, k,     ncols_B, ncols_B, (void*)d_B, CUDA_R_32F, CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&matC, nrows, ncols_B, ncols_B, (void*)d_C, CUDA_R_32F, CUSPARSE_ORDER_ROW);

    float alpha = 1.f, beta = 0.f;
    void *d_buf = nullptr; size_t buf_bytes = 0;
    cusparseSpMM_bufferSize(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC,
        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &buf_bytes);
    cudaMalloc(&d_buf, buf_bytes);
    cusparseSpMM(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC,
        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, d_buf);
    cusparseDestroyDnMat(matB); cusparseDestroyDnMat(matC);
    cudaFree(d_buf);
}
```

---

## Key Takeaways

- **cuBLAS row-major trick**: to use row-major C arrays with column-major cuBLAS, swap A↔B and swap M↔N in the `cublasSgemm` call
- **Leading dimension** is the stride between consecutive columns (column-major): for a row-major M×N matrix, the leading dimension is N
- **Batched GEMM**: `cublasGemmBatchedEx` (array of pointers) and `cublasGemmStridedBatchedEx` (fixed stride) handle independent mini-batch matrix multiplications in one call
- **Tensor Cores**: enabled via `CUBLAS_COMPUTE_32F_FAST_16F` with `CUDA_R_16F` inputs; can deliver 2-4× the FLOPS of FP32 CUDA cores
- **CSR format**: stores nnz values with corresponding column indices and a row pointer array; memory-efficient for <10% density
- **SpMV performance**: sparse is faster than dense only at high sparsity (>80-90%); structured 2:4 sparsity on Ampere delivers consistent ~2× speedup

---

**Next**: [30. Mixed Precision and Tensor Cores](./30_Mixed_Precision_and_Tensor_Cores.md) — Exploit FP16, BF16, and the WMMA API to write CUDA kernels that directly program Tensor Core matrix operations.
