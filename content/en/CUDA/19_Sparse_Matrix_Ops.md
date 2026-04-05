# 19. Sparse Matrix Operations

**Previous**: [Histogram and Binning](./18_Histogram_and_Binning.md) | **Next**: [N-Body Simulation](./20_N_Body_Simulation.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Represent sparse matrices in COO, CSR, and CSC formats and convert between them
2. Implement a naive CSR SpMV kernel (one thread per row)
3. Implement a warp-per-row SpMV kernel that eliminates load imbalance
4. Use cuSPARSE `cusparseSpMV` for production sparse matrix-vector multiplication
5. Decide when to use sparse vs dense representations based on sparsity ratio

---

## 1. Why Sparse Representation?

Dense matrices store all M×N values even when most are zero. For a 1,000,000×1,000,000 matrix with 0.001% non-zeros, dense storage requires 4 TB — impossible. Sparse formats store only the non-zero values.

**Sparsity ratio**: fraction of zeros. Use sparse formats when sparsity > ~95% and the matrix is large enough that the format overhead is worthwhile.

Common sources of sparse matrices:
- Graph adjacency matrices (typically <0.01% non-zeros for large graphs)
- Finite element stiffness matrices (bandwidth limited by mesh connectivity)
- Neural network weight matrices after pruning (70–99% zeros)
- Natural language processing (word co-occurrence, document-term matrices)

---

## 2. Sparse Matrix Formats

### COO — Coordinate Format

Stores (row, col, value) triples for every non-zero:

```c
// COO representation
typedef struct {
    int    *row_indices;   // [nnz] row index of each non-zero
    int    *col_indices;   // [nnz] column index of each non-zero
    float  *values;        // [nnz] value of each non-zero
    int     nrows, ncols, nnz;
} SparseCOO;

// Example: 4×4 matrix, 6 non-zeros
//  [1 0 2 0]
//  [0 3 0 4]
//  [5 0 6 0]
//  [0 7 0 8]
//
// row_indices: [0, 0, 1, 1, 2, 2, 3, 3]
// col_indices: [0, 2, 1, 3, 0, 2, 1, 3]
// values:      [1, 2, 3, 4, 5, 6, 7, 8]
```

COO is easy to construct incrementally but slow for SpMV (rows are not contiguous).

### CSR — Compressed Sparse Row

Compresses the row indices into a pointer array:

```c
typedef struct {
    int    *row_ptr;    // [nrows+1] row_ptr[i] = index of first nnz in row i
    int    *col_idx;    // [nnz] column index of each non-zero
    float  *values;     // [nnz] value of each non-zero
    int     nrows, ncols, nnz;
} SparseCSR;

// Same example in CSR:
// row_ptr:  [0, 2, 4, 6, 8]  (row i has nnz from row_ptr[i] to row_ptr[i+1]-1)
// col_idx:  [0, 2, 1, 3, 0, 2, 1, 3]
// values:   [1, 2, 3, 4, 5, 6, 7, 8]
//
// Row 0: cols [0,2] values [1,2]
// Row 1: cols [1,3] values [3,4]
// Row 2: cols [0,2] values [5,6]
// Row 3: cols [1,3] values [7,8]
```

CSR enables O(nnz_in_row) row access with contiguous memory layout — ideal for SpMV.

### CSC — Compressed Sparse Column

CSC is CSR transposed — pointers over columns instead of rows. Used when column-oriented access is needed (e.g., SpMM with column-major matrices, or computing A^T × x directly):

```c
typedef struct {
    int    *col_ptr;    // [ncols+1]
    int    *row_idx;    // [nnz]
    float  *values;     // [nnz]
    int     nrows, ncols, nnz;
} SparseCSC;
// CSC of A = CSR of A^T (same memory layout, different interpretation)
```

---

## 3. SpMV: One Thread Per Row (Naive CSR)

Sparse matrix-vector multiplication: y = A × x, where A is M×N sparse, x is N-vector.

```c
// Naive SpMV: one thread per row of A
__global__ void spmv_csr_scalar(
    const int   *row_ptr,    // [M+1]
    const int   *col_idx,    // [nnz]
    const float *values,     // [nnz]
    const float *x,          // [N]
    float       *y,          // [M] output
    int          M)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    float sum = 0.0f;
    int row_start = row_ptr[row];
    int row_end   = row_ptr[row + 1];

    for (int j = row_start; j < row_end; j++) {
        sum += values[j] * x[col_idx[j]];
    }
    y[row] = sum;
}
```

**Problem — load imbalance**: if row lengths vary widely (e.g., graph: some nodes have degree 1, others have degree 10,000), threads within a warp stall waiting for the longest row. The warp executes as many iterations as the maximum row length.

---

## 4. SpMV: Warp Per Row

Assign all 32 threads of a warp to a single row. They collaboratively dot-product the row against x, then reduce with warp shuffle:

```c
// Warp-per-row SpMV — better load balance for irregular matrices
__global__ void spmv_csr_warp(
    const int   *row_ptr,
    const int   *col_idx,
    const float *values,
    const float *x,
    float       *y,
    int          M)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane    = threadIdx.x & 31;

    if (warp_id >= M) return;

    int row_start = row_ptr[warp_id];
    int row_end   = row_ptr[warp_id + 1];

    float sum = 0.0f;
    // Each lane handles every 32nd element in the row
    for (int j = row_start + lane; j < row_end; j += 32) {
        sum += values[j] * x[col_idx[j]];
    }

    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    if (lane == 0) y[warp_id] = sum;
}

// Launch: blockDim.x = 256 (8 warps per block), each handles one row
// grid = (M * 32 + 255) / 256
```

**When to prefer warp-per-row**: rows with ≥ 32 non-zeros on average. For very short rows (< 8 nnz), the warp is mostly idle — use scalar per-thread instead or a vector approach that assigns variable-width SIMD groups to rows based on row length.

---

## 5. cuSPARSE SpMV

For production, use cuSPARSE — it handles all format conversions, algorithm selection, and auto-tuning:

```c
#include <cusparse.h>

void cusparse_spmv(
    const int *h_row_ptr, const int *h_col_idx, const float *h_values,
    const float *h_x, float *h_y,
    int M, int N, int nnz)
{
    // Create handle
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    // Allocate device memory and upload
    int   *d_row_ptr, *d_col_idx;
    float *d_values, *d_x, *d_y;
    cudaMalloc(&d_row_ptr, (M + 1) * sizeof(int));
    cudaMalloc(&d_col_idx, nnz     * sizeof(int));
    cudaMalloc(&d_values,  nnz     * sizeof(float));
    cudaMalloc(&d_x,       N       * sizeof(float));
    cudaMalloc(&d_y,       M       * sizeof(float));
    cudaMemcpy(d_row_ptr, h_row_ptr, (M+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, h_col_idx, nnz*sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_values,  h_values,  nnz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x,       h_x,       N*sizeof(float),   cudaMemcpyHostToDevice);

    // Create matrix and vector descriptors
    cusparseSpMatDescr_t mat_A;
    cusparseDnVecDescr_t vec_x, vec_y;

    cusparseCreateCsr(&mat_A, M, N, nnz,
                      d_row_ptr, d_col_idx, d_values,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateDnVec(&vec_x, N, d_x, CUDA_R_32F);
    cusparseCreateDnVec(&vec_y, M, d_y, CUDA_R_32F);

    // Query buffer size
    float alpha = 1.0f, beta = 0.0f;
    size_t buf_size = 0;
    cusparseSpMV_bufferSize(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, mat_A, vec_x, &beta, vec_y,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &buf_size);

    void *d_buf;
    cudaMalloc(&d_buf, buf_size);

    // Execute SpMV: y = alpha * A * x + beta * y
    cusparseSpMV(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, mat_A, vec_x, &beta, vec_y,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_buf);

    cudaMemcpy(h_y, d_y, M * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cusparseDestroySpMat(mat_A);
    cusparseDestroyDnVec(vec_x); cusparseDestroyDnVec(vec_y);
    cudaFree(d_buf); cudaFree(d_row_ptr); cudaFree(d_col_idx);
    cudaFree(d_values); cudaFree(d_x); cudaFree(d_y);
    cusparseDestroy(handle);
}
```

---

## 6. SpGEMM Concept (Sparse × Sparse)

Sparse matrix-matrix multiplication (C = A × B where both are sparse) is far more complex than SpMV because the sparsity pattern of C is not known in advance.

The cuSPARSE approach:
1. **Work estimation**: determine the number of non-zeros in C
2. **Allocate C**: based on the estimated count
3. **Compute C**: fill in the non-zero values

```c
// Conceptual cuSPARSE SpGEMM (simplified API sketch)
cusparseSpGEMMDescr_t spgemm_descr;
cusparseSpGEMM_createDescr(&spgemm_descr);

// Step 1: work estimate
cusparseSpGEMM_workEstimation(handle, opA, opB,
    &alpha, mat_A, mat_B, &beta, mat_C,
    CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
    spgemm_descr, &buf1_size, NULL);
cudaMalloc(&d_buf1, buf1_size);

// Step 2: compute
cusparseSpGEMM_compute(handle, opA, opB,
    &alpha, mat_A, mat_B, &beta, mat_C,
    CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
    spgemm_descr, &buf2_size, NULL);
// ... (allocate buf2, copy result)
```

---

## 7. Sparse vs Dense Decision Guide

```
Sparsity     Matrix Size      Recommendation
-----------------------------------------------------------
< 90%        Any              Dense (cuBLAS): fewer indirection overheads
90–99%       < 10K × 10K      Dense might still win (cuBLAS has high flop/byte)
> 99%        > 100K × 100K    Sparse (cuSPARSE): memory is dominant constraint
> 99.9%      > 1M × 1M        Sparse mandatory (dense = terabytes)

Rule of thumb: use sparse if nnz/M/N < 0.01 (1% dense)

Special cases:
  Structured sparsity (block-sparse): use block-CSR or BSRMM in cuSPARSE
  Pruned neural nets: use cuSPARSE or NVIDIA ASP (Accelerated Sparse Precision)
  Dynamic sparsity: COO or hash-map based (rebuild CSR on changes)
```

---

## Key Takeaways

- **COO** stores (row, col, val) triples — easy to build, slow to compute with
- **CSR** compresses row pointers — optimal for SpMV and row-wise access
- **CSC** compresses column pointers — optimal for column-wise access (or A^T × x)
- One-thread-per-row SpMV suffers from **load imbalance**; warp-per-row reduces this for rows with ≥32 nnz
- **cuSPARSE `cusparseSpMV`** with generic API handles CSR/CSC/COO and auto-selects the best kernel
- Use sparse formats when sparsity > 99% and the matrix is large; below that, dense cuBLAS often wins due to lower overhead

---

**Next**: [20. N-Body Simulation](./20_N_Body_Simulation.md) — Compute gravitational forces for N particles with O(N²) direct summation and tile-based shared memory optimization.
