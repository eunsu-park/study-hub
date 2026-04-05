# 03. Thread Indexing and Grids

**Previous**: [CUDA Programming Model](./02_CUDA_Programming_Model.md) | **Next**: [CUDA Memory Model](./04_CUDA_Memory_Model.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Calculate correct global thread indices for 1D, 2D, and 3D grids
2. Handle arbitrary array sizes with boundary checks
3. Implement a matrix transpose kernel using 2D indexing
4. Choose appropriate grid and block dimensions for any problem shape
5. Debug incorrect indexing with systematic test patterns

---

## 1. Why Indexing Is the #1 CUDA Bug Source

The single most common CUDA bug: **wrong index calculation**. Off-by-one errors, missing boundary guards, and confused row/column order produce silent wrong results or random crashes. Master this before writing any real kernel.

The goal: map each thread to exactly the data elements it should process.

---

## 2. 1D Indexing

For flat arrays:

```c
__global__ void kernel_1d(float *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = process(data[i]);
    }
}
```

### When N is not a multiple of blockSize

```
N = 10, blockSize = 4:
  gridSize = ceil(10/4) = 3 blocks → 12 threads total

Block 0: threads 0,1,2,3  → i = 0,1,2,3   ✓ all valid
Block 1: threads 0,1,2,3  → i = 4,5,6,7   ✓ all valid
Block 2: threads 0,1,2,3  → i = 8,9,10,11 ⚠ i=10,11 out of bounds → guard needed
```

Grid size formula: `int gridSize = (N + blockSize - 1) / blockSize;`

This is equivalent to `ceil(N / blockSize)` but avoids floating-point division.

---

## 3. 2D Indexing

For 2D problems (matrices, images), use `dim3` for both grid and block:

```c
__global__ void kernel_2d(float *data, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // x → column
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // y → row

    if (row < rows && col < cols) {
        int idx = row * cols + col;  // row-major (C convention)
        data[idx] = process(data[idx]);
    }
}

// Launch:
dim3 block(16, 16);  // 256 threads per block, 16×16 arrangement
dim3 grid(
    (cols + block.x - 1) / block.x,
    (rows + block.y - 1) / block.y
);
kernel_2d<<<grid, block>>>(d_data, rows, cols);
```

**Convention**: `x` indexes the faster-varying dimension (columns in row-major layout). This matches memory layout and maximizes coalescing — adjacent threads (same row, adjacent x) access adjacent memory addresses.

```
Thread (row=2, col=3) in a 5×8 matrix:
  blockIdx  = (0, 0), blockDim = (8, 4), threadIdx = (3, 2)
  col = 0*8 + 3 = 3
  row = 0*4 + 2 = 2
  idx = 2 * 8 + 3 = 19   ✓
```

---

## 4. 3D Indexing

For volumes (3D tensors, voxel grids, batch operations):

```c
__global__ void kernel_3d(float *data, int D, int H, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // width
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // height
    int z = blockIdx.z * blockDim.z + threadIdx.z;  // depth

    if (x < W && y < H && z < D) {
        int idx = z * (H * W) + y * W + x;
        data[idx] = process(data[idx]);
    }
}

dim3 block(8, 8, 4);  // 256 threads per block in 3D
dim3 grid(
    (W + block.x - 1) / block.x,
    (H + block.y - 1) / block.y,
    (D + block.z - 1) / block.z
);
```

**Limit**: `gridDim.z` has a maximum of 65,535. For batch processing of large batches, loop inside the kernel or use a 2D grid with the batch index in one dimension.

---

## 5. Case Study: Matrix Transpose

A matrix transpose reads A[row][col] and writes B[col][row]. The naive version has a problem: either reads or writes are non-coalesced.

### Naive (non-coalesced writes)

```c
__global__ void transpose_naive(const float *in, float *out, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        out[col * rows + row] = in[row * cols + col];
        //   ↑ strided write: thread 0→0, thread 1→rows, thread 2→2*rows ...
    }
}
```

- **Reads** `in[row * cols + col]`: coalesced ✓ (adjacent threads read adjacent memory)
- **Writes** `out[col * rows + row]`: strided ✗ (adjacent threads write with stride = rows)

This is a classic performance problem — see L08 (Memory Coalescing) for the solution using shared memory as a staging buffer.

### Tiled (using shared memory — preview)

```c
#define TILE 32

__global__ void transpose_tiled(const float *in, float *out, int rows, int cols) {
    __shared__ float tile[TILE][TILE + 1];  // +1 avoids bank conflicts

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    // Read tile: coalesced reads from global memory
    if (x < cols && y < rows)
        tile[threadIdx.y][threadIdx.x] = in[y * cols + x];

    __syncthreads();  // wait for all threads to fill the tile

    // Write tile transposed: coalesced writes to global memory
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;

    if (x < rows && y < cols)
        out[y * rows + x] = tile[threadIdx.x][threadIdx.y];
}
```

This achieves near-peak memory bandwidth. We will analyze it in detail in L05.

---

## 6. Choosing Block Dimensions

Guidelines for 1D kernels:

| Block size | Behavior |
|------------|----------|
| < 32 | Wastes a warp (32 threads always execute together) |
| 32 | Minimal — only 1 warp per block, low occupancy |
| 128 | Common choice — 4 warps per block, good occupancy |
| **256** | **Most common — 8 warps, good occupancy, widely compatible** |
| 512 | OK — check register/shared memory limits |
| 1024 | Maximum — only if registers/shared memory are very low |

For 2D kernels, `(16, 16)` is the standard — 256 threads arranged to match tile sizes.

```c
// Occupancy calculator (CUDA 6.5+)
int minGridSize, blockSize;
cudaOccupancyMaxPotentialBlockSize(
    &minGridSize,           // recommended grid size
    &blockSize,             // optimal block size
    myKernel,               // kernel function
    0,                      // dynamic shared memory per block
    0                       // block size limit (0 = no limit)
);
printf("Optimal block size: %d\n", blockSize);
```

---

## 7. Stride-Based Indexing for Arbitrary Sizes

When one kernel call must handle more work than the grid can contain, use a **grid-stride loop**:

```c
__global__ void scale_stride(float *data, float scalar, long n) {
    // Start at this thread's first element
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    // Stride by the total number of threads in the grid
    long stride = (long)gridDim.x * blockDim.x;

    for (; i < n; i += stride) {
        data[i] *= scalar;
    }
}

// Can launch with any grid size:
scale_stride<<<1024, 256>>>(d_data, 2.0f, 1e9);  // 1 billion elements
```

Benefits:
- Works correctly regardless of N
- Allows launching a fixed-size grid tuned for occupancy, independent of data size
- Simpler debugging: can use `<<<1, 1>>>` to test the loop logic

---

## 8. Debugging Index Errors

Systematic approach: write an index-verifying kernel.

```c
__global__ void verify_indexing(int *out, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        int expected_flat = row * cols + col;
        out[expected_flat] = expected_flat;  // each thread writes its own index
    }
}

// After running: verify that out[i] == i for all i
// Any mismatch reveals the index calculation bug
```

For off-by-one boundary bugs:

```c
// Sentinel approach: fill with -1, then check no -1 remains
cudaMemset(d_out, -1, bytes);
myKernel<<<grid, block>>>(d_out, N);
cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
for (int i = 0; i < N; i++) {
    assert(h_out[i] != -1);  // some elements were not written
}
```

---

## 9. Summary: Index Patterns by Problem Type

```c
// 1D array of length N
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < N) { ... }

// 2D matrix M×N (rows × cols), row-major
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
if (row < M && col < N) {
    int idx = row * N + col;
    ...
}

// Batched 2D: batch size B, each of M×N
int col   = blockIdx.x * blockDim.x + threadIdx.x;
int row   = blockIdx.y * blockDim.y + threadIdx.y;
int batch = blockIdx.z;
if (batch < B && row < M && col < N) {
    int idx = batch * (M * N) + row * N + col;
    ...
}

// Grid-stride loop
for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
         i < N;
         i += (long)gridDim.x * blockDim.x) {
    ...
}
```

---

## Key Takeaways

- **1D index**: `i = blockIdx.x * blockDim.x + threadIdx.x` — always add boundary guard `if (i < N)`
- **2D convention**: `x` → columns (fast-varying), `y` → rows; matches row-major memory for coalescing
- **Block size**: multiples of 32; 256 is the safe default; use `cudaOccupancyMaxPotentialBlockSize` for tuning
- **Grid-stride loop**: decouples problem size from grid size — use for very large arrays or reusable kernels
- Always verify indexing with a sentinel or index-check kernel before trusting results

---

**Next**: [04. CUDA Memory Model](./04_CUDA_Memory_Model.md) — Explore all GPU memory types: global, shared, registers, L1/L2, constant, and texture memory.
