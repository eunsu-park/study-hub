# Block 1 — Fundamentals

**Lessons covered**: L01 (Why GPU?), L02 (CUDA Programming Model), L03 (Memory Hierarchy),
L04 (Thread Organization), L05 (Synchronization), L06 (Warp Execution), L07 (Profiling Basics)

---

## Exercise 1.1 — Vector Scale

**Concept introduced in**: L02 (CUDA Programming Model), L03 (Memory Hierarchy)

### Problem Statement

Write a CUDA kernel `scale_vector` that multiplies every element of a float array by a scalar
value `alpha`. Allocate device memory, copy the host array to the device, launch the kernel,
copy the result back, and verify correctness against a CPU reference.

### Requirements

- Handle arrays whose length `N` is not a multiple of the block size (boundary guard).
- Use a 1D grid of 1D blocks.
- Block size: 256 threads.

### Starter Code

```cuda
// ex1_1_scale_vector.cu
// Compile: nvcc -O2 -arch=sm_80 -o ex1_1 ex1_1_scale_vector.cu

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cassert>

#define N 1 << 20    // 1M elements
#define BLOCK_SIZE 256

// TODO: Implement the kernel.
// Each thread computes: d_out[i] = alpha * d_in[i]
// Guard against out-of-bounds access when N % BLOCK_SIZE != 0.
__global__ void scale_vector(const float* d_in, float* d_out, float alpha, int n) {
    // TODO
}

int main() {
    const float alpha = 2.5f;
    const int n = N;

    // Allocate host arrays
    float* h_in  = new float[n];
    float* h_out = new float[n];
    for (int i = 0; i < n; ++i) h_in[i] = static_cast<float>(i);

    // TODO: Allocate device memory (d_in, d_out)

    // TODO: Copy h_in -> d_in

    // TODO: Compute grid size and launch scale_vector kernel

    // TODO: Copy d_out -> h_out

    // Verify
    bool ok = true;
    for (int i = 0; i < n; ++i) {
        if (fabsf(h_out[i] - alpha * h_in[i]) > 1e-5f) { ok = false; break; }
    }
    printf("Result: %s\n", ok ? "PASS" : "FAIL");

    // TODO: Free device memory
    delete[] h_in;
    delete[] h_out;
    return ok ? 0 : 1;
}
```

### Expected Output

```
Result: PASS
```

### Hints

| Step | API |
|------|-----|
| Allocate device memory | `cudaMalloc` |
| Copy host → device | `cudaMemcpy(..., cudaMemcpyHostToDevice)` |
| Launch kernel | `<<<gridDim, blockDim>>>` where `gridDim = (n + BLOCK_SIZE - 1) / BLOCK_SIZE` |
| Copy device → host | `cudaMemcpy(..., cudaMemcpyDeviceToHost)` |
| Free | `cudaFree` |

Inside the kernel: `int i = blockIdx.x * blockDim.x + threadIdx.x;` then guard with `if (i < n)`.

### Performance Target

Not the focus of this exercise — correctness only. The kernel should finish in < 5 ms for N = 1M on any Turing+ GPU.

---

## Exercise 1.2 — 2D Matrix Addition

**Concept introduced in**: L04 (Thread Organization)

### Problem Statement

Write a kernel `mat_add` that computes `C = A + B` for two row-major float matrices of size
`M × N` where `M` and `N` may differ and may not be multiples of the tile size.

### Requirements

- Use a 2D grid of 2D thread blocks (block size 16×16 recommended).
- Handle the case `M ≠ N` (non-square matrices).
- Verify the result against a CPU reference.

### Starter Code

```cuda
// ex1_2_mat_add.cu
// Compile: nvcc -O2 -arch=sm_80 -o ex1_2 ex1_2_mat_add.cu

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#define M 1000   // rows (intentionally non-power-of-2)
#define N 768    // cols
#define TILE 16

// TODO: Implement the kernel.
// A, B, C are stored in row-major order.
// Element (row, col) lives at index row * n_cols + col.
// Guard rows < m and cols < n.
__global__ void mat_add(const float* A, const float* B, float* C, int m, int n) {
    // TODO
}

int main() {
    const int size = M * N;
    float* h_A = new float[size];
    float* h_B = new float[size];
    float* h_C = new float[size];

    for (int i = 0; i < size; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(size - i);
    }

    // TODO: Allocate d_A, d_B, d_C
    // TODO: Copy h_A -> d_A, h_B -> d_B

    // TODO: Define dim3 blockDim(TILE, TILE)
    //       Define dim3 gridDim(...) — remember to ceiling-divide both M and N

    // TODO: Launch mat_add kernel

    // TODO: Copy d_C -> h_C

    // Verify
    bool ok = true;
    for (int i = 0; i < size; ++i) {
        if (fabsf(h_C[i] - (h_A[i] + h_B[i])) > 1e-5f) { ok = false; break; }
    }
    printf("Result: %s\n", ok ? "PASS" : "FAIL");

    // TODO: Free device memory
    delete[] h_A; delete[] h_B; delete[] h_C;
    return ok ? 0 : 1;
}
```

### Expected Output

```
Result: PASS
```

### Hints

- `dim3 blockDim(TILE, TILE);`
- `dim3 gridDim((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);`
- Inside kernel: `int col = blockIdx.x * blockDim.x + threadIdx.x;` and `int row = blockIdx.y * blockDim.y + threadIdx.y;`
- Guard: `if (row < m && col < n)`

### Performance Target

Correctness focus. The kernel should be memory-bandwidth bound (not compute bound) and run in < 2 ms for M=1000, N=768.

---

## Exercise 1.3 — Global Memory Bandwidth Benchmark

**Concept introduced in**: L03 (Memory Hierarchy), L07 (Profiling Basics)

### Problem Statement

Implement a streaming copy kernel `mem_copy` that reads from one device array and writes to
another. Measure the effective memory bandwidth in GB/s using CUDA events. Compare your
measured bandwidth against the theoretical peak for your GPU (check `deviceQuery` or the
datasheet).

### Requirements

- Array size: at least 256 MB (64M float elements).
- Use `cudaEventRecord` / `cudaEventElapsedTime` for timing.
- Print bandwidth in GB/s.
- Run the kernel 10 times and average to reduce jitter.

### Starter Code

```cuda
// ex1_3_bandwidth.cu
// Compile: nvcc -O2 -arch=sm_80 -o ex1_3 ex1_3_bandwidth.cu

#include <cuda_runtime.h>
#include <cstdio>

#define NELEMS (1 << 26)   // 64M floats = 256 MB
#define BLOCK_SIZE 256
#define NRUNS 10

// Streaming copy: d_out[i] = d_in[i]
// Keep it simple — the bottleneck must be memory, not arithmetic.
__global__ void mem_copy(const float* __restrict__ d_in,
                         float* __restrict__       d_out,
                         int n) {
    // TODO
}

int main() {
    const int n = NELEMS;
    const size_t bytes = n * sizeof(float);

    float *d_in, *d_out;
    // TODO: cudaMalloc d_in, d_out
    // TODO: cudaMemset d_in to some value so reads are valid

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up run (not timed)
    // TODO: launch mem_copy kernel once

    float total_ms = 0.0f;
    for (int r = 0; r < NRUNS; ++r) {
        // TODO: Record start event, launch kernel, record stop event, synchronize, accumulate elapsed time
    }

    float avg_ms = total_ms / NRUNS;
    // Each run transfers 2 * bytes (one read + one write)
    double bw_gbs = (2.0 * bytes) / (avg_ms * 1e-3) / 1e9;
    printf("Average time: %.3f ms\n", avg_ms);
    printf("Effective bandwidth: %.1f GB/s\n", bw_gbs);

    // TODO: cudaFree, cudaEventDestroy
    return 0;
}
```

### Expected Output (RTX 3080 example)

```
Average time: 4.2 ms
Effective bandwidth: 121.8 GB/s
```

Typical values: 300–700 GB/s on A100, 900+ GB/s on H100 HBM.

### Hints

- Use `__restrict__` on kernel pointers to hint the compiler they don't alias.
- `cudaEventElapsedTime(&ms, start, stop)` returns milliseconds.
- Bandwidth formula: `(2 * bytes) / (time_seconds) / 1e9` (factor 2 for read + write).
- To hit peak bandwidth the access pattern must be perfectly coalesced (stride-1 with aligned base).

### Performance Target

Achieve at least 80% of your GPU's theoretical peak memory bandwidth. If below 60%, check alignment and block size.

---

## Exercise 1.4 — Shared Memory Transpose (Bank-Conflict Free)

**Concept introduced in**: L03 (Memory Hierarchy), L05 (Synchronization)

### Problem Statement

Implement a matrix transpose kernel that uses shared memory to convert the random-access
write pattern (naive transpose) into a coalesced write. To eliminate shared memory bank
conflicts, pad the shared memory tile by 1 extra column.

Verify correctness and compare bandwidth against the naive (global-memory-only) transpose.

### Requirements

- Tile size: 32×32.
- Shared memory tile declared as `__shared__ float tile[TILE][TILE + 1]` (the `+1` avoids conflicts).
- Load phase: coalesced read from `d_in`, store into `tile[threadIdx.y][threadIdx.x]`.
- Store phase: after `__syncthreads()`, coalesced write from `tile[threadIdx.x][threadIdx.y]` into `d_out`.
- Matrix size: 4096×4096 floats.

### Starter Code

```cuda
// ex1_4_transpose.cu
// Compile: nvcc -O2 -arch=sm_80 -o ex1_4 ex1_4_transpose.cu

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#define DIM  4096
#define TILE 32

// Naive transpose — reference for bandwidth comparison
__global__ void transpose_naive(const float* in, float* out, int rows, int cols) {
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < cols && y < rows)
        out[x * rows + y] = in[y * cols + x];
}

// TODO: Implement transpose_smem using padded shared memory tile.
// The output element (x, y) of the transposed matrix maps to position (x * rows + y) in d_out.
__global__ void transpose_smem(const float* in, float* out, int rows, int cols) {
    __shared__ float tile[TILE][TILE + 1];  // +1 to avoid bank conflicts

    // TODO: compute global (x, y) coordinates
    // TODO: load tile from d_in (coalesced read)
    // TODO: __syncthreads()
    // TODO: compute transposed output coordinates
    // TODO: store from tile to d_out (coalesced write)
}

float time_kernel(void (*launch)(const float*, float*, int, int),
                  const float* d_in, float* d_out, int rows, int cols) {
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);

    dim3 block(TILE, TILE);
    dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);

    // Warm-up
    launch<<<grid, block>>>(d_in, d_out, rows, cols);

    cudaEventRecord(s);
    for (int i = 0; i < 20; ++i) launch<<<grid, block>>>(d_in, d_out, rows, cols);
    cudaEventRecord(e);
    cudaEventSynchronize(e);

    float ms; cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms / 20.0f;
}

int main() {
    const int rows = DIM, cols = DIM;
    const size_t bytes = rows * cols * sizeof(float);

    float *d_in, *d_out;
    // TODO: cudaMalloc d_in, d_out; initialize d_in with cudaMemset or a fill kernel

    // Verify correctness of transpose_smem
    // TODO: launch transpose_smem, copy result to host, check d_out[j * rows + i] == d_in[i * cols + j]

    // Measure bandwidth
    auto launch_naive = [](const float* a, float* b, int r, int c) {
        dim3 blk(TILE, TILE);
        dim3 grd((c + TILE - 1) / TILE, (r + TILE - 1) / TILE);
        transpose_naive<<<grd, blk>>>(a, b, r, c);
    };
    auto launch_smem = [](const float* a, float* b, int r, int c) {
        dim3 blk(TILE, TILE);
        dim3 grd((c + TILE - 1) / TILE, (r + TILE - 1) / TILE);
        transpose_smem<<<grd, blk>>>(a, b, r, c);
    };

    float ms_naive = time_kernel(launch_naive, d_in, d_out, rows, cols);
    float ms_smem  = time_kernel(launch_smem,  d_in, d_out, rows, cols);

    double bw_naive = (2.0 * bytes) / (ms_naive * 1e-3) / 1e9;
    double bw_smem  = (2.0 * bytes) / (ms_smem  * 1e-3) / 1e9;

    printf("Naive transpose:  %.2f ms  %.1f GB/s\n", ms_naive, bw_naive);
    printf("Smem  transpose:  %.2f ms  %.1f GB/s\n", ms_smem,  bw_smem);
    printf("Speedup: %.2fx\n", ms_naive / ms_smem);

    // TODO: cudaFree
    return 0;
}
```

### Expected Output

```
Naive transpose:  8.31 ms  16.1 GB/s
Smem  transpose:  1.74 ms  77.0 GB/s
Speedup: 4.78x
```

### Hints

- Bank conflicts occur when 32 threads in a warp access the same bank. With `float tile[32][32]`, consecutive rows are 32 floats = 128 bytes apart; banks repeat every 4 bytes → threads in the same warp hit the same bank on column access. The `+1` padding shifts every row by 4 bytes, breaking alignment.
- Load: `tile[threadIdx.y][threadIdx.x] = in[y * cols + x]` (coalesced since `threadIdx.x` strides columns).
- Store: `out[transposed_x * rows + transposed_y] = tile[threadIdx.x][threadIdx.y]` — note swapped thread indices.

### Performance Target

Shared memory version should achieve at least 70% of peak memory bandwidth. Speedup over naive should be 3–6×.

---

## Exercise 1.5 — Warp Shuffle Reduction

**Concept introduced in**: L06 (Warp Execution)

### Problem Statement

Implement a block-level sum reduction using warp shuffle intrinsics (`__shfl_down_sync`)
instead of shared memory. The reduction should work for a block of 256 threads (8 warps).

### Requirements

- Each thread loads one element from the input array.
- Perform a warp-level reduction using `__shfl_down_sync` with strides 16, 8, 4, 2, 1.
- Collect warp sums in shared memory (one entry per warp), then reduce them in lane 0 of warp 0.
- Write the block result to `d_partial[blockIdx.x]`.
- Launch a second kernel (or use `atomicAdd`) to produce the final scalar sum.
- Verify the final sum equals `N * (N - 1) / 2` for input array `[0, 1, 2, ..., N-1]`.

### Starter Code

```cuda
// ex1_5_warp_reduce.cu
// Compile: nvcc -O2 -arch=sm_80 -o ex1_5 ex1_5_warp_reduce.cu

#include <cuda_runtime.h>
#include <cstdio>

#define N          (1 << 20)   // 1M elements
#define BLOCK_SIZE 256
#define WARP_SIZE  32

// Warp-level reduction helper.
// TODO: complete the butterfly pattern.
__device__ float warp_reduce_sum(float val) {
    // Iterate strides: 16, 8, 4, 2, 1
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        // TODO: val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction using warp shuffle + shared memory for cross-warp accumulation.
__global__ void block_reduce(const float* d_in, float* d_partial, int n) {
    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE];  // one slot per warp

    int tid  = threadIdx.x;
    int gid  = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = tid % WARP_SIZE;
    int wid  = tid / WARP_SIZE;

    // Each thread loads its element (or 0 if out of range)
    float val = (gid < n) ? d_in[gid] : 0.0f;

    // TODO: call warp_reduce_sum(val) to get each warp's partial sum
    // TODO: lane 0 of each warp stores its sum to warp_sums[wid]
    // TODO: __syncthreads()
    // TODO: lane 0 of warp 0 loads from warp_sums, re-reduces, atomicAdds to d_partial[blockIdx.x]

    // Hint: only threads with tid < (BLOCK_SIZE / WARP_SIZE) participate in the cross-warp step
}

int main() {
    const int n = N;
    float* h_in = new float[n];
    for (int i = 0; i < n; ++i) h_in[i] = static_cast<float>(i);

    float* d_in;
    // TODO: cudaMalloc and copy h_in -> d_in

    const int nblocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float* d_partial;
    // TODO: cudaMalloc d_partial[nblocks], initialized to 0

    // TODO: launch block_reduce<<<nblocks, BLOCK_SIZE>>>(d_in, d_partial, n)
    // TODO: reduce d_partial on CPU (copy back and sum) or launch a second kernel

    float gpu_sum = 0.0f;
    // TODO: compute gpu_sum from d_partial

    double expected = (double)n * (n - 1) / 2.0;
    double err = (gpu_sum - expected) / expected;
    printf("GPU sum: %.0f  Expected: %.0f  Relative error: %.2e\n",
           (double)gpu_sum, expected, err);
    printf("Result: %s\n", (err < 1e-5) ? "PASS" : "FAIL");

    // TODO: cudaFree
    delete[] h_in;
    return 0;
}
```

### Expected Output

```
GPU sum: 549755289600  Expected: 549755289600  Relative error: 0.00e+00
Result: PASS
```

### Hints

| Concept | Detail |
|---------|--------|
| Full mask | Use `0xffffffff` for all 32 lanes active |
| `__shfl_down_sync` signature | `float __shfl_down_sync(unsigned mask, float val, int offset, int width=32)` |
| Warp sum storage | `if (lane == 0) warp_sums[wid] = val;` |
| Cross-warp reduce | Only run with `tid < BLOCK_SIZE/WARP_SIZE` threads, load from `warp_sums` |
| Final write | `atomicAdd(d_partial + blockIdx.x, reduced_val)` or direct assignment if tid==0 |

### Performance Target

Should outperform a naive shared-memory-only reduction by approximately 10–20% due to eliminated shared memory reads/writes in the warp-level phase.
