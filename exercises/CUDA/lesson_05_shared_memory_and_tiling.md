# Lesson 5 — Shared Memory and Tiling (per-lesson exercise)

Prerequisites: L03 (thread indexing), L04 (memory model).

Compile: `nvcc -O3 -arch=sm_80 ex.cu -o ex`

Shared memory is a small (48–128 KiB per SM), programmer-managed L1-like cache. The two uses that account for 95% of real-world kernels:

1. **Tiling**: load a block of input into shared memory once, let every thread in the block reuse it.
2. **Inter-thread communication**: threads write partial results and read neighbors' writes.

---

## Exercise 5.1 — Tiled Matrix Transpose

**Difficulty**: ★★★

### Problem

Transpose an `N × N` matrix. A naive implementation reads row-major but writes column-major, generating uncoalesced writes (a performance disaster). Tiling uses shared memory to buffer a block, transposed-locally, so global writes are coalesced again.

### Starter

```cuda
#include <cstdio>
#include <cuda_runtime.h>

#define TILE 32

__global__ void transpose_tiled(const float *in, float *out, int N) {
    __shared__ float tile[TILE][TILE + 1];   // +1 to avoid shared-memory bank conflicts

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    // Load coalesced from global into the tile
    if (x < N && y < N) tile[threadIdx.y][threadIdx.x] = in[y * N + x];
    __syncthreads();

    // Remap indices for output — writes are now coalesced
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    if (x < N && y < N) out[y * N + x] = tile[threadIdx.x][threadIdx.y];
}

int main(void) {
    const int N = 1024;
    size_t bytes = N * N * sizeof(float);
    float *h = new float[N * N];
    for (int i = 0; i < N * N; i++) h[i] = (float)i;

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h, bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    transpose_tiled<<<grid, block>>>(d_in, d_out, N);

    cudaMemcpy(h, d_out, bytes, cudaMemcpyDeviceToHost);
    // Verify: out[i*N+j] should equal original[j*N+i]
    bool ok = true;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            if (h[i * N + j] != (float)(j * N + i)) { ok = false; break; }
    printf("transpose check: %s\n", ok ? "OK" : "FAIL");

    cudaFree(d_in); cudaFree(d_out);
    delete[] h;
    return 0;
}
```

Note the `+1` padding on `tile[TILE][TILE + 1]`. Without it, 32 consecutive reads of `tile[i][threadIdx.x]` all fall in the same bank, serializing the reads 32-way. With the padding, successive threads access different banks.

---

## Exercise 5.2 — Bank-Conflict Benchmark

**Difficulty**: ★★

Remove the `+1` padding and re-run 5.1. Time both versions with `cudaEventElapsedTime`. The conflict-free version should be 2–8× faster on most GPUs. Profile with NVIDIA Nsight Compute and confirm the "Shared Bank Conflicts" counter increases without the padding.

---

## Exercise 5.3 — Tiled Convolution — Bonus

**Difficulty**: ★★★★

Implement a 2D convolution with a 3×3 kernel using tiling. Each block loads a `(TILE+2) × (TILE+2)` input patch (with halo) into shared memory; threads then compute the output for the inner `TILE × TILE` region. Compare performance to a naive implementation where every output pixel re-reads its input.
