# Lesson 2 — CUDA Programming Model (per-lesson exercise)

Prerequisites: basic C / C++.

Compile: `nvcc -O3 -arch=sm_80 ex.cu -o ex`

The CUDA programming model rests on three abstractions: **threads** (the smallest unit of execution), **blocks** (groups of threads that share memory and can synchronize), and **grids** (collections of blocks that the device schedules). The first kernel anyone writes is the right place to internalize these.

---

## Exercise 2.1 — Vector Add ("Hello, GPU")

**Difficulty**: ★

### Problem

Compute `c[i] = a[i] + b[i]` on the GPU for `N = 1 << 20` floats.

### Starter

```cuda
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do {                                                  \
    cudaError_t e = (x);                                                    \
    if (e != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e));         \
        std::exit(1);                                                       \
    }                                                                       \
} while (0)

__global__ void vec_add(const float *a, const float *b, float *c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) c[i] = a[i] + b[i];
}

int main(void) {
    int N = 1 << 20;
    size_t bytes = N * sizeof(float);

    /* Host buffers */
    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_c = new float[N];
    for (int i = 0; i < N; i++) { h_a[i] = i; h_b[i] = 2 * i; }

    /* Device buffers */
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    /* H2D copy */
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    /* Launch */
    int block = 256;
    int grid  = (N + block - 1) / block;
    vec_add<<<grid, block>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());

    /* D2H copy */
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    /* Verify */
    for (int i = 0; i < 5; i++) printf("%.1f + %.1f = %.1f\n", h_a[i], h_b[i], h_c[i]);

    delete[] h_a; delete[] h_b; delete[] h_c;
    CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_b)); CUDA_CHECK(cudaFree(d_c));
    return 0;
}
```

### Verification

The first 5 outputs should be `0+0=0, 1+2=3, 2+4=6, 3+6=9, 4+8=12` — a sanity check that the kernel ran and the H2D/D2H plumbing is correct.

---

## Exercise 2.2 — Querying the Device

**Difficulty**: ★

### Problem

Print the device's properties — these are the hard limits your kernel must respect:

```cuda
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);

printf("Device: %s (compute %d.%d)\n", prop.name, prop.major, prop.minor);
printf("  SMs: %d\n", prop.multiProcessorCount);
printf("  Max threads/block: %d\n", prop.maxThreadsPerBlock);
printf("  Max threads/SM: %d\n", prop.maxThreadsPerMultiProcessor);
printf("  Shared mem/block: %zu KiB\n", prop.sharedMemPerBlock / 1024);
printf("  Global mem: %.1f GiB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
printf("  Memory bus width: %d bits\n", prop.memoryBusWidth);
printf("  Memory clock: %d MHz\n", prop.memoryClockRate / 1000);
printf("  Peak DRAM bandwidth: %.1f GB/s\n",
        prop.memoryClockRate * 1000.0 * (prop.memoryBusWidth / 8) * 2 / 1e9);
```

Memorize: max threads per block is **always** 1024 on any modern NVIDIA GPU. Shared memory per block is 48 KB by default (96 KB with opt-in on Volta+).

---

## Exercise 2.3 — Block Size Sweep — Bonus

**Difficulty**: ★★

Run the vector add from 2.1 with `block ∈ {32, 64, 128, 256, 512, 1024}`. Time each version. The optimum is usually 128 or 256 — small blocks waste resources on per-block overhead, large blocks reduce occupancy.

Plot the throughput curve. The shape — peaked in the middle, falling at extremes — generalizes to almost every CUDA kernel.

---

## Exercise 2.4 — Grid Stride Loop — Bonus

**Difficulty**: ★★

For arrays much larger than a single grid can cover (`N > grid_size * block_size`), the grid-stride loop pattern lets a single launch process arbitrarily many elements:

```cuda
__global__ void vec_add_strided(const float *a, const float *b, float *c, int N) {
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += stride)
        c[i] = a[i] + b[i];
}
```

Launch with a fixed grid size (say 256 blocks of 256 threads = 65k threads) and run for `N = 100 million`. The grid-stride loop iterates each thread through `N / 65000 ≈ 1538` elements. This pattern is more flexible than "match grid to data" because the same kernel works on any input size.
