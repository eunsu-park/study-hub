# Lesson 4 — CUDA Memory Model (per-lesson exercise)

Prerequisites: L02 (programming model), L03 (thread indexing).

Compile: `nvcc -O3 -arch=sm_80 ex.cu -o ex`

CUDA exposes a hierarchy of memory spaces, each with different visibility, latency, and capacity:

| Memory | Scope | Latency | Capacity | Use case |
|--------|-------|---------|----------|----------|
| Register | per thread | 1 cycle | 65536 / SM | scalars, hot loop variables |
| Shared / L1 | per block | ~30 cycles | 48-96 KB / SM | tile buffers, inter-thread comm. |
| Constant | grid-wide read-only | ~30-100 (cached) | 64 KB / device | small lookup tables |
| Global / L2 | grid-wide | ~400 cycles | 4-80 GB / device | bulk data |
| Texture | grid-wide read-only | ~400 (with caching) | global mem-backed | image/lookup with filtering |

Knowing where each piece of data lives and what tier of memory accesses it is the foundation of every CUDA optimization.

---

## Exercise 4.1 — Constant Memory for Filter Coefficients

**Difficulty**: ★★

### Problem

A 1D convolution kernel reads the same filter coefficients for every output pixel. Putting them in `__constant__` memory lets the constant cache serve all threads from a single broadcast read.

```cuda
#include <cstdio>
#include <cuda_runtime.h>

__constant__ float c_filter[5];

__global__ void conv1d_constant(const float *in, float *out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 2 || i >= N - 2) return;
    float s = 0;
    for (int k = 0; k < 5; k++) s += c_filter[k] * in[i + k - 2];
    out[i] = s;
}

int main(void) {
    float h_filter[5] = {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f};
    cudaMemcpyToSymbol(c_filter, h_filter, sizeof(h_filter));

    /* Allocate, launch, etc. */
    return 0;
}
```

For comparison, write a version that passes the filter as a global-memory pointer. On uniform-broadcast access patterns (every thread reads the same address), constant memory is 5-20× faster than uncached global. For non-uniform patterns, the constant cache penalty makes it slower than ordinary global — use only for true broadcast.

---

## Exercise 4.2 — Pinned vs Pageable Host Memory

**Difficulty**: ★★

### Problem

Time `cudaMemcpyAsync` of a 256 MB buffer with:

- `malloc` (pageable host memory) — the driver must DMA through a staging buffer.
- `cudaMallocHost` (pinned/page-locked memory) — direct DMA, ~2× faster.

Pinned memory is also the prerequisite for asynchronous H2D/D2H overlap (CUDA L12). Without pinned, `cudaMemcpyAsync` falls back to synchronous behavior.

The cost of pinned: it cannot be paged out, so over-allocating starves the OS. Treat pinned buffers as a scarce resource; allocate once and reuse.

---

## Exercise 4.3 — Unified Memory

**Difficulty**: ★★

`cudaMallocManaged` allocates memory accessible from both host and device. The driver migrates pages on demand:

```cuda
float *data;
cudaMallocManaged(&data, N * sizeof(float));

/* Host writes */
for (int i = 0; i < N; i++) data[i] = i;

/* Kernel reads — driver page-migrates as needed */
my_kernel<<<grid, block>>>(data, N);
cudaDeviceSynchronize();

/* Host reads */
printf("%.1f\n", data[0]);

cudaFree(data);
```

Compare against the explicit `cudaMalloc` + `cudaMemcpy` version. Unified memory is convenient (no explicit copies) but introduces page-fault overhead. For workloads with small data (< 100 MB), the convenience usually wins; for large bulk transfers, explicit copies are still faster.

---

## Exercise 4.4 — Texture Memory for 2D Lookup — Bonus

**Difficulty**: ★★★

Bilinear-sample a 2D float array using the texture cache. The hardware does the float-coordinate-to-integer-pixel conversion and the bilinear blend in one instruction.

```cuda
texture<float, 2, cudaReadModeElementType> tex;

__global__ void sample_kernel(float *out, float *u, float *v, int N, int W, int H) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    out[i] = tex2D(tex, u[i] * W, v[i] * H);
}
```

For random scattered access into a moderate-sized image, the texture cache (a separate L1 specifically for texture fetches) outperforms loading from global memory through the regular L1 — and you get bilinear interpolation for free.
