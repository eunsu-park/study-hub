# 02. CUDA Programming Model

**Previous**: [GPU Architecture Overview](./01_GPU_Architecture_Overview.md) | **Next**: [Thread Indexing and Grids](./03_Thread_Indexing_and_Grids.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the CUDA thread hierarchy: thread → warp → block → grid
2. Write and launch a CUDA kernel using `<<<grid, block>>>` syntax
3. Allocate and transfer memory with `cudaMalloc`, `cudaFree`, `cudaMemcpy`
4. Implement `vector_add.cu` — the "Hello World" of GPU computing
5. Handle errors with `cudaGetLastError` and interpret error messages

---

## 1. The CUDA Execution Model

CUDA programs run on two processors simultaneously:

```
Host (CPU)                        Device (GPU)
─────────────────                 ─────────────────────────────────
Sequential C/C++ code             Thousands of parallel threads
Launches kernels                  Executes kernels
Manages data transfer             Has its own memory space
```

A **kernel** is a C function that runs on the GPU. Every thread executes the same kernel code but operates on different data — identified by its unique thread index.

---

## 2. Thread Hierarchy

CUDA organizes threads in a three-level hierarchy:

```
Grid (all threads launched by one kernel call)
│
├── Block 0          Block 1          Block 2 ...
│   ├── Thread 0     ├── Thread 0     ├── Thread 0
│   ├── Thread 1     ├── Thread 1     ├── Thread 1
│   ├── Thread 2     ├── Thread 2     ├── Thread 2
│   └── ...          └── ...          └── ...
```

| Level | Variable | Scope | Notes |
|-------|----------|-------|-------|
| **Thread** | `threadIdx.{x,y,z}` | Within a block | Up to 1024 threads per block |
| **Block** | `blockIdx.{x,y,z}` | Within the grid | Executes on one SM |
| **Grid** | `gridDim.{x,y,z}` | Entire kernel | Up to 2³¹−1 blocks |

**Key constraint**: One block runs entirely on **one SM** and cannot be split. Threads within a block can cooperate via **shared memory** and `__syncthreads()`. Threads in different blocks cannot directly communicate.

---

## 3. The `<<<grid, block>>>` Launch Syntax

```c
kernel<<<gridDim, blockDim>>>(args...);
```

- `gridDim`: number of blocks in the grid (`dim3` or `int`)
- `blockDim`: number of threads per block (`dim3` or `int`)

```c
// 1D example: 1 million elements
int N = 1 << 20;            // 1,048,576 elements
int blockSize = 256;        // threads per block (must be multiple of 32)
int gridSize  = (N + blockSize - 1) / blockSize;  // = 4096 blocks

myKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
```

For 2D problems (e.g., matrices):

```c
dim3 block(16, 16);           // 256 threads per block, 2D arrangement
dim3 grid(W/16, H/16);        // one block per 16×16 tile
matmulKernel<<<grid, block>>>(d_A, d_B, d_C, N);
```

---

## 4. Memory Management

GPU memory is separate from CPU memory. You must explicitly allocate and transfer:

```c
// Host (CPU) allocation
float *h_a = (float *)malloc(N * sizeof(float));

// Device (GPU) allocation
float *d_a;
cudaMalloc((void **)&d_a, N * sizeof(float));

// Transfer: Host → Device
cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);

// ... run kernels on GPU ...

// Transfer: Device → Host
cudaMemcpy(h_a, d_a, N * sizeof(float), cudaMemcpyDeviceToHost);

// Free GPU memory
cudaFree(d_a);
free(h_a);
```

### Memory Transfer Cost

```
PCI-e 4.0 ×16:  ~32 GB/s bidirectional
NVLink (A100):  ~600 GB/s GPU–GPU

Transferring 1 GB over PCIe:  ~31 ms
HBM2e bandwidth (A100):       2 TB/s — 62× faster than PCIe
```

**Rule of thumb**: minimize host↔device transfers. Keep data on GPU as long as possible.

---

## 5. Vector Addition — Your First CUDA Kernel

The canonical first CUDA program: add two arrays element-wise.

```c
// vector_add.cu
#include <stdio.h>
#include <cuda_runtime.h>

// Kernel: runs on GPU, called from CPU
__global__ void vector_add(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // global thread index
    if (i < n) {          // boundary guard
        c[i] = a[i] + b[i];
    }
}

int main(void) {
    const int N = 1 << 20;  // 1M elements
    const size_t bytes = N * sizeof(float);

    // Host arrays
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);

    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    // Device arrays
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy input data to GPU
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int gridSize  = (N + blockSize - 1) / blockSize;
    vector_add<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);

    // Copy result back
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Verify
    float max_err = 0.0f;
    for (int i = 0; i < N; i++) {
        float expected = h_a[i] + h_b[i];
        float err = fabsf(h_c[i] - expected);
        if (err > max_err) max_err = err;
    }
    printf("Max error: %e\n", max_err);  // Should be 0.000000e+00

    // Cleanup
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
```

```bash
nvcc -O2 -arch=sm_80 -o vector_add vector_add.cu
./vector_add
# Output: Max error: 0.000000e+00
```

### How the Index Calculation Works

```
blockIdx.x = 2, blockDim.x = 256, threadIdx.x = 37
→ i = 2 * 256 + 37 = 549

Each thread handles exactly one element: c[549] = a[549] + b[549]
```

For N=1,048,576 elements with blockSize=256:
- gridSize = 1,048,576 / 256 = **4,096 blocks**
- Total threads = 4,096 × 256 = **1,048,576** (one per element)

---

## 6. Function Qualifiers

CUDA uses qualifiers to control where code runs:

| Qualifier | Runs on | Called from | Notes |
|-----------|---------|-------------|-------|
| `__global__` | GPU | CPU (or GPU on CC 3.5+) | The kernel — main entry point |
| `__device__` | GPU | GPU only | Helper functions called from kernels |
| `__host__` | CPU | CPU only | Normal C functions (default) |
| `__host__ __device__` | Both | Both | Compiled for both targets |

```c
__device__ float square(float x) { return x * x; }

__global__ void squareKernel(float *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = square(data[i]);  // calls __device__ function
}
```

---

## 7. Error Handling

CUDA functions return `cudaError_t`. Always check errors in production:

```c
// Macro for checking CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Usage
CUDA_CHECK(cudaMalloc(&d_a, bytes));
CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));

// Kernel launches don't return errors directly — check after
myKernel<<<grid, block>>>(args);
CUDA_CHECK(cudaGetLastError());   // catches launch errors
CUDA_CHECK(cudaDeviceSynchronize());  // waits for kernel + catches runtime errors
```

**Why synchronize?** GPU kernels are **asynchronous** — the CPU continues after launching a kernel. `cudaDeviceSynchronize()` blocks the CPU until the GPU finishes.

### Debugging Memory Errors with compute-sanitizer

For deeper runtime error detection — beyond what `CUDA_CHECK` catches — use `compute-sanitizer`:

```bash
# Detect out-of-bounds, uninitialized memory, misaligned access, double-free
compute-sanitizer --tool memcheck ./my_program

# Other available tools:
compute-sanitizer --tool racecheck  ./my_program   # shared memory race conditions
compute-sanitizer --tool initcheck  ./my_program   # uninitialized global memory reads
compute-sanitizer --tool synccheck  ./my_program   # __syncthreads() misuse
```

`compute-sanitizer --tool memcheck` is the CUDA equivalent of Valgrind. It should be run on every new kernel during development — silent out-of-bounds writes are among the hardest CUDA bugs to diagnose without it. Expect a 5–20× slowdown during sanitizer runs.

---

## 8. Measuring Kernel Time with CUDA Events

```c
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
myKernel<<<grid, block>>>(args);
cudaEventRecord(stop);

cudaEventSynchronize(stop);  // wait for stop event

float ms;
cudaEventElapsedTime(&ms, start, stop);
printf("Kernel time: %.3f ms\n", ms);

cudaEventDestroy(start);
cudaEventDestroy(stop);
```

For vector_add with N=1M:

```
Expected output:
  Kernel time: 0.156 ms
  Effective bandwidth: 4 * 1e6 * 4 bytes / 0.156e-3 / 1e9 = 10.3 GB/s

Wait — A100 peak is 2000 GB/s, why only 10 GB/s?
Answer: Vector add is extremely short. The overhead of memory transactions
        for tiny inputs dominates. Try N=100M for realistic bandwidth:
        ≈ 1500 GB/s (75% of peak) — excellent coalesced access.
```

---

## 9. Built-in Variables Cheat Sheet

```c
// Within a __global__ or __device__ function:
threadIdx.x / .y / .z    // thread's position within its block
blockIdx.x  / .y / .z    // block's position within the grid
blockDim.x  / .y / .z    // block dimensions (set at launch)
gridDim.x   / .y / .z    // grid dimensions (set at launch)

// Common 1D index pattern:
int i = blockIdx.x * blockDim.x + threadIdx.x;

// Common 2D index pattern (e.g., matrix element [row][col]):
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
int idx = row * width + col;

// Warp lane (0–31):
int lane = threadIdx.x % 32;

// Warp ID within block:
int warpId = threadIdx.x / 32;
```

---

## 10. Complete Workflow Diagram

```
CPU (Host)                          GPU (Device)
───────────────────────────────     ──────────────────────────────────
malloc(h_a, h_b, h_c)
initialize h_a, h_b
                  ──cudaMalloc─→    allocate d_a, d_b, d_c
                  ──H2D copy──→     d_a = h_a, d_b = h_b
launch kernel<<<>>>                 │
(CPU returns immediately)           ↓
                                    Thread 0: c[0] = a[0]+b[0]
                                    Thread 1: c[1] = a[1]+b[1]
                                    ...
                                    Thread N-1: c[N-1] = a[N-1]+b[N-1]
cudaDeviceSynchronize() ←──────────  (kernel done)
                  ←──D2H copy──    h_c = d_c
verify result
cudaFree, free
```

---

## Key Takeaways

- A **kernel** is launched with `<<<gridDim, blockDim>>>` and runs on thousands of GPU threads simultaneously
- Each thread computes its global index: `i = blockIdx.x * blockDim.x + threadIdx.x`
- **Block size must be a multiple of 32** (warp size); 128 or 256 is the typical sweet spot
- GPU memory is separate — use `cudaMalloc`/`cudaMemcpy`/`cudaFree` to manage it
- Kernel launches are **asynchronous**; use `cudaDeviceSynchronize()` before accessing results
- Always check errors with `CUDA_CHECK()`; silent CUDA failures are a common trap

---

**Next**: [03. Thread Indexing and Grids](./03_Thread_Indexing_and_Grids.md) — Master 1D/2D/3D grid indexing, handle arbitrary array sizes, and implement a matrix transpose kernel.
