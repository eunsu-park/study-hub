# 04. CUDA Memory Model

**Previous**: [Thread Indexing and Grids](./03_Thread_Indexing_and_Grids.md) | **Next**: [Shared Memory and Tiling](./05_Shared_Memory_and_Tiling.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Describe all GPU memory types: global, shared, registers, L1/L2, constant, texture
2. Choose the right memory type for a given access pattern
3. Benchmark memory bandwidth with a custom kernel
4. Explain why misuse of memory types causes performance cliffs
5. Use `cudaMemcpyToSymbol` for constant memory

---

## 1. The GPU Memory Hierarchy

GPU programs live and die by their relationship with memory. Understanding the hierarchy is essential for performance:

```
Fastest ────────────────────────────────────────────── Slowest

  Registers    Shared/L1    L2 Cache    Global (HBM/GDDR)
  ─────────    ─────────    ────────    ─────────────────
  ~0 cycles    1–5 cycles   50 cycles   400–700 cycles
  256 KB/SM    48–228 KB    40–80 MB    40–80 GB total
  ~19 TB/s     ~19 TB/s     ~5 TB/s     ~2 TB/s (A100)
  per-thread   per-block    device      device
```

Each memory type has a different **scope**, **lifetime**, and **access pattern**.

---

## 2. Global Memory

The main GPU DRAM (HBM or GDDR). All threads can read/write:

```c
__global__ void kernel(float *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = data[i] * 2.0f;  // global memory read + write
    }
}
```

**Properties**:
- Allocated with `cudaMalloc`, transferred with `cudaMemcpy`
- Survives for the lifetime of the application
- Cached in L2 (and L1 for read-only accesses in Volta+)
- **Coalescing is critical** — adjacent threads must access adjacent addresses to combine into one 128-byte transaction

**Bandwidth benchmark**:

```c
__global__ void bw_benchmark(float *in, float *out, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (long)gridDim.x * blockDim.x)
        out[i] = in[i];  // copy — 1 read + 1 write per element
}

// Expected: close to device's peak memory bandwidth
// A100: ~2000 GB/s peak, achievable: ~1800 GB/s (90%)
```

---

## 3. Shared Memory

Per-block fast scratchpad, explicitly managed by the programmer:

```c
__global__ void reduction_kernel(float *data, float *result, int n) {
    __shared__ float smem[256];  // declared in the kernel — static allocation

    int tid  = threadIdx.x;
    int i    = blockIdx.x * blockDim.x + tid;

    smem[tid] = (i < n) ? data[i] : 0.0f;  // load from global → shared
    __syncthreads();                         // wait for all threads

    // Reduction within the block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid == 0) result[blockIdx.x] = smem[0];  // write block result
}
```

**Properties**:
- **Static**: `__shared__ float buf[256];` — size known at compile time
- **Dynamic**: `extern __shared__ float buf[];` + third kernel launch parameter

```c
// Dynamic shared memory — size specified at launch:
reduction_kernel<<<grid, block, sharedBytes>>>(args);
```

- Lifetime: one kernel invocation
- Scope: all threads in the **same block**
- Speed: same as L1 cache (~19 TB/s, ~1–5 cycles)
- **Bank conflicts** reduce bandwidth — see L05 for the full analysis

**Configuring shared memory / L1 ratio** (Ampere+ has a unified pool):

```c
// Set 48 KB shared memory (default) or up to 228 KB with runtime API
cudaFuncSetAttribute(myKernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize, 131072);  // 128 KB
```

---

## 4. Registers

The fastest storage — per-thread private variables with zero-cycle access:

```c
__global__ void kernel(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // register
    float x = a[i];   // register — loaded once from global
    float y = b[i];   // register
    float result = x * x + y * y;  // all register arithmetic
    c[i] = result;    // write back to global
}
```

**Properties**:
- Private to each thread — not accessible by other threads
- 64K 32-bit registers per SM on Ampere
- Too many registers per thread → **register spilling** to local memory (global memory with high latency)
- Check register usage: `nvcc -Xptxas -v kernel.cu` shows `Used N registers`

**Register pressure**: the main constraint on occupancy.

```
Registers per thread = 32 → max 2048 threads/SM (64K/32) = 64 warps → 100% occupancy
Registers per thread = 64 → max 1024 threads/SM (64K/64) = 32 warps → 50% occupancy
Registers per thread = 128 → 512 threads/SM = 16 warps → 25% occupancy
```

---

## 5. Constant Memory

Read-only memory, cached in a dedicated cache (64 KB), fast for **broadcast access** (all threads in a warp read the same address):

```c
// Declare at file scope (device-visible)
__constant__ float filter[64];   // e.g., convolution weights

__global__ void conv_kernel(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float sum = 0.0f;
        for (int k = 0; k < 64; k++)
            sum += filter[k] * in[i + k];  // all threads read filter[k] → broadcast
        out[i] = sum;
    }
}

// Initialize from CPU:
cudaMemcpyToSymbol(filter, h_filter, 64 * sizeof(float));
```

**When to use**: weights, lookup tables, constants that all threads share simultaneously.

**When NOT to use**: if different threads in a warp read different indices → serialization (32× slower than broadcast). Use global memory + L2 cache instead.

---

## 6. Texture Memory

Read-only, cached, hardware-accelerated interpolation and 2D/3D spatial locality:

```c
// Texture object API (CUDA 5.0+)
cudaTextureObject_t tex;

cudaResourceDesc resDesc;
memset(&resDesc, 0, sizeof(resDesc));
resDesc.resType = cudaResourceTypeLinear;
resDesc.res.linear.devPtr = d_data;
resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
resDesc.res.linear.desc.x = 32;  // 32-bit float
resDesc.res.linear.sizeInBytes = N * sizeof(float);

cudaTextureDesc texDesc;
memset(&texDesc, 0, sizeof(texDesc));
texDesc.readMode = cudaReadModeElementType;

cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

__global__ void texture_kernel(cudaTextureObject_t tex, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = tex1Dfetch<float>(tex, i);
}
```

**Best for**: image processing (spatial locality, boundary clamping, bilinear interpolation), irregular access patterns where the texture cache outperforms L1.

In modern code (Volta+), `__ldg()` (load via L1 read-only cache) is usually sufficient:

```c
float val = __ldg(&d_data[i]);  // read-only, cached in L1 read-only cache
```

---

## 7. Local Memory

A misnomer — **local memory is actually global memory**, used when registers overflow:

```c
__global__ void large_array_kernel(float *out) {
    float local_arr[1024];  // too large for registers → spills to local memory
    // ... (slow — 400+ cycle latency per access)
}
```

The compiler automatically handles spilling. Use `nvcc -Xptxas -v` to detect it:
```
ptxas info: Function properties for myKernel
  24 bytes stack frame, 24 bytes spill stores, 24 bytes spill loads
```
Any non-zero spill count is a performance warning.

---

## 8. Memory Access Patterns Summary

| Memory | Scope | Latency | BW | Size | Managed by |
|--------|-------|---------|-----|------|------------|
| **Registers** | Per thread | 0 | ~19 TB/s | 256 KB/SM | Compiler |
| **Shared** | Per block | 1–5 cy | ~19 TB/s | 48–228 KB/SM | Programmer |
| **L1 Cache** | Per SM | 1–5 cy | ~19 TB/s | 32 KB (part of SM) | Hardware |
| **L2 Cache** | Device | 50 cy | ~5 TB/s | 40–80 MB | Hardware |
| **Global** | Device | 400–700 cy | ~2 TB/s | 40–80 GB | Programmer |
| **Constant** | Device | 1–5 cy (broadcast) | Fast | 64 KB | Programmer |
| **Texture** | Device | 1–5 cy (cached) | Moderate | 48 KB cache | Programmer |
| **Local** | Per thread | 400–700 cy | ~2 TB/s | 512 KB/thread | Compiler |

---

## 9. Pinned (Page-Locked) Host Memory

Standard `malloc` allocates **pageable** memory that the OS can swap. For faster PCIe transfers, use **pinned** memory:

```c
float *h_pinned;
cudaHostAlloc(&h_pinned, bytes, cudaHostAllocDefault);

// Transfer is ~2× faster (3–12 GB/s vs 6–12 GB/s on PCIe 4.0)
cudaMemcpy(d_data, h_pinned, bytes, cudaMemcpyHostToDevice);

cudaFreeHost(h_pinned);  // must use cudaFreeHost, not free()
```

**Tradeoff**: pinned memory cannot be swapped out, consuming physical RAM. Don't pin excessively.

---

## 10. Unified Memory (UM)

CUDA 6.0+ allows a single pointer accessible by both CPU and GPU:

```c
float *data;
cudaMallocManaged(&data, bytes);  // single allocation

// CPU can read/write:
for (int i = 0; i < N; i++) data[i] = (float)i;

// GPU can read/write (after kernel call):
myKernel<<<grid, block>>>(data, N);
cudaDeviceSynchronize();

// CPU reads result:
printf("data[0] = %f\n", data[0]);

cudaFree(data);
```

**UM is convenient for prototyping** but may be slower due to automatic page migration. On NVLink systems (e.g., A100 SXM with direct host-GPU memory access), UM performance can be excellent. For production, prefer explicit `cudaMemcpy`.

### Improving Unified Memory Performance: Prefetching and Advise

Automatic page migration triggers on first access (page fault), adding latency. Two APIs eliminate this:

```c
// Prefetch: migrate pages to the device before the kernel launches
// Avoids page faults during kernel execution
cudaMemPrefetchAsync(data, bytes, deviceId, stream);
myKernel<<<grid, block, 0, stream>>>(data, N);

// MemAdvise: hint the driver about access patterns
// ReadMostly — driver may create read-only copies on multiple processors
cudaMemAdvise(data, bytes, cudaMemAdviseSetReadMostly, deviceId);

// PreferredLocation — keep pages on this device unless explicitly migrated
cudaMemAdvise(data, bytes, cudaMemAdviseSetPreferredLocation, deviceId);

// AccessedBy — map pages into the device's page table without migrating
// (useful when CPU and GPU both access the data frequently)
cudaMemAdvise(data, bytes, cudaMemAdviseSetAccessedBy, deviceId);
```

On Pascal+ GPUs, combining `cudaMemPrefetchAsync` with `cudaMemAdvise` can bring Unified Memory performance within 5–10% of explicit `cudaMemcpy` workflows.

---

## Key Takeaways

- **Global memory** (HBM) is large but slow — coalesced access is mandatory for performance
- **Shared memory** is the programmer's cache — use it to stage data that multiple threads reuse
- **Registers** are free in terms of latency but limited in count; spilling kills performance
- **Constant memory** excels for small, broadcast-read data (weights, filter coefficients)
- Use `cudaHostAlloc` for pinned memory to accelerate host↔device transfers
- The memory hierarchy pyramid: registers > shared/L1 > L2 > global — optimize from the bottom up

---

**Next**: [05. Shared Memory and Tiling](./05_Shared_Memory_and_Tiling.md) — Use shared memory to build a tiled matrix multiply, eliminate bank conflicts, and profile with Nsight Compute.
