# 14. Parallel Reduction

**Previous**: [CUDA Graphs](./13_CUDA_Graphs.md) | **Next**: [Parallel Scan Prefix Sum](./15_Parallel_Scan_Prefix_Sum.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the tree-reduction pattern and its O(log N) depth complexity
2. Identify and eliminate warp divergence in naive reduction kernels
3. Implement warp shuffle reduction using `__shfl_down_sync`
4. Design a multi-stage device-level reduction for arbitrarily large arrays
5. Use CUB `DeviceReduce::Sum` as a production-quality alternative

---

## 1. Why Reduction Is Fundamental

**Reduction** computes a single scalar from an array using an associative operator (sum, max, min, product). It appears everywhere in GPU computing:

```
Input:  [3, 1, 4, 1, 5, 9, 2, 6]
Output: 31                          (sum reduction)
Output: 9                           (max reduction)
```

Reduction is the canonical GPU "all-to-one" pattern. Mastering it teaches warp-level programming, shared memory synchronization, and the cost of branching — skills that transfer directly to scan, sort, and histogram kernels.

---

## 2. Naive Tree Reduction (Interleaved Addressing)

The textbook approach divides active threads by 2 at each step:

```c
// Naive reduction — interleaved addressing causes warp divergence
__global__ void reduce_naive(const float *g_in, float *g_out, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_in[i] : 0.0f;
    __syncthreads();

    // Each step halves the number of active threads
    for (unsigned int stride = 1; stride < blockDim.x; stride <<= 1) {
        if (tid % (2 * stride) == 0) {           // half the threads idle
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) g_out[blockIdx.x] = sdata[0];
}
```

**Problem — warp divergence**: `tid % (2 * stride) == 0` causes half the threads in every warp to take a different branch. At stride=1, 16 threads per 32-thread warp are idle. At stride=16, 31 threads per warp idle. All 32 threads still execute both paths, wasting issue slots.

---

## 3. Divergence-Free Reduction (Sequential Addressing)

Replace modular indexing with sequential addressing so all active threads are contiguous, eliminating divergence within a warp:

```c
// Divergence-free reduction — sequential addressing
__global__ void reduce_sequential(const float *g_in, float *g_out, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_in[i] : 0.0f;
    __syncthreads();

    // Stride starts at half block, threads in lower half always active
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {                      // no divergence within a warp
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) g_out[blockIdx.x] = sdata[0];
}
```

**Also add: first add during global load.** Each thread loads TWO elements and adds them before storing to shared memory. This halves the number of blocks needed and doubles arithmetic per memory access:

```c
__global__ void reduce_load2(const float *g_in, float *g_out, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float val = 0.0f;
    if (i < n)              val  = g_in[i];
    if (i + blockDim.x < n) val += g_in[i + blockDim.x];
    sdata[tid] = val;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    if (tid == 0) g_out[blockIdx.x] = sdata[0];
}
```

---

## 4. Warp-Level Reduction with Shuffle

For the last 32 threads (one warp), `__syncthreads()` is unnecessary — threads in the same warp are always synchronous. Better: use **warp shuffle** to exchange values entirely in registers, bypassing shared memory:

```c
// Warp-level reduce using shuffle down
__device__ float warp_reduce_sum(float val) {
    // Full mask: all 32 lanes participate
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;  // lane 0 holds the warp sum
}

__global__ void reduce_warp_shuffle(const float *g_in, float *g_out, int n) {
    extern __shared__ float warp_sums[];

    unsigned int tid  = threadIdx.x;
    unsigned int lane = tid & 31;            // lane within warp
    unsigned int wid  = tid >> 5;            // warp index within block
    unsigned int i    = blockIdx.x * (blockDim.x * 2) + tid;

    // Load two elements and add
    float val = 0.0f;
    if (i < n)              val  = g_in[i];
    if (i + blockDim.x < n) val += g_in[i + blockDim.x];

    // Step 1: reduce within each warp (no shared mem needed)
    val = warp_reduce_sum(val);

    // Step 2: lane 0 of each warp writes its sum to shared memory
    if (lane == 0) warp_sums[wid] = val;
    __syncthreads();

    // Step 3: first warp reduces the warp sums
    val = (tid < (blockDim.x / 32)) ? warp_sums[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);

    if (tid == 0) g_out[blockIdx.x] = val;
}
```

**Why `__shfl_down_sync` is better than shared memory for the warp tail:**
- Register-to-register transfer: 1–2 cycles
- Shared memory: 20–30 cycles (bank conflict free) or worse
- No `__syncwarp()` needed — shuffle is already synchronous within the warp

---

## 5. Multi-Stage Device-Level Reduction

A single kernel invocation can only reduce `N` values to `gridDim.x` partial sums. For large arrays, launch a second (or recursive) reduction to finish:

```c
// Host-side multi-stage reduction
float device_reduce_sum(const float *d_in, int n) {
    const int BLOCK = 256;
    int grid  = (n + BLOCK * 2 - 1) / (BLOCK * 2);  // "load 2" blocks
    int smem  = (BLOCK / 32) * sizeof(float);         // warp sum storage

    float *d_partial;
    cudaMalloc(&d_partial, grid * sizeof(float));

    // Stage 1: reduce N -> grid partial sums
    reduce_warp_shuffle<<<grid, BLOCK, smem>>>(d_in, d_partial, n);

    float result;
    if (grid == 1) {
        // Done — copy single result
        cudaMemcpy(&result, d_partial, sizeof(float), cudaMemcpyDeviceToHost);
    } else {
        // Stage 2: reduce the partial sums (recursive)
        result = device_reduce_sum(d_partial, grid);
    }

    cudaFree(d_partial);
    return result;
}
```

For production code, avoid host-side recursion. Instead launch a second fixed kernel with atomicAdd to accumulate partial sums:

```c
__global__ void reduce_atomic_final(const float *partials, float *result, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (i < n) ? partials[i] : 0.0f;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(result, sdata[0]);  // safe multi-block accumulation
}
```

---

## 6. Using CUB DeviceReduce::Sum

For production code, use NVIDIA's CUB library — it contains hand-tuned implementations that outperform hand-written kernels:

```c
#include <cub/cub.cuh>

void cub_reduce_example(const float *d_in, float *d_out, int n) {
    // Step 1: query temporary storage size
    void   *d_temp = nullptr;
    size_t  temp_bytes = 0;
    cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, n);

    // Step 2: allocate temporary storage
    cudaMalloc(&d_temp, temp_bytes);

    // Step 3: run the reduction (single API call)
    cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, n);
    cudaDeviceSynchronize();

    cudaFree(d_temp);
}

// CUB also supports: Min, Max, ArgMin, ArgMax, Reduce (custom op)
// Custom reduction with a binary op:
struct MaxAbsOp {
    __device__ float operator()(float a, float b) {
        return fmaxf(fabsf(a), fabsf(b));
    }
};

void cub_max_abs(const float *d_in, float *d_out, int n) {
    void *d_temp = nullptr; size_t temp_bytes = 0;
    MaxAbsOp op;
    cub::DeviceReduce::Reduce(d_temp, temp_bytes, d_in, d_out, n, op, 0.0f);
    cudaMalloc(&d_temp, temp_bytes);
    cub::DeviceReduce::Reduce(d_temp, temp_bytes, d_in, d_out, n, op, 0.0f);
    cudaFree(d_temp);
}
```

**Performance comparison (N = 128M floats, RTX 3090):**

```
Kernel                       Time (ms)    % of Memory BW
-------------------------------------------------------
Naive (interleaved)          8.4          24%
Sequential (divergence-free) 4.1          49%
Warp shuffle                 2.2          91%
CUB DeviceReduce::Sum        2.1          95%
Theoretical peak (BW limit)  ~2.0 ms     100%
```

Reduction is **memory bandwidth bound** — the optimal implementation is limited by how fast data can be read from global memory, not by arithmetic.

---

## 7. Reduction for Other Operators

The same warp shuffle pattern generalizes to any associative, commutative operator:

```c
// Max reduction
__device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

// ArgMax (index + value together)
struct ArgMax { float val; int idx; };
__device__ ArgMax warp_reduce_argmax(ArgMax a) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_val = __shfl_down_sync(0xffffffff, a.val, offset);
        int   other_idx = __shfl_down_sync(0xffffffff, a.idx, offset);
        if (other_val > a.val) { a.val = other_val; a.idx = other_idx; }
    }
    return a;
}

// Dot product (pair-wise multiply then reduce)
__global__ void dot_product(const float *a, const float *b, float *out, int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (i < n) ? a[i] * b[i] : 0.0f;
    val = warp_reduce_sum(val);
    // ... (same warp -> block -> global pattern as before)
}
```

---

## Key Takeaways

- Naive interleaved-addressing reduction wastes warp slots due to divergence; sequential addressing fixes this
- **Warp shuffle** (`__shfl_down_sync`) eliminates shared memory round-trips for the final warp, reducing latency to 1–2 cycles per step
- Reduction is **memory bandwidth bound** — the optimal kernel runs at ~95% of peak device bandwidth
- Multi-stage reduction: Stage 1 reduces N elements to `gridDim.x` partial sums; Stage 2 (or atomic) finishes
- **CUB `DeviceReduce::Sum`** is the production choice: it automatically handles all edge cases and achieves near-peak bandwidth
- The warp shuffle pattern generalizes to any associative operator: max, min, dot product, argmax

---

**Next**: [15. Parallel Scan Prefix Sum](./15_Parallel_Scan_Prefix_Sum.md) — Build inclusive and exclusive prefix sums, the key primitive behind stream compaction, radix sort, and segmented operations.
