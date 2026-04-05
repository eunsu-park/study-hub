# 31. Cooperative Groups

**Previous**: [Mixed Precision and Tensor Cores](./30_Mixed_Precision_and_Tensor_Cores.md) | **Next**: [GEMM from Scratch](./32_GEMM_from_Scratch.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Understand the Cooperative Groups programming model and why it improves on `__syncthreads()`
2. Use `cg::thread_block` and `cg::tile_partition<N>` for sub-block synchronization
3. Implement warp-level reductions using `cg::tile_partition<32>` and `group.shfl_down()`
4. Launch grid-cooperative kernels using `cudaLaunchCooperativeKernel` for `cg::grid_group`
5. Use `cg::coalesced_group` to operate on the active (non-diverged) threads within a warp

---

## 1. Why Cooperative Groups?

Traditional CUDA synchronization is rigid:
- `__syncthreads()` always synchronizes all threads in a block
- Warp intrinsics (`__shfl_sync`, `__ballot_sync`) need explicit masks
- No portable way to synchronize subsets of a block

Cooperative Groups (CG) introduced in CUDA 9 provides:
```
thread_block     — all threads in a block (replaces __syncthreads)
tile_partition<N>— groups of N threads within a block (N = power of 2)
grid_group       — all threads across all blocks (requires cooperative launch)
coalesced_group  — currently active (converged) threads within a warp
```

The key benefit: **algorithms parameterized by group size**, not hardcoded to warp/block.

---

## 2. thread_block: Block-Level Synchronization

```c
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void block_reduce_cg(const float *in, float *out, int N) {
    extern __shared__ float sdata[];

    // Obtain handle to this thread block
    cg::thread_block cta = cg::this_thread_block();

    int i   = cta.group_index().x * cta.group_dim().x + cta.thread_index().x;
    int tid = cta.thread_index().x;

    sdata[tid] = (i < N) ? in[i] : 0.f;

    // Block synchronization via CG (equivalent to __syncthreads())
    cta.sync();   // or cg::sync(cta);

    // Reduction within block
    for (unsigned stride = cta.size() / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        cta.sync();
    }

    if (tid == 0) out[cta.group_index().x] = sdata[0];
}
```

This looks the same as `__syncthreads()` but the group object can be passed to helper functions, enabling modular code:

```c
// Generic reduce function — works with any group type that has .sync()
template <typename Group>
__device__ float reduce_sum(Group g, float *shared, float val) {
    int lane = g.thread_rank();
    shared[lane] = val;
    g.sync();

    for (int stride = g.size() / 2; stride > 0; stride >>= 1) {
        if (lane < stride) shared[lane] += shared[lane + stride];
        g.sync();
    }
    return shared[0];
}
```

---

## 3. tile_partition: Sub-Block Groups

`tile_partition<N>` splits a block into fixed-size tiles of N threads (N must be a power of 2, ≤32):

```c
__global__ void warp_level_reduce(const float *in, float *out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Partition block into tiles of 32 (= warp size)
    cg::thread_block  cta  = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(cta);

    float val = (i < N) ? in[i] : 0.f;

    // Warp-level reduction using tile
    for (int offset = warp.size() / 2; offset > 0; offset >>= 1)
        val += warp.shfl_down(val, offset);

    // Lane 0 of each warp holds the partial sum
    if (warp.thread_rank() == 0)
        atomicAdd(out, val);
}

// tile_partition with N < 32 for finer-grained synchronization
__global__ void group8_example(const int *data, int *out, int N) {
    cg::thread_block cta = cg::this_thread_block();
    cg::thread_block_tile<8> g8 = cg::tiled_partition<8>(cta);

    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = g8.thread_rank();   // 0..7

    // Each group of 8 threads does an independent mini-reduction
    int val = (i < N) ? data[i] : 0;
    for (int s = g8.size()/2; s > 0; s >>= 1)
        val += g8.shfl_down(val, s);

    if (lane == 0)
        atomicAdd(out, val);
}
```

---

## 4. group.shfl_down, shfl_up, shfl_xor

CG tiles provide `shfl_*` operations matching CUDA warp intrinsics but typed through the group:

```c
__device__ float warp_reduce_max(cg::thread_block_tile<32> warp, float val) {
    // Warp-level max reduction using shfl_down
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, warp.shfl_down(val, offset));
    return val;   // only lane 0 has the true maximum
}

__device__ float warp_scan_inclusive(cg::thread_block_tile<32> warp, float val) {
    // Inclusive scan: each lane gets prefix sum up to (and including) its lane
    for (int offset = 1; offset < 32; offset <<= 1) {
        float tmp = warp.shfl_up(val, offset);
        if (warp.thread_rank() >= offset) val += tmp;
    }
    return val;
}

// Butterfly reduction (uses shfl_xor for balanced communication)
__device__ float warp_reduce_sum_butterfly(cg::thread_block_tile<32> warp, float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += warp.shfl_xor(val, offset);
    return val;  // all 32 lanes hold the total sum (broadcast result)
}
```

---

## 5. grid_group: Grid-Wide Synchronization

`grid_group` synchronizes all threads across all blocks. Requires a **cooperative launch**:

```c
#include <cooperative_groups.h>

// Kernel using grid_group for global barrier
__global__ void grid_reduce_two_pass(const float *in, float *partial, float *out, int N) {
    cg::grid_group grid = cg::this_grid();

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // --- Pass 1: local block reduction ---
    extern __shared__ float sdata[];
    sdata[threadIdx.x] = (i < N) ? in[i] : 0.f;
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[blockIdx.x] = sdata[0];

    // --- Global barrier: all blocks must finish pass 1 ---
    grid.sync();   // requires cooperative launch!

    // --- Pass 2: block 0 reduces the partial sums ---
    if (blockIdx.x == 0) {
        sdata[threadIdx.x] = (threadIdx.x < gridDim.x)
                             ? partial[threadIdx.x] : 0.f;
        __syncthreads();
        for (int s = blockDim.x/2; s > 0; s >>= 1) {
            if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0) *out = sdata[0];
    }
}

// Launch with cooperative kernel API
void launch_grid_reduce(const float *d_in, float *d_partial, float *d_out, int N) {
    int block = 256;
    int grid  = (N + block - 1) / block;

    // Check device supports cooperative launch
    int can_cooperative;
    cudaDeviceGetAttribute(&can_cooperative, cudaDevAttrCooperativeLaunch, 0);
    if (!can_cooperative) { fprintf(stderr, "No cooperative launch support\n"); return; }

    void *args[] = { (void*)&d_in, (void*)&d_partial, (void*)&d_out, (void*)&N };
    size_t shared = block * sizeof(float);

    cudaLaunchCooperativeKernel(
        (void*)grid_reduce_two_pass,
        grid, block, args, shared, nullptr);
}
```

---

## 6. coalesced_group: Active Threads Only

`cg::coalesced_threads()` captures only the currently active (non-diverged) threads in a warp. Useful for branch-heavy kernels:

```c
// Process only threads that satisfy a condition, using active group
__global__ void process_active_only(int *data, int *out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    if (data[i] > 0) {
        // Form a group of just the threads that took this branch
        cg::coalesced_group active = cg::coalesced_threads();

        // Reduce across active threads only (no idle threads wasting work)
        float val = (float)data[i];
        for (int offset = active.size()/2; offset > 0; offset >>= 1)
            val += active.shfl_down(val, offset);

        if (active.thread_rank() == 0)
            atomicAdd(out, (int)val);
    }
}

// labeled_partition: split active threads by a label (e.g., thread color)
__global__ void labeled_example(int *colors, float *vals, float *out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int   color = colors[i];
    float val   = vals[i];

    // Group threads with the same color within this warp
    cg::coalesced_group same_color = cg::labeled_partition(
        cg::coalesced_threads(), color);

    // Reduce within same-color group
    for (int off = same_color.size()/2; off > 0; off >>= 1)
        val += same_color.shfl_down(val, off);

    if (same_color.thread_rank() == 0)
        atomicAdd(&out[color], val);
}
```

---

## 7. Flexible Warp Reduction Utility

Combining CG with templates gives a truly flexible reduction function:

```c
// Works for tile_partition<32>, tile_partition<16>, coalesced_group, etc.
template <typename GroupT>
__device__ float group_reduce_sum(GroupT g, float val) {
    for (int offset = g.size() / 2; offset > 0; offset >>= 1)
        val += g.shfl_down(val, offset);
    return val;  // valid only on thread_rank()==0 for shfl_down
}

template <typename GroupT>
__device__ float group_reduce_max(GroupT g, float val) {
    for (int offset = g.size() / 2; offset > 0; offset >>= 1)
        val = fmaxf(val, g.shfl_down(val, offset));
    return val;
}

// Example: multi-stage reduction (warp → block → grid)
__global__ void multi_stage_reduce(const float *in, float *out, int N) {
    cg::thread_block  cta  = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(cta);
    extern __shared__ float sdata[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (i < N) ? in[i] : 0.f;

    // Stage 1: warp reduce
    val = group_reduce_sum(warp, val);

    // Stage 2: block reduce (one value per warp)
    if (warp.thread_rank() == 0)
        sdata[threadIdx.x / 32] = val;
    cta.sync();

    if (threadIdx.x < cta.size() / 32) {
        val = sdata[threadIdx.x];
        cg::thread_block_tile<8> last_group = cg::tiled_partition<8>(cta);
        val = group_reduce_sum(last_group, val);
    }

    if (threadIdx.x == 0) out[blockIdx.x] = val;
}
```

---

## Key Takeaways

- **Cooperative Groups** replaces hardcoded `__syncthreads()` with composable group objects: pass them to helper functions for modular, reusable code
- **`cg::this_thread_block()`** returns a handle to the full thread block; `.sync()` is equivalent to `__syncthreads()` but also carries group size and rank metadata
- **`cg::tiled_partition<N>(cta)`** creates a sub-block group of N threads; enables sub-warp and sub-block synchronization patterns
- **`group.shfl_down(val, offset)`** performs a warp shuffle within the tile; works identically to `__shfl_down_sync` but without manual mask management
- **`cg::this_grid()`** with `.sync()` provides a global grid barrier, but requires `cudaLaunchCooperativeKernel` and a device that supports cooperative launch
- **`cg::coalesced_threads()`** captures only the active threads in a warp after divergence; `labeled_partition()` further groups active threads by a key value

---

**Next**: [32. GEMM from Scratch](./32_GEMM_from_Scratch.md) — Build a high-performance matrix multiply kernel step by step, from a naive baseline to a register-tiled, float4-vectorized implementation approaching cuBLAS performance.
