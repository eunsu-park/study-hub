# Lesson 31 — Cooperative Groups (per-lesson exercise)

Prerequisites: L05 (shared memory), L14 (reduction).

Compile: `nvcc -O3 -arch=sm_80 ex.cu -o ex`

Cooperative Groups is a CUDA C++ API that gives a uniform way to talk about subgroups of threads — the entire grid, a block, a warp, a 16-thread tile, or a custom group. They replace the older mix of `__syncthreads()`, warp shuffle intrinsics, and `__threadfence()` with one consistent vocabulary.

The two practical wins:
1. Warp-level reductions become one line.
2. Grid-wide synchronization (where all blocks meet) is finally expressible without launching a new kernel.

---

## Exercise 31.1 — Warp Reduce with Cooperative Groups

**Difficulty**: ★★

### Problem

Replace your tree-reduce-then-warp-shuffle pattern (CUDA L14) with a cooperative-group `coalesced_threads().reduce(value, op)`:

```cuda
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__global__ void block_sum_cg(const float *in, float *out, int N) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (gid < N) ? in[gid] : 0.0f;
    __syncthreads();

    /* Tree reduce down to one warp */
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    /* The last warp uses cooperative-group reduce — no more shuffle boilerplate */
    if (tid < 32) {
        cg::thread_block_tile<32> warp = cg::tiled_partition<32>(cg::this_thread_block());
        float val = sdata[tid];
        val = cg::reduce(warp, val, cg::plus<float>());
        if (warp.thread_rank() == 0) out[blockIdx.x] = val;
    }
}
```

Time vs. your hand-rolled CUDA L14 code. They should be within a few percent — Cooperative Groups generates the same shuffle instructions but reads cleaner.

---

## Exercise 31.2 — Tile-Level Sub-Warps

**Difficulty**: ★★

Some algorithms (e.g., 16-element segments) want a smaller-than-warp group. Tiling a warp into 8-thread sub-warps:

```cuda
__global__ void tile_sum(const float *in, float *out, int N) {
    cg::thread_block       block = cg::this_thread_block();
    cg::thread_block_tile<8> tile = cg::tiled_partition<8>(block);

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (gid < N) ? in[gid] : 0.0f;
    v = cg::reduce(tile, v, cg::plus<float>());

    /* Each tile (8 threads) writes one output */
    if (tile.thread_rank() == 0) out[gid / 8] = v;
}
```

Verify against a CPU reference. Each output bin should equal the sum of the corresponding 8 input elements.

This is the building block under sparse-matrix kernels and graph-traversal kernels where work groups are smaller than a full warp.

---

## Exercise 31.3 — Grid Synchronization — Bonus

**Difficulty**: ★★★

The classic CUDA limitation: blocks cannot synchronize without exiting the kernel. Cooperative Groups changes this — if you launch the kernel cooperatively (`cudaLaunchCooperativeKernel`), every block can call `grid.sync()`:

```cuda
__global__ void persistent_kernel(float *data, int N) {
    cg::grid_group grid = cg::this_grid();
    /* Stage 1 */
    int idx = grid.thread_rank();
    if (idx < N) data[idx] = data[idx] * 2.0f;

    grid.sync();   /* All blocks wait here */

    /* Stage 2 */
    if (idx < N) data[idx] = data[idx] + 1.0f;
}
```

Replace a 2-kernel pipeline (kernel A then kernel B with implicit barrier) with a single cooperative kernel. The benefit: no kernel-launch overhead between stages — useful for tight loops in iterative solvers (CG, GMRES).

The constraint: total active blocks must fit on the device simultaneously, so cooperative kernels do not scale to arbitrary grid sizes. Query `cudaOccupancyMaxActiveBlocksPerMultiprocessor` and compare to the number of SMs to know your maximum grid size.
