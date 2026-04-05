/*
 * cg_reduce.cu — Lesson 31: Cooperative Groups
 *
 * Demonstrates the Cooperative Groups (CG) API for flexible synchronization:
 *   - thread_block_tile<32> for warp-level operations
 *   - this_thread_block() for block-level operations
 *   - Grid-wide sync via cooperative_groups::grid_group (requires -rdc=true)
 *   - CG-based warp reduce and block reduce
 *
 * Build:  nvcc -O2 -arch=sm_80 -rdc=true cg_reduce.cu -o cg_reduce
 * Run:    ./cg_reduce
 */

#include <cstdio>
#include <cmath>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e)); exit(1); } } while(0)

static const int N       = 1 << 22;
static const int THREADS = 256;

// ── CG warp reduce ────────────────────────────────────────────────────────────
// Uses thread_block_tile<32> to avoid hardcoded __shfl_down_sync.
template <int TILE_SZ>
__device__ float warp_reduce_cg(cg::thread_block_tile<TILE_SZ> tile, float v) {
    for (int d = tile.size() / 2; d > 0; d >>= 1)
        v += tile.shfl_down(v, d);
    return v;
}

// ── CG block reduce ───────────────────────────────────────────────────────────
__device__ float block_reduce_cg(cg::thread_block block, float v,
                                  float *smem) {
    auto warp = cg::tiled_partition<32>(block);
    float wsum = warp_reduce_cg(warp, v);

    if (warp.thread_rank() == 0)
        smem[warp.meta_group_rank()] = wsum;
    block.sync();

    // Final reduce across warps (first warp only)
    int n_warps = block.size() / 32;
    float val = (block.thread_rank() < (unsigned)n_warps) ?
                smem[block.thread_rank()] : 0.f;
    if (warp.meta_group_rank() == 0) val = warp_reduce_cg(warp, val);
    return val;
}

// ── Main kernel ───────────────────────────────────────────────────────────────
__global__ void reduce_cg(const float *g_in, float *g_out, int n) {
    extern __shared__ float smem[];

    auto block = cg::this_thread_block();
    int  gid   = block.group_index().x * block.size() + block.thread_rank();
    float v    = (gid < n) ? g_in[gid] : 0.f;

    float bsum = block_reduce_cg(block, v, smem);
    if (block.thread_rank() == 0)
        g_out[block.group_index().x] = bsum;
}

// ── Grid-cooperative kernel (requires SM with cooperative launch support) ─────
// Sums across ALL blocks in one launch (no second kernel call).
__global__ void reduce_cg_grid(const float *g_in, float *g_out, int n) {
    extern __shared__ float smem[];
    auto grid  = cg::this_grid();
    auto block = cg::this_thread_block();

    int  gid = grid.thread_rank();
    float v  = (gid < n) ? g_in[gid] : 0.f;

    float bsum = block_reduce_cg(block, v, smem);
    if (block.thread_rank() == 0)
        smem[block.group_index().x] = bsum;   // store per-block sum in smem
    grid.sync();   // wait for all blocks

    // Block 0 does final reduce
    if (block.group_index().x == 0) {
        float final_v = (block.thread_rank() < (unsigned)gridDim.x) ?
                        smem[block.thread_rank()] : 0.f;
        auto warp = cg::tiled_partition<32>(block);
        final_v   = warp_reduce_cg(warp, final_v);
        if (block.thread_rank() == 0) *g_out = final_v;
    }
}

int main(void) {
    const size_t bytes = (size_t)N * sizeof(float);
    int blocks = (N + THREADS - 1) / THREADS;

    float *h_in = (float *)malloc(bytes);
    for (int i = 0; i < N; i++) h_in[i] = 1.f;

    float *d_in, *d_tmp;
    CUDA_CHECK(cudaMalloc(&d_in,  bytes));
    CUDA_CHECK(cudaMalloc(&d_tmp, blocks * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    size_t smem = THREADS / 32 * sizeof(float);

    // Two-pass reduce using CG
    reduce_cg<<<blocks, THREADS, smem>>>(d_in, d_tmp, N);
    reduce_cg<<<1,      THREADS, smem>>>(d_tmp, d_tmp, blocks);
    cudaDeviceSynchronize();

    float result;
    CUDA_CHECK(cudaMemcpy(&result, d_tmp, sizeof(float), cudaMemcpyDeviceToHost));
    printf("CG two-pass reduce:  sum=%.0f (%s)\n",
           result, fabsf(result - (float)N) < 1.f ? "PASS" : "FAIL");

    // Grid-cooperative reduce (single launch)
    // Check if device supports cooperative launch
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (prop.cooperativeLaunch) {
        float *d_result;
        CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
        // smem must be large enough for all block sums
        size_t coop_smem = blocks * sizeof(float);
        if (coop_smem < smem) coop_smem = smem;

        void *args[] = {&d_in, &d_result, &N};
        CUDA_CHECK(cudaLaunchCooperativeKernel(
            (void*)reduce_cg_grid, blocks, THREADS, args, coop_smem));
        CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
        printf("CG grid-coop reduce: sum=%.0f (%s)\n",
               result, fabsf(result - (float)N) < 1.f ? "PASS" : "FAIL");
        cudaFree(d_result);
    } else {
        printf("CG grid-coop reduce: device does not support cooperative launch\n");
    }

    cudaFree(d_in); cudaFree(d_tmp);
    free(h_in);
    return 0;
}
