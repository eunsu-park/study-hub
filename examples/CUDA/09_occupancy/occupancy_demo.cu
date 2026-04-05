/*
 * occupancy_demo.cu — Lesson 09: Occupancy and Launch Configuration
 *
 * Demonstrates:
 *   - cudaOccupancyMaxActiveBlocksPerMultiprocessor
 *   - cudaOccupancyMaxPotentialBlockSize (auto-tuner)
 *   - How register count and shared memory size limit occupancy
 *   - Effect of thread count on latency hiding
 *
 * Build:  nvcc -O2 -arch=sm_80 occupancy_demo.cu -o occupancy_demo
 * Run:    ./occupancy_demo
 */

#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e)); exit(1); } } while(0)

static const int N = 1 << 22;

// ── Compute-bound kernel with tunable register pressure ───────────────────────
// Using __launch_bounds__ tells the compiler the maximum threads-per-block,
// which can reduce register spilling.
template <int REGS>
__global__ __launch_bounds__(1024, 1)
void compute_kernel(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Artificial register pressure: unroll into local variables
    float v = in[i];
    #pragma unroll
    for (int r = 0; r < REGS; r++)
        v = v * v + 0.001f;     // keeps 'v' live → occupies registers
    out[i] = v;
}

// ── Helper: query occupancy for a given block size ────────────────────────────
static void report_occupancy(const void *kernel_fn, int block_size,
                              size_t smem_bytes, int n_sm) {
    int max_blocks;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks, kernel_fn, block_size, smem_bytes));
    float occupancy = (float)(max_blocks * block_size) / 2048.f;   // sm_80: 2048 threads/SM
    printf("  blockSize=%4d  active_blocks/SM=%2d  occupancy=%.0f%%\n",
           block_size, max_blocks, occupancy * 100.f);
}

int main(void) {
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_in, 0, N * sizeof(float)));

    // ── 1. Manual occupancy query for different block sizes ──────────────────
    printf("Occupancy for compute_kernel<4> (sm_80, 2048 threads/SM max):\n");
    for (int bs : {32, 64, 128, 256, 512, 1024})
        report_occupancy((const void *)compute_kernel<4>, bs, 0, 1);

    // ── 2. Auto-tune with cudaOccupancyMaxPotentialBlockSize ─────────────────
    int min_grid, opt_block;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
        &min_grid, &opt_block, compute_kernel<4>, 0, 0));
    printf("\nAuto-tuned block size: %d  (min grid: %d)\n", opt_block, min_grid);

    // ── 3. Benchmark: effect of block size on throughput ─────────────────────
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    printf("\nKernel timing (N=%d):\n", N);
    printf("  %-10s %8s\n", "blockSize", "ms");
    for (int bs : {32, 64, 128, 256, 512, 1024}) {
        int grid = (N + bs - 1) / bs;
        // warmup
        compute_kernel<4><<<grid, bs>>>(d_in, d_out, N);
        cudaEventRecord(t0);
        for (int i = 0; i < 5; i++)
            compute_kernel<4><<<grid, bs>>>(d_in, d_out, N);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms, t0, t1);
        printf("  %-10d %8.3f\n", bs, ms / 5);
    }

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
