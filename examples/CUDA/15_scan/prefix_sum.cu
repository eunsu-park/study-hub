/*
 * prefix_sum.cu — Lesson 15: Parallel Scan (Prefix Sum)
 *
 * Demonstrates:
 *   - Blelloch (work-efficient) exclusive scan — single block
 *   - Multi-block scan using auxiliary array + propagation pass
 *   - Correctness verification against CPU scan
 *
 * Build:  nvcc -O2 -arch=sm_80 prefix_sum.cu -o prefix_sum
 * Run:    ./prefix_sum
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e)); exit(1); } } while(0)

static const int N       = 1 << 20;   // must be power-of-2 for this demo
static const int THREADS = 512;       // threads per block = half of elements per block

// ── Single-block Blelloch exclusive scan ─────────────────────────────────────
// Each block handles 2*THREADS elements.
// Up-sweep: build a reduction tree
// Down-sweep: propagate partial sums to produce exclusive scan
__global__ void block_scan(const float *in, float *out,
                            float *block_sums, int n) {
    extern __shared__ float s[];
    int tid   = threadIdx.x;
    int bid   = blockIdx.x;
    int base  = bid * 2 * blockDim.x;

    // Load two elements per thread
    s[2 * tid]     = (base + 2 * tid     < n) ? in[base + 2 * tid]     : 0.f;
    s[2 * tid + 1] = (base + 2 * tid + 1 < n) ? in[base + 2 * tid + 1] : 0.f;

    int offset = 1;
    // Up-sweep
    for (int d = blockDim.x; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            s[bi] += s[ai];
        }
        offset <<= 1;
    }

    // Save block total and clear last element
    if (tid == 0) {
        if (block_sums) block_sums[bid] = s[2 * blockDim.x - 1];
        s[2 * blockDim.x - 1] = 0.f;
    }

    // Down-sweep
    for (int d = 1; d <= (int)blockDim.x; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            float t = s[ai];
            s[ai]   = s[bi];
            s[bi]  += t;
        }
    }
    __syncthreads();

    // Write results
    if (base + 2 * tid     < n) out[base + 2 * tid]     = s[2 * tid];
    if (base + 2 * tid + 1 < n) out[base + 2 * tid + 1] = s[2 * tid + 1];
}

// ── Add block offset to each element within a block ───────────────────────────
__global__ void add_block_offsets(float *data, const float *offsets, int n) {
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    float off = (blockIdx.x > 0) ? offsets[blockIdx.x] : 0.f;
    if (i     < n) data[i]     += off;
    if (i + blockDim.x < n) data[i + blockDim.x] += off;
}

int main(void) {
    const size_t bytes = (size_t)N * sizeof(float);
    int blocks = N / (2 * THREADS);   // exact division for power-of-2 N

    // Host data
    float *h_in  = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);
    for (int i = 0; i < N; i++) h_in[i] = 1.f;

    // CPU reference (exclusive scan)
    h_ref[0] = 0.f;
    for (int i = 1; i < N; i++) h_ref[i] = h_ref[i-1] + h_in[i-1];

    float *d_in, *d_out, *d_block_sums, *d_scan_block;
    CUDA_CHECK(cudaMalloc(&d_in,         bytes));
    CUDA_CHECK(cudaMalloc(&d_out,        bytes));
    CUDA_CHECK(cudaMalloc(&d_block_sums, blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scan_block, blocks * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    size_t smem = 2 * THREADS * sizeof(float);

    // Pass 1: scan each block, collect block sums
    block_scan<<<blocks, THREADS, smem>>>(d_in, d_out, d_block_sums, N);

    // Pass 2: scan the block sums (single block)
    block_scan<<<1, blocks/2, blocks * sizeof(float)>>>(
        d_block_sums, d_scan_block, nullptr, blocks);

    // Pass 3: add scanned block offsets back
    add_block_offsets<<<blocks, THREADS>>>(d_out, d_scan_block, N);

    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    // Verify
    float max_err = 0.f;
    for (int i = 0; i < N; i++)
        max_err = fmaxf(max_err, fabsf(h_out[i] - h_ref[i]));
    printf("Prefix sum (exclusive, N=%d): max_err=%.1f %s\n",
           N, max_err, max_err < 1.f ? "PASS" : "FAIL");
    printf("  h_out[0..5] = %.0f %.0f %.0f %.0f %.0f\n",
           h_out[0], h_out[1], h_out[2], h_out[3], h_out[4]);

    cudaFree(d_in); cudaFree(d_out);
    cudaFree(d_block_sums); cudaFree(d_scan_block);
    free(h_in); free(h_out); free(h_ref);
    return (max_err < 1.f) ? 0 : 1;
}
