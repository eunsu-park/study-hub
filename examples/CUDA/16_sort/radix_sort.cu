/*
 * radix_sort.cu — Lesson 16: Parallel Sort
 *
 * Implements a simple LSD (least-significant-digit) radix sort on GPU:
 *   - 1 pass per 2-bit digit (16 passes for 32-bit keys)
 *   - Each pass: histogram → exclusive scan → scatter
 *   - Correctness verified against std::sort
 *
 * For production use, prefer CUB::DeviceRadixSort (Lesson 28).
 *
 * Build:  nvcc -O2 -arch=sm_80 radix_sort.cu -o radix_sort
 * Run:    ./radix_sort
 */

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e)); exit(1); } } while(0)

static const int N       = 1 << 20;   // 1 M keys
static const int THREADS = 256;
static const int RADIX   = 4;         // bits per pass
static const int BUCKETS = (1 << RADIX);  // 16 buckets

// ── Pass 1: compute per-bucket histogram ────────────────────────────────────
__global__ void histogram(const unsigned *keys, unsigned *hist,
                           int n, int bit_shift) {
    __shared__ unsigned s_hist[BUCKETS];
    if (threadIdx.x < BUCKETS) s_hist[threadIdx.x] = 0;
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned digit = (keys[i] >> bit_shift) & (BUCKETS - 1);
        atomicAdd(&s_hist[digit], 1u);
    }
    __syncthreads();
    if (threadIdx.x < BUCKETS)
        atomicAdd(&hist[threadIdx.x], s_hist[threadIdx.x]);
}

// ── Pass 2: exclusive prefix sum on histogram (single thread, small array) ──
__global__ void exclusive_scan_hist(unsigned *hist) {
    // Only 16 buckets — run on a single thread for simplicity
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned prefix = 0;
        for (int b = 0; b < BUCKETS; b++) {
            unsigned cnt = hist[b];
            hist[b] = prefix;
            prefix += cnt;
        }
    }
}

// ── Pass 3: scatter keys to output array ─────────────────────────────────────
__global__ void scatter(const unsigned *keys_in, unsigned *keys_out,
                        unsigned *hist, int n, int bit_shift) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned digit = (keys_in[i] >> bit_shift) & (BUCKETS - 1);
        unsigned pos   = atomicAdd(&hist[digit], 1u);
        keys_out[pos]  = keys_in[i];
    }
}

int main(void) {
    unsigned *h_keys = (unsigned *)malloc(N * sizeof(unsigned));
    unsigned *h_ref  = (unsigned *)malloc(N * sizeof(unsigned));
    for (int i = 0; i < N; i++) {
        h_keys[i] = (unsigned)rand();
        h_ref[i]  = h_keys[i];
    }
    std::sort(h_ref, h_ref + N);

    unsigned *d_keys, *d_tmp, *d_hist;
    CUDA_CHECK(cudaMalloc(&d_keys, N * sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&d_tmp,  N * sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&d_hist, BUCKETS * sizeof(unsigned)));
    CUDA_CHECK(cudaMemcpy(d_keys, h_keys, N * sizeof(unsigned), cudaMemcpyHostToDevice));

    int blocks = (N + THREADS - 1) / THREADS;

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    for (int bit = 0; bit < 32; bit += RADIX) {
        CUDA_CHECK(cudaMemset(d_hist, 0, BUCKETS * sizeof(unsigned)));
        histogram       <<<blocks, THREADS>>>(d_keys, d_hist, N, bit);
        exclusive_scan_hist<<<1, 1>>>(d_hist);
        scatter         <<<blocks, THREADS>>>(d_keys, d_tmp, d_hist, N, bit);
        // swap pointers
        unsigned *tmp = d_keys; d_keys = d_tmp; d_tmp = tmp;
    }

    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);

    // Copy result back (d_keys points to sorted array after even # of swaps)
    CUDA_CHECK(cudaMemcpy(h_keys, d_keys, N * sizeof(unsigned), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < N; i++) if (h_keys[i] != h_ref[i]) { ok = false; break; }
    printf("Radix sort (N=%d, %d-bit RADIX): %.3f ms  %s\n",
           N, RADIX, ms, ok ? "PASS" : "FAIL");

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_keys); cudaFree(d_tmp); cudaFree(d_hist);
    free(h_keys); free(h_ref);
    return ok ? 0 : 1;
}
