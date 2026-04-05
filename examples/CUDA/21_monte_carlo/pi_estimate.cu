/*
 * pi_estimate.cu — Lesson 21: Monte Carlo Methods
 *
 * Estimates π using the Monte Carlo dartboard method:
 *   π ≈ 4 × (points inside unit circle) / (total points)
 *
 * Demonstrates:
 *   - cuRAND device API (curand_uniform per thread)
 *   - Per-thread RNG state with curandState
 *   - Parallel reduction to aggregate results
 *
 * Build:  nvcc -O2 -arch=sm_80 pi_estimate.cu -o pi_estimate -lcurand
 * Run:    ./pi_estimate
 */

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e)); exit(1); } } while(0)

static const int THREADS     = 256;
static const int BLOCKS      = 256;
static const int SAMPLES_PER = 4096;   // samples per thread
static const long long TOTAL = (long long)THREADS * BLOCKS * SAMPLES_PER;

// ── Initialise one curandState per thread ────────────────────────────────────
__global__ void init_rng(curandState *states, unsigned long long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, id, 0, &states[id]);
}

// ── Monte Carlo π estimation ─────────────────────────────────────────────────
__global__ void mc_pi(curandState *states, unsigned long long *hits) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curandState local_state = states[id];

    unsigned long long count = 0;
    for (int s = 0; s < SAMPLES_PER; s++) {
        float x = curand_uniform(&local_state);
        float y = curand_uniform(&local_state);
        if (x*x + y*y <= 1.f) count++;
    }
    states[id] = local_state;   // write back updated state

    // Reduce within block using shared memory
    __shared__ unsigned long long s[THREADS];
    s[threadIdx.x] = count;
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicAdd(hits, s[0]);
}

int main(void) {
    curandState      *d_states;
    unsigned long long *d_hits;
    CUDA_CHECK(cudaMalloc(&d_states, THREADS * BLOCKS * sizeof(curandState)));
    CUDA_CHECK(cudaMalloc(&d_hits,   sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_hits, 0, sizeof(unsigned long long)));

    // Initialise RNG states
    init_rng<<<BLOCKS, THREADS>>>(d_states, 12345ULL);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    mc_pi<<<BLOCKS, THREADS>>>(d_states, d_hits);

    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);

    unsigned long long h_hits;
    CUDA_CHECK(cudaMemcpy(&h_hits, d_hits, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    double pi_est = 4.0 * (double)h_hits / (double)TOTAL;
    printf("Monte Carlo π estimate\n");
    printf("  Samples  : %lld\n", TOTAL);
    printf("  Hits     : %llu\n", h_hits);
    printf("  π ≈      : %.8f  (error = %.2e)\n", pi_est, fabs(pi_est - M_PI));
    printf("  Time     : %.3f ms\n", ms);

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_states); cudaFree(d_hits);
    return 0;
}
