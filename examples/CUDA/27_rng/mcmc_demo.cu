/*
 * mcmc_demo.cu — Lesson 27: Random Number Generation and Stochastic Methods
 *
 * Implements the Metropolis-Hastings MCMC algorithm on GPU to sample
 * from a bimodal Gaussian mixture distribution:
 *   p(x) ∝ exp(-0.5*(x-2)²) + exp(-0.5*(x+2)²)
 *
 * Demonstrates:
 *   - cuRAND device API for parallel Markov chains
 *   - Per-thread curandState for independent streams
 *   - Parallel histogram accumulation with atomics
 *
 * Build:  nvcc -O2 -arch=sm_80 mcmc_demo.cu -o mcmc_demo -lcurand
 * Run:    ./mcmc_demo
 */

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e)); exit(1); } } while(0)

static const int CHAINS     = 4096;
static const int WARMUP     = 1000;
static const int SAMPLES    = 4000;
static const int HIST_BINS  = 40;
static const float HIST_MIN = -6.f;
static const float HIST_MAX =  6.f;

__global__ void init_rng(curandState *states, unsigned long long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < CHAINS) curand_init(seed, id, 0, &states[id]);
}

// ── Target: bimodal Gaussian log-density ─────────────────────────────────────
__device__ float log_target(float x) {
    float g1 = -0.5f * (x - 2.f) * (x - 2.f);
    float g2 = -0.5f * (x + 2.f) * (x + 2.f);
    // log-sum-exp for numerical stability
    float m = fmaxf(g1, g2);
    return m + logf(expf(g1 - m) + expf(g2 - m));
}

// ── Metropolis-Hastings chains ────────────────────────────────────────────────
__global__ void mcmc_run(curandState *states, unsigned int *hist,
                          int n_chains, int warmup, int n_samples) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n_chains) return;

    curandState s = states[id];
    float x = curand_normal(&s);   // start near 0

    float step = 1.5f;   // proposal std dev

    // Warmup
    for (int i = 0; i < warmup; i++) {
        float x_prop = x + step * curand_normal(&s);
        float log_r  = log_target(x_prop) - log_target(x);
        if (logf(curand_uniform(&s)) < log_r) x = x_prop;
    }

    // Collect samples
    float bin_width = (HIST_MAX - HIST_MIN) / HIST_BINS;
    for (int i = 0; i < n_samples; i++) {
        float x_prop = x + step * curand_normal(&s);
        float log_r  = log_target(x_prop) - log_target(x);
        if (logf(curand_uniform(&s)) < log_r) x = x_prop;

        int b = (int)((x - HIST_MIN) / bin_width);
        if (b >= 0 && b < HIST_BINS)
            atomicAdd(&hist[b], 1u);
    }
    states[id] = s;
}

int main(void) {
    curandState  *d_states;
    unsigned int *d_hist;
    CUDA_CHECK(cudaMalloc(&d_states, CHAINS * sizeof(curandState)));
    CUDA_CHECK(cudaMalloc(&d_hist,   HIST_BINS * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_hist, 0, HIST_BINS * sizeof(unsigned int)));

    int threads = 256, blocks = (CHAINS + threads - 1) / threads;
    init_rng<<<blocks, threads>>>(d_states, 777ULL);
    mcmc_run<<<blocks, threads>>>(d_states, d_hist, CHAINS, WARMUP, SAMPLES);
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned int h_hist[HIST_BINS];
    CUDA_CHECK(cudaMemcpy(h_hist, d_hist, HIST_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    // Print ASCII histogram
    unsigned int max_cnt = 0;
    for (int b = 0; b < HIST_BINS; b++) if (h_hist[b] > max_cnt) max_cnt = h_hist[b];

    float bin_width = (HIST_MAX - HIST_MIN) / HIST_BINS;
    printf("MCMC bimodal Gaussian (%d chains × %d samples)\n", CHAINS, SAMPLES);
    printf("Distribution (normalised):\n");
    for (int b = 0; b < HIST_BINS; b++) {
        float mid = HIST_MIN + (b + 0.5f) * bin_width;
        int   bar = (int)(40.f * h_hist[b] / max_cnt);
        printf("  %+5.2f |", mid);
        for (int i = 0; i < bar; i++) printf("#");
        printf("\n");
    }

    cudaFree(d_states); cudaFree(d_hist);
    return 0;
}
