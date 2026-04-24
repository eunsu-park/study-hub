# Lesson 27 — Random Number Generation and Stochastic Methods (per-lesson exercise)

Prerequisites: L04 (memory model), L09 (occupancy).

Compile: `nvcc -O3 -arch=sm_80 ex.cu -lcurand -o ex`

GPU random number generation requires a different approach than CPU `rand()`. Each thread needs its own state (otherwise they all produce the same sequence), and the algorithm has to be parallel-friendly. NVIDIA provides cuRAND for this.

---

## Exercise 27.1 — cuRAND Per-Thread Initialization

**Difficulty**: ★★

### Problem

Initialize one cuRAND state per thread, then have each thread generate 1000 random floats and compute their mean. Verify that the means cluster around 0.5 (uniform on [0, 1)) within statistical error.

### Starter

```cuda
#include <cstdio>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void init_rng(curandState *states, unsigned long long seed, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) curand_init(seed, idx, 0, &states[idx]);
}

__global__ void rand_means(curandState *states, float *out, int n_per_thread, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    curandState s = states[idx];
    float sum = 0;
    for (int k = 0; k < n_per_thread; k++) sum += curand_uniform(&s);
    states[idx] = s;
    out[idx] = sum / n_per_thread;
}

int main(void) {
    int N = 1024;
    curandState *d_states;
    float *d_means, h_means[1024];

    cudaMalloc(&d_states, N * sizeof(curandState));
    cudaMalloc(&d_means, N * sizeof(float));

    init_rng<<<(N + 255) / 256, 256>>>(d_states, /*seed*/ 1234ULL, N);
    rand_means<<<(N + 255) / 256, 256>>>(d_states, d_means, /*n_per_thread*/ 1000, N);

    cudaMemcpy(h_means, d_means, N * sizeof(float), cudaMemcpyDeviceToHost);
    double overall = 0; for (int i = 0; i < N; i++) overall += h_means[i];
    printf("overall mean = %.4f (expected ≈ 0.5)\n", overall / N);

    cudaFree(d_states); cudaFree(d_means);
    return 0;
}
```

The standard error of a mean of 1000 uniform samples is roughly $1 / \sqrt{12000} \approx 0.0091$, so individual thread means should fall within about 3× that of 0.5.

---

## Exercise 27.2 — Monte Carlo Estimate of π

**Difficulty**: ★★

The classic stochastic exercise: estimate π by sampling points in the unit square and counting the fraction that fall inside the unit quarter-circle:

$$\pi \approx 4 \cdot \frac{N_{\text{inside}}}{N_{\text{total}}}$$

Each thread generates a batch of points and atomically adds its inside-count to a global counter. Run with $N = 10^8$ and verify accuracy is roughly $1/\sqrt{N} \approx 10^{-4}$.

```cuda
__global__ void monte_carlo_pi(curandState *states,
                               unsigned long long *inside,
                               int n_per_thread) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState s = states[tid];
    unsigned long long local = 0;
    for (int k = 0; k < n_per_thread; k++) {
        float x = curand_uniform(&s);
        float y = curand_uniform(&s);
        if (x*x + y*y <= 1.0f) local++;
    }
    states[tid] = s;
    atomicAdd(inside, local);
}
```

For comparison, write the equivalent CPU loop in a single core. The GPU should be 50–500× faster depending on hardware.

---

## Exercise 27.3 — Box-Muller for Gaussian Samples — Bonus

**Difficulty**: ★★

`curand_normal` is available, but implementing Box-Muller from two uniform samples is a useful exercise:

$$z_1 = \sqrt{-2\ln u_1} \cdot \cos(2\pi u_2), \quad z_2 = \sqrt{-2\ln u_1} \cdot \sin(2\pi u_2)$$

Implement a kernel that generates `2N` $\mathcal{N}(0,1)$ samples using $N$ Box-Muller transforms. Compute the empirical mean and variance and confirm they are 0 and 1 to within statistical error.

---

## Exercise 27.4 — Reproducibility — Bonus

**Difficulty**: ★

Run 27.1 twice with the same seed and confirm bit-exact reproducibility. Then run with `seed + 1` and observe completely different outputs. This is the "PRNG hygiene" fact that lets you ship reproducible ML experiments — same seed, same hardware, same driver = same output.
