# 27. Random Numbers and Stochastic Methods

**Previous**: [Image Processing GPU](./26_Image_Processing_GPU.md) | **Next**: [Thrust and CUB](./28_Thrust_and_CUB.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Use the cuRAND host API to generate large batches of random numbers efficiently
2. Use cuRAND device API inside kernels for per-thread random streams
3. Implement parallel Monte Carlo integration and Metropolis-Hastings MCMC on the GPU
4. Simulate the 2D Ising model using a checkerboard Metropolis update
5. Apply variance reduction techniques (antithetic variates, importance sampling) to reduce statistical error

---

## 1. cuRAND Overview

cuRAND provides GPU-accelerated random number generation. It supports two usage modes:

```
Host API:
  curandCreateGenerator() → generate directly into device memory
  Pros: simple, handles all state management
  Cons: entire batch generated at once (large memory footprint)

Device API:
  curand_init() initializes per-thread state
  curand_uniform() generates one number per call inside a kernel
  Pros: on-the-fly generation, no extra storage needed
  Cons: state initialization cost (~100 cycles); state is 48-192 bytes/thread
```

**Supported generators:**

| Generator | Period | Quality | Speed |
|-----------|--------|---------|-------|
| XORWOW    | ~2^190 | Good    | Fastest |
| Philox4   | 2^128  | High    | Fast |
| MRG32k3a  | ~2^191 | High    | Moderate |
| MTGP32    | 2^11213| Very high | Moderate (shared memory based) |
| Sobol32   | Quasi  | QMC     | Fast (low discrepancy) |

---

## 2. Host API: Batch Generation

```c
#include <curand.h>

void generate_uniform_host(float **d_rand, int N) {
    curandGenerator_t gen;

    // Create generator
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW);

    // Set seed (for reproducibility)
    curandSetPseudoRandomGeneratorSeed(gen, 12345ULL);

    // Allocate device memory
    cudaMalloc(d_rand, N * sizeof(float));

    // Generate N uniform floats in [0, 1)
    curandGenerateUniform(gen, *d_rand, N);

    // Generate normal distribution N(0, 1)
    float *d_normal;
    cudaMalloc(&d_normal, N * sizeof(float));
    curandGenerateNormal(gen, d_normal, N, 0.0f, 1.0f);  // mean=0, std=1

    // Box-Muller requires even N; for log-normal:
    // curandGenerateLogNormal(gen, d_log, N, mean, std);

    curandDestroyGenerator(gen);
    cudaFree(d_normal);
}

// Quasi-random Sobol sequences (low discrepancy — better convergence than pseudo-random)
void generate_sobol(float **d_sobol, int N, int dims) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_QUASI_SOBOL32);
    curandSetQuasiRandomGeneratorDimensions(gen, dims);
    cudaMalloc(d_sobol, N * dims * sizeof(float));
    curandGenerateUniform(gen, *d_sobol, N * dims);
    curandDestroyGenerator(gen);
}
```

---

## 3. Device API: Per-Thread Generation

```c
#include <curand_kernel.h>

// Initialize one curand state per thread
// Call once; save state to global memory for reuse across kernel launches
__global__ void init_rng(curandState *states, int N, unsigned long long seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    // sequence = tid ensures each thread has an independent stream
    curand_init(seed, /*sequence=*/tid, /*offset=*/0, &states[tid]);
}

// Monte Carlo pi estimation: count points inside unit circle
__global__ void monte_carlo_pi(curandState *states, int *counts, int samples_per_thread) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState local_state = states[tid];  // load to registers

    int inside = 0;
    for (int s = 0; s < samples_per_thread; s++) {
        float x = curand_uniform(&local_state);   // [0, 1)
        float y = curand_uniform(&local_state);
        if (x*x + y*y <= 1.0f) inside++;
    }

    states[tid] = local_state;  // save state back (important!)
    counts[tid] = inside;
}

// Host: estimate pi
void estimate_pi(int total_samples) {
    const int THREADS = 256, BLOCKS = 1024;
    const int N = THREADS * BLOCKS;
    const int spt = total_samples / N;  // samples per thread

    curandState *d_states;
    cudaMalloc(&d_states, N * sizeof(curandState));
    init_rng<<<BLOCKS, THREADS>>>(d_states, N, 42ULL);

    int *d_counts;
    cudaMalloc(&d_counts, N * sizeof(int));
    monte_carlo_pi<<<BLOCKS, THREADS>>>(d_states, d_counts, spt);

    // Reduce counts
    int total_inside = thrust_reduce_sum(d_counts, N);
    double pi = 4.0 * total_inside / (double)(N * spt);
    printf("pi ≈ %.6f (error: %.2e)\n", pi, fabs(pi - M_PI));
}
```

---

## 4. Metropolis-Hastings MCMC

Metropolis-Hastings samples from a target distribution π(x) without knowing its normalizing constant:

```
Algorithm per step:
  1. Propose x' = x + ε,   ε ~ N(0, σ²)
  2. Acceptance ratio α = min(1, π(x') / π(x))
  3. Accept x' with probability α; else stay at x

Parallel MCMC: run M independent chains simultaneously (embarrassingly parallel)
```

```c
// Target: 2D correlated Gaussian
// π(x,y) ∝ exp(-0.5 * [x,y] Σ^{-1} [x,y]^T)
__device__ float log_target(float x, float y) {
    // Σ = [[1, 0.9],[0.9, 1]]  → Σ^{-1} = [[1,-0.9],[-0.9,1]] / (1-0.81)
    float det_inv = 1.f / (1.f - 0.9f*0.9f);
    return -0.5f * det_inv * (x*x - 2.f*0.9f*x*y + y*y);
}

__global__ void metropolis_2d(
    curandState *states,
    float *chain_x, float *chain_y,   // output chains [N * n_steps]
    int N, int n_steps, float step_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    curandState local = states[tid];
    float x = curand_normal(&local);  // start from random point
    float y = curand_normal(&local);
    float log_p = log_target(x, y);

    for (int s = 0; s < n_steps; s++) {
        float xp = x + step_size * curand_normal(&local);
        float yp = y + step_size * curand_normal(&local);
        float log_pp = log_target(xp, yp);

        float log_alpha = log_pp - log_p;
        float u = curand_uniform(&local);
        if (logf(u) < log_alpha) {
            x = xp; y = yp; log_p = log_pp;
        }
        chain_x[tid * n_steps + s] = x;
        chain_y[tid * n_steps + s] = y;
    }
    states[tid] = local;
}
```

---

## 5. 2D Ising Model (Parallel Metropolis)

The Ising model on a square lattice uses a checkerboard decomposition to parallelize Metropolis updates without read-write conflicts:

```
Energy: E = -J Σ_{<i,j>} s_i s_j   (s_i = ±1)
Acceptance: P(flip) = min(1, exp(-ΔE / kT))

Checkerboard (red-black ordering):
  Even sites (i+j even):  update in parallel (no neighbor conflicts)
  Odd  sites (i+j odd):   update in parallel
```

```c
// Ising spin stored as int8: +1 or -1
__global__ void ising_sweep(
    int8_t *spins, int Nx, int Ny,
    float beta,      // β = 1/(k_B T)
    int parity,      // 0 = update even sites, 1 = update odd sites
    curandState *states)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= Nx || y >= Ny) return;
    if ((x + y) % 2 != parity) return;  // skip wrong color

    int tid = y * Nx + x;
    curandState local = states[tid];

    int8_t s = spins[tid];

    // Sum of 4 neighbors (periodic BC)
    int sum_nbr =
        spins[((y-1+Ny)%Ny) * Nx + x] +
        spins[((y+1)%Ny)    * Nx + x] +
        spins[y * Nx + (x-1+Nx)%Nx]  +
        spins[y * Nx + (x+1)%Nx];

    // ΔE = 2 * J * s * sum_nbr  (J=1)
    float dE = 2.f * s * sum_nbr;

    // Accept/reject flip
    if (dE <= 0.f || curand_uniform(&local) < expf(-beta * dE))
        spins[tid] = -s;

    states[tid] = local;
}

// Measure magnetization per spin
float measure_magnetization(const int8_t *d_spins, int N) {
    // Use Thrust reduce (see Lesson 28)
    // thrust::reduce(thrust::device_pointer_cast(d_spins), ...) / N
    return 0.f; // placeholder
}
```

---

## 6. Variance Reduction Techniques

Standard Monte Carlo converges as O(1/√N). These methods reduce variance without more samples:

### Antithetic Variates

For a function f(U) where U ~ Uniform(0,1), use pairs (U, 1-U):

```c
__global__ void mc_antithetic(curandState *states, float *results, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N/2) return;

    curandState local = states[tid];
    float sum = 0.f;
    const int S = 100;

    for (int s = 0; s < S; s++) {
        float u = curand_uniform(&local);
        float u_anti = 1.f - u;
        // f(u) = exp(-u): integrate exp(-u) from 0 to 1 → exact = 1 - 1/e ≈ 0.6321
        float f1 = expf(-u);
        float f2 = expf(-u_anti);
        sum += 0.5f * (f1 + f2);   // average of pair
    }
    results[tid] = sum / S;
    states[tid] = local;
}
// Antithetic variates reduce variance by ~50% for monotone functions (no extra random calls)
```

### Importance Sampling

Sample from a proposal q(x) that approximates |f(x)|, weight by f(x)/q(x):

```
Estimator: (1/N) Σ f(x_i) / q(x_i),  x_i ~ q
Optimal q(x) ∝ |f(x)|  → zero variance
```

```c
// Estimate tail probability P(X > t) for X~N(0,1) using exponential tilting
// Proposal q(x) = λ·exp(-λ(x-t)),  x > t
__global__ void importance_sampling_tail(
    curandState *states, float *results,
    int N, float t, float lambda)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    curandState local = states[tid];

    // Sample from Exponential(lambda) shifted to start at t
    float x = t - logf(curand_uniform(&local)) / lambda;  // inverse CDF

    // Importance weight: N(0,1) / Exponential(lambda)
    float log_p = -0.5f * x * x - 0.5f * logf(2.f * M_PI);   // log N(0,1)
    float log_q = logf(lambda) - lambda * (x - t);             // log Exp proposal
    float w = expf(log_p - log_q);                             // likelihood ratio

    results[tid] = w;   // E[w] ≈ P(X > t)
    states[tid] = local;
}
```

---

## 7. Statistical Convergence Testing

```c
// Chi-squared test for uniformity: compare histogram bins to expected counts
void chi_squared_test(const float *d_rand, int N, int bins) {
    // 1. Compute histogram (atomic, see Lesson 18)
    int *h_hist = compute_histogram_cpu(d_rand, N, bins);

    float expected = (float)N / bins;
    float chi2 = 0.f;
    for (int b = 0; b < bins; b++) {
        float diff = h_hist[b] - expected;
        chi2 += diff * diff / expected;
    }

    // Degrees of freedom = bins - 1
    // Critical value at 0.05 significance, df=bins-1
    float critical = chi2_critical(bins - 1, 0.05);
    printf("χ² = %.2f, critical = %.2f → %s\n",
           chi2, critical,
           chi2 < critical ? "PASS (uniform)" : "FAIL (non-uniform)");
    free(h_hist);
}
```

---

## Key Takeaways

- **cuRAND host API** generates batches directly on the GPU with a single call; use for pre-generated samples where memory is available
- **cuRAND device API** generates on the fly per thread; save/restore `curandState` to global memory between kernel launches to maintain stream independence
- **Parallel chains**: Metropolis-Hastings with M independent chains is embarrassingly parallel; checkerboard ordering enables single-chain spatial parallelism for grid-based models like Ising
- **2D Ising model**: checkerboard (red-black) sweeps allow half the spins to update simultaneously since only same-color neighbors interact
- **Antithetic variates** use paired samples (U, 1-U) to reduce variance by ~50% at no extra RNG cost for monotone integrands
- **Importance sampling** focuses samples on high-contribution regions; most powerful for rare-event estimation where standard MC is impractical

---

**Next**: [28. Thrust and CUB](./28_Thrust_and_CUB.md) — Use the Thrust STL-like library and CUB primitives for high-level GPU sorting, reduction, and scan without writing custom kernels.
