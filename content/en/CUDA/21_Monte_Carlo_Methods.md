# 21. Monte Carlo Methods on GPU

**Previous**: [N-Body Simulation](./20_N_Body_Simulation.md) | **Next**: [FFT on GPU](./22_FFT_on_GPU.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Use the cuRAND device API to generate random numbers inside GPU kernels
2. Choose between XORWOW, MT19937, and Sobol generators for different use cases
3. Implement a GPU parallel Monte Carlo π estimator
4. Implement the Black-Scholes European call option pricer using Monte Carlo simulation
5. Apply antithetic variates to reduce Monte Carlo variance without extra computation

---

## 1. Why GPUs Excel at Monte Carlo

Monte Carlo methods generate many independent random samples and process each sample identically. This is perfectly suited to the SIMD GPU model:

```
Sequential Monte Carlo: 1 thread × N samples × T work/sample = N×T
Parallel Monte Carlo:   N threads × 1 sample/thread × T work/thread = T  (N× speedup)

N = 10M samples: GPU generates and processes 10M samples simultaneously
vs CPU: 10M sequential samples
```

The main challenge is **random number generation (RNG)**: each thread needs an independent, high-quality random stream.

---

## 2. cuRAND Overview

cuRAND provides two APIs:

**Host API**: generates numbers on CPU, transfers to GPU (useful for pre-computed tables):
```c
curandGenerator_t gen;
curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW);
curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

float *d_random;
cudaMalloc(&d_random, N * sizeof(float));
curandGenerateUniform(gen, d_random, N);  // uniform [0,1)
curandDestroyGenerator(gen);
```

**Device API**: each thread owns its own RNG state — required for inside-kernel random number generation:
```c
#include <curand_kernel.h>

// Initialize one RNG state per thread
__global__ void init_rng(curandState *states, unsigned long long seed, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        // Each thread gets unique sequence: same seed, different sequence offset
        curand_init(seed, idx, 0, &states[idx]);
}

// Use state inside a kernel
__global__ void sample_kernel(curandState *states, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    curandState local = states[idx];       // copy to registers (faster)
    float u = curand_uniform(&local);      // uniform [0,1)
    float n = curand_normal(&local);       // standard normal N(0,1)
    out[idx] = n;
    states[idx] = local;                   // write back updated state
}
```

---

## 3. RNG Generator Types

```
Generator         Period          Quality       Cost       Use Case
---------------------------------------------------------------------------
XORWOW (default)  2^190 - 2^62   Good           Low       General purpose
MT19937           2^19937 - 1    Very good       Medium    High-quality uniform
MRG32k3a          ~2^191         Good            Medium    Multi-stream guarantee
Sobol (quasi)     N/A (det.)     Low discrepancy Medium    Integration, finance
Philox (counter)  2^128          Good            Very low  Reproducible, inline

Practical choice:
  Default simulations: CURAND_RNG_PSEUDO_XORWOW
  Financial Monte Carlo: CURAND_RNG_QUASI_SOBOL32 (faster convergence)
  Reproducible results: Philox4_32_10 (counter-based, no state to store)
```

---

## 4. Estimating π with Monte Carlo

Classic demonstration: sample uniform random (x, y) in [0,1]², count points inside the quarter-circle, estimate π ≈ 4 × (count inside) / (total samples):

```c
#include <curand_kernel.h>

__global__ void estimate_pi(
    curandState *states, int *d_count, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    curandState local = states[idx];

    int inside = 0;
    // Each thread draws SAMPLES_PER_THREAD samples
    const int SPT = 100;
    for (int k = 0; k < SPT; k++) {
        float x = curand_uniform(&local);
        float y = curand_uniform(&local);
        if (x*x + y*y <= 1.0f) inside++;
    }

    states[idx] = local;
    atomicAdd(d_count, inside);
}

double gpu_pi_estimate(int N_threads) {
    const int BLOCK = 256;
    int grid = (N_threads + BLOCK - 1) / BLOCK;

    curandState *d_states;
    cudaMalloc(&d_states, N_threads * sizeof(curandState));
    init_rng<<<grid, BLOCK>>>(d_states, 42ULL, N_threads);

    int *d_count;
    cudaMalloc(&d_count, sizeof(int));
    cudaMemset(d_count, 0, sizeof(int));

    estimate_pi<<<grid, BLOCK>>>(d_states, d_count, N_threads);

    int h_count;
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    long long total = (long long)N_threads * 100;  // 100 samples per thread
    double pi = 4.0 * h_count / total;

    cudaFree(d_states); cudaFree(d_count);
    return pi;
}
// With N_threads=100,000 (10M total samples): π ≈ 3.1416 (error ~0.01%)
```

---

## 5. Black-Scholes Option Pricing via Monte Carlo

A **European call option** on a stock pays max(S_T - K, 0) at expiry T, where S_T is the terminal stock price. Under risk-neutral pricing with geometric Brownian motion:

```
S_T = S_0 · exp((r - 0.5·σ²)·T + σ·√T·Z)    where Z ~ N(0,1)
Call price C = e^(-r·T) · E[max(S_T - K, 0)]
```

```c
__global__ void black_scholes_mc(
    curandState *states,
    float  S0,    // initial stock price
    float  K,     // strike price
    float  r,     // risk-free rate (annual)
    float  sigma, // volatility (annual)
    float  T,     // time to expiry (years)
    float *payoffs, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    curandState local = states[idx];

    // Drift and diffusion pre-computation (save register ops per sample)
    float drift   = (r - 0.5f * sigma * sigma) * T;
    float diffuse = sigma * sqrtf(T);
    float disc    = expf(-r * T);

    // Draw standard normal using Box-Muller (curand_normal uses Marsaglia)
    float Z  = curand_normal(&local);
    float ST = S0 * expf(drift + diffuse * Z);
    payoffs[idx] = disc * fmaxf(ST - K, 0.0f);

    states[idx] = local;
}

float option_price_mc(float S0, float K, float r, float sigma, float T, int N) {
    const int BLOCK = 256;
    int grid = (N + BLOCK - 1) / BLOCK;

    curandState *d_states;
    cudaMalloc(&d_states, N * sizeof(curandState));
    init_rng<<<grid, BLOCK>>>(d_states, 12345ULL, N);

    float *d_payoffs;
    cudaMalloc(&d_payoffs, N * sizeof(float));
    black_scholes_mc<<<grid, BLOCK>>>(d_states, S0, K, r, sigma, T, d_payoffs, N);

    // Reduce payoffs to mean (use CUB DeviceReduce or thrust::reduce)
    float total = thrust_reduce_sum(d_payoffs, N);
    float price = total / N;

    cudaFree(d_states); cudaFree(d_payoffs);
    return price;
}
// Closed-form Black-Scholes: ~$10.45 for S0=100, K=100, r=0.05, σ=0.2, T=1
// Monte Carlo (N=10M): matches to < 0.01
```

---

## 6. Antithetic Variates (Variance Reduction)

For every sample Z ~ N(0,1), also evaluate -Z. Since large positive Z gives a high payoff and -Z gives a low payoff, their average has lower variance than either alone:

```c
__global__ void black_scholes_antithetic(
    curandState *states,
    float S0, float K, float r, float sigma, float T,
    float *payoffs, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N / 2) return;  // N/2 threads, each produces 2 samples

    curandState local = states[idx];
    float drift   = (r - 0.5f * sigma * sigma) * T;
    float diffuse = sigma * sqrtf(T);
    float disc    = expf(-r * T);

    float Z  = curand_normal(&local);

    // Primal path (+Z)
    float ST1 = S0 * expf(drift + diffuse * Z);
    float p1  = disc * fmaxf(ST1 - K, 0.0f);

    // Antithetic path (-Z)
    float ST2 = S0 * expf(drift - diffuse * Z);
    float p2  = disc * fmaxf(ST2 - K, 0.0f);

    // Store average of the pair
    payoffs[idx] = 0.5f * (p1 + p2);
    states[idx]  = local;
}
// With antithetic variates: same N samples → ~50% variance reduction
// Equivalently: achieve same accuracy with ~50% fewer samples
```

**Why it works**: Z and -Z are negatively correlated for convex payoffs, so averaging them cancels much of the random variation. For call options (convex payoff), antithetic variates typically reduce standard error by 40–70%.

---

## 7. Sobol Quasi-Random Sequences

Pseudo-random numbers can cluster; Sobol sequences (low-discrepancy) fill the space more uniformly, giving better convergence for integration:

```c
// Generate Sobol sequence with cuRAND
curandGenerator_t gen;
curandCreateGenerator(&gen, CURAND_RNG_QUASI_SOBOL32);
curandSetQuasiRandomGeneratorDimensions(gen, 2);  // 2D Sobol for (x, y) pairs

float *d_sobol;
cudaMalloc(&d_sobol, 2 * N * sizeof(float));
curandGenerateUniform(gen, d_sobol, 2 * N);
// d_sobol[0..N-1] = x coordinates, d_sobol[N..2N-1] = y coordinates

// Scrambled Sobol (better statistical properties)
curandCreateGenerator(&gen, CURAND_RNG_QUASI_SCRAMBLED_SOBOL32);
```

**Convergence comparison** for integration error vs N samples:
```
Method              Error rate        Error at N=10M
---------------------------------------------------
Pseudo-random MC    O(1/√N)          ~0.0003
Sobol quasi-MC      O((log N)^d / N) ~0.00001  (5-10× better)
```

Sobol sequences are the industry standard for financial Monte Carlo.

---

## Key Takeaways

- The cuRAND **device API** (`curandState`, `curand_init`, `curand_uniform`) gives each thread its own independent RNG stream
- Always copy `curandState` to a local register variable before use — avoids repeated global memory round trips per draw
- **XORWOW** is the default; **Sobol** gives lower-discrepancy sequences for integration; **Philox** is counter-based and cheapest to initialize
- GPU Monte Carlo scales linearly: 10M independent samples run in the same time as 1 sample, bounded only by compute throughput
- **Antithetic variates** halve variance (or halve the required sample count) at zero extra memory cost — always worth enabling for call/put options
- **Black-Scholes MC** on GPU with N=10M samples runs in ~10 ms; the exact closed-form takes < 1 μs but MC generalizes to exotic payoffs

---

**Next**: [22. FFT on GPU](./22_FFT_on_GPU.md) — Compute 1D/2D/3D Fast Fourier Transforms with cuFFT, implement convolution via FFT, and understand normalization pitfalls.
