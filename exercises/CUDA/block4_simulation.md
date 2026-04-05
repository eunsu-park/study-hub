# Block 4 — Simulation

**Lessons covered**: L20 (N-body), L21 (Monte Carlo Methods), L22 (FFT on GPU),
L23 (PDE Solvers), L24 (Image Processing), L25 (Fluid Simulation Intro),
L26 (Lattice Boltzmann), L27 (Physics Simulation Patterns)

---

## Exercise 4.1 — N-body with Shared Memory Tile Loading

**Concept introduced in**: L20 (N-body)

### Problem Statement

Implement a tile-based N-body gravitational force computation. Rather than each thread
independently accessing all N body positions (O(N²) global memory reads), threads in a
block cooperate to load a tile of 32 bodies into shared memory and each thread computes
forces against all 32 bodies in the tile before moving to the next tile.

Verify energy conservation: total kinetic + potential energy should not drift by more than
1% over 100 integration steps (leapfrog / Euler).

### Requirements

- N = 1024 bodies, tile size = 32.
- Each body has position `(x, y, z)` and mass `m`.
- Use `float4` to pack `(x, y, z, m)` per body.
- Softening parameter `eps = 1e-3` to avoid singularities.
- Run 100 Euler steps; verify |E(100) - E(0)| / |E(0)| < 0.01.

### Starter Code

```cuda
// ex4_1_nbody.cu
// Compile: nvcc -O2 -arch=sm_80 -o ex4_1 ex4_1_nbody.cu

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>

#define NBODIES   1024
#define TILE_SIZE 32
#define G         1.0f
#define EPS       1e-3f
#define DT        1e-3f
#define NSTEPS    100

// Compute force on body i due to body j (packed in float4: x,y,z,mass)
__device__ float3 body_body_interaction(float4 bi, float4 bj) {
    float dx = bj.x - bi.x;
    float dy = bj.y - bi.y;
    float dz = bj.z - bi.z;
    float dist_sq = dx*dx + dy*dy + dz*dz + EPS*EPS;
    float inv_dist = rsqrtf(dist_sq);
    float inv_dist3 = inv_dist * inv_dist * inv_dist;
    float scale = G * bj.w * inv_dist3;   // bj.w = mass of j
    return {scale * dx, scale * dy, scale * dz};
}

// Each thread computes the net force on one body using tiled shared memory.
__global__ void compute_forces(const float4* __restrict__ pos,
                               float3* __restrict__       forces,
                               int n) {
    __shared__ float4 tile[TILE_SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float4 bi = (i < n) ? pos[i] : make_float4(0, 0, 0, 0);
    float3 f  = {0.0f, 0.0f, 0.0f};

    int ntiles = (n + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < ntiles; ++t) {
        // TODO: Cooperatively load tile t into shared memory.
        //   int j = t * TILE_SIZE + threadIdx.x;
        //   tile[threadIdx.x] = (j < n) ? pos[j] : make_float4(0, 0, 0, 0);
        // TODO: __syncthreads()

        // TODO: Loop over tile entries, accumulate force using body_body_interaction.
        //   Skip self-interaction (when global j == i).

        // TODO: __syncthreads() before loading next tile
    }

    if (i < n) forces[i] = f;
}

// Euler integration step
__global__ void integrate(float4* pos, float3* vel, const float3* forces, int n, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // TODO: vel[i] += forces[i] * dt (component-wise)
    // TODO: pos[i].xyz += vel[i] * dt
}

// Compute total energy on CPU (called before/after simulation)
double total_energy(const float4* pos, const float3* vel, int n) {
    double ke = 0.0, pe = 0.0;
    for (int i = 0; i < n; ++i) {
        ke += 0.5 * pos[i].w * (vel[i].x*vel[i].x + vel[i].y*vel[i].y + vel[i].z*vel[i].z);
        for (int j = i + 1; j < n; ++j) {
            float dx = pos[j].x - pos[i].x;
            float dy = pos[j].y - pos[i].y;
            float dz = pos[j].z - pos[i].z;
            float r = sqrtf(dx*dx + dy*dy + dz*dz + EPS*EPS);
            pe -= G * pos[i].w * pos[j].w / r;
        }
    }
    return ke + pe;
}

int main() {
    int n = NBODIES;
    float4* h_pos = new float4[n];
    float3* h_vel = new float3[n];
    srand(42);
    for (int i = 0; i < n; ++i) {
        h_pos[i] = {(float)rand()/RAND_MAX, (float)rand()/RAND_MAX,
                    (float)rand()/RAND_MAX, 1.0f};   // mass = 1
        h_vel[i] = {0.0f, 0.0f, 0.0f};
    }

    double E0 = total_energy(h_pos, h_vel, n);
    printf("Initial energy: %.6f\n", E0);

    float4 *d_pos;  float3 *d_vel, *d_forces;
    // TODO: cudaMalloc d_pos, d_vel, d_forces
    // TODO: cudaMemcpy h_pos -> d_pos, h_vel -> d_vel

    int nblocks = (n + TILE_SIZE - 1) / TILE_SIZE;
    for (int step = 0; step < NSTEPS; ++step) {
        compute_forces<<<nblocks, TILE_SIZE>>>(d_pos, d_forces, n);
        integrate<<<nblocks, TILE_SIZE>>>(d_pos, d_vel, d_forces, n, DT);
    }
    cudaDeviceSynchronize();

    // TODO: cudaMemcpy d_pos -> h_pos, d_vel -> h_vel
    double E1 = total_energy(h_pos, h_vel, n);
    double rel_err = fabs(E1 - E0) / fabs(E0);
    printf("Final   energy: %.6f\n", E1);
    printf("Energy drift:   %.4f%%\n", rel_err * 100.0);
    printf("Result: %s\n", (rel_err < 0.01) ? "PASS" : "FAIL");

    // TODO: cudaFree x3
    delete[] h_pos; delete[] h_vel;
    return 0;
}
```

### Expected Output

```
Initial energy: -487.234123
Final   energy: -487.214056
Energy drift:   0.0041%
Result: PASS
```

### Hints

- `float4` aligns to 16 bytes; loads are 128-bit wide — perfect for coalescing.
- With tile size 32, threads in a warp load one `float4` each = one 128-byte transaction.
- For the self-interaction guard: compare global body index `i` against `t * TILE_SIZE + tile_idx`.
- Euler integration accumulates error; use leapfrog for longer simulations.

### Performance Target

1024-body tiled force computation should run in < 1 ms per step. Naive (no tiling) should take ~2× longer due to redundant global memory loads.

---

## Exercise 4.2 — Monte Carlo π with cuRAND

**Concept introduced in**: L21 (Monte Carlo Methods)

### Problem Statement

Estimate π using the Monte Carlo method: generate M random (x, y) pairs uniformly in
[0, 1) × [0, 1); count points inside the unit quarter-circle (x² + y² ≤ 1); π ≈ 4 × count / M.

Use `cuRAND` with `curand_init` and `curand_uniform` inside the kernel for GPU-side RNG.

### Requirements

- M = 1,000,000 samples.
- Each thread generates its own RNG state with a unique seed based on thread index.
- Use a block-level reduction to count hits per block.
- Verify |π_estimated - π| < 0.01.

### Starter Code

```cuda
// ex4_2_monte_carlo_pi.cu
// Compile: nvcc -O2 -arch=sm_80 -o ex4_2 ex4_2_monte_carlo_pi.cu -lcurand

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cmath>

#define M          1000000
#define BLOCK_SIZE 256

// Each thread generates SAMPLES_PER_THREAD (x,y) pairs and counts hits.
__global__ void mc_pi(unsigned int* d_hits, int m) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    int samples_per_thread = (m + total_threads - 1) / total_threads;

    // Initialize cuRAND state — each thread needs a unique seed or unique sequence
    curandState state;
    // TODO: curand_init(seed=12345ULL, sequence=gid, offset=0ULL, &state)

    unsigned int hits = 0;
    for (int s = 0; s < samples_per_thread; ++s) {
        // TODO: float x = curand_uniform(&state)
        // TODO: float y = curand_uniform(&state)
        // TODO: if (x*x + y*y <= 1.0f) hits++
    }

    // Block-level reduction using shared memory
    __shared__ unsigned int s_hits[BLOCK_SIZE];
    s_hits[threadIdx.x] = hits;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) s_hits[threadIdx.x] += s_hits[threadIdx.x + s];
        __syncthreads();
    }
    // TODO: thread 0 does atomicAdd(&d_hits[0], s_hits[0])
}

int main() {
    unsigned int* d_hits;
    cudaMalloc(&d_hits, sizeof(unsigned int));
    cudaMemset(d_hits, 0, sizeof(unsigned int));

    int nblocks = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    mc_pi<<<nblocks, BLOCK_SIZE>>>(d_hits, M);
    cudaDeviceSynchronize();

    unsigned int h_hits;
    cudaMemcpy(&h_hits, d_hits, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    double pi_est = 4.0 * h_hits / M;
    double err    = fabs(pi_est - M_PI);
    printf("Samples: %d  Hits: %u\n", M, h_hits);
    printf("π estimated: %.6f  π actual: %.6f  Error: %.6f\n", pi_est, M_PI, err);
    printf("Result: %s\n", (err < 0.01) ? "PASS" : "FAIL");

    cudaFree(d_hits);
    return (err < 0.01) ? 0 : 1;
}
```

### Expected Output

```
Samples: 1000000  Hits: 785326
π estimated: 3.141304  π actual: 3.141593  Error: 0.000289
Result: PASS
```

### Hints

- `curand_init` with a unique `sequence` per thread gives independent streams from the same seed.
- `curand_uniform` returns `(0, 1]`; subtract a small epsilon if strict `[0, 1)` is needed.
- The `samples_per_thread` approach avoids launching exactly M threads (can be arbitrary M).
- Statistical error ∝ 1/√M, so at M = 1M: expected σ ≈ 0.0016.

### Performance Target

Should run in < 100 ms for M = 1M. cuRAND kernel setup (init) is the dominant cost for small M; for large M (1B+), generation dominates.

---

## Exercise 4.3 — cuFFT Low-pass Filter

**Concept introduced in**: L22 (FFT on GPU)

### Problem Statement

Apply a 1D low-pass filter using cuFFT:

1. Generate a signal: sum of a 10 Hz and 1000 Hz sine wave.
2. Forward FFT (real → complex).
3. Zero out all frequency bins above a cutoff (e.g., keep only the first 50 bins and their conjugates).
4. Inverse FFT (complex → real), normalize by N.
5. Verify that the output contains only the low-frequency component.

### Requirements

- Signal length N = 8192 (power of 2).
- Sample rate: 44100 Hz.
- Use `cufftExecR2C` (real-to-complex) and `cufftExecC2R` (complex-to-real).
- Verify: RMS of 1000 Hz component in output < 5% of the 10 Hz component.

### Starter Code

```cuda
// ex4_3_cufft.cu
// Compile: nvcc -O2 -arch=sm_80 -o ex4_3 ex4_3_cufft.cu -lcufft

#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdio>
#include <cmath>

#define N          8192
#define SAMPLE_RATE 44100.0f
#define F_LOW      10.0f      // Hz — to keep
#define F_HIGH     1000.0f    // Hz — to filter out
#define CUTOFF_BIN 50         // zero bins above this index

// Kernel: zero out high-frequency bins in the complex spectrum.
// Spectrum has N/2+1 complex elements (R2C output).
__global__ void apply_lowpass(cufftComplex* d_spectrum, int n_bins, int cutoff) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: if (i > cutoff && i < n_bins) d_spectrum[i] = {0.0f, 0.0f};
}

// Kernel: normalize FFT output by N (cuFFT does not normalize inverse).
__global__ void normalize(float* d_signal, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: if (i < n) d_signal[i] /= n;
}

int main() {
    const int n = N;
    const int n_bins = n / 2 + 1;

    // Generate signal on host
    float* h_signal_in  = new float[n];
    float* h_signal_out = new float[n];
    for (int i = 0; i < n; ++i) {
        float t = i / SAMPLE_RATE;
        h_signal_in[i] = sinf(2.0f * (float)M_PI * F_LOW  * t)
                       + sinf(2.0f * (float)M_PI * F_HIGH * t);
    }

    // Device buffers
    float*        d_signal;
    cufftComplex* d_spectrum;
    cudaMalloc(&d_signal,   n       * sizeof(float));
    cudaMalloc(&d_spectrum, n_bins  * sizeof(cufftComplex));
    cudaMemcpy(d_signal, h_signal_in, n * sizeof(float), cudaMemcpyHostToDevice);

    // cuFFT plan
    cufftHandle plan_r2c, plan_c2r;
    // TODO: cufftPlan1d(&plan_r2c, n, CUFFT_R2C, 1)
    // TODO: cufftPlan1d(&plan_c2r, n, CUFFT_C2R, 1)

    // Forward FFT
    // TODO: cufftExecR2C(plan_r2c, d_signal, d_spectrum)

    // Apply low-pass filter
    int nblocks = (n_bins + 255) / 256;
    apply_lowpass<<<nblocks, 256>>>(d_spectrum, n_bins, CUTOFF_BIN);

    // Inverse FFT
    // TODO: cufftExecC2R(plan_c2r, d_spectrum, d_signal)

    // Normalize
    nblocks = (n + 255) / 256;
    normalize<<<nblocks, 256>>>(d_signal, n);

    cudaMemcpy(h_signal_out, d_signal, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify: compute RMS of difference between output and pure F_LOW sine
    double rms_err = 0.0, rms_ref = 0.0;
    for (int i = 0; i < n; ++i) {
        float ref = sinf(2.0f * (float)M_PI * F_LOW * i / SAMPLE_RATE);
        rms_err += (h_signal_out[i] - ref) * (h_signal_out[i] - ref);
        rms_ref += ref * ref;
    }
    rms_err = sqrt(rms_err / n);
    rms_ref = sqrt(rms_ref / n);
    double snr = 20.0 * log10(rms_ref / rms_err);
    printf("RMS error vs pure F_LOW sine: %.6f  RMS ref: %.6f\n", (float)rms_err, (float)rms_ref);
    printf("SNR: %.1f dB\n", snr);
    printf("Result: %s\n", (rms_err < 0.05f * rms_ref) ? "PASS" : "FAIL");

    // TODO: cufftDestroy x2, cudaFree x2
    delete[] h_signal_in; delete[] h_signal_out;
    return 0;
}
```

### Expected Output

```
RMS error vs pure F_LOW sine: 0.000012  RMS ref: 0.707107
SNR: 95.4 dB
Result: PASS
```

### Hints

- `cufftExecR2C` treats the input as real and produces `N/2+1` complex (Hermitian) outputs.
- The normalization factor for the round-trip is `1/N`.
- Bin index `k` corresponds to frequency `k * SAMPLE_RATE / N` Hz.
- Cutoff bin 50 at 44100 Hz / 8192 = 5.37 Hz per bin → cutoff at 268 Hz, which removes the 1 kHz component.

### Performance Target

For N = 8192, the full pipeline (forward FFT + filter + inverse FFT) should complete in < 1 ms.

---

## Exercise 4.4 — 2D Heat Equation PDE

**Concept introduced in**: L23 (PDE Solvers)

### Problem Statement

Simulate the 2D heat equation using an explicit finite-difference scheme:

```
T[t+1][i][j] = T[t][i][j] + alpha * dt * Laplacian(T[t][i][j])
```

where the Laplacian is the same 5-point stencil from Exercise 3.4. Apply Dirichlet boundary
conditions (edges fixed at 0 except the bottom edge fixed at 1). Run 1000 time steps and
verify that the maximum temperature decreases monotonically after the initial transient.

### Requirements

- Grid: 512×512.
- `alpha = 0.1f`, `dt = 0.1f`, `dx = 1.0f / 512` (stability: `alpha * dt / dx² < 0.25`).
- Use double-buffering: ping-pong between two device arrays.
- Record `T_max` at every 100 steps; verify T_max at step 1000 < T_max at step 100.

### Starter Code

```cuda
// ex4_4_heat_equation.cu
// Compile: nvcc -O2 -arch=sm_80 -o ex4_4 ex4_4_heat_equation.cu

#include <cuda_runtime.h>
#include <cstdio>
#include <cfloat>
#include <cmath>

#define DIM        512
#define TILE       32
#define ALPHA      0.1f
#define DT         0.1f
#define NSTEPS     1000
#define RECORD_INT 100

// Heat equation time step kernel
__global__ void heat_step(const float* __restrict__ T_in,
                          float* __restrict__       T_out,
                          int rows, int cols,
                          float alpha, float dt, float inv_dx2) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row <= 0 || row >= rows - 1 || col <= 0 || col >= cols - 1) return;

    int idx = row * cols + col;
    float lap = T_in[idx - 1] + T_in[idx + 1]
              + T_in[idx - cols] + T_in[idx + cols]
              - 4.0f * T_in[idx];
    // TODO: T_out[idx] = T_in[idx] + alpha * dt * inv_dx2 * lap;
}

// Max reduction kernel (simplified — computes max over entire array on GPU)
__global__ void reduce_max(const float* d_in, float* d_out, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (gid < n) ? d_in[gid] : -FLT_MAX;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if (tid == 0) d_out[blockIdx.x] = sdata[0];
}

float gpu_max(const float* d_T, float* d_tmp, int n, int block_size) {
    int nb = (n + block_size - 1) / block_size;
    reduce_max<<<nb, block_size, block_size * sizeof(float)>>>(d_T, d_tmp, n);
    float* h = new float[nb];
    cudaMemcpy(h, d_tmp, nb * sizeof(float), cudaMemcpyDeviceToHost);
    float m = -FLT_MAX;
    for (int i = 0; i < nb; ++i) m = fmaxf(m, h[i]);
    delete[] h;
    return m;
}

int main() {
    const int rows = DIM, cols = DIM;
    const int n = rows * cols;
    const float dx = 1.0f / cols;
    const float inv_dx2 = 1.0f / (dx * dx);

    // Check stability: r = alpha * dt / dx^2 < 0.25
    float r = ALPHA * DT * inv_dx2;
    printf("Stability parameter r = %.4f  (must be < 0.25)\n", r);
    if (r >= 0.25f) { printf("UNSTABLE — reduce dt or alpha\n"); return 1; }

    float* h_T = new float[n]();
    // Bottom edge (row = rows-1) fixed at 1
    for (int col = 0; col < cols; ++col) h_T[(rows - 1) * cols + col] = 1.0f;

    float *d_T0, *d_T1, *d_tmp;
    // TODO: cudaMalloc d_T0, d_T1 (n floats each), d_tmp (n/256 floats for max reduction)
    // TODO: cudaMemcpy h_T -> d_T0
    // TODO: cudaMemcpy h_T -> d_T1

    dim3 block(TILE, TILE);
    dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);

    float tmax_history[NSTEPS / RECORD_INT + 1];
    int n_recorded = 0;

    for (int step = 0; step < NSTEPS; ++step) {
        // Re-apply boundary: bottom row of T0 must stay at 1
        // (Alternatively set it again each step — simpler)
        // TODO: cudaMemset or a small kernel to fix boundaries in d_T0

        heat_step<<<grid, block>>>(d_T0, d_T1, rows, cols, ALPHA, DT, inv_dx2);

        if ((step + 1) % RECORD_INT == 0) {
            float tmax = gpu_max(d_T1, d_tmp, n, 256);
            tmax_history[n_recorded++] = tmax;
            printf("Step %4d  T_max = %.6f\n", step + 1, tmax);
        }

        // Swap buffers
        float* tmp = d_T0; d_T0 = d_T1; d_T1 = tmp;
    }

    // Verify T_max decreases (after initial transient the max temp should be bounded by BC)
    // The bottom row is 1; interior heats up toward steady state — T_max should stabilize near 1
    bool monotone = true;
    for (int i = 1; i < n_recorded; ++i)
        if (tmax_history[i] > tmax_history[i - 1] + 1e-4f) { monotone = false; break; }
    printf("T_max trend monotone: %s\n", monotone ? "YES" : "NO");
    printf("Result: %s\n", monotone ? "PASS" : "FAIL");

    // TODO: cudaFree x3
    delete[] h_T;
    return monotone ? 0 : 1;
}
```

### Expected Output

```
Stability parameter r = 0.0001  (must be < 0.25)
Step  100  T_max = 1.000000
Step  200  T_max = 1.000000
...
Step 1000  T_max = 1.000000
T_max trend monotone: YES
Result: PASS
```

(T_max stays at 1.0 because the boundary enforces it; interior cells approach the steady state but never exceed 1.)

### Hints

- Double-buffering avoids read-write hazards: always read from `d_T0` and write to `d_T1`, then swap.
- Stability condition for explicit 2D heat equation: `r = alpha * dt / dx² ≤ 0.25`. Violation leads to divergent oscillations.
- Re-apply boundary conditions every step after the swap (or fix boundaries inside the kernel by not updating boundary cells).

### Performance Target

Single time step (512×512 grid) should complete in < 0.5 ms. The kernel is strongly memory-bandwidth bound.

---

## Exercise 4.5 — Gaussian Blur with Shared Memory

**Concept introduced in**: L24 (Image Processing)

### Problem Statement

Implement a 2D 3×3 Gaussian blur kernel using shared memory. The filter weights are:

```
1/16 * [ 1  2  1 ]
        [ 2  4  2 ]
        [ 1  2  1 ]
```

Load a `(TILE+2) × (TILE+2)` region into shared memory (1-pixel halo), then each interior
thread applies the weighted sum. Verify output matches a CPU reference implementation.

### Requirements

- Image: 4096×4096 grayscale float values in [0, 1].
- Tile size: 32×32.
- Boundary condition: clamp to edge (nearest neighbor).
- Max absolute error vs CPU reference < 1e-6.

### Starter Code

```cuda
// ex4_5_gaussian_blur.cu
// Compile: nvcc -O2 -arch=sm_80 -o ex4_5 ex4_5_gaussian_blur.cu

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>

#define DIM  4096
#define TILE 32
#define HALO (TILE + 2)

// 3x3 Gaussian kernel weights (1/16 normalization)
__constant__ float d_filter[3][3] = {
    {1.f/16, 2.f/16, 1.f/16},
    {2.f/16, 4.f/16, 2.f/16},
    {1.f/16, 2.f/16, 1.f/16}
};

// CPU reference blur
void gaussian_blur_cpu(const float* in, float* out, int rows, int cols) {
    float w[3][3] = {{1.f/16, 2.f/16, 1.f/16},
                     {2.f/16, 4.f/16, 2.f/16},
                     {1.f/16, 2.f/16, 1.f/16}};
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            float sum = 0.0f;
            for (int dr = -1; dr <= 1; ++dr)
                for (int dc = -1; dc <= 1; ++dc) {
                    int rr = min(max(r + dr, 0), rows - 1);
                    int cc = min(max(c + dc, 0), cols - 1);
                    sum += w[dr+1][dc+1] * in[rr * cols + cc];
                }
            out[r * cols + c] = sum;
        }
}

// GPU kernel: 3x3 Gaussian blur with shared memory halo
__global__ void gaussian_blur_gpu(const float* __restrict__ in,
                                  float* __restrict__       out,
                                  int rows, int cols) {
    // Shared memory: (TILE+2) x (TILE+2) including 1-pixel halo
    __shared__ float s[HALO][HALO];

    int tx = threadIdx.x, ty = threadIdx.y;
    int col = blockIdx.x * TILE + tx;
    int row = blockIdx.y * TILE + ty;

    // Helper lambda equivalent — clamp to valid range
    auto clamp_load = [&](int r, int c) -> float {
        r = min(max(r, 0), rows - 1);
        c = min(max(c, 0), cols - 1);
        return in[r * cols + c];
    };

    // TODO: Load shared memory including halos.
    //   Interior: s[ty+1][tx+1] = clamp_load(row, col)
    //   Halos (threads at tile boundary load the extra cells):
    //     if (tx == 0) s[ty+1][0] = clamp_load(row, col-1)
    //     if (tx == TILE-1) s[ty+1][TILE+1] = clamp_load(row, col+1)
    //     if (ty == 0) s[0][tx+1] = clamp_load(row-1, col)
    //     if (ty == TILE-1) s[TILE+1][tx+1] = clamp_load(row+1, col)
    //     Corners: thread (0,0), (0,TILE-1), (TILE-1,0), (TILE-1,TILE-1) each load one corner

    // TODO: __syncthreads()

    // Apply filter
    if (row < rows && col < cols) {
        float sum = 0.0f;
        // TODO: for dr in {-1,0,1}: for dc in {-1,0,1}: sum += d_filter[dr+1][dc+1] * s[ty+1+dr][tx+1+dc]
        out[row * cols + col] = sum;
    }
}

int main() {
    const int rows = DIM, cols = DIM;
    const int n = rows * cols;

    float* h_in      = new float[n];
    float* h_out_cpu = new float[n];
    float* h_out_gpu = new float[n];
    for (int i = 0; i < n; ++i) h_in[i] = (float)rand() / RAND_MAX;

    // CPU reference
    gaussian_blur_cpu(h_in, h_out_cpu, rows, cols);

    float *d_in, *d_out;
    // TODO: cudaMalloc d_in, d_out
    // TODO: cudaMemcpy h_in -> d_in

    dim3 block(TILE, TILE);
    dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    gaussian_blur_gpu<<<grid, block>>>(d_in, d_out, rows, cols);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms, s, e);

    // TODO: cudaMemcpy d_out -> h_out_gpu

    float max_err = 0.0f;
    for (int i = 0; i < n; ++i)
        max_err = fmaxf(max_err, fabsf(h_out_gpu[i] - h_out_cpu[i]));

    double bw = (2.0 * n * sizeof(float)) / (ms * 1e-3) / 1e9;
    printf("Time: %.3f ms  Effective BW: %.1f GB/s\n", ms, bw);
    printf("Max error: %.2e\n", max_err);
    printf("Result: %s\n", (max_err < 1e-6f) ? "PASS" : "FAIL");

    // TODO: cudaFree x2
    delete[] h_in; delete[] h_out_cpu; delete[] h_out_gpu;
    return 0;
}
```

### Expected Output

```
Time: 1.83 ms  Effective BW: 143.6 GB/s
Max error: 0.00e+00
Result: PASS
```

### Hints

- `__constant__` memory is cached and broadcast efficiently when all threads in a warp access the same index.
- Corner halo cells must also be loaded (4 corners, each loaded by a specific boundary thread).
- Alternatively, load the full `HALO × HALO` tile using a linearized loop where each thread loads one element based on `threadIdx.x + threadIdx.y * blockDim.x`.

### Performance Target

Should achieve > 100 GB/s effective bandwidth. For a 3×3 stencil, reuse ratio is 9 reads → 1 write, so shared memory saves significantly over naively reading from global memory 9 times per output.
