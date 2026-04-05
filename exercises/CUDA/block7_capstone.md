# Block 7 — Capstone

**Lessons covered**: L37 (Multi-GPU & NCCL), L38 (End-to-End Projects)

---

## Exercise 7.1 — NCCL AllReduce

**Concept introduced in**: L37 (Multi-GPU & NCCL)

### Problem Statement

Use NCCL's `ncclAllReduce` to sum a float vector across two GPUs (simulated with two CPU
threads, each managing one GPU). After the allreduce, every GPU should hold the element-wise
sum of both GPU's input vectors.

### Requirements

- Two GPUs (GPU 0 and GPU 1). Skip or print a message if only one GPU is available.
- Vector size: N = 1,000,000 floats.
- GPU 0 initializes its vector to all 1.0f; GPU 1 initializes to all 2.0f.
- After `ncclAllReduce`, both vectors should contain all 3.0f (1+2).
- Verify on both GPUs.

### Starter Code

```cuda
// ex7_1_nccl_allreduce.cu
// Compile: nvcc -O2 -arch=sm_80 -o ex7_1 ex7_1_nccl_allreduce.cu -lnccl
// Run:     ./ex7_1  (requires 2 GPUs)

#include <cuda_runtime.h>
#include <nccl.h>
#include <pthread.h>
#include <cstdio>
#include <cmath>
#include <cassert>

#define N 1000000

// Macro to check NCCL errors
#define NCCL_CHECK(cmd) do {                                         \
    ncclResult_t e = (cmd);                                          \
    if (e != ncclSuccess) {                                          \
        fprintf(stderr, "NCCL error %s:%d '%s'\n",                  \
                __FILE__, __LINE__, ncclGetErrorString(e));          \
        exit(EXIT_FAILURE);                                          \
    }                                                                \
} while(0)

struct ThreadArgs {
    int        rank;        // 0 or 1
    ncclComm_t comm;
    float*     d_data;      // device buffer for this rank
    float      fill_value;  // what to pre-fill d_data with
    bool       pass;        // verification result (written by thread)
};

void* worker(void* arg) {
    ThreadArgs* a = (ThreadArgs*)arg;
    cudaSetDevice(a->rank);

    // Allocate and fill device buffer
    cudaMalloc(&a->d_data, N * sizeof(float));
    // TODO: launch a fill kernel or use cudaMemset equivalent
    //   (cudaMemset only works for byte patterns; use a small kernel or thrust::fill)
    // For simplicity: fill with a kernel
    // fill_kernel<<<(N+255)/256, 256>>>(a->d_data, a->fill_value, N);

    // AllReduce: sum across all ranks
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    // TODO: NCCL_CHECK(ncclAllReduce(a->d_data, a->d_data, N, ncclFloat, ncclSum, a->comm, stream))
    cudaStreamSynchronize(stream);

    // Verify: all elements should equal 1.0 + 2.0 = 3.0
    float* h_data = new float[N];
    cudaMemcpy(h_data, a->d_data, N * sizeof(float), cudaMemcpyDeviceToHost);
    a->pass = true;
    for (int i = 0; i < N; ++i) {
        if (fabsf(h_data[i] - 3.0f) > 1e-5f) { a->pass = false; break; }
    }
    delete[] h_data;
    cudaFree(a->d_data);
    cudaStreamDestroy(stream);
    return nullptr;
}

int main() {
    int n_devs;
    cudaGetDeviceCount(&n_devs);
    if (n_devs < 2) {
        printf("This exercise requires 2 GPUs. Found %d. Skipping.\n", n_devs);
        return 0;
    }

    int devs[2] = {0, 1};
    ncclComm_t comms[2];

    // Initialize NCCL communicators for both GPUs
    // TODO: NCCL_CHECK(ncclCommInitAll(comms, 2, devs))

    ThreadArgs args[2];
    args[0] = {0, comms[0], nullptr, 1.0f, false};
    args[1] = {1, comms[1], nullptr, 2.0f, false};

    // TODO: Launch two pthreads, each running worker()
    pthread_t threads[2];
    // TODO: pthread_create(&threads[0], nullptr, worker, &args[0])
    // TODO: pthread_create(&threads[1], nullptr, worker, &args[1])
    // TODO: pthread_join both

    printf("GPU 0 verification: %s\n", args[0].pass ? "PASS" : "FAIL");
    printf("GPU 1 verification: %s\n", args[1].pass ? "PASS" : "FAIL");

    // TODO: NCCL_CHECK(ncclCommDestroy(comms[0]))
    // TODO: NCCL_CHECK(ncclCommDestroy(comms[1]))
    return (args[0].pass && args[1].pass) ? 0 : 1;
}
```

### Fill Kernel (add before main)

```cuda
__global__ void fill_kernel(float* d, float val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = val;
}
```

### Expected Output

```
GPU 0 verification: PASS
GPU 1 verification: PASS
```

### Hints

| Step | API |
|------|-----|
| Init comms | `ncclCommInitAll(comms, n_gpus, dev_ids)` |
| AllReduce | `ncclAllReduce(sendbuf, recvbuf, count, type, op, comm, stream)` |
| Destroy | `ncclCommDestroy(comm)` |

- Each GPU thread must call `cudaSetDevice(rank)` before any CUDA operations.
- NCCL operations are enqueued into a CUDA stream; synchronize with `cudaStreamSynchronize` after the collective.
- For single-process multi-GPU (one process owning all GPUs), `ncclCommInitAll` is the simplest initialization path.
- For multi-process (MPI-style), use `ncclCommInitRank` with an `ncclUniqueId`.

### Performance Target

AllReduce for N = 1M floats (4 MB) should complete in < 10 ms on NVLink-connected GPUs (< 1 ms on A100 NVLink 600 GB/s).

---

## Exercise 7.2 (Capstone A) — LBM D2Q9 + Cylindrical Obstacle

**Concept introduced in**: L38 (End-to-End Projects)

### Problem Statement

Extend the Lattice Boltzmann Method (LBM) D2Q9 simulation from the course to support a
cylindrical obstacle at the center of the domain. Implement the no-slip bounce-back
boundary condition at solid cells. After the simulation converges, extract and print
the velocity magnitude field so it can be visualized as streamlines.

### Background

D2Q9 LBM uses 9 discrete velocity directions per cell:

```
6  2  5
3  0  1
7  4  8
```

Each direction has a weight `w_i` and lattice velocity `(cx_i, cy_i)`. The two steps are:

1. **Collision**: relax toward the local Maxwell-Boltzmann equilibrium.
2. **Streaming**: propagate distributions along their velocity directions.

At solid (obstacle) cells, streaming applies **bounce-back**: a distribution traveling into
the wall reverses direction (e.g., direction 1 bounces back as direction 3).

### Requirements

- Domain: 400 × 200 cells.
- Cylinder: center at (100, 100), radius 20 cells.
- Inflow (left boundary): `u_x = 0.1` (lattice units), `u_y = 0`.
- Outflow (right boundary): copy from adjacent column.
- Run 5000 time steps.
- Output: print a 40×20 ASCII map of velocity magnitude (scaled to chars).

### Starter Code

```cuda
// ex7_2_lbm_obstacle.cu
// Compile: nvcc -O2 -arch=sm_80 -o ex7_2 ex7_2_lbm_obstacle.cu

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#define NX   400
#define NY   200
#define Q    9
#define TAU  0.6f     // relaxation time (viscosity nu = (tau - 0.5)/3)
#define UIN  0.1f     // inlet velocity
#define NSTEPS 5000

// D2Q9 lattice constants
__constant__ float w[Q]  = {4.f/9, 1.f/9, 1.f/9, 1.f/9, 1.f/9,
                              1.f/36, 1.f/36, 1.f/36, 1.f/36};
__constant__ int   cx[Q] = {0,  1, 0, -1, 0,  1, -1, -1,  1};
__constant__ int   cy[Q] = {0,  0, 1,  0,-1,  1,  1, -1, -1};
// Bounce-back opposite direction index
__constant__ int   opp[Q] = {0, 3, 4, 1, 2, 7, 8, 5, 6};

// Equilibrium distribution
__device__ float feq(int q, float rho, float ux, float uy) {
    float cu = cx[q] * ux + cy[q] * uy;
    float uu = ux * ux + uy * uy;
    return w[q] * rho * (1.0f + 3.0f*cu + 4.5f*cu*cu - 1.5f*uu);
}

// Collision + streaming kernel (single-step)
__global__ void lbm_step(const float* f_in, float* f_out,
                          const bool* solid, int nx, int ny) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;

    int idx = y * nx + x;

    if (solid[idx]) {
        // TODO: Bounce-back: reverse all distributions for solid cells.
        // For each direction q, f_out at this cell in direction opp[q]
        // should receive f_in[q] from this cell.
        // (Full bounce-back: f_out[opp[q]][idx] = f_in[q][idx])
        return;
    }

    // Gather: collect incoming distributions from upstream neighbors
    float f_local[Q];
    for (int q = 0; q < Q; ++q) {
        int xn = (x - cx[q] + nx) % nx;
        int yn = (y - cy[q] + ny) % ny;
        f_local[q] = f_in[q * nx * ny + yn * nx + xn];
    }

    // Compute macroscopic quantities
    float rho = 0.0f, ux = 0.0f, uy = 0.0f;
    for (int q = 0; q < Q; ++q) {
        rho += f_local[q];
        ux  += cx[q] * f_local[q];
        uy  += cy[q] * f_local[q];
    }
    ux /= rho; uy /= rho;

    // TODO: Collision: f_out[q][idx] = f_local[q] - (f_local[q] - feq(q, rho, ux, uy)) / tau
    for (int q = 0; q < Q; ++q) {
        // TODO
    }
}

// Inflow boundary (left edge): set f to equilibrium at (rho=1, ux=UIN, uy=0)
__global__ void apply_inflow(float* f, int nx, int ny) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= ny) return;
    int idx = y * nx + 0;  // x = 0
    for (int q = 0; q < Q; ++q)
        f[q * nx * ny + idx] = feq(q, 1.0f, UIN, 0.0f);
}

int main() {
    const int n = NX * NY;
    const int cx_obs = NX / 4, cy_obs = NY / 2, r_obs = 20;

    // Initialize solid mask
    bool* h_solid = new bool[n]();
    for (int y = 0; y < NY; ++y)
        for (int x = 0; x < NX; ++x) {
            int dx = x - cx_obs, dy = y - cy_obs;
            if (dx*dx + dy*dy <= r_obs*r_obs) h_solid[y * NX + x] = true;
        }

    bool* d_solid;
    float *d_f0, *d_f1;
    // TODO: cudaMalloc d_solid (n), d_f0 (Q*n), d_f1 (Q*n)
    // TODO: cudaMemcpy h_solid -> d_solid
    // TODO: Initialize d_f0 to equilibrium everywhere (launch a kernel or use cudaMemset tricks)
    //   For simplicity: fill all f[q] with w[q] (rho=1, u=0) using cudaMemset or a fill kernel

    dim3 block(16, 16);
    dim3 grid((NX + 15) / 16, (NY + 15) / 16);

    for (int step = 0; step < NSTEPS; ++step) {
        apply_inflow<<<(NY + 255) / 256, 256>>>(d_f0, NX, NY);
        lbm_step<<<grid, block>>>(d_f0, d_f1, d_solid, NX, NY);
        // Swap buffers
        float* tmp = d_f0; d_f0 = d_f1; d_f1 = tmp;
    }
    cudaDeviceSynchronize();

    // Extract velocity magnitude and print ASCII map (downsampled 10x)
    float* h_f = new float[Q * n];
    // TODO: cudaMemcpy d_f0 -> h_f

    int map_nx = 40, map_ny = 20;
    printf("\nVelocity magnitude (ASCII, 10x downsampled):\n");
    for (int my = map_ny - 1; my >= 0; --my) {
        for (int mx = 0; mx < map_nx; ++mx) {
            int x = mx * NX / map_nx + NX / map_nx / 2;
            int y = my * NY / map_ny + NY / map_ny / 2;
            int idx = y * NX + x;
            if (h_solid[idx]) { putchar('#'); continue; }
            float ux = 0.0f, uy = 0.0f, rho = 0.0f;
            for (int q = 0; q < Q; ++q) {
                float fq = h_f[q * n + idx];
                rho += fq; ux += cx[q] * fq; uy += cy[q] * fq;
            }
            ux /= rho; uy /= rho;
            float umag = sqrtf(ux*ux + uy*uy) / UIN;
            char c = (umag > 1.2f) ? '2' : (umag > 0.8f) ? '1' : (umag > 0.4f) ? '.' : ' ';
            putchar(c);
        }
        putchar('\n');
    }

    // TODO: cudaFree x3
    delete[] h_solid; delete[] h_f;
    return 0;
}
```

### Expected Output (approximate)

```
Velocity magnitude (ASCII, 10x downsampled):
.......11111111111111111.....111.......
......11122222222222211.....1111......
.....1112222222222222211....11111.....
....111222222222222222211...111111....
...11122222222222222222211..1111111...
..1112222222222222222222211.11111111..
.111222222222222222222222211111111111.
11122222222###########22221111111111..
1112222222##############2221111111111.
112222222################22111111111..
112222222################22111111111..
1112222222##############2221111111111.
11122222222###########22221111111111..
.111222222222222222222222211111111111.
..1112222222222222222222211.11111111..
...11122222222222222222211..1111111...
....111222222222222222211...111111....
.....1112222222222222211....11111.....
......11122222222222211.....1111......
.......11111111111111111.....111.......
```

### Hints

- Memory layout: `f[q * NX * NY + y * NX + x]` — store all 9 directions separately for coalesced access.
- Bounce-back at solid cells: `f_out[opp[q] * n + idx] = f_in[q * n + idx]` (the distribution "bounces" back in the opposite direction).
- The inflow boundary forces equilibrium at every step; outflow copies from the adjacent interior column.
- After convergence, the Kármán vortex street develops downstream of the cylinder (visible as alternating low/high velocity regions).

### Performance Target

5000 time steps on a 400×200 grid should complete in < 60 seconds. Each step involves 9 × 400 × 200 = 720K read+write operations.

---

## Exercise 7.3 (Capstone B) — GPT-2-style Token Generation

**Concept introduced in**: L38 (End-to-End Projects)

### Problem Statement

Implement a minimal forward pass for a single GPT-2-style transformer block in CUDA,
then run greedy token generation for 10 tokens. The pipeline is:

```
token_ids → embed → transformer_block → unembed → argmax → next_token
```

All operations must be CUDA kernels (no cuBLAS for this exercise — write the kernels
yourself to practice the full stack).

### Transformer Block (simplified)

```
LayerNorm → Self-Attention (QKV projection, scaled dot-product, output projection) → residual
    → LayerNorm → FFN (linear → GELU → linear) → residual
```

### Requirements

- Vocabulary size: V = 1024 (tiny, for correctness focus).
- Embedding dimension: d_model = 128.
- Number of heads: h = 4, head dim = 32.
- FFN hidden dim: 4 × d_model = 512.
- Sequence length for one generation step: T = 1 (single token at a time, KV cache omitted).
- Use random initialized weights (the goal is correctness of the kernel pipeline, not meaningful text).
- Verify: each generated token is in [0, V) and the pipeline runs without CUDA errors.

### Starter Code

```cuda
// ex7_3_gpt2_generate.cu
// Compile: nvcc -O2 -arch=sm_80 -o ex7_3 ex7_3_gpt2_generate.cu

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>

#define V         1024    // vocabulary size
#define D         128     // model dimension
#define H         4       // attention heads
#define D_HEAD    32      // = D / H
#define D_FFN     512     // = 4 * D
#define BLOCK     256

// ---- Kernel 1: Token Embedding ----
// Look up token embedding: x = embed_table[token_id]
__global__ void embed(const float* embed_table, const int* token_ids,
                      float* x, int d) {
    int dim = threadIdx.x;   // one thread per embedding dimension
    int tok = blockIdx.x;    // one block per token in batch
    if (dim < d)
        x[tok * d + dim] = embed_table[token_ids[tok] * d + dim];
}

// ---- Kernel 2: Layer Normalization ----
// Normalize each row of x independently: x_norm = (x - mean) / std * gamma + beta
__global__ void layernorm(const float* x, float* out,
                          const float* gamma, const float* beta,
                          int d, float eps) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    // Each block handles one row (token)
    float val = (tid < d) ? x[blockIdx.x * d + tid] : 0.0f;

    // TODO: Compute mean using shared memory reduction
    // TODO: Compute variance using shared memory reduction
    // TODO: Normalize: out[blockIdx.x * d + tid] = (val - mean) / sqrt(var + eps) * gamma[tid] + beta[tid]
}

// ---- Kernel 3: Linear (matmul) ----
// out[i] = W[i, :] @ in + bias[i]   (W is [out_dim x in_dim], row-major)
__global__ void linear(const float* W, const float* bias,
                        const float* in, float* out,
                        int out_dim, int in_dim) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_dim) return;
    float sum = 0.0f;
    for (int j = 0; j < in_dim; ++j) sum += W[row * in_dim + j] * in[j];
    out[row] = sum + (bias ? bias[row] : 0.0f);
}

// ---- Kernel 4: GELU Activation ----
// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
__global__ void gelu(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    float k = 0.7978845608f;  // sqrt(2/pi)
    // TODO: x[i] = 0.5f * v * (1.0f + tanhf(k * (v + 0.044715f * v * v * v)));
}

// ---- Kernel 5: Scaled Dot-Product Attention ----
// For T=1 (single token), this reduces to: attn = softmax(q @ k^T / sqrt(d_head)) @ v
// With T=1 and one head: q, k, v are D_HEAD-dimensional vectors.
// Output = v (since softmax of a single element is 1.0).
// For T=1, attn_out = v  (trivial case — no cross-token attention).
__global__ void self_attention_t1(const float* q, const float* k, const float* v,
                                   float* out, int d_head, int h) {
    // With T=1, each head's attention output = v[head]
    // TODO: out[head * d_head + dim] = v[head * d_head + dim]
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < h * d_head) out[i] = v[i];
}

// ---- Kernel 6: Residual Add ----
__global__ void residual_add(float* x, const float* residual, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += residual[i];
}

// ---- Kernel 7: ArgMax (Greedy Decoding) ----
__global__ void argmax(const float* logits, int* out_token, int v) {
    // Single block, scans all V logits, finds argmax
    __shared__ float s_max_val;
    __shared__ int   s_max_idx;
    if (threadIdx.x == 0) { s_max_val = -1e38f; s_max_idx = 0; }
    __syncthreads();

    float local_max = -1e38f;
    int   local_idx = 0;
    for (int i = threadIdx.x; i < v; i += blockDim.x) {
        if (logits[i] > local_max) { local_max = logits[i]; local_idx = i; }
    }

    // TODO: Block-reduce to find global max across all threads
    // Use atomicMax is not directly available for floats; use a reinterpret trick or serial fallback.
    // Simple approach: have each thread do atomicCAS to update (max_val, max_idx) pair.
    // For correctness in this exercise, use a shared memory reduction.

    // TODO: After reduction, thread 0 writes s_max_idx to out_token[0]
}

// ---- Weight initialization (random, on device) ----
__global__ void rand_init(float* w, int n, unsigned long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // Simple LCG RNG
    unsigned long x = seed + i;
    x = x * 6364136223846793005ULL + 1442695040888963407ULL;
    w[i] = (float)(int)(x >> 32) / (float)0x80000000 * 0.02f;  // scale to small values
}

void rand_init_weight(float* d_w, int n, unsigned long seed = 42) {
    rand_init<<<(n + 255) / 256, 256>>>(d_w, n, seed);
}

int main() {
    // Allocate all model weights
    float *d_embed_table;     // [V, D]
    float *d_Wq, *d_Wk, *d_Wv, *d_Wo;   // attention projections [D, D] each
    float *d_W1, *d_b1, *d_W2, *d_b2;   // FFN: [D_FFN, D], [D_FFN]; [D, D_FFN], [D]
    float *d_gamma1, *d_beta1;           // LayerNorm 1 params [D]
    float *d_gamma2, *d_beta2;           // LayerNorm 2 params [D]
    float *d_unembed;                    // [V, D] (weight-tied with embed_table typically)

    // TODO: cudaMalloc all weights above
    // TODO: Initialize with rand_init_weight (each weight with a different seed)

    // Working buffers
    float *d_x, *d_residual, *d_q, *d_k, *d_v, *d_attn_out, *d_ffn_hidden;
    float *d_logits;
    int   *d_token, *d_next_token;

    // TODO: cudaMalloc all buffers above

    // Start with token 0
    int h_token = 0;
    cudaMemcpy(d_token, &h_token, sizeof(int), cudaMemcpyHostToDevice);

    printf("Generated tokens: %d", h_token);

    for (int step = 0; step < 10; ++step) {
        // 1. Embed
        embed<<<1, D>>>(d_embed_table, d_token, d_x, D);
        // Save residual
        cudaMemcpy(d_residual, d_x, D * sizeof(float), cudaMemcpyDeviceToDevice);

        // 2. LayerNorm 1
        // TODO: layernorm<<<1, D, 3*D*sizeof(float)>>>(d_x, d_x, d_gamma1, d_beta1, D, 1e-5f)

        // 3. Self-attention
        //   Q = Wq @ x,  K = Wk @ x,  V = Wv @ x
        // TODO: linear<<<(D+BLOCK-1)/BLOCK, BLOCK>>>(d_Wq, nullptr, d_x, d_q, D, D)
        // TODO: linear for d_k, d_v
        // TODO: self_attention_t1<<<1, D>>>(d_q, d_k, d_v, d_attn_out, D_HEAD, H)
        //   Output projection: x = Wo @ attn_out
        // TODO: linear<<<(D+BLOCK-1)/BLOCK, BLOCK>>>(d_Wo, nullptr, d_attn_out, d_x, D, D)

        // 4. Residual add
        // TODO: residual_add<<<(D+BLOCK-1)/BLOCK, BLOCK>>>(d_x, d_residual, D)
        // Save new residual
        cudaMemcpy(d_residual, d_x, D * sizeof(float), cudaMemcpyDeviceToDevice);

        // 5. LayerNorm 2
        // TODO: layernorm<<<1, D, 3*D*sizeof(float)>>>(d_x, d_x, d_gamma2, d_beta2, D, 1e-5f)

        // 6. FFN: W1 @ x → GELU → W2 @ x
        // TODO: linear<<<(D_FFN+BLOCK-1)/BLOCK, BLOCK>>>(d_W1, d_b1, d_x, d_ffn_hidden, D_FFN, D)
        // TODO: gelu<<<(D_FFN+BLOCK-1)/BLOCK, BLOCK>>>(d_ffn_hidden, D_FFN)
        // TODO: linear<<<(D+BLOCK-1)/BLOCK, BLOCK>>>(d_W2, d_b2, d_ffn_hidden, d_x, D, D_FFN)

        // 7. Residual add
        // TODO: residual_add<<<(D+BLOCK-1)/BLOCK, BLOCK>>>(d_x, d_residual, D)

        // 8. Unembed → logits [V]
        // TODO: linear<<<(V+BLOCK-1)/BLOCK, BLOCK>>>(d_unembed, nullptr, d_x, d_logits, V, D)

        // 9. Greedy decoding
        argmax<<<1, BLOCK>>>(d_logits, d_next_token, V);

        int h_next;
        cudaMemcpy(&h_next, d_next_token, sizeof(int), cudaMemcpyDeviceToHost);
        printf(" → %d", h_next);

        // Feed next token back
        cudaMemcpy(d_token, d_next_token, sizeof(int), cudaMemcpyDeviceToDevice);
    }

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    printf("\n");
    if (err != cudaSuccess)
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    else
        printf("All tokens in [0, %d): PASS\n", V);

    // TODO: cudaFree all
    return 0;
}
```

### Expected Output

```
Generated tokens: 0 → 412 → 87 → 653 → 201 → 44 → 890 → 312 → 7 → 534 → 128
All tokens in [0, 1024): PASS
```

(Exact token IDs depend on random weight initialization. Any values in [0, 1024) are correct.)

### Hints

- **LayerNorm**: reduction of mean and variance requires 2 shared memory reduction passes; use `sdata[0]` after sync to broadcast.
- **Attention T=1**: with sequence length 1, the attention score is a scalar (1×1 matrix), softmax = 1.0, so `attn_out = v`. This is a degenerate case — for T > 1, you need the full QKT softmax V loop.
- **ArgMax**: float atomicMax doesn't exist natively; use `__float_as_int` + `atomicMax` on the integer representation (works for positive floats since IEEE 754 ordering is preserved).
- **GELU**: the `tanhf` approximation is standard (used in GPT-2); the exact GELU uses `erff`.
- For a real model, use cuBLAS for all linear layers and load pretrained weights from the GPT-2 HuggingFace checkpoint.

### Performance Target

10 tokens of generation with d_model=128, V=1024: should run in < 100 ms total (the bottleneck is kernel launch overhead for tiny tensors, not arithmetic). For production-scale models (d_model=768, V=50257), use cuBLAS and achieve < 5 ms/token on A100.
