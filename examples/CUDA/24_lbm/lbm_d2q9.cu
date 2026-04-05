/*
 * lbm_d2q9.cu — Lesson 24: Fluid Dynamics with Lattice Boltzmann Method
 *
 * Implements the D2Q9 Lattice Boltzmann Method (LBM) for 2-D incompressible flow.
 *
 * Scheme overview:
 *   1. Stream: shift distribution functions along lattice directions
 *   2. Collide: relax toward Maxwell-Boltzmann equilibrium (BGK)
 *   3. Boundary: no-slip (bounce-back) on solid walls
 *
 * Demonstrates:
 *   - Structure-of-arrays (SoA) layout for LBM distributions
 *   - Separate stream and collide kernels (ping-pong buffers)
 *
 * Build:  nvcc -O2 -arch=sm_80 lbm_d2q9.cu -o lbm_d2q9
 * Run:    ./lbm_d2q9
 */

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e)); exit(1); } } while(0)

static const int NX     = 256;
static const int NY     = 128;
static const int Q      = 9;
static const int STEPS  = 500;
static const float TAU  = 0.6f;       // relaxation time (ν = (τ-0.5)/3)
static const float U_LID = 0.1f;      // lid velocity

// D2Q9 lattice velocities and weights
__constant__ int   CX[9] = { 0, 1, 0,-1, 0, 1,-1,-1, 1};
__constant__ int   CY[9] = { 0, 0, 1, 0,-1, 1, 1,-1,-1};
__constant__ float W9[9] = { 4.f/9, 1.f/9, 1.f/9, 1.f/9, 1.f/9,
                              1.f/36, 1.f/36, 1.f/36, 1.f/36 };
// Opposite direction indices for bounce-back
__constant__ int   OPP[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};

// ── Equilibrium distribution ──────────────────────────────────────────────────
__device__ float feq(int q, float rho, float ux, float uy) {
    float eu = CX[q]*ux + CY[q]*uy;
    float u2 = ux*ux + uy*uy;
    return W9[q] * rho * (1.f + 3.f*eu + 4.5f*eu*eu - 1.5f*u2);
}

// ── Stream + Collide (single fused kernel) ───────────────────────────────────
// Layout: f[q * NX * NY + y * NX + x]
__global__ void stream_collide(const float *f_in, float *f_out,
                                const int *solid, int nx, int ny) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    int idx = y * nx + x;

    // Stream: pull from upstream neighbours
    float f[Q];
    for (int q = 0; q < Q; q++) {
        int xs = (x - CX[q] + nx) % nx;
        int ys = (y - CY[q] + ny) % ny;
        f[q] = f_in[q * nx * ny + ys * nx + xs];
    }

    // Bounce-back on solid nodes
    if (solid[idx]) {
        for (int q = 0; q < Q; q++)
            f_out[q * nx * ny + idx] = f[OPP[q]];
        return;
    }

    // Macroscopic variables
    float rho = 0.f, ux = 0.f, uy = 0.f;
    for (int q = 0; q < Q; q++) {
        rho += f[q];
        ux  += CX[q] * f[q];
        uy  += CY[q] * f[q];
    }
    ux /= rho; uy /= rho;

    // Lid-driven cavity: top wall with moving lid
    if (y == ny - 1) { ux = U_LID; uy = 0.f; }

    // BGK collision
    float omega = 1.f / TAU;
    for (int q = 0; q < Q; q++)
        f_out[q * nx * ny + idx] = f[q] - omega * (f[q] - feq(q, rho, ux, uy));
}

int main(void) {
    const size_t f_bytes = (size_t)Q * NX * NY * sizeof(float);
    const size_t s_bytes = (size_t)NX * NY * sizeof(int);

    // Initialise distributions to equilibrium (rho=1, u=0)
    float *h_f = (float *)malloc(f_bytes);
    int   *h_s = (int   *)calloc(NX * NY, sizeof(int));
    for (int i = 0; i < NX * NY; i++)
        for (int q = 0; q < Q; q++)
            h_f[q * NX * NY + i] = W9[q];   // ρ=1, u=0 equilibrium

    // Solid walls: bottom (y=0) and top (y=NY-1) are no-slip; lid moves
    for (int x = 0; x < NX; x++) h_s[0 * NX + x] = 1;   // bottom wall

    float *d_f0, *d_f1;
    int   *d_s;
    CUDA_CHECK(cudaMalloc(&d_f0, f_bytes));
    CUDA_CHECK(cudaMalloc(&d_f1, f_bytes));
    CUDA_CHECK(cudaMalloc(&d_s,  s_bytes));
    CUDA_CHECK(cudaMemcpy(d_f0, h_f, f_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_s,  h_s, s_bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16), grid((NX+15)/16, (NY+15)/16);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    for (int s = 0; s < STEPS; s++) {
        stream_collide<<<grid, block>>>(d_f0, d_f1, d_s, NX, NY);
        float *tmp = d_f0; d_f0 = d_f1; d_f1 = tmp;
    }

    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);

    double mlups = (double)NX * NY * STEPS / (ms * 1e-3) / 1e6;
    printf("LBM D2Q9 lid-driven cavity (%dx%d, %d steps)\n", NX, NY, STEPS);
    printf("  Time: %.2f ms  Throughput: %.1f MLUPS\n", ms, mlups);

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_f0); cudaFree(d_f1); cudaFree(d_s);
    free(h_f); free(h_s);
    return 0;
}
