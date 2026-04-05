/*
 * heat2d.cu — Lesson 23: PDE Solvers — Heat Equation
 *
 * Solves the 2-D heat equation u_t = α∇²u using:
 *   - Explicit FTCS finite difference (same as Lesson 17 stencil)
 *   - Neumann (zero-flux) boundary conditions (ghost cells)
 *   - Periodic boundary conditions variant
 *   - Outputs final center value for verification
 *
 * This lesson focuses on boundary condition handling and convergence,
 * while Lesson 17 focused on shared-memory tiling.
 *
 * Build:  nvcc -O2 -arch=sm_80 heat2d.cu -o heat2d
 * Run:    ./heat2d
 */

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e)); exit(1); } } while(0)

static const int   NX     = 256;
static const int   NY     = 256;
static const float DX     = 1.f / NX;
static const float DT     = 0.25f * DX * DX;    // stability: α*dt/dx² ≤ 0.25
static const int   STEPS  = 2000;
static const float ALPHA  = 1.f;

// ── FTCS with Neumann (zero-flux) BCs ────────────────────────────────────────
__global__ void heat_neumann(const float *u, float *u_new, int nx, int ny,
                              float alpha, float dt, float dx) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;

    // Clamp neighbours for Neumann BC (du/dn = 0 → ghost = interior)
    int xm = (x > 0)    ? x - 1 : x;
    int xp = (x < nx-1) ? x + 1 : x;
    int ym = (y > 0)    ? y - 1 : y;
    int yp = (y < ny-1) ? y + 1 : y;

    float lap = (u[y*nx+xm] + u[y*nx+xp] +
                 u[ym*nx+x] + u[yp*nx+x] - 4.f * u[y*nx+x]) / (dx*dx);
    u_new[y*nx+x] = u[y*nx+x] + alpha * dt * lap;
}

// ── FTCS with periodic BCs ────────────────────────────────────────────────────
__global__ void heat_periodic(const float *u, float *u_new, int nx, int ny,
                               float alpha, float dt, float dx) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;

    int xm = (x - 1 + nx) % nx;
    int xp = (x + 1)      % nx;
    int ym = (y - 1 + ny) % ny;
    int yp = (y + 1)      % ny;

    float lap = (u[y*nx+xm] + u[y*nx+xp] +
                 u[ym*nx+x] + u[yp*nx+x] - 4.f * u[y*nx+x]) / (dx*dx);
    u_new[y*nx+x] = u[y*nx+x] + alpha * dt * lap;
}

static float run(bool periodic, float *h_u) {
    const size_t bytes = (size_t)NX * NY * sizeof(float);
    float *d_u, *d_v;
    CUDA_CHECK(cudaMalloc(&d_u, bytes));
    CUDA_CHECK(cudaMalloc(&d_v, bytes));
    CUDA_CHECK(cudaMemcpy(d_u, h_u, bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16), grid((NX+15)/16, (NY+15)/16);
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    for (int s = 0; s < STEPS; s++) {
        if (periodic)
            heat_periodic<<<grid, block>>>(d_u, d_v, NX, NY, ALPHA, DT, DX);
        else
            heat_neumann <<<grid, block>>>(d_u, d_v, NX, NY, ALPHA, DT, DX);
        float *tmp = d_u; d_u = d_v; d_v = tmp;
    }
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);

    float center;
    CUDA_CHECK(cudaMemcpy(&center, &d_u[(NY/2)*NX + NX/2], sizeof(float), cudaMemcpyDeviceToHost));
    printf("  %-10s %.3f ms  center=%.6f\n",
           periodic ? "periodic" : "Neumann", ms, center);

    cudaFree(d_u); cudaFree(d_v);
    return ms;
}

int main(void) {
    float *h_u = (float *)calloc(NX * NY, sizeof(float));
    // Point source in center
    h_u[(NY/2)*NX + NX/2] = 1.f;

    printf("2-D Heat equation (%dx%d, %d steps, α=%.1f)\n", NX, NY, STEPS, ALPHA);
    run(false, h_u);
    run(true,  h_u);

    free(h_u);
    return 0;
}
