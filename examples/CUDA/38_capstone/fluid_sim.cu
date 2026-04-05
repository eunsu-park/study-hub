/*
 * fluid_sim.cu — Lesson 38: Capstone CUDA Application
 *
 * A miniature end-to-end GPU fluid simulation pipeline integrating
 * techniques from previous lessons:
 *
 *   Stage 1 — Advection     (stencil, Lesson 17)
 *   Stage 2 — Projection    (iterative Poisson solver using Jacobi iteration)
 *   Stage 3 — Visualisation (reduction to compute max velocity, Lesson 14)
 *
 * Fluid model: incompressible 2-D Navier-Stokes (simplified, no pressure)
 *   - Velocity field (u, v) on a staggered MAC grid
 *   - Semi-Lagrangian advection
 *   - Jacobi pressure projection
 *
 * Build:  nvcc -O2 -arch=sm_80 fluid_sim.cu -o fluid_sim
 * Run:    ./fluid_sim
 */

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e)); exit(1); } } while(0)
#define FULL_MASK 0xffffffff

static const int   NX     = 256;
static const int   NY     = 256;
static const float DX     = 1.f / NX;
static const float DT     = 0.01f;
static const float VISC   = 0.0001f;   // kinematic viscosity
static const int   STEPS  = 200;
static const int   JITER  = 40;        // Jacobi iterations per step

// ── Bilinear interpolation (for semi-Lagrangian advection) ───────────────────
__device__ float bilerp(const float *f, float x, float y, int nx, int ny) {
    x = fmaxf(0.5f, fminf(nx - 1.5f, x));
    y = fmaxf(0.5f, fminf(ny - 1.5f, y));
    int ix = (int)x, iy = (int)y;
    float fx = x - ix, fy = y - iy;
    float f00 = f[iy*nx + ix],     f10 = f[iy*nx + ix+1];
    float f01 = f[(iy+1)*nx + ix], f11 = f[(iy+1)*nx + ix+1];
    return (1-fx)*(1-fy)*f00 + fx*(1-fy)*f10 +
           (1-fx)*fy    *f01 + fx*fy    *f11;
}

// ── Advection: u^{n+1}(x) = u^n(x - dt * u^n(x)) ───────────────────────────
__global__ void advect(const float *u, const float *v,
                        const float *src, float *dst,
                        int nx, int ny, float dt, float dx) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    float px = x - dt * u[y*nx+x] / dx;
    float py = y - dt * v[y*nx+x] / dx;
    dst[y*nx+x] = bilerp(src, px, py, nx, ny);
}

// ── Diffusion: simple Jacobi (forward Euler viscosity) ───────────────────────
__global__ void diffuse(const float *f_in, float *f_out,
                         int nx, int ny, float visc, float dt, float dx) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x <= 0 || x >= nx-1 || y <= 0 || y >= ny-1) return;
    float lap = f_in[y*nx+x-1] + f_in[y*nx+x+1] +
                f_in[(y-1)*nx+x] + f_in[(y+1)*nx+x] - 4.f*f_in[y*nx+x];
    f_out[y*nx+x] = f_in[y*nx+x] + visc * dt / (dx*dx) * lap;
}

// ── Divergence of velocity field ────────────────────────────────────────────
__global__ void divergence(const float *u, const float *v, float *div,
                             int nx, int ny, float dx) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x <= 0 || x >= nx-1 || y <= 0 || y >= ny-1) return;
    div[y*nx+x] = ((u[y*nx+x+1] - u[y*nx+x-1]) +
                   (v[(y+1)*nx+x] - v[(y-1)*nx+x])) * 0.5f / dx;
}

// ── Jacobi pressure solve ────────────────────────────────────────────────────
__global__ void jacobi_pressure(const float *p_in, float *p_out,
                                  const float *div, int nx, int ny, float dx) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x <= 0 || x >= nx-1 || y <= 0 || y >= ny-1) return;
    p_out[y*nx+x] = (p_in[y*nx+x-1] + p_in[y*nx+x+1] +
                     p_in[(y-1)*nx+x] + p_in[(y+1)*nx+x] -
                     dx*dx*div[y*nx+x]) * 0.25f;
}

// ── Gradient subtraction (project onto divergence-free field) ────────────────
__global__ void project(float *u, float *v, const float *p,
                          int nx, int ny, float dx) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x <= 0 || x >= nx-1 || y <= 0 || y >= ny-1) return;
    u[y*nx+x] -= 0.5f * (p[y*nx+x+1] - p[y*nx+x-1]) / dx;
    v[y*nx+x] -= 0.5f * (p[(y+1)*nx+x] - p[(y-1)*nx+x]) / dx;
}

// ── Max velocity reduction (for CFL monitoring) ──────────────────────────────
__global__ void max_speed(const float *u, const float *v, float *out, int n) {
    extern __shared__ float s[];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;
    float speed = (i < n) ? sqrtf(u[i]*u[i] + v[i]*v[i]) : 0.f;
    s[tid] = speed;
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] = fmaxf(s[tid], s[tid+stride]);
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = s[0];
}

int main(void) {
    const size_t bytes = (size_t)NX * NY * sizeof(float);
    const int threads2d = 16;
    dim3 block(threads2d, threads2d);
    dim3 grid((NX + threads2d-1)/threads2d, (NY + threads2d-1)/threads2d);

    float *d_u, *d_v, *d_utmp, *d_vtmp;
    float *d_p, *d_ptmp, *d_div, *d_reduce;
    CUDA_CHECK(cudaMalloc(&d_u,       bytes));
    CUDA_CHECK(cudaMalloc(&d_v,       bytes));
    CUDA_CHECK(cudaMalloc(&d_utmp,    bytes));
    CUDA_CHECK(cudaMalloc(&d_vtmp,    bytes));
    CUDA_CHECK(cudaMalloc(&d_p,       bytes));
    CUDA_CHECK(cudaMalloc(&d_ptmp,    bytes));
    CUDA_CHECK(cudaMalloc(&d_div,     bytes));

    // Initial condition: lid-driven cavity (top boundary u = 1)
    CUDA_CHECK(cudaMemset(d_u, 0, bytes));
    CUDA_CHECK(cudaMemset(d_v, 0, bytes));
    CUDA_CHECK(cudaMemset(d_p, 0, bytes));

    float *h_lid = (float *)calloc(NX * NY, sizeof(float));
    for (int x = 0; x < NX; x++) h_lid[(NY-1)*NX + x] = 1.f;
    CUDA_CHECK(cudaMemcpy(d_u, h_lid, bytes, cudaMemcpyHostToDevice));
    free(h_lid);

    int n_blocks_reduce = (NX*NY + 255) / 256;
    CUDA_CHECK(cudaMalloc(&d_reduce, n_blocks_reduce * sizeof(float)));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    for (int s = 0; s < STEPS; s++) {
        // 1. Advect u and v
        advect<<<grid, block>>>(d_u, d_v, d_u, d_utmp, NX, NY, DT, DX);
        advect<<<grid, block>>>(d_u, d_v, d_v, d_vtmp, NX, NY, DT, DX);
        float *tmp;
        tmp = d_u; d_u = d_utmp; d_utmp = tmp;
        tmp = d_v; d_v = d_vtmp; d_vtmp = tmp;

        // 2. Diffuse
        diffuse<<<grid, block>>>(d_u, d_utmp, NX, NY, VISC, DT, DX);
        diffuse<<<grid, block>>>(d_v, d_vtmp, NX, NY, VISC, DT, DX);
        tmp = d_u; d_u = d_utmp; d_utmp = tmp;
        tmp = d_v; d_v = d_vtmp; d_vtmp = tmp;

        // 3. Project (pressure solve)
        divergence<<<grid, block>>>(d_u, d_v, d_div, NX, NY, DX);
        CUDA_CHECK(cudaMemset(d_p, 0, bytes));
        for (int j = 0; j < JITER; j++) {
            jacobi_pressure<<<grid, block>>>(d_p, d_ptmp, d_div, NX, NY, DX);
            tmp = d_p; d_p = d_ptmp; d_ptmp = tmp;
        }
        project<<<grid, block>>>(d_u, d_v, d_p, NX, NY, DX);
    }

    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);

    // Report max speed
    max_speed<<<n_blocks_reduce, 256, 256*sizeof(float)>>>(d_u, d_v, d_reduce, NX*NY);
    float *h_reduce = (float *)malloc(n_blocks_reduce * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_reduce, d_reduce, n_blocks_reduce*sizeof(float), cudaMemcpyDeviceToHost));
    float vmax = 0.f;
    for (int i = 0; i < n_blocks_reduce; i++) vmax = fmaxf(vmax, h_reduce[i]);
    free(h_reduce);

    printf("Capstone fluid sim (%dx%d, %d steps, %d Jacobi iters)\n",
           NX, NY, STEPS, JITER);
    printf("  Total time : %.2f ms  (%.2f ms/step)\n", ms, ms/STEPS);
    printf("  Max speed  : %.4f (CFL = %.4f)\n", vmax, vmax*DT/DX);

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_u); cudaFree(d_v); cudaFree(d_utmp); cudaFree(d_vtmp);
    cudaFree(d_p); cudaFree(d_ptmp); cudaFree(d_div); cudaFree(d_reduce);
    return 0;
}
