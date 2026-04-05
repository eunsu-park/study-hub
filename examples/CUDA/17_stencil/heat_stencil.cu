/*
 * heat_stencil.cu — Lesson 17: Stencil Computations
 *
 * Solves the 2-D heat equation with explicit FTCS:
 *   u(t+1)[i][j] = u(t)[i][j] + alpha * (nabla^2 u)
 *
 * Demonstrates:
 *   - 5-point stencil with halo-exchange via __syncthreads()
 *   - Tiled implementation with shared-memory halo padding
 *   - Double-buffering between two device arrays (ping-pong)
 *
 * Build:  nvcc -O2 -arch=sm_80 heat_stencil.cu -o heat_stencil
 * Run:    ./heat_stencil
 */

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e)); exit(1); } } while(0)

static const int   NX     = 512;      // grid width
static const int   NY     = 512;      // grid height
static const int   STEPS  = 1000;
static const float ALPHA  = 0.25f;    // stability condition: alpha <= 0.25
static const int   TILE_X = 32;
static const int   TILE_Y = 8;

// ── Naive stencil (all loads from global memory) ──────────────────────────────
__global__ void stencil_naive(const float *u, float *u_new, int nx, int ny) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x <= 0 || x >= nx - 1 || y <= 0 || y >= ny - 1) return;

    u_new[y * nx + x] = u[y * nx + x] +
        ALPHA * (u[y * nx + x - 1] + u[y * nx + x + 1] +
                 u[(y-1) * nx + x] + u[(y+1) * nx + x] -
                 4.f * u[y * nx + x]);
}

// ── Tiled stencil (shared memory with 1-cell halo) ───────────────────────────
__global__ void stencil_tiled(const float *u, float *u_new, int nx, int ny) {
    __shared__ float s[TILE_Y + 2][TILE_X + 2];

    int lx = threadIdx.x + 1;   // local index (inside halo)
    int ly = threadIdx.y + 1;
    int gx = blockIdx.x * TILE_X + threadIdx.x;
    int gy = blockIdx.y * TILE_Y + threadIdx.y;

    // Load interior + halo into shared memory
    if (gx < nx && gy < ny)
        s[ly][lx] = u[gy * nx + gx];

    // Load halo cells (boundary threads)
    if (threadIdx.x == 0 && gx > 0)
        s[ly][0] = u[gy * nx + gx - 1];
    if (threadIdx.x == TILE_X - 1 && gx < nx - 1)
        s[ly][TILE_X + 1] = u[gy * nx + gx + 1];
    if (threadIdx.y == 0 && gy > 0)
        s[0][lx] = u[(gy - 1) * nx + gx];
    if (threadIdx.y == TILE_Y - 1 && gy < ny - 1)
        s[TILE_Y + 1][lx] = u[(gy + 1) * nx + gx];
    __syncthreads();

    if (gx <= 0 || gx >= nx-1 || gy <= 0 || gy >= ny-1) return;

    u_new[gy * nx + gx] = s[ly][lx] +
        ALPHA * (s[ly][lx-1] + s[ly][lx+1] +
                 s[ly-1][lx] + s[ly+1][lx] - 4.f * s[ly][lx]);
}

int main(void) {
    const size_t bytes = (size_t)NX * NY * sizeof(float);
    float *h_u = (float *)calloc(NX * NY, sizeof(float));

    // Initial condition: hot spot in the center
    for (int y = NY/4; y < 3*NY/4; y++)
        for (int x = NX/4; x < 3*NX/4; x++)
            h_u[y * NX + x] = 1.f;

    float *d_u, *d_v;
    CUDA_CHECK(cudaMalloc(&d_u, bytes));
    CUDA_CHECK(cudaMalloc(&d_v, bytes));
    CUDA_CHECK(cudaMemcpy(d_u, h_u, bytes, cudaMemcpyHostToDevice));

    dim3 block_n(32, 8), block_t(TILE_X, TILE_Y);
    dim3 grid_n((NX + 31) / 32, (NY + 7) / 8);
    dim3 grid_t((NX + TILE_X - 1) / TILE_X, (NY + TILE_Y - 1) / TILE_Y);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    // Naive
    CUDA_CHECK(cudaMemcpy(d_u, h_u, bytes, cudaMemcpyHostToDevice));
    cudaEventRecord(t0);
    for (int s = 0; s < STEPS; s++) {
        stencil_naive<<<grid_n, block_n>>>(d_u, d_v, NX, NY);
        float *tmp = d_u; d_u = d_v; d_v = tmp;
    }
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms_naive; cudaEventElapsedTime(&ms_naive, t0, t1);

    // Tiled
    CUDA_CHECK(cudaMemcpy(d_u, h_u, bytes, cudaMemcpyHostToDevice));
    cudaEventRecord(t0);
    for (int s = 0; s < STEPS; s++) {
        stencil_tiled<<<grid_t, block_t>>>(d_u, d_v, NX, NY);
        float *tmp = d_u; d_u = d_v; d_v = tmp;
    }
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms_tiled; cudaEventElapsedTime(&ms_tiled, t0, t1);

    printf("Heat stencil (%dx%d, %d steps)\n", NX, NY, STEPS);
    printf("  Naive  : %.3f ms\n", ms_naive);
    printf("  Tiled  : %.3f ms  speedup=%.2fx\n",
           ms_tiled, ms_naive / ms_tiled);

    // Print center temperature
    float *h_out = (float *)malloc(bytes);
    CUDA_CHECK(cudaMemcpy(h_out, d_u, bytes, cudaMemcpyDeviceToHost));
    printf("  Center value after %d steps: %.6f\n", STEPS,
           h_out[(NY/2) * NX + NX/2]);

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_u); cudaFree(d_v);
    free(h_u); free(h_out);
    return 0;
}
