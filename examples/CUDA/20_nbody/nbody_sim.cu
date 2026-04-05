/*
 * nbody_sim.cu — Lesson 20: N-Body Simulation
 *
 * Direct-sum N-body gravitational simulation (O(N²) pairwise forces).
 *
 * Demonstrates:
 *   - Shared-memory tile loading for N-body (each block loads a tile of
 *     body positions into shared memory so all threads can reuse them)
 *   - Softened gravity: F = G*m*m / (r² + ε²)
 *   - Leapfrog time integration
 *
 * Build:  nvcc -O2 -arch=sm_80 nbody_sim.cu -o nbody_sim
 * Run:    ./nbody_sim
 */

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e)); exit(1); } } while(0)

static const int   N_BODIES = 8192;
static const int   TILE     = 256;
static const int   STEPS    = 10;
static const float DT       = 0.001f;
static const float EPS2     = 1e-4f;   // softening

struct Body { float x, y, z, vx, vy, vz, mass; };

// ── Tiled force accumulation ──────────────────────────────────────────────────
// Each thread computes the total force on one body from all other bodies.
// Bodies are loaded tile-by-tile into shared memory.
__global__ void compute_forces(Body *bodies, float3 *forces, int n) {
    __shared__ float4 tile[TILE];   // packed (x,y,z,mass)

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float3 pos_i = {0, 0, 0};
    if (i < n) { pos_i = {bodies[i].x, bodies[i].y, bodies[i].z}; }

    float3 acc = {0.f, 0.f, 0.f};

    int n_tiles = (n + TILE - 1) / TILE;
    for (int t = 0; t < n_tiles; t++) {
        int j = t * TILE + threadIdx.x;
        tile[threadIdx.x] = (j < n) ?
            make_float4(bodies[j].x, bodies[j].y, bodies[j].z, bodies[j].mass) :
            make_float4(0, 0, 0, 0);
        __syncthreads();

        if (i < n) {
            #pragma unroll 8
            for (int k = 0; k < TILE; k++) {
                float dx = tile[k].x - pos_i.x;
                float dy = tile[k].y - pos_i.y;
                float dz = tile[k].z - pos_i.z;
                float dist2 = dx*dx + dy*dy + dz*dz + EPS2;
                float inv_d3 = rsqrtf(dist2) / dist2;   // 1/r^3
                float f = tile[k].w * inv_d3;            // mass/r^3
                acc.x += dx * f;
                acc.y += dy * f;
                acc.z += dz * f;
            }
        }
        __syncthreads();
    }
    if (i < n) forces[i] = acc;
}

// ── Leapfrog velocity & position update ──────────────────────────────────────
__global__ void integrate(Body *bodies, const float3 *forces, int n, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    bodies[i].vx += forces[i].x * dt;
    bodies[i].vy += forces[i].y * dt;
    bodies[i].vz += forces[i].z * dt;
    bodies[i].x  += bodies[i].vx * dt;
    bodies[i].y  += bodies[i].vy * dt;
    bodies[i].z  += bodies[i].vz * dt;
}

int main(void) {
    size_t body_bytes  = N_BODIES * sizeof(Body);
    size_t force_bytes = N_BODIES * sizeof(float3);

    Body *h_bodies = (Body *)malloc(body_bytes);
    srand(42);
    for (int i = 0; i < N_BODIES; i++) {
        auto rnd = []{ return (float)rand()/RAND_MAX * 2.f - 1.f; };
        h_bodies[i] = {rnd(), rnd(), rnd(), rnd()*0.1f, rnd()*0.1f, rnd()*0.1f, 1.f};
    }

    Body   *d_bodies;
    float3 *d_forces;
    CUDA_CHECK(cudaMalloc(&d_bodies, body_bytes));
    CUDA_CHECK(cudaMalloc(&d_forces, force_bytes));
    CUDA_CHECK(cudaMemcpy(d_bodies, h_bodies, body_bytes, cudaMemcpyHostToDevice));

    int blocks = (N_BODIES + TILE - 1) / TILE;

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    for (int s = 0; s < STEPS; s++) {
        compute_forces<<<blocks, TILE>>>(d_bodies, d_forces, N_BODIES);
        integrate     <<<blocks, TILE>>>(d_bodies, d_forces, N_BODIES, DT);
    }

    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);

    double gflops = (double)STEPS * N_BODIES * N_BODIES * 20.0 / (ms * 1e-3) / 1e9;
    printf("N-Body: N=%d  steps=%d  %.2f ms  %.2f GFLOP/s\n",
           N_BODIES, STEPS, ms, gflops);

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_bodies); cudaFree(d_forces);
    free(h_bodies);
    return 0;
}
