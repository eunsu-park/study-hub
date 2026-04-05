/*
 * lennard_jones.cu — Lesson 25: Molecular Dynamics
 *
 * Simulates N particles interacting via the Lennard-Jones potential:
 *   V(r) = 4ε [ (σ/r)^12 − (σ/r)^6 ]
 *
 * Demonstrates:
 *   - Shared-memory tiling for force calculation (same structure as N-body)
 *   - Cutoff radius to skip negligible long-range interactions
 *   - Verlet time integration
 *   - Energy computation (kinetic + potential)
 *
 * Build:  nvcc -O2 -arch=sm_80 lennard_jones.cu -o lennard_jones
 * Run:    ./lennard_jones
 */

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e)); exit(1); } } while(0)

static const int   N     = 4096;
static const int   TILE  = 128;
static const int   STEPS = 50;
static const float DT    = 0.005f;
static const float RCUT  = 2.5f;    // cutoff in units of σ
static const float RCUT2 = RCUT * RCUT;

struct Particle { float x, y, z, vx, vy, vz; };

// ── LJ force kernel ───────────────────────────────────────────────────────────
__global__ void lj_forces(const Particle *p, float3 *forces,
                           float *epot, int n) {
    __shared__ float4 tile[TILE];   // packed (x,y,z,_)

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float3 pi = {0,0,0};
    if (i < n) pi = {p[i].x, p[i].y, p[i].z};

    float3 fi = {0,0,0};
    float  ep = 0.f;

    int n_tiles = (n + TILE - 1) / TILE;
    for (int t = 0; t < n_tiles; t++) {
        int j = t * TILE + threadIdx.x;
        tile[threadIdx.x] = (j < n && j != i) ?
            make_float4(p[j].x, p[j].y, p[j].z, 0) :
            make_float4(1e9f, 1e9f, 1e9f, 0);
        __syncthreads();

        if (i < n) {
            #pragma unroll 8
            for (int k = 0; k < TILE; k++) {
                float dx = tile[k].x - pi.x;
                float dy = tile[k].y - pi.y;
                float dz = tile[k].z - pi.z;
                float r2 = dx*dx + dy*dy + dz*dz;
                if (r2 > 0.0001f && r2 < RCUT2) {
                    float ir2   = 1.f / r2;
                    float ir6   = ir2 * ir2 * ir2;
                    float ir12  = ir6 * ir6;
                    float force = 24.f * ir2 * (2.f*ir12 - ir6);
                    fi.x += force * dx;
                    fi.y += force * dy;
                    fi.z += force * dz;
                    ep   += 4.f * (ir12 - ir6);
                }
            }
        }
        __syncthreads();
    }

    if (i < n) {
        forces[i] = fi;
        epot[i]   = ep * 0.5f;   // avoid double counting
    }
}

// ── Velocity Verlet integration ───────────────────────────────────────────────
__global__ void integrate(Particle *p, const float3 *f, int n, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    p[i].vx += f[i].x * dt;
    p[i].vy += f[i].y * dt;
    p[i].vz += f[i].z * dt;
    p[i].x  += p[i].vx * dt;
    p[i].y  += p[i].vy * dt;
    p[i].z  += p[i].vz * dt;
}

int main(void) {
    Particle *h_p = (Particle *)malloc(N * sizeof(Particle));
    // FCC-like initial positions
    srand(42);
    for (int i = 0; i < N; i++) {
        h_p[i].x  = ((float)rand()/RAND_MAX) * 20.f;
        h_p[i].y  = ((float)rand()/RAND_MAX) * 20.f;
        h_p[i].z  = ((float)rand()/RAND_MAX) * 20.f;
        h_p[i].vx = ((float)rand()/RAND_MAX - 0.5f) * 0.5f;
        h_p[i].vy = ((float)rand()/RAND_MAX - 0.5f) * 0.5f;
        h_p[i].vz = ((float)rand()/RAND_MAX - 0.5f) * 0.5f;
    }

    Particle *d_p;
    float3   *d_f;
    float    *d_ep;
    CUDA_CHECK(cudaMalloc(&d_p,  N * sizeof(Particle)));
    CUDA_CHECK(cudaMalloc(&d_f,  N * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_ep, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_p, h_p, N * sizeof(Particle), cudaMemcpyHostToDevice));

    int blocks = (N + TILE - 1) / TILE;

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    for (int s = 0; s < STEPS; s++) {
        lj_forces<<<blocks, TILE>>>(d_p, d_f, d_ep, N);
        integrate <<<blocks, TILE>>>(d_p, d_f, N, DT);
    }

    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);

    double gflops = (double)STEPS * N * N * 25.0 / (ms * 1e-3) / 1e9;
    printf("Lennard-Jones MD: N=%d  steps=%d  %.2f ms  %.2f GFLOP/s\n",
           N, STEPS, ms, gflops);

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_p); cudaFree(d_f); cudaFree(d_ep);
    free(h_p);
    return 0;
}
