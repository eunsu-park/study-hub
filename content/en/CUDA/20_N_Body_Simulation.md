# 20. N-Body Simulation

**Previous**: [Sparse Matrix Ops](./19_Sparse_Matrix_Ops.md) | **Next**: [Monte Carlo Methods](./21_Monte_Carlo_Methods.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement O(N²) direct gravitational force summation on the GPU
2. Apply tile-based shared memory optimization to reduce global memory loads from N² to N²/TILE_SIZE
3. Integrate particle positions using the velocity Verlet scheme
4. Understand the softening parameter and why it is physically necessary
5. Describe the Barnes-Hut O(N log N) approximation and OpenGL interop for real-time visualization

---

## 1. The N-Body Problem

The gravitational force on particle i due to particle j:

```
F_ij = G * m_i * m_j / (r_ij² + ε²)^(3/2) * r_ij_vec

where r_ij_vec = (x_j - x_i, y_j - y_i, z_j - z_i)
      r_ij²    = dot(r_ij_vec, r_ij_vec)
      ε        = softening parameter (prevents division by zero when r→0)
```

Each particle i must sum contributions from all N-1 other particles — O(N²) work total. For N=10,000 particles, that is 100M force evaluations per time step. GPUs excel here: the computation is embarrassingly parallel and compute-intensive.

```
N=10,000 particles × 10,000 interactions × 20 FLOP per interaction = 2 GFLOP per step
RTX 3090 FP32 peak = 35.6 TFLOPS → potential speedup ~10,000× vs single CPU core
```

---

## 2. Naive N-Body Kernel

```c
// Particle state
struct float4;  // x, y, z, mass

// Compute acceleration for all particles (one thread per particle)
__global__ void nbody_naive(
    const float4 *pos,   // [N] (x, y, z, mass)
    float4       *acc,   // [N] (ax, ay, az, unused)
    int N, float G, float eps2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float4 pi = pos[i];
    float ax = 0.f, ay = 0.f, az = 0.f;

    for (int j = 0; j < N; j++) {
        float4 pj = pos[j];           // global memory load for every j
        float dx = pj.x - pi.x;
        float dy = pj.y - pi.y;
        float dz = pj.z - pi.z;
        float dist2 = dx*dx + dy*dy + dz*dz + eps2;
        float inv_dist3 = G * pj.w * rsqrtf(dist2) / dist2;  // G*m_j / r^3
        ax += dx * inv_dist3;
        ay += dy * inv_dist3;
        az += dz * inv_dist3;
    }

    acc[i] = make_float4(ax, ay, az, 0.f);
}
```

**Bottleneck**: each thread i loads `pos[j]` from global memory for every j. That is N loads per thread × N threads = N² global loads = 400M loads for N=10,000. At 4 bytes per float4: 6.4 GB. At 900 GB/s bandwidth: 7 ms per step — mostly memory bound.

---

## 3. Tile-Based Shared Memory Optimization

Load TILE_SIZE particles into shared memory; all threads in the block reuse those TILE_SIZE loads:

```c
#define TILE 256

__global__ void nbody_tiled(
    const float4 *pos, float4 *acc,
    int N, float G, float eps2)
{
    __shared__ float4 sh_pos[TILE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float4 pi = (i < N) ? pos[i] : make_float4(0,0,0,0);

    float ax = 0.f, ay = 0.f, az = 0.f;

    // Process input particles in tiles of TILE_SIZE
    for (int tile = 0; tile < (N + TILE - 1) / TILE; tile++) {
        // Collaboratively load one tile of particles into shared memory
        int j = tile * TILE + threadIdx.x;
        sh_pos[threadIdx.x] = (j < N) ? pos[j] : make_float4(0,0,0,0);
        __syncthreads();

        // Each thread i accumulates force from all TILE particles in sh_pos
        // (unrolled 4× for ILP)
        #pragma unroll 8
        for (int k = 0; k < TILE; k++) {
            float dx = sh_pos[k].x - pi.x;
            float dy = sh_pos[k].y - pi.y;
            float dz = sh_pos[k].z - pi.z;
            float dist2 = dx*dx + dy*dy + dz*dz + eps2;
            float inv_dist3 = G * sh_pos[k].w * rsqrtf(dist2) / dist2;
            ax += dx * inv_dist3;
            ay += dy * inv_dist3;
            az += dz * inv_dist3;
        }
        __syncthreads();
    }

    if (i < N) acc[i] = make_float4(ax, ay, az, 0.f);
}
```

**Memory traffic reduction**: with TILE=256, each block of 256 threads loads one tile of 256 particles (256 × 16 bytes = 4 KB) and reuses it 256 times. Global loads: N/TILE tiles × N threads × 1 load/tile = N²/TILE — a TILE-fold reduction.

```
TILE=256, N=10,000:
  Naive:  N² = 100M global loads
  Tiled:  N²/TILE = 390K global loads  → 256× fewer global memory bytes
```

The tiled kernel becomes **compute-bound** rather than memory-bound. The inner loop has ~20 FLOP per particle-pair and runs at near-peak FP32 throughput.

---

## 4. Velocity Verlet Integration

The velocity Verlet integrator is second-order accurate and time-reversible — ideal for N-body physics:

```
x(t + Δt) = x(t) + v(t)·Δt + 0.5·a(t)·Δt²
a(t + Δt) = F(x(t + Δt)) / m           (recompute forces at new position)
v(t + Δt) = v(t) + 0.5·(a(t) + a(t+Δt))·Δt
```

```c
// Verlet integration step (one thread per particle)
__global__ void integrate_verlet(
    float4 *pos,    // (x, y, z, mass)
    float4 *vel,    // (vx, vy, vz, 0)
    float4 *acc_old, // a(t)
    float4 *acc_new, // a(t+Δt) — already computed
    int N, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float4 p  = pos[i];
    float4 v  = vel[i];
    float4 a0 = acc_old[i];
    float4 a1 = acc_new[i];

    // Update position (using old acceleration)
    p.x += v.x * dt + 0.5f * a0.x * dt * dt;
    p.y += v.y * dt + 0.5f * a0.y * dt * dt;
    p.z += v.z * dt + 0.5f * a0.z * dt * dt;

    // Update velocity (average of old and new acceleration)
    v.x += 0.5f * (a0.x + a1.x) * dt;
    v.y += 0.5f * (a0.y + a1.y) * dt;
    v.z += 0.5f * (a0.z + a1.z) * dt;

    pos[i] = p;
    vel[i] = v;
}

// Main simulation loop
void simulate_nbody(int N, int steps, float dt, float G, float eps) {
    float eps2 = eps * eps;
    const int BLOCK = TILE;

    float4 *d_pos, *d_vel, *d_acc0, *d_acc1;
    cudaMalloc(&d_pos,  N * sizeof(float4));
    cudaMalloc(&d_vel,  N * sizeof(float4));
    cudaMalloc(&d_acc0, N * sizeof(float4));
    cudaMalloc(&d_acc1, N * sizeof(float4));

    // Initialize pos and vel on host and upload
    // ...

    // Initial force computation
    nbody_tiled<<<(N + BLOCK - 1) / BLOCK, BLOCK>>>(d_pos, d_acc0, N, G, eps2);

    for (int t = 0; t < steps; t++) {
        // 1. Update positions (uses acc0 = a(t))
        // simplified: just advance pos with current vel and half-step acc
        // (full verlet is split into two half-velocity updates)

        // 2. Recompute forces at new positions
        nbody_tiled<<<(N + BLOCK - 1) / BLOCK, BLOCK>>>(d_pos, d_acc1, N, G, eps2);

        // 3. Complete velocity update
        integrate_verlet<<<(N + BLOCK - 1) / BLOCK, BLOCK>>>(
            d_pos, d_vel, d_acc0, d_acc1, N, dt);

        // Swap acc buffers
        float4 *tmp = d_acc0; d_acc0 = d_acc1; d_acc1 = tmp;
    }

    cudaFree(d_pos); cudaFree(d_vel); cudaFree(d_acc0); cudaFree(d_acc1);
}
```

---

## 5. Softening Parameter

Without softening, two nearby particles produce a force that diverges as r → 0, causing numerical explosion. The softening length ε sets a minimum effective distance:

```
Unsoftened:  inv_dist3 = 1 / r^3       (diverges as r → 0)
Softened:    inv_dist3 = 1 / (r² + ε²)^(3/2)   (bounded, max ≈ 1/ε³)

Typical choice: ε ≈ 0.01 to 0.1 × mean inter-particle spacing
Too small: particles fly apart after close encounters (numerical instability)
Too large: forces are underestimated at short range (unphysical softening)
```

---

## 6. Barnes-Hut O(N log N) Approximation

For large N (>100K), direct O(N²) becomes too slow. **Barnes-Hut** builds an octree (3D) or quadtree (2D) and approximates distant clusters of particles as a single super-particle:

```
Criterion (opening angle θ):
  If cluster_size / distance_to_cluster < θ:
      treat entire cluster as one particle at its center of mass
  Else:
      recursively descend into sub-nodes

θ = 0.5 is typical: achieves O(N log N) complexity with < 1% force error
```

GPU Barnes-Hut is complex to implement from scratch (tree construction on GPU is non-trivial). The NVIDIA GPU Gems 3 chapter and the CUDA SDK sample provide reference implementations. For production, consider **Fast Multipole Method** (FMM) which achieves O(N) with higher-order accuracy.

---

## 7. OpenGL Interop (Visualization Concept)

To visualize N-body trajectories without copying back to CPU every frame:

```c
// Register a CUDA buffer that is also an OpenGL VBO
GLuint vbo;
glGenBuffers(1, &vbo);
glBindBuffer(GL_ARRAY_BUFFER, vbo);
glBufferData(GL_ARRAY_BUFFER, N * sizeof(float4), NULL, GL_DYNAMIC_DRAW);

cudaGraphicsResource_t cuda_vbo;
cudaGraphicsGLRegisterBuffer(&cuda_vbo, vbo, cudaGraphicsMapFlagsWriteDiscard);

// Each frame:
cudaGraphicsMapResources(1, &cuda_vbo, 0);
float4 *d_pos_gl;
size_t  bytes;
cudaGraphicsResourceGetMappedPointer((void**)&d_pos_gl, &bytes, cuda_vbo);

// Kernels write directly into the OpenGL buffer — no CPU round trip
nbody_tiled<<<grid, TILE>>>(d_pos_gl, d_acc, N, G, eps2);
integrate<<<grid, TILE>>>(d_pos_gl, d_vel, d_acc, N, dt);

cudaGraphicsUnmapResources(1, &cuda_vbo, 0);
// OpenGL renders directly from d_pos_gl
glDrawArrays(GL_POINTS, 0, N);
```

This enables real-time interactive visualization at 60 fps for N up to ~1M particles.

---

## Key Takeaways

- Direct N-body force computation is O(N²) — perfectly parallel (no communication needed between particles except accumulation)
- **Tiled shared memory** reduces global memory traffic by a factor of TILE_SIZE (e.g., 256×), turning a memory-bound kernel into a compute-bound kernel
- **Velocity Verlet** is the standard integrator: second-order accurate, time-reversible, and energy-conserving (compared to simple Euler)
- **Softening** (ε > 0) prevents force divergence at r → 0 and is physically motivated by finite particle size
- **Barnes-Hut** (O(N log N)) approximates far-field forces via hierarchical tree — necessary for N > 100K particles
- **CUDA-OpenGL interop** eliminates CPU-GPU round trips for visualization — particles are rendered directly from GPU memory

---

**Next**: [21. Monte Carlo Methods](./21_Monte_Carlo_Methods.md) — Generate random numbers on GPU with cuRAND and implement parallel Monte Carlo simulations for π estimation and Black-Scholes option pricing.
