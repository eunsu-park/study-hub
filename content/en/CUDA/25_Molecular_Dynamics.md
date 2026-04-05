# 25. Molecular Dynamics

**Previous**: [Fluid Dynamics LBM](./24_Fluid_Dynamics_LBM.md) | **Next**: [Image Processing GPU](./26_Image_Processing_GPU.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Evaluate the Lennard-Jones potential and compute pairwise forces between atoms
2. Implement a neighbor list (Verlet list) to reduce force computation from O(N²) to O(N·k)
3. Apply periodic boundary conditions with the minimum image convention
4. Integrate equations of motion using velocity Verlet and verify energy conservation
5. Implement a velocity-rescaling thermostat for NVT ensemble simulation

---

## 1. Lennard-Jones Potential

The Lennard-Jones (LJ) 12-6 potential models van der Waals interactions between noble gas atoms (argon, etc.):

```
U(r) = 4ε · [(σ/r)^12 - (σ/r)^6]

r     = interparticle distance
ε     = depth of potential well (energy scale)
σ     = distance at which U=0 (length scale)

r_min = 2^(1/6) σ ≈ 1.122 σ   (equilibrium distance, U = -ε)
r > r_min: attractive  (r^-6 dominates)
r < r_min: repulsive   (r^-12 dominates — very steep)
```

Force (negative gradient of potential):

```
F(r) = -dU/dr = 4ε · [12σ^12/r^13 - 6σ^6/r^7] (along r̂)
     = (48ε/r²) · [(σ/r)^12 - 0.5·(σ/r)^6] · r_vec
```

In reduced LJ units (ε = σ = m = 1), the critical cutoff is typically r_cut = 2.5σ — beyond this, LJ forces are < 1% of the minimum.

---

## 2. Naive Pairwise Force Kernel

```c
// LJ force computation — O(N²), one thread per atom
__global__ void lj_forces_naive(
    const float4 *pos,    // (x, y, z, type)
    float4       *force,  // (fx, fy, fz, potential)
    int N, float L,       // box length (cubic periodic box)
    float r_cut2)         // r_cut² (avoid sqrt for initial filter)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float4 pi = pos[i];
    float fx = 0.f, fy = 0.f, fz = 0.f, pe = 0.f;

    for (int j = 0; j < N; j++) {
        if (j == i) continue;
        float4 pj = pos[j];

        // Minimum image convention (periodic BC)
        float dx = pj.x - pi.x;
        float dy = pj.y - pi.y;
        float dz = pj.z - pi.z;
        dx -= L * rintf(dx / L);   // round to nearest image
        dy -= L * rintf(dy / L);
        dz -= L * rintf(dz / L);

        float r2 = dx*dx + dy*dy + dz*dz;
        if (r2 >= r_cut2) continue;

        // LJ in reduced units (ε=σ=1):  F = 48/r² * (1/r^12 - 0.5/r^6)
        float r2i  = 1.0f / r2;
        float r6i  = r2i * r2i * r2i;
        float fscl = 48.0f * r2i * r6i * (r6i - 0.5f);

        fx += fscl * dx;
        fy += fscl * dy;
        fz += fscl * dz;
        pe += 4.0f * r6i * (r6i - 1.0f);   // potential (only half — Newton's 3rd law pair)
    }
    pe *= 0.5f;  // each pair counted twice

    force[i] = make_float4(fx, fy, fz, pe);
}
```

For N=10,000 atoms this is 100M pair evaluations per step — expensive but correct. The neighbor list below reduces this by ~100×.

---

## 3. Verlet Neighbor List

The neighbor list stores all pairs (i, j) with r < r_cut + r_skin for atom i. The skin distance r_skin (typically 0.3σ) means the list is valid for several time steps before needing a rebuild:

```c
// Build neighbor list on GPU
// d_neighbors[i * MAX_NBRS + k] = j  (kth neighbor of atom i)
// d_num_nbrs[i]                 = number of neighbors of atom i
__global__ void build_neighbor_list(
    const float4 *pos,
    int *d_neighbors, int *d_num_nbrs,
    int N, float L, float r_list2,  // (r_cut + r_skin)²
    int MAX_NBRS)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float4 pi = pos[i];
    int count = 0;

    for (int j = 0; j < N; j++) {
        if (j == i) continue;
        float4 pj = pos[j];
        float dx = pj.x - pi.x; dx -= L * rintf(dx / L);
        float dy = pj.y - pi.y; dy -= L * rintf(dy / L);
        float dz = pj.z - pi.z; dz -= L * rintf(dz / L);
        float r2 = dx*dx + dy*dy + dz*dz;

        if (r2 < r_list2 && count < MAX_NBRS)
            d_neighbors[i * MAX_NBRS + count++] = j;
    }
    d_num_nbrs[i] = count;
}

// Force computation using neighbor list — O(N * avg_neighbors)
__global__ void lj_forces_nblist(
    const float4 *pos,
    const int *d_neighbors, const int *d_num_nbrs,
    float4 *force, int N, float L, float r_cut2, int MAX_NBRS)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float4 pi = pos[i];
    float fx = 0.f, fy = 0.f, fz = 0.f, pe = 0.f;
    int nnbr = d_num_nbrs[i];

    for (int k = 0; k < nnbr; k++) {
        int j = d_neighbors[i * MAX_NBRS + k];
        float4 pj = pos[j];
        float dx = pj.x - pi.x; dx -= L * rintf(dx / L);
        float dy = pj.y - pi.y; dy -= L * rintf(dy / L);
        float dz = pj.z - pi.z; dz -= L * rintf(dz / L);
        float r2 = dx*dx + dy*dy + dz*dz;
        if (r2 >= r_cut2) continue;

        float r2i  = 1.0f / r2;
        float r6i  = r2i * r2i * r2i;
        float fscl = 48.0f * r2i * r6i * (r6i - 0.5f);
        fx += fscl * dx; fy += fscl * dy; fz += fscl * dz;
        pe += 2.0f * r6i * (r6i - 1.0f);
    }
    force[i] = make_float4(fx, fy, fz, pe);
}
```

**Rebuild frequency**: rebuild when any atom has moved more than r_skin/2 since the last rebuild. Check with a displacement kernel + reduction.

---

## 4. Velocity Verlet Integration

```c
// Step 1: half-step velocity update + full position update
__global__ void verlet_step1(float4 *pos, float4 *vel, const float4 *force,
                              int N, float dt, float L) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float4 p = pos[i], v = vel[i], f = force[i];
    float dt2 = 0.5f * dt;

    // Half-step velocity (assume mass=1 in LJ units)
    v.x += dt2 * f.x;  v.y += dt2 * f.y;  v.z += dt2 * f.z;
    // Full-step position
    p.x += dt * v.x;   p.y += dt * v.y;   p.z += dt * v.z;

    // Periodic box wrapping
    p.x -= L * floorf(p.x / L);
    p.y -= L * floorf(p.y / L);
    p.z -= L * floorf(p.z / L);

    pos[i] = p; vel[i] = v;
}

// Step 2: complete velocity update with new forces
__global__ void verlet_step2(float4 *vel, const float4 *force, int N, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float4 v = vel[i], f = force[i];
    float dt2 = 0.5f * dt;
    v.x += dt2 * f.x;  v.y += dt2 * f.y;  v.z += dt2 * f.z;
    vel[i] = v;
}
```

---

## 5. Energy Conservation and NVT Thermostat

**Energy monitoring** (NVE ensemble check):

```c
// Kinetic energy: KE = 0.5 * Σ m*v²  (m=1 in LJ units)
__global__ void kinetic_energy(const float4 *vel, float *ke_partial, int N) {
    extern __shared__ float sdata[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float4 v = (i < N) ? vel[i] : make_float4(0,0,0,0);
    sdata[threadIdx.x] = 0.5f * (v.x*v.x + v.y*v.y + v.z*v.z);
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) ke_partial[blockIdx.x] = sdata[0];
}

// NVT: velocity rescaling thermostat
// Scale velocities so KE = target_KE = 1.5 * N * k_B * T (k_B=1 in LJ units)
__global__ void rescale_velocities(float4 *vel, float scale, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    vel[i].x *= scale;
    vel[i].y *= scale;
    vel[i].z *= scale;
}

void apply_thermostat(float4 *d_vel, int N, float T_target) {
    float ke = compute_kinetic_energy(d_vel, N);  // use reduction
    float T_current = 2.0f * ke / (3.0f * N);     // k_B=1 in LJ units
    float scale = sqrtf(T_target / T_current);
    rescale_velocities<<<(N+255)/256, 256>>>(d_vel, scale, N);
}
```

---

## 6. Complete MD Loop

```c
void run_md(int N, int steps, float dt, float T, float L) {
    const int BLOCK = 256;
    const float r_cut = 2.5f, r_skin = 0.3f;
    const float r_cut2  = r_cut * r_cut;
    const float r_list2 = (r_cut + r_skin) * (r_cut + r_skin);
    const int   MAX_NBRS = 200;

    float4 *d_pos, *d_vel, *d_force;
    int    *d_nbr, *d_nnbr;
    cudaMalloc(&d_pos,  N * sizeof(float4));
    cudaMalloc(&d_vel,  N * sizeof(float4));
    cudaMalloc(&d_force, N * sizeof(float4));
    cudaMalloc(&d_nbr,  N * MAX_NBRS * sizeof(int));
    cudaMalloc(&d_nnbr, N * sizeof(int));

    // Initialize FCC lattice positions and Maxwell-Boltzmann velocities
    init_fcc_lattice(d_pos, d_vel, N, L, T);

    // Initial neighbor list
    build_neighbor_list<<<(N+BLOCK-1)/BLOCK, BLOCK>>>(
        d_pos, d_nbr, d_nnbr, N, L, r_list2, MAX_NBRS);
    lj_forces_nblist<<<(N+BLOCK-1)/BLOCK, BLOCK>>>(
        d_pos, d_nbr, d_nnbr, d_force, N, L, r_cut2, MAX_NBRS);

    for (int t = 0; t < steps; t++) {
        // Verlet step 1: v += 0.5*dt*f, x += dt*v
        verlet_step1<<<(N+BLOCK-1)/BLOCK, BLOCK>>>(d_pos, d_vel, d_force, N, dt, L);

        // Rebuild neighbor list if needed (check max displacement)
        if (need_rebuild(d_pos, d_pos_ref, N, r_skin))
            build_neighbor_list<<<(N+BLOCK-1)/BLOCK, BLOCK>>>(
                d_pos, d_nbr, d_nnbr, N, L, r_list2, MAX_NBRS);

        // Recompute forces
        lj_forces_nblist<<<(N+BLOCK-1)/BLOCK, BLOCK>>>(
            d_pos, d_nbr, d_nnbr, d_force, N, L, r_cut2, MAX_NBRS);

        // Verlet step 2: v += 0.5*dt*f_new
        verlet_step2<<<(N+BLOCK-1)/BLOCK, BLOCK>>>(d_vel, d_force, N, dt);

        // NVT thermostat (apply every 10 steps)
        if (t % 10 == 0) apply_thermostat(d_vel, N, T);

        // Diagnostic output
        if (t % 100 == 0) {
            float ke = compute_kinetic_energy(d_vel, N);
            float pe = compute_potential_energy(d_force, N);
            printf("Step %d: KE=%.4f PE=%.4f E_tot=%.4f T=%.4f\n",
                   t, ke, pe, ke+pe, 2.f*ke/(3.f*N));
        }
    }

    cudaFree(d_pos); cudaFree(d_vel); cudaFree(d_force);
    cudaFree(d_nbr); cudaFree(d_nnbr);
}
```

---

## Key Takeaways

- The **Lennard-Jones potential** models short-range repulsion (r^-12) and van der Waals attraction (r^-6); cutoff at r_cut = 2.5σ reduces computation with <1% energy error
- **Minimum image convention**: for periodic BC, shift each pair distance by the nearest box image (subtract L × round(Δr/L))
- **Verlet neighbor list** reduces O(N²) force computation to O(N·k) where k ≈ average neighbors in the cutoff sphere; rebuild every ~20 steps using the skin r_skin
- **Velocity Verlet** is the MD integrator of choice: time-reversible, second-order, and conserves energy over long runs
- Energy conservation (stable total energy) is the key correctness check for NVE ensemble; drift indicates too large a time step or a bug
- **Velocity rescaling** (NVT thermostat) instantaneously sets temperature but is not rigorously NVT; Nosé-Hoover thermostats are preferred for production

---

**Next**: [26. Image Processing GPU](./26_Image_Processing_GPU.md) — Apply Gaussian blur, Sobel edge detection, bilateral filtering, and histogram equalization using GPU kernels and texture memory.
