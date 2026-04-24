# Lesson 25 — Molecular Dynamics (per-lesson exercise)

Prerequisites: L05 (shared memory), L08 (memory coalescing), L20 (N-body familiarity helpful).

Compile: `nvcc -O3 -arch=sm_80 ex.cu -o ex`

A molecular dynamics (MD) simulation integrates Newton's equations of motion for a collection of atoms or particles under a force field. The dominant cost is computing the pairwise forces — $O(N^2)$ in the naive case, $O(N \log N)$ with neighbor lists.

The GPU is a near-perfect fit: the force calculation parallelizes per particle, and the integrator step is embarrassingly parallel.

---

## Exercise 25.1 — All-Pairs Lennard-Jones Force

**Difficulty**: ★★★

### Problem

The Lennard-Jones potential is the simplest realistic interatomic potential:

$$U(r) = 4\epsilon\left[(\sigma/r)^{12} - (\sigma/r)^6\right]$$

The force on particle $i$ from particle $j$:

$$\vec{F}_{ij} = -\nabla U = 24\epsilon\left[2(\sigma/r)^{12} - (\sigma/r)^6\right] \hat{r}_{ij} / r$$

Implement an all-pairs force kernel where each thread computes the total force on one particle by summing contributions from all others:

```cuda
__global__ void lj_force_all_pairs(const float4 *pos, float4 *force,
                                   int N, float sigma, float epsilon) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float4 pi = pos[i];
    float fx = 0, fy = 0, fz = 0;

    for (int j = 0; j < N; j++) {
        if (i == j) continue;
        float4 pj = pos[j];
        float dx = pj.x - pi.x, dy = pj.y - pi.y, dz = pj.z - pi.z;
        float r2 = dx*dx + dy*dy + dz*dz + 1e-12f;   /* avoid divide by zero */
        float inv_r2 = 1.0f / r2;
        float s2 = sigma * sigma * inv_r2;
        float s6 = s2 * s2 * s2;
        float s12 = s6 * s6;
        float f_over_r = 24.0f * epsilon * (2.0f * s12 - s6) * inv_r2;
        fx += f_over_r * dx; fy += f_over_r * dy; fz += f_over_r * dz;
    }
    force[i] = make_float4(fx, fy, fz, 0);
}
```

For $N = 4096$ particles this is roughly 16 million pair evaluations per step — perfectly fine for GPU. Time it; expect 1-5 ms per step on a modern GPU.

---

## Exercise 25.2 — Velocity Verlet Integration

**Difficulty**: ★★

The Velocity Verlet integrator is the standard for MD because it conserves energy well over long simulations:

```
v(t + dt/2) = v(t) + 0.5 * a(t) * dt
x(t + dt)   = x(t) + v(t + dt/2) * dt
a(t + dt)   = compute_force(x(t + dt)) / mass
v(t + dt)   = v(t + dt/2) + 0.5 * a(t + dt) * dt
```

Wrap your force kernel from 25.1 in this integrator:

```cuda
__global__ void verlet_step1(float4 *pos, float4 *vel, const float4 *acc,
                             int N, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    vel[i].x += 0.5f * acc[i].x * dt;
    vel[i].y += 0.5f * acc[i].y * dt;
    vel[i].z += 0.5f * acc[i].z * dt;
    pos[i].x += vel[i].x * dt;
    pos[i].y += vel[i].y * dt;
    pos[i].z += vel[i].z * dt;
}
```

Then call `lj_force_all_pairs`, then a `verlet_step2` that does the second half-step on velocity. Run for 10000 steps; the total energy (kinetic + potential) should drift by < 1% if the timestep is appropriate (`dt ≈ 0.005` in reduced LJ units).

---

## Exercise 25.3 — Tiled Force with Shared Memory

**Difficulty**: ★★★

The all-pairs kernel reads each particle's position $N$ times from global memory. Tile $N$ into blocks of `BLOCK_SIZE`, load each tile into shared memory once, and reuse it for `BLOCK_SIZE` outputs:

```cuda
__shared__ float4 s_pos[BLOCK_SIZE];

for (int tile = 0; tile < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; tile++) {
    int j = tile * BLOCK_SIZE + threadIdx.x;
    s_pos[threadIdx.x] = (j < N) ? pos[j] : zero;
    __syncthreads();

    for (int k = 0; k < BLOCK_SIZE; k++) {
        /* same LJ force computation, but reads from s_pos[k] */
    }
    __syncthreads();
}
```

This reduces per-particle global reads from $N$ to $N / \text{BLOCK\_SIZE}$. Speedup is typically 3-5× over the naive version on $N \geq 4096$.

---

## Exercise 25.4 — Cell-List Acceleration — Bonus

**Difficulty**: ★★★★

For systems with a finite cutoff (LJ has effectively zero force at $r > 2.5\sigma$), the all-pairs cost of $O(N^2)$ is wasteful. A **cell list** spatially bins particles so each particle only interacts with neighbors in its own cell + 26 surrounding cells in 3D.

Implement a cell list:
1. Bin particles into cells of side ≈ cutoff.
2. For each particle, scan its 27 cells (including its own) and apply forces only to neighbors within the cutoff.

This drops the cost to $O(N)$ for sparse systems. Most production MD codes (LAMMPS, GROMACS) use cell lists or their refinements.
