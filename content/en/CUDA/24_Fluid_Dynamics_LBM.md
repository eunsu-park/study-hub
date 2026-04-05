# 24. Fluid Dynamics — Lattice Boltzmann Method

**Previous**: [PDE Solvers Heat Equation](./23_PDE_Solvers_Heat_Equation.md) | **Next**: [Molecular Dynamics](./25_Molecular_Dynamics.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the D2Q9 velocity model and the physical meaning of distribution functions
2. Implement the BGK collision step (relaxation to equilibrium)
3. Implement the streaming step by shifting distribution functions
4. Apply no-slip bounce-back boundary conditions for solid walls
5. Recover macroscopic density and velocity from distribution functions and validate against the lid-driven cavity benchmark

---

## 1. Lattice Boltzmann Overview

The Lattice Boltzmann Method (LBM) simulates fluid dynamics by tracking the evolution of mesoscopic **particle distribution functions** f_i on a regular grid, rather than solving the Navier-Stokes equations directly.

Each grid node holds Q distribution values (one per velocity direction). At each time step:
1. **Collision**: relax distributions toward local equilibrium
2. **Streaming**: shift distributions along their velocity directions

LBM is ideal for GPU because:
- All collision steps are **local** (no neighbors needed)
- Streaming is a **regular shift pattern** (structured memory access)
- The algorithm is embarrassingly parallel per grid node

---

## 2. D2Q9 Velocity Model

The D2Q9 model uses 9 discrete velocities on a 2D grid:

```
Velocity indices and directions:
  6  2  5
  3  0  1      e_i = velocity direction i
  7  4  8

e_0 = ( 0,  0)    weight w_0 = 4/9
e_1 = ( 1,  0)    weight w_1 = 1/9
e_2 = ( 0,  1)    weight w_2 = 1/9
e_3 = (-1,  0)    weight w_3 = 1/9
e_4 = ( 0, -1)    weight w_4 = 1/9
e_5 = ( 1,  1)    weight w_5 = 1/36
e_6 = (-1,  1)    weight w_6 = 1/36
e_7 = (-1, -1)    weight w_7 = 1/36
e_8 = ( 1, -1)    weight w_8 = 1/36
```

```c
// D2Q9 constants
__constant__ int ex[9] = { 0,  1,  0, -1,  0,  1, -1, -1,  1};
__constant__ int ey[9] = { 0,  0,  1,  0, -1,  1,  1, -1, -1};
__constant__ float w[9] = {4.f/9, 1.f/9, 1.f/9, 1.f/9, 1.f/9,
                            1.f/36, 1.f/36, 1.f/36, 1.f/36};
// Opposite direction for bounce-back
__constant__ int opp[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
```

---

## 3. Macroscopic Variables

Density ρ and momentum ρu are moments of the distribution function:

```
ρ(x, t)    = Σ_i f_i(x, t)               (zeroth moment = density)
ρ·u(x, t)  = Σ_i e_i · f_i(x, t)        (first moment = momentum)
u = ρu / ρ                                (velocity)
```

```c
// Recover macroscopic density and velocity from f_i
__device__ void macro_vars(const float *f, float *rho, float *ux, float *uy) {
    *rho = 0.f;  *ux = 0.f;  *uy = 0.f;
    for (int q = 0; q < 9; q++) {
        *rho += f[q];
        *ux  += ex[q] * f[q];
        *uy  += ey[q] * f[q];
    }
    *ux /= *rho;
    *uy /= *rho;
}
```

---

## 4. BGK Equilibrium and Collision

The BGK (Bhatnagar-Gross-Krook) collision relaxes f_i toward the local Maxwell-Boltzmann equilibrium f_i^eq at rate 1/τ:

```
f_i^eq = w_i · ρ · [1 + (e_i·u)/c_s² + (e_i·u)²/(2c_s⁴) - u²/(2c_s²)]

where c_s² = 1/3  (lattice speed of sound squared in LB units)

BGK collision:
f_i*(x,t) = f_i(x,t) - (1/τ) · [f_i(x,t) - f_i^eq(x,t)]

τ = relaxation time, related to kinematic viscosity: ν = c_s²(τ - 0.5)
Reynolds number: Re = U·L / ν = U·L / [c_s²(τ - 0.5)]
```

```c
// Compute equilibrium distribution
__device__ float f_eq(int q, float rho, float ux, float uy) {
    float eu  = ex[q] * ux + ey[q] * uy;     // e_i · u
    float u2  = ux * ux + uy * uy;            // |u|²
    // c_s² = 1/3, so 1/c_s² = 3, 1/(2c_s²) = 3/2, 1/(2c_s⁴) = 9/2
    return w[q] * rho * (1.f + 3.f*eu + 4.5f*eu*eu - 1.5f*u2);
}

// Combined collision kernel (in-place on f array)
__global__ void collide(float *f, const bool *solid, int Nx, int Ny, float tau_inv) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= Nx || y >= Ny || solid[y * Nx + x]) return;

    int node = y * Nx + x;
    float fi[9];
    for (int q = 0; q < 9; q++) fi[q] = f[node * 9 + q];

    float rho, ux, uy;
    macro_vars(fi, &rho, &ux, &uy);

    for (int q = 0; q < 9; q++) {
        float feq = f_eq(q, rho, ux, uy);
        f[node * 9 + q] = fi[q] - tau_inv * (fi[q] - feq);
    }
}
```

---

## 5. Streaming Step

After collision, each distribution function f_i is shifted to the neighboring node in direction e_i:

```
f_i(x + e_i, t+1) ← f_i*(x, t)    (streaming)
```

```c
// Streaming: pull scheme — each node reads from upstream neighbors
// (avoids race conditions without double buffering)
__global__ void stream(const float *f_in, float *f_out,
                       const bool *solid, int Nx, int Ny) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= Nx || y >= Ny) return;

    int node = y * Nx + x;

    for (int q = 0; q < 9; q++) {
        // Source node: where direction q came from
        int xs = (x - ex[q] + Nx) % Nx;   // periodic wrap for open boundaries
        int ys = (y - ey[q] + Ny) % Ny;
        int src = ys * Nx + xs;

        if (solid[src]) {
            // Bounce-back: reverse direction from solid node
            f_out[node * 9 + q] = f_in[node * 9 + opp[q]];
        } else {
            f_out[node * 9 + q] = f_in[src * 9 + q];
        }
    }
}
```

The **pull scheme** reads from neighbors rather than pushing to them — this is race-condition-free and GPU-friendly.

---

## 6. Bounce-Back Boundary Condition (No-Slip)

The bounce-back rule reverses the incoming distribution at solid nodes, resulting in a no-slip condition (velocity = 0 at the wall):

```
At solid boundary: f_i(wall, t+1) = f_{opp(i)}(wall, t)

where opp(i) is the opposite direction to i:
  opp(1) = 3   (right ↔ left)
  opp(2) = 4   (up ↔ down)
  opp(5) = 7   (up-right ↔ down-left)
  etc.
```

For a moving wall (e.g., lid-driven cavity), add a momentum correction:

```c
// Moving lid (top wall at y=Ny-1 moves at velocity u_lid in x-direction)
__global__ void moving_lid_bc(float *f, int Nx, int Ny, float u_lid) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= Nx) return;
    int y = Ny - 1;
    int node = y * Nx + x;

    float rho_lid = 0.f;
    // Estimate density from known distributions
    rho_lid = f[node*9+0] + f[node*9+1] + f[node*9+3]
            + 2.f*(f[node*9+2] + f[node*9+5] + f[node*9+6]);
    rho_lid /= (1.f + 1.5f * u_lid);  // from Zou-He velocity BC

    // Apply Zou-He velocity boundary condition for f_4, f_7, f_8
    f[node*9+4] = f[node*9+2] - (2.f/3.f) * rho_lid * 0.f;       // vy=0
    f[node*9+7] = f[node*9+5] - 0.5f*(f[node*9+1]-f[node*9+3])
                              - (1.f/6.f)*rho_lid*u_lid;
    f[node*9+8] = f[node*9+6] + 0.5f*(f[node*9+1]-f[node*9+3])
                              + (1.f/6.f)*rho_lid*u_lid;
}
```

---

## 7. Main LBM Loop and Reynolds Number

```c
void run_lbm(int Nx, int Ny, int steps, float tau, float u_lid) {
    float tau_inv = 1.0f / tau;
    float nu = (1.0f/3.0f) * (tau - 0.5f);  // kinematic viscosity
    float Re = u_lid * Ny / nu;
    printf("Re = %.1f, tau = %.3f, nu = %.5f\n", Re, tau, nu);

    // Allocate: Nx * Ny nodes × 9 distributions
    size_t bytes = Nx * Ny * 9 * sizeof(float);
    float *d_f0, *d_f1;
    bool  *d_solid;
    cudaMalloc(&d_f0,   bytes);
    cudaMalloc(&d_f1,   bytes);
    cudaMalloc(&d_solid, Nx * Ny * sizeof(bool));

    // Initialize: f = f_eq(rho=1, ux=0, uy=0) everywhere
    init_equilibrium<<<dim3((Nx+15)/16,(Ny+15)/16), dim3(16,16)>>>(d_f0, Nx, Ny);

    // Mark solid nodes (walls at y=0, y=Ny-1, x=0, x=Nx-1)
    mark_solid_walls<<<(Nx*Ny+255)/256, 256>>>(d_solid, Nx, Ny);

    dim3 block(16, 16), grid((Nx+15)/16, (Ny+15)/16);

    for (int t = 0; t < steps; t++) {
        collide<<<grid, block>>>(d_f0, d_solid, Nx, Ny, tau_inv);
        moving_lid_bc<<<(Nx+255)/256, 256>>>(d_f0, Nx, Ny, u_lid);
        stream<<<grid, block>>>(d_f0, d_f1, d_solid, Nx, Ny);

        // Swap buffers
        float *tmp = d_f0; d_f0 = d_f1; d_f1 = tmp;
    }

    // Extract velocity field for visualization
    extract_velocity<<<grid, block>>>(d_f0, d_ux, d_uy, d_solid, Nx, Ny);

    cudaFree(d_f0); cudaFree(d_f1); cudaFree(d_solid);
}
```

**Benchmark target**: on an RTX 3090, a 1024×1024 D2Q9 LBM simulation achieves ~3 × 10⁹ node-updates per second (3 GNUPS).

---

## Key Takeaways

- **D2Q9 LBM** uses 9 velocity directions per node; each node stores 9 distribution values f_i
- **Collision** (BGK): relax f_i toward equilibrium f_i^eq at rate 1/τ; τ controls viscosity and Reynolds number
- **Streaming** (pull scheme): each node reads from upstream neighbors — naturally race-condition free
- **Bounce-back** reverses incoming distributions at solid walls, producing the no-slip condition
- **Moving wall** (lid-driven cavity): Zou-He velocity BC correctly specifies the wall velocity
- Reynolds number Re = U·L / ν, where ν = c_s²(τ - 0.5); keep Re < ~1000 for stable laminar flow with simple BGK

---

**Next**: [25. Molecular Dynamics](./25_Molecular_Dynamics.md) — Implement a Lennard-Jones molecular dynamics simulation with neighbor lists, velocity Verlet integration, and periodic boundary conditions.
