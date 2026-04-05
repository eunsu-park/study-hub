# 23. PDE Solvers — Heat Equation

**Previous**: [FFT on GPU](./22_FFT_on_GPU.md) | **Next**: [Fluid Dynamics LBM](./24_Fluid_Dynamics_LBM.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Discretize the 2D heat equation using explicit finite differences
2. Write a CUDA stencil kernel for the discrete Laplacian operator
3. Identify the CFL stability condition and choose a safe time step
4. Implement Dirichlet and Neumann boundary conditions on the GPU
5. Measure convergence to steady state using the L2-norm of the residual

---

## 1. The 2D Heat Equation

The heat equation governs diffusion of temperature u(x, y, t) in a 2D domain:

```
∂u/∂t = α · (∂²u/∂x² + ∂²u/∂y²)

α = thermal diffusivity [m²/s]
∇²u = Laplacian = ∂²u/∂x² + ∂²u/∂y² (second spatial derivative)

Physical interpretation:
  u increases where its neighbors are hotter (positive Laplacian)
  u decreases where its neighbors are cooler (negative Laplacian)
  At steady state: ∇²u = 0  (Laplace equation)
```

---

## 2. Explicit Finite Difference Discretization

Discretize the domain into an Nx × Ny grid with spacing Δx = Δy = h:

```
Space:  u[i][j] ≈ u(i·h, j·h)        i = 0..Ny-1, j = 0..Nx-1
Time:   u^n[i][j] = u at time step n

Forward Euler in time:
  ∂u/∂t ≈ (u^{n+1}[i][j] - u^n[i][j]) / Δt

Central difference in space:
  ∂²u/∂x² ≈ (u[i][j-1] - 2u[i][j] + u[i][j+1]) / h²
  ∂²u/∂y² ≈ (u[i-1][j] - 2u[i][j] + u[i+1][j]) / h²

Combined update rule (let r = α·Δt/h²):
  u^{n+1}[i][j] = u^n[i][j] + r · (u[i-1][j] + u[i+1][j]
                                   + u[i][j-1] + u[i][j+1]
                                   - 4·u[i][j])
```

---

## 3. CFL Stability Condition

The explicit scheme is **conditionally stable** — only stable if the time step is small enough:

```
CFL condition for 2D heat equation:
  r = α·Δt/h² ≤ 1/4

Therefore: Δt ≤ h² / (4·α)

Example: α = 0.1, h = 0.01
  Δt_max = 0.01² / (4 × 0.1) = 0.0025 seconds per step
  To simulate T = 1 second: need at least 400 time steps

Violation (r > 0.25): solution grows without bound (numerical explosion)
```

```c
// Validate parameters before launching simulation
bool check_cfl(float alpha, float dt, float h) {
    float r = alpha * dt / (h * h);
    if (r > 0.25f) {
        fprintf(stderr, "CFL violation! r=%.4f > 0.25. Reduce dt to %.6f\n",
                r, 0.25f * h * h / alpha);
        return false;
    }
    printf("CFL: r=%.4f (stable, max=0.25)\n", r);
    return true;
}
```

---

## 4. Heat Equation Kernel

```c
// 2D heat equation update: one thread per interior grid point
__global__ void heat_eq_step(
    const float *u_old, float *u_new,
    int Nx, int Ny, float r)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // column
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // row

    if (i <= 0 || i >= Ny - 1 || j <= 0 || j >= Nx - 1) return;

    int idx  = i * Nx + j;
    float u0 = u_old[idx];

    float laplacian =
        u_old[(i - 1) * Nx + j] +   // above
        u_old[(i + 1) * Nx + j] +   // below
        u_old[i * Nx + (j - 1)] +   // left
        u_old[i * Nx + (j + 1)] -   // right
        4.0f * u0;

    u_new[idx] = u0 + r * laplacian;
}

// Shared memory version (see Lesson 17 for full halo implementation)
__global__ void heat_eq_step_shared(
    const float *u_old, float *u_new,
    int Nx, int Ny, float r)
{
    __shared__ float s[18][18];  // 16×16 block + 1-cell halo on each side
    int tx = threadIdx.x, ty = threadIdx.y;
    int j  = blockIdx.x * 16 + tx;
    int i  = blockIdx.y * 16 + ty;

    // Load center
    s[ty + 1][tx + 1] = (i < Ny && j < Nx) ? u_old[i * Nx + j] : 0.f;

    // Load halos (simplified: only top/bottom shown)
    if (ty == 0)
        s[0][tx + 1] = (i > 0 && j < Nx) ? u_old[(i - 1) * Nx + j] : 0.f;
    if (ty == 15)
        s[17][tx + 1] = (i < Ny - 1 && j < Nx) ? u_old[(i + 1) * Nx + j] : 0.f;
    // (left/right halos: tx==0 and tx==15 cases — omitted for brevity)
    __syncthreads();

    if (i <= 0 || i >= Ny - 1 || j <= 0 || j >= Nx - 1) return;
    float laplacian = s[ty][tx+1] + s[ty+2][tx+1] +
                      s[ty+1][tx] + s[ty+1][tx+2] - 4.f * s[ty+1][tx+1];
    u_new[i * Nx + j] = s[ty+1][tx+1] + r * laplacian;
}
```

---

## 5. Boundary Conditions

### Dirichlet (Fixed Value)

Boundary values are fixed constants (e.g., temperature of a heated wall):

```c
// Apply Dirichlet BC: set boundary to constant values
__global__ void apply_dirichlet(float *u, int Nx, int Ny,
                                 float top, float bottom,
                                 float left, float right) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Top and bottom rows
    if (idx < Nx) {
        u[0 * Nx + idx]        = top;     // row 0
        u[(Ny-1) * Nx + idx]   = bottom;  // row Ny-1
    }
    // Left and right columns
    if (idx < Ny) {
        u[idx * Nx + 0]        = left;    // col 0
        u[idx * Nx + (Nx - 1)] = right;   // col Nx-1
    }
}
```

### Neumann (Fixed Flux / Insulating)

Zero-flux (insulating) boundary: ∂u/∂n = 0. Implemented with ghost cells (mirror):

```c
// Neumann BC on top row: u[-1][j] = u[1][j] (mirror)
// Apply after each update step
__global__ void apply_neumann_top(float *u, int Nx) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < Nx) u[0 * Nx + j] = u[1 * Nx + j];   // ghost = mirror
}

__global__ void apply_neumann_bottom(float *u, int Nx, int Ny) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < Nx) u[(Ny-1) * Nx + j] = u[(Ny-2) * Nx + j];
}
```

---

## 6. Convergence Measurement

For steady-state problems, check the L2-norm of the change per step:

```c
// L2 residual: ||u_new - u_old||_2
__global__ void l2_residual(
    const float *u_new, const float *u_old,
    float *partial_sq, int N)
{
    extern __shared__ float sdata[];
    int i   = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float diff = (i < N) ? (u_new[i] - u_old[i]) : 0.f;
    sdata[tid] = diff * diff;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    if (tid == 0) partial_sq[blockIdx.x] = sdata[0];
}

// Full convergence loop
void run_to_convergence(float *d_u0, float *d_u1, int Nx, int Ny,
                        float r, float tol) {
    dim3 block(16, 16);
    dim3 grid((Nx + 15) / 16, (Ny + 15) / 16);
    int N = Nx * Ny;
    const int BLOCK = 256;
    int n_blocks = (N + BLOCK - 1) / BLOCK;

    float *d_partial; cudaMalloc(&d_partial, n_blocks * sizeof(float));

    for (int step = 0; step < 100000; step++) {
        heat_eq_step<<<grid, block>>>(d_u0, d_u1, Nx, Ny, r);

        if (step % 100 == 0) {
            l2_residual<<<n_blocks, BLOCK, BLOCK * sizeof(float)>>>(
                d_u1, d_u0, d_partial, N);
            // reduce d_partial (use CUB) and take sqrt
            float res = sqrtf(cub_reduce_sum(d_partial, n_blocks));
            printf("Step %d: L2 residual = %.2e\n", step, res);
            if (res < tol) { printf("Converged!\n"); break; }
        }

        float *tmp = d_u0; d_u0 = d_u1; d_u1 = tmp;  // ping-pong
    }
    cudaFree(d_partial);
}
```

---

## 7. Complete Simulation Example

```c
int main() {
    const int Nx = 512, Ny = 512;
    const float alpha = 0.1f;
    const float h = 1.0f / (Nx - 1);           // grid spacing
    const float dt = 0.24f * h * h / alpha;    // r = 0.24 < 0.25 (safe)
    const float r  = alpha * dt / (h * h);

    check_cfl(alpha, dt, h);

    size_t bytes = Nx * Ny * sizeof(float);
    float *d_u0, *d_u1;
    cudaMalloc(&d_u0, bytes);
    cudaMalloc(&d_u1, bytes);
    cudaMemset(d_u0, 0, bytes);  // initial temperature = 0

    // Dirichlet BC: top wall at T=1, all others at T=0
    apply_dirichlet<<<(max(Nx,Ny)+255)/256, 256>>>(d_u0, Nx, Ny, 1.f, 0.f, 0.f, 0.f);
    cudaMemcpy(d_u1, d_u0, bytes, cudaMemcpyDeviceToDevice);

    dim3 block(16, 16), grid((Nx+15)/16, (Ny+15)/16);

    const int STEPS = 10000;
    for (int t = 0; t < STEPS; t++) {
        heat_eq_step<<<grid, block>>>(d_u0, d_u1, Nx, Ny, r);
        // Reapply Dirichlet BC (stencil kernel may overwrite boundaries)
        apply_dirichlet<<<(max(Nx,Ny)+255)/256, 256>>>(d_u1, Nx, Ny, 1.f, 0.f, 0.f, 0.f);
        float *tmp = d_u0; d_u0 = d_u1; d_u1 = tmp;
    }

    // Download and save
    std::vector<float> h_u(Nx * Ny);
    cudaMemcpy(h_u.data(), d_u0, bytes, cudaMemcpyDeviceToHost);
    save_pgm("heat.pgm", h_u.data(), Nx, Ny);

    cudaFree(d_u0); cudaFree(d_u1);
}
```

**Expected result**: after 10,000 steps the temperature field shows a gradient from u=1 (top) to u=0 (bottom), converging to the steady-state linear profile u(y) = 1 - y (for unit-square domain).

---

## Key Takeaways

- The 2D heat equation ∂u/∂t = α∇²u is discretized as a 5-point stencil update: `u_new = u_old + r * (neighbors - 4*center)`
- **CFL condition**: `r = α·Δt/h² ≤ 0.25` — violation causes exponential growth (numerical instability)
- **Ping-pong buffers**: alternate between `u_old` and `u_new` to avoid read-write conflicts in the stencil
- **Dirichlet BC**: fix boundary values after each step; **Neumann BC**: copy interior row/column to boundary (ghost cell approach)
- Convergence to steady state measured by L2-norm of `u_new - u_old`; terminate when below a tolerance
- Explicit schemes are simple to code but constrained by CFL; implicit schemes (Crank-Nicolson) allow larger Δt but require solving a linear system per step

---

**Next**: [24. Fluid Dynamics LBM](./24_Fluid_Dynamics_LBM.md) — Simulate incompressible flow with the Lattice Boltzmann Method on the D2Q9 grid, including streaming, collision, and bounce-back boundary conditions.
