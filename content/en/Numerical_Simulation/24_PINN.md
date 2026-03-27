[Previous: GPU Acceleration](./23_GPU_Acceleration.md)

---

# 24. Physics-Informed Neural Networks (PINNs)

> **Prerequisites**: Basic understanding of neural networks ([Deep Learning L01-L03](../Deep_Learning/01_PyTorch_Basics.md)) and PDEs (Lessons 6-10 of this topic).

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the PINN framework: embedding physics (PDEs) into neural network loss functions
2. Implement a PINN from scratch for solving the 1D heat equation
3. Design composite loss functions that balance data, PDE residual, and boundary conditions
4. Train PINNs for forward problems (solving PDEs) and inverse problems (discovering parameters)
5. Evaluate PINN limitations and compare with traditional numerical methods

---

## Table of Contents

1. [Motivation: Bridging Physics and ML](#1-motivation-bridging-physics-and-ml)
2. [PINN Architecture](#2-pinn-architecture)
3. [Loss Function Design](#3-loss-function-design)
4. [Implementation: 1D Heat Equation](#4-implementation-1d-heat-equation)
5. [Inverse Problems](#5-inverse-problems)
6. [Advanced Topics](#6-advanced-topics)
7. [Exercises](#7-exercises)

---

## 1. Motivation: Bridging Physics and ML

### 1.1 Traditional PDE Solving vs Neural Networks

```
Traditional numerical methods:          Neural network approach:
┌──────────────────────────┐           ┌──────────────────────────┐
│ • Discretize domain      │           │ • Train NN to approximate│
│ • Assemble system        │           │   the solution function  │
│ • Solve linear system    │           │ • No mesh needed         │
│                          │           │ • Differentiable (AD)    │
│ Pros: Accurate, proven   │           │                          │
│ Cons: Mesh-dependent,    │           │ Pros: Mesh-free, handles │
│       curse of dim.      │           │       high dimensions    │
│                          │           │ Cons: Training cost,     │
│                          │           │       accuracy issues    │
└──────────────────────────┘           └──────────────────────────┘
```

### 1.2 The PINN Idea (Raissi et al., 2019)

A PINN is a neural network trained to satisfy both **data** and **physics**:

```
Standard ML:                           PINN:

  Loss = Σ (NN(x) - y_data)²          Loss = L_data + L_pde + L_bc

  Fits data only.                      L_data: fit observed data
  No physical constraints.             L_pde:  satisfy governing PDE
  May produce unphysical results.      L_bc:   satisfy boundary conditions

  The PDE acts as a regularizer,       Even with sparse data, the
  constraining the solution to         physics constrains the solution
  be physically consistent.            to be physically meaningful.
```

### 1.3 When to Use PINNs

| Scenario | Traditional | PINN | Recommendation |
|----------|------------|------|----------------|
| Well-posed forward PDE | Fast, accurate | Slower, less accurate | Traditional |
| Sparse/noisy data + known physics | Cannot use data | Combines both | PINN |
| Inverse problems (parameter ID) | Requires optimization | Natural framework | PINN |
| High-dimensional PDEs (> 3D) | Curse of dimensionality | Handles well | PINN |
| Complex geometries, no mesh | Meshing is hard | Mesh-free | PINN |

---

## 2. PINN Architecture

### 2.1 Network Structure

```
Input: (x, t)                     Output: u(x, t)
  coordinates                       solution value

  ┌───────┐    ┌─────────┐    ┌─────────┐    ┌───────┐
  │ (x,t) │───►│ Hidden  │───►│ Hidden  │───►│ u(x,t)│
  │       │    │ Layer 1 │    │ Layer 2 │    │       │
  │       │    │ (tanh)  │    │ (tanh)  │    │       │
  └───────┘    └─────────┘    └─────────┘    └───────┘
       │                                          │
       │    Automatic Differentiation              │
       │    ┌──────────────────────────┐          │
       └───►│  ∂u/∂t, ∂u/∂x, ∂²u/∂x² │◄─────────┘
            │  (computed via AD)        │
            └──────────────────────────┘
                       │
                       ▼
              PDE Residual: f = ∂u/∂t - α∂²u/∂x²
              Goal: f ≈ 0 everywhere
```

### 2.2 Key Insight: Automatic Differentiation

The neural network u(x,t;θ) is differentiable with respect to inputs (x,t). This means we can compute ∂u/∂t, ∂u/∂x, ∂²u/∂x² exactly using automatic differentiation — no finite differences needed.

```python
import numpy as np


class SimpleNN:
    """Minimal fully-connected neural network for PINN.

    Architecture: input(2) → hidden(n) → hidden(n) → output(1)
    Activation: tanh (smooth, well-behaved derivatives)
    """

    def __init__(self, hidden_size=32, seed=42):
        rng = np.random.RandomState(seed)
        # Xavier initialization
        s1 = np.sqrt(2.0 / 2)
        s2 = np.sqrt(2.0 / hidden_size)

        self.W1 = rng.randn(2, hidden_size) * s1
        self.b1 = np.zeros(hidden_size)
        self.W2 = rng.randn(hidden_size, hidden_size) * s2
        self.b2 = np.zeros(hidden_size)
        self.W3 = rng.randn(hidden_size, 1) * s2
        self.b3 = np.zeros(1)

    def forward(self, x_t):
        """Forward pass: (x, t) → u(x, t)."""
        self.z1 = x_t @ self.W1 + self.b1
        self.h1 = np.tanh(self.z1)
        self.z2 = self.h1 @ self.W2 + self.b2
        self.h2 = np.tanh(self.z2)
        self.u = self.h2 @ self.W3 + self.b3
        return self.u

    def get_params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
```

---

## 3. Loss Function Design

### 3.1 Composite Loss

The PINN loss has three components:

```
L_total = λ_data · L_data + λ_pde · L_pde + λ_bc · L_bc

L_data = (1/N_d) Σ |u_NN(x_i, t_i) - u_observed_i|²
  → Fit observed data points (if available)

L_pde = (1/N_r) Σ |f(x_j, t_j)|²
  → PDE residual should be zero at collocation points
  → f = ∂u/∂t - α·∂²u/∂x² (for heat equation)

L_bc = (1/N_b) Σ |u_NN(x_k, t_k) - u_bc_k|²
  → Satisfy boundary conditions

Collocation points: random or quasi-random points in the domain
where we enforce the PDE (no mesh needed!)
```

### 3.2 Collocation Points

```
Domain: x ∈ [0, 1], t ∈ [0, 1]

  t
  1 ┌──────────────────────────────┐
    │  ·  ·    ·   ·  ·   ·  ·   │ ← Interior collocation
    │    ·  ·  ·   ·  ·     ·    │    points (L_pde)
    │  ·    ·    ·   ·    ·  ·   │
    │    ·  ·  ·   ·    ·    ·   │
    │  ·  ·    ·   ·  ·   ·     │
  0 ├──●──●──●──●──●──●──●──●───┤
    │  ↑                         │
    │  Initial condition (L_bc)  │
    ●                            ●  ← Boundary conditions (L_bc)
    ●                            ●
    ●                            ●
    └──────────────────────────────┘
    0                            1  x

  No grid structure needed — points can be random.
```

### 3.3 Loss Weighting

Balancing loss components is critical. Common strategies:

| Strategy | Description |
|----------|-------------|
| Fixed weights | λ_data=1, λ_pde=1, λ_bc=10 (hand-tuned) |
| Adaptive (GradNorm) | Normalize gradients of each loss component |
| Curriculum | Start with λ_bc large (learn BC first), then increase λ_pde |
| Self-adaptive | Learn λ as trainable parameters |

---

## 4. Implementation: 1D Heat Equation

### 4.1 Problem Setup

```
Heat equation: ∂u/∂t = α · ∂²u/∂x²

Domain: x ∈ [0, 1], t ∈ [0, 1]
IC: u(x, 0) = sin(πx)
BC: u(0, t) = u(1, t) = 0

Analytical solution: u(x, t) = sin(πx) · exp(-α·π²·t)
```

### 4.2 PINN Implementation

```python
class HeatPINN:
    """PINN for 1D heat equation using finite differences for derivatives.

    Uses numerical differentiation (since we're implementing in pure NumPy
    without autograd). In practice, use PyTorch or JAX for automatic
    differentiation.

    ∂u/∂t = α · ∂²u/∂x²
    """

    def __init__(self, nn, alpha=0.01, n_colloc=1000, n_bc=100,
                 n_ic=100):
        self.nn = nn
        self.alpha = alpha

        rng = np.random.RandomState(0)

        # Interior collocation points
        self.x_colloc = rng.uniform(0.01, 0.99, (n_colloc, 1))
        self.t_colloc = rng.uniform(0.01, 0.99, (n_colloc, 1))

        # Initial condition points (t=0)
        self.x_ic = rng.uniform(0, 1, (n_ic, 1))
        self.t_ic = np.zeros((n_ic, 1))
        self.u_ic = np.sin(np.pi * self.x_ic)

        # Boundary condition points (x=0 and x=1)
        self.t_bc = rng.uniform(0, 1, (n_bc, 1))
        self.x_bc_left = np.zeros((n_bc, 1))
        self.x_bc_right = np.ones((n_bc, 1))
        self.u_bc = np.zeros((n_bc, 1))  # u=0 at boundaries

    def compute_pde_residual(self, eps=1e-4):
        """Compute PDE residual using finite difference approximation.

        f = ∂u/∂t - α · ∂²u/∂x²

        In a real PINN with autograd: derivatives are exact.
        Here we approximate with central differences.
        """
        x = self.x_colloc
        t = self.t_colloc
        xt = np.hstack([x, t])

        u = self.nn.forward(xt)

        # ∂u/∂t via finite difference
        xt_tp = np.hstack([x, t + eps])
        xt_tm = np.hstack([x, t - eps])
        du_dt = (self.nn.forward(xt_tp) - self.nn.forward(xt_tm)) / (2*eps)

        # ∂²u/∂x² via finite difference
        xt_xp = np.hstack([x + eps, t])
        xt_xm = np.hstack([x - eps, t])
        d2u_dx2 = (self.nn.forward(xt_xp) - 2*u
                    + self.nn.forward(xt_xm)) / eps**2

        # PDE residual
        residual = du_dt - self.alpha * d2u_dx2
        return residual

    def compute_loss(self):
        """Composite PINN loss: L_pde + L_ic + L_bc."""
        # PDE residual loss
        residual = self.compute_pde_residual()
        L_pde = np.mean(residual**2)

        # Initial condition loss
        xt_ic = np.hstack([self.x_ic, self.t_ic])
        u_pred_ic = self.nn.forward(xt_ic)
        L_ic = np.mean((u_pred_ic - self.u_ic)**2)

        # Boundary condition loss
        xt_left = np.hstack([self.x_bc_left, self.t_bc])
        xt_right = np.hstack([self.x_bc_right, self.t_bc])
        u_left = self.nn.forward(xt_left)
        u_right = self.nn.forward(xt_right)
        L_bc = np.mean(u_left**2) + np.mean(u_right**2)

        # Weighted sum
        L_total = L_pde + 10.0 * L_ic + 10.0 * L_bc
        return L_total, L_pde, L_ic, L_bc

    def analytical_solution(self, x, t):
        """Analytical solution for comparison."""
        return np.sin(np.pi * x) * np.exp(-self.alpha * np.pi**2 * t)
```

### 4.3 Training with Random Search (Simplified)

```python
def train_pinn_simple(pinn, iterations=5000, lr=0.001):
    """Train PINN using parameter perturbation (simplified).

    In practice, use gradient-based optimization (Adam)
    with automatic differentiation (PyTorch/JAX).
    Here we use random perturbation for demonstration.
    """
    params = pinn.nn.get_params()
    best_loss = float('inf')
    best_params = [p.copy() for p in params]

    for it in range(iterations):
        # Perturb parameters
        for p in params:
            p += np.random.randn(*p.shape) * lr

        loss, l_pde, l_ic, l_bc = pinn.compute_loss()

        if loss < best_loss:
            best_loss = loss
            best_params = [p.copy() for p in params]
        else:
            # Revert
            for p, bp in zip(params, best_params):
                p[:] = bp

        if (it + 1) % 500 == 0:
            print(f"  Iter {it+1}: loss={best_loss:.6f} "
                  f"(pde={l_pde:.6f}, ic={l_ic:.6f}, bc={l_bc:.6f})")

    return best_loss
```

---

## 5. Inverse Problems

### 5.1 The Inverse Problem Framework

Instead of solving the PDE with known parameters, discover unknown parameters from observed data:

```
Forward problem:
  Known: PDE, boundary conditions, parameters (α)
  Unknown: Solution u(x, t)

Inverse problem:
  Known: PDE structure, some observations of u
  Unknown: Parameter α (e.g., thermal diffusivity)

PINN for inverse problem:
  L = L_data + L_pde(α) + L_bc
  α is a trainable parameter (like network weights)
  The optimizer simultaneously learns u and α!
```

### 5.2 Parameter Discovery Example

```python
class InversePINN:
    """PINN for inverse problem: discover thermal diffusivity α.

    Given sparse observations of temperature over time,
    find the thermal diffusivity α of the material.
    """

    def __init__(self, nn, x_data, t_data, u_data):
        self.nn = nn
        self.x_data = x_data
        self.t_data = t_data
        self.u_data = u_data

        # α is unknown — start with an initial guess
        self.log_alpha = np.log(0.05)  # log for positivity

    @property
    def alpha(self):
        return np.exp(self.log_alpha)

    def compute_loss(self):
        """Loss for inverse problem: fit data + satisfy PDE."""
        # Data loss
        xt_data = np.hstack([self.x_data, self.t_data])
        u_pred = self.nn.forward(xt_data)
        L_data = np.mean((u_pred - self.u_data)**2)

        # PDE residual (using current estimate of α)
        # Same as forward problem but α is trainable
        # L_pde = ... (compute with current self.alpha)

        return L_data  # + L_pde in full implementation
```

### 5.3 Applications of Inverse PINNs

| Application | Known | Unknown |
|-------------|-------|---------|
| Material characterization | Temperature measurements | Thermal diffusivity |
| Fluid mechanics | Velocity field (PIV) | Viscosity, pressure field |
| Structural health | Vibration data | Stiffness (damage detection) |
| Epidemiology | Case counts | SIR model parameters |

---

## 6. Advanced Topics

### 6.1 Training Challenges

| Challenge | Cause | Solution |
|-----------|-------|----------|
| Failure to converge | Loss imbalance | Adaptive loss weighting |
| Slow training | Ill-conditioned problem | Learning rate scheduling, Fourier features |
| Sharp gradients / shocks | tanh activation too smooth | Use sin activations (SIREN) |
| High-dimensional PDEs | Curse of dimensionality | Use residual connections, domain decomposition |

### 6.2 DeepXDE Library

```python
# DeepXDE provides a high-level API for PINNs
# (shown for reference — requires installation)

# import deepxde as dde
#
# def pde(x, y):
#     dy_t = dde.grad.jacobian(y, x, i=0, j=1)
#     dy_xx = dde.grad.hessian(y, x, i=0, j=0)
#     return dy_t - 0.01 * dy_xx
#
# geom = dde.geometry.Interval(0, 1)
# timedomain = dde.geometry.TimeDomain(0, 1)
# geomtime = dde.geometry.GeometryXTime(geom, timedomain)
#
# bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, ...)
# ic = dde.icbc.IC(geomtime, lambda x: np.sin(np.pi * x[:, 0:1]), ...)
#
# data = dde.data.TimePDE(geomtime, pde, [bc, ic], num_domain=2540)
# net = dde.nn.FNN([2] + [32]*3 + [1], "tanh", "Glorot uniform")
# model = dde.Model(data, net)
# model.compile("adam", lr=0.001)
# model.train(epochs=10000)
```

### 6.3 PINN vs Traditional Methods

| Aspect | FEM/FDM | PINN |
|--------|---------|------|
| Accuracy | High (well-studied error bounds) | Moderate (depends on training) |
| Speed (small PDE) | Fast | Slow (training overhead) |
| Speed (high-dim PDE) | Curse of dimensionality | Handles well |
| Mesh | Required | Mesh-free |
| Inverse problems | Requires separate optimization | Built into framework |
| Data assimilation | Add-on | Natural |
| Reliability | Proven convergence theory | No convergence guarantees |

---

## 7. Exercises

### Exercise 1: PINN for 1D Heat Equation

Implement a PINN for the heat equation:
1. Domain: x ∈ [0,1], t ∈ [0,1], α = 0.01
2. IC: u(x,0) = sin(πx), BC: u(0,t) = u(1,t) = 0
3. Use 500 interior collocation points, 50 BC points, 50 IC points
4. Train and compare with analytical solution
5. Plot error distribution across the domain

### Exercise 2: Loss Component Analysis

Study the effect of loss weighting:
1. Train with λ_pde = 1, λ_bc = {0.1, 1, 10, 100}
2. For each, plot the final solution and per-component loss curves
3. Find the optimal λ_bc that minimizes total error
4. Explain why too-low λ_bc causes boundary violations and too-high slows PDE convergence

### Exercise 3: Collocation Point Study

Investigate collocation point strategies:
1. Random uniform vs Latin Hypercube Sampling vs regular grid
2. Number of points: 100, 500, 1000, 5000
3. Compare accuracy for each strategy and point count
4. Determine the minimum number of points for 1% relative error

### Exercise 4: Inverse Problem

Discover the diffusivity α from observations:
1. Generate "observed" data from analytical solution with α = 0.02
2. Add 5% Gaussian noise to observations
3. Use only 20 sparse observation points
4. Train inverse PINN to recover α
5. How close is the estimated α to the true value?

### Exercise 5: Burgers' Equation PINN

Solve the viscous Burgers' equation with a PINN:
1. ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x², ν = 0.01/π
2. IC: u(x,0) = -sin(πx), BC: u(-1,t) = u(1,t) = 0
3. This equation develops a sharp gradient (shock-like) — harder for PINNs
4. Compare PINN solution with a reference FD solution
5. Discuss why PINNs struggle with sharp gradients

---

*End of Lesson 24*
