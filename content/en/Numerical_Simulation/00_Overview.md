# Numerical Simulation Overview

## Introduction

This folder contains learning materials for numerical simulation using Python. It covers the full range from basic ordinary differential equations (ODE) to magnetohydrodynamics (MHD) and plasma simulation.

---

## Learning Roadmap

```
Basics (01-02)
    ↓
Ordinary Differential Equations ODE (03-06)
    ↓
Partial Differential Equations PDE Basics (07-08)
    ↓
Heat/Wave/Steady-State Equations (09-12)
    ↓
Computational Fluid Dynamics CFD (13-14)
    ↓
Electromagnetic Simulation (15-16)
    ↓
Magnetohydrodynamics MHD (17-18)
    ↓
Plasma Simulation (19)
    ↓
Monte Carlo Simulation (20)
    ↓
Spectral Methods (21)
    ↓
Finite Element Method (22)
    ↓
GPU Acceleration (23)
    ↓
Physics-Informed Neural Networks PINN (24)
```

---

## File List

| File | Topic | Key Content |
|------|------|----------|
| [01_Numerical_Analysis_Basics.md](./01_Numerical_Analysis_Basics.md) | Numerical Analysis Basics | Floating-point, error analysis, numerical differentiation/integration |
| [02_Linear_Algebra_Review.md](./02_Linear_Algebra_Review.md) | Linear Algebra Review | Matrix operations, eigenvalues, decomposition (LU, QR, SVD) |
| [03_ODE_Basics.md](./03_ODE_Basics.md) | ODE Basics | ODE concepts, initial value problem, analytical solutions |
| [04_ODE_Numerical_Methods.md](./04_ODE_Numerical_Methods.md) | ODE Numerical Methods | Euler, RK2, RK4, adaptive step |
| [05_ODE_Advanced.md](./05_ODE_Advanced.md) | ODE Advanced | Stiff problems, implicit methods, scipy.integrate |
| [06_ODE_Systems.md](./06_ODE_Systems.md) | Coupled ODE and Systems | Lotka-Volterra, pendulum, chaotic systems (Lorenz) |
| [07_PDE_Overview.md](./07_PDE_Overview.md) | PDE Overview | PDE classification, boundary conditions, initial conditions |
| [08_Finite_Difference_Basics.md](./08_Finite_Difference_Basics.md) | Finite Difference Basics | Grid, discretization, stability conditions (CFL) |
| [09_Heat_Equation.md](./09_Heat_Equation.md) | Heat Equation | 1D/2D heat conduction, explicit/implicit methods |
| [10_Wave_Equation.md](./10_Wave_Equation.md) | Wave Equation | 1D/2D waves, boundary reflection, absorbing boundaries |
| [11_Laplace_Poisson.md](./11_Laplace_Poisson.md) | Laplace/Poisson | Steady-state, iterative methods (Jacobi, Gauss-Seidel, SOR) |
| [12_Advection_Equation.md](./12_Advection_Equation.md) | Advection Equation | Upwind, Lax-Wendroff, numerical dispersion/diffusion |
| [13_CFD_Basics.md](./13_CFD_Basics.md) | CFD Basics | Fluid dynamics concepts, Navier-Stokes introduction |
| [14_Incompressible_Flow.md](./14_Incompressible_Flow.md) | Incompressible Flow | Stream function-vorticity, pressure-velocity coupling, SIMPLE |
| [15_Electromagnetics_Numerical.md](./15_Electromagnetics_Numerical.md) | Electromagnetics Numerical | Maxwell equations, FDTD basics |
| [16_FDTD_Implementation.md](./16_FDTD_Implementation.md) | FDTD Implementation | 1D/2D electromagnetic wave simulation, absorbing boundaries (PML) |
| [17_MHD_Basics.md](./17_MHD_Basics.md) | MHD Basic Theory | Magnetohydrodynamics concepts, ideal MHD equations |
| [18_MHD_Numerical_Methods.md](./18_MHD_Numerical_Methods.md) | MHD Numerical Methods | Conservative form, Godunov method, MHD Riemann problem |
| [19_Plasma_Simulation.md](./19_Plasma_Simulation.md) | Plasma Simulation | PIC method basics, particle-mesh interaction |
| [20_Monte_Carlo_Simulation.md](./20_Monte_Carlo_Simulation.md) | Monte Carlo Simulation | Random number generation, MC integration, Ising model, option pricing, variance reduction |
| [21_Spectral_Methods.md](./21_Spectral_Methods.md) | Spectral Methods | Fourier spectral, FFT differentiation, Chebyshev collocation, dealiasing |
| [22_Finite_Element_Method.md](./22_Finite_Element_Method.md) | Finite Element Method | Weak form, basis functions, stiffness matrix assembly, 1D/2D FEM |
| [23_GPU_Acceleration.md](./23_GPU_Acceleration.md) | GPU Acceleration | CUDA basics, CuPy, GPU-accelerated PDE solvers, performance optimization |
| [24_Physics_Informed_Neural_Networks.md](./24_Physics_Informed_Neural_Networks.md) | Physics-Informed Neural Networks (PINN) | Neural network solvers, loss functions from PDEs, automatic differentiation, inverse problems |

---

## Required Libraries

```bash
# Basic
pip install numpy scipy matplotlib

# Performance optimization (optional)
pip install numba

# 3D visualization (optional)
pip install mayavi
```

### Library Roles

| Library | Purpose |
|-----------|------|
| NumPy | Array operations, linear algebra |
| SciPy | ODE solvers, sparse matrices, optimization |
| Matplotlib | 2D visualization, animation |
| Numba | JIT compilation, performance optimization |

---

## Recommended Learning Sequence

### Stage 1: Basics (1-2 weeks)
- 01_Numerical_Analysis_Basics.md
- 02_Linear_Algebra_Review.md

### Stage 2: ODE (2-3 weeks)
- 03_ODE_Basics.md
- 04_ODE_Numerical_Methods.md
- 05_ODE_Advanced.md
- 06_ODE_Systems.md

### Stage 3: PDE Basics (2-3 weeks)
- 07_PDE_Overview.md
- 08_Finite_Difference_Basics.md
- 09_Heat_Equation.md
- 10_Wave_Equation.md

### Stage 4: Steady-State and Advection (1-2 weeks)
- 11_Laplace_Poisson.md
- 12_Advection_Equation.md

### Stage 5: CFD (2-3 weeks)
- 13_CFD_Basics.md
- 14_Incompressible_Flow.md

### Stage 6: Electromagnetics (2 weeks)
- 15_Electromagnetics_Numerical.md
- 16_FDTD_Implementation.md

### Stage 7: MHD and Plasma (3-4 weeks)
- 17_MHD_Basics.md
- 18_MHD_Numerical_Methods.md
- 19_Plasma_Simulation.md

### Stage 8: Stochastic Simulation (2 weeks)
- 20_Monte_Carlo_Simulation.md

### Stage 9: Advanced Methods (2-3 weeks)
- 21_Spectral_Methods.md
- 22_Finite_Element_Method.md

### Stage 10: Modern Computational Techniques (2-3 weeks)
- 23_GPU_Acceleration.md
- 24_Physics_Informed_Neural_Networks.md

---

## Prerequisites

1. **Python Basics**: NumPy array operations
2. **Calculus**: Differentiation, integration, partial derivatives
3. **Linear Algebra**: Matrices, eigenvalues, decomposition
4. **Physics**: Mechanics, basic electromagnetics (for CFD/MHD)

---

## Simulation Code Structure Example

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 1. Parameter setup
nx, ny = 100, 100
dx, dy = 1.0, 1.0
dt = 0.01
n_steps = 1000

# 2. Initial conditions
u = np.zeros((nx, ny))

# 3. Time integration loop
for step in range(n_steps):
    # Apply boundary conditions
    # Calculate spatial derivatives
    # Time advancement
    pass

# 4. Result visualization
plt.imshow(u)
plt.colorbar()
plt.show()
```

---

## References

### Textbooks
- Computational Physics - Mark Newman
- Numerical Recipes - Press et al.
- CFD Python (12 Steps to Navier-Stokes) - Lorena Barba

### Online
- SciPy Official Documentation: https://docs.scipy.org
- Lorena Barba CFD Python: https://github.com/barbagroup/CFDPython
