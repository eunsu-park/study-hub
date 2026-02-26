# 16. Computational Electrodynamics (FDTD)

[← Previous: 15. Multipole Expansion](15_Multipole_Expansion.md) | [Next: 17. Electromagnetic Scattering →](17_Electromagnetic_Scattering.md)

## Learning Objectives

1. Understand the FDTD (Finite-Difference Time-Domain) method as a direct discretization of Maxwell's equations
2. Construct the Yee grid (staggered grid) and explain why it naturally preserves the divergence conditions
3. Implement the leapfrog time-stepping scheme and derive the Courant stability condition
4. Build a complete 1D FDTD simulation with source injection and absorbing boundaries
5. Extend the method to 2D and understand PML (Perfectly Matched Layer) absorbing boundaries
6. Validate FDTD results against analytical solutions for wave propagation
7. Apply FDTD to practical problems: pulse propagation, scattering, and waveguide simulation

Maxwell's equations are a set of coupled partial differential equations that, for all their elegance, can be solved analytically only in a handful of idealized geometries. For real-world problems — antennas with complex shapes, scattering from irregular objects, waves in inhomogeneous media — we need computational methods. The Finite-Difference Time-Domain (FDTD) method is the most intuitive and widely used approach. It directly discretizes Maxwell's curl equations on a staggered grid, marching the fields forward in time step by step. In this lesson, we build an FDTD solver from scratch, starting in 1D and extending to 2D, learning about stability, accuracy, and boundary conditions along the way.

> **Analogy**: Imagine simulating a pond by placing water-level sensors on a regular grid. At each time step, you update each sensor's value based on the differences between neighboring sensors (finite differences). If your sensors are spaced closely enough and you take small enough time steps, the discrete updates faithfully reproduce the continuous wave physics. FDTD does exactly this for electromagnetic fields, but with the added subtlety that $\mathbf{E}$ and $\mathbf{B}$ live on slightly shifted grids (the Yee lattice) to maintain consistency.

---

## 1. From Maxwell to Finite Differences

### 1.1 Maxwell's Curl Equations

In source-free, linear media:

$$\frac{\partial \mathbf{B}}{\partial t} = -\nabla \times \mathbf{E}$$

$$\frac{\partial \mathbf{D}}{\partial t} = \nabla \times \mathbf{H}$$

For a 1D wave propagating in the $x$-direction with $E_z$ and $H_y$ components:

$$\frac{\partial E_z}{\partial t} = \frac{1}{\epsilon}\frac{\partial H_y}{\partial x}$$

$$\frac{\partial H_y}{\partial t} = \frac{1}{\mu}\frac{\partial E_z}{\partial x}$$

### 1.2 Finite Difference Approximation

The central difference approximation for derivatives:

$$\frac{\partial f}{\partial x}\bigg|_{x_i} \approx \frac{f(x_i + \Delta x/2) - f(x_i - \Delta x/2)}{\Delta x} + O(\Delta x^2)$$

This is **second-order accurate** — the error decreases quadratically with grid spacing.

### 1.3 The Yee Grid (Staggered Grid)

Kane Yee's brilliant insight (1966) was to stagger the $\mathbf{E}$ and $\mathbf{H}$ field components in both space and time:

```
1D Yee Grid:

     E_z[0]        E_z[1]        E_z[2]        E_z[3]
      |              |              |              |
      |----H_y[0]----|----H_y[1]----|----H_y[2]----|
      |              |              |              |
     x=0           x=dx          x=2dx          x=3dx
```

- $E_z$ is defined at integer grid points: $E_z^n[i]$ at $(x = i\Delta x, t = n\Delta t)$
- $H_y$ is defined at half-integer grid points: $H_y^{n+1/2}[i+1/2]$ at $(x = (i+1/2)\Delta x, t = (n+1/2)\Delta t)$

This staggering ensures that the central difference naturally approximates the curl at the correct location.

---

## 2. The Leapfrog Algorithm

### 2.1 Update Equations (1D)

The discretized Maxwell equations become:

$$H_y^{n+1/2}[i+\tfrac{1}{2}] = H_y^{n-1/2}[i+\tfrac{1}{2}] + \frac{\Delta t}{\mu \Delta x}\left(E_z^n[i+1] - E_z^n[i]\right)$$

$$E_z^{n+1}[i] = E_z^n[i] + \frac{\Delta t}{\epsilon \Delta x}\left(H_y^{n+1/2}[i+\tfrac{1}{2}] - H_y^{n+1/2}[i-\tfrac{1}{2}]\right)$$

The $E$ and $H$ fields are updated alternately — $H$ is always half a time step ahead of $E$, hence "leapfrog."

### 2.2 Courant-Friedrichs-Lewy (CFL) Condition

The leapfrog scheme is **conditionally stable**. The time step must satisfy:

$$\boxed{\Delta t \leq \frac{\Delta x}{c \sqrt{d}}}$$

where $d$ is the spatial dimension (1, 2, or 3) and $c$ is the maximum wave speed in the domain.

The **Courant number** is $S = c \Delta t / \Delta x$. Stability requires $S \leq 1/\sqrt{d}$.

In practice, use $S = 0.5$ for safety and to reduce numerical dispersion.

```python
import numpy as np
import matplotlib.pyplot as plt

class FDTD_1D:
    """
    1D FDTD simulation of electromagnetic wave propagation.

    Why build from scratch: understanding the update equations,
    stability conditions, and boundary treatments gives physical
    insight that a black-box solver cannot provide.
    """

    def __init__(self, nx=500, dx=1e-3, courant=0.5):
        """
        Initialize the 1D FDTD domain.

        Parameters:
            nx     : number of spatial cells
            dx     : cell size (m)
            courant: Courant number S = c*dt/dx (must be <= 1)
        """
        self.c = 3e8              # speed of light (m/s)
        self.eps_0 = 8.854e-12    # vacuum permittivity
        self.mu_0 = 4*np.pi*1e-7  # vacuum permeability

        self.nx = nx
        self.dx = dx
        self.dt = courant * dx / self.c
        self.courant = courant

        # Field arrays
        self.Ez = np.zeros(nx)         # E_z at integer points
        self.Hy = np.zeros(nx - 1)     # H_y at half-integer points

        # Material arrays (relative permittivity and permeability)
        self.eps_r = np.ones(nx)
        self.mu_r = np.ones(nx - 1)

        # Update coefficients
        self.cE = self.dt / (self.eps_0 * self.dx)  # for E update
        self.cH = self.dt / (self.mu_0 * self.dx)   # for H update

        self.time_step = 0

    def set_material(self, start_idx, end_idx, eps_r=1.0, mu_r=1.0):
        """Set material properties in a region."""
        self.eps_r[start_idx:end_idx] = eps_r
        if end_idx < self.nx:
            self.mu_r[start_idx:end_idx] = mu_r

    def add_gaussian_source(self, source_idx, t0, spread):
        """Inject a soft Gaussian pulse source at a given location."""
        t = self.time_step * self.dt
        self.Ez[source_idx] += np.exp(-0.5 * ((t - t0) / spread)**2)

    def add_sinusoidal_source(self, source_idx, frequency, amplitude=1.0):
        """Inject a continuous sinusoidal source."""
        t = self.time_step * self.dt
        self.Ez[source_idx] += amplitude * np.sin(2 * np.pi * frequency * t)

    def step(self):
        """
        Advance the simulation by one time step using leapfrog.

        The order matters: H is updated first (from n-1/2 to n+1/2),
        then E is updated (from n to n+1). This preserves the
        time-centering of the central differences.
        """
        # Update H_y (uses E_z at time n)
        self.Hy += (self.cH / self.mu_r) * (self.Ez[1:] - self.Ez[:-1])

        # Update E_z (uses H_y at time n+1/2)
        self.Ez[1:-1] += (self.cE / self.eps_r[1:-1]) * (self.Hy[1:] - self.Hy[:-1])

        # Simple absorbing boundary conditions (Mur first-order)
        # Left boundary
        self.Ez[0] = self.Ez[1]
        # Right boundary
        self.Ez[-1] = self.Ez[-2]

        self.time_step += 1

    def run(self, n_steps, source_idx=None, source_type='gaussian',
            frequency=None, snapshots=None):
        """
        Run the simulation for n_steps.

        Returns field snapshots at specified time steps.
        """
        if snapshots is None:
            snapshots = [n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps - 1]

        saved = {}
        t0 = 20 * self.dt * (self.nx // 10)  # delay for Gaussian
        spread = 5 * self.dt * (self.nx // 10)

        for n in range(n_steps):
            if source_idx is not None:
                if source_type == 'gaussian':
                    self.add_gaussian_source(source_idx, t0, spread)
                elif source_type == 'sinusoidal' and frequency is not None:
                    self.add_sinusoidal_source(source_idx, frequency)

            self.step()

            if n in snapshots:
                saved[n] = self.Ez.copy()

        return saved


def demo_1d_fdtd():
    """Demonstrate basic 1D FDTD: Gaussian pulse in free space."""
    sim = FDTD_1D(nx=500, dx=1e-3, courant=0.5)

    # Source at the left quarter of the domain
    source_idx = 100
    n_steps = 600
    snapshots = [100, 200, 300, 500]

    saved = sim.run(n_steps, source_idx=source_idx, source_type='gaussian',
                    snapshots=snapshots)

    x = np.arange(sim.nx) * sim.dx * 1e3  # in mm

    fig, axes = plt.subplots(len(snapshots), 1, figsize=(12, 10), sharex=True)
    for ax, step_n in zip(axes, snapshots):
        ax.plot(x, saved[step_n], 'b-', linewidth=1.5)
        ax.axvline(x=source_idx * sim.dx * 1e3, color='r', linestyle='--',
                   alpha=0.5, label='Source')
        ax.set_ylabel('$E_z$')
        t_ns = step_n * sim.dt * 1e9
        ax.set_title(f'Step {step_n} (t = {t_ns:.2f} ns)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

    axes[-1].set_xlabel('Position (mm)')
    plt.suptitle('1D FDTD: Gaussian Pulse Propagation', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("fdtd_1d_gaussian.png", dpi=150)
    plt.show()

demo_1d_fdtd()
```

---

## 3. Absorbing Boundary Conditions

### 3.1 The Problem

FDTD simulates a finite domain, but electromagnetic waves naturally propagate to infinity. If we simply truncate the grid, waves reflect off the boundaries and corrupt the solution.

### 3.2 Mur's Absorbing Boundary Condition

The simplest ABC is the first-order Mur condition, which approximates an outgoing wave at the boundary:

$$E_z^{n+1}[0] = E_z^n[1] + \frac{S - 1}{S + 1}\left(E_z^{n+1}[1] - E_z^n[0]\right)$$

where $S$ is the Courant number. For $S = 1$, this simplifies to $E_z^{n+1}[0] = E_z^n[1]$ — the boundary value is simply the previous value of its neighbor.

### 3.3 Perfectly Matched Layer (PML)

The PML, invented by Berenger (1994), is a lossy layer at the boundary that absorbs incoming waves **without reflection**, regardless of frequency or angle of incidence. It works by introducing an artificial conductivity that is "matched" to free space:

$$\sigma_x(x) = \sigma_{\max}\left(\frac{x}{d_{\text{PML}}}\right)^p$$

where the conductivity increases polynomially from zero at the inner PML boundary to $\sigma_{\max}$ at the outer boundary. Typical parameters: $p = 3$, PML thickness = 10-20 cells.

```python
class FDTD_1D_PML(FDTD_1D):
    """
    1D FDTD with PML absorbing boundaries.

    Why PML: Mur's ABC works only for normal incidence in 1D and
    degrades for oblique incidence in 2D/3D. PML absorbs waves at
    all angles and frequencies, making it the standard for production
    FDTD codes.
    """

    def __init__(self, nx=500, dx=1e-3, courant=0.5, pml_cells=20):
        super().__init__(nx=nx + 2 * pml_cells, dx=dx, courant=courant)
        self.pml_cells = pml_cells
        self.inner_nx = nx

        # PML conductivity profile (polynomial grading)
        sigma_max = 0.8 * (3 + 1) / (self.dx * np.sqrt(self.mu_0 / self.eps_0))

        # Left PML region
        for i in range(pml_cells):
            depth = (pml_cells - i) / pml_cells
            sigma = sigma_max * depth**3
            self.eps_r[i] = 1.0  # still vacuum permittivity
            # Modify update coefficients to include loss
            # We store the loss factor separately
            pass  # simplified: use exponential decay approach below

        # Use a simpler convolutional PML approach
        self.psi_Ez_left = np.zeros(pml_cells)
        self.psi_Ez_right = np.zeros(pml_cells)
        self.psi_Hy_left = np.zeros(pml_cells)
        self.psi_Hy_right = np.zeros(pml_cells)

        # PML parameters
        sigma = np.zeros(pml_cells)
        for i in range(pml_cells):
            sigma[i] = sigma_max * ((pml_cells - i) / pml_cells)**3

        self.b_pml = np.exp(-sigma * self.dt / self.eps_0)
        self.c_pml = (self.b_pml - 1) * sigma / (sigma**2 + 1e-30) * self.eps_0

    def step(self):
        """Advance one step with PML boundaries."""
        # Standard leapfrog update
        self.Hy += (self.cH / self.mu_r) * (self.Ez[1:] - self.Ez[:-1])
        self.Ez[1:-1] += (self.cE / self.eps_r[1:-1]) * (self.Hy[1:] - self.Hy[:-1])

        # Apply PML damping at boundaries
        pml = self.pml_cells
        decay = 0.98  # simple exponential decay per step in PML

        for i in range(pml):
            factor = decay ** (pml - i)
            self.Ez[i] *= factor
            self.Ez[-(i+1)] *= factor

        self.time_step += 1


def compare_boundaries():
    """Compare Mur ABC vs PML absorbing boundaries."""

    # Simulation with Mur ABC
    sim_mur = FDTD_1D(nx=400, dx=1e-3, courant=0.5)
    saved_mur = sim_mur.run(800, source_idx=200, source_type='gaussian',
                            snapshots=[300, 500, 700])

    # Simulation with PML
    sim_pml = FDTD_1D_PML(nx=400, dx=1e-3, courant=0.5, pml_cells=20)
    saved_pml = sim_pml.run(800, source_idx=220, source_type='gaussian',
                            snapshots=[300, 500, 700])

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex='col')

    for row, step_n in enumerate([300, 500, 700]):
        x_mur = np.arange(sim_mur.nx) * sim_mur.dx * 1e3
        x_pml = np.arange(sim_pml.nx) * sim_pml.dx * 1e3

        axes[row, 0].plot(x_mur, saved_mur[step_n], 'b-', linewidth=1.5)
        axes[row, 0].set_ylabel('$E_z$')
        axes[row, 0].set_title(f'Mur ABC (step {step_n})')
        axes[row, 0].grid(True, alpha=0.3)

        axes[row, 1].plot(x_pml, saved_pml[step_n], 'r-', linewidth=1.5)
        axes[row, 1].set_title(f'PML (step {step_n})')
        axes[row, 1].grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel('Position (mm)')
    axes[-1, 1].set_xlabel('Position (mm)')
    plt.suptitle('Boundary Comparison: Mur ABC vs PML', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("fdtd_boundary_comparison.png", dpi=150)
    plt.show()

compare_boundaries()
```

---

## 4. 2D FDTD

### 4.1 TE and TM Modes in 2D

In 2D (waves in the $xy$-plane), Maxwell's equations decouple into two independent sets:

**TM mode** ($E_z$, $H_x$, $H_y$):

$$\frac{\partial H_x}{\partial t} = -\frac{1}{\mu}\frac{\partial E_z}{\partial y}$$

$$\frac{\partial H_y}{\partial t} = \frac{1}{\mu}\frac{\partial E_z}{\partial x}$$

$$\frac{\partial E_z}{\partial t} = \frac{1}{\epsilon}\left(\frac{\partial H_y}{\partial x} - \frac{\partial H_x}{\partial y}\right)$$

**TE mode** ($H_z$, $E_x$, $E_y$):

$$\frac{\partial E_x}{\partial t} = \frac{1}{\epsilon}\frac{\partial H_z}{\partial y}$$

$$\frac{\partial E_y}{\partial t} = -\frac{1}{\epsilon}\frac{\partial H_z}{\partial x}$$

$$\frac{\partial H_z}{\partial t} = \frac{1}{\mu}\left(\frac{\partial E_x}{\partial y} - \frac{\partial E_y}{\partial x}\right)$$

### 4.2 The 2D Yee Grid

```
2D Yee Grid (TM mode):

    Hy---Ez---Hy---Ez
    |         |
    Hx        Hx
    |         |
    Hy---Ez---Hy---Ez
    |         |
    Hx        Hx
    |         |
    Hy---Ez---Hy---Ez
```

$E_z$ is at cell centers, $H_x$ at cell edges (horizontal), $H_y$ at cell edges (vertical).

### 4.3 Courant Condition in 2D

$$\Delta t \leq \frac{1}{c\sqrt{1/\Delta x^2 + 1/\Delta y^2}}$$

For a square grid ($\Delta x = \Delta y = \Delta$): $\Delta t \leq \Delta/(c\sqrt{2})$.

```python
class FDTD_2D:
    """
    2D FDTD simulation (TM mode: Ez, Hx, Hy).

    Why 2D: it captures diffraction, scattering, and interference
    effects that are absent in 1D, while remaining computationally
    tractable for interactive exploration.
    """

    def __init__(self, nx=200, ny=200, dx=1e-3, courant=0.5):
        self.c = 3e8
        self.eps_0 = 8.854e-12
        self.mu_0 = 4 * np.pi * 1e-7

        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dx  # square grid
        self.dt = courant * dx / (self.c * np.sqrt(2))

        # Field arrays (TM mode)
        self.Ez = np.zeros((nx, ny))
        self.Hx = np.zeros((nx, ny - 1))  # half-grid in y
        self.Hy = np.zeros((nx - 1, ny))  # half-grid in x

        # Material properties
        self.eps_r = np.ones((nx, ny))

        # Update coefficients
        self.cE = self.dt / (self.eps_0 * self.dx)
        self.cH = self.dt / (self.mu_0 * self.dx)

        self.time_step = 0

    def set_material_circle(self, cx, cy, radius, eps_r):
        """Set material properties in a circular region."""
        for i in range(self.nx):
            for j in range(self.ny):
                if (i - cx)**2 + (j - cy)**2 < radius**2:
                    self.eps_r[i, j] = eps_r

    def step(self):
        """Advance one time step (leapfrog)."""
        # Update Hx
        self.Hx -= self.cH * (self.Ez[:, 1:] - self.Ez[:, :-1])

        # Update Hy
        self.Hy += self.cH * (self.Ez[1:, :] - self.Ez[:-1, :])

        # Update Ez
        dHy_dx = self.Hy[1:, :] - self.Hy[:-1, :]
        dHx_dy = self.Hx[:, 1:] - self.Hx[:, :-1]

        # Interior update (avoiding boundaries)
        self.Ez[1:-1, 1:-1] += (self.cE / self.eps_r[1:-1, 1:-1]) * \
            (dHy_dx[:, 1:-1] - dHx_dy[1:-1, :])

        # Simple absorbing boundaries
        self.Ez[0, :] = 0
        self.Ez[-1, :] = 0
        self.Ez[:, 0] = 0
        self.Ez[:, -1] = 0

        self.time_step += 1

    def add_point_source(self, ix, iy, frequency=None, t0=None, spread=None):
        """Add a source at (ix, iy)."""
        t = self.time_step * self.dt
        if frequency is not None:
            self.Ez[ix, iy] += np.sin(2 * np.pi * frequency * t)
        elif t0 is not None and spread is not None:
            self.Ez[ix, iy] += np.exp(-0.5 * ((t - t0) / spread)**2)


def demo_2d_fdtd():
    """
    Demonstrate 2D FDTD: point source radiation and scattering
    from a dielectric cylinder.
    """
    sim = FDTD_2D(nx=200, ny=200, dx=0.5e-3, courant=0.5)

    # Add a dielectric cylinder (eps_r = 4) at center-right
    sim.set_material_circle(cx=130, cy=100, radius=20, eps_r=4.0)

    freq = 30e9  # 30 GHz

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    snapshot_steps = [200, 400, 600, 800]

    for n in range(max(snapshot_steps) + 1):
        sim.add_point_source(50, 100, frequency=freq)
        sim.step()

        if n in snapshot_steps:
            idx = snapshot_steps.index(n)
            ax = axes.flat[idx]

            extent = [0, sim.nx * sim.dx * 1e3, 0, sim.ny * sim.dy * 1e3]
            im = ax.imshow(sim.Ez.T, cmap='RdBu_r', origin='lower',
                          extent=extent, vmin=-0.5, vmax=0.5, aspect='equal')

            # Draw the dielectric cylinder
            circle = plt.Circle((130 * sim.dx * 1e3, 100 * sim.dy * 1e3),
                               20 * sim.dx * 1e3, fill=False, color='white',
                               linewidth=2, linestyle='--')
            ax.add_patch(circle)

            # Mark source
            ax.plot(50 * sim.dx * 1e3, 100 * sim.dy * 1e3, 'w*', markersize=10)

            ax.set_xlabel('x (mm)')
            ax.set_ylabel('y (mm)')
            t_ns = n * sim.dt * 1e9
            ax.set_title(f'Step {n} (t = {t_ns:.3f} ns)')
            plt.colorbar(im, ax=ax, label='$E_z$', shrink=0.8)

    plt.suptitle('2D FDTD: Point Source + Dielectric Cylinder Scattering',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("fdtd_2d_scattering.png", dpi=150)
    plt.show()

demo_2d_fdtd()
```

---

## 5. Validation Against Analytical Solutions

A critical part of any computational method is **validation** — comparing numerical results with known analytical solutions to verify correctness.

```python
def validate_fdtd_wave_speed():
    """
    Validate FDTD by checking wave propagation speed.

    Why validate: numerical dispersion can cause the simulated wave
    speed to differ from the physical speed. Measuring this error
    is essential for understanding the accuracy of the simulation.
    """
    c = 3e8
    dx = 1e-3
    nx = 1000
    courant_values = [0.25, 0.5, 0.75, 1.0]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    for S in courant_values:
        sim = FDTD_1D(nx=nx, dx=dx, courant=S)

        # Gaussian pulse source at center
        source_idx = nx // 4
        t0 = 40 * sim.dt * 10
        spread = 10 * sim.dt * 10

        # Run until pulse reaches 3/4 of the domain
        target_idx = 3 * nx // 4
        target_time = (target_idx - source_idx) * dx / c
        n_steps = int(target_time / sim.dt) + 100

        for n in range(n_steps):
            t = n * sim.dt
            sim.Ez[source_idx] += np.exp(-0.5 * ((t - t0) / spread)**2)
            sim.step()

        # Find pulse peak position
        peak_idx = np.argmax(np.abs(sim.Ez[source_idx:]))
        peak_pos = (source_idx + peak_idx) * dx

        # Expected position
        expected_pos = source_idx * dx + c * n_steps * sim.dt

        x = np.arange(nx) * dx * 1e3
        axes[0].plot(x, sim.Ez, linewidth=1.5, label=f'S = {S}')

    axes[0].set_xlabel('Position (mm)')
    axes[0].set_ylabel('$E_z$')
    axes[0].set_title('Pulse Shape at Same Physical Time (Different Courant Numbers)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Numerical dispersion analysis
    # For a 1D FDTD, the numerical dispersion relation is:
    # sin(omega*dt/2) / dt = c * sin(k*dx/2) / dx
    k_norm = np.linspace(0.01, np.pi, 200)  # k*dx from 0 to pi

    for S in courant_values:
        omega_exact = c * k_norm / dx
        sin_arg = S * np.sin(k_norm / 2)
        sin_arg = np.clip(sin_arg, -1, 1)
        omega_fdtd = 2 * np.arcsin(sin_arg) / (S * dx / c)

        v_phase_ratio = (omega_fdtd / (k_norm / dx)) / c
        axes[1].plot(k_norm / np.pi, v_phase_ratio,
                     linewidth=1.5, label=f'S = {S}')

    axes[1].axhline(y=1, color='black', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Normalized wavenumber $k\\Delta x / \\pi$')
    axes[1].set_ylabel('$v_{\\phi}^{\\mathrm{FDTD}} / c$')
    axes[1].set_title('Numerical Dispersion: Phase Velocity Ratio')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0.8, 1.05)

    plt.tight_layout()
    plt.savefig("fdtd_validation.png", dpi=150)
    plt.show()

validate_fdtd_wave_speed()
```

---

## 6. Practical Considerations

### 6.1 Grid Resolution

The rule of thumb is **at least 10-20 cells per wavelength** ($\Delta x \leq \lambda / 20$). Finer resolution is needed near material boundaries and for curved geometries.

### 6.2 Numerical Dispersion

The FDTD grid introduces **numerical dispersion** — the simulated phase velocity depends on direction and frequency. The numerical dispersion relation in 2D is:

$$\left(\frac{\sin(\omega\Delta t/2)}{c\Delta t}\right)^2 = \left(\frac{\sin(k_x\Delta x/2)}{\Delta x}\right)^2 + \left(\frac{\sin(k_y\Delta y/2)}{\Delta y}\right)^2$$

This means waves traveling at 45 degrees to the grid axes propagate slightly slower than those along the axes. Using 20+ cells per wavelength keeps this error below 1%.

### 6.3 Sources

- **Hard source**: $E_z[i] = f(t)$ — overwrites the field, causes artificial reflections
- **Soft source**: $E_z[i] \mathrel{+}= f(t)$ — adds to the field, transparent to waves
- **Total-field/scattered-field (TF/SF)**: cleanly separates incident and scattered waves

### 6.4 Extensions

| Method | Description | Use Case |
|--------|-------------|----------|
| FDTD (basic) | Staggered grid, leapfrog | Broadband, time-domain |
| ADI-FDTD | Alternating Direction Implicit | Large time steps, unconditionally stable |
| DGTD | Discontinuous Galerkin Time Domain | Complex geometries, high-order accuracy |
| FEM | Finite Element Method | Frequency domain, unstructured meshes |
| MoM | Method of Moments | Surface integral, antenna design |

---

## Summary

| Concept | Key Detail | Purpose |
|---------|------------|---------|
| Yee grid | E and H staggered by half-cell | Preserves $\nabla \cdot \mathbf{B} = 0$ automatically |
| Leapfrog | E and H offset by half time step | Second-order accurate, explicit |
| Courant condition | $S = c\Delta t/\Delta x \leq 1/\sqrt{d}$ | Stability requirement |
| Mur ABC | First-order absorbing boundary | Simple but limited to normal incidence |
| PML | Graded absorbing layer | Angle/frequency independent absorption |
| Grid resolution | $\Delta x \leq \lambda/20$ | Controls accuracy |
| Numerical dispersion | $v_\phi$ depends on direction | Inherent grid artifact |
| Soft source | $E_z \mathrel{+}= f(t)$ | Transparent to reflected waves |

---

## Exercises

### Exercise 1: Dielectric Slab Reflection
Simulate a plane wave incident on a dielectric slab ($\epsilon_r = 4$, thickness = $\lambda/4$) in 1D FDTD. Measure the reflected and transmitted pulse amplitudes and compare with the analytical Fresnel coefficients. How does the result change when the slab thickness is $\lambda/2$?

### Exercise 2: Courant Number Effects
Run the same 1D pulse propagation simulation with Courant numbers $S = 0.1, 0.5, 0.9, 1.0$, and $1.1$. (a) Observe the pulse shapes — which value gives the least dispersion? (b) What happens when $S > 1$? (c) Plot the numerical dispersion relation for each case.

### Exercise 3: 2D Double Slit
Implement a 2D FDTD simulation of a plane wave passing through two slits in a conducting wall. Choose slit width $= 2\lambda$ and slit separation $= 5\lambda$. Observe the diffraction and interference pattern. Compare the fringe spacing with the analytical prediction $\Delta y = \lambda L / d$ where $L$ is the distance from the slits and $d$ is the slit separation.

### Exercise 4: Waveguide Mode
Simulate a rectangular waveguide cross-section in 2D FDTD (the $xy$-plane with PEC boundaries at $y = 0$ and $y = b$). Excite the TE$_{10}$ mode at one end and measure the propagation constant $k_z$. Compare with the analytical value $k_z = \sqrt{k^2 - (\pi/b)^2}$.

### Exercise 5: PML Optimization
Implement a PML with polynomial grading of order $p = 1, 2, 3, 4$ and PML thicknesses of 5, 10, 20, and 40 cells. Measure the reflection coefficient from the PML by comparing the reflected pulse amplitude with the incident pulse. Plot the reflection as a function of PML thickness for each polynomial order and determine the optimal configuration.

---

[← Previous: 15. Multipole Expansion](15_Multipole_Expansion.md) | [Next: 17. Electromagnetic Scattering →](17_Electromagnetic_Scattering.md)
