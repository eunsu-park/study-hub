#!/usr/bin/env python3
"""
FDTD for Maxwell's Equations (1D and 2D)
==========================================

Implements the Finite-Difference Time-Domain (FDTD) method for
electromagnetic wave propagation using the Yee grid.

Key Idea:
    FDTD discretizes Maxwell's curl equations in both space and time.
    E and H fields are staggered by half a cell in space and half a
    time step in time (Yee grid), creating a leapfrog scheme.

Maxwell's Equations (source-free):
    dH/dt = -(1/mu) * curl(E)
    dE/dt =  (1/eps) * curl(H)

Why FDTD?
    - Direct time-domain simulation (no frequency-domain transformation)
    - Naturally handles broadband sources (one simulation = all frequencies)
    - Simple to implement and parallelize
    - The standard method for computational electromagnetics

Key Concepts:
    - Yee grid: staggered E and H field placement
    - CFL stability condition: dt <= dx / (c * sqrt(dim))
    - PML (Perfectly Matched Layer) absorbing boundary
    - Courant number and numerical dispersion

Author: Educational example for Numerical Simulation
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Physical constants
C0 = 3e8          # Speed of light (m/s)
MU0 = 4e-7 * np.pi   # Permeability of free space (H/m)
EPS0 = 1 / (MU0 * C0 ** 2)  # Permittivity of free space (F/m)
ETA0 = np.sqrt(MU0 / EPS0)  # Impedance of free space (~377 Ohm)


class FDTD1D:
    """
    1D FDTD solver for Ex-Hy mode (TEM wave propagation in z-direction).

    The 1D case is z-propagating with:
        dHy/dt = -(1/mu) * dEx/dz
        dEx/dt =  (1/eps) * dHy/dz

    Yee grid layout:
        Ex: |--Ex[0]--|--Ex[1]--|--Ex[2]--|--...
        Hy:     |--Hy[0]--|--Hy[1]--|--Hy[2]--|--...
              (Hy is offset by dx/2)
    """

    def __init__(self, nx: int = 400, dx: float = None):
        self.nx = nx

        # Why: dx sets the spatial resolution. Rule of thumb: at least
        # 20 cells per shortest wavelength to control numerical dispersion.
        self.dx = dx if dx else 1e-3  # 1 mm default

        # CFL condition: dt < dx / c for stability in 1D
        # Why: The Courant number S = c*dt/dx must be <= 1. We use 0.5
        # for safety margin against numerical dispersion.
        self.dt = 0.5 * self.dx / C0

        # Fields
        self.Ex = np.zeros(nx)
        self.Hy = np.zeros(nx)

        # Material properties (relative permittivity and permeability)
        self.eps_r = np.ones(nx)
        self.mu_r = np.ones(nx)

        # PML parameters (absorbing boundary)
        self.sigma_e = np.zeros(nx)
        self.sigma_h = np.zeros(nx)
        self._setup_pml(30)

        # Update coefficients
        # Why: Precomputing coefficients avoids per-step division.
        self._compute_coefficients()

    def _setup_pml(self, n_pml: int):
        """
        Set up PML (Perfectly Matched Layer) absorbing boundary.

        PML is a lossy region at domain boundaries that absorbs outgoing
        waves without reflection. The conductivity increases polynomially
        from zero (no reflection at the PML interface) to a maximum.

        Why PML over simple ABC (Absorbing Boundary Condition)?
            PML absorbs waves at all angles and frequencies with near-zero
            reflection. ABCs only work well for specific incidence angles.
        """
        # Why polynomial grading: A sudden jump in conductivity causes
        # reflections. Polynomial grading (order m=3) smoothly ramps up.
        sigma_max = 0.8 * (3 + 1) / (ETA0 * self.dx)

        for i in range(n_pml):
            # Left PML
            depth = (n_pml - i) / n_pml
            self.sigma_e[i] = sigma_max * depth ** 3
            self.sigma_h[i] = sigma_max * (depth + 0.5 / n_pml) ** 3
            # Right PML
            self.sigma_e[-(i + 1)] = sigma_max * depth ** 3
            self.sigma_h[-(i + 1)] = sigma_max * (depth + 0.5 / n_pml) ** 3

    def _compute_coefficients(self):
        """Precompute FDTD update coefficients including PML loss."""
        # Why: The FDTD update with conductivity is:
        #   E_new = Ca * E_old + Cb * (dH/dz)
        #   H_new = Da * H_old + Db * (dE/dz)
        eps = self.eps_r * EPS0
        mu = self.mu_r * MU0

        self.Ca = (1 - self.sigma_e * self.dt / (2 * eps)) / \
                  (1 + self.sigma_e * self.dt / (2 * eps))
        self.Cb = (self.dt / eps / self.dx) / \
                  (1 + self.sigma_e * self.dt / (2 * eps))
        self.Da = (1 - self.sigma_h * self.dt / (2 * mu)) / \
                  (1 + self.sigma_h * self.dt / (2 * mu))
        self.Db = (self.dt / mu / self.dx) / \
                  (1 + self.sigma_h * self.dt / (2 * mu))

    def add_dielectric_slab(self, start: int, end: int, eps_r: float):
        """Add a dielectric slab (changes permittivity in a region)."""
        self.eps_r[start:end] = eps_r
        self._compute_coefficients()

    def gaussian_source(self, t_step: int, t0: int = 80, spread: float = 30.0):
        """Gaussian pulse source."""
        return np.exp(-((t_step - t0) / spread) ** 2)

    def step(self, t_step: int, source_pos: int = None):
        """
        Advance fields by one time step using leapfrog scheme.

        Leapfrog: H is updated at half-integer times, E at integer times.
        This makes the scheme second-order accurate in time.
        """
        # Update H field (half step ahead of E)
        # Why: Hy[i] sits between Ex[i] and Ex[i+1] on the Yee grid,
        # so dEx/dz at Hy's location = (Ex[i+1] - Ex[i]) / dx.
        self.Hy[:-1] = self.Da[:-1] * self.Hy[:-1] + \
                        self.Db[:-1] * (self.Ex[1:] - self.Ex[:-1])

        # Update E field
        self.Ex[1:] = self.Ca[1:] * self.Ex[1:] + \
                       self.Cb[1:] * (self.Hy[1:] - self.Hy[:-1])

        # Inject source (soft source: add to existing field)
        if source_pos is not None:
            self.Ex[source_pos] += self.gaussian_source(t_step)


class FDTD2D:
    """
    2D FDTD solver for TM mode (Ez, Hx, Hy).

    TM polarization:
        dHx/dt = -(1/mu) * dEz/dy
        dHy/dt =  (1/mu) * dEz/dx
        dEz/dt =  (1/eps) * (dHy/dx - dHx/dy)
    """

    def __init__(self, nx: int = 200, ny: int = 200, dx: float = 1e-3):
        self.nx, self.ny = nx, ny
        self.dx = self.dy = dx

        # CFL for 2D: dt < dx / (c * sqrt(2))
        self.dt = 0.5 * dx / (C0 * np.sqrt(2))

        self.Ez = np.zeros((nx, ny))
        self.Hx = np.zeros((nx, ny))
        self.Hy = np.zeros((nx, ny))

    def step(self, t_step: int, source_pos: tuple = None):
        """Advance 2D TM fields by one time step."""
        dt, dx, dy = self.dt, self.dx, self.dy

        # Update Hx: dHx/dt = -(1/mu0) * dEz/dy
        self.Hx[:, :-1] -= (dt / MU0) * (self.Ez[:, 1:] - self.Ez[:, :-1]) / dy

        # Update Hy: dHy/dt = (1/mu0) * dEz/dx
        self.Hy[:-1, :] += (dt / MU0) * (self.Ez[1:, :] - self.Ez[:-1, :]) / dx

        # Update Ez: dEz/dt = (1/eps0) * (dHy/dx - dHx/dy)
        self.Ez[1:, 1:] += (dt / EPS0) * (
            (self.Hy[1:, 1:] - self.Hy[:-1, 1:]) / dx -
            (self.Hx[1:, 1:] - self.Hx[1:, :-1]) / dy
        )

        # Simple ABC: zero tangential E at boundaries
        self.Ez[0, :] = self.Ez[-1, :] = 0
        self.Ez[:, 0] = self.Ez[:, -1] = 0

        # Inject point source
        if source_pos:
            spread = 30.0
            t0 = 80
            self.Ez[source_pos] += np.exp(-((t_step - t0) / spread) ** 2)


def demo_1d():
    """1D FDTD: pulse through a dielectric slab."""
    print("\n--- 1D FDTD: Gaussian Pulse through Dielectric Slab ---")
    nx = 400
    sim = FDTD1D(nx=nx)

    # Add a glass slab (eps_r = 4, so wave speed halves → wavelength halves)
    slab_start, slab_end = 200, 280
    sim.add_dielectric_slab(slab_start, slab_end, eps_r=4.0)
    source_pos = 100

    # Collect snapshots
    snapshots = []
    n_steps = 500
    for t in range(n_steps):
        sim.step(t, source_pos=source_pos)
        if t % 25 == 0:
            snapshots.append((t, sim.Ex.copy()))

    # Plot selected snapshots
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    times_to_show = [2, 5, 10, 15]
    for ax, idx in zip(axes.flat, times_to_show):
        t, ex = snapshots[idx]
        ax.plot(ex, 'b-', linewidth=0.8)
        ax.axvspan(slab_start, slab_end, alpha=0.2, color='orange',
                   label=f'Dielectric (eps_r=4)')
        ax.axvline(source_pos, color='r', linestyle='--', alpha=0.5,
                   label='Source')
        ax.set_ylim(-1.2, 1.2)
        ax.set_xlabel("Grid cell")
        ax.set_ylabel("Ex")
        ax.set_title(f"t = {t} steps")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("1D FDTD: Pulse Propagation through Dielectric Slab", fontsize=13)
    plt.tight_layout()
    plt.savefig("fdtd_1d.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: fdtd_1d.png")


def demo_2d():
    """2D FDTD: point source radiation."""
    print("\n--- 2D FDTD: Point Source Radiation ---")
    nx, ny = 150, 150
    sim = FDTD2D(nx=nx, ny=ny)
    source = (nx // 2, ny // 2)

    snapshots = []
    n_steps = 200
    for t in range(n_steps):
        sim.step(t, source_pos=source)
        if t % 40 == 0:
            snapshots.append((t, sim.Ez.copy()))

    fig, axes = plt.subplots(1, len(snapshots), figsize=(16, 3.5))
    vmax = max(np.max(np.abs(s[1])) for s in snapshots) * 0.8

    for ax, (t, ez) in zip(axes, snapshots):
        im = ax.imshow(ez.T, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                       origin='lower', aspect='equal')
        ax.plot(source[0], source[1], 'k+', markersize=10)
        ax.set_title(f"t = {t}")
        ax.set_xlabel("x")

    axes[0].set_ylabel("y")
    plt.suptitle("2D FDTD: Ez Field from Point Source", fontsize=13)
    plt.tight_layout()
    plt.savefig("fdtd_2d.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: fdtd_2d.png")


if __name__ == "__main__":
    print("=" * 60)
    print("FDTD for Maxwell's Equations")
    print("=" * 60)

    print(f"\nPhysical constants:")
    print(f"  c  = {C0:.3e} m/s")
    print(f"  mu0 = {MU0:.3e} H/m")
    print(f"  eps0 = {EPS0:.3e} F/m")
    print(f"  eta0 = {ETA0:.1f} Ohm")

    demo_1d()
    demo_2d()

    print("\nDone.")
