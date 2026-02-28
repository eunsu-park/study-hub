#!/usr/bin/env python3
"""
2D Incompressible Navier-Stokes: Lid-Driven Cavity Flow
=========================================================

Solves the 2D incompressible Navier-Stokes equations using the
stream function-vorticity formulation on a lid-driven cavity.

Problem Setup:
    - Square cavity [0,1] x [0,1]
    - Top wall moves rightward at velocity U = 1
    - All other walls are stationary (no-slip)
    - Fluid is incompressible (div(u) = 0)

Stream Function-Vorticity Formulation:
    dw/dt + u * dw/dx + v * dw/dy = (1/Re) * laplacian(w)
    laplacian(psi) = -w
    u = d(psi)/dy,  v = -d(psi)/dx

Why this formulation?
    - Eliminates pressure (which is tricky for incompressible flow)
    - Automatically satisfies continuity (div(u) = 0)
    - Only two scalar fields (psi, w) instead of three (u, v, p)

Key Concepts:
    - Finite difference discretization on a staggered grid
    - Jacobi/SOR iteration for Poisson equation
    - Reynolds number and flow regimes
    - Boundary conditions for vorticity

Author: Educational example for Numerical Simulation
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


class LidDrivenCavity:
    """
    2D lid-driven cavity solver using stream function-vorticity method.

    Attributes:
        N: Grid points per side
        Re: Reynolds number (ratio of inertial to viscous forces)
        dx, dy: Grid spacing
        psi: Stream function
        omega: Vorticity
    """

    def __init__(self, N: int = 41, Re: float = 100.0, U_lid: float = 1.0):
        """
        Initialize the cavity solver.

        Args:
            N: Number of grid points per side
            Re: Reynolds number. Low Re (< 100): laminar.
                High Re (> 1000): vortices, eventually turbulent.
            U_lid: Lid velocity
        """
        self.N = N
        self.Re = Re
        self.U_lid = U_lid

        self.dx = 1.0 / (N - 1)
        self.dy = 1.0 / (N - 1)
        self.x = np.linspace(0, 1, N)
        self.y = np.linspace(0, 1, N)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # Why: Initialize with zeros (fluid at rest). The lid velocity
        # is imposed as a boundary condition, driving the flow.
        self.psi = np.zeros((N, N))    # Stream function
        self.omega = np.zeros((N, N))  # Vorticity

    def compute_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute velocity from stream function using central differences.

        u = d(psi)/dy,  v = -d(psi)/dx

        Why central differences?
            Second-order accurate and sufficient for interior points.
            Boundary velocities are set by no-slip conditions.
        """
        u = np.zeros_like(self.psi)
        v = np.zeros_like(self.psi)

        # Interior points: central differences
        u[1:-1, 1:-1] = (self.psi[2:, 1:-1] - self.psi[:-2, 1:-1]) / (2 * self.dy)
        v[1:-1, 1:-1] = -(self.psi[1:-1, 2:] - self.psi[1:-1, :-2]) / (2 * self.dx)

        # Boundary conditions (no-slip on walls, moving lid on top)
        u[-1, :] = self.U_lid  # Top wall moves right
        return u, v

    def update_boundary_vorticity(self):
        """
        Set vorticity boundary conditions using Thom's formula.

        At a no-slip wall: omega_wall = -2 * (psi_interior - psi_wall) / dx^2

        Why Thom's formula?
            Vorticity at the wall is not directly prescribed. It must be
            derived from the stream function via a Taylor expansion of
            the no-slip condition (u = 0 at wall, u = U at lid).
        """
        dx2 = self.dx ** 2
        dy2 = self.dy ** 2

        # Bottom wall (y = 0, u = 0)
        self.omega[0, :] = -2 * self.psi[1, :] / dy2

        # Top wall (y = 1, u = U_lid)
        # Why: The U_lid term accounts for the moving wall. Without it,
        # the vorticity BC would be wrong and the flow would not develop.
        self.omega[-1, :] = -2 * (self.psi[-2, :] - self.U_lid * self.dy) / dy2

        # Left wall (x = 0, v = 0)
        self.omega[:, 0] = -2 * self.psi[:, 1] / dx2

        # Right wall (x = 1, v = 0)
        self.omega[:, -1] = -2 * self.psi[:, -2] / dx2

    def solve_poisson(self, max_iter: int = 500, tol: float = 1e-5):
        """
        Solve Poisson equation: laplacian(psi) = -omega using SOR.

        SOR (Successive Over-Relaxation) accelerates Jacobi/Gauss-Seidel
        by extrapolating the update: psi_new = (1-w)*psi_old + w*psi_gs

        Why SOR over direct solvers?
            SOR is simple and memory-efficient. For production codes,
            FFT-based or multigrid solvers are much faster.
        """
        dx2 = self.dx ** 2
        dy2 = self.dy ** 2
        # Why: Optimal SOR parameter for a square grid with Dirichlet BCs.
        # This formula from Young (1954) minimizes iterations.
        omega_sor = 2.0 / (1 + np.sin(np.pi * self.dx))

        for iteration in range(max_iter):
            psi_old = self.psi.copy()

            for i in range(1, self.N - 1):
                for j in range(1, self.N - 1):
                    # Gauss-Seidel update for laplacian(psi) = -omega
                    gs = ((self.psi[i + 1, j] + self.psi[i - 1, j]) / dy2 +
                          (self.psi[i, j + 1] + self.psi[i, j - 1]) / dx2 +
                          self.omega[i, j]) / (2 / dx2 + 2 / dy2)
                    # SOR relaxation
                    self.psi[i, j] = (1 - omega_sor) * self.psi[i, j] + omega_sor * gs

            residual = np.max(np.abs(self.psi - psi_old))
            if residual < tol:
                return iteration + 1

        return max_iter

    def advance_vorticity(self, dt: float):
        """
        Advance vorticity using FTCS (Forward Time, Central Space).

        dw/dt = -(u * dw/dx + v * dw/dy) + (1/Re) * laplacian(w)
                  ^^^^ advection ^^^^       ^^^^ diffusion ^^^^

        Why FTCS can be unstable:
            FTCS for the advection term is conditionally stable. For higher Re,
            upwind differencing or implicit methods are needed. We use FTCS
            here for simplicity at low Re.
        """
        u, v = self.compute_velocity()
        dx = self.dx
        dy = self.dy
        nu = 1.0 / self.Re  # Kinematic viscosity

        omega_new = self.omega.copy()

        for i in range(1, self.N - 1):
            for j in range(1, self.N - 1):
                # Advection: central differences
                dwdx = (self.omega[i, j + 1] - self.omega[i, j - 1]) / (2 * dx)
                dwdy = (self.omega[i + 1, j] - self.omega[i - 1, j]) / (2 * dy)
                advection = u[i, j] * dwdx + v[i, j] * dwdy

                # Diffusion: 5-point Laplacian
                laplacian = ((self.omega[i + 1, j] + self.omega[i - 1, j] -
                              2 * self.omega[i, j]) / dy ** 2 +
                             (self.omega[i, j + 1] + self.omega[i, j - 1] -
                              2 * self.omega[i, j]) / dx ** 2)

                omega_new[i, j] = self.omega[i, j] + dt * (-advection + nu * laplacian)

        self.omega = omega_new

    def solve(self, n_steps: int = 2000, dt: float = 0.001,
              poisson_tol: float = 1e-5) -> dict:
        """
        Run the full simulation.

        Returns:
            Dictionary with convergence history
        """
        history = {"residuals": [], "max_vorticity": []}

        for step in range(n_steps):
            # 1. Solve Poisson for stream function
            n_poisson = self.solve_poisson(tol=poisson_tol)

            # 2. Update vorticity boundary conditions
            self.update_boundary_vorticity()

            # 3. Advance vorticity in time
            self.advance_vorticity(dt)

            # Track convergence
            max_omega = np.max(np.abs(self.omega))
            history["max_vorticity"].append(max_omega)

            if (step + 1) % 500 == 0:
                print(f"  Step {step + 1:5d}: max|omega| = {max_omega:.4f}, "
                      f"Poisson iters = {n_poisson}")

        return history


def visualize_cavity(solver: LidDrivenCavity, title_suffix: str = ""):
    """Visualize the cavity flow solution."""
    u, v = solver.compute_velocity()
    speed = np.sqrt(u ** 2 + v ** 2)

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))

    # 1. Stream function (streamlines)
    levels = np.linspace(solver.psi.min(), solver.psi.max(), 30)
    cs1 = axes[0, 0].contour(solver.X, solver.Y, solver.psi, levels=levels,
                              cmap='RdBu_r')
    axes[0, 0].set_title("Stream Function (Streamlines)")
    axes[0, 0].set_aspect('equal')
    plt.colorbar(cs1, ax=axes[0, 0])

    # 2. Vorticity field
    vmax = max(abs(solver.omega.min()), abs(solver.omega.max()))
    im2 = axes[0, 1].contourf(solver.X, solver.Y, solver.omega, levels=30,
                               cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[0, 1].set_title("Vorticity Field")
    axes[0, 1].set_aspect('equal')
    plt.colorbar(im2, ax=axes[0, 1])

    # 3. Velocity magnitude with quiver
    im3 = axes[1, 0].contourf(solver.X, solver.Y, speed, levels=20, cmap='viridis')
    skip = max(1, solver.N // 15)
    axes[1, 0].quiver(solver.X[::skip, ::skip], solver.Y[::skip, ::skip],
                      u[::skip, ::skip], v[::skip, ::skip],
                      color='white', scale=15, width=0.003)
    axes[1, 0].set_title("Velocity Magnitude + Vectors")
    axes[1, 0].set_aspect('equal')
    plt.colorbar(im3, ax=axes[1, 0])

    # 4. Centerline velocity profiles (validation)
    mid = solver.N // 2
    axes[1, 1].plot(u[:, mid], solver.y, 'b-o', markersize=2, label='u along x=0.5')
    axes[1, 1].plot(solver.x, v[mid, :], 'r-s', markersize=2, label='v along y=0.5')
    axes[1, 1].set_xlabel("Velocity")
    axes[1, 1].set_ylabel("Position")
    axes[1, 1].set_title("Centerline Profiles")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f"Lid-Driven Cavity (Re={solver.Re}, N={solver.N}) {title_suffix}",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig("navier_stokes_cavity.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: navier_stokes_cavity.png")


if __name__ == "__main__":
    print("=" * 60)
    print("2D Lid-Driven Cavity — Stream Function-Vorticity Method")
    print("=" * 60)

    # Why Re=100: Low enough for FTCS stability, high enough to show
    # the primary vortex and corner eddies characteristic of cavity flow.
    Re = 100
    N = 41       # Grid resolution (41x41)
    dt = 0.001   # Time step
    n_steps = 3000

    print(f"\nReynolds number: {Re}")
    print(f"Grid: {N}x{N}, dt={dt}, steps={n_steps}")
    print(f"CFL-like condition: dt * U / dx = {dt * 1.0 / (1.0 / (N - 1)):.3f}")

    solver = LidDrivenCavity(N=N, Re=Re)
    print("\nSolving...")
    history = solver.solve(n_steps=n_steps, dt=dt)

    u, v = solver.compute_velocity()
    print(f"\nMax velocity: u_max = {np.max(np.abs(u)):.4f}, "
          f"v_max = {np.max(np.abs(v)):.4f}")

    print("\nGenerating visualization...")
    visualize_cavity(solver)
