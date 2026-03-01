"""
2D Finite-Difference Time-Domain (FDTD) Simulation
===================================================

Topics covered:
  1. 2D FDTD for TM mode (E_z, H_x, H_y)
  2. Point source excitation
  3. Simplified PML absorbing boundary layers
  4. Scattering from a circular cylinder
  5. Field snapshots (static visualization)

Why 2D FDTD?
  1D FDTD demonstrates the algorithm, but real problems are 2D or 3D.
  In 2D, we see wavefronts, diffraction, and scattering -- phenomena
  absent in 1D. The TM polarization (E_z, H_x, H_y) is the natural
  choice because E_z is a scalar field easy to visualize.

Physics background (TM mode, z-invariant):
  dE_z/dt = (1/eps) * (dH_y/dx - dH_x/dy)
  dH_x/dt = -(1/mu0) * dE_z/dy
  dH_y/dt = (1/mu0) * dE_z/dx

Yee grid (2D):
  E_z[i,j] at (i*dx, j*dy)
  H_x[i,j] at (i*dx, (j+0.5)*dy)
  H_y[i,j] at ((i+0.5)*dx, j*dy)

PML (Perfectly Matched Layer):
  An artificial absorbing material that absorbs outgoing waves with
  minimal reflection, regardless of angle or frequency. It's the
  gold standard for open-boundary FDTD simulations.
"""

import numpy as np
import matplotlib.pyplot as plt

# Physical constants
C = 2.998e8
MU0 = 4 * np.pi * 1e-7
EPS0 = 8.854e-12


class FDTD2D:
    """
    2D FDTD simulation for TM polarization with simplified PML boundaries.

    Why TM and not TE?
      In TM mode, E_z is the only electric field component. This scalar
      field is easy to visualize as a 2D color map. TE mode would have
      H_z (scalar) with E_x, E_y -- equivalent physics but E is harder
      to visualize as a single image.
    """

    def __init__(self, Nx=200, Ny=200, dx=None, courant=0.5, pml_thickness=15):
        """
        Initialize 2D FDTD grid with PML boundaries.

        Parameters
        ----------
        Nx, Ny : int
            Grid dimensions.
        dx : float
            Spatial step (same in x and y for simplicity).
        courant : float
            Courant number. For 2D stability: S <= 1/sqrt(2) ~ 0.707.
        pml_thickness : int
            Number of PML cells on each side.

        Why courant = 0.5 for 2D?
          The stability limit in 2D is c*dt/dx <= 1/sqrt(2) ~ 0.707.
          Using 0.5 provides a safety margin and is a common choice.
        """
        self.Nx = Nx
        self.Ny = Ny
        self.dx = dx if dx is not None else 1e-3  # 1 mm default
        self.dy = self.dx  # square cells
        self.dt = courant * self.dx / (C * np.sqrt(2))  # 2D stability
        self.courant = courant
        self.pml_thickness = pml_thickness

        # Material arrays
        self.eps_r = np.ones((Nx, Ny))

        # Fields
        self.Ez = np.zeros((Nx, Ny))
        self.Hx = np.zeros((Nx, Ny))
        self.Hy = np.zeros((Nx, Ny))

        # PML conductivity profiles
        # Why sigma increases polynomially into the PML?
        #   A gradual increase in absorption avoids a sharp impedance
        #   mismatch at the PML interface, which would cause reflections.
        #   The polynomial grading (typically cubic or quartic) provides
        #   a smooth transition from zero conductivity to maximum.
        self.sigma_x = np.zeros((Nx, Ny))
        self.sigma_y = np.zeros((Nx, Ny))
        self._setup_pml()

        # PML auxiliary fields (needed for split-field PML)
        # Why split-field?
        #   The standard PML (Berenger) splits each field component into
        #   two sub-components, each absorbed in one direction only.
        #   This is mathematically simple and well-tested.
        self.Ez_x = np.zeros((Nx, Ny))  # E_z split into x-absorbed part
        self.Ez_y = np.zeros((Nx, Ny))  # E_z split into y-absorbed part

        self.time_step = 0

        print(f"  FDTD2D: {Nx}x{Ny}, dx={self.dx*1e3:.2f} mm, "
              f"dt={self.dt:.4e} s, PML={pml_thickness} cells")

    def _setup_pml(self):
        """
        Set up PML conductivity profiles.

        sigma(d) = sigma_max * (d / thickness)^3

        Why cubic polynomial?
          Linear grading (exponent 1) gives poor absorption for thick PMLs.
          Higher-order (3 or 4) grading concentrates absorption near the
          outer edge, giving better performance with fewer cells.
          sigma_max is chosen based on the theoretical optimum for a
          given PML thickness and polynomial order.
        """
        t = self.pml_thickness
        if t == 0:
            return

        # Optimal sigma_max for polynomial grading of order m
        # sigma_max = -(m+1) * ln(R) / (2 * eta * thickness * dx)
        # where R is the desired reflection coefficient (e.g., 1e-6)
        # and eta = sqrt(mu0/eps0) ~ 377 ohms
        m = 3  # polynomial order
        R_target = 1e-6
        eta = np.sqrt(MU0 / EPS0)
        sigma_max = -(m + 1) * np.log(R_target) / (2 * eta * t * self.dx)

        for i in range(t):
            # Why (t - i) / t?
            #   i = 0 is the inner edge (closest to the domain), i = t-1 is
            #   the outer edge. Conductivity increases from inner to outer.
            sigma_val = sigma_max * ((t - i) / t) ** m

            # x-direction PML (left and right)
            self.sigma_x[i, :] = sigma_val          # left
            self.sigma_x[-(i + 1), :] = sigma_val   # right

            # y-direction PML (bottom and top)
            self.sigma_y[:, i] = sigma_val          # bottom
            self.sigma_y[:, -(i + 1)] = sigma_val   # top

    def add_cylinder(self, cx, cy, radius, eps_r):
        """
        Add a circular dielectric cylinder to the domain.

        Why cylinder?
          Scattering from a cylinder is a canonical problem with an
          exact analytical solution (Mie theory in 2D). This makes it
          an excellent validation case for FDTD.
        """
        x = np.arange(self.Nx) * self.dx
        y = np.arange(self.Ny) * self.dy
        X, Y = np.meshgrid(x, y, indexing='ij')

        mask = (X - cx)**2 + (Y - cy)**2 <= radius**2
        self.eps_r[mask] = eps_r
        n_cells = np.sum(mask)
        print(f"  Cylinder: center=({cx*1e3:.1f}, {cy*1e3:.1f}) mm, "
              f"r={radius*1e3:.1f} mm, eps_r={eps_r}, cells={n_cells}")

    def gaussian_source(self, t_step, t0=30, spread=8):
        """Gaussian pulse in time."""
        return np.exp(-((t_step - t0) / spread) ** 2)

    def sinusoidal_source(self, t_step, freq):
        """Continuous sinusoidal source (turned on smoothly)."""
        # Why ramp-up?
        #   An abrupt turn-on creates artificial high-frequency content.
        #   A smooth ramp (1 - exp(-t/tau)) eliminates this.
        t = t_step * self.dt
        ramp = 1.0 - np.exp(-t / (20 * self.dt))
        return ramp * np.sin(2 * np.pi * freq * t)

    def step(self, source_pos=None, source_value=0.0):
        """
        Advance one full FDTD time step.

        The update order is crucial:
          1. Update H from E (half time step)
          2. Update E from H (half time step)
          3. Add source
        This leapfrog is second-order accurate in time.
        """
        dx = self.dx
        dy = self.dy
        dt = self.dt

        # --- Update H fields ---
        # dH_x/dt = -(1/mu0) * dE_z/dy
        self.Hx[:, :-1] -= (dt / MU0) * (self.Ez[:, 1:] - self.Ez[:, :-1]) / dy

        # dH_y/dt = (1/mu0) * dE_z/dx
        self.Hy[:-1, :] += (dt / MU0) * (self.Ez[1:, :] - self.Ez[:-1, :]) / dx

        # --- Update E field (split-field PML formulation) ---
        # dE_z_x/dt = (1/eps) * dH_y/dx - sigma_x * E_z_x / eps
        # dE_z_y/dt = -(1/eps) * dH_x/dy - sigma_y * E_z_y / eps
        eps = EPS0 * self.eps_r

        # Why split E_z into E_z_x and E_z_y?
        #   In the PML, each direction must be absorbed independently.
        #   E_z_x is the part that propagates in x (absorbed by sigma_x).
        #   E_z_y is the part that propagates in y (absorbed by sigma_y).
        #   In the main domain (sigma = 0), the split is transparent:
        #   E_z = E_z_x + E_z_y behaves exactly like the unsplit equation.

        # E_z_x update
        Ca_x = (1 - self.sigma_x * dt / (2 * eps)) / \
               (1 + self.sigma_x * dt / (2 * eps))
        Cb_x = (dt / eps) / (1 + self.sigma_x * dt / (2 * eps))

        self.Ez_x[1:, :] = (Ca_x[1:, :] * self.Ez_x[1:, :] +
                             Cb_x[1:, :] * (self.Hy[1:, :] - self.Hy[:-1, :]) / dx)

        # E_z_y update
        Ca_y = (1 - self.sigma_y * dt / (2 * eps)) / \
               (1 + self.sigma_y * dt / (2 * eps))
        Cb_y = (dt / eps) / (1 + self.sigma_y * dt / (2 * eps))

        self.Ez_y[:, 1:] = (Ca_y[:, 1:] * self.Ez_y[:, 1:] -
                             Cb_y[:, 1:] * (self.Hx[:, 1:] - self.Hx[:, :-1]) / dy)

        # Reconstruct total E_z
        self.Ez = self.Ez_x + self.Ez_y

        # Add source
        if source_pos is not None:
            si, sj = source_pos
            self.Ez[si, sj] += source_value

        self.time_step += 1


# ===========================
# 1. Point Source in Free Space
# ===========================

def demo_point_source():
    """
    Simulate a point source radiating in free space.

    Why start with free space?
      A point source should produce perfectly circular wavefronts.
      This validates the grid, the PML (no reflections from boundaries),
      and the source implementation.
    """
    Nx, Ny = 200, 200
    sim = FDTD2D(Nx=Nx, Ny=Ny, dx=1e-3, courant=0.5, pml_thickness=15)

    source_pos = (Nx // 2, Ny // 2)
    N_steps = 200

    snapshots = []
    snap_times = [40, 80, 120, 160]

    for step in range(N_steps):
        src = sim.gaussian_source(step, t0=30, spread=8)
        sim.step(source_pos=source_pos, source_value=src)

        if step in snap_times:
            snapshots.append((step, sim.Ez.copy()))

    # Plot snapshots
    x = np.arange(Nx) * sim.dx * 1000
    y = np.arange(Ny) * sim.dy * 1000

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    axes = axes.flatten()

    for ax, (t, Ez) in zip(axes, snapshots):
        vmax = np.max(np.abs(Ez)) * 0.5 or 1.0
        im = ax.imshow(Ez.T, extent=[x[0], x[-1], y[0], y[-1]],
                       origin='lower', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, aspect='equal')
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_title(f'Step {t}: E_z')

        # Mark PML region
        pml_mm = sim.pml_thickness * sim.dx * 1000
        rect = plt.Rectangle((pml_mm, pml_mm),
                              x[-1] - 2 * pml_mm, y[-1] - 2 * pml_mm,
                              linewidth=1, edgecolor='gray',
                              facecolor='none', linestyle='--')
        ax.add_patch(rect)

    plt.suptitle('2D FDTD: Point Source in Free Space', fontsize=14)
    plt.tight_layout()
    plt.savefig('11_fdtd2d_point_source.png', dpi=150)
    plt.close()
    print("[Saved] 11_fdtd2d_point_source.png")


# ===========================
# 2. Scattering from a Circular Cylinder
# ===========================

def demo_cylinder_scattering():
    """
    Simulate scattering of a cylindrical wave from a dielectric cylinder.

    Why cylinder scattering?
      It's a classic problem in computational EM. The circular wavefront
      from a point source hits the cylinder, creating:
        - A reflected (scattered) wave
        - A transmitted wave inside the cylinder
        - A shadow region behind the cylinder
        - Diffraction around the edges
      All of these phenomena are visible in the FDTD field snapshots.
    """
    Nx, Ny = 250, 250
    sim = FDTD2D(Nx=Nx, Ny=Ny, dx=0.5e-3, courant=0.5, pml_thickness=20)

    # Place cylinder at center
    cx = Nx // 2 * sim.dx
    cy = Ny // 2 * sim.dy
    radius = 15e-3  # 15 mm
    eps_r_cyl = 4.0  # dielectric cylinder (like glass)
    sim.add_cylinder(cx, cy, radius, eps_r_cyl)

    # Source to the left of center
    source_pos = (Nx // 4, Ny // 2)

    N_steps = 350
    snapshots = []
    snap_times = [60, 120, 180, 250, 300, 340]

    for step in range(N_steps):
        # Use sinusoidal source for clearer scattering pattern
        freq = 30e9  # 30 GHz (lambda ~ 10 mm)
        src = sim.sinusoidal_source(step, freq)
        sim.step(source_pos=source_pos, source_value=src)

        if step in snap_times:
            snapshots.append((step, sim.Ez.copy()))

    x = np.arange(Nx) * sim.dx * 1000
    y = np.arange(Ny) * sim.dy * 1000

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for ax, (t, Ez) in zip(axes, snapshots):
        vmax = np.max(np.abs(Ez)) * 0.3 or 1.0
        im = ax.imshow(Ez.T, extent=[x[0], x[-1], y[0], y[-1]],
                       origin='lower', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, aspect='equal')
        fig.colorbar(im, ax=ax, shrink=0.8)

        # Draw cylinder outline
        circle = plt.Circle((cx * 1000, cy * 1000), radius * 1000,
                             fill=False, color='black', linewidth=1.5)
        ax.add_patch(circle)

        # Mark source
        ax.plot(source_pos[0] * sim.dx * 1000,
                source_pos[1] * sim.dy * 1000, 'g*', markersize=10)

        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_title(f'Step {t}')

    plt.suptitle(f'2D FDTD: Scattering from Dielectric Cylinder (eps_r={eps_r_cyl})',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig('11_fdtd2d_cylinder.png', dpi=150)
    plt.close()
    print("[Saved] 11_fdtd2d_cylinder.png")


# ===========================
# 3. PML Effectiveness Test
# ===========================

def demo_pml_test():
    """
    Compare simulation with and without PML to demonstrate its effectiveness.

    Why test PML separately?
      PML reflections are the most common source of error in FDTD.
      By running the same problem with and without PML, we can see
      the spurious reflections from simple (hard) boundaries and
      appreciate how well PML suppresses them.
    """
    Nx, Ny = 150, 150

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (pml_t, title) in enumerate([(0, 'No PML (hard walls)'),
                                           (15, 'With PML (15 cells)')]):
        sim = FDTD2D(Nx=Nx, Ny=Ny, dx=1e-3, courant=0.5, pml_thickness=pml_t)
        source_pos = (Nx // 2, Ny // 2)

        # Run enough steps for the pulse to hit the boundary and reflect
        for step in range(200):
            src = sim.gaussian_source(step, t0=30, spread=8)
            sim.step(source_pos=source_pos, source_value=src)

        x = np.arange(Nx) * sim.dx * 1000
        y = np.arange(Ny) * sim.dy * 1000

        ax = axes[idx]
        vmax = 0.05  # low vmax to see reflections
        im = ax.imshow(sim.Ez.T, extent=[x[0], x[-1], y[0], y[-1]],
                       origin='lower', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, aspect='equal')
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_title(title)

    plt.suptitle('PML Effectiveness: Boundary Reflections at step 200', fontsize=14)
    plt.tight_layout()
    plt.savefig('11_fdtd2d_pml_test.png', dpi=150)
    plt.close()
    print("[Saved] 11_fdtd2d_pml_test.png")


# ===========================
# 4. Steady-State Field Pattern
# ===========================

def demo_steady_state():
    """
    Run with a continuous source until a quasi-steady-state pattern forms.
    Plot the time-averaged |E_z|^2 (intensity).

    Why time-averaged intensity?
      The instantaneous field oscillates rapidly. The time-averaged
      intensity <|E_z|^2> shows the standing wave pattern and
      interference structure clearly, similar to what a detector
      would measure.
    """
    Nx, Ny = 200, 200
    sim = FDTD2D(Nx=Nx, Ny=Ny, dx=0.5e-3, courant=0.5, pml_thickness=20)

    # Two-slit setup: metallic wall with two gaps
    wall_x = Nx // 2
    gap_width = 5      # cells
    gap_separation = 30  # cells between gap centers
    eps_metal = 1.0     # We'll use high conductivity instead

    # Set high conductivity to simulate metal (not perfect, but effective)
    # Why conductivity instead of PEC?
    #   True PEC requires special update equations. High conductivity
    #   (sigma >> omega*eps) gives near-perfect reflection with the
    #   standard lossy material formulation -- simpler to implement.
    metal_sigma = 1e6
    gap1_center = Ny // 2 - gap_separation // 2
    gap2_center = Ny // 2 + gap_separation // 2

    sim.sigma_x[wall_x, :] = metal_sigma
    # Open the gaps
    sim.sigma_x[wall_x, gap1_center - gap_width // 2:gap1_center + gap_width // 2] = 0
    sim.sigma_x[wall_x, gap2_center - gap_width // 2:gap2_center + gap_width // 2] = 0

    source_pos = (Nx // 4, Ny // 2)
    freq = 60e9  # 60 GHz

    # Accumulate intensity
    intensity = np.zeros((Nx, Ny))
    N_warmup = 200  # let transient die out
    N_measure = 300

    for step in range(N_warmup + N_measure):
        src = sim.sinusoidal_source(step, freq)
        sim.step(source_pos=source_pos, source_value=src)

        if step >= N_warmup:
            intensity += sim.Ez**2

    intensity /= N_measure

    x = np.arange(Nx) * sim.dx * 1000
    y = np.arange(Ny) * sim.dy * 1000

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(np.log10(intensity.T + 1e-20),
                   extent=[x[0], x[-1], y[0], y[-1]],
                   origin='lower', cmap='hot', aspect='equal')
    fig.colorbar(im, ax=ax, label=r'$\log_{10}\langle E_z^2 \rangle$')

    # Mark slits
    ax.axvline(x=wall_x * sim.dx * 1000, color='cyan', linestyle='--',
               alpha=0.5, label='Slit wall')
    ax.plot(wall_x * sim.dx * 1000, gap1_center * sim.dy * 1000,
            'co', markersize=8)
    ax.plot(wall_x * sim.dx * 1000, gap2_center * sim.dy * 1000,
            'co', markersize=8)
    ax.plot(source_pos[0] * sim.dx * 1000, source_pos[1] * sim.dy * 1000,
            'g*', markersize=12, label='Source')

    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title('Double-Slit Diffraction: Time-Averaged Intensity')
    ax.legend()

    plt.tight_layout()
    plt.savefig('11_fdtd2d_double_slit.png', dpi=150)
    plt.close()
    print("[Saved] 11_fdtd2d_double_slit.png")


# ===========================
# Main
# ===========================

if __name__ == '__main__':
    print("=== 2D FDTD: Point Source ===")
    demo_point_source()

    print("\n=== 2D FDTD: Cylinder Scattering ===")
    demo_cylinder_scattering()

    print("\n=== 2D FDTD: PML Test ===")
    demo_pml_test()

    print("\n=== 2D FDTD: Double-Slit Diffraction ===")
    demo_steady_state()
