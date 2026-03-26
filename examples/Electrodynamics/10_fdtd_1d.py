"""
1D Finite-Difference Time-Domain (FDTD) Simulation
===================================================

Topics covered:
  1. 1D FDTD with Yee grid (staggered E and H fields)
  2. Gaussian pulse source injection
  3. Simple absorbing boundary conditions (ABC)
  4. Dielectric slab reflection and transmission

Why FDTD?
  FDTD is the most widely used numerical method for solving Maxwell's
  equations in the time domain. It directly discretizes the curl equations
  on a staggered grid (Yee grid), leapfrogging E and H updates in time.
  Advantages:
    - Simple to implement
    - Naturally broadband (one run covers all frequencies)
    - Handles complex materials (dispersive, nonlinear)
    - Scales well to large problems

Physics background:
  Maxwell's curl equations (1D, linear, non-magnetic):
    dE_x/dt = (1/eps) * dH_y/dz
    dH_y/dt = (1/mu0) * dE_x/dz

  Yee grid (1D):
    E[k] is located at z = k*dz
    H[k] is located at z = (k + 0.5)*dz
    E and H are updated at alternating half-time-steps (leapfrog)

  Courant condition (stability):
    dt <= dz / c   (Courant number S = c*dt/dz <= 1)
"""

import numpy as np
import matplotlib.pyplot as plt

# Physical constants
C = 2.998e8       # m/s
MU0 = 4 * np.pi * 1e-7  # T*m/A
EPS0 = 8.854e-12  # F/m


# ===========================
# 1. FDTD Core Engine (1D)
# ===========================

class FDTD1D:
    """
    1D FDTD simulation engine.

    Why a class?
      Encapsulating the grid, material properties, source, and update
      logic in a class makes it easy to run multiple configurations
      (free space, dielectric slab) without code duplication.
    """

    def __init__(self, Nz=500, dz=None, courant=0.5):
        """
        Initialize the 1D FDTD grid.

        Parameters
        ----------
        Nz : int
            Number of spatial cells.
        dz : float or None
            Spatial step (m). If None, set to 1e-3 m (1 mm).
        courant : float
            Courant number S = c*dt/dz. Must be <= 1 for stability.
            Using 0.5 gives a good balance of accuracy and stability.

        Why Courant number = 0.5 (not 1.0)?
          S = 1.0 gives exact dispersion in 1D (magic time step), but
          in multi-dimensional FDTD the stability limit is S <= 1/sqrt(d)
          where d is the number of dimensions. Using S = 0.5 works in
          all dimensions and teaches good habits.
        """
        self.Nz = Nz
        self.dz = dz if dz is not None else 1e-3
        self.dt = courant * self.dz / C
        self.courant = courant

        # Why arrays of eps_r instead of a single value?
        #   This allows spatially varying materials (e.g., dielectric slabs).
        #   Each cell can have a different permittivity.
        self.eps_r = np.ones(Nz)  # relative permittivity at each E-node
        self.sigma = np.zeros(Nz)  # conductivity (for lossy materials)

        # Fields: E[k] at integer positions, H[k] at half-integer positions
        # Why separate arrays (not a combined EM vector)?
        #   The Yee algorithm updates E and H at different times.
        #   Keeping them separate makes the leapfrog structure explicit.
        self.E = np.zeros(Nz)
        self.H = np.zeros(Nz)

        # Update coefficients (precomputed for efficiency)
        # Why precompute?
        #   In the time loop, we avoid recomputing eps0*eps_r and other
        #   material-dependent factors at every step. This can speed up
        #   the simulation significantly for large grids.
        self._compute_coefficients()

        # Source parameters
        self.source_pos = Nz // 4  # default source position
        self.time_step = 0

        # Storage for boundary ABC
        self._E_left_prev = 0.0
        self._E_right_prev = 0.0

        print(f"  FDTD1D: Nz={Nz}, dz={self.dz*1e3:.2f} mm, "
              f"dt={self.dt:.4e} s, S={courant}")

    def _compute_coefficients(self):
        """Precompute update coefficients from material properties."""
        # E update: E_new = Ca * E_old + Cb * (dH/dz)
        # From: eps * dE/dt + sigma * E = dH/dz (curl H equation with losses)
        eps_eff = EPS0 * self.eps_r
        self.Ca = (1 - self.sigma * self.dt / (2 * eps_eff)) / \
                  (1 + self.sigma * self.dt / (2 * eps_eff))
        self.Cb = (self.dt / eps_eff) / \
                  (1 + self.sigma * self.dt / (2 * eps_eff))

    def set_dielectric_slab(self, start, end, eps_r, sigma=0.0):
        """
        Insert a dielectric slab between grid indices start and end.

        Why this is useful:
          A dielectric slab is the simplest structure for studying
          reflection and transmission. It tests that our FDTD handles
          material interfaces correctly.
        """
        self.eps_r[start:end] = eps_r
        self.sigma[start:end] = sigma
        self._compute_coefficients()
        print(f"  Dielectric slab: indices [{start}, {end}), "
              f"eps_r={eps_r}, sigma={sigma}")

    def gaussian_source(self, t_step, t0=50, spread=12):
        """
        Gaussian pulse source: E(t) = exp(-((t - t0) / spread)^2)

        Why Gaussian pulse?
          A Gaussian pulse is smooth (no sharp edges that create
          numerical artifacts) and has a known, controllable bandwidth.
          Narrower spread = broader bandwidth = more frequencies
          excited in a single simulation.

        Why "soft source" (additive) instead of "hard source"?
          A hard source (E[k] = source_value) creates artificial
          reflections because it forces the field to a fixed value.
          A soft source (E[k] += source_value) adds to the existing
          field, allowing waves to pass through the source point
          without reflection.
        """
        return np.exp(-((t_step - t0) / spread)**2)

    def update_H(self):
        """
        Update H field (leapfrog: H at t+0.5*dt from E at t).

        dH_y/dt = (1/mu0) * dE_x/dz
        H[k]^{n+1/2} = H[k]^{n-1/2} + (dt/mu0) * (E[k+1]^n - E[k]^n) / dz

        Why H[k] uses E[k+1] - E[k]?
          On the Yee grid, H[k] sits between E[k] and E[k+1].
          The spatial derivative is naturally a forward difference
          from H's perspective.
        """
        self.H[:-1] += (self.dt / (MU0 * self.dz)) * (self.E[1:] - self.E[:-1])

    def update_E(self):
        """
        Update E field (leapfrog: E at t+dt from H at t+0.5*dt).

        eps * dE_x/dt = dH_y/dz
        E[k]^{n+1} = Ca[k]*E[k]^n + Cb[k] * (H[k]^{n+1/2} - H[k-1]^{n+1/2}) / dz

        Why E[k] uses H[k] - H[k-1]?
          On the Yee grid, E[k] sits between H[k-1] and H[k].
          This backward difference (from E's perspective) creates
          the staggered, self-consistent update scheme.
        """
        self.E[1:] = (self.Ca[1:] * self.E[1:] +
                      self.Cb[1:] * (self.H[1:] - self.H[:-1]) / self.dz)

    def apply_abc(self):
        """
        Apply simple first-order absorbing boundary conditions (ABC).

        The simplest ABC: E at the boundary at time n+1 equals E at
        the adjacent cell at time n. This "absorbs" outgoing waves
        for normal incidence, but is imperfect for oblique incidence
        (not an issue in 1D).

        Why not periodic BCs?
          Periodic BCs would wrap waves around, causing interference.
          We want waves to leave the domain cleanly, simulating an
          infinite space. PML (Perfectly Matched Layer) is more
          accurate but more complex -- we use PML in the 2D example.
        """
        # Left boundary
        self.E[0] = self._E_left_prev
        self._E_left_prev = self.E[1]

        # Right boundary
        self.E[-1] = self._E_right_prev
        self._E_right_prev = self.E[-2]

    def step(self):
        """Advance one full time step."""
        self.update_H()
        self.update_E()

        # Add source (soft source)
        self.E[self.source_pos] += self.gaussian_source(self.time_step)

        self.apply_abc()
        self.time_step += 1


# ===========================
# 2. Free Space Propagation Demo
# ===========================

def demo_free_space():
    """
    Run FDTD in free space and show the Gaussian pulse propagating.

    Why start with free space?
      This validates the basic algorithm: the pulse should propagate
      at exactly c, maintain its shape (no numerical dispersion at
      S = 0.5 in 1D), and be absorbed cleanly at the boundaries.
    """
    sim = FDTD1D(Nz=500, dz=1e-3, courant=0.5)

    # Record snapshots
    snapshots = []
    snap_times = [0, 80, 160, 240, 320, 400]

    for step in range(max(snap_times) + 1):
        sim.step()
        if step in snap_times:
            snapshots.append((step, sim.E.copy()))

    z = np.arange(sim.Nz) * sim.dz * 1000  # mm

    fig, axes = plt.subplots(len(snapshots), 1, figsize=(12, 10), sharex=True)
    for ax, (t, E) in zip(axes, snapshots):
        ax.plot(z, E, 'b-', linewidth=1.5)
        ax.set_ylabel('E')
        ax.set_title(f'Time step {t}')
        ax.set_ylim(-1.1, 1.1)
        ax.grid(True, alpha=0.3)
        # Mark source position
        ax.axvline(x=sim.source_pos * sim.dz * 1000, color='red',
                   linestyle=':', alpha=0.5)

    axes[-1].set_xlabel('z (mm)')
    plt.suptitle('1D FDTD: Gaussian Pulse in Free Space', fontsize=14)
    plt.tight_layout()
    plt.savefig('10_fdtd_free_space.png', dpi=150)
    plt.close()
    print("[Saved] 10_fdtd_free_space.png")


# ===========================
# 3. Dielectric Slab Reflection/Transmission
# ===========================

def demo_dielectric_slab():
    """
    Simulate a Gaussian pulse hitting a dielectric slab and observe
    reflection and transmission.

    Why this is a critical test:
      The analytic Fresnel coefficients give exact reflection and
      transmission for a slab. Comparing FDTD results with analytics
      validates the algorithm's handling of material interfaces.
    """
    sim = FDTD1D(Nz=600, dz=0.5e-3, courant=0.5)

    # Dielectric slab: eps_r = 4 (like glass)
    slab_start = 350
    slab_end = 450
    eps_r_slab = 4.0
    sim.set_dielectric_slab(slab_start, slab_end, eps_r_slab)

    # Detector positions
    det_inc = 200   # before slab (reflected wave detector)
    det_trans = 500  # after slab (transmitted wave detector)

    E_det_inc = []
    E_det_trans = []

    N_steps = 800
    snapshots = []
    snap_times = [100, 200, 350, 450, 550, 700]

    for step in range(N_steps):
        sim.step()
        E_det_inc.append(sim.E[det_inc])
        E_det_trans.append(sim.E[det_trans])
        if step in snap_times:
            snapshots.append((step, sim.E.copy()))

    z = np.arange(sim.Nz) * sim.dz * 1000  # mm

    # --- Snapshots ---
    fig, axes = plt.subplots(len(snapshots), 1, figsize=(12, 12), sharex=True)
    for ax, (t, E) in zip(axes, snapshots):
        ax.plot(z, E, 'b-', linewidth=1.5)
        # Mark dielectric slab
        ax.axvspan(slab_start * sim.dz * 1000, slab_end * sim.dz * 1000,
                   alpha=0.2, color='orange', label=f'Slab eps_r={eps_r_slab}')
        ax.set_ylabel('E')
        ax.set_title(f'Step {t}')
        ax.set_ylim(-1.1, 1.1)
        ax.grid(True, alpha=0.3)
        if t == snap_times[0]:
            ax.legend(fontsize=9)

    axes[-1].set_xlabel('z (mm)')
    plt.suptitle('1D FDTD: Pulse Hitting a Dielectric Slab', fontsize=14)
    plt.tight_layout()
    plt.savefig('10_fdtd_dielectric_snapshots.png', dpi=150)
    plt.close()
    print("[Saved] 10_fdtd_dielectric_snapshots.png")

    # --- Time traces at detectors ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))

    t_axis = np.arange(N_steps) * sim.dt * 1e9  # nanoseconds

    ax = axes[0]
    ax.plot(t_axis, E_det_inc, 'b-', linewidth=1.5)
    ax.set_ylabel('E at detector (before slab)')
    ax.set_title('Incident + Reflected pulse')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t_axis, E_det_trans, 'r-', linewidth=1.5)
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('E at detector (after slab)')
    ax.set_title('Transmitted pulse')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('10_fdtd_dielectric_traces.png', dpi=150)
    plt.close()
    print("[Saved] 10_fdtd_dielectric_traces.png")

    # --- Estimate reflection coefficient ---
    # Why use peak amplitude ratio?
    #   For a broadband pulse, the reflection coefficient varies with
    #   frequency. The peak ratio gives an approximate average.
    #   For normal incidence on a single interface: r = (1 - n)/(1 + n).
    E_inc_peak = max(np.abs(E_det_inc[:300]))  # incident pulse
    E_ref_peak = max(np.abs(E_det_inc[300:]))  # reflected pulse (arrives later)
    E_trans_peak = max(np.abs(E_det_trans))

    n_slab = np.sqrt(eps_r_slab)
    r_analytic = (1 - n_slab) / (1 + n_slab)  # single interface
    t_analytic = 2 / (1 + n_slab)              # single interface

    print(f"\n  Incident peak:   {E_inc_peak:.4f}")
    print(f"  Reflected peak:  {E_ref_peak:.4f}")
    print(f"  Transmitted peak: {E_trans_peak:.4f}")
    print(f"  Reflection ratio: {E_ref_peak/E_inc_peak:.4f}")
    print(f"  Analytic |r| (single interface): {abs(r_analytic):.4f}")
    print(f"  Note: slab has two interfaces -> Fabry-Perot interference")


# ===========================
# 4. Numerical Dispersion Analysis
# ===========================

def numerical_dispersion():
    """
    Measure numerical dispersion by comparing pulse arrival times
    at different Courant numbers.

    Why care about numerical dispersion?
      In FDTD, the numerical phase velocity is slightly different from
      c. This error depends on the Courant number S and the ratio
      dz/lambda. For S = 1 in 1D, the dispersion is exactly zero
      (magic time step). For S < 1, waves travel slightly slower than c.
      Understanding this helps choose grid parameters for accurate results.
    """
    courant_values = [0.25, 0.5, 0.75, 1.0]
    detector_pos = 400
    Nz = 500

    fig, ax = plt.subplots(figsize=(10, 6))

    for S in courant_values:
        sim = FDTD1D(Nz=Nz, dz=1e-3, courant=S)
        sim.source_pos = 100

        E_trace = []
        for step in range(600):
            sim.step()
            E_trace.append(sim.E[detector_pos])

        t_axis = np.arange(len(E_trace)) * sim.dt * 1e9
        ax.plot(t_axis, E_trace, linewidth=1.5, label=f'S = {S}')

    # Expected arrival time
    distance = (detector_pos - 100) * 1e-3  # m
    t_expected = distance / C * 1e9  # ns

    ax.axvline(x=t_expected, color='black', linestyle='--', alpha=0.5,
               label=f'Expected arrival: {t_expected:.3f} ns')
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('E at detector')
    ax.set_title('Numerical Dispersion: Pulse Arrival vs Courant Number')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('10_fdtd_dispersion.png', dpi=150)
    plt.close()
    print("[Saved] 10_fdtd_dispersion.png")
    print(f"  Expected arrival time: {t_expected:.4f} ns")
    print(f"  S=1.0 should arrive exactly on time (1D magic time step)")


# ===========================
# Main
# ===========================

if __name__ == '__main__':
    print("=== 1D FDTD: Free Space ===")
    demo_free_space()

    print("\n=== 1D FDTD: Dielectric Slab ===")
    demo_dielectric_slab()

    print("\n=== Numerical Dispersion Analysis ===")
    numerical_dispersion()
