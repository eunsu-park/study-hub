"""
Capacitor Simulation: Parallel Plates, Dielectrics, and Fringing Fields
=======================================================================

Topics covered:
  1. Parallel plate capacitor electric field (ideal + fringing)
  2. Effect of dielectric insertion on field and capacitance
  3. Energy stored in the electric field
  4. Fringing field visualization near plate edges

Why study capacitors in detail?
  Capacitors are the workhorse of electrostatics. They store energy in
  the electric field and appear in virtually every electronic circuit.
  Understanding the ideal parallel plate model, its limitations (fringing),
  and dielectric effects bridges the gap between textbook formulas and
  real-world behavior.

Physics background:
  - Ideal parallel plate: E = sigma/eps0 = V/d  (uniform between plates)
  - Capacitance: C = eps0 * A / d  (vacuum), C = eps0 * eps_r * A / d  (dielectric)
  - Energy: U = (1/2) * C * V^2 = (1/2) * eps0 * integral(E^2 dV)
  - Fringing: field bows outward near plate edges (non-uniform)
"""

import numpy as np
import matplotlib.pyplot as plt


# Physical constants
EPS0 = 8.854187817e-12  # F/m


# ===========================
# 1. Ideal Parallel Plate Capacitor
# ===========================

def ideal_parallel_plate(V0, d, plate_length, Nx=200, Ny=200):
    """
    Compute the electric field of an ideal parallel plate capacitor
    using a simple model: uniform field between plates, zero outside.

    Parameters
    ----------
    V0 : float
        Voltage difference between plates (V).
    d : float
        Plate separation (m).
    plate_length : float
        Length of plates (m).

    Returns
    -------
    X, Y, Ex, Ey, V : arrays
        Grid coordinates, field components, and potential.

    Why start with the ideal model?
      The ideal model (infinite plates) gives E = V0/d uniformly between
      the plates and E = 0 outside. This is the textbook result and serves
      as our baseline. Comparing with the fringing model shows where the
      ideal model breaks down.
    """
    extent = plate_length * 1.5
    x = np.linspace(-extent, extent, Nx)
    y = np.linspace(-d * 2, d * 2, Ny)
    X, Y = np.meshgrid(x, y)

    Ex = np.zeros_like(X)
    Ey = np.zeros_like(Y)
    V_field = np.zeros_like(X)

    # Why define the between-plates region explicitly?
    #   The ideal parallel plate field is a step function: constant between
    #   the plates, zero outside. This sharp boundary is what "infinite plates"
    #   gives us -- no edge effects.
    between_plates = (np.abs(X) <= plate_length / 2) & \
                     (Y >= -d / 2) & (Y <= d / 2)

    # E points from + plate to - plate (let's say +y to -y)
    # Convention: top plate at +V0/2, bottom at -V0/2
    Ey[between_plates] = -V0 / d  # Why negative? Field points from high V to low V

    # Potential (linear interpolation between plates)
    V_field[between_plates] = V0 / 2 - V0 * (Y[between_plates] + d / 2) / d

    return X, Y, Ex, Ey, V_field


# ===========================
# 2. Fringing Field (Finite Difference Solver)
# ===========================

def fringing_field_solver(V0, d, plate_length, Nx=151, Ny=151, max_iter=8000):
    """
    Solve Laplace's equation for a finite parallel plate capacitor,
    capturing the fringing fields near the plate edges.

    Why use a PDE solver for fringing?
      The ideal model assumes infinite plates, but real capacitors have
      finite extent. Near the edges, field lines "bow out" (fringe),
      increasing the effective capacitance. This effect can only be
      captured by solving Laplace's equation with the actual geometry.
    """
    # Domain extends beyond plates to capture fringing
    Lx = plate_length * 2.0
    Ly = d * 4.0
    x = np.linspace(-Lx / 2, Lx / 2, Nx)
    y = np.linspace(-Ly / 2, Ly / 2, Ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    V = np.zeros((Ny, Nx))
    mask = np.zeros((Ny, Nx), dtype=bool)

    # Find plate indices
    # Why round to nearest grid point?
    #   The plates are 1D surfaces, but on a grid they must be placed
    #   at discrete positions. Rounding ensures consistent placement.
    y_top_idx = np.argmin(np.abs(y - d / 2))
    y_bot_idx = np.argmin(np.abs(y - (-d / 2)))
    x_left_idx = np.argmin(np.abs(x - (-plate_length / 2)))
    x_right_idx = np.argmin(np.abs(x - (plate_length / 2)))

    # Set plate potentials
    V[y_top_idx, x_left_idx:x_right_idx + 1] = V0 / 2
    V[y_bot_idx, x_left_idx:x_right_idx + 1] = -V0 / 2
    mask[y_top_idx, x_left_idx:x_right_idx + 1] = True
    mask[y_bot_idx, x_left_idx:x_right_idx + 1] = True

    # Boundary: V = 0 at domain edges (far enough away)
    V[0, :] = 0
    V[-1, :] = 0
    V[:, 0] = 0
    V[:, -1] = 0
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True

    # Jacobi relaxation
    # Why not use SOR here?
    #   Jacobi is simpler to implement and debug. For a 151x151 grid
    #   the convergence time (~seconds) is acceptable for a demo.
    for it in range(max_iter):
        V_old = V.copy()
        V[1:-1, 1:-1] = 0.25 * (
            V_old[2:, 1:-1] + V_old[:-2, 1:-1] +
            V_old[1:-1, 2:] + V_old[1:-1, :-2]
        )
        V[mask] = V_old[mask]

        if it % 500 == 0:
            change = np.max(np.abs(V - V_old))
            if change < 1e-5:
                print(f"  Fringing solver converged at iteration {it}")
                break

    Ey_field, Ex_field = np.gradient(V, dy, dx)
    Ex_field = -Ex_field
    Ey_field = -Ey_field

    return x, y, V, Ex_field, Ey_field


# ===========================
# 3. Dielectric Effect on Capacitance
# ===========================

def dielectric_effect():
    """
    Demonstrate how inserting a dielectric changes capacitance and
    stored energy.

    Why do dielectrics increase capacitance?
      A dielectric material contains polar molecules (or polarizable atoms).
      In an external E field, these align to partially cancel the field
      inside the material. For the same voltage V, more charge Q can
      accumulate on the plates: C = Q/V increases by a factor eps_r.

      Energy perspective: at constant voltage, U = (1/2)CV^2 increases.
      At constant charge, U = Q^2/(2C) decreases (the field does work
      pulling the dielectric in -- this is why dielectrics are attracted
      into capacitor gaps).
    """
    # Plate parameters
    A = 0.01       # plate area (m^2)
    d = 0.001      # separation (m)
    V0 = 100.0     # voltage (V)

    eps_r_values = np.linspace(1, 10, 100)

    # Capacitance vs dielectric constant
    C_values = EPS0 * eps_r_values * A / d

    # Energy stored at constant voltage
    U_values = 0.5 * C_values * V0**2

    # Electric field inside
    # Why does E decrease with eps_r at constant V?
    #   The voltage V = E * d is fixed, so E = V/d doesn't change for
    #   the *total* field. But the *free* field (from plates alone) is
    #   still sigma/(eps0), and the dielectric reduces the net field to
    #   sigma/(eps0 * eps_r). The apparent contradiction is resolved:
    #   inserting a dielectric at constant V requires additional charge
    #   on the plates, increasing sigma to maintain V = sigma*d/(eps0*eps_r).
    E_inside = V0 / d  # Independent of eps_r at constant V!

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # C vs eps_r
    ax = axes[0]
    ax.plot(eps_r_values, C_values * 1e12, 'b-', linewidth=2)
    ax.set_xlabel(r'$\varepsilon_r$')
    ax.set_ylabel('C (pF)')
    ax.set_title('Capacitance vs Dielectric Constant')
    ax.grid(True, alpha=0.3)

    # U vs eps_r
    ax = axes[1]
    ax.plot(eps_r_values, U_values * 1e6, 'r-', linewidth=2)
    ax.set_xlabel(r'$\varepsilon_r$')
    ax.set_ylabel(r'U ($\mu$J)')
    ax.set_title('Stored Energy (constant V)')
    ax.grid(True, alpha=0.3)

    # Energy density profile (with and without dielectric)
    ax = axes[2]
    y_pos = np.linspace(0, d, 100)
    u_vacuum = 0.5 * EPS0 * (V0 / d)**2 * np.ones_like(y_pos)
    eps_r_example = 4.0
    u_dielectric = 0.5 * EPS0 * eps_r_example * (V0 / d)**2 * np.ones_like(y_pos)

    ax.plot(y_pos * 1000, u_vacuum, 'b-', linewidth=2, label='Vacuum')
    ax.plot(y_pos * 1000, u_dielectric, 'r--', linewidth=2,
            label=rf'$\varepsilon_r = {eps_r_example}$')
    ax.set_xlabel('Position y (mm)')
    ax.set_ylabel(r'Energy density $u$ (J/m$^3$)')
    ax.set_title('Energy Density Between Plates')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('03_dielectric_effect.png', dpi=150)
    plt.close()
    print("[Saved] 03_dielectric_effect.png")

    # Print summary
    C_vac = EPS0 * A / d
    C_diel = EPS0 * eps_r_example * A / d
    print(f"\n  Vacuum capacitance:     C = {C_vac * 1e12:.2f} pF")
    print(f"  With eps_r={eps_r_example}:  C = {C_diel * 1e12:.2f} pF")
    print(f"  Ratio: {C_diel / C_vac:.1f}x (equals eps_r, as expected)")


# ===========================
# 4. Energy Stored Computation
# ===========================

def compute_energy(V, Ex, Ey, dx, dy, eps_r=1.0):
    """
    Compute the total electrostatic energy from the field.

    U = (1/2) * eps0 * eps_r * integral(E^2) dV

    Why compute energy from the field rather than from C and V?
      The formula U = (1/2)CV^2 only works for simple geometries where
      C is known. Computing from the field is general: it works for
      any geometry, including fringing fields, multiple dielectrics, etc.
      Also, the field energy viewpoint is fundamental -- the energy is
      *in the field itself*, not magically stored on the conductors.
    """
    E_squared = Ex**2 + Ey**2
    # Why multiply by dx * dy?
    #   This is the area element in the 2D integral. For a true 3D
    #   calculation we'd also need a depth dz, but for a 2D simulation
    #   we compute energy per unit depth.
    u = 0.5 * EPS0 * eps_r * E_squared
    U_per_length = np.sum(u) * dx * dy  # J/m (energy per unit depth)
    return U_per_length


# ===========================
# 5. Visualization
# ===========================

def plot_fringing(x, y, V, Ex, Ey, plate_length, d):
    """
    Visualize the fringing field near plate edges.

    Why focus on fringing?
      Fringing fields are what make real capacitors different from the
      textbook ideal. They increase effective capacitance, cause
      non-uniform energy density, and can lead to dielectric breakdown
      at sharp edges. Engineers must account for fringing in precision
      capacitor design.
    """
    X, Y = np.meshgrid(x, y)
    E_mag = np.sqrt(Ex**2 + Ey**2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Potential + equipotential lines
    ax = axes[0]
    levels_V = np.linspace(V.min(), V.max(), 30)
    cf = ax.contourf(X, Y, V, levels=levels_V, cmap='RdBu_r')
    ax.contour(X, Y, V, levels=15, colors='black', linewidths=0.3)
    fig.colorbar(cf, ax=ax, label='V (Volts)')

    # Mark plates
    ax.plot([-plate_length / 2, plate_length / 2], [d / 2, d / 2],
            'k-', linewidth=3, label='Plates')
    ax.plot([-plate_length / 2, plate_length / 2], [-d / 2, -d / 2],
            'k-', linewidth=3)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Potential with Fringing')
    ax.legend()
    ax.set_aspect('equal')

    # Field lines (streamplot) -- zoomed near edge
    ax = axes[1]
    E_log = np.log10(E_mag + 1)
    strm = ax.streamplot(X, Y, Ex, Ey, color=E_log, cmap='hot',
                         density=2, linewidth=1)
    fig.colorbar(strm.lines, ax=ax, label=r'$\log_{10}(|E|+1)$')
    ax.plot([-plate_length / 2, plate_length / 2], [d / 2, d / 2],
            'cyan', linewidth=3)
    ax.plot([-plate_length / 2, plate_length / 2], [-d / 2, -d / 2],
            'cyan', linewidth=3)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Electric Field (Fringing Visible)')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('03_fringing_field.png', dpi=150)
    plt.close()
    print("[Saved] 03_fringing_field.png")


# ===========================
# Main
# ===========================

if __name__ == '__main__':
    V0 = 100.0          # Voltage (V)
    d = 0.2              # Plate separation (m)
    plate_length = 0.5   # Plate length (m)

    # --- Fringing field computation ---
    print("=== Fringing Field Solver ===")
    x, y, V, Ex, Ey = fringing_field_solver(V0, d, plate_length)
    plot_fringing(x, y, V, Ex, Ey, plate_length, d)

    # --- Energy computation ---
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    U = compute_energy(V, Ex, Ey, dx, dy)
    print(f"  Stored energy (numerical, per unit depth): {U:.4e} J/m")

    # Compare with ideal capacitance formula
    C_ideal = EPS0 * plate_length / d  # per unit depth (2D)
    U_ideal = 0.5 * C_ideal * V0**2
    print(f"  Stored energy (ideal C formula):           {U_ideal:.4e} J/m")
    print(f"  Ratio (numerical/ideal): {U / U_ideal:.3f} "
          f"(>1 due to fringing)")

    # --- Dielectric effects ---
    print("\n=== Dielectric Effects ===")
    dielectric_effect()
