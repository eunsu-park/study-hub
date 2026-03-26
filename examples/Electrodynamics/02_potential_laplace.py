"""
Electrostatic Potential and Laplace's Equation (Finite Difference Solver)
=========================================================================

Topics covered:
  1. Finite difference discretization of Laplace's equation in 2D
  2. Boundary conditions for a conducting box
  3. Jacobi relaxation iteration
  4. Equipotential line visualization

Why solve Laplace's equation numerically?
  Laplace's equation (nabla^2 V = 0) governs the potential in charge-free
  regions. Analytic solutions exist only for simple geometries (infinite
  plates, spheres, cylinders). For realistic electrode shapes, we *must*
  solve numerically. The finite-difference method on a regular grid is
  the simplest and most instructive approach.

Physics background:
  - Poisson's equation:   nabla^2 V = -rho / eps0
  - Laplace's equation:   nabla^2 V = 0  (special case, rho = 0)
  - Boundary conditions:  V specified on conductors (Dirichlet BC)
  - E = -grad(V):  once we have V, the field follows immediately
"""

import numpy as np
import matplotlib.pyplot as plt


def laplace_jacobi(V, mask, max_iter=10000, tol=1e-6):
    """
    Solve Laplace's equation on a 2D grid using Jacobi iteration.

    Parameters
    ----------
    V : 2D array
        Initial potential grid with boundary values already set.
    mask : 2D bool array
        True where the potential is *fixed* (boundary or conductor).
        The solver will not update these points.
    max_iter : int
        Maximum number of relaxation iterations.
    tol : float
        Convergence criterion: max absolute change between iterations.

    Returns
    -------
    V : 2D array
        Converged potential.
    history : list of float
        Max change at each iteration (for convergence monitoring).

    Why Jacobi iteration (not Gauss-Seidel or SOR)?
      Jacobi is the simplest relaxation method: each grid point is updated
      to the average of its four neighbors, using *only* values from the
      previous iteration. This makes the algorithm:
        - Easy to understand and implement
        - Trivially parallelizable (each update is independent)
        - Slower to converge than Gauss-Seidel or SOR
      For a teaching example, clarity trumps speed. In production codes,
      you'd use SOR (Successive Over-Relaxation) or multigrid methods.

    Why does averaging neighbors solve Laplace's equation?
      The finite-difference approximation of nabla^2 V = 0 at grid point
      (i,j) with spacing h is:
        [V(i+1,j) + V(i-1,j) + V(i,j+1) + V(i,j-1) - 4*V(i,j)] / h^2 = 0
      Rearranging:
        V(i,j) = (1/4) * [V(i+1,j) + V(i-1,j) + V(i,j+1) + V(i,j-1)]
      This is exactly the Jacobi update rule! The solution of Laplace's
      equation is the unique potential that equals the average of its
      neighbors at every interior point.
    """
    Ny, Nx = V.shape
    history = []

    for iteration in range(max_iter):
        # Why copy?
        #   Jacobi iteration requires that ALL updates use values from the
        #   *previous* step, not partially-updated values (that would be
        #   Gauss-Seidel). So we work on a copy.
        V_old = V.copy()

        # Why slice [1:-1, 1:-1]?
        #   We skip the boundary rows/columns (index 0 and -1) because
        #   they are fixed by boundary conditions.
        V[1:-1, 1:-1] = 0.25 * (
            V_old[2:, 1:-1] +    # neighbor below  (i+1)
            V_old[:-2, 1:-1] +   # neighbor above  (i-1)
            V_old[1:-1, 2:] +    # neighbor right  (j+1)
            V_old[1:-1, :-2]     # neighbor left   (j-1)
        )

        # Why restore fixed points?
        #   Conductors and boundaries have prescribed potentials.
        #   The averaging step might have overwritten them.
        V[mask] = V_old[mask]

        # Convergence check
        max_change = np.max(np.abs(V - V_old))
        history.append(max_change)

        if max_change < tol:
            print(f"  Jacobi converged in {iteration + 1} iterations "
                  f"(max change = {max_change:.2e})")
            break
    else:
        print(f"  Jacobi did NOT converge after {max_iter} iterations "
              f"(max change = {max_change:.2e})")

    return V, history


def setup_conducting_box(Nx=101, Ny=101):
    """
    Set up a 2D domain with a conducting box inside.

    Geometry:
      - Outer boundary: V = 0  (grounded walls)
      - Inner square conductor: V = 100 V

    Why this geometry?
      A conductor at a different potential from the grounded walls
      creates a non-trivial field pattern. This is a simplified model
      of a capacitor or electrode inside a shielded enclosure.

    Returns
    -------
    V : 2D array (Ny, Nx)
        Initial potential with boundaries set.
    mask : 2D bool array
        Fixed-potential points.
    x, y : 1D arrays
        Coordinate vectors.
    """
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    V = np.zeros((Ny, Nx))
    mask = np.zeros((Ny, Nx), dtype=bool)

    # Outer boundary: V = 0 (grounded)
    # Why fix all four edges?
    #   Without fully specified boundaries, Laplace's equation doesn't
    #   have a unique solution. Dirichlet BCs on a closed boundary
    #   guarantee uniqueness (Uniqueness Theorem of electrostatics).
    V[0, :] = 0.0    # bottom
    V[-1, :] = 0.0   # top
    V[:, 0] = 0.0    # left
    V[:, -1] = 0.0   # right
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True

    # Inner conducting box: V = 100 V
    # Why a box and not a circle?
    #   A rectangular conductor on a rectangular grid aligns perfectly
    #   with grid points -- no staircase approximation needed. This
    #   keeps the example simple and the results accurate.
    box_x1 = int(0.35 * Nx)
    box_x2 = int(0.65 * Nx)
    box_y1 = int(0.35 * Ny)
    box_y2 = int(0.65 * Ny)

    V[box_y1:box_y2, box_x1:box_x2] = 100.0
    mask[box_y1:box_y2, box_x1:box_x2] = True

    return V, mask, x, y


def setup_asymmetric_plates(Nx=101, Ny=101):
    """
    Set up a domain with two parallel plates at different potentials.

    Geometry:
      - Top boundary: V = 100 V
      - Bottom boundary: V = 0 V
      - Left/Right: V = 0 V (grounded)

    Why include this geometry?
      Two plates at different potentials is the simplest capacitor model.
      The solution should be nearly linear in the interior (V ~ y * 100),
      with fringing effects near the left/right walls.
    """
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    V = np.zeros((Ny, Nx))
    mask = np.zeros((Ny, Nx), dtype=bool)

    # Bottom: V = 0
    V[0, :] = 0.0
    mask[0, :] = True

    # Top: V = 100
    V[-1, :] = 100.0
    mask[-1, :] = True

    # Left and right: linear gradient (to reduce edge effects)
    # Why a linear interpolation on the sides?
    #   If we set left/right to 0, the top corners have a discontinuity
    #   (100 on top, 0 on side). A linear gradient is the physically
    #   sensible BC that avoids artificial singularities at corners.
    V[:, 0] = np.linspace(0, 100, Ny)
    V[:, -1] = np.linspace(0, 100, Ny)
    mask[:, 0] = True
    mask[:, -1] = True

    return V, mask, x, y


def compute_electric_field(V, dx, dy):
    """
    Compute E = -grad(V) using central differences.

    Why central differences?
      Central differences give O(h^2) accuracy, while forward or backward
      differences are only O(h). For the same grid spacing, central
      differences are much more accurate.
    """
    Ey, Ex = np.gradient(V, dy, dx)
    # Why the negative sign?
    #   By convention, E = -grad(V). The electric field points from
    #   high potential to low potential (downhill in the potential landscape).
    return -Ex, -Ey


def plot_results(V, Ex, Ey, x, y, title_prefix, filename_prefix):
    """
    Create a comprehensive visualization: potential + field.
    """
    X, Y = np.meshgrid(x, y)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left: Potential with equipotential lines ---
    ax = axes[0]
    # Why use both imshow and contour?
    #   imshow gives a continuous color map of V, while contour lines
    #   mark specific voltage levels. Together they give both a global
    #   overview and precise quantitative information.
    im = ax.contourf(X, Y, V, levels=50, cmap='RdYlBu_r')
    ax.contour(X, Y, V, levels=15, colors='black', linewidths=0.5, alpha=0.7)
    fig.colorbar(im, ax=ax, label='V (Volts)')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(f'{title_prefix}: Potential')
    ax.set_aspect('equal')

    # --- Right: Electric field lines ---
    ax = axes[1]
    E_mag = np.sqrt(Ex**2 + Ey**2)
    E_log = np.log10(E_mag + 1e-10)

    strm = ax.streamplot(X, Y, Ex, Ey, color=E_log, cmap='viridis',
                         density=2, linewidth=1)
    fig.colorbar(strm.lines, ax=ax, label=r'$\log_{10}|E|$ (V/m)')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(f'{title_prefix}: Electric Field')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(f'{filename_prefix}.png', dpi=150)
    plt.close()
    print(f"[Saved] {filename_prefix}.png")


def plot_convergence(history, filename='02_convergence.png'):
    """Plot the convergence history of the Jacobi iteration."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Why log scale?
    #   The max change decreases roughly exponentially with iteration
    #   number. A log scale reveals the convergence rate clearly.
    ax.semilogy(history, 'b-', linewidth=1)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Max |V_new - V_old|')
    ax.set_title('Jacobi Iteration Convergence')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"[Saved] {filename}")


# ===========================
# Main
# ===========================

if __name__ == '__main__':
    Nx, Ny = 101, 101  # Grid resolution

    # --- Problem 1: Conducting box in grounded enclosure ---
    print("=== Problem 1: Conducting box ===")
    V1, mask1, x1, y1 = setup_conducting_box(Nx, Ny)
    V1, hist1 = laplace_jacobi(V1, mask1, max_iter=15000, tol=1e-5)

    dx1 = x1[1] - x1[0]
    dy1 = y1[1] - y1[0]
    Ex1, Ey1 = compute_electric_field(V1, dx1, dy1)
    plot_results(V1, Ex1, Ey1, x1, y1, 'Conducting Box', '02_box')
    plot_convergence(hist1)

    # --- Problem 2: Parallel plates (capacitor) ---
    print("\n=== Problem 2: Parallel plates ===")
    V2, mask2, x2, y2 = setup_asymmetric_plates(Nx, Ny)
    V2, hist2 = laplace_jacobi(V2, mask2, max_iter=15000, tol=1e-5)

    dx2 = x2[1] - x2[0]
    dy2 = y2[1] - y2[0]
    Ex2, Ey2 = compute_electric_field(V2, dx2, dy2)
    plot_results(V2, Ex2, Ey2, x2, y2, 'Parallel Plates', '02_plates')

    # Verify: potential should be nearly linear in y in the interior
    mid_x = Nx // 2
    V_centerline = V2[:, mid_x]
    V_linear = np.linspace(0, 100, Ny)
    max_deviation = np.max(np.abs(V_centerline - V_linear))
    print(f"  Max deviation from linear profile at center: {max_deviation:.4f} V")
