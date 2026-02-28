"""
Exercises for Lesson 13: CFD Basics
Topic: Numerical_Simulation

Solutions to practice problems from the lesson.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# === Exercise 1: Reynolds Number and Flow Regime ===
# Problem: Calculate Re for water through a 5 cm pipe at 2 m/s. Find min velocity
# for turbulent flow.

def exercise_1():
    """Reynolds number calculation and flow regime determination."""
    # Water at 20 deg C
    rho = 998.0      # kg/m^3
    mu = 1.002e-3     # Pa s
    D = 0.05          # m (5 cm)
    U = 2.0           # m/s

    Re = rho * U * D / mu
    nu = mu / rho

    regime = "Laminar" if Re < 2300 else ("Transition" if Re < 4000 else "Turbulent")

    print(f"Water at 20 deg C: rho={rho} kg/m^3, mu={mu} Pa.s")
    print(f"Pipe diameter D = {D*100} cm, velocity U = {U} m/s")
    print(f"Reynolds number Re = {Re:.0f}")
    print(f"Flow regime: {regime}")

    # Minimum velocity for turbulent flow (Re = 4000)
    Re_turb = 4000
    U_min_turb = Re_turb * mu / (rho * D)
    print(f"\nMinimum velocity for turbulent flow (Re = {Re_turb}):")
    print(f"  U_min = {U_min_turb:.4f} m/s")

    # Also find transition onset
    Re_trans = 2300
    U_trans = Re_trans * mu / (rho * D)
    print(f"Transition onset (Re = {Re_trans}):")
    print(f"  U_trans = {U_trans:.4f} m/s")


# === Exercise 2: Navier-Stokes Term Analysis ===
# Problem: For u = sin(x)cos(y), v = -cos(x)sin(y), verify continuity
# and compute vorticity.

def exercise_2():
    """Verify continuity equation and compute vorticity for given flow field."""
    # Velocity field: u = sin(x)cos(y), v = -cos(x)sin(y)
    nx = ny = 50
    x = np.linspace(0, 2 * np.pi, nx)
    y = np.linspace(0, 2 * np.pi, ny)
    X, Y = np.meshgrid(x, y)

    u = np.sin(X) * np.cos(Y)
    v = -np.cos(X) * np.sin(Y)

    # Continuity: du/dx + dv/dy = 0
    # du/dx = cos(x)cos(y)
    # dv/dy = -cos(x)cos(y)
    # Sum = 0 -> continuity satisfied!

    dudx = np.cos(X) * np.cos(Y)
    dvdy = -np.cos(X) * np.cos(Y)
    divergence = dudx + dvdy

    print("Velocity field: u = sin(x)cos(y), v = -cos(x)sin(y)")
    print(f"  Max |div(u)| = {np.max(np.abs(divergence)):.2e}")
    print("  Continuity equation is satisfied (div = 0 analytically)")

    # Vorticity: omega = dv/dx - du/dy
    # dv/dx = sin(x)sin(y)
    # du/dy = -sin(x)sin(y)
    # omega = sin(x)sin(y) - (-sin(x)sin(y)) = 2 sin(x)sin(y)
    dvdx = np.sin(X) * np.sin(Y)
    dudy = -np.sin(X) * np.sin(Y)
    omega = dvdx - dudy

    omega_analytical = 2 * np.sin(X) * np.sin(Y)
    omega_error = np.max(np.abs(omega - omega_analytical))

    print(f"\n  Vorticity omega = dv/dx - du/dy = 2 sin(x)sin(y)")
    print(f"  Verification error: {omega_error:.2e}")
    print("  Physical meaning: vorticity measures local rotation of the fluid.")
    print("  Positive omega = counterclockwise rotation, negative = clockwise.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    skip = 3
    ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
              u[::skip, ::skip], v[::skip, ::skip], color='blue', alpha=0.7)
    ax.set_title('Velocity Field (u, v)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')

    ax = axes[1]
    im = ax.contourf(X, Y, omega, levels=20, cmap='RdBu_r')
    plt.colorbar(im, ax=ax, label='Vorticity')
    ax.set_title('Vorticity: omega = 2 sin(x)sin(y)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('ex13_vorticity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Plot saved: ex13_vorticity.png")


# === Exercise 3: Poiseuille Flow Comparison ===
# Problem: Implement Poiseuille flow and measure grid convergence.

def exercise_3():
    """Poiseuille flow FD solution vs analytical, grid convergence study."""
    H = 1.0
    mu = 0.01
    dpdx = -1.0

    def u_exact(y):
        return -(1 / (2 * mu)) * dpdx * y * (H - y)

    grid_sizes = [10, 20, 40]
    errors = []

    print("Poiseuille flow grid convergence:")
    print(f"  H={H}, mu={mu}, dp/dx={dpdx}")
    print(f"  Analytical: u_max = {H**2 / (8*mu) * abs(dpdx):.4f}")
    print()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for Ny in grid_sizes:
        dy = H / (Ny - 1)
        y = np.linspace(0, H, Ny)

        # Solve d^2u/dy^2 = (1/mu) dp/dx with u(0) = u(H) = 0
        # Finite difference: (u_{j+1} - 2u_j + u_{j-1})/dy^2 = (1/mu) dp/dx
        A = np.zeros((Ny, Ny))
        b = np.zeros(Ny)

        # BCs
        A[0, 0] = 1
        A[-1, -1] = 1

        # Interior
        for j in range(1, Ny - 1):
            A[j, j - 1] = 1 / dy**2
            A[j, j] = -2 / dy**2
            A[j, j + 1] = 1 / dy**2
            b[j] = (1 / mu) * dpdx

        u_num = np.linalg.solve(A, b)
        u_ex = u_exact(y)

        l2_err = np.sqrt(np.sum((u_num - u_ex)**2) * dy)
        errors.append(l2_err)

        axes[0].plot(u_num, y, 'o-', markersize=3, label=f'Ny={Ny}')
        print(f"  Ny={Ny:3d}, dy={dy:.4f}: L2 error = {l2_err:.6e}")

    # Exact solution on fine grid
    y_fine = np.linspace(0, H, 200)
    axes[0].plot(u_exact(y_fine), y_fine, 'k-', linewidth=2, label='Exact')
    axes[0].set_xlabel('u [m/s]')
    axes[0].set_ylabel('y [m]')
    axes[0].set_title('Poiseuille Flow: Numerical vs Exact')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Convergence order
    dy_vals = [H / (Ny - 1) for Ny in grid_sizes]
    axes[1].loglog(dy_vals, errors, 'bo-', markersize=8, linewidth=2, label='L2 Error')
    # Reference line for O(h^2)
    ref = [errors[0] * (d / dy_vals[0])**2 for d in dy_vals]
    axes[1].loglog(dy_vals, ref, 'r--', linewidth=1.5, label='O(h^2) reference')
    axes[1].set_xlabel('dy')
    axes[1].set_ylabel('L2 Error')
    axes[1].set_title('Grid Convergence')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, which='both')

    # Convergence rate
    for i in range(len(grid_sizes) - 1):
        rate = np.log(errors[i] / errors[i + 1]) / np.log(dy_vals[i] / dy_vals[i + 1])
        print(f"  Convergence rate ({grid_sizes[i]} -> {grid_sizes[i+1]}): {rate:.2f}")

    plt.tight_layout()
    plt.savefig('ex13_poiseuille_convergence.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Plot saved: ex13_poiseuille_convergence.png")
    print("  Expected: O(h^2) convergence for central differences")


# === Exercise 4: Boundary Layer Scaling ===
# Problem: Compute boundary layer thickness, Cf, and transition location.

def exercise_4():
    """Boundary layer scaling for flat plate in air flow."""
    nu = 1.5e-5  # m^2/s (air at 20 C)
    U_inf = 10.0  # m/s

    x = np.linspace(0.001, 0.5, 200)  # 0 < x < 0.5 m
    Re_x = U_inf * x / nu

    # (a) Laminar BL thickness (Blasius)
    delta = 5.0 * x / np.sqrt(Re_x)

    # (b) Local friction coefficient
    Cf = 0.664 / np.sqrt(Re_x)

    # (c) Transition location: Re_x = 5e5
    Re_trans = 5e5
    x_trans = Re_trans * nu / U_inf

    print(f"Flat plate boundary layer (air, U_inf = {U_inf} m/s):")
    print(f"  Kinematic viscosity nu = {nu} m^2/s")
    print(f"  Transition location: x_trans = {x_trans*1000:.1f} mm (Re_x = {Re_trans:.0e})")

    # At what speed does transition move to x = 0.1 m?
    x_target = 0.1  # m
    U_for_trans = Re_trans * nu / x_target
    print(f"\n  For transition at x = {x_target} m:")
    print(f"  Required velocity: U = {U_for_trans:.2f} m/s")

    # Sample values
    for xi in [0.05, 0.1, 0.2, 0.5]:
        if xi <= 0.5:
            Re_i = U_inf * xi / nu
            delta_i = 5.0 * xi / np.sqrt(Re_i)
            Cf_i = 0.664 / np.sqrt(Re_i)
            print(f"  x={xi:.2f}m: Re_x={Re_i:.0f}, delta={delta_i*1000:.2f}mm, Cf={Cf_i:.6f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.plot(x * 1000, delta * 1000, 'b-', linewidth=2)
    ax.axvline(x=x_trans * 1000, color='r', linestyle='--', label=f'Transition ({x_trans*1000:.0f} mm)')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('delta [mm]')
    ax.set_title('Boundary Layer Thickness (Blasius)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(x * 1000, Cf, 'r-', linewidth=2)
    ax.axvline(x=x_trans * 1000, color='gray', linestyle='--')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('Cf')
    ax.set_title('Local Friction Coefficient')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.semilogy(x * 1000, Re_x, 'g-', linewidth=2)
    ax.axhline(y=Re_trans, color='r', linestyle='--', label=f'Re_crit = {Re_trans:.0e}')
    ax.axvline(x=x_trans * 1000, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('Re_x')
    ax.set_title('Local Reynolds Number')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    plt.suptitle(f'Boundary Layer (U_inf = {U_inf} m/s, air)', fontsize=13)
    plt.tight_layout()
    plt.savefig('ex13_boundary_layer.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Plot saved: ex13_boundary_layer.png")


if __name__ == "__main__":
    print("=== Exercise 1: Reynolds Number and Flow Regime ===")
    exercise_1()
    print("\n=== Exercise 2: Navier-Stokes Term Analysis ===")
    exercise_2()
    print("\n=== Exercise 3: Poiseuille Flow Comparison ===")
    exercise_3()
    print("\n=== Exercise 4: Boundary Layer Scaling ===")
    exercise_4()
    print("\nAll exercises completed!")
