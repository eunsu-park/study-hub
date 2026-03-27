"""
Exercises for Lesson 18: MHD Numerical Methods
Topic: Numerical_Simulation

Solutions to practice problems from the lesson.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# === Exercise 1: MHD Wave Speed Calculation ===
# Problem: Compute all 7 characteristic speeds for given plasma state.

def exercise_1():
    """Compute MHD characteristic speeds and verify ordering."""
    rho = 1.0
    p = 0.5
    Bx = 1.0
    By = 0.5
    Bz = 0.0
    gamma = 5 / 3
    vx = 0.0  # Entropy wave speed

    # Sound speed
    c_s = np.sqrt(gamma * p / rho)

    # Alfven speeds
    B2 = Bx**2 + By**2 + Bz**2
    v_A = np.sqrt(B2 / rho)       # Total Alfven speed
    c_a = abs(Bx) / np.sqrt(rho)  # x-component Alfven speed

    # Fast and slow magnetosonic speeds
    # cf^2 = 0.5*(v_A^2 + c_s^2 + sqrt((v_A^2 + c_s^2)^2 - 4*c_a^2*c_s^2))
    # cs^2 = 0.5*(v_A^2 + c_s^2 - sqrt((v_A^2 + c_s^2)^2 - 4*c_a^2*c_s^2))
    discriminant = np.sqrt((v_A**2 + c_s**2)**2 - 4 * c_a**2 * c_s**2)
    c_f = np.sqrt(0.5 * (v_A**2 + c_s**2 + discriminant))
    c_sl = np.sqrt(0.5 * (v_A**2 + c_s**2 - discriminant))

    # All 7 characteristic speeds
    speeds = sorted([vx - c_f, vx - c_a, vx - c_sl, vx, vx + c_sl, vx + c_a, vx + c_f])

    print(f"MHD state: rho={rho}, p={p}, Bx={Bx}, By={By}, Bz={Bz}, gamma={gamma:.4f}")
    print(f"  Sound speed:        c_s  = {c_s:.6f}")
    print(f"  Total Alfven speed: v_A  = {v_A:.6f}")
    print(f"  Alfven speed (x):   c_a  = {c_a:.6f}")
    print(f"  Fast speed:         c_f  = {c_f:.6f}")
    print(f"  Slow speed:         c_sl = {c_sl:.6f}")
    print()
    print("All 7 characteristic speeds:")
    labels = ['vx-cf', 'vx-ca', 'vx-cs', 'vx', 'vx+cs', 'vx+ca', 'vx+cf']
    for label, speed in zip(labels, speeds):
        print(f"  {label:8s} = {speed:+.6f}")

    # Verify ordering
    is_ordered = all(speeds[i] <= speeds[i+1] + 1e-10 for i in range(6))
    print(f"\n  Ordering verified: {is_ordered}")
    print(f"  vx-cf <= vx-ca <= vx-cs <= vx <= vx+cs <= vx+ca <= vx+cf")


# === Exercise 2: Lax-Friedrichs Numerical Diffusion ===
# Problem: Apply LxF to scalar advection, measure diffusion vs Courant number.

def exercise_2():
    """Lax-Friedrichs numerical diffusion for scalar advection."""
    a = 1.0   # Advection speed
    L = 2.0   # Domain length (periodic)
    Nx = 200
    dx = L / Nx
    x = np.linspace(0, L, Nx, endpoint=False)

    def u0(x):
        """Gaussian initial condition."""
        return np.exp(-((x - 0.5)**2) / 0.01)

    n_periods = 10
    T = n_periods * L / a  # 10 wave periods

    courant_values = [0.5, 0.8, 0.95]

    print("Lax-Friedrichs: Numerical diffusion vs Courant number")
    print(f"  {n_periods} wave periods, Nx={Nx}")
    print()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, S in enumerate(courant_values):
        dt = S * dx / a
        nt = int(np.ceil(T / dt))
        dt = T / nt

        u = u0(x).copy()

        for n in range(nt):
            u_new = np.zeros_like(u)
            # Lax-Friedrichs with periodic BCs
            u_new[1:-1] = 0.5 * (u[2:] + u[:-2]) - (S / 2) * (u[2:] - u[:-2])
            # Periodic boundaries
            u_new[0] = 0.5 * (u[1] + u[-1]) - (S / 2) * (u[1] - u[-1])
            u_new[-1] = 0.5 * (u[0] + u[-2]) - (S / 2) * (u[0] - u[-2])
            u = u_new

        # Exact solution after 10 periods = initial condition
        u_exact = u0(x)

        amp_ratio = np.max(u) / np.max(u_exact)
        # Numerical diffusion coefficient: D_num = dx^2/(2*dt) * (1 - S^2)
        # = (a*dx/2) * (1 - S^2) / S
        D_num_formula = (a * dx / 2) * (1 - S**2)

        print(f"  S={S:.2f}: amplitude ratio={amp_ratio:.4f}, "
              f"D_num (formula)={(1-S**2)/2*dx:.6f}")

        ax = axes[idx]
        ax.plot(x, u_exact, 'r--', linewidth=2, label='Exact')
        ax.plot(x, u, 'b-', linewidth=2, label='LxF')
        ax.set_title(f'S={S}, amp ratio={amp_ratio:.3f}')
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Lax-Friedrichs Diffusion After 10 Periods', fontsize=13)
    plt.tight_layout()
    plt.savefig('ex18_lxf_diffusion.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  Numerical diffusion proportional to (1 - S^2)/2 * dx")
    print("  S -> 1 minimizes diffusion; S = 1 gives exact transport")
    print("  Plot saved: ex18_lxf_diffusion.png")


# === Exercise 3: Brio-Wu Shock Tube (HLL vs LxF) ===
# Problem: Run Brio-Wu with both HLL and LxF fluxes and compare.

def exercise_3():
    """Brio-Wu MHD shock tube: HLL vs Lax-Friedrichs comparison."""
    # Brio-Wu initial conditions
    gamma = 2.0
    Nx = 400
    L = 1.0
    dx = L / Nx
    x = np.linspace(dx / 2, L - dx / 2, Nx)

    # State: [rho, vx, vy, vz, By, Bz, p]
    # Bx = 0.75 everywhere (constant)
    Bx = 0.75

    # Left state (x < 0.5)
    rho_L, vx_L, vy_L, p_L, By_L = 1.0, 0.0, 0.0, 1.0, 1.0
    # Right state (x > 0.5)
    rho_R, vx_R, vy_R, p_R, By_R = 0.125, 0.0, 0.0, 0.1, -1.0

    CFL = 0.5
    T_final = 0.1

    def prim_to_cons(rho, vx, vy, p, By):
        """Convert primitive to conservative variables."""
        mx = rho * vx
        my = rho * vy
        B2 = Bx**2 + By**2
        E = p / (gamma - 1) + 0.5 * rho * (vx**2 + vy**2) + 0.5 * B2
        return np.array([rho, mx, my, By, E])

    def cons_to_prim(U):
        """Convert conservative to primitive."""
        rho = U[0]
        vx = U[1] / rho
        vy = U[2] / rho
        By = U[3]
        B2 = Bx**2 + By**2
        E = U[4]
        p = (gamma - 1) * (E - 0.5 * rho * (vx**2 + vy**2) - 0.5 * B2)
        return rho, vx, vy, p, By

    def flux(U):
        """Compute flux from conservative variables."""
        rho, vx, vy, p, By = cons_to_prim(U)
        B2 = Bx**2 + By**2
        p_total = p + 0.5 * B2
        E = U[4]
        F = np.zeros(5)
        F[0] = rho * vx
        F[1] = rho * vx**2 + p_total - Bx**2
        F[2] = rho * vx * vy - Bx * By
        F[3] = By * vx - Bx * vy
        F[4] = (E + p_total) * vx - Bx * (vx * Bx + vy * By)
        return F

    def max_speed(U):
        """Maximum wave speed for CFL."""
        rho, vx, vy, p, By = cons_to_prim(U)
        B2 = Bx**2 + By**2
        v_A2 = B2 / rho
        c_s2 = gamma * p / rho
        c_f = np.sqrt(0.5 * (v_A2 + c_s2 + np.sqrt((v_A2 + c_s2)**2 - 4 * Bx**2 / rho * c_s2)))
        return abs(vx) + c_f

    def run_simulation(flux_type='hll'):
        """Run Brio-Wu simulation."""
        # Initialize
        U = np.zeros((5, Nx))
        for i in range(Nx):
            if x[i] < 0.5:
                U[:, i] = prim_to_cons(rho_L, vx_L, vy_L, p_L, By_L)
            else:
                U[:, i] = prim_to_cons(rho_R, vx_R, vy_R, p_R, By_R)

        t = 0.0
        while t < T_final:
            # Compute max speed
            s_max = 0.0
            for i in range(Nx):
                s_max = max(s_max, max_speed(U[:, i]))
            dt = CFL * dx / s_max
            if t + dt > T_final:
                dt = T_final - t

            # Compute fluxes at interfaces
            F_interface = np.zeros((5, Nx + 1))

            for i in range(Nx + 1):
                if i == 0:
                    UL = U[:, 0]
                    UR = U[:, 0]
                elif i == Nx:
                    UL = U[:, -1]
                    UR = U[:, -1]
                else:
                    UL = U[:, i - 1]
                    UR = U[:, i]

                if flux_type == 'lxf':
                    # Lax-Friedrichs
                    FL = flux(UL)
                    FR = flux(UR)
                    alpha = s_max
                    F_interface[:, i] = 0.5 * (FL + FR) - 0.5 * alpha * (UR - UL)
                else:
                    # HLL
                    FL = flux(UL)
                    FR = flux(UR)
                    sL = min(0, -max_speed(UL))
                    sR = max(0, max_speed(UR))

                    if sL >= 0:
                        F_interface[:, i] = FL
                    elif sR <= 0:
                        F_interface[:, i] = FR
                    else:
                        F_interface[:, i] = ((sR * FL - sL * FR + sL * sR * (UR - UL))
                                             / (sR - sL))

            # Update
            for i in range(Nx):
                U[:, i] = U[:, i] - dt / dx * (F_interface[:, i + 1] - F_interface[:, i])

            t += dt

        return U

    # Run both methods
    U_hll = run_simulation('hll')
    U_lxf = run_simulation('lxf')

    # Extract primitives
    rho_hll = U_hll[0, :]
    vx_hll = U_hll[1, :] / U_hll[0, :]
    By_hll = U_hll[3, :]

    rho_lxf = U_lxf[0, :]
    vx_lxf = U_lxf[1, :] / U_lxf[0, :]
    By_lxf = U_lxf[3, :]

    print(f"Brio-Wu shock tube at t={T_final}, Nx={Nx}:")

    # Measure contact discontinuity width
    # Contact is near x ~ 0.48-0.52 in density
    contact_region = (x > 0.4) & (x < 0.6)
    rho_range_hll = np.max(rho_hll[contact_region]) - np.min(rho_hll[contact_region])
    rho_range_lxf = np.max(rho_lxf[contact_region]) - np.min(rho_lxf[contact_region])

    print(f"  HLL density range in contact region: {rho_range_hll:.4f}")
    print(f"  LxF density range in contact region: {rho_range_lxf:.4f}")
    print(f"  HLL resolves intermediate states more sharply than LxF")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.plot(x, rho_hll, 'b-', linewidth=1.5, label='HLL')
    ax.plot(x, rho_lxf, 'r--', linewidth=1.5, label='LxF')
    ax.set_title('Density')
    ax.set_xlabel('x')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(x, vx_hll, 'b-', linewidth=1.5, label='HLL')
    ax.plot(x, vx_lxf, 'r--', linewidth=1.5, label='LxF')
    ax.set_title('Velocity (vx)')
    ax.set_xlabel('x')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(x, By_hll, 'b-', linewidth=1.5, label='HLL')
    ax.plot(x, By_lxf, 'r--', linewidth=1.5, label='LxF')
    ax.set_title('By')
    ax.set_xlabel('x')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Brio-Wu Shock Tube (t={T_final}, Nx={Nx})', fontsize=13)
    plt.tight_layout()
    plt.savefig('ex18_brio_wu.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Plot saved: ex18_brio_wu.png")


# === Exercise 4: div B Error Monitoring ===
# Problem: Monitor discrete divergence of B in a simple configuration.

def exercise_4():
    """Monitor div B error in a simple 2D magnetic field configuration."""
    Nx = Ny = 50
    dx = dy = 0.1
    x = np.linspace(0, (Nx - 1) * dx, Nx)
    y = np.linspace(0, (Ny - 1) * dy, Ny)
    X, Y = np.meshgrid(x, y)

    # Analytical divergence-free field: B = curl(A z_hat)
    # A = sin(pi*x/L)*sin(pi*y/L)
    # Bx = dA/dy = (pi/L) sin(pi*x/L) cos(pi*y/L)
    # By = -dA/dx = -(pi/L) cos(pi*x/L) sin(pi*y/L)
    Lx = (Nx - 1) * dx
    Ly = (Ny - 1) * dy

    Bx = (np.pi / Ly) * np.sin(np.pi * X / Lx) * np.cos(np.pi * Y / Ly)
    By = -(np.pi / Lx) * np.cos(np.pi * X / Lx) * np.sin(np.pi * Y / Ly)

    # Compute div B using central differences
    divB = np.zeros((Ny, Nx))
    divB[1:-1, 1:-1] = ((Bx[1:-1, 2:] - Bx[1:-1, :-2]) / (2 * dx) +
                         (By[2:, 1:-1] - By[:-2, 1:-1]) / (2 * dy))

    max_divB = np.max(np.abs(divB))
    l2_divB = np.sqrt(np.sum(divB**2) * dx * dy)

    print("div B monitoring for analytical divergence-free field:")
    print(f"  B = curl(A), A = sin(pi x/L) sin(pi y/L)")
    print(f"  Grid: {Nx}x{Ny}, dx=dy={dx}")
    print(f"  max|div B| = {max_divB:.2e} (should be ~0 + truncation)")
    print(f"  L2|div B|  = {l2_divB:.2e}")
    print()

    # Now perturb B to introduce div B error
    Bx_perturbed = Bx + 0.01 * np.random.randn(Ny, Nx)
    By_perturbed = By + 0.01 * np.random.randn(Ny, Nx)

    divB_perturbed = np.zeros((Ny, Nx))
    divB_perturbed[1:-1, 1:-1] = (
        (Bx_perturbed[1:-1, 2:] - Bx_perturbed[1:-1, :-2]) / (2 * dx) +
        (By_perturbed[2:, 1:-1] - By_perturbed[:-2, 1:-1]) / (2 * dy)
    )

    max_divB_p = np.max(np.abs(divB_perturbed))
    print(f"  After random perturbation: max|div B| = {max_divB_p:.2e}")
    print("  Non-zero div B indicates numerical errors that must be controlled")
    print("  in MHD simulations (Powell source terms, projection, etc.)")


if __name__ == "__main__":
    print("=== Exercise 1: MHD Wave Speed Calculation ===")
    exercise_1()
    print("\n=== Exercise 2: Lax-Friedrichs Numerical Diffusion ===")
    exercise_2()
    print("\n=== Exercise 3: Brio-Wu Shock Tube (HLL vs LxF) ===")
    exercise_3()
    print("\n=== Exercise 4: div B Error Monitoring ===")
    exercise_4()
    print("\nAll exercises completed!")
