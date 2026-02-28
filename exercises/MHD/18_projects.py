"""
Lesson 18: MHD Projects
Topic: MHD
Description: Exercises integrating knowledge from the full MHD course.
             Harris sheet equilibrium verification, safety factor and
             Lundquist number scaling, tearing mode stability maps,
             dynamo number and cycle period, and CME initiation model.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import j0, j1
from scipy.integrate import solve_ivp


def exercise_1():
    """Harris Sheet Equilibrium Verification.

    Compute the total pressure p_tot(y) = p(y) + B^2(y)/2 for a Harris
    current sheet. Verify that p_tot is constant (force balance).
    """
    # Harris sheet parameters
    B_0 = 1.0            # asymptotic magnetic field
    delta = 0.5           # current sheet half-width
    beta_plasma = 1.0     # plasma beta at center (determines p_0)

    # Magnetic field profile: B_x(y) = B_0 * tanh(y / delta)
    N = 500
    y = np.linspace(-3 * delta, 3 * delta, N)

    Bx = B_0 * np.tanh(y / delta)

    # Magnetic pressure: p_mag = B^2 / 2 (normalized, mu_0 = 1)
    p_mag = Bx**2 / 2.0

    # For pressure balance: p(y) + B_x^2(y)/2 = const = p_0 + 0
    # where p_0 is the gas pressure at y = 0 (where B = 0)
    # => p(y) = p_0 * (1 - tanh^2(y/delta)) = p_0 / cosh^2(y/delta)
    # with p_0 = B_0^2 / 2 (to balance the magnetic pressure at y -> infinity)
    p_0 = B_0**2 / 2.0   # gas pressure at center
    p_gas = p_0 / np.cosh(y / delta)**2

    # Total pressure
    p_tot = p_gas + p_mag

    # Expected: p_tot = p_0 = B_0^2 / 2 everywhere
    p_tot_mean = np.mean(p_tot)
    max_deviation = np.max(np.abs(p_tot - p_tot_mean)) / p_tot_mean

    print(f"  Harris current sheet parameters:")
    print(f"    B_0 = {B_0}, delta = {delta}")
    print(f"    B_x(y) = B_0 * tanh(y/delta)")
    print(f"    p(y) = (B_0^2/2) / cosh^2(y/delta)")
    print()
    print(f"  Total pressure p_tot = p + B^2/2:")
    print(f"    <p_tot> = {p_tot_mean:.6f}")
    print(f"    Expected: B_0^2/2 = {B_0**2 / 2:.6f}")
    print(f"    Max relative deviation: {max_deviation:.2e}")

    if max_deviation < 1e-10:
        print(f"  VERIFIED: p_tot is constant to machine precision!")
    else:
        print(f"  Warning: p_tot deviation {max_deviation:.2e}")

    print()
    print(f"  Physical explanation:")
    print(f"  Inside the current sheet (y ~ 0): B ~ 0, p = p_0 (high gas pressure)")
    print(f"  Outside the sheet (|y| >> delta): B ~ B_0, p ~ 0 (magnetic pressure dominated)")
    print(f"  The gas pressure enhancement inside the sheet exactly compensates")
    print(f"  the magnetic pressure reduction, maintaining force balance.")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(y / delta, p_gas, 'r-', linewidth=2, label=r'$p_{gas}(y)$')
    ax1.plot(y / delta, p_mag, 'b-', linewidth=2, label=r'$B^2/2$')
    ax1.plot(y / delta, p_tot, 'k--', linewidth=2, label=r'$p_{tot}$')
    ax1.set_xlabel(r'$y / \delta$', fontsize=12)
    ax1.set_ylabel('Pressure', fontsize=12)
    ax1.set_title('Harris Sheet Pressure Balance', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Current density: J_z = dB_x/dy = B_0/(delta * cosh^2(y/delta))
    Jz = B_0 / (delta * np.cosh(y / delta)**2)
    ax2.plot(y / delta, Bx, 'b-', linewidth=2, label=r'$B_x$')
    ax2.plot(y / delta, Jz / np.max(Jz), 'r--', linewidth=2, label=r'$J_z$ (normalized)')
    ax2.set_xlabel(r'$y / \delta$', fontsize=12)
    ax2.set_ylabel('Field / Current', fontsize=12)
    ax2.set_title('Harris Sheet: Field and Current', fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('18_harris_equilibrium.png', dpi=150)
    plt.close()
    print("  Plot saved to 18_harris_equilibrium.png")


def exercise_2():
    """Safety Factor and Lundquist Number Scaling.

    1. Compute q(r) for uniform current profile.
    2. Compute Lundquist number S and Sweet-Parker reconnection rate.
    """
    # Part 1: Safety factor for uniform current
    # q(r) = r * B_phi / (R_0 * B_theta(r))
    # For uniform j_phi: B_theta(r) = mu_0 * j * r / 2
    # j = I_p / (pi * a^2)
    # B_theta(r) = mu_0 * I_p * r / (2 * pi * a^2)
    # q(r) = r * B_phi / (R_0 * mu_0 * I_p * r / (2*pi*a^2))
    #       = 2*pi*a^2*B_phi / (mu_0*R_0*I_p) = q_a = const

    R_0 = 3.0            # m (major radius)
    a = 1.0              # m (minor radius)
    B_phi = 5.0          # T (toroidal field)
    I_p = 5e6            # A
    mu_0 = 4 * np.pi * 1e-7

    q_a = 2 * np.pi * a**2 * B_phi / (mu_0 * R_0 * I_p)

    print(f"  Part 1: Safety Factor for Uniform Current")
    print(f"  -------------------------------------------")
    print(f"  R_0 = {R_0} m, a = {a} m, B_phi = {B_phi} T, I_p = {I_p / 1e6:.0f} MA")
    print(f"  Analytic: q(r) = 2*pi*a^2*B_phi / (mu_0*R_0*I_p) = {q_a:.3f}")
    print(f"  For uniform current, q(r) = q_a = const everywhere!")
    print(f"  (The toroidal winding of field lines is the same at all radii)")
    print()

    # Part 2: Lundquist number and Sweet-Parker reconnection
    print(f"  Part 2: Lundquist Number and Sweet-Parker Reconnection")
    print(f"  -------------------------------------------------------")

    # Reconnection parameters (from Project 1 in the lesson)
    B_0_rec = 1.0         # normalized field
    rho_0 = 1.0           # normalized density
    a_rec = 0.5           # current sheet half-width
    v_A = B_0_rec / np.sqrt(rho_0)  # Alfven speed

    eta_values = [0.01, 0.001, 0.0001]

    print(f"  v_A = B_0/sqrt(rho_0) = {v_A:.1f}, a = {a_rec}")
    print()
    print(f"  {'eta':>8s}  {'S':>10s}  {'M_A^SP':>10s}  {'delta^SP':>12s}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*12}")

    S_values = []
    for eta in eta_values:
        S = a_rec * v_A / eta
        M_A_SP = S**(-0.5)  # Sweet-Parker reconnection rate
        delta_SP = a_rec * S**(-0.5)  # current sheet thickness

        S_values.append(S)
        print(f"  {eta:8.4f}  {S:10.1f}  {M_A_SP:10.4f}  {delta_SP:12.6f}")

    print()
    print(f"  At S > 10^4, the plasmoid instability becomes important:")
    print(f"  - The Sweet-Parker current sheet becomes unstable to tearing")
    print(f"  - Multiple magnetic islands (plasmoids) form along the sheet")
    print(f"  - Reconnection rate becomes nearly independent of S: M_A ~ 0.01")
    print(f"  - This resolves the 'slow reconnection' problem of Sweet-Parker theory")

    for S, eta in zip(S_values, eta_values):
        if S > 1e4:
            print(f"  eta = {eta}: S = {S:.0e} > 10^4 => plasmoid instability active!")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    S_range = np.logspace(1, 8, 200)
    M_SP = S_range**(-0.5)
    M_plasmoid = np.where(S_range > 1e4, 0.01 * np.ones_like(S_range), M_SP)

    ax.loglog(S_range, M_SP, 'b-', linewidth=2, label='Sweet-Parker: $M_A = S^{-1/2}$')
    ax.loglog(S_range[S_range > 1e4], 0.01 * np.ones(np.sum(S_range > 1e4)),
              'r-', linewidth=2, label='Plasmoid regime: $M_A \\sim 0.01$')
    ax.axvline(1e4, color='gray', linestyle=':', alpha=0.7, label='$S \\sim 10^4$ (transition)')

    for S_val, eta_val in zip(S_values, eta_values):
        ax.plot(S_val, S_val**(-0.5), 'ko', markersize=8)
        ax.annotate(f'$\\eta$ = {eta_val}', (S_val, S_val**(-0.5)),
                    textcoords="offset points", xytext=(10, 10), fontsize=9)

    ax.set_xlabel('Lundquist Number S', fontsize=12)
    ax.set_ylabel('Reconnection Rate $M_A$', fontsize=12)
    ax.set_title('Reconnection Rate vs Lundquist Number', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('18_lundquist_scaling.png', dpi=150)
    plt.close()
    print("  Plot saved to 18_lundquist_scaling.png")


def exercise_3():
    """Tearing Mode Stability Map.

    Scan over current peaking index nu and q_0 to build a 2D stability map.
    """
    # Current profile: j(r) = j_0 * (1 - (r/a)^2)^nu
    # Safety factor: q(r) depends on the integrated current inside r

    a = 1.0              # minor radius
    R_0 = 3.0            # major radius
    B_phi = 5.0          # T
    mu_0 = 4 * np.pi * 1e-7

    nu_values = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
    q0_values = np.array([1.2, 1.5, 2.0, 3.0])

    N_r = 200
    r = np.linspace(0.001, a, N_r)
    dr = r[1] - r[0]

    # Stability map: q_min and Delta' sign
    stability_map = np.zeros((len(nu_values), len(q0_values)))
    q_min_map = np.zeros_like(stability_map)
    r_q2_map = np.zeros_like(stability_map)

    for i, nu in enumerate(nu_values):
        for j_idx, q0_target in enumerate(q0_values):
            # Current profile: j(r) = j_0 * (1 - (r/a)^2)^nu
            # Integrate to get B_theta(r):
            # B_theta(r) = mu_0/(r) * int_0^r j(r')*r' dr'
            j_profile = (1 - (r / a)**2)**nu

            # Integrated current inside radius r: I(r) = 2*pi * int_0^r j(r')*r' dr'
            I_r = np.cumsum(j_profile * r * dr) * 2 * np.pi

            # B_theta(r) = mu_0 * I(r) / (2*pi*r)
            B_theta = mu_0 * I_r / (2 * np.pi * r)

            # Total current: I_p = I(a)
            I_p_total = I_r[-1]

            # Safety factor: q(r) = r * B_phi / (R_0 * B_theta(r))
            q_profile = r * B_phi / (R_0 * B_theta)

            # Scale j_0 so that q(0) = q0_target
            # q(r->0) ~ r * B_phi / (R_0 * mu_0 * j_0 * r / 2) = 2*B_phi / (R_0*mu_0*j_0)
            # For the profile: q_0_computed = q_profile near r=0 (first few points average)
            q_0_computed = np.mean(q_profile[:5])

            # Rescale: new j_0 such that q_0 = q0_target
            # q_0 scales inversely with j_0, so:
            scale = q_0_computed / q0_target
            q_profile_scaled = q_profile / scale

            # Find q_min
            q_min = np.min(q_profile_scaled)
            q_min_map[i, j_idx] = q_min

            # Find q=2 surface
            idx_q2 = np.argmin(np.abs(q_profile_scaled - 2.0))
            r_q2 = r[idx_q2] if q_profile_scaled[idx_q2] < 2.5 else np.nan
            r_q2_map[i, j_idx] = r_q2

            # Stability criterion:
            # - Kruskal-Shafranov: q_min > 1 (stable to internal kink)
            # - Tearing: depends on Delta' (current gradient at rational surface)
            # Simplified: unstable if q has a rational surface (q=2)
            # inside the plasma AND current gradient is significant there

            if q_min < 1.0:
                stability_map[i, j_idx] = -1  # kink unstable
            elif not np.isnan(r_q2) and r_q2 < 0.9 * a:
                # Check current gradient at q=2 surface
                if nu > 0.5 and q0_target < 2.5:
                    stability_map[i, j_idx] = 0  # tearing unstable (marginal)
                else:
                    stability_map[i, j_idx] = 1  # stable
            else:
                stability_map[i, j_idx] = 1  # stable (no rational surface)

    # Print results
    print(f"  Tearing Mode Stability Map:")
    print(f"  {'':>6s}", end='')
    for q0 in q0_values:
        print(f"  q0={q0:4.1f}", end='')
    print()
    print(f"  {'nu':>6s}", end='')
    for _ in q0_values:
        print(f"  {'-----':>8s}", end='')
    print()

    for i, nu in enumerate(nu_values):
        print(f"  {nu:6.1f}", end='')
        for j_idx in range(len(q0_values)):
            s = stability_map[i, j_idx]
            if s == -1:
                label = "KINK"
            elif s == 0:
                label = "TEAR"
            else:
                label = "STABLE"
            print(f"  {label:>8s}", end='')
        print()

    print()
    print(f"  Legend: KINK = kink unstable (q_min < 1)")
    print(f"          TEAR = tearing unstable (q=2 rational surface)")
    print(f"          STABLE = MHD stable")
    print()
    print(f"  Most disruption-prone: high nu (peaked current) + low q_0")
    print(f"  Peaked current creates large gradients at rational surfaces.")

    # Plot stability map
    fig, ax = plt.subplots(figsize=(8, 6))
    # Color map: green=stable, yellow=marginal/tearing, red=kink unstable
    cmap = plt.cm.RdYlGn
    im = ax.pcolormesh(q0_values, nu_values, stability_map, cmap=cmap,
                        shading='nearest', vmin=-1, vmax=1)
    ax.set_xlabel('$q_0$ (central safety factor)', fontsize=12)
    ax.set_ylabel(r'$\nu$ (current peaking index)', fontsize=12)
    ax.set_title('Tearing Mode Stability Map', fontsize=13)

    # Add text labels
    for i, nu in enumerate(nu_values):
        for j_idx, q0 in enumerate(q0_values):
            s = stability_map[i, j_idx]
            label = "KINK" if s == -1 else ("TEAR" if s == 0 else "OK")
            color = 'white' if s == -1 else 'black'
            ax.text(q0, nu, label, ha='center', va='center', fontsize=10,
                    fontweight='bold', color=color)

    plt.colorbar(im, ax=ax, label='Stability (green=stable, red=unstable)')
    plt.tight_layout()
    plt.savefig('18_stability_map.png', dpi=150)
    plt.close()
    print("  Plot saved to 18_stability_map.png")


def exercise_4():
    """Dynamo Number and Cycle Period.

    Compute the dynamo number D = C_alpha * C_Omega and study how
    the magnetic field growth rate and cycle period depend on D.
    """
    # Mean-field dynamo equation (1D in r, simplified):
    # dA/dt = alpha*B + eta*D^2*A   (poloidal from toroidal via alpha-effect)
    # dB/dt = Omega'*dA/dr + eta*D^2*B  (toroidal from poloidal via Omega-effect)

    # Parameters from Project 3
    r_i = 0.5    # inner radius
    r_o = 1.0    # outer radius
    L = r_o - r_i
    eta = 1e-3
    Omega_0 = 10.0

    alpha_values = [0.5, 1.0, 2.0, 4.0]

    print(f"  Dynamo Number and Cycle Period Analysis:")
    print(f"  =========================================")
    print(f"  Parameters: L = {L}, eta = {eta}, Omega_0 = {Omega_0}")
    print()

    # Grid
    Nr = 50
    r = np.linspace(r_i, r_o, Nr)
    dr = r[1] - r[0]

    # Omega profile (differential rotation): Omega'(r) = -Omega_0 * sin(pi*(r-r_i)/L)
    Omega_prime = -Omega_0 * np.sin(np.pi * (r - r_i) / L)

    results = {}

    for alpha_0 in alpha_values:
        # Dynamo number
        C_alpha = alpha_0 * L / eta
        C_Omega = Omega_0 * L**2 / eta
        D = C_alpha * C_Omega

        # Alpha profile: alpha(r) = alpha_0 * sin(pi*(r-r_i)/L) * cos(pi*(r-r_i)/L)
        alpha_profile = alpha_0 * np.sin(np.pi * (r - r_i) / L)

        # Time integration (simplified 1D dynamo)
        dt = 0.5 * dr**2 / eta  # diffusion CFL
        dt = min(dt, 0.01)
        N_steps = 80000
        N_save = 100

        A = 1e-4 * np.sin(np.pi * (r - r_i) / L)  # seed
        B = np.zeros(Nr)

        E_history = []
        t_history = []

        for step in range(N_steps):
            # Diffusion
            d2A = np.zeros(Nr)
            d2B = np.zeros(Nr)
            dA_dr = np.zeros(Nr)
            for k in range(1, Nr - 1):
                d2A[k] = (A[k + 1] - 2 * A[k] + A[k - 1]) / dr**2
                d2B[k] = (B[k + 1] - 2 * B[k] + B[k - 1]) / dr**2
                dA_dr[k] = (A[k + 1] - A[k - 1]) / (2 * dr)

            # Source terms
            source_A = alpha_profile * B
            source_B = Omega_prime * dA_dr

            # Update
            A += dt * (eta * d2A + source_A)
            B += dt * (eta * d2B + source_B)

            # Boundary conditions
            A[0] = A[-1] = 0
            B[0] = B[-1] = 0

            # Quench if needed
            B_max = np.max(np.abs(B))
            A_max = np.max(np.abs(A))
            field_max = max(B_max, A_max)
            if field_max > 100:
                A *= 100 / field_max
                B *= 100 / field_max

            # Record energy
            if step % N_save == 0:
                E = np.sum(A**2 + B**2) * dr
                E_history.append(E)
                t_history.append(step * dt)

        E_arr = np.array(E_history)
        t_arr = np.array(t_history)

        # Determine growth rate (fit exponential to early phase)
        if len(E_arr) > 10 and E_arr[-1] > E_arr[0]:
            # Find growth rate from log(E) slope
            idx_start = len(E_arr) // 4
            idx_end = 3 * len(E_arr) // 4
            log_E = np.log(E_arr[idx_start:idx_end] + 1e-30)
            t_fit = t_arr[idx_start:idx_end]
            if len(t_fit) > 2:
                coeffs = np.polyfit(t_fit, log_E, 1)
                gamma = coeffs[0]
            else:
                gamma = 0
        else:
            gamma = 0

        # Determine cycle period (if oscillating)
        if len(E_arr) > 20:
            # Look for oscillations in the second half
            E_half = E_arr[len(E_arr) // 2:]
            t_half = t_arr[len(E_arr) // 2:]
            # Find peaks
            peaks = []
            for k in range(1, len(E_half) - 1):
                if E_half[k] > E_half[k - 1] and E_half[k] > E_half[k + 1]:
                    peaks.append(t_half[k])
            if len(peaks) >= 2:
                periods = np.diff(peaks)
                T_cycle = np.mean(periods)
            else:
                T_cycle = np.nan
        else:
            T_cycle = np.nan

        results[alpha_0] = {
            'D': D, 'gamma': gamma, 'T_cycle': T_cycle,
            'E': E_arr, 't': t_arr
        }

        status = "growing" if gamma > 0.01 else ("decaying" if gamma < -0.01 else "marginal")
        period_str = f"{T_cycle:.2f}" if not np.isnan(T_cycle) else "N/A"
        print(f"  alpha_0 = {alpha_0:.1f}: D = {D:.0f}, gamma = {gamma:.4f} ({status}), "
              f"T_cycle = {period_str}")

    # Find critical D
    D_values = [results[a]['D'] for a in alpha_values]
    gamma_values = [results[a]['gamma'] for a in alpha_values]

    # Interpolate to find D_c (where gamma = 0)
    for k in range(len(gamma_values) - 1):
        if gamma_values[k] * gamma_values[k + 1] < 0:
            # Linear interpolation
            D_c = D_values[k] - gamma_values[k] * (D_values[k + 1] - D_values[k]) / (gamma_values[k + 1] - gamma_values[k])
            print(f"\n  Critical dynamo number D_c ~ {D_c:.0f}")
            break
    else:
        if all(g > 0 for g in gamma_values):
            print(f"\n  All dynamo numbers produce growth; D_c < {min(D_values):.0f}")
        else:
            print(f"\n  Could not determine D_c from this scan")

    print()
    print(f"  Physical explanation:")
    print(f"  Increasing alpha_0 strengthens the alpha-effect (poloidal -> toroidal coupling).")
    print(f"  This both increases the growth rate (more efficient field amplification)")
    print(f"  and shortens the cycle period (faster oscillation between poloidal and toroidal).")
    print(f"  The cycle period T ~ 1/sqrt(D) for supercritical dynamos.")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for alpha_0 in alpha_values:
        res = results[alpha_0]
        ax1.semilogy(res['t'], res['E'] + 1e-30, linewidth=1.5,
                     label=f'$\\alpha_0$ = {alpha_0:.1f}, D = {res["D"]:.0f}')

    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Magnetic Energy E(t)', fontsize=12)
    ax1.set_title('Dynamo Growth vs Dynamo Number', fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(D_values, gamma_values, 'bo-', markersize=8, linewidth=2)
    ax2.axhline(0, color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Dynamo Number D', fontsize=12)
    ax2.set_ylabel('Growth Rate $\\gamma$', fontsize=12)
    ax2.set_title('Growth Rate vs Dynamo Number', fontsize=13)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('18_dynamo_number.png', dpi=150)
    plt.close()
    print("  Plot saved to 18_dynamo_number.png")


def exercise_5():
    """CME Initiation Model.

    Create a simplified 2D flux rope equilibrium, compute the torus
    instability criterion, and analyze the eruption dynamics.
    """
    # Part 1: Equilibrium setup
    Nx, Ny = 200, 200
    Lx, Ly = 4.0, 4.0
    x = np.linspace(-Lx / 2, Lx / 2, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Background arcade field: B_y(x) = B_ext * tanh(x / L_arcade)
    B_ext = 1.0
    L_arcade = 0.5
    By_arcade = B_ext * np.tanh(X / L_arcade)
    Bx_arcade = np.zeros_like(X)

    # Flux rope: localized current loop at (x0, y0) with radius R_rope
    x0, y0 = 0.0, 1.0   # flux rope center
    R_rope = 0.5         # rope radius
    I_rope = 2.0         # rope current (determines field strength)

    # Flux rope field (2D cross-section of infinite wire, simplified):
    # Inside rope: B_theta = (mu_0 * I / (2*pi*R_rope^2)) * r  (uniform current density)
    # Outside: B_theta = mu_0 * I / (2*pi*r)
    R = np.sqrt((X - x0)**2 + (Y - y0)**2) + 1e-10

    # Vector potential for the rope: A_z = -(mu_0*I/(4*pi)) * [r^2/R_rope^2 - 1] inside
    #                                     = -(mu_0*I/(4*pi)) * ln(r/R_rope) outside
    # In normalized units (mu_0 = 4*pi):
    A_rope = np.where(R < R_rope,
                      -I_rope * (R**2 / R_rope**2 - 1),
                      -I_rope * np.log(R / R_rope))

    # B_x = -dA/dy, B_y = dA/dx (for the rope)
    Bx_rope = np.zeros_like(X)
    By_rope = np.zeros_like(X)
    Bx_rope[1:-1, :] = -(A_rope[2:, :] - A_rope[:-2, :]) / (2 * dy)
    By_rope[:, 1:-1] = (A_rope[:, 2:] - A_rope[:, :-2]) / (2 * dx)

    # Total field
    Bx_total = Bx_arcade + Bx_rope
    By_total = By_arcade + By_rope
    B_mag = np.sqrt(Bx_total**2 + By_total**2)

    print(f"  Part 1: Flux Rope Equilibrium")
    print(f"  =============================")
    print(f"  Arcade: B_ext = {B_ext}, L = {L_arcade}")
    print(f"  Rope: center = ({x0}, {y0}), R = {R_rope}, I = {I_rope}")
    print(f"  Max |B| = {np.max(B_mag):.2f}")
    print()

    # Part 2: Torus instability criterion
    # Decay index n = -d(ln B_ext) / d(ln h)
    # where B_ext is the external (arcade) field, h is height
    # n > n_crit ~ 1.5 => eruption

    h_range = np.linspace(0.3, 3.0, 200)
    # External field at x=0 as function of height:
    # B_ext(h) = B_ext * tanh(0/L) at x=0, which is 0. Use the |B| at (0, h).
    # Better: use the horizontal component at x = small offset
    x_offset = 0.1
    B_ext_profile = B_ext * np.abs(np.tanh(x_offset / L_arcade)) / np.cosh(0.5 * h_range)**2
    # More physical model: arcade field decays with height
    B_ext_height = B_ext * np.exp(-h_range / L_arcade)

    # Decay index
    n = np.zeros(len(h_range))
    for k in range(1, len(h_range) - 1):
        dln_B = (np.log(B_ext_height[k + 1] + 1e-20) - np.log(B_ext_height[k - 1] + 1e-20))
        dln_h = (np.log(h_range[k + 1]) - np.log(h_range[k - 1]))
        n[k] = -dln_B / dln_h

    n[0] = n[1]
    n[-1] = n[-2]

    # Critical height
    n_crit = 1.5
    idx_crit = np.argmin(np.abs(n - n_crit))
    h_crit = h_range[idx_crit] if n[idx_crit] > 0.5 else np.nan

    print(f"  Part 2: Torus Instability")
    print(f"  =========================")
    print(f"  Decay index n = -d(ln B_ext)/d(ln h)")
    print(f"  n_crit = {n_crit}")
    if not np.isnan(h_crit):
        print(f"  Critical height h_c = {h_crit:.2f}")
        print(f"  Rope center at h = {y0}: {'ABOVE h_c => UNSTABLE' if y0 > h_crit else 'BELOW h_c => STABLE'}")
    else:
        print(f"  Could not determine h_c from this model")
    print()

    # Part 3: Simplified eruption dynamics
    # Track rope height using force balance:
    # m * d^2h/dt^2 = F_hoop - F_tension - F_gravity(magnetic)
    # F_hoop ~ I^2 / (2*R_rope) (hoop force, upward)
    # F_tension ~ B_ext(h) * I (downward strapping by arcade)

    print(f"  Part 3: Eruption Dynamics (simplified)")
    print(f"  =======================================")

    m_rope = 1.0  # effective mass (normalized)
    perturbation = 0.1  # upward kick

    def rope_eom(t, state):
        h, v = state
        if h < 0.1:
            return [0, 0]
        # Hoop force (upward): F_hoop = I^2 / (2*h) (decreases with height)
        F_hoop = I_rope**2 / (2 * h)
        # Arcade tension (downward): F_tension = B_ext(h) * I
        B_arcade_h = B_ext * np.exp(-h / L_arcade)
        F_tension = B_arcade_h * I_rope
        # Gravity-like restoring force
        F_gravity = 0.5 * m_rope / h**2
        F_net = F_hoop - F_tension - F_gravity
        return [v, F_net / m_rope]

    # Integrate
    t_span = (0, 10)
    y0_ode = [y0 + perturbation, 0.0]  # perturbed position, zero velocity
    sol = solve_ivp(rope_eom, t_span, y0_ode, max_step=0.01, dense_output=True)

    h_t = sol.y[0]
    v_t = sol.y[1]
    t_t = sol.t

    # Determine if eruption occurs
    if h_t[-1] > 2 * y0:
        print(f"  Rope ERUPTS! Final height: {h_t[-1]:.2f} (initial: {y0 + perturbation:.2f})")
        print(f"  Maximum velocity: {np.max(v_t):.3f}")
    else:
        print(f"  Rope OSCILLATES and returns to equilibrium.")
        print(f"  Height range: [{np.min(h_t):.2f}, {np.max(h_t):.2f}]")

    print()
    print(f"  Part 4: Physical Interpretation")
    print(f"  ================================")
    print(f"  This model captures the basic CME initiation physics:")
    print(f"  - The flux rope carries current that creates an upward hoop force")
    print(f"  - The overlying arcade field provides a downward tension (strapping)")
    print(f"  - If the arcade field decays fast enough (n > n_crit), the rope erupts")
    print(f"  - Reconnection below the rope reduces the strapping field further")
    print(f"  - This creates a positive feedback loop (catastrophic eruption)")
    print(f"  - Energy: Delta_E_mag is converted to Delta_E_kin (CME kinetic energy)")
    print(f"  - This is consistent with the CSHKP flare model:")
    print(f"    (Carmichael-Sturrock-Hirayama-Kopp-Pneuman)")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Magnetic field
    ax = axes[0, 0]
    ax.pcolormesh(X, Y, B_mag, cmap='inferno', shading='auto', vmin=0, vmax=3)
    # Streamlines (use coarser grid)
    skip = 5
    ax.streamplot(x[::skip], y[::skip], By_total[::skip, ::skip].T, Bx_total[::skip, ::skip].T,
                  color='white', linewidth=0.5, density=2)
    circle = plt.Circle((x0, y0), R_rope, fill=False, color='cyan', linewidth=2)
    ax.add_patch(circle)
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y (height)', fontsize=11)
    ax.set_title('Magnetic Field |B| with Flux Rope', fontsize=12)
    ax.set_aspect('equal')

    # Decay index
    ax = axes[0, 1]
    ax.plot(h_range, n, 'b-', linewidth=2)
    ax.axhline(n_crit, color='red', linestyle='--', label=f'$n_{{crit}}$ = {n_crit}')
    if not np.isnan(h_crit):
        ax.axvline(h_crit, color='green', linestyle=':', label=f'$h_c$ = {h_crit:.2f}')
    ax.axvline(y0, color='orange', linestyle='--', alpha=0.7, label=f'Rope height = {y0}')
    ax.set_xlabel('Height h', fontsize=11)
    ax.set_ylabel('Decay index n', fontsize=11)
    ax.set_title('Torus Instability Criterion', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Eruption dynamics: height vs time
    ax = axes[1, 0]
    ax.plot(t_t, h_t, 'b-', linewidth=2)
    ax.axhline(y0, color='orange', linestyle='--', alpha=0.7, label='Initial height')
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Rope height h(t)', fontsize=11)
    ax.set_title('Flux Rope Dynamics', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Velocity vs time
    ax = axes[1, 1]
    ax.plot(t_t, v_t, 'r-', linewidth=2)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Rope velocity v(t)', fontsize=11)
    ax.set_title('Rope Velocity', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.suptitle('CME Initiation Model', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('18_cme_initiation.png', dpi=150)
    plt.close()
    print("  Plot saved to 18_cme_initiation.png")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Harris Sheet Equilibrium Verification", exercise_1),
        ("Exercise 2: Safety Factor and Lundquist Number Scaling", exercise_2),
        ("Exercise 3: Tearing Mode Stability Map", exercise_3),
        ("Exercise 4: Dynamo Number and Cycle Period", exercise_4),
        ("Exercise 5: Integrated CME Initiation Model", exercise_5),
    ]
    for title, func in exercises:
        print(f"\n{'=' * 60}")
        print(f" {title}")
        print(f"{'=' * 60}")
        func()
