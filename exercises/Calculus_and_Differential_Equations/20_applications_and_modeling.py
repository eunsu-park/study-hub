"""
Exercise Solutions: Lesson 20 - Applications and Modeling
Calculus and Differential Equations

Topics covered:
- Modified logistic model with harvesting (bifurcation)
- Competing species phase portrait
- RLC circuit resonance
- Heat equation with source (steady-state)
- Diffusion time scale estimation
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# Problem 1: Logistic with Harvesting
# ============================================================
def exercise_1():
    """
    P' = r*P*(1 - P/K) - H. Find equilibria as function of H.
    Show collapse when H > rK/4.
    """
    print("=" * 60)
    print("Problem 1: Logistic with Harvesting (Bifurcation)")
    print("=" * 60)

    r, K = 1.0, 100.0

    # P' = r*P*(1 - P/K) - H
    # At equilibrium: r*P*(1-P/K) = H
    # r*P - r*P^2/K = H
    # P^2 - K*P + K*H/r = 0
    # P = (K +/- sqrt(K^2 - 4KH/r)) / 2

    print(f"\n  P' = r*P*(1 - P/K) - H")
    print(f"  r = {r}, K = {K}")
    print(f"\n  Equilibria: r*P*(1-P/K) = H")
    print(f"  P^2 - K*P + KH/r = 0")
    print(f"  P = (K +/- sqrt(K^2 - 4KH/r)) / 2")
    print(f"\n  Discriminant = K^2 - 4KH/r = K(K - 4H/r)")
    print(f"  Two equilibria when H < rK/4 = {r*K/4}")
    print(f"  One equilibrium (saddle-node) when H = rK/4 = {r*K/4}")
    print(f"  No equilibria when H > rK/4 => population COLLAPSES")

    H_critical = r * K / 4
    print(f"\n  Critical harvesting rate: H_c = rK/4 = {H_critical}")

    # Bifurcation diagram
    H_values = np.linspace(0, 30, 300)
    P_upper = []
    P_lower = []

    for H in H_values:
        disc = K**2 - 4*K*H/r
        if disc >= 0:
            P_upper.append((K + np.sqrt(disc)) / 2)
            P_lower.append((K - np.sqrt(disc)) / 2)
        else:
            P_upper.append(np.nan)
            P_lower.append(np.nan)

    # Simulation for various H
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Bifurcation diagram
    ax1.plot(H_values, P_upper, 'b-', linewidth=2, label='Stable equilibrium')
    ax1.plot(H_values, P_lower, 'r--', linewidth=2, label='Unstable equilibrium')
    ax1.axvline(x=H_critical, color='g', linestyle=':', linewidth=2, label=f'$H_c$ = {H_critical}')
    ax1.set_xlabel('Harvest rate H', fontsize=12)
    ax1.set_ylabel('Equilibrium P*', fontsize=12)
    ax1.set_title('Bifurcation Diagram', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Time series
    t_span = [0, 50]
    t_eval = np.linspace(0, 50, 1000)
    P0 = 80

    for H_val, color, style in [(10, 'b', '-'), (H_critical, 'g', '--'), (30, 'r', '-.')]:
        def ode(t, P, H=H_val):
            return [r*P[0]*(1 - P[0]/K) - H]
        sol = solve_ivp(ode, t_span, [P0], t_eval=t_eval, method='RK45')
        ax2.plot(sol.t, sol.y[0], color=color, linestyle=style, linewidth=2,
                 label=f'H = {H_val}')

    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Population P', fontsize=12)
    ax2.set_title('Population Dynamics', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex20_logistic_harvesting.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  [Plot saved: ex20_logistic_harvesting.png]")


# ============================================================
# Problem 2: Competing Species
# ============================================================
def exercise_2():
    """
    x' = x(3 - x - 2y), y' = y(2 - y - x).
    Equilibria, stability, phase portrait.
    """
    print("\n" + "=" * 60)
    print("Problem 2: Competing Species")
    print("=" * 60)

    # Equilibria: x(3-x-2y)=0, y(2-y-x)=0
    # (0,0), (3,0), (0,2), and intersection of 3-x-2y=0, 2-y-x=0
    # From 2-y-x=0: x = 2-y; substituting: 3-(2-y)-2y = 1-y = 0 => y=1, x=1
    print(f"\n  x' = x(3-x-2y), y' = y(2-y-x)")
    print(f"\n  Equilibria:")
    equilibria = [(0, 0), (3, 0), (0, 2), (1, 1)]
    for eq in equilibria:
        print(f"    ({eq[0]}, {eq[1]})")

    # Jacobian
    # J = [[3-2x-2y, -2x], [-y, 2-2y-x]]
    print(f"\n  Jacobian:")
    print(f"  J = [[3-2x-2y, -2x], [-y, 2-2y-x]]")

    for xp, yp in equilibria:
        J = np.array([[3 - 2*xp - 2*yp, -2*xp],
                       [-yp, 2 - 2*yp - xp]])
        eigenvals = np.linalg.eigvals(J)
        trace = J[0, 0] + J[1, 1]
        det = J[0, 0]*J[1, 1] - J[0, 1]*J[1, 0]

        if all(np.real(eigenvals) < 0):
            stability = "STABLE NODE"
        elif all(np.real(eigenvals) > 0):
            stability = "UNSTABLE NODE"
        elif np.real(eigenvals[0]) * np.real(eigenvals[1]) < 0:
            stability = "SADDLE"
        else:
            stability = "UNKNOWN"

        print(f"\n    ({xp}, {yp}): eigenvalues = {eigenvals}, {stability}")

    print(f"\n  The coexistence point (1,1) is a SADDLE -- unstable.")
    print(f"  Both (3,0) and (0,2) are stable nodes.")
    print(f"  Outcome depends on initial conditions (competitive exclusion).")

    # Phase portrait
    def system(t, state):
        x_v, y_v = state
        return [x_v*(3 - x_v - 2*y_v), y_v*(2 - y_v - x_v)]

    fig, ax = plt.subplots(figsize=(10, 8))

    ics = [(0.2, 0.2), (0.5, 0.5), (1.5, 0.5), (0.5, 1.5), (2.5, 0.5),
           (0.5, 2), (2, 1), (1, 1.5), (2.5, 1.5), (0.1, 1.8)]
    for x0, y0 in ics:
        sol = solve_ivp(system, [0, 30], [x0, y0],
                        t_eval=np.linspace(0, 30, 3000), method='RK45')
        ax.plot(sol.y[0], sol.y[1], 'b-', linewidth=0.8, alpha=0.6)
        ax.plot(x0, y0, 'go', markersize=4)

    # Nullclines
    x_null = np.linspace(0, 3.5, 200)
    ax.plot(x_null, (3 - x_null)/2, 'r--', linewidth=1.5, label='x-nullcline')
    ax.plot(x_null, 2 - x_null, 'g--', linewidth=1.5, label='y-nullcline')

    for xp, yp in equilibria:
        ax.plot(xp, yp, 'rs', markersize=12)

    ax.set_xlim(0, 3.5)
    ax.set_ylim(0, 2.5)
    ax.set_xlabel('x (species 1)', fontsize=12)
    ax.set_ylabel('y (species 2)', fontsize=12)
    ax.set_title('Competing Species Phase Portrait', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex20_competing_species.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  [Plot saved: ex20_competing_species.png]")


# ============================================================
# Problem 3: RLC Resonance
# ============================================================
def exercise_3():
    """
    L=1, R=0.1, C=0.04. Resonant frequency.
    Amplitude vs omega plot. Effect of R.
    """
    print("\n" + "=" * 60)
    print("Problem 3: RLC Circuit Resonance")
    print("=" * 60)

    L_val, R_val, C_val = 1.0, 0.1, 0.04

    # L*q'' + R*q' + q/C = E(t) = sin(omega*t)
    # Natural frequency: omega_0 = 1/sqrt(LC) = 1/sqrt(0.04) = 5
    omega_0 = 1 / np.sqrt(L_val * C_val)
    print(f"\n  L={L_val}, R={R_val}, C={C_val}")
    print(f"  LCq'' + RCq' + q = CE(t)")
    print(f"  omega_0 = 1/sqrt(LC) = {omega_0}")

    # Resonant frequency for current amplitude (same as omega_0 for series RLC)
    print(f"  Resonant frequency: omega_r = omega_0 = {omega_0} rad/s")
    print(f"  f_r = omega_0/(2*pi) = {omega_0/(2*np.pi):.4f} Hz")

    # Steady-state amplitude of charge:
    # |q| = C*E_0 / sqrt((1 - omega^2*LC)^2 + (omega*RC)^2)
    # Steady-state current amplitude:
    # |I| = omega*|q| = omega*C*E_0 / sqrt((1 - omega^2*LC)^2 + (omega*RC)^2)
    # Or equivalently: |I| = E_0 / sqrt(R^2 + (omega*L - 1/(omega*C))^2)
    E_0 = 1.0
    omega_range = np.linspace(0.5, 10, 1000)

    def current_amplitude(omega, R):
        Z = np.sqrt(R**2 + (omega*L_val - 1/(omega*C_val))**2)
        return E_0 / Z

    I_amp = current_amplitude(omega_range, R_val)
    I_max = E_0 / R_val  # at resonance: impedance = R
    print(f"\n  At resonance: |I|_max = E_0/R = {I_max}")

    # Effect of increasing R
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    R_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    for R in R_values:
        I_amp_R = current_amplitude(omega_range, R)
        ax1.plot(omega_range, I_amp_R, linewidth=2, label=f'R = {R}')
        ax2.plot(omega_range, I_amp_R, linewidth=2, label=f'R = {R}')

    ax1.axvline(x=omega_0, color='gray', linestyle='--', alpha=0.5, label=f'$\\omega_0$ = {omega_0}')
    ax1.set_xlabel(r'$\omega$ (rad/s)', fontsize=12)
    ax1.set_ylabel('Current amplitude |I|', fontsize=12)
    ax1.set_title('Resonance Curves', fontsize=14)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel(r'$\omega$ (rad/s)', fontsize=12)
    ax2.set_ylabel('Current amplitude |I|', fontsize=12)
    ax2.set_title('Effect of Resistance on Resonance', fontsize=14)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 3)

    plt.tight_layout()
    plt.savefig('ex20_rlc_resonance.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  Effect of increasing R:")
    print(f"  - Peak amplitude decreases (I_max = E_0/R)")
    print(f"  - Resonance peak broadens (lower Q factor)")
    print(f"  - Q = omega_0*L/R = {omega_0*L_val/R_val:.1f} (for R={R_val})")
    print("  [Plot saved: ex20_rlc_resonance.png]")


# ============================================================
# Problem 4: Heat Equation with Source
# ============================================================
def exercise_4():
    """
    u_t = 0.01*u_xx + sin(pi*x) on [0,1], u(0,t)=u(1,t)=0, u(x,0)=0.
    Find steady-state analytically.
    """
    print("\n" + "=" * 60)
    print("Problem 4: Heat Equation with Source")
    print("=" * 60)

    alpha = 0.01
    Nx = 100
    dx = 1.0 / Nx
    dt = 0.4 * dx**2 / alpha  # CFL-safe
    r = alpha * dt / dx**2

    x = np.linspace(0, 1, Nx + 1)
    u = np.zeros(Nx + 1)

    # Analytical steady-state: u_ss
    # Set u_t = 0: 0.01*u_xx = -sin(pi*x)
    # u_xx = -100*sin(pi*x)
    # u = 100/(pi^2) * sin(pi*x) (since d^2/dx^2[sin(pi*x)] = -pi^2*sin(pi*x))
    u_ss = 100 / (np.pi**2) * np.sin(np.pi * x)

    print(f"\n  u_t = {alpha}*u_xx + sin(pi*x), u(0,t)=u(1,t)=0, u(x,0)=0")
    print(f"\n  Steady-state: u_t = 0 => {alpha}*u_xx = -sin(pi*x)")
    print(f"  u_xx = -{1/alpha}*sin(pi*x)")
    print(f"  Since d^2/dx^2[sin(pi*x)] = -pi^2*sin(pi*x):")
    print(f"  u_ss(x) = {1/(alpha*np.pi**2):.6f} * sin(pi*x)")
    print(f"  u_ss(1/2) = {u_ss[Nx//2]:.6f}")

    # Numerical simulation (forward Euler)
    Nt = 50000
    source = np.sin(np.pi * x)
    u_history = [u.copy()]
    t_current = 0

    for n in range(Nt):
        u_new = u.copy()
        for i in range(1, Nx):
            u_new[i] = u[i] + r*(u[i+1] - 2*u[i] + u[i-1]) + dt*source[i]
        u_new[0] = 0
        u_new[Nx] = 0
        u = u_new
        t_current += dt
        if (n + 1) % 10000 == 0:
            u_history.append(u.copy())

    max_err = np.max(np.abs(u - u_ss))
    print(f"\n  Numerical simulation: {Nt} steps, t_final = {t_current:.2f}")
    print(f"  Max error vs steady-state: {max_err:.6e}")
    print(f"  u_numerical(1/2) = {u[Nx//2]:.6f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for i, uh in enumerate(u_history):
        alpha_val = 0.3 + 0.7*i/len(u_history)
        t_label = f't = {i*10000*dt:.1f}' if i < len(u_history)-1 else f't = {t_current:.1f} (final)'
        ax1.plot(x, uh, '-', alpha=alpha_val, linewidth=1.5, label=t_label)
    ax1.plot(x, u_ss, 'k--', linewidth=2, label='Steady-state (analytical)')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('u', fontsize=12)
    ax1.set_title('Temperature Evolution', fontsize=14)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(x, u, 'b-', linewidth=2, label='Numerical (final)')
    ax2.plot(x, u_ss, 'r--', linewidth=2, label='Analytical steady-state')
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('u', fontsize=12)
    ax2.set_title('Steady-State Comparison', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ex20_heat_with_source.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [Plot saved: ex20_heat_with_source.png]")


# ============================================================
# Problem 5: Diffusion Time Scale
# ============================================================
def exercise_5():
    """
    Pollutant at center of 1 km lake, D = 1e-5 m^2/s.
    Estimate time to spread 100 m using sigma ~ sqrt(2*D*t).
    Verify with 1D diffusion simulation.
    """
    print("\n" + "=" * 60)
    print("Problem 5: Diffusion Time Scale")
    print("=" * 60)

    D = 1e-5  # m^2/s
    sigma_target = 100  # m

    # sigma ~ sqrt(2*D*t) => t ~ sigma^2 / (2*D)
    t_est = sigma_target**2 / (2 * D)

    print(f"\n  Diffusion coefficient D = {D} m^2/s")
    print(f"  Target spread distance: {sigma_target} m")
    print(f"\n  Estimate: sigma ~ sqrt(2*D*t)")
    print(f"  t ~ sigma^2 / (2*D) = {sigma_target}^2 / (2*{D})")
    print(f"  t ~ {t_est:.0f} seconds")
    print(f"  t ~ {t_est/3600:.1f} hours")
    print(f"  t ~ {t_est/86400:.1f} days")
    print(f"  t ~ {t_est/(86400*365.25):.2f} years")

    # Numerical verification: 1D diffusion
    # u_t = D*u_xx, u(x,0) = delta(x) (initially concentrated at center)
    # Exact: u(x,t) = 1/sqrt(4*pi*D*t) * exp(-x^2/(4*D*t))

    L = 500  # half-domain (m)
    Nx = 500
    dx = 2*L / Nx
    dt = 0.4 * dx**2 / D
    x = np.linspace(-L, L, Nx + 1)

    # Initial condition: Gaussian with small sigma (approximating delta)
    sigma_0 = 5  # m
    u = (1/(sigma_0*np.sqrt(2*np.pi))) * np.exp(-x**2/(2*sigma_0**2))
    u[0] = 0
    u[-1] = 0

    # Find time when concentration at x=100 reaches 10% of peak
    target_x_idx = np.argmin(np.abs(x - sigma_target))
    peak_0 = np.max(u)
    threshold = 0.10 * peak_0

    t_current = 0
    t_threshold = None
    n_steps = int(t_est / dt) + 10000  # simulate a bit past estimate
    report_interval = max(n_steps // 10, 1)

    print(f"\n  Simulating 1D diffusion (dx={dx:.1f} m, dt={dt:.0f} s)...")
    print(f"  Initial peak concentration: {peak_0:.6e}")
    print(f"  10% threshold at x={sigma_target}m: {threshold:.6e}")

    for n in range(min(n_steps, 500000)):
        r = D * dt / dx**2
        u_new = u.copy()
        for i in range(1, Nx):
            u_new[i] = u[i] + r*(u[i+1] - 2*u[i] + u[i-1])
        u_new[0] = 0
        u_new[-1] = 0
        u = u_new
        t_current += dt

        if t_threshold is None and u[target_x_idx] >= threshold:
            t_threshold = t_current

        if (n + 1) % report_interval == 0:
            current_peak = np.max(u)
            u_at_100 = u[target_x_idx]
            # Don't print every step, just summarize

    if t_threshold is not None:
        print(f"\n  Numerical result:")
        print(f"    Concentration at x={sigma_target}m reaches 10% of initial peak at:")
        print(f"    t = {t_threshold:.0f} s = {t_threshold/3600:.1f} hours = {t_threshold/86400:.1f} days")
        print(f"\n  Comparison:")
        print(f"    Analytical estimate: {t_est:.0f} s ({t_est/86400:.1f} days)")
        print(f"    Numerical result:    {t_threshold:.0f} s ({t_threshold/86400:.1f} days)")
    else:
        print(f"\n  Simulation ended at t = {t_current:.0f} s ({t_current/86400:.1f} days)")
        print(f"  10% threshold not yet reached.")
        print(f"  (Expected around t ~ {t_est:.0f} s = {t_est/86400:.1f} days)")

    # Exact Gaussian comparison
    # For Gaussian initial condition sigma_0:
    # u(x,t) = 1/sqrt(2*pi*sigma(t)^2) * exp(-x^2/(2*sigma(t)^2))
    # sigma(t)^2 = sigma_0^2 + 2*D*t
    sigma_at_est = np.sqrt(sigma_0**2 + 2*D*t_est)
    print(f"\n  At analytical t_est:")
    print(f"    Gaussian sigma(t) = sqrt(sigma_0^2 + 2Dt) = sqrt({sigma_0**2} + {2*D*t_est:.0f}) = {sigma_at_est:.1f} m")
    print(f"    This confirms spread of ~{sigma_target} m")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
    print("\n" + "=" * 60)
    print("All exercises for Lesson 20 completed.")
    print("=" * 60)
