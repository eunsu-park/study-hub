"""
Lesson 10: Turbulent Dynamo
Topic: MHD
Description: Exercises on Kazantsev theory, small-scale dynamo growth,
             magnetic Prandtl number scaling, helicity evolution, and
             turbulent dynamo saturation mechanisms.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def exercise_1():
    """Kazantsev Growth Rate.

    For turbulence with u_rms = 10 m/s, l = 10^6 m, eta = 10^4 m^2/s,
    compute Rm and estimate the growth rate assuming Rm_c = 100.
    """
    u_rms = 10.0        # m/s
    ell = 1e6            # m (outer scale)
    eta = 1e4            # m^2/s (magnetic diffusivity)
    Rm_c = 100.0         # critical magnetic Reynolds number

    # Magnetic Reynolds number: Rm = u_rms * l / eta
    Rm = u_rms * ell / eta
    print(f"  Magnetic Reynolds number Rm = u_rms * l / eta = {Rm:.1f}")

    # Eddy turnover time at outer scale
    tau_eddy = ell / u_rms
    print(f"  Eddy turnover time tau_eddy = l / u_rms = {tau_eddy:.1f} s")

    # Growth rate of kinematic small-scale dynamo (Kazantsev regime):
    # gamma ~ (1/tau_eddy) * (Rm/Rm_c - 1)^(1/2) for Rm > Rm_c
    # More simply: gamma ~ u_rms / l for Rm >> Rm_c
    if Rm > Rm_c:
        gamma_approx = u_rms / ell
        print(f"  Growth rate (approximate) gamma ~ u_rms / l = {gamma_approx:.2e} s^-1")
        print(f"  e-folding time = {1.0 / gamma_approx:.1f} s")

        # Refined estimate including Rm dependence
        # At the resistive scale l_eta = l * Rm^(-3/4) (Kolmogorov),
        # the eddy turnover rate is gamma ~ (u_rms/l) * Rm^(1/2)
        gamma_resistive = (u_rms / ell) * np.sqrt(Rm)
        print(f"  Growth rate at resistive scale ~ (u_rms/l) * Rm^(1/2) = {gamma_resistive:.2e} s^-1")
    else:
        print(f"  Rm = {Rm:.1f} < Rm_c = {Rm_c:.1f} => No dynamo action")

    print(f"  Rm/Rm_c = {Rm / Rm_c:.1f} => Dynamo is {'active' if Rm > Rm_c else 'inactive'}")


def exercise_2():
    """Critical Rm for Low Pm.

    If Rm_c ~ 100 * Pm^(-1/2) for Pm < 1, what is Rm_c for liquid sodium
    with Pm = 10^(-5)?
    """
    Pm = 1e-5  # magnetic Prandtl number for liquid sodium

    # Critical magnetic Reynolds number scaling for low Pm
    Rm_c = 100.0 * Pm**(-0.5)

    print(f"  Magnetic Prandtl number Pm = {Pm:.1e}")
    print(f"  Rm_c = 100 * Pm^(-1/2) = 100 * {Pm**(-0.5):.1f} = {Rm_c:.1f}")
    print(f"  This is very large! Achieving Rm ~ {Rm_c:.0f} in liquid sodium")
    print(f"  requires very high flow velocities or large system sizes.")

    # For comparison, estimate required flow speed
    # Liquid sodium: eta ~ 0.1 m^2/s, typical experiment size L ~ 1 m
    eta_Na = 0.1   # m^2/s
    L_Na = 1.0     # m
    v_required = Rm_c * eta_Na / L_Na
    print(f"  For eta = {eta_Na} m^2/s, L = {L_Na} m:")
    print(f"  Required v = Rm_c * eta / L = {v_required:.1f} m/s")


def exercise_3():
    """Resistive Scale.

    For Re = 10^4 and Pm = 0.01, compute the ratio eta_R / eta_K.
    Which dissipates at smaller scales?
    """
    Re = 1e4    # Reynolds number
    Pm = 0.01   # magnetic Prandtl number

    # Rm = Re * Pm
    Rm = Re * Pm
    print(f"  Re = {Re:.0e}, Pm = {Pm}, Rm = Re * Pm = {Rm:.1f}")

    # Kolmogorov dissipation scale (viscous): eta_K = l * Re^(-3/4)
    # Resistive dissipation scale: eta_R = l * Rm^(-3/4)
    # Ratio: eta_R / eta_K = (Re / Rm)^(3/4) = Pm^(-3/4)
    ratio = Pm**(-3.0 / 4.0)
    print(f"  eta_R / eta_K = Pm^(-3/4) = {Pm}^(-3/4) = {ratio:.2f}")

    if Pm < 1:
        print(f"  Since Pm < 1, eta_R > eta_K")
        print(f"  Resistive dissipation occurs at LARGER scales than viscous dissipation.")
        print(f"  The magnetic field is smoothed out at scales larger than the velocity field.")
        print(f"  This makes small-scale dynamo harder (higher Rm_c needed).")
    else:
        print(f"  Since Pm > 1, eta_R < eta_K")
        print(f"  Resistive dissipation occurs at SMALLER scales than viscous dissipation.")

    # Also compute the actual scales if l = 1 (normalized)
    eta_K = Re**(-3.0 / 4.0)
    eta_R = Rm**(-3.0 / 4.0)
    print(f"  In units of outer scale l:")
    print(f"    eta_K / l = Re^(-3/4) = {eta_K:.4e}")
    print(f"    eta_R / l = Rm^(-3/4) = {eta_R:.4e}")


def exercise_4():
    """Equipartition Field.

    In the ISM with rho = 10^(-21) kg/m^3, v = 10 km/s,
    compute the equipartition magnetic field in Gauss.
    """
    rho = 1e-21   # kg/m^3 (ISM density)
    v = 10e3      # m/s (turbulent velocity, 10 km/s)
    mu_0 = 4 * np.pi * 1e-7  # H/m

    # Equipartition: B^2 / (2 * mu_0) = (1/2) * rho * v^2
    # => B = sqrt(mu_0 * rho) * v
    B_SI = v * np.sqrt(mu_0 * rho)
    print(f"  rho = {rho:.1e} kg/m^3")
    print(f"  v = {v / 1e3:.1f} km/s")

    print(f"  Equipartition: B^2 / (2 mu_0) = (1/2) rho v^2")
    print(f"  B = v * sqrt(mu_0 * rho) = {B_SI:.4e} T")

    # Convert to Gauss: 1 T = 10^4 G
    B_Gauss = B_SI * 1e4
    print(f"  B = {B_Gauss:.2f} microGauss = {B_Gauss:.2e} G")
    print(f"  This is consistent with observed ISM fields of a few microGauss.")


def exercise_5():
    """Helicity Dissipation.

    For a domain of size L = 1 kpc with eta = 10^26 cm^2/s (ISM),
    estimate the resistive decay timescale of magnetic helicity.
    """
    L_kpc = 1.0
    L_cm = L_kpc * 3.086e21  # 1 kpc in cm
    eta = 1e26               # cm^2/s

    # Resistive decay timescale: tau = L^2 / eta
    tau_s = L_cm**2 / eta
    tau_yr = tau_s / (365.25 * 24 * 3600)
    tau_Gyr = tau_yr / 1e9

    print(f"  Domain size L = {L_kpc} kpc = {L_cm:.3e} cm")
    print(f"  Magnetic diffusivity eta = {eta:.1e} cm^2/s")
    print(f"  Resistive decay timescale tau = L^2 / eta")
    print(f"  tau = {tau_s:.3e} s")
    print(f"  tau = {tau_yr:.3e} yr")
    print(f"  tau = {tau_Gyr:.1f} Gyr")
    print(f"  This is much longer than the age of the universe (~13.8 Gyr),")
    print(f"  so magnetic helicity is approximately conserved in the ISM.")


def exercise_6():
    """Kazantsev Spectrum.

    Plot the expected E_B(k) for a kinematic small-scale dynamo and compare
    to a saturated state with E_B(k) ~ k^(-3/2). Find the crossing wavenumber.
    """
    # Wavenumber range
    k = np.logspace(-1, 3, 1000)

    # Kinematic Kazantsev spectrum: E_B(k) ~ k^(3/2) for k < k_eta
    # with exponential cutoff at resistive scale
    k_eta = 100.0  # resistive cutoff wavenumber
    k_peak = 50.0  # peak wavenumber
    E_kinematic = k**(3.0 / 2.0) * np.exp(-2.0 * (k / k_eta)**2)
    E_kinematic /= np.max(E_kinematic)  # normalize

    # Saturated (Kolmogorov-like) spectrum: E_B(k) ~ k^(-5/3)
    # or often E_B ~ k^(-3/2) for MHD turbulence
    k_L = 1.0  # energy injection scale
    E_saturated = np.where(k > k_L, k**(-3.0 / 2.0), k_L**(-3.0 / 2.0))
    # Normalize so they cross somewhere
    E_saturated *= 0.3

    # Find crossing point
    diff = np.abs(np.log10(E_kinematic + 1e-30) - np.log10(E_saturated + 1e-30))
    # Look for crossing in the region where both are nonzero
    mask = (E_kinematic > 1e-10) & (E_saturated > 1e-10)
    if np.any(mask):
        idx = np.argmin(diff[mask])
        k_cross = k[mask][idx]
        print(f"  Spectra cross at approximately k ~ {k_cross:.1f}")
    else:
        k_cross = None
        print(f"  No clear crossing found in this parameter range")

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.loglog(k, E_kinematic, 'b-', label=r'Kinematic: $E_B(k) \propto k^{3/2}$', linewidth=2)
    ax.loglog(k, E_saturated, 'r--', label=r'Saturated: $E_B(k) \propto k^{-3/2}$', linewidth=2)
    if k_cross is not None:
        ax.axvline(k_cross, color='gray', linestyle=':', alpha=0.7, label=f'Crossing k ~ {k_cross:.1f}')
    ax.set_xlabel('Wavenumber k', fontsize=12)
    ax.set_ylabel(r'$E_B(k)$ (normalized)', fontsize=12)
    ax.set_title('Kazantsev Spectrum: Kinematic vs Saturated', fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim(0.1, 1000)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('10_kazantsev_spectrum.png', dpi=150)
    plt.close()
    print("  Plot saved to 10_kazantsev_spectrum.png")
    print("  Kinematic phase: E_B(k) ~ k^(3/2) peaks at resistive scale")
    print("  Saturated phase: E_B(k) ~ k^(-3/2) (inverse cascade shifts energy to larger scales)")


def exercise_7():
    """Pm Scaling.

    If saturation field strength scales as B_sat ~ Pm^(1/2) for low Pm,
    by what factor does B_sat decrease when going from Pm = 1 to Pm = 10^(-6)?
    """
    Pm_1 = 1.0
    Pm_2 = 1e-6

    # B_sat ~ Pm^(1/2)
    ratio = (Pm_2 / Pm_1)**(0.5)
    print(f"  B_sat ~ Pm^(1/2)")
    print(f"  Pm_1 = {Pm_1}, Pm_2 = {Pm_2:.1e}")
    print(f"  B_sat(Pm_2) / B_sat(Pm_1) = (Pm_2/Pm_1)^(1/2) = {ratio:.1e}")
    print(f"  B_sat decreases by a factor of {1.0 / ratio:.0f}")
    print()
    print(f"  Physical interpretation:")
    print(f"  At low Pm, viscous damping destroys velocity fluctuations at scales")
    print(f"  larger than the resistive scale, reducing the efficiency of field")
    print(f"  amplification. The saturated field is much weaker than equipartition.")

    # Show scaling for a range of Pm values
    Pm_range = np.logspace(-6, 0, 7)
    print(f"\n  Pm scaling table:")
    print(f"  {'Pm':>12s}  {'B_sat/B_sat(Pm=1)':>20s}")
    print(f"  {'-'*12}  {'-'*20}")
    for Pm in Pm_range:
        factor = (Pm / 1.0)**0.5
        print(f"  {Pm:>12.1e}  {factor:>20.4e}")


def exercise_8():
    """Dynamo Growth with Saturation.

    Modify the small-scale dynamo growth to include saturation via
    gamma(B) = gamma_0 * (1 - B^2 / B_eq^2). Observe exponential
    growth -> saturation transition.
    """
    # Parameters
    gamma_0 = 0.1       # linear growth rate (1/s)
    B_eq = 1.0           # equipartition field strength
    B_0 = 1e-6           # initial (seed) field
    dt = 0.1             # timestep
    t_max = 200.0        # total time

    t = np.arange(0, t_max, dt)
    N = len(t)
    B = np.zeros(N)
    B[0] = B_0

    # Time integration: dB/dt = gamma(B) * B = gamma_0 * (1 - B^2/B_eq^2) * B
    for i in range(N - 1):
        gamma = gamma_0 * (1.0 - (B[i] / B_eq)**2)
        B[i + 1] = B[i] + dt * gamma * B[i]
        # Prevent overshoot
        B[i + 1] = min(B[i + 1], B_eq * 1.01)

    # Analytical kinematic phase: B(t) = B_0 * exp(gamma_0 * t)
    B_kinematic = B_0 * np.exp(gamma_0 * t)

    # Find transition time (when B reaches e.g. 0.1 * B_eq)
    idx_transition = np.argmin(np.abs(B - 0.1 * B_eq))
    t_transition = t[idx_transition]

    # Find saturation time (when B reaches 0.99 * B_eq)
    idx_sat = np.argmin(np.abs(B - 0.99 * B_eq))
    t_sat = t[idx_sat] if B[idx_sat] > 0.5 * B_eq else t_max

    print(f"  Parameters: gamma_0 = {gamma_0}, B_eq = {B_eq}, B_0 = {B_0:.1e}")
    print(f"  Kinematic e-folding time: tau = 1/gamma_0 = {1.0 / gamma_0:.1f} s")
    print(f"  Transition time (B = 0.1 B_eq): t ~ {t_transition:.1f} s")
    print(f"  Saturation time (B = 0.99 B_eq): t ~ {t_sat:.1f} s")
    print(f"  Final B / B_eq = {B[-1] / B_eq:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Linear plot
    ax1.plot(t, B, 'b-', label='With saturation', linewidth=2)
    ax1.axhline(B_eq, color='r', linestyle='--', label=r'$B_{eq}$', alpha=0.7)
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('B', fontsize=12)
    ax1.set_title('Dynamo Growth with Saturation (Linear)', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Log plot
    ax2.semilogy(t, B, 'b-', label='With saturation', linewidth=2)
    ax2.semilogy(t, np.clip(B_kinematic, 0, 1e10), 'g--',
                 label='Kinematic (no saturation)', linewidth=1.5, alpha=0.7)
    ax2.axhline(B_eq, color='r', linestyle='--', label=r'$B_{eq}$', alpha=0.7)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('B', fontsize=12)
    ax2.set_title('Dynamo Growth with Saturation (Log)', fontsize=13)
    ax2.set_ylim(B_0 * 0.1, B_eq * 10)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('10_dynamo_saturation.png', dpi=150)
    plt.close()
    print("  Plot saved to 10_dynamo_saturation.png")


def exercise_9():
    """Helicity Flux.

    In the magnetic helicity evolution, increase the flux rate and observe
    how it affects the saturation level of the magnetic field.
    """
    # Magnetic helicity evolution: dH/dt = 2*alpha*E_B - 2*eta*mu*E_B - F_H
    # where F_H is the helicity flux through boundaries
    # Simplified model: dB^2/dt = gamma*B^2 - (eta/L^2)*B^2*(B^2/B_eq_0^2 + F_factor)
    # F_factor represents helicity flux reducing alpha-quenching

    gamma_0 = 0.1
    B_eq_0 = 1.0
    dt = 0.1
    t_max = 300.0
    t = np.arange(0, t_max, dt)
    N = len(t)

    flux_rates = [0.0, 0.5, 1.0, 2.0, 5.0]
    results = {}

    for F_factor in flux_rates:
        B = np.zeros(N)
        B[0] = 1e-6

        for i in range(N - 1):
            # With helicity flux, the effective quenching is reduced:
            # gamma_eff = gamma_0 * (1 - B^2 / (B_eq^2 * (1 + F_factor*Rm)))
            # Simplified: larger F_factor allows higher saturation
            B_sat_eff = B_eq_0 * np.sqrt(1.0 + F_factor)
            gamma = gamma_0 * (1.0 - (B[i] / B_sat_eff)**2)
            B[i + 1] = B[i] + dt * gamma * B[i]
            B[i + 1] = min(B[i + 1], B_sat_eff * 1.01)

        results[F_factor] = B.copy()
        print(f"  F_factor = {F_factor:.1f}: B_sat/B_eq = {B[-1] / B_eq_0:.3f}, "
              f"B_sat_eff = {B_eq_0 * np.sqrt(1 + F_factor):.3f}")

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for F_factor, B in results.items():
        ax.semilogy(t, B, label=f'F = {F_factor}', linewidth=2)
    ax.axhline(B_eq_0, color='k', linestyle=':', alpha=0.5, label=r'$B_{eq,0}$')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('B', fontsize=12)
    ax.set_title('Effect of Helicity Flux on Dynamo Saturation', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_ylim(1e-7, 10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('10_helicity_flux.png', dpi=150)
    plt.close()
    print("  Plot saved to 10_helicity_flux.png")
    print("  Conclusion: Higher helicity flux alleviates catastrophic quenching,")
    print("  allowing the dynamo to saturate at a higher field strength.")


def exercise_10():
    """Shell-Model MHD Turbulence with Dynamo.

    Implement a simple shell-model for MHD turbulence. Use logarithmically
    spaced wavenumber shells and model the nonlinear transfer between shells.
    Study the energy cascade and dynamo onset as Rm is varied.
    """
    # GOY (Gledzer-Ohkitani-Yamada) shell model for MHD
    # Shells: k_n = k_0 * lambda^n, lambda = 2
    N_shells = 16
    k0 = 1.0
    lam = 2.0
    k = k0 * lam**np.arange(N_shells)

    # Parameters
    nu = 1e-6      # viscosity
    dt = 1e-4
    N_steps = 50000

    print(f"  Shell model for MHD turbulence with dynamo")
    print(f"  Number of shells: {N_shells}")
    print(f"  Wavenumber range: k = [{k[0]:.1f}, {k[-1]:.1f}]")
    print(f"  Viscosity nu = {nu:.1e}")
    print()

    # Scan over different magnetic diffusivities (eta)
    eta_values = [1e-3, 1e-4, 1e-5, 1e-6]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, eta in enumerate(eta_values):
        Rm_eff = 1.0 / (eta * k0)  # approximate Rm

        # Initialize velocity and magnetic shells with small random amplitudes
        rng = np.random.default_rng(42)
        u = np.zeros(N_shells, dtype=complex)
        b = np.zeros(N_shells, dtype=complex)

        # Seed velocity at large scales
        u[0] = 1.0 + 0.0j
        u[1] = 0.5 + 0.3j

        # Seed magnetic field at small amplitude
        b[0] = 1e-6 + 0.0j
        b[1] = 1e-6 + 0.0j

        # Forcing at shell 0
        f0 = 1.0

        E_v_history = []
        E_b_history = []
        times = []

        for step in range(N_steps):
            # Dissipation
            diss_v = -nu * k**2 * u
            diss_b = -eta * k**2 * b

            # Simplified nonlinear coupling (nearest-neighbor interaction)
            # du_n/dt = i*k_n * (u_{n+1}*u_{n+2} - b_{n+1}*b_{n+2}) + ...
            du = np.zeros(N_shells, dtype=complex)
            db = np.zeros(N_shells, dtype=complex)

            for n in range(1, N_shells - 2):
                # Velocity: forward cascade + Lorentz force
                du[n] = 1j * k[n] * (
                    np.conj(u[n + 1]) * u[n + 2] / lam
                    - np.conj(b[n + 1]) * b[n + 2] / lam
                )
                # Magnetic: induction (stretching)
                db[n] = 1j * k[n] * (
                    np.conj(u[n + 1]) * b[n + 2] / lam
                    - np.conj(b[n + 1]) * u[n + 2] / lam
                )

            # Euler step (simplified)
            u += dt * (du + diss_v)
            b += dt * (db + diss_b)

            # Forcing
            u[0] += dt * f0

            # Record energies periodically
            if step % 100 == 0:
                E_v = 0.5 * np.sum(np.abs(u)**2)
                E_b = 0.5 * np.sum(np.abs(b)**2)
                E_v_history.append(E_v)
                E_b_history.append(E_b)
                times.append(step * dt)

        E_v_arr = np.array(E_v_history)
        E_b_arr = np.array(E_b_history)
        t_arr = np.array(times)

        # Compute spectra at final time
        spec_v = 0.5 * np.abs(u)**2
        spec_b = 0.5 * np.abs(b)**2

        # Plot energy spectra
        ax = axes[idx]
        ax.loglog(k, spec_v, 'b-o', markersize=4, label=r'$E_v(k)$')
        ax.loglog(k, np.maximum(spec_b, 1e-30), 'r-s', markersize=4, label=r'$E_B(k)$')
        # Reference slopes
        k_ref = k[2:8]
        ax.loglog(k_ref, 0.1 * k_ref**(-5.0 / 3.0), 'k--', alpha=0.5, label=r'$k^{-5/3}$')
        ax.set_xlabel('k', fontsize=10)
        ax.set_ylabel('E(k)', fontsize=10)
        ax.set_title(f'eta = {eta:.1e}, Rm ~ {Rm_eff:.0f}', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Check if dynamo is active
        dynamo_active = E_b_arr[-1] > 100 * E_b_arr[0] if len(E_b_arr) > 1 else False
        status = "ACTIVE" if dynamo_active else "INACTIVE"
        print(f"  eta = {eta:.1e}, Rm ~ {Rm_eff:.0e}: Dynamo {status}, "
              f"E_v = {E_v_arr[-1]:.3e}, E_B = {E_b_arr[-1]:.3e}")

    plt.suptitle('Shell-Model MHD: Dynamo vs Rm', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('10_shell_model_dynamo.png', dpi=150)
    plt.close()
    print("  Plot saved to 10_shell_model_dynamo.png")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Kazantsev Growth Rate", exercise_1),
        ("Exercise 2: Critical Rm for Low Pm", exercise_2),
        ("Exercise 3: Resistive Scale", exercise_3),
        ("Exercise 4: Equipartition Field", exercise_4),
        ("Exercise 5: Helicity Dissipation", exercise_5),
        ("Exercise 6: Kazantsev Spectrum", exercise_6),
        ("Exercise 7: Pm Scaling", exercise_7),
        ("Exercise 8: Dynamo Growth with Saturation", exercise_8),
        ("Exercise 9: Helicity Flux", exercise_9),
        ("Exercise 10: Shell-Model MHD Turbulence", exercise_10),
    ]
    for title, func in exercises:
        print(f"\n{'=' * 60}")
        print(f" {title}")
        print(f"{'=' * 60}")
        func()
