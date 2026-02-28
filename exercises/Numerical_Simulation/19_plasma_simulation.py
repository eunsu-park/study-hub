"""
Exercises for Lesson 19: Plasma Simulation
Topic: Numerical_Simulation

Solutions to practice problems from the lesson.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq


# ---------------------------------------------------------------------------
# Shared utility functions used across exercises
# ---------------------------------------------------------------------------

def boris_pusher(x, v, E, B, q, m, dt):
    """
    Boris particle pusher for charged particle motion.

    Parameters:
    -----------
    x, v : array, particle position and velocity (3D vectors)
    E, B : array, electric and magnetic fields at particle position
    q, m : float, charge and mass
    dt : float, time step

    Returns:
    --------
    x_new, v_new : updated position and velocity
    """
    qmdt2 = q * dt / (2 * m)

    # Half-acceleration (E)
    v_minus = v + qmdt2 * E

    # Rotation (B)
    t = qmdt2 * B
    s = 2 * t / (1 + np.dot(t, t))
    v_prime = v_minus + np.cross(v_minus, t)
    v_plus = v_minus + np.cross(v_prime, s)

    # Half-acceleration (E)
    v_new = v_plus + qmdt2 * E

    # Position update
    x_new = x + v_new * dt

    return x_new, v_new


def solve_poisson_1d_periodic(rho, dx, eps0=1.0):
    """Solve 1D Poisson equation with periodic BC using FFT."""
    Nx = len(rho)
    rho_k = fft(rho)
    k = fftfreq(Nx, dx) * 2 * np.pi

    phi_k = np.zeros_like(rho_k, dtype=complex)
    phi_k[1:] = rho_k[1:] / (eps0 * k[1:]**2)
    phi_k[0] = 0

    phi = np.real(ifft(phi_k))
    return phi


def electric_field_from_potential(phi, dx):
    """Calculate electric field from potential (central difference, periodic)."""
    Nx = len(phi)
    E = np.zeros(Nx)
    E[1:-1] = -(phi[2:] - phi[:-2]) / (2 * dx)
    E[0] = -(phi[1] - phi[-1]) / (2 * dx)
    E[-1] = -(phi[0] - phi[-2]) / (2 * dx)
    return E


# === Exercise 1: E x B Drift with Boris Algorithm ===
# Problem: Simulate a proton in crossed uniform E and B fields.
# Verify E x B drift velocity matches vd = E/B.

def exercise_1():
    """E x B drift simulation with Boris algorithm."""

    # Proton parameters (SI units)
    q = 1.602e-19    # Coulomb
    m = 1.673e-27    # kg

    # Fields
    E0 = 1000.0      # V/m in x-direction
    B0 = 0.01         # T in z-direction
    E = np.array([E0, 0.0, 0.0])
    B = np.array([0.0, 0.0, B0])

    # Theoretical E x B drift velocity
    v_drift_theory = E0 / B0  # = 1e5 m/s in y-direction
    print(f"Theoretical E x B drift speed: {v_drift_theory:.2e} m/s (y-direction)")

    # Cyclotron frequency and period
    omega_c = q * B0 / m
    T_c = 2 * np.pi / omega_c
    print(f"Cyclotron frequency: {omega_c:.2e} rad/s")
    print(f"Cyclotron period: {T_c:.2e} s")

    # Simulation
    dt = T_c / 100   # 100 steps per cyclotron period
    n_periods = 10
    n_steps = int(n_periods * T_c / dt)

    x = np.array([0.0, 0.0, 0.0])
    v = np.array([0.0, 0.0, 0.0])  # Start at rest

    trajectory = [x.copy()]
    velocity = [v.copy()]

    for _ in range(n_steps):
        x, v = boris_pusher(x, v, E, B, q, m, dt)
        trajectory.append(x.copy())
        velocity.append(v.copy())

    trajectory = np.array(trajectory)
    velocity = np.array(velocity)
    t = np.arange(len(trajectory)) * dt

    # Measure guiding-center drift in y-direction
    # Average vy over complete cycles
    n_full = int(n_periods * T_c / dt)
    vy_avg = (trajectory[n_full, 1] - trajectory[0, 1]) / (n_full * dt)

    print(f"\nMeasured drift speed (vy avg): {vy_avg:.2e} m/s")
    print(f"Relative error: {abs(vy_avg - v_drift_theory) / v_drift_theory:.4%}")

    # Verify energy conservation in pure magnetic field (E=0)
    print("\n--- Energy conservation test (pure B field) ---")
    x2 = np.array([0.0, 0.0, 0.0])
    v2 = np.array([1e5, 0.0, 0.0])  # Initial velocity in x
    E_zero = np.array([0.0, 0.0, 0.0])

    KE_initial = 0.5 * m * np.dot(v2, v2)
    for _ in range(int(5 * T_c / dt)):
        x2, v2 = boris_pusher(x2, v2, E_zero, B, q, m, dt)
    KE_final = 0.5 * m * np.dot(v2, v2)

    print(f"Initial KE: {KE_initial:.6e} J")
    print(f"Final KE:   {KE_final:.6e} J")
    print(f"Energy change: {abs(KE_final - KE_initial) / KE_initial:.2e}")

    # Plot trajectory
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=0.8)
    axes[0].plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=8, label='Start')
    axes[0].set_xlabel('x (m)')
    axes[0].set_ylabel('y (m)')
    axes[0].set_title('Proton Trajectory (E x B Drift)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal')

    axes[1].plot(t / T_c, velocity[:, 1], 'r-', linewidth=0.8)
    axes[1].axhline(y=v_drift_theory, color='k', linestyle='--',
                    label=f'Theory: {v_drift_theory:.0e} m/s')
    axes[1].set_xlabel('t / T_c')
    axes[1].set_ylabel('v_y (m/s)')
    axes[1].set_title('y-Velocity Component')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('exercise_19_1_exb_drift.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved to exercise_19_1_exb_drift.png")


# === Exercise 2: CIC vs NGP Charge Density Comparison ===
# Problem: Compare CIC and NGP charge assignment for the same distribution.
# Show that CIC reduces noise compared to NGP.

def exercise_2():
    """CIC vs NGP charge assignment comparison."""

    Nx = 100
    dx = 1.0
    L = Nx * dx
    n_particles = 10000
    q = -1.0  # Electron charge (normalized)

    # Uniform particle distribution with Maxwellian velocity
    np.random.seed(42)
    x_particles = np.random.uniform(0, L, n_particles)

    # NGP charge assignment
    rho_ngp = np.zeros(Nx)
    for xp in x_particles:
        xp_mod = xp % L
        i = int(round(xp_mod / dx)) % Nx
        rho_ngp[i] += q / dx

    # CIC charge assignment
    rho_cic = np.zeros(Nx)
    for xp in x_particles:
        xp_mod = xp % L
        i = int(xp_mod / dx)
        i_next = (i + 1) % Nx
        frac = (xp_mod / dx) - i
        rho_cic[i] += q * (1 - frac) / dx
        rho_cic[i_next] += q * frac / dx

    # Expected mean density
    mean_density = n_particles * q / L

    # Noise analysis: standard deviation relative to mean
    noise_ngp = np.std(rho_ngp) / abs(np.mean(rho_ngp))
    noise_cic = np.std(rho_cic) / abs(np.mean(rho_cic))

    print("CIC vs NGP Charge Assignment")
    print("=" * 50)
    print(f"Grid cells: {Nx}, Particles: {n_particles}")
    print(f"Mean density (theory): {mean_density:.4f}")
    print(f"Mean density (NGP):    {np.mean(rho_ngp):.4f}")
    print(f"Mean density (CIC):    {np.mean(rho_cic):.4f}")
    print(f"\nRelative noise (std/mean):")
    print(f"  NGP: {noise_ngp:.4f}")
    print(f"  CIC: {noise_cic:.4f}")
    print(f"  Noise reduction factor: {noise_ngp / noise_cic:.2f}x")
    print(f"\nTheory: 1/sqrt(N_ppc) = 1/sqrt({n_particles/Nx:.0f}) = {1/np.sqrt(n_particles/Nx):.4f}")

    # Visualization
    x_grid = np.arange(Nx) * dx

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(x_grid, rho_ngp, 'b-', alpha=0.7, linewidth=0.8, label='NGP')
    axes[0].plot(x_grid, rho_cic, 'r-', alpha=0.7, linewidth=0.8, label='CIC')
    axes[0].axhline(y=mean_density, color='k', linestyle='--', label='Mean density')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('Charge density')
    axes[0].set_title('Charge Density: NGP vs CIC')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Power spectra
    spec_ngp = np.abs(fft(rho_ngp - np.mean(rho_ngp)))**2
    spec_cic = np.abs(fft(rho_cic - np.mean(rho_cic)))**2
    k = fftfreq(Nx, dx)
    mask = k > 0

    axes[1].semilogy(k[mask], spec_ngp[mask], 'b-', alpha=0.7, label='NGP')
    axes[1].semilogy(k[mask], spec_cic[mask], 'r-', alpha=0.7, label='CIC')
    axes[1].set_xlabel('Wavenumber k')
    axes[1].set_ylabel('Power spectrum')
    axes[1].set_title('Noise Spectrum Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('exercise_19_2_cic_vs_ngp.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved to exercise_19_2_cic_vs_ngp.png")


# === Exercise 3: Langmuir Wave Frequency Measurement ===
# Problem: Measure plasma oscillation frequency from PIC simulation
# and compare with theoretical omega_pe.

def exercise_3():
    """Langmuir wave frequency measurement from 1D PIC simulation."""

    # Simulation parameters (normalized units: lambda_D, omega_pe^-1)
    Nx = 64
    L = 4.0 * 2 * np.pi  # ~4 Debye lengths worth of wavelengths
    dx = L / Nx
    dt = 0.1
    n_electrons = 10000

    # Initialize electrons uniformly with Maxwellian velocities
    np.random.seed(42)
    x_p = np.random.uniform(0, L, n_electrons)
    v_p = np.random.normal(0, 1.0, n_electrons)  # vth = 1 in normalized units
    q_p = np.full(n_electrons, -1.0)
    m_p = np.full(n_electrons, 1.0)

    # Density perturbation: delta_n/n = 0.01 at k = 2*pi/L
    k_wave = 2 * np.pi / L
    amplitude = 0.01
    x_p += amplitude * np.sin(k_wave * x_p) / k_wave
    x_p = x_p % L

    weight = L / n_electrons

    # Storage for field energy time series
    n_steps = 500
    FE_history = []
    t_history = []

    for step in range(n_steps):
        # 1. Charge deposition (CIC)
        rho = np.zeros(Nx)
        for i in range(n_electrons):
            xmod = x_p[i] % L
            j = int(xmod / dx)
            j_next = (j + 1) % Nx
            frac = (xmod / dx) - j
            rho[j] += q_p[i] * (1 - frac) * weight / dx
            rho[j_next] += q_p[i] * frac * weight / dx
        rho -= np.mean(rho)

        # 2. Field solve
        phi = solve_poisson_1d_periodic(rho, dx)
        E_grid = electric_field_from_potential(phi, dx)

        # Record field energy
        FE = 0.5 * np.sum(E_grid**2) * dx
        FE_history.append(FE)
        t_history.append(step * dt)

        # 3. Interpolate field to particles and push
        for i in range(n_electrons):
            xmod = x_p[i] % L
            j = int(xmod / dx)
            j_next = (j + 1) % Nx
            frac = (xmod / dx) - j
            E_particle = (1 - frac) * E_grid[j] + frac * E_grid[j_next]

            # Leapfrog push
            v_p[i] += (q_p[i] / m_p[i]) * E_particle * dt
            x_p[i] += v_p[i] * dt
            x_p[i] = x_p[i] % L

    FE_history = np.array(FE_history)
    t_history = np.array(t_history)

    # Frequency measurement via FFT of field energy
    FE_detrended = FE_history - np.mean(FE_history)
    spectrum = np.abs(fft(FE_detrended))**2
    freqs = fftfreq(len(FE_detrended), dt)
    positive = freqs > 0

    # Find peak frequency (field energy oscillates at 2*omega)
    peak_idx = np.argmax(spectrum[positive])
    omega_measured = 2 * np.pi * freqs[positive][peak_idx] / 2  # Divide by 2 since FE ~ cos^2

    print("Langmuir Wave Frequency Measurement")
    print("=" * 50)
    print(f"Measured omega: {omega_measured:.3f} omega_pe")
    print(f"Theoretical omega_pe: 1.000")
    print(f"Relative error: {abs(omega_measured - 1.0):.4f}")

    # Bohm-Gross correction: omega^2 = omega_pe^2 * (1 + 3*k^2*lambda_D^2)
    k_sim = k_wave
    omega_BG = np.sqrt(1 + 3 * k_sim**2)
    print(f"\nBohm-Gross corrected omega: {omega_BG:.3f}")
    print(f"k * lambda_D = {k_sim:.4f}")

    # At what k does Bohm-Gross predict 10% correction?
    # 1 + 3*k^2 = 1.1^2 = 1.21  =>  k^2 = 0.07  =>  k = 0.265
    k_10pct = np.sqrt(0.1 / 3 * (2 + 0.1))  # from omega^2/omega_pe^2 = 1.1 => 3k^2 ~= 0.21
    k_10pct_approx = np.sqrt(0.21 / 3)
    print(f"\n10% Bohm-Gross correction at k*lambda_D ~ {k_10pct_approx:.3f}")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].semilogy(t_history, FE_history, 'b-', linewidth=0.8)
    axes[0].set_xlabel('t [omega_pe^-1]')
    axes[0].set_ylabel('Field Energy')
    axes[0].set_title('Langmuir Wave: Field Energy')
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(freqs[positive] * 2 * np.pi, spectrum[positive], 'r-', linewidth=0.8)
    axes[1].axvline(x=2.0, color='k', linestyle='--', label='2*omega_pe (expected)')
    axes[1].set_xlabel('omega')
    axes[1].set_ylabel('Power')
    axes[1].set_title('FFT of Field Energy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 10)

    plt.tight_layout()
    plt.savefig('exercise_19_3_langmuir_freq.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved to exercise_19_3_langmuir_freq.png")


# === Exercise 4: Two-Stream Instability Growth Rate Measurement ===
# Problem: Measure growth rate gamma for various drift velocities v0
# and compare with theoretical gamma_max ~ sqrt(3)/2 * omega_pe.

def exercise_4():
    """Two-stream instability growth rate vs drift velocity."""

    def run_two_stream(v0, Nx=128, L=2*np.pi*8, dt=0.1, n_per_beam=5000, n_steps=300):
        """Run two-stream simulation and return field energy time series."""
        dx = L / Nx
        n_total = 2 * n_per_beam

        np.random.seed(42)
        x_p = np.random.uniform(0, L, n_total)
        v_p = np.concatenate([
            np.random.normal(v0, 0.1, n_per_beam),     # Beam 1 (right)
            np.random.normal(-v0, 0.1, n_per_beam)     # Beam 2 (left)
        ])
        q_p = np.full(n_total, -1.0)
        m_p = np.full(n_total, 1.0)
        weight = L / n_total

        # Small perturbation
        x_p += 0.001 * np.random.randn(n_total)
        x_p = x_p % L

        FE_list = []
        for step in range(n_steps):
            # Charge deposition
            rho = np.zeros(Nx)
            for i in range(n_total):
                xmod = x_p[i] % L
                j = int(xmod / dx)
                j_next = (j + 1) % Nx
                frac = (xmod / dx) - j
                rho[j] += q_p[i] * (1 - frac) * weight / dx
                rho[j_next] += q_p[i] * frac * weight / dx
            rho -= np.mean(rho)

            # Field solve
            phi = solve_poisson_1d_periodic(rho, dx)
            E_grid = electric_field_from_potential(phi, dx)
            FE_list.append(0.5 * np.sum(E_grid**2) * dx)

            # Push particles
            for i in range(n_total):
                xmod = x_p[i] % L
                j = int(xmod / dx)
                j_next = (j + 1) % Nx
                frac = (xmod / dx) - j
                E_particle = (1 - frac) * E_grid[j] + frac * E_grid[j_next]
                v_p[i] += (q_p[i] / m_p[i]) * E_particle * dt
                x_p[i] += v_p[i] * dt
                x_p[i] = x_p[i] % L

        return np.array(FE_list)

    v0_values = [1.0, 2.0, 3.0, 5.0]
    dt = 0.1
    n_steps = 300

    print("Two-Stream Instability Growth Rate Measurement")
    print("=" * 60)
    print(f"{'v0':<8}{'gamma (measured)':<20}{'gamma_max (theory)':<20}")
    print("-" * 60)

    gamma_theory = np.sqrt(3) / 2  # ~ 0.866 omega_pe
    gammas_measured = []

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for v0 in v0_values:
        FE = run_two_stream(v0, n_steps=n_steps)
        t = np.arange(n_steps) * dt

        # Find linear growth region: look for exponential growth in field energy
        log_FE = np.log(FE + 1e-30)
        # Find region where FE is growing: use a middle section
        growth_start = int(0.1 * n_steps)
        growth_end = int(0.5 * n_steps)

        # Check if there is actual growth
        if FE[growth_end] > 10 * FE[growth_start]:
            mask = slice(growth_start, growth_end)
            coeffs = np.polyfit(t[mask], log_FE[mask], 1)
            gamma_meas = coeffs[0] / 2  # FE ~ exp(2*gamma*t)
        else:
            gamma_meas = 0.0

        gammas_measured.append(gamma_meas)
        print(f"{v0:<8.1f}{gamma_meas:<20.4f}{gamma_theory:<20.4f}")

        axes[0].semilogy(t, FE, linewidth=1.5, label=f'v0={v0}')

    axes[0].set_xlabel('t [omega_pe^-1]')
    axes[0].set_ylabel('Field Energy')
    axes[0].set_title('Two-Stream: Field Energy Growth')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(v0_values, gammas_measured, 'bo-', markersize=8, label='Measured')
    axes[1].axhline(y=gamma_theory, color='r', linestyle='--',
                    label=f'Theory: sqrt(3)/2 = {gamma_theory:.3f}')
    axes[1].set_xlabel('Drift velocity v0')
    axes[1].set_ylabel('Growth rate gamma')
    axes[1].set_title('Growth Rate vs Drift Velocity')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('exercise_19_4_growth_rate.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nTheoretical max growth rate: gamma_max = sqrt(3)/2 = {gamma_theory:.4f}")
    print("At high v0 the growth rate saturates because the instability bandwidth narrows.")
    print("Plot saved to exercise_19_4_growth_rate.png")


# === Exercise 5: Particle Number and Numerical Heating ===
# Problem: Run single-species Maxwellian plasma with different N_ppc
# and measure numerical heating rate.

def exercise_5():
    """Numerical heating dependence on particles per cell."""

    def run_thermal_plasma(n_ppc, Nx=32, n_periods=100):
        """Run single-species Maxwellian plasma and return KE time series."""
        L = Nx * 1.0  # Grid spacing = 1 Debye length
        dx = L / Nx
        dt = 0.2
        n_particles = n_ppc * Nx
        weight = L / n_particles
        steps_per_period = int(2 * np.pi / dt)
        n_steps = n_periods * steps_per_period

        np.random.seed(12345)
        x_p = np.random.uniform(0, L, n_particles)
        v_p = np.random.normal(0, 1.0, n_particles)
        q_p = np.full(n_particles, -1.0)
        m_p = np.full(n_particles, 1.0)

        KE_list = []
        sample_interval = steps_per_period  # Sample once per plasma period

        for step in range(n_steps):
            # Charge deposition
            rho = np.zeros(Nx)
            for i in range(n_particles):
                xmod = x_p[i] % L
                j = int(xmod / dx)
                j_next = (j + 1) % Nx
                frac = (xmod / dx) - j
                rho[j] += q_p[i] * (1 - frac) * weight / dx
                rho[j_next] += q_p[i] * frac * weight / dx
            rho -= np.mean(rho)

            # Field solve
            phi = solve_poisson_1d_periodic(rho, dx)
            E_grid = electric_field_from_potential(phi, dx)

            # Record KE at sample intervals
            if step % sample_interval == 0:
                KE = 0.5 * np.sum(m_p * v_p**2) * weight
                KE_list.append(KE)

            # Push
            for i in range(n_particles):
                xmod = x_p[i] % L
                j = int(xmod / dx)
                j_next = (j + 1) % Nx
                frac = (xmod / dx) - j
                E_particle = (1 - frac) * E_grid[j] + frac * E_grid[j_next]
                v_p[i] += (q_p[i] / m_p[i]) * E_particle * dt
                x_p[i] += v_p[i] * dt
                x_p[i] = x_p[i] % L

        return np.array(KE_list)

    n_ppc_values = [10, 50, 200]
    n_periods = 100  # Reduced from 500 for practical runtime

    print("Numerical Heating vs Particles Per Cell")
    print("=" * 60)

    fig, ax = plt.subplots(figsize=(10, 6))

    heating_rates = []

    for n_ppc in n_ppc_values:
        print(f"Running N_ppc = {n_ppc}...", end=" ", flush=True)
        KE = run_thermal_plasma(n_ppc, n_periods=n_periods)
        periods = np.arange(len(KE))

        # Heating rate: fractional energy increase per period
        if len(KE) > 1 and KE[0] > 0:
            coeffs = np.polyfit(periods, KE / KE[0], 1)
            rate_per_period = coeffs[0]
        else:
            rate_per_period = 0.0

        heating_rates.append(rate_per_period)
        print(f"heating rate = {rate_per_period:.6f} per period")

        ax.plot(periods, KE / KE[0], linewidth=1.5,
                label=f'N_ppc={n_ppc} (rate={rate_per_period:.4f}/period)')

    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='No heating')
    ax.set_xlabel('Plasma Period')
    ax.set_ylabel('KE / KE_0')
    ax.set_title('Numerical Heating vs Particles Per Cell')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('exercise_19_5_numerical_heating.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nHeating rates: {heating_rates}")
    if len(heating_rates) >= 2 and heating_rates[0] > 0 and heating_rates[1] > 0:
        ratio = heating_rates[0] / heating_rates[1]
        n_ratio = n_ppc_values[1] / n_ppc_values[0]
        print(f"Rate ratio ({n_ppc_values[0]} vs {n_ppc_values[1]}): {ratio:.2f} "
              f"(expected ~ {n_ratio:.1f} for 1/N_ppc scaling)")
    print("Numerical heating decreases approximately as 1/N_ppc.")
    print("Plot saved to exercise_19_5_numerical_heating.png")


if __name__ == "__main__":
    print("=== Exercise 1: E x B Drift with Boris Algorithm ===")
    exercise_1()

    print("\n=== Exercise 2: CIC vs NGP Charge Density ===")
    exercise_2()

    print("\n=== Exercise 3: Langmuir Wave Frequency ===")
    exercise_3()

    print("\n=== Exercise 4: Two-Stream Growth Rate ===")
    exercise_4()

    print("\n=== Exercise 5: Numerical Heating ===")
    exercise_5()

    print("\nAll exercises completed!")
