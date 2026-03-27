"""
Exercises for Lesson 15: Computational Electromagnetics
Topic: Numerical_Simulation

Solutions to practice problems from the lesson.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Physical constants
eps0 = 8.854187817e-12   # Vacuum permittivity [F/m]
mu0 = 4 * np.pi * 1e-7   # Vacuum permeability [H/m]
c0 = 1 / np.sqrt(mu0 * eps0)  # Speed of light [m/s]


# === Exercise 1: Wave Equation Derivation ===
# Problem: Derive the wave equation for E from Faraday's and Ampere-Maxwell laws.

def exercise_1():
    """Derive and verify the electromagnetic wave equation numerically."""
    print("Derivation of the wave equation for E:")
    print()
    print("Step 1: Start with Faraday's law: curl E = -dB/dt")
    print("Step 2: Take the curl of both sides: curl(curl E) = -d(curl B)/dt")
    print("Step 3: Substitute Ampere-Maxwell (vacuum, no sources):")
    print("        curl B = mu0 eps0 dE/dt")
    print("        => curl(curl E) = -mu0 eps0 d^2E/dt^2")
    print("Step 4: Apply vector identity:")
    print("        curl(curl F) = grad(div F) - nabla^2 F")
    print("Step 5: In vacuum with no charges, div E = 0 (Gauss's law)")
    print("        => -nabla^2 E = -mu0 eps0 d^2E/dt^2")
    print("Step 6: Rearranging:")
    print("        nabla^2 E = (1/c^2) d^2E/dt^2")
    print(f"        where c = 1/sqrt(mu0 eps0) = {c0:.3e} m/s")
    print()

    # Numerical verification: plane wave E = E0 sin(kx - wt) satisfies wave eq
    L = 1.0
    nx = 200
    dx = L / nx
    x = np.linspace(0, L, nx)

    k = 2 * np.pi / 0.2  # wavenumber
    omega = c0 * k        # angular frequency (dispersion relation)
    t = 0.0

    # E field
    E = np.sin(k * x - omega * t)

    # d^2E/dx^2 (analytically: -k^2 sin(kx - wt))
    d2E_dx2 = -k**2 * np.sin(k * x - omega * t)

    # (1/c^2) d^2E/dt^2 (analytically: -omega^2/c^2 sin(kx - wt) = -k^2 sin(...))
    d2E_dt2_over_c2 = -(omega**2 / c0**2) * np.sin(k * x - omega * t)

    residual = np.max(np.abs(d2E_dx2 - d2E_dt2_over_c2))
    print(f"Verification: plane wave E = sin(kx - wt)")
    print(f"  |nabla^2 E - (1/c^2) d^2E/dt^2| = {residual:.2e}")
    print(f"  Wave equation satisfied: {residual < 1e-10}")


# === Exercise 2: 1D FDTD Dielectric Interface ===
# Problem: Simulate a Gaussian pulse hitting a dielectric (eps_r=4) and compare
# reflection/transmission coefficients with Fresnel formulas.

def exercise_2():
    """1D FDTD with dielectric interface, Fresnel coefficient comparison."""
    Nx = 400
    dx = 1e-3  # 1 mm
    dt = 0.99 * dx / c0
    n_steps = 600

    # Fields
    Ey = np.zeros(Nx)
    Hz = np.zeros(Nx)

    # Material: vacuum on left, dielectric (eps_r=4) on right half
    eps_r = np.ones(Nx)
    eps_r[Nx // 2:] = 4.0

    # Coefficients
    Ca = dt / (mu0 * dx)
    Cb = dt / (eps0 * eps_r * dx)

    # Source
    source_pos = Nx // 4
    t0 = 80 * dt
    tau = 25 * dt

    # Probes
    probe_inc = Nx // 4 - 20   # Before interface (incident + reflected)
    probe_trans = 3 * Nx // 4   # After interface (transmitted)

    Ey_inc = []
    Ey_trans = []

    # Previous values for Mur ABC
    Ey_left_prev = [0.0, 0.0]
    Ey_right_prev = [0.0, 0.0]

    for n in range(n_steps):
        # H update
        Hz[:-1] = Hz[:-1] - Ca * (Ey[1:] - Ey[:-1])

        # Source (soft)
        t = n * dt
        source = np.exp(-((t - t0) / tau)**2)
        Ey[source_pos] += source

        # E update
        Ey[1:-1] = Ey[1:-1] - Cb[1:-1] * (Hz[1:-1] - Hz[:-2])

        # Mur ABC
        coeff = (c0 * dt - dx) / (c0 * dt + dx)
        Ey_new_left = Ey_left_prev[1] + coeff * (Ey[1] - Ey_left_prev[0])
        Ey_left_prev = [Ey_new_left, Ey[1]]
        Ey[0] = Ey_new_left

        coeff_r = (c0 / np.sqrt(eps_r[-1]) * dt - dx) / (c0 / np.sqrt(eps_r[-1]) * dt + dx)
        Ey_new_right = Ey_right_prev[1] + coeff_r * (Ey[-2] - Ey_right_prev[0])
        Ey_right_prev = [Ey_new_right, Ey[-2]]
        Ey[-1] = Ey_new_right

        Ey_inc.append(Ey[probe_inc])
        Ey_trans.append(Ey[probe_trans])

    # Fresnel formulas (normal incidence)
    n1 = 1.0   # sqrt(eps_r1)
    n2 = 2.0   # sqrt(eps_r2) = sqrt(4)
    r_fresnel = (n1 - n2) / (n1 + n2)
    t_fresnel = 2 * n1 / (n1 + n2)

    # Measure from simulation
    Ey_inc = np.array(Ey_inc)
    Ey_trans = np.array(Ey_trans)

    E_incident_max = np.max(np.abs(Ey_inc[:200]))
    E_reflected_max = np.max(np.abs(Ey_inc[300:]))
    E_transmitted_max = np.max(np.abs(Ey_trans))

    r_sim = E_reflected_max / E_incident_max
    t_sim = E_transmitted_max / E_incident_max

    print(f"1D FDTD: Dielectric interface (eps_r1=1, eps_r2=4)")
    print(f"  Fresnel: |r| = {abs(r_fresnel):.4f}, |t| = {t_fresnel:.4f}")
    print(f"  FDTD:    |r| ~ {r_sim:.4f}, |t| ~ {t_sim:.4f}")
    print(f"  (Note: approximate due to pulse broadening and ABC imperfections)")

    fig, ax = plt.subplots(figsize=(10, 4))
    t_arr = np.arange(n_steps) * dt * 1e9
    ax.plot(t_arr, Ey_inc, 'b-', label=f'Incident + Reflected (x={probe_inc})')
    ax.plot(t_arr, Ey_trans, 'r-', label=f'Transmitted (x={probe_trans})')
    ax.set_xlabel('Time [ns]')
    ax.set_ylabel('Ey')
    ax.set_title('1D FDTD: Gaussian Pulse at Dielectric Interface')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ex15_fdtd_dielectric.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Plot saved: ex15_fdtd_dielectric.png")


# === Exercise 3: Courant Number and Stability ===
# Problem: Run 1D FDTD with S=0.5, 0.99, 1.01 and observe stability.

def exercise_3():
    """Courant number stability test for 1D FDTD."""
    Nx = 200
    dx = 1e-3
    n_steps = 500

    courant_numbers = [0.5, 0.99, 1.01]

    print("Courant number stability test (1D FDTD):")
    fig, ax = plt.subplots(figsize=(10, 5))

    for S in courant_numbers:
        dt = S * dx / c0
        Ey = np.zeros(Nx)
        Hz = np.zeros(Nx)

        Ca = dt / (mu0 * dx)
        Cb = dt / (eps0 * dx)

        max_vals = []

        for n in range(n_steps):
            Hz[:-1] = Hz[:-1] - Ca * (Ey[1:] - Ey[:-1])

            # Initial pulse
            if n == 0:
                Ey[Nx // 4] = 1.0

            Ey[1:-1] = Ey[1:-1] - Cb * (Hz[1:-1] - Hz[:-2])
            Ey[0] = 0
            Ey[-1] = 0

            max_vals.append(np.max(np.abs(Ey)))

        stable = max_vals[-1] < 10
        label = f'S={S} ({"stable" if stable else "UNSTABLE"})'
        ax.semilogy(max_vals, linewidth=1.5, label=label)

        # Find instability onset
        if not stable:
            for step, val in enumerate(max_vals):
                if val > 2.0:
                    print(f"  S={S}: unstable at step {step} "
                          f"(max|Ey|={val:.2f})")
                    break
        else:
            print(f"  S={S}: stable, final max|Ey| = {max_vals[-1]:.4f}")

    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time step')
    ax.set_ylabel('max|Ey|')
    ax.set_title('1D FDTD Stability vs Courant Number')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim(1e-4, 1e10)
    plt.tight_layout()
    plt.savefig('ex15_courant_stability.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  Physical meaning: the CFL condition c*dt/dx <= 1 ensures that")
    print("  numerical information propagates at least as fast as the physical wave.")
    print("  Plot saved: ex15_courant_stability.png")


# === Exercise 4: Numerical Dispersion Measurement ===
# Problem: Measure wavelength in FDTD and compare to analytical for different dx.

def exercise_4():
    """Numerical dispersion error measurement for 1D FDTD."""
    freq = 5e9  # 5 GHz
    wavelength = c0 / freq
    print(f"Frequency: {freq/1e9} GHz, Wavelength: {wavelength*1000:.2f} mm")

    resolutions = [10, 20]  # cells per wavelength

    for cpw in resolutions:
        dx = wavelength / cpw
        Nx = 1000
        dt = 0.99 * dx / c0
        n_steps = 5000

        Ey = np.zeros(Nx)
        Hz = np.zeros(Nx)

        Ca = dt / (mu0 * dx)
        Cb = dt / (eps0 * dx)

        source_pos = 50

        for n in range(n_steps):
            Hz[:-1] = Hz[:-1] - Ca * (Ey[1:] - Ey[:-1])

            t = n * dt
            # CW source with smooth ramp
            ramp = min(1.0, t / (50 * dt))
            Ey[source_pos] += ramp * np.sin(2 * np.pi * freq * t)

            Ey[1:-1] = Ey[1:-1] - Cb * (Hz[1:-1] - Hz[:-2])
            Ey[0] = 0
            Ey[-1] = 0

        # Measure wavelength from steady-state pattern
        # Find zero crossings in the propagating region
        region = Ey[source_pos + 100:source_pos + 500]
        x_region = np.arange(len(region)) * dx

        # Find zero crossings
        crossings = []
        for i in range(len(region) - 1):
            if region[i] * region[i + 1] < 0:
                # Linear interpolation
                x_cross = x_region[i] - region[i] * (x_region[i+1] - x_region[i]) / (region[i+1] - region[i])
                crossings.append(x_cross)

        if len(crossings) >= 3:
            # Wavelength = distance between every other zero crossing
            half_wavelengths = np.diff(crossings)
            measured_wavelength = 2 * np.mean(half_wavelengths)
            dispersion_error = abs(measured_wavelength - wavelength) / wavelength * 100

            print(f"\n  dx = lambda/{cpw} ({dx*1000:.3f} mm):")
            print(f"    Analytical wavelength: {wavelength*1000:.4f} mm")
            print(f"    Measured wavelength:   {measured_wavelength*1000:.4f} mm")
            print(f"    Dispersion error: {dispersion_error:.2f}%")
        else:
            print(f"\n  dx = lambda/{cpw}: insufficient zero crossings to measure")

    print("\n  Conclusion: Higher resolution (more cells per wavelength) reduces")
    print("  numerical dispersion. FDTD rule of thumb: 10-20 cells per wavelength.")


if __name__ == "__main__":
    print("=== Exercise 1: Wave Equation Derivation ===")
    exercise_1()
    print("\n=== Exercise 2: 1D FDTD Dielectric Interface ===")
    exercise_2()
    print("\n=== Exercise 3: Courant Number and Stability ===")
    exercise_3()
    print("\n=== Exercise 4: Numerical Dispersion Measurement ===")
    exercise_4()
    print("\nAll exercises completed!")
