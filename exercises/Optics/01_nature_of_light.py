"""
Exercises for Lesson 01: Nature of Light
Topic: Optics
Solutions to practice problems from the lesson.
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: Photon Energy of H-alpha Line
    Calculate the energy, frequency, and momentum of a photon at the
    hydrogen Balmer-alpha wavelength (656.28 nm).
    """
    h = 6.626e-34       # Planck's constant (J*s)
    c = 3.0e8            # Speed of light (m/s)
    eV = 1.602e-19       # Electron-volt (J)
    lam = 656.28e-9      # H-alpha wavelength (m)

    # Frequency
    nu = c / lam
    print(f"Wavelength: {lam*1e9:.2f} nm")
    print(f"Frequency: {nu:.4e} Hz")

    # Energy
    E_J = h * nu
    E_eV = E_J / eV
    print(f"Photon energy: {E_J:.4e} J = {E_eV:.4f} eV")

    # Momentum
    p = h / lam
    print(f"Photon momentum: {p:.4e} kg*m/s")

    # Verify E = pc
    E_check = p * c
    print(f"Verification E = pc: {E_check:.4e} J (matches {E_J:.4e} J)")


def exercise_2():
    """
    Exercise 2: Speed of Light in Diamond
    Calculate the speed of light in diamond (n=2.42) and the
    wavelength inside the medium.
    """
    c = 3.0e8       # Speed of light in vacuum (m/s)
    n = 2.42        # Refractive index of diamond
    lam_0 = 589e-9  # Sodium D-line wavelength in vacuum (m)

    v = c / n
    lam_medium = lam_0 / n

    print(f"Refractive index of diamond: {n}")
    print(f"Speed of light in diamond: {v:.4e} m/s")
    print(f"  = {v/c:.4f} c")
    print(f"Wavelength in vacuum: {lam_0*1e9:.1f} nm")
    print(f"Wavelength in diamond: {lam_medium*1e9:.2f} nm")
    print(f"Note: Frequency remains unchanged at {c/lam_0:.4e} Hz")


def exercise_3():
    """
    Exercise 3: Dispersion Analysis Using the Sellmeier Equation
    Compute the refractive index of BK7 glass at multiple wavelengths,
    calculate the Abbe number, and analyze dispersion.
    """
    # Sellmeier coefficients for BK7
    B1, C1 = 1.03961212, 0.00600069867  # um^2
    B2, C2 = 0.231792344, 0.0200179144
    B3, C3 = 1.01046945, 103.560653

    def sellmeier_bk7(lam_um):
        """Sellmeier equation for BK7 glass. lam_um in micrometers."""
        l2 = lam_um**2
        n2 = 1 + B1*l2/(l2-C1) + B2*l2/(l2-C2) + B3*l2/(l2-C3)
        return np.sqrt(n2)

    # Compute n at several wavelengths
    wavelengths_nm = np.array([400, 450, 500, 550, 589.3, 600, 650, 700, 800, 1000])
    wavelengths_um = wavelengths_nm / 1000.0
    n_values = sellmeier_bk7(wavelengths_um)

    print("BK7 Refractive Index (Sellmeier equation):")
    print(f"{'Wavelength (nm)':>15} {'n':>10}")
    print("-" * 27)
    for lam_nm, n in zip(wavelengths_nm, n_values):
        print(f"{lam_nm:>15.1f} {n:>10.6f}")

    # Abbe number: V_d = (n_d - 1) / (n_F - n_C)
    n_d = sellmeier_bk7(0.5876)   # d-line (587.6 nm)
    n_F = sellmeier_bk7(0.4861)   # F-line (486.1 nm)
    n_C = sellmeier_bk7(0.6563)   # C-line (656.3 nm)
    V_d = (n_d - 1) / (n_F - n_C)

    print(f"\nn_d (587.6 nm) = {n_d:.6f}")
    print(f"n_F (486.1 nm) = {n_F:.6f}")
    print(f"n_C (656.3 nm) = {n_C:.6f}")
    print(f"Abbe number V_d = {V_d:.2f}")
    print(f"  (Literature value for BK7: ~64.17)")

    # Group index: n_g = n - lambda * dn/dlambda
    lam_center = 0.5876  # um
    dlam = 0.0001
    dn_dlam = (sellmeier_bk7(lam_center + dlam) - sellmeier_bk7(lam_center - dlam)) / (2*dlam)
    n_g = n_d - lam_center * dn_dlam
    print(f"\nGroup index at 587.6 nm: n_g = {n_g:.6f}")
    print(f"dn/dlambda = {dn_dlam:.4f} /um")


def exercise_4():
    """
    Exercise 4: Photon Counting with a CCD
    Calculate the photon flux from a star of magnitude V=20 collected
    by a 1-meter telescope with a CCD detector.
    """
    # Parameters
    V_mag = 20             # Visual magnitude
    D = 1.0                # Telescope diameter (m)
    eta = 0.3              # Total system efficiency (QE * throughput)
    t_exp = 300            # Exposure time (s)
    lam = 550e-9           # Effective wavelength (m)
    dlam = 100e-9          # Bandwidth (m)
    h = 6.626e-34
    c = 3.0e8

    # Zero-magnitude flux (Vega system) at V-band
    # F_0 ~ 3.64e-23 W/m^2/Hz (Bessell 1998)
    # In photons: F_0 ~ 1000 photons/s/cm^2/Angstrom for V=0
    F_0_photons = 1000  # photons/s/cm^2/Angstrom at V=0

    # Flux at magnitude V
    F_V = F_0_photons * 10**(-V_mag / 2.5)
    print(f"Star magnitude: V = {V_mag}")
    print(f"Flux at V={V_mag}: {F_V:.4e} photons/s/cm^2/A")

    # Collecting area
    A = np.pi * (D/2)**2  # m^2
    A_cm2 = A * 1e4        # cm^2
    print(f"Telescope area: {A:.4f} m^2 = {A_cm2:.1f} cm^2")

    # Total photons collected
    bandwidth_A = dlam * 1e10  # Angstroms
    N_total = F_V * A_cm2 * bandwidth_A * t_exp
    N_detected = N_total * eta

    print(f"Bandwidth: {bandwidth_A:.0f} A")
    print(f"Exposure time: {t_exp} s")
    print(f"Total photons (no losses): {N_total:.1f}")
    print(f"Detected photons (eta={eta}): {N_detected:.1f}")

    # Signal-to-noise (photon noise only)
    SNR = np.sqrt(N_detected) if N_detected > 0 else 0
    print(f"SNR (photon noise limited): {SNR:.1f}")

    # Energy per photon
    E_photon = h * c / lam
    print(f"\nEnergy per photon at {lam*1e9:.0f} nm: {E_photon:.3e} J = {E_photon/1.602e-19:.3f} eV")


def exercise_5():
    """
    Exercise 5: Phase Velocity vs Group Velocity
    Compute and compare phase and group velocities in a dispersive medium
    using a simple model n(omega) = A + B*omega^2.
    """
    c = 3.0e8  # m/s

    # Model parameters for a dispersive medium
    # n(omega) = n0 + n2 * (omega - omega0)^2
    # This is a Taylor expansion around a central frequency
    n0 = 1.50        # Refractive index at central frequency
    # dn/domega at omega0 (normal dispersion)
    dn_domega = 1e-17  # s (typical for glass)

    # Central frequency (green light, 550 nm)
    lam0 = 550e-9
    omega0 = 2 * np.pi * c / lam0

    print(f"Central wavelength: {lam0*1e9:.0f} nm")
    print(f"Central angular frequency: {omega0:.4e} rad/s")
    print(f"Refractive index at center: n0 = {n0}")
    print(f"dn/domega = {dn_domega:.2e} s")

    # Phase velocity at central frequency
    v_p = c / n0
    print(f"\nPhase velocity: v_p = c/n = {v_p:.4e} m/s = {v_p/c:.4f}c")

    # Group velocity: v_g = c / (n + omega * dn/domega)
    n_g = n0 + omega0 * dn_domega
    v_g = c / n_g
    print(f"Group index: n_g = n + omega*(dn/domega) = {n_g:.6f}")
    print(f"Group velocity: v_g = c/n_g = {v_g:.4e} m/s = {v_g/c:.6f}c")

    print(f"\nv_p - v_g = {(v_p - v_g):.2e} m/s")
    print(f"Ratio v_g/v_p = {v_g/v_p:.6f}")

    # GVD (Group Velocity Dispersion)
    # For propagation through L meters of medium:
    L = 0.01  # 1 cm of glass
    # Pulse broadening for a transform-limited pulse of duration tau_0
    tau_0 = 100e-15  # 100 fs pulse

    # GVD parameter: beta_2 = d^2k/domega^2
    # For our model, beta_2 ~ (2*dn/domega + omega*d^2n/domega^2)/c
    # Simplified: beta_2 ~ 2*dn_domega/c (first-order approximation)
    beta_2 = 2 * dn_domega / c  # s^2/m
    print(f"\nGVD parameter beta_2 ~ {beta_2:.2e} s^2/m")
    print(f"  = {beta_2*1e30:.2f} fs^2/mm")

    # Broadened pulse duration
    tau_out = tau_0 * np.sqrt(1 + (beta_2 * L / tau_0**2)**2)
    print(f"\nInput pulse duration: {tau_0*1e15:.0f} fs")
    print(f"After {L*100:.0f} cm of glass:")
    print(f"Output pulse duration: {tau_out*1e15:.2f} fs")
    print(f"Broadening factor: {tau_out/tau_0:.4f}")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Photon Energy of H-alpha Line", exercise_1),
        ("Exercise 2: Speed of Light in Diamond", exercise_2),
        ("Exercise 3: Dispersion Analysis (Sellmeier/Abbe)", exercise_3),
        ("Exercise 4: Photon Counting with a CCD", exercise_4),
        ("Exercise 5: Phase Velocity vs Group Velocity", exercise_5),
    ]
    for title, func in exercises:
        print(f"\n{'='*60}")
        print(f"=== {title} ===")
        print(f"{'='*60}")
        func()

    print(f"\n{'='*60}")
    print("All exercises completed!")
