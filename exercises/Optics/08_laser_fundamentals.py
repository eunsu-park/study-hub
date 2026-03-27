"""
Exercises for Lesson 08: Laser Fundamentals
Topic: Optics
Solutions to practice problems from the lesson.
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: Einstein Coefficients for Ruby Laser
    Calculate the Einstein A and B coefficients, stimulated emission
    cross section, and gain for a ruby laser at 694.3 nm.
    """
    h = 6.626e-34    # Planck's constant (J*s)
    c = 3.0e8         # Speed of light (m/s)
    k_B = 1.381e-23   # Boltzmann constant (J/K)

    lam = 694.3e-9    # Ruby laser wavelength (m)
    tau_sp = 3.0e-3   # Spontaneous emission lifetime (s)
    n_medium = 1.76   # Refractive index of ruby
    delta_nu = 330e9  # Gain bandwidth (Hz) at 300K

    nu = c / lam
    print("Einstein Coefficients for Ruby Laser:")
    print(f"Wavelength: {lam*1e9:.1f} nm")
    print(f"Frequency: {nu:.4e} Hz")
    print(f"Spontaneous lifetime: {tau_sp*1e3:.1f} ms")

    # Einstein A coefficient
    A_21 = 1.0 / tau_sp
    print(f"\nEinstein A_21 = 1/tau = {A_21:.2f} s^-1")

    # Einstein B coefficient: A_21 = (8*pi*h*nu^3/c^3) * B_21
    B_21 = A_21 * c**3 / (8 * np.pi * h * nu**3)
    print(f"Einstein B_21 = {B_21:.4e} m^3/(J*s^2)")

    # Stimulated emission cross section
    # sigma = (A_21 * lambda^2) / (8*pi*n^2 * delta_nu)
    sigma_em = A_21 * lam**2 / (8 * np.pi * n_medium**2 * delta_nu)
    print(f"Stimulated emission cross section: sigma = {sigma_em:.4e} m^2")
    print(f"  = {sigma_em*1e4:.4e} cm^2")

    # Population inversion density for gain
    # g(nu) = sigma * delta_N
    # For threshold, gain = loss: sigma * delta_N * L = -ln(R1*R2) + 2*alpha*L
    L = 0.10       # Crystal length (m)
    R1, R2 = 0.999, 0.80  # Mirror reflectivities
    alpha = 1.0     # Internal loss (1/m)

    loss_total = -np.log(R1 * R2) / (2 * L) + alpha
    delta_N_th = loss_total / sigma_em

    print(f"\nThreshold analysis (L = {L*100:.0f} cm):")
    print(f"Mirror reflectivities: R1 = {R1}, R2 = {R2}")
    print(f"Internal loss: alpha = {alpha} /m")
    print(f"Total loss per pass: {loss_total:.2f} /m")
    print(f"Threshold inversion density: dN_th = {delta_N_th:.4e} /m^3")
    print(f"  = {delta_N_th*1e-6:.4e} /cm^3")

    # Thermal population at room temperature
    T = 300  # K
    ratio_thermal = np.exp(-h * nu / (k_B * T))
    print(f"\nBoltzmann ratio N2/N1 at {T} K: {ratio_thermal:.2e}")
    print(f"  (Negligible thermal population in upper level)")


def exercise_2():
    """
    Exercise 2: Laser Cavity Design (Nd:YAG)
    Design a stable laser cavity and calculate mode structure,
    beam waist, and stability condition.
    """
    lam = 1064e-9    # Nd:YAG wavelength (m)
    n = 1.82         # YAG refractive index

    # Cavity parameters
    R1 = 2.0         # Mirror 1 radius of curvature (m), concave
    R2 = 1.0         # Mirror 2 radius of curvature (m), concave
    L = 0.30         # Cavity length (m)

    print("Nd:YAG Laser Cavity Design:")
    print(f"Wavelength: {lam*1e9:.0f} nm")
    print(f"Cavity length: L = {L*100:.0f} cm")
    print(f"Mirror radii: R1 = {R1} m, R2 = {R2} m")

    # Stability parameters
    g1 = 1 - L / R1
    g2 = 1 - L / R2
    g_product = g1 * g2

    print(f"\nStability parameters:")
    print(f"  g1 = 1 - L/R1 = {g1:.4f}")
    print(f"  g2 = 1 - L/R2 = {g2:.4f}")
    print(f"  g1*g2 = {g_product:.4f}")
    print(f"  Stable: {0 <= g_product <= 1}")

    # Beam waist sizes (TEM00)
    # w1^2 = (lambda*L/pi) * sqrt(g2/(g1*(1-g1*g2)))
    w1_sq = (lam * L / np.pi) * np.sqrt(g2 / (g1 * (1 - g_product)))
    w2_sq = (lam * L / np.pi) * np.sqrt(g1 / (g2 * (1 - g_product)))
    w0_sq = (lam * L / np.pi) * np.sqrt(g_product * (1 - g_product) / (g1 + g2 - 2*g_product)**2)

    w1 = np.sqrt(abs(w1_sq))
    w2 = np.sqrt(abs(w2_sq))

    print(f"\nBeam spot sizes:")
    print(f"  At mirror 1: w1 = {w1*1e3:.3f} mm")
    print(f"  At mirror 2: w2 = {w2*1e3:.3f} mm")

    # Mode spacing
    c_light = 3.0e8
    delta_nu_FSR = c_light / (2 * L)  # Longitudinal mode spacing
    N_modes = int(330e9 / delta_nu_FSR)  # Number within Nd:YAG gain bandwidth

    print(f"\nLongitudinal mode spacing: {delta_nu_FSR/1e9:.3f} GHz")
    print(f"Nd:YAG gain bandwidth: ~0.45 nm (~120 GHz)")
    print(f"Number of longitudinal modes: ~{int(120e9/delta_nu_FSR)}")

    # Stability diagram regions
    print(f"\n--- Stability of common cavity configurations ---")
    configs = [
        ("Plane-plane", np.inf, np.inf),
        ("Confocal (symmetric)", L, L),
        ("Concentric", L/2 + 0.001, L/2 + 0.001),
        ("Hemispherical", np.inf, L + 0.001),
        ("Current design", R1, R2),
    ]
    print(f"{'Configuration':>25} {'g1':>8} {'g2':>8} {'g1*g2':>8} {'Stable?':>8}")
    print("-" * 61)
    for name, r1, r2 in configs:
        gg1 = 1 - L/r1 if np.isfinite(r1) else 1.0
        gg2 = 1 - L/r2 if np.isfinite(r2) else 1.0
        gp = gg1 * gg2
        stable = "Yes" if 0 <= gp <= 1 else "No*"
        print(f"{name:>25} {gg1:>8.4f} {gg2:>8.4f} {gp:>8.4f} {stable:>8}")


def exercise_3():
    """
    Exercise 3: Gaussian Beam Propagation
    Calculate beam parameters as a Gaussian beam propagates through
    free space and through a thin lens.
    """
    lam = 632.8e-9   # He-Ne wavelength (m)
    w0 = 0.5e-3      # Beam waist (m)

    print("Gaussian Beam Propagation:")
    print(f"Wavelength: {lam*1e9:.1f} nm")
    print(f"Beam waist: w0 = {w0*1e3:.2f} mm")

    # Rayleigh range
    z_R = np.pi * w0**2 / lam
    print(f"Rayleigh range: z_R = {z_R:.4f} m = {z_R*100:.2f} cm")

    # Beam divergence
    theta_div = lam / (np.pi * w0)
    print(f"Far-field divergence half-angle: {theta_div*1e3:.4f} mrad")

    # Beam size and radius of curvature vs distance
    print(f"\n{'z (m)':>8} {'w(z) (mm)':>12} {'R(z) (m)':>12} {'Gouy (deg)':>12}")
    print("-" * 46)

    distances = [0, 0.1, 0.5, z_R, 1.0, 2.0, 5.0, 10.0]
    for z in distances:
        w_z = w0 * np.sqrt(1 + (z/z_R)**2)
        if z == 0:
            R_z_str = "inf"
        else:
            R_z = z * (1 + (z_R/z)**2)
            R_z_str = f"{R_z:.3f}"
        gouy = np.degrees(np.arctan(z / z_R))
        print(f"{z:>8.3f} {w_z*1e3:>12.4f} {R_z_str:>12} {gouy:>12.2f}")

    # Focusing by a lens
    print(f"\n--- Focusing with a thin lens ---")
    f = 0.10  # Focal length (m)
    # Input beam at waist (z = 0)
    # Output beam waist: w0' = w0 / sqrt(1 + (z_R/f)^2)
    # Output waist location: z' = f / (1 + (f/z_R)^2)

    for f_lens in [0.05, 0.10, 0.20, 0.50]:
        w0_prime = w0 / np.sqrt(1 + (z_R/f_lens)**2)
        z_prime = f_lens / (1 + (f_lens/z_R)**2)

        print(f"\nf = {f_lens*100:.0f} cm:")
        print(f"  Focused waist: w0' = {w0_prime*1e6:.2f} um")
        print(f"  Waist location: z' = {z_prime*100:.2f} cm from lens")
        print(f"  New Rayleigh range: z_R' = {np.pi*w0_prime**2/lam*100:.4f} cm")
        print(f"  Spot size at focus: {2*w0_prime*1e6:.2f} um diameter")


def exercise_4():
    """
    Exercise 4: Laser Comparison Table
    Compare key parameters of different laser types.
    """
    print("Laser Type Comparison:")
    print("-" * 100)
    header = (f"{'Laser':>20} {'Lambda':>10} {'Medium':>12} {'Pump':>12} "
              f"{'CW/Pulsed':>10} {'Power':>12} {'App.':>15}")
    print(header)
    print("-" * 100)

    lasers = [
        ("He-Ne", "632.8 nm", "Gas", "Discharge", "CW", "1-50 mW", "Alignment"),
        ("Nd:YAG", "1064 nm", "Solid", "Flash/Diode", "Both", "W-kW", "Cutting"),
        ("CO2", "10.6 um", "Gas", "Discharge", "Both", "W-kW", "Industrial"),
        ("Ar-ion", "488/515 nm", "Gas", "Discharge", "CW", "1-20 W", "Spectroscopy"),
        ("Ti:Sapphire", "700-1050 nm", "Solid", "Laser", "Pulsed", "W (fs)", "Ultrafast"),
        ("Diode (GaAs)", "780-980 nm", "Semicond.", "Current", "Both", "mW-W", "Telecom"),
        ("Fiber", "1.0-2.0 um", "Fiber", "Diode", "Both", "W-kW", "Industrial"),
        ("Excimer", "193-351 nm", "Gas", "Discharge", "Pulsed", "W (ns)", "Lithography"),
    ]

    for laser_data in lasers:
        name, lam_str, medium, pump, mode, power, app = laser_data
        print(f"{name:>20} {lam_str:>10} {medium:>12} {pump:>12} "
              f"{mode:>10} {power:>12} {app:>15}")

    print(f"\n{'='*60}")
    print("Detailed Parameter Comparison:")

    # Efficiency comparison
    print(f"\n{'Laser':>20} {'Wall-plug eff.':>16} {'Slope eff.':>14} {'Linewidth':>12}")
    print("-" * 64)
    efficiencies = [
        ("He-Ne", "0.01-0.1%", "~10%", "~1 MHz"),
        ("Nd:YAG (diode)", "1-5%", "30-50%", "~120 GHz"),
        ("CO2", "5-20%", "~15%", "~60 MHz"),
        ("Diode laser", "30-70%", "50-80%", "1-10 nm"),
        ("Fiber laser", "20-40%", "70-85%", "~10 kHz"),
    ]
    for name, wall, slope, lw in efficiencies:
        print(f"{name:>20} {wall:>16} {slope:>14} {lw:>12}")


def exercise_5():
    """
    Exercise 5: ABCD Matrix Cascade
    Propagate a Gaussian beam through a multi-element optical system
    using the ABCD matrix formalism and the q-parameter.
    """
    lam = 1064e-9  # Nd:YAG wavelength
    w0_input = 1.0e-3  # Input beam waist (1 mm)

    print("ABCD Matrix Beam Propagation:")
    print(f"Wavelength: {lam*1e9:.0f} nm")
    print(f"Input beam waist: w0 = {w0_input*1e3:.1f} mm")

    # Rayleigh range
    z_R = np.pi * w0_input**2 / lam
    print(f"Rayleigh range: z_R = {z_R:.3f} m")

    # Complex beam parameter at waist: 1/q = 1/R - i*lambda/(pi*w^2)
    # At waist: R = infinity, so q = i*z_R
    q_input = 1j * z_R

    def propagate_q(q, M):
        """Propagate beam parameter through ABCD matrix."""
        A, B, C, D = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
        return (A * q + B) / (C * q + D)

    def q_to_params(q, lam):
        """Extract beam parameters from q."""
        inv_q = 1.0 / q
        R = 1.0 / np.real(inv_q) if abs(np.real(inv_q)) > 1e-15 else np.inf
        w = np.sqrt(-lam / (np.pi * np.imag(inv_q)))
        return w, R

    # Optical system: free space -> lens -> free space -> lens -> free space
    f1 = 0.100  # First lens (m)
    f2 = 0.050  # Second lens (m)
    d1 = 0.200  # Input to lens 1 (m)
    d2 = 0.150  # Lens 1 to lens 2 (m)
    d3 = 0.100  # Lens 2 to output (m)

    # ABCD matrices
    M_free1 = np.array([[1, d1], [0, 1]])
    M_lens1 = np.array([[1, 0], [-1/f1, 1]])
    M_free2 = np.array([[1, d2], [0, 1]])
    M_lens2 = np.array([[1, 0], [-1/f2, 1]])
    M_free3 = np.array([[1, d3], [0, 1]])

    # System matrix
    M_sys = M_free3 @ M_lens2 @ M_free2 @ M_lens1 @ M_free1

    print(f"\nOptical system: d1={d1*100:.0f}cm -> f1={f1*100:.0f}cm -> "
          f"d2={d2*100:.0f}cm -> f2={f2*100:.0f}cm -> d3={d3*100:.0f}cm")

    A, B, C, D = M_sys[0, 0], M_sys[0, 1], M_sys[1, 0], M_sys[1, 1]
    print(f"\nSystem ABCD matrix:")
    print(f"  A = {A:.4f}, B = {B:.6f}")
    print(f"  C = {C:.4f}, D = {D:.4f}")
    print(f"  Determinant = {A*D - B*C:.6f} (should be 1)")

    # Propagate step by step
    print(f"\n{'Location':>20} {'w (mm)':>10} {'R (m)':>10}")
    print("-" * 42)

    q = q_input
    w, R = q_to_params(q, lam)
    R_str = f"{R:.3f}" if np.isfinite(R) else "inf"
    print(f"{'Input (waist)':>20} {w*1e3:>10.4f} {R_str:>10}")

    q = propagate_q(q, M_free1)
    w, R = q_to_params(q, lam)
    print(f"{'Before lens 1':>20} {w*1e3:>10.4f} {R:.3f}")

    q = propagate_q(q, M_lens1)
    w, R = q_to_params(q, lam)
    print(f"{'After lens 1':>20} {w*1e3:>10.4f} {R:.3f}")

    q = propagate_q(q, M_free2)
    w, R = q_to_params(q, lam)
    print(f"{'Before lens 2':>20} {w*1e3:>10.4f} {R:.3f}")

    q = propagate_q(q, M_lens2)
    w, R = q_to_params(q, lam)
    print(f"{'After lens 2':>20} {w*1e3:>10.4f} {R:.3f}")

    q = propagate_q(q, M_free3)
    w, R = q_to_params(q, lam)
    print(f"{'Output':>20} {w*1e3:>10.4f} {R:.3f}")

    # Find the output waist location
    q_after_L2 = propagate_q(propagate_q(propagate_q(
        propagate_q(propagate_q(q_input, M_free1), M_lens1), M_free2), M_lens2),
        np.array([[1, 0], [0, 1]]))
    # Search for waist (R = infinity => real(1/q) = 0)
    z_scan = np.linspace(0, 0.3, 1000)
    for z in z_scan:
        q_z = propagate_q(q_after_L2, np.array([[1, z], [0, 1]]))
        w_z, R_z = q_to_params(q_z, lam)
        if abs(1/R_z) < 1e-3:
            print(f"\nOutput beam waist found at z = {z*100:.2f} cm after lens 2")
            print(f"  Waist size: w0 = {w_z*1e3:.4f} mm = {w_z*1e6:.2f} um")
            break


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Einstein Coefficients (Ruby Laser)", exercise_1),
        ("Exercise 2: Laser Cavity Design (Nd:YAG)", exercise_2),
        ("Exercise 3: Gaussian Beam Propagation", exercise_3),
        ("Exercise 4: Laser Comparison Table", exercise_4),
        ("Exercise 5: ABCD Matrix Cascade", exercise_5),
    ]
    for title, func in exercises:
        print(f"\n{'='*60}")
        print(f"=== {title} ===")
        print(f"{'='*60}")
        func()

    print(f"\n{'='*60}")
    print("All exercises completed!")
