"""
Exercises for Lesson 09: Fiber Optics
Topic: Optics
Solutions to practice problems from the lesson.
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: Fiber Design Parameters
    Calculate V-number, NA, number of modes, and mode field diameter
    for step-index and graded-index fibers.
    """
    lam = 1.55e-6     # Telecom wavelength (m)
    n_core = 1.4500    # Core refractive index
    n_clad = 1.4440    # Cladding refractive index
    a = 4.1e-6         # Core radius (m) - typical SMF

    # Numerical aperture
    NA = np.sqrt(n_core**2 - n_clad**2)
    print("Step-Index Fiber Design:")
    print(f"Core index: n1 = {n_core}")
    print(f"Cladding index: n2 = {n_clad}")
    print(f"Core radius: a = {a*1e6:.1f} um")
    print(f"Wavelength: lambda = {lam*1e6:.2f} um")
    print(f"\nNumerical aperture: NA = {NA:.4f}")

    # Acceptance angle
    theta_max = np.arcsin(NA)
    print(f"Maximum acceptance angle: {np.degrees(theta_max):.2f} deg")

    # V-number (normalized frequency)
    V = 2 * np.pi * a * NA / lam
    print(f"V-number: V = {V:.4f}")
    print(f"Single-mode condition: V < 2.405")
    print(f"This fiber is {'single-mode' if V < 2.405 else 'multi-mode'}")

    # Number of modes (for multimode, N ~ V^2/2 for step-index)
    if V >= 2.405:
        N_modes = int(V**2 / 2)
        print(f"Approximate number of modes: {N_modes}")

    # Mode Field Diameter (Marcuse approximation for SMF)
    # MFD ~ 2*a * (0.65 + 1.619/V^1.5 + 2.879/V^6)
    MFD = 2 * a * (0.65 + 1.619 / V**1.5 + 2.879 / V**6)
    print(f"\nMode Field Diameter (Marcuse): MFD = {MFD*1e6:.2f} um")
    print(f"  (Standard SMF-28: MFD ~ 10.4 um at 1550 nm)")

    # Cutoff wavelength
    lam_cutoff = 2 * np.pi * a * NA / 2.405
    print(f"Cutoff wavelength: {lam_cutoff*1e6:.3f} um")

    # Multimode fiber comparison
    print(f"\n--- Multimode Fiber (OM3) ---")
    a_mm = 25e-6    # 50 um core diameter
    NA_mm = 0.20
    V_mm = 2 * np.pi * a_mm * NA_mm / 0.85e-6  # at 850 nm
    N_modes_mm = int(V_mm**2 / 2)
    print(f"Core radius: {a_mm*1e6:.0f} um, NA = {NA_mm}")
    print(f"V-number at 850 nm: {V_mm:.1f}")
    print(f"Approximate number of modes: {N_modes_mm}")


def exercise_2():
    """
    Exercise 2: Fiber Link Budget
    Calculate the power budget for a fiber-optic communication link
    including all losses and margins.
    """
    # Link parameters
    P_tx = 0.0  # Transmitter power (dBm) = 1 mW
    distance = 40.0  # km
    alpha_fiber = 0.2  # Fiber attenuation (dB/km) at 1550 nm
    n_splices = 8
    loss_splice = 0.1  # dB per splice
    n_connectors = 4
    loss_connector = 0.5  # dB per connector
    sensitivity = -28.0  # Receiver sensitivity (dBm)
    margin = 3.0  # System margin (dB)

    print("Fiber-Optic Link Budget Analysis:")
    print(f"{'='*45}")

    # Fiber loss
    loss_fiber = alpha_fiber * distance
    loss_splices = n_splices * loss_splice
    loss_connectors = n_connectors * loss_connector
    loss_total = loss_fiber + loss_splices + loss_connectors

    print(f"Transmitter power: {P_tx:.1f} dBm ({10**(P_tx/10):.1f} mW)")
    print(f"\nLoss breakdown:")
    print(f"  Fiber ({distance:.0f} km x {alpha_fiber} dB/km): {loss_fiber:.1f} dB")
    print(f"  Splices ({n_splices} x {loss_splice} dB): {loss_splices:.1f} dB")
    print(f"  Connectors ({n_connectors} x {loss_connector} dB): {loss_connectors:.1f} dB")
    print(f"  Total loss: {loss_total:.1f} dB")

    # Power at receiver
    P_rx = P_tx - loss_total
    print(f"\nReceived power: {P_rx:.1f} dBm ({10**(P_rx/10)*1000:.3f} uW)")
    print(f"Receiver sensitivity: {sensitivity:.1f} dBm")

    # Power margin
    power_margin = P_rx - sensitivity
    print(f"\nPower margin: {power_margin:.1f} dB")
    print(f"Required system margin: {margin:.1f} dB")
    print(f"Excess margin: {power_margin - margin:.1f} dB")

    if power_margin >= margin:
        print(f"\nLink budget: PASS (margin = {power_margin:.1f} dB >= {margin:.1f} dB)")
    else:
        print(f"\nLink budget: FAIL (need {margin - power_margin:.1f} dB more)")

    # Maximum distance
    available_loss = P_tx - sensitivity - margin - loss_splices - loss_connectors
    max_distance = available_loss / alpha_fiber
    print(f"\nMaximum link distance: {max_distance:.1f} km")

    # Wavelength comparison
    print(f"\n--- Attenuation at Different Wavelengths ---")
    wavelengths = [("850 nm", 2.5), ("1310 nm", 0.35), ("1550 nm", 0.20), ("1625 nm", 0.23)]
    for wl_name, alpha in wavelengths:
        max_dist = available_loss / alpha if alpha > 0 else np.inf
        print(f"  {wl_name}: alpha = {alpha} dB/km, max distance = {max_dist:.1f} km")


def exercise_3():
    """
    Exercise 3: Dispersion-Limited Distance
    Calculate the maximum distance for a 10 Gb/s link considering
    chromatic dispersion and PMD.
    """
    # System parameters
    bit_rate = 10e9     # 10 Gb/s
    lam = 1.55e-6       # Wavelength (m)
    c = 3.0e8

    # Chromatic dispersion parameters
    D_fiber = 17.0      # ps/(nm*km) for SMF-28 at 1550 nm
    S_0 = 0.092         # Dispersion slope ps/(nm^2*km)

    print("Dispersion-Limited Distance Analysis:")
    print(f"Bit rate: {bit_rate/1e9:.0f} Gb/s")
    print(f"Wavelength: {lam*1e6:.2f} um")
    print(f"Chromatic dispersion: D = {D_fiber} ps/(nm*km)")

    # Maximum allowed pulse broadening: sigma < T_bit / 4
    T_bit = 1.0 / bit_rate  # Bit period (s)
    sigma_max = T_bit / 4    # Maximum tolerable broadening

    print(f"\nBit period: T_bit = {T_bit*1e12:.1f} ps")
    print(f"Max pulse broadening: sigma_max = {sigma_max*1e12:.2f} ps")

    # For directly modulated laser (spectral width ~1 nm)
    delta_lam_DML = 1.0  # nm
    L_max_DML = sigma_max * 1e12 / (D_fiber * delta_lam_DML)

    print(f"\n--- Directly Modulated Laser (delta_lambda = {delta_lam_DML} nm) ---")
    print(f"Dispersion-limited distance: L_max = {L_max_DML:.1f} km")

    # For externally modulated laser (transform-limited, ~0.08 nm at 10 Gb/s)
    delta_lam_EML = 0.08  # nm
    L_max_EML = sigma_max * 1e12 / (D_fiber * delta_lam_EML)

    print(f"\n--- Externally Modulated Laser (delta_lambda = {delta_lam_EML} nm) ---")
    print(f"Dispersion-limited distance: L_max = {L_max_EML:.1f} km")

    # PMD (Polarization Mode Dispersion)
    PMD_coeff = 0.1  # ps/sqrt(km) (modern fiber)
    print(f"\n--- Polarization Mode Dispersion ---")
    print(f"PMD coefficient: {PMD_coeff} ps/sqrt(km)")

    distances_check = [50, 100, 200, 500, 1000]
    print(f"{'Distance (km)':>14} {'PMD (ps)':>10} {'PMD/T_bit':>10}")
    print("-" * 36)
    for L_check in distances_check:
        pmd = PMD_coeff * np.sqrt(L_check)
        print(f"{L_check:>14} {pmd:>10.2f} {pmd/(T_bit*1e12):>10.4f}")

    # PMD limit: DGD < T_bit / 10
    L_max_PMD = (T_bit * 1e12 / (10 * PMD_coeff))**2
    print(f"\nPMD-limited distance: L_max = {L_max_PMD:.0f} km")

    # Combined limit
    print(f"\n--- Summary ---")
    print(f"Chromatic dispersion limit (DML): {L_max_DML:.1f} km")
    print(f"Chromatic dispersion limit (EML): {L_max_EML:.1f} km")
    print(f"PMD limit: {L_max_PMD:.0f} km")
    print(f"Practical limit: {min(L_max_EML, L_max_PMD):.0f} km (with EML)")


def exercise_4():
    """
    Exercise 4: Fiber Bragg Grating Sensor
    Calculate the Bragg wavelength, sensitivity to strain and temperature,
    and design a sensor interrogation scheme.
    """
    # FBG parameters
    n_eff = 1.4682     # Effective refractive index
    Lambda = 535.0e-9  # Grating period (m)

    # Bragg wavelength: lambda_B = 2 * n_eff * Lambda
    lam_B = 2 * n_eff * Lambda

    print("Fiber Bragg Grating Sensor:")
    print(f"Effective index: n_eff = {n_eff:.4f}")
    print(f"Grating period: Lambda = {Lambda*1e9:.1f} nm")
    print(f"Bragg wavelength: lambda_B = {lam_B*1e9:.3f} nm")

    # Strain sensitivity: delta_lambda/lambda = (1 - p_e) * epsilon
    # p_e ~ 0.22 for silica fiber (effective photo-elastic coefficient)
    p_e = 0.22
    strain_sensitivity = lam_B * (1 - p_e)  # m per unit strain
    print(f"\n--- Strain Sensitivity ---")
    print(f"Photo-elastic coefficient: p_e = {p_e}")
    print(f"Sensitivity: {strain_sensitivity*1e9/1e-6:.4f} pm/ustrain")

    # Example: 1000 microstrain
    epsilon = 1000e-6
    delta_lam_strain = lam_B * (1 - p_e) * epsilon
    print(f"For {epsilon*1e6:.0f} microstrain:")
    print(f"  Wavelength shift: {delta_lam_strain*1e9:.4f} nm = {delta_lam_strain*1e12:.2f} pm")

    # Temperature sensitivity: delta_lambda/lambda = (alpha + xi) * delta_T
    # alpha ~ 0.55e-6 /K (thermal expansion)
    # xi ~ 6.67e-6 /K (thermo-optic coefficient)
    alpha_T = 0.55e-6   # /K
    xi = 6.67e-6         # /K
    temp_sensitivity = lam_B * (alpha_T + xi)  # m/K

    print(f"\n--- Temperature Sensitivity ---")
    print(f"Thermal expansion: alpha = {alpha_T*1e6:.2f} ppm/K")
    print(f"Thermo-optic: xi = {xi*1e6:.2f} ppm/K")
    print(f"Sensitivity: {temp_sensitivity*1e12:.2f} pm/K")

    # Example: 50 K temperature change
    delta_T = 50  # K
    delta_lam_temp = temp_sensitivity * delta_T
    print(f"For delta_T = {delta_T} K:")
    print(f"  Wavelength shift: {delta_lam_temp*1e9:.4f} nm = {delta_lam_temp*1e12:.1f} pm")

    # Reflectivity
    print(f"\n--- FBG Reflectivity ---")
    # R_max = tanh^2(kappa * L), where kappa = pi*delta_n/lambda_B
    delta_n_values = [1e-4, 5e-4, 1e-3]
    L_grating = 10e-3  # 10 mm grating length

    print(f"Grating length: L = {L_grating*1e3:.0f} mm")
    print(f"{'delta_n':>12} {'kappa (1/m)':>14} {'R_max':>10} {'BW (nm)':>10}")
    print("-" * 48)

    for dn in delta_n_values:
        kappa = np.pi * dn / lam_B
        R_max = np.tanh(kappa * L_grating)**2
        # 3dB bandwidth: delta_lambda ~ lambda_B * sqrt((delta_n/(2*n_eff))^2 + (1/(N))^2)
        N = L_grating / Lambda
        BW = lam_B * np.sqrt((dn / (2*n_eff))**2 + (1/N)**2)
        print(f"{dn:>12.1e} {kappa:>14.1f} {R_max:>10.4f} {BW*1e9:>10.4f}")

    # Multiplexed sensor array
    print(f"\n--- Wavelength Division Multiplexing ---")
    n_sensors = 10
    channel_spacing = 4.0  # nm
    start_lam = 1530.0     # nm
    print(f"Number of FBG sensors: {n_sensors}")
    print(f"Channel spacing: {channel_spacing} nm")
    for i in range(n_sensors):
        center = start_lam + i * channel_spacing
        print(f"  Sensor {i+1}: lambda_B = {center:.1f} nm "
              f"(range: {center - channel_spacing/2:.1f} - {center + channel_spacing/2:.1f} nm)")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Fiber Design Parameters", exercise_1),
        ("Exercise 2: Fiber Link Budget", exercise_2),
        ("Exercise 3: Dispersion-Limited Distance", exercise_3),
        ("Exercise 4: FBG Sensor", exercise_4),
    ]
    for title, func in exercises:
        print(f"\n{'='*60}")
        print(f"=== {title} ===")
        print(f"{'='*60}")
        func()

    print(f"\n{'='*60}")
    print("All exercises completed!")
