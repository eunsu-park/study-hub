"""
Exercises for Lesson 13: Quantum Optics Primer
Topic: Optics
Solutions to practice problems from the lesson.
"""

import numpy as np
from scipy.special import factorial


def exercise_1():
    """
    Exercise 1: Coherent State Properties
    For a coherent state |alpha> with alpha = 3 + 4i, calculate the
    mean photon number, standard deviation, and photon number distribution.
    """
    alpha = 3.0 + 4.0j
    n_bar = np.abs(alpha)**2  # Mean photon number

    print("Coherent State Properties:")
    print(f"alpha = {alpha}")
    print(f"|alpha|^2 = {n_bar:.1f}")

    # (a) Mean photon number
    print(f"\n(a) Mean photon number: <n> = |alpha|^2 = {n_bar:.1f}")

    # (b) Standard deviation
    # For coherent state, variance = mean (Poisson distribution)
    sigma_n = np.sqrt(n_bar)
    print(f"(b) Standard deviation: sigma_n = sqrt(<n>) = {sigma_n:.4f}")
    print(f"    Relative fluctuation: sigma_n/<n> = {sigma_n/n_bar:.4f}")

    # (c) Probability of detecting exactly 25 photons
    # P(n) = e^(-n_bar) * n_bar^n / n!
    n_target = 25
    # Use log to avoid overflow: log(P) = -n_bar + n*log(n_bar) - log(n!)
    log_P = -n_bar + n_target * np.log(n_bar) - np.sum(np.log(np.arange(1, n_target + 1)))
    P_25 = np.exp(log_P)
    print(f"(c) P(n=25) = {P_25:.6f}")
    print(f"    (Poisson distribution with mean {n_bar:.0f})")

    # (d) Photon number distribution for n = 0 to 50
    print(f"\n(d) Photon number distribution P(n):")
    print(f"{'n':>4} {'P(n)':>12} {'Cumulative':>12}")
    print("-" * 30)

    cumulative = 0
    n_values = np.arange(0, 51)
    P_n = np.zeros(51)

    for n in n_values:
        if n == 0:
            log_P_n = -n_bar
        else:
            log_P_n = -n_bar + n * np.log(n_bar) - np.sum(np.log(np.arange(1, n + 1)))
        P_n[n] = np.exp(log_P_n)
        cumulative += P_n[n]
        if n % 5 == 0 or n == n_target:
            print(f"{n:>4} {P_n[n]:>12.6f} {cumulative:>12.6f}")

    # Peak of distribution
    n_peak = n_values[np.argmax(P_n)]
    print(f"\nPeak at n = {n_peak} (P = {P_n[n_peak]:.6f})")
    print(f"FWHM: ~{2.35*sigma_n:.1f} photons")

    # Verify normalization
    print(f"Sum of P(n) for n=0..50: {np.sum(P_n):.8f}")


def exercise_2():
    """
    Exercise 2: g^(2)(0) Measurement (Hanbury Brown-Twiss)
    Analyze detection records from an HBT setup to determine the
    photon statistics of the light source.
    """
    # Given data
    T = 100.0        # Measurement time (s)
    N_A = 5000       # Counts on detector A
    N_B = 4800       # Counts on detector B
    N_AB = 150       # Coincidence counts
    delta_t = 10e-9  # Coincidence window (s)

    print("g^(2)(0) Measurement:")
    print(f"Measurement time: T = {T} s")
    print(f"Detector A counts: N_A = {N_A}")
    print(f"Detector B counts: N_B = {N_B}")
    print(f"Coincidences: N_AB = {N_AB}")
    print(f"Coincidence window: delta_t = {delta_t*1e9:.0f} ns")

    # (a) Estimate g^(2)(0)
    g2 = (N_AB * T) / (N_A * N_B * delta_t)
    print(f"\n(a) g^(2)(0) = N_AB * T / (N_A * N_B * delta_t)")
    print(f"    = ({N_AB} * {T}) / ({N_A} * {N_B} * {delta_t*1e9:.0f}e-9)")
    print(f"    = {g2:.4f}")

    # Statistical uncertainty
    sigma_g2 = g2 / np.sqrt(N_AB)
    print(f"    Statistical uncertainty: +/- {sigma_g2:.4f}")

    # (b) Classical or quantum?
    print(f"\n(b) Classification:")
    if g2 >= 2:
        print(f"    g^(2)(0) = {g2:.2f} >= 2: THERMAL (bunched) light")
        print(f"    Classical: Yes (super-Poissonian)")
    elif abs(g2 - 1) < 0.1:
        print(f"    g^(2)(0) = {g2:.2f} ~ 1: COHERENT (Poissonian) light")
        print(f"    Classical: Yes")
    elif g2 < 1:
        print(f"    g^(2)(0) = {g2:.2f} < 1: QUANTUM (antibunched) light")
        print(f"    Classical: NO - violates classical inequality g^(2)(0) >= 1")
    else:
        print(f"    g^(2)(0) = {g2:.2f}: Between coherent and thermal")
        print(f"    Classical: Yes (partially bunched)")

    # (c) Source type
    print(f"\n(c) Possible source types:")
    print(f"    g^(2)(0) reference values:")
    print(f"    - Coherent (laser): g^(2)(0) = 1")
    print(f"    - Thermal:          g^(2)(0) = 2")
    print(f"    - Single photon:    g^(2)(0) = 0")
    print(f"    - N-photon Fock:    g^(2)(0) = 1 - 1/N")
    print(f"    - Squeezed vacuum:  g^(2)(0) > 2 (super-bunched)")

    if g2 > 1.5:
        print(f"\n    Measured g^(2)(0) = {g2:.2f} -> Likely thermal or pseudo-thermal source")
    elif g2 < 0.5:
        print(f"\n    Measured g^(2)(0) = {g2:.2f} -> Strong antibunching (single photon source)")


def exercise_3():
    """
    Exercise 3: Hong-Ou-Mandel (HOM) Visibility
    Calculate HOM dip parameters for photons with given spectral bandwidth.
    """
    c = 3.0e8
    lam = 810e-9          # Center wavelength (m)
    delta_lam = 2.0e-9    # Spectral bandwidth (m)

    print("Hong-Ou-Mandel Experiment:")
    print(f"Center wavelength: {lam*1e9:.0f} nm")
    print(f"Spectral bandwidth: {delta_lam*1e9:.1f} nm")

    # (a) Coherence time
    tau_c = lam**2 / (c * delta_lam)
    print(f"\n(a) Coherence time: tau_c = lambda^2 / (c * delta_lambda)")
    print(f"    = ({lam*1e9:.0f}e-9)^2 / ({c:.1e} * {delta_lam*1e9:.1f}e-9)")
    print(f"    = {tau_c*1e12:.2f} ps")
    print(f"    Coherence length: L_c = c*tau_c = {c*tau_c*1e6:.2f} um")

    # (b) Time delay for visibility = 1/e
    # HOM visibility: V(tau) = V_max * exp(-(tau/tau_c)^2) (for Gaussian spectrum)
    # V(tau) = V_max / e when tau = tau_c
    tau_1e = tau_c
    print(f"\n(b) Delay for V = V_max/e:")
    print(f"    tau = tau_c = {tau_1e*1e12:.2f} ps")
    print(f"    Corresponding path difference: {c*tau_1e*1e6:.2f} um")

    # HOM dip profile
    print(f"\n    HOM visibility vs. time delay:")
    print(f"{'tau (ps)':>10} {'tau/tau_c':>10} {'V/V_max':>10}")
    print("-" * 32)

    for tau_ratio in [0, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]:
        tau = tau_ratio * tau_c
        V = np.exp(-(tau/tau_c)**2)
        print(f"{tau*1e12:>10.2f} {tau_ratio:>10.2f} {V:>10.4f}")

    # (c) Measured visibility of 92%
    V_measured = 0.92
    V_perfect = 1.0  # Perfect indistinguishability
    print(f"\n(c) Measured HOM visibility: V = {V_measured*100:.0f}%")
    print(f"    Perfect visibility: V = {V_perfect*100:.0f}%")

    # Indistinguishability
    # V_HOM = V_max * M, where M is the mode overlap (indistinguishability)
    # For 50:50 BS, V_max = 1, so V = M
    M = V_measured / V_perfect
    print(f"\n    Photon indistinguishability: M = V/V_max = {M:.2f}")
    print(f"    = {M*100:.0f}% indistinguishable")

    print(f"\n    Sources of imperfect visibility ({(1-M)*100:.0f}% loss):")
    print(f"    - Spectral mismatch between photon sources")
    print(f"    - Temporal jitter in photon generation")
    print(f"    - Spatial mode mismatch at beamsplitter")
    print(f"    - Multi-photon events (if using SPDC)")
    print(f"    - Detector timing resolution > tau_c")


def exercise_4():
    """
    Exercise 4: BB84 QKD Key Rate
    Calculate the key rate for a BB84 quantum key distribution system
    over a fiber-optic channel.
    """
    c = 3.0e8

    # System parameters
    f_rep = 1e9          # Pulse rate (Hz)
    eta_source = 0.5     # Source efficiency
    alpha_fiber = 0.2    # Fiber loss (dB/km)
    distance = 50        # km
    eta_det = 0.10       # Detector efficiency (10%)
    p_dark = 1e-6        # Dark count probability per gate

    print("BB84 QKD System Analysis:")
    print(f"Pulse rate: {f_rep/1e9:.0f} GHz")
    print(f"Source efficiency: {eta_source}")
    print(f"Fiber loss: {alpha_fiber} dB/km")
    print(f"Distance: {distance} km")
    print(f"Detector efficiency: {eta_det*100:.0f}%")
    print(f"Dark count rate: {p_dark} per gate")

    # (a) Photon arrival rate at Bob
    total_fiber_loss_dB = alpha_fiber * distance
    total_fiber_loss = 10**(-total_fiber_loss_dB / 10)

    print(f"\n(a) Photon arrival rate:")
    print(f"  Total fiber loss: {total_fiber_loss_dB:.1f} dB")
    print(f"  Transmission: {total_fiber_loss:.2e}")
    print(f"  Photon arrival rate: {f_rep * eta_source * total_fiber_loss:.2e} photons/s")

    R_detect = f_rep * eta_source * total_fiber_loss * eta_det
    print(f"  Detection rate: {R_detect:.2e} counts/s")
    print(f"    = {R_detect/1e3:.2f} kcounts/s")

    # Dark count rate
    R_dark = f_rep * p_dark
    print(f"  Dark count rate: {R_dark:.2e} counts/s")

    # (b) Raw sifted key rate
    # Sifting: Alice and Bob use matching bases 50% of the time
    R_sifted = R_detect / 2
    print(f"\n(b) Raw sifted key rate:")
    print(f"  R_sifted = R_detect / 2 = {R_sifted:.2e} bits/s")
    print(f"    = {R_sifted/1e3:.2f} kbits/s")

    # QBER from dark counts
    QBER_dark = p_dark * f_rep / (2 * (R_detect + R_dark))
    print(f"\n  QBER contribution from dark counts: {QBER_dark*100:.4f}%")
    print(f"  (BB84 is secure for QBER < 11%)")

    # (c) Maximum distance (where dark counts dominate)
    print(f"\n(c) Maximum distance analysis:")
    print(f"{'Distance (km)':>14} {'Signal':>12} {'Dark':>12} {'QBER (%)':>10} {'Key (bits/s)':>14}")
    print("-" * 64)

    for d in [10, 20, 50, 100, 150, 200, 250, 300]:
        loss_dB = alpha_fiber * d
        transmission = 10**(-loss_dB / 10)
        R_sig = f_rep * eta_source * transmission * eta_det
        R_dk = f_rep * p_dark
        R_total = R_sig + R_dk

        if R_total > 0:
            qber = R_dk / (2 * R_total)
        else:
            qber = 0.5

        # Secure key rate (simplified asymptotic formula)
        # R_key = R_sifted * [1 - 2*h(QBER)] where h is binary entropy
        if 0 < qber < 0.5:
            h_qber = -qber * np.log2(qber) - (1-qber) * np.log2(1-qber)
        else:
            h_qber = 1.0

        R_key = (R_sig / 2) * max(0, 1 - 2 * h_qber)

        print(f"{d:>14} {R_sig:>12.2e} {R_dk:>12.2e} {qber*100:>10.4f} {R_key:>14.2e}")

    # Find crossover distance
    # Signal = Dark when eta_source * 10^(-alpha*d/10) * eta_det = p_dark
    d_max = -10 * np.log10(p_dark / (eta_source * eta_det)) / alpha_fiber
    print(f"\nSignal = Dark count at distance: {d_max:.1f} km")
    print(f"Practical limit (QBER < 11%): somewhat shorter")


def exercise_5():
    """
    Exercise 5: Squeezing and LIGO Sensitivity
    Calculate LIGO's shot-noise sensitivity and the improvement
    from squeezed light injection.
    """
    h_bar = 1.055e-34   # Reduced Planck's constant (J*s)
    c = 3.0e8
    lam = 1064e-9        # LIGO laser wavelength (m)
    omega = 2 * np.pi * c / lam

    L = 4000             # Arm length (m)
    P = 750e3            # Circulating power (W)
    tau_meas = 1e-3      # Measurement time (s)

    print("LIGO Sensitivity with Squeezed Light:")
    print(f"Arm length: L = {L} m")
    print(f"Wavelength: {lam*1e9:.0f} nm")
    print(f"Circulating power: P = {P/1e3:.0f} kW")
    print(f"Measurement time: tau = {tau_meas*1e3:.0f} ms")

    # (a) Shot-noise-limited displacement sensitivity
    # delta_x_SQL = sqrt(h_bar * c * lambda / (4 * pi * P * tau))
    # From photon counting: N_photons = P * tau / (h_bar * omega)
    N_photons = P * tau_meas / (h_bar * omega)
    print(f"\n(a) Shot-noise-limited sensitivity:")
    print(f"  Photons per measurement: N = {N_photons:.2e}")

    # Shot noise limited phase sensitivity
    delta_phi_shot = 1 / np.sqrt(N_photons)
    # Convert to displacement: delta_x = delta_phi * lambda / (4*pi)
    delta_x_SQL = delta_phi_shot * lam / (4 * np.pi)

    print(f"  Phase sensitivity: delta_phi = 1/sqrt(N) = {delta_phi_shot:.2e} rad")
    print(f"  Displacement sensitivity: delta_x = {delta_x_SQL:.2e} m")
    print(f"    = {delta_x_SQL*1e18:.4f} am (attometers)")

    # (b) With 10 dB squeezing
    squeeze_dB = 10
    squeeze_factor = 10**(squeeze_dB / 20)  # Amplitude squeeze factor
    delta_x_squeezed = delta_x_SQL / squeeze_factor

    print(f"\n(b) With {squeeze_dB} dB squeezing:")
    print(f"  Squeeze factor: {squeeze_factor:.2f}x in amplitude")
    print(f"  Improved sensitivity: delta_x = {delta_x_squeezed:.2e} m")
    print(f"    = {delta_x_squeezed*1e18:.4f} am")
    print(f"  Improvement: {squeeze_factor:.1f}x better ({squeeze_dB} dB)")

    # (c) Compare with gravitational wave strain
    h_min = 1e-21  # Minimum detectable strain
    delta_L = h_min * L / 2
    print(f"\n(c) Gravitational wave comparison:")
    print(f"  Minimum strain: h = {h_min:.0e}")
    print(f"  Corresponding displacement: delta_L = h*L/2 = {delta_L:.2e} m")
    print(f"    = {delta_L*1e18:.4f} am")

    print(f"\n  Comparison:")
    print(f"    SQL sensitivity:      {delta_x_SQL*1e18:.4f} am")
    print(f"    Squeezed sensitivity: {delta_x_squeezed*1e18:.4f} am")
    print(f"    GW displacement:      {delta_L*1e18:.4f} am")

    ratio_SQL = delta_L / delta_x_SQL
    ratio_sq = delta_L / delta_x_squeezed
    print(f"\n    GW signal / SQL noise: {ratio_SQL:.2f}")
    print(f"    GW signal / squeezed noise: {ratio_sq:.2f}")

    if ratio_sq > 1:
        print(f"    -> Detectable with squeezing (SNR = {ratio_sq:.1f})")
    else:
        print(f"    -> Requires additional techniques (power recycling, signal recycling)")

    # LIGO actual sensitivity
    print(f"\n  Note: LIGO achieves ~{1e-19*1e18:.0f} am sensitivity")
    print(f"  through power recycling, signal recycling,")
    print(f"  and frequency-dependent squeezing (since 2019)")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Coherent State Properties", exercise_1),
        ("Exercise 2: g^(2)(0) Measurement", exercise_2),
        ("Exercise 3: HOM Visibility", exercise_3),
        ("Exercise 4: BB84 Key Rate", exercise_4),
        ("Exercise 5: Squeezing and LIGO", exercise_5),
    ]
    for title, func in exercises:
        print(f"\n{'='*60}")
        print(f"=== {title} ===")
        print(f"{'='*60}")
        func()

    print(f"\n{'='*60}")
    print("All exercises completed!")
