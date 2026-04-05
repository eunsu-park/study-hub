"""
Control Theory — Lesson 7: Bode Plot Analysis and Frequency-Domain Design

Demonstrates:
1. Asymptotic Bode plot construction (straight-line approximation)
2. Exact vs asymptotic magnitude comparison
3. Bandwidth and speed-of-response relationship
4. Gain and phase margin computation from Bode data
5. Loop shaping concepts
"""
import numpy as np


# ── 1. Asymptotic Bode Magnitude ──────────────────────────────────────

def asymptotic_bode_mag(zeros: list[float], poles: list[float],
                        K: float, omega: np.ndarray) -> np.ndarray:
    """
    Compute asymptotic (straight-line) Bode magnitude in dB.

    Each real zero at -z contributes +20 dB/dec above ω = z.
    Each real pole at -p contributes -20 dB/dec above ω = p.
    Integrators (pole at 0) contribute -20 dB/dec everywhere.

    Args:
        zeros: list of break frequencies |z_i| (positive values)
        poles: list of break frequencies |p_i| (positive values, 0 for integrator)
        K: DC gain (or Bode gain for type > 0 systems)
        omega: frequency array [rad/s]
    """
    mag_db = 20 * np.log10(abs(K)) * np.ones_like(omega)

    for z in zeros:
        if z > 0:
            mag_db += np.where(omega < z, 0.0,
                               20 * np.log10(omega / z))
        # z == 0 means a zero at origin (differentiator)
        else:
            mag_db += 20 * np.log10(omega)

    for p in poles:
        if p > 0:
            mag_db -= np.where(omega < p, 0.0,
                               20 * np.log10(omega / p))
        else:
            mag_db -= 20 * np.log10(omega)

    return mag_db


# ── 2. Exact Bode from Transfer Function ──────────────────────────────

def exact_bode(num: list[float], den: list[float],
               omega: np.ndarray) -> dict:
    """
    Compute exact Bode magnitude and phase from polynomial coefficients.

    Args:
        num: numerator coefficients [a_n, ..., a_1, a_0]
        den: denominator coefficients [b_m, ..., b_1, b_0]

    Returns:
        dict with 'mag_db', 'phase_deg'
    """
    s = 1j * omega
    H_num = np.polyval(num, s)
    H_den = np.polyval(den, s)
    H = H_num / H_den

    mag_db = 20 * np.log10(np.abs(H) + 1e-30)
    phase_deg = np.degrees(np.unwrap(np.angle(H)))
    return {"mag_db": mag_db, "phase_deg": phase_deg}


# ── 3. Bandwidth Computation ──────────────────────────────────────────

def bandwidth(num: list[float], den: list[float],
              omega: np.ndarray) -> float:
    """
    Find -3 dB bandwidth of a closed-loop transfer function.

    The bandwidth is the frequency where |T(jω)| drops to -3 dB
    relative to its DC value.
    """
    bode = exact_bode(num, den, omega)
    dc_mag = bode["mag_db"][0]
    threshold = dc_mag - 3.0

    for i in range(len(omega) - 1):
        if bode["mag_db"][i] >= threshold and bode["mag_db"][i + 1] < threshold:
            # Linear interpolation
            frac = ((bode["mag_db"][i] - threshold) /
                    (bode["mag_db"][i] - bode["mag_db"][i + 1]))
            return omega[i] + frac * (omega[i + 1] - omega[i])

    return float("nan")


# ── 4. Gain and Phase Margins ─────────────────────────────────────────

def stability_margins(num: list[float], den: list[float],
                      omega: np.ndarray) -> dict:
    """
    Compute gain margin (dB) and phase margin (deg) from open-loop
    transfer function L(jω).

    Gain margin: at phase crossover (∠L = -180°), GM = -20log|L|
    Phase margin: at gain crossover (|L| = 1), PM = 180° + ∠L
    """
    bode = exact_bode(num, den, omega)
    mag_db = bode["mag_db"]
    phase = bode["phase_deg"]

    # Phase crossover: where phase crosses -180°
    gm_db = float("inf")
    wpc = float("nan")
    for i in range(len(phase) - 1):
        if phase[i] > -180 and phase[i + 1] <= -180:
            frac = (phase[i] + 180) / (phase[i] - phase[i + 1])
            wpc = omega[i] + frac * (omega[i + 1] - omega[i])
            mag_at_wpc = np.interp(wpc, omega, mag_db)
            gm_db = -mag_at_wpc
            break

    # Gain crossover: where magnitude crosses 0 dB
    pm_deg = float("inf")
    wgc = float("nan")
    for i in range(len(mag_db) - 1):
        if mag_db[i] >= 0 and mag_db[i + 1] < 0:
            frac = mag_db[i] / (mag_db[i] - mag_db[i + 1])
            wgc = omega[i] + frac * (omega[i + 1] - omega[i])
            phase_at_wgc = np.interp(wgc, omega, phase)
            pm_deg = 180.0 + phase_at_wgc
            break

    return {
        "gain_margin_db": gm_db,
        "phase_crossover_freq": wpc,
        "phase_margin_deg": pm_deg,
        "gain_crossover_freq": wgc,
    }


# ── 5. Loop Shaping: Desired Loop Transfer Function ──────────────────

def loop_shape_specs(wc_desired: float, pm_desired: float,
                     low_freq_slope: int = -1) -> dict:
    """
    Generate loop-shaping specifications.

    A well-shaped loop L(s) has:
    - Slope of -20 dB/dec near crossover
    - Sufficient phase margin
    - High gain at low frequencies for tracking

    Args:
        wc_desired: desired crossover frequency [rad/s]
        pm_desired: desired phase margin [deg]
        low_freq_slope: slope in units of -20 dB/dec (1 = type 1, 2 = type 2)

    Returns:
        Specification dict for controller design
    """
    # For -20 dB/dec at crossover with phase margin pm:
    # The phase of an integrator is -90°, so PM = 90° by default
    # Additional lag reduces PM; lead compensator restores it

    phase_needed = pm_desired - 90.0 * abs(low_freq_slope)
    needs_lead = phase_needed < 0

    return {
        "wc_desired": wc_desired,
        "pm_desired": pm_desired,
        "system_type": abs(low_freq_slope),
        "slope_at_crossover_db_dec": -20,
        "needs_lead_compensation": needs_lead,
        "phase_deficit": abs(phase_needed) if needs_lead else 0,
    }


# ── 6. Rise Time and Bandwidth Relationship ──────────────────────────

def estimate_rise_time(bw: float) -> float:
    """
    Approximate rise time from bandwidth.

    For a second-order system: t_r ≈ 1.8 / ω_bw
    """
    return 1.8 / bw


def estimate_settling_time(wn: float, zeta: float,
                           criterion: float = 0.02) -> float:
    """
    Settling time for a second-order underdamped system.

    t_s ≈ -ln(criterion) / (ζ * ω_n)
    """
    return -np.log(criterion) / (zeta * wn)


# ── Demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    omega = np.logspace(-2, 4, 10000)

    # Plant: G(s) = 100 / [(s+1)(s+10)]
    num = [100.0]
    den = np.polymul([1, 1], [1, 10]).tolist()

    print("=== Exact vs Asymptotic Bode ===")
    print(f"  G(s) = 100 / [(s+1)(s+10)]")
    print(f"  DC gain = 100/(1*10) = 10 = {20*np.log10(10):.1f} dB")

    # Asymptotic
    asym = asymptotic_bode_mag(zeros=[], poles=[1.0, 10.0],
                               K=10.0, omega=omega)
    # Exact
    ex = exact_bode(num, den, omega)

    # Compare at specific frequencies
    test_freqs = [0.1, 1.0, 3.16, 10.0, 100.0]
    print(f"\n  {'ω [rad/s]':>12} {'Asymptotic [dB]':>16} {'Exact [dB]':>12} {'Error [dB]':>12}")
    for wt in test_freqs:
        idx = np.argmin(np.abs(omega - wt))
        print(f"  {wt:12.2f} {asym[idx]:16.2f} {ex['mag_db'][idx]:12.2f}"
              f" {abs(asym[idx] - ex['mag_db'][idx]):12.2f}")
    print("  (Max error at break frequencies ≈ 3 dB, as expected)")

    # Stability margins for open-loop L(s) = 50/[s(s+1)(s+5)]
    print("\n=== Stability Margins ===")
    ol_num = [50.0]
    ol_den = np.polymul([1, 0], np.polymul([1, 1], [1, 5])).tolist()
    margins = stability_margins(ol_num, ol_den, omega)
    print(f"  L(s) = 50 / [s(s+1)(s+5)]")
    print(f"  Gain margin: {margins['gain_margin_db']:.2f} dB"
          f"  at ω = {margins['phase_crossover_freq']:.2f} rad/s")
    print(f"  Phase margin: {margins['phase_margin_deg']:.2f}°"
          f"  at ω = {margins['gain_crossover_freq']:.2f} rad/s")

    # Bandwidth and rise time
    print("\n=== Bandwidth / Rise Time ===")
    # Closed-loop: T(s) = G/(1+G) for G = 100/[(s+1)(s+10)]
    cl_num = [100.0]
    cl_den = [1, 11, 110]  # s^2 + 11s + 10 + 100
    bw = bandwidth(cl_num, cl_den, omega)
    tr = estimate_rise_time(bw)
    print(f"  Closed-loop BW: {bw:.2f} rad/s")
    print(f"  Estimated rise time: {tr:.3f} s")

    # Loop shaping specs
    print("\n=== Loop Shaping Specs ===")
    specs = loop_shape_specs(wc_desired=10.0, pm_desired=50.0,
                             low_freq_slope=-2)
    print(f"  Desired crossover: {specs['wc_desired']} rad/s")
    print(f"  System type: {specs['system_type']}")
    print(f"  Needs lead: {specs['needs_lead_compensation']}")
    print(f"  Phase deficit: {specs['phase_deficit']:.1f}°")
