"""
Control Theory — Lesson 10: Lead-Lag Compensation

Demonstrates:
1. Lead compensator design (phase-lead)
2. Lag compensator design (gain adjustment)
3. Lead-lag compensator combination
4. Frequency-domain verification
"""
import numpy as np


# ── 1. Lead Compensator ───────────────────────────────────────────────

def lead_compensator(pm_required: float, pm_uncompensated: float,
                     wc_new: float, plant_gain_at_wc: float) -> dict:
    """
    Design a lead compensator C(s) = Kc * (s + z) / (s + p).

    Uses the maximum phase-lead method:
      1. Compute required phase lead φ_max
      2. Compute α = (1 - sin(φ_max)) / (1 + sin(φ_max))
      3. Place zero and pole around new crossover frequency

    Args:
        pm_required: desired phase margin [deg]
        pm_uncompensated: current phase margin [deg]
        wc_new: desired new crossover frequency [rad/s]
        plant_gain_at_wc: |G(jω_c_new)| of the plant

    Returns:
        dict with compensator parameters
    """
    # Extra margin for safety
    phi_max_deg = pm_required - pm_uncompensated + 5.0
    phi_max = np.radians(phi_max_deg)

    alpha = (1 - np.sin(phi_max)) / (1 + np.sin(phi_max))

    # Zero and pole placement
    zero = wc_new * np.sqrt(alpha)   # z = ω_c √α
    pole = wc_new / np.sqrt(alpha)   # p = ω_c / √α

    # Gain to ensure |C(jω_c) * G(jω_c)| = 1
    comp_mag_at_wc = np.sqrt(pole / zero)  # |C(jω_c)| without Kc
    Kc = 1.0 / (plant_gain_at_wc * comp_mag_at_wc)

    # Achieved phase lead at crossover
    phase_at_wc = np.degrees(np.arctan(wc_new / zero) - np.arctan(wc_new / pole))

    return {
        "alpha": alpha,
        "zero": zero,
        "pole": pole,
        "Kc": Kc,
        "phi_max_deg": phi_max_deg,
        "phase_at_wc": phase_at_wc,
        "num": [Kc, Kc * zero],         # Kc * (s + z)
        "den": [1.0, pole],             # (s + p)
    }


# ── 2. Lag Compensator ────────────────────────────────────────────────

def lag_compensator(gain_deficit_db: float, wc: float,
                    decade_factor: float = 10.0) -> dict:
    """
    Design a lag compensator C(s) = (s + z) / (s + p) with z > p.

    The lag compensator adds low-frequency gain without significantly
    affecting the crossover frequency.

    Args:
        gain_deficit_db: how much additional DC gain is needed [dB]
        wc: crossover frequency [rad/s] (place zero/pole well below this)
        decade_factor: how many times below wc to place the zero

    Returns:
        dict with compensator parameters
    """
    beta = 10 ** (gain_deficit_db / 20)  # gain ratio

    # Place zero one decade below crossover
    zero = wc / decade_factor
    pole = zero / beta

    return {
        "beta": beta,
        "zero": zero,
        "pole": pole,
        "dc_gain_boost_db": gain_deficit_db,
        "num": [1.0, zero],
        "den": [1.0, pole],
    }


# ── 3. Lead-Lag Compensator ───────────────────────────────────────────

def lead_lag_compensator(lead_params: dict, lag_params: dict) -> dict:
    """
    Combine lead and lag compensators.

    C(s) = Kc * [(s + z_lead)(s + z_lag)] / [(s + p_lead)(s + p_lag)]
    """
    Kc = lead_params["Kc"]
    z_lead = lead_params["zero"]
    p_lead = lead_params["pole"]
    z_lag = lag_params["zero"]
    p_lag = lag_params["pole"]

    # Combined transfer function (expanded)
    num = np.polymul([1, z_lead], [1, z_lag]) * Kc
    den = np.polymul([1, p_lead], [1, p_lag])

    return {
        "Kc": Kc,
        "zeros": [-z_lead, -z_lag],
        "poles": [-p_lead, -p_lag],
        "num": num.tolist(),
        "den": den.tolist(),
    }


# ── 4. Frequency Response Evaluation ──────────────────────────────────

def eval_tf(num: list[float], den: list[float],
            omega: np.ndarray) -> np.ndarray:
    """Evaluate transfer function G(jω) at given frequencies."""
    s = 1j * omega
    numerator = sum(c * s**k for k, c in enumerate(reversed(num)))
    denominator = sum(c * s**k for k, c in enumerate(reversed(den)))
    return numerator / denominator


def bode_data(num: list[float], den: list[float],
              omega: np.ndarray) -> dict:
    """Compute Bode magnitude (dB) and phase (deg)."""
    H = eval_tf(num, den, omega)
    mag_db = 20 * np.log10(np.abs(H) + 1e-30)
    phase_deg = np.degrees(np.unwrap(np.angle(H)))
    return {"omega": omega, "mag_db": mag_db, "phase_deg": phase_deg}


def find_gain_crossover(omega: np.ndarray, mag_db: np.ndarray) -> float:
    """Find the frequency where magnitude crosses 0 dB."""
    for i in range(len(mag_db) - 1):
        if mag_db[i] >= 0 and mag_db[i + 1] < 0:
            # Linear interpolation
            frac = mag_db[i] / (mag_db[i] - mag_db[i + 1])
            return omega[i] + frac * (omega[i + 1] - omega[i])
    return float("nan")


def phase_margin(omega: np.ndarray, mag_db: np.ndarray,
                 phase_deg: np.ndarray) -> float:
    """Compute phase margin at gain crossover frequency."""
    wc = find_gain_crossover(omega, mag_db)
    if np.isnan(wc):
        return float("inf")
    # Interpolate phase at crossover
    idx = np.searchsorted(omega, wc)
    if idx >= len(omega):
        idx = len(omega) - 1
    phase_at_wc = np.interp(wc, omega, phase_deg)
    return 180.0 + phase_at_wc


# ── 5. Step Response via Euler (Closed-Loop) ──────────────────────────

def closed_loop_step(plant_num: list[float], plant_den: list[float],
                     comp_num: list[float], comp_den: list[float],
                     dt: float = 0.01, t_end: float = 10.0) -> dict:
    """
    Simulate unit step response of the closed-loop system.
    Uses state-space conversion of the open-loop C(s)*G(s) and
    feedback closure.

    For simplicity, simulates using direct polynomial division
    and Euler integration of the closed-loop transfer function.
    """
    # Open-loop: L(s) = C(s)*G(s)
    ol_num = np.polymul(comp_num, plant_num)
    ol_den = np.polymul(comp_den, plant_den)

    # Closed-loop: T(s) = L(s) / (1 + L(s)) = ol_num / (ol_den + ol_num)
    cl_num = ol_num
    # Pad to same length
    max_len = max(len(ol_den), len(ol_num))
    ol_den_pad = np.pad(ol_den, (max_len - len(ol_den), 0))
    ol_num_pad = np.pad(ol_num, (max_len - len(ol_num), 0))
    cl_den = ol_den_pad + ol_num_pad

    # Normalize
    cl_num = cl_num / cl_den[0]
    cl_den = cl_den / cl_den[0]

    # Simulate via companion-form state space
    n = len(cl_den) - 1
    A = np.zeros((n, n))
    for i in range(n - 1):
        A[i, i + 1] = 1.0
    for i in range(n):
        A[n - 1, i] = -cl_den[n - i]

    B = np.zeros(n)
    B[0] = 1.0

    # Pad cl_num to length n+1
    cl_num_pad = np.pad(cl_num, (n + 1 - len(cl_num), 0))
    C = np.zeros(n)
    for i in range(n):
        C[i] = cl_num_pad[n - i] - cl_num_pad[0] * cl_den[n - i]

    D = cl_num_pad[0]

    steps = int(t_end / dt)
    t = np.linspace(0, t_end, steps)
    x = np.zeros(n)
    y = np.zeros(steps)

    for i in range(steps):
        u_in = 1.0  # unit step
        y[i] = C @ x + D * u_in
        x = x + dt * (A @ x + B * u_in)

    return {"t": t, "y": y}


# ── Demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    omega = np.logspace(-2, 3, 5000)

    # Plant: G(s) = 10 / [s(s+1)(s+5)]
    plant_num = [10.0]
    plant_den = np.polymul([1, 0], np.polymul([1, 1], [1, 5])).tolist()

    # Uncompensated Bode data
    bode_unc = bode_data(plant_num, plant_den, omega)
    pm_unc = phase_margin(omega, bode_unc["mag_db"], bode_unc["phase_deg"])
    wc_unc = find_gain_crossover(omega, bode_unc["mag_db"])

    print("=== Uncompensated Plant ===")
    print(f"  G(s) = 10 / [s(s+1)(s+5)]")
    print(f"  Gain crossover: {wc_unc:.2f} rad/s")
    print(f"  Phase margin: {pm_unc:.1f}°")

    # Design lead compensator for PM ≥ 45°
    plant_gain_at_3 = abs(eval_tf(plant_num, plant_den, np.array([3.0]))[0])
    lead = lead_compensator(45.0, pm_unc, wc_new=3.0,
                            plant_gain_at_wc=plant_gain_at_3)

    print(f"\n=== Lead Compensator ===")
    print(f"  α = {lead['alpha']:.4f}")
    print(f"  Zero: s = {-lead['zero']:.2f}")
    print(f"  Pole: s = {-lead['pole']:.2f}")
    print(f"  Gain Kc: {lead['Kc']:.4f}")
    print(f"  Max phase lead: {lead['phi_max_deg']:.1f}°")

    # Verify compensated phase margin
    comp_num = np.polymul(lead["num"], plant_num).tolist()
    comp_den = np.polymul(lead["den"], plant_den).tolist()
    bode_comp = bode_data(comp_num, comp_den, omega)
    pm_comp = phase_margin(omega, bode_comp["mag_db"], bode_comp["phase_deg"])
    print(f"  Compensated phase margin: {pm_comp:.1f}°")

    # Lag compensator for additional low-frequency gain
    lag = lag_compensator(gain_deficit_db=10.0, wc=3.0)
    print(f"\n=== Lag Compensator ===")
    print(f"  β = {lag['beta']:.2f}")
    print(f"  Zero: s = {-lag['zero']:.4f}")
    print(f"  Pole: s = {-lag['pole']:.4f}")

    # Combined lead-lag
    ll = lead_lag_compensator(lead, lag)
    print(f"\n=== Lead-Lag Combined ===")
    print(f"  Zeros: {ll['zeros']}")
    print(f"  Poles: {ll['poles']}")

    # Closed-loop step response comparison
    print(f"\n=== Step Response (final values) ===")
    step_unc = closed_loop_step(plant_num, plant_den,
                                [1.0], [1.0], t_end=15.0)
    step_lead = closed_loop_step(plant_num, plant_den,
                                 lead["num"], lead["den"], t_end=15.0)
    print(f"  Uncompensated y(∞): {step_unc['y'][-1]:.4f}")
    print(f"  Lead-compensated y(∞): {step_lead['y'][-1]:.4f}")
