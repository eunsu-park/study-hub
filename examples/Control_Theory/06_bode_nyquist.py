"""
Control Theory — Lessons 7-8: Bode Plots and Nyquist Stability

Demonstrates:
1. Frequency response computation
2. Bode magnitude and phase calculation
3. Gain margin and phase margin
4. Nyquist plot data generation
"""
import numpy as np


def frequency_response(num: list[float], den: list[float],
                       omega: np.ndarray) -> np.ndarray:
    """Evaluate G(jω) for an array of frequencies."""
    s = 1j * omega
    G = np.array([np.polyval(num, si) / np.polyval(den, si) for si in s])
    return G


def bode_data(num: list[float], den: list[float],
              omega: np.ndarray) -> dict:
    """
    Compute Bode plot data.

    Returns dict with: omega, magnitude_db, phase_deg, G_jw
    """
    G = frequency_response(num, den, omega)
    mag_db = 20 * np.log10(np.abs(G))
    phase_deg = np.degrees(np.unwrap(np.angle(G)))

    return {
        "omega": omega,
        "magnitude_db": mag_db,
        "phase_deg": phase_deg,
        "G_jw": G,
    }


def stability_margins(num: list[float], den: list[float],
                      omega: np.ndarray = None) -> dict:
    """
    Compute gain margin and phase margin from frequency response.

    Returns dict with: GM_db, PM_deg, omega_gc, omega_pc
    """
    if omega is None:
        omega = np.logspace(-3, 3, 10000)

    data = bode_data(num, den, omega)
    mag = np.abs(data["G_jw"])
    phase = data["phase_deg"]

    # Gain crossover: |G(jω)| = 1  (0 dB)
    idx_gc = None
    for i in range(len(mag) - 1):
        if (mag[i] - 1) * (mag[i + 1] - 1) < 0:
            # Linear interpolation
            frac = (1 - mag[i]) / (mag[i + 1] - mag[i])
            omega_gc = omega[i] + frac * (omega[i + 1] - omega[i])
            phase_gc = phase[i] + frac * (phase[i + 1] - phase[i])
            idx_gc = i
            break

    # Phase crossover: ∠G(jω) = -180°
    idx_pc = None
    for i in range(len(phase) - 1):
        if (phase[i] + 180) * (phase[i + 1] + 180) < 0:
            frac = (-180 - phase[i]) / (phase[i + 1] - phase[i])
            omega_pc = omega[i] + frac * (omega[i + 1] - omega[i])
            mag_pc = mag[i] + frac * (mag[i + 1] - mag[i])
            idx_pc = i
            break

    result = {}

    if idx_gc is not None:
        result["omega_gc"] = omega_gc
        result["PM_deg"] = 180 + phase_gc
    else:
        result["omega_gc"] = None
        result["PM_deg"] = float('inf')  # gain never crosses 0 dB

    if idx_pc is not None:
        result["omega_pc"] = omega_pc
        result["GM_db"] = -20 * np.log10(mag_pc)
    else:
        result["omega_pc"] = None
        result["GM_db"] = float('inf')  # phase never crosses -180°

    return result


def nyquist_data(num: list[float], den: list[float],
                 omega: np.ndarray) -> dict:
    """
    Compute Nyquist plot data (positive frequencies only).
    The negative frequency portion is the complex conjugate.
    """
    G = frequency_response(num, den, omega)

    return {
        "omega": omega,
        "real": G.real,
        "imag": G.imag,
        "G_jw": G,
    }


def count_encirclements(real: np.ndarray, imag: np.ndarray,
                        point: tuple = (-1, 0)) -> int:
    """
    Count net clockwise encirclements of a point by a closed curve.
    Uses the winding number formula.
    """
    # Include both positive and negative frequency (mirror)
    full_real = np.concatenate([real, real[::-1]])
    full_imag = np.concatenate([imag, -imag[::-1]])

    px, py = point
    dx = full_real - px
    dy = full_imag - py

    # Compute angle changes
    angles = np.arctan2(dy, dx)
    d_angles = np.diff(angles)

    # Handle angle wrapping
    d_angles = np.where(d_angles > np.pi, d_angles - 2 * np.pi, d_angles)
    d_angles = np.where(d_angles < -np.pi, d_angles + 2 * np.pi, d_angles)

    winding_number = np.sum(d_angles) / (2 * np.pi)
    # Clockwise = negative winding number in math convention
    return -int(np.round(winding_number))


# ── Demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    omega = np.logspace(-2, 3, 5000)

    # Example 1: G(s) = 100(s+1) / [s(s+10)]
    print("=== Bode Plot: G(s) = 100(s+1)/[s(s+10)] ===")
    num = [100, 100]   # 100(s+1) = 100s + 100
    den = [1, 10, 0]   # s(s+10) = s² + 10s
    data = bode_data(num, den, omega)
    margins = stability_margins(num, den, omega)
    print(f"  Gain margin: {margins['GM_db']:.1f} dB "
          f"at ω = {margins.get('omega_pc', 'N/A')}")
    print(f"  Phase margin: {margins['PM_deg']:.1f}° "
          f"at ω = {margins.get('omega_gc', 'N/A'):.2f}")

    # Example 2: G(s) = K / [s(s+1)(s+5)]
    print("\n=== Stability Margins: G(s) = K/[s(s+1)(s+5)] ===")
    for K in [1, 5, 10, 20, 30]:
        num_k = [K]
        den_k = [1, 6, 5, 0]
        m = stability_margins(num_k, den_k, omega)
        pm = m['PM_deg']
        gm = m['GM_db']
        stable = pm > 0 and gm > 0
        print(f"  K={K:3d}: GM={gm:6.1f} dB, PM={pm:6.1f}°, "
              f"stable={'Yes' if stable else 'No'}")

    # Nyquist plot
    print("\n=== Nyquist Plot: G(s) = 10/[s(s+1)(s+2)] ===")
    num_ny = [10]
    den_ny = [1, 3, 2, 0]
    omega_ny = np.logspace(-2, 2, 5000)
    ny = nyquist_data(num_ny, den_ny, omega_ny)

    # Find real-axis crossing
    sign_changes = np.where(np.diff(np.sign(ny["imag"])))[0]
    for idx in sign_changes:
        if ny["real"][idx] < 0:  # negative real axis crossing
            print(f"  Real-axis crossing at ω ≈ {omega_ny[idx]:.2f}: "
                  f"G(jω) ≈ {ny['real'][idx]:.4f}")

    N = count_encirclements(ny["real"], ny["imag"])
    print(f"  Encirclements of (-1, 0): N = {N}")
    P = 0  # open-loop stable (pole at origin handled by indentation)
    Z = N + P
    print(f"  P (OL RHP poles) = {P}")
    print(f"  Z (CL RHP poles) = N + P = {Z}")
    print(f"  Closed-loop stable: {'Yes' if Z == 0 else 'No'}")

    # Verify with actual poles
    char_poly = np.polyadd(den_ny, num_ny)
    cl_poles = np.roots(char_poly)
    print(f"  Actual CL poles: {np.sort_complex(cl_poles)}")
