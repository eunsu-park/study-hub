"""
Control Theory — Lesson 8: Nyquist Stability Criterion

Demonstrates:
1. Nyquist contour mapping for rational transfer functions
2. Encirclement counting (winding number)
3. Nyquist stability criterion application
4. Cauchy's argument principle verification
5. Conditional stability detection
"""
import numpy as np


# ── 1. Nyquist Plot Data Generation ───────────────────────────────────

def nyquist_plot(num: list[float], den: list[float],
                 omega_max: float = 1000.0,
                 n_points: int = 10000) -> dict:
    """
    Generate Nyquist plot data for L(jω), ω ∈ (0, ω_max].

    Returns real and imaginary parts for the positive-frequency
    portion. The negative-frequency portion is the complex conjugate.

    Args:
        num: numerator polynomial coefficients
        den: denominator polynomial coefficients
        omega_max: upper frequency limit [rad/s]
        n_points: number of frequency points

    Returns:
        dict with 'omega', 'real', 'imag', 'mag', 'phase_deg'
    """
    omega = np.logspace(-4, np.log10(omega_max), n_points)
    s = 1j * omega
    H = np.polyval(num, s) / np.polyval(den, s)

    return {
        "omega": omega,
        "real": H.real,
        "imag": H.imag,
        "mag": np.abs(H),
        "phase_deg": np.degrees(np.angle(H)),
    }


# ── 2. Winding Number Computation ────────────────────────────────────

def winding_number(real: np.ndarray, imag: np.ndarray,
                   point: tuple[float, float] = (-1, 0)) -> int:
    """
    Count the number of clockwise encirclements of a point
    by the Nyquist contour.

    Uses the angle accumulation method: the total angle swept
    by the vector from the point to L(jω) as ω goes from 0 to ∞
    (and back via the conjugate) gives 2π × N_encirclements.

    Args:
        real: real part of L(jω) for ω > 0
        imag: imaginary part of L(jω) for ω > 0
        point: the critical point (default: -1+j0)

    Returns:
        N: number of clockwise encirclements (positive = clockwise)
    """
    px, py = point

    # Positive frequency: ω from 0+ to +∞
    dx_pos = real - px
    dy_pos = imag - py
    angles_pos = np.arctan2(dy_pos, dx_pos)

    # Negative frequency: conjugate (ω from +∞ to 0+)
    dx_neg = real[::-1] - px
    dy_neg = -imag[::-1] - py
    angles_neg = np.arctan2(dy_neg, dx_neg)

    # Concatenate full contour
    angles = np.concatenate([angles_pos, angles_neg])

    # Compute total angle change using unwrapped angles
    unwrapped = np.unwrap(angles)
    total_angle = unwrapped[-1] - unwrapped[0]

    # Clockwise encirclements: negative total angle / (2π)
    N = -int(np.round(total_angle / (2 * np.pi)))
    return N


# ── 3. Nyquist Stability Criterion ───────────────────────────────────

def nyquist_stability(num: list[float], den: list[float],
                      omega_max: float = 10000.0) -> dict:
    """
    Apply the Nyquist stability criterion.

    N = P - Z
    where:
      N = number of CW encirclements of -1+j0
      P = number of open-loop RHP poles
      Z = number of closed-loop RHP poles

    The closed-loop system is stable iff Z = 0, i.e., N = P.

    Returns:
        dict with P, N, Z, and stability determination
    """
    # Count open-loop RHP poles
    ol_poles = np.roots(den)
    P = int(np.sum(ol_poles.real > 0))

    # Generate Nyquist data and count encirclements
    data = nyquist_plot(num, den, omega_max)
    N = winding_number(data["real"], data["imag"])

    Z = P - N
    stable = (Z == 0)

    return {
        "open_loop_rhp_poles": P,
        "encirclements_N": N,
        "closed_loop_rhp_poles": Z,
        "closed_loop_stable": stable,
        "open_loop_poles": np.sort_complex(ol_poles).tolist(),
    }


# ── 4. Cauchy's Argument Principle ────────────────────────────────────

def cauchys_principle(num: list[float], den: list[float]) -> dict:
    """
    Verify Cauchy's argument principle for 1 + L(s).

    The characteristic polynomial is den(s) + num(s).
    Z_char = number of RHP zeros of 1+L(s) = closed-loop RHP poles.
    P_char = number of RHP poles of 1+L(s) = open-loop RHP poles.

    The winding number of L(s) around -1 equals Z_char - P_char.
    (Convention: positive = counterclockwise here)
    """
    char_poly = np.polyadd(den, num)
    char_roots = np.roots(char_poly)
    Z_char = int(np.sum(char_roots.real > 0))

    ol_poles = np.roots(den)
    P_char = int(np.sum(ol_poles.real > 0))

    return {
        "char_polynomial": char_poly.tolist(),
        "char_roots": np.sort_complex(char_roots).tolist(),
        "Z_closed_loop_rhp": Z_char,
        "P_open_loop_rhp": P_char,
        "expected_encirclements": P_char - Z_char,
    }


# ── 5. Conditional Stability Check ───────────────────────────────────

def conditional_stability_range(num: list[float], den: list[float],
                                K_values: np.ndarray) -> dict:
    """
    For L(s) = K * num(s)/den(s), find ranges of K where the
    closed-loop system is stable. Systems that are stable only
    for a bounded range of K exhibit conditional stability.

    Returns:
        dict with stable_ranges and classification
    """
    results = []
    for K in K_values:
        char_poly = np.polyadd(den, (K * np.array(num)).tolist())
        roots = np.roots(char_poly)
        n_rhp = int(np.sum(roots.real > 1e-10))
        results.append(n_rhp == 0)

    stable = np.array(results)

    # Find transitions
    ranges = []
    in_stable = False
    start_K = None
    for i, (K, s) in enumerate(zip(K_values, stable)):
        if s and not in_stable:
            start_K = K
            in_stable = True
        elif not s and in_stable:
            ranges.append((start_K, K_values[i - 1]))
            in_stable = False
    if in_stable:
        ranges.append((start_K, K_values[-1]))

    conditionally_stable = len(ranges) > 1 or (
        len(ranges) == 1 and ranges[0][0] > K_values[0] + 0.01)

    return {
        "stable_ranges": ranges,
        "conditionally_stable": conditionally_stable,
        "n_stable_ranges": len(ranges),
    }


# ── 6. Distance to Critical Point ────────────────────────────────────

def minimum_distance_to_critical(num: list[float], den: list[float],
                                 omega_max: float = 1000.0) -> dict:
    """
    Find the minimum distance from the Nyquist contour to -1+j0.

    This distance is 1/M_peak where M_peak is the resonant peak
    of the closed-loop sensitivity function.
    """
    data = nyquist_plot(num, den, omega_max)
    distances = np.sqrt((data["real"] + 1)**2 + data["imag"]**2)
    idx_min = np.argmin(distances)

    return {
        "min_distance": distances[idx_min],
        "at_omega": data["omega"][idx_min],
        "sensitivity_peak": 1.0 / distances[idx_min],
        "sensitivity_peak_db": 20 * np.log10(1.0 / distances[idx_min]),
    }


# ── Demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Example 1: Stable system — L(s) = 10 / [s(s+1)(s+5)]
    print("=== Example 1: Stable System ===")
    print("  L(s) = 10 / [s(s+1)(s+5)]")
    num1 = [10.0]
    den1 = np.polymul([1, 0], np.polymul([1, 1], [1, 5])).tolist()
    result1 = nyquist_stability(num1, den1)
    print(f"  Open-loop RHP poles (P): {result1['open_loop_rhp_poles']}")
    print(f"  Encirclements of -1 (N): {result1['encirclements_N']}")
    print(f"  Closed-loop RHP poles (Z=P-N): {result1['closed_loop_rhp_poles']}")
    print(f"  Stable: {result1['closed_loop_stable']}")

    # Verify with Cauchy
    cauchy1 = cauchys_principle(num1, den1)
    print(f"  Cauchy verification — CL RHP zeros: {cauchy1['Z_closed_loop_rhp']}")

    # Example 2: Unstable system — L(s) = 100 / [s(s+1)(s+5)]
    print("\n=== Example 2: Higher Gain (K=100) ===")
    print("  L(s) = 100 / [s(s+1)(s+5)]")
    num2 = [100.0]
    result2 = nyquist_stability(num2, den1)
    print(f"  Encirclements of -1 (N): {result2['encirclements_N']}")
    print(f"  Closed-loop RHP poles (Z): {result2['closed_loop_rhp_poles']}")
    print(f"  Stable: {result2['closed_loop_stable']}")

    cauchy2 = cauchys_principle(num2, den1)
    print(f"  Cauchy verification — CL RHP zeros: {cauchy2['Z_closed_loop_rhp']}")

    # Example 3: Open-loop unstable plant
    print("\n=== Example 3: Open-Loop Unstable ===")
    print("  L(s) = 5(s+2) / [(s-1)(s+3)]")
    num3 = [5, 10]
    den3 = np.polymul([1, -1], [1, 3]).tolist()
    result3 = nyquist_stability(num3, den3)
    print(f"  Open-loop RHP poles (P): {result3['open_loop_rhp_poles']}")
    print(f"  Encirclements of -1 (N): {result3['encirclements_N']}")
    print(f"  Closed-loop RHP poles (Z): {result3['closed_loop_rhp_poles']}")
    print(f"  Stable: {result3['closed_loop_stable']}")

    # Example 4: Minimum distance / sensitivity peak
    print("\n=== Sensitivity Peak ===")
    dist = minimum_distance_to_critical(num1, den1)
    print(f"  L(s) = 10 / [s(s+1)(s+5)]")
    print(f"  Min distance to -1+j0: {dist['min_distance']:.4f}")
    print(f"  At ω = {dist['at_omega']:.2f} rad/s")
    print(f"  Peak sensitivity: {dist['sensitivity_peak']:.2f}"
          f" ({dist['sensitivity_peak_db']:.1f} dB)")

    # Example 5: Conditional stability
    print("\n=== Conditional Stability ===")
    print("  L(s) = K(s+2) / [s(s+1)(s+3)(s+10)]")
    num5 = [1, 2]
    den5 = np.polymul(np.polymul([1, 0], [1, 1]),
                      np.polymul([1, 3], [1, 10])).tolist()
    K_arr = np.linspace(0.1, 500, 5000)
    cond = conditional_stability_range(num5, den5, K_arr)
    print(f"  Conditionally stable: {cond['conditionally_stable']}")
    print(f"  Number of stable ranges: {cond['n_stable_ranges']}")
    for lo, hi in cond["stable_ranges"]:
        print(f"    Stable for K ∈ ({lo:.1f}, {hi:.1f})")
