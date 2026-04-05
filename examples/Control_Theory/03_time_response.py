"""
Control Theory — Lesson 4: Time-Domain Analysis

Demonstrates:
1. First-order step response
2. Second-order step response (underdamped, critically damped, overdamped)
3. Time-domain specifications (tp, Mp, ts, tr)
4. Steady-state error analysis (system type, error constants)
"""
import numpy as np


# ── 1. First-Order Step Response ─────────────────────────────────────────

def first_order_step(K: float, tau: float, t: np.ndarray) -> np.ndarray:
    """Step response of G(s) = K/(τs+1)."""
    return K * (1 - np.exp(-t / tau))


# ── 2. Second-Order Step Response ────────────────────────────────────────

def second_order_step(wn: float, zeta: float, t: np.ndarray) -> np.ndarray:
    """Unit step response of G(s) = ωn²/(s²+2ζωn·s+ωn²)."""
    if zeta < 1:  # underdamped
        wd = wn * np.sqrt(1 - zeta**2)
        sigma = zeta * wn
        phi = np.arccos(zeta)
        return 1 - np.exp(-sigma * t) / np.sqrt(1 - zeta**2) * np.sin(wd * t + phi)
    elif abs(zeta - 1) < 1e-10:  # critically damped
        return 1 - (1 + wn * t) * np.exp(-wn * t)
    else:  # overdamped
        s1 = -zeta * wn + wn * np.sqrt(zeta**2 - 1)
        s2 = -zeta * wn - wn * np.sqrt(zeta**2 - 1)
        return 1 + (s1 * np.exp(s2 * t) - s2 * np.exp(s1 * t)) / (s2 - s1)


# ── 3. Time-Domain Specifications ────────────────────────────────────────

def compute_specs(wn: float, zeta: float) -> dict:
    """Compute time-domain specs for underdamped second-order system."""
    if zeta >= 1 or zeta <= 0:
        raise ValueError("Specs are for underdamped case: 0 < ζ < 1")

    wd = wn * np.sqrt(1 - zeta**2)
    sigma = zeta * wn

    tp = np.pi / wd
    Mp = np.exp(-np.pi * zeta / np.sqrt(1 - zeta**2))
    ts_2 = 4 / sigma
    ts_5 = 3 / sigma
    tr = 1.8 / wn  # approximate

    return {
        "peak_time": tp,
        "overshoot_pct": Mp * 100,
        "settling_2pct": ts_2,
        "settling_5pct": ts_5,
        "rise_time_approx": tr,
    }


# ── 4. Steady-State Error ────────────────────────────────────────────────

def steady_state_error(num: list[float], den: list[float],
                       input_type: str = "step") -> float:
    """
    Compute steady-state error for unity-feedback system.

    Args:
        num, den: open-loop G(s) polynomial coefficients [highest ... lowest]
        input_type: "step", "ramp", or "parabola"
    """
    # Evaluate limit using polynomial evaluation at s → 0
    # G(s) = num(s)/den(s)
    num = np.array(num, dtype=float)
    den = np.array(den, dtype=float)

    # Count integrators (factors of s in denominator)
    # = number of trailing zeros in den
    system_type = 0
    d = den.copy()
    while len(d) > 1 and abs(d[-1]) < 1e-12:
        system_type += 1
        d = d[:-1]

    if input_type == "step":
        if system_type >= 1:
            return 0.0
        # Kp = lim s→0 G(s) = num[-1]/den[-1]
        Kp = num[-1] / den[-1]
        return 1 / (1 + Kp)

    elif input_type == "ramp":
        if system_type >= 2:
            return 0.0
        if system_type == 0:
            return float('inf')
        # Kv = lim s→0 s*G(s)
        # For Type 1: G(s) = N(s)/(s*D'(s)), so s*G(s)|s=0 = N(0)/D'(0)
        # Multiply num by [1, 0] (multiply by s), then evaluate at 0
        sG_num = np.append(num, 0)  # multiply num by s
        Kv = np.polyval(sG_num, 0) / np.polyval(den, 0) if abs(np.polyval(den, 0)) > 1e-12 else float('inf')
        # For type 1, need to be more careful
        # Kv = lim s→0 s * num(s)/den(s)
        eps = 1e-8
        Kv = abs(eps * np.polyval(num, eps) / np.polyval(den, eps))
        return 1 / Kv if Kv > 0 else float('inf')

    elif input_type == "parabola":
        if system_type >= 3:
            return 0.0
        if system_type < 2:
            return float('inf')
        eps = 1e-8
        Ka = abs(eps**2 * np.polyval(num, eps) / np.polyval(den, eps))
        return 1 / Ka if Ka > 0 else float('inf')

    raise ValueError(f"Unknown input_type: {input_type}")


# ── Demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t = np.linspace(0, 10, 2000)

    # First-order
    print("=== First-Order Step Response ===")
    K, tau = 2.0, 0.5
    y = first_order_step(K, tau, t)
    idx_tau = np.argmin(np.abs(t - tau))
    print(f"  G(s) = {K}/(s·{tau}+1)")
    print(f"  At t = τ = {tau}: y = {y[idx_tau]:.3f} (expect {K * 0.632:.3f})")
    print(f"  Final value: {y[-1]:.3f} (expect {K})")

    # Second-order with different damping
    print("\n=== Second-Order Step Responses ===")
    wn = 5.0
    for zeta in [0.2, 0.5, 0.707, 1.0, 2.0]:
        y = second_order_step(wn, zeta, t)
        peak = np.max(y)
        overshoot = (peak - 1.0) * 100
        case = ("underdamped" if zeta < 1
                else "critically damped" if abs(zeta - 1) < 0.01
                else "overdamped")
        print(f"  ζ = {zeta:.3f} ({case:>20s}): "
              f"peak = {peak:.3f}, overshoot = {overshoot:.1f}%")

    # Specifications
    print("\n=== Time-Domain Specifications (ωn=5, ζ=0.5) ===")
    specs = compute_specs(wn=5, zeta=0.5)
    for name, val in specs.items():
        print(f"  {name}: {val:.3f}")

    # Steady-state error
    print("\n=== Steady-State Error Analysis ===")
    # Type 0: G(s) = 10/(s+5)
    print("  Type 0: G(s) = 10/(s+5)")
    print(f"    Step error:     {steady_state_error([10], [1, 5], 'step'):.4f}")
    print(f"    Ramp error:     {steady_state_error([10], [1, 5], 'ramp')}")

    # Type 1: G(s) = 100/[s(s+5)]
    print("  Type 1: G(s) = 100/[s(s+5)]")
    print(f"    Step error:     {steady_state_error([100], [1, 5, 0], 'step'):.4f}")
    print(f"    Ramp error:     {steady_state_error([100], [1, 5, 0], 'ramp'):.4f}")
    print(f"    Parabola error: {steady_state_error([100], [1, 5, 0], 'parabola')}")
