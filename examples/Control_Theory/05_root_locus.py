"""
Control Theory — Lesson 6: Root Locus Method

Demonstrates:
1. Root locus computation (brute-force pole tracking)
2. Asymptote calculation
3. Breakaway point finding
4. Gain at a specific pole location
"""
import numpy as np


def compute_root_locus(num: list[float], den: list[float],
                       K_range: np.ndarray) -> np.ndarray:
    """
    Compute root locus by finding closed-loop poles for each K.

    Characteristic equation: den(s) + K * num(s) = 0

    Args:
        num: numerator coefficients of G(s)H(s) [highest ... lowest]
        den: denominator coefficients of G(s)H(s)
        K_range: array of gain values

    Returns:
        Array of shape (len(K_range), degree) with pole locations.
    """
    num = np.array(num, dtype=float)
    den = np.array(den, dtype=float)

    # Pad shorter polynomial
    max_len = max(len(num), len(den))
    num_padded = np.zeros(max_len)
    den_padded = np.zeros(max_len)
    num_padded[max_len - len(num):] = num
    den_padded[max_len - len(den):] = den

    degree = len(den) - 1
    poles = np.zeros((len(K_range), degree), dtype=complex)

    for i, K in enumerate(K_range):
        char_poly = den_padded + K * num_padded
        # Trim leading zeros
        char_poly = np.trim_zeros(char_poly, 'f')
        if len(char_poly) <= 1:
            continue
        r = np.roots(char_poly)
        # Sort for consistent ordering
        r = np.sort_complex(r)
        poles[i, :len(r)] = r

    return poles


def asymptotes(open_loop_poles: list[complex],
               open_loop_zeros: list[complex]) -> dict:
    """
    Compute root locus asymptote angles and centroid.

    Returns dict with: angles (degrees), centroid (real number), n_minus_m
    """
    n = len(open_loop_poles)
    m = len(open_loop_zeros)
    diff = n - m

    if diff <= 0:
        return {"angles": [], "centroid": 0, "n_minus_m": diff}

    angles = [(2 * k + 1) * 180 / diff for k in range(diff)]
    centroid = (sum(p.real for p in open_loop_poles)
                - sum(z.real for z in open_loop_zeros)) / diff

    return {"angles": angles, "centroid": centroid, "n_minus_m": diff}


def breakaway_points(num: list[float], den: list[float]) -> np.ndarray:
    """
    Find breakaway/break-in points: solve dK/ds = 0.

    K(s) = -den(s)/num(s), so dK/ds = 0 means
    num(s)*den'(s) - den(s)*num'(s) = 0
    """
    num = np.array(num, dtype=float)
    den = np.array(den, dtype=float)

    num_d = np.polyder(num)
    den_d = np.polyder(den)

    # Numerator of dK/ds = 0:  num * den' - den * num'
    poly = np.polysub(np.polymul(num, den_d), np.polymul(den, num_d))
    candidates = np.roots(poly)

    # Filter: keep only real candidates on the real axis
    real_candidates = []
    for c in candidates:
        if abs(c.imag) < 1e-6:
            s = c.real
            # Check if this point is on the locus (K > 0)
            den_val = np.polyval(den, s)
            num_val = np.polyval(num, s)
            if abs(num_val) > 1e-10:
                K = -den_val / num_val
                if K > 0:
                    real_candidates.append((s, K))

    return real_candidates


def gain_at_point(num: list[float], den: list[float],
                  s: complex) -> float:
    """Compute K at a point on the root locus: K = -den(s)/num(s)."""
    return abs(np.polyval(den, s) / np.polyval(num, s))


# ── Demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Example: G(s) = K / [s(s+1)(s+3)]
    print("=== Root Locus: G(s) = K/[s(s+1)(s+3)] ===")
    num = [1]
    den = [1, 4, 3, 0]  # s³ + 4s² + 3s

    # Open-loop poles and zeros
    ol_poles = np.roots(den)
    ol_zeros = np.roots(num) if len(num) > 1 else []
    print(f"  Open-loop poles: {np.sort(ol_poles.real)}")
    print(f"  Open-loop zeros: {list(ol_zeros)}")

    # Asymptotes
    asym = asymptotes(list(ol_poles), list(ol_zeros))
    print(f"\n  Asymptotes:")
    print(f"    n - m = {asym['n_minus_m']}")
    print(f"    Angles: {asym['angles']}°")
    print(f"    Centroid: {asym['centroid']:.3f}")

    # Breakaway points
    bp = breakaway_points(num, den)
    print(f"\n  Breakaway points:")
    for s_val, K_val in bp:
        print(f"    s = {s_val:.3f}, K = {K_val:.3f}")

    # Imaginary axis crossing (from Routh: K_crit = 12, ω = √3)
    print(f"\n  Imaginary axis crossing:")
    K_crit = 12
    roots_crit = np.roots([1, 4, 3, K_crit])
    print(f"    K_crit = {K_crit}")
    print(f"    Roots: {np.sort_complex(roots_crit)}")

    # Root locus computation
    K_values = np.linspace(0.01, 15, 500)
    poles = compute_root_locus(num, den, K_values)

    print(f"\n  Root locus sample points:")
    for K in [0.1, 1, 5, 10, 12]:
        idx = np.argmin(np.abs(K_values - K))
        p = poles[idx]
        print(f"    K={K:5.1f}: poles = {np.sort_complex(p)}")

    # Gain at a desired pole location
    print(f"\n  Gain selection:")
    s_desired = -1 + 2j  # desired pole
    K_desired = gain_at_point(num, den, s_desired)
    print(f"    Desired pole: s = {s_desired}")
    print(f"    Required K = {K_desired:.3f}")
    # Verify
    cl_roots = np.roots(np.polyadd(den, [K_desired * c for c in num]))
    print(f"    CL poles at K={K_desired:.3f}: {np.sort_complex(cl_roots)}")
