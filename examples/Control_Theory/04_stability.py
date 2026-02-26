"""
Control Theory — Lesson 5: Stability Analysis

Demonstrates:
1. Routh-Hurwitz criterion (Routh array construction)
2. Stability range finding (gain K for stability)
3. Pole analysis and stability verification
"""
import numpy as np


def routh_array(coeffs: list[float]) -> list[list[float]]:
    """
    Construct the Routh array for a polynomial.

    Args:
        coeffs: polynomial coefficients [a_n, a_{n-1}, ..., a_1, a_0]

    Returns:
        List of rows, each row is a list of first-column and subsequent entries.
    """
    n = len(coeffs) - 1  # degree
    if n < 1:
        return [coeffs]

    # Number of columns
    cols = (n + 2) // 2

    # Initialize array
    array = np.zeros((n + 1, cols))

    # Fill first two rows
    array[0, :] = [coeffs[i] if i < len(coeffs) else 0
                   for i in range(0, 2 * cols, 2)]
    array[1, :] = [coeffs[i] if i < len(coeffs) else 0
                   for i in range(1, 2 * cols, 2)]

    # Compute remaining rows
    eps = 1e-10  # for zero first-column element
    for i in range(2, n + 1):
        pivot = array[i - 1, 0]
        if abs(pivot) < eps:
            pivot = eps  # replace zero with small positive number

        for j in range(cols - 1):
            array[i, j] = (pivot * array[i - 2, j + 1]
                          - array[i - 1, j + 1] * array[i - 2, 0]) / pivot
        # Last column is always zero from the formula

    return array


def count_sign_changes(array: np.ndarray) -> int:
    """Count sign changes in the first column of the Routh array."""
    first_col = array[:, 0]
    changes = 0
    for i in range(1, len(first_col)):
        if first_col[i - 1] * first_col[i] < 0:
            changes += 1
    return changes


def routh_stability(coeffs: list[float]) -> dict:
    """
    Analyze stability using Routh-Hurwitz criterion.

    Returns dict with: array, sign_changes, is_stable, first_column
    """
    array = routh_array(coeffs)
    first_col = array[:, 0].tolist()
    changes = count_sign_changes(array)
    stable = changes == 0 and all(c > 0 for c in first_col)

    return {
        "array": array,
        "first_column": first_col,
        "sign_changes": changes,
        "rhp_poles": changes,
        "is_stable": stable,
    }


def find_stability_range(plant_den: list[float], K_range=(0, 100),
                         steps=10000) -> tuple[float, float]:
    """
    Find the range of K > 0 for which 1 + K*G(s) = 0 is stable.

    Args:
        plant_den: denominator coefficients of G(s) (open-loop, without K)
                   The characteristic equation is: den(s) + K * num(s) = 0
                   For G(s) = K/den(s), char eq = den(s) + K = 0
    """
    K_lo, K_hi = K_range
    K_values = np.linspace(K_lo + 0.001, K_hi, steps)

    stable_min = None
    stable_max = None

    for K in K_values:
        # Characteristic polynomial: add K to the constant term
        char_poly = list(plant_den)
        char_poly[-1] += K
        result = routh_stability(char_poly)
        if result["is_stable"]:
            if stable_min is None:
                stable_min = K
            stable_max = K
        elif stable_min is not None and stable_max is not None:
            break  # Found the boundary

    return (stable_min, stable_max) if stable_min else (None, None)


# ── Demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Example 1: s^4 + 2s^3 + 3s^2 + 4s + 5
    print("=== Example 1: s⁴ + 2s³ + 3s² + 4s + 5 ===")
    result = routh_stability([1, 2, 3, 4, 5])
    print(f"  First column: {[f'{x:.2f}' for x in result['first_column']]}")
    print(f"  Sign changes: {result['sign_changes']}")
    print(f"  RHP poles: {result['rhp_poles']}")
    print(f"  Stable: {result['is_stable']}")
    # Verify with numpy
    roots = np.roots([1, 2, 3, 4, 5])
    rhp = sum(1 for r in roots if r.real > 0)
    print(f"  Actual roots: {np.sort_complex(roots)}")
    print(f"  Actual RHP poles: {rhp}")

    # Example 2: Stable polynomial s^3 + 6s^2 + 11s + 6 = (s+1)(s+2)(s+3)
    print("\n=== Example 2: s³ + 6s² + 11s + 6 ===")
    result = routh_stability([1, 6, 11, 6])
    print(f"  First column: {[f'{x:.2f}' for x in result['first_column']]}")
    print(f"  Stable: {result['is_stable']}")

    # Example 3: Finding stability range for K
    # G(s) = K / [s(s+1)(s+5)]  →  char eq: s³ + 6s² + 5s + K = 0
    print("\n=== Stability Range: G(s) = K/[s(s+1)(s+5)] ===")
    # Analytical: Routh array gives 0 < K < 30
    K_min, K_max = find_stability_range([1, 6, 5, 0], K_range=(0, 50))
    print(f"  Stable for K ∈ ({K_min:.2f}, {K_max:.2f})")
    print(f"  Expected: (0, 30)")

    # Verify at boundary
    print("\n  Verification at K = 30:")
    result = routh_stability([1, 6, 5, 30])
    print(f"    First column: {[f'{x:.4f}' for x in result['first_column']]}")
    roots_30 = np.roots([1, 6, 5, 30])
    print(f"    Roots at K=30: {np.sort_complex(roots_30)}")

    # Example 4: System with parameter
    print("\n=== Stability with Parameter ===")
    print("  s³ + 10s² + 31s + 30")
    result = routh_stability([1, 10, 31, 30])
    print(f"  Stable: {result['is_stable']}")
    roots = np.roots([1, 10, 31, 30])
    print(f"  Poles: {np.sort(roots.real)}")
    print(f"  All Re(p) < -1? {all(r.real < -1 for r in roots)}")
