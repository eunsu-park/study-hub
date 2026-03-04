"""
Regression and ANOVA

Demonstrates:
1. Simple Linear Regression (OLS)
2. Multiple Linear Regression
3. R² and Residual Analysis
4. One-Way ANOVA

Theory:
- OLS: β̂₁ = Σ(xi-x̄)(yi-ȳ) / Σ(xi-x̄)²
- R² = 1 - SSE/SST = explained variance ratio
- ANOVA: F = MSB/MSW, compares between-group to within-group variance

Adapted from Probability and Statistics Lesson 17.
"""

import math
import random


# ─────────────────────────────────────────────────
# 1. SIMPLE LINEAR REGRESSION
# ─────────────────────────────────────────────────

def simple_linear_regression(x: list[float],
                              y: list[float]) -> dict:
    """OLS: y = β₀ + β₁x."""
    n = len(x)
    x_bar = sum(x) / n
    y_bar = sum(y) / n

    ss_xy = sum((xi - x_bar) * (yi - y_bar) for xi, yi in zip(x, y))
    ss_xx = sum((xi - x_bar) ** 2 for xi in x)

    beta1 = ss_xy / ss_xx
    beta0 = y_bar - beta1 * x_bar

    # Predictions and residuals
    y_hat = [beta0 + beta1 * xi for xi in x]
    residuals = [yi - yhi for yi, yhi in zip(y, y_hat)]

    # R²
    ss_res = sum(r ** 2 for r in residuals)
    ss_tot = sum((yi - y_bar) ** 2 for yi in y)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Standard error of β₁
    s2 = ss_res / (n - 2) if n > 2 else 0
    se_beta1 = math.sqrt(s2 / ss_xx) if ss_xx > 0 else 0

    return {
        "beta0": beta0, "beta1": beta1,
        "r_squared": r_squared,
        "residuals": residuals,
        "se_beta1": se_beta1,
        "y_hat": y_hat,
    }


def demo_simple_regression():
    print("=" * 60)
    print("  Simple Linear Regression")
    print("=" * 60)

    # Study hours vs exam score
    random.seed(42)
    hours = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    scores = [45 + 5 * h + random.gauss(0, 5) for h in hours]

    result = simple_linear_regression(hours, scores)

    print(f"\n  Study hours vs exam score (n={len(hours)}):")
    print(f"  ŷ = {result['beta0']:.2f} + {result['beta1']:.2f}x")
    print(f"  R² = {result['r_squared']:.4f}")
    print(f"  SE(β₁) = {result['se_beta1']:.4f}")

    # t-test for β₁ ≠ 0
    t = result['beta1'] / result['se_beta1'] if result['se_beta1'] > 0 else 0
    print(f"  t-stat for H₀: β₁=0: {t:.3f}")

    # Scatter with fit line
    print(f"\n  {'Hours':>6} {'Score':>8} {'Predicted':>10} {'Residual':>10}")
    print(f"  {'─' * 36}")
    for h, s, yh, r in zip(hours, scores, result['y_hat'], result['residuals']):
        print(f"  {h:>6} {s:>8.1f} {yh:>10.1f} {r:>10.1f}")


# ─────────────────────────────────────────────────
# 2. MULTIPLE LINEAR REGRESSION
# ─────────────────────────────────────────────────

def mat_multiply(A: list[list[float]],
                  B: list[list[float]]) -> list[list[float]]:
    """Matrix multiplication."""
    rows_a, cols_a = len(A), len(A[0])
    cols_b = len(B[0])
    result = [[0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += A[i][k] * B[k][j]
    return result


def mat_transpose(A: list[list[float]]) -> list[list[float]]:
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]


def solve_2x2(A: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """Solve Ax = b for 2x2 A."""
    det = A[0][0]*A[1][1] - A[0][1]*A[1][0]
    inv = [[A[1][1]/det, -A[0][1]/det],
           [-A[1][0]/det, A[0][0]/det]]
    return mat_multiply(inv, b)


def demo_multiple_regression():
    print("\n" + "=" * 60)
    print("  Multiple Linear Regression (2 predictors)")
    print("=" * 60)

    random.seed(42)
    n = 20
    # y = 10 + 3*x1 + 2*x2 + noise
    x1 = [random.uniform(1, 10) for _ in range(n)]
    x2 = [random.uniform(1, 10) for _ in range(n)]
    y = [10 + 3*a + 2*b + random.gauss(0, 3) for a, b in zip(x1, x2)]

    # Build design matrix [1, x1, x2] and solve via normal equations
    # For simplicity, use two separate simple regressions to show concept
    y_bar = sum(y) / n
    ss_tot = sum((yi - y_bar)**2 for yi in y)

    # Using normal equations: β = (XᵀX)⁻¹Xᵀy (show for 2 predictors)
    # Center variables for illustration
    x1_bar = sum(x1) / n
    x2_bar = sum(x2) / n

    s11 = sum((a - x1_bar)**2 for a in x1)
    s22 = sum((b - x2_bar)**2 for b in x2)
    s12 = sum((a - x1_bar)*(b - x2_bar) for a, b in zip(x1, x2))
    s1y = sum((a - x1_bar)*(yi - y_bar) for a, yi in zip(x1, y))
    s2y = sum((b - x2_bar)*(yi - y_bar) for b, yi in zip(x2, y))

    A = [[s11, s12], [s12, s22]]
    b_vec = [[s1y], [s2y]]
    betas = solve_2x2(A, b_vec)
    b1, b2 = betas[0][0], betas[1][0]
    b0 = y_bar - b1 * x1_bar - b2 * x2_bar

    # R²
    y_hat = [b0 + b1*a + b2*b for a, b in zip(x1, x2)]
    ss_res = sum((yi - yh)**2 for yi, yh in zip(y, y_hat))
    r2 = 1 - ss_res / ss_tot

    print(f"\n  True: y = 10 + 3x₁ + 2x₂ + ε")
    print(f"  Estimated: ŷ = {b0:.2f} + {b1:.2f}x₁ + {b2:.2f}x₂")
    print(f"  R² = {r2:.4f}")
    print(f"  Adjusted R² = {1 - (1-r2)*(n-1)/(n-3):.4f}")


# ─────────────────────────────────────────────────
# 3. R² AND RESIDUAL ANALYSIS
# ─────────────────────────────────────────────────

def demo_residuals():
    print("\n" + "=" * 60)
    print("  R² and Residual Diagnostics")
    print("=" * 60)

    random.seed(42)
    n = 30
    x = [i for i in range(1, n + 1)]
    # Linear case
    y_lin = [2 + 0.5 * xi + random.gauss(0, 2) for xi in x]
    # Quadratic case (wrong model)
    y_quad = [2 + 0.01 * xi**2 + random.gauss(0, 2) for xi in x]

    for name, y_data in [("Linear data", y_lin), ("Quadratic data (linear fit)", y_quad)]:
        result = simple_linear_regression(x, y_data)
        resid = result['residuals']

        # Residual statistics
        mean_r = sum(resid) / len(resid)
        var_r = sum((r - mean_r)**2 for r in resid) / len(resid)

        print(f"\n  {name}: R² = {result['r_squared']:.4f}")
        print(f"  Residual mean = {mean_r:.4f} (should ≈ 0)")

        # Check for pattern (residuals vs fitted)
        n_pos_run = sum(1 for i in range(1, len(resid))
                        if resid[i] * resid[i-1] > 0)
        print(f"  Same-sign runs: {n_pos_run}/{len(resid)-1} "
              f"({'pattern' if n_pos_run > 0.7 * (len(resid)-1) else 'random'})")


# ─────────────────────────────────────────────────
# 4. ONE-WAY ANOVA
# ─────────────────────────────────────────────────

def one_way_anova(*groups: list[float]) -> dict:
    """One-way ANOVA: test if group means differ."""
    k = len(groups)
    N = sum(len(g) for g in groups)
    grand_mean = sum(sum(g) for g in groups) / N

    # Between-group: SSB = Σ nj(x̄j - x̄)²
    ssb = sum(len(g) * (sum(g)/len(g) - grand_mean)**2 for g in groups)

    # Within-group: SSW = ΣΣ (xij - x̄j)²
    ssw = sum(sum((x - sum(g)/len(g))**2 for x in g) for g in groups)

    sst = ssb + ssw
    df_b = k - 1
    df_w = N - k
    msb = ssb / df_b
    msw = ssw / df_w
    f_stat = msb / msw if msw > 0 else float('inf')

    # Approximate p-value using Wilson-Hilferty on F → χ²
    # F(d1,d2) approximation
    x = f_stat
    d1, d2 = df_b, df_w
    z = ((x * d1 / d2)**(1/3) - (1 - 2/(9*d2))) / math.sqrt(2/(9*d2))
    p_value = 1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))

    return {
        "SSB": ssb, "SSW": ssw, "SST": sst,
        "df_between": df_b, "df_within": df_w,
        "MSB": msb, "MSW": msw,
        "F": f_stat, "p_value": p_value,
    }


def demo_anova():
    print("\n" + "=" * 60)
    print("  One-Way ANOVA")
    print("=" * 60)

    random.seed(42)

    # Three teaching methods
    method_a = [random.gauss(75, 8) for _ in range(15)]
    method_b = [random.gauss(82, 8) for _ in range(15)]
    method_c = [random.gauss(78, 8) for _ in range(15)]

    result = one_way_anova(method_a, method_b, method_c)

    print(f"\n  Three teaching methods (n=15 each):")
    for name, data in [("A", method_a), ("B", method_b), ("C", method_c)]:
        mean = sum(data) / len(data)
        sd = math.sqrt(sum((x - mean)**2 for x in data) / (len(data) - 1))
        print(f"  Method {name}: mean={mean:.1f}, sd={sd:.1f}")

    print(f"\n  ANOVA Table:")
    print(f"  {'Source':<12} {'SS':>10} {'df':>5} {'MS':>10} {'F':>8} {'p':>8}")
    print(f"  {'─' * 55}")
    print(f"  {'Between':<12} {result['SSB']:>10.2f} {result['df_between']:>5} "
          f"{result['MSB']:>10.2f} {result['F']:>8.3f} {result['p_value']:>8.4f}")
    print(f"  {'Within':<12} {result['SSW']:>10.2f} {result['df_within']:>5} "
          f"{result['MSW']:>10.2f}")
    print(f"  {'Total':<12} {result['SST']:>10.2f} {result['df_between']+result['df_within']:>5}")
    print(f"\n  Decision: {'Reject H₀' if result['p_value'] < 0.05 else 'Fail to reject'} (α=0.05)")
    print(f"  → {'Group means differ' if result['p_value'] < 0.05 else 'No significant difference'}")


if __name__ == "__main__":
    demo_simple_regression()
    demo_multiple_regression()
    demo_residuals()
    demo_anova()
