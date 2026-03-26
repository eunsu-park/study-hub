"""
Probability and Statistics — Regression and ANOVA
Exercises covering OLS coefficient computation, R-squared calculation,
residual analysis, and one-way ANOVA F-test.
"""
import math
import random
from typing import List, Tuple


# === Exercise 1: OLS Coefficient Computation ===
def exercise_1() -> None:
    """Compute the OLS regression coefficients beta0 and beta1 from scratch
    for a simple linear regression y = beta0 + beta1*x."""
    print("=== Exercise 1: OLS Coefficient Computation ===")

    # Data: study hours vs exam score
    x = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]
    y = [55.0, 60.0, 62.0, 68.0, 72.0, 75.0, 78.0, 82.0, 88.0, 90.0, 93.0, 95.0]
    n = len(x)

    x_bar = sum(x) / n
    y_bar = sum(y) / n

    # beta1 = Sxy / Sxx
    Sxy = sum((xi - x_bar) * (yi - y_bar) for xi, yi in zip(x, y))
    Sxx = sum((xi - x_bar) ** 2 for xi in x)
    Syy = sum((yi - y_bar) ** 2 for yi in y)

    beta1 = Sxy / Sxx
    beta0 = y_bar - beta1 * x_bar

    # Fitted values and residuals
    y_hat = [beta0 + beta1 * xi for xi in x]
    residuals = [yi - yhi for yi, yhi in zip(y, y_hat)]

    # Standard error of beta1
    ss_res = sum(r ** 2 for r in residuals)
    s2 = ss_res / (n - 2) if n > 2 else 0
    se_beta1 = math.sqrt(s2 / Sxx) if Sxx > 0 else 0

    # t-statistic for beta1
    t_stat = beta1 / se_beta1 if se_beta1 > 0 else 0

    print(f"\n  Data: n={n} observations (study hours vs exam score)")
    print(f"  x_bar = {x_bar:.4f}, y_bar = {y_bar:.4f}")
    print(f"\n  Sxy = {Sxy:.4f}")
    print(f"  Sxx = {Sxx:.4f}")

    print(f"\n  OLS coefficients:")
    print(f"    beta1 = Sxy / Sxx = {Sxy:.4f} / {Sxx:.4f} = {beta1:.4f}")
    print(f"    beta0 = y_bar - beta1 * x_bar = {y_bar:.4f} - "
          f"{beta1:.4f} * {x_bar:.4f} = {beta0:.4f}")
    print(f"\n  Regression equation: y_hat = {beta0:.4f} + {beta1:.4f} * x")

    print(f"\n  SE(beta1) = {se_beta1:.4f}")
    print(f"  t-stat (H0: beta1=0) = {t_stat:.4f}")

    # Prediction
    x_new = 15.0
    y_new = beta0 + beta1 * x_new
    print(f"\n  Prediction for x={x_new}: y_hat = {y_new:.2f}")

    print(f"\n  {'x':>4} {'y':>6} {'y_hat':>8} {'Residual':>10}")
    print(f"  {'--':>4} {'--':>6} {'-----':>8} {'--------':>10}")
    for xi, yi, yhi, ri in zip(x, y, y_hat, residuals):
        print(f"  {xi:>4.0f} {yi:>6.1f} {yhi:>8.2f} {ri:>+10.2f}")
    print()


# === Exercise 2: R-Squared Calculation ===
def exercise_2() -> None:
    """Compute R-squared and adjusted R-squared, and interpret the results
    for both a linear and quadratic fit."""
    print("=== Exercise 2: R-Squared Calculation ===")

    random.seed(42)
    n = 30
    x = list(range(1, n + 1))

    # Case 1: Linear data (good fit)
    y_linear = [2.0 + 0.5 * xi + random.gauss(0, 2) for xi in x]

    # Case 2: Quadratic data with linear fit (poor fit)
    y_quad = [0.01 * xi ** 2 + random.gauss(0, 1) for xi in x]

    def compute_r_squared(x_data: List[float], y_data: List[float]) -> dict:
        m = len(x_data)
        xm = sum(x_data) / m
        ym = sum(y_data) / m

        sxy = sum((xi - xm) * (yi - ym) for xi, yi in zip(x_data, y_data))
        sxx = sum((xi - xm) ** 2 for xi in x_data)

        b1 = sxy / sxx
        b0 = ym - b1 * xm

        yh = [b0 + b1 * xi for xi in x_data]
        ss_res = sum((yi - yhi) ** 2 for yi, yhi in zip(y_data, yh))
        ss_tot = sum((yi - ym) ** 2 for yi in y_data)
        ss_reg = ss_tot - ss_res

        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        # Adjusted R^2 = 1 - (1-R^2)(n-1)/(n-p-1), p = number of predictors
        r2_adj = 1 - (1 - r2) * (m - 1) / (m - 2) if m > 2 else r2

        return {"beta0": b0, "beta1": b1, "R2": r2, "R2_adj": r2_adj,
                "SS_tot": ss_tot, "SS_reg": ss_reg, "SS_res": ss_res}

    for name, y_data in [("Linear data (y ~ x + noise)", y_linear),
                         ("Quadratic data (y ~ x^2 + noise)", y_quad)]:
        result = compute_r_squared([float(xi) for xi in x], y_data)
        print(f"\n  {name}:")
        print(f"    y_hat = {result['beta0']:.4f} + {result['beta1']:.4f} * x")
        print(f"    SS_tot = {result['SS_tot']:.4f}")
        print(f"    SS_reg = {result['SS_reg']:.4f}")
        print(f"    SS_res = {result['SS_res']:.4f}")
        print(f"    R^2 = 1 - SS_res/SS_tot = {result['R2']:.4f}")
        print(f"    Adjusted R^2 = {result['R2_adj']:.4f}")
        print(f"    Interpretation: {result['R2']*100:.1f}% of variance explained")

    print(f"\n  Key insight:")
    print(f"  R^2 is high for linear data but may be lower for quadratic data")
    print(f"  when fitted with a linear model (model misspecification).\n")


# === Exercise 3: Residual Analysis ===
def exercise_3() -> None:
    """Perform a comprehensive residual analysis to check regression
    assumptions: zero mean, constant variance, and normality."""
    print("=== Exercise 3: Residual Analysis ===")

    random.seed(42)
    n = 20
    x = [float(i) for i in range(1, n + 1)]

    # Generate data with mild heteroscedasticity
    y = [3.0 + 2.5 * xi + random.gauss(0, 1 + 0.1 * xi) for xi in x]

    # OLS fit
    xm = sum(x) / n
    ym = sum(y) / n
    sxy = sum((xi - xm) * (yi - ym) for xi, yi in zip(x, y))
    sxx = sum((xi - xm) ** 2 for xi in x)
    beta1 = sxy / sxx
    beta0 = ym - beta1 * xm

    y_hat = [beta0 + beta1 * xi for xi in x]
    residuals = [yi - yhi for yi, yhi in zip(y, y_hat)]

    print(f"\n  Regression: y_hat = {beta0:.4f} + {beta1:.4f} * x")

    # 1. Mean of residuals
    res_mean = sum(residuals) / n
    print(f"\n  1. Mean of residuals = {res_mean:.8f} (should be ~0)")

    # 2. Residual statistics
    res_var = sum((r - res_mean) ** 2 for r in residuals) / n
    res_sd = math.sqrt(res_var) if res_var > 0 else 1

    # Standardized residuals
    ss_res = sum(r ** 2 for r in residuals)
    se_reg = math.sqrt(ss_res / (n - 2))
    leverages = [1 / n + (xi - xm) ** 2 / sxx for xi in x]
    std_residuals = [r / (se_reg * math.sqrt(max(1 - h, 0.001)))
                     for r, h in zip(residuals, leverages)]

    print(f"\n  2. Residual table:")
    print(f"  {'x':>4} {'y':>8} {'y_hat':>8} {'e_i':>8} {'e_i*':>8}")
    print(f"  {'--':>4} {'--':>8} {'-----':>8} {'---':>8} {'----':>8}")
    outliers = 0
    for xi, yi, yhi, ri, sri in zip(x, y, y_hat, residuals, std_residuals):
        flag = " *" if abs(sri) > 2 else ""
        if abs(sri) > 2:
            outliers += 1
        print(f"  {xi:>4.0f} {yi:>8.2f} {yhi:>8.2f} {ri:>+8.2f} {sri:>+8.3f}{flag}")
    print(f"  (* = potential outlier, |e*| > 2)")
    print(f"  Outliers found: {outliers}")

    # 3. Normality check: skewness and kurtosis
    skew = sum(((r - res_mean) / res_sd) ** 3 for r in residuals) / n
    kurt = sum(((r - res_mean) / res_sd) ** 4 for r in residuals) / n

    print(f"\n  3. Normality diagnostics:")
    print(f"     Skewness = {skew:.4f} (normal: 0)")
    print(f"     Kurtosis = {kurt:.4f} (normal: 3)")
    print(f"     Excess kurtosis = {kurt - 3:.4f}")

    # 4. Runs test for independence
    signs = [1 if r >= 0 else -1 for r in residuals]
    runs = 1
    for i in range(1, n):
        if signs[i] != signs[i - 1]:
            runs += 1
    n_pos = sum(1 for s in signs if s == 1)
    n_neg = n - n_pos
    expected_runs = 1 + 2 * n_pos * n_neg / n

    print(f"\n  4. Runs test for independence:")
    print(f"     Number of runs = {runs}")
    print(f"     Expected runs = {expected_runs:.2f}")
    verdict = "random" if abs(runs - expected_runs) <= 3 else "pattern detected"
    print(f"     Assessment: {verdict}\n")


# === Exercise 4: One-Way ANOVA F-Test ===
def exercise_4() -> None:
    """Perform a one-way ANOVA F-test to determine whether three group
    means differ significantly."""
    print("=== Exercise 4: One-Way ANOVA F-Test ===")

    random.seed(42)
    # Three teaching methods
    method_a = [random.gauss(75, 8) for _ in range(15)]
    method_b = [random.gauss(82, 8) for _ in range(15)]
    method_c = [random.gauss(78, 8) for _ in range(15)]

    groups: List[Tuple[str, List[float]]] = [
        ("Method A", method_a),
        ("Method B", method_b),
        ("Method C", method_c),
    ]

    k = len(groups)
    N = sum(len(g) for _, g in groups)
    all_data = [x for _, g in groups for x in g]
    grand_mean = sum(all_data) / N

    # Between-group sum of squares
    SSB = sum(len(g) * (sum(g) / len(g) - grand_mean) ** 2 for _, g in groups)

    # Within-group sum of squares
    SSW = sum(sum((x - sum(g) / len(g)) ** 2 for x in g) for _, g in groups)

    SST = SSB + SSW
    df_b = k - 1
    df_w = N - k
    MSB = SSB / df_b
    MSW = SSW / df_w
    F_stat = MSB / MSW if MSW > 0 else float('inf')

    # Approximate p-value using Wilson-Hilferty transformation
    x_val = F_stat
    d1, d2 = df_b, df_w
    z_wh = ((x_val * d1 / d2) ** (1 / 3) - (1 - 2 / (9 * d2))) / \
            math.sqrt(2 / (9 * d2))
    p_value = 1 - 0.5 * (1 + math.erf(z_wh / math.sqrt(2)))

    print(f"\n  Three teaching methods (n=15 each):")
    print(f"  {'Group':>10} {'Mean':>8} {'SD':>8} {'n':>4}")
    print(f"  {'-' * 32}")
    for name, g in groups:
        m = sum(g) / len(g)
        sd = math.sqrt(sum((x - m) ** 2 for x in g) / (len(g) - 1))
        print(f"  {name:>10} {m:>8.2f} {sd:>8.2f} {len(g):>4}")

    print(f"\n  Grand mean = {grand_mean:.4f}")

    # ANOVA table
    print(f"\n  ANOVA Table:")
    print(f"  {'Source':<10} {'SS':>10} {'df':>5} {'MS':>10} {'F':>8} {'p':>8}")
    print(f"  {'-' * 53}")
    print(f"  {'Between':<10} {SSB:>10.2f} {df_b:>5} {MSB:>10.2f} "
          f"{F_stat:>8.3f} {p_value:>8.4f}")
    print(f"  {'Within':<10} {SSW:>10.2f} {df_w:>5} {MSW:>10.2f}")
    print(f"  {'Total':<10} {SST:>10.2f} {df_b + df_w:>5}")

    # Verify SST = SSB + SSW
    assert abs(SST - (SSB + SSW)) < 1e-6, "SST != SSB + SSW"
    print(f"\n  Check: SSB + SSW = {SSB:.2f} + {SSW:.2f} = {SSB + SSW:.2f} = SST")

    # Effect size: eta-squared
    eta_sq = SSB / SST
    print(f"\n  Effect size: eta^2 = SSB/SST = {eta_sq:.4f}")
    print(f"  ({eta_sq * 100:.1f}% of variance explained by group)")

    alpha = 0.05
    print(f"\n  Decision (alpha={alpha}): "
          f"{'Reject H0' if p_value < alpha else 'Fail to reject H0'}")
    if p_value < alpha:
        print(f"  At least one group mean differs significantly.\n")
    else:
        print(f"  No significant difference among group means.\n")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
