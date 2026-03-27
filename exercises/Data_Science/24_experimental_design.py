"""
Exercises for Lesson 24: Experimental Design
Topic: Data_Science

Solutions to practice problems from the lesson.
"""
import numpy as np


# === Exercise 1: Sample Size Calculation ===
# Problem: Calculate required sample size for an A/B test comparing
#          two proportions with various power and MDE settings.
def exercise_1():
    """Solution for sample size calculation using power analysis.

    For comparing two proportions:
    n = (z_{alpha/2} * sqrt(2*p_bar*(1-p_bar)) + z_beta * sqrt(p1*(1-p1) + p2*(1-p2)))^2
        / (p1 - p2)^2

    where p_bar = (p1 + p2) / 2, z_alpha/2 and z_beta are critical values.
    """
    from scipy.stats import norm

    def sample_size_two_proportions(p1, p2, alpha=0.05, power=0.80):
        """Calculate sample size per group for two-proportion comparison.

        Parameters
        ----------
        p1 : float
            Baseline (control) proportion
        p2 : float
            Expected (treatment) proportion
        alpha : float
            Significance level (two-sided)
        power : float
            Statistical power (1 - beta)

        Returns
        -------
        int
            Required sample size per group
        """
        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power)
        p_bar = (p1 + p2) / 2

        numerator = (
            z_alpha * np.sqrt(2 * p_bar * (1 - p_bar))
            + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
        ) ** 2
        denominator = (p1 - p2) ** 2

        return int(np.ceil(numerator / denominator))

    # Problem 1: Baseline 3%, detect 20% relative lift (to 3.6%)
    p_control = 0.03
    p_treat_20 = 0.036  # 20% relative lift
    p_treat_10 = 0.033  # 10% relative lift

    print("Sample Size Calculations:")
    print(f"  Baseline conversion rate: {p_control:.1%}")

    # Part 1: alpha=0.05, power=0.80, 20% lift
    n1 = sample_size_two_proportions(p_control, p_treat_20, alpha=0.05, power=0.80)
    print(f"\n  1a. 20% lift ({p_control:.1%} -> {p_treat_20:.1%}), "
          f"alpha=0.05, power=0.80")
    print(f"      Sample size per group: {n1:,}")
    print(f"      Total sample size: {2 * n1:,}")

    # Part 2: Increase power to 0.90
    n2 = sample_size_two_proportions(p_control, p_treat_20, alpha=0.05, power=0.90)
    print(f"\n  1b. Same but power=0.90")
    print(f"      Sample size per group: {n2:,}")
    print(f"      Increase vs 0.80 power: {(n2 - n1) / n1 * 100:.1f}%")

    # Part 3: Reduce MDE to 10% lift
    n3 = sample_size_two_proportions(p_control, p_treat_10, alpha=0.05, power=0.80)
    print(f"\n  1c. 10% lift ({p_control:.1%} -> {p_treat_10:.1%}), "
          f"alpha=0.05, power=0.80")
    print(f"      Sample size per group: {n3:,}")
    print(f"      Increase vs 20% lift: {(n3 - n1) / n1 * 100:.1f}%")

    # Summary table
    print("\n  Summary of relationships:")
    print(f"    {'Scenario':<30} {'n/group':>10} {'Total':>10}")
    print(f"    {'-' * 52}")
    print(f"    {'20% lift, power=0.80':<30} {n1:>10,} {2*n1:>10,}")
    print(f"    {'20% lift, power=0.90':<30} {n2:>10,} {2*n2:>10,}")
    print(f"    {'10% lift, power=0.80':<30} {n3:>10,} {2*n3:>10,}")
    print(f"\n  Key insight: Halving the effect size roughly quadruples "
          f"the required n.")


# === Exercise 2: Randomization and Balance Check ===
# Problem: Implement stratified randomization and verify
#          covariate balance between treatment and control groups.
def exercise_2():
    """Solution for stratified randomization with balance checking.

    Simple randomization can produce imbalanced groups by chance,
    especially with small samples. Stratified randomization ensures
    balance on known prognostic factors by randomizing within strata.

    Balance is checked using standardized mean differences (SMD):
    SMD = (mean_treat - mean_control) / pooled_SD
    |SMD| < 0.1 is considered well-balanced.
    """
    np.random.seed(42)

    n_participants = 100

    # Generate participant characteristics
    ages = np.random.normal(45, 12, n_participants)
    genders = np.random.choice(['M', 'F'], n_participants, p=[0.55, 0.45])
    risk_scores = np.random.uniform(0, 10, n_participants)

    # Simple randomization
    simple_assignment = np.random.choice([0, 1], n_participants)

    print("=== Simple Randomization ===")
    treat_mask = simple_assignment == 1
    ctrl_mask = simple_assignment == 0
    print(f"  Treatment: n={np.sum(treat_mask)}, Control: n={np.sum(ctrl_mask)}")

    def compute_smd(values, assignment):
        """Compute standardized mean difference."""
        treat_vals = values[assignment == 1]
        ctrl_vals = values[assignment == 0]
        pooled_sd = np.sqrt(
            (np.var(treat_vals, ddof=1) + np.var(ctrl_vals, ddof=1)) / 2
        )
        if pooled_sd == 0:
            return 0.0
        return (np.mean(treat_vals) - np.mean(ctrl_vals)) / pooled_sd

    smd_age_simple = compute_smd(ages, simple_assignment)
    smd_risk_simple = compute_smd(risk_scores, simple_assignment)
    gender_pct_treat = np.mean(genders[treat_mask] == 'M')
    gender_pct_ctrl = np.mean(genders[ctrl_mask] == 'M')

    print(f"  Age SMD: {smd_age_simple:.4f} "
          f"({'balanced' if abs(smd_age_simple) < 0.1 else 'IMBALANCED'})")
    print(f"  Risk SMD: {smd_risk_simple:.4f} "
          f"({'balanced' if abs(smd_risk_simple) < 0.1 else 'IMBALANCED'})")
    print(f"  Male %: treat={gender_pct_treat:.1%}, ctrl={gender_pct_ctrl:.1%}")

    # Stratified randomization by age group and gender
    print("\n=== Stratified Randomization ===")
    age_strata = np.where(ages < 35, 'young',
                          np.where(ages < 55, 'middle', 'senior'))

    stratified_assignment = np.zeros(n_participants, dtype=int)

    strata_keys = set()
    for i in range(n_participants):
        strata_keys.add((age_strata[i], genders[i]))

    for stratum_age, stratum_gender in strata_keys:
        mask = (age_strata == stratum_age) & (genders == stratum_gender)
        indices = np.where(mask)[0]
        n_stratum = len(indices)
        # Block randomization within stratum
        half = n_stratum // 2
        perm = np.random.permutation(n_stratum)
        stratified_assignment[indices[perm[:half]]] = 1
        stratified_assignment[indices[perm[half:]]] = 0

    treat_mask_s = stratified_assignment == 1
    ctrl_mask_s = stratified_assignment == 0
    print(f"  Treatment: n={np.sum(treat_mask_s)}, "
          f"Control: n={np.sum(ctrl_mask_s)}")

    smd_age_strat = compute_smd(ages, stratified_assignment)
    smd_risk_strat = compute_smd(risk_scores, stratified_assignment)
    gender_pct_treat_s = np.mean(genders[treat_mask_s] == 'M')
    gender_pct_ctrl_s = np.mean(genders[ctrl_mask_s] == 'M')

    print(f"  Age SMD: {smd_age_strat:.4f} "
          f"({'balanced' if abs(smd_age_strat) < 0.1 else 'IMBALANCED'})")
    print(f"  Risk SMD: {smd_risk_strat:.4f} "
          f"({'balanced' if abs(smd_risk_strat) < 0.1 else 'IMBALANCED'})")
    print(f"  Male %: treat={gender_pct_treat_s:.1%}, ctrl={gender_pct_ctrl_s:.1%}")

    print(f"\n  Comparison of balance (|SMD|):")
    print(f"    {'Covariate':<15} {'Simple':>10} {'Stratified':>12}")
    print(f"    {'-' * 40}")
    print(f"    {'Age':<15} {abs(smd_age_simple):>10.4f} {abs(smd_age_strat):>12.4f}")
    print(f"    {'Risk score':<15} {abs(smd_risk_simple):>10.4f} "
          f"{abs(smd_risk_strat):>12.4f}")


# === Exercise 3: Sequential Testing (Alpha Spending) ===
# Problem: Implement O'Brien-Fleming and Pocock alpha spending
#          functions for interim analyses.
def exercise_3():
    """Solution for sequential testing with alpha spending functions.

    Sequential testing allows interim analyses without inflating
    the overall Type I error rate. Alpha spending functions allocate
    the total alpha budget across planned analysis points.

    O'Brien-Fleming: Very conservative early, relaxed late.
    Boundary ~ z_{alpha/(2*t)} where t is information fraction.

    Pocock: Constant boundary at each interim look.
    Uses the same critical value at each analysis.
    """
    from scipy.stats import norm

    def obrien_fleming_boundary(alpha, n_looks, look_index):
        """Calculate O'Brien-Fleming boundary at a given look.

        The O'Brien-Fleming spending function:
        alpha_spent(t) = 2 * (1 - Phi(z_{alpha/2} / sqrt(t)))
        where t = information fraction.
        """
        t = (look_index + 1) / n_looks  # information fraction
        z_nominal = norm.ppf(1 - alpha / 2)
        z_boundary = z_nominal / np.sqrt(t)
        alpha_at_look = 2 * (1 - norm.cdf(z_boundary))
        return z_boundary, alpha_at_look

    def pocock_boundary(alpha, n_looks):
        """Approximate Pocock boundary (constant z at each look).

        Pocock uses a constant boundary z_p at each look such
        that the overall alpha is maintained. This requires
        numerical integration; we approximate here.
        """
        # Pocock boundaries for common n_looks (pre-tabulated)
        pocock_z = {
            2: 2.178, 3: 2.289, 4: 2.361, 5: 2.413,
        }
        z_val = pocock_z.get(n_looks, norm.ppf(1 - alpha / (2 * n_looks)))
        alpha_at_look = 2 * (1 - norm.cdf(z_val))
        return z_val, alpha_at_look

    alpha = 0.05
    n_looks = 5

    print("Sequential Testing: Alpha Spending Functions")
    print(f"  Overall alpha: {alpha}")
    print(f"  Number of interim analyses: {n_looks}")

    print(f"\n  {'Look':>6} {'Frac':>6} {'OBF z':>8} {'OBF alpha':>10} "
          f"{'Pocock z':>10} {'Pocock alpha':>13}")
    print(f"  {'-' * 58}")

    cumulative_obf = 0.0
    z_pocock, a_pocock = pocock_boundary(alpha, n_looks)

    for i in range(n_looks):
        frac = (i + 1) / n_looks
        z_obf, a_obf = obrien_fleming_boundary(alpha, n_looks, i)
        cumulative_obf += a_obf

        print(f"  {i+1:>6} {frac:>6.2f} {z_obf:>8.3f} {a_obf:>10.6f} "
              f"{z_pocock:>10.3f} {a_pocock:>13.6f}")

    print(f"\n  Key observations:")
    print(f"    - OBF: Very strict early (z ~ 4-5), relaxes to z ~ 2 at final look")
    print(f"    - Pocock: Constant z = {z_pocock:.3f} at every look")
    print(f"    - OBF is preferred when early stopping is undesirable")
    print(f"    - Pocock is preferred when equal chances of stopping are wanted")

    # Simulate an A/B test with interim looks
    print("\n  === Simulated A/B Test with 5 Interim Looks ===")
    np.random.seed(42)
    n_per_look = 500
    p_control = 0.05
    p_treatment = 0.065  # 30% relative lift

    for i in range(n_looks):
        n_accumulated = n_per_look * (i + 1)
        control_successes = np.random.binomial(n_accumulated, p_control)
        treat_successes = np.random.binomial(n_accumulated, p_treatment)

        p_hat_c = control_successes / n_accumulated
        p_hat_t = treat_successes / n_accumulated
        p_pooled = (control_successes + treat_successes) / (2 * n_accumulated)
        se = np.sqrt(2 * p_pooled * (1 - p_pooled) / n_accumulated)
        z_test = (p_hat_t - p_hat_c) / se if se > 0 else 0.0

        z_obf, _ = obrien_fleming_boundary(alpha, n_looks, i)
        stop_obf = abs(z_test) > z_obf
        stop_pocock = abs(z_test) > z_pocock

        print(f"    Look {i+1} (n={n_accumulated:,}): z={z_test:.3f}, "
              f"OBF boundary={z_obf:.3f} ({'STOP' if stop_obf else 'continue'}), "
              f"Pocock={'STOP' if stop_pocock else 'continue'}")


# === Exercise 4: Multiple Comparison Correction ===
# Problem: Apply Bonferroni and Holm-Bonferroni corrections
#          to a set of p-values from segment-level A/B test results.
def exercise_4():
    """Solution for multiple comparison corrections.

    When testing multiple hypotheses simultaneously, the probability
    of at least one false positive (familywise error rate, FWER)
    increases dramatically: FWER = 1 - (1-alpha)^m.

    Bonferroni: Reject if p_i < alpha/m (most conservative).
    Holm-Bonferroni: Step-down procedure (more powerful).
      Sort p-values: p_(1) <= p_(2) <= ... <= p_(m)
      Reject p_(i) if p_(i) < alpha / (m - i + 1)
      Stop at first non-rejection.
    """
    # Simulated p-values from testing 5 age segments
    segments = ['18-25', '26-35', '36-45', '46-55', '56+']
    p_values = np.array([0.003, 0.020, 0.048, 0.150, 0.510])
    alpha = 0.05
    m = len(p_values)

    print("Multiple Comparison Corrections:")
    print(f"  Number of tests (m): {m}")
    print(f"  Nominal alpha: {alpha}")
    print(f"  FWER without correction: {1 - (1 - alpha)**m:.4f}")

    # Uncorrected results
    print(f"\n  === Uncorrected (Naive) ===")
    for seg, p in zip(segments, p_values):
        sig = "*" if p < alpha else "ns"
        print(f"    {seg}: p={p:.4f} {sig}")
    n_sig_naive = np.sum(p_values < alpha)
    print(f"    Significant: {n_sig_naive}/{m}")

    # Bonferroni correction
    print(f"\n  === Bonferroni Correction ===")
    alpha_bonf = alpha / m
    print(f"    Corrected alpha: {alpha_bonf:.4f}")
    bonf_adjusted = np.minimum(p_values * m, 1.0)
    for seg, p, p_adj in zip(segments, p_values, bonf_adjusted):
        sig = "*" if p_adj < alpha else "ns"
        print(f"    {seg}: p={p:.4f}, p_adj={p_adj:.4f} {sig}")
    n_sig_bonf = np.sum(bonf_adjusted < alpha)
    print(f"    Significant: {n_sig_bonf}/{m}")

    # Holm-Bonferroni (step-down)
    print(f"\n  === Holm-Bonferroni Correction (Step-Down) ===")
    sort_idx = np.argsort(p_values)
    sorted_p = p_values[sort_idx]
    sorted_segments = [segments[i] for i in sort_idx]
    holm_adjusted = np.zeros(m)
    holm_sig = np.zeros(m, dtype=bool)

    cummax = 0.0
    for i in range(m):
        adjusted = sorted_p[i] * (m - i)
        cummax = max(cummax, adjusted)
        holm_adjusted[i] = min(cummax, 1.0)
        holm_sig[i] = holm_adjusted[i] < alpha

    print(f"    {'Step':<6} {'Segment':<10} {'p':>8} {'alpha_i':>10} "
          f"{'p_adj':>8} {'Sig?':>6}")
    print(f"    {'-' * 52}")
    for i in range(m):
        alpha_i = alpha / (m - i)
        sig = "*" if holm_sig[i] else "ns"
        print(f"    {i+1:<6} {sorted_segments[i]:<10} {sorted_p[i]:>8.4f} "
              f"{alpha_i:>10.4f} {holm_adjusted[i]:>8.4f} {sig:>6}")
    n_sig_holm = np.sum(holm_sig)
    print(f"    Significant: {n_sig_holm}/{m}")

    # Summary
    print(f"\n  Summary:")
    print(f"    {'Method':<25} {'Significant':>12}")
    print(f"    {'-' * 40}")
    print(f"    {'Uncorrected':<25} {n_sig_naive:>12}/{m}")
    print(f"    {'Bonferroni':<25} {n_sig_bonf:>12}/{m}")
    print(f"    {'Holm-Bonferroni':<25} {n_sig_holm:>12}/{m}")
    print(f"\n    Holm-Bonferroni is uniformly more powerful than Bonferroni")
    print(f"    while still controlling FWER at alpha={alpha}.")


if __name__ == "__main__":
    print("=== Exercise 1: Sample Size Calculation ===")
    exercise_1()
    print("\n=== Exercise 2: Randomization and Balance Check ===")
    exercise_2()
    print("\n=== Exercise 3: Sequential Testing (Alpha Spending) ===")
    exercise_3()
    print("\n=== Exercise 4: Multiple Comparison Correction ===")
    exercise_4()
    print("\nAll exercises completed!")
