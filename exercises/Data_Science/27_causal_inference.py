"""
Exercises for Lesson 27: Causal Inference
Topic: Data_Science

Solutions to practice problems from the lesson.
"""
import numpy as np


# === Exercise 1: Propensity Score Matching ===
# Problem: Estimate the causal effect of a job training program using
#          propensity score matching and IPW on observational data.
def exercise_1():
    """Solution for propensity score analysis with matching and IPW.

    In observational studies, treatment is not randomly assigned.
    Propensity score methods attempt to mimic randomization by
    balancing covariates between treated and control groups.

    Propensity score: e(X) = P(T=1 | X)
    ATE via matching: pair each treated unit with nearest control by e(X)
    ATE via IPW: weight by 1/e(X) for treated, 1/(1-e(X)) for control
    """
    np.random.seed(42)

    # Simulate observational data
    n = 1000
    age = np.random.normal(40, 10, n)
    education = np.random.normal(12, 3, n)
    experience = np.random.normal(10, 5, n).clip(0, None)

    # Treatment assignment depends on covariates (confounding)
    logit = -2 + 0.03 * age + 0.15 * education + 0.05 * experience
    p_treat = 1 / (1 + np.exp(-logit))
    treatment = np.random.binomial(1, p_treat)

    # Outcome: true treatment effect = 3000
    true_ate = 3000
    income = (20000 + 500 * education + 300 * experience + 100 * age
              + true_ate * treatment + np.random.normal(0, 5000, n))

    print("Propensity Score Analysis:")
    print(f"  True ATE: {true_ate}")
    print(f"  N total: {n}, N treated: {np.sum(treatment)}, "
          f"N control: {np.sum(1 - treatment)}")

    # Naive estimate (biased)
    naive_ate = np.mean(income[treatment == 1]) - np.mean(income[treatment == 0])
    print(f"  Naive ATE: {naive_ate:.0f} (biased due to confounding)")

    # Step 1: Estimate propensity score via logistic regression
    # Manual logistic regression using gradient descent
    X = np.column_stack([np.ones(n), age, education, experience])
    y = treatment.astype(float)

    # Gradient descent for logistic regression
    beta = np.zeros(X.shape[1])
    lr = 0.0001
    for _ in range(5000):
        z = X @ beta
        p = 1 / (1 + np.exp(-z.clip(-500, 500)))
        grad = X.T @ (y - p) / n
        beta += lr * grad

    propensity = 1 / (1 + np.exp(-(X @ beta).clip(-500, 500)))
    propensity = propensity.clip(0.01, 0.99)  # trim extreme values

    print(f"\n  Propensity score range: [{propensity.min():.3f}, "
          f"{propensity.max():.3f}]")
    print(f"  Mean propensity (treated): {propensity[treatment == 1].mean():.3f}")
    print(f"  Mean propensity (control): {propensity[treatment == 0].mean():.3f}")

    # Step 2: Matching (1-nearest-neighbor on propensity score)
    treated_idx = np.where(treatment == 1)[0]
    control_idx = np.where(treatment == 0)[0]
    control_ps = propensity[control_idx]

    matched_outcomes = np.zeros(len(treated_idx))
    for i, t_idx in enumerate(treated_idx):
        # Find nearest control by propensity score
        distances = np.abs(control_ps - propensity[t_idx])
        nearest = np.argmin(distances)
        matched_outcomes[i] = income[control_idx[nearest]]

    ate_matched = np.mean(income[treated_idx]) - np.mean(matched_outcomes)
    print(f"\n  Matched ATE: {ate_matched:.0f}")

    # Step 3: IPW estimate
    weights = np.where(treatment == 1,
                       1 / propensity,
                       1 / (1 - propensity))
    # Trim extreme weights
    weight_cap = np.percentile(weights, 99)
    weights = weights.clip(max=weight_cap)

    treated_weighted = np.sum(treatment * income * weights) / \
                       np.sum(treatment * weights)
    control_weighted = np.sum((1 - treatment) * income * weights) / \
                       np.sum((1 - treatment) * weights)
    ate_ipw = treated_weighted - control_weighted
    print(f"  IPW ATE: {ate_ipw:.0f}")

    print(f"\n  Summary: True={true_ate}, Naive={naive_ate:.0f}, "
          f"Matched={ate_matched:.0f}, IPW={ate_ipw:.0f}")


# === Exercise 2: Difference-in-Differences ===
# Problem: Estimate the effect of a policy change using DID design
#          and test the parallel trends assumption.
def exercise_2():
    """Solution for Difference-in-Differences estimation.

    DID = (Y_treat_post - Y_treat_pre) - (Y_ctrl_post - Y_ctrl_pre)

    This is equivalent to the coefficient on treated*post in:
    Y = b0 + b1*treated + b2*post + b3*treated*post + e

    Key assumption: parallel trends (without treatment, both groups
    would have followed the same trajectory).
    """
    np.random.seed(42)

    # Simulate panel data: effect of minimum wage increase
    n_per_group = 300
    true_effect = 2500

    # Pre-treatment period (4 time points for trend checking)
    # Both groups share a common trend
    common_trend = np.array([0, 500, 1000, 1500])
    treat_base = 35000
    ctrl_base = 32000

    # Pre-treatment data (check parallel trends)
    print("Difference-in-Differences Analysis:")
    print(f"  True treatment effect: {true_effect}")
    print(f"\n  Pre-Treatment Trend Check:")
    print(f"    {'Period':>8} {'Treatment':>12} {'Control':>12} {'Diff':>10}")
    print(f"    {'-' * 45}")

    for t, trend in enumerate(common_trend):
        t_mean = treat_base + trend + np.random.normal(0, 200)
        c_mean = ctrl_base + trend + np.random.normal(0, 200)
        print(f"    {t:>8} {t_mean:>12,.0f} {c_mean:>12,.0f} "
              f"{t_mean - c_mean:>10,.0f}")

    # Generate pre/post data
    wage_treat_pre = np.random.normal(treat_base + 1500, 5000, n_per_group)
    wage_ctrl_pre = np.random.normal(ctrl_base + 1500, 5000, n_per_group)

    # Post-treatment: both groups get common time trend + treatment effect
    time_trend = 2000
    wage_treat_post = np.random.normal(
        treat_base + 1500 + time_trend + true_effect, 5000, n_per_group
    )
    wage_ctrl_post = np.random.normal(
        ctrl_base + 1500 + time_trend, 5000, n_per_group
    )

    # DID calculation (manual)
    mean_tp = np.mean(wage_treat_post)
    mean_tc = np.mean(wage_treat_pre)
    mean_cp = np.mean(wage_ctrl_post)
    mean_cc = np.mean(wage_ctrl_pre)

    did_manual = (mean_tp - mean_tc) - (mean_cp - mean_cc)

    print(f"\n  DID Manual Calculation:")
    print(f"    Treat post mean: {mean_tp:,.0f}")
    print(f"    Treat pre mean:  {mean_tc:,.0f}")
    print(f"    Treat change:    {mean_tp - mean_tc:,.0f}")
    print(f"    Ctrl post mean:  {mean_cp:,.0f}")
    print(f"    Ctrl pre mean:   {mean_cc:,.0f}")
    print(f"    Ctrl change:     {mean_cp - mean_cc:,.0f}")
    print(f"    DID estimate:    {did_manual:,.0f}")

    # DID via OLS regression
    n_total = 4 * n_per_group
    wages = np.concatenate([wage_treat_pre, wage_treat_post,
                            wage_ctrl_pre, wage_ctrl_post])
    treated = np.repeat([1, 1, 0, 0], n_per_group)
    post = np.repeat([0, 1, 0, 1], n_per_group)
    interaction = treated * post

    # OLS: Y = b0 + b1*treated + b2*post + b3*interaction
    X = np.column_stack([np.ones(n_total), treated, post, interaction])
    beta = np.linalg.lstsq(X, wages, rcond=None)[0]

    # Standard errors
    residuals = wages - X @ beta
    sigma2 = np.sum(residuals ** 2) / (n_total - 4)
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))

    print(f"\n  DID Regression:")
    labels = ['Intercept', 'Treated', 'Post', 'Treated x Post (DID)']
    for label, b, s in zip(labels, beta, se):
        t_stat = b / s
        p_val = 2 * (1 - _normal_cdf(abs(t_stat)))
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else \
              "*" if p_val < 0.05 else "ns"
        print(f"    {label:<25} {b:>10,.0f} (SE={s:>7,.0f}) {sig}")

    print(f"\n  DID estimate: {beta[3]:,.0f} (true: {true_effect:,})")

    # Placebo test: apply fake treatment one period earlier
    print(f"\n  Placebo Test (fake treatment in pre-period):")
    print(f"    If the model is valid, a placebo test should yield")
    print(f"    a DID estimate close to 0.")


# === Exercise 3: Regression Discontinuity Design ===
# Problem: Estimate a local treatment effect at a cutoff using RDD.
def exercise_3():
    """Solution for sharp regression discontinuity design.

    RDD exploits a known cutoff: treatment is assigned when a running
    variable crosses a threshold. Near the threshold, assignment is
    effectively random, enabling causal identification.

    Local linear regression within a bandwidth around the cutoff:
    Y = a + b*(X-c) + tau*D + gamma*D*(X-c) + e
    where D = 1{X >= c}, c = cutoff, tau = treatment effect.
    """
    np.random.seed(42)

    # Simulate: scholarship effect on college GPA
    n = 800
    cutoff = 75
    true_effect = 0.4

    # Running variable: test score
    test_score = np.random.uniform(50, 100, n)
    treatment = (test_score >= cutoff).astype(int)

    # Outcome: GPA (continuous relationship + treatment jump)
    gpa = (2.0 + 0.025 * test_score + true_effect * treatment
           + np.random.normal(0, 0.3, n))

    centered = test_score - cutoff

    print("Regression Discontinuity Design:")
    print(f"  Cutoff: {cutoff}")
    print(f"  True treatment effect: {true_effect}")
    print(f"  N total: {n}, N treated: {np.sum(treatment)}, "
          f"N control: {np.sum(1 - treatment)}")

    # Estimate with different bandwidths
    print(f"\n  RDD Estimates by Bandwidth:")
    print(f"    {'BW':>6} {'N_local':>8} {'Tau':>8} {'SE':>8} {'95% CI':>20}")
    print(f"    {'-' * 54}")

    for bw in [5, 10, 15, 20]:
        local_mask = np.abs(centered) <= bw
        n_local = np.sum(local_mask)

        x_local = centered[local_mask]
        d_local = treatment[local_mask]
        y_local = gpa[local_mask]

        # Local linear regression: Y = a + b*X_centered + tau*D + gamma*D*X
        X_rdd = np.column_stack([
            np.ones(n_local),
            x_local,
            d_local,
            d_local * x_local
        ])
        beta_rdd = np.linalg.lstsq(X_rdd, y_local, rcond=None)[0]

        # Standard error for tau
        resid = y_local - X_rdd @ beta_rdd
        sigma2 = np.sum(resid ** 2) / (n_local - 4)
        cov_matrix = sigma2 * np.linalg.inv(X_rdd.T @ X_rdd)
        se_tau = np.sqrt(cov_matrix[2, 2])

        ci_lower = beta_rdd[2] - 1.96 * se_tau
        ci_upper = beta_rdd[2] + 1.96 * se_tau

        print(f"    {bw:>6} {n_local:>8} {beta_rdd[2]:>8.3f} {se_tau:>8.3f} "
              f"[{ci_lower:>8.3f}, {ci_upper:>8.3f}]")

    print(f"\n  Best practice: check density continuity at cutoff (McCrary test)")
    print(f"  and covariate balance just above vs below the threshold.")


# === Exercise 4: Heterogeneous Treatment Effects (T-Learner) ===
# Problem: Estimate treatment effects that vary across subgroups
#          using the T-Learner approach.
def exercise_4():
    """Solution for heterogeneous treatment effect estimation.

    T-Learner: Train separate models for treated and control groups.
    CATE(x) = E[Y|X=x, T=1] - E[Y|X=x, T=0]
            = mu_1(x) - mu_0(x)

    This exercise uses simple polynomial regression as the base
    learner to estimate conditional means.
    """
    np.random.seed(42)

    # Simulate RCT with heterogeneous effects
    n = 1500
    age = np.random.uniform(20, 70, n)
    income = np.random.uniform(20000, 100000, n)
    treatment = np.random.binomial(1, 0.5, n)  # randomized

    # True CATE: effect depends on age (larger for younger people)
    true_cate = 8.0 - 0.15 * (age - 20)  # ranges from 8 to 0.5

    # Outcome
    outcome = (50 + 0.1 * age + 0.0002 * income
               + true_cate * treatment
               + np.random.normal(0, 3, n))

    print("Heterogeneous Treatment Effects (T-Learner):")
    print(f"  True CATE function: 8.0 - 0.15*(age - 20)")
    print(f"  N={n}, randomized treatment")

    # Overall ATE
    ate = np.mean(outcome[treatment == 1]) - np.mean(outcome[treatment == 0])
    true_ate = np.mean(true_cate)
    print(f"  True ATE: {true_ate:.3f}")
    print(f"  Observed ATE: {ate:.3f}")

    # T-Learner: fit separate polynomial regression for each group
    treated_mask = treatment == 1
    control_mask = treatment == 0

    def fit_polynomial_2d(x1, x2, y, degree=2):
        """Fit polynomial regression with two features."""
        features = [np.ones(len(x1))]
        for d in range(1, degree + 1):
            features.append(x1 ** d)
            features.append(x2 ** d)
        if degree >= 2:
            features.append(x1 * x2)
        X = np.column_stack(features)
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        return beta, X

    def predict_poly_2d(x1, x2, beta, degree=2):
        """Predict using fitted polynomial."""
        features = [np.ones(len(x1))]
        for d in range(1, degree + 1):
            features.append(x1 ** d)
            features.append(x2 ** d)
        if degree >= 2:
            features.append(x1 * x2)
        X = np.column_stack(features)
        return X @ beta

    # Normalize features for stability
    age_norm = (age - np.mean(age)) / np.std(age)
    inc_norm = (income - np.mean(income)) / np.std(income)

    beta_1, _ = fit_polynomial_2d(age_norm[treated_mask],
                                   inc_norm[treated_mask],
                                   outcome[treated_mask])
    beta_0, _ = fit_polynomial_2d(age_norm[control_mask],
                                   inc_norm[control_mask],
                                   outcome[control_mask])

    # Estimate CATE for all individuals
    mu_1 = predict_poly_2d(age_norm, inc_norm, beta_1)
    mu_0 = predict_poly_2d(age_norm, inc_norm, beta_0)
    cate_hat = mu_1 - mu_0

    print(f"\n  T-Learner CATE Estimates by Age Group:")
    print(f"    {'Age Group':<12} {'True CATE':>10} {'Est CATE':>10} "
          f"{'Abs Error':>10} {'N':>6}")
    print(f"    {'-' * 52}")

    age_bins = [(20, 30), (30, 40), (40, 50), (50, 60), (60, 70)]
    for lo, hi in age_bins:
        mask = (age >= lo) & (age < hi)
        true_mean = np.mean(true_cate[mask])
        est_mean = np.mean(cate_hat[mask])
        err = abs(true_mean - est_mean)
        print(f"    {lo}-{hi:<8} {true_mean:>10.3f} {est_mean:>10.3f} "
              f"{err:>10.3f} {np.sum(mask):>6}")

    corr = np.corrcoef(true_cate, cate_hat)[0, 1]
    rmse = np.sqrt(np.mean((true_cate - cate_hat) ** 2))
    print(f"\n  Correlation(true, est): {corr:.4f}, RMSE: {rmse:.4f}")
    print(f"  T-Learner works well in RCTs; high variance in observational data.")


def _normal_cdf(x):
    """Approximate standard normal CDF using the error function."""
    return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


if __name__ == "__main__":
    print("=== Exercise 1: Propensity Score Matching ===")
    exercise_1()
    print("\n=== Exercise 2: Difference-in-Differences ===")
    exercise_2()
    print("\n=== Exercise 3: Regression Discontinuity Design ===")
    exercise_3()
    print("\n=== Exercise 4: Heterogeneous Treatment Effects ===")
    exercise_4()
    print("\nAll exercises completed!")
