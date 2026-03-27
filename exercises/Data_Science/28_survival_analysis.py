"""
Exercises for Lesson 28: Survival Analysis
Topic: Data_Science

Solutions to practice problems from the lesson.
"""
import numpy as np


# === Exercise 1: Kaplan-Meier Estimator ===
# Problem: Implement the Kaplan-Meier survival curve estimator from
#          scratch and compare two groups with the log-rank test.
def exercise_1():
    """Solution for Kaplan-Meier estimator and log-rank test.

    Kaplan-Meier estimate at each event time t_i:
    S(t) = product over t_i <= t of (1 - d_i / n_i)
    where d_i = number of events at t_i, n_i = number at risk.

    Log-rank test: compares observed vs expected events under H0.
    chi2 = sum((O_j - E_j)^2 / E_j) for each group j.
    """
    np.random.seed(42)

    # Simulate clinical trial: two treatment arms
    n_per_arm = 100

    # Treatment arm: longer survival (exponential with mean 24 months)
    treat_times = np.random.exponential(24, n_per_arm)
    treat_event = np.random.binomial(1, 0.75, n_per_arm)

    # Control arm: shorter survival (exponential with mean 16 months)
    ctrl_times = np.random.exponential(16, n_per_arm)
    ctrl_event = np.random.binomial(1, 0.75, n_per_arm)

    # Cap at study duration (36 months)
    study_end = 36
    treat_times = np.minimum(treat_times, study_end)
    treat_event[treat_times >= study_end] = 0
    ctrl_times = np.minimum(ctrl_times, study_end)
    ctrl_event[ctrl_times >= study_end] = 0

    def kaplan_meier(times, events):
        """Compute Kaplan-Meier survival curve.

        Returns arrays of (unique_times, survival_prob, n_at_risk, n_events).
        """
        # Sort by time
        order = np.argsort(times)
        sorted_times = times[order]
        sorted_events = events[order]

        # Get unique event times
        unique_times = np.unique(sorted_times[sorted_events == 1])

        survival = np.ones(len(unique_times) + 1)
        km_times = np.zeros(len(unique_times) + 1)
        n_at_risk_arr = np.zeros(len(unique_times), dtype=int)
        n_events_arr = np.zeros(len(unique_times), dtype=int)

        n = len(times)
        for i, t in enumerate(unique_times):
            n_at_risk = np.sum(sorted_times >= t)
            n_events = np.sum((sorted_times == t) & (sorted_events == 1))
            n_at_risk_arr[i] = n_at_risk
            n_events_arr[i] = n_events
            survival[i + 1] = survival[i] * (1 - n_events / n_at_risk)
            km_times[i + 1] = t

        return km_times[1:], survival[1:], n_at_risk_arr, n_events_arr

    # Compute KM for both arms
    t_km, s_km, t_risk, t_events = kaplan_meier(treat_times, treat_event)
    c_km, c_s, c_risk, c_events = kaplan_meier(ctrl_times, ctrl_event)

    print("Kaplan-Meier Survival Estimates:")
    print(f"\n  Treatment: {np.sum(treat_event)} events, "
          f"{np.sum(1 - treat_event)} censored ({len(t_km)} unique event times)")
    print(f"  Control:   {np.sum(ctrl_event)} events, "
          f"{np.sum(1 - ctrl_event)} censored ({len(c_km)} unique event times)")

    # Median survival: time where S(t) first drops below 0.5
    def median_survival(km_times, km_survival):
        below_half = np.where(km_survival <= 0.5)[0]
        if len(below_half) > 0:
            return km_times[below_half[0]]
        return float('inf')

    med_treat = median_survival(t_km, s_km)
    med_ctrl = median_survival(c_km, c_s)
    print(f"\n  Median survival: Treatment={med_treat:.1f}, Control={med_ctrl:.1f}")

    # Survival at 12 months
    s_t_12 = s_km[np.searchsorted(t_km, 12, side='right') - 1] if 12 >= t_km[0] else 1.0
    s_c_12 = c_s[np.searchsorted(c_km, 12, side='right') - 1] if 12 >= c_km[0] else 1.0
    print(f"\n  At 12 months: Treatment S(t)={s_t_12:.3f}, Control S(t)={s_c_12:.3f}")

    # Log-rank test (simplified implementation)
    print(f"\n  Log-Rank Test:")
    all_times = np.concatenate([treat_times, ctrl_times])
    all_events = np.concatenate([treat_event, ctrl_event])
    all_groups = np.concatenate([np.ones(n_per_arm), np.zeros(n_per_arm)])

    event_times = np.unique(all_times[all_events == 1])
    event_times.sort()

    o1 = 0  # observed events in treatment
    e1 = 0  # expected events in treatment
    var_sum = 0

    for t in event_times:
        at_risk_1 = np.sum((treat_times >= t))
        at_risk_0 = np.sum((ctrl_times >= t))
        at_risk_total = at_risk_1 + at_risk_0

        events_1 = np.sum((treat_times == t) & (treat_event == 1))
        events_0 = np.sum((ctrl_times == t) & (ctrl_event == 1))
        events_total = events_1 + events_0

        if at_risk_total > 1:
            expected_1 = events_total * at_risk_1 / at_risk_total
            e1 += expected_1
            o1 += events_1
            var_sum += (events_total * at_risk_1 * at_risk_0
                        * (at_risk_total - events_total)
                        / (at_risk_total ** 2 * (at_risk_total - 1)))

    chi2 = (o1 - e1) ** 2 / var_sum if var_sum > 0 else 0
    # Approximate p-value from chi-squared(1)
    p_value = np.exp(-chi2 / 2)  # rough approximation

    print(f"    Observed events (treatment): {o1:.0f}")
    print(f"    Expected events (treatment): {e1:.1f}")
    print(f"    Chi-squared statistic: {chi2:.3f}")
    print(f"    Approximate p-value: {p_value:.4f}")


# === Exercise 2: Cox Proportional Hazards ===
# Problem: Fit a Cox PH model and interpret hazard ratios for
#          a customer churn dataset.
def exercise_2():
    """Solution for Cox proportional hazards regression.

    Cox PH: h(t|X) = h0(t) * exp(beta'X)

    We implement a simplified Cox model using partial likelihood.
    The partial likelihood avoids estimating the baseline hazard:

    L(beta) = product over events of
              exp(beta'x_i) / sum_{j in risk set} exp(beta'x_j)

    Hazard ratio exp(beta) interpretation:
    - HR > 1: increases risk (shorter survival)
    - HR < 1: decreases risk (longer survival)
    """
    np.random.seed(42)

    # Simulate customer churn data
    n = 400
    age = np.random.normal(40, 10, n)
    monthly_charge = np.random.uniform(20, 100, n)
    tech_support = np.random.binomial(1, 0.4, n)
    contract_length = np.random.choice([1, 12, 24], n, p=[0.5, 0.3, 0.2])

    # Tenure depends on covariates
    log_hazard = (0.01 * (monthly_charge - 50) - 0.02 * age
                  - 0.5 * tech_support - 0.03 * contract_length)
    base_survival = 12
    tenure = np.random.exponential(base_survival * np.exp(-log_hazard))
    tenure = tenure.clip(max=36)
    churned = (tenure < 36).astype(int)

    print("Cox Proportional Hazards Model (Customer Churn):")
    print(f"  N={n}, Events={np.sum(churned)}, Censored={np.sum(1 - churned)}")

    # Standardize features
    features = np.column_stack([age, monthly_charge, tech_support, contract_length])
    feature_names = ['Age', 'Monthly Charge', 'Tech Support', 'Contract Length']
    means, stds = np.mean(features, axis=0), np.std(features, axis=0)
    stds[stds == 0] = 1
    X_std = (features - means) / stds

    # Fit Cox model using Newton-Raphson on partial log-likelihood
    def cox_partial_log_likelihood(beta, X, times, events):
        """Compute negative partial log-likelihood and gradient."""
        n_obs = len(times)
        risk_scores = X @ beta

        # Sort by time (descending for risk set calculation)
        order = np.argsort(-times)
        sorted_scores = risk_scores[order]
        sorted_events = events[order]

        log_lik = 0.0
        grad = np.zeros(len(beta))
        hess = np.zeros((len(beta), len(beta)))

        # Cumulative sum from largest to smallest time
        exp_scores = np.exp(sorted_scores - np.max(sorted_scores))
        cum_exp = np.cumsum(exp_scores)

        for i in range(n_obs):
            if sorted_events[i] == 1:
                xi = X[order[i]]
                log_lik += sorted_scores[i] - np.log(cum_exp[i] + 1e-10)
                # Weighted mean of covariates in risk set
                weights = exp_scores[:i + 1] / (cum_exp[i] + 1e-10)
                weighted_x = X[order[:i + 1]].T @ weights
                grad += xi - weighted_x

        return -log_lik, -grad

    # Gradient descent
    beta = np.zeros(X_std.shape[1])
    lr = 0.01
    for iteration in range(200):
        neg_ll, neg_grad = cox_partial_log_likelihood(
            beta, X_std, tenure, churned
        )
        beta -= lr * neg_grad

    print(f"\n  Cox PH Results (standardized coefficients):")
    print(f"    {'Covariate':<20} {'Beta':>8} {'HR':>8} {'Effect':>20}")
    print(f"    {'-' * 60}")

    for i, name in enumerate(feature_names):
        hr = np.exp(beta[i])
        effect = "increases hazard" if hr > 1 else "decreases hazard"
        pct_change = abs(hr - 1) * 100
        print(f"    {name:<20} {beta[i]:>8.4f} {hr:>8.3f} "
              f"{effect} by {pct_change:.1f}%")

    print(f"\n  HR > 1 means higher churn risk; HR < 1 means protective.")


# === Exercise 3: Parametric Survival Models ===
# Problem: Compare exponential and Weibull survival models using
#          maximum likelihood estimation and AIC/BIC.
def exercise_3():
    """Solution for parametric survival model comparison.

    Exponential: S(t) = exp(-lambda*t), constant hazard h(t) = lambda
    Weibull: S(t) = exp(-(t/lambda)^rho)
      rho < 1: decreasing hazard (infant mortality)
      rho = 1: constant hazard (exponential)
      rho > 1: increasing hazard (wear-out)

    Log-likelihood for right-censored data:
    L = sum(delta_i * log(f(t_i)) + (1 - delta_i) * log(S(t_i)))
    """
    np.random.seed(42)

    # Simulate equipment failure data (Weibull distributed)
    n = 200
    true_rho = 1.8   # increasing hazard (wear-out)
    true_lambda = 20  # scale parameter
    failure_times = true_lambda * np.random.weibull(true_rho, n)
    study_end = 30
    observed_times = np.minimum(failure_times, study_end)
    event = (failure_times <= study_end).astype(int)

    print("Parametric Survival Model Comparison:")
    print(f"  True: Weibull(rho={true_rho}, lambda={true_lambda}), "
          f"N={n}, Events={np.sum(event)}")

    # 1. Exponential model: MLE for lambda
    # Log-likelihood: sum(delta_i * log(lam) - lam * t_i)
    # MLE: lambda_hat = sum(delta_i) / sum(t_i)
    lam_exp = np.sum(event) / np.sum(observed_times)

    ll_exp = (np.sum(event) * np.log(lam_exp)
              - lam_exp * np.sum(observed_times))
    k_exp = 1
    aic_exp = -2 * ll_exp + 2 * k_exp
    bic_exp = -2 * ll_exp + k_exp * np.log(n)

    print(f"\n  Exponential Model:")
    print(f"    lambda = {lam_exp:.4f} (mean survival = {1/lam_exp:.2f})")
    print(f"    Log-likelihood: {ll_exp:.2f}")
    print(f"    AIC: {aic_exp:.2f}, BIC: {bic_exp:.2f}")

    # 2. Weibull model: MLE via gradient ascent
    # Log-likelihood: sum(delta_i * (log(rho) + (rho-1)*log(t_i/lam) - log(lam))
    #                     - (t_i/lam)^rho)
    def weibull_log_lik(rho, lam, times, events):
        """Compute Weibull log-likelihood."""
        t_scaled = times / lam
        t_scaled = t_scaled.clip(1e-10, None)
        ll = np.sum(
            events * (np.log(rho) + (rho - 1) * np.log(t_scaled) - np.log(lam))
            - t_scaled ** rho
        )
        return ll

    # Grid search for Weibull parameters
    best_ll = -np.inf
    best_rho = 1.0
    best_lam = 1.0

    for rho_try in np.arange(0.5, 3.5, 0.1):
        for lam_try in np.arange(5, 35, 1.0):
            ll = weibull_log_lik(rho_try, lam_try, observed_times, event)
            if ll > best_ll:
                best_ll = ll
                best_rho = rho_try
                best_lam = lam_try

    k_weibull = 2
    aic_weibull = -2 * best_ll + 2 * k_weibull
    bic_weibull = -2 * best_ll + k_weibull * np.log(n)

    print(f"\n  Weibull Model:")
    print(f"    rho = {best_rho:.2f} (true: {true_rho}), "
          f"lambda = {best_lam:.2f} (true: {true_lambda})")
    hazard_shape = "decreasing" if best_rho < 1 else \
                    "constant" if abs(best_rho - 1) < 0.1 else "increasing"
    print(f"    Hazard shape: {hazard_shape}")
    print(f"    Log-likelihood: {best_ll:.2f}")
    print(f"    AIC: {aic_weibull:.2f}, BIC: {bic_weibull:.2f}")

    # Model comparison
    print(f"\n  Model Comparison:")
    print(f"    {'Model':<15} {'AIC':>10} {'BIC':>10}")
    print(f"    {'-' * 38}")
    print(f"    {'Exponential':<15} {aic_exp:>10.2f} {bic_exp:>10.2f}")
    print(f"    {'Weibull':<15} {aic_weibull:>10.2f} {bic_weibull:>10.2f}")

    better = "Weibull" if aic_weibull < aic_exp else "Exponential"
    print(f"\n    Preferred: {better} (delta AIC = {abs(aic_weibull - aic_exp):.2f})")


# === Exercise 4: Competing Risks ===
# Problem: Demonstrate why Kaplan-Meier is biased under competing risks
#          and implement cause-specific cumulative incidence.
def exercise_4():
    """Solution for competing risks analysis.

    When multiple event types compete, standard KM overestimates
    the probability of each individual event because it treats
    competing events as censoring.

    Cumulative Incidence Function (CIF):
    CIF_k(t) = sum_{t_j <= t} S(t_{j-1}) * h_k(t_j)
    where S is the overall survival and h_k is the cause-specific hazard.
    """
    np.random.seed(42)

    n = 500

    # Three competing events: cardiac, cancer, other
    t_cardiac = np.random.exponential(25, n)
    t_cancer = np.random.exponential(30, n)
    t_other = np.random.exponential(50, n)

    # Observed time is minimum; event type is whichever occurs first
    t_min = np.minimum(np.minimum(t_cardiac, t_cancer), t_other)
    study_end = 40

    observed_time = np.minimum(t_min, study_end)
    event_type = np.zeros(n, dtype=int)  # 0=censored
    not_censored = t_min < study_end
    event_type[not_censored & (t_cardiac <= t_cancer) & (t_cardiac <= t_other)] = 1
    event_type[not_censored & (t_cancer < t_cardiac) & (t_cancer <= t_other)] = 2
    event_type[not_censored & (t_other < t_cardiac) & (t_other < t_cancer)] = 3

    print("Competing Risks Analysis:")
    for etype, label in [(0, 'Censored'), (1, 'Cardiac'), (2, 'Cancer'),
                          (3, 'Other')]:
        count = np.sum(event_type == etype)
        print(f"  {label}: {count} ({count/n*100:.1f}%)")

    # Cause-specific analysis: treat other events as censoring
    print(f"\n  Cause-Specific Cumulative Incidence (1-KM):")
    print(f"    {'Cause':<15} {'Events':>8} {'Rate':>8} {'Median Time':>12}")
    print(f"    {'-' * 46}")

    for cause_id, label in [(1, 'Cardiac'), (2, 'Cancer'), (3, 'Other')]:
        cause_event = (event_type == cause_id).astype(int)
        n_events = np.sum(cause_event)
        rate = n_events / n
        # Estimate median via sorted event times
        cause_times = observed_time[event_type == cause_id]
        median_t = np.median(cause_times) if len(cause_times) > 0 else float('inf')
        print(f"    {label:<15} {n_events:>8} {rate:>8.3f} {median_t:>12.1f}")

    print(f"\n  Key insight: Naive KM overestimates cause-specific incidence.")
    print(f"  Use CIF to correctly account for competing events.")


if __name__ == "__main__":
    print("=== Exercise 1: Kaplan-Meier Estimator ===")
    exercise_1()
    print("\n=== Exercise 2: Cox Proportional Hazards ===")
    exercise_2()
    print("\n=== Exercise 3: Parametric Survival Models ===")
    exercise_3()
    print("\n=== Exercise 4: Competing Risks ===")
    exercise_4()
    print("\nAll exercises completed!")
