# Survival Analysis

[Previous: Causal Inference](./27_Causal_Inference.md) | [Next: Modern Data Tools](./29_Modern_Data_Tools.md)

## Overview

Survival analysis models time-to-event data where some observations are censored (event not yet observed). This lesson covers the Kaplan-Meier estimator, log-rank test, Cox proportional hazards model, parametric models, competing risks, and practical applications using the lifelines library.

---

## 1. Survival Analysis Concepts

### 1.1 Core Definitions

```python
"""
Key Concepts:

1. Survival Time (T): Time from a starting point to an event
   - Medical: diagnosis → death/relapse
   - Business: signup → churn
   - Engineering: deployment → failure

2. Censoring: Event not observed during study period
   - Right censoring (most common): study ends before event
   - Left censoring: event happened before observation started
   - Interval censoring: event between two observation times

   Example (right censoring):
   Patient A: ──────────×  (died at month 8)
   Patient B: ────────────── (alive at month 12, censored)
   Patient C: ──×            (died at month 3)
   Patient D: ────────────── (alive at month 12, censored)

3. Survival Function: S(t) = P(T > t)
   Probability of surviving beyond time t.
   S(0) = 1, S(∞) = 0, monotonically decreasing.

4. Hazard Function: h(t) = lim_{dt→0} P(t ≤ T < t+dt | T ≥ t) / dt
   Instantaneous rate of event at time t, given survival to time t.
   NOT a probability (can exceed 1).

5. Cumulative Hazard: H(t) = ∫₀ᵗ h(s)ds = -log S(t)
   S(t) = exp(-H(t))
"""
```

---

## 2. Kaplan-Meier Estimator

### 2.1 Nonparametric Survival Curve

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

# Simulated clinical trial data
np.random.seed(42)
n = 200

# Treatment group: longer survival
treatment_times = np.random.exponential(scale=24, size=n)
treatment_event = np.random.binomial(1, 0.7, n)  # 30% censored

# Control group: shorter survival
control_times = np.random.exponential(scale=16, size=n)
control_event = np.random.binomial(1, 0.7, n)

# Cap at study duration (36 months)
study_end = 36
treatment_times = np.minimum(treatment_times, study_end)
treatment_event[treatment_times >= study_end] = 0
control_times = np.minimum(control_times, study_end)
control_event[control_times >= study_end] = 0

# Fit Kaplan-Meier
kmf_treat = KaplanMeierFitter()
kmf_treat.fit(treatment_times, treatment_event, label="Treatment")

kmf_control = KaplanMeierFitter()
kmf_control.fit(control_times, control_event, label="Control")

# Plot survival curves
fig, ax = plt.subplots(figsize=(10, 6))
kmf_treat.plot_survival_function(ax=ax, ci_show=True)
kmf_control.plot_survival_function(ax=ax, ci_show=True)
ax.set_xlabel("Time (months)")
ax.set_ylabel("Survival Probability")
ax.set_title("Kaplan-Meier Survival Curves")
plt.tight_layout()
plt.show()

# Median survival times
print(f"Treatment median survival: {kmf_treat.median_survival_time_:.1f} months")
print(f"Control median survival: {kmf_control.median_survival_time_:.1f} months")

# Summary at specific times
for t in [6, 12, 24]:
    s_treat = kmf_treat.predict(t)
    s_control = kmf_control.predict(t)
    print(f"  At {t} months: Treatment S(t)={s_treat:.3f}, Control S(t)={s_control:.3f}")
```

### 2.2 Log-Rank Test

```python
# Log-rank test: Are the survival curves significantly different?
result = logrank_test(
    treatment_times, control_times,
    treatment_event, control_event,
)
print(f"\nLog-Rank Test:")
print(f"  Test statistic: {result.test_statistic:.3f}")
print(f"  p-value: {result.p_value:.4f}")
print(f"  Significant (α=0.05): {result.p_value < 0.05}")
```

---

## 3. Cox Proportional Hazards Model

### 3.1 Cox PH Regression

```python
"""
Cox PH Model: h(t|X) = h₀(t) * exp(β₁X₁ + β₂X₂ + ...)

  - h₀(t): baseline hazard (unspecified, nonparametric)
  - exp(βX): proportional effect of covariates
  - Semi-parametric: no assumption about baseline hazard shape

Proportional Hazards Assumption:
  The hazard ratio between any two individuals is constant over time.
  h(t|X₁) / h(t|X₂) = exp(β(X₁ - X₂))  (doesn't depend on t)

Interpretation:
  - exp(β) > 1: covariate increases hazard (worse prognosis)
  - exp(β) < 1: covariate decreases hazard (protective)
  - exp(β) = 1: no effect
"""

from lifelines import CoxPHFitter
import numpy as np
import pandas as pd

np.random.seed(42)
N = 500

# Simulate: Churn analysis
df = pd.DataFrame({
    "tenure": np.random.exponential(12, N).clip(max=36),
    "churned": np.random.binomial(1, 0.6, N),
    "age": np.random.normal(40, 10, N),
    "monthly_charge": np.random.uniform(20, 100, N),
    "contract_type": np.random.choice(["month-to-month", "1-year", "2-year"], N,
                                       p=[0.5, 0.3, 0.2]),
    "tech_support": np.random.binomial(1, 0.4, N),
})

# Higher charges → higher churn
df.loc[df.monthly_charge > 70, "churned"] = np.random.binomial(1, 0.8, (df.monthly_charge > 70).sum())
# Longer contracts → lower churn
df.loc[df.contract_type == "2-year", "tenure"] *= 1.5
df["tenure"] = df["tenure"].clip(max=36)

# One-hot encode
df_encoded = pd.get_dummies(df, columns=["contract_type"], drop_first=True)

# Fit Cox PH model
cph = CoxPHFitter()
cph.fit(df_encoded, duration_col="tenure", event_col="churned")

# Results
cph.print_summary()

# Hazard ratios
print("\nHazard Ratios (exp(coef)):")
for var, hr in cph.hazard_ratios_.items():
    interpretation = "increases" if hr > 1 else "decreases"
    print(f"  {var}: HR={hr:.3f} → {interpretation} hazard by {abs(hr-1)*100:.1f}%")

# Plot coefficients
cph.plot()
plt.title("Cox PH Coefficients")
plt.tight_layout()
plt.show()
```

### 3.2 Checking the PH Assumption

```python
# Test proportional hazards assumption
ph_test = cph.check_assumptions(df_encoded, p_value_threshold=0.05, show_plots=True)

"""
If PH assumption is violated for a variable:
  Options:
  1. Stratify by that variable: CoxPHFitter(strata=['variable'])
  2. Include time-varying coefficient: add interaction with time
  3. Use a parametric model (AFT) instead
"""
```

### 3.3 Prediction and Risk Scores

```python
# Predict survival function for new individuals
new_data = pd.DataFrame({
    "age": [30, 60],
    "monthly_charge": [50, 90],
    "tech_support": [1, 0],
    "contract_type_1-year": [1, 0],
    "contract_type_2-year": [0, 0],
})

# Survival curves for new individuals
surv = cph.predict_survival_function(new_data)
fig, ax = plt.subplots(figsize=(10, 6))
surv.plot(ax=ax)
ax.set_xlabel("Time (months)")
ax.set_ylabel("Survival Probability")
ax.set_title("Predicted Survival Curves")
ax.legend(["Young, low charge, tech support", "Old, high charge, no support"])
plt.tight_layout()
plt.show()

# Median survival prediction
median_surv = cph.predict_median(new_data)
print(f"Predicted median survival:")
print(f"  Profile 1: {median_surv.iloc[0]:.1f} months")
print(f"  Profile 2: {median_surv.iloc[1]:.1f} months")

# Risk scores (partial hazard)
risk = cph.predict_partial_hazard(new_data)
print(f"\nRelative risk scores:")
print(f"  Profile 1: {risk.iloc[0]:.3f}")
print(f"  Profile 2: {risk.iloc[1]:.3f}")
```

---

## 4. Parametric Models

### 4.1 Common Distributions

```python
"""
Parametric Survival Models — assume a distribution for survival times.

| Distribution    | Hazard Shape        | Use Case                     |
|----------------|---------------------|------------------------------|
| Exponential    | Constant h(t)=λ     | Memoryless (rare in practice)|
| Weibull        | Monotone increase/decrease | Equipment failure, aging |
| Log-normal     | Non-monotone (rise then fall) | Disease progression  |
| Log-logistic   | Non-monotone          | Biological systems           |
| Gompertz       | Exponentially increasing | Human mortality             |

Weibull: h(t) = (ρ/λ)(t/λ)^(ρ-1)
  ρ < 1: Decreasing hazard (infant mortality)
  ρ = 1: Constant (exponential)
  ρ > 1: Increasing hazard (wear-out)
"""

from lifelines import (
    WeibullFitter, LogNormalFitter,
    LogLogisticFitter, ExponentialFitter,
    WeibullAFTFitter,
)

# Fit multiple parametric models
fitters = {
    "Exponential": ExponentialFitter(),
    "Weibull": WeibullFitter(),
    "Log-Normal": LogNormalFitter(),
    "Log-Logistic": LogLogisticFitter(),
}

fig, ax = plt.subplots(figsize=(10, 6))
for name, fitter in fitters.items():
    fitter.fit(df["tenure"], df["churned"], label=name)
    fitter.plot_survival_function(ax=ax)
    print(f"{name:15s} AIC={fitter.AIC_:.1f}  BIC={fitter.BIC_:.1f}")

# Add Kaplan-Meier for comparison
kmf = KaplanMeierFitter()
kmf.fit(df["tenure"], df["churned"], label="Kaplan-Meier")
kmf.plot_survival_function(ax=ax, ci_show=False, linestyle="--", color="black")

ax.set_title("Parametric vs Nonparametric Survival")
ax.set_xlabel("Time (months)")
ax.set_ylabel("Survival Probability")
plt.tight_layout()
plt.show()
```

### 4.2 Accelerated Failure Time (AFT)

```python
"""
AFT Model: log(T) = β₀ + β₁X₁ + ... + σε

  Alternative to Cox PH.
  Covariates accelerate or decelerate survival time.
  exp(β) > 1 → slows failure (longer survival)
  exp(β) < 1 → accelerates failure (shorter survival)
"""

aft = WeibullAFTFitter()
aft.fit(df_encoded, duration_col="tenure", event_col="churned")
aft.print_summary()

print("\nAcceleration Factors:")
for var in aft.params_.index.get_level_values(1).unique():
    if var != "Intercept" and var in aft.params_["lambda_"].index:
        af = np.exp(aft.params_["lambda_"][var])
        effect = "extends" if af > 1 else "shortens"
        print(f"  {var}: AF={af:.3f} → {effect} survival by {abs(af-1)*100:.1f}%")
```

---

## 5. Competing Risks

### 5.1 Cumulative Incidence Function

```python
"""
Competing Risks: Multiple possible events (only the first one matters).

Example: Patient can die from:
  - Heart disease (event type 1)
  - Cancer (event type 2)
  - Other causes (event type 3)

The Kaplan-Meier estimator is BIASED with competing risks.
Use the Cumulative Incidence Function (CIF) instead.

CIF_k(t) = P(T ≤ t, event = k)
  = Probability of event k occurring by time t,
    accounting for competing events.

Sum of all CIFs ≤ 1 at any time point.
"""

# Simulate competing risks data
np.random.seed(42)
N = 500

# Time to each event type
t1 = np.random.exponential(20, N)  # Heart disease
t2 = np.random.exponential(25, N)  # Cancer
t3 = np.random.exponential(40, N)  # Other

# Observed time = minimum
T_obs = np.minimum(np.minimum(t1, t2), t3).clip(max=36)
event_type = np.where(T_obs == t1, 1,
             np.where(T_obs == t2, 2,
             np.where(T_obs == t3, 3, 0)))
# Censoring
censored = T_obs >= 36
event_type[censored] = 0

cr_df = pd.DataFrame({
    "time": T_obs,
    "event": event_type,
})

# Crude (cause-specific) Kaplan-Meier for each event
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: KM per cause (treat other events as censored)
for event_id, label in [(1, "Heart Disease"), (2, "Cancer"), (3, "Other")]:
    kmf = KaplanMeierFitter()
    kmf.fit(
        cr_df["time"],
        event_observed=(cr_df["event"] == event_id).astype(int),
        label=label,
    )
    # Plot as 1-S(t) = cumulative incidence
    ci = 1 - kmf.survival_function_
    ci.plot(ax=axes[0])

axes[0].set_title("Cause-Specific Cumulative Incidence")
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Cumulative Incidence")

# Right: Event distribution
event_counts = cr_df["event"].value_counts().sort_index()
labels = ["Censored", "Heart Disease", "Cancer", "Other"]
axes[1].bar(labels, [event_counts.get(i, 0) for i in range(4)])
axes[1].set_title("Event Distribution")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.show()

print(f"Events: {dict(cr_df['event'].value_counts().sort_index())}")
```

---

## 6. Practice Problems

### Exercise 1: Customer Churn Survival

```python
"""
Analyze customer churn using survival analysis:
1. Load telecom churn dataset (or simulate with 5+ covariates)
2. Fit Kaplan-Meier curves by customer segment
3. Perform log-rank tests between segments
4. Build a Cox PH model with all covariates
5. Check the proportional hazards assumption
6. Predict median survival for 3 customer profiles
7. Identify the strongest predictors of churn
"""
```

### Exercise 2: Clinical Trial Analysis

```python
"""
Analyze a simulated clinical trial:
1. Generate data: 2 treatment arms, 300 patients, 3 years follow-up
2. Include covariates: age, stage, biomarker level
3. Compare arms with Kaplan-Meier + log-rank test
4. Fit Cox PH model adjusting for covariates
5. Fit Weibull AFT and compare with Cox PH
6. Calculate number-needed-to-treat (NNT) at 12 months
"""
```

---

## 7. Summary

### Key Takeaways

| Concept | Description |
|---------|-------------|
| **Censoring** | Event not observed during study; must account for in analysis |
| **Kaplan-Meier** | Nonparametric survival curve; no covariates |
| **Log-rank test** | Compare survival between groups |
| **Cox PH** | Semi-parametric regression; hazard ratios |
| **Weibull AFT** | Parametric; acceleration factors instead of hazard ratios |
| **Competing risks** | Multiple event types; use CIF not KM |

### Best Practices

1. **Always check censoring** — heavily censored data yields unreliable estimates
2. **Check PH assumption** — violated PH leads to misleading hazard ratios
3. **Use CIF for competing risks** — Kaplan-Meier is biased when events compete
4. **Compare parametric fits** — use AIC/BIC to select the best distribution
5. **Report median survival** — more interpretable than hazard ratios for non-technical audiences

### Next Steps

- **L29**: Modern Data Tools — Polars, DuckDB for efficient data processing
- Return to **L20** (Time Series Basics) for related temporal analysis
