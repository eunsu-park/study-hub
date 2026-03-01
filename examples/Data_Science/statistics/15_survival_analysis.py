"""
Survival Analysis
=================
Demonstrates:
- Kaplan-Meier estimator
- Log-rank test
- Cox Proportional Hazards model
- Parametric models (Weibull, Log-Normal)
- Competing risks (cumulative incidence)

Requirements:
    pip install lifelines numpy pandas matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import (
    KaplanMeierFitter, CoxPHFitter,
    WeibullFitter, LogNormalFitter, LogLogisticFitter, ExponentialFitter,
    WeibullAFTFitter,
)
from lifelines.statistics import logrank_test


# ── 1. Kaplan-Meier ───────────────────────────────────────────────

def demo_kaplan_meier():
    """Kaplan-Meier survival curves and log-rank test."""
    np.random.seed(42)
    N = 200
    study_end = 36

    # Treatment group (longer survival)
    t_times = np.minimum(np.random.exponential(24, N), study_end)
    t_event = np.random.binomial(1, 0.7, N)
    t_event[t_times >= study_end] = 0

    # Control group (shorter survival)
    c_times = np.minimum(np.random.exponential(16, N), study_end)
    c_event = np.random.binomial(1, 0.7, N)
    c_event[c_times >= study_end] = 0

    # Fit KM
    kmf_t = KaplanMeierFitter().fit(t_times, t_event, label="Treatment")
    kmf_c = KaplanMeierFitter().fit(c_times, c_event, label="Control")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    kmf_t.plot_survival_function(ax=axes[0], ci_show=True)
    kmf_c.plot_survival_function(ax=axes[0], ci_show=True)
    axes[0].set_xlabel("Time (months)")
    axes[0].set_ylabel("Survival Probability")
    axes[0].set_title("Kaplan-Meier Survival Curves")

    # Cumulative hazard
    kmf_t.plot_cumulative_density(ax=axes[1])
    kmf_c.plot_cumulative_density(ax=axes[1])
    axes[1].set_xlabel("Time (months)")
    axes[1].set_ylabel("Cumulative Incidence")
    axes[1].set_title("Cumulative Incidence (1 - Survival)")

    plt.tight_layout()
    plt.show()

    # Median survival
    print(f"Treatment median survival: {kmf_t.median_survival_time_:.1f} months")
    print(f"Control median survival: {kmf_c.median_survival_time_:.1f} months")

    # Log-rank test
    result = logrank_test(t_times, c_times, t_event, c_event)
    print(f"\nLog-Rank Test: stat={result.test_statistic:.3f}, p={result.p_value:.4f}")


# ── 2. Cox PH Model ───────────────────────────────────────────────

def demo_cox_ph():
    """Cox Proportional Hazards regression."""
    np.random.seed(42)
    N = 500

    df = pd.DataFrame({
        "tenure": np.random.exponential(12, N).clip(max=36),
        "churned": np.random.binomial(1, 0.6, N),
        "age": np.random.normal(40, 10, N).clip(18, 80),
        "monthly_charge": np.random.uniform(20, 100, N),
        "tech_support": np.random.binomial(1, 0.4, N),
    })

    # Make outcome correlated with features
    high_charge = df.monthly_charge > 70
    df.loc[high_charge, "churned"] = np.random.binomial(1, 0.8, high_charge.sum())
    df.loc[df.tech_support == 1, "tenure"] *= 1.3
    df["tenure"] = df["tenure"].clip(max=36)

    # Fit
    cph = CoxPHFitter()
    cph.fit(df, duration_col="tenure", event_col="churned")
    cph.print_summary()

    # Hazard ratios
    print("\nHazard Ratios:")
    for var, hr in cph.hazard_ratios_.items():
        direction = "increases" if hr > 1 else "decreases"
        print(f"  {var}: HR={hr:.3f} → {direction} risk by {abs(hr-1)*100:.1f}%")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cph.plot(ax=axes[0])
    axes[0].set_title("Cox PH Coefficients")

    # Predict for profiles
    profiles = pd.DataFrame({
        "age": [30, 50, 65],
        "monthly_charge": [40, 70, 95],
        "tech_support": [1, 1, 0],
    })
    surv = cph.predict_survival_function(profiles)
    surv.columns = ["Young/Low/Support", "Mid/Med/Support", "Old/High/NoSupport"]
    surv.plot(ax=axes[1])
    axes[1].set_xlabel("Time (months)")
    axes[1].set_ylabel("Survival Probability")
    axes[1].set_title("Predicted Survival by Profile")

    plt.tight_layout()
    plt.show()

    return df


# ── 3. Parametric Models ──────────────────────────────────────────

def demo_parametric(df):
    """Compare parametric survival models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    fitters = {
        "Exponential": ExponentialFitter(),
        "Weibull": WeibullFitter(),
        "Log-Normal": LogNormalFitter(),
        "Log-Logistic": LogLogisticFitter(),
    }

    print("Parametric Model Comparison:")
    print(f"{'Model':>15} {'AIC':>10} {'BIC':>10}")
    print("-" * 40)
    for name, fitter in fitters.items():
        fitter.fit(df["tenure"], df["churned"], label=name)
        fitter.plot_survival_function(ax=ax)
        print(f"{name:>15} {fitter.AIC_:>10.1f} {fitter.BIC_:>10.1f}")

    # KM reference
    KaplanMeierFitter().fit(df["tenure"], df["churned"], label="Kaplan-Meier") \
        .plot_survival_function(ax=ax, ci_show=False, linestyle="--", color="black")

    ax.set_title("Parametric vs Nonparametric Survival")
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Survival Probability")
    plt.tight_layout()
    plt.show()


# ── 4. Competing Risks ────────────────────────────────────────────

def demo_competing_risks():
    """Competing risks with cumulative incidence."""
    np.random.seed(42)
    N = 500

    t1 = np.random.exponential(20, N)
    t2 = np.random.exponential(25, N)
    t3 = np.random.exponential(40, N)

    T = np.minimum(np.minimum(t1, t2), t3).clip(max=36)
    event = np.where(T == t1, 1, np.where(T == t2, 2, np.where(T == t3, 3, 0)))
    event[T >= 36] = 0

    df = pd.DataFrame({"time": T, "event": event})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    labels = {1: "Heart Disease", 2: "Cancer", 3: "Other"}
    colors = {1: "red", 2: "blue", 3: "green"}

    for eid, label in labels.items():
        kmf = KaplanMeierFitter()
        kmf.fit(df["time"], (df["event"] == eid).astype(int), label=label)
        ci = 1 - kmf.survival_function_
        ci.plot(ax=axes[0], color=colors[eid])

    axes[0].set_title("Cause-Specific Cumulative Incidence")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Cumulative Incidence")

    # Event distribution
    counts = df["event"].value_counts().sort_index()
    bar_labels = ["Censored"] + [labels.get(i, f"Event {i}") for i in range(1, 4)]
    bar_counts = [counts.get(i, 0) for i in range(4)]
    bar_colors = ["gray"] + [colors[i] for i in range(1, 4)]
    axes[1].bar(bar_labels, bar_counts, color=bar_colors)
    axes[1].set_title("Event Distribution")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.show()

    print("Event counts:")
    for eid, label in {0: "Censored", **labels}.items():
        print(f"  {label}: {(df.event == eid).sum()}")


# ── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    demos = {
        "km": demo_kaplan_meier,
        "cox": lambda: demo_parametric(demo_cox_ph()),
        "competing": demo_competing_risks,
    }

    if len(sys.argv) < 2 or sys.argv[1] not in demos:
        print("Usage: python 15_survival_analysis.py <demo>")
        print(f"Available: {', '.join(demos.keys())}")
        print("\nRunning all demos...")
        demo_kaplan_meier()
        print()
        df = demo_cox_ph()
        print()
        demo_parametric(df)
        print()
        demo_competing_risks()
    else:
        demos[sys.argv[1]]()
