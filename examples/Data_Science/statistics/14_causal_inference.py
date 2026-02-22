"""
Causal Inference
================
Demonstrates:
- Propensity score matching and IPW
- Difference-in-Differences (DID)
- Regression Discontinuity Design (RDD)
- Heterogeneous treatment effects (T-Learner)

Requirements:
    pip install numpy pandas scikit-learn statsmodels matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import GradientBoostingRegressor
import statsmodels.formula.api as smf


# ── 1. Propensity Score Methods ────────────────────────────────────

def demo_propensity_score():
    """Propensity score matching and IPW."""
    np.random.seed(42)
    N = 2000
    TRUE_ATE = 2.0

    # Covariates
    age = np.random.normal(45, 12, N)
    income = np.random.normal(50000, 15000, N)

    # Confounded treatment assignment
    logit = 0.03 * (age - 45) + 0.00002 * (income - 50000)
    p_treat = 1 / (1 + np.exp(-logit))
    treatment = np.random.binomial(1, p_treat)

    # Outcome
    outcome = 5 + TRUE_ATE * treatment + 0.1 * age + 0.0001 * income + np.random.normal(0, 3, N)

    df = pd.DataFrame({"age": age, "income": income, "treatment": treatment, "outcome": outcome})

    # Naive estimate
    naive = df[df.treatment == 1].outcome.mean() - df[df.treatment == 0].outcome.mean()
    print(f"True ATE: {TRUE_ATE}")
    print(f"Naive ATE: {naive:.3f} (biased)")

    # Propensity score
    X = df[["age", "income"]]
    ps = LogisticRegression().fit(X, df.treatment)
    df["ps"] = ps.predict_proba(X)[:, 1]

    # Matching
    treated = df[df.treatment == 1]
    control = df[df.treatment == 0]
    nn = NearestNeighbors(n_neighbors=1).fit(control[["ps"]])
    _, idx = nn.kneighbors(treated[["ps"]])
    ate_match = treated.outcome.mean() - control.iloc[idx.flatten()].outcome.mean()
    print(f"Matched ATE: {ate_match:.3f}")

    # IPW
    w = np.where(df.treatment == 1, 1/df.ps, 1/(1-df.ps))
    w = np.clip(w, 0, np.percentile(w, 99))
    ate_ipw = (
        (df.treatment * df.outcome * w).sum() / (df.treatment * w).sum() -
        ((1-df.treatment) * df.outcome * w).sum() / ((1-df.treatment) * w).sum()
    )
    print(f"IPW ATE: {ate_ipw:.3f}")

    # Balance check
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, var in zip(axes, ["age", "income"]):
        ax.hist(df[df.treatment == 1][var], bins=30, alpha=0.5, label="Treated", density=True)
        ax.hist(df[df.treatment == 0][var], bins=30, alpha=0.5, label="Control", density=True)
        ax.set_title(f"{var} Distribution")
        ax.legend()
    plt.suptitle("Covariate Balance (before matching)")
    plt.tight_layout()
    plt.show()


# ── 2. Difference-in-Differences ──────────────────────────────────

def demo_did():
    """Difference-in-Differences estimation."""
    np.random.seed(42)
    N = 500
    TRUE_EFFECT = 3000

    # Pre-treatment
    wage_treat_pre = np.random.normal(40000, 5000, N)
    wage_ctrl_pre = np.random.normal(38000, 5000, N)

    # Post-treatment (common trend + treatment effect)
    common_trend = 5000
    wage_treat_post = wage_treat_pre + np.random.normal(common_trend, 2000, N) + TRUE_EFFECT
    wage_ctrl_post = wage_ctrl_pre + np.random.normal(common_trend, 2000, N)

    df = pd.DataFrame({
        "wage": np.concatenate([wage_treat_pre, wage_treat_post, wage_ctrl_pre, wage_ctrl_post]),
        "treated": np.repeat([1, 1, 0, 0], N),
        "post": np.repeat([0, 1, 0, 1], N),
    })
    df["treated_x_post"] = df["treated"] * df["post"]

    # DID regression
    model = smf.ols("wage ~ treated + post + treated_x_post", data=df).fit()
    did_estimate = model.params["treated_x_post"]
    ci = model.conf_int().loc["treated_x_post"]

    print(f"True effect: {TRUE_EFFECT}")
    print(f"DID estimate: {did_estimate:.0f}")
    print(f"95% CI: [{ci[0]:.0f}, {ci[1]:.0f}]")
    print(f"p-value: {model.pvalues['treated_x_post']:.4f}")

    # Visualize
    fig, ax = plt.subplots(figsize=(8, 5))
    means = df.groupby(["treated", "post"]).wage.mean().unstack()
    ax.plot([0, 1], means.loc[1], "ro-", markersize=10, linewidth=2, label="Treatment")
    ax.plot([0, 1], means.loc[0], "bs-", markersize=10, linewidth=2, label="Control")
    # Counterfactual
    counterfactual = means.loc[1, 0] + (means.loc[0, 1] - means.loc[0, 0])
    ax.plot([0, 1], [means.loc[1, 0], counterfactual], "r--", alpha=0.5, label="Counterfactual")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pre-treatment", "Post-treatment"])
    ax.set_ylabel("Wage")
    ax.set_title(f"DID: Estimated Effect = {did_estimate:.0f}")
    ax.legend()
    plt.tight_layout()
    plt.show()


# ── 3. Regression Discontinuity ───────────────────────────────────

def demo_rdd():
    """Sharp Regression Discontinuity Design."""
    np.random.seed(42)
    N = 1000
    CUTOFF = 80
    TRUE_EFFECT = 0.5

    score = np.random.uniform(50, 100, N)
    treatment = (score >= CUTOFF).astype(int)
    gpa = 2.0 + 0.02 * score + TRUE_EFFECT * treatment + np.random.normal(0, 0.3, N)

    df = pd.DataFrame({"score": score, "treatment": treatment, "gpa": gpa,
                        "centered": score - CUTOFF})

    # Local linear regression
    bw = 10
    local = df[(df.score >= CUTOFF - bw) & (df.score <= CUTOFF + bw)]
    model = smf.ols("gpa ~ treatment + centered + treatment:centered", data=local).fit()

    print(f"True effect: {TRUE_EFFECT}")
    print(f"RDD estimate: {model.params['treatment']:.3f}")
    print(f"95% CI: [{model.conf_int().loc['treatment'][0]:.3f}, "
          f"{model.conf_int().loc['treatment'][1]:.3f}]")

    # Visualize
    fig, ax = plt.subplots(figsize=(10, 6))
    below = df[df.score < CUTOFF]
    above = df[df.score >= CUTOFF]
    ax.scatter(below.score, below.gpa, alpha=0.2, s=10, c="blue", label="Control")
    ax.scatter(above.score, above.gpa, alpha=0.2, s=10, c="red", label="Treated")
    ax.axvline(CUTOFF, color="black", linestyle="--", alpha=0.5)

    # Fit lines
    for subset, color in [(below, "blue"), (above, "red")]:
        z = np.polyfit(subset.score, subset.gpa, 1)
        xs = np.linspace(subset.score.min(), subset.score.max(), 100)
        ax.plot(xs, np.poly1d(z)(xs), color=color, linewidth=2)

    ax.set_xlabel("Test Score")
    ax.set_ylabel("GPA")
    ax.set_title(f"RDD: Effect = {model.params['treatment']:.3f} (true = {TRUE_EFFECT})")
    ax.legend()
    plt.tight_layout()
    plt.show()


# ── 4. Heterogeneous Treatment Effects ────────────────────────────

def demo_cate():
    """T-Learner for heterogeneous treatment effects."""
    np.random.seed(42)
    N = 2000

    age = np.random.uniform(20, 70, N)
    income = np.random.uniform(20000, 100000, N)
    treatment = np.random.binomial(1, 0.5, N)

    # CATE varies by age
    true_cate = 5 - 0.1 * (age - 20)
    outcome = 10 + 0.05 * age + 0.0001 * income + true_cate * treatment + np.random.normal(0, 2, N)

    df = pd.DataFrame({"age": age, "income": income, "treatment": treatment, "outcome": outcome})

    # T-Learner
    t1 = df[df.treatment == 1]
    t0 = df[df.treatment == 0]

    m1 = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    m0 = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    m1.fit(t1[["age", "income"]], t1.outcome)
    m0.fit(t0[["age", "income"]], t0.outcome)

    df["cate_hat"] = m1.predict(df[["age", "income"]]) - m0.predict(df[["age", "income"]])
    df["cate_true"] = true_cate

    print(f"True ATE: {true_cate.mean():.3f}")
    print(f"Estimated ATE: {df.cate_hat.mean():.3f}")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter
    axes[0].scatter(df.age, df.cate_true, alpha=0.2, s=10, label="True CATE")
    axes[0].scatter(df.age, df.cate_hat, alpha=0.2, s=10, label="Estimated CATE")
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Treatment Effect")
    axes[0].set_title("CATE by Age")
    axes[0].legend()

    # Binned average
    df["age_bin"] = pd.cut(df.age, bins=10)
    binned = df.groupby("age_bin")[["cate_true", "cate_hat"]].mean()
    binned.plot(kind="bar", ax=axes[1])
    axes[1].set_title("Average CATE by Age Group")
    axes[1].set_ylabel("Treatment Effect")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


# ── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    demos = {
        "ps": demo_propensity_score,
        "did": demo_did,
        "rdd": demo_rdd,
        "cate": demo_cate,
    }

    if len(sys.argv) < 2 or sys.argv[1] not in demos:
        print("Usage: python 14_causal_inference.py <demo>")
        print(f"Available: {', '.join(demos.keys())}")
        print("\nRunning all demos...")
        for name, fn in demos.items():
            print(f"\n{'='*60}")
            print(f"  {name.upper()}")
            print(f"{'='*60}")
            fn()
    else:
        demos[sys.argv[1]]()
