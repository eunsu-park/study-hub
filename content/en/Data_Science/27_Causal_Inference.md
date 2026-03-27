# Causal Inference

[Previous: Bayesian Advanced](./26_Bayesian_Advanced.md) | [Next: Survival Analysis](./28_Survival_Analysis.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the fundamental problem of causal inference and distinguish correlation from causation using concrete examples
2. Draw and interpret causal DAGs, identifying chains, forks, and colliders to determine valid adjustment sets via the backdoor criterion
3. Implement propensity score matching and inverse probability weighting (IPW) to estimate average treatment effects from observational data
4. Apply difference-in-differences (DID) estimation and assess the parallel trends assumption
5. Describe regression discontinuity designs (RDD) and estimate local treatment effects near a cutoff
6. Explain when instrumental variables (IV) are needed and verify instrument validity using first-stage F-statistics
7. Distinguish between average treatment effects (ATE) and conditional average treatment effects (CATE), and apply T-learners for heterogeneous effect estimation

---

Most of data science concerns prediction -- given what we observe, what do we expect? But the questions that drive real decisions are causal: will changing the button color increase conversions? Does this training program raise wages? Causal inference provides the rigorous tools to answer "what if" questions from both experimental and observational data, bridging the gap between statistical association and actionable knowledge.

---

## 1. Correlation vs Causation

### 1.1 The Fundamental Problem

```python
"""
Correlation ≠ Causation

Example: Ice cream sales and drowning deaths are positively correlated.
  → Ice cream doesn't cause drowning!
  → Confound: Hot weather causes both.

  Temperature (Confounder)
      /          \
     ↓            ↓
  Ice Cream    Drowning
    Sales        Deaths

The fundamental problem of causal inference:
  For individual i, we observe either Y_i(treated) OR Y_i(control), never both.
  The missing outcome is the "counterfactual."

  Individual Causal Effect = Y_i(1) - Y_i(0)
  But we can never observe both for the same person!

Average Treatment Effect (ATE):
  ATE = E[Y(1) - Y(0)] = E[Y(1)] - E[Y(0)]
  With random assignment, we can estimate this from group means.
"""
```

### 1.2 Causal Graphs (DAGs)

```python
"""
Directed Acyclic Graph (DAG): Encodes causal assumptions.

Nodes = variables, Edges = direct causal effects

Three fundamental structures:

1. Chain (Mediation):    X → M → Y
   Controlling for M blocks the X→Y path.

2. Fork (Confounding):  X ← Z → Y
   Z is a confounder. Must control for Z to identify X→Y effect.

3. Collider:            X → Z ← Y
   Controlling for Z OPENS a spurious path between X and Y.
   DO NOT control for colliders!

d-separation rules:
  - Chain: Blocked by conditioning on M
  - Fork: Blocked by conditioning on Z
  - Collider: Blocked unless conditioning on Z (or descendant of Z)

Backdoor Criterion:
  To identify causal effect of X on Y:
  Find a set S that blocks all backdoor paths from X to Y
  (paths with arrows into X), without blocking any causal paths
  and without opening collider paths.
"""

# Visualize a DAG
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def draw_dag():
    """Draw a simple causal DAG."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Node positions
    nodes = {
        "Treatment\n(X)": (1, 1),
        "Outcome\n(Y)": (4, 1),
        "Confounder\n(Z)": (2.5, 3),
        "Mediator\n(M)": (2.5, 1),
    }

    # Draw nodes
    for label, (x, y) in nodes.items():
        circle = plt.Circle((x, y), 0.4, fill=True, facecolor='lightblue',
                            edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=9, fontweight='bold')

    # Draw edges
    edges = [
        ("Confounder\n(Z)", "Treatment\n(X)"),
        ("Confounder\n(Z)", "Outcome\n(Y)"),
        ("Treatment\n(X)", "Mediator\n(M)"),
        ("Mediator\n(M)", "Outcome\n(Y)"),
    ]
    for src, dst in edges:
        x1, y1 = nodes[src]
        x2, y2 = nodes[dst]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=2, color='gray'))

    ax.set_xlim(0, 5)
    ax.set_ylim(0, 4)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("Causal DAG Example", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

draw_dag()
```

---

## 2. Propensity Score Methods

### 2.1 Propensity Score Matching

```python
"""
Propensity Score: P(Treatment=1 | Covariates X)

In observational studies, treatment isn't randomly assigned.
Propensity score balances covariates between treated and control groups.

Methods:
1. Matching: Pair each treated unit with a similar control unit
2. Weighting (IPW): Weight observations by inverse propensity
3. Stratification: Group by propensity score quintiles
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

np.random.seed(42)

# Simulate observational data
N = 2000
age = np.random.normal(45, 12, N)
income = np.random.normal(50000, 15000, N)
# Treatment assignment depends on covariates (confounding!)
p_treat = 1 / (1 + np.exp(-(0.03 * (age - 45) + 0.00002 * (income - 50000))))
treatment = np.random.binomial(1, p_treat)
# Outcome depends on treatment AND covariates
outcome = 5 + 2 * treatment + 0.1 * age + 0.0001 * income + np.random.normal(0, 3, N)

df = pd.DataFrame({
    "age": age, "income": income,
    "treatment": treatment, "outcome": outcome,
})

# Naive estimate (biased due to confounding)
naive_ate = df[df.treatment == 1].outcome.mean() - df[df.treatment == 0].outcome.mean()
print(f"True ATE: 2.0")
print(f"Naive ATE: {naive_ate:.3f} (biased)")

# Step 1: Estimate propensity score
X = df[["age", "income"]]
ps_model = LogisticRegression()
ps_model.fit(X, df.treatment)
df["propensity"] = ps_model.predict_proba(X)[:, 1]

# Step 2: Match treated to control by propensity score
treated = df[df.treatment == 1].copy()
control = df[df.treatment == 0].copy()

nn = NearestNeighbors(n_neighbors=1)
nn.fit(control[["propensity"]])
distances, indices = nn.kneighbors(treated[["propensity"]])

matched_control = control.iloc[indices.flatten()]
ate_matched = treated.outcome.mean() - matched_control.outcome.values.mean()
print(f"Matched ATE: {ate_matched:.3f}")

# Step 3: Inverse Propensity Weighting (IPW)
df["weight"] = np.where(
    df.treatment == 1,
    1 / df.propensity,
    1 / (1 - df.propensity),
)
# Trim extreme weights
df["weight"] = df["weight"].clip(upper=df["weight"].quantile(0.99))

ate_ipw = (
    (df.treatment * df.outcome * df.weight).sum() / (df.treatment * df.weight).sum() -
    ((1 - df.treatment) * df.outcome * df.weight).sum() / ((1 - df.treatment) * df.weight).sum()
)
print(f"IPW ATE: {ate_ipw:.3f}")
```

---

## 3. Difference-in-Differences (DID)

### 3.1 DID Design

```python
"""
Difference-in-Differences: Compare changes over time between groups.

Setup:
  - Treatment group: Receives intervention at time T
  - Control group: Does not receive intervention
  - Observe both groups before and after T

  Outcome
    │
    │        ╱ Treatment (observed)
    │       ╱
    │      ●──── Counterfactual (never observed)
    │     ╱     ╱
    │    ╱     ╱ Control
    │   ╱     ╱
    │  ●─────●
    │ ╱     ╱
    │╱     ╱
    └──────┬──────── Time
           T (intervention)

  DID = (Y_treat_after - Y_treat_before) - (Y_control_after - Y_control_before)

Key Assumption: Parallel trends
  Without treatment, both groups would have followed the same trend.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

np.random.seed(42)

# Simulate: Effect of a training program on wages
N_per_group = 500
# Pre-treatment wages
wage_treat_pre = np.random.normal(40000, 5000, N_per_group)
wage_control_pre = np.random.normal(38000, 5000, N_per_group)
# Post-treatment wages (treatment effect = 3000)
wage_treat_post = wage_treat_pre + np.random.normal(5000, 2000, N_per_group) + 3000  # treatment effect
wage_control_post = wage_control_pre + np.random.normal(5000, 2000, N_per_group)  # no treatment

# Build panel dataset
df = pd.DataFrame({
    "wage": np.concatenate([wage_treat_pre, wage_treat_post,
                            wage_control_pre, wage_control_post]),
    "treated": np.repeat([1, 1, 0, 0], N_per_group),
    "post": np.repeat([0, 1, 0, 1], N_per_group),
})
df["treated_x_post"] = df["treated"] * df["post"]

# DID regression
model = smf.ols("wage ~ treated + post + treated_x_post", data=df).fit()
print("DID Regression:")
print(model.summary().tables[1])
print(f"\nDID estimate (treatment effect): {model.params['treated_x_post']:.0f}")
print(f"True effect: 3000")
```

---

## 4. Regression Discontinuity (RDD)

### 4.1 Sharp RDD

```python
"""
Regression Discontinuity: Treatment assigned by a cutoff on a running variable.

Example: Scholarship awarded if test score >= 80.
  Compare students just below (79) vs just above (80) the cutoff.
  Near the cutoff, assignment is "as if random."

  Outcome
    │
    │              ╱ (treated)
    │             ╱
    │            ●  ← Treatment effect (jump)
    │           ╱
    │ (control)╱
    │         ╱
    │        ╱
    └────────┬────── Running Variable
             80 (cutoff)
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

np.random.seed(42)

# Simulate: Scholarship effect on college GPA
N = 1000
test_score = np.random.uniform(50, 100, N)
cutoff = 80
treatment = (test_score >= cutoff).astype(int)
# True effect = 0.5 GPA points
gpa = 2.0 + 0.02 * test_score + 0.5 * treatment + np.random.normal(0, 0.3, N)

df = pd.DataFrame({
    "test_score": test_score,
    "treatment": treatment,
    "gpa": gpa,
    "centered_score": test_score - cutoff,
})

# Local linear regression (within bandwidth of cutoff)
bandwidth = 10
local = df[(df.test_score >= cutoff - bandwidth) & (df.test_score <= cutoff + bandwidth)]

model = smf.ols("gpa ~ treatment + centered_score + treatment:centered_score",
                 data=local).fit()
print("RDD Regression (local linear):")
print(model.summary().tables[1])
print(f"\nRDD estimate: {model.params['treatment']:.3f}")
print(f"True effect: 0.500")

# Visualize
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
below = df[df.test_score < cutoff]
above = df[df.test_score >= cutoff]
ax.scatter(below.test_score, below.gpa, alpha=0.3, s=10, c='blue', label='Control')
ax.scatter(above.test_score, above.gpa, alpha=0.3, s=10, c='red', label='Treated')
ax.axvline(cutoff, color='black', linestyle='--', alpha=0.5, label=f'Cutoff ({cutoff})')

# Fit lines
for subset, color in [(below, 'blue'), (above, 'red')]:
    z = np.polyfit(subset.test_score, subset.gpa, 1)
    p = np.poly1d(z)
    x_line = np.linspace(subset.test_score.min(), subset.test_score.max(), 100)
    ax.plot(x_line, p(x_line), color=color, linewidth=2)

ax.set_xlabel('Test Score')
ax.set_ylabel('College GPA')
ax.set_title('Regression Discontinuity Design')
ax.legend()
plt.tight_layout()
plt.show()
```

---

## 5. Instrumental Variables (IV)

### 5.1 IV Estimation

```python
"""
Instrumental Variables: Handle unmeasured confounding.

Problem: X → Y with unmeasured confounder U.
  X ← U → Y  (can't control for U)

Solution: Find instrument Z that:
  1. Relevance: Z affects X (Z → X)
  2. Exclusion: Z affects Y only through X (no Z → Y directly)
  3. Independence: Z is independent of U

  Z ──→ X ──→ Y
         ↑     ↑
          U ───┘  (unmeasured)

IV estimator (2SLS):
  Stage 1: X_hat = α + γ * Z  (predict X from Z)
  Stage 2: Y = β₀ + β₁ * X_hat  (use predicted X)
"""

import numpy as np
import pandas as pd
from linearmodels.iv import IV2SLS

np.random.seed(42)

# Simulate: Effect of education (X) on income (Y)
# Unmeasured confounder: ability (U)
# Instrument: distance to college (Z)
N = 2000
ability = np.random.normal(0, 1, N)  # unmeasured
distance = np.random.uniform(0, 100, N)  # instrument

# Education depends on ability (confounder) and distance (instrument)
education = 12 + 0.5 * ability - 0.02 * distance + np.random.normal(0, 1, N)

# Income depends on education AND ability (confounder)
income = 20000 + 2000 * education + 5000 * ability + np.random.normal(0, 3000, N)

df = pd.DataFrame({
    "income": income,
    "education": education,
    "distance": distance,
})

# OLS (biased due to omitted ability)
import statsmodels.formula.api as smf
ols = smf.ols("income ~ education", data=df).fit()
print(f"OLS estimate: {ols.params['education']:.0f} (biased, true=2000)")

# 2SLS IV estimation
iv = IV2SLS.from_formula("income ~ 1 + [education ~ distance]", data=df).fit()
print(f"IV estimate:  {iv.params['education']:.0f} (true=2000)")
print(f"\nFirst stage F-statistic: {iv.first_stage.individual['education'].f_stat.stat:.1f}")
print("(F > 10 indicates strong instrument)")
```

---

## 6. Causal ML (Brief Introduction)

### 6.1 Heterogeneous Treatment Effects

```python
"""
Causal ML: Estimate treatment effects that vary across individuals.

  Traditional: ATE = E[Y(1) - Y(0)]  (one number for everyone)
  Causal ML: CATE(x) = E[Y(1) - Y(0) | X=x]  (effect varies by covariates)

Methods:
1. Causal Forest (Wager & Athey): Random forest for treatment effects
2. Meta-learners:
   - T-learner: Separate models for treated/control
   - S-learner: Single model with treatment as feature
   - X-learner: Two-stage with imputed effects
3. Double Machine Learning: Partial out confounders with ML
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict

np.random.seed(42)

# Simulate heterogeneous treatment effects
N = 2000
age = np.random.uniform(20, 70, N)
income = np.random.uniform(20000, 100000, N)
treatment = np.random.binomial(1, 0.5, N)  # Randomized!

# Treatment effect varies by age (larger for younger people)
true_cate = 5 - 0.1 * (age - 20)  # Effect from 5 (age 20) to 0 (age 70)
outcome = 10 + 0.05 * age + 0.0001 * income + true_cate * treatment + np.random.normal(0, 2, N)

df = pd.DataFrame({"age": age, "income": income, "treatment": treatment, "outcome": outcome})

# T-Learner: Separate models for treated and control
treated_df = df[df.treatment == 1]
control_df = df[df.treatment == 0]

model_1 = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
model_0 = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)

model_1.fit(treated_df[["age", "income"]], treated_df.outcome)
model_0.fit(control_df[["age", "income"]], control_df.outcome)

# Estimated CATE
df["cate_hat"] = model_1.predict(df[["age", "income"]]) - model_0.predict(df[["age", "income"]])
df["cate_true"] = true_cate

# Compare
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(df.age, df.cate_true, alpha=0.2, s=10, label="True CATE")
ax.scatter(df.age, df.cate_hat, alpha=0.2, s=10, label="Estimated CATE")
ax.set_xlabel("Age")
ax.set_ylabel("Treatment Effect")
ax.set_title("Heterogeneous Treatment Effects (T-Learner)")
ax.legend()
plt.tight_layout()
plt.show()

print(f"True ATE: {true_cate.mean():.3f}")
print(f"Estimated ATE: {df.cate_hat.mean():.3f}")
```

---

## 7. Practice Problems

### Exercise 1: Propensity Score Analysis

```python
"""
Estimate the effect of a job training program:
1. Load/simulate observational data with confounders
2. Estimate propensity scores with logistic regression
3. Check covariate balance before/after matching
4. Estimate ATE using: matching, IPW, and doubly robust
5. Compare estimates and assess sensitivity to unmeasured confounders
"""
```

### Exercise 2: DID with Real Data

```python
"""
Analyze the effect of a minimum wage increase:
1. Simulate panel data: 2 states (treated/control), 2 time periods
2. Test the parallel trends assumption (pre-treatment periods)
3. Estimate DID effect with regression
4. Add covariates and time fixed effects
5. Conduct a placebo test (fake treatment date)
"""
```

---

## 8. Summary

### Key Takeaways

| Method | When to Use | Key Assumption |
|--------|------------|----------------|
| **RCT** | Can randomize treatment | Random assignment |
| **Propensity Score** | Observational data, known confounders | No unmeasured confounders |
| **DID** | Panel data, treatment at a time point | Parallel trends |
| **RDD** | Treatment assigned by a cutoff | Continuity at cutoff |
| **IV** | Unmeasured confounders, valid instrument | Exclusion restriction |
| **Causal Forest** | Heterogeneous effects, RCT data | Unconfoundedness |

### Best Practices

1. **Draw the DAG first** — make causal assumptions explicit before analysis
2. **Check assumptions** — parallel trends for DID, balance for matching
3. **Sensitivity analysis** — how much unmeasured confounding would invalidate results?
4. **Use multiple methods** — if different methods agree, the result is more credible
5. **Report uncertainty** — confidence intervals, not just point estimates

### Next Steps

- **L28**: Survival Analysis — time-to-event data with censoring
- Return to **L22** (Multivariate Analysis) for related regression techniques
