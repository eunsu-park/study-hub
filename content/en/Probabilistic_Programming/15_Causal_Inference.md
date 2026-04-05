# 15. Causal Inference

[Previous: Bayesian Deep Learning](./14_Bayesian_Deep_Learning.md) | [Next: Model Comparison](./16_Model_Comparison.md)

---

> **Framework Note**: This lesson uses DoWhy, PyMC, and NumPy for causal inference.
>
> Installation: `pip install dowhy pymc arviz numpy matplotlib networkx`

## Learning Objectives

- Distinguish correlation from causation using structural causal models (SCMs)
- Apply do-calculus to compute interventional distributions
- Implement the backdoor and frontdoor adjustment criteria
- Use instrumental variables for causal estimation
- Build Bayesian causal models with PyMC

---

## 1. Structural Causal Models

An SCM defines the causal mechanism generating each variable as a function of its direct causes plus noise.

### 1.1 SCM Definition

$$X_i := f_i(\text{Pa}(X_i), U_i)$$

```python
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# SCM: Z → X → Y, with confounder C → X, C → Y
np.random.seed(42)
n = 1000

Z = np.random.normal(0, 1, n)            # instrument
C = np.random.normal(0, 1, n)            # confounder
X = 0.5 * Z + 0.8 * C + np.random.normal(0, 0.5, n)  # treatment
Y = 1.5 * X - 0.5 * C + np.random.normal(0, 0.5, n)  # outcome

# Naive correlation is biased by the confounder
naive_effect = np.polyfit(X, Y, 1)[0]
print(f"Naive regression coefficient: {naive_effect:.3f}")
print(f"True causal effect: 1.5")
# The naive estimate is biased because C confounds X → Y
```

### 1.2 Causal Graph

```python
G = nx.DiGraph()
G.add_edges_from([("Z", "X"), ("C", "X"), ("C", "Y"), ("X", "Y")])

pos = {"Z": (0, 0), "C": (1, 1), "X": (1, 0), "Y": (2, 0)}
fig, ax = plt.subplots(figsize=(8, 5))
nx.draw(G, pos, with_labels=True, node_color="lightblue",
        node_size=2000, font_size=14, arrows=True, arrowsize=25, ax=ax)
ax.set_title("Causal DAG: Z→X→Y, C→X, C→Y")
plt.tight_layout()
plt.savefig("causal_dag.png", dpi=100)
plt.show()
```

---

## 2. The Do-Operator

The key distinction: $P(Y | X=x)$ (observational) vs $P(Y | do(X=x))$ (interventional).

### 2.1 Intervention vs Conditioning

```python
# Observing X=x: we see the world where X happened to be x
# do(X=x): we SET X to x, breaking all incoming arrows to X

# In our SCM:
# P(Y | X=2) includes the effect of C (confounded)
# P(Y | do(X=2)) removes C→X, isolating X→Y

# After do(X=x), the modified SCM is:
# X := x  (overwritten, no dependence on Z or C)
# Y := 1.5 * x - 0.5 * C + noise_Y

# So E[Y | do(X=x)] = 1.5*x - 0.5*E[C] = 1.5*x
```

---

## 3. Backdoor Criterion and Adjustment

### 3.1 Backdoor Criterion

A set Z satisfies the backdoor criterion relative to (X, Y) if:
1. No node in Z is a descendant of X
2. Z blocks every path between X and Y that contains an arrow into X

```python
def backdoor_adjustment(X, Y, C, n_bins=20):
    """Estimate causal effect E[Y|do(X)] via backdoor adjustment."""
    # E[Y|do(x)] = sum_c E[Y|X=x, C=c] * P(C=c)
    # For continuous variables, use regression: control for C

    from sklearn.linear_model import LinearRegression
    # Adjusted regression: Y ~ X + C
    reg = LinearRegression()
    features = np.column_stack([X, C])
    reg.fit(features, Y)

    print(f"Backdoor-adjusted effect of X on Y: {reg.coef_[0]:.3f}")
    print(f"True causal effect: 1.5")
    return reg.coef_[0]

adjusted_effect = backdoor_adjustment(X, Y, C)
```

### 3.2 Bayesian Backdoor Adjustment with PyMC

```python
import pymc as pm
import arviz as az

with pm.Model() as causal_model:
    # Priors
    beta_x = pm.Normal("beta_x", mu=0, sigma=5)  # causal effect
    beta_c = pm.Normal("beta_c", mu=0, sigma=5)
    intercept = pm.Normal("intercept", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=5)

    # Adjusted regression
    mu = intercept + beta_x * X + beta_c * C
    y_obs = pm.Normal("y", mu=mu, sigma=sigma, observed=Y)

    trace = pm.sample(3000, tune=1000, chains=4, random_seed=42)

summary = az.summary(trace, var_names=["beta_x", "beta_c"])
print(summary)
# beta_x should be close to 1.5 (true causal effect)
```

---

## 4. Frontdoor Criterion

When there is no valid backdoor adjustment set, the frontdoor criterion may still enable causal identification.

```python
# Frontdoor: X → M → Y, with unobserved confounder U → X, U → Y
# If M satisfies:
# 1. X blocks all paths from M to U
# 2. There is no backdoor path from X to M
# 3. All backdoor paths from M to Y are blocked by X

# Frontdoor adjustment:
# P(Y|do(X=x)) = sum_m P(M=m|X=x) * sum_{x'} P(Y|X=x',M=m) * P(X=x')

def frontdoor_adjustment(X, M, Y, n_samples=10000):
    """Estimate causal effect via frontdoor criterion."""
    from sklearn.linear_model import LinearRegression

    # Step 1: P(M|X)
    reg_mx = LinearRegression().fit(X.reshape(-1, 1), M)
    beta_mx = reg_mx.coef_[0]

    # Step 2: P(Y|X, M)
    reg_ymx = LinearRegression().fit(np.column_stack([M, X]), Y)
    beta_m_on_y = reg_ymx.coef_[0]

    # Causal effect = beta_mx * beta_m_on_y
    causal_effect = beta_mx * beta_m_on_y
    print(f"Frontdoor-adjusted causal effect: {causal_effect:.3f}")
    return causal_effect
```

---

## 5. Instrumental Variables

```python
def iv_estimation(Z, X, Y):
    """Two-stage least squares (2SLS) for IV estimation."""
    from sklearn.linear_model import LinearRegression

    # Stage 1: regress X on Z
    reg1 = LinearRegression().fit(Z.reshape(-1, 1), X)
    X_hat = reg1.predict(Z.reshape(-1, 1))

    # Stage 2: regress Y on X_hat
    reg2 = LinearRegression().fit(X_hat.reshape(-1, 1), Y)
    iv_effect = reg2.coef_[0]

    print(f"IV estimate (2SLS): {iv_effect:.3f}")
    print(f"True causal effect: 1.5")
    return iv_effect

iv_effect = iv_estimation(Z, X, Y)
```

### 5.1 Bayesian IV with PyMC

```python
with pm.Model() as iv_model:
    # Stage 1: X ~ Z
    gamma = pm.Normal("gamma", mu=0, sigma=5)
    sigma_x = pm.HalfNormal("sigma_x", sigma=5)
    x_pred = gamma * Z
    x_obs = pm.Normal("x_obs", mu=x_pred, sigma=sigma_x, observed=X)

    # Stage 2: Y ~ X_hat
    beta = pm.Normal("beta", mu=0, sigma=5)  # causal effect
    intercept = pm.Normal("intercept", mu=0, sigma=10)
    sigma_y = pm.HalfNormal("sigma_y", sigma=5)
    mu_y = intercept + beta * x_pred
    y_obs = pm.Normal("y_obs", mu=mu_y, sigma=sigma_y, observed=Y)

    trace_iv = pm.sample(3000, tune=1000, chains=4, random_seed=42)

print(az.summary(trace_iv, var_names=["beta", "gamma"]))
```

---

## 6. Regression Discontinuity and Difference-in-Differences

### 6.1 Regression Discontinuity

```python
# Bayesian RDD: effect of a treatment assigned at a threshold
np.random.seed(42)
n = 500
running_var = np.random.uniform(0, 10, n)
treated = (running_var >= 5).astype(float)
y_rdd = 2 + 0.5 * running_var + 3 * treated + np.random.normal(0, 1, n)

with pm.Model() as rdd_model:
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta_run = pm.Normal("beta_run", mu=0, sigma=5)
    tau = pm.Normal("tau", mu=0, sigma=5)  # treatment effect
    sigma = pm.HalfNormal("sigma", sigma=5)

    mu = alpha + beta_run * running_var + tau * treated
    y_obs = pm.Normal("y", mu=mu, sigma=sigma, observed=y_rdd)

    trace_rdd = pm.sample(3000, tune=1000, random_seed=42)

print(f"Estimated treatment effect (tau): "
      f"{trace_rdd.posterior['tau'].values.mean():.3f} (true: 3.0)")
```

---

## 7. Causal Discovery

```python
# PC Algorithm: learn causal structure from data
def pc_algorithm_sketch(data, alpha=0.05):
    """Simplified PC algorithm for causal discovery."""
    from scipy.stats import pearsonr
    from itertools import combinations

    variables = list(data.columns)
    n = len(variables)

    # Start with complete undirected graph
    edges = {(i, j) for i in variables for j in variables if i != j}

    # Remove edges based on unconditional independence
    to_remove = set()
    for i, j in combinations(variables, 2):
        r, p = pearsonr(data[i], data[j])
        if p > alpha:
            to_remove.add((i, j))
            to_remove.add((j, i))
    edges -= to_remove

    # Orient edges using v-structures (colliders)
    # ... (simplified)

    return edges

import pandas as pd
data_df = pd.DataFrame({"Z": Z, "C": C, "X": X, "Y": Y})
learned_edges = pc_algorithm_sketch(data_df)
print(f"Learned edges: {learned_edges}")
```

---

## 8. Average Treatment Effect (ATE) with Propensity Scores

```python
def propensity_score_matching(X_features, treatment, outcome, n_neighbors=5):
    """Estimate ATE using propensity score matching."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import NearestNeighbors

    # Estimate propensity scores
    lr = LogisticRegression()
    lr.fit(X_features, treatment)
    propensity = lr.predict_proba(X_features)[:, 1]

    # Match treated and control units by propensity score
    treated_idx = np.where(treatment == 1)[0]
    control_idx = np.where(treatment == 0)[0]

    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(propensity[control_idx].reshape(-1, 1))
    distances, indices = nn.kneighbors(propensity[treated_idx].reshape(-1, 1))

    # ATE estimate
    y_treated = outcome[treated_idx]
    y_matched_control = outcome[control_idx[indices]].mean(axis=1)
    ate = (y_treated - y_matched_control).mean()

    print(f"Estimated ATE (propensity matching): {ate:.3f}")
    return ate
```

---

## Summary

| Method | When to Use | Key Assumption |
|--------|-------------|----------------|
| Backdoor adjustment | Measured confounders | No unobserved confounders |
| Frontdoor criterion | Mediator available | Specific graph structure |
| Instrumental variables | Unobserved confounders | Valid instrument |
| RDD | Treatment at threshold | Continuity at cutoff |
| DiD | Before/after treatment | Parallel trends |
| Propensity score | Observational data | Unconfoundedness |

---

## References

1. Pearl, J. (2009). *Causality*, 2nd Edition. Cambridge University Press.
2. Imbens, G. & Rubin, D. (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences*. Cambridge.
3. Cunningham, S. (2021). *Causal Inference: The Mixtape*. Yale University Press.
4. Peters, J., Janzing, D., & Scholkopf, B. (2017). *Elements of Causal Inference*. MIT Press.

---

[Previous: Bayesian Deep Learning](./14_Bayesian_Deep_Learning.md) | [Next: Model Comparison →](./16_Model_Comparison.md)
