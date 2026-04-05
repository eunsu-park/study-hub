# 16. Model Comparison

[Previous: Causal Inference](./15_Causal_Inference.md) | [Next: Uncertainty Quantification](./17_Uncertainty_Quantification.md)

---

> **Framework Note**: This lesson uses PyMC and ArviZ for Bayesian model comparison.
>
> Installation: `pip install pymc arviz numpy scipy matplotlib`

## Learning Objectives

- Understand information criteria (WAIC, LOO-CV) for model comparison
- Compute and interpret Bayes factors
- Perform posterior predictive checks for model validation
- Use ArviZ's model comparison tools
- Avoid common pitfalls in Bayesian model selection

---

## 1. Why Model Comparison Matters

In Bayesian modeling, we often have multiple candidate models. We need principled methods to assess which model best explains the data while avoiding overfitting.

### 1.1 The Bayesian Model Selection Framework

```python
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from scipy import stats

# Generate data from a quadratic relationship
np.random.seed(42)
n = 50
x = np.random.uniform(-3, 3, n)
y = 1.0 + 0.5 * x + 0.3 * x**2 + np.random.normal(0, 0.5, n)

# Candidate models with increasing complexity
models = {}
traces = {}

for degree in [1, 2, 3, 5]:
    X = np.column_stack([x**d for d in range(1, degree + 1)])
    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=5)
        beta = pm.Normal("beta", mu=0, sigma=2, shape=degree)
        sigma = pm.HalfNormal("sigma", sigma=3)
        mu = alpha + pm.math.dot(X, beta)
        y_obs = pm.Normal("y", mu=mu, sigma=sigma, observed=y)
        trace = pm.sample(3000, tune=1000, chains=4, random_seed=42)
        pm.compute_log_likelihood(trace)

    models[f"poly_{degree}"] = model
    traces[f"poly_{degree}"] = trace
```

---

## 2. Information Criteria

### 2.1 WAIC (Widely Applicable Information Criterion)

WAIC estimates out-of-sample predictive accuracy using the log pointwise predictive density (lppd) with a penalty for effective number of parameters.

$$\text{WAIC} = -2(\text{lppd} - p_\text{WAIC})$$

```python
# Compute WAIC for all models
for name, trace in traces.items():
    waic = az.waic(trace)
    print(f"{name}: WAIC = {waic.waic:.2f}, p_waic = {waic.p_waic:.2f}")
```

### 2.2 LOO-CV (Leave-One-Out Cross-Validation)

LOO-CV approximated via Pareto-Smoothed Importance Sampling (PSIS-LOO).

```python
# LOO-CV comparison
comparison = az.compare(traces, ic="loo")
print(comparison)

# Visualize
az.plot_compare(comparison)
plt.title("Model Comparison: LOO-CV")
plt.savefig("model_comparison_loo.png", dpi=100)
plt.show()
```

### 2.3 Interpreting Results

```python
# Key columns in the comparison table:
# - rank: model ranking (0 = best)
# - elpd_loo: expected log pointwise predictive density
# - p_loo: effective number of parameters
# - d_loo: difference from best model
# - se: standard error of the difference
# - dse: standard error of the difference relative to best
# - warning: True if Pareto k > 0.7 (unreliable estimate)

# Rule of thumb:
# - If d_loo / dse > 2: significant difference
# - If warning=True: use K-fold CV instead
```

---

## 3. Bayes Factors

The Bayes factor directly compares the evidence for two models.

$$BF_{12} = \frac{P(D | M_1)}{P(D | M_2)} = \frac{\int P(D|\theta_1, M_1) P(\theta_1 | M_1) d\theta_1}{\int P(D|\theta_2, M_2) P(\theta_2 | M_2) d\theta_2}$$

### 3.1 Interpretation Scale

| Bayes Factor | Evidence |
|-------------|---------|
| 1-3 | Barely worth mentioning |
| 3-10 | Substantial |
| 10-30 | Strong |
| 30-100 | Very strong |
| > 100 | Decisive |

### 3.2 Computing Bayes Factors (Savage-Dickey Ratio)

```python
def savage_dickey_bf(trace, param_name, null_value, prior_density_at_null):
    """
    Compute Bayes factor for point null hypothesis using Savage-Dickey ratio.
    BF_01 = p(theta=null | data) / p(theta=null | prior)
    """
    from scipy.stats import gaussian_kde

    posterior_samples = trace.posterior[param_name].values.flatten()
    kde = gaussian_kde(posterior_samples)
    posterior_density_at_null = kde(null_value)[0]

    bf_01 = posterior_density_at_null / prior_density_at_null
    bf_10 = 1 / bf_01

    print(f"BF_01 (in favor of null): {bf_01:.4f}")
    print(f"BF_10 (against null):     {bf_10:.4f}")
    return bf_01, bf_10

# Test: is the quadratic coefficient zero?
prior_density = stats.norm.pdf(0, 0, 2)  # Normal(0, 2) prior at 0
bf_01, bf_10 = savage_dickey_bf(traces["poly_2"], "beta", 0, prior_density)
```

### 3.3 Bridge Sampling

```python
def bridge_sampling_estimate(trace, model, n_bridge=10000):
    """Estimate marginal likelihood via bridge sampling (simplified)."""
    # This is a simplified version; use bridgesampling package for production
    log_lik = trace.log_likelihood
    # Approximate: harmonic mean estimator (known to be unstable)
    ll_values = sum(log_lik[v].values.flatten() for v in log_lik.data_vars)
    log_ml = -np.log(np.mean(np.exp(-ll_values)))
    return log_ml
```

---

## 4. Posterior Predictive Checks (PPC)

### 4.1 Visual PPC

```python
# Generate posterior predictive samples for the quadratic model
with models["poly_2"]:
    ppc = pm.sample_posterior_predictive(traces["poly_2"], random_seed=42)

# Overlay plot
az.plot_ppc(az.from_pymc(trace=traces["poly_2"],
                          posterior_predictive=ppc,
                          model=models["poly_2"]))
plt.title("Posterior Predictive Check: Quadratic Model")
plt.savefig("ppc_quadratic.png", dpi=100)
plt.show()
```

### 4.2 Test Statistics

```python
def ppc_test_statistics(y_obs, y_rep, statistics=None):
    """Compute posterior predictive p-values for test statistics."""
    if statistics is None:
        statistics = {
            "mean": np.mean,
            "std": np.std,
            "skewness": lambda x: stats.skew(x),
            "min": np.min,
            "max": np.max,
        }

    print("Posterior Predictive p-values:")
    print("-" * 50)
    for name, stat_fn in statistics.items():
        observed_stat = stat_fn(y_obs)
        rep_stats = np.array([stat_fn(y_rep[i]) for i in range(len(y_rep))])
        p_value = np.mean(rep_stats >= observed_stat)
        print(f"  {name:12s}: observed={observed_stat:.3f}, p-value={p_value:.3f}")
        # p close to 0 or 1 indicates model misfit for this statistic

y_rep = ppc.posterior_predictive["y"].values.reshape(-1, n)
ppc_test_statistics(y, y_rep[:1000])
```

### 4.3 Calibration Check

```python
def pit_calibration(y_obs, y_rep):
    """Probability Integral Transform calibration check."""
    n_obs = len(y_obs)
    pit_values = np.array([
        np.mean(y_rep[:, i] <= y_obs[i])
        for i in range(n_obs)
    ])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.hist(pit_values, bins=20, density=True, alpha=0.7)
    ax1.axhline(1.0, color='r', linestyle='--', label='Ideal (Uniform)')
    ax1.set_title("PIT Histogram (should be uniform)")
    ax1.legend()

    # ECDF vs uniform
    pit_sorted = np.sort(pit_values)
    ax2.plot(pit_sorted, np.linspace(0, 1, len(pit_sorted)), 'b-', label='PIT ECDF')
    ax2.plot([0, 1], [0, 1], 'r--', label='Ideal')
    ax2.set_title("PIT ECDF vs Uniform")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("pit_calibration.png", dpi=100)
    plt.show()
    return pit_values

pit = pit_calibration(y, y_rep[:2000])
```

---

## 5. Stacking and Model Averaging

Instead of selecting one model, combine multiple models weighted by their predictive performance.

```python
# Bayesian stacking weights (ArviZ)
weights = az.compare(traces, ic="loo")["weight"].values
model_names = list(traces.keys())

print("Stacking weights:")
for name, w in zip(model_names, weights):
    print(f"  {name}: {w:.3f}")

# Model-averaged predictions
def model_averaged_prediction(traces, weights, X_new):
    """Combine predictions from multiple models."""
    predictions = []
    for name, trace in traces.items():
        alpha = trace.posterior["alpha"].values.flatten()
        beta = trace.posterior["beta"].values.reshape(-1, trace.posterior["beta"].shape[-1])
        degree = beta.shape[1]
        X_poly = np.column_stack([X_new**d for d in range(1, degree + 1)])
        pred = alpha[:, None] + beta @ X_poly.T
        predictions.append(pred)

    # Weighted average
    avg_pred = sum(w * pred for w, pred in zip(weights, predictions))
    return avg_pred.mean(axis=0), avg_pred.std(axis=0)
```

---

## 6. Cross-Validation

### 6.1 K-Fold CV in Bayesian Setting

```python
def bayesian_kfold_cv(X, y, model_fn, K=5, n_samples=2000):
    """K-fold cross-validation for Bayesian models."""
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=K, shuffle=True, random_state=42)
    log_scores = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        with model_fn(X_train, y_train) as model:
            trace = pm.sample(n_samples, tune=500, chains=2, random_seed=42)

        # Compute predictive log-likelihood on test set
        alpha = trace.posterior["alpha"].values.flatten()
        beta = trace.posterior["beta"].values.reshape(-1, X.shape[1])
        sigma = trace.posterior["sigma"].values.flatten()

        pred_mu = alpha[:, None] + beta @ X_test.T
        log_lik = stats.norm.logpdf(y_test[None, :], pred_mu, sigma[:, None])
        log_score = np.log(np.exp(log_lik).mean(axis=0)).sum()
        log_scores.append(log_score)

        print(f"Fold {fold}: log score = {log_score:.2f}")

    print(f"\nMean log score: {np.mean(log_scores):.2f} ± {np.std(log_scores):.2f}")
    return log_scores
```

---

## 7. Common Pitfalls

### 7.1 Prior Sensitivity of Bayes Factors

```python
# Bayes factors are VERY sensitive to prior choice
# (unlike posterior inference, which is often robust)

# Example: same data, different priors → different Bayes factors
priors = [
    ("Narrow: N(0, 1)", 1.0),
    ("Medium: N(0, 5)", 5.0),
    ("Wide: N(0, 100)", 100.0),
]

for name, prior_sd in priors:
    prior_density = stats.norm.pdf(0, 0, prior_sd)
    bf_01, _ = savage_dickey_bf(traces["poly_2"], "beta", 0, prior_density)
    print(f"  {name}: BF_01 = {bf_01:.4f}")
```

### 7.2 Overfitting Warning Signs

```python
# Warning signs in model comparison:
# 1. p_loo or p_waic >> actual number of parameters
# 2. Large Pareto k values (> 0.7)
# 3. Wide standard errors on elpd differences
# 4. Posterior predictive checks fail
```

---

## Summary

| Criterion | Pros | Cons | Use When |
|-----------|------|------|----------|
| WAIC | Fast, automatic | Less reliable than LOO | Quick comparison |
| PSIS-LOO | Gold standard, reliable | Can fail (high Pareto k) | Default choice |
| Bayes factor | Direct evidence comparison | Very prior-sensitive | Sharp hypotheses |
| K-fold CV | No PSIS issues | Expensive (refit K times) | LOO fails |
| PPC | Intuitive, visual | Subjective | Model checking |
| Stacking | Combines model strengths | More complex | Multiple good models |

---

## References

1. Vehtari, A., Gelman, A., & Gabry, J. (2017). "Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC." *Statistics and Computing*, 27, 1413-1432.
2. Gelman, A., et al. (2013). *Bayesian Data Analysis*, 3rd Ed., Ch. 7.
3. Yao, Y., et al. (2018). "Using Stacking to Average Bayesian Predictive Distributions." *Bayesian Analysis*.
4. Gronau, Q., et al. (2017). "A Tutorial on Bridge Sampling." *Journal of Mathematical Psychology*.

---

[Previous: Causal Inference](./15_Causal_Inference.md) | [Next: Uncertainty Quantification →](./17_Uncertainty_Quantification.md)
