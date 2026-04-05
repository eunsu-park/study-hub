# 04. PyMC Introduction

[Previous: MCMC Fundamentals](./03_MCMC_Fundamentals.md) | [Next: Hierarchical Models](./05_Hierarchical_Models.md)

---

> **Framework Note**: This lesson uses PyMC 5.x with the PyTensor backend and ArviZ for diagnostics.
>
> Installation: `pip install pymc arviz matplotlib`

## Learning Objectives

- Build probabilistic models using PyMC's declarative API
- Run MCMC sampling and understand sampler configuration
- Analyze posterior traces with ArviZ
- Perform posterior predictive checks
- Use PyMC for real-world inference problems

---

## 1. PyMC Overview

PyMC is the most popular Python framework for Bayesian modeling. It provides a high-level API for specifying probabilistic models and runs NUTS (No-U-Turn Sampler) by default.

### 1.1 Core Concepts

```python
import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt

print(f"PyMC version: {pm.__version__}")

# A PyMC model has three ingredients:
# 1. Priors: pm.Normal("mu", mu=0, sigma=10)
# 2. Likelihood: pm.Normal("y", mu=mu, sigma=sigma, observed=data)
# 3. Inference: pm.sample()
```

### 1.2 Model Context Manager

```python
# Every PyMC model is defined inside a context manager
with pm.Model() as simple_model:
    # Prior
    mu = pm.Normal("mu", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=5)

    # Likelihood
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=np.random.randn(100))

# Inspect the model
print(simple_model.basic_RVs)    # [mu, sigma]
print(simple_model.observed_RVs) # [y]
```

---

## 2. Building Your First Model

### 2.1 Coin Flip Model (Beta-Binomial)

```python
# Data: 14 heads out of 20 flips
n_flips = 20
n_heads = 14

with pm.Model() as coin_model:
    # Prior: Beta(1, 1) = Uniform(0, 1)
    theta = pm.Beta("theta", alpha=1, beta=1)

    # Likelihood: Binomial
    y = pm.Binomial("y", n=n_flips, p=theta, observed=n_heads)

    # Sample from posterior
    trace = pm.sample(2000, tune=1000, chains=4, random_seed=42)

# Posterior summary
summary = az.summary(trace, var_names=["theta"])
print(summary)

# Plot posterior
az.plot_posterior(trace, var_names=["theta"],
                  ref_val=0.5,  # reference line at fair coin
                  hdi_prob=0.95)
plt.title("Posterior: P(heads)")
plt.savefig("pymc_coin_posterior.png", dpi=100)
plt.show()
```

### 2.2 Normal Model (Estimating Mean and Variance)

```python
# Generate data
np.random.seed(42)
data = np.random.normal(loc=5.0, scale=2.0, size=100)

with pm.Model() as normal_model:
    # Priors
    mu = pm.Normal("mu", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=5)

    # Likelihood
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=data)

    # Sample
    trace = pm.sample(3000, tune=1000, chains=4, random_seed=42)

# Summary with diagnostics
summary = az.summary(trace, var_names=["mu", "sigma"],
                     stat_funcs={"median": np.median},
                     hdi_prob=0.95)
print(summary)
# Check: R-hat should be ~1.00, ESS should be > 400
```

---

## 3. Sampling and Configuration

### 3.1 Sampler Options

```python
with pm.Model() as model:
    mu = pm.Normal("mu", mu=0, sigma=10)
    y = pm.Normal("y", mu=mu, sigma=1, observed=data)

    # Default: NUTS sampler (best for continuous parameters)
    trace_nuts = pm.sample(
        draws=2000,           # number of posterior samples per chain
        tune=1000,            # warm-up / adaptation steps (discarded)
        chains=4,             # number of independent chains
        cores=4,              # parallel cores (1 per chain)
        target_accept=0.8,    # target acceptance rate (increase to 0.95 for difficult models)
        random_seed=42,
        return_inferencedata=True,
    )

    # For models with discrete parameters: use Metropolis
    # step = pm.Metropolis()
    # trace_mh = pm.sample(5000, step=step, chains=4)
```

### 3.2 Understanding the Trace Object

```python
# The trace (InferenceData) contains:
print(trace_nuts)
# - posterior: sampled parameter values
# - sample_stats: NUTS diagnostics (divergences, tree depth, etc.)
# - observed_data: the data we conditioned on

# Access posterior samples
mu_samples = trace_nuts.posterior["mu"].values  # shape: (chains, draws)
print(f"Shape: {mu_samples.shape}")  # (4, 2000)
print(f"Posterior mean of mu: {mu_samples.mean():.3f}")
print(f"Posterior std of mu:  {mu_samples.std():.3f}")

# Flatten across chains
mu_flat = mu_samples.flatten()
print(f"Total samples: {len(mu_flat)}")
```

### 3.3 Checking for Divergences

```python
# Divergent transitions indicate problems
divergences = trace_nuts.sample_stats["diverging"].values.sum()
print(f"Number of divergent transitions: {divergences}")

# If divergences > 0:
# 1. Increase target_accept (e.g., 0.95 or 0.99)
# 2. Reparameterize the model (non-centered parameterization)
# 3. Check for conflicting priors
```

---

## 4. ArviZ Diagnostics and Visualization

### 4.1 Trace Plots

```python
# Trace plot: visual convergence check
az.plot_trace(trace, var_names=["mu", "sigma"])
plt.tight_layout()
plt.savefig("pymc_trace.png", dpi=100)
plt.show()
# Look for: "fuzzy caterpillars" with all chains overlapping
```

### 4.2 Forest Plot

```python
# Forest plot: compare parameter estimates across chains
az.plot_forest(trace, var_names=["mu", "sigma"],
               combined=True, hdi_prob=0.95)
plt.title("Forest Plot: 95% HDI")
plt.savefig("pymc_forest.png", dpi=100)
plt.show()
```

### 4.3 Pair Plot

```python
# Pair plot: visualize joint posterior and correlations
az.plot_pair(trace, var_names=["mu", "sigma"],
             kind="kde", marginals=True)
plt.savefig("pymc_pair.png", dpi=100)
plt.show()
```

### 4.4 Energy Plot

```python
# Energy plot: diagnose HMC sampling efficiency
az.plot_energy(trace)
plt.title("Energy Plot (overlapping is good)")
plt.savefig("pymc_energy.png", dpi=100)
plt.show()
```

---

## 5. Posterior Predictive Checks

The gold standard for model validation: can the fitted model generate data that looks like the observed data?

### 5.1 Sampling from the Posterior Predictive

```python
with normal_model:
    # Generate predictions from the posterior
    ppc = pm.sample_posterior_predictive(trace, random_seed=42)

# Compare observed data with posterior predictions
az.plot_ppc(az.from_pymc(
    trace=trace,
    posterior_predictive=ppc,
    model=normal_model,
))
plt.title("Posterior Predictive Check")
plt.savefig("pymc_ppc.png", dpi=100)
plt.show()
```

### 5.2 Custom Posterior Predictive Checks

```python
# Test statistic: does the model capture the data's skewness?
from scipy.stats import skew

observed_skew = skew(data)
ppc_skews = [skew(ppc.posterior_predictive["y"].values[0, i, :])
             for i in range(ppc.posterior_predictive["y"].shape[1])]

p_value = np.mean(np.array(ppc_skews) >= observed_skew)
print(f"Observed skewness: {observed_skew:.3f}")
print(f"Posterior predictive p-value for skewness: {p_value:.3f}")
# p close to 0 or 1 indicates model misfit
```

---

## 6. Common Distributions in PyMC

### 6.1 Continuous Distributions

```python
with pm.Model() as dist_demo:
    # Location-scale
    normal = pm.Normal("normal", mu=0, sigma=1)
    student_t = pm.StudentT("student_t", nu=3, mu=0, sigma=1)
    cauchy = pm.Cauchy("cauchy", alpha=0, beta=1)
    laplace = pm.Laplace("laplace", mu=0, b=1)

    # Positive
    halfnormal = pm.HalfNormal("halfnormal", sigma=1)
    exponential = pm.Exponential("exponential", lam=1)
    gamma = pm.Gamma("gamma", alpha=2, beta=1)
    inv_gamma = pm.InverseGamma("inv_gamma", alpha=2, beta=1)
    lognormal = pm.LogNormal("lognormal", mu=0, sigma=1)

    # Bounded
    beta = pm.Beta("beta", alpha=2, beta=5)
    uniform = pm.Uniform("uniform", lower=0, upper=10)

    # Multivariate
    mvnormal = pm.MvNormal("mvnormal", mu=np.zeros(2),
                            cov=np.eye(2), shape=2)
```

### 6.2 Discrete Distributions

```python
with pm.Model() as discrete_demo:
    bernoulli = pm.Bernoulli("bernoulli", p=0.5)
    binomial = pm.Binomial("binomial", n=10, p=0.3)
    poisson = pm.Poisson("poisson", mu=5)
    categorical = pm.Categorical("categorical", p=[0.3, 0.5, 0.2])
    neg_binomial = pm.NegativeBinomial("neg_binomial", mu=5, alpha=2)
```

---

## 7. Practical Example: A/B Test Analysis

```python
# E-commerce A/B test: conversion rates
np.random.seed(42)
n_a, n_b = 1000, 1000
conversions_a = 120  # 12% conversion
conversions_b = 145  # 14.5% conversion

with pm.Model() as ab_model:
    # Priors: weakly informative Beta
    p_a = pm.Beta("p_a", alpha=1, beta=1)
    p_b = pm.Beta("p_b", alpha=1, beta=1)

    # Derived quantities
    delta = pm.Deterministic("delta", p_b - p_a)
    relative_lift = pm.Deterministic("relative_lift", (p_b - p_a) / p_a)

    # Likelihood
    obs_a = pm.Binomial("obs_a", n=n_a, p=p_a, observed=conversions_a)
    obs_b = pm.Binomial("obs_b", n=n_b, p=p_b, observed=conversions_b)

    # Sample
    trace_ab = pm.sample(5000, tune=1000, chains=4, random_seed=42)

# Results
summary = az.summary(trace_ab, var_names=["p_a", "p_b", "delta", "relative_lift"])
print(summary)

# Probability that B is better than A
delta_samples = trace_ab.posterior["delta"].values.flatten()
prob_b_wins = (delta_samples > 0).mean()
print(f"\nP(B > A) = {prob_b_wins:.4f}")
print(f"Expected lift: {delta_samples.mean()*100:.2f}%")
print(f"95% HDI for lift: [{np.percentile(delta_samples, 2.5)*100:.2f}%, "
      f"{np.percentile(delta_samples, 97.5)*100:.2f}%]")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
az.plot_posterior(trace_ab, var_names=["delta"], ref_val=0, ax=axes[0])
axes[0].set_title("Δ = p_B - p_A")
az.plot_posterior(trace_ab, var_names=["relative_lift"], ref_val=0, ax=axes[1])
axes[1].set_title("Relative Lift: (p_B - p_A) / p_A")
plt.tight_layout()
plt.savefig("pymc_ab_test.png", dpi=100)
plt.show()
```

---

## 8. Practical Example: Poisson Regression

```python
# Count data: modeling website traffic
np.random.seed(42)
n_days = 100
day = np.arange(n_days)
is_weekend = ((day % 7 == 5) | (day % 7 == 6)).astype(float)
true_rate = np.exp(4.0 + 0.01 * day + 0.5 * is_weekend)
visits = np.random.poisson(true_rate)

with pm.Model() as poisson_model:
    # Priors
    intercept = pm.Normal("intercept", mu=0, sigma=5)
    beta_trend = pm.Normal("beta_trend", mu=0, sigma=1)
    beta_weekend = pm.Normal("beta_weekend", mu=0, sigma=2)

    # Linear predictor (log link)
    log_rate = intercept + beta_trend * day + beta_weekend * is_weekend

    # Likelihood
    y = pm.Poisson("y", mu=pm.math.exp(log_rate), observed=visits)

    # Sample
    trace_poisson = pm.sample(3000, tune=1000, chains=4, random_seed=42)

summary = az.summary(trace_poisson,
                     var_names=["intercept", "beta_trend", "beta_weekend"])
print(summary)
print(f"\nTrue values: intercept=4.0, beta_trend=0.01, beta_weekend=0.5")
```

---

## 9. Prior Predictive Simulation

```python
with pm.Model() as prior_check_model:
    mu = pm.Normal("mu", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=5)
    y = pm.Normal("y", mu=mu, sigma=sigma, shape=100)

    # Sample from priors only (no data conditioning)
    prior_samples = pm.sample_prior_predictive(1000, random_seed=42)

# Inspect prior predictive range
y_prior = prior_samples.prior_predictive["y"].values[0]  # (draws, obs)
print(f"Prior predictive y range: [{y_prior.min():.1f}, {y_prior.max():.1f}]")
print(f"Prior predictive y mean range: [{y_prior.mean(axis=1).min():.1f}, "
      f"{y_prior.mean(axis=1).max():.1f}]")

# If these ranges are unreasonable for your domain, tighten priors!
```

---

## 10. Model Persistence and Reproducibility

### 10.1 Saving and Loading Traces

```python
# Save trace to NetCDF
trace.to_netcdf("normal_model_trace.nc")

# Load trace
loaded_trace = az.from_netcdf("normal_model_trace.nc")
print(az.summary(loaded_trace, var_names=["mu", "sigma"]))
```

### 10.2 Reproducibility Tips

```python
# 1. Always set random_seed in pm.sample()
# 2. Record PyMC and dependency versions
# 3. Save both the model code and the trace
# 4. Use pm.model_to_graphviz() for documentation

# Model graph visualization
graph = pm.model_to_graphviz(normal_model)
graph.render("normal_model_graph", format="png")
```

---

## Summary

| PyMC Component | Purpose |
|---------------|---------|
| `pm.Model()` | Context manager for model definition |
| `pm.Normal()`, `pm.Beta()`, etc. | Prior and likelihood distributions |
| `pm.Deterministic()` | Derived quantities tracked in the trace |
| `pm.sample()` | Run MCMC (default: NUTS) |
| `pm.sample_posterior_predictive()` | Generate predictions from fitted model |
| `pm.sample_prior_predictive()` | Check prior implications |
| `az.summary()` | Posterior summaries with R-hat, ESS |
| `az.plot_trace()` | Visual convergence check |
| `az.plot_ppc()` | Posterior predictive check |

---

## References

1. Salvatier, J., Wiecki, T. V., & Fonnesbeck, C. (2016). "Probabilistic programming in Python using PyMC3." *PeerJ Computer Science*, 2, e55.
2. PyMC documentation: https://www.pymc.io/
3. ArviZ documentation: https://arviz-devs.github.io/arviz/
4. Kumar, R., et al. (2019). "ArviZ: a unified library for exploratory analysis of Bayesian models." *JOSS*, 4(33), 1143.

---

[Previous: MCMC Fundamentals](./03_MCMC_Fundamentals.md) | [Next: Hierarchical Models →](./05_Hierarchical_Models.md)
