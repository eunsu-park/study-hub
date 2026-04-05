# 05. Hierarchical Models

[Previous: PyMC Introduction](./04_PyMC_Introduction.md) | [Next: Bayesian Regression](./06_Bayesian_Regression.md)

---

> **Framework Note**: This lesson uses PyMC 5.x for hierarchical model building.
>
> Installation: `pip install pymc arviz numpy matplotlib pandas`

## Learning Objectives

- Understand the motivation for multilevel / hierarchical models
- Distinguish between complete pooling, no pooling, and partial pooling
- Implement hierarchical models in PyMC
- Understand shrinkage estimation and its benefits
- Apply non-centered parameterization to improve sampling

---

## 1. The Pooling Problem

When data comes from multiple groups (schools, hospitals, regions), we face a fundamental modeling choice.

### 1.1 Three Approaches

```python
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd

# Example: batting averages for 18 baseball players
# (Efron & Morris, 1975 — classic shrinkage example)
np.random.seed(42)
n_players = 18
true_ability = np.random.beta(80, 220, size=n_players)  # true talent ~0.265
at_bats = np.random.randint(30, 100, size=n_players)
hits = np.random.binomial(at_bats, true_ability)
observed_avg = hits / at_bats

print("Player batting averages (first 6):")
for i in range(6):
    print(f"  Player {i}: {hits[i]}/{at_bats[i]} = {observed_avg[i]:.3f} (true: {true_ability[i]:.3f})")
```

### 1.2 Complete Pooling (One Parameter for All)

```python
with pm.Model() as pooled_model:
    theta = pm.Beta("theta", alpha=1, beta=1)
    y = pm.Binomial("y", n=at_bats, p=theta, observed=hits)
    trace_pooled = pm.sample(2000, tune=1000, random_seed=42)

pooled_mean = trace_pooled.posterior["theta"].values.mean()
print(f"Pooled estimate (same for all): {pooled_mean:.3f}")
# Problem: ignores player-level variation
```

### 1.3 No Pooling (Separate Parameter per Group)

```python
with pm.Model() as unpooled_model:
    theta = pm.Beta("theta", alpha=1, beta=1, shape=n_players)
    y = pm.Binomial("y", n=at_bats, p=theta, observed=hits)
    trace_unpooled = pm.sample(2000, tune=1000, random_seed=42)

unpooled_means = trace_unpooled.posterior["theta"].values.mean(axis=(0, 1))
print(f"Unpooled estimates (first 6): {unpooled_means[:6].round(3)}")
# Problem: extreme estimates for players with few at-bats
```

### 1.4 Partial Pooling (Hierarchical Model)

```python
with pm.Model() as hierarchical_model:
    # Hyperpriors: population-level parameters
    alpha_pop = pm.Gamma("alpha_pop", alpha=2, beta=0.1)
    beta_pop = pm.Gamma("beta_pop", alpha=2, beta=0.1)

    # Group-level parameters: each player drawn from the population
    theta = pm.Beta("theta", alpha=alpha_pop, beta=beta_pop, shape=n_players)

    # Likelihood
    y = pm.Binomial("y", n=at_bats, p=theta, observed=hits)

    # Sample
    trace_hier = pm.sample(3000, tune=1000, chains=4, random_seed=42)

hier_means = trace_hier.posterior["theta"].values.mean(axis=(0, 1))
print(f"Hierarchical estimates (first 6): {hier_means[:6].round(3)}")
```

---

## 2. Shrinkage

The hallmark of hierarchical models: group estimates are "shrunk" toward the population mean, with more shrinkage for groups with less data.

```python
def plot_shrinkage(observed, pooled, unpooled, hierarchical, true_vals, at_bats):
    """Visualize shrinkage from no-pooling to hierarchical estimates."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Sort by sample size
    order = np.argsort(at_bats)

    for idx, i in enumerate(order):
        ax.plot([observed[i], hierarchical[i]], [idx, idx], 'b-', alpha=0.5)
        ax.scatter(observed[i], idx, color='red', s=at_bats[i]*2, alpha=0.6, label="Observed" if idx==0 else "")
        ax.scatter(hierarchical[i], idx, color='blue', s=50, zorder=5, label="Hierarchical" if idx==0 else "")
        ax.scatter(true_vals[i], idx, color='green', marker='x', s=80, label="True" if idx==0 else "")

    ax.axvline(pooled, color='gray', linestyle='--', label="Pooled mean")
    ax.set_xlabel("Batting Average")
    ax.set_ylabel("Player (sorted by at-bats)")
    ax.set_title("Shrinkage: Observed → Hierarchical Estimates")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("shrinkage.png", dpi=100)
    plt.show()

plot_shrinkage(observed_avg, pooled_mean, unpooled_means, hier_means, true_ability, at_bats)
```

### 2.1 Measuring Shrinkage

```python
# Shrinkage factor for each player
shrinkage = 1 - (hier_means - pooled_mean) / (observed_avg - pooled_mean + 1e-10)
print("Shrinkage factors (0=no shrinkage, 1=complete pooling):")
for i in range(6):
    print(f"  Player {i} (n={at_bats[i]}): shrinkage = {shrinkage[i]:.3f}")
# Players with fewer at-bats get more shrinkage
```

---

## 3. The Eight Schools Problem

The classic hierarchical modeling example from Gelman et al. (2013).

```python
# SAT coaching effects at 8 schools
schools = ["A", "B", "C", "D", "E", "F", "G", "H"]
y_obs = np.array([28, 8, -3, 7, -1, 1, 18, 12])     # estimated treatment effects
sigma_obs = np.array([15, 10, 16, 11, 9, 11, 10, 18]) # standard errors

with pm.Model() as eight_schools:
    # Hyperpriors
    mu = pm.Normal("mu", mu=0, sigma=20)           # population mean effect
    tau = pm.HalfNormal("tau", sigma=20)            # between-school std

    # Non-centered parameterization (crucial for good sampling!)
    z = pm.Normal("z", mu=0, sigma=1, shape=8)     # standardized offsets
    theta = pm.Deterministic("theta", mu + tau * z) # school effects

    # Likelihood
    y = pm.Normal("y", mu=theta, sigma=sigma_obs, observed=y_obs)

    trace_schools = pm.sample(5000, tune=2000, chains=4,
                              target_accept=0.95, random_seed=42)

# Summary
summary = az.summary(trace_schools, var_names=["mu", "tau", "theta"])
print(summary)

# Check for divergences
divergences = trace_schools.sample_stats["diverging"].values.sum()
print(f"\nDivergent transitions: {divergences}")
```

### 3.1 Centered vs Non-Centered Parameterization

```python
# CENTERED (can cause divergences):
# theta_j ~ Normal(mu, tau)
#
# NON-CENTERED (usually better):
# z_j ~ Normal(0, 1)
# theta_j = mu + tau * z_j
#
# Both define the same model, but the non-centered version
# decouples theta from tau in the posterior geometry.

# Let's compare
with pm.Model() as centered_model:
    mu = pm.Normal("mu", mu=0, sigma=20)
    tau = pm.HalfNormal("tau", sigma=20)
    theta = pm.Normal("theta", mu=mu, sigma=tau, shape=8)  # centered
    y = pm.Normal("y", mu=theta, sigma=sigma_obs, observed=y_obs)
    trace_centered = pm.sample(3000, tune=1000, chains=4, random_seed=42)

div_centered = trace_centered.sample_stats["diverging"].values.sum()
div_noncentered = trace_schools.sample_stats["diverging"].values.sum()
print(f"Divergences — Centered: {div_centered}, Non-centered: {div_noncentered}")
```

---

## 4. Hierarchical Regression

### 4.1 Varying Intercepts Model

```python
# Example: student test scores across schools
np.random.seed(42)
n_schools = 10
n_students_per = 30
n_total = n_schools * n_students_per

school_ids = np.repeat(np.arange(n_schools), n_students_per)
true_school_effects = np.random.normal(0, 3, size=n_schools)
study_hours = np.random.uniform(1, 10, size=n_total)

scores = (50 + true_school_effects[school_ids] +
          5 * study_hours +
          np.random.normal(0, 5, size=n_total))

with pm.Model() as varying_intercepts:
    # Hyperpriors
    mu_school = pm.Normal("mu_school", mu=50, sigma=20)
    sigma_school = pm.HalfNormal("sigma_school", sigma=10)

    # School-level intercepts (non-centered)
    z_school = pm.Normal("z_school", mu=0, sigma=1, shape=n_schools)
    alpha = pm.Deterministic("alpha", mu_school + sigma_school * z_school)

    # Fixed slope for study hours
    beta = pm.Normal("beta", mu=0, sigma=10)

    # Observation noise
    sigma = pm.HalfNormal("sigma", sigma=10)

    # Linear model
    mu = alpha[school_ids] + beta * study_hours

    # Likelihood
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=scores)

    trace_vi = pm.sample(3000, tune=1000, chains=4, random_seed=42)

summary = az.summary(trace_vi, var_names=["mu_school", "sigma_school", "beta", "sigma"])
print(summary)
```

### 4.2 Varying Intercepts and Slopes

```python
with pm.Model() as varying_slopes:
    # Hyperpriors for intercepts
    mu_alpha = pm.Normal("mu_alpha", mu=50, sigma=20)
    sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=10)

    # Hyperpriors for slopes
    mu_beta = pm.Normal("mu_beta", mu=0, sigma=10)
    sigma_beta = pm.HalfNormal("sigma_beta", sigma=5)

    # School-level parameters (non-centered)
    z_alpha = pm.Normal("z_alpha", mu=0, sigma=1, shape=n_schools)
    z_beta = pm.Normal("z_beta", mu=0, sigma=1, shape=n_schools)
    alpha = pm.Deterministic("alpha", mu_alpha + sigma_alpha * z_alpha)
    beta = pm.Deterministic("beta", mu_beta + sigma_beta * z_beta)

    sigma = pm.HalfNormal("sigma", sigma=10)

    mu = alpha[school_ids] + beta[school_ids] * study_hours
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=scores)

    trace_vs = pm.sample(3000, tune=1000, chains=4,
                         target_accept=0.9, random_seed=42)

# Visualize school-specific slopes
fig, ax = plt.subplots(figsize=(10, 6))
hours_grid = np.linspace(1, 10, 50)
for school in range(n_schools):
    a = trace_vs.posterior["alpha"].values[:, :, school].mean()
    b = trace_vs.posterior["beta"].values[:, :, school].mean()
    ax.plot(hours_grid, a + b * hours_grid, alpha=0.6, label=f"School {school}")
ax.set_xlabel("Study Hours")
ax.set_ylabel("Predicted Score")
ax.set_title("Varying Slopes: Each School Has Its Own Relationship")
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("varying_slopes.png", dpi=100)
plt.show()
```

---

## 5. Correlated Group Parameters

When intercepts and slopes are correlated, we can model this with a multivariate normal hyperprior.

```python
with pm.Model() as correlated_model:
    # Hyperpriors
    mu_ab = pm.Normal("mu_ab", mu=[50, 5], sigma=10, shape=2)

    # Covariance structure (LKJ prior for correlation)
    sd_ab = pm.HalfNormal("sd_ab", sigma=10, shape=2)
    chol, corr, sigmas = pm.LKJCholeskyCov(
        "chol_cov", n=2, eta=2, sd_dist=pm.HalfNormal.dist(sigma=10)
    )
    cov = pm.Deterministic("cov", chol @ chol.T)

    # School parameters from multivariate normal
    z = pm.Normal("z", mu=0, sigma=1, shape=(n_schools, 2))
    ab = pm.Deterministic("ab", mu_ab + z @ chol.T)
    alpha_corr = ab[:, 0]
    beta_corr = ab[:, 1]

    sigma = pm.HalfNormal("sigma", sigma=10)
    mu = alpha_corr[school_ids] + beta_corr[school_ids] * study_hours
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=scores)

    trace_corr = pm.sample(3000, tune=2000, chains=4,
                           target_accept=0.95, random_seed=42)

# Extract correlation between intercepts and slopes
corr_samples = trace_corr.posterior["cov"].values
corr_01 = corr_samples[:, :, 0, 1] / np.sqrt(corr_samples[:, :, 0, 0] * corr_samples[:, :, 1, 1])
print(f"Posterior correlation(α, β): {corr_01.mean():.3f} [{np.percentile(corr_01, 2.5):.3f}, {np.percentile(corr_01, 97.5):.3f}]")
```

---

## 6. Hierarchical Models for Count Data

```python
# Example: modeling defect rates across manufacturing lines
np.random.seed(42)
n_lines = 8
n_batches_per = np.random.randint(20, 50, size=n_lines)
true_rates = np.random.gamma(3, 1, size=n_lines)

line_ids = np.concatenate([np.full(n, i) for i, n in enumerate(n_batches_per)])
defects = np.concatenate([
    np.random.poisson(true_rates[i], n)
    for i, n in enumerate(n_batches_per)
])

with pm.Model() as hier_poisson:
    # Hyperpriors for Gamma distribution of rates
    alpha_hyper = pm.Exponential("alpha_hyper", lam=0.5)
    beta_hyper = pm.Exponential("beta_hyper", lam=0.5)

    # Line-level rates
    rate = pm.Gamma("rate", alpha=alpha_hyper, beta=beta_hyper, shape=n_lines)

    # Likelihood
    y = pm.Poisson("y", mu=rate[line_ids], observed=defects)

    trace_hp = pm.sample(3000, tune=1000, chains=4, random_seed=42)

summary = az.summary(trace_hp, var_names=["rate"])
print(summary)
```

---

## 7. Model Comparison: Pooled vs Hierarchical

```python
# Compare models using WAIC or LOO-CV
with pooled_model:
    pm.compute_log_likelihood(trace_pooled)
with unpooled_model:
    pm.compute_log_likelihood(trace_unpooled)
with hierarchical_model:
    pm.compute_log_likelihood(trace_hier)

comparison = az.compare({
    "pooled": trace_pooled,
    "unpooled": trace_unpooled,
    "hierarchical": trace_hier,
}, ic="loo")
print(comparison)

az.plot_compare(comparison)
plt.title("Model Comparison: LOO-CV")
plt.savefig("model_comparison.png", dpi=100)
plt.show()
```

---

## 8. When to Use Hierarchical Models

### 8.1 Indicators

1. **Grouped data**: observations belong to natural clusters
2. **Small group sizes**: some groups have very few observations
3. **Interest in both group and population**: want estimates for individual groups AND overall trends
4. **Prior information**: groups are exchangeable (drawn from a common population)

### 8.2 Hierarchical Models in Practice

| Domain | Groups | Outcome |
|--------|--------|---------|
| Education | Schools | Test scores |
| Medicine | Hospitals | Mortality rates |
| Sports | Players | Performance metrics |
| Marketing | Regions | Conversion rates |
| Manufacturing | Machines | Defect rates |
| Ecology | Species | Abundance counts |

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| Complete pooling | One parameter for all; ignores group variation |
| No pooling | Separate parameters; noisy for small groups |
| Partial pooling | Hierarchical model; automatic shrinkage toward population |
| Shrinkage | Small-sample groups shrink more toward the grand mean |
| Non-centered param. | `theta = mu + sigma * z` avoids divergences |
| LKJ prior | For modeling correlations between group-level parameters |
| When to use | Grouped data with varying sample sizes |

---

## References

1. Gelman, A. & Hill, J. (2006). *Data Analysis Using Regression and Multilevel/Hierarchical Models*. Cambridge.
2. Gelman, A., et al. (2013). *Bayesian Data Analysis*, 3rd Ed., Ch. 5.
3. Betancourt, M. & Girolami, M. (2015). "Hamiltonian Monte Carlo for Hierarchical Models." arXiv:1312.0906.
4. McElreath, R. (2020). *Statistical Rethinking*, 2nd Ed., Ch. 13.

---

[Previous: PyMC Introduction](./04_PyMC_Introduction.md) | [Next: Bayesian Regression →](./06_Bayesian_Regression.md)
