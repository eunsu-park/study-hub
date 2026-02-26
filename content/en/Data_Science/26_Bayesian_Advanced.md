# Bayesian Advanced Methods

[Previous: Practical Projects](./25_Practical_Projects.md) | [Next: Causal Inference](./27_Causal_Inference.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain how Hamiltonian Monte Carlo (HMC) uses gradient information to sample more efficiently than Metropolis-Hastings
2. Describe how the NUTS sampler automatically tunes the trajectory length and why it is the default in modern probabilistic programming frameworks
3. Compare MCMC and Variational Inference (ADVI) in terms of accuracy, speed, and appropriate use cases
4. Implement a hierarchical (multilevel) model in PyMC and explain the concept of partial pooling and shrinkage
5. Apply Bayesian model comparison using LOO-CV and WAIC, and interpret Pareto k diagnostics
6. Evaluate MCMC convergence using R-hat, effective sample size (ESS), divergence checks, and posterior predictive checks

---

Basic MCMC methods like Metropolis-Hastings get the job done for simple models, but they struggle with high-dimensional parameter spaces and scale poorly to large datasets. Advanced computational Bayesian methods -- HMC/NUTS for efficient sampling, Variational Inference for speed, and hierarchical models for grouped data -- unlock the full power of the Bayesian framework, letting you build richer models and draw reliable conclusions even when the problem is complex.

---

## 1. Beyond Basic MCMC

### 1.1 Limitations of Metropolis-Hastings

```python
"""
Metropolis-Hastings (from L16) works but has limitations:

1. Random walk proposal → slow exploration of high-dimensional spaces
2. Requires tuning step size manually
3. High autocorrelation → need many samples for good estimates
4. Scales poorly: O(d²) samples needed for d dimensions

Solution: Use gradient information to guide proposals
→ Hamiltonian Monte Carlo (HMC)
"""
```

### 1.2 Hamiltonian Monte Carlo (HMC)

```python
"""
HMC Intuition:

Imagine a ball rolling on a surface shaped like the negative log-posterior.
The ball's position = parameter values.
The ball's momentum = auxiliary variable for exploration.

Algorithm:
1. Sample random momentum p ~ N(0, I)
2. Simulate Hamiltonian dynamics for L steps with step size ε:
   - θ_new = θ + ε * p
   - p_new = p - ε * ∇U(θ)    where U(θ) = -log π(θ|data)
3. Accept/reject with Metropolis correction

Key Parameters:
- Step size (ε): Too large → rejected, too small → slow
- Number of steps (L): Too few → random walk, too many → wasted computation

Advantages over Metropolis-Hastings:
- Uses gradient → moves efficiently through parameter space
- Much lower autocorrelation
- Scales better with dimensionality: O(d^{5/4}) vs O(d²)
"""
```

### 1.3 NUTS (No-U-Turn Sampler)

```python
"""
NUTS: Automatically tunes L (number of leapfrog steps).

Problem with HMC: Choosing L is difficult.
- Too few steps → behaves like random walk
- Too many → the trajectory loops back (U-turn), wasting computation

NUTS solution:
- Build a binary tree of leapfrog steps
- Stop when the trajectory starts to turn back (U-turn criterion)
- Select a point from the trajectory with correct stationary distribution

NUTS is the default sampler in PyMC, Stan, and NumPyro.
No need to tune L — only ε (step size), which is auto-tuned during warmup.
"""

import pymc as pm
import numpy as np
import arviz as az

# Generate data
np.random.seed(42)
N = 200
true_mu = [3.0, -1.0]
true_sigma = [1.5, 0.8]
data = np.concatenate([
    np.random.normal(true_mu[0], true_sigma[0], N),
    np.random.normal(true_mu[1], true_sigma[1], N),
])

# PyMC model with NUTS (default sampler)
with pm.Model() as mixture_model:
    # Priors
    mu = pm.Normal("mu", mu=0, sigma=5, shape=2)
    sigma = pm.HalfNormal("sigma", sigma=3, shape=2)
    weights = pm.Dirichlet("weights", a=np.ones(2))

    # Likelihood (mixture)
    likelihood = pm.Mixture(
        "obs",
        w=weights,
        comp_dists=[
            pm.Normal.dist(mu=mu[0], sigma=sigma[0]),
            pm.Normal.dist(mu=mu[1], sigma=sigma[1]),
        ],
        observed=data,
    )

    # Sample with NUTS (default)
    trace = pm.sample(2000, tune=1000, cores=2, random_seed=42)

# Diagnostics
print(az.summary(trace, var_names=["mu", "sigma", "weights"]))
az.plot_trace(trace, var_names=["mu", "sigma"])
```

---

## 2. Variational Inference (VI)

### 2.1 VI Concept

```python
"""
Variational Inference: Approximate posterior with optimization instead of sampling.

MCMC: Sample from p(θ|data) directly → exact but slow
VI:   Find q(θ) ≈ p(θ|data) by minimizing KL divergence → fast but approximate

  Objective: minimize KL(q(θ) || p(θ|data))
  Equivalent to: maximize ELBO (Evidence Lower Bound)

  ELBO = E_q[log p(data, θ)] - E_q[log q(θ)]
       = E_q[log p(data|θ)] - KL(q(θ) || p(θ))
         ↑ data fit            ↑ stay close to prior

Mean-Field VI:
  Assume q(θ) = ∏ q_i(θ_i)  (fully factorized)
  Each factor is a simple distribution (e.g., Gaussian)
  Optimize the parameters of each q_i

ADVI (Automatic Differentiation VI):
  - Automatically transforms constrained parameters to unconstrained space
  - Uses stochastic gradient descent to optimize ELBO
  - Available in PyMC via pm.fit()

When to use VI vs MCMC:
  - VI: Large datasets, complex models, approximate answer OK, need speed
  - MCMC: Small-medium datasets, need exact posterior, model diagnostics important
"""
```

### 2.2 ADVI in PyMC

```python
import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

# Generate logistic regression data
np.random.seed(42)
N = 5000
d = 10  # features
X = np.random.randn(N, d)
true_beta = np.random.randn(d) * 0.5
p = 1 / (1 + np.exp(-X @ true_beta))
y = np.random.binomial(1, p)

# Model
with pm.Model() as logistic_model:
    beta = pm.Normal("beta", mu=0, sigma=1, shape=d)
    logit_p = pm.math.dot(X, beta)
    obs = pm.Bernoulli("obs", logit_p=logit_p, observed=y)

    # ADVI (fast approximate inference)
    approx = pm.fit(
        n=30000,
        method="advi",
        random_seed=42,
    )

# Plot ELBO convergence
plt.figure(figsize=(10, 4))
plt.plot(approx.hist)
plt.xlabel("Iteration")
plt.ylabel("ELBO")
plt.title("ADVI Convergence")
plt.tight_layout()
plt.show()

# Sample from approximate posterior
vi_trace = approx.sample(2000)

# Compare VI vs true parameters
vi_means = vi_trace.posterior["beta"].mean(dim=["chain", "draw"]).values
print("\nParameter comparison:")
print(f"{'':>6} {'True':>8} {'VI Mean':>8}")
for i in range(d):
    print(f"  β_{i}: {true_beta[i]:>8.3f} {vi_means[i]:>8.3f}")
```

### 2.3 MCMC vs VI Comparison

```python
"""
MCMC vs VI Comparison:

| Aspect         | MCMC (NUTS)                | VI (ADVI)                 |
|----------------|----------------------------|---------------------------|
| Output         | Samples from posterior     | Approximate distribution  |
| Accuracy       | Exact (given enough samples) | Approximate             |
| Speed          | Slow (sequential)          | Fast (optimization)       |
| Scalability    | O(N) per sample            | Mini-batch possible       |
| Diagnostics    | R-hat, ESS, trace plots    | ELBO convergence          |
| Multimodal     | Can explore multiple modes | May miss modes            |
| Uncertainty    | Full posterior uncertainty  | May underestimate         |
| Best for       | Small-medium N, accuracy   | Large N, speed, screening |
"""

# Side-by-side comparison
with pm.Model() as comparison_model:
    mu = pm.Normal("mu", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=5)
    obs = pm.Normal("obs", mu=mu, sigma=sigma,
                    observed=np.random.normal(3, 1.5, 100))

    # MCMC
    mcmc_trace = pm.sample(2000, tune=1000, random_seed=42)

    # VI
    approx = pm.fit(20000, method="advi", random_seed=42)
    vi_trace = approx.sample(2000)

# Compare posteriors
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, var in zip(axes, ["mu", "sigma"]):
    mcmc_vals = mcmc_trace.posterior[var].values.flatten()
    vi_vals = vi_trace.posterior[var].values.flatten()
    ax.hist(mcmc_vals, bins=50, alpha=0.5, density=True, label="MCMC")
    ax.hist(vi_vals, bins=50, alpha=0.5, density=True, label="VI")
    ax.set_title(var)
    ax.legend()
plt.tight_layout()
plt.show()
```

---

## 3. Hierarchical Models

### 3.1 Partial Pooling

```python
"""
Hierarchical (multilevel) models: share information across groups.

Three approaches:
  1. Complete pooling: Ignore groups, one model for all
     → Ignores group differences
  2. No pooling: Separate model per group
     → Noisy estimates for small groups
  3. Partial pooling (hierarchical): Groups share a common prior
     → Borrows strength across groups (shrinkage)

Example: Estimating batting averages in baseball
  - Player with 4/10 hits (small sample) → shrunk toward league average
  - Player with 100/300 hits (large sample) → close to raw average
"""

import pymc as pm
import numpy as np
import arviz as az

# Simulated data: test scores across 8 schools
np.random.seed(42)
n_schools = 8
true_mu = 50
true_tau = 10
true_theta = np.random.normal(true_mu, true_tau, n_schools)  # school-level means
n_per_school = np.random.randint(10, 100, n_schools)

data = []
school_idx = []
for j in range(n_schools):
    scores = np.random.normal(true_theta[j], 15, n_per_school[j])
    data.extend(scores)
    school_idx.extend([j] * n_per_school[j])

data = np.array(data)
school_idx = np.array(school_idx)

# Hierarchical model
with pm.Model() as hierarchical_model:
    # Hyperpriors (population-level)
    mu = pm.Normal("mu", mu=50, sigma=20)           # population mean
    tau = pm.HalfNormal("tau", sigma=15)              # between-school SD

    # School-level parameters (partial pooling)
    theta = pm.Normal("theta", mu=mu, sigma=tau, shape=n_schools)

    # Within-school variation
    sigma = pm.HalfNormal("sigma", sigma=20)

    # Likelihood
    obs = pm.Normal("obs", mu=theta[school_idx], sigma=sigma, observed=data)

    # Sample
    trace = pm.sample(2000, tune=1000, random_seed=42)

# Compare raw means vs hierarchical estimates
raw_means = [data[school_idx == j].mean() for j in range(n_schools)]
hier_means = trace.posterior["theta"].mean(dim=["chain", "draw"]).values

print(f"\n{'School':>8} {'N':>5} {'Raw Mean':>10} {'Hier Mean':>10} {'True':>8}")
print("-" * 50)
for j in range(n_schools):
    print(f"  {j:>5} {n_per_school[j]:>5} {raw_means[j]:>10.2f} "
          f"{hier_means[j]:>10.2f} {true_theta[j]:>8.2f}")

print(f"\n  Population mean: {trace.posterior['mu'].mean().values:.2f} "
      f"(true: {true_mu})")
print(f"  Between-school SD: {trace.posterior['tau'].mean().values:.2f} "
      f"(true: {true_tau})")
```

---

## 4. Model Comparison

### 4.1 WAIC and LOO-CV

```python
"""
Bayesian Model Comparison:

1. WAIC (Widely Applicable Information Criterion):
   WAIC = -2 * (lppd - p_waic)
   - lppd: log pointwise predictive density
   - p_waic: effective number of parameters
   - Lower is better

2. LOO-CV (Leave-One-Out Cross-Validation via PSIS):
   - Pareto-Smoothed Importance Sampling (PSIS)
   - Approximates LOO-CV without refitting
   - elpd_loo: expected log pointwise predictive density
   - Higher elpd_loo is better
   - k diagnostic: k > 0.7 indicates unreliable estimate

Both are available in ArviZ and preferred over DIC/BIC for Bayesian models.
"""

import pymc as pm
import numpy as np
import arviz as az

# Generate data with a quadratic relationship
np.random.seed(42)
N = 100
x = np.random.uniform(-3, 3, N)
y = 2 + 0.5 * x - 0.3 * x**2 + np.random.normal(0, 1, N)

# Model 1: Linear
with pm.Model() as linear_model:
    a = pm.Normal("a", mu=0, sigma=5)
    b = pm.Normal("b", mu=0, sigma=5)
    sigma = pm.HalfNormal("sigma", sigma=3)
    mu = a + b * x
    obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
    trace_linear = pm.sample(2000, tune=1000, random_seed=42,
                              idata_kwargs={"log_likelihood": True})

# Model 2: Quadratic
with pm.Model() as quad_model:
    a = pm.Normal("a", mu=0, sigma=5)
    b = pm.Normal("b", mu=0, sigma=5)
    c = pm.Normal("c", mu=0, sigma=5)
    sigma = pm.HalfNormal("sigma", sigma=3)
    mu = a + b * x + c * x**2
    obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
    trace_quad = pm.sample(2000, tune=1000, random_seed=42,
                            idata_kwargs={"log_likelihood": True})

# Compare with LOO
comparison = az.compare(
    {"linear": trace_linear, "quadratic": trace_quad},
    ic="loo",
)
print("Model Comparison (LOO-CV):")
print(comparison)

# Plot comparison
az.plot_compare(comparison, insample_dev=False)
```

---

## 5. Practical Considerations

### 5.1 Diagnostics Checklist

```python
"""
Bayesian Diagnostics Checklist:

1. Convergence:
   - R-hat < 1.01 for all parameters
   - Trace plots: chains should mix well (fuzzy caterpillars)
   - No divergences (NUTS specific)

2. Effective Sample Size (ESS):
   - ESS > 400 for reliable posterior estimates
   - ESS_bulk: overall posterior estimation
   - ESS_tail: tail probability estimation

3. NUTS Diagnostics:
   - Divergences = 0 (or very few)
   - Tree depth not hitting max (default 10)
   - Energy plot: marginal energy ≈ energy transition

4. Posterior Predictive Check:
   - Simulate data from posterior
   - Compare with observed data
   - If mismatched → model misspecification

5. Prior Predictive Check:
   - Sample from prior → simulate data
   - Check if prior generates plausible data
   - If absurd → revise priors
"""

# Comprehensive diagnostics
import pymc as pm
import numpy as np
import arviz as az

np.random.seed(42)
y_obs = np.random.normal(5, 2, 50)

with pm.Model() as diag_model:
    mu = pm.Normal("mu", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=5)
    obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=y_obs)

    # Prior predictive check
    prior_pred = pm.sample_prior_predictive(500, random_seed=42)

    # Sample
    trace = pm.sample(2000, tune=1000, random_seed=42)

    # Posterior predictive check
    post_pred = pm.sample_posterior_predictive(trace, random_seed=42)

# R-hat and ESS
summary = az.summary(trace)
print("Parameter Summary:")
print(summary)

# Check for issues
rhat_ok = (summary["r_hat"] < 1.01).all()
ess_ok = (summary["ess_bulk"] > 400).all()
print(f"\nR-hat OK: {rhat_ok}")
print(f"ESS OK: {ess_ok}")

# Divergences
n_divergences = trace.sample_stats["diverging"].sum().values
print(f"Divergences: {n_divergences}")
```

---

## 6. Practice Problems

### Exercise 1: Hierarchical Regression

```python
"""
Build a hierarchical linear regression:
1. Data: Student test scores across 20 schools
   - Each school has 10-50 students
   - Covariates: study hours, prior GPA
2. Model: School-specific slopes and intercepts
   - Partial pooling: share strength across schools
3. Compare: no pooling vs partial pooling vs complete pooling
4. Visualize shrinkage: how do small-school estimates change?
5. Run diagnostics: R-hat, ESS, posterior predictive check
"""
```

### Exercise 2: MCMC vs VI

```python
"""
Compare MCMC and VI on the same model:
1. Generate data from a mixture of 3 Gaussians (N=1000)
2. Fit with NUTS (pm.sample) and ADVI (pm.fit)
3. Compare: posterior means, uncertainties, computation time
4. When does VI's approximation break down?
5. Try FullRank ADVI (method='fullrank_advi') and compare
"""
```

---

## 7. Summary

### Key Takeaways

| Concept | Description |
|---------|-------------|
| **HMC** | Uses gradient for efficient sampling; scales better than Metropolis |
| **NUTS** | Auto-tunes HMC trajectory length; default in PyMC/Stan |
| **ADVI** | Fast approximate inference via optimization; useful for large data |
| **Hierarchical models** | Partial pooling: share information across groups |
| **LOO-CV** | Bayesian model comparison via Pareto-smoothed importance sampling |
| **Prior predictive** | Sanity check: does the prior generate plausible data? |

### Best Practices

1. **Start with NUTS** — it's the gold standard for moderate-dimensional problems
2. **Use VI for screening** — explore model space quickly, then validate with MCMC
3. **Check diagnostics** — R-hat < 1.01, ESS > 400, zero divergences
4. **Prior predictive checks** — always verify priors before fitting
5. **Hierarchical when grouped** — partial pooling almost always beats no pooling
6. **Compare with LOO** — prefer LOO-CV over WAIC; check Pareto k values

### Next Steps

- **L27**: Causal Inference — move from correlation to causation
- Return to **L15-L16** (Bayesian Basics/Inference) for foundational concepts
