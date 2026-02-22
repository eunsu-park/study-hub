"""
Bayesian Advanced Methods
=========================
Demonstrates:
- HMC/NUTS sampling with PyMC
- ADVI (Variational Inference) comparison
- Hierarchical models (partial pooling)
- Model comparison with LOO-CV
- Diagnostic checks

Requirements:
    pip install pymc arviz matplotlib numpy
"""

import numpy as np
import matplotlib.pyplot as plt

# ── 1. NUTS Sampling ───────────────────────────────────────────────

def demo_nuts():
    """Demonstrate NUTS sampling with PyMC."""
    import pymc as pm
    import arviz as az

    np.random.seed(42)
    N = 100
    true_mu, true_sigma = 5.0, 2.0
    data = np.random.normal(true_mu, true_sigma, N)

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=5)
        obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=data)

        # NUTS (default sampler)
        trace = pm.sample(2000, tune=1000, cores=2, random_seed=42)

    # Diagnostics
    summary = az.summary(trace)
    print("Parameter Summary:")
    print(summary)
    print(f"\nTrue values: mu={true_mu}, sigma={true_sigma}")

    # Trace plot
    az.plot_trace(trace)
    plt.suptitle("NUTS Trace Plot", y=1.02)
    plt.tight_layout()
    plt.show()

    # Posterior plot
    az.plot_posterior(trace, ref_val={"mu": true_mu, "sigma": true_sigma})
    plt.tight_layout()
    plt.show()


# ── 2. MCMC vs VI ─────────────────────────────────────────────────

def demo_mcmc_vs_vi():
    """Compare MCMC (NUTS) and VI (ADVI)."""
    import pymc as pm
    import arviz as az
    import time

    np.random.seed(42)
    N = 500
    X = np.random.randn(N, 3)
    true_beta = np.array([2.0, -1.0, 0.5])
    y = X @ true_beta + np.random.normal(0, 1, N)

    with pm.Model() as model:
        beta = pm.Normal("beta", mu=0, sigma=5, shape=3)
        sigma = pm.HalfNormal("sigma", sigma=3)
        obs = pm.Normal("obs", mu=pm.math.dot(X, beta), sigma=sigma, observed=y)

        # MCMC
        t0 = time.time()
        mcmc_trace = pm.sample(2000, tune=1000, cores=2, random_seed=42)
        mcmc_time = time.time() - t0

        # VI (ADVI)
        t0 = time.time()
        approx = pm.fit(30000, method="advi", random_seed=42)
        vi_trace = approx.sample(2000)
        vi_time = time.time() - t0

    # Compare
    print(f"MCMC time: {mcmc_time:.1f}s")
    print(f"VI time:   {vi_time:.1f}s")
    print(f"Speedup:   {mcmc_time/vi_time:.1f}x")

    print(f"\n{'':>8} {'True':>8} {'MCMC':>8} {'VI':>8}")
    mcmc_means = mcmc_trace.posterior["beta"].mean(dim=["chain", "draw"]).values
    vi_means = vi_trace.posterior["beta"].mean(dim=["chain", "draw"]).values
    for i in range(3):
        print(f"  beta_{i}: {true_beta[i]:>8.3f} {mcmc_means[i]:>8.3f} {vi_means[i]:>8.3f}")

    # Plot ELBO convergence
    plt.figure(figsize=(10, 4))
    plt.plot(approx.hist[-5000:])
    plt.xlabel("Iteration")
    plt.ylabel("ELBO")
    plt.title("ADVI Convergence (last 5000 iterations)")
    plt.tight_layout()
    plt.show()

    # Compare posteriors
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, ax in enumerate(axes):
        mcmc_vals = mcmc_trace.posterior["beta"].values[:, :, i].flatten()
        vi_vals = vi_trace.posterior["beta"].values[:, :, i].flatten()
        ax.hist(mcmc_vals, bins=50, alpha=0.5, density=True, label="MCMC")
        ax.hist(vi_vals, bins=50, alpha=0.5, density=True, label="VI")
        ax.axvline(true_beta[i], color="red", linestyle="--", label="True")
        ax.set_title(f"beta_{i}")
        ax.legend()
    plt.suptitle("MCMC vs VI Posterior Comparison")
    plt.tight_layout()
    plt.show()


# ── 3. Hierarchical Model ─────────────────────────────────────────

def demo_hierarchical():
    """Hierarchical model with partial pooling."""
    import pymc as pm
    import arviz as az

    np.random.seed(42)
    n_groups = 8
    true_mu = 50
    true_tau = 10
    true_theta = np.random.normal(true_mu, true_tau, n_groups)
    n_per_group = np.array([5, 10, 15, 20, 30, 50, 75, 100])

    # Generate data
    data = []
    group_idx = []
    for j in range(n_groups):
        obs = np.random.normal(true_theta[j], 15, n_per_group[j])
        data.extend(obs)
        group_idx.extend([j] * n_per_group[j])

    data = np.array(data)
    group_idx = np.array(group_idx)

    # Hierarchical model
    with pm.Model() as hier_model:
        mu = pm.Normal("mu", mu=50, sigma=20)
        tau = pm.HalfNormal("tau", sigma=15)
        theta = pm.Normal("theta", mu=mu, sigma=tau, shape=n_groups)
        sigma = pm.HalfNormal("sigma", sigma=20)
        obs = pm.Normal("obs", mu=theta[group_idx], sigma=sigma, observed=data)
        trace = pm.sample(2000, tune=1000, random_seed=42)

    # Compare raw means vs hierarchical estimates
    raw_means = [data[group_idx == j].mean() for j in range(n_groups)]
    hier_means = trace.posterior["theta"].mean(dim=["chain", "draw"]).values
    hier_sd = trace.posterior["theta"].std(dim=["chain", "draw"]).values

    # Shrinkage plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for j in range(n_groups):
        ax.plot([raw_means[j], hier_means[j]], [j, j], "k-", alpha=0.3)
    ax.scatter(raw_means, range(n_groups), s=100, c="blue", zorder=5,
               label="Raw mean (no pooling)")
    ax.scatter(hier_means, range(n_groups), s=100, c="red", zorder=5,
               label="Hierarchical mean (partial pooling)")
    ax.scatter(true_theta, range(n_groups), s=100, c="green", marker="x", zorder=5,
               label="True value")
    ax.axvline(true_mu, color="gray", linestyle="--", alpha=0.5, label="Population mean")

    for j in range(n_groups):
        ax.errorbar(hier_means[j], j, xerr=hier_sd[j]*2, fmt="none", c="red", alpha=0.3)

    ax.set_yticks(range(n_groups))
    ax.set_yticklabels([f"Group {j} (n={n_per_group[j]})" for j in range(n_groups)])
    ax.set_xlabel("Estimated Mean")
    ax.set_title("Shrinkage: Small Groups Pulled Toward Population Mean")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    print(f"Population mean: {trace.posterior['mu'].mean().values:.2f} (true: {true_mu})")
    print(f"Between-group SD: {trace.posterior['tau'].mean().values:.2f} (true: {true_tau})")


# ── 4. Model Comparison ───────────────────────────────────────────

def demo_model_comparison():
    """Compare models with LOO-CV."""
    import pymc as pm
    import arviz as az

    np.random.seed(42)
    N = 100
    x = np.random.uniform(-3, 3, N)
    y = 2 + 0.5 * x - 0.3 * x**2 + np.random.normal(0, 1, N)

    # Linear model
    with pm.Model() as linear:
        a = pm.Normal("a", 0, 5)
        b = pm.Normal("b", 0, 5)
        sigma = pm.HalfNormal("sigma", 3)
        mu = a + b * x
        obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
        trace_lin = pm.sample(2000, tune=1000, random_seed=42,
                              idata_kwargs={"log_likelihood": True})

    # Quadratic model
    with pm.Model() as quadratic:
        a = pm.Normal("a", 0, 5)
        b = pm.Normal("b", 0, 5)
        c = pm.Normal("c", 0, 5)
        sigma = pm.HalfNormal("sigma", 3)
        mu = a + b * x + c * x**2
        obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
        trace_quad = pm.sample(2000, tune=1000, random_seed=42,
                               idata_kwargs={"log_likelihood": True})

    # Compare
    comparison = az.compare({"linear": trace_lin, "quadratic": trace_quad}, ic="loo")
    print("Model Comparison (LOO-CV):")
    print(comparison)

    az.plot_compare(comparison, insample_dev=False)
    plt.title("Model Comparison")
    plt.tight_layout()
    plt.show()


# ── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    demos = {
        "nuts": demo_nuts,
        "vi": demo_mcmc_vs_vi,
        "hier": demo_hierarchical,
        "compare": demo_model_comparison,
    }

    if len(sys.argv) < 2 or sys.argv[1] not in demos:
        print("Usage: python 13_bayesian_advanced.py <demo>")
        print(f"Available: {', '.join(demos.keys())}")
        print("\nRunning 'hier' demo...")
        demo_hierarchical()
    else:
        demos[sys.argv[1]]()
