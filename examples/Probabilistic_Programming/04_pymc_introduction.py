"""
PyMC Introduction Examples
- Model building, sampling, trace analysis, posterior predictive checks
"""
import numpy as np

# Note: This example requires pymc and arviz
# pip install pymc arviz matplotlib

def coin_flip_model():
    """Beta-Binomial coin flip model with PyMC."""
    import pymc as pm
    import arviz as az

    with pm.Model() as model:
        theta = pm.Beta("theta", alpha=1, beta=1)
        y = pm.Binomial("y", n=20, p=theta, observed=14)
        trace = pm.sample(2000, tune=1000, chains=4, random_seed=42)

    print(az.summary(trace, var_names=["theta"]))
    return trace


def normal_model():
    """Estimate mean and variance of Normal data."""
    import pymc as pm
    import arviz as az

    np.random.seed(42)
    data = np.random.normal(5.0, 2.0, 100)

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=5)
        y = pm.Normal("y", mu=mu, sigma=sigma, observed=data)
        trace = pm.sample(3000, tune=1000, chains=4, random_seed=42)

    print(az.summary(trace, var_names=["mu", "sigma"]))
    return trace


def ab_test():
    """Bayesian A/B test analysis."""
    import pymc as pm
    import arviz as az

    with pm.Model() as model:
        p_a = pm.Beta("p_a", alpha=1, beta=1)
        p_b = pm.Beta("p_b", alpha=1, beta=1)
        delta = pm.Deterministic("delta", p_b - p_a)
        pm.Binomial("obs_a", n=1000, p=p_a, observed=120)
        pm.Binomial("obs_b", n=1000, p=p_b, observed=145)
        trace = pm.sample(5000, tune=1000, chains=4, random_seed=42)

    delta_samples = trace.posterior["delta"].values.flatten()
    print(f"P(B > A) = {(delta_samples > 0).mean():.4f}")
    print(f"Expected lift = {delta_samples.mean()*100:.2f}%")


if __name__ == "__main__":
    print("=== Coin Flip Model ===")
    coin_flip_model()
    print("\n=== Normal Model ===")
    normal_model()
    print("\n=== A/B Test ===")
    ab_test()
