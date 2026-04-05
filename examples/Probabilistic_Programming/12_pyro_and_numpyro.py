"""
Pyro and NumPyro Examples
- Model/guide pattern, SVI, MCMC
Note: Requires pyro-ppl and/or numpyro installed.
"""
import numpy as np


def pyro_coin_flip():
    """Bayesian coin flip with Pyro SVI."""
    try:
        import torch
        import pyro
        import pyro.distributions as dist
        from pyro.infer import SVI, Trace_ELBO
        from pyro.optim import Adam

        pyro.clear_param_store()
        data = torch.tensor([1.0]*7 + [0.0]*3)

        def model(data):
            theta = pyro.sample("theta", dist.Beta(1, 1))
            with pyro.plate("data", len(data)):
                pyro.sample("obs", dist.Bernoulli(theta), obs=data)

        def guide(data):
            a = pyro.param("a", torch.tensor(1.0), constraint=dist.constraints.positive)
            b = pyro.param("b", torch.tensor(1.0), constraint=dist.constraints.positive)
            pyro.sample("theta", dist.Beta(a, b))

        svi = SVI(model, guide, Adam({"lr": 0.01}), Trace_ELBO())
        for step in range(2000):
            svi.step(data)

        a = pyro.param("a").item()
        b = pyro.param("b").item()
        print(f"Pyro SVI: Beta({a:.2f}, {b:.2f}), mean={a/(a+b):.3f}")
        print(f"Exact:    Beta(8, 4), mean={8/12:.3f}")
    except ImportError:
        print("Pyro not installed. Run: pip install pyro-ppl torch")


def numpyro_regression():
    """Bayesian regression with NumPyro NUTS."""
    try:
        import jax
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS

        def model(X, y=None):
            w = numpyro.sample("w", dist.Normal(0, 5))
            b = numpyro.sample("b", dist.Normal(0, 10))
            sigma = numpyro.sample("sigma", dist.HalfNormal(5))
            mu = w * X + b
            numpyro.sample("y", dist.Normal(mu, sigma), obs=y)

        np.random.seed(42)
        X = np.random.randn(50).astype(np.float32)
        y = (2.5 * X - 1.0 + np.random.normal(0, 0.5, 50)).astype(np.float32)

        kernel = NUTS(model)
        mcmc = MCMC(kernel, num_warmup=500, num_samples=2000, num_chains=1)
        mcmc.run(jax.random.PRNGKey(42), jnp.array(X), jnp.array(y))
        mcmc.print_summary()
    except ImportError:
        print("NumPyro not installed. Run: pip install numpyro jax jaxlib")


if __name__ == "__main__":
    print("=== Pyro SVI ===")
    pyro_coin_flip()
    print("\n=== NumPyro MCMC ===")
    numpyro_regression()
