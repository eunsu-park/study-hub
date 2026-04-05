# 12. Pyro and NumPyro

[Previous: Bayesian Optimization](./11_Bayesian_Optimization.md) | [Next: Normalizing Flows](./13_Normalizing_Flows.md)

---

> **Framework Note**: This lesson uses Pyro (PyTorch backend) and NumPyro (JAX backend).
>
> Installation: `pip install pyro-ppl torch numpyro jax jaxlib`

## Learning Objectives

- Understand Pyro's model/guide paradigm for probabilistic programming
- Write models using `pyro.sample` and effect handlers
- Implement Stochastic Variational Inference (SVI) in Pyro
- Use NumPyro for JAX-accelerated MCMC and SVI
- Compare Pyro/NumPyro with PyMC for different use cases

---

## 1. Pyro Fundamentals

Pyro is a deep probabilistic programming language built on PyTorch. Its key innovation is combining neural networks with probabilistic inference.

### 1.1 Model and Guide

```python
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import Adam
import matplotlib.pyplot as plt
import numpy as np

pyro.set_rng_seed(42)
pyro.clear_param_store()

# A Pyro model is a regular Python function with pyro.sample statements
def coin_model(data=None):
    """Bayesian coin flip model."""
    theta = pyro.sample("theta", dist.Beta(1, 1))
    with pyro.plate("data", len(data) if data is not None else 1):
        obs = pyro.sample("obs", dist.Bernoulli(theta), obs=data)
    return obs

# A guide is the variational approximation (for SVI)
def coin_guide(data=None):
    """Variational guide for coin model."""
    alpha_q = pyro.param("alpha_q", torch.tensor(1.0), constraint=dist.constraints.positive)
    beta_q = pyro.param("beta_q", torch.tensor(1.0), constraint=dist.constraints.positive)
    pyro.sample("theta", dist.Beta(alpha_q, beta_q))
```

### 1.2 Stochastic Variational Inference

```python
# Data: 7 heads out of 10
data = torch.tensor([1.0, 1, 1, 1, 1, 1, 1, 0, 0, 0])

# SVI setup
optimizer = Adam({"lr": 0.01})
svi = SVI(coin_model, coin_guide, optimizer, loss=Trace_ELBO())

# Training loop
losses = []
for step in range(2000):
    loss = svi.step(data)
    losses.append(loss)
    if step % 500 == 0:
        print(f"Step {step}: loss = {loss:.3f}")

# Learned variational parameters
alpha_q = pyro.param("alpha_q").item()
beta_q = pyro.param("beta_q").item()
print(f"\nLearned posterior: Beta({alpha_q:.2f}, {beta_q:.2f})")
print(f"Posterior mean: {alpha_q / (alpha_q + beta_q):.3f}")
print(f"Exact posterior: Beta(8, 4), mean = {8/12:.3f}")
```

---

## 2. Pyro Effect Handlers

Effect handlers are Pyro's mechanism for transforming model behavior without modifying the model code.

```python
# trace: record all sample sites
from pyro.poutine import trace, replay, condition

traced = trace(coin_model).get_trace(data)
print("Sample sites:")
for name, site in traced.nodes.items():
    if site["type"] == "sample":
        print(f"  {name}: value={site['value']}, log_prob={site['log_prob_sum']:.3f}")

# condition: fix a latent variable to a value
conditioned_model = condition(coin_model, data={"theta": torch.tensor(0.7)})

# replay: replay one execution's choices in another
```

---

## 3. Bayesian Linear Regression in Pyro

```python
# Generate data
np.random.seed(42)
N = 100
X = torch.randn(N, 2)
true_w = torch.tensor([2.5, -1.0])
true_b = torch.tensor(1.5)
y = X @ true_w + true_b + torch.randn(N) * 0.5

def regression_model(X, y=None):
    """Bayesian linear regression."""
    D = X.shape[1]
    w = pyro.sample("w", dist.Normal(torch.zeros(D), 5 * torch.ones(D)).to_event(1))
    b = pyro.sample("b", dist.Normal(0.0, 10.0))
    sigma = pyro.sample("sigma", dist.HalfNormal(5.0))
    mu = X @ w + b
    with pyro.plate("data", len(X)):
        pyro.sample("y", dist.Normal(mu, sigma), obs=y)

def regression_guide(X, y=None):
    """Mean-field variational guide."""
    D = X.shape[1]
    w_loc = pyro.param("w_loc", torch.zeros(D))
    w_scale = pyro.param("w_scale", torch.ones(D), constraint=dist.constraints.positive)
    b_loc = pyro.param("b_loc", torch.tensor(0.0))
    b_scale = pyro.param("b_scale", torch.tensor(1.0), constraint=dist.constraints.positive)
    sigma_loc = pyro.param("sigma_loc", torch.tensor(1.0), constraint=dist.constraints.positive)

    pyro.sample("w", dist.Normal(w_loc, w_scale).to_event(1))
    pyro.sample("b", dist.Normal(b_loc, b_scale))
    pyro.sample("sigma", dist.LogNormal(torch.log(sigma_loc), 0.1))

# Train
pyro.clear_param_store()
svi = SVI(regression_model, regression_guide, Adam({"lr": 0.01}), Trace_ELBO())
for step in range(3000):
    loss = svi.step(X, y)
    if step % 1000 == 0:
        print(f"Step {step}: ELBO loss = {loss:.2f}")

print(f"\nLearned w: {pyro.param('w_loc').detach().numpy().round(3)}")
print(f"True w:    {true_w.numpy()}")
print(f"Learned b: {pyro.param('b_loc').item():.3f}, True b: {true_b.item():.3f}")
```

---

## 4. AutoGuide: Automatic Variational Families

```python
from pyro.infer.autoguide import AutoDiagonalNormal, AutoMultivariateNormal, AutoNormal

# Instead of writing guides manually:
auto_guide = AutoDiagonalNormal(regression_model)

pyro.clear_param_store()
svi = SVI(regression_model, auto_guide, Adam({"lr": 0.01}), Trace_ELBO())
for step in range(3000):
    loss = svi.step(X, y)

# Get posterior samples
predictive = Predictive(regression_model, guide=auto_guide, num_samples=1000)
posterior = predictive(X)
print(f"Posterior w mean: {posterior['w'].mean(0).detach().numpy().round(3)}")
```

---

## 5. MCMC in Pyro

```python
from pyro.infer import MCMC, NUTS

# NUTS sampling (like Stan/PyMC)
kernel = NUTS(regression_model)
mcmc = MCMC(kernel, num_samples=2000, warmup_steps=500, num_chains=4)
mcmc.run(X, y)

# Get samples
posterior_samples = mcmc.get_samples()
print(f"MCMC w mean: {posterior_samples['w'].mean(0).numpy().round(3)}")
print(f"MCMC b mean: {posterior_samples['b'].mean().item():.3f}")
print(f"MCMC sigma mean: {posterior_samples['sigma'].mean().item():.3f}")

mcmc.summary()
```

---

## 6. NumPyro: JAX-Accelerated Inference

NumPyro is Pyro's JAX-based sibling. It is significantly faster for MCMC because JAX compiles the model to XLA.

### 6.1 NumPyro Model

```python
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as ndist
from numpyro.infer import MCMC as NMCMC, NUTS as NNUTS, SVI as NSVI, Predictive as NPredictive

numpyro.set_host_device_count(4)

def numpyro_regression(X, y=None):
    """Bayesian regression in NumPyro."""
    D = X.shape[1]
    w = numpyro.sample("w", ndist.Normal(jnp.zeros(D), 5 * jnp.ones(D)))
    b = numpyro.sample("b", ndist.Normal(0.0, 10.0))
    sigma = numpyro.sample("sigma", ndist.HalfNormal(5.0))
    mu = X @ w + b
    with numpyro.plate("data", len(X)):
        numpyro.sample("y", ndist.Normal(mu, sigma), obs=y)

# Convert to JAX arrays
X_jax = jnp.array(X.numpy())
y_jax = jnp.array(y.numpy())

# NUTS sampling
kernel = NNUTS(numpyro_regression)
mcmc = NMCMC(kernel, num_warmup=500, num_samples=2000, num_chains=4)
mcmc.run(jax.random.PRNGKey(42), X_jax, y_jax)
mcmc.print_summary()
```

### 6.2 NumPyro SVI

```python
from numpyro.infer import SVI as NSVI, Trace_ELBO as NTrace_ELBO
from numpyro.infer.autoguide import AutoDiagonalNormal as NAutoDiag
from numpyro.optim import Adam as NAdam

guide = NAutoDiag(numpyro_regression)
svi = NSVI(numpyro_regression, guide, NAdam(0.01), NTrace_ELBO())
svi_result = svi.run(jax.random.PRNGKey(42), 5000, X_jax, y_jax)
```

---

## 7. Pyro for Deep Probabilistic Models

### 7.1 Bayesian Neural Network

```python
import torch.nn as nn

class BNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def bnn_model(X, y=None):
    """Bayesian Neural Network with weight priors."""
    net = BNN(2, 20, 1)
    # Place priors on all parameters
    lifted_module = pyro.random_module("module", net, prior={
        "fc1.weight": dist.Normal(0, 1).expand([20, 2]).to_event(2),
        "fc1.bias": dist.Normal(0, 1).expand([20]).to_event(1),
        "fc2.weight": dist.Normal(0, 1).expand([1, 20]).to_event(2),
        "fc2.bias": dist.Normal(0, 1).expand([1]).to_event(1),
    })
    sampled_net = lifted_module()
    pred = sampled_net(X).squeeze(-1)
    sigma = pyro.sample("sigma", dist.HalfNormal(1.0))
    with pyro.plate("data", len(X)):
        pyro.sample("y", dist.Normal(pred, sigma), obs=y)
```

---

## 8. Pyro vs PyMC vs Stan

| Feature | Pyro/NumPyro | PyMC | Stan |
|---------|-------------|------|------|
| Backend | PyTorch/JAX | PyTensor | C++ |
| Deep learning | Native | Limited | None |
| SVI | Yes (primary) | ADVI only | ADVI |
| MCMC | NUTS | NUTS | NUTS (reference) |
| Mini-batch | Easy | pm.Minibatch | No |
| Discrete latents | Enumeration | Metropolis | Marginalize |
| GPU | Yes | Limited | No |
| Best for | Deep + probabilistic | Traditional Bayes | Gold standard MCMC |

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| Pyro model | Python function with `pyro.sample` statements |
| Guide | Variational approximation; mirrors the model |
| SVI | Scalable inference via mini-batch stochastic gradient ELBO |
| AutoGuide | Automatic variational family construction |
| Effect handlers | Transform model execution (trace, condition, replay) |
| NumPyro | JAX backend for fast compiled MCMC |
| Use case | Deep probabilistic models, large-scale inference |

---

## References

1. Bingham, E., et al. (2019). "Pyro: Deep Universal Probabilistic Programming." *JMLR*, 20(28), 1-6.
2. Phan, D., Pradhan, N., & Jankowiak, M. (2019). "Composable Effects for Flexible and Accelerated Probabilistic Programming in NumPyro." arXiv:1912.11554.
3. Pyro documentation: https://pyro.ai/
4. NumPyro documentation: https://num.pyro.ai/

---

[Previous: Bayesian Optimization](./11_Bayesian_Optimization.md) | [Next: Normalizing Flows →](./13_Normalizing_Flows.md)
