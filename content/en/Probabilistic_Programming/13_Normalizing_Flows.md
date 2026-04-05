# 13. Normalizing Flows

[Previous: Pyro and NumPyro](./12_Pyro_and_NumPyro.md) | [Next: Bayesian Deep Learning](./14_Bayesian_Deep_Learning.md)

---

> **Framework Note**: This lesson uses PyTorch for flow implementations and Pyro for integration with PPLs.
>
> Installation: `pip install torch pyro-ppl numpy matplotlib`

## Learning Objectives

- Understand normalizing flows as invertible transformations of simple distributions
- Implement planar flows, RealNVP, and Neural Spline Flows
- Use flows as flexible variational posteriors
- Apply flows for density estimation and generative modeling
- Integrate flows with probabilistic programming frameworks

---

## 1. The Normalizing Flow Idea

A normalizing flow transforms a simple base distribution (e.g., standard Gaussian) through a series of invertible, differentiable transformations to produce a complex target distribution.

### 1.1 Change of Variables

$$\mathbf{z}_K = f_K \circ f_{K-1} \circ \cdots \circ f_1(\mathbf{z}_0), \quad \mathbf{z}_0 \sim q_0(\mathbf{z}_0)$$

$$\log q_K(\mathbf{z}_K) = \log q_0(\mathbf{z}_0) - \sum_{k=1}^{K} \log \left| \det \frac{\partial f_k}{\partial \mathbf{z}_{k-1}} \right|$$

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Simple example: transform standard normal to match a target
z0 = torch.randn(10000, 2)  # base distribution

# Affine flow: z1 = exp(s) * z0 + t
s = torch.tensor([0.5, -0.3])
t = torch.tensor([2.0, -1.0])
z1 = torch.exp(s) * z0 + t

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.scatter(z0[:, 0], z0[:, 1], alpha=0.1, s=1)
ax1.set_title("Base: N(0, I)")
ax2.scatter(z1[:, 0].detach(), z1[:, 1].detach(), alpha=0.1, s=1)
ax2.set_title("After affine flow")
plt.tight_layout()
plt.savefig("flow_basic.png", dpi=100)
plt.show()
```

---

## 2. Planar Flows

The simplest non-trivial flow. Each layer applies:

$$f(\mathbf{z}) = \mathbf{z} + \mathbf{u} \cdot h(\mathbf{w}^T \mathbf{z} + b)$$

```python
class PlanarFlow(nn.Module):
    """Single planar flow layer."""

    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim))
        self.u = nn.Parameter(torch.randn(dim))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, z):
        linear = z @ self.w + self.b
        f_z = z + self.u * torch.tanh(linear).unsqueeze(-1)

        # Log-determinant of Jacobian
        psi = (1 - torch.tanh(linear)**2) * self.w
        log_det = torch.log(torch.abs(1 + psi @ self.u) + 1e-8)
        return f_z, log_det


class PlanarFlowSequence(nn.Module):
    """Stack of planar flows."""

    def __init__(self, dim, n_flows):
        super().__init__()
        self.flows = nn.ModuleList([PlanarFlow(dim) for _ in range(n_flows)])

    def forward(self, z):
        log_det_sum = 0
        for flow in self.flows:
            z, log_det = flow(z)
            log_det_sum += log_det
        return z, log_det_sum


# Transform N(0,I) through 10 planar flows
flow = PlanarFlowSequence(dim=2, n_flows=10)
z0 = torch.randn(5000, 2)
with torch.no_grad():
    zK, log_det = flow(z0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.scatter(z0[:, 0], z0[:, 1], alpha=0.1, s=1)
ax1.set_title("Base distribution")
ax2.scatter(zK[:, 0], zK[:, 1], alpha=0.1, s=1)
ax2.set_title("After 10 planar flows (untrained)")
plt.tight_layout()
plt.savefig("planar_flow.png", dpi=100)
plt.show()
```

---

## 3. RealNVP (Real-valued Non-Volume Preserving)

RealNVP uses affine coupling layers, which are easy to invert and have tractable Jacobians.

```python
class AffineCouplingLayer(nn.Module):
    """Affine coupling layer for RealNVP."""

    def __init__(self, dim, hidden_dim=64, mask_type='even'):
        super().__init__()
        self.dim = dim
        # Mask: which dimensions to keep fixed
        if mask_type == 'even':
            self.mask = torch.tensor([i % 2 == 0 for i in range(dim)]).float()
        else:
            self.mask = torch.tensor([i % 2 == 1 for i in range(dim)]).float()

        # Scale and translation networks
        self.s_net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, dim), nn.Tanh(),
        )
        self.t_net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, z):
        z_masked = z * self.mask
        s = self.s_net(z_masked) * (1 - self.mask)
        t = self.t_net(z_masked) * (1 - self.mask)
        z_out = z_masked + (1 - self.mask) * (z * torch.exp(s) + t)
        log_det = s.sum(dim=-1)
        return z_out, log_det

    def inverse(self, z_out):
        z_masked = z_out * self.mask
        s = self.s_net(z_masked) * (1 - self.mask)
        t = self.t_net(z_masked) * (1 - self.mask)
        z = z_masked + (1 - self.mask) * (z_out - t) * torch.exp(-s)
        return z


class RealNVP(nn.Module):
    """RealNVP normalizing flow."""

    def __init__(self, dim, n_layers=6, hidden_dim=64):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            mask_type = 'even' if i % 2 == 0 else 'odd'
            self.layers.append(AffineCouplingLayer(dim, hidden_dim, mask_type))

    def forward(self, z):
        log_det_sum = 0
        for layer in self.layers:
            z, log_det = layer(z)
            log_det_sum += log_det
        return z, log_det_sum

    def inverse(self, x):
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x

    def log_prob(self, x):
        z = self.inverse(x)
        log_pz = -0.5 * (z**2 + np.log(2 * np.pi)).sum(dim=-1)
        # Compute log_det through forward pass
        _, log_det = self.forward(z)
        return log_pz + log_det
```

### 3.1 Training RealNVP for Density Estimation

```python
def train_realnvp(flow, target_samples, n_epochs=2000, lr=1e-3):
    """Train RealNVP by maximizing log-likelihood."""
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)

    for epoch in range(n_epochs):
        # Negative log-likelihood loss
        loss = -flow.log_prob(target_samples).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}: NLL = {loss.item():.3f}")

# Generate target: two moons
from sklearn.datasets import make_moons
target_data, _ = make_moons(n_samples=5000, noise=0.05)
target_tensor = torch.tensor(target_data, dtype=torch.float32)

flow_model = RealNVP(dim=2, n_layers=8, hidden_dim=64)
train_realnvp(flow_model, target_tensor)

# Generate samples
with torch.no_grad():
    z = torch.randn(5000, 2)
    x_gen, _ = flow_model(z)
```

---

## 4. Neural Spline Flows

The state-of-the-art flow architecture using rational-quadratic spline transformations.

```python
# Neural Spline Flows use monotonic rational-quadratic splines
# as the coupling transform instead of affine transforms.
# This gives much more expressive transformations.

# Key idea:
# - Divide the domain into K bins
# - In each bin, use a rational-quadratic spline (parameterized by widths, heights, derivatives)
# - The network predicts these spline parameters
# - Guaranteed monotonic → invertible with analytic Jacobian

# In practice, use the implementation from nflows or Pyro:
# pip install nflows
#
# from nflows.flows import MaskedAutoregressiveFlow
# from nflows.transforms import MaskedPiecewiseRationalQuadraticAutoregressiveTransform
```

---

## 5. Flows as Variational Posteriors

The most important application for probabilistic programming: using flows to create flexible variational distributions.

```python
import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
from pyro.infer import SVI, Trace_ELBO

def model(data):
    """Model with multi-modal posterior."""
    z = pyro.sample("z", dist.Normal(torch.zeros(2), torch.ones(2)).to_event(1))
    with pyro.plate("data", len(data)):
        pyro.sample("x", dist.Normal(z, 0.5 * torch.ones(2)).to_event(1), obs=data)

# Flow-based guide
def flow_guide(data):
    """Normalizing flow variational posterior."""
    # Base distribution
    base_dist = dist.Normal(torch.zeros(2), torch.ones(2)).to_event(1)

    # Spline flow transform
    transforms = [
        T.spline_autoregressive(2, hidden_dims=[32, 32])
        for _ in range(4)
    ]
    flow_dist = dist.TransformedDistribution(base_dist, transforms)
    pyro.sample("z", flow_dist)
```

---

## 6. Continuous Normalizing Flows (CNFs)

Instead of discrete flow steps, CNFs parameterize the transformation as an ODE.

```python
# Neural ODE approach:
# dz/dt = f_theta(z(t), t)
# z(0) ~ base distribution
# z(1) = transformed sample
#
# Log-probability:
# log p(z(1)) = log p(z(0)) - integral_0^1 tr(df/dz) dt
#
# Implementation via torchdiffeq or Pyro's experimental CNF support

# Advantages:
# - Arbitrary architecture for f_theta (no invertibility constraint)
# - Memory efficient (adjoint method)
# - Continuous interpolation between base and target

# Disadvantages:
# - Slow (requires ODE solver at each step)
# - Trace estimation for Jacobian
```

---

## 7. Applications

### 7.1 Density Estimation

```python
# Evaluate learned density on a grid
with torch.no_grad():
    xx, yy = torch.meshgrid(torch.linspace(-3, 3, 100), torch.linspace(-2, 3, 100))
    grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    log_probs = flow_model.log_prob(grid)
    probs = torch.exp(log_probs).reshape(100, 100)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.scatter(target_data[:, 0], target_data[:, 1], alpha=0.1, s=1)
ax1.set_title("Target distribution")
ax2.contourf(xx.numpy(), yy.numpy(), probs.numpy(), levels=30, cmap='viridis')
ax2.set_title("Learned density (RealNVP)")
plt.tight_layout()
plt.savefig("flow_density.png", dpi=100)
plt.show()
```

### 7.2 Anomaly Detection

```python
def flow_anomaly_detection(flow, data, threshold_percentile=5):
    """Detect anomalies using flow-based density estimation."""
    with torch.no_grad():
        log_probs = flow.log_prob(data)
    threshold = np.percentile(log_probs.numpy(), threshold_percentile)
    anomalies = log_probs < threshold
    return anomalies, log_probs
```

---

## Summary

| Flow Type | Expressiveness | Speed | Invertibility |
|-----------|---------------|-------|--------------|
| Planar | Low | Fast | Approximate |
| RealNVP | Medium | Fast | Exact |
| MAF | High | Slow generation | Exact |
| IAF | High | Fast generation | Exact |
| Neural Spline | Very high | Moderate | Exact |
| CNF | Unlimited | Slow | Exact |

| Use Case | Recommended Flow |
|----------|-----------------|
| Variational posterior | Spline autoregressive |
| Fast generation | IAF or RealNVP |
| Density estimation | MAF or Neural Spline |
| Image generation | Glow (multi-scale RealNVP) |

---

## References

1. Rezende, D. & Mohamed, S. (2015). "Variational Inference with Normalizing Flows." *ICML*.
2. Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2017). "Density Estimation Using Real-NVP." *ICLR*.
3. Durkan, C., et al. (2019). "Neural Spline Flows." *NeurIPS*.
4. Papamakarios, G., et al. (2021). "Normalizing Flows for Probabilistic Modeling and Inference." *JMLR*.

---

[Previous: Pyro and NumPyro](./12_Pyro_and_NumPyro.md) | [Next: Bayesian Deep Learning →](./14_Bayesian_Deep_Learning.md)
