# 14. Bayesian Deep Learning

[Previous: Normalizing Flows](./13_Normalizing_Flows.md) | [Next: Causal Inference](./15_Causal_Inference.md)

---

> **Framework Note**: This lesson uses PyTorch and Pyro for Bayesian neural network implementations.
>
> Installation: `pip install torch pyro-ppl numpy matplotlib`

## Learning Objectives

- Understand why uncertainty matters in deep learning
- Implement MC Dropout as approximate Bayesian inference
- Build Bayesian Neural Networks with Bayes by Backprop
- Decompose uncertainty into aleatoric and epistemic components
- Apply BDL to real-world tasks requiring uncertainty estimates

---

## 1. The Need for Uncertainty in Deep Learning

Standard neural networks output point predictions with no uncertainty. This is dangerous for safety-critical applications.

### 1.1 Types of Uncertainty

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Epistemic uncertainty: model uncertainty (reducible with more data)
# - "I don't know because I haven't seen enough examples like this"
# - High in regions far from training data

# Aleatoric uncertainty: data uncertainty (irreducible)
# - "The data itself is noisy/ambiguous"
# - High even with infinite training data

# Example: regression with heteroscedastic noise
np.random.seed(42)
x_train = np.sort(np.random.uniform(-3, 3, 100))
noise_std = 0.1 + 0.3 * np.abs(x_train)  # noise increases with |x|
y_train = np.sin(x_train) + np.random.normal(0, noise_std)

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(x_train, y_train, alpha=0.5, s=20, label='Training data')
ax.plot(x_train, np.sin(x_train), 'r-', label='True function')
ax.set_title("Heteroscedastic Data: Noise Increases with |x|")
ax.legend()
plt.tight_layout()
plt.savefig("heteroscedastic_data.png", dpi=100)
plt.show()
```

---

## 2. MC Dropout

Dropout at test time approximates a Bayesian posterior over network weights (Gal & Ghahramani, 2016).

```python
class MCDropoutNet(nn.Module):
    """Neural network with MC Dropout."""

    def __init__(self, in_dim, hidden_dim, out_dim, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # dropout ALSO at test time
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

    def predict_with_uncertainty(self, x, n_forward=100):
        """Multiple forward passes with dropout for uncertainty."""
        self.train()  # keep dropout active
        predictions = torch.stack([self(x) for _ in range(n_forward)])
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        return mean, std


# Train the MC Dropout model
X_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(-1)
y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)

model = MCDropoutNet(1, 64, 1, dropout_rate=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(1000):
    model.train()
    pred = model(X_tensor)
    loss = loss_fn(pred, y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Predict with uncertainty
x_test = torch.linspace(-5, 5, 200).unsqueeze(-1)
with torch.no_grad():
    mean, std = model.predict_with_uncertainty(x_test, n_forward=200)

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(x_train, y_train, alpha=0.3, s=10, color='gray')
ax.plot(x_test.numpy(), mean.numpy(), 'b-', linewidth=2, label='MC Dropout mean')
ax.fill_between(x_test.squeeze().numpy(),
                (mean - 2*std).squeeze().numpy(),
                (mean + 2*std).squeeze().numpy(),
                alpha=0.2, label='±2σ (epistemic)')
ax.set_title("MC Dropout: Uncertainty Grows Outside Training Data")
ax.legend()
plt.tight_layout()
plt.savefig("mc_dropout.png", dpi=100)
plt.show()
```

---

## 3. Bayes by Backprop

Weight Uncertainty in Neural Networks (Blundell et al., 2015). Each weight has a mean and variance, trained by minimizing the variational free energy.

```python
class BayesLinear(nn.Module):
    """Bayesian linear layer with learnable weight distributions."""

    def __init__(self, in_features, out_features, prior_sigma=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight parameters (variational)
        self.w_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.w_rho = nn.Parameter(torch.full((out_features, in_features), -3.0))
        self.b_mu = nn.Parameter(torch.zeros(out_features))
        self.b_rho = nn.Parameter(torch.full((out_features,), -3.0))

        # Prior
        self.prior_sigma = prior_sigma
        self.kl = 0

    def forward(self, x):
        # Reparameterization trick
        w_sigma = torch.log1p(torch.exp(self.w_rho))  # softplus
        b_sigma = torch.log1p(torch.exp(self.b_rho))

        w = self.w_mu + w_sigma * torch.randn_like(w_sigma)
        b = self.b_mu + b_sigma * torch.randn_like(b_sigma)

        # KL divergence: KL(q(w) || p(w))
        kl_w = self._kl_gaussian(self.w_mu, w_sigma, 0, self.prior_sigma)
        kl_b = self._kl_gaussian(self.b_mu, b_sigma, 0, self.prior_sigma)
        self.kl = kl_w + kl_b

        return nn.functional.linear(x, w, b)

    def _kl_gaussian(self, mu_q, sigma_q, mu_p, sigma_p):
        """KL divergence between two Gaussians."""
        return torch.sum(
            torch.log(sigma_p / sigma_q) +
            (sigma_q**2 + (mu_q - mu_p)**2) / (2 * sigma_p**2) - 0.5
        )


class BayesianNN(nn.Module):
    """Bayesian Neural Network with Bayes by Backprop."""

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = BayesLinear(in_dim, hidden_dim)
        self.fc2 = BayesLinear(hidden_dim, hidden_dim)
        self.fc3 = BayesLinear(hidden_dim, out_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

    @property
    def kl(self):
        return self.fc1.kl + self.fc2.kl + self.fc3.kl


# Training
bnn = BayesianNN(1, 50, 1)
optimizer = torch.optim.Adam(bnn.parameters(), lr=0.005)
n_train = len(x_train)

for epoch in range(2000):
    pred = bnn(X_tensor)
    nll = nn.functional.mse_loss(pred, y_tensor, reduction='sum') / 2
    kl = bnn.kl / n_train  # KL weight: 1/N
    loss = nll + kl

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}: NLL={nll.item():.3f}, KL={kl.item():.3f}")

# Predict
preds = torch.stack([bnn(x_test) for _ in range(200)]).detach()
mean_bnn = preds.mean(dim=0)
std_bnn = preds.std(dim=0)
```

---

## 4. Uncertainty Decomposition

```python
class HeteroscedasticBNN(nn.Module):
    """BNN that outputs both mean and variance (aleatoric + epistemic)."""

    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.log_var_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.shared(x)
        mean = self.mean_head(h)
        log_var = self.log_var_head(h)
        return mean, log_var

    def predict_with_decomposition(self, x, n_forward=200):
        self.train()
        means, log_vars = [], []
        for _ in range(n_forward):
            m, lv = self(x)
            means.append(m)
            log_vars.append(lv)

        means = torch.stack(means)
        log_vars = torch.stack(log_vars)
        vars_alea = torch.exp(log_vars)

        # Epistemic: variance of the means across forward passes
        epistemic = means.var(dim=0)
        # Aleatoric: mean of the predicted variances
        aleatoric = vars_alea.mean(dim=0)
        # Total
        total = epistemic + aleatoric

        return means.mean(dim=0), epistemic, aleatoric, total


het_model = HeteroscedasticBNN(1, 64)
optimizer = torch.optim.Adam(het_model.parameters(), lr=0.005)

for epoch in range(2000):
    het_model.train()
    mean_pred, log_var_pred = het_model(X_tensor)
    # Heteroscedastic Gaussian NLL
    loss = 0.5 * (log_var_pred + (y_tensor - mean_pred)**2 / torch.exp(log_var_pred)).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Decompose uncertainty
with torch.no_grad():
    mean_h, epist, alea, total = het_model.predict_with_decomposition(x_test)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, unc, title in zip(axes,
    [epist.squeeze().numpy(), alea.squeeze().numpy(), total.squeeze().numpy()],
    ["Epistemic (model)", "Aleatoric (data)", "Total"]):
    ax.scatter(x_train, y_train, alpha=0.2, s=5, color='gray')
    ax.plot(x_test.numpy(), mean_h.numpy(), 'b-', linewidth=2)
    ax.fill_between(x_test.squeeze().numpy(),
                    mean_h.squeeze().numpy() - 2*np.sqrt(unc),
                    mean_h.squeeze().numpy() + 2*np.sqrt(unc), alpha=0.3)
    ax.set_title(title)
plt.tight_layout()
plt.savefig("uncertainty_decomposition.png", dpi=100)
plt.show()
```

---

## 5. BNN with Pyro

```python
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import MCMC, NUTS, Predictive

class PyroRegression(PyroModule):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = PyroModule[nn.Linear](in_dim, hidden_dim)
        self.fc1.weight = PyroSample(dist.Normal(0., 1.).expand([hidden_dim, in_dim]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0., 1.).expand([hidden_dim]).to_event(1))
        self.fc2 = PyroModule[nn.Linear](hidden_dim, out_dim)
        self.fc2.weight = PyroSample(dist.Normal(0., 1.).expand([out_dim, hidden_dim]).to_event(2))
        self.fc2.bias = PyroSample(dist.Normal(0., 1.).expand([out_dim]).to_event(1))

    def forward(self, x, y=None):
        x = torch.relu(self.fc1(x))
        mu = self.fc2(x).squeeze(-1)
        sigma = pyro.sample("sigma", dist.HalfNormal(1.0))
        with pyro.plate("data", len(x)):
            obs = pyro.sample("obs", dist.Normal(mu, sigma), obs=y)
        return mu
```

---

## 6. Practical Considerations

### 6.1 When to Use BDL

| Application | Why BDL | Method |
|-------------|---------|--------|
| Medical diagnosis | Need to flag uncertain predictions | MC Dropout / Deep Ensemble |
| Autonomous driving | Safety-critical decisions | Heteroscedastic BNN |
| Active learning | Select most informative samples | Epistemic uncertainty |
| Out-of-distribution detection | Flag unseen inputs | Epistemic uncertainty |
| Calibrated forecasting | Reliable confidence intervals | All BDL methods |

### 6.2 Deep Ensembles

```python
class DeepEnsemble:
    """Ensemble of neural networks for uncertainty estimation."""

    def __init__(self, n_models=5, in_dim=1, hidden_dim=64, out_dim=1):
        self.models = [
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, out_dim)
            ) for _ in range(n_models)
        ]

    def train_all(self, X, y, n_epochs=1000, lr=0.01):
        for i, model in enumerate(self.models):
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            # Bootstrap: train each model on a random subset
            idx = torch.randint(0, len(X), (len(X),))
            for epoch in range(n_epochs):
                pred = model(X[idx])
                loss = nn.functional.mse_loss(pred, y[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def predict(self, X):
        preds = torch.stack([m(X) for m in self.models])
        return preds.mean(dim=0), preds.std(dim=0)
```

---

## Summary

| Method | Type | Pros | Cons |
|--------|------|------|------|
| MC Dropout | Approximate BNN | Simple, no architecture change | Weak approximation |
| Bayes by Backprop | Weight distributions | Principled VI | 2x parameters, training instability |
| Deep Ensembles | Non-Bayesian | Best calibration, simple | N× compute/memory |
| Pyro BNN (MCMC) | Exact BNN | Gold standard | Very slow for large nets |
| SWAG | Approximate | Low overhead | Gaussian assumption |

| Uncertainty | Source | Reducible? | Detect via |
|-------------|--------|-----------|------------|
| Epistemic | Limited data | Yes (more data) | Variance of predictions across samples |
| Aleatoric | Inherent noise | No | Predicted variance (heteroscedastic) |

---

## References

1. Gal, Y. & Ghahramani, Z. (2016). "Dropout as a Bayesian Approximation." *ICML*.
2. Blundell, C., et al. (2015). "Weight Uncertainty in Neural Networks." *ICML*.
3. Lakshminarayanan, B., et al. (2017). "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." *NeurIPS*.
4. Kendall, A. & Gal, Y. (2017). "What Uncertainties Do We Need in Bayesian Deep Learning?" *NeurIPS*.

---

[Previous: Normalizing Flows](./13_Normalizing_Flows.md) | [Next: Causal Inference →](./15_Causal_Inference.md)
