# 14. 베이지안 딥러닝(Bayesian Deep Learning)

**주제**: Probabilistic_Programming
**레슨**: 19개 중 14번째

[이전: 정규화 플로우](./13_Normalizing_Flows.md) | [다음: 인과 추론](./15_Causal_Inference.md)

---

> **프레임워크 참고**: 이 레슨에서는 PyTorch와 Pyro를 사용한 베이지안 신경망(Bayesian neural network) 구현을 다룹니다.
>
> 설치: `pip install torch pyro-ppl numpy matplotlib`

## 학습 목표(Learning Objectives)

- 딥러닝에서 불확실성(uncertainty)이 왜 중요한지 이해
- 근사 베이지안 추론으로서 MC 드롭아웃(MC Dropout) 구현
- Bayes by Backprop으로 베이지안 신경망(Bayesian Neural Network) 구축
- 불확실성을 우연적(aleatoric)과 인식론적(epistemic) 구성 요소로 분해
- 불확실성 추정이 필요한 실전 과제에 베이지안 딥러닝(BDL) 적용

---

## 1. 딥러닝에서의 불확실성의 필요성(The Need for Uncertainty in Deep Learning)

표준 신경망은 불확실성 없이 점 예측만 출력합니다. 이는 안전이 중요한 응용 분야에서 위험할 수 있습니다.

### 1.1 불확실성의 유형(Types of Uncertainty)

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

두 가지 유형의 불확실성이 있습니다:
- **인식론적 불확실성(Epistemic uncertainty)**: 모델 불확실성으로, 더 많은 데이터로 줄일 수 있습니다. "이런 예를 충분히 보지 못해서 모르겠다"는 의미입니다. 학습 데이터에서 먼 영역에서 높습니다.
- **우연적 불확실성(Aleatoric uncertainty)**: 데이터 불확실성으로, 줄일 수 없습니다. "데이터 자체가 노이즈/모호하다"는 의미입니다. 무한한 학습 데이터가 있어도 높습니다.

---

## 2. MC 드롭아웃(MC Dropout)

테스트 시간에 드롭아웃을 적용하면 네트워크 가중치에 대한 베이지안 사후분포를 근사합니다 (Gal & Ghahramani, 2016).

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

MC 드롭아웃(MC Dropout)의 핵심은 테스트 시에도 드롭아웃을 활성화하여 여러 번 순방향 전파를 수행하는 것입니다. 예측들의 분산이 인식론적 불확실성을 나타내며, 학습 데이터 범위 밖에서 불확실성이 증가하는 것을 확인할 수 있습니다.

---

## 3. Bayes by Backprop

신경망의 가중치 불확실성 (Blundell et al., 2015). 각 가중치가 평균과 분산을 가지며, 변분 자유 에너지를 최소화하여 학습됩니다.

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

Bayes by Backprop은 각 가중치를 점 추정이 아닌 확률 분포(평균 mu와 표준편차 sigma)로 모델링합니다. 재매개변수화 트릭(reparameterization trick)을 사용하여 역전파를 통해 변분 매개변수를 학습하며, 손실 함수는 음의 로그 우도(NLL)와 KL 발산의 합입니다.

---

## 4. 불확실성 분해(Uncertainty Decomposition)

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

불확실성 분해의 핵심 원리:
- **인식론적 불확실성**: 여러 순방향 전파에서 평균값들의 분산으로 측정됩니다. 학습 데이터 밖 영역에서 높습니다.
- **우연적 불확실성**: 예측된 분산들의 평균으로 측정됩니다. 데이터 노이즈가 큰 영역에서 높습니다.
- **총 불확실성**: 인식론적 + 우연적 불확실성의 합입니다.

이질적 분산(heteroscedastic) 모델은 평균과 분산을 모두 출력하여 데이터 의존적 노이즈를 포착합니다.

---

## 5. Pyro를 사용한 BNN(BNN with Pyro)

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

Pyro의 `PyroModule`과 `PyroSample`을 사용하면 기존 PyTorch 모듈을 선언적으로 베이지안 모델로 변환할 수 있습니다. 각 가중치에 사전분포를 지정하고 NUTS MCMC 또는 SVI로 사후분포를 추론합니다.

---

## 6. 실용적 고려사항(Practical Considerations)

### 6.1 BDL을 사용해야 할 때(When to Use BDL)

| 응용 분야 | BDL을 사용하는 이유 | 방법 |
|-------------|---------|--------|
| 의료 진단 | 불확실한 예측에 플래그 필요 | MC 드롭아웃 / 딥 앙상블 |
| 자율 주행 | 안전이 중요한 의사결정 | 이질적 분산 BNN |
| 능동 학습(Active learning) | 가장 정보량이 많은 샘플 선택 | 인식론적 불확실성 |
| 분포 외 탐지(OOD detection) | 미관측 입력 플래그 | 인식론적 불확실성 |
| 교정된 예측(Calibrated forecasting) | 신뢰할 수 있는 신뢰 구간 | 모든 BDL 방법 |

### 6.2 딥 앙상블(Deep Ensembles)

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

딥 앙상블(Deep Ensemble)은 엄밀히 베이지안 방법은 아니지만 실전에서 최고의 불확실성 교정을 보여줍니다. 여러 모델을 서로 다른 초기화와 부트스트랩 데이터로 학습하여 예측의 분산을 불확실성으로 사용합니다. 단점은 N배의 계산 및 메모리 비용입니다.

---

## 요약(Summary)

| 방법 | 타입 | 장점 | 단점 |
|--------|------|------|------|
| MC 드롭아웃(MC Dropout) | 근사 BNN | 단순, 아키텍처 변경 없음 | 약한 근사 |
| Bayes by Backprop | 가중치 분포 | 원칙적 변분 추론(VI) | 2배 매개변수, 학습 불안정 |
| 딥 앙상블(Deep Ensembles) | 비베이지안 | 최고의 교정, 단순 | N배 계산/메모리 |
| Pyro BNN (MCMC) | 정확 BNN | 표준 | 대규모 네트워크에서 매우 느림 |
| SWAG | 근사 | 낮은 오버헤드 | 가우시안 가정 |

| 불확실성 | 출처 | 감소 가능? | 탐지 방법 |
|-------------|--------|-----------|------------|
| 인식론적(Epistemic) | 제한된 데이터 | 예 (더 많은 데이터) | 샘플 간 예측 분산 |
| 우연적(Aleatoric) | 내재적 노이즈 | 아니오 | 예측된 분산 (이질적 분산) |

---

## 참고 문헌(References)

1. Gal, Y. & Ghahramani, Z. (2016). "Dropout as a Bayesian Approximation." *ICML*.
2. Blundell, C., et al. (2015). "Weight Uncertainty in Neural Networks." *ICML*.
3. Lakshminarayanan, B., et al. (2017). "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." *NeurIPS*.
4. Kendall, A. & Gal, Y. (2017). "What Uncertainties Do We Need in Bayesian Deep Learning?" *NeurIPS*.

---

[이전: 정규화 플로우](./13_Normalizing_Flows.md) | [다음: 인과 추론 →](./15_Causal_Inference.md)
