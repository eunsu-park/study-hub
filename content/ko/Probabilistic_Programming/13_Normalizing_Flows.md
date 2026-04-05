# 13. 정규화 플로우(Normalizing Flows)

**주제**: Probabilistic_Programming
**레슨**: 19개 중 13번째

[이전: Pyro와 NumPyro](./12_Pyro_and_NumPyro.md) | [다음: 베이지안 딥러닝](./14_Bayesian_Deep_Learning.md)

---

> **프레임워크 참고**: 이 레슨에서는 PyTorch로 플로우를 구현하고 Pyro로 PPL과 통합합니다.
>
> 설치: `pip install torch pyro-ppl numpy matplotlib`

## 학습 목표(Learning Objectives)

- 단순 분포의 가역 변환으로서 정규화 플로우(normalizing flows) 이해
- 평면 플로우(planar flow), RealNVP, 신경 스플라인 플로우(Neural Spline Flow) 구현
- 유연한 변분 사후분포(variational posterior)로서 플로우 사용
- 밀도 추정(density estimation)과 생성 모델링(generative modeling)에 플로우 적용
- 확률적 프로그래밍 프레임워크와 플로우 통합

---

## 1. 정규화 플로우의 아이디어(The Normalizing Flow Idea)

정규화 플로우는 단순한 기저 분포(예: 표준 가우시안)를 일련의 가역적이고 미분 가능한 변환을 통해 복잡한 목표 분포로 변환합니다.

### 1.1 변수 변환(Change of Variables)

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

핵심은 변수 변환 공식입니다. 변환된 분포의 로그 확률을 계산하려면 기저 분포의 로그 확률에서 야코비안 행렬식의 로그 절대값을 빼야 합니다. 효율적인 플로우 설계의 핵심은 이 야코비안 행렬식을 빠르게 계산할 수 있도록 변환을 구성하는 것입니다.

---

## 2. 평면 플로우(Planar Flows)

가장 단순한 비자명 플로우입니다. 각 레이어는 다음을 적용합니다:

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

평면 플로우는 간단하지만 표현력이 제한적입니다. 각 레이어는 하이퍼플레인 주변에서 분포를 "접는" 효과를 가지며, 여러 레이어를 쌓아 더 복잡한 변환을 달성합니다.

---

## 3. RealNVP(Real-valued Non-Volume Preserving)

RealNVP는 아핀 결합 레이어(affine coupling layer)를 사용하며, 역변환이 용이하고 야코비안이 다루기 쉽습니다.

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

RealNVP의 핵심 아이디어는 입력 차원을 두 그룹으로 나누어, 한 그룹은 그대로 유지하고 다른 그룹을 첫 번째 그룹에 의존하는 아핀 변환으로 변환하는 것입니다. 마스크를 번갈아 사용하여 모든 차원이 변환될 수 있도록 합니다.

### 3.1 밀도 추정을 위한 RealNVP 학습(Training RealNVP for Density Estimation)

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

학습은 음의 로그 우도(negative log-likelihood)를 최소화하여 수행됩니다. 학습 후에는 기저 분포에서 샘플링하고 순방향 변환을 적용하여 새로운 샘플을 생성할 수 있습니다.

---

## 4. 신경 스플라인 플로우(Neural Spline Flows)

유리 이차 스플라인(rational-quadratic spline) 변환을 사용하는 최신 플로우 아키텍처입니다.

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

핵심 아이디어는 다음과 같습니다:
- 정의역을 K개의 구간으로 분할합니다
- 각 구간에서 유리 이차 스플라인(너비, 높이, 도함수로 매개변수화)을 사용합니다
- 신경망이 이 스플라인 매개변수를 예측합니다
- 단조 증가가 보장되므로 해석적 야코비안을 가진 가역 변환입니다

---

## 5. 변분 사후분포로서의 플로우(Flows as Variational Posteriors)

확률적 프로그래밍에서 가장 중요한 응용: 플로우를 사용하여 유연한 변분 분포를 생성합니다.

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

전통적인 평균장 변분 추론(variational inference)은 독립 가우시안으로 사후분포를 근사하여 다중 모드 분포를 포착하지 못합니다. 플로우 기반 가이드는 훨씬 더 유연한 분포를 표현할 수 있어 복잡한 사후분포를 더 정확하게 근사합니다.

---

## 6. 연속 정규화 플로우(Continuous Normalizing Flows, CNFs)

이산 플로우 단계 대신 변환을 ODE로 매개변수화합니다.

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

CNF의 핵심은 신경 ODE 접근법입니다:
- **장점**: f_theta에 임의의 아키텍처 사용 가능(가역성 제약 없음), 수반(adjoint) 방법으로 메모리 효율적, 기저와 목표 사이의 연속적 보간
- **단점**: 느림(각 단계에서 ODE 솔버 필요), 야코비안의 트레이스 추정 필요

---

## 7. 응용(Applications)

### 7.1 밀도 추정(Density Estimation)

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

### 7.2 이상 탐지(Anomaly Detection)

```python
def flow_anomaly_detection(flow, data, threshold_percentile=5):
    """Detect anomalies using flow-based density estimation."""
    with torch.no_grad():
        log_probs = flow.log_prob(data)
    threshold = np.percentile(log_probs.numpy(), threshold_percentile)
    anomalies = log_probs < threshold
    return anomalies, log_probs
```

플로우 기반 밀도 추정을 활용하면 데이터의 로그 확률을 계산하여 학습된 밀도가 낮은 데이터 포인트를 이상치로 탐지할 수 있습니다.

---

## 요약(Summary)

| 플로우 타입 | 표현력 | 속도 | 가역성 |
|-----------|---------------|-------|--------------|
| 평면(Planar) | 낮음 | 빠름 | 근사 |
| RealNVP | 중간 | 빠름 | 정확 |
| MAF | 높음 | 느린 생성 | 정확 |
| IAF | 높음 | 빠른 생성 | 정확 |
| 신경 스플라인(Neural Spline) | 매우 높음 | 중간 | 정확 |
| CNF | 무제한 | 느림 | 정확 |

| 사용 사례 | 추천 플로우 |
|----------|-----------------|
| 변분 사후분포 | 스플라인 자기회귀(Spline autoregressive) |
| 빠른 생성 | IAF 또는 RealNVP |
| 밀도 추정 | MAF 또는 신경 스플라인 |
| 이미지 생성 | Glow (다중 스케일 RealNVP) |

---

## 참고 문헌(References)

1. Rezende, D. & Mohamed, S. (2015). "Variational Inference with Normalizing Flows." *ICML*.
2. Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2017). "Density Estimation Using Real-NVP." *ICLR*.
3. Durkan, C., et al. (2019). "Neural Spline Flows." *NeurIPS*.
4. Papamakarios, G., et al. (2021). "Normalizing Flows for Probabilistic Modeling and Inference." *JMLR*.

---

[이전: Pyro와 NumPyro](./12_Pyro_and_NumPyro.md) | [다음: 베이지안 딥러닝 →](./14_Bayesian_Deep_Learning.md)
