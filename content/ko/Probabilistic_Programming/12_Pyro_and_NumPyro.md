# 12. Pyro와 NumPyro(Pyro and NumPyro)

**주제**: Probabilistic_Programming
**레슨**: 19개 중 12번째

[이전: 베이지안 최적화](./11_Bayesian_Optimization.md) | [다음: 정규화 플로우](./13_Normalizing_Flows.md)

---

> **프레임워크 참고**: 이 레슨에서는 Pyro(PyTorch 백엔드)와 NumPyro(JAX 백엔드)를 사용합니다.
>
> 설치: `pip install pyro-ppl torch numpyro jax jaxlib`

## 학습 목표(Learning Objectives)

- Pyro의 모델/가이드(model/guide) 패러다임 이해
- `pyro.sample`과 이펙트 핸들러(effect handler)를 사용한 모델 작성
- Pyro에서 확률적 변분 추론(Stochastic Variational Inference, SVI) 구현
- JAX 가속 MCMC와 SVI를 위한 NumPyro 사용
- 다양한 사용 사례에서 Pyro/NumPyro와 PyMC 비교

---

## 1. Pyro 기초(Pyro Fundamentals)

Pyro는 PyTorch 위에 구축된 딥 확률적 프로그래밍 언어입니다. 핵심 혁신은 신경망과 확률적 추론을 결합한 것입니다.

### 1.1 모델과 가이드(Model and Guide)

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

Pyro 모델은 `pyro.sample` 문이 포함된 일반 Python 함수입니다. 가이드(guide)는 SVI를 위한 변분 근사(variational approximation)로, 모델의 잠재 변수와 동일한 이름을 가진 `pyro.sample` 문을 포함해야 합니다.

### 1.2 확률적 변분 추론(Stochastic Variational Inference)

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

SVI는 ELBO(Evidence Lower Bound)를 확률적 경사 하강법으로 최대화하여 변분 매개변수를 학습합니다. 위 예제에서 10번의 동전 던지기 중 7번 앞면이 나온 데이터에 대해 사후분포를 학습하며, 정확한 사후분포 Beta(8, 4)에 수렴합니다.

---

## 2. Pyro 이펙트 핸들러(Pyro Effect Handlers)

이펙트 핸들러는 모델 코드를 수정하지 않고 모델 동작을 변환하는 Pyro의 메커니즘입니다.

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

주요 이펙트 핸들러는 다음과 같습니다:
- **trace**: 모든 샘플 사이트를 기록합니다
- **condition**: 잠재 변수를 특정 값으로 고정합니다
- **replay**: 한 실행의 선택을 다른 실행에서 재현합니다

---

## 3. Pyro에서의 베이지안 선형 회귀(Bayesian Linear Regression in Pyro)

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

평균장(mean-field) 변분 가이드는 각 잠재 변수를 독립적인 분포로 근사합니다. 가중치 `w`, 편향 `b`, 노이즈 표준편차 `sigma` 각각에 대해 위치(loc)와 스케일(scale) 매개변수를 학습합니다.

---

## 4. AutoGuide: 자동 변분 패밀리(AutoGuide: Automatic Variational Families)

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

AutoGuide는 가이드를 수동으로 작성하는 대신 모델의 잠재 변수를 자동으로 분석하여 변분 패밀리를 구성합니다. `AutoDiagonalNormal`은 모든 잠재 변수에 대해 독립 정규 분포를, `AutoMultivariateNormal`은 상관관계가 있는 다변량 정규 분포를 사용합니다.

---

## 5. Pyro에서의 MCMC(MCMC in Pyro)

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

Pyro는 SVI뿐만 아니라 Stan이나 PyMC와 유사한 NUTS(No-U-Turn Sampler) 기반 MCMC도 지원합니다. 정확한 사후분포 샘플이 필요한 경우 유용하지만, SVI보다 느릴 수 있습니다.

---

## 6. NumPyro: JAX 가속 추론(NumPyro: JAX-Accelerated Inference)

NumPyro는 Pyro의 JAX 기반 형제 라이브러리입니다. JAX가 모델을 XLA로 컴파일하기 때문에 MCMC에서 상당히 빠릅니다.

### 6.1 NumPyro 모델(NumPyro Model)

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

NumPyro는 Pyro와 거의 동일한 API를 제공하면서도 JAX의 JIT 컴파일과 자동 벡터화를 활용하여 훨씬 빠른 추론을 수행합니다. 특히 MCMC에서 그 차이가 두드러집니다.

---

## 7. 딥 확률 모델을 위한 Pyro(Pyro for Deep Probabilistic Models)

### 7.1 베이지안 신경망(Bayesian Neural Network)

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

Pyro의 `random_module`은 기존 PyTorch 모듈의 모든 매개변수에 사전분포를 배치하여 베이지안 신경망으로 변환합니다. 이를 통해 딥러닝과 확률적 프로그래밍을 자연스럽게 결합할 수 있습니다.

---

## 8. Pyro vs PyMC vs Stan 비교(Pyro vs PyMC vs Stan)

| 특징 | Pyro/NumPyro | PyMC | Stan |
|---------|-------------|------|------|
| 백엔드 | PyTorch/JAX | PyTensor | C++ |
| 딥러닝 | 네이티브 | 제한적 | 없음 |
| SVI | 예 (주요 방법) | ADVI만 | ADVI |
| MCMC | NUTS | NUTS | NUTS (기준 구현) |
| 미니배치 | 용이 | pm.Minibatch | 불가 |
| 이산 잠재 변수 | 열거(enumeration) | Metropolis | 주변화(marginalize) |
| GPU | 예 | 제한적 | 아니오 |
| 적합한 경우 | 딥+확률 모델 | 전통적 베이즈 | 표준 MCMC |

---

## 요약(Summary)

| 개념 | 핵심 요점 |
|---------|-------------|
| Pyro 모델 | `pyro.sample` 문이 있는 Python 함수 |
| 가이드(Guide) | 모델을 미러링하는 변분 근사 |
| SVI | 미니배치 확률적 기울기 ELBO를 통한 확장 가능한 추론 |
| AutoGuide | 자동 변분 패밀리 구성 |
| 이펙트 핸들러 | trace, condition, replay로 모델 실행 변환 |
| NumPyro | 빠른 컴파일된 MCMC를 위한 JAX 백엔드 |
| 사용 사례 | 딥 확률 모델, 대규모 추론 |

---

## 참고 문헌(References)

1. Bingham, E., et al. (2019). "Pyro: Deep Universal Probabilistic Programming." *JMLR*, 20(28), 1-6.
2. Phan, D., Pradhan, N., & Jankowiak, M. (2019). "Composable Effects for Flexible and Accelerated Probabilistic Programming in NumPyro." arXiv:1912.11554.
3. Pyro documentation: https://pyro.ai/
4. NumPyro documentation: https://num.pyro.ai/

---

[이전: 베이지안 최적화](./11_Bayesian_Optimization.md) | [다음: 정규화 플로우 →](./13_Normalizing_Flows.md)
