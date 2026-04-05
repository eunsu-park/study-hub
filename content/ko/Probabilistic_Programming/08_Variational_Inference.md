# 08. 변분 추론(Variational Inference)

[이전: Stan과 CmdStanPy](./07_Stan_and_CmdStanPy.md) | [다음: 가우시안 프로세스](./09_Gaussian_Processes.md)

---

> **프레임워크 참고**: 이 레슨은 NumPy를 사용한 VI 이론 구현과 PyMC에서의 실전 사용을 다룹니다.
>
> 설치: `pip install pymc arviz numpy scipy matplotlib`

## 학습 목표

- 증거 하한(ELBO)과 KL 발산 이해
- 평균장 변분 추론을 처음부터 구현
- PyMC에서 자동 미분 변분 추론(ADVI) 사용
- VI와 MCMC의 속도/정확도 트레이드오프 비교
- VI가 MCMC보다 적합한 경우 이해

---

## 1. 변분 추론의 핵심 아이디어(The Variational Inference Idea)

사후분포에서 샘플링(MCMC)하는 대신, 최적화를 통해 더 단순한 분포로 **근사**합니다.

### 1.1 적분에서 최적화로(From Integration to Optimization)

MCMC: $P(\theta | D)$에서 직접 샘플을 추출합니다.
VI: 다루기 쉬운 분포군 $\mathcal{Q}$에서 $P(\theta | D)$에 가장 가까운 분포 $q(\theta)$를 찾습니다.

$$q^*(\theta) = \arg\min_{q \in \mathcal{Q}} \text{KL}(q(\theta) \| P(\theta | D))$$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize

# Visualize: approximate a bimodal posterior with a Gaussian
theta = np.linspace(-6, 8, 1000)
true_posterior = 0.4 * stats.norm.pdf(theta, -1, 0.8) + 0.6 * stats.norm.pdf(theta, 3, 1.2)

# Best Gaussian approximation (mean-field VI)
def kl_divergence_approx(params):
    """Approximate KL(q || p) using Monte Carlo."""
    mu, log_sigma = params
    sigma = np.exp(log_sigma)
    samples = np.random.normal(mu, sigma, 10000)
    log_q = stats.norm.logpdf(samples, mu, sigma)
    log_p = np.log(0.4 * stats.norm.pdf(samples, -1, 0.8) +
                   0.6 * stats.norm.pdf(samples, 3, 1.2) + 1e-300)
    return np.mean(log_q - log_p)

result = minimize(kl_divergence_approx, [1.0, 0.5], method='Nelder-Mead')
vi_mu, vi_sigma = result.x[0], np.exp(result.x[1])
vi_approx = stats.norm.pdf(theta, vi_mu, vi_sigma)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(theta, true_posterior, 'b-', linewidth=2, label='True posterior')
ax.plot(theta, vi_approx, 'r--', linewidth=2, label=f'VI approx N({vi_mu:.1f}, {vi_sigma:.1f}²)')
ax.fill_between(theta, true_posterior, alpha=0.2, color='blue')
ax.fill_between(theta, vi_approx, alpha=0.2, color='red')
ax.legend(fontsize=12)
ax.set_title("Variational Inference: Gaussian Approximation to Bimodal Posterior")
plt.tight_layout()
plt.savefig("vi_approximation.png", dpi=100)
plt.show()
```

---

## 2. KL 발산과 ELBO(KL Divergence and ELBO)

### 2.1 KL 발산(KL Divergence)

$$\text{KL}(q \| p) = \int q(\theta) \log \frac{q(\theta)}{P(\theta | D)} d\theta$$

KL 발산의 성질:
- 비음수: KL(q || p) >= 0
- q = p일 때만 0
- **비대칭**: KL(q || p) != KL(p || q)

### 2.2 증거 하한(Evidence Lower Bound, ELBO)

KL(q || 사후분포)를 직접 계산할 수 없으므로(증거 P(D)가 필요), 대신 ELBO를 최대화합니다:

$$\text{ELBO}(q) = \mathbb{E}_q[\log P(D, \theta)] - \mathbb{E}_q[\log q(\theta)]$$
$$= \mathbb{E}_q[\log P(D | \theta)] - \text{KL}(q(\theta) \| P(\theta))$$

ELBO 최대화 = KL(q || 사후분포) 최소화.

```python
def compute_elbo(q_samples, log_joint_fn, q_log_prob_fn, n_samples=1000):
    """Estimate ELBO via Monte Carlo."""
    log_joint = np.array([log_joint_fn(s) for s in q_samples[:n_samples]])
    log_q = np.array([q_log_prob_fn(s) for s in q_samples[:n_samples]])
    elbo = np.mean(log_joint - log_q)
    return elbo
```

### 2.3 ELBO 분해(ELBO Decomposition)

```python
def elbo_decomposition(q_samples, log_likelihood_fn, log_prior_fn, q_log_prob_fn):
    """Decompose ELBO into expected log-likelihood and KL(q||prior)."""
    expected_ll = np.mean([log_likelihood_fn(s) for s in q_samples])
    kl_prior = np.mean([q_log_prob_fn(s) - log_prior_fn(s) for s in q_samples])

    print(f"E_q[log P(D|θ)] = {expected_ll:.2f} (data fit)")
    print(f"KL(q || prior)  = {kl_prior:.2f} (complexity penalty)")
    print(f"ELBO            = {expected_ll - kl_prior:.2f}")
    return expected_ll, kl_prior
```

---

## 3. 평균장 변분 추론(Mean-Field Variational Inference)

가장 단순한 VI 접근법: 변분 분포에서 모든 매개변수가 독립이라고 가정합니다.

$$q(\theta_1, \ldots, \theta_d) = \prod_{j=1}^d q_j(\theta_j)$$

### 3.1 처음부터 구현(Implementation from Scratch)

```python
class MeanFieldVI:
    """Mean-field Gaussian variational inference via gradient ascent on ELBO."""

    def __init__(self, log_joint_fn, n_params, learning_rate=0.01):
        self.log_joint = log_joint_fn
        self.d = n_params
        self.lr = learning_rate

        # Variational parameters: mean and log(std) for each parameter
        self.mu = np.zeros(n_params)
        self.log_sigma = np.zeros(n_params)

    def sample(self, n_samples=100):
        """Sample from the variational distribution q(theta)."""
        sigma = np.exp(self.log_sigma)
        epsilon = np.random.normal(size=(n_samples, self.d))
        return self.mu + sigma * epsilon  # reparameterization trick

    def log_q(self, theta):
        """Log probability under variational distribution."""
        sigma = np.exp(self.log_sigma)
        return np.sum(stats.norm.logpdf(theta, self.mu, sigma))

    def estimate_elbo(self, n_samples=100):
        """Monte Carlo estimate of ELBO."""
        samples = self.sample(n_samples)
        elbo = np.mean([
            self.log_joint(s) - self.log_q(s)
            for s in samples
        ])
        return elbo

    def fit(self, n_steps=1000, n_samples=50, verbose=True):
        """Optimize ELBO using stochastic gradient ascent."""
        elbo_history = []

        for step in range(n_steps):
            sigma = np.exp(self.log_sigma)
            epsilon = np.random.normal(size=(n_samples, self.d))
            samples = self.mu + sigma * epsilon

            # Estimate gradients via score function estimator
            log_joints = np.array([self.log_joint(s) for s in samples])
            log_qs = np.array([self.log_q(s) for s in samples])
            advantages = log_joints - log_qs

            # Gradient w.r.t. mu
            grad_mu = np.mean(
                advantages[:, None] * epsilon / sigma, axis=0
            )

            # Gradient w.r.t. log_sigma
            grad_log_sigma = np.mean(
                advantages[:, None] * (epsilon**2 - 1), axis=0
            )

            # Update
            self.mu += self.lr * grad_mu
            self.log_sigma += self.lr * 0.1 * grad_log_sigma

            elbo = np.mean(advantages)
            elbo_history.append(elbo)

            if verbose and step % 100 == 0:
                print(f"Step {step}: ELBO = {elbo:.3f}, "
                      f"mu = {self.mu.round(3)}, sigma = {np.exp(self.log_sigma).round(3)}")

        return elbo_history


# Example: Bayesian linear regression
np.random.seed(42)
n = 50
x = np.random.randn(n)
true_w, true_b = 2.5, -1.0
y = true_w * x + true_b + np.random.normal(0, 0.5, n)

def log_joint(params):
    w, b, log_s = params
    sigma = np.exp(log_s)
    log_prior = stats.norm.logpdf(w, 0, 5) + stats.norm.logpdf(b, 0, 5)
    log_prior += stats.halfnorm.logpdf(sigma, scale=5) + log_s  # Jacobian
    log_lik = np.sum(stats.norm.logpdf(y, w * x + b, sigma))
    return log_prior + log_lik

vi = MeanFieldVI(log_joint, n_params=3, learning_rate=0.005)
elbo_history = vi.fit(n_steps=2000, n_samples=100)

print(f"\nVI results: w={vi.mu[0]:.3f}, b={vi.mu[1]:.3f}, sigma={np.exp(vi.mu[2]):.3f}")
print(f"True values: w={true_w}, b={true_b}, sigma=0.5")

plt.figure(figsize=(10, 4))
plt.plot(elbo_history)
plt.xlabel("Step")
plt.ylabel("ELBO")
plt.title("ELBO Convergence")
plt.savefig("elbo_convergence.png", dpi=100)
plt.show()
```

---

## 4. 재매개변수화 트릭(The Reparameterization Trick)

기울기 기반 VI를 가능하게 하는 핵심 통찰: $\theta \sim q_\phi(\theta)$에서 직접 샘플링하는 대신, $\theta = g(\epsilon, \phi)$로 재매개변수화합니다. 여기서 $\epsilon \sim p(\epsilon)$입니다.

```python
# Without reparameterization:
# theta ~ Normal(mu, sigma)
# gradient of E_q[f(theta)] w.r.t. mu requires score function estimator (high variance)

# With reparameterization:
# epsilon ~ Normal(0, 1)
# theta = mu + sigma * epsilon
# gradient of E[f(mu + sigma * epsilon)] w.r.t. mu is simply E[f'(mu + sigma * epsilon)]
# (low variance, enables backpropagation)

def reparameterized_elbo_gradient(log_joint, mu, log_sigma, n_samples=100):
    """Compute ELBO gradient using the reparameterization trick."""
    sigma = np.exp(log_sigma)
    d = len(mu)
    epsilon = np.random.normal(size=(n_samples, d))
    theta = mu + sigma * epsilon

    # Numerical gradients of log_joint w.r.t. theta
    h = 1e-5
    grad_mu = np.zeros(d)
    grad_log_sigma = np.zeros(d)

    for j in range(d):
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[:, j] += h
        theta_minus[:, j] -= h

        dlj = np.array([(log_joint(tp) - log_joint(tm)) / (2*h)
                        for tp, tm in zip(theta_plus, theta_minus)])

        grad_mu[j] = np.mean(dlj)
        grad_log_sigma[j] = np.mean(dlj * epsilon[:, j] * sigma[j])

    # Entropy gradient: d/d(log_sigma) [log(sigma)] = 1
    grad_log_sigma += 1.0  # entropy term

    return grad_mu, grad_log_sigma
```

---

## 5. PyMC에서의 ADVI(ADVI in PyMC)

자동 미분 변분 추론(ADVI)은 평균장 VI를 자동화합니다:
1. 모든 매개변수를 비제약 공간으로 변환
2. 변환된 공간에서 평균장 가우시안을 적합
3. 변환을 역변환하여 사후분포 근사를 얻음

### 5.1 ADVI 실행(Running ADVI)

```python
import pymc as pm
import arviz as az

np.random.seed(42)
data = np.random.normal(5.0, 2.0, 100)

with pm.Model() as vi_model:
    mu = pm.Normal("mu", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=5)
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=data)

    # ADVI (mean-field)
    approx = pm.fit(
        n=30000,              # optimization steps
        method="advi",         # mean-field ADVI
        random_seed=42,
    )

# Plot convergence (ELBO)
plt.figure(figsize=(10, 4))
plt.plot(approx.hist)
plt.xlabel("Iteration")
plt.ylabel("-ELBO")
plt.title("ADVI Convergence")
plt.savefig("advi_convergence.png", dpi=100)
plt.show()

# Draw samples from the approximate posterior
vi_trace = approx.sample(5000)
print(az.summary(vi_trace, var_names=["mu", "sigma"]))
```

### 5.2 풀랭크 ADVI(Full-Rank ADVI)

```python
with vi_model:
    # Full-rank ADVI (captures correlations between parameters)
    approx_fullrank = pm.fit(
        n=30000,
        method="fullrank_advi",
        random_seed=42,
    )

    vi_trace_fr = approx_fullrank.sample(5000)
    print("Full-rank ADVI:")
    print(az.summary(vi_trace_fr, var_names=["mu", "sigma"]))
```

---

## 6. VI vs MCMC 비교(VI vs MCMC Comparison)

```python
# Compare VI and MCMC on the same model
with vi_model:
    mcmc_trace = pm.sample(5000, tune=1000, chains=4, random_seed=42)

# Side-by-side comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for i, var in enumerate(["mu", "sigma"]):
    vi_samples = vi_trace.posterior[var].values.flatten()
    mcmc_samples = mcmc_trace.posterior[var].values.flatten()

    axes[i].hist(mcmc_samples, bins=50, density=True, alpha=0.5, label="MCMC (NUTS)")
    axes[i].hist(vi_samples, bins=50, density=True, alpha=0.5, label="VI (ADVI)")
    axes[i].set_title(var)
    axes[i].legend()

plt.suptitle("VI vs MCMC Posterior Comparison")
plt.tight_layout()
plt.savefig("vi_vs_mcmc.png", dpi=100)
plt.show()

# Quantitative comparison
for var in ["mu", "sigma"]:
    vi_mean = vi_trace.posterior[var].values.mean()
    mcmc_mean = mcmc_trace.posterior[var].values.mean()
    vi_std = vi_trace.posterior[var].values.std()
    mcmc_std = mcmc_trace.posterior[var].values.std()
    print(f"{var}: VI mean={vi_mean:.3f}±{vi_std:.3f}, MCMC mean={mcmc_mean:.3f}±{mcmc_std:.3f}")
```

### 6.1 VI vs MCMC 사용 시점(When to Use VI vs MCMC)

| 기준 | VI | MCMC |
|------|-----|------|
| 속도 | 빠름 (분) | 느림 (시간) |
| 정확도 | 근사적 | 정확 (점근적으로) |
| 다봉 사후분포 | 모드를 놓칠 수 있음 | 모드 탐색 가능 (믹싱 시) |
| 사후 상관관계 | 평균장은 무시 | 완전히 포착 |
| 대규모 데이터 | 미니배치로 잘 확장 | 반복당 비용 높음 |
| 모델 개발 | 빠른 반복에 적합 | 최종 결과의 표준 |
| 수렴 | 평가 용이 (ELBO) | 여러 진단 필요 |

---

## 7. 확률적 VI와 미니배칭(Stochastic VI and Mini-Batching)

대규모 데이터셋의 경우, 미니배치를 사용하여 VI를 확장할 수 있습니다.

```python
# Stochastic ADVI with minibatch
np.random.seed(42)
large_data = np.random.normal(3.0, 1.5, 10000)

with pm.Model() as minibatch_model:
    mu = pm.Normal("mu", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=5)

    # Create minibatch shared variable
    data_minibatch = pm.Minibatch(large_data, batch_size=200)

    # Likelihood with minibatch
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=data_minibatch,
                  total_size=len(large_data))

    approx_mini = pm.fit(n=20000, method="advi", random_seed=42)

mini_trace = approx_mini.sample(5000)
print(az.summary(mini_trace, var_names=["mu", "sigma"]))
```

---

## 8. VI를 위한 정규화 흐름 미리보기(Normalizing Flows for VI - Preview)

평균장 VI는 단순한 분포로 제한됩니다. 정규화 흐름(레슨 13)은 유연한 변분군을 만들 수 있습니다.

```python
# Conceptual overview:
# Instead of q(theta) = Normal(mu, sigma^2),
# we use q(theta) = T_K ∘ T_{K-1} ∘ ... ∘ T_1 (z), where z ~ Normal(0, I)
#
# Each T_k is an invertible, differentiable transform
# This can approximate complex, multimodal posteriors
#
# ELBO with flow:
# ELBO = E_z[log p(T(z), D)] + E_z[sum_k log|det J_k|] + H[q_0]
```

---

## 9. 상각 추론 미리보기(Amortized Inference - Preview)

상각 추론에서는 관측값을 근사 사후분포 매개변수로 직접 매핑하는 신경망을 훈련합니다.

```python
# Concept: encoder network
# Given data x, output variational parameters (mu, sigma)
# phi(x) → (mu_q, sigma_q)
# q(theta | x; phi) = Normal(mu_q(x), sigma_q(x))
#
# This is the inference network used in Variational Autoencoders (VAEs)
# and in Pyro's SVI (Lesson 12)
```

---

## 요약(Summary)

| 개념 | 핵심 요점 |
|------|----------|
| ELBO | 증거 하한; ELBO 최대화 ≈ KL 발산 최소화 |
| 평균장 VI | 독립 매개변수 가정; 빠르지만 상관관계 놓침 |
| 풀랭크 ADVI | 매개변수 상관관계 포착; 더 정확 |
| 재매개변수화 트릭 | 확률적 목적함수의 기울기 기반 최적화 가능하게 함 |
| 미니배치 VI | 확률적 기울기로 대규모 데이터셋에 확장 |
| VI vs MCMC | VI: 빠르고 근사적; MCMC: 느리고 정확 |
| ADVI | PyMC의 자동 VI; 비제약 공간으로 변환 |

---

## 참고문헌(References)

1. Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). "Variational Inference: A Review for Statisticians." *JASA*, 112(518), 859-877.
2. Kucukelbir, A., et al. (2017). "Automatic Differentiation Variational Inference." *JMLR*, 18(1), 430-474.
3. Kingma, D. P. & Welling, M. (2014). "Auto-Encoding Variational Bayes." arXiv:1312.6114.
4. Zhang, C., et al. (2018). "Advances in Variational Inference." *IEEE TPAMI*.

---

[이전: Stan과 CmdStanPy](./07_Stan_and_CmdStanPy.md) | [다음: 가우시안 프로세스 →](./09_Gaussian_Processes.md)
