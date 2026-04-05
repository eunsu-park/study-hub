# 03. MCMC 기초(MCMC Fundamentals)

**주제**: Probabilistic_Programming
**레슨**: 19개 중 3번째

[이전: 확률적 그래프 모델](./02_Probabilistic_Graphical_Models.md) | [다음: PyMC 소개](./04_PyMC_Introduction.md)

---

> **프레임워크 참고**: 이 레슨은 NumPy를 사용하여 MCMC를 처음부터 구현합니다.
> 프로덕션급 MCMC는 레슨 04(PyMC)와 07(Stan)에서 다룹니다.
>
> 설치: `pip install numpy scipy matplotlib arviz`

## 학습 목표(Learning Objectives)

- 마르코프 체인 몬테카를로(MCMC) 방법의 동기 이해
- 메트로폴리스-헤이스팅스(Metropolis-Hastings) 알고리즘을 처음부터 구현
- 다중 매개변수 모델을 위한 깁스 샘플링 구현
- 추적 플롯, R-hat, 유효 표본 크기를 사용한 수렴 진단
- 해밀턴 몬테카를로(HMC)의 원리 이해

---

## 1. 몬테카를로 원리(The Monte Carlo Principle)

적분을 해석적으로 계산할 수 없을 때, 샘플로 근사합니다.

### 1.1 샘플링이 필요한 이유(Why We Need Sampling)

베이지안 추론에서는 사후분포 하의 기댓값을 자주 계산해야 합니다:

$$\mathbb{E}[\theta | D] = \int \theta \, P(\theta | D) \, d\theta$$

대부분의 실제 모델에서 이 적분은 닫힌 형태의 해가 없습니다. 몬테카를로 적분이 해법을 제공합니다: $P(\theta | D)$에서 샘플 $\theta^{(1)}, \ldots, \theta^{(S)}$를 추출할 수 있다면:

$$\mathbb{E}[\theta | D] \approx \frac{1}{S} \sum_{s=1}^{S} \theta^{(s)}$$

### 1.2 직접 샘플링 vs MCMC(Direct Sampling vs MCMC)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Direct sampling: easy when the posterior is a known distribution
# Beta(9, 5) posterior from Lesson 01
direct_samples = np.random.beta(9, 5, size=10000)
print(f"Direct sampling mean: {direct_samples.mean():.4f} (exact: {9/14:.4f})")

# But what if the posterior is NOT a standard distribution?
# e.g., P(theta | data) ∝ exp(-theta^4 + 2*theta^2)
# We cannot sample from this directly → we need MCMC
```

---

## 2. 마르코프 체인(Markov Chains)

### 2.1 마르코프 체인이란?(What is a Markov Chain?)

마르코프 체인은 다음 상태가 현재 상태에만 의존하는 확률 변수의 시퀀스입니다:

$$P(X_{t+1} | X_t, X_{t-1}, \ldots, X_0) = P(X_{t+1} | X_t)$$

```python
def simple_markov_chain(transition_matrix, initial_state, n_steps):
    """Simulate a discrete Markov chain."""
    n_states = len(transition_matrix)
    chain = [initial_state]

    for _ in range(n_steps):
        current = chain[-1]
        next_state = np.random.choice(n_states, p=transition_matrix[current])
        chain.append(next_state)

    return np.array(chain)


# Weather model: 0=Sunny, 1=Cloudy, 2=Rainy
T = np.array([
    [0.7, 0.2, 0.1],  # Sunny → ...
    [0.3, 0.4, 0.3],  # Cloudy → ...
    [0.2, 0.3, 0.5],  # Rainy → ...
])

chain = simple_markov_chain(T, initial_state=0, n_steps=10000)
# Stationary distribution: solve π = πT
eigenvalues, eigenvectors = np.linalg.eig(T.T)
stationary = eigenvectors[:, np.isclose(eigenvalues, 1)].real.flatten()
stationary /= stationary.sum()
print(f"Stationary distribution: {stationary.round(4)}")
print(f"Empirical frequencies:   {np.bincount(chain, minlength=3) / len(chain)}")
```

### 2.2 에르고딕성과 수렴(Ergodicity and Convergence)

MCMC가 작동하려면, 마르코프 체인은 다음 조건을 만족해야 합니다:
- **비환원성(Irreducible)**: 어떤 상태에서든 다른 어떤 상태로든 도달 가능
- **비주기성(Aperiodic)**: 순환에 갇히지 않음
- **양의 재귀성(Positive recurrent)**: 어떤 상태로의 기대 복귀 시간이 유한

이 조건들은 체인이 고유한 **정상 분포(stationary distribution)**로 수렴함을 보장합니다.

---

## 3. 메트로폴리스-헤이스팅스 알고리즘(Metropolis-Hastings Algorithm)

가장 기본적인 MCMC 알고리즘입니다. 정상 분포가 목표 사후분포인 마르코프 체인을 구성합니다.

### 3.1 알고리즘(The Algorithm)

```
1. Initialize θ₀
2. For t = 1, 2, ..., T:
   a. Propose θ* ~ q(θ* | θ_{t-1})        (proposal distribution)
   b. Compute acceptance ratio:
      α = min(1, [p(θ*) · q(θ_{t-1}|θ*)] / [p(θ_{t-1}) · q(θ*|θ_{t-1})])
   c. Accept: θ_t = θ* with probability α
      Reject: θ_t = θ_{t-1} with probability 1-α
```

### 3.2 처음부터 구현(Implementation from Scratch)

```python
class MetropolisHastings:
    """Metropolis-Hastings MCMC sampler."""

    def __init__(self, log_target, proposal_std=1.0):
        """
        Args:
            log_target: function θ → log P(θ|data) (up to constant)
            proposal_std: standard deviation of Gaussian proposal
        """
        self.log_target = log_target
        self.proposal_std = proposal_std

    def sample(self, initial, n_samples, burn_in=1000):
        """Run MCMC sampling."""
        ndim = len(initial) if hasattr(initial, '__len__') else 1
        scalar = ndim == 1

        if scalar:
            initial = np.array([initial])

        samples = np.zeros((n_samples + burn_in, ndim))
        samples[0] = initial
        current_log_p = self.log_target(initial)
        n_accepted = 0

        for t in range(1, n_samples + burn_in):
            # Propose (symmetric Gaussian → simplified acceptance ratio)
            proposal = samples[t-1] + np.random.normal(0, self.proposal_std, size=ndim)
            proposed_log_p = self.log_target(proposal)

            # Acceptance ratio (log scale)
            log_alpha = proposed_log_p - current_log_p

            # Accept or reject
            if np.log(np.random.uniform()) < log_alpha:
                samples[t] = proposal
                current_log_p = proposed_log_p
                n_accepted += 1
            else:
                samples[t] = samples[t-1]

        acceptance_rate = n_accepted / (n_samples + burn_in)
        posterior_samples = samples[burn_in:]

        if scalar:
            posterior_samples = posterior_samples.flatten()

        return posterior_samples, acceptance_rate


# Example 1: Sample from a mixture of Gaussians
def log_target_mixture(theta):
    """Log target: 0.3 * N(-2, 0.5) + 0.7 * N(3, 1)"""
    p1 = 0.3 * stats.norm.pdf(theta[0], -2, 0.5)
    p2 = 0.7 * stats.norm.pdf(theta[0], 3, 1)
    return np.log(p1 + p2 + 1e-300)


mh = MetropolisHastings(log_target_mixture, proposal_std=1.5)
samples, acc_rate = mh.sample(initial=0.0, n_samples=20000, burn_in=2000)
print(f"Acceptance rate: {acc_rate:.3f}")
print(f"Sample mean: {samples.mean():.3f}")
print(f"Sample std:  {samples.std():.3f}")

# Plot trace and histogram
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
ax1.plot(samples[:2000], linewidth=0.5)
ax1.set_title("Trace Plot (first 2000 samples)")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("θ")

ax2.hist(samples, bins=100, density=True, alpha=0.7, label="MCMC samples")
theta_grid = np.linspace(-5, 7, 1000)
true_pdf = 0.3 * stats.norm.pdf(theta_grid, -2, 0.5) + 0.7 * stats.norm.pdf(theta_grid, 3, 1)
ax2.plot(theta_grid, true_pdf, 'r-', linewidth=2, label="True density")
ax2.legend()
ax2.set_title("Posterior Histogram vs True Density")
plt.tight_layout()
plt.savefig("mh_mixture.png", dpi=100)
plt.show()
```

### 3.3 다차원 메트로폴리스-헤이스팅스(Multi-dimensional Metropolis-Hastings)

```python
# Example 2: Bayesian linear regression
def log_posterior_regression(params, X, y, prior_std=10.0):
    """Log posterior for Bayesian linear regression."""
    beta = params[:-1]
    log_sigma = params[-1]
    sigma = np.exp(log_sigma)

    # Prior: beta ~ Normal(0, prior_std), sigma ~ HalfNormal(5)
    log_prior = np.sum(stats.norm.logpdf(beta, 0, prior_std))
    log_prior += stats.halfnorm.logpdf(sigma, scale=5)
    log_prior += log_sigma  # Jacobian for log-transform

    # Likelihood: y ~ Normal(X @ beta, sigma)
    log_lik = np.sum(stats.norm.logpdf(y, X @ beta, sigma))

    return log_prior + log_lik


# Generate synthetic data
np.random.seed(42)
n = 50
X = np.column_stack([np.ones(n), np.random.randn(n)])
true_beta = np.array([2.0, -1.5])
true_sigma = 0.8
y = X @ true_beta + np.random.normal(0, true_sigma, n)

# Run MCMC
log_target = lambda params: log_posterior_regression(params, X, y)
mh_reg = MetropolisHastings(log_target, proposal_std=0.1)
samples_reg, acc_rate = mh_reg.sample(
    initial=np.array([0.0, 0.0, 0.0]),  # [beta0, beta1, log_sigma]
    n_samples=30000, burn_in=5000
)

print(f"Acceptance rate: {acc_rate:.3f}")
print(f"Posterior means: β₀={samples_reg[:, 0].mean():.3f}, "
      f"β₁={samples_reg[:, 1].mean():.3f}, "
      f"σ={np.exp(samples_reg[:, 2]).mean():.3f}")
print(f"True values:     β₀={true_beta[0]:.3f}, β₁={true_beta[1]:.3f}, σ={true_sigma:.3f}")
```

### 3.4 제안 분포 조정(Tuning the Proposal Distribution)

수용률은 고차원 목표에서 약 23%, 1차원 목표에서 44%가 이상적입니다(Roberts, Gelman, & Gilks, 1997).

```python
def tune_proposal_std(log_target, initial, target_rate=0.234, n_tune=5000):
    """Adaptively tune proposal standard deviation."""
    ndim = len(initial)
    proposal_std = 1.0
    theta = initial.copy()
    log_p = log_target(theta)

    for batch in range(20):
        n_accept = 0
        for _ in range(n_tune):
            proposal = theta + np.random.normal(0, proposal_std, size=ndim)
            log_p_proposal = log_target(proposal)
            if np.log(np.random.uniform()) < log_p_proposal - log_p:
                theta = proposal
                log_p = log_p_proposal
                n_accept += 1

        rate = n_accept / n_tune
        # Adjust proposal_std
        if rate < target_rate:
            proposal_std *= 0.8
        else:
            proposal_std *= 1.2
        print(f"Batch {batch}: acceptance={rate:.3f}, proposal_std={proposal_std:.4f}")

        if abs(rate - target_rate) < 0.05:
            break

    return proposal_std

optimal_std = tune_proposal_std(log_target, np.array([0.0, 0.0, 0.0]))
print(f"Optimal proposal std: {optimal_std:.4f}")
```

---

## 4. 깁스 샘플링(Gibbs Sampling)

깁스 샘플링은 한 번에 한 변수씩, 다른 모든 변수가 주어진 **전조건부 분포(full conditional distribution)**에서 샘플링합니다.

### 4.1 알고리즘(Algorithm)

```
1. Initialize θ₁⁰, θ₂⁰, ..., θ_d⁰
2. For t = 1, 2, ..., T:
   θ₁ᵗ ~ P(θ₁ | θ₂^{t-1}, θ₃^{t-1}, ..., θ_d^{t-1})
   θ₂ᵗ ~ P(θ₂ | θ₁^t, θ₃^{t-1}, ..., θ_d^{t-1})
   ...
   θ_dᵗ ~ P(θ_d | θ₁^t, θ₂^t, ..., θ_{d-1}^t)
```

### 4.2 정규 분포를 위한 깁스 샘플러(Gibbs Sampler for Normal Distribution)

```python
def gibbs_normal(data, n_samples=10000, burn_in=2000):
    """
    Gibbs sampler for Normal model: y ~ N(mu, sigma^2)
    Priors: mu ~ N(mu_0, tau_0^2), sigma^2 ~ InvGamma(a, b)
    """
    # Hyperparameters
    mu_0, tau_0_sq = 0.0, 100.0  # prior on mu
    a, b = 2.0, 1.0              # prior on sigma^2

    n = len(data)
    data_mean = data.mean()
    data_var = data.var()

    # Initialize
    mu = data_mean
    sigma_sq = data_var

    samples_mu = np.zeros(n_samples + burn_in)
    samples_sigma_sq = np.zeros(n_samples + burn_in)

    for t in range(n_samples + burn_in):
        # Sample mu | sigma^2, data
        precision_prior = 1 / tau_0_sq
        precision_lik = n / sigma_sq
        precision_post = precision_prior + precision_lik
        mu_post = (precision_prior * mu_0 + precision_lik * data_mean) / precision_post
        sigma_post = np.sqrt(1 / precision_post)
        mu = np.random.normal(mu_post, sigma_post)

        # Sample sigma^2 | mu, data
        a_post = a + n / 2
        b_post = b + np.sum((data - mu)**2) / 2
        sigma_sq = 1 / np.random.gamma(a_post, 1 / b_post)

        samples_mu[t] = mu
        samples_sigma_sq[t] = sigma_sq

    return (samples_mu[burn_in:], samples_sigma_sq[burn_in:])


# Generate data
np.random.seed(42)
data = np.random.normal(5.0, 2.0, size=50)

mu_samples, sigma_sq_samples = gibbs_normal(data)
print(f"Posterior mean of μ:  {mu_samples.mean():.3f} (true: 5.0)")
print(f"Posterior mean of σ²: {sigma_sq_samples.mean():.3f} (true: 4.0)")
print(f"Posterior mean of σ:  {np.sqrt(sigma_sq_samples).mean():.3f} (true: 2.0)")
```

### 4.3 가우시안 혼합 모델을 위한 깁스 샘플러(Gibbs Sampler for Gaussian Mixture Model)

```python
def gibbs_gmm(data, K=2, n_samples=5000, burn_in=1000):
    """Gibbs sampler for a K-component Gaussian Mixture Model."""
    n = len(data)

    # Initialize
    labels = np.random.randint(0, K, size=n)
    mus = np.random.randn(K) * 2
    sigmas = np.ones(K)
    weights = np.ones(K) / K

    trace_mus = np.zeros((n_samples, K))
    trace_weights = np.zeros((n_samples, K))

    for t in range(n_samples + burn_in):
        # Step 1: Sample labels (cluster assignments)
        log_probs = np.zeros((n, K))
        for k in range(K):
            log_probs[:, k] = (np.log(weights[k] + 1e-300)
                              + stats.norm.logpdf(data, mus[k], sigmas[k]))
        # Normalize
        log_probs -= log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs)
        probs /= probs.sum(axis=1, keepdims=True)
        labels = np.array([np.random.choice(K, p=probs[i]) for i in range(n)])

        # Step 2: Sample weights (Dirichlet posterior)
        counts = np.bincount(labels, minlength=K) + 1.0  # Dirichlet prior alpha=1
        weights = np.random.dirichlet(counts)

        # Step 3: Sample means (Normal-Normal conjugacy)
        for k in range(K):
            mask = labels == k
            nk = mask.sum()
            if nk > 0:
                data_mean_k = data[mask].mean()
                precision = nk / sigmas[k]**2 + 1 / 100.0  # prior precision
                mu_post = (nk / sigmas[k]**2 * data_mean_k) / precision
                mus[k] = np.random.normal(mu_post, 1 / np.sqrt(precision))

        # Step 4: Sample variances (Inverse-Gamma posterior)
        for k in range(K):
            mask = labels == k
            nk = mask.sum()
            if nk > 1:
                a_post = 2 + nk / 2
                b_post = 1 + np.sum((data[mask] - mus[k])**2) / 2
                sigmas[k] = np.sqrt(1 / np.random.gamma(a_post, 1 / b_post))

        if t >= burn_in:
            idx = t - burn_in
            trace_mus[idx] = np.sort(mus)  # label switching: sort for identifiability
            trace_weights[idx] = weights[np.argsort(mus)]

    return trace_mus, trace_weights


# Generate mixture data
np.random.seed(42)
data_gmm = np.concatenate([
    np.random.normal(-2, 0.8, 200),
    np.random.normal(3, 1.2, 300),
])

mus_trace, weights_trace = gibbs_gmm(data_gmm, K=2)
print(f"Posterior means: μ₁={mus_trace[:, 0].mean():.3f}, μ₂={mus_trace[:, 1].mean():.3f}")
print(f"Posterior weights: w₁={weights_trace[:, 0].mean():.3f}, w₂={weights_trace[:, 1].mean():.3f}")
```

---

## 5. 수렴 진단(Convergence Diagnostics)

MCMC를 실행하는 것만으로는 충분하지 않습니다 — 체인이 목표 분포로 수렴했는지 반드시 확인해야 합니다.

### 5.1 추적 플롯(Trace Plots)

```python
def plot_diagnostics(samples, param_name="θ"):
    """Plot trace, histogram, autocorrelation, and running mean."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Trace plot
    axes[0, 0].plot(samples, linewidth=0.3)
    axes[0, 0].set_title(f"Trace Plot: {param_name}")
    axes[0, 0].set_xlabel("Iteration")

    # Histogram
    axes[0, 1].hist(samples, bins=50, density=True, alpha=0.7)
    axes[0, 1].set_title(f"Posterior: {param_name}")
    axes[0, 1].set_xlabel(param_name)

    # Autocorrelation
    max_lag = min(100, len(samples) // 4)
    acf = np.correlate(samples - samples.mean(), samples - samples.mean(), mode='full')
    acf = acf[len(acf)//2:]
    acf /= acf[0]
    axes[1, 0].bar(range(max_lag), acf[:max_lag], width=1.0, alpha=0.7)
    axes[1, 0].set_title("Autocorrelation")
    axes[1, 0].set_xlabel("Lag")
    axes[1, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    # Running mean
    running_mean = np.cumsum(samples) / np.arange(1, len(samples) + 1)
    axes[1, 1].plot(running_mean, linewidth=1)
    axes[1, 1].axhline(y=samples.mean(), color='r', linestyle='--', alpha=0.7)
    axes[1, 1].set_title("Running Mean")
    axes[1, 1].set_xlabel("Iteration")

    plt.suptitle(f"MCMC Diagnostics: {param_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"mcmc_diagnostics_{param_name}.png", dpi=100)
    plt.show()

plot_diagnostics(mu_samples, "μ")
```

### 5.2 R-hat(Gelman-Rubin 진단)

R-hat은 체인 내 분산과 체인 간 분산을 비교합니다. 1.0에 가까운 값은 수렴을 나타냅니다.

```python
def compute_rhat(chains):
    """
    Compute R-hat (potential scale reduction factor).
    chains: array of shape (n_chains, n_samples)
    """
    n_chains, n_samples = chains.shape

    # Within-chain variance
    W = np.mean(np.var(chains, axis=1, ddof=1))

    # Between-chain variance
    chain_means = np.mean(chains, axis=1)
    B = n_samples * np.var(chain_means, ddof=1)

    # Pooled variance estimate
    var_hat = (1 - 1/n_samples) * W + (1/n_samples) * B

    # R-hat
    rhat = np.sqrt(var_hat / W)
    return rhat


# Run 4 chains from different starting points
chains = []
for start in [-5.0, 0.0, 5.0, 10.0]:
    mh_chain = MetropolisHastings(log_target_mixture, proposal_std=1.5)
    s, _ = mh_chain.sample(initial=start, n_samples=10000, burn_in=2000)
    chains.append(s)

chains = np.array(chains)
rhat = compute_rhat(chains)
print(f"R-hat: {rhat:.4f} (target: < 1.01)")
```

### 5.3 유효 표본 크기(Effective Sample Size, ESS)

자기상관으로 인해 MCMC 샘플은 독립적이지 않습니다. ESS는 "실질적으로 독립적인" 샘플 수를 추정합니다.

```python
def effective_sample_size(samples):
    """Compute effective sample size using initial positive sequence estimator."""
    n = len(samples)
    acf = np.correlate(samples - samples.mean(), samples - samples.mean(), mode='full')
    acf = acf[len(acf)//2:] / acf[len(acf)//2]

    # Sum autocorrelations until they become negative (in pairs)
    tau = 1.0
    for lag in range(1, n // 2):
        rho = acf[lag]
        if rho < 0:
            break
        tau += 2 * rho

    ess = n / tau
    return ess


ess = effective_sample_size(mu_samples)
print(f"Effective sample size: {ess:.0f} out of {len(mu_samples)} total")
print(f"Efficiency: {ess / len(mu_samples) * 100:.1f}%")
```

### 5.4 ArviZ를 사용한 진단(Using ArviZ for Diagnostics)

```python
import arviz as az

# Convert samples to ArviZ InferenceData
trace_data = az.from_dict(
    posterior={
        "mu": mu_samples.reshape(1, -1),           # (chains, draws)
        "sigma": np.sqrt(sigma_sq_samples).reshape(1, -1),
    }
)

# Summary statistics with diagnostics
summary = az.summary(trace_data)
print(summary)

# R-hat and ESS are automatically computed
# Target: R-hat < 1.01, ESS > 400 (per chain)
```

---

## 6. 해밀턴 몬테카를로 미리보기(Hamiltonian Monte Carlo Preview)

HMC는 기울기 정보를 사용하여 효율적인 제안을 만들어, 랜덤 워크 행동을 극적으로 줄입니다.

### 6.1 물리학 비유(The Physics Analogy)

HMC는 매개변수 공간을 물리적 시스템으로 취급합니다:
- **위치(Position)** q = 매개변수 값 θ
- **운동량(Momentum)** p = 보조 변수 (정규 분포에서 추출)
- **해밀토니안(Hamiltonian)** H(q, p) = U(q) + K(p), 여기서 U(q) = -log P(θ|D) (위치 에너지)이고 K(p) = p²/2 (운동 에너지)

```python
def hmc_step(current_q, log_prob_fn, grad_log_prob_fn, step_size, n_leapfrog):
    """Single HMC step with leapfrog integration."""
    q = current_q.copy()
    # Sample momentum
    p = np.random.normal(size=len(q))
    current_p = p.copy()

    # Leapfrog integration
    p += 0.5 * step_size * grad_log_prob_fn(q)  # half step for momentum
    for i in range(n_leapfrog - 1):
        q += step_size * p                        # full step for position
        p += step_size * grad_log_prob_fn(q)       # full step for momentum
    q += step_size * p                            # final full step for position
    p += 0.5 * step_size * grad_log_prob_fn(q)    # final half step for momentum
    p = -p  # negate momentum for reversibility

    # Metropolis acceptance
    current_H = -log_prob_fn(current_q) + 0.5 * np.sum(current_p**2)
    proposed_H = -log_prob_fn(q) + 0.5 * np.sum(p**2)

    if np.log(np.random.uniform()) < current_H - proposed_H:
        return q, True   # accept
    else:
        return current_q, False  # reject


def hmc_sampler(log_prob_fn, grad_log_prob_fn, initial, n_samples,
                step_size=0.1, n_leapfrog=20, burn_in=500):
    """Full HMC sampler."""
    ndim = len(initial)
    samples = np.zeros((n_samples + burn_in, ndim))
    samples[0] = initial
    n_accepted = 0

    for t in range(1, n_samples + burn_in):
        samples[t], accepted = hmc_step(
            samples[t-1], log_prob_fn, grad_log_prob_fn, step_size, n_leapfrog
        )
        if accepted:
            n_accepted += 1

    acceptance_rate = n_accepted / (n_samples + burn_in)
    return samples[burn_in:], acceptance_rate


# Example: 2D Gaussian
def log_prob_2d(q):
    """Log probability of a correlated 2D Gaussian."""
    Sigma_inv = np.array([[1.0, -0.8], [-0.8, 1.0]])  # precision matrix
    return -0.5 * q @ Sigma_inv @ q

def grad_log_prob_2d(q):
    """Gradient of log probability."""
    Sigma_inv = np.array([[1.0, -0.8], [-0.8, 1.0]])
    return -Sigma_inv @ q

hmc_samples, hmc_acc = hmc_sampler(
    log_prob_2d, grad_log_prob_2d,
    initial=np.array([0.0, 0.0]),
    n_samples=5000, step_size=0.15, n_leapfrog=25
)
print(f"HMC acceptance rate: {hmc_acc:.3f}")
print(f"HMC sample covariance:\n{np.cov(hmc_samples.T).round(3)}")
```

### 6.2 HMC가 랜덤 워크 MH보다 나은 이유(Why HMC is Better Than Random-Walk MH)

```python
def compare_mh_vs_hmc():
    """Visual comparison of MH vs HMC sampling trajectories."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # MH samples
    mh = MetropolisHastings(lambda q: log_prob_2d(q), proposal_std=0.5)
    mh_samples, mh_acc = mh.sample(initial=np.array([0.0, 0.0]), n_samples=2000, burn_in=500)

    axes[0].plot(mh_samples[:, 0], mh_samples[:, 1], 'b-', alpha=0.3, linewidth=0.5)
    axes[0].scatter(mh_samples[:, 0], mh_samples[:, 1], c=range(len(mh_samples)),
                    cmap='viridis', s=1)
    axes[0].set_title(f"Metropolis-Hastings (acc={mh_acc:.2f})")
    axes[0].set_xlabel("θ₁")
    axes[0].set_ylabel("θ₂")

    # HMC samples
    axes[1].plot(hmc_samples[:2000, 0], hmc_samples[:2000, 1], 'r-', alpha=0.3, linewidth=0.5)
    axes[1].scatter(hmc_samples[:2000, 0], hmc_samples[:2000, 1], c=range(2000),
                    cmap='viridis', s=1)
    axes[1].set_title(f"HMC (acc={hmc_acc:.2f})")
    axes[1].set_xlabel("θ₁")
    axes[1].set_ylabel("θ₂")

    for ax in axes:
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')

    plt.suptitle("MH vs HMC: Sampling from Correlated 2D Gaussian")
    plt.tight_layout()
    plt.savefig("mh_vs_hmc.png", dpi=100)
    plt.show()

compare_mh_vs_hmc()
```

### 6.3 NUTS: No-U-Turn 샘플러(The No-U-Turn Sampler)

NUTS는 궤적이 되돌아가기 시작하는 것을 감지하여 리프프로그 단계 수를 자동으로 조정합니다. 이를 통해 `n_leapfrog`를 수동으로 설정할 필요가 없어집니다.

NUTS는 PyMC과 Stan의 기본 샘플러입니다(레슨 04와 07에서 다룸).

---

## 7. 실전 MCMC 팁(Practical MCMC Tips)

### 7.1 매개변수화가 중요합니다(Parameterization Matters)

```python
# Bad: sampling sigma directly (bounded at 0)
# Good: sample log_sigma (unbounded) and transform

# Bad: sampling a correlation matrix directly
# Good: use the Cholesky decomposition (LKJ prior in Stan)

# Example: centered vs non-centered parameterization
# Centered (can be slow for hierarchical models):
#   mu ~ N(0, sigma_mu)
#   theta_i ~ N(mu, sigma_theta)
#
# Non-centered (often faster):
#   mu ~ N(0, sigma_mu)
#   z_i ~ N(0, 1)
#   theta_i = mu + sigma_theta * z_i
```

### 7.2 초기화 전략(Initialization Strategies)

```python
def initialize_chains(n_chains, n_params, strategy="dispersed"):
    """Generate initial values for multiple MCMC chains."""
    if strategy == "dispersed":
        # Uniform in [-2, 2] (recommended for Stan)
        return np.random.uniform(-2, 2, size=(n_chains, n_params))
    elif strategy == "jitter_mle":
        # Start near MLE with small perturbation
        mle = np.zeros(n_params)  # placeholder for actual MLE
        return mle + np.random.normal(0, 0.1, size=(n_chains, n_params))
    elif strategy == "prior":
        # Sample from the prior
        return np.random.normal(0, 1, size=(n_chains, n_params))

inits = initialize_chains(4, 3, "dispersed")
print(f"Initial values for 4 chains, 3 params:\n{inits}")
```

### 7.3 씨닝과 저장(Thinning and Storage)

```python
def thin_samples(samples, thin_factor=10):
    """Thin MCMC samples to reduce autocorrelation and storage."""
    return samples[::thin_factor]

# Generally NOT recommended — it's better to run longer chains
# and keep all samples. Only thin for storage constraints.
```

---

## 8. 일반적인 MCMC 병리현상(Common MCMC Pathologies)

### 8.1 느린 혼합(Slow Mixing)

```python
def demonstrate_slow_mixing():
    """Show slow mixing in a bimodal target."""
    def log_bimodal(theta):
        return np.log(
            0.5 * stats.norm.pdf(theta[0], -5, 0.5) +
            0.5 * stats.norm.pdf(theta[0], 5, 0.5) + 1e-300
        )

    # With small proposal std: gets stuck in one mode
    mh_small = MetropolisHastings(log_bimodal, proposal_std=0.3)
    samples_stuck, acc = mh_small.sample(initial=-5.0, n_samples=10000)

    # With large proposal std: can jump between modes but low acceptance
    mh_large = MetropolisHastings(log_bimodal, proposal_std=8.0)
    samples_jump, acc2 = mh_large.sample(initial=-5.0, n_samples=10000)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    ax1.plot(samples_stuck, linewidth=0.5)
    ax1.set_title(f"Small proposal (σ=0.3): Stuck in one mode (acc={acc:.2f})")
    ax2.plot(samples_jump, linewidth=0.5)
    ax2.set_title(f"Large proposal (σ=8.0): Jumps but low acceptance (acc={acc2:.2f})")
    plt.tight_layout()
    plt.savefig("slow_mixing.png", dpi=100)
    plt.show()

demonstrate_slow_mixing()
```

### 8.2 발산 전이(Divergent Transitions)

HMC에서의 발산 전이는 사후분포 기하학의 급격한 곡률로 인해 리프프로그 적분기가 실패함을 나타냅니다. 이는 보통 모델의 재매개변수화가 필요함을 의미합니다.

---

## 요약(Summary)

| 알고리즘 | 장점 | 단점 | 적합한 경우 |
|---------|------|------|-----------|
| 메트로폴리스-헤이스팅스 | 단순, 범용 | 랜덤 워크, 고차원에서 느림 | 저차원, 교육 |
| 깁스 샘플링 | 조정 불필요, 켤레성 활용 | 상관된 매개변수에서 느림 | 켤레 모델, GMM |
| HMC | 고차원에서 효율적 | 기울기 필요, 조정 | 연속 매개변수 |
| NUTS | 자동 조정된 HMC | 기울기 필요 | 대부분의 모델에 기본 선택 |

| 진단 | 목표 | 해석 |
|------|------|------|
| R-hat | < 1.01 | 체인들이 같은 분포로 수렴 |
| ESS | > 400 | 체인당 충분한 독립 정보 |
| 추적 플롯 | 퍼지 애벌레 | 좋은 혼합, 추세나 정체 없음 |
| 발산 | 0 | 기하학적 문제 없음 |

---

## 참고 문헌(References)

1. Robert, C. P. & Casella, G. (2004). *Monte Carlo Statistical Methods*, 2nd Ed. Springer.
2. Gelman, A., et al. (2013). *Bayesian Data Analysis*, 3rd Ed., Ch. 11-12.
3. Neal, R. M. (2011). "MCMC using Hamiltonian dynamics." *Handbook of MCMC*.
4. Hoffman, M. D. & Gelman, A. (2014). "The No-U-Turn Sampler." *JMLR*, 15, 1593-1623.

---

[이전: 확률적 그래프 모델](./02_Probabilistic_Graphical_Models.md) | [다음: PyMC 소개 →](./04_PyMC_Introduction.md)
