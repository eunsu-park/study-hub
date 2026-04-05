# 01. 베이지안 사고(Bayesian Thinking)

**주제**: Probabilistic_Programming
**레슨**: 19개 중 1번째

[다음: 확률적 그래프 모델](./02_Probabilistic_Graphical_Models.md)

---

> **프레임워크 참고**: 이 레슨은 실습 베이지안 계산을 위해 NumPy와 SciPy를 사용합니다.
> 이후 레슨에서 확장 가능한 추론을 위한 PyMC, Stan, Pyro를 소개합니다.
>
> 설치: `pip install numpy scipy matplotlib`

## 학습 목표(Learning Objectives)

- 베이즈 정리와 그 구성요소(사전분포, 가능도, 사후분포) 이해
- 빈도주의와 베이지안 확률 해석의 차이 구분
- 켤레 사전분포(conjugate prior)를 사용한 베이지안 갱신 구현
- 사전분포 강도가 사후분포 추론에 미치는 영향 시각화

---

## 1. 확률의 두 학파(Two Schools of Probability)

확률에는 두 가지 주요 해석이 있으며, 확률적 프로그래밍에 들어가기 전에 이 차이를 이해하는 것이 필수적입니다.

### 1.1 빈도주의 해석(Frequentist Interpretation)

빈도주의 관점에서 확률은 사건의 **장기 빈도**입니다. 매개변수는 고정된 미지의 상수이며, 데이터만 확률적입니다. 추론은 표집 분포와 p-값에 의존합니다.

```python
import numpy as np

# Frequentist: estimate p(heads) by repeating the experiment
np.random.seed(42)
n_flips = 10000
flips = np.random.binomial(1, 0.7, size=n_flips)
freq_estimate = flips.mean()
print(f"Frequentist estimate of p(heads): {freq_estimate:.4f}")
# Close to 0.7 as n_flips → ∞
```

### 1.2 베이지안 해석(Bayesian Interpretation)

베이지안 관점에서 확률은 **믿음의 정도**를 나타냅니다. 매개변수 자체가 확률 분포를 가집니다. 사전 믿음에서 시작하여, 데이터를 관찰하고, 사후 믿음으로 갱신합니다.

```python
# Bayesian: we express uncertainty about p(heads) as a distribution
# Before seeing any data, we might believe p ~ Uniform(0, 1)
# After seeing data, we update our belief using Bayes' theorem
```

### 1.3 핵심 철학적 차이(Key Philosophical Differences)

| 측면 | 빈도주의 | 베이지안 |
|------|---------|---------|
| 확률 | 장기 빈도 | 믿음의 정도 |
| 매개변수 | 고정, 미지 | 확률 변수 |
| 추론 | MLE, 신뢰구간 | 사후분포 |
| 사전 정보 | 사용 안 함 | 명시적으로 포함 |
| 소표본 | 신뢰하기 어려움 | 사전분포로 자연스럽게 정규화 |

---

## 2. 베이즈 정리(Bayes' Theorem)

베이지안 추론의 초석은 베이즈 정리로, 새로운 증거에 비추어 믿음을 갱신하는 방법을 알려줍니다.

### 2.1 공식(The Formula)

$$P(\theta | D) = \frac{P(D | \theta) \cdot P(\theta)}{P(D)}$$

구성요소:
- $P(\theta | D)$: **사후분포(Posterior)** — 데이터 $D$를 본 후 $\theta$에 대한 갱신된 믿음
- $P(D | \theta)$: **가능도(Likelihood)** — $\theta$가 주어질 때 데이터의 확률
- $P(\theta)$: **사전분포(Prior)** — 데이터를 보기 전 $\theta$에 대한 믿음
- $P(D)$: **증거(Evidence)** (주변 가능도) — 정규화 상수

### 2.2 정규화 상수(The Normalizing Constant)

증거 $P(D)$는 사후분포가 1로 적분되도록 보장합니다:

$$P(D) = \int P(D | \theta) P(\theta) \, d\theta$$

실제로 이 적분은 종종 다루기 어렵습니다 — 이것이 마르코프 체인 몬테카를로(MCMC)와 변분 추론(variational inference)이 필요한 이유입니다(이후 레슨에서 다룸).

### 2.3 비례식(Proportionality Form)

$P(D)$는 $\theta$에 대한 상수이므로, 다음과 같이 쓸 수 있습니다:

$$P(\theta | D) \propto P(D | \theta) \cdot P(\theta)$$

**사후분포 ∝ 가능도 × 사전분포**

이것은 베이지안 통계에서 가장 중요한 공식입니다.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Example: Coin flip inference
# Prior: Beta(2, 2) — slight belief that coin is fair
# Likelihood: Binomial
# Posterior: Beta(2 + heads, 2 + tails)  [conjugacy!]

alpha_prior, beta_prior = 2, 2
n_heads, n_tails = 7, 3

alpha_post = alpha_prior + n_heads   # 9
beta_post = beta_prior + n_tails     # 5

theta = np.linspace(0, 1, 1000)
prior = stats.beta.pdf(theta, alpha_prior, beta_prior)
likelihood = stats.binom.pmf(n_heads, n_heads + n_tails, theta)
posterior = stats.beta.pdf(theta, alpha_post, beta_post)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, dist, title, color in zip(
    axes,
    [prior, likelihood, posterior],
    ["Prior Beta(2,2)", "Likelihood (7H, 3T)", "Posterior Beta(9,5)"],
    ["blue", "green", "red"]
):
    ax.plot(theta, dist, color=color, linewidth=2)
    ax.fill_between(theta, dist, alpha=0.3, color=color)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("θ (probability of heads)")
    ax.set_ylabel("Density")
plt.tight_layout()
plt.savefig("bayesian_updating.png", dpi=100)
plt.show()
```

---

## 3. 사전분포(Prior Distributions)

사전분포는 데이터를 관찰하기 전에 알고 있는(또는 가정하는) 것을 인코딩합니다. 적절한 사전분포를 선택하는 것은 베이지안 모델링의 핵심 기술입니다.

### 3.1 정보적 사전분포(Informative Priors)

정보적 사전분포는 특정 도메인 지식을 인코딩합니다. 특히 소규모 데이터셋에서 사후분포에 큰 영향을 미칠 수 있습니다.

```python
# Informative prior: We know this coin is roughly fair
# Beta(50, 50) is concentrated around 0.5
alpha_info, beta_info = 50, 50
prior_info = stats.beta.pdf(theta, alpha_info, beta_info)

# Even with 7/10 heads, posterior stays closer to 0.5
alpha_post_info = alpha_info + n_heads   # 57
beta_post_info = beta_info + n_tails     # 53
posterior_info = stats.beta.pdf(theta, alpha_post_info, beta_post_info)
print(f"Posterior mean (informative prior): {alpha_post_info / (alpha_post_info + beta_post_info):.4f}")
# ~0.518, pulled toward 0.5
```

### 3.2 약하게 정보적인 사전분포(Weakly Informative Priors)

약하게 정보적인 사전분포는 특정 값에 전념하지 않으면서 매개변수를 합리적인 범위로 제한합니다. 현대 베이지안 실무에서 권장되는 기본값입니다.

```python
# Weakly informative: Normal(0, 10) for regression coefficients
# Rules out extreme values but doesn't favor any particular value strongly
x = np.linspace(-30, 30, 1000)
weak_prior = stats.norm.pdf(x, 0, 10)
```

### 3.3 비정보적(평탄) 사전분포(Non-informative / Flat Priors)

비정보적 사전분포는 "데이터가 스스로 말하게 하려는" 시도입니다. 균일 분포와 제프리스 사전분포가 일반적인 선택입니다.

```python
# Flat prior: Beta(1, 1) = Uniform(0, 1)
# Jeffreys prior for Bernoulli: Beta(0.5, 0.5)
flat_prior = stats.beta.pdf(theta, 1, 1)
jeffreys_prior = stats.beta.pdf(theta, 0.5, 0.5)
```

### 3.4 사전분포 민감도 분석(Prior Sensitivity Analysis)

책임감 있는 베이지안 실무자는 항상 결론이 사전분포 선택에 얼마나 민감한지 확인합니다.

```python
def prior_sensitivity(n_heads, n_tails, priors):
    """Compare posteriors under different priors."""
    theta = np.linspace(0, 1, 1000)
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, (a, b) in priors.items():
        a_post = a + n_heads
        b_post = b + n_tails
        posterior = stats.beta.pdf(theta, a_post, b_post)
        mean = a_post / (a_post + b_post)
        ax.plot(theta, posterior, label=f"{name}: post mean={mean:.3f}", linewidth=2)

    ax.set_xlabel("θ")
    ax.set_ylabel("Posterior density")
    ax.set_title(f"Prior Sensitivity (data: {n_heads}H, {n_tails}T)")
    ax.legend()
    plt.tight_layout()
    return fig

priors = {
    "Flat Beta(1,1)": (1, 1),
    "Jeffreys Beta(0.5,0.5)": (0.5, 0.5),
    "Weak Beta(2,2)": (2, 2),
    "Informative Beta(50,50)": (50, 50),
}
fig = prior_sensitivity(7, 3, priors)
plt.savefig("prior_sensitivity.png", dpi=100)
plt.show()
```

---

## 4. 가능도 함수(Likelihood Functions)

가능도 함수는 서로 다른 매개변수 값이 데이터를 얼마나 잘 설명하는지 측정합니다.

### 4.1 일반적인 가능도 함수(Common Likelihood Functions)

```python
# Bernoulli / Binomial likelihood
def binomial_likelihood(theta, n_heads, n_total):
    """P(data | theta) for coin flips."""
    from scipy.special import comb
    return comb(n_total, n_heads) * theta**n_heads * (1-theta)**(n_total - n_heads)

# Normal likelihood
def normal_likelihood(data, mu, sigma):
    """P(data | mu, sigma) for Gaussian observations."""
    return np.prod(stats.norm.pdf(data, mu, sigma))

# Poisson likelihood
def poisson_likelihood(data, lam):
    """P(data | lambda) for count data."""
    return np.prod(stats.poisson.pmf(data, lam))
```

### 4.2 수치 안정성을 위한 로그 가능도(Log-Likelihood for Numerical Stability)

실무에서는 수치적 언더플로를 방지하기 위해 항상 로그 가능도로 작업합니다.

```python
def log_likelihood_normal(data, mu, sigma):
    """Log P(data | mu, sigma)."""
    n = len(data)
    return -n/2 * np.log(2 * np.pi * sigma**2) - np.sum((data - mu)**2) / (2 * sigma**2)

# Example
data = np.random.normal(5.0, 2.0, size=100)
mu_grid = np.linspace(3, 7, 200)
ll_values = [log_likelihood_normal(data, mu, 2.0) for mu in mu_grid]
mle_mu = mu_grid[np.argmax(ll_values)]
print(f"MLE estimate of mu: {mle_mu:.3f} (true: 5.0)")
```

---

## 5. 켤레 사전분포(Conjugate Priors)

사전분포와 사후분포가 같은 분포 계열에 속하면 **켤레성(conjugacy)**이 있습니다. 이를 통해 수치 적분 없이 닫힌 형태의 베이지안 갱신이 가능합니다.

### 5.1 일반적인 켤레 쌍(Common Conjugate Pairs)

| 가능도 | 사전분포 | 사후분포 | 매개변수 |
|--------|---------|---------|---------|
| 베르누이/이항 | Beta(α, β) | Beta(α+k, β+n-k) | n번 시행 중 k번 성공 |
| 포아송 | Gamma(α, β) | Gamma(α+Σx, β+n) | n개 관측 |
| 정규(σ 알려짐) | Normal(μ₀, σ₀²) | Normal(μₙ, σₙ²) | 정밀도 가중 평균 |
| 정규(μ 알려짐) | Inverse-Gamma(α, β) | Inverse-Gamma(α+n/2, β+SS/2) | SS = 제곱합 |
| 다항 | Dirichlet(α) | Dirichlet(α+counts) | 범주 카운트 |
| 지수 | Gamma(α, β) | Gamma(α+n, β+Σx) | n개 관측 |

### 5.2 베타-이항 켤레성 상세(Beta-Binomial Conjugacy in Detail)

가장 흔하게 사용되는 켤레 쌍입니다. 순차적 갱신을 구현해 봅시다.

```python
class BetaBinomialModel:
    """Sequential Bayesian updating with Beta-Binomial conjugacy."""

    def __init__(self, alpha_prior=1.0, beta_prior=1.0):
        self.alpha = alpha_prior
        self.beta = beta_prior
        self.history = [(alpha_prior, beta_prior)]

    def update(self, n_successes, n_trials):
        """Update posterior after observing data."""
        self.alpha += n_successes
        self.beta += (n_trials - n_successes)
        self.history.append((self.alpha, self.beta))
        return self

    @property
    def mean(self):
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self):
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b)**2 * (a + b + 1))

    def credible_interval(self, level=0.95):
        """Compute credible interval."""
        tail = (1 - level) / 2
        lo = stats.beta.ppf(tail, self.alpha, self.beta)
        hi = stats.beta.ppf(1 - tail, self.alpha, self.beta)
        return lo, hi

    def plot_history(self):
        """Visualize sequential updating."""
        theta = np.linspace(0, 1, 500)
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, (a, b) in enumerate(self.history):
            pdf = stats.beta.pdf(theta, a, b)
            ax.plot(theta, pdf, label=f"Step {i}: Beta({a:.0f},{b:.0f})")
        ax.set_xlabel("θ")
        ax.set_ylabel("Density")
        ax.set_title("Sequential Bayesian Updating")
        ax.legend()
        plt.tight_layout()
        return fig


# Sequential updating example
model = BetaBinomialModel(alpha_prior=2, beta_prior=2)

# Observe batches of coin flips
batches = [(6, 10), (8, 10), (5, 10), (7, 10)]
for heads, total in batches:
    model.update(heads, total)
    lo, hi = model.credible_interval()
    print(f"After {total} flips ({heads}H): "
          f"mean={model.mean:.3f}, 95% CI=[{lo:.3f}, {hi:.3f}]")

model.plot_history()
plt.savefig("sequential_updating.png", dpi=100)
plt.show()
```

### 5.3 정규-정규 켤레성(Normal-Normal Conjugacy)

분산이 알려진 가우시안 데이터의 경우, 평균의 사후분포도 가우시안입니다.

```python
class NormalNormalModel:
    """Bayesian updating for Normal likelihood with known variance."""

    def __init__(self, mu_prior, sigma_prior, sigma_likelihood):
        self.mu = mu_prior
        self.sigma = sigma_prior
        self.sigma_lik = sigma_likelihood

    def update(self, data):
        """Update posterior after observing data points."""
        n = len(data)
        data_mean = np.mean(data)

        # Precision = 1/variance
        prior_precision = 1 / self.sigma**2
        lik_precision = n / self.sigma_lik**2

        post_precision = prior_precision + lik_precision
        post_sigma = np.sqrt(1 / post_precision)
        post_mu = (prior_precision * self.mu + lik_precision * data_mean) / post_precision

        self.mu = post_mu
        self.sigma = post_sigma
        return self

    def credible_interval(self, level=0.95):
        z = stats.norm.ppf(1 - (1 - level) / 2)
        return self.mu - z * self.sigma, self.mu + z * self.sigma


# Example: Estimate the mean temperature
model = NormalNormalModel(mu_prior=20.0, sigma_prior=5.0, sigma_likelihood=2.0)
temperature_data = np.random.normal(22.5, 2.0, size=30)
model.update(temperature_data)
lo, hi = model.credible_interval()
print(f"Posterior mean: {model.mu:.2f}, 95% CI: [{lo:.2f}, {hi:.2f}]")
```

---

## 6. 베이지안 vs 빈도주의: 실전 비교(Bayesian vs Frequentist: Practical Comparison)

### 6.1 신뢰구간 vs 신용구간(Confidence Interval vs Credible Interval)

```python
from scipy.stats import t as t_dist

# Frequentist 95% confidence interval
data = np.array([23.1, 22.5, 24.0, 21.8, 23.5, 22.9, 24.2, 23.0])
n = len(data)
mean = data.mean()
se = data.std(ddof=1) / np.sqrt(n)
t_crit = t_dist.ppf(0.975, df=n-1)
ci_freq = (mean - t_crit * se, mean + t_crit * se)
print(f"Frequentist 95% CI: [{ci_freq[0]:.3f}, {ci_freq[1]:.3f}]")
# Interpretation: 95% of such intervals would contain the true mean

# Bayesian 95% credible interval
bayes_model = NormalNormalModel(mu_prior=22.0, sigma_prior=5.0, sigma_likelihood=1.0)
bayes_model.update(data)
ci_bayes = bayes_model.credible_interval()
print(f"Bayesian 95% CI:    [{ci_bayes[0]:.3f}, {ci_bayes[1]:.3f}]")
# Interpretation: There is a 95% probability that the true mean lies in this interval
```

### 6.2 베이지안이 유리한 경우(When Bayesian Wins)

1. **소표본**: 사전분포가 추정치를 정규화
2. **순차적 갱신**: 배치별 자연스러운 학습
3. **불확실성 정량화**: 점추정이 아닌 전체 사후분포
4. **의사결정**: 사후분포로 기대 효용 계산 가능

---

## 7. 최대 사후확률(MAP) 추정(Maximum A Posteriori Estimation)

MAP 추정은 사후분포의 최빈값을 찾습니다 — MLE와 완전한 베이지안 추론 사이의 다리입니다.

```python
def map_vs_mle_demo():
    """Compare MAP and MLE for a biased coin."""
    # True bias = 0.3, but we only have 5 observations
    np.random.seed(42)
    data = np.random.binomial(1, 0.3, size=5)
    k = data.sum()  # number of heads
    n = len(data)

    # MLE: k/n
    mle = k / n
    print(f"Data: {data}, k={k}, n={n}")
    print(f"MLE:  {mle:.3f}")

    # MAP with Beta(2, 5) prior (we believe coin is biased toward tails)
    alpha, beta = 2, 5
    map_estimate = (alpha + k - 1) / (alpha + beta + n - 2)
    print(f"MAP (Beta(2,5)):  {map_estimate:.3f}")

    # Posterior mean (different from MAP for skewed distributions)
    post_mean = (alpha + k) / (alpha + beta + n)
    print(f"Posterior mean:   {post_mean:.3f}")

    # Visualize
    theta = np.linspace(0, 1, 1000)
    a_post, b_post = alpha + k, beta + n - k
    posterior = stats.beta.pdf(theta, a_post, b_post)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(theta, posterior, 'b-', linewidth=2, label="Posterior")
    ax.axvline(mle, color='r', linestyle='--', label=f"MLE = {mle:.3f}")
    ax.axvline(map_estimate, color='g', linestyle='--', label=f"MAP = {map_estimate:.3f}")
    ax.axvline(post_mean, color='purple', linestyle='--', label=f"Post. mean = {post_mean:.3f}")
    ax.axvline(0.3, color='k', linestyle=':', label="True θ = 0.3")
    ax.legend()
    ax.set_xlabel("θ")
    ax.set_ylabel("Density")
    ax.set_title("MLE vs MAP vs Posterior Mean")
    plt.tight_layout()
    plt.savefig("map_vs_mle.png", dpi=100)
    plt.show()

map_vs_mle_demo()
```

---

## 8. 예측 분포(Predictive Distributions)

베이지안 접근법의 핵심 장점: 단일 매개변수 추정치로 예측하는 대신, 전체 사후분포에 걸쳐 예측을 평균합니다.

### 8.1 사전 예측 분포(Prior Predictive Distribution)

```python
def prior_predictive(alpha, beta, n_trials, n_samples=10000):
    """Sample from the prior predictive distribution."""
    # Step 1: Sample theta from the prior
    thetas = np.random.beta(alpha, beta, size=n_samples)
    # Step 2: For each theta, sample the number of successes
    predictions = np.random.binomial(n_trials, thetas)
    return predictions

prior_pred = prior_predictive(2, 2, n_trials=10)
print(f"Prior predictive mean: {prior_pred.mean():.2f}")
print(f"Prior predictive std:  {prior_pred.std():.2f}")
```

### 8.2 사후 예측 분포(Posterior Predictive Distribution)

```python
def posterior_predictive(alpha_post, beta_post, n_trials, n_samples=10000):
    """Sample from the posterior predictive distribution."""
    thetas = np.random.beta(alpha_post, beta_post, size=n_samples)
    predictions = np.random.binomial(n_trials, thetas)
    return predictions

# After observing 7/10 heads with Beta(2,2) prior
# Posterior: Beta(9, 5)
post_pred = posterior_predictive(9, 5, n_trials=10)
print(f"Posterior predictive: {post_pred.mean():.2f} ± {post_pred.std():.2f}")

# Compare with plug-in prediction (using point estimate)
plugin_pred = np.random.binomial(10, 9/14, size=10000)
print(f"Plug-in prediction:  {plugin_pred.mean():.2f} ± {plugin_pred.std():.2f}")
# Posterior predictive has wider uncertainty (accounts for parameter uncertainty)
```

---

## 9. 격자 근사(Grid Approximation)

레슨 03에서 마르코프 체인 몬테카를로(MCMC)에 도달하기 전, 격자 근사는 수치적으로 사후분포를 계산하는 간단한 방법입니다.

```python
def grid_approximation(data, n_grid=1000):
    """Compute posterior via grid approximation for a Bernoulli model."""
    theta_grid = np.linspace(0, 1, n_grid)

    # Prior: Beta(2, 2)
    log_prior = stats.beta.logpdf(theta_grid, 2, 2)

    # Likelihood: product of Bernoulli
    k = data.sum()
    n = len(data)
    log_likelihood = k * np.log(theta_grid + 1e-10) + (n - k) * np.log(1 - theta_grid + 1e-10)

    # Unnormalized log-posterior
    log_posterior = log_prior + log_likelihood

    # Normalize (in log space for stability)
    log_posterior -= log_posterior.max()
    posterior = np.exp(log_posterior)
    posterior /= np.trapz(posterior, theta_grid)

    return theta_grid, posterior


# Example
data = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 1])
theta_grid, posterior = grid_approximation(data)
post_mean = np.trapz(theta_grid * posterior, theta_grid)
print(f"Grid approximation posterior mean: {post_mean:.4f}")

# Compare with analytical result
alpha_post = 2 + data.sum()
beta_post = 2 + len(data) - data.sum()
print(f"Analytical posterior mean:         {alpha_post / (alpha_post + beta_post):.4f}")
```

---

## 10. 베이지안 워크플로우(The Bayesian Workflow)

Gelman 등이 제안한 체계적 베이지안 모델링 접근법:

### 10.1 워크플로우 단계(The Workflow Steps)

```
┌─────────────────┐
│ 1. 모델 정의    │  가능도, 사전분포, 구조 선택
└────────┬────────┘
         │
┌────────▼────────┐
│ 2. 사전 검사    │  사전 예측 시뮬레이션
└────────┬────────┘
         │
┌────────▼────────┐
│ 3. 모델 적합    │  MCMC, VI, 또는 켤레 갱신
└────────┬────────┘
         │
┌────────▼────────┐
│ 4. 진단         │  수렴 검사, R-hat, ESS
└────────┬────────┘
         │
┌────────▼────────────┐
│ 5. 사후 검사        │  사후 예측 검사
└────────┬────────────┘
         │
┌────────▼────────┐
│ 6. 모델 비교    │  WAIC, LOO-CV, 베이즈 인자
└────────┬────────┘
         │
┌────────▼────────┐
│ 7. 소통         │  사후분포 요약, 의사결정 보고
└─────────────────┘
```

### 10.2 사전 예측 검사(Prior Predictive Checking)

모델 적합 전에, 사전분포에서 데이터를 시뮬레이션하여 모델이 합리적인 데이터를 생성하는지 확인합니다.

```python
def prior_predictive_check():
    """Check if our model specification is reasonable."""
    n_simulations = 1000
    n_observations = 50

    # Model: y ~ Normal(mu, sigma)
    # Priors: mu ~ Normal(0, 10), sigma ~ HalfNormal(5)
    simulated_means = []
    for _ in range(n_simulations):
        mu = np.random.normal(0, 10)
        sigma = abs(np.random.normal(0, 5))
        y_sim = np.random.normal(mu, sigma, size=n_observations)
        simulated_means.append(y_sim.mean())

    simulated_means = np.array(simulated_means)
    print(f"Prior predictive mean range: [{simulated_means.min():.1f}, {simulated_means.max():.1f}]")
    print(f"Prior predictive mean of means: {simulated_means.mean():.2f}")

    # If these ranges are unreasonable for your domain, adjust priors!

prior_predictive_check()
```

---

## 11. 베이지안 의사결정 이론(Bayesian Decision Theory)

사후분포는 불확실성 하에서 원칙적인 의사결정을 가능하게 합니다.

### 11.1 손실 함수(Loss Functions)

```python
def bayesian_decision(posterior_samples, loss_fn="squared"):
    """Find the optimal point estimate under a given loss function."""
    if loss_fn == "squared":
        # Optimal: posterior mean (minimizes expected squared error)
        return np.mean(posterior_samples)
    elif loss_fn == "absolute":
        # Optimal: posterior median (minimizes expected absolute error)
        return np.median(posterior_samples)
    elif loss_fn == "zero_one":
        # Optimal: posterior mode (MAP)
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(posterior_samples)
        grid = np.linspace(posterior_samples.min(), posterior_samples.max(), 1000)
        return grid[np.argmax(kde(grid))]


# Generate posterior samples from Beta(9, 5)
posterior_samples = np.random.beta(9, 5, size=50000)

for loss in ["squared", "absolute", "zero_one"]:
    estimate = bayesian_decision(posterior_samples, loss)
    print(f"Optimal estimate ({loss} loss): {estimate:.4f}")
```

### 11.2 기대 효용(Expected Utility)

```python
def ab_test_decision(posterior_a, posterior_b):
    """A/B test: which variant is better?"""
    prob_b_better = np.mean(posterior_b > posterior_a)
    expected_lift = np.mean(posterior_b - posterior_a)
    risk_b = np.mean(np.maximum(posterior_a - posterior_b, 0))

    print(f"P(B > A):       {prob_b_better:.4f}")
    print(f"Expected lift:  {expected_lift:.4f}")
    print(f"Risk of B:      {risk_b:.4f}")
    return prob_b_better, expected_lift, risk_b


# Variant A: 120/1000 conversions, Variant B: 145/1000
posterior_a = np.random.beta(1 + 120, 1 + 880, size=50000)
posterior_b = np.random.beta(1 + 145, 1 + 855, size=50000)
ab_test_decision(posterior_a, posterior_b)
```

---

## 12. 베이지안 사고의 흔한 함정(Common Pitfalls in Bayesian Thinking)

### 12.1 기저율 무시(Base Rate Neglect)

```python
def medical_test_example():
    """The classic medical testing problem."""
    prevalence = 0.001       # P(disease) = 0.1%
    sensitivity = 0.99       # P(positive | disease) = 99%
    false_positive = 0.05    # P(positive | no disease) = 5%

    # P(disease | positive) via Bayes' theorem
    p_positive = sensitivity * prevalence + false_positive * (1 - prevalence)
    p_disease_given_positive = (sensitivity * prevalence) / p_positive

    print(f"P(disease | positive test) = {p_disease_given_positive:.4f}")
    print(f"Despite a 99% sensitive test, only {p_disease_given_positive*100:.1f}% "
          f"of positive tests indicate disease!")
    print(f"This is because the base rate (prevalence) is so low.")

medical_test_example()
```

### 12.2 사전분포의 영향 무시(Ignoring the Prior's Influence)

```python
def prior_dominance_demo():
    """Show when the prior dominates vs when data dominates."""
    n_data_points = [1, 5, 10, 50, 200, 1000]
    true_theta = 0.7
    alpha_prior, beta_prior = 50, 50  # Strong prior centered at 0.5

    print(f"Strong prior: Beta({alpha_prior},{beta_prior}), true θ = {true_theta}")
    print("-" * 60)

    for n in n_data_points:
        k = np.random.binomial(n, true_theta)
        post_mean = (alpha_prior + k) / (alpha_prior + beta_prior + n)
        mle = k / n if n > 0 else 0
        print(f"n={n:4d}: k={k:4d}, MLE={mle:.3f}, Posterior mean={post_mean:.3f}")

prior_dominance_demo()
```

---

## 요약(Summary)

| 개념 | 핵심 요점 |
|------|---------|
| 베이즈 정리 | 사후분포 ∝ 가능도 × 사전분포 |
| 사전분포 | 사전 데이터 지식을 인코딩; 항상 민감도 분석 수행 |
| 켤레 사전분포 | 닫힌 형태 갱신 가능 (Beta-이항, Normal-Normal 등) |
| 사후 예측 | 정직한 예측을 위해 매개변수 불확실성에 대해 평균 |
| 격자 근사 | 단순하지만 확장성 낮음; 복잡한 모델에는 MCMC 필요 |
| 베이지안 워크플로우 | 사전 검사 → 적합 → 진단 → 사후 검사 → 비교 |
| 의사결정 이론 | 최적 의사결정을 위해 전체 사후분포 활용 |

---

## 참고 문헌(References)

1. Gelman, A., et al. (2013). *Bayesian Data Analysis*, 3rd Edition. CRC Press.
2. McElreath, R. (2020). *Statistical Rethinking*, 2nd Edition. CRC Press.
3. Kruschke, J. K. (2014). *Doing Bayesian Data Analysis*, 2nd Edition. Academic Press.
4. Betancourt, M. (2017). "A Conceptual Introduction to Hamiltonian Monte Carlo." arXiv:1701.02434.

---

[다음: 확률적 그래프 모델 →](./02_Probabilistic_Graphical_Models.md)
