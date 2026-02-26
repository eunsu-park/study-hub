# 18. 베이지안 통계 기초 (Introduction to Bayesian Statistics)

[이전: 일반화 선형모형](./17_Generalized_Linear_Models.md) | [다음: 베이지안 추론](./19_Bayesian_Inference.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 확률, 모수, 구간에 대한 빈도주의(frequentist)와 베이지안(Bayesian) 해석을 비교할 수 있다
2. 이산형 및 연속형 문제에 베이즈 정리(Bayes' theorem)를 적용하여 사후 확률(posterior probability)을 계산할 수 있다
3. 순차적 베이지안 업데이트(sequential Bayesian updating)를 시연하고, 새로운 데이터에 따라 사후분포가 어떻게 변화하는지 설명할 수 있다
4. 켤레 사전분포(conjugate prior) 쌍(Beta-Binomial, Normal-Normal, Gamma-Poisson)을 식별하고 구현할 수 있다
5. MLE, MAP, 사후 평균(posterior mean) 추정법을 구분하고, 각각이 적합한 상황을 설명할 수 있다
6. MAP 추정과 정규화(regularization)(Ridge, Lasso)의 연관성을 설명할 수 있다
7. 베이지안 A/B 테스트를 설계하고, 개선 확률(probability of improvement)과 기대 상승률(expected lift)을 포함하여 해석할 수 있다

---

> **증거로 믿음을 갱신하기.** 의사가 처음에 당신이 희귀 질환에 걸렸을 확률을 1%로 추정한다고 가정합니다(사전 확률(Prior)). 양성 검사 결과가 도착했지만, 이 검사의 위양성률(false-positive rate)은 5%입니다(가능도(likelihood)). 베이즈 정리는 사전 믿음과 새로운 증거를 결합하여 갱신된 확률(사후 확률(Posterior))인 약 17%를 산출합니다. 핵심 통찰: 최종 믿음은 *이전에 알고 있던 것*과 *데이터가 말해주는 것* 모두에 의존합니다. 이것이 베이지안 사고 — 증거로부터 학습하는 공식적 프레임워크입니다.

빈도주의 방법은 모수를 고정된 미지수로 취급하여 데이터에 관한 질문에 답하는 반면, 베이지안 방법은 모수를 불확실한 양으로 취급하여 모수 자체에 관한 질문에 직접 답합니다. 이러한 관점의 전환은 사전 지식을 반영하고, 가설에 대한 직관적인 확률 진술을 생성하며, 새로운 데이터가 들어올 때 자연스럽게 믿음을 갱신할 수 있게 합니다. 이 레슨은 베이지안 사고의 개념적·계산적 기초를 다집니다.

---

## 1. 빈도주의 vs 베이지안 패러다임

### 1.1 확률 해석의 차이

| 관점 | 빈도주의 (Frequentist) | 베이지안 (Bayesian) |
|------|------------------------|---------------------|
| **확률의 의미** | 장기적인 빈도 (무한 반복 시) | 불확실성의 정도 (믿음) |
| **모수** | 고정된 미지의 상수 | 확률변수 (분포를 가짐) |
| **추론 목표** | 점추정, 신뢰구간 | 사후분포 전체 |
| **사전 정보** | 사용하지 않음 | 사전분포로 반영 |
| **해석** | "95% 신뢰구간" (반복 시 95%가 참값 포함) | "95% 확률로 참값이 구간 내 존재" |

### 1.2 비교 예제: 동전 던지기

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

np.random.seed(42)

# 데이터: 10번 던져서 7번 앞면
n_trials = 10
n_heads = 7

# === 빈도주의 접근 ===
# 점추정: MLE
p_mle = n_heads / n_trials
print(f"[빈도주의] MLE 추정치: {p_mle:.3f}")

# 95% 신뢰구간 (Wald interval)
se = np.sqrt(p_mle * (1 - p_mle) / n_trials)
ci_freq = (p_mle - 1.96 * se, p_mle + 1.96 * se)
print(f"[빈도주의] 95% 신뢰구간: ({ci_freq[0]:.3f}, {ci_freq[1]:.3f})")

# === 베이지안 접근 ===
# 사전분포: Beta(1, 1) = Uniform(0, 1)
alpha_prior, beta_prior = 1, 1

# 사후분포: Beta(alpha + heads, beta + tails)
alpha_post = alpha_prior + n_heads
beta_post = beta_prior + (n_trials - n_heads)

posterior = stats.beta(alpha_post, beta_post)

# 사후 평균
p_bayes = posterior.mean()
print(f"\n[베이지안] 사후 평균: {p_bayes:.3f}")

# 95% 신용구간 (Credible Interval)
ci_bayes = posterior.interval(0.95)
print(f"[베이지안] 95% 신용구간: ({ci_bayes[0]:.3f}, {ci_bayes[1]:.3f})")
```

### 1.3 해석의 차이

```python
# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

p_range = np.linspace(0, 1, 200)

# 빈도주의: 가능도 함수
likelihood = stats.binom.pmf(n_heads, n_trials, p_range)
likelihood = likelihood / likelihood.max()  # 정규화

axes[0].plot(p_range, likelihood, 'b-', lw=2, label='Likelihood')
axes[0].axvline(p_mle, color='r', linestyle='--', label=f'MLE = {p_mle:.2f}')
axes[0].axvspan(ci_freq[0], ci_freq[1], alpha=0.3, color='r', label='95% CI')
axes[0].set_xlabel('p (동전의 앞면 확률)')
axes[0].set_ylabel('정규화된 가능도')
axes[0].set_title('빈도주의: 가능도 함수와 신뢰구간')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 베이지안: 사후분포
prior_pdf = stats.beta(alpha_prior, beta_prior).pdf(p_range)
posterior_pdf = posterior.pdf(p_range)

axes[1].plot(p_range, prior_pdf, 'g--', lw=2, label='Prior: Beta(1,1)')
axes[1].plot(p_range, posterior_pdf, 'b-', lw=2, label=f'Posterior: Beta({alpha_post},{beta_post})')
axes[1].axvline(p_bayes, color='r', linestyle='--', label=f'사후 평균 = {p_bayes:.2f}')
axes[1].axvspan(ci_bayes[0], ci_bayes[1], alpha=0.3, color='b', label='95% Credible Interval')
axes[1].set_xlabel('p (동전의 앞면 확률)')
axes[1].set_ylabel('밀도')
axes[1].set_title('베이지안: 사전분포와 사후분포')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 2. 베이즈 정리 (Bayes' Theorem)

### 2.1 베이즈 정리의 유도

조건부 확률의 정의에서 출발:

$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

$$P(B|A) = \frac{P(A \cap B)}{P(A)}$$

이 두 식에서 P(A ∩ B)를 소거하면:

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

**모수 θ와 데이터 D의 관계로 표현:**

$$P(\theta|D) = \frac{P(D|\theta) \cdot P(\theta)}{P(D)}$$

### 2.2 용어 정리

| 용어 | 수식 | 의미 |
|------|------|------|
| **사후분포 (Posterior)** | P(θ\|D) | 데이터를 관측한 후 모수에 대한 믿음 |
| **가능도 (Likelihood)** | P(D\|θ) | 주어진 모수에서 데이터가 관측될 확률 |
| **사전분포 (Prior)** | P(θ) | 데이터 관측 전 모수에 대한 사전 지식 |
| **주변 가능도 (Evidence)** | P(D) | 정규화 상수 (모든 θ에 대한 적분) |

### 2.3 베이즈 정리 구현

```python
def bayes_theorem_discrete(prior: dict, likelihood: dict) -> dict:
    """
    이산형 베이즈 정리 계산

    Parameters:
    -----------
    prior : dict
        사전확률 {가설: 확률}
    likelihood : dict
        가능도 {가설: P(데이터|가설)}

    Returns:
    --------
    posterior : dict
        사후확률 {가설: P(가설|데이터)}
    """
    # 비정규화 사후확률
    unnormalized = {h: prior[h] * likelihood[h] for h in prior}

    # 주변 가능도 (정규화 상수)
    evidence = sum(unnormalized.values())

    # 정규화
    posterior = {h: p / evidence for h, p in unnormalized.items()}

    return posterior

# 예제: 질병 진단
# 가설: 질병이 있다(D), 질병이 없다(~D)
prior = {
    'D': 0.001,   # 유병률 0.1%
    '~D': 0.999
}

# 검사 결과가 양성(+)일 때의 가능도
likelihood = {
    'D': 0.99,    # 민감도: P(+|D) = 99%
    '~D': 0.05    # 위양성률: P(+|~D) = 5%
}

posterior = bayes_theorem_discrete(prior, likelihood)

print("=== 질병 진단 베이즈 업데이트 ===")
print(f"사전확률 P(질병) = {prior['D']:.4f}")
print(f"가능도 P(양성|질병) = {likelihood['D']:.2f}")
print(f"가능도 P(양성|정상) = {likelihood['~D']:.2f}")
print(f"\n검사 양성 후 사후확률:")
print(f"P(질병|양성) = {posterior['D']:.4f} ({posterior['D']*100:.2f}%)")
print(f"P(정상|양성) = {posterior['~D']:.4f}")
```

### 2.4 연속적인 업데이트

```python
def sequential_bayes_update(prior_dist, data_points, likelihood_func):
    """
    순차적 베이즈 업데이트 (데이터가 하나씩 들어올 때)

    Parameters:
    -----------
    prior_dist : scipy.stats distribution
        초기 사전분포
    data_points : array-like
        관측된 데이터
    likelihood_func : callable
        likelihood_func(x, theta) -> P(x|theta)
    """
    theta_range = np.linspace(0, 1, 1000)
    current_prior = prior_dist.pdf(theta_range)

    history = [current_prior.copy()]

    for x in data_points:
        # 가능도 계산
        likelihood = likelihood_func(x, theta_range)

        # 비정규화 사후분포
        unnormalized_posterior = current_prior * likelihood

        # 정규화 (수치적 적분)
        posterior = unnormalized_posterior / np.trapz(unnormalized_posterior, theta_range)

        history.append(posterior.copy())
        current_prior = posterior

    return theta_range, history

# 동전 던지기 시뮬레이션
np.random.seed(123)
true_p = 0.7
data = np.random.binomial(1, true_p, size=20)  # 20번 던지기

# 베르누이 가능도
def bernoulli_likelihood(x, theta):
    return theta**x * (1 - theta)**(1 - x)

# 순차적 업데이트
theta_range, history = sequential_bayes_update(
    stats.beta(1, 1),  # 균등 사전분포
    data,
    bernoulli_likelihood
)

# 시각화
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

update_points = [0, 1, 5, 10, 15, 20]
for i, n in enumerate(update_points):
    axes[i].plot(theta_range, history[n], 'b-', lw=2)
    axes[i].axvline(true_p, color='r', linestyle='--', alpha=0.7, label=f'True p = {true_p}')
    axes[i].fill_between(theta_range, history[n], alpha=0.3)
    axes[i].set_xlabel('θ')
    axes[i].set_ylabel('밀도')
    axes[i].set_title(f'n = {n} 관측 후')
    axes[i].legend()
    axes[i].set_xlim(0, 1)

plt.tight_layout()
plt.suptitle('순차적 베이즈 업데이트: 사후분포의 변화', y=1.02)
plt.show()

print(f"데이터: {data}")
print(f"총 앞면 수: {data.sum()}, 전체: {len(data)}")
```

---

## 3. 사전분포, 가능도, 사후분포

### 3.1 사전분포의 종류

```python
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
x = np.linspace(0, 1, 200)

# 1. 무정보 사전분포 (Non-informative Prior)
ax = axes[0, 0]
uniform_prior = stats.beta(1, 1).pdf(x)
ax.plot(x, uniform_prior, 'b-', lw=2)
ax.set_title('무정보 사전분포\nBeta(1, 1) = Uniform(0, 1)')
ax.set_xlabel('θ')
ax.set_ylabel('밀도')
ax.fill_between(x, uniform_prior, alpha=0.3)

# 2. Jeffreys 사전분포
ax = axes[0, 1]
jeffreys_prior = stats.beta(0.5, 0.5).pdf(x)
ax.plot(x, jeffreys_prior, 'b-', lw=2)
ax.set_title('Jeffreys 사전분포\nBeta(0.5, 0.5)')
ax.set_xlabel('θ')
ax.set_ylabel('밀도')
ax.fill_between(x, jeffreys_prior, alpha=0.3)

# 3. 약한 정보 사전분포 (Weakly Informative)
ax = axes[0, 2]
weak_prior = stats.beta(2, 2).pdf(x)
ax.plot(x, weak_prior, 'b-', lw=2)
ax.set_title('약한 정보 사전분포\nBeta(2, 2)')
ax.set_xlabel('θ')
ax.set_ylabel('밀도')
ax.fill_between(x, weak_prior, alpha=0.3)

# 4. 정보적 사전분포 (Informative Prior) - 0.3 중심
ax = axes[1, 0]
informative_prior1 = stats.beta(3, 7).pdf(x)
ax.plot(x, informative_prior1, 'b-', lw=2)
ax.set_title('정보적 사전분포 (낮은 p)\nBeta(3, 7), 평균=0.3')
ax.set_xlabel('θ')
ax.set_ylabel('밀도')
ax.fill_between(x, informative_prior1, alpha=0.3)

# 5. 정보적 사전분포 - 0.7 중심
ax = axes[1, 1]
informative_prior2 = stats.beta(7, 3).pdf(x)
ax.plot(x, informative_prior2, 'b-', lw=2)
ax.set_title('정보적 사전분포 (높은 p)\nBeta(7, 3), 평균=0.7')
ax.set_xlabel('θ')
ax.set_ylabel('밀도')
ax.fill_between(x, informative_prior2, alpha=0.3)

# 6. 강한 정보적 사전분포
ax = axes[1, 2]
strong_prior = stats.beta(30, 30).pdf(x)
ax.plot(x, strong_prior, 'b-', lw=2)
ax.set_title('강한 정보적 사전분포\nBeta(30, 30), 평균=0.5')
ax.set_xlabel('θ')
ax.set_ylabel('밀도')
ax.fill_between(x, strong_prior, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 3.2 사전분포의 영향

```python
# 같은 데이터, 다른 사전분포
n_trials, n_heads = 10, 7

priors = [
    ('무정보 Beta(1,1)', 1, 1),
    ('Jeffreys Beta(0.5,0.5)', 0.5, 0.5),
    ('약정보 Beta(2,2)', 2, 2),
    ('정보적 Beta(5,5)', 5, 5),
    ('강정보 Beta(20,20)', 20, 20),
]

fig, axes = plt.subplots(1, 5, figsize=(18, 4))
theta = np.linspace(0, 1, 200)

for i, (name, a, b) in enumerate(priors):
    # 사전분포
    prior = stats.beta(a, b)

    # 사후분포
    a_post = a + n_heads
    b_post = b + (n_trials - n_heads)
    posterior = stats.beta(a_post, b_post)

    # 시각화
    axes[i].plot(theta, prior.pdf(theta), 'g--', lw=2, label='Prior')
    axes[i].plot(theta, posterior.pdf(theta), 'b-', lw=2, label='Posterior')
    axes[i].axvline(0.7, color='r', linestyle=':', alpha=0.7, label='MLE')
    axes[i].axvline(posterior.mean(), color='purple', linestyle='--',
                    alpha=0.7, label=f'Post Mean={posterior.mean():.2f}')
    axes[i].set_xlabel('θ')
    axes[i].set_title(f'{name}')
    axes[i].legend(fontsize=8)
    axes[i].set_xlim(0, 1)

plt.suptitle(f'사전분포의 영향 (데이터: {n_heads}/{n_trials} 성공)', y=1.02)
plt.tight_layout()
plt.show()
```

### 3.3 가능도 함수

```python
def plot_likelihood_function(data, distribution='binomial'):
    """
    가능도 함수 시각화
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if distribution == 'binomial':
        n = len(data)
        k = sum(data)
        theta_range = np.linspace(0.001, 0.999, 200)

        # 가능도 함수: L(θ) = θ^k * (1-θ)^(n-k)
        likelihood = theta_range**k * (1 - theta_range)**(n - k)
        log_likelihood = k * np.log(theta_range) + (n - k) * np.log(1 - theta_range)

        # MLE
        mle = k / n

        # 가능도 함수
        axes[0].plot(theta_range, likelihood, 'b-', lw=2)
        axes[0].axvline(mle, color='r', linestyle='--', label=f'MLE = {mle:.3f}')
        axes[0].set_xlabel('θ')
        axes[0].set_ylabel('L(θ)')
        axes[0].set_title('가능도 함수 (Likelihood)')
        axes[0].legend()
        axes[0].fill_between(theta_range, likelihood, alpha=0.3)

        # 로그 가능도 함수
        axes[1].plot(theta_range, log_likelihood, 'b-', lw=2)
        axes[1].axvline(mle, color='r', linestyle='--', label=f'MLE = {mle:.3f}')
        axes[1].set_xlabel('θ')
        axes[1].set_ylabel('log L(θ)')
        axes[1].set_title('로그 가능도 함수 (Log-Likelihood)')
        axes[1].legend()

    plt.tight_layout()
    plt.show()

    return mle

# 예제
data = [1, 1, 1, 0, 1, 0, 1, 1, 0, 1]  # 7/10 성공
mle = plot_likelihood_function(data)
print(f"데이터: {data}")
print(f"성공 수: {sum(data)}, 시행 수: {len(data)}")
print(f"MLE: {mle}")
```

---

## 4. 켤레 사전분포 (Conjugate Priors)

### 4.1 켤레 사전분포란?

**정의**: 사전분포와 사후분포가 같은 분포족에 속하면 "켤레(conjugate)"라고 합니다.

**장점**:
- 사후분포를 해석적으로(closed-form) 계산 가능
- 수치적 방법 불필요
- 순차적 업데이트가 간단

### 4.2 Beta-Binomial 켤레

**모델**:
- 가능도: X ~ Binomial(n, θ)
- 사전분포: θ ~ Beta(α, β)
- 사후분포: θ|X ~ Beta(α + x, β + n - x)

```python
class BetaBinomialModel:
    """Beta-Binomial 켤레 모델"""

    def __init__(self, alpha_prior=1, beta_prior=1):
        """
        Parameters:
        -----------
        alpha_prior : float
            Beta 사전분포의 alpha 파라미터
        beta_prior : float
            Beta 사전분포의 beta 파라미터
        """
        self.alpha = alpha_prior
        self.beta = beta_prior
        self.n_observations = 0
        self.n_successes = 0

    @property
    def prior(self):
        return stats.beta(self.alpha, self.beta)

    @property
    def posterior(self):
        return stats.beta(
            self.alpha + self.n_successes,
            self.beta + self.n_observations - self.n_successes
        )

    def update(self, n_successes, n_trials):
        """데이터로 사후분포 업데이트"""
        self.n_successes += n_successes
        self.n_observations += n_trials
        return self

    def posterior_mean(self):
        return self.posterior.mean()

    def posterior_std(self):
        return self.posterior.std()

    def credible_interval(self, alpha=0.05):
        """신용구간 계산"""
        return self.posterior.interval(1 - alpha)

    def posterior_predictive(self, n_trials):
        """
        사후 예측분포: 새로운 n_trials 시행에서 성공 수
        Beta-Binomial 분포
        """
        a = self.alpha + self.n_successes
        b = self.beta + self.n_observations - self.n_successes
        return stats.betabinom(n_trials, a, b)

    def summary(self):
        """모델 요약"""
        print("=== Beta-Binomial Model Summary ===")
        print(f"Prior: Beta({self.alpha}, {self.beta})")
        print(f"Data: {self.n_successes} successes / {self.n_observations} trials")
        print(f"Posterior: Beta({self.alpha + self.n_successes}, "
              f"{self.beta + self.n_observations - self.n_successes})")
        print(f"Posterior Mean: {self.posterior_mean():.4f}")
        print(f"Posterior Std: {self.posterior_std():.4f}")
        ci = self.credible_interval()
        print(f"95% Credible Interval: ({ci[0]:.4f}, {ci[1]:.4f})")


# 예제: 클릭률 추정
model = BetaBinomialModel(alpha_prior=2, beta_prior=8)  # 사전: 평균 20% 클릭률

# 1차 데이터: 100명 중 35명 클릭
model.update(35, 100)
print("1차 데이터 후:")
model.summary()

# 2차 데이터: 추가 50명 중 22명 클릭
model.update(22, 50)
print("\n2차 데이터 후:")
model.summary()

# 시각화
theta = np.linspace(0, 1, 200)

fig, ax = plt.subplots(figsize=(10, 6))

# 사전분포
prior = stats.beta(2, 8)
ax.plot(theta, prior.pdf(theta), 'g--', lw=2, label='Prior: Beta(2,8)')

# 중간 사후분포 (1차 데이터 후)
post1 = stats.beta(2 + 35, 8 + 100 - 35)
ax.plot(theta, post1.pdf(theta), 'orange', lw=2, linestyle='-.',
        label='After 1st data: Beta(37, 73)')

# 최종 사후분포
ax.plot(theta, model.posterior.pdf(theta), 'b-', lw=2,
        label=f'Final Posterior: Beta(59, 95)')

ax.fill_between(theta, model.posterior.pdf(theta), alpha=0.3)
ax.axvline(model.posterior_mean(), color='r', linestyle=':',
           label=f'Posterior Mean = {model.posterior_mean():.3f}')

ax.set_xlabel('θ (클릭률)')
ax.set_ylabel('밀도')
ax.set_title('Beta-Binomial 켤레: 클릭률 추정')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

### 4.3 Normal-Normal 켤레 (알려진 분산)

**모델**:
- 가능도: X ~ N(μ, σ²)  (σ² 알려짐)
- 사전분포: μ ~ N(μ₀, σ₀²)
- 사후분포: μ|X ~ N(μₙ, σₙ²)

$$\mu_n = \frac{\frac{\mu_0}{\sigma_0^2} + \frac{n\bar{x}}{\sigma^2}}{\frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}}$$

$$\sigma_n^2 = \frac{1}{\frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}}$$

```python
class NormalNormalModel:
    """Normal-Normal 켤레 모델 (알려진 분산)"""

    def __init__(self, mu_prior=0, sigma_prior=10, sigma_known=1):
        """
        Parameters:
        -----------
        mu_prior : float
            사전분포 평균
        sigma_prior : float
            사전분포 표준편차
        sigma_known : float
            알려진 데이터 표준편차
        """
        self.mu_0 = mu_prior
        self.sigma_0 = sigma_prior
        self.sigma = sigma_known
        self.data = []

    @property
    def n(self):
        return len(self.data)

    @property
    def x_bar(self):
        return np.mean(self.data) if self.data else 0

    @property
    def prior(self):
        return stats.norm(self.mu_0, self.sigma_0)

    @property
    def posterior_precision(self):
        """사후 정밀도 (분산의 역수)"""
        return 1/self.sigma_0**2 + self.n/self.sigma**2

    @property
    def posterior_variance(self):
        return 1 / self.posterior_precision

    @property
    def posterior_mean(self):
        if self.n == 0:
            return self.mu_0
        return (self.mu_0/self.sigma_0**2 + self.n*self.x_bar/self.sigma**2) / \
               self.posterior_precision

    @property
    def posterior_std(self):
        return np.sqrt(self.posterior_variance)

    @property
    def posterior(self):
        return stats.norm(self.posterior_mean, self.posterior_std)

    def update(self, data):
        """데이터 추가"""
        if isinstance(data, (list, np.ndarray)):
            self.data.extend(data)
        else:
            self.data.append(data)
        return self

    def credible_interval(self, alpha=0.05):
        return self.posterior.interval(1 - alpha)

    def summary(self):
        print("=== Normal-Normal Model Summary ===")
        print(f"Prior: N({self.mu_0}, {self.sigma_0}²)")
        print(f"Known σ: {self.sigma}")
        print(f"Data: n={self.n}, x̄={self.x_bar:.4f}" if self.n > 0 else "Data: none")
        print(f"Posterior: N({self.posterior_mean:.4f}, {self.posterior_std:.4f}²)")
        ci = self.credible_interval()
        print(f"95% Credible Interval: ({ci[0]:.4f}, {ci[1]:.4f})")


# 예제: 체온 측정
# 사전 지식: 평균 체온은 대략 36.5도, 불확실성 σ=0.5
# 측정 오차 σ=0.2 (알려진 값)

model = NormalNormalModel(mu_prior=36.5, sigma_prior=0.5, sigma_known=0.2)
print("사전분포:")
model.summary()

# 데이터 수집: 환자 체온 측정
measurements = [37.1, 37.3, 36.9, 37.2, 37.0]
model.update(measurements)
print("\n데이터 업데이트 후:")
model.summary()

# 시각화
mu_range = np.linspace(35.5, 38, 200)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(mu_range, model.prior.pdf(mu_range), 'g--', lw=2,
        label=f'Prior: N({model.mu_0}, {model.sigma_0}²)')
ax.plot(mu_range, model.posterior.pdf(mu_range), 'b-', lw=2,
        label=f'Posterior: N({model.posterior_mean:.2f}, {model.posterior_std:.3f}²)')
ax.fill_between(mu_range, model.posterior.pdf(mu_range), alpha=0.3)

# 개별 측정값 표시
for i, x in enumerate(measurements):
    ax.axvline(x, color='orange', linestyle=':', alpha=0.5,
               label='Measurements' if i == 0 else '')

ax.axvline(np.mean(measurements), color='r', linestyle='--',
           label=f'Sample Mean = {np.mean(measurements):.2f}')

ax.set_xlabel('μ (체온)')
ax.set_ylabel('밀도')
ax.set_title('Normal-Normal 켤레: 체온 추정')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

### 4.4 Gamma-Poisson 켤레

**모델**:
- 가능도: X ~ Poisson(λ)
- 사전분포: λ ~ Gamma(α, β)
- 사후분포: λ|X ~ Gamma(α + Σxᵢ, β + n)

```python
class GammaPoissonModel:
    """Gamma-Poisson 켤레 모델"""

    def __init__(self, alpha_prior=1, beta_prior=1):
        """
        Parameters:
        -----------
        alpha_prior : float (shape)
        beta_prior : float (rate)
        """
        self.alpha = alpha_prior
        self.beta = beta_prior
        self.data = []

    @property
    def n(self):
        return len(self.data)

    @property
    def sum_x(self):
        return sum(self.data)

    @property
    def prior(self):
        return stats.gamma(self.alpha, scale=1/self.beta)

    @property
    def posterior(self):
        alpha_post = self.alpha + self.sum_x
        beta_post = self.beta + self.n
        return stats.gamma(alpha_post, scale=1/beta_post)

    def update(self, data):
        if isinstance(data, (list, np.ndarray)):
            self.data.extend(data)
        else:
            self.data.append(data)
        return self

    def posterior_mean(self):
        return (self.alpha + self.sum_x) / (self.beta + self.n)

    def posterior_std(self):
        return self.posterior.std()

    def credible_interval(self, alpha=0.05):
        return self.posterior.interval(1 - alpha)

    def posterior_predictive(self):
        """
        사후 예측분포: Negative Binomial
        """
        alpha_post = self.alpha + self.sum_x
        beta_post = self.beta + self.n
        p = beta_post / (beta_post + 1)
        return stats.nbinom(alpha_post, p)

    def summary(self):
        print("=== Gamma-Poisson Model Summary ===")
        print(f"Prior: Gamma({self.alpha}, {self.beta})")
        print(f"Data: n={self.n}, Σx={self.sum_x}")
        alpha_post = self.alpha + self.sum_x
        beta_post = self.beta + self.n
        print(f"Posterior: Gamma({alpha_post}, {beta_post})")
        print(f"Posterior Mean (λ): {self.posterior_mean():.4f}")
        print(f"Posterior Std: {self.posterior_std():.4f}")
        ci = self.credible_interval()
        print(f"95% Credible Interval: ({ci[0]:.4f}, {ci[1]:.4f})")


# 예제: 콜센터 통화 수 추정
# 사전: 시간당 약 10통 예상, 불확실성 있음
model = GammaPoissonModel(alpha_prior=10, beta_prior=1)  # 평균 10, 분산 10
print("사전분포:")
model.summary()

# 5시간 동안의 통화 수
calls = [8, 12, 15, 9, 11]
model.update(calls)
print("\n데이터 후:")
model.summary()

# 시각화
lambda_range = np.linspace(0, 25, 200)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 사후분포
ax = axes[0]
ax.plot(lambda_range, model.prior.pdf(lambda_range), 'g--', lw=2, label='Prior')
ax.plot(lambda_range, model.posterior.pdf(lambda_range), 'b-', lw=2, label='Posterior')
ax.fill_between(lambda_range, model.posterior.pdf(lambda_range), alpha=0.3)
ax.axvline(model.posterior_mean(), color='r', linestyle='--',
           label=f'Post Mean = {model.posterior_mean():.2f}')
ax.axvline(np.mean(calls), color='orange', linestyle=':',
           label=f'Sample Mean = {np.mean(calls):.1f}')
ax.set_xlabel('λ (시간당 통화 수)')
ax.set_ylabel('밀도')
ax.set_title('Gamma-Poisson 켤레: 통화율 추정')
ax.legend()
ax.grid(True, alpha=0.3)

# 사후 예측분포
ax = axes[1]
x_pred = np.arange(0, 30)
pred_dist = model.posterior_predictive()
ax.bar(x_pred, pred_dist.pmf(x_pred), alpha=0.6, label='Posterior Predictive')
ax.axvline(pred_dist.mean(), color='r', linestyle='--',
           label=f'Expected = {pred_dist.mean():.1f}')
ax.set_xlabel('다음 시간 통화 수')
ax.set_ylabel('확률')
ax.set_title('사후 예측분포')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 4.5 켤레 사전분포 요약표

| 가능도 | 사전분포 | 사후분포 | 업데이트 규칙 |
|--------|----------|----------|---------------|
| Binomial(n, p) | Beta(α, β) | Beta(α+x, β+n-x) | α←α+x, β←β+(n-x) |
| Poisson(λ) | Gamma(α, β) | Gamma(α+Σx, β+n) | α←α+Σx, β←β+n |
| Normal(μ, σ²) | Normal(μ₀, σ₀²) | Normal(μₙ, σₙ²) | 정밀도 가중 평균 |
| Exponential(λ) | Gamma(α, β) | Gamma(α+n, β+Σx) | α←α+n, β←β+Σx |
| Multinomial | Dirichlet | Dirichlet | α←α+counts |

---

## 5. MAP 추정 (Maximum A Posteriori)

### 5.1 MAP vs MLE

**MLE**: 가능도를 최대화
$$\hat{\theta}_{MLE} = \arg\max_\theta P(D|\theta)$$

**MAP**: 사후분포를 최대화
$$\hat{\theta}_{MAP} = \arg\max_\theta P(\theta|D) = \arg\max_\theta P(D|\theta)P(\theta)$$

```python
def compare_mle_map(n_trials, n_successes, alpha_prior, beta_prior):
    """MLE와 MAP 비교"""

    # MLE
    mle = n_successes / n_trials

    # MAP for Beta-Binomial
    # posterior: Beta(alpha + k, beta + n - k)
    # mode of Beta(a, b) = (a - 1) / (a + b - 2) when a, b > 1
    a = alpha_prior + n_successes
    b = beta_prior + (n_trials - n_successes)

    if a > 1 and b > 1:
        map_est = (a - 1) / (a + b - 2)
    else:
        # 모드가 경계에 있는 경우
        map_est = a / (a + b)  # 사후 평균 사용

    # 사후 평균
    posterior_mean = a / (a + b)

    return mle, map_est, posterior_mean

# 다양한 사전분포와 데이터 크기에서 비교
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

scenarios = [
    (10, 7, 'n=10, k=7'),
    (100, 70, 'n=100, k=70'),
    (10, 3, 'n=10, k=3'),
    (100, 30, 'n=100, k=30'),
]

priors = [
    ('Uniform', 1, 1),
    ('Weak', 2, 2),
    ('Informative 0.5', 10, 10),
    ('Strong 0.5', 50, 50),
]

for i, (n, k, title) in enumerate(scenarios):
    ax = axes.flatten()[i]

    x = np.arange(len(priors))
    width = 0.25

    mles, maps, means = [], [], []
    for name, a, b in priors:
        mle, map_est, post_mean = compare_mle_map(n, k, a, b)
        mles.append(mle)
        maps.append(map_est)
        means.append(post_mean)

    ax.bar(x - width, mles, width, label='MLE', alpha=0.8)
    ax.bar(x, maps, width, label='MAP', alpha=0.8)
    ax.bar(x + width, means, width, label='Posterior Mean', alpha=0.8)

    ax.axhline(k/n, color='r', linestyle='--', alpha=0.5, label=f'True ratio = {k/n:.2f}')
    ax.set_xticks(x)
    ax.set_xticklabels([p[0] for p in priors], rotation=15)
    ax.set_ylabel('추정치')
    ax.set_title(title)
    ax.legend(loc='best', fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('MLE vs MAP vs Posterior Mean: 사전분포의 영향', y=1.02)
plt.tight_layout()
plt.show()
```

### 5.2 MAP 추정의 수치적 계산

```python
from scipy.optimize import minimize_scalar, minimize

def map_estimation_normal(data, prior_mean, prior_std, known_std):
    """
    정규분포 데이터의 MAP 추정 (수치적 방법)
    """
    n = len(data)
    x_bar = np.mean(data)

    # 음의 로그 사후확률 (최소화)
    def neg_log_posterior(mu):
        # 로그 가능도
        log_likelihood = -n * np.log(known_std) - \
                         0.5 * np.sum((data - mu)**2) / known_std**2
        # 로그 사전확률
        log_prior = -0.5 * ((mu - prior_mean)**2) / prior_std**2

        return -(log_likelihood + log_prior)

    # 최적화
    result = minimize_scalar(neg_log_posterior, bounds=(x_bar - 3*known_std, x_bar + 3*known_std))

    return result.x

# 예제
np.random.seed(42)
true_mu = 5
data = np.random.normal(true_mu, 1, 10)

# 다양한 사전분포로 MAP 추정
priors = [
    (5, 100),   # 매우 약한 사전분포
    (5, 2),     # 약한 사전분포
    (0, 1),     # 잘못된 강한 사전분포
    (5, 0.5),   # 올바른 강한 사전분포
]

print(f"True μ: {true_mu}")
print(f"Sample mean: {np.mean(data):.4f}")
print(f"Sample size: {len(data)}")
print("\nMAP estimates with different priors:")

for prior_mean, prior_std in priors:
    map_est = map_estimation_normal(data, prior_mean, prior_std, known_std=1)
    print(f"Prior N({prior_mean}, {prior_std}²): MAP = {map_est:.4f}")
```

### 5.3 MAP과 정규화의 관계

```python
# MAP 추정은 정규화(regularization)와 동등
# Beta(α, β) prior for Binomial → L2 정규화와 유사한 효과

def show_map_regularization_connection():
    """MAP과 정규화의 연결"""

    print("=== MAP과 정규화의 관계 ===")
    print()
    print("1. 로지스틱 회귀에서:")
    print("   - 정규 사전분포 N(0, σ²) → L2 정규화 (Ridge)")
    print("   - 라플라스 사전분포 → L1 정규화 (Lasso)")
    print()
    print("2. 선형 회귀에서:")
    print("   - MAP with N(0, σ²) prior ≡ Ridge regression")
    print("   - λ = σ²_noise / σ²_prior")
    print()
    print("3. 베이지안 관점:")
    print("   - 정규화 강도 = 사전분포의 확신도")
    print("   - 강한 사전분포 → 강한 정규화")
    print("   - 무정보 사전분포 → 정규화 없음 (MLE)")

show_map_regularization_connection()

# 시각적 예시
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Ridge vs MAP
theta = np.linspace(-3, 3, 200)

# Ridge 관점: L2 패널티
ridge_penalty = theta**2
axes[0].plot(theta, ridge_penalty, 'b-', lw=2, label='L2 penalty: λθ²')
axes[0].set_xlabel('θ')
axes[0].set_ylabel('Penalty')
axes[0].set_title('Ridge Regression: L2 Penalty')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# MAP 관점: 정규 사전분포
prior_pdf = stats.norm(0, 1).pdf(theta)
neg_log_prior = -np.log(prior_pdf + 1e-10)
axes[1].plot(theta, neg_log_prior, 'r-', lw=2, label='-log P(θ) for N(0, 1)')
axes[1].set_xlabel('θ')
axes[1].set_ylabel('-log Prior')
axes[1].set_title('MAP: Negative Log Prior')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 6. 실습 예제

### 6.1 A/B 테스트의 베이지안 분석

```python
class BayesianABTest:
    """베이지안 A/B 테스트"""

    def __init__(self, alpha_prior=1, beta_prior=1):
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior

    def fit(self, successes_A, trials_A, successes_B, trials_B):
        """데이터 적합"""
        self.successes_A = successes_A
        self.trials_A = trials_A
        self.successes_B = successes_B
        self.trials_B = trials_B

        # 사후분포
        self.posterior_A = stats.beta(
            self.alpha_prior + successes_A,
            self.beta_prior + trials_A - successes_A
        )
        self.posterior_B = stats.beta(
            self.alpha_prior + successes_B,
            self.beta_prior + trials_B - successes_B
        )

    def prob_B_better(self, n_samples=100000):
        """P(B > A) 시뮬레이션으로 계산"""
        samples_A = self.posterior_A.rvs(n_samples)
        samples_B = self.posterior_B.rvs(n_samples)
        return (samples_B > samples_A).mean()

    def expected_lift(self, n_samples=100000):
        """기대 상승률"""
        samples_A = self.posterior_A.rvs(n_samples)
        samples_B = self.posterior_B.rvs(n_samples)
        lift = (samples_B - samples_A) / samples_A
        return lift.mean(), np.percentile(lift, [2.5, 97.5])

    def summary(self):
        print("=== Bayesian A/B Test Results ===")
        print(f"\nGroup A: {self.successes_A}/{self.trials_A} "
              f"({self.successes_A/self.trials_A*100:.1f}%)")
        print(f"Group B: {self.successes_B}/{self.trials_B} "
              f"({self.successes_B/self.trials_B*100:.1f}%)")

        print(f"\nPosterior Mean A: {self.posterior_A.mean():.4f}")
        print(f"Posterior Mean B: {self.posterior_B.mean():.4f}")

        prob_b_better = self.prob_B_better()
        print(f"\nP(B > A): {prob_b_better:.4f} ({prob_b_better*100:.1f}%)")

        lift_mean, lift_ci = self.expected_lift()
        print(f"Expected Lift: {lift_mean*100:.2f}%")
        print(f"95% CI for Lift: ({lift_ci[0]*100:.2f}%, {lift_ci[1]*100:.2f}%)")

    def plot(self):
        """결과 시각화"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        p_range = np.linspace(0, 1, 200)

        # 사후분포 비교
        ax = axes[0]
        ax.plot(p_range, self.posterior_A.pdf(p_range), 'b-', lw=2, label='A')
        ax.plot(p_range, self.posterior_B.pdf(p_range), 'r-', lw=2, label='B')
        ax.fill_between(p_range, self.posterior_A.pdf(p_range), alpha=0.3)
        ax.fill_between(p_range, self.posterior_B.pdf(p_range), alpha=0.3)
        ax.set_xlabel('전환율')
        ax.set_ylabel('밀도')
        ax.set_title('전환율의 사후분포')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 차이 분포
        ax = axes[1]
        samples_A = self.posterior_A.rvs(50000)
        samples_B = self.posterior_B.rvs(50000)
        diff = samples_B - samples_A
        ax.hist(diff, bins=50, density=True, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='r', linestyle='--', label='No difference')
        ax.axvline(diff.mean(), color='g', linestyle='-',
                   label=f'Mean diff = {diff.mean():.4f}')
        ax.set_xlabel('B - A')
        ax.set_ylabel('밀도')
        ax.set_title('전환율 차이 분포')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 상승률 분포
        ax = axes[2]
        lift = (samples_B - samples_A) / samples_A
        lift = lift[(lift > -1) & (lift < 2)]  # 극단값 제거
        ax.hist(lift, bins=50, density=True, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='r', linestyle='--', label='No lift')
        ax.axvline(np.median(lift), color='g', linestyle='-',
                   label=f'Median = {np.median(lift)*100:.1f}%')
        ax.set_xlabel('상승률 (B-A)/A')
        ax.set_ylabel('밀도')
        ax.set_title('상승률 분포')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


# 예제: 웹사이트 버튼 색상 A/B 테스트
ab_test = BayesianABTest(alpha_prior=1, beta_prior=1)
ab_test.fit(
    successes_A=48, trials_A=500,   # 기존 버튼: 48/500 클릭
    successes_B=63, trials_B=500    # 새 버튼: 63/500 클릭
)
ab_test.summary()
ab_test.plot()
```

### 6.2 품질 관리의 베이지안 접근

```python
def bayesian_quality_control():
    """제조 품질 관리의 베이지안 분석"""

    # 시나리오: 불량률 추정
    # 사전 지식: 과거 경험으로 불량률 약 5%, 신뢰도 중간

    # 사전분포: Beta(2, 38) → 평균 5%, 표본크기 40에 해당하는 정보
    alpha_prior = 2
    beta_prior = 38
    prior = stats.beta(alpha_prior, beta_prior)

    print("=== 제조 품질 관리 베이지안 분석 ===")
    print(f"사전분포: Beta({alpha_prior}, {beta_prior})")
    print(f"사전 평균 (예상 불량률): {prior.mean()*100:.1f}%")
    print(f"사전 95% 구간: ({prior.ppf(0.025)*100:.2f}%, {prior.ppf(0.975)*100:.2f}%)")

    # 새 데이터: 100개 샘플 중 8개 불량
    n_samples = 100
    n_defects = 8

    # 사후분포
    alpha_post = alpha_prior + n_defects
    beta_post = beta_prior + (n_samples - n_defects)
    posterior = stats.beta(alpha_post, beta_post)

    print(f"\n데이터: {n_defects}/{n_samples} 불량")
    print(f"사후분포: Beta({alpha_post}, {beta_post})")
    print(f"사후 평균 (추정 불량률): {posterior.mean()*100:.2f}%")
    ci = posterior.interval(0.95)
    print(f"사후 95% 신용구간: ({ci[0]*100:.2f}%, {ci[1]*100:.2f}%)")

    # 의사결정: 불량률이 10%를 초과할 확률
    prob_exceed_10 = 1 - posterior.cdf(0.10)
    print(f"\nP(불량률 > 10%): {prob_exceed_10*100:.2f}%")

    if prob_exceed_10 > 0.05:
        print("⚠️ 권고: 품질 점검 필요 (10% 초과 확률이 5% 이상)")
    else:
        print("✓ 정상: 품질 기준 충족")

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    p_range = np.linspace(0, 0.25, 200)

    # 사전/사후 비교
    ax = axes[0]
    ax.plot(p_range, prior.pdf(p_range), 'g--', lw=2, label='Prior')
    ax.plot(p_range, posterior.pdf(p_range), 'b-', lw=2, label='Posterior')
    ax.fill_between(p_range, posterior.pdf(p_range), alpha=0.3)
    ax.axvline(0.10, color='r', linestyle=':', lw=2, label='Threshold (10%)')
    ax.axvline(posterior.mean(), color='purple', linestyle='--',
               label=f'Post Mean = {posterior.mean()*100:.1f}%')
    ax.set_xlabel('불량률')
    ax.set_ylabel('밀도')
    ax.set_title('불량률의 사전/사후 분포')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 기준 초과 확률
    ax = axes[1]
    thresholds = np.linspace(0.01, 0.20, 50)
    exceed_probs = [1 - posterior.cdf(t) for t in thresholds]
    ax.plot(thresholds * 100, exceed_probs, 'b-', lw=2)
    ax.axvline(10, color='r', linestyle='--', label='현재 기준 (10%)')
    ax.axhline(0.05, color='orange', linestyle=':', label='유의수준 (5%)')
    ax.set_xlabel('불량률 기준 (%)')
    ax.set_ylabel('초과 확률')
    ax.set_title('불량률 기준에 따른 초과 확률')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

bayesian_quality_control()
```

---

## 7. 연습 문제

### 문제 1: 베이즈 정리 적용
스팸 필터를 구현하려 합니다.
- P(스팸) = 0.3 (30%가 스팸)
- P("무료"|스팸) = 0.8 (스팸 메일의 80%에 "무료" 포함)
- P("무료"|정상) = 0.1 (정상 메일의 10%에 "무료" 포함)

"무료"가 포함된 메일이 스팸일 확률을 계산하세요.

### 문제 2: 켤레 사전분포 선택
다음 상황에 적합한 켤레 사전분포를 선택하고 이유를 설명하세요:
1. 웹사이트 클릭률 추정
2. 콜센터 시간당 통화 수 추정
3. 제품 무게의 평균 추정 (분산은 알려짐)

### 문제 3: 사전분포 민감도 분석
동일한 데이터(100회 시행, 30회 성공)에 대해 다음 사전분포로 MAP 추정을 수행하세요:
1. Beta(1, 1)
2. Beta(5, 5)
3. Beta(1, 9)
4. Beta(9, 1)

각 경우의 MAP 추정치와 MLE를 비교하고, 사전분포가 결과에 미치는 영향을 분석하세요.

### 문제 4: 순차적 업데이트
매일 고객 10명이 방문하고, 3일간의 구매 데이터가 [3, 5, 4]입니다.
Beta(1, 1) 사전분포를 시작으로 매일 데이터가 들어올 때마다 사후분포를 업데이트하세요.
최종 사후분포의 평균과 95% 신용구간을 구하세요.

---

## 8. 핵심 요약

### 베이지안 통계의 핵심 개념

1. **확률의 의미**: 불확실성의 정도 (믿음)
2. **베이즈 정리**: 사전 지식 + 데이터 → 사후 지식
3. **켤레 사전분포**: 해석적 사후분포 계산 가능

### 주요 켤레쌍

```
Binomial + Beta → Beta
Poisson + Gamma → Gamma
Normal(μ, σ²_known) + Normal → Normal
```

### 추정 방법 비교

| 방법 | 목적 | 특징 |
|------|------|------|
| MLE | max P(D\|θ) | 데이터만 사용 |
| MAP | max P(θ\|D) | 사전분포 반영 |
| 사후 평균 | E[θ\|D] | 전체 불확실성 반영 |

### 다음 장 미리보기

09장 **베이지안 추론**에서는:
- MCMC 방법 (Metropolis-Hastings, Gibbs Sampling)
- PyMC를 사용한 베이지안 모델링
- 베이지안 회귀분석
- 모델 비교와 선택
