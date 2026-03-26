# 베이즈 추론

**이전**: [가설 검정](./14_Hypothesis_Testing.md) | **다음**: [비모수적 방법](./16_Nonparametric_Methods.md)

---

## 학습 목표

이 단원을 완료하면 다음을 할 수 있습니다:

1. 베이즈 패러다임을 설명하고 빈도주의 접근법과 대비하기
2. 베이즈 정리를 적용하여 사전분포와 가능도로부터 사후분포 계산하기
3. 일반적인 모형에 대한 켤레 사전분포를 식별하고 사용하기
4. 제프리스 사전분포 (Jeffreys Prior)를 포함한 비정보적 사전분포를 설명하기
5. MAP 추정량과 사후 평균 점 추정량을 계산하기
6. 신용구간 (등꼬리, HPD) 구성하기
7. 사후 예측 분포를 유도하고 해석하기
8. Python에서 베타-이항 갱신 구현하기

---

베이즈 추론 (Bayesian Inference)은 미지의 모수를 확률분포를 가진 확률 변수로 취급하여, 관측된 데이터에 비추어 사전 믿음을 갱신합니다. 이는 사전 지식과 증거를 결합하기 위한 일관된 프레임워크를 제공합니다.

---

## 1. 베이즈 패러다임 vs. 빈도주의

### 1.1 근본적 차이

| 측면 | 빈도주의 | 베이즈 |
|------|----------|--------|
| 모수 | 고정된 미지의 상수 | 분포를 가진 확률 변수 |
| 확률 | 사건의 장기적 빈도 | 믿음의 정도 |
| 추론 | 추정량의 표본 분포에 기반 | 모수의 사후분포에 기반 |
| 사전 정보 | 공식적으로 편입하지 않음 | 사전분포로 부호화 |
| 구간 | 신뢰구간 (포함률 성질) | 신용구간 ($\theta$에 대한 확률 진술) |

### 1.2 베이즈 방법이 선호되는 경우

- 사전 정보가 있고 이를 편입해야 할 때
- 표본 크기가 작고 사전분포가 추정을 안정화할 수 있을 때
- 모수에 대한 직접적 확률 진술이 필요할 때
- 계층적 또는 복잡한 모형이 필요할 때
- 믿음의 순차적 갱신이 자연스러운 경우

---

## 2. 추론을 위한 베이즈 정리

### 2.1 핵심 공식

데이터 $x$와 모수 $\theta$가 주어지면:

$$\pi(\theta \mid x) = \frac{f(x \mid \theta) \, \pi(\theta)}{f(x)} = \frac{f(x \mid \theta) \, \pi(\theta)}{\int f(x \mid \theta) \, \pi(\theta) \, d\theta}$$

줄여서:

$$\text{사후분포} \propto \text{가능도} \times \text{사전분포}$$

$$\pi(\theta \mid x) \propto f(x \mid \theta) \, \pi(\theta)$$

### 2.2 구성 요소

- **사전분포 (Prior)** $\pi(\theta)$: 데이터를 보기 전의 $\theta$에 대한 믿음을 부호화합니다.
- **가능도 (Likelihood)** $f(x \mid \theta) = L(\theta \mid x)$: 주어진 $\theta$ 하에서 관측 데이터의 확률.
- **사후분포 (Posterior)** $\pi(\theta \mid x)$: 데이터 관측 후 갱신된 믿음.
- **주변 가능도 (Marginal Likelihood)** $f(x) = \int f(x \mid \theta)\pi(\theta) \, d\theta$: 정규화 상수 (모형 비교에 중요).

### 2.3 순차적 갱신

베이즈 추론의 강력한 특징은 오늘의 사후분포가 내일의 사전분포가 된다는 것입니다. 데이터 $x_1$을 관측한 후 나중에 $x_2$를 관측하면:

$$\pi(\theta \mid x_1, x_2) \propto f(x_2 \mid \theta) \cdot \underbrace{f(x_1 \mid \theta) \pi(\theta)}_{\propto \, \pi(\theta \mid x_1)}$$

이것은 베이즈 방법을 온라인 또는 스트리밍 데이터에 자연스럽게 적합하게 만듭니다.

---

## 3. 켤레 사전분포

사전분포 $\pi(\theta)$가 가능도 $f(x \mid \theta)$에 대해 **켤레 (Conjugate)**라 함은, 사후분포 $\pi(\theta \mid x)$가 사전분포와 같은 족에 속하는 것입니다. 이는 닫힌 형태의 사후분포 갱신을 제공합니다.

### 3.1 베타-이항

**모형**: $X \mid p \sim \text{Binomial}(n, p)$

**사전분포**: $p \sim \text{Beta}(\alpha, \beta)$

**사후분포**: $p \mid x \sim \text{Beta}(\alpha + x, \, \beta + n - x)$

베타 사전분포는 유연합니다: $\alpha = \beta = 1$은 균등 사전분포; $\alpha, \beta > 1$은 질량을 0과 1로부터 멀리 집중시킵니다.

**사전 평균**: $E[p] = \frac{\alpha}{\alpha + \beta}$

**사후 평균**: $E[p \mid x] = \frac{\alpha + x}{\alpha + \beta + n}$

이것은 사전 평균과 표본 비율 $\hat{p} = x/n$의 가중 평균입니다.

### 3.2 정규-정규

**모형**: $X_1, \ldots, X_n \mid \mu \sim N(\mu, \sigma^2)$이고 $\sigma^2$ 기지.

**사전분포**: $\mu \sim N(\mu_0, \tau^2)$

**사후분포**: $\mu \mid x \sim N(\mu_n, \tau_n^2)$ 여기서:

$$\mu_n = \frac{\frac{\mu_0}{\tau^2} + \frac{n\bar{x}}{\sigma^2}}{\frac{1}{\tau^2} + \frac{n}{\sigma^2}}, \quad \tau_n^2 = \frac{1}{\frac{1}{\tau^2} + \frac{n}{\sigma^2}}$$

사후 평균은 사전 평균과 표본 평균의 **정밀도 가중 평균 (Precision-Weighted Average)**이며, 여기서 정밀도 = $1/\text{분산}$입니다.

### 3.3 감마-포아송

**모형**: $X_1, \ldots, X_n \mid \lambda \sim \text{Poisson}(\lambda)$

**사전분포**: $\lambda \sim \text{Gamma}(\alpha, \beta)$

**사후분포**: $\lambda \mid x \sim \text{Gamma}(\alpha + \sum x_i, \, \beta + n)$

### 3.4 요약 표

| 가능도 | 켤레 사전분포 | 사후분포 |
|--------|--------------|----------|
| Binomial$(n, p)$ | Beta$(\alpha, \beta)$ | Beta$(\alpha + x, \beta + n - x)$ |
| Poisson$(\lambda)$ | Gamma$(\alpha, \beta)$ | Gamma$(\alpha + \sum x_i, \beta + n)$ |
| Normal$(\mu, \sigma^2)$ ($\sigma^2$ 기지) | Normal$(\mu_0, \tau^2)$ | Normal$(\mu_n, \tau_n^2)$ |
| Normal$(\mu, \sigma^2)$ ($\mu$ 기지) | Inverse-Gamma$(a, b)$ | Inverse-Gamma$(a + n/2, b + \sum(x_i-\mu)^2/2)$ |
| Exponential$(\lambda)$ | Gamma$(\alpha, \beta)$ | Gamma$(\alpha + n, \beta + \sum x_i)$ |

---

## 4. 비정보적 사전분포

사전 정보가 없을 때, "데이터가 스스로 말하게" 하는 사전분포를 구합니다.

### 4.1 평탄 (균등) 사전분포

모수 공간에 걸쳐 $\pi(\theta) \propto 1$. 단순하지만 재모수화에 불변이 아닙니다: $\pi(\theta) \propto 1$이면, $\phi = g(\theta)$에 대해 $\pi(\phi) \propto |d\theta/d\phi|$이며, 이는 일반적으로 평탄하지 않습니다.

### 4.2 제프리스 사전분포

제프리스 사전분포 (Jeffreys Prior)는 재모수화에 불변입니다:

$$\pi_J(\theta) \propto \sqrt{I(\theta)}$$

여기서 $I(\theta) = -E\left[\frac{\partial^2 \ln f(X \mid \theta)}{\partial \theta^2}\right]$는 피셔 정보량입니다.

**예시**:
- 이항: $\pi_J(p) \propto p^{-1/2}(1-p)^{-1/2} = \text{Beta}(1/2, 1/2)$
- 정규 평균 ($\sigma$ 기지): $\pi_J(\mu) \propto 1$
- 정규 분산 ($\mu$ 기지): $\pi_J(\sigma^2) \propto 1/\sigma^2$

### 4.3 약정보적 사전분포

실제로 완전한 비정보적 사전분포는 비적절 사후분포나 나쁜 행동을 초래할 수 있습니다. **약정보적 사전분포 (Weakly Informative Priors)**는 데이터를 압도하지 않으면서 약한 정칙화를 제공합니다. 예를 들어, 회귀 계수에 $N(0, 100)$ 사전분포를 사용하면 합리적 범위로 제한하면서 사후분포에 강한 영향을 주지 않습니다.

---

## 5. 점 추정

### 5.1 최대사후확률 추정 (MAP)

$$\hat{\theta}_{\text{MAP}} = \arg\max_\theta \pi(\theta \mid x) = \arg\max_\theta [f(x \mid \theta) \pi(\theta)]$$

MAP는 사전분포를 편입하여 MLE를 일반화합니다. 평탄 사전분포의 경우, MAP = MLE입니다.

### 5.2 사후 평균

$$\hat{\theta}_{\text{PM}} = E[\theta \mid x] = \int \theta \, \pi(\theta \mid x) \, d\theta$$

사후 평균은 기대 제곱 오차 손실 $E[(\theta - a)^2 \mid x]$를 최소화합니다.

### 5.3 사후 중앙값

사후 중앙값은 기대 절대 오차 손실 $E[|\theta - a| \mid x]$를 최소화합니다.

### 5.4 비교

| 추정량 | 최적 조건 | 성질 |
|--------|-----------|------|
| MAP | 0-1 손실 | 사후분포의 최빈값; 다봉 사후분포에서 존재하지 않을 수 있음 |
| 사후 평균 | 제곱 오차 손실 | 적절한 사후분포에서 항상 존재; 꼬리에 민감 |
| 사후 중앙값 | 절대 오차 손실 | 비대칭에 강건 |

대칭 단봉 사후분포에서는 셋 모두 일치합니다.

---

## 6. 신용구간

### 6.1 정의

$(1-\alpha)$ **신용구간 (Credible Interval)** $[a, b]$은 다음을 만족합니다:

$$P(a \leq \theta \leq b \mid x) = 1 - \alpha$$

신뢰구간과 달리, 이것은 $\theta$에 대한 직접적 확률 진술입니다 (데이터와 모형이 주어졌을 때).

### 6.2 등꼬리 구간

$a$와 $b$를 다음과 같이 선택합니다:

$$P(\theta < a \mid x) = \alpha/2, \quad P(\theta > b \mid x) = \alpha/2$$

각 꼬리에 동일한 확률을 배치합니다.

### 6.3 최고사후밀도 (HPD) 구간

HPD 구간은 $1-\alpha$ 확률을 포함하는 가장 짧은 구간입니다:

$$C = \{\theta : \pi(\theta \mid x) \geq c\}$$

여기서 $c$는 $P(\theta \in C \mid x) = 1-\alpha$가 되도록 선택됩니다.

대칭 단봉 사후분포에서는 HPD 구간과 등꼬리 구간이 일치합니다. 비대칭 사후분포에서는 HPD 구간이 더 짧습니다.

---

## 7. 사후 예측 분포

관측 데이터 $x$가 주어졌을 때 새로운 관측 $\tilde{x}$를 예측하려면:

$$f(\tilde{x} \mid x) = \int f(\tilde{x} \mid \theta) \, \pi(\theta \mid x) \, d\theta$$

이것은 사후분포에 걸쳐 가능도를 평균하여, 모수 불확실성을 예측에 편입합니다.

**예시 (베타-이항)**: $p \mid x \sim \text{Beta}(\alpha', \beta')$이면, $m$번의 새 시행에서 $k$번 성공할 예측 확률은:

$$P(\tilde{X} = k \mid x) = \binom{m}{k} \frac{B(\alpha' + k, \beta' + m - k)}{B(\alpha', \beta')}$$

여기서 $B(\cdot, \cdot)$은 베타 함수입니다. 이것이 **베타-이항 분포 (Beta-Binomial Distribution)**입니다.

---

## 8. 베이즈 vs. 빈도주의 비교

| 특징 | 빈도주의 | 베이즈 |
|------|----------|--------|
| 모수의 확률 | 정의되지 않음 | $\pi(\theta \mid x)$ |
| 신뢰/신용 구간 | 반복에 대한 포함률 보장 | 직접적 확률 진술 |
| 사전 정보 | 사용하지 않음 (또는 비공식적 사용) | 공식적으로 편입 |
| 소표본 | 불안정할 수 있음 | 사전분포가 정칙화에 도움 |
| 계산 부담 | 보통 더 낮음 | 높을 수 있음 (복잡한 모형의 MCMC) |
| 일치성 | 정칙 조건 하에서 일치 추정량 | 사후분포가 참 $\theta$에 집중 |
| 주관성 | 추정량, 검정의 선택 | 사전분포의 선택 |
| 가설 검정 | p-값, 기각 영역 | 베이즈 인자, 사후 확률 |

실제로 대표본과 모호한 사전분포에서는 베이즈와 빈도주의 결과가 밀접하게 일치하는 경우가 많습니다.

---

## 9. Python 예제: 베타-이항 갱신

```python
import math
import random

class BetaBinomial:
    """Beta-Binomial conjugate model for Bayesian updating."""

    def __init__(self, alpha=1.0, beta=1.0):
        """Initialize with Beta(alpha, beta) prior.

        Default: uniform prior (alpha=1, beta=1).
        """
        self.alpha = alpha
        self.beta = beta

    def update(self, successes, trials):
        """Update posterior with observed data."""
        self.alpha += successes
        self.beta += trials - successes
        return self

    def posterior_mean(self):
        return self.alpha / (self.alpha + self.beta)

    def posterior_mode(self):
        """MAP estimate (mode of Beta distribution)."""
        if self.alpha > 1 and self.beta > 1:
            return (self.alpha - 1) / (self.alpha + self.beta - 2)
        return None  # Mode at boundary if alpha or beta <= 1

    def posterior_variance(self):
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b)**2 * (a + b + 1))

    def _beta_function(self, a, b):
        """Compute Beta function B(a, b) = Gamma(a)*Gamma(b)/Gamma(a+b)."""
        return math.exp(math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))

    def pdf(self, p):
        """Evaluate posterior Beta PDF at p."""
        a, b = self.alpha, self.beta
        if p <= 0 or p >= 1:
            return 0.0
        return p**(a-1) * (1-p)**(b-1) / self._beta_function(a, b)

    def equal_tailed_interval(self, level=0.95, n_grid=10000):
        """Approximate equal-tailed credible interval via grid."""
        alpha_tail = (1 - level) / 2
        grid = [i / n_grid for i in range(1, n_grid)]
        pdf_values = [self.pdf(p) for p in grid]
        total = sum(pdf_values)
        cdf = 0
        lower, upper = 0.0, 1.0
        for p, pdf_val in zip(grid, pdf_values):
            cdf += pdf_val / total
            if cdf >= alpha_tail and lower == 0.0:
                lower = p
            if cdf >= 1 - alpha_tail and upper == 1.0:
                upper = p
                break
        return (lower, upper)

    def __repr__(self):
        return f"Beta({self.alpha:.2f}, {self.beta:.2f})"


# ----- Example: Sequential coin-flip updating -----
print("=== Beta-Binomial Sequential Updating ===\n")
model = BetaBinomial(alpha=2, beta=2)  # Mild prior: expect p near 0.5
print(f"Prior:          {model}")
print(f"Prior mean:     {model.posterior_mean():.4f}\n")

# Observe batches of coin flips
batches = [(7, 10), (6, 10), (14, 20)]  # (successes, trials)
total_s, total_t = 0, 0
for successes, trials in batches:
    total_s += successes
    total_t += trials
    model.update(successes, trials)
    print(f"After {total_t} trials ({total_s} successes):")
    print(f"  Posterior:     {model}")
    print(f"  Post. mean:    {model.posterior_mean():.4f}")
    print(f"  Post. mode:    {model.posterior_mode():.4f}")
    print(f"  Post. var:     {model.posterior_variance():.6f}")
    ci = model.equal_tailed_interval()
    print(f"  95% CI:        ({ci[0]:.4f}, {ci[1]:.4f})\n")


# ----- Example: Prior sensitivity analysis -----
print("=== Prior Sensitivity Analysis ===\n")
print("Observed: 3 successes in 10 trials\n")

priors = [
    ("Uniform", 1, 1),
    ("Jeffreys", 0.5, 0.5),
    ("Informative (p~0.5)", 10, 10),
    ("Informative (p~0.8)", 16, 4),
]

for name, a, b in priors:
    m = BetaBinomial(a, b)
    m.update(3, 10)
    print(f"Prior: {name:30s} -> Posterior mean = {m.posterior_mean():.4f}")


# ----- Example: Posterior predictive -----
print("\n=== Posterior Predictive ===\n")
model = BetaBinomial(2, 2)
model.update(7, 10)
print(f"Posterior: {model}")

# P(next flip = heads) = posterior mean of p
print(f"P(next flip = heads) = {model.posterior_mean():.4f}")

# P(k heads in next 5 flips) using Beta-Binomial predictive
def predictive_pmf(k, m, alpha, beta):
    """P(X=k) for Beta-Binomial(m, alpha, beta)."""
    log_binom = math.lgamma(m+1) - math.lgamma(k+1) - math.lgamma(m-k+1)
    log_beta_num = math.lgamma(alpha+k) + math.lgamma(beta+m-k) - math.lgamma(alpha+beta+m)
    log_beta_den = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha+beta)
    return math.exp(log_binom + log_beta_num - log_beta_den)

m_new = 5
print(f"\nPredictive distribution for {m_new} new trials:")
for k in range(m_new + 1):
    prob = predictive_pmf(k, m_new, model.alpha, model.beta)
    bar = "#" * int(prob * 50)
    print(f"  P(X={k}) = {prob:.4f}  {bar}")
```

---

## 10. 핵심 요약

| 개념 | 핵심 사항 |
|------|-----------|
| 베이즈 정리 | 사후분포 $\propto$ 가능도 $\times$ 사전분포 |
| 켤레 사전분포 | 닫힌 형태의 사후분포; 쉬운 순차적 갱신 |
| 제프리스 사전분포 | 비정보적; 재모수화에 불변 |
| MAP 추정 | 사후분포의 최빈값; 평탄 사전분포에서 MLE와 동일 |
| 사후 평균 | 제곱 오차 손실을 최소화; 정밀도 가중 평균 |
| 신용구간 | 직접적 확률 진술: $P(\theta \in C \mid x) = 1 - \alpha$ |
| HPD 구간 | 주어진 신용 수준에서 가장 짧은 구간 |
| 사후 예측 | 예측을 위해 모수 불확실성을 적분으로 제거 |
| 사전분포 민감도 | 항상 다른 사전분포에 따라 결과가 어떻게 변하는지 확인 |

**핵심 통찰**: 표본 크기가 커지면 가능도가 사전분포를 압도하여, 베이즈와 빈도주의 추정이 수렴합니다. 사전분포는 데이터가 부족할 때 가장 중요합니다.

---

**이전**: [가설 검정](./14_Hypothesis_Testing.md) | **다음**: [비모수적 방법](./16_Nonparametric_Methods.md)
