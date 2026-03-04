# 구간 추정

**이전**: [점 추정](./12_Point_Estimation.md) | **다음**: [가설 검정](./14_Hypothesis_Testing.md)

---

## 학습 목표

이 단원을 완료하면 다음을 할 수 있습니다:

1. 점 추정만으로는 불충분한 이유와 구간이 필요한 이유를 설명하기
2. 빈도주의적 프레임워크에서 신뢰구간을 정의하고 올바르게 해석하기
3. 정규분포 평균에 대한 신뢰구간 구성하기 (분산이 알려진 경우와 미지인 경우)
4. 카이제곱 분포를 사용하여 분산에 대한 신뢰구간 유도하기
5. 월드, 윌슨, 클로퍼-피어슨 방법을 사용하여 비율에 대한 신뢰구간 계산하기
6. 평균의 차이와 분산의 비에 대한 구간 구성하기
7. 원하는 오차 한계에 필요한 표본 크기 결정하기
8. 부트스트랩 방법 (백분위수법, BCa)을 신뢰구간 구성에 적용하기

---

점 추정값 $\hat{\theta}$는 모수에 대한 단일 최적 추측을 제공하지만, 그 추측의 불확실성에 대해서는 아무것도 알려주지 않습니다. 구간 추정 (Interval Estimation)은 참 모수가 그 범위 안에 있다는 신뢰 정도와 함께 합리적인 값의 범위를 제공하여 이 문제를 해결합니다.

---

## 1. 점 추정의 한계

모수 $\theta$의 점 추정값 $\hat{\theta}$는 표본 데이터에서 계산된 단일 수치입니다. 유용하지만 주요 한계가 있습니다:

- **정밀도의 척도 없음**: 두 표본이 동일한 점 추정값을 산출하더라도 변동성이 크게 다를 수 있습니다.
- **표본 변동성**: 다른 표본은 다른 추정값을 생성합니다. 범위 없이는 추정값이 얼마나 변할 수 있는지 표현할 수 없습니다.
- **의사결정 위험**: 불확실성을 이해하지 않고 단일 수치에 따라 행동하면 잘못된 결정을 내릴 수 있습니다.

**예시**: 서버의 평균 응답 시간을 $\hat{\mu} = 120$ ms로 추정했다고 합시다. 참 평균이 115에서 125 ms 사이일 가능성이 높은가, 아니면 50에서 190 ms 사이인가? 점 추정값만으로는 이 질문에 답할 수 없습니다.

---

## 2. 신뢰구간: 정의와 해석

### 2.1 정의

신뢰 수준 $1 - \alpha$에서의 **신뢰구간 (Confidence Interval, CI)** 은 다음을 만족하는 확률 구간 $[L(X), U(X)]$입니다:

$$P(L(X) \leq \theta \leq U(X)) = 1 - \alpha$$

여기서 $L(X)$와 $U(X)$는 통계량 (표본 데이터의 함수)이고, $\theta$는 고정되었지만 미지인 모수입니다.

### 2.2 빈도주의적 해석

올바른 해석은 특정 구간이 아닌 **절차**에 대한 것입니다:

> 표본 추출 과정을 여러 번 반복하고 매번 95% 신뢰구간을 계산하면, 그 구간 중 약 95%가 참 모수 $\theta$를 포함할 것입니다.

계산된 특정 구간, 예를 들어 $[112, 128]$은 $\theta$를 포함하거나 포함하지 않습니다. "$\theta$가 $[112, 128]$에 있을 확률이 95%이다"라고 말하지 **않습니다**.

### 2.3 폭과 정밀도

신뢰구간의 폭은 다음에 의존합니다:

- **신뢰 수준** $1 - \alpha$: 높은 신뢰도는 더 넓은 구간을 만듭니다.
- **표본 크기** $n$: 큰 표본은 더 좁은 구간을 만듭니다.
- **모집단 변동성** $\sigma$: 더 큰 변동성은 더 넓은 구간을 만듭니다.

---

## 3. 피봇 양

**피봇 양 (Pivot Quantity)** 은 데이터와 모수의 함수 $Q(X, \theta)$로서 그 분포가 완전히 알려져 있는 ($\theta$에 의존하지 않는) 것입니다.

**일반적 방법**: $Q(X, \theta)$가 피봇이고 $P(a \leq Q \leq b) = 1 - \alpha$이면, 부등식을 역전하면 $\theta$에 대한 신뢰구간을 얻습니다.

**예시**: $X_1, \ldots, X_n \sim N(\mu, \sigma^2)$이고 $\sigma$가 알려져 있으면:

$$Q = \frac{\bar{X} - \mu}{\sigma / \sqrt{n}} \sim N(0, 1)$$

분포가 $\mu$에 의존하지 않으므로 이것은 피봇입니다. $P(-z_{\alpha/2} \leq Q \leq z_{\alpha/2}) = 1 - \alpha$를 역전하면:

$$\bar{X} - z_{\alpha/2} \frac{\sigma}{\sqrt{n}} \leq \mu \leq \bar{X} + z_{\alpha/2} \frac{\sigma}{\sqrt{n}}$$

---

## 4. 정규 평균에 대한 신뢰구간

### 4.1 분산이 알려진 경우 (z-구간)

$X_1, \ldots, X_n \sim N(\mu, \sigma^2)$이고 $\sigma^2$가 알려져 있을 때:

$$\bar{X} \pm z_{\alpha/2} \frac{\sigma}{\sqrt{n}}$$

여기서 $z_{\alpha/2}$는 표준 정규의 상위 $\alpha/2$ 분위수입니다. 95% 신뢰구간의 경우, $z_{0.025} = 1.96$입니다.

### 4.2 분산이 미지인 경우 (t-구간)

$\sigma^2$가 미지인 경우, 이를 표본 표준편차 $S$로 대체합니다. 피봇은:

$$T = \frac{\bar{X} - \mu}{S / \sqrt{n}} \sim t_{n-1}$$

$(1-\alpha)$ 신뢰구간은:

$$\bar{X} \pm t_{\alpha/2, n-1} \frac{S}{\sqrt{n}}$$

t-분포 (t-distribution)는 표준 정규보다 두꺼운 꼬리를 가지므로, $\sigma$를 추정하는 추가 불확실성을 고려한 더 넓은 구간을 생성합니다.

```python
import math
import statistics

def z_confidence_interval(data, sigma, confidence=0.95):
    """CI for mean with known population std dev."""
    n = len(data)
    x_bar = sum(data) / n
    # Approximate z-value using inverse error function
    # For common levels: 0.90->1.645, 0.95->1.960, 0.99->2.576
    z_values = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
    z = z_values.get(confidence, 1.960)
    margin = z * sigma / math.sqrt(n)
    return (x_bar - margin, x_bar + margin)

def t_confidence_interval(data, confidence=0.95):
    """CI for mean with unknown variance using t-distribution."""
    n = len(data)
    x_bar = statistics.mean(data)
    s = statistics.stdev(data)
    # t critical values for common df (approximation for moderate n)
    # For large n, t approaches z
    from math import sqrt
    # Use normal approximation for demonstration
    z_values = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
    # Adjust for small samples with Satterthwaite-like correction
    df = n - 1
    z = z_values.get(confidence, 1.960)
    # Simple t-approximation: t ≈ z + (z + z^3)/(4*df) for moderate df
    t_approx = z + (z + z**3) / (4 * df) if df < 30 else z
    margin = t_approx * s / sqrt(n)
    return (x_bar - margin, x_bar + margin)

# Example: Server response times (ms)
data = [118, 122, 131, 115, 127, 124, 119, 130, 126, 121]
print("Sample mean:", statistics.mean(data))
print("Sample std:", statistics.stdev(data))
print("t-interval (95%):", t_confidence_interval(data, 0.95))
```

---

## 5. 정규 분산에 대한 신뢰구간

$X_1, \ldots, X_n \sim N(\mu, \sigma^2)$이면, $\sigma^2$에 대한 피봇은:

$$\frac{(n-1)S^2}{\sigma^2} \sim \chi^2_{n-1}$$

$\sigma^2$에 대한 $(1 - \alpha)$ 신뢰구간은:

$$\left[\frac{(n-1)S^2}{\chi^2_{\alpha/2, n-1}}, \quad \frac{(n-1)S^2}{\chi^2_{1-\alpha/2, n-1}}\right]$$

비대칭성에 주의하세요: 카이제곱 분포 (Chi-squared Distribution)는 오른쪽으로 치우쳐 있으므로, 분산에 대한 신뢰구간은 $S^2$에 대해 대칭이 아닙니다.

---

## 6. 비율에 대한 신뢰구간

$X \sim \text{Binomial}(n, p)$이고 $\hat{p} = X/n$이라 합시다.

### 6.1 월드 구간

가장 단순하지만 (작은 $n$이나 극단적 $p$에서 부정확한 경우가 많음):

$$\hat{p} \pm z_{\alpha/2} \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$$

**문제**: 특히 $p$가 0이나 1에 가깝거나 $n$이 작을 때 포함 확률이 명목 수준보다 훨씬 낮을 수 있습니다.

### 6.2 윌슨 스코어 구간

스코어 검정을 역전한 더 신뢰할 수 있는 대안입니다:

$$\frac{\hat{p} + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$

이는 월드 구간 (Wald Interval)보다 훨씬 나은 포함 확률 성질을 가집니다.

### 6.3 클로퍼-피어슨 (정확) 구간

이항 검정을 역전한 것에 기반합니다. 최소 $1-\alpha$ 포함 확률을 보장합니다 (보수적):

$$\left[B\left(\frac{\alpha}{2}; x, n-x+1\right), \quad B\left(1 - \frac{\alpha}{2}; x+1, n-x\right)\right]$$

여기서 $B(q; a, b)$는 $\text{Beta}(a, b)$ 분포의 $q$번째 분위수입니다.

```python
import math

def wald_interval(x, n, confidence=0.95):
    """Wald confidence interval for proportion."""
    p_hat = x / n
    z = 1.960 if confidence == 0.95 else 1.645
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n)
    return (max(0, p_hat - margin), min(1, p_hat + margin))

def wilson_interval(x, n, confidence=0.95):
    """Wilson score confidence interval for proportion."""
    p_hat = x / n
    z = 1.960 if confidence == 0.95 else 1.645
    z2 = z ** 2
    denom = 1 + z2 / n
    center = (p_hat + z2 / (2 * n)) / denom
    spread = z * math.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n**2)) / denom
    return (center - spread, center + spread)

# Example: 23 successes out of 100 trials
x, n = 23, 100
print(f"Wald:   {wald_interval(x, n)}")
print(f"Wilson: {wilson_interval(x, n)}")

# Small sample: 2 out of 10 -- Wald breaks down
x2, n2 = 2, 10
print(f"\nSmall sample (x=2, n=10):")
print(f"Wald:   {wald_interval(x2, n2)}")
print(f"Wilson: {wilson_interval(x2, n2)}")
```

---

## 7. 평균의 차이와 분산의 비에 대한 신뢰구간

### 7.1 두 평균의 차이 (독립 표본)

$N(\mu_1, \sigma_1^2)$과 $N(\mu_2, \sigma_2^2)$에서의 독립 표본에 대해:

**등분산 가정** (합동 t-구간):

$$(\bar{X}_1 - \bar{X}_2) \pm t_{\alpha/2,\, n_1+n_2-2} \cdot S_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}$$

여기서 $S_p^2 = \frac{(n_1-1)S_1^2 + (n_2-1)S_2^2}{n_1+n_2-2}$입니다.

**이분산 가정** (웰치 t-구간):

$$(\bar{X}_1 - \bar{X}_2) \pm t_{\alpha/2,\, \nu} \sqrt{\frac{S_1^2}{n_1} + \frac{S_2^2}{n_2}}$$

여기서 자유도 $\nu$는 웰치-새터스웨이트 공식 (Welch-Satterthwaite Formula)으로 주어집니다.

### 7.2 두 분산의 비

피봇은:

$$F = \frac{S_1^2 / \sigma_1^2}{S_2^2 / \sigma_2^2} \sim F_{n_1-1, n_2-1}$$

$\sigma_1^2 / \sigma_2^2$에 대한 신뢰구간은:

$$\left[\frac{S_1^2}{S_2^2} \cdot \frac{1}{F_{\alpha/2,\, n_1-1,\, n_2-1}}, \quad \frac{S_1^2}{S_2^2} \cdot \frac{1}{F_{1-\alpha/2,\, n_1-1,\, n_2-1}}\right]$$

---

## 8. 표본 크기 결정

### 8.1 평균 추정을 위한 경우

$\sigma$가 알려져 있을 때 신뢰 수준 $1-\alpha$에서 오차 한계 $E$를 달성하려면:

$$n \geq \left(\frac{z_{\alpha/2} \cdot \sigma}{E}\right)^2$$

**예시**: $\sigma = 15$, 원하는 오차 한계 $E = 3$, $\alpha = 0.05$인 경우:

$$n \geq \left(\frac{1.96 \times 15}{3}\right)^2 = (9.8)^2 = 96.04 \implies n = 97$$

### 8.2 비율 추정을 위한 경우

$$n \geq \left(\frac{z_{\alpha/2}}{E}\right)^2 \hat{p}(1-\hat{p})$$

$p$의 사전 추정이 없으면, $\hat{p} = 0.5$ (최악의 경우)를 사용합니다:

$$n \geq \frac{z_{\alpha/2}^2}{4E^2}$$

```python
import math

def sample_size_mean(sigma, margin, confidence=0.95):
    """Minimum sample size for estimating a mean."""
    z = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}[confidence]
    n = (z * sigma / margin) ** 2
    return math.ceil(n)

def sample_size_proportion(margin, p_hat=0.5, confidence=0.95):
    """Minimum sample size for estimating a proportion."""
    z = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}[confidence]
    n = (z ** 2) * p_hat * (1 - p_hat) / (margin ** 2)
    return math.ceil(n)

print("Mean (sigma=15, E=3):", sample_size_mean(15, 3))
print("Proportion (E=0.03):", sample_size_proportion(0.03))
print("Proportion (E=0.05, p=0.2):", sample_size_proportion(0.05, p_hat=0.2))
```

---

## 9. 부트스트랩 신뢰구간

추정량의 표본 분포를 해석적으로 유도하기 어려울 때, **부트스트랩 (Bootstrap)** 방법이 계산적 대안을 제공합니다.

### 9.1 부트스트랩 원리

1. 크기 $n$인 원래 표본에서 $B$개의 부트스트랩 표본을 추출합니다 (복원 추출).
2. 각 부트스트랩 표본 $b = 1, \ldots, B$에 대해 관심 통계량 $\hat{\theta}^*_b$를 계산합니다.
3. $\{\hat{\theta}^*_1, \ldots, \hat{\theta}^*_B\}$의 경험적 분포를 사용하여 표본 분포를 추정합니다.

### 9.2 백분위수법

$(1-\alpha)$ 백분위수 부트스트랩 신뢰구간은 단순히:

$$[\hat{\theta}^*_{(\alpha/2)}, \quad \hat{\theta}^*_{(1-\alpha/2)}]$$

여기서 $\hat{\theta}^*_{(q)}$는 부트스트랩 분포의 $q$번째 분위수입니다.

### 9.3 BCa (편향 보정 가속화)

BCa 방법 (Bias-Corrected and Accelerated)은 부트스트랩 분포의 편향과 비대칭을 보정하여, 단순 백분위수법보다 더 나은 포함 확률을 제공합니다. 편향 보정 계수 $z_0$과 가속 계수 $a$를 사용합니다.

```python
import random
import statistics

def bootstrap_percentile_ci(data, statistic_fn, B=10000, confidence=0.95, seed=42):
    """Bootstrap percentile confidence interval.

    Args:
        data: list of observations
        statistic_fn: callable that computes the statistic from a sample
        B: number of bootstrap replicates
        confidence: confidence level
        seed: random seed for reproducibility
    """
    random.seed(seed)
    n = len(data)
    alpha = 1 - confidence

    boot_stats = []
    for _ in range(B):
        boot_sample = random.choices(data, k=n)
        boot_stats.append(statistic_fn(boot_sample))

    boot_stats.sort()
    lower_idx = int(B * alpha / 2)
    upper_idx = int(B * (1 - alpha / 2))
    return (boot_stats[lower_idx], boot_stats[upper_idx])

# Example: CI for the median
data = [2.3, 4.1, 1.8, 5.7, 3.2, 6.4, 2.9, 4.8, 3.5, 7.1,
        2.1, 5.3, 3.8, 4.5, 6.0, 1.9, 3.1, 5.5, 4.2, 3.7]

ci_mean = bootstrap_percentile_ci(data, statistics.mean)
ci_median = bootstrap_percentile_ci(data, statistics.median)
print(f"Bootstrap CI for mean:   {ci_mean}")
print(f"Bootstrap CI for median: {ci_median}")
print(f"Sample mean:   {statistics.mean(data):.3f}")
print(f"Sample median: {statistics.median(data):.3f}")
```

---

## 10. 핵심 요약

| 개념 | 핵심 사항 |
|------|-----------|
| 점 추정의 한계 | 불확실성의 척도를 제공하지 않음 |
| 신뢰 수준 $1-\alpha$ | 반복 표본에서 $\theta$를 포함하는 신뢰구간의 비율 |
| 피봇 양 | 알려진 분포를 가진 데이터와 모수의 함수 |
| z-구간 | $\sigma$가 알려져 있을 때 사용; 표준 정규에 기반 |
| t-구간 | $\sigma$가 미지일 때 사용; 추가 불확실성으로 인해 더 넓음 |
| 카이제곱 신뢰구간 | 분산에 대한 것; 비대칭 구간 |
| 월드 vs 윌슨 | 비율에 대해서는 윌슨이 선호됨, 특히 작은 $n$에서 |
| 표본 크기 | $n \propto (z \cdot \sigma / E)^2$; $E$가 감소하면 이차적으로 증가 |
| 부트스트랩 신뢰구간 | 비모수적; 해석적 신뢰구간을 사용할 수 없을 때 유용 |

**흔한 함정**: "$\theta$가 이 구간에 있을 확률이 95%이다"라고 말하는 것. 올바른 빈도주의적 진술은 **절차**의 장기적 포함 확률에 대한 것입니다.

---

**이전**: [점 추정](./12_Point_Estimation.md) | **다음**: [가설 검정](./14_Hypothesis_Testing.md)
