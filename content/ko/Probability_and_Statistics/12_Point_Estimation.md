# 점 추정

**이전**: [큰 수의 법칙과 중심극한정리](./11_Law_of_Large_Numbers_and_CLT.md) | **다음**: [구간 추정](./13_Interval_Estimation.md)

---

## 학습 목표

이 단원을 완료하면 다음을 할 수 있습니다:

1. 추정 문제를 형식화하기: 모수, 추정량, 추정값 구별하기
2. 추정량의 성질을 정의하고 평가하기: 비편향성, 일치성, 효율성
3. 적률법을 사용하여 추정량 유도하기
4. 가능도 함수와 로그 가능도 함수를 통해 최대가능도 추정량 구성하기
5. 최대가능도 추정량의 점근적 성질 진술하기: 일치성, 정규성, 불변성
6. 피셔 정보량을 계산하고 크래머-라오 하한을 적용하기
7. 네이만 인수분해 정리를 적용하여 충분통계량 식별하기
8. 완비성, 라오-블랙웰 정리, UMVUE를 설명하기

---

점 추정 (Point Estimation)은 관측된 데이터를 사용하여 미지의 모집단 모수에 대한 단일 "최적 추측"을 생성하는 문제입니다. 이 단원에서는 추정량을 평가하고 구성하는 이론적 프레임워크를 개발하며, 강력한 최대가능도 방법과 최적 추정량을 식별하는 최적성 이론을 다룹니다.

---

## 1. 추정 문제

### 1.1 설정

- **모집단 모형**: 데이터 $X_1, X_2, \ldots, X_n$이 분포 $f(x; \theta)$에서 i.i.d.이며, $\theta \in \Theta$는 미지의 **모수** (스칼라 또는 벡터)입니다.
- **추정량 (Estimator)**: 데이터의 함수 $\hat{\theta} = T(X_1, \ldots, X_n)$ ($\theta$에 의존하지 않음).
- **추정값 (Estimate)**: 특정 표본에서 계산된 $\hat{\theta}$의 수치 값.

### 1.2 예시

$X_1, \ldots, X_n \sim N(\mu, \sigma^2)$이고 두 모수가 모두 미지인 경우, $\theta = (\mu, \sigma^2)$입니다. 표본 평균 $\bar{X}$는 $\mu$의 추정량이고, 표본 분산 $S^2 = \frac{1}{n-1}\sum(X_i - \bar{X})^2$는 $\sigma^2$의 추정량입니다.

### 1.3 표본 분포

$\hat{\theta}$는 확률적 데이터의 함수이므로 그 자체가 **표본 분포 (Sampling Distribution)** 를 가진 확률 변수입니다. 이 분포의 성질이 추정량의 우수성을 결정합니다.

---

## 2. 추정량의 성질

### 2.1 편향과 비편향성

추정량 $\hat{\theta}$의 $\theta$에 대한 **편향 (Bias)** 은:

$$\text{Bias}(\hat{\theta}) = E[\hat{\theta}] - \theta$$

$E[\hat{\theta}] = \theta$이 모든 $\theta \in \Theta$에 대해 성립하면 추정량은 **비편향 (Unbiased)** 입니다.

**예시**:

- $\bar{X}$는 $\mu$에 대해 비편향입니다: $E[\bar{X}] = \mu$.
- $S^2 = \frac{1}{n-1}\sum(X_i - \bar{X})^2$는 $\sigma^2$에 대해 비편향입니다. $n-1$ 분모 (베셀 보정, Bessel's Correction)가 $\frac{1}{n}\sum(X_i - \bar{X})^2$에 존재하는 편향을 제거합니다.

### 2.2 평균 제곱 오차

**평균 제곱 오차 (Mean Squared Error, MSE)** 는 편향과 분산을 결합합니다:

$$\text{MSE}(\hat{\theta}) = E[(\hat{\theta} - \theta)^2] = \text{Var}(\hat{\theta}) + [\text{Bias}(\hat{\theta})]^2$$

편향된 추정량이라도 분산이 충분히 줄어들면 비편향 추정량보다 낮은 MSE를 가질 수 있습니다. 이것이 **편향-분산 트레이드오프 (Bias-Variance Tradeoff)** 입니다.

### 2.3 일치성

추정량 $\hat{\theta}_n$은 $n \to \infty$일 때 $\hat{\theta}_n \xrightarrow{P} \theta$이면 **일치적 (Consistent)** 입니다.

충분 조건: $\text{Bias}(\hat{\theta}_n) \to 0$이고 $\text{Var}(\hat{\theta}_n) \to 0$이면, $\hat{\theta}_n$은 일치적입니다 ($\text{MSE} \to 0$이 확률 수렴을 함의하므로).

### 2.4 효율성

모든 비편향 추정량 중에서, 가능한 가장 작은 분산을 달성하는 추정량이 **효율적 (Efficient)** 입니다. 크래머-라오 하한 (7절)이 이 최소 분산을 제공합니다. 두 추정량의 **상대 효율성 (Relative Efficiency)** 은:

$$e(\hat{\theta}_1, \hat{\theta}_2) = \frac{\text{Var}(\hat{\theta}_2)}{\text{Var}(\hat{\theta}_1)}$$

$e > 1$이면 $\hat{\theta}_1$이 더 효율적입니다.

---

## 3. 적률법 (MoM)

### 3.1 절차

1. 처음 $k$개의 **모집단 적률**을 계산합니다: $\mu_j' = E[X^j]$ ($j = 1, \ldots, k$), 모수 $\theta_1, \ldots, \theta_k$의 함수로 표현합니다.
2. 이를 해당 **표본 적률**과 같다고 놓습니다: $m_j' = \frac{1}{n}\sum_{i=1}^n X_i^j$.
3. 연립방정식 $m_j' = \mu_j'(\theta_1, \ldots, \theta_k)$를 $\hat{\theta}_1, \ldots, \hat{\theta}_k$에 대해 풉니다.

### 3.2 예시: 정규분포

$X \sim N(\mu, \sigma^2)$인 경우:

- 1차 적률: $E[X] = \mu$이므로, $\hat{\mu}_{MoM} = \bar{X}$.
- 2차 적률: $E[X^2] = \mu^2 + \sigma^2$이므로, $\hat{\sigma}^2_{MoM} = \frac{1}{n}\sum X_i^2 - \bar{X}^2 = \frac{1}{n}\sum(X_i - \bar{X})^2$.

참고: $\hat{\sigma}^2_{MoM}$은 제수 $n$을 사용하므로 약간의 편향이 있습니다.

### 3.3 예시: 감마분포

$X \sim \text{Gamma}(\alpha, \beta)$ (비율 모수화)인 경우: $E[X] = \alpha/\beta$이고 $E[X^2] = \alpha(\alpha+1)/\beta^2$.

풀면:

$$\hat{\beta}_{MoM} = \frac{\bar{X}}{m_2' - \bar{X}^2}, \qquad \hat{\alpha}_{MoM} = \bar{X}\, \hat{\beta}_{MoM}$$

### 3.4 성질

- 적률법 추정량은 일반적으로 **일치적**입니다 (큰 수의 법칙에 의해 표본 적률이 모집단 적률로 수렴).
- 보통 **효율적이지 않습니다** (MLE보다 큰 분산을 가질 수 있음).
- 계산이 쉽고 반복적 MLE 알고리즘의 좋은 초기값을 제공합니다.

---

## 4. 최대가능도 추정 (MLE)

### 4.1 가능도 함수

관측된 데이터 $x_1, \ldots, x_n$이 주어지면, **가능도 함수 (Likelihood Function)** 는:

$$L(\theta) = L(\theta; x_1, \ldots, x_n) = \prod_{i=1}^n f(x_i; \theta)$$

이는 관측된 데이터에서 평가된 결합 밀도를 $\theta$의 함수로 본 것입니다.

### 4.2 로그 가능도

곱은 최대화하기 어려우므로, **로그 가능도 (Log-Likelihood)** 로 작업합니다:

$$\ell(\theta) = \ln L(\theta) = \sum_{i=1}^n \ln f(x_i; \theta)$$

$\ell(\theta)$를 최대화하는 것은 $L(\theta)$를 최대화하는 것과 동치입니다.

### 4.3 스코어 함수

**스코어 함수 (Score Function)** 는 로그 가능도의 기울기입니다:

$$S(\theta) = \frac{\partial \ell(\theta)}{\partial \theta}$$

최대가능도 추정량 (MLE) $\hat{\theta}_{MLE}$는 **스코어 방정식** (가능도 방정식)을 만족합니다:

$$S(\hat{\theta}) = \sum_{i=1}^n \frac{\partial}{\partial\theta} \ln f(x_i; \hat{\theta}) = 0$$

### 4.4 예시: 정규분포

$X_1, \ldots, X_n \sim N(\mu, \sigma^2)$인 경우:

$$\ell(\mu, \sigma^2) = -\frac{n}{2}\ln(2\pi) - \frac{n}{2}\ln(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n(x_i - \mu)^2$$

편미분을 0으로 놓으면:

$$\hat{\mu}_{MLE} = \bar{X}, \qquad \hat{\sigma}^2_{MLE} = \frac{1}{n}\sum_{i=1}^n(X_i - \bar{X})^2$$

참고: $\hat{\sigma}^2_{MLE}$는 편향되어 있지만 ($n-1$ 대신 $n$으로 나눔), 일치적입니다.

### 4.5 예시: 포아송 분포

$X_1, \ldots, X_n \sim \text{Poisson}(\lambda)$인 경우:

$$\ell(\lambda) = \sum_{i=1}^n \left[x_i \ln\lambda - \lambda - \ln(x_i!)\right]$$

$$\frac{d\ell}{d\lambda} = \frac{\sum x_i}{\lambda} - n = 0 \implies \hat{\lambda}_{MLE} = \bar{X}$$

---

## 5. MLE의 성질

### 5.1 일치성

정칙 조건 하에서, $\hat{\theta}_{MLE} \xrightarrow{P} \theta_0$ (참 모수 값)입니다.

### 5.2 점근 정규성

$$\sqrt{n}(\hat{\theta}_{MLE} - \theta_0) \xrightarrow{d} N\!\left(0, \frac{1}{I(\theta_0)}\right)$$

여기서 $I(\theta)$는 피셔 정보량 (6절 참조)입니다. MLE는 **점근적으로 효율적**입니다: 극한에서 크래머-라오 하한을 달성합니다.

### 5.3 불변 성질

$\hat{\theta}$가 $\theta$의 MLE이면, 임의의 함수 $g$에 대해 $g(\hat{\theta})$는 $g(\theta)$의 MLE입니다. 이를 **불변 원리 (Invariance Principle)** 라 합니다.

예를 들어, $\hat{\sigma}^2$가 $\sigma^2$의 MLE이면, $\hat{\sigma} = \sqrt{\hat{\sigma}^2}$는 $\sigma$의 MLE입니다.

### 5.4 점근 효율성

모든 일치적이고 점근적으로 정규인 추정량 중에서, MLE는 가장 작은 점근 분산을 가집니다. 대표본 극한에서 어떤 정칙 추정량도 이보다 더 잘할 수 없습니다.

---

## 6. 피셔 정보량

### 6.1 정의

단일 관측에 포함된 $\theta$에 대한 **피셔 정보량 (Fisher Information)** 은:

$$I(\theta) = E\!\left[\left(\frac{\partial}{\partial\theta} \ln f(X; \theta)\right)^2\right] = E[S(\theta)^2]$$

정칙 조건 하에서, 이는 다음과 같습니다:

$$I(\theta) = -E\!\left[\frac{\partial^2}{\partial\theta^2} \ln f(X; \theta)\right]$$

### 6.2 해석

- 피셔 정보량은 참 모수 값에서 로그 가능도의 **곡률**을 측정합니다.
- 높은 곡률은 로그 가능도가 뾰족하게 정점을 이루어 $\theta$를 정확히 찾기 쉬움을 의미합니다.
- $n$개의 i.i.d. 관측에 대해, 총 피셔 정보량은 $I_n(\theta) = n \cdot I(\theta)$입니다.

### 6.3 예시: 베르누이

$X \sim \text{Bernoulli}(p)$인 경우: $\ln f(x; p) = x\ln p + (1-x)\ln(1-p)$.

$$\frac{\partial^2}{\partial p^2} \ln f = -\frac{x}{p^2} - \frac{1-x}{(1-p)^2}$$

$$I(p) = E\!\left[\frac{X}{p^2} + \frac{1-X}{(1-p)^2}\right] = \frac{1}{p} + \frac{1}{1-p} = \frac{1}{p(1-p)}$$

정보량은 $p = 0.5$일 때 (불확실성이 가장 클 때) 가장 높고, $p = 0$ 또는 $p = 1$ 근처에서 가장 낮습니다.

### 6.4 다변량 피셔 정보량

모수 벡터 $\boldsymbol{\theta} \in \mathbb{R}^k$에 대해, 피셔 정보량은 $k \times k$ **행렬**입니다:

$$[\mathbf{I}(\boldsymbol{\theta})]_{jk} = -E\!\left[\frac{\partial^2 \ell}{\partial\theta_j \partial\theta_k}\right]$$

---

## 7. 크래머-라오 하한

### 7.1 진술

$\theta$의 임의의 **비편향** 추정량 $\hat{\theta}$에 대해:

$$\text{Var}(\hat{\theta}) \ge \frac{1}{I_n(\theta)} = \frac{1}{n \cdot I(\theta)}$$

### 7.2 해석

- CRLB는 모든 비편향 추정량의 **가능한 최소 분산**을 제공합니다.
- $\text{Var}(\hat{\theta}) = 1/(nI(\theta))$이면, 추정량은 **효율적**이거나 CRLB를 달성한다고 합니다.
- MLE는 점근적으로 (큰 $n$에 대해) CRLB를 달성하며, 때로는 유한 $n$에서도 정확히 달성합니다.

### 7.3 예시: 베르누이

$X_1, \ldots, X_n \sim \text{Bernoulli}(p)$인 경우, MLE는 $\hat{p} = \bar{X}$입니다. 그 분산은 $p(1-p)/n$입니다.

CRLB는 $1/(nI(p)) = p(1-p)/n$입니다.

$\text{Var}(\hat{p}) = \text{CRLB}$이므로, 표본 비율은 $p$를 추정하는 데 효율적입니다.

### 7.4 편향된 추정량에 대한 확장

$E[\hat{\theta}] = g(\theta)$ (편향 가능)이면, 한계는 다음으로 일반화됩니다:

$$\text{Var}(\hat{\theta}) \ge \frac{[g'(\theta)]^2}{nI(\theta)}$$

---

## 8. 충분통계량

### 8.1 정의

통계량 $T = T(X_1, \ldots, X_n)$이 $\theta$에 대해 **충분 (Sufficient)** 하다는 것은, $T$가 주어졌을 때 데이터의 조건부 분포가 $\theta$에 의존하지 않을 때입니다:

$$f(x_1, \ldots, x_n \mid T = t; \theta) \text{ 가 } \theta\text{와 무관}$$

직관적으로, $T$는 $\theta$에 대한 데이터의 **모든 정보**를 담고 있습니다. $T$가 알려지면 데이터의 나머지 변동은 순수 잡음입니다.

### 8.2 네이만 인수분해 정리

$T(X_1, \ldots, X_n)$이 $\theta$에 대해 충분인 것과 결합 밀도가 다음과 같이 인수분해되는 것은 동치입니다:

$$f(x_1, \ldots, x_n; \theta) = g(T(\mathbf{x}), \theta) \cdot h(\mathbf{x})$$

여기서 $g$는 데이터에 $T$를 통해서만 의존하고, $h$는 $\theta$에 의존하지 않습니다.

### 8.3 예시

**정규** ($\mu$ 미지, $\sigma^2$ 기지): $T = \sum X_i$ (또는 동치적으로 $\bar{X}$)가 충분합니다.

$$\prod_{i=1}^n \frac{1}{\sqrt{2\pi}\sigma}e^{-(x_i-\mu)^2/(2\sigma^2)} = \underbrace{\exp\!\left(\frac{\mu \sum x_i}{\sigma^2} - \frac{n\mu^2}{2\sigma^2}\right)}_{g(T, \mu)} \cdot \underbrace{\frac{1}{(2\pi\sigma^2)^{n/2}} \exp\!\left(-\frac{\sum x_i^2}{2\sigma^2}\right)}_{h(\mathbf{x})}$$

**포아송** ($\lambda$ 미지): $T = \sum X_i$가 충분합니다.

**정규** ($\mu$와 $\sigma^2$ 모두 미지): $T = (\sum X_i, \sum X_i^2)$가 결합 충분합니다.

---

## 9. 완비성, UMVUE, 라오-블랙웰

### 9.1 완비성

충분통계량 $T$가 **완비 (Complete)** 하다는 것은, 모든 함수 $g$에 대해:

$$E_\theta[g(T)] = 0 \text{ for all } \theta \implies P(g(T) = 0) = 1 \text{ for all } \theta$$

즉, $T$에 기반한 영(zero)의 유일한 비편향 추정량은 영 함수입니다. 완비성은 "중복된" 충분통계량을 배제합니다.

**지수족 (Exponential Family)** 분포는 (완만한 조건 하에서) 완비 충분통계량을 가집니다.

### 9.2 라오-블랙웰 정리

$\hat{\theta}$가 $\theta$의 비편향 추정량이고 $T$가 충분통계량이면, 다음을 정의합니다:

$$\hat{\theta}^* = E[\hat{\theta} \mid T]$$

그러면:

1. $\hat{\theta}^*$는 비편향입니다: $E[\hat{\theta}^*] = \theta$.
2. $\text{Var}(\hat{\theta}^*) \le \text{Var}(\hat{\theta})$이며, 등호는 $\hat{\theta}$가 이미 $T$의 함수일 때만 성립합니다.

라오-블랙웰 정리 (Rao-Blackwell Theorem)는 다음을 말합니다: **임의의 비편향 추정량을 충분통계량에 대해 조건부 기대값을 취하면 (약하게) 더 나은 추정량을 얻습니다**.

### 9.3 UMVUE (균일 최소 분산 비편향 추정량)

비편향 추정량이 **UMVUE (Uniformly Minimum Variance Unbiased Estimator)** 라 함은, $\theta$의 모든 값에 대해 모든 비편향 추정량 중 가장 작은 분산을 가질 때입니다.

**레만-셰페 정리 (Lehmann-Scheffe Theorem)**: $T$가 완비 충분통계량이고 $\hat{\theta}^* = g(T)$가 $\theta$에 대해 비편향이면, $\hat{\theta}^*$가 UMVUE입니다.

### 9.4 UMVUE를 찾는 절차

1. 충분통계량 $T$를 찾습니다 (네이만 인수분해를 통해).
2. 완비성을 확인합니다 (지수족에 의해 보장되는 경우가 많음).
3. $\theta$에 대해 비편향인 함수 $g(T)$를 찾습니다.
4. 레만-셰페에 의해 $g(T)$가 UMVUE입니다.

### 9.5 예시: 정규 평균의 UMVUE

$X_1, \ldots, X_n \sim N(\mu, \sigma^2)$ ($\sigma^2$ 기지)인 경우:

- 충분통계량: $T = \bar{X}$
- 완비: 예 (지수족)
- 비편향: $E[\bar{X}] = \mu$
- 결론: $\bar{X}$가 $\mu$의 UMVUE입니다

---

## 10. Python 예제

### 10.1 감마분포에 대한 적률법

```python
import random
import math

random.seed(42)

# True parameters
alpha_true = 3.0
beta_true = 2.0  # rate parameterisation
# Gamma(alpha, beta): mean = alpha/beta, var = alpha/beta^2

# Generate Gamma samples as sum of Exponentials (alpha must be integer for this)
n = 500
samples = []
for _ in range(n):
    # Gamma(3, 2) = sum of 3 independent Exp(2)
    val = sum(random.expovariate(beta_true) for _ in range(int(alpha_true)))
    samples.append(val)

# Method of Moments
m1 = sum(samples) / n                              # first sample moment
m2 = sum(x ** 2 for x in samples) / n              # second sample moment
sample_var = m2 - m1 ** 2                           # = E[X^2] - (E[X])^2

alpha_mom = m1 ** 2 / sample_var
beta_mom = m1 / sample_var

print("Method of Moments for Gamma(alpha, beta):")
print(f"  alpha_hat = {alpha_mom:.4f}  (true: {alpha_true})")
print(f"  beta_hat  = {beta_mom:.4f}  (true: {beta_true})")
print(f"  sample mean     = {m1:.4f}  (theoretical: {alpha_true/beta_true:.4f})")
print(f"  sample variance = {sample_var:.4f}  "
      f"(theoretical: {alpha_true/beta_true**2:.4f})")
```

### 10.2 정규분포에 대한 MLE

```python
import random
import math

random.seed(123)

mu_true = 5.0
sigma_true = 2.0
n = 200
samples = [random.gauss(mu_true, sigma_true) for _ in range(n)]

# MLE
mu_mle = sum(samples) / n
sigma2_mle = sum((x - mu_mle) ** 2 for x in samples) / n       # biased
sigma2_unbiased = sum((x - mu_mle) ** 2 for x in samples) / (n - 1)  # unbiased

print("MLE for Normal(mu, sigma^2):")
print(f"  mu_MLE     = {mu_mle:.4f}  (true: {mu_true})")
print(f"  sigma2_MLE = {sigma2_mle:.4f}  (true: {sigma_true**2})")
print(f"  sigma2_S2  = {sigma2_unbiased:.4f}  (unbiased, true: {sigma_true**2})")
print(f"  sigma_MLE  = {math.sqrt(sigma2_mle):.4f}  "
      f"(true: {sigma_true}, by invariance)")
```

### 10.3 포아송분포에 대한 MLE

```python
import random
import math

random.seed(77)

lambda_true = 4.5
n = 300

# Generate Poisson samples via inverse transform
def poisson_sample(lam):
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        p *= random.random()
        if p < L:
            return k
        k += 1

samples = [poisson_sample(lambda_true) for _ in range(n)]

# MLE for Poisson is the sample mean
lambda_mle = sum(samples) / n

# Fisher information: I(lambda) = 1/lambda
# CRLB: Var(lambda_hat) >= lambda / n
fisher_info = 1.0 / lambda_true
crlb = lambda_true / n

# Simulate variance of MLE
n_sim = 10_000
mle_values = []
for _ in range(n_sim):
    sim_samples = [poisson_sample(lambda_true) for _ in range(n)]
    mle_values.append(sum(sim_samples) / n)

var_mle = sum((v - lambda_true) ** 2 for v in mle_values) / (n_sim - 1)

print(f"\nMLE for Poisson(lambda = {lambda_true}):")
print(f"  lambda_MLE = {lambda_mle:.4f}")
print(f"  Fisher info I(lambda) = {fisher_info:.4f}")
print(f"  CRLB = {crlb:.6f}")
print(f"  Simulated Var(MLE) = {var_mle:.6f}")
print(f"  MLE achieves CRLB: {abs(var_mle - crlb) < 0.005}")
```

### 10.4 정규 평균에 대한 로그 가능도 시각화

```python
import random
import math

random.seed(2024)

mu_true = 3.0
sigma = 1.5
n = 50
samples = [random.gauss(mu_true, sigma) for _ in range(n)]

def log_likelihood_normal_mean(mu, data, sigma):
    """Log-likelihood for Normal mean with known sigma."""
    n = len(data)
    ss = sum((x - mu) ** 2 for x in data)
    return -n / 2 * math.log(2 * math.pi * sigma ** 2) - ss / (2 * sigma ** 2)

# Evaluate over a grid
mu_grid = [2.0 + 0.05 * i for i in range(41)]  # 2.0 to 4.0
ll_values = [log_likelihood_normal_mean(mu, samples, sigma) for mu in mu_grid]

# Find MLE
mu_mle = sum(samples) / n
ll_max = log_likelihood_normal_mean(mu_mle, samples, sigma)

print(f"\nLog-likelihood for Normal mean (sigma = {sigma} known):")
print(f"  MLE: mu_hat = {mu_mle:.4f}")
print(f"  Max log-likelihood: {ll_max:.2f}")

# ASCII plot of log-likelihood
ll_min = min(ll_values)
ll_range = ll_max - ll_min
print("\n  mu    | log-likelihood")
print("  " + "-" * 55)
for mu, ll in zip(mu_grid, ll_values):
    bar_len = int((ll - ll_min) / ll_range * 40) if ll_range > 0 else 0
    marker = " <-- MLE" if abs(mu - mu_mle) < 0.03 else ""
    print(f"  {mu:5.2f} | {'#' * bar_len}{marker}")
```

### 10.5 크래머-라오 하한: 베르누이

```python
import random
import math

random.seed(555)

p_true = 0.3
n_obs = 100
n_sim = 50_000

# CRLB for Bernoulli: Var(p_hat) >= p(1-p)/n
crlb = p_true * (1 - p_true) / n_obs

# Simulate sampling distribution of p_hat = X_bar
p_hat_values = []
for _ in range(n_sim):
    successes = sum(1 for _ in range(n_obs) if random.random() < p_true)
    p_hat_values.append(successes / n_obs)

mean_phat = sum(p_hat_values) / n_sim
var_phat = sum((p - mean_phat) ** 2 for p in p_hat_values) / (n_sim - 1)

print(f"\nCramer-Rao Bound for Bernoulli(p = {p_true}), n = {n_obs}:")
print(f"  CRLB = {crlb:.6f}")
print(f"  Simulated Var(p_hat) = {var_phat:.6f}")
print(f"  Ratio Var/CRLB = {var_phat / crlb:.4f}  (should be ~1.0)")
print(f"  E[p_hat] = {mean_phat:.4f}  (unbiased: true p = {p_true})")
```

### 10.6 라오-블랙웰 개선

```python
import random

random.seed(888)

# Estimating P(X=0) for Poisson(lambda)
# True value: e^(-lambda)
# Naive unbiased estimator: T(X1) = 1 if X1=0, else 0
# Sufficient statistic: S = sum(Xi)
# Rao-Blackwell: E[T | S] = P(X1=0 | sum=s) = ((n-1)/n)^s * C(n-1,s-0)/C(n,s)
# For Poisson, E[1(X1=0) | S=s] = ((n-1)/n)^s

import math

lambda_true = 2.0
n = 20
n_sim = 50_000
true_value = math.exp(-lambda_true)

def poisson_sample(lam):
    L = math.exp(-lam)
    k, p = 0, 1.0
    while True:
        p *= random.random()
        if p < L:
            return k
        k += 1

naive_estimates = []
rb_estimates = []

for _ in range(n_sim):
    data = [poisson_sample(lambda_true) for _ in range(n)]

    # Naive: use only first observation
    naive = 1.0 if data[0] == 0 else 0.0
    naive_estimates.append(naive)

    # Rao-Blackwell: E[1(X1=0) | S=s] = ((n-1)/n)^s
    s = sum(data)
    rb = ((n - 1) / n) ** s
    rb_estimates.append(rb)

mean_naive = sum(naive_estimates) / n_sim
var_naive = sum((x - mean_naive) ** 2 for x in naive_estimates) / (n_sim - 1)
mean_rb = sum(rb_estimates) / n_sim
var_rb = sum((x - mean_rb) ** 2 for x in rb_estimates) / (n_sim - 1)

print(f"\nRao-Blackwell improvement for estimating e^(-lambda):")
print(f"  True value: {true_value:.6f}")
print(f"  Naive estimator (uses X1 only):")
print(f"    Mean = {mean_naive:.6f},  Var = {var_naive:.6f}")
print(f"  Rao-Blackwell estimator (conditions on sum):")
print(f"    Mean = {mean_rb:.6f},  Var = {var_rb:.6f}")
print(f"  Variance reduction: {(1 - var_rb / var_naive) * 100:.1f}%")
```

---

## 핵심 요약

1. **비편향성, 일치성, 효율성**은 추정량 평가의 세 기둥입니다. 일치성이 가장 중요한 것으로 간주되는 경우가 많습니다: 좋은 추정량은 최소한 참값으로 수렴해야 합니다.
2. **적률법**은 표본 적률과 모집단 적률을 같다고 놓습니다. 단순하고 일치적이지만 일반적으로 가장 효율적이지는 않습니다.
3. **최대가능도 추정**은 관측된 데이터의 확률을 최대화합니다. MLE는 일치적이고, 점근적으로 정규이며, 점근적으로 효율적이고, 재모수화에 대해 불변입니다.
4. **피셔 정보량**은 단일 관측이 $\theta$에 대해 알려주는 정보의 양을 측정합니다. 이는 크래머-라오 하한을 통해 정밀도 한계를 결정합니다.
5. **크래머-라오 하한**은 모든 비편향 추정량의 분산 하한을 제공합니다: $\text{Var}(\hat{\theta}) \ge 1/(nI(\theta))$.
6. **충분통계량**은 $\theta$에 대한 정보를 잃지 않고 데이터를 압축합니다. 네이만 인수분해 정리가 실용적인 판별법을 제공합니다.
7. **라오-블랙웰 정리**는 임의의 비편향 추정량을 충분통계량에 대해 조건부 기대하면 분산이 증가하지 않음을 보여줍니다. 완비성 (레만-셰페)과 결합하면 UMVUE를 식별합니다.

---

*다음 단원: [구간 추정](./13_Interval_Estimation.md)*
