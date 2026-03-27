# 12. 표본과 추정 (Sampling and Estimation)

[이전: 확률론 복습](./11_Probability_Review.md) | [다음: 신뢰구간](./13_Confidence_Intervals.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 모수(population parameter)와 표본 통계량(sample statistic)을 구분하고, 표본추출(sampling)이 필요한 이유를 설명할 수 있습니다.
2. 단순무작위추출(simple random sampling), 층화추출(stratified sampling), 계통추출(systematic sampling) 방법을 설명할 수 있습니다.
3. 표본평균의 표본분포(sampling distribution)를 유도하고 표준오차(standard error)를 계산할 수 있습니다.
4. 불편성(unbiasedness), 일치성(consistency), 효율성(efficiency) 성질을 이용하여 추정량(estimator)의 품질을 평가할 수 있습니다.
5. 베르누이(Bernoulli), 정규(normal), 포아송(Poisson), 지수(exponential) 분포에 대해 최대가능도 추정법(Maximum Likelihood Estimation, MLE)을 구현할 수 있습니다.
6. 감마(gamma) 및 베타(beta) 분포의 파라미터를 적률추정법(Method of Moments, MoM)으로 추정할 수 있습니다.
7. 시뮬레이션을 통해 편향(bias), 분산(variance), 평균제곱오차(MSE) 측면에서 MLE와 MoM을 비교할 수 있습니다.

---

실제로는 모집단(population) 전체에 접근하는 경우가 거의 없으며, 항상 표본(sample)을 가지고 작업합니다. 통계학의 핵심 과제는 유한한 표본이 말해주는 것과 모집단 전체에 대한 사실 사이의 간극을 메우는 것입니다. 이 레슨에서는 그 간극을 잇는 이론적 도구를 다룹니다. 표본 통계량의 행동 방식을 정량화하는 표본분포(sampling distribution)와, 제한된 데이터로부터 최선의 파라미터 추정값을 추출하는 추정 방법론(estimation methods)이 그것입니다.

---

## 1. 모집단과 표본

### 1.1 기본 개념

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# 모집단: 관심 있는 전체 대상
# 표본: 모집단에서 추출한 일부

# 예시: 대한민국 성인 남성의 키 (모집단)
population_mean = 173.5  # 모평균 μ
population_std = 6.0     # 모표준편차 σ

# 실제로는 모집단 전체를 알 수 없음
# 표본을 통해 모수(parameter)를 추정

# 표본 추출 (단순무작위추출)
np.random.seed(42)
sample_size = 100
sample = np.random.normal(population_mean, population_std, sample_size)

print("모집단 파라미터 (실제로는 미지):")
print(f"  모평균 (μ): {population_mean}")
print(f"  모표준편차 (σ): {population_std}")

print(f"\n표본 통계량 (n={sample_size}):")
print(f"  표본평균 (x̄): {sample.mean():.2f}")
print(f"  표본표준편차 (s): {sample.std(ddof=1):.2f}")  # ddof=1 for sample std
```

### 1.2 표본추출 방법

```python
# 다양한 표본추출 방법

# 1. 단순무작위추출 (Simple Random Sampling)
np.random.seed(42)
population = np.arange(1, 1001)  # 1부터 1000까지의 모집단
simple_random_sample = np.random.choice(population, size=50, replace=False)
print(f"단순무작위추출: {simple_random_sample[:10]}...")

# 2. 층화추출 (Stratified Sampling)
# 모집단을 층(strata)으로 나누고 각 층에서 추출
strata_A = np.random.normal(50, 10, 500)  # 층 A
strata_B = np.random.normal(80, 15, 500)  # 층 B

# 각 층에서 비례 추출
sample_A = np.random.choice(strata_A, size=25, replace=False)
sample_B = np.random.choice(strata_B, size=25, replace=False)
stratified_sample = np.concatenate([sample_A, sample_B])

print(f"\n층화추출:")
print(f"  층 A 표본 평균: {sample_A.mean():.2f}")
print(f"  층 B 표본 평균: {sample_B.mean():.2f}")
print(f"  전체 표본 평균: {stratified_sample.mean():.2f}")

# 3. 계통추출 (Systematic Sampling)
# k번째마다 추출
k = 20  # 추출 간격
start = np.random.randint(0, k)
systematic_sample = population[start::k]
print(f"\n계통추출 (k={k}): {systematic_sample[:5]}...")
```

---

## 2. 표본분포 (Sampling Distribution)

### 2.1 표본평균의 분포

```python
# 표본평균의 표본분포 시뮬레이션
np.random.seed(42)

# 모집단 파라미터
mu = 100
sigma = 20

# 다양한 표본 크기
sample_sizes = [5, 30, 100]
n_simulations = 10000

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, n in enumerate(sample_sizes):
    # 많은 표본을 추출하여 각각의 평균 계산
    sample_means = []
    for _ in range(n_simulations):
        sample = np.random.normal(mu, sigma, n)
        sample_means.append(sample.mean())

    sample_means = np.array(sample_means)

    # 시각화
    axes[i].hist(sample_means, bins=50, density=True, alpha=0.7, color='steelblue')

    # 이론적 분포: X̄ ~ N(μ, σ²/n)
    theoretical_std = sigma / np.sqrt(n)
    x = np.linspace(mu - 4*theoretical_std, mu + 4*theoretical_std, 100)
    axes[i].plot(x, stats.norm.pdf(x, mu, theoretical_std), 'r-', lw=2)

    axes[i].axvline(mu, color='k', linestyle='--', alpha=0.5)
    axes[i].set_xlabel('표본평균')
    axes[i].set_ylabel('밀도')
    axes[i].set_title(f'n = {n}\nSD(X̄) = σ/√n = {theoretical_std:.2f}')

plt.suptitle('표본평균의 표본분포', fontsize=14)
plt.tight_layout()
plt.show()

# 수치 확인
print("\n표본평균 분포의 특성:")
for n in sample_sizes:
    sample_means = [np.random.normal(mu, sigma, n).mean() for _ in range(n_simulations)]
    print(f"n={n:3d}: E[X̄]={np.mean(sample_means):.2f}, SD(X̄)={np.std(sample_means):.2f}, "
          f"이론값={sigma/np.sqrt(n):.2f}")
```

### 2.2 표준오차 (Standard Error)

```python
# 표준오차 = 표본통계량의 표준편차
# 표본평균의 표준오차: SE(X̄) = σ/√n (또는 s/√n)

def standard_error(sample):
    """표본평균의 표준오차 계산"""
    n = len(sample)
    s = np.std(sample, ddof=1)  # 표본표준편차
    se = s / np.sqrt(n)
    return se

# 예시
np.random.seed(42)
sample_sizes = [10, 30, 100, 500]

print("표본 크기에 따른 표준오차:")
print("-" * 50)

for n in sample_sizes:
    sample = np.random.normal(100, 20, n)
    se = standard_error(sample)
    theoretical_se = 20 / np.sqrt(n)
    print(f"n = {n:4d}: SE = {se:.3f}, 이론값 = {theoretical_se:.3f}")

# 시각화
fig, ax = plt.subplots(figsize=(10, 5))
n_range = np.arange(10, 501)
se_values = 20 / np.sqrt(n_range)

ax.plot(n_range, se_values, 'b-', linewidth=2)
ax.fill_between(n_range, 0, se_values, alpha=0.2)
ax.set_xlabel('표본 크기 (n)')
ax.set_ylabel('표준오차 (SE)')
ax.set_title('표본 크기와 표준오차의 관계: SE = σ/√n')
ax.grid(alpha=0.3)

# 특정 점 표시
for n in [30, 100, 400]:
    se = 20 / np.sqrt(n)
    ax.scatter([n], [se], color='red', s=100, zorder=5)
    ax.annotate(f'n={n}, SE={se:.2f}', (n, se), xytext=(n+20, se+0.5))

plt.show()
```

### 2.3 다른 표본통계량의 분포

```python
# 표본분산의 분포
np.random.seed(42)

# 정규모집단에서 (n-1)S²/σ² ~ χ²(n-1)
n = 20
sigma_squared = 100  # 모분산
n_simulations = 10000

chi_squared_values = []
for _ in range(n_simulations):
    sample = np.random.normal(0, np.sqrt(sigma_squared), n)
    s_squared = np.var(sample, ddof=1)  # 표본분산
    chi_squared = (n - 1) * s_squared / sigma_squared
    chi_squared_values.append(chi_squared)

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 표본분산 분포
sample_variances = [np.var(np.random.normal(0, 10, n), ddof=1) for _ in range(n_simulations)]
axes[0].hist(sample_variances, bins=50, density=True, alpha=0.7)
axes[0].axvline(sigma_squared, color='r', linestyle='--', label=f'σ² = {sigma_squared}')
axes[0].set_xlabel('표본분산 (S²)')
axes[0].set_ylabel('밀도')
axes[0].set_title('표본분산의 분포')
axes[0].legend()

# 카이제곱 변환
axes[1].hist(chi_squared_values, bins=50, density=True, alpha=0.7, label='시뮬레이션')
x = np.linspace(0, 50, 100)
axes[1].plot(x, stats.chi2.pdf(x, df=n-1), 'r-', lw=2, label=f'χ²({n-1})')
axes[1].set_xlabel('(n-1)S²/σ²')
axes[1].set_ylabel('밀도')
axes[1].set_title('표본분산의 카이제곱 변환')
axes[1].legend()

plt.tight_layout()
plt.show()

print(f"시뮬레이션 평균: {np.mean(chi_squared_values):.2f}, 이론값 (df=n-1): {n-1}")
print(f"시뮬레이션 분산: {np.var(chi_squared_values):.2f}, 이론값 (2*df): {2*(n-1)}")
```

---

## 3. 점추정 (Point Estimation)

### 3.1 추정량의 성질

```python
# 좋은 추정량의 조건: 불편성, 일치성, 효율성

# 1. 불편성 (Unbiasedness): E[θ̂] = θ
# 표본평균은 모평균의 불편추정량
# 표본분산(n-1로 나눔)은 모분산의 불편추정량

np.random.seed(42)
mu, sigma = 50, 10
n = 30
n_simulations = 10000

# 표본평균의 불편성
sample_means = [np.random.normal(mu, sigma, n).mean() for _ in range(n_simulations)]
print(f"표본평균의 기대값: {np.mean(sample_means):.4f}, 모평균: {mu}")
print(f"→ 표본평균은 불편추정량")

# 표본분산 비교 (n으로 나눔 vs n-1로 나눔)
var_n = []    # n으로 나눈 분산 (편향)
var_n_1 = []  # n-1로 나눈 분산 (불편)

for _ in range(n_simulations):
    sample = np.random.normal(mu, sigma, n)
    var_n.append(np.var(sample, ddof=0))
    var_n_1.append(np.var(sample, ddof=1))

print(f"\nddof=0 (n으로 나눔) 기대값: {np.mean(var_n):.2f}")
print(f"ddof=1 (n-1로 나눔) 기대값: {np.mean(var_n_1):.2f}")
print(f"모분산: {sigma**2}")
print(f"→ n-1로 나눈 표본분산이 불편추정량")
```

```python
# 2. 일치성 (Consistency): n → ∞ 일 때 θ̂ → θ
np.random.seed(42)
mu = 100

sample_sizes = [10, 50, 100, 500, 1000, 5000]
mean_estimates = []
var_estimates = []

for n in sample_sizes:
    sample = np.random.normal(mu, 20, n)
    mean_estimates.append(sample.mean())
    var_estimates.append(np.var(sample, ddof=1))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(sample_sizes, mean_estimates, 'bo-')
axes[0].axhline(mu, color='r', linestyle='--', label=f'μ = {mu}')
axes[0].set_xlabel('표본 크기 (n)')
axes[0].set_ylabel('표본평균')
axes[0].set_title('일치성: 표본평균 → 모평균')
axes[0].legend()
axes[0].set_xscale('log')

axes[1].plot(sample_sizes, var_estimates, 'go-')
axes[1].axhline(400, color='r', linestyle='--', label='σ² = 400')
axes[1].set_xlabel('표본 크기 (n)')
axes[1].set_ylabel('표본분산')
axes[1].set_title('일치성: 표본분산 → 모분산')
axes[1].legend()
axes[1].set_xscale('log')

plt.tight_layout()
plt.show()
```

```python
# 3. 효율성 (Efficiency): 분산이 작을수록 효율적
# 정규분포에서 평균 추정: 표본평균 vs 표본중앙값

np.random.seed(42)
n = 30
n_simulations = 10000
mu = 50

means = []
medians = []

for _ in range(n_simulations):
    sample = np.random.normal(mu, 10, n)
    means.append(sample.mean())
    medians.append(np.median(sample))

print("효율성 비교 (정규분포에서 모평균 추정):")
print(f"표본평균: Var = {np.var(means):.4f}")
print(f"표본중앙값: Var = {np.var(medians):.4f}")
print(f"상대효율성: {np.var(means) / np.var(medians):.4f}")
print("→ 정규분포에서 표본평균이 더 효율적 (ARE ≈ π/2 ≈ 1.57)")
```

### 3.2 편향-분산 트레이드오프 (Bias-Variance Tradeoff)

```python
# MSE = Bias² + Variance

def mse_analysis(estimates, true_value):
    """MSE 분해"""
    estimates = np.array(estimates)
    bias = np.mean(estimates) - true_value
    variance = np.var(estimates)
    mse = np.mean((estimates - true_value)**2)
    return bias, variance, mse

# 예: 분산 추정 (n vs n-1)
np.random.seed(42)
n = 10
sigma_squared = 100
n_simulations = 10000

var_n = [np.var(np.random.normal(0, 10, n), ddof=0) for _ in range(n_simulations)]
var_n_1 = [np.var(np.random.normal(0, 10, n), ddof=1) for _ in range(n_simulations)]

print("분산 추정량의 MSE 분석:")
print("-" * 60)
print(f"{'추정량':<15} {'Bias':<12} {'Variance':<12} {'MSE':<12}")
print("-" * 60)

for name, estimates in [("n으로 나눔", var_n), ("n-1로 나눔", var_n_1)]:
    bias, var, mse = mse_analysis(estimates, sigma_squared)
    print(f"{name:<15} {bias:<12.4f} {var:<12.4f} {mse:<12.4f}")

# 시각화
fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(var_n, bins=50, alpha=0.5, label=f'n으로 나눔 (편향)', density=True)
ax.hist(var_n_1, bins=50, alpha=0.5, label=f'n-1로 나눔 (불편)', density=True)
ax.axvline(sigma_squared, color='k', linestyle='--', linewidth=2, label=f'σ² = {sigma_squared}')
ax.set_xlabel('추정값')
ax.set_ylabel('밀도')
ax.set_title('편향-분산 트레이드오프')
ax.legend()
plt.show()
```

---

## 4. 최대가능도 추정법 (Maximum Likelihood Estimation)

### 4.1 MLE 개념

```python
# 가능도 함수: L(θ|x) = P(X=x|θ)
# MLE: L(θ)를 최대화하는 θ 찾기

# 예시: 베르누이 분포의 MLE
def bernoulli_likelihood(p, data):
    """베르누이 분포의 가능도 함수"""
    n = len(data)
    k = sum(data)  # 성공 횟수
    return (p ** k) * ((1 - p) ** (n - k))

def bernoulli_log_likelihood(p, data):
    """베르누이 분포의 로그 가능도 함수"""
    n = len(data)
    k = sum(data)
    if p <= 0 or p >= 1:
        return -np.inf
    return k * np.log(p) + (n - k) * np.log(1 - p)

# 데이터 생성
np.random.seed(42)
true_p = 0.3
data = np.random.binomial(1, true_p, size=50)
print(f"실제 p = {true_p}, 관측된 성공 비율 = {data.mean():.2f}")

# 가능도 함수 시각화
p_values = np.linspace(0.01, 0.99, 100)
likelihoods = [bernoulli_likelihood(p, data) for p in p_values]
log_likelihoods = [bernoulli_log_likelihood(p, data) for p in p_values]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(p_values, likelihoods, 'b-', linewidth=2)
axes[0].axvline(data.mean(), color='r', linestyle='--', label=f'MLE p̂ = {data.mean():.2f}')
axes[0].axvline(true_p, color='g', linestyle=':', label=f'True p = {true_p}')
axes[0].set_xlabel('p')
axes[0].set_ylabel('L(p)')
axes[0].set_title('가능도 함수')
axes[0].legend()

axes[1].plot(p_values, log_likelihoods, 'b-', linewidth=2)
axes[1].axvline(data.mean(), color='r', linestyle='--', label=f'MLE p̂ = {data.mean():.2f}')
axes[1].axvline(true_p, color='g', linestyle=':', label=f'True p = {true_p}')
axes[1].set_xlabel('p')
axes[1].set_ylabel('log L(p)')
axes[1].set_title('로그 가능도 함수')
axes[1].legend()

plt.tight_layout()
plt.show()

print(f"\n베르누이 MLE: p̂ = x̄ = {data.mean():.4f}")
```

### 4.2 정규분포의 MLE

```python
from scipy.optimize import minimize

# 정규분포 N(μ, σ²)의 MLE
# MLE: μ̂ = x̄, σ̂² = (1/n)Σ(x_i - x̄)²

def normal_neg_log_likelihood(params, data):
    """정규분포의 음의 로그 가능도 함수"""
    mu, sigma = params
    if sigma <= 0:
        return np.inf
    n = len(data)
    ll = -n/2 * np.log(2*np.pi) - n*np.log(sigma) - np.sum((data - mu)**2) / (2*sigma**2)
    return -ll

# 데이터 생성
np.random.seed(42)
true_mu, true_sigma = 50, 10
data = np.random.normal(true_mu, true_sigma, 100)

# 수치적 최적화
initial_guess = [0, 1]
result = minimize(normal_neg_log_likelihood, initial_guess, args=(data,),
                  method='L-BFGS-B', bounds=[(-np.inf, np.inf), (0.001, np.inf)])

mu_mle, sigma_mle = result.x

# 해석적 해와 비교
mu_analytical = data.mean()
sigma_analytical = np.std(data, ddof=0)  # MLE는 n으로 나눔

print("정규분포 MLE:")
print("-" * 50)
print(f"실제 파라미터: μ = {true_mu}, σ = {true_sigma}")
print(f"해석적 MLE:   μ̂ = {mu_analytical:.4f}, σ̂ = {sigma_analytical:.4f}")
print(f"수치적 MLE:   μ̂ = {mu_mle:.4f}, σ̂ = {sigma_mle:.4f}")

# scipy.stats를 이용한 피팅
mu_fit, sigma_fit = stats.norm.fit(data)
print(f"scipy.stats:  μ̂ = {mu_fit:.4f}, σ̂ = {sigma_fit:.4f}")
```

### 4.3 포아송 분포의 MLE

```python
# 포아송 분포 Poisson(λ)의 MLE
# MLE: λ̂ = x̄

def poisson_neg_log_likelihood(lam, data):
    """포아송 분포의 음의 로그 가능도 함수"""
    if lam <= 0:
        return np.inf
    return -np.sum(stats.poisson.logpmf(data, lam))

# 데이터 생성
np.random.seed(42)
true_lambda = 5
data = np.random.poisson(true_lambda, 100)

# 수치적 최적화
result = minimize(poisson_neg_log_likelihood, x0=[1], args=(data,),
                  method='L-BFGS-B', bounds=[(0.001, np.inf)])

lambda_mle = result.x[0]
lambda_analytical = data.mean()

print("포아송 분포 MLE:")
print(f"실제 λ = {true_lambda}")
print(f"해석적 MLE: λ̂ = {lambda_analytical:.4f}")
print(f"수치적 MLE: λ̂ = {lambda_mle:.4f}")

# 시각화
fig, ax = plt.subplots(figsize=(10, 5))

x_vals = np.arange(0, 15)
ax.bar(x_vals - 0.2, np.bincount(data, minlength=15)[:15] / len(data),
       width=0.4, alpha=0.7, label='관측 빈도')
ax.bar(x_vals + 0.2, stats.poisson.pmf(x_vals, lambda_mle),
       width=0.4, alpha=0.7, label=f'MLE Poisson(λ̂={lambda_mle:.2f})')

ax.set_xlabel('x')
ax.set_ylabel('확률/빈도')
ax.set_title('포아송 분포 MLE 피팅')
ax.legend()
ax.set_xticks(x_vals)
plt.show()
```

### 4.4 지수분포의 MLE

```python
# 지수분포 Exp(λ)의 MLE
# MLE: λ̂ = 1/x̄

def exponential_neg_log_likelihood(lam, data):
    """지수분포의 음의 로그 가능도 함수"""
    if lam <= 0:
        return np.inf
    n = len(data)
    return -n * np.log(lam) + lam * np.sum(data)

# 데이터 생성
np.random.seed(42)
true_lambda = 0.5
data = np.random.exponential(scale=1/true_lambda, size=100)

# MLE
lambda_mle = 1 / data.mean()

print("지수분포 MLE:")
print(f"실제 λ = {true_lambda}")
print(f"MLE: λ̂ = 1/x̄ = {lambda_mle:.4f}")

# 시각화
fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(data, bins=30, density=True, alpha=0.7, label='데이터')
x = np.linspace(0, data.max(), 100)
ax.plot(x, stats.expon.pdf(x, scale=1/lambda_mle), 'r-', lw=2,
        label=f'MLE Exp(λ̂={lambda_mle:.3f})')
ax.plot(x, stats.expon.pdf(x, scale=1/true_lambda), 'g--', lw=2,
        label=f'True Exp(λ={true_lambda})')

ax.set_xlabel('x')
ax.set_ylabel('밀도')
ax.set_title('지수분포 MLE 피팅')
ax.legend()
plt.show()
```

### 4.5 MLE의 점근적 성질

```python
# MLE의 점근적 정규성
# √n(θ̂_MLE - θ) → N(0, I(θ)^{-1}) (피셔 정보량)

# 시뮬레이션: 정규분포 평균 MLE의 점근적 분포
np.random.seed(42)
true_mu = 50
true_sigma = 10
sample_sizes = [10, 30, 100, 500]
n_simulations = 5000

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for i, n in enumerate(sample_sizes):
    mle_estimates = []
    for _ in range(n_simulations):
        sample = np.random.normal(true_mu, true_sigma, n)
        mle_estimates.append(sample.mean())

    # 표준화
    standardized = np.sqrt(n) * (np.array(mle_estimates) - true_mu) / true_sigma

    axes[i].hist(standardized, bins=50, density=True, alpha=0.7)
    x = np.linspace(-4, 4, 100)
    axes[i].plot(x, stats.norm.pdf(x, 0, 1), 'r-', lw=2, label='N(0,1)')
    axes[i].set_xlabel('√n(μ̂ - μ)/σ')
    axes[i].set_ylabel('밀도')
    axes[i].set_title(f'n = {n}')
    axes[i].legend()
    axes[i].set_xlim(-4, 4)

plt.suptitle('MLE의 점근적 정규성', fontsize=14)
plt.tight_layout()
plt.show()
```

---

## 5. 적률추정법 (Method of Moments)

### 5.1 적률추정법 개념

```python
# 적률추정법: 표본적률 = 모적률로 놓고 파라미터 추정
# k차 표본적률: m_k = (1/n)Σx_i^k
# k차 중심적률: m'_k = (1/n)Σ(x_i - x̄)^k

def sample_moments(data, k):
    """k차 표본적률 계산"""
    return np.mean(data ** k)

def sample_central_moments(data, k):
    """k차 표본중심적률 계산"""
    return np.mean((data - data.mean()) ** k)

# 예시 데이터
np.random.seed(42)
data = np.random.gamma(shape=5, scale=2, size=1000)

print("표본적률:")
for k in range(1, 5):
    print(f"  {k}차 적률: {sample_moments(data, k):.4f}")

print("\n표본중심적률:")
for k in range(2, 5):
    print(f"  {k}차 중심적률: {sample_central_moments(data, k):.4f}")
```

### 5.2 감마분포의 적률추정

```python
# 감마분포 Gamma(α, β): E[X] = α/β, Var(X) = α/β²
# 적률추정: α̂ = x̄² / s², β̂ = x̄ / s²

def gamma_mom_estimator(data):
    """감마분포의 적률추정"""
    x_bar = data.mean()
    s_squared = data.var(ddof=1)

    alpha_hat = x_bar ** 2 / s_squared
    beta_hat = x_bar / s_squared

    return alpha_hat, beta_hat

# 데이터 생성
np.random.seed(42)
true_alpha, true_beta = 5, 2
data = np.random.gamma(shape=true_alpha, scale=1/true_beta, size=500)

# 적률추정
alpha_mom, beta_mom = gamma_mom_estimator(data)

# MLE (scipy)
alpha_mle, loc_mle, scale_mle = stats.gamma.fit(data, floc=0)
beta_mle = 1 / scale_mle

print("감마분포 파라미터 추정:")
print("-" * 50)
print(f"{'방법':<15} {'α':<12} {'β':<12}")
print("-" * 50)
print(f"{'실제값':<15} {true_alpha:<12} {true_beta:<12}")
print(f"{'적률추정':<15} {alpha_mom:<12.4f} {beta_mom:<12.4f}")
print(f"{'MLE':<15} {alpha_mle:<12.4f} {beta_mle:<12.4f}")

# 시각화
fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(data, bins=50, density=True, alpha=0.7, label='데이터')
x = np.linspace(0, data.max(), 100)
ax.plot(x, stats.gamma.pdf(x, a=true_alpha, scale=1/true_beta), 'g--', lw=2,
        label=f'실제: α={true_alpha}, β={true_beta}')
ax.plot(x, stats.gamma.pdf(x, a=alpha_mom, scale=1/beta_mom), 'r-', lw=2,
        label=f'MoM: α={alpha_mom:.2f}, β={beta_mom:.2f}')
ax.plot(x, stats.gamma.pdf(x, a=alpha_mle, scale=1/beta_mle), 'b:', lw=2,
        label=f'MLE: α={alpha_mle:.2f}, β={beta_mle:.2f}')

ax.set_xlabel('x')
ax.set_ylabel('밀도')
ax.set_title('감마분포 추정: 적률추정 vs MLE')
ax.legend()
plt.show()
```

### 5.3 베타분포의 적률추정

```python
# 베타분포 Beta(α, β): E[X] = α/(α+β), Var(X) = αβ/((α+β)²(α+β+1))
# 적률추정 유도

def beta_mom_estimator(data):
    """베타분포의 적률추정"""
    x_bar = data.mean()
    s_squared = data.var(ddof=1)

    # 적률 방정식 해
    temp = (x_bar * (1 - x_bar) / s_squared) - 1
    alpha_hat = x_bar * temp
    beta_hat = (1 - x_bar) * temp

    return alpha_hat, beta_hat

# 데이터 생성
np.random.seed(42)
true_alpha, true_beta = 2, 5
data = np.random.beta(true_alpha, true_beta, size=500)

# 적률추정
alpha_mom, beta_mom = beta_mom_estimator(data)

# MLE (scipy)
alpha_mle, beta_mle, loc_mle, scale_mle = stats.beta.fit(data, floc=0, fscale=1)

print("베타분포 파라미터 추정:")
print("-" * 50)
print(f"{'방법':<15} {'α':<12} {'β':<12}")
print("-" * 50)
print(f"{'실제값':<15} {true_alpha:<12} {true_beta:<12}")
print(f"{'적률추정':<15} {alpha_mom:<12.4f} {beta_mom:<12.4f}")
print(f"{'MLE':<15} {alpha_mle:<12.4f} {beta_mle:<12.4f}")

# 시각화
fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(data, bins=30, density=True, alpha=0.7, label='데이터')
x = np.linspace(0.001, 0.999, 100)
ax.plot(x, stats.beta.pdf(x, true_alpha, true_beta), 'g--', lw=2,
        label=f'실제: α={true_alpha}, β={true_beta}')
ax.plot(x, stats.beta.pdf(x, alpha_mom, beta_mom), 'r-', lw=2,
        label=f'MoM: α={alpha_mom:.2f}, β={beta_mom:.2f}')
ax.plot(x, stats.beta.pdf(x, alpha_mle, beta_mle), 'b:', lw=2,
        label=f'MLE: α={alpha_mle:.2f}, β={beta_mle:.2f}')

ax.set_xlabel('x')
ax.set_ylabel('밀도')
ax.set_title('베타분포 추정: 적률추정 vs MLE')
ax.legend()
plt.show()
```

### 5.4 MLE vs 적률추정 비교

```python
# MLE vs MoM 비교 시뮬레이션

np.random.seed(42)
true_alpha, true_beta = 3, 2
sample_sizes = [20, 50, 100, 500]
n_simulations = 1000

results = []

for n in sample_sizes:
    mle_alpha, mle_beta = [], []
    mom_alpha, mom_beta = [], []

    for _ in range(n_simulations):
        data = np.random.gamma(shape=true_alpha, scale=1/true_beta, size=n)

        # MoM
        a_mom, b_mom = gamma_mom_estimator(data)
        mom_alpha.append(a_mom)
        mom_beta.append(b_mom)

        # MLE
        try:
            a_mle, _, scale_mle = stats.gamma.fit(data, floc=0)
            b_mle = 1 / scale_mle
            mle_alpha.append(a_mle)
            mle_beta.append(b_mle)
        except:
            pass

    results.append({
        'n': n,
        'MoM_alpha_bias': np.mean(mom_alpha) - true_alpha,
        'MoM_alpha_var': np.var(mom_alpha),
        'MLE_alpha_bias': np.mean(mle_alpha) - true_alpha,
        'MLE_alpha_var': np.var(mle_alpha),
    })

# 결과 출력
import pandas as pd
df_results = pd.DataFrame(results)
print("MLE vs MoM 비교 (감마분포 α 추정):")
print(df_results.to_string(index=False))

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Bias
axes[0].plot(df_results['n'], np.abs(df_results['MoM_alpha_bias']), 'r-o', label='MoM')
axes[0].plot(df_results['n'], np.abs(df_results['MLE_alpha_bias']), 'b-s', label='MLE')
axes[0].set_xlabel('표본 크기 (n)')
axes[0].set_ylabel('|Bias|')
axes[0].set_title('편향 비교')
axes[0].legend()
axes[0].set_xscale('log')

# Variance
axes[1].plot(df_results['n'], df_results['MoM_alpha_var'], 'r-o', label='MoM')
axes[1].plot(df_results['n'], df_results['MLE_alpha_var'], 'b-s', label='MLE')
axes[1].set_xlabel('표본 크기 (n)')
axes[1].set_ylabel('Variance')
axes[1].set_title('분산 비교')
axes[1].legend()
axes[1].set_xscale('log')
axes[1].set_yscale('log')

plt.tight_layout()
plt.show()
```

---

## 6. 실전 예제: 분포 피팅

### 6.1 여러 분포 비교 피팅

```python
# 데이터에 여러 분포를 피팅하고 비교

np.random.seed(42)

# 실제 데이터 (감마분포로 생성)
true_data = np.random.gamma(shape=3, scale=2, size=500)

# 여러 분포 피팅
distributions = {
    'Normal': stats.norm,
    'Gamma': stats.gamma,
    'Lognormal': stats.lognorm,
    'Exponential': stats.expon,
    'Weibull': stats.weibull_min
}

fit_results = {}

for name, dist in distributions.items():
    try:
        params = dist.fit(true_data)
        # 로그 가능도 계산
        log_likelihood = np.sum(dist.logpdf(true_data, *params))
        # AIC = -2*logL + 2*k
        k = len(params)
        aic = -2 * log_likelihood + 2 * k
        fit_results[name] = {'params': params, 'logL': log_likelihood, 'AIC': aic}
    except Exception as e:
        fit_results[name] = {'error': str(e)}

# 결과 출력
print("분포 피팅 결과 (AIC 기준 정렬):")
print("-" * 60)
print(f"{'분포':<15} {'Log-Likelihood':<18} {'AIC':<12}")
print("-" * 60)

sorted_results = sorted(fit_results.items(), key=lambda x: x[1].get('AIC', np.inf))
for name, result in sorted_results:
    if 'AIC' in result:
        print(f"{name:<15} {result['logL']:<18.2f} {result['AIC']:<12.2f}")

# 시각화
fig, ax = plt.subplots(figsize=(12, 6))

ax.hist(true_data, bins=50, density=True, alpha=0.5, label='데이터', color='gray')
x = np.linspace(0.01, true_data.max(), 200)

colors = plt.cm.Set1(np.linspace(0, 1, len(distributions)))
for (name, dist), color in zip(distributions.items(), colors):
    if name in fit_results and 'params' in fit_results[name]:
        params = fit_results[name]['params']
        ax.plot(x, dist.pdf(x, *params), linewidth=2, color=color,
                label=f'{name} (AIC={fit_results[name]["AIC"]:.1f})')

ax.set_xlabel('x')
ax.set_ylabel('밀도')
ax.set_title('여러 분포 피팅 비교')
ax.legend()
ax.set_xlim(0, true_data.max())
plt.show()

print(f"\n최적 분포: {sorted_results[0][0]}")
```

### 6.2 적합도 검정 (Goodness-of-Fit)

```python
# Kolmogorov-Smirnov 검정과 Anderson-Darling 검정

# 정규분포 적합도 검정
np.random.seed(42)
data_normal = np.random.normal(50, 10, 200)
data_skewed = np.random.gamma(2, 5, 200)

def goodness_of_fit_tests(data, dist_name='norm'):
    """적합도 검정 수행"""
    # 분포 피팅
    if dist_name == 'norm':
        params = stats.norm.fit(data)
        dist = stats.norm(*params)
    elif dist_name == 'expon':
        params = stats.expon.fit(data)
        dist = stats.expon(*params)

    # KS 검정
    ks_stat, ks_pvalue = stats.kstest(data, dist.cdf)

    # Shapiro-Wilk 검정 (정규성)
    if dist_name == 'norm':
        sw_stat, sw_pvalue = stats.shapiro(data)
    else:
        sw_stat, sw_pvalue = np.nan, np.nan

    return {
        'KS_statistic': ks_stat,
        'KS_pvalue': ks_pvalue,
        'SW_statistic': sw_stat,
        'SW_pvalue': sw_pvalue
    }

print("적합도 검정 결과:")
print("=" * 60)

print("\n정규 데이터:")
results_normal = goodness_of_fit_tests(data_normal, 'norm')
print(f"  KS 검정: 통계량={results_normal['KS_statistic']:.4f}, p={results_normal['KS_pvalue']:.4f}")
print(f"  SW 검정: 통계량={results_normal['SW_statistic']:.4f}, p={results_normal['SW_pvalue']:.4f}")
print(f"  → 정규분포에 적합 (p > 0.05)")

print("\n비대칭 데이터:")
results_skewed = goodness_of_fit_tests(data_skewed, 'norm')
print(f"  KS 검정: 통계량={results_skewed['KS_statistic']:.4f}, p={results_skewed['KS_pvalue']:.4f}")
print(f"  SW 검정: 통계량={results_skewed['SW_statistic']:.4f}, p={results_skewed['SW_pvalue']:.4f}")
print(f"  → 정규분포에 부적합 (p < 0.05)")

# QQ 플롯
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

stats.probplot(data_normal, dist="norm", plot=axes[0])
axes[0].set_title('정규 데이터 Q-Q 플롯')
axes[0].grid(alpha=0.3)

stats.probplot(data_skewed, dist="norm", plot=axes[1])
axes[1].set_title('비대칭 데이터 Q-Q 플롯')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 연습 문제

### 문제 1: 표본분포
모평균 100, 모표준편차 15인 정규모집단에서 크기 36인 표본을 추출할 때:
- (a) 표본평균의 기대값과 표준오차는?
- (b) 표본평균이 95와 105 사이일 확률은?

### 문제 2: MLE
다음 데이터가 포아송 분포를 따른다고 할 때, λ의 MLE를 구하고 95% 신뢰구간을 추정하시오.
```python
data = [3, 5, 4, 2, 6, 3, 4, 5, 2, 4]
```

### 문제 3: 적률추정
균등분포 U(a, b)에서 a와 b의 적률추정량을 유도하시오.
(힌트: E[X] = (a+b)/2, Var(X) = (b-a)²/12)

---

## 정리

| 개념 | 핵심 내용 | Python 함수 |
|------|-----------|-------------|
| 표본평균 분포 | X̄ ~ N(μ, σ²/n) | 시뮬레이션 |
| 표준오차 | SE = σ/√n | `s / np.sqrt(n)` |
| 불편성 | E[θ̂] = θ | `ddof=1` 사용 |
| 일치성 | θ̂ → θ as n → ∞ | 시뮬레이션 |
| MLE | L(θ)를 최대화 | `scipy.stats.fit()` |
| 적률추정 | 표본적률 = 모적률 | 수식 유도 |
| 적합도 검정 | KS, Anderson-Darling | `stats.kstest()` |
