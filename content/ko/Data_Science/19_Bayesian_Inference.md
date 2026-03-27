# 19. 베이지안 추론 (Bayesian Inference)

[이전: 베이지안 통계 기초](./18_Bayesian_Statistics_Basics.md) | [다음: 시계열 분석 기초](./20_Time_Series_Basics.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 켤레 사전분포(conjugate prior)를 사용할 수 없고 사후분포의 닫힌 형식(closed form)이 존재하지 않을 때 MCMC가 필요한 이유를 설명할 수 있습니다
2. Metropolis-Hastings 알고리즘을 처음부터 구현하고, 적절한 수락률(acceptance rate)을 위해 제안 분포(proposal distribution)를 조정할 수 있습니다
3. PyMC를 사용하여 사전분포(prior), 가능도(likelihood), 샘플링 매개변수를 지정하는 베이지안 모델을 구축하고 샘플링할 수 있습니다
4. 베이지안 선형 회귀를 구현하고 사후 추정치를 빈도주의 OLS 결과와 비교할 수 있습니다
5. WAIC, LOO-CV를 사용하여 경쟁 모델을 비교하고 베이즈 팩터(Bayes factor)를 해석할 수 있습니다
6. 사후 예측 검사(posterior predictive checks)를 수행하여 모델 적합도를 평가하고 모델 오특정(misspecification)을 탐지할 수 있습니다
7. R-hat, 유효 표본 크기(effective sample size), 트레이스 플롯(trace plot), 자기상관(autocorrelation) 진단을 사용하여 MCMC 수렴을 평가할 수 있습니다

---

이전 레슨에서는 켤레 사전분포가 우아한 닫힌 형식의 사후분포를 산출하는 방법을 보여주었지만, 대부분의 실제 모델은 그런 편리함을 누리기에는 너무 복잡합니다. 사후분포를 해석적으로 계산할 수 없을 때, MCMC 알고리즘은 대신 샘플을 추출할 수 있게 해줍니다 — 다루기 어려운 적분 문제를 시뮬레이션 문제로 변환하는 것입니다. 이 레슨에서는 MCMC의 이론적 배경과 PyMC를 활용한 현대적 베이지안 모델링의 실제 워크플로우를 모두 배웁니다.

---

## 1. MCMC 소개

### 1.1 왜 MCMC가 필요한가?

**문제**: 복잡한 사후분포의 정규화 상수를 계산하기 어려움

$$P(\theta|D) = \frac{P(D|\theta)P(\theta)}{\int P(D|\theta)P(\theta)d\theta}$$

분모의 적분이 고차원에서 계산 불가능할 수 있음.

**해결**: 사후분포에서 직접 샘플링하여 분포를 근사

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# MCMC의 핵심 아이디어 시각화
np.random.seed(42)

# 목표 분포: 혼합 정규분포 (정규화 상수 없이도 비율만 알면 됨)
def target_unnormalized(x):
    """비정규화 목표 분포"""
    return 0.3 * np.exp(-0.5 * ((x - 2) / 0.5)**2) + \
           0.7 * np.exp(-0.5 * ((x + 1) / 1)**2)

x_range = np.linspace(-5, 5, 200)
y = target_unnormalized(x_range)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 비정규화 목표 분포
axes[0].plot(x_range, y, 'b-', lw=2, label='Unnormalized target')
axes[0].fill_between(x_range, y, alpha=0.3)
axes[0].set_xlabel('x')
axes[0].set_ylabel('Unnormalized density')
axes[0].set_title('비정규화 목표 분포')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# MCMC 샘플의 히스토그램 (나중에 채울 예정)
axes[1].text(0.5, 0.5, 'MCMC 샘플\n(다음 섹션에서 채움)',
             transform=axes[1].transAxes, ha='center', va='center', fontsize=14)
axes[1].set_xlabel('x')
axes[1].set_ylabel('밀도')
axes[1].set_title('MCMC 샘플 히스토그램')

plt.tight_layout()
plt.show()
```

### 1.2 마르코프 체인 기초

**정의**: 다음 상태가 현재 상태에만 의존하는 확률적 과정

$$P(X_{t+1}|X_1, X_2, ..., X_t) = P(X_{t+1}|X_t)$$

```python
def simple_markov_chain_demo():
    """간단한 마르코프 체인 시뮬레이션"""

    # 날씨 전이 확률: 맑음(0), 흐림(1), 비(2)
    transition_matrix = np.array([
        [0.7, 0.2, 0.1],  # 맑음 → 맑음/흐림/비
        [0.3, 0.4, 0.3],  # 흐림 → 맑음/흐림/비
        [0.2, 0.3, 0.5],  # 비 → 맑음/흐림/비
    ])

    states = ['맑음', '흐림', '비']

    # 시뮬레이션
    n_steps = 1000
    current_state = 0  # 맑음에서 시작
    chain = [current_state]

    for _ in range(n_steps):
        probs = transition_matrix[current_state]
        current_state = np.random.choice([0, 1, 2], p=probs)
        chain.append(current_state)

    # 정상 분포 추정
    unique, counts = np.unique(chain, return_counts=True)
    empirical_dist = counts / len(chain)

    # 이론적 정상 분포 (전이행렬의 왼쪽 고유벡터)
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    stationary_idx = np.argmin(np.abs(eigenvalues - 1))
    stationary_dist = eigenvectors[:, stationary_idx].real
    stationary_dist = stationary_dist / stationary_dist.sum()

    print("=== 마르코프 체인 날씨 시뮬레이션 ===")
    print(f"시뮬레이션 단계: {n_steps}")
    print("\n상태별 빈도:")
    for i, state in enumerate(states):
        print(f"  {state}: 경험적={empirical_dist[i]:.3f}, 이론적={stationary_dist[i]:.3f}")

    return chain

chain = simple_markov_chain_demo()
```

---

## 2. Metropolis-Hastings 알고리즘

### 2.1 알고리즘 개요

**목표**: 목표 분포 π(x)에서 샘플 추출

**알고리즘**:
1. 초기값 x₀ 선택
2. 제안 분포 q(x'|x)에서 후보 x' 생성
3. 수락 확률 계산: α = min(1, [π(x')q(x|x')] / [π(x)q(x'|x)])
4. U ~ Uniform(0,1)이면:
   - U < α: x' 수락 (x_{t+1} = x')
   - 그렇지 않으면: x 유지 (x_{t+1} = x)
5. 2-4 반복

### 2.2 구현

```python
def metropolis_hastings(target, proposal_std, n_samples, initial_value=0, burn_in=1000):
    """
    Metropolis-Hastings 알고리즘

    Parameters:
    -----------
    target : callable
        비정규화 목표 밀도 함수
    proposal_std : float
        제안 분포(정규분포)의 표준편차
    n_samples : int
        생성할 샘플 수
    initial_value : float
        초기값
    burn_in : int
        burn-in 기간 (버릴 초기 샘플 수)

    Returns:
    --------
    samples : ndarray
        MCMC 샘플
    acceptance_rate : float
        수락률
    """
    samples = np.zeros(n_samples + burn_in)
    samples[0] = initial_value
    n_accepted = 0

    for i in range(1, n_samples + burn_in):
        current = samples[i-1]

        # 제안 (대칭 정규분포)
        proposal = np.random.normal(current, proposal_std)

        # 수락 확률 (대칭 제안분포이므로 q 취소됨)
        acceptance_ratio = target(proposal) / target(current)
        acceptance_prob = min(1, acceptance_ratio)

        # 수락/거부
        if np.random.uniform() < acceptance_prob:
            samples[i] = proposal
            n_accepted += 1
        else:
            samples[i] = current

    # burn-in 제거
    samples = samples[burn_in:]
    acceptance_rate = n_accepted / (n_samples + burn_in)

    return samples, acceptance_rate


# 예제: 혼합 정규분포에서 샘플링
def mixture_target(x):
    """혼합 정규분포 (비정규화)"""
    return 0.3 * np.exp(-0.5 * ((x - 2) / 0.5)**2) + \
           0.7 * np.exp(-0.5 * ((x + 1) / 1)**2)

# 다양한 제안 분포로 실험
proposal_stds = [0.1, 1.0, 5.0]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

np.random.seed(42)

for i, proposal_std in enumerate(proposal_stds):
    samples, acc_rate = metropolis_hastings(
        mixture_target,
        proposal_std=proposal_std,
        n_samples=10000,
        burn_in=1000
    )

    # 트레이스 플롯
    axes[0, i].plot(samples[:500], 'b-', alpha=0.7, lw=0.5)
    axes[0, i].set_xlabel('Iteration')
    axes[0, i].set_ylabel('Value')
    axes[0, i].set_title(f'Trace (σ={proposal_std}, acc={acc_rate:.2%})')
    axes[0, i].grid(True, alpha=0.3)

    # 히스토그램
    x_range = np.linspace(-5, 5, 200)
    true_density = mixture_target(x_range)
    true_density = true_density / np.trapz(true_density, x_range)

    axes[1, i].hist(samples, bins=50, density=True, alpha=0.7, label='MCMC samples')
    axes[1, i].plot(x_range, true_density, 'r-', lw=2, label='True density')
    axes[1, i].set_xlabel('Value')
    axes[1, i].set_ylabel('Density')
    axes[1, i].set_title(f'Histogram (σ={proposal_std})')
    axes[1, i].legend()
    axes[1, i].grid(True, alpha=0.3)

plt.suptitle('Metropolis-Hastings: 제안 분포 표준편차의 영향')
plt.tight_layout()
plt.show()
```

### 2.3 다변량 확장

```python
def metropolis_hastings_2d(target, proposal_cov, n_samples, initial_value=None, burn_in=1000):
    """
    2차원 Metropolis-Hastings

    Parameters:
    -----------
    target : callable
        비정규화 2D 목표 밀도 함수
    proposal_cov : ndarray
        제안 분포의 공분산 행렬 (2x2)
    n_samples : int
        생성할 샘플 수
    """
    if initial_value is None:
        initial_value = np.array([0.0, 0.0])

    samples = np.zeros((n_samples + burn_in, 2))
    samples[0] = initial_value
    n_accepted = 0

    for i in range(1, n_samples + burn_in):
        current = samples[i-1]

        # 다변량 정규분포에서 제안
        proposal = np.random.multivariate_normal(current, proposal_cov)

        # 수락 확률
        acceptance_ratio = target(proposal) / (target(current) + 1e-10)
        acceptance_prob = min(1, acceptance_ratio)

        if np.random.uniform() < acceptance_prob:
            samples[i] = proposal
            n_accepted += 1
        else:
            samples[i] = current

    samples = samples[burn_in:]
    acceptance_rate = n_accepted / (n_samples + burn_in)

    return samples, acceptance_rate


# 2D 혼합 정규분포
def target_2d(x):
    """2D 혼합 정규분포"""
    # 첫 번째 성분
    mu1 = np.array([2, 2])
    cov1 = np.array([[0.5, 0.3], [0.3, 0.5]])
    term1 = stats.multivariate_normal(mu1, cov1).pdf(x)

    # 두 번째 성분
    mu2 = np.array([-1, -1])
    cov2 = np.array([[0.8, -0.2], [-0.2, 0.8]])
    term2 = stats.multivariate_normal(mu2, cov2).pdf(x)

    return 0.4 * term1 + 0.6 * term2

# 샘플링
np.random.seed(42)
proposal_cov = np.array([[0.5, 0], [0, 0.5]])
samples_2d, acc_rate = metropolis_hastings_2d(
    target_2d,
    proposal_cov=proposal_cov,
    n_samples=20000,
    burn_in=5000
)

print(f"수락률: {acc_rate:.2%}")

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 트레이스 플롯
axes[0].plot(samples_2d[:1000, 0], 'b-', alpha=0.7, lw=0.5, label='x1')
axes[0].plot(samples_2d[:1000, 1], 'r-', alpha=0.7, lw=0.5, label='x2')
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Value')
axes[0].set_title('Trace Plot (first 1000)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2D 산점도
axes[1].scatter(samples_2d[::10, 0], samples_2d[::10, 1], alpha=0.3, s=5)
axes[1].set_xlabel('x1')
axes[1].set_ylabel('x2')
axes[1].set_title('MCMC Samples (2D)')
axes[1].grid(True, alpha=0.3)

# 등고선과 비교
x1_range = np.linspace(-4, 5, 100)
x2_range = np.linspace(-4, 5, 100)
X1, X2 = np.meshgrid(x1_range, x2_range)
positions = np.dstack((X1, X2))
Z = np.array([[target_2d(p) for p in row] for row in positions])

axes[2].contour(X1, X2, Z, levels=10, cmap='viridis')
axes[2].scatter(samples_2d[::20, 0], samples_2d[::20, 1], alpha=0.2, s=3, c='red')
axes[2].set_xlabel('x1')
axes[2].set_ylabel('x2')
axes[2].set_title('Samples vs True Density')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 3. PyMC 소개

### 3.1 설치 및 기본 설정

```bash
# PyMC 설치 (Python 3.9+)
pip install pymc arviz

# 또는 conda
conda install -c conda-forge pymc arviz
```

```python
# 기본 import
import pymc as pm
import arviz as az

# 시각화 설정
az.style.use("arviz-darkgrid")

print(f"PyMC version: {pm.__version__}")
print(f"ArviZ version: {az.__version__}")
```

### 3.2 첫 번째 PyMC 모델: 동전 던지기

```python
import pymc as pm
import arviz as az

# 데이터
n_trials = 100
n_heads = 65

# PyMC 모델 정의
with pm.Model() as coin_model:
    # 사전분포: Beta(1, 1) = Uniform(0, 1)
    p = pm.Beta('p', alpha=1, beta=1)

    # 가능도: Binomial
    y = pm.Binomial('y', n=n_trials, p=p, observed=n_heads)

    # 사후분포 샘플링
    trace = pm.sample(2000, tune=1000, cores=1, random_seed=42)

# 결과 확인
print(az.summary(trace, var_names=['p']))

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 트레이스 플롯
az.plot_trace(trace, var_names=['p'], axes=axes.reshape(1, 2))

plt.tight_layout()
plt.show()

# 사후분포 플롯
fig, ax = plt.subplots(figsize=(10, 5))
az.plot_posterior(trace, var_names=['p'], ax=ax)
ax.axvline(0.65, color='r', linestyle='--', label=f'MLE = 0.65')
ax.legend()
plt.show()
```

### 3.3 정규분포 모수 추정

```python
# 데이터 생성
np.random.seed(42)
true_mu = 5.0
true_sigma = 2.0
data = np.random.normal(true_mu, true_sigma, 50)

print(f"True μ: {true_mu}, σ: {true_sigma}")
print(f"Sample mean: {data.mean():.3f}, std: {data.std():.3f}")

# PyMC 모델
with pm.Model() as normal_model:
    # 사전분포
    mu = pm.Normal('mu', mu=0, sigma=10)  # 약한 정보 사전분포
    sigma = pm.HalfNormal('sigma', sigma=5)  # 양수 제약

    # 가능도
    y = pm.Normal('y', mu=mu, sigma=sigma, observed=data)

    # 샘플링
    trace = pm.sample(2000, tune=1000, cores=1, random_seed=42)

# 결과 요약
print("\n=== 사후분포 요약 ===")
print(az.summary(trace, var_names=['mu', 'sigma']))

# 시각화
fig = plt.figure(figsize=(14, 8))

# 트레이스 플롯
axes = fig.subplots(2, 2)
az.plot_trace(trace, var_names=['mu', 'sigma'], axes=axes)
plt.tight_layout()
plt.show()

# 결합 사후분포
fig, ax = plt.subplots(figsize=(8, 8))
az.plot_pair(trace, var_names=['mu', 'sigma'], kind='kde', ax=ax)
ax.axvline(true_mu, color='r', linestyle='--', alpha=0.5)
ax.axhline(true_sigma, color='r', linestyle='--', alpha=0.5)
ax.scatter([true_mu], [true_sigma], color='r', s=100, marker='x', label='True values')
ax.legend()
plt.show()
```

---

## 4. 베이지안 선형 회귀

### 4.1 모델 정의

$$y_i = \beta_0 + \beta_1 x_i + \epsilon_i$$

$$\epsilon_i \sim N(0, \sigma^2)$$

**베이지안 설정**:
- β₀ ~ N(0, 10²)
- β₁ ~ N(0, 10²)
- σ ~ HalfNormal(5)

### 4.2 PyMC 구현

```python
# 데이터 생성
np.random.seed(42)
n = 100
x = np.random.uniform(0, 10, n)
true_beta0 = 2.5
true_beta1 = 1.5
true_sigma = 1.0
y = true_beta0 + true_beta1 * x + np.random.normal(0, true_sigma, n)

print(f"True parameters: β0={true_beta0}, β1={true_beta1}, σ={true_sigma}")

# 빈도주의 OLS 비교
from scipy import stats as scipy_stats
slope_ols, intercept_ols, _, _, _ = scipy_stats.linregress(x, y)
print(f"OLS estimates: β0={intercept_ols:.3f}, β1={slope_ols:.3f}")

# 베이지안 선형 회귀
with pm.Model() as linear_model:
    # 사전분포
    beta0 = pm.Normal('beta0', mu=0, sigma=10)
    beta1 = pm.Normal('beta1', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=5)

    # 선형 예측
    mu = beta0 + beta1 * x

    # 가능도
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

    # 샘플링
    trace = pm.sample(3000, tune=1000, cores=1, random_seed=42)

# 결과
print("\n=== 베이지안 추정 결과 ===")
summary = az.summary(trace, var_names=['beta0', 'beta1', 'sigma'])
print(summary)

# 시각화
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# 사후분포
az.plot_posterior(trace, var_names=['beta0'], ax=axes[0, 0])
axes[0, 0].axvline(true_beta0, color='r', linestyle='--', label='True')
axes[0, 0].axvline(intercept_ols, color='g', linestyle=':', label='OLS')
axes[0, 0].legend()

az.plot_posterior(trace, var_names=['beta1'], ax=axes[0, 1])
axes[0, 1].axvline(true_beta1, color='r', linestyle='--', label='True')
axes[0, 1].axvline(slope_ols, color='g', linestyle=':', label='OLS')
axes[0, 1].legend()

az.plot_posterior(trace, var_names=['sigma'], ax=axes[0, 2])
axes[0, 2].axvline(true_sigma, color='r', linestyle='--', label='True')
axes[0, 2].legend()

# 회귀선 불확실성
ax = axes[1, 0]
ax.scatter(x, y, alpha=0.5, s=20)

# 사후 샘플에서 회귀선
x_plot = np.linspace(0, 10, 100)
posterior_samples = trace.posterior

# 100개의 사후 샘플로 회귀선 그리기
for i in range(100):
    b0 = posterior_samples['beta0'].values.flatten()[i * 30]  # thin
    b1 = posterior_samples['beta1'].values.flatten()[i * 30]
    ax.plot(x_plot, b0 + b1 * x_plot, 'b-', alpha=0.05)

# 사후 평균 회귀선
mean_b0 = posterior_samples['beta0'].mean().values
mean_b1 = posterior_samples['beta1'].mean().values
ax.plot(x_plot, mean_b0 + mean_b1 * x_plot, 'r-', lw=2, label='Posterior mean')
ax.plot(x_plot, true_beta0 + true_beta1 * x_plot, 'g--', lw=2, label='True')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('회귀선 불확실성')
ax.legend()
ax.grid(True, alpha=0.3)

# 결합 사후분포: beta0 vs beta1
ax = axes[1, 1]
az.plot_pair(trace, var_names=['beta0', 'beta1'], kind='kde', ax=ax)
ax.scatter([true_beta0], [true_beta1], color='r', s=100, marker='x', zorder=5)
ax.set_title('β0 vs β1 결합분포')

# 트레이스 플롯
ax = axes[1, 2]
az.plot_trace(trace, var_names=['beta1'], axes=np.array([[ax, ax]]))
ax.set_title('β1 트레이스')

plt.tight_layout()
plt.show()
```

### 4.3 사후 예측

```python
# 사후 예측분포
with linear_model:
    # 새로운 x 값에 대한 예측
    x_new = np.array([3, 5, 7])

    # 사후 예측 샘플
    pm.set_data({'x': x_new})  # 만약 shared 변수 사용 시

    # 수동으로 사후 예측
    posterior = trace.posterior

    beta0_samples = posterior['beta0'].values.flatten()
    beta1_samples = posterior['beta1'].values.flatten()
    sigma_samples = posterior['sigma'].values.flatten()

# 예측
predictions = {}
for x_val in x_new:
    mu_pred = beta0_samples + beta1_samples * x_val
    y_pred = np.random.normal(mu_pred, sigma_samples)
    predictions[x_val] = {
        'mean': mu_pred.mean(),
        'std': mu_pred.std(),
        'pred_mean': y_pred.mean(),
        'pred_std': y_pred.std(),
        'ci_95': (np.percentile(mu_pred, 2.5), np.percentile(mu_pred, 97.5)),
        'pred_95': (np.percentile(y_pred, 2.5), np.percentile(y_pred, 97.5))
    }

print("=== 사후 예측 ===")
for x_val, pred in predictions.items():
    print(f"\nx = {x_val}:")
    print(f"  E[y|x] = {pred['mean']:.3f} ± {pred['std']:.3f}")
    print(f"  95% CI for E[y|x]: ({pred['ci_95'][0]:.3f}, {pred['ci_95'][1]:.3f})")
    print(f"  95% PI for y: ({pred['pred_95'][0]:.3f}, {pred['pred_95'][1]:.3f})")
```

---

## 5. 모델 비교

### 5.1 WAIC (Widely Applicable Information Criterion)

```python
# 두 모델 비교: 단순 vs 이차 회귀

# 이차 데이터 생성
np.random.seed(42)
n = 100
x = np.random.uniform(0, 10, n)
y_quad = 2 + 0.5 * x + 0.1 * x**2 + np.random.normal(0, 1.5, n)

# 모델 1: 선형
with pm.Model() as model_linear:
    beta0 = pm.Normal('beta0', mu=0, sigma=10)
    beta1 = pm.Normal('beta1', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=5)

    mu = beta0 + beta1 * x
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_quad)

    trace_linear = pm.sample(2000, tune=1000, cores=1, random_seed=42)
    # WAIC 계산을 위한 log_likelihood
    pm.compute_log_likelihood(trace_linear)

# 모델 2: 이차
with pm.Model() as model_quadratic:
    beta0 = pm.Normal('beta0', mu=0, sigma=10)
    beta1 = pm.Normal('beta1', mu=0, sigma=10)
    beta2 = pm.Normal('beta2', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=5)

    mu = beta0 + beta1 * x + beta2 * x**2
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_quad)

    trace_quadratic = pm.sample(2000, tune=1000, cores=1, random_seed=42)
    pm.compute_log_likelihood(trace_quadratic)

# WAIC 비교
print("=== WAIC 비교 ===")
waic_linear = az.waic(trace_linear)
waic_quadratic = az.waic(trace_quadratic)

print(f"Linear model WAIC: {waic_linear.waic:.2f}")
print(f"Quadratic model WAIC: {waic_quadratic.waic:.2f}")
print(f"\n낮은 WAIC이 더 좋은 모델")

# 모델 비교
comparison = az.compare({
    'Linear': trace_linear,
    'Quadratic': trace_quadratic
}, ic='waic')
print("\n=== 모델 비교 ===")
print(comparison)

# 시각화
fig, ax = plt.subplots(figsize=(10, 5))
az.plot_compare(comparison, ax=ax)
plt.title('WAIC 모델 비교')
plt.tight_layout()
plt.show()
```

### 5.2 LOO-CV (Leave-One-Out Cross-Validation)

```python
# LOO-CV (Pareto Smoothed Importance Sampling LOO)
print("=== LOO-CV 비교 ===")

loo_linear = az.loo(trace_linear)
loo_quadratic = az.loo(trace_quadratic)

print(f"Linear model LOO: {loo_linear.loo:.2f}")
print(f"Quadratic model LOO: {loo_quadratic.loo:.2f}")

# LOO 진단
print("\n=== LOO 진단: Pareto k ===")
print(f"Linear - k > 0.7인 관측치: {(loo_linear.pareto_k > 0.7).sum()}")
print(f"Quadratic - k > 0.7인 관측치: {(loo_quadratic.pareto_k > 0.7).sum()}")

# 비교
comparison_loo = az.compare({
    'Linear': trace_linear,
    'Quadratic': trace_quadratic
}, ic='loo')
print("\n=== LOO 모델 비교 ===")
print(comparison_loo)
```

### 5.3 베이즈 팩터 (개념)

```python
def bayes_factor_example():
    """베이즈 팩터 개념 설명"""

    print("=== 베이즈 팩터 (Bayes Factor) ===")
    print()
    print("정의: BF₁₀ = P(D|M₁) / P(D|M₀)")
    print()
    print("해석:")
    print("  BF > 100   : 결정적 증거 (M₁ 지지)")
    print("  30 < BF < 100: 매우 강한 증거")
    print("  10 < BF < 30 : 강한 증거")
    print("  3 < BF < 10  : 중간 증거")
    print("  1 < BF < 3   : 약한 증거")
    print("  BF = 1       : 증거 없음")
    print("  BF < 1       : M₀ 지지")
    print()
    print("주의: 베이즈 팩터 계산은 일반적으로 어려움")
    print("      PyMC에서는 Savage-Dickey ratio 등의 방법 사용")

bayes_factor_example()
```

---

## 6. 사후 예측 검사 (Posterior Predictive Checks)

### 6.1 개념

사후 예측 분포: 관측된 데이터를 학습한 모델이 생성할 것으로 예상되는 데이터

$$p(\tilde{y}|y) = \int p(\tilde{y}|\theta) p(\theta|y) d\theta$$

### 6.2 구현

```python
# 포아송 회귀 예제
np.random.seed(42)
n = 100
x = np.random.uniform(0, 5, n)
true_rate = np.exp(0.5 + 0.3 * x)
y_count = np.random.poisson(true_rate)

# 잘못된 모델: 정규분포 가정
with pm.Model() as wrong_model:
    beta0 = pm.Normal('beta0', mu=0, sigma=5)
    beta1 = pm.Normal('beta1', mu=0, sigma=5)
    sigma = pm.HalfNormal('sigma', sigma=5)

    mu = beta0 + beta1 * x
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_count)

    trace_wrong = pm.sample(2000, tune=1000, cores=1, random_seed=42)

    # 사후 예측
    ppc_wrong = pm.sample_posterior_predictive(trace_wrong, random_seed=42)

# 올바른 모델: 포아송 가정
with pm.Model() as correct_model:
    beta0 = pm.Normal('beta0', mu=0, sigma=5)
    beta1 = pm.Normal('beta1', mu=0, sigma=5)

    log_rate = beta0 + beta1 * x
    rate = pm.math.exp(log_rate)

    y_obs = pm.Poisson('y_obs', mu=rate, observed=y_count)

    trace_correct = pm.sample(2000, tune=1000, cores=1, random_seed=42)

    ppc_correct = pm.sample_posterior_predictive(trace_correct, random_seed=42)

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 잘못된 모델의 PPC
ax = axes[0, 0]
ppc_data_wrong = ppc_wrong.posterior_predictive['y_obs'].values.flatten()
ax.hist(y_count, bins=20, density=True, alpha=0.7, label='Observed')
ax.hist(ppc_data_wrong[:len(y_count)*10], bins=20, density=True, alpha=0.5, label='PPC')
ax.set_xlabel('y')
ax.set_ylabel('Density')
ax.set_title('잘못된 모델 (Normal): PPC')
ax.legend()

# 올바른 모델의 PPC
ax = axes[0, 1]
ppc_data_correct = ppc_correct.posterior_predictive['y_obs'].values.flatten()
ax.hist(y_count, bins=20, density=True, alpha=0.7, label='Observed')
ax.hist(ppc_data_correct[:len(y_count)*10], bins=20, density=True, alpha=0.5, label='PPC')
ax.set_xlabel('y')
ax.set_ylabel('Density')
ax.set_title('올바른 모델 (Poisson): PPC')
ax.legend()

# 잔차 분석 - 잘못된 모델
ax = axes[1, 0]
mean_pred_wrong = ppc_wrong.posterior_predictive['y_obs'].mean(dim=['chain', 'draw']).values
residuals_wrong = y_count - mean_pred_wrong
ax.scatter(mean_pred_wrong, residuals_wrong, alpha=0.5)
ax.axhline(0, color='r', linestyle='--')
ax.set_xlabel('Predicted')
ax.set_ylabel('Residual')
ax.set_title('잘못된 모델: 잔차')

# 잔차 분석 - 올바른 모델
ax = axes[1, 1]
mean_pred_correct = ppc_correct.posterior_predictive['y_obs'].mean(dim=['chain', 'draw']).values
residuals_correct = y_count - mean_pred_correct
ax.scatter(mean_pred_correct, residuals_correct, alpha=0.5)
ax.axhline(0, color='r', linestyle='--')
ax.set_xlabel('Predicted')
ax.set_ylabel('Residual')
ax.set_title('올바른 모델: 잔차')

plt.tight_layout()
plt.show()
```

### 6.3 ArviZ를 사용한 PPC

```python
# ArviZ PPC 플롯
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 관측 데이터를 InferenceData에 추가
idata_wrong = az.from_pymc3(trace_wrong, posterior_predictive=ppc_wrong)
idata_wrong.add_groups({'observed_data': {'y_obs': y_count}})

idata_correct = az.from_pymc3(trace_correct, posterior_predictive=ppc_correct)
idata_correct.add_groups({'observed_data': {'y_obs': y_count}})

# PPC 플롯
az.plot_ppc(idata_wrong, ax=axes[0], num_pp_samples=100)
axes[0].set_title('잘못된 모델 (Normal)')

az.plot_ppc(idata_correct, ax=axes[1], num_pp_samples=100)
axes[1].set_title('올바른 모델 (Poisson)')

plt.tight_layout()
plt.show()
```

---

## 7. 수렴 진단

### 7.1 R-hat (Gelman-Rubin 통계량)

```python
def demonstrate_convergence_diagnostics():
    """수렴 진단 시연"""

    # 여러 체인으로 샘플링
    np.random.seed(42)
    data = np.random.normal(5, 2, 50)

    with pm.Model() as model:
        mu = pm.Normal('mu', mu=0, sigma=10)
        sigma = pm.HalfNormal('sigma', sigma=5)
        y = pm.Normal('y', mu=mu, sigma=sigma, observed=data)

        # 여러 체인
        trace = pm.sample(2000, tune=1000, chains=4, cores=1, random_seed=42)

    # R-hat 계산
    summary = az.summary(trace)
    print("=== 수렴 진단 ===")
    print(summary[['mean', 'sd', 'r_hat', 'ess_bulk', 'ess_tail']])

    print("\n해석:")
    print("  R-hat < 1.01: 수렴 양호")
    print("  ESS_bulk > 400: 벌크 ESS 충분")
    print("  ESS_tail > 400: 꼬리 ESS 충분")

    return trace

trace_diagnostic = demonstrate_convergence_diagnostics()

# 체인별 트레이스 플롯
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
az.plot_trace(trace_diagnostic, var_names=['mu', 'sigma'], axes=axes)
plt.suptitle('다중 체인 트레이스 플롯')
plt.tight_layout()
plt.show()
```

### 7.2 자기상관 분석

```python
# 자기상관 플롯
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

az.plot_autocorr(trace_diagnostic, var_names=['mu'], ax=axes[0])
axes[0].set_title('μ 자기상관')

az.plot_autocorr(trace_diagnostic, var_names=['sigma'], ax=axes[1])
axes[1].set_title('σ 자기상관')

plt.tight_layout()
plt.show()

print("자기상관이 빠르게 0으로 감소하면 좋은 혼합을 의미")
```

### 7.3 유효 표본 크기 (ESS)

```python
# ESS 계산
ess = az.ess(trace_diagnostic)
print("=== 유효 표본 크기 (ESS) ===")
for var in ['mu', 'sigma']:
    print(f"{var}: bulk={ess[var].values:.0f}, tail={az.ess(trace_diagnostic, method='tail')[var].values:.0f}")

print("\n권장: ESS > 400 (최소 100 이상)")
print("ESS가 낮으면 더 많은 샘플이 필요하거나 모델/샘플러 개선 필요")
```

---

## 8. 실습 예제

### 8.1 계층적 모델 (Hierarchical Model)

```python
# 여러 학교의 학생 성적 데이터
np.random.seed(42)

n_schools = 8
n_students_per_school = np.random.randint(15, 30, n_schools)

# 진짜 학교 효과
true_school_mean = 70
true_school_std = 5
true_school_effects = np.random.normal(0, true_school_std, n_schools)
true_noise_std = 10

# 데이터 생성
schools = []
scores = []
for i in range(n_schools):
    school_mean = true_school_mean + true_school_effects[i]
    student_scores = np.random.normal(school_mean, true_noise_std, n_students_per_school[i])
    schools.extend([i] * n_students_per_school[i])
    scores.extend(student_scores)

schools = np.array(schools)
scores = np.array(scores)

print(f"학교 수: {n_schools}")
print(f"총 학생 수: {len(scores)}")
print(f"학교별 평균 점수: {[scores[schools == i].mean():.1f for i in range(n_schools)]}")

# 계층적 모델
with pm.Model() as hierarchical_model:
    # 초모수 (hyperparameters)
    mu_school = pm.Normal('mu_school', mu=70, sigma=20)
    sigma_school = pm.HalfNormal('sigma_school', sigma=10)

    # 학교 효과
    school_effect = pm.Normal('school_effect', mu=0, sigma=sigma_school, shape=n_schools)

    # 잔차 표준편차
    sigma_residual = pm.HalfNormal('sigma_residual', sigma=10)

    # 기대값
    mu = mu_school + school_effect[schools]

    # 가능도
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma_residual, observed=scores)

    # 샘플링
    trace_hier = pm.sample(3000, tune=1500, cores=1, random_seed=42)

# 결과
print("\n=== 계층적 모델 결과 ===")
print(az.summary(trace_hier, var_names=['mu_school', 'sigma_school', 'sigma_residual']))

# 학교 효과 비교
school_effects_post = trace_hier.posterior['school_effect'].mean(dim=['chain', 'draw']).values
school_effects_ci = az.hdi(trace_hier, var_names=['school_effect'])['school_effect'].values

fig, ax = plt.subplots(figsize=(10, 6))
school_ids = np.arange(n_schools)

# 사후 평균과 신용구간
ax.errorbar(school_ids, school_effects_post,
            yerr=[school_effects_post - school_effects_ci[:, 0],
                  school_effects_ci[:, 1] - school_effects_post],
            fmt='o', capsize=5, label='Posterior (hierarchical)')

# 실제 효과
ax.scatter(school_ids + 0.1, true_school_effects, marker='x', s=100,
           color='r', label='True effects', zorder=5)

# 개별 학교 평균 (no pooling)
school_means = [scores[schools == i].mean() - true_school_mean for i in range(n_schools)]
ax.scatter(school_ids - 0.1, school_means, marker='s', s=60,
           color='g', alpha=0.5, label='No pooling (raw)')

ax.axhline(0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('School ID')
ax.set_ylabel('School Effect')
ax.set_title('학교 효과: 계층적 모델 vs 개별 추정')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()

print("\n수축(Shrinkage) 효과:")
print("계층적 모델은 극단적인 추정치를 전체 평균 방향으로 수축시킴")
```

---

## 9. 연습 문제

### 문제 1: MH 알고리즘 구현
다음 목표 분포에서 Metropolis-Hastings로 샘플을 추출하세요:
- 목표: Gamma(5, 1)
- 제안 분포: 현재 값에서 ±0.5 균등분포

### 문제 2: PyMC 모델링
다음 데이터에 대해 PyMC로 베이지안 추론을 수행하세요:
- 20개 제품 중 3개 불량
- Beta(1,1) 사전분포
- 불량률의 사후 평균과 95% 신용구간 계산

### 문제 3: 모델 비교
선형 회귀와 다항 회귀(2차, 3차)를 비교하세요:
- 데이터 생성: y = 2 + 3x + 0.5x² + noise
- WAIC로 모델 선택
- 사후 예측 검사 수행

### 문제 4: 수렴 진단
여러 체인으로 샘플링하고 다음을 확인하세요:
- R-hat이 1.01 미만인지
- ESS가 충분한지
- 트레이스 플롯이 잘 섞이는지

---

## 10. 핵심 요약

### MCMC 핵심

1. **Metropolis-Hastings**: 제안-수락/거부 반복
2. **수락률**: 20-50%가 적절
3. **Burn-in**: 초기 불안정한 샘플 제거

### PyMC 워크플로우

```python
with pm.Model() as model:
    # 1. 사전분포 정의
    theta = pm.Distribution('theta', ...)

    # 2. 가능도 정의
    y = pm.Distribution('y', ..., observed=data)

    # 3. 샘플링
    trace = pm.sample(...)

    # 4. 사후 예측
    ppc = pm.sample_posterior_predictive(trace)

# 5. 진단 및 요약
az.summary(trace)
az.plot_trace(trace)
az.plot_ppc(...)
```

### 수렴 진단 체크리스트

- [ ] R-hat < 1.01
- [ ] ESS > 400
- [ ] 트레이스 플롯이 안정적
- [ ] 자기상관이 빠르게 감소
- [ ] 체인들이 잘 섞임

### 다음 장 미리보기

10장 **시계열 분석 기초**에서는:
- 시계열 구성요소 (추세, 계절성, 잡음)
- 정상성과 단위근 검정
- ACF/PACF 분석
- 시계열 분해
