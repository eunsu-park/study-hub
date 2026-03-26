# 12. 샘플링과 몬테카를로 방법

## 학습 목표

- 몬테카를로 방법의 기본 원리와 수렴 속도를 이해한다
- 기본 샘플링 방법(역변환, 기각 샘플링)을 구현하고 한계를 파악한다
- 중요도 샘플링을 통한 분산 감소 기법을 습득한다
- MCMC(메트로폴리스-헤이스팅스, 깁스 샘플링)의 원리와 구현을 이해한다
- 재매개변수화 트릭의 수학적 배경과 VAE에서의 역할을 파악한다
- 머신러닝에서 샘플링이 사용되는 다양한 사례를 학습한다

---

## 1. 왜 샘플링인가?

### 1.1 해석적으로 계산 불가능한 적분

많은 머신러닝 문제에서 다음과 같은 적분을 계산해야 합니다:

$$
\mathbb{E}_{p(x)}[f(x)] = \int f(x) p(x) dx
$$

예를 들어:
- **베이지안 추론**: 사후 분포의 정규화 상수 $\int p(x|\theta) p(\theta) d\theta$
- **강화학습**: 정책의 기대 보상 $\mathbb{E}_{\pi}[R]$
- **VAE**: ELBO의 기대값 $\mathbb{E}_{q(z|x)}[\log p(x|z)]$

이러한 적분은 고차원에서 해석적으로 계산이 불가능합니다.

### 1.2 몬테카를로 추정

**몬테카를로 원리**: 분포 $p(x)$에서 $N$개의 샘플 $x_1, \ldots, x_N$을 뽑으면:

$$
\mathbb{E}_{p(x)}[f(x)] \approx \frac{1}{N} \sum_{i=1}^{N} f(x_i)
$$

이는 **대수의 법칙(Law of Large Numbers)**에 의해 보장됩니다.

### 1.3 수렴 속도

몬테카를로 추정의 표준 오차는:

$$
\text{SE} = \frac{\sigma}{\sqrt{N}}
$$

여기서 $\sigma^2 = \text{Var}_{p(x)}[f(x)]$입니다.

**중요한 관찰**:
- 오차가 $O(1/\sqrt{N})$로 감소 → 정확도를 10배 높이려면 샘플이 100배 필요
- 차원에 **무관**: 고차원 적분에도 적용 가능 (격자 방법은 차원의 저주를 겪음)

```python
import numpy as np
import matplotlib.pyplot as plt

# 몬테카를로로 원주율 π 추정
def estimate_pi(n_samples):
    """단위 사각형 내 랜덤 점을 생성하고 단위 원 내부 비율로 π 추정"""
    x = np.random.uniform(-1, 1, n_samples)
    y = np.random.uniform(-1, 1, n_samples)
    inside_circle = (x**2 + y**2) <= 1
    pi_estimate = 4 * np.mean(inside_circle)
    return pi_estimate

# 샘플 수에 따른 수렴
sample_sizes = [10, 100, 1000, 10000, 100000]
estimates = [estimate_pi(n) for n in sample_sizes]

print("몬테카를로 π 추정:")
for n, est in zip(sample_sizes, estimates):
    error = abs(est - np.pi)
    print(f"N={n:6d}: π ≈ {est:.6f}, 오차 = {error:.6f}")

# 수렴 시각화
n_trials = 100
sample_range = np.logspace(1, 5, 50, dtype=int)
errors = []

for n in sample_range:
    trial_estimates = [estimate_pi(n) for _ in range(n_trials)]
    errors.append(np.std(trial_estimates))

plt.figure(figsize=(10, 6))
plt.loglog(sample_range, errors, 'b-', label='실제 표준 오차')
plt.loglog(sample_range, 1/np.sqrt(sample_range), 'r--', label=r'$O(1/\sqrt{N})$')
plt.xlabel('샘플 수 (N)')
plt.ylabel('표준 오차')
plt.title('몬테카를로 추정의 수렴 속도')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 2. 기본 샘플링 방법

### 2.1 역변환 샘플링 (Inverse Transform Sampling)

**정리**: $U \sim \text{Uniform}(0, 1)$이고 $F$가 누적분포함수(CDF)일 때, $X = F^{-1}(U)$는 CDF $F$를 따릅니다.

**알고리즘**:
1. $u \sim \text{Uniform}(0, 1)$ 생성
2. $x = F^{-1}(u)$ 계산

```python
# 역변환 샘플링 예제: 지수분포
def inverse_transform_exponential(n_samples, lambda_param=1.0):
    """지수분포 Exp(λ)에서 샘플링"""
    u = np.random.uniform(0, 1, n_samples)
    # F^(-1)(u) = -ln(1-u)/λ
    x = -np.log(1 - u) / lambda_param
    return x

# 검증
samples = inverse_transform_exponential(10000, lambda_param=2.0)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(samples, bins=50, density=True, alpha=0.7, label='샘플링 결과')
x_range = np.linspace(0, 5, 100)
plt.plot(x_range, 2 * np.exp(-2 * x_range), 'r-', linewidth=2, label='이론 PDF')
plt.xlabel('x')
plt.ylabel('밀도')
plt.title('역변환 샘플링: 지수분포')
plt.legend()

plt.subplot(1, 2, 2)
# Q-Q plot
from scipy import stats
stats.probplot(samples, dist=stats.expon(scale=0.5), plot=plt)
plt.title('Q-Q Plot')
plt.tight_layout()
plt.show()
```

**한계**: $F^{-1}$를 계산하기 어려운 경우가 많습니다.

### 2.2 기각 샘플링 (Rejection Sampling)

목표 분포 $p(x)$에서 직접 샘플링하기 어려울 때, 제안 분포(proposal) $q(x)$를 사용합니다.

**알고리즘**:
1. 상수 $M$을 선택하여 $M q(x) \geq p(x)$ (모든 $x$에 대해)
2. $x \sim q(x)$ 생성
3. $u \sim \text{Uniform}(0, 1)$ 생성
4. $u \leq \frac{p(x)}{M q(x)}$이면 $x$ 수락, 아니면 거부

**수락률**: $\frac{1}{M}$

```python
# 기각 샘플링 예제: 베타분포 Beta(2, 5)
def target_pdf(x):
    """목표 분포: Beta(2, 5)"""
    if 0 <= x <= 1:
        return 30 * x * (1-x)**4  # 정규화된 Beta(2,5)
    return 0

def proposal_pdf(x):
    """제안 분포: Uniform(0, 1)"""
    return 1.0 if 0 <= x <= 1 else 0

# M 찾기 (최대값)
x_grid = np.linspace(0, 1, 1000)
M = max(target_pdf(x) / proposal_pdf(x) for x in x_grid)
print(f"M = {M:.2f}")

def rejection_sampling(n_samples):
    """기각 샘플링 구현"""
    samples = []
    n_rejected = 0

    while len(samples) < n_samples:
        x = np.random.uniform(0, 1)  # 제안 분포에서 샘플
        u = np.random.uniform(0, 1)

        if u <= target_pdf(x) / (M * proposal_pdf(x)):
            samples.append(x)
        else:
            n_rejected += 1

    acceptance_rate = n_samples / (n_samples + n_rejected)
    print(f"수락률: {acceptance_rate:.2%} (이론값: {1/M:.2%})")
    return np.array(samples)

samples = rejection_sampling(10000)

plt.figure(figsize=(10, 6))
plt.hist(samples, bins=50, density=True, alpha=0.7, label='샘플링 결과')
x_range = np.linspace(0, 1, 100)
plt.plot(x_range, [target_pdf(x) for x in x_range], 'r-', linewidth=2, label='목표 PDF')
plt.xlabel('x')
plt.ylabel('밀도')
plt.title('기각 샘플링: Beta(2, 5)')
plt.legend()
plt.show()
```

### 2.3 고차원에서의 한계

기각 샘플링의 수락률은 $1/M$입니다. 고차원에서 $M$은 기하급수적으로 증가하여 비효율적입니다.

**예**: 10차원 가우스 분포에서 $M \sim e^{10}$ → 수락률 $< 0.01\%$

---

## 3. 중요도 샘플링 (Importance Sampling)

### 3.1 기본 원리

목표: $\mathbb{E}_{p(x)}[f(x)]$ 계산

**중요도 샘플링 항등식**:

$$
\mathbb{E}_{p(x)}[f(x)] = \int f(x) p(x) dx = \int f(x) \frac{p(x)}{q(x)} q(x) dx = \mathbb{E}_{q(x)}\left[f(x) \frac{p(x)}{q(x)}\right]
$$

$q(x)$를 **제안 분포**, $w(x) = \frac{p(x)}{q(x)}$를 **중요도 가중치**라고 합니다.

**몬테카를로 추정**:
$$
\mathbb{E}_{p(x)}[f(x)] \approx \frac{1}{N} \sum_{i=1}^{N} f(x_i) w(x_i), \quad x_i \sim q(x)
$$

### 3.2 제안 분포 선택

**좋은 제안 분포**:
- $q(x)$가 $|f(x)| p(x)$와 비례하면 분산 최소
- 실제로는 $p(x)$의 꼬리가 두꺼운 곳을 커버해야 함

**나쁜 선택**: $q(x)$가 $p(x)$보다 얇은 꼬리 → 가중치가 폭발하여 높은 분산

### 3.3 자기정규화 중요도 샘플링

$p(x)$를 정규화 상수까지만 알 때 ($p(x) = \tilde{p}(x)/Z$):

$$
\mathbb{E}_{p(x)}[f(x)] \approx \frac{\sum_{i=1}^{N} f(x_i) \tilde{w}(x_i)}{\sum_{i=1}^{N} \tilde{w}(x_i)}, \quad \tilde{w}(x_i) = \frac{\tilde{p}(x_i)}{q(x_i)}
$$

```python
# 중요도 샘플링 예제
def target_unnormalized(x):
    """정규화되지 않은 목표 분포: 혼합 가우스"""
    return 0.3 * np.exp(-0.5 * ((x-2)/0.5)**2) + \
           0.7 * np.exp(-0.5 * ((x+2)/1.0)**2)

def proposal(x):
    """제안 분포: N(0, 3)"""
    return np.exp(-0.5 * (x/3)**2) / (3 * np.sqrt(2*np.pi))

def f(x):
    """계산하려는 함수: x^2"""
    return x**2

# 일반 몬테카를로 (목표 분포에서 직접 샘플링이 가능하다고 가정)
def naive_monte_carlo(n_samples):
    # 기각 샘플링으로 목표 분포에서 샘플
    samples = []
    while len(samples) < n_samples:
        x = np.random.normal(0, 3)
        u = np.random.uniform(0, target_unnormalized(x))
        if u <= target_unnormalized(x):
            samples.append(x)
    samples = np.array(samples)
    return np.mean(f(samples))

# 중요도 샘플링
def importance_sampling(n_samples):
    # 제안 분포에서 샘플
    samples = np.random.normal(0, 3, n_samples)
    weights = target_unnormalized(samples) / proposal(samples)
    weights = weights / np.sum(weights)  # 자기정규화
    return np.sum(f(samples) * weights * n_samples)

# 비교
n_trials = 100
n_samples = 1000

naive_results = [naive_monte_carlo(n_samples) for _ in range(n_trials)]
is_results = [importance_sampling(n_samples) for _ in range(n_trials)]

print(f"일반 MC: 평균 = {np.mean(naive_results):.4f}, 표준편차 = {np.std(naive_results):.4f}")
print(f"중요도 샘플링: 평균 = {np.mean(is_results):.4f}, 표준편차 = {np.std(is_results):.4f}")
print(f"분산 감소율: {np.var(naive_results) / np.var(is_results):.2f}x")
```

### 3.4 강화학습과의 연결

정책 경사법(Policy Gradient)에서 중요도 샘플링이 핵심입니다:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_{\theta_{\text{old}}}} \left[ \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} \nabla_\theta \log \pi_\theta(a|s) Q(s,a) \right]
$$

이는 **off-policy learning**을 가능하게 합니다 (PPO, TRPO).

---

## 4. 마르코프 체인 몬테카를로 (MCMC)

### 4.1 마르코프 체인 기초

**마르코프 체인**: 확률적 상태 전이 $P(x_{t+1} | x_t)$

**정상 분포(Stationary Distribution)**: $\pi(x) = \int \pi(x') P(x|x') dx'$

**세부 균형 조건(Detailed Balance)**:
$$
\pi(x) P(x'|x) = \pi(x') P(x|x')
$$

세부 균형을 만족하면 $\pi$가 정상 분포입니다.

### 4.2 메트로폴리스-헤이스팅스 알고리즘

**목표**: 목표 분포 $\pi(x)$ (정규화 상수 불필요)에서 샘플링

**알고리즘**:
1. 현재 상태 $x_t$에서 시작
2. 제안 분포 $q(x'|x_t)$에서 후보 $x'$ 생성
3. 수락 확률 계산:
$$
\alpha(x', x_t) = \min\left(1, \frac{\pi(x') q(x_t|x')}{\pi(x_t) q(x'|x_t)}\right)
$$
4. 확률 $\alpha$로 $x_{t+1} = x'$ (수락), 아니면 $x_{t+1} = x_t$ (거부)

```python
# 메트로폴리스-헤이스팅스 구현
def target_distribution(x):
    """목표 분포: 이봉 분포 (bimodal)"""
    return np.exp(-0.5 * ((x-3)/0.8)**2) + np.exp(-0.5 * ((x+3)/0.8)**2)

def metropolis_hastings(n_samples, proposal_std=1.0):
    """MH 알고리즘 (대칭 제안 분포)"""
    samples = np.zeros(n_samples)
    x = 0.0  # 초기 상태
    n_accepted = 0

    for i in range(n_samples):
        # 제안 생성 (대칭 가우스)
        x_proposed = x + np.random.normal(0, proposal_std)

        # 수락 확률 계산 (대칭 제안이므로 q 항 소거)
        acceptance_prob = min(1, target_distribution(x_proposed) / target_distribution(x))

        # 수락/거부 결정
        if np.random.uniform() < acceptance_prob:
            x = x_proposed
            n_accepted += 1

        samples[i] = x

    acceptance_rate = n_accepted / n_samples
    print(f"수락률: {acceptance_rate:.2%}")
    return samples

# 샘플링
samples = metropolis_hastings(50000, proposal_std=2.0)

# 번인(burn-in) 제거
burn_in = 5000
samples_burned = samples[burn_in:]

plt.figure(figsize=(15, 5))

# 궤적
plt.subplot(1, 3, 1)
plt.plot(samples[:1000])
plt.xlabel('반복')
plt.ylabel('x')
plt.title('MCMC 궤적 (처음 1000개)')
plt.axvline(burn_in, color='r', linestyle='--', label='번인 종료')

# 히스토그램
plt.subplot(1, 3, 2)
plt.hist(samples_burned, bins=50, density=True, alpha=0.7, label='MCMC 샘플')
x_range = np.linspace(-6, 6, 200)
plt.plot(x_range, target_distribution(x_range) /
         (np.sqrt(2*np.pi*0.8**2) * 2), 'r-', linewidth=2, label='목표 분포')
plt.xlabel('x')
plt.ylabel('밀도')
plt.title('샘플 분포')
plt.legend()

# 자기상관
plt.subplot(1, 3, 3)
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(samples_burned[::10], lags=50, ax=plt.gca())
plt.title('자기상관 함수 (ACF)')
plt.tight_layout()
plt.show()
```

### 4.3 깁스 샘플링 (Gibbs Sampling)

다변량 분포 $p(x_1, \ldots, x_d)$에서 조건부 분포 $p(x_i | x_{-i})$를 알 때 사용합니다.

**알고리즘**:
1. 각 변수를 순회하며
2. $x_i^{(t+1)} \sim p(x_i | x_1^{(t+1)}, \ldots, x_{i-1}^{(t+1)}, x_{i+1}^{(t)}, \ldots, x_d^{(t)})$

```python
# 깁스 샘플링: 이변량 가우스
def gibbs_sampling_bivariate_gaussian(n_samples, rho=0.9):
    """이변량 가우스 N([0,0], [[1,ρ],[ρ,1]])에서 샘플링"""
    samples = np.zeros((n_samples, 2))
    x, y = 0.0, 0.0  # 초기값

    for i in range(n_samples):
        # x | y ~ N(ρy, 1-ρ²)
        x = np.random.normal(rho * y, np.sqrt(1 - rho**2))
        # y | x ~ N(ρx, 1-ρ²)
        y = np.random.normal(rho * x, np.sqrt(1 - rho**2))
        samples[i] = [x, y]

    return samples

samples = gibbs_sampling_bivariate_gaussian(10000, rho=0.9)
samples = samples[1000:]  # 번인 제거

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('깁스 샘플링: 이변량 가우스 (ρ=0.9)')
plt.axis('equal')

plt.subplot(1, 2, 2)
# 이론적 등고선
from scipy.stats import multivariate_normal
x_grid = np.linspace(-4, 4, 100)
y_grid = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x_grid, y_grid)
pos = np.dstack((X, Y))
rv = multivariate_normal([0, 0], [[1, 0.9], [0.9, 1]])
plt.contour(X, Y, rv.pdf(pos), levels=10)
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.1, s=1, c='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('이론 분포와 비교')
plt.axis('equal')
plt.tight_layout()
plt.show()
```

### 4.4 번인과 자기상관

- **번인(Burn-in)**: 초기 샘플은 정상 분포에 도달하지 않았으므로 버립니다
- **자기상관**: 연속 샘플이 상관관계를 가짐 → 간격을 두고 샘플링 (thinning)
- **유효 샘플 크기**: $n_{\text{eff}} = \frac{n}{1 + 2\sum_{k=1}^{\infty} \rho_k}$

---

## 5. 재매개변수화 트릭 (Reparameterization Trick)

### 5.1 문제: 확률적 노드의 역전파

VAE에서 인코더는 $q_\phi(z|x)$를 출력하고, $z \sim q_\phi(z|x)$를 샘플링합니다. 손실 함수:

$$
\mathcal{L}(\phi) = \mathbb{E}_{z \sim q_\phi(z|x)}[f(z)]
$$

**문제**: $\frac{\partial}{\partial \phi} \mathbb{E}_{z \sim q_\phi}[f(z)]$를 어떻게 계산할까?

샘플링 연산은 **미분 불가능**합니다.

### 5.2 재매개변수화 해법

**아이디어**: 확률성을 $\phi$와 무관한 노이즈로 분리합니다.

**가우스 분포**: $z \sim \mathcal{N}(\mu_\phi, \sigma_\phi^2)$

재매개변수화: $z = \mu_\phi + \sigma_\phi \cdot \epsilon$, 여기서 $\epsilon \sim \mathcal{N}(0, 1)$

이제:
$$
\frac{\partial}{\partial \phi} \mathbb{E}_{\epsilon \sim \mathcal{N}(0,1)}[f(\mu_\phi + \sigma_\phi \epsilon)] = \mathbb{E}_{\epsilon}\left[\frac{\partial f(z)}{\partial z} \frac{\partial z}{\partial \phi}\right]
$$

**기울기가 샘플링 내부로 들어갈 수 있습니다!**

```python
import torch
import torch.nn as nn

# 재매개변수화 트릭 구현
class VAEEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x):
        h = torch.relu(self.fc(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """재매개변수화 트릭"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # ε ~ N(0, 1)
        z = mu + std * eps           # z = μ + σε
        return z

# 기울기 흐름 테스트
encoder = VAEEncoder(input_dim=10, latent_dim=2)
x = torch.randn(32, 10)

mu, logvar = encoder(x)
z = encoder.reparameterize(mu, logvar)

# 임의의 손실 함수
loss = (z ** 2).sum()
loss.backward()

print("기울기가 성공적으로 계산됨:")
print(f"  mu의 기울기: {encoder.fc_mu.weight.grad is not None}")
print(f"  logvar의 기울기: {encoder.fc_logvar.weight.grad is not None}")
print(f"  기울기 norm: {encoder.fc_mu.weight.grad.norm():.4f}")
```

### 5.3 다른 분포의 재매개변수화

**베르누이**: Gumbel-Softmax 트릭
$$
z = \text{one-hot}\left(\arg\max_i \left[\log \pi_i + G_i\right]\right)
$$
여기서 $G_i \sim \text{Gumbel}(0, 1)$

**감마 분포**: Shape augmentation

**일반적 원칙**: $z = g(\epsilon; \theta)$로 표현 가능하면 재매개변수화 가능

### 5.4 VAE의 ELBO

VAE 손실 함수 (ELBO):

$$
\mathcal{L} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))
$$

**재매개변수화 적용**:
- 첫 번째 항: 재매개변수화 트릭으로 기울기 계산
- 두 번째 항: 가우스 사전분포 가정 시 해석적으로 계산 가능

```python
def vae_loss(x, x_recon, mu, logvar):
    """VAE 손실 함수 (ELBO)"""
    # 재구성 손실 (log p(x|z))
    recon_loss = nn.functional.binary_cross_entropy(x_recon, x, reduction='sum')

    # KL 발산 (가우스 가정 하에 해석적 계산)
    # KL(N(μ,σ²) || N(0,1)) = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_loss

# 전체 VAE 학습 루프 (간략화)
def train_vae(encoder, decoder, data_loader, optimizer, epochs):
    for epoch in range(epochs):
        for x in data_loader:
            # 인코더
            mu, logvar = encoder(x)
            z = encoder.reparameterize(mu, logvar)

            # 디코더
            x_recon = decoder(z)

            # 손실 계산 및 역전파
            loss = vae_loss(x, x_recon, mu, logvar)
            optimizer.zero_grad()
            loss.backward()  # 재매개변수화 덕분에 기울기 계산 가능!
            optimizer.step()
```

---

## 6. 머신러닝 응용

### 6.1 VAE: ELBO 최적화

재매개변수화 트릭이 없었다면 VAE는 불가능했습니다. REINFORCE (점수 함수 추정)는 분산이 너무 높습니다.

### 6.2 강화학습: REINFORCE = 몬테카를로

정책 경사 정리:
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) R(\tau)\right]
$$

이는 몬테카를로 샘플링으로 추정합니다.

**분산 감소**: 베이스라인 사용, GAE (Generalized Advantage Estimation)

### 6.3 베이지안 추론: MCMC vs 변분 추론

**MCMC**:
- 장점: 이론적으로 정확한 샘플
- 단점: 느림, 수렴 진단 어려움

**변분 추론**:
- 장점: 빠름, 확장 가능
- 단점: 근사 품질이 $q$의 표현력에 제한됨

### 6.4 Dropout = 베르누이 샘플링

Dropout은 훈련 중 베르누이 샘플링입니다:
$$
h_{\text{drop}} = h \odot \text{Bernoulli}(p)
$$

**베이지안 해석**: Dropout은 근사 베이지안 추론으로 볼 수 있습니다 (Gal & Ghahramani, 2016).

```python
# MC Dropout으로 불확실성 추정
class MCDropoutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # 추론 시에도 dropout 적용
        x = self.fc2(x)
        return x

def predict_with_uncertainty(model, x, n_samples=100):
    """MC Dropout으로 예측 불확실성 추정"""
    model.train()  # Dropout 활성화
    predictions = []

    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(x)
            predictions.append(pred)

    predictions = torch.cat(predictions, dim=1)
    mean = predictions.mean(dim=1)
    std = predictions.std(dim=1)
    return mean, std

# 테스트
model = MCDropoutModel()
x_test = torch.randn(10, 10)
mean, std = predict_with_uncertainty(model, x_test, n_samples=100)
print(f"예측 평균: {mean[:5]}")
print(f"예측 표준편차 (불확실성): {std[:5]}")
```

---

## 연습 문제

### 문제 1: 몬테카를로 적분
다음 적분을 몬테카를로 방법으로 추정하세요:
$$
I = \int_0^1 e^{-x^2} dx
$$

(a) 10, 100, 1000, 10000개 샘플을 사용하여 추정하고 오차를 계산하세요.
(b) 오차가 $O(1/\sqrt{N})$로 감소하는지 log-log 플롯으로 확인하세요.
(c) `scipy.integrate.quad`의 결과와 비교하세요.

### 문제 2: 중요도 샘플링
목표 분포 $p(x) \propto e^{-|x|^3}$에서 $\mathbb{E}[x^2]$를 계산하려고 합니다.

(a) 제안 분포로 $q(x) = \mathcal{N}(0, 1)$을 사용한 중요도 샘플링을 구현하세요.
(b) 제안 분포로 $q(x) = \text{Laplace}(0, 1)$을 사용한 경우와 분산을 비교하세요.
(c) 어떤 제안 분포가 더 효율적인가요? 이유를 설명하세요.

### 문제 3: MCMC 진단
이봉 분포 $p(x) \propto e^{-(x-3)^2/2} + e^{-(x+3)^2/2}$에서 메트로폴리스-헤이스팅스 샘플링을 수행하세요.

(a) 제안 분포 표준편차를 0.1, 1.0, 10.0으로 변경하며 수락률을 관찰하세요.
(b) 각 경우의 자기상관 함수(ACF)를 플롯하고 유효 샘플 크기를 추정하세요.
(c) 최적의 제안 분포 표준편차는 무엇인가요?

### 문제 4: 재매개변수화 트릭
간단한 VAE를 구현하고 재매개변수화 트릭의 효과를 검증하세요.

(a) MNIST 데이터셋에서 작은 VAE를 훈련하세요 (latent_dim=2).
(b) 재매개변수화 트릭을 사용하지 않고 REINFORCE로 훈련할 때와 비교하세요.
(c) 잠재 공간을 2D 플롯으로 시각화하세요.

### 문제 5: 깁스 샘플링 확장
3변량 가우스 분포에서 깁스 샘플링을 구현하세요.

$$
\mathbf{x} \sim \mathcal{N}\left(\mathbf{0}, \begin{bmatrix} 1 & 0.8 & 0.5 \\ 0.8 & 1 & 0.7 \\ 0.5 & 0.7 & 1 \end{bmatrix}\right)
$$

(a) 조건부 분포 $p(x_i | x_{-i})$를 유도하세요 (가우스 조건부 공식 사용).
(b) 깁스 샘플링을 10,000회 반복하고 샘플 공분산을 계산하세요.
(c) 이론적 공분산과 비교하세요.

---

## 참고 자료

### 서적
- **Pattern Recognition and Machine Learning** (Bishop, 2006) - Chapter 11: Sampling Methods
- **Monte Carlo Statistical Methods** (Robert & Casella, 2004)
- **Deep Learning** (Goodfellow et al., 2016) - Chapter 17: Monte Carlo Methods

### 논문
- Kingma & Welling (2013), "Auto-Encoding Variational Bayes" - VAE와 재매개변수화 트릭
- Metropolis et al. (1953), "Equation of State Calculations by Fast Computing Machines"
- Hastings (1970), "Monte Carlo Sampling Methods Using Markov Chains"
- Gal & Ghahramani (2016), "Dropout as a Bayesian Approximation"

### 온라인 자료
- [MCMC Interactive Gallery](https://chi-feng.github.io/mcmc-demo/)
- [Distill.pub: Visualizing Sampling Methods](https://distill.pub)
- [PyMC Documentation](https://www.pymc.io) - 실제 베이지안 추론 라이브러리

### 라이브러리
- **PyMC3/PyMC4**: 베이지안 추론 및 MCMC
- **Stan**: 강력한 MCMC 추론 엔진
- **PyTorch**: VAE 구현
- **NumPyro**: JAX 기반 확률적 프로그래밍
