# 09. 최대우도추정과 MAP (Maximum Likelihood and MAP)

## 학습 목표

- 우도 함수와 확률의 차이를 이해하고, 로그 우도의 역할을 파악한다
- 최대우도추정(MLE)의 정의와 다양한 확률분포에 대한 MLE를 유도할 수 있다
- MAP 추정의 원리를 이해하고 MLE와의 차이점을 설명할 수 있다
- 정규화 항이 사전분포와 어떻게 연결되는지 수학적으로 유도할 수 있다
- EM 알고리즘의 원리를 이해하고 잠재 변수가 있는 모델에 적용할 수 있다
- 머신러닝에서 MLE와 MAP가 어떻게 활용되는지 실제 예제를 통해 학습한다

---

## 1. 우도 함수 (Likelihood Function)

### 1.1 우도 vs 확률

**확률(Probability)**과 **우도(Likelihood)**는 같은 식으로 계산되지만, 의미가 다릅니다.

- **확률**: 파라미터 $\theta$가 고정되어 있을 때, 데이터 $D$가 발생할 확률
  $$P(D|\theta)$$

- **우도**: 데이터 $D$가 관측된 후, 파라미터 $\theta$에 대한 함수
  $$\mathcal{L}(\theta|D) = P(D|\theta)$$

**핵심 차이점**:
- 확률: $\theta$ 고정, $D$ 변수 → $\sum_D P(D|\theta) = 1$
- 우도: $D$ 고정, $\theta$ 변수 → $\int \mathcal{L}(\theta|D) d\theta \neq 1$ (일반적으로)

**예제**: 동전 던지기
- 동전의 앞면 확률이 $\theta = 0.7$일 때, 10번 던져서 7번 앞면이 나올 **확률**
- 10번 던져서 7번 앞면이 관측되었을 때, $\theta$가 0.7일 **우도**

### 1.2 로그 우도 (Log-Likelihood)

실제 계산에서는 **로그 우도**를 사용합니다:

$$\ell(\theta|D) = \log \mathcal{L}(\theta|D) = \log P(D|\theta)$$

**로그 우도를 사용하는 이유**:

1. **수치적 안정성**: 확률의 곱은 매우 작은 값이 되어 underflow 발생
2. **곱을 합으로 변환**: $\log(ab) = \log a + \log b$
3. **미분 계산 용이**: 지수 함수의 미분이 간단해짐
4. **최적화 동일**: $\log$는 단조증가 함수이므로 argmax는 동일

### 1.3 i.i.d. 가정과 로그를 통한 곱의 합 변환

데이터 $D = \{x_1, x_2, \ldots, x_n\}$이 **독립 항등 분포(i.i.d.)**를 따른다면:

$$P(D|\theta) = \prod_{i=1}^n P(x_i|\theta)$$

로그를 취하면:

$$\ell(\theta|D) = \log \prod_{i=1}^n P(x_i|\theta) = \sum_{i=1}^n \log P(x_i|\theta)$$

이 변환이 MLE의 핵심입니다.

## 2. 최대우도추정 (MLE)

### 2.1 MLE 정의

**최대우도추정(Maximum Likelihood Estimation, MLE)**은 관측된 데이터의 우도를 최대화하는 파라미터를 찾는 방법입니다:

$$\theta_{\text{MLE}} = \arg\max_{\theta} \mathcal{L}(\theta|D) = \arg\max_{\theta} \log P(D|\theta)$$

**계산 방법**:
1. 우도 함수 $\mathcal{L}(\theta|D)$ 작성
2. 로그 우도 $\ell(\theta|D)$ 계산
3. $\frac{\partial \ell}{\partial \theta} = 0$ 풀기
4. 2차 미분으로 최대값 확인

### 2.2 정규분포의 MLE

**문제**: $n$개의 데이터 $\{x_1, \ldots, x_n\}$이 $\mathcal{N}(\mu, \sigma^2)$를 따를 때, $\mu$와 $\sigma^2$의 MLE를 구하시오.

**해결**:

로그 우도:
$$\ell(\mu, \sigma^2) = \sum_{i=1}^n \log \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_i-\mu)^2}{2\sigma^2}\right)$$

$$= -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i-\mu)^2$$

**$\mu$에 대한 최적화**:

$$\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^n (x_i - \mu) = 0$$

$$\Rightarrow \mu_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n x_i$$

**$\sigma^2$에 대한 최적화**:

$$\frac{\partial \ell}{\partial \sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2(\sigma^2)^2}\sum_{i=1}^n (x_i-\mu)^2 = 0$$

$$\Rightarrow \sigma^2_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n (x_i - \mu_{\text{MLE}})^2$$

**결과**: 표본 평균과 표본 분산!

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 데이터 생성
np.random.seed(42)
true_mu = 5.0
true_sigma = 2.0
n_samples = 100
data = np.random.normal(true_mu, true_sigma, n_samples)

# MLE 계산
mu_mle = np.mean(data)
sigma2_mle = np.var(data, ddof=0)  # ddof=0: MLE (biased)
sigma_mle = np.sqrt(sigma2_mle)

print(f"True parameters: μ={true_mu}, σ={true_sigma}")
print(f"MLE estimates: μ={mu_mle:.3f}, σ={sigma_mle:.3f}")

# 로그 우도 함수 시각화
def log_likelihood(mu, sigma, data):
    n = len(data)
    return -n/2 * np.log(2*np.pi) - n * np.log(sigma) - \
           np.sum((data - mu)**2) / (2 * sigma**2)

# μ에 대한 로그 우도
mu_range = np.linspace(3, 7, 100)
ll_mu = [log_likelihood(m, true_sigma, data) for m in mu_range]

plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.plot(mu_range, ll_mu)
plt.axvline(mu_mle, color='r', linestyle='--', label=f'MLE={mu_mle:.2f}')
plt.xlabel('μ')
plt.ylabel('Log-Likelihood')
plt.title('Log-Likelihood vs μ')
plt.legend()
plt.grid(True)

# σ에 대한 로그 우도
sigma_range = np.linspace(0.5, 4, 100)
ll_sigma = [log_likelihood(true_mu, s, data) for s in sigma_range]

plt.subplot(132)
plt.plot(sigma_range, ll_sigma)
plt.axvline(sigma_mle, color='r', linestyle='--', label=f'MLE={sigma_mle:.2f}')
plt.xlabel('σ')
plt.ylabel('Log-Likelihood')
plt.title('Log-Likelihood vs σ')
plt.legend()
plt.grid(True)

# 히스토그램과 MLE 분포
plt.subplot(133)
plt.hist(data, bins=30, density=True, alpha=0.7, label='Data')
x = np.linspace(data.min(), data.max(), 100)
plt.plot(x, stats.norm.pdf(x, mu_mle, sigma_mle), 'r-',
         linewidth=2, label='MLE fit')
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Data vs MLE Distribution')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('mle_gaussian.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 2.3 베르누이 분포의 MLE

**문제**: $n$번의 동전 던지기에서 $k$번 앞면이 나왔을 때, 앞면 확률 $\theta$의 MLE는?

**해결**:

$$\mathcal{L}(\theta) = \theta^k (1-\theta)^{n-k}$$

$$\ell(\theta) = k \log \theta + (n-k) \log(1-\theta)$$

$$\frac{d\ell}{d\theta} = \frac{k}{\theta} - \frac{n-k}{1-\theta} = 0$$

$$\Rightarrow \theta_{\text{MLE}} = \frac{k}{n}$$

**직관**: 관측된 상대 빈도!

### 2.4 MLE의 성질

1. **일치성(Consistency)**: $n \to \infty$일 때, $\theta_{\text{MLE}} \to \theta_{\text{true}}$ (확률적으로)

2. **점근 정규성(Asymptotic Normality)**:
   $$\sqrt{n}(\theta_{\text{MLE}} - \theta_{\text{true}}) \xrightarrow{d} \mathcal{N}(0, I(\theta)^{-1})$$
   여기서 $I(\theta)$는 Fisher 정보 행렬

3. **불편성은 보장되지 않음**: 예를 들어, $\sigma^2_{\text{MLE}}$는 편향 추정량 (ddof=0)

4. **변환 불변성**: $\theta_{\text{MLE}}$가 $\theta$의 MLE이면, $g(\theta_{\text{MLE}})$는 $g(\theta)$의 MLE

## 3. MAP 추정 (Maximum A Posteriori)

### 3.1 베이즈 정리와 사후 분포

베이즈 정리:

$$P(\theta|D) = \frac{P(D|\theta)P(\theta)}{P(D)}$$

- $P(\theta|D)$: 사후 분포 (posterior)
- $P(D|\theta)$: 우도 (likelihood)
- $P(\theta)$: 사전 분포 (prior)
- $P(D)$: 증거 (evidence), 정규화 상수

### 3.2 MAP 정의

**MAP 추정(Maximum A Posteriori)**은 사후 분포를 최대화하는 파라미터를 찾습니다:

$$\theta_{\text{MAP}} = \arg\max_{\theta} P(\theta|D)$$

$$= \arg\max_{\theta} P(D|\theta)P(\theta)$$

$$= \arg\max_{\theta} \left[\log P(D|\theta) + \log P(\theta)\right]$$

**MLE와의 관계**:
- MLE: $\arg\max_{\theta} \log P(D|\theta)$
- MAP: $\arg\max_{\theta} \left[\log P(D|\theta) + \log P(\theta)\right]$

MAP = MLE + prior term

### 3.3 MLE vs MAP 비교

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 데이터: 10번 중 8번 앞면
n, k = 10, 8

# 우도 함수
theta_range = np.linspace(0.01, 0.99, 100)
likelihood = stats.binom.pmf(k, n, theta_range)

# MLE
theta_mle = k / n

# MAP with Beta prior (α, β)
alpha, beta = 2, 2  # 약한 사전분포
prior = stats.beta.pdf(theta_range, alpha, beta)
posterior = likelihood * prior
posterior = posterior / np.trapz(posterior, theta_range)  # 정규화
theta_map = theta_range[np.argmax(posterior)]

plt.figure(figsize=(14, 4))

# 우도
plt.subplot(131)
plt.plot(theta_range, likelihood / np.max(likelihood), label='Likelihood')
plt.axvline(theta_mle, color='r', linestyle='--', label=f'MLE={theta_mle:.2f}')
plt.xlabel('θ')
plt.ylabel('Normalized Likelihood')
plt.title('Likelihood Function')
plt.legend()
plt.grid(True)

# 사전분포
plt.subplot(132)
plt.plot(theta_range, prior, label=f'Beta({alpha},{beta})')
plt.xlabel('θ')
plt.ylabel('Density')
plt.title('Prior Distribution')
plt.legend()
plt.grid(True)

# 사후분포
plt.subplot(133)
plt.plot(theta_range, posterior, label='Posterior', linewidth=2)
plt.axvline(theta_mle, color='r', linestyle='--', alpha=0.7, label=f'MLE={theta_mle:.2f}')
plt.axvline(theta_map, color='g', linestyle='--', alpha=0.7, label=f'MAP={theta_map:.2f}')
plt.xlabel('θ')
plt.ylabel('Density')
plt.title('Posterior Distribution')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('mle_vs_map.png', dpi=150, bbox_inches='tight')
plt.show()

# 데이터가 적을 때의 차이
print("데이터가 적을 때 (n=10, k=8):")
print(f"MLE: {theta_mle:.3f}")
print(f"MAP: {theta_map:.3f}")

# 데이터가 많을 때
n2, k2 = 1000, 800
theta_mle2 = k2 / n2
likelihood2 = stats.binom.pmf(k2, n2, theta_range)
posterior2 = likelihood2 * prior
theta_map2 = theta_range[np.argmax(posterior2)]

print("\n데이터가 많을 때 (n=1000, k=800):")
print(f"MLE: {theta_mle2:.3f}")
print(f"MAP: {theta_map2:.3f}")
print("→ 데이터가 많으면 MLE와 MAP가 수렴")
```

### 3.4 사전분포의 역할

- **데이터가 적을 때**: 사전분포의 영향이 큼 (정규화 효과)
- **데이터가 많을 때**: 우도가 지배적, MAP ≈ MLE
- **적절한 사전분포**: 도메인 지식 반영, 과적합 방지
- **무정보 사전분포**: $P(\theta) \propto 1$ → MAP = MLE

## 4. 정규화 = 사전분포

### 4.1 L2 정규화와 가우시안 사전분포

**선형 회귀의 L2 정규화** (Ridge):

$$\min_w \left[\frac{1}{2}\sum_{i=1}^n (y_i - w^T x_i)^2 + \frac{\lambda}{2}\|w\|^2\right]$$

이는 다음 MAP와 동일합니다:

**사전분포**: $w \sim \mathcal{N}(0, \sigma_w^2 I)$

$$\log P(w) = -\frac{1}{2\sigma_w^2}\|w\|^2 + \text{const}$$

**우도** (가우시안 노이즈): $y|x,w \sim \mathcal{N}(w^T x, \sigma^2)$

$$\log P(D|w) = -\frac{1}{2\sigma^2}\sum_{i=1}^n (y_i - w^T x_i)^2 + \text{const}$$

**사후분포 로그**:

$$\log P(w|D) = \log P(D|w) + \log P(w)$$

$$= -\frac{1}{2\sigma^2}\sum_{i=1}^n (y_i - w^T x_i)^2 - \frac{1}{2\sigma_w^2}\|w\|^2 + \text{const}$$

**최대화 = 최소화**:

$$\arg\max_w \log P(w|D) = \arg\min_w \left[\sum_{i=1}^n (y_i - w^T x_i)^2 + \frac{\sigma^2}{\sigma_w^2}\|w\|^2\right]$$

따라서 $\lambda = \frac{\sigma^2}{\sigma_w^2}$

**결론**: L2 정규화 = 가우시안 사전분포를 가진 MAP

### 4.2 L1 정규화와 라플라스 사전분포

**Lasso 회귀**:

$$\min_w \left[\frac{1}{2}\sum_{i=1}^n (y_i - w^T x_i)^2 + \lambda\|w\|_1\right]$$

**라플라스 사전분포**: $P(w_j) = \frac{1}{2b}\exp\left(-\frac{|w_j|}{b}\right)$

$$\log P(w) = -\frac{1}{b}\|w\|_1 + \text{const}$$

따라서 $\lambda = \frac{\sigma^2}{b}$

**결론**: L1 정규화 = 라플라스 사전분포를 가진 MAP

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 사전분포 시각화
w_range = np.linspace(-3, 3, 1000)

# 가우시안 사전분포 (L2)
sigma_w = 1.0
gaussian_prior = stats.norm.pdf(w_range, 0, sigma_w)
log_gaussian = stats.norm.logpdf(w_range, 0, sigma_w)

# 라플라스 사전분포 (L1)
b = 1.0
laplace_prior = stats.laplace.pdf(w_range, 0, b)
log_laplace = stats.laplace.logpdf(w_range, 0, b)

plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.plot(w_range, gaussian_prior, label='Gaussian (L2)', linewidth=2)
plt.plot(w_range, laplace_prior, label='Laplace (L1)', linewidth=2)
plt.xlabel('w')
plt.ylabel('Density')
plt.title('Prior Distributions')
plt.legend()
plt.grid(True)

plt.subplot(132)
plt.plot(w_range, log_gaussian, label='log Gaussian', linewidth=2)
plt.plot(w_range, log_laplace, label='log Laplace', linewidth=2)
plt.xlabel('w')
plt.ylabel('Log Density')
plt.title('Log Prior (Regularization Term)')
plt.legend()
plt.grid(True)

plt.subplot(133)
plt.plot(w_range, -w_range**2 / (2*sigma_w**2),
         label=r'$-\frac{w^2}{2\sigma_w^2}$ (L2)', linewidth=2)
plt.plot(w_range, -np.abs(w_range) / b,
         label=r'$-\frac{|w|}{b}$ (L1)', linewidth=2)
plt.xlabel('w')
plt.ylabel('Regularization Penalty')
plt.title('Penalty Terms')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('regularization_prior.png', dpi=150, bbox_inches='tight')
plt.show()

print("L2 (Gaussian)는 큰 가중치에 quadratic penalty")
print("L1 (Laplace)는 모든 가중치에 linear penalty → 희소성")
```

## 5. EM 알고리즘 (Expectation-Maximization)

잠재(관측되지 않은) 변수가 있는 모델에서는 MLE와 MAP 모두 닫힌 형태의 해를 갖지 않습니다. 기댓값-최대화(Expectation-Maximization, EM) 알고리즘은 이러한 경우를 처리하기 위한 반복적 접근법을 제공합니다.

### 5.1 잠재 변수 문제

**관측 변수** $X$와 **잠재 변수** $Z$가 있을 때:

$$P(X|\theta) = \sum_Z P(X, Z|\theta)$$

직접 최대화하기 어려움 (sum 안의 log).

**EM 알고리즘**은 반복적으로 하한을 최대화합니다.

### 5.2 ELBO와 EM 유도

Jensen 부등식을 사용하여:

$$\log P(X|\theta) = \log \sum_Z P(X, Z|\theta)$$

$$= \log \sum_Z Q(Z) \frac{P(X, Z|\theta)}{Q(Z)}$$

$$\geq \sum_Z Q(Z) \log \frac{P(X, Z|\theta)}{Q(Z)}$$

$$= \mathbb{E}_{Q(Z)} [\log P(X, Z|\theta)] + H(Q)$$

이를 **ELBO** (Evidence Lower BOund)라고 합니다.

**E-step**: $Q(Z)$를 현재 $\theta^{(t)}$에 대해 최적화
$$Q^{(t+1)}(Z) = P(Z|X, \theta^{(t)})$$

**M-step**: $\theta$를 현재 $Q^{(t+1)}$에 대해 최적화
$$\theta^{(t+1)} = \arg\max_{\theta} \mathbb{E}_{Q^{(t+1)}(Z)} [\log P(X, Z|\theta)]$$

### 5.3 가우시안 혼합 모델 (GMM)

**모델**:
- $K$개의 가우시안 컴포넌트
- 잠재 변수 $z_i \in \{1, \ldots, K\}$: 데이터 $x_i$의 컴포넌트
- 파라미터: $\pi_k$ (혼합 비율), $\mu_k, \Sigma_k$ (각 가우시안의 평균, 공분산)

**생성 과정**:
1. $z_i \sim \text{Categorical}(\pi)$
2. $x_i | z_i=k \sim \mathcal{N}(\mu_k, \Sigma_k)$

**E-step**: 책임도(responsibility) 계산

$$\gamma_{ik} = P(z_i=k | x_i, \theta^{(t)}) = \frac{\pi_k \mathcal{N}(x_i|\mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x_i|\mu_j, \Sigma_j)}$$

**M-step**: 파라미터 업데이트

$$\pi_k^{\text{new}} = \frac{1}{n}\sum_{i=1}^n \gamma_{ik}$$

$$\mu_k^{\text{new}} = \frac{\sum_{i=1}^n \gamma_{ik} x_i}{\sum_{i=1}^n \gamma_{ik}}$$

$$\Sigma_k^{\text{new}} = \frac{\sum_{i=1}^n \gamma_{ik} (x_i - \mu_k^{\text{new}})(x_i - \mu_k^{\text{new}})^T}{\sum_{i=1}^n \gamma_{ik}}$$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class GMM:
    def __init__(self, K, max_iter=100, tol=1e-4):
        self.K = K
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        n, d = X.shape

        # 초기화: k-means++
        self.pi = np.ones(self.K) / self.K
        # 랜덤 초기화
        idx = np.random.choice(n, self.K, replace=False)
        self.mu = X[idx]
        self.Sigma = np.array([np.eye(d) for _ in range(self.K)])

        log_likelihood_old = -np.inf

        for iteration in range(self.max_iter):
            # E-step
            gamma = self._e_step(X)

            # M-step
            self._m_step(X, gamma)

            # 로그 우도 계산
            log_likelihood = self._compute_log_likelihood(X)

            if iteration % 10 == 0:
                print(f"Iteration {iteration}: log-likelihood = {log_likelihood:.2f}")

            # 수렴 확인
            if abs(log_likelihood - log_likelihood_old) < self.tol:
                print(f"Converged at iteration {iteration}")
                break

            log_likelihood_old = log_likelihood

        return self

    def _e_step(self, X):
        n = X.shape[0]
        gamma = np.zeros((n, self.K))

        for k in range(self.K):
            gamma[:, k] = self.pi[k] * stats.multivariate_normal.pdf(
                X, self.mu[k], self.Sigma[k])

        # 정규화
        gamma /= gamma.sum(axis=1, keepdims=True)
        return gamma

    def _m_step(self, X, gamma):
        n = X.shape[0]

        # Effective number of points assigned to each cluster
        Nk = gamma.sum(axis=0)

        # 혼합 비율 업데이트
        self.pi = Nk / n

        # 평균 업데이트
        self.mu = (gamma.T @ X) / Nk[:, np.newaxis]

        # 공분산 업데이트
        for k in range(self.K):
            diff = X - self.mu[k]
            self.Sigma[k] = (gamma[:, k:k+1] * diff).T @ diff / Nk[k]
            # 수치 안정성
            self.Sigma[k] += 1e-6 * np.eye(X.shape[1])

    def _compute_log_likelihood(self, X):
        n = X.shape[0]
        log_likelihood = 0

        for i in range(n):
            prob = 0
            for k in range(self.K):
                prob += self.pi[k] * stats.multivariate_normal.pdf(
                    X[i], self.mu[k], self.Sigma[k])
            log_likelihood += np.log(prob + 1e-10)

        return log_likelihood

    def predict(self, X):
        gamma = self._e_step(X)
        return np.argmax(gamma, axis=1)

# 테스트: 3개의 가우시안 혼합
np.random.seed(42)
n_samples = 300

# 3개의 클러스터 생성
X1 = np.random.randn(n_samples, 2) + np.array([0, 0])
X2 = np.random.randn(n_samples, 2) + np.array([5, 5])
X3 = np.random.randn(n_samples, 2) + np.array([0, 5])
X = np.vstack([X1, X2, X3])

# GMM 학습
gmm = GMM(K=3, max_iter=100)
gmm.fit(X)

# 예측
labels = gmm.predict(X)

# 시각화
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
plt.title('Original Data')
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(True)

plt.subplot(132)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.5)
for k in range(gmm.K):
    plt.scatter(gmm.mu[k, 0], gmm.mu[k, 1],
                marker='x', s=200, linewidths=3, color='red')
plt.title('GMM Clustering')
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(True)

# 결정 경계
plt.subplot(133)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
Z = gmm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis',
            alpha=0.7, edgecolors='k')
plt.title('Decision Boundaries')
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(True)

plt.tight_layout()
plt.savefig('gmm_em.png', dpi=150, bbox_inches='tight')
plt.show()
```

## 6. 머신러닝 응용

### 6.1 로지스틱 회귀 = 베르누이 MLE

**모델**: $P(y=1|x,w) = \sigma(w^T x)$, 여기서 $\sigma(z) = \frac{1}{1+e^{-z}}$

**로그 우도**:

$$\ell(w) = \sum_{i=1}^n \left[y_i \log \sigma(w^T x_i) + (1-y_i) \log(1-\sigma(w^T x_i))\right]$$

이는 베르누이 분포의 로그 우도입니다!

**최적화**: 경사 하강법

$$\nabla_w \ell = \sum_{i=1}^n (y_i - \sigma(w^T x_i)) x_i$$

### 6.2 신경망 학습 = MLE

**분류 문제**의 신경망 출력을 softmax로 해석:

$$P(y=k|x,\theta) = \frac{\exp(f_k(x;\theta))}{\sum_j \exp(f_j(x;\theta))}$$

**교차 엔트로피 손실** = **음의 로그 우도**:

$$\mathcal{L}(\theta) = -\sum_{i=1}^n \log P(y_i|x_i, \theta)$$

따라서 신경망 학습 = MLE!

### 6.3 베이지안 신경망 소개

**문제**: 신경망은 점 추정만 제공, 불확실성을 모름

**베이지안 신경망**: 가중치에 대한 사후 분포 추론

$$P(\theta|D) = \frac{P(D|\theta)P(\theta)}{P(D)}$$

**예측**: 사후 분포에 대한 적분

$$P(y|x, D) = \int P(y|x, \theta) P(\theta|D) d\theta$$

**도전 과제**: 고차원 적분 계산 → 변분 추론, MCMC 필요

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 간단한 로지스틱 회귀 예제
np.random.seed(42)
torch.manual_seed(42)

# 데이터 생성
n = 200
X = np.random.randn(n, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)

X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y)

# 로지스틱 회귀 모델
model = nn.Linear(2, 1)
criterion = nn.BCEWithLogitsLoss()  # 이진 교차 엔트로피 = 베르누이 MLE
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 학습
losses = []
for epoch in range(1000):
    optimizer.zero_grad()
    output = model(X_tensor).squeeze()
    loss = criterion(output, y_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

# 시각화
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss (Negative Log-Likelihood)')
plt.title('Training Loss = -Log-Likelihood')
plt.grid(True)

plt.subplot(132)
plt.scatter(X[y==0, 0], X[y==0, 1], label='Class 0', alpha=0.6)
plt.scatter(X[y==1, 0], X[y==1, 1], label='Class 1', alpha=0.6)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Data')
plt.legend()
plt.grid(True)

# 결정 경계
plt.subplot(133)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
with torch.no_grad():
    Z = torch.sigmoid(model(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]))).numpy()
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, levels=20)
plt.scatter(X[y==0, 0], X[y==0, 1], label='Class 0', alpha=0.7, edgecolors='k')
plt.scatter(X[y==1, 0], X[y==1, 1], label='Class 1', alpha=0.7, edgecolors='k')
plt.colorbar(label='P(y=1|x)')
plt.title('MLE Decision Boundary')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('logistic_mle.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Learned weights: {model.weight.data.numpy()}")
print(f"Learned bias: {model.bias.data.numpy()}")
```

## 연습 문제

### 문제 1: 지수분포의 MLE

지수분포 $\text{Exp}(\lambda)$의 PDF는 $p(x|\lambda) = \lambda e^{-\lambda x}$ (for $x \geq 0$)입니다.

$n$개의 i.i.d. 샘플 $\{x_1, \ldots, x_n\}$이 주어졌을 때:

(a) 로그 우도 함수를 유도하시오.
(b) $\lambda$의 MLE를 구하시오.
(c) Python으로 검증하시오 (시뮬레이션 데이터 생성 후 MLE 계산).

### 문제 2: MAP with Different Priors

선형 회귀 문제에서:
- 데이터: $(x_1, y_1), \ldots, (x_n, y_n)$
- 모델: $y = wx + b + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma^2)$

다음 두 경우의 MAP를 비교하시오:

(a) 가우시안 사전분포: $w \sim \mathcal{N}(0, \sigma_w^2)$ → L2 정규화
(b) 라플라스 사전분포: $p(w) \propto \exp(-|w|/b)$ → L1 정규화

각각의 최적화 문제를 작성하고, Python으로 구현하여 차이를 시각화하시오.

### 문제 3: EM for Coin Toss

두 개의 동전 A와 B가 있고, 앞면 확률이 각각 $\theta_A, \theta_B$입니다.

실험:
- 5번의 시행에서, 각 시행마다 동전을 랜덤하게 선택 (관측 불가)
- 결과: {H, T, T, H, H}

EM 알고리즘을 사용하여 $\theta_A, \theta_B$를 추정하시오:

(a) E-step: 각 시행에서 동전 A일 확률 계산
(b) M-step: $\theta_A, \theta_B$ 업데이트
(c) Python으로 구현하고 수렴 과정을 시각화하시오

### 문제 4: Fisher Information

Fisher 정보는 다음과 같이 정의됩니다:

$$I(\theta) = -\mathbb{E}\left[\frac{\partial^2 \log p(X|\theta)}{\partial \theta^2}\right]$$

베르누이 분포 $p(x|\theta) = \theta^x (1-\theta)^{1-x}$에 대해:

(a) Fisher 정보 $I(\theta)$를 계산하시오.
(b) Cramér-Rao 하한 $\text{Var}(\hat{\theta}) \geq \frac{1}{nI(\theta)}$을 확인하시오.
(c) MLE $\hat{\theta} = \frac{k}{n}$의 분산이 이 하한에 도달함을 보이시오.

### 문제 5: EM for Mixture of Exponentials

지수분포의 혼합 모델:

$$p(x) = \pi \lambda_1 e^{-\lambda_1 x} + (1-\pi) \lambda_2 e^{-\lambda_2 x}$$

EM 알고리즘을 유도하고 Python으로 구현하시오:

(a) E-step의 책임도 공식 유도
(b) M-step의 $\pi, \lambda_1, \lambda_2$ 업데이트 공식 유도
(c) 시뮬레이션 데이터로 검증 (진짜 파라미터를 알고 있는 상태에서 복원)

## 참고 자료

1. **Bishop, C. M. (2006).** *Pattern Recognition and Machine Learning*. Chapter 9 (EM Algorithm).
2. **Murphy, K. P. (2022).** *Probabilistic Machine Learning: An Introduction*. Chapter 8 (MLE), Chapter 10 (MAP).
3. **Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020).** *Mathematics for Machine Learning*. Chapter 8.
4. **MacKay, D. J. C. (2003).** *Information Theory, Inference, and Learning Algorithms*. Chapter 22 (EM).
5. **논문**: Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). "Maximum Likelihood from Incomplete Data via the EM Algorithm". *Journal of the Royal Statistical Society*.
6. **scikit-learn 문서**: Gaussian Mixture Models - https://scikit-learn.org/stable/modules/mixture.html
7. **PyTorch 문서**: Loss Functions - https://pytorch.org/docs/stable/nn.html#loss-functions
