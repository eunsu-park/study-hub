# 11. 고급 확률 분포 (Advanced Probability Distributions)

## 학습 목표

- 지수족의 일반 형태와 성질을 이해하고 다양한 분포를 지수족으로 표현할 수 있다
- 충분 통계량과 자연 매개변수의 개념을 파악하고 GLM과의 연결고리를 이해한다
- 켤레 사전분포의 원리를 이해하고 베이지안 추론에서의 활용을 학습한다
- 다변량 가우시안 분포의 성질과 조건부/주변화 연산을 수행할 수 있다
- 가우시안 혼합 모델(GMM)의 구조를 이해하고 EM 알고리즘을 적용할 수 있다
- 머신러닝에서 고급 확률 분포가 어떻게 활용되는지 실제 예제로 학습한다

---

## 1. 지수족 (Exponential Family)

### 1.1 일반 형태

확률 분포가 **지수족(exponential family)**에 속한다는 것은 다음 형태로 표현 가능함을 의미합니다:

$$p(x|\eta) = h(x) \exp\left(\eta^T T(x) - A(\eta)\right)$$

여기서:
- $\eta$: **자연 매개변수(natural parameter)** 또는 정준 매개변수
- $T(x)$: **충분 통계량(sufficient statistic)**
- $A(\eta)$: **로그 분배 함수(log partition function)** 또는 누적 생성 함수
- $h(x)$: 기저 측도(base measure)

**정규화 조건**:

$$\int h(x) \exp\left(\eta^T T(x) - A(\eta)\right) dx = 1$$

따라서:

$$A(\eta) = \log \int h(x) \exp\left(\eta^T T(x)\right) dx$$

### 1.2 왜 지수족인가?

지수족은 다음과 같은 훌륭한 성질을 가집니다:

1. **충분 통계량**: 데이터를 $T(x)$로 요약 가능
2. **해석적 MLE**: 닫힌 형태의 해
3. **켤레 사전분포 존재**: 베이지안 추론이 쉬움
4. **모멘트 계산**: $A(\eta)$의 미분으로 얻음
5. **GLM의 기초**: 일반화 선형 모델

**로그 분배 함수의 성질**:

$$\frac{\partial A}{\partial \eta} = \mathbb{E}[T(X)]$$

$$\frac{\partial^2 A}{\partial \eta^2} = \text{Var}(T(X))$$

### 1.3 충분 통계량

**정의**: $T(x)$는 $\eta$에 대한 **충분 통계량**입니다.

**의미**: 파라미터 추정에 필요한 모든 정보를 $T(x)$가 담고 있음.

$$P(\theta | x) = P(\theta | T(x))$$

**예**: $n$개의 정규분포 샘플에서, $\bar{x} = \frac{1}{n}\sum x_i$와 $s^2 = \frac{1}{n}\sum (x_i - \bar{x})^2$가 충분 통계량.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 지수족 분포들의 로그 분배 함수 시각화

# 베르누이: η = log(p/(1-p)), A(η) = log(1 + e^η)
eta_bernoulli = np.linspace(-5, 5, 100)
A_bernoulli = np.log(1 + np.exp(eta_bernoulli))
p_bernoulli = 1 / (1 + np.exp(-eta_bernoulli))  # p = sigmoid(η)

# 가우시안 (평균만): η = μ, A(η) = η²/2 (분산=1 고정)
eta_gaussian = np.linspace(-3, 3, 100)
A_gaussian = eta_gaussian**2 / 2

plt.figure(figsize=(14, 4))

# 베르누이
plt.subplot(131)
plt.plot(eta_bernoulli, A_bernoulli, linewidth=2)
plt.xlabel('η (자연 매개변수)')
plt.ylabel('A(η)')
plt.title('Bernoulli: A(η) = log(1 + exp(η))')
plt.grid(True)

plt.subplot(132)
plt.plot(eta_bernoulli, p_bernoulli, linewidth=2, color='green')
plt.xlabel('η')
plt.ylabel('p = dA/dη')
plt.title('Bernoulli: E[X] = sigmoid(η)')
plt.grid(True)

# 가우시안
plt.subplot(133)
plt.plot(eta_gaussian, A_gaussian, linewidth=2, color='red')
plt.xlabel('η (= μ)')
plt.ylabel('A(η)')
plt.title('Gaussian (σ=1): A(η) = η²/2')
plt.grid(True)

plt.tight_layout()
plt.savefig('exponential_family_log_partition.png', dpi=150, bbox_inches='tight')
plt.show()

print("A(η)의 미분 = 기대값 E[T(X)]")
print("A(η)의 2차 미분 = 분산 Var(T(X))")
```

## 2. 지수족의 예

### 2.1 베르누이 분포 → 지수족

**일반 형태**:
$$p(x|p) = p^x (1-p)^{1-x}$$

**지수족 변환**:

$$= \exp\left[x \log p + (1-x) \log(1-p)\right]$$

$$= \exp\left[x \log \frac{p}{1-p} + \log(1-p)\right]$$

**식별**:
- $\eta = \log \frac{p}{1-p}$ (로짓)
- $T(x) = x$
- $A(\eta) = -\log(1-p) = \log(1 + e^\eta)$
- $h(x) = 1$

**역변환**: $p = \sigma(\eta) = \frac{1}{1+e^{-\eta}}$ (시그모이드!)

### 2.2 가우시안 분포 → 지수족

**일반 형태** ($\sigma^2$ 알려진 경우):

$$p(x|\mu) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

$$= \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{x^2}{2\sigma^2}\right) \exp\left(\frac{\mu x}{\sigma^2} - \frac{\mu^2}{2\sigma^2}\right)$$

**식별**:
- $\eta = \frac{\mu}{\sigma^2}$
- $T(x) = x$
- $A(\eta) = \frac{\mu^2}{2\sigma^2} = \frac{\sigma^2 \eta^2}{2}$
- $h(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{x^2}{2\sigma^2}\right)$

**$\mu$와 $\sigma^2$ 모두 미지인 경우**:

$$\eta = \begin{bmatrix} \mu/\sigma^2 \\ -1/(2\sigma^2) \end{bmatrix}, \quad T(x) = \begin{bmatrix} x \\ x^2 \end{bmatrix}$$

### 2.3 포아송 분포

$$p(x|\lambda) = \frac{\lambda^x e^{-\lambda}}{x!}$$

$$= \frac{1}{x!} \exp(x \log \lambda - \lambda)$$

- $\eta = \log \lambda$
- $T(x) = x$
- $A(\eta) = \lambda = e^\eta$
- $h(x) = 1/x!$

### 2.4 감마 분포

$$p(x|\alpha, \beta) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}$$

($\alpha$ 고정, $\beta$ 변수):
- $\eta = -\beta$
- $T(x) = x$
- $A(\eta) = -\alpha \log(-\eta)$

### 2.5 베타 분포

$$p(x|\alpha, \beta) = \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} x^{\alpha-1} (1-x)^{\beta-1}$$

- $\eta = \begin{bmatrix} \alpha-1 \\ \beta-1 \end{bmatrix}$
- $T(x) = \begin{bmatrix} \log x \\ \log(1-x) \end{bmatrix}$

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# 다양한 지수족 분포
x_range = np.linspace(0, 10, 1000)

# 포아송 분포들 (λ 변화)
lambdas = [1, 2, 4]
x_poisson = np.arange(0, 15)

plt.figure(figsize=(14, 9))

# 포아송
plt.subplot(331)
for lam in lambdas:
    pmf = stats.poisson.pmf(x_poisson, lam)
    plt.plot(x_poisson, pmf, 'o-', label=f'λ={lam}')
plt.xlabel('x')
plt.ylabel('P(X=x)')
plt.title('Poisson Distribution')
plt.legend()
plt.grid(True)

# 지수 분포 (감마의 특수 케이스, α=1)
plt.subplot(332)
for lam in lambdas:
    pdf = stats.expon.pdf(x_range, scale=1/lam)
    plt.plot(x_range, pdf, label=f'λ={lam}')
plt.xlabel('x')
plt.ylabel('p(x)')
plt.title('Exponential Distribution')
plt.legend()
plt.grid(True)

# 감마 분포
plt.subplot(333)
alphas = [1, 2, 5]
beta = 1
for alpha in alphas:
    pdf = stats.gamma.pdf(x_range, alpha, scale=1/beta)
    plt.plot(x_range, pdf, label=f'α={alpha}, β={beta}')
plt.xlabel('x')
plt.ylabel('p(x)')
plt.title('Gamma Distribution')
plt.legend()
plt.grid(True)

# 베타 분포
x_beta = np.linspace(0, 1, 1000)
plt.subplot(334)
params = [(0.5, 0.5), (2, 2), (5, 2)]
for alpha, beta in params:
    pdf = stats.beta.pdf(x_beta, alpha, beta)
    plt.plot(x_beta, pdf, label=f'α={alpha}, β={beta}')
plt.xlabel('x')
plt.ylabel('p(x)')
plt.title('Beta Distribution')
plt.legend()
plt.grid(True)

# 가우시안 (다양한 μ)
plt.subplot(335)
mus = [-1, 0, 1]
sigma = 1
x_gauss = np.linspace(-4, 4, 1000)
for mu in mus:
    pdf = stats.norm.pdf(x_gauss, mu, sigma)
    plt.plot(x_gauss, pdf, label=f'μ={mu}, σ={sigma}')
plt.xlabel('x')
plt.ylabel('p(x)')
plt.title('Gaussian Distribution')
plt.legend()
plt.grid(True)

# 베르누이 (다양한 p)
plt.subplot(336)
ps = [0.2, 0.5, 0.8]
x_bern = [0, 1]
for p in ps:
    pmf = [1-p, p]
    plt.plot(x_bern, pmf, 'o-', markersize=10, label=f'p={p}')
plt.xlabel('x')
plt.ylabel('P(X=x)')
plt.title('Bernoulli Distribution')
plt.xticks([0, 1])
plt.legend()
plt.grid(True)

# 카테고리컬 (다항)
plt.subplot(337)
categories = ['A', 'B', 'C', 'D']
probs1 = [0.25, 0.25, 0.25, 0.25]
probs2 = [0.5, 0.3, 0.15, 0.05]
x_pos = np.arange(len(categories))
width = 0.35
plt.bar(x_pos - width/2, probs1, width, label='Uniform', alpha=0.7)
plt.bar(x_pos + width/2, probs2, width, label='Skewed', alpha=0.7)
plt.xlabel('Category')
plt.ylabel('Probability')
plt.title('Categorical Distribution')
plt.xticks(x_pos, categories)
plt.legend()
plt.grid(True, axis='y')

# 다항 정규분포 (2D 등고선)
plt.subplot(338)
x1 = np.linspace(-3, 3, 100)
x2 = np.linspace(-3, 3, 100)
X1, X2 = np.meshgrid(x1, x2)
pos = np.dstack((X1, X2))
rv = stats.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]])
plt.contourf(X1, X2, rv.pdf(pos), levels=15, cmap='viridis')
plt.colorbar()
plt.xlabel('x₁')
plt.ylabel('x₂')
plt.title('Bivariate Gaussian')
plt.grid(True)

# 디리클레 (단순화: 3D를 2D로 투영)
plt.subplot(339)
from matplotlib.patches import Polygon
# 2-simplex 시각화 (합=1 제약)
alpha_params = [(1, 1, 1), (2, 5, 2), (10, 5, 3)]
for alpha in alpha_params:
    # 간단한 표현: 각 모서리에 가중치
    label = f'α=({alpha[0]},{alpha[1]},{alpha[2]})'
    plt.text(0.5, sum(alpha)/30, label, ha='center')
plt.text(0.5, 0.5, 'Dirichlet\n(simplex)', ha='center', va='center',
         fontsize=14, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title('Dirichlet Distribution (conceptual)')
plt.axis('off')

plt.tight_layout()
plt.savefig('exponential_family_examples.png', dpi=150, bbox_inches='tight')
plt.show()

print("모든 지수족 분포는 동일한 수학적 구조를 공유합니다!")
```

## 3. 켤레 사전분포 (Conjugate Priors)

### 3.1 정의

사전분포 $p(\theta)$와 우도 $p(x|\theta)$에 대해, 사후분포 $p(\theta|x)$가 사전분포와 **같은 분포족**에 속하면, 사전분포를 **켤레 사전분포(conjugate prior)**라고 합니다.

$$p(\theta), p(\theta|x) \in \text{same family}$$

**장점**:
1. 사후분포가 닫힌 형태
2. 순차적 업데이트 가능
3. 베이지안 추론이 해석적

### 3.2 베타-베르누이 켤레

**우도**: 베르누이 $p(x|\theta) = \theta^x (1-\theta)^{1-x}$

**사전분포**: 베타 $p(\theta) = \text{Beta}(\alpha, \beta)$

$$p(\theta) = \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} \theta^{\alpha-1} (1-\theta)^{\beta-1}$$

**사후분포** ($n$번 시행, $k$번 성공):

$$p(\theta|D) \propto p(D|\theta) p(\theta)$$

$$\propto \theta^k (1-\theta)^{n-k} \cdot \theta^{\alpha-1} (1-\theta)^{\beta-1}$$

$$= \theta^{k+\alpha-1} (1-\theta)^{n-k+\beta-1}$$

$$= \text{Beta}(\alpha + k, \beta + n - k)$$

**해석**:
- $\alpha$: 사전 "성공 횟수" (가상 관측)
- $\beta$: 사전 "실패 횟수"
- 사후분포: 사전 가상 관측 + 실제 관측

### 3.3 정규-정규 켤레

**우도**: $x \sim \mathcal{N}(\mu, \sigma^2)$ ($\sigma^2$ 알려짐)

**사전분포**: $\mu \sim \mathcal{N}(\mu_0, \sigma_0^2)$

**사후분포**: $\mu | D \sim \mathcal{N}(\mu_n, \sigma_n^2)$

$$\mu_n = \frac{\sigma^2 \mu_0 + n \sigma_0^2 \bar{x}}{\sigma^2 + n \sigma_0^2}$$

$$\sigma_n^2 = \frac{\sigma^2 \sigma_0^2}{\sigma^2 + n \sigma_0^2}$$

**해석**:
- $\mu_n$: 사전 평균과 표본 평균의 가중 평균
- 데이터가 많아지면 ($n \to \infty$) $\mu_n \to \bar{x}$

### 3.4 디리클레-카테고리컬 켤레

**우도**: 카테고리컬 $p(x|\theta) = \prod_{k=1}^K \theta_k^{x_k}$ (one-hot $x$)

**사전분포**: 디리클레 $p(\theta) = \text{Dir}(\alpha)$

$$p(\theta|\alpha) = \frac{\Gamma(\sum_k \alpha_k)}{\prod_k \Gamma(\alpha_k)} \prod_{k=1}^K \theta_k^{\alpha_k - 1}$$

**사후분포** (각 카테고리 $k$가 $n_k$번 관측):

$$p(\theta|D) = \text{Dir}(\alpha_1 + n_1, \ldots, \alpha_K + n_K)$$

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# 베타-베르누이 켤레 시연
np.random.seed(42)

# 사전분포: Beta(2, 2) (약간 중앙 선호)
alpha_prior, beta_prior = 2, 2

# 진짜 동전: p = 0.7
true_p = 0.7
n_flips = [0, 1, 5, 20, 100]

theta_range = np.linspace(0, 1, 1000)

plt.figure(figsize=(14, 8))

for i, n in enumerate(n_flips):
    # 데이터 생성
    if n > 0:
        data = np.random.binomial(1, true_p, n)
        k = data.sum()  # 성공 횟수
    else:
        k = 0

    # 사후분포
    alpha_post = alpha_prior + k
    beta_post = beta_prior + (n - k)

    # 시각화
    plt.subplot(2, 3, i+1)

    # 사전분포 (첫 번째 플롯에만)
    if i == 0:
        prior = stats.beta.pdf(theta_range, alpha_prior, beta_prior)
        plt.plot(theta_range, prior, 'b--', linewidth=2, label='Prior')

    # 사후분포
    posterior = stats.beta.pdf(theta_range, alpha_post, beta_post)
    plt.plot(theta_range, posterior, 'r-', linewidth=2, label='Posterior')

    # 진짜 값
    plt.axvline(true_p, color='g', linestyle='--', alpha=0.7,
                label=f'True p={true_p}')

    # 사후 평균
    post_mean = alpha_post / (alpha_post + beta_post)
    plt.axvline(post_mean, color='orange', linestyle='--', alpha=0.7,
                label=f'Post mean={post_mean:.3f}')

    plt.xlabel('θ')
    plt.ylabel('Density')
    plt.title(f'n={n}, k={k if n > 0 else 0}\nBeta({alpha_post}, {beta_post})')
    plt.legend(fontsize=8)
    plt.grid(True)

# 6번째 플롯: 순차적 업데이트
plt.subplot(2, 3, 6)
alphas = [alpha_prior]
betas = [beta_prior]

n_total = 50
for flip in range(n_total):
    result = np.random.binomial(1, true_p)
    alphas.append(alphas[-1] + result)
    betas.append(betas[-1] + (1 - result))

# 평균의 변화
means = [a / (a + b) for a, b in zip(alphas, betas)]
plt.plot(means, linewidth=2, label='Posterior mean')
plt.axhline(true_p, color='g', linestyle='--', label=f'True p={true_p}')
plt.xlabel('Number of flips')
plt.ylabel('Posterior mean of θ')
plt.title('Sequential Bayesian Update')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('conjugate_beta_bernoulli.png', dpi=150, bbox_inches='tight')
plt.show()

print("켤레 사전분포의 장점:")
print("1. 사후분포가 닫힌 형태 (Beta 유지)")
print("2. 순차적 업데이트 가능")
print("3. 데이터가 많아지면 진짜 값으로 수렴")
```

### 3.5 주요 켤레 쌍

| 우도 | 사전분포 | 사후분포 |
|------|---------|---------|
| Bernoulli | Beta | Beta |
| Binomial | Beta | Beta |
| Categorical | Dirichlet | Dirichlet |
| Multinomial | Dirichlet | Dirichlet |
| Gaussian (μ, σ² 알려짐) | Gaussian | Gaussian |
| Gaussian (σ² 알려짐) | Gaussian (on μ) | Gaussian |
| Gaussian (μ 알려짐) | Inverse-Gamma (on σ²) | Inverse-Gamma |
| Poisson | Gamma | Gamma |
| Exponential | Gamma | Gamma |

## 4. 다변량 가우시안 분포 (Multivariate Gaussian)

지수족 분포 중에서도 다변량 가우시안(multivariate Gaussian)은 머신러닝에서 특히 핵심적인 위치를 차지합니다. 그 기하학적·대수적 성질 덕분에 선형 회귀부터 VAE에 이르는 다양한 모델에서 없어서는 안 될 존재입니다.

### 4.1 정의

$d$차원 **다변량 가우시안 분포** $\mathcal{N}(\mu, \Sigma)$:

$$p(x) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x-\mu)^T \Sigma^{-1} (x-\mu)\right)$$

- $\mu \in \mathbb{R}^d$: 평균 벡터
- $\Sigma \in \mathbb{R}^{d \times d}$: 공분산 행렬 (양정부호)
- $|\Sigma|$: 행렬식
- $\Sigma^{-1}$: 정밀도 행렬 (precision matrix)

### 4.2 공분산 행렬의 역할

공분산 행렬 $\Sigma$:

$$\Sigma_{ij} = \mathbb{E}[(X_i - \mu_i)(X_j - \mu_j)]$$

**성질**:
1. 대칭: $\Sigma = \Sigma^T$
2. 양정부호: $x^T \Sigma x > 0$ for all $x \neq 0$
3. 고유값 분해: $\Sigma = Q \Lambda Q^T$

**등고선**: $(x-\mu)^T \Sigma^{-1} (x-\mu) = c$는 타원

### 4.3 주변화 (Marginalization)

$x = \begin{bmatrix} x_a \\ x_b \end{bmatrix}$로 분할하고,

$$\mu = \begin{bmatrix} \mu_a \\ \mu_b \end{bmatrix}, \quad \Sigma = \begin{bmatrix} \Sigma_{aa} & \Sigma_{ab} \\ \Sigma_{ba} & \Sigma_{bb} \end{bmatrix}$$

**주변 분포**:

$$p(x_a) = \int p(x_a, x_b) dx_b = \mathcal{N}(x_a | \mu_a, \Sigma_{aa})$$

→ 가우시안의 주변분포도 가우시안!

### 4.4 조건부 분포 (Conditional Distribution)

**조건부 분포**:

$$p(x_a | x_b) = \mathcal{N}(x_a | \mu_{a|b}, \Sigma_{a|b})$$

$$\mu_{a|b} = \mu_a + \Sigma_{ab} \Sigma_{bb}^{-1} (x_b - \mu_b)$$

$$\Sigma_{a|b} = \Sigma_{aa} - \Sigma_{ab} \Sigma_{bb}^{-1} \Sigma_{ba}$$

→ 가우시안의 조건부도 가우시안!

**중요 성질**: $x_a$와 $x_b$가 무상관 ($\Sigma_{ab} = 0$) $\Leftrightarrow$ 독립

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 2D 가우시안 시각화
mu = np.array([0, 0])
Sigma = np.array([[1, 0.8],
                  [0.8, 1]])

# 샘플링
np.random.seed(42)
samples = np.random.multivariate_normal(mu, Sigma, 500)

# 그리드
x1 = np.linspace(-3, 3, 100)
x2 = np.linspace(-3, 3, 100)
X1, X2 = np.meshgrid(x1, x2)
pos = np.dstack((X1, X2))

# PDF
rv = stats.multivariate_normal(mu, Sigma)
pdf = rv.pdf(pos)

plt.figure(figsize=(14, 4))

# 등고선과 샘플
plt.subplot(131)
plt.contour(X1, X2, pdf, levels=10, cmap='viridis')
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=10)
plt.xlabel('x₁')
plt.ylabel('x₂')
plt.title('Bivariate Gaussian\n(ρ=0.8)')
plt.axis('equal')
plt.grid(True)

# 주변분포
plt.subplot(132)
# x1의 주변분포
mu_x1 = mu[0]
sigma_x1 = np.sqrt(Sigma[0, 0])
x1_range = np.linspace(-3, 3, 100)
pdf_x1 = stats.norm.pdf(x1_range, mu_x1, sigma_x1)

plt.plot(x1_range, pdf_x1, linewidth=2, label='Marginal p(x₁)')
plt.hist(samples[:, 0], bins=30, density=True, alpha=0.5,
         label='Samples')
plt.xlabel('x₁')
plt.ylabel('Density')
plt.title('Marginal Distribution')
plt.legend()
plt.grid(True)

# 조건부 분포
plt.subplot(133)
# x2 = 1로 조건부
x2_cond = 1.0

# 조건부 평균과 분산
mu_cond = mu[0] + Sigma[0, 1] / Sigma[1, 1] * (x2_cond - mu[1])
sigma_cond = np.sqrt(Sigma[0, 0] - Sigma[0, 1]**2 / Sigma[1, 1])

pdf_cond = stats.norm.pdf(x1_range, mu_cond, sigma_cond)

plt.plot(x1_range, pdf_cond, linewidth=2,
         label=f'p(x₁|x₂={x2_cond})')

# 조건부 샘플 추출
cond_samples = samples[np.abs(samples[:, 1] - x2_cond) < 0.1, 0]
plt.hist(cond_samples, bins=20, density=True, alpha=0.5,
         label='Conditional samples')

plt.xlabel('x₁')
plt.ylabel('Density')
plt.title(f'Conditional Distribution\np(x₁|x₂={x2_cond})')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('multivariate_gaussian.png', dpi=150, bbox_inches='tight')
plt.show()

# 수식 검증
print("조건부 분포 파라미터:")
print(f"  μ_(x₁|x₂={x2_cond}) = {mu_cond:.3f}")
print(f"  σ_(x₁|x₂={x2_cond}) = {sigma_cond:.3f}")

print("\n공분산 행렬:")
print(Sigma)
print(f"\n상관계수: ρ = {Sigma[0,1] / np.sqrt(Sigma[0,0] * Sigma[1,1]):.3f}")
```

### 4.5 공분산 행렬의 기하학

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def plot_gaussian_ellipse(mu, Sigma, ax, color='blue', label=''):
    """가우시안의 등고선 타원 그리기"""
    # 고유값 분해
    eigenvalues, eigenvectors = np.linalg.eig(Sigma)

    # 타원의 각도
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    angle = np.degrees(angle)

    # 2-sigma 타원
    width, height = 2 * 2 * np.sqrt(eigenvalues)

    ellipse = Ellipse(mu, width, height, angle=angle,
                      facecolor='none', edgecolor=color,
                      linewidth=2, label=label)
    ax.add_patch(ellipse)

    # 주축 그리기
    for i in range(2):
        v = eigenvectors[:, i] * 2 * np.sqrt(eigenvalues[i])
        ax.arrow(mu[0], mu[1], v[0], v[1],
                head_width=0.1, head_length=0.1,
                fc=color, ec=color, alpha=0.5)

mu = np.array([0, 0])

# 다양한 공분산 행렬
Sigmas = [
    (np.array([[1, 0], [0, 1]]), 'Isotropic'),
    (np.array([[2, 0], [0, 0.5]]), 'Diagonal'),
    (np.array([[1, 0.8], [0.8, 1]]), 'Correlated (+)'),
    (np.array([[1, -0.8], [-0.8, 1]]), 'Correlated (-)'),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()

for ax, (Sigma, title) in zip(axes, Sigmas):
    # 등고선 타원
    plot_gaussian_ellipse(mu, Sigma, ax, color='blue', label='2σ ellipse')

    # 샘플
    samples = np.random.multivariate_normal(mu, Sigma, 200)
    ax.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=10)

    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title(title)
    ax.axis('equal')
    ax.grid(True)
    ax.legend()

    # 공분산 행렬 표시
    ax.text(0.05, 0.95, f'Σ =\n{Sigma}',
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('covariance_geometry.png', dpi=150, bbox_inches='tight')
plt.show()
```

## 5. 가우시안 혼합 모델 (GMM)

### 5.1 모델 정의

**가우시안 혼합 모델(Gaussian Mixture Model, GMM)**:

$$p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)$$

- $K$: 컴포넌트 개수
- $\pi_k$: 혼합 비율 ($\sum_k \pi_k = 1, \pi_k \geq 0$)
- $\mu_k, \Sigma_k$: $k$번째 가우시안의 평균, 공분산

**잠재 변수 해석**:
- $z_i \in \{1, \ldots, K\}$: 데이터 $x_i$가 어느 컴포넌트에서 왔는지
- $p(z_i = k) = \pi_k$
- $p(x_i | z_i = k) = \mathcal{N}(x_i | \mu_k, \Sigma_k)$

### 5.2 EM 알고리즘 (09강과 연결)

**E-step**: 책임도(responsibility) 계산

$$\gamma_{ik} = p(z_i = k | x_i) = \frac{\pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x_i | \mu_j, \Sigma_j)}$$

**M-step**: 파라미터 업데이트

$$N_k = \sum_{i=1}^n \gamma_{ik}$$

$$\pi_k = \frac{N_k}{n}$$

$$\mu_k = \frac{1}{N_k} \sum_{i=1}^n \gamma_{ik} x_i$$

$$\Sigma_k = \frac{1}{N_k} \sum_{i=1}^n \gamma_{ik} (x_i - \mu_k)(x_i - \mu_k)^T$$

### 5.3 구현

09강의 GMM 코드를 다변량으로 확장:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class GaussianMixture:
    def __init__(self, n_components=3, max_iter=100, tol=1e-4):
        self.K = n_components
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        n, d = X.shape

        # 초기화
        self.pi = np.ones(self.K) / self.K
        # k-means++ 스타일 초기화
        idx = np.random.choice(n, self.K, replace=False)
        self.mu = X[idx]
        self.Sigma = np.array([np.eye(d) for _ in range(self.K)])

        log_likelihood_old = -np.inf

        for iteration in range(self.max_iter):
            # E-step
            gamma = self._e_step(X)

            # M-step
            self._m_step(X, gamma)

            # Log-likelihood
            log_likelihood = self._compute_log_likelihood(X)

            if iteration % 10 == 0:
                print(f"Iter {iteration}: log-likelihood = {log_likelihood:.2f}")

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

        gamma /= gamma.sum(axis=1, keepdims=True)
        return gamma

    def _m_step(self, X, gamma):
        n = X.shape[0]
        Nk = gamma.sum(axis=0)

        self.pi = Nk / n
        self.mu = (gamma.T @ X) / Nk[:, np.newaxis]

        for k in range(self.K):
            diff = X - self.mu[k]
            self.Sigma[k] = (gamma[:, k:k+1] * diff).T @ diff / Nk[k]
            self.Sigma[k] += 1e-6 * np.eye(X.shape[1])

    def _compute_log_likelihood(self, X):
        n = X.shape[0]
        log_likelihood = 0

        for i in range(n):
            prob = sum(self.pi[k] * stats.multivariate_normal.pdf(
                X[i], self.mu[k], self.Sigma[k]) for k in range(self.K))
            log_likelihood += np.log(prob + 1e-10)

        return log_likelihood

    def predict(self, X):
        gamma = self._e_step(X)
        return np.argmax(gamma, axis=1)

    def predict_proba(self, X):
        return self._e_step(X)

# 3개의 가우시안으로 데이터 생성
np.random.seed(42)
n_samples = 300

X1 = np.random.randn(n_samples, 2) @ np.array([[1, 0.5], [0.5, 1]]) + np.array([0, 0])
X2 = np.random.randn(n_samples, 2) @ np.array([[1, -0.3], [-0.3, 0.5]]) + np.array([5, 5])
X3 = np.random.randn(n_samples, 2) @ np.array([[0.5, 0], [0, 1.5]]) + np.array([0, 5])
X = np.vstack([X1, X2, X3])

# GMM 학습
gmm = GaussianMixture(n_components=3, max_iter=100)
gmm.fit(X)

# 예측
labels = gmm.predict(X)
proba = gmm.predict_proba(X)

# 시각화
fig = plt.figure(figsize=(16, 5))

# 데이터
ax1 = plt.subplot(131)
ax1.scatter(X[:, 0], X[:, 1], alpha=0.3, s=10)
ax1.set_title('Original Data')
ax1.set_xlabel('x₁')
ax1.set_ylabel('x₂')
ax1.grid(True)

# GMM 결과
ax2 = plt.subplot(132)
ax2.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.5, s=10)
for k in range(gmm.K):
    ax2.scatter(gmm.mu[k, 0], gmm.mu[k, 1],
               marker='x', s=300, linewidths=4, color='red')

    # 2-sigma 타원
    from matplotlib.patches import Ellipse
    eigenvalues, eigenvectors = np.linalg.eig(gmm.Sigma[k])
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * 2 * np.sqrt(eigenvalues)

    ellipse = Ellipse(gmm.mu[k], width, height, angle=angle,
                     facecolor='none', edgecolor='red', linewidth=2)
    ax2.add_patch(ellipse)

ax2.set_title('GMM Clustering (with 2σ ellipses)')
ax2.set_xlabel('x₁')
ax2.set_ylabel('x₂')
ax2.grid(True)

# 확률 밀도
ax3 = plt.subplot(133)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# GMM의 확률 밀도
Z = np.zeros_like(xx)
for k in range(gmm.K):
    Z += gmm.pi[k] * stats.multivariate_normal.pdf(
        np.c_[xx.ravel(), yy.ravel()], gmm.mu[k], gmm.Sigma[k]
    ).reshape(xx.shape)

contour = ax3.contourf(xx, yy, Z, levels=20, cmap='viridis', alpha=0.6)
ax3.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis',
           alpha=0.3, s=10, edgecolors='k', linewidth=0.3)
plt.colorbar(contour, ax=ax3)
ax3.set_title('GMM Probability Density')
ax3.set_xlabel('x₁')
ax3.set_ylabel('x₂')
ax3.grid(True)

plt.tight_layout()
plt.savefig('gmm_2d.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nLearned parameters:")
for k in range(gmm.K):
    print(f"\nComponent {k}:")
    print(f"  π_{k} = {gmm.pi[k]:.3f}")
    print(f"  μ_{k} = {gmm.mu[k]}")
    print(f"  Σ_{k} =\n{gmm.Sigma[k]}")
```

## 6. 머신러닝 응용

### 6.1 GLM (Generalized Linear Models)

**일반화 선형 모델**은 지수족 + 링크 함수:

$$\mathbb{E}[Y|X] = g^{-1}(w^T x)$$

- $g$: 링크 함수 (link function)
- 응답 변수 $Y$는 지수족 분포

**예**:
- 선형 회귀: 가우시안, $g$ = identity
- 로지스틱 회귀: 베르누이, $g$ = logit
- 포아송 회귀: 포아송, $g$ = log

### 6.2 베이지안 신경망

가중치에 사전분포:

$$w \sim \mathcal{N}(0, \sigma_w^2 I)$$

사후분포 근사 (변분 추론):

$$q(w) \approx p(w|D)$$

예측 시 불확실성 정량화 가능.

### 6.3 가우시안 과정 (Gaussian Process)

**함수 공간에서의 사전분포**:

$$f \sim \mathcal{GP}(m, k)$$

- $m(x)$: 평균 함수
- $k(x, x')$: 커널 함수 (공분산)

**성질**: 임의의 유한 점 집합에서의 함수 값은 다변량 가우시안.

**장점**: 불확실성 정량화, 커널 트릭

### 6.4 정규화 흐름 (Normalizing Flows)

가우시안에서 복잡한 분포로 변환:

$$x = f_\theta(z), \quad z \sim \mathcal{N}(0, I)$$

$$p(x) = p(z) \left|\det \frac{\partial f_\theta}{\partial z}\right|^{-1}$$

가역 변환 $f_\theta$를 학습.

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 간단한 GLM: 포아송 회귀
from scipy import stats

# 데이터 생성
np.random.seed(42)
n = 200
X = np.random.randn(n, 2)
true_w = np.array([1.5, -0.8])
true_b = 0.5

# 로그 링크: log(λ) = w^T x + b
log_lambda = X @ true_w + true_b
lambda_true = np.exp(log_lambda)

# 포아송 샘플링
y = np.random.poisson(lambda_true)

# PyTorch로 포아송 회귀
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y)

class PoissonRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        log_lambda = self.linear(x)
        return torch.exp(log_lambda)

model = PoissonRegression(2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 포아송 NLL
def poisson_nll(pred, target):
    return -torch.mean(target * torch.log(pred + 1e-8) - pred)

# 학습
losses = []
for epoch in range(1000):
    optimizer.zero_grad()
    pred = model(X_tensor).squeeze()
    loss = poisson_nll(pred, y_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

# 결과
learned_w = model.linear.weight.data.numpy().squeeze()
learned_b = model.linear.bias.data.numpy()[0]

plt.figure(figsize=(14, 4))

plt.subplot(131)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Negative Log-Likelihood')
plt.title('Training Loss (Poisson Regression)')
plt.grid(True)

plt.subplot(132)
with torch.no_grad():
    pred_lambda = model(X_tensor).squeeze().numpy()

plt.scatter(lambda_true, y, alpha=0.5, label='True λ vs Observed y')
plt.scatter(pred_lambda, y, alpha=0.5, label='Predicted λ vs Observed y')
plt.plot([0, lambda_true.max()], [0, lambda_true.max()], 'r--', label='y=λ')
plt.xlabel('λ (rate parameter)')
plt.ylabel('y (count)')
plt.title('Poisson Regression Predictions')
plt.legend()
plt.grid(True)

plt.subplot(133)
plt.scatter(range(2), true_w, s=100, label='True weights', zorder=5)
plt.scatter(range(2), learned_w, s=100, marker='x', label='Learned weights', zorder=5)
plt.scatter([2], [true_b], s=100)
plt.scatter([2], [learned_b], s=100, marker='x')
plt.xticks([0, 1, 2], ['w₁', 'w₂', 'b'])
plt.ylabel('Value')
plt.title('Parameter Recovery')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('glm_poisson.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"True weights: {true_w}, bias: {true_b:.3f}")
print(f"Learned weights: {learned_w}, bias: {learned_b:.3f}")
```

## 연습 문제

### 문제 1: 지수족 변환

다음 분포를 지수족 형태로 변환하고, $\eta, T(x), A(\eta), h(x)$를 식별하시오:

(a) 기하 분포: $p(x|p) = (1-p)^{x-1} p$ for $x = 1, 2, \ldots$

(b) 라플라스 분포: $p(x|\mu, b) = \frac{1}{2b} \exp\left(-\frac{|x-\mu|}{b}\right)$ ($b$ 고정)

### 문제 2: 로그 분배 함수의 성질

지수족 분포에서 $A(\eta)$의 1차, 2차 미분이 기대값과 분산임을 증명하시오:

(a) $\frac{\partial A}{\partial \eta} = \mathbb{E}[T(X)]$

(b) $\frac{\partial^2 A}{\partial \eta^2} = \text{Var}(T(X))$

**힌트**: 정규화 조건 $\int h(x) \exp(\eta^T T(x) - A(\eta)) dx = 1$을 $\eta$에 대해 미분.

### 문제 3: 감마-포아송 켤레

포아송 분포 $p(x|\lambda) = \frac{\lambda^x e^{-\lambda}}{x!}$의 켤레 사전분포가 감마 분포 $\text{Gamma}(\alpha, \beta)$임을 보이고, 사후분포를 유도하시오.

**데이터**: $n$번 관측, 총 $\sum x_i = s$

**사후**: $\lambda | D \sim \text{Gamma}(\alpha + s, \beta + n)$

Python으로 검증하시오.

### 문제 4: 다변량 가우시안의 조건부

3차원 가우시안 $\mathcal{N}(\mu, \Sigma)$에서:

$$\mu = \begin{bmatrix} 0 \\ 1 \\ 2 \end{bmatrix}, \quad \Sigma = \begin{bmatrix} 1 & 0.5 & 0.3 \\ 0.5 & 2 & 0.4 \\ 0.3 & 0.4 & 1 \end{bmatrix}$$

$x_3 = 1.5$로 주어졌을 때, $(x_1, x_2)$의 조건부 분포 파라미터를 계산하시오.

### 문제 5: GMM vs K-means

(a) GMM과 K-means의 차이점을 설명하시오.

(b) K-means를 GMM의 특수 케이스로 해석할 수 있음을 보이시오.
    **힌트**: $\Sigma_k = \sigma^2 I$이고 $\sigma \to 0$인 극한.

(c) Python으로 동일한 데이터에 대해 K-means와 GMM을 비교하시오.

## 참고 자료

1. **Bishop, C. M. (2006).** *Pattern Recognition and Machine Learning*. Chapter 2 (확률 분포), Chapter 9 (EM과 GMM).
2. **Murphy, K. P. (2022).** *Probabilistic Machine Learning: An Introduction*. Chapter 3 (확률 분포), Chapter 21 (GLM).
3. **Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020).** *Mathematics for Machine Learning*. Chapter 6 (확률과 분포).
4. **Rasmussen, C. E., & Williams, C. K. I. (2006).** *Gaussian Processes for Machine Learning*. MIT Press.
5. **논문**: McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman and Hall.
6. **scikit-learn 문서**: Gaussian Mixture - https://scikit-learn.org/stable/modules/mixture.html
7. **튜토리얼**: Rezende, D. J., & Mohamed, S. (2015). "Variational Inference with Normalizing Flows". *ICML*.
8. **블로그**: Distill.pub - Exponential Family - https://distill.pub/
