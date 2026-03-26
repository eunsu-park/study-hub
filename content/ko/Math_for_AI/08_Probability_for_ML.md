# 08. 머신러닝을 위한 확률론 (Probability for Machine Learning)

## 학습 목표

- 확률의 기본 공리와 조건부 확률, 베이즈 정리를 이해하고 응용한다
- 확률 변수의 개념과 이산/연속 분포의 차이를 학습한다
- 기댓값, 분산, 공분산 등 확률 변수의 주요 통계량을 계산하고 해석한다
- 머신러닝에서 자주 사용되는 확률 분포들의 특성과 응용을 학습한다
- 베이즈 정리를 사용한 확률적 추론과 베이지안 업데이트를 구현한다
- 생성 모델과 판별 모델의 차이를 확률론적 관점에서 이해한다

---

## 1. 확률의 기초

### 1.1 확률의 공리

**표본공간 (Sample Space)** $\Omega$: 모든 가능한 결과의 집합

**사건 (Event)** $A$: 표본공간의 부분집합

**확률 측도 (Probability Measure)** $P$는 다음 공리를 만족합니다:

1. **비음성 (Non-negativity)**: $P(A) \geq 0$ for all $A$
2. **정규성 (Normalization)**: $P(\Omega) = 1$
3. **가산 가법성 (Countable Additivity)**: 서로 배반인 사건 $A_1, A_2, \ldots$에 대해
   $$P\left(\bigcup_{i=1}^\infty A_i\right) = \sum_{i=1}^\infty P(A_i)$$

### 1.2 조건부 확률 (Conditional Probability)

사건 $B$가 일어났을 때 사건 $A$가 일어날 확률:

$$
P(A|B) = \frac{P(A \cap B)}{P(B)}, \quad \text{if } P(B) > 0
$$

**직관**: $B$가 일어난 것을 알 때, 표본공간이 $\Omega$에서 $B$로 축소됩니다.

### 1.3 독립성 (Independence)

사건 $A$와 $B$가 독립이라는 것은:

$$
P(A \cap B) = P(A) \cdot P(B)
$$

또는 동등하게:
$$
P(A|B) = P(A)
$$

### 1.4 전확률 법칙 (Law of Total Probability)

$B_1, \ldots, B_n$이 표본공간의 분할 (partition)이면:

$$
P(A) = \sum_{i=1}^n P(A|B_i)P(B_i)
$$

### 1.5 베이즈 정리 (Bayes' Theorem)

$$
P(A|B) = \frac{P(B|A)P(A)}{P(B)}
$$

또는 전확률 법칙을 사용하여:

$$
P(A|B) = \frac{P(B|A)P(A)}{\sum_{i} P(B|A_i)P(A_i)}
$$

**용어:**
- $P(A)$: **사전 확률 (prior)**
- $P(B|A)$: **우도 (likelihood)**
- $P(A|B)$: **사후 확률 (posterior)**
- $P(B)$: **주변 확률 (marginal)** 또는 **증거 (evidence)**

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 베이즈 정리 예제: 의료 진단
# 질병 유병률: 1%
P_disease = 0.01
P_no_disease = 1 - P_disease

# 검사 정확도
# 민감도 (sensitivity): 병이 있을 때 양성 확률
P_positive_given_disease = 0.95
# 특이도 (specificity): 병이 없을 때 음성 확률
P_negative_given_no_disease = 0.95
P_positive_given_no_disease = 1 - P_negative_given_no_disease

# 전확률: 양성 검사 확률
P_positive = (P_positive_given_disease * P_disease +
              P_positive_given_no_disease * P_no_disease)

# 베이즈 정리: 양성일 때 실제 병이 있을 확률
P_disease_given_positive = (P_positive_given_disease * P_disease) / P_positive

print("의료 진단 예제 (베이즈 정리)")
print(f"질병 유병률 (사전 확률): {P_disease:.1%}")
print(f"검사 민감도: {P_positive_given_disease:.1%}")
print(f"검사 특이도: {P_negative_given_no_disease:.1%}")
print(f"\n양성 검사 확률 (전확률): {P_positive:.4f}")
print(f"양성일 때 실제 병이 있을 확률 (사후 확률): {P_disease_given_positive:.1%}")
print(f"\n해석: 검사가 양성이어도 실제 병이 있을 확률은 {P_disease_given_positive:.1%}에 불과")
print("       (낮은 유병률로 인해 위양성이 많음)")

# 시각화: 베이즈 정리
fig, ax = plt.subplots(figsize=(12, 6))

categories = ['사전 확률\n(병 있음)', '우도\n(양성|병)', '사후 확률\n(병|양성)']
probabilities = [P_disease, P_positive_given_disease, P_disease_given_positive]
colors = ['skyblue', 'lightgreen', 'salmon']

bars = ax.bar(categories, probabilities, color=colors, edgecolor='black', linewidth=2)

# 값 표시
for bar, prob in zip(bars, probabilities):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{prob:.1%}', ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_ylabel('확률', fontsize=13)
ax.set_title('베이즈 정리: 의료 진단 예제', fontsize=15)
ax.set_ylim(0, 1.0)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('bayes_theorem_medical.png', dpi=150)
plt.show()
```

## 2. 확률 변수 (Random Variables)

### 2.1 확률 변수의 정의

**확률 변수 (Random Variable)**: 표본공간에서 실수로의 함수
$$X: \Omega \to \mathbb{R}$$

**이산 확률변수**: 가산개의 값을 가짐 (예: 주사위, 동전)
**연속 확률변수**: 연속적인 값을 가짐 (예: 키, 온도)

### 2.2 확률 질량 함수 (PMF)

이산 확률변수 $X$에 대해:

$$
p_X(x) = P(X = x)
$$

**성질:**
- $p_X(x) \geq 0$ for all $x$
- $\sum_{x} p_X(x) = 1$

### 2.3 확률 밀도 함수 (PDF)

연속 확률변수 $X$에 대해:

$$
P(a \leq X \leq b) = \int_a^b f_X(x) dx
$$

**성질:**
- $f_X(x) \geq 0$ for all $x$
- $\int_{-\infty}^{\infty} f_X(x) dx = 1$
- $P(X = x) = 0$ (단일 점의 확률은 0)

### 2.4 누적 분포 함수 (CDF)

$$
F_X(x) = P(X \leq x)
$$

**성질:**
- 비감소 함수
- $\lim_{x \to -\infty} F_X(x) = 0$, $\lim_{x \to \infty} F_X(x) = 1$
- 연속 확률변수: $f_X(x) = \frac{d}{dx}F_X(x)$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. 이산: 이항 분포
n, p = 10, 0.5
x_binom = np.arange(0, n+1)
pmf_binom = stats.binom.pmf(x_binom, n, p)
cdf_binom = stats.binom.cdf(x_binom, n, p)

axes[0, 0].bar(x_binom, pmf_binom, color='skyblue', edgecolor='black')
axes[0, 0].set_title('이항 분포 PMF\n$n=10, p=0.5$', fontsize=12)
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('P(X=x)')
axes[0, 0].grid(True, alpha=0.3)

axes[1, 0].step(x_binom, cdf_binom, where='post', linewidth=2, color='blue')
axes[1, 0].set_title('이항 분포 CDF', fontsize=12)
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('P(X≤x)')
axes[1, 0].grid(True, alpha=0.3)

# 2. 연속: 정규 분포
mu, sigma = 0, 1
x_norm = np.linspace(-4, 4, 1000)
pdf_norm = stats.norm.pdf(x_norm, mu, sigma)
cdf_norm = stats.norm.cdf(x_norm, mu, sigma)

axes[0, 1].plot(x_norm, pdf_norm, linewidth=2, color='red')
axes[0, 1].fill_between(x_norm, pdf_norm, alpha=0.3, color='red')
axes[0, 1].set_title('정규 분포 PDF\n$\mu=0, \sigma=1$', fontsize=12)
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('f(x)')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 1].plot(x_norm, cdf_norm, linewidth=2, color='darkred')
axes[1, 1].set_title('정규 분포 CDF', fontsize=12)
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('F(x)')
axes[1, 1].grid(True, alpha=0.3)

# 3. 연속: 지수 분포
lam = 1.0
x_exp = np.linspace(0, 5, 1000)
pdf_exp = stats.expon.pdf(x_exp, scale=1/lam)
cdf_exp = stats.expon.cdf(x_exp, scale=1/lam)

axes[0, 2].plot(x_exp, pdf_exp, linewidth=2, color='green')
axes[0, 2].fill_between(x_exp, pdf_exp, alpha=0.3, color='green')
axes[0, 2].set_title('지수 분포 PDF\n$\lambda=1$', fontsize=12)
axes[0, 2].set_xlabel('x')
axes[0, 2].set_ylabel('f(x)')
axes[0, 2].grid(True, alpha=0.3)

axes[1, 2].plot(x_exp, cdf_exp, linewidth=2, color='darkgreen')
axes[1, 2].set_title('지수 분포 CDF', fontsize=12)
axes[1, 2].set_xlabel('x')
axes[1, 2].set_ylabel('F(x)')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pmf_pdf_cdf.png', dpi=150)
plt.show()

print("PMF vs PDF:")
print("  PMF (이산): 특정 값의 확률 P(X=x)")
print("  PDF (연속): 확률 밀도, 구간 확률은 적분으로 계산")
print("  CDF: 누적 확률 P(X≤x), 이산/연속 모두 정의")
```

### 2.5 결합/주변/조건부 분포

**결합 분포 (Joint Distribution):**
$$P(X = x, Y = y)$$ 또는 $$f_{X,Y}(x, y)$$

**주변 분포 (Marginal Distribution):**
$$p_X(x) = \sum_y p_{X,Y}(x, y)$$ 또는 $$f_X(x) = \int f_{X,Y}(x, y) dy$$

**조건부 분포 (Conditional Distribution):**
$$p_{X|Y}(x|y) = \frac{p_{X,Y}(x, y)}{p_Y(y)}$$

```python
# 결합 분포 예제: 이변량 정규분포
from scipy.stats import multivariate_normal

# 파라미터
mu = np.array([0, 0])
cov = np.array([[1, 0.7],
                [0.7, 1]])

# 그리드
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# 결합 PDF
rv = multivariate_normal(mu, cov)
Z = rv.pdf(pos)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 등고선
ax = axes[0]
contour = ax.contourf(X, Y, Z, levels=15, cmap='viridis')
plt.colorbar(contour, ax=ax)
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title('결합 분포 $f_{X,Y}(x,y)$ (이변량 정규)', fontsize=14)
ax.grid(True, alpha=0.3)

# 주변 분포
ax = axes[1]
marginal_X = stats.norm.pdf(x, mu[0], np.sqrt(cov[0, 0]))
marginal_Y = stats.norm.pdf(y, mu[1], np.sqrt(cov[1, 1]))
ax.plot(x, marginal_X, linewidth=3, label='주변 분포 $f_X(x)$', color='blue')
ax.plot(y, marginal_Y, linewidth=3, label='주변 분포 $f_Y(y)$', color='red')
ax.set_xlabel('값', fontsize=12)
ax.set_ylabel('밀도', fontsize=12)
ax.set_title('주변 분포', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('joint_marginal_distributions.png', dpi=150)
plt.show()

print(f"공분산 행렬:\n{cov}")
print(f"상관계수: {cov[0,1] / np.sqrt(cov[0,0] * cov[1,1]):.2f}")
```

## 3. 기댓값과 분산

### 3.1 기댓값 (Expectation)

**이산:**
$$\mathbb{E}[X] = \sum_x x \cdot p_X(x)$$

**연속:**
$$\mathbb{E}[X] = \int_{-\infty}^{\infty} x \cdot f_X(x) dx$$

**함수의 기댓값 (LOTUS - Law of the Unconscious Statistician):**
$$\mathbb{E}[g(X)] = \sum_x g(x) \cdot p_X(x) \quad \text{or} \quad \int g(x) \cdot f_X(x) dx$$

### 3.2 기댓값의 성질

1. **선형성 (Linearity):**
   $$\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$$

2. **독립 변수의 곱:**
   $X, Y$ 독립이면 $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$

### 3.3 분산 (Variance)

$$
\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2
$$

**표준편차 (Standard Deviation):**
$$\sigma_X = \sqrt{\text{Var}(X)}$$

**분산의 성질:**
- $\text{Var}(aX + b) = a^2 \text{Var}(X)$
- $X, Y$ 독립이면 $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$

### 3.4 공분산 (Covariance)

$$
\text{Cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])] = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]
$$

**상관계수 (Correlation Coefficient):**
$$
\rho_{X,Y} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y} \in [-1, 1]
$$

```python
import numpy as np
import matplotlib.pyplot as plt

# 몬테카를로로 기댓값과 분산 추정
np.random.seed(42)

# 정규 분포 샘플링
mu, sigma = 2, 1.5
samples = np.random.normal(mu, sigma, 10000)

# 기댓값과 분산 추정
estimated_mean = np.mean(samples)
estimated_var = np.var(samples, ddof=0)
estimated_std = np.std(samples, ddof=0)

print("몬테카를로 추정")
print(f"이론적 평균: {mu}, 추정 평균: {estimated_mean:.4f}")
print(f"이론적 분산: {sigma**2}, 추정 분산: {estimated_var:.4f}")
print(f"이론적 표준편차: {sigma}, 추정 표준편차: {estimated_std:.4f}")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 히스토그램 + 이론적 PDF
ax = axes[0]
ax.hist(samples, bins=50, density=True, alpha=0.7, color='skyblue',
        edgecolor='black', label='샘플 히스토그램')
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
pdf = stats.norm.pdf(x, mu, sigma)
ax.plot(x, pdf, linewidth=3, color='red', label='이론적 PDF')
ax.axvline(estimated_mean, color='green', linestyle='--', linewidth=2,
           label=f'추정 평균 = {estimated_mean:.2f}')
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('밀도', fontsize=12)
ax.set_title(f'정규 분포 샘플링 (μ={mu}, σ={sigma})', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# 샘플 크기에 따른 수렴
ax = axes[1]
sample_sizes = np.arange(10, 10001, 10)
running_means = [np.mean(samples[:n]) for n in sample_sizes]

ax.plot(sample_sizes, running_means, linewidth=2, color='blue',
        label='누적 평균')
ax.axhline(mu, color='red', linestyle='--', linewidth=2, label=f'이론적 평균 = {mu}')
ax.set_xlabel('샘플 크기', fontsize=12)
ax.set_ylabel('누적 평균', fontsize=12)
ax.set_title('대수의 법칙 (Law of Large Numbers)', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('expectation_variance_estimation.png', dpi=150)
plt.show()

# 공분산 예제
np.random.seed(42)
n = 1000

# 양의 상관관계
X1 = np.random.randn(n)
Y1 = 0.8 * X1 + 0.3 * np.random.randn(n)

# 음의 상관관계
X2 = np.random.randn(n)
Y2 = -0.8 * X2 + 0.3 * np.random.randn(n)

# 독립
X3 = np.random.randn(n)
Y3 = np.random.randn(n)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

datasets = [(X1, Y1, '양의 상관'), (X2, Y2, '음의 상관'), (X3, Y3, '독립 (상관 없음)')]
for idx, (X, Y, title) in enumerate(datasets):
    ax = axes[idx]
    ax.scatter(X, Y, alpha=0.5, s=20, edgecolors='k', linewidths=0.5)

    # 통계량 계산
    cov = np.cov(X, Y)[0, 1]
    corr = np.corrcoef(X, Y)[0, 1]

    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title(f'{title}\nCov={cov:.3f}, ρ={corr:.3f}', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('covariance_correlation.png', dpi=150)
plt.show()

print("\n공분산과 상관계수:")
print("  Cov > 0: 양의 관계 (X 증가 → Y 증가)")
print("  Cov < 0: 음의 관계 (X 증가 → Y 감소)")
print("  Cov = 0: 선형 관계 없음 (독립이면 Cov=0, 역은 성립 안 함)")
print("  ρ ∈ [-1, 1]: 정규화된 공분산 (단위 무관)")
```

## 4. 주요 확률 분포

### 4.1 이산 분포

**베르누이 분포 (Bernoulli):**
$$X \sim \text{Ber}(p), \quad P(X=1) = p, \; P(X=0) = 1-p$$
- 평균: $p$, 분산: $p(1-p)$

**이항 분포 (Binomial):**
$$X \sim \text{Bin}(n, p), \quad P(X=k) = \binom{n}{k}p^k(1-p)^{n-k}$$
- $n$번 독립 베르누이 시행의 성공 횟수
- 평균: $np$, 분산: $np(1-p)$

**포아송 분포 (Poisson):**
$$X \sim \text{Pois}(\lambda), \quad P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}$$
- 단위 시간/공간에서의 사건 발생 횟수
- 평균: $\lambda$, 분산: $\lambda$

**카테고리컬 분포 (Categorical):**
$$X \sim \text{Cat}(p_1, \ldots, p_K), \quad P(X=k) = p_k, \; \sum p_k = 1$$
- 다항 분류의 기본 분포

### 4.2 연속 분포

**균등 분포 (Uniform):**
$$X \sim \text{Unif}(a, b), \quad f(x) = \frac{1}{b-a} \text{ for } x \in [a, b]$$
- 평균: $\frac{a+b}{2}$, 분산: $\frac{(b-a)^2}{12}$

**정규 분포 (Normal/Gaussian):**
$$X \sim \mathcal{N}(\mu, \sigma^2), \quad f(x) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$
- 평균: $\mu$, 분산: $\sigma^2$
- 중심극한정리에 의해 자연적으로 나타남

**지수 분포 (Exponential):**
$$X \sim \text{Exp}(\lambda), \quad f(x) = \lambda e^{-\lambda x} \text{ for } x \geq 0$$
- 포아송 과정의 대기 시간
- 평균: $1/\lambda$, 분산: $1/\lambda^2$

**베타 분포 (Beta):**
$$X \sim \text{Beta}(\alpha, \beta), \quad f(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)} \text{ for } x \in [0, 1]$$
- 확률 자체의 분포 (베이지안 추론에서 사전분포)

```python
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 3, figsize=(18, 15))

# 1. 베르누이
ax = axes[0, 0]
p = 0.7
x = [0, 1]
pmf = [1-p, p]
ax.bar(x, pmf, color='skyblue', edgecolor='black', width=0.4)
ax.set_title(f'베르누이 (p={p})', fontsize=12)
ax.set_xticks([0, 1])
ax.set_ylabel('P(X=x)')

# 2. 이항
ax = axes[0, 1]
n, p = 20, 0.5
x = np.arange(0, n+1)
pmf = stats.binom.pmf(x, n, p)
ax.bar(x, pmf, color='lightgreen', edgecolor='black')
ax.set_title(f'이항 (n={n}, p={p})', fontsize=12)
ax.set_xlabel('x')

# 3. 포아송
ax = axes[0, 2]
lam = 5
x = np.arange(0, 20)
pmf = stats.poisson.pmf(x, lam)
ax.bar(x, pmf, color='salmon', edgecolor='black')
ax.set_title(f'포아송 (λ={lam})', fontsize=12)
ax.set_xlabel('x')

# 4. 균등
ax = axes[1, 0]
a, b = 0, 1
x = np.linspace(-0.5, 1.5, 1000)
pdf = stats.uniform.pdf(x, a, b-a)
ax.plot(x, pdf, linewidth=3, color='blue')
ax.fill_between(x, pdf, alpha=0.3, color='blue')
ax.set_title(f'균등 (a={a}, b={b})', fontsize=12)
ax.set_ylabel('f(x)')

# 5. 정규 (여러 파라미터)
ax = axes[1, 1]
x = np.linspace(-5, 5, 1000)
params = [(0, 1), (0, 0.5), (1, 1)]
for mu, sigma in params:
    pdf = stats.norm.pdf(x, mu, sigma)
    ax.plot(x, pdf, linewidth=2, label=f'μ={mu}, σ={sigma}')
ax.set_title('정규 분포', fontsize=12)
ax.legend(fontsize=9)

# 6. 지수
ax = axes[1, 2]
x = np.linspace(0, 5, 1000)
lambdas = [0.5, 1, 2]
for lam in lambdas:
    pdf = stats.expon.pdf(x, scale=1/lam)
    ax.plot(x, pdf, linewidth=2, label=f'λ={lam}')
ax.set_title('지수 분포', fontsize=12)
ax.legend(fontsize=9)

# 7. 감마
ax = axes[2, 0]
x = np.linspace(0, 20, 1000)
params = [(1, 1), (2, 2), (5, 1)]
for k, theta in params:
    pdf = stats.gamma.pdf(x, k, scale=theta)
    ax.plot(x, pdf, linewidth=2, label=f'k={k}, θ={theta}')
ax.set_title('감마 분포', fontsize=12)
ax.set_ylabel('f(x)')
ax.legend(fontsize=9)

# 8. 베타
ax = axes[2, 1]
x = np.linspace(0, 1, 1000)
params = [(0.5, 0.5), (2, 2), (5, 2)]
for alpha, beta in params:
    pdf = stats.beta.pdf(x, alpha, beta)
    ax.plot(x, pdf, linewidth=2, label=f'α={alpha}, β={beta}')
ax.set_title('베타 분포', fontsize=12)
ax.legend(fontsize=9)

# 9. 카이제곱
ax = axes[2, 2]
x = np.linspace(0, 15, 1000)
dfs = [2, 4, 6]
for df in dfs:
    pdf = stats.chi2.pdf(x, df)
    ax.plot(x, pdf, linewidth=2, label=f'df={df}')
ax.set_title('카이제곱 분포', fontsize=12)
ax.legend(fontsize=9)

for ax in axes.flat:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('common_distributions.png', dpi=150)
plt.show()

print("머신러닝에서의 활용:")
print("  베르누이/이항: 이진 분류")
print("  카테고리컬: 다항 분류")
print("  정규: 연속 데이터, 오차 모델, VAE 잠재 공간")
print("  포아송: 카운트 데이터 (추천 시스템, 웹 트래픽)")
print("  베타: 베이지안 추론의 사전분포")
print("  지수/감마: 대기 시간, 생존 분석")
```

### 4.3 다변량 정규분포

$$
\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}), \quad f(\mathbf{x}) = \frac{1}{\sqrt{(2\pi)^d |\boldsymbol{\Sigma}|}}\exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)
$$

- $\boldsymbol{\mu} \in \mathbb{R}^d$: 평균 벡터
- $\boldsymbol{\Sigma} \in \mathbb{R}^{d \times d}$: 공분산 행렬 (양정치)

```python
from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt

# 다변량 정규분포 시각화
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

mu = np.array([0, 0])
covs = [
    np.array([[1, 0], [0, 1]]),      # 독립
    np.array([[1, 0.8], [0.8, 1]]),  # 양의 상관
    np.array([[1, -0.8], [-0.8, 1]]) # 음의 상관
]
titles = ['독립 (ρ=0)', '양의 상관 (ρ=0.8)', '음의 상관 (ρ=-0.8)']

x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

for ax, cov, title in zip(axes, covs, titles):
    rv = multivariate_normal(mu, cov)
    Z = rv.pdf(pos)

    contour = ax.contourf(X, Y, Z, levels=15, cmap='viridis')
    ax.contour(X, Y, Z, levels=15, colors='white', alpha=0.3, linewidths=0.5)
    ax.set_xlabel('$X_1$', fontsize=12)
    ax.set_ylabel('$X_2$', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('multivariate_normal.png', dpi=150)
plt.show()

print("다변량 정규분포:")
print("  - 고차원 데이터 모델링의 기본")
print("  - 가우시안 프로세스, GMM, VAE 등에서 핵심")
print("  - 공분산 행렬로 변수 간 의존성 표현")
```

## 5. 베이즈 정리 심화

### 5.1 베이지안 업데이트

**사전분포 (Prior)** → **데이터 (Data)** → **사후분포 (Posterior)**

$$
P(\theta | D) = \frac{P(D | \theta) P(\theta)}{P(D)} \propto P(D | \theta) P(\theta)
$$

- $\theta$: 파라미터 (확률변수로 취급)
- $D$: 관측 데이터
- $P(\theta)$: 사전 확률 (데이터 전 믿음)
- $P(D | \theta)$: 우도 (데이터가 주어졌을 때 파라미터의 가능도)
- $P(\theta | D)$: 사후 확률 (데이터 후 업데이트된 믿음)

### 5.2 예제: 동전 던지기 (베타-이항 모델)

```python
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# 베타-이항 모델: 동전의 앞면 확률 추정
# 사전분포: Beta(α, β)
# 우도: Binomial
# 사후분포: Beta(α + n_heads, β + n_tails)

np.random.seed(42)

# 진짜 동전 확률 (알 수 없다고 가정)
true_p = 0.7

# 사전분포 (균등 사전: Beta(1, 1))
alpha_prior, beta_prior = 1, 1

# 동전 던지기 시뮬레이션
n_flips_list = [0, 1, 5, 20, 100]
data = np.random.binomial(1, true_p, 100)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

p_vals = np.linspace(0, 1, 1000)

for idx, n_flips in enumerate(n_flips_list):
    ax = axes[idx]

    if n_flips == 0:
        # 사전분포만
        prior_pdf = stats.beta.pdf(p_vals, alpha_prior, beta_prior)
        ax.plot(p_vals, prior_pdf, linewidth=3, color='blue', label='사전분포')
    else:
        # 데이터
        observed_data = data[:n_flips]
        n_heads = np.sum(observed_data)
        n_tails = n_flips - n_heads

        # 사후분포
        alpha_post = alpha_prior + n_heads
        beta_post = beta_prior + n_tails
        posterior_pdf = stats.beta.pdf(p_vals, alpha_post, beta_post)

        # 사전분포
        prior_pdf = stats.beta.pdf(p_vals, alpha_prior, beta_prior)

        ax.plot(p_vals, prior_pdf, linewidth=2, color='blue', linestyle='--',
                label='사전분포', alpha=0.7)
        ax.plot(p_vals, posterior_pdf, linewidth=3, color='red', label='사후분포')

        # MAP 추정 (최대 사후 확률)
        map_estimate = (alpha_post - 1) / (alpha_post + beta_post - 2)
        ax.axvline(map_estimate, color='red', linestyle=':', linewidth=2,
                   label=f'MAP = {map_estimate:.3f}')

    # 진짜 확률
    ax.axvline(true_p, color='green', linestyle='--', linewidth=2,
               label=f'진짜 p = {true_p}')

    ax.set_xlabel('p (앞면 확률)', fontsize=11)
    ax.set_ylabel('밀도', fontsize=11)
    ax.set_title(f'동전 {n_flips}번 던진 후' if n_flips > 0 else '사전분포', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

# 수렴 곡선
ax = axes[-1]
n_range = np.arange(1, 101)
map_estimates = []
for n in n_range:
    n_heads = np.sum(data[:n])
    n_tails = n - n_heads
    alpha_post = alpha_prior + n_heads
    beta_post = beta_prior + n_tails
    map_est = (alpha_post - 1) / (alpha_post + beta_post - 2)
    map_estimates.append(map_est)

ax.plot(n_range, map_estimates, linewidth=2, color='red', label='MAP 추정')
ax.axhline(true_p, color='green', linestyle='--', linewidth=2, label=f'진짜 p = {true_p}')
ax.set_xlabel('동전 던진 횟수', fontsize=11)
ax.set_ylabel('추정된 p', fontsize=11)
ax.set_title('베이지안 학습 수렴', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bayesian_update_coin.png', dpi=150)
plt.show()

print("베이지안 업데이트:")
print("  - 데이터가 늘어날수록 사후분포가 진짜 값 주변에 집중")
print("  - 사전분포의 영향은 데이터가 많아지면 감소")
print("  - 불확실성을 분포로 표현 (점 추정이 아님)")
```

## 6. 머신러닝에서의 확률

### 6.1 생성 모델 vs 판별 모델

**생성 모델 (Generative Model):**
- $P(X, Y) = P(Y)P(X|Y)$를 모델링
- 클래스별 데이터 분포를 학습
- 예측: 베이즈 정리로 $P(Y|X) = \frac{P(X|Y)P(Y)}{P(X)}$
- 예: 나이브 베이즈, GMM, VAE, GAN

**판별 모델 (Discriminative Model):**
- $P(Y|X)$를 직접 모델링
- 결정 경계만 학습
- 예: 로지스틱 회귀, SVM, 신경망

```python
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# 데이터 생성
X, y = make_classification(n_samples=300, n_features=2, n_informative=2,
                           n_redundant=0, n_clusters_per_class=1,
                           random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 생성 모델: 나이브 베이즈
generative_model = GaussianNB()
generative_model.fit(X_train, y_train)

# 판별 모델: 로지스틱 회귀
discriminative_model = LogisticRegression()
discriminative_model.fit(X_train, y_train)

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 그리드
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

models = [
    (generative_model, '생성 모델 (나이브 베이즈)', axes[0]),
    (discriminative_model, '판별 모델 (로지스틱 회귀)', axes[1])
]

for model, title, ax in models:
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    ax.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1],
               c='blue', marker='o', s=50, edgecolors='k', label='Class 0')
    ax.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1],
               c='red', marker='s', s=50, edgecolors='k', label='Class 1')

    score = model.score(X_test, y_test)
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.set_title(f'{title}\nTest Accuracy: {score:.3f}', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('generative_vs_discriminative.png', dpi=150)
plt.show()

print("생성 vs 판별:")
print("  생성: P(X,Y) 전체 분포 모델링 → 샘플 생성 가능")
print("  판별: P(Y|X) 조건부만 → 예측만 가능, 보통 더 높은 성능")
```

### 6.2 나이브 베이즈 분류기

**가정**: 피처들이 클래스가 주어졌을 때 조건부 독립

$$
P(X_1, \ldots, X_d | Y) = \prod_{i=1}^d P(X_i | Y)
$$

**예측:**
$$
\hat{y} = \arg\max_y P(Y=y) \prod_{i=1}^d P(X_i | Y=y)
$$

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# 텍스트 분류 예제
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
train = fetch_20newsgroups(subset='train', categories=categories, random_state=42)
test = fetch_20newsgroups(subset='test', categories=categories, random_state=42)

# 특징 추출 (Bag-of-Words)
vectorizer = CountVectorizer(max_features=1000)
X_train = vectorizer.fit_transform(train.data)
X_test = vectorizer.transform(test.data)

# 나이브 베이즈 학습
nb_model = MultinomialNB()
nb_model.fit(X_train, train.target)

# 예측
y_pred = nb_model.predict(X_test)

print("나이브 베이즈 텍스트 분류:")
print(classification_report(test.target, y_pred, target_names=test.target_names))

print("\n나이브 베이즈의 특징:")
print("  - 조건부 독립 가정 (naive) → 계산 효율적")
print("  - 고차원 데이터에서도 잘 작동 (텍스트 분류)")
print("  - 확률적 해석 가능")
print("  - 작은 데이터셋에서도 합리적 성능")
```

### 6.3 확률적 그래프 모델 (Probabilistic Graphical Models)

- **베이지안 네트워크**: 방향성 비순환 그래프 (DAG)로 변수 간 조건부 독립성 표현
- **마르코프 랜덤 필드**: 무방향 그래프
- **Hidden Markov Model (HMM)**: 시계열 데이터의 숨겨진 상태 추론
- **응용**: 음성 인식, 자연어 처리, 컴퓨터 비전

```python
# 간단한 베이지안 네트워크 예제 (개념적)
import networkx as nx
import matplotlib.pyplot as plt

# 베이지안 네트워크 구조
# Rain → Sprinkler, Rain → Grass Wet, Sprinkler → Grass Wet
G = nx.DiGraph()
G.add_edges_from([('Rain', 'Sprinkler'), ('Rain', 'Grass Wet'),
                  ('Sprinkler', 'Grass Wet')])

plt.figure(figsize=(10, 6))
pos = {'Rain': (0.5, 1), 'Sprinkler': (0, 0), 'Grass Wet': (1, 0)}
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue',
        font_size=12, font_weight='bold', arrowsize=20, arrows=True)
plt.title('베이지안 네트워크: 비 → 스프링클러, 잔디 젖음', fontsize=14)
plt.tight_layout()
plt.savefig('bayesian_network_example.png', dpi=150)
plt.show()

print("확률적 그래프 모델:")
print("  - 변수 간 의존성을 그래프로 표현")
print("  - 조건부 독립성으로 계산 효율화")
print("  - 추론: 관측된 변수로 숨겨진 변수 추정")
print("  - 학습: 데이터로부터 그래프 구조와 확률 파라미터 학습")
```

## 연습 문제

1. **베이즈 정리 응용**: 스팸 필터를 베이즈 정리로 설계하시오. 특정 단어가 포함되었을 때 스팸일 확률을 계산하는 공식을 유도하고, 간단한 예제 데이터로 구현하시오.

2. **확률 분포 피팅**: `scipy.stats`를 사용하여 실제 데이터(예: 키, 시험 점수)에 정규분포를 피팅하고, Q-Q plot으로 적합도를 검증하시오. 정규분포가 적합하지 않다면 다른 분포를 시도하시오.

3. **몬테카를로 적분**: 확률 변수 $X \sim \mathcal{N}(0, 1)$에 대해 $\mathbb{E}[e^X]$를 (1) 해석적으로 계산하고, (2) 몬테카를로 샘플링으로 추정하시오. 샘플 크기를 늘려가며 수렴을 확인하시오.

4. **베이지안 선형 회귀**: 선형 회귀를 베이지안 관점에서 구현하시오. 가중치에 정규분포 사전분포를 부여하고, 데이터를 관측할 때마다 사후분포를 업데이트하시오. 사후분포의 평균과 불확실성을 시각화하시오.

5. **나이브 베이즈 vs 로지스틱 회귀**: Iris 데이터셋에서 나이브 베이즈(생성)와 로지스틱 회귀(판별)의 성능을 비교하시오. 학습 데이터 크기를 변화시키며 두 모델의 학습 곡선을 그리시오. 어떤 상황에서 각 모델이 유리한지 분석하시오.

## 참고 자료

- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press
  - 확률론적 관점에서의 머신러닝 바이블
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer
  - Chapter 1-2: 확률론 기초와 확률 분포
- Wasserman, L. (2004). *All of Statistics*. Springer
  - 통계와 확률의 간결한 요약
- Koller, D., & Friedman, N. (2009). *Probabilistic Graphical Models*. MIT Press
  - 확률적 그래프 모델의 포괄적 교과서
- SciPy Stats Documentation: https://docs.scipy.org/doc/scipy/reference/stats.html
- Seeing Theory (확률/통계 시각화): https://seeing-theory.brown.edu/
- Bayesian Methods for Hackers (온라인 책): https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers
