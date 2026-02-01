# 01. 확률론 복습 (Probability Review)

## 개요

확률론은 통계학의 수학적 기초입니다. 이 장에서는 고급 통계학을 학습하기 전에 반드시 이해해야 할 확률론의 핵심 개념들을 복습합니다.

---

## 1. 확률의 공리 (Axioms of Probability)

### 1.1 표본공간과 사건

**표본공간 (Sample Space)**: 모든 가능한 결과의 집합 Ω

**사건 (Event)**: 표본공간의 부분집합

```python
# 예: 주사위 던지기
sample_space = {1, 2, 3, 4, 5, 6}  # Ω

# 사건 정의
event_even = {2, 4, 6}  # 짝수가 나오는 사건
event_greater_than_4 = {5, 6}  # 4보다 큰 사건

# 합사건과 곱사건
event_union = event_even | event_greater_than_4  # {2, 4, 5, 6}
event_intersection = event_even & event_greater_than_4  # {6}
```

### 1.2 콜모고로프 공리 (Kolmogorov Axioms)

세 가지 공리:

1. **비음수성**: P(A) ≥ 0 (모든 사건 A에 대해)
2. **정규성**: P(Ω) = 1 (전체 표본공간의 확률은 1)
3. **가산 가법성**: 서로소인 사건들의 합의 확률 = 각 확률의 합

```python
import numpy as np

def verify_probability_axioms(probabilities: dict) -> bool:
    """확률 공리 검증"""
    probs = list(probabilities.values())

    # 공리 1: 비음수성
    axiom1 = all(p >= 0 for p in probs)

    # 공리 2: 정규성 (합이 1)
    axiom2 = np.isclose(sum(probs), 1.0)

    print(f"공리 1 (비음수성): {axiom1}")
    print(f"공리 2 (정규성, 합={sum(probs):.4f}): {axiom2}")

    return axiom1 and axiom2

# 주사위 확률
dice_probs = {i: 1/6 for i in range(1, 7)}
verify_probability_axioms(dice_probs)
```

### 1.3 확률의 기본 성질

```python
# P(A^c) = 1 - P(A) (여사건)
P_A = 0.3
P_A_complement = 1 - P_A
print(f"P(A) = {P_A}, P(A^c) = {P_A_complement}")

# P(A ∪ B) = P(A) + P(B) - P(A ∩ B) (포함-배제 원리)
P_A, P_B, P_A_and_B = 0.5, 0.4, 0.2
P_A_or_B = P_A + P_B - P_A_and_B
print(f"P(A ∪ B) = {P_A_or_B}")
```

---

## 2. 확률변수 (Random Variables)

### 2.1 이산확률변수 (Discrete Random Variable)

확률변수 X가 셀 수 있는 값들만 가질 때

**확률질량함수 (PMF)**: P(X = x) = f(x)

```python
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np

# 이산확률변수 예시: 이항분포
n, p = 10, 0.3
X = stats.binom(n=n, p=p)

# PMF 계산
x_values = np.arange(0, n+1)
pmf_values = X.pmf(x_values)

# 기대값과 분산
print(f"이항분포 B({n}, {p})")
print(f"E[X] = np = {n*p}")
print(f"Var(X) = np(1-p) = {n*p*(1-p)}")
print(f"scipy 계산: 평균={X.mean():.2f}, 분산={X.var():.2f}")

# 시각화
plt.figure(figsize=(10, 4))
plt.bar(x_values, pmf_values, color='steelblue', alpha=0.7)
plt.xlabel('x')
plt.ylabel('P(X = x)')
plt.title(f'이항분포 PMF: B({n}, {p})')
plt.xticks(x_values)
plt.grid(axis='y', alpha=0.3)
plt.show()
```

### 2.2 연속확률변수 (Continuous Random Variable)

확률변수 X가 연속적인 값을 가질 때

**확률밀도함수 (PDF)**: f(x), P(a ≤ X ≤ b) = ∫[a,b] f(x)dx

```python
# 연속확률변수 예시: 정규분포
mu, sigma = 0, 1
X = stats.norm(loc=mu, scale=sigma)

# PDF 계산
x_values = np.linspace(-4, 4, 1000)
pdf_values = X.pdf(x_values)

# 특정 구간의 확률
prob_between = X.cdf(1) - X.cdf(-1)  # P(-1 ≤ X ≤ 1)
print(f"P(-1 ≤ X ≤ 1) = {prob_between:.4f}")  # 약 68.27%

# 시각화
plt.figure(figsize=(10, 4))
plt.plot(x_values, pdf_values, 'b-', linewidth=2)
plt.fill_between(x_values, pdf_values, where=(x_values >= -1) & (x_values <= 1),
                  alpha=0.3, color='steelblue')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('표준정규분포 PDF')
plt.axhline(y=0, color='k', linewidth=0.5)
plt.grid(alpha=0.3)
plt.show()
```

### 2.3 누적분포함수 (CDF)

```python
# CDF: F(x) = P(X ≤ x)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 이산분포 CDF (이항분포)
n, p = 10, 0.5
X_binom = stats.binom(n=n, p=p)
x_discrete = np.arange(0, n+2)
cdf_discrete = X_binom.cdf(x_discrete)

axes[0].step(x_discrete, cdf_discrete, where='post', color='steelblue', linewidth=2)
axes[0].scatter(x_discrete[:-1], cdf_discrete[:-1], color='steelblue', s=50, zorder=5)
axes[0].set_xlabel('x')
axes[0].set_ylabel('F(x)')
axes[0].set_title(f'이항분포 CDF: B({n}, {p})')
axes[0].grid(alpha=0.3)

# 연속분포 CDF (정규분포)
X_norm = stats.norm(0, 1)
x_cont = np.linspace(-4, 4, 1000)
cdf_cont = X_norm.cdf(x_cont)

axes[1].plot(x_cont, cdf_cont, 'b-', linewidth=2)
axes[1].set_xlabel('x')
axes[1].set_ylabel('F(x)')
axes[1].set_title('표준정규분포 CDF')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 3. 주요 확률분포

### 3.1 이산분포

#### 이항분포 (Binomial Distribution)

n번의 독립 베르누이 시행에서 성공 횟수

```python
# 이항분포: X ~ B(n, p)
n, p = 20, 0.4

X = stats.binom(n=n, p=p)

print(f"이항분포 B({n}, {p})")
print(f"평균: E[X] = {X.mean():.2f}")
print(f"분산: Var(X) = {X.var():.2f}")
print(f"P(X = 8) = {X.pmf(8):.4f}")
print(f"P(X ≤ 8) = {X.cdf(8):.4f}")
print(f"P(X > 8) = {1 - X.cdf(8):.4f}")

# 난수 생성
samples = X.rvs(size=1000, random_state=42)
print(f"\n1000개 샘플의 평균: {np.mean(samples):.2f}")
```

#### 포아송분포 (Poisson Distribution)

단위 시간/공간당 평균 λ번 발생하는 사건의 횟수

```python
# 포아송분포: X ~ Poisson(λ)
lambda_param = 5

X = stats.poisson(mu=lambda_param)

print(f"포아송분포 Poisson({lambda_param})")
print(f"평균: E[X] = {X.mean():.2f}")
print(f"분산: Var(X) = {X.var():.2f}")
print(f"P(X = 3) = {X.pmf(3):.4f}")
print(f"P(X ≤ 3) = {X.cdf(3):.4f}")

# 포아송 분포와 이항분포 비교 (n이 크고 p가 작을 때)
n, p = 100, 0.05  # np = 5 = λ
X_binom = stats.binom(n=n, p=p)
X_poisson = stats.poisson(mu=n*p)

x_vals = np.arange(0, 15)
plt.figure(figsize=(10, 4))
plt.bar(x_vals - 0.2, X_binom.pmf(x_vals), width=0.4, label=f'Binom({n},{p})', alpha=0.7)
plt.bar(x_vals + 0.2, X_poisson.pmf(x_vals), width=0.4, label=f'Poisson({n*p})', alpha=0.7)
plt.xlabel('x')
plt.ylabel('P(X = x)')
plt.title('이항분포와 포아송분포 비교 (n이 크고 p가 작을 때)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

#### 기하분포 (Geometric Distribution)

첫 번째 성공까지의 시행 횟수

```python
# 기하분포: X ~ Geom(p)
p = 0.3

X = stats.geom(p=p)

print(f"기하분포 Geom({p})")
print(f"평균: E[X] = 1/p = {1/p:.2f}")
print(f"분산: Var(X) = (1-p)/p² = {(1-p)/p**2:.2f}")
print(f"P(X = 3) = {X.pmf(3):.4f}")  # 3번째 시행에서 첫 성공

# 무기억성 (Memoryless property)
# P(X > s+t | X > s) = P(X > t)
s, t = 2, 3
P_Xgt_s_plus_t = 1 - X.cdf(s+t)
P_Xgt_s = 1 - X.cdf(s)
P_Xgt_t = 1 - X.cdf(t)

print(f"\n무기억성 검증:")
print(f"P(X > {s+t}) / P(X > {s}) = {P_Xgt_s_plus_t / P_Xgt_s:.4f}")
print(f"P(X > {t}) = {P_Xgt_t:.4f}")
```

### 3.2 연속분포

#### 정규분포 (Normal Distribution)

가장 중요한 연속분포, 중심극한정리의 핵심

```python
# 정규분포: X ~ N(μ, σ²)
mu, sigma = 100, 15  # IQ 예시

X = stats.norm(loc=mu, scale=sigma)

print(f"정규분포 N({mu}, {sigma}²)")
print(f"평균: E[X] = {X.mean():.2f}")
print(f"표준편차: SD(X) = {X.std():.2f}")

# 표준화와 백분위수
x_val = 130
z_score = (x_val - mu) / sigma
print(f"\nX = {x_val}의 Z-점수: {z_score:.2f}")
print(f"P(X ≤ {x_val}) = {X.cdf(x_val):.4f}")

# 백분위수 구하기
percentiles = [0.25, 0.50, 0.75, 0.95]
for p in percentiles:
    print(f"{int(p*100)}백분위수: {X.ppf(p):.2f}")

# 경험적 규칙 (68-95-99.7 rule)
print("\n경험적 규칙:")
print(f"P(μ-σ < X < μ+σ) = {X.cdf(mu+sigma) - X.cdf(mu-sigma):.4f} (≈ 68%)")
print(f"P(μ-2σ < X < μ+2σ) = {X.cdf(mu+2*sigma) - X.cdf(mu-2*sigma):.4f} (≈ 95%)")
print(f"P(μ-3σ < X < μ+3σ) = {X.cdf(mu+3*sigma) - X.cdf(mu-3*sigma):.4f} (≈ 99.7%)")
```

#### 지수분포 (Exponential Distribution)

사건 발생까지의 대기시간

```python
# 지수분포: X ~ Exp(λ) - scipy에서는 scale = 1/λ 사용
lambda_param = 0.5  # 평균 대기시간 = 2
scale = 1 / lambda_param

X = stats.expon(scale=scale)

print(f"지수분포 Exp({lambda_param})")
print(f"평균: E[X] = 1/λ = {X.mean():.2f}")
print(f"분산: Var(X) = 1/λ² = {X.var():.2f}")

# 무기억성 (포아송 과정과 연결)
print("\n무기억성: 이미 t시간 기다렸어도 추가 대기시간 분포는 동일")

# 지수분포와 포아송분포의 관계
# 포아송: 단위 시간당 사건 수 | 지수: 사건 발생까지 대기시간
plt.figure(figsize=(10, 4))
x_vals = np.linspace(0, 10, 1000)
for lam in [0.5, 1.0, 2.0]:
    pdf_vals = stats.expon(scale=1/lam).pdf(x_vals)
    plt.plot(x_vals, pdf_vals, label=f'λ = {lam}')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('지수분포 PDF (다양한 λ)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

#### 감마분포와 베타분포

```python
# 감마분포: X ~ Gamma(α, β)
# 평균 = α/β, 분산 = α/β²
alpha, beta_param = 5, 2

X_gamma = stats.gamma(a=alpha, scale=1/beta_param)
print(f"감마분포 Gamma({alpha}, {beta_param})")
print(f"평균: {X_gamma.mean():.2f}, 분산: {X_gamma.var():.2f}")

# 베타분포: X ~ Beta(α, β) - 0과 1 사이의 확률 모델링
# 평균 = α/(α+β)
alpha, beta_param = 2, 5

X_beta = stats.beta(a=alpha, b=beta_param)
print(f"\n베타분포 Beta({alpha}, {beta_param})")
print(f"평균: {X_beta.mean():.2f}, 분산: {X_beta.var():.4f}")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 감마분포
x_gamma = np.linspace(0, 10, 1000)
for a in [1, 2, 5]:
    axes[0].plot(x_gamma, stats.gamma(a=a, scale=1).pdf(x_gamma), label=f'α={a}, β=1')
axes[0].set_xlabel('x')
axes[0].set_ylabel('f(x)')
axes[0].set_title('감마분포 PDF')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 베타분포
x_beta = np.linspace(0, 1, 1000)
beta_params = [(0.5, 0.5), (1, 1), (2, 5), (5, 2)]
for a, b in beta_params:
    axes[1].plot(x_beta, stats.beta(a=a, b=b).pdf(x_beta), label=f'α={a}, β={b}')
axes[1].set_xlabel('x')
axes[1].set_ylabel('f(x)')
axes[1].set_title('베타분포 PDF')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 4. 결합확률과 조건부확률

### 4.1 결합확률분포 (Joint Distribution)

```python
import pandas as pd

# 이산 결합확률분포 예시
# X: 교육수준 (1=고졸, 2=대졸, 3=대학원)
# Y: 소득수준 (1=저, 2=중, 3=고)

# 결합확률질량함수 (joint PMF)
joint_pmf = np.array([
    [0.10, 0.15, 0.05],  # X=1
    [0.05, 0.20, 0.15],  # X=2
    [0.02, 0.08, 0.20]   # X=3
])

df_joint = pd.DataFrame(joint_pmf,
                        index=['X=1(고졸)', 'X=2(대졸)', 'X=3(대학원)'],
                        columns=['Y=1(저)', 'Y=2(중)', 'Y=3(고)'])

print("결합확률분포:")
print(df_joint)
print(f"\n합계: {joint_pmf.sum():.2f}")

# 주변확률 (Marginal Probability)
marginal_X = joint_pmf.sum(axis=1)
marginal_Y = joint_pmf.sum(axis=0)

print(f"\nX의 주변확률: {marginal_X}")
print(f"Y의 주변확률: {marginal_Y}")
```

### 4.2 조건부확률 (Conditional Probability)

P(A|B) = P(A ∩ B) / P(B)

```python
# 조건부확률질량함수
# P(Y|X=2): 대졸인 경우 소득수준 분포
P_Y_given_X2 = joint_pmf[1, :] / marginal_X[1]
print(f"P(Y|X=대졸) = {P_Y_given_X2}")
print(f"합계: {P_Y_given_X2.sum():.2f}")

# P(X|Y=3): 고소득인 경우 교육수준 분포
P_X_given_Y3 = joint_pmf[:, 2] / marginal_Y[2]
print(f"P(X|Y=고소득) = {P_X_given_Y3}")

# 베이즈 정리
# P(X=대학원|Y=고소득) = P(Y=고소득|X=대학원) * P(X=대학원) / P(Y=고소득)
P_Y3_given_X3 = joint_pmf[2, 2] / marginal_X[2]
P_X3 = marginal_X[2]
P_Y3 = marginal_Y[2]

P_X3_given_Y3_bayes = (P_Y3_given_X3 * P_X3) / P_Y3
print(f"\n베이즈 정리를 이용한 P(X=대학원|Y=고소득) = {P_X3_given_Y3_bayes:.4f}")
print(f"직접 계산: {P_X_given_Y3[2]:.4f}")
```

### 4.3 독립성 (Independence)

두 확률변수 X, Y가 독립이면: P(X, Y) = P(X)P(Y)

```python
def check_independence(joint_pmf):
    """결합분포의 독립성 검정"""
    marginal_X = joint_pmf.sum(axis=1)
    marginal_Y = joint_pmf.sum(axis=0)

    # 독립이면 P(X,Y) = P(X)P(Y)
    expected_if_independent = np.outer(marginal_X, marginal_Y)

    print("독립 가정 시 기대 결합확률:")
    print(np.round(expected_if_independent, 4))
    print("\n실제 결합확률:")
    print(np.round(joint_pmf, 4))

    # 차이 계산
    diff = np.abs(joint_pmf - expected_if_independent)
    print(f"\n최대 차이: {diff.max():.4f}")
    print(f"독립 여부: {'예' if np.allclose(joint_pmf, expected_if_independent) else '아니오'}")

check_independence(joint_pmf)
```

---

## 5. 기대값, 분산, 공분산

### 5.1 기대값 (Expected Value)

E[X] = Σ x·P(X=x) (이산) 또는 ∫ x·f(x)dx (연속)

```python
# 이산확률변수의 기대값
def expected_value_discrete(values, probabilities):
    """이산확률변수의 기대값 계산"""
    return np.sum(values * probabilities)

# 예: 복권 기대값
# 상금: 0원(확률 0.9), 1000원(0.08), 10000원(0.019), 100000원(0.001)
prizes = np.array([0, 1000, 10000, 100000])
probs = np.array([0.9, 0.08, 0.019, 0.001])

E_X = expected_value_discrete(prizes, probs)
print(f"복권 기대 상금: {E_X:.0f}원")
print(f"복권 가격이 500원이면 기대 이익: {E_X - 500:.0f}원")

# 기대값의 선형성
# E[aX + b] = aE[X] + b
a, b = 2, 100
print(f"\nE[2X + 100] = 2*E[X] + 100 = {2*E_X + 100:.0f}원")
```

### 5.2 분산과 표준편차 (Variance and Standard Deviation)

Var(X) = E[(X - μ)²] = E[X²] - (E[X])²

```python
def variance_discrete(values, probabilities):
    """이산확률변수의 분산 계산"""
    E_X = np.sum(values * probabilities)
    E_X2 = np.sum(values**2 * probabilities)
    return E_X2 - E_X**2

Var_X = variance_discrete(prizes, probs)
SD_X = np.sqrt(Var_X)

print(f"분산: Var(X) = {Var_X:.0f}")
print(f"표준편차: SD(X) = {SD_X:.0f}원")

# 분산의 성질
# Var(aX + b) = a²Var(X)
print(f"\nVar(2X + 100) = 4*Var(X) = {4*Var_X:.0f}")
```

### 5.3 공분산과 상관계수 (Covariance and Correlation)

```python
# 연속변수에서의 공분산과 상관계수
np.random.seed(42)

# 양의 상관관계 데이터 생성
n = 500
X = np.random.normal(50, 10, n)
Y = 0.8 * X + np.random.normal(0, 5, n)

# 공분산: Cov(X,Y) = E[(X-μ_X)(Y-μ_Y)]
cov_XY = np.cov(X, Y, ddof=1)[0, 1]
print(f"공분산: Cov(X,Y) = {cov_XY:.2f}")

# 상관계수: ρ = Cov(X,Y) / (SD(X) * SD(Y))
corr_XY = np.corrcoef(X, Y)[0, 1]
print(f"상관계수: ρ = {corr_XY:.4f}")

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 양의 상관관계
axes[0].scatter(X, Y, alpha=0.5, s=10)
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
axes[0].set_title(f'양의 상관관계 (ρ = {corr_XY:.2f})')

# 음의 상관관계
Y_neg = -0.8 * X + 100 + np.random.normal(0, 5, n)
corr_neg = np.corrcoef(X, Y_neg)[0, 1]
axes[1].scatter(X, Y_neg, alpha=0.5, s=10, color='red')
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')
axes[1].set_title(f'음의 상관관계 (ρ = {corr_neg:.2f})')

# 상관관계 없음
Y_none = np.random.normal(50, 10, n)
corr_none = np.corrcoef(X, Y_none)[0, 1]
axes[2].scatter(X, Y_none, alpha=0.5, s=10, color='green')
axes[2].set_xlabel('X')
axes[2].set_ylabel('Y')
axes[2].set_title(f'상관관계 없음 (ρ = {corr_none:.2f})')

plt.tight_layout()
plt.show()
```

### 5.4 기대값과 분산의 성질

```python
# 두 확률변수의 합
# E[X + Y] = E[X] + E[Y] (항상 성립)
# Var(X + Y) = Var(X) + Var(Y) + 2Cov(X,Y)
# 독립이면: Var(X + Y) = Var(X) + Var(Y)

np.random.seed(42)
X = np.random.normal(100, 10, 10000)
Y = np.random.normal(50, 5, 10000)  # 독립

print("독립인 두 확률변수 X, Y:")
print(f"E[X] = {X.mean():.2f}, E[Y] = {Y.mean():.2f}")
print(f"E[X+Y] = {(X+Y).mean():.2f} ≈ E[X] + E[Y] = {X.mean() + Y.mean():.2f}")

print(f"\nVar(X) = {X.var():.2f}, Var(Y) = {Y.var():.2f}")
print(f"Var(X+Y) = {(X+Y).var():.2f} ≈ Var(X) + Var(Y) = {X.var() + Y.var():.2f}")
```

---

## 6. 대수의 법칙과 중심극한정리

### 6.1 대수의 법칙 (Law of Large Numbers)

표본 크기가 커지면 표본평균은 모평균에 수렴

```python
# 대수의 법칙 시뮬레이션
np.random.seed(42)

# 모집단: 주사위 (기대값 = 3.5)
population_mean = 3.5

# 다양한 표본 크기로 평균 계산
sample_sizes = np.logspace(1, 5, 50, dtype=int)
sample_means = []

for n in sample_sizes:
    sample = np.random.randint(1, 7, size=n)
    sample_means.append(sample.mean())

# 시각화
plt.figure(figsize=(10, 5))
plt.semilogx(sample_sizes, sample_means, 'b-', alpha=0.7, linewidth=1)
plt.axhline(y=population_mean, color='r', linestyle='--', label=f'모평균 = {population_mean}')
plt.xlabel('표본 크기 (n)')
plt.ylabel('표본평균')
plt.title('대수의 법칙: 표본평균의 모평균 수렴')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print(f"n=10: 표본평균 = {np.random.randint(1, 7, 10).mean():.2f}")
print(f"n=100: 표본평균 = {np.random.randint(1, 7, 100).mean():.2f}")
print(f"n=10000: 표본평균 = {np.random.randint(1, 7, 10000).mean():.2f}")
```

### 6.2 중심극한정리 (Central Limit Theorem)

표본 크기가 충분히 크면, 표본평균의 분포는 정규분포에 근사

X̄ ~ N(μ, σ²/n) approximately, for large n

```python
def demonstrate_clt(distribution, params, dist_name, n_samples=1000):
    """중심극한정리 시각화"""
    sample_sizes = [1, 5, 30, 100]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for i, n in enumerate(sample_sizes):
        # 표본평균 생성
        sample_means = []
        for _ in range(n_samples):
            sample = distribution.rvs(**params, size=n)
            sample_means.append(sample.mean())

        # 원본 분포 (첫 번째 행)
        if i == 0:
            x = np.linspace(distribution.ppf(0.001, **params),
                           distribution.ppf(0.999, **params), 100)
            axes[0, 0].plot(x, distribution.pdf(x, **params), 'b-', lw=2)
            axes[0, 0].set_title(f'원본 분포: {dist_name}')
            axes[0, 0].set_ylabel('밀도')

        axes[0, i].hist(distribution.rvs(**params, size=1000), bins=30,
                        density=True, alpha=0.7, color='steelblue')
        axes[0, i].set_title(f'n=1 (원본)')

        # 표본평균 분포 (두 번째 행)
        axes[1, i].hist(sample_means, bins=30, density=True, alpha=0.7, color='coral')

        # 이론적 정규분포 (CLT에 의한)
        mu = distribution.mean(**params)
        sigma = distribution.std(**params)
        x = np.linspace(mu - 4*sigma/np.sqrt(n), mu + 4*sigma/np.sqrt(n), 100)
        axes[1, i].plot(x, stats.norm.pdf(x, mu, sigma/np.sqrt(n)),
                        'k--', lw=2, label='이론적 정규분포')

        axes[1, i].set_title(f'n={n}의 표본평균 분포')
        axes[1, i].set_xlabel('표본평균')
        if i == 0:
            axes[1, i].set_ylabel('밀도')
        axes[1, i].legend()

    plt.suptitle(f'중심극한정리: {dist_name}', fontsize=14)
    plt.tight_layout()
    plt.show()

# 지수분포 (비대칭)에서 CLT 확인
demonstrate_clt(stats.expon, {'scale': 2}, '지수분포(λ=0.5)')
```

```python
# 균등분포에서 CLT
demonstrate_clt(stats.uniform, {'loc': 0, 'scale': 1}, '균등분포 U(0,1)')
```

### 6.3 CLT의 수학적 형태

```python
# 표준화된 형태
# Z = (X̄ - μ) / (σ/√n) ~ N(0, 1)

np.random.seed(42)
n = 30
mu, sigma = 50, 10  # 모집단 파라미터

# 10000개의 표본평균 생성
sample_means = []
for _ in range(10000):
    sample = np.random.normal(mu, sigma, n)
    sample_means.append(sample.mean())

sample_means = np.array(sample_means)

# 표준화
Z = (sample_means - mu) / (sigma / np.sqrt(n))

# 표준정규분포와 비교
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 표본평균 분포
axes[0].hist(sample_means, bins=50, density=True, alpha=0.7, color='steelblue')
x = np.linspace(mu - 4*sigma/np.sqrt(n), mu + 4*sigma/np.sqrt(n), 100)
axes[0].plot(x, stats.norm.pdf(x, mu, sigma/np.sqrt(n)), 'r-', lw=2)
axes[0].set_xlabel('표본평균')
axes[0].set_ylabel('밀도')
axes[0].set_title(f'표본평균 분포 (n={n})')
axes[0].axvline(mu, color='k', linestyle='--', label=f'μ={mu}')
axes[0].legend()

# 표준화된 분포
axes[1].hist(Z, bins=50, density=True, alpha=0.7, color='coral')
x_std = np.linspace(-4, 4, 100)
axes[1].plot(x_std, stats.norm.pdf(x_std, 0, 1), 'r-', lw=2, label='N(0,1)')
axes[1].set_xlabel('Z = (X̄ - μ) / (σ/√n)')
axes[1].set_ylabel('밀도')
axes[1].set_title('표준화된 표본평균 분포')
axes[1].legend()

plt.tight_layout()
plt.show()

# 정규성 검정
stat, p_value = stats.shapiro(Z[:5000])  # Shapiro-Wilk는 최대 5000개
print(f"Shapiro-Wilk 검정: 통계량={stat:.4f}, p-value={p_value:.4f}")
```

---

## 7. scipy.stats 분포 종합 예제

### 7.1 분포 객체 사용법

```python
# scipy.stats 분포 객체의 공통 메서드

# 정규분포 예시
X = stats.norm(loc=0, scale=1)

print("scipy.stats 분포 객체 메서드:")
print(f"pdf(0): {X.pdf(0):.4f}  # 확률밀도함수")
print(f"cdf(1.96): {X.cdf(1.96):.4f}  # 누적분포함수")
print(f"ppf(0.975): {X.ppf(0.975):.4f}  # 분위수함수 (CDF의 역함수)")
print(f"sf(1.96): {X.sf(1.96):.4f}  # 생존함수 = 1 - CDF")
print(f"mean(): {X.mean():.4f}  # 평균")
print(f"var(): {X.var():.4f}  # 분산")
print(f"std(): {X.std():.4f}  # 표준편차")
print(f"rvs(5): {X.rvs(5, random_state=42)}  # 난수 생성")

# 구간 확률
print(f"\nP(-1 < X < 1): {X.cdf(1) - X.cdf(-1):.4f}")
print(f"interval(0.95): {X.interval(0.95)}  # 중앙 95% 구간")
```

### 7.2 분포 피팅 (Distribution Fitting)

```python
# 데이터에 분포 피팅하기
np.random.seed(42)

# 실제 데이터 (지수분포에서 생성)
true_lambda = 0.5
data = stats.expon(scale=1/true_lambda).rvs(size=500)

# 지수분포 피팅
loc_fit, scale_fit = stats.expon.fit(data)
lambda_fit = 1 / scale_fit

print(f"실제 λ: {true_lambda}")
print(f"추정된 λ: {lambda_fit:.4f}")

# 피팅 결과 시각화
plt.figure(figsize=(10, 5))
plt.hist(data, bins=30, density=True, alpha=0.7, label='데이터')

x = np.linspace(0, data.max(), 100)
plt.plot(x, stats.expon(loc=loc_fit, scale=scale_fit).pdf(x),
         'r-', lw=2, label=f'피팅된 지수분포 (λ={lambda_fit:.3f})')
plt.plot(x, stats.expon(scale=1/true_lambda).pdf(x),
         'g--', lw=2, label=f'실제 분포 (λ={true_lambda})')

plt.xlabel('x')
plt.ylabel('밀도')
plt.title('분포 피팅 예시')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

### 7.3 주요 분포 요약 테이블

```python
# 주요 분포 요약
distributions = [
    ('정규분포', 'norm', {'loc': 0, 'scale': 1}),
    ('지수분포', 'expon', {'scale': 2}),
    ('감마분포', 'gamma', {'a': 2, 'scale': 2}),
    ('베타분포', 'beta', {'a': 2, 'b': 5}),
    ('카이제곱분포', 'chi2', {'df': 5}),
    ('t-분포', 't', {'df': 10}),
    ('F-분포', 'f', {'dfn': 5, 'dfd': 20}),
]

print("주요 연속분포 요약:")
print("-" * 70)
print(f"{'분포명':<15} {'scipy.stats':<12} {'평균':<12} {'분산':<12}")
print("-" * 70)

for name, dist_name, params in distributions:
    dist = getattr(stats, dist_name)(**params)
    print(f"{name:<15} {dist_name:<12} {dist.mean():<12.4f} {dist.var():<12.4f}")

print("\n주요 이산분포:")
print("-" * 70)
discrete_distributions = [
    ('이항분포', 'binom', {'n': 10, 'p': 0.3}),
    ('포아송분포', 'poisson', {'mu': 5}),
    ('기하분포', 'geom', {'p': 0.3}),
    ('음이항분포', 'nbinom', {'n': 5, 'p': 0.3}),
]

for name, dist_name, params in discrete_distributions:
    dist = getattr(stats, dist_name)(**params)
    print(f"{name:<15} {dist_name:<12} {dist.mean():<12.4f} {dist.var():<12.4f}")
```

---

## 연습 문제

### 문제 1: 확률 계산
어떤 공장에서 생산되는 부품의 수명은 평균 1000시간, 표준편차 100시간인 정규분포를 따릅니다.
- (a) 부품의 수명이 900시간 이상일 확률은?
- (b) 수명이 상위 5%에 해당하는 최소 시간은?

### 문제 2: 중심극한정리
포아송분포 Poisson(λ=4)에서 크기 n=50인 표본을 추출할 때, 표본평균이 3.5와 4.5 사이일 확률을 CLT를 사용하여 구하시오.

### 문제 3: 결합분포
X와 Y의 결합확률질량함수가 다음과 같을 때:
- P(X=0, Y=0) = 0.1, P(X=0, Y=1) = 0.2
- P(X=1, Y=0) = 0.3, P(X=1, Y=1) = 0.4

(a) 주변확률분포를 구하시오.
(b) X와 Y는 독립인가?
(c) Cov(X, Y)를 구하시오.

---

## 정리

| 개념 | 핵심 내용 | Python 함수 |
|------|-----------|-------------|
| 확률공리 | 비음수성, 정규성, 가산가법성 | - |
| PMF/PDF | 이산/연속 확률분포 함수 | `dist.pmf()`, `dist.pdf()` |
| CDF | 누적분포함수 P(X ≤ x) | `dist.cdf()` |
| 기대값 | E[X] = Σ x·P(x) | `dist.mean()`, `np.mean()` |
| 분산 | Var(X) = E[(X-μ)²] | `dist.var()`, `np.var()` |
| 공분산 | Cov(X,Y) = E[(X-μ_X)(Y-μ_Y)] | `np.cov()` |
| 상관계수 | ρ = Cov(X,Y) / (σ_X·σ_Y) | `np.corrcoef()` |
| LLN | X̄ → μ as n → ∞ | 시뮬레이션 |
| CLT | X̄ ~ N(μ, σ²/n) for large n | 시뮬레이션 |
