# 통계 분석 기초

## 개요

데이터 분석에 필요한 통계 개념인 확률분포, 가설검정, 신뢰구간, 상관분석을 다룹니다.

---

## 1. 확률 분포

### 1.1 이산 확률 분포

```python
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# 이항분포 (Binomial)
# 성공 확률 p인 n번의 독립 시행에서 성공 횟수
n, p = 10, 0.5
binom = stats.binom(n, p)

x = np.arange(0, n+1)
plt.figure(figsize=(10, 4))
plt.bar(x, binom.pmf(x), edgecolor='black')
plt.title(f'Binomial Distribution (n={n}, p={p})')
plt.xlabel('Number of Successes')
plt.ylabel('Probability')
plt.show()

print(f"평균: {binom.mean():.2f}")
print(f"분산: {binom.var():.2f}")
print(f"P(X=5): {binom.pmf(5):.4f}")
print(f"P(X≤5): {binom.cdf(5):.4f}")

# 포아송 분포 (Poisson)
# 단위 시간/공간 내 발생 횟수 (평균 λ)
lam = 5
poisson = stats.poisson(lam)

x = np.arange(0, 15)
plt.figure(figsize=(10, 4))
plt.bar(x, poisson.pmf(x), edgecolor='black')
plt.title(f'Poisson Distribution (λ={lam})')
plt.xlabel('Events')
plt.ylabel('Probability')
plt.show()

# 기하분포 (Geometric)
# 첫 번째 성공까지의 시행 횟수
p = 0.3
geom = stats.geom(p)
print(f"기하분포 P(X=3): {geom.pmf(3):.4f}")
```

### 1.2 연속 확률 분포

```python
# 정규분포 (Normal/Gaussian)
mu, sigma = 0, 1
norm = stats.norm(mu, sigma)

x = np.linspace(-4, 4, 100)
plt.figure(figsize=(10, 4))
plt.plot(x, norm.pdf(x), 'b-', linewidth=2)
plt.fill_between(x, norm.pdf(x), alpha=0.3)
plt.title(f'Normal Distribution (μ={mu}, σ={sigma})')
plt.xlabel('X')
plt.ylabel('Probability Density')
plt.show()

print(f"P(-1 < X < 1): {norm.cdf(1) - norm.cdf(-1):.4f}")  # 약 68%
print(f"P(-2 < X < 2): {norm.cdf(2) - norm.cdf(-2):.4f}")  # 약 95%
print(f"Z-value for 95%: {norm.ppf(0.975):.4f}")  # 역CDF

# 지수분포 (Exponential)
# 포아송 과정에서 다음 사건까지의 시간
lam = 0.5
expon = stats.expon(scale=1/lam)

x = np.linspace(0, 10, 100)
plt.figure(figsize=(10, 4))
plt.plot(x, expon.pdf(x), 'r-', linewidth=2)
plt.fill_between(x, expon.pdf(x), alpha=0.3)
plt.title(f'Exponential Distribution (λ={lam})')
plt.show()

# 균등분포 (Uniform)
a, b = 0, 10
uniform = stats.uniform(a, b-a)
print(f"균등분포 평균: {uniform.mean():.2f}")
print(f"균등분포 분산: {uniform.var():.2f}")

# t 분포
df = 5
t = stats.t(df)

x = np.linspace(-4, 4, 100)
plt.figure(figsize=(10, 4))
plt.plot(x, norm.pdf(x), 'b-', linewidth=2, label='Normal')
plt.plot(x, t.pdf(x), 'r-', linewidth=2, label=f't (df={df})')
plt.legend()
plt.title('Normal vs t Distribution')
plt.show()

# 카이제곱 분포
df = 5
chi2 = stats.chi2(df)

x = np.linspace(0, 20, 100)
plt.figure(figsize=(10, 4))
plt.plot(x, chi2.pdf(x), 'g-', linewidth=2)
plt.title(f'Chi-square Distribution (df={df})')
plt.show()
```

### 1.3 분포 적합도 검정

```python
# 데이터가 특정 분포를 따르는지 검정
np.random.seed(42)
data = np.random.normal(100, 15, 200)

# Shapiro-Wilk 정규성 검정
stat, p_value = stats.shapiro(data)
print(f"Shapiro-Wilk 검정: 통계량={stat:.4f}, p-value={p_value:.4f}")
if p_value > 0.05:
    print("정규분포를 따름 (귀무가설 채택)")

# Kolmogorov-Smirnov 검정
stat, p_value = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
print(f"K-S 검정: 통계량={stat:.4f}, p-value={p_value:.4f}")

# Anderson-Darling 검정
result = stats.anderson(data, dist='norm')
print(f"Anderson-Darling: 통계량={result.statistic:.4f}")
for i, (cv, sig) in enumerate(zip(result.critical_values, result.significance_level)):
    print(f"  유의수준 {sig}%: 임계값={cv:.3f}")
```

---

## 2. 가설 검정

### 2.1 가설 검정 기초

```python
"""
가설 검정 절차:
1. 귀무가설(H0)과 대립가설(H1) 설정
2. 유의수준(α) 결정 (보통 0.05)
3. 검정통계량 계산
4. p-value 또는 기각역 확인
5. 결론 도출

용어:
- 제1종 오류(α): 참인 귀무가설을 기각 (위양성)
- 제2종 오류(β): 거짓인 귀무가설을 채택 (위음성)
- 검정력(1-β): 거짓인 귀무가설을 기각할 확률
"""
```

### 2.2 단일 표본 t-검정

```python
# 모집단 평균에 대한 검정
np.random.seed(42)
sample = np.random.normal(105, 15, 30)  # 실제 평균 105

# H0: μ = 100, H1: μ ≠ 100
hypothesized_mean = 100
stat, p_value = stats.ttest_1samp(sample, hypothesized_mean)

print("=== 단일 표본 t-검정 ===")
print(f"표본 평균: {sample.mean():.2f}")
print(f"표본 표준편차: {sample.std():.2f}")
print(f"t-통계량: {stat:.4f}")
print(f"p-value: {p_value:.4f}")

alpha = 0.05
if p_value < alpha:
    print(f"결론: H0 기각 (μ ≠ {hypothesized_mean})")
else:
    print(f"결론: H0 채택 (μ = {hypothesized_mean})")
```

### 2.3 독립 표본 t-검정

```python
# 두 그룹의 평균 비교
np.random.seed(42)
group1 = np.random.normal(100, 15, 50)  # 그룹 1
group2 = np.random.normal(108, 15, 50)  # 그룹 2

# H0: μ1 = μ2, H1: μ1 ≠ μ2
stat, p_value = stats.ttest_ind(group1, group2)

print("=== 독립 표본 t-검정 ===")
print(f"그룹 1 평균: {group1.mean():.2f}")
print(f"그룹 2 평균: {group2.mean():.2f}")
print(f"t-통계량: {stat:.4f}")
print(f"p-value: {p_value:.4f}")

# 등분산 검정 (Levene's test)
stat_levene, p_levene = stats.levene(group1, group2)
print(f"\nLevene 등분산 검정: p-value={p_levene:.4f}")

# 등분산이 아닌 경우 Welch's t-test
stat_welch, p_welch = stats.ttest_ind(group1, group2, equal_var=False)
print(f"Welch's t-test p-value: {p_welch:.4f}")
```

### 2.4 대응 표본 t-검정

```python
# 동일 대상의 사전-사후 비교
np.random.seed(42)
before = np.random.normal(100, 15, 30)
after = before + np.random.normal(5, 5, 30)  # 평균 5점 상승

# H0: μd = 0, H1: μd ≠ 0
stat, p_value = stats.ttest_rel(before, after)

print("=== 대응 표본 t-검정 ===")
print(f"사전 평균: {before.mean():.2f}")
print(f"사후 평균: {after.mean():.2f}")
print(f"차이 평균: {(after - before).mean():.2f}")
print(f"t-통계량: {stat:.4f}")
print(f"p-value: {p_value:.4f}")
```

### 2.5 ANOVA (분산분석)

```python
# 세 그룹 이상의 평균 비교
np.random.seed(42)
group1 = np.random.normal(100, 15, 30)
group2 = np.random.normal(105, 15, 30)
group3 = np.random.normal(110, 15, 30)

# 일원배치 분산분석
stat, p_value = stats.f_oneway(group1, group2, group3)

print("=== 일원배치 ANOVA ===")
print(f"그룹 1 평균: {group1.mean():.2f}")
print(f"그룹 2 평균: {group2.mean():.2f}")
print(f"그룹 3 평균: {group3.mean():.2f}")
print(f"F-통계량: {stat:.4f}")
print(f"p-value: {p_value:.4f}")

# 사후검정 (Tukey HSD)
from scipy.stats import tukey_hsd
result = tukey_hsd(group1, group2, group3)
print("\n사후검정 (Tukey HSD):")
print(result)
```

### 2.6 비모수 검정

```python
# Mann-Whitney U 검정 (독립 표본)
np.random.seed(42)
group1 = np.random.exponential(10, 30)
group2 = np.random.exponential(15, 30)

stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
print("=== Mann-Whitney U 검정 ===")
print(f"U-통계량: {stat:.4f}")
print(f"p-value: {p_value:.4f}")

# Wilcoxon 부호순위 검정 (대응 표본)
before = np.random.exponential(10, 30)
after = before + np.random.exponential(2, 30)

stat, p_value = stats.wilcoxon(before, after)
print("\n=== Wilcoxon 부호순위 검정 ===")
print(f"통계량: {stat:.4f}")
print(f"p-value: {p_value:.4f}")

# Kruskal-Wallis 검정 (비모수 ANOVA)
group1 = np.random.exponential(10, 30)
group2 = np.random.exponential(12, 30)
group3 = np.random.exponential(15, 30)

stat, p_value = stats.kruskal(group1, group2, group3)
print("\n=== Kruskal-Wallis 검정 ===")
print(f"H-통계량: {stat:.4f}")
print(f"p-value: {p_value:.4f}")
```

---

## 3. 신뢰구간

### 3.1 평균의 신뢰구간

```python
import scipy.stats as stats
import numpy as np

np.random.seed(42)
sample = np.random.normal(100, 15, 50)

# 평균과 표준오차
mean = sample.mean()
sem = stats.sem(sample)  # 표준오차
n = len(sample)

# 95% 신뢰구간
confidence = 0.95
alpha = 1 - confidence
t_critical = stats.t.ppf(1 - alpha/2, df=n-1)

margin_of_error = t_critical * sem
ci_lower = mean - margin_of_error
ci_upper = mean + margin_of_error

print("=== 평균의 95% 신뢰구간 ===")
print(f"표본 평균: {mean:.2f}")
print(f"표준오차: {sem:.2f}")
print(f"t 임계값: {t_critical:.4f}")
print(f"오차한계: {margin_of_error:.2f}")
print(f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")

# scipy 함수 사용
ci = stats.t.interval(confidence, df=n-1, loc=mean, scale=sem)
print(f"scipy CI: [{ci[0]:.2f}, {ci[1]:.2f}]")
```

### 3.2 비율의 신뢰구간

```python
# 이항분포 기반 신뢰구간
successes = 45
n = 100
p_hat = successes / n

# 정규근사 신뢰구간 (Wald)
z = stats.norm.ppf(0.975)
se = np.sqrt(p_hat * (1 - p_hat) / n)
ci_wald = (p_hat - z * se, p_hat + z * se)

print("=== 비율의 95% 신뢰구간 ===")
print(f"표본 비율: {p_hat:.2f}")
print(f"Wald CI: [{ci_wald[0]:.4f}, {ci_wald[1]:.4f}]")

# Wilson 신뢰구간 (더 정확)
from statsmodels.stats.proportion import proportion_confint
ci_wilson = proportion_confint(successes, n, method='wilson')
print(f"Wilson CI: [{ci_wilson[0]:.4f}, {ci_wilson[1]:.4f}]")

# Clopper-Pearson (정확) 신뢰구간
ci_exact = proportion_confint(successes, n, method='beta')
print(f"Exact CI: [{ci_exact[0]:.4f}, {ci_exact[1]:.4f}]")
```

### 3.3 두 평균 차이의 신뢰구간

```python
np.random.seed(42)
group1 = np.random.normal(100, 15, 30)
group2 = np.random.normal(108, 15, 30)

mean_diff = group1.mean() - group2.mean()
n1, n2 = len(group1), len(group2)
s1, s2 = group1.std(ddof=1), group2.std(ddof=1)

# 풀드 표준오차 (등분산 가정)
sp = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
se = sp * np.sqrt(1/n1 + 1/n2)
df = n1 + n2 - 2

t_critical = stats.t.ppf(0.975, df)
ci = (mean_diff - t_critical * se, mean_diff + t_critical * se)

print("=== 두 평균 차이의 95% 신뢰구간 ===")
print(f"평균 차이: {mean_diff:.2f}")
print(f"95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")
```

---

## 4. 상관분석

### 4.1 피어슨 상관계수

```python
np.random.seed(42)
x = np.random.randn(100)
y = 2 * x + np.random.randn(100) * 0.5

# 피어슨 상관계수
corr, p_value = stats.pearsonr(x, y)

print("=== 피어슨 상관분석 ===")
print(f"상관계수 (r): {corr:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"결정계수 (r²): {corr**2:.4f}")

# 해석
if abs(corr) < 0.3:
    strength = "약한"
elif abs(corr) < 0.7:
    strength = "중간"
else:
    strength = "강한"

direction = "양의" if corr > 0 else "음의"
print(f"해석: {strength} {direction} 상관관계")

# 시각화
plt.figure(figsize=(10, 5))
plt.scatter(x, y, alpha=0.6)
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x, p(x), "r--", linewidth=2, label=f'r = {corr:.3f}')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Pearson Correlation')
plt.legend()
plt.show()
```

### 4.2 스피어만 순위 상관계수

```python
# 비선형 관계나 순서형 데이터에 적합
np.random.seed(42)
x = np.random.randn(50)
y = x**2 + np.random.randn(50) * 0.5  # 비선형 관계

# 피어슨 vs 스피어만
pearson_corr, _ = stats.pearsonr(x, y)
spearman_corr, spearman_p = stats.spearmanr(x, y)

print("=== 상관계수 비교 ===")
print(f"피어슨: {pearson_corr:.4f}")
print(f"스피어만: {spearman_corr:.4f}, p-value: {spearman_p:.4f}")
```

### 4.3 켄달 타우 상관계수

```python
# 순서형 데이터나 작은 샘플에 적합
kendall_corr, kendall_p = stats.kendalltau(x, y)

print("=== 켄달 타우 ===")
print(f"타우: {kendall_corr:.4f}, p-value: {kendall_p:.4f}")
```

### 4.4 편상관분석

```python
import pandas as pd
from scipy import stats

# 제3의 변수를 통제한 상관관계
np.random.seed(42)
z = np.random.randn(100)  # 교란변수
x = z + np.random.randn(100) * 0.3
y = z + np.random.randn(100) * 0.3

# 단순 상관
simple_corr = stats.pearsonr(x, y)[0]
print(f"단순 상관: {simple_corr:.4f}")

# 편상관 (z를 통제)
def partial_corr(x, y, z):
    """z를 통제한 x와 y의 편상관계수"""
    # 잔차 계산
    x_resid = x - np.polyval(np.polyfit(z, x, 1), z)
    y_resid = y - np.polyval(np.polyfit(z, y, 1), z)
    return stats.pearsonr(x_resid, y_resid)[0]

partial = partial_corr(x, y, z)
print(f"편상관 (z 통제): {partial:.4f}")
```

---

## 5. 카이제곱 검정

### 5.1 적합도 검정

```python
# 관측 빈도가 기대 빈도와 일치하는지 검정
observed = [20, 30, 25, 25]  # 관측 빈도
expected = [25, 25, 25, 25]  # 기대 빈도 (균등분포 가정)

stat, p_value = stats.chisquare(observed, expected)

print("=== 카이제곱 적합도 검정 ===")
print(f"χ² 통계량: {stat:.4f}")
print(f"p-value: {p_value:.4f}")
```

### 5.2 독립성 검정

```python
# 두 범주형 변수의 독립성 검정
# 교차표 (성별 x 선호도)
contingency_table = np.array([
    [30, 20, 10],  # 남성: A, B, C
    [15, 35, 20]   # 여성: A, B, C
])

chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

print("=== 카이제곱 독립성 검정 ===")
print(f"χ² 통계량: {chi2:.4f}")
print(f"자유도: {dof}")
print(f"p-value: {p_value:.4f}")
print("\n기대빈도:")
print(expected)

# 효과 크기 (Cramér's V)
n = contingency_table.sum()
min_dim = min(contingency_table.shape) - 1
cramers_v = np.sqrt(chi2 / (n * min_dim))
print(f"\nCramér's V: {cramers_v:.4f}")
```

---

## 6. 효과 크기

```python
"""
효과 크기 (Effect Size):
- 통계적 유의성과 별개로 실질적 의미를 나타냄
- Cohen's d, r, η² 등
"""

def cohens_d(group1, group2):
    """Cohen's d: 두 그룹 평균 차이의 효과 크기"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()

    # 풀드 표준편차
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    d = (group1.mean() - group2.mean()) / pooled_std
    return d

np.random.seed(42)
group1 = np.random.normal(100, 15, 50)
group2 = np.random.normal(108, 15, 50)

d = cohens_d(group1, group2)
print(f"Cohen's d: {d:.4f}")

# 해석
if abs(d) < 0.2:
    effect = "작은"
elif abs(d) < 0.8:
    effect = "중간"
else:
    effect = "큰"
print(f"해석: {effect} 효과")

# 상관계수의 효과 크기
r = 0.5
# r < 0.1: 작은, 0.1-0.3: 중간, > 0.3: 큰

# ANOVA의 효과 크기 (η²)
# η² < 0.01: 작은, 0.01-0.06: 중간, > 0.14: 큰
```

---

## 7. 검정력 분석

```python
from statsmodels.stats.power import TTestIndPower

# 검정력 분석: 필요 표본 크기 계산
power_analysis = TTestIndPower()

# 필요 표본 크기 계산
# 효과 크기 d=0.5, 유의수준 0.05, 검정력 0.8
n = power_analysis.solve_power(effect_size=0.5, alpha=0.05, power=0.8)
print(f"필요 표본 크기 (각 그룹): {np.ceil(n)}")

# 검정력 계산 (표본 크기가 주어졌을 때)
power = power_analysis.solve_power(effect_size=0.5, alpha=0.05, nobs1=50)
print(f"검정력: {power:.4f}")

# 효과 크기별 필요 표본 크기
print("\n효과 크기별 필요 표본 크기 (α=0.05, power=0.8):")
for d in [0.2, 0.5, 0.8]:
    n = power_analysis.solve_power(effect_size=d, alpha=0.05, power=0.8)
    print(f"  d={d}: n={np.ceil(n)}")
```

---

## 연습 문제

### 문제 1: 가설 검정
새로운 교육 방법의 효과를 검정하세요.

```python
np.random.seed(42)
control = np.random.normal(75, 10, 30)
treatment = np.random.normal(82, 10, 30)

# 풀이
stat, p_value = stats.ttest_ind(control, treatment)
d = cohens_d(control, treatment)

print(f"t-통계량: {stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Cohen's d: {d:.4f}")
print(f"결론: {'유의한 차이 있음' if p_value < 0.05 else '유의한 차이 없음'}")
```

### 문제 2: 신뢰구간
설문조사에서 찬성률의 95% 신뢰구간을 구하세요.

```python
n = 500
successes = 280

# 풀이
from statsmodels.stats.proportion import proportion_confint
ci = proportion_confint(successes, n, method='wilson')
print(f"찬성률: {successes/n:.1%}")
print(f"95% CI: [{ci[0]:.1%}, {ci[1]:.1%}]")
```

### 문제 3: 상관분석
두 변수의 관계를 분석하세요.

```python
np.random.seed(42)
hours_studied = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
test_score = np.array([50, 55, 60, 65, 70, 75, 85, 90, 95])

# 풀이
corr, p_value = stats.pearsonr(hours_studied, test_score)
print(f"상관계수: {corr:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"결정계수: {corr**2:.4f}")
```

---

## 요약

| 검정 유형 | 함수 | 조건 |
|----------|------|------|
| 단일 표본 t | `ttest_1samp()` | 모평균 검정 |
| 독립 표본 t | `ttest_ind()` | 두 그룹 평균 비교 |
| 대응 표본 t | `ttest_rel()` | 쌍체 비교 |
| ANOVA | `f_oneway()` | 3그룹 이상 평균 비교 |
| Mann-Whitney U | `mannwhitneyu()` | 비모수 독립 표본 |
| Wilcoxon | `wilcoxon()` | 비모수 대응 표본 |
| 카이제곱 | `chi2_contingency()` | 범주형 독립성 |
| 피어슨 | `pearsonr()` | 선형 상관 |
| 스피어만 | `spearmanr()` | 순위 상관 |
