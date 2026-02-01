# 04. 가설검정 심화 (Advanced Hypothesis Testing)

## 개요

이 장에서는 기본적인 가설검정을 넘어서 **검정력(Power)**, **효과크기(Effect Size)**, **표본크기 결정(Sample Size Determination)**, 그리고 **다중검정 문제(Multiple Testing Problem)**를 다룹니다.

---

## 1. 가설검정 복습

### 1.1 가설검정의 기본 프레임워크

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# 가설검정의 기본 요소
# H₀: 귀무가설 (null hypothesis) - 효과가 없다
# H₁: 대립가설 (alternative hypothesis) - 효과가 있다
# α: 유의수준 (significance level) - 보통 0.05
# p-value: 귀무가설이 참일 때, 관찰된 결과보다 극단적인 결과를 얻을 확률

# 예시: 단일표본 t-검정
np.random.seed(42)
sample = np.random.normal(102, 15, 30)  # 실제 평균 102
mu_0 = 100  # 귀무가설: 평균 = 100

# t-검정 수행
t_stat, p_value = stats.ttest_1samp(sample, mu_0)

print("단일표본 t-검정:")
print(f"표본평균: {sample.mean():.2f}")
print(f"귀무가설 평균: {mu_0}")
print(f"t-통계량: {t_stat:.3f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("→ 귀무가설 기각 (α=0.05)")
else:
    print("→ 귀무가설 기각 실패 (α=0.05)")
```

### 1.2 제1종 오류와 제2종 오류

```python
# 제1종 오류 (Type I Error): H₀가 참인데 기각 → α (유의수준)
# 제2종 오류 (Type II Error): H₀가 거짓인데 기각 실패 → β
# 검정력 (Power): 1 - β = H₀가 거짓일 때 기각할 확률

def visualize_errors(mu_0, mu_1, sigma, n, alpha=0.05):
    """제1종, 제2종 오류 시각화"""
    se = sigma / np.sqrt(n)

    # 귀무가설 하의 분포 (H₀: μ = μ₀)
    null_dist = stats.norm(mu_0, se)

    # 대립가설 하의 분포 (H₁: μ = μ₁)
    alt_dist = stats.norm(mu_1, se)

    # 임계값 (양측검정)
    critical_upper = null_dist.ppf(1 - alpha/2)
    critical_lower = null_dist.ppf(alpha/2)

    # 시각화
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.linspace(mu_0 - 4*se, mu_1 + 4*se, 500)

    # 귀무가설 분포
    ax.plot(x, null_dist.pdf(x), 'b-', lw=2, label=f'H₀: μ = {mu_0}')
    ax.fill_between(x, null_dist.pdf(x),
                    where=(x >= critical_upper) | (x <= critical_lower),
                    alpha=0.3, color='red', label=f'Type I Error (α = {alpha})')

    # 대립가설 분포
    ax.plot(x, alt_dist.pdf(x), 'g-', lw=2, label=f'H₁: μ = {mu_1}')
    ax.fill_between(x, alt_dist.pdf(x),
                    where=(x < critical_upper) & (x > critical_lower),
                    alpha=0.3, color='orange', label='Type II Error (β)')

    # 임계값 표시
    ax.axvline(critical_upper, color='red', linestyle='--', lw=1.5)
    ax.axvline(critical_lower, color='red', linestyle='--', lw=1.5)

    # β와 Power 계산
    beta = alt_dist.cdf(critical_upper) - alt_dist.cdf(critical_lower)
    power = 1 - beta

    ax.set_xlabel('표본평균', fontsize=12)
    ax.set_ylabel('밀도', fontsize=12)
    ax.set_title(f'가설검정의 오류\nβ = {beta:.3f}, Power = {power:.3f}', fontsize=14)
    ax.legend(fontsize=10)

    return beta, power

# 예시
mu_0, mu_1 = 100, 105
sigma = 15
n = 30

beta, power = visualize_errors(mu_0, mu_1, sigma, n, alpha=0.05)
print(f"제2종 오류 (β): {beta:.3f}")
print(f"검정력 (Power): {power:.3f}")
plt.show()
```

---

## 2. 검정력과 효과크기

### 2.1 효과크기 (Effect Size)

```python
# Cohen's d: 평균 차이를 표준편차로 표준화
# d = (μ₁ - μ₀) / σ

def cohens_d(group1, group2):
    """두 그룹의 Cohen's d 계산"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # 합동 표준편차 (pooled standard deviation)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))

    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    return d

# 예시
np.random.seed(42)
control = np.random.normal(100, 15, 50)
treatment = np.random.normal(107, 15, 50)

d = cohens_d(treatment, control)

print("Cohen's d 계산:")
print(f"대조군 평균: {control.mean():.2f}")
print(f"처리군 평균: {treatment.mean():.2f}")
print(f"Cohen's d: {d:.3f}")

# 해석 기준 (Cohen, 1988)
print("\nCohen's d 해석 기준:")
print("  |d| ≈ 0.2: 작은 효과")
print("  |d| ≈ 0.5: 중간 효과")
print("  |d| ≈ 0.8: 큰 효과")
```

```python
# 다양한 효과크기 지표

def effect_sizes(group1, group2):
    """다양한 효과크기 계산"""
    # Cohen's d
    d = cohens_d(group1, group2)

    # Hedges' g (작은 표본 보정)
    n1, n2 = len(group1), len(group2)
    correction = 1 - (3 / (4*(n1+n2) - 9))
    g = d * correction

    # Glass's delta (대조군 표준편차만 사용)
    delta = (np.mean(group1) - np.mean(group2)) / np.std(group2, ddof=1)

    # r (상관계수로 변환)
    t, _ = stats.ttest_ind(group1, group2)
    df = n1 + n2 - 2
    r = np.sqrt(t**2 / (t**2 + df))

    return {
        "Cohen's d": d,
        "Hedges' g": g,
        "Glass's delta": delta,
        "r (effect size)": r
    }

effects = effect_sizes(treatment, control)
print("\n다양한 효과크기 지표:")
for name, value in effects.items():
    print(f"  {name}: {value:.4f}")
```

### 2.2 pingouin을 이용한 효과크기 계산

```python
import pingouin as pg

# pingouin으로 쉽게 효과크기 계산
np.random.seed(42)
group1 = np.random.normal(100, 15, 40)
group2 = np.random.normal(108, 15, 45)

# t-검정과 함께 효과크기 계산
result = pg.ttest(group1, group2, correction=False)
print("pingouin t-검정 결과:")
print(result.to_string())

# 효과크기 별도 계산
d = pg.compute_effsize(group1, group2, eftype='cohen')
hedges = pg.compute_effsize(group1, group2, eftype='hedges')
cles = pg.compute_effsize(group1, group2, eftype='CLES')  # Common Language Effect Size

print(f"\n효과크기:")
print(f"  Cohen's d: {d:.4f}")
print(f"  Hedges' g: {hedges:.4f}")
print(f"  CLES: {cles:.4f}")
print(f"  (CLES: group1에서 무작위로 뽑은 값이 group2보다 클 확률)")
```

### 2.3 검정력 분석

```python
# 검정력: H₁이 참일 때 H₀를 기각할 확률

def power_ttest(mu_0, mu_1, sigma, n, alpha=0.05, alternative='two-sided'):
    """t-검정의 검정력 계산"""
    se = sigma / np.sqrt(n)
    d = (mu_1 - mu_0) / sigma  # 효과크기

    if alternative == 'two-sided':
        critical_upper = stats.norm.ppf(1 - alpha/2)
        critical_lower = -critical_upper

        # 비중심 t-분포 사용 (근사적으로 정규분포)
        ncp = d * np.sqrt(n)  # 비중심 모수
        power = 1 - (stats.norm.cdf(critical_upper - ncp) -
                     stats.norm.cdf(critical_lower - ncp))
    elif alternative == 'greater':
        critical = stats.norm.ppf(1 - alpha)
        ncp = d * np.sqrt(n)
        power = 1 - stats.norm.cdf(critical - ncp)
    else:  # 'less'
        critical = stats.norm.ppf(alpha)
        ncp = d * np.sqrt(n)
        power = stats.norm.cdf(critical - ncp)

    return power

# 예시
mu_0, mu_1 = 100, 105
sigma = 15
n = 30

power = power_ttest(mu_0, mu_1, sigma, n, alpha=0.05)
print(f"검정력 분석:")
print(f"H₀: μ = {mu_0}, H₁: μ = {mu_1}")
print(f"σ = {sigma}, n = {n}, α = 0.05")
print(f"검정력 = {power:.4f}")

# 표본 크기에 따른 검정력 변화
sample_sizes = np.arange(10, 201, 5)
powers = [power_ttest(mu_0, mu_1, sigma, n) for n in sample_sizes]

plt.figure(figsize=(10, 5))
plt.plot(sample_sizes, powers, 'b-', linewidth=2)
plt.axhline(0.8, color='r', linestyle='--', label='Power = 0.8')
plt.xlabel('표본 크기 (n)')
plt.ylabel('검정력')
plt.title('표본 크기와 검정력의 관계')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# 80% 검정력을 위한 최소 표본 크기 찾기
for n in sample_sizes:
    if power_ttest(mu_0, mu_1, sigma, n) >= 0.8:
        print(f"\n80% 검정력을 위한 최소 표본 크기: {n}")
        break
```

---

## 3. 표본크기 결정

### 3.1 statsmodels를 이용한 표본크기 계산

```python
from statsmodels.stats.power import TTestIndPower, TTestPower

# 독립표본 t-검정의 검정력 분석
power_analysis = TTestIndPower()

# 필요 표본크기 계산 (각 그룹)
effect_size = 0.5  # 중간 효과크기
alpha = 0.05
power = 0.8

n = power_analysis.solve_power(effect_size=effect_size,
                                alpha=alpha,
                                power=power,
                                ratio=1,  # 두 그룹 크기 비율
                                alternative='two-sided')

print("독립표본 t-검정 표본크기 계산:")
print(f"효과크기 (d): {effect_size}")
print(f"유의수준 (α): {alpha}")
print(f"검정력 (1-β): {power}")
print(f"필요 표본크기 (각 그룹): {np.ceil(n):.0f}")
print(f"총 필요 표본크기: {np.ceil(n)*2:.0f}")

# 검정력 곡선 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 표본크기 vs 검정력 (다양한 효과크기)
sample_sizes = np.arange(10, 151)
effect_sizes = [0.2, 0.5, 0.8]
colors = ['blue', 'green', 'red']

for d, color in zip(effect_sizes, colors):
    powers = [power_analysis.solve_power(effect_size=d, nobs1=n, alpha=0.05)
              for n in sample_sizes]
    axes[0].plot(sample_sizes, powers, color=color, linewidth=2, label=f'd = {d}')

axes[0].axhline(0.8, color='k', linestyle='--', alpha=0.5)
axes[0].set_xlabel('표본 크기 (각 그룹)')
axes[0].set_ylabel('검정력')
axes[0].set_title('표본 크기와 검정력')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 효과크기 vs 필요 표본크기
effect_range = np.linspace(0.1, 1.5, 50)
required_n = [power_analysis.solve_power(effect_size=d, alpha=0.05, power=0.8)
              for d in effect_range]

axes[1].plot(effect_range, required_n, 'b-', linewidth=2)
axes[1].set_xlabel('효과크기 (d)')
axes[1].set_ylabel('필요 표본크기 (각 그룹)')
axes[1].set_title('효과크기와 필요 표본크기 (Power=0.8)')
axes[1].grid(alpha=0.3)
axes[1].set_ylim(0, 500)

# 주요 효과크기 표시
for d in [0.2, 0.5, 0.8]:
    n_req = power_analysis.solve_power(effect_size=d, alpha=0.05, power=0.8)
    axes[1].scatter([d], [n_req], color='red', s=100, zorder=5)
    axes[1].annotate(f'd={d}\nn={n_req:.0f}', (d, n_req), xytext=(d+0.05, n_req+30))

plt.tight_layout()
plt.show()
```

### 3.2 비율 비교를 위한 표본크기

```python
from statsmodels.stats.power import zt_ind_solve_power
from statsmodels.stats.proportion import proportion_effectsize

# 두 비율 비교를 위한 표본크기
p1 = 0.30  # 대조군 비율
p2 = 0.40  # 처리군 비율 (예상)
alpha = 0.05
power = 0.80

# 효과크기 계산 (Cohen's h)
h = proportion_effectsize(p2, p1)
print(f"Cohen's h: {h:.4f}")

# 필요 표본크기
n = zt_ind_solve_power(effect_size=h, alpha=alpha, power=power)

print(f"\n두 비율 비교 표본크기 계산:")
print(f"대조군 비율: {p1}")
print(f"처리군 비율 (예상): {p2}")
print(f"필요 표본크기 (각 그룹): {np.ceil(n):.0f}")
```

### 3.3 pingouin을 이용한 검정력 분석

```python
# pingouin의 power 함수들

# t-검정 검정력
power_result = pg.power_ttest(d=0.5, n=30, power=None, alpha=0.05,
                               alternative='two-sided')
print(f"검정력 (d=0.5, n=30): {power_result:.4f}")

# 필요 표본크기
n_result = pg.power_ttest(d=0.5, n=None, power=0.8, alpha=0.05,
                           alternative='two-sided')
print(f"필요 표본크기 (d=0.5, power=0.8): {n_result:.1f}")

# 탐지 가능한 효과크기
d_result = pg.power_ttest(d=None, n=50, power=0.8, alpha=0.05,
                           alternative='two-sided')
print(f"탐지 가능 효과크기 (n=50, power=0.8): {d_result:.4f}")

# 상관분석 검정력
power_corr = pg.power_corr(r=0.3, n=50, power=None, alpha=0.05)
print(f"\n상관분석 검정력 (r=0.3, n=50): {power_corr:.4f}")

n_corr = pg.power_corr(r=0.3, n=None, power=0.8, alpha=0.05)
print(f"상관분석 필요 표본크기 (r=0.3, power=0.8): {n_corr:.1f}")
```

---

## 4. 다중검정 문제

### 4.1 다중검정의 문제점

```python
# 여러 번 검정할 때 제1종 오류율 증가

def multiple_testing_demo(n_tests, alpha=0.05, n_simulations=10000):
    """다중검정 시뮬레이션"""
    np.random.seed(42)

    at_least_one_significant = 0

    for _ in range(n_simulations):
        # H₀가 모두 참인 상황 (효과 없음)
        p_values = np.random.uniform(0, 1, n_tests)

        # 하나라도 유의하면 제1종 오류
        if np.any(p_values < alpha):
            at_least_one_significant += 1

    familywise_error_rate = at_least_one_significant / n_simulations

    # 이론적 FWER: 1 - (1-α)^n
    theoretical_fwer = 1 - (1 - alpha) ** n_tests

    return familywise_error_rate, theoretical_fwer

# 다양한 검정 횟수에 대해 시뮬레이션
test_counts = [1, 5, 10, 20, 50, 100]

print("다중검정 시 제1종 오류율 (FWER):")
print("-" * 50)
print(f"{'검정 횟수':<12} {'시뮬레이션':<15} {'이론값':<15}")
print("-" * 50)

for n_tests in test_counts:
    simulated, theoretical = multiple_testing_demo(n_tests)
    print(f"{n_tests:<12} {simulated:<15.4f} {theoretical:<15.4f}")

# 시각화
fig, ax = plt.subplots(figsize=(10, 5))

n_tests_range = np.arange(1, 101)
fwer = 1 - (1 - 0.05) ** n_tests_range

ax.plot(n_tests_range, fwer, 'b-', linewidth=2)
ax.axhline(0.05, color='r', linestyle='--', label='α = 0.05')
ax.fill_between(n_tests_range, 0.05, fwer, alpha=0.3, color='red')
ax.set_xlabel('검정 횟수')
ax.set_ylabel('FWER (Family-Wise Error Rate)')
ax.set_title('다중검정의 문제: FWER 증가')
ax.legend()
ax.grid(alpha=0.3)
plt.show()

print(f"\n20번 검정 시 FWER: {1 - (1-0.05)**20:.2%}")
print(f"100번 검정 시 FWER: {1 - (1-0.05)**100:.2%}")
```

### 4.2 Bonferroni 보정

```python
from statsmodels.stats.multitest import multipletests

def bonferroni_correction(p_values, alpha=0.05):
    """Bonferroni 보정"""
    n = len(p_values)
    adjusted_alpha = alpha / n
    reject = p_values < adjusted_alpha
    adjusted_p = np.minimum(p_values * n, 1.0)
    return reject, adjusted_p, adjusted_alpha

# 예시: 10개의 p-value
np.random.seed(42)
p_values = np.array([0.001, 0.008, 0.025, 0.04, 0.06, 0.10, 0.15, 0.35, 0.50, 0.80])

# Bonferroni 보정
reject, adjusted_p, adj_alpha = bonferroni_correction(p_values)

print("Bonferroni 보정:")
print(f"원래 α = 0.05, 보정된 α = {adj_alpha:.4f}")
print("-" * 60)
print(f"{'i':<5} {'p-value':<12} {'조정 p-value':<15} {'기각여부':<10}")
print("-" * 60)
for i, (p, adj_p, rej) in enumerate(zip(p_values, adjusted_p, reject), 1):
    print(f"{i:<5} {p:<12.4f} {adj_p:<15.4f} {'Yes' if rej else 'No':<10}")

# statsmodels 사용
reject_sm, adjusted_p_sm, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')
print(f"\nstatsmodels 결과 확인: {np.allclose(adjusted_p, adjusted_p_sm)}")
```

### 4.3 Holm-Bonferroni 보정 (Step-down)

```python
def holm_correction(p_values, alpha=0.05):
    """Holm-Bonferroni 보정 (step-down)"""
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    reject = np.zeros(n, dtype=bool)
    adjusted_p = np.zeros(n)

    for i, (idx, p) in enumerate(zip(sorted_indices, sorted_p)):
        # Holm의 보정된 α: α / (n - i)
        holm_alpha = alpha / (n - i)

        if p < holm_alpha:
            reject[idx] = True
        else:
            # 이후 모든 가설 기각 실패
            break

        # 조정된 p-value
        adjusted_p[idx] = min((n - i) * p, 1.0)

    # 단조성 보장
    for i in range(1, n):
        idx = sorted_indices[i]
        prev_idx = sorted_indices[i-1]
        adjusted_p[idx] = max(adjusted_p[idx], adjusted_p[prev_idx])

    return reject, adjusted_p

# 비교
p_values = np.array([0.001, 0.008, 0.025, 0.04, 0.06, 0.10, 0.15, 0.35, 0.50, 0.80])

reject_bonf, adj_p_bonf, _ = bonferroni_correction(p_values)
reject_holm, adj_p_holm = holm_correction(p_values)

# statsmodels로 확인
_, adj_p_holm_sm, _, _ = multipletests(p_values, alpha=0.05, method='holm')

print("Bonferroni vs Holm 비교:")
print("-" * 70)
print(f"{'p-value':<10} {'Bonf p':<12} {'Bonf 기각':<12} {'Holm p':<12} {'Holm 기각':<10}")
print("-" * 70)
for p, bp, br, hp, hr in zip(p_values, adj_p_bonf, reject_bonf,
                              adj_p_holm_sm, reject_holm):
    print(f"{p:<10.4f} {bp:<12.4f} {'Yes' if br else 'No':<12} {hp:<12.4f} {'Yes' if hr else 'No':<10}")

print(f"\n→ Holm 방법이 Bonferroni보다 덜 보수적 (더 많은 기각 가능)")
```

### 4.4 False Discovery Rate (FDR) 보정

```python
def benjamini_hochberg(p_values, alpha=0.05):
    """Benjamini-Hochberg FDR 보정"""
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    # BH 임계값: (i/n) * α
    bh_thresholds = (np.arange(1, n+1) / n) * alpha

    # 기각 결정
    below_threshold = sorted_p <= bh_thresholds
    if np.any(below_threshold):
        max_i = np.max(np.where(below_threshold)[0])
        reject = np.zeros(n, dtype=bool)
        reject[sorted_indices[:max_i+1]] = True
    else:
        reject = np.zeros(n, dtype=bool)

    # 조정된 p-value
    adjusted_p = np.zeros(n)
    for i, idx in enumerate(sorted_indices):
        adjusted_p[idx] = sorted_p[i] * n / (i + 1)

    # 단조성 보장 (역순으로)
    for i in range(n-2, -1, -1):
        idx = sorted_indices[i]
        next_idx = sorted_indices[i+1]
        adjusted_p[idx] = min(adjusted_p[idx], adjusted_p[next_idx])

    adjusted_p = np.minimum(adjusted_p, 1.0)

    return reject, adjusted_p

# 비교
p_values = np.array([0.001, 0.008, 0.025, 0.04, 0.06, 0.10, 0.15, 0.35, 0.50, 0.80])

# statsmodels 사용
_, adj_p_bh, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

print("FDR (Benjamini-Hochberg) 보정:")
print("-" * 60)
print(f"{'순위':<6} {'p-value':<10} {'BH 임계값':<12} {'조정 p-value':<15}")
print("-" * 60)

sorted_p = np.sort(p_values)
n = len(p_values)
for i, p in enumerate(sorted_p, 1):
    threshold = (i / n) * 0.05
    adj_p = p_values[np.argsort(p_values)[i-1]]
    print(f"{i:<6} {p:<10.4f} {threshold:<12.4f} {adj_p_bh[np.argsort(p_values)[i-1]]:<15.4f}")
```

### 4.5 다중검정 방법 종합 비교

```python
# 모든 방법 비교
methods = ['bonferroni', 'holm', 'fdr_bh', 'fdr_by']
method_names = ['Bonferroni', 'Holm', 'BH (FDR)', 'BY (FDR)']

p_values = np.array([0.001, 0.008, 0.025, 0.04, 0.055, 0.10])

print("다중검정 보정 방법 비교:")
print("=" * 80)
print(f"{'원본 p-value':<15}", end='')
for name in method_names:
    print(f"{name:<15}", end='')
print("\n" + "=" * 80)

results = {}
for method in methods:
    _, adj_p, _, _ = multipletests(p_values, alpha=0.05, method=method)
    results[method] = adj_p

for i, p in enumerate(p_values):
    print(f"{p:<15.4f}", end='')
    for method in methods:
        adj = results[method][i]
        sig = "*" if adj < 0.05 else ""
        print(f"{adj:<13.4f}{sig:<2}", end='')
    print()

print("=" * 80)
print("* 표시: 조정된 p-value < 0.05 (유의)")

# 시각화
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(p_values))
width = 0.15

for i, (method, name) in enumerate(zip(methods, method_names)):
    adj_p = results[method]
    ax.bar(x + i*width, adj_p, width, label=name, alpha=0.8)

ax.bar(x - width, p_values, width, label='원본', color='gray', alpha=0.5)
ax.axhline(0.05, color='red', linestyle='--', label='α = 0.05')

ax.set_xlabel('가설')
ax.set_ylabel('p-value')
ax.set_title('다중검정 보정 방법 비교')
ax.set_xticks(x + width)
ax.set_xticklabels([f'H{i+1}' for i in range(len(p_values))])
ax.legend(loc='upper right')
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
```

### 4.6 실제 데이터 예시

```python
# 유전자 발현 데이터 시뮬레이션
np.random.seed(42)

n_genes = 100
n_samples = 20

# 대부분의 유전자는 차이 없음 (H₀ 참)
# 10개의 유전자만 실제 차등 발현 (H₁ 참)
truly_different = 10

# p-value 생성
p_values = np.zeros(n_genes)

# H₀가 참인 유전자들 (균등분포에서 p-value)
p_values[truly_different:] = np.random.uniform(0, 1, n_genes - truly_different)

# H₁이 참인 유전자들 (낮은 p-value 경향)
p_values[:truly_different] = np.random.beta(0.5, 5, truly_different) * 0.1

# 다중검정 보정
_, adj_p_bonf, _, _ = multipletests(p_values, method='bonferroni')
_, adj_p_bh, _, _ = multipletests(p_values, method='fdr_bh')

# 결과 비교
def count_discoveries(p_values, adj_p, alpha=0.05, n_true=truly_different):
    """발견 수 계산"""
    significant_original = p_values < alpha
    significant_adjusted = adj_p < alpha

    # 실제 양성 중 발견된 것 (True Positive)
    tp_original = np.sum(significant_original[:n_true])
    tp_adjusted = np.sum(significant_adjusted[:n_true])

    # 실제 음성 중 잘못 발견된 것 (False Positive)
    fp_original = np.sum(significant_original[n_true:])
    fp_adjusted = np.sum(significant_adjusted[n_true:])

    return {
        'total_discoveries': np.sum(significant_adjusted),
        'true_positives': tp_adjusted,
        'false_positives': fp_adjusted,
        'false_discovery_rate': fp_adjusted / max(np.sum(significant_adjusted), 1)
    }

print(f"시뮬레이션: {n_genes}개 유전자, {truly_different}개가 실제 차등 발현")
print("=" * 60)

# 보정 없음
results_none = count_discoveries(p_values, p_values)
print(f"\n보정 없음:")
print(f"  총 발견: {np.sum(p_values < 0.05)}")
print(f"  True Positives: {np.sum(p_values[:truly_different] < 0.05)}")
print(f"  False Positives: {np.sum(p_values[truly_different:] < 0.05)}")

# Bonferroni
results_bonf = count_discoveries(p_values, adj_p_bonf)
print(f"\nBonferroni 보정:")
print(f"  총 발견: {results_bonf['total_discoveries']}")
print(f"  True Positives: {results_bonf['true_positives']}")
print(f"  False Positives: {results_bonf['false_positives']}")

# BH (FDR)
results_bh = count_discoveries(p_values, adj_p_bh)
print(f"\nBH (FDR) 보정:")
print(f"  총 발견: {results_bh['total_discoveries']}")
print(f"  True Positives: {results_bh['true_positives']}")
print(f"  False Positives: {results_bh['false_positives']}")
print(f"  실제 FDR: {results_bh['false_discovery_rate']:.2%}")
```

---

## 5. pingouin을 활용한 종합 분석

### 5.1 다양한 검정

```python
import pingouin as pg
import pandas as pd

# 데이터 준비
np.random.seed(42)
df = pd.DataFrame({
    'group': ['A']*30 + ['B']*30 + ['C']*30,
    'value': np.concatenate([
        np.random.normal(100, 15, 30),
        np.random.normal(105, 15, 30),
        np.random.normal(110, 15, 30)
    ])
})

# 정규성 검정
print("정규성 검정 (Shapiro-Wilk):")
for group in ['A', 'B', 'C']:
    data = df[df['group'] == group]['value']
    result = pg.normality(data)
    print(f"  그룹 {group}: W={result['W'].values[0]:.4f}, p={result['pval'].values[0]:.4f}")

# 등분산성 검정
print("\n등분산성 검정 (Levene):")
levene_result = pg.homoscedasticity(df, dv='value', group='group')
print(levene_result)

# t-검정 (A vs B)
print("\nt-검정 (A vs B):")
ttest_result = pg.ttest(df[df['group']=='A']['value'],
                        df[df['group']=='B']['value'])
print(ttest_result)

# 비모수 검정 (Mann-Whitney)
print("\nMann-Whitney U 검정 (A vs B):")
mwu_result = pg.mwu(df[df['group']=='A']['value'],
                    df[df['group']=='B']['value'])
print(mwu_result)
```

### 5.2 효과크기 해석 가이드

```python
# 효과크기 해석 함수
def interpret_effect_size(d, measure='cohen_d'):
    """효과크기 해석"""
    d = abs(d)

    if measure == 'cohen_d':
        if d < 0.2:
            return "무시할 수준 (negligible)"
        elif d < 0.5:
            return "작은 효과 (small)"
        elif d < 0.8:
            return "중간 효과 (medium)"
        else:
            return "큰 효과 (large)"
    elif measure == 'r':
        if d < 0.1:
            return "무시할 수준"
        elif d < 0.3:
            return "작은 효과"
        elif d < 0.5:
            return "중간 효과"
        else:
            return "큰 효과"

# 예시
effect_sizes = [0.15, 0.35, 0.65, 0.95]

print("Cohen's d 해석:")
print("-" * 40)
for d in effect_sizes:
    print(f"d = {d:.2f}: {interpret_effect_size(d)}")
```

---

## 연습 문제

### 문제 1: 검정력 분석
두 그룹의 평균 차이가 5점이고, 표준편차가 12일 때:
- (a) 각 그룹 30명으로 검정할 때 검정력은?
- (b) 80% 검정력을 위한 최소 표본크기는?

### 문제 2: 다중검정
다음 10개의 p-value에 대해 Bonferroni와 BH 보정을 적용하시오.
```python
p_values = [0.001, 0.005, 0.010, 0.020, 0.040, 0.080, 0.120, 0.200, 0.500, 0.800]
```

### 문제 3: 효과크기
두 그룹 데이터에서 Cohen's d와 Hedges' g를 계산하고 해석하시오.
```python
group1 = [23, 25, 27, 22, 28, 26, 24, 29, 25, 27]
group2 = [30, 32, 28, 31, 33, 29, 35, 31, 30, 32]
```

---

## 정리

| 개념 | 핵심 내용 | Python |
|------|-----------|--------|
| 제1종 오류 (α) | H₀ 참인데 기각 | 유의수준 |
| 제2종 오류 (β) | H₁ 참인데 미기각 | 1 - 검정력 |
| 검정력 | 1 - β | `pg.power_ttest()` |
| Cohen's d | 효과크기 | `pg.compute_effsize()` |
| Bonferroni | FWER 보정 | `multipletests(method='bonferroni')` |
| Holm | FWER 보정 (덜 보수적) | `multipletests(method='holm')` |
| BH (FDR) | FDR 보정 | `multipletests(method='fdr_bh')` |
