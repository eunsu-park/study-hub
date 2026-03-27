# 13. 신뢰구간 (Confidence Intervals)

[이전: 표본과 추정](./12_Sampling_and_Estimation.md) | [다음: 가설검정 심화](./14_Hypothesis_Testing_Advanced.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 신뢰구간의 올바른 빈도주의적 해석과 흔한 오해를 설명할 수 있다
2. 모평균에 대한 Z-구간과 t-구간을 구성하고, 각각이 적용되는 상황을 기술할 수 있다
3. 정규 근사(normal approximation)와 윌슨(Wilson) 방법을 이용해 모비율에 대한 신뢰구간을 계산할 수 있다
4. 카이제곱(chi-square) 분포를 이용해 모분산에 대한 신뢰구간을 유도할 수 있다
5. 두 평균의 차(독립 표본과 대응 표본)와 두 비율의 차에 대한 신뢰구간을 계산할 수 있다
6. 임의의 통계량에 대한 부트스트랩(bootstrap) 신뢰구간(백분위수법과 BCa법)을 구현할 수 있다
7. `scipy.stats`와 `statsmodels` 함수를 활용해 신뢰구간을 효율적으로 계산할 수 있다

---

표본 평균과 같은 점추정(point estimate)은 단일 최선 추정값을 제공하지만, 그 추정의 불확실성이 얼마나 큰지는 알려주지 않습니다. 신뢰구간(Confidence Interval)은 모수에 대한 그럴듯한 범위를 제공함으로써 그 불확실성을 정량화합니다. 서로 다른 구간 추정 방법 중 어떤 것을 선택하고, 어떻게 구성하고, 어떻게 해석해야 하는지를 이해하는 것은 모든 데이터 분석 결과를 정직하게 보고하는 데 필수적입니다.

---

## 1. 구간추정의 기본 개념

### 1.1 신뢰구간의 정의

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# 신뢰구간의 올바른 해석
# "95% 신뢰구간"의 의미:
# - 동일한 방법으로 100번 표본추출하여 구간을 구하면,
# - 약 95번은 모수를 포함할 것이다

# 시뮬레이션으로 이해하기
np.random.seed(42)

# 모수 설정 (실제로는 미지)
mu = 100
sigma = 15
n = 30
confidence_level = 0.95

# 100번 표본 추출하여 신뢰구간 계산
n_simulations = 100
intervals = []
contains_mu = 0

z_critical = stats.norm.ppf((1 + confidence_level) / 2)

for i in range(n_simulations):
    sample = np.random.normal(mu, sigma, n)
    x_bar = sample.mean()
    se = sigma / np.sqrt(n)  # 모표준편차를 안다고 가정

    lower = x_bar - z_critical * se
    upper = x_bar + z_critical * se

    intervals.append((lower, upper))
    if lower <= mu <= upper:
        contains_mu += 1

print(f"100개의 95% 신뢰구간 중 모평균을 포함하는 구간: {contains_mu}개")

# 시각화 (첫 20개 구간)
fig, ax = plt.subplots(figsize=(12, 8))

for i in range(20):
    lower, upper = intervals[i]
    color = 'blue' if lower <= mu <= upper else 'red'
    ax.plot([lower, upper], [i, i], color=color, linewidth=2)
    ax.scatter([(lower + upper)/2], [i], color=color, s=30)

ax.axvline(mu, color='green', linestyle='--', linewidth=2, label=f'μ = {mu}')
ax.set_xlabel('값')
ax.set_ylabel('시뮬레이션 번호')
ax.set_title('신뢰구간의 의미: 20개 중 모평균을 포함하지 않는 구간(빨강)')
ax.legend()
ax.set_xlim(90, 110)
plt.show()
```

### 1.2 신뢰수준과 구간 폭의 관계

```python
# 신뢰수준 ↑ → 구간 폭 ↑
# 표본 크기 ↑ → 구간 폭 ↓

np.random.seed(42)
sample = np.random.normal(100, 15, 50)
x_bar = sample.mean()
s = sample.std(ddof=1)
n = len(sample)
se = s / np.sqrt(n)

confidence_levels = [0.80, 0.90, 0.95, 0.99]

print("신뢰수준에 따른 신뢰구간:")
print("-" * 60)
print(f"표본평균 = {x_bar:.2f}, 표준오차 = {se:.2f}")
print("-" * 60)

for cl in confidence_levels:
    t_critical = stats.t.ppf((1 + cl) / 2, df=n-1)
    margin = t_critical * se
    lower = x_bar - margin
    upper = x_bar + margin
    width = upper - lower
    print(f"{int(cl*100)}% CI: [{lower:.2f}, {upper:.2f}], 폭 = {width:.2f}")

# 시각화
fig, ax = plt.subplots(figsize=(10, 5))

colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(confidence_levels)))
for i, (cl, color) in enumerate(zip(confidence_levels, colors)):
    t_critical = stats.t.ppf((1 + cl) / 2, df=n-1)
    margin = t_critical * se
    ax.barh(i, 2*margin, left=x_bar-margin, height=0.6, color=color,
            label=f'{int(cl*100)}% CI', alpha=0.7)

ax.axvline(x_bar, color='red', linestyle='-', linewidth=2, label='표본평균')
ax.set_yticks(range(len(confidence_levels)))
ax.set_yticklabels([f'{int(cl*100)}%' for cl in confidence_levels])
ax.set_xlabel('값')
ax.set_ylabel('신뢰수준')
ax.set_title('신뢰수준과 구간 폭의 관계')
ax.legend(loc='upper right')
plt.show()
```

---

## 2. 모평균의 신뢰구간

### 2.1 모분산이 알려진 경우 (Z-구간)

```python
def ci_mean_z(sample, sigma, confidence=0.95):
    """모분산이 알려진 경우 모평균의 신뢰구간 (Z-구간)"""
    n = len(sample)
    x_bar = np.mean(sample)
    se = sigma / np.sqrt(n)

    z_critical = stats.norm.ppf((1 + confidence) / 2)
    margin = z_critical * se

    return x_bar - margin, x_bar + margin, margin

# 예시: 제조 공정의 불량률 모니터링
# 과거 데이터로 σ = 2.5임을 알고 있음
np.random.seed(42)
sigma_known = 2.5
sample = np.random.normal(50, sigma_known, 40)

lower, upper, margin = ci_mean_z(sample, sigma_known, 0.95)

print("모분산이 알려진 경우 (Z-구간):")
print(f"표본평균: {sample.mean():.3f}")
print(f"95% 신뢰구간: [{lower:.3f}, {upper:.3f}]")
print(f"오차 한계: ±{margin:.3f}")
```

### 2.2 모분산이 알려지지 않은 경우 (t-구간)

```python
def ci_mean_t(sample, confidence=0.95):
    """모분산이 미지인 경우 모평균의 신뢰구간 (t-구간)"""
    n = len(sample)
    x_bar = np.mean(sample)
    s = np.std(sample, ddof=1)
    se = s / np.sqrt(n)

    t_critical = stats.t.ppf((1 + confidence) / 2, df=n-1)
    margin = t_critical * se

    return x_bar - margin, x_bar + margin, margin

# 예시
np.random.seed(42)
sample = np.random.normal(50, 2.5, 40)

lower_t, upper_t, margin_t = ci_mean_t(sample, 0.95)

print("모분산이 미지인 경우 (t-구간):")
print(f"표본평균: {sample.mean():.3f}")
print(f"표본표준편차: {sample.std(ddof=1):.3f}")
print(f"95% 신뢰구간: [{lower_t:.3f}, {upper_t:.3f}]")
print(f"오차 한계: ±{margin_t:.3f}")

# scipy.stats 활용
sem = stats.sem(sample)  # 표준오차
ci_scipy = stats.t.interval(0.95, df=len(sample)-1, loc=sample.mean(), scale=sem)
print(f"\nscipy 결과: [{ci_scipy[0]:.3f}, {ci_scipy[1]:.3f}]")
```

### 2.3 t-분포 이해

```python
# t-분포: 표본 크기가 작을 때 사용
# t = (X̄ - μ) / (S/√n) ~ t(n-1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# t-분포 vs 정규분포
x = np.linspace(-4, 4, 200)
axes[0].plot(x, stats.norm.pdf(x), 'b-', lw=2, label='N(0,1)')

dfs = [3, 10, 30]
colors = ['red', 'green', 'purple']
for df, color in zip(dfs, colors):
    axes[0].plot(x, stats.t.pdf(x, df), linestyle='--', lw=2,
                 color=color, label=f't(df={df})')

axes[0].set_xlabel('x')
axes[0].set_ylabel('밀도')
axes[0].set_title('t-분포 vs 표준정규분포')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 자유도에 따른 임계값 변화
dfs_range = np.arange(2, 101)
t_criticals = [stats.t.ppf(0.975, df) for df in dfs_range]
z_critical = stats.norm.ppf(0.975)

axes[1].plot(dfs_range, t_criticals, 'b-', lw=2, label='t(0.975, df)')
axes[1].axhline(z_critical, color='r', linestyle='--', lw=2,
                label=f'z(0.975) = {z_critical:.3f}')
axes[1].set_xlabel('자유도 (df)')
axes[1].set_ylabel('임계값')
axes[1].set_title('자유도에 따른 95% 임계값')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# 표로 정리
print("자유도에 따른 95% 신뢰구간 임계값:")
print("-" * 30)
for df in [5, 10, 20, 30, 50, 100, np.inf]:
    if df == np.inf:
        t_val = stats.norm.ppf(0.975)
        print(f"df = ∞ (정규): t = {t_val:.4f}")
    else:
        t_val = stats.t.ppf(0.975, df)
        print(f"df = {df:3.0f}: t = {t_val:.4f}")
```

---

## 3. 모비율의 신뢰구간

### 3.1 정규근사를 이용한 신뢰구간

```python
def ci_proportion(successes, n, confidence=0.95):
    """모비율의 신뢰구간 (정규근사)"""
    p_hat = successes / n
    se = np.sqrt(p_hat * (1 - p_hat) / n)

    z_critical = stats.norm.ppf((1 + confidence) / 2)
    margin = z_critical * se

    return p_hat - margin, p_hat + margin

# 예시: 여론조사
# 1000명 중 420명이 찬성
successes = 420
n = 1000

lower, upper = ci_proportion(successes, n, 0.95)

print("모비율의 95% 신뢰구간 (정규근사):")
print(f"표본비율: p̂ = {successes/n:.3f}")
print(f"95% CI: [{lower:.3f}, {upper:.3f}]")
print(f"해석: 모비율은 {lower*100:.1f}%에서 {upper*100:.1f}% 사이일 것으로 추정")

# 정규근사 조건 확인: np ≥ 10 and n(1-p) ≥ 10
p_hat = successes / n
print(f"\n정규근사 조건 확인:")
print(f"np = {n * p_hat:.1f} ≥ 10? {n * p_hat >= 10}")
print(f"n(1-p) = {n * (1-p_hat):.1f} ≥ 10? {n * (1-p_hat) >= 10}")
```

### 3.2 Wilson 신뢰구간 (개선된 방법)

```python
def ci_proportion_wilson(successes, n, confidence=0.95):
    """Wilson 신뢰구간 (더 정확한 방법)"""
    p_hat = successes / n
    z = stats.norm.ppf((1 + confidence) / 2)

    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2*n)) / denominator
    margin = (z / denominator) * np.sqrt(p_hat*(1-p_hat)/n + z**2/(4*n**2))

    return center - margin, center + margin

# 작은 표본에서 비교
successes = 8
n = 20
p_hat = successes / n

lower_normal, upper_normal = ci_proportion(successes, n, 0.95)
lower_wilson, upper_wilson = ci_proportion_wilson(successes, n, 0.95)

print(f"표본비율: p̂ = {p_hat:.3f} (n={n})")
print(f"\n정규근사 CI: [{lower_normal:.3f}, {upper_normal:.3f}]")
print(f"Wilson CI:   [{lower_wilson:.3f}, {upper_wilson:.3f}]")

# scipy의 proportion_confint
from statsmodels.stats.proportion import proportion_confint

ci_wilson = proportion_confint(successes, n, alpha=0.05, method='wilson')
ci_normal = proportion_confint(successes, n, alpha=0.05, method='normal')

print(f"\nstatsmodels 결과:")
print(f"  Normal: [{ci_normal[0]:.3f}, {ci_normal[1]:.3f}]")
print(f"  Wilson: [{ci_wilson[0]:.3f}, {ci_wilson[1]:.3f}]")
```

### 3.3 비율의 신뢰구간 비교

```python
from statsmodels.stats.proportion import proportion_confint

# 다양한 방법 비교
successes = 15
n = 50
p_hat = successes / n

methods = ['normal', 'wilson', 'jeffreys', 'agresti_coull', 'beta']

print(f"표본비율 = {p_hat:.3f} (successes={successes}, n={n})")
print("\n비율 신뢰구간 방법 비교:")
print("-" * 50)

for method in methods:
    lower, upper = proportion_confint(successes, n, alpha=0.05, method=method)
    width = upper - lower
    print(f"{method:<15}: [{lower:.4f}, {upper:.4f}], 폭={width:.4f}")

# 시각화
fig, ax = plt.subplots(figsize=(10, 6))

y_positions = range(len(methods))
for i, method in enumerate(methods):
    lower, upper = proportion_confint(successes, n, alpha=0.05, method=method)
    ax.barh(i, upper-lower, left=lower, height=0.6, alpha=0.7)
    ax.scatter([p_hat], [i], color='red', s=50, zorder=5)

ax.axvline(p_hat, color='red', linestyle='--', alpha=0.5, label='표본비율')
ax.set_yticks(y_positions)
ax.set_yticklabels(methods)
ax.set_xlabel('비율')
ax.set_title('비율 신뢰구간 방법 비교')
ax.legend()
plt.tight_layout()
plt.show()
```

---

## 4. 모분산의 신뢰구간

### 4.1 카이제곱 분포를 이용한 신뢰구간

```python
def ci_variance(sample, confidence=0.95):
    """모분산의 신뢰구간 (정규모집단 가정)"""
    n = len(sample)
    s_squared = np.var(sample, ddof=1)

    alpha = 1 - confidence
    chi2_lower = stats.chi2.ppf(alpha/2, df=n-1)
    chi2_upper = stats.chi2.ppf(1 - alpha/2, df=n-1)

    # 신뢰구간: ((n-1)s² / χ²_upper, (n-1)s² / χ²_lower)
    var_lower = (n - 1) * s_squared / chi2_upper
    var_upper = (n - 1) * s_squared / chi2_lower

    return var_lower, var_upper

def ci_std(sample, confidence=0.95):
    """모표준편차의 신뢰구간"""
    var_lower, var_upper = ci_variance(sample, confidence)
    return np.sqrt(var_lower), np.sqrt(var_upper)

# 예시: 품질관리에서 분산 추정
np.random.seed(42)
sample = np.random.normal(100, 5, 30)  # 실제 σ = 5

s_squared = np.var(sample, ddof=1)
s = np.std(sample, ddof=1)

var_lower, var_upper = ci_variance(sample, 0.95)
std_lower, std_upper = ci_std(sample, 0.95)

print("모분산의 95% 신뢰구간:")
print(f"표본분산: s² = {s_squared:.3f}")
print(f"분산 CI: [{var_lower:.3f}, {var_upper:.3f}]")

print(f"\n모표준편차의 95% 신뢰구간:")
print(f"표본표준편차: s = {s:.3f}")
print(f"표준편차 CI: [{std_lower:.3f}, {std_upper:.3f}]")
print(f"실제 σ = 5가 구간에 포함되는가? {std_lower <= 5 <= std_upper}")
```

### 4.2 카이제곱 분포의 비대칭성

```python
# 분산의 신뢰구간이 평균 중심이 아닌 이유

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 카이제곱 분포
df = 20
x = np.linspace(0, 50, 200)
chi2_pdf = stats.chi2.pdf(x, df)

# 임계값
alpha = 0.05
chi2_lower = stats.chi2.ppf(alpha/2, df)
chi2_upper = stats.chi2.ppf(1 - alpha/2, df)

axes[0].plot(x, chi2_pdf, 'b-', lw=2)
axes[0].fill_between(x, chi2_pdf, where=(x >= chi2_lower) & (x <= chi2_upper),
                      alpha=0.3, color='blue')
axes[0].axvline(chi2_lower, color='r', linestyle='--', label=f'χ²_L = {chi2_lower:.2f}')
axes[0].axvline(chi2_upper, color='r', linestyle='--', label=f'χ²_U = {chi2_upper:.2f}')
axes[0].axvline(df, color='g', linestyle=':', label=f'E[χ²] = {df}')
axes[0].set_xlabel('χ²')
axes[0].set_ylabel('밀도')
axes[0].set_title(f'카이제곱 분포 (df={df})')
axes[0].legend()

# 분산 신뢰구간의 시뮬레이션
np.random.seed(42)
true_variance = 25  # σ² = 25
n = 21  # df = 20
n_simulations = 1000

var_estimates = []
ci_lowers = []
ci_uppers = []

for _ in range(n_simulations):
    sample = np.random.normal(0, np.sqrt(true_variance), n)
    s2 = np.var(sample, ddof=1)
    var_estimates.append(s2)

    lower = (n-1) * s2 / chi2_upper
    upper = (n-1) * s2 / chi2_lower
    ci_lowers.append(lower)
    ci_uppers.append(upper)

axes[1].hist(var_estimates, bins=50, density=True, alpha=0.7, label='표본분산 분포')
axes[1].axvline(true_variance, color='r', linestyle='--', lw=2,
                label=f'σ² = {true_variance}')
axes[1].axvline(np.mean(var_estimates), color='g', linestyle=':',
                label=f'E[s²] = {np.mean(var_estimates):.2f}')
axes[1].set_xlabel('표본분산')
axes[1].set_ylabel('밀도')
axes[1].set_title('표본분산의 분포')
axes[1].legend()

plt.tight_layout()
plt.show()

# 커버리지 확인
coverage = np.mean([(l <= true_variance <= u)
                    for l, u in zip(ci_lowers, ci_uppers)])
print(f"95% 신뢰구간의 실제 커버리지: {coverage*100:.1f}%")
```

---

## 5. 두 집단 비교를 위한 신뢰구간

### 5.1 두 모평균 차이의 신뢰구간

```python
def ci_two_means_independent(sample1, sample2, confidence=0.95, equal_var=True):
    """독립 두 표본 평균 차이의 신뢰구간"""
    n1, n2 = len(sample1), len(sample2)
    x1_bar, x2_bar = sample1.mean(), sample2.mean()
    s1, s2 = sample1.std(ddof=1), sample2.std(ddof=1)

    if equal_var:
        # 합동분산 (pooled variance)
        sp_squared = ((n1-1)*s1**2 + (n2-1)*s2**2) / (n1 + n2 - 2)
        se = np.sqrt(sp_squared * (1/n1 + 1/n2))
        df = n1 + n2 - 2
    else:
        # Welch's t-test (등분산 가정 X)
        se = np.sqrt(s1**2/n1 + s2**2/n2)
        # Welch-Satterthwaite 자유도
        df = (s1**2/n1 + s2**2/n2)**2 / \
             ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))

    t_critical = stats.t.ppf((1 + confidence) / 2, df)
    margin = t_critical * se
    diff = x1_bar - x2_bar

    return diff - margin, diff + margin, diff, df

# 예시: A/B 테스트 결과
np.random.seed(42)
group_A = np.random.normal(105, 15, 50)  # 처리군
group_B = np.random.normal(100, 15, 50)  # 대조군

# 등분산 가정
lower_eq, upper_eq, diff, df = ci_two_means_independent(group_A, group_B,
                                                        equal_var=True)
print("두 모평균 차이의 95% 신뢰구간:")
print(f"그룹 A 평균: {group_A.mean():.2f}, 그룹 B 평균: {group_B.mean():.2f}")
print(f"차이 (A - B): {diff:.2f}")
print(f"등분산 가정 CI (df={df:.0f}): [{lower_eq:.2f}, {upper_eq:.2f}]")

# 등분산 가정 X (Welch)
lower_welch, upper_welch, diff, df_welch = ci_two_means_independent(group_A, group_B,
                                                                     equal_var=False)
print(f"Welch CI (df={df_welch:.1f}): [{lower_welch:.2f}, {upper_welch:.2f}]")

# scipy 확인
from scipy.stats import ttest_ind
t_stat, p_val = ttest_ind(group_A, group_B, equal_var=False)
print(f"\nscipy t-검정 통계량: {t_stat:.3f}, p-value: {p_val:.4f}")
```

### 5.2 대응표본 평균 차이의 신뢰구간

```python
def ci_paired_difference(before, after, confidence=0.95):
    """대응표본 평균 차이의 신뢰구간"""
    diff = after - before
    n = len(diff)
    d_bar = diff.mean()
    s_d = diff.std(ddof=1)
    se = s_d / np.sqrt(n)

    t_critical = stats.t.ppf((1 + confidence) / 2, df=n-1)
    margin = t_critical * se

    return d_bar - margin, d_bar + margin, d_bar

# 예시: 다이어트 프로그램 전후 체중
np.random.seed(42)
weight_before = np.random.normal(75, 10, 30)
weight_after = weight_before - np.random.normal(3, 2, 30)  # 평균 3kg 감소

lower, upper, mean_diff = ci_paired_difference(weight_before, weight_after, 0.95)

print("대응표본 평균 차이의 95% 신뢰구간:")
print(f"전: {weight_before.mean():.2f} kg, 후: {weight_after.mean():.2f} kg")
print(f"평균 변화: {mean_diff:.2f} kg")
print(f"95% CI: [{lower:.2f}, {upper:.2f}] kg")

if upper < 0:
    print("→ 신뢰구간이 0을 포함하지 않으므로, 체중 감소가 유의함")
```

### 5.3 두 모비율 차이의 신뢰구간

```python
def ci_two_proportions(successes1, n1, successes2, n2, confidence=0.95):
    """두 모비율 차이의 신뢰구간"""
    p1 = successes1 / n1
    p2 = successes2 / n2
    diff = p1 - p2

    se = np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
    z_critical = stats.norm.ppf((1 + confidence) / 2)
    margin = z_critical * se

    return diff - margin, diff + margin, diff

# 예시: 두 광고의 클릭률 비교
# 광고 A: 1000명 노출, 150명 클릭
# 광고 B: 1200명 노출, 132명 클릭

lower, upper, diff = ci_two_proportions(150, 1000, 132, 1200, 0.95)

print("두 모비율 차이의 95% 신뢰구간:")
print(f"광고 A 클릭률: {150/1000:.3f}")
print(f"광고 B 클릭률: {132/1200:.3f}")
print(f"차이: {diff:.4f} ({diff*100:.2f}%p)")
print(f"95% CI: [{lower:.4f}, {upper:.4f}]")

if lower > 0:
    print("→ 광고 A의 클릭률이 유의하게 높음")
elif upper < 0:
    print("→ 광고 B의 클릭률이 유의하게 높음")
else:
    print("→ 두 광고의 클릭률에 유의한 차이 없음")
```

---

## 6. 부트스트랩 신뢰구간

### 6.1 부트스트랩 방법 소개

```python
def bootstrap_ci(data, statistic_func, confidence=0.95, n_bootstrap=10000):
    """부트스트랩 신뢰구간 (백분위수 방법)"""
    np.random.seed(42)
    n = len(data)
    bootstrap_statistics = []

    for _ in range(n_bootstrap):
        # 복원추출로 부트스트랩 표본 생성
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_statistics.append(statistic_func(bootstrap_sample))

    # 백분위수 신뢰구간
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_statistics, 100 * alpha / 2)
    upper = np.percentile(bootstrap_statistics, 100 * (1 - alpha / 2))

    return lower, upper, np.array(bootstrap_statistics)

# 예시: 평균의 부트스트랩 신뢰구간
np.random.seed(42)
sample = np.random.exponential(scale=5, size=50)  # 비대칭 분포

lower_boot, upper_boot, boot_means = bootstrap_ci(sample, np.mean, 0.95)

# 기존 t-분포 방법과 비교
lower_t, upper_t, _ = ci_mean_t(sample, 0.95)

print("평균의 95% 신뢰구간 비교:")
print(f"표본평균: {sample.mean():.3f}")
print(f"t-분포:   [{lower_t:.3f}, {upper_t:.3f}]")
print(f"부트스트랩: [{lower_boot:.3f}, {upper_boot:.3f}]")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 원본 데이터
axes[0].hist(sample, bins=20, density=True, alpha=0.7, edgecolor='black')
axes[0].axvline(sample.mean(), color='r', linestyle='--', linewidth=2, label='표본평균')
axes[0].set_xlabel('값')
axes[0].set_ylabel('밀도')
axes[0].set_title('원본 데이터 (지수분포)')
axes[0].legend()

# 부트스트랩 분포
axes[1].hist(boot_means, bins=50, density=True, alpha=0.7, edgecolor='black')
axes[1].axvline(lower_boot, color='r', linestyle='--', label=f'95% CI: [{lower_boot:.2f}, {upper_boot:.2f}]')
axes[1].axvline(upper_boot, color='r', linestyle='--')
axes[1].axvline(sample.mean(), color='g', linestyle='-', linewidth=2, label='표본평균')
axes[1].set_xlabel('부트스트랩 표본평균')
axes[1].set_ylabel('밀도')
axes[1].set_title('부트스트랩 분포 (10,000회)')
axes[1].legend()

plt.tight_layout()
plt.show()
```

### 6.2 다양한 부트스트랩 방법

```python
def bootstrap_ci_bca(data, statistic_func, confidence=0.95, n_bootstrap=10000):
    """BCa (Bias-Corrected and Accelerated) 부트스트랩 신뢰구간"""
    from scipy.stats import norm

    np.random.seed(42)
    n = len(data)
    original_stat = statistic_func(data)

    # 부트스트랩 통계량
    boot_stats = []
    for _ in range(n_bootstrap):
        boot_sample = np.random.choice(data, size=n, replace=True)
        boot_stats.append(statistic_func(boot_sample))
    boot_stats = np.array(boot_stats)

    # 편향 수정 (bias correction)
    z0 = norm.ppf(np.mean(boot_stats < original_stat))

    # 가속 계수 (acceleration) - jackknife 사용
    jackknife_stats = []
    for i in range(n):
        jack_sample = np.delete(data, i)
        jackknife_stats.append(statistic_func(jack_sample))
    jackknife_stats = np.array(jackknife_stats)
    jack_mean = jackknife_stats.mean()
    a = np.sum((jack_mean - jackknife_stats)**3) / \
        (6 * (np.sum((jack_mean - jackknife_stats)**2))**1.5 + 1e-10)

    # BCa 백분위수 계산
    alpha = 1 - confidence
    z_alpha_lower = norm.ppf(alpha / 2)
    z_alpha_upper = norm.ppf(1 - alpha / 2)

    alpha_lower = norm.cdf(z0 + (z0 + z_alpha_lower) / (1 - a * (z0 + z_alpha_lower)))
    alpha_upper = norm.cdf(z0 + (z0 + z_alpha_upper) / (1 - a * (z0 + z_alpha_upper)))

    lower = np.percentile(boot_stats, 100 * alpha_lower)
    upper = np.percentile(boot_stats, 100 * alpha_upper)

    return lower, upper

# 비교
np.random.seed(42)
sample = np.random.exponential(scale=5, size=30)

# 백분위수 방법
lower_pct, upper_pct, _ = bootstrap_ci(sample, np.mean, 0.95)

# BCa 방법
lower_bca, upper_bca = bootstrap_ci_bca(sample, np.mean, 0.95)

print("부트스트랩 방법 비교:")
print(f"표본평균: {sample.mean():.3f}")
print(f"백분위수:  [{lower_pct:.3f}, {upper_pct:.3f}]")
print(f"BCa:       [{lower_bca:.3f}, {upper_bca:.3f}]")
```

### 6.3 중앙값의 부트스트랩 신뢰구간

```python
# 중앙값처럼 분포를 모르는 통계량에 부트스트랩 활용

np.random.seed(42)
sample = np.random.lognormal(mean=3, sigma=0.5, size=100)

lower_median, upper_median, boot_medians = bootstrap_ci(sample, np.median, 0.95)

print("중앙값의 95% 부트스트랩 신뢰구간:")
print(f"표본중앙값: {np.median(sample):.3f}")
print(f"95% CI: [{lower_median:.3f}, {upper_median:.3f}]")

# 상관계수의 부트스트랩 신뢰구간
np.random.seed(42)
x = np.random.normal(0, 1, 50)
y = 0.7 * x + np.random.normal(0, 0.5, 50)
data_combined = np.column_stack([x, y])

def correlation(data):
    return np.corrcoef(data[:, 0], data[:, 1])[0, 1]

lower_corr, upper_corr, boot_corrs = bootstrap_ci(data_combined, correlation, 0.95)

print(f"\n상관계수의 95% 부트스트랩 신뢰구간:")
print(f"표본상관계수: {np.corrcoef(x, y)[0, 1]:.3f}")
print(f"95% CI: [{lower_corr:.3f}, {upper_corr:.3f}]")
```

---

## 7. scipy를 이용한 신뢰구간 계산

### 7.1 기본 신뢰구간 함수

```python
# scipy.stats의 interval 메서드 활용

# 정규분포에서 모평균 신뢰구간
np.random.seed(42)
sample = np.random.normal(100, 15, 50)

# t-분포 이용
mean = sample.mean()
sem = stats.sem(sample)  # 표준오차

ci_95 = stats.t.interval(confidence=0.95, df=len(sample)-1, loc=mean, scale=sem)
ci_99 = stats.t.interval(confidence=0.99, df=len(sample)-1, loc=mean, scale=sem)

print("scipy.stats.t.interval 사용:")
print(f"표본평균: {mean:.2f}")
print(f"95% CI: [{ci_95[0]:.2f}, {ci_95[1]:.2f}]")
print(f"99% CI: [{ci_99[0]:.2f}, {ci_99[1]:.2f}]")

# bootstrap 메서드 (scipy >= 1.9)
from scipy.stats import bootstrap

np.random.seed(42)
sample_tuple = (sample,)  # tuple로 전달해야 함
result = bootstrap(sample_tuple, np.mean, confidence_level=0.95, n_resamples=9999)

print(f"\nscipy.stats.bootstrap 사용:")
print(f"95% CI: [{result.confidence_interval.low:.2f}, {result.confidence_interval.high:.2f}]")
```

### 7.2 statsmodels 활용

```python
import statsmodels.api as sm
from statsmodels.stats.weightstats import DescrStatsW

# DescrStatsW로 신뢰구간 계산
np.random.seed(42)
sample = np.random.normal(100, 15, 50)

d = DescrStatsW(sample)

print("statsmodels DescrStatsW 사용:")
print(f"표본평균: {d.mean:.2f}")
print(f"95% CI: {d.tconfint_mean(alpha=0.05)}")
print(f"99% CI: {d.tconfint_mean(alpha=0.01)}")

# 두 표본 비교
group1 = np.random.normal(100, 15, 40)
group2 = np.random.normal(105, 15, 45)

from statsmodels.stats.weightstats import CompareMeans

cm = CompareMeans(DescrStatsW(group1), DescrStatsW(group2))
print(f"\n두 평균 차이의 95% CI: {cm.tconfint_diff(alpha=0.05)}")
```

---

## 연습 문제

### 문제 1: t-신뢰구간
다음 표본에서 모평균의 95% 신뢰구간을 구하시오.
```python
sample = [23, 25, 28, 22, 26, 24, 27, 25, 29, 24]
```

### 문제 2: 비율의 신뢰구간
500명을 대상으로 설문조사한 결과 230명이 찬성했습니다.
- (a) 모비율의 95% 신뢰구간을 구하시오 (정규근사)
- (b) Wilson 방법으로 다시 계산하시오
- (c) 표본 크기를 2000명으로 늘리면 구간 폭이 어떻게 변하는가?

### 문제 3: 부트스트랩
다음 비대칭 분포에서 평균과 중앙값의 부트스트랩 95% 신뢰구간을 각각 구하시오.
```python
np.random.seed(42)
data = np.random.exponential(10, 40)
```

---

## 정리

| 신뢰구간 유형 | 조건 | 공식 | Python |
|--------------|------|------|--------|
| 평균 (σ 기지) | Z-구간 | x̄ ± z·σ/√n | `stats.norm.interval()` |
| 평균 (σ 미지) | t-구간 | x̄ ± t·s/√n | `stats.t.interval()` |
| 비율 | np ≥ 10 | p̂ ± z·√(p̂(1-p̂)/n) | `proportion_confint()` |
| 분산 | 정규모집단 | χ² 분포 이용 | 직접 계산 |
| 부트스트랩 | 비모수적 | 재표본추출 | `stats.bootstrap()` |
