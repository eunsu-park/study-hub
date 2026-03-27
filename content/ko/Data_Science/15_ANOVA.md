# 15. 분산분석 (ANOVA - Analysis of Variance)

[이전: 가설검정 심화](./14_Hypothesis_Testing_Advanced.md) | [다음: 회귀분석 심화](./16_Regression_Analysis_Advanced.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 세 개 이상의 그룹 평균을 비교할 때 여러 t-검정 대신 분산분석(ANOVA)을 사용하는 이유를 설명할 수 있다
2. 총 변동을 그룹 간 변동과 그룹 내 변동으로 분해하고 F-통계량을 계산할 수 있다
3. 분산분석(ANOVA) 가정(정규성, 등분산성)을 검증하고, 가정 위반 시 Welch's ANOVA를 적용할 수 있다
4. 사후검정(Post-hoc Tests)(Tukey HSD, Bonferroni, Dunnett, Games-Howell)을 수행하여 어느 그룹 간에 차이가 있는지 식별할 수 있다
5. 이원배치 분산분석(Two-way ANOVA)을 구현하여 두 요인의 주효과(Main Effect)와 상호작용 효과(Interaction Effect)를 분석할 수 있다
6. 반복측정 분산분석(Repeated Measures ANOVA)과 혼합 설계(Mixed-design ANOVA)를 피험자 내 설계 및 혼합 설계에 적용할 수 있다
7. 비모수 대안(Kruskal-Wallis, Friedman)의 사용 시점을 파악하고 분산분석 효과크기(Effect Size)를 계산할 수 있다

---

> **신호 대 잡음(Signal vs. Noise).** 시끄러운 방에서 세 명의 발표자가 하는 말을 들으려 한다고 상상해 보세요. 분산분석(ANOVA)은 이렇게 묻습니다: "발표자들 사이의 차이(그룹 간 분산)가 배경 소음(그룹 내 분산)보다 더 크게 들리는가?" 신호 대 잡음비(signal-to-noise ratio)가 충분히 높다면, 발표자들이 실제로 서로 다른 말을 하고 있다고 결론 내릴 수 있습니다 — 단순히 임의의 잡음에 묻혀버리는 것이 아니라.

두 그룹 이상을 비교할 때 쌍별 t-검정을 반복하면 제1종 오류율(Type I error rate)이 빠르게 높아집니다. 분산분석(ANOVA)은 변동을 의미 있는 출처로 분해하는 우아한 단일 검정 체계를 제공합니다 — 관심 있는 신호와 통제할 수 없는 잡음으로 구분하는 것입니다. 분산분석(ANOVA)과 그 확장 기법을 익히면 여러 요인과 반복 측정을 갖춘 실험을 설계하고 분석할 수 있는 능력을 갖추게 됩니다.

---

## 1. 일원배치 분산분석 (One-way ANOVA)

### 1.1 기본 개념

```python
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
import statsmodels.api as sm
from statsmodels.formula.api import ols

# ANOVA의 기본 아이디어:
# 총 변동 = 그룹 간 변동 + 그룹 내 변동
# H₀: μ₁ = μ₂ = ... = μₖ (모든 그룹 평균이 같다)
# H₁: 적어도 하나의 그룹 평균이 다르다

# 예시 데이터 생성
np.random.seed(42)

# 세 가지 교수법의 효과 비교
method_A = np.random.normal(75, 10, 25)  # 전통적 교수법
method_B = np.random.normal(80, 10, 25)  # 토론식 교수법
method_C = np.random.normal(85, 10, 25)  # 플립러닝

# DataFrame 생성
df = pd.DataFrame({
    'score': np.concatenate([method_A, method_B, method_C]),
    'method': ['A']*25 + ['B']*25 + ['C']*25
})

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Box plot
sns.boxplot(data=df, x='method', y='score', ax=axes[0], palette='Set2')
axes[0].set_xlabel('교수법')
axes[0].set_ylabel('점수')
axes[0].set_title('교수법별 점수 분포')

# 각 그룹의 평균과 개별 점수
for i, method in enumerate(['A', 'B', 'C']):
    data = df[df['method'] == method]['score']
    axes[1].scatter([i]*len(data), data, alpha=0.5, s=30)
    axes[1].scatter([i], [data.mean()], color='red', s=100, marker='D', zorder=5)

axes[1].set_xticks([0, 1, 2])
axes[1].set_xticklabels(['A', 'B', 'C'])
axes[1].set_xlabel('교수법')
axes[1].set_ylabel('점수')
axes[1].set_title('개별 점수와 그룹 평균')
axes[1].axhline(df['score'].mean(), color='blue', linestyle='--', label='전체 평균')
axes[1].legend()

plt.tight_layout()
plt.show()

print(f"전체 평균: {df['score'].mean():.2f}")
print(f"그룹별 평균:")
print(df.groupby('method')['score'].mean())
```

### 1.2 ANOVA 계산

```python
def one_way_anova_manual(groups):
    """일원배치 ANOVA 수동 계산"""
    # 전체 데이터
    all_data = np.concatenate(groups)
    n_total = len(all_data)
    k = len(groups)  # 그룹 수
    grand_mean = all_data.mean()

    # 그룹 간 변동 (SSB: Sum of Squares Between)
    SSB = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)

    # 그룹 내 변동 (SSW: Sum of Squares Within)
    SSW = sum(sum((x - g.mean())**2 for x in g) for g in groups)

    # 총 변동 (SST: Sum of Squares Total)
    SST = sum((x - grand_mean)**2 for x in all_data)

    # 자유도
    df_between = k - 1
    df_within = n_total - k
    df_total = n_total - 1

    # 평균제곱 (Mean Squares)
    MSB = SSB / df_between
    MSW = SSW / df_within

    # F-통계량
    F_stat = MSB / MSW

    # p-value
    p_value = 1 - stats.f.cdf(F_stat, df_between, df_within)

    return {
        'SSB': SSB, 'SSW': SSW, 'SST': SST,
        'df_between': df_between, 'df_within': df_within,
        'MSB': MSB, 'MSW': MSW,
        'F_stat': F_stat, 'p_value': p_value
    }

# 계산
groups = [method_A, method_B, method_C]
result = one_way_anova_manual(groups)

print("일원배치 ANOVA 결과 (수동 계산):")
print("=" * 60)
print("분산분석표:")
print("-" * 60)
print(f"{'변동원':<12} {'SS':<12} {'df':<8} {'MS':<12} {'F':<10} {'p-value':<10}")
print("-" * 60)
print(f"{'처리(그룹간)':<12} {result['SSB']:<12.2f} {result['df_between']:<8} "
      f"{result['MSB']:<12.2f} {result['F_stat']:<10.3f} {result['p_value']:<10.4f}")
print(f"{'오차(그룹내)':<12} {result['SSW']:<12.2f} {result['df_within']:<8} {result['MSW']:<12.2f}")
print(f"{'총계':<12} {result['SST']:<12.2f} {result['df_between'] + result['df_within']:<8}")
print("=" * 60)
```

### 1.3 scipy와 statsmodels 활용

```python
# scipy.stats.f_oneway
F_scipy, p_scipy = stats.f_oneway(method_A, method_B, method_C)
print(f"scipy.stats.f_oneway:")
print(f"  F = {F_scipy:.3f}, p = {p_scipy:.4f}")

# statsmodels OLS를 이용한 ANOVA
model = ols('score ~ C(method)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

print(f"\nstatsmodels ANOVA Table:")
print(anova_table)

# pingouin
anova_pg = pg.anova(data=df, dv='score', between='method', detailed=True)
print(f"\npingouin ANOVA:")
print(anova_pg)
```

### 1.4 F-분포 이해

```python
# F-분포: 두 카이제곱 변수의 비율
# F = (χ²₁/df₁) / (χ²₂/df₂)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# F-분포 (다양한 자유도)
x = np.linspace(0, 5, 200)
df_pairs = [(2, 20), (5, 20), (10, 20), (5, 50)]

for df1, df2 in df_pairs:
    axes[0].plot(x, stats.f.pdf(x, df1, df2), label=f'F({df1},{df2})')

axes[0].set_xlabel('F')
axes[0].set_ylabel('밀도')
axes[0].set_title('F-분포 (다양한 자유도)')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 현재 분석의 F-분포와 관찰된 F-값
df1, df2 = result['df_between'], result['df_within']
x = np.linspace(0, 8, 200)
y = stats.f.pdf(x, df1, df2)

axes[1].plot(x, y, 'b-', lw=2, label=f'F({df1},{df2})')
axes[1].fill_between(x, y, where=(x >= result['F_stat']), alpha=0.3, color='red',
                      label=f'p-value = {result["p_value"]:.4f}')
axes[1].axvline(result['F_stat'], color='red', linestyle='--',
                label=f'F = {result["F_stat"]:.2f}')

# 임계값
f_critical = stats.f.ppf(0.95, df1, df2)
axes[1].axvline(f_critical, color='green', linestyle=':',
                label=f'F_crit (α=0.05) = {f_critical:.2f}')

axes[1].set_xlabel('F')
axes[1].set_ylabel('밀도')
axes[1].set_title('F-분포와 검정 결과')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 2. ANOVA 가정 확인

### 2.1 정규성 검정

```python
# 각 그룹의 정규성 검정

print("정규성 검정 (Shapiro-Wilk):")
print("-" * 40)

for method in ['A', 'B', 'C']:
    data = df[df['method'] == method]['score']
    stat, p = stats.shapiro(data)
    print(f"그룹 {method}: W = {stat:.4f}, p = {p:.4f}")

# 잔차의 정규성 (더 중요)
residuals = model.resid
stat, p = stats.shapiro(residuals)
print(f"\n잔차 정규성: W = {stat:.4f}, p = {p:.4f}")

# Q-Q plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 그룹별 Q-Q plot
for i, method in enumerate(['A', 'B', 'C']):
    data = df[df['method'] == method]['score']
    stats.probplot(data, dist="norm", plot=axes[0])

axes[0].set_title('그룹별 Q-Q Plot')

# 잔차 Q-Q plot
stats.probplot(residuals, dist="norm", plot=axes[1])
axes[1].set_title('잔차 Q-Q Plot')

plt.tight_layout()
plt.show()
```

### 2.2 등분산성 검정

```python
# Levene's test (중앙값 기반, 더 로버스트)
stat_levene, p_levene = stats.levene(method_A, method_B, method_C)
print(f"Levene's Test: F = {stat_levene:.4f}, p = {p_levene:.4f}")

# Bartlett's test (정규성 가정)
stat_bartlett, p_bartlett = stats.bartlett(method_A, method_B, method_C)
print(f"Bartlett's Test: χ² = {stat_bartlett:.4f}, p = {p_bartlett:.4f}")

# pingouin
homoscedasticity_result = pg.homoscedasticity(df, dv='score', group='method')
print(f"\npingouin 등분산성 검정:")
print(homoscedasticity_result)

# 잔차 vs 적합값 그림
fig, ax = plt.subplots(figsize=(10, 5))

fitted = model.fittedvalues
residuals = model.resid

ax.scatter(fitted, residuals, alpha=0.6)
ax.axhline(0, color='red', linestyle='--')
ax.set_xlabel('적합값')
ax.set_ylabel('잔차')
ax.set_title('잔차 vs 적합값 (등분산성 확인)')

# 각 그룹의 분산 표시
for method in ['A', 'B', 'C']:
    data = df[df['method'] == method]
    mean_val = data['score'].mean()
    ax.axvline(mean_val, color='gray', linestyle=':', alpha=0.5)

plt.show()

print("\n그룹별 표준편차:")
print(df.groupby('method')['score'].std())
```

### 2.3 등분산 가정 위반 시: Welch's ANOVA

```python
# 등분산 가정이 위반된 경우
np.random.seed(42)

# 분산이 다른 그룹들
group1 = np.random.normal(50, 5, 30)   # σ = 5
group2 = np.random.normal(55, 15, 30)  # σ = 15
group3 = np.random.normal(60, 25, 30)  # σ = 25

df_hetero = pd.DataFrame({
    'value': np.concatenate([group1, group2, group3]),
    'group': ['G1']*30 + ['G2']*30 + ['G3']*30
})

# 등분산성 검정
stat, p = stats.levene(group1, group2, group3)
print(f"Levene's Test: p = {p:.4f} (등분산 가정 위반)")

# 일반 ANOVA
F_normal, p_normal = stats.f_oneway(group1, group2, group3)
print(f"\n일반 ANOVA: F = {F_normal:.3f}, p = {p_normal:.4f}")

# Welch's ANOVA (pingouin)
welch_result = pg.welch_anova(data=df_hetero, dv='value', between='group')
print(f"\nWelch's ANOVA:")
print(welch_result)

# Games-Howell 사후검정 (등분산 가정 X)
games_howell = pg.pairwise_gameshowell(data=df_hetero, dv='value', between='group')
print(f"\nGames-Howell 사후검정:")
print(games_howell)
```

---

## 3. 사후검정 (Post-hoc Tests)

### 3.1 Tukey HSD

```python
# ANOVA가 유의하면 → 어떤 그룹이 다른지 확인
# Tukey's Honest Significant Difference

from statsmodels.stats.multicomp import pairwise_tukeyhsd

# 원본 데이터로
tukey_result = pairwise_tukeyhsd(df['score'], df['method'], alpha=0.05)

print("Tukey HSD 사후검정:")
print(tukey_result)

# 시각화
fig = tukey_result.plot_simultaneous(figsize=(10, 4))
plt.title('Tukey HSD: 95% 신뢰구간')
plt.xlabel('점수')
plt.tight_layout()
plt.show()

# pingouin 사용
tukey_pg = pg.pairwise_tukey(data=df, dv='score', between='method')
print("\npingouin Tukey HSD:")
print(tukey_pg)
```

### 3.2 다양한 사후검정 방법

```python
# 다양한 사후검정 비교

print("다양한 사후검정 방법 비교:")
print("=" * 70)

# 1. Bonferroni
bonf = pg.pairwise_tests(data=df, dv='score', between='method',
                          padjust='bonf', effsize='cohen')
print("\n1. Bonferroni:")
print(bonf[['A', 'B', 'T', 'p-unc', 'p-corr', 'cohen']])

# 2. Holm
holm = pg.pairwise_tests(data=df, dv='score', between='method',
                          padjust='holm', effsize='cohen')
print("\n2. Holm:")
print(holm[['A', 'B', 'p-unc', 'p-corr']])

# 3. Scheffe (더 보수적)
# statsmodels에서 직접 계산 필요

# 4. Dunnett (대조군 비교)
# A를 대조군으로 설정
from scipy.stats import dunnett

dunnett_result = dunnett(method_B, method_C, control=method_A)
print("\n4. Dunnett (A가 대조군):")
print(f"  B vs A: statistic={dunnett_result.statistic[0]:.3f}, p={dunnett_result.pvalue[0]:.4f}")
print(f"  C vs A: statistic={dunnett_result.statistic[1]:.3f}, p={dunnett_result.pvalue[1]:.4f}")
```

### 3.3 효과크기

```python
# ANOVA 효과크기

# Eta-squared (η²)
ss_between = result['SSB']
ss_total = result['SST']
eta_squared = ss_between / ss_total

# Omega-squared (ω²) - 편향 보정
ms_within = result['MSW']
n_total = len(df)
k = 3
omega_squared = (ss_between - (k-1)*ms_within) / (ss_total + ms_within)

# Partial eta-squared (이 경우 eta-squared와 동일)
partial_eta_squared = ss_between / (ss_between + result['SSW'])

print("ANOVA 효과크기:")
print("-" * 40)
print(f"η² (Eta-squared): {eta_squared:.4f}")
print(f"ω² (Omega-squared): {omega_squared:.4f}")
print(f"Partial η²: {partial_eta_squared:.4f}")

print("\n해석 기준 (Cohen):")
print("  η² ≈ 0.01: 작은 효과")
print("  η² ≈ 0.06: 중간 효과")
print("  η² ≈ 0.14: 큰 효과")

# pingouin 결과에서 효과크기
print(f"\npingouin 결과의 η²: {anova_pg['np2'].values[0]:.4f}")
```

---

## 4. 이원배치 분산분석 (Two-way ANOVA)

### 4.1 기본 개념

```python
# 두 가지 독립변수(요인)의 효과를 동시에 분석
# 주효과(Main Effect) + 상호작용(Interaction)

np.random.seed(42)

# 예시: 교수법(A, B) × 학습시간(Low, High)의 효과
n_per_cell = 20

data = {
    'score': [],
    'method': [],
    'time': []
}

# 2x2 설계
# 상호작용 효과 포함
effects = {
    ('A', 'Low'): 70,
    ('A', 'High'): 80,
    ('B', 'Low'): 75,
    ('B', 'High'): 90,  # B + High의 시너지 효과
}

for (method, time), mean in effects.items():
    scores = np.random.normal(mean, 8, n_per_cell)
    data['score'].extend(scores)
    data['method'].extend([method] * n_per_cell)
    data['time'].extend([time] * n_per_cell)

df_two = pd.DataFrame(data)

# 셀 평균 확인
cell_means = df_two.groupby(['method', 'time'])['score'].mean().unstack()
print("셀 평균:")
print(cell_means)

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Box plot
sns.boxplot(data=df_two, x='method', y='score', hue='time', ax=axes[0])
axes[0].set_title('교수법 × 학습시간')
axes[0].legend(title='학습시간')

# 상호작용 플롯
for time in ['Low', 'High']:
    means = df_two[df_two['time'] == time].groupby('method')['score'].mean()
    axes[1].plot(['A', 'B'], means.values, 'o-', label=time, markersize=10)

axes[1].set_xlabel('교수법')
axes[1].set_ylabel('평균 점수')
axes[1].set_title('상호작용 플롯')
axes[1].legend(title='학습시간')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

### 4.2 이원 ANOVA 분석

```python
# statsmodels를 이용한 이원 ANOVA
model_two = ols('score ~ C(method) * C(time)', data=df_two).fit()
anova_two = sm.stats.anova_lm(model_two, typ=2)

print("이원배치 ANOVA 결과:")
print(anova_two)

# pingouin
anova_two_pg = pg.anova(data=df_two, dv='score', between=['method', 'time'])
print("\npingouin 이원 ANOVA:")
print(anova_two_pg)

# 효과 해석
print("\n효과 해석:")
print("-" * 50)
for idx, row in anova_two.iterrows():
    if 'Residual' not in idx:
        sig = "***" if row['PR(>F)'] < 0.001 else ("**" if row['PR(>F)'] < 0.01 else ("*" if row['PR(>F)'] < 0.05 else ""))
        print(f"{idx}: F = {row['F']:.2f}, p = {row['PR(>F)']:.4f} {sig}")
```

### 4.3 상호작용 효과 해석

```python
# 상호작용이 유의할 때: 단순 주효과 분석

# 학습시간별 교수법 효과
print("단순 주효과 분석: 학습시간별 교수법 효과")
print("-" * 50)

for time in ['Low', 'High']:
    subset = df_two[df_two['time'] == time]
    t_stat, p_val = stats.ttest_ind(
        subset[subset['method'] == 'A']['score'],
        subset[subset['method'] == 'B']['score']
    )
    print(f"{time} 학습시간: t = {t_stat:.3f}, p = {p_val:.4f}")

# 교수법별 학습시간 효과
print("\n단순 주효과 분석: 교수법별 학습시간 효과")
print("-" * 50)

for method in ['A', 'B']:
    subset = df_two[df_two['method'] == method]
    t_stat, p_val = stats.ttest_ind(
        subset[subset['time'] == 'Low']['score'],
        subset[subset['time'] == 'High']['score']
    )
    d = pg.compute_effsize(
        subset[subset['time'] == 'High']['score'],
        subset[subset['time'] == 'Low']['score'],
        eftype='cohen'
    )
    print(f"교수법 {method}: t = {t_stat:.3f}, p = {p_val:.4f}, d = {d:.3f}")
```

### 4.4 상호작용 없는 경우

```python
# 상호작용이 없는 데이터 생성
np.random.seed(42)

data_no_int = {
    'score': [],
    'method': [],
    'time': []
}

# 상호작용 없음: 각 요인의 효과가 독립적
effects_no_int = {
    ('A', 'Low'): 70,
    ('A', 'High'): 80,   # +10 (시간 효과)
    ('B', 'Low'): 78,    # +8 (방법 효과)
    ('B', 'High'): 88,   # +8 + 10 (가산적)
}

for (method, time), mean in effects_no_int.items():
    scores = np.random.normal(mean, 8, 20)
    data_no_int['score'].extend(scores)
    data_no_int['method'].extend([method] * 20)
    data_no_int['time'].extend([time] * 20)

df_no_int = pd.DataFrame(data_no_int)

# 상호작용 플롯
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 상호작용 있는 경우
for time in ['Low', 'High']:
    means = df_two[df_two['time'] == time].groupby('method')['score'].mean()
    axes[0].plot(['A', 'B'], means.values, 'o-', label=time, markersize=10)
axes[0].set_xlabel('교수법')
axes[0].set_ylabel('평균 점수')
axes[0].set_title('상호작용 있음 (선이 교차/비평행)')
axes[0].legend(title='학습시간')
axes[0].grid(alpha=0.3)

# 상호작용 없는 경우
for time in ['Low', 'High']:
    means = df_no_int[df_no_int['time'] == time].groupby('method')['score'].mean()
    axes[1].plot(['A', 'B'], means.values, 'o-', label=time, markersize=10)
axes[1].set_xlabel('교수법')
axes[1].set_ylabel('평균 점수')
axes[1].set_title('상호작용 없음 (선이 평행)')
axes[1].legend(title='학습시간')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ANOVA 비교
print("상호작용 없는 경우 ANOVA:")
model_no_int = ols('score ~ C(method) * C(time)', data=df_no_int).fit()
print(sm.stats.anova_lm(model_no_int, typ=2))
```

---

## 5. 반복측정 분산분석 (Repeated Measures ANOVA)

### 5.1 기본 개념

```python
# 같은 피험자를 여러 조건에서 측정
# 피험자 내 설계 (within-subjects design)

np.random.seed(42)

# 예시: 동일 피험자가 3가지 약물 조건에서 측정
n_subjects = 20

# 개인차 (피험자 효과)
subject_effect = np.random.normal(0, 10, n_subjects)

# 각 조건의 평균 효과
condition_effects = {'Placebo': 0, 'Drug_A': 5, 'Drug_B': 10}

data_rm = []
for subj in range(n_subjects):
    for condition, effect in condition_effects.items():
        score = 50 + subject_effect[subj] + effect + np.random.normal(0, 5)
        data_rm.append({
            'subject': f'S{subj+1:02d}',
            'condition': condition,
            'score': score
        })

df_rm = pd.DataFrame(data_rm)

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Box plot
sns.boxplot(data=df_rm, x='condition', y='score', ax=axes[0],
            order=['Placebo', 'Drug_A', 'Drug_B'])
axes[0].set_title('조건별 점수 분포')

# 개인별 변화
for subj in df_rm['subject'].unique()[:10]:  # 처음 10명만
    subj_data = df_rm[df_rm['subject'] == subj]
    subj_data = subj_data.set_index('condition').loc[['Placebo', 'Drug_A', 'Drug_B']]
    axes[1].plot(range(3), subj_data['score'].values, 'o-', alpha=0.5)

# 평균 추가
means = df_rm.groupby('condition')['score'].mean()
means = means.loc[['Placebo', 'Drug_A', 'Drug_B']]
axes[1].plot(range(3), means.values, 'rs-', markersize=12, linewidth=3, label='평균')

axes[1].set_xticks(range(3))
axes[1].set_xticklabels(['Placebo', 'Drug_A', 'Drug_B'])
axes[1].set_xlabel('조건')
axes[1].set_ylabel('점수')
axes[1].set_title('개인별 변화 패턴')
axes[1].legend()

plt.tight_layout()
plt.show()
```

### 5.2 반복측정 ANOVA 분석

```python
# pingouin을 이용한 반복측정 ANOVA
rm_anova = pg.rm_anova(data=df_rm, dv='score', within='condition',
                        subject='subject', detailed=True)

print("반복측정 ANOVA 결과:")
print(rm_anova)

# 구형성 검정 (Mauchly's test)
# 반복측정 ANOVA의 가정
print("\n구형성 가정:")
print("  구형성이 위반되면 Greenhouse-Geisser 또는 Huynh-Feldt 보정 사용")
print(f"  GG epsilon: {rm_anova['eps'].values[0]:.4f}")

# 사후검정
posthoc_rm = pg.pairwise_tests(data=df_rm, dv='score', within='condition',
                                subject='subject', padjust='bonf')
print("\n사후검정 (Bonferroni):")
print(posthoc_rm[['A', 'B', 'T', 'p-unc', 'p-corr', 'BF10']])
```

### 5.3 혼합 설계 ANOVA

```python
# 혼합 설계: 피험자 간 요인 + 피험자 내 요인
np.random.seed(42)

# 2 (그룹: 실험/통제) × 3 (시점: 전/중/후) 혼합 설계
n_per_group = 15

data_mixed = []
for group, group_effect in [('Experimental', 5), ('Control', 0)]:
    for subj in range(n_per_group):
        subject_id = f'{group[0]}_{subj+1:02d}'
        base = 50 + np.random.normal(0, 8)

        for time_idx, (time, time_effect) in enumerate([('Pre', 0), ('Mid', 3), ('Post', 6)]):
            # 상호작용: 실험군은 시간에 따른 효과가 더 큼
            interaction = time_idx * 3 if group == 'Experimental' else 0
            score = base + group_effect + time_effect + interaction + np.random.normal(0, 5)

            data_mixed.append({
                'subject': subject_id,
                'group': group,
                'time': time,
                'score': score
            })

df_mixed = pd.DataFrame(data_mixed)

# 시각화
fig, ax = plt.subplots(figsize=(10, 5))

for group in ['Experimental', 'Control']:
    means = df_mixed[df_mixed['group'] == group].groupby('time')['score'].mean()
    sems = df_mixed[df_mixed['group'] == group].groupby('time')['score'].sem()
    means = means.loc[['Pre', 'Mid', 'Post']]
    sems = sems.loc[['Pre', 'Mid', 'Post']]

    ax.errorbar(range(3), means.values, yerr=sems.values, fmt='o-',
                label=group, markersize=10, capsize=5)

ax.set_xticks(range(3))
ax.set_xticklabels(['Pre', 'Mid', 'Post'])
ax.set_xlabel('시점')
ax.set_ylabel('점수')
ax.set_title('혼합 설계: 그룹 × 시점')
ax.legend()
ax.grid(alpha=0.3)
plt.show()

# 혼합 ANOVA
mixed_anova = pg.mixed_anova(data=df_mixed, dv='score', between='group',
                              within='time', subject='subject')
print("혼합 설계 ANOVA:")
print(mixed_anova)
```

---

## 6. 비모수 대안

### 6.1 Kruskal-Wallis 검정

```python
# 정규성 가정을 충족하지 못할 때

# 비정규 데이터 생성
np.random.seed(42)
group1_nonnorm = np.random.exponential(10, 30)
group2_nonnorm = np.random.exponential(15, 30)
group3_nonnorm = np.random.exponential(20, 30)

# 정규성 검정
print("정규성 검정:")
for i, g in enumerate([group1_nonnorm, group2_nonnorm, group3_nonnorm], 1):
    stat, p = stats.shapiro(g)
    print(f"  그룹 {i}: p = {p:.4f}")

# Kruskal-Wallis 검정
H_stat, p_kw = stats.kruskal(group1_nonnorm, group2_nonnorm, group3_nonnorm)
print(f"\nKruskal-Wallis 검정:")
print(f"  H = {H_stat:.3f}, p = {p_kw:.4f}")

# 비교: 일원 ANOVA
F_stat, p_anova = stats.f_oneway(group1_nonnorm, group2_nonnorm, group3_nonnorm)
print(f"\n일원 ANOVA (비교용):")
print(f"  F = {F_stat:.3f}, p = {p_anova:.4f}")

# 사후검정: Dunn's test
df_nonnorm = pd.DataFrame({
    'value': np.concatenate([group1_nonnorm, group2_nonnorm, group3_nonnorm]),
    'group': ['G1']*30 + ['G2']*30 + ['G3']*30
})

dunn_result = pg.pairwise_tests(data=df_nonnorm, dv='value', between='group',
                                 parametric=False, padjust='bonf')
print("\nDunn's 사후검정:")
print(dunn_result[['A', 'B', 'U-val', 'p-unc', 'p-corr']])
```

### 6.2 Friedman 검정

```python
# 반복측정의 비모수 대안

# 비정규 반복측정 데이터
np.random.seed(42)
n_subjects = 20

cond1 = np.random.exponential(10, n_subjects)
cond2 = np.random.exponential(15, n_subjects) + cond1 * 0.3  # 상관 있는 측정
cond3 = np.random.exponential(20, n_subjects) + cond1 * 0.3

# Friedman 검정
chi2_stat, p_friedman = stats.friedmanchisquare(cond1, cond2, cond3)
print(f"Friedman 검정:")
print(f"  χ² = {chi2_stat:.3f}, p = {p_friedman:.4f}")

# 효과크기: Kendall's W
n_conditions = 3
W = chi2_stat / (n_subjects * (n_conditions - 1))
print(f"  Kendall's W = {W:.4f}")
```

---

## 연습 문제

### 문제 1: 일원 ANOVA
세 가지 비료(A, B, C)가 식물 성장에 미치는 효과를 분석하시오.
```python
fertilizer_A = [20, 22, 19, 24, 25, 23, 21, 22, 26, 24]
fertilizer_B = [28, 30, 27, 29, 31, 28, 30, 29, 32, 31]
fertilizer_C = [25, 27, 26, 28, 24, 26, 27, 25, 29, 26]
```

### 문제 2: 이원 ANOVA
성별(남/여) × 학습방법(온라인/오프라인)의 효과를 분석하시오.
- 주효과와 상호작용 효과를 해석하시오.
- 상호작용 플롯을 그리시오.

### 문제 3: 사후검정
문제 1의 결과가 유의할 경우, Tukey HSD 사후검정을 수행하고 해석하시오.

---

## 정리

| ANOVA 유형 | 설계 | Python 함수 |
|------------|------|-------------|
| 일원배치 | 1 요인, k 수준 | `stats.f_oneway()`, `pg.anova()` |
| 이원배치 | 2 요인 | `ols() + anova_lm()` |
| 반복측정 | 피험자 내 요인 | `pg.rm_anova()` |
| 혼합설계 | 피험자 간+내 | `pg.mixed_anova()` |
| Kruskal-Wallis | 비모수 일원 | `stats.kruskal()` |
| Friedman | 비모수 반복측정 | `stats.friedmanchisquare()` |

| 사후검정 | 특징 | 함수 |
|----------|------|------|
| Tukey HSD | 모든 쌍 비교 | `pairwise_tukeyhsd()` |
| Bonferroni | 보수적 | `pg.pairwise_tests(padjust='bonf')` |
| Dunnett | 대조군 비교 | `stats.dunnett()` |
| Games-Howell | 등분산 X | `pg.pairwise_gameshowell()` |
