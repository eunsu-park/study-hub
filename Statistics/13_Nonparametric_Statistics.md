# 13. 비모수 통계 (Nonparametric Statistics)

## 개요

비모수 통계는 모집단의 분포에 대한 가정 없이 데이터를 분석하는 방법입니다. 정규성을 만족하지 않거나 표본 크기가 작을 때 유용합니다.

---

## 1. 비모수 검정이 필요한 경우

### 1.1 언제 비모수 검정을 사용하는가?

```python
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

np.random.seed(42)

def when_to_use_nonparametric():
    """비모수 검정 사용 상황"""
    print("""
    ================================================
    비모수 검정을 사용해야 하는 경우
    ================================================

    1. 정규성 위반
       - 데이터가 정규분포를 따르지 않음
       - Shapiro-Wilk 검정 등에서 기각

    2. 작은 표본 크기
       - n < 30인 경우 CLT 적용 어려움
       - 정규성 검정도 검정력 부족

    3. 순위/서열 데이터
       - 리커트 척도 (1~5점)
       - 순위 데이터

    4. 이상치 존재
       - 극단값에 민감한 평균 대신 중위수 비교
       - 순위 기반 방법은 이상치에 강건

    5. 동질성 가정 위반
       - 분산 동질성 (등분산) 가정 위반

    ================================================
    비모수 검정의 장단점
    ================================================

    장점:
    - 분포 가정 불필요
    - 이상치에 강건
    - 순위 데이터에 적합

    단점:
    - 가정이 만족되면 모수적 검정보다 검정력 낮음
    - 효과 크기 해석이 어려울 수 있음
    - 일부 복잡한 설계에 적용 어려움
    """)

when_to_use_nonparametric()

# 정규성 검정 예시
def check_normality(data, alpha=0.05):
    """정규성 검정 수행"""
    # Shapiro-Wilk
    stat_sw, p_sw = stats.shapiro(data)

    # D'Agostino-Pearson
    if len(data) >= 20:
        stat_da, p_da = stats.normaltest(data)
    else:
        stat_da, p_da = np.nan, np.nan

    print("=== 정규성 검정 ===")
    print(f"표본 크기: {len(data)}")
    print(f"Shapiro-Wilk: W={stat_sw:.4f}, p={p_sw:.4f}")
    if not np.isnan(p_da):
        print(f"D'Agostino-Pearson: p={p_da:.4f}")

    is_normal = p_sw > alpha
    print(f"결론: {'정규분포로 볼 수 있음' if is_normal else '정규분포 아님'} (α={alpha})")

    return is_normal

# 정규 데이터
normal_data = np.random.normal(50, 10, 30)
check_normality(normal_data)

print()

# 비정규 데이터 (지수분포)
skewed_data = np.random.exponential(10, 30)
check_normality(skewed_data)
```

### 1.2 모수적 vs 비모수적 검정 대응

| 모수적 검정 | 비모수적 검정 | 상황 |
|------------|--------------|------|
| 1-표본 t-검정 | Wilcoxon 부호순위 검정 | 단일 표본, 중위수 검정 |
| 독립표본 t-검정 | Mann-Whitney U | 두 독립 표본 비교 |
| 대응표본 t-검정 | Wilcoxon 부호순위 검정 | 대응 표본 비교 |
| 일원 ANOVA | Kruskal-Wallis H | 3개 이상 독립 표본 |
| 반복측정 ANOVA | Friedman | 3개 이상 대응 표본 |
| Pearson 상관 | Spearman/Kendall | 상관관계 |

---

## 2. Mann-Whitney U 검정

### 2.1 개념

**목적**: 두 독립 표본의 분포 비교 (중위수 또는 분포 위치)

**가설**:
- H₀: 두 집단의 분포가 동일
- H₁: 두 집단의 분포가 다름 (또는 한쪽이 확률적으로 더 큼)

**검정 통계량**: U (순위 합 기반)

```python
def mann_whitney_example():
    """Mann-Whitney U 검정 예시"""
    np.random.seed(42)

    # 시나리오: 두 그룹의 처리 효과 비교 (정규성 위반)
    # 그룹 A: 대조군
    # 그룹 B: 처리군

    group_a = np.random.exponential(20, 25)  # 비정규
    group_b = np.random.exponential(25, 25) + 5  # 비정규, 더 큰 값

    print("=== Mann-Whitney U 검정 ===")
    print(f"\n그룹 A: n={len(group_a)}, 중위수={np.median(group_a):.2f}")
    print(f"그룹 B: n={len(group_b)}, 중위수={np.median(group_b):.2f}")

    # 정규성 검정
    print("\n정규성 검정:")
    _, p_a = stats.shapiro(group_a)
    _, p_b = stats.shapiro(group_b)
    print(f"  그룹 A: p={p_a:.4f}")
    print(f"  그룹 B: p={p_b:.4f}")

    # Mann-Whitney U 검정
    statistic, p_value = stats.mannwhitneyu(group_a, group_b, alternative='two-sided')

    print(f"\nMann-Whitney U 검정:")
    print(f"  U 통계량: {statistic:.2f}")
    print(f"  p-value: {p_value:.4f}")

    if p_value < 0.05:
        print("  결론: 두 그룹 간 유의한 차이 있음 (p < 0.05)")
    else:
        print("  결론: 두 그룹 간 유의한 차이 없음 (p >= 0.05)")

    # 효과 크기: rank-biserial correlation
    n1, n2 = len(group_a), len(group_b)
    r = 1 - (2 * statistic) / (n1 * n2)
    print(f"\n효과 크기 (rank-biserial r): {r:.3f}")
    print(f"  |r| < 0.1: 작은 효과")
    print(f"  0.1 <= |r| < 0.3: 중간 효과")
    print(f"  |r| >= 0.3: 큰 효과")

    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 박스플롯
    ax = axes[0]
    ax.boxplot([group_a, group_b], labels=['Group A', 'Group B'])
    ax.set_ylabel('값')
    ax.set_title('박스플롯')
    ax.grid(True, alpha=0.3)

    # 히스토그램
    ax = axes[1]
    ax.hist(group_a, bins=15, alpha=0.5, label='Group A', density=True)
    ax.hist(group_b, bins=15, alpha=0.5, label='Group B', density=True)
    ax.axvline(np.median(group_a), color='blue', linestyle='--', label='A 중위수')
    ax.axvline(np.median(group_b), color='orange', linestyle='--', label='B 중위수')
    ax.set_xlabel('값')
    ax.set_ylabel('밀도')
    ax.set_title('분포 비교')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 순위 분포
    ax = axes[2]
    all_data = np.concatenate([group_a, group_b])
    ranks = stats.rankdata(all_data)
    ranks_a = ranks[:len(group_a)]
    ranks_b = ranks[len(group_a):]
    ax.hist(ranks_a, bins=15, alpha=0.5, label='Group A ranks')
    ax.hist(ranks_b, bins=15, alpha=0.5, label='Group B ranks')
    ax.set_xlabel('순위')
    ax.set_ylabel('빈도')
    ax.set_title('순위 분포')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return group_a, group_b

group_a, group_b = mann_whitney_example()
```

### 2.2 단측 검정

```python
# 단측 검정: 그룹 B가 그룹 A보다 큰지 검정
stat_greater, p_greater = stats.mannwhitneyu(group_a, group_b, alternative='less')
print(f"단측 검정 (B > A): p = {p_greater:.4f}")

stat_less, p_less = stats.mannwhitneyu(group_a, group_b, alternative='greater')
print(f"단측 검정 (A > B): p = {p_less:.4f}")
```

---

## 3. Wilcoxon 부호순위 검정

### 3.1 대응표본 비교

**목적**: 대응되는 두 측정값의 차이가 0인지 검정

**사용 상황**: 전/후 비교, 짝지어진 표본

```python
def wilcoxon_signed_rank_example():
    """Wilcoxon 부호순위 검정 예시"""
    np.random.seed(42)

    # 시나리오: 다이어트 프로그램 전후 체중 변화
    n = 20
    before = np.random.normal(80, 10, n)
    # 평균 3kg 감소 효과 + 비정규 오차
    after = before - 3 + np.random.exponential(2, n) - np.random.exponential(2, n)

    diff = after - before

    print("=== Wilcoxon 부호순위 검정 ===")
    print(f"\n표본 크기: {n}")
    print(f"전: 평균={before.mean():.2f}, 중위수={np.median(before):.2f}")
    print(f"후: 평균={after.mean():.2f}, 중위수={np.median(after):.2f}")
    print(f"차이: 평균={diff.mean():.2f}, 중위수={np.median(diff):.2f}")

    # 차이의 정규성 검정
    _, p_norm = stats.shapiro(diff)
    print(f"\n차이의 정규성: p={p_norm:.4f}")

    # Wilcoxon 검정
    statistic, p_value = stats.wilcoxon(before, after, alternative='two-sided')

    print(f"\nWilcoxon 검정:")
    print(f"  W 통계량: {statistic:.2f}")
    print(f"  p-value: {p_value:.4f}")

    # 대응표본 t-검정과 비교
    t_stat, t_p = stats.ttest_rel(before, after)
    print(f"\n대응 t-검정 (비교용):")
    print(f"  t 통계량: {t_stat:.2f}")
    print(f"  p-value: {t_p:.4f}")

    # 효과 크기
    r = statistic / (n * (n + 1) / 2)
    print(f"\n효과 크기 (r): {abs(1-2*r):.3f}")

    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 전후 비교
    ax = axes[0]
    ax.boxplot([before, after], labels=['Before', 'After'])
    ax.set_ylabel('체중 (kg)')
    ax.set_title('전후 비교')
    ax.grid(True, alpha=0.3)

    # 개인별 변화
    ax = axes[1]
    for i in range(n):
        ax.plot([0, 1], [before[i], after[i]], 'b-', alpha=0.5)
    ax.plot([0, 1], [before.mean(), after.mean()], 'r-', linewidth=2, label='평균')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Before', 'After'])
    ax.set_ylabel('체중 (kg)')
    ax.set_title('개인별 변화')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 차이 분포
    ax = axes[2]
    ax.hist(diff, bins=10, density=True, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='r', linestyle='--', label='변화 없음')
    ax.axvline(np.median(diff), color='g', linestyle='-', label=f'중위수={np.median(diff):.2f}')
    ax.set_xlabel('차이 (After - Before)')
    ax.set_ylabel('밀도')
    ax.set_title('차이 분포')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return before, after

before, after = wilcoxon_signed_rank_example()
```

### 3.2 단일 표본 검정 (중위수 검정)

```python
# 단일 표본: 중위수가 특정 값인지 검정
def one_sample_wilcoxon(data, hypothesized_median):
    """
    단일 표본 Wilcoxon 검정
    H0: 중위수 = hypothesized_median
    """
    diff_from_median = data - hypothesized_median

    # 0인 값 제외
    diff_from_median = diff_from_median[diff_from_median != 0]

    if len(diff_from_median) == 0:
        print("모든 값이 가설 중위수와 같습니다.")
        return

    stat, p_value = stats.wilcoxon(diff_from_median)

    print(f"=== 단일 표본 Wilcoxon 검정 ===")
    print(f"H0: 중위수 = {hypothesized_median}")
    print(f"표본 중위수: {np.median(data):.2f}")
    print(f"W 통계량: {stat:.2f}")
    print(f"p-value: {p_value:.4f}")

    if p_value < 0.05:
        print(f"결론: 중위수는 {hypothesized_median}과 유의하게 다름")
    else:
        print(f"결론: 중위수가 {hypothesized_median}이라는 증거 부족")

# 예시
sample_data = np.random.exponential(10, 30) + 5
one_sample_wilcoxon(sample_data, hypothesized_median=10)
```

---

## 4. Kruskal-Wallis H 검정

### 4.1 개념

**목적**: 3개 이상 독립 그룹의 분포 비교 (일원 ANOVA의 비모수적 대안)

**가설**:
- H₀: 모든 그룹의 분포가 동일
- H₁: 최소 하나의 그룹이 다름

```python
def kruskal_wallis_example():
    """Kruskal-Wallis H 검정 예시"""
    np.random.seed(42)

    # 시나리오: 3가지 교육 방법의 효과 비교
    method_a = np.random.exponential(10, 25) + 60  # 전통적
    method_b = np.random.exponential(10, 25) + 65  # 온라인
    method_c = np.random.exponential(10, 25) + 70  # 혼합

    print("=== Kruskal-Wallis H 검정 ===")
    print(f"\n방법 A: n={len(method_a)}, 중위수={np.median(method_a):.2f}")
    print(f"방법 B: n={len(method_b)}, 중위수={np.median(method_b):.2f}")
    print(f"방법 C: n={len(method_c)}, 중위수={np.median(method_c):.2f}")

    # 정규성 검정
    print("\n정규성 검정:")
    for name, data in [('A', method_a), ('B', method_b), ('C', method_c)]:
        _, p = stats.shapiro(data)
        print(f"  {name}: p={p:.4f}")

    # Kruskal-Wallis 검정
    H_stat, p_value = stats.kruskal(method_a, method_b, method_c)

    print(f"\nKruskal-Wallis 검정:")
    print(f"  H 통계량: {H_stat:.2f}")
    print(f"  p-value: {p_value:.4f}")

    # 효과 크기: eta-squared
    N = len(method_a) + len(method_b) + len(method_c)
    k = 3
    eta_sq = (H_stat - k + 1) / (N - k)
    print(f"\n효과 크기 (η²): {eta_sq:.3f}")
    print("  0.01: 작은 효과")
    print("  0.06: 중간 효과")
    print("  0.14: 큰 효과")

    # 일원 ANOVA와 비교
    F_stat, anova_p = stats.f_oneway(method_a, method_b, method_c)
    print(f"\n일원 ANOVA (비교용):")
    print(f"  F 통계량: {F_stat:.2f}")
    print(f"  p-value: {anova_p:.4f}")

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 박스플롯
    ax = axes[0]
    ax.boxplot([method_a, method_b, method_c],
               labels=['Method A', 'Method B', 'Method C'])
    ax.set_ylabel('점수')
    ax.set_title('교육 방법별 점수 분포')
    ax.grid(True, alpha=0.3)

    # 바이올린 플롯
    ax = axes[1]
    parts = ax.violinplot([method_a, method_b, method_c], positions=[1, 2, 3],
                           showmeans=True, showmedians=True)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Method A', 'Method B', 'Method C'])
    ax.set_ylabel('점수')
    ax.set_title('바이올린 플롯')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return method_a, method_b, method_c

method_a, method_b, method_c = kruskal_wallis_example()
```

### 4.2 사후검정 (Post-hoc Tests)

```python
from itertools import combinations
from scipy.stats import mannwhitneyu

def dunn_test_alternative(groups, group_names, alpha=0.05):
    """
    Dunn 검정의 대안: Bonferroni 보정된 Mann-Whitney U 검정
    """
    n_comparisons = len(list(combinations(range(len(groups)), 2)))
    adjusted_alpha = alpha / n_comparisons

    print(f"=== 사후검정 (Bonferroni 보정) ===")
    print(f"비교 횟수: {n_comparisons}")
    print(f"보정된 α: {adjusted_alpha:.4f}")
    print()

    results = []
    for (i, j) in combinations(range(len(groups)), 2):
        stat, p = mannwhitneyu(groups[i], groups[j], alternative='two-sided')
        significant = p < adjusted_alpha
        results.append({
            'comparison': f'{group_names[i]} vs {group_names[j]}',
            'U': stat,
            'p-value': p,
            'significant': significant
        })
        print(f"{group_names[i]} vs {group_names[j]}: U={stat:.1f}, p={p:.4f} {'*' if significant else ''}")

    return pd.DataFrame(results)

groups = [method_a, method_b, method_c]
group_names = ['A', 'B', 'C']
posthoc_results = dunn_test_alternative(groups, group_names)
```

---

## 5. Friedman 검정

### 5.1 개념

**목적**: 반복측정 또는 블록 설계에서 3개 이상 조건 비교

**사용 상황**: 같은 피험자가 여러 조건에서 측정됨

```python
def friedman_example():
    """Friedman 검정 예시"""
    np.random.seed(42)

    # 시나리오: 같은 학생들의 3개 시험 점수
    n_students = 20

    # 상관된 데이터 생성 (같은 학생의 여러 시험)
    ability = np.random.normal(70, 10, n_students)  # 기저 능력
    exam1 = ability + np.random.normal(0, 5, n_students)
    exam2 = ability + np.random.normal(3, 5, n_students)  # 약간 더 어려움
    exam3 = ability + np.random.normal(6, 5, n_students)  # 더 어려움

    print("=== Friedman 검정 ===")
    print(f"\n학생 수: {n_students}")
    print(f"Exam 1: 중위수={np.median(exam1):.2f}")
    print(f"Exam 2: 중위수={np.median(exam2):.2f}")
    print(f"Exam 3: 중위수={np.median(exam3):.2f}")

    # Friedman 검정
    stat, p_value = stats.friedmanchisquare(exam1, exam2, exam3)

    print(f"\nFriedman 검정:")
    print(f"  χ² 통계량: {stat:.2f}")
    print(f"  p-value: {p_value:.4f}")

    # 효과 크기: Kendall's W
    k = 3  # 조건 수
    W = stat / (n_students * (k - 1))
    print(f"\n효과 크기 (Kendall's W): {W:.3f}")
    print("  W < 0.3: 약한 일치")
    print("  0.3 <= W < 0.5: 중간 일치")
    print("  W >= 0.5: 강한 일치")

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 박스플롯
    ax = axes[0]
    ax.boxplot([exam1, exam2, exam3], labels=['Exam 1', 'Exam 2', 'Exam 3'])
    ax.set_ylabel('점수')
    ax.set_title('시험별 점수 분포')
    ax.grid(True, alpha=0.3)

    # 개인별 프로파일
    ax = axes[1]
    for i in range(min(10, n_students)):  # 처음 10명만
        ax.plot([1, 2, 3], [exam1[i], exam2[i], exam3[i]], 'o-', alpha=0.5)
    ax.plot([1, 2, 3], [exam1.mean(), exam2.mean(), exam3.mean()],
            'rs-', linewidth=2, markersize=8, label='평균')
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Exam 1', 'Exam 2', 'Exam 3'])
    ax.set_ylabel('점수')
    ax.set_title('개인별 프로파일')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return exam1, exam2, exam3

exam1, exam2, exam3 = friedman_example()
```

### 5.2 Nemenyi 사후검정

```python
def nemenyi_posthoc(groups, group_names, alpha=0.05):
    """
    Nemenyi 사후검정 (Friedman 이후)
    Wilcoxon 부호순위 검정의 Bonferroni 보정
    """
    n_comparisons = len(list(combinations(range(len(groups)), 2)))
    adjusted_alpha = alpha / n_comparisons

    print(f"=== Nemenyi 사후검정 ===")
    print(f"비교 횟수: {n_comparisons}")
    print(f"보정된 α: {adjusted_alpha:.4f}")
    print()

    results = []
    for (i, j) in combinations(range(len(groups)), 2):
        stat, p = stats.wilcoxon(groups[i], groups[j])
        significant = p < adjusted_alpha
        results.append({
            'comparison': f'{group_names[i]} vs {group_names[j]}',
            'W': stat,
            'p-value': p,
            'significant': significant
        })
        print(f"{group_names[i]} vs {group_names[j]}: W={stat:.1f}, p={p:.4f} {'*' if significant else ''}")

    return pd.DataFrame(results)

groups = [exam1, exam2, exam3]
group_names = ['Exam1', 'Exam2', 'Exam3']
nemenyi_results = nemenyi_posthoc(groups, group_names)
```

---

## 6. 비모수적 상관 (Spearman, Kendall)

### 6.1 Spearman 순위 상관

**특징**: 순위 기반, 단조 관계 측정 (선형일 필요 없음)

```python
def spearman_correlation_example():
    """Spearman 상관 예시"""
    np.random.seed(42)

    # 비선형 관계 데이터
    n = 50
    x = np.random.uniform(0, 10, n)
    y = np.log(x + 1) + np.random.normal(0, 0.3, n)  # 로그 관계 + 잡음

    # Pearson 상관
    pearson_r, pearson_p = stats.pearsonr(x, y)

    # Spearman 상관
    spearman_r, spearman_p = stats.spearmanr(x, y)

    print("=== 상관 분석 ===")
    print(f"\nPearson 상관: r={pearson_r:.4f}, p={pearson_p:.4f}")
    print(f"Spearman 상관: ρ={spearman_r:.4f}, p={spearman_p:.4f}")
    print("\n비선형 관계에서 Spearman이 더 적합")

    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 원본 데이터
    ax = axes[0]
    ax.scatter(x, y, alpha=0.7)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'원본 데이터\nPearson r={pearson_r:.3f}, Spearman ρ={spearman_r:.3f}')
    ax.grid(True, alpha=0.3)

    # 순위 변환
    ax = axes[1]
    x_ranks = stats.rankdata(x)
    y_ranks = stats.rankdata(y)
    ax.scatter(x_ranks, y_ranks, alpha=0.7)
    ax.set_xlabel('X 순위')
    ax.set_ylabel('Y 순위')
    ax.set_title('순위 변환 후')
    ax.grid(True, alpha=0.3)

    # 이상치가 있는 경우
    ax = axes[2]
    x_outlier = np.append(x, [50])  # 극단적 이상치
    y_outlier = np.append(y, [y.mean()])

    pearson_out, _ = stats.pearsonr(x_outlier, y_outlier)
    spearman_out, _ = stats.spearmanr(x_outlier, y_outlier)

    ax.scatter(x_outlier[:-1], y_outlier[:-1], alpha=0.7, label='일반 데이터')
    ax.scatter(x_outlier[-1], y_outlier[-1], color='r', s=100, label='이상치', marker='x')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'이상치 포함\nPearson={pearson_out:.3f}, Spearman={spearman_out:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return x, y

x, y = spearman_correlation_example()
```

### 6.2 Kendall 순위 상관

**특징**: 쌍별 비교 기반, Spearman보다 강건

```python
def kendall_correlation_example():
    """Kendall tau 상관 예시"""
    np.random.seed(42)

    # 순위 데이터 (리커트 척도 등)
    n = 30
    rater1 = np.random.randint(1, 6, n)  # 1~5점
    # 어느 정도 일치하지만 완전히 같지는 않음
    rater2 = np.clip(rater1 + np.random.randint(-1, 2, n), 1, 5)

    # Pearson (부적절)
    pearson_r, _ = stats.pearsonr(rater1, rater2)

    # Spearman
    spearman_r, _ = stats.spearmanr(rater1, rater2)

    # Kendall
    kendall_tau, kendall_p = stats.kendalltau(rater1, rater2)

    print("=== 순위 데이터 상관 분석 ===")
    print(f"\n평가자 1 분포: {np.bincount(rater1)[1:]}")
    print(f"평가자 2 분포: {np.bincount(rater2)[1:]}")
    print(f"\nPearson r: {pearson_r:.4f} (순위 데이터에 부적절)")
    print(f"Spearman ρ: {spearman_r:.4f}")
    print(f"Kendall τ: {kendall_tau:.4f}, p={kendall_p:.4f}")

    print("\n해석:")
    print("  τ = 0: 일치/불일치 쌍이 균등")
    print("  τ = 1: 완전한 순위 일치")
    print("  τ = -1: 완전한 역순위")

    # 시각화
    fig, ax = plt.subplots(figsize=(8, 6))

    # 지터 추가 (같은 점수가 겹치지 않게)
    jitter1 = rater1 + np.random.uniform(-0.1, 0.1, n)
    jitter2 = rater2 + np.random.uniform(-0.1, 0.1, n)

    ax.scatter(jitter1, jitter2, alpha=0.7, s=50)
    ax.plot([0, 6], [0, 6], 'r--', label='완전 일치선')
    ax.set_xlabel('평가자 1')
    ax.set_ylabel('평가자 2')
    ax.set_title(f'평가자 간 일치도\nKendall τ = {kendall_tau:.3f}')
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.5, 5.5)
    ax.set_xticks(range(1, 6))
    ax.set_yticks(range(1, 6))
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.show()

    return rater1, rater2

rater1, rater2 = kendall_correlation_example()
```

### 6.3 상관계수 비교

```python
def compare_correlations():
    """세 가지 상관계수 비교"""
    print("""
    =================================================
    상관계수 비교
    =================================================

    | 특성 | Pearson | Spearman | Kendall |
    |------|---------|----------|---------|
    | 관계 | 선형 | 단조 | 단조 |
    | 데이터 | 연속형 | 순서형/연속형 | 순서형 |
    | 이상치 | 민감 | 강건 | 매우 강건 |
    | 동률 처리 | 해당없음 | 평균 순위 | 보정 가능 |
    | 계산 | O(n) | O(n log n) | O(n²) |
    | 범위 | [-1, 1] | [-1, 1] | [-1, 1] |

    선택 기준:
    - 연속형 & 선형 관계 → Pearson
    - 비정규/이상치/비선형 → Spearman
    - 순위/서열 데이터 → Kendall
    - 작은 표본 크기 → Kendall
    """)

compare_correlations()
```

---

## 7. 실습 예제

### 7.1 종합 비모수 분석

```python
def comprehensive_nonparametric_analysis(data_dict, alpha=0.05):
    """
    종합 비모수 분석 수행

    Parameters:
    -----------
    data_dict : dict
        {그룹명: 데이터} 형태
    """
    print("="*60)
    print("종합 비모수 분석")
    print("="*60)

    groups = list(data_dict.values())
    group_names = list(data_dict.keys())
    n_groups = len(groups)

    # 1. 기술 통계
    print("\n[1] 기술 통계")
    for name, data in data_dict.items():
        print(f"  {name}: n={len(data)}, 중위수={np.median(data):.2f}, "
              f"IQR={np.percentile(data, 75)-np.percentile(data, 25):.2f}")

    # 2. 정규성 검정
    print("\n[2] 정규성 검정 (Shapiro-Wilk)")
    all_normal = True
    for name, data in data_dict.items():
        _, p = stats.shapiro(data)
        is_normal = p > alpha
        all_normal = all_normal and is_normal
        print(f"  {name}: p={p:.4f} {'(정규)' if is_normal else '(비정규)'}")

    # 3. 적절한 검정 선택 및 수행
    print(f"\n[3] 그룹 비교 검정 (그룹 수: {n_groups})")

    if n_groups == 2:
        # 두 그룹: Mann-Whitney U
        stat, p = stats.mannwhitneyu(groups[0], groups[1], alternative='two-sided')
        print(f"  Mann-Whitney U: U={stat:.2f}, p={p:.4f}")
        test_name = "Mann-Whitney U"
    else:
        # 세 그룹 이상: Kruskal-Wallis
        stat, p = stats.kruskal(*groups)
        print(f"  Kruskal-Wallis H: H={stat:.2f}, p={p:.4f}")
        test_name = "Kruskal-Wallis"

        # 유의하면 사후검정
        if p < alpha:
            print("\n  사후검정 (Bonferroni 보정):")
            n_comp = n_groups * (n_groups - 1) // 2
            adj_alpha = alpha / n_comp
            for (i, j) in combinations(range(n_groups), 2):
                _, ph_p = stats.mannwhitneyu(groups[i], groups[j])
                sig = '*' if ph_p < adj_alpha else ''
                print(f"    {group_names[i]} vs {group_names[j]}: p={ph_p:.4f} {sig}")

    # 4. 시각화
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 박스플롯
    ax = axes[0]
    ax.boxplot(groups, labels=group_names)
    ax.set_ylabel('값')
    ax.set_title(f'{test_name} 검정\np = {p:.4f}')
    ax.grid(True, alpha=0.3)

    # 히스토그램
    ax = axes[1]
    for name, data in data_dict.items():
        ax.hist(data, bins=15, alpha=0.5, label=name, density=True)
    ax.set_xlabel('값')
    ax.set_ylabel('밀도')
    ax.set_title('분포 비교')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# 사용 예시
np.random.seed(42)
data = {
    'Control': np.random.exponential(10, 30) + 50,
    'Treatment A': np.random.exponential(10, 30) + 55,
    'Treatment B': np.random.exponential(10, 30) + 60
}
comprehensive_nonparametric_analysis(data)
```

---

## 8. 연습 문제

### 문제 1: 검정 선택
다음 상황에 적합한 비모수 검정을 선택하세요:
1. 남녀 학생의 시험 점수 비교 (점수 분포가 편포)
2. 3개 병원의 대기 시간 비교
3. 동일 환자의 치료 전후 통증 점수 비교

### 문제 2: Mann-Whitney U
두 그룹의 데이터가 주어졌을 때:
- Group A: [23, 28, 31, 35, 39, 42]
- Group B: [18, 22, 25, 29, 33]

수동으로 U 통계량을 계산하고 scipy 결과와 비교하세요.

### 문제 3: 상관 분석
10개의 순위 쌍 데이터에 대해:
1. Spearman 상관계수 계산
2. Kendall tau 계산
3. 두 계수의 차이 해석

### 문제 4: Kruskal-Wallis 후 사후검정
4개 그룹 비교에서 Kruskal-Wallis가 유의할 때:
1. 필요한 사후검정 비교 횟수
2. Bonferroni 보정된 유의수준
3. 어떤 쌍이 유의하게 다른지 판단

---

## 9. 핵심 요약

### 검정 선택 플로우차트

```
데이터 유형 확인
    │
    ├── 정규성 만족? ───┬── Yes → 모수적 검정
    │                   └── No → 비모수 검정
    │
비모수 검정 선택:
    │
    ├── 2 독립 그룹 → Mann-Whitney U
    │
    ├── 2 대응 그룹 → Wilcoxon 부호순위
    │
    ├── 3+ 독립 그룹 → Kruskal-Wallis H → 사후: Dunn
    │
    ├── 3+ 대응 그룹 → Friedman → 사후: Nemenyi
    │
    └── 상관 → Spearman (연속) / Kendall (순위)
```

### scipy.stats 함수

| 검정 | 함수 |
|------|------|
| Mann-Whitney U | `mannwhitneyu(x, y)` |
| Wilcoxon | `wilcoxon(x, y)` |
| Kruskal-Wallis | `kruskal(g1, g2, g3, ...)` |
| Friedman | `friedmanchisquare(g1, g2, g3, ...)` |
| Spearman | `spearmanr(x, y)` |
| Kendall | `kendalltau(x, y)` |

### 효과 크기 해석

| 검정 | 효과 크기 | 작음 | 중간 | 큼 |
|------|----------|------|------|-----|
| Mann-Whitney | rank-biserial r | 0.1 | 0.3 | 0.5 |
| Kruskal-Wallis | η² | 0.01 | 0.06 | 0.14 |
| Friedman | Kendall's W | 0.1 | 0.3 | 0.5 |

### 다음 장 미리보기

14장 **실험 설계**에서는:
- 실험 설계의 기본 원리
- A/B 테스트
- 표본 크기 결정 (검정력 분석)
- 순차적 검정
