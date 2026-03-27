# 탐색적 자료 분석에서 통계적 추론으로

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 일반적인 결론을 도출하는 데 있어 기술통계(descriptive statistics)와 탐색적 자료 분석(EDA)의 한계를 설명할 수 있습니다
2. 모집단(population)과 표본(sample)을 구분하고, 표본 변동성(sampling variability)이 통계적 추론을 필요로 하는 이유를 설명할 수 있습니다
3. 추정(estimation), 가설 검정(hypothesis testing), 예측(prediction), 연관성(association) 등 다양한 통계적 질문 유형을 구분할 수 있습니다
4. 데이터 유형과 연구 질문에 따라 적절한 통계 방법을 선택하기 위한 의사결정 가이드를 적용할 수 있습니다
5. 탐색적 분석에서 확증적 가설 검정까지 이어지는 전체 워크플로우(end-to-end workflow)를 구현할 수 있습니다
6. p-해킹(p-hacking), 상관관계와 인과관계의 혼동, 가정 무시 등 흔한 함정을 식별할 수 있습니다
7. 효과 크기(effect size)를 활용하여 통계적 유의성(statistical significance)과 실용적 유의성(practical significance)을 구분할 수 있습니다

---

이전 레슨들에서 데이터 조작(data manipulation), 시각화(visualization), 기술통계(descriptive statistics) 등 데이터셋에서 무슨 일이 일어나고 있는지 탐색하는 강력한 도구들을 익혔습니다. 그러나 탐색만으로는 관찰된 패턴이 실제인지, 아니면 단순히 무작위 오차의 산물인지 알 수 없으며, 분석 결과가 현재 데이터 너머로 일반화될 수 있는지도 알 수 없습니다. 이 레슨은 탐색적 분석에서 공식적인 통계적 추론(statistical inference)으로 넘어가는 가교 역할을 하며, 데이터로부터 엄밀하고 방어 가능한 결론을 도출하는 데 필요한 사고 체계를 제공합니다.

---

## 1. 서론: "그냥 보는 것"의 한계

이전 강의에서 우리는 **탐색적 자료 분석(Exploratory Data Analysis, EDA)**을 위한 강력한 도구들을 배웠습니다:
- Pandas를 활용한 데이터 조작
- Matplotlib과 Seaborn을 사용한 시각화
- 기술통계 (평균, 중앙값, 표준편차)
- 패턴 탐지 및 이상치 식별

하지만 EDA만으로는 중요한 질문에 답할 수 없습니다:
- "이 차이가 **진짜**인가, 아니면 그냥 무작위 노이즈인가?"
- "이 결과를 데이터셋을 넘어서 **일반화**할 수 있는가?"
- "우리의 결론에 대해 얼마나 **확신**할 수 있는가?"
- "미래 관측값에 대해 무엇을 **예측**할 수 있는가?"

여기서 **통계적 추론(Statistical Inference)**이 필요합니다.

### 탐정 비유

데이터 과학을 탐정 업무라고 생각해 봅시다:
- **EDA** = 단서 수집, 범죄 현장 조사, 가설 형성
- **통계적 추론** = 그 가설을 엄격하게 검증하고, 법정용 증거 구축

EDA는 *데이터에서 무슨 일이 일어났는지*를 말해줍니다. 추론은 *그것이 데이터 너머의 세계에 대해 무엇을 의미하는지*를 말해줍니다.

---

## 2. 모집단 대 표본: 왜 추론이 필요한가

### 핵심 개념

- **모집단(Population)**: 우리가 연구하고자 하는 모든 개체/항목의 완전한 집합
  - 예: 전자상거래 플랫폼의 *모든* 고객
  - 예: 물리 상수의 *모든* 가능한 측정값

- **표본(Sample)**: 실제로 관측하는 모집단의 부분집합
  - 예: 데이터베이스에서 10,000명의 고객
  - 예: 실험에서 100개의 측정값

- **표본 변동성(Sampling Variability)**: 같은 모집단의 서로 다른 표본은 서로 다른 결과를 제공

### 왜 표본 통계량을 그대로 사용할 수 없나요?

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 모집단 시뮬레이션: 평균 구매액 $50, 표준편차 $15인 100만 고객
np.random.seed(42)
population = np.random.normal(loc=50, scale=15, size=1_000_000)
true_mean = population.mean()
print(f"True population mean: ${true_mean:.2f}")

# 크기 100인 5개의 서로 다른 표본 추출
sample_means = []
for i in range(5):
    sample = np.random.choice(population, size=100, replace=False)
    sample_mean = sample.mean()
    sample_means.append(sample_mean)
    print(f"Sample {i+1} mean: ${sample_mean:.2f}")

print(f"\nRange of sample means: ${min(sample_means):.2f} to ${max(sample_means):.2f}")
```

**출력:**
```
True population mean: $50.00
Sample 1 mean: $48.93
Sample 2 mean: $51.24
Sample 3 mean: $49.67
Sample 4 mean: $50.82
Sample 5 mean: $49.15

Range of sample means: $48.93 to $51.24
```

**핵심 통찰**: 각 표본마다 *다른* 추정값을 제공합니다! 통계적 추론은 다음을 도와줍니다:
1. 이 불확실성을 정량화
2. 모집단에 대한 확률적 진술 작성
3. 통제된 오류율로 가설 검정

---

## 3. 통계적 사고의 전환

### 기술통계에서 추론통계로

| **기술통계(EDA)** | **추론통계** |
|----------------------------------|----------------------------|
| "표본 평균은 50.2입니다" | "모집단 평균은 48.5에서 51.9 사이일 가능성이 높습니다 (95% CI)" |
| "그룹 A가 더 높은 평균을 가집니다" | "그룹 A의 평균이 유의하게 더 높습니다 (p < 0.01)" |
| "변수 X와 Y가 0.7의 상관관계를 가집니다" | "모집단 상관관계는 양수입니다 (p < 0.001)" |
| "이 패턴이 우리 데이터에 나타납니다" | "이 패턴은 표본을 넘어 일반화됩니다 (AIC 비교)" |

### 추론 마인드셋

EDA에서 추론으로 넘어갈 때 다음을 질문하세요:

1. **관심 모집단이 무엇인가?**
   - 단순히 "내 데이터셋"이 아니라 더 넓은 맥락

2. **표본은 어떻게 얻었는가?**
   - 무작위 표집? 편의 표본? 이것이 타당성에 영향을 미침

3. **어떤 가정을 하고 있는가?**
   - 정규성? 독립성? 분산의 동질성?

4. **불확실성이 무엇인가?**
   - 신뢰구간, p-값, 신용구간

5. **틀렸을 때의 실제적 결과는 무엇인가?**
   - 제1종 오류 대 제2종 오류, 효과 크기

---

## 4. 통계적 질문의 유형

### 4.1 추정 질문

**질문**: "모집단 모수의 값은 무엇인가?"

**예시**:
- 고객 생애가치의 평균은 무엇인가?
- 사용자 중 버튼을 클릭하는 비율은?

**도구**: 신뢰구간(Confidence Intervals), 점추정(Point Estimates)

```python
# 예시: 신뢰구간으로 평균 고객 지출 추정
sample = np.random.choice(population, size=100, replace=False)
sample_mean = sample.mean()
sample_se = stats.sem(sample)  # Standard error of the mean
ci_95 = stats.t.interval(0.95, len(sample)-1, loc=sample_mean, scale=sample_se)

print(f"Sample mean: ${sample_mean:.2f}")
print(f"95% Confidence Interval: ${ci_95[0]:.2f} to ${ci_95[1]:.2f}")
print(f"Interpretation: We are 95% confident the true population mean is in this range")
```

### 4.2 가설 검정 질문

**질문**: "유의한 차이/효과가 있는가?"

**예시**:
- 처치 A가 처치 B보다 더 잘 작동하는가?
- 웹사이트 재디자인이 전환율을 증가시켰는가?

**도구**: t-검정, 카이제곱 검정, ANOVA, 순열 검정

```python
# 예시: A/B 테스트 - 새 디자인이 전환율을 증가시켰는가?
# 통제 그룹 (구 디자인)
control_conversions = np.random.binomial(1, 0.10, size=1000)  # 10% conversion
# 처치 그룹 (신 디자인)
treatment_conversions = np.random.binomial(1, 0.12, size=1000)  # 12% conversion

# 가설 검정
from statsmodels.stats.proportion import proportions_ztest

count = np.array([treatment_conversions.sum(), control_conversions.sum()])
nobs = np.array([len(treatment_conversions), len(control_conversions)])

z_stat, p_value = proportions_ztest(count, nobs)
print(f"Treatment conversion: {treatment_conversions.mean():.3f}")
print(f"Control conversion: {control_conversions.mean():.3f}")
print(f"p-value: {p_value:.4f}")
print(f"Result: {'Significant' if p_value < 0.05 else 'Not significant'} at α=0.05")
```

### 4.3 예측 질문

**질문**: "새로운 관측값에 대해 무엇이 일어날 것인가?"

**예시**:
- 이 고객이 다음 달에 무엇을 지출할 것인가?
- 몇 개의 유닛을 판매할 것인가?

**도구**: 회귀, 시계열 모델, 기계 학습

### 4.4 연관성 질문

**질문**: "변수들이 어떻게 관련되어 있는가?"

**예시**:
- 교육 수준이 소득과 상관관계가 있는가?
- 변수들이 독립적인가 아니면 종속적인가?

**도구**: 상관관계, 회귀, 분할표

---

## 5. 언제 어떤 방법을 사용할지: 결정 가이드

### 5.1 데이터 유형 기반

```
┌─── 결과 변수의 유형은? ───┐
│                                       │
│  연속형 (수치형)                      │
│  ├─ 한 그룹 → 일표본 t-검정          │
│  ├─ 두 그룹 → 이표본 t-검정          │
│  ├─ 3개 이상 그룹 → ANOVA            │
│  └─ 예측 변수들 → 회귀               │
│                                       │
│  범주형 (이항/개수)                   │
│  ├─ 한 비율 → 비율 검정              │
│  ├─ 두 비율 → 카이제곱               │
│  └─ 예측 변수들 → 로지스틱           │
│                                       │
│  시계열                               │
│  └─ 시간적 패턴 → ARIMA 등           │
└───────────────────────────────────────┘
```

### 5.2 연구 질문 기반

```python
def suggest_test(data_type, num_groups, paired=False, question_type="difference"):
    """
    통계 검정 선택을 위한 간단한 결정 트리

    Parameters:
    -----------
    data_type : str
        'continuous' 또는 'categorical'
    num_groups : int
        비교할 그룹 수
    paired : bool
        관측값이 대응되는가?
    question_type : str
        'difference', 'association', 'prediction'
    """

    if question_type == "association":
        if data_type == "continuous":
            return "Pearson/Spearman correlation, Linear regression"
        else:
            return "Chi-square test of independence, Odds ratio"

    if question_type == "prediction":
        return "Regression (linear/logistic), Machine learning"

    # For difference questions
    if data_type == "continuous":
        if num_groups == 1:
            return "One-sample t-test"
        elif num_groups == 2:
            if paired:
                return "Paired t-test"
            else:
                return "Independent two-sample t-test (or Mann-Whitney if not normal)"
        else:
            return "One-way ANOVA (or Kruskal-Wallis if not normal)"
    else:  # categorical
        if num_groups == 1:
            return "One-proportion z-test, Binomial test"
        elif num_groups == 2:
            return "Two-proportion z-test, Chi-square test"
        else:
            return "Chi-square test for multiple groups"

# Examples
print(suggest_test('continuous', 2, paired=False))
# → Independent two-sample t-test (or Mann-Whitney if not normal)

print(suggest_test('categorical', 2, question_type='difference'))
# → Two-proportion z-test, Chi-square test

print(suggest_test('continuous', 1, question_type='association'))
# → Pearson/Spearman correlation, Linear regression
```

---

## 6. EDA를 추론과 연결하기

### 워크플로: 탐색에서 확증으로

```python
import pandas as pd
import seaborn as sns

# 단계 1: 탐색적 - 데이터 불러오고 시각화
np.random.seed(42)
data = pd.DataFrame({
    'group': ['A']*50 + ['B']*50,
    'score': np.concatenate([
        np.random.normal(75, 10, 50),  # Group A
        np.random.normal(80, 10, 50)   # Group B
    ])
})

# EDA: 차이 시각화
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.boxplot(data=data, x='group', y='score')
plt.title('EDA: Boxplot suggests Group B scores higher')

plt.subplot(1, 2, 2)
sns.histplot(data=data, x='score', hue='group', kde=True)
plt.title('EDA: Distributions appear roughly normal')

plt.tight_layout()
plt.savefig('/tmp/eda_to_inference.png', dpi=100, bbox_inches='tight')
plt.close()

# EDA: 기술통계
print("=== EXPLORATORY PHASE ===")
print(data.groupby('group')['score'].describe())
print("\nObservation: Group B has a higher mean (80.5 vs 74.9)")
print("Question: Is this difference statistically significant?\n")

# 단계 2: 확증적 - 가설 검정
print("=== INFERENTIAL PHASE ===")

# 가정 확인
group_a = data[data['group'] == 'A']['score']
group_b = data[data['group'] == 'B']['score']

# 정규성 검정
_, p_norm_a = stats.shapiro(group_a)
_, p_norm_b = stats.shapiro(group_b)
print(f"Normality test (Shapiro-Wilk):")
print(f"  Group A: p={p_norm_a:.3f} → {'Normal' if p_norm_a > 0.05 else 'Not normal'}")
print(f"  Group B: p={p_norm_b:.3f} → {'Normal' if p_norm_b > 0.05 else 'Not normal'}")

# 분산 동질성 검정
_, p_var = stats.levene(group_a, group_b)
print(f"\nEqual variance test (Levene):")
print(f"  p={p_var:.3f} → {'Equal variances' if p_var > 0.05 else 'Unequal variances'}")

# 이표본 t-검정
t_stat, p_value = stats.ttest_ind(group_a, group_b)
print(f"\nTwo-sample t-test:")
print(f"  t-statistic: {t_stat:.3f}")
print(f"  p-value: {p_value:.4f}")
print(f"  Result: {'Reject H0' if p_value < 0.05 else 'Fail to reject H0'} at α=0.05")

# 효과 크기 (Cohen's d)
pooled_std = np.sqrt((group_a.std()**2 + group_b.std()**2) / 2)
cohens_d = (group_b.mean() - group_a.mean()) / pooled_std
print(f"  Cohen's d: {cohens_d:.3f} ({'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'} effect)")

# 차이에 대한 신뢰구간
diff_mean = group_b.mean() - group_a.mean()
se_diff = np.sqrt(group_a.var()/len(group_a) + group_b.var()/len(group_b))
ci_diff = stats.t.interval(0.95, len(group_a)+len(group_b)-2, loc=diff_mean, scale=se_diff)
print(f"  95% CI for difference: ({ci_diff[0]:.2f}, {ci_diff[1]:.2f})")

print("\n=== CONCLUSION ===")
print(f"Group B scores are significantly higher than Group A (p={p_value:.4f}).")
print(f"The mean difference is {diff_mean:.2f} points (95% CI: {ci_diff[0]:.2f} to {ci_diff[1]:.2f}).")
print(f"This represents a {('small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large')} effect size.")
```

**핵심 요점**: EDA는 추론 전략을 안내합니다:
- 히스토그램 → 정규성 가정 확인
- 박스플롯 → 적절한 검정 식별 (모수적 대 비모수적)
- 산점도 → 회귀 선택 정보 제공
- 결측 데이터 패턴 → 추론 전 처리

---

## 7. 통계적 추론에서 흔한 함정

### 7.1 p-해킹 (데이터 드레징, Data Dredging)

**문제**: p < 0.05를 찾을 때까지 많은 가설을 검정

```python
# 나쁜 관행: 보정 없이 많은 변수 검정
np.random.seed(123)
num_tests = 20
p_values = []

for i in range(num_tests):
    # 무작위 데이터 생성 (진짜 효과 없음)
    group1 = np.random.normal(0, 1, 30)
    group2 = np.random.normal(0, 1, 30)
    _, p = stats.ttest_ind(group1, group2)
    p_values.append(p)
    if p < 0.05:
        print(f"Test {i+1}: p={p:.4f} 🎉 Significant!")

print(f"\nFound {sum(p < 0.05 for p in p_values)} 'significant' results out of {num_tests} tests")
print("But all data was random! This is Type I error (false positive)")
```

**해결책**:
- 다중 검정 보정 사용 (Bonferroni, Benjamini-Hochberg)
- 가설 사전 등록
- 수행된 모든 검정 보고

```python
from statsmodels.stats.multitest import multipletests

# 다중 비교 보정
rejected, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')
print(f"\nAfter Bonferroni correction: {sum(rejected)} significant results")
```

### 7.2 상관관계와 인과관계 혼동

**문제**: "X와 Y가 상관관계를 가지므로, X가 Y를 유발한다"

```python
# 허위 상관관계 예시
np.random.seed(42)
years = np.arange(2000, 2020)
ice_cream_sales = 100 + 2*years + np.random.normal(0, 50, len(years)) - 4000
drowning_deaths = 50 + 1*years + np.random.normal(0, 20, len(years)) - 2000

corr, p_corr = stats.pearsonr(ice_cream_sales, drowning_deaths)
print(f"Correlation between ice cream sales and drowning deaths: r={corr:.3f}, p={p_corr:.4f}")
print("Conclusion: Ice cream causes drowning? NO!")
print("Explanation: Both are caused by a confounding variable (summer/temperature)")
```

**기억하세요**:
- 상관관계 ≠ 인과관계
- 인과적 주장을 위해서는 실험 설계 (무작위화, 통제) 필요
- 교란 변수, 역 인과관계, 제3 변수 고려

### 7.3 가정 무시

**문제**: 가정을 확인하지 않고 검정 사용

```python
# 예시: 심하게 치우친 데이터에 t-검정 사용
np.random.seed(42)
skewed_data1 = np.random.exponential(scale=2, size=30)
skewed_data2 = np.random.exponential(scale=2.5, size=30)

# 잘못된 방법: 정규성 확인 없이 t-검정 사용
t_stat, p_ttest = stats.ttest_ind(skewed_data1, skewed_data2)
print(f"t-test p-value: {p_ttest:.4f}")

# 올바른 방법: 먼저 가정 확인
_, p_norm = stats.shapiro(skewed_data1)
print(f"Shapiro-Wilk test for normality: p={p_norm:.4f}")
if p_norm < 0.05:
    print("Data is not normal! Use Mann-Whitney U test instead")
    u_stat, p_mann = stats.mannwhitneyu(skewed_data1, skewed_data2)
    print(f"Mann-Whitney U test p-value: {p_mann:.4f}")
```

### 7.4 통계적 유의성과 실제적 유의성 혼동

**문제**: "p < 0.05이므로 중요하다!"

```python
# 큰 표본은 작은 효과를 "유의하게" 만들 수 있음
np.random.seed(42)
large_group1 = np.random.normal(100, 15, 10000)
large_group2 = np.random.normal(100.5, 15, 10000)  # Tiny difference

t_stat, p_value = stats.ttest_ind(large_group1, large_group2)
cohens_d = (large_group2.mean() - large_group1.mean()) / large_group1.std()

print(f"Mean difference: {large_group2.mean() - large_group1.mean():.3f}")
print(f"p-value: {p_value:.4f} → Statistically significant!")
print(f"Cohen's d: {cohens_d:.3f} → Practically negligible (tiny effect)")
print("\nAlways report effect sizes, not just p-values!")
```

---

## 8. 실전 예시: EDA에서 완전한 추론까지

### 시나리오: 전자상거래 A/B 테스트

새로운 결제 플로우가 구매액을 증가시키는지 알고 싶습니다.

```python
# 현실적인 데이터 생성
np.random.seed(42)
n = 200

data_ab = pd.DataFrame({
    'user_id': range(n),
    'variant': ['control']*100 + ['treatment']*100,
    'purchase_amount': np.concatenate([
        np.random.gamma(shape=2, scale=25, size=100),  # Control
        np.random.gamma(shape=2.3, scale=25, size=100)  # Treatment (slightly higher)
    ])
})

# 교란 변수 추가: 사용자 재직 기간
data_ab['tenure_months'] = np.random.poisson(lam=12, size=n)

print("=== STAGE 1: EXPLORATORY DATA ANALYSIS ===\n")

# 1. 요약 통계
print(data_ab.groupby('variant')['purchase_amount'].describe())

# 2. 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 히스토그램
data_ab.hist(column='purchase_amount', by='variant', bins=20, ax=axes[0:2], alpha=0.7)
axes[0].set_title('Control Group')
axes[1].set_title('Treatment Group')

# 박스플롯
axes[2].boxplot([
    data_ab[data_ab['variant']=='control']['purchase_amount'],
    data_ab[data_ab['variant']=='treatment']['purchase_amount']
], labels=['Control', 'Treatment'])
axes[2].set_ylabel('Purchase Amount')
axes[2].set_title('Distribution Comparison')

plt.tight_layout()
plt.savefig('/tmp/ab_test_eda.png', dpi=100, bbox_inches='tight')
plt.close()

print("\n=== STAGE 2: FORMULATE STATISTICAL QUESTION ===\n")
print("Research Question: Does the new checkout flow (treatment) increase purchase amounts?")
print("Null Hypothesis (H0): μ_treatment = μ_control")
print("Alternative Hypothesis (H1): μ_treatment > μ_control (one-tailed)")
print("Significance level: α = 0.05")

print("\n=== STAGE 3: CHECK ASSUMPTIONS ===\n")

control = data_ab[data_ab['variant']=='control']['purchase_amount']
treatment = data_ab[data_ab['variant']=='treatment']['purchase_amount']

# 정규성 (주의: n=100이면 중심극한정리 적용, 하지만 확인해보자)
_, p_norm_c = stats.shapiro(control)
_, p_norm_t = stats.shapiro(treatment)
print(f"Normality (Shapiro-Wilk):")
print(f"  Control: p={p_norm_c:.4f}")
print(f"  Treatment: p={p_norm_t:.4f}")
print(f"  → Data is {'normal' if min(p_norm_c, p_norm_t) > 0.05 else 'not normal, but n=100 so CLT applies'}")

# 분산 동질성
_, p_var = stats.levene(control, treatment)
print(f"\nEqual variance (Levene): p={p_var:.4f}")
print(f"  → Variances are {'equal' if p_var > 0.05 else 'unequal'}")

print("\n=== STAGE 4: CONDUCT INFERENCE ===\n")

# 이표본 t-검정 (단측)
t_stat, p_twotail = stats.ttest_ind(treatment, control, equal_var=(p_var>0.05))
p_onetail = p_twotail / 2 if t_stat > 0 else 1 - p_twotail / 2

print(f"Two-sample t-test (one-tailed):")
print(f"  t-statistic: {t_stat:.3f}")
print(f"  p-value: {p_onetail:.4f}")
print(f"  Decision: {'Reject H0' if p_onetail < 0.05 else 'Fail to reject H0'}")

# 효과 크기
cohens_d = (treatment.mean() - control.mean()) / np.sqrt((control.var() + treatment.var())/2)
print(f"\nEffect size (Cohen's d): {cohens_d:.3f}")

# 신뢰구간
diff_mean = treatment.mean() - control.mean()
se_diff = np.sqrt(control.var()/len(control) + treatment.var()/len(treatment))
ci_95 = stats.t.interval(0.95, len(control)+len(treatment)-2, loc=diff_mean, scale=se_diff)
print(f"95% CI for difference: (${ci_95[0]:.2f}, ${ci_95[1]:.2f})")

# 실제적 유의성
revenue_increase = (diff_mean / control.mean()) * 100
print(f"\nRevenue increase: {revenue_increase:.1f}%")

print("\n=== STAGE 5: REPORT RESULTS ===\n")
print(f"The treatment group had significantly higher purchase amounts than the control group")
print(f"(M_treatment = ${treatment.mean():.2f}, M_control = ${control.mean():.2f}, t({len(control)+len(treatment)-2}) = {t_stat:.2f}, p = {p_onetail:.4f}).")
print(f"The mean difference was ${diff_mean:.2f} (95% CI: ${ci_95[0]:.2f} to ${ci_95[1]:.2f}),")
print(f"representing a {revenue_increase:.1f}% increase in average purchase amount.")
print(f"The effect size was {('small' if abs(cohens_d)<0.5 else 'medium' if abs(cohens_d)<0.8 else 'large')} (Cohen's d = {cohens_d:.2f}).")
```

---

## 9. 연습 문제

### 연습 문제 1: 올바른 검정 선택

각 시나리오에 대해 적절한 통계 검정을 식별하세요:

a) 한 회사가 세 서비스 센터 간 고객 만족도 점수(1-10)가 다른지 알고 싶어 합니다.

b) 한 연구자가 남성과 여성 간 왼손잡이의 비율이 다른지 검정하고 싶어 합니다.

c) 한 데이터 과학자가 평방 피트, 침실 수, 위치를 기반으로 주택 가격을 예측하고 싶어 합니다.

d) 한 분석가가 공부 시간과 시험 점수 간 관계가 있는지 알고 싶어 합니다.

**답변**:
- a) 일원 분산분석(One-way ANOVA) (연속형 결과, 3개 그룹)
- b) 이표본 비율 z-검정 또는 카이제곱 검정 (범주형 결과, 2개 그룹)
- c) 다중 선형 회귀(Multiple Linear Regression) (연속형 결과, 다중 예측 변수)
- d) 피어슨 상관관계 / 단순 선형 회귀 (두 연속형 변수)

### 연습 문제 2: EDA에서 가설로

직원 데이터에 대한 EDA를 수행하고 다음을 발견했습니다:
- 부서 A의 중앙값 급여는 $75,000입니다
- 부서 B의 중앙값 급여는 $82,000입니다
- 박스플롯에서 일부 겹침이 보이지만 부서 B가 더 높아 보입니다

**질문**:
1. 관심 모집단은 무엇인가?
2. 귀무가설과 대립가설을 공식화하세요
3. 어떤 검정을 사용하겠습니까? 어떤 가정을 확인하겠습니까?
4. p = 0.03이면 무엇을 결론 내리겠습니까?
5. 실제적 유의성을 평가하는 데 어떤 추가 정보가 도움이 될까요?

### 연습 문제 3: 함정 식별

각 시나리오에서 통계적 함정을 식별하세요:

a) 한 연구자가 50개의 서로 다른 변수를 검정하고 p < 0.05인 3개만 보고합니다.

b) 한 연구가 커피 소비가 심장병과 상관관계가 있다는 것을 발견하고 커피가 심장병을 유발한다고 결론 내립니다.

c) 한 회사가 "통계적으로 유의한 개선"(p=0.04)을 보고하지만 실제 전환율 증가는 0.1%였습니다.

d) 한 분석가가 심하게 오른쪽으로 치우친 소득 데이터에 가정을 확인하지 않고 t-검정을 사용합니다.

**답변**:
- a) p-해킹 / 다중 비교 문제
- b) 상관관계와 인과관계 혼동
- c) 통계적 유의성과 실제적 유의성 혼동
- d) 검정 가정 위반

### 연습 문제 4: 완전한 분석 파이프라인

tips 데이터셋(seaborn에서 사용 가능)을 사용하여 완전한 분석을 수행하세요:

```python
import seaborn as sns
tips = sns.load_dataset('tips')

# Your task:
# 1. EDA: 흡연자가 비흡연자와 다르게 팁을 주는지 탐색
# 2. 가설 공식화
# 3. 가정 확인
# 4. 적절한 검정 수행
# 5. 효과 크기와 신뢰구간으로 결과 보고
```

---

## 10. 요약

### 핵심 요점

1. **EDA는 탐색; 추론은 확증**
   - EDA는 가설을 생성; 추론은 이를 엄격하게 검증

2. **표본은 모집단에 대한 불완전한 창**
   - 표본 변동성은 확률적 진술이 필요함을 의미
   - 신뢰구간과 p-값은 불확실성을 정량화

3. **다른 질문은 다른 방법이 필요**
   - 추정 → 신뢰구간
   - 검정 → 가설 검정 (t-검정, ANOVA, 카이제곱 등)
   - 예측 → 회귀, 머신러닝 모델
   - 연관성 → 상관관계, 분할표

4. **항상 가정을 확인**
   - 정규성, 독립성, 분산 동질성
   - 가정이 실패할 때 강건한 대안 사용

5. **p-값뿐만 아니라 효과 크기 보고**
   - 통계적 유의성 ≠ 실제적 유의성
   - 해석에는 맥락이 중요

6. **흔한 함정 주의**
   - 보정 없는 다중 검정
   - 상관관계 ≠ 인과관계
   - 가정 위반
   - p-값 과잉 해석

### 건넌 다리

이제 여러분은 다음을 이해합니다:
- ✅ 왜 추론 없이 "데이터를 신뢰"할 수 없는가
- ✅ 기술적 패턴에서 정식 통계 질문으로 이동하는 방법
- ✅ 언제 어떤 통계 방법을 사용할지
- ✅ EDA 결과를 엄격한 검정과 연결하는 방법
- ✅ 피해야 할 흔한 실수

### 다음은?

나머지 강의는 다음을 더 깊이 다룹니다:
- **L11-L13**: 확률 기초와 분포
- **L14-L16**: 가설 검정 프레임워크와 검정력 분석
- **L17-L18**: 회귀와 모델 평가
- **L19-L21**: 베이지안 추론
- **L22-L24**: 시계열과 고급 주제

이제 "데이터가 보여주는 것"을 넘어 "확신을 가지고 결론 내릴 수 있는 것"으로 나아갈 준비가 되었습니다.

---

## 11. 추가 자료

### 책
- **"The Art of Statistics"** by David Spiegelhalter - 통계적 사고에 대한 접근하기 쉬운 소개
- **"Statistical Rethinking"** by Richard McElreath - 직관적 예시를 통한 베이지안 접근
- **"Naked Statistics"** by Charles Wheelan - 무거운 수학 없이 개념적 이해

### 온라인 자료
- [Seeing Theory](https://seeing-theory.brown.edu/) - 확률과 통계에 대한 시각적 소개
- [StatQuest](https://statquest.org/) - 통계 개념에 대한 비디오 설명
- [Cross Validated](https://stats.stackexchange.com/) - 통계 Q&A

### Python 라이브러리
- **scipy.stats**: 통계 검정 및 분포
- **statsmodels**: 회귀, 가설 검정, 시계열
- **pingouin**: 효과 크기가 포함된 사용자 친화적 통계 검정
- **scikit-learn**: 기계 학습 및 예측 모델링

---

## 내비게이션
- [이전: 09_Data_Visualization_Advanced](./09_Data_Visualization_Advanced.md)
- [다음: 11_Probability_Review](./11_Probability_Review.md)
- [개요: 00_Overview](./00_Overview.md)
