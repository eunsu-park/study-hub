# 7. 기술통계와 EDA (탐색적 데이터 분석)

[이전: 데이터 전처리](./06_Data_Preprocessing.md) | [다음: 데이터 시각화 기초](./08_Data_Visualization_Basics.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 평균(mean), 중앙값(median), 최빈값(mode), 절사평균(trimmed mean) 등 중심 경향 측도(measures of central tendency)를 계산할 수 있습니다
2. 분산(variance), 표준편차(standard deviation), IQR, 변동계수(coefficient of variation) 등 산포 측도(measures of dispersion)를 계산할 수 있습니다
3. 왜도(skewness)와 첨도(kurtosis)를 이용해 분포 형태(distribution shape)를 설명하고 그 값을 해석할 수 있습니다
4. DataFrame에서 `describe` 메서드를 적용하고 상관행렬(correlation matrix) 및 공분산행렬(covariance matrix)을 계산할 수 있습니다
5. 단변량(univariate), 이변량(bivariate), 다변량(multivariate) 분석을 포괄하는 체계적인 EDA(탐색적 데이터 분석) 워크플로우를 구현할 수 있습니다
6. 정규성 검정(normality tests)(Shapiro-Wilk, Kolmogorov-Smirnov, Anderson-Darling)을 적용하고 Q-Q 플롯(Q-Q plot)을 해석할 수 있습니다
7. 탐색적 맥락에서 t-검정(t-test), 분산분석(ANOVA), 카이제곱 검정(chi-square test)을 이용해 집단을 비교할 수 있습니다
8. ydata-profiling, sweetviz 등 EDA 자동화 라이브러리를 언제 사용해야 하는지 판단할 수 있습니다

---

데이터를 모델링하거나 결론을 도출하기 전에, 먼저 다루는 데이터를 이해해야 합니다. 기술통계(descriptive statistics)는 수천 개의 데이터 포인트를 몇 가지 해석 가능한 숫자로 압축하고, EDA(탐색적 데이터 분석)는 시각화와 체계적인 탐색을 통해 숨겨진 패턴, 이상값, 관계를 드러냅니다. 두 가지를 함께 활용하면 모든 데이터 사이언스 프로젝트에서 빠뜨릴 수 없는 첫 번째 단계가 완성되며, 이후의 모든 의사결정을 이끄는 나침반이 됩니다.

---

## 1. 기술통계량

### 1.1 중심 경향 측도

```python
import pandas as pd
import numpy as np
from scipy import stats

data = [10, 15, 20, 25, 30, 35, 40, 100]  # 이상치 포함
s = pd.Series(data)

# 평균 (Mean)
print(f"평균: {s.mean():.2f}")  # 34.38

# 중앙값 (Median)
print(f"중앙값: {s.median():.2f}")  # 27.50

# 최빈값 (Mode)
data_with_mode = [1, 2, 2, 3, 3, 3, 4, 4]
print(f"최빈값: {pd.Series(data_with_mode).mode().values}")  # [3]

# 절사평균 (Trimmed Mean) - 이상치 영향 감소
# 언제 절사평균을 선택하나요? 이상치(outlier)가 의심되지만 완전히 제거하고 싶지 않을 때입니다.
# 일반 평균은 이상치(여기서는 100)에 끌려가지만, 절사평균은 상하위 10%를 제외한 후
# 평균을 구하므로 데이터를 영구 삭제하지 않고도 더 안정적인 '전형적 값'을 얻을 수 있습니다.
print(f"절사평균(10%): {stats.trim_mean(data, 0.1):.2f}")

# 가중평균
values = [10, 20, 30]
weights = [0.2, 0.3, 0.5]
weighted_mean = np.average(values, weights=weights)
print(f"가중평균: {weighted_mean}")
```

### 1.2 산포 측도

```python
s = pd.Series([10, 15, 20, 25, 30, 35, 40])

# 범위 (Range)
print(f"범위: {s.max() - s.min()}")

# 분산 (Variance)
print(f"분산(표본): {s.var():.2f}")
print(f"분산(모집단): {s.var(ddof=0):.2f}")

# 표준편차 (Standard Deviation)
print(f"표준편차(표본): {s.std():.2f}")
print(f"표준편차(모집단): {s.std(ddof=0):.2f}")

# 사분위범위 (IQR)
Q1 = s.quantile(0.25)
Q3 = s.quantile(0.75)
IQR = Q3 - Q1
print(f"IQR: {IQR}")

# 변동계수 (CV, Coefficient of Variation) - 상대적 산포
# CV는 표준편차(std dev) 단독으로는 알 수 없는 것을 알려줍니다: 평균 대비 산포가 얼마나 큰지.
# 단위나 규모가 다른 데이터셋을 비교할 때 이것이 중요합니다 —
# 예를 들어 소득(std ≈ 20,000달러)과 키(std ≈ 8cm)는 표준편차로 비교할 수 없지만,
# CV(std/mean)는 두 값을 단위 없는 비율로 표현해 의미 있는 교차 비교가 가능합니다.
cv = s.std() / s.mean()
print(f"변동계수: {cv:.4f}")

# 평균절대편차 (MAD)
mad = (s - s.mean()).abs().mean()
print(f"MAD: {mad:.2f}")
```

### 1.3 분포 형태 측도

```python
s = pd.Series([1, 2, 2, 3, 3, 3, 4, 4, 5, 10])

# 왜도 (Skewness) - 비대칭 정도
print(f"왜도: {s.skew():.4f}")
# > 0: 오른쪽 꼬리 (양의 왜도)
# < 0: 왼쪽 꼬리 (음의 왜도)
# = 0: 대칭

# 첨도 (Kurtosis) - 꼬리 두께
print(f"첨도: {s.kurtosis():.4f}")
# > 0: 정규분포보다 뾰족 (두꺼운 꼬리)
# < 0: 정규분포보다 평평 (얇은 꼬리)
# = 0: 정규분포와 유사

# scipy 사용 (Fisher 정의)
print(f"왜도 (scipy): {stats.skew(s):.4f}")
print(f"첨도 (scipy): {stats.kurtosis(s):.4f}")
```

### 1.4 백분위수와 분위수

```python
s = pd.Series(range(1, 101))

# 백분위수
print(f"25번째 백분위수: {s.quantile(0.25)}")
print(f"50번째 백분위수: {s.quantile(0.50)}")
print(f"75번째 백분위수: {s.quantile(0.75)}")
print(f"90번째 백분위수: {s.quantile(0.90)}")

# 여러 분위수 한번에
print(s.quantile([0.1, 0.25, 0.5, 0.75, 0.9]))

# 5수 요약 (Five-number summary)
print("5수 요약:")
print(f"최소: {s.min()}")
print(f"Q1: {s.quantile(0.25)}")
print(f"중앙값: {s.median()}")
print(f"Q3: {s.quantile(0.75)}")
print(f"최대: {s.max()}")
```

---

## 2. DataFrame 기술통계

### 2.1 describe 메서드

```python
df = pd.DataFrame({
    'age': [25, 30, 35, 40, 45, 50, 55],
    'salary': [40000, 45000, 50000, 60000, 70000, 80000, 100000],
    'department': ['IT', 'HR', 'IT', 'Sales', 'IT', 'HR', 'Sales']
})

# 수치형 열만
print(df.describe())

# 모든 열 (범주형 포함)
print(df.describe(include='all'))

# 특정 백분위수 지정
print(df.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]))
```

### 2.2 상관분석

```python
df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [2, 4, 5, 4, 5],
    'C': [5, 4, 3, 2, 1]
})

# 피어슨 상관계수 (선형 관계)
print("피어슨 상관계수:")
print(df.corr())

# 스피어만 상관계수 (순위 기반, 비선형 관계)
print("\n스피어만 상관계수:")
print(df.corr(method='spearman'))

# 켄달 상관계수 (순위 기반)
print("\n켄달 상관계수:")
print(df.corr(method='kendall'))

# 특정 열 간 상관계수
print(f"\nA와 B의 상관계수: {df['A'].corr(df['B']):.4f}")

# p-value와 함께
from scipy.stats import pearsonr, spearmanr
corr, p_value = pearsonr(df['A'], df['B'])
print(f"상관계수: {corr:.4f}, p-value: {p_value:.4f}")
```

### 2.3 공분산

```python
df = pd.DataFrame({
    'X': [1, 2, 3, 4, 5],
    'Y': [2, 4, 5, 4, 5]
})

# 공분산 행렬
print(df.cov())

# 두 변수 간 공분산
print(f"X와 Y의 공분산: {df['X'].cov(df['Y']):.4f}")
```

---

## 3. 탐색적 데이터 분석 (EDA)

### 3.1 데이터 개요 파악

```python
import pandas as pd
import numpy as np

# 샘플 데이터 생성
np.random.seed(42)
df = pd.DataFrame({
    'id': range(1, 1001),
    'age': np.random.randint(18, 70, 1000),
    'income': np.random.lognormal(10, 1, 1000),
    'gender': np.random.choice(['M', 'F'], 1000),
    'category': np.random.choice(['A', 'B', 'C', 'D'], 1000),
    'score': np.random.normal(75, 15, 1000)
})
df.loc[np.random.choice(1000, 50), 'income'] = np.nan  # 결측치 추가

# 1. 기본 정보
print("="*50)
print("1. 데이터 기본 정보")
print("="*50)
print(f"행 수: {len(df)}")
print(f"열 수: {len(df.columns)}")
print(f"\n컬럼 목록: {df.columns.tolist()}")
print(f"\n데이터 타입:\n{df.dtypes}")

# 2. 결측치 현황
print("\n" + "="*50)
print("2. 결측치 현황")
print("="*50)
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    '결측치 수': missing,
    '결측치 비율(%)': missing_pct
})
print(missing_df[missing_df['결측치 수'] > 0])

# 3. 기술통계
print("\n" + "="*50)
print("3. 수치형 변수 기술통계")
print("="*50)
print(df.describe())

# 4. 범주형 변수 빈도
print("\n" + "="*50)
print("4. 범주형 변수 빈도")
print("="*50)
for col in df.select_dtypes(include='object').columns:
    print(f"\n[{col}]")
    print(df[col].value_counts())
```

### 3.2 단변량 분석

```python
import matplotlib.pyplot as plt

# 수치형 변수
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove('id')

fig, axes = plt.subplots(2, len(numeric_cols), figsize=(15, 8))

for i, col in enumerate(numeric_cols):
    # 히스토그램
    axes[0, i].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
    axes[0, i].set_title(f'{col} - Histogram')
    axes[0, i].set_xlabel(col)
    axes[0, i].set_ylabel('Frequency')

    # 박스플롯
    axes[1, i].boxplot(df[col].dropna())
    axes[1, i].set_title(f'{col} - Boxplot')

plt.tight_layout()
plt.show()

# 범주형 변수
categorical_cols = df.select_dtypes(include='object').columns.tolist()

fig, axes = plt.subplots(1, len(categorical_cols), figsize=(12, 4))

for i, col in enumerate(categorical_cols):
    df[col].value_counts().plot(kind='bar', ax=axes[i], edgecolor='black')
    axes[i].set_title(f'{col} - Bar Chart')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Count')
    axes[i].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.show()
```

### 3.3 이변량 분석

```python
import seaborn as sns

# 수치형 vs 수치형: 산점도
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(df['age'], df['income'], alpha=0.5)
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Income')
axes[0].set_title('Age vs Income')

axes[1].scatter(df['age'], df['score'], alpha=0.5)
axes[1].set_xlabel('Age')
axes[1].set_ylabel('Score')
axes[1].set_title('Age vs Score')

plt.tight_layout()
plt.show()

# 범주형 vs 수치형: 박스플롯
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

df.boxplot(column='income', by='gender', ax=axes[0])
axes[0].set_title('Income by Gender')

df.boxplot(column='score', by='category', ax=axes[1])
axes[1].set_title('Score by Category')

plt.suptitle('')  # 자동 생성된 제목 제거
plt.tight_layout()
plt.show()

# 범주형 vs 범주형: 교차표
print("Gender와 Category 교차표:")
print(pd.crosstab(df['gender'], df['category']))
print(pd.crosstab(df['gender'], df['category'], normalize='index'))  # 행 기준 비율
```

### 3.4 다변량 분석

```python
# 상관행렬 히트맵
numeric_df = df[['age', 'income', 'score']].dropna()
correlation_matrix = numeric_df.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm',
            center=0, vmin=-1, vmax=1, fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# 산점도 행렬 (Pair Plot)
sns.pairplot(df[['age', 'income', 'score', 'gender']].dropna(),
             hue='gender', diag_kind='hist')
plt.suptitle('Pair Plot', y=1.02)
plt.tight_layout()
plt.show()
```

---

## 4. 분포 분석

### 4.1 분포 확인

```python
from scipy import stats

data = df['income'].dropna()

# 정규성 검정 - Shapiro-Wilk (n < 5000)
if len(data) < 5000:
    stat, p_value = stats.shapiro(data[:5000])
    print(f"Shapiro-Wilk 검정: 통계량={stat:.4f}, p-value={p_value:.4f}")

# 정규성 검정 - Kolmogorov-Smirnov
stat, p_value = stats.kstest(data, 'norm',
                             args=(data.mean(), data.std()))
print(f"K-S 검정: 통계량={stat:.4f}, p-value={p_value:.4f}")

# 정규성 검정 - Anderson-Darling
result = stats.anderson(data, dist='norm')
print(f"Anderson-Darling 검정: 통계량={result.statistic:.4f}")

# 시각적 확인: Q-Q plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 히스토그램과 정규분포 곡선
axes[0].hist(data, bins=50, density=True, alpha=0.7, edgecolor='black')
x = np.linspace(data.min(), data.max(), 100)
axes[0].plot(x, stats.norm.pdf(x, data.mean(), data.std()),
             'r-', linewidth=2, label='Normal')
axes[0].set_title('Histogram with Normal Curve')
axes[0].legend()

# Q-Q plot
stats.probplot(data, dist="norm", plot=axes[1])
axes[1].set_title('Q-Q Plot')

plt.tight_layout()
plt.show()
```

### 4.2 분포 변환

```python
# 로그 정규분포 데이터
data = df['income'].dropna()

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 원본
axes[0, 0].hist(data, bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].set_title(f'Original (Skew: {stats.skew(data):.2f})')

# 로그 변환
log_data = np.log1p(data)
axes[0, 1].hist(log_data, bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].set_title(f'Log Transform (Skew: {stats.skew(log_data):.2f})')

# 제곱근 변환
sqrt_data = np.sqrt(data)
axes[1, 0].hist(sqrt_data, bins=50, edgecolor='black', alpha=0.7)
axes[1, 0].set_title(f'Square Root Transform (Skew: {stats.skew(sqrt_data):.2f})')

# Box-Cox 변환
boxcox_data, lambda_param = stats.boxcox(data)
axes[1, 1].hist(boxcox_data, bins=50, edgecolor='black', alpha=0.7)
axes[1, 1].set_title(f'Box-Cox Transform (λ={lambda_param:.2f}, Skew: {stats.skew(boxcox_data):.2f})')

plt.tight_layout()
plt.show()
```

---

## 5. 그룹별 분석

### 5.1 그룹별 기술통계

```python
# 그룹별 요약
print("성별 그룹별 통계:")
print(df.groupby('gender')[['age', 'income', 'score']].agg(['mean', 'std', 'count']))

print("\n카테고리별 통계:")
print(df.groupby('category')[['income', 'score']].describe())

# 다중 그룹
print("\n성별 & 카테고리별 평균 소득:")
print(df.groupby(['gender', 'category'])['income'].mean().unstack())
```

### 5.2 그룹 간 비교

```python
# t-검정 (두 그룹)
male_income = df[df['gender'] == 'M']['income'].dropna()
female_income = df[df['gender'] == 'F']['income'].dropna()

stat, p_value = stats.ttest_ind(male_income, female_income)
print(f"독립표본 t-검정: t={stat:.4f}, p-value={p_value:.4f}")

# ANOVA (세 그룹 이상)
groups = [df[df['category'] == cat]['score'].dropna() for cat in df['category'].unique()]
stat, p_value = stats.f_oneway(*groups)
print(f"ANOVA: F={stat:.4f}, p-value={p_value:.4f}")

# 카이제곱 검정 (범주형 변수)
contingency_table = pd.crosstab(df['gender'], df['category'])
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
print(f"카이제곱 검정: χ²={chi2:.4f}, p-value={p_value:.4f}")
```

---

## 6. EDA 자동화 라이브러리

### 6.1 pandas-profiling (ydata-profiling)

자동화 프로파일링은 낯선 데이터셋을 처음 접할 때 빠르게 전체 현황을 파악하기에 가장 적합합니다 — 결측치 패턴, 분포, 상관관계, 중복값을 한 번에 잡아냅니다. 프로덕션 파이프라인이나 도메인 특화 검사가 필요한 경우에는 수동 EDA가 더 높은 제어력과 재현성을 제공합니다.

```python
# pip install ydata-profiling

# from ydata_profiling import ProfileReport

# report = ProfileReport(df, title="EDA Report", explorative=True)
# report.to_file("eda_report.html")

# 간소화 버전 (대용량 데이터)
# report = ProfileReport(df, minimal=True)
```

### 6.2 sweetviz

```python
# pip install sweetviz

# import sweetviz as sv

# report = sv.analyze(df)
# report.show_html("sweetviz_report.html")

# 두 데이터셋 비교 — 훈련/테스트 분할 검증에 특히 유용합니다:
# 두 분포가 크게 다르면 모델이 분포 변화(distribution shift)에 취약할 수 있습니다.
# report = sv.compare(df1, df2)
```

---

## 7. EDA 체크리스트

```markdown
## EDA 체크리스트

### 1. 데이터 개요
- [ ] 행/열 수 확인
- [ ] 데이터 타입 확인
- [ ] 메모리 사용량 확인

### 2. 결측치
- [ ] 결측치 존재 여부
- [ ] 결측치 비율
- [ ] 결측치 패턴 (MCAR, MAR, MNAR)

### 3. 수치형 변수
- [ ] 기술통계 (평균, 중앙값, 표준편차 등)
- [ ] 분포 형태 (왜도, 첨도)
- [ ] 이상치 존재 여부
- [ ] 히스토그램/박스플롯

### 4. 범주형 변수
- [ ] 카테고리 수
- [ ] 빈도 분포
- [ ] 희소 카테고리 존재 여부

### 5. 변수 간 관계
- [ ] 상관분석 (수치형)
- [ ] 교차표 (범주형)
- [ ] 그룹별 비교

### 6. 타겟 변수
- [ ] 타겟 분포 (불균형 여부)
- [ ] 타겟과 특성 간 관계
```

---

## 연습 문제

### 문제 1: 기술통계
다음 데이터의 5수 요약을 구하세요.

```python
data = [12, 15, 18, 22, 25, 28, 30, 35, 40, 100]
s = pd.Series(data)

# 풀이
print(f"최소: {s.min()}")
print(f"Q1: {s.quantile(0.25)}")
print(f"중앙값: {s.median()}")
print(f"Q3: {s.quantile(0.75)}")
print(f"최대: {s.max()}")
```

### 문제 2: 상관분석
두 변수 간의 상관계수를 구하고 해석하세요.

```python
df = pd.DataFrame({
    'study_hours': [1, 2, 3, 4, 5, 6, 7, 8],
    'score': [50, 55, 60, 65, 70, 80, 85, 95]
})

# 풀이
corr = df['study_hours'].corr(df['score'])
print(f"상관계수: {corr:.4f}")
# 강한 양의 상관관계 (공부 시간이 늘수록 점수 증가)
```

### 문제 3: 그룹 비교
그룹별 평균과 표준편차를 구하세요.

```python
df = pd.DataFrame({
    'group': ['A', 'A', 'B', 'B', 'A', 'B'],
    'value': [10, 12, 20, 22, 11, 21]
})

# 풀이
print(df.groupby('group')['value'].agg(['mean', 'std']))
```

---

## 요약

| 측도 유형 | 측도 | 함수 |
|----------|------|------|
| 중심 경향 | 평균, 중앙값, 최빈값 | `mean()`, `median()`, `mode()` |
| 산포 | 분산, 표준편차, IQR | `var()`, `std()`, `quantile()` |
| 분포 형태 | 왜도, 첨도 | `skew()`, `kurtosis()` |
| 관계 | 상관계수, 공분산 | `corr()`, `cov()` |
| 요약 | 기술통계 | `describe()` |
