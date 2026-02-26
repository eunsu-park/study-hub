# 25. 실전 프로젝트 - 데이터 분석 종합 실습

[이전: 실험 설계](./24_Experimental_Design.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 실제 데이터셋에 대해 데이터 로딩부터 인사이트 도출까지 체계적인 EDA 워크플로우를 적용할 수 있다
2. 결측 데이터(missing data) 패턴을 식별하고 시각화하며, 처리 방법에 대한 근거 있는 판단을 내릴 수 있다
3. 적절한 시각화 기법을 사용하여 단변량(univariate), 이변량(bivariate), 다변량(multivariate) 분석을 수행할 수 있다
4. 카이제곱(chi-square) 검정, t-검정 등 통계적 검정을 통해 데이터에서 관찰된 패턴을 검증할 수 있다
5. 분석 결과를 명확하고 실행 가능한 인사이트로 종합할 수 있다
6. 일반적인 탐색 작업을 자동화하는 재사용 가능한 EDA 보고서 템플릿을 구축할 수 있다

---

개별 통계 기법을 학습하는 것은 첫걸음에 불과합니다. 진짜 과제는 원시 데이터에서 시작해 실행 가능한 인사이트로 끝나는 일관된 엔드-투-엔드 분석으로 기법들을 결합하는 방법을 아는 것입니다. 이 실전 프로젝트는 실제 데이터셋에 대한 완전한 분석 과정을 단계별로 안내하며, 교과서 독자와 현장 데이터 과학자를 구분짓는 판단력과 워크플로우 습관을 길러줍니다.

---

## 프로젝트 1: Titanic 생존 분석

### 1.1 데이터 로딩 및 개요

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Seaborn 내장 데이터셋 로드
titanic = sns.load_dataset('titanic')

# 데이터 개요
print("="*50)
print("데이터 기본 정보")
print("="*50)
print(f"행 수: {len(titanic)}")
print(f"열 수: {len(titanic.columns)}")
print(f"\n컬럼 목록:\n{titanic.columns.tolist()}")
print(f"\n데이터 타입:\n{titanic.dtypes}")

# 처음 5행
print("\n" + "="*50)
print("데이터 미리보기")
print("="*50)
print(titanic.head())
```

### 1.2 결측치 분석

```python
print("="*50)
print("결측치 분석")
print("="*50)

# 결측치 현황
missing = titanic.isnull().sum()
missing_pct = (missing / len(titanic) * 100).round(2)
missing_df = pd.DataFrame({
    '결측치 수': missing,
    '결측치 비율(%)': missing_pct
}).sort_values('결측치 비율(%)', ascending=False)

print(missing_df[missing_df['결측치 수'] > 0])

# 시각화
fig, ax = plt.subplots(figsize=(10, 6))
missing_cols = missing_df[missing_df['결측치 수'] > 0].index
missing_vals = missing_df.loc[missing_cols, '결측치 비율(%)']
ax.barh(missing_cols, missing_vals, color='coral')
ax.set_xlabel('결측치 비율 (%)')
ax.set_title('결측치 현황')
for i, v in enumerate(missing_vals):
    ax.text(v + 0.5, i, f'{v}%', va='center')
plt.tight_layout()
plt.show()
```

### 1.3 타겟 변수 분석

```python
print("="*50)
print("타겟 변수 (생존 여부) 분석")
print("="*50)

# 생존율
survival_rate = titanic['survived'].value_counts(normalize=True)
print(f"생존율: {survival_rate[1]:.1%}")
print(f"사망율: {survival_rate[0]:.1%}")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 빈도
titanic['survived'].value_counts().plot(kind='bar', ax=axes[0],
                                         color=['coral', 'steelblue'])
axes[0].set_title('생존 여부 빈도')
axes[0].set_xticklabels(['사망', '생존'], rotation=0)
axes[0].set_ylabel('인원 수')

# 비율
titanic['survived'].value_counts().plot(kind='pie', ax=axes[1],
                                         autopct='%1.1f%%',
                                         colors=['coral', 'steelblue'],
                                         labels=['사망', '생존'])
axes[1].set_title('생존 여부 비율')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()
```

### 1.4 범주형 변수별 생존율

```python
print("="*50)
print("범주형 변수별 생존율")
print("="*50)

categorical_vars = ['sex', 'pclass', 'embarked', 'alone']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, var in zip(axes.flat, categorical_vars):
    survival_by_var = titanic.groupby(var)['survived'].mean().sort_values(ascending=False)
    survival_by_var.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
    ax.set_title(f'{var}별 생존율')
    ax.set_ylabel('생존율')
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x', rotation=45)

    # 값 표시
    for i, v in enumerate(survival_by_var):
        ax.text(i, v + 0.02, f'{v:.1%}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# 통계 요약
print("\n성별 생존율:")
print(titanic.groupby('sex')['survived'].agg(['mean', 'count']))

print("\n객실 등급별 생존율:")
print(titanic.groupby('pclass')['survived'].agg(['mean', 'count']))
```

### 1.5 수치형 변수 분석

```python
print("="*50)
print("수치형 변수 분석")
print("="*50)

numeric_vars = ['age', 'fare']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for i, var in enumerate(numeric_vars):
    # 히스토그램 (생존 여부별)
    for survived, label, color in [(0, '사망', 'coral'), (1, '생존', 'steelblue')]:
        data = titanic[titanic['survived'] == survived][var].dropna()
        axes[i, 0].hist(data, bins=30, alpha=0.6, label=label, color=color)
    axes[i, 0].set_title(f'{var} 분포 (생존 여부별)')
    axes[i, 0].set_xlabel(var)
    axes[i, 0].legend()

    # 박스플롯
    titanic.boxplot(column=var, by='survived', ax=axes[i, 1])
    axes[i, 1].set_title(f'{var} (생존 여부별)')
    axes[i, 1].set_xlabel('생존 여부')

plt.suptitle('')
plt.tight_layout()
plt.show()

# 통계 요약
print("\n나이별 생존 통계:")
print(titanic.groupby('survived')['age'].describe())
```

### 1.6 다변량 분석

```python
print("="*50)
print("다변량 분석")
print("="*50)

# 성별 & 객실 등급별 생존율
pivot = pd.pivot_table(titanic, values='survived',
                       index='pclass', columns='sex', aggfunc='mean')
print("성별 & 객실 등급별 생존율:")
print(pivot)

# 히트맵
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(pivot, annot=True, cmap='RdYlGn', fmt='.1%',
            vmin=0, vmax=1, ax=ax)
ax.set_title('성별 & 객실 등급별 생존율')
plt.show()

# 나이 그룹 생성
titanic['age_group'] = pd.cut(titanic['age'],
                              bins=[0, 12, 18, 35, 60, 100],
                              labels=['어린이', '청소년', '청년', '중년', '노년'])

# 나이 그룹별 생존율
age_survival = titanic.groupby('age_group')['survived'].mean()
print("\n나이 그룹별 생존율:")
print(age_survival)
```

### 1.7 통계 검정

```python
from scipy import stats

print("="*50)
print("통계 검정")
print("="*50)

# 성별에 따른 생존율 차이 (카이제곱 검정)
contingency = pd.crosstab(titanic['sex'], titanic['survived'])
chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
print(f"\n성별-생존 카이제곱 검정:")
print(f"χ² = {chi2:.4f}, p-value = {p_value:.4f}")

# 생존 여부에 따른 나이 차이 (t-검정)
survived_age = titanic[titanic['survived'] == 1]['age'].dropna()
died_age = titanic[titanic['survived'] == 0]['age'].dropna()

stat, p_value = stats.ttest_ind(survived_age, died_age)
print(f"\n나이-생존 t-검정:")
print(f"t = {stat:.4f}, p-value = {p_value:.4f}")
print(f"생존자 평균 나이: {survived_age.mean():.1f}")
print(f"사망자 평균 나이: {died_age.mean():.1f}")
```

### 1.8 인사이트 정리

```python
print("="*50)
print("주요 인사이트")
print("="*50)

insights = """
1. 전체 생존율: 약 38%

2. 성별:
   - 여성 생존율(74%)이 남성(19%)보다 현저히 높음
   - "여성과 아이 먼저" 원칙의 영향

3. 객실 등급:
   - 1등석(63%) > 2등석(47%) > 3등석(24%)
   - 상위 등급일수록 생존율 높음

4. 나이:
   - 어린이 생존율이 가장 높음
   - 생존자 평균 나이가 사망자보다 약간 낮음

5. 동반자:
   - 혼자 탑승한 승객의 생존율이 낮음

6. 운임:
   - 높은 운임을 지불한 승객의 생존율이 높음
   - (객실 등급과 상관관계)
"""
print(insights)
```

---

## 프로젝트 2: 팁 데이터 분석

### 2.1 데이터 탐색

```python
tips = sns.load_dataset('tips')

print("="*50)
print("Tips 데이터셋 개요")
print("="*50)
print(tips.info())
print("\n기술 통계:")
print(tips.describe())
```

### 2.2 팁 금액 분석

```python
# 팁 비율 계산
tips['tip_pct'] = tips['tip'] / tips['total_bill'] * 100

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 팁 금액 분포
axes[0, 0].hist(tips['tip'], bins=30, edgecolor='black', alpha=0.7)
axes[0, 0].axvline(tips['tip'].mean(), color='red', linestyle='--',
                   label=f'평균: ${tips["tip"].mean():.2f}')
axes[0, 0].set_title('팁 금액 분포')
axes[0, 0].set_xlabel('팁 ($)')
axes[0, 0].legend()

# 팁 비율 분포
axes[0, 1].hist(tips['tip_pct'], bins=30, edgecolor='black', alpha=0.7)
axes[0, 1].axvline(tips['tip_pct'].mean(), color='red', linestyle='--',
                   label=f'평균: {tips["tip_pct"].mean():.1f}%')
axes[0, 1].set_title('팁 비율 분포')
axes[0, 1].set_xlabel('팁 비율 (%)')
axes[0, 1].legend()

# 요일별 팁
tips.groupby('day')['tip'].mean().plot(kind='bar', ax=axes[1, 0],
                                        color='steelblue', edgecolor='black')
axes[1, 0].set_title('요일별 평균 팁')
axes[1, 0].set_ylabel('평균 팁 ($)')
axes[1, 0].tick_params(axis='x', rotation=45)

# 시간대별 팁
tips.groupby('time')['tip'].mean().plot(kind='bar', ax=axes[1, 1],
                                         color='coral', edgecolor='black')
axes[1, 1].set_title('시간대별 평균 팁')
axes[1, 1].set_ylabel('평균 팁 ($)')
axes[1, 1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.show()

print(f"평균 팁: ${tips['tip'].mean():.2f}")
print(f"평균 팁 비율: {tips['tip_pct'].mean():.1f}%")
```

### 2.3 청구 금액과 팁의 관계

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 산점도
sns.scatterplot(data=tips, x='total_bill', y='tip', hue='time',
                size='size', ax=axes[0])
axes[0].set_title('청구 금액 vs 팁')

# 회귀선
sns.regplot(data=tips, x='total_bill', y='tip', ax=axes[1],
            scatter_kws={'alpha': 0.5})
axes[1].set_title('청구 금액 vs 팁 (회귀선)')

plt.tight_layout()
plt.show()

# 상관계수
corr, p_value = stats.pearsonr(tips['total_bill'], tips['tip'])
print(f"상관계수: {corr:.4f} (p-value: {p_value:.4f})")
```

### 2.4 그룹별 비교

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 성별
sns.boxplot(data=tips, x='sex', y='tip_pct', ax=axes[0, 0])
axes[0, 0].set_title('성별 팁 비율')

# 흡연 여부
sns.boxplot(data=tips, x='smoker', y='tip_pct', ax=axes[0, 1])
axes[0, 1].set_title('흡연 여부별 팁 비율')

# 요일
sns.boxplot(data=tips, x='day', y='tip_pct', ax=axes[1, 0])
axes[1, 0].set_title('요일별 팁 비율')

# 인원수
sns.boxplot(data=tips, x='size', y='tip_pct', ax=axes[1, 1])
axes[1, 1].set_title('인원수별 팁 비율')

plt.tight_layout()
plt.show()

# 통계 검정: 성별 차이
male_tip = tips[tips['sex'] == 'Male']['tip_pct']
female_tip = tips[tips['sex'] == 'Female']['tip_pct']
stat, p_value = stats.ttest_ind(male_tip, female_tip)
print(f"\n성별 팁 비율 t-검정: t={stat:.4f}, p={p_value:.4f}")
```

---

## 프로젝트 3: 분석 보고서 템플릿

```python
def generate_eda_report(df, target=None):
    """
    EDA 보고서 자동 생성 함수

    Parameters:
    -----------
    df : DataFrame
        분석할 데이터프레임
    target : str, optional
        타겟 변수명
    """
    print("="*60)
    print("           탐색적 데이터 분석 (EDA) 보고서")
    print("="*60)

    # 1. 기본 정보
    print("\n1. 데이터 기본 정보")
    print("-"*40)
    print(f"   행 수: {len(df):,}")
    print(f"   열 수: {len(df.columns)}")
    print(f"   메모리 사용량: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # 2. 데이터 타입
    print("\n2. 데이터 타입 요약")
    print("-"*40)
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"   {dtype}: {count}개")

    # 3. 결측치
    print("\n3. 결측치 현황")
    print("-"*40)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100)
    for col, (cnt, pct) in zip(df.columns, zip(missing, missing_pct)):
        if cnt > 0:
            print(f"   {col}: {cnt}개 ({pct:.1f}%)")
    if missing.sum() == 0:
        print("   결측치 없음")

    # 4. 수치형 변수
    print("\n4. 수치형 변수 통계")
    print("-"*40)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        for col in numeric_cols[:5]:  # 상위 5개만
            print(f"\n   [{col}]")
            print(f"   평균: {df[col].mean():.2f}, 중앙값: {df[col].median():.2f}")
            print(f"   표준편차: {df[col].std():.2f}")
            print(f"   범위: [{df[col].min():.2f}, {df[col].max():.2f}]")

    # 5. 범주형 변수
    print("\n5. 범주형 변수 요약")
    print("-"*40)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols[:5]:  # 상위 5개만
        print(f"\n   [{col}]")
        print(f"   고유값 수: {df[col].nunique()}")
        print(f"   최빈값: {df[col].mode().values[0]}")

    # 6. 타겟 변수 (있는 경우)
    if target and target in df.columns:
        print(f"\n6. 타겟 변수 ({target}) 분석")
        print("-"*40)
        if df[target].dtype in ['int64', 'float64']:
            print(f"   평균: {df[target].mean():.2f}")
            print(f"   분포: 연속형")
        else:
            print(f"   클래스 분포:")
            for val, cnt in df[target].value_counts().items():
                print(f"   - {val}: {cnt} ({cnt/len(df)*100:.1f}%)")

    print("\n" + "="*60)
    print("                     보고서 끝")
    print("="*60)

# 사용 예시
# generate_eda_report(titanic, target='survived')
```

---

## 분석 체크리스트

```markdown
## 데이터 분석 체크리스트

### 1단계: 데이터 이해
- [ ] 데이터 출처와 수집 방법 확인
- [ ] 각 변수의 의미 파악
- [ ] 비즈니스 문맥 이해

### 2단계: 데이터 품질 확인
- [ ] 결측치 확인 및 처리 계획
- [ ] 이상치 탐지
- [ ] 데이터 타입 확인
- [ ] 중복 데이터 확인

### 3단계: 단변량 분석
- [ ] 수치형 변수 분포 확인
- [ ] 범주형 변수 빈도 확인
- [ ] 기술통계량 계산

### 4단계: 이변량/다변량 분석
- [ ] 변수 간 상관관계 분석
- [ ] 그룹별 비교 분석
- [ ] 타겟 변수와의 관계 분석

### 5단계: 통계 검정
- [ ] 적절한 검정 방법 선택
- [ ] 가정 검증
- [ ] 결과 해석

### 6단계: 인사이트 도출
- [ ] 주요 발견 정리
- [ ] 비즈니스 의미 해석
- [ ] 추가 분석 제안
```

---

## 요약

| 단계 | 주요 작업 | 도구/함수 |
|------|----------|----------|
| 데이터 로딩 | CSV/Excel/DB 로드 | `pd.read_*()` |
| 개요 파악 | 형태, 타입 확인 | `info()`, `describe()` |
| 결측치 | 확인 및 처리 | `isna()`, `fillna()` |
| 단변량 | 분포, 빈도 분석 | `histplot()`, `countplot()` |
| 이변량 | 관계 분석 | `scatterplot()`, `boxplot()` |
| 다변량 | 패턴 발견 | `heatmap()`, `pairplot()` |
| 통계 검정 | 유의성 검정 | `scipy.stats` |
| 인사이트 | 결과 정리 | 마크다운 보고서 |

---

## 연습 문제

### 연습 1: Iris 데이터셋 전체 EDA

Iris 데이터셋에 대한 완전한 탐색적 데이터 분석(EDA, Exploratory Data Analysis)을 수행하세요.

1. 데이터셋을 로드하세요:
   ```python
   from sklearn.datasets import load_iris
   import pandas as pd
   iris = pd.DataFrame(load_iris().data, columns=load_iris().feature_names)
   iris['species'] = load_iris().target_names[load_iris().target]
   ```
2. `generate_eda_report(iris, target='species')`를 실행하고 출력의 각 섹션을 해석하세요.
3. `hue='species'`, `diag_kind='kde'`로 페어 플롯(`sns.pairplot`)을 생성하세요. 두 특징 중 어느 것이 품종 간 가장 깔끔한 선형 분리를 보여주나요?
4. 식별한 두 특징에 대해 세 품종 그룹에 걸쳐 일원 분산 분석(ANOVA, Analysis of Variance) F-통계량(`scipy.stats.f_oneway`)을 계산하세요. p-값을 보고하고 결론을 서술하세요.
5. Section 1.8(핵심 인사이트)의 형식에 따라 3-5개의 불릿으로 발견 사항을 요약하세요.

### 연습 2: 결측치 처리 전략 비교

다양한 결측치 대체(imputation) 전략이 후속 통계에 미치는 영향을 조사하세요.

1. 타이타닉 데이터셋을 가져와 `fare` 열에 무작위로 20%의 결측치를 인위적으로 도입하세요.
2. 네 가지 대체 전략을 적용하세요:
   - 결측 `fare`가 있는 행 삭제
   - 열 평균으로 채우기
   - 열 중앙값으로 채우기
   - 그룹 중앙값으로 채우기 (`pclass`로 그룹화)
3. 각 전략에 대해 계산하세요: (a) 평균 운임(fare), (b) 운임의 표준 편차, (c) `fare`와 `survived` 간 상관관계.
4. 네 가지 통계 세트를 비교 표로 표시하세요.
5. 어느 전략이 원래 상관관계를 가장 잘 보존하나요? 이상치(outlier)가 있을 때 평균 대체보다 중앙값 대체가 일반적으로 선호되는 이유를 설명하세요.

### 연습 3: 다중 데이터셋 비교 분석

새 데이터셋에 분석 워크플로를 적용하고 타이타닉 결과와 비교하세요.

1. Seaborn에서 `penguins` 데이터셋을 로드하세요: `penguins = sns.load_dataset('penguins')`.
2. 레슨의 전체 체크리스트(1-6단계)를 따르세요:
   - 결측치를 문서화하고 결측치가 있는 행을 삭제하세요.
   - 네 개의 수치형 열 전체에 단변량 분석을 수행하세요.
   - 이변량 분석: `species`로 색상 구분된 `bill_length_mm` vs. `flipper_length_mm` 산점도를 생성하세요.
   - 카이제곱 검정(chi-square test): `island`와 `species` 간에 유의미한 연관성이 있나요?
3. 품종별로 색상 구분된 네 개의 수치형 특징 히스토그램을 보여주는 2×2 서브플롯을 생성하세요.
4. Section 1.8의 타이타닉 인사이트에 유사한 5개의 불릿 인사이트 섹션을 작성하세요.
5. 비교하세요: 목표 변수가 범주형(타이타닉 `survived`)인 경우와 지정된 목표가 없는 경우(펭귄) 분석 워크플로는 어떻게 달라지나요?

### 연습 4: EDA 보고서 생성기 확장

`generate_eda_report()` 함수에 세 개의 새로운 분석 섹션을 추가하세요.

1. **섹션 7: 이상치(Outlier) 탐지** 추가 — 각 수치형 열에 대해 IQR을 계산하고 [Q1 − 1.5·IQR, Q3 + 1.5·IQR] 범위 밖의 값을 이상치로 표시하세요. 열별 이상치 수와 비율을 출력하세요.
2. **섹션 8: 상관관계 요약** 추가 — 전체 상관 행렬을 계산하고 가장 강한 양의 상관관계 5개와 가장 강한 음의 상관관계 5개를 출력하세요 (자기 상관 제외).
3. **섹션 9: 왜도(Skewness)와 첨도(Kurtosis)** 추가 — 각 수치형 열의 왜도와 첨도 값을 출력하세요. |왜도| > 1인 열을 "높은 왜도"로 표시하고 로그 변환(log transformation)을 제안하세요.
4. 타이타닉과 tips 데이터셋 모두에서 확장된 함수를 테스트하세요.
5. 각 섹션을 키워드 인수로 켜고 끌 수 있도록 함수를 리팩토링하세요 (예: `outliers=True, correlation=True, skewness=True`).

### 연습 5: 실제 데이터셋에서의 엔드-투-엔드 분석

직접 선택한 데이터셋(또는 Seaborn의 `diamonds` 데이터셋)에 전체 워크플로를 적용하세요.

1. 데이터셋을 로드하세요: `diamonds = sns.load_dataset('diamonds')`. 53,940행과 price, carat, cut, color, clarity 등 10개의 열이 있습니다.
2. EDA 보고서 생성기를 실행하세요. 데이터 타입, 결측치, 기본 통계에 주목하세요.
3. 다이아몬드 가격을 결정하는 요인에 대한 세 가지 가설을 세우세요 (예: "높은 캐럿은 더 높은 가격을 예측한다" 또는 "컷(cut) 품질은 가격과 연관된다").
4. 각 가설에 대해:
   - 적절한 시각화를 선택하세요 (산점도, 박스 플롯, 오차 막대가 있는 막대 차트).
   - 적절한 통계 검정을 선택하세요 (피어슨(Pearson) 상관관계, t-검정, ANOVA).
   - 검정 통계량, p-값, 효과 크기(Cohen's d 또는 η²)를 보고하세요.
5. 세 가지 발견 사항과 그 실용적 의미를 요약하는 간결한 분석 보고서(≤ 300 단어)를 작성하세요 (예: 구매자가 우선시해야 할 요소는 무엇인가요?).

[이전: 실험 설계](./24_Experimental_Design.md)
