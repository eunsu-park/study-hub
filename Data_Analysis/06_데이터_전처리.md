# 데이터 전처리

## 개요

데이터 전처리는 분석이나 모델링 전에 데이터를 정제하고 변환하는 과정입니다. 결측치 처리, 이상치 탐지, 정규화, 인코딩 등 핵심 기법을 다룹니다.

---

## 1. 결측치 처리

### 1.1 결측치 확인

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'A': [1, 2, None, 4, 5],
    'B': [None, 2, 3, None, 5],
    'C': ['a', 'b', None, 'd', 'e'],
    'D': [1.0, 2.0, 3.0, 4.0, 5.0]
})

# 결측치 확인
print(df.isna())          # 불리언 마스크
print(df.isna().sum())    # 열별 결측치 수
print(df.isna().sum().sum())  # 전체 결측치 수

# 결측치 비율
print(df.isna().mean() * 100)

# 결측치가 있는 행/열
print(df[df.isna().any(axis=1)])  # 결측치가 있는 행
print(df.columns[df.isna().any()])  # 결측치가 있는 열

# 결측치 시각화 (missingno 라이브러리)
# import missingno as msno
# msno.matrix(df)
```

### 1.2 결측치 제거

```python
df = pd.DataFrame({
    'A': [1, 2, None, 4],
    'B': [None, 2, 3, 4],
    'C': [1, None, 3, None]
})

# 결측치가 있는 행 제거
print(df.dropna())

# 모든 값이 결측인 행 제거
print(df.dropna(how='all'))

# 특정 열 기준
print(df.dropna(subset=['A']))
print(df.dropna(subset=['A', 'B']))

# 임계값 설정 (최소 비결측값 개수)
print(df.dropna(thresh=2))  # 최소 2개의 비결측값
```

### 1.3 결측치 대체

```python
df = pd.DataFrame({
    'numeric': [1, 2, None, 4, 5, None],
    'category': ['A', 'B', None, 'A', 'B', 'A']
})

# 특정 값으로 대체
df_filled = df.fillna(0)
df_filled = df.fillna({'numeric': 0, 'category': 'Unknown'})

# 통계값으로 대체
df['numeric'] = df['numeric'].fillna(df['numeric'].mean())     # 평균
df['numeric'] = df['numeric'].fillna(df['numeric'].median())   # 중앙값
df['category'] = df['category'].fillna(df['category'].mode()[0])  # 최빈값

# 앞/뒤 값으로 대체
df_ffill = df.fillna(method='ffill')  # 앞의 값으로
df_bfill = df.fillna(method='bfill')  # 뒤의 값으로

# 보간 (interpolation)
df['numeric'] = df['numeric'].interpolate(method='linear')
df['numeric'] = df['numeric'].interpolate(method='polynomial', order=2)
```

### 1.4 그룹별 결측치 처리

```python
df = pd.DataFrame({
    'group': ['A', 'A', 'B', 'B', 'A', 'B'],
    'value': [1, None, 3, None, 5, 6]
})

# 그룹별 평균으로 대체
df['value'] = df.groupby('group')['value'].transform(
    lambda x: x.fillna(x.mean())
)
print(df)
```

---

## 2. 이상치 탐지

### 2.1 통계적 방법

```python
df = pd.DataFrame({
    'value': [10, 12, 11, 13, 100, 11, 12, 10, 9, 11]
})

# IQR 방법
Q1 = df['value'].quantile(0.25)
Q3 = df['value'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['value'] < lower_bound) | (df['value'] > upper_bound)]
print("이상치:", outliers)

# Z-score 방법
from scipy import stats

z_scores = np.abs(stats.zscore(df['value']))
outliers = df[z_scores > 3]  # |z| > 3인 경우
print("이상치:", outliers)

# 수정된 Z-score (MAD 기반)
median = df['value'].median()
mad = np.median(np.abs(df['value'] - median))
modified_z = 0.6745 * (df['value'] - median) / mad
outliers = df[np.abs(modified_z) > 3.5]
```

### 2.2 시각적 방법

```python
import matplotlib.pyplot as plt

df = pd.DataFrame({
    'value': np.concatenate([np.random.randn(100), [10, -10]])
})

# 박스플롯
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].boxplot(df['value'])
axes[0].set_title('Box Plot')

# 히스토그램
axes[1].hist(df['value'], bins=30, edgecolor='black')
axes[1].set_title('Histogram')

plt.tight_layout()
plt.show()
```

### 2.3 이상치 처리

```python
df = pd.DataFrame({
    'value': [10, 12, 11, 13, 100, 11, 12, 10, 9, -50]
})

# 1. 제거
Q1, Q3 = df['value'].quantile([0.25, 0.75])
IQR = Q3 - Q1
df_clean = df[(df['value'] >= Q1 - 1.5 * IQR) &
              (df['value'] <= Q3 + 1.5 * IQR)]

# 2. 대체 (클리핑)
lower = df['value'].quantile(0.05)
upper = df['value'].quantile(0.95)
df['value_clipped'] = df['value'].clip(lower, upper)

# 3. 윈저화 (Winsorizing)
from scipy.stats import mstats
df['value_winsorized'] = mstats.winsorize(df['value'], limits=[0.05, 0.05])

# 4. 로그 변환 (왜도가 큰 데이터)
df['value_log'] = np.log1p(df['value'] - df['value'].min() + 1)
```

---

## 3. 데이터 정규화/표준화

### 3.1 Min-Max 정규화

```python
df = pd.DataFrame({
    'A': [10, 20, 30, 40, 50],
    'B': [100, 200, 300, 400, 500]
})

# 수동 구현
df_normalized = (df - df.min()) / (df.max() - df.min())
print(df_normalized)

# sklearn 사용
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_normalized = pd.DataFrame(
    scaler.fit_transform(df),
    columns=df.columns
)
print(df_normalized)
```

### 3.2 표준화 (Z-score)

```python
df = pd.DataFrame({
    'A': [10, 20, 30, 40, 50],
    'B': [100, 200, 300, 400, 500]
})

# 수동 구현
df_standardized = (df - df.mean()) / df.std()
print(df_standardized)

# sklearn 사용
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_standardized = pd.DataFrame(
    scaler.fit_transform(df),
    columns=df.columns
)
print(df_standardized)
```

### 3.3 다양한 스케일링 방법

```python
from sklearn.preprocessing import (
    MinMaxScaler, StandardScaler, RobustScaler,
    MaxAbsScaler, QuantileTransformer, PowerTransformer
)

df = pd.DataFrame({
    'value': [1, 2, 3, 4, 5, 100]  # 이상치 포함
})

# RobustScaler (이상치에 강건)
scaler = RobustScaler()  # 중앙값과 IQR 사용
robust_scaled = scaler.fit_transform(df)

# MaxAbsScaler (절댓값 최대로 스케일링)
scaler = MaxAbsScaler()
maxabs_scaled = scaler.fit_transform(df)

# QuantileTransformer (분위수 기반)
scaler = QuantileTransformer(output_distribution='normal')
quantile_scaled = scaler.fit_transform(df)

# PowerTransformer (정규분포에 가깝게)
scaler = PowerTransformer(method='yeo-johnson')
power_scaled = scaler.fit_transform(df)
```

---

## 4. 범주형 변수 인코딩

### 4.1 레이블 인코딩

```python
from sklearn.preprocessing import LabelEncoder

df = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'red', 'blue']
})

# sklearn LabelEncoder
le = LabelEncoder()
df['color_encoded'] = le.fit_transform(df['color'])
print(df)
print("클래스:", le.classes_)

# 역변환
original = le.inverse_transform(df['color_encoded'])
print("원본:", original)

# pandas factorize
codes, uniques = pd.factorize(df['color'])
df['color_factorized'] = codes
print(df)
```

### 4.2 원-핫 인코딩

```python
df = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'red'],
    'size': ['S', 'M', 'L', 'M']
})

# pandas get_dummies
df_encoded = pd.get_dummies(df, columns=['color', 'size'])
print(df_encoded)

# drop_first 옵션 (다중공선성 방지)
df_encoded = pd.get_dummies(df, columns=['color'], drop_first=True)
print(df_encoded)

# sklearn OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded = encoder.fit_transform(df[['color', 'size']])
print(encoded)
print("특성 이름:", encoder.get_feature_names_out())
```

### 4.3 순서형 인코딩

```python
from sklearn.preprocessing import OrdinalEncoder

df = pd.DataFrame({
    'education': ['high school', 'bachelor', 'master', 'phd', 'bachelor']
})

# 순서 지정
order = ['high school', 'bachelor', 'master', 'phd']

# sklearn OrdinalEncoder
encoder = OrdinalEncoder(categories=[order])
df['education_encoded'] = encoder.fit_transform(df[['education']])
print(df)

# pandas Categorical
df['education_cat'] = pd.Categorical(
    df['education'],
    categories=order,
    ordered=True
)
df['education_codes'] = df['education_cat'].cat.codes
print(df)
```

### 4.4 빈도 인코딩

```python
df = pd.DataFrame({
    'category': ['A', 'B', 'A', 'C', 'B', 'A', 'A', 'C']
})

# 빈도 계산
freq_map = df['category'].value_counts() / len(df)
df['category_freq'] = df['category'].map(freq_map)
print(df)
```

### 4.5 타겟 인코딩

```python
df = pd.DataFrame({
    'category': ['A', 'B', 'A', 'C', 'B', 'A'],
    'target': [1, 0, 1, 0, 1, 1]
})

# 카테고리별 타겟 평균
target_mean = df.groupby('category')['target'].mean()
df['category_target_encoded'] = df['category'].map(target_mean)
print(df)

# 과적합 방지를 위한 스무딩
def target_encode_smoothed(df, col, target, weight=10):
    global_mean = df[target].mean()
    agg = df.groupby(col)[target].agg(['mean', 'count'])
    smoothed = (agg['count'] * agg['mean'] + weight * global_mean) / (agg['count'] + weight)
    return df[col].map(smoothed)

df['category_smoothed'] = target_encode_smoothed(df, 'category', 'target')
```

---

## 5. 수치형 변환

### 5.1 로그 변환

```python
df = pd.DataFrame({
    'value': [1, 10, 100, 1000, 10000]
})

# 로그 변환
df['log'] = np.log(df['value'])
df['log10'] = np.log10(df['value'])
df['log1p'] = np.log1p(df['value'])  # log(1 + x), 0 처리 가능

print(df)
```

### 5.2 Box-Cox / Yeo-Johnson 변환

```python
from scipy import stats
from sklearn.preprocessing import PowerTransformer

df = pd.DataFrame({
    'value': [1, 2, 5, 10, 50, 100, 500]
})

# Box-Cox (양수만 가능)
df['boxcox'], lambda_param = stats.boxcox(df['value'])
print(f"최적 람다: {lambda_param}")

# Yeo-Johnson (음수도 가능)
pt = PowerTransformer(method='yeo-johnson')
df['yeojohnson'] = pt.fit_transform(df[['value']])

print(df)
```

### 5.3 구간화 (Binning)

```python
df = pd.DataFrame({
    'age': [15, 22, 35, 45, 55, 65, 75, 85]
})

# 동일 간격 구간화
df['age_bin_equal'] = pd.cut(df['age'], bins=4)

# 사용자 정의 구간
bins = [0, 20, 40, 60, 100]
labels = ['youth', 'adult', 'middle', 'senior']
df['age_bin_custom'] = pd.cut(df['age'], bins=bins, labels=labels)

# 동일 빈도 구간화
df['age_qcut'] = pd.qcut(df['age'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

print(df)
```

---

## 6. 날짜/시간 처리

### 6.1 날짜 파싱

```python
df = pd.DataFrame({
    'date_str': ['2023-01-15', '2023/02/20', '15-03-2023', '04.25.2023']
})

# 자동 파싱
df['date1'] = pd.to_datetime(df['date_str'].iloc[0:2])

# 형식 지정
df['date'] = pd.to_datetime(df['date_str'], format='mixed', dayfirst=True)

# 오류 처리
df['date'] = pd.to_datetime(df['date_str'], errors='coerce')  # 오류 시 NaT
```

### 6.2 날짜 특성 추출

```python
df = pd.DataFrame({
    'timestamp': pd.date_range('2023-01-01', periods=100, freq='D')
})

# 기본 특성
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
df['day'] = df['timestamp'].dt.day
df['dayofweek'] = df['timestamp'].dt.dayofweek  # 0=월요일
df['dayofyear'] = df['timestamp'].dt.dayofyear
df['weekofyear'] = df['timestamp'].dt.isocalendar().week
df['quarter'] = df['timestamp'].dt.quarter

# 불리언 특성
df['is_weekend'] = df['timestamp'].dt.dayofweek >= 5
df['is_month_start'] = df['timestamp'].dt.is_month_start
df['is_month_end'] = df['timestamp'].dt.is_month_end

# 주기적 특성 (삼각함수 인코딩)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

print(df.head())
```

---

## 7. 텍스트 전처리

### 7.1 기본 정제

```python
df = pd.DataFrame({
    'text': ['  Hello, World!  ', 'PYTHON 3.9', 'data-science', None]
})

# 소문자 변환
df['lower'] = df['text'].str.lower()

# 공백 제거
df['stripped'] = df['text'].str.strip()

# 특수문자 제거
df['cleaned'] = df['text'].str.replace(r'[^\w\s]', '', regex=True)

# 숫자 제거
df['no_numbers'] = df['text'].str.replace(r'\d+', '', regex=True)

print(df)
```

### 7.2 토큰화와 불용어 제거

```python
import re

# 간단한 토큰화
df = pd.DataFrame({
    'text': ['This is a sample text.', 'Another example here.']
})

df['tokens'] = df['text'].str.lower().str.split()

# 불용어 제거
stopwords = {'a', 'an', 'the', 'is', 'this', 'here'}
df['filtered'] = df['tokens'].apply(
    lambda x: [word for word in x if word not in stopwords] if x else []
)

print(df)
```

---

## 8. 전처리 파이프라인

### 8.1 sklearn Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 예제 데이터
df = pd.DataFrame({
    'age': [25, None, 35, 45, None],
    'salary': [50000, 60000, None, 80000, 70000],
    'department': ['IT', 'HR', 'IT', None, 'Sales']
})

# 수치형 파이프라인
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 범주형 파이프라인
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 전체 전처리기
preprocessor = ColumnTransformer([
    ('numeric', numeric_pipeline, ['age', 'salary']),
    ('categorical', categorical_pipeline, ['department'])
])

# 변환 실행
X_transformed = preprocessor.fit_transform(df)
print(X_transformed)
```

### 8.2 사용자 정의 변환기

```python
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        self.lower_ = Q1 - self.factor * IQR
        self.upper_ = Q3 + self.factor * IQR
        return self

    def transform(self, X):
        X_clipped = np.clip(X, self.lower_, self.upper_)
        return X_clipped

# 사용
remover = OutlierRemover(factor=1.5)
data = np.array([[1], [2], [3], [100], [4], [5]])
transformed = remover.fit_transform(data)
print(transformed)
```

---

## 연습 문제

### 문제 1: 결측치 처리
다음 데이터의 결측치를 적절히 처리하세요.

```python
df = pd.DataFrame({
    'A': [1, 2, None, 4, 5],
    'B': [None, 'X', 'Y', 'X', None]
})

# 풀이
df['A'] = df['A'].fillna(df['A'].median())
df['B'] = df['B'].fillna(df['B'].mode()[0])
print(df)
```

### 문제 2: 이상치 탐지
IQR 방법으로 이상치를 찾고 제거하세요.

```python
df = pd.DataFrame({
    'value': [10, 12, 11, 13, 100, 11, 12, 10]
})

# 풀이
Q1 = df['value'].quantile(0.25)
Q3 = df['value'].quantile(0.75)
IQR = Q3 - Q1
df_clean = df[(df['value'] >= Q1 - 1.5 * IQR) &
              (df['value'] <= Q3 + 1.5 * IQR)]
print(df_clean)
```

### 문제 3: 인코딩
범주형 변수를 원-핫 인코딩하세요.

```python
df = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'red']
})

# 풀이
df_encoded = pd.get_dummies(df, columns=['color'], prefix='color')
print(df_encoded)
```

---

## 요약

| 기능 | 방법 |
|------|------|
| 결측치 확인 | `isna()`, `isnull()` |
| 결측치 처리 | `dropna()`, `fillna()`, `interpolate()` |
| 이상치 탐지 | IQR, Z-score, 박스플롯 |
| 정규화/표준화 | `MinMaxScaler`, `StandardScaler`, `RobustScaler` |
| 범주형 인코딩 | `LabelEncoder`, `OneHotEncoder`, `get_dummies()` |
| 수치형 변환 | 로그 변환, Box-Cox, 구간화 |
| 날짜 처리 | `to_datetime()`, `dt` 접근자 |
