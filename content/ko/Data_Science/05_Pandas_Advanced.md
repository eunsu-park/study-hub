# 5. Pandas 고급

[이전: Pandas 데이터 조작](./04_Pandas_Data_Manipulation.md) | [다음: 데이터 전처리](./06_Data_Preprocessing.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `from_tuples`, `from_arrays`, `from_product`를 사용하여 MultiIndex DataFrame을 생성하고 조작할 수 있습니다
2. 계층적 인덱스에 크로스 섹션 선택(`xs`), 레벨 교환, 레벨 기반 집계를 적용할 수 있습니다
3. 날짜 생성, 날짜시간 인덱싱, 날짜 산술 연산을 포함한 시계열 연산을 구현할 수 있습니다
4. 시계열 데이터에 리샘플링(Resampling), 롤링 윈도우(Rolling Window), 지수 가중 이동 평균(EWMA)을 적용할 수 있습니다
5. 범주형(Categorical) 데이터 타입을 설명하고 문자열 열 대비 메모리 절약 효과를 구현할 수 있습니다
6. 고급 `.str` 접근자 메서드를 사용하여 정규식 기반 문자열 추출, 분리, 포맷팅을 적용할 수 있습니다
7. 벡터화 연산, `eval`, `query`, 청크 처리를 사용한 성능 최적화를 구현할 수 있습니다
8. `pipe`와 메서드 체이닝을 사용하여 데이터 변환 파이프라인을 구축할 수 있습니다

---

데이터셋이 복잡해질수록 기본적인 선택과 집계를 넘어서는 도구가 필요합니다. MultiIndex 구조는 계층적 데이터를 자연스럽게 모델링하고, 시계열 연산은 시간적 패턴을 손쉽게 처리하며, 성능 최적화 기법은 수백만 행의 데이터에서도 분석 속도를 유지해 줍니다. 이러한 Pandas 고급 기능은 초급 사용자와 숙련된 데이터 실무자를 구분하는 핵심 역량입니다.

---

## 1. 멀티인덱스 (MultiIndex)

### 1.1 멀티인덱스 생성

```python
import pandas as pd
import numpy as np

# 튜플 리스트로 생성
arrays = [
    ['A', 'A', 'B', 'B'],
    [1, 2, 1, 2]
]
tuples = list(zip(*arrays))
index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])

s = pd.Series([10, 20, 30, 40], index=index)
print(s)
# first  second
# A      1         10
#        2         20
# B      1         30
#        2         40

# from_arrays
index = pd.MultiIndex.from_arrays(arrays, names=['first', 'second'])

# from_product (카르테시안 곱)
index = pd.MultiIndex.from_product(
    [['A', 'B'], [1, 2, 3]],
    names=['letter', 'number']
)
print(index)

# DataFrame에 적용
df = pd.DataFrame({
    'value': [10, 20, 30, 40, 50, 60]
}, index=index)
print(df)
```

### 1.2 멀티인덱스 DataFrame

```python
# 열에도 멀티인덱스
col_index = pd.MultiIndex.from_product(
    [['2023', '2024'], ['Q1', 'Q2']],
    names=['year', 'quarter']
)
row_index = pd.MultiIndex.from_product(
    [['Sales', 'IT'], ['Seoul', 'Busan']],
    names=['dept', 'city']
)

data = np.random.randint(100, 1000, (4, 4))
df = pd.DataFrame(data, index=row_index, columns=col_index)
print(df)
```

### 1.3 멀티인덱스 선택

```python
df = pd.DataFrame({
    'year': [2022, 2022, 2023, 2023, 2022, 2023],
    'quarter': ['Q1', 'Q2', 'Q1', 'Q2', 'Q1', 'Q1'],
    'department': ['Sales', 'Sales', 'IT', 'IT', 'IT', 'Sales'],
    'revenue': [100, 150, 200, 250, 180, 160]
})
df = df.set_index(['year', 'quarter', 'department'])
print(df)

# 단일 레벨 선택
print(df.loc[2022])
print(df.loc[(2022, 'Q1')])
print(df.loc[(2022, 'Q1', 'Sales')])

# xs 메서드 (크로스 섹션)
print(df.xs('Q1', level='quarter'))
print(df.xs('Sales', level='department'))
print(df.xs((2022, 'Sales'), level=['year', 'department']))

# 슬라이싱
print(df.loc[2022:2023])
print(df.loc[(2022, 'Q1'):(2023, 'Q1')])
```

### 1.4 멀티인덱스 조작

```python
df = pd.DataFrame({
    'year': [2022, 2022, 2023, 2023],
    'quarter': ['Q1', 'Q2', 'Q1', 'Q2'],
    'revenue': [100, 150, 200, 250]
}).set_index(['year', 'quarter'])

# 레벨 교환
print(df.swaplevel())

# 레벨 정렬
df_unsorted = df.iloc[[2, 0, 3, 1]]
print(df_unsorted.sort_index())
print(df_unsorted.sort_index(level=1))

# 인덱스 리셋
print(df.reset_index())
print(df.reset_index(level='quarter'))

# 레벨 이름 변경
df.index = df.index.set_names(['연도', '분기'])
print(df)

# 레벨 값 변경
df.index = df.index.set_levels([['2022년', '2023년'], ['1분기', '2분기']])
```

### 1.5 멀티인덱스 집계

```python
df = pd.DataFrame({
    'year': [2022, 2022, 2022, 2023, 2023, 2023],
    'quarter': ['Q1', 'Q1', 'Q2', 'Q1', 'Q1', 'Q2'],
    'department': ['Sales', 'IT', 'Sales', 'Sales', 'IT', 'Sales'],
    'revenue': [100, 150, 120, 200, 180, 220]
}).set_index(['year', 'quarter', 'department'])

# 레벨별 합계
print(df.groupby(level='year').sum())
print(df.groupby(level=['year', 'quarter']).sum())

# unstack으로 피벗
print(df.unstack(level='department'))
print(df.unstack(level=['quarter', 'department']))

# stack으로 역피벗
df_wide = df.unstack(level='department')
print(df_wide.stack())
```

---

## 2. 시계열 데이터

### 2.1 날짜/시간 생성

```python
# Timestamp
ts = pd.Timestamp('2023-01-15')
ts = pd.Timestamp('2023-01-15 10:30:00')
ts = pd.Timestamp(year=2023, month=1, day=15, hour=10)

# to_datetime
dates = pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01'])
dates = pd.to_datetime(['01/15/2023', '02/15/2023'], format='%m/%d/%Y')

# date_range
dates = pd.date_range('2023-01-01', periods=10, freq='D')
dates = pd.date_range('2023-01-01', '2023-12-31', freq='M')  # 월말
dates = pd.date_range('2023-01-01', '2023-12-31', freq='MS')  # 월초
dates = pd.date_range('2023-01-01', periods=5, freq='W-MON')  # 매주 월요일

# 주요 freq 옵션
# 'D': 일, 'W': 주, 'M': 월말, 'MS': 월초
# 'Q': 분기말, 'QS': 분기초, 'Y': 연말, 'YS': 연초
# 'H': 시간, 'T' or 'min': 분, 'S': 초
# 'B': 영업일, 'BM': 영업일 월말

# period_range (기간)
periods = pd.period_range('2023-01', periods=12, freq='M')
print(periods)
```

### 2.2 시계열 인덱싱

```python
# DatetimeIndex를 가진 Series
dates = pd.date_range('2023-01-01', periods=365, freq='D')
ts = pd.Series(np.random.randn(365), index=dates)

# 문자열로 선택
print(ts['2023-03-15'])
print(ts['2023-03'])  # 3월 전체
print(ts['2023'])     # 2023년 전체

# 범위 선택
print(ts['2023-03-01':'2023-03-10'])
print(ts['2023-03':'2023-06'])

# loc 사용
print(ts.loc['2023-03-15'])
print(ts.loc['2023-03'])
```

### 2.3 날짜/시간 속성

```python
df = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=10, freq='D')
})

# dt 접근자
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek  # 0=월요일
df['day_name'] = df['date'].dt.day_name()
df['month_name'] = df['date'].dt.month_name()
df['quarter'] = df['date'].dt.quarter
df['is_month_end'] = df['date'].dt.is_month_end
df['is_month_start'] = df['date'].dt.is_month_start

print(df)
```

### 2.4 시간 연산

```python
# Timedelta
td = pd.Timedelta('1 days')
td = pd.Timedelta(days=1, hours=2, minutes=30)

# 날짜 연산
dates = pd.date_range('2023-01-01', periods=5, freq='D')
print(dates + pd.Timedelta('1 days'))
print(dates + pd.DateOffset(months=1))

# 날짜 차이
df = pd.DataFrame({
    'start': pd.to_datetime(['2023-01-01', '2023-02-15']),
    'end': pd.to_datetime(['2023-01-10', '2023-03-20'])
})
df['duration'] = df['end'] - df['start']
df['days'] = df['duration'].dt.days

# DateOffset
from pandas.tseries.offsets import MonthEnd, BDay

date = pd.Timestamp('2023-01-15')
print(date + MonthEnd())  # 월말
print(date + BDay(5))     # 5 영업일 후
```

### 2.5 리샘플링

```python
dates = pd.date_range('2023-01-01', periods=100, freq='D')
ts = pd.Series(np.random.randn(100), index=dates)

# 다운샘플링 (더 큰 간격으로)
print(ts.resample('W').mean())    # 주간 평균
print(ts.resample('M').sum())     # 월간 합계
print(ts.resample('M').agg(['mean', 'std', 'min', 'max']))

# 업샘플링 (더 작은 간격으로)
monthly = pd.Series([100, 110, 120],
                    index=pd.date_range('2023-01-01', periods=3, freq='M'))
print(monthly.resample('D').ffill())   # 앞의 값으로 채움
print(monthly.resample('D').bfill())   # 뒤의 값으로 채움
print(monthly.resample('D').interpolate())  # 보간

# OHLC 집계
print(ts.resample('W').ohlc())  # Open, High, Low, Close
```

### 2.6 이동 윈도우

```python
dates = pd.date_range('2023-01-01', periods=30, freq='D')
ts = pd.Series(np.random.randn(30).cumsum(), index=dates)

# 이동 평균
print(ts.rolling(window=7).mean())

# 다양한 집계
print(ts.rolling(window=7).std())
print(ts.rolling(window=7).min())
print(ts.rolling(window=7).max())
print(ts.rolling(window=7).sum())

# 중심 이동 평균
print(ts.rolling(window=7, center=True).mean())

# 지수 가중 이동 평균 (EWMA)
print(ts.ewm(span=7).mean())
print(ts.ewm(alpha=0.3).mean())

# 확장 윈도우 (처음부터 현재까지)
print(ts.expanding().mean())  # 누적 평균
print(ts.expanding().sum())   # 누적 합
```

### 2.7 시간대 처리

```python
# 시간대 지정
ts = pd.Timestamp('2023-01-15 10:00', tz='Asia/Seoul')
print(ts)

dates = pd.date_range('2023-01-01', periods=5, freq='D', tz='UTC')
print(dates)

# 시간대 변환
ts_utc = pd.Timestamp('2023-01-15 10:00', tz='UTC')
ts_seoul = ts_utc.tz_convert('Asia/Seoul')
print(ts_seoul)

# 시간대 지역화
ts_naive = pd.Timestamp('2023-01-15 10:00')
ts_localized = ts_naive.tz_localize('Asia/Seoul')
print(ts_localized)

# Series/DataFrame
s = pd.Series([1, 2, 3], index=pd.date_range('2023-01-01', periods=3, freq='D'))
s_utc = s.tz_localize('UTC')
s_seoul = s_utc.tz_convert('Asia/Seoul')
```

---

## 3. 범주형 데이터

### 3.1 Categorical 타입

```python
# Categorical 생성
cat = pd.Categorical(['a', 'b', 'c', 'a', 'b'])
print(cat)
print(cat.categories)
print(cat.codes)

# 순서 지정
cat = pd.Categorical(['low', 'medium', 'high', 'low'],
                     categories=['low', 'medium', 'high'],
                     ordered=True)
print(cat)
print(cat.min())  # low
print(cat.max())  # high

# DataFrame에서 사용
df = pd.DataFrame({
    'grade': pd.Categorical(['A', 'B', 'A', 'C', 'B'],
                            categories=['C', 'B', 'A'],
                            ordered=True)
})
print(df.sort_values('grade'))
```

### 3.2 타입 변환

```python
df = pd.DataFrame({
    'category': ['apple', 'banana', 'apple', 'cherry', 'banana']
})

# category 타입으로 변환
df['category'] = df['category'].astype('category')
print(df['category'].dtype)
print(df['category'].cat.categories)

# 메모리 절약 확인
df_str = pd.DataFrame({'col': ['A'] * 1000000})
df_cat = pd.DataFrame({'col': pd.Categorical(['A'] * 1000000)})
print(f"문자열: {df_str.memory_usage(deep=True).sum():,} bytes")
print(f"카테고리: {df_cat.memory_usage(deep=True).sum():,} bytes")
```

### 3.3 범주 조작

```python
s = pd.Series(['a', 'b', 'c', 'a', 'b']).astype('category')

# 카테고리 추가
s = s.cat.add_categories(['d', 'e'])
print(s.cat.categories)

# 카테고리 제거
s = s.cat.remove_categories(['e'])

# 카테고리 이름 변경
s = s.cat.rename_categories({'a': 'A', 'b': 'B', 'c': 'C'})
print(s)

# 사용되지 않는 카테고리 제거
s = s.cat.remove_unused_categories()

# 카테고리 재정렬
s = s.cat.reorder_categories(['C', 'B', 'A'])
```

---

## 4. 문자열 고급

### 4.1 정규 표현식

```python
df = pd.DataFrame({
    'text': ['apple 123', 'banana 456', 'cherry 789', 'date'],
    'email': ['test@example.com', 'user@domain.org', 'invalid', 'admin@site.net']
})

# 패턴 매칭
print(df['text'].str.contains(r'\d+', regex=True))

# 패턴 추출
print(df['text'].str.extract(r'(\w+)\s(\d+)'))

# 모든 매치 추출
print(df['text'].str.findall(r'\d'))

# 이메일 도메인 추출
print(df['email'].str.extract(r'@(.+)$'))

# 교체
print(df['text'].str.replace(r'\d+', 'NUM', regex=True))
```

### 4.2 문자열 분리와 결합

```python
df = pd.DataFrame({
    'full_name': ['John Smith', 'Jane Doe', 'Bob Johnson']
})

# 분리
names = df['full_name'].str.split(' ', expand=True)
names.columns = ['first', 'last']
print(names)

# 결합
df['formatted'] = df['full_name'].str.replace(' ', ', ')

# 문자열 결합 (Series)
s = pd.Series(['a', 'b', 'c'])
print(s.str.cat(sep='-'))  # 'a-b-c'

# 두 Series 결합
s1 = pd.Series(['a', 'b', 'c'])
s2 = pd.Series(['1', '2', '3'])
print(s1.str.cat(s2, sep='-'))  # ['a-1', 'b-2', 'c-3']
```

### 4.3 문자열 포맷팅

```python
df = pd.DataFrame({
    'name': ['Alice', 'Bob'],
    'score': [95.5, 87.3]
})

# 포맷팅
df['formatted'] = df['name'] + ': ' + df['score'].astype(str)
df['formatted2'] = df.apply(lambda x: f"{x['name']}: {x['score']:.1f}", axis=1)
print(df)

# 패딩
s = pd.Series(['1', '22', '333'])
print(s.str.pad(5, side='left', fillchar='0'))  # ['00001', '00022', '00333']
print(s.str.zfill(5))  # ['00001', '00022', '00333']
print(s.str.center(7, '*'))  # ['***1***', '**22***', '*333**']
```

---

## 5. 성능 최적화

### 5.1 데이터 타입 최적화

```python
def reduce_mem_usage(df):
    """DataFrame 메모리 사용량 최적화"""
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)

    end_mem = df.memory_usage().sum() / 1024**2
    print(f'메모리 사용량: {start_mem:.2f} MB -> {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% 감소)')

    return df
```

### 5.2 벡터화 연산

```python
import time

df = pd.DataFrame({
    'A': np.random.randn(100000),
    'B': np.random.randn(100000)
})

# 나쁜 예: iterrows
start = time.time()
result = []
for idx, row in df.iterrows():
    result.append(row['A'] + row['B'])
print(f"iterrows: {time.time() - start:.4f}초")

# 좋은 예: 벡터화
start = time.time()
result = df['A'] + df['B']
print(f"벡터화: {time.time() - start:.4f}초")

# apply vs 벡터화
start = time.time()
result = df.apply(lambda x: x['A'] + x['B'], axis=1)
print(f"apply: {time.time() - start:.4f}초")
```

### 5.3 eval과 query

```python
df = pd.DataFrame({
    'A': np.random.randn(100000),
    'B': np.random.randn(100000),
    'C': np.random.randn(100000)
})

# eval 사용 (복잡한 수식)
df['D'] = pd.eval('df.A + df.B * df.C')

# 더 복잡한 계산
result = df.eval('(A + B) / (C + 1)')

# query와 함께
result = df.query('A > 0 and B < 0')

# 지역 변수 사용
threshold = 0.5
result = df.query('A > @threshold')
```

### 5.4 대용량 데이터 처리

```python
# 청크 단위로 처리
def process_large_file(filename, chunksize=10000):
    results = []
    for chunk in pd.read_csv(filename, chunksize=chunksize):
        # 각 청크 처리
        processed = chunk[chunk['value'] > 0].groupby('category')['value'].sum()
        results.append(processed)

    return pd.concat(results).groupby(level=0).sum()

# 특정 열만 읽기
df = pd.read_csv('large_file.csv', usecols=['col1', 'col2', 'col3'])

# 데이터 타입 지정하여 읽기
dtypes = {'col1': 'int32', 'col2': 'float32', 'col3': 'category'}
df = pd.read_csv('large_file.csv', dtype=dtypes)
```

---

## 6. 파이프라인

### 6.1 pipe 메서드

```python
def remove_outliers(df, column, n_std=3):
    """이상치 제거"""
    mean = df[column].mean()
    std = df[column].std()
    return df[(df[column] - mean).abs() <= n_std * std]

def add_features(df):
    """특성 추가"""
    df = df.copy()
    df['log_value'] = np.log1p(df['value'])
    df['squared'] = df['value'] ** 2
    return df

def normalize(df, columns):
    """정규화"""
    df = df.copy()
    for col in columns:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df

# 파이프라인 실행
df = pd.DataFrame({
    'value': np.random.randn(1000) * 10 + 50,
    'category': np.random.choice(['A', 'B', 'C'], 1000)
})

result = (df
    .pipe(remove_outliers, 'value')
    .pipe(add_features)
    .pipe(normalize, ['value', 'log_value'])
)
print(result.head())
```

### 6.2 메서드 체이닝

```python
df = pd.DataFrame({
    'name': ['  Alice  ', 'BOB', 'charlie', None],
    'age': [25, 30, 35, 28],
    'salary': [50000, 60000, 70000, 55000]
})

result = (df
    .dropna()
    .assign(name=lambda x: x['name'].str.strip().str.title())
    .query('age >= 25')
    .sort_values('salary', ascending=False)
    .reset_index(drop=True)
)
print(result)
```

---

## 연습 문제

### 문제 1: 멀티인덱스 활용
연도별, 분기별 매출 데이터에서 2023년 데이터만 선택하세요.

```python
df = pd.DataFrame({
    'year': [2022, 2022, 2023, 2023, 2022, 2023],
    'quarter': ['Q1', 'Q2', 'Q1', 'Q2', 'Q3', 'Q3'],
    'sales': [100, 120, 150, 180, 110, 200]
}).set_index(['year', 'quarter'])

# 풀이
print(df.loc[2023])
```

### 문제 2: 시계열 리샘플링
일별 데이터를 주간 평균으로 리샘플링하세요.

```python
dates = pd.date_range('2023-01-01', periods=30, freq='D')
ts = pd.Series(np.random.randn(30), index=dates)

# 풀이
weekly_avg = ts.resample('W').mean()
print(weekly_avg)
```

### 문제 3: 이동 평균
7일 이동 평균을 계산하고 원본과 함께 표시하세요.

```python
dates = pd.date_range('2023-01-01', periods=30, freq='D')
ts = pd.Series(np.random.randn(30).cumsum(), index=dates)

# 풀이
df = pd.DataFrame({
    'original': ts,
    'ma_7': ts.rolling(window=7).mean()
})
print(df)
```

---

## 요약

| 기능 | 함수/메서드 |
|------|------------|
| 멀티인덱스 | `MultiIndex.from_*()`, `xs()`, `swaplevel()`, `stack()`, `unstack()` |
| 시계열 | `to_datetime()`, `date_range()`, `resample()`, `rolling()`, `ewm()` |
| 범주형 | `Categorical()`, `astype('category')`, `cat` 접근자 |
| 문자열 | `str` 접근자, 정규표현식, `extract()`, `split()` |
| 성능 | 벡터화 연산, `eval()`, `query()`, 청크 처리 |
| 파이프라인 | `pipe()`, 메서드 체이닝 |
