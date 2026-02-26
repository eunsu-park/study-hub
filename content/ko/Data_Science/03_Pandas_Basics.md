# 3. Pandas 기초

[이전: NumPy 고급](./02_NumPy_Advanced.md) | [다음: Pandas 데이터 조작](./04_Pandas_Data_Manipulation.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 시리즈(Series)와 데이터프레임(DataFrame) 자료구조를 구분하고 다양한 소스에서 각각 생성할 수 있습니다
2. `shape`, `columns`, `dtypes`, 메모리 사용량 등 데이터프레임(DataFrame)의 주요 속성을 설명할 수 있습니다
3. `read_csv`, `read_excel`, `read_json`, `read_sql`을 적용하여 여러 형식의 데이터를 불러올 수 있습니다
4. 레이블 기반(`loc`), 정수 기반(`iloc`), 단일 값(`at`/`iat`) 선택을 구현할 수 있습니다
5. 조건 필터링, 고유값 카운팅, 결측값 탐지를 적용하여 데이터를 탐색할 수 있습니다
6. 데이터프레임(DataFrame)에 산술 연산, 집계 함수, 정렬을 수행할 수 있습니다
7. 열과 행의 추가, 수정, 삭제를 구현할 수 있습니다
8. 텍스트 정제 및 추출을 위한 문자열 접근자 메서드(`.str`)를 적용할 수 있습니다

---

실세계 데이터는 거의 항상 스프레드시트, 데이터베이스 내보내기, CSV 로그와 같은 표 형태로 제공됩니다. Pandas는 이러한 데이터를 불러오고, 탐색하고, 변환하기 위한 강력하고 표현력 있는 인터페이스를 제공합니다. 모든 분석 파이프라인이 데이터 전처리(data wrangling)로 시작되는 만큼, Pandas를 익히는 것은 데이터 과학을 지망하는 누구에게나 가장 큰 영향을 미치는 단일 기술이라 할 수 있습니다.

---

## 1. Pandas 자료구조

### 1.1 Series

Series는 1차원 레이블이 있는 배열입니다.

```python
import pandas as pd
import numpy as np

# 리스트로부터 Series 생성
s = pd.Series([10, 20, 30, 40, 50])
print(s)
# 0    10
# 1    20
# 2    30
# 3    40
# 4    50
# dtype: int64

# 인덱스 지정
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
print(s)
# a    10
# b    20
# c    30

# 딕셔너리로부터 생성
d = {'apple': 100, 'banana': 200, 'cherry': 150}
s = pd.Series(d)
print(s)

# Series 속성
print(s.values)  # 값 배열
print(s.index)   # 인덱스
print(s.dtype)   # 데이터 타입
print(s.name)    # Series 이름
```

### 1.2 DataFrame

DataFrame은 2차원 테이블 형태의 자료구조입니다.

```python
# 딕셔너리로부터 DataFrame 생성
data = {
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['Seoul', 'Busan', 'Incheon']
}
df = pd.DataFrame(data)
print(df)
#       name  age     city
# 0    Alice   25    Seoul
# 1      Bob   30    Busan
# 2  Charlie   35  Incheon

# 리스트의 리스트로 생성
data = [
    ['Alice', 25, 'Seoul'],
    ['Bob', 30, 'Busan'],
    ['Charlie', 35, 'Incheon']
]
df = pd.DataFrame(data, columns=['name', 'age', 'city'])

# NumPy 배열로 생성
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
df = pd.DataFrame(arr, columns=['A', 'B', 'C'])

# 인덱스 지정
df = pd.DataFrame(data,
                  columns=['name', 'age', 'city'],
                  index=['p1', 'p2', 'p3'])
```

### 1.3 DataFrame 속성

```python
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'salary': [50000, 60000, 70000]
})

# 기본 속성
print(df.shape)      # (3, 3)
print(df.columns)    # Index(['name', 'age', 'salary'], dtype='object')
print(df.index)      # RangeIndex(start=0, stop=3, step=1)
print(df.dtypes)     # 각 열의 데이터 타입
print(df.values)     # NumPy 배열
print(df.size)       # 9 (전체 요소 수)
print(len(df))       # 3 (행 수)

# 메모리 사용량
print(df.memory_usage())

# 데이터 요약
print(df.info())
print(df.describe())  # 수치형 열의 통계 요약
```

---

## 2. 데이터 로딩

### 2.1 CSV 파일

```python
# CSV 읽기
df = pd.read_csv('data.csv')

# 옵션 지정
df = pd.read_csv('data.csv',
                 sep=',',           # 구분자
                 header=0,          # 헤더 행 (None이면 없음)
                 index_col=0,       # 인덱스로 사용할 열
                 usecols=['A', 'B'], # 읽을 열 지정
                 dtype={'A': int},   # 데이터 타입 지정
                 na_values=['NA', 'N/A'],  # 결측값으로 처리할 값
                 encoding='utf-8',   # 인코딩
                 nrows=100)          # 읽을 행 수

# 대용량 파일 청크로 읽기
chunks = pd.read_csv('large_data.csv', chunksize=10000)
for chunk in chunks:
    process(chunk)

# CSV 저장
df.to_csv('output.csv', index=False)
```

### 2.2 Excel 파일

```python
# Excel 읽기 (openpyxl 또는 xlrd 필요)
df = pd.read_excel('data.xlsx')

# 시트 지정
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# 여러 시트 읽기
sheets = pd.read_excel('data.xlsx', sheet_name=None)  # 딕셔너리 반환

# Excel 저장
df.to_excel('output.xlsx', index=False, sheet_name='Data')

# 여러 시트 저장
with pd.ExcelWriter('output.xlsx') as writer:
    df1.to_excel(writer, sheet_name='Sheet1')
    df2.to_excel(writer, sheet_name='Sheet2')
```

### 2.3 JSON 파일

```python
# JSON 읽기
df = pd.read_json('data.json')

# JSON 형식 지정
df = pd.read_json('data.json', orient='records')
# orient: 'split', 'records', 'index', 'columns', 'values'

# JSON 저장
df.to_json('output.json', orient='records')

# 줄바꿈으로 구분된 JSON (JSON Lines)
df = pd.read_json('data.jsonl', lines=True)
df.to_json('output.jsonl', orient='records', lines=True)
```

### 2.4 SQL 데이터베이스

```python
import sqlite3
from sqlalchemy import create_engine

# SQLite 연결
conn = sqlite3.connect('database.db')
df = pd.read_sql('SELECT * FROM users', conn)
conn.close()

# SQLAlchemy 엔진 사용
engine = create_engine('postgresql://user:pass@host:5432/db')
df = pd.read_sql('SELECT * FROM users', engine)

# 테이블 읽기
df = pd.read_sql_table('users', engine)

# 쿼리 실행
df = pd.read_sql_query('SELECT * FROM users WHERE age > 30', engine)

# DataFrame을 SQL로 저장
df.to_sql('users', engine, if_exists='replace', index=False)
# if_exists: 'fail', 'replace', 'append'
```

### 2.5 기타 형식

```python
# HTML 테이블
dfs = pd.read_html('https://example.com/table.html')
df = dfs[0]  # 첫 번째 테이블

# 클립보드
df = pd.read_clipboard()

# Parquet (pyarrow 필요)
df = pd.read_parquet('data.parquet')
df.to_parquet('output.parquet')

# Pickle
df = pd.read_pickle('data.pkl')
df.to_pickle('output.pkl')

# HDF5 (tables 필요)
df = pd.read_hdf('data.h5', key='df')
df.to_hdf('output.h5', key='df')
```

---

## 3. 데이터 선택과 접근

### 3.1 열 선택

```python
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['Seoul', 'Busan', 'Incheon']
})

# 단일 열 선택 (Series 반환)
print(df['name'])
print(df.name)  # 속성 접근 (열 이름이 파이썬 식별자일 때)

# 여러 열 선택 (DataFrame 반환)
print(df[['name', 'age']])
```

### 3.2 행 선택

```python
# 슬라이싱
print(df[0:2])  # 처음 2행

# 조건 필터링
print(df[df['age'] > 25])
print(df[df['city'].isin(['Seoul', 'Busan'])])
```

### 3.3 loc - 레이블 기반 선택

```python
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['Seoul', 'Busan', 'Incheon']
}, index=['a', 'b', 'c'])

# 단일 행
print(df.loc['a'])

# 여러 행
print(df.loc[['a', 'c']])

# 행과 열
print(df.loc['a', 'name'])        # 단일 값
print(df.loc['a':'b', 'name':'age'])  # 범위 슬라이싱

# 조건과 함께
print(df.loc[df['age'] > 25, ['name', 'city']])
```

### 3.4 iloc - 정수 기반 선택

```python
# 단일 행
print(df.iloc[0])

# 여러 행
print(df.iloc[[0, 2]])

# 행과 열
print(df.iloc[0, 1])        # 단일 값
print(df.iloc[0:2, 0:2])    # 범위 슬라이싱
print(df.iloc[[0, 2], [0, 2]])  # 특정 위치

# 음수 인덱스
print(df.iloc[-1])  # 마지막 행
```

### 3.5 at과 iat - 단일 값 접근

```python
# at: 레이블 기반 단일 값
print(df.at['a', 'name'])

# iat: 정수 기반 단일 값
print(df.iat[0, 0])

# 값 수정
df.at['a', 'age'] = 26
df.iat[0, 1] = 27
```

---

## 4. 데이터 확인과 탐색

### 4.1 데이터 미리보기

```python
df = pd.DataFrame({
    'A': range(100),
    'B': range(100, 200),
    'C': range(200, 300)
})

# 처음/끝 확인
print(df.head())     # 처음 5행
print(df.head(10))   # 처음 10행
print(df.tail())     # 마지막 5행
print(df.tail(3))    # 마지막 3행

# 랜덤 샘플
print(df.sample(5))  # 랜덤 5행
print(df.sample(frac=0.1))  # 10% 샘플
```

### 4.2 데이터 정보

```python
df = pd.DataFrame({
    'name': ['Alice', 'Bob', None, 'Diana'],
    'age': [25, 30, 35, None],
    'salary': [50000.0, 60000.0, 70000.0, 80000.0]
})

# 기본 정보
print(df.info())

# 출력 예시:
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 4 entries, 0 to 3
# Data columns (total 3 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   name    3 non-null      object
#  1   age     3 non-null      float64
#  2   salary  4 non-null      float64
# dtypes: float64(2), object(1)
# memory usage: 224.0+ bytes

# 통계 요약
print(df.describe())
print(df.describe(include='all'))  # 모든 열 포함
```

### 4.3 고유값과 빈도

```python
df = pd.DataFrame({
    'category': ['A', 'B', 'A', 'C', 'B', 'A', 'A', 'C'],
    'value': [10, 20, 30, 40, 50, 60, 70, 80]
})

# 고유값
print(df['category'].unique())    # ['A' 'B' 'C']
print(df['category'].nunique())   # 3

# 빈도
print(df['category'].value_counts())
# A    4
# B    2
# C    2

# 정규화된 빈도
print(df['category'].value_counts(normalize=True))
```

### 4.4 결측값 확인

```python
df = pd.DataFrame({
    'A': [1, 2, None, 4],
    'B': [None, 2, 3, 4],
    'C': [1, 2, 3, None]
})

# 결측값 확인
print(df.isna())      # 불리언 DataFrame
print(df.isnull())    # isna와 동일

# 결측값 개수
print(df.isna().sum())        # 열별 결측값 수
print(df.isna().sum().sum())  # 전체 결측값 수

# 결측값이 있는 행/열
print(df[df.isna().any(axis=1)])  # 결측값이 있는 행
```

---

## 5. 기본 연산

### 5.1 산술 연산

```python
df = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [10, 20, 30, 40]
})

# 스칼라 연산
print(df + 10)
print(df * 2)
print(df ** 2)

# 열 간 연산
df['C'] = df['A'] + df['B']
df['D'] = df['B'] / df['A']

# DataFrame 간 연산 (인덱스 정렬)
df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]}, index=[0, 1])
df2 = pd.DataFrame({'A': [10, 20], 'B': [30, 40]}, index=[1, 2])
print(df1 + df2)  # 인덱스가 일치하는 부분만 연산

# 결측값 처리하며 연산
print(df1.add(df2, fill_value=0))
```

### 5.2 집계 함수

```python
df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50]
})

# 기본 집계
print(df.sum())      # 열별 합계
print(df.mean())     # 열별 평균
print(df.median())   # 중앙값
print(df.std())      # 표준편차
print(df.var())      # 분산
print(df.min())      # 최솟값
print(df.max())      # 최댓값
print(df.count())    # 비결측값 개수

# 축 지정
print(df.sum(axis=0))  # 열별 (기본값)
print(df.sum(axis=1))  # 행별

# 누적 함수
print(df.cumsum())   # 누적 합
print(df.cumprod())  # 누적 곱
print(df.cummax())   # 누적 최대
print(df.cummin())   # 누적 최소
```

### 5.3 정렬

```python
df = pd.DataFrame({
    'name': ['Charlie', 'Alice', 'Bob'],
    'age': [35, 25, 30],
    'score': [85, 95, 75]
})

# 값 기준 정렬
print(df.sort_values('age'))
print(df.sort_values('age', ascending=False))

# 여러 열 기준
print(df.sort_values(['age', 'score']))
print(df.sort_values(['age', 'score'], ascending=[True, False]))

# 인덱스 정렬
df = df.set_index('name')
print(df.sort_index())
print(df.sort_index(ascending=False))

# 정렬 순서
print(df.rank())  # 순위
```

---

## 6. 데이터 수정

### 6.1 열 추가/수정

```python
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

# 새 열 추가
df['C'] = [7, 8, 9]
df['D'] = df['A'] + df['B']
df['E'] = 10  # 스칼라 값

# assign 메서드 (원본 유지)
df2 = df.assign(F=lambda x: x['A'] * 2,
                G=[10, 20, 30])

# insert (특정 위치에 삽입)
df.insert(1, 'new_col', [100, 200, 300])
```

### 6.2 열 삭제

```python
# drop 메서드
df = df.drop('C', axis=1)
df = df.drop(['D', 'E'], axis=1)

# del 키워드
del df['B']

# pop 메서드 (삭제하고 반환)
col = df.pop('A')
```

### 6.3 행 추가/수정

```python
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

# 행 추가 (concat 사용)
new_row = pd.DataFrame({'A': [4], 'B': [7]})
df = pd.concat([df, new_row], ignore_index=True)

# loc으로 추가
df.loc[len(df)] = [5, 8]

# 행 삭제
df = df.drop(0)  # 인덱스 0 삭제
df = df.drop([1, 2])  # 여러 행 삭제
```

### 6.4 값 수정

```python
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

# 조건에 따른 수정
df.loc[df['A'] > 1, 'B'] = 0

# replace
df['A'] = df['A'].replace(1, 100)
df = df.replace({2: 200, 3: 300})

# where (조건이 False인 곳을 수정)
df['A'] = df['A'].where(df['A'] > 100, 0)

# mask (조건이 True인 곳을 수정)
df['B'] = df['B'].mask(df['B'] < 5, -1)
```

---

## 7. 문자열 처리

Pandas는 `.str` 접근자를 통해 문자열 메서드를 제공합니다.

```python
df = pd.DataFrame({
    'name': ['  Alice  ', 'BOB', 'charlie'],
    'email': ['alice@test.com', 'bob@example.com', 'charlie@test.com']
})

# 대소문자
print(df['name'].str.lower())
print(df['name'].str.upper())
print(df['name'].str.title())
print(df['name'].str.capitalize())

# 공백 제거
print(df['name'].str.strip())
print(df['name'].str.lstrip())
print(df['name'].str.rstrip())

# 문자열 길이
print(df['name'].str.len())

# 문자열 포함 여부
print(df['email'].str.contains('test'))
print(df['name'].str.startswith('A'))
print(df['name'].str.endswith('e'))

# 문자열 분리
print(df['email'].str.split('@'))
print(df['email'].str.split('@').str[0])  # 첫 번째 요소

# 문자열 교체
print(df['email'].str.replace('test', 'example'))

# 정규 표현식
print(df['email'].str.extract(r'@(.+)\.com'))
print(df['email'].str.findall(r'\w+'))
```

---

## 연습 문제

### 문제 1: 데이터 로딩과 탐색
다음 데이터를 DataFrame으로 생성하고 기본 정보를 확인하세요.

```python
data = {
    'product': ['Apple', 'Banana', 'Cherry', 'Date'],
    'price': [1000, 500, 2000, 1500],
    'quantity': [50, 100, 30, 45]
}

# 풀이
df = pd.DataFrame(data)
print(df.info())
print(df.describe())
print(df['price'].mean())  # 평균 가격
```

### 문제 2: 데이터 선택
price가 1000 이상인 제품의 이름과 수량만 선택하세요.

```python
# 풀이
result = df.loc[df['price'] >= 1000, ['product', 'quantity']]
print(result)
```

### 문제 3: 열 추가
총 금액(price * quantity) 열을 추가하세요.

```python
# 풀이
df['total'] = df['price'] * df['quantity']
print(df)
```

---

## 요약

| 기능 | 함수/메서드 |
|------|------------|
| 데이터 로딩 | `pd.read_csv()`, `pd.read_excel()`, `pd.read_json()`, `pd.read_sql()` |
| 데이터 저장 | `to_csv()`, `to_excel()`, `to_json()`, `to_sql()` |
| 열 선택 | `df['col']`, `df[['col1', 'col2']]` |
| 행 선택 | `df.loc[]`, `df.iloc[]`, `df[condition]` |
| 데이터 확인 | `head()`, `tail()`, `info()`, `describe()` |
| 집계 | `sum()`, `mean()`, `count()`, `min()`, `max()` |
| 정렬 | `sort_values()`, `sort_index()` |
| 문자열 | `df['col'].str.method()` |
