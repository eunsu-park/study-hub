# 4. Pandas 데이터 조작

[이전: Pandas 기초](./03_Pandas_Basics.md) | [다음: Pandas 고급](./05_Pandas_Advanced.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 조건 필터링, 문자열 기반 필터링, `query` 메서드를 적용하여 데이터프레임(DataFrame)에서 행을 선택할 수 있습니다
2. 오름차순/내림차순을 자유롭게 지정하며 단일 열 및 다중 열 정렬을 구현할 수 있습니다
3. 집계(aggregation), 변환(transformation), 그룹 필터링을 포함한 그룹바이(GroupBy) 연산을 수행할 수 있습니다
4. `agg` 메서드에 이름이 지정된 집계(named aggregation)를 적용하여 그룹별 여러 통계를 계산할 수 있습니다
5. `merge`와 `concat`을 사용하여 SQL 스타일 조인(내부, 왼쪽, 오른쪽, 외부)을 구현할 수 있습니다
6. 데이터 재구조화를 위한 `pivot`, `pivot_table`, `melt`, `stack`, `unstack`을 비교하고 적용할 수 있습니다
7. `duplicated`와 `drop_duplicates`를 사용하여 중복 행을 식별하고 제거할 수 있습니다
8. 범주형 변수 간의 관계를 요약하기 위해 교차 집계(cross-tabulation)를 적용할 수 있습니다

---

데이터를 불러오는 것은 시작에 불과합니다. 진정한 분석 능력은 데이터를 재구조화하고, 결합하고, 요약하는 데서 나옵니다. 행 필터링, 카테고리별 그룹화, 여러 테이블 조인, 넓은 형식(wide)과 긴 형식(long) 간의 피벗은 데이터 분석가가 매일 사용하는 연산들입니다. 이러한 조작 기법을 익히면 복잡한 비즈니스 질문에 단 몇 줄의 코드로 답할 수 있습니다.

---

## 1. 데이터 필터링

### 1.1 조건 필터링

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'age': [25, 30, 35, 28, 32],
    'department': ['Sales', 'IT', 'IT', 'HR', 'Sales'],
    'salary': [50000, 60000, 70000, 55000, 65000]
})

# 단일 조건
print(df[df['age'] > 30])
print(df[df['department'] == 'IT'])

# 복합 조건 (AND: &, OR: |)
print(df[(df['age'] > 25) & (df['salary'] >= 60000)])
print(df[(df['department'] == 'IT') | (df['department'] == 'Sales')])

# NOT 조건
print(df[~(df['department'] == 'HR')])

# 범위 조건
print(df[df['age'].between(25, 30)])  # 25 <= age <= 30
```

### 1.2 isin을 이용한 필터링

```python
# 여러 값 중 하나와 일치
departments = ['IT', 'Sales']
print(df[df['department'].isin(departments)])

# 일치하지 않는 경우
print(df[~df['department'].isin(departments)])
```

### 1.3 문자열 조건

```python
# 문자열 포함
print(df[df['name'].str.contains('a', case=False)])

# 시작/끝 문자열
print(df[df['name'].str.startswith('A')])
print(df[df['name'].str.endswith('e')])

# 정규 표현식
print(df[df['name'].str.match(r'^[A-C]')])  # A, B, C로 시작
```

### 1.4 query 메서드

```python
# SQL 스타일 쿼리
print(df.query('age > 30'))
print(df.query('department == "IT"'))
print(df.query('age > 25 and salary >= 60000'))

# 변수 사용
min_age = 30
print(df.query('age >= @min_age'))

# 인덱스 참조
df_indexed = df.set_index('name')
print(df_indexed.query('index == "Alice"'))
```

### 1.5 결측값 필터링

```python
df_with_na = pd.DataFrame({
    'A': [1, 2, None, 4],
    'B': [None, 2, 3, 4],
    'C': [1, 2, 3, None]
})

# 결측값이 있는 행
print(df_with_na[df_with_na['A'].isna()])

# 결측값이 없는 행
print(df_with_na[df_with_na['A'].notna()])

# 모든 열에 결측값이 없는 행
print(df_with_na.dropna())

# 특정 열에 결측값이 없는 행
print(df_with_na.dropna(subset=['A', 'B']))
```

---

## 2. 정렬

### 2.1 값 기준 정렬

```python
df = pd.DataFrame({
    'name': ['Charlie', 'Alice', 'Bob', 'Diana'],
    'age': [35, 25, 30, 25],
    'score': [85, 95, 75, 90]
})

# 단일 열 정렬
print(df.sort_values('age'))
print(df.sort_values('age', ascending=False))

# 여러 열 정렬
print(df.sort_values(['age', 'score']))
print(df.sort_values(['age', 'score'], ascending=[True, False]))

# 결측값 위치
df_na = df.copy()
df_na.loc[0, 'age'] = None
print(df_na.sort_values('age', na_position='first'))  # 결측값 맨 앞
print(df_na.sort_values('age', na_position='last'))   # 결측값 맨 뒤

# inplace 정렬
df.sort_values('age', inplace=True)
```

### 2.2 인덱스 정렬

```python
df = pd.DataFrame({
    'value': [10, 20, 30, 40]
}, index=['d', 'b', 'c', 'a'])

# 인덱스 오름차순
print(df.sort_index())

# 인덱스 내림차순
print(df.sort_index(ascending=False))

# 열 인덱스 정렬
df_wide = pd.DataFrame({
    'C': [1, 2],
    'A': [3, 4],
    'B': [5, 6]
})
print(df_wide.sort_index(axis=1))
```

### 2.3 순위 매기기

```python
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'score': [85, 95, 85, 90]
})

# 기본 순위 (동점 시 평균)
df['rank'] = df['score'].rank(ascending=False)
print(df)

# 순위 방법
# method='average': 평균 순위 (기본값)
# method='min': 최소 순위
# method='max': 최대 순위
# method='first': 먼저 나온 순서대로
# method='dense': 밀집 순위 (간격 없음)

df['rank_min'] = df['score'].rank(ascending=False, method='min')
df['rank_dense'] = df['score'].rank(ascending=False, method='dense')
print(df)
```

---

## 3. 그룹화 (GroupBy)

### 3.1 기본 그룹화

```python
df = pd.DataFrame({
    'department': ['Sales', 'IT', 'IT', 'HR', 'Sales', 'HR'],
    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank'],
    'salary': [50000, 60000, 70000, 55000, 65000, 52000],
    'bonus': [5000, 8000, 10000, 6000, 7000, 5500]
})

# 그룹화
grouped = df.groupby('department')

# 그룹 확인
print(grouped.groups)
print(grouped.ngroups)  # 그룹 수

# 특정 그룹 가져오기
print(grouped.get_group('IT'))
```

### 3.2 집계 함수

```python
# 단일 집계
print(df.groupby('department')['salary'].mean())
print(df.groupby('department')['salary'].sum())

# 여러 열 집계
print(df.groupby('department')[['salary', 'bonus']].mean())

# 여러 집계 함수
print(df.groupby('department')['salary'].agg(['mean', 'sum', 'count']))

# 사용자 정의 함수
print(df.groupby('department')['salary'].agg(lambda x: x.max() - x.min()))
```

### 3.3 agg 메서드

```python
# 열마다 다른 집계
agg_result = df.groupby('department').agg({
    'salary': ['mean', 'max'],
    'bonus': 'sum',
    'name': 'count'
})
print(agg_result)

# 이름 지정
agg_result = df.groupby('department').agg(
    avg_salary=('salary', 'mean'),
    max_salary=('salary', 'max'),
    total_bonus=('bonus', 'sum'),
    employee_count=('name', 'count')
)
print(agg_result)
```

### 3.4 transform과 apply

```python
# transform: 원본과 같은 크기의 결과
df['dept_avg_salary'] = df.groupby('department')['salary'].transform('mean')
print(df)

# 그룹 내 정규화
df['salary_normalized'] = df.groupby('department')['salary'].transform(
    lambda x: (x - x.mean()) / x.std()
)

# apply: 유연한 그룹 연산
def top_n(group, n=2, column='salary'):
    return group.nlargest(n, column)

print(df.groupby('department').apply(top_n, n=1))
```

### 3.5 여러 열로 그룹화

```python
df = pd.DataFrame({
    'year': [2022, 2022, 2023, 2023, 2022, 2023],
    'quarter': ['Q1', 'Q2', 'Q1', 'Q2', 'Q1', 'Q1'],
    'sales': [100, 150, 120, 180, 110, 130]
})

# 다중 열 그룹화
print(df.groupby(['year', 'quarter'])['sales'].sum())

# 결과를 DataFrame으로
print(df.groupby(['year', 'quarter'])['sales'].sum().reset_index())

# 언스택
print(df.groupby(['year', 'quarter'])['sales'].sum().unstack())
```

### 3.6 필터링

```python
df = pd.DataFrame({
    'department': ['Sales', 'IT', 'IT', 'HR', 'Sales'],
    'salary': [50000, 60000, 70000, 55000, 65000]
})

# 조건을 만족하는 그룹만 필터링
# 평균 급여가 55000 이상인 부서
result = df.groupby('department').filter(lambda x: x['salary'].mean() >= 55000)
print(result)
```

---

## 4. 데이터 병합

### 4.1 merge (SQL 스타일 조인)

```python
# 예제 데이터
employees = pd.DataFrame({
    'emp_id': [1, 2, 3, 4],
    'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'dept_id': [10, 20, 10, 30]
})

departments = pd.DataFrame({
    'dept_id': [10, 20, 40],
    'dept_name': ['Sales', 'IT', 'Marketing']
})

# 내부 조인 (기본값)
result = pd.merge(employees, departments, on='dept_id')
print(result)

# 왼쪽 조인
result = pd.merge(employees, departments, on='dept_id', how='left')
print(result)

# 오른쪽 조인
result = pd.merge(employees, departments, on='dept_id', how='right')
print(result)

# 외부 조인
result = pd.merge(employees, departments, on='dept_id', how='outer')
print(result)
```

### 4.2 다른 열 이름으로 조인

```python
employees = pd.DataFrame({
    'emp_id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie'],
    'department': [10, 20, 10]
})

departments = pd.DataFrame({
    'id': [10, 20],
    'dept_name': ['Sales', 'IT']
})

# 다른 열 이름으로 조인
result = pd.merge(employees, departments,
                  left_on='department', right_on='id')
print(result)
```

### 4.3 인덱스 기반 조인

```python
employees = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'salary': [50000, 60000, 70000]
}, index=[1, 2, 3])

bonuses = pd.DataFrame({
    'bonus': [5000, 8000, 10000]
}, index=[1, 2, 4])

# 인덱스로 조인
result = pd.merge(employees, bonuses, left_index=True, right_index=True, how='outer')
print(result)

# join 메서드 (인덱스 기반)
result = employees.join(bonuses, how='outer')
print(result)
```

### 4.4 concat (연결)

```python
df1 = pd.DataFrame({
    'A': [1, 2],
    'B': [3, 4]
})

df2 = pd.DataFrame({
    'A': [5, 6],
    'B': [7, 8]
})

# 수직 연결 (행 방향)
result = pd.concat([df1, df2])
print(result)

# 인덱스 재설정
result = pd.concat([df1, df2], ignore_index=True)
print(result)

# 수평 연결 (열 방향)
result = pd.concat([df1, df2], axis=1)
print(result)

# 키 추가
result = pd.concat([df1, df2], keys=['first', 'second'])
print(result)
```

### 4.5 append (행 추가) - deprecated in pandas 2.0

```python
# concat 사용 권장
new_row = pd.DataFrame({'A': [9], 'B': [10]})
result = pd.concat([df1, new_row], ignore_index=True)
```

---

## 5. 피벗과 멜트

### 5.1 pivot

```python
df = pd.DataFrame({
    'date': ['2023-01', '2023-01', '2023-02', '2023-02'],
    'city': ['Seoul', 'Busan', 'Seoul', 'Busan'],
    'sales': [100, 80, 120, 90]
})

# 피벗 테이블
pivot = df.pivot(index='date', columns='city', values='sales')
print(pivot)
#          Busan  Seoul
# date
# 2023-01     80    100
# 2023-02     90    120
```

### 5.2 pivot_table

```python
df = pd.DataFrame({
    'date': ['2023-01', '2023-01', '2023-01', '2023-02'],
    'city': ['Seoul', 'Seoul', 'Busan', 'Seoul'],
    'category': ['A', 'B', 'A', 'A'],
    'sales': [100, 150, 80, 120]
})

# 집계 함수 적용
pivot = pd.pivot_table(df, values='sales', index='date',
                       columns='city', aggfunc='sum')
print(pivot)

# 여러 집계 함수
pivot = pd.pivot_table(df, values='sales', index='date',
                       columns='city', aggfunc=['sum', 'mean'])
print(pivot)

# 여러 인덱스
pivot = pd.pivot_table(df, values='sales',
                       index=['date', 'category'],
                       columns='city',
                       aggfunc='sum',
                       fill_value=0)
print(pivot)

# 마진 추가
pivot = pd.pivot_table(df, values='sales', index='date',
                       columns='city', aggfunc='sum', margins=True)
print(pivot)
```

### 5.3 melt (언피벗)

```python
df_wide = pd.DataFrame({
    'name': ['Alice', 'Bob'],
    'math': [90, 85],
    'english': [80, 95],
    'science': [85, 90]
})

# Wide → Long 변환
df_long = pd.melt(df_wide,
                  id_vars=['name'],
                  value_vars=['math', 'english', 'science'],
                  var_name='subject',
                  value_name='score')
print(df_long)
#     name  subject  score
# 0  Alice     math     90
# 1    Bob     math     85
# 2  Alice  english     80
# 3    Bob  english     95
# 4  Alice  science     85
# 5    Bob  science     90
```

### 5.4 stack과 unstack

```python
df = pd.DataFrame({
    'A': [1, 2],
    'B': [3, 4]
}, index=['x', 'y'])

# stack: 열을 행으로
stacked = df.stack()
print(stacked)
# x  A    1
#    B    3
# y  A    2
#    B    4

# unstack: 행을 열로
unstacked = stacked.unstack()
print(unstacked)
```

---

## 6. 중복 처리

```python
df = pd.DataFrame({
    'A': [1, 1, 2, 2, 3],
    'B': ['a', 'a', 'b', 'c', 'c'],
    'C': [10, 10, 20, 30, 40]
})

# 중복 확인
print(df.duplicated())
print(df.duplicated(subset=['A', 'B']))
print(df[df.duplicated(keep=False)])  # 모든 중복 행

# 중복 개수
print(df.duplicated().sum())

# 중복 제거
print(df.drop_duplicates())
print(df.drop_duplicates(subset=['A']))
print(df.drop_duplicates(subset=['A'], keep='last'))  # 마지막 유지
print(df.drop_duplicates(subset=['A'], keep=False))   # 모든 중복 제거
```

---

## 7. 교차 테이블

```python
df = pd.DataFrame({
    'gender': ['M', 'F', 'M', 'F', 'M', 'F'],
    'department': ['Sales', 'IT', 'IT', 'Sales', 'HR', 'IT'],
    'salary': [50000, 60000, 55000, 58000, 52000, 62000]
})

# 빈도 교차 테이블
ct = pd.crosstab(df['gender'], df['department'])
print(ct)

# 마진 추가
ct = pd.crosstab(df['gender'], df['department'], margins=True)
print(ct)

# 정규화
ct = pd.crosstab(df['gender'], df['department'], normalize=True)
print(ct)

# 집계 함수 적용
ct = pd.crosstab(df['gender'], df['department'],
                 values=df['salary'], aggfunc='mean')
print(ct)
```

---

## 8. Pandas 2.0+ PyArrow 백엔드

Pandas 2.0(2023년 4월 릴리스)에서 Apache Arrow를 백엔드 dtype 시스템으로 일급 지원(First-class support)하기 시작했습니다. 이것은 Pandas가 만들어진 이래 가장 큰 내부 변경으로, 지원되는 타입에 대해 레거시 NumPy 기반 메모리 레이아웃을 Arrow의 컬럼형 포맷으로 대체합니다. 대규모 데이터셋을 효율적으로 다루려면 PyArrow 백엔드를 언제, 어떻게 사용하는지 이해하는 것이 필수적입니다.

### 8.1 왜 PyArrow인가?

전통적인 Pandas 백엔드는 NumPy 배열을 사용하는데, 여러 한계가 있습니다:
- **네이티브 nullable 정수/실수 없음**: 정수 열에 `NaN`이 하나만 있어도 전체 열이 `float64`로 강제 변환되어 메모리 낭비와 미묘한 버그를 유발합니다.
- **문자열이 Python 객체로 저장**: 각 문자열이 힙(Heap)에 별도의 Python 객체로 존재하여, 높은 메모리 오버헤드와 느린 연산을 초래합니다.
- **제로 카피 읽기 불가**: Parquet 파일을 로드할 때 Arrow에서 NumPy 포맷으로 데이터를 복사해야 합니다.

PyArrow는 컬럼형(Columnar), 캐시 친화적(Cache-friendly) 메모리 레이아웃으로 이 세 가지 문제를 모두 해결합니다.

### 8.2 PyArrow 백엔드 활성화

```python
import pandas as pd
import numpy as np

# 방법 1: PyArrow 백엔드로 직접 읽기
# 왜 engine="pyarrow"을 사용하는가? Arrow에서 NumPy로의 복사를 피하여
# 전체 파이프라인에서 데이터를 Arrow 포맷으로 유지할 수 있기 때문이다.
# df = pd.read_csv("data.csv", dtype_backend="pyarrow")
# df = pd.read_parquet("data.parquet", dtype_backend="pyarrow")

# 방법 2: 명시적 Arrow 타입으로 DataFrame 생성
df_arrow = pd.DataFrame({
    'id': pd.array([1, 2, 3, None], dtype="int64[pyarrow]"),
    'name': pd.array(["Alice", "Bob", None, "Diana"], dtype="string[pyarrow]"),
    'score': pd.array([95.5, None, 88.0, 92.3], dtype="float64[pyarrow]"),
})

print(df_arrow.dtypes)
# id        int64[pyarrow]
# name     string[pyarrow]
# score   float64[pyarrow]

print(df_arrow)
# 참고: None 값이 타입 강제 변환 없이 NA로 보존된다
```

### 8.3 Nullable 타입: NaN 강제 변환 없음

```python
# --- 전통적 NumPy 백엔드: NaN이 float 변환을 강제 ---
df_numpy = pd.DataFrame({'values': [1, 2, None, 4]})
print(f"NumPy dtype: {df_numpy['values'].dtype}")  # float64 (강제!)
print(df_numpy['values'].tolist())  # [1.0, 2.0, nan, 4.0]

# --- PyArrow 백엔드: 정수는 정수로 유지 ---
df_pa = pd.DataFrame({'values': pd.array([1, 2, None, 4], dtype="int64[pyarrow]")})
print(f"PyArrow dtype: {df_pa['values'].dtype}")  # int64[pyarrow]
print(df_pa['values'].tolist())  # [1, 2, <NA>, 4]

# 왜 이것이 중요한가: 데이터 파이프라인에서 정수 ID(user_id, order_id)는
# 절대 float으로 변환되어서는 안 된다. PyArrow에서는 1이 1.0이 아니라 1로 유지된다.
# 이는 1 != 1.0인 조인(Join)과 그룹바이(GroupBy) 연산에서 버그를 방지한다.
```

### 8.4 PyArrow의 문자열 성능

```python
import time

# 대규모 문자열 Series 생성
n = 500_000
strings_object = pd.Series(["hello_world_" + str(i) for i in range(n)], dtype="object")
strings_arrow = pd.Series(
    ["hello_world_" + str(i) for i in range(n)], dtype="string[pyarrow]"
)

# 메모리 비교
print(f"object dtype 메모리:       {strings_object.memory_usage(deep=True) / 1e6:.1f} MB")
print(f"string[pyarrow] 메모리:    {strings_arrow.memory_usage(deep=True) / 1e6:.1f} MB")

# 속도 비교: str.contains
start = time.time()
_ = strings_object.str.contains("999")
time_object = time.time() - start

start = time.time()
_ = strings_arrow.str.contains("999")
time_arrow = time.time() - start

print(f"\nstr.contains 속도:")
print(f"  object dtype:       {time_object:.3f}s")
print(f"  string[pyarrow]:    {time_arrow:.3f}s")
print(f"  속도 향상:            {time_object / time_arrow:.1f}x")

# 왜 PyArrow 문자열이 빠른가: Arrow는 문자열을 연속 버퍼에 오프셋 배열과
# 함께 저장하여 SIMD 가속 연산이 가능하다. Python 객체 문자열은 힙에
# 분산되어 캐시 미스(Cache miss)를 유발한다.
```

### 8.5 PyArrow 엔진으로 Parquet 입출력

```python
# Parquet은 Arrow의 네이티브 파일 포맷이다 — PyArrow 백엔드를 사용하면
# 비용이 큰 직렬화(Serialization)를 피하는 제로 카피 읽기가 가능하다.

# Parquet 쓰기
df_sample = pd.DataFrame({
    'id': pd.array(range(1000), dtype="int64[pyarrow]"),
    'category': pd.array(["A", "B", "C"] * 333 + ["A"], dtype="string[pyarrow]"),
    'value': pd.array(np.random.randn(1000), dtype="float64[pyarrow]"),
})

df_sample.to_parquet("sample.parquet", engine="pyarrow")

# PyArrow 백엔드로 Parquet 읽기 — 거의 제로 카피
df_loaded = pd.read_parquet("sample.parquet", dtype_backend="pyarrow")
print(df_loaded.dtypes)
print(f"메모리 사용량: {df_loaded.memory_usage(deep=True).sum() / 1e3:.1f} KB")
```

### 8.6 PyArrow 백엔드 사용 시점

| 시나리오 | 권장사항 | 이유 |
|----------|---------|------|
| 대규모 데이터셋 (>100K행) | PyArrow 사용 | 낮은 메모리, 빠른 연산 |
| 문자열이 많은 데이터 (로그, 텍스트) | PyArrow 사용 | 2-10배 적은 메모리, 빠른 `.str` |
| 결측 정수가 있는 데이터 | PyArrow 사용 | float 강제 변환 방지 |
| Parquet 기반 파이프라인 | PyArrow 사용 | 제로 카피 읽기 |
| 소규모 데이터셋 (<10K행) | 어느 것이든 가능 | 오버헤드 무시 가능 |
| NumPy 상호운용이 필요한 레거시 코드 | NumPy 유지 | 일부 라이브러리가 NumPy 배열을 기대 |
| GPU 워크플로우 (cuDF) | PyArrow 사용 | cuDF가 Arrow 포맷을 네이티브로 사용 |

**호환성 참고**: Pandas 2.2+ 기준으로 대부분의 연산이 PyArrow 백엔드 DataFrame과 매끄럽게 동작합니다. 그러나 일부 서드파티 라이브러리는 `.to_numpy()` 변환이 필요할 수 있습니다. 마이그레이션 전에 항상 전체 파이프라인을 테스트하세요.

---

## 연습 문제

### 문제 1: 그룹별 통계
부서별 평균 급여와 직원 수를 구하세요.

```python
df = pd.DataFrame({
    'department': ['Sales', 'IT', 'IT', 'HR', 'Sales'],
    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'salary': [50000, 60000, 70000, 55000, 65000]
})

# 풀이
result = df.groupby('department').agg(
    avg_salary=('salary', 'mean'),
    count=('name', 'count')
)
print(result)
```

### 문제 2: 데이터 병합
두 DataFrame을 조인하여 직원의 부서명을 포함하세요.

```python
employees = pd.DataFrame({
    'emp_id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie'],
    'dept_id': [10, 20, 10]
})

departments = pd.DataFrame({
    'dept_id': [10, 20],
    'dept_name': ['Sales', 'IT']
})

# 풀이
result = pd.merge(employees, departments, on='dept_id')
print(result)
```

### 문제 3: 피벗 테이블
월별, 카테고리별 매출 합계를 피벗 테이블로 만드세요.

```python
sales = pd.DataFrame({
    'month': ['Jan', 'Jan', 'Feb', 'Feb', 'Jan', 'Feb'],
    'category': ['A', 'B', 'A', 'B', 'A', 'A'],
    'amount': [100, 150, 120, 180, 110, 130]
})

# 풀이
pivot = pd.pivot_table(sales, values='amount',
                       index='month', columns='category',
                       aggfunc='sum')
print(pivot)
```

---

## 요약

| 기능 | 함수/메서드 |
|------|------------|
| 필터링 | `df[condition]`, `query()`, `isin()` |
| 정렬 | `sort_values()`, `sort_index()`, `rank()` |
| 그룹화 | `groupby()`, `agg()`, `transform()`, `apply()` |
| 병합 | `merge()`, `join()`, `concat()` |
| 피벗 | `pivot()`, `pivot_table()`, `melt()`, `stack()`, `unstack()` |
| 중복 | `duplicated()`, `drop_duplicates()` |
| 교차표 | `crosstab()` |
