# 현대 데이터 도구: Polars와 DuckDB

[이전: 생존 분석](./28_Survival_Analysis.md)

## 개요

Polars와 DuckDB는 중대형 데이터셋에서 Pandas보다 훨씬 빠른 속도를 제공하는 차세대 데이터 처리 도구입니다. 이 레슨에서는 Polars의 지연 평가(lazy evaluation)와 표현식 API(expression API), DuckDB의 인프로세스 SQL 엔진(in-process SQL engine), Apache Arrow 상호 운용성(interoperability), 그리고 Pandas에서의 실전 마이그레이션 패턴을 다룹니다.

---

## 1. 왜 현대 도구인가?

### 1.1 Pandas의 한계

```python
"""
Pandas의 문제점 (중대형 데이터):

1. 단일 스레드: 멀티코어 CPU 활용 불가
2. 메모리 낭비: 데이터를 자주 복사, GIL 오버헤드
3. 즉시 평가(Eager evaluation): 모든 연산이 즉시 실행됨
4. 일관성 없는 API: 같은 작업을 여러 방법으로 수행 가능
5. 느린 문자열 연산: 문자열 컬럼에 최적화되지 않음

Pandas가 여전히 적합한 경우:
  - 소규모 데이터셋 (< 1M 행)
  - 빠른 탐색 / 프로토타이핑
  - 깊은 에코시스템 (sklearn, statsmodels 등)

Polars/DuckDB를 고려할 경우:
  - 데이터셋 > 1M 행
  - 변환 작업의 빠른 반복이 필요
  - 1000만 행 이상 데이터에서 집계 수행
  - 서버 없이 SQL 기반 분석이 필요

성능 비교 (1000만 행, 그룹별 집계):
  Pandas:  ~5초
  Polars:  ~0.3초 (16배 빠름)
  DuckDB:  ~0.2초 (25배 빠름)
"""
```

---

## 2. Polars

### 2.1 기본 사용법

```python
import polars as pl
import numpy as np

# DataFrame 생성
df = pl.DataFrame({
    "name": ["Alice", "Bob", "Carol", "Dave", "Eve"],
    "age": [30, 25, 35, 28, 32],
    "city": ["NYC", "LA", "NYC", "LA", "NYC"],
    "salary": [75000, 65000, 90000, 70000, 85000],
})
print(df)

# 파일에서 읽기 (대용량 파일의 경우 Pandas보다 훨씬 빠름)
# df = pl.read_csv("data.csv")
# df = pl.read_parquet("data.parquet")

# 기본 연산
print(df.filter(pl.col("age") > 28))           # 필터링
print(df.select(["name", "salary"]))            # 컬럼 선택
print(df.sort("salary", descending=True))       # 정렬
print(df.with_columns(                          # 컬럼 추가
    (pl.col("salary") * 1.1).alias("new_salary")
))
```

### 2.2 표현식 API(Expression API)

```python
"""
Polars 표현식 API:
  - pl.col("name")           → 컬럼 참조
  - pl.lit(value)            → 리터럴 값
  - 체이닝 가능한 변환       → .filter().group_by().agg()
  - 인덱스 없음 (Pandas와 달리) → 더 명확한 의미
"""

import polars as pl
import numpy as np

# 더 큰 데이터셋 생성
np.random.seed(42)
N = 1_000_000
df = pl.DataFrame({
    "user_id": np.random.randint(1, 10001, N),
    "product": np.random.choice(["A", "B", "C", "D", "E"], N),
    "amount": np.random.exponential(50, N).round(2),
    "timestamp": pl.Series(
        np.random.choice(
            pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 12, 31), eager=True),
            N,
        )
    ),
})

# 복잡한 표현식
result = (
    df
    .filter(pl.col("amount") > 10)
    .with_columns(
        pl.col("timestamp").dt.month().alias("month"),
        pl.col("amount").log().alias("log_amount"),
        (pl.col("amount") > pl.col("amount").mean()).alias("above_avg"),
    )
    .group_by(["product", "month"])
    .agg(
        pl.col("amount").sum().alias("total_revenue"),
        pl.col("amount").mean().alias("avg_order"),
        pl.col("user_id").n_unique().alias("unique_users"),
        pl.col("amount").quantile(0.95).alias("p95_amount"),
    )
    .sort(["product", "month"])
)
print(result.head(10))
```

### 2.3 지연 평가(Lazy Evaluation)

```python
"""
지연 평가(Lazy Evaluation): 쿼리 계획을 수립하고, 최적화한 후 실행.

즉시 평가(Eager, Pandas):
  df = read_csv("big.csv")          # 전체 데이터를 메모리로 읽기
  df = df[df.amount > 100]          # 필터링 (모든 행 처리)
  df = df.groupby("product").sum()  # 집계

지연 평가(Lazy, Polars):
  q = scan_csv("big.csv")           # 쿼리 계획 수립 (I/O 없음)
  q = q.filter(col("amount") > 100) # 계획에 필터 추가
  q = q.group_by("product").agg(sum("amount"))  # 집계 추가
  result = q.collect()               # 최적화된 계획 실행

자동으로 적용되는 최적화:
  - 술어 푸시다운(Predicate pushdown): 읽기 전에 필터 적용
  - 프로젝션 푸시다운(Projection pushdown): 필요한 컬럼만 읽기
  - 공통 부분식 제거(Common subexpression elimination)
  - 조인 최적화(Join optimization)
  - 병렬 실행(Parallel execution)
"""

# 지연 쿼리 (실서비스에서는 scan_csv 또는 scan_parquet 사용)
lazy_df = df.lazy()

query = (
    lazy_df
    .filter(pl.col("amount") > 20)
    .group_by("product")
    .agg(
        pl.col("amount").sum().alias("total"),
        pl.col("amount").mean().alias("avg"),
        pl.len().alias("count"),
    )
    .sort("total", descending=True)
)

# 최적화된 쿼리 계획 확인
print("Query Plan:")
print(query.explain())

# 실행
result = query.collect()
print("\nResult:")
print(result)

# RAM보다 큰 데이터셋에 대한 스트리밍 모드
# result = query.collect(streaming=True)
```

### 2.4 윈도우 함수(Window Functions)

```python
# Polars의 윈도우 함수
result = df.with_columns(
    # 제품별 순위
    pl.col("amount").rank().over("product").alias("rank_in_product"),

    # 사용자별 누적 합계
    pl.col("amount").cum_sum().over("user_id").alias("cumulative_spend"),

    # 제품 평균과의 차이
    (pl.col("amount") - pl.col("amount").mean().over("product")).alias("diff_from_avg"),

    # Lead/lag
    pl.col("amount").shift(1).over("user_id").alias("prev_amount"),
)
print(result.head(10))
```

---

## 3. DuckDB

### 3.1 인프로세스 SQL 엔진(In-Process SQL Engine)

```python
"""
DuckDB: 인프로세스 OLAP 데이터베이스.
  - 서버 불필요 (SQLite와 유사하지만 분석용)
  - 컬럼 기반 저장 → 빠른 집계
  - Pandas, Polars, Parquet, CSV 직접 쿼리 가능
  - 완전한 SQL 지원 (윈도우 함수, CTE 등)
"""

import duckdb

# SQL로 Polars DataFrame 직접 쿼리!
result = duckdb.sql("""
    SELECT
        product,
        COUNT(*) as order_count,
        ROUND(SUM(amount), 2) as total_revenue,
        ROUND(AVG(amount), 2) as avg_order,
        ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY amount), 2) as p95
    FROM df
    WHERE amount > 10
    GROUP BY product
    ORDER BY total_revenue DESC
""")
print(result)

# Parquet 파일 직접 쿼리 (메모리 적재 없음)
# duckdb.sql("SELECT * FROM 'data/*.parquet' WHERE year = 2024")

# 자동 감지로 CSV 파일 쿼리
# duckdb.sql("SELECT * FROM read_csv_auto('data.csv')")
```

### 3.2 고급 SQL 기능(Advanced SQL Features)

```python
import duckdb

# 윈도우 함수
result = duckdb.sql("""
    WITH monthly AS (
        SELECT
            product,
            EXTRACT(MONTH FROM timestamp) as month,
            SUM(amount) as revenue
        FROM df
        GROUP BY product, month
    )
    SELECT
        product,
        month,
        revenue,
        LAG(revenue) OVER (PARTITION BY product ORDER BY month) as prev_month,
        ROUND(100.0 * (revenue - LAG(revenue) OVER (PARTITION BY product ORDER BY month))
              / NULLIF(LAG(revenue) OVER (PARTITION BY product ORDER BY month), 0), 1)
              as growth_pct,
        SUM(revenue) OVER (PARTITION BY product ORDER BY month) as ytd_revenue
    FROM monthly
    ORDER BY product, month
""")
print(result)
```

### 3.3 영구 저장소를 사용하는 DuckDB(DuckDB with Persistent Storage)

```python
import duckdb

# 영구 데이터베이스 생성
con = duckdb.connect("analytics.duckdb")

# 테이블 생성 및 데이터 삽입
con.execute("""
    CREATE TABLE IF NOT EXISTS orders AS
    SELECT * FROM df
""")

# 빠른 쿼리를 위한 인덱스 생성
con.execute("CREATE INDEX IF NOT EXISTS idx_product ON orders(product)")

# 분석 쿼리
result = con.execute("""
    SELECT
        product,
        DATE_TRUNC('month', timestamp) as month,
        COUNT(DISTINCT user_id) as unique_users,
        COUNT(*) as orders,
        ROUND(SUM(amount), 2) as revenue
    FROM orders
    GROUP BY product, month
    ORDER BY month, product
""").fetchdf()  # Pandas DataFrame 반환
print(result.head(10))

# 결과 내보내기
con.execute("COPY (SELECT * FROM orders WHERE amount > 100) TO 'high_value.parquet' (FORMAT PARQUET)")

con.close()

# 정리
import os
os.remove("analytics.duckdb")
```

---

## 4. Arrow 상호 운용성(Arrow Interoperability)

### 4.1 제로 복사 데이터 교환(Zero-Copy Data Exchange)

```python
"""
Apache Arrow: 컬럼형 데이터를 위한 공통 인메모리 포맷.

  Polars DataFrame ←→ Arrow Table ←→ DuckDB
       ↕                  ↕
  Pandas DataFrame   Parquet File

제로 복사(Zero-copy): 포맷 간 변환 시 데이터 중복 없음.
"""

import polars as pl
import pandas as pd
import pyarrow as pa
import duckdb

# Polars → Arrow → Pandas (가능한 경우 제로 복사)
polars_df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

# Polars → Arrow
arrow_table = polars_df.to_arrow()
print(f"Arrow: {arrow_table.schema}")

# Arrow → Pandas
pandas_df = arrow_table.to_pandas()
print(f"Pandas: {pandas_df.dtypes.to_dict()}")

# Pandas → Polars
polars_from_pandas = pl.from_pandas(pandas_df)
print(f"Polars: {polars_from_pandas.dtypes}")

# DuckDB는 세 가지 모두 직접 쿼리 가능
result = duckdb.sql("SELECT * FROM polars_df WHERE a > 1")
print(result)
```

---

## 5. Pandas에서 마이그레이션(Migration from Pandas)

### 5.1 주요 연산 비교(Common Operations Comparison)

```python
"""
Pandas → Polars 전환 가이드:

| 연산(Operation)  | Pandas                         | Polars                               |
|-----------------|--------------------------------|--------------------------------------|
| CSV 읽기        | pd.read_csv("f.csv")           | pl.read_csv("f.csv")                |
| Parquet 읽기    | pd.read_parquet("f.parquet")   | pl.read_parquet("f.parquet")         |
| 필터링          | df[df.age > 30]                | df.filter(pl.col("age") > 30)       |
| 컬럼 선택       | df[["a", "b"]]                 | df.select(["a", "b"])                |
| 컬럼 추가       | df["new"] = df.a + 1           | df.with_columns((pl.col("a")+1).alias("new")) |
| 그룹 + 집계     | df.groupby("g").agg({"a":"sum"})| df.group_by("g").agg(pl.col("a").sum()) |
| 정렬            | df.sort_values("a")            | df.sort("a")                         |
| 이름 변경       | df.rename(columns={"a":"b"})   | df.rename({"a": "b"})                |
| 결측값 제거     | df.dropna()                    | df.drop_nulls()                      |
| 결측값 채우기   | df.fillna(0)                   | df.fill_null(0)                      |
| 값 개수         | df.a.value_counts()            | df["a"].value_counts()               |
| 조인            | pd.merge(df1, df2, on="key")   | df1.join(df2, on="key")              |
| Apply           | df.a.apply(func)               | df.with_columns(pl.col("a").map_elements(func)) |
"""

import pandas as pd
import polars as pl
import numpy as np

# 성능 비교
np.random.seed(42)
N = 5_000_000

# 동일한 데이터를 두 라이브러리에 생성
data = {
    "category": np.random.choice(["A", "B", "C", "D", "E"], N),
    "value": np.random.randn(N),
    "amount": np.random.exponential(100, N),
}

import time

# Pandas
pd_df = pd.DataFrame(data)
t0 = time.time()
pd_result = pd_df.groupby("category").agg(
    total=("amount", "sum"),
    avg_val=("value", "mean"),
    count=("value", "count"),
)
pd_time = time.time() - t0

# Polars
pl_df = pl.DataFrame(data)
t0 = time.time()
pl_result = pl_df.group_by("category").agg(
    pl.col("amount").sum().alias("total"),
    pl.col("value").mean().alias("avg_val"),
    pl.len().alias("count"),
)
pl_time = time.time() - t0

# DuckDB
import duckdb
t0 = time.time()
db_result = duckdb.sql("""
    SELECT category,
           SUM(amount) as total,
           AVG(value) as avg_val,
           COUNT(*) as count
    FROM pl_df
    GROUP BY category
""")
db_time = time.time() - t0

print(f"Group-by aggregation on {N:,} rows:")
print(f"  Pandas:  {pd_time:.3f}s")
print(f"  Polars:  {pl_time:.3f}s  ({pd_time/pl_time:.1f}x faster)")
print(f"  DuckDB:  {db_time:.3f}s  ({pd_time/db_time:.1f}x faster)")
```

---

## 6. 실전 패턴(Practical Patterns)

### 6.1 대용량 파일 처리(Large File Processing)

```python
"""
RAM보다 큰 파일 처리:

Polars 스트리밍:
  pl.scan_parquet("big_file.parquet")
    .filter(...)
    .group_by(...)
    .agg(...)
    .collect(streaming=True)

DuckDB:
  duckdb.sql("SELECT ... FROM 'big_file.parquet' WHERE ...")
  → 청크 단위로 처리, 전체 파일을 메모리에 적재하지 않음
"""

# Polars: 여러 파일의 지연 스캔
# query = (
#     pl.scan_parquet("data/year=*/*.parquet")
#     .filter(pl.col("amount") > 100)
#     .group_by("product")
#     .agg(pl.col("amount").sum())
#     .collect(streaming=True)
# )

# DuckDB: 파티셔닝된 Parquet 직접 쿼리
# duckdb.sql("""
#     SELECT product, SUM(amount)
#     FROM 'data/year=*/month=*/*.parquet'
#     WHERE year = 2024 AND amount > 100
#     GROUP BY product
# """)
```

### 6.2 도구 선택 기준(When to Use What)

```python
"""
선택 기준:

┌─────────────────────────────────────────────────┐
│                데이터셋 크기                    │
├──────────┬──────────────┬───────────────────────┤
│ < 1M 행  │ 1M - 1억 행  │ > 1억 행              │
│          │              │                       │
│ Pandas   │ Polars 또는  │ Polars (스트리밍)     │
│ (충분)   │ DuckDB       │ DuckDB                │
│          │              │ Spark (분산)          │
└──────────┴──────────────┴───────────────────────┘

Pandas 사용 시:
  - 소규모 데이터 + 빠른 프로토타이핑
  - sklearn/statsmodels 연동 필요
  - 팀이 Pandas에 익숙

Polars 사용 시:
  - 변환/집계 속도가 필요
  - 데이터 파이프라인 구축 (지연 평가)
  - 메서드 체이닝 API 선호

DuckDB 사용 시:
  - Python API보다 SQL 선호
  - 파일 직접 쿼리 필요 (Parquet/CSV)
  - 영구 분석 데이터베이스 원함
  - SQL + Python 혼합 워크플로우
"""
```

---

## 7. 연습 문제

### 연습 1: Polars 파이프라인

```python
"""
Polars로 데이터 파이프라인 구축:
1. 1000만 행 판매 데이터셋 생성 (product, region, date, amount, quantity)
2. 지연 평가를 사용하여 다음을 계산:
   a. 제품 및 지역별 월간 매출
   b. 제품별 전월 대비 성장률
   c. 지역별 3개월 이동 평균
3. 실행 시간 비교: 즉시 평가 vs 지연 평가
4. 파티셔닝을 적용하여 결과를 Parquet으로 내보내기
"""
```

### 연습 2: DuckDB 분석

```python
"""
DuckDB SQL을 사용한 분석:
1. 3개 테이블이 있는 DuckDB 데이터베이스 생성 (orders, customers, products)
2. 다음 쿼리 작성:
   a. 생애 가치(lifetime value) 기준 상위 10명 고객
   b. 월별 코호트 유지율(cohort retention) 분석
   c. 제품 친화도(product affinity, 자주 함께 구매되는 제품)
3. DuckDB 쿼리 시간과 동등한 Pandas 연산 시간 비교
4. 결과를 Parquet으로 내보내고 파일 간 쿼리 수행
"""
```

---

## 8. 요약

### 핵심 정리

| 도구 | 유형 | 최적 용도 |
|------|------|-----------|
| **Polars** | DataFrame 라이브러리 | 빠른 변환, 지연 평가, 파이프라인 |
| **DuckDB** | 인프로세스 SQL | SQL 분석, 파일 직접 쿼리 |
| **Arrow** | 메모리 포맷 | 도구 간 제로 복사 상호 운용 |
| **Pandas** | DataFrame 라이브러리 | 프로토타이핑, 소규모 데이터, 에코시스템 |

### 모범 사례

1. **Polars에서 지연 평가 사용** — 옵티마이저가 쿼리를 계획하도록 맡기기
2. **DuckDB로 파일 직접 쿼리** — 메모리에 적재할 필요 없음
3. **교환 시 Arrow 사용** — Polars/Pandas/DuckDB 간 복사 없이 변환
4. **먼저 프로파일링** — 성능 문제가 없다면 Pandas에서 전환하지 말 것
5. **Parquet 사용** — 컬럼 기반 포맷은 세 가지 도구 모두에 유리

### 탐색

- **이전**: L28 — 생존 분석(Survival Analysis)
- **L01** (NumPy 기초, NumPy Fundamentals)로 돌아가 데이터 도구의 기초 학습
