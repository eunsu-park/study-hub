# PySpark DataFrame

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Python 리스트, Pandas DataFrame, CSV 파일, Parquet 파일 등 다양한 소스에서 명시적 스키마 정의와 함께 Spark DataFrame을 생성할 수 있다
2. PySpark API를 사용하여 select, filter, withColumn, groupBy, agg, join 등 핵심 DataFrame 변환을 적용할 수 있다
3. Spark SQL 내장 함수(Built-in Function)와 윈도우 함수(Window Function)를 사용하여 복잡한 컬럼 수준 연산을 수행할 수 있다
4. 분산 DataFrame에서 fillna, dropna, 대입(Imputation) 전략을 사용하여 결측 데이터를 처리할 수 있다
5. Catalyst 옵티마이저(Catalyst Optimizer)가 논리적 및 물리적 실행 계획을 생성하고 최적화하는 방식을 설명할 수 있다
6. 성능을 위한 파티션 전략을 설정하면서 DataFrame을 다양한 출력 형식과 스토리지 시스템에 저장할 수 있다

---

## 개요

Spark DataFrame은 분산된 데이터를 테이블 형태로 표현하는 고수준 API입니다. SQL과 유사한 연산을 제공하며, Catalyst 옵티마이저를 통해 자동으로 최적화됩니다.

---

## 1. SparkSession과 DataFrame 생성

### 1.1 SparkSession 초기화

```python
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType

# SparkSession은 Spark 2.0 이후 통합 진입점이다 — 별도의 SQLContext/HiveContext를 대체한다
spark = SparkSession.builder \
    .appName("PySpark DataFrame Tutorial") \
    .config("spark.sql.shuffle.partitions", 100) \  # 기본 200개 셔플 파티션은 소규모 데이터셋에 너무 많을 수 있다 — 총 코어 수의 2-3배로 조정한다
    .config("spark.sql.adaptive.enabled", True) \    # AQE는 런타임에 파티션을 자동 조정한다 — 데이터 볼륨이 변동하는 경우 필수적이다
    .getOrCreate()

# Spark 버전 확인
print(f"Spark Version: {spark.version}")
```

### 1.2 DataFrame 생성 방법

```python
# 방법 1: Python 리스트에서 생성 — 테스트에는 편리하지만 Spark가 데이터를 스캔하여
# 스키마를 추론해야 하므로 대규모 데이터셋에서는 느리다
data = [
    ("Alice", 30, "Engineering"),
    ("Bob", 25, "Marketing"),
    ("Charlie", 35, "Engineering"),
]
df1 = spark.createDataFrame(data, ["name", "age", "department"])

# 방법 2: 명시적 스키마는 스키마 추론 스캔을 피하고 타입 불일치를 방지한다
# (예: 널 값으로 인한 잘못된 타입 추론). 프로덕션에서는 항상 사용한다.
# nullable=False는 Spark가 쓰기 시점에 강제하는 NOT NULL 제약 조건을 추가한다.
schema = StructType([
    StructField("name", StringType(), nullable=False),
    StructField("age", IntegerType(), nullable=True),
    StructField("department", StringType(), nullable=True),
])
df2 = spark.createDataFrame(data, schema)

# 방법 3: 딕셔너리 리스트에서 생성 — Spark가 딕셔너리 키에서 스키마를 추론한다.
# 튜플보다 가독성이 높지만 딕셔너리 오버헤드로 인해 약간 느리다.
dict_data = [
    {"name": "Alice", "age": 30, "department": "Engineering"},
    {"name": "Bob", "age": 25, "department": "Marketing"},
]
df3 = spark.createDataFrame(dict_data)

# 방법 4: Pandas에서 생성 — 가능한 경우 Arrow를 사용하여 효율적으로 전송한다
# (spark.sql.execution.arrow.pyspark.enabled=true). 드라이버 메모리에 맞는 데이터만 가능하며,
# 대용량 데이터는 분산 스토리지에서 직접 읽는다.
import pandas as pd
pdf = pd.DataFrame(data, columns=["name", "age", "department"])
df4 = spark.createDataFrame(pdf)

# 방법 5: RDD에서 생성 — 레거시 RDD 코드를 DataFrame으로 마이그레이션할 때 유용하다.
# toDF()는 Python 객체에서 타입을 추론하므로 불안정할 수 있다; 명시적 스키마와 함께
# createDataFrame(rdd, schema)를 사용하는 것이 더 낫다.
rdd = spark.sparkContext.parallelize(data)
df5 = rdd.toDF(["name", "age", "department"])
```

### 1.3 파일에서 DataFrame 읽기

```python
# CSV 파일
df_csv = spark.read.csv(
    "data.csv",
    header=True,           # 첫 행을 헤더로
    inferSchema=True,      # 전체 파일을 스캔하여 타입을 감지한다 — 읽기 시간이 2배가 된다.
                           # 이 오버헤드를 피하려면 프로덕션에서 명시적 스키마를 사용한다.
    sep=",",               # 구분자
    nullValue="NA",        # "NA" 문자열을 Spark 널로 매핑한다 — 유효한 문자열로 처리되는 것을 방지한다
    dateFormat="yyyy-MM-dd"
)

# 명시적 스키마는 전체 파일 추론 스캔을 건너뛰고 실행 간 일관된
# 타입을 보장한다 (inferSchema는 데이터에 따라 int vs long을 다르게 감지할 수 있다).
schema = StructType([
    StructField("id", IntegerType()),
    StructField("name", StringType()),
    StructField("amount", DoubleType()),
    StructField("date", DateType()),
])
df_csv = spark.read.csv("data.csv", header=True, schema=schema)

# Parquet은 파일 메타데이터에 스키마를 내장한다 — 추론이 필요 없다. 컬럼형 형식은
# 컬럼 가지치기(요청한 컬럼만 읽기)와 조건절 푸시다운(min/max 통계로 행 그룹 건너뛰기)을 가능하게 한다.
df_parquet = spark.read.parquet("data.parquet")

# JSON 파일
df_json = spark.read.json("data.json")

# ORC 파일
df_orc = spark.read.orc("data.orc")

# JDBC 읽기는 소스 DB에 쿼리를 실행한다 — 기본적으로 단일 파티션(단일 JDBC 연결)을
# 사용하여 병목이 발생한다. 대규모 테이블에서는 병렬 읽기를 위해
# partitionColumn/lowerBound/upperBound/numPartitions를 추가한다.
df_jdbc = spark.read.format("jdbc") \
    .option("url", "jdbc:postgresql://localhost:5432/mydb") \
    .option("dbtable", "public.users") \
    .option("user", "user") \
    .option("password", "password") \
    .option("driver", "org.postgresql.Driver") \
    .load()

# Delta Lake는 Parquet 위에 ACID 트랜잭션과 타임 트래블을 추가한다 —
# 스키마 강제 및 진화로 조용한 데이터 손상을 방지한다
df_delta = spark.read.format("delta").load("path/to/delta")
```

---

## 2. DataFrame 기본 연산

### 2.1 데이터 확인

```python
# show()는 실행을 트리거하지만 출력을 제한한다 — collect()와 달리 어떤 데이터 크기에도 안전하다
df.show()           # 상위 20행
df.show(5)          # 상위 5행
df.show(truncate=False)  # 컬럼 잘림 없이 — 값이 긴 문자열일 때 유용하다

# printSchema()는 메타데이터만 읽는다 (데이터 스캔 없음) — 항상 빠르다
df.printSchema()
df.dtypes           # [(컬럼명, 타입), ...]
df.columns          # 컬럼 목록

# describe()는 전체 스캔을 트리거한다 — count, mean, stddev, min, max를 계산한다.
# summary()는 백분위수(25%, 50%, 75%)를 추가하지만 더 비용이 크다.
df.describe().show()        # 기술 통계
df.summary().show()         # 확장 통계

# count()는 모든 파티션을 스캔하는 액션이다 — 여러 번 호출되는 경우
# 불필요한 스캔을 피하기 위해 결과를 캐시한다
df.count()

# distinct()는 중복 제거를 위해 셔플이 필요하다 — 고카디널리티 컬럼에서 비용이 크다
df.select("department").distinct().count()

# first/head는 첫 번째 파티션에서만 가져온다 — 전체 스캔보다 훨씬 저렴하다
df.first()
df.head(5)

# toPandas()는 모든 데이터를 드라이버 메모리에 Pandas DataFrame으로 수집한다 —
# DataFrame이 드라이버 메모리보다 크면 OOM이 발생한다. 소규모 결과에만 사용한다.
pdf = df.toPandas()
```

### 2.2 컬럼 선택

```python
from pyspark.sql.functions import col, lit

# 단일 컬럼
df.select("name")
df.select(col("name"))
df.select(df.name)
df.select(df["name"])

# 여러 컬럼
df.select("name", "age")
df.select(["name", "age"])
df.select(col("name"), col("age"))

# 모든 컬럼 + 추가 컬럼
df.select("*", lit(1).alias("constant"))

# 컬럼 제외
df.drop("department")

# 컬럼 이름 변경
df.withColumnRenamed("name", "full_name")

# 여러 컬럼 이름 변경
df.toDF("name_new", "age_new", "dept_new")

# alias 사용
df.select(col("name").alias("employee_name"))
```

### 2.3 필터링

```python
from pyspark.sql.functions import col

# 기본 필터
df.filter(col("age") > 30)
df.filter(df.age > 30)
df.filter("age > 30")           # SQL 표현식
df.where(col("age") > 30)       # filter와 동일

# 복합 조건
df.filter((col("age") > 25) & (col("department") == "Engineering"))
df.filter((col("age") < 25) | (col("department") == "Marketing"))
df.filter(~(col("age") > 30))   # NOT

# 문자열 필터
df.filter(col("name").startswith("A"))
df.filter(col("name").endswith("e"))
df.filter(col("name").contains("li"))
df.filter(col("name").like("%li%"))
df.filter(col("name").rlike("^[A-C].*"))  # 정규식

# IN 조건
df.filter(col("department").isin(["Engineering", "Marketing"]))

# NULL 처리
df.filter(col("age").isNull())
df.filter(col("age").isNotNull())

# BETWEEN
df.filter(col("age").between(25, 35))
```

---

## 3. 변환 (Transformations)

### 3.1 컬럼 추가/수정

```python
from pyspark.sql.functions import col, lit, when, concat, upper, lower, length

# withColumn은 새로운 DataFrame을 반환한다 — DataFrame은 불변이다. 각 호출은
# 논리적 계획에 프로젝션을 추가하며; Catalyst는 연속적인 withColumn 호출을 병합한다.
df.withColumn("bonus", col("salary") * 0.1)

# lit()은 Python 스칼라를 Spark Column으로 감싼다 — Spark 표현식은 분산
# Column 객체에서 동작하므로 로컬 Python 값이 필요하다
df.withColumn("country", lit("USA"))

# 같은 컬럼 이름을 사용하면 논리적 계획에서 제자리에 교체된다 (변경이 아님)
df.withColumn("name", upper(col("name")))

# when/otherwise는 SQL CASE WHEN에 매핑된다 — 쿼리 계획의 일부로 지연 평가된다.
# 조건은 순서대로 확인되며; 첫 번째 일치가 적용된다.
df.withColumn("age_group",
    when(col("age") < 30, "Young")
    .when(col("age") < 50, "Middle")
    .otherwise("Senior")
)

# withColumns (Spark 3.3+)는 하나의 호출로 여러 변환을 적용한다 — withColumn 체이닝보다
# 깔끔하며 Catalyst가 함께 최적화하는 데 도움이 될 수 있다
df.withColumns({
    "name_upper": upper(col("name")),
    "age_plus_10": col("age") + 10,
})

# 문자열 결합
df.withColumn("full_info", concat(col("name"), lit(" - "), col("department")))

# cast()는 논리적 계획에서 컬럼 타입을 변경한다 — Spark가 실행 시점에 변환을 처리한다.
# 잘못된 캐스트(예: "abc"를 int로)는 오류 대신 널을 생성한다.
df.withColumn("age_double", col("age").cast("double"))
df.withColumn("age_string", col("age").cast(StringType()))
```

### 3.2 집계 연산

```python
from pyspark.sql.functions import (
    count, sum as _sum, avg, min as _min, max as _max,
    countDistinct, collect_list, collect_set,
    first, last, stddev, variance
)

# groupBy 없는 집계는 전체 DataFrame에서 계산한다 —
# 단일 행 결과를 생성한다 (GROUP BY 없는 SQL SELECT처럼)
df.agg(
    count("*").alias("total_count"),
    _sum("salary").alias("total_salary"),
    avg("salary").alias("avg_salary"),
    _min("salary").alias("min_salary"),
    _max("salary").alias("max_salary"),
).show()

# groupBy는 같은 키를 가진 행을 함께 배치하기 위해 셔플을 트리거하고 각 그룹 내에서 집계를 적용한다.
# Catalyst는 셔플 볼륨을 줄이기 위해 파티션 내에서 부분 집계(해시 기반)를 사용한다.
df.groupBy("department").agg(
    count("*").alias("employee_count"),
    avg("salary").alias("avg_salary"),
    _sum("salary").alias("total_salary"),
    countDistinct("name").alias("unique_names"),  # 고유 값 추적이 필요 — count보다 메모리 사용량이 많다
)

# 다중 컬럼 그룹화는 그룹 키의 교차곱을 생성한다 — 카디널리티가 높으면
# 많은 그룹이 생겨 OOM이 발생할 수 있다
df.groupBy("department", "age_group").count()

# collect_list/collect_set은 그룹당 모든 값을 배열로 수집한다 — 주의:
# 그룹에 수백만 개의 값이 있으면 executor에서 OOM이 발생할 수 있다.
# 그룹 크기가 제한되어 있을 때만 사용한다.
df.groupBy("department").agg(
    collect_list("name").alias("employee_names"),  # 중복과 삽입 순서를 보존한다
    collect_set("age").alias("unique_ages"),        # 값을 중복 제거한다
)

# pivot()은 행 값을 컬럼으로 변환한다 — 피벗 컬럼의 전체 사전 스캔을 피하고
# 출력 스키마를 제어하기 위해 허용 값을 명시적으로 지정한다
df.groupBy("department") \
    .pivot("age_group", ["Young", "Middle", "Senior"]) \
    .agg(count("*"))
```

### 3.3 정렬

```python
from pyspark.sql.functions import col, asc, desc

# 단일 컬럼 정렬
df.orderBy("age")                    # 오름차순 (기본)
df.orderBy(col("age").desc())        # 내림차순
df.orderBy(desc("age"))

# 여러 컬럼 정렬
df.orderBy(["department", "age"])
df.orderBy(col("department").asc(), col("age").desc())

# NULL 처리
df.orderBy(col("age").asc_nulls_first())
df.orderBy(col("age").desc_nulls_last())

# sort는 orderBy와 동일
df.sort("age")
```

### 3.4 조인

```python
# 테스트 데이터
employees = spark.createDataFrame([
    (1, "Alice", 101),
    (2, "Bob", 102),
    (3, "Charlie", 101),
], ["id", "name", "dept_id"])

departments = spark.createDataFrame([
    (101, "Engineering"),
    (102, "Marketing"),
    (103, "Finance"),
], ["dept_id", "dept_name"])

# Inner Join — Spark가 전략을 자동 선택한다: 한쪽이 10MB 미만이면 브로드캐스트,
# 그렇지 않으면 소트-머지 조인. explain()으로 선택된 전략을 확인한다.
employees.join(departments, employees.dept_id == departments.dept_id)
employees.join(departments, "dept_id")  # 문자열 형식으로 조인 컬럼이 자동 중복 제거된다

# Left Join — 모든 왼쪽 행을 보존한다; 누락된 관계를 감지할 때 사용한다
# (예: 유효한 부서가 없는 직원)
employees.join(departments, "dept_id", "left")

# Right Join
employees.join(departments, "dept_id", "right")

# Full Outer Join — 가장 비용이 크다: 양쪽의 모든 행을 구체화해야 한다
employees.join(departments, "dept_id", "full")

# Cross Join은 N*M 행을 생성한다 — 대형 테이블에서 매우 주의해서 사용한다.
# Spark는 우발적인 카테시안 곱을 방지하기 위해 명시적 crossJoin()을 요구한다.
employees.crossJoin(departments)

# Semi Join은 일치하는 왼쪽 행을 반환하지만 오른쪽 컬럼은 포함하지 않는다 —
# 오른쪽이 구체화되지 않고 프로브만 되므로 inner join + drop보다 효율적이다
employees.join(departments, "dept_id", "left_semi")

# Anti Join은 일치하지 않는 왼쪽 행을 반환한다 — 고아 레코드 찾기에 유용하다
# (예: 삭제된 제품을 참조하는 주문)
employees.join(departments, "dept_id", "left_anti")

# 복합 조건 — 이렇게 하면 단일 컬럼 조인 최적화가 비활성화되고
# 출력에 중복 조인 컬럼이 생성될 수 있다
employees.join(
    departments,
    (employees.dept_id == departments.dept_id) & (employees.id > 1),
    "inner"
)
```

---

## 4. 액션 (Actions)

### 4.1 데이터 수집

```python
# collect()는 모든 데이터를 드라이버 메모리로 가져온다 — 데이터가 드라이버 RAM을 초과하면 OOM이 발생한다.
# 검사에는 take()/show()를, 대용량 출력에는 write()를 사용한다.
result = df.collect()           # 전체 데이터 (주의: 메모리)
result = df.take(10)            # 10개 행을 찾을 때까지만 파티션을 처리한다 — 저렴하다
result = df.first()             # 첫 번째 행
result = df.head(5)             # 상위 5개

# .rdd 변환은 Catalyst 최적화를 깨뜨린다 — 프로덕션 파이프라인에서는 피한다.
# DataFrame 네이티브 대안으로 df.select("age").collect() + 리스트 컴프리헨션을 사용한다.
ages = df.select("age").rdd.flatMap(lambda x: x).collect()

# toPandas()는 드라이버 메모리에 전체 데이터셋을 구체화한다 — Arrow를 활성화하면
# (spark.sql.execution.arrow.pyspark.enabled=true) ~10배 빠른 전송이 가능하다
pdf = df.toPandas()             # 작은 데이터만

# toLocalIterator()는 한 번에 하나의 파티션을 가져온다 — 메모리 사용량이 제한되지만
# 순차 파티션 페치로 인해 collect()보다 훨씬 느리다
for row in df.toLocalIterator():
    print(row)
```

### 4.2 파일 저장

```python
# Parquet은 권장 형식이다 — 압축, 스키마 메타데이터, 조건절 푸시다운을 지원하는 컬럼형 스토리지.
# 읽기 속도가 CSV보다 ~10배 빠르다.
df.write.parquet("output/data.parquet")

# "overwrite"는 전체 디렉토리를 교체한다 — Delta Lake 없이는 원자적이지 않다.
# "append"는 증분 쓰기에 더 안전하지만 재시도 시 중복 데이터가 생길 수 있다.
df.write.mode("overwrite").parquet("output/data.parquet")
# overwrite: 기존 덮어쓰기
# append: 기존에 추가
# ignore: 존재하면 무시
# error: 존재하면 에러 (기본)

# partitionBy는 디렉토리 계층 구조(year=2024/month=01/)를 생성한다 — 이 컬럼을
# 필터링하는 쿼리가 전체 디렉토리를 건너뛸 수 있도록 파티션 가지치기를 가능하게 한다.
# 저카디널리티 컬럼(날짜, 지역)을 선택한다 — 높은 카디널리티는 너무 많은 작은 파일을 생성한다.
df.write.partitionBy("date", "department").parquet("output/partitioned")

# CSV
df.write.csv("output/data.csv", header=True)

# JSON
df.write.json("output/data.json")

# coalesce(1)은 모든 파티션을 하나로 병합한다 — 단일 출력 파일을 생성하지만
# 쓰기 병렬성을 잃는다. 소규모 출력에만 사용하며 대용량 데이터는 여러 파일을 유지하여
# 리더가 병렬화할 수 있게 한다.
df.coalesce(1).write.csv("output/single_file.csv", header=True)

# JDBC 쓰기는 파티션당 하나의 연결을 사용한다 — 파티션이 많으면 데이터베이스에
# 과부하가 걸릴 수 있다. 연결 수를 제어하기 위해 coalesce() 또는 repartition()을 사용한다.
df.write.format("jdbc") \
    .option("url", "jdbc:postgresql://localhost:5432/mydb") \
    .option("dbtable", "public.output_table") \
    .option("user", "user") \
    .option("password", "password") \
    .mode("overwrite") \
    .save()
```

---

## 5. UDF (User Defined Functions)

### 5.1 기본 UDF

```python
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType, IntegerType

# UDF는 최후의 수단이다 — Spark가 Python 함수 내부를 검사할 수 없으므로
# Catalyst 최적화가 비활성화된다. 이 분류 패턴에는 내장 함수(when/otherwise)를
# 선호한다. Spark 내장 함수로 표현할 수 없는 로직에만 UDF를 사용한다.
def categorize_age(age):
    if age is None:
        return "Unknown"
    elif age < 30:
        return "Young"
    elif age < 50:
        return "Middle"
    else:
        return "Senior"

# 데코레이터 방식 — UDF로만 사용되는 함수에 더 깔끔하다.
# returnType은 Spark가 Python 함수 반환 타입을 추론할 수 없으므로 필수다.
@udf(returnType=StringType())
def categorize_age_udf(age):
    if age is None:
        return "Unknown"
    elif age < 30:
        return "Young"
    elif age < 50:
        return "Middle"
    else:
        return "Senior"

# 함수 방식 — 일반 Python 함수로도 테스트해야 할 때 유용하다
categorize_udf = udf(categorize_age, StringType())

# 두 접근 방식은 동일한 실행 계획을 생성한다 — 각 행이 Python으로 직렬화되고,
# 처리된 후, 다시 직렬화된다. 이 직렬화/역직렬화 오버헤드로 인해 UDF가
# 동등한 내장 함수보다 10-100배 느리다.
df.withColumn("age_category", categorize_udf(col("age")))
df.withColumn("age_category", categorize_age_udf(col("age")))
```

### 5.2 Pandas UDF (성능 향상)

```python
from pyspark.sql.functions import pandas_udf
import pandas as pd

# Pandas UDF는 JVM과 Python 간의 벡터화된 데이터 전송을 위해 Apache Arrow를 사용한다 —
# 행별이 아닌 배치 단위로 데이터를 처리하여 일반 UDF보다 3-100배 빠르다.
# 여전히 네이티브 Spark 함수보다는 느리지만, 복잡한 Python 로직에는 최선의 선택이다.
@pandas_udf(StringType())
def categorize_pandas_udf(age_series: pd.Series) -> pd.Series:
    return age_series.apply(
        lambda x: "Unknown" if x is None
        else "Young" if x < 30
        else "Middle" if x < 50
        else "Senior"
    )

# 사용
df.withColumn("age_category", categorize_pandas_udf(col("age")))

# GROUPED_MAP은 그룹당 모든 행을 Pandas DataFrame으로 받는다 — 내장 Spark
# 집계 함수로 표현할 수 없는 복잡한 그룹별 분석(회귀, 커스텀 집계)에 유용하다.
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

# 출력 스키마는 Spark가 Python 함수 실행 전에 쿼리를 계획하는 데 필요하므로
# 명시적으로 선언해야 한다
result_schema = StructType([
    StructField("department", StringType()),
    StructField("avg_salary", DoubleType()),
    StructField("employee_count", IntegerType()),
])

@pandas_udf(result_schema, PandasUDFType.GROUPED_MAP)
def analyze_department(pdf: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "department": [pdf["department"].iloc[0]],
        "avg_salary": [pdf["salary"].mean()],
        "employee_count": [len(pdf)],
    })

# apply()는 각 그룹의 데이터를 Python으로 전송한다 — 전체 그룹이 executor
# 메모리에 맞아야 한다. 그룹이 매우 크면 내장 함수로 사전 집계를 고려한다.
df.groupby("department").apply(analyze_department)
```

### 5.3 SQL에서 UDF 사용

```python
# SQL용 UDF 등록
spark.udf.register("categorize_age", categorize_age, StringType())

# SQL에서 사용
df.createOrReplaceTempView("employees")
spark.sql("""
    SELECT name, age, categorize_age(age) as age_category
    FROM employees
""").show()
```

---

## 6. 윈도우 함수

```python
from pyspark.sql.window import Window
from pyspark.sql.functions import (
    row_number, rank, dense_rank,
    lead, lag, sum as _sum, avg,
    first, last, ntile
)

# 윈도우 정의는 "무엇을 계산할지"를 "어떤 행에서"와 분리한다 —
# 여러 계산에 동일한 윈도우 스펙을 재사용하여 DRY하게 유지한다.
# partitionBy는 독립적인 그룹을 결정하고; orderBy는 각 그룹 내의 행 순서를 정의한다.
window_dept = Window.partitionBy("department").orderBy("salary")
window_all = Window.orderBy("salary")  # 파티션 없음 = 단일 전역 윈도우 (대용량 데이터에서 비용이 크다)

# row_number는 고유한 순차 번호를 부여한다 (동점 없음) — 상위 N개 쿼리에 유용하다.
# rank/dense_rank는 동점을 다르게 처리한다: rank는 번호를 건너뛴다 (1,2,2,4), dense_rank는 건너뛰지 않는다 (1,2,2,3).
df.withColumn("row_num", row_number().over(window_dept))
df.withColumn("rank", rank().over(window_dept))
df.withColumn("dense_rank", dense_rank().over(window_dept))
df.withColumn("ntile_4", ntile(4).over(window_dept))  # 대략 동일한 4개 버킷으로 분할

# lag/lead는 셀프 조인 없이 인접 행에 접근한다 — 연속 레코드 간의 차이를
# 계산하는 데 훨씬 효율적이다 (예: 일별 변화량)
df.withColumn("prev_salary", lag("salary", 1).over(window_dept))
df.withColumn("next_salary", lead("salary", 1).over(window_dept))

# rowsBetween은 프레임을 정의한다: unboundedPreceding부터 currentRow = 누적 합계.
# 명시적 프레임 범위 없이는 Spark가 RANGE(값 기반)를 사용하여 동점이 예상치 않게
# 포함될 수 있다. 결정적인 누적 합계를 위해 ROWS(위치 기반)를 사용한다.
window_cumsum = Window.partitionBy("department") \
    .orderBy("date") \
    .rowsBetween(Window.unboundedPreceding, Window.currentRow)

df.withColumn("cumsum_salary", _sum("salary").over(window_cumsum))

# 고정 윈도우로 이동 평균: 현재 행 + 이전 2개 행.
# rowsBetween(-2, 0)은 "현재 행에서 2행 전부터 현재 행까지"를 의미한다 (3행 윈도우).
window_moving = Window.partitionBy("department") \
    .orderBy("date") \
    .rowsBetween(-2, 0)

df.withColumn("moving_avg", avg("salary").over(window_moving))

# 윈도우 내 첫 번째/마지막 값 — 앞채우기(forward-fill) 또는 뒤채우기(back-fill) 패턴에 유용하다
df.withColumn("first_name", first("name").over(window_dept))
df.withColumn("last_name", last("name").over(window_dept))
```

---

## 연습 문제

### 문제 1: 데이터 변환
판매 데이터에서 월별, 카테고리별 총 매출과 평균 매출을 계산하세요.

### 문제 2: 윈도우 함수
각 부서별로 급여 순위를 매기고, 부서 내 급여 상위 3명을 추출하세요.

### 문제 3: UDF 작성
이메일 주소에서 도메인을 추출하는 UDF를 작성하고 적용하세요.

---

## 요약

| 연산 | 설명 | 예시 |
|------|------|------|
| **select** | 컬럼 선택 | `df.select("name", "age")` |
| **filter** | 행 필터링 | `df.filter(col("age") > 30)` |
| **groupBy** | 그룹화 | `df.groupBy("dept").agg(...)` |
| **join** | 테이블 조인 | `df1.join(df2, "key")` |
| **orderBy** | 정렬 | `df.orderBy(desc("salary"))` |
| **withColumn** | 컬럼 추가/수정 | `df.withColumn("new", ...)` |

---

## 참고 자료

- [PySpark DataFrame Guide](https://spark.apache.org/docs/latest/sql-getting-started.html)
- [PySpark Functions](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/functions.html)
