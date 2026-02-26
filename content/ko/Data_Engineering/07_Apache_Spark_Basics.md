# Apache Spark 기초

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Apache Spark의 아키텍처(Driver, Executor, Cluster Manager)를 설명하고, Spark의 인메모리 처리가 Hadoop MapReduce 대비 성능 향상을 달성하는 원리를 기술할 수 있다
2. RDD(Resilient Distributed Dataset), DataFrame, Dataset의 차이를 구분하고 각 추상화를 언제 사용해야 하는지 설명할 수 있다
3. SparkSession을 생성하고 기본적인 데이터 로딩, 변환(Transformation), 액션(Action) 연산을 수행할 수 있다
4. PySpark API를 사용하여 필터링, 그룹화, 조인(Join), 집계(Aggregation) 등 일반적인 DataFrame 변환을 적용할 수 있다
5. 지연 평가(Lazy Evaluation)를 설명하고 Spark의 DAG 실행 모델이 쿼리 계획을 최적화하는 방식을 기술할 수 있다
6. Spark 작업 파라미터를 설정하고 HDFS, S3, Parquet 파일 등 분산 스토리지 소스에서 데이터를 읽을 수 있다

---

## 개요

Apache Spark는 대규모 데이터 처리를 위한 통합 분석 엔진입니다. 인메모리 처리로 Hadoop MapReduce보다 빠른 성능을 제공하며, 배치 처리와 스트리밍을 모두 지원합니다.

---

## 1. Spark 개요

### 1.1 Spark의 특징

```
┌────────────────────────────────────────────────────────────────┐
│                    Apache Spark 특징                            │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   1. 속도 (Speed)                                              │
│      - 인메모리 처리로 Hadoop보다 100배 빠름                     │
│      - 디스크 기반보다 10배 빠름                                 │
│                                                                │
│   2. 사용 편의성 (Ease of Use)                                  │
│      - Python, Scala, Java, R 지원                             │
│      - SQL 인터페이스 제공                                      │
│                                                                │
│   3. 범용성 (Generality)                                       │
│      - SQL, 스트리밍, ML, 그래프 처리                           │
│      - 하나의 엔진으로 다양한 워크로드                           │
│                                                                │
│   4. 호환성 (Compatibility)                                     │
│      - HDFS, S3, Cassandra 등 다양한 데이터 소스                │
│      - YARN, Kubernetes, Standalone 클러스터                    │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 1.2 Spark 생태계

```
┌─────────────────────────────────────────────────────────────────┐
│                     Spark Ecosystem                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐  │
│   │  Spark SQL │ │ Streaming  │ │   MLlib    │ │  GraphX    │  │
│   │    + DF    │ │ (Structured)│ │(Machine   │ │  (Graph)   │  │
│   │            │ │             │ │ Learning) │ │            │  │
│   └────────────┘ └────────────┘ └────────────┘ └────────────┘  │
│   ─────────────────────────────────────────────────────────────│
│   │                     Spark Core                           │  │
│   │                 (RDD, Task Scheduling)                   │  │
│   ─────────────────────────────────────────────────────────────│
│   ─────────────────────────────────────────────────────────────│
│   │    Standalone    │    YARN    │    Kubernetes    │ Mesos │  │
│   ─────────────────────────────────────────────────────────────│
│   ─────────────────────────────────────────────────────────────│
│   │  HDFS  │   S3   │   GCS   │  Cassandra  │  JDBC  │ etc │  │
│   ─────────────────────────────────────────────────────────────│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Spark 아키텍처

### 2.1 클러스터 구성

```
┌─────────────────────────────────────────────────────────────────┐
│                    Spark Cluster Architecture                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌───────────────────────────────────────────────────────┐    │
│   │                    Driver Program                      │    │
│   │   ┌─────────────────────────────────────────────────┐ │    │
│   │   │              SparkContext                        │ │    │
│   │   │   - 애플리케이션 진입점                          │ │    │
│   │   │   - 클러스터와 연결                              │ │    │
│   │   │   - Job 생성 및 스케줄링                         │ │    │
│   │   └─────────────────────────────────────────────────┘ │    │
│   └───────────────────────────────────────────────────────┘    │
│                              ↓                                  │
│   ┌───────────────────────────────────────────────────────┐    │
│   │                  Cluster Manager                       │    │
│   │       (Standalone, YARN, Kubernetes, Mesos)            │    │
│   └───────────────────────────────────────────────────────┘    │
│                              ↓                                  │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
│   │   Worker    │  │   Worker    │  │   Worker    │           │
│   │  ┌───────┐  │  │  ┌───────┐  │  │  ┌───────┐  │           │
│   │  │Executor│ │  │  │Executor│ │  │  │Executor│ │           │
│   │  │ Task  │  │  │  │ Task  │  │  │  │ Task  │  │           │
│   │  │ Task  │  │  │  │ Task  │  │  │  │ Task  │  │           │
│   │  │ Cache │  │  │  │ Cache │  │  │  │ Cache │  │           │
│   │  └───────┘  │  │  └───────┘  │  │  └───────┘  │           │
│   └─────────────┘  └─────────────┘  └─────────────┘           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 핵심 개념

| 개념 | 설명 |
|------|------|
| **Driver** | 메인 프로그램 실행, SparkContext 생성 |
| **Executor** | Worker 노드에서 Task 실행 |
| **Task** | 실행의 기본 단위 |
| **Job** | Action에 의해 생성되는 병렬 계산 |
| **Stage** | Job 내의 Task 그룹 (Shuffle 경계) |
| **Partition** | 데이터의 논리적 분할 단위 |

### 2.3 실행 흐름

```python
"""
Spark 실행 흐름:
1. Driver에서 SparkContext 생성
2. 애플리케이션 코드 해석
3. Transformation → DAG (Directed Acyclic Graph) 생성
4. Action 호출 시 Job 생성
5. Job → Stages → Tasks로 분해
6. Cluster Manager가 Executor에 Task 할당
7. Executor에서 Task 실행
8. 결과를 Driver로 반환
"""

# 예시 코드 흐름
from pyspark.sql import SparkSession

# SparkSession은 Spark 2.0 이후 통합 진입점이다 — 이전 버전에서 별도로 필요했던
# SparkContext, SQLContext, HiveContext를 대체한다
spark = SparkSession.builder.appName("Example").getOrCreate()

# 트랜스포메이션은 지연 평가된다 — Spark는 DAG(실행 계획)를 구성하지만 아직
# 데이터를 읽거나 처리하지 않는다. 이를 통해 Catalyst 옵티마이저가 실행 전에
# 연산을 재정렬하고 병합할 수 있다.
df = spark.read.csv("data.csv", header=True)  # 읽기 계획
df2 = df.filter(df.age > 20)                  # 필터 계획
df3 = df2.groupBy("city").count()             # 집계 계획

# 액션은 전체 DAG 실행을 트리거한다. collect()는 모든 데이터를 드라이버로 가져온다 —
# 작은 결과에는 안전하지만, 큰 데이터셋에서는 OOM이 발생할 수 있다
# (대용량 출력에는 .show()나 .write를 사용한다).
result = df3.collect()  # Job 생성 → Stages → Tasks → 실행
```

---

## 3. RDD (Resilient Distributed Dataset)

### 3.1 RDD 개념

RDD는 Spark의 기본 데이터 구조로, 분산된 불변 데이터 컬렉션입니다.

```python
from pyspark import SparkContext

# "local[*]"은 모든 가용 CPU 코어를 사용하여 로컬 모드로 Spark를 실행한다 —
# 개발/테스트에 이상적이다. 프로덕션에서는 클러스터 모드를 위해 "yarn"이나 "k8s://..."를 사용한다.
sc = SparkContext("local[*]", "RDD Example")

# RDD 생성 방법
# 1. parallelize()는 로컬 Python 컬렉션을 파티션에 분산한다.
# 기본 파티션 수 = 코어 수. 테스트에 유용하며, 프로덕션에서는 외부 소스에서 데이터를 가져온다.
rdd1 = sc.parallelize([1, 2, 3, 4, 5])

# 2. textFile은 HDFS 블록(기본 128MB)당 하나의 파티션을 생성한다 — Spark가
# 파일 크기에 따라 자동으로 읽기를 병렬화한다
rdd2 = sc.textFile("data.txt")

# 3. 트랜스포메이션은 항상 새로운 RDD를 생성한다 — RDD는 불변이며, 이는 리니지 기반
# 장애 복구를 가능하게 한다: 파티션이 손실되면 Spark는 해당 파티션을 재계산하는 데
# 필요한 트랜스포메이션만 재실행한다
rdd3 = rdd1.map(lambda x: x * 2)

# RDD 특성
"""
R - Resilient: 장애 복구 가능 (Lineage로 재계산)
D - Distributed: 클러스터에 분산 저장
D - Dataset: 데이터 컬렉션
"""
```

### 3.2 RDD 연산

```python
# Transformations (Lazy)
# - 실행하지 않고 새로운 RDD를 반환한다 — Spark는 계산을 리니지 그래프로 기록한다.
#   이 지연 실행 덕분에 옵티마이저가 실행 전에 연산을 결합하고 데이터 이동을 최소화할 수 있다.

rdd = sc.parallelize([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# map: 1대1 변환, 파티션 수 유지
mapped = rdd.map(lambda x: x * 2)  # [2, 4, 6, ...]

# filter: 데이터를 일찍 좁혀 다운스트림 처리 감소 — 파이프라인에서
# 항상 가능한 한 빨리 필터를 적용한다
filtered = rdd.filter(lambda x: x % 2 == 0)  # [2, 4, 6, 8, 10]

# flatMap: 1대다 매핑 — 토크나이제이션(예: 줄을 단어로 분할)에 유용
flat = rdd.flatMap(lambda x: [x, x*2])  # [1, 2, 2, 4, 3, 6, ...]

# distinct: 셔플이 필요하다(비용이 크다) — 중복이 실제로 중요한 경우에만 사용한다
distinct = rdd.distinct()

# union: 데이터 이동 없는 논리적 병합 — 두 RDD의 파티션이 연결된다
union = rdd.union(sc.parallelize([11, 12]))

# groupByKey: 키의 파티션으로 모든 값을 셔플한다 — 키의 모든 값이 메모리에 맞아야
# 하므로 메모리 집약적이다. 가능하면 reduceByKey를 선호한다.
pairs = sc.parallelize([("a", 1), ("b", 2), ("a", 3)])
grouped = pairs.groupByKey()  # [("a", [1, 3]), ("b", [2])]

# reduceByKey: 셔플 전에 각 파티션 내에서 로컬 결합을 수행한다 (미니 MapReduce 컴바이너처럼),
# 네트워크 전송을 대폭 줄인다. groupByKey + reduce보다 항상 선호된다.
reduced = pairs.reduceByKey(lambda a, b: a + b)  # [("a", 4), ("b", 2)]


# Actions (Eager)
# - 전체 리니지 실행을 트리거한다 — Spark가 클러스터에 Job을 제출하고,
#   셔플 경계에서 스테이지와 파티션별 태스크로 분해한다.

# collect: 모든 데이터를 드라이버 메모리로 가져온다 — 데이터가 드라이버 RAM을 초과하면 OOM이 발생한다.
# 검사에는 take()/show()를, 대용량 출력에는 write()를 사용한다.
result = rdd.collect()  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# count: 실행을 트리거하지만 단일 숫자를 반환한다 — 어떤 데이터 크기에도 안전하다
count = rdd.count()  # 10

# first / take: N개 요소를 찾을 때까지만 파티션을 처리한다 — 데이터 미리보기에
# collect()보다 훨씬 저렴하다
first = rdd.first()  # 1
take3 = rdd.take(3)  # [1, 2, 3]

# reduce: 모든 요소를 하나로 결합한다 — 각 파티션 내에서 병렬로 실행한 후
# 드라이버에서 결과를 집계한다
total = rdd.reduce(lambda a, b: a + b)  # 55

# foreach: 익스큐터에서 실행된다 (드라이버 아님) — 외부 시스템에 쓰기 등
# 부수 효과에 사용한다. 반환 값은 버려진다.
rdd.foreach(lambda x: print(x))

# saveAsTextFile: 파티션당 하나의 파일을 쓴다 — 출력 파일 수를 줄이고 싶다면
# 먼저 coalesce()를 사용한다
rdd.saveAsTextFile("output/")
```

### 3.3 Pair RDD 연산

```python
# Key-Value 쌍 RDD 연산
sales = sc.parallelize([
    ("Electronics", 100),
    ("Clothing", 50),
    ("Electronics", 200),
    ("Clothing", 75),
    ("Food", 30),
])

# reduceByKey는 셔플 전에 각 파티션 내에서 사전 집계를 수행한다 —
# 먼저 원시 값을 셔플하는 groupByKey().mapValues(sum)보다 훨씬 효율적이다
total_by_category = sales.reduceByKey(lambda a, b: a + b)
# [("Electronics", 300), ("Clothing", 125), ("Food", 30)]

# combineByKey는 가장 일반적인 집계다 — 누적 타입이 값 타입과 다를 때 사용한다
# (여기서: value=int, accumulator=(sum, count) 튜플).
# 세 가지 함수가 처리한다: 첫 번째 어큐뮬레이터 생성, 기존 어큐뮬레이터에 값 병합,
# 파티션 간 두 어큐뮬레이터 병합.
count_sum = sales.combineByKey(
    lambda v: (v, 1),                      # createCombiner: 파티션의 첫 번째 값
    lambda acc, v: (acc[0] + v, acc[1] + 1),  # mergeValue: 기존 어큐뮬레이터에 추가
    lambda acc1, acc2: (acc1[0] + acc2[0], acc1[1] + acc2[1])  # mergeCombiner: 파티션 간 병합
)
avg_by_category = count_sum.mapValues(lambda x: x[0] / x[1])

# sortByKey는 데이터를 범위 파티셔닝하기 위해 전체 셔플이 필요하다 — 대용량 데이터셋에서 비용이 크다.
# 소비자가 정렬된 출력을 실제로 필요로 할 때만 정렬한다.
sorted_rdd = sales.sortByKey()

# Join은 두 RDD를 키로 셔플하고 레코드를 매칭한다 — 키당 카테시안 곱을 생성한다.
# 한쪽이 작은 경우(< 10MB), sc.broadcast()를 통해 브로드캐스트 조인을 고려하여
# 비싼 셔플을 피한다.
inventory = sc.parallelize([
    ("Electronics", 50),
    ("Clothing", 100),
])

joined = sales.join(inventory)
# [("Electronics", (100, 50)), ("Electronics", (200, 50)), ...]
```

---

## 4. 설치 및 실행

### 4.1 로컬 설치 (PySpark)

```bash
# pip 설치
pip install pyspark

# 버전 확인
pyspark --version

# PySpark 셸 시작
pyspark

# spark-submit으로 스크립트 실행
spark-submit my_script.py
```

### 4.2 Docker 설치

```yaml
# docker-compose.yaml
version: '3'

services:
  spark-master:
    image: bitnami/spark:3.4
    environment:
      - SPARK_MODE=master
    ports:
      - "8080:8080"
      - "7077:7077"

  spark-worker:
    image: bitnami/spark:3.4
    environment:
      - SPARK_MODE=worker
      - SPARK_MASTER_URL=spark://spark-master:7077
    depends_on:
      - spark-master
```

```bash
# 실행
docker-compose up -d

# 클러스터에 작업 제출
spark-submit --master spark://localhost:7077 my_script.py
```

### 4.3 클러스터 모드

```bash
# Standalone 클러스터
spark-submit \
    --master spark://master:7077 \
    --deploy-mode cluster \
    --executor-memory 4G \
    --executor-cores 2 \
    --num-executors 10 \
    my_script.py

# YARN 클러스터
spark-submit \
    --master yarn \
    --deploy-mode cluster \
    --executor-memory 4G \
    my_script.py

# Kubernetes 클러스터
spark-submit \
    --master k8s://https://k8s-master:6443 \
    --deploy-mode cluster \
    --conf spark.kubernetes.container.image=my-spark-image \
    my_script.py
```

---

## 5. SparkSession

### 5.1 SparkSession 생성

```python
from pyspark.sql import SparkSession

# getOrCreate()는 JVM에 이미 SparkSession이 존재하면 재사용한다 —
# 노트북에서 여러 세션을 생성하는 일반적인 오류를 방지한다
spark = SparkSession.builder \
    .appName("My Application") \
    .getOrCreate()

# 프로덕션 설정 — 세션 생성 전에 설정해야 한다;
# 일부 설정(예: executor 메모리)은 JVM 시작 후 변경할 수 없다.
spark = SparkSession.builder \
    .appName("My Application") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", 200) \   # 200은 기본값 — 소형 클러스터에서는 코어 수의 2-3배로 조정한다
    .config("spark.executor.memory", "4g") \          # executor당 힙 메모리 — 노드의 가용 RAM에 따라 설정한다
    .config("spark.driver.memory", "2g") \            # 드라이버는 collect() 결과와 브로드캐스트 변수를 위한 충분한 RAM이 필요하다
    .config("spark.sql.adaptive.enabled", "true") \   # AQE는 런타임에 동적으로 최적화한다 — Spark 3.x에서 강력히 권장된다
    .enableHiveSupport() \                            # Hive 메타스토어 접근에만 필요 — 사용하지 않으면 시작 오버헤드가 추가된다
    .getOrCreate()

# SparkContext는 저수준 RDD API다 — 브로드캐스트 변수,
# 어큐뮬레이터, DataFrame API로 사용할 수 없는 RDD 연산에 여전히 필요하다
sc = spark.sparkContext

# 설정 확인
print(spark.conf.get("spark.sql.shuffle.partitions"))

# 완료 시 항상 세션을 중지하여 클러스터 리소스를 해제하고 로그를 플러시한다
spark.stop()
```

### 5.2 주요 설정

```python
# 자주 사용하는 설정
common_configs = {
    # 메모리 설정 — executor 메모리는 실행(셔플, 조인)과
    # 스토리지(캐시) 간에 분할된다. memoryOverhead는 오프힙 메모리(Python 프로세스, JVM 오버헤드)를 커버한다.
    "spark.executor.memory": "4g",
    "spark.driver.memory": "2g",
    "spark.executor.memoryOverhead": "512m",  # PySpark에서 증가시킨다 — Python 워커는 오프힙을 사용한다

    # 병렬성 — 이 값들이 태스크 수를 결정한다. 너무 적으면 코어가 유휴 상태가 된다.
    # 너무 많으면 과도한 스케줄링 오버헤드와 작은 태스크가 발생한다.
    "spark.executor.cores": "4",              # executor당 코어 수 — 4-5가 일반적인 최적값
    "spark.default.parallelism": "100",       # RDD 연산용 (SQL 아님)
    "spark.sql.shuffle.partitions": "200",    # DataFrame/SQL 셔플용 — 총 코어 수의 2-3배로 시작한다

    # Kryo는 Java 직렬화보다 10배 빠르고 더 컴팩트하다 — 최상의 결과를 위해
    # kryo.classesToRegister에 클래스를 등록한다
    "spark.serializer": "org.apache.spark.serializer.KryoSerializer",

    # AQE는 실제 데이터 통계를 기반으로 런타임에 쿼리 계획을 재최적화한다 —
    # 정적 계획으로 예측할 수 없는 데이터 스큐와 파티션 크기 조정을 처리한다
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.coalescePartitions.enabled": "true",   # 셔플 후 작은 파티션을 병합한다
    "spark.sql.adaptive.skewJoin.enabled": "true",             # 스큐된 파티션을 자동으로 분할한다

    # 캐시 설정
    "spark.storage.memoryFraction": "0.6",    # 캐싱에 executor 메모리의 60% — 조인/셔플에 더 많은 실행 메모리가 필요하면 낮춘다

    # 셔플 압축은 CPU 비용으로 네트워크 I/O를 줄인다 — 셔플은 일반적으로
    # CPU가 아닌 네트워크 바운드이므로 거의 항상 이득이다
    "spark.shuffle.compress": "true",
}

# 설정 적용 예시
spark = SparkSession.builder \
    .config("spark.sql.shuffle.partitions", 100) \
    .config("spark.sql.adaptive.enabled", True) \
    .getOrCreate()
```

---

## 6. 기본 예제

### 6.1 Word Count

```python
from pyspark.sql import SparkSession

# SparkSession 생성
spark = SparkSession.builder \
    .appName("Word Count") \
    .getOrCreate()

sc = spark.sparkContext

# textFile은 HDFS 블록 경계로 입력을 분할한다 — 각 블록이 파티션이 되어
# 클러스터 전체에서 병렬 읽기를 가능하게 한다
text_rdd = sc.textFile("input.txt")

# RDD 트랜스포메이션으로 표현된 고전 MapReduce 패턴:
# flatMap → map → reduceByKey가 표준 워드 카운트 파이프라인이다.
word_counts = text_rdd \
    .flatMap(lambda line: line.split()) \     # 1줄 → 여러 단어 (1대N 매핑)
    .map(lambda word: (word.lower(), 1)) \    # "The"와 "the"가 다른 키가 되지 않도록 대소문자 정규화
    .reduceByKey(lambda a, b: a + b) \        # 로컬 결합 + 셔플 — groupByKey보다 훨씬 효율적
    .sortBy(lambda x: x[1], ascending=False)  # 전역 정렬은 전체 셔플이 필요하다 — 마지막에 수행한다

# take(10)은 10개 결과를 반환할 때까지만 파티션을 스캔한다 —
# 전체 어휘를 드라이버로 가져오는 collect()를 피한다
for word, count in word_counts.take(10):
    print(f"{word}: {count}")

# 파티션당 part-NNNNN 파일 하나를 쓴다. 단일 파일을 원하면 coalesce(1)을 사용하되,
# 출력이 작을 때만 사용한다 — 단일 파일 쓰기는 병렬화할 수 없다.
word_counts.saveAsTextFile("output/word_counts")

spark.stop()
```

### 6.2 DataFrame 기본

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as _sum, avg

spark = SparkSession.builder.appName("DataFrame Example").getOrCreate()

# DataFrame은 RDD보다 선호되는 API다 — Catalyst 옵티마이저를 활용하여
# 자동 쿼리 최적화(조건절 푸시다운, 컬럼 가지치기 등)를 수행한다
data = [
    ("Alice", "Engineering", 50000),
    ("Bob", "Engineering", 60000),
    ("Charlie", "Marketing", 45000),
    ("Diana", "Marketing", 55000),
]

# Python 튜플에서 스키마 추론 — 프로토타이핑에 유용하지만 프로덕션에서는
# 타입 안정성과 더 나은 Parquet 성능을 위해 명시적 StructType 스키마를 사용한다
df = spark.createDataFrame(data, ["name", "department", "salary"])

# show()는 액션이다 — 실행을 트리거하지만 출력을 제한한다(기본 20행)
# 모든 것을 드라이버로 가져오는 collect()와 달리
df.show()
df.printSchema()

# 컬럼 기반 필터링은 Catalyst를 사용하여 가능한 경우 데이터 소스에 조건절을
# 푸시다운한다 (예: Parquet 행 그룹 필터링, JDBC WHERE 절 푸시다운)
df.filter(col("salary") > 50000).show()

# groupBy는 같은 키를 가진 행을 함께 배치하기 위해 셔플을 트리거한다 —
# 대부분의 Spark 작업에서 가장 비싼 연산이다.
df.groupBy("department") \
    .agg(
        _sum("salary").alias("total_salary"),
        avg("salary").alias("avg_salary")
    ) \
    .show()

# SQL과 DataFrame API는 내부적으로 동일한 실행 계획을 생성한다 —
# 해당 쿼리에서 더 읽기 쉬운 방식을 선택한다
df.createOrReplaceTempView("employees")
spark.sql("""
    SELECT department, AVG(salary) as avg_salary
    FROM employees
    GROUP BY department
""").show()

spark.stop()
```

---

## 연습 문제

### 문제 1: RDD 기본 연산
1부터 100까지의 숫자 중 짝수만 선택하여 제곱의 합을 구하세요.

```python
# 풀이
sc = spark.sparkContext
result = sc.parallelize(range(1, 101)) \
    .filter(lambda x: x % 2 == 0) \
    .map(lambda x: x ** 2) \
    .reduce(lambda a, b: a + b)
print(result)  # 171700
```

### 문제 2: Pair RDD
로그 파일에서 에러 수준별 로그 수를 집계하세요.

```python
# 입력: "2024-01-01 ERROR: Connection failed"
logs = sc.textFile("logs.txt")
error_counts = logs \
    .map(lambda line: line.split()[1].replace(":", "")) \
    .map(lambda level: (level, 1)) \
    .reduceByKey(lambda a, b: a + b) \
    .collect()
```

---

## 요약

| 개념 | 설명 |
|------|------|
| **Spark** | 대규모 데이터 처리 통합 엔진 |
| **RDD** | 기본 분산 데이터 구조 |
| **Transformation** | 새 RDD 생성 (Lazy) |
| **Action** | 결과 반환 (Eager) |
| **Driver** | 메인 프로그램 실행 노드 |
| **Executor** | Task 실행 워커 |

---

## 참고 자료

- [Apache Spark Documentation](https://spark.apache.org/docs/latest/)
- [PySpark API Reference](https://spark.apache.org/docs/latest/api/python/)
- [Learning Spark (O'Reilly)](https://www.oreilly.com/library/view/learning-spark-2nd/9781492050032/)
