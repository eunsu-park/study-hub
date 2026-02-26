# Spark SQL 최적화

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. explain() 메서드를 사용하여 Spark 실행 계획(Execution Plan)을 읽고 해석하여 성능 병목 지점을 파악할 수 있다
2. Catalyst 옵티마이저의 4단계(분석, 논리적 최적화, 물리적 계획, 코드 생성)를 기술하고 조건절 푸시다운(Predicate Pushdown)과 컬럼 가지치기(Column Pruning) 등 핵심 최적화를 설명할 수 있다
3. 파티셔닝 전략(repartition, coalesce, partitionBy)과 캐싱(cache, persist)을 적용하여 작업 성능을 향상시킬 수 있다
4. 데이터 크기에 따라 올바른 조인 전략(Broadcast, Sort-Merge, Shuffle Hash)을 선택하고 스큐 조인(Skew Join) 문제와 완화 기법을 설명할 수 있다
5. 런타임 쿼리 계획 최적화를 활성화하기 위해 적응형 쿼리 실행(AQE, Adaptive Query Execution) 설정을 구성할 수 있다
6. Spark UI와 모범 사례를 사용하여 데이터 쏠림(Data Skew), 과도한 셔플(Shuffle), 작은 파일 문제 등 일반적인 성능 이슈를 진단하고 해결할 수 있다

---

## 개요

Spark SQL의 성능을 최적화하기 위해서는 Catalyst 옵티마이저의 동작 원리를 이해하고, 파티셔닝, 캐싱, 조인 전략 등을 적절히 활용해야 합니다.

---

## 1. Catalyst Optimizer

### 1.1 실행 계획 이해

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("Optimization").getOrCreate()

df = spark.read.parquet("sales.parquet")

# 쿼리를 구성한다 — 아직 실행이 발생하지 않는다. Spark는 Catalyst 옵티마이저가
# 물리적 실행 전략을 선택하기 전에 재작성할 논리적 계획을 구성한다.
query = df.filter(col("amount") > 100) \
          .groupBy("category") \
          .sum("amount")

# "simple"은 물리적 계획만 표시한다 — Spark가 실제로 실행하는 것을 보려면 여기서 시작한다
query.explain(mode="simple")

# "extended"는 4가지 계획을 모두 표시한다 (파싱 → 분석 → 최적화 → 물리적) —
# Catalyst가 조건절 푸시다운 같은 예상 최적화를 적용했는지 확인하는 데 사용한다
query.explain(mode="extended")

# "cost"는 행 수와 크기 추정치를 추가한다 — Spark가 왜 브로드캐스트 조인 대신
# 소트-머지 조인을 선택했는지 이해하는 데 도움이 된다 (예상 테이블 크기 기반)
query.explain(mode="cost")

# "formatted"는 연산자 세부 정보가 있는 가장 읽기 쉬운 출력을 생성한다 — 셔플을
# 나타내는 Exchange 노드를 찾아 병목을 식별하는 데 최적이다
query.explain(mode="formatted")
```

### 1.2 Catalyst 최적화 단계

```
┌─────────────────────────────────────────────────────────────────┐
│                   Catalyst Optimizer 단계                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   1. Analysis (분석)                                            │
│      - 컬럼/테이블 이름 확인                                     │
│      - 타입 검증                                                │
│      ↓                                                          │
│   2. Logical Optimization (논리적 최적화)                        │
│      - Predicate Pushdown (조건절 푸시다운)                      │
│      - Column Pruning (컬럼 가지치기)                            │
│      - Constant Folding (상수 폴딩)                             │
│      ↓                                                          │
│   3. Physical Planning (물리적 계획)                             │
│      - 조인 전략 선택                                           │
│      - 집계 전략 선택                                           │
│      ↓                                                          │
│   4. Code Generation (코드 생성)                                │
│      - Whole-Stage Code Generation                              │
│      - JIT 컴파일                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 주요 최적화 기법

```python
# 1. 조건절 푸시다운(Predicate Pushdown) — Catalyst가 필터를 데이터 소스 스캔으로 이동시킨다.
# Parquet의 경우 min/max 통계가 일치하지 않는 전체 행 그룹을 건너뛰어
# 디스크에서 10-100배 적은 데이터를 읽을 수 있다.
df = spark.read.parquet("data.parquet")
filtered = df.filter(col("date") == "2024-01-01")  # Parquet이 이 날짜 범위 밖의 행 그룹을 건너뛴다

# 2. 컬럼 가지치기(Column Pruning) — Spark는 컬럼형 형식에서 참조된 컬럼만 읽는다.
# Parquet에서 참조되지 않은 컬럼은 디스크에서 역직렬화되지 않아 I/O와 메모리를 절약한다.
# SELECT * 대신 항상 특정 컬럼을 선택한다.
df.select("name", "amount")  # 다른 컬럼은 읽지 않음

# 3. 프로젝션 푸시다운(Projection Pushdown) — JDBC 소스의 경우 WHERE와 SELECT를
# 생성된 SQL 쿼리에 푸시하여 데이터베이스가 네트워크를 통해 필요한 데이터만 반환한다
df = spark.read.format("jdbc") \
    .option("pushDownPredicate", "true") \
    .load()

# 4. 상수 폴딩(Constant Folding) — Catalyst가 계획 시점에 상수 표현식을 평가하여
# 행별 반복 계산을 방지한다. 자동으로 발생하므로 튜닝이 필요 없다.
df.filter(col("value") > 1 + 2)  # 실행 전에 > 3으로 최적화됨
```

---

## 2. 파티셔닝

### 2.1 파티션 개념

```python
# 파티션 수 확인
df.rdd.getNumPartitions()

# repartition()은 완전한 셔플을 트리거한다 — 모든 레코드가 재분배된다.
# 파티션 수를 늘리거나 특정 컬럼으로 재분배해야 할 때 사용한다.
df.repartition(100)                      # 해시 기반으로 100개 파티션에 재분배
df.repartition("date")                   # 같은 날짜의 행을 함께 배치 — 날짜 기반 조인 전에 유용하다
df.repartition(100, "date", "category")  # 균등 분산을 위해 복합 키로 해시

# coalesce()는 인접 파티션을 병합하여 파티션 수만 줄인다 — 셔플이
# 필요 없다. filter() 연산 후 많은 거의 빈 파티션이 남을 때 사용한다.
# 파티션 수를 늘릴 수 없다 (그 경우 repartition을 사용한다).
df.coalesce(10)

# 진단 헬퍼 — 데이터 스큐 감지에 사용한다 (일부 파티션이 다른 것보다 훨씬 큰 경우).
# glom().collect()는 파티션 데이터를 드라이버로 가져오므로 작은 데이터셋에만 사용한다.
def print_partition_info(df):
    print(f"Partitions: {df.rdd.getNumPartitions()}")
    for idx, partition in enumerate(df.rdd.glom().collect()):
        print(f"Partition {idx}: {len(partition)} rows")
```

### 2.2 파티션 전략

```python
# 적절한 파티션 수 계산
"""
권장 공식:
- 파티션 수 = 데이터 크기(MB) / 128MB  → 파티션당 ~128MB를 목표로 한다 (Spark의 최적 지점)
- 또는: 클러스터 코어 수 * 2~4  → 모든 코어가 태스크 수준 병렬성으로 바쁘게 유지된다

너무 적은 파티션 → OOM이 발생하거나 코어를 충분히 활용하지 못하는 대형 태스크.
너무 많은 파티션 → 과도한 스케줄링 오버헤드와 작은 파일 문제.

예시:
- 10GB 데이터 → 10,000MB / 128MB ≈ 80 파티션
- 100 코어 클러스터 → 200~400 파티션
"""

# 이것은 모든 셔플(조인, groupBy 등)의 파티션 수를 설정한다 — 전역
# 기본값이다. AQE가 활성화되면 Spark가 런타임에 작은 파티션을 자동으로 병합할 수 있다.
spark.conf.set("spark.sql.shuffle.partitions", 200)

# repartitionByRange는 정렬된 비중첩 파티션을 생성한다 — 각 파티션이 연속적인
# 키 범위를 커버하므로 범위 기반 쿼리(예: 날짜 범위)에 효율적인 파티션 가지치기를 가능하게 한다.
df.repartitionByRange(100, "date")

# 해시 파티셔닝은 동일한 키가 항상 동일한 파티션으로 간다는 것을 보장한다 —
# 조인 자체 중에 셔플을 피하려면 user_id로 조인하기 전에 필수적이다.
df.repartition(100, "user_id")
```

### 2.3 파티션 저장

```python
# partitionBy는 Hive 스타일 디렉토리 구조를 생성한다 — year/month를 필터링하는 쿼리가
# 파일을 하나도 읽지 않고 전체 디렉토리를 건너뛸 수 있는 파티션 가지치기를 가능하게 한다.
# 파티션 컬럼을 신중하게 선택한다: 너무 높은 카디널리티 → 수백만 개의 작은 파일.
df.write \
    .partitionBy("year", "month") \
    .parquet("output/partitioned_data")

# 결과 디렉토리 구조:
# output/partitioned_data/
#   year=2024/
#     month=01/
#       part-00000.parquet
#     month=02/
#       part-00000.parquet

# Spark는 디렉토리 이름에서 파티션 컬럼을 인식한다 — 이 필터는
# year=2024/month=01 하위 디렉토리만 읽고 다른 모든 월/년은 완전히 건너뛴다
df = spark.read.parquet("output/partitioned_data")
df.filter((col("year") == 2024) & (col("month") == 1))

# 버킷팅은 쓰기 시점에 데이터를 사전 셔플하고 정렬한다 — 동일하게 버킷된 두 테이블 간의
# 후속 user_id 조인은 읽기 시점에 셔플이 전혀 필요하지 않다.
# 트레이드오프: 쓰기는 느리지만 동일한 키에 대한 반복 조인이 극적으로 빨라진다.
df.write \
    .bucketBy(100, "user_id") \      # 100개 버킷 = 100개 파일, 각각 user_id의 해시 범위를 포함
    .sortBy("timestamp") \            # 버킷 내 사전 정렬로 머지 조인과 범위 스캔 속도 향상
    .saveAsTable("bucketed_table")    # saveAsTable을 사용해야 함 (write.parquet 아님) — 버킷 메타데이터가 Hive 메타스토어에 저장됨
```

---

## 3. 캐싱

### 3.1 캐시 기본

```python
# cache()는 지연 평가된다 — 실제 캐싱은 첫 번째 액션에서 발생한다. 그 후 DataFrame이
# 이후 액션에서 executor 메모리에 유지되어 재계산을 방지한다.
df.cache()           # persist(MEMORY_AND_DISK)의 별칭
df.persist()         # cache()와 동일

# 메모리/CPU/디스크 트레이드오프에 따라 스토리지 레벨을 선택한다:
from pyspark import StorageLevel

df.persist(StorageLevel.MEMORY_ONLY)           # 가장 빠른 읽기지만 메모리가 부족하면 파티션을 퇴출한다 — 미스 시 재계산
df.persist(StorageLevel.MEMORY_AND_DISK)       # 재계산 대신 디스크로 스필 — 대부분의 워크로드에 최적의 기본값
df.persist(StorageLevel.MEMORY_ONLY_SER)       # 직렬화 = ~2-5배 적은 메모리지만 역직렬화를 위한 CPU 비용이 추가됨
df.persist(StorageLevel.DISK_ONLY)             # 데이터가 메모리에 너무 클 때 — 소스에서 재계산하는 것보다는 빠름
df.persist(StorageLevel.MEMORY_AND_DISK_SER)   # 직렬화 + 디스크 폴백 — 최대 메모리 효율

# 완료 시 항상 unpersist한다 — 캐시된 데이터는 셔플/실행에 사용될 수 있는
# executor 메모리를 차지한다. 오래된 캐시는 OOM의 일반적인 원인이다.
df.unpersist()

# 캐시 상태 확인
spark.catalog.isCached("table_name")
```

### 3.2 캐시 전략

```python
# 캐시 사용 시: 동일한 DataFrame이 2개 이상의 액션에서 사용되고 재계산이 비용이 클 때.
# 캐시하지 말아야 할 때: 일회성 변환, 또는 데이터가 메모리에 맞지 않을 때
# (캐싱이 퇴출 스레싱을 유발하여 성능이 더 나빠진다).

# 예시: 이 조인은 비용이 크다 (두 큰 테이블을 셔플). 캐싱 없이는
# 아래의 세 집계 각각이 전체 읽기 + 필터 + 조인을 반복한다.
expensive_df = spark.read.parquet("large_data.parquet") \
    .filter(col("status") == "active") \
    .join(other_df, "key")

# cache()는 DAG 중단점을 표시한다 — 첫 번째 액션이 결과를 구체화하고 저장한다
expensive_df.cache()

# 세 액션 모두 캐시된 결과를 재사용한다 — 매번 조인을 재계산하는 것보다 3배 빠르다
result1 = expensive_df.groupBy("category").count()
result2 = expensive_df.groupBy("region").sum("amount")
result3 = expensive_df.filter(col("amount") > 1000).count()

# 명시적 unpersist가 중요하다 — Spark의 LRU 퇴출은 메모리를 낭비하는 오래된 캐시를
# 유지할 수 있다. 오래 실행되는 애플리케이션에서는 항상 정리한다.
expensive_df.unpersist()
```

### 3.3 캐시 모니터링

```python
# Spark UI에서 확인 (http://localhost:4040/storage)

# 프로그래밍 방식 확인
sc = spark.sparkContext

# 캐시된 RDD 목록
for rdd_id, rdd_info in sc._jsc.sc().getRDDStorageInfo():
    print(f"RDD {rdd_id}: {rdd_info}")

# 전체 캐시 클리어
spark.catalog.clearCache()
```

---

## 4. 조인 전략

### 4.1 조인 유형별 특성

```python
# Spark 조인 전략:
join_strategies = {
    "Broadcast Hash Join": {
        "condition": "작은 테이블 (< 10MB 기본)",
        "performance": "가장 빠름",
        "shuffle": "없음 (작은 테이블 브로드캐스트)"
    },
    "Sort Merge Join": {
        "condition": "큰 테이블 간 조인",
        "performance": "안정적",
        "shuffle": "양쪽 테이블 셔플 + 정렬"
    },
    "Shuffle Hash Join": {
        "condition": "한쪽이 작을 때",
        "performance": "중간",
        "shuffle": "양쪽 셔플"
    },
    "Broadcast Nested Loop Join": {
        "condition": "조인 조건 없음 (Cross)",
        "performance": "느림",
        "shuffle": "없음 (브로드캐스트)"
    }
}
```

### 4.2 Broadcast Join 강제

```python
from pyspark.sql.functions import broadcast

# broadcast()는 전체 소규모 DataFrame을 모든 executor로 전송한다 — 대규모 측에서의
# 셔플을 완전히 제거한다. 소규모 테이블이 드라이버 + executor 메모리에 맞아야 한다.
large_df.join(broadcast(small_df), "key")

# 기본 자동 브로드캐스트 임계값은 10MB다 — 메모리에 여전히 맞는 더 큰 차원 테이블에서 늘린다.
# 너무 높게 설정하면 executor에서 OOM이 발생할 위험이 있다.
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", 100 * 1024 * 1024)  # 100MB

# 자동 브로드캐스트를 비활성화하여 소트-머지 조인을 강제한다 — 벤치마킹이나
# Spark의 크기 추정이 부정확할 때 유용하다 (복잡한 서브쿼리에서 일반적)
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)

# SQL 힌트는 옵티마이저의 결정을 재정의한다 — Spark의 비용 모델보다 더 잘 알 때 사용한다
# (예: 필터링 후 테이블이 브로드캐스트 크기로 줄어든 경우)
spark.sql("""
    SELECT /*+ BROADCAST(small_table) */
        large_table.*, small_table.name
    FROM large_table
    JOIN small_table ON large_table.id = small_table.id
""")
```

### 4.3 조인 최적화 팁

```python
# 1. 조인 전 필터링 — 셔플로 들어가는 행 수를 줄인다.
# Catalyst가 종종 자동으로 이를 수행하지만(조건절 푸시다운), 필터를 일찍 배치하면
# 의도가 명확해지고 푸시다운이 실패하는 경우에도 도움이 된다.
# 나쁜 예
df1.join(df2, "key").filter(col("status") == "active")

# 좋은 예 — 셔플하고 조인할 행이 적다
df1.filter(col("status") == "active").join(df2, "key")


# 2. 타입 불일치는 모든 행에 대해 암시적 캐스팅을 강제한다 — 조건절 푸시다운을 비활성화하고
# Spark가 최적화된 조인 경로를 사용하는 것을 방지한다.
# 나쁜 예 (타입 불일치로 암시적 캐스팅)
df1.join(df2, df1.id == df2.id)  # id가 string vs int

# 좋은 예 — 한 번 캐스트하고, 효율적으로 조인한다
df1 = df1.withColumn("id", col("id").cast("int"))
df1.join(df2, "id")


# 3. 스큐 조인은 런타임에 과도하게 큰 파티션을 분할한다 — 파티션이
# skewedPartitionFactor(5배)보다 크고 임계값(256MB)을 초과하면
# AQE가 자동으로 더 작은 서브 파티션으로 분할한다.
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", True)
spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionFactor", 5)
spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", "256MB")


# 4. 버킷팅은 쓰기 시점에 데이터를 사전 파티셔닝한다 — 두 테이블 모두 동일한
# 버킷 수와 키를 사용해야 한다. 이후 조인은 셔플을 완전히 건너뛴다.
# 일치하는 키가 두 측의 동일한 버킷 번호에 있음이 보장되기 때문이다.
df.write.bucketBy(100, "user_id").saveAsTable("users_bucketed")
other_df.write.bucketBy(100, "user_id").saveAsTable("orders_bucketed")

# 실행 계획에 Exchange(셔플) 노드가 없다 — .explain()으로 확인한다
spark.table("users_bucketed").join(spark.table("orders_bucketed"), "user_id")
```

---

## 5. 성능 튜닝

### 5.1 설정 최적화

```python
# 메모리 설정 — executor 메모리는 실행(셔플, 조인, 정렬)과 스토리지(캐시) 간에 분할된다.
# memory.fraction은 총 사용 가능한 비율(힙의 80%)을 제어하고,
# storageFraction은 캐싱을 위해 그 중 30%를 예약한다 — 조인/셔플이 OOM이 나면 낮춘다.
spark = SparkSession.builder \
    .config("spark.executor.memory", "8g") \
    .config("spark.executor.memoryOverhead", "2g") \   # Python, 네트워크 버퍼를 위한 오프힙 — PySpark 워크로드에서 증가시킨다
    .config("spark.driver.memory", "4g") \              # 브로드캐스트 변수와 collect() 결과를 위한 여유 공간이 필요하다
    .config("spark.memory.fraction", "0.8") \           # 20%는 사용자 데이터 구조와 내부 메타데이터를 위해 예약됨
    .config("spark.memory.storageFraction", "0.3") \    # 스토리지는 실행에서 빌릴 수 있지만 그 반대는 불가 (통합 메모리 모델)
    .getOrCreate()

# default.parallelism은 RDD 연산에 영향을 주고; shuffle.partitions는 DataFrame 연산에 영향을 준다.
# 균형 잡힌 태스크 수준 병렬성을 위해 둘 다 총 클러스터 코어 수의 2-3배로 설정한다.
spark.conf.set("spark.default.parallelism", 200)
spark.conf.set("spark.sql.shuffle.partitions", 200)

# AQE는 스테이지 경계에서 실제 런타임 통계를 사용하여 계획을 재최적화한다 —
# 정적 최적화로 예측할 수 없는 문제(스큐, 파티션 크기)를 처리한다.
spark.conf.set("spark.sql.adaptive.enabled", True)
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", True)    # 셔플 후 작은 파티션을 목표 크기로 병합
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", True)              # 조인 중 과도하게 큰 파티션을 분할
spark.conf.set("spark.sql.adaptive.localShuffleReader.enabled", True)    # 가능한 경우 셔플 데이터를 로컬에서 읽는다 (네트워크 방지)

# Kryo 직렬화는 Java 기본보다 10배 빠르고 2-5배 더 컴팩트하다 —
# 성능이 중요한 작업에 필수적이다. 오버헤드를 피하기 위해 커스텀 클래스를 등록한다.
spark.conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

# 동적 할당은 워크로드에 따라 executor를 자동 스케일링한다 — minExecutors는
# 콜드 스타트 지연을 방지하고, maxExecutors는 비용 제어를 위해 리소스 사용량을 제한한다.
spark.conf.set("spark.dynamicAllocation.enabled", True)
spark.conf.set("spark.dynamicAllocation.minExecutors", 2)
spark.conf.set("spark.dynamicAllocation.maxExecutors", 100)
```

### 5.2 데이터 형식 최적화

```python
# Snappy는 적당한 비율(~2배)로 빠른 압축/해제를 제공한다.
# zstd는 더 높은 CPU 비용으로 더 나은 압축(~3-4배)을 달성한다 — 콜드 스토리지에는 zstd를,
# 자주 읽는 핫 데이터에는 snappy를 선호한다.
spark.conf.set("spark.sql.parquet.compression.codec", "snappy")
spark.conf.set("spark.sql.parquet.filterPushdown", True)  # Parquet 행 그룹의 min/max 통계로 데이터를 건너뛴다

# maxPartitionBytes는 입력 분할 크기를 제한한다 — 128MB는 HDFS 블록 크기에 맞춰
# 블록당 하나의 태스크를 생성한다. openCostInBytes는 많은 소규모 파일 열기에 패널티를 주어
# Spark가 더 큰 분할로 결합하도록 유도한다.
spark.conf.set("spark.sql.files.maxPartitionBytes", "128MB")
spark.conf.set("spark.sql.files.openCostInBytes", "4MB")

# parallelismFirst=False는 AQE가 병렬성보다 목표 파티션 크기(128MB)를 우선시하도록 한다 —
# 다운스트림 읽기 성능을 저하시키는 "너무 많은 작은 파일" 문제 해결에 필수적이다.
spark.conf.set("spark.sql.adaptive.coalescePartitions.parallelismFirst", False)
spark.conf.set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "128MB")

# 항상 explain()으로 컬럼 가지치기를 확인한다 — ReadSchema에 요청한 컬럼만 보이면
# 가지치기가 작동 중이다. 전체 스키마 읽기는 문제를 나타낸다.
df.select("needed_column1", "needed_column2").explain()
```

### 5.3 셔플 최적화

```python
# AQE가 최선의 첫 번째 접근 방식이다 — 런타임에 작은 파티션을 동적으로 병합하고
# 큰 파티션을 분할하여 실제 데이터 분포에 적응한다.
spark.conf.set("spark.sql.adaptive.enabled", True)

# AQE를 사용할 수 없을 때 수동 계산: 파티션당 ~128MB를 목표로
# 태스크 스케줄링 오버헤드와 메모리 사용량의 균형을 맞춘다
data_size_gb = 10
partition_size_mb = 128
optimal_partitions = (data_size_gb * 1024) // partition_size_mb
spark.conf.set("spark.sql.shuffle.partitions", optimal_partitions)

# 셔플 압축은 CPU를 네트워크 I/O와 교환한다 — 셔플은 일반적으로
# CPU가 아닌 네트워크 바운드이므로 거의 항상 이득이다
spark.conf.set("spark.shuffle.compress", True)

# 스필 압축은 셔플 데이터가 executor 메모리를 초과할 때 디스크 사용량을 줄인다 —
# 없으면 스필된 데이터가 원시 디스크 공간을 사용하여 빠르게 채울 수 있다
spark.conf.set("spark.shuffle.spill.compress", True)

# 외부 셔플 서비스는 executor와 독립적으로 실행된다 — 동적 할당을 가능하게 한다
# (executor가 셔플 데이터를 잃지 않고 제거될 수 있다)
spark.conf.set("spark.shuffle.service.enabled", True)
```

---

## 6. 성능 모니터링

### 6.1 Spark UI 활용

```python
# Spark UI 접근: http://<driver-host>:4040

# UI 탭별 정보:
"""
Jobs: Job 실행 현황, 시간
Stages: Stage별 상세 (셔플, 데이터 크기)
Storage: 캐시된 RDD/DataFrame
Environment: 설정 값
Executors: Executor 상태, 메모리
SQL: SQL 쿼리 계획
"""

# 이력 서버 (완료된 작업)
# spark.eventLog.enabled=true
# spark.history.fs.logDirectory=hdfs:///spark-history
```

### 6.2 프로그래밍 방식 모니터링

```python
# 실행 시간 측정
import time

start = time.time()
result = df.groupBy("category").count().collect()
end = time.time()
print(f"Execution time: {end - start:.2f} seconds")

# 실행 계획에서 셔플 확인
df.explain(mode="formatted")

# 물리적 계획에서 조인 전략 확인
# Exchange = 셔플 발생
# BroadcastHashJoin = 브로드캐스트 조인
# SortMergeJoin = 소트 머지 조인
```

### 6.3 메트릭 수집

```python
# 크기 추정은 Catalyst 통계(실제 데이터 스캔이 아님)를 사용한다 — ANALYZE TABLE 없이는
# 결과가 부정확할 수 있다. 브로드캐스트 조인 vs 소트-머지 조인 결정에 유용하다.
def estimate_size(df):
    """DataFrame 크기 추정 (바이트)"""
    return df._jdf.queryExecution().optimizedPlan().stats().sizeInBytes()

# 파티션 수준 행 수는 데이터 스큐를 드러낸다 — 스큐 비율이 3-5배를 초과하면
# 일부 태스크가 다른 것보다 훨씬 느려져 병목이 된다.
# 참고: mapPartitions + collect는 DataFrame을 구체화해야 한다 —
# 전체 재계산을 트리거하지 않으려면 캐시된 데이터에서 사용한다.
partition_counts = df.rdd.mapPartitions(
    lambda it: [sum(1 for _ in it)]
).collect()

print(f"Min: {min(partition_counts)}, Max: {max(partition_counts)}")
print(f"Skew ratio: {max(partition_counts) / (sum(partition_counts) / len(partition_counts)):.2f}")
```

---

## 7. 일반적인 성능 문제와 해결

### 7.1 데이터 스큐 (Skew)

```python
# 문제: 특정 키에 데이터가 집중된다 (예: 한 고객의 주문이 90%)
# 증상: 하나의 태스크가 다른 것보다 100배 더 오래 실행된다 — Spark UI 스테이지 타임라인에서 확인 가능

# 해결 1: AQE가 런타임에 스큐를 감지하고 큰 파티션을 자동으로 서브 파티션으로 분할한다 —
# 가장 간단한 수정, 먼저 시도한다
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", True)

# 해결 2: 솔트 키 — 핫 키를 N개의 서브키로 수동으로 분리하여 데이터를
# N개의 파티션에 분산시킨다. 트레이드오프: 크로스 조인으로 다른 테이블을 확장해야 하여 크기가 N×가 된다.
from pyspark.sql.functions import rand, floor

num_salts = 10  # 스큐 심각도에 따라 선택 — 높을수록 더 많은 병렬성이지만 더 큰 크로스 조인
df_salted = df.withColumn("salt", floor(rand() * num_salts))

# 소규모 테이블은 N번 복제된다 (솔트 값당 하나) — 대규모 측의 모든 솔트된
# 키-파티션이 소규모 측에서 일치하는 행을 찾을 수 있다
result = df_salted.join(
    other_df.crossJoin(
        spark.range(num_salts).withColumnRenamed("id", "salt")
    ),
    ["key", "salt"]
).drop("salt")

# 해결 3: 브로드캐스트는 셔플을 완전히 피한다 — 스큐된 테이블이 executor
# 메모리에 맞는 소규모 차원 테이블과 조인될 때 최선의 방법이다
result = df.join(broadcast(small_df), "key")
```

### 7.2 OOM (Out of Memory)

```python
# 문제: 메모리 부족 — executor(파티션당 데이터가 너무 많음)나
# 드라이버(collect/broadcast가 너무 큼)에서 발생할 수 있다. 어디서 발생했는지 확인하기 위해 에러 스택트레이스를 확인한다.

# 해결 1: executor당 더 많은 메모리 — 빠른 수정이지만 비용이 많이 든다.
# PySpark에서는 Python 워커가 오프힙 메모리를 사용하므로 오버헤드도 증가시킨다.
spark.conf.set("spark.executor.memory", "8g")
spark.conf.set("spark.executor.memoryOverhead", "2g")

# 해결 2: 더 많은 파티션 = 태스크당 적은 데이터 = 낮은 태스크당 메모리.
# 메모리 추가보다 종종 더 효과적이다 — RAM을 투입하는 대신 문제를 분산시킨다.
df.repartition(500)

# 해결 3: 오래된 캐시는 숨겨진 OOM 원인이다 — 캐시된 DataFrame이 실행(셔플/조인)이
# 회수할 수 없는 스토리지 메모리를 차지한다
spark.catalog.clearCache()

# 해결 4: 브로드캐스트 조인이 OOM을 일으키면 "소규모" 테이블이 예상보다 컸다.
# 임계값을 낮추면 전체 테이블을 메모리에 로드하는 대신 데이터를 스트리밍하는 소트-머지 조인이 강제된다.
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "10MB")
```

### 7.3 셔플 과다

```python
# 문제: 셔플로 인한 네트워크/디스크 I/O
# 증상: Stage 간 대기 시간 증가

# 해결 1: 셔플 전 필터링
df.filter(col("status") == "active").groupBy("key").count()

# 해결 2: 파티셔닝 전략 변경
# 같은 키로 파티셔닝된 데이터는 셔플 없이 조인
df1.repartition(100, "key").join(df2.repartition(100, "key"), "key")

# 해결 3: 버킷팅 사용
df.write.bucketBy(100, "key").saveAsTable("bucketed_table")
```

---

## 연습 문제

### 문제 1: 실행 계획 분석
주어진 쿼리의 실행 계획을 분석하고 최적화 포인트를 찾으세요.

### 문제 2: 조인 최적화
1억 건의 트랜잭션 테이블과 100만 건의 고객 테이블을 조인하는 최적의 방법을 설계하세요.

### 문제 3: 스큐 처리
특정 카테고리에 데이터가 집중된 상황에서 집계 성능을 개선하세요.

---

## 요약

| 최적화 영역 | 기법 |
|-------------|------|
| **Catalyst** | Predicate Pushdown, Column Pruning |
| **파티셔닝** | repartition, coalesce, partitionBy |
| **캐싱** | cache, persist, StorageLevel |
| **조인** | Broadcast, Sort Merge, 버킷팅 |
| **AQE** | 자동 파티션 병합, 스큐 처리 |

---

## 참고 자료

- [Spark SQL Tuning](https://spark.apache.org/docs/latest/sql-performance-tuning.html)
- [Spark Configuration](https://spark.apache.org/docs/latest/configuration.html)
- [Adaptive Query Execution](https://spark.apache.org/docs/latest/sql-performance-tuning.html#adaptive-query-execution)
