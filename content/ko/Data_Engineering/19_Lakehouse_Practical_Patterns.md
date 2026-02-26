# 레이크하우스 실전 패턴(Lakehouse Practical Patterns)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 각 레이어에서 적절한 스키마 제약, 중복 제거, 데이터 품질 규칙을 적용하는 메달리온 아키텍처(브론즈, 실버, 골드)를 설계할 수 있습니다.
2. Delta Lake의 DeltaTable API 또는 Apache Iceberg의 MergeIntoTable을 사용하여 대용량 테이블을 효율적으로 업데이트하는 증분 MERGE(업서트) 연산을 구현할 수 있습니다.
3. SCD Type 2 패턴을 적용하여 유효 날짜(Effective Date)와 현재 레코드 플래그(Current Record Flag)를 캡처함으로써 차원 테이블의 이력 변경 사항을 추적할 수 있습니다.
4. Delta Lake와 Iceberg의 시간 여행(Time Travel) 기능을 사용하여 과거 스냅샷을 쿼리하고, 변경 사항을 감사하며, 실수로 인한 쓰기를 롤백할 수 있습니다.
5. 대용량 레이크하우스 테이블의 쿼리 성능을 극대화하기 위해 테이블 압축(OPTIMIZE / rewrite_data_files), Z-ORDER 클러스터링, 파티션 가지치기(Partition Pruning)를 구성할 수 있습니다.
6. ACID 보장, 스키마 진화(Schema Evolution) 기능, 다중 엔진 상호 운용성(Spark, Trino, Flink) 측면에서 Delta Lake와 Apache Iceberg를 비교할 수 있습니다.

---

## 개요

레이크하우스(Lakehouse) 아키텍처는 데이터 웨어하우스(Data Warehouse)의 신뢰성과 데이터 레이크(Data Lake)의 확장성을 결합합니다. 이 레슨에서는 Delta Lake와 Apache Iceberg의 프로덕션 패턴을 다룹니다: 메달리온 아키텍처(Medallion Architecture), MERGE를 활용한 증분 처리(Incremental Processing), 천천히 변하는 차원(SCD Type 2), 압축(Compaction), 시간 여행(Time Travel), 그리고 다중 엔진 상호 운용성(Multi-engine Interoperability).

---

## 1. 메달리온 아키텍처(Medallion Architecture)

### 1.1 3계층 설계

```python
"""
Medallion Architecture (Bronze → Silver → Gold):

┌────────────┐    ┌────────────┐    ┌────────────┐
│   Bronze    │───→│   Silver    │───→│    Gold     │
│  (Raw)      │    │ (Cleaned)   │    │ (Business)  │
└────────────┘    └────────────┘    └────────────┘

Bronze Layer:
  - Raw ingestion (append-only)
  - Preserves original format
  - Includes metadata: ingestion timestamp, source, batch ID
  - Schema enforcement: minimal (accept all)
  - Retention: long (years) for replay capability

Silver Layer:
  - Deduplicated, validated, conformed
  - Standard schemas, data types fixed
  - Null handling, quality checks applied
  - Joins with reference data
  - Retention: medium (months to years)

Gold Layer:
  - Business-level aggregations
  - Pre-computed KPIs, metrics
  - Optimized for BI queries (star schema)
  - Retention: as needed by business
"""
```

### 1.2 Delta Lake로 메달리온 구현하기

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, current_timestamp, input_file_name, lit,
    from_json, to_timestamp, when, count, sum as spark_sum,
    window, row_number,
)
from pyspark.sql.window import Window
from delta.tables import DeltaTable

# Delta Lake는 커스텀 SQL 명령(OPTIMIZE, VACUUM, MERGE)을 등록하고
# spark.sql()이 Delta 테이블을 인식하기 위해 이 두 Spark 설정이 필요하다.
# 없으면 Delta 연산이 ClassNotFoundException으로 실패한다.
spark = SparkSession.builder \
    .appName("MedallionArchitecture") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()


# ── Bronze: Raw Ingestion ───────────────────────────────────────────

def ingest_bronze(source_path, bronze_path):
    """Ingest raw data into Bronze layer (append-only)."""
    raw = spark.read.json(source_path)

    # 메타데이터 컬럼(_ingested_at, _source_file, _batch_id)은 디버깅과
    # 재처리(replay)를 가능하게 한다: 어떤 행이 언제 도착했는지,
    # 어떤 파일에서 왔는지, 어떤 배치가 생성했는지 추적할 수 있다.
    bronze = raw \
        .withColumn("_ingested_at", current_timestamp()) \
        .withColumn("_source_file", input_file_name()) \
        .withColumn("_batch_id", lit("batch_001"))

    # mergeSchema=true는 Bronze 레이어가 업스트림 소스의 스키마 변경(예: 프로듀서가
    # 새 컬럼 추가)을 수용할 수 있게 한다. Bronze에서는 의도적인 선택이며 —
    # 스키마 적용은 Silver에서 이루어진다.
    bronze.write \
        .format("delta") \
        .mode("append") \
        .option("mergeSchema", "true") \
        .save(bronze_path)

    print(f"Bronze: {bronze.count()} rows ingested")


# ── Silver: Clean and Deduplicate ───────────────────────────────────

def process_silver(bronze_path, silver_path):
    """Process Bronze → Silver: clean, validate, deduplicate."""
    bronze = spark.read.format("delta").load(bronze_path)

    # 1. Parse and type-cast
    # 금융 계산에서 부동소수점 반올림 오류를 피하기 위해 float 대신
    # decimal(10,2)을 사용한다 (예: 99.99가 99.98999...로 저장되는 문제).
    cleaned = bronze \
        .filter(col("order_id").isNotNull()) \
        .withColumn("amount", col("amount").cast("decimal(10,2)")) \
        .withColumn("order_time", to_timestamp(col("timestamp")))

    # 2. Deduplicate: keep latest record per order_id
    # Bronze는 추가 전용(append-only)이므로 같은 order_id가 여러 번 나타날 수 있다
    # (예: 재수집, 늦은 수정). _ingested_at DESC 정렬의 row_number()는
    # 항상 가장 최신 버전을 선택하도록 보장한다.
    w = Window.partitionBy("order_id").orderBy(col("_ingested_at").desc())
    deduplicated = cleaned \
        .withColumn("_rn", row_number().over(w)) \
        .filter(col("_rn") == 1) \
        .drop("_rn")

    # 3. Quality checks — 이 조건들을 통과하지 못한 행은 조용히 제거된다.
    # 프로덕션에서는 거부된 행을 격리(quarantine) 테이블에 기록하여 조사할 것.
    valid = deduplicated.filter(
        (col("amount") > 0) &
        (col("order_time").isNotNull()) &
        (col("status").isin("pending", "shipped", "delivered", "cancelled"))
    )

    # 4. MERGE는 Silver를 멱등적(idempotent)으로 만든다: 같은 배치를 재실행해도
    # 기존 행이 업데이트될 뿐 중복되지 않는다. _ingested_at 조건은 더 오래된
    # 재처리 배치가 최신 버전을 덮어쓰는 것을 방지한다.
    if DeltaTable.isDeltaTable(spark, silver_path):
        silver_table = DeltaTable.forPath(spark, silver_path)
        silver_table.alias("target").merge(
            valid.alias("source"),
            "target.order_id = source.order_id"
        ).whenMatchedUpdateAll(
            condition="source._ingested_at > target._ingested_at"
        ).whenNotMatchedInsertAll().execute()
    else:
        valid.write.format("delta").save(silver_path)

    print(f"Silver: {valid.count()} valid rows processed")


# ── Gold: Business Aggregations ─────────────────────────────────────

def build_gold_daily_summary(silver_path, gold_path):
    """Build Gold layer: daily order summary."""
    silver = spark.read.format("delta").load(silver_path)

    daily = silver \
        .withColumn("order_date", col("order_time").cast("date")) \
        .groupBy("order_date", "status") \
        .agg(
            count("*").alias("order_count"),
            spark_sum("amount").alias("total_amount"),
        )

    # replaceWhere는 이번 배치에 있는 날짜 파티션만 선택적으로 덮어쓰고
    # 오래된 날짜는 그대로 유지한다. 이는 모든 Gold 데이터를 삭제하는
    # mode="overwrite"보다 더 효율적이고 안전하다. 또한 동일한 날짜를
    # 재실행해도 행이 중복되지 않아 멱등적(idempotent)인 연산이 된다.
    daily.write \
        .format("delta") \
        .mode("overwrite") \
        .option("replaceWhere",
                f"order_date >= '{daily.agg({'order_date': 'min'}).first()[0]}'") \
        .save(gold_path)

    print(f"Gold daily summary: {daily.count()} rows")
```

---

## 2. MERGE를 활용한 증분 처리(Incremental Processing with MERGE)

### 2.1 Delta Lake MERGE (업서트)

```python
def upsert_orders(new_data_path, target_path):
    """Incremental upsert: INSERT new rows, UPDATE existing ones."""
    new_data = spark.read.json(new_data_path)

    target = DeltaTable.forPath(spark, target_path)

    target.alias("t").merge(
        new_data.alias("s"),
        "t.order_id = s.order_id"
    ).whenMatchedUpdate(
        # updated_at 가드는 순서가 어긋난 재처리나 병렬 쓰기 시
        # 오래된 배치가 최신 데이터를 덮어쓰는 것을 방지한다.
        condition="s.updated_at > t.updated_at",
        # 컬럼을 명시적으로 나열하면 MERGE가 수정하는 필드를 명확히 문서화한다 —
        # 스키마 변경 후 새 소스 컬럼을 조용히 포함시키는 UpdateAll보다 안전하다.
        set={
            "status": "s.status",
            "amount": "s.amount",
            "updated_at": "s.updated_at",
        }
    ).whenNotMatchedInsert(
        values={
            "order_id": "s.order_id",
            "customer_id": "s.customer_id",
            "status": "s.status",
            "amount": "s.amount",
            "created_at": "s.created_at",
            "updated_at": "s.updated_at",
        }
    ).whenNotMatchedBySourceDelete(
        # 이 절은 소스 배치에 더 이상 존재하지 않는 대상 행을 삭제하지만,
        # 30일 이상 된 경우에만 적용한다. 나이 제한(age guard)은 특정
        # 증분 배치에 포함되지 않은 최근 행이 실수로 삭제되는 것을 방지한다.
        condition="t.updated_at < current_date() - INTERVAL 30 DAYS"
    ).execute()
```

### 2.2 CDC 이벤트를 Delta Lake에 적용하기

```python
def apply_cdc_to_delta(cdc_batch_df, batch_id, target_path):
    """Apply Debezium CDC events to a Delta table via foreachBatch.

    Handles INSERT (op=c/r), UPDATE (op=u), and DELETE (op=d).
    """
    if cdc_batch_df.isEmpty():
        return

    target = DeltaTable.forPath(spark, target_path)

    # 삭제와 업서트는 별도로 처리해야 한다. MERGE는 소스가 다른 연산을 위한
    # 일치 및 비일치 키를 모두 포함할 때 하나의 연산으로 행 삭제와 업서트를
    # 동시에 수행할 수 없기 때문이다.
    deletes = cdc_batch_df.filter(col("op") == "d")
    upserts = cdc_batch_df.filter(col("op").isin("c", "u", "r"))

    # Apply upserts (INSERT or UPDATE)
    if upserts.count() > 0:
        # 삽입/업데이트의 경우 행의 현재 상태는 'after' 필드에 있다.
        # select("after.*")는 중첩된 struct를 대상 테이블 스키마와 일치하는
        # 최상위 컬럼으로 평탄화(flatten)한다.
        upsert_data = upserts.select("after.*")
        target.alias("t").merge(
            upsert_data.alias("s"),
            "t.id = s.id"
        ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()

    # Apply deletes
    if deletes.count() > 0:
        # 삭제 시 'after' 필드가 null이므로 마지막으로 알려진 행 상태는
        # 'before' 필드에 있다. 대상 테이블에서 제거할 행을 식별하려면
        # 기본 키만 필요하다.
        delete_keys = deletes.select("before.id").collect()
        delete_ids = [row["id"] for row in delete_keys]
        target.delete(col("id").isin(delete_ids))

    print(f"CDC batch {batch_id}: "
          f"{upserts.count()} upserts, {deletes.count()} deletes")
```

---

## 3. 천천히 변하는 차원 타입 2(Slowly Changing Dimensions — SCD Type 2)

### 3.1 Delta Lake로 SCD Type 2 구현하기

```python
"""
SCD Type 2: Preserve full history of dimension changes.

Each row has:
  - effective_from: when this version became active
  - effective_to:   when this version was superseded (null = current)
  - is_current:     boolean flag for the active version

Customer changes address:
  id=1, name='Alice', city='NYC', effective_from='2024-01-01', effective_to=null, is_current=true
  ↓ Alice moves to LA on 2024-06-15
  id=1, name='Alice', city='NYC', effective_from='2024-01-01', effective_to='2024-06-14', is_current=false
  id=1, name='Alice', city='LA',  effective_from='2024-06-15', effective_to=null, is_current=true
"""


def scd_type2_merge(updates_df, target_path, key_col, tracked_cols,
                    effective_date_col="effective_date"):
    """Apply SCD Type 2 logic to a Delta table.

    Args:
        updates_df: DataFrame with new/changed records
        target_path: Delta table path
        key_col: Business key column (e.g., "customer_id")
        tracked_cols: Columns to track for changes (e.g., ["city", "email"])
        effective_date_col: Column with the change date
    """
    target = DeltaTable.forPath(spark, target_path)

    # tracked_cols에서 변경 감지 조건을 동적으로 구성한다.
    # 이 특정 컬럼들의 변경만이 새 SCD2 버전을 트리거한다 —
    # tracked_cols에 없는 컬럼(예: last_login)은 이력 레코드를 생성하지 않고
    # 변경될 수 있어 버전 폭발(version explosion)을 줄인다.
    change_condition = " OR ".join(
        f"target.{c} != source.{c}" for c in tracked_cols
    )

    # 현재 레코드(is_current=True)만 비교한다 — 이력 버전은 닫혀 있어
    # 다시 매칭되어서는 안 된다.
    current = spark.read.format("delta").load(target_path) \
        .filter(col("is_current") == True)

    # LEFT 조인은 대상에 매칭이 없는 신규 레코드도 포함시킨다.
    # 필터는 두 가지 경우를 포착한다: 완전히 새로운 엔티티 AND
    # 추적 컬럼이 변경된 기존 엔티티.
    changes = updates_df.alias("source").join(
        current.alias("target"),
        col(f"source.{key_col}") == col(f"target.{key_col}"),
        "left",
    ).filter(
        col(f"target.{key_col}").isNull() | expr(change_condition)
    )

    # Rows to close (expire old version)
    rows_to_close = changes.filter(
        col(f"target.{key_col}").isNotNull()
    ).select(
        col(f"target.{key_col}").alias(key_col),
        col(f"source.{effective_date_col}"),
    )

    # Rows to insert (new version)
    rows_to_insert = changes.select(
        col(f"source.{key_col}"),
        *[col(f"source.{c}") for c in tracked_cols],
        col(f"source.{effective_date_col}").alias("effective_from"),
        lit(None).cast("date").alias("effective_to"),
        lit(True).alias("is_current"),
    )

    # SCD2는 두 단계가 필요하다: 먼저 닫은 후 삽입. 순서가 중요하다 —
    # 먼저 닫으면 같은 엔티티에 대해 두 레코드가 모두 is_current=True로
    # 표시되는 순간이 절대 없도록 보장한다.
    if rows_to_close.count() > 0:
        close_keys = [row[key_col] for row in rows_to_close.collect()]
        target.update(
            condition=(col(key_col).isin(close_keys)) & (col("is_current") == True),
            set={
                "effective_to": expr(f"date_sub(current_date(), 1)"),
                "is_current": lit(False),
            },
        )

    # 새 버전은 MERGE가 아닌 Append로 삽입한다. 각 새 버전은 다른
    # effective_from을 가진 완전히 새로운 행이므로 매칭할 기존 행이 없다.
    rows_to_insert.write \
        .format("delta") \
        .mode("append") \
        .save(target_path)

    print(f"SCD2: {rows_to_close.count()} closed, {rows_to_insert.count()} inserted")
```

---

## 4. 테이블 유지 관리(Table Maintenance)

### 4.1 압축(Compaction)과 최적화(Optimization)

```python
"""
Delta Lake creates many small files (especially with streaming).
Compaction combines small files into larger ones for better read performance.
"""


def optimize_delta_table(table_path):
    """Run compaction and Z-ordering on a Delta table."""
    # OPTIMIZE는 소형 파일들을 ~1GB 파일로 병합한다. 스트리밍 쓰기는
    # 많은 작은 파일을 생성하며(마이크로배치당 하나), 각 파일은 쿼리 플래닝에
    # 오버헤드를 추가한다. 압축(compaction)은 읽기 성능을 10~100배 향상시킬 수 있다.
    spark.sql(f"OPTIMIZE delta.`{table_path}`")

    # Z-ORDER는 유사한 (order_date, customer_id) 값을 가진 행을 동일 파일에
    # 물리적으로 함께 배치한다. 이 컬럼으로 필터링하는 쿼리는 min/max 통계로
    # 전체 파일을 건너뛰어 I/O를 크게 줄인다. 가장 자주 사용되는 WHERE 절에
    # 기반하여 Z-ORDER 컬럼을 선택할 것.
    spark.sql(f"""
        OPTIMIZE delta.`{table_path}`
        ZORDER BY (order_date, customer_id)
    """)


def vacuum_old_versions(table_path, retention_hours=168):
    """Remove old file versions no longer referenced.

    Default retention: 7 days (168 hours).
    Files older than retention are deleted.
    Time travel will no longer work for versions beyond retention.
    """
    # VACUUM은 되돌릴 수 없다 — 오래된 파일이 삭제되면 해당 버전으로
    # 시간 여행(time travel)이 불가능하다. 보존 기간을 168시간 미만으로
    # 설정하려면 안전 검사를 비활성화해야 한다:
    # spark.databricks.delta.retentionDurationCheck.enabled
    spark.sql(f"VACUUM delta.`{table_path}` RETAIN {retention_hours} HOURS")


def auto_compact_config():
    """Configure auto-compaction for streaming writes."""
    # optimizeWrite는 각 쓰기 내에서 데이터를 재파티셔닝하여 더 적고 큰 파일을
    # 생성한다 — 소형 파일 문제를 소스에서 줄인다.
    spark.conf.set("spark.databricks.delta.optimizeWrite.enabled", "true")
    # autoCompact는 작은 파일이 너무 많이 쌓이면 각 쓰기 후 경량 압축을 트리거한다.
    # 별도의 OPTIMIZE 작업이 필요 없어지지만 쓰기 지연(write latency)이
    # ~5~10% 증가한다.
    spark.conf.set("spark.databricks.delta.autoCompact.enabled", "true")
    # 128MB 목표는 읽기 성능(열 파일 수 감소)과 압축 중 메모리 사용 간의
    # 균형을 맞춘다. 대형 테이블에는 높이고, 소형 또는 자주 쿼리되는 테이블에는 낮출 것.
    spark.conf.set("spark.databricks.delta.autoCompact.minFileSize", "134217728")
```

### 4.2 시간 여행(Time Travel)

```python
def time_travel_queries(table_path):
    """Query historical versions of a Delta table."""
    # versionAsOf는 정확한 버전 번호를 읽는다. 버전 0은 테이블의 초기 생성
    # 상태다 — 현재 데이터와 원본을 비교하여 드리프트(drift)를 감지하거나
    # 변환 결과를 검증하는 데 유용하다.
    v0 = spark.read.format("delta") \
        .option("versionAsOf", 0) \
        .load(table_path)

    # timestampAsOf는 "버전 번호"가 아닌 "시점"을 알 때 더 직관적이다.
    # Delta는 주어진 타임스탬프 이전의 최신 버전을 찾는다.
    # 감사 쿼리에 유용하다: "어제 CFO가 본 데이터를 보여줘."
    historical = spark.read.format("delta") \
        .option("timestampAsOf", "2024-06-15T10:00:00") \
        .load(table_path)

    # DESCRIBE HISTORY는 모든 연산(INSERT, MERGE, OPTIMIZE 등)을
    # 행 수와 실행 시간과 함께 보여준다 — 테이블의 행 수가 예상치 못하게
    # 변경되었을 때 디버깅에 필수적이다.
    history = spark.sql(f"DESCRIBE HISTORY delta.`{table_path}`")
    history.select("version", "timestamp", "operation", "operationMetrics").show()

    # RESTORE는 테이블을 버전 5와 일치하도록 물리적으로 재작성한다. 이는
    # 이후 버전 삭제가 아닌 새 버전 생성이므로 전체 감사 추적(audit trail)을
    # 보존한다. RESTORE 후 VACUUM이 참조되지 않은 파일을 최종적으로 회수한다.
    spark.sql(f"RESTORE TABLE delta.`{table_path}` TO VERSION AS OF 5")

    return v0, historical
```

---

## 5. Apache Iceberg 패턴

### 5.1 Iceberg 테이블 연산

```python
"""
Apache Iceberg provides similar capabilities with a catalog-centric approach.
Iceberg works with Spark, Trino, Flink, and other engines simultaneously.
"""

# "local" 카탈로그 이름은 임의적이다 — 프로덕션에서는 REST 카탈로그나
# AWS Glue를 가리키는 "iceberg_prod"를 사용할 수 있다. hadoop 타입은
# 메타데이터를 파일로 저장하여(서버 불필요) 단순하지만 동시 쓰기 안전성이 부족하다.
# 다중 엔진 프로덕션 사용에는 REST 또는 Hive 카탈로그를 사용할 것.
spark_iceberg = SparkSession.builder \
    .appName("IcebergPatterns") \
    .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.local.type", "hadoop") \
    .config("spark.sql.catalog.local.warehouse", "/data/iceberg/warehouse") \
    .getOrCreate()


def iceberg_crud():
    """Basic Iceberg table operations."""
    # Iceberg는 "숨겨진 파티셔닝(hidden partitioning)"을 사용한다 — months(order_date)는
    # 중복 파티션 컬럼 추가 없이 월별 파티션을 생성한다.
    # order_date로 필터링하는 쿼리는 파티션 변환을 명시적으로 참조하지 않아도
    # 자동으로 파티션 가지치기(pruning)의 혜택을 받는다.
    spark_iceberg.sql("""
        CREATE TABLE IF NOT EXISTS local.db.orders (
            order_id    BIGINT,
            customer_id STRING,
            amount      DECIMAL(10, 2),
            status      STRING,
            order_date  DATE
        )
        USING iceberg
        PARTITIONED BY (months(order_date))
    """)

    # Insert data
    spark_iceberg.sql("""
        INSERT INTO local.db.orders VALUES
        (1, 'C001', 99.99, 'shipped', DATE '2024-06-15'),
        (2, 'C002', 149.50, 'pending', DATE '2024-06-16')
    """)

    # MERGE (upsert)
    spark_iceberg.sql("""
        MERGE INTO local.db.orders t
        USING updates s
        ON t.order_id = s.order_id
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """)

    # Time travel
    spark_iceberg.sql("""
        SELECT * FROM local.db.orders
        FOR SYSTEM_VERSION AS OF 1
    """)

    # Snapshots
    spark_iceberg.sql("""
        SELECT * FROM local.db.orders.snapshots
    """).show()


def iceberg_partition_evolution():
    """Iceberg partition evolution: change partitioning without rewriting."""
    # 데이터 볼륨이 낮을 때 적합한 월별 파티셔닝으로 시작한다.
    spark_iceberg.sql("""
        CREATE TABLE local.db.events (
            event_id  BIGINT,
            event_ts  TIMESTAMP,
            payload   STRING
        )
        USING iceberg
        PARTITIONED BY (months(event_ts))
    """)

    # 파티션 진화(partition evolution)는 Delta Lake 대비 Iceberg의 핵심 기능이다.
    # 기존 데이터 파일을 재작성하지 않고도 미래 쓰기를 일별 세분성으로 변경한다.
    # Delta Lake에서 파티셔닝을 변경하려면 전체 테이블 재작성이 필요하다
    # (대형 테이블에서는 몇 시간이 걸릴 수 있다).
    spark_iceberg.sql("""
        ALTER TABLE local.db.events
        ADD PARTITION FIELD days(event_ts)
    """)
    spark_iceberg.sql("""
        ALTER TABLE local.db.events
        DROP PARTITION FIELD months(event_ts)
    """)
    # 오래된 데이터는 월별 파티션에 남고, 새 데이터는 일별 파티션으로 간다.
    # Iceberg의 메타데이터 레이어는 어떤 파티션 스펙이 어떤 데이터 파일에
    # 적용되는지 추적하므로 쿼리가 두 레이아웃을 투명하게 읽는다.
    # Hive 스타일 파티셔닝은 파티션 정보를 디렉토리 경로에 인코딩하는 것과 달리,
    # Iceberg는 파티션 정보를 매니페스트 파일에 파일별로 저장하므로 이것이 가능하다.
```

### 5.2 Iceberg 테이블 유지 관리

```python
def iceberg_maintenance():
    """Iceberg table maintenance operations."""
    # 스냅샷 만료는 메타데이터 스토리지를 확보하고 쿼리 플래닝 시간을 줄인다.
    # retain_last=10은 시간 여행과 디버깅을 위한 충분한 스냅샷을 유지하면서,
    # older_than은 동시 쿼리가 아직 읽고 있을 수 있는 매우 최근 스냅샷을
    # 삭제하지 않도록 방지한다.
    spark_iceberg.sql("""
        CALL local.system.expire_snapshots(
            table => 'db.orders',
            older_than => TIMESTAMP '2024-06-10 00:00:00',
            retain_last => 10
        )
    """)

    # 압축(compaction)은 소형 파일들을 ~128MB 파일로 병합한다. Delta의 OPTIMIZE와
    # 달리 Iceberg의 rewrite_data_files는 Iceberg 트랜잭션 모델 내에서 동작한다 —
    # 압축된 파일이 새 스냅샷이 되므로 동시 읽기 작업이 일관된 데이터를 계속 볼 수 있다.
    spark_iceberg.sql("""
        CALL local.system.rewrite_data_files(
            table => 'db.orders',
            options => map('target-file-size-bytes', '134217728')
        )
    """)

    # 매니페스트 파일은 각 스냅샷에 속한 데이터 파일을 인덱싱한다.
    # 시간이 지남에 따라 많은 소형 매니페스트가 축적된다. 이를 더 적고 큰
    # 매니페스트로 재작성하면 쿼리 플래닝이 빨라진다(스캔 계획 구성 시
    # 읽어야 할 파일 수 감소).
    spark_iceberg.sql("""
        CALL local.system.rewrite_manifests('db.orders')
    """)

    # 고아 파일(orphan files)은 어떤 스냅샷에도 참조되지 않는 데이터 파일이다 —
    # 일반적으로 실패한 쓰기나 만료된 스냅샷이 남긴다. 제거하면 스토리지를 회수한다.
    # older_than 가드는 진행 중인 쓰기의 파일이 삭제되는 것을 방지한다.
    spark_iceberg.sql("""
        CALL local.system.remove_orphan_files(
            table => 'db.orders',
            older_than => TIMESTAMP '2024-06-01 00:00:00'
        )
    """)
```

---

## 6. Delta Lake vs Iceberg 비교

```python
"""
Delta Lake vs Apache Iceberg:

| Feature              | Delta Lake               | Apache Iceberg            |
|----------------------|--------------------------|---------------------------|
| Originated by        | Databricks               | Netflix → Apache          |
| ACID Transactions    | Yes                      | Yes                       |
| Time Travel          | Yes (version/timestamp)  | Yes (snapshot-based)      |
| Schema Evolution     | Yes (merge on read)      | Yes (full evolution)      |
| Partition Evolution   | No (must rewrite)        | Yes (no rewrite needed)   |
| Multi-Engine         | Limited (Spark-centric)  | Excellent (Spark/Trino/Flink) |
| Catalog              | Hive Metastore / Unity   | REST / Hive / Glue / Nessie |
| File Format          | Parquet only             | Parquet, ORC, Avro        |
| Compaction           | OPTIMIZE command         | rewrite_data_files proc   |
| Z-Ordering           | Built-in                 | Sort order (similar)      |
| Merge Performance    | Optimized (Delta 3.0+)   | Copy-on-write / MoR       |
| Community            | Databricks ecosystem     | Broad multi-vendor        |

When to use Delta Lake:
  - Databricks ecosystem
  - Spark-only workloads
  - Simpler setup

When to use Iceberg:
  - Multi-engine (Spark + Trino + Flink)
  - Need partition evolution
  - Cloud-native data platform
"""
```

---

## 7. 엔드 투 엔드 파이프라인 예제(End-to-End Pipeline Example)

### 7.1 스트리밍 CDC에서 레이크하우스로

```python
def streaming_cdc_to_lakehouse():
    """Complete pipeline: Kafka CDC → Bronze → Silver → Gold."""
    from pyspark.sql.types import StructType, StructField, StringType

    cdc_schema = StructType([
        StructField("op", StringType()),
        StructField("before", StringType()),
        StructField("after", StringType()),
        StructField("source", StringType()),
        StructField("ts_ms", StringType()),
    ])

    # "earliest"는 Debezium 초기 스냅샷 이벤트(op="r")를 캡처하여
    # 기존 데이터베이스 상태로 Silver 테이블을 채우도록 보장한다.
    cdc_stream = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "dbserver1.public.orders") \
        .option("startingOffsets", "earliest") \
        .load()

    # Parse CDC events
    parsed = cdc_stream.select(
        from_json(col("value").cast("string"), cdc_schema).alias("cdc"),
        col("timestamp").alias("kafka_timestamp"),
    ).select("cdc.*", "kafka_timestamp")

    # Bronze는 원시 CDC 이벤트를 변경 없이 추가한다 — 모든 삽입, 업데이트,
    # 삭제의 완전한 변경 이력을 감사(auditing)와 재처리(replay)를 위해 보존한다.
    # Silver 로직이 변경되더라도 Bronze에서 재계산할 수 있다.
    bronze_query = parsed.writeStream \
        .format("delta") \
        .outputMode("append") \
        .option("checkpointLocation", "/checkpoints/bronze_orders") \
        .trigger(processingTime="30 seconds") \
        .start("/data/bronze/orders")

    # Silver는 표준 append/update 출력 모드가 MERGE 연산을 수행할 수 없기 때문에
    # foreachBatch를 사용한다. foreachBatch는 DeltaTable.merge()를 호출할 수 있는
    # 일반 DataFrame에 접근을 제공한다.
    def silver_batch(batch_df, batch_id):
        apply_cdc_to_delta(batch_df, batch_id, "/data/silver/orders")

    # Silver는 MERGE가 append보다 비용이 많이 들기 때문에 덜 자주 트리거된다
    # (1분 vs 30초). MERGE당 더 많은 이벤트를 배치 처리하면 대상 파일 읽기 및
    # 재작성 비용을 분산(amortize)할 수 있다.
    silver_query = parsed.writeStream \
        .foreachBatch(silver_batch) \
        .option("checkpointLocation", "/checkpoints/silver_orders") \
        .trigger(processingTime="1 minute") \
        .start()

    bronze_query.awaitTermination()
    silver_query.awaitTermination()
```

---

## 8. 연습 문제(Practice Problems)

### 연습 1: 메달리온 아키텍처

```python
"""
Build a complete medallion pipeline:
1. Bronze: Ingest JSON files from a landing zone into Delta Lake
   - Add metadata columns: ingestion timestamp, source file, batch ID
2. Silver: Clean and deduplicate
   - Cast data types, handle nulls
   - Deduplicate by primary key (keep latest)
   - Validate with quality checks
3. Gold: Build a daily sales summary
   - Total sales, order count, average order value per day
4. Schedule: Run Silver after Bronze, Gold after Silver
5. Verify: Time travel to compare versions
"""
```

### 연습 2: CDC를 활용한 SCD Type 2

```python
"""
Implement SCD Type 2 using CDC events:
1. Create a 'customers' dimension table with SCD2 columns
2. Consume Debezium CDC events from Kafka
3. On INSERT: add row with is_current=true
4. On UPDATE: close current row, insert new version
5. On DELETE: close current row (soft delete)
6. Query: "What was customer X's address on date Y?"
7. Bonus: add data quality checks between Bronze and Silver
"""
```

---

## 연습 문제

### 연습 1: 3계층 메달리온 파이프라인 구축

소매 데이터셋을 위한 완전한 Bronze → Silver → Gold 파이프라인을 구현하세요:

1. **Bronze**: `/data/landing/orders/`에서 JSON 파일을 읽어 세 개의 메타데이터 컬럼을 추가하여 `/data/bronze/orders/`의 Delta Lake 테이블에 append 모드로 쓰세요: `_ingested_at`(현재 타임스탬프), `_source_file`(입력 파일명), `_batch_id`(런타임 파라미터)
2. **Silver**: Bronze에서 읽어 다음 변환을 적용하세요:
   - `order_id`가 null이거나 `amount`가 음수인 행을 필터링하세요
   - `amount`를 `DECIMAL(10,2)`로, `order_time`을 `TIMESTAMP`로 형변환하세요
   - `order_id` 기준으로 중복 제거하여 `_ingested_at`이 가장 최신인 행을 유지하세요
   - `DeltaTable.merge()`를 사용하여 Silver 테이블에 업서트(upsert)하세요 — `source._ingested_at > target._ingested_at`일 때만 업데이트하세요
3. **Gold**: Silver 데이터를 `order_date`, `status`, `order_count`, `total_amount`, `avg_amount` 컬럼의 일별 요약 테이블로 집계하세요 — `replaceWhere`를 사용하여 영향받는 날짜 파티션만 덮어쓰세요
4. 멱등성(idempotency) 검증: 동일한 입력으로 전체 파이프라인을 두 번 실행하고 Gold 행 수가 두 배가 되지 않는지 확인하세요
5. Silver 테이블에 `DESCRIBE HISTORY`를 사용하여 모든 MERGE 연산과 해당 행 수를 표시하세요

### 연습 2: 충돌 해결을 포함한 증분 MERGE

순서가 어긋난 데이터를 처리하는 강력한 증분 업서트 파이프라인을 구현하세요:

1. `order_id`, `status`, `amount`, `updated_at`, `_last_seen_batch` 컬럼을 가진 대상 Delta Lake 테이블 `orders_silver`를 생성하세요
2. 다음을 수행하는 `upsert_batch(batch_df, batch_id)` 함수를 작성하세요:
   - `source.updated_at > target.updated_at`일 때만 기존 행을 업데이트합니다 (오래된 배치가 최신 데이터를 덮어쓰는 것 방지)
   - 대상에 없는 새 행을 삽입합니다
   - `source.status = 'cancelled'`이고 `target.updated_at < current_date - INTERVAL 7 DAYS`인 행을 `whenNotMatchedBySourceDelete`로 삭제합니다
3. 순서가 어긋난 전달을 시뮬레이션하세요: 배치 2 전에 배치 3을 실행하고, 그 다음 배치 2를 실행하세요 — 배치 2가 배치 3이 쓴 데이터를 덮어쓰지 않는지 검증하세요
4. `DESCRIBE HISTORY`를 사용하여 각 배치가 정확히 하나의 MERGE 커밋을 생성했는지 확인하세요
5. 주석으로 설명하세요: `whenMatchedUpdate(set={...})`에서 컬럼을 명시적으로 나열하는 것이 `whenMatchedUpdateAll()`을 사용하는 것보다 파이프라인을 더 안전하게 만드는 이유는 무엇인가요?

### 연습 3: SCD Type 2 차원 테이블

고객 주소 이력을 위한 천천히 변하는 차원 타입 2(SCD Type 2) 테이블을 구현하세요:

1. `customer_id`, `name`, `city`, `email`, `effective_from`, `effective_to`, `is_current` 컬럼을 가진 초기 Delta Lake `customers_dim` 테이블을 생성하세요
2. `is_current=True`, `effective_from='2024-01-01'`, `effective_to=None`으로 5명의 초기 고객을 로드하세요
3. 2명의 고객이 `city`를 변경하고 1명이 신규 고객인 업데이트 배치를 처리하기 위해 3.1절의 `scd_type2_merge()` 함수를 적용하세요
4. 결과를 검증하세요: 변경된 2명의 고객은 각각 2개의 행(하나는 `is_current=False`로 닫힌 행, 하나는 `is_current=True`인 현재 행)을 가져야 하고; 신규 고객은 1개의 행을 가져야 합니다
5. `effective_from`과 `effective_to` 컬럼을 사용하여 "2024-06-30에 각 고객의 도시는 무엇이었나?"라는 질문에 답하는 쿼리를 작성하세요
6. 시간 여행(`versionAsOf=0`)을 사용하여 SCD2 업데이트 전의 테이블 상태를 확인하고 현재 상태와 비교하세요

### 연습 4: 테이블 유지 관리와 쿼리 성능

시뮬레이션된 대형 테이블에서 압축(compaction)과 Z-ORDER의 영향을 측정하세요:

1. Delta Lake 테이블을 생성하고 100행씩 1,000개의 소형 배치를 쓰세요 (스트리밍 쓰기 시뮬레이션) — `DESCRIBE DETAIL`을 사용하여 생성된 파일 수를 확인하세요
2. `(order_date, customer_id)`로 필터링하는 쿼리를 실행하고 `spark.sql("EXPLAIN ...").show(truncate=False)`로 쿼리 플랜을 기록하세요 — 스캔되는 파일 수를 확인하세요
3. 테이블에 `OPTIMIZE` 후 `ZORDER BY (order_date, customer_id)`를 실행하세요
4. 동일한 필터 쿼리를 재실행하고 최적화 전후의 스캔 파일 수를 비교하세요
5. 1시간 보존 기간으로 `VACUUM`을 실행하고 (안전 검사 비활성화 필요: `spark.conf.set("spark.databricks.delta.retentionDurationCheck.enabled", "false")`) 오래된 파일이 제거되었는지 확인하세요
6. OPTIMIZE+VACUUM 전후의 `DESCRIBE DETAIL` 출력을 비교하고 `numFiles`와 `sizeInBytes`의 변화를 설명하세요

### 연습 5: Iceberg 파티션 진화와 다중 엔진 접근

Iceberg의 파티션 진화(partition evolution) 기능을 시연하고 Delta Lake와 비교하세요:

1. `months(event_ts)`로 파티셔닝된 Iceberg 테이블을 생성하고 6개월치 샘플 이벤트 데이터를 쓰세요
2. `CALL local.system.rewrite_data_files()`를 사용하여 테이블을 압축하고 `SELECT * FROM local.db.events.files`에서 전후 파일 수를 기록하세요
3. `ALTER TABLE ... ADD/DROP PARTITION FIELD`를 사용하여 파티션 구성을 `days(event_ts)`로 진화시키세요 — 기존 데이터 파일이 재작성되지 않았는지 확인하세요 (`SELECT * FROM local.db.events.files` 재확인)
4. 일별 파티셔닝을 사용하여 새 이벤트를 쓰고 쿼리가 기존 월별 파티션과 새 일별 파티션을 투명하게 읽는지 확인하세요
5. `SELECT * FROM local.db.events.snapshots`로 전체 스냅샷 이력을 조회한 후, `CALL local.system.expire_snapshots()`를 사용하여 3개월 이상된 스냅샷을 만료시키세요
6. 주석 블록으로 설명하세요: Delta Lake는 파티셔닝을 변경할 때 전체 테이블 재작성이 필요한 반면 Iceberg는 그렇지 않은 이유는 무엇인가요? Iceberg에서 이를 가능하게 하는 메타데이터 구조는 무엇인가요?

---

## 9. 요약(Summary)

### 핵심 정리

| 개념 | 설명 |
|------|------|
| **메달리온(Medallion)** | Bronze (원시) → Silver (정제) → Gold (비즈니스) |
| **MERGE** | 증분 처리를 위한 원자적 업서트(Upsert) |
| **SCD Type 2** | effective_from/to 날짜로 전체 이력 추적 |
| **압축(Compaction)** | OPTIMIZE / rewrite_data_files로 소형 파일 감소 |
| **시간 여행(Time Travel)** | 감사(Auditing) 및 디버깅을 위한 과거 버전 조회 |
| **파티션 진화(Partition Evolution)** | Iceberg는 데이터 재작성 없이 파티셔닝 변경 가능 |

### 모범 사례(Best Practices)

1. **Bronze는 추가 전용(append-only)** — 원시 데이터는 절대 수정하지 않으며, 재처리(Replay)의 원천입니다
2. **Silver에는 MERGE 사용** — 멱등적(Idempotent) 업서트로 재처리를 안전하게 처리합니다
3. **압축 일정 수립** — OPTIMIZE / VACUUM을 정기적으로 실행합니다(예: 매일 밤)
4. **현명한 파티셔닝** — 시계열 데이터는 날짜 기준으로 파티셔닝하고, 높은 카디널리티(Cardinality) 키는 지양합니다
5. **파일 수 모니터링** — 소형 파일이 너무 많으면 쿼리 성능이 저하됩니다
6. **시간 여행으로 테스트** — 버전 비교를 통해 변환 결과를 검증합니다

### 탐색

- **이전**: L18 — Debezium을 활용한 CDC
- **다음**: [L20 — Dagster 자산 기반 오케스트레이션](20_Dagster_Asset_Orchestration.md)
- **L11** (Delta Lake & Iceberg)로 돌아가서 기초 API 지식을 복습하세요
