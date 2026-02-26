# Lakehouse Practical Patterns

## Learning Objectives

After completing this lesson, you will be able to:

1. Design a medallion architecture (bronze, silver, gold) that enforces appropriate schema constraints, deduplication, and data quality rules at each layer.
2. Implement incremental MERGE (upsert) operations using Delta Lake's DeltaTable API or Apache Iceberg's MergeIntoTable to efficiently update large tables.
3. Apply the SCD Type 2 pattern to track historical changes in dimension tables, capturing effective dates and current record flags.
4. Use Delta Lake and Iceberg time travel features to query historical snapshots, audit changes, and roll back accidental writes.
5. Configure table compaction (OPTIMIZE / rewrite_data_files), Z-ORDER clustering, and partition pruning to maximize query performance on large Lakehouse tables.
6. Compare Delta Lake and Apache Iceberg in terms of ACID guarantees, schema evolution capabilities, and multi-engine interoperability (Spark, Trino, Flink).

---

## Overview

The Lakehouse architecture combines the reliability of data warehouses with the scalability of data lakes. This lesson covers production patterns for Delta Lake and Apache Iceberg: the medallion architecture, incremental processing with MERGE, Slowly Changing Dimensions (SCD Type 2), compaction, time travel, and multi-engine interoperability.

---

## 1. Medallion Architecture

### 1.1 Three-Layer Design

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

### 1.2 Implementing Medallion with Delta Lake

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, current_timestamp, input_file_name, lit,
    from_json, to_timestamp, when, count, sum as spark_sum,
    window, row_number,
)
from pyspark.sql.window import Window
from delta.tables import DeltaTable

# Delta Lake requires these two Spark configs to register its custom SQL
# commands (OPTIMIZE, VACUUM, MERGE) and to make spark.sql() recognize
# Delta tables. Without them, Delta operations fail with ClassNotFoundException.
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

    # Metadata columns (_ingested_at, _source_file, _batch_id) enable
    # debugging and replay: you can trace any row back to when it arrived,
    # which file it came from, and which batch produced it.
    bronze = raw \
        .withColumn("_ingested_at", current_timestamp()) \
        .withColumn("_source_file", input_file_name()) \
        .withColumn("_batch_id", lit("batch_001"))

    # mergeSchema=true allows the Bronze layer to accept schema drift from
    # upstream sources (e.g., a new column added by the producer). This is
    # intentional at Bronze — schema enforcement happens at Silver.
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
    # decimal(10,2) is used instead of float to avoid floating-point rounding
    # errors in financial calculations (e.g., 99.99 stored as 99.98999...).
    cleaned = bronze \
        .filter(col("order_id").isNotNull()) \
        .withColumn("amount", col("amount").cast("decimal(10,2)")) \
        .withColumn("order_time", to_timestamp(col("timestamp")))

    # 2. Deduplicate: keep latest record per order_id
    # Bronze is append-only, so the same order_id may appear multiple times
    # (e.g., re-ingestion, late corrections). row_number() ordered by
    # _ingested_at DESC ensures we always pick the most recent version.
    w = Window.partitionBy("order_id").orderBy(col("_ingested_at").desc())
    deduplicated = cleaned \
        .withColumn("_rn", row_number().over(w)) \
        .filter(col("_rn") == 1) \
        .drop("_rn")

    # 3. Quality checks — rows failing these conditions are silently dropped.
    # In production, write rejected rows to a quarantine table for investigation.
    valid = deduplicated.filter(
        (col("amount") > 0) &
        (col("order_time").isNotNull()) &
        (col("status").isin("pending", "shipped", "delivered", "cancelled"))
    )

    # 4. MERGE makes Silver idempotent: re-running the same batch produces
    # the same result because existing rows are updated rather than duplicated.
    # The _ingested_at condition prevents overwriting a newer version with
    # an older re-processed batch.
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

    # replaceWhere selectively overwrites only the date partitions present
    # in this batch, leaving older dates untouched. This is more efficient
    # and safer than a full mode="overwrite" which would delete ALL gold data.
    # It also makes the operation idempotent: re-running the same dates
    # produces identical results without duplicating rows.
    daily.write \
        .format("delta") \
        .mode("overwrite") \
        .option("replaceWhere",
                f"order_date >= '{daily.agg({'order_date': 'min'}).first()[0]}'") \
        .save(gold_path)

    print(f"Gold daily summary: {daily.count()} rows")
```

---

## 2. Incremental Processing with MERGE

### 2.1 Delta Lake MERGE (Upsert)

```python
def upsert_orders(new_data_path, target_path):
    """Incremental upsert: INSERT new rows, UPDATE existing ones."""
    new_data = spark.read.json(new_data_path)

    target = DeltaTable.forPath(spark, target_path)

    target.alias("t").merge(
        new_data.alias("s"),
        "t.order_id = s.order_id"
    ).whenMatchedUpdate(
        # The updated_at guard prevents an older batch from overwriting
        # newer data during out-of-order reprocessing or parallel writes.
        condition="s.updated_at > t.updated_at",
        # Explicitly listing columns (instead of UpdateAll) documents
        # exactly which fields the MERGE modifies — safer than UpdateAll
        # which silently picks up new source columns after schema changes.
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
        # This clause removes target rows that no longer exist in the source
        # batch, but only if they are older than 30 days. The age guard
        # prevents accidental deletion of recent rows that simply weren't
        # included in this particular incremental batch.
        condition="t.updated_at < current_date() - INTERVAL 30 DAYS"
    ).execute()
```

### 2.2 CDC Events to Delta Lake

```python
def apply_cdc_to_delta(cdc_batch_df, batch_id, target_path):
    """Apply Debezium CDC events to a Delta table via foreachBatch.

    Handles INSERT (op=c/r), UPDATE (op=u), and DELETE (op=d).
    """
    if cdc_batch_df.isEmpty():
        return

    target = DeltaTable.forPath(spark, target_path)

    # Deletes and upserts must be processed separately because MERGE cannot
    # delete rows and upsert rows in the same operation when the source
    # contains both matching and non-matching keys for different operations.
    deletes = cdc_batch_df.filter(col("op") == "d")
    upserts = cdc_batch_df.filter(col("op").isin("c", "u", "r"))

    # Apply upserts (INSERT or UPDATE)
    if upserts.count() > 0:
        # For inserts/updates, the row's current state is in the 'after' field.
        # select("after.*") flattens the nested struct into top-level columns
        # matching the target table's schema.
        upsert_data = upserts.select("after.*")
        target.alias("t").merge(
            upsert_data.alias("s"),
            "t.id = s.id"
        ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()

    # Apply deletes
    if deletes.count() > 0:
        # For deletes, the row's last known state is in the 'before' field
        # (the 'after' field is null). We only need the primary key to identify
        # which rows to remove from the target table.
        delete_keys = deletes.select("before.id").collect()
        delete_ids = [row["id"] for row in delete_keys]
        target.delete(col("id").isin(delete_ids))

    print(f"CDC batch {batch_id}: "
          f"{upserts.count()} upserts, {deletes.count()} deletes")
```

---

## 3. Slowly Changing Dimensions (SCD Type 2)

### 3.1 SCD Type 2 with Delta Lake

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

    # Dynamically build the change-detection condition from tracked_cols.
    # Only changes to these specific columns trigger a new SCD2 version —
    # columns not in tracked_cols (e.g., last_login) can change without
    # creating historical records, reducing version explosion.
    change_condition = " OR ".join(
        f"target.{c} != source.{c}" for c in tracked_cols
    )

    # Only compare against current records (is_current=True) — historical
    # versions are closed and should never be matched again.
    current = spark.read.format("delta").load(target_path) \
        .filter(col("is_current") == True)

    # LEFT join ensures new records (no match in target) are included.
    # The filter captures two cases: entirely new entities AND existing
    # entities whose tracked columns have changed.
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

    # SCD2 requires two operations: close then insert. The order matters —
    # closing first ensures there is never a window where two records for
    # the same entity are both marked is_current=True.
    if rows_to_close.count() > 0:
        close_keys = [row[key_col] for row in rows_to_close.collect()]
        target.update(
            condition=(col(key_col).isin(close_keys)) & (col("is_current") == True),
            set={
                "effective_to": expr(f"date_sub(current_date(), 1)"),
                "is_current": lit(False),
            },
        )

    # Append (not MERGE) for the new versions because each new version
    # is a brand new row with a different effective_from — there is no
    # existing row to match against.
    rows_to_insert.write \
        .format("delta") \
        .mode("append") \
        .save(target_path)

    print(f"SCD2: {rows_to_close.count()} closed, {rows_to_insert.count()} inserted")
```

---

## 4. Table Maintenance

### 4.1 Compaction and Optimization

```python
"""
Delta Lake creates many small files (especially with streaming).
Compaction combines small files into larger ones for better read performance.
"""


def optimize_delta_table(table_path):
    """Run compaction and Z-ordering on a Delta table."""
    # OPTIMIZE merges small files into ~1GB files. Streaming writes create
    # many tiny files (one per micro-batch), and each file adds overhead to
    # query planning. Compaction can improve read performance by 10-100x.
    spark.sql(f"OPTIMIZE delta.`{table_path}`")

    # Z-ORDER physically co-locates rows with similar (order_date, customer_id)
    # values in the same files. Queries filtering on these columns skip entire
    # files via min/max statistics, dramatically reducing I/O. Choose Z-ORDER
    # columns based on your most common WHERE clauses.
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
    # VACUUM is irreversible — once old files are deleted, you cannot
    # time-travel to those versions. Setting retention < 168 hours requires
    # disabling the safety check: spark.databricks.delta.retentionDurationCheck.enabled
    spark.sql(f"VACUUM delta.`{table_path}` RETAIN {retention_hours} HOURS")


def auto_compact_config():
    """Configure auto-compaction for streaming writes."""
    # optimizeWrite repartitions data within each write to produce
    # fewer, larger files — reduces the small-file problem at the source.
    spark.conf.set("spark.databricks.delta.optimizeWrite.enabled", "true")
    # autoCompact triggers a lightweight compaction after each write
    # if too many small files accumulate. This eliminates the need for
    # a separate OPTIMIZE job, but adds ~5-10% write latency.
    spark.conf.set("spark.databricks.delta.autoCompact.enabled", "true")
    # 128MB target balances read performance (fewer files to open) against
    # memory usage during compaction. Increase for large tables, decrease
    # for small or frequently-queried tables.
    spark.conf.set("spark.databricks.delta.autoCompact.minFileSize", "134217728")
```

### 4.2 Time Travel

```python
def time_travel_queries(table_path):
    """Query historical versions of a Delta table."""
    # versionAsOf reads an exact version number. Version 0 is the table's
    # initial creation state — useful for comparing current data against
    # the original to detect drift or validate transformations.
    v0 = spark.read.format("delta") \
        .option("versionAsOf", 0) \
        .load(table_path)

    # timestampAsOf is more intuitive when you know "when" but not "which version."
    # Delta finds the latest version at or before the given timestamp.
    # Useful for audit queries: "show me the data as the CFO saw it yesterday."
    historical = spark.read.format("delta") \
        .option("timestampAsOf", "2024-06-15T10:00:00") \
        .load(table_path)

    # DESCRIBE HISTORY shows all operations (INSERT, MERGE, OPTIMIZE, etc.)
    # with row counts and execution times — essential for debugging when
    # a table's row count unexpectedly changed.
    history = spark.sql(f"DESCRIBE HISTORY delta.`{table_path}`")
    history.select("version", "timestamp", "operation", "operationMetrics").show()

    # RESTORE physically rewrites the table to match version 5. This creates
    # a NEW version (not a deletion of later versions), preserving the full
    # audit trail. After RESTORE, VACUUM will eventually reclaim unreferenced files.
    spark.sql(f"RESTORE TABLE delta.`{table_path}` TO VERSION AS OF 5")

    return v0, historical
```

---

## 5. Apache Iceberg Patterns

### 5.1 Iceberg Table Operations

```python
"""
Apache Iceberg provides similar capabilities with a catalog-centric approach.
Iceberg works with Spark, Trino, Flink, and other engines simultaneously.
"""

# The "local" catalog name is arbitrary — in production you might use
# "iceberg_prod" pointing to a REST catalog or AWS Glue. The hadoop type
# stores metadata as files (simple, no server needed) but lacks concurrent
# write safety. Use a REST or Hive catalog for multi-engine production use.
spark_iceberg = SparkSession.builder \
    .appName("IcebergPatterns") \
    .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.local.type", "hadoop") \
    .config("spark.sql.catalog.local.warehouse", "/data/iceberg/warehouse") \
    .getOrCreate()


def iceberg_crud():
    """Basic Iceberg table operations."""
    # Iceberg uses "hidden partitioning" — months(order_date) creates monthly
    # partitions without requiring the user to add a redundant partition column.
    # Queries filtering on order_date automatically benefit from partition pruning,
    # even without explicitly referencing the partition transform.
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
    # Start with monthly partitioning — appropriate when data volume is low.
    spark_iceberg.sql("""
        CREATE TABLE local.db.events (
            event_id  BIGINT,
            event_ts  TIMESTAMP,
            payload   STRING
        )
        USING iceberg
        PARTITIONED BY (months(event_ts))
    """)

    # Partition evolution is Iceberg's killer feature over Delta Lake.
    # This changes future writes to daily granularity WITHOUT rewriting
    # any existing data files. In Delta Lake, changing partitioning
    # requires a full table rewrite (potentially hours for large tables).
    spark_iceberg.sql("""
        ALTER TABLE local.db.events
        ADD PARTITION FIELD days(event_ts)
    """)
    spark_iceberg.sql("""
        ALTER TABLE local.db.events
        DROP PARTITION FIELD months(event_ts)
    """)
    # Old data stays in monthly partitions, new data goes to daily partitions.
    # Iceberg's metadata layer tracks which partition spec applies to which
    # data files, so queries transparently read both layouts. This is possible
    # because Iceberg stores partition info per-file in manifest files, unlike
    # Hive-style partitioning which encodes it in the directory path.
```

### 5.2 Iceberg Table Maintenance

```python
def iceberg_maintenance():
    """Iceberg table maintenance operations."""
    # Expiring snapshots frees metadata storage and reduces query planning
    # time. retain_last=10 keeps enough snapshots for time travel and
    # debugging, while older_than prevents deleting very recent snapshots
    # that concurrent queries might still be reading.
    spark_iceberg.sql("""
        CALL local.system.expire_snapshots(
            table => 'db.orders',
            older_than => TIMESTAMP '2024-06-10 00:00:00',
            retain_last => 10
        )
    """)

    # Compaction merges small files into ~128MB files. Unlike Delta's
    # OPTIMIZE, Iceberg's rewrite_data_files works within the Iceberg
    # transaction model — the compacted files become a new snapshot,
    # so concurrent readers continue seeing consistent data.
    spark_iceberg.sql("""
        CALL local.system.rewrite_data_files(
            table => 'db.orders',
            options => map('target-file-size-bytes', '134217728')
        )
    """)

    # Manifest files index which data files belong to each snapshot.
    # Over time, many small manifests accumulate. Rewriting them into
    # fewer, larger manifests speeds up query planning (fewer files to read
    # when building the scan plan).
    spark_iceberg.sql("""
        CALL local.system.rewrite_manifests('db.orders')
    """)

    # Orphan files are data files not referenced by any snapshot — typically
    # left behind by failed writes or expired snapshots. Removing them
    # reclaims storage. The older_than guard prevents deleting files from
    # in-progress writes.
    spark_iceberg.sql("""
        CALL local.system.remove_orphan_files(
            table => 'db.orders',
            older_than => TIMESTAMP '2024-06-01 00:00:00'
        )
    """)
```

---

## 6. Delta Lake vs Iceberg Comparison

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

## 7. End-to-End Pipeline Example

### 7.1 Streaming CDC to Lakehouse

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

    # "earliest" ensures we capture the Debezium initial snapshot events
    # (op="r") which populate the Silver table with the existing database state.
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

    # Bronze appends raw CDC events unchanged — this preserves the complete
    # change history (every insert, update, delete) for auditing and replay.
    # Even if Silver logic changes, you can recompute from Bronze.
    bronze_query = parsed.writeStream \
        .format("delta") \
        .outputMode("append") \
        .option("checkpointLocation", "/checkpoints/bronze_orders") \
        .trigger(processingTime="30 seconds") \
        .start("/data/bronze/orders")

    # Silver uses foreachBatch because standard append/update output modes
    # cannot perform MERGE operations. foreachBatch gives access to a regular
    # DataFrame where DeltaTable.merge() can be called.
    def silver_batch(batch_df, batch_id):
        apply_cdc_to_delta(batch_df, batch_id, "/data/silver/orders")

    # Silver triggers less frequently (1 min vs 30s) because MERGE is
    # more expensive than append — batching more events per MERGE amortizes
    # the cost of reading and rewriting target files.
    silver_query = parsed.writeStream \
        .foreachBatch(silver_batch) \
        .option("checkpointLocation", "/checkpoints/silver_orders") \
        .trigger(processingTime="1 minute") \
        .start()

    bronze_query.awaitTermination()
    silver_query.awaitTermination()
```

---

## 8. Practice Problems

### Exercise 1: Medallion Architecture

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

### Exercise 2: SCD Type 2 with CDC

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

## Exercises

### Exercise 1: Build a Three-Layer Medallion Pipeline

Implement a complete Bronze → Silver → Gold pipeline for a retail dataset:

1. **Bronze**: Read JSON files from `/data/landing/orders/` and write to a Delta Lake table at `/data/bronze/orders/` in append mode, adding three metadata columns: `_ingested_at` (current timestamp), `_source_file` (input file name), and `_batch_id` (a runtime parameter)
2. **Silver**: Read from Bronze and apply the following transformations:
   - Filter out rows where `order_id` is null or `amount` is negative
   - Cast `amount` to `DECIMAL(10,2)` and `order_time` to `TIMESTAMP`
   - Deduplicate by `order_id`, keeping the row with the latest `_ingested_at`
   - Use `DeltaTable.merge()` to upsert into the Silver table — update only when `source._ingested_at > target._ingested_at`
3. **Gold**: Aggregate Silver data into a daily summary table with columns: `order_date`, `status`, `order_count`, `total_amount`, `avg_amount` — use `replaceWhere` to overwrite only the affected date partitions
4. Verify idempotency: run the full pipeline twice with the same input and confirm the Gold row counts do not double
5. Use `DESCRIBE HISTORY` on the Silver table to show all MERGE operations and their row counts

### Exercise 2: Incremental MERGE with Conflict Resolution

Implement a robust incremental upsert pipeline that handles out-of-order data:

1. Create a target Delta Lake table `orders_silver` with columns: `order_id`, `status`, `amount`, `updated_at`, `_last_seen_batch`
2. Write an `upsert_batch(batch_df, batch_id)` function that:
   - Updates existing rows only when `source.updated_at > target.updated_at` (prevents older batches from overwriting newer data)
   - Inserts new rows that do not exist in the target
   - Deletes rows from the target where `source.status = 'cancelled'` AND `target.updated_at < current_date - INTERVAL 7 DAYS` using `whenNotMatchedBySourceDelete`
3. Simulate out-of-order delivery: run batch 3 before batch 2, then batch 2 — verify that batch 2 does NOT overwrite the data written by batch 3
4. Use `DESCRIBE HISTORY` to confirm that each batch created exactly one MERGE commit
5. Explain in comments: why does explicitly listing columns in `whenMatchedUpdate(set={...})` make the pipeline safer than using `whenMatchedUpdateAll()`?

### Exercise 3: SCD Type 2 Dimension Table

Implement a Slowly Changing Dimension Type 2 table for customer address history:

1. Create an initial Delta Lake `customers_dim` table with columns: `customer_id`, `name`, `city`, `email`, `effective_from`, `effective_to`, `is_current`
2. Load 5 initial customers with `is_current=True`, `effective_from='2024-01-01'`, `effective_to=None`
3. Apply the `scd_type2_merge()` function from Section 3.1 to process a batch of updates where 2 customers changed their `city` and 1 is a new customer
4. Verify the result: the 2 changed customers should each have 2 rows (one closed with `is_current=False`, one current with `is_current=True`); the new customer should have 1 row
5. Write a query that answers: "What was each customer's city on 2024-06-30?" using the `effective_from` and `effective_to` columns
6. Use time travel (`versionAsOf=0`) to see the table state before the SCD2 update and compare it with the current state

### Exercise 4: Table Maintenance and Query Performance

Measure the impact of compaction and Z-ordering on a simulated large table:

1. Create a Delta Lake table and write 1,000 small batches of 100 rows each (simulating streaming writes) — note the number of files created using `DESCRIBE DETAIL`
2. Run a query that filters by `(order_date, customer_id)` and record the query plan using `spark.sql("EXPLAIN ...").show(truncate=False)` — note how many files are scanned
3. Run `OPTIMIZE` followed by `ZORDER BY (order_date, customer_id)` on the table
4. Re-run the same filter query and compare the number of files scanned before and after optimization
5. Run `VACUUM` with a 1-hour retention (you will need to disable the safety check: `spark.conf.set("spark.databricks.delta.retentionDurationCheck.enabled", "false")`) and verify the old files are removed
6. Compare the output of `DESCRIBE DETAIL` before and after OPTIMIZE+VACUUM and explain the changes in `numFiles` and `sizeInBytes`

### Exercise 5: Iceberg Partition Evolution and Multi-Engine Access

Demonstrate Iceberg's partition evolution capability and compare it with Delta Lake:

1. Create an Iceberg table partitioned by `months(event_ts)` and write 6 months of sample event data
2. Use `CALL local.system.rewrite_data_files()` to compact the table and record the before/after file counts from `SELECT * FROM local.db.events.files`
3. Evolve the partition scheme to `days(event_ts)` using `ALTER TABLE ... ADD/DROP PARTITION FIELD` — verify that existing data files are NOT rewritten (check `SELECT * FROM local.db.events.files` again)
4. Write new events using the daily partitioning and verify that queries transparently read both the old monthly and new daily partitions
5. Use `SELECT * FROM local.db.events.snapshots` to view the full snapshot history, then expire snapshots older than 3 months using `CALL local.system.expire_snapshots()`
6. Explain in a comment block: why does Delta Lake require a full table rewrite when changing partitioning, while Iceberg does not? What metadata structure makes this possible in Iceberg?

---

## 9. Summary

### Key Takeaways

| Concept | Description |
|---------|-------------|
| **Medallion** | Bronze (raw) → Silver (clean) → Gold (business) |
| **MERGE** | Atomic upsert for incremental processing |
| **SCD Type 2** | Track full history with effective_from/to dates |
| **Compaction** | OPTIMIZE / rewrite_data_files to reduce small files |
| **Time Travel** | Query historical versions for auditing and debugging |
| **Partition Evolution** | Iceberg can change partitioning without data rewrite |

### Best Practices

1. **Bronze is append-only** — never modify raw data; it's your replay source
2. **Use MERGE for Silver** — idempotent upserts handle reprocessing safely
3. **Schedule compaction** — run OPTIMIZE / VACUUM regularly (e.g., nightly)
4. **Partition wisely** — by date for time-series, avoid high-cardinality keys
5. **Monitor file counts** — too many small files degrade query performance
6. **Test with time travel** — verify transformations by comparing versions

### Navigation

- **Previous**: L18 — CDC with Debezium
- **Next**: [L20 — Dagster Asset Orchestration](20_Dagster_Asset_Orchestration.md)
- Return to **L11** (Delta Lake & Iceberg) for foundational API knowledge
