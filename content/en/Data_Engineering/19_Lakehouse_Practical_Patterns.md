# Lakehouse Practical Patterns

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

    bronze = raw \
        .withColumn("_ingested_at", current_timestamp()) \
        .withColumn("_source_file", input_file_name()) \
        .withColumn("_batch_id", lit("batch_001"))

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
    cleaned = bronze \
        .filter(col("order_id").isNotNull()) \
        .withColumn("amount", col("amount").cast("decimal(10,2)")) \
        .withColumn("order_time", to_timestamp(col("timestamp")))

    # 2. Deduplicate: keep latest record per order_id
    w = Window.partitionBy("order_id").orderBy(col("_ingested_at").desc())
    deduplicated = cleaned \
        .withColumn("_rn", row_number().over(w)) \
        .filter(col("_rn") == 1) \
        .drop("_rn")

    # 3. Quality checks
    valid = deduplicated.filter(
        (col("amount") > 0) &
        (col("order_time").isNotNull()) &
        (col("status").isin("pending", "shipped", "delivered", "cancelled"))
    )

    # 4. Write with MERGE for idempotency
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

    # Overwrite partition for idempotency
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
        condition="s.updated_at > t.updated_at",
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

    # Separate deletes from upserts
    deletes = cdc_batch_df.filter(col("op") == "d")
    upserts = cdc_batch_df.filter(col("op").isin("c", "u", "r"))

    # Apply upserts (INSERT or UPDATE)
    if upserts.count() > 0:
        # Extract 'after' fields for upserts
        upsert_data = upserts.select("after.*")
        target.alias("t").merge(
            upsert_data.alias("s"),
            "t.id = s.id"
        ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()

    # Apply deletes
    if deletes.count() > 0:
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

    # Build change detection condition
    change_condition = " OR ".join(
        f"target.{c} != source.{c}" for c in tracked_cols
    )

    # Join updates with current records to find changes
    current = spark.read.format("delta").load(target_path) \
        .filter(col("is_current") == True)

    changes = updates_df.alias("source").join(
        current.alias("target"),
        col(f"source.{key_col}") == col(f"target.{key_col}"),
        "left",
    ).filter(
        # New records (no match) or changed records
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

    # Step 1: Close existing current records
    if rows_to_close.count() > 0:
        close_keys = [row[key_col] for row in rows_to_close.collect()]
        target.update(
            condition=(col(key_col).isin(close_keys)) & (col("is_current") == True),
            set={
                "effective_to": expr(f"date_sub(current_date(), 1)"),
                "is_current": lit(False),
            },
        )

    # Step 2: Insert new versions
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
    # OPTIMIZE: compact small files (target 1GB per file)
    spark.sql(f"OPTIMIZE delta.`{table_path}`")

    # OPTIMIZE with Z-ORDER: co-locate related data for faster queries
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
    spark.sql(f"VACUUM delta.`{table_path}` RETAIN {retention_hours} HOURS")


def auto_compact_config():
    """Configure auto-compaction for streaming writes."""
    # Auto-optimize on write
    spark.conf.set("spark.databricks.delta.optimizeWrite.enabled", "true")
    # Auto-compact small files after write
    spark.conf.set("spark.databricks.delta.autoCompact.enabled", "true")
    # Target file size (128MB default)
    spark.conf.set("spark.databricks.delta.autoCompact.minFileSize", "134217728")
```

### 4.2 Time Travel

```python
def time_travel_queries(table_path):
    """Query historical versions of a Delta table."""
    # Read a specific version
    v0 = spark.read.format("delta") \
        .option("versionAsOf", 0) \
        .load(table_path)

    # Read as of a timestamp
    historical = spark.read.format("delta") \
        .option("timestampAsOf", "2024-06-15T10:00:00") \
        .load(table_path)

    # View table history
    history = spark.sql(f"DESCRIBE HISTORY delta.`{table_path}`")
    history.select("version", "timestamp", "operation", "operationMetrics").show()

    # Restore to a previous version
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

# Configure Iceberg catalog
spark_iceberg = SparkSession.builder \
    .appName("IcebergPatterns") \
    .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.local.type", "hadoop") \
    .config("spark.sql.catalog.local.warehouse", "/data/iceberg/warehouse") \
    .getOrCreate()


def iceberg_crud():
    """Basic Iceberg table operations."""
    # Create table
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
    # Start with monthly partitioning
    spark_iceberg.sql("""
        CREATE TABLE local.db.events (
            event_id  BIGINT,
            event_ts  TIMESTAMP,
            payload   STRING
        )
        USING iceberg
        PARTITIONED BY (months(event_ts))
    """)

    # Later, change to daily partitioning (no data rewrite!)
    spark_iceberg.sql("""
        ALTER TABLE local.db.events
        ADD PARTITION FIELD days(event_ts)
    """)
    spark_iceberg.sql("""
        ALTER TABLE local.db.events
        DROP PARTITION FIELD months(event_ts)
    """)
    # Old data stays in monthly partitions
    # New data goes to daily partitions
    # Queries transparently read both
```

### 5.2 Iceberg Table Maintenance

```python
def iceberg_maintenance():
    """Iceberg table maintenance operations."""
    # Expire old snapshots (keep last 5 days)
    spark_iceberg.sql("""
        CALL local.system.expire_snapshots(
            table => 'db.orders',
            older_than => TIMESTAMP '2024-06-10 00:00:00',
            retain_last => 10
        )
    """)

    # Rewrite small data files (compaction)
    spark_iceberg.sql("""
        CALL local.system.rewrite_data_files(
            table => 'db.orders',
            options => map('target-file-size-bytes', '134217728')
        )
    """)

    # Rewrite manifests for faster planning
    spark_iceberg.sql("""
        CALL local.system.rewrite_manifests('db.orders')
    """)

    # Remove orphan files
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

    # Read CDC events from Kafka
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

    # Bronze: append raw CDC events
    bronze_query = parsed.writeStream \
        .format("delta") \
        .outputMode("append") \
        .option("checkpointLocation", "/checkpoints/bronze_orders") \
        .trigger(processingTime="30 seconds") \
        .start("/data/bronze/orders")

    # Silver: foreachBatch to apply CDC with MERGE
    def silver_batch(batch_df, batch_id):
        apply_cdc_to_delta(batch_df, batch_id, "/data/silver/orders")

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
- Return to **L11** (Delta Lake & Iceberg) for foundational API knowledge
