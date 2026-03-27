"""
Delta Lake / Lakehouse Practical Patterns
==========================================
Demonstrates:
- Medallion architecture (Bronze → Silver → Gold)
- Incremental MERGE (upsert)
- SCD Type 2 with Delta Lake
- Table maintenance (OPTIMIZE, VACUUM, Z-ORDER)
- Time travel queries
- CDC events to Delta Lake

Requirements:
    pip install pyspark delta-spark
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, current_timestamp, input_file_name, lit,
    to_timestamp, when, count, sum as spark_sum, avg,
    row_number, expr, date_sub, current_date,
)
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
    IntegerType, TimestampType, DateType, BooleanType,
)
from pyspark.sql.window import Window


def get_spark():
    """Create SparkSession with Delta Lake support."""
    return SparkSession.builder \
        .appName("LakehousePatterns") \
        .master("local[*]") \
        .config("spark.sql.extensions",
                "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.sql.shuffle.partitions", "4") \
        .getOrCreate()


# ── 1. Medallion Architecture ──────────────────────────────────────

def demo_medallion(spark):
    """Demonstrate Bronze → Silver → Gold pipeline."""
    import tempfile
    import os

    base = tempfile.mkdtemp(prefix="lakehouse_")
    bronze_path = os.path.join(base, "bronze", "orders")
    silver_path = os.path.join(base, "silver", "orders")
    gold_path = os.path.join(base, "gold", "daily_summary")

    # Create sample data
    data = [
        (1, "C001", "shipped", 99.99, "2024-06-15 10:00:00"),
        (2, "C002", "pending", 149.50, "2024-06-15 11:00:00"),
        (3, "C001", "delivered", 75.00, "2024-06-16 09:00:00"),
        (4, "C003", "shipped", 200.00, "2024-06-16 14:00:00"),
        (5, "C002", "cancelled", -10.00, "2024-06-16 15:00:00"),  # Bad: negative
        (None, "C004", "pending", 50.00, "2024-06-17 08:00:00"),  # Bad: null ID
        (6, "C001", "pending", 120.00, "2024-06-17 12:00:00"),
    ]
    schema = ["order_id", "customer_id", "status", "amount", "timestamp"]
    raw_df = spark.createDataFrame(data, schema)

    # ── Bronze: Raw ingestion ──
    bronze = raw_df \
        .withColumn("_ingested_at", current_timestamp()) \
        .withColumn("_source", lit("sample_data")) \
        .withColumn("_batch_id", lit("batch_001"))

    bronze.write.format("delta").mode("overwrite").save(bronze_path)
    print(f"Bronze: {bronze.count()} rows written")

    # ── Silver: Clean + deduplicate ──
    bronze_read = spark.read.format("delta").load(bronze_path)

    # Clean: filter nulls, cast types, validate
    cleaned = bronze_read \
        .filter(col("order_id").isNotNull()) \
        .withColumn("amount", col("amount").cast("decimal(10,2)")) \
        .withColumn("order_time", to_timestamp(col("timestamp"))) \
        .filter(col("amount") > 0) \
        .filter(col("status").isin("pending", "shipped", "delivered", "cancelled"))

    # Deduplicate: keep latest per order_id
    w = Window.partitionBy("order_id").orderBy(col("_ingested_at").desc())
    silver = cleaned \
        .withColumn("_rn", row_number().over(w)) \
        .filter(col("_rn") == 1) \
        .drop("_rn", "timestamp", "_source", "_batch_id")

    silver.write.format("delta").mode("overwrite").save(silver_path)
    print(f"Silver: {silver.count()} valid rows (removed {bronze.count() - silver.count()} bad rows)")

    # ── Gold: Business aggregation ──
    silver_read = spark.read.format("delta").load(silver_path)

    gold = silver_read \
        .withColumn("order_date", col("order_time").cast("date")) \
        .groupBy("order_date") \
        .agg(
            count("*").alias("order_count"),
            spark_sum("amount").alias("total_revenue"),
            avg("amount").alias("avg_order_value"),
        ) \
        .orderBy("order_date")

    gold.write.format("delta").mode("overwrite").save(gold_path)
    print(f"Gold: {gold.count()} daily summaries")
    gold.show()

    return base


# ── 2. Incremental MERGE (Upsert) ─────────────────────────────────

def demo_merge(spark):
    """Demonstrate Delta Lake MERGE for incremental upserts."""
    import tempfile
    import os
    from delta.tables import DeltaTable

    base = tempfile.mkdtemp(prefix="merge_")
    target_path = os.path.join(base, "orders")

    # Initial data
    initial = spark.createDataFrame([
        (1, "C001", "pending", 99.99, "2024-06-15"),
        (2, "C002", "pending", 149.50, "2024-06-15"),
        (3, "C003", "shipped", 75.00, "2024-06-16"),
    ], ["order_id", "customer_id", "status", "amount", "updated_at"])
    initial.write.format("delta").save(target_path)
    print("Initial table:")
    spark.read.format("delta").load(target_path).show()

    # New data: order 1 updated, order 4 is new
    updates = spark.createDataFrame([
        (1, "C001", "shipped", 99.99, "2024-06-16"),   # UPDATE
        (4, "C004", "pending", 200.00, "2024-06-16"),   # INSERT
    ], ["order_id", "customer_id", "status", "amount", "updated_at"])

    # MERGE
    target = DeltaTable.forPath(spark, target_path)
    target.alias("t").merge(
        updates.alias("s"),
        "t.order_id = s.order_id"
    ).whenMatchedUpdate(
        condition="s.updated_at > t.updated_at",
        set={
            "status": "s.status",
            "amount": "s.amount",
            "updated_at": "s.updated_at",
        }
    ).whenNotMatchedInsertAll().execute()

    print("After MERGE:")
    spark.read.format("delta").load(target_path).orderBy("order_id").show()

    # Check history
    print("Table history:")
    history = spark.sql(f"DESCRIBE HISTORY delta.`{target_path}`")
    history.select("version", "timestamp", "operation",
                   "operationMetrics").show(truncate=False)

    return base


# ── 3. SCD Type 2 ──────────────────────────────────────────────────

def demo_scd_type2(spark):
    """Demonstrate Slowly Changing Dimension Type 2."""
    import tempfile
    import os
    from delta.tables import DeltaTable

    base = tempfile.mkdtemp(prefix="scd2_")
    dim_path = os.path.join(base, "dim_customers")

    # Initial dimension data
    initial = spark.createDataFrame([
        (1, "Alice", "NYC", "alice@email.com", "2024-01-01", None, True),
        (2, "Bob", "LA", "bob@email.com", "2024-01-01", None, True),
        (3, "Carol", "Chicago", "carol@email.com", "2024-01-01", None, True),
    ], ["customer_id", "name", "city", "email",
        "effective_from", "effective_to", "is_current"])

    initial = initial \
        .withColumn("effective_from", col("effective_from").cast("date")) \
        .withColumn("effective_to", col("effective_to").cast("date"))

    initial.write.format("delta").save(dim_path)
    print("Initial dimension:")
    spark.read.format("delta").load(dim_path).show()

    # Updates: Alice moved to LA, Dave is new
    updates = spark.createDataFrame([
        (1, "Alice", "LA", "alice@email.com", "2024-06-15"),
        (4, "Dave", "Houston", "dave@email.com", "2024-06-15"),
    ], ["customer_id", "name", "city", "email", "change_date"])
    updates = updates.withColumn("change_date", col("change_date").cast("date"))

    # Apply SCD Type 2
    target = DeltaTable.forPath(spark, dim_path)
    current = spark.read.format("delta").load(dim_path) \
        .filter(col("is_current") == True)

    # Find changes
    changes = updates.alias("s").join(
        current.alias("t"),
        col("s.customer_id") == col("t.customer_id"),
        "left",
    ).filter(
        col("t.customer_id").isNull() |  # New customer
        (col("s.city") != col("t.city")) |  # City changed
        (col("s.email") != col("t.email"))  # Email changed
    )

    # Close existing records
    changed_ids = [row.customer_id for row in
                   changes.filter(col("t.customer_id").isNotNull())
                   .select("s.customer_id").collect()]

    if changed_ids:
        target.update(
            condition=(col("customer_id").isin(changed_ids)) &
                      (col("is_current") == True),
            set={
                "effective_to": lit("2024-06-14").cast("date"),
                "is_current": lit(False),
            },
        )

    # Insert new versions
    new_rows = changes.select(
        col("s.customer_id"),
        col("s.name"),
        col("s.city"),
        col("s.email"),
        col("s.change_date").alias("effective_from"),
        lit(None).cast("date").alias("effective_to"),
        lit(True).alias("is_current"),
    )
    new_rows.write.format("delta").mode("append").save(dim_path)

    print(f"SCD2: {len(changed_ids)} closed, {new_rows.count()} inserted")
    print("\nDimension after SCD2:")
    spark.read.format("delta").load(dim_path) \
        .orderBy("customer_id", "effective_from").show()

    # Query: what was Alice's city on 2024-03-01?
    as_of_date = "2024-03-01"
    result = spark.read.format("delta").load(dim_path) \
        .filter(
            (col("customer_id") == 1) &
            (col("effective_from") <= as_of_date) &
            (col("effective_to").isNull() | (col("effective_to") >= as_of_date))
        )
    print(f"Alice's city on {as_of_date}:")
    result.show()

    return base


# ── 4. Table Maintenance ──────────────────────────────────────────

def demo_maintenance(spark, table_path):
    """Demonstrate Delta Lake maintenance operations."""
    # View current table info
    print("=== Table History ===")
    spark.sql(f"DESCRIBE HISTORY delta.`{table_path}`") \
        .select("version", "timestamp", "operation").show()

    print("=== Table Detail ===")
    spark.sql(f"DESCRIBE DETAIL delta.`{table_path}`") \
        .select("format", "numFiles", "sizeInBytes").show()

    # OPTIMIZE: compact small files
    print("Running OPTIMIZE...")
    spark.sql(f"OPTIMIZE delta.`{table_path}`")

    # VACUUM: remove old files (set retention to 0 for demo)
    spark.conf.set("spark.databricks.delta.retentionDurationCheck.enabled", "false")
    print("Running VACUUM (0 hours retention for demo)...")
    spark.sql(f"VACUUM delta.`{table_path}` RETAIN 0 HOURS")


# ── 5. Time Travel ────────────────────────────────────────────────

def demo_time_travel(spark, table_path):
    """Demonstrate Delta Lake time travel."""
    # Current version
    current = spark.read.format("delta").load(table_path)
    print(f"Current version: {current.count()} rows")

    # Version 0
    v0 = spark.read.format("delta") \
        .option("versionAsOf", 0) \
        .load(table_path)
    print(f"Version 0: {v0.count()} rows")

    # Compare
    print("\nVersion 0 data:")
    v0.show()
    print("Current data:")
    current.show()

    # History
    print("Full history:")
    spark.sql(f"DESCRIBE HISTORY delta.`{table_path}`").show(truncate=False)


# ── 6. CDC to Delta Lake ──────────────────────────────────────────

def demo_cdc_to_delta(spark):
    """Apply simulated CDC events to a Delta table."""
    import tempfile
    import os
    from delta.tables import DeltaTable

    base = tempfile.mkdtemp(prefix="cdc_delta_")
    target_path = os.path.join(base, "orders")

    # Initial table
    initial = spark.createDataFrame([
        (1, "C001", "pending", 99.99),
        (2, "C002", "shipped", 149.50),
    ], ["id", "customer_id", "status", "amount"])
    initial.write.format("delta").save(target_path)
    print("Initial:")
    spark.read.format("delta").load(target_path).show()

    # Simulated CDC events (Debezium format)
    cdc_events = [
        {"op": "c", "after": {"id": 3, "customer_id": "C003", "status": "pending", "amount": 75.0}},
        {"op": "u", "before": {"id": 1}, "after": {"id": 1, "customer_id": "C001", "status": "shipped", "amount": 99.99}},
        {"op": "d", "before": {"id": 2, "customer_id": "C002", "status": "shipped", "amount": 149.50}},
    ]

    cdc_df = spark.createDataFrame(cdc_events)

    # Separate operations
    inserts = cdc_df.filter(col("op").isin("c", "r")).select("after.*")
    updates = cdc_df.filter(col("op") == "u").select("after.*")
    deletes = cdc_df.filter(col("op") == "d").select("before.id")

    target = DeltaTable.forPath(spark, target_path)

    # Apply upserts
    upserts = inserts.union(updates)
    if upserts.count() > 0:
        target.alias("t").merge(
            upserts.alias("s"),
            "t.id = s.id"
        ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()

    # Apply deletes
    if deletes.count() > 0:
        delete_ids = [row.id for row in deletes.collect()]
        target.delete(col("id").isin(delete_ids))

    print("After CDC events (1 insert, 1 update, 1 delete):")
    spark.read.format("delta").load(target_path).show()

    return base


# ── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    demos = {
        "medallion": lambda: demo_medallion(spark),
        "merge": lambda: demo_merge(spark),
        "scd2": lambda: demo_scd_type2(spark),
        "cdc": lambda: demo_cdc_to_delta(spark),
    }

    if len(sys.argv) < 2 or sys.argv[1] not in demos:
        print("Usage: spark-submit --packages io.delta:delta-spark_2.12:3.1.0 "
              "delta_lake_patterns.py <demo>")
        print(f"Available: {', '.join(demos.keys())}")
        print("\nRunning all demos...")
        for name, fn in demos.items():
            print(f"\n{'='*60}")
            print(f"  Demo: {name}")
            print(f"{'='*60}")
            fn()
    else:
        demos[sys.argv[1]]()

    spark.stop()
