"""
Spark Structured Streaming Examples
====================================
Demonstrates:
- Basic streaming from socket/rate source
- Windowed aggregations with watermark
- Stream-stream joins
- foreachBatch for multi-sink writes
- Kafka integration (read + write)
- Monitoring and checkpointing

Requirements:
    pip install pyspark

For Kafka examples:
    pip install pyspark[kafka]  # or include kafka jar manually
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, from_json, to_json, struct, to_timestamp,
    window, count, sum as spark_sum, avg, expr,
    explode, split, upper, current_timestamp, lit,
)
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
    IntegerType, TimestampType,
)


# ── 1. Basic Word Count (Socket Source) ─────────────────────────────

def word_count_socket(spark: SparkSession):
    """Classic word count on a streaming socket source.

    Run first:  nc -lk 9999
    Then type words into the terminal.
    """
    lines = spark.readStream \
        .format("socket") \
        .option("host", "localhost") \
        .option("port", 9999) \
        .load()

    words = lines.select(
        explode(split(col("value"), " ")).alias("word")
    )
    word_counts = words.groupBy("word").count()

    query = word_counts.writeStream \
        .outputMode("complete") \
        .format("console") \
        .trigger(processingTime="5 seconds") \
        .start()

    query.awaitTermination()


# ── 2. Rate Source with Windowed Aggregation ────────────────────────

def rate_source_windowed(spark: SparkSession):
    """Generate synthetic data and compute windowed aggregations."""
    # Rate source generates (timestamp, value) rows
    stream = spark.readStream \
        .format("rate") \
        .option("rowsPerSecond", 100) \
        .load()

    # Add a category column for grouping
    categorized = stream.withColumn(
        "category", (col("value") % 5).cast("string")
    )

    # Tumbling 30-second window with 10-second watermark
    windowed = categorized \
        .withWatermark("timestamp", "10 seconds") \
        .groupBy(
            window(col("timestamp"), "30 seconds"),
            col("category"),
        ) \
        .agg(
            count("*").alias("event_count"),
            spark_sum("value").alias("total_value"),
            avg("value").alias("avg_value"),
        )

    query = windowed.writeStream \
        .outputMode("update") \
        .format("console") \
        .option("truncate", "false") \
        .trigger(processingTime="10 seconds") \
        .start()

    query.awaitTermination()


# ── 3. JSON Parsing from Kafka ──────────────────────────────────────

def kafka_json_processing(spark: SparkSession):
    """Read JSON events from Kafka, parse, filter, and write back."""
    event_schema = StructType([
        StructField("user_id", StringType()),
        StructField("action", StringType()),
        StructField("amount", DoubleType()),
        StructField("timestamp", StringType()),
    ])

    # Read from Kafka
    raw = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "events") \
        .option("startingOffsets", "latest") \
        .option("maxOffsetsPerTrigger", 10000) \
        .load()

    # Parse JSON value column
    events = raw.select(
        col("key").cast("string").alias("kafka_key"),
        from_json(col("value").cast("string"), event_schema).alias("data"),
        col("timestamp").alias("kafka_timestamp"),
    ).select("kafka_key", "data.*", "kafka_timestamp")

    # Transform
    processed = events \
        .filter(col("amount") > 0) \
        .withColumn("event_time", to_timestamp(col("timestamp"))) \
        .withColumn("action_upper", upper(col("action")))

    # Write processed data back to Kafka
    output = processed.select(
        col("user_id").alias("key"),
        to_json(struct(
            col("action_upper").alias("action"),
            col("amount"),
            col("event_time"),
        )).alias("value"),
    )

    query = output.writeStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("topic", "processed_events") \
        .option("checkpointLocation", "/tmp/checkpoints/kafka_output") \
        .trigger(processingTime="10 seconds") \
        .start()

    query.awaitTermination()


# ── 4. Windowed Aggregation with Watermark ──────────────────────────

def windowed_aggregation(spark: SparkSession):
    """Tumbling and sliding windows with watermark for late data."""
    event_schema = StructType([
        StructField("user_id", StringType()),
        StructField("action", StringType()),
        StructField("amount", DoubleType()),
        StructField("event_time", TimestampType()),
    ])

    # Simulate with rate source + transformation
    stream = spark.readStream \
        .format("rate") \
        .option("rowsPerSecond", 50) \
        .load()

    events = stream.select(
        (col("value") % 10).cast("string").alias("user_id"),
        expr("CASE WHEN value % 3 = 0 THEN 'click' "
             "WHEN value % 3 = 1 THEN 'view' "
             "ELSE 'purchase' END").alias("action"),
        (col("value") * 0.5).alias("amount"),
        col("timestamp").alias("event_time"),
    )

    # Tumbling window: non-overlapping 1-minute windows
    tumbling = events \
        .withWatermark("event_time", "2 minutes") \
        .groupBy(
            window(col("event_time"), "1 minute"),
            col("action"),
        ) \
        .agg(
            count("*").alias("event_count"),
            spark_sum("amount").alias("total_amount"),
            avg("amount").alias("avg_amount"),
        )

    # Sliding window: 2-minute window sliding every 30 seconds
    sliding = events \
        .withWatermark("event_time", "2 minutes") \
        .groupBy(
            window(col("event_time"), "2 minutes", "30 seconds"),
            col("user_id"),
        ) \
        .agg(
            count("*").alias("event_count"),
            spark_sum("amount").alias("total_amount"),
        )

    # Write tumbling results
    query = tumbling.writeStream \
        .outputMode("update") \
        .format("console") \
        .option("truncate", "false") \
        .trigger(processingTime="15 seconds") \
        .start()

    query.awaitTermination()


# ── 5. Stream-Stream Join ───────────────────────────────────────────

def stream_stream_join(spark: SparkSession):
    """Join two streams: orders and shipments within a time window."""
    order_schema = StructType([
        StructField("order_id", StringType()),
        StructField("product", StringType()),
        StructField("amount", DoubleType()),
        StructField("order_time", TimestampType()),
    ])

    shipment_schema = StructType([
        StructField("order_id", StringType()),
        StructField("carrier", StringType()),
        StructField("ship_time", TimestampType()),
    ])

    # In production, these would come from Kafka topics
    orders = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "orders") \
        .load() \
        .select(from_json(col("value").cast("string"), order_schema).alias("o")) \
        .select("o.*")

    shipments = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "shipments") \
        .load() \
        .select(from_json(col("value").cast("string"), shipment_schema).alias("s")) \
        .select("s.*")

    # Both streams need watermarks for the join
    orders_wm = orders.withWatermark("order_time", "2 hours")
    shipments_wm = shipments.withWatermark("ship_time", "3 hours")

    # Inner join with time constraint
    joined = orders_wm.join(
        shipments_wm,
        expr("""
            o.order_id = s.order_id AND
            ship_time >= order_time AND
            ship_time <= order_time + interval 24 hours
        """),
        "inner",
    )

    query = joined.writeStream \
        .outputMode("append") \
        .format("console") \
        .option("truncate", "false") \
        .trigger(processingTime="30 seconds") \
        .start()

    query.awaitTermination()


# ── 6. foreachBatch: Multi-Sink Write ───────────────────────────────

def foreach_batch_multi_sink(spark: SparkSession):
    """Use foreachBatch to write each micro-batch to multiple sinks."""

    def process_batch(batch_df, batch_id):
        """Process each micro-batch with custom logic."""
        if batch_df.isEmpty():
            return

        # Cache to avoid recomputation across multiple writes
        batch_df.persist()

        # Sink 1: Write raw data to Parquet
        batch_df.write \
            .mode("append") \
            .parquet("/tmp/output/raw_events/")

        # Sink 2: Write aggregated summary
        summary = batch_df.groupBy("category").agg(
            count("*").alias("count"),
            spark_sum("value").alias("total"),
        )
        summary.write \
            .mode("append") \
            .json("/tmp/output/summaries/")

        batch_df.unpersist()
        print(f"Batch {batch_id}: {batch_df.count()} rows → 2 sinks")

    stream = spark.readStream \
        .format("rate") \
        .option("rowsPerSecond", 50) \
        .load() \
        .withColumn("category", (col("value") % 5).cast("string"))

    query = stream.writeStream \
        .foreachBatch(process_batch) \
        .option("checkpointLocation", "/tmp/checkpoints/multi_sink/") \
        .trigger(processingTime="30 seconds") \
        .start()

    query.awaitTermination()


# ── 7. Deduplication ────────────────────────────────────────────────

def deduplication_example(spark: SparkSession):
    """Deduplicate streaming events by event_id within a watermark."""
    event_schema = StructType([
        StructField("event_id", StringType()),
        StructField("user_id", StringType()),
        StructField("action", StringType()),
        StructField("event_time", TimestampType()),
    ])

    events = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "events") \
        .load() \
        .select(from_json(col("value").cast("string"), event_schema).alias("e")) \
        .select("e.*")

    # Deduplicate within a 10-minute watermark window
    deduplicated = events \
        .withWatermark("event_time", "10 minutes") \
        .dropDuplicates(["event_id", "event_time"])

    query = deduplicated.writeStream \
        .outputMode("append") \
        .format("console") \
        .trigger(processingTime="10 seconds") \
        .start()

    query.awaitTermination()


# ── 8. Monitoring Query Progress ────────────────────────────────────

def monitored_query(spark: SparkSession):
    """Start a query and monitor its progress."""
    stream = spark.readStream \
        .format("rate") \
        .option("rowsPerSecond", 100) \
        .load()

    query = stream.writeStream \
        .format("console") \
        .trigger(processingTime="10 seconds") \
        .start()

    import time
    for _ in range(5):
        time.sleep(15)

        # Query status
        print(f"\n--- Query Status ---")
        print(f"ID:       {query.id}")
        print(f"Run ID:   {query.runId}")
        print(f"Active:   {query.isActive}")
        print(f"Status:   {query.status}")

        # Last progress metrics
        progress = query.lastProgress
        if progress:
            print(f"Batch:    {progress.get('batchId', 'N/A')}")
            print(f"Input:    {progress.get('numInputRows', 0)} rows")
            print(f"Rate in:  {progress.get('inputRowsPerSecond', 0):.1f} rows/s")
            print(f"Rate out: {progress.get('processedRowsPerSecond', 0):.1f} rows/s")
            duration = progress.get("durationMs", {})
            print(f"Duration: {duration}")

    query.stop()


# ── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("StructuredStreamingExamples") \
        .master("local[*]") \
        .config("spark.sql.shuffle.partitions", "4") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    import sys
    examples = {
        "wordcount": word_count_socket,
        "rate": rate_source_windowed,
        "kafka": kafka_json_processing,
        "window": windowed_aggregation,
        "join": stream_stream_join,
        "foreach": foreach_batch_multi_sink,
        "dedup": deduplication_example,
        "monitor": monitored_query,
    }

    if len(sys.argv) < 2 or sys.argv[1] not in examples:
        print("Usage: spark-submit structured_streaming.py <example>")
        print(f"Available: {', '.join(examples.keys())}")
        print("\nRunning 'rate' example by default...")
        rate_source_windowed(spark)
    else:
        examples[sys.argv[1]](spark)
