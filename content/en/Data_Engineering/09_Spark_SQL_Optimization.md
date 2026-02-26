# Spark SQL Optimization

## Learning Objectives

After completing this lesson, you will be able to:

1. Read and interpret Spark execution plans using the explain() method to identify performance bottlenecks
2. Describe the four phases of the Catalyst optimizer (Analysis, Logical Optimization, Physical Planning, Code Generation) and explain key optimizations like predicate pushdown and column pruning
3. Apply partitioning strategies (repartition, coalesce, partitionBy) and caching (cache, persist) to improve job performance
4. Choose the correct join strategy (Broadcast, Sort-Merge, Shuffle Hash) based on data size and explain the skew join problem and mitigation techniques
5. Configure Adaptive Query Execution (AQE) settings to enable runtime query plan optimizations
6. Diagnose and resolve common performance issues such as data skew, excessive shuffles, and small file problems using Spark UI and best practices

---

## Overview

To optimize Spark SQL performance, you need to understand how the Catalyst optimizer works and properly utilize partitioning, caching, join strategies, and other techniques.

---

## 1. Catalyst Optimizer

### 1.1 Understanding Execution Plans

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("Optimization").getOrCreate()

df = spark.read.parquet("sales.parquet")

# Build query — no execution happens yet. Spark constructs a logical plan that
# the Catalyst optimizer will rewrite before choosing a physical execution strategy.
query = df.filter(col("amount") > 100) \
          .groupBy("category") \
          .sum("amount")

# "simple" shows only the physical plan — start here to see what Spark actually executes
query.explain(mode="simple")

# "extended" shows all 4 plans (parsed → analyzed → optimized → physical) — use
# to verify that Catalyst applied expected optimizations like predicate pushdown
query.explain(mode="extended")

# "cost" adds row count and size estimates — helps understand why Spark chose
# broadcast vs sort-merge join (based on estimated table sizes)
query.explain(mode="cost")

# "formatted" produces the most readable output with operator details — best for
# identifying bottlenecks (look for Exchange nodes which indicate shuffles)
query.explain(mode="formatted")
```

### 1.2 Catalyst Optimization Phases

```
┌─────────────────────────────────────────────────────────────────┐
│                   Catalyst Optimizer Phases                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   1. Analysis                                                   │
│      - Verify column/table names                               │
│      - Type validation                                         │
│      ↓                                                          │
│   2. Logical Optimization                                       │
│      - Predicate Pushdown                                      │
│      - Column Pruning                                          │
│      - Constant Folding                                        │
│      ↓                                                          │
│   3. Physical Planning                                          │
│      - Select join strategy                                    │
│      - Select aggregation strategy                             │
│      ↓                                                          │
│   4. Code Generation                                            │
│      - Whole-Stage Code Generation                             │
│      - JIT compilation                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 Key Optimization Techniques

```python
# 1. Predicate Pushdown — Catalyst moves filters into the data source scan.
# For Parquet, this skips entire row groups whose min/max stats don't match,
# potentially reading 10-100x less data from disk.
df = spark.read.parquet("data.parquet")
filtered = df.filter(col("date") == "2024-01-01")  # Parquet skips row groups outside this date

# 2. Column Pruning — Spark reads only referenced columns from columnar formats.
# In Parquet, unreferenced columns are never deserialized from disk, saving both
# I/O and memory. Always select specific columns instead of SELECT *.
df.select("name", "amount")  # Other columns not read

# 3. Projection Pushdown — for JDBC sources, pushes WHERE and SELECT into the
# generated SQL query, so the database returns only needed data over the network
df = spark.read.format("jdbc") \
    .option("pushDownPredicate", "true") \
    .load()

# 4. Constant Folding — Catalyst evaluates constant expressions at plan time,
# avoiding repeated computation per row. Happens automatically; no tuning needed.
df.filter(col("value") > 1 + 2)  # Optimized to > 3 before execution
```

---

## 2. Partitioning

### 2.1 Partition Concepts

```python
# Check number of partitions
df.rdd.getNumPartitions()

# repartition() triggers a FULL shuffle — every record is redistributed.
# Use when you need to INCREASE partition count or redistribute by specific columns.
df.repartition(100)                      # Hash-based redistribution into 100 partitions
df.repartition("date")                   # Co-locates same-date rows — useful before date-based joins
df.repartition(100, "date", "category")  # Hash on composite key for even distribution

# coalesce() only DECREASES partitions by merging adjacent ones — no shuffle
# required. Use after filter() operations that leave many nearly-empty partitions.
# Cannot increase partition count (use repartition for that).
df.coalesce(10)

# Diagnostic helper — use to detect data skew (some partitions much larger than others).
# glom().collect() pulls partition data to Driver, so only use on small datasets.
def print_partition_info(df):
    print(f"Partitions: {df.rdd.getNumPartitions()}")
    for idx, partition in enumerate(df.rdd.glom().collect()):
        print(f"Partition {idx}: {len(partition)} rows")
```

### 2.2 Partitioning Strategies

```python
# Calculate appropriate number of partitions
"""
Recommended formula:
- Number of partitions = Data size (MB) / 128MB  → targets ~128MB per partition (Spark's sweet spot)
- Or: Cluster cores * 2~4  → ensures all cores stay busy with task-level parallelism

Too few partitions → large tasks that OOM or underutilize cores.
Too many partitions → excessive scheduling overhead and small-file problem.

Examples:
- 10GB data → 10,000MB / 128MB ≈ 80 partitions
- 100 core cluster → 200~400 partitions
"""

# This sets partition count for ALL shuffles (joins, groupBy, etc.) — a global
# default. With AQE enabled, Spark can auto-coalesce small partitions at runtime.
spark.conf.set("spark.sql.shuffle.partitions", 200)

# repartitionByRange creates sorted, non-overlapping partitions — ideal for
# range-based queries (e.g., date ranges) because each partition covers a contiguous
# key range, enabling efficient partition pruning.
df.repartitionByRange(100, "date")

# Hash partitioning ensures same key always goes to same partition — essential
# before joins on user_id to avoid shuffles during the join itself.
df.repartition(100, "user_id")
```

### 2.3 Partition Storage

```python
# partitionBy creates Hive-style directory structure — enables partition pruning
# where queries filtering on year/month skip entire directories without reading any files.
# Choose partition columns carefully: too high cardinality → millions of tiny files.
df.write \
    .partitionBy("year", "month") \
    .parquet("output/partitioned_data")

# Resulting directory structure:
# output/partitioned_data/
#   year=2024/
#     month=01/
#       part-00000.parquet
#     month=02/
#       part-00000.parquet

# Spark recognizes partition columns from directory names — this filter reads
# only the year=2024/month=01 subdirectory, skipping all other months/years entirely
df = spark.read.parquet("output/partitioned_data")
df.filter((col("year") == 2024) & (col("month") == 1))

# Bucketing pre-shuffles and sorts data at write time — subsequent joins on
# user_id between two identically bucketed tables require NO shuffle at read time.
# Trade-off: slower writes but dramatically faster repeated joins on the same key.
df.write \
    .bucketBy(100, "user_id") \      # 100 buckets = 100 files, each containing a hash range of user_ids
    .sortBy("timestamp") \            # Pre-sorting within buckets speeds up merge joins and range scans
    .saveAsTable("bucketed_table")    # Must use saveAsTable (not write.parquet) — bucket metadata stored in Hive metastore
```

---

## 3. Caching

### 3.1 Cache Basics

```python
# cache() is lazy — actual caching happens on the first action. The DataFrame is
# then kept in executor memory across subsequent actions, avoiding recomputation.
df.cache()           # Alias for persist(MEMORY_AND_DISK)
df.persist()         # Same as cache()

# Choose storage level based on your memory/CPU/disk trade-offs:
from pyspark import StorageLevel

df.persist(StorageLevel.MEMORY_ONLY)           # Fastest reads but evicts partitions if memory is tight — recomputes on miss
df.persist(StorageLevel.MEMORY_AND_DISK)       # Spills to disk instead of recomputing — best default for most workloads
df.persist(StorageLevel.MEMORY_ONLY_SER)       # Serialized = ~2-5x less memory but adds CPU cost for deserialization
df.persist(StorageLevel.DISK_ONLY)             # When data is too large for memory — still faster than recomputing from source
df.persist(StorageLevel.MEMORY_AND_DISK_SER)   # Serialized + disk fallback — maximum memory efficiency

# Always unpersist when done — cached data occupies executor memory that could
# be used for shuffle/execution. Stale caches are a common cause of OOM.
df.unpersist()

# Check cache status
spark.catalog.isCached("table_name")
```

### 3.2 Caching Strategies

```python
# Cache when: the same DataFrame is used in 2+ actions AND recomputation is expensive.
# Don't cache: one-shot transformations, or when the data doesn't fit in memory
# (caching causes eviction thrashing that makes performance worse).

# Example: This join is expensive (shuffles two large tables). Without caching,
# each of the three aggregations below would repeat the full read + filter + join.
expensive_df = spark.read.parquet("large_data.parquet") \
    .filter(col("status") == "active") \
    .join(other_df, "key")

# cache() marks the DAG breakpoint — first action materializes and stores the result
expensive_df.cache()

# All three actions reuse the cached result — 3x speedup vs recomputing the join each time
result1 = expensive_df.groupBy("category").count()
result2 = expensive_df.groupBy("region").sum("amount")
result3 = expensive_df.filter(col("amount") > 1000).count()

# Explicit unpersist is important — Spark's LRU eviction may keep stale caches
# that waste memory. In long-running applications, always clean up.
expensive_df.unpersist()
```

### 3.3 Cache Monitoring

```python
# Check in Spark UI (http://localhost:4040/storage)

# Programmatic checking
sc = spark.sparkContext

# List cached RDDs
for rdd_id, rdd_info in sc._jsc.sc().getRDDStorageInfo():
    print(f"RDD {rdd_id}: {rdd_info}")

# Clear all caches
spark.catalog.clearCache()
```

---

## 4. Join Strategies

### 4.1 Join Type Characteristics

```python
# Spark join strategies:
join_strategies = {
    "Broadcast Hash Join": {
        "condition": "Small table (< 10MB default)",
        "performance": "Fastest",
        "shuffle": "None (broadcast small table)"
    },
    "Sort Merge Join": {
        "condition": "Join between large tables",
        "performance": "Stable",
        "shuffle": "Shuffle + sort both tables"
    },
    "Shuffle Hash Join": {
        "condition": "When one side is smaller",
        "performance": "Medium",
        "shuffle": "Shuffle both sides"
    },
    "Broadcast Nested Loop Join": {
        "condition": "No join condition (Cross)",
        "performance": "Slow",
        "shuffle": "None (broadcast)"
    }
}
```

### 4.2 Force Broadcast Join

```python
from pyspark.sql.functions import broadcast

# broadcast() sends the entire small DataFrame to every executor — eliminates the
# shuffle on the large side entirely. The small table must fit in Driver + executor memory.
large_df.join(broadcast(small_df), "key")

# Default auto-broadcast threshold is 10MB — increase for larger dimension tables
# that still fit in memory. Set too high and you risk OOM on executors.
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", 100 * 1024 * 1024)  # 100MB

# Disable auto-broadcast to force sort-merge join — useful for benchmarking
# or when Spark's size estimates are inaccurate (common with complex subqueries)
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)

# SQL hint overrides the optimizer's decision — use when you know better than
# Spark's cost model (e.g., after filtering reduces a table to broadcast size)
spark.sql("""
    SELECT /*+ BROADCAST(small_table) */
        large_table.*, small_table.name
    FROM large_table
    JOIN small_table ON large_table.id = small_table.id
""")
```

### 4.3 Join Optimization Tips

```python
# 1. Filter before join — reduces the number of rows entering the shuffle.
# Catalyst often does this automatically (predicate pushdown), but placing
# filters early makes intent clear and helps in cases where pushdown fails.
# Bad
df1.join(df2, "key").filter(col("status") == "active")

# Good — fewer rows to shuffle and join
df1.filter(col("status") == "active").join(df2, "key")


# 2. Type mismatches force implicit casting on EVERY row — this disables
# predicate pushdown and prevents Spark from using optimized join paths.
# Bad (type mismatch causes implicit casting)
df1.join(df2, df1.id == df2.id)  # id is string vs int

# Good — cast once upfront, then join efficiently
df1 = df1.withColumn("id", col("id").cast("int"))
df1.join(df2, "id")


# 3. Skew join splits oversized partitions at runtime — if a partition is
# skewedPartitionFactor (5x) larger than the median AND exceeds the threshold
# (256MB), AQE automatically splits it into smaller sub-partitions.
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", True)
spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionFactor", 5)
spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", "256MB")


# 4. Bucketing pre-partitions data at write time — both tables must use the same
# bucket count and key. Subsequent joins skip the shuffle entirely because matching
# keys are guaranteed to be in the same bucket number on both sides.
df.write.bucketBy(100, "user_id").saveAsTable("users_bucketed")
other_df.write.bucketBy(100, "user_id").saveAsTable("orders_bucketed")

# No Exchange (shuffle) node in the execution plan — verify with .explain()
spark.table("users_bucketed").join(spark.table("orders_bucketed"), "user_id")
```

---

## 5. Performance Tuning

### 5.1 Configuration Optimization

```python
# Memory settings — executor memory is split into execution (shuffles, joins, sorts)
# and storage (cache). memory.fraction controls the total usable fraction (80% of heap),
# storageFraction reserves 30% of that for caching — lower this if joins/shuffles OOM.
spark = SparkSession.builder \
    .config("spark.executor.memory", "8g") \
    .config("spark.executor.memoryOverhead", "2g") \   # Off-heap for Python, network buffers — increase for PySpark workloads
    .config("spark.driver.memory", "4g") \              # Needs headroom for broadcast vars and collect() results
    .config("spark.memory.fraction", "0.8") \           # 20% reserved for user data structures and internal metadata
    .config("spark.memory.storageFraction", "0.3") \    # Storage can borrow from execution (unified memory model) but not vice versa
    .getOrCreate()

# default.parallelism affects RDD operations; shuffle.partitions affects DataFrame operations.
# Set both to 2-3x total cluster cores for balanced task-level parallelism.
spark.conf.set("spark.default.parallelism", 200)
spark.conf.set("spark.sql.shuffle.partitions", 200)

# AQE re-optimizes plans at stage boundaries using actual runtime statistics —
# handles problems that static optimization cannot predict (skew, partition sizing).
spark.conf.set("spark.sql.adaptive.enabled", True)
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", True)    # Merges small post-shuffle partitions to target size
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", True)              # Splits oversized partitions during joins
spark.conf.set("spark.sql.adaptive.localShuffleReader.enabled", True)    # Reads shuffle data locally when possible (avoids network)

# Kryo serialization is 10x faster and 2-5x more compact than Java default —
# essential for performance-sensitive jobs. Register custom classes to avoid overhead.
spark.conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

# Dynamic allocation auto-scales executors based on workload — minExecutors prevents
# cold-start latency, maxExecutors caps resource usage for cost control.
spark.conf.set("spark.dynamicAllocation.enabled", True)
spark.conf.set("spark.dynamicAllocation.minExecutors", 2)
spark.conf.set("spark.dynamicAllocation.maxExecutors", 100)
```

### 5.2 Data Format Optimization

```python
# Snappy offers fast compression/decompression with moderate ratio (~2x).
# zstd achieves better compression (~3-4x) at higher CPU cost — prefer zstd
# for cold storage, snappy for frequently-read hot data.
spark.conf.set("spark.sql.parquet.compression.codec", "snappy")
spark.conf.set("spark.sql.parquet.filterPushdown", True)  # Uses Parquet row group min/max stats to skip data

# maxPartitionBytes caps input split size — 128MB aligns with HDFS block size,
# creating one task per block. openCostInBytes penalizes opening many small files,
# encouraging Spark to combine them into larger splits.
spark.conf.set("spark.sql.files.maxPartitionBytes", "128MB")
spark.conf.set("spark.sql.files.openCostInBytes", "4MB")

# parallelismFirst=False tells AQE to prioritize target partition size (128MB)
# over parallelism — essential for fixing the "too many small files" problem
# that degrades downstream read performance.
spark.conf.set("spark.sql.adaptive.coalescePartitions.parallelismFirst", False)
spark.conf.set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "128MB")

# Always verify column pruning with explain() — if you see ReadSchema with only
# your requested columns, pruning is working. Full schema reads indicate a problem.
df.select("needed_column1", "needed_column2").explain()
```

### 5.3 Shuffle Optimization

```python
# AQE is the best first approach — it dynamically coalesces small partitions
# and splits large ones at runtime, adapting to actual data distribution.
spark.conf.set("spark.sql.adaptive.enabled", True)

# Manual calculation when AQE is unavailable: target ~128MB per partition
# to balance between task scheduling overhead and memory usage
data_size_gb = 10
partition_size_mb = 128
optimal_partitions = (data_size_gb * 1024) // partition_size_mb  # = 80 partitions
spark.conf.set("spark.sql.shuffle.partitions", optimal_partitions)

# Shuffle compression trades CPU for network I/O — almost always a net win
# because shuffles are typically network-bound, not CPU-bound
spark.conf.set("spark.shuffle.compress", True)

# Spill compression reduces disk usage when shuffle data exceeds executor memory —
# without it, spilled data uses raw disk space which can fill up fast
spark.conf.set("spark.shuffle.spill.compress", True)

# External shuffle service runs independently of executors — enables dynamic
# allocation (executors can be removed without losing their shuffle data)
spark.conf.set("spark.shuffle.service.enabled", True)
```

---

## 6. Performance Monitoring

### 6.1 Using Spark UI

```python
# Access Spark UI: http://<driver-host>:4040

# Information by UI tab:
"""
Jobs: Job execution status, time
Stages: Stage details (shuffle, data size)
Storage: Cached RDD/DataFrame
Environment: Configuration values
Executors: Executor status, memory
SQL: SQL query plans
"""

# History server (for completed jobs)
# spark.eventLog.enabled=true
# spark.history.fs.logDirectory=hdfs:///spark-history
```

### 6.2 Programmatic Monitoring

```python
# Measure execution time
import time

start = time.time()
result = df.groupBy("category").count().collect()
end = time.time()
print(f"Execution time: {end - start:.2f} seconds")

# Check shuffle in execution plan
df.explain(mode="formatted")

# Check join strategy in physical plan
# Exchange = shuffle occurs
# BroadcastHashJoin = broadcast join
# SortMergeJoin = sort merge join
```

### 6.3 Metrics Collection

```python
# Size estimation uses Catalyst statistics (not actual data scan) — results may
# be inaccurate without ANALYZE TABLE. Useful for deciding broadcast vs sort-merge join.
def estimate_size(df):
    """Estimate DataFrame size (bytes)"""
    return df._jdf.queryExecution().optimizedPlan().stats().sizeInBytes()

# Partition-level row counts reveal data skew — a skew ratio > 3-5x indicates
# some tasks will be much slower than others, becoming the bottleneck.
# Note: mapPartitions + collect requires materializing the DataFrame — use on
# cached data to avoid triggering a full recomputation.
partition_counts = df.rdd.mapPartitions(
    lambda it: [sum(1 for _ in it)]
).collect()

print(f"Min: {min(partition_counts)}, Max: {max(partition_counts)}")
print(f"Skew ratio: {max(partition_counts) / (sum(partition_counts) / len(partition_counts)):.2f}")
```

---

## 7. Common Performance Issues and Solutions

### 7.1 Data Skew

```python
# Problem: Data concentrated in specific keys (e.g., 90% of orders from one customer)
# Symptom: One task runs 100x longer than others — visible in Spark UI stage timeline

# Solution 1: AQE detects skew at runtime and automatically splits large partitions
# into sub-partitions — simplest fix, try this first
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", True)

# Solution 2: Salt key — manually break hot keys into N sub-keys to distribute
# their data across N partitions. Trade-off: requires expanding the other table
# via cross join, increasing its size by N×.
from pyspark.sql.functions import rand, floor

num_salts = 10  # Choose based on skew severity — higher = more parallelism but larger cross join
df_salted = df.withColumn("salt", floor(rand() * num_salts))

# The small table is replicated N times (one per salt value) so every salted
# key-partition on the large side finds a matching row on the small side
result = df_salted.join(
    other_df.crossJoin(
        spark.range(num_salts).withColumnRenamed("id", "salt")
    ),
    ["key", "salt"]
).drop("salt")

# Solution 3: Broadcast avoids the shuffle entirely — best when the skewed table
# is being joined with a small dimension table that fits in executor memory
result = df.join(broadcast(small_df), "key")
```

### 7.2 OOM (Out of Memory)

```python
# Problem: Memory shortage — can occur in executor (too much data per partition)
# or Driver (collect/broadcast too large). Check the error stacktrace to identify which.

# Solution 1: More memory per executor — quick fix but expensive.
# Also increase overhead for PySpark (Python workers use off-heap memory).
spark.conf.set("spark.executor.memory", "8g")
spark.conf.set("spark.executor.memoryOverhead", "2g")

# Solution 2: More partitions = less data per task = lower per-task memory.
# Often more effective than adding memory — distributes the problem instead of throwing RAM at it.
df.repartition(500)

# Solution 3: Stale caches are a hidden OOM cause — cached DataFrames occupy
# storage memory that execution (shuffles/joins) cannot reclaim
spark.catalog.clearCache()

# Solution 4: If broadcast join OOMs, the "small" table was larger than expected.
# Lowering the threshold forces sort-merge join which streams data instead of
# loading the entire table into memory.
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "10MB")
```

### 7.3 Excessive Shuffling

```python
# Problem: Network/disk I/O due to shuffle
# Symptom: Increased wait time between stages

# Solution 1: Filter before shuffle
df.filter(col("status") == "active").groupBy("key").count()

# Solution 2: Change partitioning strategy
# Data partitioned by same key can join without shuffle
df1.repartition(100, "key").join(df2.repartition(100, "key"), "key")

# Solution 3: Use bucketing
df.write.bucketBy(100, "key").saveAsTable("bucketed_table")
```

---

## Practice Problems

### Problem 1: Execution Plan Analysis
Analyze the execution plan of a given query and find optimization points.

### Problem 2: Join Optimization
Design the optimal method to join a transaction table with 100 million records and a customer table with 1 million records.

### Problem 3: Skew Handling
Improve aggregation performance when data is concentrated in specific categories.

---

## Summary

| Optimization Area | Techniques |
|-------------------|------------|
| **Catalyst** | Predicate Pushdown, Column Pruning |
| **Partitioning** | repartition, coalesce, partitionBy |
| **Caching** | cache, persist, StorageLevel |
| **Join** | Broadcast, Sort Merge, Bucketing |
| **AQE** | Automatic partition coalescing, skew handling |

---

## References

- [Spark SQL Tuning](https://spark.apache.org/docs/latest/sql-performance-tuning.html)
- [Spark Configuration](https://spark.apache.org/docs/latest/configuration.html)
- [Adaptive Query Execution](https://spark.apache.org/docs/latest/sql-performance-tuning.html#adaptive-query-execution)
