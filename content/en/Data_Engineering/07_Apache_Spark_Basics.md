# Apache Spark Basics

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the Apache Spark architecture including the Driver, Executors, and Cluster Manager, and describe how Spark's in-memory processing achieves performance gains over Hadoop MapReduce
2. Distinguish between RDDs (Resilient Distributed Datasets), DataFrames, and Datasets, and explain when to use each abstraction
3. Create a SparkSession and perform basic data loading, transformation, and action operations
4. Apply common DataFrame transformations such as filtering, grouping, joining, and aggregation using the PySpark API
5. Explain lazy evaluation and describe how Spark's DAG execution model optimizes query plans
6. Configure Spark job parameters and read data from distributed storage sources such as HDFS, S3, and Parquet files

---

## Overview

Apache Spark is a unified analytics engine for large-scale data processing. It provides faster performance than Hadoop MapReduce through in-memory processing and supports both batch processing and streaming.

---

## 1. Spark Overview

### 1.1 Spark Features

```
┌────────────────────────────────────────────────────────────────┐
│                    Apache Spark Features                        │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   1. Speed                                                     │
│      - 100x faster than Hadoop with in-memory processing       │
│      - 10x faster than disk-based processing                   │
│                                                                │
│   2. Ease of Use                                               │
│      - Supports Python, Scala, Java, R                         │
│      - Provides SQL interface                                  │
│                                                                │
│   3. Generality                                                │
│      - SQL, streaming, ML, graph processing                    │
│      - Diverse workloads with one engine                       │
│                                                                │
│   4. Compatibility                                             │
│      - Various data sources: HDFS, S3, Cassandra, etc.         │
│      - YARN, Kubernetes, Standalone clusters                   │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 1.2 Spark Ecosystem

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

## 2. Spark Architecture

### 2.1 Cluster Configuration

```
┌─────────────────────────────────────────────────────────────────┐
│                    Spark Cluster Architecture                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌───────────────────────────────────────────────────────┐    │
│   │                    Driver Program                      │    │
│   │   ┌─────────────────────────────────────────────────┐ │    │
│   │   │              SparkContext                        │ │    │
│   │   │   - Application entry point                      │ │    │
│   │   │   - Connects to cluster                          │ │    │
│   │   │   - Job creation and scheduling                  │ │    │
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

### 2.2 Core Concepts

| Concept | Description |
|---------|-------------|
| **Driver** | Executes main program, creates SparkContext |
| **Executor** | Executes tasks on worker nodes |
| **Task** | Basic unit of execution |
| **Job** | Parallel computation triggered by an action |
| **Stage** | Group of tasks within a job (shuffle boundary) |
| **Partition** | Logical division unit of data |

### 2.3 Execution Flow

```python
"""
Spark execution flow:
1. Create SparkContext in Driver
2. Parse application code
3. Transformations → Create DAG (Directed Acyclic Graph)
4. Create job when action is called
5. Decompose job → Stages → Tasks
6. Cluster Manager assigns tasks to Executors
7. Executors execute tasks
8. Return results to Driver
"""

# Example code flow
from pyspark.sql import SparkSession

# SparkSession is the unified entry point since Spark 2.0 — replaces the separate
# SparkContext, SQLContext, and HiveContext that earlier versions required
spark = SparkSession.builder.appName("Example").getOrCreate()

# Transformations are lazy — Spark builds a DAG (execution plan) but does NOT
# read or process any data yet. This enables the Catalyst optimizer to reorder
# and fuse operations before execution.
df = spark.read.csv("data.csv", header=True)  # Read plan
df2 = df.filter(df.age > 20)                  # Filter plan
df3 = df2.groupBy("city").count()             # Aggregation plan

# Actions trigger the full DAG execution. collect() materializes all data to the
# Driver — safe for small results but will OOM on large datasets (use .show() or
# .write instead for large outputs).
result = df3.collect()
```

---

## 3. RDD (Resilient Distributed Dataset)

### 3.1 RDD Concept

RDD is Spark's fundamental data structure, an immutable distributed collection of data.

```python
from pyspark import SparkContext

# "local[*]" runs Spark in local mode using all available CPU cores — ideal for
# development/testing. In production, use "yarn" or "k8s://..." for cluster mode.
sc = SparkContext("local[*]", "RDD Example")

# Ways to create RDD
# 1. parallelize() distributes a local Python collection across partitions.
# Default partition count = number of cores. Useful for testing; in production
# data comes from external sources.
rdd1 = sc.parallelize([1, 2, 3, 4, 5])

# 2. textFile creates one partition per HDFS block (128MB default) — Spark
# automatically parallelizes reading based on file size
rdd2 = sc.textFile("data.txt")

# 3. Transformations always produce new RDDs — RDDs are immutable, which enables
# lineage-based fault recovery: if a partition is lost, Spark replays only
# the transformations needed to recompute that partition
rdd3 = rdd1.map(lambda x: x * 2)

# RDD properties
"""
R - Resilient: Fault-recoverable (recompute via lineage)
D - Distributed: Distributed across cluster
D - Dataset: Data collection
"""
```

### 3.2 RDD Operations

```python
# Transformations (Lazy)
# - Return new RDD without executing — Spark records the computation as a lineage
#   graph. This laziness enables the optimizer to combine operations and minimize
#   data movement before any work happens.

rdd = sc.parallelize([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# map: 1-to-1 transformation, preserves partition count
mapped = rdd.map(lambda x: x * 2)  # [2, 4, 6, ...]

# filter: Narrows data early to reduce downstream processing — always push
# filters as early as possible in the pipeline
filtered = rdd.filter(lambda x: x % 2 == 0)  # [2, 4, 6, 8, 10]

# flatMap: 1-to-many mapping — useful for tokenization (e.g., splitting lines into words)
flat = rdd.flatMap(lambda x: [x, x*2])  # [1, 2, 2, 4, 3, 6, ...]

# distinct: Requires shuffle (expensive) — only use when duplicates actually matter
distinct = rdd.distinct()

# union: Logical merge without data movement — partitions from both RDDs are concatenated
union = rdd.union(sc.parallelize([11, 12]))

# groupByKey: Shuffles ALL values to the key's partition — memory-intensive because
# all values for a key must fit in memory. Prefer reduceByKey when possible.
pairs = sc.parallelize([("a", 1), ("b", 2), ("a", 3)])
grouped = pairs.groupByKey()  # [("a", [1, 3]), ("b", [2])]

# reduceByKey: Performs local combine BEFORE shuffle (like a mini-MapReduce combiner),
# drastically reducing network transfer. Always prefer over groupByKey + reduce.
reduced = pairs.reduceByKey(lambda a, b: a + b)  # [("a", 4), ("b", 2)]


# Actions (Eager)
# - Trigger the full lineage execution — Spark submits a job to the cluster,
#   breaking it into stages at shuffle boundaries and tasks per partition.

# collect: Pulls ALL data to Driver memory — use only for small results.
# For large datasets, use take() or write to storage instead.
result = rdd.collect()  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# count: Triggers execution but returns a single number — safe for any data size
count = rdd.count()  # 10

# first / take: Only processes enough partitions to get N elements — much
# cheaper than collect() for previewing data
first = rdd.first()  # 1
take3 = rdd.take(3)  # [1, 2, 3]

# reduce: Combines all elements into one — runs in parallel within each partition
# then aggregates results on the Driver
total = rdd.reduce(lambda a, b: a + b)  # 55

# foreach: Runs on executors (not Driver) — use for side effects like writing
# to external systems. Return values are discarded.
rdd.foreach(lambda x: print(x))

# saveAsTextFile: Writes one file per partition — use coalesce() first if you
# want fewer output files
rdd.saveAsTextFile("output/")
```

### 3.3 Pair RDD Operations

```python
# Key-Value pair RDD operations
sales = sc.parallelize([
    ("Electronics", 100),
    ("Clothing", 50),
    ("Electronics", 200),
    ("Clothing", 75),
    ("Food", 30),
])

# reduceByKey pre-aggregates within each partition before shuffling — far more
# efficient than groupByKey().mapValues(sum) which shuffles raw values first
total_by_category = sales.reduceByKey(lambda a, b: a + b)
# [("Electronics", 300), ("Clothing", 125), ("Food", 30)]

# combineByKey is the most general aggregation — use when the accumulated type
# differs from the value type (here: value=int, accumulator=(sum, count) tuple).
# Three functions handle: creating the first accumulator, merging a value into
# an existing accumulator, and merging two accumulators across partitions.
count_sum = sales.combineByKey(
    lambda v: (v, 1),                      # createCombiner: first value in partition
    lambda acc, v: (acc[0] + v, acc[1] + 1),  # mergeValue: add to existing accumulator
    lambda acc1, acc2: (acc1[0] + acc2[0], acc1[1] + acc2[1])  # mergeCombiner: cross-partition merge
)
avg_by_category = count_sum.mapValues(lambda x: x[0] / x[1])

# sortByKey requires full shuffle to range-partition data — expensive for large datasets.
# Only sort when the consumer truly needs ordered output.
sorted_rdd = sales.sortByKey()

# Join shuffles both RDDs by key and matches records — produces a Cartesian product
# per key. If one side is small (< 10MB), consider broadcast join via sc.broadcast()
# to avoid the expensive shuffle.
inventory = sc.parallelize([
    ("Electronics", 50),
    ("Clothing", 100),
])

joined = sales.join(inventory)
# [("Electronics", (100, 50)), ("Electronics", (200, 50)), ...]
```

---

## 4. Installation and Execution

### 4.1 Local Installation (PySpark)

```bash
# pip installation
pip install pyspark

# Check version
pyspark --version

# Start PySpark shell
pyspark

# Execute script with spark-submit
spark-submit my_script.py
```

### 4.2 Docker Installation

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
# Run
docker-compose up -d

# Submit job to cluster
spark-submit --master spark://localhost:7077 my_script.py
```

### 4.3 Cluster Mode

```bash
# Standalone cluster
spark-submit \
    --master spark://master:7077 \
    --deploy-mode cluster \
    --executor-memory 4G \
    --executor-cores 2 \
    --num-executors 10 \
    my_script.py

# YARN cluster
spark-submit \
    --master yarn \
    --deploy-mode cluster \
    --executor-memory 4G \
    my_script.py

# Kubernetes cluster
spark-submit \
    --master k8s://https://k8s-master:6443 \
    --deploy-mode cluster \
    --conf spark.kubernetes.container.image=my-spark-image \
    my_script.py
```

---

## 5. SparkSession

### 5.1 Creating SparkSession

```python
from pyspark.sql import SparkSession

# getOrCreate() reuses an existing SparkSession if one exists in the JVM —
# prevents the common error of creating multiple sessions in notebooks
spark = SparkSession.builder \
    .appName("My Application") \
    .getOrCreate()

# Production configuration — set these BEFORE creating the session;
# some configs (like executor memory) are immutable after JVM startup.
spark = SparkSession.builder \
    .appName("My Application") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", 200) \   # 200 is the default — tune to ~2-3x core count for small clusters
    .config("spark.executor.memory", "4g") \          # Heap memory per executor — set based on available node RAM
    .config("spark.driver.memory", "2g") \            # Driver needs enough RAM for collect() results and broadcast variables
    .config("spark.sql.adaptive.enabled", "true") \   # AQE dynamically optimizes at runtime — strongly recommended for Spark 3.x
    .enableHiveSupport() \                            # Required only for Hive metastore access — adds startup overhead if unused
    .getOrCreate()

# SparkContext is the low-level RDD API — still needed for broadcast variables,
# accumulators, and RDD operations not available through DataFrame API
sc = spark.sparkContext

# Check configuration
print(spark.conf.get("spark.sql.shuffle.partitions"))

# Always stop the session when done to release cluster resources and flush logs
spark.stop()
```

### 5.2 Common Configurations

```python
# Frequently used configurations
common_configs = {
    # Memory settings — executor memory is split between execution (shuffles, joins)
    # and storage (caches). memoryOverhead covers off-heap memory (Python processes, JVM overhead).
    "spark.executor.memory": "4g",
    "spark.driver.memory": "2g",
    "spark.executor.memoryOverhead": "512m",  # Increase for PySpark — Python workers use off-heap

    # Parallelism — these determine task count. Too few = underutilized cores.
    # Too many = excessive scheduling overhead and small tasks.
    "spark.executor.cores": "4",              # Cores per executor — 4-5 is typical sweet spot
    "spark.default.parallelism": "100",       # For RDD operations (not SQL)
    "spark.sql.shuffle.partitions": "200",    # For DataFrame/SQL shuffles — start with 2-3x total cores

    # Kryo is 10x faster and more compact than Java serialization — register
    # your classes with kryo.classesToRegister for best results
    "spark.serializer": "org.apache.spark.serializer.KryoSerializer",

    # AQE re-optimizes the query plan at runtime based on actual data statistics —
    # handles data skew and partition coalescing that static planning cannot predict
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.coalescePartitions.enabled": "true",   # Merges small post-shuffle partitions
    "spark.sql.adaptive.skewJoin.enabled": "true",             # Splits skewed partitions automatically

    # Cache settings
    "spark.storage.memoryFraction": "0.6",    # 60% of executor memory for caching — lower if you need more execution memory

    # Shuffle compression reduces network I/O at the cost of CPU — almost always
    # a net win because shuffles are typically network-bound, not CPU-bound
    "spark.shuffle.compress": "true",
}

# Apply configuration example
spark = SparkSession.builder \
    .config("spark.sql.shuffle.partitions", 100) \
    .config("spark.sql.adaptive.enabled", True) \
    .getOrCreate()
```

---

## 6. Basic Examples

### 6.1 Word Count

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Word Count") \
    .getOrCreate()

sc = spark.sparkContext

# textFile splits input by HDFS block boundaries — each block becomes a partition,
# enabling parallel reading across the cluster
text_rdd = sc.textFile("input.txt")

# Classic MapReduce pattern expressed as RDD transformations:
# flatMap → map → reduceByKey is the canonical word count pipeline.
word_counts = text_rdd \
    .flatMap(lambda line: line.split()) \     # 1 line → many words (1-to-N mapping)
    .map(lambda word: (word.lower(), 1)) \    # Normalize case to avoid "The" vs "the" as separate keys
    .reduceByKey(lambda a, b: a + b) \        # Local combine + shuffle — far more efficient than groupByKey
    .sortBy(lambda x: x[1], ascending=False)  # Global sort requires full shuffle — do last

# take(10) only scans enough partitions to return 10 results —
# avoid collect() which would pull the entire vocabulary to the Driver
for word, count in word_counts.take(10):
    print(f"{word}: {count}")

# Writes one part-NNNNN file per partition. Use coalesce(1) for a single file,
# but only if output is small — single-file writes cannot be parallelized.
word_counts.saveAsTextFile("output/word_counts")

spark.stop()
```

### 6.2 DataFrame Basics

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as _sum, avg

spark = SparkSession.builder.appName("DataFrame Example").getOrCreate()

# DataFrames are the preferred API over RDDs — they leverage the Catalyst optimizer
# for automatic query optimization (predicate pushdown, column pruning, etc.)
data = [
    ("Alice", "Engineering", 50000),
    ("Bob", "Engineering", 60000),
    ("Charlie", "Marketing", 45000),
    ("Diana", "Marketing", 55000),
]

# Schema inference from Python tuples — works for prototyping but use explicit
# StructType schemas in production for type safety and better Parquet performance
df = spark.createDataFrame(data, ["name", "department", "salary"])

# show() is an action — triggers execution but limits output (default 20 rows)
# unlike collect() which pulls everything to the Driver
df.show()
df.printSchema()

# Column-based filtering uses Catalyst to push predicates down to the data source
# when possible (e.g., Parquet row group filtering, JDBC WHERE clause pushdown)
df.filter(col("salary") > 50000).show()

# groupBy triggers a shuffle to co-locate rows with the same key — the most
# expensive operation in most Spark jobs. Catalyst may rewrite this as a
# partial aggregate + final aggregate to reduce shuffle volume.
df.groupBy("department") \
    .agg(
        _sum("salary").alias("total_salary"),
        avg("salary").alias("avg_salary")
    ) \
    .show()

# SQL and DataFrame APIs produce identical execution plans under the hood —
# choose whichever is more readable for the query at hand
df.createOrReplaceTempView("employees")
spark.sql("""
    SELECT department, AVG(salary) as avg_salary
    FROM employees
    GROUP BY department
""").show()

spark.stop()
```

---

## Practice Problems

### Problem 1: Basic RDD Operations
Find the sum of squares of even numbers from 1 to 100.

```python
# Solution
sc = spark.sparkContext
result = sc.parallelize(range(1, 101)) \
    .filter(lambda x: x % 2 == 0) \
    .map(lambda x: x ** 2) \
    .reduce(lambda a, b: a + b)
print(result)  # 171700
```

### Problem 2: Pair RDD
Aggregate log counts by error level from a log file.

```python
# Input: "2024-01-01 ERROR: Connection failed"
logs = sc.textFile("logs.txt")
error_counts = logs \
    .map(lambda line: line.split()[1].replace(":", "")) \
    .map(lambda level: (level, 1)) \
    .reduceByKey(lambda a, b: a + b) \
    .collect()
```

---

## Summary

| Concept | Description |
|---------|-------------|
| **Spark** | Unified engine for large-scale data processing |
| **RDD** | Basic distributed data structure |
| **Transformation** | Creates new RDD (Lazy) |
| **Action** | Returns result (Eager) |
| **Driver** | Main program execution node |
| **Executor** | Task execution worker |

---

## References

- [Apache Spark Documentation](https://spark.apache.org/docs/latest/)
- [PySpark API Reference](https://spark.apache.org/docs/latest/api/python/)
- [Learning Spark (O'Reilly)](https://www.oreilly.com/library/view/learning-spark-2nd/9781492050032/)
