[← Previous: 10. Kafka Streaming](10_Kafka_Streaming.md) | [Next: 12. dbt Transformation →](12_dbt_Transformation.md)

# Data Lake and Data Warehouse

## Learning Objectives

After completing this lesson, you will be able to:

1. Compare the architectural differences between Data Lakes, Data Warehouses, and Lakehouse systems, including Schema-on-Write vs. Schema-on-Read approaches.
2. Explain the role of dimensional modeling (star and snowflake schemas) in analytical query performance.
3. Evaluate trade-offs in selecting a storage architecture based on data volume, variety, and query patterns.
4. Design a Lakehouse architecture using open table formats such as Delta Lake or Apache Iceberg.
5. Implement ETL/ELT pipelines that move data across bronze, silver, and gold layers in a medallion architecture.
6. Analyze the cost and scalability implications of major cloud data warehouse and data lake solutions.

---

## Theory & Principles

The story of analytical storage is a back-and-forth between two extremes: the *warehouse* (rigid schema, expensive compute, ACID, SQL-only) and the *lake* (flexible files, cheap storage, no transactions, open compute). Each stage of the industry's evolution responded to a real limitation of the previous. Understanding *why* each architecture won, then was superseded, is more useful than memorizing the differences.

- **(A) The warehouse-lake-lakehouse evolution** — what each addressed, what it left unsolved.
- **(B) Schema-on-write vs schema-on-read** — the central trade-off with cost paid at different times.
- **(C) Table formats: Delta Lake, Iceberg, Hudi** — how ACID transactions exist on top of object storage.
- **(D) The medallion (bronze/silver/gold) discipline** — refinement levels and idempotent rebuild semantics.
- **(E) Storage economics** — why the cloud pricing model favors lakehouse architectures.

### A. The Warehouse-Lake-Lakehouse Evolution

Three architectural waves, each addressing the previous's pain.

#### A.1 Wave 1: data warehouse (1990s-2010s)

Teradata, Oracle, SQL Server, then Redshift, Snowflake, BigQuery. The pitch: a relational database tuned for analytics. Schema enforced at write, columnar storage, MPP execution, optimizer with statistics. Analysts run SQL, get fast results.

Limitations that drove the next wave:
- **Storage and compute coupled** (early generation). Adding storage required adding compute hardware. Cost scaled badly.
- **Schema rigidity.** Adding a column meant ALTER TABLE on petabytes; rejecting non-SQL data (images, free-text logs).
- **Vendor lock-in.** Data lived in proprietary formats; getting it out for ML or non-SQL processing was painful.

#### A.2 Wave 2: data lake (2010s)

Hadoop HDFS, then S3/GCS/ADLS. Just dump files (Parquet, JSON, Avro, CSV) into cheap object storage. Multiple compute engines (Spark, Hive, Presto, Python notebooks) read the same files. Schema flexible, storage cheap, no vendor lock-in.

What the lake won: storage cost (10-100x cheaper than warehouse), schema flexibility, multi-engine analytics, ML-friendliness.

What it lost:
- **No ACID transactions.** Concurrent writers corrupt each other. A reader during a write sees half-written data.
- **No schema enforcement.** Garbage accumulates ("data swamp").
- **No fine-grained updates / deletes.** Delete one row in a Parquet file requires rewriting the file.
- **No indexing or stats.** Every query is a full scan unless someone built indexes externally.

The "data swamp" became a real organizational problem: petabytes of data nobody can trust or query efficiently.

#### A.3 Wave 3: lakehouse (2020-)

The synthesis. Keep the lake's storage layout (files in S3) but add a *table format* on top that provides ACID transactions, schema evolution, time travel, and indexing. Three open implementations: **Delta Lake** (Databricks), **Apache Iceberg** (Netflix origin, now Apache), **Apache Hudi** (Uber origin, now Apache).

The pitch: warehouse semantics (ACID, SQL, performance) on lake economics (cheap storage, multi-engine, open formats).

### B. Schema-on-Write vs Schema-on-Read

The fundamental axis these architectures arrange themselves on.

#### B.1 Schema-on-write (warehouse default)

Validate and structure data at load time. Bad rows are rejected or quarantined. The warehouse table always conforms to its declared schema.

- **Cost paid at write time.** Producers must conform. New fields require migration.
- **Confidence at read time.** Every query sees clean, typed data. Analysts move fast.
- **Failure mode:** schema migrations are painful. A column rename in upstream source breaks the loader.

#### B.2 Schema-on-read (lake default)

Dump raw bytes; consumers parse and validate at read time. Adding a new field is free at write time.

- **Cost paid at read time.** Every consumer pays parsing/validation cost. Schema bugs surface late.
- **Flexibility at write time.** Producers move fast. New fields appear in JSON without migration.
- **Failure mode:** garbage accumulates. Different consumers parse fields differently. Trust erodes.

#### B.3 Lakehouse: schema-on-read at bronze, schema-on-write at silver/gold

The pragmatic synthesis. Raw tier (bronze) accepts anything; validation happens during the bronze→silver transformation. Silver and gold tiers are typed, validated, schema-enforced. The "data swamp" risk is contained to bronze; everything downstream is clean.

### C. Table Formats: Delta Lake, Iceberg, Hudi

The technology that makes lakehouse work. All three solve the same problem with the same general approach: a *transaction log* that records each commit, alongside the actual data files.

#### C.1 The transaction log pattern

```
/table_root/
  _delta_log/                ← transaction log (Delta Lake naming; Iceberg uses metadata/)
    000000.json              ← commit 0: "added file part-001.parquet"
    000001.json              ← commit 1: "added file part-002.parquet, removed part-001.parquet"
  part-001.parquet           ← actual data
  part-002.parquet
```

To read the table at the current state, a reader:
1. Lists the log files.
2. Replays them in order to compute the current set of "live" data files.
3. Reads only those files.

To write, a writer:
1. Stages new data files.
2. Atomically appends a new log entry recording the change.
3. Optimistic concurrency: if another writer's log entry won, retry.

This gives:
- **ACID writes.** The log entry is atomic; partial writes are invisible.
- **Snapshot reads.** A reader at time T sees a consistent snapshot, regardless of concurrent writers.
- **Time travel.** Read the table as of an old commit number — useful for debugging, audit, replay.
- **Schema evolution.** Schema is in the log; old data files have their own schema; reader merges.
- **Updates and deletes.** Implemented as "rewrite affected files + record old as removed in log".

#### C.2 Differences in practice

- **Delta Lake:** simplest, tightly tied to Spark/Databricks (good third-party support now). Default for Databricks customers.
- **Apache Iceberg:** designed for multi-engine from day one. Better support in Trino, Flink, Snowflake. Most "open" choice.
- **Apache Hudi:** best for streaming + frequent upserts. Has incremental processing primitives baked in.

The choice often comes down to ecosystem rather than raw capability. All three are converging.

### D. The Medallion Discipline

The standard organizing pattern for lakehouse pipelines.

#### D.1 The three tiers

- **Bronze:** raw ingested data, append-only, schema-on-read. The source of truth; never deleted. CDC feeds, raw API responses, dumped files. Bronze data is messy; that's its purpose.
- **Silver:** cleaned, deduplicated, joined, conformed to a schema. Type-checked, null-handled, business-key-enforced. Analyst-ready.
- **Gold:** business-aggregated, denormalized, optimized for specific consumption — dashboards, ML features, reports. Often pre-joined wide tables.

#### D.2 Idempotent rebuild

The medallion's superpower: each tier is **rebuildable from the prior tier**. If silver has a bug, fix the bronze→silver code, drop silver, rebuild from bronze. Bronze never changes (append-only); silver and gold are derived.

This is what makes the architecture survivable. Bug in production gold? Rebuild from silver in an hour. Bug in silver? Rebuild from bronze in a day. Bronze is your replay log.

#### D.3 The orchestration model

Each tier is built by orchestrated jobs (Airflow, Dagster, Prefect). The natural pattern: bronze ingestion runs continuously or hourly; silver and gold rebuild incrementally on a schedule. Failure of a downstream tier does not affect upstream — bronze keeps accumulating.

### E. Storage Economics

The economic reality that makes lakehouse possible.

- **Object storage:** $0.023/GB-month (S3). Effectively unlimited capacity. Multi-region replication built in.
- **Warehouse storage** (managed): typically 5-20x more expensive than raw object storage.
- **Compute:** elastic, per-second billing. Spin up 100 nodes for an hour, pay for an hour, shut down.

In this regime, **storage is free; compute is the bill**. The optimal architecture: store everything raw (bronze) at lake prices, transform on-demand with elastic compute, materialize only the silver/gold tiers most queries hit.

The old warehouse model where storage was the dominant cost is obsolete. Lakehouse architectures align with how cloud pricing actually works.

### From Theory to the Practice Below

Each section that follows operationalizes one piece of this framework:

- §1 (Data Lake) is §A.2 — the lake's promise and pitfalls.
- §2 (Data Warehouse) is §A.1 — the warehouse's strengths.
- §3 (Lakehouse) is §A.3 + §C — the synthesis and the table formats that enable it.
- §4 (Storage Format Comparison) is the practical Parquet/Delta/Iceberg/Hudi choice.
- §5 (Cloud Solutions) maps §A waves to specific products (Snowflake, BigQuery, Databricks, etc.).
- §6 (Architecture Patterns) is §D — the medallion in practice.

---

## Overview

Data storage architecture is central to an organization's data strategy. Understanding the characteristics and use cases of Data Lakes, Data Warehouses, and the Lakehouse architecture that combines both.

---

## 1. Data Warehouse

### 1.1 Concept

```
┌────────────────────────────────────────────────────────────────┐
│                    Data Warehouse                              │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   Characteristics:                                             │
│   - Structured data (schema definition required)               │
│   - Schema-on-Write (apply schema on write)                    │
│   - Optimized for analytics (OLAP)                             │
│   - SQL-based queries                                          │
│                                                                │
│   ┌──────────────────────────────────────────────────────┐    │
│   │                    Data Warehouse                     │    │
│   │   ┌─────────────────────────────────────────────────┐│    │
│   │   │  Dim Tables    │    Fact Tables                 ││    │
│   │   │  ┌──────────┐  │  ┌──────────┐                 ││    │
│   │   │  │dim_date  │  │  │fact_sales│                 ││    │
│   │   │  │dim_product│  │  │fact_orders│                ││    │
│   │   │  │dim_customer│ │                               ││    │
│   │   │  └──────────┘  │  └──────────┘                 ││    │
│   │   └─────────────────────────────────────────────────┘│    │
│   └──────────────────────────────────────────────────────┘    │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 1.2 Major Solutions

| Solution | Type | Features |
|----------|------|----------|
| **Snowflake** | Cloud | Separated storage/compute, auto-scaling |
| **BigQuery** | Cloud (GCP) | Serverless, petabyte scale |
| **Redshift** | Cloud (AWS) | Columnar, MPP architecture |
| **Synapse** | Cloud (Azure) | Unified analytics platform |
| **PostgreSQL** | On-premise | Small-scale, open source |

### 1.3 Data Warehouse SQL Examples

```sql
-- Snowflake/BigQuery style analytical queries

-- Monthly sales trend
-- Why a star-schema join? Joining facts to dimensions enables slicing metrics
-- by any dimension attribute without denormalizing the entire dataset.
SELECT
    d.year,
    d.month,
    d.month_name,
    SUM(f.sales_amount) AS total_sales,
    COUNT(DISTINCT f.customer_sk) AS unique_customers,
    AVG(f.sales_amount) AS avg_order_value,
    -- NULLIF prevents division-by-zero when the prior month has no sales
    -- (e.g., first month in the dataset or seasonal gaps).
    -- LAG window function compares sequential months without a self-join,
    -- which is far more efficient on columnar warehouses.
    (SUM(f.sales_amount) - LAG(SUM(f.sales_amount)) OVER (ORDER BY d.year, d.month))
        / NULLIF(LAG(SUM(f.sales_amount)) OVER (ORDER BY d.year, d.month), 0) * 100
        AS mom_growth_pct
FROM fact_sales f
JOIN dim_date d ON f.date_sk = d.date_sk
WHERE d.year >= 2023
GROUP BY d.year, d.month, d.month_name
ORDER BY d.year, d.month;


-- Customer LTV by segment
-- CTE isolates per-customer aggregation so the outer query can compute
-- segment-level statistics cleanly — avoids nested subqueries and
-- makes the logic testable independently.
WITH customer_metrics AS (
    SELECT
        c.customer_sk,
        c.customer_segment,
        MIN(d.full_date) AS first_purchase_date,
        MAX(d.full_date) AS last_purchase_date,
        COUNT(DISTINCT f.order_id) AS total_orders,
        SUM(f.sales_amount) AS total_revenue
    FROM fact_sales f
    JOIN dim_customer c ON f.customer_sk = c.customer_sk
    JOIN dim_date d ON f.date_sk = d.date_sk
    GROUP BY c.customer_sk, c.customer_segment
)
SELECT
    customer_segment,
    COUNT(*) AS customer_count,
    AVG(total_orders) AS avg_orders,
    AVG(total_revenue) AS avg_ltv,
    -- Median is more robust than AVG for skewed revenue distributions;
    -- a few whale customers can inflate AVG dramatically.
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_revenue) AS median_ltv
FROM customer_metrics
GROUP BY customer_segment
ORDER BY avg_ltv DESC;
```

---

## 2. Data Lake

### 2.1 Concept

```
┌────────────────────────────────────────────────────────────────┐
│                      Data Lake                                 │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   Characteristics:                                             │
│   - All types of data (structured, semi-structured, unstructured)│
│   - Schema-on-Read (apply schema on read)                      │
│   - Raw data preservation                                      │
│   - Low-cost storage                                           │
│                                                                │
│   ┌──────────────────────────────────────────────────────┐    │
│   │                     Data Lake                         │    │
│   │  ┌────────────────────────────────────────────────┐  │    │
│   │  │  Raw Zone (Bronze)                              │  │    │
│   │  │  - Raw data (JSON, CSV, Logs, Images)           │  │    │
│   │  └────────────────────────────────────────────────┘  │    │
│   │                         ↓                             │    │
│   │  ┌────────────────────────────────────────────────┐  │    │
│   │  │  Processed Zone (Silver)                        │  │    │
│   │  │  - Cleaned data (Parquet, Delta)                │  │    │
│   │  └────────────────────────────────────────────────┘  │    │
│   │                         ↓                             │    │
│   │  ┌────────────────────────────────────────────────┐  │    │
│   │  │  Curated Zone (Gold)                            │  │    │
│   │  │  - Analytics/ML ready data                      │  │    │
│   │  └────────────────────────────────────────────────┘  │    │
│   └──────────────────────────────────────────────────────┘    │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 2.2 Major Storage

| Storage | Cloud | Features |
|---------|-------|----------|
| **S3** | AWS | Object storage, high durability |
| **GCS** | GCP | Google Cloud Storage |
| **ADLS** | Azure | Azure Data Lake Storage |
| **HDFS** | On-premise | Hadoop Distributed File System |

### 2.3 Data Lake File Structure

```
s3://my-data-lake/
├── raw/                          # Bronze layer
│   ├── orders/
│   │   ├── year=2024/
│   │   │   ├── month=01/
│   │   │   │   ├── day=15/
│   │   │   │   │   ├── orders_20240115_001.json
│   │   │   │   │   └── orders_20240115_002.json
│   ├── customers/
│   │   └── snapshot_20240115.csv
│   └── logs/
│       └── app_logs_20240115.log
│
├── processed/                    # Silver layer
│   ├── orders/
│   │   └── year=2024/
│   │       └── month=01/
│   │           └── part-00000.parquet
│   └── customers/
│       └── part-00000.parquet
│
└── curated/                      # Gold layer
    ├── fact_sales/
    │   └── year=2024/
    │       └── month=01/
    └── dim_customers/
        └── current/
```

```python
# Processing Data Lake layers with PySpark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

spark = SparkSession.builder \
    .appName("DataLakeProcessing") \
    .getOrCreate()

# Raw → Processed (Bronze → Silver)
# Bronze holds immutable raw data for auditability and reprocessing.
# Silver applies cleaning and deduplication so downstream consumers
# don't repeat these steps — a single source of clean truth.
def process_raw_orders():
    # Read raw JSON — schema-on-read means we accept any shape here
    # and apply structure only during processing (Data Lake philosophy)
    raw_df = spark.read.json("s3://my-data-lake/raw/orders/")

    # Filter nulls and deduplicate: raw sources often contain duplicates
    # from at-least-once delivery; dedup on business key prevents
    # inflated counts in downstream aggregations
    processed_df = raw_df \
        .filter(col("order_id").isNotNull()) \
        .withColumn("processed_at", current_timestamp()) \
        .dropDuplicates(["order_id"])

    # Parquet provides columnar compression (10-30x vs JSON) and
    # predicate pushdown for efficient analytical queries.
    # Partitioning by year/month enables partition pruning — queries
    # that filter on date only scan relevant directories.
    processed_df.write \
        .mode("overwrite") \
        .partitionBy("year", "month") \
        .parquet("s3://my-data-lake/processed/orders/")


# Processed → Curated (Silver → Gold)
# Gold layer contains business-ready aggregates and dimensional models
# optimized for fast queries — BI tools read from here directly.
def create_fact_sales():
    orders = spark.read.parquet("s3://my-data-lake/processed/orders/")
    customers = spark.read.parquet("s3://my-data-lake/processed/customers/")

    # Join at the Gold layer to pre-compute the dimensional model;
    # doing this once here avoids repeated expensive joins in every
    # downstream query or dashboard
    fact_sales = orders \
        .join(customers, "customer_id") \
        .select(
            col("order_id"),
            col("customer_sk"),
            col("order_date"),
            col("amount").alias("sales_amount")
        )

    fact_sales.write \
        .mode("overwrite") \
        .partitionBy("year", "month") \
        .parquet("s3://my-data-lake/curated/fact_sales/")
```

---

## 3. Data Warehouse vs Data Lake

### 3.1 Comparison

| Characteristic | Data Warehouse | Data Lake |
|----------------|----------------|-----------|
| **Data Type** | Structured | All types |
| **Schema** | Schema-on-Write | Schema-on-Read |
| **Users** | Business analysts | Data scientists, engineers |
| **Processing** | OLAP | Batch, streaming, ML |
| **Cost** | High | Low |
| **Query Performance** | Optimized | Variable |
| **Data Quality** | High (cleaned) | Variable |

### 3.2 Selection Criteria

```python
def choose_architecture(requirements: dict) -> str:
    """Architecture selection guide"""

    # Weighted scoring by counting matching factors.
    # In practice, factors should be weighted differently (e.g., governance
    # compliance may be non-negotiable), but equal weights keep this
    # heuristic simple and easy to extend.
    warehouse_factors = [
        requirements.get('structured_data_only', False),
        requirements.get('sql_analytics_primary', False),
        requirements.get('strict_governance', False),
        requirements.get('fast_query_response', False),
    ]

    # Lake has more factors because it serves a wider range of use cases;
    # this naturally biases the score toward Lake when requirements are mixed,
    # reflecting the industry trend of starting with a Lake/Lakehouse.
    lake_factors = [
        requirements.get('unstructured_data', False),
        requirements.get('ml_workloads', False),
        requirements.get('raw_data_preservation', False),
        requirements.get('cost_sensitive', False),
        requirements.get('schema_flexibility', False),
    ]

    if sum(warehouse_factors) > sum(lake_factors):
        return "Data Warehouse recommended"
    elif sum(lake_factors) > sum(warehouse_factors):
        return "Data Lake recommended"
    else:
        # Tie-breaking toward Lakehouse: when neither side dominates,
        # Lakehouse gives you both SQL performance and raw-data flexibility
        return "Consider Lakehouse"
```

---

## 4. Lakehouse

### 4.1 Concept

Lakehouse is an architecture that combines the flexibility of Data Lakes with the performance and management capabilities of Data Warehouses.

```
┌────────────────────────────────────────────────────────────────┐
│                      Lakehouse Architecture                    │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   ┌────────────────────────────────────────────────────────┐  │
│   │                   Applications                          │  │
│   │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │  │
│   │  │    BI    │ │    ML    │ │  SQL     │ │ Streaming│  │  │
│   │  └──────────┘ └──────────┘ └──────────┘ └──────────┘  │  │
│   └────────────────────────────────────────────────────────┘  │
│                              ↓                                 │
│   ┌────────────────────────────────────────────────────────┐  │
│   │                  Query Engine                           │  │
│   │        (Spark, Presto, Trino, Dremio)                  │  │
│   └────────────────────────────────────────────────────────┘  │
│                              ↓                                 │
│   ┌────────────────────────────────────────────────────────┐  │
│   │              Lakehouse Format Layer                     │  │
│   │     ┌──────────────────────────────────────────────┐   │  │
│   │     │  ACID Transactions │ Schema Enforcement      │   │  │
│   │     │  Time Travel       │ Unified Batch/Streaming │   │  │
│   │     └──────────────────────────────────────────────┘   │  │
│   │           Delta Lake / Apache Iceberg / Apache Hudi    │  │
│   └────────────────────────────────────────────────────────┘  │
│                              ↓                                 │
│   ┌────────────────────────────────────────────────────────┐  │
│   │              Object Storage (Data Lake)                 │  │
│   │                  S3 / GCS / ADLS / HDFS                 │  │
│   └────────────────────────────────────────────────────────┘  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 4.2 Key Features

| Feature | Description |
|---------|-------------|
| **ACID Transactions** | Data integrity guarantee |
| **Schema Evolution** | Schema change support |
| **Time Travel** | Query historical data versions |
| **Upsert/Merge** | Efficient data updates |
| **Unified Processing** | Batch + streaming in single table |

---

## 5. Delta Lake

### 5.1 Delta Lake Basics

```python
from pyspark.sql import SparkSession
from delta import *

# Delta Lake configuration
# Both extensions are required: DeltaSparkSessionExtension adds Delta-specific
# SQL commands (MERGE, OPTIMIZE), while DeltaCatalog enables Spark to resolve
# Delta tables by name rather than only by path.
spark = SparkSession.builder \
    .appName("DeltaLake") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

# Create Delta table
# Writing as Delta (vs raw Parquet) adds a _delta_log transaction log that
# provides ACID transactions, schema enforcement, and time travel — all
# impossible with plain Parquet files.
df = spark.createDataFrame([
    (1, "Alice", 100),
    (2, "Bob", 200),
], ["id", "name", "amount"])

df.write.format("delta").save("/data/delta/users")

# Read
delta_df = spark.read.format("delta").load("/data/delta/users")

# Registering a SQL table on top of the Delta path enables BI tools and
# SQL analysts to query the data without knowing the physical file location.
spark.sql("CREATE TABLE users USING DELTA LOCATION '/data/delta/users'")
spark.sql("SELECT * FROM users").show()
```

### 5.2 Delta Lake Advanced Features

```python
from delta.tables import DeltaTable

# MERGE (Upsert)
# MERGE is the key operation that distinguishes Lakehouse from plain Data Lakes.
# Without it, you'd need to read-all → filter → union → write-all for any update,
# which is expensive and not atomic.
delta_table = DeltaTable.forPath(spark, "/data/delta/users")

new_data = spark.createDataFrame([
    (1, "Alice Updated", 150),  # Update
    (3, "Charlie", 300),        # Insert
], ["id", "name", "amount"])

# The merge condition defines the business key for matching.
# whenMatchedUpdate handles existing records (SCD Type 1 overwrites),
# while whenNotMatchedInsert handles new records — both in a single atomic pass.
delta_table.alias("target").merge(
    new_data.alias("source"),
    "target.id = source.id"
).whenMatchedUpdate(set={
    "name": "source.name",
    "amount": "source.amount"
}).whenNotMatchedInsert(values={
    "id": "source.id",
    "name": "source.name",
    "amount": "source.amount"
}).execute()


# Time Travel (query historical versions)
# Time travel is invaluable for debugging data issues — you can compare
# the current state with a prior version to see exactly what changed.
# By version number
df_v0 = spark.read.format("delta") \
    .option("versionAsOf", 0) \
    .load("/data/delta/users")

# By timestamp — useful when you know "the data was correct yesterday"
# but don't know the exact version number
df_yesterday = spark.read.format("delta") \
    .option("timestampAsOf", "2024-01-14") \
    .load("/data/delta/users")


# Check history
delta_table.history().show()


# Vacuum (cleanup old files)
# 168 hours (7 days) balances storage cost against the need for time travel.
# Shorter retention saves storage but loses the ability to query or rollback
# to older versions. Never vacuum below the default 7-day threshold without
# confirming no concurrent readers depend on old files.
delta_table.vacuum(retentionHours=168)  # 7 days retention


# Schema evolution — mergeSchema=true allows adding new columns from
# incoming data without breaking existing readers.
# Without this, writes with new columns would fail with a schema mismatch error.
spark.read.format("delta") \
    .option("mergeSchema", "true") \
    .load("/data/delta/users")


# Z-Order optimization (query performance)
# Z-ordering co-locates related data in the same files based on the specified
# columns, dramatically improving query pruning for filters on those columns.
# Choose columns that appear most frequently in WHERE clauses.
delta_table.optimize().executeZOrderBy("date", "customer_id")
```

---

## 6. Apache Iceberg

### 6.1 Iceberg Basics

```python
from pyspark.sql import SparkSession

# Iceberg uses a catalog-centric design: all table metadata lives in the catalog
# (here Hive Metastore), which makes tables engine-independent — the same table
# can be queried from Spark, Trino, Flink, or Dremio without data duplication.
spark = SparkSession.builder \
    .appName("Iceberg") \
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
    .config("spark.sql.catalog.iceberg", "org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.iceberg.type", "hive") \
    .config("spark.sql.catalog.iceberg.uri", "thrift://localhost:9083") \
    .getOrCreate()

# Create Iceberg table
# bucket(16, id) is a "hidden partition": Iceberg hashes the id into 16 buckets
# for even data distribution. Unlike Hive-style partitioning, the user doesn't
# need to know the partitioning scheme when querying — Iceberg rewrites the
# filter predicate automatically (partition evolution).
spark.sql("""
    CREATE TABLE iceberg.db.users (
        id INT,
        name STRING,
        amount DECIMAL(10, 2)
    ) USING ICEBERG
    PARTITIONED BY (bucket(16, id))
""")

# Insert data
spark.sql("""
    INSERT INTO iceberg.db.users VALUES
    (1, 'Alice', 100.00),
    (2, 'Bob', 200.00)
""")

# Time Travel — Iceberg uses a snapshot-based model (vs Delta's log-based),
# storing a manifest list per snapshot. This makes multi-engine time travel
# possible since any engine can read the snapshot metadata directly.
spark.sql("SELECT * FROM iceberg.db.users VERSION AS OF 1").show()
spark.sql("SELECT * FROM iceberg.db.users TIMESTAMP AS OF '2024-01-15'").show()

# Snapshots metadata table exposes the full version history including
# operation type, added/deleted files, and summary statistics
spark.sql("SELECT * FROM iceberg.db.users.snapshots").show()
```

### 6.2 Delta Lake vs Iceberg Comparison

| Feature | Delta Lake | Iceberg |
|---------|------------|---------|
| **Developer** | Databricks | Netflix → Apache |
| **Compatibility** | Spark-centric | Engine-independent |
| **Metadata** | Transaction log | Snapshot-based |
| **Partition Evolution** | Limited | Strong support |
| **Hidden Partitioning** | Not supported | Supported |
| **Community** | Databricks ecosystem | Multiple vendors |

---

## 7. Modern Data Stack

### 7.1 Architecture Pattern

```
┌─────────────────────────────────────────────────────────────────┐
│                   Modern Data Stack                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Data Sources                                                  │
│   ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐                  │
│   │ SaaS   │ │Database│ │  API   │ │  IoT   │                  │
│   └────┬───┘ └───┬────┘ └───┬────┘ └───┬────┘                  │
│        └─────────┴──────────┴──────────┘                        │
│                         ↓                                       │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │              Ingestion (EL)                              │  │
│   │        Fivetran / Airbyte / Stitch                       │  │
│   └─────────────────────────────────────────────────────────┘  │
│                         ↓                                       │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │           Cloud Data Warehouse / Lakehouse              │  │
│   │        Snowflake / BigQuery / Databricks                │  │
│   └─────────────────────────────────────────────────────────┘  │
│                         ↓                                       │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │              Transformation (T)                          │  │
│   │                      dbt                                 │  │
│   └─────────────────────────────────────────────────────────┘  │
│                         ↓                                       │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                 BI / Analytics                           │  │
│   │        Looker / Tableau / Metabase / Mode               │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Practice Problems

### Problem 1: Architecture Selection
Choose and justify an architecture for the following requirements:
- 10TB daily log data
- Used for ML model training
- Raw data retention for 5 years

### Problem 2: Delta Lake Implementation
Implement SCD Type 2 for customer data using Delta Lake MERGE.

---

## Summary

| Architecture | Features | Use Cases |
|--------------|----------|-----------|
| **Data Warehouse** | Structured, SQL optimized | BI, reporting |
| **Data Lake** | All data types, low cost | ML, raw preservation |
| **Lakehouse** | Lake + Warehouse benefits | Unified analytics |

---

## References

- [Delta Lake Documentation](https://docs.delta.io/)
- [Apache Iceberg Documentation](https://iceberg.apache.org/)
- [Databricks Lakehouse](https://www.databricks.com/product/data-lakehouse)

---

[← Previous: 10. Kafka Streaming](10_Kafka_Streaming.md) | [Next: 12. dbt Transformation →](12_dbt_Transformation.md)
