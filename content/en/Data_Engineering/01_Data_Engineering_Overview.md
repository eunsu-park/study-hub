[← Previous: Overview](00_Overview.md) | [Next: 2. Data Modeling Basics →](02_Data_Modeling_Basics.md)

# Data Engineering Overview

## Learning Objectives

After completing this lesson, you will be able to:

1. Define data engineering and explain the core responsibilities of a data engineer compared to data scientists and analysts
2. Describe the components of a data pipeline and implement a basic ETL pipeline in Python
3. Compare batch processing and stream processing, and select the appropriate approach for a given use case
4. Explain major data architecture patterns including Data Warehouse, Data Lake, Lambda, and Kappa architectures
5. Identify the key tools in the data engineering ecosystem and map them to their cloud service equivalents
6. Apply pipeline design best practices such as idempotency, atomicity, and error handling with retry logic

---

## Overview

Data Engineering is the field of designing and building systems that collect, store, process, and deliver organizational data. Data engineers build data pipelines that transform raw data into analyzable formats.

---

## 1. Role of a Data Engineer

### 1.1 Core Responsibilities

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Engineer Role                        │
├─────────────────────────────────────────────────────────────┤
│  1. Data Ingestion                                          │
│     - Extract data from various sources                     │
│     - API, databases, files, streaming                      │
│                                                             │
│  2. Data Storage                                            │
│     - Design Data Lake, Data Warehouse                      │
│     - Schema design and optimization                        │
│                                                             │
│  3. Data Transformation                                     │
│     - Build ETL/ELT pipelines                              │
│     - Ensure data quality                                   │
│                                                             │
│  4. Data Serving                                            │
│     - Provide data to analysts/scientists                   │
│     - Integrate with BI tools, API, dashboards             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Data Engineer vs Data Scientist vs Data Analyst

| Role | Main Responsibilities | Required Skills |
|------|----------|----------|
| **Data Engineer** | Pipeline construction, infrastructure management | Python, SQL, Spark, Airflow, Kafka |
| **Data Scientist** | Model development, predictive analytics | Python, ML/DL, statistics, mathematics |
| **Data Analyst** | Business insight extraction | SQL, BI tools, visualization, statistics |

### 1.3 Essential Skills for Data Engineers

```python
# Example data engineer tech stack
# Grouped by function rather than vendor — this mirrors how teams typically
# divide ownership (e.g., a "platform" team owns infra, a "data" team owns
# orchestration + processing).  Knowing the *category* helps you identify
# transferable skills when switching between cloud providers.
tech_stack = {
    "programming": ["Python", "SQL", "Scala", "Java"],
    "databases": ["PostgreSQL", "MySQL", "MongoDB", "Redis"],
    "big_data": ["Spark", "Hadoop", "Flink", "Hive"],
    "orchestration": ["Airflow", "Prefect", "Dagster"],
    "streaming": ["Kafka", "Kinesis", "Pub/Sub"],
    "cloud": ["AWS", "GCP", "Azure"],
    "infrastructure": ["Docker", "Kubernetes", "Terraform"],
    "storage": ["S3", "GCS", "HDFS", "Delta Lake"]
}
```

---

## 2. Data Pipeline Concepts

### Theory: Pipeline as a Directed Graph of Stages

A data pipeline is a DAG (directed acyclic graph) of stages, each consuming inputs and producing outputs. The canonical decomposition is *extract* (pull from a source), *transform* (clean, join, aggregate), and *load* (write to a sink) — but the deeper truth is that every stage in the graph has the same shape: read inputs, do work, write outputs.

#### A.1 Why a DAG, not a linear script

A linear script (`extract.py && transform.py && load.py`) breaks the moment you have:

- More than one source feeding the same transform.
- A transform that fans out to multiple sinks.
- A failure in stage 5 that needs to rerun without redoing stages 1-4.

A DAG makes the dependency explicit, lets the scheduler skip already-completed stages, and allows parallel execution of independent branches. This is why every modern orchestrator (Airflow, Prefect, Dagster) models pipelines as DAGs — see Lessons 4-6.

#### A.2 Idempotency: the single most important pipeline discipline

A pipeline stage is *idempotent* when running it twice with the same input produces the same output as running it once. This matters because failures are inevitable — a worker crashes, a network blip kills the connection, the orchestrator retries. If the stage is idempotent, retry is safe; if not, retry corrupts.

Three patterns make a stage idempotent:

1. **Deterministic partitioning by date.** "Load yesterday's data into the `orders_2024_03_15` partition." Rerunning overwrites the same partition; downstream sees the same final state.
2. **Upsert (merge) instead of insert.** Use `INSERT ... ON CONFLICT DO UPDATE` (PostgreSQL) or `MERGE` (data warehouse SQL). The second run finds the row already there and updates rather than duplicating.
3. **Content-addressed outputs.** Hash the input; if a file with that hash already exists, skip writing. Common in ML feature pipelines.

The pattern to avoid: `INSERT INTO sales SELECT * FROM raw_sales WHERE date = today()`. If this runs twice, you get every row twice.

#### A.3 Atomicity at the stage boundary

A stage either fully completes or leaves no partial state behind. Concretely: write to a temporary location, then atomically rename to the final location only when the write succeeds. On HDFS/S3 this is `_SUCCESS` markers; in SQL warehouses it is transactions; in Delta Lake / Iceberg it is the table commit log. A reader that sees a stage's output sees either *all* of it or *none* of it — never a half-written file.

### 2.1 What is a Pipeline?

A data pipeline is a series of processing steps that move data from source to destination.

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Source  │ → │  Extract │ → │Transform │ → │   Load   │
│          │    │          │    │          │    │          │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
     ↓               ↓               ↓               ↓
  Database        Raw Data      Cleaned Data    Warehouse
  API, Files      Staging       Processed       Analytics
```

### 2.2 Pipeline Components

```python
# Simple pipeline example
from datetime import datetime
import pandas as pd

class DataPipeline:
    """Basic data pipeline class"""

    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None

    def extract(self, source: str) -> pd.DataFrame:
        """Data extraction step"""
        print(f"[{datetime.now()}] Extracting from {source}")
        # In practice, extract data from DB, API, files, etc.
        data = pd.read_csv(source)
        return data

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Data transformation step"""
        print(f"[{datetime.now()}] Transforming data")
        # Drop rather than impute: this simple pipeline assumes upstream
        # systems handle partial data; dropping keeps the logic idempotent
        # since we don't need to guess fill values.
        df = df.dropna()
        # Stamp processing time so downstream consumers can detect stale data
        # and distinguish between two runs on the same source file.
        df['processed_at'] = datetime.now()
        return df

    def load(self, df: pd.DataFrame, destination: str):
        """Data loading step"""
        print(f"[{datetime.now()}] Loading to {destination}")
        # Parquet chosen over CSV: columnar format compresses better and
        # preserves dtypes, avoiding type-inference issues on reload.
        df.to_parquet(destination, index=False)

    def run(self, source: str, destination: str):
        """Execute entire pipeline"""
        self.start_time = datetime.now()
        print(f"Pipeline '{self.name}' started")

        # Sequential E→T→L: each step receives the previous step's output,
        # making it easy to add validation or checkpoints between stages.
        raw_data = self.extract(source)
        transformed_data = self.transform(raw_data)
        self.load(transformed_data, destination)

        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).seconds
        print(f"Pipeline completed in {duration} seconds")

# Execute pipeline
if __name__ == "__main__":
    pipeline = DataPipeline("daily_sales")
    pipeline.run("sales_raw.csv", "sales_processed.parquet")
```

### 2.3 Pipeline Types

| Type | Description | Use Cases |
|------|------|----------|
| **Batch** | Process large volumes of data at scheduled times | Daily reports, monthly aggregations |
| **Streaming** | Real-time data processing | Real-time dashboards, anomaly detection |
| **Micro-batch** | Small batches at short intervals | Near real-time analytics (5-15 min) |
| **Event-driven** | Process on specific event occurrence | Trigger-based processing |

---

## 3. Batch Processing vs Stream Processing

### Theory: Batch vs Streaming Trade-offs

The choice is not "stream is modern, batch is old" — it is a deliberate trade-off triangle.

| Property | Batch | Streaming |
|----------|-------|-----------|
| Latency | minutes to hours | milliseconds to seconds |
| Throughput per dollar | very high (compute amortized) | lower (always-on workers) |
| Operational complexity | low (cron + DAG) | high (state, watermarks, exactly-once) |
| Reprocessing | trivial (rerun the job) | complex (replay from log, manage state) |
| Best for | analytics, ML training, reports | alerts, dashboards, real-time features |

The pragmatic rule: use batch by default; introduce streaming only when latency requirements truly demand it. Streaming compounds operational burden — watermarks (event-time vs processing-time), state management, late-arriving data — that does not exist in batch. See Lessons 10, 16, 17 for streaming-specific theory (event time, exactly-once, windowing).

### 3.1 Batch Processing

```python
# Batch processing example: Daily sales aggregation
from datetime import datetime, timedelta
import pandas as pd

def daily_sales_batch():
    """Daily sales batch processing"""

    # Process *yesterday's* data: the full day must be complete before
    # aggregation, otherwise totals would be partial and misleading.
    yesterday = datetime.now() - timedelta(days=1)
    date_str = yesterday.strftime('%Y-%m-%d')

    # Pre-aggregate at the source DB level to minimize data transfer —
    # only aggregated rows cross the network rather than raw transactions.
    query = f"""
    SELECT
        product_id,
        SUM(quantity) as total_quantity,
        SUM(amount) as total_amount
    FROM sales
    WHERE DATE(created_at) = '{date_str}'
    GROUP BY product_id
    """

    # Date-partitioned output file enables idempotent re-runs:
    # re-running the same date simply overwrites the same file.
    print(f"Processing batch for {date_str}")
    # df = execute_query(query)
    # df.to_parquet(f"sales_summary_{date_str}.parquet")

    return {"status": "success", "date": date_str}

# Batch processing characteristics
batch_characteristics = {
    "latency": "High (minutes to hours)",
    "throughput": "High (efficient for large volumes)",
    "use_cases": ["Daily reports", "Weekly aggregations", "Data migration"],
    "tools": ["Spark", "Airflow", "dbt", "AWS Glue"]
}
```

### 3.2 Stream Processing

```python
# Stream processing example: Real-time event processing
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Any
import json

@dataclass
class Event:
    """Streaming event"""
    event_type: str
    data: dict
    timestamp: datetime

class StreamProcessor:
    """Simple stream processor"""

    def __init__(self):
        # Multiple handlers per event type: the observer pattern lets us
        # add new reactions (alerting, logging, metrics) without modifying
        # existing handler code — critical in streaming where downtime is costly.
        self.handlers: dict[str, list[Callable]] = {}

    def register_handler(self, event_type: str, handler: Callable):
        """Register event handler"""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)

    def process(self, event: Event):
        """Process event"""
        # Fan-out: every registered handler runs on the same event, enabling
        # independent processing paths (e.g., log AND alert simultaneously).
        handlers = self.handlers.get(event.event_type, [])
        for handler in handlers:
            handler(event)

    def consume(self, stream):
        """Consume events from stream (simulation)"""
        for message in stream:
            # Timestamp assigned at consume-time, not source-time — in production
            # you'd use the source event timestamp to avoid clock-skew issues.
            event = Event(
                event_type=message['type'],
                data=message['data'],
                timestamp=datetime.now()
            )
            self.process(event)

# Handler examples
def log_handler(event: Event):
    """Event logging"""
    print(f"[{event.timestamp}] {event.event_type}: {event.data}")

def alert_handler(event: Event):
    """Anomaly detection alert"""
    if event.data.get('amount', 0) > 10000:
        print(f"ALERT: High value transaction detected!")

# Streaming characteristics
streaming_characteristics = {
    "latency": "Low (milliseconds to seconds)",
    "throughput": "Medium (record-level)",
    "use_cases": ["Real-time dashboards", "Anomaly detection", "Notifications"],
    "tools": ["Kafka", "Flink", "Spark Streaming", "Kinesis"]
}
```

### 3.3 Batch vs Streaming Comparison

| Characteristic | Batch Processing | Stream Processing |
|------|----------|--------------|
| **Latency** | Minutes to hours | Milliseconds to seconds |
| **Data Throughput** | Large volumes | Small/continuous |
| **Complexity** | Relatively simple | Relatively complex |
| **Reprocessing** | Easy | Difficult |
| **Cost** | Lower | Higher |
| **Use Cases** | Reports, aggregations | Real-time analytics, alerts |

---

## 4. Data Architecture Patterns

### Theory: Storage Tiers: Warehouse, Lake, Lakehouse

Three storage architectures dominate analytical workloads. They differ in *when* schema is enforced and *what* compute can run against them.

#### B.1 Data warehouse (schema-on-write)

A relational database tuned for analytics: columnar storage, MPP (massively parallel processing) execution, optimizer that pushes predicates down. Examples: Snowflake, BigQuery, Redshift.

- **Schema enforced at write.** Loading data requires it to fit a defined table schema; bad rows get rejected or quarantined.
- **SQL is the only interface.** Great for analysts; clumsy for unstructured data (images, logs, JSON).
- **Storage and compute coupled (historically), decoupled (modern).** Snowflake/BigQuery separate storage from compute, so you can scale compute independently.

#### B.2 Data lake (schema-on-read)

A blob store (S3, GCS, ADLS) holding raw files: Parquet, JSON, CSV, images, anything. Examples: AWS S3 + Glue catalog, Databricks lake.

- **No schema at write.** You dump bytes; consumers parse on read.
- **Any compute can read.** Spark, Presto/Trino, Python notebooks, ML training pipelines.
- **Cheap storage; expensive analytics.** Without indexing or table format, scanning is full-file.

The lake's flexibility comes at a cost: without enforced schema, garbage accumulates ("data swamp"). Without ACID, concurrent writers corrupt each other.

#### B.3 Lakehouse (schema-on-read with table format)

The synthesis: keep data in cheap blob storage but add a *table format* layer (Delta Lake, Apache Iceberg, Apache Hudi) that provides ACID transactions, schema evolution, and time travel. Examples: Databricks Lakehouse, Snowflake Iceberg tables.

- **ACID on top of object storage.** A transaction log file (`_delta_log/`) records each commit; readers see consistent snapshots.
- **Schema enforced at write** (after you turn it on), but you can also dump raw files and refine later.
- **Both SQL and ML compute.** Spark/Trino read the same tables that BI tools query.

#### B.4 Medallion architecture

The lakehouse-era convention for organizing refinement levels:

- **Bronze:** raw ingested data, append-only, schema-on-read. Source of truth; never deleted.
- **Silver:** cleaned, deduplicated, joined, conformed to a schema. Analyst-ready.
- **Gold:** business-aggregated, denormalized, optimized for specific consumption (dashboards, ML features).

Each tier reads from the previous and is independently rebuildable from bronze. This is the modern operational pattern for lakehouse pipelines — covered in detail in Lessons 11 and 19.

### Theory: Lambda vs Kappa Architectures

Two architectural patterns for combining batch and streaming. The choice shapes everything from team structure to monitoring.

#### C.1 Lambda architecture

```
              ┌──── batch layer (Spark) ──── batch view ────┐
source ──────┤                                              ├── serving layer (query)
              └──── speed layer (Storm)  ──── realtime view ┘
```

- **Batch layer** processes the entire dataset (or large windows) for high accuracy.
- **Speed layer** processes recent events in real time for low latency.
- **Serving layer** merges the two views at query time.

Pros: combines accuracy of batch with freshness of streaming. Cons: two codebases (batch + stream) implementing the same logic, plus the merge logic. Operationally heavy.

#### C.2 Kappa architecture

```
source ──── stream processor (Flink/Spark Structured Streaming) ──── materialized view
                              ↑
                        replay from log
                        (Kafka retention)
```

A single stream processor handles both real-time and historical reprocessing. To "rebuild" a view, replay the stream from the beginning (Kafka can retain forever; or replay from the lakehouse). One codebase, one operational story.

Pros: simpler, single source of truth. Cons: stream processor must be capable of high throughput batch-style replay; some workloads (heavy joins across years of data) are still better as nightly batch.

#### C.3 Which won

In practice, modern stacks lean Kappa-with-batch-supplement: streaming for low-latency views and continuous ETL, scheduled batch (Airflow + Spark + dbt) for complex aggregations and historical rebuilds. The Lambda pattern survives in legacy systems but is rarely the choice for new builds, because the cost of maintaining two implementations exceeds the latency benefit for most analytics use cases.

### 4.1 Traditional Data Warehouse Architecture

```
┌──────────────────────────────────────────────────────────────┐
│              Traditional Data Warehouse Architecture          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────┐   ┌─────────┐   ┌─────────────────────────┐    │
│  │ Source 1│   │ Source 2│   │       Source N          │    │
│  │  (ERP)  │   │  (CRM)  │   │      (Other)            │    │
│  └────┬────┘   └────┬────┘   └───────────┬─────────────┘    │
│       │             │                     │                  │
│       └─────────────┼─────────────────────┘                  │
│                     ↓                                        │
│           ┌─────────────────┐                                │
│           │   ETL Process   │                                │
│           │ (Extract-Transform-Load)                         │
│           └────────┬────────┘                                │
│                    ↓                                         │
│           ┌─────────────────┐                                │
│           │  Data Warehouse │                                │
│           │   (Star Schema) │                                │
│           └────────┬────────┘                                │
│                    ↓                                         │
│           ┌─────────────────┐                                │
│           │    BI Tools     │                                │
│           │ (Tableau, Power BI)                              │
│           └─────────────────┘                                │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 Modern Data Lake Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                  Modern Data Lake Architecture                │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Sources                                                     │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                    │
│  │ API │ │ DB  │ │ IoT │ │ Log │ │Files│                    │
│  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘                    │
│     └───────┴───────┴───────┴───────┘                        │
│                     ↓                                        │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                    Data Lake                         │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐             │    │
│  │  │  Bronze │→│  Silver │→│  Gold   │              │    │
│  │  │   Raw   │  │ Cleaned │  │Curated │              │    │
│  │  └─────────┘  └─────────┘  └─────────┘             │    │
│  └─────────────────────────────────────────────────────┘    │
│                     ↓                                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │BI/Reports│ │ ML/AI    │ │ Data Apps│ │ API      │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
└──────────────────────────────────────────────────────────────┘
```

### 4.3 Lambda Architecture

A hybrid architecture combining batch and streaming.

> **When to choose Lambda vs Kappa?**
> - **Lambda**: Use when you need historical reprocessing (e.g., correcting past aggregations after a schema change) and real-time results simultaneously. The batch layer acts as the "source of truth" that the speed layer's approximate results converge toward.
> - **Kappa**: Use when a streaming-first approach suffices and the event log is replayable (e.g., Kafka with long retention). Kappa avoids the operational burden of maintaining two separate codebases for the same logic.

```python
# Lambda architecture concept implementation
class LambdaArchitecture:
    """Lambda architecture: Batch + Streaming layers"""

    def __init__(self):
        # Two parallel layers solve a fundamental tension: batch gives
        # *accurate* results (full dataset recomputation) while speed gives
        # *timely* results (sub-second latency). Neither alone satisfies
        # use cases like fraud detection dashboards that need both properties.
        self.batch_layer = BatchLayer()
        self.speed_layer = SpeedLayer()
        self.serving_layer = ServingLayer()

    def ingest(self, data):
        """Data ingestion: Send to both layers simultaneously"""
        # Dual-write: the same event feeds both layers so they stay in sync.
        # The batch layer stores the immutable master copy for reprocessing;
        # the speed layer processes it immediately for low-latency queries.
        self.batch_layer.append(data)
        self.speed_layer.process(data)

    def query(self, params):
        """Query: Merge batch view + real-time view"""
        batch_result = self.serving_layer.get_batch_view(params)
        realtime_result = self.speed_layer.get_realtime_view(params)

        # Merge logic: the batch view covers all data up to the last batch run;
        # the real-time view covers only the gap since then. Merging ensures
        # the query result is both complete *and* up-to-date.
        return self.merge_views(batch_result, realtime_result)

class BatchLayer:
    """Batch layer: Process entire dataset"""

    def append(self, data):
        """Append to master dataset"""
        # Append-only (immutable) storage preserves the raw event history,
        # enabling full recomputation if business logic changes later.
        pass

    def compute_batch_views(self):
        """Compute batch views (periodic execution)"""
        # Recomputes over the *entire* dataset — expensive but guarantees
        # correctness. Typically runs on a schedule (e.g., hourly/daily)
        # using Spark or MapReduce on the master dataset.
        pass

class SpeedLayer:
    """Speed layer: Real-time data processing"""

    def process(self, data):
        """Real-time processing"""
        # Incremental updates only — fast but potentially approximate.
        # Once the batch layer catches up, its results supersede the
        # speed layer's, so any approximation errors are self-correcting.
        pass

    def get_realtime_view(self, params):
        """Return real-time view"""
        pass

class ServingLayer:
    """Serving layer: Query processing"""

    def get_batch_view(self, params):
        """Return batch view"""
        # Serves pre-computed batch results from a low-latency store
        # (e.g., Cassandra, HBase) — the heavy computation happened
        # during the batch run, so reads are fast.
        pass
```

### 4.4 Kappa Architecture

A simplified architecture using only streaming.

```
┌──────────────────────────────────────────────────────────────┐
│                    Kappa Architecture                        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Sources                                                     │
│  ┌─────┐ ┌─────┐ ┌─────┐                                    │
│  │Event│ │Event│ │Event│                                    │
│  └──┬──┘ └──┬──┘ └──┬──┘                                    │
│     └───────┴───────┘                                        │
│             ↓                                                │
│  ┌─────────────────────────────────────┐                    │
│  │         Message Queue (Kafka)       │                    │
│  │         - Event Log                 │                    │
│  │         - Replayable                │                    │
│  └─────────────────┬───────────────────┘                    │
│                    ↓                                         │
│  ┌─────────────────────────────────────┐                    │
│  │      Stream Processing Layer        │                    │
│  │      (Flink, Spark Streaming)       │                    │
│  └─────────────────┬───────────────────┘                    │
│                    ↓                                         │
│  ┌─────────────────────────────────────┐                    │
│  │          Serving Layer              │                    │
│  │    (Database, Cache, API)           │                    │
│  └─────────────────────────────────────┘                    │
└──────────────────────────────────────────────────────────────┘
```

---

## 5. Data Engineering Tool Ecosystem

### 5.1 Major Tool Categories

```python
# Organized into functional layers rather than vendor buckets:
# this reflects how a real data platform is assembled — you pick one
# tool per layer and swap vendors without changing the overall design.
data_engineering_tools = {
    "orchestration": {
        # Batch orchestrators schedule DAGs; streaming ones manage
        # long-running topologies — different failure/retry models.
        "batch": ["Apache Airflow", "Prefect", "Dagster", "Luigi"],
        "streaming": ["Apache Kafka", "Apache Flink", "Spark Streaming"]
    },
    "processing": {
        "batch": ["Apache Spark", "Apache Hive", "Presto/Trino"],
        "streaming": ["Apache Kafka Streams", "Apache Flink", "Apache Storm"]
    },
    "storage": {
        # Lake vs warehouse vs DB is a latency/cost/flexibility trade-off:
        # lakes are cheapest for raw data, warehouses optimise for analytics,
        # OLTP databases serve low-latency application reads.
        "data_lake": ["S3", "GCS", "HDFS", "Azure Blob"],
        "data_warehouse": ["Snowflake", "BigQuery", "Redshift", "Databricks"],
        "databases": ["PostgreSQL", "MySQL", "MongoDB", "Cassandra"]
    },
    "transformation": {
        # SQL-based tools (dbt) let analysts own transforms without Python;
        # code-based tools (PySpark) handle ML feature engineering or complex
        # business logic that is awkward in pure SQL.
        "sql_based": ["dbt", "SQLMesh"],
        "code_based": ["PySpark", "Pandas", "Polars"]
    },
    "quality": {
        "testing": ["Great Expectations", "dbt tests", "Soda"],
        "monitoring": ["Monte Carlo", "Datadog", "Grafana"]
    },
    "catalog": ["Apache Atlas", "DataHub", "Amundsen", "OpenMetadata"]
}
```

### 5.2 Cloud Service Mapping

| Function | AWS | GCP | Azure |
|------|-----|-----|-------|
| **Orchestration** | Step Functions, MWAA | Cloud Composer | Data Factory |
| **Streaming** | Kinesis | Pub/Sub, Dataflow | Event Hubs |
| **Batch Processing** | EMR, Glue | Dataproc, Dataflow | HDInsight |
| **Data Lake** | S3 + Lake Formation | GCS + BigLake | ADLS + Synapse |
| **Data Warehouse** | Redshift | BigQuery | Synapse Analytics |

---

## 6. Data Engineering Best Practices

### 6.1 Pipeline Design Principles

```python
# Good pipeline design principles
# These are ordered roughly by how often violations cause production incidents:
# idempotency failures cause duplicate data, atomicity failures cause partial
# loads, and missing monitoring causes silent failures that go unnoticed for days.
pipeline_best_practices = {
    "idempotency": "Same input produces same result",
    "atomicity": "All succeed or all fail",
    "incremental": "Ensure efficiency with incremental processing",
    "monitoring": "Monitor at every stage",
    "error_handling": "Retry and alert on failure",
    "documentation": "Manage code and documentation together"
}

# Idempotency example
def idempotent_upsert(df, table_name, key_columns):
    """Upsert function ensuring idempotency"""
    # DELETE-then-INSERT rather than INSERT alone: if the pipeline is re-run
    # (e.g., after a partial failure), this approach prevents duplicate rows.
    # An alternative is MERGE/UPSERT, but DELETE+INSERT is simpler to reason
    # about and works identically across most SQL dialects.
    delete_query = f"""
    DELETE FROM {table_name}
    WHERE (key1, key2) IN (
        SELECT DISTINCT key1, key2 FROM staging_table
    )
    """
    # execute(delete_query)
    # insert_dataframe(df, table_name)
    pass
```

### 6.2 Error Handling and Retry

```python
import time
from functools import wraps
from typing import Callable, Type

def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    # Catch only specific exceptions in production (e.g., ConnectionError)
    # to avoid retrying on programming bugs like TypeError.
    exceptions: tuple[Type[Exception], ...] = (Exception,)
):
    """Retry decorator"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        print(f"Attempt {attempt} failed: {e}")
                        # Linear backoff (delay * attempt): gives transient
                        # failures time to resolve without hammering the
                        # upstream service. For true exponential, use
                        # delay * (2 ** attempt) + random jitter.
                        time.sleep(delay * attempt)
            raise last_exception
        return wrapper
    return decorator

# max_attempts=3, delay=2.0 → waits 2s, then 4s before giving up.
# Total worst-case wait: 6s, which balances fast recovery against
# overwhelming an already-struggling API.
@retry(max_attempts=3, delay=2.0)
def fetch_data_from_api(url: str):
    """Fetch data from API (with retry)"""
    import requests
    # timeout=30: prevents the pipeline from hanging indefinitely if
    # the API is alive but slow; 30s is generous enough for most REST APIs.
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()
```

---

## Practice Problems

### Problem 1: Pipeline Design
Design a pipeline that generates daily sales reports for an online shopping mall.

```python
# Example solution
class DailySalesReportPipeline:
    def extract(self):
        """Extract order, product, customer data"""
        pass

    def transform(self):
        """Sales aggregation, category analysis"""
        pass

    def load(self):
        """Load report table"""
        pass
```

### Problem 2: Batch vs Streaming Selection
Choose the appropriate approach (batch or streaming) for the following cases and explain why:
- Daily sales report generation
- Real-time low stock alerts
- Monthly customer segmentation

---

## Summary

| Concept | Description |
|------|------|
| **Data Pipeline** | Moving and transforming data from source to destination |
| **Batch Processing** | Periodically processing large volumes of data |
| **Stream Processing** | Processing data in real-time |
| **Data Lake** | Storage for raw data |
| **Data Warehouse** | Analytics storage for cleaned data |
| **ETL/ELT** | Extract, transform, load data process |

---

## References

- [Fundamentals of Data Engineering (O'Reilly)](https://www.oreilly.com/library/view/fundamentals-of-data/9781098108298/)
- [The Data Engineering Cookbook](https://github.com/andkret/Cookbook)
- [Data Engineering Weekly Newsletter](https://dataengineeringweekly.com/)

---

[← Previous: Overview](00_Overview.md) | [Next: 2. Data Modeling Basics →](02_Data_Modeling_Basics.md)
