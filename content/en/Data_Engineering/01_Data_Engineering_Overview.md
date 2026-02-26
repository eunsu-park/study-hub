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
