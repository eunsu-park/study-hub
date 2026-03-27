[← Previous: 19. Lakehouse Practical Patterns](19_Lakehouse_Practical_Patterns.md) | [Next: 21. Data Versioning and Data Contracts →](21_Data_Versioning_and_Contracts.md)

# Dagster — Asset-Based Orchestration

## Learning Objectives

1. Understand Dagster's software-defined asset philosophy and how it differs from task-based DAGs
2. Master core Dagster concepts: Assets, Ops, Jobs, Graphs, Resources, and IO Managers
3. Compare Dagster and Airflow across key dimensions (developer experience, testing, observability)
4. Set up a Dagster project with proper structure (`dagster.yaml`, `definitions.py`)
5. Build end-to-end data pipelines using Dagster's asset graph
6. Implement partitioned assets for incremental and time-windowed processing
7. Integrate Dagster with dbt, Spark, and pandas for real-world pipelines

---

## Overview

Most orchestration tools — Airflow, Prefect, Luigi — think in terms of **tasks**: "run this Python function, then run that one." Dagster flips the mental model entirely. Instead of asking "what steps should I run?", Dagster asks "what data assets should exist, and how are they derived?" This seemingly small shift has profound consequences for how we build, test, debug, and observe data pipelines.

Dagster was created by Nick Schrock (co-creator of GraphQL at Facebook) and first released in 2019. It reached production maturity with Dagster 1.0 in 2022, and has since become one of the fastest-growing orchestration frameworks. Its asset-centric model aligns naturally with the modern data stack — where dbt models, ML features, and analytics tables are all **assets** that consumers depend on.

In this lesson, we will build a complete understanding of Dagster from philosophy to production deployment. By the end, you will be able to architect and implement asset-based pipelines that are testable, observable, and maintainable.

> **Analogy**: Dagster assets are like a recipe book where each dish (asset) declares its ingredients (upstream assets). The kitchen (Dagster) figures out the cooking order automatically. If you update a recipe for the sauce, the kitchen knows which dishes need to be re-prepared — without you manually listing every step.

---

## 1. The Software-Defined Asset Philosophy

### 1.1 Task-Based vs Asset-Based Thinking

The fundamental difference between Dagster and traditional orchestrators lies in what they consider the primary abstraction.

```python
"""
Task-Based Orchestration (Airflow model):
─────────────────────────────────────────
"What TASKS should I run, and in what ORDER?"

  extract_orders() → clean_orders() → compute_metrics() → load_dashboard()
       Task 1           Task 2            Task 3             Task 4

- The DAG describes COMPUTATION steps
- Data is a side effect of running tasks
- If Task 3 fails, you know WHICH TASK failed
- But do you know WHICH DATASET is stale? Not directly.


Asset-Based Orchestration (Dagster model):
──────────────────────────────────────────
"What DATA ASSETS should exist, and how are they derived?"

  raw_orders → cleaned_orders → order_metrics → dashboard_summary
    Asset 1       Asset 2          Asset 3          Asset 4

- The graph describes DATA DEPENDENCIES
- Computation is the means to produce assets
- If order_metrics is stale, you immediately see:
  - When it was last materialized
  - What upstream assets it depends on
  - Who consumes it downstream
"""
```

Why does this matter? Consider a real scenario: your daily pipeline fails at 3 AM. With a task-based system, you see "Task `compute_revenue` failed." You then need to figure out which datasets are affected. With Dagster, you see "Asset `daily_revenue` is stale since yesterday" — and every downstream consumer is visually marked as potentially stale too.

### 1.2 The Three Pillars of Dagster

Dagster is built on three core ideas that reinforce each other:

```python
"""
Dagster's Three Pillars:

1. SOFTWARE-DEFINED ASSETS
   - Assets are first-class citizens, not side effects
   - Each asset declares: what it produces, what it depends on, how to compute it
   - The asset graph IS the pipeline definition

2. DECLARATIVE DATA MANAGEMENT
   - You declare WHAT should exist, Dagster figures out HOW to get there
   - Reconciliation: compare desired state vs actual state
   - Freshness policies: "this asset should be no more than 1 hour old"

3. TESTABLE BY DESIGN
   - Every asset is a Python function — unit-testable in isolation
   - Resources are injectable — swap a production DB for a test DB
   - No need to spin up a scheduler to run your pipeline locally

Together these create a development experience where:
  ┌────────────────┐     ┌────────────────┐     ┌────────────────┐
  │ Define Assets  │ ──→ │  Test Locally   │ ──→ │  Deploy with   │
  │ as Python      │     │  (pytest)       │     │  Confidence    │
  └────────────────┘     └────────────────┘     └────────────────┘
"""
```

### 1.3 When to Choose Dagster

Dagster excels in certain scenarios and is less ideal in others:

| Scenario | Dagster | Airflow |
|----------|---------|---------|
| Data-centric pipelines (ELT, analytics) | Excellent | Good |
| ML feature pipelines | Excellent | Adequate |
| dbt integration | Native (`dagster-dbt`) | Via operators |
| General task orchestration (non-data) | Adequate | Excellent |
| Existing Airflow investment | Migration cost | Stay |
| Team size < 5, new project | Recommended | Also fine |
| Testing & local development | Superior | Requires setup |
| Event-driven / sensor-heavy | Good | Good |
| Mature ecosystem / community size | Growing | Very large |

---

## 2. Core Concepts Deep Dive

### 2.1 Assets — The Foundation

An **asset** is a persistent object in your data platform — a table, a file, a model artifact, a dashboard. In Dagster, you define assets as decorated Python functions.

```python
import dagster as dg
import pandas as pd

# Why @asset decorator? It tells Dagster:
# 1. This function PRODUCES a named data asset
# 2. Its parameters are UPSTREAM dependencies (by name)
# 3. Its return value IS the asset's data

@dg.asset(
    description="Raw order data ingested from the e-commerce API",
    metadata={"source": "api.store.com/orders", "owner": "data-team"},
    group_name="bronze",          # Visual grouping in Dagster UI
)
def raw_orders() -> pd.DataFrame:
    """Ingest raw order data from the e-commerce API.

    This asset has no upstream dependencies — it's a SOURCE asset.
    """
    # In production, this would call an API or read from S3
    return pd.DataFrame({
        "order_id": [1, 2, 3, 4, 5],
        "customer_id": [101, 102, 101, 103, 102],
        "amount": [99.99, 149.50, 25.00, 299.99, 75.00],
        "status": ["completed", "completed", "refunded", "completed", "pending"],
        "created_at": pd.date_range("2024-01-01", periods=5, freq="D"),
    })


@dg.asset(
    description="Orders with invalid records removed and types standardized",
    group_name="silver",
)
def cleaned_orders(raw_orders: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate order data.

    Why the parameter name matters:
      - `raw_orders` matches the upstream asset name exactly
      - Dagster resolves this as a dependency automatically
      - No need to manually wire dependencies like Airflow's >>
    """
    df = raw_orders.copy()

    # Why filter completed only at silver layer?
    # Bronze preserves ALL data; Silver applies business rules
    df = df[df["status"] == "completed"]

    # Type standardization — ensures downstream assets get consistent types
    df["amount"] = df["amount"].astype(float)
    df["created_at"] = pd.to_datetime(df["created_at"])

    return df


@dg.asset(
    description="Aggregated order metrics per customer",
    group_name="gold",
)
def order_metrics(cleaned_orders: pd.DataFrame) -> pd.DataFrame:
    """Compute per-customer order metrics.

    Why aggregate at the gold layer?
      - Gold assets serve business consumers (BI tools, dashboards)
      - Pre-aggregation avoids repeated computation by analysts
      - Clear ownership: data team owns gold, analysts consume it
    """
    metrics = cleaned_orders.groupby("customer_id").agg(
        total_orders=("order_id", "count"),
        total_revenue=("amount", "sum"),
        avg_order_value=("amount", "mean"),
        first_order=("created_at", "min"),
        last_order=("created_at", "max"),
    ).reset_index()

    return metrics
```

The asset graph that Dagster constructs from this code:

```
raw_orders ──→ cleaned_orders ──→ order_metrics
 (bronze)         (silver)           (gold)
```

### 2.2 Ops, Graphs, and Jobs — The Computation Layer

While assets are the primary abstraction, Dagster also supports **ops** (operations) for cases where you need fine-grained control over computation steps.

```python
import dagster as dg

# Why use Ops instead of Assets?
# - When the computation doesn't produce a persistent data asset
# - When you need fine-grained retry/timeout per step
# - When migrating from Airflow (ops map to tasks)

@dg.op(
    description="Validate that the source API is reachable",
    retry_policy=dg.RetryPolicy(max_retries=3, delay=10),
)
def check_api_health(context: dg.OpExecutionContext) -> bool:
    """Check if the external API is responding."""
    context.log.info("Checking API health...")
    # In production: requests.get("https://api.store.com/health")
    return True


@dg.op
def extract_data(context: dg.OpExecutionContext, api_healthy: bool) -> dict:
    """Extract data only if the API is healthy."""
    if not api_healthy:
        raise dg.Failure(description="API is not healthy")
    context.log.info("Extracting data from API")
    return {"orders": [1, 2, 3], "extracted_at": "2024-01-15"}


@dg.op
def validate_data(context: dg.OpExecutionContext, raw_data: dict) -> dict:
    """Validate extracted data meets expectations."""
    assert len(raw_data["orders"]) > 0, "No orders extracted"
    context.log.info(f"Validated {len(raw_data['orders'])} orders")
    return raw_data


# A Graph composes Ops into a computation DAG
# Why separate Graph from Job?
# - Graph = the logical computation (reusable)
# - Job = Graph + configuration (environment-specific)

@dg.graph
def etl_graph():
    """Define the ETL computation graph."""
    healthy = check_api_health()
    raw = extract_data(healthy)
    validate_data(raw)


# Job = Graph bound to specific resources/config
etl_job = etl_graph.to_job(
    name="etl_job",
    description="Daily ETL job for order data",
    config={
        "ops": {
            "check_api_health": {"config": {}},
        }
    },
)
```

**When to use what:**

| Concept | Use When | Think of It As |
|---------|----------|----------------|
| `@asset` | Producing persistent data | "What data should exist?" |
| `@op` | Step in a computation | "What computation should run?" |
| `@graph` | Composing ops | "How do steps connect?" |
| `Job` | Running a graph | "When and how to execute?" |

### 2.3 Resources — Dependency Injection

Resources are Dagster's mechanism for dependency injection. They allow you to swap implementations between environments (dev, staging, production) without changing asset code.

```python
import dagster as dg

# Why Resources?
# 1. Decouple assets from infrastructure (S3 client, DB connection)
# 2. Enable testing — inject a mock resource in tests
# 3. Share connections across assets (connection pooling)
# 4. Configure per-environment (dev uses local files, prod uses S3)


class DatabaseResource(dg.ConfigurableResource):
    """A configurable database connection resource.

    Why ConfigurableResource?
      - Type-safe configuration via Pydantic
      - Validated at pipeline startup, not at runtime
      - Self-documenting in the Dagster UI
    """
    host: str
    port: int = 5432
    database: str
    username: str
    password: str

    def get_connection_string(self) -> str:
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

    def execute_query(self, query: str) -> list:
        """Execute a SQL query (simplified for illustration)."""
        # In production: use sqlalchemy or psycopg2
        return [{"result": "mock_data"}]


class S3Resource(dg.ConfigurableResource):
    """S3 client resource for reading/writing data."""
    bucket: str
    region: str = "us-east-1"
    endpoint_url: str = ""  # For MinIO/LocalStack in dev

    def read_parquet(self, key: str) -> "pd.DataFrame":
        """Read a Parquet file from S3."""
        import pandas as pd
        path = f"s3://{self.bucket}/{key}"
        return pd.read_parquet(path)

    def write_parquet(self, df: "pd.DataFrame", key: str) -> None:
        """Write a DataFrame as Parquet to S3."""
        path = f"s3://{self.bucket}/{key}"
        df.to_parquet(path, index=False)


# Using resources in assets:

@dg.asset
def revenue_report(
    context: dg.AssetExecutionContext,
    database: DatabaseResource,      # Injected automatically by name
    s3: S3Resource,                   # Injected automatically by name
) -> None:
    """Generate revenue report from database to S3.

    Why declare resources as parameters?
      - Dagster injects them automatically at runtime
      - In tests, you can provide mock resources
      - The asset is self-documenting: you see its dependencies
    """
    results = database.execute_query("SELECT * FROM daily_revenue")
    import pandas as pd
    df = pd.DataFrame(results)
    s3.write_parquet(df, "reports/revenue/latest.parquet")
    context.log.info(f"Wrote {len(df)} rows to S3")
```

### 2.4 IO Managers — Controlling Asset Persistence

IO Managers define **how** assets are stored and loaded. They separate the "what" (asset logic) from the "where" (storage).

```python
import dagster as dg
import pandas as pd
from pathlib import Path


class ParquetIOManager(dg.ConfigurableIOManager):
    """Store assets as Parquet files on the local filesystem.

    Why IO Managers?
      - Asset functions return DataFrames; IO Managers handle persistence
      - Swap between local Parquet (dev) and S3 Parquet (prod) by config
      - Upstream assets are automatically loaded before downstream runs
    """
    base_path: str

    def handle_output(self, context: dg.OutputContext, obj: pd.DataFrame) -> None:
        """Save the asset's output to a Parquet file."""
        # Why use asset_key for the path?
        # Each asset gets a unique, deterministic file path
        path = Path(self.base_path) / f"{context.asset_key.to_python_identifier()}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        obj.to_parquet(path, index=False)
        context.log.info(f"Wrote {len(obj)} rows to {path}")

        # Attach metadata visible in the Dagster UI
        context.add_output_metadata({
            "num_rows": len(obj),
            "columns": list(obj.columns),
            "path": str(path),
        })

    def load_input(self, context: dg.InputContext) -> pd.DataFrame:
        """Load an upstream asset from its Parquet file."""
        path = Path(self.base_path) / f"{context.asset_key.to_python_identifier()}.parquet"
        df = pd.read_parquet(path)
        context.log.info(f"Loaded {len(df)} rows from {path}")
        return df


class CsvIOManager(dg.ConfigurableIOManager):
    """Store assets as CSV files (useful for debugging / small datasets)."""
    base_path: str

    def handle_output(self, context: dg.OutputContext, obj: pd.DataFrame) -> None:
        path = Path(self.base_path) / f"{context.asset_key.to_python_identifier()}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        obj.to_csv(path, index=False)

    def load_input(self, context: dg.InputContext) -> pd.DataFrame:
        path = Path(self.base_path) / f"{context.asset_key.to_python_identifier()}.csv"
        return pd.read_csv(path)
```

---

## 3. Dagster vs Airflow — Detailed Comparison

Understanding the trade-offs helps you make an informed choice for your team and project.

### 3.1 Feature Comparison

| Dimension | Dagster | Airflow |
|-----------|---------|---------|
| **Primary abstraction** | Software-defined assets | Task DAGs |
| **Data awareness** | First-class (lineage, freshness) | Limited (via metadata plugins) |
| **Local development** | `dagster dev` (instant) | Docker Compose or standalone |
| **Testing** | `pytest` native, in-process | Requires running scheduler |
| **Dynamic pipelines** | Native (Python control flow) | `expand()` in 2.4+ |
| **Backfills** | Asset-level partitioned backfills | DAG-level reruns |
| **UI** | Asset graph + lineage + freshness | Task graph + Gantt charts |
| **Configuration** | Pydantic-based, type-safe | Airflow Variables/Connections |
| **Scheduling** | Cron, sensors, freshness policies | Cron, timetables, sensors |
| **Community** | ~10K GitHub stars, growing | ~37K GitHub stars, mature |
| **Managed cloud** | Dagster Cloud (Dagster+) | Astronomer, MWAA, Cloud Composer |
| **Plugin ecosystem** | ~50 integrations | ~1000+ providers |

### 3.2 Code Comparison — Same Pipeline in Both

```python
"""
The same pipeline — "ingest orders, clean, aggregate" — in both systems.
This illustrates the philosophical difference.
"""

# ── Airflow Version ──────────────────────────────────────────────
# Focus: WHAT TASKS to run, in WHAT ORDER

# from airflow.decorators import dag, task
# @dag(schedule='@daily', start_date=datetime(2024, 1, 1))
# def order_pipeline():
#     @task()
#     def extract():
#         return fetch_orders()
#
#     @task()
#     def transform(raw):
#         return clean(raw)
#
#     @task()
#     def load(cleaned):
#         write_to_warehouse(cleaned)
#
#     load(transform(extract()))
#
# order_pipeline()

# Key observations:
# - You define TASKS (verbs): extract, transform, load
# - Data flow is implicit via return values
# - No built-in concept of "what asset does this produce?"
# - Testing requires Airflow context

# ── Dagster Version ──────────────────────────────────────────────
# Focus: WHAT DATA should exist, HOW it's derived

# import dagster as dg
# @dg.asset
# def raw_orders():
#     return fetch_orders()
#
# @dg.asset
# def cleaned_orders(raw_orders):
#     return clean(raw_orders)
#
# @dg.asset
# def order_metrics(cleaned_orders):
#     return aggregate(cleaned_orders)

# Key observations:
# - You define ASSETS (nouns): raw_orders, cleaned_orders, order_metrics
# - Dependencies are explicit via function parameters
# - Each asset is independently materializable and testable
# - No orchestrator context needed for testing
```

### 3.3 Migration Path: Airflow to Dagster

Dagster provides `dagster-airflow` for incremental migration:

```python
"""
Migration Strategy (incremental, not big-bang):

Phase 1: Run Dagster alongside Airflow
  - Wrap existing Airflow DAGs as Dagster jobs
  - Use dagster-airflow to import DAGs
  - Both systems run in parallel

Phase 2: Convert critical pipelines
  - Rewrite high-value DAGs as Dagster assets
  - Keep legacy DAGs in Airflow
  - Dagster sensors watch for Airflow completions

Phase 3: Full migration
  - All pipelines in Dagster
  - Decommission Airflow
  - Consolidate monitoring

Timeline estimate: 3-6 months for a team of 5-10 engineers
"""

# Example: Importing an Airflow DAG into Dagster
# from dagster_airflow import make_dagster_definitions_from_airflow_dags
#
# definitions = make_dagster_definitions_from_airflow_dags(
#     airflow_dags_path="/path/to/airflow/dags",
#     connections=[...],
# )
```

---

## 4. Dagster Project Structure

### 4.1 Standard Layout

A well-organized Dagster project follows this structure:

```
my_dagster_project/
├── my_dagster_project/          # Python package
│   ├── __init__.py
│   ├── definitions.py           # Entry point — Dagster loads this
│   ├── assets/                  # Asset definitions
│   │   ├── __init__.py
│   │   ├── bronze.py            # Raw ingestion assets
│   │   ├── silver.py            # Cleaned/validated assets
│   │   └── gold.py              # Business-level aggregations
│   ├── resources/               # Resource definitions
│   │   ├── __init__.py
│   │   ├── database.py
│   │   └── storage.py
│   ├── jobs.py                  # Job definitions (if using ops)
│   ├── schedules.py             # Schedule definitions
│   └── sensors.py               # Sensor definitions
├── tests/                       # Unit and integration tests
│   ├── test_assets.py
│   └── test_resources.py
├── dagster.yaml                 # Dagster instance configuration
├── pyproject.toml               # Project metadata + Dagster config
└── workspace.yaml               # Multi-project workspace config
```

### 4.2 The Definitions Object

The `definitions.py` file is the entry point — it tells Dagster about all your assets, resources, jobs, schedules, and sensors.

```python
# definitions.py — the central registry for your Dagster project

import dagster as dg
from my_dagster_project.assets.bronze import raw_orders, raw_customers
from my_dagster_project.assets.silver import cleaned_orders, cleaned_customers
from my_dagster_project.assets.gold import order_metrics, customer_lifetime_value

# Why a single Definitions object?
# 1. Dagster knows everything about your project from one place
# 2. Validation happens at import time, not at runtime
# 3. The UI/CLI reads this to render the asset graph

defs = dg.Definitions(
    assets=[
        # Bronze layer (raw ingestion)
        raw_orders,
        raw_customers,
        # Silver layer (cleaned)
        cleaned_orders,
        cleaned_customers,
        # Gold layer (aggregated)
        order_metrics,
        customer_lifetime_value,
    ],
    resources={
        # Why define resources here?
        # - Assets reference resources by NAME (string key)
        # - Swap implementations for different environments
        "database": DatabaseResource(
            host="localhost",
            database="analytics",
            username="dagster",
            password=dg.EnvVar("DB_PASSWORD"),   # Read from environment
        ),
        "io_manager": ParquetIOManager(
            base_path="/data/dagster/assets",
        ),
    },
    schedules=[daily_refresh_schedule],
    sensors=[new_file_sensor],
)
```

### 4.3 Configuration Files

```yaml
# dagster.yaml — Dagster instance configuration
# This configures the Dagster INSTANCE (storage, telemetry, etc.)
# Not to be confused with definitions.py which configures your CODE

storage:
  # Where Dagster stores run history, event logs, etc.
  postgres:
    postgres_url:
      env: DAGSTER_PG_URL
    # Why Postgres instead of SQLite?
    # - Concurrent access (multiple workers)
    # - Persistent across restarts
    # - Required for production deployments

run_coordinator:
  module: dagster.core.run_coordinator
  class: QueuedRunCoordinator
  config:
    max_concurrent_runs: 10
    # Why limit concurrent runs?
    # - Prevent resource exhaustion
    # - Control database connection count
    # - Predictable pipeline execution

telemetry:
  enabled: false
```

```toml
# pyproject.toml — Project configuration
[project]
name = "my_dagster_project"
version = "0.1.0"
dependencies = [
    "dagster>=1.6",
    "dagster-pandas",
    "dagster-dbt",
    "pandas",
    "pyarrow",
]

[tool.dagster]
module_name = "my_dagster_project.definitions"
# Why specify module_name?
# - `dagster dev` uses this to find your Definitions
# - No need for workspace.yaml in single-project setups
```

---

## 5. Building a Data Pipeline with Dagster Assets

### 5.1 Multi-Source Ingestion Pipeline

Let's build a realistic e-commerce analytics pipeline with multiple data sources.

```python
import dagster as dg
import pandas as pd
from datetime import datetime, timedelta


# ── Source Assets (Bronze Layer) ──────────────────────────────────

@dg.asset(
    group_name="bronze",
    description="Raw order data from the transactional database",
    metadata={"source": "postgres://orders_db", "refresh": "daily"},
)
def raw_orders(database: DatabaseResource) -> pd.DataFrame:
    """Ingest order data from the OLTP database.

    Why pull from a replica, not the primary?
      - Avoid query load on the production database
      - Read replicas can handle long-running analytical queries
      - No risk of locking production tables
    """
    return database.execute_query("""
        SELECT order_id, customer_id, product_id, quantity, unit_price,
               discount, status, created_at, updated_at
        FROM orders
        WHERE updated_at >= CURRENT_DATE - INTERVAL '1 day'
    """)


@dg.asset(
    group_name="bronze",
    description="Raw product catalog from the product API",
)
def raw_products() -> pd.DataFrame:
    """Ingest product catalog (changes infrequently)."""
    # Why separate asset for products?
    # - Different refresh frequency (weekly vs daily for orders)
    # - Different source system (API vs database)
    # - Independent materialization
    return pd.DataFrame({
        "product_id": [1, 2, 3],
        "name": ["Widget A", "Widget B", "Premium Widget"],
        "category": ["basic", "basic", "premium"],
        "cost": [10.0, 15.0, 50.0],
    })


# ── Cleaned Assets (Silver Layer) ─────────────────────────────────

@dg.asset(
    group_name="silver",
    description="Validated orders with enriched product information",
)
def enriched_orders(
    raw_orders: pd.DataFrame,
    raw_products: pd.DataFrame,
) -> pd.DataFrame:
    """Join orders with product data and compute derived fields.

    Why join at the silver layer?
      - Bronze stores raw, un-joined data for replay capability
      - Silver is where we create the "single source of truth" view
      - Downstream gold assets don't need to repeat this join
    """
    df = raw_orders.merge(raw_products, on="product_id", how="left")

    # Compute total amount per line item
    df["line_total"] = df["quantity"] * df["unit_price"] * (1 - df["discount"])

    # Flag high-value orders for business attention
    # Why $500? This is a configurable business rule
    df["is_high_value"] = df["line_total"] > 500

    return df


# ── Business Assets (Gold Layer) ──────────────────────────────────

@dg.asset(
    group_name="gold",
    description="Daily revenue metrics by product category",
)
def daily_category_revenue(enriched_orders: pd.DataFrame) -> pd.DataFrame:
    """Aggregate revenue by category for the BI dashboard.

    Business logic:
      - Only completed orders count toward revenue
      - Revenue = sum of line_total (already discounted)
      - AOV (Average Order Value) = revenue / unique orders
    """
    completed = enriched_orders[enriched_orders["status"] == "completed"]

    metrics = completed.groupby("category").agg(
        total_revenue=("line_total", "sum"),
        order_count=("order_id", "nunique"),
        avg_discount=("discount", "mean"),
    ).reset_index()

    metrics["aov"] = metrics["total_revenue"] / metrics["order_count"]
    return metrics


@dg.asset(
    group_name="gold",
    description="Customer segmentation based on purchase behavior",
)
def customer_segments(enriched_orders: pd.DataFrame) -> pd.DataFrame:
    """Segment customers into value tiers.

    Why RFM-lite segmentation?
      - Recency, Frequency, Monetary — standard retail analytics
      - Simple yet effective for targeting and retention campaigns
      - Gold layer is the right place for business logic
    """
    customer_stats = enriched_orders.groupby("customer_id").agg(
        total_spent=("line_total", "sum"),
        order_count=("order_id", "nunique"),
        last_order=("created_at", "max"),
    ).reset_index()

    # Segment based on total spend
    # Why these thresholds? They should be calibrated to your business
    conditions = [
        customer_stats["total_spent"] >= 1000,
        customer_stats["total_spent"] >= 500,
        customer_stats["total_spent"] >= 100,
    ]
    labels = ["platinum", "gold", "silver"]
    customer_stats["segment"] = pd.np.select(conditions, labels, default="bronze")

    return customer_stats
```

### 5.2 The Complete Asset Graph

```
                    ┌──────────────┐
                    │ raw_products  │  (bronze)
                    └──────┬───────┘
                           │
┌──────────────┐    ┌──────▼───────────┐    ┌────────────────────────┐
│  raw_orders  │───→│ enriched_orders  │───→│ daily_category_revenue │
│  (bronze)    │    │    (silver)       │    │        (gold)          │
└──────────────┘    └──────┬───────────┘    └────────────────────────┘
                           │
                    ┌──────▼───────────┐
                    │customer_segments │
                    │     (gold)       │
                    └──────────────────┘
```

---

## 6. Partitioned Assets and Incremental Processing

### 6.1 Why Partitions?

In production, you rarely materialize an entire dataset every run. Instead, you process data in **partitions** — typically by date, region, or category. This enables:

- **Incremental processing**: Only process new/changed data ($O(\Delta n)$ instead of $O(n)$ per run)
- **Targeted backfills**: Re-process specific date ranges without touching everything
- **Parallel execution**: Different partitions can run simultaneously
- **Cost control**: Process only what's needed (especially important in cloud)

### 6.2 Time-Partitioned Assets

```python
import dagster as dg
import pandas as pd
from datetime import datetime

# Define a daily partition scheme
# Why DailyPartitionsDefinition?
# - Most data pipelines operate on daily batches
# - Each partition represents one day of data
# - Dagster tracks materialization status per partition

daily_partitions = dg.DailyPartitionsDefinition(
    start_date="2024-01-01",
    # Why set a start_date?
    # - Defines the earliest partition Dagster will recognize
    # - Prevents accidental backfills to the dawn of time
    # - Aligns with when your data source started collecting data
)


@dg.asset(
    partitions_def=daily_partitions,
    group_name="bronze",
    description="Daily order ingestion, partitioned by order date",
)
def daily_raw_orders(context: dg.AssetExecutionContext) -> pd.DataFrame:
    """Ingest orders for a specific date partition.

    Why partition at ingestion?
      - Source query is bounded: WHERE date = '2024-01-15'
      - No need to re-read the entire table
      - Backfill a specific day = re-run that one partition
    """
    # context.partition_key gives us the date string, e.g., "2024-01-15"
    partition_date = context.partition_key
    context.log.info(f"Ingesting orders for {partition_date}")

    # In production: query with WHERE clause on partition_date
    return pd.DataFrame({
        "order_id": range(100),
        "amount": [50.0] * 100,
        "order_date": [partition_date] * 100,
    })


@dg.asset(
    partitions_def=daily_partitions,
    group_name="silver",
)
def daily_cleaned_orders(
    context: dg.AssetExecutionContext,
    daily_raw_orders: pd.DataFrame,
) -> pd.DataFrame:
    """Clean orders for a specific partition.

    Key insight: When Dagster materializes partition "2024-01-15" for this asset,
    it automatically loads partition "2024-01-15" from daily_raw_orders upstream.
    Partition alignment is automatic!
    """
    partition_date = context.partition_key
    df = daily_raw_orders.copy()
    df = df[df["amount"] > 0]  # Remove invalid amounts
    context.log.info(
        f"Cleaned {len(df)} orders for {partition_date} "
        f"(dropped {len(daily_raw_orders) - len(df)} invalid)"
    )
    return df
```

### 6.3 Multi-Dimensional Partitions

For complex scenarios, Dagster supports multi-dimensional partitions:

```python
import dagster as dg

# Why multi-dimensional partitions?
# - Data varies by both time AND category
# - Enable backfills by date, by region, or by both
# - Example: re-process all "europe" data for January

region_time_partitions = dg.MultiPartitionsDefinition({
    "date": dg.DailyPartitionsDefinition(start_date="2024-01-01"),
    "region": dg.StaticPartitionsDefinition(["us", "europe", "asia"]),
})


@dg.asset(partitions_def=region_time_partitions)
def regional_orders(context: dg.AssetExecutionContext) -> pd.DataFrame:
    """Orders partitioned by date AND region.

    Each materialization processes one (date, region) pair.
    Total partitions = days × regions. For 365 days × 3 regions = 1,095 partitions.
    Dagster tracks each independently.
    """
    keys = context.partition_key.keys_by_dimension
    date = keys["date"]
    region = keys["region"]
    context.log.info(f"Processing {region} orders for {date}")

    return pd.DataFrame({
        "order_id": range(50),
        "region": [region] * 50,
        "date": [date] * 50,
    })
```

### 6.4 Partition-to-Partition Mapping

When downstream assets aggregate across upstream partitions (e.g., weekly metrics from daily data), you use partition mappings:

```python
import dagster as dg

weekly_partitions = dg.WeeklyPartitionsDefinition(start_date="2024-01-01")

@dg.asset(
    partitions_def=weekly_partitions,
    ins={
        "daily_cleaned_orders": dg.AssetIn(
            partition_mapping=dg.TimeWindowPartitionMapping(
                # Why TimeWindowPartitionMapping?
                # - Maps one weekly partition to 7 daily upstream partitions
                # - Dagster loads all 7 days automatically
                start_offset=0,
                end_offset=0,
            ),
        ),
    },
)
def weekly_order_summary(daily_cleaned_orders: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily orders into weekly summaries.

    This asset reads 7 daily partitions and produces 1 weekly partition.
    Dagster handles the fan-in automatically via the partition mapping.
    """
    return daily_cleaned_orders.groupby("order_date").agg(
        daily_revenue=("amount", "sum"),
        daily_orders=("order_id", "count"),
    ).reset_index()
```

---

## 7. Sensors and Schedules

### 7.1 Schedules — Time-Based Triggers

```python
import dagster as dg

# Simple cron-based schedule
daily_refresh = dg.ScheduleDefinition(
    name="daily_asset_refresh",
    # Why target the asset selection instead of a job?
    # - Directly schedule asset materializations
    # - No need to create a separate job object
    target=dg.AssetSelection.groups("bronze", "silver", "gold"),
    cron_schedule="0 6 * * *",    # Every day at 6 AM UTC
    default_status=dg.DefaultScheduleStatus.RUNNING,
)


# Schedule with partition awareness
@dg.schedule(
    cron_schedule="0 6 * * *",
    job_name="daily_etl_job",
)
def daily_partition_schedule(context: dg.ScheduleEvaluationContext):
    """Schedule that automatically targets yesterday's partition.

    Why yesterday? Data for day N is typically complete by day N+1.
    Running at 6 AM gives a buffer for late-arriving events.
    """
    yesterday = (context.scheduled_execution_time - timedelta(days=1)).strftime("%Y-%m-%d")
    return dg.RunRequest(
        partition_key=yesterday,
        tags={"source": "daily_schedule"},
    )
```

### 7.2 Sensors — Event-Based Triggers

Sensors poll for external conditions and trigger runs when conditions are met.

```python
import dagster as dg
from pathlib import Path

@dg.sensor(
    job_name="ingest_new_files_job",
    minimum_interval_seconds=60,  # Poll every 60 seconds
)
def new_file_sensor(context: dg.SensorEvaluationContext):
    """Watch for new files in the landing zone and trigger ingestion.

    Why sensors instead of schedules?
      - Data arrives at unpredictable times (partner uploads, API pushes)
      - Event-driven: process immediately when data appears
      - More efficient than running on a fixed schedule "just in case"
    """
    landing_zone = Path("/data/landing/orders/")
    last_mtime = float(context.cursor or "0")

    new_files = []
    max_mtime = last_mtime

    # Using modification time (mtime) as the cursor avoids maintaining a
    # separate database of processed files. The trade-off: if a file is
    # modified in-place (same mtime), it won't be reprocessed.
    for f in landing_zone.glob("*.csv"):
        if f.stat().st_mtime > last_mtime:
            new_files.append(str(f))
            max_mtime = max(max_mtime, f.stat().st_mtime)

    if new_files:
        context.log.info(f"Found {len(new_files)} new files")
        # Cursor is persisted across sensor ticks. If the daemon restarts,
        # the cursor resumes from the last committed value — no re-scanning.
        context.update_cursor(str(max_mtime))
        # run_key deduplicates: if the same key is yielded again (e.g.,
        # sensor tick runs twice before the run starts), Dagster skips
        # the duplicate instead of launching a second run.
        yield dg.RunRequest(
            run_key=f"new_files_{max_mtime}",
            run_config={
                "ops": {
                    "ingest": {"config": {"files": new_files}}
                }
            },
        )


# Freshness-based sensor: trigger when an asset is stale
@dg.freshness_policy_sensor(
    asset_selection=dg.AssetSelection.groups("gold"),
)
def freshness_sensor(context: dg.FreshnessPolicySensorContext):
    """Automatically materialize gold assets that have gone stale.

    This is Dagster's DECLARATIVE scheduling:
      - Instead of "run at 6 AM", you say "this asset should be fresh within 2 hours"
      - Dagster checks freshness and triggers materialization when needed
    """
    pass  # The sensor framework handles the logic
```

---

## 8. Testing in Dagster

One of Dagster's strongest advantages is its testing story. Assets are plain Python functions — you can test them with `pytest` without any orchestration infrastructure.

### 8.1 Unit Testing Assets

```python
import pytest
import pandas as pd

# Testing an asset is as simple as calling the function!

def test_cleaned_orders():
    """Test that cleaned_orders removes non-completed orders."""
    # Arrange: create test input
    raw = pd.DataFrame({
        "order_id": [1, 2, 3],
        "customer_id": [101, 102, 103],
        "amount": [100.0, 200.0, 50.0],
        "status": ["completed", "refunded", "completed"],
        "created_at": pd.date_range("2024-01-01", periods=3),
    })

    # Act: call the asset function directly
    # Why can we do this? Because an @asset is just a Python function!
    result = cleaned_orders(raw)

    # Assert
    assert len(result) == 2              # Refunded order removed
    assert "refunded" not in result["status"].values
    assert result["amount"].dtype == float


def test_order_metrics_aggregation():
    """Test that order_metrics correctly aggregates per customer."""
    cleaned = pd.DataFrame({
        "order_id": [1, 2, 3],
        "customer_id": [101, 101, 102],
        "amount": [100.0, 200.0, 50.0],
        "status": ["completed"] * 3,
        "created_at": pd.date_range("2024-01-01", periods=3),
    })

    result = order_metrics(cleaned)

    assert len(result) == 2  # Two unique customers
    cust_101 = result[result["customer_id"] == 101].iloc[0]
    assert cust_101["total_orders"] == 2
    assert cust_101["total_revenue"] == 300.0
    assert cust_101["avg_order_value"] == 150.0
```

### 8.2 Testing with Resources (Mocking)

```python
import dagster as dg

def test_revenue_report_with_mock_resources():
    """Test an asset that depends on external resources.

    Why mock resources?
      - Don't need a real database or S3 bucket in tests
      - Tests run in milliseconds, not seconds
      - Deterministic: no flaky tests from network issues
    """
    # Create mock resources
    mock_db = MockDatabaseResource(data=[{"revenue": 1000}])
    mock_s3 = MockS3Resource()

    # Execute the asset with mock resources
    result = dg.materialize(
        assets=[revenue_report],
        resources={
            "database": mock_db,
            "s3": mock_s3,
        },
    )

    assert result.success
    assert mock_s3.last_written_path == "reports/revenue/latest.parquet"


def test_full_pipeline_integration():
    """Test the entire asset graph end-to-end.

    Why materialize() for integration tests?
      - Runs the full asset graph in the correct order
      - Uses IO Managers to pass data between assets
      - Validates the complete pipeline, not just individual assets
    """
    result = dg.materialize(
        assets=[raw_orders, cleaned_orders, order_metrics],
        resources={
            "io_manager": dg.mem_io_manager,
            # Why mem_io_manager?
            # - Stores assets in memory (no disk I/O)
            # - Perfect for fast integration tests
            # - No cleanup needed
        },
    )

    assert result.success
    # Check that all assets materialized
    assert result.output_for_node("raw_orders") is not None
    assert result.output_for_node("order_metrics") is not None
```

---

## 9. Dagster Cloud vs OSS Deployment

### 9.1 Deployment Options

```python
"""
Dagster Deployment Comparison:

┌─────────────────────────────────────────────────────────────────────┐
│                    Dagster OSS (Self-Hosted)                        │
├─────────────────────────────────────────────────────────────────────┤
│ Components you manage:                                              │
│ ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│ │ Dagster   │  │ Dagster   │  │PostgreSQL│  │ User Code│            │
│ │ Webserver │  │ Daemon    │  │ (storage)│  │ (gRPC)   │            │
│ └──────────┘  └──────────┘  └──────────┘  └──────────┘            │
│                                                                     │
│ Pros:                          Cons:                                │
│ ✓ Full control                 ✗ Operational overhead               │
│ ✓ No vendor lock-in            ✗ Scaling is your problem           │
│ ✓ Free                         ✗ No built-in alerting/insights     │
│ ✓ Air-gapped environments      ✗ Must manage upgrades              │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    Dagster Cloud (Managed)                           │
├─────────────────────────────────────────────────────────────────────┤
│ Dagster manages:    │  You manage:                                   │
│ ┌──────────┐        │  ┌──────────────┐                              │
│ │ Webserver │        │  │ User Code    │ (runs in YOUR infra)        │
│ │ Daemon    │        │  │ (Hybrid or   │                              │
│ │ Storage   │        │  │  Serverless) │                              │
│ │ Monitoring│        │  └──────────────┘                              │
│ └──────────┘        │                                                │
│                                                                     │
│ Deployment modes:                                                   │
│ 1. Hybrid: Agent runs in your cloud, code stays in your VPC        │
│ 2. Serverless: Dagster runs everything (simplest)                  │
│                                                                     │
│ Pros:                          Cons:                                │
│ ✓ Zero ops overhead            ✗ Cost (pay per compute)             │
│ ✓ Built-in alerting            ✗ Less control over infra           │
│ ✓ Branch deployments           ✗ Internet connectivity required     │
│ ✓ Insights & cost tracking                                          │
└─────────────────────────────────────────────────────────────────────┘
"""
```

### 9.2 OSS Deployment with Docker Compose

```yaml
# docker-compose.yaml for Dagster OSS
# Why Docker Compose? Good for small teams and dev/staging environments.
# For production at scale, use Kubernetes (Helm chart: dagster/dagster).

version: "3.8"
services:
  # The webserver serves the Dagster UI (asset graph, run history, logs).
  # It is stateless and can be scaled horizontally behind a load balancer.
  dagster-webserver:
    image: dagster/dagster-k8s:latest
    command: ["dagster-webserver", "-h", "0.0.0.0", "-p", "3000"]
    ports:
      - "3000:3000"
    environment:
      DAGSTER_PG_URL: "postgresql://dagster:dagster@postgres:5432/dagster"
    depends_on:
      - postgres

  # The daemon handles schedules, sensors, and run queuing. Unlike the
  # webserver, the daemon must be a SINGLE instance — running multiple
  # daemons causes duplicate schedule triggers and sensor evaluations.
  dagster-daemon:
    image: dagster/dagster-k8s:latest
    command: ["dagster-daemon", "run"]
    environment:
      DAGSTER_PG_URL: "postgresql://dagster:dagster@postgres:5432/dagster"
    depends_on:
      - postgres

  # Dagster stores run metadata, event logs, and schedule state in Postgres.
  # SQLite is supported for local dev but lacks concurrent write safety,
  # so always use Postgres when running webserver + daemon together.
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: dagster
      POSTGRES_PASSWORD: dagster
      POSTGRES_DB: dagster
    volumes:
      - dagster-pg-data:/var/lib/postgresql/data

volumes:
  dagster-pg-data:
```

---

## 10. Integration with dbt, Spark, and pandas

### 10.1 dagster-dbt: First-Class dbt Support

Dagster's dbt integration is one of its strongest selling points. Each dbt model automatically becomes a Dagster asset, creating a unified lineage graph.

```python
from dagster_dbt import DbtCliResource, dbt_assets
from dagster import AssetExecutionContext, Definitions
from pathlib import Path

# Point to your dbt project
DBT_PROJECT_DIR = Path(__file__).parent / "dbt_project"

# Why parse the dbt manifest?
# - Dagster reads dbt's manifest.json to discover all models
# - Each model becomes a Dagster asset automatically
# - Dependencies between models become asset dependencies
# - dbt tests become Dagster asset checks

@dbt_assets(manifest=DBT_PROJECT_DIR / "target" / "manifest.json")
def my_dbt_assets(context: AssetExecutionContext, dbt: DbtCliResource):
    """All dbt models as Dagster assets.

    Why use @dbt_assets instead of running dbt manually?
      - Unified lineage: Python assets + dbt models in one graph
      - Selective execution: materialize specific dbt models
      - Observation: Dagster tracks freshness of dbt models
    """
    yield from dbt.cli(["build"], context=context).stream()


# Combine Python assets with dbt assets
@dg.asset(
    deps=["stg_orders"],  # This is a dbt model name!
    description="ML features derived from dbt staging tables",
)
def ml_features(database: DatabaseResource) -> pd.DataFrame:
    """Compute ML features from dbt-transformed data.

    Why mix Python and dbt?
      - dbt excels at SQL transformations (staging, joins, aggregations)
      - Python excels at ML feature engineering (embeddings, custom logic)
      - Dagster unifies both in a single dependency graph
    """
    staged = database.execute_query("SELECT * FROM stg_orders")
    return compute_features(pd.DataFrame(staged))
```

### 10.2 Spark Integration

```python
from dagster_spark import SparkResource
import dagster as dg

@dg.asset(
    group_name="silver",
    description="Large-scale order cleaning via Spark",
)
def spark_cleaned_orders(
    context: dg.AssetExecutionContext,
    spark: SparkResource,
) -> None:
    """Clean orders using Spark for large datasets.

    Why Spark for this asset?
      - When data volume exceeds single-machine memory (>10 GB)
      - When you need distributed joins or aggregations
      - Dagster orchestrates; Spark does the heavy lifting
    """
    session = spark.spark_session

    raw = session.read.parquet("s3://data-lake/bronze/orders/")

    # Chaining multiple filters instead of combining with AND is functionally
    # identical but more readable. Spark's optimizer (Catalyst) automatically
    # merges them into a single predicate pushdown.
    cleaned = (
        raw
        .filter("status = 'completed'")
        .filter("amount > 0")
        # dropDuplicates on order_id keeps an arbitrary row per key.
        # If you need the LATEST version, add an orderBy before dedup
        # or use a window function instead.
        .dropDuplicates(["order_id"])
        .withColumn("processed_at", dg.F.current_timestamp())
    )

    # mode("overwrite") replaces the entire Silver table. For incremental
    # processing, use Delta Lake MERGE instead — but full overwrite is
    # acceptable when the dataset fits in a single Spark job (<100 GB).
    cleaned.write.mode("overwrite").parquet("s3://data-lake/silver/orders/")
    context.log.info(f"Wrote {cleaned.count()} cleaned orders")
```

---

## Summary

```
Dagster Key Concepts:
─────────────────────
Asset           = "What data should exist?"
Op              = "What computation should run?"
Graph           = "How do ops connect?"
Job             = "When/how to execute a graph?"
Resource        = "What infrastructure do I need?"
IO Manager      = "How/where to store asset data?"
Partition       = "How to slice data for incremental processing?"
Sensor          = "When should I react to external events?"
Schedule        = "When should I run on a timer?"
Definitions     = "The complete registry of my Dagster project"

When to choose Dagster over Airflow:
  ✓ Data-centric pipelines (analytics, ML features, ELT)
  ✓ Strong testing requirements
  ✓ dbt integration needed
  ✓ Greenfield project with a small team
  ✓ Asset freshness tracking matters

When to stick with Airflow:
  ✓ Large existing Airflow investment
  ✓ General-purpose task orchestration (non-data)
  ✓ Need for extensive provider ecosystem (1000+ operators)
  ✓ Team already expert in Airflow
```

---

## Exercises

### Exercise 1: Build Your Own Asset Pipeline

Create a Dagster asset pipeline for a blog analytics platform:

1. Define a `raw_page_views` source asset that generates mock page view data (columns: `page_url`, `user_id`, `timestamp`, `session_id`)
2. Define a `cleaned_page_views` asset that removes bot traffic (user agents containing "bot") and deduplicates by session
3. Define a `page_popularity` gold asset that ranks pages by unique visitor count
4. Add metadata and descriptions to each asset
5. Test all three assets using `pytest`

### Exercise 2: Partitioned Incremental Pipeline

Extend Exercise 1 with time partitioning:

1. Add `DailyPartitionsDefinition` to `raw_page_views` and `cleaned_page_views`
2. Create a `weekly_page_report` asset with `WeeklyPartitionsDefinition` and a `TimeWindowPartitionMapping` from the daily assets
3. Write a schedule that materializes yesterday's partitions at 2 AM UTC
4. Simulate a backfill for the last 7 days using `dagster dev`

### Exercise 3: Resource Injection and Testing

Practice dependency injection:

1. Create a `PostgresResource` and an `S3Resource` (mock implementations)
2. Create an asset `user_profiles` that reads from Postgres and writes to S3
3. Write a test that uses mock resources to verify the asset logic
4. Write a second test using `dg.materialize()` with `mem_io_manager`

### Exercise 4: Dagster-dbt Integration Design

Design (on paper or code) a hybrid pipeline:

1. dbt handles: staging models (`stg_users`, `stg_events`), marts (`fct_user_activity`)
2. Python handles: ML feature computation, model scoring
3. Draw the asset dependency graph showing both dbt and Python assets
4. Identify which assets should be partitioned and why

### Exercise 5: Sensor-Driven Pipeline

Build an event-driven ingestion pipeline:

1. Create a sensor that watches a directory for new CSV files
2. When new files appear, trigger a job that ingests, cleans, and aggregates the data
3. Use `context.cursor` to track which files have been processed
4. Handle edge cases: empty files, duplicate files, malformed CSV

---

## References

- [Dagster Documentation](https://docs.dagster.io/)
- [Dagster GitHub Repository](https://github.com/dagster-io/dagster)
- [dagster-dbt Integration Guide](https://docs.dagster.io/integrations/dbt)
- [Software-Defined Assets Concepts](https://docs.dagster.io/concepts/assets/software-defined-assets)
- [Dagster Cloud](https://dagster.io/cloud)
- [Dagster vs Airflow — Official Comparison](https://dagster.io/vs/dagster-vs-airflow)
- [Dagster University (Free Courses)](https://courses.dagster.io/)

---

[← Previous: 19. Lakehouse Practical Patterns](19_Lakehouse_Practical_Patterns.md) | [Next: 21. Data Versioning and Data Contracts →](21_Data_Versioning_and_Contracts.md)
