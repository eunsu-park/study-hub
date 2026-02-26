# Practical Pipeline Project

## Learning Objectives

After completing this lesson, you will be able to:

1. Architect an end-to-end data pipeline integrating multiple tools (Airflow, Spark, dbt, Great Expectations) to satisfy real-world business requirements.
2. Implement a medallion architecture (bronze, silver, gold layers) on a cloud data lake to progressively refine raw data into analytics-ready datasets.
3. Orchestrate multi-step pipeline DAGs in Airflow that coordinate ingestion, transformation, quality validation, and alerting tasks.
4. Design data ingestion patterns for heterogeneous sources including relational databases, object storage (S3), and event streams (Kafka).
5. Apply monitoring and alerting strategies to detect pipeline failures and data quality regressions, and route notifications via Slack or email.
6. Evaluate the trade-offs of batch vs. streaming ingestion when building production analytics pipelines.

---

## Overview

In this lesson, we'll integrate all the technologies learned so far to build a real data pipeline. We'll design an end-to-end pipeline using Airflow for orchestration, Spark for large-scale processing, dbt for transformations, and Great Expectations for quality validation.

---

## 1. Project Overview

### 1.1 Scenario

```
┌────────────────────────────────────────────────────────────────┐
│                    E-Commerce Analytics Pipeline                │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   Business Requirements:                                        │
│   - Daily sales analysis dashboard                             │
│   - Customer segmentation                                      │
│   - Inventory optimization alerts                              │
│                                                                │
│   Data Sources:                                                 │
│   - PostgreSQL: Orders, customers, products                     │
│   - S3: Clickstream logs (JSON)                                │
│   - Kafka: Real-time inventory events                          │
│                                                                │
│   Outputs:                                                      │
│   - Data Warehouse: Snowflake/BigQuery                         │
│   - BI Dashboard: Looker/Tableau                               │
│   - Alert System: Slack/Email                                  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 1.2 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Pipeline Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Data Sources                                                  │
│   ┌────────┐ ┌────────┐ ┌────────┐                             │
│   │PostgreSQL│ S3 Logs│ │ Kafka  │                              │
│   └────┬───┘ └───┬────┘ └───┬────┘                             │
│        │         │          │                                   │
│        └─────────┴──────────┘                                   │
│                   ↓                                             │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                    Airflow                               │  │
│   │    (Orchestration)                                       │  │
│   └─────────────────────────────────────────────────────────┘  │
│                   ↓                                             │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                 Data Lake (S3)                           │  │
│   │   ┌─────────┐  ┌─────────┐  ┌─────────┐                 │  │
│   │   │ Bronze  │→│ Silver  │→│  Gold   │                  │  │
│   │   │  (Raw)  │  │(Cleaned)│  │(Curated)│                  │  │
│   │   └─────────┘  └─────────┘  └─────────┘                 │  │
│   │                                                            │  │
│   │   Why three layers?                                        │  │
│   │   - Bronze: immutable raw data for auditability — if a     │  │
│   │     transformation bug corrupts Silver/Gold, you can       │  │
│   │     always reprocess from Bronze without re-extracting     │  │
│   │   - Silver: cleaned, deduplicated, type-standardized —     │  │
│   │     a single source of validated truth that multiple       │  │
│   │     Gold tables can build upon                             │  │
│   │   - Gold: business-ready aggregates and dimensional        │  │
│   │     models optimized for fast BI queries — avoids          │  │
│   │     expensive ad-hoc JOINs on raw data                     │  │
│   └─────────────────────────────────────────────────────────┘  │
│                   ↓                                             │
│   ┌───────────────────────────────────────────────────────────┐│
│   │ Spark (Processing) │ dbt (Transform) │ GE (Quality)      ││
│   └───────────────────────────────────────────────────────────┘│
│                   ↓                                             │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │               Data Warehouse                             │  │
│   │         (Snowflake / BigQuery)                           │  │
│   └─────────────────────────────────────────────────────────┘  │
│                   ↓                                             │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐                      │
│   │ BI Tool  │ │ ML Models│ │ Alerts   │                      │
│   └──────────┘ └──────────┘ └──────────┘                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Project Structure

### 2.1 Directory Structure

```
ecommerce_pipeline/
├── airflow/
│   ├── dags/
│   │   ├── daily_etl_dag.py
│   │   ├── hourly_streaming_dag.py
│   │   └── data_quality_dag.py
│   └── plugins/
│       └── custom_operators.py
│
├── spark/
│   ├── jobs/
│   │   ├── extract_postgres.py
│   │   ├── process_clickstream.py
│   │   └── aggregate_daily.py
│   └── utils/
│       └── spark_utils.py
│
├── dbt/
│   ├── dbt_project.yml
│   ├── models/
│   │   ├── staging/
│   │   │   ├── stg_orders.sql
│   │   │   ├── stg_customers.sql
│   │   │   └── stg_products.sql
│   │   ├── intermediate/
│   │   │   └── int_order_items.sql
│   │   └── marts/
│   │       ├── fct_orders.sql
│   │       ├── dim_customers.sql
│   │       └── agg_daily_sales.sql
│   └── tests/
│       └── assert_positive_amounts.sql
│
├── great_expectations/
│   ├── expectations/
│   │   ├── orders_suite.json
│   │   └── customers_suite.json
│   └── checkpoints/
│       └── daily_checkpoint.yml
│
├── docker/
│   ├── docker-compose.yml
│   └── Dockerfile.spark
│
├── tests/
│   ├── test_spark_jobs.py
│   └── test_dbt_models.py
│
└── requirements.txt
```

---

## 3. Airflow DAG Implementation

### 3.1 Main ETL DAG

```python
# airflow/dags/daily_etl_dag.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.providers.dbt.cloud.operators.dbt import DbtCloudRunJobOperator
from airflow.utils.task_group import TaskGroup

# default_args are inherited by every task in the DAG, reducing boilerplate.
# retries=2 with 5-min delay handles transient failures (network blips,
# temporary resource contention) without human intervention.
# depends_on_past=False means each daily run is independent — a Monday
# failure won't block Tuesday's run from starting.
default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'email': ['data-alerts@company.com'],
    'email_on_failure': True,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='daily_ecommerce_pipeline',
    default_args=default_args,
    description='Daily e-commerce data pipeline',
    schedule_interval='0 6 * * *',  # Daily at 6 AM
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['production', 'etl', 'daily'],
    max_active_runs=1,
) as dag:

    start = EmptyOperator(task_id='start')

    # ============================================
    # Extract: Extract from data sources
    # ============================================
    # TaskGroup organizes related tasks into a collapsible unit in the Airflow UI.
    # This is purely organizational (not a scheduling boundary) — all tasks inside
    # still execute independently. Without grouping, a pipeline with 15+ tasks
    # becomes unreadable in the Graph view.
    with TaskGroup(group_id='extract') as extract_group:

        extract_orders = SparkSubmitOperator(
            task_id='extract_orders',
            application='/opt/spark/jobs/extract_postgres.py',
            conn_id='spark_default',
            application_args=[
                '--table', 'orders',
                '--date', '{{ ds }}',
                '--output', 's3://data-lake/bronze/orders/{{ ds }}/'
            ],
        )

        extract_customers = SparkSubmitOperator(
            task_id='extract_customers',
            application='/opt/spark/jobs/extract_postgres.py',
            conn_id='spark_default',
            application_args=[
                '--table', 'customers',
                '--output', 's3://data-lake/bronze/customers/'
            ],
        )

        extract_products = SparkSubmitOperator(
            task_id='extract_products',
            application='/opt/spark/jobs/extract_postgres.py',
            conn_id='spark_default',
            application_args=[
                '--table', 'products',
                '--output', 's3://data-lake/bronze/products/'
            ],
        )

        extract_clickstream = SparkSubmitOperator(
            task_id='extract_clickstream',
            application='/opt/spark/jobs/process_clickstream.py',
            conn_id='spark_default',
            application_args=[
                '--date', '{{ ds }}',
                '--input', 's3://raw-logs/clickstream/{{ ds }}/',
                '--output', 's3://data-lake/bronze/clickstream/{{ ds }}/'
            ],
        )

    # ============================================
    # Quality Check: Bronze layer quality validation
    # ============================================
    # Quality gate between Bronze and Silver: catches data issues (missing files,
    # schema drift, null spikes) before they propagate downstream. Fixing bad data
    # in Bronze is 10x cheaper than fixing corrupted Gold aggregates that already
    # fed dashboards and ML models.
    with TaskGroup(group_id='quality_bronze') as quality_bronze:

        def run_great_expectations(checkpoint_name: str, **kwargs):
            import great_expectations as gx
            context = gx.get_context()
            result = context.run_checkpoint(checkpoint_name=checkpoint_name)
            if not result.success:
                raise ValueError(f"Quality check failed: {checkpoint_name}")

        check_orders = PythonOperator(
            task_id='check_orders_quality',
            python_callable=run_great_expectations,
            op_kwargs={'checkpoint_name': 'bronze_orders_checkpoint'},
        )

        check_customers = PythonOperator(
            task_id='check_customers_quality',
            python_callable=run_great_expectations,
            op_kwargs={'checkpoint_name': 'bronze_customers_checkpoint'},
        )

    # ============================================
    # Transform: Create Silver layer with Spark
    # ============================================
    # Silver layer = cleaned, deduplicated, type-standardized data.
    # Spark is used here (not dbt) because Spark handles large-scale file-based
    # transformations on the data lake, while dbt operates on the warehouse SQL engine.
    with TaskGroup(group_id='transform_spark') as transform_spark:

        process_orders = SparkSubmitOperator(
            task_id='process_orders',
            application='/opt/spark/jobs/process_orders.py',
            application_args=[
                '--input', 's3://data-lake/bronze/orders/{{ ds }}/',
                '--output', 's3://data-lake/silver/orders/{{ ds }}/'
            ],
        )

        aggregate_daily = SparkSubmitOperator(
            task_id='aggregate_daily',
            application='/opt/spark/jobs/aggregate_daily.py',
            application_args=[
                '--date', '{{ ds }}',
                '--output', 's3://data-lake/silver/daily_aggregates/{{ ds }}/'
            ],
        )

    # ============================================
    # Transform: Create Gold layer with dbt
    # ============================================
    # Gold layer = business-ready aggregates and dimensional models.
    # dbt handles warehouse-side SQL transformations (staging → marts),
    # while Spark handled the file-based lake transformations above.
    # This separation lets each tool operate in its strength zone.
    with TaskGroup(group_id='transform_dbt') as transform_dbt:

        def run_dbt_command(command: str, **kwargs):
            # Subprocess execution because dbt CLI is the standard interface.
            # In production, prefer DbtCloudRunJobOperator or the dbt Python API
            # to avoid shell injection risks and get better error reporting.
            import subprocess
            result = subprocess.run(
                f"cd /opt/dbt && dbt {command} --profiles-dir /opt/dbt",
                shell=True,
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                raise Exception(f"dbt failed: {result.stderr}")
            print(result.stdout)

        dbt_run = PythonOperator(
            task_id='dbt_run',
            python_callable=run_dbt_command,
            op_kwargs={'command': 'run --select staging marts'},
        )

        # dbt test runs AFTER dbt run to validate the freshly materialized models.
        # This catches data quality regressions introduced by the transformation
        # logic itself (e.g., a JOIN producing unexpected NULLs).
        dbt_test = PythonOperator(
            task_id='dbt_test',
            python_callable=run_dbt_command,
            op_kwargs={'command': 'test'},
        )

        dbt_run >> dbt_test

    # ============================================
    # Quality Check: Gold layer quality validation
    # ============================================
    # Second quality gate at the Gold layer catches issues introduced by
    # transformations (incorrect joins, aggregation bugs). Having quality
    # gates at both Bronze AND Gold provides defense-in-depth: Bronze
    # catches source problems, Gold catches transformation problems.
    quality_gold = PythonOperator(
        task_id='quality_gold',
        python_callable=run_great_expectations,
        op_kwargs={'checkpoint_name': 'gold_checkpoint'},
    )

    # ============================================
    # Notify: Completion notification
    # ============================================
    def send_completion_notification(**kwargs):
        import requests
        webhook_url = "https://hooks.slack.com/services/xxx"
        message = {
            "text": f"Daily pipeline completed for {kwargs['ds']}"
        }
        requests.post(webhook_url, json=message)

    notify = PythonOperator(
        task_id='notify_completion',
        python_callable=send_completion_notification,
    )

    end = EmptyOperator(task_id='end')

    # Task dependencies follow the medallion architecture:
    # Extract → Quality(Bronze) → Transform(Silver) → Transform(Gold) → Quality(Gold) → Notify
    # This linear chain ensures data quality is validated before each layer transition,
    # preventing bad data from propagating to downstream consumers.
    start >> extract_group >> quality_bronze >> transform_spark >> transform_dbt >> quality_gold >> notify >> end
```

---

## 4. Spark Processing Jobs

### 4.1 Data Extraction Job

```python
# spark/jobs/extract_postgres.py
from pyspark.sql import SparkSession
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--table', required=True)
    parser.add_argument('--date', required=False)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    spark = SparkSession.builder \
        .appName(f"Extract {args.table}") \
        .getOrCreate()

    # JDBC read configuration
    # In production, store credentials in a secrets manager (AWS Secrets Manager,
    # Vault) and inject via environment variables — never hardcode passwords.
    jdbc_url = "jdbc:postgresql://postgres:5432/ecommerce"
    properties = {
        "user": "postgres",
        "password": "password",
        "driver": "org.postgresql.Driver"
    }

    # Incremental extraction when date is specified: only pull rows updated on
    # that date, avoiding full table scans that would be expensive on large tables.
    # The subquery alias "AS t" is required by Spark's JDBC reader for pushdown queries.
    if args.date:
        query = f"""
            (SELECT * FROM {args.table}
             WHERE DATE(updated_at) = '{args.date}') AS t
        """
    else:
        query = args.table

    # Read data
    df = spark.read.jdbc(
        url=jdbc_url,
        table=query,
        properties=properties
    )

    # Save to Bronze layer as Parquet.
    # mode("overwrite") is safe here because each date partition is written to
    # its own path (via Airflow templating), so we only overwrite that day's data.
    df.write \
        .mode("overwrite") \
        .parquet(args.output)

    print(f"Extracted {df.count()} rows from {args.table}")
    spark.stop()


if __name__ == "__main__":
    main()
```

### 4.2 Clickstream Processing

```python
# spark/jobs/process_clickstream.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    # AQE (Adaptive Query Execution) dynamically adjusts shuffle partitions and
    # join strategies at runtime based on actual data sizes — eliminates the need
    # to manually tune spark.sql.shuffle.partitions for varying daily volumes.
    spark = SparkSession.builder \
        .appName("Process Clickstream") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()

    # Explicitly defining the schema avoids a costly schema-inference pass that
    # requires reading the entire JSON dataset twice. For large clickstream logs
    # (millions of events/day), this can save minutes of processing time.
    schema = StructType([
        StructField("event_id", StringType()),
        StructField("user_id", StringType()),
        StructField("session_id", StringType()),
        StructField("event_type", StringType()),
        StructField("page_url", StringType()),
        StructField("timestamp", TimestampType()),
        StructField("properties", MapType(StringType(), StringType())),
    ])

    # Read JSON with pre-defined schema (no inference overhead)
    df = spark.read.schema(schema).json(args.input)

    # Clean and transform:
    # - Null filters remove malformed events that would cause join failures downstream
    # - Extracting event_date/hour enables time-based partitioning for efficient queries
    # - getItem("product_id") flattens the nested properties map into a top-level column,
    #   which is much faster to query than repeatedly parsing nested JSON
    processed_df = df \
        .filter(col("event_id").isNotNull()) \
        .filter(col("user_id").isNotNull()) \
        .withColumn("event_date", to_date(col("timestamp"))) \
        .withColumn("event_hour", hour(col("timestamp"))) \
        .withColumn("product_id", col("properties").getItem("product_id")) \
        .dropDuplicates(["event_id"]) \
        .select(
            "event_id",
            "user_id",
            "session_id",
            "event_type",
            "page_url",
            "product_id",
            "event_date",
            "event_hour",
            "timestamp"
        )

    # Dual partitioning (date + hour) enables both daily batch queries and
    # hourly drill-down queries to prune efficiently. Over-partitioning
    # (e.g., by minute) would create too many small files, degrading read performance.
    processed_df.write \
        .mode("overwrite") \
        .partitionBy("event_date", "event_hour") \
        .parquet(args.output)

    print(f"Processed {processed_df.count()} events")
    spark.stop()


if __name__ == "__main__":
    main()
```

### 4.3 Daily Aggregation

```python
# spark/jobs/aggregate_daily.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    spark = SparkSession.builder \
        .appName("Daily Aggregation") \
        .getOrCreate()

    # Read Silver layer — customers and products are loaded as full snapshots
    # (no date partition) because they are slowly-changing dimensions;
    # orders are date-partitioned for incremental processing.
    orders = spark.read.parquet(f"s3://data-lake/silver/orders/{args.date}/")
    customers = spark.read.parquet("s3://data-lake/silver/customers/")
    products = spark.read.parquet("s3://data-lake/silver/products/")

    # Daily sales aggregation — pre-computing these aggregates in the Gold layer
    # means BI dashboards can read directly without running expensive GROUP BYs
    # on raw fact tables, reducing query latency from minutes to seconds.
    daily_sales = orders \
        .filter(col("order_date") == args.date) \
        .join(products, "product_id") \
        .groupBy(
            col("order_date"),
            col("category"),
            col("region")
        ) \
        .agg(
            count("order_id").alias("order_count"),
            sum("amount").alias("total_revenue"),
            avg("amount").alias("avg_order_value"),
            # countDistinct is more expensive than count but critical for
            # understanding customer reach vs order volume
            countDistinct("customer_id").alias("unique_customers")
        )

    # Customer segment aggregation — separate from daily_sales because
    # it has a different grain (segment-level vs category/region),
    # serving different business questions (segmentation vs product analytics)
    customer_segments = orders \
        .filter(col("order_date") == args.date) \
        .join(customers, "customer_id") \
        .groupBy("customer_segment") \
        .agg(
            count("order_id").alias("orders"),
            sum("amount").alias("revenue")
        )

    # Save
    daily_sales.write \
        .mode("overwrite") \
        .parquet(f"{args.output}/daily_sales/")

    customer_segments.write \
        .mode("overwrite") \
        .parquet(f"{args.output}/customer_segments/")

    spark.stop()


if __name__ == "__main__":
    main()
```

---

## 5. dbt Models

### 5.1 Staging Models

```sql
-- dbt/models/staging/stg_orders.sql
{{
    config(
        -- Materialized as view for fast iteration: rebuilds instantly during
        -- development. For production with high query volume, consider switching
        -- to table or incremental to avoid re-reading Silver on every query.
        materialized='view',
        schema='staging'
    )
}}

WITH source AS (
    SELECT * FROM {{ source('silver', 'orders') }}
),

cleaned AS (
    SELECT
        order_id,
        customer_id,
        product_id,
        -- Explicit CAST ensures consistent types regardless of how the source
        -- system serialized the data (some sources emit dates as strings)
        CAST(order_date AS DATE) AS order_date,
        CAST(amount AS DECIMAL(12, 2)) AS amount,
        CAST(quantity AS INT) AS quantity,
        status,
        CURRENT_TIMESTAMP AS loaded_at
    FROM source
    -- Filter out null PKs and non-positive amounts at the staging layer
    -- so all downstream models can trust these invariants without rechecking
    WHERE order_id IS NOT NULL
      AND amount > 0
)

SELECT * FROM cleaned
```

### 5.2 Mart Models

```sql
-- dbt/models/marts/fct_orders.sql
{{
    config(
        -- Incremental materialization processes only new data since last run,
        -- critical for fact tables that grow by millions of rows daily.
        -- A full rebuild (dbt run --full-refresh) can be forced when needed.
        materialized='incremental',
        unique_key='order_id',
        schema='marts',
        -- Day-level partitioning matches the pipeline's daily schedule;
        -- most BI queries filter by date range, so partition pruning
        -- delivers major cost savings on cloud warehouses.
        partition_by={
            'field': 'order_date',
            'data_type': 'date',
            'granularity': 'day'
        }
    )
}}

WITH orders AS (
    SELECT * FROM {{ ref('stg_orders') }}
),

customers AS (
    SELECT * FROM {{ ref('dim_customers') }}
),

products AS (
    SELECT * FROM {{ ref('dim_products') }}
)

SELECT
    o.order_id,
    o.order_date,

    -- Customer information — denormalized into the fact table for query
    -- convenience. BI tools can filter/group by segment and region without
    -- requiring users to write manual JOINs.
    o.customer_id,
    c.customer_name,
    c.customer_segment,
    c.region,

    -- Product information
    o.product_id,
    p.product_name,
    p.category,

    -- Metrics — computing cost and profit at the fact level enables direct
    -- aggregation in Gold-layer summaries without re-joining product costs
    o.quantity,
    o.amount AS order_amount,
    p.unit_cost * o.quantity AS cost_amount,
    o.amount - (p.unit_cost * o.quantity) AS profit_amount,

    -- Status
    o.status,

    -- Metadata
    o.loaded_at

-- LEFT JOINs preserve orders even when dimension data is missing (e.g.,
-- a new customer not yet in dim_customers). INNER JOIN would silently
-- drop these orders, causing revenue undercounting.
FROM orders o
LEFT JOIN customers c ON o.customer_id = c.customer_id
LEFT JOIN products p ON o.product_id = p.product_id

{% if is_incremental() %}
WHERE o.order_date > (SELECT MAX(order_date) FROM {{ this }})
{% endif %}
```

```sql
-- dbt/models/marts/agg_daily_sales.sql
{{
    config(
        materialized='table',
        schema='marts'
    )
}}

WITH orders AS (
    SELECT * FROM {{ ref('fct_orders') }}
)

SELECT
    order_date,
    category,
    region,
    customer_segment,

    -- Order metrics
    COUNT(DISTINCT order_id) AS total_orders,
    COUNT(DISTINCT customer_id) AS unique_customers,
    SUM(quantity) AS total_quantity,

    -- Revenue metrics
    SUM(order_amount) AS total_revenue,
    AVG(order_amount) AS avg_order_value,
    SUM(profit_amount) AS total_profit,

    -- Prevents division-by-zero when total revenue is 0 (e.g., a day with only
    -- cancelled orders or a newly launched region with no completed sales).
    -- NULLIF returns NULL instead of 0, making the division yield NULL rather
    -- than a database error, which is safer for downstream BI tools.
    ROUND(SUM(profit_amount) / NULLIF(SUM(order_amount), 0) * 100, 2) AS profit_margin_pct,

    -- Period comparison (with dbt_utils)
    -- {{ dbt_utils.date_spine(...) }}

    CURRENT_TIMESTAMP AS updated_at

FROM orders
GROUP BY
    order_date,
    category,
    region,
    customer_segment
```

---

## 6. Quality Validation

### 6.1 Great Expectations Suite

```python
# great_expectations/create_expectations.py
import great_expectations as gx

context = gx.get_context()

# Orders Suite
orders_suite = context.add_expectation_suite("orders_suite")
validator = context.get_validator(
    batch_request={"datasource": "orders_datasource", ...},
    expectation_suite_name="orders_suite"
)

# Basic validation
validator.expect_column_values_to_not_be_null("order_id")
validator.expect_column_values_to_be_unique("order_id")
validator.expect_column_values_to_not_be_null("customer_id")
validator.expect_column_values_to_not_be_null("amount")

# Value range
validator.expect_column_values_to_be_between("amount", min_value=0, max_value=1000000)
validator.expect_column_values_to_be_between("quantity", min_value=1, max_value=100)

# Accepted values
validator.expect_column_values_to_be_in_set(
    "status",
    ["pending", "processing", "shipped", "delivered", "cancelled"]
)

# Table level
validator.expect_table_row_count_to_be_between(min_value=1000, max_value=10000000)

# Referential integrity
# validator.expect_column_values_to_be_in_set(
#     "customer_id",
#     customer_ids_from_dim_table
# )

validator.save_expectation_suite(discard_failed_expectations=False)
```

### 6.2 dbt Tests

```yaml
# dbt/models/marts/_schema.yml
version: 2

models:
  - name: fct_orders
    description: "Orders fact table"
    tests:
      - dbt_utils.recency:
          datepart: day
          field: order_date
          interval: 1
    columns:
      - name: order_id
        tests:
          - unique
          - not_null
      - name: customer_id
        tests:
          - not_null
          - relationships:
              to: ref('dim_customers')
              field: customer_id
      - name: order_amount
        tests:
          - not_null
          - dbt_utils.expression_is_true:
              expression: ">= 0"
      - name: profit_amount
        tests:
          - dbt_utils.expression_is_true:
              expression: "<= order_amount"

  - name: agg_daily_sales
    tests:
      - dbt_utils.unique_combination_of_columns:
          combination_of_columns:
            - order_date
            - category
            - region
            - customer_segment
```

---

## 7. Monitoring and Alerts

### 7.1 Monitoring Dashboard

```python
# monitoring/metrics_collector.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import json

@dataclass
class PipelineMetrics:
    """Pipeline metrics"""
    pipeline_name: str
    run_date: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"
    records_processed: int = 0
    quality_score: float = 0.0
    errors: list = None

    def to_dict(self):
        return {
            "pipeline_name": self.pipeline_name,
            "run_date": self.run_date,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": (self.end_time - self.start_time).seconds if self.end_time else None,
            "status": self.status,
            "records_processed": self.records_processed,
            "quality_score": self.quality_score,
            "errors": self.errors or []
        }


def push_metrics_to_prometheus(metrics: PipelineMetrics):
    """Push metrics to Prometheus"""
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

    # Fresh registry per push to avoid stale metrics from previous runs.
    # Without this, metric labels from a failed pipeline could linger
    # and confuse Grafana dashboards.
    registry = CollectorRegistry()

    # Gauge (not Counter) because pipeline duration is a point-in-time value,
    # not an ever-increasing counter. Gauges allow Prometheus to track
    # whether runs are getting slower over time.
    duration = Gauge(
        'pipeline_duration_seconds',
        'Pipeline duration',
        ['pipeline_name'],
        registry=registry
    )
    duration.labels(pipeline_name=metrics.pipeline_name).set(
        (metrics.end_time - metrics.start_time).seconds if metrics.end_time else 0
    )

    records = Gauge(
        'pipeline_records_processed',
        'Records processed',
        ['pipeline_name'],
        registry=registry
    )
    records.labels(pipeline_name=metrics.pipeline_name).set(metrics.records_processed)

    # Quality score enables Grafana alerts on degradation trends — e.g.,
    # alert if quality drops below 95% for 3 consecutive runs
    quality = Gauge(
        'pipeline_quality_score',
        'Quality score',
        ['pipeline_name'],
        registry=registry
    )
    quality.labels(pipeline_name=metrics.pipeline_name).set(metrics.quality_score)

    # Push gateway is used (instead of pull) because pipeline jobs are short-lived
    # and may not be running when Prometheus scrapes. The push gateway acts as
    # an intermediary that holds metrics until Prometheus collects them.
    push_to_gateway('localhost:9091', job='data_pipeline', registry=registry)
```

### 7.2 Alert Configuration

```python
# monitoring/alerts.py
import requests
from typing import Optional

class AlertManager:
    """Alert management"""

    def __init__(self, slack_webhook: str, pagerduty_key: Optional[str] = None):
        self.slack_webhook = slack_webhook
        self.pagerduty_key = pagerduty_key

    def send_slack_alert(self, message: str, severity: str = "info"):
        """Slack notification"""
        # Color-coded attachments provide instant visual severity in Slack —
        # on-call engineers can triage red (error) vs orange (warning) at a glance
        # without reading the full message.
        color = {
            "info": "#36a64f",
            "warning": "#ffa500",
            "error": "#ff0000"
        }.get(severity, "#36a64f")

        payload = {
            "attachments": [{
                "color": color,
                "text": message,
                "footer": "Data Pipeline Alert"
            }]
        }
        requests.post(self.slack_webhook, json=payload)

    def send_pagerduty_alert(self, message: str):
        """PagerDuty notification (for critical issues)"""
        # PagerDuty is reserved for critical failures that need immediate human
        # response (e.g., pipeline down for >1 hour). Routing non-critical issues
        # to Slack only prevents alert fatigue and keeps PagerDuty signals meaningful.
        if not self.pagerduty_key:
            return

        payload = {
            "routing_key": self.pagerduty_key,
            "event_action": "trigger",
            "payload": {
                "summary": message,
                "severity": "critical",
                "source": "data-pipeline"
            }
        }
        requests.post(
            "https://events.pagerduty.com/v2/enqueue",
            json=payload
        )


# Use in Airflow
def alert_on_failure(context):
    """Alert on task failure"""
    alert_manager = AlertManager(
        slack_webhook="https://hooks.slack.com/services/xxx"
    )

    message = f"""
    Pipeline Failed!
    DAG: {context['dag'].dag_id}
    Task: {context['task'].task_id}
    Execution Date: {context['ds']}
    Error: {context.get('exception', 'Unknown')}
    """

    alert_manager.send_slack_alert(message, severity="error")
```

---

## 8. Deployment

### 8.1 Docker Compose

```yaml
# docker/docker-compose.yml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow
      POSTGRES_DB: airflow
    volumes:
      - postgres_data:/var/lib/postgresql/data

  airflow-webserver:
    image: apache/airflow:2.7.0
    depends_on:
      - postgres
    environment:
      AIRFLOW__CORE__EXECUTOR: CeleryExecutor
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
    volumes:
      - ./airflow/dags:/opt/airflow/dags
      - ./dbt:/opt/dbt
    ports:
      - "8080:8080"
    command: webserver

  airflow-scheduler:
    image: apache/airflow:2.7.0
    depends_on:
      - airflow-webserver
    volumes:
      - ./airflow/dags:/opt/airflow/dags
    command: scheduler

  spark-master:
    image: bitnami/spark:3.4
    environment:
      - SPARK_MODE=master
    ports:
      - "7077:7077"
      - "8081:8080"
    volumes:
      - ./spark/jobs:/opt/spark/jobs

  spark-worker:
    image: bitnami/spark:3.4
    environment:
      - SPARK_MODE=worker
      - SPARK_MASTER_URL=spark://spark-master:7077
    depends_on:
      - spark-master

volumes:
  postgres_data:
```

---

## Practice Problems

### Problem 1: Extend Pipeline
Add a streaming pipeline that processes real-time inventory events from Kafka and sends low-stock alerts.

### Problem 2: Quality Dashboard
Visualize daily data quality scores in a Grafana dashboard.

### Problem 3: Cost Optimization
Optimize Spark partition count and resource settings for large-scale data processing.

---

## Summary

Key integrations covered in this project:

| Tool | Role |
|------|------|
| **Airflow** | Pipeline orchestration |
| **Spark** | Large-scale data processing |
| **dbt** | SQL-based transformation |
| **Great Expectations** | Data quality validation |
| **Data Lake** | Layered storage |

---

## References

- [Airflow Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
- [Spark Performance Tuning](https://spark.apache.org/docs/latest/sql-performance-tuning.html)
- [dbt Best Practices](https://docs.getdbt.com/guides/best-practices)
- [Data Engineering Cookbook](https://github.com/andkret/Cookbook)

---

## Example Code

Runnable example project with Docker Compose, synthetic data, and all pipeline components:

**[`examples/Data_Engineering/practical_pipeline/`](../../../examples/Data_Engineering/practical_pipeline/)**

Includes Airflow DAGs, Spark jobs, dbt models, and Great Expectations suites — all adapted for local execution.
