# Data Quality and Governance

## Learning Objectives

After completing this lesson, you will be able to:

1. Define the six core dimensions of data quality (accuracy, completeness, consistency, timeliness, uniqueness, validity) and explain how each impacts downstream analytics.
2. Implement automated data quality checks using tools such as Great Expectations or dbt tests to detect anomalies in production pipelines.
3. Design a data governance framework that includes data cataloging, lineage tracking, and ownership policies.
4. Apply data quality metrics collection and alerting patterns to monitor pipeline health in real time.
5. Evaluate data observability platforms and compare their capabilities for detecting schema changes, null rate spikes, and distribution shifts.
6. Analyze the relationship between data governance policies and regulatory compliance requirements such as GDPR or CCPA.

---

## Overview

Data quality ensures the accuracy, completeness, and consistency of data, while data governance is a framework for systematically managing data assets. Both are essential for building trustworthy data pipelines.

---

## 1. Data Quality Dimensions

### 1.1 Quality Dimension Definitions

```
┌────────────────────────────────────────────────────────────────┐
│                   6 Dimensions of Data Quality                  │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   1. Accuracy                                                  │
│      - Does the data correctly reflect the actual value?       │
│      - Example: Is the customer email in a valid format?       │
│                                                                │
│   2. Completeness                                              │
│      - Is all necessary data present?                          │
│      - Example: Are there no NULLs in required fields?         │
│                                                                │
│   3. Consistency                                               │
│      - Is the data consistent across systems?                  │
│      - Example: Do order counts match between order table      │
│        and aggregate table?                                    │
│                                                                │
│   4. Timeliness                                                │
│      - Is data provided within an appropriate timeframe?       │
│      - Example: Is the real-time dashboard updated within      │
│        5 minutes?                                              │
│                                                                │
│   5. Uniqueness                                                │
│      - Is there no duplicate data?                             │
│      - Example: Is the same order not recorded multiple times? │
│                                                                │
│   6. Validity                                                  │
│      - Does the data comply with defined rules?                │
│      - Example: Is the date in the correct format?             │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 1.2 Quality Metrics Example

```python
from dataclasses import dataclass
from typing import Optional
import pandas as pd

@dataclass
class DataQualityMetrics:
    """Data quality metrics"""
    table_name: str
    row_count: int
    null_count: dict[str, int]
    duplicate_count: int
    freshness_hours: float
    schema_valid: bool

def calculate_quality_metrics(df: pd.DataFrame, table_name: str) -> DataQualityMetrics:
    """Calculate quality metrics"""

    # Completeness: count NULLs per column rather than just overall,
    # because a single critical column (e.g., primary key) being 50% null
    # is far worse than many optional columns having a few nulls each
    null_count = {col: df[col].isna().sum() for col in df.columns}

    # Uniqueness: whole-row deduplication catches exact duplicate ingestion,
    # which commonly occurs in at-least-once delivery systems (Kafka, S3 events)
    duplicate_count = df.duplicated().sum()

    return DataQualityMetrics(
        table_name=table_name,
        row_count=len(df),
        null_count=null_count,
        duplicate_count=duplicate_count,
        freshness_hours=0,  # Requires separate calculation
        schema_valid=True    # Requires separate validation
    )


def quality_score(metrics: DataQualityMetrics) -> float:
    """Calculate 0-100 quality score"""
    scores = []

    # Completeness score (NULL ratio)
    # Guard against division-by-zero when the table is empty or has no columns
    total_cells = metrics.row_count * len(metrics.null_count)
    total_nulls = sum(metrics.null_count.values())
    completeness = (1 - total_nulls / total_cells) * 100 if total_cells > 0 else 100
    scores.append(completeness)

    # Uniqueness score (duplicate ratio)
    # Simple average of two dimensions here; in production, weight these
    # based on business impact (e.g., uniqueness failures in financial data
    # may be catastrophic, while completeness gaps in logs are tolerable)
    uniqueness = (1 - metrics.duplicate_count / metrics.row_count) * 100 if metrics.row_count > 0 else 100
    scores.append(uniqueness)

    return sum(scores) / len(scores)
```

---

## 2. Great Expectations

### 2.1 Installation and Initialization

```bash
# Installation
pip install great_expectations

# Project initialization
great_expectations init
```

### 2.2 Basic Usage

```python
import great_expectations as gx
import pandas as pd

# GX uses a "Context" as the central entry point that manages all
# configuration, data sources, and expectation suites — similar to
# how dbt uses profiles.yml as a single configuration hub.
context = gx.get_context()

# Add data source — Pandas datasource for local/development validation.
# In production, swap for a SQLAlchemy or Spark datasource to validate
# data in-place without loading it into memory.
datasource = context.sources.add_pandas("my_datasource")

# Define data asset
data_asset = datasource.add_dataframe_asset(name="orders")

# Load DataFrame
df = pd.read_csv("orders.csv")

# Batch Request — decouples "which data" from "which expectations",
# enabling the same expectation suite to validate different batches
# (e.g., yesterday's data, today's data, a backfill batch)
batch_request = data_asset.build_batch_request(dataframe=df)

# Create Expectation Suite
suite = context.add_expectation_suite("orders_suite")

# Create Validator — the bridge between data (batch) and rules (suite).
# The validator materializes expectations into concrete SQL/Pandas checks.
validator = context.get_validator(
    batch_request=batch_request,
    expectation_suite_name="orders_suite"
)
```

### 2.3 Define Expectations

```python
# Each expectation maps to one of the six quality dimensions.
# Structuring expectations by dimension makes it clear which
# aspect of quality fails, enabling targeted remediation.

# Completeness: primary key must never be null — a null order_id
# would make downstream joins and aggregations unreliable
validator.expect_column_values_to_not_be_null("order_id")

# Uniqueness: duplicate order_ids inflate revenue/count metrics.
# If this fails, investigate at-least-once delivery from the source.
validator.expect_column_values_to_be_unique("order_id")

# Validity: bounds check catches data corruption (negative amounts)
# and outliers (>1M) that may indicate unit-of-measure errors (cents vs dollars)
validator.expect_column_values_to_be_between(
    "amount",
    min_value=0,
    max_value=1000000
)

# Validity: enum check detects unexpected status values that may indicate
# schema drift in the source system or unhandled business state transitions
validator.expect_column_values_to_be_in_set(
    "status",
    ["pending", "completed", "cancelled", "refunded"]
)

# Accuracy: regex validates email format — catches data entry errors
# and garbage values that would break email-based customer matching
validator.expect_column_values_to_match_regex(
    "email",
    r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
)

# Completeness: volume check detects upstream extraction failures.
# A row count drop below 1000 likely means the source query failed
# or returned partial results.
validator.expect_table_row_count_to_be_between(
    min_value=1000,
    max_value=1000000
)

# Consistency: schema drift detection — catches when upstream systems
# add, remove, or rename columns without coordinating with the data team
validator.expect_table_columns_to_match_set(
    ["order_id", "customer_id", "amount", "status", "order_date"]
)

# Validity: date format check prevents parsing failures downstream.
# Mixed formats (MM/DD vs YYYY-MM-DD) are a common silent data corruption.
validator.expect_column_values_to_match_strftime_format(
    "order_date",
    "%Y-%m-%d"
)

# Consistency: referential integrity between tables — orphaned customer_ids
# indicate sync lag between the orders and customers extraction jobs
validator.expect_column_values_to_be_in_set(
    "customer_id",
    customer_ids_list  # List of IDs from customer table
)

# discard_failed_expectations=False preserves all expectations including
# failed ones, which is important for iterative suite development
validator.save_expectation_suite(discard_failed_expectations=False)
```

### 2.4 Run Validation

```python
# Create and run Checkpoint
checkpoint = context.add_or_update_checkpoint(
    name="orders_checkpoint",
    validations=[
        {
            "batch_request": batch_request,
            "expectation_suite_name": "orders_suite"
        }
    ]
)

# Run validation
result = checkpoint.run()

# Check results
print(f"Success: {result.success}")
print(f"Statistics: {result.statistics}")

# Check failed Expectations
for validation_result in result.list_validation_results():
    for exp_result in validation_result.results:
        if not exp_result.success:
            print(f"Failed: {exp_result.expectation_config.expectation_type}")
            print(f"  Column: {exp_result.expectation_config.kwargs.get('column')}")
            print(f"  Result: {exp_result.result}")
```

### 2.5 Generate Data Docs

```python
# Build and open Data Docs
context.build_data_docs()
context.open_data_docs()
```

---

## 3. Airflow Integration

### 3.1 Great Expectations Operator

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import great_expectations as gx

def validate_data(**kwargs):
    """Great Expectations validation Task"""
    context = gx.get_context()

    # Checkpoints bundle a batch request + expectation suite + actions (e.g., store
    # results, send Slack alerts). Running via checkpoint ensures consistent
    # validation behavior across manual runs and scheduled executions.
    result = context.run_checkpoint(
        checkpoint_name="orders_checkpoint"
    )

    # Raising an exception on failure causes Airflow to mark this task as FAILED,
    # which blocks downstream tasks and triggers email/Slack alerts configured
    # in default_args — acting as a quality gate in the pipeline.
    if not result.success:
        raise ValueError("Data quality check failed!")

    return result.statistics


with DAG(
    'data_quality_dag',
    start_date=datetime(2024, 1, 1),
    schedule_interval='@daily',
) as dag:

    validate = PythonOperator(
        task_id='validate_orders',
        python_callable=validate_data,
    )
```

### 3.2 Custom Quality Checks

```python
from airflow.operators.python import PythonOperator, BranchPythonOperator

def check_row_count(**kwargs):
    """Row count validation"""
    import pandas as pd

    df = pd.read_parquet(f"/data/{kwargs['ds']}/orders.parquet")
    row_count = len(df)

    # Push to XCom so downstream tasks (like the branching decision) can
    # access the count without re-reading the file — avoids duplicate I/O
    kwargs['ti'].xcom_push(key='row_count', value=row_count)

    # Threshold of 1000 is a sanity floor: real daily order volume is ~10K+.
    # A count below 1000 usually indicates a partial extraction failure
    # rather than genuinely low business activity.
    if row_count < 1000:
        raise ValueError(f"Row count too low: {row_count}")

    return row_count


def check_freshness(**kwargs):
    """Data freshness validation"""
    from datetime import datetime, timedelta

    # Using file mtime as a proxy for data freshness — simple but effective.
    # In production, prefer checking MAX(event_timestamp) inside the data
    # for more accurate freshness measurement.
    import os
    file_path = f"/data/{kwargs['ds']}/orders.parquet"
    mtime = datetime.fromtimestamp(os.path.getmtime(file_path))

    age_hours = (datetime.now() - mtime).total_seconds() / 3600

    # 24-hour threshold aligns with the daily pipeline SLA.
    # If data is older than 24h, the pipeline likely didn't run or
    # the upstream source is stale.
    if age_hours > 24:
        raise ValueError(f"Data too old: {age_hours:.1f} hours")

    return age_hours


def decide_next_step(**kwargs):
    """Branching based on quality results"""
    ti = kwargs['ti']
    row_count = ti.xcom_pull(task_ids='check_row_count', key='row_count')

    # Adaptive processing: large batches use Spark for distributed compute,
    # while small batches use Pandas for lower overhead — optimizes cost
    # and avoids Spark startup latency on small datasets
    if row_count > 10000:
        return 'process_large_batch'
    else:
        return 'process_small_batch'


with DAG('quality_checks_dag', ...) as dag:

    check_rows = PythonOperator(
        task_id='check_row_count',
        python_callable=check_row_count,
    )

    check_fresh = PythonOperator(
        task_id='check_freshness',
        python_callable=check_freshness,
    )

    branch = BranchPythonOperator(
        task_id='decide_processing',
        python_callable=decide_next_step,
    )

    # Both checks run in parallel (fan-in) before branching — if either
    # fails, the branch task never executes, preventing bad data from
    # flowing downstream
    [check_rows, check_fresh] >> branch
```

---

## 4. Data Catalog

### 4.1 Catalog Concept

```
┌────────────────────────────────────────────────────────────────┐
│                    Data Catalog                                 │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   Metadata Management System:                                  │
│                                                                │
│   ┌────────────────────────────────────────────────────────┐  │
│   │  Technical Metadata                                     │  │
│   │  - Schema, data types, partitions                       │  │
│   │  - Location, format, size                               │  │
│   │  - Creation date, modification date                     │  │
│   └────────────────────────────────────────────────────────┘  │
│                                                                │
│   ┌────────────────────────────────────────────────────────┐  │
│   │  Business Metadata                                      │  │
│   │  - Description, definition, terminology                 │  │
│   │  - Owner, administrator                                 │  │
│   │  - Tags, classification                                 │  │
│   └────────────────────────────────────────────────────────┘  │
│                                                                │
│   ┌────────────────────────────────────────────────────────┐  │
│   │  Operational Metadata                                   │  │
│   │  - Usage frequency, query patterns                      │  │
│   │  - Quality score, issues                                │  │
│   │  - Access permissions                                   │  │
│   └────────────────────────────────────────────────────────┘  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 4.2 Catalog Tools

| Tool | Type | Features |
|------|------|------|
| **DataHub** | Open Source | Developed by LinkedIn, general purpose |
| **Apache Atlas** | Open Source | Hadoop ecosystem |
| **Amundsen** | Open Source | Developed by Lyft, search-focused |
| **OpenMetadata** | Open Source | All-in-one platform |
| **Atlan** | Commercial | Collaboration-focused |
| **Alation** | Commercial | Enterprise |

### 4.3 DataHub Example

```python
# DataHub metadata collection example
# Programmatic metadata emission enables automated catalog updates as part of
# CI/CD pipelines — no manual data entry needed when schemas or ownership change.
from datahub.emitter.mce_builder import make_dataset_urn
from datahub.emitter.rest_emitter import DatahubRestEmitter
from datahub.metadata.schema_classes import (
    DatasetPropertiesClass,
    SchemaMetadataClass,
    SchemaFieldClass,
    StringTypeClass,
    NumberTypeClass,
)

# REST emitter sends metadata to the DataHub GMS (Generalized Metadata Service).
# Alternatively, use KafkaEmitter for async, higher-throughput emission.
emitter = DatahubRestEmitter(gms_server="http://localhost:8080")

# URN (Uniform Resource Name) uniquely identifies each dataset across all platforms.
# The three-part structure (platform.name.env) prevents naming collisions when
# the same table name exists in dev and prod environments.
dataset_urn = make_dataset_urn(
    platform="postgres",
    name="analytics.public.fact_orders",
    env="PROD"
)

# Custom properties store governance metadata that DataHub's UI can filter/search on.
# Tagging PII status here enables automatic access control enforcement downstream.
properties = DatasetPropertiesClass(
    description="Orders fact table",
    customProperties={
        "owner": "data-team@company.com",
        "sla": "daily",
        "pii": "false"
    }
)

# Schema definition — registering field-level metadata enables impact analysis:
# when you rename "amount" to "order_amount", DataHub can identify all dashboards
# and downstream models that reference this field.
schema = SchemaMetadataClass(
    schemaName="fact_orders",
    platform=f"urn:li:dataPlatform:postgres",
    fields=[
        SchemaFieldClass(
            fieldPath="order_id",
            type=StringTypeClass(),
            description="Unique order ID"
        ),
        SchemaFieldClass(
            fieldPath="amount",
            type=NumberTypeClass(),
            description="Order amount"
        ),
    ]
)

# Emit metadata
emitter.emit_mce(properties)
emitter.emit_mce(schema)
```

---

## 5. Data Lineage

### 5.1 Lineage Concept

```
┌────────────────────────────────────────────────────────────────┐
│                     Data Lineage                                │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   Track data origin and transformation process:                │
│                                                                │
│   Raw Sources          Staging           Marts                 │
│   ┌──────────┐        ┌──────────┐      ┌──────────┐          │
│   │ orders   │───────→│stg_orders│─────→│fct_orders│          │
│   │ (raw)    │        │          │      │          │          │
│   └──────────┘        └──────────┘      └────┬─────┘          │
│                                               │                │
│   ┌──────────┐        ┌──────────┐           │                │
│   │customers │───────→│stg_customers│────────→│                │
│   │ (raw)    │        │          │           │                │
│   └──────────┘        └──────────┘           │                │
│                                               ↓                │
│                                         ┌──────────┐          │
│                                         │ dashboard│          │
│                                         │ (BI)     │          │
│                                         └──────────┘          │
│                                                                │
│   Uses:                                                        │
│   - Impact analysis: Identify affected targets when source     │
│     changes                                                    │
│   - Root cause analysis: Track source of data issues           │
│   - Compliance: Audit data flows                               │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 5.2 dbt Lineage

```bash
# Generate dbt lineage
dbt docs generate

# View lineage (docs server)
dbt docs serve
```

```yaml
# dbt model metadata
version: 2

models:
  - name: fct_orders
    description: "Orders fact table"
    meta:
      owner: "data-team"
      upstream:
        - stg_orders
        - stg_customers
      downstream:
        - sales_dashboard
        - ml_model_features
```

### 5.3 OpenLineage

```python
# Lineage tracking using OpenLineage
# OpenLineage is an open standard (not tied to any single vendor) for lineage events.
# This enables heterogeneous pipelines (Spark + Airflow + dbt) to emit lineage to
# a single backend (e.g., Marquez, DataHub) without vendor lock-in.
from openlineage.client import OpenLineageClient
from openlineage.client.run import Run, Job, RunEvent, RunState
from openlineage.client.facet import (
    SqlJobFacet,
    SchemaDatasetFacet,
    SchemaField,
)
from datetime import datetime
import uuid

client = OpenLineageClient(url="http://localhost:5000")

# Namespace + name uniquely identifies a job across the organization.
# Use a consistent naming convention (e.g., team.pipeline_name) to enable
# cross-team lineage discovery.
job = Job(
    namespace="my_pipeline",
    name="transform_orders"
)

# UUID per run ensures each execution is independently trackable,
# enabling debugging of specific pipeline failures via their run_id.
run_id = str(uuid.uuid4())
run = Run(runId=run_id)

# Input datasets — declaring inputs with schema facets enables automatic
# impact analysis: when raw.orders changes, the lineage graph shows which
# downstream jobs and datasets are affected.
input_datasets = [
    {
        "namespace": "postgres",
        "name": "raw.orders",
        "facets": {
            "schema": SchemaDatasetFacet(
                fields=[
                    SchemaField(name="order_id", type="string"),
                    SchemaField(name="amount", type="decimal"),
                ]
            )
        }
    }
]

# Output datasets
output_datasets = [
    {
        "namespace": "postgres",
        "name": "analytics.fct_orders",
    }
]

# START event marks the beginning of a run — lineage consumers can track
# in-progress jobs and detect stuck/hanging pipelines
client.emit(
    RunEvent(
        eventType=RunState.START,
        eventTime=datetime.now().isoformat(),
        run=run,
        job=job,
        inputs=input_datasets,
        outputs=output_datasets,
    )
)

# ... actual transformation work ...

# COMPLETE event closes the run. Emitting both START and COMPLETE enables
# duration tracking. If a COMPLETE is never received, monitoring can flag
# the run as potentially failed.
client.emit(
    RunEvent(
        eventType=RunState.COMPLETE,
        eventTime=datetime.now().isoformat(),
        run=run,
        job=job,
    )
)
```

---

## 6. Governance Framework

### 6.1 Data Governance Components

```
┌────────────────────────────────────────────────────────────────┐
│                 Data Governance Framework                       │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   1. Organization                                              │
│      - Designate data stewards                                 │
│      - Define roles and responsibilities                       │
│      - Governance committee                                    │
│                                                                │
│   2. Policies                                                  │
│      - Data classification policy                              │
│      - Access control policy                                   │
│      - Retention/deletion policy                               │
│      - Quality standards                                       │
│                                                                │
│   3. Processes                                                 │
│      - Data request/approval process                           │
│      - Issue management process                                │
│      - Change management process                               │
│                                                                │
│   4. Technology                                                │
│      - Data catalog                                            │
│      - Quality monitoring                                      │
│      - Access control systems                                  │
│      - Audit logs                                              │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 6.2 Data Classification

```python
from enum import Enum

class DataClassification(Enum):
    """Data sensitivity classification
    Four levels from least to most sensitive. This hierarchy directly maps
    to access control policies: e.g., RESTRICTED requires encryption at rest
    and column-level masking for non-privileged users.
    """
    PUBLIC = "public"           # Publicly available
    INTERNAL = "internal"       # Internal use
    CONFIDENTIAL = "confidential"  # Confidential
    RESTRICTED = "restricted"   # Restricted (PII, financial)

class DataClassifier:
    """Automatic data classification"""

    # Regex patterns for common PII types. These are intentionally broad
    # to minimize false negatives — it's better to over-classify (restrict
    # non-PII) than to under-classify (expose actual PII). False positives
    # can be whitelisted during manual review.
    PII_PATTERNS = {
        'email': r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+',
        'phone': r'\d{3}-\d{3,4}-\d{4}',
        'ssn': r'\d{3}-\d{2}-\d{4}',
        'credit_card': r'\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}',
    }

    PII_COLUMN_NAMES = [
        'email', 'phone', 'ssn', 'social_security',
        'credit_card', 'password', 'address'
    ]

    @classmethod
    def classify_column(cls, column_name: str, sample_values: list) -> DataClassification:
        """Classify column"""
        column_lower = column_name.lower()

        # Name-based check runs first because it's O(1) and catches the
        # most common cases without reading any data
        if any(pii in column_lower for pii in cls.PII_COLUMN_NAMES):
            return DataClassification.RESTRICTED

        # Value-based check as fallback: catches PII in generically-named columns
        # (e.g., 'field_1' that actually contains SSNs).
        # Sampling 100 values balances accuracy with performance — scanning
        # every row would be prohibitively slow on large tables.
        import re
        for value in sample_values[:100]:
            if value is None:
                continue
            for pii_type, pattern in cls.PII_PATTERNS.items():
                if re.match(pattern, str(value)):
                    return DataClassification.RESTRICTED

        # Default to INTERNAL (not PUBLIC) as a safe default: data should be
        # explicitly promoted to PUBLIC after human review
        return DataClassification.INTERNAL
```

---

## Practice Problems

### Problem 1: Great Expectations
Write an Expectation Suite for order data (NULL check, unique, value range, referential integrity).

### Problem 2: Quality Dashboard
Design a pipeline that calculates and visualizes daily data quality scores.

### Problem 3: Lineage Tracking
Design a system that automatically tracks lineage in an ETL pipeline.

---

## Summary

| Concept | Description |
|------|------|
| **Data Quality** | Ensuring accuracy, completeness, consistency, timeliness |
| **Great Expectations** | Python-based data quality framework |
| **Data Catalog** | Metadata management system |
| **Data Lineage** | Tracking data origin and transformations |
| **Data Governance** | Systematic management of data assets |

---

## References

- [Great Expectations Documentation](https://docs.greatexpectations.io/)
- [DataHub Documentation](https://datahubproject.io/docs/)
- [OpenLineage](https://openlineage.io/)
- [DMBOK (Data Management Body of Knowledge)](https://www.dama.org/cpages/body-of-knowledge)
