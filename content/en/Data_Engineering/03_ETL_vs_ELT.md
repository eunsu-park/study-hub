# ETL vs ELT

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the ETL (Extract, Transform, Load) and ELT (Extract, Load, Transform) patterns and describe their fundamental differences
2. Implement an ETL pipeline in Python using pandas and SQLAlchemy
3. Implement an ELT pipeline leveraging SQL-based transformations within a data warehouse
4. Compare ETL and ELT across dimensions such as performance, scalability, cost, and use cases
5. Identify when to use ETL versus ELT based on data volume, transformation complexity, and infrastructure constraints
6. Analyze real-world scenarios and choose the appropriate pattern for a given data engineering problem

---

## Overview

ETL (Extract, Transform, Load) and ELT (Extract, Load, Transform) are two major patterns in data pipelines. Traditional ETL transforms then loads data, while modern ELT loads then transforms data.

---

## 1. ETL (Extract, Transform, Load)

### 1.1 ETL Process

```
┌─────────────────────────────────────────────────────────────┐
│                      ETL Process                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────┐    ┌──────────────┐    ┌──────────┐         │
│   │ Sources  │ → │ ETL Server   │ → │ Target   │          │
│   │          │    │              │    │ (DW)     │          │
│   │ - DB     │    │ 1. Extract   │    │          │          │
│   │ - Files  │    │ 2. Transform │    │ Clean    │          │
│   │ - APIs   │    │ 3. Load      │    │ Data     │          │
│   └──────────┘    └──────────────┘    └──────────┘         │
│                                                             │
│   Transformation performed on intermediate server           │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 ETL Example Code

```python
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime

class ETLPipeline:
    """Traditional ETL pipeline"""

    def __init__(self, source_conn: str, target_conn: str):
        # Separate engines for source vs target: prevents accidental writes
        # to the source DB and makes connection pooling independent.
        self.source_engine = create_engine(source_conn)
        self.target_engine = create_engine(target_conn)

    def extract(self, query: str) -> pd.DataFrame:
        """
        Extract: Extract data from source
        """
        print(f"[Extract] Starting at {datetime.now()}")
        df = pd.read_sql(query, self.source_engine)
        print(f"[Extract] Extracted {len(df)} rows")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform: Clean and transform data
        - This step is performed on ETL server (resource consuming)
        """
        print(f"[Transform] Starting at {datetime.now()}")

        # 1. Drop rows missing required fields (customer_id, amount) but
        # fill optional fields (email) — losing a sale record is worse than
        # having a placeholder email.
        df = df.dropna(subset=['customer_id', 'amount'])
        df['email'] = df['email'].fillna('unknown@example.com')

        # 2. Explicit type casting catches source-side schema drift early:
        # if order_date arrives as an unexpected format, this fails loudly
        # rather than silently loading corrupt data.
        df['order_date'] = pd.to_datetime(df['order_date'])
        df['amount'] = df['amount'].astype(float)

        # 3. Generate derived columns
        df['year'] = df['order_date'].dt.year
        df['month'] = df['order_date'].dt.month
        df['day_of_week'] = df['order_date'].dt.dayofweek

        # 4. Apply business logic — segmenting here (on the ETL server)
        # ensures all consumers see the same segment definitions, unlike
        # ELT where each analyst might apply thresholds differently.
        df['customer_segment'] = df['total_purchases'].apply(
            lambda x: 'Gold' if x > 10000 else ('Silver' if x > 5000 else 'Bronze')
        )

        # 5. Fail-fast assertion: cheaper to abort the pipeline than to
        # load negative amounts that corrupt downstream revenue reports.
        assert df['amount'].min() >= 0, "Negative amounts found"

        print(f"[Transform] Transformed {len(df)} rows")
        return df

    def load(self, df: pd.DataFrame, table_name: str):
        """
        Load: Load into target data warehouse
        """
        print(f"[Load] Starting at {datetime.now()}")

        # Full refresh (if_exists='replace'): simple and idempotent, but
        # not suitable for large tables — use incremental upserts for those.
        # chunksize=10000 avoids building a single enormous INSERT statement
        # that could exceed DB memory limits.
        df.to_sql(
            table_name,
            self.target_engine,
            if_exists='replace',
            index=False,
            chunksize=10000
        )

        print(f"[Load] Loaded {len(df)} rows to {table_name}")

    def run(self, source_query: str, target_table: str):
        """Execute ETL pipeline"""
        start_time = datetime.now()
        print(f"ETL Pipeline started at {start_time}")

        # Execute in E-T-L order
        raw_data = self.extract(source_query)
        transformed_data = self.transform(raw_data)
        self.load(transformed_data, target_table)

        end_time = datetime.now()
        print(f"ETL Pipeline completed in {(end_time - start_time).seconds} seconds")


# Usage example
if __name__ == "__main__":
    pipeline = ETLPipeline(
        source_conn="postgresql://user:pass@source-db:5432/sales",
        target_conn="postgresql://user:pass@warehouse:5432/analytics"
    )

    pipeline.run(
        source_query="""
            SELECT
                o.order_id,
                o.customer_id,
                c.email,
                o.order_date,
                o.amount,
                c.total_purchases
            FROM orders o
            JOIN customers c ON o.customer_id = c.customer_id
            WHERE o.order_date >= CURRENT_DATE - INTERVAL '1 day'
        """,
        target_table="fact_daily_orders"
    )
```

### 1.3 ETL Tools

| Tool | Type | Features |
|------|------|------|
| **Informatica** | Commercial | Enterprise-grade, GUI-based |
| **Talend** | Open Source/Commercial | Java-based, various connectors |
| **SSIS** | Commercial (MS) | SQL Server integration |
| **Pentaho** | Open Source | Lightweight, user-friendly |
| **Apache NiFi** | Open Source | Data flow, real-time |

---

## 2. ELT (Extract, Load, Transform)

### 2.1 ELT Process

```
┌─────────────────────────────────────────────────────────────┐
│                      ELT Process                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────┐    ┌──────────────┐    ┌──────────────┐     │
│   │ Sources  │ → │ Load Raw     │ → │ Transform    │      │
│   │          │    │ (Data Lake)  │    │ (in DW)      │      │
│   │ - DB     │    │              │    │              │      │
│   │ - Files  │    │ Raw Zone     │    │ SQL/Spark    │      │
│   │ - APIs   │    │ (as-is)      │    │ (DW resource)│      │
│   └──────────┘    └──────────────┘    └──────────────┘     │
│                                                             │
│   Transformation performed in target system (DW/Lake)       │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 ELT Example Code

```python
import pandas as pd
from datetime import datetime

class ELTPipeline:
    """Modern ELT pipeline"""

    def __init__(self, source_conn: str, warehouse_conn: str):
        self.source_conn = source_conn
        self.warehouse_conn = warehouse_conn

    def extract_and_load(self, source_query: str, raw_table: str):
        """
        Extract & Load: Load raw data as-is
        - Quickly load raw data without transformation
        """
        print(f"[Extract & Load] Starting at {datetime.now()}")

        # Extract data from source
        df = pd.read_sql(source_query, self.source_conn)

        # Load as-is to raw table (no transformation)
        df.to_sql(
            raw_table,
            self.warehouse_conn,
            if_exists='replace',
            index=False
        )

        print(f"[Extract & Load] Loaded {len(df)} rows to {raw_table}")

    def transform_in_warehouse(self, transform_sql: str):
        """
        Transform: Transform with SQL in warehouse
        - Utilize DW computing power
        - SQL-based transformation (using dbt, etc.)
        """
        print(f"[Transform] Starting at {datetime.now()}")

        # Execute SQL in warehouse
        with self.warehouse_conn.connect() as conn:
            conn.execute(transform_sql)

        print(f"[Transform] Transformation completed")


# dbt model example (SQL-based transformation)
# Notice how this SQL applies the *same* logic as the Python ETL transform()
# above — the key difference is WHERE it runs: inside the warehouse engine,
# leveraging its MPP (massively parallel processing) compute.
DBT_MODEL_EXAMPLE = """
-- models/staging/stg_orders.sql
-- ELT transformation using dbt

WITH source AS (
    -- {{ source() }} macro provides lineage tracking and freshness checks
    -- so dbt can alert you if the raw table hasn't been updated.
    SELECT * FROM {{ source('raw', 'orders_raw') }}
),

cleaned AS (
    SELECT
        order_id,
        customer_id,
        COALESCE(email, 'unknown@example.com') AS email,
        CAST(order_date AS DATE) AS order_date,
        CAST(amount AS DECIMAL(10, 2)) AS amount,
        total_purchases,
        -- Derived columns computed at query time — no ETL server needed;
        -- the warehouse handles this in parallel across its nodes.
        EXTRACT(YEAR FROM order_date) AS order_year,
        EXTRACT(MONTH FROM order_date) AS order_month,
        EXTRACT(DOW FROM order_date) AS day_of_week,
        -- Business logic
        CASE
            WHEN total_purchases > 10000 THEN 'Gold'
            WHEN total_purchases > 5000 THEN 'Silver'
            ELSE 'Bronze'
        END AS customer_segment,
        -- loaded_at tracks *when* each row was processed, enabling
        -- incremental models to pick up only newly arrived data.
        CURRENT_TIMESTAMP AS loaded_at
    FROM source
    -- Quality filter pushed into SQL: rejects bad rows before they
    -- enter the staging layer, equivalent to the ETL assertion approach.
    WHERE customer_id IS NOT NULL
      AND amount IS NOT NULL
      AND amount >= 0
)

SELECT * FROM cleaned
"""


# Actual ELT pipeline (Snowflake/BigQuery style)
# Three-layer architecture (raw → staging → mart) follows the medallion
# pattern: each layer increases data quality while preserving the raw copy
# for debugging and reprocessing.
class ModernELTWithSQL:
    """SQL-based modern ELT"""

    def __init__(self, warehouse):
        self.warehouse = warehouse

    def extract_load(self, source: str, target_raw: str):
        """Source → Raw layer"""
        # COPY INTO is a warehouse-native bulk load command — orders of
        # magnitude faster than row-by-row INSERT because it reads
        # Parquet files directly from cloud storage in parallel.
        copy_sql = f"""
        COPY INTO {target_raw}
        FROM @{source}
        FILE_FORMAT = (TYPE = 'PARQUET')
        """
        self.warehouse.execute(copy_sql)

    def transform_staging(self):
        """Raw → Staging layer"""
        # PARSE_JSON + explicit casts: raw layer stores semi-structured data
        # as-is; the staging layer enforces a schema, catching type mismatches
        # early while still keeping the raw copy untouched for reprocessing.
        staging_sql = """
        CREATE OR REPLACE TABLE staging.orders AS
        SELECT
            order_id,
            customer_id,
            PARSE_JSON(raw_data):email::STRING AS email,
            TO_DATE(raw_data:order_date) AS order_date,
            raw_data:amount::NUMBER(10,2) AS amount
        FROM raw.orders_raw
        """
        self.warehouse.execute(staging_sql)

    def transform_mart(self):
        """Staging → Mart layer"""
        # Mart joins staging data with dimension tables and adds analytics
        # columns. Window functions like cumulative_amount run efficiently
        # on the warehouse's distributed compute — no external processing needed.
        mart_sql = """
        CREATE OR REPLACE TABLE mart.fact_orders AS
        SELECT
            o.order_id,
            d.date_sk,
            c.customer_sk,
            o.amount,
            -- Running total per customer: enables lifetime-value queries
            -- without rescanning the entire fact table.
            SUM(o.amount) OVER (
                PARTITION BY o.customer_id
                ORDER BY o.order_date
            ) AS cumulative_amount
        FROM staging.orders o
        JOIN dim_date d ON o.order_date = d.full_date
        JOIN dim_customer c ON o.customer_id = c.customer_id
        """
        self.warehouse.execute(mart_sql)
```

### 2.3 ELT Tools

| Tool | Type | Features |
|------|------|------|
| **dbt** | Open Source | SQL-based transformation, testing, documentation |
| **Fivetran** | Commercial | Automatic schema management, 150+ connectors |
| **Airbyte** | Open Source | Custom connectors, EL specialized |
| **Stitch** | Commercial | Easy setup, SaaS friendly |
| **AWS Glue** | Cloud | Serverless, Spark-based |

---

## 3. ETL vs ELT Comparison

### 3.1 Detailed Comparison

| Characteristic | ETL | ELT |
|------|-----|-----|
| **Transformation Location** | Intermediate server | Target system (DW/Lake) |
| **Data Movement** | Transformed data only | Entire raw data |
| **Schema** | Must be predefined | Flexible (Schema-on-Read) |
| **Processing Speed** | Slow (intermediate processing) | Fast (parallel processing) |
| **Cost** | Separate infrastructure needed | Uses DW resources |
| **Flexibility** | Low | High (raw data preserved) |
| **Complex Transforms** | Suitable | Limited |
| **Real-time Processing** | Difficult | Relatively easy |

### 3.2 Selection Criteria

```python
def choose_etl_or_elt(requirements: dict) -> str:
    """ETL/ELT selection guide"""

    # ETL factors: scenarios where transformation MUST happen outside
    # the warehouse (privacy, non-SQL logic, or legacy format parsing).
    etl_factors = [
        requirements.get('data_privacy', False),      # PII must be masked before loading
        requirements.get('complex_transforms', False), # Logic too complex for SQL (ML, NLP)
        requirements.get('legacy_systems', False),    # Fixed-width/mainframe formats
        requirements.get('small_data', False),        # DW overhead not justified
    ]

    # ELT factors: scenarios where the warehouse engine should do the work
    # because it can parallelise and the raw data should be preserved.
    elt_factors = [
        requirements.get('big_data', False),          # Warehouse MPP handles scale
        requirements.get('cloud_dw', False),          # Pay-per-query compute
        requirements.get('data_lake', False),         # Schema-on-read flexibility
        requirements.get('flexible_schema', False),   # Sources change often
        requirements.get('raw_data_access', False),   # Analysts need raw data
        requirements.get('sql_transforms', False),    # Transformations expressible in SQL
    ]

    # Simple scoring — in practice, weight factors differently (e.g., data_privacy
    # might be a hard constraint that overrides the score entirely).
    etl_score = sum(etl_factors)
    elt_score = sum(elt_factors)

    if etl_score > elt_score:
        return "ETL recommended"
    elif elt_score > etl_score:
        return "ELT recommended"
    else:
        return "Consider hybrid"


# Usage example
project_requirements = {
    'big_data': True,
    'cloud_dw': True,  # Snowflake, BigQuery
    'sql_transforms': True,
    'raw_data_access': True
}

recommendation = choose_etl_or_elt(project_requirements)
print(recommendation)  # "ELT recommended"
```

---

## 4. Hybrid Approach

### 4.1 ETLT Pattern

```
┌─────────────────────────────────────────────────────────────┐
│                    ETLT (Hybrid) Pattern                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Sources → [E] → [T] → [L] → [T] → Mart                   │
│                    ↑          ↑                             │
│                    │          │                             │
│            Light Transform   Heavy Transform                │
│            (masking, validation)  (aggregation, join)       │
│            ETL Server        Data Warehouse                 │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Hybrid Implementation Example

```python
class HybridPipeline:
    """ETL + ELT hybrid pipeline"""

    def __init__(self, source, staging_area, warehouse):
        self.source = source
        self.staging = staging_area
        self.warehouse = warehouse

    def extract_with_light_transform(self):
        """
        E + Light T: Perform light transformation during extraction
        - PII masking (privacy protection)
        - Basic data type conversion
        - Essential field validation
        """
        # PII masking MUST happen before data leaves the source system:
        # once raw PII lands in the warehouse, it's much harder to control
        # access and comply with GDPR/CCPA data-minimization requirements.
        query = """
        SELECT
            order_id,
            -- Hash at source so the warehouse never sees raw emails;
            -- MD5 is sufficient for pseudonymization (not security).
            MD5(customer_email) AS customer_email_hash,
            SUBSTRING(phone, 1, 3) || '****' || SUBSTRING(phone, -4) AS phone_masked,
            -- Basic type casting here catches malformed data early,
            -- before it consumes warehouse storage and compute.
            CAST(order_date AS DATE) AS order_date,
            CAST(amount AS DECIMAL(10, 2)) AS amount
        FROM orders
        WHERE amount IS NOT NULL
        """
        return self.source.execute(query)

    def load_to_staging(self, data):
        """L: Load to staging area"""
        self.staging.load(data, 'orders_staging')

    def transform_in_warehouse(self):
        """
        Heavy T: Complex transformation in warehouse
        - Joins, aggregations, and window functions are pushed to the DW
          because it can parallelise these operations across many nodes —
          doing this on the ETL server would be a single-machine bottleneck.
        """
        heavy_transform_sql = """
        CREATE TABLE mart.order_analysis AS
        SELECT
            o.order_date,
            c.customer_segment,
            p.product_category,
            COUNT(*) AS order_count,
            SUM(o.amount) AS total_amount,
            AVG(o.amount) AS avg_order_value,
            -- 7-day rolling sum by segment: ROWS BETWEEN 6 PRECEDING AND
            -- CURRENT ROW gives exactly 7 rows (today + 6 prior days).
            -- This runs efficiently on columnar engines with sort keys.
            SUM(o.amount) OVER (
                PARTITION BY c.customer_segment
                ORDER BY o.order_date
                ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
            ) AS rolling_7day_amount
        FROM orders_staging o
        JOIN dim_customer c ON o.customer_id = c.customer_id
        JOIN dim_product p ON o.product_id = p.product_id
        GROUP BY o.order_date, c.customer_segment, p.product_category
        """
        self.warehouse.execute(heavy_transform_sql)

    def run(self):
        """Execute hybrid pipeline"""
        # Phase 1: ETL — light transforms (PII masking) on the ETL server
        data = self.extract_with_light_transform()

        # Phase 2: Load masked data to staging area
        self.load_to_staging(data)

        # Phase 3: ELT — heavy transforms in the DW engine
        self.transform_in_warehouse()
```

---

## 5. Real-world Use Cases

### 5.1 ETL Use Cases

```python
# Case 1: Privacy handling (GDPR compliance)
# ETL is the right choice here because PII must be masked *before* leaving
# the source environment — an ELT approach would land raw PII in the warehouse,
# violating data-minimization principles even if access controls are in place.
class GDPRCompliantETL:
    """GDPR-compliant ETL - Mask PII before loading"""

    def transform(self, df):
        # Mask on the ETL server so the warehouse never stores raw PII.
        # Partial masking (keeping last 4 digits) balances privacy with
        # the ability to verify records during support escalations.
        df['email'] = df['email'].apply(self.mask_email)
        df['ssn'] = df['ssn'].apply(lambda x: 'XXX-XX-' + x[-4:])
        df['credit_card'] = df['credit_card'].apply(lambda x: '**** **** **** ' + x[-4:])

        # Consent filter: only transfer data for users who opted in.
        # Applying this filter *before* cross-border transfer ensures
        # compliance with GDPR Article 6 (lawful basis for processing).
        df = df[df['consent_given'] == True]

        return df

    def mask_email(self, email):
        if pd.isna(email):
            return None
        local, domain = email.split('@')
        return local[:2] + '***@' + domain


# Case 2: Legacy system integration
# ETL is necessary here because mainframe fixed-width formats can't be
# loaded directly into modern warehouses — they need structural parsing
# on an intermediate server first.
class LegacySystemETL:
    """Legacy mainframe data integration"""

    def transform(self, raw_data):
        # Fixed-width field offsets are defined by the mainframe COBOL copybook;
        # even a 1-byte shift corrupts every downstream field.
        records = []
        for line in raw_data.split('\n'):
            record = {
                'account_no': line[0:10].strip(),
                'account_type': line[10:12],
                # Mainframes store currency as integers (no decimals),
                # so divide by 100 to recover the true dollar amount.
                'balance': int(line[12:24]) / 100,
                'status': 'A' if line[24:25] == '1' else 'I',
                'date': self.parse_legacy_date(line[25:33])
            }
            records.append(record)
        return pd.DataFrame(records)

    def parse_legacy_date(self, date_str):
        # YYYYMMDD → YYYY-MM-DD
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
```

### 5.2 ELT Use Cases

```sql
-- Case 1: E-commerce analytics using dbt

-- models/staging/stg_orders.sql
WITH raw_orders AS (
    SELECT * FROM {{ source('raw', 'orders') }}
)
SELECT
    order_id,
    customer_id,
    order_date,
    status,
    total_amount
FROM raw_orders
WHERE order_date IS NOT NULL

-- models/marts/core/fct_orders.sql
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
    c.customer_segment,
    p.product_category,
    o.total_amount,
    -- Utilize window functions in DW
    ROW_NUMBER() OVER (PARTITION BY o.customer_id ORDER BY o.order_date) AS order_sequence,
    LAG(o.order_date) OVER (PARTITION BY o.customer_id ORDER BY o.order_date) AS prev_order_date
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON o.product_id = p.product_id


-- Case 2: BigQuery ELT
-- Large-scale log analysis (serverless processing)
CREATE OR REPLACE TABLE analytics.user_behavior AS
SELECT
    user_id,
    DATE(timestamp) AS event_date,
    event_type,
    COUNT(*) AS event_count,
    COUNTIF(event_type = 'purchase') AS purchase_count,
    SUM(CASE WHEN event_type = 'purchase' THEN revenue ELSE 0 END) AS total_revenue,
    -- Session analysis (complex window function)
    ARRAY_AGG(
        STRUCT(timestamp, event_type, page_url)
        ORDER BY timestamp
    ) AS event_sequence
FROM raw.events
WHERE DATE(timestamp) = CURRENT_DATE() - 1
GROUP BY user_id, DATE(timestamp), event_type;
```

---

## 6. Tool Selection Guide

### 6.1 Recommended Tools by Data Scale

| Data Scale | ETL Tools | ELT Tools |
|-------------|----------|----------|
| **Small** (< 1GB) | Python + Pandas | dbt + PostgreSQL |
| **Medium** (1GB-100GB) | Airflow + Python | dbt + Snowflake |
| **Large** (> 100GB) | Spark | dbt + BigQuery/Databricks |

### 6.2 Recommendations by Architecture

```python
# Match the approach to the target system's strengths:
# traditional DWs enforce schema on write (ETL fits naturally),
# cloud DWs offer elastic compute (ELT leverages that),
# data lakes accept any format (ELT with schema-on-read).
architecture_recommendations = {
    "traditional_dw": {
        "approach": "ETL",
        "tools": ["Informatica", "Talend", "SSIS"],
        "reason": "Strict schema, load after transformation"
    },
    "cloud_dw": {
        "approach": "ELT",
        "tools": ["dbt", "Fivetran + dbt", "Airbyte + dbt"],
        # Cloud DWs charge for compute-seconds — running transforms inside
        # the DW avoids paying for a separate ETL cluster.
        "reason": "Utilize DW computing power, preserve raw data"
    },
    "data_lake": {
        "approach": "ELT",
        "tools": ["Spark", "AWS Glue", "Databricks"],
        "reason": "Flexible schema, large-scale processing"
    },
    "hybrid": {
        "approach": "ETLT",
        "tools": ["Airflow + dbt", "Prefect + dbt"],
        # Orchestrator (Airflow/Prefect) handles light ETL transforms
        # (masking, validation), then dbt handles heavy SQL transforms in DW.
        "reason": "Sensitive data processing + DW transformation"
    }
}
```

---

## Practice Problems

### Problem 1: ETL vs ELT Selection
Choose between ETL and ELT for the following situations and explain your reasoning:
- Loading 100GB of daily log data to BigQuery
- Processing customer data containing personal information

### Problem 2: ELT SQL Writing
Write ELT SQL to create a daily sales aggregation table from raw table `raw_orders`.

---

## Summary

| Concept | Description |
|------|------|
| **ETL** | Transform then load, process on intermediate server |
| **ELT** | Load then transform, process in target system |
| **ETL Advantages** | Data quality assurance, sensitive data processing |
| **ELT Advantages** | Fast loading, flexible schema, raw data preservation |
| **Hybrid** | ETL + ELT combination, select based on situation |

---

## References

- [dbt Documentation](https://docs.getdbt.com/)
- [Modern Data Stack](https://www.moderndatastack.xyz/)
- [ETL vs ELT: The Difference](https://www.fivetran.com/blog/etl-vs-elt)
