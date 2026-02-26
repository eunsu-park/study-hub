# dbt Transformation Tool

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the role of dbt in an ELT pipeline and how it differs from traditional ETL transformation tools.
2. Define dbt models, materializations (view, table, incremental, ephemeral), and manage inter-model dependencies using the ref() function.
3. Implement dbt tests (schema tests and custom singular tests) to validate data quality and integrity.
4. Organize a dbt project using sources, staging models, intermediate models, and mart models following best practices.
5. Generate and publish automated documentation and data lineage graphs using dbt docs.
6. Configure incremental models and snapshots to efficiently handle large datasets and track slowly changing dimensions (SCD).

---

## Overview

dbt (data build tool) is a SQL-based data transformation tool. It handles the Transform step in the ELT pattern and applies software engineering best practices (version control, testing, documentation) to data transformations.

---

## 1. dbt Overview

### 1.1 What is dbt?

```
┌────────────────────────────────────────────────────────────────┐
│                        dbt Role                                 │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   Handles T(Transform) in ELT Pipeline                         │
│                                                                │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐               │
│   │ Extract  │ →  │   Load   │ →  │Transform │               │
│   │ (Fivetran│    │  (to DW) │    │  (dbt)   │               │
│   │  Airbyte)│    │          │    │          │               │
│   └──────────┘    └──────────┘    └──────────┘               │
│                                                                │
│   Core dbt Features:                                           │
│   - SQL-based model definition                                 │
│   - Automatic dependency management                            │
│   - Testing and documentation                                  │
│   - Jinja template support                                     │
│   - Version control (Git)                                      │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 1.2 dbt Core vs dbt Cloud

| Feature | dbt Core | dbt Cloud |
|------|----------|-----------|
| **Cost** | Free (Open Source) | Paid (SaaS) |
| **Execution** | CLI | Web UI + API |
| **Scheduling** | External tool required (Airflow) | Built-in scheduler |
| **IDE** | VS Code etc. | Built-in IDE |
| **Collaboration** | Git-based | Built-in collaboration |

### 1.3 Installation

```bash
# Install dbt Core
pip install dbt-core

# Install database-specific adapters
pip install dbt-postgres      # PostgreSQL
pip install dbt-snowflake     # Snowflake
pip install dbt-bigquery      # BigQuery
pip install dbt-redshift      # Redshift
pip install dbt-databricks    # Databricks

# Check version
dbt --version
```

---

## 2. Project Structure

### 2.1 Project Initialization

```bash
# Create new project
dbt init my_project
cd my_project

# Project structure
my_project/
├── dbt_project.yml          # Project configuration
├── profiles.yml             # Connection settings (~/.dbt/profiles.yml)
├── models/                  # SQL models
│   ├── staging/            # Staging models
│   ├── intermediate/       # Intermediate models
│   └── marts/              # Final models
├── tests/                   # Custom tests
├── macros/                  # Reusable macros
├── seeds/                   # Seed data (CSV)
├── snapshots/               # SCD snapshots
├── analyses/                # Analysis queries
└── target/                  # Compiled results
```

### 2.2 Configuration Files

```yaml
# dbt_project.yml
name: 'my_project'
version: '1.0.0'
config-version: 2

profile: 'my_project'

model-paths: ["models"]
analysis-paths: ["analyses"]
test-paths: ["tests"]
seed-paths: ["seeds"]
macro-paths: ["macros"]
snapshot-paths: ["snapshots"]

target-path: "target"
clean-targets:
  - "target"
  - "dbt_packages"

# Materialization strategy follows the medallion pattern:
# - staging as view: fast iteration during dev, always reflects source data,
#   and avoids duplicating storage for 1:1 source mirrors
# - intermediate as ephemeral: inlined as CTEs — no physical table created,
#   reducing warehouse clutter for intermediate logic that's never queried directly
# - marts as table: pre-computed for BI queries, trading storage for speed
models:
  my_project:
    staging:
      +materialized: view
      +schema: staging
    intermediate:
      +materialized: ephemeral
    marts:
      +materialized: table
      +schema: marts
```

```yaml
# profiles.yml (~/.dbt/profiles.yml)
# Stored outside the project repo (~/.dbt/) to prevent credentials from
# being committed to version control. env_var() reads secrets from the
# environment at runtime, supporting both local development and CI/CD.
my_project:
  target: dev
  outputs:
    dev:
      type: postgres
      host: localhost
      port: 5432
      user: postgres
      password: "{{ env_var('DB_PASSWORD') }}"
      dbname: analytics
      schema: dbt_dev
      # 4 threads for dev: balances speed with local resource limits.
      # Each thread runs one model concurrently.
      threads: 4

    prod:
      type: postgres
      host: prod-db.example.com
      port: 5432
      user: "{{ env_var('PROD_USER') }}"
      password: "{{ env_var('PROD_PASSWORD') }}"
      dbname: analytics
      schema: dbt_prod
      # 8 threads for prod: more parallelism since prod warehouses
      # have more compute capacity. Tune based on your warehouse's
      # concurrent query limit to avoid throttling.
      threads: 8
```

---

## 3. Models

### 3.1 Basic Models

```sql
-- models/staging/stg_orders.sql
-- Staging model: Clean source data
-- Staging models are a thin cleaning layer over raw sources. They handle
-- type casting and null filtering so that downstream models can assume
-- clean, consistently typed inputs without repeating this logic.

SELECT
    order_id,
    customer_id,
    -- Explicit CASTs prevent implicit type coercion surprises that differ
    -- across warehouse engines (e.g., Snowflake vs PostgreSQL date handling)
    CAST(order_date AS DATE) AS order_date,
    CAST(amount AS DECIMAL(10, 2)) AS amount,
    status,
    -- loaded_at enables data freshness monitoring: if this timestamp is stale,
    -- the pipeline likely failed upstream
    CURRENT_TIMESTAMP AS loaded_at
FROM {{ source('raw', 'orders') }}
WHERE order_id IS NOT NULL
```

```sql
-- models/staging/stg_customers.sql
SELECT
    customer_id,
    TRIM(first_name) AS first_name,
    TRIM(last_name) AS last_name,
    LOWER(email) AS email,
    created_at
FROM {{ source('raw', 'customers') }}
```

```sql
-- models/marts/core/fct_orders.sql
-- Fact table: Orders

{{
    config(
        materialized='table',
        unique_key='order_id',
        -- Partitioning by month enables partition pruning: queries filtering
        -- on order_date only scan the relevant monthly partitions, reducing
        -- query cost by 10-100x on large tables in BigQuery/Snowflake.
        partition_by={
            'field': 'order_date',
            'data_type': 'date',
            'granularity': 'month'
        }
    )
}}

-- ref() creates an explicit dependency graph: dbt will build stg_orders
-- before fct_orders, and `dbt run --select +fct_orders` automatically
-- includes all upstream models.
WITH orders AS (
    SELECT * FROM {{ ref('stg_orders') }}
),

customers AS (
    SELECT * FROM {{ ref('stg_customers') }}
)

SELECT
    o.order_id,
    o.customer_id,
    c.first_name,
    c.last_name,
    c.email,
    o.order_date,
    o.amount,
    o.status,
    -- Pre-computed derived columns avoid repetitive EXTRACT() calls in
    -- every downstream query and enable simple GROUP BY year/month filters
    EXTRACT(YEAR FROM o.order_date) AS order_year,
    EXTRACT(MONTH FROM o.order_date) AS order_month,
    -- Business-defined tiers: thresholds should match the segmentation
    -- agreed upon with the business team. Changing these requires
    -- rebuilding the table and updating downstream dashboards.
    CASE
        WHEN o.amount > 1000 THEN 'high'
        WHEN o.amount > 100 THEN 'medium'
        ELSE 'low'
    END AS order_tier
-- LEFT JOIN (not INNER) preserves orders even when customer data is missing,
-- preventing silent data loss from referential integrity gaps in the source
FROM orders o
LEFT JOIN customers c ON o.customer_id = c.customer_id
```

### 3.2 Source Definition

```yaml
# models/staging/_sources.yml
version: 2

sources:
  - name: raw
    description: "Raw data sources"
    database: raw_db
    schema: public
    tables:
      - name: orders
        description: "Raw orders table"
        columns:
          - name: order_id
            description: "Unique order ID"
            tests:
              - unique
              - not_null
          - name: customer_id
            description: "Customer ID"
          - name: amount
            description: "Order amount"

      - name: customers
        description: "Raw customers table"
        freshness:
          warn_after: {count: 12, period: hour}
          error_after: {count: 24, period: hour}
        loaded_at_field: updated_at
```

### 3.3 Materialization Types

```sql
-- View (default): no storage cost, always fresh, but recomputes every query.
-- Best for: staging models or dev environments where iteration speed matters.
{{ config(materialized='view') }}

-- Table: physical table rebuilt on every dbt run. Trades storage for fast reads.
-- Best for: mart models queried frequently by BI tools.
{{ config(materialized='table') }}

-- Incremental: only processes new/changed rows since last run, dramatically
-- reducing compute cost on large fact tables (e.g., 1B+ row event tables).
-- unique_key enables upsert semantics (update existing, insert new).
-- on_schema_change='append_new_columns' auto-adds new source columns
-- without requiring a full refresh — avoids pipeline breakage from upstream
-- schema evolution.
{{ config(
    materialized='incremental',
    unique_key='order_id',
    on_schema_change='append_new_columns'
) }}

SELECT *
FROM {{ source('raw', 'orders') }}
{% if is_incremental() %}
-- Only fetch rows newer than the latest already-loaded date.
-- {{ this }} refers to the current materialized table, creating a
-- self-referencing filter that makes the model idempotent.
WHERE order_date > (SELECT MAX(order_date) FROM {{ this }})
{% endif %}

-- Ephemeral: compiled inline as a CTE — zero warehouse objects created.
-- Best for: intermediate transformations referenced by only one downstream model.
-- Avoid for models referenced by many models (causes CTE duplication).
{{ config(materialized='ephemeral') }}
```

---

## 4. Tests

### 4.1 Schema Tests

```yaml
# models/marts/core/_schema.yml
version: 2

models:
  - name: fct_orders
    description: "Orders fact table"
    columns:
      - name: order_id
        description: "Unique order ID"
        tests:
          - unique
          - not_null

      - name: customer_id
        tests:
          - not_null
          - relationships:
              to: ref('dim_customers')
              field: customer_id

      - name: amount
        tests:
          - not_null
          - dbt_utils.expression_is_true:
              expression: ">= 0"

      - name: status
        tests:
          - accepted_values:
              values: ['pending', 'completed', 'cancelled', 'refunded']
```

### 4.2 Custom Tests

```sql
-- tests/assert_positive_amounts.sql
-- Verify all order amounts are positive

SELECT
    order_id,
    amount
FROM {{ ref('fct_orders') }}
WHERE amount < 0
```

```sql
-- macros/test_row_count_equal.sql
{% test row_count_equal(model, compare_model) %}

WITH model_count AS (
    SELECT COUNT(*) AS cnt FROM {{ model }}
),

compare_count AS (
    SELECT COUNT(*) AS cnt FROM {{ compare_model }}
)

SELECT
    m.cnt AS model_count,
    c.cnt AS compare_count
FROM model_count m
CROSS JOIN compare_count c
WHERE m.cnt != c.cnt

{% endtest %}
```

### 4.3 Running Tests

```bash
# Run all tests
dbt test

# Test specific model
dbt test --select fct_orders

# Test source freshness
dbt source freshness
```

---

## 5. Documentation

### 5.1 Documentation Definition

```yaml
# models/marts/core/_schema.yml
version: 2

models:
  - name: fct_orders
    description: |
      ## Orders Fact Table

      This table contains all order transactions.

      ### Use Cases
      - Daily/monthly revenue analysis
      - Customer purchase pattern analysis
      - Repeat purchase rate calculation

      ### Refresh Schedule
      - Daily at 06:00 UTC

    meta:
      owner: "data-team@company.com"
      contains_pii: false

    columns:
      - name: order_id
        description: "Unique order identifier (UUID)"
        meta:
          dimension: true

      - name: amount
        description: "Total order amount (USD)"
        meta:
          measure: true
          aggregation: sum
```

### 5.2 Generate and Serve Documentation

```bash
# Generate documentation
dbt docs generate

# Serve documentation
dbt docs serve --port 8080

# View at http://localhost:8080
```

---

## 6. Jinja Templates

### 6.1 Basic Jinja Syntax

```sql
-- Variables
{% set my_var = 'value' %}
SELECT '{{ my_var }}' AS col

-- Conditionals
SELECT
    CASE
        {% if target.name == 'prod' %}
        WHEN amount > 1000 THEN 'high'
        {% else %}
        WHEN amount > 100 THEN 'high'
        {% endif %}
        ELSE 'low'
    END AS tier
FROM orders

-- Loops
SELECT
    order_id,
    {% for col in ['amount', 'quantity', 'discount'] %}
    SUM({{ col }}) AS total_{{ col }}{% if not loop.last %},{% endif %}
    {% endfor %}
FROM order_items
GROUP BY order_id
```

### 6.2 Macros

```sql
-- macros/generate_schema_name.sql
-- Overrides dbt's default schema naming to prefix custom schemas with
-- the target schema (e.g., dbt_dev_staging). This prevents dev and prod
-- models from colliding in the same database, since each environment
-- has its own target.schema prefix.
{% macro generate_schema_name(custom_schema_name, node) -%}
    {%- set default_schema = target.schema -%}
    {%- if custom_schema_name is none -%}
        {{ default_schema }}
    {%- else -%}
        {{ default_schema }}_{{ custom_schema_name }}
    {%- endif -%}
{%- endmacro %}


-- macros/cents_to_dollars.sql
-- Centralizes currency conversion logic: if the precision or divisor changes
-- (e.g., switch from cents to basis points), update here once rather than
-- in every model that handles money.
{% macro cents_to_dollars(column_name, precision=2) %}
    ROUND({{ column_name }} / 100.0, {{ precision }})
{% endmacro %}


-- macros/limit_data_in_dev.sql
-- Automatically caps query results in dev to speed up iteration and reduce
-- warehouse costs. In prod (target.name != 'dev'), no LIMIT is applied,
-- so production models process the full dataset.
{% macro limit_data_in_dev() %}
    {% if target.name == 'dev' %}
        LIMIT 1000
    {% endif %}
{% endmacro %}
```

```sql
-- Using macros
SELECT
    order_id,
    {{ cents_to_dollars('amount_cents') }} AS amount_dollars
FROM orders
{{ limit_data_in_dev() }}
```

### 6.3 Built-in dbt Functions

```sql
-- ref(): Reference another model
SELECT * FROM {{ ref('stg_orders') }}

-- source(): Reference source table
SELECT * FROM {{ source('raw', 'orders') }}

-- this: Reference current model (useful in incremental)
{% if is_incremental() %}
SELECT MAX(updated_at) FROM {{ this }}
{% endif %}

-- config(): Access configuration values
{{ config.get('materialized') }}

-- target: Target environment information
{{ target.name }}    -- dev, prod
{{ target.schema }}  -- dbt_dev
{{ target.type }}    -- postgres, snowflake
```

---

## 7. Incremental Processing

### 7.1 Basic Incremental Model

```sql
-- models/marts/fct_events.sql
{{
    config(
        materialized='incremental',
        unique_key='event_id',
        -- delete+insert is chosen over merge because it performs better on
        -- warehouses that lack native MERGE support (e.g., older PostgreSQL).
        -- It deletes rows matching the unique_key, then inserts fresh versions,
        -- achieving upsert semantics in two steps.
        incremental_strategy='delete+insert'
    )
}}

SELECT
    event_id,
    user_id,
    event_type,
    event_data,
    created_at
FROM {{ source('raw', 'events') }}

{% if is_incremental() %}
-- Only process events newer than the latest already loaded.
-- This assumes events arrive in roughly chronological order; for out-of-order
-- events, consider a lookback window: created_at > MAX(created_at) - INTERVAL '1 hour'
WHERE created_at > (SELECT MAX(created_at) FROM {{ this }})
{% endif %}
```

### 7.2 Incremental Strategies

```sql
-- Append (default): Only add new data, never update existing rows.
-- Fastest strategy (no deduplication overhead) — ideal for immutable event logs
-- where duplicates are handled upstream or don't matter.
{{ config(
    materialized='incremental',
    incremental_strategy='append'
) }}

-- Delete+Insert: Delete by key then insert.
-- Works on all warehouses (no MERGE required). Use when you need upsert
-- semantics on databases like PostgreSQL that have limited MERGE support.
{{ config(
    materialized='incremental',
    unique_key='id',
    incremental_strategy='delete+insert'
) }}

-- Merge (Snowflake, BigQuery): Use MERGE statement.
-- Most flexible: atomically inserts new rows and updates existing ones.
-- merge_update_columns limits which columns are updated, preventing
-- accidental overwrites of columns managed by other processes.
{{ config(
    materialized='incremental',
    unique_key='id',
    incremental_strategy='merge',
    merge_update_columns=['name', 'amount', 'updated_at']
) }}
```

---

## 8. Execution Commands

### 8.1 Basic Commands

```bash
# Test connection
dbt debug

# Run all models
dbt run

# Run specific model only
dbt run --select fct_orders
dbt run --select staging.*
dbt run --select +fct_orders+  # Include dependencies

# Run tests
dbt test

# Build (run + test)
dbt build

# Load seed data
dbt seed

# Compile only (no execution)
dbt compile

# Clean
dbt clean
```

### 8.2 Selectors

```bash
# By model name
dbt run --select my_model

# By path
dbt run --select models/staging/*

# By tag
dbt run --select tag:daily

# Include upstream dependencies
dbt run --select +my_model

# Include downstream dependencies
dbt run --select my_model+

# Both directions
dbt run --select +my_model+

# Exclude specific model
dbt run --exclude my_model
```

---

## 9. Advanced dbt: Semantic Layer and Metrics

### 9.1 The Semantic Layer: A Single Source of Truth for Metrics

In traditional analytics workflows, business metrics like "revenue" or "active users" are defined independently in each BI tool, dashboard, or ad-hoc query. This leads to inconsistent numbers across teams — the CFO's revenue figure differs from the product team's because each used a slightly different SQL definition.

> **Analogy**: Think of the semantic layer as a **dictionary for your business metrics**. Just as a dictionary provides the single authoritative definition for each word, the semantic layer provides one canonical definition for each metric. Without it, every analyst writes their own "dialect" of revenue — and nobody agrees on what the word means.

The **dbt Semantic Layer** (available in dbt Cloud, powered by MetricFlow) solves this by defining metrics once in your dbt project and exposing them through a query API that any downstream tool can consume.

```
┌─────────────────────────────────────────────────────────────────┐
│                  Traditional vs Semantic Layer                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Traditional:                                                  │
│   ┌────────┐  each tool defines    ┌──────────┐                │
│   │Looker  │──its own "revenue"──→ │Dashboard1│  (revenue=$1M) │
│   │Tableau │──its own "revenue"──→ │Dashboard2│  (revenue=$1.2M)│
│   │Notebook│──its own "revenue"──→ │Report    │  (revenue=$980K)│
│   └────────┘                       └──────────┘                │
│                                                                 │
│   Semantic Layer:                                               │
│   ┌────────┐      ┌───────────┐    ┌──────────┐               │
│   │Looker  │─┐    │  dbt      │    │Dashboard1│               │
│   │Tableau │─┼───→│  Semantic │───→│Dashboard2│  (all=$1M)    │
│   │Notebook│─┘    │  Layer    │    │Report    │               │
│   └────────┘      └───────────┘    └──────────┘               │
│                   (single def)                                  │
└─────────────────────────────────────────────────────────────────┘
```

| Aspect | Traditional Approach | Semantic Layer Approach |
|--------|---------------------|------------------------|
| **Metric definition** | Duplicated in every BI tool / query | Defined once in dbt YAML |
| **Consistency** | Prone to drift across teams | Guaranteed single source of truth |
| **Governance** | Hard to audit who defined what | Version-controlled in Git |
| **Maintenance** | Change logic in N places | Change once, propagates everywhere |
| **Time-to-insight** | Analysts must know exact SQL | Query by metric name + dimensions |
| **Testing** | Manual / per-tool | dbt tests apply to metric definitions |

### 9.2 MetricFlow and Semantic Models

dbt acquired **Transform** (the company behind MetricFlow) in 2023 to power its semantic layer. MetricFlow compiles metric requests into optimized SQL. The building blocks are **semantic models** and **metrics**.

#### Semantic Model Definition

A semantic model maps a dbt model to semantic concepts: entities (join keys), dimensions (grouping columns), and measures (aggregatable columns).

```yaml
# models/marts/core/_semantic_models.yml
# Why semantic models? They declare the "meaning" of columns so that
# MetricFlow knows how to join tables and aggregate measures automatically.

semantic_models:
  - name: orders
    defaults:
      agg_time_dimension: order_date
    description: "Order transactions for metric computation"
    model: ref('fct_orders')

    entities:
      # Entities define join keys — MetricFlow uses these to
      # automatically join semantic models when a metric needs
      # dimensions from multiple tables.
      - name: order_id
        type: primary
      - name: customer_id
        type: foreign

    dimensions:
      - name: order_date
        type: time
        type_params:
          time_granularity: day
      - name: order_tier
        type: categorical
      - name: status
        type: categorical

    measures:
      # Measures are the raw aggregatable building blocks.
      # Metrics (defined separately) reference these measures.
      - name: order_count
        agg: count
        expr: order_id
      - name: total_revenue
        agg: sum
        expr: amount
      - name: average_order_value
        agg: average
        expr: amount
```

#### Metric Definitions

Metrics reference measures from semantic models and come in four types:

```yaml
# models/marts/core/_metrics.yml
metrics:
  # --- Simple metric: directly references one measure ---
  - name: revenue
    description: "Total revenue from completed orders"
    type: simple
    label: "Revenue"
    type_params:
      measure: total_revenue
    filter: |
      {{ Dimension('order_id__status') }} = 'completed'

  - name: order_count
    description: "Total number of orders"
    type: simple
    label: "Order Count"
    type_params:
      measure: order_count

  # --- Derived metric: combines other metrics with arithmetic ---
  # Why derived? AOV = revenue / order_count, but defining it as
  # derived ensures both numerator and denominator use the exact
  # same filters and grain, preventing subtle mismatches.
  - name: average_order_value
    description: "Average order value (revenue / orders)"
    type: derived
    label: "AOV"
    type_params:
      expr: revenue / order_count
      metrics:
        - name: revenue
        - name: order_count

  # --- Cumulative metric: running total over time ---
  - name: cumulative_revenue
    description: "Cumulative revenue year-to-date"
    type: cumulative
    label: "Cumulative Revenue"
    type_params:
      measure: total_revenue
      window: 1 year

  # --- Conversion metric: measures funnel conversion rates ---
  - name: checkout_conversion_rate
    description: "Ratio of orders to cart-creation events"
    type: conversion
    label: "Checkout Conversion"
    type_params:
      entity: customer_id
      calculation: conversions / opportunities
      base_measure: cart_creations    # opportunities
      conversion_measure: order_count # conversions
      window: 7 days
```

### 9.3 Querying the Semantic Layer

#### dbt Cloud Semantic Layer API

Once metrics are defined, downstream tools query them through the **Semantic Layer API** (GraphQL or JDBC) without writing raw SQL. MetricFlow compiles each request into optimized SQL behind the scenes.

```graphql
# Example: Query revenue by month and order tier
# The API consumer never writes SQL — just declares
# what metric, dimensions, and time grain they need.
{
  createQuery(
    metrics: [{name: "revenue"}]
    groupBy: [
      {name: "metric_time", grain: MONTH},
      {name: "order_tier"}
    ]
    where: [{sql: "{{ TimeDimension('metric_time', 'MONTH') }} >= '2024-01-01'"}]
    orderBy: [{name: "metric_time"}]
  ) {
    queryId
    result {
      data
    }
  }
}
```

```bash
# Using the dbt Cloud CLI to query metrics locally during development
# Why query locally? To validate metric definitions before deploying.
dbt sl query --metrics revenue --group-by metric_time__month,order_tier \
  --where "metric_time__month >= '2024-01-01'" --order-by metric_time__month
```

#### BI Tool Integration

| BI Tool | Integration Method | Notes |
|---------|-------------------|-------|
| **Looker** | dbt Semantic Layer connection | Native integration since Looker 2024.2 |
| **Tableau** | JDBC connector | Metrics appear as data source fields |
| **Hex** | Native dbt integration | Query metrics directly in notebooks |
| **Google Sheets** | Google Sheets add-on | Pull metrics into spreadsheets |
| **Custom apps** | GraphQL / JDBC API | Build internal tools against metrics |

### 9.4 dbt Mesh: Multi-Project dbt

As organizations scale, a single monolithic dbt project becomes difficult to manage. **dbt Mesh** (introduced in dbt 1.6+) enables multiple dbt projects to reference each other's models while maintaining clear ownership boundaries.

#### Cross-Project References

```yaml
# project_b/dbt_project.yml
# Why cross-project refs? Team A owns the core orders model,
# Team B (marketing analytics) needs to build on top of it
# without duplicating the SQL or breaking Team A's contract.
name: marketing_analytics
version: '1.0.0'

dependencies:
  - project: core_analytics
    # This declares a dependency on another dbt project.
    # dbt resolves the ref() at compile time across projects.
```

```sql
-- project_b/models/marts/marketing/mkt_campaign_attribution.sql
-- Cross-project ref: access core_analytics.fct_orders from this project.
-- The two-argument ref() tells dbt the model lives in another project.
SELECT
    o.order_id,
    o.customer_id,
    o.amount,
    c.campaign_id,
    c.utm_source
FROM {{ ref('core_analytics', 'fct_orders') }} o
LEFT JOIN {{ ref('stg_campaign_touches') }} c
    ON o.customer_id = c.customer_id
    AND o.order_date BETWEEN c.touch_date AND c.touch_date + INTERVAL '7 days'
```

#### Public Models and Contracts

```yaml
# project_a/models/marts/core/_models.yml
# Why contracts? Public models are consumed by other projects,
# so their schema is a promise — breaking it would cascade failures.
models:
  - name: fct_orders
    access: public          # Exposed to other projects (default is "protected")
    group: core_team        # Ownership group
    latest_version: 2       # Enables model versioning for safe migrations

    config:
      contract:
        enforced: true      # Columns must match the declared schema exactly

    columns:
      - name: order_id
        data_type: varchar
        description: "Primary key"
        constraints:
          - type: not_null
          - type: primary_key
      - name: amount
        data_type: numeric
        description: "Order amount in USD"
        constraints:
          - type: not_null
```

#### Groups and Access Control

```yaml
# dbt_project.yml — define ownership groups
groups:
  - name: core_team
    owner:
      name: "Core Data Team"
      email: "core-data@company.com"

  - name: marketing_team
    owner:
      name: "Marketing Analytics"
      email: "mkt-analytics@company.com"
```

```yaml
# models/marts/core/_models.yml
# Access levels control who can ref() a model:
#   - private:   only within the same group
#   - protected: only within the same project (default)
#   - public:    any project can reference it
models:
  - name: fct_orders
    access: public
    group: core_team

  - name: int_order_enriched
    access: protected      # Other projects cannot ref() this
    group: core_team

  - name: _stg_orders_deduped
    access: private        # Only core_team models can ref() this
    group: core_team
```

---

## Practice Problems

### Problem 1: Staging Model
Create a stg_products model from the raw products table. Convert prices to dollars and handle NULL values.

### Problem 2: Incremental Model
Write a model that incrementally processes daily sales aggregates.

### Problem 3: Write Tests
Write tests for the fct_sales model (unique, not_null, positive amount check).

---

## Summary

| Concept | Description |
|------|------|
| **Model** | SQL-based data transformation definition |
| **Source** | Raw data reference |
| **ref()** | Reference between models (automatic dependency management) |
| **Test** | Data quality validation |
| **Materialization** | view, table, incremental, ephemeral |
| **Macro** | Reusable SQL templates |

---

## References

- [dbt Documentation](https://docs.getdbt.com/)
- [dbt Learn](https://courses.getdbt.com/)
- [dbt Best Practices](https://docs.getdbt.com/guides/best-practices)
