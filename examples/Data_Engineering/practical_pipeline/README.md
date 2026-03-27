# Practical Data Pipeline Project

End-to-end e-commerce analytics pipeline using Airflow, Spark, dbt, and Great Expectations.

Companion code for **Data_Engineering Lesson 14**.

## Architecture

```
PostgreSQL (source) → Spark (extract/process) → dbt (transform) → PostgreSQL (warehouse)
       ↑                                              ↓
  generate_data.py                          Great Expectations (validate)
       ↑
  Airflow (orchestrate all)
```

## Prerequisites

- Docker & Docker Compose
- Python 3.9+

## Quick Start

```bash
# 1. Copy environment variables
cp .env.example .env

# 2. Start all services
docker compose up -d

# 3. Generate sample data (run from host or inside airflow container)
pip install -r requirements.txt
python init/generate_data.py

# 4. Initialize database schema
docker compose exec source-db psql -U ecommerce -d ecommerce -f /docker-entrypoint-initdb.d/init_db.sql

# 5. Access Airflow UI
open http://localhost:8080
# Login: admin / admin
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| Airflow Webserver | 8080 | DAG management UI |
| Spark Master | 8081 | Spark cluster UI |
| Source DB | 5433 | PostgreSQL (e-commerce data) |
| Warehouse DB | 5434 | PostgreSQL (analytics warehouse) |

## Directory Structure

```
practical_pipeline/
├── docker-compose.yml        # All services
├── .env.example              # Environment template
├── requirements.txt          # Python dependencies
├── init/
│   ├── init_db.sql           # Schema + seed data
│   └── generate_data.py      # Synthetic data generator
├── dags/
│   ├── daily_etl_dag.py      # Main ETL orchestration
│   └── data_quality_dag.py   # Quality validation
├── spark_jobs/
│   ├── extract_postgres.py   # Extract from source DB
│   ├── process_clickstream.py # Process JSON events
│   └── aggregate_daily.py    # Daily aggregations
├── dbt_project/
│   ├── dbt_project.yml       # dbt configuration
│   ├── profiles.yml          # Database connection
│   └── models/
│       ├── staging/stg_orders.sql
│       └── marts/fct_daily_sales.sql
└── great_expectations/
    ├── great_expectations.yml
    └── expectations/orders_suite.json
```

## Customization

- **Scale data**: Edit `NUM_ORDERS` in `init/generate_data.py`
- **Add sources**: Extend `docker-compose.yml` with Kafka/MinIO
- **Cloud migration**: Replace local paths with S3/GCS URIs, swap PostgreSQL warehouse with Snowflake/BigQuery
