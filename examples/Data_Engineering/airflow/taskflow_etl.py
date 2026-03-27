"""
TaskFlow API ETL Pipeline Example
==================================
A complete ETL pipeline using Airflow 2.x TaskFlow API with:
- @dag and @task decorators
- Dynamic task mapping (expand/partial)
- TaskGroups for organization
- Multiple outputs
- Error handling

Usage:
    Place this file in your Airflow DAGs folder.
    Requires: apache-airflow >= 2.4
"""

from airflow.decorators import dag, task, task_group
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)

# Default arguments for all tasks
default_args = {
    'owner': 'data_team',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(minutes=30),
}


@dag(
    schedule='@daily',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=['taskflow', 'etl', 'example'],
    description='TaskFlow API ETL pipeline with dynamic mapping',
)
def taskflow_etl_pipeline():
    """
    E-commerce data ETL pipeline using TaskFlow API.
    Demonstrates modern Airflow patterns.
    """

    # ── Extract Phase ──────────────────────────────────────────────

    @task_group()
    def extract():
        """Extract data from multiple sources."""

        @task()
        def get_data_sources() -> list[dict]:
            """Discover available data sources."""
            return [
                {"name": "users", "type": "api", "endpoint": "/api/v2/users"},
                {"name": "orders", "type": "database", "table": "raw.orders"},
                {"name": "products", "type": "s3", "path": "s3://bucket/products/"},
            ]

        @task(multiple_outputs=True)
        def extract_source(source: dict) -> dict:
            """Extract data from a single source (dynamically mapped)."""
            logger.info(f"Extracting from {source['name']} ({source['type']})")

            # Simulate extraction (replace with real extraction logic)
            if source['type'] == 'api':
                records = [{"id": i, "name": f"user_{i}"} for i in range(100)]
            elif source['type'] == 'database':
                records = [{"id": i, "amount": i * 10.5} for i in range(500)]
            elif source['type'] == 's3':
                records = [{"sku": f"P{i:04d}", "price": i * 5.0} for i in range(50)]
            else:
                records = []

            return {
                "source_name": source['name'],
                "record_count": len(records),
                "sample": records[:3],  # Only pass metadata, not full data
            }

        sources = get_data_sources()
        # Dynamic mapping: one extract_source task per data source
        return extract_source.expand(source=sources)

    # ── Transform Phase ────────────────────────────────────────────

    @task_group()
    def transform(extractions: list[dict]):
        """Transform and validate extracted data."""

        @task()
        def validate(extractions: list[dict]) -> dict:
            """Validate all extractions completed successfully."""
            total_records = sum(e['record_count'] for e in extractions)
            sources = [e['source_name'] for e in extractions]

            logger.info(f"Validated {len(extractions)} sources, {total_records} total records")

            if total_records == 0:
                raise ValueError("No records extracted from any source!")

            return {
                "sources": sources,
                "total_records": total_records,
                "status": "validated",
            }

        @task()
        def compute_metrics(validation: dict) -> dict:
            """Compute business metrics from validated data."""
            return {
                "sources_count": len(validation['sources']),
                "total_records": validation['total_records'],
                "metrics_computed": True,
                "pipeline_date": "{{ ds }}",
            }

        validated = validate(extractions)
        return compute_metrics(validated)

    # ── Load Phase ─────────────────────────────────────────────────

    @task()
    def load(metrics: dict):
        """Load transformed data to the warehouse."""
        logger.info(
            f"Loading metrics: {metrics['total_records']} records "
            f"from {metrics['sources_count']} sources"
        )
        # In production: write to database, S3, etc.
        print(json.dumps(metrics, indent=2, default=str))

    @task()
    def notify(metrics: dict):
        """Send pipeline completion notification."""
        message = (
            f"Pipeline completed: {metrics['total_records']} records "
            f"from {metrics['sources_count']} sources"
        )
        logger.info(f"Notification: {message}")
        # In production: send Slack/email notification

    # ── DAG Flow ───────────────────────────────────────────────────

    raw_data = extract()
    metrics = transform(raw_data)
    load(metrics)
    notify(metrics)


# Instantiate the DAG
taskflow_etl_pipeline()
