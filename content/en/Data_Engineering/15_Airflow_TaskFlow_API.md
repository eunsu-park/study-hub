# Airflow TaskFlow API

## Overview

The TaskFlow API, introduced in Airflow 2.0, provides a Python-native way to define DAGs using decorators. It replaces the traditional Operator-based pattern with `@task` decorators, enabling automatic XCom passing, cleaner code, and better type safety. This is the modern standard for writing Airflow DAGs.

---

## 1. TaskFlow vs Traditional Operators

### 1.1 Side-by-Side Comparison

```python
"""
=== Traditional Operator Pattern (Airflow 1.x style) ===
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def extract_fn(**kwargs):
    data = {"user_count": 100, "revenue": 5000}
    kwargs['ti'].xcom_push(key='extracted_data', value=data)

def transform_fn(**kwargs):
    ti = kwargs['ti']
    data = ti.xcom_pull(task_ids='extract', key='extracted_data')
    data['revenue_per_user'] = data['revenue'] / data['user_count']
    ti.xcom_push(key='transformed_data', value=data)

def load_fn(**kwargs):
    ti = kwargs['ti']
    data = ti.xcom_pull(task_ids='transform', key='transformed_data')
    print(f"Loading data: {data}")

with DAG('traditional_etl', start_date=datetime(2024, 1, 1), schedule='@daily') as dag:
    extract = PythonOperator(task_id='extract', python_callable=extract_fn)
    transform = PythonOperator(task_id='transform', python_callable=transform_fn)
    load = PythonOperator(task_id='load', python_callable=load_fn)
    extract >> transform >> load
```

```python
"""
=== TaskFlow API Pattern (Airflow 2.x modern style) ===
"""
from airflow.decorators import dag, task
from datetime import datetime

@dag(schedule='@daily', start_date=datetime(2024, 1, 1), catchup=False)
def taskflow_etl():

    @task()
    def extract() -> dict:
        return {"user_count": 100, "revenue": 5000}

    @task()
    def transform(data: dict) -> dict:
        data['revenue_per_user'] = data['revenue'] / data['user_count']
        return data

    @task()
    def load(data: dict):
        print(f"Loading data: {data}")

    # Dependencies are inferred from function calls!
    raw_data = extract()
    transformed = transform(raw_data)
    load(transformed)

taskflow_etl()  # Instantiate the DAG
```

### 1.2 Key Differences

| Aspect | Traditional | TaskFlow API |
|--------|------------|--------------|
| **Task definition** | `PythonOperator(...)` | `@task()` decorator |
| **XCom passing** | Manual `xcom_push/pull` | Automatic via return values |
| **Dependencies** | Explicit `>>` operators | Inferred from function calls |
| **Type hints** | Not enforced | Supported (return type annotations) |
| **Code readability** | Boilerplate-heavy | Pythonic, concise |
| **DAG definition** | `with DAG(...) as dag:` | `@dag()` decorator |

---

## 2. @task Decorator Basics

### 2.1 Return Values and Automatic XCom

```python
from airflow.decorators import dag, task
from datetime import datetime
import json

@dag(schedule='@daily', start_date=datetime(2024, 1, 1), catchup=False,
     tags=['taskflow', 'example'])
def xcom_demo():

    @task()
    def generate_data() -> dict:
        """Return value is automatically stored as XCom."""
        return {
            "users": [
                {"id": 1, "name": "Alice", "score": 95},
                {"id": 2, "name": "Bob", "score": 87},
                {"id": 3, "name": "Charlie", "score": 92},
            ]
        }

    @task()
    def compute_stats(data: dict) -> dict:
        """Input is automatically pulled from XCom."""
        scores = [u["score"] for u in data["users"]]
        return {
            "mean": sum(scores) / len(scores),
            "max": max(scores),
            "min": min(scores),
            "count": len(scores),
        }

    @task()
    def report(stats: dict):
        print(f"Stats: mean={stats['mean']:.1f}, "
              f"max={stats['max']}, min={stats['min']}, "
              f"count={stats['count']}")

    data = generate_data()
    stats = compute_stats(data)
    report(stats)

xcom_demo()
```

### 2.2 Multiple Outputs

```python
from airflow.decorators import dag, task
from datetime import datetime

@dag(schedule='@daily', start_date=datetime(2024, 1, 1), catchup=False)
def multiple_outputs_demo():

    @task(multiple_outputs=True)
    def split_data() -> dict:
        """Each key in the returned dict becomes a separate XCom."""
        return {
            "users": [{"id": 1}, {"id": 2}],
            "metadata": {"source": "api", "timestamp": "2024-01-01"},
            "count": 2,
        }

    @task()
    def process_users(users: list):
        print(f"Processing {len(users)} users")

    @task()
    def process_metadata(metadata: dict):
        print(f"Source: {metadata['source']}")

    # Access individual keys from the dict
    result = split_data()
    process_users(result["users"])
    process_metadata(result["metadata"])

multiple_outputs_demo()
```

### 2.3 Task with Custom Parameters

```python
from airflow.decorators import dag, task
from datetime import datetime

@dag(schedule='@daily', start_date=datetime(2024, 1, 1), catchup=False)
def custom_params_demo():

    @task(
        task_id='custom_extract',
        retries=3,
        retry_delay=60,  # seconds
        execution_timeout=300,  # 5 minutes
        pool='data_pool',
        queue='high_priority',
    )
    def extract(source: str, limit: int = 100) -> list:
        """Task with custom Airflow parameters and function arguments."""
        print(f"Extracting from {source} with limit {limit}")
        return [{"id": i} for i in range(limit)]

    @task(trigger_rule='all_success')
    def validate(records: list) -> bool:
        assert len(records) > 0, "No records extracted"
        return True

    data = extract(source="api_v2", limit=50)
    validate(data)

custom_params_demo()
```

---

## 3. TaskFlow with Different Runtimes

### 3.1 Runtime Decorators

```python
from airflow.decorators import dag, task
from datetime import datetime

@dag(schedule='@daily', start_date=datetime(2024, 1, 1), catchup=False)
def runtime_demo():

    # Standard Python task (runs in Airflow worker)
    @task()
    def python_task():
        return {"source": "python"}

    # Virtual environment task (isolated dependencies)
    @task.virtualenv(
        requirements=["pandas==2.1.0", "numpy==1.25.0"],
        python_version="3.11",
        system_site_packages=False,
    )
    def virtualenv_task():
        import pandas as pd
        import numpy as np
        df = pd.DataFrame(np.random.randn(100, 3), columns=['a', 'b', 'c'])
        return {"rows": len(df), "mean_a": float(df['a'].mean())}

    # Docker task (containerized execution)
    @task.docker(
        image="python:3.11-slim",
        auto_remove="success",
        mount_tmp_dir=False,
    )
    def docker_task():
        import json
        result = {"environment": "docker", "status": "ok"}
        print(json.dumps(result))

    # Branch task (conditional execution)
    @task.branch()
    def decide_path() -> str:
        import random
        return "fast_path" if random.random() > 0.5 else "slow_path"

    @task(task_id="fast_path")
    def fast():
        return "fast result"

    @task(task_id="slow_path")
    def slow():
        return "slow result"

    # Build DAG
    py_result = python_task()
    venv_result = virtualenv_task()
    branch = decide_path()
    fast()
    slow()

runtime_demo()
```

---

## 4. Dynamic Task Mapping

### 4.1 expand() for Dynamic Parallelism

```python
from airflow.decorators import dag, task
from datetime import datetime

@dag(schedule='@daily', start_date=datetime(2024, 1, 1), catchup=False)
def dynamic_mapping_demo():

    @task()
    def get_partitions() -> list[str]:
        """Returns a list — each element generates a mapped task instance."""
        return ["us-east", "us-west", "eu-west", "ap-south"]

    @task()
    def process_partition(partition: str) -> dict:
        """This task runs once per partition (4 parallel instances)."""
        import random
        records = random.randint(100, 1000)
        return {"partition": partition, "records": records}

    @task()
    def aggregate(results: list[dict]) -> dict:
        """Receives a list of all mapped task outputs."""
        total = sum(r["records"] for r in results)
        return {"total_records": total, "partitions": len(results)}

    # Dynamic mapping: expand creates one task instance per list element
    partitions = get_partitions()
    processed = process_partition.expand(partition=partitions)
    aggregate(processed)

dynamic_mapping_demo()
```

### 4.2 expand() with partial()

```python
from airflow.decorators import dag, task
from datetime import datetime

@dag(schedule='@daily', start_date=datetime(2024, 1, 1), catchup=False)
def partial_expand_demo():

    @task()
    def get_files() -> list[str]:
        return ["data_2024_01.csv", "data_2024_02.csv", "data_2024_03.csv"]

    @task()
    def process_file(file_path: str, output_format: str, validate: bool) -> dict:
        """Process a file with shared configuration."""
        return {
            "file": file_path,
            "format": output_format,
            "validated": validate,
            "status": "success",
        }

    files = get_files()

    # partial() provides shared arguments, expand() provides varying arguments
    results = process_file.partial(
        output_format="parquet",  # Same for all instances
        validate=True,            # Same for all instances
    ).expand(
        file_path=files,          # Different for each instance
    )

partial_expand_demo()
```

### 4.3 Mapping Over Multiple Arguments

```python
from airflow.decorators import dag, task
from datetime import datetime

@dag(schedule='@daily', start_date=datetime(2024, 1, 1), catchup=False)
def zip_mapping_demo():

    @task()
    def get_configs() -> list[dict]:
        return [
            {"table": "users", "schema": "public"},
            {"table": "orders", "schema": "sales"},
            {"table": "products", "schema": "catalog"},
        ]

    @task()
    def extract_table(config: dict) -> dict:
        return {
            "table": f"{config['schema']}.{config['table']}",
            "rows": 1000,
        }

    @task()
    def summarize(extractions: list[dict]):
        for e in extractions:
            print(f"Extracted {e['rows']} from {e['table']}")

    configs = get_configs()
    results = extract_table.expand(config=configs)
    summarize(results)

zip_mapping_demo()
```

---

## 5. TaskFlow with TaskGroups

### 5.1 Organizing Tasks

```python
from airflow.decorators import dag, task, task_group
from datetime import datetime

@dag(schedule='@daily', start_date=datetime(2024, 1, 1), catchup=False)
def taskgroup_demo():

    @task_group()
    def extract_group():
        """Group related extraction tasks."""

        @task()
        def extract_users() -> list:
            return [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]

        @task()
        def extract_orders() -> list:
            return [{"id": 101, "user_id": 1, "amount": 50.0}]

        @task()
        def extract_products() -> list:
            return [{"id": "P1", "name": "Widget", "price": 25.0}]

        return {
            "users": extract_users(),
            "orders": extract_orders(),
            "products": extract_products(),
        }

    @task_group()
    def transform_group(data: dict):
        """Group related transformation tasks."""

        @task()
        def enrich_orders(orders: list, users: list) -> list:
            user_map = {u["id"]: u["name"] for u in users}
            for order in orders:
                order["user_name"] = user_map.get(order["user_id"], "Unknown")
            return orders

        @task()
        def calculate_totals(orders: list) -> dict:
            return {"total_revenue": sum(o["amount"] for o in orders)}

        enriched = enrich_orders(data["orders"], data["users"])
        return calculate_totals(enriched)

    @task()
    def load(totals: dict):
        print(f"Total revenue: ${totals['total_revenue']:.2f}")

    raw_data = extract_group()
    totals = transform_group(raw_data)
    load(totals)

taskgroup_demo()
```

---

## 6. Mixing TaskFlow with Traditional Operators

### 6.1 Hybrid DAGs

```python
from airflow.decorators import dag, task
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
from datetime import datetime

@dag(schedule='@daily', start_date=datetime(2024, 1, 1), catchup=False)
def hybrid_dag():

    # Sensor (traditional operator)
    wait_for_file = FileSensor(
        task_id='wait_for_file',
        filepath='/data/input/daily_{{ ds }}.csv',
        poke_interval=60,
        timeout=3600,
    )

    # TaskFlow task
    @task()
    def process_file(ds=None) -> dict:
        """ds is automatically injected by Airflow (logical date)."""
        return {"file": f"/data/input/daily_{ds}.csv", "rows": 1000}

    # SQL task (traditional operator)
    create_table = PostgresOperator(
        task_id='create_staging_table',
        postgres_conn_id='warehouse',
        sql="""
            CREATE TABLE IF NOT EXISTS staging.daily_data (
                id SERIAL PRIMARY KEY,
                data JSONB,
                loaded_at TIMESTAMP DEFAULT NOW()
            );
        """,
    )

    # TaskFlow task
    @task()
    def load_to_staging(metadata: dict):
        print(f"Loading {metadata['rows']} rows from {metadata['file']}")

    # Bash task (traditional operator)
    cleanup = BashOperator(
        task_id='cleanup',
        bash_command='echo "Cleaning up temp files"',
    )

    # Mix traditional and TaskFlow dependencies
    metadata = process_file()
    wait_for_file >> metadata >> create_table >> load_to_staging(metadata) >> cleanup

hybrid_dag()
```

---

## 7. Testing TaskFlow DAGs

### 7.1 Unit Testing Tasks

```python
"""
TaskFlow tasks can be tested as regular Python functions.
"""
import pytest

# The decorated task function
from airflow.decorators import task

@task()
def calculate_metrics(data: list) -> dict:
    if not data:
        raise ValueError("Empty data")
    values = [d["value"] for d in data]
    return {
        "mean": sum(values) / len(values),
        "count": len(values),
        "total": sum(values),
    }

# Test by calling the underlying function directly
def test_calculate_metrics():
    data = [{"value": 10}, {"value": 20}, {"value": 30}]
    # Access the raw function via .function
    result = calculate_metrics.function(data)
    assert result["mean"] == 20.0
    assert result["count"] == 3
    assert result["total"] == 60

def test_calculate_metrics_empty():
    with pytest.raises(ValueError, match="Empty data"):
        calculate_metrics.function([])

# Run: pytest test_tasks.py -v
```

### 7.2 DAG Validation Testing

```python
"""
Test that DAG loads without errors and has correct structure.
"""
import pytest
from airflow.models import DagBag

def test_dag_loaded():
    """Test DAG file can be imported without errors."""
    dagbag = DagBag(dag_folder='dags/', include_examples=False)
    assert len(dagbag.import_errors) == 0, f"DAG import errors: {dagbag.import_errors}"

def test_dag_structure():
    """Test DAG has expected tasks."""
    dagbag = DagBag(dag_folder='dags/', include_examples=False)
    dag = dagbag.get_dag('taskflow_etl')

    assert dag is not None
    task_ids = [t.task_id for t in dag.tasks]
    assert 'extract' in task_ids
    assert 'transform' in task_ids
    assert 'load' in task_ids

def test_dag_dependencies():
    """Test task dependencies are correct."""
    dagbag = DagBag(dag_folder='dags/', include_examples=False)
    dag = dagbag.get_dag('taskflow_etl')

    extract = dag.get_task('extract')
    transform = dag.get_task('transform')
    assert 'extract' in [t.task_id for t in transform.upstream_list]
```

---

## 8. Migration Guide: Traditional to TaskFlow

### 8.1 Step-by-Step Migration

```python
"""
Migration Checklist:

1. Replace `with DAG(...):` with `@dag(...)` decorator
2. Replace PythonOperator with @task() decorator
3. Remove manual xcom_push/xcom_pull
4. Return values instead of pushing XCom
5. Accept function parameters instead of pulling XCom
6. Let dependency be inferred from function calls
7. Remove explicit >> operators (unless mixing with traditional)

Common Pitfalls:
- @task functions must be serializable (no lambdas, no closures over non-serializable objects)
- Return values are stored in XCom (default: metadata DB) — avoid large returns
  → Use custom XCom backend for large data (S3, GCS)
- @task functions run in the worker process — avoid global state
"""
```

### 8.2 Custom XCom Backend for Large Data

```python
"""
Default XCom stores data in the Airflow metadata DB (serialized JSON/pickle).
For large datasets, use a custom XCom backend:

# airflow.cfg
[core]
xcom_backend = airflow.providers.amazon.aws.xcom_backends.s3.S3XComBackend

# Environment variables
AIRFLOW__CORE__XCOM_BACKEND=airflow.providers.amazon.aws.xcom_backends.s3.S3XComBackend
XCOM_BACKEND_BUCKET_NAME=my-airflow-xcom
XCOM_BACKEND_PREFIX=xcom/

Now @task return values are automatically stored in S3 instead of the DB.

Alternative backends:
  - S3: airflow.providers.amazon.aws.xcom_backends.s3.S3XComBackend
  - GCS: airflow.providers.google.cloud.xcom_backends.gcs.GCSXComBackend
  - Custom: Subclass BaseXCom and implement serialize/deserialize
"""
```

---

## 9. Best Practices

### 9.1 TaskFlow Guidelines

```python
"""
1. PREFER TaskFlow for Python tasks
   ✓ @task() for Python logic
   ✓ Traditional operators for external systems (SQL, Bash, sensors)

2. KEEP tasks small and focused
   ✓ Each task does one thing
   ✗ Avoid monolithic tasks that do extract + transform + load

3. USE type hints
   ✓ def extract() -> dict:
   ✓ def transform(data: dict) -> list:

4. AVOID large XCom values
   ✓ Pass metadata/references (file paths, S3 keys)
   ✗ Pass entire DataFrames or large datasets

5. USE task_group for organization
   ✓ Group related tasks (extract_group, transform_group)
   ✓ Keeps the DAG graph readable

6. USE dynamic mapping for parallel processing
   ✓ process_partition.expand(partition=get_partitions())
   ✗ Hardcoded parallel tasks (process_1, process_2, ...)

7. TEST tasks as functions
   ✓ calculate_metrics.function(test_data)
   ✓ pytest for both function logic and DAG structure
"""
```

---

## 10. Practice Problems

### Exercise 1: Convert a Traditional DAG

```python
"""
Convert this traditional DAG to TaskFlow API:

with DAG('legacy_pipeline', ...) as dag:
    def fetch(**ctx):
        ctx['ti'].xcom_push(key='records', value=[1, 2, 3, 4, 5])

    def double(**ctx):
        records = ctx['ti'].xcom_pull(task_ids='fetch', key='records')
        ctx['ti'].xcom_push(key='doubled', value=[r * 2 for r in records])

    def save(**ctx):
        data = ctx['ti'].xcom_pull(task_ids='double', key='doubled')
        print(f"Saving {data}")

    t1 = PythonOperator(task_id='fetch', python_callable=fetch)
    t2 = PythonOperator(task_id='double', python_callable=double)
    t3 = PythonOperator(task_id='save', python_callable=save)
    t1 >> t2 >> t3

Requirements:
1. Use @dag and @task decorators
2. Use return values instead of xcom_push/pull
3. Add type hints
4. Add a task_group for fetch + double
"""
```

### Exercise 2: Dynamic ETL Pipeline

```python
"""
Build a TaskFlow DAG that:
1. Reads a config file listing 5 database tables to sync
2. Uses dynamic task mapping to process each table in parallel
3. Each mapped task: extracts row count + schema info
4. An aggregation task collects all results
5. A final task generates a sync report

Use: @task, expand(), partial(), task_group
"""
```

---

## 11. Summary

### Key Takeaways

| Concept | Description |
|---------|-------------|
| **@task decorator** | Define tasks as decorated Python functions |
| **Automatic XCom** | Return values are automatically stored and passed |
| **@dag decorator** | Define DAGs as decorated functions |
| **Dynamic mapping** | `expand()` creates parallel task instances at runtime |
| **partial()** | Share common arguments across mapped tasks |
| **task_group** | Organize related tasks visually and logically |
| **Runtime variants** | `@task.virtualenv`, `@task.docker`, `@task.branch` |
| **Hybrid DAGs** | Mix TaskFlow with traditional operators freely |

### Best Practices

1. **Use TaskFlow** for all Python tasks — it's the modern standard
2. **Keep return values small** — pass references, not data
3. **Use dynamic mapping** instead of hardcoded parallel tasks
4. **Test tasks as functions** — `my_task.function(args)` for unit testing
5. **Migrate gradually** — mix old and new styles in the same DAG

### Next Steps

- **L16**: Kafka Streams and ksqlDB — real-time stream processing
- **L17**: Spark Structured Streaming — streaming with Spark DataFrames
