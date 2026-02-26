# Airflow TaskFlow API

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the key differences between the traditional Operator-based pattern and the TaskFlow API in Airflow 2.x, and justify when to use each approach.
2. Define DAGs and tasks using `@dag` and `@task` decorators, with automatic XCom passing between tasks via function return values.
3. Implement dynamic task mapping with `expand()` and `partial()` to generate parameterized task instances at runtime.
4. Apply task group organization and dependency management to build maintainable, modular DAG structures.
5. Configure task-level retries, timeouts, SLA callbacks, and failure handlers for production-grade pipeline reliability.
6. Integrate the TaskFlow API with external operators (BashOperator, SparkSubmitOperator) and sensors within a single DAG.

---

## Overview

The TaskFlow API, introduced in Airflow 2.0, provides a Python-native way to define DAGs using decorators. It replaces the traditional Operator-based pattern with `@task` decorators, enabling automatic XCom passing, cleaner code, and better type safety. This is the modern standard for writing Airflow DAGs.

---

## 1. TaskFlow vs Traditional Operators

### 1.1 Side-by-Side Comparison

```python
"""
=== Traditional Operator Pattern (Airflow 1.x style) ===
Notice how much boilerplate is needed: manual xcom_push/pull in every function,
string-based task_id references (error-prone), and no type safety on the data
being passed between tasks. The TaskFlow pattern below solves all of these.
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def extract_fn(**kwargs):
    data = {"user_count": 100, "revenue": 5000}
    # Must manually push to XCom — easy to forget or misspell the key
    kwargs['ti'].xcom_push(key='extracted_data', value=data)

def transform_fn(**kwargs):
    ti = kwargs['ti']
    # Must manually pull and specify the exact task_id string — if the task
    # is renamed, this silently returns None instead of raising an error
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
Same ETL logic as above, but XCom passing is automatic (via return values),
dependencies are inferred from function call chains, and type hints document
the data contract between tasks.
"""
from airflow.decorators import dag, task
from datetime import datetime

# catchup=False prevents Airflow from running all past dates since start_date
# on first deployment — without this, a DAG created today with start_date=2024-01-01
# would immediately queue 400+ backfill runs.
@dag(schedule='@daily', start_date=datetime(2024, 1, 1), catchup=False)
def taskflow_etl():

    @task()
    def extract() -> dict:
        # Return value is automatically serialized to XCom (JSON by default)
        return {"user_count": 100, "revenue": 5000}

    @task()
    def transform(data: dict) -> dict:
        # `data` is automatically deserialized from XCom — no manual pull needed
        data['revenue_per_user'] = data['revenue'] / data['user_count']
        return data

    @task()
    def load(data: dict):
        print(f"Loading data: {data}")

    # Dependencies are inferred from function calls — passing `raw_data` to
    # transform() tells Airflow that transform depends on extract.
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

    # multiple_outputs=True stores each dict key as a separate XCom entry,
    # enabling downstream tasks to depend on specific keys rather than the
    # entire dict. This avoids pulling unnecessary data and enables partial
    # dependency: process_users only re-runs if "users" changes.
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

    # Dict key access creates fine-grained dependencies: process_users
    # depends only on the "users" XCom, not on "metadata" or "count"
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
        # execution_timeout prevents runaway tasks (e.g., stuck API calls)
        # from consuming worker slots indefinitely
        execution_timeout=300,  # 5 minutes
        # pool limits concurrency: 'data_pool' might have 5 slots, preventing
        # this DAG from overwhelming the source database with parallel reads
        pool='data_pool',
        # queue routes the task to specific workers (e.g., workers with
        # more memory or network access to the data source)
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

    # Standard Python task (runs in Airflow worker process).
    # Fastest startup but shares the worker's Python environment —
    # dependency conflicts with Airflow itself are possible.
    @task()
    def python_task():
        return {"source": "python"}

    # Virtual environment task: creates an isolated Python env per execution.
    # Solves dependency conflicts (e.g., task needs pandas 2.1 but Airflow
    # requires an older version). Trade-off: ~10-30s startup overhead for
    # venv creation on each run.
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

    # Docker task: full container isolation — guarantees reproducible execution
    # regardless of the host environment. Ideal for tasks with complex native
    # dependencies (C libraries, GPU drivers). Higher startup cost than virtualenv.
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

    # Dynamic mapping with expand(): the number of task instances is determined
    # at runtime from the upstream task's output — not hardcoded in the DAG file.
    # This is critical when the partition list changes (e.g., new regions added)
    # without requiring a DAG code change and redeploy.
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

    # partial() fixes arguments shared across all mapped instances, while expand()
    # varies the per-instance argument. This separation avoids repeating config
    # in every mapped call and makes it easy to change shared settings in one place.
    # Without partial(), you'd need to zip shared values with the varying list.
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

    # @task_group creates a visual boundary in the Airflow UI Graph view.
    # Within extract_group, all three extracts run in parallel (no dependencies
    # between them), maximizing throughput. The group itself acts as a single
    # node when connecting to downstream groups.
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

        # Returning a dict of task outputs lets the transform_group
        # access each dataset by key, maintaining clear data contracts
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
            # Build a lookup map for O(1) user resolution instead of
            # O(n*m) nested loops — important for large datasets
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

    # Sensor (traditional operator) — waits for an external dependency.
    # Sensors have no TaskFlow equivalent because they poll external state,
    # not execute Python logic. poke_interval=60 checks every minute;
    # timeout=3600 fails the task if the file hasn't appeared after 1 hour.
    wait_for_file = FileSensor(
        task_id='wait_for_file',
        filepath='/data/input/daily_{{ ds }}.csv',
        poke_interval=60,
        timeout=3600,
    )

    # TaskFlow task — ds=None is a special Airflow pattern: at runtime,
    # Airflow injects the logical execution date as a string. This enables
    # the task to process the correct date's file without hardcoding.
    @task()
    def process_file(ds=None) -> dict:
        """ds is automatically injected by Airflow (logical date)."""
        return {"file": f"/data/input/daily_{ds}.csv", "rows": 1000}

    # SQL task — PostgresOperator is preferred over @task() for SQL because
    # it uses the Airflow connection manager (encrypted credentials, connection
    # pooling) and renders the SQL in Airflow's UI for debugging.
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

    # Mixing traditional operators and TaskFlow tasks in the same dependency chain
    # is fully supported. The >> operator connects them seamlessly — Airflow
    # treats both as first-class Task instances regardless of how they were defined.
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

# Test by calling the underlying function directly.
# .function bypasses the Airflow decorator, so you can test pure business logic
# without spinning up an Airflow environment — tests run in milliseconds.
def test_calculate_metrics():
    data = [{"value": 10}, {"value": 20}, {"value": 30}]
    result = calculate_metrics.function(data)
    assert result["mean"] == 20.0
    assert result["count"] == 3
    assert result["total"] == 60

def test_calculate_metrics_empty():
    # Testing the error path is critical: if empty data silently passes,
    # downstream tasks would process garbage and produce incorrect results
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

## Exercises

### Exercise 1: Migrate a Legacy DAG to TaskFlow

Take the following traditional DAG and convert it to TaskFlow API:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def score_users(**kwargs):
    ti = kwargs['ti']
    users = ti.xcom_pull(task_ids='load_users', key='users')
    scores = {u['id']: u['clicks'] * 2 + u['purchases'] * 10 for u in users}
    ti.xcom_push(key='scores', value=scores)

def filter_vip(**kwargs):
    ti = kwargs['ti']
    scores = ti.xcom_pull(task_ids='score_users', key='scores')
    vips = {uid: s for uid, s in scores.items() if s >= 50}
    ti.xcom_push(key='vips', value=vips)

def send_campaign(**kwargs):
    ti = kwargs['ti']
    vips = ti.xcom_pull(task_ids='filter_vip', key='vips')
    print(f"Sending campaign to {len(vips)} VIP users")

with DAG('legacy_campaign', start_date=datetime(2024, 1, 1), schedule='@weekly') as dag:
    t1 = PythonOperator(task_id='load_users', python_callable=lambda **kw: kw['ti'].xcom_push(key='users', value=[]))
    t2 = PythonOperator(task_id='score_users', python_callable=score_users)
    t3 = PythonOperator(task_id='filter_vip', python_callable=filter_vip)
    t4 = PythonOperator(task_id='send_campaign', python_callable=send_campaign)
    t1 >> t2 >> t3 >> t4
```

Requirements:
1. Use `@dag` and `@task` decorators with proper type hints
2. Replace all `xcom_push`/`xcom_pull` with automatic return-value passing
3. Wrap `score_users` and `filter_vip` in a `@task_group` called `scoring_group`
4. Add `retries=2` and `execution_timeout=120` to the `send_campaign` task
5. Write a unit test using `.function()` that verifies the scoring logic

### Exercise 2: Dynamic Multi-Region ETL

Build a TaskFlow DAG that dynamically processes data from multiple regions:

1. A `get_regions()` task returns a list of region configs: `[{"name": "us-east", "bucket": "s3://us-east/data"}, ...]`
2. Use `expand()` to fan out a `process_region(config: dict)` task that simulates extracting row counts per region
3. Use `partial()` to fix a shared `output_format="parquet"` parameter across all mapped instances
4. An `aggregate_results(results: list[dict])` task collects all outputs and computes the total row count
5. A `generate_report(summary: dict)` task prints a formatted summary table

Verify that adding a new region to `get_regions()` automatically creates a new task instance without any DAG code changes.

### Exercise 3: Hybrid DAG with SLA and Alerting

Design a hybrid DAG that combines TaskFlow and traditional operators with production-grade reliability:

1. A `FileSensor` waits for a daily input file (`/data/input/{{ ds }}.json`)
2. A `@task` reads the file and returns parsed records (use `ds` context variable)
3. A `@task.virtualenv` with `pandas==2.1.0` performs statistical analysis on the records
4. A `PostgresOperator` writes a summary to a staging table
5. A `@task` reads the staging table and sends an alert if any metric exceeds a threshold
6. Configure SLA of 2 hours on the entire DAG and implement an `on_failure_callback` that logs the failed task ID and execution date
7. Write a `DagBag` test that verifies all 5+ task IDs exist and their dependency order is correct

### Exercise 4: Custom XCom Backend for Large Payloads

The default XCom backend stores data in Airflow's metadata database, which becomes a bottleneck for large DataFrames. Solve this problem:

1. Explain in comments why passing a large DataFrame directly via XCom would cause performance issues
2. Redesign a DAG where an `extract()` task produces a large dataset by having it write to a local Parquet file and return only the **file path** as XCom
3. A `transform(path: str)` task reads the file, performs aggregation, and writes the result to a new path
4. A `load(path: str)` task reads the final file and "loads" it (print the row count)
5. Add a `cleanup(paths: list[str])` task using `multiple_outputs=False` that deletes all intermediate files using `expand()`

### Exercise 5: Branch and Parallel Merge Pattern

Implement a conditional processing DAG that models a real-world A/B decision:

1. A `@task.branch` task inspects an Airflow Variable named `processing_mode` and returns either `"fast_path"` or `"full_path"`
2. The `fast_path` group uses a single `@task` that approximates results in under 5 seconds
3. The `full_path` group uses three parallel `@task` instances (one per data segment), then an aggregation task
4. Both branches converge at a `report(result: dict)` task that uses `trigger_rule="none_failed_min_one_success"`
5. Write tests for both the branch decision logic and the fast-path aggregation using `.function()`

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
