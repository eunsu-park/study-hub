# Prefect Modern Orchestration

## Learning Objectives

After completing this lesson, you will be able to:

1. Compare Prefect and Airflow across key dimensions such as workflow definition, scheduling model, and dynamic workflow support
2. Define Prefect flows and tasks using Python decorators and configure their dependencies
3. Implement caching, retries, and concurrency controls in Prefect flows
4. Deploy Prefect flows using Work Pools and configure schedules via the Prefect Cloud UI or API
5. Integrate Prefect with external systems using blocks and build parameterized, reusable flows
6. Monitor flow run states and set up notifications and automations for production pipelines

---

## Overview

Prefect is a modern workflow orchestration tool that builds data pipelines in a Python-native way. Compared to Airflow, it offers simpler setup and supports dynamic workflows.

---

## 1. Prefect Overview

### 1.1 Prefect vs Airflow

```
┌────────────────────────────────────────────────────────────────┐
│                   Prefect vs Airflow Comparison                 │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Airflow:                    Prefect:                          │
│  ┌──────────────┐           ┌──────────────┐                  │
│  │ DAG (Static) │           │ Flow (Dynamic)│                  │
│  │              │           │               │                  │
│  │ - Static def │           │ - Dynamic     │                  │
│  │ - File-based │           │ - Python code │                  │
│  │ - Scheduler  │           │ - Event-driven│                  │
│  └──────────────┘           └──────────────┘                  │
│                                                                │
│  Execution model:           Execution model:                   │
│  Scheduler → Worker         Trigger → Work Pool → Worker       │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

| Feature | Airflow | Prefect |
|---------|---------|---------|
| **Definition** | DAG files | Python decorators |
| **Scheduling** | Scheduler process | Event-based, serverless |
| **Dynamic workflows** | Limited | Native support |
| **Local execution** | Complex setup | Instant |
| **State management** | DB required | Optional |
| **Learning curve** | Steep | Gentle |

### 1.2 Prefect Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    Prefect Architecture                        │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   ┌─────────────────────────────────────────────┐             │
│   │              Prefect Cloud / Server         │             │
│   │  ┌─────────┐ ┌─────────┐ ┌─────────┐       │             │
│   │  │  UI     │ │  API    │ │ Automations    │             │
│   │  └─────────┘ └─────────┘ └─────────┘       │             │
│   └─────────────────────────────────────────────┘             │
│                          ↑ ↓                                   │
│   ┌─────────────────────────────────────────────┐             │
│   │               Work Pools                     │             │
│   │  ┌─────────┐ ┌─────────┐ ┌─────────┐       │             │
│   │  │ Process │ │ Docker  │ │  K8s    │       │             │
│   │  └─────────┘ └─────────┘ └─────────┘       │             │
│   └─────────────────────────────────────────────┘             │
│                          ↑ ↓                                   │
│   ┌─────────────────────────────────────────────┐             │
│   │               Workers                        │             │
│   │         (Flow execution agents)              │             │
│   └─────────────────────────────────────────────┘             │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 2. Installation and Getting Started

### 2.1 Installation

```bash
# Basic installation
pip install prefect

# Install additional integrations
pip install "prefect[aws]"      # AWS integration
pip install "prefect[gcp]"      # GCP integration
pip install "prefect[dask]"     # Dask integration

# Check version
prefect version
```

### 2.2 Connect to Prefect Cloud (Optional)

```bash
# Login to Prefect Cloud
prefect cloud login

# Or use API key
prefect cloud login --key YOUR_API_KEY

# Connect to self-hosted server
prefect config set PREFECT_API_URL="http://localhost:4200/api"
```

### 2.3 Run Local Server

```bash
# Start Prefect server (includes UI)
prefect server start

# Access UI: http://localhost:4200
```

---

## 3. Flow and Task Basics

### 3.1 Basic Flow

```python
from prefect import flow, task
from prefect.logging import get_run_logger

# @task decorator turns a function into a trackable unit — Prefect records its
# state (Pending → Running → Completed/Failed), logs, and return values automatically.
# Unlike Airflow operators, tasks are plain Python functions with no boilerplate.
@task
def extract_data(source: str) -> dict:
    """Data extraction task"""
    # get_run_logger() gives a logger scoped to the current task run, so log messages
    # automatically include task name/run ID — essential for debugging in parallel flows
    logger = get_run_logger()
    logger.info(f"Extracting from {source}")

    # In practice, extract from DB, API, etc.
    data = {"source": source, "records": [1, 2, 3, 4, 5]}
    return data


@task
def transform_data(data: dict) -> dict:
    """Data transformation task"""
    logger = get_run_logger()
    logger.info(f"Transforming {len(data['records'])} records")

    # Transformation logic
    data["records"] = [x * 2 for x in data["records"]]
    data["transformed"] = True
    return data


@task
def load_data(data: dict, destination: str) -> bool:
    """Data loading task"""
    logger = get_run_logger()
    logger.info(f"Loading to {destination}")

    # In practice, save to DB, file, etc.
    print(f"Loaded data: {data}")
    return True


# @flow wraps the orchestration logic. Prefect infers task dependencies from the
# data flow between function calls — no need for explicit >> operators like Airflow.
# Default parameters make the flow callable both programmatically and via deployment.
@flow(name="ETL Pipeline")
def etl_pipeline(source: str = "database", destination: str = "warehouse"):
    """ETL pipeline flow"""
    # Prefect automatically tracks data dependencies: transform waits for extract,
    # load waits for transform. No explicit dependency wiring required.
    raw_data = extract_data(source)
    transformed = transform_data(raw_data)
    result = load_data(transformed, destination)
    return result


# Flows are regular Python functions — run directly for local testing without
# needing a scheduler, database, or web server (unlike Airflow's full stack)
if __name__ == "__main__":
    etl_pipeline()
```

### 3.2 Task Options

```python
from prefect import task
from datetime import timedelta

@task(
    name="My Task",
    description="Task description",
    tags=["etl", "production"],         # Tags enable filtering in Prefect UI and can trigger automations
    retries=3,                          # Automatic retries handle transient failures (network timeouts, API rate limits)
    retry_delay_seconds=60,             # Fixed 60s delay gives downstream services time to recover
    timeout_seconds=3600,               # 1-hour timeout prevents zombie tasks from holding resources indefinitely
    cache_key_fn=lambda: "static_key",  # Static key means same result for any input — use for idempotent lookups
    cache_expiration=timedelta(hours=1), # 1-hour expiry balances freshness vs avoiding redundant API calls
    log_prints=True,                    # Redirects print() to Prefect logs — useful for legacy code that uses print
)
def my_task(param: str) -> str:
    print(f"Processing: {param}")
    return f"Result: {param}"


# Exponential backoff avoids thundering-herd retries against recovering services —
# each retry waits exponentially longer (10s, 20s, 40s, 80s, 160s) to reduce load
from prefect.tasks import exponential_backoff

@task(
    retries=5,
    retry_delay_seconds=exponential_backoff(backoff_factor=10),
)
def flaky_task():
    """Unstable external API call"""
    import random
    if random.random() < 0.7:
        raise Exception("Random failure")
    return "Success"
```

### 3.3 Flow Options

```python
from prefect import flow
from prefect.task_runners import ConcurrentTaskRunner, SequentialTaskRunner

@flow(
    name="My Flow",
    description="Flow description",
    version="1.0.0",
    retries=2,                           # Flow-level retries re-run the entire flow, not individual tasks
    retry_delay_seconds=300,             # 5-minute delay gives dependent systems time to stabilize
    timeout_seconds=7200,                # 2-hour hard limit prevents runaway pipelines from blocking work pools
    task_runner=ConcurrentTaskRunner(),  # Runs tasks in threads — use for I/O-bound tasks (API calls, DB queries)
    log_prints=True,
    persist_result=True,                 # Persists flow return value to storage — required for flow-of-flows patterns
)
def my_flow():
    pass


# SequentialTaskRunner forces one-at-a-time execution — use when tasks have
# hidden shared state (e.g., writing to same file) that concurrency would corrupt
@flow(task_runner=SequentialTaskRunner())
def sequential_flow():
    pass
```

---

## 4. Dynamic Workflows

### 4.1 Dynamic Task Creation

```python
from prefect import flow, task

@task
def process_item(item: str) -> str:
    return f"Processed: {item}"


# Unlike Airflow DAGs where task count must be defined at parse time, Prefect
# creates tasks at runtime — the loop body is real Python, not a static graph.
@flow
def dynamic_tasks_flow(items: list[str]):
    """Dynamically determine number of tasks"""
    results = []
    for item in items:
        # Each call creates a separate task run — Prefect tracks them independently
        result = process_item(item)
        results.append(result)
    return results


# Execute
dynamic_tasks_flow(["a", "b", "c", "d"])


# .submit() enables concurrent execution via the flow's task_runner.
# Use this for I/O-bound workloads (API calls, file downloads) where
# waiting sequentially wastes time.
@flow
def parallel_tasks_flow(items: list[str]):
    """Execute tasks in parallel"""
    futures = []
    for item in items:
        # .submit() returns a PrefectFuture immediately without blocking,
        # unlike direct calls which wait for the result
        future = process_item.submit(item)
        futures.append(future)

    # .result() blocks until the future resolves — gather all results after
    # submitting everything to maximize parallelism
    results = [f.result() for f in futures]
    return results
```

### 4.2 Conditional Execution

```python
from prefect import flow, task

@task
def check_condition(data: dict) -> bool:
    return data.get("count", 0) > 100


@task
def process_large(data: dict):
    print(f"Processing large dataset: {data['count']} records")


@task
def process_small(data: dict):
    print(f"Processing small dataset: {data['count']} records")


@flow
def conditional_flow(data: dict):
    """Conditional branching"""
    is_large = check_condition(data)

    if is_large:
        process_large(data)
    else:
        process_small(data)


# Execute
conditional_flow({"count": 150})  # Runs process_large
conditional_flow({"count": 50})   # Runs process_small
```

### 4.3 Subflows

```python
from prefect import flow, task

@task
def extract(source: str) -> list:
    return [1, 2, 3, 4, 5]


@task
def transform(data: list) -> list:
    return [x * 2 for x in data]


@task
def load(data: list, target: str):
    print(f"Loading {len(data)} records to {target}")


# Subflows are flows called from other flows. Each subflow gets its own flow run
# with independent state tracking, retries, and logs — making them composable and reusable.
@flow(name="ETL Subflow")
def etl_subflow(source: str, target: str):
    """Reusable ETL subflow"""
    data = extract(source)
    transformed = transform(data)
    load(transformed, target)
    return len(transformed)


# Parent flow orchestrates subflows like function calls. If a subflow fails,
# only that subflow retries — other completed subflows are not re-executed.
@flow(name="Main Pipeline")
def main_pipeline():
    """Orchestrate multiple subflows"""
    # Each call creates a nested flow run visible in the UI hierarchy.
    # For parallel subflow execution, use .submit() with ConcurrentTaskRunner.
    count_a = etl_subflow("source_a", "target_a")
    count_b = etl_subflow("source_b", "target_b")
    count_c = etl_subflow("source_c", "target_c")

    print(f"Total processed: {count_a + count_b + count_c}")


main_pipeline()
```

---

## 5. Deployment

### 5.1 Creating Deployment

```python
from prefect import flow
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule

@flow
def my_etl_flow(date: str = None):
    """Daily ETL flow"""
    from datetime import datetime
    date = date or datetime.now().strftime("%Y-%m-%d")
    print(f"Running ETL for {date}")


# Deployments decouple flow definition from execution scheduling — the same flow
# can have multiple deployments with different schedules, parameters, and infrastructure.
deployment = Deployment.build_from_flow(
    flow=my_etl_flow,
    name="daily-etl",
    version="1.0",
    tags=["production", "etl"],           # Tags enable filtering and RBAC in Prefect Cloud
    schedule=CronSchedule(cron="0 6 * * *"),  # 6 AM ensures data is ready before business hours
    parameters={"date": None},            # None triggers datetime.now() fallback for daily runs
    work_pool_name="default-agent-pool",  # Routes execution to the correct infrastructure
)

# apply() registers the deployment with the Prefect server/cloud — workers
# poll for scheduled runs and execute them in the specified work pool
deployment.apply()
```

### 5.2 Create Deployment via CLI

```bash
# Generate prefect.yaml
prefect init

# Build and apply deployment
prefect deploy --name daily-etl
```

```yaml
# prefect.yaml example
name: my-project
prefect-version: 2.14.0

deployments:
  - name: daily-etl
    entrypoint: flows/etl.py:my_etl_flow
    work_pool:
      name: default-agent-pool
    schedule:
      cron: "0 6 * * *"
    parameters:
      date: null
    tags:
      - production
      - etl
```

### 5.3 Work Pools and Workers

```bash
# Create work pool
prefect work-pool create my-pool --type process

# Start worker
prefect worker start --pool my-pool

# Docker-based work pool
prefect work-pool create docker-pool --type docker

# Kubernetes-based work pool
prefect work-pool create k8s-pool --type kubernetes
```

---

## 6. Comparison Examples with Airflow

### 6.1 Airflow Version

```python
# Airflow DAG
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

# Airflow tasks communicate via XCom (cross-communication) — an implicit key-value
# store backed by the metadata DB. This indirection makes data flow harder to trace
# and limits payload size (default 48KB in the DB, though configurable).
def extract(**kwargs):
    ti = kwargs['ti']
    data = [1, 2, 3, 4, 5]
    ti.xcom_push(key='data', value=data)

def transform(**kwargs):
    ti = kwargs['ti']
    # Must know the exact task_id string — a source of runtime errors if renamed
    data = ti.xcom_pull(key='data', task_ids='extract')
    result = [x * 2 for x in data]
    ti.xcom_push(key='result', value=result)

def load(**kwargs):
    ti = kwargs['ti']
    result = ti.xcom_pull(key='result', task_ids='transform')
    print(f"Loading: {result}")

with DAG(
    'etl_airflow',
    start_date=datetime(2024, 1, 1),
    schedule_interval='@daily',
    catchup=False,  # Without this, Airflow backfills ALL missed runs since start_date
) as dag:
    t1 = PythonOperator(task_id='extract', python_callable=extract)
    t2 = PythonOperator(task_id='transform', python_callable=transform)
    t3 = PythonOperator(task_id='load', python_callable=load)

    # Dependencies must be declared explicitly — Airflow cannot infer them from code
    t1 >> t2 >> t3
```

### 6.2 Prefect Version

```python
# Prefect Flow — same ETL logic with no boilerplate. Data passes as function
# return values (not XCom), making the code testable as plain Python.
from prefect import flow, task

@task
def extract() -> list:
    return [1, 2, 3, 4, 5]

# Type hints serve double duty: Python type checking + Prefect schema validation
@task
def transform(data: list) -> list:
    return [x * 2 for x in data]

@task
def load(data: list):
    print(f"Loading: {data}")

@flow
def etl_prefect():
    # Dependencies are implicit from data flow — Prefect builds the DAG automatically.
    # This eliminates the class of bugs where explicit >> ordering disagrees with
    # actual data dependencies.
    data = extract()
    transformed = transform(data)
    load(transformed)

# No scheduler/webserver/DB needed — run and debug locally like any Python script
etl_prefect()
```

### 6.3 Key Differences

```python
"""
1. Data passing:
   - Airflow: Use XCom (explicit push/pull)
   - Prefect: Direct use of function return values (natural Python)

2. Dependencies:
   - Airflow: Explicit with >> operator
   - Prefect: Automatically inferred from function call order

3. Scheduling:
   - Airflow: Scheduler process required
   - Prefect: Optional, event-based possible

4. Local testing:
   - Airflow: Complex setup required
   - Prefect: Execute like regular Python functions

5. Dynamic workflows:
   - Airflow: Limited support
   - Prefect: Native Python control flow
"""
```

---

## 7. Advanced Features

### 7.1 State Handlers

```python
from prefect import flow, task
from prefect.states import State, Completed, Failed

# State handlers fire on state transitions — use them for alerting, cleanup, or
# custom logging without cluttering the main task logic.
def custom_state_handler(task, task_run, state: State):
    """Called on task state change"""
    if state.is_failed():
        # In production, integrate with Slack/PagerDuty here. Returning state
        # (not raising) lets Prefect continue its normal retry/failure handling.
        print(f"Task {task.name} failed!")
    return state


# on_failure hooks only fire on failure — more efficient than checking every
# state transition when you only need error notifications
@task(on_failure=[custom_state_handler])
def risky_task():
    raise ValueError("Something went wrong")


# Flow-level handlers catch failures from any task in the flow — useful for
# sending a single "pipeline failed" alert instead of per-task notifications
@flow(on_failure=[lambda flow, flow_run, state: print("Flow failed!")])
def my_flow():
    risky_task()
```

### 7.2 Result Storage

```python
from prefect import flow, task
from prefect.filesystems import S3, LocalFileSystem
from prefect.serializers import JSONSerializer

# Result storage persists task outputs beyond the flow run lifetime — enables
# caching across runs, debugging past results, and flow-of-flows data passing.
# LocalFileSystem is convenient for development; use S3/GCS in production.
@task(result_storage=LocalFileSystem(basepath="/tmp/prefect"))
def save_locally():
    return {"data": [1, 2, 3]}


# S3 storage with JSON serialization — JSONSerializer produces human-readable
# results (useful for debugging), unlike the default pickle which is faster but opaque.
# persist_result=True is required to actually write to storage; without it, results
# stay in memory and are lost when the process exits.
@task(
    persist_result=True,
    result_storage=S3(bucket_path="my-bucket/results"),
    result_serializer=JSONSerializer(),
)
def save_to_s3():
    return {"large": "data"}
```

### 7.3 Secret Management

```python
from prefect.blocks.system import Secret

# Save secret as Block (via UI or CLI)
# prefect block register -m prefect.blocks.system

# Use in code
@task
def use_secret():
    api_key = Secret.load("my-api-key").get()
    # Use for API calls
    return f"Using key: {api_key[:4]}..."


# Use environment variables
import os

@task
def use_env_var():
    return os.getenv("MY_SECRET")
```

---

## Practice Problems

### Problem 1: Basic Flow
Write an ETL flow with 3 tasks (data extraction, transformation, loading).

### Problem 2: Dynamic Tasks
Write a flow that accepts a list of files and processes each file in parallel.

### Problem 3: Conditional Execution
Write a flow that selects different processing methods based on data size.

---

## Summary

| Concept | Description |
|---------|-------------|
| **Flow** | Workflow definition (Airflow's DAG) |
| **Task** | Individual unit of work |
| **Deployment** | Deployment configuration for a flow |
| **Work Pool** | Worker group management |
| **Worker** | Flow execution agent |

---

## References

- [Prefect Documentation](https://docs.prefect.io/)
- [Prefect Tutorials](https://docs.prefect.io/tutorials/)
- [Prefect GitHub](https://github.com/PrefectHQ/prefect)
