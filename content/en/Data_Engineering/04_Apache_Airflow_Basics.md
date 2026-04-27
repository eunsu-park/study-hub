[← Previous: 3. ETL vs ELT](03_ETL_vs_ELT.md) | [Next: 5. Airflow Advanced →](05_Airflow_Advanced.md)

# Apache Airflow Basics

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the Apache Airflow architecture and describe the roles of each core component (Web Server, Scheduler, Executor, Worker, Metadata DB)
2. Define a DAG (Directed Acyclic Graph) in Python and configure task dependencies using Airflow operators
3. Implement common Airflow operators including PythonOperator, BashOperator, and PostgresOperator
4. Use XComs and Airflow Variables to share data and configuration between tasks
5. Apply scheduling with cron expressions and configure backfilling and catchup behavior
6. Debug and monitor DAG runs using the Airflow Web UI and logs

---

## Overview

Apache Airflow is a platform for programmatically authoring, scheduling, and monitoring workflows. It manages complex data pipelines by defining DAGs (Directed Acyclic Graphs) in Python.

---

## 1. Airflow Architecture

### Theory: Executor Models

The executor decides where tasks actually run. Airflow ships with four production-grade options:

#### C.1 SequentialExecutor

Runs one task at a time in the same process as the scheduler. Useful only for testing — no parallelism.

#### C.2 LocalExecutor

Forks subprocesses on the same machine as the scheduler. Parallelism = number of cores. Good for small deployments; no distribution across machines.

#### C.3 CeleryExecutor

The traditional production setup. A Celery broker (Redis or RabbitMQ) queues task instances; a fleet of Celery workers pulls from the queue and executes. Workers are long-lived processes; you scale by adding workers.

- **Pros:** mature, well-understood, low per-task overhead.
- **Cons:** workers are stateful (need Python deps pre-installed); resource isolation between tasks is limited (one bad task can OOM the whole worker); scaling requires capacity planning.

#### C.4 KubernetesExecutor

Each task instance becomes a Kubernetes pod, scheduled by Kubernetes, killed when the task ends.

- **Pros:** perfect resource isolation (each task gets its own container, memory limit, CPU limit); per-task Python deps via per-task images; auto-scaling via Kubernetes cluster autoscaler.
- **Cons:** per-task pod startup latency (10-30 seconds) — bad for many small tasks; Kubernetes operational complexity.

Hybrid: CeleryKubernetesExecutor lets you mix — small fast tasks via Celery, heavy isolated tasks via Kubernetes.

#### C.5 Picking an executor

| Workload | Executor |
|----------|----------|
| Local development | SequentialExecutor or LocalExecutor |
| Small production (1 machine, few tasks) | LocalExecutor |
| Medium-large production, mostly fast tasks | CeleryExecutor |
| Heterogeneous tasks needing isolation | KubernetesExecutor |
| Mix of fast and isolated | CeleryKubernetesExecutor |

Before diving into components, it helps to understand the problem Airflow solves. A plain cron job can schedule a single script, but it has no built-in support for complex task dependencies, automatic retries on failure, backfilling historical date ranges, or a centralized UI for observability. Airflow addresses all of these: it models pipelines as DAGs with explicit dependencies, provides configurable retry/alerting policies, supports backfill with a single CLI command, and ships with a web UI that shows task status, logs, and execution history in one place.

### 1.1 Core Components

```
┌──────────────────────────────────────────────────────────────┐
│                    Airflow Architecture                      │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────────┐         ┌─────────────┐                   │
│   │  Web Server │         │  Scheduler  │                   │
│   │    (UI)     │         │             │                   │
│   └──────┬──────┘         └──────┬──────┘                   │
│          │                       │                          │
│          │    ┌─────────────┐    │                          │
│          └───→│  Metadata   │←───┘                          │
│               │  Database   │                               │
│               │ (PostgreSQL)│                               │
│               └──────┬──────┘                               │
│                      │                                      │
│          ┌───────────┴───────────┐                          │
│          ↓                       ↓                          │
│   ┌─────────────┐         ┌─────────────┐                   │
│   │   Worker    │         │   Worker    │                   │
│   │  (Celery)   │         │  (Celery)   │                   │
│   └─────────────┘         └─────────────┘                   │
│                                                              │
│   DAGs Folder: /opt/airflow/dags/                           │
└──────────────────────────────────────────────────────────────┘
```

### 1.2 Component Roles

| Component | Role |
|-----------|------|
| **Web Server** | Provide UI, visualize DAGs, view logs |
| **Scheduler** | Parse DAGs, schedule tasks, trigger execution |
| **Executor** | Determine task execution method (Local, Celery, K8s) |
| **Worker** | Execute actual tasks (Celery/K8s Executor) |
| **Metadata DB** | Store DAG metadata and execution history |

### 1.3 Executor Types

```python
# airflow.cfg settings
# The executor determines how many tasks can run in parallel and whether
# they run on the same machine or across a cluster.  Choosing the wrong
# executor is the #1 cause of "my DAGs are slow" complaints.
executor_types = {
    "SequentialExecutor": "Single process, for development",
    "LocalExecutor": "Multi-process, single machine",
    "CeleryExecutor": "Distributed processing, production",
    "KubernetesExecutor": "Run as K8s Pods"
}

# Recommended configuration:
# Development → LocalExecutor (no external broker needed, still parallel)
# Production  → CeleryExecutor (persistent workers, lower cold-start)
#            or KubernetesExecutor (per-task isolation, auto-scaling to zero)
```

---

## 2. Installation and Environment Setup

### 2.1 Docker Compose Installation (Recommended)

```yaml
# docker-compose.yaml
version: '3.8'

# YAML anchor (&airflow-common) avoids duplicating config across services —
# all Airflow components share the same image, env vars, and volume mounts.
x-airflow-common: &airflow-common
  image: apache/airflow:2.7.0
  environment:
    &airflow-common-env
    AIRFLOW__CORE__EXECUTOR: CeleryExecutor
    AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
    AIRFLOW__CELERY__RESULT_BACKEND: db+postgresql://airflow:airflow@postgres/airflow
    # Redis as Celery broker: lightweight, no authentication for local dev.
    # In production, use a managed Redis or RabbitMQ with TLS.
    AIRFLOW__CELERY__BROKER_URL: redis://:@redis:6379/0
    AIRFLOW__CORE__FERNET_KEY: ''
    # Pause new DAGs by default so they don't start running immediately on
    # deploy — gives operators time to review before enabling.
    AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION: 'true'
    AIRFLOW__CORE__LOAD_EXAMPLES: 'false'
  volumes:
    # Mount local directories so DAG code changes are picked up without
    # rebuilding the Docker image — essential for a fast dev loop.
    - ./dags:/opt/airflow/dags
    - ./logs:/opt/airflow/logs
    - ./plugins:/opt/airflow/plugins

services:
  postgres:
    image: postgres:13
    environment:
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow
      POSTGRES_DB: airflow
    volumes:
      - postgres-db-volume:/var/lib/postgresql/data

  redis:
    image: redis:latest

  airflow-webserver:
    <<: *airflow-common
    command: webserver
    ports:
      - 8080:8080
    depends_on:
      - postgres
      - redis

  airflow-scheduler:
    <<: *airflow-common
    command: scheduler
    depends_on:
      - postgres
      - redis

  airflow-worker:
    <<: *airflow-common
    command: celery worker
    depends_on:
      - airflow-scheduler

  airflow-init:
    <<: *airflow-common
    entrypoint: /bin/bash
    command:
      - -c
      - |
        airflow db init
        airflow users create \
          --username admin \
          --password admin \
          --firstname Admin \
          --lastname User \
          --role Admin \
          --email admin@example.com

volumes:
  postgres-db-volume:
```

### 2.2 pip Installation (Local Development)

```bash
# Create virtual environment
python -m venv airflow-venv
source airflow-venv/bin/activate

# Install Airflow
pip install "apache-airflow[celery,postgres,redis]==2.7.0" \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.7.0/constraints-3.9.txt"

# Initialize
export AIRFLOW_HOME=~/airflow
airflow db init

# Create user
airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com

# Start services
airflow webserver --port 8080 &
airflow scheduler &
```

---

## 3. DAG (Directed Acyclic Graph)

### Theory: DAG Semantics

A DAG is a directed acyclic graph: nodes are tasks, edges are dependencies. Airflow guarantees that a task does not start until all upstream tasks have finished.

#### A.1 Why acyclic

Cycles are forbidden because a cycle has no valid execution order — task A waits for B which waits for A. Airflow validates DAG definitions at parse time and rejects cycles. The acyclic constraint is what makes scheduling decidable; without it, "is this DAG complete?" has no answer.

If you need a loop, you express it as repeated DAG runs (one per scheduled interval) or, in newer Airflow, as a *dynamic task mapping* — a task expanded into N parallel instances at runtime.

#### A.2 Definition time vs execution time

The DAG file is **Python code that gets parsed every few seconds** by the scheduler. That is not just a quirk — it shapes everything:

- Top-level code (imports, variables, DAG construction) runs **on every parse**, in the scheduler. Slow imports = slow scheduler.
- Task code (functions decorated with `@task` or wrapped in operators) runs **once per task instance**, on a worker.
- *Never* put database queries or API calls in top-level DAG code. They get hammered every parse cycle.

The mental model: the DAG file is a *definition* of the task graph; tasks are the actual work. Keep them separate.

#### A.3 Templating and Jinja

Airflow injects context (logical_date, run_id, task_instance) into operator parameters via Jinja templates: `bash_command="extract.sh {{ ds }}"`. The template is rendered at task execution time, not DAG parse time. This is how a single DAG handles 365 daily runs — each run substitutes its own date.

### 3.1 Basic DAG Structure

```python
# dags/simple_dag.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

# default_args are inherited by every task in the DAG, reducing boilerplate.
# Override per-task when a specific operator needs different retry behavior.
default_args = {
    'owner': 'data_team',
    # depends_on_past=False: each run is independent. Set to True only when
    # a task genuinely needs the previous day's run to have succeeded first
    # (e.g., incremental aggregation that reads yesterday's output).
    'depends_on_past': False,
    'email': ['data-team@company.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    # 3 retries with 5-min delay: gives transient issues (network blips,
    # temporary DB locks) time to resolve without human intervention.
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

# DAG definition
with DAG(
    dag_id='simple_example_dag',
    default_args=default_args,
    description='Simple example DAG',
    schedule_interval='0 9 * * *',  # Daily at 9 AM
    start_date=datetime(2024, 1, 1),
    # Prevent backfill flooding: without catchup=False, Airflow schedules
    # ALL missed runs since start_date on first deployment — if start_date
    # is 2024-01-01 and today is 2024-06-15, that's ~165 concurrent runs.
    catchup=False,
    tags=['example', 'tutorial'],
) as dag:

    # Task 1: Execute Python function
    def print_hello():
        print("Hello, Airflow!")
        return "Hello returned"

    task_hello = PythonOperator(
        task_id='print_hello',
        python_callable=print_hello,
    )

    # Task 2: Execute Bash command
    task_date = BashOperator(
        task_id='print_date',
        bash_command='date',
    )

    # Task 3: Python function (with arguments)
    def greet(name, **kwargs):
        execution_date = kwargs['ds']
        print(f"Hello, {name}! Today is {execution_date}")

    task_greet = PythonOperator(
        task_id='greet_user',
        python_callable=greet,
        op_kwargs={'name': 'Data Engineer'},
    )

    # Define task dependencies
    task_hello >> task_date >> task_greet
    # Or: task_hello.set_downstream(task_date)
```

### 3.2 DAG Parameters

```python
from airflow import DAG

dag = DAG(
    # Required parameters
    dag_id='my_dag',                    # Unique identifier (must be unique across all DAGs)
    start_date=datetime(2024, 1, 1),    # Earliest data_interval the scheduler will create

    # Schedule related
    schedule_interval='@daily',         # Execution frequency
    # schedule_interval='0 0 * * *'     # Cron expression (more precise control)
    # schedule_interval=timedelta(days=1)  # timedelta for non-calendar intervals

    # Execution control
    catchup=False,                      # See note above about backfill flooding
    # max_active_runs=1: prevents overlapping runs for non-idempotent pipelines.
    # Increase for idempotent DAGs that can safely run in parallel.
    max_active_runs=1,
    # max_active_tasks limits parallelism *within* a single run — useful to
    # avoid overwhelming a shared resource (e.g., a database connection pool).
    max_active_tasks=10,

    # Other
    default_args=default_args,          # Default arguments
    description='DAG description',
    tags=['production', 'etl'],         # Tags enable filtering in the web UI
    doc_md="""
    ## DAG Documentation
    This DAG performs daily ETL.
    """
)

# Schedule presets
schedule_presets = {
    '@once': 'Run once',
    '@hourly': 'Every hour (0 * * * *)',
    '@daily': 'Daily at midnight (0 0 * * *)',
    '@weekly': 'Every Sunday (0 0 * * 0)',
    '@monthly': 'First of month (0 0 1 * *)',
    '@yearly': 'January 1st (0 0 1 1 *)',
    None: 'Manual trigger only'
}
```

---

## 4. Operator Types

### Theory: Task Lifecycle and Idempotency Contract

A task instance moves through states. The interesting transitions:

```
none → scheduled → queued → running → success
                                    ↘ failed → up_for_retry → queued → running → success/failed
                                    ↘ skipped (upstream failed/skipped, branch decided)
```

Each state transition is recorded in the metadata DB; the UI shows the current state of every task in every run.

#### D.1 Retry semantics

Tasks declare `retries=N` and `retry_delay`. If a task fails, Airflow schedules a retry after `retry_delay`. After N retries, the task is marked `failed` and downstream tasks are `upstream_failed`.

This works only if **the task is idempotent**: running it twice produces the same final state. Concretely:

- **Bad:** `INSERT INTO sales SELECT * FROM raw WHERE date='{{ ds }}'`. Retry → duplicate rows.
- **Good:** `DELETE FROM sales WHERE date='{{ ds }}'; INSERT INTO sales SELECT ...` in one transaction. Retry → same final state.
- **Good:** writing to a date-partitioned location. Retry overwrites the partition.

The idempotency-on-`{{ ds }}` discipline is what makes Airflow's retry, backfill, and rerun work.

#### D.2 Pools and concurrency

Two knobs control concurrency:

- **DAG-level `max_active_runs`** — how many runs of this DAG can be in-flight at once. Default 16.
- **Pools** — named buckets with a slot limit. A task assigned to a pool consumes a slot while running. Use to throttle access to scarce resources (e.g., `database_pool` with 5 slots prevents DB overload).

A task that hits a pool limit is `queued` until a slot frees. This is how you prevent runaway parallelism from killing downstream systems.

### 4.1 Main Operators

```python
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.email import EmailOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.http.operators.http import SimpleHttpOperator

# 1. PythonOperator — best when you need complex logic, library imports, or
# DataFrame manipulation.  Use this over BashOperator when the task involves
# more than a one-liner shell command.
def my_function(arg1, arg2):
    return arg1 + arg2

python_task = PythonOperator(
    task_id='python_task',
    python_callable=my_function,
    op_args=[1, 2],              # Positional arguments
    op_kwargs={'arg1': 1},       # Keyword arguments
)

# 2. BashOperator — ideal for calling CLI tools (dbt run, spark-submit),
# running shell scripts, or quick file operations.  Prefer this over
# PythonOperator when the task is essentially a shell command.
bash_task = BashOperator(
    task_id='bash_task',
    bash_command='echo "Hello" && date',
    env={'MY_VAR': 'value'},     # Environment variables
    cwd='/tmp',                  # Working directory
)

# 3. EmptyOperator — zero-cost DAG structure nodes.  Use as start/end
# markers or to fan-in/fan-out parallel branches without running any logic.
start = EmptyOperator(task_id='start')
end = EmptyOperator(task_id='end')

# 4. PostgresOperator — executes SQL directly against a managed connection.
# Prefer this over PythonOperator + psycopg2 for simple SQL statements
# because it handles connection lifecycle and templating automatically.
sql_task = PostgresOperator(
    task_id='sql_task',
    postgres_conn_id='my_postgres',
    sql="""
        INSERT INTO logs (message, created_at)
        VALUES ('Task executed', NOW());
    """,
)

# 5. EmailOperator — sends notification emails via the configured SMTP
# connection.  Use for success summaries or reports; for failure alerts,
# prefer email_on_failure in default_args (fires automatically).
email_task = EmailOperator(
    task_id='send_email',
    to='user@example.com',
    subject='Airflow Notification',
    html_content='<h1>Task completed!</h1>',
)

# 6. SimpleHttpOperator — calls external REST APIs.  The response_check
# lambda lets you define custom success criteria beyond HTTP 2xx status.
http_task = SimpleHttpOperator(
    task_id='http_task',
    http_conn_id='my_api',
    endpoint='/api/data',
    method='GET',
    response_check=lambda response: response.status_code == 200,
)
```

### 4.2 Branch Operator

```python
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator

def choose_branch(**kwargs):
    """Choose task to execute based on condition"""
    execution_date = kwargs['ds']
    day_of_week = datetime.strptime(execution_date, '%Y-%m-%d').weekday()

    if day_of_week < 5:  # Weekday
        return 'weekday_task'
    else:  # Weekend
        return 'weekend_task'

with DAG('branch_example', ...) as dag:

    branch_task = BranchPythonOperator(
        task_id='branch',
        python_callable=choose_branch,
    )

    weekday_task = EmptyOperator(task_id='weekday_task')
    weekend_task = EmptyOperator(task_id='weekend_task')
    # trigger_rule='none_failed_min_one_success': the join task runs as long
    # as at least one branch succeeded and none failed.  Default 'all_success'
    # would never trigger because the un-chosen branch is always "skipped".
    join_task = EmptyOperator(task_id='join', trigger_rule='none_failed_min_one_success')

    branch_task >> [weekday_task, weekend_task] >> join_task
```

### 4.3 Custom Operator

```python
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from typing import Any

class MyCustomOperator(BaseOperator):
    """Custom operator example"""

    # template_fields: Airflow renders Jinja templates in these fields before
    # execute() runs, enabling dynamic values like {{ ds }} or {{ params.x }}.
    # Any field NOT listed here will be treated as a literal string.
    template_fields = ['param']

    @apply_defaults
    def __init__(
        self,
        param: str,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.param = param

    def execute(self, context: dict) -> Any:
        """Task execution logic"""
        self.log.info(f"Executing with param: {self.param}")

        # context dict provides runtime metadata (dates, task instance,
        # DAG run info) — avoids hardcoding values that change per run.
        execution_date = context['ds']
        task_instance = context['ti']

        # Business logic
        result = f"Processed {self.param} on {execution_date}"

        # Returning a value automatically pushes it to XCom with
        # key='return_value', making it available to downstream tasks.
        return result

# Usage
custom_task = MyCustomOperator(
    task_id='custom_task',
    param='my_value',
)
```

---

## 5. Task Dependencies

### 5.1 Dependency Definition Methods

```python
from airflow import DAG
from airflow.operators.empty import EmptyOperator

with DAG('dependency_example', ...) as dag:

    task_a = EmptyOperator(task_id='task_a')
    task_b = EmptyOperator(task_id='task_b')
    task_c = EmptyOperator(task_id='task_c')
    task_d = EmptyOperator(task_id='task_d')
    task_e = EmptyOperator(task_id='task_e')

    # Method 1: >> operator (recommended)
    task_a >> task_b >> task_c

    # Method 2: << operator (reverse)
    task_c << task_b << task_a  # Same as above

    # Method 3: set_downstream / set_upstream
    task_a.set_downstream(task_b)
    task_b.set_downstream(task_c)

    # Parallel execution
    task_a >> [task_b, task_c] >> task_d

    # Complex dependencies
    #     ┌→ B ─┐
    # A ──┤     ├──→ E
    #     └→ C → D ─┘

    task_a >> task_b >> task_e
    task_a >> task_c >> task_d >> task_e
```

### 5.2 Trigger Rules

```python
from airflow.utils.trigger_rule import TriggerRule

# Trigger rule types
trigger_rules = {
    'all_success': 'All upstream tasks succeeded (default)',
    'all_failed': 'All upstream tasks failed',
    'all_done': 'All upstream tasks completed (success/failure irrelevant)',
    'one_success': 'At least one succeeded',
    'one_failed': 'At least one failed',
    'none_failed': 'No failures (skips allowed)',
    'none_failed_min_one_success': 'No failures and at least one success',
    'none_skipped': 'No skips',
    'always': 'Always run',
}

# Usage example: join after a branch — runs even if one branch was skipped,
# as long as none actually *failed*.
task_join = EmptyOperator(
    task_id='join',
    trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
)

# Error handling task: only runs when at least one upstream failed —
# useful for cleanup or alert tasks that should not run on success.
task_error_handler = EmptyOperator(
    task_id='error_handler',
    trigger_rule=TriggerRule.ONE_FAILED,
)
```

---

## 6. Scheduling

### Theory: Logical Date / Data Interval

The single most confusing concept for new Airflow users. A scheduled DAG run has two distinct timestamps:

- **Logical date / data interval start** — the conceptual date the run *represents*. For a daily DAG scheduled at midnight UTC, the run for 2024-03-15 has `logical_date = 2024-03-15 00:00 UTC`. The run *processes data for* March 15.
- **Wall-clock time** — when the run actually executed. The 2024-03-15 run typically starts at 2024-03-16 00:00 UTC, because Airflow runs at the *end* of the data interval (after the data has fully arrived).

This shift trips up everyone. The rule: tasks should always reference the data they are processing (`{{ ds }}` = logical date as YYYY-MM-DD), not "today" or "now". Tasks must be deterministic functions of their data interval.

#### B.1 Why "end of interval" scheduling

If the DAG is "process yesterday's events", you cannot start at 2024-03-15 00:00 — yesterday's events are still arriving. You wait until the interval *ends* (2024-03-16 00:00) and then process the closed interval [2024-03-15 00:00, 2024-03-16 00:00).

#### B.2 Backfill and catchup

Two related concepts:

- **Catchup:** when a paused DAG is unpaused, run all the missed scheduled intervals. Default `catchup=True`. For DAGs that should only run going forward, set `catchup=False`.
- **Backfill:** explicitly trigger historical runs via CLI: `airflow dags backfill --start-date 2024-01-01 --end-date 2024-03-15`. Useful for replaying logic on old data.

Both depend on tasks being **idempotent functions of `logical_date`**. Running the 2024-01-15 task today must produce the same final state as running it on 2024-01-16. If the task says "load yesterday" instead of "load `{{ ds }}`", backfill is corrupted.

### 6.1 Cron Expressions

```python
# Cron format: minute hour day month day_of_week
cron_examples = {
    '0 0 * * *': 'Daily at midnight',
    '0 9 * * 1-5': 'Weekdays at 9 AM',
    '0 */2 * * *': 'Every 2 hours',
    '30 8 1 * *': 'First of month at 8:30 AM',
    '0 0 * * 0': 'Every Sunday at midnight',
}

# Use in DAG
dag = DAG(
    dag_id='scheduled_dag',
    schedule_interval='0 9 * * 1-5',  # Weekdays at 9 AM
    start_date=datetime(2024, 1, 1),
    ...
)
```

### 6.2 Data Interval

```python
# Airflow 2.0+ data interval concept
# Understanding this is critical: the DAG runs AFTER the interval ends,
# not at the start. This "end-of-period" convention ensures the full
# day's data exists before the pipeline processes it.
"""
schedule_interval = @daily, start_date = 2024-01-01

Execution time: 2024-01-02 00:00
data_interval_start: 2024-01-01 00:00
data_interval_end: 2024-01-02 00:00
logical_date (execution_date): 2024-01-01 00:00

→ Runs on 2024-01-02 to process 2024-01-01 data
"""

def process_daily_data(**kwargs):
    # Data period to process
    data_interval_start = kwargs['data_interval_start']
    data_interval_end = kwargs['data_interval_end']

    print(f"Processing data from {data_interval_start} to {data_interval_end}")

# Using Jinja templates
sql_task = PostgresOperator(
    task_id='load_data',
    sql="""
        SELECT * FROM sales
        WHERE sale_date >= '{{ data_interval_start }}'
          AND sale_date < '{{ data_interval_end }}'
    """,
)
```

---

## 7. Basic DAG Writing Example

### 7.1 Daily ETL DAG

```python
# dags/daily_etl_dag.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.operators.empty import EmptyOperator

default_args = {
    'owner': 'data_team',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': True,
    'email': ['data-alerts@company.com'],
}

def extract_data(**kwargs):
    """Extract data"""
    import pandas as pd

    # kwargs['ds'] is the logical date (YYYY-MM-DD) — Airflow injects this
    # automatically, so the same DAG code works for any date during backfills.
    ds = kwargs['ds']

    # Filter by ds to ensure idempotent extraction: re-running this task
    # for the same date always pulls the same data partition.
    query = f"""
        SELECT * FROM source_table
        WHERE date = '{ds}'
    """

    # Parquet preserves column types across the E→T boundary;
    # CSV would lose datetime/decimal precision.
    # df = pd.read_sql(query, source_conn)
    # df.to_parquet(f'/tmp/extract_{ds}.parquet')

    print(f"Extracted data for {ds}")
    return f"/tmp/extract_{ds}.parquet"

def transform_data(**kwargs):
    """Transform data"""
    import pandas as pd

    ti = kwargs['ti']
    extract_path = ti.xcom_pull(task_ids='extract')

    # df = pd.read_parquet(extract_path)
    # Transformation logic
    # df['new_column'] = df['column'].apply(transform_func)
    # df.to_parquet(f'/tmp/transform_{kwargs["ds"]}.parquet')

    print("Data transformed")
    return f"/tmp/transform_{kwargs['ds']}.parquet"

with DAG(
    dag_id='daily_etl_pipeline',
    default_args=default_args,
    description='Daily ETL pipeline',
    schedule_interval='0 6 * * *',  # Daily at 6 AM
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['etl', 'daily', 'production'],
) as dag:

    start = EmptyOperator(task_id='start')

    extract = PythonOperator(
        task_id='extract',
        python_callable=extract_data,
    )

    transform = PythonOperator(
        task_id='transform',
        python_callable=transform_data,
    )

    load = PostgresOperator(
        task_id='load',
        postgres_conn_id='warehouse',
        sql="""
            COPY target_table FROM '/tmp/transform_{{ ds }}.parquet'
            WITH (FORMAT 'parquet');
        """,
    )

    # Post-load validation: fail loudly if no rows were loaded.
    # The 1/0 trick causes a division-by-zero error that Airflow interprets
    # as a task failure, triggering the configured retry and alert policies.
    validate = PostgresOperator(
        task_id='validate',
        postgres_conn_id='warehouse',
        sql="""
            SELECT
                CASE WHEN COUNT(*) > 0 THEN 1
                     ELSE 1/0  -- Intentional error to fail the task
                END
            FROM target_table
            WHERE date = '{{ ds }}';
        """,
    )

    end = EmptyOperator(task_id='end')

    # Define dependencies
    start >> extract >> transform >> load >> validate >> end
```

---

## Practice Problems

### Problem 1: Basic DAG Creation
Create a DAG that runs hourly. It should include two tasks: one that logs the current time and another that creates a temporary file.

### Problem 2: Conditional Execution
Create a DAG using BranchPythonOperator that executes different tasks on weekdays versus weekends.

---

## Summary

| Concept | Description |
|------|------|
| **DAG** | Directed Acyclic Graph defining task dependencies |
| **Operator** | Task execution type (Python, Bash, SQL, etc.) |
| **Task** | Individual work unit within a DAG |
| **Scheduler** | DAG parsing and task scheduling |
| **Executor** | Task execution method (Local, Celery, K8s) |

---

## References

- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [Airflow Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
- [Astronomer Guides](https://www.astronomer.io/guides/)

---

[← Previous: 3. ETL vs ELT](03_ETL_vs_ELT.md) | [Next: 5. Airflow Advanced →](05_Airflow_Advanced.md)
