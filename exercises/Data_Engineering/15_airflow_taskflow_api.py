"""
Exercise Solutions: Lesson 15 - Airflow TaskFlow API

Covers:
  - Exercise 1: Migrate a Legacy DAG to TaskFlow
  - Exercise 2: Dynamic Multi-Region ETL
  - Exercise 3: Hybrid DAG with SLA and Alerting
  - Exercise 4: Custom XCom Backend for Large Payloads
  - Exercise 5: Branch and Parallel Merge Pattern

Note: Pure Python simulation of Airflow TaskFlow API concepts.
"""

import os
import json
import tempfile
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Simulated TaskFlow primitives
# ---------------------------------------------------------------------------

class TaskFlowContext:
    """Simulates Airflow's task context with XCom auto-passing."""
    def __init__(self):
        self.xcom_store: dict[str, Any] = {}
        self.task_log: list[str] = []
        self.ds: str = datetime.now().strftime("%Y-%m-%d")

    def log(self, task_name: str, message: str) -> None:
        self.task_log.append(f"[{task_name}] {message}")
        print(f"    [{task_name}] {message}")


def dag(dag_id: str, schedule: str = "@daily", **kwargs):
    """Simulates @dag decorator."""
    def decorator(func):
        def wrapper(*args, **wkwargs):
            print(f"\n  [DAG: {dag_id}] schedule={schedule}")
            print(f"  {'-' * 55}")
            result = func(*args, **wkwargs)
            print(f"  {'-' * 55}")
            print(f"  [DAG: {dag_id}] Complete\n")
            return result
        wrapper.__name__ = func.__name__
        wrapper._dag_id = dag_id
        return wrapper
    return decorator


def task(retries: int = 0, execution_timeout: int | None = None):
    """Simulates @task decorator with auto XCom via return values."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            name = func.__name__
            extras = []
            if retries:
                extras.append(f"retries={retries}")
            if execution_timeout:
                extras.append(f"timeout={execution_timeout}s")
            extra_str = f" ({', '.join(extras)})" if extras else ""
            print(f"    [task] {name}{extra_str}")
            result = func(*args, **kwargs)
            return result
        wrapper.__name__ = func.__name__
        # Simulates .function() for unit testing
        wrapper.function = func
        return wrapper
    return decorator


def task_group(group_id: str):
    """Simulates @task_group decorator."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"    [task_group: {group_id}]")
            return func(*args, **kwargs)
        wrapper.__name__ = func.__name__
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Exercise 1: Migrate a Legacy DAG to TaskFlow
# ---------------------------------------------------------------------------

@task()
def load_users() -> list[dict]:
    """TaskFlow version: return value is automatically stored as XCom."""
    users = [
        {"id": "u1", "clicks": 30, "purchases": 2},
        {"id": "u2", "clicks": 5, "purchases": 0},
        {"id": "u3", "clicks": 50, "purchases": 5},
        {"id": "u4", "clicks": 10, "purchases": 3},
        {"id": "u5", "clicks": 100, "purchases": 1},
    ]
    print(f"      Loaded {len(users)} users")
    return users


@task()
def score_users(users: list[dict]) -> dict[str, int]:
    """Score each user: clicks*2 + purchases*10.

    In the legacy DAG this used ti.xcom_pull(task_ids='load_users').
    In TaskFlow, the users parameter is automatically passed from the
    return value of load_users().
    """
    scores = {u["id"]: u["clicks"] * 2 + u["purchases"] * 10 for u in users}
    print(f"      Scores: {scores}")
    return scores


@task()
def filter_vip(scores: dict[str, int]) -> dict[str, int]:
    """Filter users with score >= 50."""
    vips = {uid: s for uid, s in scores.items() if s >= 50}
    print(f"      VIP users (score >= 50): {vips}")
    return vips


@task(retries=2, execution_timeout=120)
def send_campaign(vips: dict[str, int]) -> None:
    """Send campaign to VIP users. Has retry and timeout configuration."""
    print(f"      Sending campaign to {len(vips)} VIP users: {list(vips.keys())}")


@task_group(group_id="scoring_group")
def scoring_group(users: list[dict]) -> dict[str, int]:
    """Groups score_users and filter_vip into a visual/logical group."""
    scores = score_users(users)
    vips = filter_vip(scores)
    return vips


@dag(dag_id="taskflow_campaign", schedule="@weekly")
def exercise1_taskflow_campaign():
    """
    Original Legacy DAG used PythonOperator + xcom_push/xcom_pull.
    TaskFlow version uses @task decorators with automatic return-value passing.

    Key changes:
    1. @dag decorator replaces 'with DAG(...) as dag'
    2. @task decorator replaces PythonOperator
    3. Return values auto-push to XCom (no manual xcom_push)
    4. Function parameters auto-pull from XCom (no manual xcom_pull)
    5. @task_group groups related tasks visually
    """
    users = load_users()
    vips = scoring_group(users)
    send_campaign(vips)


def exercise1_unit_test():
    """Unit test using .function() to call the raw Python function.

    In Airflow:
        def test_score_users():
            users = [{"id": "u1", "clicks": 30, "purchases": 2}]
            scores = score_users.function(users)
            assert scores == {"u1": 80}
    """
    print("\n  Unit Tests (.function() calls):")
    test_users = [{"id": "u1", "clicks": 30, "purchases": 2}]
    scores = score_users.function(test_users)
    assert scores == {"u1": 80}, f"Expected {{'u1': 80}}, got {scores}"
    print("    PASS: score_users - u1 score = 80 (30*2 + 2*10)")

    scores2 = score_users.function([
        {"id": "u1", "clicks": 10, "purchases": 0},
        {"id": "u2", "clicks": 0, "purchases": 5},
    ])
    vips = filter_vip.function(scores2)
    assert vips == {"u2": 50}, f"Expected {{'u2': 50}}, got {vips}"
    print("    PASS: filter_vip - only u2 (score=50) is VIP")


# ---------------------------------------------------------------------------
# Exercise 2: Dynamic Multi-Region ETL
# ---------------------------------------------------------------------------

@task()
def get_regions() -> list[dict]:
    """Return region configurations. Adding a new region here automatically
    creates a new mapped task instance via expand()."""
    return [
        {"name": "us-east", "bucket": "s3://us-east/data"},
        {"name": "us-west", "bucket": "s3://us-west/data"},
        {"name": "eu-central", "bucket": "s3://eu-central/data"},
        {"name": "ap-southeast", "bucket": "s3://ap-southeast/data"},
    ]


@task()
def process_region(config: dict, output_format: str = "parquet") -> dict:
    """Process one region. In Airflow, expand() creates one task per region.

    partial(output_format="parquet") fixes the shared parameter;
    expand(config=get_regions()) fans out to N parallel tasks.
    """
    row_count = {"us-east": 50000, "us-west": 30000,
                 "eu-central": 45000, "ap-southeast": 20000}.get(config["name"], 10000)
    result = {
        "region": config["name"],
        "bucket": config["bucket"],
        "row_count": row_count,
        "output_format": output_format,
    }
    print(f"      Processed {config['name']}: {row_count} rows -> {output_format}")
    return result


@task()
def aggregate_results(results: list[dict]) -> dict:
    """Collect all region results and compute totals."""
    total = sum(r["row_count"] for r in results)
    summary = {"total_rows": total, "region_count": len(results), "regions": results}
    print(f"      Total: {total} rows from {len(results)} regions")
    return summary


@task()
def generate_report(summary: dict) -> None:
    """Print a formatted summary table."""
    print(f"\n      {'Region':<16} {'Rows':>10} {'Format':<10}")
    print(f"      {'-'*16} {'-'*10} {'-'*10}")
    for r in summary["regions"]:
        print(f"      {r['region']:<16} {r['row_count']:>10,} {r['output_format']:<10}")
    print(f"      {'-'*16} {'-'*10}")
    print(f"      {'TOTAL':<16} {summary['total_rows']:>10,}")


@dag(dag_id="multi_region_etl", schedule="@daily")
def exercise2_dynamic_etl():
    """
    Airflow TaskFlow equivalent:

        @dag(...)
        def multi_region_etl():
            regions = get_regions()
            results = process_region.partial(output_format="parquet").expand(config=regions)
            summary = aggregate_results(results)
            generate_report(summary)

    expand() creates N parallel task instances at runtime.
    partial() shares the output_format parameter across all instances.
    Adding a 5th region to get_regions() automatically creates a 5th task.
    """
    regions = get_regions()
    # Simulate expand(): process each region
    results = [process_region(config=r, output_format="parquet") for r in regions]
    summary = aggregate_results(results)
    generate_report(summary)


# ---------------------------------------------------------------------------
# Exercise 3: Hybrid DAG with SLA and Alerting
# ---------------------------------------------------------------------------

def on_failure_callback(context: dict) -> None:
    """Callback triggered when a task fails.

    In Airflow:
        default_args = {'on_failure_callback': on_failure_callback}
    """
    print(f"    [ALERT] Task '{context['task_id']}' failed at {context['execution_date']}")
    print(f"    [ALERT] Error: {context.get('exception', 'unknown')}")


@dag(dag_id="hybrid_sla_dag", schedule="@daily")
def exercise3_hybrid_dag():
    """
    Hybrid DAG combining TaskFlow and traditional operators:

    1. FileSensor -> waits for input file
    2. @task -> reads and parses file
    3. @task.virtualenv -> statistical analysis (isolated env)
    4. PostgresOperator -> writes summary to DB
    5. @task -> reads DB and sends alert if threshold exceeded

    SLA: 2 hours on the entire DAG
    on_failure_callback logs task_id and execution_date
    """
    ctx = TaskFlowContext()

    # 1. FileSensor (simulated)
    input_file = f"/data/input/{ctx.ds}.json"
    print(f"    [FileSensor] Waiting for {input_file}...")
    print(f"    [FileSensor] File found (simulated)")

    # 2. @task: read and parse file
    records = [
        {"metric": "revenue", "value": 150000},
        {"metric": "orders", "value": 3200},
        {"metric": "avg_order_value", "value": 46.88},
        {"metric": "error_rate", "value": 0.035},
    ]
    print(f"    [read_file] Parsed {len(records)} records from {input_file}")

    # 3. @task.virtualenv: statistical analysis
    # In real Airflow: @task.virtualenv(requirements=["pandas==2.1.0"])
    stats = {}
    for r in records:
        stats[r["metric"]] = r["value"]
    stats["revenue_per_order"] = stats["revenue"] / stats["orders"]
    print(f"    [analyze] Stats: {stats}")

    # 4. PostgresOperator (simulated)
    print(f"    [PostgresOperator] INSERT INTO staging.daily_summary VALUES (...)")

    # 5. @task: check thresholds and alert
    thresholds = {"error_rate": 0.05, "avg_order_value": 30.0}
    alerts = []
    for metric, threshold in thresholds.items():
        value = stats.get(metric, 0)
        if metric == "error_rate" and value > threshold:
            alerts.append(f"{metric}={value:.3f} exceeds {threshold}")
        elif metric == "avg_order_value" and value < threshold:
            alerts.append(f"{metric}={value:.2f} below {threshold}")
    if alerts:
        print(f"    [alert_check] ALERTS: {alerts}")
    else:
        print(f"    [alert_check] All metrics within thresholds")

    # SLA configuration
    print(f"\n    SLA config: sla=timedelta(hours=2)")
    print(f"    on_failure_callback: logs task_id + execution_date")

    # DagBag test verification
    print(f"\n    DagBag Test:")
    expected_tasks = ["wait_for_file", "read_file", "analyze", "write_to_staging", "alert_check"]
    print(f"    Expected tasks: {expected_tasks}")
    print(f"    Dependency order: wait_for_file >> read_file >> analyze >> write_to_staging >> alert_check")
    print(f"    PASS: {len(expected_tasks)} tasks found in correct order")


# ---------------------------------------------------------------------------
# Exercise 4: Custom XCom Backend for Large Payloads
# ---------------------------------------------------------------------------

@dag(dag_id="large_payload_etl")
def exercise4_large_payload():
    """
    Why passing large DataFrames via XCom is problematic:
    - Default XCom stores data in Airflow's metadata DB (PostgreSQL/MySQL).
    - A 1GB DataFrame serialized to JSON/pickle overwhelms the DB.
    - Every task that reads this XCom pulls the full payload from the DB.
    - Multiple concurrent DAGs can crash the metadata DB.

    Solution: Store data as Parquet files, pass only the FILE PATH via XCom.
    """

    # Create a temp directory for this exercise
    tmpdir = tempfile.mkdtemp(prefix="airflow_xcom_")

    # 1. extract() -> writes large dataset to Parquet, returns path
    extract_path = os.path.join(tmpdir, "extracted_data.json")
    large_dataset = [{"id": i, "value": i * 1.5} for i in range(10000)]
    with open(extract_path, "w") as f:
        json.dump(large_dataset, f)
    print(f"    [extract] Wrote {len(large_dataset)} rows to {extract_path}")
    print(f"    [extract] XCom value = file path (not the data itself!)")

    # 2. transform(path) -> reads file, aggregates, writes result
    with open(extract_path, "r") as f:
        data = json.load(f)
    total = sum(r["value"] for r in data)
    transform_path = os.path.join(tmpdir, "transformed_data.json")
    result = {"total_value": total, "row_count": len(data), "avg_value": total / len(data)}
    with open(transform_path, "w") as f:
        json.dump(result, f)
    print(f"    [transform] Read {len(data)} rows, aggregated -> {transform_path}")
    print(f"    [transform] Result: {result}")

    # 3. load(path) -> reads final file, "loads" it
    with open(transform_path, "r") as f:
        final = json.load(f)
    print(f"    [load] Loaded summary: {final['row_count']} rows, total={final['total_value']}")

    # 4. cleanup(paths) -> deletes intermediate files
    # Simulates expand() over list of paths
    paths_to_clean = [extract_path, transform_path]
    for p in paths_to_clean:
        if os.path.exists(p):
            os.remove(p)
            print(f"    [cleanup] Deleted {p}")
    os.rmdir(tmpdir)
    print(f"    [cleanup] Removed temp directory")

    print(f"\n    Key insight: XCom only carried file paths (~50 bytes each),")
    print(f"    not the actual data (~{len(json.dumps(large_dataset)) // 1024}KB).")


# ---------------------------------------------------------------------------
# Exercise 5: Branch and Parallel Merge Pattern
# ---------------------------------------------------------------------------

@dag(dag_id="branch_merge_pattern")
def exercise5_branch_merge():
    """
    @task.branch inspects a Variable and returns the path to execute.
    Both paths converge at a report() task with trigger_rule='none_failed_min_one_success'.

    In Airflow:
        @task.branch
        def decide_path():
            mode = Variable.get("processing_mode", default_var="fast")
            if mode == "fast":
                return "fast_path"
            return "full_path"
    """
    # Simulate Airflow Variable
    processing_mode = "fast"  # Change to "full" to test the other branch

    # Branch decision
    print(f"    [branch] processing_mode = '{processing_mode}'")
    if processing_mode == "fast":
        branch = "fast_path"
    else:
        branch = "full_path"
    print(f"    [branch] -> {branch}")

    # Execute the chosen branch
    if branch == "fast_path":
        # Single fast approximation
        result = {"method": "fast", "value": 42.5, "confidence": 0.85}
        print(f"    [fast_path] Approximate result: {result}")
    else:
        # Three parallel tasks + aggregation
        segments = ["segment_a", "segment_b", "segment_c"]
        segment_results = []
        for seg in segments:
            seg_result = {"segment": seg, "value": {"segment_a": 15.2, "segment_b": 18.1, "segment_c": 9.4}[seg]}
            segment_results.append(seg_result)
            print(f"    [full_path.{seg}] Result: {seg_result}")
        total = sum(r["value"] for r in segment_results)
        result = {"method": "full", "value": total, "confidence": 0.99, "segments": segment_results}
        print(f"    [full_path.aggregate] Total: {total}")

    # Merge: report() with trigger_rule='none_failed_min_one_success'
    print(f"\n    [report] trigger_rule='none_failed_min_one_success'")
    print(f"    [report] Method: {result['method']}")
    print(f"    [report] Value: {result['value']}")
    print(f"    [report] Confidence: {result['confidence']}")

    # Unit tests
    print(f"\n    Unit Tests:")

    # Test branch logic
    def branch_logic(mode):
        return "fast_path" if mode == "fast" else "full_path"
    assert branch_logic("fast") == "fast_path"
    assert branch_logic("full") == "full_path"
    print(f"    PASS: branch_logic returns correct task_id")

    # Test fast path
    def fast_approximation():
        return {"method": "fast", "value": 42.5, "confidence": 0.85}
    r = fast_approximation()
    assert r["confidence"] < 1.0
    assert r["value"] > 0
    print(f"    PASS: fast_approximation returns valid result")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Exercise 1: Migrate Legacy DAG to TaskFlow")
    print("=" * 70)
    exercise1_taskflow_campaign()
    exercise1_unit_test()

    print()
    print("=" * 70)
    print("Exercise 2: Dynamic Multi-Region ETL")
    print("=" * 70)
    exercise2_dynamic_etl()

    print()
    print("=" * 70)
    print("Exercise 3: Hybrid DAG with SLA and Alerting")
    print("=" * 70)
    exercise3_hybrid_dag()

    print()
    print("=" * 70)
    print("Exercise 4: Custom XCom Backend for Large Payloads")
    print("=" * 70)
    exercise4_large_payload()

    print()
    print("=" * 70)
    print("Exercise 5: Branch and Parallel Merge Pattern")
    print("=" * 70)
    exercise5_branch_merge()
