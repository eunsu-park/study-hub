"""
Exercise Solutions: Lesson 06 - Prefect Modern Orchestration

Covers:
  - Problem 1: Basic Flow (ETL with 3 tasks)
  - Problem 2: Dynamic Tasks (parallel file processing)
  - Problem 3: Conditional Execution (different methods by data size)

Note: Simulates Prefect @flow/@task decorators in pure Python.
"""

from datetime import datetime
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


# ---------------------------------------------------------------------------
# Simulated Prefect primitives
# ---------------------------------------------------------------------------

def flow(name: str | None = None, retries: int = 0):
    """Simulates the @flow decorator.

    In Prefect:
        @flow(name="my_etl", retries=2)
        def my_etl():
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            flow_name = name or func.__name__
            print(f"\n[Flow: {flow_name}] Starting (retries={retries})")
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            print(f"[Flow: {flow_name}] Completed in {elapsed:.2f}s")
            return result
        wrapper.__name__ = func.__name__
        return wrapper
    return decorator


def task(name: str | None = None, retries: int = 0, tags: list[str] | None = None):
    """Simulates the @task decorator.

    In Prefect:
        @task(name="extract", retries=3, tags=["io"])
        def extract():
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            task_name = name or func.__name__
            tag_str = f" tags={tags}" if tags else ""
            print(f"  [Task: {task_name}]{tag_str} Running...")
            result = func(*args, **kwargs)
            print(f"  [Task: {task_name}] Done")
            return result
        wrapper.__name__ = func.__name__
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Problem 1: Basic Flow
# Write an ETL flow with 3 tasks (extraction, transformation, loading).
# ---------------------------------------------------------------------------

@task(name="extract_data", retries=2, tags=["io", "source"])
def extract_data() -> list[dict]:
    """Extract raw order data from source system.

    In Prefect, the @task decorator gives you:
    - Automatic retry on failure
    - State tracking (Pending -> Running -> Completed/Failed)
    - Result caching (optional)
    - Logging integration
    """
    orders = []
    for i in range(50):
        orders.append({
            "order_id": i + 1,
            "customer": f"customer_{random.randint(1, 20)}",
            "amount": round(random.uniform(10, 300), 2),
            "category": random.choice(["A", "B", "C"]),
            "status": random.choice(["completed", "completed", "cancelled"]),
        })
    print(f"    Extracted {len(orders)} orders")
    return orders


@task(name="transform_data", tags=["compute"])
def transform_data(raw_orders: list[dict]) -> dict:
    """Transform: filter completed orders and aggregate by category."""
    completed = [o for o in raw_orders if o["status"] == "completed"]
    summary: dict[str, dict] = {}
    for o in completed:
        cat = o["category"]
        if cat not in summary:
            summary[cat] = {"count": 0, "revenue": 0.0}
        summary[cat]["count"] += 1
        summary[cat]["revenue"] = round(summary[cat]["revenue"] + o["amount"], 2)
    print(f"    Transformed {len(completed)} completed orders into {len(summary)} categories")
    return summary


@task(name="load_data", tags=["io", "warehouse"])
def load_data(summary: dict) -> None:
    """Load aggregated data into destination."""
    print(f"    Loading {len(summary)} category summaries:")
    for cat, stats in sorted(summary.items()):
        print(f"      Category {cat}: {stats['count']} orders, ${stats['revenue']:.2f}")


@flow(name="basic_etl_flow", retries=1)
def basic_etl_flow():
    """
    Prefect equivalent:

        @flow(name="basic_etl_flow", retries=1)
        def basic_etl_flow():
            raw = extract_data()
            summary = transform_data(raw)
            load_data(summary)
    """
    raw = extract_data()
    summary = transform_data(raw)
    load_data(summary)
    return summary


# ---------------------------------------------------------------------------
# Problem 2: Dynamic Tasks
# Write a flow that accepts a list of files and processes each in parallel.
# ---------------------------------------------------------------------------

@task(name="process_file", retries=1)
def process_file(filepath: str) -> dict:
    """Process a single file: read, validate, count rows.

    In Prefect, you achieve parallelism by submitting tasks to a
    ConcurrentTaskRunner or DaskTaskRunner:

        @flow(task_runner=ConcurrentTaskRunner())
        def parallel_flow(files):
            futures = process_file.map(files)
            return [f.result() for f in futures]
    """
    # Simulate file processing
    simulated_rows = random.randint(100, 10000)
    simulated_errors = random.randint(0, int(simulated_rows * 0.02))
    time.sleep(random.uniform(0.05, 0.2))  # Simulate I/O
    result = {
        "filepath": filepath,
        "total_rows": simulated_rows,
        "error_rows": simulated_errors,
        "valid_rows": simulated_rows - simulated_errors,
    }
    print(f"    Processed {filepath}: {result['valid_rows']} valid / {result['total_rows']} total")
    return result


@flow(name="parallel_file_processing")
def parallel_file_processing(file_list: list[str]):
    """
    Prefect equivalent using .map() for fan-out:

        @flow(task_runner=ConcurrentTaskRunner())
        def parallel_file_processing(file_list):
            # .map() creates one task per element and runs them concurrently
            results = process_file.map(file_list)
            # Aggregate results
            total_valid = sum(r.result()['valid_rows'] for r in results)
            print(f'Total valid rows: {total_valid}')
    """
    # Simulate parallel execution with ThreadPoolExecutor
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_file, f): f for f in file_list}
        for future in as_completed(futures):
            results.append(future.result())

    # Aggregate
    total_valid = sum(r["valid_rows"] for r in results)
    total_rows = sum(r["total_rows"] for r in results)
    print(f"\n  Aggregate: {total_valid} valid rows out of {total_rows} total across {len(file_list)} files")
    return results


# ---------------------------------------------------------------------------
# Problem 3: Conditional Execution
# Write a flow that selects different processing methods based on data size.
# ---------------------------------------------------------------------------

@task(name="get_data_size")
def get_data_size(source: str) -> int:
    """Check the size of the incoming data."""
    sizes = {"small_source": 500, "medium_source": 50_000, "large_source": 5_000_000}
    size = sizes.get(source, 10_000)
    print(f"    Data source '{source}' has {size:,} rows")
    return size


@task(name="process_small")
def process_small(source: str, size: int) -> str:
    """Process small datasets in-memory with pandas-like logic.

    Suitable for <10,000 rows. Simple, fast, no distributed overhead.
    """
    print(f"    [Small] Processing {size:,} rows from '{source}' in-memory")
    print(f"    [Small] Using: pandas read -> transform -> write CSV")
    return "small_processing_complete"


@task(name="process_medium")
def process_medium(source: str, size: int) -> str:
    """Process medium datasets with chunked reading.

    Suitable for 10K-1M rows. Reads in chunks to manage memory.
    """
    chunk_size = 10_000
    chunks = (size + chunk_size - 1) // chunk_size
    print(f"    [Medium] Processing {size:,} rows in {chunks} chunks of {chunk_size:,}")
    print(f"    [Medium] Using: chunked pandas read -> incremental transform -> batch write")
    return "medium_processing_complete"


@task(name="process_large")
def process_large(source: str, size: int) -> str:
    """Process large datasets with distributed compute (Spark).

    Suitable for >1M rows. Requires distributed resources.
    """
    partitions = max(1, size // 100_000)
    print(f"    [Large] Processing {size:,} rows with {partitions} Spark partitions")
    print(f"    [Large] Using: Spark read -> distributed transform -> Parquet write")
    return "large_processing_complete"


@flow(name="conditional_processing")
def conditional_processing(source: str):
    """
    Prefect equivalent:

        @flow
        def conditional_processing(source: str):
            size = get_data_size(source)
            if size < 10_000:
                result = process_small(source, size)
            elif size < 1_000_000:
                result = process_medium(source, size)
            else:
                result = process_large(source, size)
            return result

    Prefect natively supports if/else inside flows — no special branching
    operator needed (unlike Airflow's BranchPythonOperator). This is one
    of Prefect's design advantages: flows are just Python functions.
    """
    size = get_data_size(source)

    if size < 10_000:
        result = process_small(source, size)
    elif size < 1_000_000:
        result = process_medium(source, size)
    else:
        result = process_large(source, size)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Problem 1: Basic ETL Flow")
    print("=" * 70)
    basic_etl_flow()

    print()
    print("=" * 70)
    print("Problem 2: Dynamic Tasks (Parallel File Processing)")
    print("=" * 70)
    files = [f"/data/input/file_{i:03d}.csv" for i in range(8)]
    parallel_file_processing(files)

    print()
    print("=" * 70)
    print("Problem 3: Conditional Execution (by Data Size)")
    print("=" * 70)
    for src in ["small_source", "medium_source", "large_source"]:
        conditional_processing(src)
