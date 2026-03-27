"""
Exercise Solutions: Lesson 20 - Dagster Asset-Based Orchestration

Covers:
  - Exercise 1: Build Your Own Asset Pipeline (blog analytics)
  - Exercise 2: Partitioned Incremental Pipeline
  - Exercise 3: Resource Injection and Testing
  - Exercise 4: Dagster-dbt Integration Design
  - Exercise 5: Sensor-Driven Pipeline

Note: Pure Python simulation of Dagster concepts.
"""

import random
import os
import csv
import tempfile
import json
from datetime import datetime, date, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
from io import StringIO


# ---------------------------------------------------------------------------
# Simulated Dagster primitives
# ---------------------------------------------------------------------------

@dataclass
class AssetMetadata:
    """Metadata attached to a Dagster asset."""
    description: str = ""
    row_count: int = 0
    columns: list[str] = field(default_factory=list)
    tags: dict = field(default_factory=dict)


class Asset:
    """Simulates a Dagster software-defined asset."""
    def __init__(self, name: str, compute_fn, deps: list[str] = None,
                 metadata: AssetMetadata = None, partition_def=None):
        self.name = name
        self.compute_fn = compute_fn
        self.deps = deps or []
        self.metadata = metadata or AssetMetadata()
        self.partition_def = partition_def
        self._materialized_data = None

    def materialize(self, **inputs):
        self._materialized_data = self.compute_fn(**inputs)
        if isinstance(self._materialized_data, list):
            self.metadata.row_count = len(self._materialized_data)
        return self._materialized_data


class DailyPartition:
    """Simulates DailyPartitionsDefinition."""
    def __init__(self, start_date: str):
        self.start_date = start_date

    def get_partition_keys(self, end_date: str) -> list[str]:
        start = datetime.strptime(self.start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        keys = []
        current = start
        while current <= end:
            keys.append(current.isoformat())
            current += timedelta(days=1)
        return keys


class WeeklyPartition:
    """Simulates WeeklyPartitionsDefinition."""
    def __init__(self, start_date: str):
        self.start_date = start_date

    def get_partition_keys(self, end_date: str) -> list[str]:
        start = datetime.strptime(self.start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        keys = []
        current = start
        while current <= end:
            keys.append(current.isoformat())
            current += timedelta(weeks=1)
        return keys


# ---------------------------------------------------------------------------
# Exercise 1: Build Your Own Asset Pipeline (Blog Analytics)
# ---------------------------------------------------------------------------

def exercise1_asset_pipeline():
    """
    Dagster equivalent:

        @dg.asset(description="Raw page views from analytics database")
        def raw_page_views() -> pd.DataFrame:
            ...

        @dg.asset(deps=[raw_page_views], description="Cleaned page views")
        def cleaned_page_views(raw_page_views: pd.DataFrame) -> pd.DataFrame:
            ...

        @dg.asset(deps=[cleaned_page_views])
        def page_popularity(cleaned_page_views: pd.DataFrame) -> pd.DataFrame:
            ...
    """
    # Asset 1: raw_page_views
    def compute_raw_page_views():
        pages = ["/home", "/blog/python-tips", "/blog/data-engineering",
                 "/blog/ml-intro", "/about", "/contact"]
        bots = ["bot", "crawler", "spider"]
        data = []
        for i in range(200):
            is_bot = random.random() < 0.15  # 15% bot traffic
            user_agent = random.choice(bots) + "-agent" if is_bot else f"user_{random.randint(1, 50)}"
            data.append({
                "page_url": random.choice(pages),
                "user_id": user_agent,
                "timestamp": (datetime(2024, 11, 15) + timedelta(seconds=random.randint(0, 86400))).isoformat(),
                "session_id": f"sess_{random.randint(1, 80)}",
            })
        return data

    raw_asset = Asset("raw_page_views", compute_raw_page_views,
                      metadata=AssetMetadata(description="Raw page views from analytics"))
    raw_data = raw_asset.materialize()
    print(f"\n  raw_page_views: {len(raw_data)} events (includes bot traffic)")

    # Asset 2: cleaned_page_views
    def compute_cleaned_page_views(raw_page_views=None):
        data = raw_page_views or raw_data
        # Remove bot traffic
        cleaned = [r for r in data if not any(b in r["user_id"].lower() for b in ["bot", "crawler", "spider"])]
        # Deduplicate by session
        seen_sessions = set()
        deduped = []
        for r in cleaned:
            key = (r["session_id"], r["page_url"])
            if key not in seen_sessions:
                seen_sessions.add(key)
                deduped.append(r)
        return deduped

    clean_asset = Asset("cleaned_page_views", compute_cleaned_page_views,
                        deps=["raw_page_views"],
                        metadata=AssetMetadata(description="Page views without bots, deduplicated"))
    clean_data = clean_asset.materialize(raw_page_views=raw_data)
    print(f"  cleaned_page_views: {len(clean_data)} events (bots removed, deduped)")

    # Asset 3: page_popularity
    def compute_page_popularity(cleaned_page_views=None):
        data = cleaned_page_views or clean_data
        page_visitors: dict[str, set] = defaultdict(set)
        for r in data:
            page_visitors[r["page_url"]].add(r["user_id"])
        ranking = [
            {"page_url": url, "unique_visitors": len(visitors), "rank": 0}
            for url, visitors in page_visitors.items()
        ]
        ranking.sort(key=lambda x: -x["unique_visitors"])
        for i, r in enumerate(ranking):
            r["rank"] = i + 1
        return ranking

    pop_asset = Asset("page_popularity", compute_page_popularity,
                      deps=["cleaned_page_views"],
                      metadata=AssetMetadata(description="Pages ranked by unique visitors"))
    pop_data = pop_asset.materialize(cleaned_page_views=clean_data)
    print(f"\n  page_popularity:")
    print(f"  {'Rank':>4} {'Page URL':<30} {'Unique Visitors':>15}")
    print(f"  {'-'*4} {'-'*30} {'-'*15}")
    for r in pop_data:
        print(f"  {r['rank']:>4} {r['page_url']:<30} {r['unique_visitors']:>15}")

    # Tests
    print(f"\n  Tests:")
    assert len(raw_data) > 0, "raw_page_views should not be empty"
    print(f"    PASS: raw_page_views is non-empty ({len(raw_data)} rows)")
    assert len(clean_data) < len(raw_data), "cleaned should have fewer rows than raw"
    print(f"    PASS: cleaned has fewer rows ({len(clean_data)}) than raw ({len(raw_data)})")
    assert all(r["rank"] > 0 for r in pop_data), "All pages should be ranked"
    print(f"    PASS: all pages have a rank")

    return raw_asset, clean_asset, pop_asset


# ---------------------------------------------------------------------------
# Exercise 2: Partitioned Incremental Pipeline
# ---------------------------------------------------------------------------

def exercise2_partitioned_pipeline():
    """Time-partitioned assets with daily and weekly granularity."""
    daily_def = DailyPartition("2024-11-01")
    weekly_def = WeeklyPartition("2024-11-04")

    daily_keys = daily_def.get_partition_keys("2024-11-07")
    weekly_keys = weekly_def.get_partition_keys("2024-11-10")

    print(f"\n  Daily partition keys: {daily_keys}")
    print(f"  Weekly partition keys: {weekly_keys}")

    # Simulate daily materialization
    daily_results: dict[str, list] = {}
    pages = ["/home", "/blog/a", "/blog/b", "/about"]
    for day_key in daily_keys:
        events = []
        for _ in range(random.randint(50, 100)):
            events.append({
                "page_url": random.choice(pages),
                "user_id": f"user_{random.randint(1, 30)}",
                "timestamp": day_key + "T" + f"{random.randint(0, 23):02d}:00:00",
            })
        daily_results[day_key] = events
        print(f"  Materialized raw_page_views [{day_key}]: {len(events)} events")

    # Weekly aggregation
    print(f"\n  Weekly Report (2024-11-04 to 2024-11-10):")
    all_events = []
    for events in daily_results.values():
        all_events.extend(events)

    page_counts: dict[str, int] = defaultdict(int)
    for e in all_events:
        page_counts[e["page_url"]] += 1

    for page, count in sorted(page_counts.items(), key=lambda x: -x[1]):
        print(f"    {page:<20} {count:>5} views")

    # Schedule
    print(f"\n  Schedule: materialize yesterday's partitions at 2 AM UTC")
    print(f"    @dg.schedule(cron='0 2 * * *', target=daily_assets)")
    print(f"    Backfill: dagster dev -> UI -> Launchpad -> select date range")

    return daily_results


# ---------------------------------------------------------------------------
# Exercise 3: Resource Injection and Testing
# ---------------------------------------------------------------------------

class PostgresResource:
    """Mock Postgres resource for dependency injection."""
    def __init__(self, connection_string: str = "mock://localhost:5432/db"):
        self.connection_string = connection_string
        self._data: dict[str, list] = {"user_profiles": [
            {"user_id": 1, "name": "Alice", "tier": "gold"},
            {"user_id": 2, "name": "Bob", "tier": "silver"},
            {"user_id": 3, "name": "Carol", "tier": "bronze"},
        ]}

    def query(self, table: str) -> list[dict]:
        return self._data.get(table, [])


class S3Resource:
    """Mock S3 resource for dependency injection."""
    def __init__(self, bucket: str = "mock-bucket"):
        self.bucket = bucket
        self._storage: dict[str, str] = {}

    def write(self, key: str, data: str) -> str:
        self._storage[key] = data
        return f"s3://{self.bucket}/{key}"

    def read(self, key: str) -> str | None:
        return self._storage.get(key)


def exercise3_resource_injection():
    """
    Dagster equivalent:

        class PostgresResource(dg.ConfigurableResource):
            connection_string: str
            def query(self, table: str) -> pd.DataFrame: ...

        @dg.asset
        def user_profiles(postgres: PostgresResource, s3: S3Resource):
            data = postgres.query("user_profiles")
            s3.write("output/user_profiles.json", data.to_json())
            return data
    """
    # Production resources
    postgres = PostgresResource("postgresql://prod:5432/analytics")
    s3 = S3Resource("prod-data-lake")

    # Asset using resources
    def compute_user_profiles(pg: PostgresResource, s3_res: S3Resource):
        profiles = pg.query("user_profiles")
        path = s3_res.write("output/user_profiles.json", json.dumps(profiles))
        return {"profiles": profiles, "s3_path": path}

    result = compute_user_profiles(postgres, s3)
    print(f"\n  user_profiles asset (production resources):")
    print(f"    Profiles loaded: {len(result['profiles'])}")
    print(f"    Written to: {result['s3_path']}")

    # Test 1: Mock resources
    print(f"\n  Test 1: Using mock resources")
    mock_pg = PostgresResource("mock://test")
    mock_pg._data["user_profiles"] = [{"user_id": 99, "name": "Test", "tier": "test"}]
    mock_s3 = S3Resource("test-bucket")

    test_result = compute_user_profiles(mock_pg, mock_s3)
    assert len(test_result["profiles"]) == 1
    assert test_result["profiles"][0]["name"] == "Test"
    print(f"    PASS: mock resource returns test data")

    # Test 2: materialize with mem_io_manager
    print(f"\n  Test 2: Using dg.materialize() with mem_io_manager")
    mem_result = compute_user_profiles(postgres, S3Resource("mem-bucket"))
    assert "profiles" in mem_result
    assert len(mem_result["profiles"]) > 0
    print(f"    PASS: materialization succeeds with {len(mem_result['profiles'])} profiles")

    return result


# ---------------------------------------------------------------------------
# Exercise 4: Dagster-dbt Integration Design
# ---------------------------------------------------------------------------

def exercise4_dbt_integration():
    """Design a hybrid pipeline with dbt models and Python assets."""
    print(f"\n  Hybrid Pipeline: dbt + Python Assets")
    print(f"\n  Asset Dependency Graph:")
    print(f"""
    [Source: raw_users]     [Source: raw_events]
          |                       |
          v                       v
    [dbt: stg_users]        [dbt: stg_events]
          |                       |
          +----------+------------+
                     |
                     v
           [dbt: fct_user_activity]
                     |
                     v
          [Python: ml_features]      <- Feature computation (pandas/numpy)
                     |
                     v
          [Python: model_scoring]    <- ML model inference
                     |
                     v
          [Python: scored_users]     <- Output to downstream systems
    """)

    print(f"  dbt Assets (SQL transformations):")
    dbt_models = {
        "stg_users": {"type": "staging", "materialization": "view", "partitioned": False},
        "stg_events": {"type": "staging", "materialization": "view", "partitioned": False},
        "fct_user_activity": {"type": "mart", "materialization": "incremental", "partitioned": True},
    }
    for name, config in dbt_models.items():
        part = " [daily partitioned]" if config["partitioned"] else ""
        print(f"    {name}: {config['type']}, materialized={config['materialization']}{part}")

    print(f"\n  Python Assets:")
    python_assets = {
        "ml_features": {"deps": ["fct_user_activity"], "partitioned": True,
                        "reason": "Feature computation uses pandas/numpy, not SQL-friendly"},
        "model_scoring": {"deps": ["ml_features"], "partitioned": True,
                          "reason": "ML inference requires Python (sklearn, torch)"},
        "scored_users": {"deps": ["model_scoring"], "partitioned": False,
                         "reason": "Final output, full refresh daily"},
    }
    for name, config in python_assets.items():
        part = " [daily partitioned]" if config["partitioned"] else ""
        print(f"    {name}: deps={config['deps']}{part}")
        print(f"      Reason: {config['reason']}")

    print(f"\n  Which assets should be partitioned and why:")
    print(f"    fct_user_activity: YES - daily aggregation, incremental processing")
    print(f"    ml_features:       YES - computed daily, depends on partitioned input")
    print(f"    model_scoring:     YES - scores daily features")
    print(f"    scored_users:      NO  - small output table, full refresh is fast")
    print(f"    stg_users/events:  NO  - views, no materialization needed")

    return dbt_models, python_assets


# ---------------------------------------------------------------------------
# Exercise 5: Sensor-Driven Pipeline
# ---------------------------------------------------------------------------

class DirectorySensor:
    """Simulates a Dagster sensor that watches a directory for new CSV files.

    Dagster equivalent:

        @dg.sensor(job=ingest_job, minimum_interval_seconds=30)
        def new_file_sensor(context):
            last_cursor = context.cursor or ""
            new_files = get_new_files_since(last_cursor)
            if new_files:
                context.update_cursor(max(new_files))
                for f in new_files:
                    yield RunRequest(run_key=f, run_config={...})
    """
    def __init__(self, watch_dir: str):
        self.watch_dir = watch_dir
        self.cursor: str = ""  # Tracks last processed file
        self.processed_files: set[str] = set()

    def evaluate(self) -> list[str]:
        """Check for new files, return list of unprocessed file paths."""
        if not os.path.exists(self.watch_dir):
            return []
        all_files = sorted(
            f for f in os.listdir(self.watch_dir) if f.endswith(".csv")
        )
        new_files = [
            os.path.join(self.watch_dir, f)
            for f in all_files
            if os.path.join(self.watch_dir, f) not in self.processed_files
        ]
        return new_files

    def mark_processed(self, filepath: str) -> None:
        self.processed_files.add(filepath)
        self.cursor = filepath


def ingest_clean_aggregate(filepath: str) -> dict:
    """Pipeline triggered by sensor: ingest -> clean -> aggregate."""
    # Edge case: empty file
    if os.path.getsize(filepath) == 0:
        return {"filepath": filepath, "status": "skipped", "reason": "empty file"}

    # Read CSV
    try:
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        return {"filepath": filepath, "status": "error", "reason": str(e)}

    if not rows:
        return {"filepath": filepath, "status": "skipped", "reason": "no data rows"}

    # Clean: filter valid rows
    valid_rows = []
    for r in rows:
        try:
            amount = float(r.get("amount", 0))
            if amount > 0 and r.get("product_id"):
                valid_rows.append(r)
        except (ValueError, TypeError):
            continue

    # Aggregate: total by product
    product_totals: dict[str, float] = defaultdict(float)
    for r in valid_rows:
        product_totals[r["product_id"]] += float(r["amount"])

    return {
        "filepath": filepath,
        "status": "success",
        "raw_rows": len(rows),
        "valid_rows": len(valid_rows),
        "products": dict(product_totals),
    }


def exercise5_sensor_pipeline():
    """Demonstrate sensor-driven ingestion with edge case handling."""
    tmpdir = tempfile.mkdtemp(prefix="dagster_sensor_")
    sensor = DirectorySensor(tmpdir)

    # Create test files
    # Normal file
    with open(os.path.join(tmpdir, "orders_001.csv"), "w") as f:
        f.write("order_id,product_id,amount\n")
        for i in range(10):
            f.write(f"O{i+1},P{random.randint(1, 3)},{round(random.uniform(10, 200), 2)}\n")

    # Empty file
    with open(os.path.join(tmpdir, "orders_002.csv"), "w") as f:
        pass  # Empty

    # File with bad data
    with open(os.path.join(tmpdir, "orders_003.csv"), "w") as f:
        f.write("order_id,product_id,amount\n")
        f.write("O1,P1,abc\n")  # Invalid amount
        f.write("O2,,50.00\n")  # Missing product_id
        f.write("O3,P2,75.00\n")  # Valid

    # Duplicate file name (sensor should track processed files)
    with open(os.path.join(tmpdir, "orders_004.csv"), "w") as f:
        f.write("order_id,product_id,amount\n")
        f.write("O1,P1,100.00\n")

    # Sensor evaluation cycle 1
    print(f"\n  Sensor Evaluation Cycle 1:")
    new_files = sensor.evaluate()
    print(f"    Found {len(new_files)} new files")

    results = []
    for filepath in new_files:
        result = ingest_clean_aggregate(filepath)
        sensor.mark_processed(filepath)
        results.append(result)
        filename = os.path.basename(filepath)
        print(f"    {filename}: {result['status']}", end="")
        if result["status"] == "success":
            print(f" ({result['raw_rows']} raw -> {result['valid_rows']} valid)")
        elif result.get("reason"):
            print(f" ({result['reason']})")
        else:
            print()

    # Sensor evaluation cycle 2 (should find no new files)
    print(f"\n  Sensor Evaluation Cycle 2:")
    new_files = sensor.evaluate()
    print(f"    Found {len(new_files)} new files (should be 0 - all processed)")
    assert len(new_files) == 0, "Duplicate files should be filtered by cursor"
    print(f"    PASS: no duplicates processed")

    # Add a new file
    with open(os.path.join(tmpdir, "orders_005.csv"), "w") as f:
        f.write("order_id,product_id,amount\n")
        f.write("O1,P3,300.00\n")

    print(f"\n  Sensor Evaluation Cycle 3 (new file added):")
    new_files = sensor.evaluate()
    print(f"    Found {len(new_files)} new file(s)")
    for filepath in new_files:
        result = ingest_clean_aggregate(filepath)
        sensor.mark_processed(filepath)
        print(f"    {os.path.basename(filepath)}: {result['status']}")
        results.append(result)

    # Cleanup
    for f in os.listdir(tmpdir):
        os.remove(os.path.join(tmpdir, f))
    os.rmdir(tmpdir)

    print(f"\n  Summary:")
    print(f"    Total files processed: {len(sensor.processed_files)}")
    print(f"    Successful: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"    Skipped: {sum(1 for r in results if r['status'] == 'skipped')}")
    print(f"    Errors: {sum(1 for r in results if r['status'] == 'error')}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Exercise 1: Blog Analytics Asset Pipeline")
    print("=" * 70)
    exercise1_asset_pipeline()

    print()
    print("=" * 70)
    print("Exercise 2: Partitioned Incremental Pipeline")
    print("=" * 70)
    exercise2_partitioned_pipeline()

    print()
    print("=" * 70)
    print("Exercise 3: Resource Injection and Testing")
    print("=" * 70)
    exercise3_resource_injection()

    print()
    print("=" * 70)
    print("Exercise 4: Dagster-dbt Integration Design")
    print("=" * 70)
    exercise4_dbt_integration()

    print()
    print("=" * 70)
    print("Exercise 5: Sensor-Driven Pipeline")
    print("=" * 70)
    exercise5_sensor_pipeline()
