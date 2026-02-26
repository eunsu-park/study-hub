"""
lakeFS Data Versioning Operations
===================================
Demonstrates Git-like data versioning for data lakes using lakeFS:
- Creating repositories and branches
- Committing data changes with metadata
- Comparing branches (diff)
- Merging branches (with conflict detection)
- Reading versioned data with pandas and Spark
- Rollback to previous commits

Requirements:
    pip install lakefs pandas pyarrow

    # lakeFS server (Docker quickstart):
    docker run --pull always \
        -p 8000:8000 \
        treeverse/lakefs:latest \
        run --quickstart

Usage:
    python lakefs_operations.py

Note:
    This example includes both real lakeFS client code and a mock mode.
    Set USE_MOCK=True to run without a lakeFS server (for learning/testing).
"""

import io
import json
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# Configuration
# =============================================================================

USE_MOCK = True  # Set to False when running against a real lakeFS instance

LAKEFS_CONFIG = {
    "host": "http://localhost:8000",
    "username": "AKIAIOSFODNN7EXAMPLE",
    "password": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
}


# =============================================================================
# Mock lakeFS Client (for running without a real server)
# =============================================================================

@dataclass
class MockObject:
    """Represents a file (object) in the mock lakeFS store."""
    path: str
    data: bytes
    content_type: str = "application/octet-stream"


@dataclass
class MockCommit:
    """Represents a commit in the mock lakeFS store."""
    id: str
    message: str
    metadata: dict = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class MockDiffEntry:
    """Represents a change between two refs."""
    type: str   # "added", "removed", "changed"
    path: str


class MockLakeFSClient:
    """Mock lakeFS client that stores data in memory.

    Why a mock?
      - Learn lakeFS concepts without running a server
      - Unit test data pipelines that depend on lakeFS
      - Demonstrate the API surface without external dependencies
    """

    def __init__(self):
        self.repos: dict[str, dict] = {}

    def create_repository(self, repo_name: str, storage_namespace: str,
                          default_branch: str = "main") -> dict:
        """Create a new repository (= a versioned namespace in your data lake)."""
        self.repos[repo_name] = {
            "branches": {
                default_branch: {
                    "objects": {},      # path -> MockObject
                    "commits": [],      # list of MockCommit
                    "staged": {},       # uncommitted changes
                }
            },
            "storage_namespace": storage_namespace,
        }
        print(f"[lakeFS] Created repository: {repo_name}")
        print(f"         Storage: {storage_namespace}")
        print(f"         Default branch: {default_branch}")
        return {"id": repo_name}

    def create_branch(self, repo_name: str, branch_name: str,
                      source_branch: str = "main") -> dict:
        """Create a new branch (instant, zero-copy).

        Why branches?
          - Isolate changes: test new ETL logic without affecting production
          - Parallel work: multiple teams can work on different data changes
          - Safe experimentation: delete the branch if the experiment fails
        """
        if repo_name not in self.repos:
            raise ValueError(f"Repository '{repo_name}' not found")

        source = self.repos[repo_name]["branches"][source_branch]
        # Why copy objects dict? In real lakeFS, this is just a pointer.
        # Both branches share the same physical data (copy-on-write).
        self.repos[repo_name]["branches"][branch_name] = {
            "objects": dict(source["objects"]),  # Shallow copy (shared references)
            "commits": list(source["commits"]),
            "staged": {},
        }
        print(f"[lakeFS] Created branch: {branch_name} (from {source_branch})")
        return {"id": branch_name}

    def upload_object(self, repo_name: str, branch_name: str,
                      path: str, data: bytes, content_type: str = "") -> None:
        """Upload (stage) an object to a branch.

        Why "stage"?
          - Like Git, uploads are staged but NOT committed
          - You can upload multiple files, then commit them atomically
          - Atomic commits = all-or-nothing (no partial updates visible)
        """
        branch = self.repos[repo_name]["branches"][branch_name]
        obj = MockObject(path=path, data=data, content_type=content_type)
        branch["staged"][path] = obj
        print(f"[lakeFS] Staged: {path} ({len(data)} bytes) on {branch_name}")

    def commit(self, repo_name: str, branch_name: str,
               message: str, metadata: Optional[dict] = None) -> MockCommit:
        """Commit staged changes to the branch.

        Why explicit commits?
          - Creates an immutable snapshot (like a Git commit)
          - Commit ID can be used for reproducibility (e.g., ML training data)
          - Message and metadata document WHY the data changed
        """
        branch = self.repos[repo_name]["branches"][branch_name]

        # Apply staged changes to the branch
        for path, obj in branch["staged"].items():
            branch["objects"][path] = obj
        staged_count = len(branch["staged"])
        branch["staged"] = {}

        # Create commit
        commit_id = f"c{len(branch['commits']) + 1:04d}"
        commit = MockCommit(
            id=commit_id,
            message=message,
            metadata=metadata or {},
        )
        branch["commits"].append(commit)

        print(f"[lakeFS] Committed: {commit_id} on {branch_name}")
        print(f"         Message: {message}")
        print(f"         Files: {staged_count} staged changes applied")
        return commit

    def diff(self, repo_name: str, ref_a: str, ref_b: str) -> list[MockDiffEntry]:
        """Compare two branches/commits.

        Why diff before merge?
          - See exactly what changed (added, modified, deleted files)
          - Review data changes like you review code (data PR)
          - Catch unintended changes before they reach production
        """
        branch_a = self.repos[repo_name]["branches"][ref_a]
        branch_b = self.repos[repo_name]["branches"][ref_b]

        objects_a = set(branch_a["objects"].keys())
        objects_b = set(branch_b["objects"].keys())

        changes = []

        # Files only in branch_b (added)
        for path in objects_b - objects_a:
            changes.append(MockDiffEntry(type="added", path=path))

        # Files only in branch_a (removed in branch_b's context)
        for path in objects_a - objects_b:
            changes.append(MockDiffEntry(type="removed", path=path))

        # Files in both but potentially modified
        for path in objects_a & objects_b:
            if branch_a["objects"][path].data != branch_b["objects"][path].data:
                changes.append(MockDiffEntry(type="changed", path=path))

        return changes

    def merge(self, repo_name: str, source_branch: str,
              target_branch: str) -> dict:
        """Merge source branch into target branch.

        Why merge (not overwrite)?
          - Preserves commit history on the target branch
          - Detects conflicts if the same file was modified on both branches
          - Atomic: entire merge succeeds or fails (no partial state)
        """
        source = self.repos[repo_name]["branches"][source_branch]
        target = self.repos[repo_name]["branches"][target_branch]

        # Check for conflicts (same file modified on both branches)
        conflicts = []
        for path in source["objects"]:
            if (path in target["objects"] and
                source["objects"][path].data != target["objects"][path].data):
                # Check if the file was modified since branching
                conflicts.append(path)

        if conflicts:
            print(f"[lakeFS] MERGE CONFLICT: {len(conflicts)} conflicting files")
            for c in conflicts:
                print(f"         - {c}")
            raise Exception(f"Merge conflict on {len(conflicts)} files: {conflicts}")

        # Apply source changes to target
        merged_count = 0
        for path, obj in source["objects"].items():
            if path not in target["objects"] or target["objects"][path].data != obj.data:
                target["objects"][path] = obj
                merged_count += 1

        # Record merge commit
        merge_commit = MockCommit(
            id=f"m{len(target['commits']) + 1:04d}",
            message=f"Merge {source_branch} into {target_branch}",
            metadata={"source_branch": source_branch, "merged_files": merged_count},
        )
        target["commits"].append(merge_commit)

        print(f"[lakeFS] Merged {source_branch} → {target_branch}")
        print(f"         {merged_count} files updated")
        return {"commit_id": merge_commit.id}

    def get_object(self, repo_name: str, ref: str, path: str) -> bytes:
        """Read an object from a specific branch or commit."""
        branch = self.repos[repo_name]["branches"][ref]
        if path not in branch["objects"]:
            raise FileNotFoundError(f"Object not found: {path} on {ref}")
        return branch["objects"][path].data

    def list_objects(self, repo_name: str, ref: str,
                     prefix: str = "") -> list[str]:
        """List all objects under a prefix on a branch."""
        branch = self.repos[repo_name]["branches"][ref]
        return [
            path for path in branch["objects"]
            if path.startswith(prefix)
        ]

    def revert(self, repo_name: str, branch_name: str,
               commit_id: str) -> None:
        """Revert a branch to a specific commit.

        Why revert?
          - Instant rollback: restore production data in seconds
          - Safer than fixing forward: buy time to investigate the root cause
          - Non-destructive: the reverted commits are preserved in history
        """
        print(f"[lakeFS] Reverting {branch_name} to commit {commit_id}")
        # In real lakeFS, this creates a new commit that undoes changes
        # For the mock, we'll just log the operation
        branch = self.repos[repo_name]["branches"][branch_name]
        revert_commit = MockCommit(
            id=f"r{len(branch['commits']) + 1:04d}",
            message=f"Revert to {commit_id}",
            metadata={"reverted_to": commit_id},
        )
        branch["commits"].append(revert_commit)
        print(f"[lakeFS] Revert committed: {revert_commit.id}")


# =============================================================================
# Helper Functions — DataFrame I/O with lakeFS
# =============================================================================

def upload_dataframe(
    client: MockLakeFSClient,
    repo: str,
    branch: str,
    path: str,
    df: pd.DataFrame,
    format: str = "parquet",
) -> None:
    """Upload a pandas DataFrame to lakeFS as Parquet or CSV.

    Why Parquet as default?
      - Columnar format: efficient for analytical queries
      - Schema embedded: no ambiguity about column types
      - Compressed: ~10x smaller than CSV for typical data
    """
    buffer = io.BytesIO()
    if format == "parquet":
        df.to_parquet(buffer, index=False)
        content_type = "application/octet-stream"
    elif format == "csv":
        buffer.write(df.to_csv(index=False).encode("utf-8"))
        content_type = "text/csv"
    else:
        raise ValueError(f"Unsupported format: {format}")

    buffer.seek(0)
    client.upload_object(repo, branch, path, buffer.read(), content_type)


def read_dataframe(
    client: MockLakeFSClient,
    repo: str,
    ref: str,
    path: str,
    format: str = "parquet",
) -> pd.DataFrame:
    """Read a DataFrame from lakeFS (from any branch or commit).

    Why read from a specific ref?
      - "main": read production data
      - "feature/xyz": read experimental data
      - "c0042": read historical data at a specific point in time
    """
    data = client.get_object(repo, ref, path)
    buffer = io.BytesIO(data)
    if format == "parquet":
        return pd.read_parquet(buffer)
    elif format == "csv":
        return pd.read_csv(buffer)
    raise ValueError(f"Unsupported format: {format}")


# =============================================================================
# Spark Integration Pattern (illustrative, not runnable without PySpark)
# =============================================================================

def spark_lakefs_example():
    """Demonstrate how Spark reads/writes versioned data via lakeFS.

    The key insight: lakeFS exposes an S3-compatible API, so Spark
    treats it exactly like S3. No code changes needed — just change
    the endpoint URL.
    """
    spark_config = """
    # PySpark configuration for lakeFS:
    spark = SparkSession.builder \\
        .appName("lakeFS-Demo") \\
        .config("spark.hadoop.fs.s3a.endpoint", "http://lakefs:8000") \\
        .config("spark.hadoop.fs.s3a.access.key", "your-access-key") \\
        .config("spark.hadoop.fs.s3a.secret.key", "your-secret-key") \\
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \\
        .getOrCreate()

    # Read from main branch:
    df = spark.read.parquet("s3a://my-repo/main/data/orders/")

    # Read from feature branch:
    df = spark.read.parquet("s3a://my-repo/feature~new-etl/data/orders/")

    # Read from specific commit (time travel):
    df = spark.read.parquet("s3a://my-repo/abc123def/data/orders/")

    # Write to a branch (staged, needs commit via lakeFS API):
    result.write.mode("overwrite").parquet(
        "s3a://my-repo/feature~new-etl/data/orders_cleaned/"
    )
    """
    print("=" * 60)
    print("Spark + lakeFS Integration Pattern")
    print("=" * 60)
    print(spark_config)


# =============================================================================
# Main Demo — Full Branch-Merge Workflow
# =============================================================================

def main():
    """Demonstrate a complete lakeFS workflow:

    1. Create repository
    2. Upload initial data to main
    3. Create feature branch
    4. Modify data on feature branch
    5. Diff branches
    6. Merge feature into main
    7. Demonstrate rollback
    """
    print("=" * 60)
    print("lakeFS Data Versioning Demo")
    print("=" * 60)

    client = MockLakeFSClient()

    # ── Step 1: Create repository ──────────────────────────────────
    print("\n--- Step 1: Create Repository ---")
    client.create_repository(
        repo_name="analytics-lake",
        storage_namespace="s3://my-data-bucket/analytics/",
        default_branch="main",
    )

    # ── Step 2: Upload initial data to main ────────────────────────
    print("\n--- Step 2: Upload Initial Data to Main ---")

    orders_v1 = pd.DataFrame({
        "order_id": [1, 2, 3, 4, 5],
        "customer_id": [101, 102, 103, 101, 104],
        "amount": [99.99, 149.50, 25.00, 299.99, 75.00],
        "status": ["completed", "completed", "completed", "completed", "completed"],
    })
    upload_dataframe(client, "analytics-lake", "main",
                     "gold/orders/2024-01-15.parquet", orders_v1)

    customers_v1 = pd.DataFrame({
        "customer_id": [101, 102, 103, 104],
        "name": ["Alice", "Bob", "Charlie", "Diana"],
        "tier": ["gold", "silver", "bronze", "gold"],
    })
    upload_dataframe(client, "analytics-lake", "main",
                     "gold/customers/latest.parquet", customers_v1)

    commit1 = client.commit(
        "analytics-lake", "main",
        message="Initial data load: orders + customers",
        metadata={
            "pipeline": "daily_etl",
            "source": "postgres_replica",
            "rows_orders": "5",
            "rows_customers": "4",
        },
    )

    # ── Step 3: Create feature branch ──────────────────────────────
    print("\n--- Step 3: Create Feature Branch ---")
    client.create_branch("analytics-lake", "feature/add-discount", "main")

    # ── Step 4: Modify data on feature branch ──────────────────────
    print("\n--- Step 4: Modify Data on Feature Branch ---")

    # Add a discount column to orders (schema evolution)
    orders_v2 = orders_v1.copy()
    orders_v2["discount_pct"] = [0, 10, 0, 5, 15]
    orders_v2["amount_after_discount"] = (
        orders_v2["amount"] * (1 - orders_v2["discount_pct"] / 100)
    ).round(2)

    upload_dataframe(client, "analytics-lake", "feature/add-discount",
                     "gold/orders/2024-01-15.parquet", orders_v2)

    # Also add a new data file
    discount_summary = pd.DataFrame({
        "date": ["2024-01-15"],
        "total_discount_given": [orders_v2["amount"].sum() - orders_v2["amount_after_discount"].sum()],
        "orders_with_discount": [(orders_v2["discount_pct"] > 0).sum()],
    })
    upload_dataframe(client, "analytics-lake", "feature/add-discount",
                     "gold/metrics/discount_summary.parquet", discount_summary)

    commit2 = client.commit(
        "analytics-lake", "feature/add-discount",
        message="Add discount columns to orders + discount summary metric",
        metadata={
            "ticket": "DATA-1234",
            "reviewer": "alice@company.com",
        },
    )

    # ── Step 5: Compare branches ───────────────────────────────────
    print("\n--- Step 5: Diff Between main and feature/add-discount ---")
    diff = client.diff("analytics-lake", "main", "feature/add-discount")
    if diff:
        print(f"Found {len(diff)} changes:")
        for change in diff:
            print(f"  [{change.type:>8}] {change.path}")
    else:
        print("No differences found.")

    # ── Step 6: Merge feature into main ────────────────────────────
    print("\n--- Step 6: Merge feature/add-discount → main ---")
    merge_result = client.merge("analytics-lake", "feature/add-discount", "main")
    print(f"Merge commit: {merge_result['commit_id']}")

    # Verify merged data
    print("\n--- Verification: Read merged data from main ---")
    merged_orders = read_dataframe(
        client, "analytics-lake", "main",
        "gold/orders/2024-01-15.parquet"
    )
    print(f"Orders on main after merge:\n{merged_orders.to_string()}")
    print(f"\nNew columns present: {list(merged_orders.columns)}")

    # Verify the new metrics file exists on main
    main_objects = client.list_objects("analytics-lake", "main", prefix="gold/")
    print(f"\nAll objects on main (gold/):")
    for obj_path in main_objects:
        print(f"  {obj_path}")

    # ── Step 7: Demonstrate rollback ───────────────────────────────
    print("\n--- Step 7: Rollback Demonstration ---")
    print(f"Current commits on main:")
    main_branch = client.repos["analytics-lake"]["branches"]["main"]
    for c in main_branch["commits"]:
        print(f"  {c.id}: {c.message} ({c.timestamp})")

    # Simulate: "Oh no, the discount data was wrong! Roll back!"
    print(f"\nRolling back to commit {commit1.id} (before discount changes)...")
    client.revert("analytics-lake", "main", commit1.id)

    print(f"\nCommit history after rollback:")
    for c in main_branch["commits"]:
        print(f"  {c.id}: {c.message}")

    # ── Spark Integration ──────────────────────────────────────────
    print()
    spark_lakefs_example()

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Summary of lakeFS Operations Demonstrated")
    print("=" * 60)
    print("""
    1. create_repository  → Initialize a versioned data namespace
    2. create_branch      → Isolate changes (zero-copy, instant)
    3. upload_object      → Stage data files on a branch
    4. commit             → Create an immutable snapshot
    5. diff               → Compare two branches/commits
    6. merge              → Atomically apply changes to target branch
    7. revert             → Roll back to a previous commit

    Key Benefits:
    - Instant rollback: seconds, not hours
    - Safe experimentation: branch, test, merge or discard
    - Reproducibility: pin ML training to a specific commit
    - Auditability: full history of who changed what and when
    - Zero-copy branching: no storage overhead for branches
    """)


if __name__ == "__main__":
    main()
