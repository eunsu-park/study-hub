"""
Exercise Solutions: Lesson 21 - Data Versioning and Data Contracts

Covers:
  - Exercise 1: lakeFS Branch-Merge Workflow
  - Exercise 2: Build a Pydantic Contract (user_events validation)
  - Exercise 3: Schema Evolution Assessment
  - Exercise 4: Contract-First Pipeline with Dagster
  - Exercise 5: DVC + lakeFS Integration Design

Note: Pure Python simulation of lakeFS, Pydantic contracts, and schema evolution.
"""

import json
import random
from datetime import datetime
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Simulated lakeFS
# ---------------------------------------------------------------------------

class LakeFSRepository:
    """Simulates a lakeFS repository with branches, commits, and diffs."""
    def __init__(self, name: str):
        self.name = name
        self.branches: dict[str, dict] = {
            "main": {"data": {}, "commits": [], "parent": None},
        }

    def create_branch(self, branch_name: str, source: str = "main") -> None:
        """Create a new branch from source (copy-on-write)."""
        self.branches[branch_name] = {
            "data": deepcopy(self.branches[source]["data"]),
            "commits": list(self.branches[source]["commits"]),
            "parent": source,
        }

    def write(self, branch: str, path: str, data: Any) -> None:
        """Write data to a path on a branch."""
        self.branches[branch]["data"][path] = data

    def read(self, branch: str, path: str) -> Any:
        return self.branches[branch]["data"].get(path)

    def commit(self, branch: str, message: str) -> dict:
        commit = {
            "id": f"commit_{len(self.branches[branch]['commits']) + 1:03d}",
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "branch": branch,
        }
        self.branches[branch]["commits"].append(commit)
        return commit

    def diff(self, branch_a: str, branch_b: str) -> list[dict]:
        """Compare two branches and return differences."""
        data_a = self.branches[branch_a]["data"]
        data_b = self.branches[branch_b]["data"]
        diffs = []
        all_paths = set(list(data_a.keys()) + list(data_b.keys()))
        for path in sorted(all_paths):
            in_a = path in data_a
            in_b = path in data_b
            if in_a and not in_b:
                diffs.append({"path": path, "type": "deleted"})
            elif not in_a and in_b:
                diffs.append({"path": path, "type": "added"})
            elif data_a[path] != data_b[path]:
                diffs.append({"path": path, "type": "modified"})
        return diffs

    def merge(self, source: str, target: str) -> dict:
        """Merge source branch into target."""
        source_data = self.branches[source]["data"]
        self.branches[target]["data"].update(source_data)
        merge_commit = self.commit(target, f"Merge '{source}' into '{target}'")
        return merge_commit


# ---------------------------------------------------------------------------
# Exercise 1: lakeFS Branch-Merge Workflow
# ---------------------------------------------------------------------------

def exercise1_lakefs_workflow():
    """Simulate a data pipeline change using lakeFS branches."""
    repo = LakeFSRepository("my-data-repo")

    # Write initial data to main
    orders = [
        {"order_id": 1, "customer": "Alice", "amount": 150.00, "status": "completed"},
        {"order_id": 2, "customer": "Bob", "amount": 89.99, "status": "pending"},
        {"order_id": 3, "customer": "Carol", "amount": 250.00, "status": "completed"},
    ]
    repo.write("main", "data/orders.json", orders)
    repo.commit("main", "Initial order data")
    print(f"\n  Step 1-2: Written initial data to main ({len(orders)} orders)")
    print(f"    Commit: {repo.branches['main']['commits'][-1]['id']}")

    # Create feature branch
    repo.create_branch("feature/add-discount-column", "main")
    print(f"\n  Step 3: Created branch 'feature/add-discount-column'")

    # Add discount column on feature branch
    branch_data = repo.read("feature/add-discount-column", "data/orders.json")
    for order in branch_data:
        order["discount"] = round(order["amount"] * random.uniform(0, 0.15), 2)
    repo.write("feature/add-discount-column", "data/orders.json", branch_data)
    repo.commit("feature/add-discount-column", "Add discount column to orders")
    print(f"\n  Step 4: Added 'discount' column on feature branch")
    for order in branch_data:
        print(f"    order_id={order['order_id']}: discount=${order['discount']}")

    # Compare branches
    diffs = repo.diff("main", "feature/add-discount-column")
    print(f"\n  Step 5: Diff (main vs feature branch):")
    for d in diffs:
        print(f"    {d['path']}: {d['type']}")

    # Merge
    merge_commit = repo.merge("feature/add-discount-column", "main")
    print(f"\n  Step 6: Merged feature -> main")
    print(f"    Merge commit: {merge_commit['id']}")

    # Verify
    merged_data = repo.read("main", "data/orders.json")
    has_discount = all("discount" in order for order in merged_data)
    print(f"\n  Step 7: Verification")
    print(f"    Orders on main have 'discount' column: {has_discount}")
    print(f"    Sample: {merged_data[0]}")

    return repo


# ---------------------------------------------------------------------------
# Exercise 2: Build a Pydantic Contract (user_events)
# ---------------------------------------------------------------------------

class EventType(Enum):
    CLICK = "click"
    VIEW = "view"
    PURCHASE = "purchase"


@dataclass
class ValidationError:
    row_index: int
    field: str
    value: Any
    error: str


def validate_user_event(row: dict, index: int) -> list[ValidationError]:
    """Validate a single user event against the contract.

    Pydantic equivalent:

        class UserEvent(BaseModel):
            event_id: str
            user_id: str
            event_type: EventType
            timestamp: datetime
            page_url: str
            metadata: Optional[dict] = None

            @field_validator('page_url')
            def validate_page_url(cls, v):
                if not v.startswith('/'):
                    raise ValueError('page_url must start with /')
                return v
    """
    errors = []

    # Required fields
    for field_name in ["event_id", "user_id", "event_type", "timestamp", "page_url"]:
        if field_name not in row or row[field_name] is None:
            errors.append(ValidationError(index, field_name, None, f"Required field '{field_name}' is missing"))

    if errors:
        return errors  # Can't validate further if required fields are missing

    # event_type must be in enum
    valid_types = {"click", "view", "purchase"}
    if row["event_type"] not in valid_types:
        errors.append(ValidationError(index, "event_type", row["event_type"],
                                      f"Must be one of {valid_types}"))

    # page_url must start with /
    if not str(row.get("page_url", "")).startswith("/"):
        errors.append(ValidationError(index, "page_url", row["page_url"],
                                      "Must start with '/'"))

    # timestamp must be valid ISO format
    try:
        datetime.fromisoformat(row["timestamp"])
    except (ValueError, TypeError):
        errors.append(ValidationError(index, "timestamp", row["timestamp"],
                                      "Invalid ISO timestamp format"))

    return errors


def validate_dataframe(data: list[dict], error_threshold: float = 0.1) -> dict:
    """Validate a batch of events and return valid records + error details.

    If more than error_threshold% of rows fail, reject the entire batch.
    """
    valid_records = []
    all_errors: list[ValidationError] = []

    for i, row in enumerate(data):
        errors = validate_user_event(row, i)
        if errors:
            all_errors.extend(errors)
        else:
            valid_records.append(row)

    error_rate = len(all_errors) / len(data) if data else 0
    batch_rejected = error_rate > error_threshold

    return {
        "total_rows": len(data),
        "valid_rows": len(valid_records),
        "error_count": len(all_errors),
        "error_rate": round(error_rate, 4),
        "batch_rejected": batch_rejected,
        "threshold": error_threshold,
        "valid_records": valid_records if not batch_rejected else [],
        "errors": all_errors,
    }


def exercise2_pydantic_contract():
    """Build and test a data contract for user_events."""
    events = [
        {"event_id": "e1", "user_id": "u1", "event_type": "click", "timestamp": "2024-11-15T10:00:00", "page_url": "/home", "metadata": {"ref": "google"}},
        {"event_id": "e2", "user_id": "u2", "event_type": "view", "timestamp": "2024-11-15T10:01:00", "page_url": "/product/1"},
        {"event_id": "e3", "user_id": "u3", "event_type": "purchase", "timestamp": "2024-11-15T10:02:00", "page_url": "/checkout"},
        {"event_id": "e4", "user_id": "u4", "event_type": "invalid_type", "timestamp": "2024-11-15T10:03:00", "page_url": "/home"},
        {"event_id": "e5", "user_id": "u5", "event_type": "click", "timestamp": "2024-11-15T10:04:00", "page_url": "http://external.com"},
        {"event_id": "e6", "user_id": None, "event_type": "view", "timestamp": "2024-11-15T10:05:00", "page_url": "/about"},
        {"event_id": "e7", "user_id": "u7", "event_type": "click", "timestamp": "not-a-date", "page_url": "/home"},
        {"event_id": "e8", "user_id": "u8", "event_type": "view", "timestamp": "2024-11-15T10:06:00", "page_url": "/blog"},
    ]

    result = validate_dataframe(events, error_threshold=0.5)

    print(f"\n  Validation Results:")
    print(f"    Total rows:     {result['total_rows']}")
    print(f"    Valid rows:     {result['valid_rows']}")
    print(f"    Error count:    {result['error_count']}")
    print(f"    Error rate:     {result['error_rate']:.2%}")
    print(f"    Threshold:      {result['threshold']:.0%}")
    print(f"    Batch rejected: {result['batch_rejected']}")

    if result["errors"]:
        print(f"\n  Errors:")
        for e in result["errors"]:
            print(f"    Row {e.row_index}: {e.field}={e.value!r} -> {e.error}")

    # Test with high error rate (should reject batch)
    print(f"\n  Testing batch rejection (threshold=10%):")
    bad_events = [{"event_id": f"e{i}"} for i in range(10)]  # All missing fields
    bad_result = validate_dataframe(bad_events, error_threshold=0.1)
    print(f"    Error rate: {bad_result['error_rate']:.0%} > threshold 10%")
    print(f"    Batch rejected: {bad_result['batch_rejected']}")

    return result


# ---------------------------------------------------------------------------
# Exercise 3: Schema Evolution Assessment
# ---------------------------------------------------------------------------

def exercise3_schema_evolution():
    """Detect and classify schema changes between two versions."""
    ORDER_SCHEMA_V1 = {
        "order_id": {"type": "int", "nullable": False},
        "customer_name": {"type": "string", "nullable": False},
        "amount": {"type": "float", "nullable": False},
        "status": {"type": "string", "nullable": True},
        "created_at": {"type": "timestamp", "nullable": False},
    }

    ORDER_SCHEMA_V2 = {
        "order_id": {"type": "int", "nullable": False},
        "customer_id": {"type": "int", "nullable": False},  # Renamed from customer_name
        "amount": {"type": "decimal", "nullable": False},    # Type changed
        "status": {"type": "string", "nullable": True},
        "created_at": {"type": "timestamp", "nullable": False},
        "priority": {"type": "string", "nullable": True},    # New optional column
    }

    def detect_changes(v1: dict, v2: dict) -> list[dict]:
        changes = []
        all_fields = set(list(v1.keys()) + list(v2.keys()))

        for field_name in sorted(all_fields):
            in_v1 = field_name in v1
            in_v2 = field_name in v2

            if in_v1 and not in_v2:
                changes.append({
                    "field": field_name,
                    "change_type": "removed",
                    "breaking": not v1[field_name]["nullable"],
                    "detail": f"Column '{field_name}' removed",
                    "migration": f"ALTER TABLE DROP COLUMN {field_name}; UPDATE consumers to remove references",
                })
            elif not in_v1 and in_v2:
                is_breaking = not v2[field_name]["nullable"]
                changes.append({
                    "field": field_name,
                    "change_type": "added",
                    "breaking": is_breaking,
                    "detail": f"Column '{field_name}' added ({'required' if is_breaking else 'optional'})",
                    "migration": f"ALTER TABLE ADD COLUMN {field_name} {v2[field_name]['type']}"
                                 + (" NOT NULL DEFAULT ..." if is_breaking else ""),
                })
            elif in_v1 and in_v2:
                if v1[field_name]["type"] != v2[field_name]["type"]:
                    changes.append({
                        "field": field_name,
                        "change_type": "type_changed",
                        "breaking": True,
                        "detail": f"Type: {v1[field_name]['type']} -> {v2[field_name]['type']}",
                        "migration": f"ALTER TABLE ALTER COLUMN {field_name} TYPE {v2[field_name]['type']}; "
                                     f"Verify all consumers handle new type",
                    })
                if v1[field_name]["nullable"] != v2[field_name]["nullable"]:
                    nullable_change = f"{'nullable' if v1[field_name]['nullable'] else 'not-null'} -> " \
                                      f"{'nullable' if v2[field_name]['nullable'] else 'not-null'}"
                    changes.append({
                        "field": field_name,
                        "change_type": "nullability_changed",
                        "breaking": not v2[field_name]["nullable"],
                        "detail": f"Nullability: {nullable_change}",
                        "migration": f"ALTER TABLE ALTER COLUMN {field_name} SET {'NOT NULL' if not v2[field_name]['nullable'] else 'NULL'}",
                    })

        return changes

    changes = detect_changes(ORDER_SCHEMA_V1, ORDER_SCHEMA_V2)

    print(f"\n  Schema V1 -> V2 Changes:")
    print(f"  {'Field':<18} {'Change':<20} {'Breaking':>8} {'Detail'}")
    print(f"  {'-'*18} {'-'*20} {'-'*8} {'-'*40}")
    for c in changes:
        breaking = "YES" if c["breaking"] else "no"
        print(f"  {c['field']:<18} {c['change_type']:<20} {breaking:>8} {c['detail']}")

    # Migration plan for breaking changes
    breaking_changes = [c for c in changes if c["breaking"]]
    print(f"\n  Migration Plan ({len(breaking_changes)} breaking changes):")
    for i, c in enumerate(breaking_changes, 1):
        print(f"    {i}. {c['field']}: {c['migration']}")

    # Test
    print(f"\n  Tests:")
    assert any(c["change_type"] == "removed" for c in changes)
    print(f"    PASS: detected removed column (customer_name)")
    assert any(c["change_type"] == "added" for c in changes)
    print(f"    PASS: detected added column (customer_id, priority)")
    assert any(c["change_type"] == "type_changed" for c in changes)
    print(f"    PASS: detected type change (amount: float -> decimal)")
    assert sum(1 for c in changes if c["breaking"]) >= 2
    print(f"    PASS: correctly identified {len(breaking_changes)} breaking changes")

    return changes


# ---------------------------------------------------------------------------
# Exercise 4: Contract-First Pipeline with Dagster
# ---------------------------------------------------------------------------

def exercise4_contract_first_pipeline():
    """Contract-first Dagster pipeline with asset checks."""

    # Contract (Pydantic model)
    TRANSACTION_CONTRACT = {
        "txn_id": {"type": "string", "nullable": False},
        "user_id": {"type": "string", "nullable": False},
        "amount": {"type": "float", "nullable": False, "min": 0.01},
        "category": {"type": "string", "nullable": False, "allowed": ["A", "B", "C", "D"]},
        "timestamp": {"type": "timestamp", "nullable": False},
    }

    # Asset: produce mock data
    def compute_transactions() -> list[dict]:
        """@dg.asset producing transaction data."""
        categories = ["A", "B", "C", "D"]
        data = []
        for i in range(20):
            data.append({
                "txn_id": f"T{i+1:04d}",
                "user_id": f"user_{random.randint(1, 10)}",
                "amount": round(random.uniform(5, 500), 2),
                "category": random.choice(categories),
                "timestamp": datetime(2024, 11, 15, 10 + i // 5, i * 3 % 60).isoformat(),
            })
        # Add one bad row for testing
        data.append({"txn_id": "T9999", "user_id": None, "amount": -10, "category": "Z", "timestamp": "bad"})
        return data

    # Asset check: validate against contract
    def check_transactions(data: list[dict]) -> dict:
        """@dg.asset_check validating transactions against the contract."""
        errors = []
        for i, row in enumerate(data):
            for field_name, rules in TRANSACTION_CONTRACT.items():
                value = row.get(field_name)
                if value is None and not rules["nullable"]:
                    errors.append(f"Row {i}: {field_name} is null")
                if "min" in rules and value is not None:
                    try:
                        if float(value) < rules["min"]:
                            errors.append(f"Row {i}: {field_name}={value} < {rules['min']}")
                    except (ValueError, TypeError):
                        pass
                if "allowed" in rules and value is not None and value not in rules["allowed"]:
                    errors.append(f"Row {i}: {field_name}={value!r} not in {rules['allowed']}")
        passed = len(errors) == 0
        return {"passed": passed, "errors": errors}

    # Run pipeline
    print(f"\n  Contract:")
    for field_name, rules in TRANSACTION_CONTRACT.items():
        constraints = ", ".join(f"{k}={v}" for k, v in rules.items() if k != "type")
        print(f"    {field_name}: {rules['type']} ({constraints})")

    data = compute_transactions()
    print(f"\n  Asset 'transactions': {len(data)} rows produced")

    check_result = check_transactions(data)
    print(f"\n  Asset Check Result: {'PASSED' if check_result['passed'] else 'FAILED'}")
    if check_result["errors"]:
        print(f"  Errors ({len(check_result['errors'])}):")
        for e in check_result["errors"][:5]:
            print(f"    {e}")

    # Downstream asset only runs if check passes
    if check_result["passed"]:
        print(f"\n  Downstream asset 'enriched_transactions': RUNNING")
    else:
        print(f"\n  Downstream asset 'enriched_transactions': SKIPPED (check failed)")
        # In practice, remove bad rows and retry
        clean_data = [r for r in data if r.get("user_id") and r.get("amount", 0) > 0]
        clean_check = check_transactions(clean_data)
        print(f"  After cleaning: {len(clean_data)} rows, check={'PASSED' if clean_check['passed'] else 'FAILED'}")

    return check_result


# ---------------------------------------------------------------------------
# Exercise 5: DVC + lakeFS Integration Design
# ---------------------------------------------------------------------------

def exercise5_hybrid_versioning():
    """Design a hybrid versioning strategy for an ML platform."""
    print(f"\n  Hybrid Versioning Architecture:")
    print(f"""
    +-----------------+     +-----------------+     +------------------+
    |  Raw Data       |     |  Features       |     |  Models          |
    |  (lakeFS)       |     |  (lakeFS)       |     |  (DVC + MLflow)  |
    |                 |     |                 |     |                  |
    |  - S3 storage   |     |  - S3 storage   |     |  - Git-tracked   |
    |  - Branch/merge |     |  - Branch/merge |     |    metadata      |
    |  - Commit IDs   |     |  - Commit IDs   |     |  - S3 storage    |
    +-----------------+     +-----------------+     +------------------+
           |                       |                       |
           +----------+------------+-----------+-----------+
                      |                        |
                +-----------+            +-----------+
                |  CI/CD    |            |  MLflow   |
                |  Pipeline |            |  Registry |
                +-----------+            +-----------+
    """)

    print(f"  Versioning Workflow:")
    steps = [
        ("1. Data Scientist creates feature branch on lakeFS",
         "lakectl branch create lakefs://repo/feature/new-features -s lakefs://repo/main"),
        ("2. Write new feature set to lakeFS branch",
         "df.write.parquet('lakefs://repo/feature/new-features/features/user_features.parquet')"),
        ("3. Validate data contract (CI check)",
         "python validate_contract.py --schema features/schema.json --data lakefs://..."),
        ("4. Train model on feature branch data",
         "mlflow.start_run(); model = train(features_from_lakefs); mlflow.log_model(model)"),
        ("5. Track model artifacts with DVC",
         "dvc add models/model.pkl && git add models/model.pkl.dvc && git commit"),
        ("6. Merge feature data to main",
         "lakectl merge lakefs://repo/feature/new-features lakefs://repo/main"),
        ("7. Promote model to production in MLflow",
         "mlflow.register_model(model_uri, 'production')"),
    ]

    for step, cmd in steps:
        print(f"    {step}")
        print(f"      $ {cmd}")
        print()

    print(f"  CI/CD Pipeline Pseudocode:")
    pipeline = """
    on_lakefs_merge_to_main:
      - validate_data_contract:
          schema: schemas/features.json
          data: lakefs://repo/main/features/
          action: BLOCK merge if validation fails

      - run_data_quality_checks:
          suite: great_expectations/features_suite.json
          threshold: 99%

      - trigger_model_retrain:
          if: feature_schema_changed
          job: train_and_evaluate
          data_ref: lakefs://repo/main/features/

      - register_model:
          if: model_metrics > production_baseline
          action: promote to staging
    """
    print(pipeline)

    print(f"  Artifact Versioning Summary:")
    print(f"  {'Artifact':<20} {'Tool':<15} {'Why'}")
    print(f"  {'-'*20} {'-'*15} {'-'*40}")
    artifacts = [
        ("Raw data", "lakeFS", "Large files, branch/merge workflow, S3-native"),
        ("Features", "lakeFS", "Same as raw data, co-versioned with raw data"),
        ("Model weights", "DVC", "Git-like versioning, works with any storage"),
        ("Model metadata", "MLflow", "Experiment tracking, registry, deployment"),
        ("Code", "Git", "Source code versioning (standard)"),
        ("Schemas/Contracts", "Git", "Versioned with code, CI/CD validation"),
    ]
    for artifact, tool, why in artifacts:
        print(f"  {artifact:<20} {tool:<15} {why}")

    return artifacts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Exercise 1: lakeFS Branch-Merge Workflow")
    print("=" * 70)
    exercise1_lakefs_workflow()

    print()
    print("=" * 70)
    print("Exercise 2: Build a Pydantic Contract (user_events)")
    print("=" * 70)
    exercise2_pydantic_contract()

    print()
    print("=" * 70)
    print("Exercise 3: Schema Evolution Assessment")
    print("=" * 70)
    exercise3_schema_evolution()

    print()
    print("=" * 70)
    print("Exercise 4: Contract-First Pipeline with Dagster")
    print("=" * 70)
    exercise4_contract_first_pipeline()

    print()
    print("=" * 70)
    print("Exercise 5: DVC + lakeFS Integration Design")
    print("=" * 70)
    exercise5_hybrid_versioning()
