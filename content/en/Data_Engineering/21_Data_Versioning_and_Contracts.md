[← Previous: 20. Dagster Asset Orchestration](20_Dagster_Asset_Orchestration.md) | [Next: Overview →](00_Overview.md)

# Data Versioning and Data Contracts

## Learning Objectives

1. Understand why data versioning is essential for reproducibility, rollback, and auditability
2. Master lakeFS concepts: branching, committing, merging, and diffing data like Git
3. Compare DVC and lakeFS for different data versioning use cases
4. Define and enforce data contracts between data producers and consumers
5. Implement schema contracts using Pydantic, JSON Schema, and Avro
6. Build a contract-first data pipeline with automated validation gates
7. Connect data contracts to data mesh principles and domain ownership

---

## Overview

Modern data platforms suffer from two related problems. First, **data is mutable by default** — when a Spark job overwrites a partition, the previous version vanishes unless you explicitly preserved it. There is no "undo" button, no blame history, no way to compare today's output with yesterday's. Second, **data interfaces are implicit** — upstream teams change column names, data types, or business logic without warning, and downstream pipelines break silently at 3 AM.

Data versioning and data contracts are the answers to these problems. Versioning gives data the same safety net that Git gives code: branches, commits, diffs, and rollbacks. Contracts formalize the agreements between data producers and consumers, ensuring that changes are intentional, communicated, and validated.

Together, these practices transform a fragile, trust-based data ecosystem into a robust, engineering-grade platform. In this lesson, we will build practical skills in both areas — from branching a data lake with lakeFS to enforcing schema contracts with automated tests.

> **Analogy**: Data contracts are like SLAs between departments — the finance team promises to deliver clean transaction data in an agreed format, and the analytics team knows exactly what to expect. If the finance team needs to change the format, they propose an update, both sides review it, and the transition is coordinated. No more surprise breakages.

---

## 1. Why Data Versioning Matters

### 1.1 The Problem with Mutable Data

```python
"""
The Mutable Data Problem:
═════════════════════════

Scenario: Your daily ETL pipeline writes to s3://data-lake/gold/revenue/

Monday:    ETL runs → writes $1.2M revenue ✓
Tuesday:   ETL runs → writes $1.5M revenue ✓ (Monday's data overwritten)
Wednesday: Bug deployed → writes $0 revenue ✗ (Tuesday's data overwritten)
Thursday:  Bug discovered!

Questions you can't answer without versioning:
  1. What did Tuesday's data look like? (GONE — overwritten)
  2. When exactly did the bug corrupt the data? (NO HISTORY)
  3. Can I roll back to Tuesday's version? (MANUAL RESTORE from backup, if exists)
  4. Which downstream dashboards consumed the $0 data? (UNKNOWN)

With versioning (lakeFS, Delta Lake time travel, Iceberg snapshots):
  1. Tuesday's data → checkout commit abc123
  2. Bug introduction → diff between commits shows exact changes
  3. Rollback → revert to commit abc123 (instant, zero-copy)
  4. Lineage → version metadata tracks consumers

Versioning cost: ~1-5% storage overhead (copy-on-write / metadata only)
Versioning value: hours to days saved per incident
"""
```

### 1.2 Use Cases for Data Versioning

| Use Case | Without Versioning | With Versioning |
|----------|-------------------|-----------------|
| **Bug rollback** | Restore from backup (hours) | Revert commit (seconds) |
| **Reproducible ML** | "Which data trained model v2?" | Pinned to commit `abc123` |
| **A/B testing data changes** | Run two pipelines in parallel | Branch, test, merge |
| **Regulatory audit** | Manual export snapshots | Full commit history |
| **Schema migration** | Risky big-bang update | Branch, migrate, validate, merge |
| **Multi-team collaboration** | Conflicts overwrite silently | Merge conflicts detected |

### 1.3 The Versioning Spectrum

Not all versioning approaches are equal. The right choice depends on your scale and use case.

```python
"""
Data Versioning Approaches (from simple to comprehensive):

Level 0: No Versioning
  └── Data is overwritten in place. No history.
  └── Risk: HIGH. Recovery: backup restore (if exists).

Level 1: Timestamp-Based Snapshots
  └── s3://bucket/table/snapshot=2024-01-15/
  └── Full copies per snapshot. Storage: $O(n \times s)$ where $s$ = snapshots
  └── Simple but expensive. No diff capability.

Level 2: Format-Level Versioning (Delta Lake, Iceberg)
  └── Transaction log tracks file-level changes
  └── Time travel via snapshot IDs
  └── Storage: $O(n + \Delta)$ (only changed files stored)
  └── Limited to single-table scope

Level 3: Repository-Level Versioning (lakeFS)
  └── Git-like branches/commits across the ENTIRE data lake
  └── Cross-table atomic commits
  └── Storage: $O(n + \Delta)$ via copy-on-write
  └── Full diff, merge, rollback across all datasets

Level 4: Full Lineage + Versioning
  └── Version data + track how it was produced (pipeline code version)
  └── Tools: lakeFS + Dagster, DVC + Git, Pachyderm
  └── Enables full reproducibility: same code + same data = same result
"""
```

---

## 2. lakeFS: Git for Data Lakes

### 2.1 lakeFS Architecture

lakeFS adds a Git-like version control layer on top of your existing object store (S3, GCS, Azure Blob) without copying data.

```python
"""
lakeFS Architecture:
════════════════════

┌─────────────────────────────────────────────────────────────┐
│                        lakeFS Server                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  S3-Compatible│  │  Version     │  │  Merge       │     │
│  │  API Gateway  │  │  Control     │  │  Engine      │     │
│  │  (read/write) │  │  (commits,   │  │  (3-way diff │     │
│  │               │  │   branches)  │  │   + merge)   │     │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘     │
│         │                  │                                 │
│  ┌──────▼──────────────────▼───────┐                       │
│  │        Metadata Store            │                       │
│  │  (PostgreSQL / DynamoDB)         │                       │
│  │  - Branch pointers               │                       │
│  │  - Commit objects                 │                       │
│  │  - Object deduplication index     │                       │
│  └──────────────┬──────────────────┘                       │
└─────────────────┼───────────────────────────────────────────┘
                  │
    ┌─────────────▼─────────────┐
    │   Underlying Object Store  │
    │   (S3 / GCS / Azure Blob / │
    │    MinIO / local)           │
    │                             │
    │   Actual data files live    │
    │   here, unchanged.          │
    │   lakeFS only manages       │
    │   METADATA (pointers).      │
    └────────────────────────────┘

Key insight: lakeFS does NOT copy data on branch/commit.
It uses copy-on-write semantics:
  - Creating a branch = creating a pointer (instant, ~0 bytes)
  - Writing new data = new files in object store + metadata update
  - Unchanged files = shared between branches (zero duplication)

Storage overhead formula:
  Total storage = Original data + Changed data (delta)
  NOT: Original data × Number of branches
"""
```

### 2.2 Core Operations

```python
import lakefs
from lakefs.client import Client

# Initialize lakeFS client
# Why S3-compatible API? So existing Spark/pandas code works unchanged.
# Just change the endpoint URL from s3.amazonaws.com to lakefs.example.com.

client = Client(
    host="http://localhost:8000",
    username="AKIAIOSFODNN7EXAMPLE",
    password="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
)


# ── Repository Operations ─────────────────────────────────────────

def setup_repository():
    """Create a lakeFS repository backed by S3.

    Why a repository?
      - A repository maps to one storage namespace (e.g., one S3 bucket prefix)
      - All branches, commits, and tags live within a repository
      - Think of it like a Git repository, but for data files
    """
    repo = lakefs.Repository("analytics-lake", client=client)
    repo.create(
        storage_namespace="s3://my-bucket/analytics-lake/",
        default_branch="main",
    )
    print(f"Repository created: {repo.id}")
    return repo


# ── Branching ──────────────────────────────────────────────────────

def create_feature_branch(repo):
    """Create a branch for isolated experimentation.

    Why branch data?
      - Test a new ETL transformation without affecting production
      - Multiple teams can work on different data changes simultaneously
      - Failed experiments are discarded (delete branch), not rolled back
    """
    main = repo.branch("main")
    dev_branch = repo.branch("feature/new-cleaning-logic").create(source_reference="main")
    print(f"Branch created: {dev_branch.id}")

    # At this point, dev_branch has identical data to main
    # No data was copied — just a pointer was created
    return dev_branch


# ── Committing Changes ────────────────────────────────────────────

def write_and_commit(branch):
    """Write data to a branch and commit the changes.

    Why explicit commits?
      - Uncommitted writes are "staged" but not visible to other branches
      - Commits are atomic: all files in a commit succeed or fail together
      - Commit messages document WHY the data changed (just like Git)
    """
    import io
    import pandas as pd

    # Create sample data
    df = pd.DataFrame({
        "order_id": [1, 2, 3],
        "amount": [100.0, 200.0, 300.0],
        "status": ["completed", "completed", "pending"],
    })

    # Upload data to the branch
    # Why Parquet? Columnar format is efficient for analytics queries
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)

    branch.object("orders/daily/2024-01-15.parquet").upload(
        data=buffer,
        content_type="application/octet-stream",
    )

    # Commit the changes
    commit = branch.commit(
        message="Add orders for 2024-01-15",
        metadata={
            "pipeline": "daily_order_ingestion",
            "source": "postgres_replica",
            "row_count": "3",
        },
    )
    print(f"Committed: {commit.id} - {commit.message}")
    return commit


# ── Diffing and Comparing ─────────────────────────────────────────

def compare_branches(repo):
    """Compare data between two branches.

    Why diff before merge?
      - See exactly what changed (new files, modified files, deleted files)
      - Catch unintended changes before they reach production
      - Review data changes like code reviews (data PRs)
    """
    main = repo.branch("main")
    feature = repo.branch("feature/new-cleaning-logic")

    diff = main.diff(other_ref=feature)

    for change in diff:
        # change.type: 'added', 'removed', 'changed'
        # change.path: the object key
        print(f"  [{change.type}] {change.path}")

    return diff


# ── Merging ────────────────────────────────────────────────────────

def merge_to_main(repo):
    """Merge a feature branch into main after validation.

    Why merge (not overwrite)?
      - Preserves commit history on main
      - Detects conflicts if main was updated since branch creation
      - Atomic: entire merge succeeds or fails (no partial updates)
    """
    feature = repo.branch("feature/new-cleaning-logic")
    main = repo.branch("main")

    try:
        merge_result = feature.merge_into(main)
        print(f"Merge successful: {merge_result}")
    except lakefs.exceptions.ConflictException as e:
        # Why handle conflicts?
        # If someone updated the same files on main since we branched,
        # lakeFS detects the conflict (just like Git).
        print(f"Merge conflict: {e}")
        print("Resolve manually: update the conflicting files on the feature branch")


# ── Rollback ───────────────────────────────────────────────────────

def rollback_to_commit(repo, commit_id: str):
    """Roll back the main branch to a specific commit.

    Why rollback instead of fixing forward?
      - Instant recovery: production data is restored in seconds
      - Fix-forward takes time: investigate, fix, re-run pipeline
      - Rollback buys you time to investigate while users see correct data
    """
    main = repo.branch("main")
    main.revert(parent_number=1, reference=commit_id)
    print(f"Rolled back main to commit {commit_id}")
```

### 2.3 lakeFS with Spark Integration

```python
"""
lakeFS + Spark Integration:

The beauty of lakeFS is that Spark reads/writes using the standard S3 API.
You just change the endpoint URL. No code changes needed!
"""

from pyspark.sql import SparkSession

# Pointing Spark's S3 endpoint to lakeFS means zero code changes to existing
# Spark jobs — only the endpoint URL changes. This is why lakeFS chose S3
# compatibility: it eliminates adoption friction for teams already using S3.
# path.style.access=true is required because lakeFS doesn't support
# virtual-hosted-style URLs (bucket.endpoint) used by real S3.

spark = SparkSession.builder \
    .appName("lakeFS-Spark") \
    .config("spark.hadoop.fs.s3a.endpoint", "http://lakefs:8000") \
    .config("spark.hadoop.fs.s3a.access.key", "AKIAIOSFODNN7EXAMPLE") \
    .config("spark.hadoop.fs.s3a.secret.key", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY") \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .getOrCreate()

# URI format: s3a://<repository>/<branch>/path/to/data
# Reading from "main" always gives the latest committed state.
main_orders = spark.read.parquet("s3a://analytics-lake/main/orders/")

# Branch names with slashes use ~ as separator in URIs (feature~new-cleaning-logic
# represents the branch "feature/new-cleaning-logic"). This avoids ambiguity
# with the object path that follows.
feature_orders = spark.read.parquet("s3a://analytics-lake/feature~new-cleaning-logic/orders/")

# Referencing a commit hash instead of a branch gives immutable access —
# even if the branch advances, this always reads the exact same data.
# Essential for reproducible ML training and regulatory audits.
historical = spark.read.parquet("s3a://analytics-lake/abc123def/orders/")

# Writes to a branch are "staged" (visible only on that branch) until
# explicitly committed via the lakeFS API. This means a failed Spark job
# leaves no partially-written data visible to other users.
result_df = main_orders.filter("amount > 100")
result_df.write.mode("overwrite").parquet(
    "s3a://analytics-lake/feature~new-cleaning-logic/orders_filtered/"
)
# Then commit via lakeFS API (see above)
```

---

## 3. DVC vs lakeFS Comparison

Both DVC (Data Version Control) and lakeFS version data, but they target different use cases.

### 3.1 Architecture Comparison

```python
"""
DVC (Data Version Control):
════════════════════════════
- Git extension: .dvc files in Git track data file hashes
- Data lives in remote storage (S3, GCS, NFS)
- Versioning = Git commits containing .dvc pointer files
- Best for: ML experiments, small teams, model/dataset versioning

  Git repo:                Remote storage:
  ┌──────────────┐        ┌──────────────┐
  │ data.csv.dvc │──hash─→│ data.csv     │
  │ model.pkl.dvc│──hash─→│ model.pkl    │
  │ pipeline.py  │        │              │
  └──────────────┘        └──────────────┘

lakeFS:
═══════
- Standalone server: manages object store directly
- No Git dependency (has its own commit/branch model)
- S3-compatible API: transparent to Spark/Presto/Trino
- Best for: data lake versioning, multi-team, production pipelines

  lakeFS server:           Object store:
  ┌──────────────┐        ┌──────────────┐
  │ branches     │──meta─→│ actual data   │
  │ commits      │        │ files (shared │
  │ merge engine │        │ via CoW)      │
  └──────────────┘        └──────────────┘
"""
```

### 3.2 Feature Matrix

| Feature | DVC | lakeFS |
|---------|-----|--------|
| **Primary use case** | ML experiments | Data lake management |
| **Version model** | Git-based (.dvc files) | Standalone (own branches/commits) |
| **Branching** | Via Git branches | Native, instant (zero-copy) |
| **API** | CLI + Python | S3-compatible + REST + Python |
| **Spark/Trino integration** | Manual path management | Transparent (S3 API) |
| **Atomic multi-file commits** | Yes (via Git) | Yes (native) |
| **Merge conflicts** | File-level (via Git) | Object-level (3-way merge) |
| **Pipeline tracking** | `dvc.yaml` pipelines | External (Dagster/Airflow) |
| **Data diffing** | Limited (hash comparison) | Rich (file-level + optional content) |
| **Scale** | Hundreds of files | Millions of objects |
| **Team size** | Small (1-10) | Any (enterprise-ready) |
| **Deployment** | None (client-only) | Server required |
| **Cost** | Free (OSS) | Free OSS / lakeFS Cloud |

### 3.3 Decision Guide

```python
"""
Choose DVC when:
  ✓ Your team already uses Git heavily
  ✓ You're versioning ML datasets and models
  ✓ Dataset count is in the hundreds (not millions)
  ✓ You want pipeline reproducibility (dvc repro)
  ✓ You don't need real-time branch switching for Spark/Trino

Choose lakeFS when:
  ✓ You have a data lake with millions of objects
  ✓ Multiple teams read/write to shared storage
  ✓ You need Spark/Trino/Presto to read versioned data transparently
  ✓ You want atomic cross-table commits
  ✓ You need instant branching for CI/CD testing of data pipelines
  ✓ You want to enforce pre-merge hooks (data quality gates)

Choose both (DVC for ML, lakeFS for data lake):
  ✓ ML team uses DVC for experiment tracking + model versioning
  ✓ Data engineering team uses lakeFS for the shared data lake
  ✓ ML training reads from lakeFS via Spark, versions models with DVC
"""
```

---

## 4. Data Contracts

### 4.1 The Problem: Implicit Interfaces

In a typical data platform, the interface between data producers and consumers is implicit — defined only by whatever the pipeline happens to produce today.

```python
"""
The Implicit Interface Problem:
═══════════════════════════════

Team A (Producer):                    Team B (Consumer):
┌───────────────────┐                ┌───────────────────┐
│ Writes orders table│                │ Reads orders table │
│ Columns:           │                │ Expects:           │
│  - order_id (int)  │──── table ───→│  - order_id (int)  │
│  - amount (float)  │               │  - amount (float)  │
│  - status (string) │               │  - status (string) │
└───────────────────┘                └───────────────────┘

Day 1: Everything works ✓

Day 30: Team A renames "amount" → "total_amount"
  - Team A: "We improved the column name!"
  - Team B: pipeline fails with KeyError: 'amount'
  - Dashboard shows no data for 6 hours until someone notices

Day 60: Team A adds nullable "discount" column
  - Team B's aggregation returns NaN for all metrics
  - Root cause takes 3 hours to diagnose

Root cause: NO CONTRACT defines what Team A promises to deliver.
"""
```

### 4.2 What is a Data Contract?

A data contract is a formal agreement between a data producer and its consumers that specifies:

```python
"""
Data Contract Components:

┌─────────────────────────────────────────────────────────────┐
│                       DATA CONTRACT                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. SCHEMA                                                  │
│     - Column names, data types, nullability                 │
│     - Primary keys, foreign keys                            │
│     - Valid value ranges and enums                          │
│                                                             │
│  2. SEMANTICS                                               │
│     - Business meaning of each field                        │
│     - Calculation logic (e.g., "amount = quantity × price") │
│     - Temporal semantics (event time vs processing time)    │
│                                                             │
│  3. SLA (Service Level Agreement)                           │
│     - Freshness: "updated within 1 hour of source change"  │
│     - Completeness: ">99% of source records present"       │
│     - Availability: "queryable 99.9% of the time"          │
│                                                             │
│  4. EVOLUTION POLICY                                        │
│     - How changes are proposed and reviewed                 │
│     - Backward compatibility requirements                   │
│     - Deprecation timelines                                 │
│                                                             │
│  5. OWNERSHIP                                               │
│     - Producer team and contact                             │
│     - Consumer teams (registered)                           │
│     - Escalation path for violations                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
"""
```

### 4.3 Schema Contracts with Different Formats

```python
# ── Approach 1: Pydantic Models (Python-native) ───────────────────
# Why Pydantic? Type-safe, auto-validation, great error messages.
# Best for: Python pipelines, FastAPI data services.

from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from enum import Enum
from typing import Optional


class OrderStatus(str, Enum):
    """Valid order statuses — the contract guarantees only these values."""
    PENDING = "pending"
    COMPLETED = "completed"
    REFUNDED = "refunded"
    CANCELLED = "cancelled"


class OrderRecord(BaseModel):
    """Schema contract for the orders dataset.

    This Pydantic model IS the contract. If data doesn't match,
    validation fails with a clear error message.

    Version: 2.0
    Owner: data-platform-team
    Consumers: analytics, ML, finance
    """
    order_id: int = Field(..., gt=0, description="Unique order identifier")
    customer_id: int = Field(..., gt=0, description="Customer FK")
    amount: float = Field(..., ge=0, description="Order total in USD")
    currency: str = Field(default="USD", pattern=r"^[A-Z]{3}$")
    status: OrderStatus
    created_at: datetime
    updated_at: Optional[datetime] = None

    # Why field validators? They encode BUSINESS RULES into the contract.
    # The schema guarantees not just types, but semantic correctness.
    @field_validator("amount")
    @classmethod
    def amount_must_be_reasonable(cls, v: float) -> float:
        """Business rule: orders above $1M require manual review."""
        if v > 1_000_000:
            raise ValueError(f"Amount ${v:,.2f} exceeds $1M limit — needs review")
        return round(v, 2)


def validate_dataframe(df, model_class):
    """Validate every row of a DataFrame against a Pydantic contract.

    Why validate at pipeline boundaries?
      - Catch contract violations EARLY (at ingestion, not in dashboards)
      - Clear error messages: which row, which field, what's wrong
      - Fail fast: better a pipeline failure than corrupt data downstream
    """
    errors = []
    valid_records = []

    for idx, row in df.iterrows():
        try:
            record = model_class(**row.to_dict())
            valid_records.append(record.model_dump())
        except Exception as e:
            errors.append({"row": idx, "error": str(e)})

    if errors:
        error_rate = len(errors) / len(df)
        print(f"Validation: {len(errors)}/{len(df)} rows failed ({error_rate:.1%})")
        # Why a threshold? Some bad records are expected (data quality)
        # But if >5% fail, something systematic is wrong
        if error_rate > 0.05:
            raise ValueError(
                f"Contract violation: {error_rate:.1%} error rate exceeds 5% threshold"
            )

    return valid_records, errors
```

```python
# ── Approach 2: JSON Schema (Language-Agnostic) ───────────────────
# Why JSON Schema? Works across Python, Java, Go, JavaScript.
# Best for: cross-team contracts, API boundaries, schema registries.

import json
from jsonschema import validate, ValidationError

# JSON Schema contracts are language-agnostic — the same schema file can be
# validated by Python, Java, Go, or JavaScript consumers. This makes JSON
# Schema ideal for cross-team contracts where producer and consumer use
# different languages.
ORDER_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "OrderRecord",
    "description": "Schema contract for the orders dataset (v2.0)",
    "type": "object",
    # "required" lists fields that MUST be present in every record.
    # Fields NOT listed here are optional — they may or may not appear.
    # Keep the required list minimal to maximize backward compatibility.
    "required": ["order_id", "customer_id", "amount", "status", "created_at"],
    "properties": {
        "order_id": {
            "type": "integer",
            "minimum": 1,
            "description": "Unique order identifier",
        },
        "customer_id": {
            "type": "integer",
            "minimum": 1,
            "description": "Customer foreign key",
        },
        "amount": {
            "type": "number",
            "minimum": 0,
            "maximum": 1000000,
            "description": "Order total in USD",
        },
        "status": {
            "type": "string",
            "enum": ["pending", "completed", "refunded", "cancelled"],
        },
        "created_at": {
            "type": "string",
            "format": "date-time",
        },
    },
    # additionalProperties: False is the strict mode — it rejects records
    # containing unexpected columns. This prevents "schema drift" where
    # new columns appear silently and downstream consumers unknowingly
    # depend on undocumented fields. Set to True during migration periods
    # when the producer is adding new columns incrementally.
    "additionalProperties": False,
}


def validate_record_json_schema(record: dict) -> bool:
    """Validate a single record against the JSON Schema contract."""
    try:
        validate(instance=record, schema=ORDER_SCHEMA)
        return True
    except ValidationError as e:
        print(f"Contract violation: {e.message}")
        print(f"  Path: {'.'.join(str(p) for p in e.absolute_path)}")
        return False
```

```python
# ── Approach 3: Avro Schema (Streaming / Kafka) ──────────────────
# Why Avro? Native schema evolution, compact binary format, Kafka standard.
# Best for: Kafka topics, schema registry integration, cross-language streaming.

# Avro's compact binary serialization reduces Kafka message size by 50-80%
# versus JSON. The schema is registered once in the Schema Registry, and
# only a 4-byte schema ID is transmitted with each message.
AVRO_SCHEMA = {
    "type": "record",
    "name": "OrderRecord",
    # Namespace prevents name collisions when multiple teams publish schemas
    # to the same registry — "com.company.analytics.OrderRecord" is unique
    # even if the payments team also has an "OrderRecord."
    "namespace": "com.company.analytics",
    "doc": "Schema contract for order events (v2.0)",
    "fields": [
        {"name": "order_id", "type": "long", "doc": "Unique order identifier"},
        {"name": "customer_id", "type": "long", "doc": "Customer FK"},
        {"name": "amount", "type": "double", "doc": "Order total in USD"},
        {
            "name": "status",
            # Avro enum is more restrictive than a string — the reader rejects
            # any value not in "symbols." This catches invalid status values
            # at serialization time rather than downstream.
            "type": {
                "type": "enum",
                "name": "OrderStatus",
                "symbols": ["PENDING", "COMPLETED", "REFUNDED", "CANCELLED"],
            },
        },
        {"name": "created_at", "type": "long", "logicalType": "timestamp-millis"},
        {
            "name": "updated_at",
            # Union type ["null", "long"] makes this field nullable. The null
            # must appear FIRST in the union for the default=None to be valid.
            # This is an Avro-specific ordering requirement.
            "type": ["null", "long"],
            "default": None,
            "logicalType": "timestamp-millis",
        },
    ],
}

"""
Avro Schema Evolution Rules:
─────────────────────────────
BACKWARD compatible changes (consumers can read new data with old schema):
  ✓ Add a field with a default value
  ✓ Remove a field that had a default value
  ✗ Rename a field (breaking!)
  ✗ Change a field's type (breaking!)

FORWARD compatible changes (consumers with new schema can read old data):
  ✓ Add a field with a default value
  ✓ Remove a field
  ✗ Add a required field without default (breaking!)

FULL compatible = both backward AND forward compatible.

Why does this matter?
  - Kafka consumers may be running different code versions
  - A producer update should NOT require simultaneous consumer updates
  - Schema Registry enforces compatibility at write time
"""
```

---

## 5. Contract Testing and Enforcement

### 5.1 Contract Validation in Pipelines

```python
"""
Where to enforce contracts in a pipeline:

  Producer                    Contract Gate              Consumer
  ┌────────┐    ┌─────────────────────────────┐    ┌────────┐
  │ Source  │───→│ 1. Schema validation        │───→│ Target │
  │ system  │    │ 2. Semantic checks          │    │ system │
  │         │    │ 3. Freshness verification   │    │        │
  └────────┘    │ 4. Completeness check       │    └────────┘
                │                             │
                │ PASS → data flows through   │
                │ FAIL → alert + quarantine   │
                └─────────────────────────────┘

Why validate at the boundary (not inside the consumer)?
  - Fail fast: catch problems before data spreads
  - Single point of enforcement: one gate, many consumers
  - Clear responsibility: producer fixes violations
"""
```

### 5.2 Great Expectations Integration

```python
import great_expectations as gx

# Why Great Expectations?
# - Declarative data quality checks (expectations)
# - Rich built-in expectations (200+)
# - HTML data docs for stakeholder communication
# - Integrates with Airflow, Dagster, dbt

def create_order_contract_suite():
    """Define contract expectations for the orders dataset.

    These expectations encode the contract as executable tests.
    Each expectation maps to a contract clause.
    """
    context = gx.get_context()

    # Create a data source and batch
    datasource = context.data_sources.add_pandas("orders_source")

    # An expectation suite groups all contract checks for one dataset.
    # Versioning the suite name (v2) lets you run old and new contracts
    # in parallel during migration periods.
    suite = context.suites.add_expectation_suite("orders_contract_v2")

    # Column ordering check catches silent schema drift — if the producer
    # reorders or renames columns, this expectation fails immediately rather
    # than letting downstream queries read the wrong column by position.
    suite.add_expectation(
        gx.expectations.ExpectTableColumnsToMatchOrderedList(
            column_list=[
                "order_id", "customer_id", "amount",
                "status", "created_at", "updated_at",
            ]
        )
    )

    # Type expectations
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeOfType(
            column="amount", type_="float64"
        )
    )

    # mostly=0.99 allows up to 1% of rows to fall outside the range.
    # This prevents a single outlier from failing the entire validation,
    # while still catching systematic data quality issues.
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="amount", min_value=0, max_value=1_000_000,
            mostly=0.99,
        )
    )

    # Enum expectations
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="status",
            value_set=["pending", "completed", "refunded", "cancelled"],
        )
    )

    # Uniqueness expectations
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeUnique(column="order_id")
    )

    # Different "mostly" thresholds reflect different business criticality:
    # order_id at 1.0 (zero tolerance) because it's the primary key,
    # amount at 0.99 because a small fraction of null amounts may be
    # legitimate (e.g., free promotional orders).
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="order_id", mostly=1.0,
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="amount", mostly=0.99,
        )
    )

    return suite


def run_contract_validation(df, suite_name: str = "orders_contract_v2"):
    """Run contract validation against a DataFrame.

    Returns a validation result that can be:
    - Checked programmatically (pass/fail)
    - Rendered as HTML data docs (for stakeholders)
    - Sent as an alert (via webhook/email)
    """
    context = gx.get_context()
    # Checkpoints combine a data source, expectation suite, and validation
    # actions (store results, send alerts) into a single reusable unit.
    # This decouples "what to validate" from "when/how to validate."
    result = context.run_checkpoint(
        checkpoint_name="orders_checkpoint",
        batch_request={
            "datasource_name": "orders_source",
            "data_asset_name": "orders",
            # batch_data accepts an in-memory DataFrame — no need to write
            # to disk first. This enables contract checks inside streaming
            # foreachBatch handlers or CI/CD test pipelines.
            "batch_data": df,
        },
    )

    if not result.success:
        failed = [
            r for r in result.results
            if not r.success
        ]
        print(f"CONTRACT VIOLATION: {len(failed)} expectations failed")
        for f in failed:
            print(f"  - {f.expectation_config.expectation_type}: {f.result}")
        # Raising an exception halts the pipeline, preventing contract-violating
        # data from propagating downstream. This is the "fail-fast" pattern:
        # better a pipeline failure than silently corrupted dashboards.
        raise ContractViolationError(f"{len(failed)} contract expectations failed")

    print("Contract validation PASSED")
    return result
```

### 5.3 Soda Integration

```python
# Soda provides a YAML-based contract definition language
# Why Soda? Simpler syntax, built for data contracts specifically

"""
# soda_contract.yaml — Data contract definition

dataset: orders
owner: data-platform-team
version: 2.0

schema:
  # Strict schema check — fails if columns don't match exactly
  fail:
    when mismatching columns:
      - order_id: integer
      - customer_id: integer
      - amount: float
      - status: string
      - created_at: timestamp
      - updated_at: timestamp

checks:
  # Uniqueness
  - duplicate_count(order_id) = 0

  # Completeness
  - missing_count(order_id) = 0
  - missing_percent(amount) < 1%

  # Value ranges
  - min(amount) >= 0
  - max(amount) < 1000000

  # Enum values
  - invalid_count(status) = 0:
      valid values: [pending, completed, refunded, cancelled]

  # Freshness (data should be < 2 hours old)
  - freshness(created_at) < 2h

  # Row count (anomaly detection)
  - row_count > 0
  - anomaly score for row_count < 3  # Alert if 3+ std devs from normal
"""
```

---

## 6. Building a Contract-First Pipeline

### 6.1 The Contract-First Approach

Instead of defining contracts after building pipelines, the contract-first approach defines contracts first and builds pipelines to satisfy them.

```python
"""
Contract-First Development Cycle:

1. DEFINE the contract (producer + consumer agree)
   ├── Schema: columns, types, constraints
   ├── SLA: freshness, completeness, availability
   └── Evolution: how changes are proposed

2. IMPLEMENT the producer pipeline
   └── Must produce data matching the contract

3. VALIDATE at every pipeline run
   └── Contract checks run as a pipeline step

4. EVOLVE the contract when business needs change
   ├── Propose change (PR/RFC)
   ├── Assess consumer impact
   ├── Implement with backward compatibility
   └── Deprecate old version after transition period

This is analogous to API-first development:
  - API-first: define OpenAPI spec → implement endpoints → test against spec
  - Contract-first: define data contract → implement pipeline → validate against contract
"""
```

### 6.2 Contract-First Pipeline Implementation

```python
import dagster as dg
import pandas as pd
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# Step 1: Define the contract FIRST
class OrderContract(BaseModel):
    """The ORDER contract — agreed upon by producer and consumers.

    This model serves as:
    - Documentation (field descriptions)
    - Validation logic (type checks, constraints)
    - Schema evolution tracker (version field)
    """
    class Config:
        json_schema_extra = {
            "version": "2.0",
            "owner": "data-platform-team",
            "consumers": ["analytics", "ml-team", "finance"],
            "sla": {
                "freshness": "1 hour",
                "completeness": "99%",
                "availability": "99.9%",
            },
        }

    order_id: int = Field(..., gt=0)
    customer_id: int = Field(..., gt=0)
    amount: float = Field(..., ge=0, le=1_000_000)
    status: str = Field(..., pattern=r"^(pending|completed|refunded|cancelled)$")
    created_at: datetime
    updated_at: Optional[datetime] = None


# Step 2: Contract validation as a Dagster asset check
@dg.asset_check(asset=dg.AssetKey("cleaned_orders"))
def orders_contract_check(context: dg.AssetCheckExecutionContext, cleaned_orders: pd.DataFrame):
    """Validate cleaned_orders against the OrderContract.

    Why an asset check (not just inline validation)?
      - Visible in Dagster UI: green check = contract met
      - Historical tracking: see when contracts started failing
      - Blocking: can prevent downstream materialization on failure
    """
    errors = []
    for idx, row in cleaned_orders.iterrows():
        try:
            OrderContract(**row.to_dict())
        except Exception as e:
            errors.append({"row": idx, "error": str(e)})

    error_rate = len(errors) / len(cleaned_orders) if len(cleaned_orders) > 0 else 0

    if error_rate > 0.01:  # >1% error rate = contract violation
        yield dg.AssetCheckResult(
            passed=False,
            metadata={
                "error_count": len(errors),
                "error_rate": f"{error_rate:.2%}",
                "sample_errors": str(errors[:5]),
            },
        )
    else:
        yield dg.AssetCheckResult(
            passed=True,
            metadata={
                "validated_rows": len(cleaned_orders),
                "error_count": len(errors),
                "contract_version": "2.0",
            },
        )
```

---

## 7. Schema Evolution

### 7.1 Managing Breaking vs Non-Breaking Changes

Schema evolution is inevitable. The key is distinguishing changes that break consumers from those that don't.

```python
"""
Schema Evolution Classification:
═════════════════════════════════

NON-BREAKING (safe to deploy without consumer coordination):
  ✓ Add an OPTIONAL column with a default value
  ✓ Widen a type (int32 → int64, float32 → float64)
  ✓ Add a new enum value (if consumers use 'default' handling)
  ✓ Increase a max length constraint
  ✓ Relax a NOT NULL to nullable

BREAKING (requires consumer coordination):
  ✗ Remove a column
  ✗ Rename a column
  ✗ Narrow a type (int64 → int32)
  ✗ Add a NOT NULL column without default
  ✗ Change a column's semantic meaning
  ✗ Remove an enum value

Migration Strategy for Breaking Changes:
  Phase 1: Add new column alongside old (both populated)
  Phase 2: Consumers migrate to new column (tracked via contract registry)
  Phase 3: Old column deprecated (warning period: 30-90 days)
  Phase 4: Old column removed (after all consumers migrated)

Timeline: $T_{migration} \approx 30 + N_{consumers} \times 7$ days
  where $N_{consumers}$ is the number of consuming teams
"""
```

### 7.2 Schema Evolution Implementation

```python
from dataclasses import dataclass, field
from typing import Any
from enum import Enum


class ChangeType(Enum):
    ADD_COLUMN = "add_column"
    REMOVE_COLUMN = "remove_column"
    RENAME_COLUMN = "rename_column"
    CHANGE_TYPE = "change_type"
    ADD_CONSTRAINT = "add_constraint"
    REMOVE_CONSTRAINT = "remove_constraint"


@dataclass
class SchemaChange:
    """Represents a proposed change to a data contract."""
    change_type: ChangeType
    column_name: str
    details: dict = field(default_factory=dict)
    is_breaking: bool = False
    migration_plan: str = ""


def assess_schema_changes(
    old_schema: dict,
    new_schema: dict,
) -> list[SchemaChange]:
    """Compare two schema versions and classify each change.

    Why automated assessment?
      - Humans miss breaking changes in large schemas
      - Consistent classification across all contract updates
      - Feeds into CI/CD gates (block breaking changes without approval)
    """
    changes = []

    old_cols = set(old_schema.get("required", []))
    new_cols = set(new_schema.get("required", []))

    old_props = old_schema.get("properties", {})
    new_props = new_schema.get("properties", {})

    # Column removal is ALWAYS breaking — even if no consumer currently uses
    # the column, removing it prevents any consumer from reading old data
    # that includes the column (schema mismatch in Avro/Parquet readers).
    for col in old_props:
        if col not in new_props:
            changes.append(SchemaChange(
                change_type=ChangeType.REMOVE_COLUMN,
                column_name=col,
                is_breaking=True,
                migration_plan=f"Phase out '{col}' over 30 days. "
                              f"Notify consumers: analytics, ML, finance.",
            ))

    # Adding an OPTIONAL column is non-breaking because existing consumers
    # simply ignore it. But a REQUIRED column without a default forces all
    # consumers to update their code before they can read the new schema.
    for col in new_props:
        if col not in old_props:
            is_required = col in new_cols
            changes.append(SchemaChange(
                change_type=ChangeType.ADD_COLUMN,
                column_name=col,
                details={"required": is_required, "schema": new_props[col]},
                is_breaking=is_required and "default" not in new_props[col],
                migration_plan="" if not is_required else
                    f"Add '{col}' with default value first, "
                    f"then make required after consumers adopt.",
            ))

    # Type changes are breaking even when widening (int32 -> int64) because
    # downstream consumers may have hardcoded the original type in their
    # Pydantic models, Spark schemas, or database DDL. The safe pattern
    # is a parallel column: keep the old column, add a new typed column,
    # then deprecate the old one after all consumers migrate.
    for col in old_props:
        if col in new_props:
            old_type = old_props[col].get("type")
            new_type = new_props[col].get("type")
            if old_type != new_type:
                changes.append(SchemaChange(
                    change_type=ChangeType.CHANGE_TYPE,
                    column_name=col,
                    details={"old_type": old_type, "new_type": new_type},
                    is_breaking=True,
                    migration_plan=f"Add '{col}_v2' ({new_type}) alongside "
                                  f"'{col}' ({old_type}). Deprecate after migration.",
                ))

    return changes


# Usage:
# changes = assess_schema_changes(ORDER_SCHEMA_V1, ORDER_SCHEMA_V2)
# breaking = [c for c in changes if c.is_breaking]
# if breaking:
#     print(f"BLOCKED: {len(breaking)} breaking changes require consumer approval")
```

---

## 8. Data Mesh and Domain Ownership

### 8.1 Data Mesh Principles

Data contracts are a cornerstone of the **data mesh** architecture, where data ownership is decentralized to domain teams.

```python
"""
Data Mesh Architecture:
═══════════════════════

Traditional (Centralized):
  ┌───────────┐    ┌────────────────┐    ┌───────────┐
  │ Team A    │───→│ Central Data   │───→│ Team C    │
  │ (source)  │    │ Team           │    │ (consumer)│
  └───────────┘    │                │    └───────────┘
  ┌───────────┐    │ - Owns ALL     │    ┌───────────┐
  │ Team B    │───→│   pipelines    │───→│ Team D    │
  │ (source)  │    │ - Bottleneck   │    │ (consumer)│
  └───────────┘    └────────────────┘    └───────────┘

  Problem: Central team = bottleneck. Doesn't scale with org growth.

Data Mesh (Decentralized):
  ┌───────────────────┐        ┌───────────────────┐
  │ Orders Domain     │        │ Products Domain    │
  │ ┌──────────────┐  │        │ ┌──────────────┐  │
  │ │ Data Product: │  │        │ │ Data Product: │  │
  │ │ orders_clean  │──┼────────┼→│ product_catalog│ │
  │ │ CONTRACT: v2  │  │        │ │ CONTRACT: v3  │  │
  │ └──────────────┘  │        │ └──────────────┘  │
  └───────────────────┘        └───────────────────┘
          │                            │
          └──────────┬─────────────────┘
                     ▼
  ┌───────────────────────────┐
  │ Analytics Domain          │
  │ ┌──────────────────────┐  │
  │ │ Data Product:        │  │
  │ │ revenue_dashboard    │  │
  │ │ Consumes: orders v2  │  │
  │ │           products v3│  │
  │ └──────────────────────┘  │
  └───────────────────────────┘

Four Principles of Data Mesh:
  1. Domain Ownership: Each domain owns its data pipelines and contracts
  2. Data as a Product: Treat data outputs as products with SLAs
  3. Self-Serve Platform: Shared infrastructure for storage, compute, contracts
  4. Federated Governance: Shared standards (naming, security, compliance)
"""
```

### 8.2 Data Products and Contracts

```python
"""
A Data Product = Data + Contract + Metadata + SLA

Example: Orders Domain publishes the "orders_clean" data product

  ┌────────────────────────────────────────────┐
  │ Data Product: orders_clean                 │
  ├────────────────────────────────────────────┤
  │ Owner: Orders Team (@orders-eng)           │
  │ Contract version: 2.0                      │
  │ Schema: OrderContract (Pydantic model)     │
  │ Location: s3://data-lake/gold/orders_clean │
  │ Format: Parquet (partitioned by date)      │
  │ Freshness SLA: < 1 hour from source       │
  │ Completeness SLA: > 99%                   │
  │ Availability: 99.9%                        │
  │ Access: self-service via data catalog      │
  │ Consumers: analytics, ML, finance          │
  │                                            │
  │ Change policy:                             │
  │   Non-breaking: deploy freely              │
  │   Breaking: 30-day notice + RFC            │
  └────────────────────────────────────────────┘

The contract is the API of the data product.
Without it, data mesh degenerates into data chaos.
"""
```

---

## 9. Versioned ML Datasets

### 9.1 Reproducible Training with Versioned Data

One of the most impactful applications of data versioning is ML reproducibility.

```python
"""
The ML Reproducibility Problem:
═══════════════════════════════

You trained model_v3 last month. It performed great.
Today you need to retrain with the same data + new features.
But the training data has changed (new records, corrections, schema updates).

Without versioning:
  - "Which exact data trained model_v3?" → Unknown
  - "Can I reproduce model_v3's results?" → No
  - "What changed between model_v3 and model_v4's training data?" → Manual diff

With versioning (lakeFS + DVC):
  - model_v3 trained on lakeFS commit abc123 (or DVC tag v3-data)
  - Reproduce: checkout that exact version, retrain → identical results
  - Compare: diff commits abc123 vs def456 → see data changes
  - Audit: full lineage from raw data → training set → model → predictions
"""

# ── lakeFS approach: Pin training data to a commit ────────────────

def train_model_with_versioned_data(
    lakefs_repo: str,
    commit_id: str,
    model_version: str,
):
    """Train an ML model using a pinned data version.

    Why pin to a commit (not a branch)?
      - Branches move (new commits are added)
      - Commits are immutable (same data forever)
      - Reproducible: same commit = same training data = same model
    """
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.appName("ML-Training").getOrCreate()

    # Reading from a commit hash (not a branch) guarantees immutability:
    # even if new data is added to the branch tomorrow, this training run
    # reads exactly the same files. This is the foundation of ML reproducibility.
    training_data = spark.read.parquet(
        f"s3a://{lakefs_repo}/{commit_id}/ml/training/features/"
    )

    # Logging the data version in MLflow creates a bidirectional link:
    # from model → data (which data trained this model?) and from
    # data → model (which models were trained on this data version?).
    # Without this link, debugging model quality regressions is guesswork.
    import mlflow
    with mlflow.start_run():
        mlflow.log_param("data_lakefs_repo", lakefs_repo)
        mlflow.log_param("data_lakefs_commit", commit_id)
        mlflow.log_param("model_version", model_version)
        # Row count serves as a quick sanity check — if it differs
        # significantly from the expected count, the data version may
        # be wrong or the feature pipeline may have a bug.
        mlflow.log_metric("training_rows", training_data.count())

        # Train model...
        # model = train(training_data)
        # mlflow.sklearn.log_model(model, "model")


# ── DVC approach: Tag dataset versions in Git ─────────────────────

"""
# DVC workflow for ML data versioning:

# 1. Track a dataset with DVC
$ dvc add data/training_features.parquet
# Creates: data/training_features.parquet.dvc (pointer file in Git)

# 2. Commit the pointer to Git
$ git add data/training_features.parquet.dvc
$ git commit -m "Training data v3: added click features"
$ git tag data-v3

# 3. Train model, referencing the data version
$ python train.py --data-version data-v3

# 4. Later, reproduce:
$ git checkout data-v3
$ dvc checkout          # Downloads the exact data version
$ python train.py       # Identical results

# 5. Compare data versions:
$ dvc diff data-v2 data-v3
#   Modified: data/training_features.parquet
#   +5000 rows, +2 columns (click_count, session_duration)
"""
```

### 9.2 Dataset Lineage

```python
"""
Complete ML Lineage with Versioning:

  Raw Data             Feature Store          Model Registry
  (lakeFS)             (lakeFS/DVC)           (MLflow)
  ┌─────────┐         ┌─────────────┐        ┌─────────────┐
  │commit a1 │────────→│commit f1    │───────→│ model_v1    │
  │(raw data)│ ETL +   │(features v1)│ Train  │ metric: 0.85│
  └─────────┘ Feature  └─────────────┘        └─────────────┘
              Eng.

  ┌─────────┐         ┌─────────────┐        ┌─────────────┐
  │commit a2 │────────→│commit f2    │───────→│ model_v2    │
  │(+new data)│ ETL + │(features v2)│ Train  │ metric: 0.87│
  └─────────┘ Feature  └─────────────┘        └─────────────┘
              Eng.     (+click feats)

Full traceability:
  model_v2 → trained on features f2 → derived from raw data a2
  Diff f1→f2: +2 columns (click_count, session_duration)
  Diff a1→a2: +50K new rows from January
"""
```

---

## Summary

```
Data Versioning Key Concepts:
─────────────────────────────
lakeFS          = Git-like versioning for data lakes (branches, commits, merges)
DVC             = Git extension for versioning datasets and ML models
Copy-on-Write   = Only store changes, not full copies (efficient branching)
Time Travel     = Query historical data versions (also available in Delta/Iceberg)
Atomic Commits  = All-or-nothing changes to multiple files

Data Contracts Key Concepts:
────────────────────────────
Schema Contract = Formal definition of columns, types, constraints
Semantic Contract = Business meaning and calculation logic
SLA             = Freshness, completeness, availability guarantees
Evolution Policy = How breaking vs non-breaking changes are managed
Contract Testing = Automated validation at pipeline boundaries

Tools for Contracts:
  Pydantic     → Python-native, great for Python pipelines
  JSON Schema  → Language-agnostic, API boundaries
  Avro         → Streaming/Kafka, built-in evolution
  Great Expectations → Rich expectation library, data docs
  Soda         → YAML-based, purpose-built for contracts

Data Mesh Connection:
  Data Product = Data + Contract + SLA + Ownership
  Domain teams own their data products and contracts
  Contracts enable decentralized, self-serve data consumption
```

---

## Exercises

### Exercise 1: lakeFS Branch-Merge Workflow

Simulate a data pipeline change using lakeFS:

1. Create a lakeFS repository (use the Docker quickstart or mock the API)
2. Write initial order data to the `main` branch and commit
3. Create a `feature/add-discount-column` branch
4. Add a `discount` column to the order data on the feature branch
5. Compare the branches using the diff API
6. Merge the feature branch into `main`
7. Verify the merged data has the new column

### Exercise 2: Build a Pydantic Contract

Define a complete data contract for a `user_events` dataset:

1. Create a Pydantic model with fields: `event_id`, `user_id`, `event_type` (enum: click, view, purchase), `timestamp`, `page_url`, `metadata` (optional dict)
2. Add field validators for business rules (e.g., `page_url` must start with `/`)
3. Write a `validate_dataframe` function that returns valid records and error details
4. Test with a DataFrame containing both valid and invalid rows
5. Implement a configurable error threshold (reject the entire batch if >X% fail)

### Exercise 3: Schema Evolution Assessment

Implement a schema evolution analyzer:

1. Define `ORDER_SCHEMA_V1` and `ORDER_SCHEMA_V2` (V2 renames a column and adds a new optional column)
2. Write a function that detects all changes between versions
3. Classify each change as breaking or non-breaking
4. Generate a migration plan for breaking changes
5. Write a test that verifies your classifier correctly identifies breaking changes

### Exercise 4: Contract-First Pipeline with Dagster

Build a contract-first Dagster pipeline:

1. Define the contract (Pydantic model) for a `transactions` dataset
2. Create a Dagster asset that produces mock transaction data
3. Add a `@dg.asset_check` that validates the asset against the contract
4. Create a downstream asset that consumes `transactions` (only runs if the check passes)
5. Write tests for both the asset and the contract check

### Exercise 5: DVC + lakeFS Integration Design

Design a hybrid versioning strategy for an ML platform:

1. Draw the architecture showing: raw data (lakeFS), features (lakeFS), models (DVC + MLflow)
2. Define the versioning workflow: how a data scientist creates a new feature set, trains a model, and promotes to production
3. Write pseudocode for the CI/CD pipeline that validates data contracts before merging data changes
4. Identify which artifacts are versioned where and why

---

## References

- [lakeFS Documentation](https://docs.lakefs.io/)
- [lakeFS GitHub Repository](https://github.com/treeverse/lakeFS)
- [DVC Documentation](https://dvc.org/doc)
- [Data Contracts — PayPal Engineering Blog](https://medium.com/paypal-tech/the-next-big-thing-in-data-engineering-data-contracts-17a55e7a0b89)
- [Great Expectations Documentation](https://docs.greatexpectations.io/)
- [Soda Data Contracts](https://docs.soda.io/soda/data-contracts.html)
- [Avro Schema Evolution](https://avro.apache.org/docs/current/specification/)
- [Data Mesh by Zhamak Dehghani](https://www.datamesh-architecture.com/)
- [JSON Schema Specification](https://json-schema.org/)
- [Confluent Schema Registry](https://docs.confluent.io/platform/current/schema-registry/)

---

[← Previous: 20. Dagster Asset Orchestration](20_Dagster_Asset_Orchestration.md) | [Next: Overview →](00_Overview.md)
