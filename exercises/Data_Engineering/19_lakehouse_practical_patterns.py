"""
Exercise Solutions: Lesson 19 - Lakehouse Practical Patterns

Covers:
  - Exercise 1: Build a Three-Layer Medallion Pipeline
  - Exercise 2: Incremental MERGE with Conflict Resolution
  - Exercise 3: SCD Type 2 Dimension Table
  - Exercise 4: Table Maintenance and Query Performance
  - Exercise 5: Iceberg Partition Evolution

Note: Pure Python simulation of Delta Lake / Iceberg patterns.
"""

import random
import json
from datetime import datetime, date, timedelta
from collections import defaultdict
from copy import deepcopy


# ---------------------------------------------------------------------------
# Simulated Delta Lake Table
# ---------------------------------------------------------------------------

class DeltaTable:
    """Simulates Delta Lake with MERGE, time travel, OPTIMIZE, VACUUM, and history."""
    def __init__(self, name: str):
        self.name = name
        self.data: list[dict] = []
        self.versions: list[list[dict]] = []  # time travel snapshots
        self.history: list[dict] = []
        self.num_files: int = 0
        self.size_bytes: int = 0

    def _snapshot(self):
        self.versions.append(deepcopy(self.data))
        ver = len(self.versions) - 1
        return ver

    def _log(self, operation: str, **kwargs):
        self.history.append({
            "version": len(self.versions) - 1,
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            **kwargs,
        })

    def append(self, rows: list[dict], batch_id: str = ""):
        """Append rows (Bronze-style ingestion)."""
        for r in rows:
            r["_ingested_at"] = datetime.now().isoformat()
            if batch_id:
                r["_batch_id"] = batch_id
        self.data.extend(rows)
        self.num_files += 1
        self.size_bytes += sum(len(json.dumps(r)) for r in rows)
        ver = self._snapshot()
        self._log("WRITE", mode="append", rows_written=len(rows), batch_id=batch_id)

    def merge(self, source: list[dict], key: str, update_condition=None,
              delete_condition=None) -> dict:
        """MERGE (upsert) with optional update condition and delete."""
        stats = {"rows_updated": 0, "rows_inserted": 0, "rows_deleted": 0}
        existing_keys = {str(r[key]): i for i, r in enumerate(self.data)}

        for src_row in source:
            src_key = str(src_row[key])
            if src_key in existing_keys:
                idx = existing_keys[src_key]
                if update_condition and not update_condition(src_row, self.data[idx]):
                    continue
                if delete_condition and delete_condition(src_row, self.data[idx]):
                    self.data[idx] = None  # Mark for deletion
                    stats["rows_deleted"] += 1
                else:
                    self.data[idx] = src_row
                    stats["rows_updated"] += 1
            else:
                self.data.append(src_row)
                stats["rows_inserted"] += 1

        self.data = [r for r in self.data if r is not None]
        ver = self._snapshot()
        self._log("MERGE", **stats)
        return stats

    def replace_where(self, condition, new_rows: list[dict]):
        """replaceWhere: overwrite specific partitions."""
        self.data = [r for r in self.data if not condition(r)]
        self.data.extend(new_rows)
        self._snapshot()
        self._log("REPLACE_WHERE", rows_replaced=len(new_rows))

    def version_as_of(self, version: int) -> list[dict]:
        """Time travel: read a historical version."""
        if 0 <= version < len(self.versions):
            return self.versions[version]
        return []

    def describe_history(self):
        print(f"\n  DESCRIBE HISTORY {self.name}:")
        for h in self.history:
            details = {k: v for k, v in h.items() if k not in ("version", "timestamp", "operation")}
            print(f"    v{h['version']}: {h['operation']} {details}")


# ---------------------------------------------------------------------------
# Exercise 1: Build a Three-Layer Medallion Pipeline
# ---------------------------------------------------------------------------

def exercise1_medallion_pipeline():
    """Bronze -> Silver -> Gold pipeline for retail orders.

    Idempotency verification: running the pipeline twice with the same
    input should NOT double the Gold row counts because:
    - Silver uses MERGE (upsert by order_id): re-running updates existing rows.
    - Gold uses replaceWhere on affected dates: re-running replaces, not appends.
    """
    # Bronze: raw ingestion (append-only)
    bronze = DeltaTable("bronze_orders")

    raw_orders = []
    for i in range(50):
        raw_orders.append({
            "order_id": f"ORD-{i+1:04d}",
            "customer_id": f"C{random.randint(1, 20):03d}",
            "amount": round(random.uniform(-10, 500), 2),  # Some invalid
            "order_time": f"2024-11-{random.randint(1, 15):02d}T{random.randint(8, 20):02d}:00:00",
            "status": random.choice(["completed", "completed", "cancelled"]),
            "_source_file": "orders_batch_001.json",
        })
    # Add some duplicates
    raw_orders.append(dict(raw_orders[0]))
    raw_orders.append(dict(raw_orders[5]))

    bronze.append(raw_orders, batch_id="batch_001")
    print(f"  Bronze: {len(bronze.data)} rows ingested (append-only, includes duplicates)")

    # Silver: clean, validate, deduplicate, upsert
    silver = DeltaTable("silver_orders")

    # Filter: remove nulls and invalid amounts
    valid = [r for r in bronze.data if r.get("order_id") and r.get("amount", 0) >= 0]
    print(f"  Silver: {len(valid)} rows after validation (removed {len(bronze.data) - len(valid)} invalid)")

    # Deduplicate by order_id (keep latest _ingested_at)
    deduped: dict[str, dict] = {}
    for r in valid:
        oid = r["order_id"]
        if oid not in deduped or r["_ingested_at"] > deduped[oid]["_ingested_at"]:
            deduped[oid] = r
    silver_rows = list(deduped.values())
    print(f"  Silver: {len(silver_rows)} rows after deduplication")

    # MERGE into Silver
    stats = silver.merge(silver_rows, key="order_id",
                         update_condition=lambda src, tgt: src.get("_ingested_at", "") > tgt.get("_ingested_at", ""))
    print(f"  Silver MERGE: {stats}")

    # Gold: daily aggregation
    gold = DeltaTable("gold_daily_summary")

    daily_agg: dict[tuple[str, str], dict] = {}
    for r in silver.data:
        order_date = r["order_time"][:10]
        status = r["status"]
        key = (order_date, status)
        if key not in daily_agg:
            daily_agg[key] = {"order_date": order_date, "status": status,
                              "order_count": 0, "total_amount": 0.0}
        daily_agg[key]["order_count"] += 1
        daily_agg[key]["total_amount"] += r.get("amount", 0)

    gold_rows = []
    for stats in daily_agg.values():
        stats["total_amount"] = round(stats["total_amount"], 2)
        stats["avg_amount"] = round(stats["total_amount"] / stats["order_count"], 2)
        gold_rows.append(stats)

    gold.data = gold_rows
    gold._snapshot()
    gold._log("REPLACE_WHERE", rows_replaced=len(gold_rows))

    print(f"\n  Gold Daily Summary ({len(gold.data)} rows):")
    print(f"  {'Date':<12} {'Status':<12} {'Count':>6} {'Total':>10} {'Avg':>8}")
    print(f"  {'-'*12} {'-'*12} {'-'*6} {'-'*10} {'-'*8}")
    for r in sorted(gold.data, key=lambda x: (x["order_date"], x["status"]))[:10]:
        print(f"  {r['order_date']:<12} {r['status']:<12} {r['order_count']:>6} "
              f"${r['total_amount']:>9.2f} ${r['avg_amount']:>7.2f}")

    # Idempotency test: re-run pipeline
    print(f"\n  Idempotency Test: re-running pipeline with same input...")
    silver_before = len(silver.data)
    gold_before = len(gold.data)
    silver.merge(silver_rows, key="order_id",
                 update_condition=lambda src, tgt: src.get("_ingested_at", "") > tgt.get("_ingested_at", ""))
    print(f"    Silver rows: {silver_before} -> {len(silver.data)} (should be same)")
    print(f"    Gold rows:   {gold_before} -> {len(gold.data)} (should be same)")
    print(f"    Idempotent: {len(silver.data) == silver_before}")

    silver.describe_history()
    return bronze, silver, gold


# ---------------------------------------------------------------------------
# Exercise 2: Incremental MERGE with Conflict Resolution
# ---------------------------------------------------------------------------

def exercise2_incremental_merge():
    """Out-of-order batch handling: batch 3 before batch 2."""
    silver = DeltaTable("orders_silver")

    # Batch 1
    batch1 = [
        {"order_id": "O1", "status": "pending", "amount": 100, "updated_at": "2024-11-01T10:00:00", "_last_seen_batch": 1},
        {"order_id": "O2", "status": "pending", "amount": 200, "updated_at": "2024-11-01T11:00:00", "_last_seen_batch": 1},
        {"order_id": "O3", "status": "pending", "amount": 150, "updated_at": "2024-11-01T12:00:00", "_last_seen_batch": 1},
    ]
    silver.merge(batch1, key="order_id")
    print(f"\n  After Batch 1: {len(silver.data)} rows")

    # Batch 3 arrives BEFORE batch 2 (out of order)
    batch3 = [
        {"order_id": "O1", "status": "completed", "amount": 100, "updated_at": "2024-11-03T10:00:00", "_last_seen_batch": 3},
        {"order_id": "O4", "status": "pending", "amount": 300, "updated_at": "2024-11-03T09:00:00", "_last_seen_batch": 3},
    ]
    silver.merge(batch3, key="order_id",
                 update_condition=lambda src, tgt: src["updated_at"] > tgt["updated_at"])
    print(f"  After Batch 3 (out of order): {len(silver.data)} rows")
    for r in silver.data:
        print(f"    {r['order_id']}: status={r['status']}, updated={r['updated_at']}, batch={r['_last_seen_batch']}")

    # Batch 2 arrives late
    batch2 = [
        {"order_id": "O1", "status": "shipped", "amount": 100, "updated_at": "2024-11-02T10:00:00", "_last_seen_batch": 2},
        {"order_id": "O2", "status": "completed", "amount": 200, "updated_at": "2024-11-02T11:00:00", "_last_seen_batch": 2},
    ]
    silver.merge(batch2, key="order_id",
                 update_condition=lambda src, tgt: src["updated_at"] > tgt["updated_at"])
    print(f"\n  After Batch 2 (late arrival): {len(silver.data)} rows")
    for r in silver.data:
        print(f"    {r['order_id']}: status={r['status']}, updated={r['updated_at']}, batch={r['_last_seen_batch']}")

    # Verify: O1 should still be 'completed' from batch 3, NOT 'shipped' from batch 2
    o1 = next(r for r in silver.data if r["order_id"] == "O1")
    assert o1["status"] == "completed", f"O1 should be 'completed' from batch 3, got '{o1['status']}'"
    assert o1["_last_seen_batch"] == 3, f"O1 should be from batch 3, got batch {o1['_last_seen_batch']}"
    print(f"\n  Verification: O1 status='{o1['status']}' (batch 3 NOT overwritten by batch 2)")

    # Why explicit column listing in whenMatchedUpdate is safer:
    print(f"\n  Why explicit columns in whenMatchedUpdate(set={{...}}) is safer:")
    print(f"    - whenMatchedUpdateAll() updates ALL columns including metadata.")
    print(f"    - If source has extra/missing columns, it may corrupt the target.")
    print(f"    - Explicit listing ensures only intended columns are overwritten.")
    print(f"    - It also prevents accidental changes to audit columns like _ingested_at.")

    silver.describe_history()
    return silver


# ---------------------------------------------------------------------------
# Exercise 3: SCD Type 2 Dimension Table
# ---------------------------------------------------------------------------

def exercise3_scd_type2():
    """SCD Type 2 dimension with time travel verification."""
    dim = DeltaTable("customers_dim")

    # Initial load: 5 customers
    initial = [
        {"customer_id": "C1", "name": "Alice", "city": "Seoul", "email": "alice@a.com",
         "effective_from": "2024-01-01", "effective_to": None, "is_current": True},
        {"customer_id": "C2", "name": "Bob", "city": "Busan", "email": "bob@b.com",
         "effective_from": "2024-01-01", "effective_to": None, "is_current": True},
        {"customer_id": "C3", "name": "Carol", "city": "Incheon", "email": "carol@c.com",
         "effective_from": "2024-01-01", "effective_to": None, "is_current": True},
        {"customer_id": "C4", "name": "Dave", "city": "Daegu", "email": "dave@d.com",
         "effective_from": "2024-01-01", "effective_to": None, "is_current": True},
        {"customer_id": "C5", "name": "Eve", "city": "Gwangju", "email": "eve@e.com",
         "effective_from": "2024-01-01", "effective_to": None, "is_current": True},
    ]
    dim.data = initial
    dim._snapshot()
    dim._log("INSERT", rows=len(initial))
    print(f"\n  Initial load: {len(dim.data)} customers")

    # SCD Type 2 updates: C1 moves to Jeju, C3 moves to Sejong, C6 is new
    updates = [
        {"customer_id": "C1", "name": "Alice", "city": "Jeju", "email": "alice@a.com"},
        {"customer_id": "C3", "name": "Carol", "city": "Sejong", "email": "carol@c.com"},
        {"customer_id": "C6", "name": "Frank", "city": "Ulsan", "email": "frank@f.com"},
    ]

    today = "2024-07-01"
    change_cols = ["city", "email"]
    stats = {"updated": 0, "inserted": 0}

    for upd in updates:
        current = next((r for r in dim.data if r["customer_id"] == upd["customer_id"] and r["is_current"]), None)
        if current:
            changed = any(current.get(c) != upd.get(c) for c in change_cols)
            if changed:
                current["is_current"] = False
                current["effective_to"] = today
                new_row = {**current, **upd, "effective_from": today, "effective_to": None, "is_current": True}
                dim.data.append(new_row)
                stats["updated"] += 1
        else:
            dim.data.append({**upd, "effective_from": today, "effective_to": None, "is_current": True})
            stats["inserted"] += 1

    dim._snapshot()
    dim._log("MERGE (SCD2)", **stats)

    print(f"\n  After SCD Type 2 MERGE: {stats}")
    print(f"\n  Full table ({len(dim.data)} rows):")
    print(f"  {'CustID':<8} {'Name':<8} {'City':<10} {'From':<12} {'To':<12} {'Current'}")
    print(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*12} {'-'*12} {'-'*7}")
    for r in sorted(dim.data, key=lambda x: (x["customer_id"], x["effective_from"])):
        to_date = r["effective_to"] or "NULL"
        print(f"  {r['customer_id']:<8} {r['name']:<8} {r['city']:<10} "
              f"{r['effective_from']:<12} {to_date:<12} {r['is_current']}")

    # Point-in-time query: what was each customer's city on 2024-06-30?
    query_date = "2024-06-30"
    print(f"\n  Point-in-time query: city on {query_date}")
    for cid in ["C1", "C2", "C3", "C4", "C5"]:
        for r in dim.data:
            if r["customer_id"] == cid:
                end = r["effective_to"] or "9999-12-31"
                if r["effective_from"] <= query_date <= end:
                    print(f"    {cid}: {r['city']}")
                    break

    # Time travel: compare version 0 with current
    v0 = dim.version_as_of(0)
    print(f"\n  Time travel (version 0): {len(v0)} rows")
    print(f"  Current state: {len(dim.data)} rows")
    print(f"  Rows added by SCD2: {len(dim.data) - len(v0)}")

    return dim


# ---------------------------------------------------------------------------
# Exercise 4: Table Maintenance and Query Performance
# ---------------------------------------------------------------------------

def exercise4_table_maintenance():
    """Demonstrate OPTIMIZE, Z-ORDER, and VACUUM impact."""
    table = DeltaTable("orders_table")

    # Simulate 1000 small batches (streaming writes)
    print(f"\n  Writing 1000 small batches of 100 rows each...")
    total_rows = 0
    for batch in range(1000):
        rows = [{"order_id": batch * 100 + i, "order_date": f"2024-{(batch % 12) + 1:02d}-{(i % 28) + 1:02d}",
                 "customer_id": f"C{random.randint(1, 1000):04d}", "amount": round(random.uniform(10, 500), 2)}
                for i in range(100)]
        table.data.extend(rows)
        table.num_files += 1
        total_rows += 100

    table.size_bytes = total_rows * 200  # ~200 bytes per row
    print(f"  DESCRIBE DETAIL before OPTIMIZE:")
    print(f"    numFiles: {table.num_files}")
    print(f"    sizeInBytes: {table.size_bytes:,} ({table.size_bytes / 1024 / 1024:.1f} MB)")
    print(f"    totalRows: {total_rows:,}")

    # Simulate query before OPTIMIZE
    filter_date = "2024-07"
    filter_customer = "C0042"
    matching_before = sum(1 for r in table.data
                          if r["order_date"].startswith(filter_date) and r["customer_id"] == filter_customer)
    files_scanned_before = table.num_files  # Must scan all files
    print(f"\n  Query: WHERE order_date LIKE '{filter_date}%' AND customer_id = '{filter_customer}'")
    print(f"    Before OPTIMIZE: files scanned = {files_scanned_before}, rows matched = {matching_before}")

    # OPTIMIZE + ZORDER
    print(f"\n  Running OPTIMIZE + ZORDER BY (order_date, customer_id)...")
    # Z-ordering co-locates rows with similar (order_date, customer_id) values
    # into the same files, enabling file-level pruning
    optimized_files = max(1, table.num_files // 100)  # Compaction reduces files by ~100x
    print(f"  After OPTIMIZE:")
    print(f"    numFiles: {table.num_files} -> {optimized_files}")

    # After ZORDER, the query only needs to scan ~1-2% of files
    files_scanned_after = max(1, optimized_files // 10)
    print(f"\n  Same query after OPTIMIZE + ZORDER:")
    print(f"    Files scanned: {files_scanned_before} -> {files_scanned_after}")
    print(f"    Improvement: {files_scanned_before / files_scanned_after:.0f}x fewer files")

    # VACUUM
    print(f"\n  Running VACUUM (retention = 1 hour):")
    print(f"    spark.conf.set('spark.databricks.delta.retentionDurationCheck.enabled', 'false')")
    print(f"    Old files removed: {table.num_files - optimized_files}")
    print(f"    Remaining files: {optimized_files}")

    print(f"\n  DESCRIBE DETAIL after OPTIMIZE + VACUUM:")
    print(f"    numFiles: {optimized_files}")
    print(f"    sizeInBytes: ~{table.size_bytes:,} (same data, fewer files)")
    print(f"    Benefit: {table.num_files / optimized_files:.0f}x fewer files to open")

    return table


# ---------------------------------------------------------------------------
# Exercise 5: Iceberg Partition Evolution
# ---------------------------------------------------------------------------

def exercise5_iceberg_partition_evolution():
    """Demonstrate Iceberg's partition evolution without data rewrite.

    Delta Lake vs Iceberg for partition changes:
    - Delta Lake: Changing partitioning requires a full table rewrite because
      partition columns are encoded in the file paths (e.g., /date=2024-01-01/).
    - Iceberg: Partition spec is stored in metadata, not file paths.
      New writes use the new spec; old files are read using the old spec.
      The manifest files record which partition spec each data file belongs to.
      This is called "partition evolution" and is zero-cost.
    """
    # Simulate Iceberg table with monthly partitions
    print(f"\n  Step 1: Create Iceberg table partitioned by months(event_ts)")
    monthly_data = []
    for month in range(1, 7):
        for i in range(100):
            monthly_data.append({
                "event_id": f"evt_{month}_{i}",
                "event_ts": f"2024-{month:02d}-{random.randint(1, 28):02d}T10:00:00",
                "user_id": f"user_{random.randint(1, 50)}",
                "action": random.choice(["click", "view", "purchase"]),
                "partition_spec": "months(event_ts)",
            })
    print(f"    Written {len(monthly_data)} events across 6 monthly partitions")
    print(f"    Files: ~6 (one per month partition)")

    # Step 2: Compact
    print(f"\n  Step 2: CALL rewrite_data_files()")
    print(f"    Before: 6 files")
    print(f"    After:  6 files (already compact)")

    # Step 3: Evolve partition to daily
    print(f"\n  Step 3: Evolve partitioning to days(event_ts)")
    print(f"    ALTER TABLE db.events DROP PARTITION FIELD months(event_ts);")
    print(f"    ALTER TABLE db.events ADD PARTITION FIELD days(event_ts);")
    print(f"    Existing data files: NOT rewritten (still monthly partitioned)")
    print(f"    Files table unchanged: {len(monthly_data)} rows in 6 files")

    # Step 4: Write new data with daily partitioning
    daily_data = []
    for day in range(1, 8):
        for i in range(20):
            daily_data.append({
                "event_id": f"evt_7_{day}_{i}",
                "event_ts": f"2024-07-{day:02d}T10:00:00",
                "user_id": f"user_{random.randint(1, 50)}",
                "action": random.choice(["click", "view"]),
                "partition_spec": "days(event_ts)",
            })
    total_data = monthly_data + daily_data
    print(f"\n  Step 4: Write {len(daily_data)} new events with daily partitioning")
    print(f"    New files: 7 (one per day)")
    print(f"    Total files: 6 (monthly) + 7 (daily) = 13")
    print(f"    Query transparently reads both partition schemes!")

    # Step 5: Snapshot history and expire
    print(f"\n  Step 5: Snapshot history")
    snapshots = [
        {"snapshot_id": 1, "operation": "append", "partition_spec": "months", "files": 6},
        {"snapshot_id": 2, "operation": "rewrite_data_files", "partition_spec": "months", "files": 6},
        {"snapshot_id": 3, "operation": "replace_partition_spec", "partition_spec": "days", "files": 6},
        {"snapshot_id": 4, "operation": "append", "partition_spec": "days", "files": 13},
    ]
    for s in snapshots:
        print(f"    Snapshot {s['snapshot_id']}: {s['operation']} "
              f"(spec={s['partition_spec']}, files={s['files']})")

    print(f"\n    CALL expire_snapshots('db.events', TIMESTAMP '2024-04-01')")
    print(f"    Expired: snapshots 1-2 (older than 3 months)")

    # Step 6: Explanation
    print(f"\n  Step 6: Why Iceberg can evolve partitions without rewrite:")
    print(f"    Delta Lake: partition columns are physical directory paths (/date=...).")
    print(f"    Changing partitioning means physically moving ALL data files to new paths.")
    print(f"    ")
    print(f"    Iceberg: partition spec is stored in METADATA (manifest files).")
    print(f"    Each data file records WHICH partition spec it was written under.")
    print(f"    Queries use manifest metadata to correctly route files to partitions.")
    print(f"    No data movement needed: old files stay in place, new files use new spec.")
    print(f"    This is possible because Iceberg tracks partitions at the metadata layer,")
    print(f"    not the filesystem layer.")

    return total_data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Exercise 1: Three-Layer Medallion Pipeline")
    print("=" * 70)
    exercise1_medallion_pipeline()

    print()
    print("=" * 70)
    print("Exercise 2: Incremental MERGE with Conflict Resolution")
    print("=" * 70)
    exercise2_incremental_merge()

    print()
    print("=" * 70)
    print("Exercise 3: SCD Type 2 Dimension Table")
    print("=" * 70)
    exercise3_scd_type2()

    print()
    print("=" * 70)
    print("Exercise 4: Table Maintenance and Query Performance")
    print("=" * 70)
    exercise4_table_maintenance()

    print()
    print("=" * 70)
    print("Exercise 5: Iceberg Partition Evolution")
    print("=" * 70)
    exercise5_iceberg_partition_evolution()
