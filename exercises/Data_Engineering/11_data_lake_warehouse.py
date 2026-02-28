"""
Exercise Solutions: Lesson 11 - Data Lake & Warehouse

Covers:
  - Problem 1: Architecture Selection (10TB daily log data + ML)
  - Problem 2: Delta Lake Implementation (SCD Type 2 with MERGE)

Note: Pure Python simulation of lakehouse concepts.
"""

from datetime import datetime
from copy import deepcopy


# ---------------------------------------------------------------------------
# Problem 1: Architecture Selection
# Choose and justify an architecture for:
# - 10TB daily log data
# - Used for ML model training
# - Raw data retention for 5 years
# ---------------------------------------------------------------------------

def problem1_architecture_selection():
    """Evaluate Data Warehouse vs Data Lake vs Lakehouse for the given requirements.

    Requirements analysis:
    1. 10TB/day = 3.65PB/year = ~18PB over 5 years -> cost is critical
    2. ML model training -> needs raw/semi-structured data, not just SQL
    3. 5-year retention -> cold storage tier needed

    A traditional data warehouse (Redshift, BigQuery) would be too expensive
    for 18PB of raw log storage and cannot efficiently serve ML frameworks.
    A raw data lake (S3/GCS) has low cost but no ACID, no schema enforcement,
    and querying 10TB daily is slow without optimization.
    A lakehouse (Delta Lake / Iceberg on object storage) combines both.
    """

    architectures = [
        {
            "name": "Data Warehouse (e.g., BigQuery, Snowflake)",
            "score": 4,  # out of 10
            "storage_cost": "HIGH (~$23/TB/month for active storage)",
            "5yr_cost_estimate": "~$5M for 18PB",
            "ml_support": "LIMITED (SQL-only, export needed for Python/Spark)",
            "raw_retention": "EXPENSIVE (storing raw logs in columnar format wastes resources)",
            "query_performance": "EXCELLENT (optimized for SQL analytics)",
            "verdict": "NOT RECOMMENDED - cost prohibitive for raw log retention",
        },
        {
            "name": "Data Lake (e.g., S3 + Athena)",
            "score": 6,
            "storage_cost": "LOW (~$0.023/GB/month S3, ~$0.004 Glacier)",
            "5yr_cost_estimate": "~$100K-$500K with tiering",
            "ml_support": "GOOD (Spark/Python read directly from S3)",
            "raw_retention": "EXCELLENT (store anything cheaply)",
            "query_performance": "VARIABLE (no indexing, full scans unless partitioned)",
            "verdict": "VIABLE but lacks ACID and schema enforcement",
        },
        {
            "name": "Lakehouse (e.g., Delta Lake on S3 / Iceberg on GCS)",
            "score": 9,
            "storage_cost": "LOW (same as data lake - object storage)",
            "5yr_cost_estimate": "~$100K-$500K storage + compute costs",
            "ml_support": "EXCELLENT (Spark DataFrames, direct Parquet access for ML)",
            "raw_retention": "EXCELLENT (Bronze layer = raw, lifecycle policies for tiering)",
            "query_performance": "GOOD-EXCELLENT (Z-ordering, partition pruning, time travel)",
            "verdict": "RECOMMENDED - best balance of cost, flexibility, and governance",
        },
    ]

    print("\n  Requirements:")
    print("    - 10TB daily log data")
    print("    - ML model training")
    print("    - Raw data retention for 5 years")
    print()

    for arch in architectures:
        marker = " *** RECOMMENDED ***" if arch["score"] >= 9 else ""
        print(f"  {arch['name']}{marker}")
        print(f"    Storage Cost   : {arch['storage_cost']}")
        print(f"    5-Year Estimate: {arch['5yr_cost_estimate']}")
        print(f"    ML Support     : {arch['ml_support']}")
        print(f"    Raw Retention  : {arch['raw_retention']}")
        print(f"    Query Perf     : {arch['query_performance']}")
        print(f"    Verdict        : {arch['verdict']}")
        print()

    # Recommended architecture diagram
    print("  Recommended Lakehouse Architecture:")
    print("  " + "=" * 55)
    print("  | Ingestion      | Bronze (Raw)     | Kafka/Flume -> S3     |")
    print("  | Cleaning       | Silver (Clean)   | Spark + Delta MERGE   |")
    print("  | Analytics       | Gold (Business)  | Aggregated tables     |")
    print("  | ML Training    | Silver/Gold      | Spark ML / SageMaker  |")
    print("  | Cold Storage   | Bronze (>1 yr)   | S3 Glacier lifecycle  |")
    print("  " + "=" * 55)

    return architectures


# ---------------------------------------------------------------------------
# Problem 2: Delta Lake Implementation
# Implement SCD Type 2 for customer data using Delta Lake MERGE.
# ---------------------------------------------------------------------------

class SimulatedDeltaTable:
    """Simulates a Delta Lake table with MERGE, time travel, and history.

    In PySpark with Delta Lake:
        from delta.tables import DeltaTable

        delta_table = DeltaTable.forPath(spark, "/data/silver/customers")

        delta_table.alias("target").merge(
            source_df.alias("source"),
            "target.customer_id = source.customer_id AND target.is_current = true"
        ).whenMatchedUpdate(
            condition="target.city != source.city OR target.email != source.email",
            set={
                "is_current": "false",
                "effective_end": "current_date()",
            }
        ).whenNotMatchedInsert(
            values={...}
        ).execute()
    """

    def __init__(self, table_name: str):
        self.table_name = table_name
        self.data: list[dict] = []
        self.history: list[dict] = []
        self._version = 0
        self._next_sk = 1

    def _snapshot(self) -> list[dict]:
        """Create a deep copy for time travel."""
        return deepcopy(self.data)

    def _record_history(self, operation: str, details: dict) -> None:
        self.history.append({
            "version": self._version,
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            **details,
        })
        self._version += 1

    def insert(self, rows: list[dict]) -> None:
        """Bulk insert rows."""
        for row in rows:
            row["_surrogate_key"] = self._next_sk
            self._next_sk += 1
            self.data.append(row)
        self._record_history("INSERT", {"rows_affected": len(rows)})

    def merge_scd_type2(self, source_rows: list[dict], merge_key: str,
                        change_columns: list[str]) -> dict:
        """Execute an SCD Type 2 MERGE operation.

        For each source row:
        1. If a current record exists AND any change_column differs:
           - Close the old record (is_current=False, set effective_end)
           - Insert a new current record
        2. If no current record exists:
           - Insert as a new record

        Delta Lake SQL equivalent:

            MERGE INTO customers_dim AS target
            USING (
                SELECT *, true AS _is_new
                FROM source_updates
            ) AS source
            ON target.customer_id = source.customer_id AND target.is_current = true
            WHEN MATCHED AND (target.city != source.city OR target.email != source.email) THEN
                UPDATE SET is_current = false, effective_end = current_date()
            WHEN NOT MATCHED THEN
                INSERT (customer_id, name, city, email, effective_start, effective_end, is_current)
                VALUES (source.customer_id, source.name, source.city, source.email,
                        current_date(), null, true)
        """
        today = datetime.now().strftime("%Y-%m-%d")
        stats = {"updated": 0, "inserted": 0, "unchanged": 0}

        for src in source_rows:
            src_key = src[merge_key]

            # Find current record for this key
            current_record = None
            for row in self.data:
                if row[merge_key] == src_key and row.get("is_current", False):
                    current_record = row
                    break

            if current_record is not None:
                # Check if any change column differs
                changed = any(
                    current_record.get(col) != src.get(col)
                    for col in change_columns
                )
                if changed:
                    # Close old record
                    current_record["is_current"] = False
                    current_record["effective_end"] = today

                    # Insert new current record
                    new_record = {
                        "_surrogate_key": self._next_sk,
                        merge_key: src_key,
                        "effective_start": today,
                        "effective_end": None,
                        "is_current": True,
                    }
                    for col in change_columns:
                        new_record[col] = src[col]
                    # Carry forward non-change columns
                    for col in current_record:
                        if col not in new_record and col != "_surrogate_key":
                            new_record[col] = current_record[col]
                    self._next_sk += 1
                    self.data.append(new_record)
                    stats["updated"] += 1
                else:
                    stats["unchanged"] += 1
            else:
                # New record
                new_record = {
                    "_surrogate_key": self._next_sk,
                    merge_key: src_key,
                    "effective_start": today,
                    "effective_end": None,
                    "is_current": True,
                    **{col: src[col] for col in change_columns},
                }
                # Include any extra columns from source
                for col in src:
                    if col not in new_record:
                        new_record[col] = src[col]
                self._next_sk += 1
                self.data.append(new_record)
                stats["inserted"] += 1

        self._record_history("MERGE (SCD Type 2)", stats)
        return stats

    def display(self, title: str = "") -> None:
        if title:
            print(f"\n  {title}")
        cols = ["_surrogate_key", "customer_id", "name", "city", "email",
                "effective_start", "effective_end", "is_current"]
        available_cols = [c for c in cols if any(c in row for row in self.data)]

        header = "".join(f"{c:<18}" for c in available_cols)
        print(f"  {header}")
        print(f"  {'-' * len(header)}")
        for row in self.data:
            line = "".join(f"{str(row.get(c, 'NULL')):<18}" for c in available_cols)
            print(f"  {line}")

    def describe_history(self) -> None:
        """Display the table's transaction history (like DESCRIBE HISTORY in Delta Lake)."""
        print(f"\n  DESCRIBE HISTORY {self.table_name}:")
        print(f"  {'Version':<10} {'Operation':<25} {'Details'}")
        print(f"  {'-'*10} {'-'*25} {'-'*30}")
        for h in self.history:
            details = {k: v for k, v in h.items() if k not in ("version", "timestamp", "operation")}
            print(f"  {h['version']:<10} {h['operation']:<25} {details}")


def problem2_delta_lake_scd2():
    """Demonstrate SCD Type 2 using simulated Delta Lake MERGE."""

    table = SimulatedDeltaTable("customers_dim")

    # Initial load
    initial_customers = [
        {"customer_id": "C001", "name": "Alice", "city": "Seoul", "email": "alice@a.com",
         "effective_start": "2024-01-01", "effective_end": None, "is_current": True},
        {"customer_id": "C002", "name": "Bob", "city": "Busan", "email": "bob@b.com",
         "effective_start": "2024-01-01", "effective_end": None, "is_current": True},
        {"customer_id": "C003", "name": "Carol", "city": "Incheon", "email": "carol@c.com",
         "effective_start": "2024-01-01", "effective_end": None, "is_current": True},
    ]
    table.insert(initial_customers)
    table.display("After Initial Load (version 0):")

    # Apply updates: C001 moved, C003 changed email, C004 is new
    updates = [
        {"customer_id": "C001", "name": "Alice", "city": "Jeju", "email": "alice@a.com"},
        {"customer_id": "C003", "name": "Carol", "city": "Incheon", "email": "carol@new.com"},
        {"customer_id": "C004", "name": "Dave", "city": "Daegu", "email": "dave@d.com"},
    ]

    print("\n  Applying MERGE with updates:")
    print("    C001: city Seoul -> Jeju")
    print("    C003: email carol@c.com -> carol@new.com")
    print("    C004: new customer (insert)")

    stats = table.merge_scd_type2(
        source_rows=updates,
        merge_key="customer_id",
        change_columns=["name", "city", "email"],
    )
    print(f"\n  MERGE stats: {stats}")
    table.display("After MERGE (version 1):")

    # Show history
    table.describe_history()

    # Query: current state
    print("\n  Current Customers (is_current = True):")
    current = [r for r in table.data if r.get("is_current")]
    for r in current:
        print(f"    {r['customer_id']}: {r['name']} in {r['city']} ({r['email']})")

    return table


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Problem 1: Architecture Selection (10TB Log Data + ML)")
    print("=" * 70)
    problem1_architecture_selection()

    print()
    print("=" * 70)
    print("Problem 2: Delta Lake SCD Type 2 with MERGE")
    print("=" * 70)
    problem2_delta_lake_scd2()
