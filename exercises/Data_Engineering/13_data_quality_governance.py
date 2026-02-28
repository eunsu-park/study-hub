"""
Exercise Solutions: Lesson 13 - Data Quality & Governance

Covers:
  - Problem 1: Great Expectations (Expectation Suite for order data)
  - Problem 2: Quality Dashboard (daily data quality scoring pipeline)
  - Problem 3: Lineage Tracking (ETL lineage system)

Note: Pure Python simulation of data quality frameworks.
"""

from datetime import datetime, date, timedelta
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Problem 1: Great Expectations
# Write an Expectation Suite for order data:
# NULL check, unique, value range, referential integrity.
# ---------------------------------------------------------------------------

@dataclass
class ExpectationResult:
    """Result of a single expectation check."""
    expectation_type: str
    column: str
    success: bool
    observed_value: Any
    details: str


class ExpectationSuite:
    """Simulates a Great Expectations ExpectationSuite.

    In Great Expectations:
        import great_expectations as gx

        context = gx.get_context()
        suite = context.add_expectation_suite("order_data_suite")

        # Add expectations
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(column="order_id")
        )
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeUnique(column="order_id")
        )
        # ... etc

        # Validate
        checkpoint = context.add_or_update_checkpoint(...)
        result = checkpoint.run()
    """
    def __init__(self, name: str):
        self.name = name
        self.expectations: list[dict] = []
        self.results: list[ExpectationResult] = []

    def expect_column_values_to_not_be_null(self, column: str) -> None:
        self.expectations.append({
            "type": "expect_column_values_to_not_be_null",
            "column": column,
        })

    def expect_column_values_to_be_unique(self, column: str) -> None:
        self.expectations.append({
            "type": "expect_column_values_to_be_unique",
            "column": column,
        })

    def expect_column_values_to_be_between(self, column: str,
                                            min_value: float,
                                            max_value: float) -> None:
        self.expectations.append({
            "type": "expect_column_values_to_be_between",
            "column": column,
            "min_value": min_value,
            "max_value": max_value,
        })

    def expect_column_values_to_be_in_set(self, column: str,
                                           value_set: list) -> None:
        self.expectations.append({
            "type": "expect_column_values_to_be_in_set",
            "column": column,
            "value_set": value_set,
        })

    def expect_column_pair_values_a_to_be_greater_than_b(
            self, column_a: str, column_b: str) -> None:
        self.expectations.append({
            "type": "expect_column_pair_values_a_to_be_greater_than_b",
            "column_a": column_a,
            "column_b": column_b,
        })

    def expect_compound_columns_to_be_unique(self, column_list: list[str]) -> None:
        self.expectations.append({
            "type": "expect_compound_columns_to_be_unique",
            "column_list": column_list,
        })

    def validate(self, data: list[dict]) -> dict:
        """Run all expectations against the data."""
        self.results = []

        for exp in self.expectations:
            exp_type = exp["type"]

            if exp_type == "expect_column_values_to_not_be_null":
                col = exp["column"]
                nulls = sum(1 for r in data if r.get(col) is None)
                self.results.append(ExpectationResult(
                    exp_type, col, nulls == 0, nulls,
                    f"{nulls} null values found" if nulls > 0 else "No nulls",
                ))

            elif exp_type == "expect_column_values_to_be_unique":
                col = exp["column"]
                values = [r.get(col) for r in data if r.get(col) is not None]
                dupes = len(values) - len(set(values))
                self.results.append(ExpectationResult(
                    exp_type, col, dupes == 0, dupes,
                    f"{dupes} duplicate values" if dupes > 0 else "All unique",
                ))

            elif exp_type == "expect_column_values_to_be_between":
                col = exp["column"]
                min_v, max_v = exp["min_value"], exp["max_value"]
                out_of_range = sum(
                    1 for r in data
                    if r.get(col) is not None and not (min_v <= r[col] <= max_v)
                )
                self.results.append(ExpectationResult(
                    exp_type, col, out_of_range == 0, out_of_range,
                    f"{out_of_range} out of [{min_v}, {max_v}]" if out_of_range > 0 else "All in range",
                ))

            elif exp_type == "expect_column_values_to_be_in_set":
                col = exp["column"]
                value_set = set(exp["value_set"])
                invalid = sum(
                    1 for r in data
                    if r.get(col) is not None and r[col] not in value_set
                )
                self.results.append(ExpectationResult(
                    exp_type, col, invalid == 0, invalid,
                    f"{invalid} values not in set" if invalid > 0 else "All valid",
                ))

            elif exp_type == "expect_column_pair_values_a_to_be_greater_than_b":
                col_a, col_b = exp["column_a"], exp["column_b"]
                violations = sum(
                    1 for r in data
                    if r.get(col_a) is not None and r.get(col_b) is not None
                    and r[col_a] <= r[col_b]
                )
                self.results.append(ExpectationResult(
                    exp_type, f"{col_a} > {col_b}", violations == 0, violations,
                    f"{violations} violations" if violations > 0 else "OK",
                ))

        passed = sum(1 for r in self.results if r.success)
        total = len(self.results)
        return {
            "suite_name": self.name,
            "success": passed == total,
            "statistics": {"evaluated": total, "successful": passed, "failed": total - passed},
        }

    def report(self) -> None:
        """Display validation results."""
        print(f"\n  Expectation Suite: '{self.name}'")
        print(f"  {'Expectation':<50} {'Column':<20} {'Status':<8} {'Detail'}")
        print(f"  {'-'*50} {'-'*20} {'-'*8} {'-'*25}")
        for r in self.results:
            status = "PASS" if r.success else "FAIL"
            print(f"  {r.expectation_type:<50} {r.column:<20} {status:<8} {r.details}")


def problem1_great_expectations():
    """Build and run an Expectation Suite for order data."""
    # Sample order data with intentional quality issues
    orders = [
        {"order_id": "ORD-001", "customer_id": "C001", "amount": 150.00, "discount": 15.00, "status": "completed", "order_date": "2024-11-01"},
        {"order_id": "ORD-002", "customer_id": "C002", "amount": 89.99, "discount": 0.00, "status": "completed", "order_date": "2024-11-01"},
        {"order_id": "ORD-003", "customer_id": None, "amount": 250.00, "discount": 25.00, "status": "completed", "order_date": "2024-11-02"},
        {"order_id": "ORD-004", "customer_id": "C001", "amount": -10.00, "discount": 0.00, "status": "cancelled", "order_date": "2024-11-02"},
        {"order_id": "ORD-005", "customer_id": "C003", "amount": 45.00, "discount": 50.00, "status": "shipped", "order_date": "2024-11-03"},
        {"order_id": "ORD-005", "customer_id": "C004", "amount": 320.00, "discount": 32.00, "status": "completed", "order_date": "2024-11-03"},
        {"order_id": "ORD-006", "customer_id": "C005", "amount": 75.00, "discount": 7.50, "status": "invalid_status", "order_date": "2024-11-04"},
    ]

    suite = ExpectationSuite("order_data_suite")

    # NULL checks
    suite.expect_column_values_to_not_be_null("order_id")
    suite.expect_column_values_to_not_be_null("customer_id")
    suite.expect_column_values_to_not_be_null("amount")

    # Uniqueness
    suite.expect_column_values_to_be_unique("order_id")

    # Value range
    suite.expect_column_values_to_be_between("amount", min_value=0.01, max_value=100000.00)

    # Referential / valid set
    suite.expect_column_values_to_be_in_set(
        "status", ["completed", "cancelled", "shipped", "returned", "pending"]
    )

    # Business rule: amount > discount
    suite.expect_column_pair_values_a_to_be_greater_than_b("amount", "discount")

    # Run validation
    summary = suite.validate(orders)
    suite.report()

    print(f"\n  Overall: {'PASSED' if summary['success'] else 'FAILED'}")
    print(f"  Statistics: {summary['statistics']}")

    return summary


# ---------------------------------------------------------------------------
# Problem 2: Quality Dashboard
# Design a pipeline that calculates and visualizes daily quality scores.
# ---------------------------------------------------------------------------

@dataclass
class QualityDimension:
    """Represents one of the 6 data quality dimensions.

    The 6 dimensions:
    1. Completeness: ratio of non-null values
    2. Uniqueness:   ratio of unique values where expected
    3. Validity:     ratio of values conforming to rules
    4. Accuracy:     ratio of values matching a known truth (hard to automate)
    5. Consistency:  cross-source or cross-column agreement
    6. Timeliness:   data freshness (age of most recent record)
    """
    name: str
    score: float  # 0.0 to 1.0
    details: str


def calculate_quality_scores(data: list[dict], check_date: str) -> dict:
    """Calculate quality scores across all 6 dimensions for a dataset."""
    total = len(data)
    if total == 0:
        return {"date": check_date, "overall_score": 0.0, "dimensions": []}

    # 1. Completeness: % of non-null across key columns
    key_columns = ["order_id", "customer_id", "amount", "status", "order_date"]
    total_cells = total * len(key_columns)
    null_cells = sum(
        1 for row in data for col in key_columns if row.get(col) is None
    )
    completeness = (total_cells - null_cells) / total_cells

    # 2. Uniqueness: order_id should be unique
    order_ids = [r.get("order_id") for r in data if r.get("order_id")]
    uniqueness = len(set(order_ids)) / len(order_ids) if order_ids else 0

    # 3. Validity: amount > 0, status in valid set
    valid_statuses = {"completed", "cancelled", "shipped", "returned", "pending"}
    valid_count = sum(
        1 for r in data
        if (r.get("amount") is not None and r["amount"] > 0)
        and (r.get("status") in valid_statuses)
    )
    validity = valid_count / total

    # 4. Accuracy: simplified - check if amounts are reasonable ($0.01-$100K)
    reasonable = sum(
        1 for r in data
        if r.get("amount") is not None and 0.01 <= r["amount"] <= 100000
    )
    accuracy = reasonable / total

    # 5. Consistency: order_date should be in valid format
    consistent = sum(
        1 for r in data
        if r.get("order_date") and len(r["order_date"]) == 10
    )
    consistency = consistent / total

    # 6. Timeliness: most recent order_date relative to check_date
    dates = [r["order_date"] for r in data if r.get("order_date")]
    if dates:
        max_date = max(dates)
        days_old = (datetime.strptime(check_date, "%Y-%m-%d") -
                    datetime.strptime(max_date, "%Y-%m-%d")).days
        timeliness = max(0, 1.0 - days_old / 7)  # Full score if < 1 day old
    else:
        timeliness = 0.0

    dimensions = [
        QualityDimension("Completeness", round(completeness, 4), f"{null_cells} nulls in {total_cells} cells"),
        QualityDimension("Uniqueness", round(uniqueness, 4), f"{len(set(order_ids))}/{len(order_ids)} unique order_ids"),
        QualityDimension("Validity", round(validity, 4), f"{valid_count}/{total} rows pass rules"),
        QualityDimension("Accuracy", round(accuracy, 4), f"{reasonable}/{total} reasonable amounts"),
        QualityDimension("Consistency", round(consistency, 4), f"{consistent}/{total} valid date formats"),
        QualityDimension("Timeliness", round(timeliness, 4), f"Most recent: {max(dates) if dates else 'N/A'}"),
    ]

    weights = [0.25, 0.15, 0.20, 0.15, 0.10, 0.15]
    overall = sum(d.score * w for d, w in zip(dimensions, weights))

    return {
        "date": check_date,
        "overall_score": round(overall, 4),
        "dimensions": dimensions,
        "record_count": total,
    }


def problem2_quality_dashboard():
    """Simulate a multi-day quality dashboard pipeline."""
    statuses = ["completed", "completed", "completed", "cancelled", "shipped"]

    daily_scores = []
    print("\n  Daily Quality Score Pipeline (7 days):")
    print(f"  {'Date':<12} {'Records':>8} {'Overall':>8} {'Complete':>8} {'Unique':>8} {'Valid':>8} {'Accurate':>8} {'Consist':>8} {'Timely':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    base_date = date(2024, 11, 1)
    for day in range(7):
        check_date = (base_date + timedelta(days=day)).isoformat()

        # Generate daily data with varying quality
        n = random.randint(80, 120)
        data = []
        for i in range(n):
            row = {
                "order_id": f"ORD-{day*200+i+1:05d}",
                "customer_id": f"C{random.randint(1, 50):03d}" if random.random() > 0.02 else None,
                "amount": round(random.uniform(10, 500), 2) if random.random() > 0.03 else -10.0,
                "status": random.choice(statuses) if random.random() > 0.02 else "invalid",
                "order_date": check_date,
            }
            # Introduce occasional duplicates
            if random.random() < 0.01 and data:
                row["order_id"] = data[-1]["order_id"]
            data.append(row)

        scores = calculate_quality_scores(data, check_date)
        daily_scores.append(scores)

        dims = {d.name: d.score for d in scores["dimensions"]}
        print(f"  {check_date:<12} {scores['record_count']:>8} {scores['overall_score']:>8.4f} "
              f"{dims['Completeness']:>8.4f} {dims['Uniqueness']:>8.4f} "
              f"{dims['Validity']:>8.4f} {dims['Accuracy']:>8.4f} "
              f"{dims['Consistency']:>8.4f} {dims['Timeliness']:>8.4f}")

    # Trend analysis
    overall_trend = [s["overall_score"] for s in daily_scores]
    avg_score = sum(overall_trend) / len(overall_trend)
    print(f"\n  Average Overall Score: {avg_score:.4f}")
    print(f"  Trend: {'Improving' if overall_trend[-1] > overall_trend[0] else 'Declining'}")

    # ASCII sparkline
    min_s, max_s = min(overall_trend), max(overall_trend)
    if max_s > min_s:
        bars = [int((s - min_s) / (max_s - min_s) * 8) for s in overall_trend]
        chars = " _.-=^*#@"
        sparkline = "".join(chars[b] for b in bars)
    else:
        sparkline = "=" * len(overall_trend)
    print(f"  Sparkline: [{sparkline}]")

    return daily_scores


# ---------------------------------------------------------------------------
# Problem 3: Lineage Tracking
# Design a system that automatically tracks lineage in an ETL pipeline.
# ---------------------------------------------------------------------------

@dataclass
class LineageNode:
    """A node in the data lineage graph (dataset or transformation)."""
    node_id: str
    node_type: str  # "dataset" or "transform"
    name: str
    metadata: dict = field(default_factory=dict)


@dataclass
class LineageEdge:
    """A directed edge in the lineage graph (data flow direction)."""
    source_id: str
    target_id: str
    metadata: dict = field(default_factory=dict)


class LineageTracker:
    """Tracks data lineage through an ETL pipeline.

    Lineage answers three questions:
    1. Where did this data come from? (upstream lineage)
    2. What downstream datasets depend on this? (downstream lineage / impact analysis)
    3. What transformations were applied? (transformation lineage)

    In production, you would use:
    - OpenLineage (open standard for lineage events)
    - Apache Atlas (metadata governance)
    - DataHub (metadata platform by LinkedIn)
    """
    def __init__(self):
        self.nodes: dict[str, LineageNode] = {}
        self.edges: list[LineageEdge] = []
        self.execution_log: list[dict] = []

    def register_dataset(self, dataset_id: str, name: str, **metadata) -> None:
        """Register a dataset node in the lineage graph."""
        self.nodes[dataset_id] = LineageNode(dataset_id, "dataset", name, metadata)

    def register_transform(self, transform_id: str, name: str,
                           inputs: list[str], outputs: list[str],
                           **metadata) -> None:
        """Register a transformation and its input/output datasets."""
        self.nodes[transform_id] = LineageNode(transform_id, "transform", name, metadata)
        for inp in inputs:
            self.edges.append(LineageEdge(inp, transform_id))
        for out in outputs:
            self.edges.append(LineageEdge(transform_id, out))

    def log_execution(self, transform_id: str, status: str,
                      rows_in: int, rows_out: int) -> None:
        """Log a transformation execution event."""
        self.execution_log.append({
            "transform_id": transform_id,
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "rows_in": rows_in,
            "rows_out": rows_out,
        })

    def get_upstream(self, node_id: str, depth: int = 10) -> list[str]:
        """Find all upstream dependencies of a node (backward lineage)."""
        visited = set()
        queue = [node_id]
        result = []
        while queue and depth > 0:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            for edge in self.edges:
                if edge.target_id == current and edge.source_id not in visited:
                    result.append(edge.source_id)
                    queue.append(edge.source_id)
            depth -= 1
        return result

    def get_downstream(self, node_id: str, depth: int = 10) -> list[str]:
        """Find all downstream dependents of a node (impact analysis)."""
        visited = set()
        queue = [node_id]
        result = []
        while queue and depth > 0:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            for edge in self.edges:
                if edge.source_id == current and edge.target_id not in visited:
                    result.append(edge.target_id)
                    queue.append(edge.target_id)
            depth -= 1
        return result

    def display_graph(self) -> None:
        """Display the lineage graph as text."""
        print("\n  Lineage Graph:")
        print(f"  {'Node ID':<25} {'Type':<12} {'Name'}")
        print(f"  {'-'*25} {'-'*12} {'-'*30}")
        for nid, node in self.nodes.items():
            print(f"  {nid:<25} {node.node_type:<12} {node.name}")
        print(f"\n  Edges (data flow):")
        for edge in self.edges:
            src_name = self.nodes[edge.source_id].name if edge.source_id in self.nodes else edge.source_id
            tgt_name = self.nodes[edge.target_id].name if edge.target_id in self.nodes else edge.target_id
            print(f"    {src_name} -> {tgt_name}")


def problem3_lineage_tracking():
    """Build and query a lineage graph for a sample ETL pipeline."""
    tracker = LineageTracker()

    # Register source datasets
    tracker.register_dataset("src_orders", "raw_orders", source="postgres", schema="public")
    tracker.register_dataset("src_customers", "raw_customers", source="postgres", schema="public")
    tracker.register_dataset("src_products", "raw_products", source="postgres", schema="public")

    # Register transformations and intermediate/final datasets
    tracker.register_dataset("stg_orders", "stg_orders", layer="staging")
    tracker.register_dataset("stg_customers", "stg_customers", layer="staging")
    tracker.register_dataset("stg_products", "stg_products", layer="staging")
    tracker.register_dataset("fct_sales", "fct_sales", layer="mart")
    tracker.register_dataset("dim_customers", "dim_customers", layer="mart")
    tracker.register_dataset("rpt_daily_sales", "rpt_daily_sales", layer="report")

    tracker.register_transform(
        "t_clean_orders", "Clean Orders",
        inputs=["src_orders"], outputs=["stg_orders"],
        logic="Filter nulls, cast types, deduplicate",
    )
    tracker.register_transform(
        "t_clean_customers", "Clean Customers",
        inputs=["src_customers"], outputs=["stg_customers"],
        logic="Trim whitespace, standardize names",
    )
    tracker.register_transform(
        "t_clean_products", "Clean Products",
        inputs=["src_products"], outputs=["stg_products"],
        logic="Convert prices, handle NULL categories",
    )
    tracker.register_transform(
        "t_build_fct_sales", "Build fct_sales",
        inputs=["stg_orders", "stg_products"], outputs=["fct_sales"],
        logic="Join orders+products, calculate revenue",
    )
    tracker.register_transform(
        "t_build_dim_customers", "Build dim_customers",
        inputs=["stg_customers"], outputs=["dim_customers"],
        logic="SCD Type 2 merge",
    )
    tracker.register_transform(
        "t_daily_report", "Daily Sales Report",
        inputs=["fct_sales", "dim_customers"], outputs=["rpt_daily_sales"],
        logic="Aggregate sales by date and customer segment",
    )

    # Log some executions
    tracker.log_execution("t_clean_orders", "success", rows_in=10000, rows_out=9800)
    tracker.log_execution("t_build_fct_sales", "success", rows_in=9800, rows_out=9800)
    tracker.log_execution("t_daily_report", "success", rows_in=9800, rows_out=30)

    # Display the graph
    tracker.display_graph()

    # Lineage queries
    print("\n  --- Lineage Queries ---")

    # Upstream of fct_sales: where does the data come from?
    upstream = tracker.get_upstream("fct_sales")
    print(f"\n  Upstream of fct_sales:")
    for uid in upstream:
        node = tracker.nodes.get(uid)
        print(f"    <- {node.name} ({node.node_type})" if node else f"    <- {uid}")

    # Downstream of raw_orders: what does changing this table affect?
    downstream = tracker.get_downstream("src_orders")
    print(f"\n  Downstream of raw_orders (impact analysis):")
    for did in downstream:
        node = tracker.nodes.get(did)
        print(f"    -> {node.name} ({node.node_type})" if node else f"    -> {did}")

    # Execution history
    print(f"\n  Recent Execution Log:")
    for log in tracker.execution_log:
        print(f"    {log['transform_id']}: {log['status']} "
              f"({log['rows_in']} in -> {log['rows_out']} out)")

    return tracker


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Problem 1: Great Expectations (Order Data Suite)")
    print("=" * 70)
    problem1_great_expectations()

    print()
    print("=" * 70)
    print("Problem 2: Quality Dashboard Pipeline")
    print("=" * 70)
    problem2_quality_dashboard()

    print()
    print("=" * 70)
    print("Problem 3: Lineage Tracking System")
    print("=" * 70)
    problem3_lineage_tracking()
