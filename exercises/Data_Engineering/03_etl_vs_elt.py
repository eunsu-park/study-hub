"""
Exercise Solutions: Lesson 03 - ETL vs ELT

Covers:
  - Problem 1: ETL vs ELT Selection (two scenarios)
  - Problem 2: ELT SQL Writing (daily sales aggregation)
"""

from datetime import datetime, date
import random


# ---------------------------------------------------------------------------
# Problem 1: ETL vs ELT Selection
# Choose between ETL and ELT for the following situations.
# ---------------------------------------------------------------------------

def etl_vs_elt_selection() -> list[dict]:
    """Analyze two scenarios and justify ETL vs ELT choice.

    Decision framework:
      ETL when: data must be cleaned/masked BEFORE landing in the warehouse
                (e.g., PII, compliance), or the target system has limited
                compute power.
      ELT when: the target warehouse has strong compute (BigQuery, Snowflake,
                Redshift), data volume is large, and analysts need raw data.
    """
    scenarios = [
        {
            "scenario": "Loading 100GB of daily log data to BigQuery",
            "recommendation": "ELT",
            "reasoning": (
                "BigQuery is a massively parallel warehouse optimized for "
                "large-scale SQL transformations. Loading the raw 100GB "
                "directly (Extract-Load) and then transforming with dbt/SQL "
                "inside BigQuery leverages its distributed compute engine. "
                "Pre-transforming 100GB on an intermediate server would be "
                "slow and expensive. ELT also preserves the raw data in the "
                "warehouse, enabling ad-hoc re-processing without re-ingestion."
            ),
        },
        {
            "scenario": "Processing customer data containing personal information",
            "recommendation": "ETL",
            "reasoning": (
                "Personal information (PII) such as names, emails, and SSNs "
                "must be masked or tokenized BEFORE landing in the warehouse "
                "to comply with GDPR / CCPA. If raw PII enters the warehouse, "
                "every analyst has potential access, violating least-privilege. "
                "ETL transforms (encrypt, hash, or redact) the sensitive "
                "fields on an intermediate server, so only safe data arrives "
                "at the destination. The raw PII is never stored outside the "
                "source system."
            ),
        },
    ]

    print("ETL vs ELT Selection Analysis")
    print("=" * 60)
    for s in scenarios:
        print(f"\nScenario     : {s['scenario']}")
        print(f"Recommendation: {s['recommendation']}")
        print(f"Reasoning    : {s['reasoning']}")
    return scenarios


# ---------------------------------------------------------------------------
# Problem 2: ELT SQL Writing
# Write ELT SQL (simulated in Python) to create a daily sales aggregation
# table from raw table raw_orders.
# ---------------------------------------------------------------------------

def generate_raw_orders(n: int = 200) -> list[dict]:
    """Generate synthetic raw_orders data."""
    categories = ["Electronics", "Clothing", "Books", "Food", "Home"]
    statuses = ["completed", "completed", "completed", "cancelled", "returned"]
    orders = []
    base_date = date(2024, 11, 1)
    for i in range(n):
        day_offset = random.randint(0, 29)
        order_date = date(2024, 11, 1 + day_offset % 28)
        orders.append({
            "order_id": i + 1,
            "order_date": order_date.isoformat(),
            "category": random.choice(categories),
            "amount": round(random.uniform(5, 500), 2),
            "quantity": random.randint(1, 10),
            "status": random.choice(statuses),
        })
    return orders


def elt_daily_sales_aggregation(raw_orders: list[dict]) -> list[dict]:
    """Simulate the ELT transformation that would run as SQL in the warehouse.

    Equivalent SQL (e.g., for BigQuery / dbt):

        CREATE OR REPLACE TABLE daily_sales_summary AS
        SELECT
            order_date,
            category,
            COUNT(*)                                     AS total_orders,
            SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END)
                                                         AS completed_orders,
            SUM(CASE WHEN status = 'completed' THEN amount ELSE 0 END)
                                                         AS total_revenue,
            AVG(CASE WHEN status = 'completed' THEN amount END)
                                                         AS avg_order_value,
            SUM(CASE WHEN status = 'completed' THEN quantity ELSE 0 END)
                                                         AS total_quantity
        FROM raw_orders
        GROUP BY order_date, category
        ORDER BY order_date, category;

    Why ELT?
    - The raw_orders table is already loaded in the warehouse.
    - The transformation is a straightforward GROUP BY that the warehouse
      engine parallelizes efficiently.
    - Keeping raw_orders intact allows re-running this SQL at any time
      with different logic (e.g., including 'returned' orders).
    """
    # GROUP BY (order_date, category)
    agg: dict[tuple[str, str], dict] = {}
    for o in raw_orders:
        key = (o["order_date"], o["category"])
        if key not in agg:
            agg[key] = {
                "order_date": o["order_date"],
                "category": o["category"],
                "total_orders": 0,
                "completed_orders": 0,
                "total_revenue": 0.0,
                "avg_order_value": 0.0,
                "total_quantity": 0,
                "_completed_amounts": [],
            }
        row = agg[key]
        row["total_orders"] += 1
        if o["status"] == "completed":
            row["completed_orders"] += 1
            row["total_revenue"] += o["amount"]
            row["total_quantity"] += o["quantity"]
            row["_completed_amounts"].append(o["amount"])

    results = []
    for row in agg.values():
        amounts = row.pop("_completed_amounts")
        row["total_revenue"] = round(row["total_revenue"], 2)
        row["avg_order_value"] = (
            round(sum(amounts) / len(amounts), 2) if amounts else 0.0
        )
        results.append(row)

    results.sort(key=lambda r: (r["order_date"], r["category"]))
    return results


def display_aggregation(results: list[dict], max_rows: int = 15) -> None:
    """Pretty-print the daily sales aggregation table."""
    header = (
        f"{'Date':<12} {'Category':<14} {'Total':>6} {'Done':>5} "
        f"{'Revenue':>10} {'AvgOV':>8} {'Qty':>5}"
    )
    print(header)
    print("-" * len(header))
    for i, r in enumerate(results):
        if i >= max_rows:
            print(f"  ... ({len(results) - max_rows} more rows)")
            break
        print(
            f"{r['order_date']:<12} {r['category']:<14} "
            f"{r['total_orders']:>6} {r['completed_orders']:>5} "
            f"${r['total_revenue']:>9.2f} ${r['avg_order_value']:>7.2f} "
            f"{r['total_quantity']:>5}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Problem 1: ETL vs ELT Selection")
    print("=" * 70)
    etl_vs_elt_selection()

    print()
    print("=" * 70)
    print("Problem 2: ELT SQL Writing - Daily Sales Aggregation")
    print("=" * 70)
    raw = generate_raw_orders(200)
    print(f"\nRaw orders loaded: {len(raw)}")
    print(f"\nExecuting ELT transformation (GROUP BY order_date, category)...\n")
    agg = elt_daily_sales_aggregation(raw)
    display_aggregation(agg)
    print(f"\nTotal aggregated rows: {len(agg)}")
