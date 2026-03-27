"""
Exercise Solutions: Lesson 01 - Data Engineering Overview

Covers:
  - Problem 1: Pipeline Design (daily sales report)
  - Problem 2: Batch vs Streaming Selection
"""

from datetime import datetime, timedelta
import random
import json


# ---------------------------------------------------------------------------
# Problem 1: Pipeline Design
# Design a pipeline that generates daily sales reports for an online
# shopping mall. Implement extract, transform, and load phases.
# ---------------------------------------------------------------------------

class DailySalesReportPipeline:
    """A simulated ETL pipeline that produces a daily sales report.

    Why three explicit phases?
    - Extract isolates the data source (could be a DB, API, or file).
    - Transform applies business logic independently of I/O.
    - Load writes output to the target (warehouse, file, etc.).
    Separating concerns makes each phase testable and replaceable.
    """

    def __init__(self, report_date: str | None = None):
        self.report_date = report_date or datetime.now().strftime("%Y-%m-%d")
        self.raw_orders: list[dict] = []
        self.report: dict = {}

    # -- Extract phase -------------------------------------------------------
    def extract(self) -> list[dict]:
        """Simulate extracting order, product, and customer data for one day.

        In production this would query the OLTP database or read from a data
        lake landing zone.
        """
        categories = ["Electronics", "Clothing", "Books", "Food", "Home"]
        customers = [f"customer_{i}" for i in range(1, 21)]

        orders = []
        for order_id in range(1, random.randint(50, 150)):
            orders.append({
                "order_id": f"ORD-{self.report_date}-{order_id:04d}",
                "customer_id": random.choice(customers),
                "category": random.choice(categories),
                "amount": round(random.uniform(10, 500), 2),
                "quantity": random.randint(1, 5),
                "order_date": self.report_date,
                "status": random.choice(["completed", "completed", "completed", "cancelled", "returned"]),
            })
        self.raw_orders = orders
        print(f"[Extract] Loaded {len(orders)} orders for {self.report_date}")
        return orders

    # -- Transform phase -----------------------------------------------------
    def transform(self) -> dict:
        """Aggregate sales by category and compute KPIs.

        Business rules:
        - Only include orders with status='completed'.
        - Report total revenue, order count, and average order value per category.
        """
        completed = [o for o in self.raw_orders if o["status"] == "completed"]

        # Per-category aggregation
        category_stats: dict[str, dict] = {}
        for order in completed:
            cat = order["category"]
            if cat not in category_stats:
                category_stats[cat] = {"revenue": 0.0, "order_count": 0, "quantities": 0}
            category_stats[cat]["revenue"] += order["amount"]
            category_stats[cat]["order_count"] += 1
            category_stats[cat]["quantities"] += order["quantity"]

        for cat, stats in category_stats.items():
            stats["revenue"] = round(stats["revenue"], 2)
            stats["avg_order_value"] = (
                round(stats["revenue"] / stats["order_count"], 2)
                if stats["order_count"] > 0 else 0.0
            )

        self.report = {
            "report_date": self.report_date,
            "total_orders": len(self.raw_orders),
            "completed_orders": len(completed),
            "cancelled_orders": len([o for o in self.raw_orders if o["status"] == "cancelled"]),
            "returned_orders": len([o for o in self.raw_orders if o["status"] == "returned"]),
            "total_revenue": round(sum(o["amount"] for o in completed), 2),
            "category_breakdown": category_stats,
        }
        print(f"[Transform] Processed {len(completed)} completed orders -> report ready")
        return self.report

    # -- Load phase ----------------------------------------------------------
    def load(self) -> str:
        """Write the report to a JSON file (simulating warehouse load)."""
        filename = f"daily_sales_report_{self.report_date}.json"
        # In production: INSERT INTO report_table ...
        output = json.dumps(self.report, indent=2)
        print(f"[Load] Report written to {filename}")
        print(output[:500])
        return output

    # -- Orchestrate ---------------------------------------------------------
    def run(self) -> dict:
        """Execute the full pipeline: extract -> transform -> load."""
        self.extract()
        self.transform()
        self.load()
        return self.report


# ---------------------------------------------------------------------------
# Problem 2: Batch vs Streaming Selection
# Choose the appropriate approach for each scenario and explain why.
# ---------------------------------------------------------------------------

def batch_vs_streaming_analysis() -> list[dict]:
    """Analyze three scenarios and recommend batch or streaming processing.

    Key decision factors:
    - Latency requirement: seconds (streaming) vs hours/days (batch)
    - Data volume per event: high aggregate (batch) vs individual events (streaming)
    - Business impact of delay: high urgency favours streaming
    """
    scenarios = [
        {
            "scenario": "Daily sales report generation",
            "recommendation": "Batch",
            "reasoning": (
                "Sales reports are consumed once per day by business analysts. "
                "There is no latency requirement shorter than 24 hours, and the "
                "workload involves heavy aggregation over a full day of data. "
                "Batch processing (e.g., a nightly Airflow DAG running Spark) is "
                "the natural fit: cheaper, simpler, and easier to debug."
            ),
        },
        {
            "scenario": "Real-time low stock alerts",
            "recommendation": "Streaming",
            "reasoning": (
                "Inventory going below a threshold requires immediate action "
                "(reorder, disable purchase button). A delay of minutes can cause "
                "overselling. A Kafka-based streaming pipeline consuming inventory "
                "change events and triggering alerts within seconds is the right "
                "approach. The per-event logic is lightweight (threshold check)."
            ),
        },
        {
            "scenario": "Monthly customer segmentation",
            "recommendation": "Batch",
            "reasoning": (
                "Segmentation uses ML clustering over an entire month of customer "
                "behaviour data. The output is consumed by marketing once per month. "
                "There is no benefit to processing this in real time. A monthly "
                "Spark batch job that reads aggregated features and writes segment "
                "labels to the warehouse is optimal."
            ),
        },
    ]

    print("=" * 70)
    print("Batch vs Streaming Selection Analysis")
    print("=" * 70)
    for s in scenarios:
        print(f"\nScenario : {s['scenario']}")
        print(f"Choice   : {s['recommendation']}")
        print(f"Reasoning: {s['reasoning']}")
    print()
    return scenarios


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Problem 1: Daily Sales Report Pipeline")
    print("=" * 70)
    pipeline = DailySalesReportPipeline(report_date="2024-11-15")
    pipeline.run()

    print()
    batch_vs_streaming_analysis()
