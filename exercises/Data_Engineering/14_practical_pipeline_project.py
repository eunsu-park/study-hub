"""
Exercise Solutions: Lesson 14 - Practical Pipeline Project

Covers:
  - Problem 1: Extend Pipeline (Kafka streaming + low-stock alerts)
  - Problem 2: Quality Dashboard (daily quality scores + visualization)
  - Problem 3: Cost Optimization (Spark partition & resource tuning)

Note: Pure Python simulation of integrated pipeline concepts.
"""

import random
import time
import threading
from collections import defaultdict
from datetime import datetime, timedelta
from queue import Queue


# ---------------------------------------------------------------------------
# Problem 1: Extend Pipeline
# Add a streaming pipeline that processes real-time inventory events
# from Kafka and sends low-stock alerts.
# ---------------------------------------------------------------------------

class InventoryEvent:
    """Represents an inventory change event from Kafka."""
    def __init__(self, product_id: str, product_name: str,
                 warehouse: str, quantity_change: int, current_stock: int):
        self.product_id = product_id
        self.product_name = product_name
        self.warehouse = warehouse
        self.quantity_change = quantity_change
        self.current_stock = current_stock
        self.timestamp = datetime.now().isoformat()


class LowStockAlertSystem:
    """Streaming pipeline that monitors inventory and triggers alerts.

    Architecture:
      [POS / Warehouse] -> Kafka(inventory_events) -> [StreamProcessor] -> Kafka(alerts)
                                                            |
                                                    [Alert Service] -> Slack/Email

    This simulates the Kafka consumer that processes inventory events,
    checks against thresholds, and produces alert messages.
    """
    def __init__(self, low_stock_threshold: int = 10, critical_threshold: int = 3):
        self.low_stock_threshold = low_stock_threshold
        self.critical_threshold = critical_threshold
        self.alerts: list[dict] = []
        self.stock_state: dict[str, dict] = {}  # product_id -> current state

    def process_event(self, event: InventoryEvent) -> dict | None:
        """Process a single inventory event and check thresholds.

        In a real Kafka Streams / Faust application:
            @app.agent(inventory_topic)
            async def process_inventory(stream):
                async for event in stream:
                    if event.current_stock < LOW_THRESHOLD:
                        await alerts_topic.send(value=create_alert(event))
        """
        self.stock_state[event.product_id] = {
            "product_name": event.product_name,
            "warehouse": event.warehouse,
            "current_stock": event.current_stock,
            "last_updated": event.timestamp,
        }

        alert = None
        if event.current_stock <= self.critical_threshold:
            alert = {
                "severity": "CRITICAL",
                "product_id": event.product_id,
                "product_name": event.product_name,
                "warehouse": event.warehouse,
                "current_stock": event.current_stock,
                "threshold": self.critical_threshold,
                "message": f"CRITICAL: {event.product_name} at {event.current_stock} units "
                           f"in {event.warehouse} (threshold: {self.critical_threshold})",
                "timestamp": event.timestamp,
            }
        elif event.current_stock <= self.low_stock_threshold:
            alert = {
                "severity": "WARNING",
                "product_id": event.product_id,
                "product_name": event.product_name,
                "warehouse": event.warehouse,
                "current_stock": event.current_stock,
                "threshold": self.low_stock_threshold,
                "message": f"WARNING: {event.product_name} at {event.current_stock} units "
                           f"in {event.warehouse} (threshold: {self.low_stock_threshold})",
                "timestamp": event.timestamp,
            }

        if alert:
            self.alerts.append(alert)

        return alert


def problem1_streaming_alerts():
    """Simulate a Kafka streaming pipeline for inventory monitoring."""
    system = LowStockAlertSystem(low_stock_threshold=10, critical_threshold=3)

    # Simulate inventory events
    products = [
        ("P001", "Laptop Pro"),
        ("P002", "Wireless Mouse"),
        ("P003", "USB-C Hub"),
        ("P004", "Monitor Stand"),
        ("P005", "Keyboard"),
    ]
    warehouses = ["WH-East", "WH-West", "WH-Central"]

    print("\n  Streaming Inventory Events:")
    print(f"  {'Time':<12} {'Product':<16} {'Warehouse':<12} {'Change':>7} {'Stock':>6} {'Alert'}")
    print(f"  {'-'*12} {'-'*16} {'-'*12} {'-'*7} {'-'*6} {'-'*20}")

    # Simulate 20 events with decreasing stock
    stock_levels: dict[str, int] = {p[0]: random.randint(15, 30) for p in products}

    for i in range(20):
        pid, pname = random.choice(products)
        wh = random.choice(warehouses)
        change = -random.randint(1, 8)
        stock_levels[pid] = max(0, stock_levels[pid] + change)

        event = InventoryEvent(pid, pname, wh, change, stock_levels[pid])
        alert = system.process_event(event)

        alert_str = alert["severity"] if alert else ""
        print(f"  {datetime.now().strftime('%H:%M:%S.%f')[:12]:<12} {pname:<16} {wh:<12} "
              f"{change:>+7} {stock_levels[pid]:>6} {alert_str}")

    # Summary
    print(f"\n  Alerts Generated: {len(system.alerts)}")
    critical = [a for a in system.alerts if a["severity"] == "CRITICAL"]
    warnings = [a for a in system.alerts if a["severity"] == "WARNING"]
    print(f"    CRITICAL: {len(critical)}")
    print(f"    WARNING:  {len(warnings)}")

    if critical:
        print(f"\n  Critical Alerts:")
        for a in critical[:5]:
            print(f"    {a['message']}")

    return system


# ---------------------------------------------------------------------------
# Problem 2: Quality Dashboard
# Visualize daily data quality scores in a dashboard format.
# ---------------------------------------------------------------------------

def calculate_pipeline_quality(date_str: str) -> dict:
    """Calculate quality metrics for all pipeline stages."""
    stages = {
        "bronze_ingestion": {
            "completeness": random.uniform(0.95, 1.0),
            "freshness_minutes": random.randint(1, 15),
            "row_count": random.randint(9000, 11000),
            "error_rate": random.uniform(0, 0.02),
        },
        "silver_transformation": {
            "completeness": random.uniform(0.97, 1.0),
            "dedup_rate": random.uniform(0.01, 0.05),
            "schema_valid": random.uniform(0.98, 1.0),
            "row_count": random.randint(8500, 10500),
        },
        "gold_aggregation": {
            "completeness": 1.0,
            "accuracy": random.uniform(0.96, 1.0),
            "timeliness_minutes": random.randint(5, 45),
            "row_count": random.randint(20, 50),
        },
    }
    overall = (
        stages["bronze_ingestion"]["completeness"] * 0.3 +
        stages["silver_transformation"]["schema_valid"] * 0.4 +
        stages["gold_aggregation"]["accuracy"] * 0.3
    )
    return {"date": date_str, "overall_score": round(overall, 4), "stages": stages}


def problem2_quality_dashboard():
    """Display a multi-day quality dashboard with ASCII visualization."""
    print("\n  === Data Quality Dashboard ===\n")

    days = 7
    base = datetime(2024, 11, 1)
    daily_metrics = []
    for d in range(days):
        dt = (base + timedelta(days=d)).strftime("%Y-%m-%d")
        daily_metrics.append(calculate_pipeline_quality(dt))

    # Overall score trend
    print("  Overall Quality Score Trend:")
    print(f"  {'Date':<12} {'Score':>7} {'Bar'}")
    print(f"  {'-'*12} {'-'*7} {'-'*40}")
    for m in daily_metrics:
        bar_len = int(m["overall_score"] * 40)
        bar = "#" * bar_len
        indicator = "OK" if m["overall_score"] >= 0.98 else "WARN" if m["overall_score"] >= 0.95 else "ALERT"
        print(f"  {m['date']:<12} {m['overall_score']:>7.4f} |{bar:<40}| {indicator}")

    # Per-stage breakdown for latest day
    latest = daily_metrics[-1]
    print(f"\n  Stage Breakdown ({latest['date']}):")
    for stage_name, metrics in latest["stages"].items():
        print(f"\n    {stage_name}:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"      {k:<20} {v:.4f}")
            else:
                print(f"      {k:<20} {v}")

    # SLA check
    print(f"\n  SLA Checks:")
    sla_rules = [
        ("Bronze freshness < 10 min", latest["stages"]["bronze_ingestion"]["freshness_minutes"] < 10),
        ("Silver schema valid > 99%", latest["stages"]["silver_transformation"]["schema_valid"] > 0.99),
        ("Gold accuracy > 98%", latest["stages"]["gold_aggregation"]["accuracy"] > 0.98),
        ("Overall score > 97%", latest["overall_score"] > 0.97),
    ]
    for rule, passed in sla_rules:
        status = "PASS" if passed else "FAIL"
        print(f"    [{status}] {rule}")

    return daily_metrics


# ---------------------------------------------------------------------------
# Problem 3: Cost Optimization
# Optimize Spark partition count and resource settings.
# ---------------------------------------------------------------------------

def problem3_cost_optimization():
    """Demonstrate Spark partition and resource optimization strategies.

    Key principles:
    1. Partition count = 2-3x the number of cores
    2. Each partition should be 128MB-200MB for optimal performance
    3. Executor memory: 4-8GB per executor, leave headroom for overhead
    4. Too many small partitions = scheduling overhead
    5. Too few large partitions = memory pressure and poor parallelism
    """

    # Scenario: Processing 500GB of data
    data_size_gb = 500
    data_size_mb = data_size_gb * 1024

    cluster_configs = [
        {
            "name": "Before Optimization (wasteful)",
            "executors": 20,
            "cores_per_executor": 5,
            "memory_per_executor_gb": 16,
            "partitions": 200,  # default or poorly set
            "partition_size_mb": data_size_mb / 200,
            "notes": "200 partitions for 100 cores -> 2 waves. Partition size=2.5GB is too large.",
        },
        {
            "name": "After Optimization (efficient)",
            "executors": 10,
            "cores_per_executor": 4,
            "memory_per_executor_gb": 8,
            "partitions": 300,  # ~3x cores
            "partition_size_mb": data_size_mb / 300,
            "notes": "300 partitions for 40 cores -> ~8 waves. Partition size=~1.7GB still large but manageable.",
        },
        {
            "name": "Optimal Configuration",
            "executors": 10,
            "cores_per_executor": 4,
            "memory_per_executor_gb": 8,
            "partitions": 3200,  # ~128MB per partition
            "partition_size_mb": data_size_mb / 3200,
            "notes": "3200 partitions -> ~160MB each (optimal range). 80 waves with 40 cores.",
        },
    ]

    print(f"\n  Scenario: Processing {data_size_gb}GB dataset\n")

    for config in cluster_configs:
        total_cores = config["executors"] * config["cores_per_executor"]
        total_memory = config["executors"] * config["memory_per_executor_gb"]
        waves = config["partitions"] / total_cores
        cost_units = config["executors"] * config["memory_per_executor_gb"]

        print(f"  --- {config['name']} ---")
        print(f"    Executors      : {config['executors']} x {config['cores_per_executor']} cores = {total_cores} total cores")
        print(f"    Memory         : {config['executors']} x {config['memory_per_executor_gb']}GB = {total_memory}GB total")
        print(f"    Partitions     : {config['partitions']}")
        print(f"    Partition size : {config['partition_size_mb']:.0f}MB")
        print(f"    Waves          : {waves:.1f}")
        print(f"    Cost units     : {cost_units} (executors * GB)")
        print(f"    Notes          : {config['notes']}")
        print()

    # Spark configuration recommendations
    print("  Spark Configuration Recommendations:")
    print("  " + "-" * 55)
    configs = [
        ("spark.sql.shuffle.partitions", "200 -> auto (AQE)", "Let AQE auto-tune after shuffle"),
        ("spark.sql.adaptive.enabled", "true", "Enable Adaptive Query Execution"),
        ("spark.sql.adaptive.coalescePartitions.enabled", "true", "Auto-merge small partitions"),
        ("spark.sql.adaptive.skewJoin.enabled", "true", "Auto-handle skewed joins"),
        ("spark.sql.files.maxPartitionBytes", "128MB", "Target partition size for file reads"),
        ("spark.sql.files.openCostInBytes", "4MB", "Cost of opening a file (affects split decisions)"),
        ("spark.executor.memory", "8g", "Per-executor memory (4-8GB is sweet spot)"),
        ("spark.executor.cores", "4", "Cores per executor (4-5 optimal)"),
        ("spark.executor.memoryOverhead", "1g", "Off-heap memory for Python UDFs, etc."),
        ("spark.dynamicAllocation.enabled", "true", "Scale executors up/down based on load"),
    ]

    print(f"  {'Config Key':<50} {'Value':<20} {'Reason'}")
    print(f"  {'-'*50} {'-'*20} {'-'*30}")
    for key, value, reason in configs:
        print(f"  {key:<50} {value:<20} {reason}")

    # Cost savings summary
    before = cluster_configs[0]
    after = cluster_configs[2]
    before_cost = before["executors"] * before["memory_per_executor_gb"]
    after_cost = after["executors"] * after["memory_per_executor_gb"]
    savings_pct = (1 - after_cost / before_cost) * 100

    print(f"\n  Cost Savings Summary:")
    print(f"    Before: {before['executors']} executors x {before['memory_per_executor_gb']}GB = {before_cost} cost units")
    print(f"    After:  {after['executors']} executors x {after['memory_per_executor_gb']}GB = {after_cost} cost units")
    print(f"    Savings: {savings_pct:.0f}% reduction in cluster resources")
    print(f"    Performance: Better due to optimal partition sizing (160MB vs 2.5GB)")

    return cluster_configs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Problem 1: Extend Pipeline (Kafka Streaming + Low-Stock Alerts)")
    print("=" * 70)
    problem1_streaming_alerts()

    print()
    print("=" * 70)
    print("Problem 2: Quality Dashboard")
    print("=" * 70)
    problem2_quality_dashboard()

    print()
    print("=" * 70)
    print("Problem 3: Cost Optimization (Spark Partitions & Resources)")
    print("=" * 70)
    problem3_cost_optimization()
