"""
Cloud Monitoring and Alerting Simulation

Demonstrates cloud observability concepts:
- Metric collection with time-series data (CloudWatch / Stackdriver style)
- Threshold-based alerting with configurable evaluation periods
- Simple anomaly detection using rolling statistics
- Cost optimization recommendations based on utilization patterns
- Alert state machine (OK -> ALARM -> OK transitions)

No cloud account required -- all behavior is simulated locally.
"""

import random
import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class AlarmState(Enum):
    OK = "OK"                         # Metric within normal range
    ALARM = "ALARM"                   # Threshold breached
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"  # Not enough data points yet


class ComparisonOperator(Enum):
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_OR_EQUAL = ">="
    LESS_OR_EQUAL = "<="


@dataclass
class MetricDataPoint:
    """A single metric data point with a timestamp.
    Cloud monitoring services collect metrics at regular intervals
    (typically 1 minute for basic, 1 second for detailed monitoring)."""
    timestamp: int      # Simulated epoch seconds
    value: float
    unit: str = ""


@dataclass
class MetricSeries:
    """A time series of metric values for a specific resource.
    Metrics are the foundation of observability -- without them,
    you're flying blind in production."""
    namespace: str          # e.g., "AWS/EC2", "AWS/RDS"
    metric_name: str        # e.g., "CPUUtilization", "NetworkIn"
    dimensions: Dict[str, str]  # e.g., {"InstanceId": "i-abc123"}
    unit: str = "%"
    data_points: List[MetricDataPoint] = field(default_factory=list)

    @property
    def resource_id(self) -> str:
        return f"{self.namespace}/{self.metric_name}/{self.dimensions}"

    def add_point(self, timestamp: int, value: float) -> None:
        self.data_points.append(MetricDataPoint(timestamp, value, self.unit))

    def get_statistics(self, last_n: int = 10) -> dict:
        """Compute statistics over the most recent N data points.
        CloudWatch uses 'periods' and 'statistics' to aggregate raw data."""
        recent = [dp.value for dp in self.data_points[-last_n:]]
        if not recent:
            return {"count": 0}
        return {
            "count": len(recent),
            "min": round(min(recent), 2),
            "max": round(max(recent), 2),
            "average": round(sum(recent) / len(recent), 2),
            "sum": round(sum(recent), 2),
        }


@dataclass
class AlarmConfig:
    """Configuration for a CloudWatch-style alarm.
    The evaluation_periods and datapoints_to_alarm parameters prevent
    transient spikes from triggering false alarms (reducing alert fatigue)."""
    alarm_name: str
    metric_name: str
    namespace: str
    threshold: float
    comparison: ComparisonOperator
    evaluation_periods: int = 3    # How many consecutive periods to evaluate
    datapoints_to_alarm: int = 2   # How many must breach to trigger alarm
    description: str = ""


class CloudWatchAlarm:
    """Simulates a CloudWatch alarm with state transitions.

    State machine: INSUFFICIENT_DATA -> OK or ALARM
                   OK <-> ALARM (based on metric evaluation)

    Key concept: alarms evaluate over multiple periods to avoid false positives.
    A single spike should not page your on-call engineer at 3 AM."""

    def __init__(self, config: AlarmConfig):
        self.config = config
        self.state = AlarmState.INSUFFICIENT_DATA
        self.state_reason = "Waiting for data"
        self.history: List[Tuple[int, AlarmState]] = []
        self._breach_count = 0
        self._ok_count = 0

    def _check_threshold(self, value: float) -> bool:
        """Check if a value breaches the configured threshold."""
        ops = {
            ComparisonOperator.GREATER_THAN: lambda v, t: v > t,
            ComparisonOperator.LESS_THAN: lambda v, t: v < t,
            ComparisonOperator.GREATER_OR_EQUAL: lambda v, t: v >= t,
            ComparisonOperator.LESS_OR_EQUAL: lambda v, t: v <= t,
        }
        return ops[self.config.comparison](value, self.config.threshold)

    def evaluate(self, timestamp: int, value: float) -> Optional[str]:
        """Evaluate the alarm against a new data point.
        Returns an alert message if the alarm state changes."""
        breached = self._check_threshold(value)
        old_state = self.state

        if breached:
            self._breach_count += 1
            self._ok_count = 0
        else:
            self._ok_count += 1
            self._breach_count = 0

        # Transition to ALARM if enough consecutive breaches
        if self._breach_count >= self.config.datapoints_to_alarm:
            self.state = AlarmState.ALARM
            self.state_reason = (
                f"Threshold crossed: {value:.1f} {self.config.comparison.value} "
                f"{self.config.threshold} for {self._breach_count} period(s)")
        # Transition back to OK if enough consecutive OK readings
        elif self._ok_count >= self.config.evaluation_periods:
            self.state = AlarmState.OK
            self.state_reason = f"Metric within normal range ({value:.1f})"
        # First valid data -> leave INSUFFICIENT_DATA
        elif self.state == AlarmState.INSUFFICIENT_DATA:
            self.state = AlarmState.OK if not breached else AlarmState.ALARM

        self.history.append((timestamp, self.state))

        # Generate notification on state change
        if self.state != old_state:
            return (f"  ALERT [{self.config.alarm_name}]: "
                    f"{old_state.value} -> {self.state.value} | {self.state_reason}")
        return None


class AnomalyDetector:
    """Simple anomaly detection using rolling mean and standard deviation.
    Real cloud services (CloudWatch Anomaly Detection, GCP) use ML models,
    but the core idea is the same: flag values that deviate significantly
    from historical patterns."""

    def __init__(self, window_size: int = 30, z_threshold: float = 2.5):
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.values: deque = deque(maxlen=window_size)

    def check(self, value: float) -> Optional[dict]:
        """Check if a value is anomalous based on recent history."""
        self.values.append(value)
        if len(self.values) < self.window_size // 2:
            return None  # Not enough history

        mean = sum(self.values) / len(self.values)
        variance = sum((v - mean) ** 2 for v in self.values) / len(self.values)
        std = math.sqrt(variance) if variance > 0 else 0.01

        z_score = (value - mean) / std
        is_anomaly = abs(z_score) > self.z_threshold

        if is_anomaly:
            return {
                "value": round(value, 2),
                "mean": round(mean, 2),
                "std": round(std, 2),
                "z_score": round(z_score, 2),
                "direction": "above" if z_score > 0 else "below",
            }
        return None


def generate_cpu_metrics(hours: int = 24) -> MetricSeries:
    """Generate realistic CPU utilization data with daily patterns.
    Simulates: baseline ~30%, business hours peak ~70%, overnight dip ~15%."""
    series = MetricSeries(
        namespace="AWS/EC2", metric_name="CPUUtilization",
        dimensions={"InstanceId": "i-0abc1234"}, unit="%",
    )
    for minute in range(hours * 60):
        hour = (minute // 60) % 24
        # Daily pattern: peak during business hours (9-17), low at night
        if 9 <= hour <= 17:
            base = 55 + 15 * math.sin((hour - 9) / 8 * math.pi)
        else:
            base = 15
        noise = random.gauss(0, 5)
        # Inject occasional spikes (simulating traffic bursts)
        spike = 30 if random.random() < 0.02 else 0
        value = max(0, min(100, base + noise + spike))
        series.add_point(minute * 60, value)
    return series


def demo_metric_collection():
    """Show metric collection and statistics computation."""
    print("=" * 70)
    print("Metric Collection and Statistics")
    print("=" * 70)

    series = generate_cpu_metrics(hours=24)
    stats = series.get_statistics(last_n=60)  # Last hour
    print(f"\n  Metric: {series.namespace}/{series.metric_name}")
    print(f"  Resource: {series.dimensions}")
    print(f"  Total data points: {len(series.data_points)} (24h at 1-min intervals)")
    print(f"\n  Last hour statistics:")
    for key, val in stats.items():
        print(f"    {key}: {val}")
    print()


def demo_threshold_alerting():
    """Demonstrate threshold-based alerting with state transitions."""
    print("=" * 70)
    print("Threshold-Based Alerting")
    print("=" * 70)

    alarm = CloudWatchAlarm(AlarmConfig(
        alarm_name="HighCPU-WebServer",
        metric_name="CPUUtilization",
        namespace="AWS/EC2",
        threshold=80.0,
        comparison=ComparisonOperator.GREATER_THAN,
        evaluation_periods=3,
        datapoints_to_alarm=2,
        description="Alert when CPU exceeds 80% for 2 of 3 periods",
    ))

    series = generate_cpu_metrics(hours=6)
    alerts = []
    # Evaluate every 5 minutes (aggregated)
    for i in range(0, len(series.data_points), 5):
        chunk = series.data_points[i:i+5]
        avg = sum(dp.value for dp in chunk) / len(chunk)
        timestamp = chunk[0].timestamp
        alert = alarm.evaluate(timestamp, avg)
        if alert:
            alerts.append(alert)

    print(f"\n  Alarm: {alarm.config.alarm_name}")
    print(f"  Threshold: CPU {alarm.config.comparison.value} {alarm.config.threshold}%")
    print(f"  Evaluation: {alarm.config.datapoints_to_alarm} of "
          f"{alarm.config.evaluation_periods} periods")
    print(f"\n  State transitions detected: {len(alerts)}")
    for alert in alerts[:10]:  # Show first 10
        print(alert)
    print()


def demo_anomaly_detection():
    """Show anomaly detection on metric data."""
    print("=" * 70)
    print("Anomaly Detection (Z-Score Method)")
    print("=" * 70)

    detector = AnomalyDetector(window_size=60, z_threshold=2.5)
    series = generate_cpu_metrics(hours=12)

    anomalies = []
    for dp in series.data_points:
        result = detector.check(dp.value)
        if result:
            anomalies.append({"timestamp": dp.timestamp, **result})

    print(f"\n  Analyzed {len(series.data_points)} data points over 12 hours")
    print(f"  Anomalies detected: {len(anomalies)}")
    print(f"  Detection parameters: window=60, z_threshold=2.5")
    print(f"\n  Sample anomalies:")
    for a in anomalies[:8]:
        hour = a["timestamp"] // 3600
        minute = (a["timestamp"] % 3600) // 60
        print(f"    {hour:02d}:{minute:02d} - CPU={a['value']}% "
              f"(mean={a['mean']}%, z={a['z_score']:+.1f}, {a['direction']})")
    print()


def demo_cost_optimization():
    """Generate cost optimization recommendations based on utilization."""
    print("=" * 70)
    print("Cost Optimization Recommendations")
    print("=" * 70)

    # Simulate utilization data for multiple instances
    instances = [
        {"id": "i-web-01", "type": "m5.2xlarge", "cost_hr": 0.384,
         "avg_cpu": random.uniform(5, 15), "avg_mem": random.uniform(10, 20)},
        {"id": "i-web-02", "type": "m5.xlarge", "cost_hr": 0.192,
         "avg_cpu": random.uniform(60, 85), "avg_mem": random.uniform(70, 90)},
        {"id": "i-batch-01", "type": "c5.4xlarge", "cost_hr": 0.680,
         "avg_cpu": random.uniform(2, 8), "avg_mem": random.uniform(5, 10)},
        {"id": "i-db-01", "type": "r5.2xlarge", "cost_hr": 0.504,
         "avg_cpu": random.uniform(30, 50), "avg_mem": random.uniform(80, 95)},
        {"id": "i-dev-01", "type": "t3.large", "cost_hr": 0.0832,
         "avg_cpu": random.uniform(1, 5), "avg_mem": random.uniform(3, 10)},
    ]

    print(f"\n  {'Instance':<16} {'Type':<14} {'CPU%':>6} {'Mem%':>6} "
          f"{'$/month':>10} {'Recommendation':<30}")
    print(f"  {'-'*85}")

    total_monthly = 0
    potential_savings = 0

    for inst in instances:
        monthly = inst["cost_hr"] * 730
        total_monthly += monthly

        # Generate recommendations based on utilization patterns
        if inst["avg_cpu"] < 10 and inst["avg_mem"] < 20:
            rec = "DOWNSIZE or TERMINATE (idle)"
            savings = monthly * 0.7
        elif inst["avg_cpu"] < 30:
            rec = "DOWNSIZE (underutilized)"
            savings = monthly * 0.4
        elif inst["avg_cpu"] > 80:
            rec = "Monitor (high utilization)"
            savings = 0
        else:
            rec = "Right-sized"
            savings = 0

        potential_savings += savings
        print(f"  {inst['id']:<16} {inst['type']:<14} {inst['avg_cpu']:>5.1f} "
              f"{inst['avg_mem']:>5.1f} ${monthly:>9.2f} {rec}")

    print(f"\n  Total monthly cost:      ${total_monthly:>10.2f}")
    print(f"  Potential monthly savings: ${potential_savings:>10.2f} "
          f"({potential_savings/total_monthly*100:.0f}%)")
    print(f"\n  Additional recommendations:")
    print(f"    - Use Reserved Instances for steady-state workloads (save ~40%)")
    print(f"    - Use Spot Instances for batch processing (save ~60-90%)")
    print(f"    - Enable auto-scaling to match capacity to demand")
    print(f"    - Schedule dev/staging instances to stop outside business hours")
    print()


if __name__ == "__main__":
    random.seed(42)
    demo_metric_collection()
    demo_threshold_alerting()
    demo_anomaly_detection()
    demo_cost_optimization()
