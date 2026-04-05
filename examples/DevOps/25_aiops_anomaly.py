#!/usr/bin/env python3
"""Example: AIOps Anomaly Detection — Statistical & ML-Based Alerting

Demonstrates AIOps concepts: Z-score and IQR anomaly detection, seasonal
decomposition, dynamic thresholds, alert correlation/deduplication, and
automated root-cause ranking.
Related lesson: 27_AIOps_Anomaly_Detection.md
"""

# =============================================================================
# WHY AIOps ANOMALY DETECTION?
# Static thresholds ("alert if CPU > 80%") generate floods of false positives
# and miss subtle degradation. AIOps uses statistical methods and ML to learn
# normal behavior, detect anomalies adaptively, correlate related alerts, and
# suggest root causes — moving from reactive to predictive operations.
# =============================================================================

import math
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


# =============================================================================
# 1. STATISTICAL ANOMALY DETECTORS
# =============================================================================

@dataclass
class ZScoreDetector:
    """Detect anomalies using rolling Z-score."""
    window_size: int = 100
    threshold: float = 3.0
    samples: deque = field(default_factory=lambda: deque(maxlen=100))

    def __post_init__(self):
        self.samples = deque(maxlen=self.window_size)

    def add(self, value: float) -> dict[str, Any]:
        self.samples.append(value)
        if len(self.samples) < 10:
            return {"value": value, "anomaly": False, "z_score": 0.0}
        mean = sum(self.samples) / len(self.samples)
        variance = sum((x - mean) ** 2 for x in self.samples) / len(self.samples)
        std = math.sqrt(variance) if variance > 0 else 1e-10
        z = (value - mean) / std
        return {
            "value": value,
            "z_score": round(z, 3),
            "mean": round(mean, 3),
            "std": round(std, 3),
            "anomaly": abs(z) > self.threshold,
        }


@dataclass
class IQRDetector:
    """Detect anomalies using Interquartile Range (robust to outliers)."""
    window_size: int = 200
    k: float = 1.5  # IQR multiplier (1.5 = standard, 3.0 = extreme)
    samples: list[float] = field(default_factory=list)

    def add(self, value: float) -> dict[str, Any]:
        self.samples.append(value)
        if len(self.samples) > self.window_size:
            self.samples = self.samples[-self.window_size:]
        if len(self.samples) < 20:
            return {"value": value, "anomaly": False}
        sorted_s = sorted(self.samples)
        n = len(sorted_s)
        q1 = sorted_s[n // 4]
        q3 = sorted_s[3 * n // 4]
        iqr = q3 - q1
        lower = q1 - self.k * iqr
        upper = q3 + self.k * iqr
        return {
            "value": value,
            "q1": round(q1, 2), "q3": round(q3, 2), "iqr": round(iqr, 2),
            "lower_bound": round(lower, 2), "upper_bound": round(upper, 2),
            "anomaly": value < lower or value > upper,
        }


# =============================================================================
# 2. SEASONAL DECOMPOSITION (SIMPLE)
# =============================================================================

@dataclass
class SeasonalDetector:
    """Detect anomalies accounting for periodic patterns (e.g., daily cycles)."""
    period: int = 24  # e.g., 24 hours
    threshold_sigma: float = 2.5
    history: list[float] = field(default_factory=list)

    def add(self, value: float) -> dict[str, Any]:
        self.history.append(value)
        idx = len(self.history) - 1
        phase = idx % self.period

        # Collect all values at the same phase
        phase_values = [self.history[i] for i in range(phase, len(self.history), self.period)]
        if len(phase_values) < 3:
            return {"value": value, "anomaly": False, "phase": phase}

        mean = sum(phase_values) / len(phase_values)
        std = math.sqrt(
            sum((x - mean) ** 2 for x in phase_values) / len(phase_values)
        ) or 1e-10
        deviation = (value - mean) / std
        return {
            "value": value,
            "phase": phase,
            "seasonal_mean": round(mean, 2),
            "deviation_sigma": round(deviation, 2),
            "anomaly": abs(deviation) > self.threshold_sigma,
        }


# =============================================================================
# 3. DYNAMIC THRESHOLD CALCULATOR
# =============================================================================

def compute_dynamic_thresholds(
    data: list[float], window: int = 50, num_sigma: float = 2.0
) -> list[dict[str, float]]:
    """Compute rolling dynamic thresholds (upper/lower bounds)."""
    results = []
    for i in range(len(data)):
        start = max(0, i - window)
        window_data = data[start:i + 1]
        mean = sum(window_data) / len(window_data)
        std = math.sqrt(
            sum((x - mean) ** 2 for x in window_data) / len(window_data)
        ) if len(window_data) > 1 else 0
        results.append({
            "value": data[i],
            "upper": round(mean + num_sigma * std, 2),
            "lower": round(mean - num_sigma * std, 2),
            "anomaly": data[i] > mean + num_sigma * std or data[i] < mean - num_sigma * std,
        })
    return results


# =============================================================================
# 4. ALERT CORRELATION & DEDUPLICATION
# =============================================================================

@dataclass
class Alert:
    """A raw alert from a monitoring system."""
    name: str
    service: str
    metric: str
    value: float
    timestamp: float = field(default_factory=time.time)
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class AlertCorrelator:
    """Correlate and deduplicate related alerts."""
    time_window_s: float = 300.0  # 5-minute correlation window
    alerts: list[Alert] = field(default_factory=list)

    def ingest(self, alert: Alert) -> None:
        self.alerts.append(alert)

    def correlate(self) -> list[dict[str, Any]]:
        """Group alerts that likely share a root cause."""
        groups: dict[str, list[Alert]] = {}
        for alert in self.alerts:
            # Group by service + time window
            key = alert.service
            if key not in groups:
                groups[key] = []
            groups[key].append(alert)

        correlated = []
        for service, alerts in groups.items():
            alerts.sort(key=lambda a: a.timestamp)
            # Sub-group by time proximity
            current_group: list[Alert] = [alerts[0]]
            for a in alerts[1:]:
                if a.timestamp - current_group[-1].timestamp <= self.time_window_s:
                    current_group.append(a)
                else:
                    correlated.append(self._summarize_group(service, current_group))
                    current_group = [a]
            correlated.append(self._summarize_group(service, current_group))
        return correlated

    @staticmethod
    def _summarize_group(service: str, alerts: list[Alert]) -> dict[str, Any]:
        metrics = list({a.metric for a in alerts})
        return {
            "service": service,
            "alert_count": len(alerts),
            "metrics": metrics,
            "likely_root_cause": metrics[0] if len(metrics) == 1 else "multi-signal",
            "first_alert": alerts[0].name,
        }


# =============================================================================
# 5. ROOT CAUSE RANKER
# =============================================================================

def rank_root_causes(correlated_groups: list[dict], service_graph: dict[str, list[str]]
                     ) -> list[dict[str, Any]]:
    """Rank potential root causes using graph topology + alert density."""
    scores: dict[str, float] = {}
    for group in correlated_groups:
        svc = group["service"]
        # Base score: number of alerts
        scores[svc] = scores.get(svc, 0) + group["alert_count"]
        # Bonus for upstream services (they affect more downstream)
        for downstream in service_graph.get(svc, []):
            scores[svc] = scores.get(svc, 0) + 0.5

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{"service": svc, "score": round(score, 1), "rank": i + 1}
            for i, (svc, score) in enumerate(ranked)]


# =============================================================================
# 6. DEMO
# =============================================================================

if __name__ == "__main__":
    random.seed(42)

    # --- Z-Score Detector ---
    print("=" * 60)
    print("Z-Score Anomaly Detection")
    print("=" * 60)
    detector = ZScoreDetector(window_size=50, threshold=2.5)
    data = [random.gauss(100, 5) for _ in range(100)]
    data[45] = 135  # Inject anomaly
    data[78] = 60   # Inject anomaly
    anomalies = []
    for i, val in enumerate(data):
        result = detector.add(val)
        if result["anomaly"]:
            anomalies.append((i, result))
    print(f"  100 samples, {len(anomalies)} anomalies detected:")
    for idx, res in anomalies:
        print(f"    idx={idx}: value={res['value']:.1f}, z={res['z_score']:.2f}")

    # --- Seasonal Detector ---
    print(f"\n{'=' * 60}")
    print("Seasonal Anomaly Detection (24h cycle)")
    print("=" * 60)
    seasonal = SeasonalDetector(period=24, threshold_sigma=2.5)
    # 3 days of data with daily pattern + one anomaly
    seasonal_data = []
    for day in range(3):
        for hour in range(24):
            # Simulate daily traffic pattern: peak at hour 12
            base = 100 + 50 * math.sin(2 * math.pi * hour / 24)
            noise = random.gauss(0, 5)
            val = base + noise
            if day == 2 and hour == 14:
                val += 80  # Anomaly
            seasonal_data.append(val)
    seasonal_anomalies = []
    for i, val in enumerate(seasonal_data):
        result = seasonal.add(val)
        if result["anomaly"]:
            seasonal_anomalies.append((i, result))
    print(f"  {len(seasonal_data)} samples, {len(seasonal_anomalies)} anomalies:")
    for idx, res in seasonal_anomalies[-3:]:
        print(f"    idx={idx} (hour={res['phase']}): value={res['value']:.1f}, "
              f"seasonal_mean={res['seasonal_mean']:.1f}, "
              f"dev={res['deviation_sigma']:.2f}sigma")

    # --- Alert Correlation ---
    print(f"\n{'=' * 60}")
    print("Alert Correlation & Root Cause Ranking")
    print("=" * 60)
    correlator = AlertCorrelator(time_window_s=60)
    base_ts = time.time()
    # Simulate cascading failure
    for i, (svc, metric) in enumerate([
        ("database", "connection_pool_exhausted"),
        ("database", "query_latency_high"),
        ("order-svc", "error_rate_high"),
        ("order-svc", "latency_p99_high"),
        ("api-gateway", "5xx_rate_high"),
        ("payment-svc", "timeout_rate_high"),
    ]):
        correlator.ingest(Alert(
            name=f"{svc}-{metric}", service=svc, metric=metric,
            value=random.uniform(50, 100), timestamp=base_ts + i * 10,
        ))
    groups = correlator.correlate()
    for g in groups:
        print(f"  {g['service']}: {g['alert_count']} alerts, "
              f"metrics={g['metrics']}, cause={g['likely_root_cause']}")

    service_graph = {
        "database": ["order-svc", "payment-svc"],
        "order-svc": ["api-gateway"],
        "payment-svc": ["api-gateway"],
    }
    ranked = rank_root_causes(groups, service_graph)
    print(f"\n  Root cause ranking:")
    for r in ranked:
        print(f"    #{r['rank']}: {r['service']} (score={r['score']})")
