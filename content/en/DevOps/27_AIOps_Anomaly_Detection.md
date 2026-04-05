# 27. AIOps and Anomaly Detection

**Previous**: [Incident Response](./26_Incident_Response.md) | **Next**: [Capstone: Full-Stack Observability](./28_Capstone_Full_Stack_Observability.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Define AIOps and explain how machine learning enhances observability and operations
2. Implement statistical anomaly detection techniques for time-series metrics
3. Design intelligent alerting systems that reduce alert fatigue through ML-based thresholds
4. Evaluate AIOps platforms and their core capabilities (anomaly detection, correlation, root cause analysis)
5. Apply automated remediation patterns with appropriate safeguards
6. Distinguish between hype and practical value in AIOps tooling

---

Static alert thresholds ("alert if CPU > 80%") work for simple, predictable systems. But modern distributed systems exhibit complex, dynamic behavior -- seasonal traffic patterns, gradual degradation, and correlated failures across services. AIOps applies machine learning to observability data to detect anomalies that static rules miss, correlate related alerts to reduce noise, and automate routine remediation.

> **Analogy -- Weather Forecasting**: Static alerts are like a fixed rain warning: "Alert if humidity > 90%." This produces false alarms on humid-but-clear days and misses rain on dry-but-stormy days. ML-based anomaly detection is like modern weather forecasting: it considers pressure, wind, temperature trends, historical patterns, and satellite imagery to predict rain with much higher accuracy. The model learns what "normal" looks like for each context and alerts only on genuine anomalies.

## 1. AIOps Overview

### 1.1 What AIOps Covers

```
┌─────────────────────────────────────────────────┐
│                   AIOps Stack                    │
├─────────────────────────────────────────────────┤
│ Layer 4: Automated Remediation                   │
│   - Auto-scaling based on predicted demand       │
│   - Self-healing (restart, rollback, failover)   │
│   - Runbook automation triggered by anomalies    │
├─────────────────────────────────────────────────┤
│ Layer 3: Root Cause Analysis                     │
│   - Causal inference from correlated anomalies   │
│   - Topology-aware fault localization            │
│   - Change-to-impact correlation                 │
├─────────────────────────────────────────────────┤
│ Layer 2: Alert Intelligence                      │
│   - Alert correlation and grouping               │
│   - Noise reduction (suppress non-actionable)    │
│   - Severity prediction                          │
├─────────────────────────────────────────────────┤
│ Layer 1: Anomaly Detection                       │
│   - Baseline learning (what is "normal"?)        │
│   - Statistical anomaly detection                │
│   - Trend detection and forecasting              │
├─────────────────────────────────────────────────┤
│ Foundation: Observability Data                   │
│   - Metrics, Logs, Traces, Events, Changes       │
└─────────────────────────────────────────────────┘
```

### 1.2 AIOps vs Traditional Monitoring

| Aspect | Traditional Monitoring | AIOps |
|--------|----------------------|-------|
| **Thresholds** | Static (manually configured) | Dynamic (learned from data) |
| **Baselines** | None or manual | Automated, seasonal-aware |
| **Alert volume** | High (many false positives) | Reduced (correlated, deduplicated) |
| **Root cause** | Manual investigation | Assisted by correlation and topology |
| **Remediation** | Manual (follow runbook) | Automated for known patterns |
| **Scaling** | Breaks down at 100+ services | Designed for large-scale systems |

---

## 2. Anomaly Detection Techniques

### 2.1 Statistical Methods

**Moving Average with Standard Deviation:**

```python
"""Simple anomaly detection using moving average and standard deviation."""
import numpy as np
from dataclasses import dataclass

@dataclass
class AnomalyResult:
    timestamp: float
    value: float
    expected: float
    lower_bound: float
    upper_bound: float
    is_anomaly: bool
    z_score: float

def detect_anomalies_zscore(
    values: list[float],
    window_size: int = 60,
    threshold_sigma: float = 3.0,
) -> list[AnomalyResult]:
    """Detect anomalies using z-score with rolling statistics."""
    results = []
    for i in range(window_size, len(values)):
        window = values[i - window_size:i]
        mean = np.mean(window)
        std = np.std(window)

        if std == 0:
            std = 1e-10  # Avoid division by zero

        z_score = (values[i] - mean) / std
        is_anomaly = abs(z_score) > threshold_sigma

        results.append(AnomalyResult(
            timestamp=i,
            value=values[i],
            expected=mean,
            lower_bound=mean - threshold_sigma * std,
            upper_bound=mean + threshold_sigma * std,
            is_anomaly=is_anomaly,
            z_score=z_score,
        ))
    return results

# Example: detect latency anomalies
latency_samples = [100, 102, 98, 105, 99, 101, 97, 103, 100, 98,
                   # ... normal traffic ...
                   500, 480, 520,  # ← anomaly (latency spike)
                   101, 99, 102]  # ← back to normal
```

### 2.2 Seasonal Decomposition

Many metrics have daily, weekly, or monthly patterns:

```python
"""Seasonal anomaly detection using STL decomposition."""
from statsmodels.tsa.seasonal import STL

def detect_seasonal_anomalies(
    values: np.ndarray,
    period: int = 1440,      # 1440 minutes = 1 day for minute-resolution data
    threshold_sigma: float = 3.0,
) -> np.ndarray:
    """Detect anomalies accounting for seasonal patterns."""
    # STL decomposition: value = trend + seasonal + residual
    stl = STL(values, period=period, robust=True)
    result = stl.fit()

    # Anomalies are in the residual (what is left after removing trend + season)
    residual = result.resid
    residual_mean = np.mean(residual)
    residual_std = np.std(residual)

    # Points where residual exceeds threshold are anomalies
    z_scores = (residual - residual_mean) / residual_std
    is_anomaly = np.abs(z_scores) > threshold_sigma

    return is_anomaly
```

### 2.3 EWMA (Exponentially Weighted Moving Average)

```python
"""EWMA-based anomaly detection: more sensitive to recent changes."""

def detect_anomalies_ewma(
    values: list[float],
    alpha: float = 0.1,        # Smoothing factor (0.01=slow, 0.5=fast)
    threshold_sigma: float = 3.0,
) -> list[bool]:
    """Detect anomalies using EWMA."""
    ewma = values[0]
    ewma_var = 0.0
    anomalies = []

    for value in values:
        # Update EWMA
        diff = value - ewma
        ewma = alpha * value + (1 - alpha) * ewma
        ewma_var = (1 - alpha) * (ewma_var + alpha * diff * diff)
        ewma_std = np.sqrt(ewma_var)

        # Check for anomaly
        is_anomaly = abs(value - ewma) > threshold_sigma * ewma_std if ewma_std > 0 else False
        anomalies.append(is_anomaly)

    return anomalies
```

### 2.4 Comparison of Techniques

| Technique | Strengths | Weaknesses | Best For |
|-----------|-----------|------------|----------|
| **Z-score** | Simple, fast, interpretable | No seasonal awareness | Stationary metrics |
| **EWMA** | Adapts to trends, lightweight | Slow to react to sudden shifts | Gradually changing metrics |
| **STL decomposition** | Handles seasonality | Requires sufficient history | Daily/weekly patterns |
| **Isolation Forest** | Handles multivariate data | Less interpretable | Multi-metric anomalies |
| **Prophet** | Handles trends + seasonality + holidays | Heavier, requires training | Capacity planning |

---

## 3. Intelligent Alerting

### 3.1 Dynamic Thresholds in Prometheus

```yaml
# Instead of static: expr: cpu_usage > 80
# Use dynamic baseline: alert when > 3 standard deviations above average

groups:
  - name: dynamic_alerts
    rules:
      # Pre-compute 7-day baseline statistics
      - record: job:http_request_duration:avg_over_7d
        expr: avg_over_time(job:http_request_duration_seconds:p99[7d])

      - record: job:http_request_duration:stddev_over_7d
        expr: stddev_over_time(job:http_request_duration_seconds:p99[7d])

      # Alert when current value > baseline + 3 standard deviations
      - alert: LatencyAnomaly
        expr: |
          job:http_request_duration_seconds:p99
          > (job:http_request_duration:avg_over_7d + 3 * job:http_request_duration:stddev_over_7d)
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Latency anomaly detected (> 3 sigma above 7-day baseline)"
          description: |
            Current p99: {{ $value }}s
            Baseline avg: {{ with query "job:http_request_duration:avg_over_7d" }}{{ . | first | value }}{{ end }}s
```

### 3.2 Alert Correlation

Reduce alert storms by grouping related alerts:

```yaml
# Alertmanager: group correlated alerts
route:
  receiver: "default"
  group_by: ["cluster", "service"]   # Group alerts by cluster and service
  group_wait: 30s                      # Wait 30s to collect related alerts
  group_interval: 5m                   # Between grouped notifications
  repeat_interval: 4h

  routes:
    # SEV1: immediate, individual alerts
    - match:
        severity: critical
      receiver: "pagerduty-critical"
      group_wait: 10s

    # SEV2+: grouped by service
    - match:
        severity: warning
      receiver: "slack-warnings"
      group_by: ["service", "alertname"]
      group_wait: 60s

# Inhibition: suppress low-severity alerts when high-severity fires
inhibit_rules:
  # If a critical alert fires for a service, suppress warnings for the same service
  - source_matchers:
      - severity="critical"
    target_matchers:
      - severity="warning"
    equal: ["service"]

  # If the entire cluster is down, suppress individual service alerts
  - source_matchers:
      - alertname="ClusterDown"
    target_matchers:
      - severity=~"warning|critical"
    equal: ["cluster"]
```

### 3.3 Noise Reduction Metrics

| Metric | Before AIOps | After AIOps | Improvement |
|--------|-------------|-------------|-------------|
| Alerts per week | 500 | 50 | 90% reduction |
| Actionable alerts | 25% | 85% | 3.4x improvement |
| MTTD (mean time to detect) | 15 minutes | 3 minutes | 5x faster |
| False positive rate | 60% | 10% | 6x reduction |

---

## 4. Change-Impact Correlation

### 4.1 Linking Deployments to Anomalies

```python
"""Correlate deployments with metric anomalies."""
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class Deployment:
    service: str
    version: str
    deployed_at: datetime
    deployer: str
    changed_files: list[str]

@dataclass
class Anomaly:
    metric: str
    service: str
    detected_at: datetime
    severity: str
    value: float
    baseline: float

def correlate_deployments_with_anomalies(
    deployments: list[Deployment],
    anomalies: list[Anomaly],
    correlation_window: timedelta = timedelta(minutes=30),
) -> list[tuple[Anomaly, list[Deployment]]]:
    """Find deployments that occurred shortly before anomalies."""
    correlated = []

    for anomaly in anomalies:
        related_deployments = [
            d for d in deployments
            if d.deployed_at <= anomaly.detected_at
            and anomaly.detected_at - d.deployed_at <= correlation_window
            and (d.service == anomaly.service
                 or d.service in get_dependencies(anomaly.service))
        ]

        if related_deployments:
            correlated.append((anomaly, related_deployments))

    return correlated

# Usage:
# "Latency anomaly on order-service at 14:15 is correlated with
#  payment-service deployment at 14:02 (13 minutes before)"
```

### 4.2 Grafana Annotations for Change Correlation

```yaml
# Automatically annotate Grafana dashboards with deployment events
# deployment-annotator (runs as a Kubernetes admission webhook or CI step)

annotations:
  - datasource: prometheus
    expr: |
      changes(kube_deployment_status_observed_generation{
        namespace="production"
      }[5m]) > 0
    name: "Deployment"
    color: "blue"
    tags: ["deployment"]

  - datasource: prometheus
    expr: |
      ALERTS{alertstate="firing", severity="critical"}
    name: "Alert"
    color: "red"
    tags: ["alert"]
```

---

## 5. Automated Remediation

### 5.1 Remediation Safety Levels

| Level | Automation | Human Role | Example |
|-------|-----------|------------|---------|
| **L0: Manual** | Alert fires | Human investigates and fixes | Follow runbook manually |
| **L1: Assisted** | Diagnostic commands pre-run | Human reviews and approves | Auto-gather logs, human decides |
| **L2: Supervised** | Remediation prepared, human approves | Human clicks "approve" | Auto-prepare rollback, human approves |
| **L3: Automatic** | Full auto-remediation with guard rails | Human notified after the fact | Auto-restart crashed pod (Kubernetes default) |
| **L4: Predictive** | Act before the problem occurs | Human monitors trends | Auto-scale before traffic spike |

### 5.2 Safe Automated Remediation Patterns

```python
"""Automated remediation with safety guardrails."""
from datetime import datetime, timedelta

class RemediationGuardrails:
    """Prevent runaway automation."""

    def __init__(self):
        self.actions_taken: list[dict] = []
        self.max_actions_per_hour = 3
        self.max_concurrent_remediations = 1
        self.cooldown_minutes = 15
        self.active_remediations = 0

    def can_remediate(self, action: str, service: str) -> tuple[bool, str]:
        """Check if remediation is safe to execute."""
        # Guard 1: Rate limit
        recent = [a for a in self.actions_taken
                  if a["time"] > datetime.utcnow() - timedelta(hours=1)]
        if len(recent) >= self.max_actions_per_hour:
            return False, f"Rate limit: {len(recent)}/{self.max_actions_per_hour} actions in last hour"

        # Guard 2: Cooldown after previous action on same service
        same_service = [a for a in self.actions_taken
                        if a["service"] == service
                        and a["time"] > datetime.utcnow() - timedelta(minutes=self.cooldown_minutes)]
        if same_service:
            return False, f"Cooldown: last action on {service} was {same_service[-1]['time']}"

        # Guard 3: No concurrent remediations
        if self.active_remediations >= self.max_concurrent_remediations:
            return False, f"Concurrent limit: {self.active_remediations} active"

        # Guard 4: Business hours check (no auto-remediation during peak)
        hour = datetime.utcnow().hour
        if action == "rollback" and 9 <= hour <= 17:
            return False, "Auto-rollback disabled during business hours (manual approval required)"

        return True, "OK"

    def execute(self, action: str, service: str, details: str):
        can, reason = self.can_remediate(action, service)
        if not can:
            notify_oncall(f"Auto-remediation blocked: {reason}. Manual intervention needed.")
            return

        self.active_remediations += 1
        try:
            # Execute the remediation
            result = run_remediation(action, service, details)
            self.actions_taken.append({
                "action": action, "service": service,
                "time": datetime.utcnow(), "result": result
            })
            # Notify
            notify_oncall(f"Auto-remediation executed: {action} on {service}. Result: {result}")
        finally:
            self.active_remediations -= 1
```

### 5.3 Common Auto-Remediation Actions

| Trigger | Action | Safety Check |
|---------|--------|-------------|
| Pod crash loop | Restart pod with increased memory | Max 3 restarts/hour |
| High error rate after deploy | Auto-rollback to previous version | Only if < 10 min since deploy |
| Connection pool exhaustion | Restart application pods (rolling) | Max 1 restart per 15 minutes |
| Disk space > 90% | Delete old logs and temp files | Never delete data directories |
| Certificate expiring in 7 days | Auto-renew via cert-manager | Verify new cert before applying |
| Traffic spike detected | Scale up replicas | Max scale factor 3x; human approval above |

---

## 6. AIOps Platforms

### 6.1 Platform Landscape

| Platform | Type | Key Capability | Best For |
|----------|------|---------------|----------|
| **Datadog** | SaaS | Watchdog anomaly detection, correlation | Full-stack SaaS observability |
| **Dynatrace** | SaaS | Davis AI engine, auto-topology | Enterprise, Java-heavy environments |
| **New Relic** | SaaS | Applied Intelligence, anomaly detection | Full-stack with APM |
| **Grafana ML** | OSS/SaaS | Metric forecasting, anomaly alerting | Prometheus/Grafana users |
| **Elastic** | OSS/SaaS | ML anomaly detection on logs and metrics | Log-heavy environments |

### 6.2 Evaluating AIOps Claims

| Claim | Reality Check |
|-------|--------------|
| "Our AI detects all anomalies" | No system catches everything; ask about false positive rates |
| "Zero-configuration ML" | Models still need tuning (sensitivity, training window) |
| "Automatic root cause analysis" | Usually correlation, not true causal analysis; human validation needed |
| "90% alert reduction" | Often achievable but includes deduplication and grouping, not just ML |
| "Self-healing infrastructure" | Works for known patterns (restart, scale); novel failures still need humans |

---

## 7. Practical Implementation

### 7.1 Starting with AIOps (Phased Approach)

```
Phase 1 (Month 1-2): Foundation
  - Ensure clean, reliable metric data (fix gaps, standardize naming)
  - Implement SLO-based alerting (Lesson 20) -- reduces alert volume 60-80%
  - Add deployment annotations to dashboards
  → Biggest impact with zero ML

Phase 2 (Month 3-4): Statistical Anomaly Detection
  - Enable Grafana ML forecasting on key SLIs
  - Replace static thresholds with dynamic baselines (z-score or EWMA)
  - Implement alert correlation in Alertmanager
  → Reduces false positives by 40-60%

Phase 3 (Month 5-6): Correlation and Assisted Remediation
  - Deploy change-impact correlation (deployment → anomaly linking)
  - Implement L1-L2 automated diagnostics (pre-gather logs, traces)
  - Build auto-remediation for known patterns (restart, scale, rollback)
  → Reduces MTTR by 30-50%

Phase 4 (Month 7+): Advanced ML
  - Evaluate AIOps platform for topology-aware RCA
  - Implement predictive scaling
  - Train custom models on your incident history
  → Additional 20-30% MTTR reduction
```

---

## 8. Next Steps

- [28_Capstone_Full_Stack_Observability.md](./28_Capstone_Full_Stack_Observability.md) -- End-to-end observability platform design

---

## Exercises

### Exercise 1: Anomaly Detection Algorithm

Implement an EWMA-based anomaly detector for the following latency data (in milliseconds). Use alpha=0.1 and threshold=3 sigma. Identify which data points are anomalies and explain why.

```python
latency_data = [
    100, 102, 98, 105, 99, 101, 97, 103, 100, 98,  # Normal (minutes 1-10)
    102, 99, 101, 100, 103, 97, 105, 98, 100, 101,  # Normal (minutes 11-20)
    250, 280, 260,                                     # Spike (minutes 21-23)
    102, 100, 98, 101, 99,                             # Recovery (minutes 24-28)
    100, 101, 99, 102, 100, 98, 103, 101, 100, 99,   # Normal (minutes 29-38)
    115, 118, 120, 122, 125, 128, 130, 133, 135, 138, # Gradual increase (minutes 39-48)
]
```

<details>
<summary>Show Answer</summary>

```python
import numpy as np

latency_data = [
    100, 102, 98, 105, 99, 101, 97, 103, 100, 98,
    102, 99, 101, 100, 103, 97, 105, 98, 100, 101,
    250, 280, 260,
    102, 100, 98, 101, 99,
    100, 101, 99, 102, 100, 98, 103, 101, 100, 99,
    115, 118, 120, 122, 125, 128, 130, 133, 135, 138,
]

alpha = 0.1
threshold_sigma = 3.0

ewma = latency_data[0]
ewma_var = 0.0
anomalies = []

for i, value in enumerate(latency_data):
    diff = value - ewma
    ewma = alpha * value + (1 - alpha) * ewma
    ewma_var = (1 - alpha) * (ewma_var + alpha * diff * diff)
    ewma_std = np.sqrt(ewma_var)

    is_anomaly = abs(value - ewma) > threshold_sigma * ewma_std if ewma_std > 0 else False

    if is_anomaly:
        anomalies.append((i, value, ewma, ewma_std))
        print(f"Minute {i+1}: value={value}, ewma={ewma:.1f}, std={ewma_std:.1f}, ANOMALY")
```

**Expected anomalies:**
- **Minute 21 (value=250)**: Sudden spike from ~100 to 250. EWMA is ~101, std is ~2.5. Z-score ≈ (250-101)/2.5 = 59.6 >> 3. Clear anomaly.
- **Minute 22 (value=280)**: Even higher. EWMA has slightly adjusted up (~116) but still far below 280. Anomaly.
- **Minute 23 (value=260)**: Still well above EWMA. Anomaly.

**Gradual increase (minutes 39-48)**: These are likely NOT flagged as anomalies because:
- EWMA tracks the gradual increase (each point only 2-3ms above the previous)
- The standard deviation adjusts upward to accommodate the trend
- No single point is > 3 sigma above the EWMA

This demonstrates a limitation of EWMA: it adapts to gradual changes and may miss slow degradation. For gradual trends, use trend detection (e.g., linear regression on the EWMA slope) or compare against a longer baseline.

</details>

### Exercise 2: Alert Correlation Design

You manage a microservices platform with 50 services. During an incident, you received 47 alerts in 5 minutes. Design an alert correlation strategy that groups these into a manageable number of actionable alerts. Specify: grouping keys, time windows, inhibition rules, and how you would present the correlated alerts to the on-call engineer.

<details>
<summary>Show Answer</summary>

**Alert correlation strategy:**

**1. Grouping configuration:**
```yaml
route:
  group_by: ["cluster", "namespace", "service"]
  group_wait: 60s       # Wait 60s to collect related alerts before sending
  group_interval: 5m
```
This collapses 47 alerts into ~5-10 groups (one per affected service/namespace).

**2. Inhibition rules:**
```yaml
inhibit_rules:
  # If infrastructure is down, suppress application alerts
  - source_matchers:
      - alertname=~"NodeDown|ClusterUnreachable"
    target_matchers:
      - severity=~"warning|critical"
    equal: ["cluster"]

  # If a database is down, suppress all services that depend on it
  - source_matchers:
      - alertname="DatabaseDown"
    target_matchers:
      - severity=~"warning|critical"
    equal: ["database_dependency"]

  # Critical suppresses warning for same service
  - source_matchers:
      - severity="critical"
    target_matchers:
      - severity="warning"
    equal: ["service"]
```

**3. Topology-aware grouping:**
Use the service dependency graph to identify the root service. If `payment-service` is the first to alert and `order-service`, `checkout-service`, `api-gateway` alert afterward, group them under "payment-service dependency failure."

**4. Presentation to on-call:**

```
🚨 Incident Alert Summary (47 alerts consolidated into 3 groups)

Group 1: CRITICAL -- payment-service (root cause candidate)
  - PaymentServiceHighErrorRate (14:00:00)     ← First alert
  - PaymentServiceHighLatency (14:00:15)
  - PaymentServiceSLOBurnRate (14:00:30)
  Related downstream alerts: order-service (5 alerts), checkout-service (3 alerts)
  [View dashboard] [View traces] [Runbook]

Group 2: WARNING -- Cascading from Group 1
  - OrderServiceHighErrorRate (14:01:00)
  - OrderServiceTimeouts (14:01:10)
  - CheckoutServiceUnavailable (14:01:30)
  ... and 25 more (suppressed, caused by Group 1)
  Likely impact of payment-service failure. Will resolve when Group 1 is fixed.

Group 3: INFO -- Unrelated
  - DiskSpaceWarning on monitoring-node-3 (14:02:00)
  Separate issue, not correlated with Groups 1-2.
```

**Result: 47 alerts → 3 actionable groups. On-call focuses on Group 1.**

</details>

---

## References

- [Moogsoft -- AIOps Platform](https://www.moogsoft.com/)
- [Datadog Watchdog](https://docs.datadoghq.com/watchdog/)
- [Grafana Machine Learning](https://grafana.com/docs/grafana-cloud/alerting-and-irm/machine-learning/)
- [Google SRE Book -- Practical Alerting](https://sre.google/sre-book/practical-alerting/)
- [Statistical Anomaly Detection (Netflix Blog)](https://netflixtechblog.com/rad-outlier-detection-on-big-data-d6b0494371cc)
- [Chaos Engineering + AIOps (Gremlin)](https://www.gremlin.com/blog/aiops-and-chaos-engineering/)
