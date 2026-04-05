# 20. SLO Engineering

**Previous**: [Observability Engineering](./19_Observability_Engineering.md) | **Next**: [Signal Correlation](./21_Signal_Correlation.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Define SLIs, SLOs, and SLAs and explain the relationship between them
2. Select appropriate SLIs for different service types and calculate SLO compliance
3. Implement error budgets as a decision framework for balancing reliability and velocity
4. Design burn rate alerts that detect SLO violations with appropriate urgency
5. Build SLO dashboards that communicate reliability posture to stakeholders
6. Operate an error budget policy that governs feature releases and reliability investments

---

SLO engineering transforms reliability from a vague aspiration ("we want to be reliable") into a measurable, actionable engineering discipline. By defining exactly how reliable a service must be, you create a shared language between engineering, product, and business stakeholders -- and a decision framework (error budgets) that makes trade-offs explicit.

> **Analogy -- Manufacturing Tolerances**: A machinist does not aim for "a good bolt." They specify a tolerance: "10mm diameter +/- 0.05mm." This tolerance is the SLO. The measurement instrument (caliper) is the SLI. The contract with the customer guaranteeing the bolt meets spec is the SLA. If 99.5% of bolts are within tolerance, and the SLO is 99%, they have budget to experiment with faster machining. If they drop to 98.5%, they stop experimenting and fix the process.

## 1. SLIs, SLOs, and SLAs

### 1.1 Definitions

```
SLA (Service Level Agreement)
  └── Business contract with consequences (refunds, penalties)
  └── "99.9% availability or customer gets 10% credit"

SLO (Service Level Objective)
  └── Internal reliability target (stricter than SLA)
  └── "99.95% availability over a 30-day rolling window"

SLI (Service Level Indicator)
  └── The metric that measures the SLO
  └── "Proportion of requests completed successfully in < 300ms"
```

**Key relationship**: SLA >= SLO (SLO is always stricter than SLA, providing a buffer).

### 1.2 SLI Specification vs Implementation

| Concept | Definition | Example |
|---------|-----------|---------|
| **SLI Specification** | What you want to measure (abstract) | "The proportion of valid requests served successfully" |
| **SLI Implementation** | How you measure it (concrete) | `sum(rate(http_requests_total{status!~"5.."}[5m])) / sum(rate(http_requests_total[5m]))` |

### 1.3 Common SLI Types

| SLI Type | Definition | Good For |
|----------|-----------|----------|
| **Availability** | Proportion of valid requests that succeed | Request/response services (APIs) |
| **Latency** | Proportion of requests faster than threshold | User-facing services |
| **Throughput** | Rate of successful processing | Data processing pipelines |
| **Freshness** | Proportion of data updated within threshold | Databases, caches, search indexes |
| **Correctness** | Proportion of responses returning correct data | Data pipelines, ML inference |
| **Durability** | Proportion of data recoverable after storage | Storage systems |

### 1.4 Choosing SLIs by Service Type

| Service Type | Primary SLIs | Example |
|-------------|-------------|---------|
| **REST API** | Availability, latency (p50, p99) | Payment service: 99.99% availability, p99 < 500ms |
| **Streaming pipeline** | Throughput, freshness | Kafka consumer: 99.9% of events processed within 30s |
| **Batch processing** | Throughput, correctness, freshness | ETL job: 99.5% of runs complete within 2h, < 0.01% error rate |
| **Storage system** | Availability, latency, durability | Database: 99.99% availability, p99 read < 10ms, 99.999999% durability |
| **Frontend (web)** | Availability, latency (Core Web Vitals) | LCP < 2.5s for 75% of page loads |

---

## 2. SLO Design

### 2.1 The SLO Document

Every service should have a written SLO document:

```yaml
# slo-document.yaml
service: payment-service
owner: payments-team
last_review: 2025-01-15

slos:
  - name: availability
    description: "Proportion of non-5xx responses to valid requests"
    sli:
      type: availability
      specification: "Good events: status < 500. Total events: all HTTP requests excluding health checks."
      implementation:
        numerator: 'sum(rate(http_requests_total{job="payment-service",status!~"5.."}[5m]))'
        denominator: 'sum(rate(http_requests_total{job="payment-service"}[5m]))'
    objective: 99.95%
    window: 30d rolling
    consequences:
      budget_exhausted: "Freeze feature deployments until budget recovers"
      budget_below_25pct: "Cancel next sprint's feature work; focus on reliability"

  - name: latency
    description: "Proportion of requests completed within 300ms"
    sli:
      type: latency
      specification: "Good events: response time < 300ms. Total events: all HTTP requests."
      implementation:
        numerator: 'sum(rate(http_request_duration_seconds_bucket{job="payment-service",le="0.3"}[5m]))'
        denominator: 'sum(rate(http_request_duration_seconds_count{job="payment-service"}[5m]))'
    objective: 99.0%
    window: 30d rolling
    consequences:
      budget_exhausted: "Initiate performance review sprint"
```

### 2.2 Choosing the Right Objective

| Objective | Downtime/month | Error Budget/month | Suitable For |
|-----------|---------------|-------------------|-------------|
| 99% | 7h 18m | 7h 18m | Internal tools, batch jobs |
| 99.5% | 3h 39m | 3h 39m | Non-critical services |
| 99.9% | 43.8m | 43.8m | Standard production services |
| 99.95% | 21.9m | 21.9m | Important customer-facing services |
| 99.99% | 4.38m | 4.38m | Critical infrastructure (auth, payments) |
| 99.999% | 26.3s | 26.3s | DNS, core routing (very expensive to achieve) |

**Guidelines for choosing objectives:**

1. **Start with user expectations**: How much unreliability will users tolerate?
2. **Consider dependencies**: Your SLO cannot exceed your least reliable dependency
3. **Factor in cost**: Each additional nine roughly 10x the engineering cost
4. **Leave buffer before SLA**: If SLA is 99.9%, set SLO at 99.95%
5. **Start lower, tighten later**: It is easier to tighten an SLO than to relax one

### 2.3 SLO Windows

| Window Type | Description | Pros | Cons |
|------------|-------------|------|------|
| **Rolling** (e.g., 30 days) | Continuous sliding window | Smooth, no cliff edges | Incident effects persist for full window |
| **Calendar** (e.g., monthly) | Resets at period boundary | Fresh start each month | Cliff edge at month boundary; incident at month end vs. start treated differently |

**Best practice**: Use rolling windows for operational decisions, calendar windows for business reporting.

---

## 3. Error Budgets

### 3.1 The Error Budget Concept

```
Error Budget = 1 - SLO

Example:
  SLO = 99.9% availability
  Error Budget = 0.1% of requests can fail

  If you serve 10,000,000 requests/month:
  Error Budget = 10,000 allowed failures/month

  Or in time:
  Error Budget = 30 days × 24h × 60m × 0.001 = 43.2 minutes of downtime/month
```

### 3.2 Error Budget Consumption

```python
"""Error budget calculator for request-based SLIs."""
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class ErrorBudgetStatus:
    slo_target: float
    window_days: int
    total_requests: int
    failed_requests: int

    @property
    def current_sli(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return (self.total_requests - self.failed_requests) / self.total_requests

    @property
    def error_budget_total(self) -> int:
        """Total allowed failures in the window."""
        return int(self.total_requests * (1 - self.slo_target))

    @property
    def error_budget_remaining(self) -> int:
        """Remaining failures before SLO violation."""
        return max(0, self.error_budget_total - self.failed_requests)

    @property
    def error_budget_remaining_pct(self) -> float:
        """Percentage of error budget remaining."""
        if self.error_budget_total == 0:
            return 0.0
        return self.error_budget_remaining / self.error_budget_total * 100

    @property
    def is_slo_met(self) -> bool:
        return self.current_sli >= self.slo_target

# Example usage
status = ErrorBudgetStatus(
    slo_target=0.999,       # 99.9%
    window_days=30,
    total_requests=10_000_000,
    failed_requests=3_500,
)

print(f"SLO target:             {status.slo_target:.3%}")
print(f"Current SLI:            {status.current_sli:.4%}")
print(f"Error budget (total):   {status.error_budget_total:,} requests")
print(f"Error budget (used):    {status.failed_requests:,} requests")
print(f"Error budget remaining: {status.error_budget_remaining:,} requests ({status.error_budget_remaining_pct:.1f}%)")
print(f"SLO met:                {status.is_slo_met}")
```

Output:

```
SLO target:             99.900%
Current SLI:            99.965%
Error budget (total):   10,000 requests
Error budget (used):    3,500 requests
Error budget remaining: 6,500 requests (65.0%)
SLO met:                True
```

### 3.3 Error Budget Policy

An error budget policy defines what happens at different budget levels:

| Budget Remaining | Action |
|-----------------|--------|
| **> 50%** | Normal operations. Ship features at full velocity. |
| **25% -- 50%** | Caution. Increase testing rigor. No risky deployments on Fridays. |
| **5% -- 25%** | Slow down. Reliability work takes priority. Feature freeze for non-critical changes. |
| **0% -- 5%** | Emergency. Full feature freeze. All engineering effort on reliability. |
| **Exhausted (< 0%)** | SLO violated. Halt all deployments. Conduct incident review. Publish postmortem. |

### 3.4 Error Budget and Deployment Decisions

```
Decision Framework:
────────────────────────────────────────────────
Feature deployment request arrives
    │
    ├── Check error budget remaining
    │   │
    │   ├── Budget > 50%? → APPROVE: Deploy normally
    │   │
    │   ├── Budget 25-50%? → APPROVE with conditions:
    │   │     - Canary deployment required
    │   │     - Automated rollback enabled
    │   │     - Not during peak traffic
    │   │
    │   ├── Budget 5-25%? → REVIEW required:
    │   │     - Risk assessment by SRE
    │   │     - Only reliability-improving changes approved
    │   │
    │   └── Budget < 5%? → DENY:
    │         - Only critical security patches
    │         - Reliability fixes only
    │
    └── Post-deployment:
        - Monitor SLI for 30 minutes
        - Auto-rollback if SLI degrades
```

---

## 4. Burn Rate Alerts

### 4.1 Why Threshold Alerts Fail for SLOs

Traditional threshold alerts ("error rate > 1%") do not account for the SLO window:

- A 1% error rate for 1 minute consumes almost no budget
- A 1% error rate for 1 hour is concerning
- A 1% error rate for 1 day is critical

**Burn rate** measures how fast you are consuming your error budget relative to the window.

### 4.2 Burn Rate Definition

```
Burn Rate = (Observed error rate) / (SLO-allowed error rate)

Example:
  SLO = 99.9% (allowed error rate = 0.1%)
  Observed error rate = 0.5%

  Burn Rate = 0.5% / 0.1% = 5x

  At 5x burn rate, the 30-day error budget would be exhausted in 6 days.
  Time to exhaustion = Window / Burn Rate = 30 days / 5 = 6 days
```

### 4.3 Multi-Window, Multi-Burn-Rate Alerts

Google's recommended approach uses multiple burn rates with matching lookback windows:

| Severity | Burn Rate | Long Window | Short Window | Time to Exhaust |
|----------|-----------|-------------|--------------|-----------------|
| **Page (critical)** | 14.4x | 1 hour | 5 minutes | 2 days |
| **Page (urgent)** | 6x | 6 hours | 30 minutes | 5 days |
| **Ticket (warning)** | 3x | 1 day | 2 hours | 10 days |
| **Ticket (info)** | 1x | 3 days | 6 hours | 30 days |

Both windows must be in violation for the alert to fire. The short window prevents stale alerts (the problem may have already resolved).

### 4.4 Prometheus Implementation

```yaml
# Burn rate alerting rules for payment-service availability SLO (99.9%)
groups:
  - name: payment_slo_burn_rate
    rules:
      # --- Recording rules for error ratios ---
      - record: payment_service:error_ratio:rate5m
        expr: |
          sum(rate(http_requests_total{job="payment-service",status=~"5.."}[5m]))
          / sum(rate(http_requests_total{job="payment-service"}[5m]))

      - record: payment_service:error_ratio:rate30m
        expr: |
          sum(rate(http_requests_total{job="payment-service",status=~"5.."}[30m]))
          / sum(rate(http_requests_total{job="payment-service"}[30m]))

      - record: payment_service:error_ratio:rate1h
        expr: |
          sum(rate(http_requests_total{job="payment-service",status=~"5.."}[1h]))
          / sum(rate(http_requests_total{job="payment-service"}[1h]))

      - record: payment_service:error_ratio:rate6h
        expr: |
          sum(rate(http_requests_total{job="payment-service",status=~"5.."}[6h]))
          / sum(rate(http_requests_total{job="payment-service"}[6h]))

      - record: payment_service:error_ratio:rate1d
        expr: |
          sum(rate(http_requests_total{job="payment-service",status=~"5.."}[1d]))
          / sum(rate(http_requests_total{job="payment-service"}[1d]))

      - record: payment_service:error_ratio:rate3d
        expr: |
          sum(rate(http_requests_total{job="payment-service",status=~"5.."}[3d]))
          / sum(rate(http_requests_total{job="payment-service"}[3d]))

      # --- Burn rate alerts ---
      # Critical: 14.4x burn rate (2-day exhaustion)
      - alert: PaymentSLOBurnRateCritical
        expr: |
          payment_service:error_ratio:rate1h > (14.4 * 0.001)
          and
          payment_service:error_ratio:rate5m > (14.4 * 0.001)
        for: 2m
        labels:
          severity: critical
          slo: payment-availability
        annotations:
          summary: "Payment service SLO burn rate critical (14.4x)"
          description: |
            Error budget will be exhausted in {{ printf "%.0f" (div 720.0 14.4) }} hours
            at current burn rate. 1h error ratio: {{ $value }}.

      # Urgent: 6x burn rate (5-day exhaustion)
      - alert: PaymentSLOBurnRateHigh
        expr: |
          payment_service:error_ratio:rate6h > (6 * 0.001)
          and
          payment_service:error_ratio:rate30m > (6 * 0.001)
        for: 5m
        labels:
          severity: warning
          slo: payment-availability
        annotations:
          summary: "Payment service SLO burn rate high (6x)"
          description: |
            Error budget will be exhausted in 5 days at current burn rate.
            6h error ratio: {{ $value }}.

      # Ticket: 3x burn rate (10-day exhaustion)
      - alert: PaymentSLOBurnRateElevated
        expr: |
          payment_service:error_ratio:rate1d > (3 * 0.001)
          and
          payment_service:error_ratio:rate2h > (3 * 0.001)
        for: 15m
        labels:
          severity: info
          slo: payment-availability
        annotations:
          summary: "Payment service SLO burn rate elevated (3x)"
```

### 4.5 Latency SLO Burn Rate

For latency SLOs, the "good event" is a request faster than the threshold:

```yaml
# Latency SLO: 99% of requests < 300ms
groups:
  - name: payment_latency_slo
    rules:
      - record: payment_service:latency_good_ratio:rate1h
        expr: |
          sum(rate(http_request_duration_seconds_bucket{
            job="payment-service", le="0.3"
          }[1h]))
          / sum(rate(http_request_duration_seconds_count{
            job="payment-service"
          }[1h]))

      - alert: PaymentLatencySLOBurnRateCritical
        expr: |
          (1 - payment_service:latency_good_ratio:rate1h) > (14.4 * 0.01)
          and
          (1 - payment_service:latency_good_ratio:rate5m) > (14.4 * 0.01)
        for: 2m
        labels:
          severity: critical
          slo: payment-latency
        annotations:
          summary: "Payment service latency SLO burn rate critical"
```

---

## 5. SLO Dashboards

### 5.1 Dashboard Design Principles

An SLO dashboard should answer three questions in under 10 seconds:

1. **Are we meeting our SLOs?** (Current SLI vs target)
2. **How much error budget remains?** (Budget gauge)
3. **What is the trend?** (Burn rate over time)

### 5.2 Dashboard Layout

```
┌─────────────────────────────────────────────────────────┐
│ Payment Service SLO Dashboard                           │
├─────────────────┬───────────────────┬───────────────────┤
│ Availability    │ Latency (p99)     │ Error Budget      │
│ SLO: 99.95%     │ SLO: 99% < 300ms  │ ████████░░ 65%   │
│ Current: 99.97% │ Current: 99.3%    │ 6,500 / 10,000   │
│ Status: ✓ OK    │ Status: ✓ OK      │ remaining         │
├─────────────────┴───────────────────┴───────────────────┤
│ Error Budget Consumption (30-day rolling)               │
│ ▁▁▂▁▁▁▁▃▁▁▁▅▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁                       │
│ Day 1        Day 10 (incident)    Day 30               │
├─────────────────────────────────────────────────────────┤
│ Burn Rate (current)                                     │
│ 1h: 0.3x  │  6h: 0.5x  │  1d: 0.8x  │  3d: 0.7x     │
│ All within normal range                                 │
├─────────────────────────────────────────────────────────┤
│ Top Error Contributors (last 24h)                       │
│  1. POST /payments/charge: 0.15% error rate (timeout)   │
│  2. GET /payments/:id: 0.02% error rate (404s)          │
│  3. POST /payments/refund: 0.01% error rate             │
│     [View traces →]  [View logs →]                      │
└─────────────────────────────────────────────────────────┘
```

### 5.3 Grafana PromQL Queries for SLO Dashboards

```promql
# Current availability SLI (instant)
1 - (
  sum(rate(http_requests_total{job="payment-service",status=~"5.."}[30d]))
  / sum(rate(http_requests_total{job="payment-service"}[30d]))
)

# Error budget remaining (percentage)
1 - (
  (1 - (
    sum(increase(http_requests_total{job="payment-service",status!~"5.."}[30d]))
    / sum(increase(http_requests_total{job="payment-service"}[30d]))
  ))
  / (1 - 0.9995)  # 1 - SLO target
)

# Burn rate over time (for time-series panel)
(
  sum(rate(http_requests_total{job="payment-service",status=~"5.."}[1h]))
  / sum(rate(http_requests_total{job="payment-service"}[1h]))
) / 0.0005  # SLO budget rate (1 - 0.9995)
```

---

## 6. SLO Operations

### 6.1 SLO Review Cadence

| Review | Frequency | Participants | Agenda |
|--------|-----------|-------------|--------|
| **Weekly SLO check** | Weekly | On-call engineer | Review burn rates, budget status |
| **Monthly SLO review** | Monthly | Team lead + SRE | Review trends, incident impact, adjust thresholds |
| **Quarterly SLO audit** | Quarterly | Engineering leadership | Review all SLOs, propose new ones, retire stale ones |

### 6.2 Handling SLO Violations

```
SLO Violation Detected
    │
    ├── 1. Immediate (within 1 hour)
    │   └── Page on-call engineer
    │   └── Determine if ongoing or past incident
    │   └── If ongoing: follow incident response (Lesson 26)
    │
    ├── 2. Short-term (within 24 hours)
    │   └── Conduct root cause analysis
    │   └── Apply error budget policy (feature freeze if needed)
    │   └── Communicate to stakeholders
    │
    ├── 3. Medium-term (within 1 week)
    │   └── Postmortem with action items
    │   └── Plan reliability improvements
    │   └── Review if SLO target is appropriate
    │
    └── 4. Long-term (next quarter)
        └── Track action item completion
        └── Review SLO targets during quarterly audit
        └── Consider architectural changes if violations recur
```

### 6.3 SLO for Dependencies

When your service depends on other services, model the dependency chain:

```python
"""SLO dependency chain calculator."""

def composite_availability(service_slos: dict[str, float], topology: str) -> float:
    """Calculate composite availability based on dependency topology."""
    slos = list(service_slos.values())

    if topology == "serial":
        # All services must be available: multiply availabilities
        result = 1.0
        for slo in slos:
            result *= slo
        return result

    elif topology == "parallel_redundant":
        # Any one service sufficient: 1 - product of unavailabilities
        result = 1.0
        for slo in slos:
            result *= (1 - slo)
        return 1 - result

    else:
        raise ValueError(f"Unknown topology: {topology}")

# Serial chain: API → Payment → Database
serial = composite_availability({
    "api_gateway": 0.9999,
    "payment_service": 0.9995,
    "database": 0.9999,
}, topology="serial")
print(f"Serial availability: {serial:.4%}")
# 99.93% -- worse than any individual service

# Redundant: Primary DB + Replica DB
redundant = composite_availability({
    "primary_db": 0.999,
    "replica_db": 0.999,
}, topology="parallel_redundant")
print(f"Redundant availability: {redundant:.6%}")
# 99.9999% -- much better than either alone
```

---

## 7. Advanced SLO Patterns

### 7.1 Multi-SLO Services

Real services need multiple SLOs covering different aspects:

```yaml
service: search-service
slos:
  - name: availability
    objective: 99.9%
    description: "Search queries return non-error responses"

  - name: latency-p50
    objective: 99%
    threshold: 100ms
    description: "Median search latency under 100ms"

  - name: latency-p99
    objective: 95%
    threshold: 1000ms
    description: "Tail search latency under 1 second"

  - name: freshness
    objective: 99%
    threshold: 60s
    description: "Search index updated within 60s of source change"

  - name: relevance
    objective: 95%
    description: "First result matches user intent (measured by click-through rate)"
```

### 7.2 User-Journey SLOs

Rather than per-service SLOs, define SLOs for user-visible journeys:

| Journey | SLI | SLO |
|---------|-----|-----|
| **Checkout** | Proportion of checkout attempts that complete successfully within 5s | 99.9% |
| **Search** | Proportion of searches that return results within 500ms | 99.5% |
| **Login** | Proportion of login attempts that succeed or fail definitively within 2s | 99.99% |
| **File upload** | Proportion of uploads < 100MB that complete within 30s | 99.0% |

User-journey SLOs require synthetic monitoring or real-user monitoring (RUM) to measure.

### 7.3 SLO as Code

Define SLOs in version-controlled configuration, consumed by alerting and dashboard tools:

```yaml
# sloth.yaml (Sloth -- SLO-to-Prometheus-rules generator)
version: "prometheus/v1"
service: "payment-service"
labels:
  owner: "payments-team"
  tier: "critical"
slos:
  - name: "requests-availability"
    objective: 99.95
    description: "Payment API availability"
    sli:
      events:
        error_query: sum(rate(http_requests_total{job="payment-service",status=~"5.."}[{{.window}}]))
        total_query: sum(rate(http_requests_total{job="payment-service"}[{{.window}}]))
    alerting:
      name: PaymentAvailability
      labels:
        category: availability
      annotations:
        runbook: "https://wiki.example.com/runbooks/payment-availability"
      page_alert:
        labels:
          severity: critical
      ticket_alert:
        labels:
          severity: warning
```

```bash
# Generate Prometheus rules from SLO definition
sloth generate -i sloth.yaml -o /etc/prometheus/rules/payment-slo.yml

# Output: recording rules + burn rate alert rules automatically generated
```

---

## 8. Organizational Adoption

### 8.1 Getting Buy-In

| Stakeholder | Message |
|-------------|---------|
| **Engineering leadership** | "SLOs let us make data-driven decisions about reliability vs. features" |
| **Product managers** | "Error budgets tell you exactly how much risk you can take on new features" |
| **Engineers** | "SLOs protect you from being blamed for outages that are within budget" |
| **Customer support** | "SLO dashboards tell you immediately if a reported issue is real or isolated" |

### 8.2 Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| **Too many SLOs** | Diluted attention, conflicting objectives | Start with 1-3 SLOs per service |
| **SLO too tight** | Constant violations, team ignores SLO | Start loose, tighten based on data |
| **SLO too loose** | Never violated, provides no information | Tighten until budget is occasionally consumed |
| **No error budget policy** | SLO is measured but has no consequences | Define and enforce the policy before setting the SLO |
| **Measuring server-side only** | SLO says 99.9% but users experience 95% | Measure at the edge or use synthetic monitoring |

---

## 9. Next Steps

- [21_Signal_Correlation.md](./21_Signal_Correlation.md) -- Correlate metrics, logs, and traces for faster debugging
- [22_Advanced_Metrics_Architecture.md](./22_Advanced_Metrics_Architecture.md) -- Scale metrics infrastructure with federation and long-term storage

---

## Exercises

### Exercise 1: SLI Selection

For each service below, choose 2-3 SLIs and justify your choices. Specify both the SLI specification (what to measure) and implementation (how to measure it in Prometheus/OTel).

1. An image upload and processing service that resizes images to multiple sizes
2. A real-time chat messaging service
3. A nightly batch job that generates financial reports

<details>
<summary>Show Answer</summary>

**1. Image upload and processing service:**

| SLI | Specification | Implementation | Justification |
|-----|--------------|----------------|---------------|
| Availability | Proportion of upload requests that return non-error responses | `sum(rate(http_requests_total{job="image-service",status!~"5.."}[5m])) / sum(rate(http_requests_total{job="image-service"}[5m]))` | Users need uploads to succeed |
| Latency | Proportion of uploads that return within 10s (for images < 5MB) | `sum(rate(http_request_duration_seconds_bucket{job="image-service",le="10"}[5m])) / sum(rate(http_request_duration_seconds_count{job="image-service"}[5m]))` | Upload latency directly affects user experience |
| Freshness | Proportion of images where all resized variants are available within 60s of upload | Custom metric: `sum(rate(image_processing_completed_within_slo_total[5m])) / sum(rate(image_uploads_total[5m]))` | Users expect resized images quickly |

**2. Real-time chat messaging service:**

| SLI | Specification | Implementation | Justification |
|-----|--------------|----------------|---------------|
| Availability | Proportion of message send requests that succeed | `sum(rate(chat_messages_sent_total{status="success"}[5m])) / sum(rate(chat_messages_sent_total[5m]))` | Message delivery is the core function |
| Latency | Proportion of messages delivered to the recipient within 500ms | Custom metric: `sum(rate(chat_message_delivery_duration_seconds_bucket{le="0.5"}[5m])) / sum(rate(chat_message_delivery_duration_seconds_count[5m]))` | Real-time chat requires low latency |
| Freshness | Proportion of message history requests that return data less than 5s stale | `sum(rate(chat_history_freshness_within_slo_total[5m])) / sum(rate(chat_history_requests_total[5m]))` | Users expect message history to be current |

**3. Nightly batch job (financial reports):**

| SLI | Specification | Implementation | Justification |
|-----|--------------|----------------|---------------|
| Freshness | Proportion of reports available by 6:00 AM deadline | `sum(report_generation_completed_before_deadline_total) / sum(report_generation_attempts_total)` | Business users need reports before market open |
| Correctness | Proportion of report rows that match source data within rounding tolerance | Custom validation: `sum(report_rows_correct_total) / sum(report_rows_total)` | Financial data must be accurate |
| Throughput | Proportion of scheduled reports that complete successfully | `sum(reports_completed_successfully_total) / sum(reports_scheduled_total)` | All reports must be generated |

</details>

### Exercise 2: Error Budget Calculation

A service has a 99.9% availability SLO over a 30-day rolling window. In the past 30 days:
- Total requests: 50,000,000
- 5xx responses: 42,000
- An incident on Day 15 caused 30,000 of those errors in a 2-hour period

Calculate: (a) current SLI, (b) total error budget, (c) budget consumed, (d) budget remaining percentage, (e) burn rate during the incident, and (f) what actions the error budget policy should trigger.

<details>
<summary>Show Answer</summary>

**(a) Current SLI:**
```
SLI = (50,000,000 - 42,000) / 50,000,000 = 49,958,000 / 50,000,000 = 99.916%
```

**(b) Total error budget:**
```
Budget = 50,000,000 × (1 - 0.999) = 50,000,000 × 0.001 = 50,000 errors
```

**(c) Budget consumed:**
```
Consumed = 42,000 / 50,000 = 84%
```

**(d) Budget remaining:**
```
Remaining = (50,000 - 42,000) / 50,000 = 8,000 / 50,000 = 16%
```

**(e) Burn rate during the incident:**
```
Normal error rate = (42,000 - 30,000) / 50,000,000 = 0.024% (background errors)
Incident request rate = 50,000,000 / 30 / 24 = ~69,444 requests/hour
Incident 2-hour requests ≈ 138,889
Incident error rate = 30,000 / 138,889 = 21.6%

Burn rate = 21.6% / 0.1% = 216x

At 216x burn rate, a 14.4x page alert would fire immediately.
The 30-day budget would be exhausted in 30/216 = 3.3 hours.
```

**(f) Error budget policy actions:**
With 16% budget remaining (between 5-25%):
- Feature freeze for non-critical changes
- Reliability work takes priority
- Conduct postmortem for the Day 15 incident
- Review whether the 99.9% target is appropriate given the architecture
- The team must demonstrate reliability improvements before resuming feature velocity
- Next sprint should be a "reliability sprint" focused on preventing similar incidents

</details>

### Exercise 3: Burn Rate Alert Design

Design a complete set of burn rate alerts for a service with a 99.5% latency SLO (99.5% of requests must complete within 500ms) over a 30-day window. Write the Prometheus recording rules and alerting rules. Include: critical (page), warning (ticket), and informational severity levels.

<details>
<summary>Show Answer</summary>

```yaml
groups:
  - name: service_latency_slo_recording
    rules:
      # Good events: requests completing within 500ms
      - record: service:latency_slo_error_ratio:rate5m
        expr: |
          1 - (
            sum(rate(http_request_duration_seconds_bucket{job="my-service",le="0.5"}[5m]))
            / sum(rate(http_request_duration_seconds_count{job="my-service"}[5m]))
          )

      - record: service:latency_slo_error_ratio:rate30m
        expr: |
          1 - (
            sum(rate(http_request_duration_seconds_bucket{job="my-service",le="0.5"}[30m]))
            / sum(rate(http_request_duration_seconds_count{job="my-service"}[30m]))
          )

      - record: service:latency_slo_error_ratio:rate1h
        expr: |
          1 - (
            sum(rate(http_request_duration_seconds_bucket{job="my-service",le="0.5"}[1h]))
            / sum(rate(http_request_duration_seconds_count{job="my-service"}[1h]))
          )

      - record: service:latency_slo_error_ratio:rate6h
        expr: |
          1 - (
            sum(rate(http_request_duration_seconds_bucket{job="my-service",le="0.5"}[6h]))
            / sum(rate(http_request_duration_seconds_count{job="my-service"}[6h]))
          )

      - record: service:latency_slo_error_ratio:rate1d
        expr: |
          1 - (
            sum(rate(http_request_duration_seconds_bucket{job="my-service",le="0.5"}[1d]))
            / sum(rate(http_request_duration_seconds_count{job="my-service"}[1d]))
          )

      - record: service:latency_slo_error_ratio:rate2h
        expr: |
          1 - (
            sum(rate(http_request_duration_seconds_bucket{job="my-service",le="0.5"}[2h]))
            / sum(rate(http_request_duration_seconds_count{job="my-service"}[2h]))
          )

  - name: service_latency_slo_alerts
    rules:
      # SLO budget rate = 1 - 0.995 = 0.005

      # Critical page: 14.4x burn rate, 1h + 5m windows
      # Threshold: 14.4 * 0.005 = 0.072
      - alert: ServiceLatencySLOCritical
        expr: |
          service:latency_slo_error_ratio:rate1h > 0.072
          and
          service:latency_slo_error_ratio:rate5m > 0.072
        for: 2m
        labels:
          severity: critical
          slo: service-latency
        annotations:
          summary: "Service latency SLO critical burn rate (14.4x)"
          description: |
            Budget will exhaust in ~2 days. Current 1h slow-request ratio: {{ $value }}.
            SLO: 99.5% of requests < 500ms.
          runbook: "https://wiki.example.com/runbooks/latency-slo"

      # Warning ticket: 6x burn rate, 6h + 30m windows
      # Threshold: 6 * 0.005 = 0.030
      - alert: ServiceLatencySLOWarning
        expr: |
          service:latency_slo_error_ratio:rate6h > 0.030
          and
          service:latency_slo_error_ratio:rate30m > 0.030
        for: 5m
        labels:
          severity: warning
          slo: service-latency
        annotations:
          summary: "Service latency SLO elevated burn rate (6x)"
          description: "Budget will exhaust in ~5 days at current rate."

      # Info ticket: 3x burn rate, 1d + 2h windows
      # Threshold: 3 * 0.005 = 0.015
      - alert: ServiceLatencySLOElevated
        expr: |
          service:latency_slo_error_ratio:rate1d > 0.015
          and
          service:latency_slo_error_ratio:rate2h > 0.015
        for: 15m
        labels:
          severity: info
          slo: service-latency
        annotations:
          summary: "Service latency SLO slightly elevated burn rate (3x)"
          description: "Budget will exhaust in ~10 days if trend continues."
```

**Key design points:**
- Each alert uses two windows: long window detects sustained issues, short window prevents stale alerts.
- Threshold calculation: burn_rate * (1 - SLO target) = burn_rate * 0.005.
- `for` duration increases with lower severity to reduce noise.
- Critical alerts page immediately; warning and info create tickets.

</details>

---

## References

- [The Art of SLOs (Google Cloud)](https://sre.google/resources/practices-and-processes/art-of-slos/)
- [Google SRE Workbook -- Implementing SLOs](https://sre.google/workbook/implementing-slos/)
- [Implementing Service Level Objectives (O'Reilly)](https://www.oreilly.com/library/view/implementing-service-level/9781492076803/)
- [Sloth -- SLO as Code](https://github.com/slok/sloth)
- [OpenSLO Specification](https://openslo.com/)
- [Google SRE Book -- Service Level Objectives](https://sre.google/sre-book/service-level-objectives/)
