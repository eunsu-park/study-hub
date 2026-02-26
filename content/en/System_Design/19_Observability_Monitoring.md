# 19. Observability and Monitoring

**Previous**: [Practical Design Examples 2](./18_Design_Example_2.md) | **Next**: [Search Systems](./20_Search_Systems.md)

Difficulty: ⭐⭐⭐⭐

## Learning Objectives

1. Define observability and distinguish it from traditional monitoring, explaining why it matters for distributed systems
2. Describe the three pillars of observability (metrics, logs, traces) and how they complement each other during incident investigation
3. Design a metrics pipeline using Prometheus-style pull-based collection, time-series storage, and Grafana dashboards
4. Architect a centralized logging system using the ELK stack or Loki for ingestion, indexing, and querying at scale
5. Implement distributed tracing with OpenTelemetry to correlate requests across service boundaries
6. Define SLIs, SLOs, and error budgets, and design alerting strategies that minimize noise while catching real incidents

---

When a user reports that "the app is slow," an observable system lets you pinpoint in minutes whether the bottleneck is a saturated database connection pool, a slow third-party API, or a garbage collection pause on a specific pod. Without observability, teams resort to guesswork and deploy "fix attempts" blindly. Google's SRE practices, which formalized SLIs, SLOs, and error budgets, have shown that investing in observability directly reduces mean time to recovery and on-call burnout across the industry.

## Table of Contents

1. [Observability Fundamentals](#1-observability-fundamentals)
2. [Metrics and Time-Series Data](#2-metrics-and-time-series-data)
3. [Logging at Scale](#3-logging-at-scale)
4. [Distributed Tracing](#4-distributed-tracing)
5. [Alerting and SLOs](#5-alerting-and-slos)
6. [OpenTelemetry](#6-opentelemetry)
7. [Practice Problems](#7-practice-problems)

---

## 1. Observability Fundamentals

### 1.1 Three Pillars of Observability

```
┌─────────────────────────────────────────────────────────────────┐
│              Three Pillars of Observability                      │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Metrics     │  │    Logs      │  │   Traces     │          │
│  │              │  │              │  │              │          │
│  │ "What is     │  │ "What       │  │ "What path   │          │
│  │  happening?" │  │  happened?"  │  │  did it      │          │
│  │              │  │              │  │  take?"      │          │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤          │
│  │ Numeric      │  │ Structured   │  │ Request-     │          │
│  │ time-series  │  │ events with  │  │ scoped       │          │
│  │ data         │  │ context      │  │ causality    │          │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤          │
│  │ Low cost     │  │ Medium cost  │  │ Higher cost  │          │
│  │ per signal   │  │ per signal   │  │ per signal   │          │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤          │
│  │ Prometheus   │  │ ELK Stack    │  │ Jaeger       │          │
│  │ Grafana      │  │ Loki         │  │ Zipkin       │          │
│  │ Datadog      │  │ Fluentd      │  │ Tempo        │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                 │
│             Correlation via Trace IDs and Labels                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Monitoring vs Observability

```
┌─────────────────────────────┬───────────────────────────────────┐
│ Monitoring                  │ Observability                      │
├─────────────────────────────┼───────────────────────────────────┤
│ Known unknowns              │ Unknown unknowns                   │
│ "Is CPU > 90%?"             │ "Why is latency high?"             │
│ Dashboard-driven            │ Exploration-driven                 │
│ Predefined alerts           │ Ad-hoc investigation               │
│ Reactive                    │ Proactive                          │
│ Works for simple systems    │ Essential for distributed systems  │
└─────────────────────────────┴───────────────────────────────────┘
```

---

## 2. Metrics and Time-Series Data

### 2.1 Metric Types

```
┌─────────────────────────────────────────────────────────────────┐
│              Four Golden Signals (Google SRE)                    │
│                                                                 │
│  1. Latency     — Time to serve a request                       │
│  2. Traffic     — Demand on the system (RPS)                    │
│  3. Errors      — Rate of failed requests                       │
│  4. Saturation  — How "full" the system is                      │
│                                                                 │
│              RED Method (for microservices)                      │
│                                                                 │
│  1. Rate     — Requests per second                              │
│  2. Errors   — Failed requests per second                       │
│  3. Duration — Distribution of request latencies                │
│                                                                 │
│              USE Method (for infrastructure)                     │
│                                                                 │
│  1. Utilization  — % of resource busy                           │
│  2. Saturation   — Queue depth, pending work                    │
│  3. Errors       — Error count                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Prometheus Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              Prometheus Ecosystem                                │
│                                                                 │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │ Application │     │ Application │     │   Node      │       │
│  │  /metrics   │     │  /metrics   │     │  Exporter   │       │
│  └──────┬──────┘     └──────┬──────┘     └──────┬──────┘       │
│         │                   │                   │               │
│         └───────────┬───────┘───────────────────┘               │
│                     │  scrape (pull)                             │
│              ┌──────▼──────┐                                    │
│              │  Prometheus  │                                    │
│              │   Server     │──────▶ AlertManager ──▶ PagerDuty │
│              │  (TSDB)      │               │        Slack      │
│              └──────┬──────┘               │        Email      │
│                     │                                           │
│              ┌──────▼──────┐                                    │
│              │   Grafana    │                                    │
│              │ (Dashboard)  │                                    │
│              └─────────────┘                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Prometheus Metric Types

```yaml
# Counter — monotonically increasing (requests, errors)
http_requests_total{method="GET", path="/api/users", status="200"} 12345

# Gauge — goes up and down (temperature, queue size)
process_memory_bytes 1073741824

# Histogram — samples in configurable buckets
http_request_duration_seconds_bucket{le="0.1"} 1000
http_request_duration_seconds_bucket{le="0.5"} 1200
http_request_duration_seconds_bucket{le="1.0"} 1250
http_request_duration_seconds_count 1280
http_request_duration_seconds_sum 320.5

# Summary — similar to histogram with quantiles
http_request_duration_seconds{quantile="0.5"} 0.042
http_request_duration_seconds{quantile="0.9"} 0.087
http_request_duration_seconds{quantile="0.99"} 0.235
```

### 2.4 PromQL Queries

```promql
# Request rate over 5 minutes
rate(http_requests_total[5m])

# Error rate percentage
sum(rate(http_requests_total{status=~"5.."}[5m])) /
sum(rate(http_requests_total[5m])) * 100

# 95th percentile latency
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Top 5 endpoints by request rate
topk(5, sum by (path)(rate(http_requests_total[5m])))

# Memory usage percentage
(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes)
/ node_memory_MemTotal_bytes * 100

# Predict disk full in 4 hours
predict_linear(node_filesystem_free_bytes[1h], 4*3600) < 0
```

---

## 3. Logging at Scale

### 3.1 Structured Logging

```json
{
  "timestamp": "2026-02-15T10:30:00Z",
  "level": "ERROR",
  "service": "order-service",
  "trace_id": "abc123def456",
  "span_id": "789ghi",
  "user_id": "u-42",
  "message": "Failed to process order",
  "error": "PaymentDeclined",
  "order_id": "ord-789",
  "amount": 129.99,
  "duration_ms": 1250
}
```

### 3.2 ELK Stack Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              ELK Stack (Elastic Stack)                           │
│                                                                 │
│  Applications                                                   │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐                           │
│  │ Service  │ │ Service  │ │ Service  │                           │
│  │    A     │ │    B     │ │    C     │                           │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘                           │
│       │            │            │                               │
│       └────────────┼────────────┘                               │
│                    ▼                                            │
│  ┌──────────────────────────────────┐                           │
│  │     Filebeat / Fluentd           │  Log Shippers             │
│  └──────────────┬───────────────────┘                           │
│                 ▼                                               │
│  ┌──────────────────────────────────┐                           │
│  │        Logstash / Kafka          │  Processing / Buffer      │
│  │   (parse, filter, enrich)        │                           │
│  └──────────────┬───────────────────┘                           │
│                 ▼                                               │
│  ┌──────────────────────────────────┐                           │
│  │       Elasticsearch              │  Storage & Search         │
│  │   (index, full-text search)      │                           │
│  └──────────────┬───────────────────┘                           │
│                 ▼                                               │
│  ┌──────────────────────────────────┐                           │
│  │          Kibana                  │  Visualization            │
│  │   (dashboards, queries)          │                           │
│  └──────────────────────────────────┘                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Grafana Loki (Lightweight Alternative)

```
┌─────────────────────────────────────────────────────────────────┐
│              Grafana Loki Stack                                  │
│                                                                 │
│  • Does NOT index log content (only labels)                     │
│  • Much cheaper storage than Elasticsearch                      │
│  • LogQL query language (similar to PromQL)                     │
│  • Ideal for Kubernetes environments                            │
│                                                                 │
│  Promtail ──▶ Loki ──▶ Grafana                                  │
│  (agent)      (store)   (query/visualize)                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

```
# LogQL examples
{service="order-service"} |= "error"                # contains "error"
{service="order-service"} | json | status >= 500     # JSON parsing + filter
{service="order-service"} | json | line_format "{{.message}}"
rate({service="order-service"} |= "error" [5m])      # error rate
```

### 3.4 Log Levels and Best Practices

```
Level    │ When to Use
─────────┼──────────────────────────────────────────
TRACE    │ Very fine-grained (usually disabled)
DEBUG    │ Development troubleshooting
INFO     │ Normal operations, business events
WARN     │ Recoverable issues, degraded service
ERROR    │ Failures requiring attention
FATAL    │ Application cannot continue
```

**Best Practices:**
- Use structured logging (JSON) over plain text
- Include correlation IDs (trace_id) in every log
- Log at appropriate levels (no INFO spam in production)
- Set retention policies (7d hot, 30d warm, 90d cold)
- Avoid logging sensitive data (PII, credentials)

---

## 4. Distributed Tracing

### 4.1 Trace Anatomy

```
┌─────────────────────────────────────────────────────────────────┐
│              Distributed Trace Example                           │
│                                                                 │
│  Trace ID: abc-123-def                                          │
│                                                                 │
│  ┌────────────────────────────────────────────────────┐         │
│  │ Span A: API Gateway (200ms)                        │         │
│  │  ├─────────────────────────┐                       │         │
│  │  │ Span B: Auth (30ms)     │                       │         │
│  │  └─────────────────────────┘                       │         │
│  │         ├──────────────────────────────┐            │         │
│  │         │ Span C: Order Service (120ms)│            │         │
│  │         │  ├───────────────┐           │            │         │
│  │         │  │ Span D: DB   │           │            │         │
│  │         │  │ Query (15ms) │           │            │         │
│  │         │  └───────────────┘           │            │         │
│  │         │       ├──────────────┐       │            │         │
│  │         │       │ Span E:     │       │            │         │
│  │         │       │ Payment     │       │            │         │
│  │         │       │ (80ms)      │       │            │         │
│  │         │       └──────────────┘       │            │         │
│  │         └──────────────────────────────┘            │         │
│  └────────────────────────────────────────────────────┘         │
│                                                                 │
│  0ms            100ms           200ms                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Tracing Systems

```
┌────────────────┬──────────────────────────────────────────────────┐
│ Tool           │ Description                                      │
├────────────────┼──────────────────────────────────────────────────┤
│ Jaeger         │ Open source, Uber-originated, CNCF project       │
│ Zipkin         │ Open source, Twitter-originated                   │
│ Grafana Tempo  │ Cost-efficient, only stores trace IDs             │
│ AWS X-Ray      │ Managed service for AWS workloads                 │
│ Datadog APM    │ Commercial, integrated with metrics/logs          │
└────────────────┴──────────────────────────────────────────────────┘
```

### 4.3 Context Propagation

```
┌─────────────────────────────────────────────────────────────────┐
│              W3C Trace Context Headers                           │
│                                                                 │
│  HTTP Request:                                                  │
│  ┌─────────────────────────────────────────────────────┐        │
│  │ traceparent: 00-{trace-id}-{span-id}-{flags}       │        │
│  │ tracestate: vendor1=value1,vendor2=value2            │        │
│  └─────────────────────────────────────────────────────┘        │
│                                                                 │
│  Example:                                                       │
│  traceparent: 00-abc123def456-789ghi012-01                      │
│  ┌──────┬──────────────┬────────────┬──────┐                    │
│  │ ver  │  trace-id    │  span-id   │flags │                    │
│  │ 00   │ abc123def456 │ 789ghi012  │ 01   │                    │
│  └──────┴──────────────┴────────────┴──────┘                    │
│                                                                 │
│  Service A ──(traceparent)──▶ Service B ──(traceparent)──▶ C    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.4 Sampling Strategies

```
┌─────────────────────────────────────────────────────────────────┐
│              Sampling Strategies                                 │
│                                                                 │
│  1. Head-based sampling                                         │
│     Decision at trace start: sample 10% of all requests         │
│     + Simple, low overhead                                      │
│     − May miss important traces                                 │
│                                                                 │
│  2. Tail-based sampling                                         │
│     Decision after trace completes: keep errors + slow traces   │
│     + Captures interesting traces                               │
│     − Higher memory usage (buffer all spans)                    │
│                                                                 │
│  3. Rate-limited sampling                                       │
│     Keep N traces per second per service                        │
│     + Predictable cost                                          │
│     − May miss bursts                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Alerting and SLOs

### 5.1 SLI / SLO / SLA

```
┌─────────────────────────────────────────────────────────────────┐
│              SLI → SLO → SLA                                    │
│                                                                 │
│  SLI (Service Level Indicator)                                  │
│  ├── What you measure                                           │
│  ├── "Proportion of requests < 200ms"                           │
│  └── "Proportion of requests returning 2xx"                     │
│                                                                 │
│  SLO (Service Level Objective)                                  │
│  ├── Internal target                                            │
│  ├── "99.9% of requests < 200ms over 30 days"                  │
│  └── "99.95% availability per month"                            │
│                                                                 │
│  SLA (Service Level Agreement)                                  │
│  ├── External contract with consequences                        │
│  └── "99.9% uptime or service credits issued"                  │
│                                                                 │
│  Rule: SLO should be stricter than SLA                          │
│        (e.g., SLO = 99.95% when SLA = 99.9%)                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Error Budgets

```
┌─────────────────────────────────────────────────────────────────┐
│              Error Budget Concept                                │
│                                                                 │
│  SLO = 99.9% availability                                      │
│  Error Budget = 100% - 99.9% = 0.1%                            │
│                                                                 │
│  Per 30 days: 0.1% × 30 × 24 × 60 = 43.2 minutes of downtime │
│                                                                 │
│  Error Budget Remaining:                                        │
│  ┌────────────────────────────────────────────────────┐         │
│  │████████████████████████████████████░░░░░░░░░░░░░░░│         │
│  │ 70% remaining              30% consumed           │         │
│  └────────────────────────────────────────────────────┘         │
│                                                                 │
│  Policy:                                                        │
│  • Budget > 50%: Deploy freely, experiment                      │
│  • Budget 20-50%: Careful deployments, extra testing            │
│  • Budget < 20%: Freeze features, focus on reliability          │
│  • Budget = 0%: Emergency freeze until budget replenishes       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 Alerting Best Practices

```yaml
# Prometheus alerting rules example
groups:
  - name: slo-alerts
    rules:
      # Multi-window, multi-burn-rate alert
      - alert: HighErrorRate
        expr: |
          (
            sum(rate(http_requests_total{status=~"5.."}[5m]))
            / sum(rate(http_requests_total[5m]))
          ) > 14.4 * 0.001  # 14.4x burn rate for 5m window
          AND
          (
            sum(rate(http_requests_total{status=~"5.."}[1h]))
            / sum(rate(http_requests_total[1h]))
          ) > 14.4 * 0.001
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Error rate exceeds SLO burn rate"

      - alert: HighLatency
        expr: |
          histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m])) > 1.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "P99 latency above 1s"
```

**Alerting Anti-Patterns:**
- Alert fatigue: too many non-actionable alerts
- Missing runbooks: alerts without remediation steps
- No owner: alerts routed to "everyone"
- Threshold-only: static thresholds without trend analysis

---

## 6. OpenTelemetry

### 6.1 OpenTelemetry Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              OpenTelemetry (OTel) Architecture                   │
│                                                                 │
│  Application                                                    │
│  ┌─────────────────────────────────────────────────┐            │
│  │  OTel SDK                                       │            │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐        │            │
│  │  │ Traces   │ │ Metrics  │ │  Logs    │        │            │
│  │  │ API      │ │ API      │ │ API      │        │            │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘        │            │
│  │       └─────────────┼────────────┘              │            │
│  │                     ▼                           │            │
│  │              OTLP Exporter                      │            │
│  └─────────────────────┬───────────────────────────┘            │
│                        ▼                                        │
│  ┌─────────────────────────────────────────────────┐            │
│  │           OTel Collector                        │            │
│  │  ┌────────┐  ┌────────────┐  ┌──────────┐      │            │
│  │  │Receivers│  │ Processors │  │ Exporters│      │            │
│  │  │ OTLP   │→│ Batch      │→│ Jaeger   │      │            │
│  │  │ Zipkin │  │ Filter     │  │ Prometheus│      │            │
│  │  │ Kafka  │  │ Tail-sample│  │ Loki     │      │            │
│  │  └────────┘  └────────────┘  └──────────┘      │            │
│  └─────────────────────────────────────────────────┘            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 OTel Collector Configuration

```yaml
# otel-collector-config.yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  batch:
    timeout: 5s
    send_batch_size: 1000
  memory_limiter:
    check_interval: 1s
    limit_mib: 512

exporters:
  prometheus:
    endpoint: 0.0.0.0:8889
  jaeger:
    endpoint: jaeger:14250
    tls:
      insecure: true
  loki:
    endpoint: http://loki:3100/loki/api/v1/push

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [jaeger]
    metrics:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [prometheus]
    logs:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [loki]
```

### 6.3 Instrumentation Example (Python)

```python
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Setup tracing
provider = TracerProvider()
provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint="localhost:4317"))
)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

# Setup metrics
meter = metrics.get_meter(__name__)
request_counter = meter.create_counter(
    "http.requests", description="Total HTTP requests"
)
request_duration = meter.create_histogram(
    "http.request.duration", description="Request duration in ms"
)

# Usage
@tracer.start_as_current_span("process_order")
def process_order(order_id):
    request_counter.add(1, {"endpoint": "/orders", "method": "POST"})

    with tracer.start_as_current_span("validate_order") as span:
        span.set_attribute("order.id", order_id)
        validate(order_id)

    with tracer.start_as_current_span("charge_payment"):
        charge(order_id)
```

---

## 7. Practice Problems

### Problem 1: Design Monitoring for a Microservices Platform
You are designing the observability stack for an e-commerce platform with 20 microservices.

**Key considerations:**
- What metrics would you collect from each service?
- How would you correlate logs across services?
- What sampling strategy for traces?
- Define SLOs for the checkout service.

```
Example approach:

Metrics (RED for each service):
- Rate: http_requests_total by service, method, status
- Errors: http_requests_total{status=~"5.."} / total
- Duration: http_request_duration_seconds histogram

Logging:
- Structured JSON with trace_id in every log
- Centralized via Loki or Elasticsearch
- Retention: 7d hot, 30d warm, 90d archive

Tracing:
- Tail-based sampling: keep all errors + p99 latencies
- Head-based: 10% sample for normal traffic
- Jaeger or Tempo as backend

Checkout SLOs:
- Availability: 99.95% success rate (30-day window)
- Latency: p99 < 2s, p50 < 500ms
- Error budget: ~21.6 min/month
```

### Problem 2: Alert Design
Design an alerting strategy that avoids alert fatigue.

```
Example answer:

Multi-burn-rate alerting:
- 2% budget consumed in 1 hour  → page (critical)
- 5% budget consumed in 6 hours → page (warning)
- 10% budget consumed in 3 days → ticket (low)

Routing:
- Critical → PagerDuty → on-call engineer
- Warning  → Slack #alerts → team lead
- Low      → Jira ticket → backlog

Every alert must have:
- Runbook link
- Dashboard link
- Expected impact
- Suggested remediation
```

### Problem 3: Observability Cost Optimization
Your team spends $50K/month on observability. Reduce costs by 40%.

```
Example answer:

1. Metrics: Drop unused metrics (audit dashboards)
   - Reduce cardinality (fewer label values)
   - Increase scrape intervals for non-critical services (15s → 60s)

2. Logs: Switch from ELK to Loki
   - Stop indexing full log content
   - Reduce log verbosity (DEBUG → INFO in production)
   - Shorter retention (90d → 30d for non-regulated data)

3. Traces: Implement tail-based sampling
   - Keep 100% of errors and slow traces
   - Sample 1% of successful traces
   - Use Grafana Tempo (cheaper than Jaeger at scale)

4. Architecture:
   - Self-host OpenTelemetry Collector
   - Use object storage (S3) for cold data
   - Aggregate metrics at collector level
```

---

**Previous**: [Practical Design Examples 2](./18_Design_Example_2.md) | **Next**: [Search Systems](./20_Search_Systems.md)

---

## References
- [Google SRE Book: Monitoring](https://sre.google/sre-book/monitoring-distributed-systems/)
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Loki](https://grafana.com/oss/loki/)
