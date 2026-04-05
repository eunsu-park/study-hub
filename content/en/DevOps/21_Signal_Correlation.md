# 21. Signal Correlation

**Previous**: [SLO Engineering](./20_SLO_Engineering.md) | **Next**: [Advanced Metrics Architecture](./22_Advanced_Metrics_Architecture.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why correlating metrics, logs, and traces is essential for effective debugging in distributed systems
2. Implement trace-to-log linking using trace IDs and structured logging
3. Use Prometheus exemplars to connect aggregated metrics to specific traces
4. Design a unified telemetry architecture where all signals share correlation identifiers
5. Build Grafana dashboards that enable seamless navigation between metrics, logs, and traces
6. Apply correlation techniques to reduce mean time to resolution (MTTR) during incidents

---

Individual telemetry signals are powerful in isolation, but their true value emerges when they are correlated. A metric tells you *something is wrong*. A trace tells you *which request path is affected*. A log tells you *exactly what happened*. Correlation is the glue that lets you navigate from a metric spike to the specific traces and log entries that explain the root cause -- turning hours of debugging into minutes.

> **Analogy -- Crime Scene Investigation**: A detective does not solve a case with fingerprints alone (metrics), witness statements alone (logs), or security camera footage alone (traces). They solve it by *correlating* all three: the fingerprint on the door handle (metric spike at 14:00), cross-referenced with camera footage showing a specific person entering at 14:00 (trace), and the witness statement describing what happened inside (log entries). Without correlation, each piece of evidence is a disconnected clue.

## 1. The Correlation Problem

### 1.1 Debugging Without Correlation

A typical debugging session without correlation:

```
1. Alert fires: "Payment service error rate > 1%"
   → Open Grafana, see the spike in the error rate panel
   → But WHICH requests are failing?

2. Open Kibana/Loki, search for payment-service errors
   → Find thousands of error log lines
   → But which ones correspond to the metric spike?
   → Filter by time window... still hundreds of entries

3. Open Jaeger, search for payment-service traces with errors
   → Find error traces
   → But do these traces correspond to the log entries?

4. Manual correlation: match timestamps, request IDs, guess
   → Takes 30-60 minutes of context-switching between tools
```

### 1.2 Debugging With Correlation

```
1. Alert fires: "Payment service error rate > 1%"
   → Open Grafana, see the spike
   → Click on the spike → see exemplar traces

2. Click exemplar → opens Jaeger with the specific trace
   → See that payment-service → stripe-api call is timing out
   → See span attributes: stripe_api_version, endpoint, timeout_ms

3. Click "View logs" on the span → opens Loki
   → See the exact log entries for this trace:
     "Stripe API timeout after 30s, retry 3/3 exhausted"
     "Circuit breaker tripped for stripe-payments endpoint"

4. Root cause identified in under 5 minutes
```

### 1.3 Correlation Identifiers

The foundation of signal correlation is shared identifiers across all telemetry:

| Identifier | Scope | Propagation |
|-----------|-------|-------------|
| **Trace ID** | Single request across all services | W3C Trace Context header |
| **Span ID** | Single operation within a service | Part of trace context |
| **Request ID** | Application-level request tracking | Custom header (X-Request-ID) |
| **Session ID** | User session across multiple requests | Cookie or token |
| **Deployment ID** | Release version that generated the telemetry | Resource attribute |

---

## 2. Trace-to-Log Linking

### 2.1 Injecting Trace Context into Logs

The most impactful correlation technique is adding trace_id and span_id to every log entry:

```python
"""Trace-to-log linking with OpenTelemetry and structlog."""
import structlog
from opentelemetry import trace

def add_trace_context(logger, method_name, event_dict):
    """Structlog processor that injects OTel trace context into logs."""
    span = trace.get_current_span()
    if span.is_recording():
        ctx = span.get_span_context()
        event_dict["trace_id"] = format(ctx.trace_id, "032x")
        event_dict["span_id"] = format(ctx.span_id, "016x")
        event_dict["trace_flags"] = ctx.trace_flags
    return event_dict

# Configure structlog with trace context injection
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        add_trace_context,                          # Inject trace context
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]
)

logger = structlog.get_logger()

# Every log entry now includes trace_id and span_id automatically
def process_payment(payment_id: str):
    with trace.get_tracer("payment-service").start_as_current_span("process_payment"):
        logger.info("payment.processing_started", payment_id=payment_id)
        # ... business logic ...
        logger.info("payment.completed", payment_id=payment_id, amount=100.0)
```

**Resulting log entry:**

```json
{
  "event": "payment.processing_started",
  "payment_id": "pay_abc123",
  "trace_id": "0af7651916cd43dd8448eb211c80319c",
  "span_id": "b7ad6b7169203331",
  "trace_flags": 1,
  "level": "info",
  "timestamp": "2025-03-15T10:30:00.000Z"
}
```

### 2.2 Log Framework Integration

| Framework | OTel Integration |
|-----------|-----------------|
| **Python structlog** | Custom processor (shown above) |
| **Python logging** | `opentelemetry-instrumentation-logging` auto-injection |
| **Java Log4j2** | `opentelemetry-log4j-context-data-2.17-autoconfigure` |
| **Java Logback** | `opentelemetry-logback-mdc-1.0` |
| **Go slog** | Manual injection via `trace.SpanFromContext(ctx)` |
| **Node.js Pino** | `@opentelemetry/instrumentation-pino` |

### 2.3 Loki Label Configuration for Trace Linking

```yaml
# Loki configuration: extract trace_id for indexed lookup
# promtail pipeline stage
scrape_configs:
  - job_name: kubernetes-pods
    kubernetes_sd_configs:
      - role: pod
    pipeline_stages:
      - json:
          expressions:
            trace_id: trace_id
            span_id: span_id
            level: level
            service: service_name

      - labels:
          level:
          service:

      - structured_metadata:
          trace_id:
          span_id:

      # Derive trace link for Grafana
      - template:
          source: trace_link
          template: '{{ .trace_id }}'
      - output:
          source: message
```

### 2.4 Grafana Trace-to-Log Configuration

```jsonc
// Grafana data source configuration: Tempo → Loki linking
{
  "name": "Tempo",
  "type": "tempo",
  "jsonData": {
    "tracesToLogs": {
      "datasourceUid": "loki-datasource-uid",
      "filterByTraceID": true,
      "filterBySpanID": true,
      "mapTagNamesEnabled": true,
      "mappedTags": [
        { "key": "service.name", "value": "service" }
      ],
      "lokiSearch": {
        "datasourceUid": "loki-datasource-uid"
      }
    },
    "tracesToMetrics": {
      "datasourceUid": "prometheus-datasource-uid",
      "queries": [
        {
          "name": "Request rate",
          "query": "sum(rate(http_requests_total{service=\"${__span.tags.service.name}\"}[$__rate_interval]))"
        }
      ]
    }
  }
}
```

---

## 3. Exemplars

### 3.1 What Are Exemplars

Exemplars are references from aggregated metric data points to specific trace IDs. They answer: "Which specific request contributed to this metric value?"

```
Metric: http_request_duration_seconds (histogram)

Without exemplars:
  bucket{le="0.5"}: 9500
  bucket{le="1.0"}: 9800
  bucket{le="5.0"}: 9990
  → "200 requests were slower than 1 second" -- but WHICH requests?

With exemplars:
  bucket{le="0.5"}: 9500
  bucket{le="1.0"}: 9800  ← exemplar: trace_id=abc123, value=0.85s
  bucket{le="5.0"}: 9990  ← exemplar: trace_id=def456, value=3.2s
  → "Request abc123 took 850ms, request def456 took 3.2s -- click to view traces"
```

### 3.2 Implementing Exemplars in Go

```go
package main

import (
    "math/rand"
    "net/http"
    "time"

    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promhttp"
    "go.opentelemetry.io/otel/trace"
)

var requestDuration = prometheus.NewHistogramVec(
    prometheus.HistogramOpts{
        Name:    "http_request_duration_seconds",
        Help:    "HTTP request duration in seconds",
        Buckets: prometheus.DefBuckets,
    },
    []string{"method", "endpoint", "status"},
)

func handleRequest(w http.ResponseWriter, r *http.Request) {
    start := time.Now()
    // ... handle request ...
    duration := time.Since(start).Seconds()

    // Get trace context for exemplar
    span := trace.SpanFromContext(r.Context())
    traceID := span.SpanContext().TraceID().String()

    // Record metric WITH exemplar
    requestDuration.WithLabelValues(
        r.Method, r.URL.Path, "200",
    ).(prometheus.ExemplarObserver).ObserveWithExemplar(
        duration,
        prometheus.Labels{"trace_id": traceID},
    )
}
```

### 3.3 Implementing Exemplars in Python

```python
"""Exemplar support with prometheus_client."""
from prometheus_client import Histogram, REGISTRY
from opentelemetry import trace

request_duration = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration",
    labelnames=["method", "endpoint", "status"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

def handle_request(method: str, endpoint: str):
    """Record request duration with exemplar linking to trace."""
    start = time.monotonic()

    # ... handle the request ...

    duration = time.monotonic() - start
    span = trace.get_current_span()
    trace_id = format(span.get_span_context().trace_id, "032x")

    # Record with exemplar
    request_duration.labels(
        method=method, endpoint=endpoint, status="200"
    ).observe(duration, exemplar={"trace_id": trace_id})
```

### 3.4 Prometheus Exemplar Storage

Exemplars require Prometheus to be configured with the `--enable-feature=exemplar-storage` flag:

```yaml
# prometheus.yml -- enable exemplar support
global:
  scrape_interval: 15s

# Exemplar storage configuration (Prometheus startup flags)
# --storage.exemplars.exemplars-limit=100000
# --enable-feature=exemplar-storage

# Scrape config must use OpenMetrics format for exemplar support
scrape_configs:
  - job_name: "webapp"
    scrape_interval: 15s
    metrics_path: /metrics
    # Required for exemplar support
    honor_timestamps: true
    scrape_protocols: ["OpenMetricsText1.0.0"]
    static_configs:
      - targets: ["webapp:8080"]
```

### 3.5 Querying Exemplars in Grafana

In Grafana, enable exemplars on a time-series panel:

```
Panel settings:
  Data source: Prometheus
  Query: histogram_quantile(0.99, sum by (le) (rate(http_request_duration_seconds_bucket[5m])))

  Options:
    ☑ Exemplars  (toggle ON)

  Exemplar data source: Tempo (or Jaeger)
  URL label: trace_id
```

When exemplars are enabled, clickable dots appear on the time-series graph. Each dot represents a specific request. Clicking opens the corresponding trace.

---

## 4. Metrics-to-Traces Correlation

### 4.1 From Metric Spike to Root Cause

```
Step 1: Dashboard shows latency spike at 14:00
  PromQL: histogram_quantile(0.99, ...)
  → p99 jumped from 200ms to 2s

Step 2: Click exemplar dot on the spike
  → Trace ID: abc123 (duration: 2.1s)

Step 3: View trace in Jaeger/Tempo
  → api-gateway (50ms)
    → order-service (100ms)
      → inventory-service (1,800ms)  ← BOTTLENECK
        → postgres: SELECT ... (1,750ms)  ← SLOW QUERY

Step 4: Click "View logs" on the postgres span
  → Log: "Slow query detected: sequential scan on orders table
          (missing index on customer_id column)"

Step 5: Root cause identified:
  → Missing database index caused sequential scans under load
  → Fix: CREATE INDEX idx_orders_customer_id ON orders(customer_id)
```

### 4.2 Trace-to-Metrics Navigation

The reverse direction is also valuable -- from a trace, navigate to related metrics:

```python
# In Grafana Tempo data source config: "Traces to Metrics"
# When viewing a trace, add links to related metrics dashboards

traces_to_metrics_queries = [
    {
        "name": "Service Request Rate",
        "query": 'sum(rate(http_requests_total{service="${__span.tags["service.name"]}"}[5m]))',
    },
    {
        "name": "Service Error Rate",
        "query": 'sum(rate(http_requests_total{service="${__span.tags["service.name"]}",status=~"5.."}[5m])) / sum(rate(http_requests_total{service="${__span.tags["service.name"]}"}[5m]))',
    },
    {
        "name": "Database Connection Pool",
        "query": 'db_pool_active_connections{service="${__span.tags["service.name"]}"}',
    },
]
```

---

## 5. Log-to-Trace Correlation

### 5.1 From Log Entry to Trace

When investigating a log entry, the trace_id field enables one-click navigation to the full distributed trace:

```
Log entry in Loki:
{
  "level": "error",
  "service": "payment-service",
  "message": "Payment failed: insufficient funds",
  "trace_id": "abc123def456",
  "span_id": "789ghi",
  "customer_id": "cust_001",
  "amount": 150.00
}

→ Click trace_id → Opens Tempo/Jaeger with trace abc123def456
→ See the full request path that led to this error
→ Was the error caused by the customer or by a system issue?
```

### 5.2 Derived Fields in Grafana Loki

```jsonc
// Grafana Loki data source: derived fields configuration
{
  "name": "Loki",
  "type": "loki",
  "jsonData": {
    "derivedFields": [
      {
        "name": "TraceID",
        "matcherRegex": "\"trace_id\":\"(\\w+)\"",
        "url": "${__value.raw}",
        "datasourceUid": "tempo-datasource-uid",
        "urlDisplayLabel": "View Trace"
      },
      {
        "name": "SpanID",
        "matcherRegex": "\"span_id\":\"(\\w+)\"",
        "url": "${__value.raw}",
        "datasourceUid": "tempo-datasource-uid"
      }
    ]
  }
}
```

---

## 6. Unified Telemetry Architecture

### 6.1 The Correlation Stack

```
┌─────────────────────────────────────────────────────┐
│                  Application Code                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │ OTel     │  │ Structured│  │ OTel     │          │
│  │ Traces   │  │ Logs     │  │ Metrics  │          │
│  │ (SDK)    │  │ (structlog)│ │ (SDK)    │          │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘          │
│       │  trace_id    │  trace_id   │  exemplars     │
│       └──────────────┴─────────────┘                │
└───────────────────────┬─────────────────────────────┘
                        │ OTLP (all signals)
                        ▼
              ┌─────────────────┐
              │  OTel Collector  │
              │  ┌─────────────┐ │
              │  │ Connectors: │ │
              │  │ spanmetrics │ │  Generates metrics FROM traces
              │  │ servicegraph│ │  Generates service graph FROM traces
              │  └─────────────┘ │
              └───┬─────┬────┬──┘
                  │     │    │
         ┌────────┘     │    └────────┐
         ▼              ▼             ▼
   ┌──────────┐  ┌──────────┐  ┌──────────┐
   │  Tempo   │  │   Loki   │  │Prometheus│
   │ (traces) │  │  (logs)  │  │(metrics) │
   └────┬─────┘  └────┬─────┘  └────┬─────┘
        │              │             │
        └──────────────┴─────────────┘
                       │
                ┌──────▼──────┐
                │   Grafana    │
                │ (unified UI) │
                │              │
                │ Metrics ←→ Traces ←→ Logs │
                └──────────────┘
```

### 6.2 OTel Collector Connectors

Connectors generate new telemetry from existing telemetry, enabling cross-signal correlation:

```yaml
# OTel Collector config with connectors
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317

connectors:
  # Generate RED metrics from trace spans
  spanmetrics:
    histogram:
      explicit:
        buckets: [5ms, 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 2.5s, 5s, 10s]
    dimensions:
      - name: http.method
      - name: http.status_code
      - name: http.route
    exemplars:
      enabled: true
    metrics_flush_interval: 15s

  # Generate service dependency graph from traces
  servicegraph:
    latency_histogram_buckets: [5ms, 10ms, 50ms, 100ms, 500ms, 1s, 5s]
    dimensions:
      - http.method
    store:
      ttl: 2s
      max_items: 1000

exporters:
  otlp/tempo:
    endpoint: tempo:4317
  prometheus:
    endpoint: 0.0.0.0:8889
  loki:
    endpoint: http://loki:3100/loki/api/v1/push

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [otlp/tempo, spanmetrics, servicegraph]

    # spanmetrics connector outputs metrics
    metrics/spanmetrics:
      receivers: [spanmetrics]
      exporters: [prometheus]

    # servicegraph connector outputs metrics
    metrics/servicegraph:
      receivers: [servicegraph]
      exporters: [prometheus]

    logs:
      receivers: [otlp]
      processors: [batch]
      exporters: [loki]

    metrics:
      receivers: [otlp]
      processors: [batch]
      exporters: [prometheus]
```

### 6.3 Span Metrics: RED Metrics from Traces

The `spanmetrics` connector automatically generates request rate, error rate, and duration metrics from trace spans:

```promql
# Request rate (generated from traces)
sum(rate(traces_spanmetrics_calls_total{service_name="payment-service"}[5m]))

# Error rate (generated from traces)
sum(rate(traces_spanmetrics_calls_total{service_name="payment-service",status_code="STATUS_CODE_ERROR"}[5m]))
/ sum(rate(traces_spanmetrics_calls_total{service_name="payment-service"}[5m]))

# Latency p99 (generated from traces)
histogram_quantile(0.99,
  sum by (le) (rate(traces_spanmetrics_duration_milliseconds_bucket{service_name="payment-service"}[5m]))
)
```

This eliminates the need to manually instrument metrics for basic RED signals -- they are derived directly from traces.

---

## 7. Service Dependency Mapping

### 7.1 Service Graph from Traces

The `servicegraph` connector builds a live dependency map from trace data:

```
┌─────────┐     ┌──────────┐     ┌───────────┐
│ frontend │────→│ order-svc│────→│ payment-svc│
│  50 rps  │     │  45 rps  │     │   40 rps   │
│  10ms p50│     │  25ms p50│     │  100ms p50 │
└─────────┘     └──────┬───┘     └───────────┘
                       │
                       ▼
                ┌──────────┐
                │inventory │
                │  45 rps  │
                │  15ms p50│
                └──────────┘
```

### 7.2 Using Service Graphs for Impact Analysis

During an incident, the service graph answers:

1. **Which services are affected?** Follow the dependency arrows from the failing service.
2. **What is the blast radius?** Count downstream services and their request rates.
3. **Where is the bottleneck?** Find the service with the highest latency increase.

---

## 8. Correlation in Practice: Incident Walkthrough

### 8.1 Scenario Setup

```
14:00 UTC - Alert fires: "Payment service availability SLO burn rate critical (14.4x)"
```

### 8.2 Step-by-Step Investigation

```
Step 1: Open SLO dashboard
  → Payment availability SLI dropped from 99.97% to 98.5%
  → Error budget consumed: 85% → 30% in 15 minutes
  → Burn rate: 14.4x

Step 2: Check error rate panel (metrics)
  → Spike in 5xx responses starting at 13:58
  → Affected endpoint: POST /payments/charge
  → Unaffected: GET /payments/:id, POST /payments/refund

Step 3: Click exemplar on error rate spike (metrics → traces)
  → Trace ID: 7a8b9c0d1e2f
  → View in Tempo:
    api-gateway (5ms)
    → payment-service: POST /payments/charge (30,015ms) [ERROR]
      → stripe-client: POST /v1/charges (30,000ms) [TIMEOUT]

Step 4: Click "View logs" on stripe-client span (traces → logs)
  → Loki query: {service="payment-service"} |= "7a8b9c0d1e2f"
  → Logs:
    13:58:01 WARN  "Stripe API response time elevated: 5200ms (threshold: 3000ms)"
    13:58:15 ERROR "Stripe API timeout after 30s, attempt 1/3"
    13:58:45 ERROR "Stripe API timeout after 30s, attempt 2/3"
    13:59:15 ERROR "Stripe API timeout after 30s, attempt 3/3, giving up"
    13:59:15 ERROR "Circuit breaker tripped for stripe-payments"

Step 5: Check Stripe status page (external)
  → Stripe status: "Investigating increased latency on Charges API"

Step 6: Root cause confirmed
  → Stripe API degradation causing timeouts
  → Circuit breaker tripped after 3 retries
  → All /payments/charge requests failing

Step 7: Mitigation
  → Enable fallback payment processor (Adyen)
  → Set circuit breaker to route to fallback automatically
  → Monitor recovery

Total investigation time: 8 minutes (vs. estimated 45+ minutes without correlation)
```

---

## 9. Correlation Best Practices

### 9.1 Implementation Checklist

| Practice | Priority | Effort |
|----------|----------|--------|
| Add trace_id to all log entries | P0 | Low (one-time setup) |
| Configure Grafana trace-to-log linking | P0 | Low |
| Enable exemplars on key histograms | P1 | Medium |
| Deploy spanmetrics connector | P1 | Medium |
| Configure trace-to-metrics links | P2 | Low |
| Deploy servicegraph connector | P2 | Medium |
| Build correlated SLO dashboards | P1 | Medium |

### 9.2 Common Pitfalls

| Pitfall | Impact | Solution |
|---------|--------|----------|
| Missing trace_id in logs | Cannot navigate from logs to traces | Use OTel log bridge or structlog processor |
| Different trace ID formats | Correlation fails (hex vs decimal) | Standardize on 32-char lowercase hex |
| Log timestamps out of sync | Time-based correlation fails | Use NTP on all nodes, log in UTC |
| Exemplars not stored | Metric-to-trace links unavailable | Enable `--enable-feature=exemplar-storage` in Prometheus |
| Too many derived fields | Grafana UI cluttered | Only link trace_id and span_id |

---

## 10. Next Steps

- [22_Advanced_Metrics_Architecture.md](./22_Advanced_Metrics_Architecture.md) -- Scale Prometheus with federation, Thanos, and Mimir
- [23_OpenTelemetry_Pipelines.md](./23_OpenTelemetry_Pipelines.md) -- Design production-grade OTel Collector pipelines

---

## Exercises

### Exercise 1: Implement Trace-to-Log Linking

You have a Python Flask application using the `logging` module and OpenTelemetry for tracing. Currently, logs do not contain trace IDs. Write the code to:

1. Create a custom logging formatter that injects trace_id and span_id
2. Configure the root logger to use this formatter
3. Demonstrate that a log entry emitted within a traced request contains the correct trace_id

<details>
<summary>Show Answer</summary>

```python
import logging
import json
from datetime import datetime, timezone
from opentelemetry import trace

class TraceContextFormatter(logging.Formatter):
    """Custom formatter that injects OTel trace context into log records."""

    def format(self, record: logging.LogRecord) -> str:
        # Get current span context
        span = trace.get_current_span()
        ctx = span.get_span_context()

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Inject trace context if available
        if ctx.is_valid:
            log_entry["trace_id"] = format(ctx.trace_id, "032x")
            log_entry["span_id"] = format(ctx.span_id, "016x")
            log_entry["trace_flags"] = ctx.trace_flags
        else:
            log_entry["trace_id"] = "0" * 32
            log_entry["span_id"] = "0" * 16

        # Add any extra fields from the log record
        if hasattr(record, "extra_data"):
            log_entry.update(record.extra_data)

        return json.dumps(log_entry)

# Configure root logger
handler = logging.StreamHandler()
handler.setFormatter(TraceContextFormatter())
logging.root.handlers = [handler]
logging.root.setLevel(logging.INFO)

logger = logging.getLogger("payment-service")

# Demonstration
from opentelemetry.sdk.trace import TracerProvider
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer("payment-service")

with tracer.start_as_current_span("process_payment") as span:
    trace_id = format(span.get_span_context().trace_id, "032x")
    logger.info("Payment processing started")
    # Output: {"timestamp": "...", "level": "INFO", "message": "Payment processing started",
    #          "trace_id": "<32-char hex>", "span_id": "<16-char hex>", ...}

# Outside of span
logger.info("No trace context here")
# Output: {"timestamp": "...", "level": "INFO", "message": "No trace context here",
#          "trace_id": "00000000000000000000000000000000", ...}
```

</details>

### Exercise 2: Exemplar Analysis

Given the following Prometheus metric with exemplars:

```
http_request_duration_seconds_bucket{le="0.1",service="api"} 9500 # {trace_id="aaa"} 0.08
http_request_duration_seconds_bucket{le="0.5",service="api"} 9900 # {trace_id="bbb"} 0.35
http_request_duration_seconds_bucket{le="1.0",service="api"} 9980 # {trace_id="ccc"} 0.72
http_request_duration_seconds_bucket{le="5.0",service="api"} 9998 # {trace_id="ddd"} 3.10
http_request_duration_seconds_bucket{le="+Inf",service="api"} 10000 # {trace_id="eee"} 12.5
http_request_duration_seconds_count{service="api"} 10000
http_request_duration_seconds_sum{service="api"} 5200
```

Answer: (a) How many requests took between 0.5s and 1.0s? (b) What is the p95 latency (approximate)? (c) Which trace_id should you investigate first to understand tail latency? (d) What percentage of requests completed within 100ms?

<details>
<summary>Show Answer</summary>

**(a) Requests between 0.5s and 1.0s:**
```
bucket[le="1.0"] - bucket[le="0.5"] = 9980 - 9900 = 80 requests
```

**(b) Approximate p95 latency:**
```
p95 means the value at the 95th percentile = 0.95 * 10000 = 9500th request.
bucket[le="0.1"] = 9500, so the 9500th request falls exactly at the 0.1s boundary.
p95 ≈ 0.1s (100ms).

Note: This is approximate. Histogram interpolation would place it at:
p95 = 0.1 * (9500 - 0) / (9500 - 0) = 0.1s
```

**(c) Which trace_id to investigate for tail latency:**
```
trace_id="eee" with value 12.5s -- this is the most extreme outlier.
It is in the +Inf bucket (above 5s), which is the longest-running request.
Only 2 requests (10000 - 9998) took longer than 5s.

However, trace_id="ddd" at 3.1s is also worth investigating as it represents
the 99.98th percentile range. Both traces should be reviewed.
```

**(d) Percentage of requests within 100ms:**
```
bucket[le="0.1"] / count = 9500 / 10000 = 95%
```

</details>

### Exercise 3: Unified Telemetry Design

Design a complete unified telemetry architecture for a 3-service system (API Gateway, Order Service, Inventory Service). Specify:

1. What OTel instrumentation each service needs
2. The OTel Collector pipeline configuration (receivers, processors, connectors, exporters)
3. The Grafana data source configuration for cross-signal correlation
4. A sample debugging workflow showing how you would use all three signals together

<details>
<summary>Show Answer</summary>

**1. Instrumentation per service:**

| Service | Auto-instrumentation | Manual Spans | Custom Metrics | Structured Logs |
|---------|---------------------|-------------|----------------|-----------------|
| API Gateway | HTTP (Flask/FastAPI), Redis | `authenticate_request`, `rate_limit_check` | `gateway_requests_total`, `gateway_auth_failures_total` | All logs with trace_id, user_id |
| Order Service | HTTP, PostgreSQL, Kafka producer | `create_order`, `validate_inventory`, `calculate_total` | `orders_created_total`, `order_value_dollars` | All logs with trace_id, order_id |
| Inventory Service | HTTP, PostgreSQL, Kafka consumer | `reserve_items`, `check_stock`, `update_inventory` | `inventory_reservations_total`, `stock_level` (gauge) | All logs with trace_id, item_id |

**2. OTel Collector config:**

```yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317

processors:
  batch:
    timeout: 5s
    send_batch_size: 1024
  memory_limiter:
    check_interval: 1s
    limit_mib: 512
  attributes:
    actions:
      - key: environment
        value: production
        action: upsert

connectors:
  spanmetrics:
    histogram:
      explicit:
        buckets: [5ms, 10ms, 50ms, 100ms, 500ms, 1s, 5s]
    dimensions:
      - name: http.method
      - name: http.route
      - name: http.status_code
    exemplars:
      enabled: true
  servicegraph:
    latency_histogram_buckets: [10ms, 50ms, 100ms, 500ms, 1s, 5s]

exporters:
  otlp/tempo:
    endpoint: tempo:4317
  prometheus:
    endpoint: 0.0.0.0:8889
  loki:
    endpoint: http://loki:3100/loki/api/v1/push

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [memory_limiter, attributes, batch]
      exporters: [otlp/tempo, spanmetrics, servicegraph]
    metrics:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [prometheus]
    metrics/derived:
      receivers: [spanmetrics, servicegraph]
      exporters: [prometheus]
    logs:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [loki]
```

**3. Grafana configuration:**
- Tempo data source: tracesToLogs → Loki (filter by trace_id), tracesToMetrics → Prometheus
- Loki data source: derivedFields → trace_id links to Tempo
- Prometheus data source: exemplars enabled, linked to Tempo

**4. Debugging workflow:**

```
Alert: "Order Service availability SLO burn rate 6x"

Step 1: SLO Dashboard (Prometheus metrics)
  → Order service error rate spiked to 2% at 10:15
  → Affected endpoint: POST /orders

Step 2: Click exemplar on error spike (Prometheus → Tempo)
  → Trace abc123: api-gateway → order-service → inventory-service [ERROR]
  → inventory-service span: "reserve_items" failed after 5s timeout

Step 3: View inventory-service span details (Tempo)
  → Attribute: db.statement = "UPDATE inventory SET reserved = reserved + $1 WHERE item_id = $2"
  → Attribute: db.duration_ms = 4980
  → Status: DEADLINE_EXCEEDED

Step 4: View logs for trace abc123 (Tempo → Loki)
  → inventory-service: "Lock wait timeout on inventory table, item_id=PROD-001"
  → inventory-service: "Concurrent reservation conflict, 15 waiters in queue"

Step 5: Check inventory-service metrics (Loki → Prometheus)
  → db_pool_active_connections: 50/50 (pool exhausted)
  → inventory_lock_wait_seconds: p99 = 4.8s (normally 10ms)

Root cause: Flash sale on PROD-001 caused lock contention on inventory table.
Fix: Use optimistic locking or queue-based reservation for high-demand items.
```

</details>

---

## References

- [Grafana Tempo -- Traces to Logs](https://grafana.com/docs/tempo/latest/metrics-generator/traces-to-logs/)
- [OpenTelemetry Collector Connectors](https://opentelemetry.io/docs/collector/connectors/)
- [Prometheus Exemplars](https://prometheus.io/docs/prometheus/latest/feature_flags/#exemplars-storage)
- [Grafana Unified Alerting](https://grafana.com/docs/grafana/latest/alerting/)
- [Correlating Signals in Grafana Cloud (Blog)](https://grafana.com/blog/2022/08/18/how-to-correlate-metrics-logs-and-traces-in-grafana/)
