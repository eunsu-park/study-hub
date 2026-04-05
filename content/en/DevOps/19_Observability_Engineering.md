# 19. Observability Engineering

**Previous**: [SRE Practices](./18_SRE_Practices.md) | **Next**: [SLO Engineering](./20_SLO_Engineering.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Distinguish between monitoring and observability and explain why observability matters for modern distributed systems
2. Describe the three pillars of observability (metrics, logs, traces) and their complementary roles
3. Explain the OpenTelemetry project architecture and its role as the industry-standard telemetry framework
4. Design an instrumentation strategy that balances signal coverage with performance overhead
5. Apply the concept of cardinality to make informed decisions about telemetry data dimensions
6. Implement structured telemetry that supports debugging novel failures in production

---

Monitoring tells you *when* something is broken. Observability tells you *why*. In a monolithic world, monitoring was sufficient -- you had a handful of servers and a predictable set of failure modes. In a distributed microservices world, failures emerge from complex interactions between dozens of services, and you cannot predict every failure mode in advance. Observability is the property of a system that allows you to understand its internal state by examining its external outputs -- telemetry data.

> **Analogy -- Medical Diagnostics**: Monitoring is like a hospital alarm that beeps when a patient's heart rate exceeds 120 BPM. It tells you *something is wrong*. Observability is the full diagnostic suite -- ECG traces, blood panels, imaging -- that lets a doctor ask arbitrary questions about the patient's condition and diagnose problems they have never seen before. You cannot pre-configure an alarm for every possible disease, but you can collect enough diagnostic data to investigate any symptom.

## 1. Monitoring vs Observability

### 1.1 The Shift from Monitoring to Observability

| Aspect | Monitoring | Observability |
|--------|-----------|--------------|
| **Question** | "Is the system working?" | "Why is the system behaving this way?" |
| **Approach** | Pre-defined checks and thresholds | Explore arbitrary questions from telemetry |
| **Failure model** | Known failure modes (predicted) | Unknown-unknowns (emergent) |
| **Data model** | Aggregated metrics, static dashboards | High-cardinality, high-dimensionality data |
| **Debugging** | Check the dashboard → find the alert → follow the runbook | Explore telemetry → form hypotheses → drill down |
| **Scale** | Works for monoliths and simple architectures | Essential for distributed systems |

### 1.2 Why Monitoring Alone Is Insufficient

In distributed systems, failures often emerge from interactions:

- Service A is slow because Service B's connection pool is exhausted
- Service B's pool is exhausted because Service C deployed a schema change
- Service C's schema change is only slow for queries involving tenant X's data

No dashboard can pre-visualize this chain. You need the ability to:
1. Start from a symptom (high latency on Service A)
2. Drill into specific traces showing slow requests
3. Correlate with logs from Service B showing pool exhaustion
4. Jump to Service C metrics showing increased query times for specific tenants

### 1.3 The Observability Maturity Model

| Level | Capability | Characteristics |
|-------|-----------|----------------|
| **L0: Reactive** | Basic health checks | Alerts on up/down, no structured telemetry |
| **L1: Informed** | Metrics + dashboards | Golden signals monitored, static dashboards |
| **L2: Investigative** | Correlated signals | Metrics, logs, and traces linked together |
| **L3: Proactive** | Anomaly detection | ML-based alerting, SLO-driven decisions |
| **L4: Predictive** | Capacity forecasting | Trend analysis, automated scaling recommendations |

---

## 2. The Three Pillars of Observability

### 2.1 Metrics

Metrics are numerical measurements collected at regular intervals. They are the most efficient telemetry type -- highly compressible and fast to query.

**Strengths:**
- Low storage cost (fixed per time series regardless of traffic)
- Fast aggregation across dimensions
- Ideal for alerting (thresholds on aggregated values)
- Well-suited for dashboards and trend analysis

**Weaknesses:**
- Aggregation loses individual event detail
- Cannot answer "which specific request was slow?"
- Cardinality explosion when adding too many label dimensions

**Key metric types (Prometheus model):**

| Type | Behavior | Example |
|------|----------|---------|
| **Counter** | Monotonically increasing | `http_requests_total` |
| **Gauge** | Goes up and down | `temperature_celsius` |
| **Histogram** | Counts values in buckets | `request_duration_seconds_bucket` |
| **Summary** | Client-side quantile calculation | `request_duration_seconds{quantile="0.99"}` |

### 2.2 Logs

Logs are discrete, timestamped event records. They provide the richest context but are the most expensive to store and query.

**Strengths:**
- Rich context per event (request ID, user ID, stack trace)
- Human-readable debugging information
- Can record arbitrary data structures

**Weaknesses:**
- Expensive storage at high volume (bytes per event vs. bytes per time series)
- Slow to query without indexing
- Difficult to aggregate numerically
- Unstructured logs are nearly useless at scale

**Structured logging best practices:**

```python
import structlog
import uuid

logger = structlog.get_logger()

def process_order(order_id: str, user_id: str):
    """Example of structured logging with correlation context."""
    log = logger.bind(
        order_id=order_id,
        user_id=user_id,
        trace_id=get_current_trace_id(),
        span_id=get_current_span_id(),
    )

    log.info("order.processing_started", item_count=len(order.items))

    try:
        result = payment_service.charge(order.total)
        log.info("order.payment_completed",
                 amount=order.total,
                 payment_id=result.id,
                 duration_ms=result.duration_ms)
    except PaymentError as e:
        log.error("order.payment_failed",
                  error_type=type(e).__name__,
                  error_message=str(e),
                  amount=order.total)
        raise
```

### 2.3 Traces

Traces record the path of a request through a distributed system. A trace consists of spans -- each span represents a unit of work (an HTTP call, a database query, a message publish).

**Strengths:**
- Show the full request lifecycle across services
- Reveal latency bottlenecks and dependency chains
- Enable root cause analysis for distributed failures

**Weaknesses:**
- High storage cost (one trace per request can mean millions per hour)
- Sampling is required at scale (not every request is traced)
- Complex instrumentation (every service must propagate context)

**Trace anatomy:**

```
Trace ID: abc123
├── [Span 1] api-gateway: POST /orders (250ms)
│   ├── [Span 2] order-service: createOrder (200ms)
│   │   ├── [Span 3] postgres: INSERT orders (15ms)
│   │   ├── [Span 4] payment-service: charge (150ms)
│   │   │   ├── [Span 5] stripe-client: POST /charges (120ms)
│   │   │   └── [Span 6] postgres: INSERT payments (8ms)
│   │   └── [Span 7] notification-service: sendEmail (30ms)
│   │       └── [Span 8] ses-client: SendEmail (25ms)
```

### 2.4 Comparing the Three Pillars

| Dimension | Metrics | Logs | Traces |
|-----------|---------|------|--------|
| **Data type** | Numeric time series | Structured event records | Span trees |
| **Granularity** | Aggregated (per interval) | Per event | Per request |
| **Storage cost** | Low | High | Medium-High |
| **Query speed** | Fast | Slow (without index) | Medium |
| **Best for** | Alerting, dashboards | Debugging, auditing | Request flow analysis |
| **Cardinality concern** | Label explosion | Volume explosion | Sampling trade-offs |

---

## 3. OpenTelemetry Overview

### 3.1 What Is OpenTelemetry

OpenTelemetry (OTel) is a CNCF project that provides a vendor-neutral, open-standard framework for generating, collecting, and exporting telemetry data. It merges the former OpenTracing and OpenCensus projects.

**Key components:**

```
Application Code
    │
    ├── OTel API (vendor-neutral interfaces)
    │   ├── TracerProvider → Tracer → Span
    │   ├── MeterProvider → Meter → Counter/Histogram/Gauge
    │   └── LoggerProvider → Logger → LogRecord
    │
    ├── OTel SDK (reference implementation)
    │   ├── SpanProcessor (batch, simple)
    │   ├── SpanExporter (OTLP, Jaeger, Zipkin)
    │   ├── MetricReader (periodic, manual)
    │   └── MetricExporter (OTLP, Prometheus)
    │
    └── OTel Collector (standalone agent/gateway)
        ├── Receivers (OTLP, Jaeger, Prometheus, etc.)
        ├── Processors (batch, filter, attributes, tail_sampling)
        └── Exporters (OTLP, Jaeger, Prometheus, Loki, etc.)
```

### 3.2 OTLP (OpenTelemetry Protocol)

OTLP is the native protocol for transmitting telemetry data:

| Feature | OTLP/gRPC | OTLP/HTTP |
|---------|-----------|-----------|
| **Transport** | gRPC (HTTP/2) | HTTP/1.1 or HTTP/2 |
| **Encoding** | Protobuf | Protobuf or JSON |
| **Performance** | Higher throughput, streaming | Simpler, firewall-friendly |
| **Port** | 4317 | 4318 |
| **Use case** | Service-to-collector | Browser, edge, restricted networks |

### 3.3 Language Support

```python
# Python: OpenTelemetry instrumentation example
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

# --- Tracing setup ---
trace_provider = TracerProvider()
trace_provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint="otel-collector:4317"))
)
trace.set_tracer_provider(trace_provider)

# --- Metrics setup ---
metric_reader = PeriodicExportingMetricReader(
    OTLPMetricExporter(endpoint="otel-collector:4317"),
    export_interval_millis=60000,
)
metrics.set_meter_provider(MeterProvider(metric_readers=[metric_reader]))

# --- Usage ---
tracer = trace.get_tracer("order-service")
meter = metrics.get_meter("order-service")

order_counter = meter.create_counter(
    "orders.created",
    description="Total orders created",
    unit="1",
)

order_duration = meter.create_histogram(
    "orders.processing_duration",
    description="Order processing duration",
    unit="ms",
)

def create_order(order_data: dict) -> Order:
    with tracer.start_as_current_span("create_order") as span:
        span.set_attribute("order.customer_id", order_data["customer_id"])
        span.set_attribute("order.item_count", len(order_data["items"]))

        start = time.monotonic()
        order = Order.from_dict(order_data)
        order.save()

        duration_ms = (time.monotonic() - start) * 1000
        order_counter.add(1, {"order.type": order.type})
        order_duration.record(duration_ms, {"order.type": order.type})

        span.set_attribute("order.id", order.id)
        return order
```

### 3.4 Auto-Instrumentation

OpenTelemetry provides automatic instrumentation for common libraries, reducing manual effort:

```bash
# Python: install auto-instrumentation packages
pip install opentelemetry-distro opentelemetry-exporter-otlp
opentelemetry-bootstrap -a install

# Run with auto-instrumentation
opentelemetry-instrument \
    --service_name order-service \
    --exporter_otlp_endpoint otel-collector:4317 \
    python app.py
```

Auto-instrumented libraries (Python):
- `requests`, `urllib3`, `httpx` -- outgoing HTTP calls
- `flask`, `django`, `fastapi` -- incoming HTTP requests
- `psycopg2`, `sqlalchemy`, `pymongo` -- database calls
- `celery`, `redis`, `kafka-python` -- message queues
- `grpc` -- gRPC client and server calls

---

## 4. Instrumentation Strategies

### 4.1 The Instrumentation Pyramid

```
                    /\
                   /  \
                  / Biz \          Business metrics (conversion rate, revenue)
                 / Logic  \
                /----------\
               / Application \     Request latency, error rates, throughput
              /   Metrics     \
             /----------------\
            / Infrastructure    \   CPU, memory, disk, network
           /   Metrics           \
          /----------------------\
         / Auto-Instrumented       \  HTTP, DB, queue spans (from OTel auto)
        /   Telemetry               \
       /----------------------------\
```

### 4.2 Manual vs Automatic Instrumentation

| Approach | Coverage | Effort | Quality |
|----------|----------|--------|---------|
| **Auto-instrumentation** | Library boundaries (HTTP, DB, queues) | Zero code changes | Generic span names, basic attributes |
| **Manual spans** | Business logic, custom operations | Code changes required | Rich context, domain-specific attributes |
| **Semantic conventions** | Standardized attribute names | Follow OTel spec | Consistent across services |

**Best practice**: Start with auto-instrumentation for breadth, then add manual spans for depth in critical business paths.

### 4.3 What to Instrument

**Always instrument (high value, low effort):**
- HTTP request/response (auto-instrumented)
- Database queries (auto-instrumented)
- External API calls (auto-instrumented)
- Message queue publish/consume (auto-instrumented)
- Authentication and authorization decisions
- Business-critical operations (order creation, payment processing)

**Instrument selectively (medium value, medium effort):**
- Cache hits/misses
- Feature flag evaluations
- Retry attempts
- Circuit breaker state changes
- Background job execution

**Avoid instrumenting (low value, high cost):**
- Tight inner loops (per-item processing in a batch)
- Trivial getters/setters
- Every line of code (use profiling instead)

### 4.4 Span Design Best Practices

```python
# BAD: Too granular -- creates noise
with tracer.start_as_current_span("validate_email"):
    validate_email(user.email)
with tracer.start_as_current_span("validate_name"):
    validate_name(user.name)
with tracer.start_as_current_span("validate_address"):
    validate_address(user.address)

# GOOD: Meaningful unit of work with attributes
with tracer.start_as_current_span("validate_user_input") as span:
    errors = validate_user(user)
    span.set_attribute("validation.error_count", len(errors))
    span.set_attribute("validation.fields_checked", ["email", "name", "address"])
    if errors:
        span.set_status(StatusCode.ERROR, f"{len(errors)} validation errors")
        span.record_exception(ValidationError(errors))
```

---

## 5. Telemetry Data Design

### 5.1 Cardinality and Dimensionality

**Cardinality** is the number of unique time series created by a metric. High cardinality is the most common cause of observability system failures.

```
# LOW cardinality (safe): ~50 time series
http_requests_total{method="GET|POST|PUT|DELETE", status="2xx|4xx|5xx"}
# ~4 methods × ~3 status classes × ~4 services = ~48

# MEDIUM cardinality (manageable): ~5,000 time series
http_requests_total{method, status, endpoint, service}
# ~4 × 5 × 50 × 5 = 5,000

# HIGH cardinality (dangerous): ~10,000,000 time series
http_requests_total{method, status, endpoint, service, user_id}
# ~4 × 5 × 50 × 5 × 10,000 = 50,000,000 -- DO NOT DO THIS
```

**Rules for cardinality management:**

| Rule | Description |
|------|-------------|
| **Never use unbounded values as metric labels** | user_id, request_id, email -- use traces for these |
| **Bucket continuous values** | Response time → histogram buckets, not exact values |
| **Use recording rules** | Pre-aggregate high-cardinality queries |
| **Monitor cardinality** | Track `prometheus_tsdb_head_series` |
| **Set cardinality limits** | Use `sample_limit` in Prometheus scrape config |

### 5.2 Semantic Conventions

OpenTelemetry defines semantic conventions -- standardized attribute names for common concepts:

```python
# OTel Semantic Conventions (subset)
from opentelemetry.semconv.trace import SpanAttributes
from opentelemetry.semconv.resource import ResourceAttributes

# HTTP
span.set_attribute(SpanAttributes.HTTP_METHOD, "POST")
span.set_attribute(SpanAttributes.HTTP_URL, "https://api.example.com/orders")
span.set_attribute(SpanAttributes.HTTP_STATUS_CODE, 201)

# Database
span.set_attribute(SpanAttributes.DB_SYSTEM, "postgresql")
span.set_attribute(SpanAttributes.DB_STATEMENT, "SELECT * FROM orders WHERE id = $1")
span.set_attribute(SpanAttributes.DB_NAME, "orders_db")

# Resource (service identity)
resource = Resource.create({
    ResourceAttributes.SERVICE_NAME: "order-service",
    ResourceAttributes.SERVICE_VERSION: "1.4.2",
    ResourceAttributes.DEPLOYMENT_ENVIRONMENT: "production",
    ResourceAttributes.HOST_NAME: socket.gethostname(),
    ResourceAttributes.K8S_POD_NAME: os.environ.get("POD_NAME"),
    ResourceAttributes.K8S_NAMESPACE_NAME: os.environ.get("NAMESPACE"),
})
```

### 5.3 Context Propagation

Context propagation ensures that trace IDs and span IDs are passed across service boundaries:

```
Service A                         Service B
┌────────────────────┐            ┌────────────────────┐
│ Span: handle_request│ ─HTTP──→  │ Span: process_order│
│ trace_id: abc123   │  Headers:  │ trace_id: abc123   │
│ span_id: span_001  │  traceparent: │ parent: span_001│
└────────────────────┘  00-abc123-span_001-01          │
                                  └────────────────────┘
```

**W3C Trace Context header format:**

```
traceparent: 00-<trace-id>-<parent-span-id>-<trace-flags>
             │   │           │                │
             │   │           │                └── 01 = sampled
             │   │           └── 16 hex chars (8 bytes)
             │   └── 32 hex chars (16 bytes)
             └── version (always 00)

Example:
traceparent: 00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01
```

---

## 6. Observability Architecture Patterns

### 6.1 Agent-Based Collection

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  App Pod 1   │     │  App Pod 2   │     │  App Pod 3   │
│  + OTel SDK  │     │  + OTel SDK  │     │  + OTel SDK  │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       └──────────┬─────────┴────────────────────┘
                  │ OTLP
       ┌──────────▼───────────┐
       │  OTel Collector      │  (DaemonSet -- one per node)
       │  Agent Mode          │
       │  - Batch processor   │
       │  - Memory limiter    │
       └──────────┬───────────┘
                  │ OTLP
       ┌──────────▼───────────┐
       │  OTel Collector      │  (Deployment -- centralized)
       │  Gateway Mode        │
       │  - Tail sampling     │
       │  - Routing           │
       └──────┬───────┬───────┘
              │       │
     ┌────────▼┐  ┌───▼────────┐
     │ Jaeger  │  │ Prometheus │
     │ (traces)│  │ (metrics)  │
     └─────────┘  └────────────┘
```

### 6.2 Sidecar Pattern

Each application pod gets its own OTel Collector sidecar. This provides isolation but increases resource usage:

```yaml
# Kubernetes pod spec with OTel Collector sidecar
apiVersion: v1
kind: Pod
metadata:
  name: order-service
spec:
  containers:
    - name: app
      image: order-service:1.4.2
      env:
        - name: OTEL_EXPORTER_OTLP_ENDPOINT
          value: "http://localhost:4317"
      ports:
        - containerPort: 8080

    - name: otel-collector
      image: otel/opentelemetry-collector-contrib:0.96.0
      args: ["--config=/etc/otel/config.yaml"]
      volumeMounts:
        - name: otel-config
          mountPath: /etc/otel
      resources:
        limits:
          cpu: 200m
          memory: 256Mi
        requests:
          cpu: 50m
          memory: 64Mi

  volumes:
    - name: otel-config
      configMap:
        name: otel-sidecar-config
```

### 6.3 Choosing a Collection Pattern

| Pattern | Resource Overhead | Isolation | Complexity | Best For |
|---------|------------------|-----------|------------|----------|
| **Direct export** (app → backend) | Lowest | None | Lowest | Small deployments |
| **Agent** (DaemonSet) | Medium | Per-node | Medium | Most Kubernetes deployments |
| **Sidecar** | Highest | Per-pod | Highest | Multi-tenant, strict isolation |
| **Gateway** (centralized collector) | Low (centralized) | Shared | Medium | Large-scale with tail sampling |

---

## 7. Observability Culture

### 7.1 Shifting Left on Observability

Observability should not be an afterthought bolted on in production. It must be part of the development process:

```
Development Lifecycle with Observability
─────────────────────────────────────────
Design Phase:
  → Define SLIs and SLOs for the service
  → Identify critical paths that need instrumentation
  → Choose telemetry data model (what attributes, what metrics)

Implementation Phase:
  → Add OTel SDK and auto-instrumentation
  → Add manual spans for business-critical paths
  → Add structured logging with correlation IDs
  → Write metric recording for business KPIs

Testing Phase:
  → Verify spans appear in local Jaeger
  → Load test and verify metric accuracy
  → Test alert rules against synthetic failures

Deployment Phase:
  → Deploy with feature flags to control sampling
  → Verify telemetry in staging before production
  → Create dashboards and alert rules

Operations Phase:
  → Use telemetry to debug incidents
  → Refine instrumentation based on actual debugging needs
  → Review cardinality and optimize storage
```

### 7.2 Observability-Driven Development

Write instrumentation *before* the feature code:

```python
def process_payment(payment_request: PaymentRequest) -> PaymentResult:
    """Observability-driven: instrumentation written first."""
    with tracer.start_as_current_span("process_payment") as span:
        # 1. Record input dimensions (written before business logic)
        span.set_attribute("payment.method", payment_request.method)
        span.set_attribute("payment.currency", payment_request.currency)
        span.set_attribute("payment.amount_cents", payment_request.amount_cents)

        # 2. Business logic (written after instrumentation skeleton)
        result = _execute_payment(payment_request)

        # 3. Record outcome dimensions
        span.set_attribute("payment.status", result.status)
        span.set_attribute("payment.processor_latency_ms", result.processor_latency_ms)

        # 4. Record metrics
        payment_counter.add(1, {
            "method": payment_request.method,
            "status": result.status,
        })
        payment_latency.record(result.processor_latency_ms, {
            "method": payment_request.method,
        })

        return result
```

---

## 8. Cost Management

### 8.1 The Telemetry Cost Equation

```
Monthly Cost = (Ingestion Volume × Ingestion Price)
             + (Storage Volume × Storage Price)
             + (Query Volume × Query Price)
```

| Signal | Typical Volume | Cost Driver |
|--------|---------------|-------------|
| **Metrics** | ~10K time series per service | Cardinality (unique time series) |
| **Logs** | ~1-10 GB/day per service | Volume (bytes ingested) |
| **Traces** | ~100K-1M spans/day per service | Span count × attributes size |

### 8.2 Cost Optimization Strategies

| Strategy | Signal | Impact |
|----------|--------|--------|
| **Sampling** (head or tail) | Traces | 10-100x cost reduction |
| **Log level management** | Logs | 2-10x reduction (drop DEBUG in prod) |
| **Metric aggregation** | Metrics | Reduce high-cardinality series |
| **Recording rules** | Metrics | Pre-aggregate instead of query-time aggregation |
| **Retention policies** | All | Keep detailed data for 7 days, aggregated for 90 days |
| **Data tiering** | All | Hot (SSD) → Warm (HDD) → Cold (object storage) |
| **Attribute filtering** | Traces/Logs | Drop non-essential attributes at collector |

### 8.3 Sampling Strategies

```yaml
# OTel Collector: tail-based sampling
processors:
  tail_sampling:
    decision_wait: 10s
    policies:
      # Always keep traces with errors
      - name: errors-policy
        type: status_code
        status_code: {status_codes: [ERROR]}

      # Always keep slow traces (> 2 seconds)
      - name: latency-policy
        type: latency
        latency: {threshold_ms: 2000}

      # Sample 10% of successful traces
      - name: probabilistic-policy
        type: probabilistic
        probabilistic: {sampling_percentage: 10}

      # Always keep traces from critical services
      - name: critical-services
        type: string_attribute
        string_attribute:
          key: service.name
          values: [payment-service, auth-service]
```

---

## 9. Observability Anti-Patterns

### 9.1 Common Mistakes

| Anti-Pattern | Problem | Solution |
|-------------|---------|----------|
| **Dashboard-driven development** | Build dashboards for everything, debug nothing | Focus on exploratory tooling (trace search, log correlation) |
| **Alert fatigue** | Too many alerts, most ignored | SLO-based alerting with error budgets (Lesson 20) |
| **High-cardinality labels** | user_id as a Prometheus label | Use traces for high-cardinality data |
| **Unstructured logs** | `log.info(f"Order {order_id} processed")` | Use structured logging with key-value pairs |
| **Missing correlation** | Cannot link a metric spike to specific traces | Exemplars, trace-to-log linking (Lesson 21) |
| **Over-sampling** | 100% trace sampling in production | Use tail sampling to keep interesting traces |
| **Vendor lock-in** | Proprietary agents and protocols | Use OpenTelemetry for vendor-neutral telemetry |
| **Neglecting context propagation** | Traces break at service boundaries | Ensure W3C Trace Context headers are propagated |

### 9.2 The "Christmas Tree" Dashboard

A common failure mode is building dashboards covered in green/red status indicators:

```
BAD: Christmas Tree Dashboard
┌─────────────────────────────────────┐
│ ● Service A: UP  ● Service D: UP   │
│ ● Service B: UP  ● Service E: UP   │
│ ● Service C: DOWN ● Service F: UP  │
│ ● Database: UP   ● Cache: UP       │
└─────────────────────────────────────┘
Problem: "Service C is DOWN" -- but WHY? What failed? What's affected?

GOOD: SLO-Based Dashboard
┌─────────────────────────────────────┐
│ Availability SLO: 99.9%            │
│ Current: 99.85% ▼ (budget: 43min)  │
│ Burn rate: 2.1x (alert threshold)  │
│                                     │
│ Top error contributors:            │
│  1. /api/payments POST: 0.8% err   │
│  2. /api/orders GET: 0.3% err      │
│                                     │
│ [View traces for payment errors →]  │
│ [View error budget history →]       │
└─────────────────────────────────────┘
```

---

## 10. Next Steps

- [20_SLO_Engineering.md](./20_SLO_Engineering.md) -- Define and operationalize SLOs with error budgets
- [21_Signal_Correlation.md](./21_Signal_Correlation.md) -- Correlate metrics, logs, and traces for faster debugging

---

## Exercises

### Exercise 1: Monitoring vs Observability Assessment

Your team operates a microservices e-commerce platform with 15 services. Currently, you have Prometheus metrics and Grafana dashboards. A recent incident took 4 hours to diagnose because the team could not determine which service caused a cascade of 500 errors.

Evaluate your current observability maturity level and propose a roadmap to reach Level 3 (Proactive). Include specific tools, instrumentation changes, and cultural shifts.

<details>
<summary>Show Answer</summary>

**Current state: Level 1 (Informed)**
- Has metrics and dashboards (Prometheus + Grafana)
- Missing distributed tracing (cannot follow requests across services)
- Missing structured logging correlation (cannot link logs to traces)
- No SLO-based alerting

**Roadmap to Level 3:**

**Phase 1 (Month 1-2): Add Distributed Tracing → Level 2**
- Deploy OpenTelemetry Collector as a DaemonSet
- Add OTel auto-instrumentation to all 15 services
- Deploy Jaeger or Tempo for trace storage
- Configure W3C Trace Context propagation
- Result: Can now follow a request across all services

**Phase 2 (Month 2-3): Correlate Signals → Level 2 (mature)**
- Add trace_id to all structured log entries
- Configure Grafana to link from metrics → traces → logs
- Add exemplars to Prometheus metrics
- Create runbooks that reference telemetry exploration steps

**Phase 3 (Month 3-6): SLO-Based Operations → Level 3**
- Define SLOs for each service (availability, latency)
- Implement error budget tracking
- Replace threshold alerts with burn-rate alerts
- Deploy anomaly detection on key SLIs
- Train the team on exploratory debugging (not just dashboard checking)

**Cultural shifts:**
- Engineers write instrumentation as part of feature development (not after)
- Postmortems include "what telemetry was missing" as a standard section
- On-call responders use trace exploration as the first debugging step

</details>

### Exercise 2: Instrumentation Design

Design the instrumentation for a user registration endpoint that:
1. Accepts a POST request with user details
2. Validates the input
3. Checks for duplicate email in the database
4. Hashes the password (bcrypt)
5. Creates the user record
6. Sends a welcome email via an external service
7. Returns the user ID

Define: what spans to create, what attributes to set on each span, what metrics to record, and what log events to emit. Justify your choices.

<details>
<summary>Show Answer</summary>

**Spans:**

| Span | Parent | Attributes | Justification |
|------|--------|-----------|---------------|
| `register_user` | Root | `user.email_domain`, `user.source` (web/mobile/api) | Top-level span for the entire operation |
| `validate_input` | `register_user` | `validation.error_count`, `validation.fields` | Separate span because validation can have business-relevant failures |
| `check_duplicate_email` | `register_user` | `db.system=postgresql`, `db.operation=SELECT`, `user.duplicate_found` | Auto-instrumented DB span + custom attribute |
| `hash_password` | `register_user` | `bcrypt.cost_factor=12` | Bcrypt is CPU-intensive; important to track its contribution to latency |
| `create_user_record` | `register_user` | `db.system=postgresql`, `db.operation=INSERT`, `user.id` (after creation) | Auto-instrumented |
| `send_welcome_email` | `register_user` | `email.provider=ses`, `email.template=welcome` | External dependency; critical to track separately |

**Metrics:**

| Metric | Type | Labels | Justification |
|--------|------|--------|---------------|
| `user.registrations_total` | Counter | `status` (success/duplicate/validation_error/internal_error), `source` | Business KPI |
| `user.registration_duration_seconds` | Histogram | `source` | Performance tracking |
| `user.password_hash_duration_seconds` | Histogram | (none) | CPU-bound operation tracking |

**Log events:**
- `user.registration.started` (INFO): source, email_domain
- `user.validation.failed` (WARN): error details (no PII)
- `user.duplicate_email` (WARN): email_domain (not full email)
- `user.created` (INFO): user_id, source
- `user.welcome_email.sent` (INFO): user_id
- `user.welcome_email.failed` (ERROR): user_id, error type (email send is non-critical)

**Key decisions:**
- Never log the full email or password (PII/security)
- `email_domain` as an attribute helps detect registration spam patterns
- `send_welcome_email` is a separate span because it is an external dependency that may fail independently
- Password hashing gets its own span because bcrypt at cost 12 takes ~250ms and is a significant latency contributor

</details>

### Exercise 3: Cardinality Analysis

A developer proposes adding the following Prometheus metric to track API usage:

```python
api_requests = Counter(
    "api_requests_total",
    "Total API requests",
    labelnames=["method", "endpoint", "status_code", "user_id", "request_id", "region"]
)
```

Analyze the cardinality. Assume: 5 methods, 200 endpoints, 50 status codes, 100,000 users, unlimited request IDs, 3 regions. What is the theoretical maximum cardinality? Which labels should be removed or modified? Propose a revised metric definition.

<details>
<summary>Show Answer</summary>

**Cardinality analysis:**

```
Theoretical max = 5 × 200 × 50 × 100,000 × ∞ × 3 = ∞ (unbounded!)

Without request_id = 5 × 200 × 50 × 100,000 × 3 = 15,000,000,000
Still far too high.

Without user_id  = 5 × 200 × 50 × 3 = 150,000
Getting manageable but still high.

Bucketed status = 5 × 200 × 5 (2xx/3xx/4xx/5xx/other) × 3 = 15,000
Good range.
```

**Labels to remove:**
- `request_id` -- MUST remove. Unbounded cardinality. Use trace ID in traces instead.
- `user_id` -- MUST remove. 100K unique values. Use traces for per-user analysis.

**Labels to modify:**
- `status_code` → `status_class` -- Bucket into 2xx, 3xx, 4xx, 5xx (5 values instead of 50).
- `endpoint` -- Consider grouping parameterized paths: `/users/123` → `/users/:id` (reduces from 200 to ~30 route templates).

**Revised metric:**

```python
api_requests = Counter(
    "api_requests_total",
    "Total API requests",
    labelnames=["method", "route", "status_class", "region"]
)

# Cardinality: 5 × 30 × 5 × 3 = 2,250 time series (excellent)

# For per-user and per-request analysis, use traces:
with tracer.start_as_current_span("api_request") as span:
    span.set_attribute("user.id", user_id)  # High cardinality is fine in traces
    span.set_attribute("http.request_id", request_id)
```

</details>

---

## References

- [Observability Engineering (O'Reilly, Charity Majors et al.)](https://www.oreilly.com/library/view/observability-engineering/9781492076438/)
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [OpenTelemetry Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/)
- [CNCF Observability Whitepaper](https://www.cncf.io/blog/2021/09/01/cncf-end-user-technology-radar-observability/)
- [Google SRE Book -- Monitoring Distributed Systems](https://sre.google/sre-book/monitoring-distributed-systems/)
- [Distributed Systems Observability (O'Reilly)](https://www.oreilly.com/library/view/distributed-systems-observability/9781492033431/)
