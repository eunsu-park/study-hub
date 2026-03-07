# Distributed Tracing

**Previous**: [Logging Infrastructure](./11_Logging_Infrastructure.md) | **Next**: [Deployment Strategies](./13_Deployment_Strategies.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why distributed tracing is necessary for debugging microservice architectures
2. Define the core concepts of distributed tracing: traces, spans, context propagation, and baggage
3. Instrument applications using OpenTelemetry SDKs and auto-instrumentation
4. Deploy and query Jaeger as a tracing backend for visualizing request flows
5. Configure trace sampling strategies to balance observability with overhead
6. Correlate logs, metrics, and traces into a unified observability pipeline

---

In a monolithic application, a stack trace shows the complete call path from request to response. In a microservice architecture, a single user request might traverse 10 or more services, and a stack trace from one service reveals only a fragment of the story. Distributed tracing solves this by assigning a unique trace ID to each request and propagating it across every service boundary, creating a complete map of the request's journey through the system.

> **Analogy -- Package Tracking**: Distributed tracing works like a shipping carrier's tracking system. When you ship a package, the carrier assigns a tracking number (trace ID). At each transit hub (service), the package is scanned (span created) with a timestamp and location. If the package is delayed, you can view the full journey and pinpoint exactly which hub caused the delay. Without tracking, you would only know the package is late but not where it got stuck.

## 1. Why Distributed Tracing

### 1.1 The Debugging Problem in Microservices

```
User Request: "Why is the checkout page slow?"

┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│  API     │──→│  Order   │──→│ Inventory│──→│ Payment  │
│ Gateway  │   │ Service  │   │ Service  │   │ Service  │
│  (12ms)  │   │  (45ms)  │   │ (230ms!) │   │  (89ms)  │
└──────────┘   └──────────┘   └──────────┘   └──────────┘
                                    ↓
                              ┌──────────┐
                              │  Cache   │
                              │  (miss!) │
                              │ (215ms)  │
                              └──────────┘

Without tracing: "Something is slow, check all services"
With tracing:    "Inventory service → cache miss → 215ms DB query"
```

### 1.2 What Tracing Tells You That Metrics and Logs Cannot

| Question | Metrics | Logs | Traces |
|----------|---------|------|--------|
| "Is the system slow?" | Yes (p95 latency) | No | No |
| "Which service is slow?" | Partially (per-service latency) | No | **Yes (span breakdown)** |
| "Why is it slow?" | No | Partially (error messages) | **Yes (call graph + timing)** |
| "Which downstream call causes the bottleneck?" | No | No | **Yes (child span durations)** |
| "How does this request flow through the system?" | No | No | **Yes (trace visualization)** |

---

## 2. Core Concepts

### 2.1 Traces and Spans

```
Trace (trace_id: abc123)
│
├── Span A: API Gateway (0ms - 376ms)          [root span]
│   ├── Span B: Order Service (5ms - 350ms)     [child of A]
│   │   ├── Span C: Inventory Check (10ms - 240ms) [child of B]
│   │   │   ├── Span D: Cache Lookup (12ms - 15ms) [child of C]
│   │   │   └── Span E: DB Query (16ms - 235ms)    [child of C]
│   │   └── Span F: Payment Processing (245ms - 340ms) [child of B]
│   │       └── Span G: Stripe API Call (250ms - 335ms) [child of F]
│   └── Span H: Response Serialization (355ms - 370ms) [child of A]
```

**Key terminology:**

| Concept | Definition |
|---------|-----------|
| **Trace** | The complete journey of a request through all services. Identified by a `trace_id`. |
| **Span** | A single unit of work within a trace (e.g., an HTTP call, a DB query). Has a start time, duration, and metadata. |
| **Root span** | The first span in a trace (typically created by the entry-point service). |
| **Child span** | A span that is causally linked to a parent span. |
| **Span context** | The data propagated across service boundaries: `trace_id`, `span_id`, `trace_flags`. |
| **Baggage** | User-defined key-value pairs propagated across all spans in a trace (e.g., `user_id`, `tenant_id`). |

### 2.2 Span Attributes

Each span carries structured metadata:

```json
{
  "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
  "span_id": "00f067aa0ba902b7",
  "parent_span_id": "a3ce929d0e0e4736",
  "operation_name": "HTTP GET /api/orders",
  "service_name": "order-service",
  "start_time": "2024-03-15T14:23:45.123Z",
  "duration_ms": 345,
  "status": "OK",
  "attributes": {
    "http.method": "GET",
    "http.url": "/api/orders/123",
    "http.status_code": 200,
    "db.system": "postgresql",
    "db.statement": "SELECT * FROM orders WHERE id = $1",
    "net.peer.name": "db-primary.internal",
    "net.peer.port": 5432
  },
  "events": [
    {
      "name": "cache.miss",
      "timestamp": "2024-03-15T14:23:45.130Z",
      "attributes": { "cache.key": "order:123" }
    }
  ]
}
```

### 2.3 Context Propagation

Context propagation is the mechanism that links spans across service boundaries. The trace context is injected into outgoing requests (HTTP headers, message headers) and extracted by the receiving service.

```
Service A                              Service B
┌─────────────────────┐                ┌─────────────────────┐
│                     │                │                     │
│  Create Span A      │                │                     │
│       │             │   HTTP Request │                     │
│       │  Inject ────┼──────────────→ │  Extract            │
│       │  context    │   Headers:     │  context            │
│       │  into       │   traceparent: │  from               │
│       │  headers    │   00-abc123... │  headers            │
│       │             │                │       │             │
│       │             │                │  Create Span B      │
│       │             │                │  (child of A)       │
│       │             │                │       │             │
│       │             │   HTTP Response│       │             │
│       │  ◄──────────┼───────────────┤│       │             │
│       │             │                │                     │
└─────────────────────┘                └─────────────────────┘
```

**W3C Trace Context header format:**
```
traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01
             ──  ────────────────────────────────  ────────────────  ──
             ver           trace-id                   parent-id     flags
                                                                    (01 = sampled)
```

---

## 3. OpenTelemetry

### 3.1 What is OpenTelemetry

OpenTelemetry (OTel) is a vendor-neutral, open-source observability framework that provides APIs, SDKs, and tools for generating, collecting, and exporting telemetry data (traces, metrics, logs).

```
┌─────────────────────────────────────────────────────────────────┐
│                    OpenTelemetry Architecture                     │
│                                                                  │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐              │
│  │   App +    │   │   App +    │   │   App +    │              │
│  │ OTel SDK   │   │ OTel SDK   │   │ OTel SDK   │              │
│  └─────┬──────┘   └─────┬──────┘   └─────┬──────┘              │
│        │ OTLP           │ OTLP           │ OTLP                │
│        └────────────────┼────────────────┘                      │
│                         ▼                                        │
│              ┌──────────────────┐                                │
│              │  OTel Collector  │                                │
│              │ ┌──────────────┐ │                                │
│              │ │  Receivers   │ │  (OTLP, Jaeger, Zipkin)       │
│              │ ├──────────────┤ │                                │
│              │ │  Processors  │ │  (batch, filter, sample)      │
│              │ ├──────────────┤ │                                │
│              │ │  Exporters   │ │  (Jaeger, Zipkin, OTLP)       │
│              │ └──────────────┘ │                                │
│              └────────┬─────────┘                                │
│           ┌───────────┼───────────┐                              │
│           ▼           ▼           ▼                              │
│     ┌──────────┐ ┌──────────┐ ┌──────────┐                     │
│     │  Jaeger  │ │  Tempo   │ │  Zipkin  │                     │
│     └──────────┘ └──────────┘ └──────────┘                     │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Instrumentation in Python

```python
# Install: pip install opentelemetry-api opentelemetry-sdk
#          opentelemetry-exporter-otlp opentelemetry-instrumentation-flask
#          opentelemetry-instrumentation-requests

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

# Configure the tracer provider
resource = Resource.create({
    "service.name": "order-service",
    "service.version": "1.2.0",
    "deployment.environment": "production",
})

provider = TracerProvider(resource=resource)
processor = BatchSpanProcessor(
    OTLPSpanExporter(endpoint="http://otel-collector:4317")
)
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

# Get a tracer
tracer = trace.get_tracer("order-service")

# Auto-instrument Flask and requests library
from flask import Flask
app = Flask(__name__)
FlaskInstrumentor().instrument_app(app)
RequestsInstrumentor().instrument()

# Manual instrumentation
@app.route("/api/orders/<order_id>")
def get_order(order_id):
    # Flask auto-instrumentation creates the parent span

    # Create a child span for business logic
    with tracer.start_as_current_span("fetch_order_from_db") as span:
        span.set_attribute("order.id", order_id)
        order = db.query("SELECT * FROM orders WHERE id = %s", order_id)

        if order is None:
            span.set_attribute("order.found", False)
            span.set_status(trace.StatusCode.ERROR, "Order not found")
            return {"error": "not found"}, 404

        span.set_attribute("order.found", True)
        span.set_attribute("order.total", order.total)

    # Create another span for enrichment
    with tracer.start_as_current_span("enrich_order") as span:
        span.add_event("fetching_customer_details", {"customer_id": order.customer_id})
        customer = requests.get(f"http://customer-service/api/customers/{order.customer_id}")
        order.customer = customer.json()

    return order.to_dict()
```

### 3.3 Instrumentation in Go

```go
package main

import (
    "context"
    "net/http"

    "go.opentelemetry.io/otel"
    "go.opentelemetry.io/otel/attribute"
    "go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
    "go.opentelemetry.io/otel/sdk/resource"
    sdktrace "go.opentelemetry.io/otel/sdk/trace"
    semconv "go.opentelemetry.io/otel/semconv/v1.21.0"
    "go.opentelemetry.io/otel/trace"
    "go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
)

func initTracer() (*sdktrace.TracerProvider, error) {
    exporter, err := otlptracegrpc.New(context.Background(),
        otlptracegrpc.WithEndpoint("otel-collector:4317"),
        otlptracegrpc.WithInsecure(),
    )
    if err != nil {
        return nil, err
    }

    tp := sdktrace.NewTracerProvider(
        sdktrace.WithBatcher(exporter),
        sdktrace.WithResource(resource.NewWithAttributes(
            semconv.SchemaURL,
            semconv.ServiceNameKey.String("order-service"),
            semconv.ServiceVersionKey.String("1.2.0"),
            attribute.String("environment", "production"),
        )),
    )
    otel.SetTracerProvider(tp)
    return tp, nil
}

func getOrder(w http.ResponseWriter, r *http.Request) {
    ctx := r.Context()
    tracer := otel.Tracer("order-service")

    // Create a child span
    ctx, span := tracer.Start(ctx, "fetch_order_from_db")
    span.SetAttributes(attribute.String("order.id", "123"))
    order, err := fetchOrder(ctx, "123")
    if err != nil {
        span.RecordError(err)
        span.SetStatus(trace.StatusCodeError, err.Error())
    }
    span.End()

    // Another child span
    ctx, span = tracer.Start(ctx, "enrich_order")
    enrichOrder(ctx, order)
    span.End()
}

func main() {
    tp, _ := initTracer()
    defer tp.Shutdown(context.Background())

    // Wrap handler with OTel HTTP instrumentation
    handler := otelhttp.NewHandler(http.HandlerFunc(getOrder), "GET /api/orders")
    http.Handle("/api/orders", handler)
    http.ListenAndServe(":8080", nil)
}
```

### 3.4 Auto-Instrumentation (Zero-Code)

OpenTelemetry supports auto-instrumentation for many languages, which instruments common libraries without code changes:

```bash
# Python: auto-instrument any Flask/Django/FastAPI app
pip install opentelemetry-distro opentelemetry-exporter-otlp
opentelemetry-bootstrap -a install    # Install all applicable instrumentations

# Run with auto-instrumentation
opentelemetry-instrument \
  --service_name order-service \
  --exporter_otlp_endpoint http://otel-collector:4317 \
  python app.py

# Java: attach the agent JAR
java -javaagent:opentelemetry-javaagent.jar \
  -Dotel.service.name=order-service \
  -Dotel.exporter.otlp.endpoint=http://otel-collector:4317 \
  -jar app.jar

# Node.js: require the SDK before the app
node --require @opentelemetry/auto-instrumentations-node/register app.js
```

---

## 4. OpenTelemetry Collector

### 4.1 Collector Configuration

```yaml
# otel-collector-config.yml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

  # Also accept Jaeger and Zipkin formats
  jaeger:
    protocols:
      thrift_http:
        endpoint: 0.0.0.0:14268
  zipkin:
    endpoint: 0.0.0.0:9411

processors:
  # Batch spans for efficient export
  batch:
    send_batch_size: 1024
    timeout: 5s

  # Add resource attributes
  resource:
    attributes:
      - key: environment
        value: production
        action: upsert

  # Tail-based sampling (keep errors and slow traces)
  tail_sampling:
    decision_wait: 10s
    policies:
      - name: errors
        type: status_code
        status_code:
          status_codes: [ERROR]
      - name: slow-traces
        type: latency
        latency:
          threshold_ms: 1000
      - name: probabilistic
        type: probabilistic
        probabilistic:
          sampling_percentage: 10

  # Filter out health check traces
  filter:
    traces:
      span:
        - 'attributes["http.target"] == "/health"'
        - 'attributes["http.target"] == "/ready"'

exporters:
  # Export to Jaeger
  jaeger:
    endpoint: jaeger:14250
    tls:
      insecure: true

  # Export to Grafana Tempo
  otlp/tempo:
    endpoint: tempo:4317
    tls:
      insecure: true

  # Debug exporter (stdout)
  debug:
    verbosity: detailed

service:
  pipelines:
    traces:
      receivers: [otlp, jaeger, zipkin]
      processors: [filter, resource, tail_sampling, batch]
      exporters: [jaeger, otlp/tempo]
```

---

## 5. Jaeger

### 5.1 Jaeger Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Jaeger Architecture                         │
│                                                                  │
│  ┌──────────┐                                                   │
│  │  App +   │──OTLP──→ ┌──────────────┐                       │
│  │  OTel    │           │  Collector   │                       │
│  └──────────┘           │  (validate,  │                       │
│                         │   index,     │                       │
│  ┌──────────┐           │   transform) │                       │
│  │  App +   │──OTLP──→ └──────┬───────┘                       │
│  │  OTel    │                  │                                │
│  └──────────┘                  ▼                                │
│                         ┌──────────────┐    ┌──────────────┐   │
│                         │   Storage    │    │   Jaeger UI  │   │
│                         │ (Cassandra / │◄──→│  (query &    │   │
│                         │ Elasticsearch│    │  visualize)  │   │
│                         │  / Badger)   │    └──────────────┘   │
│                         └──────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Deploying Jaeger (All-in-One for Development)

```yaml
# docker-compose-jaeger.yml
version: "3.8"
services:
  jaeger:
    image: jaegertracing/all-in-one:1.54
    ports:
      - "16686:16686"    # Jaeger UI
      - "14268:14268"    # Jaeger collector HTTP
      - "14250:14250"    # Jaeger collector gRPC
      - "4317:4317"      # OTLP gRPC
      - "4318:4318"      # OTLP HTTP
    environment:
      - COLLECTOR_OTLP_ENABLED=true
      - SPAN_STORAGE_TYPE=badger
      - BADGER_EPHEMERAL=false
      - BADGER_DIRECTORY_VALUE=/badger/data
      - BADGER_DIRECTORY_KEY=/badger/key
    volumes:
      - jaeger-data:/badger

  otel-collector:
    image: otel/opentelemetry-collector-contrib:0.95.0
    ports:
      - "4317:4317"
      - "4318:4318"
    volumes:
      - ./otel-collector-config.yml:/etc/otel/config.yaml
    command: ["--config", "/etc/otel/config.yaml"]
    depends_on:
      - jaeger

volumes:
  jaeger-data:
```

### 5.3 Querying Traces in Jaeger UI

Jaeger UI provides several ways to find traces:

| Query Method | Use Case | Example |
|-------------|----------|---------|
| **Service + Operation** | Find traces for a specific endpoint | Service: `order-service`, Operation: `GET /api/orders` |
| **Tags** | Filter by span attributes | `http.status_code=500`, `user.id=u-123` |
| **Duration** | Find slow traces | Min: `1s`, Max: `10s` |
| **Trace ID** | Look up a specific trace | `4bf92f3577b34da6a3ce929d0e0e4736` |
| **Time range** | Narrow the search window | Last 1 hour, last 24 hours |

### 5.4 Jaeger API Queries

```bash
# Find traces for a service
curl "http://jaeger:16686/api/traces?service=order-service&limit=20"

# Find traces with specific tags
curl "http://jaeger:16686/api/traces?service=order-service&tags=%7B%22http.status_code%22%3A%22500%22%7D"

# Get a specific trace
curl "http://jaeger:16686/api/traces/4bf92f3577b34da6a3ce929d0e0e4736"

# List services
curl "http://jaeger:16686/api/services"

# List operations for a service
curl "http://jaeger:16686/api/services/order-service/operations"
```

---

## 6. Trace Sampling

### 6.1 Why Sampling is Necessary

At scale, tracing every request is impractical:

| Metric | Without Sampling | With 10% Sampling |
|--------|-----------------|-------------------|
| **Traces/second** | 10,000 | 1,000 |
| **Storage/day** | ~100 GB | ~10 GB |
| **Network overhead** | High | Low |
| **Cost** | $$$$ | $ |

### 6.2 Sampling Strategies

```
┌─────────────────────────────────────────────────────────────┐
│                   Sampling Strategies                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Head-Based Sampling (decision at trace start)               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  • Probabilistic: sample 10% of traces randomly     │    │
│  │  • Rate-limiting: max N traces per second            │    │
│  │  • Pros: simple, low overhead                        │    │
│  │  • Cons: may miss interesting (error/slow) traces    │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  Tail-Based Sampling (decision after trace is complete)      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  • Keep all error traces                             │    │
│  │  • Keep all traces > latency threshold               │    │
│  │  • Keep all traces matching specific attributes      │    │
│  │  • Pros: never miss important traces                 │    │
│  │  • Cons: requires buffering complete traces          │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  Hybrid: Head-based 10% + tail-based for errors/slow        │
└─────────────────────────────────────────────────────────────┘
```

### 6.3 Configuring Sampling

**Head-based sampling (in SDK):**

```python
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.sampling import (
    TraceIdRatioBased,
    ParentBasedTraceIdRatio,
)

# Sample 10% of traces (but follow parent's sampling decision)
sampler = ParentBasedTraceIdRatio(0.1)

provider = TracerProvider(
    sampler=sampler,
    resource=resource,
)
```

**Tail-based sampling (in Collector):**

```yaml
# otel-collector tail sampling processor
processors:
  tail_sampling:
    decision_wait: 10s
    num_traces: 100000
    policies:
      # Always keep error traces
      - name: errors-policy
        type: status_code
        status_code:
          status_codes: [ERROR]

      # Always keep slow traces (> 2 seconds)
      - name: latency-policy
        type: latency
        latency:
          threshold_ms: 2000

      # Sample 5% of remaining traces
      - name: probabilistic-policy
        type: probabilistic
        probabilistic:
          sampling_percentage: 5
```

---

## 7. Correlating Logs, Metrics, and Traces

### 7.1 The Correlation Model

```
┌─────────────────────────────────────────────────────────────┐
│              Unified Observability Correlation                │
│                                                              │
│  Metrics (Prometheus)                                        │
│  ┌──────────────────────────────────────────────────┐       │
│  │  http_request_duration_seconds{service="orders"} │       │
│  │  → "p99 latency spiked at 14:23"                │       │
│  └──────────────────┬───────────────────────────────┘       │
│                     │ "Show me traces from 14:23"            │
│                     ▼                                        │
│  Traces (Jaeger)                                             │
│  ┌──────────────────────────────────────────────────┐       │
│  │  trace_id: abc123                                │       │
│  │  → "Inventory service DB query took 2.3s"        │       │
│  └──────────────────┬───────────────────────────────┘       │
│                     │ "Show me logs for trace abc123"         │
│                     ▼                                        │
│  Logs (Loki / ELK)                                           │
│  ┌──────────────────────────────────────────────────┐       │
│  │  {trace_id="abc123"} "Connection pool exhausted, │       │
│  │  waited 2100ms for available connection"          │       │
│  └──────────────────────────────────────────────────┘       │
│                                                              │
│  Correlation Key: trace_id links all three signals           │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 Injecting Trace Context into Logs

**Python:**

```python
import structlog
from opentelemetry import trace

def add_trace_context(logger, method_name, event_dict):
    """Structlog processor that adds trace context to every log entry."""
    span = trace.get_current_span()
    if span.is_recording():
        ctx = span.get_span_context()
        event_dict["trace_id"] = format(ctx.trace_id, "032x")
        event_dict["span_id"] = format(ctx.span_id, "016x")
    return event_dict

structlog.configure(
    processors=[
        add_trace_context,   # Add trace context to all logs
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)

log = structlog.get_logger()
# Now every log automatically includes trace_id and span_id
log.info("processing_order", order_id="ord-123")
# Output: {"event": "processing_order", "order_id": "ord-123",
#          "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
#          "span_id": "00f067aa0ba902b7", "level": "info"}
```

### 7.3 Exemplars: Linking Metrics to Traces

Exemplars attach a trace ID to a specific metric data point, enabling direct navigation from a metric to the trace that produced it:

```python
from prometheus_client import Histogram
from opentelemetry import trace

REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "Request duration",
    ["method", "endpoint"],
)

def handle_request(method, endpoint, duration):
    span = trace.get_current_span()
    ctx = span.get_span_context()
    trace_id = format(ctx.trace_id, "032x")

    # Observe with exemplar (links metric → trace)
    REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(
        duration,
        exemplar={"trace_id": trace_id}
    )
```

In Grafana, clicking an exemplar data point on a Prometheus graph navigates directly to the corresponding trace in Jaeger or Tempo.

### 7.4 Grafana Unified View

Configure Grafana data source correlations:

```yaml
# grafana/provisioning/datasources/correlations.yml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    url: http://prometheus:9090
    jsonData:
      exemplarTraceIdDestinations:
        - name: trace_id
          datasourceUid: tempo
          urlDisplayLabel: "View Trace"

  - name: Tempo
    uid: tempo
    type: tempo
    url: http://tempo:3200
    jsonData:
      tracesToLogsV2:
        datasourceUid: loki
        filterByTraceID: true
        filterBySpanID: false
      tracesToMetrics:
        datasourceUid: prometheus
        queries:
          - name: "Request rate"
            query: "sum(rate(http_requests_total{$$__tags}[5m]))"

  - name: Loki
    uid: loki
    type: loki
    url: http://loki:3100
    jsonData:
      derivedFields:
        - datasourceUid: tempo
          matcherRegex: "trace_id=(\\w+)"
          name: TraceID
          url: "$${__value.raw}"
```

---

## 8. Best Practices

### 8.1 Instrumentation Guidelines

| Practice | Description |
|----------|-------------|
| **Name spans by operation, not parameters** | `GET /api/users/{id}` not `GET /api/users/12345` (avoids high cardinality) |
| **Set meaningful attributes** | Include `user.id`, `order.id`, `db.statement` for debugging |
| **Record errors on spans** | Use `span.record_exception(error)` and `span.set_status(ERROR)` |
| **Keep span count reasonable** | 10-50 spans per trace is typical; avoid creating spans for every function call |
| **Use semantic conventions** | Follow OpenTelemetry semantic conventions for attribute names |
| **Propagate context to async work** | Pass trace context to background jobs, message consumers, and thread pools |

### 8.2 Span Naming Best Practices

```python
# Good span names (low cardinality, descriptive)
"HTTP GET /api/users/{id}"
"db.query SELECT orders"
"cache.get user_profile"
"queue.publish order_created"
"grpc.server /payment.PaymentService/Charge"

# Bad span names (high cardinality or vague)
"HTTP GET /api/users/12345"     # Parameter in name
"processing"                     # Too vague
"function_call"                  # Not meaningful
"step_1"                         # Not descriptive
```

---

## 9. Next Steps

- [13_Deployment_Strategies.md](./13_Deployment_Strategies.md) - Safe deployment patterns
- [10_Monitoring_and_Alerting.md](./10_Monitoring_and_Alerting.md) - Metrics and alerting with Prometheus
- [11_Logging_Infrastructure.md](./11_Logging_Infrastructure.md) - Centralized logging

---

## Exercises

### Exercise 1: Trace Analysis

Given the following trace, identify the performance bottleneck and suggest a fix:

```
Trace ID: abc123 (total: 2,450ms)
│
├── API Gateway: GET /api/checkout (0-2450ms)
│   ├── Auth Service: validate_token (5-25ms)
│   ├── Cart Service: get_cart (30-180ms)
│   │   ├── Redis: GET cart:user123 (35-38ms) [cache miss]
│   │   └── PostgreSQL: SELECT * FROM cart_items (40-175ms)
│   ├── Inventory Service: check_stock (185-2200ms)
│   │   ├── PostgreSQL: SELECT * FROM inventory (190-200ms)
│   │   ├── External API: supplier_stock_check (205-2190ms) ← SLOW
│   │   └── Cache: SET inventory:result (2195-2200ms)
│   └── Payment Service: process_payment (2205-2440ms)
│       └── Stripe API: charge (2210-2435ms)
```

<details>
<summary>Show Answer</summary>

**Bottleneck**: The `Inventory Service → External API: supplier_stock_check` span takes 1,985ms (205ms to 2,190ms), which is 81% of the total trace duration.

**Root cause analysis**:
1. The Redis cache miss on the cart means the cart had to be fetched from PostgreSQL (but this is only 175ms -- not the main issue).
2. The inventory check itself (DB query) is fast (10ms), but the external supplier stock check API call is extremely slow at 1,985ms.
3. The payment processing (235ms) and Stripe call (225ms) are reasonable.

**Recommended fixes**:
1. **Cache the supplier stock response**: The trace shows a `Cache: SET` after the supplier call, meaning caching is already planned. Ensure the cache TTL is appropriate (e.g., 5 minutes for stock levels) so subsequent requests hit the cache.
2. **Add a timeout**: Set a 500ms timeout on the external supplier API call with a circuit breaker. If the supplier is slow, fall back to the cached inventory count.
3. **Make it asynchronous**: If real-time supplier stock is not required for checkout, fetch supplier stock asynchronously and use locally cached inventory for the checkout flow.
4. **Parallelize**: The cart fetch and inventory check are independent. Execute them concurrently to save ~180ms: `total = max(cart_time, inventory_time) + payment_time` instead of `cart_time + inventory_time + payment_time`.

</details>

### Exercise 2: OpenTelemetry Instrumentation

Write Python code to instrument a function that calls two downstream services sequentially. Include proper span creation, attribute setting, error handling, and context propagation.

<details>
<summary>Show Answer</summary>

```python
import requests
from opentelemetry import trace
from opentelemetry.trace import StatusCode

tracer = trace.get_tracer("checkout-service")

def process_checkout(user_id: str, order_id: str):
    """Process a checkout by validating inventory and charging payment."""

    with tracer.start_as_current_span("process_checkout") as root_span:
        root_span.set_attribute("user.id", user_id)
        root_span.set_attribute("order.id", order_id)

        # Step 1: Check inventory
        with tracer.start_as_current_span("check_inventory") as inv_span:
            inv_span.set_attribute("service.downstream", "inventory-service")
            try:
                # Context propagation happens automatically via
                # RequestsInstrumentor (trace headers injected)
                resp = requests.get(
                    f"http://inventory-service/api/check/{order_id}",
                    timeout=5.0
                )
                inv_span.set_attribute("http.status_code", resp.status_code)

                if resp.status_code != 200:
                    inv_span.set_status(StatusCode.ERROR, "Inventory check failed")
                    inv_span.add_event("inventory_unavailable", {
                        "order_id": order_id,
                        "response_code": resp.status_code
                    })
                    raise ValueError(f"Inventory unavailable: {resp.text}")

                inventory = resp.json()
                inv_span.set_attribute("inventory.available", inventory["available"])

            except requests.Timeout as e:
                inv_span.record_exception(e)
                inv_span.set_status(StatusCode.ERROR, "Inventory service timeout")
                raise

        # Step 2: Process payment
        with tracer.start_as_current_span("process_payment") as pay_span:
            pay_span.set_attribute("service.downstream", "payment-service")
            pay_span.set_attribute("payment.amount", inventory["total"])
            pay_span.set_attribute("payment.currency", "USD")
            try:
                resp = requests.post(
                    "http://payment-service/api/charge",
                    json={
                        "order_id": order_id,
                        "user_id": user_id,
                        "amount": inventory["total"],
                    },
                    timeout=10.0
                )
                pay_span.set_attribute("http.status_code", resp.status_code)

                if resp.status_code != 200:
                    pay_span.set_status(StatusCode.ERROR, "Payment failed")
                    raise ValueError(f"Payment failed: {resp.text}")

                payment = resp.json()
                pay_span.set_attribute("payment.transaction_id", payment["transaction_id"])
                pay_span.add_event("payment_success", {
                    "transaction_id": payment["transaction_id"]
                })

            except requests.Timeout as e:
                pay_span.record_exception(e)
                pay_span.set_status(StatusCode.ERROR, "Payment service timeout")
                raise

        root_span.set_status(StatusCode.OK)
        return {"status": "success", "transaction_id": payment["transaction_id"]}
```

**Key points in the answer:**
- Each downstream call gets its own span with meaningful attributes.
- `record_exception()` captures the full stack trace on the span.
- `set_status(ERROR)` marks the span as failed in the Jaeger UI.
- `add_event()` records notable events within a span.
- Context propagation to downstream services is handled automatically by the `RequestsInstrumentor`.

</details>

### Exercise 3: Sampling Strategy Design

A company has three environments with different tracing needs:

| Environment | Traffic | Budget | Requirements |
|------------|---------|--------|-------------|
| Development | 100 req/s | Low | Full visibility for debugging |
| Staging | 1,000 req/s | Medium | Catch performance regressions |
| Production | 50,000 req/s | Limited | Never miss errors; keep costs manageable |

Design a sampling strategy for each environment.

<details>
<summary>Show Answer</summary>

**Development (100 req/s):**
- **Strategy**: 100% sampling (sample everything)
- **Rationale**: Low traffic means storage cost is negligible. Full visibility helps developers debug issues during development. At 100 req/s, even with all traces stored, daily storage is only ~1 GB.
- **Configuration**: `sampler = ALWAYS_ON`

**Staging (1,000 req/s):**
- **Strategy**: Head-based 50% + tail-based for errors and slow traces
- **Rationale**: Staging is used for performance testing, so you need enough samples to detect regressions. 50% gives statistically significant data. Tail-based sampling ensures all errors are captured.
- **Configuration**:
  ```yaml
  # SDK: head-based 50%
  sampler: ParentBasedTraceIdRatio(0.5)

  # Collector: tail-based additions
  tail_sampling:
    policies:
      - name: errors
        type: status_code
        status_codes: [ERROR]
      - name: slow
        type: latency
        threshold_ms: 500
  ```

**Production (50,000 req/s):**
- **Strategy**: Head-based 1% + tail-based for errors (100%) and slow traces (> 2s)
- **Rationale**: At 50K req/s, 1% sampling gives 500 traces/second -- more than enough for trend analysis. Tail-based sampling guarantees every error and slow trace is captured regardless of the head-based decision. This keeps storage at ~5 GB/day instead of ~500 GB/day.
- **Configuration**:
  ```yaml
  # SDK: head-based 1%
  sampler: ParentBasedTraceIdRatio(0.01)

  # Collector: tail-based ensures important traces survive
  tail_sampling:
    policies:
      - name: always-errors
        type: status_code
        status_codes: [ERROR]
      - name: slow-traces
        type: latency
        threshold_ms: 2000
      - name: probabilistic-fallback
        type: probabilistic
        sampling_percentage: 1
  ```

**Storage estimate at production scale:**
- Full sampling: 50K traces/s * 86400s/day * ~10KB/trace = ~43 TB/day
- 1% head + tail: ~500 traces/s + ~100 error traces/s = ~5 GB/day

</details>

### Exercise 4: Correlation Pipeline

Design a complete observability correlation setup for a three-service architecture (API Gateway, Order Service, Payment Service). Describe:

1. How trace context flows between services
2. How logs include the trace ID
3. How to navigate from a Prometheus alert to the relevant trace

<details>
<summary>Show Answer</summary>

**1. Trace context flow:**

```
Client → API Gateway → Order Service → Payment Service
         │                │                │
         │ Creates root   │ Extracts       │ Extracts
         │ span, injects  │ traceparent    │ traceparent
         │ traceparent    │ from header,   │ from header,
         │ header         │ creates child  │ creates child
         │                │ span, injects  │ span
         │                │ traceparent    │
         │                │ in outgoing    │
         │                │ request        │

HTTP Headers at each hop:
traceparent: 00-{trace_id}-{span_id}-01
```

All three services use OpenTelemetry SDKs with auto-instrumentation for HTTP clients and servers. The `traceparent` header is automatically injected/extracted.

**2. Logs include trace ID:**

Each service configures structured logging with a trace context processor:
```python
# In each service's logging setup
log.info("order_created",
         order_id="ord-123",
         trace_id="4bf92f...",   # Auto-injected from current span
         span_id="00f067...")
```

In Loki, these logs are queryable by trace ID:
```logql
{service="order-service"} | json | trace_id="4bf92f3577b34da6a3ce929d0e0e4736"
```

**3. Prometheus alert to trace navigation:**

Flow: Prometheus Alert → Grafana Dashboard → Exemplar → Jaeger Trace → Loki Logs

Step-by-step:
1. Prometheus fires `HighLatency` alert for `order-service` at 14:23.
2. Engineer opens the Grafana dashboard, sees the latency spike on the `http_request_duration_seconds` panel.
3. The panel shows exemplar dots on the graph. Clicking an exemplar shows `trace_id: 4bf92f...` and a "View Trace" link.
4. Clicking "View Trace" opens the trace in Jaeger/Tempo, showing which downstream call caused the latency.
5. From the trace view, clicking "View Logs" shows all logs from all services for that trace, filtered by `trace_id`.

This is enabled by:
- Prometheus exemplars linking metrics to traces
- Grafana data source correlations (Prometheus → Tempo → Loki)
- Consistent `trace_id` field in all logs

</details>

---

## References

- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [OpenTelemetry Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)
- [W3C Trace Context Specification](https://www.w3.org/TR/trace-context/)
- [Grafana Tempo Documentation](https://grafana.com/docs/tempo/latest/)
- [Google Dapper Paper](https://research.google/pubs/pub36356/)
