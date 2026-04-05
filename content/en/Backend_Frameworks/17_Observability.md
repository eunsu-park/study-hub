# 17. Observability

**Previous**: [Production Deployment](./16_Production_Deployment.md) | **Next**: [Project: REST API](./18_Project_REST_API.md)

**Difficulty**: ⭐⭐⭐

## Learning Objectives

- Explain the three pillars of observability (logs, metrics, traces) and how they complement each other
- Implement structured logging with correlation IDs using structlog (Python) or pino (Node.js)
- Instrument a backend application with Prometheus metrics and create meaningful Grafana dashboards
- Add distributed tracing with OpenTelemetry to trace requests across microservices
- Design alerting strategies that minimize noise and maximize actionable signal

## Table of Contents

1. [Three Pillars of Observability](#1-three-pillars-of-observability)
2. [Structured Logging](#2-structured-logging)
3. [Metrics with Prometheus](#3-metrics-with-prometheus)
4. [Distributed Tracing with OpenTelemetry](#4-distributed-tracing-with-opentelemetry)
5. [Grafana Dashboards](#5-grafana-dashboards)
6. [Alerting Strategies](#6-alerting-strategies)
7. [Error Tracking with Sentry](#7-error-tracking-with-sentry)
8. [Practice Problems](#8-practice-problems)

---

## 1. Three Pillars of Observability

Observability is the ability to understand the internal state of a system by examining its external outputs. The three pillars provide complementary views of system behavior.

### Logs: What Happened

Logs record discrete events. They answer questions like "What happened at 3:14 PM?" and "Why did this request fail?"

```
2025-01-15T15:14:23Z INFO  user.login user_id=42 ip=192.168.1.5 method=password
2025-01-15T15:14:24Z ERROR payment.charge user_id=42 amount=99.99 error="card_declined"
```

### Metrics: How Much / How Fast

Metrics are numerical measurements aggregated over time. They answer questions like "How many requests per second?" and "What is the 99th percentile latency?"

```
http_requests_total{method="GET", endpoint="/users", status="200"} 15234
http_request_duration_seconds{quantile="0.99"} 0.250
```

### Traces: The Journey of a Request

Traces follow a single request as it travels through multiple services. They answer questions like "Where is the bottleneck in this slow request?" and "Which downstream service is causing errors?"

```
[Trace: abc123]
  |-- API Gateway (5ms)
  |   |-- Auth Service (12ms)
  |   |-- User Service (45ms)
  |       |-- PostgreSQL query (30ms)
  |       |-- Redis cache set (2ms)
  |-- Response sent (total: 62ms)
```

### How They Complement Each Other

| Scenario | Start With | Then Use |
|----------|-----------|----------|
| "Error rate spiked" | Metrics (detect) | Logs (root cause) |
| "This request is slow" | Traces (find bottleneck) | Metrics (is it systemic?) |
| "User reports failure" | Logs (find the event) | Traces (see full path) |

The key to connecting these pillars is a **correlation ID** (also called trace ID or request ID) that links a log entry to a trace span and a metric label.

---

## 2. Structured Logging

Structured logging outputs log events as key-value pairs (typically JSON) instead of free-form text. This makes logs parseable by machines while remaining readable by humans.

### Python: structlog

```python
import structlog
import uuid
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware

# Configure structlog
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Middleware to inject request context into every log entry
class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

        # Bind context variables that will appear in all log entries
        # within this request's scope
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client_ip=request.client.host,
        )

        logger.info("request.started")

        response = await call_next(request)

        logger.info(
            "request.completed",
            status_code=response.status_code,
        )

        response.headers["X-Request-ID"] = request_id
        return response

app = FastAPI()
app.add_middleware(RequestContextMiddleware)

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    logger.info("user.fetch", user_id=user_id)
    user = await fetch_user(user_id)
    if not user:
        logger.warning("user.not_found", user_id=user_id)
    return user
```

**Output (JSON, one line per event):**

```json
{"request_id": "a1b2c3", "method": "GET", "path": "/users/42", "client_ip": "10.0.0.1", "event": "request.started", "level": "info", "timestamp": "2025-01-15T15:14:23.000Z"}
{"request_id": "a1b2c3", "method": "GET", "path": "/users/42", "client_ip": "10.0.0.1", "user_id": 42, "event": "user.fetch", "level": "info", "timestamp": "2025-01-15T15:14:23.005Z"}
{"request_id": "a1b2c3", "method": "GET", "path": "/users/42", "client_ip": "10.0.0.1", "status_code": 200, "event": "request.completed", "level": "info", "timestamp": "2025-01-15T15:14:23.050Z"}
```

### Node.js: pino

```javascript
const pino = require("pino");
const express = require("express");
const { randomUUID } = require("crypto");

const logger = pino({
  level: process.env.LOG_LEVEL || "info",
  // Redact sensitive fields
  redact: ["req.headers.authorization", "req.headers.cookie"],
  serializers: {
    err: pino.stdSerializers.err,
  },
});

const app = express();

// Request logging middleware
app.use((req, res, next) => {
  const requestId = req.headers["x-request-id"] || randomUUID();
  // Create a child logger with request-scoped context
  req.log = logger.child({
    requestId,
    method: req.method,
    path: req.path,
    clientIp: req.ip,
  });

  req.log.info("request.started");

  const startTime = process.hrtime.bigint();
  res.on("finish", () => {
    const duration = Number(process.hrtime.bigint() - startTime) / 1e6;
    req.log.info({
      statusCode: res.statusCode,
      durationMs: duration.toFixed(2),
      event: "request.completed",
    });
  });

  res.setHeader("X-Request-ID", requestId);
  next();
});

app.get("/users/:id", (req, res) => {
  req.log.info({ userId: req.params.id }, "user.fetch");
  // ... handler logic
});
```

### Log Level Guidelines

| Level   | When to Use                                          | Examples                             |
|---------|------------------------------------------------------|--------------------------------------|
| DEBUG   | Detailed diagnostic info (disabled in production)    | SQL queries, cache hits/misses       |
| INFO    | Normal operational events                            | Request start/end, user actions      |
| WARNING | Unexpected but recoverable situations                | Deprecated API usage, slow query     |
| ERROR   | Failures that need attention                         | Unhandled exception, service timeout |
| CRITICAL| System-level failures requiring immediate action     | Database connection lost, OOM        |

---

## 3. Metrics with Prometheus

Prometheus collects metrics by scraping HTTP endpoints. Your application exposes a `/metrics` endpoint that returns metric values in Prometheus text format.

### Metric Types

| Type      | Description                          | Example                                    |
|-----------|--------------------------------------|--------------------------------------------|
| Counter   | Monotonically increasing value       | Total requests, total errors               |
| Gauge     | Value that goes up and down          | Active connections, queue depth             |
| Histogram | Distribution of values in buckets    | Request duration, response size            |
| Summary   | Similar to histogram, calculates quantiles client-side | Request duration percentiles |

### FastAPI with prometheus-fastapi-instrumentator

```python
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge
import time

app = FastAPI()

# Auto-instrument all endpoints with standard HTTP metrics
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# Custom business metrics
USERS_CREATED = Counter(
    "users_created_total",
    "Total number of users created",
    ["registration_method"],  # Labels for breaking down by method
)

PAYMENT_AMOUNT = Histogram(
    "payment_amount_dollars",
    "Payment amounts in dollars",
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000],
)

ACTIVE_WEBSOCKET_CONNECTIONS = Gauge(
    "active_websocket_connections",
    "Number of active WebSocket connections",
)

DB_QUERY_DURATION = Histogram(
    "db_query_duration_seconds",
    "Database query execution time",
    ["query_type", "table"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)


@app.post("/users")
async def create_user(user: UserCreate):
    result = await save_user(user)
    USERS_CREATED.labels(registration_method="email").inc()
    return result


@app.post("/payments")
async def process_payment(payment: PaymentRequest):
    result = await charge(payment)
    PAYMENT_AMOUNT.observe(payment.amount)
    return result


# Database query timing helper
class QueryTimer:
    def __init__(self, query_type: str, table: str):
        self.query_type = query_type
        self.table = table
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        duration = time.perf_counter() - self.start_time
        DB_QUERY_DURATION.labels(
            query_type=self.query_type, table=self.table
        ).observe(duration)


# Usage
async def get_user_by_id(user_id: int):
    with QueryTimer("select", "users"):
        return await db.fetch_one("SELECT * FROM users WHERE id = $1", user_id)
```

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: "fastapi-app"
    static_configs:
      - targets: ["app:8000"]
    metrics_path: /metrics

  - job_name: "node-exporter"
    static_configs:
      - targets: ["node-exporter:9100"]

  - job_name: "postgres-exporter"
    static_configs:
      - targets: ["postgres-exporter:9187"]
```

---

## 4. Distributed Tracing with OpenTelemetry

OpenTelemetry (OTel) is the industry standard for distributed tracing. It provides vendor-neutral instrumentation that can export to Jaeger, Zipkin, Grafana Tempo, or any OTLP-compatible backend.

### Key Concepts

- **Trace**: The complete journey of a request through all services
- **Span**: A single operation within a trace (e.g., HTTP request, database query)
- **Context propagation**: Passing the trace ID between services via HTTP headers

### FastAPI Instrumentation

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.resources import Resource

# Initialize the tracer provider
resource = Resource.create({
    "service.name": "user-service",
    "service.version": "1.2.0",
    "deployment.environment": "production",
})

provider = TracerProvider(resource=resource)
processor = BatchSpanProcessor(
    OTLPSpanExporter(endpoint="http://otel-collector:4317")
)
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer(__name__)

# Auto-instrument frameworks
app = FastAPI()
FastAPIInstrumentor.instrument_app(app)
SQLAlchemyInstrumentor().instrument(engine=engine)
HTTPXClientInstrumentor().instrument()

# Custom spans for business logic
@app.post("/orders")
async def create_order(order: OrderCreate):
    with tracer.start_as_current_span("validate_order") as span:
        span.set_attribute("order.item_count", len(order.items))
        validated = validate_order(order)

    with tracer.start_as_current_span("check_inventory") as span:
        span.set_attribute("order.total", order.total)
        available = await check_inventory(order.items)
        if not available:
            span.set_status(trace.Status(trace.StatusCode.ERROR, "Out of stock"))
            raise HTTPException(status_code=409, detail="Item out of stock")

    with tracer.start_as_current_span("process_payment") as span:
        payment = await charge_payment(order)
        span.set_attribute("payment.id", payment.id)
        span.set_attribute("payment.method", payment.method)

    return {"order_id": order.id, "status": "confirmed"}
```

### Express.js Instrumentation

```javascript
// tracing.js — must be imported BEFORE any other modules
const { NodeSDK } = require("@opentelemetry/sdk-node");
const { getNodeAutoInstrumentations } = require("@opentelemetry/auto-instrumentations-node");
const { OTLPTraceExporter } = require("@opentelemetry/exporter-trace-otlp-grpc");
const { Resource } = require("@opentelemetry/resources");

const sdk = new NodeSDK({
  resource: new Resource({
    "service.name": "order-service",
    "service.version": "2.1.0",
  }),
  traceExporter: new OTLPTraceExporter({
    url: "http://otel-collector:4317",
  }),
  instrumentations: [
    getNodeAutoInstrumentations({
      // Auto-instruments express, http, pg, redis, etc.
      "@opentelemetry/instrumentation-fs": { enabled: false },
    }),
  ],
});

sdk.start();
process.on("SIGTERM", () => sdk.shutdown());
```

```bash
# Start the app with tracing enabled
node --require ./tracing.js server.js
```

---

## 5. Grafana Dashboards

Grafana visualizes metrics from Prometheus (and other data sources) as dashboards. A well-designed dashboard provides at-a-glance visibility into system health.

### The RED Method for Services

The RED method defines three key metrics for every service:

| Metric       | Description                    | PromQL Example                                         |
|--------------|--------------------------------|--------------------------------------------------------|
| **R**ate     | Requests per second            | `rate(http_requests_total[5m])`                       |
| **E**rrors   | Error rate                     | `rate(http_requests_total{status=~"5.."}[5m])`        |
| **D**uration | Request latency distribution   | `histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))` |

### The USE Method for Resources

The USE method defines three key metrics for every resource (CPU, memory, disk, network):

| Metric           | Description                  | PromQL Example                                 |
|------------------|------------------------------|------------------------------------------------|
| **U**tilization  | Percentage of capacity used  | `process_cpu_seconds_total`                    |
| **S**aturation   | Queue depth, pending work    | `node_load1` (1-minute load average)           |
| **E**rrors       | Error events on the resource | `node_disk_io_errors_total`                    |

### Essential PromQL Queries

```promql
# Request rate (requests per second)
rate(http_requests_total[5m])

# Error rate as a percentage
100 * sum(rate(http_requests_total{status=~"5.."}[5m]))
    / sum(rate(http_requests_total[5m]))

# 99th percentile latency
histogram_quantile(0.99,
    sum(rate(http_request_duration_seconds_bucket[5m])) by (le)
)

# Latency by endpoint (top 5 slowest)
topk(5,
    histogram_quantile(0.95,
        sum(rate(http_request_duration_seconds_bucket[5m])) by (le, endpoint)
    )
)

# Active database connections
pg_stat_activity_count{state="active"}

# Memory usage percentage
100 * process_resident_memory_bytes / machine_memory_bytes
```

### Dashboard Layout Recommendation

```
Row 1: Overview
  [Request Rate] [Error Rate %] [P50/P95/P99 Latency]

Row 2: Endpoints
  [Top endpoints by request volume]  [Slowest endpoints (P99)]

Row 3: Resources
  [CPU Usage] [Memory Usage] [Active DB Connections]

Row 4: Business Metrics
  [Users Created/min] [Orders Processed/min] [Payment Failures]
```

---

## 6. Alerting Strategies

Good alerting notifies you of problems that need human action, without drowning you in noise.

### Alert Design Principles

- **Alert on symptoms, not causes**: Alert on "error rate > 5%" rather than "CPU > 90%". High CPU might be normal under load.
- **Actionable alerts only**: Every alert should have a clear action. If you routinely ignore an alert, delete it.
- **Include context**: The alert message should contain enough information to start debugging without opening a dashboard.
- **Tiered severity**: Not everything needs to wake someone up at 3 AM.

### Severity Levels

| Level    | Response Time | Channel      | Example                         |
|----------|---------------|-------------|---------------------------------|
| Critical | Immediate     | PagerDuty   | Service down, data loss risk    |
| Warning  | Business hours| Slack        | Error rate elevated, disk 80%   |
| Info     | Next sprint   | Email/ticket | Certificate expires in 30 days  |

### Prometheus Alert Rules

```yaml
# alerts.yml
groups:
  - name: api-alerts
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: |
          100 * sum(rate(http_requests_total{status=~"5.."}[5m]))
              / sum(rate(http_requests_total[5m]))
          > 5
        for: 5m    # Must persist for 5 minutes to fire
        labels:
          severity: critical
        annotations:
          summary: "High error rate: {{ $value | printf \"%.1f\" }}%"
          description: "Error rate has been above 5% for 5 minutes."
          runbook: "https://wiki.example.com/runbooks/high-error-rate"

      # Slow responses
      - alert: HighLatencyP99
        expr: |
          histogram_quantile(0.99,
              sum(rate(http_request_duration_seconds_bucket[5m])) by (le)
          ) > 1.0
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "P99 latency above 1s: {{ $value | printf \"%.2f\" }}s"

      # Service down
      - alert: ServiceDown
        expr: up{job="fastapi-app"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "{{ $labels.instance }} is down"

      # Database connection exhaustion
      - alert: DatabaseConnectionsHigh
        expr: pg_stat_activity_count{state="active"} > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Active DB connections: {{ $value }}/100"
```

---

## 7. Error Tracking with Sentry

Sentry captures unhandled exceptions with full context: stack traces, request data, user information, and breadcrumbs (a trail of events leading to the error).

### FastAPI Integration

```python
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

sentry_sdk.init(
    dsn="https://examplePublicKey@o0.ingest.sentry.io/0",
    environment="production",
    release="api-server@1.2.0",
    traces_sample_rate=0.1,      # Sample 10% of transactions for performance
    profiles_sample_rate=0.1,    # Sample 10% for profiling
    integrations=[
        FastApiIntegration(),
        SqlalchemyIntegration(),
    ],
    # Filter out sensitive data
    before_send=filter_sensitive_data,
)

def filter_sensitive_data(event, hint):
    """Remove sensitive information before sending to Sentry."""
    if "request" in event and "headers" in event["request"]:
        headers = event["request"]["headers"]
        # Remove authorization headers
        event["request"]["headers"] = {
            k: v for k, v in headers.items()
            if k.lower() not in ("authorization", "cookie", "x-api-key")
        }
    return event
```

### Express.js Integration

```javascript
const Sentry = require("@sentry/node");

Sentry.init({
  dsn: "https://examplePublicKey@o0.ingest.sentry.io/0",
  environment: process.env.NODE_ENV,
  release: `api-server@${process.env.npm_package_version}`,
  tracesSampleRate: 0.1,
  integrations: [
    Sentry.expressIntegration(),
    Sentry.httpIntegration(),
  ],
});

const app = express();

// Sentry request handler must be the first middleware
app.use(Sentry.expressRequestHandler());

// ... routes ...

// Sentry error handler must be before any other error middleware
app.use(Sentry.expressErrorHandler());

// Custom error handler
app.use((err, req, res, next) => {
  res.status(500).json({ error: "Internal server error" });
});
```

### Capturing Context

```python
# Add user context to all events in a request
@app.middleware("http")
async def sentry_user_context(request: Request, call_next):
    if hasattr(request.state, "user"):
        sentry_sdk.set_user({
            "id": request.state.user.id,
            "username": request.state.user.username,
            "email": request.state.user.email,
        })
    response = await call_next(request)
    return response

# Add breadcrumbs for debugging context
sentry_sdk.add_breadcrumb(
    category="payment",
    message=f"Processing payment for order {order_id}",
    level="info",
    data={"amount": 99.99, "currency": "USD"},
)

# Capture handled exceptions with extra context
try:
    result = await external_api.call()
except ExternalAPIError as e:
    sentry_sdk.capture_exception(e)
    # Continue with fallback logic
    result = get_cached_response()
```

---

## 8. Practice Problems

### Problem 1: Structured Logging Pipeline

Implement a complete logging pipeline for a FastAPI application that:
- Uses structlog with JSON output
- Includes request ID, user ID, and trace ID in every log entry
- Masks sensitive fields (passwords, tokens, credit card numbers)
- Writes to stdout (for container log collection)
- Includes a middleware that logs request/response with timing

### Problem 2: Custom Prometheus Metrics

Design and implement Prometheus metrics for an e-commerce API with the following requirements:
- Track order processing time by payment method (histogram)
- Track inventory levels by product category (gauge)
- Track failed payment attempts by reason (counter with labels)
- Track cart abandonment rate (derive from counters)
- Write the PromQL queries to power a dashboard for each metric

### Problem 3: Distributed Tracing Scenario

You have three services: API Gateway, Order Service, and Payment Service. A user creates an order, which triggers inventory check and payment processing. Implement OpenTelemetry instrumentation for all three services (use FastAPI) that:
- Propagates trace context across HTTP calls
- Creates custom spans for business logic
- Records span attributes for debugging
- Handles errors with appropriate span status

### Problem 4: Alerting Ruleset

Design a complete Prometheus alerting configuration for a production API service. Include alerts for:
- SLA violations (99.9% availability, P99 latency < 500ms)
- Resource exhaustion (CPU, memory, disk, connections)
- Business anomalies (order rate drops > 50% from rolling average)
- Security events (authentication failure rate spikes)
For each alert, specify severity, `for` duration, and write a runbook outline.

### Problem 5: Observability Stack with Docker Compose

Create a Docker Compose file that sets up a complete observability stack:
- Prometheus (metrics collection)
- Grafana (dashboards, pre-provisioned with a RED dashboard)
- Jaeger (trace visualization)
- OpenTelemetry Collector (receives traces, exports to Jaeger)
- A sample FastAPI application instrumented with all three pillars

Include the configuration files for each service and verify the entire pipeline works end-to-end.

---

## References

- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [structlog Documentation](https://www.structlog.org/)
- [pino Documentation](https://getpino.io/)
- [Sentry Documentation](https://docs.sentry.io/)
- [Google SRE Book: Monitoring Distributed Systems](https://sre.google/sre-book/monitoring-distributed-systems/)
- [The RED Method](https://www.weave.works/blog/the-red-method-key-metrics-for-microservices-architecture/)
- [The USE Method](https://www.brendangregg.com/usemethod.html)

---

**Previous**: [Production Deployment](./16_Production_Deployment.md) | **Next**: [Project: REST API](./18_Project_REST_API.md)
