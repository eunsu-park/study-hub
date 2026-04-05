# 21. 신호 상관 관계(Signal Correlation)

**이전**: [SLO 엔지니어링](./20_SLO_Engineering.md) | **다음**: [고급 메트릭 아키텍처](./22_Advanced_Metrics_Architecture.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있다:

1. 분산 시스템에서 효과적인 디버깅을 위해 메트릭, 로그, 트레이스를 상관시키는 것이 필수적인 이유를 설명한다
2. 트레이스 ID와 구조화된 로깅을 사용하여 트레이스-로그(trace-to-log) 연결을 구현한다
3. Prometheus exemplar를 사용하여 집계된 메트릭을 특정 트레이스에 연결한다
4. 모든 신호가 상관 관계 식별자를 공유하는 통합 텔레메트리 아키텍처를 설계한다
5. 메트릭, 로그, 트레이스 간 원활한 탐색이 가능한 Grafana 대시보드를 구축한다
6. 상관 관계 기법을 적용하여 인시던트 중 평균 해결 시간(MTTR)을 줄인다

---

개별 텔레메트리 신호는 단독으로도 강력하지만, 진정한 가치는 상관 관계를 맺을 때 나타난다. 메트릭은 *무언가 잘못되었다*고 알려준다. 트레이스는 *어떤 요청 경로가 영향을 받는지* 알려준다. 로그는 *정확히 무슨 일이 일어났는지* 알려준다. 상관 관계는 메트릭 스파이크에서 근본 원인을 설명하는 특정 트레이스와 로그 항목으로 이동할 수 있게 하는 접착제로, 수 시간의 디버깅을 수 분으로 바꾼다.

> **비유 -- 범죄 현장 수사(Crime Scene Investigation)**: 형사는 지문(메트릭), 목격자 진술(로그), 보안 카메라 영상(트레이스)만으로는 사건을 해결하지 않는다. 세 가지를 *상관시켜* 해결한다: 문 손잡이의 지문(14:00의 메트릭 스파이크), 14:00에 특정 인물이 진입하는 카메라 영상(트레이스), 내부에서 일어난 일을 설명하는 목격자 진술(로그 항목). 상관 관계 없이 각 증거는 단절된 단서일 뿐이다.

## 1. 상관 관계 문제

### 1.1 상관 관계 없는 디버깅

상관 관계 없는 일반적인 디버깅 세션:

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

### 1.2 상관 관계 있는 디버깅

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

### 1.3 상관 관계 식별자(Correlation Identifiers)

신호 상관 관계의 기반은 모든 텔레메트리에 걸친 공유 식별자이다:

| 식별자 | 범위 | 전파 |
|--------|------|------|
| **Trace ID** | 모든 서비스에 걸친 단일 요청 | W3C Trace Context 헤더 |
| **Span ID** | 서비스 내 단일 작업 | 트레이스 컨텍스트의 일부 |
| **Request ID** | 애플리케이션 수준 요청 추적 | 커스텀 헤더 (X-Request-ID) |
| **Session ID** | 여러 요청에 걸친 사용자 세션 | 쿠키 또는 토큰 |
| **Deployment ID** | 텔레메트리를 생성한 릴리스 버전 | 리소스 속성 |

---

## 2. 트레이스-로그 연결(Trace-to-Log Linking)

### 2.1 로그에 트레이스 컨텍스트 주입

가장 영향력 있는 상관 관계 기법은 모든 로그 항목에 trace_id와 span_id를 추가하는 것이다:

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

**결과 로그 항목:**

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

### 2.2 로그 프레임워크 통합(Log Framework Integration)

| 프레임워크 | OTel 통합 |
|-----------|----------|
| **Python structlog** | 커스텀 프로세서 (위에 표시) |
| **Python logging** | `opentelemetry-instrumentation-logging` 자동 주입 |
| **Java Log4j2** | `opentelemetry-log4j-context-data-2.17-autoconfigure` |
| **Java Logback** | `opentelemetry-logback-mdc-1.0` |
| **Go slog** | `trace.SpanFromContext(ctx)`를 통한 수동 주입 |
| **Node.js Pino** | `@opentelemetry/instrumentation-pino` |

### 2.3 트레이스 연결을 위한 Loki 레이블 설정

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

### 2.4 Grafana 트레이스-로그 구성

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

## 3. Exemplar

### 3.1 Exemplar란

Exemplar는 집계된 메트릭 데이터 포인트에서 특정 트레이스 ID로의 참조이다. "이 메트릭 값에 어떤 특정 요청이 기여했는가?"에 답한다.

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

### 3.2 Go에서 Exemplar 구현

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

### 3.3 Python에서 Exemplar 구현

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

### 3.4 Prometheus Exemplar 저장소

Exemplar는 Prometheus에서 `--enable-feature=exemplar-storage` 플래그로 구성해야 한다:

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

### 3.5 Grafana에서 Exemplar 쿼리

Grafana에서 시계열 패널에 exemplar를 활성화한다:

```
Panel settings:
  Data source: Prometheus
  Query: histogram_quantile(0.99, sum by (le) (rate(http_request_duration_seconds_bucket[5m])))

  Options:
    ☑ Exemplars  (toggle ON)

  Exemplar data source: Tempo (or Jaeger)
  URL label: trace_id
```

Exemplar가 활성화되면 시계열 그래프에 클릭 가능한 점이 나타난다. 각 점은 특정 요청을 나타낸다. 클릭하면 해당 트레이스가 열린다.

---

## 4. 메트릭-트레이스 상관 관계(Metrics-to-Traces Correlation)

### 4.1 메트릭 스파이크에서 근본 원인까지

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

### 4.2 트레이스-메트릭 탐색(Trace-to-Metrics Navigation)

역방향도 유용하다 -- 트레이스에서 관련 메트릭으로 탐색:

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

## 5. 로그-트레이스 상관 관계(Log-to-Trace Correlation)

### 5.1 로그 항목에서 트레이스로

로그 항목을 조사할 때, trace_id 필드를 통해 전체 분산 트레이스로 원클릭 탐색이 가능하다:

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

### 5.2 Grafana Loki의 파생 필드(Derived Fields)

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

## 6. 통합 텔레메트리 아키텍처(Unified Telemetry Architecture)

### 6.1 상관 관계 스택

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

### 6.2 OTel Collector 커넥터

커넥터(connector)는 기존 텔레메트리에서 새로운 텔레메트리를 생성하여 교차 신호 상관 관계를 가능하게 한다:

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

### 6.3 스팬 메트릭: 트레이스에서 RED 메트릭 생성

`spanmetrics` 커넥터는 트레이스 스팬에서 자동으로 요청 속도, 오류율, 지속 시간 메트릭을 생성한다:

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

이는 기본 RED 신호에 대한 메트릭을 수동으로 계측할 필요를 없앤다 -- 트레이스에서 직접 파생된다.

---

## 7. 서비스 의존성 매핑(Service Dependency Mapping)

### 7.1 트레이스에서 서비스 그래프 생성

`servicegraph` 커넥터는 트레이스 데이터에서 실시간 의존성 맵을 구축한다:

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

### 7.2 영향 분석을 위한 서비스 그래프 활용

인시던트 중 서비스 그래프는 다음에 답한다:

1. **어떤 서비스가 영향을 받는가?** 장애 서비스에서 의존성 화살표를 따라간다.
2. **폭발 반경은 얼마인가?** 다운스트림 서비스와 요청 속도를 파악한다.
3. **병목은 어디인가?** 지연 시간이 가장 많이 증가한 서비스를 찾는다.

---

## 8. 실전 인시던트 워크쓰루(Incident Walkthrough)

### 8.1 시나리오 설정

```
14:00 UTC - Alert fires: "Payment service availability SLO burn rate critical (14.4x)"
```

### 8.2 단계별 조사

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

## 9. 상관 관계 모범 사례

### 9.1 구현 체크리스트

| 실천 사항 | 우선순위 | 노력 |
|----------|---------|------|
| 모든 로그에 trace_id 추가 | P0 | 낮음 (일회성 설정) |
| Grafana 트레이스-로그 연결 구성 | P0 | 낮음 |
| 주요 히스토그램에 exemplar 활성화 | P1 | 중간 |
| spanmetrics 커넥터 배포 | P1 | 중간 |
| 트레이스-메트릭 링크 구성 | P2 | 낮음 |
| servicegraph 커넥터 배포 | P2 | 중간 |
| 상관된 SLO 대시보드 구축 | P1 | 중간 |

### 9.2 흔한 함정

| 함정 | 영향 | 해결책 |
|------|------|--------|
| 로그에 trace_id 누락 | 로그에서 트레이스로 탐색 불가 | OTel 로그 브릿지 또는 structlog 프로세서 사용 |
| 다른 트레이스 ID 형식 | 상관 관계 실패 (hex vs decimal) | 32자 소문자 hex로 표준화 |
| 로그 타임스탬프 동기화 안됨 | 시간 기반 상관 관계 실패 | 모든 노드에서 NTP 사용, UTC로 로깅 |
| Exemplar 저장 안됨 | 메트릭-트레이스 링크 불가 | Prometheus에서 `--enable-feature=exemplar-storage` 활성화 |
| 파생 필드가 너무 많음 | Grafana UI가 어수선해짐 | trace_id와 span_id만 연결 |

---

## 10. 다음 단계

- [22_Advanced_Metrics_Architecture.md](./22_Advanced_Metrics_Architecture.md) -- 페더레이션, Thanos, Mimir로 Prometheus 확장
- [23_OpenTelemetry_Pipelines.md](./23_OpenTelemetry_Pipelines.md) -- 프로덕션급 OTel Collector 파이프라인 설계

---

## 연습 문제

### 연습 문제 1: 트레이스-로그 연결 구현

Python Flask 애플리케이션에서 `logging` 모듈과 OpenTelemetry를 사용하고 있다. 현재 로그에 트레이스 ID가 포함되어 있지 않다. 다음을 수행하는 코드를 작성하라:

1. trace_id와 span_id를 주입하는 커스텀 로깅 포매터 생성
2. 이 포매터를 사용하도록 루트 로거 구성
3. 트레이싱된 요청 내에서 발생한 로그 항목에 올바른 trace_id가 포함되어 있는지 시연

<details>
<summary>정답 보기</summary>

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

### 연습 문제 2: Exemplar 분석

다음 exemplar가 포함된 Prometheus 메트릭이 주어진다:

```
http_request_duration_seconds_bucket{le="0.1",service="api"} 9500 # {trace_id="aaa"} 0.08
http_request_duration_seconds_bucket{le="0.5",service="api"} 9900 # {trace_id="bbb"} 0.35
http_request_duration_seconds_bucket{le="1.0",service="api"} 9980 # {trace_id="ccc"} 0.72
http_request_duration_seconds_bucket{le="5.0",service="api"} 9998 # {trace_id="ddd"} 3.10
http_request_duration_seconds_bucket{le="+Inf",service="api"} 10000 # {trace_id="eee"} 12.5
http_request_duration_seconds_count{service="api"} 10000
http_request_duration_seconds_sum{service="api"} 5200
```

답하라: (a) 0.5초와 1.0초 사이의 요청은 몇 개인가? (b) p95 지연 시간(근사치)은 얼마인가? (c) 테일 지연 시간을 이해하기 위해 어떤 trace_id를 먼저 조사해야 하는가? (d) 100ms 이내에 완료된 요청의 비율은?

<details>
<summary>정답 보기</summary>

**(a) 0.5초와 1.0초 사이의 요청:**
```
bucket[le="1.0"] - bucket[le="0.5"] = 9980 - 9900 = 80 requests
```

**(b) 근사 p95 지연 시간:**
```
p95 means the value at the 95th percentile = 0.95 * 10000 = 9500th request.
bucket[le="0.1"] = 9500, so the 9500th request falls exactly at the 0.1s boundary.
p95 ≈ 0.1s (100ms).

Note: This is approximate. Histogram interpolation would place it at:
p95 = 0.1 * (9500 - 0) / (9500 - 0) = 0.1s
```

**(c) 테일 지연 시간 조사를 위한 trace_id:**
```
trace_id="eee" with value 12.5s -- this is the most extreme outlier.
It is in the +Inf bucket (above 5s), which is the longest-running request.
Only 2 requests (10000 - 9998) took longer than 5s.

However, trace_id="ddd" at 3.1s is also worth investigating as it represents
the 99.98th percentile range. Both traces should be reviewed.
```

**(d) 100ms 이내에 완료된 요청 비율:**
```
bucket[le="0.1"] / count = 9500 / 10000 = 95%
```

</details>

### 연습 문제 3: 통합 텔레메트리 설계

3개 서비스 시스템(API Gateway, Order Service, Inventory Service)을 위한 완전한 통합 텔레메트리 아키텍처를 설계하라. 다음을 지정하라:

1. 각 서비스에 필요한 OTel 계측
2. OTel Collector 파이프라인 설정 (receiver, processor, connector, exporter)
3. 교차 신호 상관 관계를 위한 Grafana 데이터 소스 설정
4. 세 가지 신호를 모두 함께 사용하는 방법을 보여주는 샘플 디버깅 워크플로우

<details>
<summary>정답 보기</summary>

**1. 서비스별 계측:**

| 서비스 | 자동 계측 | 수동 스팬 | 커스텀 메트릭 | 구조화된 로그 |
|--------|---------|---------|------------|-----------|
| API Gateway | HTTP (Flask/FastAPI), Redis | `authenticate_request`, `rate_limit_check` | `gateway_requests_total`, `gateway_auth_failures_total` | 모든 로그에 trace_id, user_id |
| Order Service | HTTP, PostgreSQL, Kafka producer | `create_order`, `validate_inventory`, `calculate_total` | `orders_created_total`, `order_value_dollars` | 모든 로그에 trace_id, order_id |
| Inventory Service | HTTP, PostgreSQL, Kafka consumer | `reserve_items`, `check_stock`, `update_inventory` | `inventory_reservations_total`, `stock_level` (gauge) | 모든 로그에 trace_id, item_id |

**2. OTel Collector 설정:**

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

**3. Grafana 설정:**
- Tempo 데이터 소스: tracesToLogs → Loki (trace_id로 필터링), tracesToMetrics → Prometheus
- Loki 데이터 소스: derivedFields → trace_id 링크를 Tempo로
- Prometheus 데이터 소스: exemplar 활성화, Tempo에 연결

**4. 디버깅 워크플로우:**

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

## 참고 자료

- [Grafana Tempo -- Traces to Logs](https://grafana.com/docs/tempo/latest/metrics-generator/traces-to-logs/)
- [OpenTelemetry Collector Connectors](https://opentelemetry.io/docs/collector/connectors/)
- [Prometheus Exemplars](https://prometheus.io/docs/prometheus/latest/feature_flags/#exemplars-storage)
- [Grafana Unified Alerting](https://grafana.com/docs/grafana/latest/alerting/)
- [Correlating Signals in Grafana Cloud (Blog)](https://grafana.com/blog/2022/08/18/how-to-correlate-metrics-logs-and-traces-in-grafana/)
