# 분산 트레이싱

**이전**: [로깅 인프라](./11_Logging_Infrastructure.md) | **다음**: [배포 전략](./13_Deployment_Strategies.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 마이크로서비스 아키텍처 디버깅에 분산 트레이싱이 필요한 이유를 설명할 수 있습니다
2. 분산 트레이싱의 핵심 개념인 trace, span, context propagation, baggage를 정의할 수 있습니다
3. OpenTelemetry SDK와 자동 계측(auto-instrumentation)을 사용하여 애플리케이션을 계측할 수 있습니다
4. 요청 흐름 시각화를 위한 트레이싱 백엔드로 Jaeger를 배포하고 쿼리할 수 있습니다
5. 관측 가능성과 오버헤드 간의 균형을 위한 트레이스 샘플링 전략을 구성할 수 있습니다
6. 로그, 메트릭, 트레이스를 통합 관측 가능성 파이프라인으로 상관시킬 수 있습니다

---

모놀리식 애플리케이션에서는 스택 트레이스가 요청부터 응답까지의 전체 호출 경로를 보여줍니다. 마이크로서비스 아키텍처에서는 하나의 사용자 요청이 10개 이상의 서비스를 거칠 수 있으며, 하나의 서비스에서 얻은 스택 트레이스는 전체 이야기의 일부만 드러냅니다. 분산 트레이싱은 각 요청에 고유한 trace ID를 할당하고 모든 서비스 경계에 걸쳐 전파하여 시스템을 통과하는 요청의 여정에 대한 완전한 맵을 생성함으로써 이 문제를 해결합니다.

> **비유 -- 택배 추적**: 분산 트레이싱은 택배 회사의 추적 시스템과 같이 작동합니다. 택배를 보내면 택배 회사가 운송장 번호(trace ID)를 부여합니다. 각 중계 허브(서비스)에서 택배가 스캔되고(span 생성) 타임스탬프와 위치가 기록됩니다. 택배가 지연되면 전체 여정을 확인하여 정확히 어느 허브에서 지연이 발생했는지 파악할 수 있습니다. 추적 없이는 택배가 늦다는 것만 알 뿐 어디서 멈췄는지 알 수 없습니다.

## 1. 분산 트레이싱이 필요한 이유

### 1.1 마이크로서비스의 디버깅 문제

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

### 1.2 메트릭과 로그가 알려줄 수 없는 것을 트레이싱이 알려줌

| 질문 | 메트릭 | 로그 | 트레이스 |
|------|--------|------|----------|
| "시스템이 느린가?" | 예 (p95 지연 시간) | 아니오 | 아니오 |
| "어느 서비스가 느린가?" | 부분적 (서비스별 지연 시간) | 아니오 | **예 (span 분석)** |
| "왜 느린가?" | 아니오 | 부분적 (오류 메시지) | **예 (호출 그래프 + 타이밍)** |
| "어느 다운스트림 호출이 병목인가?" | 아니오 | 아니오 | **예 (자식 span 기간)** |
| "이 요청이 시스템을 어떻게 통과하는가?" | 아니오 | 아니오 | **예 (트레이스 시각화)** |

---

## 2. 핵심 개념

### 2.1 Trace와 Span

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

**핵심 용어:**

| 개념 | 정의 |
|------|------|
| **Trace** | 모든 서비스를 통과하는 요청의 전체 여정. `trace_id`로 식별됩니다. |
| **Span** | 트레이스 내의 단일 작업 단위(예: HTTP 호출, DB 쿼리). 시작 시간, 기간, 메타데이터를 가집니다. |
| **Root span** | 트레이스의 첫 번째 span(보통 진입점 서비스에서 생성됩니다). |
| **Child span** | 부모 span에 인과적으로 연결된 span입니다. |
| **Span context** | 서비스 경계에 걸쳐 전파되는 데이터: `trace_id`, `span_id`, `trace_flags`. |
| **Baggage** | 트레이스의 모든 span에 걸쳐 전파되는 사용자 정의 키-값 쌍(예: `user_id`, `tenant_id`). |

### 2.2 Span 속성

각 span은 구조화된 메타데이터를 전달합니다:

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

Context propagation은 서비스 경계에 걸쳐 span을 연결하는 메커니즘입니다. 트레이스 컨텍스트는 나가는 요청(HTTP 헤더, 메시지 헤더)에 주입되고 수신 서비스에서 추출됩니다.

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

**W3C Trace Context 헤더 형식:**
```
traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01
             ──  ────────────────────────────────  ────────────────  ──
             ver           trace-id                   parent-id     flags
                                                                    (01 = sampled)
```

---

## 3. OpenTelemetry

### 3.1 OpenTelemetry란

OpenTelemetry(OTel)는 텔레메트리 데이터(트레이스, 메트릭, 로그)의 생성, 수집, 내보내기를 위한 API, SDK, 도구를 제공하는 벤더 중립적인 오픈소스 관측 가능성 프레임워크입니다.

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

### 3.2 Python에서의 계측

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

### 3.3 Go에서의 계측

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

### 3.4 자동 계측(Zero-Code)

OpenTelemetry는 많은 언어에 대해 코드 변경 없이 공통 라이브러리를 계측하는 자동 계측을 지원합니다:

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

### 4.1 Collector 구성

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

### 5.1 Jaeger 아키텍처

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

### 5.2 Jaeger 배포 (개발용 All-in-One)

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

### 5.3 Jaeger UI에서 트레이스 쿼리

Jaeger UI는 트레이스를 찾기 위한 여러 방법을 제공합니다:

| 쿼리 방법 | 사용 사례 | 예시 |
|-----------|----------|------|
| **Service + Operation** | 특정 엔드포인트의 트레이스 찾기 | Service: `order-service`, Operation: `GET /api/orders` |
| **Tags** | span 속성으로 필터링 | `http.status_code=500`, `user.id=u-123` |
| **Duration** | 느린 트레이스 찾기 | Min: `1s`, Max: `10s` |
| **Trace ID** | 특정 트레이스 조회 | `4bf92f3577b34da6a3ce929d0e0e4736` |
| **Time range** | 검색 시간 범위 축소 | 최근 1시간, 최근 24시간 |

### 5.4 Jaeger API 쿼리

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

## 6. 트레이스 샘플링

### 6.1 샘플링이 필요한 이유

대규모 환경에서 모든 요청을 트레이싱하는 것은 비현실적입니다:

| 지표 | 샘플링 없음 | 10% 샘플링 |
|------|------------|-----------|
| **트레이스/초** | 10,000 | 1,000 |
| **저장소/일** | ~100 GB | ~10 GB |
| **네트워크 오버헤드** | 높음 | 낮음 |
| **비용** | $$$$ | $ |

### 6.2 샘플링 전략

```
┌─────────────────────────────────────────────────────────────┐
│                   Sampling Strategies                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Head-Based Sampling (트레이스 시작 시 결정)                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  • Probabilistic: 트레이스의 10%를 무작위 샘플링      │    │
│  │  • Rate-limiting: 초당 최대 N개 트레이스              │    │
│  │  • 장점: 단순하고 오버헤드가 낮음                      │    │
│  │  • 단점: 흥미로운(오류/느린) 트레이스를 놓칠 수 있음    │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  Tail-Based Sampling (트레이스 완료 후 결정)                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  • 모든 오류 트레이스 유지                             │    │
│  │  • 지연 시간 임계값 초과 트레이스 유지                   │    │
│  │  • 특정 속성과 일치하는 트레이스 유지                    │    │
│  │  • 장점: 중요한 트레이스를 절대 놓치지 않음              │    │
│  │  • 단점: 완전한 트레이스 버퍼링 필요                    │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  Hybrid: Head-based 10% + 오류/느린 트레이스 tail-based       │
└─────────────────────────────────────────────────────────────┘
```

### 6.3 샘플링 구성

**Head-based 샘플링 (SDK에서):**

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

**Tail-based 샘플링 (Collector에서):**

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

## 7. 로그, 메트릭, 트레이스 상관

### 7.1 상관 모델

```
┌─────────────────────────────────────────────────────────────┐
│              Unified Observability Correlation                │
│                                                              │
│  Metrics (Prometheus)                                        │
│  ┌──────────────────────────────────────────────────┐       │
│  │  http_request_duration_seconds{service="orders"} │       │
│  │  → "p99 지연 시간이 14:23에 급등"                 │       │
│  └──────────────────┬───────────────────────────────┘       │
│                     │ "14:23의 트레이스를 보여줘"             │
│                     ▼                                        │
│  Traces (Jaeger)                                             │
│  ┌──────────────────────────────────────────────────┐       │
│  │  trace_id: abc123                                │       │
│  │  → "Inventory 서비스 DB 쿼리가 2.3초 소요"        │       │
│  └──────────────────┬───────────────────────────────┘       │
│                     │ "trace abc123의 로그를 보여줘"          │
│                     ▼                                        │
│  Logs (Loki / ELK)                                           │
│  ┌──────────────────────────────────────────────────┐       │
│  │  {trace_id="abc123"} "Connection pool exhausted, │       │
│  │  waited 2100ms for available connection"          │       │
│  └──────────────────────────────────────────────────┘       │
│                                                              │
│  상관 키: trace_id가 세 가지 신호를 모두 연결                  │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 로그에 트레이스 컨텍스트 주입

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

### 7.3 Exemplar: 메트릭과 트레이스 연결

Exemplar는 특정 메트릭 데이터 포인트에 trace ID를 첨부하여 메트릭에서 해당 메트릭을 생성한 트레이스로 직접 탐색할 수 있게 합니다:

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

Grafana에서 Prometheus 그래프의 exemplar 데이터 포인트를 클릭하면 Jaeger 또는 Tempo의 해당 트레이스로 직접 이동합니다.

### 7.4 Grafana 통합 뷰

Grafana 데이터 소스 상관 관계를 구성합니다:

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

## 8. 모범 사례

### 8.1 계측 가이드라인

| 실천 사항 | 설명 |
|-----------|------|
| **span 이름은 매개변수가 아닌 작업으로 지정** | `GET /api/users/{id}` (좋음), `GET /api/users/12345` (나쁨 -- 높은 카디널리티 방지) |
| **의미 있는 속성 설정** | 디버깅을 위해 `user.id`, `order.id`, `db.statement` 포함 |
| **span에 오류 기록** | `span.record_exception(error)`와 `span.set_status(ERROR)` 사용 |
| **span 수를 적정하게 유지** | 트레이스당 10-50개의 span이 일반적. 모든 함수 호출에 span을 만들지 않기 |
| **시맨틱 규칙 사용** | 속성 이름에 OpenTelemetry 시맨틱 규칙을 따르기 |
| **비동기 작업에 컨텍스트 전파** | 백그라운드 작업, 메시지 소비자, 스레드 풀에 트레이스 컨텍스트 전달 |

### 8.2 Span 명명 모범 사례

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

## 9. 다음 단계

- [13_Deployment_Strategies.md](./13_Deployment_Strategies.md) - 안전한 배포 패턴
- [10_Monitoring_and_Alerting.md](./10_Monitoring_and_Alerting.md) - Prometheus를 이용한 메트릭과 알림
- [11_Logging_Infrastructure.md](./11_Logging_Infrastructure.md) - 중앙 집중식 로깅

---

## 연습 문제

### 연습 문제 1: 트레이스 분석

다음 트레이스가 주어졌을 때, 성능 병목을 식별하고 해결 방안을 제안하십시오:

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
<summary>정답 보기</summary>

**병목**: `Inventory Service -> External API: supplier_stock_check` span이 1,985ms(205ms ~ 2,190ms)로, 전체 트레이스 기간의 81%를 차지합니다.

**근본 원인 분석**:
1. Redis 캐시 미스로 인해 장바구니를 PostgreSQL에서 가져와야 했지만 이는 175ms에 불과하며 주요 문제가 아닙니다.
2. 재고 확인 자체(DB 쿼리)는 10ms로 빠르지만, 외부 공급업체 재고 확인 API 호출이 1,985ms로 매우 느립니다.
3. 결제 처리(235ms)와 Stripe 호출(225ms)은 합리적인 수준입니다.

**권장 해결 방안**:
1. **공급업체 재고 응답 캐싱**: 트레이스에서 공급업체 호출 후 `Cache: SET`이 표시되어 캐싱이 이미 계획되어 있음을 알 수 있습니다. 캐시 TTL이 적절한지 확인합니다(예: 재고 수준의 경우 5분). 이후 요청은 캐시를 활용합니다.
2. **타임아웃 추가**: 외부 공급업체 API 호출에 500ms 타임아웃과 서킷 브레이커를 설정합니다. 공급업체가 느리면 캐시된 재고 수량으로 폴백합니다.
3. **비동기화**: 체크아웃에 실시간 공급업체 재고가 필요하지 않다면 공급업체 재고를 비동기적으로 가져오고 체크아웃 플로우에는 로컬 캐시된 재고를 사용합니다.
4. **병렬화**: 장바구니 조회와 재고 확인은 독립적입니다. 동시에 실행하면 ~180ms를 절약할 수 있습니다: `total = max(cart_time, inventory_time) + payment_time` (기존: `cart_time + inventory_time + payment_time`).

</details>

### 연습 문제 2: OpenTelemetry 계측

두 개의 다운스트림 서비스를 순차적으로 호출하는 함수를 계측하는 Python 코드를 작성하십시오. 적절한 span 생성, 속성 설정, 오류 처리, 컨텍스트 전파를 포함해야 합니다.

<details>
<summary>정답 보기</summary>

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

**정답의 핵심 포인트:**
- 각 다운스트림 호출마다 의미 있는 속성을 가진 별도의 span이 있습니다.
- `record_exception()`은 span에 전체 스택 트레이스를 캡처합니다.
- `set_status(ERROR)`는 Jaeger UI에서 span을 실패로 표시합니다.
- `add_event()`는 span 내에서 주목할 만한 이벤트를 기록합니다.
- 다운스트림 서비스로의 컨텍스트 전파는 `RequestsInstrumentor`에 의해 자동으로 처리됩니다.

</details>

### 연습 문제 3: 샘플링 전략 설계

한 회사에 트레이싱 요구사항이 다른 세 가지 환경이 있습니다:

| 환경 | 트래픽 | 예산 | 요구사항 |
|------|--------|------|----------|
| 개발 | 100 req/s | 낮음 | 디버깅을 위한 완전한 가시성 |
| 스테이징 | 1,000 req/s | 중간 | 성능 회귀 감지 |
| 프로덕션 | 50,000 req/s | 제한적 | 오류를 절대 놓치지 않으면서 비용 관리 |

각 환경에 대한 샘플링 전략을 설계하십시오.

<details>
<summary>정답 보기</summary>

**개발 (100 req/s):**
- **전략**: 100% 샘플링 (모든 것을 샘플링)
- **근거**: 낮은 트래픽은 저장 비용이 무시할 수 있는 수준입니다. 완전한 가시성은 개발자가 개발 중 문제를 디버깅하는 데 도움이 됩니다. 100 req/s에서는 모든 트레이스를 저장하더라도 일일 저장량은 약 1 GB에 불과합니다.
- **구성**: `sampler = ALWAYS_ON`

**스테이징 (1,000 req/s):**
- **전략**: Head-based 50% + 오류 및 느린 트레이스에 대한 tail-based
- **근거**: 스테이징은 성능 테스트에 사용되므로 회귀를 감지하기에 충분한 샘플이 필요합니다. 50%는 통계적으로 유의미한 데이터를 제공합니다. Tail-based 샘플링은 모든 오류가 캡처되도록 보장합니다.
- **구성**:
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

**프로덕션 (50,000 req/s):**
- **전략**: Head-based 1% + 오류(100%) 및 느린 트레이스(> 2s)에 대한 tail-based
- **근거**: 50K req/s에서 1% 샘플링은 초당 500개의 트레이스를 제공하며, 이는 추세 분석에 충분합니다. Tail-based 샘플링은 head-based 결정과 관계없이 모든 오류와 느린 트레이스가 캡처되도록 보장합니다. 이를 통해 일일 저장량이 ~500 GB/일 대신 ~5 GB/일로 유지됩니다.
- **구성**:
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

**프로덕션 규모의 저장소 추정:**
- 전체 샘플링: 50K 트레이스/초 * 86400초/일 * ~10KB/트레이스 = ~43 TB/일
- 1% head + tail: ~500 트레이스/초 + ~100 오류 트레이스/초 = ~5 GB/일

</details>

### 연습 문제 4: 상관 파이프라인

세 개 서비스 아키텍처(API Gateway, Order Service, Payment Service)에 대한 완전한 관측 가능성 상관 설정을 설계하십시오. 다음을 설명하십시오:

1. 서비스 간 트레이스 컨텍스트가 어떻게 흐르는지
2. 로그에 trace ID가 어떻게 포함되는지
3. Prometheus 알림에서 관련 트레이스로 어떻게 탐색하는지

<details>
<summary>정답 보기</summary>

**1. 트레이스 컨텍스트 흐름:**

```
Client → API Gateway → Order Service → Payment Service
         │                │                │
         │ root span      │ traceparent    │ traceparent
         │ 생성,          │ 헤더에서       │ 헤더에서
         │ traceparent    │ 추출,          │ 추출,
         │ 헤더 주입      │ child span     │ child span
         │                │ 생성,          │ 생성
         │                │ traceparent    │
         │                │ 나가는 요청에   │
         │                │ 주입           │

HTTP Headers at each hop:
traceparent: 00-{trace_id}-{span_id}-01
```

세 서비스 모두 HTTP 클라이언트와 서버에 대한 자동 계측이 포함된 OpenTelemetry SDK를 사용합니다. `traceparent` 헤더가 자동으로 주입/추출됩니다.

**2. 로그에 trace ID 포함:**

각 서비스는 트레이스 컨텍스트 프로세서를 사용하여 구조화된 로깅을 구성합니다:
```python
# In each service's logging setup
log.info("order_created",
         order_id="ord-123",
         trace_id="4bf92f...",   # Auto-injected from current span
         span_id="00f067...")
```

Loki에서 이 로그를 trace ID로 쿼리할 수 있습니다:
```logql
{service="order-service"} | json | trace_id="4bf92f3577b34da6a3ce929d0e0e4736"
```

**3. Prometheus 알림에서 트레이스 탐색:**

흐름: Prometheus Alert -> Grafana Dashboard -> Exemplar -> Jaeger Trace -> Loki Logs

단계별:
1. Prometheus가 14:23에 `order-service`에 대해 `HighLatency` 알림을 발생시킵니다.
2. 엔지니어가 Grafana 대시보드를 열고 `http_request_duration_seconds` 패널에서 지연 시간 급등을 확인합니다.
3. 패널에 exemplar 점이 그래프에 표시됩니다. exemplar를 클릭하면 `trace_id: 4bf92f...`와 "View Trace" 링크가 표시됩니다.
4. "View Trace"를 클릭하면 Jaeger/Tempo에서 트레이스가 열리고, 어떤 다운스트림 호출이 지연을 유발했는지 보여줍니다.
5. 트레이스 뷰에서 "View Logs"를 클릭하면 해당 트레이스의 모든 서비스에서 나온 모든 로그가 `trace_id`로 필터링되어 표시됩니다.

이것은 다음에 의해 가능합니다:
- 메트릭을 트레이스에 연결하는 Prometheus exemplar
- Grafana 데이터 소스 상관 관계 (Prometheus -> Tempo -> Loki)
- 모든 로그에서 일관된 `trace_id` 필드

</details>

---

## 참고 자료

- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [OpenTelemetry Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)
- [W3C Trace Context Specification](https://www.w3.org/TR/trace-context/)
- [Grafana Tempo Documentation](https://grafana.com/docs/tempo/latest/)
- [Google Dapper Paper](https://research.google/pubs/pub36356/)
