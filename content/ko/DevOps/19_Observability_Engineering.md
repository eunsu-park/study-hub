# 19. 관측 가능성 엔지니어링(Observability Engineering)

**이전**: [SRE 실무](./18_SRE_Practices.md) | **다음**: [SLO 엔지니어링](./20_SLO_Engineering.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있다:

1. 모니터링과 관측 가능성(observability)의 차이를 구별하고, 현대 분산 시스템에서 관측 가능성이 중요한 이유를 설명한다
2. 관측 가능성의 세 가지 기둥(메트릭, 로그, 트레이스)과 상호 보완적 역할을 설명한다
3. OpenTelemetry 프로젝트 아키텍처와 업계 표준 텔레메트리 프레임워크로서의 역할을 설명한다
4. 신호 커버리지와 성능 오버헤드의 균형을 맞추는 계측(instrumentation) 전략을 설계한다
5. 카디널리티(cardinality) 개념을 적용하여 텔레메트리 데이터 차원에 대한 정보에 기반한 결정을 내린다
6. 프로덕션에서 새로운 장애를 디버깅하는 데 도움이 되는 구조화된 텔레메트리를 구현한다

---

모니터링은 *언제* 고장났는지 알려준다. 관측 가능성은 *왜* 고장났는지 알려준다. 모놀리식 세계에서는 모니터링만으로 충분했다 -- 몇 대의 서버와 예측 가능한 장애 모드만 있었기 때문이다. 분산 마이크로서비스 세계에서는 수십 개의 서비스 간 복잡한 상호작용에서 장애가 발생하며, 모든 장애 모드를 사전에 예측할 수 없다. 관측 가능성은 시스템의 외부 출력(텔레메트리 데이터)을 검사하여 내부 상태를 이해할 수 있게 하는 시스템의 속성이다.

> **비유 -- 의료 진단**: 모니터링은 환자의 심박수가 120 BPM을 초과할 때 울리는 병원 알람과 같다. *무언가 잘못되었다*는 것을 알려준다. 관측 가능성은 의사가 환자의 상태에 대해 임의의 질문을 하고 이전에 본 적 없는 문제를 진단할 수 있게 하는 전체 진단 도구(ECG 트레이스, 혈액 검사, 영상) 세트이다. 가능한 모든 질병에 대해 알람을 미리 구성할 수는 없지만, 어떤 증상이든 조사할 수 있을 만큼 충분한 진단 데이터를 수집할 수 있다.

## 1. 모니터링 vs 관측 가능성

### 1.1 모니터링에서 관측 가능성으로의 전환

| 측면 | 모니터링 | 관측 가능성 |
|------|---------|-----------|
| **질문** | "시스템이 작동하고 있는가?" | "시스템이 왜 이렇게 동작하는가?" |
| **접근법** | 사전 정의된 검사 및 임계값 | 텔레메트리에서 임의의 질문을 탐색 |
| **장애 모델** | 알려진 장애 모드(예측됨) | 미지의 미지(unknown-unknowns, 발현적) |
| **데이터 모델** | 집계된 메트릭, 정적 대시보드 | 높은 카디널리티, 높은 차원의 데이터 |
| **디버깅** | 대시보드 확인 → 알림 찾기 → 런북 따르기 | 텔레메트리 탐색 → 가설 수립 → 드릴다운 |
| **규모** | 모놀리스 및 단순 아키텍처에 적합 | 분산 시스템에 필수적 |

### 1.2 모니터링만으로는 불충분한 이유

분산 시스템에서 장애는 종종 상호작용에서 발생한다:

- 서비스 A가 느린 이유는 서비스 B의 커넥션 풀(connection pool)이 고갈되었기 때문
- 서비스 B의 풀이 고갈된 이유는 서비스 C가 스키마 변경을 배포했기 때문
- 서비스 C의 스키마 변경은 테넌트 X의 데이터와 관련된 쿼리에서만 느림

어떤 대시보드도 이 체인을 사전에 시각화할 수 없다. 다음 능력이 필요하다:
1. 증상에서 시작 (서비스 A의 높은 지연 시간)
2. 느린 요청을 보여주는 특정 트레이스로 드릴다운
3. 서비스 B의 풀 고갈을 보여주는 로그와 상관 관계 분석
4. 특정 테넌트에 대한 쿼리 시간 증가를 보여주는 서비스 C 메트릭으로 이동

### 1.3 관측 가능성 성숙도 모델(Observability Maturity Model)

| 수준 | 역량 | 특성 |
|------|------|------|
| **L0: 반응적(Reactive)** | 기본 헬스 체크 | 업/다운 알림, 구조화된 텔레메트리 없음 |
| **L1: 정보 기반(Informed)** | 메트릭 + 대시보드 | 골든 시그널 모니터링, 정적 대시보드 |
| **L2: 조사적(Investigative)** | 상관된 신호 | 메트릭, 로그, 트레이스가 연결됨 |
| **L3: 선제적(Proactive)** | 이상 탐지 | ML 기반 알림, SLO 기반 의사결정 |
| **L4: 예측적(Predictive)** | 용량 예측 | 추세 분석, 자동화된 스케일링 권장 |

---

## 2. 관측 가능성의 세 가지 기둥(Three Pillars)

### 2.1 메트릭(Metrics)

메트릭은 정기적인 간격으로 수집되는 수치 측정값이다. 가장 효율적인 텔레메트리 유형으로, 높은 압축률과 빠른 쿼리 속도를 제공한다.

**장점:**
- 낮은 저장 비용 (트래픽에 관계없이 시계열당 고정)
- 차원 간 빠른 집계
- 알림에 이상적 (집계된 값에 대한 임계값)
- 대시보드 및 추세 분석에 적합

**단점:**
- 집계 시 개별 이벤트 세부 정보 손실
- "어떤 특정 요청이 느렸는가?"에 답할 수 없음
- 레이블 차원이 너무 많으면 카디널리티 폭발(cardinality explosion)

**주요 메트릭 유형 (Prometheus 모델):**

| 유형 | 동작 | 예시 |
|------|------|------|
| **Counter** | 단조 증가 | `http_requests_total` |
| **Gauge** | 올라가고 내려감 | `temperature_celsius` |
| **Histogram** | 버킷에 값을 집계 | `request_duration_seconds_bucket` |
| **Summary** | 클라이언트 측 분위수 계산 | `request_duration_seconds{quantile="0.99"}` |

### 2.2 로그(Logs)

로그는 타임스탬프가 찍힌 이산적 이벤트 기록이다. 가장 풍부한 컨텍스트를 제공하지만 저장 및 쿼리 비용이 가장 높다.

**장점:**
- 이벤트당 풍부한 컨텍스트 (요청 ID, 사용자 ID, 스택 트레이스)
- 사람이 읽을 수 있는 디버깅 정보
- 임의의 데이터 구조를 기록할 수 있음

**단점:**
- 높은 볼륨에서 비싼 저장소 (시계열당 바이트 vs 이벤트당 바이트)
- 인덱싱 없이는 쿼리가 느림
- 수치적으로 집계하기 어려움
- 비구조화된 로그는 대규모에서 거의 쓸모없음

**구조화된 로깅(structured logging) 모범 사례:**

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

### 2.3 트레이스(Traces)

트레이스는 분산 시스템을 통과하는 요청의 경로를 기록한다. 트레이스는 스팬(span)으로 구성되며, 각 스팬은 작업 단위(HTTP 호출, 데이터베이스 쿼리, 메시지 발행)를 나타낸다.

**장점:**
- 서비스 간 전체 요청 생명주기를 보여줌
- 지연 시간 병목 및 의존성 체인을 드러냄
- 분산 장애에 대한 근본 원인 분석(root cause analysis) 가능

**단점:**
- 높은 저장 비용 (요청당 하나의 트레이스는 시간당 수백만 개를 의미할 수 있음)
- 대규모에서는 샘플링이 필요 (모든 요청이 트레이싱되지 않음)
- 복잡한 계측 (모든 서비스가 컨텍스트를 전파해야 함)

**트레이스 구조:**

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

### 2.4 세 가지 기둥 비교

| 차원 | 메트릭 | 로그 | 트레이스 |
|------|--------|------|---------|
| **데이터 유형** | 수치 시계열 | 구조화된 이벤트 기록 | 스팬 트리 |
| **세분성** | 집계 (간격별) | 이벤트별 | 요청별 |
| **저장 비용** | 낮음 | 높음 | 중-높음 |
| **쿼리 속도** | 빠름 | 느림 (인덱스 없이) | 중간 |
| **최적 용도** | 알림, 대시보드 | 디버깅, 감사 | 요청 흐름 분석 |
| **카디널리티 우려** | 레이블 폭발 | 볼륨 폭발 | 샘플링 트레이드오프 |

---

## 3. OpenTelemetry 개요

### 3.1 OpenTelemetry란

OpenTelemetry(OTel)는 텔레메트리 데이터를 생성, 수집, 내보내기 위한 벤더 중립적이고 개방형 표준 프레임워크를 제공하는 CNCF 프로젝트이다. 이전의 OpenTracing과 OpenCensus 프로젝트를 통합했다.

**핵심 구성 요소:**

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

OTLP는 텔레메트리 데이터 전송을 위한 네이티브 프로토콜이다:

| 특성 | OTLP/gRPC | OTLP/HTTP |
|------|-----------|-----------|
| **전송** | gRPC (HTTP/2) | HTTP/1.1 또는 HTTP/2 |
| **인코딩** | Protobuf | Protobuf 또는 JSON |
| **성능** | 높은 처리량, 스트리밍 | 더 단순, 방화벽 친화적 |
| **포트** | 4317 | 4318 |
| **사용 사례** | 서비스-컬렉터 간 | 브라우저, 엣지, 제한된 네트워크 |

### 3.3 언어 지원(Language Support)

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

### 3.4 자동 계측(Auto-Instrumentation)

OpenTelemetry는 일반적인 라이브러리에 대한 자동 계측을 제공하여 수동 작업을 줄인다:

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

자동 계측되는 라이브러리 (Python):
- `requests`, `urllib3`, `httpx` -- 발신 HTTP 호출
- `flask`, `django`, `fastapi` -- 수신 HTTP 요청
- `psycopg2`, `sqlalchemy`, `pymongo` -- 데이터베이스 호출
- `celery`, `redis`, `kafka-python` -- 메시지 큐
- `grpc` -- gRPC 클라이언트 및 서버 호출

---

## 4. 계측 전략(Instrumentation Strategies)

### 4.1 계측 피라미드

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

### 4.2 수동 vs 자동 계측

| 접근법 | 커버리지 | 노력 | 품질 |
|--------|---------|------|------|
| **자동 계측(auto-instrumentation)** | 라이브러리 경계 (HTTP, DB, 큐) | 코드 변경 없음 | 일반적인 스팬 이름, 기본 속성 |
| **수동 스팬(manual spans)** | 비즈니스 로직, 커스텀 작업 | 코드 변경 필요 | 풍부한 컨텍스트, 도메인별 속성 |
| **시맨틱 규약(semantic conventions)** | 표준화된 속성 이름 | OTel 스펙 따르기 | 서비스 간 일관성 |

**모범 사례**: 자동 계측으로 넓은 범위를 시작하고, 중요한 비즈니스 경로에 수동 스팬으로 깊이를 추가한다.

### 4.3 무엇을 계측할 것인가

**항상 계측 (높은 가치, 낮은 노력):**
- HTTP 요청/응답 (자동 계측)
- 데이터베이스 쿼리 (자동 계측)
- 외부 API 호출 (자동 계측)
- 메시지 큐 발행/소비 (자동 계측)
- 인증 및 권한 부여 결정
- 비즈니스에 중요한 작업 (주문 생성, 결제 처리)

**선택적으로 계측 (중간 가치, 중간 노력):**
- 캐시 적중/실패(cache hit/miss)
- 피처 플래그(feature flag) 평가
- 재시도 횟수
- 서킷 브레이커(circuit breaker) 상태 변경
- 백그라운드 작업 실행

**계측을 피해야 하는 것 (낮은 가치, 높은 비용):**
- 타이트한 내부 루프 (배치 내 항목별 처리)
- 사소한 getter/setter
- 코드의 모든 줄 (대신 프로파일링 사용)

### 4.4 스팬 설계 모범 사례

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

## 5. 텔레메트리 데이터 설계

### 5.1 카디널리티와 차원성(Cardinality and Dimensionality)

**카디널리티**는 메트릭에 의해 생성되는 고유한 시계열의 수이다. 높은 카디널리티는 관측 가능성 시스템 장애의 가장 흔한 원인이다.

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

**카디널리티 관리 규칙:**

| 규칙 | 설명 |
|------|------|
| **무한한 값을 메트릭 레이블로 사용하지 않기** | user_id, request_id, email -- 이런 것은 트레이스를 사용한다 |
| **연속 값을 버킷으로 묶기** | 응답 시간 → 히스토그램 버킷, 정확한 값이 아님 |
| **레코딩 규칙(recording rules) 사용** | 높은 카디널리티 쿼리를 사전 집계 |
| **카디널리티 모니터링** | `prometheus_tsdb_head_series` 추적 |
| **카디널리티 제한 설정** | Prometheus 스크레이프 설정에서 `sample_limit` 사용 |

### 5.2 시맨틱 규약(Semantic Conventions)

OpenTelemetry는 일반적인 개념에 대한 표준화된 속성 이름인 시맨틱 규약을 정의한다:

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

### 5.3 컨텍스트 전파(Context Propagation)

컨텍스트 전파는 트레이스 ID와 스팬 ID가 서비스 경계를 넘어 전달되도록 보장한다:

```
Service A                         Service B
┌────────────────────┐            ┌────────────────────┐
│ Span: handle_request│ ─HTTP──→  │ Span: process_order│
│ trace_id: abc123   │  Headers:  │ trace_id: abc123   │
│ span_id: span_001  │  traceparent: │ parent: span_001│
└────────────────────┘  00-abc123-span_001-01          │
                                  └────────────────────┘
```

**W3C Trace Context 헤더 형식:**

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

## 6. 관측 가능성 아키텍처 패턴

### 6.1 에이전트 기반 수집(Agent-Based Collection)

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

### 6.2 사이드카 패턴(Sidecar Pattern)

각 애플리케이션 파드(pod)가 자체 OTel Collector 사이드카를 갖는다. 격리를 제공하지만 리소스 사용이 증가한다:

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

### 6.3 수집 패턴 선택

| 패턴 | 리소스 오버헤드 | 격리 | 복잡성 | 최적 사용 사례 |
|------|--------------|------|--------|-------------|
| **직접 내보내기(direct export)** (앱 → 백엔드) | 최저 | 없음 | 최저 | 소규모 배포 |
| **에이전트(agent)** (DaemonSet) | 중간 | 노드별 | 중간 | 대부분의 Kubernetes 배포 |
| **사이드카(sidecar)** | 최고 | 파드별 | 최고 | 멀티 테넌트, 엄격한 격리 |
| **게이트웨이(gateway)** (중앙 집중 컬렉터) | 낮음 (중앙 집중) | 공유 | 중간 | 대규모, 테일 샘플링 필요 |

---

## 7. 관측 가능성 문화

### 7.1 관측 가능성을 왼쪽으로 이동(Shifting Left on Observability)

관측 가능성은 프로덕션에서 나중에 추가하는 것이 아니라 개발 프로세스의 일부여야 한다:

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

### 7.2 관측 가능성 주도 개발(Observability-Driven Development)

기능 코드 *이전에* 계측을 작성한다:

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

## 8. 비용 관리(Cost Management)

### 8.1 텔레메트리 비용 방정식

```
Monthly Cost = (Ingestion Volume × Ingestion Price)
             + (Storage Volume × Storage Price)
             + (Query Volume × Query Price)
```

| 신호 | 일반적인 볼륨 | 비용 동인 |
|------|-------------|----------|
| **메트릭** | 서비스당 ~10K 시계열 | 카디널리티 (고유 시계열) |
| **로그** | 서비스당 ~1-10 GB/일 | 볼륨 (수집 바이트) |
| **트레이스** | 서비스당 ~100K-1M 스팬/일 | 스팬 수 x 속성 크기 |

### 8.2 비용 최적화 전략

| 전략 | 신호 | 영향 |
|------|------|------|
| **샘플링(sampling)** (헤드 또는 테일) | 트레이스 | 10-100배 비용 절감 |
| **로그 레벨 관리** | 로그 | 2-10배 감소 (프로덕션에서 DEBUG 제거) |
| **메트릭 집계** | 메트릭 | 높은 카디널리티 시리즈 감소 |
| **레코딩 규칙(recording rules)** | 메트릭 | 쿼리 시간 집계 대신 사전 집계 |
| **보존 정책(retention policies)** | 전체 | 상세 데이터 7일, 집계 90일 보관 |
| **데이터 계층화(data tiering)** | 전체 | Hot (SSD) → Warm (HDD) → Cold (객체 스토리지) |
| **속성 필터링** | 트레이스/로그 | 컬렉터에서 비필수 속성 삭제 |

### 8.3 샘플링 전략(Sampling Strategies)

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

## 9. 관측 가능성 안티패턴(Anti-Patterns)

### 9.1 흔한 실수

| 안티패턴 | 문제 | 해결책 |
|---------|------|--------|
| **대시보드 주도 개발** | 모든 것에 대시보드 구축, 디버깅은 못함 | 탐색적 도구에 집중 (트레이스 검색, 로그 상관 관계) |
| **알림 피로(alert fatigue)** | 너무 많은 알림, 대부분 무시됨 | 오류 예산(error budget) 기반 SLO 알림 (레슨 20) |
| **높은 카디널리티 레이블** | user_id를 Prometheus 레이블로 사용 | 높은 카디널리티 데이터는 트레이스 사용 |
| **비구조화 로그** | `log.info(f"Order {order_id} processed")` | 키-값 쌍으로 구조화된 로깅 사용 |
| **상관 관계 누락** | 메트릭 스파이크를 특정 트레이스에 연결 불가 | 예시(exemplar), 트레이스-로그 연결 (레슨 21) |
| **과다 샘플링(over-sampling)** | 프로덕션에서 100% 트레이스 샘플링 | 관심 있는 트레이스를 유지하기 위해 테일 샘플링 사용 |
| **벤더 종속(vendor lock-in)** | 독점 에이전트 및 프로토콜 | 벤더 중립 텔레메트리를 위해 OpenTelemetry 사용 |
| **컨텍스트 전파 누락** | 서비스 경계에서 트레이스가 끊김 | W3C Trace Context 헤더가 전파되도록 보장 |

### 9.2 "크리스마스 트리" 대시보드

녹색/빨간색 상태 표시기로 가득 찬 대시보드를 만드는 것은 흔한 실패 모드이다:

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

## 10. 다음 단계

- [20_SLO_Engineering.md](./20_SLO_Engineering.md) -- 오류 예산을 활용한 SLO 정의 및 운영화
- [21_Signal_Correlation.md](./21_Signal_Correlation.md) -- 빠른 디버깅을 위한 메트릭, 로그, 트레이스 상관 관계

---

## 연습 문제

### 연습 문제 1: 모니터링 vs 관측 가능성 평가

15개의 마이크로서비스로 구성된 전자상거래 플랫폼을 운영하고 있다. 현재 Prometheus 메트릭과 Grafana 대시보드가 있다. 최근 인시던트에서 어떤 서비스가 500 오류의 연쇄를 일으켰는지 판별하지 못해 진단에 4시간이 걸렸다.

현재 관측 가능성 성숙도 수준을 평가하고, Level 3(선제적)에 도달하기 위한 로드맵을 제안하라. 구체적인 도구, 계측 변경, 문화적 전환을 포함한다.

<details>
<summary>정답 보기</summary>

**현재 상태: Level 1 (정보 기반)**
- 메트릭과 대시보드 있음 (Prometheus + Grafana)
- 분산 트레이싱 없음 (서비스 간 요청 추적 불가)
- 구조화된 로그 상관 관계 없음 (로그를 트레이스에 연결 불가)
- SLO 기반 알림 없음

**Level 3 로드맵:**

**Phase 1 (1-2개월): 분산 트레이싱 추가 → Level 2**
- OTel Collector를 DaemonSet으로 배포
- 모든 15개 서비스에 OTel 자동 계측 추가
- 트레이스 저장을 위해 Jaeger 또는 Tempo 배포
- W3C Trace Context 전파 구성
- 결과: 이제 모든 서비스에 걸쳐 요청을 따라갈 수 있음

**Phase 2 (2-3개월): 신호 상관 관계 → Level 2 (성숙)**
- 모든 구조화된 로그 항목에 trace_id 추가
- Grafana에서 메트릭 → 트레이스 → 로그 연결 구성
- Prometheus 메트릭에 exemplar 추가
- 텔레메트리 탐색 단계를 참조하는 런북 생성

**Phase 3 (3-6개월): SLO 기반 운영 → Level 3**
- 각 서비스에 SLO 정의 (가용성, 지연 시간)
- 오류 예산 추적 구현
- 임계값 알림을 번 레이트(burn-rate) 알림으로 교체
- 주요 SLI에 이상 탐지 배포
- 팀을 탐색적 디버깅에 대해 교육 (단순한 대시보드 확인이 아님)

**문화적 전환:**
- 엔지니어가 기능 개발의 일부로 계측을 작성 (이후가 아님)
- 포스트모템에 "어떤 텔레메트리가 누락되었는가"를 표준 섹션으로 포함
- 온콜 대응자가 트레이스 탐색을 첫 번째 디버깅 단계로 사용

</details>

### 연습 문제 2: 계측 설계

다음을 수행하는 사용자 등록 엔드포인트의 계측을 설계하라:
1. 사용자 세부 정보가 포함된 POST 요청 수락
2. 입력 검증
3. 데이터베이스에서 중복 이메일 확인
4. 비밀번호 해싱 (bcrypt)
5. 사용자 레코드 생성
6. 외부 서비스를 통해 환영 이메일 전송
7. 사용자 ID 반환

어떤 스팬을 만들 것인지, 각 스팬에 어떤 속성을 설정할 것인지, 어떤 메트릭을 기록할 것인지, 어떤 로그 이벤트를 발생시킬 것인지 정의하라. 선택의 이유를 설명하라.

<details>
<summary>정답 보기</summary>

**스팬:**

| 스팬 | 부모 | 속성 | 근거 |
|------|------|------|------|
| `register_user` | 루트 | `user.email_domain`, `user.source` (web/mobile/api) | 전체 작업의 최상위 스팬 |
| `validate_input` | `register_user` | `validation.error_count`, `validation.fields` | 검증은 비즈니스 관련 실패를 가질 수 있으므로 별도 스팬 |
| `check_duplicate_email` | `register_user` | `db.system=postgresql`, `db.operation=SELECT`, `user.duplicate_found` | 자동 계측된 DB 스팬 + 커스텀 속성 |
| `hash_password` | `register_user` | `bcrypt.cost_factor=12` | bcrypt는 CPU 집약적이므로 지연 시간 기여도를 추적하는 것이 중요 |
| `create_user_record` | `register_user` | `db.system=postgresql`, `db.operation=INSERT`, `user.id` (생성 후) | 자동 계측됨 |
| `send_welcome_email` | `register_user` | `email.provider=ses`, `email.template=welcome` | 외부 의존성이므로 별도 추적이 중요 |

**메트릭:**

| 메트릭 | 유형 | 레이블 | 근거 |
|--------|------|--------|------|
| `user.registrations_total` | Counter | `status` (success/duplicate/validation_error/internal_error), `source` | 비즈니스 KPI |
| `user.registration_duration_seconds` | Histogram | `source` | 성능 추적 |
| `user.password_hash_duration_seconds` | Histogram | (없음) | CPU 바운드 작업 추적 |

**로그 이벤트:**
- `user.registration.started` (INFO): source, email_domain
- `user.validation.failed` (WARN): 오류 세부 정보 (PII 없음)
- `user.duplicate_email` (WARN): email_domain (전체 이메일이 아님)
- `user.created` (INFO): user_id, source
- `user.welcome_email.sent` (INFO): user_id
- `user.welcome_email.failed` (ERROR): user_id, error type (이메일 전송은 비핵심적)

**핵심 결정:**
- 전체 이메일이나 비밀번호를 절대 로깅하지 않음 (PII/보안)
- `email_domain`을 속성으로 사용하여 등록 스팸 패턴 감지에 도움
- `send_welcome_email`은 독립적으로 실패할 수 있는 외부 의존성이므로 별도 스팬
- 비밀번호 해싱은 bcrypt cost 12에서 ~250ms가 소요되어 상당한 지연 시간 기여자이므로 자체 스팬을 갖는다

</details>

### 연습 문제 3: 카디널리티 분석

개발자가 API 사용량을 추적하기 위해 다음 Prometheus 메트릭을 추가할 것을 제안한다:

```python
api_requests = Counter(
    "api_requests_total",
    "Total API requests",
    labelnames=["method", "endpoint", "status_code", "user_id", "request_id", "region"]
)
```

카디널리티를 분석하라. 가정: 5개 메서드, 200개 엔드포인트, 50개 상태 코드, 100,000명 사용자, 무한한 request ID, 3개 리전. 이론적 최대 카디널리티는 얼마인가? 어떤 레이블을 제거하거나 수정해야 하는가? 수정된 메트릭 정의를 제안하라.

<details>
<summary>정답 보기</summary>

**카디널리티 분석:**

```
Theoretical max = 5 × 200 × 50 × 100,000 × ∞ × 3 = ∞ (unbounded!)

Without request_id = 5 × 200 × 50 × 100,000 × 3 = 15,000,000,000
Still far too high.

Without user_id  = 5 × 200 × 50 × 3 = 150,000
Getting manageable but still high.

Bucketed status = 5 × 200 × 5 (2xx/3xx/4xx/5xx/other) × 3 = 15,000
Good range.
```

**제거해야 할 레이블:**
- `request_id` -- 반드시 제거. 무한한 카디널리티. 대신 트레이스에서 trace ID를 사용한다.
- `user_id` -- 반드시 제거. 100K 고유 값. 사용자별 분석에는 트레이스를 사용한다.

**수정해야 할 레이블:**
- `status_code` → `status_class` -- 2xx, 3xx, 4xx, 5xx로 버킷화 (50 대신 5개 값).
- `endpoint` -- 매개변수화된 경로를 그룹화 고려: `/users/123` → `/users/:id` (200에서 ~30개 라우트 템플릿으로 감소).

**수정된 메트릭:**

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

## 참고 자료

- [Observability Engineering (O'Reilly, Charity Majors et al.)](https://www.oreilly.com/library/view/observability-engineering/9781492076438/)
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [OpenTelemetry Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/)
- [CNCF Observability Whitepaper](https://www.cncf.io/blog/2021/09/01/cncf-end-user-technology-radar-observability/)
- [Google SRE Book -- Monitoring Distributed Systems](https://sre.google/sre-book/monitoring-distributed-systems/)
- [Distributed Systems Observability (O'Reilly)](https://www.oreilly.com/library/view/distributed-systems-observability/9781492033431/)
