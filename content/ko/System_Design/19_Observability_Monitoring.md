# 19. 관측 가능성과 모니터링

**이전**: [실전 설계 예제 2](./18_Design_Example_2.md) | **다음**: [검색 시스템](./20_Search_Systems.md)

난이도: ⭐⭐⭐⭐

## 학습 목표(Learning Objectives)

**난이도**: ⭐⭐⭐⭐ (고급)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 관측 가능성(Observability)을 정의하고 전통적인 모니터링(Monitoring)과 구별하며, 분산 시스템에서 중요한 이유를 설명할 수 있다
2. 관측 가능성의 세 가지 기둥(메트릭, 로그, 트레이스)을 설명하고, 장애 조사 중 이들이 서로 어떻게 보완하는지 설명할 수 있다
3. Prometheus 방식의 풀 기반(pull-based) 수집, 시계열 저장소, Grafana 대시보드를 활용하여 메트릭 파이프라인(metrics pipeline)을 설계할 수 있다
4. ELK 스택 또는 Loki를 이용해 대규모 로그 수집, 인덱싱, 쿼리를 위한 중앙 집중식 로깅 시스템(centralized logging system)을 아키텍처로 구성할 수 있다
5. OpenTelemetry를 사용하여 분산 트레이싱(distributed tracing)을 구현하고 서비스 경계를 넘어 요청을 연관지을 수 있다
6. SLI(서비스 수준 지표), SLO(서비스 수준 목표), 오류 예산(error budget)을 정의하고, 노이즈를 최소화하면서 실제 장애를 감지하는 경보(alerting) 전략을 설계할 수 있다

---

사용자가 "앱이 느려요"라고 보고할 때, 관측 가능성이 갖춰진 시스템은 병목이 데이터베이스 커넥션 풀 포화인지, 느린 서드파티 API인지, 아니면 특정 파드의 가비지 컬렉션 일시 중지인지 몇 분 안에 파악하게 해줍니다. 관측 가능성 없이는 팀이 추측에 의존하여 "수정 시도"를 맹목적으로 배포하게 됩니다. SLI, SLO, 오류 예산을 공식화한 Google의 SRE 관행은, 관측 가능성에 투자하면 업계 전반에 걸쳐 평균 복구 시간(Mean Time to Recovery)과 온콜 번아웃(on-call burnout)이 직접적으로 줄어든다는 것을 보여주었습니다.

---

## 목차

1. [관측 가능성의 기초](#1-관측-가능성의-기초)
2. [메트릭과 시계열 데이터](#2-메트릭과-시계열-데이터)
3. [대규모 로깅](#3-대규모-로깅)
4. [분산 트레이싱](#4-분산-트레이싱)
5. [경보와 SLO](#5-경보와-slo)
6. [OpenTelemetry](#6-opentelemetry)
7. [연습 문제](#7-연습-문제)

---

## 1. 관측 가능성의 기초

### 1.1 관측 가능성의 세 가지 기둥

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

### 1.2 모니터링 vs 관측 가능성

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

## 2. 메트릭과 시계열 데이터

### 2.1 메트릭 유형

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

### 2.2 Prometheus 아키텍처

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

### 2.3 Prometheus 메트릭 유형

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

### 2.4 PromQL 쿼리

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

## 3. 대규모 로깅

### 3.1 구조화된 로깅

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

### 3.2 ELK 스택 아키텍처

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

### 3.3 Grafana Loki (경량 대안)

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

### 3.4 로그 레벨과 모범 사례

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

**모범 사례:**
- 일반 텍스트보다 구조화된 로깅(JSON) 사용
- 모든 로그에 상관 관계 ID(trace_id) 포함
- 적절한 레벨로 로깅 (프로덕션에서 INFO 스팸 금지)
- 보존 정책 설정 (7일 hot, 30일 warm, 90일 cold)
- 민감한 데이터 로깅 금지 (PII, 자격 증명)

---

## 4. 분산 트레이싱

### 4.1 트레이스 구조

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

### 4.2 트레이싱 시스템

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

### 4.3 컨텍스트 전파

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

### 4.4 샘플링 전략

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

## 5. 경보와 SLO

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

### 5.2 오류 예산

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

### 5.3 경보 모범 사례

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

**경보 안티 패턴:**
- 경보 피로(Alert fatigue): 실행 불가능한 경보가 너무 많음
- 런북(Runbook) 누락: 해결 단계 없는 경보
- 소유자 없음: "모두"에게 전달되는 경보
- 임계값만 사용: 추세 분석 없이 정적 임계값만 사용

---

## 6. OpenTelemetry

### 6.1 OpenTelemetry 아키텍처

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

### 6.2 OTel Collector 설정

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

### 6.3 계측 예제 (Python)

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

## 7. 연습 문제

### 문제 1: 마이크로서비스 플랫폼 모니터링 설계
20개의 마이크로서비스가 있는 이커머스 플랫폼의 관측 가능성 스택을 설계하세요.

**주요 고려 사항:**
- 각 서비스에서 어떤 메트릭을 수집할 것인가?
- 서비스 간 로그를 어떻게 상관시킬 것인가?
- 트레이스에 대한 샘플링 전략은?
- 체크아웃 서비스에 대한 SLO를 정의하세요.

```
예시 접근법:

메트릭 (각 서비스에 대한 RED):
- Rate: http_requests_total by service, method, status
- Errors: http_requests_total{status=~"5.."} / total
- Duration: http_request_duration_seconds histogram

로깅:
- 모든 로그에 trace_id를 포함한 구조화된 JSON
- Loki 또는 Elasticsearch를 통한 중앙화
- 보존: 7일 hot, 30일 warm, 90일 archive

트레이싱:
- 꼬리 기반 샘플링(Tail-based sampling): 모든 오류 + p99 지연시간 유지
- 헤드 기반: 정상 트래픽의 10% 샘플링
- 백엔드로 Jaeger 또는 Tempo

체크아웃 SLO:
- 가용성(Availability): 99.95% 성공률 (30일 윈도우)
- 지연시간(Latency): p99 < 2s, p50 < 500ms
- 오류 예산: ~21.6분/월
```

### 문제 2: 경보 설계
경보 피로를 피하는 경보 전략을 설계하세요.

```
예시 답변:

멀티 번-레이트(Multi-burn-rate) 경보:
- 1시간 내 2% 예산 소모  → 페이징 (critical)
- 6시간 내 5% 예산 소모 → 페이징 (warning)
- 3일 내 10% 예산 소모 → 티켓 (low)

라우팅:
- Critical → PagerDuty → 온콜 엔지니어
- Warning  → Slack #alerts → 팀 리드
- Low      → Jira ticket → 백로그

모든 경보에는 다음이 포함되어야 함:
- 런북 링크
- 대시보드 링크
- 예상 영향
- 제안된 해결 방법
```

### 문제 3: 관측 가능성 비용 최적화
팀이 관측 가능성에 월 $50K를 지출합니다. 비용을 40% 절감하세요.

```
예시 답변:

1. 메트릭: 사용하지 않는 메트릭 삭제 (대시보드 감사)
   - 카디널리티 감소 (라벨 값 줄이기)
   - 중요하지 않은 서비스의 스크레이프 간격 증가 (15s → 60s)

2. 로그: ELK에서 Loki로 전환
   - 전체 로그 콘텐츠 인덱싱 중지
   - 로그 상세도 감소 (프로덕션에서 DEBUG → INFO)
   - 보존 기간 단축 (규제 없는 데이터에 대해 90d → 30d)

3. 트레이스: 꼬리 기반 샘플링 구현
   - 오류 및 느린 트레이스 100% 유지
   - 성공한 트레이스의 1% 샘플링
   - Grafana Tempo 사용 (대규모에서 Jaeger보다 저렴)

4. 아키텍처:
   - OpenTelemetry Collector 자체 호스팅
   - 콜드 데이터에 객체 스토리지(S3) 사용
   - 컬렉터 레벨에서 메트릭 집계
```

---

**이전**: [실전 설계 예제 2](./18_Design_Example_2.md) | **다음**: [검색 시스템](./20_Search_Systems.md)

---

## 참고 자료
- [Google SRE Book: Monitoring](https://sre.google/sre-book/monitoring-distributed-systems/)
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Loki](https://grafana.com/oss/loki/)
