# 23. OpenTelemetry 파이프라인(Pipelines)

**이전**: [고급 메트릭 아키텍처](./22_Advanced_Metrics_Architecture.md) | **다음**: [eBPF 관측 가능성](./24_eBPF_Observability.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. OpenTelemetry Collector 아키텍처와 리시버-프로세서-익스포터(receiver-processor-exporter) 파이프라인 모델을 설명할 수 있습니다
2. 배치 처리(batching), 메모리 제한(memory limiting), 재시도 로직(retry logic)을 갖춘 프로덕션급(production-grade) Collector 파이프라인을 구성할 수 있습니다
3. 비용을 제어하면서 고가치 트레이스를 유지하는 테일 기반 샘플링(tail-based sampling) 전략을 구현할 수 있습니다
4. 에이전트(agent)와 게이트웨이(gateway) 패턴을 사용하는 멀티 티어(multi-tier) Collector 배포를 설계할 수 있습니다
5. 속성 조작(attribute manipulation), 필터링(filtering), 라우팅(routing)을 위한 커스텀 프로세서를 구축할 수 있습니다
6. Collector 상태를 모니터링하고 파이프라인 병목(bottleneck)을 트러블슈팅할 수 있습니다

---

OpenTelemetry Collector는 현대 관측 가능성(observability) 스택의 중추 신경계입니다. 애플리케이션에서 텔레메트리(telemetry) 데이터를 수신하고, 처리(필터, 변환, 샘플링, 강화)하고, 하나 이상의 백엔드(backend)로 내보냅니다. 잘 설계된 Collector 파이프라인은 전체 관측 가능성 플랫폼의 품질, 비용, 신뢰성을 결정합니다.

> **비유 -- 정수 처리장**: 원수(텔레메트리)가 취수관(리시버)을 통해 처리장으로 들어옵니다. 여과 단계(필터링은 이물질 제거), 염소 소독(샘플링은 노이즈 제거), 불소 첨가(속성 강화는 유익한 미네랄 추가) 등 처리 단계를 거칩니다. 최종적으로 깨끗한 물이 배수관(익스포터)을 통해 가정(백엔드)으로 배급됩니다. 정수 처리장이 없으면 물이 없거나 오염된 물을 받게 됩니다.

## 1. Collector 아키텍처

### 1.1 파이프라인 모델(Pipeline Model)

```
┌─────────────────────────────────────────────────────────┐
│                  OTel Collector                           │
│                                                          │
│  ┌───────────┐    ┌────────────┐    ┌───────────────┐   │
│  │ Receivers  │───→│ Processors │───→│  Exporters    │   │
│  │            │    │            │    │               │   │
│  │ - OTLP     │    │ - batch    │    │ - OTLP        │   │
│  │ - Jaeger   │    │ - filter   │    │ - Prometheus  │   │
│  │ - Prometheus│   │ - attributes│   │ - Loki        │   │
│  │ - Kafka    │    │ - sampling │    │ - Kafka       │   │
│  │ - Filelog  │    │ - transform│    │ - Debug       │   │
│  └───────────┘    └────────────┘    └───────────────┘   │
│                                                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │              Connectors                            │  │
│  │  (파이프라인 연결: traces→metrics 등)               │  │
│  └───────────────────────────────────────────────────┘  │
│                                                          │
│  ┌────────────────────┐  ┌────────────────────────┐     │
│  │ Extensions          │  │ Service                │     │
│  │ - health_check      │  │ - telemetry            │     │
│  │ - pprof             │  │ - pipelines            │     │
│  │ - zpages            │  │                        │     │
│  └────────────────────┘  └────────────────────────┘     │
└─────────────────────────────────────────────────────────┘
```

### 1.2 Collector 배포판(Distributions)

| 배포판 | 내용 | 용도 |
|--------|------|------|
| **Core** (`otelcol`) | 최소 리시버/익스포터 (OTLP만) | 단순 배포 |
| **Contrib** (`otelcol-contrib`) | 100+ 커뮤니티 구성 요소 | 대부분의 프로덕션 배포 |
| **Custom** (OCB 빌더) | 필요한 구성 요소만 | 최소 공격 표면, 작은 바이너리 크기 |

### 1.3 커스텀 Collector 빌드

```yaml
# builder-config.yaml -- OpenTelemetry Collector Builder (OCB)
dist:
  name: my-otelcol
  description: Custom collector for production
  output_path: ./build
  version: 0.96.0

receivers:
  - gomod: go.opentelemetry.io/collector/receiver/otlpreceiver v0.96.0
  - gomod: github.com/open-telemetry/opentelemetry-collector-contrib/receiver/prometheusreceiver v0.96.0
  - gomod: github.com/open-telemetry/opentelemetry-collector-contrib/receiver/filelogreceiver v0.96.0

processors:
  - gomod: go.opentelemetry.io/collector/processor/batchprocessor v0.96.0
  - gomod: go.opentelemetry.io/collector/processor/memorylimiterprocessor v0.96.0
  - gomod: github.com/open-telemetry/opentelemetry-collector-contrib/processor/filterprocessor v0.96.0
  - gomod: github.com/open-telemetry/opentelemetry-collector-contrib/processor/attributesprocessor v0.96.0
  - gomod: github.com/open-telemetry/opentelemetry-collector-contrib/processor/tailsamplingprocessor v0.96.0

exporters:
  - gomod: go.opentelemetry.io/collector/exporter/otlpexporter v0.96.0
  - gomod: go.opentelemetry.io/collector/exporter/otlphttpexporter v0.96.0
  - gomod: github.com/open-telemetry/opentelemetry-collector-contrib/exporter/prometheusexporter v0.96.0

connectors:
  - gomod: github.com/open-telemetry/opentelemetry-collector-contrib/connector/spanmetricsconnector v0.96.0

extensions:
  - gomod: go.opentelemetry.io/collector/extension/zpagesextension v0.96.0
  - gomod: github.com/open-telemetry/opentelemetry-collector-contrib/extension/healthcheckextension v0.96.0
```

```bash
# 커스텀 Collector 빌드
ocb --config builder-config.yaml
./build/my-otelcol --config collector-config.yaml
```

---

## 2. 리시버(Receivers)

### 2.1 OTLP 리시버

OpenTelemetry 네이티브 애플리케이션을 위한 기본 리시버입니다:

```yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
        max_recv_msg_size_mib: 4        # 최대 메시지 크기
        max_concurrent_streams: 100      # gRPC 스트림 제한
        keepalive:
          server_parameters:
            max_connection_idle: 11s
            max_connection_age: 30s
        tls:                             # 프로덕션에서 TLS 활성화
          cert_file: /certs/server.crt
          key_file: /certs/server.key

      http:
        endpoint: 0.0.0.0:4318
        cors:
          allowed_origins: ["https://app.example.com"]
          allowed_headers: ["Content-Type"]
```

### 2.2 Prometheus 리시버

Prometheus 형식의 메트릭을 대상에서 스크레이프(scrape)합니다:

```yaml
receivers:
  prometheus:
    config:
      scrape_configs:
        - job_name: "kubernetes-pods"
          kubernetes_sd_configs:
            - role: pod
          relabel_configs:
            - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
              action: keep
              regex: true
            - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_port]
              action: replace
              target_label: __address__
              regex: (.+)
              replacement: ${1}:${2}
```

### 2.3 Filelog 리시버

파일에서 로그를 수집합니다 (일부 사용 사례에서 Promtail/Fluentd를 대체):

```yaml
receivers:
  filelog:
    include:
      - /var/log/pods/*/*/*.log
    exclude:
      - /var/log/pods/*/otel-collector/*.log    # 자기 수집 방지
    start_at: end                                # 파일 끝에서 시작
    include_file_path: true
    include_file_name: true
    operators:
      # JSON 로그 파싱
      - type: json_parser
        timestamp:
          parse_from: attributes.timestamp
          layout: "%Y-%m-%dT%H:%M:%S.%LZ"
        severity:
          parse_from: attributes.level

      # 상관 관계를 위한 trace_id 추출
      - type: move
        from: attributes.trace_id
        to: attributes["trace_id"]

      # Kubernetes 메타데이터 추가
      - type: regex_parser
        regex: '^/var/log/pods/(?P<namespace>[^_]+)_(?P<pod>[^_]+)_'
        parse_from: attributes["log.file.path"]
```

---

## 3. 프로세서(Processors)

### 3.1 필수 프로세서

**메모리 리미터(Memory Limiter)** (항상 파이프라인 첫 번째):

```yaml
processors:
  memory_limiter:
    check_interval: 1s
    limit_mib: 512             # 하드 리밋
    spike_limit_mib: 128       # 소프트 리밋 (GC 트리거)
    # 리밋에 도달하면: 수신 데이터 거부 (백프레셔)
```

**배치 프로세서(Batch Processor)** (항상 익스포터 바로 전):

```yaml
processors:
  batch:
    send_batch_size: 1024       # 스팬/메트릭/로그 단위 배치 크기
    send_batch_max_size: 2048   # 최대 배치 크기 (과대 배치 방지)
    timeout: 5s                  # 불완전 배치 전송 전 최대 대기 시간
```

### 3.2 필터 프로세서(Filter Processor)

```yaml
processors:
  filter/traces:
    error_mode: ignore
    traces:
      span:
        # 헬스체크 트레이스 삭제
        - 'attributes["http.route"] == "/healthz"'
        - 'attributes["http.route"] == "/readyz"'
        # 내부 서비스 메시 트레이스 삭제
        - 'attributes["http.user_agent"] == "kube-probe/1.28"'

  filter/metrics:
    error_mode: ignore
    metrics:
      metric:
        # Go 런타임 메트릭 삭제 (높은 카디널리티, 거의 유용하지 않음)
        - 'name == "go_gc_duration_seconds"'
        - 'name == "go_goroutines"'
        - 'name == "go_memstats_alloc_bytes"'
        - 'HasAttrKeyOnDatapoint("user_id")'  # user_id 레이블이 있는 메트릭 삭제

  filter/logs:
    error_mode: ignore
    logs:
      log_record:
        # 프로덕션에서 DEBUG 로그 삭제
        - 'severity_number < SEVERITY_NUMBER_INFO'
        # 시끄러운 헬스체크 로그 삭제
        - 'body == "GET /healthz 200"'
```

### 3.3 속성 프로세서(Attributes Processor)

```yaml
processors:
  attributes/insert:
    actions:
      # 환경 및 배포 정보 추가
      - key: deployment.environment
        value: "production"
        action: insert
      - key: deployment.region
        from_context: metadata.region
        action: insert

  attributes/delete:
    actions:
      # 민감한 데이터 제거
      - key: http.request.header.authorization
        action: delete
      - key: db.statement
        action: delete    # SQL 쿼리 제거 (PII 포함 가능)
      - key: user.email
        action: delete    # PII 제거

  attributes/hash:
    actions:
      # 삭제 대신 민감한 값 해싱
      - key: user.id
        action: hash      # SHA-256 해시로 카디널리티 유지하면서 값 비노출
```

### 3.4 변환 프로세서(Transform Processor)

OTTL(OpenTelemetry Transformation Language)을 사용한 복잡한 변환:

```yaml
processors:
  transform/traces:
    error_mode: ignore
    trace_statements:
      - context: span
        statements:
          # 긴 속성 값 잘라내기
          - truncate_all(attributes, 256)
          # HTTP 라우트 정규화 (경로 매개변수 제거)
          - replace_pattern(attributes["http.route"], "/users/[0-9]+", "/users/:id")
          - replace_pattern(attributes["http.route"], "/orders/[a-f0-9-]+", "/orders/:id")
          # HTTP 상태 코드 기반 스팬 상태 설정
          - set(status.code, STATUS_CODE_ERROR) where attributes["http.status_code"] >= 500

  transform/logs:
    error_mode: ignore
    log_statements:
      - context: log
        statements:
          # 신용카드 번호 마스킹
          - replace_pattern(body, "\\b\\d{4}[- ]?\\d{4}[- ]?\\d{4}[- ]?\\d{4}\\b", "****-****-****-****")
          # 이메일 주소 마스킹
          - replace_pattern(body, "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}", "***@***.***")
```

---

## 4. 테일 기반 샘플링(Tail-Based Sampling)

### 4.1 헤드 샘플링(Head Sampling) vs 테일 샘플링(Tail Sampling)

| 측면 | 헤드 샘플링 | 테일 샘플링 |
|------|----------|----------|
| **결정 시점** | 트레이스 시작 시 (처리 전) | 트레이스 완료 후 |
| **사용 가능 정보** | 없음 (랜덤 결정) | 전체 트레이스: 지속 시간, 상태, 속성 |
| **리소스 비용** | 매우 낮음 | 높음 (완전한 트레이스 버퍼링 필요) |
| **구현** | SDK 수준 `TraceIdRatioBased` 샘플러 | Collector 수준 `tailsampling` 프로세서 |
| **품질** | 통계적으로 대표적이나 희귀 이벤트 누락 | 흥미로운 트레이스 유지, 지루한 것 삭제 |

### 4.2 테일 샘플링 구성

```yaml
processors:
  tail_sampling:
    decision_wait: 30s          # 트레이스 완료 대기 시간
    num_traces: 100000          # 메모리 내 최대 트레이스 수
    expected_new_traces_per_sec: 1000

    policies:
      # 정책 1: 항상 오류 트레이스 유지
      - name: errors
        type: status_code
        status_code:
          status_codes: [ERROR]

      # 정책 2: 항상 느린 트레이스 유지 (> 2초)
      - name: slow-traces
        type: latency
        latency:
          threshold_ms: 2000

      # 정책 3: 항상 중요 서비스의 트레이스 유지
      - name: critical-services
        type: string_attribute
        string_attribute:
          key: service.name
          values:
            - payment-service
            - auth-service
            - order-service

      # 정책 4: 성공 트레이스의 5% 샘플링
      - name: probabilistic-sample
        type: probabilistic
        probabilistic:
          sampling_percentage: 5

      # 정책 5: 특정 플래그가 있는 트레이스 항상 유지
      - name: debug-flag
        type: string_attribute
        string_attribute:
          key: debug
          values: ["true"]

      # 정책 6: 대용량 서비스를 위한 속도 제한 샘플링
      - name: rate-limited
        type: rate_limiting
        rate_limiting:
          spans_per_second: 100

      # 복합: AND/OR 로직으로 정책 결합
      - name: composite-policy
        type: and
        and:
          and_sub_policy:
            - name: is-health-check
              type: string_attribute
              string_attribute:
                key: http.route
                values: ["/healthz", "/readyz"]
            - name: drop-all
              type: probabilistic
              probabilistic:
                sampling_percentage: 0
```

### 4.3 테일 샘플링 모범 사례

| 실천 사항 | 이유 |
|----------|------|
| 테일 샘플링을 에이전트가 아닌 **게이트웨이** Collector에 배치 | 완전한 트레이스 필요 (모든 서비스의 모든 스팬) |
| `decision_wait`를 최대 예상 트레이스 지속 시간 이상으로 설정 | 불완전한 트레이스에 대한 조기 결정 방지 |
| `otelcol_processor_tail_sampling_count_traces_sampled` 모니터링 | 샘플링 효과 추적 |
| `num_traces`를 메모리 예산에 기반하여 설정 | 버퍼링된 각 트레이스는 ~1-10 KB 사용 |
| SDK 수준에서 헤드 샘플링과 결합 | Collector에 도달하기 전에 볼륨 감소 |

---

## 5. 멀티 티어 배포(Multi-Tier Deployment)

### 5.1 에이전트 + 게이트웨이 아키텍처

```
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│  Pod 1  │  │  Pod 2  │  │  Pod 3  │  │  Pod N  │
│  + App  │  │  + App  │  │  + App  │  │  + App  │
└────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘
     │            │            │            │
     └────────┬───┴──────┬─────┘            │
              │          │                  │
     ┌────────▼──┐  ┌────▼──────┐  ┌───────▼───┐
     │ Agent     │  │ Agent     │  │ Agent     │  ← DaemonSet
     │ (Node 1)  │  │ (Node 2)  │  │ (Node 3)  │    (노드당 하나)
     │ - batch   │  │ - batch   │  │ - batch   │
     │ - filter  │  │ - filter  │  │ - filter  │
     │ - memory  │  │ - memory  │  │ - memory  │
     │   limiter │  │   limiter │  │   limiter │
     └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
           │               │               │
           └───────────────┼───────────────┘
                           │ OTLP
                    ┌──────▼──────┐
                    │   Gateway    │ ← Deployment (2+ 레플리카)
                    │  - tail     │
                    │    sampling │
                    │  - routing  │
                    │  - transform│
                    └──┬──────┬──┘
                       │      │
              ┌────────▼┐  ┌──▼────────┐
              │  Tempo  │  │Prometheus │
              └─────────┘  └───────────┘
```

### 5.2 에이전트 구성 (DaemonSet)

```yaml
# otel-agent-config.yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  memory_limiter:
    check_interval: 1s
    limit_mib: 256
    spike_limit_mib: 64

  batch:
    send_batch_size: 512
    timeout: 5s

  # 에이전트 수준 필터링 (네트워크 트래픽 감소)
  filter/drop-health:
    error_mode: ignore
    traces:
      span:
        - 'attributes["http.route"] == "/healthz"'
    logs:
      log_record:
        - 'severity_number < SEVERITY_NUMBER_INFO'

  resource/add-node:
    attributes:
      - key: k8s.node.name
        from_attribute: HOSTNAME
        action: insert

exporters:
  otlp/gateway:
    endpoint: otel-gateway.observability.svc:4317
    tls:
      insecure: true
    retry_on_failure:
      enabled: true
      initial_interval: 5s
      max_interval: 30s
      max_elapsed_time: 300s
    sending_queue:
      enabled: true
      num_consumers: 10
      queue_size: 5000

extensions:
  health_check:
    endpoint: 0.0.0.0:13133

service:
  extensions: [health_check]
  pipelines:
    traces:
      receivers: [otlp]
      processors: [memory_limiter, resource/add-node, filter/drop-health, batch]
      exporters: [otlp/gateway]
    metrics:
      receivers: [otlp]
      processors: [memory_limiter, resource/add-node, batch]
      exporters: [otlp/gateway]
    logs:
      receivers: [otlp]
      processors: [memory_limiter, resource/add-node, filter/drop-health, batch]
      exporters: [otlp/gateway]
```

### 5.3 게이트웨이 구성 (Deployment)

```yaml
# otel-gateway-config.yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317

processors:
  memory_limiter:
    check_interval: 1s
    limit_mib: 1024
    spike_limit_mib: 256

  batch:
    send_batch_size: 2048
    timeout: 10s

  tail_sampling:
    decision_wait: 30s
    num_traces: 200000
    policies:
      - name: errors
        type: status_code
        status_code: {status_codes: [ERROR]}
      - name: slow
        type: latency
        latency: {threshold_ms: 2000}
      - name: sample-rest
        type: probabilistic
        probabilistic: {sampling_percentage: 10}

  transform/pii:
    error_mode: ignore
    trace_statements:
      - context: span
        statements:
          - replace_pattern(attributes["db.statement"], "\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}\\b", "***@***.***")

connectors:
  spanmetrics:
    histogram:
      explicit:
        buckets: [5ms, 10ms, 50ms, 100ms, 500ms, 1s, 5s]
    exemplars:
      enabled: true

exporters:
  otlp/tempo:
    endpoint: tempo.observability.svc:4317
    tls:
      insecure: true

  prometheus:
    endpoint: 0.0.0.0:8889
    resource_to_telemetry_conversion:
      enabled: true

  loki:
    endpoint: http://loki.observability.svc:3100/loki/api/v1/push

extensions:
  health_check:
    endpoint: 0.0.0.0:13133
  zpages:
    endpoint: 0.0.0.0:55679
  pprof:
    endpoint: 0.0.0.0:1888

service:
  extensions: [health_check, zpages, pprof]
  telemetry:
    metrics:
      level: detailed
      address: 0.0.0.0:8888
  pipelines:
    traces:
      receivers: [otlp]
      processors: [memory_limiter, transform/pii, tail_sampling, batch]
      exporters: [otlp/tempo, spanmetrics]
    metrics:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [prometheus]
    metrics/spanmetrics:
      receivers: [spanmetrics]
      exporters: [prometheus]
    logs:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [loki]
```

---

## 6. 파이프라인 라우팅(Pipeline Routing)

### 6.1 서비스 또는 환경별 라우팅

```yaml
processors:
  routing:
    from_attribute: service.name
    table:
      # 중요 서비스 → 더 긴 보존 기간의 전용 Tempo 인스턴스
      - value: payment-service
        exporters: [otlp/tempo-critical]
      - value: auth-service
        exporters: [otlp/tempo-critical]
    default_exporters: [otlp/tempo-standard]

exporters:
  otlp/tempo-critical:
    endpoint: tempo-critical.observability.svc:4317
  otlp/tempo-standard:
    endpoint: tempo-standard.observability.svc:4317
```

### 6.2 다중 백엔드로 팬아웃(Fan-Out)

```yaml
# 트레이스를 Tempo와 Jaeger 모두에 전송 (마이그레이션 기간)
service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [otlp/tempo, otlp/jaeger]  # 양쪽에 팬아웃

exporters:
  otlp/tempo:
    endpoint: tempo:4317
  otlp/jaeger:
    endpoint: jaeger-collector:4317
```

---

## 7. Collector 모니터링

### 7.1 핵심 Collector 메트릭

```promql
# 리시버: 수신된 데이터
rate(otelcol_receiver_accepted_spans[5m])          # 초당 수신 트레이스
rate(otelcol_receiver_refused_spans[5m])           # 초당 거부 트레이스 (백프레셔)

# 프로세서: 처리된 데이터
otelcol_processor_batch_batch_send_size_sum        # 배치 크기
rate(otelcol_processor_dropped_spans[5m])          # 필터에 의해 삭제된 스팬
otelcol_processor_tail_sampling_count_traces_sampled  # 샘플링 결정

# 익스포터: 내보낸 데이터
rate(otelcol_exporter_sent_spans[5m])              # 초당 내보낸 트레이스
rate(otelcol_exporter_send_failed_spans[5m])       # 초당 내보내기 실패
otelcol_exporter_queue_size                         # 내보내기 큐 깊이
otelcol_exporter_queue_capacity                     # 내보내기 큐 용량

# 전반적 상태
process_runtime_total_alloc_bytes                   # 메모리 사용량
otelcol_process_uptime                              # Collector 가동 시간
```

### 7.2 Collector 대시보드

```
┌─────────────────────────────────────────────────┐
│ OTel Collector 상태 대시보드                       │
├──────────────┬──────────────┬───────────────────┤
│ 수신됨       │ 처리됨        │ 내보냄            │
│ 15,230 /sec  │ 14,100 /sec  │ 14,050 /sec      │
│ (트레이스)    │ (필터 후)     │ (백엔드로)        │
├──────────────┴──────────────┴───────────────────┤
│ 파이프라인: traces                                │
│ 리시버 → 필터 → 샘플링 → 배치 → 내보내기           │
│   15230    14100    7050     7050    7050        │
│            삭제:1130 샘플:50% 큐:정상              │
├─────────────────────────────────────────────────┤
│ 메모리: 380/512 MB  │  CPU: 0.8 코어             │
│ 큐: 120/5000        │  오류: 2/분                │
└─────────────────────────────────────────────────┘
```

### 7.3 Collector 상태 알림

```yaml
# OTel Collector용 Prometheus 알림 규칙
groups:
  - name: otel_collector_alerts
    rules:
      - alert: OTelCollectorHighMemory
        expr: process_runtime_total_alloc_bytes / 1024 / 1024 > 450  # 450 MB
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "OTel Collector 메모리 사용량 높음"

      - alert: OTelCollectorExportFailures
        expr: rate(otelcol_exporter_send_failed_spans[5m]) > 0
        for: 3m
        labels:
          severity: critical
        annotations:
          summary: "OTel Collector 스팬 내보내기 실패"

      - alert: OTelCollectorQueueFull
        expr: otelcol_exporter_queue_size / otelcol_exporter_queue_capacity > 0.8
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "OTel Collector 내보내기 큐 거의 가득 참"

      - alert: OTelCollectorDataLoss
        expr: rate(otelcol_receiver_refused_spans[5m]) > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "OTel Collector 수신 스팬 거부 중 (데이터 손실)"
```

---

## 8. Kubernetes 배포

### 8.1 DaemonSet 에이전트

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: otel-agent
  namespace: observability
spec:
  selector:
    matchLabels:
      app: otel-agent
  template:
    metadata:
      labels:
        app: otel-agent
    spec:
      serviceAccountName: otel-agent
      containers:
        - name: otel-agent
          image: otel/opentelemetry-collector-contrib:0.96.0
          args: ["--config=/etc/otel/config.yaml"]
          ports:
            - containerPort: 4317    # OTLP gRPC
            - containerPort: 4318    # OTLP HTTP
            - containerPort: 13133   # 헬스체크
          resources:
            limits:
              cpu: 500m
              memory: 512Mi
            requests:
              cpu: 100m
              memory: 128Mi
          livenessProbe:
            httpGet:
              path: /
              port: 13133
            initialDelaySeconds: 10
          readinessProbe:
            httpGet:
              path: /
              port: 13133
          volumeMounts:
            - name: config
              mountPath: /etc/otel
            - name: varlog
              mountPath: /var/log
              readOnly: true
      volumes:
        - name: config
          configMap:
            name: otel-agent-config
        - name: varlog
          hostPath:
            path: /var/log
```

### 8.2 HPA를 적용한 게이트웨이 Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: otel-gateway
  namespace: observability
spec:
  replicas: 2
  selector:
    matchLabels:
      app: otel-gateway
  template:
    metadata:
      labels:
        app: otel-gateway
    spec:
      containers:
        - name: otel-gateway
          image: otel/opentelemetry-collector-contrib:0.96.0
          args: ["--config=/etc/otel/config.yaml"]
          resources:
            limits:
              cpu: 2
              memory: 2Gi
            requests:
              cpu: 500m
              memory: 512Mi
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: otel-gateway-hpa
  namespace: observability
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: otel-gateway
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 75
```

---

## 9. 트러블슈팅(Troubleshooting)

### 9.1 일반적인 문제

| 문제 | 증상 | 해결책 |
|------|------|--------|
| **데이터 손실** | 거부된 스팬(refused spans) > 0 | 메모리 한도 증가 또는 Collector 레플리카 추가 |
| **높은 지연** | 내보내기 큐 증가 | `num_consumers` 증가, 배치 프로세서 추가 |
| **OOM 킬** | Collector 파드 재시작 | `memory_limiter` 활성화, 테일 샘플링 `num_traces` 감소 |
| **누락된 스팬** | 트레이스에 빈 구간 존재 | 컨텍스트 전파(context propagation) 확인, 모든 서비스가 Collector로 내보내는지 확인 |
| **중복 데이터** | 동일한 스팬 두 번 내보냄 | 겹치는 파이프라인 확인, 최소 1회 전달(at-least-once delivery) 확인 |

### 9.2 디버그 익스포터(Debug Exporter)

```yaml
# 파이프라인을 통해 흐르는 데이터를 보기 위해 임시로 디버그 익스포터 추가
exporters:
  debug:
    verbosity: detailed     # basic | normal | detailed
    sampling_initial: 5     # 첫 N개 항목 로깅
    sampling_thereafter: 100 # 이후 N개 중 1개 항목

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [otlp/tempo, debug]  # 실제 익스포터와 함께 디버그 추가
```

### 9.3 zPages

zPages는 프로세스 내 디버깅 페이지를 제공합니다:

```yaml
extensions:
  zpages:
    endpoint: 0.0.0.0:55679

# 접근:
# http://collector:55679/debug/tracez    -- Collector를 통한 최근 트레이스
# http://collector:55679/debug/pipelinez -- 파이프라인 상태 및 통계
```

---

## 10. 다음 단계

- [24_eBPF_Observability.md](./24_eBPF_Observability.md) -- eBPF를 활용한 커널 수준 관측 가능성
- [25_Continuous_Profiling.md](./25_Continuous_Profiling.md) -- 프로덕션 CPU 및 메모리 프로파일링

---

## 연습 문제

### 연습 1: 파이프라인 설계

다음 조건의 회사를 위한 OTel Collector 파이프라인을 설계하세요:
- 50개 마이크로서비스에서 ~10,000 스팬/초 생성
- 요구 사항: 모든 오류 트레이스 유지, 성공 트레이스 10% 샘플링, 헬스체크 트레이스 삭제
- 비용 목표: 트레이스 저장소 80% 감소
- 메트릭에서 트레이스로의 exemplar 링크 유지 필수

완전한 Collector 구성을 작성하세요.

<details>
<summary>정답 보기</summary>

```yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317

processors:
  memory_limiter:
    check_interval: 1s
    limit_mib: 1024
    spike_limit_mib: 256

  # 1단계: 헬스체크 삭제 (트레이스 ~20% 제거)
  filter/health:
    error_mode: ignore
    traces:
      span:
        - 'attributes["http.route"] == "/healthz"'
        - 'attributes["http.route"] == "/readyz"'
        - 'attributes["http.route"] == "/livez"'

  # 2단계: 테일 샘플링 (나머지 ~85% 감소)
  tail_sampling:
    decision_wait: 30s
    num_traces: 200000
    expected_new_traces_per_sec: 8000  # 헬스체크 필터링 후
    policies:
      # 항상 오류 유지
      - name: keep-errors
        type: status_code
        status_code:
          status_codes: [ERROR]
      # 항상 느린 트레이스 유지
      - name: keep-slow
        type: latency
        latency:
          threshold_ms: 2000
      # 나머지 10% 샘플링
      - name: sample-success
        type: probabilistic
        probabilistic:
          sampling_percentage: 10

  batch:
    send_batch_size: 2048
    timeout: 10s

connectors:
  # 샘플링 전에 트레이스에서 메트릭 생성
  # 이를 통해 exemplar가 샘플링된 트레이스를 참조하도록 보장
  spanmetrics:
    histogram:
      explicit:
        buckets: [5ms, 10ms, 50ms, 100ms, 500ms, 1s, 5s]
    exemplars:
      enabled: true

exporters:
  otlp/tempo:
    endpoint: tempo:4317
    tls:
      insecure: true
  prometheus:
    endpoint: 0.0.0.0:8889

service:
  pipelines:
    # 트레이스 흐름: 수신 → 필터 → spanmetrics (메트릭 생성) → 샘플링 → 내보내기
    traces/pre-sample:
      receivers: [otlp]
      processors: [memory_limiter, filter/health]
      exporters: [spanmetrics]  # 모든 트레이스에서 메트릭 생성

    traces/post-sample:
      receivers: [otlp]
      processors: [memory_limiter, filter/health, tail_sampling, batch]
      exporters: [otlp/tempo]   # 샘플링된 트레이스만 저장소로

    metrics/spanmetrics:
      receivers: [spanmetrics]
      processors: [batch]
      exporters: [prometheus]
```

**비용 분석:**
- 헬스체크 필터링: 10,000 → 8,000 스팬/초 (20% 감소)
- 테일 샘플링 (10% + 오류 + 느린 요청): 8,000 → ~1,500 스팬/초 (~81% 감소)
- 총 감소: ~85% (80% 목표 달성)
- spanmetrics가 샘플링 전 트레이스에서 생성하므로 exemplar가 작동

</details>

### 연습 2: 트러블슈팅

OTel Collector에서 다음 증상이 발생하고 있습니다:
- `otelcol_receiver_refused_spans`가 초당 500으로 증가 중
- `process_runtime_total_alloc_bytes`가 1.8 GB (한도: 2 GB)
- `otelcol_exporter_queue_size`가 5,000 용량 중 4,800
- Tempo로의 내보내기 지연이 50ms에서 5s로 증가

각 증상의 근본 원인을 진단하고 수정 방안을 제안하세요.

<details>
<summary>정답 보기</summary>

**근본 원인 분석:**

증상들은 내보내기 경로에서 시작되는 연쇄적 문제를 형성합니다:

1. **내보내기 지연 증가 (50ms → 5s)**: Tempo가 느림 (과부하, 네트워크 문제, 또는 디스크 I/O 병목). 이것이 연쇄 효과의 **근본 원인**입니다.

2. **큐 가득 참 (4,800/5,000)**: 내보내기가 느리므로 큐가 채워집니다. 항목들이 빠르게 내보내지지 않고 큐에서 대기합니다.

3. **높은 메모리 (1.8 GB)**: 가득 찬 큐와 버퍼링된 데이터가 메모리를 소비합니다. 테일 샘플링(사용 중인 경우)도 메모리에 트레이스를 버퍼링합니다.

4. **거부된 스팬 (500/초)**: 메모리가 한도에 근접하면 `memory_limiter` 프로세서가 OOM을 방지하기 위해 수신 데이터를 거부하기 시작합니다.

**수정:**

| 증상 | 즉각적 수정 | 장기적 수정 |
|------|-----------|-----------|
| 내보내기 지연 | Tempo 상태 확인; 필요 시 재시작; 네트워크 확인 | Tempo 스케일링 (더 많은 ingester); Tempo 쿼리 프런트엔드 추가 |
| 큐 가득 참 | `queue_size`를 10000으로 증가 | Collector 레플리카 추가; `num_consumers` 증가 |
| 높은 메모리 | `limit_mib`를 2048로, 파드 메모리를 3Gi로 증가 | 테일 샘플링의 `num_traces` 감소; 상위에서 더 많은 데이터 필터링 |
| 거부된 스팬 | 부하 분산을 위해 에이전트 수준 Collector 추가 배포 | 로드 밸런싱 익스포터를 통한 백프레셔 인식 라우팅 구현 |

**구성 변경:**

```yaml
processors:
  memory_limiter:
    limit_mib: 2048          # 현재보다 증가
    spike_limit_mib: 512

exporters:
  otlp/tempo:
    endpoint: tempo:4317
    retry_on_failure:
      enabled: true
      initial_interval: 1s
      max_interval: 30s
    sending_queue:
      enabled: true
      num_consumers: 20      # 기본값 10에서 증가
      queue_size: 10000      # 5000에서 증가
    timeout: 30s             # 느린 백엔드를 위해 타임아웃 증가
```

</details>

---

## 참고 자료

- [OpenTelemetry Collector Documentation](https://opentelemetry.io/docs/collector/)
- [OpenTelemetry Collector Contrib](https://github.com/open-telemetry/opentelemetry-collector-contrib)
- [OTel Collector Builder (OCB)](https://opentelemetry.io/docs/collector/custom-collector/)
- [Tail Sampling Processor](https://github.com/open-telemetry/opentelemetry-collector-contrib/tree/main/processor/tailsamplingprocessor)
- [OTTL (OpenTelemetry Transformation Language)](https://github.com/open-telemetry/opentelemetry-collector-contrib/tree/main/pkg/ottl)
