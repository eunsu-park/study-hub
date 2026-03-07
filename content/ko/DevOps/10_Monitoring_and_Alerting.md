# 모니터링과 알림

**이전**: [Kubernetes 오케스트레이션](./09_Kubernetes_Orchestration.md) | **다음**: [로깅 인프라](./11_Logging_Infrastructure.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. DevOps에서 모니터링의 역할을 설명하고 네 가지 메트릭 유형(counter, gauge, histogram, summary)을 구별할 수 있습니다
2. Prometheus 아키텍처와 메트릭 수집을 위한 풀 기반(pull-based) 모델을 설명할 수 있습니다
3. 시계열 데이터를 집계, 필터링, 알림하기 위한 PromQL 쿼리를 작성할 수 있습니다
4. 주요 인프라 및 애플리케이션 메트릭을 시각화하는 Grafana 대시보드를 설계할 수 있습니다
5. 적절한 임계값, 심각도, Alertmanager를 통한 라우팅으로 알림 규칙을 구성할 수 있습니다
6. USE 방법, RED 방법, 네 가지 골든 시그널을 포함한 모니터링 모범 사례를 적용할 수 있습니다

---

모니터링은 운영 가시성의 기반입니다. 모니터링 없이는 눈을 가리고 비행하는 것과 같습니다 -- 문제가 사용자에게 영향을 미치기 전에 감지할 수 없고, 부하 상황에서 시스템 동작을 이해할 수 없으며, 데이터 기반 용량 결정을 내릴 수 없습니다. 이 레슨에서는 사실상의 오픈소스 모니터링 표준인 Prometheus, 메트릭 쿼리를 위한 PromQL, 시각화를 위한 Grafana, 지능적인 알림 라우팅을 위한 Alertmanager를 다룹니다.

> **비유 -- 자동차 계기판**: 모니터링은 자동차의 계기판과 같습니다. 게이지는 속도(처리량), 연료 잔량(리소스 사용량), 엔진 온도(포화도)를 보여줍니다. 경고등(알림)은 임계값을 넘을 때만 켜지며, 각 경고등은 다른 조치로 연결됩니다: 연료 부족은 주유, 과열은 정차를 의미합니다. 계기판 없이는 연기가 보닛에서 피어오를 때까지 엔진이 고장나고 있다는 것을 알 수 없습니다.

## 1. 모니터링이 중요한 이유

### 1.1 모니터링 피드백 루프

모니터링은 배포와 운영 사이의 피드백 루프를 완성합니다:

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  변경     │────→│ 메트릭   │────→│ 이상     │────→│ 대응     │
│  배포     │     │ 관찰     │     │ 감지     │     │  / 수정  │
└──────────┘     └──────────┘     └──────────┘     └──────────┘
      ↑                                                  │
      └──────────────────────────────────────────────────┘
                         피드백 루프
```

### 1.2 관찰 가능성의 세 가지 기둥

| 기둥 | 목적 | 도구 |
|------|------|------|
| **메트릭** | 시간에 따른 수치 측정 (CPU, 지연 시간, 오류율) | Prometheus, Datadog, CloudWatch |
| **로그** | 컨텍스트가 포함된 개별 이벤트 기록 | ELK Stack, Loki, Splunk |
| **트레이스** | 분산 서비스 간 요청 흐름 | Jaeger, Zipkin, OpenTelemetry |

이 레슨은 **메트릭과 알림**에 중점을 둡니다. 로그와 트레이스는 레슨 11과 12에서 다룹니다.

### 1.3 모니터링 방법론

**네 가지 골든 시그널** (Google SRE 참조):

| 시그널 | 설명 | 예시 메트릭 |
|--------|------|------------|
| **지연 시간(Latency)** | 요청을 처리하는 데 걸리는 시간 | `http_request_duration_seconds` |
| **트래픽(Traffic)** | 시스템에 대한 수요 | `http_requests_total` |
| **오류(Errors)** | 실패한 요청의 비율 | `http_requests_total{status=~"5.."}` |
| **포화도(Saturation)** | 시스템이 얼마나 가득 찼는지 | CPU 사용률, 메모리 압력 |

**USE 방법** (인프라 리소스용):
- **U**tilization -- 리소스가 사용 중인 시간의 비율
- **S**aturation -- 리소스가 처리할 수 없는 작업의 양 (큐 깊이)
- **E**rrors -- 오류 이벤트 수

**RED 방법** (요청 기반 서비스용):
- **R**ate -- 초당 요청 수
- **E**rrors -- 초당 실패한 요청 수
- **D**uration -- 요청 지연 시간 분포

---

## 2. Prometheus 아키텍처

### 2.1 핵심 구성 요소

Prometheus는 신뢰성과 다차원 데이터 수집을 위해 구축된 오픈소스 모니터링 시스템입니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                     Prometheus Ecosystem                         │
│                                                                  │
│  ┌────────────┐    pull     ┌──────────────┐                    │
│  │  Targets   │◄───────────│  Prometheus   │                    │
│  │ (exporters │    /metrics │   Server      │                    │
│  │  & apps)   │            │  ┌──────────┐ │    ┌────────────┐  │
│  └────────────┘            │  │  TSDB    │ │───→│ Alertmanager│  │
│                            │  │(storage) │ │    │  (라우팅,   │  │
│  ┌────────────┐            │  └──────────┘ │    │  그룹핑)    │  │
│  │ Pushgateway│───push────→│  ┌──────────┐ │    └────────────┘  │
│  │(단기 실행  │            │  │  PromQL  │ │           │        │
│  │   작업)    │            │  │ (query)  │ │    ┌──────┴─────┐  │
│  └────────────┘            │  └──────────┘ │    │   Slack /  │  │
│                            └──────────────┘    │  PagerDuty │  │
│  ┌────────────┐                   │            │   / Email  │  │
│  │  Service   │                   │            └────────────┘  │
│  │ Discovery  │──────────────────→│                             │
│  │(K8s, DNS,  │                   ▼                             │
│  │ Consul)    │            ┌──────────────┐                    │
│  └────────────┘            │   Grafana    │                    │
│                            │(시각화)      │                    │
│                            └──────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 풀 기반 vs 푸시 기반 모델

| 측면 | 풀 기반 (Prometheus) | 푸시 기반 (Datadog, Graphite) |
|------|---------------------|------------------------------|
| **방향** | 서버가 타겟을 스크레이프 | 타겟이 서버에 푸시 |
| **디스커버리** | Prometheus가 타겟을 발견 | 타겟이 서버를 알아야 함 |
| **상태 감지** | 스크레이프 실패 = 타겟 다운 | 데이터 없음 = 모호함 |
| **단기 실행 작업** | Pushgateway 필요 | 자연스러운 적합 |
| **네트워크** | Prometheus가 타겟에 접근 필요 | 타겟이 서버에 접근 필요 |
| **스케일링** | 대규모를 위한 Federation | 집계 프록시 |

### 2.3 Prometheus 구성

```yaml
# prometheus.yml
global:
  scrape_interval: 15s          # How often to scrape targets
  evaluation_interval: 15s      # How often to evaluate rules
  scrape_timeout: 10s           # Timeout per scrape

# Alertmanager configuration
alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - "alertmanager:9093"

# Rule files for alerts and recording rules
rule_files:
  - "alerts/*.yml"
  - "recording_rules/*.yml"

# Scrape targets
scrape_configs:
  # Prometheus self-monitoring
  - job_name: "prometheus"
    static_configs:
      - targets: ["localhost:9090"]

  # Node Exporter (system metrics)
  - job_name: "node"
    static_configs:
      - targets:
          - "node1:9100"
          - "node2:9100"
          - "node3:9100"

  # Application metrics
  - job_name: "webapp"
    metrics_path: "/metrics"
    scrape_interval: 10s
    static_configs:
      - targets: ["webapp:8080"]
        labels:
          environment: "production"
          team: "backend"

  # Kubernetes service discovery
  - job_name: "kubernetes-pods"
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
```

### 2.4 애플리케이션 계측

**Python (prometheus_client):**

```python
from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server
import time

# Counter: 단조 증가 값 (총 요청 수, 오류)
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)

# Gauge: 증가하거나 감소할 수 있는 값 (온도, 연결 수)
ACTIVE_CONNECTIONS = Gauge(
    "active_connections",
    "Number of active connections"
)

# Histogram: 구성 가능한 버킷에 값 분포
REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# Summary: histogram과 유사하지만 클라이언트 측에서 분위수를 계산
REQUEST_LATENCY = Summary(
    "http_request_latency_seconds",
    "HTTP request latency in seconds",
    ["method"]
)

def handle_request(method, endpoint):
    ACTIVE_CONNECTIONS.inc()
    start = time.time()

    try:
        # Process request
        process(method, endpoint)
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status="200").inc()
    except Exception:
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status="500").inc()
        raise
    finally:
        duration = time.time() - start
        REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
        REQUEST_LATENCY.labels(method=method).observe(duration)
        ACTIVE_CONNECTIONS.dec()

# Start metrics server on port 8000
start_http_server(8000)
```

**Go (prometheus/client_golang):**

```go
package main

import (
    "net/http"
    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
    requestsTotal = prometheus.NewCounterVec(
        prometheus.CounterOpts{
            Name: "http_requests_total",
            Help: "Total HTTP requests",
        },
        []string{"method", "endpoint", "status"},
    )

    requestDuration = prometheus.NewHistogramVec(
        prometheus.HistogramOpts{
            Name:    "http_request_duration_seconds",
            Help:    "HTTP request duration in seconds",
            Buckets: prometheus.DefBuckets,
        },
        []string{"method", "endpoint"},
    )
)

func init() {
    prometheus.MustRegister(requestsTotal)
    prometheus.MustRegister(requestDuration)
}

func main() {
    http.Handle("/metrics", promhttp.Handler())
    http.ListenAndServe(":8080", nil)
}
```

---

## 3. 메트릭 유형

### 3.1 네 가지 메트릭 유형

| 유형 | 동작 | 사용 사례 | 예시 |
|------|------|----------|------|
| **Counter** | 단조 증가 | 총 요청 수, 오류, 바이트 | `http_requests_total` |
| **Gauge** | 증가 및 감소 | 온도, 메모리, 큐 크기 | `node_memory_MemAvailable_bytes` |
| **Histogram** | 값을 버킷에 분배 | 지연 시간 분포, 응답 크기 | `http_request_duration_seconds` |
| **Summary** | 클라이언트 측에서 분위수 계산 | 지연 시간 백분위수 | `go_gc_duration_seconds` |

### 3.2 Histogram vs Summary

| 측면 | Histogram | Summary |
|------|-----------|---------|
| **분위수 계산** | 서버 측 (PromQL) | 클라이언트 측 (애플리케이션) |
| **집계** | 인스턴스 간 집계 가능 | 집계 불가 |
| **버킷 구성** | 미리 버킷을 정의해야 함 | 분위수 대상 정의 (예: 0.5, 0.95, 0.99) |
| **정확도** | 버킷 경계에 의존 | 구성 가능한 오차 범위 |
| **CPU 비용** | 클라이언트에서 낮음 | 클라이언트에서 높음 |
| **적합한 경우** | 대부분의 사용 사례, SLO 추적 | 인스턴스별 정확한 분위수가 필요할 때 |

**권장 사항**: 대부분의 경우 histogram을 사용하십시오. 여러 인스턴스 간에 집계할 수 있으며, Prometheus가 `histogram_quantile()`을 사용하여 버킷 데이터에서 분위수를 계산할 수 있습니다.

### 3.3 네이밍 규칙

Prometheus 네이밍 모범 사례를 따르십시오:

```
# 형식: <namespace>_<name>_<unit>
# - snake_case 사용
# - 접미사로 단위 포함 (_seconds, _bytes, _total)
# - counter에는 _total 접미사
# - 정보 메트릭에는 _info 접미사 (값이 1인 gauge)

# 좋은 예
http_requests_total
http_request_duration_seconds
node_memory_MemAvailable_bytes
process_cpu_seconds_total

# 나쁜 예
httpRequests              # camelCase, 단위 없음
request_latency           # 모호한 단위
num_errors                # counter에는 _total 접미사 사용
```

---

## 4. PromQL (Prometheus Query Language)

### 4.1 기본 쿼리

```promql
# Instant vector: 메트릭의 현재 값
http_requests_total

# 레이블로 필터링
http_requests_total{method="GET", status="200"}

# 정규식 매칭
http_requests_total{status=~"5.."}        # All 5xx status codes
http_requests_total{method!="OPTIONS"}     # Exclude OPTIONS
http_requests_total{endpoint=~"/api/.*"}   # Endpoints starting with /api/
```

### 4.2 범위 벡터와 함수

```promql
# Range vector: 시간 범위에 걸친 값
http_requests_total[5m]     # Last 5 minutes of data points

# Rate: 초당 평균 증가율 (counter용)
rate(http_requests_total[5m])

# irate: 마지막 두 데이터 포인트 기반 순간 속도 (변동성이 더 큼)
irate(http_requests_total[5m])

# increase: 범위 동안의 총 증가량
increase(http_requests_total[1h])   # Total requests in last hour

# rate vs irate 사용 시기:
# - rate()  → 평활화된 평균, 알림과 대시보드에 적합
# - irate() → 스파이크를 포착, 변동성이 큰 단기 그래프에 적합
```

### 4.3 집계 연산자

```promql
# 모든 인스턴스에 걸쳐 합계
sum(rate(http_requests_total[5m]))

# 상태 코드별 그룹화 합계
sum by (status) (rate(http_requests_total[5m]))

# 엔드포인트별 평균 요청 기간
avg by (endpoint) (rate(http_request_duration_seconds_sum[5m])
  / rate(http_request_duration_seconds_count[5m]))

# 요청 속도 기준 상위 5개 엔드포인트
topk(5, sum by (endpoint) (rate(http_requests_total[5m])))

# 높은 오류율을 가진 타겟 수
count(rate(http_requests_total{status=~"5.."}[5m]) > 0.1)

# histogram에서 분위수
histogram_quantile(0.95,
  sum by (le) (rate(http_request_duration_seconds_bucket[5m]))
)

# 엔드포인트별 99 백분위수 지연 시간
histogram_quantile(0.99,
  sum by (le, endpoint) (rate(http_request_duration_seconds_bucket[5m]))
)
```

### 4.4 유용한 PromQL 패턴

```promql
# 오류율 (5xx 응답의 비율)
sum(rate(http_requests_total{status=~"5.."}[5m]))
  / sum(rate(http_requests_total[5m])) * 100

# 가용성 (성공 응답의 비율)
1 - (
  sum(rate(http_requests_total{status=~"5.."}[5m]))
  / sum(rate(http_requests_total[5m]))
)

# 포화도: CPU 사용률
100 - (avg by (instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)

# 메모리 사용률
(1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100

# 디스크 여유 공간 (지난 24시간 추세 기반으로 가득 찰 때까지의 시간)
predict_linear(node_filesystem_avail_bytes[24h], 3600 * 24)

# 1주 전과 비교한 요청 속도 변화
rate(http_requests_total[5m])
  / rate(http_requests_total[5m] offset 1w)
```

---

## 5. Grafana 대시보드

### 5.1 데이터 소스 구성

```yaml
# grafana/provisioning/datasources/prometheus.yml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    jsonData:
      timeInterval: "15s"
      httpMethod: POST
```

### 5.2 대시보드 설계 원칙

**레이아웃 전략:**

```
┌─────────────────────────────────────────────────────────────┐
│  Row 1: 개요 (단일 통계 패널)                                │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐   │
│  │ RPS  │ │ p50  │ │ p95  │ │ p99  │ │Error%│ │Uptime│   │
│  │ 1.2k │ │ 45ms │ │120ms │ │350ms │ │ 0.3% │ │99.97%│   │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘   │
├─────────────────────────────────────────────────────────────┤
│  Row 2: 요청 및 오류율 (시계열 그래프)                       │
│  ┌────────────────────────┐ ┌────────────────────────┐     │
│  │  엔드포인트별           │ │  상태 코드별            │     │
│  │  요청 속도 (스택)       │ │  오류율                 │     │
│  └────────────────────────┘ └────────────────────────┘     │
├─────────────────────────────────────────────────────────────┤
│  Row 3: 지연 시간 (히트맵 + 백분위수 그래프)                  │
│  ┌────────────────────────┐ ┌────────────────────────┐     │
│  │  지연 시간 히트맵       │ │  p50/p95/p99 지연 시간  │     │
│  └────────────────────────┘ └────────────────────────┘     │
├─────────────────────────────────────────────────────────────┤
│  Row 4: 인프라 (CPU, 메모리, 디스크, 네트워크)               │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │ CPU %    │ │ Memory % │ │ Disk I/O │ │ Network  │      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
└─────────────────────────────────────────────────────────────┘
```

**모범 사례:**
1. 가장 중요한 패널(골든 시그널)을 맨 위에 배치합니다
2. 일관된 색상 체계를 사용합니다 (녹색 = 정상, 노란색 = 경고, 빨간색 = 심각)
3. 시간 비교를 포함합니다 (현재 vs 1일 전 / 1주 전)
4. 필터링을 위한 템플릿 변수를 사용합니다 (환경, 서비스, 인스턴스)
5. 배포 및 인시던트에 대한 어노테이션을 추가합니다

### 5.3 코드로서의 대시보드 (JSON 모델)

```json
{
  "dashboard": {
    "title": "Service Overview",
    "uid": "service-overview",
    "tags": ["production", "backend"],
    "timezone": "browser",
    "refresh": "30s",
    "templating": {
      "list": [
        {
          "name": "environment",
          "type": "query",
          "query": "label_values(http_requests_total, environment)",
          "current": { "text": "production", "value": "production" }
        },
        {
          "name": "service",
          "type": "query",
          "query": "label_values(http_requests_total{environment=\"$environment\"}, job)"
        }
      ]
    },
    "panels": [
      {
        "title": "Request Rate",
        "type": "timeseries",
        "gridPos": { "h": 8, "w": 12, "x": 0, "y": 0 },
        "targets": [
          {
            "expr": "sum by (endpoint) (rate(http_requests_total{environment=\"$environment\", job=\"$service\"}[5m]))",
            "legendFormat": "{{endpoint}}"
          }
        ]
      },
      {
        "title": "Error Rate %",
        "type": "timeseries",
        "gridPos": { "h": 8, "w": 12, "x": 12, "y": 0 },
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{environment=\"$environment\", job=\"$service\", status=~\"5..\"}[5m])) / sum(rate(http_requests_total{environment=\"$environment\", job=\"$service\"}[5m])) * 100",
            "legendFormat": "Error %"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "thresholds": {
              "steps": [
                { "color": "green", "value": null },
                { "color": "yellow", "value": 1 },
                { "color": "red", "value": 5 }
              ]
            }
          }
        }
      }
    ]
  }
}
```

---

## 6. 알림 규칙

### 6.1 Prometheus 알림 규칙

```yaml
# alerts/application.yml
groups:
  - name: application_alerts
    rules:
      # 높은 오류율
      - alert: HighErrorRate
        expr: |
          sum(rate(http_requests_total{status=~"5.."}[5m]))
          / sum(rate(http_requests_total[5m])) > 0.05
        for: 5m
        labels:
          severity: critical
          team: backend
        annotations:
          summary: "높은 오류율이 감지되었습니다"
          description: >
            오류율이 지난 5분 동안 {{ $value | humanizePercentage }}입니다
            (임계값: 5%).
          runbook_url: "https://wiki.example.com/runbooks/high-error-rate"

      # 높은 지연 시간 (p95 > 1초)
      - alert: HighLatency
        expr: |
          histogram_quantile(0.95,
            sum by (le) (rate(http_request_duration_seconds_bucket[5m]))
          ) > 1.0
        for: 10m
        labels:
          severity: warning
          team: backend
        annotations:
          summary: "p95 지연 시간이 1초를 초과합니다"
          description: "p95 지연 시간이 {{ $value }}s입니다 (임계값: 1s)."

      # 타겟 다운
      - alert: TargetDown
        expr: up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "타겟 {{ $labels.instance }}이(가) 다운되었습니다"
          description: "{{ $labels.job }}/{{ $labels.instance }}이(가) 1분 동안 접근 불가능합니다."

  - name: infrastructure_alerts
    rules:
      # 높은 CPU 사용률
      - alert: HighCPUUsage
        expr: |
          100 - (avg by (instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 85
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "{{ $labels.instance }}에서 높은 CPU 사용률"
          description: "CPU 사용률이 {{ $value }}%입니다 (임계값: 85%)."

      # 디스크 공간 부족
      - alert: DiskSpaceLow
        expr: |
          (node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"}) * 100 < 10
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "{{ $labels.instance }}에서 디스크 공간이 10% 미만입니다"
          description: "사용 가능한 디스크 공간이 {{ $value }}%입니다."

      # 24시간 내 디스크가 가득 찰 예정
      - alert: DiskWillFillIn24Hours
        expr: |
          predict_linear(node_filesystem_avail_bytes{mountpoint="/"}[6h], 24 * 3600) < 0
        for: 1h
        labels:
          severity: warning
        annotations:
          summary: "{{ $labels.instance }}의 디스크가 24시간 내에 가득 찰 것으로 예상됩니다"
```

### 6.2 `for` 절과 Pending 상태

```
알림 상태:
┌──────────┐         ┌──────────┐         ┌──────────┐
│ Inactive │──expr──→│ Pending  │──for────→│  Firing  │──────→ 알림 전송
│          │  true   │          │ elapsed  │          │
└──────────┘         └──────────┘         └──────────┘
      ↑                   │                     │
      └───────────────────┘                     │
        expr가 false가 됨                       │
      ↑                                         │
      └─────────────────────────────────────────┘
        expr가 false가 됨 ("resolved" 전송)
```

- **`for` 없음**: 표현식이 true가 되면 즉시 발동 (노이즈 발생)
- **`for: 5m`**: 발동 전에 표현식이 5분 동안 연속으로 true여야 함 (오탐 감소)
- **너무 긴 `for`**: 실제 알림을 지연시킴; critical에는 1-5분, warning에는 10-15분 사용

---

## 7. Alertmanager

### 7.1 Alertmanager 구성

```yaml
# alertmanager.yml
global:
  resolve_timeout: 5m
  smtp_smarthost: "smtp.example.com:587"
  smtp_from: "alerts@example.com"
  smtp_auth_username: "alerts@example.com"

# 억제 규칙: 상위 심각도 알림이 발동되면 하위 심각도 알림을 억제
inhibit_rules:
  - source_matchers:
      - severity="critical"
    target_matchers:
      - severity="warning"
    equal: ["alertname", "instance"]

# 라우팅 트리
route:
  receiver: "default-slack"
  group_by: ["alertname", "team"]
  group_wait: 30s        # Wait before sending first notification
  group_interval: 5m     # Wait before sending updates to a group
  repeat_interval: 4h    # Re-send if alert is still firing
  routes:
    # 심각한 알림 → PagerDuty
    - matchers:
        - severity="critical"
      receiver: "pagerduty-critical"
      repeat_interval: 1h
      continue: false

    # 경고 알림 → Slack
    - matchers:
        - severity="warning"
      receiver: "team-slack"
      repeat_interval: 4h

    # 인프라 알림 → 인프라 팀
    - matchers:
        - team="infrastructure"
      receiver: "infra-slack"

receivers:
  - name: "default-slack"
    slack_configs:
      - api_url: "https://hooks.slack.com/services/T.../B.../..."
        channel: "#alerts-general"
        title: '{{ .CommonAnnotations.summary }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

  - name: "pagerduty-critical"
    pagerduty_configs:
      - routing_key: "<pagerduty-service-key>"
        severity: "critical"

  - name: "team-slack"
    slack_configs:
      - api_url: "https://hooks.slack.com/services/T.../B.../..."
        channel: "#alerts-backend"

  - name: "infra-slack"
    slack_configs:
      - api_url: "https://hooks.slack.com/services/T.../B.../..."
        channel: "#alerts-infrastructure"
```

### 7.2 알림 라우팅 흐름

```
              수신 알림
                    │
                    ▼
           ┌───────────────┐
           │  그룹화       │  group_by: [alertname, team]
           │  (집계)       │
           └───────┬───────┘
                   │
                   ▼
           ┌───────────────┐
           │  라우트 매칭   │  severity="critical"?
           │  (상단에서)    │
           └───────┬───────┘
                   │
          ┌────────┼────────┐
          ▼        ▼        ▼
    PagerDuty   Slack    Email
    (critical) (warning) (default)
```

### 7.3 사일런스와 유지보수 창

```bash
# 계획된 유지보수를 위한 사일런스 생성 (amtool 사용)
amtool silence add \
  alertname="HighCPUUsage" \
  instance="node3:9100" \
  --duration=2h \
  --comment="Scheduled maintenance on node3" \
  --author="ops-team"

# 활성 사일런스 조회
amtool silence query

# 사일런스 조기 만료
amtool silence expire <silence-id>
```

---

## 8. 주요 Exporter

### 8.1 인기 있는 Prometheus Exporter

| Exporter | 메트릭 | 기본 포트 |
|----------|--------|----------|
| **Node Exporter** | CPU, 메모리, 디스크, 네트워크 (Linux) | 9100 |
| **Windows Exporter** | CPU, 메모리, 디스크 (Windows) | 9182 |
| **Blackbox Exporter** | HTTP/DNS/TCP/ICMP 프로브 | 9115 |
| **MySQL Exporter** | 쿼리, 연결, 복제 | 9104 |
| **PostgreSQL Exporter** | 연결, 잠금, 복제 지연 | 9187 |
| **Redis Exporter** | 메모리, 명령, 키스페이스 | 9121 |
| **Nginx Exporter** | 연결, 요청 | 9113 |
| **cAdvisor** | 컨테이너 CPU, 메모리, 네트워크 | 8080 |
| **kube-state-metrics** | Kubernetes 객체 상태 | 8080 |

### 8.2 Blackbox Exporter (엔드포인트 프로빙)

```yaml
# blackbox.yml
modules:
  http_2xx:
    prober: http
    timeout: 5s
    http:
      valid_http_versions: ["HTTP/1.1", "HTTP/2.0"]
      valid_status_codes: [200]
      method: GET
      follow_redirects: true
  tcp_connect:
    prober: tcp
    timeout: 5s

# prometheus.yml scrape config for blackbox
scrape_configs:
  - job_name: "blackbox-http"
    metrics_path: /probe
    params:
      module: [http_2xx]
    static_configs:
      - targets:
          - "https://example.com"
          - "https://api.example.com/health"
    relabel_configs:
      - source_labels: [__address__]
        target_label: __param_target
      - source_labels: [__param_target]
        target_label: instance
      - target_label: __address__
        replacement: "blackbox-exporter:9115"
```

---

## 9. 기록 규칙(Recording Rules)과 Federation

### 9.1 기록 규칙

기록 규칙은 비용이 많이 드는 쿼리를 미리 계산하고 결과를 새로운 시계열로 저장합니다:

```yaml
# recording_rules/aggregations.yml
groups:
  - name: request_rate_rules
    interval: 30s
    rules:
      # 서비스별 요청 속도 사전 계산
      - record: job:http_requests_total:rate5m
        expr: sum by (job) (rate(http_requests_total[5m]))

      # 오류 비율 사전 계산
      - record: job:http_errors:ratio_rate5m
        expr: |
          sum by (job) (rate(http_requests_total{status=~"5.."}[5m]))
          / sum by (job) (rate(http_requests_total[5m]))

      # p95 지연 시간 사전 계산
      - record: job:http_request_duration_seconds:p95
        expr: |
          histogram_quantile(0.95,
            sum by (job, le) (rate(http_request_duration_seconds_bucket[5m]))
          )
```

**기록 규칙의 이점:**
- 대시보드 쿼리가 더 빨리 로드됩니다 (사전 계산 vs 온디맨드)
- 높은 카디널리티 오버헤드 없이 장기간 쿼리를 가능하게 합니다
- 알림 규칙이 일관성을 위해 기록 규칙을 참조합니다

### 9.2 Federation

대규모 배포의 경우, Prometheus federation을 사용하여 여러 Prometheus 서버의 메트릭을 집계합니다:

```yaml
# 글로벌 Prometheus가 리전별 Prometheus 인스턴스에서 스크레이프
scrape_configs:
  - job_name: "federate"
    honor_labels: true
    metrics_path: "/federate"
    params:
      'match[]':
        - '{job=~".+"}'               # All job metrics
        - 'job:http_requests_total:rate5m'  # Recording rules
    static_configs:
      - targets:
          - "prometheus-us-east:9090"
          - "prometheus-eu-west:9090"
          - "prometheus-ap-east:9090"
```

---

## 10. 다음 단계

- [11_Logging_Infrastructure.md](./11_Logging_Infrastructure.md) - ELK와 Loki를 사용한 중앙 집중식 로깅
- [12_Distributed_Tracing.md](./12_Distributed_Tracing.md) - OpenTelemetry와 Jaeger를 사용한 요청 트레이싱

---

## 연습 문제

### 연습 문제 1: 메트릭 유형 선택

각 시나리오에 대해 올바른 Prometheus 메트릭 유형(Counter, Gauge, Histogram, Summary)을 선택하고 이유를 설명하십시오:

1. 프로세스 시작 이후 총 HTTP 500 오류 수 추적.
2. 메시지 큐의 현재 항목 수 측정.
3. 여러 인스턴스에 걸쳐 백분위수를 계산하기 위한 API 응답 시간 분포 기록.
4. 전송된 총 데이터 바이트 추적.

<details>
<summary>정답 보기</summary>

1. **Counter** -- 오류 수는 증가만 합니다 (또는 프로세스 재시작 시 리셋). `rate()`를 사용하여 초당 오류율을 계산합니다. Counter는 모든 누적 총계에 올바른 유형입니다.

2. **Gauge** -- 큐 깊이는 항목이 추가되고 제거됨에 따라 증가하고 감소합니다. Gauge는 변동하는 모든 값에 올바른 유형입니다.

3. **Histogram** -- 여러 인스턴스에 걸쳐 백분위수를 계산해야 하므로 서버 측 집계가 필요합니다. Histogram은 인스턴스 간에 합산할 수 있는 버킷에 값을 저장하며, `histogram_quantile()`이 백분위수를 계산합니다. Summary는 인스턴스별로 분위수를 계산하며 의미 있게 집계할 수 없습니다.

4. **Counter** -- 전송된 총 바이트는 단조 증가하는 값입니다. `rate(bytes_transferred_total[5m])`를 사용하여 초당 처리량(바이트)을 계산합니다.

</details>

### 연습 문제 2: PromQL 쿼리 작성

다음 시나리오에 대한 PromQL 쿼리를 작성하십시오:

1. `webapp` 작업의 모든 인스턴스에 걸친 총 요청 속도(초당 요청 수)를 계산합니다.
2. 지난 10분 동안 `/api/orders` 엔드포인트의 99 백분위수 요청 기간을 찾습니다.
3. `payment-service` 작업의 오류율을 백분율(5xx 응답 / 총 응답)로 계산합니다.
4. 어떤 인스턴스의 사용 가능한 디스크 공간이 15% 미만일 때 알림을 발생시킵니다.

<details>
<summary>정답 보기</summary>

1. 총 요청 속도:
```promql
sum(rate(http_requests_total{job="webapp"}[5m]))
```

2. `/api/orders`의 99 백분위수 지연 시간:
```promql
histogram_quantile(0.99,
  sum by (le) (rate(http_request_duration_seconds_bucket{endpoint="/api/orders"}[10m]))
)
```

3. 오류율 백분율:
```promql
sum(rate(http_requests_total{job="payment-service", status=~"5.."}[5m]))
/ sum(rate(http_requests_total{job="payment-service"}[5m])) * 100
```

4. 디스크 공간 알림 표현식:
```promql
(node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"}) * 100 < 15
```

</details>

### 연습 문제 3: 알림 설계

다음 시나리오에 대한 알림 규칙을 설계하십시오:

결제 처리 서비스는 0.1% 미만의 오류율과 500ms 미만의 p99 지연 시간을 유지해야 합니다. 두 SLO 중 하나라도 3분 이상 위반되면 critical 알림이 발동되어야 합니다. 0.05% 오류율과 300ms p99 지연 시간에서 10분 윈도우로 warning이 발동되어야 합니다.

전체 Prometheus 알림 규칙 YAML을 작성하십시오.

<details>
<summary>정답 보기</summary>

```yaml
groups:
  - name: payment_slo_alerts
    rules:
      # Critical: 오류율 > 0.1%가 3분 동안
      - alert: PaymentHighErrorRate
        expr: |
          sum(rate(http_requests_total{job="payment-service", status=~"5.."}[5m]))
          / sum(rate(http_requests_total{job="payment-service"}[5m])) > 0.001
        for: 3m
        labels:
          severity: critical
          team: payments
        annotations:
          summary: "결제 서비스 오류율 SLO 위반"
          description: >
            오류율이 {{ $value | humanizePercentage }}입니다
            (SLO 임계값: 0.1%).
          runbook_url: "https://wiki.example.com/runbooks/payment-errors"

      # Critical: p99 지연 시간 > 500ms가 3분 동안
      - alert: PaymentHighLatency
        expr: |
          histogram_quantile(0.99,
            sum by (le) (rate(http_request_duration_seconds_bucket{job="payment-service"}[5m]))
          ) > 0.5
        for: 3m
        labels:
          severity: critical
          team: payments
        annotations:
          summary: "결제 서비스 p99 지연 시간 SLO 위반"
          description: "p99 지연 시간이 {{ $value }}s입니다 (SLO 임계값: 500ms)."

      # Warning: 오류율 > 0.05%가 10분 동안
      - alert: PaymentElevatedErrorRate
        expr: |
          sum(rate(http_requests_total{job="payment-service", status=~"5.."}[5m]))
          / sum(rate(http_requests_total{job="payment-service"}[5m])) > 0.0005
        for: 10m
        labels:
          severity: warning
          team: payments
        annotations:
          summary: "결제 서비스 오류율이 SLO 임계값에 근접하고 있습니다"
          description: "오류율이 {{ $value | humanizePercentage }}입니다 (경고 기준: 0.05%)."

      # Warning: p99 지연 시간 > 300ms가 10분 동안
      - alert: PaymentElevatedLatency
        expr: |
          histogram_quantile(0.99,
            sum by (le) (rate(http_request_duration_seconds_bucket{job="payment-service"}[5m]))
          ) > 0.3
        for: 10m
        labels:
          severity: warning
          team: payments
        annotations:
          summary: "결제 서비스 p99 지연 시간이 SLO 임계값에 근접하고 있습니다"
          description: "p99 지연 시간이 {{ $value }}s입니다 (경고 기준: 300ms)."
```

**핵심 설계 결정:**
- critical 알림은 위반을 빠르게 포착하기 위해 `for: 3m`을 사용합니다.
- warning 알림은 일시적 스파이크로 인한 노이즈를 방지하기 위해 `for: 10m`을 사용합니다.
- 둘 다 평가 윈도우 내의 짧은 변동을 평활화하기 위해 `rate()[5m]`을 사용합니다.
- Alertmanager의 억제 규칙(섹션 7.1)은 critical 알림이 발동되면 warning 알림이 억제되도록 보장합니다.

</details>

### 연습 문제 4: 대시보드 계획

사용자 인증을 처리하는 마이크로서비스에 대한 Grafana 대시보드를 구축하고 있습니다. 포함할 패널, 각 패널의 PromQL 쿼리, 그리고 이 특정 서비스에 대해 각 메트릭이 왜 중요한지 설명하십시오.

<details>
<summary>정답 보기</summary>

**Row 1: 개요 (Stat 패널)**

| 패널 | PromQL | 근거 |
|------|--------|------|
| 로그인 속도 | `sum(rate(auth_login_attempts_total[5m]))` | 인증 서비스에 대한 현재 수요를 보여줍니다 |
| 로그인 성공률 | `sum(rate(auth_login_attempts_total{result="success"}[5m])) / sum(rate(auth_login_attempts_total[5m])) * 100` | 감소 시 자격 증명 문제 또는 공격을 나타냅니다 |
| p95 지연 시간 | `histogram_quantile(0.95, sum by (le) (rate(auth_request_duration_seconds_bucket[5m])))` | 인증 지연 시간은 모든 다운스트림 서비스에 직접 영향을 줍니다 |
| 활성 세션 | `auth_active_sessions` (gauge) | 용량 지표 |

**Row 2: 요청 및 오류 상세 (Time-series)**

| 패널 | PromQL | 근거 |
|------|--------|------|
| 결과별 로그인 시도 | `sum by (result) (rate(auth_login_attempts_total[5m]))` | 성공, 잘못된 비밀번호, 계정 잠금, MFA 실패를 구분합니다 |
| 토큰 작업 | `sum by (operation) (rate(auth_token_operations_total[5m]))` | 토큰 발급, 갱신, 폐기 속도를 추적합니다 |

**Row 3: 보안 (Time-series)**

| 패널 | PromQL | 근거 |
|------|--------|------|
| IP별 실패 로그인 속도 | `topk(10, sum by (source_ip) (rate(auth_login_attempts_total{result="failed"}[5m])))` | 특정 IP에서의 무차별 대입 공격을 감지합니다 |
| 계정 잠금 | `sum(rate(auth_account_lockouts_total[5m]))` | 잠금 급증은 자격 증명 스터핑을 나타낼 수 있습니다 |

**Row 4: 인프라 (Time-series)**

| 패널 | PromQL | 근거 |
|------|--------|------|
| CPU | `rate(process_cpu_seconds_total{job="auth-service"}[5m])` | 인증 서비스는 CPU 집약적일 수 있습니다 (bcrypt 해싱) |
| 메모리 | `process_resident_memory_bytes{job="auth-service"}` | 세션 저장소는 메모리 증가를 유발할 수 있습니다 |
| DB 연결 풀 | `auth_db_pool_active_connections` | 인증 서비스는 DB 바운드입니다; 풀 소진은 연쇄 장애를 유발합니다 |

</details>

---

## 참고 자료

- [Prometheus Documentation](https://prometheus.io/docs/)
- [PromQL Cheat Sheet](https://promlabs.com/promql-cheat-sheet/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Google SRE Book - Monitoring Distributed Systems](https://sre.google/sre-book/monitoring-distributed-systems/)
- [The USE Method](https://www.brendangregg.com/usemethod.html)
- [The RED Method](https://grafana.com/blog/2018/08/02/the-red-method-how-to-instrument-your-services/)
