# 22. 고급 메트릭 아키텍처(Advanced Metrics Architecture)

**이전**: [신호 상관 관계](./21_Signal_Correlation.md) | **다음**: [OpenTelemetry 파이프라인](./23_OpenTelemetry_Pipelines.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있다:

1. 멀티 클러스터 및 멀티 리전 배포를 위한 Prometheus 페더레이션(federation) 아키텍처를 설계한다
2. 운영 요구사항에 따라 장기 저장 솔루션(Thanos, Cortex, Mimir)을 비교하고 선택한다
3. 메트릭 폭발을 방지하기 위한 카디널리티(cardinality) 관리 전략을 구현한다
4. 쿼리 성능을 최적화하고 장기 분석을 가능하게 하는 레코딩 규칙(recording rules)을 작성한다
5. 내구성 있는 메트릭 저장을 위한 remote write/read를 구성한다
6. 수집 볼륨과 비용을 제어하기 위해 메트릭 리레이블링(relabeling)을 적용한다

---

단일 Prometheus 서버는 소규모 배포에서 잘 작동하지만, 인프라가 수백 개 서비스를 넘어 성장하면 근본적인 도전에 직면한다: 단일 노드 저장 한계, 클러스터 간 쿼리, 장기 보존, 고가용성, 카디널리티 폭발. 이 레슨에서는 Prometheus 메트릭을 엔터프라이즈급 배포로 확장하는 아키텍처와 도구를 다룬다.

> **비유 -- 도서관 시스템**: 단일 도서관(Prometheus 인스턴스)은 작은 마을에서 잘 작동한다. 하지만 수십 개의 캠퍼스 도서관을 가진 대학 시스템에는 모든 도서관을 한 데스크에서 검색할 수 있는 카탈로그 시스템(페더레이션/Thanos), 활성 서가에 더 이상 없는 책을 위한 아카이브(장기 저장), 여러 분관에 동일한 책이 있을 때를 처리하는 중복 제거 시스템이 필요하다.

## 1. Prometheus 확장 과제

### 1.1 단일 인스턴스 한계

| 과제 | 증상 | 임계값 |
|------|------|--------|
| **저장소** | 디스크 가득 참, 이전 데이터 삭제 | 1M 활성 시리즈로 ~2주 보존 |
| **메모리** | OOM 킬, 느린 쿼리 | ~10M 활성 시계열 |
| **쿼리 성능** | 대시보드 타임아웃, 느린 알림 | 7일 이상 복잡한 쿼리 |
| **가용성** | 단일 장애 지점 | Prometheus 재시작 시 갭 발생 |
| **멀티 클러스터** | 클러스터 간 쿼리 불가 | 1개 이상의 K8s 클러스터 |

### 1.2 확장 전략 개요

```
                    ┌──────────────────┐
                    │   Requirements   │
                    └─────────┬────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
     ┌────────▼────────┐ ┌───▼────────┐ ┌────▼────────┐
     │ Multi-cluster   │ │ Long-term  │ │ High        │
     │ querying        │ │ storage    │ │ availability│
     └────────┬────────┘ └───┬────────┘ └────┬────────┘
              │               │               │
              └───────────────┼───────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
     ┌────────▼────────┐ ┌───▼────────┐ ┌────▼────────┐
     │ Federation      │ │ Thanos     │ │ Mimir       │
     │ (built-in)      │ │ Cortex     │ │ (Grafana)   │
     │                 │ │ Mimir      │ │             │
     │ Simple, limited │ │ Sidecar or │ │ Write-path  │
     │                 │ │ receive    │ │ scaling     │
     └─────────────────┘ └────────────┘ └─────────────┘
```

---

## 2. Prometheus 페더레이션(Federation)

### 2.1 계층적 페더레이션(Hierarchical Federation)

페더레이션은 글로벌 Prometheus가 하위 수준 Prometheus 인스턴스에서 집계된 메트릭을 스크레이프할 수 있게 한다:

```
                    ┌──────────────────┐
                    │ Global Prometheus│
                    │ (cross-cluster   │
                    │  dashboards,     │
                    │  global alerts)  │
                    └───┬─────────┬────┘
                        │         │
              ┌─────────▼─┐   ┌──▼──────────┐
              │ Prom-US    │   │ Prom-EU     │
              │ (us-east   │   │ (eu-west    │
              │  cluster)  │   │  cluster)   │
              └─────┬──────┘   └──────┬──────┘
                    │                  │
              ┌─────▼──────┐   ┌──────▼──────┐
              │ Targets    │   │ Targets     │
              │ (pods,     │   │ (pods,      │
              │  nodes)    │   │  nodes)     │
              └────────────┘   └─────────────┘
```

### 2.2 페더레이션 구성

```yaml
# Global Prometheus configuration
scrape_configs:
  - job_name: "federate-us-east"
    honor_labels: true
    metrics_path: "/federate"
    params:
      'match[]':
        # Only federate recording rules and key aggregates
        - '{__name__=~"job:.*"}'                    # Recording rules
        - '{__name__="up"}'                          # Target health
        - '{__name__=~".*:.*:rate5m"}'              # Pre-aggregated rates
        - '{__name__="kube_pod_status_phase"}'       # K8s metadata
    static_configs:
      - targets: ["prometheus-us-east.internal:9090"]
        labels:
          cluster: "us-east"
          region: "us"

  - job_name: "federate-eu-west"
    honor_labels: true
    metrics_path: "/federate"
    params:
      'match[]':
        - '{__name__=~"job:.*"}'
        - '{__name__="up"}'
        - '{__name__=~".*:.*:rate5m"}'
    static_configs:
      - targets: ["prometheus-eu-west.internal:9090"]
        labels:
          cluster: "eu-west"
          region: "eu"
```

### 2.3 페더레이션 한계

| 한계 | 영향 | 완화 |
|------|------|------|
| **풀 기반(Pull-based)** | 글로벌 Prometheus가 모든 인스턴스에 도달해야 함 | VPN/메시 네트워킹 |
| **단일 장애 지점** | 글로벌 Prometheus 다운 = 클러스터 간 뷰 없음 | HA 쌍 배포 |
| **데이터 중복** | 로컬 + 글로벌에 같은 메트릭 저장 | 레코딩 규칙만 페더레이션 |
| **쿼리 제한** | 클러스터 간 원시 메트릭 조인 불가 | 대신 Thanos/Mimir 사용 |
| **확장성 한계** | 글로벌 Prometheus의 메모리 한계 | 기능 영역별 분할 |

---

## 3. Thanos

### 3.1 아키텍처 개요

Thanos는 Prometheus 자체를 수정하지 않고 장기 저장, 글로벌 쿼리, 고가용성을 확장한다:

```
┌──────────────────────────────────────────────────────┐
│                    Thanos Query                       │
│            (global PromQL query endpoint)             │
│    Deduplicates data from multiple Prometheus         │
└───────┬──────────────┬──────────────┬────────────────┘
        │              │              │
┌───────▼──────┐ ┌─────▼──────┐ ┌────▼─────────────┐
│ Thanos       │ │ Thanos     │ │ Thanos Store     │
│ Sidecar      │ │ Sidecar    │ │ Gateway          │
│ (Prom US)    │ │ (Prom EU)  │ │ (Object Storage) │
│              │ │            │ │                   │
│ Uploads TSDB │ │ Uploads    │ │ Reads historical  │
│ blocks to    │ │ blocks     │ │ data from S3/GCS  │
│ object store │ │            │ │                   │
└──────┬───────┘ └─────┬──────┘ └────────┬──────────┘
       │               │                  │
       └───────┬───────┘                  │
               ▼                          ▼
       ┌───────────────┐         ┌────────────────┐
       │ Object Storage│         │ Object Storage │
       │ (S3/GCS/Azure)│ ◄──────│ (same bucket)  │
       └───────────────┘         └────────────────┘
```

### 3.2 Thanos 컴포넌트

| 컴포넌트 | 역할 | 배포 |
|---------|------|------|
| **Sidecar** | Prometheus TSDB 블록을 오브젝트 스토리지에 업로드; Prometheus에 대한 쿼리 프록시 | Prometheus 파드의 사이드카 컨테이너 |
| **Store Gateway** | 오브젝트 스토리지에서 과거 데이터 제공 | 스테이트리스 디플로이먼트 |
| **Query** | 글로벌 PromQL 엔드포인트; 결과 중복 제거 및 병합 | 스테이트리스 디플로이먼트 |
| **Compactor** | 오브젝트 스토리지의 블록 다운샘플링 및 컴팩션 | 단일 인스턴스 (또는 샤딩) |
| **Ruler** | 모든 데이터에 대한 레코딩 및 알림 규칙 평가 | 스테이트풀 디플로이먼트 |
| **Receive** | Prometheus의 remote_write 수신 (사이드카 대안) | 스테이트풀 디플로이먼트 |

### 3.3 Thanos 사이드카 설정

```yaml
# Prometheus deployment with Thanos sidecar
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: prometheus
spec:
  template:
    spec:
      containers:
        - name: prometheus
          image: prom/prometheus:v2.51.0
          args:
            - "--config.file=/etc/prometheus/prometheus.yml"
            - "--storage.tsdb.path=/prometheus"
            - "--storage.tsdb.retention.time=2h"      # Short retention (sidecar uploads)
            - "--storage.tsdb.min-block-duration=2h"   # Required for Thanos
            - "--storage.tsdb.max-block-duration=2h"   # Required for Thanos
            - "--web.enable-lifecycle"
          volumeMounts:
            - name: prometheus-data
              mountPath: /prometheus

        - name: thanos-sidecar
          image: quay.io/thanos/thanos:v0.34.1
          args:
            - "sidecar"
            - "--tsdb.path=/prometheus"
            - "--prometheus.url=http://localhost:9090"
            - "--objstore.config-file=/etc/thanos/objstore.yml"
          volumeMounts:
            - name: prometheus-data
              mountPath: /prometheus
            - name: thanos-objstore-config
              mountPath: /etc/thanos

      volumes:
        - name: thanos-objstore-config
          secret:
            secretName: thanos-objstore
```

```yaml
# thanos-objstore.yml (S3 backend)
type: S3
config:
  bucket: "thanos-metrics"
  endpoint: "s3.us-east-1.amazonaws.com"
  region: "us-east-1"
  access_key: "${AWS_ACCESS_KEY_ID}"
  secret_key: "${AWS_SECRET_ACCESS_KEY}"
```

### 3.4 Thanos Compactor와 다운샘플링

Compactor는 이전 데이터를 다운샘플링하여 저장 비용을 줄인다:

| 해상도 | 보존 | 일별 데이터 포인트 | 용도 |
|--------|------|------------------:|------|
| **원시(Raw)** (스크레이프 간격) | 14일 | ~5,760 (15초 간격) | 최근 디버깅 |
| **5분** | 90일 | 288 | 중간 범위 분석 |
| **1시간** | 1년+ | 24 | 장기 추세 |

```yaml
# Thanos Compactor deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: thanos-compactor
spec:
  replicas: 1   # MUST be single instance
  template:
    spec:
      containers:
        - name: compactor
          image: quay.io/thanos/thanos:v0.34.1
          args:
            - "compact"
            - "--data-dir=/var/thanos/compact"
            - "--objstore.config-file=/etc/thanos/objstore.yml"
            - "--retention.resolution-raw=14d"
            - "--retention.resolution-5m=90d"
            - "--retention.resolution-1h=365d"
            - "--compact.concurrency=4"
            - "--downsample.concurrency=4"
            - "--wait"
```

---

## 4. Grafana Mimir

### 4.1 Mimir vs Thanos

| 기능 | Thanos | Mimir |
|------|--------|-------|
| **아키텍처** | 사이드카 + 오브젝트 스토리지 | 수신 경로 (remote_write) |
| **Prometheus 변경** | 최소 (사이드카) | remote_write 설정만 |
| **멀티 테넌시** | 기본 (외부 레이블) | 네이티브 (X-Scope-OrgID 헤더) |
| **확장 모델** | 컴포넌트별 확장 | 마이크로서비스 또는 모놀리식 모드 |
| **쿼리 성능** | 양호 (Store Gateway 캐싱) | 우수 (쿼리 프론트엔드 + 캐싱) |
| **운영 복잡성** | 중간 (여러 컴포넌트) | 중-상 (더 많은 이동 부분) |
| **최적 대상** | 기존 Prometheus 확장 | 그린필드, 멀티 테넌트 |

### 4.2 Mimir 아키텍처

```
┌──────────────┐     ┌──────────────┐
│ Prometheus 1 │     │ Prometheus 2 │
│ remote_write │     │ remote_write │
└──────┬───────┘     └──────┬───────┘
       │                    │
       └──────────┬─────────┘
                  │
       ┌──────────▼───────────┐
       │   Mimir Distributor   │  (Receives, validates, shards)
       └──────────┬───────────┘
                  │
       ┌──────────▼───────────┐
       │    Mimir Ingester     │  (Writes to TSDB, replicates)
       │   (Stateful, 3 replicas)│
       └──────────┬───────────┘
                  │
       ┌──────────▼───────────┐
       │   Object Storage      │  (S3/GCS/Azure -- long-term)
       └──────────┬───────────┘
                  │
       ┌──────────▼───────────┐
       │  Store Gateway        │  (Reads historical blocks)
       └──────────┬───────────┘
                  │
       ┌──────────▼───────────┐
       │   Query Frontend      │  (Splitting, caching, scheduling)
       └──────────┬───────────┘
                  │
       ┌──────────▼───────────┐
       │      Querier          │  (Merges ingester + store gateway)
       └──────────────────────┘
```

### 4.3 Prometheus Remote Write를 Mimir으로

```yaml
# prometheus.yml -- remote write to Mimir
remote_write:
  - url: "http://mimir-distributor:8080/api/v1/push"
    headers:
      X-Scope-OrgID: "team-payments"    # Multi-tenant identifier
    queue_config:
      capacity: 10000
      max_shards: 30
      max_samples_per_send: 5000
      batch_send_deadline: 5s
    write_relabel_configs:
      # Drop expensive metrics before sending
      - source_labels: [__name__]
        regex: "go_.*"
        action: drop
```

---

## 5. 카디널리티 관리(Cardinality Management)

### 5.1 카디널리티 이해

카디널리티 = 고유 시계열의 수. 메트릭 이름 + 레이블 값의 각 고유 조합이 새로운 시계열을 생성한다.

```
# 1 metric × 3 methods × 5 endpoints × 3 statuses = 45 time series
http_requests_total{method="GET", endpoint="/api/users", status="200"}
http_requests_total{method="GET", endpoint="/api/users", status="404"}
http_requests_total{method="POST", endpoint="/api/orders", status="201"}
... (45 total)

# Adding user_id with 100K users: 45 × 100,000 = 4,500,000 time series!
```

### 5.2 카디널리티 모니터링

```promql
# Total active time series
prometheus_tsdb_head_series

# Time series created per scrape
sum(scrape_series_added) by (job)

# Top 10 metrics by cardinality
topk(10, count by (__name__) ({__name__!=""}))

# Cardinality per label (find the explosion source)
count by (__name__) ({__name__=~"http_requests_total.*"})

# TSDB head chunks (memory pressure indicator)
prometheus_tsdb_head_chunks
```

### 5.3 카디널리티 감소 기법

**1. 메트릭 리레이블링 (수집 시 삭제):**

```yaml
# prometheus.yml -- drop high-cardinality labels
scrape_configs:
  - job_name: "webapp"
    metric_relabel_configs:
      # Drop metrics entirely
      - source_labels: [__name__]
        regex: "go_gc_.*|process_.*"
        action: drop

      # Remove a specific label (reduce cardinality)
      - regex: "instance"
        action: labeldrop

      # Replace high-cardinality endpoint with route template
      - source_labels: [endpoint]
        regex: "/api/users/[0-9]+"
        target_label: endpoint
        replacement: "/api/users/:id"

      # Limit number of series per scrape
    sample_limit: 10000
```

**2. 레코딩 규칙 (사전 집계):**

```yaml
groups:
  - name: cardinality_reduction
    rules:
      # Pre-aggregate per-instance metrics to per-job
      - record: job:http_requests_total:rate5m
        expr: sum by (job, method, status) (rate(http_requests_total[5m]))

      # Pre-aggregate per-pod to per-deployment
      - record: deployment:cpu_usage:avg
        expr: avg by (deployment) (rate(container_cpu_usage_seconds_total[5m]))
```

**3. 히스토그램 버킷 최적화:**

```python
# BAD: Default buckets create too many series for high-cardinality metrics
# (.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10) = 11 buckets + sum + count = 13 series per label set

# GOOD: Tailored buckets for your service's latency distribution
from prometheus_client import Histogram

# For a fast API (most requests < 100ms)
api_latency = Histogram(
    "http_request_duration_seconds",
    "Request duration",
    labelnames=["method", "route"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],  # 7 buckets
)

# For a batch job (seconds to minutes)
batch_duration = Histogram(
    "batch_job_duration_seconds",
    "Batch job duration",
    labelnames=["job_type"],
    buckets=[1, 5, 15, 30, 60, 120, 300, 600],  # 8 buckets
)
```

### 5.4 카디널리티 제한

```yaml
# Prometheus global limits
global:
  scrape_interval: 15s
  # Per-scrape limit (drops entire scrape if exceeded)
  sample_limit: 50000
  # Per-target label limit
  label_limit: 30
  label_name_length_limit: 200
  label_value_length_limit: 1000

# Mimir per-tenant limits
limits:
  max_global_series_per_user: 5000000
  max_global_series_per_metric: 50000
  max_label_names_per_series: 30
  max_label_name_length: 1024
  max_label_value_length: 2048
```

---

## 6. 레코딩 규칙(Recording Rules)

### 6.1 레코딩 규칙의 필요성

레코딩 규칙은 자주 필요하거나 비용이 큰 PromQL 표현식을 사전 계산한다:

| 레코딩 규칙 없이 | 레코딩 규칙 있음 |
|----------------|---------------|
| 대시보드 쿼리가 매 새로고침마다 계산됨 | `evaluation_interval`마다 사전 계산 |
| 큰 시간 범위에서 느림 | 범위에 관계없이 빠름 |
| Prometheus에서 높은 메모리/CPU | 쿼리 시간 비용 최소 |
| 대시보드와 알림 간 불일치 | 단일 진실 소스(single source of truth) |

### 6.2 레코딩 규칙 네이밍 컨벤션

Prometheus 네이밍 컨벤션을 따른다:

```
level:metric:operations

Examples:
  job:http_requests_total:rate5m          # per-job request rate
  instance:node_cpu:ratio                  # per-instance CPU ratio
  cluster:http_request_duration:p99        # per-cluster p99 latency
```

### 6.3 종합 레코딩 규칙

```yaml
groups:
  - name: service_sli_recording_rules
    interval: 30s
    rules:
      # --- Request Rate ---
      - record: job:http_requests_total:rate5m
        expr: sum by (job) (rate(http_requests_total[5m]))

      - record: job_method:http_requests_total:rate5m
        expr: sum by (job, method) (rate(http_requests_total[5m]))

      # --- Error Rate ---
      - record: job:http_errors:ratio_rate5m
        expr: |
          sum by (job) (rate(http_requests_total{status=~"5.."}[5m]))
          / sum by (job) (rate(http_requests_total[5m]))

      # --- Latency Percentiles ---
      - record: job:http_request_duration_seconds:p50
        expr: |
          histogram_quantile(0.50,
            sum by (job, le) (rate(http_request_duration_seconds_bucket[5m]))
          )

      - record: job:http_request_duration_seconds:p95
        expr: |
          histogram_quantile(0.95,
            sum by (job, le) (rate(http_request_duration_seconds_bucket[5m]))
          )

      - record: job:http_request_duration_seconds:p99
        expr: |
          histogram_quantile(0.99,
            sum by (job, le) (rate(http_request_duration_seconds_bucket[5m]))
          )

  - name: infrastructure_recording_rules
    interval: 60s
    rules:
      # --- CPU ---
      - record: instance:node_cpu_utilisation:ratio_rate5m
        expr: |
          1 - avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m]))

      - record: cluster:node_cpu_utilisation:avg
        expr: avg(instance:node_cpu_utilisation:ratio_rate5m)

      # --- Memory ---
      - record: instance:node_memory_utilisation:ratio
        expr: |
          1 - (
            node_memory_MemAvailable_bytes
            / node_memory_MemTotal_bytes
          )

      # --- Disk ---
      - record: instance:node_filesystem_avail:ratio
        expr: |
          node_filesystem_avail_bytes{mountpoint="/"}
          / node_filesystem_size_bytes{mountpoint="/"}
```

---

## 7. Remote Write와 Remote Read

### 7.1 Remote Write 아키텍처

```yaml
# prometheus.yml -- remote write to multiple backends
remote_write:
  # Primary: Mimir for long-term storage
  - url: "http://mimir:8080/api/v1/push"
    queue_config:
      capacity: 10000
      max_shards: 30
      min_shards: 1
      max_samples_per_send: 5000
      batch_send_deadline: 5s
      min_backoff: 30ms
      max_backoff: 5s
    write_relabel_configs:
      # Only send important metrics to long-term storage
      - source_labels: [__name__]
        regex: "(http_requests_total|http_request_duration_seconds_bucket|up|kube_.*|node_.*)"
        action: keep

  # Secondary: Datadog for business dashboards
  - url: "https://api.datadoghq.com/api/v1/series"
    bearer_token: "${DD_API_KEY}"
    write_relabel_configs:
      - source_labels: [__name__]
        regex: "(business_.*|revenue_.*|orders_.*)"
        action: keep
```

### 7.2 과거 쿼리를 위한 Remote Read

```yaml
# prometheus.yml -- remote read from Thanos/Mimir
remote_read:
  - url: "http://thanos-query:9090/api/v1/read"
    read_recent: false    # Only read remote for data older than local retention
    required_matchers:
      job: ".*"           # Read all jobs from remote
```

---

## 8. 고가용성(High Availability)

### 8.1 Thanos 중복 제거를 활용한 Prometheus HA

동일한 타겟을 스크레이프하는 두 개의 동일한 Prometheus 인스턴스를 실행한다. Thanos Query가 중복을 제거한다:

```yaml
# Prometheus instance A
global:
  external_labels:
    cluster: "production"
    replica: "A"          # Different per replica

# Prometheus instance B
global:
  external_labels:
    cluster: "production"
    replica: "B"          # Different per replica

# Thanos Query deduplication
# thanos query --query.replica-label="replica"
# Result: identical series from A and B are merged, gaps are filled
```

### 8.2 HA 알림

HA Prometheus에서는 알림이 정확히 한 번만 발생하도록 보장한다 (두 번이 아님):

```yaml
# Alertmanager cluster mode
# alertmanager --cluster.listen-address=0.0.0.0:9094
# alertmanager --cluster.peer=alertmanager-1:9094
# alertmanager --cluster.peer=alertmanager-2:9094

# Both Prometheus replicas send alerts to the Alertmanager cluster.
# Alertmanager deduplicates based on alert fingerprint (name + labels).
```

---

## 9. 비용 최적화(Cost Optimization)

### 9.1 메트릭 비용 모델

```
Monthly Cost = Active Series × Ingestion Price
             + Stored Samples × Storage Price
             + Queries × Query Price

Example (Grafana Cloud pricing model):
  1M active series × $8/1000 series = $8,000/mo
  Reducing to 500K series = $4,000/mo (50% savings)
```

### 9.2 비용 절감 전략

| 전략 | 영향 | 노력 |
|------|------|------|
| 미사용 메트릭 삭제 (`action: drop`) | 20-40% 감소 | 낮음 |
| 히스토그램 버킷 감소 | 10-30% 감소 | 중간 |
| 레코딩 규칙으로 사전 집계 | 10-20% 감소 | 중간 |
| 비핵심 타겟의 스크레이프 간격 연장 | 5-15% 감소 | 낮음 |
| 중복 레이블 제거 (`action: labeldrop`) | 10-25% 감소 | 낮음 |

### 9.3 미사용 메트릭 식별

```promql
# Metrics that exist but are never queried (requires Grafana Mimirtool or similar)
# Run mimirtool analyze to find unused metrics:
# mimirtool analyze prometheus --address http://prometheus:9090 --grafana-address http://grafana:3000

# Manual: check if a metric is used in any dashboard or alert
# If topk(1, count_over_time(some_metric_total[30d])) returns data but
# the metric appears in zero dashboards and zero alert rules, consider dropping it.

# Find metrics with highest cardinality (candidates for reduction)
topk(20,
  count by (__name__) ({__name__!=""})
)
```

---

## 10. 다음 단계

- [23_OpenTelemetry_Pipelines.md](./23_OpenTelemetry_Pipelines.md) -- 프로덕션용 OTel Collector 파이프라인 설계
- [24_eBPF_Observability.md](./24_eBPF_Observability.md) -- eBPF를 활용한 커널 수준 관측 가능성

---

## 연습 문제

### 연습 문제 1: 아키텍처 설계

다음 조건의 조직을 위한 메트릭 아키텍처를 설계하라:
- 3개 Kubernetes 클러스터 (us-east, eu-west, ap-southeast)
- 200개 마이크로서비스 (평균 500개 메트릭 각각)
- 30일 상세 보존, 1년 집계 보존
- 글로벌 대시보드 및 클러스터 간 알림 필요

Thanos와 Mimir 중 선택하고, 결정을 정당화하며, 아키텍처를 그려라. 총 활성 시계열 수를 추정하라.

<details>
<summary>정답 보기</summary>

**카디널리티 추정:**
```
200 services × 500 metrics × 3 clusters × ~5 label combinations = ~1,500,000 active time series
With recording rules and aggregations: ~2,000,000 active time series
```

**아키텍처 선택: Thanos** (정당화):
- 조직이 이미 각 클러스터에 Prometheus를 배포했음 (최소한의 혼란)
- 사이드카 모델은 기존 Prometheus 구성 변경 불필요
- 오브젝트 스토리지(S3)가 비용 효과적인 장기 보존 제공
- Compactor가 자동으로 다운샘플링 처리 (5분 → 1시간)

**아키텍처:**
```
                    ┌─────────────────────────┐
                    │     Thanos Query         │
                    │  (global PromQL endpoint) │
                    │  --query.replica-label=   │
                    │    "replica"              │
                    └──┬────────┬────────┬─────┘
                       │        │        │
              ┌────────▼──┐  ┌──▼──────┐ ┌▼──────────┐
              │ US-EAST    │  │ EU-WEST │ │ AP-SE     │
              │ Prom A + B │  │ Prom A+B│ │ Prom A+B  │
              │ + Sidecar  │  │ +Sidecar│ │ +Sidecar  │
              └─────┬──────┘  └────┬────┘ └──────┬────┘
                    │              │              │
                    └──────────────┼──────────────┘
                                   ▼
                    ┌─────────────────────────┐
                    │    S3 Bucket             │
                    │  (Thanos Store Gateway)  │
                    └─────────────────────────┘
                                   │
                    ┌──────────────▼──────────┐
                    │    Thanos Compactor      │
                    │  Raw: 14d, 5m: 90d,     │
                    │  1h: 365d               │
                    └─────────────────────────┘
```

**핵심 결정:**
- 각 클러스터에 2개 Prometheus 레플리카(HA) + Thanos 사이드카
- Thanos Query가 레플리카 간 중복 제거
- Store Gateway가 S3에서 과거 데이터 제공
- Compactor가 싱글톤으로 실행, 보존 및 다운샘플링 적용
- 각 클러스터의 레코딩 규칙이 사전 집계하여 페더레이션 부하 감소

</details>

### 연습 문제 2: 카디널리티 폭발

개발자가 배포한 새 메트릭으로 Prometheus 메모리 사용량이 3배가 되었다. 메트릭:

```python
request_trace = Counter(
    "request_trace_total",
    "Request traces",
    labelnames=["method", "path", "status", "trace_id", "user_agent", "source_ip"]
)
```

카디널리티 문제를 진단하고, 이론적 최대 시리즈 수를 계산하고 (5개 메서드, 1000개 고유 경로, 20개 상태, 무한한 trace_id, 500개 user agent, 10000개 IP), 수정을 위한 metric_relabel_configs를 작성하라.

<details>
<summary>정답 보기</summary>

**진단:**
```
Theoretical max = 5 × 1000 × 20 × ∞ × 500 × 10,000 = ∞ (unbounded!)

Even without trace_id: 5 × 1000 × 20 × 500 × 10,000 = 500,000,000,000
This is catastrophic.
```

**근본 원인:**
1. `trace_id` -- 무한 카디널리티 (요청당 하나). 반드시 제거.
2. `source_ip` -- 10K 고유 값. 레이블로는 너무 높음.
3. `user_agent` -- 500 고유 값. 레이블로는 너무 높음.
4. `path` -- 1000 고유 경로 (ID 포함 가능성). 라우트 템플릿이어야 함.
5. `status` -- 20개 상태. 클래스로 버킷화해야 함.

**수정 -- metric_relabel_configs:**

```yaml
metric_relabel_configs:
  # Option 1: Drop the entire metric (immediate fix)
  - source_labels: [__name__]
    regex: "request_trace_total"
    action: drop

  # Option 2: Fix the metric (requires code change too)
  # Remove unbounded and high-cardinality labels
  - source_labels: [__name__]
    regex: "request_trace_total"
    action: keep
  - regex: "trace_id|source_ip|user_agent"
    action: labeldrop
  # Normalize paths to route templates
  - source_labels: [path]
    regex: "/api/users/[0-9]+"
    target_label: path
    replacement: "/api/users/:id"
  - source_labels: [path]
    regex: "/api/orders/[0-9]+"
    target_label: path
    replacement: "/api/orders/:id"
```

**권장 코드 변경:**

```python
# FIXED metric
request_counter = Counter(
    "http_requests_total",
    "HTTP requests",
    labelnames=["method", "route", "status_class"],  # Low cardinality only
)

# For per-request data, use traces (not metrics)
with tracer.start_as_current_span("handle_request") as span:
    span.set_attribute("http.user_agent", user_agent)  # Fine in traces
    span.set_attribute("net.peer.ip", source_ip)       # Fine in traces
    span.set_attribute("http.trace_id", trace_id)      # Fine in traces
```

**수정된 카디널리티:** 5 methods x 30 routes x 5 status classes = 750 시계열

</details>

### 연습 문제 3: 레코딩 규칙

다음을 사전 계산하는 전자상거래 플랫폼의 완전한 레코딩 규칙 세트를 작성하라:
1. 서비스별 요청 속도, 오류율, p99 지연 시간
2. 엔드포인트별 요청 속도 (상위 50개 엔드포인트만)
3. 인프라 활용률 (CPU, 메모리, 디스크) -- 노드별 및 클러스터별
4. 비즈니스 메트릭: 분당 주문 수, 평균 주문 가치, 결제 성공률

<details>
<summary>정답 보기</summary>

```yaml
groups:
  - name: service_sli_rules
    interval: 30s
    rules:
      # Per-service request rate
      - record: job:http_requests:rate5m
        expr: sum by (job) (rate(http_requests_total[5m]))

      # Per-service error rate
      - record: job:http_errors:ratio_rate5m
        expr: |
          sum by (job) (rate(http_requests_total{status=~"5.."}[5m]))
          / sum by (job) (rate(http_requests_total[5m]))

      # Per-service p99 latency
      - record: job:http_request_duration:p99_5m
        expr: |
          histogram_quantile(0.99,
            sum by (job, le) (rate(http_request_duration_seconds_bucket[5m])))

  - name: endpoint_rules
    interval: 30s
    rules:
      # Per-endpoint request rate (keep top 50 by volume)
      - record: job_route:http_requests:rate5m
        expr: |
          topk(50,
            sum by (job, route) (rate(http_requests_total[5m]))
          )

      # Per-endpoint error rate (only for top endpoints)
      - record: job_route:http_errors:ratio_rate5m
        expr: |
          sum by (job, route) (rate(http_requests_total{status=~"5.."}[5m]))
          / sum by (job, route) (rate(http_requests_total[5m]))

  - name: infrastructure_rules
    interval: 60s
    rules:
      # Per-node CPU utilization
      - record: instance:node_cpu:ratio_rate5m
        expr: 1 - avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m]))

      # Cluster average CPU
      - record: cluster:node_cpu:avg_ratio
        expr: avg(instance:node_cpu:ratio_rate5m)

      # Per-node memory utilization
      - record: instance:node_memory:ratio
        expr: |
          1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)

      # Cluster average memory
      - record: cluster:node_memory:avg_ratio
        expr: avg(instance:node_memory:ratio)

      # Per-node disk utilization (root filesystem)
      - record: instance:node_disk:ratio
        expr: |
          1 - (
            node_filesystem_avail_bytes{mountpoint="/"}
            / node_filesystem_size_bytes{mountpoint="/"}
          )

  - name: business_rules
    interval: 30s
    rules:
      # Orders per minute
      - record: business:orders:rate1m
        expr: sum(rate(orders_created_total[1m])) * 60

      # Average order value (5-minute window for smoothing)
      - record: business:order_value:avg5m
        expr: |
          rate(order_value_dollars_sum[5m])
          / rate(order_value_dollars_count[5m])

      # Payment success rate
      - record: business:payment_success:ratio_rate5m
        expr: |
          sum(rate(payments_total{status="success"}[5m]))
          / sum(rate(payments_total[5m]))

      # Revenue rate (dollars per minute)
      - record: business:revenue:rate1m
        expr: sum(rate(order_value_dollars_sum[1m])) * 60
```

</details>

---

## 참고 자료

- [Prometheus Federation](https://prometheus.io/docs/prometheus/latest/federation/)
- [Thanos Documentation](https://thanos.io/tip/thanos/getting-started.md/)
- [Grafana Mimir Documentation](https://grafana.com/docs/mimir/latest/)
- [Prometheus Recording Rules](https://prometheus.io/docs/prometheus/latest/configuration/recording_rules/)
- [Robust Perception -- Cardinality](https://www.robustperception.io/cardinality-is-key/)
- [Prometheus Remote Write Specification](https://prometheus.io/docs/concepts/remote_write_spec/)
