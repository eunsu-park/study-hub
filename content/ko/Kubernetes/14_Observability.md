# 14. 관측 가능성(Observability)

**이전**: [오토스케일링](./13_Autoscaling.md) | **다음**: [멀티 클러스터](./15_Multi_Cluster.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Grafana 대시보드와 함께 Prometheus 기반 메트릭 파이프라인을 배포하고 구성할 수 있다
2. EFK 스택과 Grafana Loki를 사용하여 중앙 집중식 로깅을 설정할 수 있다
3. OpenTelemetry와 Jaeger를 사용한 분산 트레이싱(distributed tracing)을 구현할 수 있다
4. 견고한 서비스 관리를 위해 헬스 체크(liveness, readiness, startup 프로브)를 구성할 수 있다
5. Alertmanager로 알림 파이프라인을 구축하고 프로덕션 인시던트에 대한 디버깅 기법을 적용할 수 있다

---

관측할 수 없는 것은 운영할 수 없습니다. Kubernetes 클러스터는 엄청난 양의 데이터를 생성합니다 -- 모든 Pod, 노드, 컨트롤 플레인 구성 요소의 메트릭; 모든 컨테이너의 로그; 모든 요청의 트레이스. 과제는 데이터를 수집하는 것이 아니라 시스템의 상태, 성능, 동작에 대한 질문에 답할 수 있는 일관된 관측 가능성(observability) 시스템을 구축하는 것입니다. 이 레슨에서는 관측 가능성의 세 가지 기둥 -- 메트릭(metrics), 로그(logs), 트레이스(traces) -- 과 함께 헬스 체크, 알림, 디버깅 기법을 다룹니다.

## 목차

- [이론과 원리](#이론과-원리)
- [1. 관측 가능성의 세 가지 기둥](#1-the-three-pillars-of-observability)
- [2. Prometheus와 Grafana를 이용한 메트릭](#2-metrics-with-prometheus-and-grafana)
- [3. Kubernetes 메트릭 파이프라인](#3-kubernetes-metrics-pipeline)
- [4. EFK와 Loki를 이용한 로깅](#4-logging-with-efk-and-loki)
- [5. 분산 트레이싱](#5-distributed-tracing)
- [6. 헬스 체크](#6-health-checks)
- [7. Alertmanager를 이용한 알림](#7-alerting-with-alertmanager)
- [8. 디버깅 기법](#8-debugging-techniques)
- [연습문제](#exercises)

---

## 1. 관측 가능성의 세 가지 기둥

### 이론: 메트릭 — 집계된 시계열, 저렴한 저장, 설계상 손실

메트릭은 일정 간격으로 샘플링된 레이블 있는 숫자 값입니다 — `http_requests_total{status="200", path="/api"} 12345`. 근본 속성은 **설계상 집계**입니다 — 요청당 하나의 값을 저장하지 않고, 고유 레이블 조합당 하나의 카운터를 저장하고 각 요청에서 증가시킵니다.

이것이 메트릭을 빠르고 저렴하게 만듭니다. 100K req/s를 처리하는 웹 서비스는 100K 이벤트를 생성하지만 소수의 구별되는 레이블 조합만 가지므로, Prometheus는 스크랩당 100K가 아니라 약 10개 숫자를 봅니다. 비용은 정보 손실 — "12345번째 요청의 user agent가 무엇이었나?"를 메트릭만으로 답할 수 없습니다.

Prometheus 관습의 네 가지 메트릭 유형:

- **Counter** — 단조 증가(요청 수, 오류, 처리된 바이트). 프로세스 재시작 시 리셋. 초당 도함수를 얻기 위해 `rate(counter[5m])`을 쿼리.
- **Gauge** — 위아래로 변할 수 있는 순간 값(사용 중 메모리, 큐 길이, 온도).
- **Histogram** — 사전 정의된 버킷의 카운트(요청 지속 시간 ≤ 10ms, ≤ 100ms, ≤ 1s, +Inf). `histogram_quantile`을 통해 근사 백분위수 계산 가능.
- **Summary** — histogram과 유사하지만 백분위수를 클라이언트 측에서 계산. 쿼리는 더 저렴하지만 파드를 가로질러 집계할 수 없음(백분위수에 대해 수학이 동작하지 않음).

"올바른" 메트릭 유형은 나중에 어떤 쿼리를 답할 수 있는지에 중요합니다. 일단 카운터를 선택하면, 개별 지속 시간을 복구할 수 없습니다 — 백분위수를 원할 모든 것에 histogram을 고르세요.

### 1.1 개요

```
                    ┌───────────────────────────────────────┐
                    │          Observability                 │
                    │                                       │
        ┌───────────┼────────────┬──────────────┐          │
        │           │            │              │          │
        ▼           ▼            ▼              ▼          │
   ┌─────────┐ ┌─────────┐ ┌─────────┐  ┌───────────┐    │
   │ Metrics  │ │  Logs   │ │ Traces  │  │  Health   │    │
   │          │ │         │ │         │  │  Checks   │    │
   │Prometheus│ │EFK/Loki │ │Jaeger/  │  │Liveness/  │    │
   │ Grafana  │ │         │ │OTel     │  │Readiness  │    │
   └─────────┘ └─────────┘ └─────────┘  └───────────┘    │
        │           │            │              │          │
        └───────────┴────────────┴──────────────┘          │
                            │                              │
                    ┌───────▼──────┐                       │
                    │  Alerting    │                       │
                    │ Alertmanager │                       │
                    └──────────────┘                       │
                    └───────────────────────────────────────┘
```

| 기둥 | 답하는 질문 | 도구 |
|---|---|---|
| 메트릭(Metrics) | 무엇이 일어나고 있는가? 얼마나? | Prometheus, Grafana |
| 로그(Logs) | 왜 일어났는가? | EFK, Loki |
| 트레이스(Traces) | 호출 체인에서 어디에서 일어났는가? | Jaeger, OpenTelemetry |

---

## 2. Prometheus와 Grafana를 이용한 메트릭

### 이론: Pull vs Push — Prometheus가 Pull하는 이유

메트릭을 스토어에 넣기 위한 두 아키텍처 철학:

**Push** (StatsD, InfluxDB 원래) — 애플리케이션이 각 메트릭 값을 중앙 collector에 보냅니다. 장점 — 클라이언트 작성 단순, 단명 프로세스(CronJob, batch) 지원. 단점 — 모든 앱이 collector 주소를 알아야 함, collector의 혼잡 또는 backpressure가 모든 앱에 영향, 앱이 건강한지 알기 어려움(메트릭 없음이 "요청 없음" 또는 "앱 다운"을 의미할 수 있음).

**Pull** (Prometheus) — 애플리케이션이 `/metrics` HTTP 엔드포인트를 노출 — collector가 스케줄대로 스크랩. 장점 — 스크랩 자체가 헬스 체크(스크랩 실패 = 타겟 다운); 중앙 collector가 자체 부하 관리를 위해 스크랩 비율 제어; 서비스 디스커버리(3강 §D)가 앱 변경 없이 collector에 현재 타겟 목록 제공. 단점 — 단명 프로세스(Job)는 다음 스크랩 전에 종료 — Prometheus는 그 케이스를 위해 별도 `pushgateway`로 해결.

쿠버네티스에서는 Service 디스커버리가 이미 모든 파드의 위치를 알기에 pull 기반이 절묘하게 잘 동작합니다 — Prometheus는 쿠버네티스 서비스 디스커버리를 사용하여 스크랩 타겟을 자동으로 열거합니다. `ServiceMonitor`와 `PodMonitor` CRD(prometheus-operator의)는 "이 레이블에 매치되는 파드를, 30초마다, `metrics` 포트에서 스크랩하라"를 선언적으로 기술합니다. 그게 전체 통합입니다.

멘탈 모델 — **Prometheus는 PromQL이라는 쿼리 언어와 쿠버네티스 서비스 디스커버리를 사용하는 스크래퍼를 가진 pull 기반 시계열 데이터베이스입니다.** 다른 모든 것(Alertmanager, Grafana, 장기 스토리지를 위한 Thanos)은 그 코어 주위로 합성됩니다.

### 2.1 Prometheus 아키텍처

```
┌──────────────────────────────────────────────────────────────┐
│                     Prometheus Server                         │
│                                                              │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────┐  │
│  │  Retrieval    │  │    TSDB       │  │  HTTP Server     │  │
│  │  (scrape      │  │  (time series │  │  (PromQL API)    │  │
│  │   targets)    │  │   database)   │  │                  │  │
│  └──────┬───────┘  └───────────────┘  └────────┬─────────┘  │
│         │                                       │            │
└─────────┼───────────────────────────────────────┼────────────┘
          │ scrape                                │ query
          ▼                                       ▼
   ┌──────────────┐                      ┌──────────────────┐
   │  Targets      │                      │   Grafana        │
   │  - Pods       │                      │   Alertmanager   │
   │  - Nodes      │                      │   API clients    │
   │  - Services   │                      └──────────────────┘
   └──────────────┘
```

### 2.2 Prometheus 스택 설치

```bash
# Install kube-prometheus-stack (Prometheus + Grafana + Alertmanager + exporters)
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set prometheus.prometheusSpec.retention=30d \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.storageClassName=gp3 \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=100Gi \
  --set grafana.adminPassword=admin \
  --set grafana.persistence.enabled=true \
  --set grafana.persistence.size=10Gi

# Verify
kubectl get pods -n monitoring
kubectl get svc -n monitoring
```

### 2.3 애플리케이션 메트릭을 위한 ServiceMonitor

```yaml
# Tell Prometheus to scrape your application
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: web-app-monitor
  namespace: production
  labels:
    release: monitoring  # Must match Prometheus operator label selector
spec:
  selector:
    matchLabels:
      app: web-app
  endpoints:
  - port: metrics
    path: /metrics
    interval: 15s
    scrapeTimeout: 10s
  namespaceSelector:
    matchNames:
    - production
```

### 2.4 사이드카 메트릭을 위한 PodMonitor

```yaml
# For pods without a Service (e.g., Jobs, sidecars)
apiVersion: monitoring.coreos.com/v1
kind: PodMonitor
metadata:
  name: envoy-sidecar-monitor
  namespace: production
spec:
  selector:
    matchLabels:
      sidecar: envoy
  podMetricsEndpoints:
  - port: admin
    path: /stats/prometheus
    interval: 30s
```

### 2.5 주요 PromQL 쿼리

```promql
# CPU usage by namespace
sum(rate(container_cpu_usage_seconds_total{namespace="production"}[5m])) by (pod)

# Memory usage percentage
sum(container_memory_working_set_bytes{namespace="production"}) by (pod) /
sum(container_spec_memory_limit_bytes{namespace="production"}) by (pod) * 100

# Pod restart count
increase(kube_pod_container_status_restarts_total{namespace="production"}[1h])

# Request rate per service (from application metrics)
sum(rate(http_requests_total{namespace="production"}[5m])) by (service)

# Error rate (5xx)
sum(rate(http_requests_total{status=~"5.."}[5m])) /
sum(rate(http_requests_total[5m])) * 100

# P99 request latency
histogram_quantile(0.99,
  sum(rate(http_request_duration_seconds_bucket{namespace="production"}[5m])) by (le, service)
)

# Node disk pressure
kube_node_status_condition{condition="DiskPressure", status="true"} == 1

# Pods not ready for more than 5 minutes
min_over_time(kube_pod_status_ready{condition="true"}[5m]) == 0
```

### 2.6 Grafana 대시보드 구성

```json
{
  "dashboard": {
    "title": "Kubernetes Cluster Overview",
    "panels": [
      {
        "title": "CPU Usage by Namespace",
        "type": "timeseries",
        "targets": [
          {
            "expr": "sum(rate(container_cpu_usage_seconds_total{container!=\"\"}[5m])) by (namespace)",
            "legendFormat": "{{namespace}}"
          }
        ]
      },
      {
        "title": "Memory Usage by Namespace",
        "type": "timeseries",
        "targets": [
          {
            "expr": "sum(container_memory_working_set_bytes{container!=\"\"}) by (namespace) / 1024 / 1024 / 1024",
            "legendFormat": "{{namespace}} (GiB)"
          }
        ]
      },
      {
        "title": "Pod Restart Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(increase(kube_pod_container_status_restarts_total[1h]))"
          }
        ]
      }
    ]
  }
}
```

---

## 3. Kubernetes 메트릭 파이프라인

### 3.1 Metrics-Server

Metrics-server는 `kubectl top`과 HPA에서 사용하는 실시간 CPU 및 메모리 메트릭을 제공합니다:

```bash
# Install metrics-server
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# For local clusters (minikube/kind), add --kubelet-insecure-tls
kubectl patch deployment metrics-server -n kube-system \
  --type='json' \
  -p='[{"op": "add", "path": "/spec/template/spec/containers/0/args/-", "value": "--kubelet-insecure-tls"}]'

# Verify
kubectl top nodes
# NAME          CPU(cores)   CPU%   MEMORY(bytes)   MEMORY%
# node-1        450m         11%    3200Mi          41%
# node-2        380m         9%     2800Mi          36%

kubectl top pods -n production --sort-by=cpu
```

### 3.2 kube-state-metrics

kube-state-metrics는 Kubernetes 객체의 상태(리소스 사용량이 아닌)에 대한 메트릭을 생성합니다:

```bash
# Installed automatically with kube-prometheus-stack
# Or install standalone:
helm install kube-state-metrics prometheus-community/kube-state-metrics \
  --namespace monitoring
```

kube-state-metrics의 주요 메트릭:

```promql
# Deployment replicas status
kube_deployment_spec_replicas
kube_deployment_status_replicas_available
kube_deployment_status_replicas_unavailable

# Pod status
kube_pod_status_phase{phase="Running"}
kube_pod_status_phase{phase="Pending"}
kube_pod_status_phase{phase="Failed"}

# Container status
kube_pod_container_status_waiting_reason
kube_pod_container_status_terminated_reason

# Resource quotas
kube_resourcequota{type="hard"}
kube_resourcequota{type="used"}
```

### 3.3 Node Exporter

node-exporter는 각 노드의 하드웨어 및 OS 수준 메트릭을 제공합니다:

```promql
# Node CPU usage
100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) by (instance) * 100)

# Node memory usage
(1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100

# Disk usage percentage
(1 - node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"}) * 100

# Network traffic
rate(node_network_receive_bytes_total{device="eth0"}[5m])
rate(node_network_transmit_bytes_total{device="eth0"}[5m])
```

---

## 4. EFK와 Loki를 이용한 로깅

### 이론: 로그 — 높은 카디널리티, 무엇으로 인덱싱?

로그는 이벤트 기록입니다 — 보통 자유 형식 텍스트와 구조화된 필드. 도전 — 바쁜 서비스는 분당 수백만 로그 라인을 생성하고, 검색할 수 있어야 합니다.

매우 다른 비용 프로파일을 가진 두 인덱싱 전략:

**모든 것 인덱싱 (Elasticsearch / EFK).** 로그 본문을 토큰화하고 모든 단어에 대한 inverted 인덱스를 빌드하고, 모든 필드에 인덱싱. 장점 — 임의 검색이 빠름(`error AND timeout AND user_id:42`). 단점 — 대규모에서 극도로 비쌈 — 인덱스 스토리지가 종종 raw 로그 볼륨의 5-10배; 높은 카디널리티 필드(user ID, request ID, IP)가 인덱스를 폭발시킴; 모든 단어가 인덱싱되어야 하므로 ingestion이 느림.

**레이블만 인덱싱 (Loki / Grafana).** 작은 상위 수준 레이블 집합(`namespace`, `pod`, `app`, `level`)에만 inverted 인덱스 빌드. raw 로그 본문을 압축된 blob으로 저장. 장점 — ingestion이 저렴(전체 텍스트 인덱싱 없음); 스토리지가 ES보다 약 10배 저렴. 단점 — 레이블에 있지 않은 쿼리는 레이블로 좁혀진 시간 범위 내에서 로그 본문의 full-scan이 됨 — `{app="api"} |= "error"`(레이블이 좁힌 후 텍스트 grep)에는 빠름, 레이블 없이 "모든 로그에서 `user_id=42` 검색"에는 느림.

Loki는 클라우드 네이티브 로그의 카디널리티를 위해 명시적으로 설계되었습니다. Prometheus 스타일 레이블(낮은 카디널리티, 레이블당 약 수십 값)은 동작 — 요청별 고유 ID는 안 됩니다. 아키텍처 교훈 — **인덱싱하는 것에 보수적이 되어 로그를 저렴히 저장하고, Loki가 가장 약한 "특정 한 요청 보기" 사용 사례에는 트레이스(§D)를 사용하라.**

EFK는 여전히 자리가 있습니다 — 비용에 관계없이 전체 텍스트 검색이 필요할 때(보안 포렌식, 컴플라이언스 audit 로그). 전형적 앱/인프라 로그에는 Loki의 경제학이 이깁니다.

### 4.1 EFK 스택 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│  Node 1              Node 2              Node 3          │
│  ┌──────────┐       ┌──────────┐       ┌──────────┐    │
│  │ Fluentd  │       │ Fluentd  │       │ Fluentd  │    │
│  │(DaemonSet)│       │(DaemonSet)│       │(DaemonSet)│    │
│  └────┬─────┘       └────┬─────┘       └────┬─────┘    │
│       │                  │                  │           │
│       └──────────────────┼──────────────────┘           │
│                          │                              │
│                    ┌─────▼──────┐                       │
│                    │Elasticsearch│                       │
│                    │ (storage)   │                       │
│                    └─────┬──────┘                       │
│                          │                              │
│                    ┌─────▼──────┐                       │
│                    │  Kibana    │                       │
│                    │ (visualize)│                       │
│                    └────────────┘                       │
└─────────────────────────────────────────────────────────┘
```

### 4.2 Fluentd DaemonSet

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: fluentd
  namespace: logging
spec:
  selector:
    matchLabels:
      app: fluentd
  template:
    metadata:
      labels:
        app: fluentd
    spec:
      serviceAccountName: fluentd
      tolerations:
      - key: node-role.kubernetes.io/control-plane
        effect: NoSchedule
      containers:
      - name: fluentd
        image: fluent/fluentd-kubernetes-daemonset:v1.16-debian-elasticsearch8-1
        env:
        - name: FLUENT_ELASTICSEARCH_HOST
          value: "elasticsearch.logging.svc"
        - name: FLUENT_ELASTICSEARCH_PORT
          value: "9200"
        - name: FLUENT_ELASTICSEARCH_SCHEME
          value: "https"
        - name: FLUENT_ELASTICSEARCH_SSL_VERIFY
          value: "false"
        resources:
          requests:
            cpu: 100m
            memory: 200Mi
          limits:
            cpu: 500m
            memory: 500Mi
        volumeMounts:
        - name: varlog
          mountPath: /var/log
        - name: containers
          mountPath: /var/lib/docker/containers
          readOnly: true
        - name: fluentd-config
          mountPath: /fluentd/etc/conf.d
      volumes:
      - name: varlog
        hostPath:
          path: /var/log
      - name: containers
        hostPath:
          path: /var/lib/docker/containers
      - name: fluentd-config
        configMap:
          name: fluentd-config
```

### 4.3 Fluentd 구성

```yaml
# fluentd-config ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluentd-config
  namespace: logging
data:
  kubernetes.conf: |
    <source>
      @type tail
      path /var/log/containers/*.log
      pos_file /var/log/fluentd-containers.log.pos
      tag kubernetes.*
      read_from_head true
      <parse>
        @type json
        time_key time
        time_format %Y-%m-%dT%H:%M:%S.%NZ
      </parse>
    </source>

    <filter kubernetes.**>
      @type kubernetes_metadata
      @id filter_kube_metadata
    </filter>

    # Drop health check logs to reduce noise
    <filter kubernetes.**>
      @type grep
      <exclude>
        key $.kubernetes.container_name
        pattern /^(healthz|readyz)$/
      </exclude>
    </filter>

    <match kubernetes.**>
      @type elasticsearch
      host elasticsearch.logging.svc
      port 9200
      logstash_format true
      logstash_prefix k8s
      include_tag_key true
      <buffer>
        @type file
        path /var/log/fluentd-buffers/kubernetes.system.buffer
        flush_mode interval
        flush_interval 5s
        retry_max_interval 30
        chunk_limit_size 8M
        total_limit_size 512M
        overflow_action drop_oldest_chunk
      </buffer>
    </match>
```

### 4.4 Grafana Loki (경량 대안)

Loki는 비용 효율적이고 운영이 쉽도록 설계된 로그 집계 시스템입니다:

```bash
# Install Loki stack (Loki + Promtail)
helm repo add grafana https://grafana.github.io/helm-charts
helm install loki grafana/loki-stack \
  --namespace logging \
  --create-namespace \
  --set loki.persistence.enabled=true \
  --set loki.persistence.size=50Gi \
  --set promtail.enabled=true \
  --set grafana.enabled=false  # Use existing Grafana
```

### 4.5 Promtail 구성

```yaml
# Promtail pipeline to enrich and filter logs
apiVersion: v1
kind: ConfigMap
metadata:
  name: promtail-config
  namespace: logging
data:
  promtail.yaml: |
    server:
      http_listen_port: 3101
    positions:
      filename: /tmp/positions.yaml
    clients:
    - url: http://loki.logging.svc:3100/loki/api/v1/push
    scrape_configs:
    - job_name: kubernetes-pods
      kubernetes_sd_configs:
      - role: pod
      pipeline_stages:
      - cri: {}
      - json:
          expressions:
            level: level
            msg: msg
            timestamp: timestamp
      - labels:
          level:
      - match:
          selector: '{app="nginx"}'
          stages:
          - regex:
              expression: '^(?P<remote_addr>[\w.]+) - .* "(?P<method>\w+) (?P<path>[^ ]+)'
          - labels:
              method:
              path:
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        target_label: app
      - source_labels: [__meta_kubernetes_namespace]
        target_label: namespace
      - source_labels: [__meta_kubernetes_pod_name]
        target_label: pod
```

### 4.6 LogQL 쿼리 (Loki)

```logql
# All logs from a specific pod
{namespace="production", pod="web-app-7f8b9c6d-x2k4q"}

# Error logs from all pods in production
{namespace="production"} |= "error"

# JSON structured logs with level filter
{app="api-server"} | json | level="error"

# Rate of error logs per minute
rate({namespace="production"} |= "error" [1m])

# Top 10 error messages
topk(10, sum by (msg) (rate({app="api-server"} | json | level="error" [5m])))

# Log volume by namespace
sum by (namespace) (rate({job="kubernetes-pods"}[5m]))
```

---

## 5. 분산 트레이싱

### 이론: 트레이스와 OpenTelemetry — 서비스 간 인과성

분산 시스템은 단일 사용자 요청을 많은 서비스 호출을 통해 처리합니다 — Ingress → API → Auth → DB. **분산 트레이싱**은 이 그래프를 캡처합니다 — 각 서비스가 자기 작업을 기록하는 "span"을 emit; span은 `trace_id`(요청)을 공유하고 부모 `span_id`(인과성)를 가집니다. 시각화하면 시간이 어디에 쓰였는지 보여주는 flame graph를 얻습니다.

트레이스의 세 속성:

- **기본 샘플링.** 요청당 트레이스 저장은 고볼륨 시스템에 비실용적. 전형적 샘플링 — 1% head 샘플링(루트에서 결정), 또는 tail 샘플링(느리거나 오류 트레이스 유지, 정상 폐기). 샘플링되지 않은 트레이스는 사라집니다 — 샘플링되지 않은 요청을 소급하여 트레이스할 수 없습니다.
- **서비스 간 전파에는 W3C Trace Context 표준 필요.** 체인의 모든 서비스가 `traceparent` 헤더를 읽고 써야 합니다. 전파하지 않는 한 서비스가 체인을 깨뜨립니다.
- **계측이 있어야만 유용.** 네트워크에서만 트레이스 수집은 인프라 수준 span(HTTP 요청 in, HTTP 요청 out)을 보여줍니다. 유용한 애플리케이션 span(DB 쿼리, 캐시 히트, 비즈니스 로직)은 코드 수준 계측을 필요로 합니다.

**OpenTelemetry (OTel)**가 통일입니다 — 세 기둥 모두를 다루는 벤더 중립적 SDK + 프로토콜(OTLP) + collector. 같은 라이브러리가 메트릭, 로그, 트레이스를 emit; 같은 collector가 셋 모두 수신; collector를 백엔드(Prometheus, Loki, Jaeger, Datadog, ...)로 가리킵니다. 이는 옛 "도구당 다른 SDK" 고통을 죽입니다.

쿠버네티스에서 OTel의 **자동 계측 operator**는 mutating 웹훅(12강)을 통해 Java/Python/Node 에이전트를 파드에 주입할 수 있어, 앱 코드 변경 없이 트레이스와 메트릭을 얻을 수 있습니다. Collector는 보통 DaemonSet(노드당 하나) 또는 Deployment로 실행되어 메트릭을 스크랩하고 로컬 파드에서 OTLP를 수신합니다.

최종 상태 — 앱의 한 SDK 집합이 모든 것을 emit, 한 collector 파이프라인이 각 신호를 적절한 스토어로 라우팅. 이것이 현대 아키텍처입니다 — 레거시 "신호당 한 도구" 배포는 천천히 마이그레이션 중입니다.

### 5.1 OpenTelemetry 아키텍처

```
┌──────────────────────────────────────────────────────────┐
│                   Application Pods                        │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐         │
│  │  Service A  │──│  Service B  │──│  Service C  │         │
│  │  (OTel SDK) │  │  (OTel SDK) │  │  (OTel SDK) │         │
│  └──────┬─────┘  └──────┬─────┘  └──────┬─────┘         │
│         │               │               │                │
│         └───────────────┼───────────────┘                │
│                         │ OTLP                           │
│                         ▼                                │
│              ┌──────────────────────┐                    │
│              │  OTel Collector      │                    │
│              │  (DaemonSet or       │                    │
│              │   Deployment)        │                    │
│              └──────────┬───────────┘                    │
│                         │                                │
│           ┌─────────────┼──────────────┐                 │
│           ▼             ▼              ▼                 │
│    ┌──────────┐  ┌──────────┐  ┌──────────────┐         │
│    │  Jaeger  │  │Prometheus│  │    Loki      │         │
│    │ (traces) │  │ (metrics)│  │   (logs)     │         │
│    └──────────┘  └──────────┘  └──────────────┘         │
└──────────────────────────────────────────────────────────┘
```

### 5.2 OpenTelemetry Collector

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: otel-collector
  namespace: observability
spec:
  selector:
    matchLabels:
      app: otel-collector
  template:
    metadata:
      labels:
        app: otel-collector
    spec:
      containers:
      - name: collector
        image: otel/opentelemetry-collector-contrib:0.92.0
        ports:
        - containerPort: 4317  # OTLP gRPC
        - containerPort: 4318  # OTLP HTTP
        - containerPort: 8888  # Metrics
        volumeMounts:
        - name: config
          mountPath: /etc/otelcol-contrib
        resources:
          requests:
            cpu: 200m
            memory: 256Mi
          limits:
            cpu: "1"
            memory: 1Gi
      volumes:
      - name: config
        configMap:
          name: otel-collector-config
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: otel-collector-config
  namespace: observability
data:
  config.yaml: |
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
        send_batch_size: 1024
      memory_limiter:
        check_interval: 1s
        limit_mib: 800
        spike_limit_mib: 200
      resource:
        attributes:
        - key: cluster.name
          value: production
          action: upsert

    exporters:
      otlp/jaeger:
        endpoint: jaeger-collector.observability.svc:4317
        tls:
          insecure: true
      prometheus:
        endpoint: 0.0.0.0:8889
      loki:
        endpoint: http://loki.logging.svc:3100/loki/api/v1/push

    service:
      pipelines:
        traces:
          receivers: [otlp]
          processors: [memory_limiter, batch, resource]
          exporters: [otlp/jaeger]
        metrics:
          receivers: [otlp]
          processors: [memory_limiter, batch]
          exporters: [prometheus]
        logs:
          receivers: [otlp]
          processors: [memory_limiter, batch]
          exporters: [loki]
```

### 5.3 Go 애플리케이션 계측

```go
package main

import (
    "context"
    "log"
    "net/http"
    "time"

    "go.opentelemetry.io/otel"
    "go.opentelemetry.io/otel/attribute"
    "go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
    "go.opentelemetry.io/otel/propagation"
    "go.opentelemetry.io/otel/sdk/resource"
    sdktrace "go.opentelemetry.io/otel/sdk/trace"
    semconv "go.opentelemetry.io/otel/semconv/v1.24.0"
    "go.opentelemetry.io/otel/trace"
    "go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
)

func initTracer(ctx context.Context) (*sdktrace.TracerProvider, error) {
    exporter, err := otlptracegrpc.New(ctx,
        otlptracegrpc.WithEndpoint("otel-collector.observability.svc:4317"),
        otlptracegrpc.WithInsecure(),
    )
    if err != nil {
        return nil, err
    }

    tp := sdktrace.NewTracerProvider(
        sdktrace.WithBatcher(exporter,
            sdktrace.WithBatchTimeout(5*time.Second),
        ),
        sdktrace.WithResource(resource.NewWithAttributes(
            semconv.SchemaURL,
            semconv.ServiceNameKey.String("order-service"),
            semconv.ServiceVersionKey.String("1.0.0"),
            attribute.String("environment", "production"),
        )),
        sdktrace.WithSampler(sdktrace.TraceIDRatioBased(0.1)), // Sample 10%
    )

    otel.SetTracerProvider(tp)
    otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
        propagation.TraceContext{},
        propagation.Baggage{},
    ))

    return tp, nil
}

var tracer = otel.Tracer("order-service")

func handleOrder(w http.ResponseWriter, r *http.Request) {
    ctx := r.Context()

    // Create a child span for database query
    ctx, span := tracer.Start(ctx, "process-order",
        trace.WithAttributes(
            attribute.String("order.id", r.URL.Query().Get("id")),
        ),
    )
    defer span.End()

    // Simulate database call
    _, dbSpan := tracer.Start(ctx, "db.query",
        trace.WithAttributes(
            attribute.String("db.system", "postgresql"),
            attribute.String("db.statement", "SELECT * FROM orders WHERE id = ?"),
        ),
    )
    time.Sleep(50 * time.Millisecond)
    dbSpan.End()

    w.WriteHeader(http.StatusOK)
    w.Write([]byte(`{"status": "processed"}`))
}

func main() {
    ctx := context.Background()
    tp, err := initTracer(ctx)
    if err != nil {
        log.Fatal(err)
    }
    defer tp.Shutdown(ctx)

    handler := otelhttp.NewHandler(http.HandlerFunc(handleOrder), "order-handler")
    http.Handle("/order", handler)
    log.Fatal(http.ListenAndServe(":8080", nil))
}
```

### 5.4 Jaeger 설치

```bash
# Install Jaeger operator
kubectl create namespace observability
helm repo add jaegertracing https://jaegertracing.github.io/helm-charts
helm install jaeger-operator jaegertracing/jaeger-operator \
  --namespace observability \
  --set rbac.clusterRole=true

# Create a Jaeger instance
kubectl apply -f - <<EOF
apiVersion: jaegertracing.io/v1
kind: Jaeger
metadata:
  name: jaeger
  namespace: observability
spec:
  strategy: production
  storage:
    type: elasticsearch
    options:
      es:
        server-urls: https://elasticsearch.logging.svc:9200
    secretName: jaeger-es-secret
  collector:
    replicas: 2
    resources:
      requests:
        cpu: 200m
        memory: 256Mi
  query:
    replicas: 1
EOF
```

---

## 6. 헬스 체크

### 6.1 프로브 유형

Kubernetes는 세 가지 유형의 헬스 프로브를 제공합니다:

| 프로브 | 목적 | 실패 시 효과 |
|---|---|---|
| **Liveness** | 컨테이너가 올바르게 실행 중인가? | 컨테이너 재시작 |
| **Readiness** | 컨테이너가 트래픽을 처리할 수 있는가? | Service 엔드포인트에서 제거 |
| **Startup** | 컨테이너가 시작을 완료했는가? | Liveness/readiness 프로브 일시 중지 |

### 6.2 프로브 구성

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: web-app
spec:
  containers:
  - name: web-app
    image: example.com/web-app:v1
    ports:
    - containerPort: 8080

    # Startup probe: allow slow startup (up to 5 min)
    startupProbe:
      httpGet:
        path: /healthz
        port: 8080
      initialDelaySeconds: 5
      periodSeconds: 10
      failureThreshold: 30    # 30 * 10s = 5 min max startup time

    # Liveness probe: restart if stuck
    livenessProbe:
      httpGet:
        path: /healthz
        port: 8080
      initialDelaySeconds: 0   # Starts after startup probe passes
      periodSeconds: 10
      timeoutSeconds: 5
      failureThreshold: 3      # Restart after 3 consecutive failures
      successThreshold: 1

    # Readiness probe: manage traffic
    readinessProbe:
      httpGet:
        path: /readyz
        port: 8080
      initialDelaySeconds: 0
      periodSeconds: 5
      timeoutSeconds: 3
      failureThreshold: 2      # Remove from endpoints after 2 failures
      successThreshold: 2      # Require 2 successes to be added back
```

### 6.3 프로브 메커니즘

```yaml
# HTTP GET probe
livenessProbe:
  httpGet:
    path: /healthz
    port: 8080
    httpHeaders:
    - name: Custom-Header
      value: "probe"

# TCP socket probe (useful for databases)
livenessProbe:
  tcpSocket:
    port: 5432

# Exec probe (run a command inside the container)
livenessProbe:
  exec:
    command:
    - /bin/sh
    - -c
    - pg_isready -U postgres

# gRPC probe (Kubernetes 1.27+)
livenessProbe:
  grpc:
    port: 50051
    service: "health"  # gRPC health checking protocol
```

### 6.4 헬스 체크 구현 (Go)

```go
package main

import (
    "database/sql"
    "encoding/json"
    "net/http"
    "sync/atomic"
)

var (
    ready   int32 = 0
    healthy int32 = 1
)

type HealthResponse struct {
    Status string            `json:"status"`
    Checks map[string]string `json:"checks,omitempty"`
}

func healthzHandler(db *sql.DB) http.HandlerFunc {
    return func(w http.ResponseWriter, r *http.Request) {
        if atomic.LoadInt32(&healthy) == 0 {
            w.WriteHeader(http.StatusServiceUnavailable)
            json.NewEncoder(w).Encode(HealthResponse{Status: "unhealthy"})
            return
        }

        checks := map[string]string{}

        // Check database connectivity
        if err := db.Ping(); err != nil {
            checks["database"] = "down: " + err.Error()
            w.WriteHeader(http.StatusServiceUnavailable)
            json.NewEncoder(w).Encode(HealthResponse{Status: "unhealthy", Checks: checks})
            return
        }
        checks["database"] = "up"

        w.WriteHeader(http.StatusOK)
        json.NewEncoder(w).Encode(HealthResponse{Status: "healthy", Checks: checks})
    }
}

func readyzHandler() http.HandlerFunc {
    return func(w http.ResponseWriter, r *http.Request) {
        if atomic.LoadInt32(&ready) == 0 {
            w.WriteHeader(http.StatusServiceUnavailable)
            json.NewEncoder(w).Encode(HealthResponse{Status: "not ready"})
            return
        }
        w.WriteHeader(http.StatusOK)
        json.NewEncoder(w).Encode(HealthResponse{Status: "ready"})
    }
}

// Call this after startup tasks complete
func markReady() {
    atomic.StoreInt32(&ready, 1)
}
```

### 6.5 안티패턴

| 안티패턴 | 문제 | 해결책 |
|---|---|---|
| Liveness 프로브가 의존성을 확인 | 외부 서비스 다운 시 연쇄 재시작 | 프로세스 상태만 확인, 의존성 확인하지 않음 |
| liveness와 readiness에 동일한 엔드포인트 사용 | 트래픽과 재시작을 독립적으로 제어할 수 없음 | 별도의 `/healthz`와 `/readyz` 엔드포인트 사용 |
| 느린 시작 앱에 startup 프로브 없음 | Liveness 프로브가 시작 전 컨테이너를 종료 | 넉넉한 failureThreshold의 startup 프로브 사용 |
| 공격적인 프로브 간격 | 지속적인 헬스 체크로 인한 높은 CPU | liveness에 10-30초 간격 사용 |

---

## 7. Alertmanager를 이용한 알림

### 7.1 Alertmanager 아키텍처

```
Prometheus ──(firing alerts)──▶ Alertmanager
                                    │
                              ┌─────┼─────┐
                              ▼     ▼     ▼
                           Route  Group  Silence
                              │     │     │
                              ▼     ▼     ▼
                        ┌──────────────────────┐
                        │     Receivers         │
                        │  - Slack              │
                        │  - PagerDuty          │
                        │  - Email              │
                        │  - Webhook            │
                        └──────────────────────┘
```

### 7.2 알림을 위한 PrometheusRule

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: kubernetes-alerts
  namespace: monitoring
  labels:
    release: monitoring
spec:
  groups:
  - name: kubernetes.pod.alerts
    rules:
    - alert: PodCrashLooping
      expr: |
        increase(kube_pod_container_status_restarts_total[1h]) > 5
      for: 10m
      labels:
        severity: warning
      annotations:
        summary: "Pod {{ $labels.namespace }}/{{ $labels.pod }} is crash looping"
        description: "Pod has restarted {{ $value }} times in the last hour."
        runbook_url: "https://runbooks.example.com/pod-crash-loop"

    - alert: PodNotReady
      expr: |
        kube_pod_status_ready{condition="true"} == 0
      for: 15m
      labels:
        severity: warning
      annotations:
        summary: "Pod {{ $labels.namespace }}/{{ $labels.pod }} has been not ready for 15 minutes"

    - alert: DeploymentReplicasMismatch
      expr: |
        kube_deployment_spec_replicas != kube_deployment_status_ready_replicas
      for: 10m
      labels:
        severity: critical
      annotations:
        summary: "Deployment {{ $labels.namespace }}/{{ $labels.deployment }} has mismatched replicas"
        description: "Expected {{ $value }} replicas but only {{ $labels.ready_replicas }} are ready."

  - name: kubernetes.node.alerts
    rules:
    - alert: NodeHighCPU
      expr: |
        100 - (avg by(instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 85
      for: 10m
      labels:
        severity: warning
      annotations:
        summary: "Node {{ $labels.instance }} CPU usage is above 85%"

    - alert: NodeHighMemory
      expr: |
        (1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100 > 90
      for: 10m
      labels:
        severity: critical
      annotations:
        summary: "Node {{ $labels.instance }} memory usage is above 90%"

    - alert: NodeDiskPressure
      expr: |
        (1 - node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"}) * 100 > 85
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "Node {{ $labels.instance }} disk usage is above 85%"
```

### 7.3 Alertmanager 구성

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: alertmanager-config
  namespace: monitoring
stringData:
  alertmanager.yaml: |
    global:
      resolve_timeout: 5m
      slack_api_url: 'https://hooks.slack.com/services/T00/B00/XXX'

    route:
      receiver: 'default'
      group_by: ['alertname', 'namespace']
      group_wait: 30s
      group_interval: 5m
      repeat_interval: 4h
      routes:
      - receiver: 'critical-pagerduty'
        match:
          severity: critical
        group_wait: 10s
        repeat_interval: 1h
      - receiver: 'warning-slack'
        match:
          severity: warning
        repeat_interval: 4h

    receivers:
    - name: 'default'
      slack_configs:
      - channel: '#alerts-default'
        title: '{{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}\n{{ end }}'

    - name: 'critical-pagerduty'
      pagerduty_configs:
      - service_key: '<pagerduty-service-key>'
        severity: 'critical'
        description: '{{ .GroupLabels.alertname }}: {{ .CommonAnnotations.summary }}'

    - name: 'warning-slack'
      slack_configs:
      - channel: '#alerts-warning'
        title: '[{{ .Status | toUpper }}] {{ .GroupLabels.alertname }}'
        text: >-
          {{ range .Alerts }}
          *Alert:* {{ .Annotations.summary }}
          *Namespace:* {{ .Labels.namespace }}
          *Description:* {{ .Annotations.description }}
          {{ end }}

    inhibit_rules:
    - source_match:
        severity: 'critical'
      target_match:
        severity: 'warning'
      equal: ['alertname', 'namespace']
```

---

## 8. 디버깅 기법

### 8.1 Pod 디버깅 플로우차트

```
Pod issue detected
    │
    ├── Status: Pending?
    │       └── kubectl describe pod → Check Events
    │           ├── Insufficient resources → Scale nodes or reduce requests
    │           ├── Unschedulable → Check taints/tolerations, node selector
    │           └── Image pull error → Check image name, pull secrets
    │
    ├── Status: CrashLoopBackOff?
    │       └── kubectl logs <pod> --previous → Check crash logs
    │           ├── OOMKilled → Increase memory limits
    │           ├── Application error → Fix code
    │           └── Missing config → Check ConfigMap/Secret mounts
    │
    ├── Status: Running but unhealthy?
    │       └── kubectl describe pod → Check probe failures
    │           ├── Liveness failing → Check /healthz endpoint
    │           ├── Readiness failing → Check dependencies
    │           └── High restart count → Check resource limits
    │
    └── Performance issue?
            └── kubectl top pod → Check resource usage
                ├── High CPU → Profile application, scale out
                └── High memory → Check for leaks, increase limits
```

### 8.2 필수 디버깅 명령어

```bash
# Pod diagnostics
kubectl get pods -o wide                       # Show node placement
kubectl describe pod <pod>                     # Full pod details and events
kubectl logs <pod> -c <container>              # Container logs
kubectl logs <pod> --previous                  # Previous container logs (crash)
kubectl logs <pod> --tail=100 -f               # Follow last 100 lines

# Resource usage
kubectl top pods --sort-by=cpu                 # Sort by CPU
kubectl top pods --sort-by=memory              # Sort by memory
kubectl top pods --containers                  # Per-container breakdown

# Network debugging
kubectl exec -it <pod> -- curl localhost:8080/healthz
kubectl exec -it <pod> -- nslookup kubernetes.default
kubectl exec -it <pod> -- wget -qO- http://service-name:port/path

# Events (cluster-wide issues)
kubectl get events --sort-by=.lastTimestamp -A
kubectl get events --field-selector type=Warning -A
```

### 8.3 임시 디버그 컨테이너(Ephemeral Debug Container)

```bash
# Attach a debug container to a running pod (without restarting it)
kubectl debug -it <pod> --image=busybox --target=<container>

# Create a copy of the pod with a debug container
kubectl debug <pod> -it --copy-to=debug-pod --container=debug --image=ubuntu

# Debug a node
kubectl debug node/<node-name> -it --image=ubuntu

# Use nicolaka/netshoot for network debugging
kubectl debug -it <pod> --image=nicolaka/netshoot --target=<container>
# Inside the debug container:
# tcpdump -i eth0 port 8080
# ss -tlnp
# curl -v http://service:port
```

### 8.4 일반적인 문제와 해결책

| 증상 | 진단 | 해결책 |
|---|---|---|
| `ImagePullBackOff` | `kubectl describe pod` | 이미지 이름 수정, imagePullSecret 추가 |
| `CrashLoopBackOff` | `kubectl logs --previous` | 애플리케이션 크래시 수정, 구성 확인 |
| `OOMKilled` | `kubectl describe pod` (last state) | 메모리 제한 증가 |
| `Pending` (이벤트 없음) | `kubectl describe pod` | 클러스터 리소스 부족 |
| `Evicted` | `kubectl describe pod` | 노드에 디스크/메모리 압박 |
| DNS 해결 실패 | `kubectl exec -- nslookup` | CoreDNS Pod와 ConfigMap 확인 |
| 서비스에 연결 불가 | `kubectl get endpoints` | 셀렉터 레이블, Pod readiness 확인 |

---

## 연습문제

### 연습문제 1: Prometheus 모니터링 설정

커스텀 애플리케이션을 모니터링하기 위한 완전한 매니페스트를 작성하세요: (a) `app: payment-service` 레이블이 있는 Pod에서 `metrics` 포트로 30초마다 메트릭을 스크래핑하는 ServiceMonitor, (b) 세 가지 알림을 가진 PrometheusRule: 높은 오류율(5분 동안 >5%), 높은 지연(10분 동안 p99 > 500ms), Pod 재시작(1시간에 >3회), (c) 요청 비율, 오류율, 지연 백분위수를 보여주는 Grafana 대시보드용 PromQL 쿼리.

<details>
<summary>정답 보기</summary>

```yaml
# (a) ServiceMonitor
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: payment-service-monitor
  namespace: production
  labels:
    release: monitoring
spec:
  selector:
    matchLabels:
      app: payment-service
  endpoints:
  - port: metrics
    path: /metrics
    interval: 30s
    scrapeTimeout: 10s
  namespaceSelector:
    matchNames:
    - production
---
# (b) PrometheusRule
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: payment-service-alerts
  namespace: monitoring
  labels:
    release: monitoring
spec:
  groups:
  - name: payment-service
    rules:
    - alert: PaymentHighErrorRate
      expr: |
        sum(rate(http_requests_total{app="payment-service", status=~"5.."}[5m]))
        /
        sum(rate(http_requests_total{app="payment-service"}[5m])) > 0.05
      for: 5m
      labels:
        severity: critical
        service: payment
      annotations:
        summary: "Payment service error rate is above 5%"
        description: "Current error rate: {{ $value | humanizePercentage }}"

    - alert: PaymentHighLatency
      expr: |
        histogram_quantile(0.99,
          sum(rate(http_request_duration_seconds_bucket{app="payment-service"}[5m])) by (le)
        ) > 0.5
      for: 10m
      labels:
        severity: warning
        service: payment
      annotations:
        summary: "Payment service p99 latency above 500ms"
        description: "Current p99: {{ $value | humanizeDuration }}"

    - alert: PaymentPodRestarts
      expr: |
        increase(kube_pod_container_status_restarts_total{
          namespace="production",
          pod=~"payment-service.*"
        }[1h]) > 3
      for: 5m
      labels:
        severity: warning
        service: payment
      annotations:
        summary: "Payment service pod {{ $labels.pod }} restarting frequently"
```

```promql
# (c) Grafana dashboard queries

# Request rate
sum(rate(http_requests_total{app="payment-service"}[5m])) by (method, path)

# Error rate percentage
sum(rate(http_requests_total{app="payment-service", status=~"5.."}[5m]))
/
sum(rate(http_requests_total{app="payment-service"}[5m])) * 100

# Latency percentiles
histogram_quantile(0.50, sum(rate(http_request_duration_seconds_bucket{app="payment-service"}[5m])) by (le))
histogram_quantile(0.90, sum(rate(http_request_duration_seconds_bucket{app="payment-service"}[5m])) by (le))
histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket{app="payment-service"}[5m])) by (le))
```

</details>

### 연습문제 2: 로깅 파이프라인

Grafana Loki를 사용한 로깅 파이프라인을 설계하고 구성하세요: (a) 모든 Pod에서 로그를 수집하고, JSON 형식 로그를 파싱하여 `level`, `msg`, `trace_id` 필드를 레이블로 추출하는 Promtail 구성 작성, (b) 다음을 찾는 5개의 LogQL 쿼리 작성: 지난 1시간의 모든 오류, 느린 요청(>1s), 특정 트레이스 ID의 로그, 서비스별 오류 비율, 상위 5개 오류 메시지, (c) Grafana에서 로그 기반 알림을 설정하는 방법 설명.

<details>
<summary>정답 보기</summary>

**(a) Promtail 구성:**

```yaml
scrape_configs:
- job_name: kubernetes-pods
  kubernetes_sd_configs:
  - role: pod
  pipeline_stages:
  - cri: {}
  - json:
      expressions:
        level: level
        msg: msg
        trace_id: trace_id
        duration_ms: duration_ms
  - labels:
      level:
  - template:
      source: level
      template: '{{ ToLower .Value }}'
  - output:
      source: msg
  relabel_configs:
  - source_labels: [__meta_kubernetes_pod_label_app]
    target_label: app
  - source_labels: [__meta_kubernetes_namespace]
    target_label: namespace
  - source_labels: [__meta_kubernetes_pod_name]
    target_label: pod
  - source_labels: [__meta_kubernetes_pod_container_name]
    target_label: container
```

**(b) LogQL 쿼리:**

```logql
# 1. All errors in the last hour
{namespace="production"} | json | level="error" | line_format "{{.msg}}"

# 2. Slow requests (duration > 1000ms)
{namespace="production"} | json | duration_ms > 1000

# 3. Logs for a specific trace ID
{namespace="production"} |= "abc123-trace-id-here"

# 4. Rate of errors per service (per minute)
sum by (app) (rate({namespace="production"} | json | level="error" [1m]))

# 5. Top 5 error messages
topk(5, sum by (msg) (count_over_time({namespace="production"} | json | level="error" [1h])))
```

**(c) Grafana에서의 로그 기반 알림:** Grafana에서 데이터 소스를 Loki로 설정한 새 Alert Rule을 생성합니다. `sum(rate({namespace="production"} | json | level="error" [5m])) > 10`과 같은 LogQL 메트릭 쿼리를 조건으로 사용합니다. 평가 간격(예: 1분마다)과 보류 기간(예: 5분)을 설정합니다. 연락처에 알림 채널(Slack, PagerDuty)을 구성합니다. 이렇게 하면 오류 로그 비율이 5분 연속으로 초당 10을 초과할 때 알림이 발동됩니다.

</details>

### 연습문제 3: 분산 트레이싱 구현

두 서비스 트레이싱 설정을 위한 Go 코드를 작성하세요: (a) Service A는 HTTP 요청을 받고, 루트 스팬(root span)을 생성하며, Service B에 HTTP 호출을 하고, 응답을 반환합니다, (b) Service B는 요청을 받고, 자식 스팬(child span)을 생성하며, 데이터베이스를 쿼리(시뮬레이션)하고, 데이터를 반환합니다, (c) 두 서비스 모두 OpenTelemetry Collector로 트레이스를 내보냅니다. OTel Collector ConfigMap과 Jaeger 배포를 포함하세요.

<details>
<summary>정답 보기</summary>

```go
// Service A - api-gateway
package main

import (
    "context"
    "io"
    "log"
    "net/http"
    "time"

    "go.opentelemetry.io/otel"
    "go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
    "go.opentelemetry.io/otel/propagation"
    "go.opentelemetry.io/otel/sdk/resource"
    sdktrace "go.opentelemetry.io/otel/sdk/trace"
    semconv "go.opentelemetry.io/otel/semconv/v1.24.0"
    "go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
)

var httpClient = &http.Client{Transport: otelhttp.NewTransport(http.DefaultTransport)}

func initTracer(ctx context.Context, serviceName string) *sdktrace.TracerProvider {
    exp, _ := otlptracegrpc.New(ctx,
        otlptracegrpc.WithEndpoint("otel-collector:4317"),
        otlptracegrpc.WithInsecure(),
    )
    tp := sdktrace.NewTracerProvider(
        sdktrace.WithBatcher(exp),
        sdktrace.WithResource(resource.NewWithAttributes(
            semconv.SchemaURL, semconv.ServiceNameKey.String(serviceName),
        )),
    )
    otel.SetTracerProvider(tp)
    otel.SetTextMapPropagator(propagation.TraceContext{})
    return tp
}

func handleRequest(w http.ResponseWriter, r *http.Request) {
    ctx := r.Context()
    tracer := otel.Tracer("api-gateway")
    ctx, span := tracer.Start(ctx, "handle-request")
    defer span.End()

    req, _ := http.NewRequestWithContext(ctx, "GET", "http://user-service:8081/user", nil)
    resp, err := httpClient.Do(req)
    if err != nil {
        http.Error(w, err.Error(), 500)
        return
    }
    defer resp.Body.Close()
    body, _ := io.ReadAll(resp.Body)
    w.Write(body)
}

func main() {
    ctx := context.Background()
    tp := initTracer(ctx, "api-gateway")
    defer tp.Shutdown(ctx)

    http.Handle("/api", otelhttp.NewHandler(http.HandlerFunc(handleRequest), "api"))
    log.Fatal(http.ListenAndServe(":8080", nil))
}
```

```go
// Service B - user-service
package main

import (
    "context"
    "log"
    "net/http"
    "time"

    "go.opentelemetry.io/otel"
    "go.opentelemetry.io/otel/attribute"
    "go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
)

func handleUser(w http.ResponseWriter, r *http.Request) {
    ctx := r.Context()
    tracer := otel.Tracer("user-service")

    _, dbSpan := tracer.Start(ctx, "db.query.user",
        // Add semantic attributes
    )
    dbSpan.SetAttributes(
        attribute.String("db.system", "postgresql"),
        attribute.String("db.statement", "SELECT * FROM users WHERE id=$1"),
    )
    time.Sleep(30 * time.Millisecond) // Simulate DB call
    dbSpan.End()

    w.Write([]byte(`{"id": 1, "name": "Alice"}`))
}

func main() {
    ctx := context.Background()
    tp := initTracer(ctx, "user-service") // Same initTracer as Service A
    defer tp.Shutdown(ctx)

    http.Handle("/user", otelhttp.NewHandler(http.HandlerFunc(handleUser), "user"))
    log.Fatal(http.ListenAndServe(":8081", nil))
}
```

```yaml
# OTel Collector ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: otel-collector-config
  namespace: observability
data:
  config.yaml: |
    receivers:
      otlp:
        protocols:
          grpc:
            endpoint: 0.0.0.0:4317
    processors:
      batch:
        timeout: 5s
    exporters:
      otlp/jaeger:
        endpoint: jaeger-collector:4317
        tls:
          insecure: true
    service:
      pipelines:
        traces:
          receivers: [otlp]
          processors: [batch]
          exporters: [otlp/jaeger]
---
# Jaeger all-in-one (for development)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: jaeger
  namespace: observability
spec:
  replicas: 1
  selector:
    matchLabels:
      app: jaeger
  template:
    metadata:
      labels:
        app: jaeger
    spec:
      containers:
      - name: jaeger
        image: jaegertracing/all-in-one:1.53
        ports:
        - containerPort: 16686  # UI
        - containerPort: 4317   # OTLP gRPC
        - containerPort: 14268  # HTTP collector
        env:
        - name: COLLECTOR_OTLP_ENABLED
          value: "true"
```

</details>

### 연습문제 4: 헬스 체크 설계

다음 특성을 가진 Java Spring Boot 애플리케이션을 위한 헬스 체크를 설계하세요: (a) 시작하는 데 90초 소요 (ML 모델을 메모리에 로드), (b) PostgreSQL과 Redis에 의존, (c) 우아한 종료(graceful shutdown) 중에는 트래픽을 받지 않아야 함. 세 가지 프로브 유형이 모두 포함된 완전한 pod spec을 작성하세요. 프로브 매개변수 선택의 근거를 설명하세요. 또한 연쇄 장애를 일으키지 않고 의존성 상태를 확인하는 커스텀 헬스 체크 엔드포인트의 Go 코드를 작성하세요.

<details>
<summary>정답 보기</summary>

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: ml-service
spec:
  terminationGracePeriodSeconds: 30
  containers:
  - name: ml-service
    image: example.com/ml-service:v1
    ports:
    - containerPort: 8080
    lifecycle:
      preStop:
        exec:
          command: ["/bin/sh", "-c", "sleep 5"]  # Allow in-flight requests to complete

    # Startup: allow up to 120s (12 * 10s) for ML model loading
    startupProbe:
      httpGet:
        path: /actuator/health/startup
        port: 8080
      initialDelaySeconds: 10
      periodSeconds: 10
      failureThreshold: 12
      timeoutSeconds: 5

    # Liveness: only check if the process is healthy (NOT dependencies)
    livenessProbe:
      httpGet:
        path: /actuator/health/liveness
        port: 8080
      periodSeconds: 15
      failureThreshold: 3
      timeoutSeconds: 5

    # Readiness: check if the service can handle traffic (includes dependencies)
    readinessProbe:
      httpGet:
        path: /actuator/health/readiness
        port: 8080
      periodSeconds: 5
      failureThreshold: 2
      successThreshold: 2
      timeoutSeconds: 3
```

**매개변수 근거:**
- Startup 프로브: `failureThreshold: 12`와 `periodSeconds: 10`으로 ML 모델 로딩에 120초를 줍니다(90초 필요 + 30초 여유).
- Liveness 프로브: `periodSeconds: 15`로 너무 공격적이지 않습니다. PostgreSQL이 잠시 사용 불가능할 때 연쇄 재시작을 방지하기 위해 프로세스 상태만 확인하고 의존성은 확인하지 않습니다.
- Readiness 프로브: 의존성 장애에 빠르게 대응하기 위해 `periodSeconds: 5`. 플래핑을 방지하기 위해 `successThreshold: 2`.

```go
// Custom health check that avoids cascading failures
func readinessHandler(db *sql.DB, redisClient *redis.Client) http.HandlerFunc {
    return func(w http.ResponseWriter, r *http.Request) {
        ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
        defer cancel()

        checks := map[string]string{}
        healthy := true

        // Check PostgreSQL with timeout
        dbCh := make(chan error, 1)
        go func() { dbCh <- db.PingContext(ctx) }()
        select {
        case err := <-dbCh:
            if err != nil {
                checks["postgresql"] = "down"
                healthy = false
            } else {
                checks["postgresql"] = "up"
            }
        case <-ctx.Done():
            checks["postgresql"] = "timeout"
            healthy = false
        }

        // Check Redis with timeout
        redisCh := make(chan error, 1)
        go func() { redisCh <- redisClient.Ping(ctx).Err() }()
        select {
        case err := <-redisCh:
            if err != nil {
                checks["redis"] = "down"
                healthy = false
            } else {
                checks["redis"] = "up"
            }
        case <-ctx.Done():
            checks["redis"] = "timeout"
            healthy = false
        }

        resp := HealthResponse{Checks: checks}
        if healthy {
            resp.Status = "ready"
            w.WriteHeader(http.StatusOK)
        } else {
            resp.Status = "not ready"
            w.WriteHeader(http.StatusServiceUnavailable)
        }
        json.NewEncoder(w).Encode(resp)
    }
}

// Liveness handler -- only checks process health, never dependencies
func livenessHandler() http.HandlerFunc {
    return func(w http.ResponseWriter, r *http.Request) {
        // Only check that the process can respond
        w.WriteHeader(http.StatusOK)
        json.NewEncoder(w).Encode(HealthResponse{Status: "alive"})
    }
}
```

</details>

### 연습문제 5: 프로덕션 디버깅 시나리오

10개 Pod를 가진 프로덕션 배포가 간헐적인 503 오류를 경험하고 있습니다. 약 20%의 요청이 실패합니다. 단계별 디버깅 절차를 작성하세요: (a) 어떤 Pod가 실패하는지 식별하는 명령어, (b) 문제가 네트워킹, 애플리케이션, 리소스 관련인지 확인하는 방법, (c) 실행 중인 Pod를 검사하기 위해 임시 디버그 컨테이너를 사용하는 방법, (d) 근본 원인을 찾기 위해 로그, 메트릭, 트레이스를 상관시키는 방법. 모든 kubectl 명령어와 PromQL 쿼리를 제공하세요.

<details>
<summary>정답 보기</summary>

```bash
# Step 1: Identify failing pods
# Check pod status and restarts
kubectl get pods -l app=my-app -o wide
kubectl get pods -l app=my-app -o custom-columns="NAME:.metadata.name,STATUS:.status.phase,RESTARTS:.status.containerStatuses[0].restartCount,NODE:.spec.nodeName,READY:.status.containerStatuses[0].ready"

# Check endpoints -- which pods are ready to receive traffic?
kubectl get endpoints my-app-service
kubectl get pods -l app=my-app -o jsonpath='{range .items[*]}{.metadata.name} ready={.status.conditions[?(@.type=="Ready")].status}{"\n"}{end}'

# Step 2: Check resource usage
kubectl top pods -l app=my-app --sort-by=cpu
kubectl top pods -l app=my-app --sort-by=memory

# Check for OOM events
kubectl get events --field-selector reason=OOMKilling -A

# Step 3: Check recent logs from failing pods
for pod in $(kubectl get pods -l app=my-app -o jsonpath='{.items[*].metadata.name}'); do
  echo "=== $pod ==="
  kubectl logs "$pod" --tail=20 | grep -i error
done

# Step 4: Check networking
# Test connectivity from a debug pod to each application pod
kubectl run debug --image=nicolaka/netshoot --rm -it -- bash
# Inside: for i in $(seq 0 9); do curl -s -o /dev/null -w "%{http_code}\n" http://my-app-$i.my-app-headless:8080/healthz; done

# Step 5: Ephemeral debug container
kubectl debug -it <failing-pod> --image=nicolaka/netshoot --target=my-app
# Inside the debug container:
# ss -tlnp          # Check listening ports
# curl localhost:8080/healthz   # Test local health
# tcpdump -i eth0 port 8080 -c 20  # Capture traffic

# Step 6: Check node health for affected pods
kubectl describe node <node-of-failing-pods> | grep -A5 Conditions

# Step 7: Check readiness probe results
kubectl describe pod <failing-pod> | grep -A10 "Readiness"
```

```promql
# PromQL queries to correlate

# Per-pod error rate (identify the 20% that are failing)
sum by (pod) (rate(http_requests_total{app="my-app", status=~"5.."}[5m]))
/
sum by (pod) (rate(http_requests_total{app="my-app"}[5m]))

# Per-node error rate (check if issue is node-specific)
sum by (node) (rate(http_requests_total{app="my-app", status=~"5.."}[5m]))

# Memory usage approaching limits
container_memory_working_set_bytes{container="my-app"}
/
container_spec_memory_limit_bytes{container="my-app"}

# CPU throttling
rate(container_cpu_cfs_throttled_seconds_total{container="my-app"}[5m])

# Network errors
rate(container_network_receive_errors_total{pod=~"my-app.*"}[5m])
```

```logql
# Loki queries for log correlation

# Errors from the specific failing pods
{app="my-app", pod=~"my-app-pod-abc.*"} |= "error"

# Correlate with trace ID from a failed request
{app="my-app"} | json | status >= 500 | line_format "trace={{.trace_id}} status={{.status}} msg={{.msg}}"
```

**20% 실패율의 일반적인 근본 원인:**
1. **10개 중 2개 Pod가 성능이 저하된 노드에 있음** -- 노드별 오류율 확인
2. **Pod가 메모리 제한에 도달** -- OOM 이벤트와 메모리 사용률 비율 확인
3. **Readiness 프로브는 통과하지만 앱이 부분적으로 저하** -- 프로브 엔드포인트 vs 실제 트래픽 경로 확인
4. **연결 풀 고갈** -- 활성 연결 메트릭과 데이터베이스 로그 확인

</details>

---

**이전**: [오토스케일링](./13_Autoscaling.md) | **다음**: [멀티 클러스터](./15_Multi_Cluster.md)
