# 로깅 인프라

**이전**: [모니터링과 알림](./10_Monitoring_and_Alerting.md) | **다음**: [분산 트레이싱](./12_Distributed_Tracing.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. DevOps에서 중앙 집중식 로깅의 역할과 분산 시스템에서 로그 집계가 필수적인 이유를 설명할 수 있습니다
2. ELK 스택(Elasticsearch, Logstash, Kibana)을 사용하여 로깅 파이프라인을 설계하고 각 구성 요소의 책임을 이해할 수 있습니다
3. ELK 스택과 Loki/Grafana 스택을 비교하고 규모와 비용 제약에 따라 올바른 솔루션을 선택할 수 있습니다
4. JSON 형식과 일관된 필드 규칙을 사용하여 애플리케이션에서 구조화된 로깅을 구현할 수 있습니다
5. 프로덕션 환경에 대한 로그 레벨, 보존 정책, 인덱스 수명 주기 관리를 구성할 수 있습니다
6. 인시던트 조사와 운영 가시성을 위한 효과적인 로그 쿼리와 대시보드를 구축할 수 있습니다

---

로그는 시스템에서 발생하는 모든 것의 서술 기록입니다. 메트릭이 무엇이 잘못되었는지(오류율이 5%로 급등)를 알려주는 반면, 로그는 왜 그런지(마이그레이션이 테이블을 잠갔기 때문에 데이터베이스 연결 풀이 소진됨)를 알려줍니다. 분산 시스템에서는 수십 개의 서비스에서 나오는 로그를 단일 검색 가능한 저장소에 집계해야 합니다. 그렇지 않으면 디버깅은 개별 머신에 SSH로 접속하는 절망적인 작업이 됩니다. 이 레슨에서는 두 가지 주요 오픈소스 로깅 스택인 ELK와 Loki, 그리고 구조화된 로깅 실천과 보존 전략을 다룹니다.

> **비유 -- 비행 데이터 기록 장치**: 로그는 비행기의 블랙박스 비행 기록 장치와 같습니다. 모든 이벤트가 타임스탬프와 함께 순서대로 기록되며, 무언가 잘못되면 조사관은 기록에서 이벤트의 연쇄를 재구성합니다. 블랙박스(중앙 집중식 로깅) 없이는 조사관이 승객을 한 명씩 면담(각 서버에 SSH)해야 합니다 -- 느리고, 불완전하며, 신뢰할 수 없습니다.

## 1. 중앙 집중식 로깅이 필요한 이유

### 1.1 로컬 로그의 문제

분산 시스템에서 개별 호스트에 흩어진 로그는 심각한 문제를 만듭니다:

```
┌─────────────────────────────────────────────────────────────┐
│  중앙 집중식 로깅 없이                                        │
│                                                              │
│  Service A (host-1)     Service B (host-2)     Service C     │
│  /var/log/app.log       /var/log/app.log     (host-3, 4, 5) │
│                                                              │
│  문제점:                                                     │
│  1. 요청 X의 로그가 어느 호스트에 있는가?                      │
│  2. 컨테이너가 종료되면 → 로그가 손실됨                       │
│  3. 서비스 간 이벤트를 상관시킬 수 없음                       │
│  4. 모든 로그에 대한 전문 검색이 불가능                        │
│  5. 로그 패턴에 대한 알림을 설정할 수 없음                    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  중앙 집중식 로깅 사용 시                                     │
│                                                              │
│  Service A ──┐                                               │
│  Service B ──┼──→ 로그 집계기 ──→ 검색 및 시각화              │
│  Service C ──┘    (단일 저장소)   (단일 대시보드)              │
│                                                              │
│  이점:                                                       │
│  1. 하나의 인터페이스에서 모든 로그 검색                       │
│  2. 요청 ID를 사용하여 서비스 간 이벤트 상관                   │
│  3. 컨테이너 재시작과 호스트 장애에서도 유지                   │
│  4. 오류 패턴에 대한 알림                                     │
│  5. 감사 및 보존 요구사항 준수                                │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 로그 레벨

표준 로그 레벨은 심각도 계층 구조를 따릅니다:

| 레벨 | 숫자 | 목적 | 예시 |
|------|------|------|------|
| **TRACE** | 5 | 세밀한 디버깅 | 함수 진입/종료, 변수 값 |
| **DEBUG** | 10 | 진단 정보 | SQL 쿼리 세부사항, 캐시 히트/미스 |
| **INFO** | 20 | 정상 운영 | 요청 처리 완료, 작업 완료, 사용자 로그인 |
| **WARNING** | 30 | 예상치 못했지만 처리됨 | 재시도 성공, 더 이상 사용되지 않는 API 호출 |
| **ERROR** | 40 | 작업 실패 | 데이터베이스 연결 실패, API 호출이 500 반환 |
| **CRITICAL/FATAL** | 50 | 시스템 사용 불가 | 메모리 부족, 포트 바인딩 불가 |

**프로덕션 가이드라인:**
- 기본 로그 레벨을 `INFO`로 설정합니다
- `DEBUG`는 개발 환경이나 인시던트 중 일시적으로만 사용합니다
- 어떤 레벨에서도 민감한 데이터(비밀번호, 토큰, PII)를 절대 로깅하지 않습니다
- 모든 `ERROR` 로그는 조치 가능해야 합니다 -- 조치할 수 없다면 `WARNING`이어야 합니다

---

## 2. 구조화된 로깅

### 2.1 비구조화 vs 구조화 로그

**비구조화 (파싱에 불리함):**
```
2024-03-15 14:23:45 INFO User john@example.com logged in from 192.168.1.100 using Chrome
```

**구조화 JSON (기계 파싱 가능):**
```json
{
  "timestamp": "2024-03-15T14:23:45.123Z",
  "level": "INFO",
  "logger": "auth.service",
  "message": "User logged in",
  "user_email": "john@example.com",
  "source_ip": "192.168.1.100",
  "user_agent": "Chrome/122.0",
  "request_id": "req-abc123",
  "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
  "service": "auth-service",
  "environment": "production"
}
```

### 2.2 Python에서 구조화된 로깅

```python
import structlog
import logging

# Configure structlog
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.ConsoleRenderer()        # Human-readable in dev
        # structlog.processors.JSONRenderer()  # JSON in production
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

log = structlog.get_logger()

# Usage
log.info("user_login", user_email="john@example.com", source_ip="192.168.1.100")
log.error("payment_failed", order_id="ord-456", reason="insufficient_funds", amount=99.99)

# Bind context for the duration of a request
log = log.bind(request_id="req-abc123", service="auth-service")
log.info("processing_request")    # Automatically includes request_id and service
log.info("request_complete", duration_ms=45)
```

### 2.3 Go에서 구조화된 로깅

```go
package main

import (
    "go.uber.org/zap"
)

func main() {
    // Production logger (JSON output)
    logger, _ := zap.NewProduction()
    defer logger.Sync()

    // Structured fields
    logger.Info("user_login",
        zap.String("user_email", "john@example.com"),
        zap.String("source_ip", "192.168.1.100"),
        zap.String("request_id", "req-abc123"),
    )

    // With context (child logger)
    reqLogger := logger.With(
        zap.String("request_id", "req-abc123"),
        zap.String("service", "auth-service"),
    )
    reqLogger.Info("processing_request")
    reqLogger.Info("request_complete", zap.Int("duration_ms", 45))
}
```

### 2.4 권장 표준 필드

| 필드 | 유형 | 설명 |
|------|------|------|
| `timestamp` | ISO 8601 | UTC 기준 이벤트 시간 |
| `level` | string | 로그 레벨 (INFO, ERROR 등) |
| `message` | string | 사람이 읽을 수 있는 이벤트 설명 |
| `service` | string | 서비스 이름 |
| `environment` | string | prod, staging, dev |
| `request_id` | string | 고유 요청 식별자 |
| `trace_id` | string | 분산 트레이스 ID (트레이스와의 상관관계용) |
| `user_id` | string | 인증된 사용자 (해당되는 경우) |
| `duration_ms` | number | 작업 기간 |
| `error` | string | 오류 메시지 (ERROR 레벨용) |
| `stack_trace` | string | 스택 트레이스 (ERROR 레벨용) |

---

## 3. ELK 스택 (Elasticsearch, Logstash, Kibana)

### 3.1 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────────┐
│                       ELK Stack Architecture                     │
│                                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │ App Logs │──→│ Filebeat │──→│  Kafka /  │──→│ Logstash │    │
│  │ (stdout) │   │ (shipper)│   │  Redis    │   │(transform│    │
│  └──────────┘   └──────────┘   │ (buffer)  │   │  & enrich│    │
│                                └──────────┘   └─────┬────┘    │
│  ┌──────────┐   ┌──────────┐                        │          │
│  │Container │──→│Fluentd / │────────────────────────┘          │
│  │  Logs    │   │Fluent Bit│                        │          │
│  └──────────┘   └──────────┘                        ▼          │
│                                              ┌──────────┐      │
│                                              │Elastic-  │      │
│                                              │search    │      │
│                                              │(store &  │      │
│                                              │ index)   │      │
│                                              └─────┬────┘      │
│                                                    │           │
│                                                    ▼           │
│                                              ┌──────────┐      │
│                                              │  Kibana  │      │
│                                              │(visualize│      │
│                                              │& search) │      │
│                                              └──────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 구성 요소 역할

| 구성 요소 | 역할 | 주요 기능 |
|----------|------|----------|
| **Beats (Filebeat)** | 경량 로그 전달자 | 로그 파일 추적, Logstash/Elasticsearch로 전달, 백프레셔 처리 |
| **Logstash** | 로그 처리 파이프라인 | 로그 파싱, 변환, 보강, 필터링; 플러그인 생태계 |
| **Elasticsearch** | 검색 및 분석 엔진 | 전문 검색, 역 인덱스, 분산형, 거의 실시간 |
| **Kibana** | 시각화 및 탐색 | 대시보드, 로그 탐색 (Discover), 알림 |

### 3.3 Filebeat 구성

```yaml
# filebeat.yml
filebeat.inputs:
  - type: log
    enabled: true
    paths:
      - /var/log/app/*.log
    json.keys_under_root: true
    json.add_error_key: true
    fields:
      service: "webapp"
      environment: "production"
    fields_under_root: true

  - type: container
    paths:
      - /var/lib/docker/containers/*/*.log
    processors:
      - add_kubernetes_metadata: ~

output.logstash:
  hosts: ["logstash:5044"]
  loadbalance: true

# Or send directly to Elasticsearch (skip Logstash)
# output.elasticsearch:
#   hosts: ["elasticsearch:9200"]
#   index: "app-logs-%{+yyyy.MM.dd}"
```

### 3.4 Logstash 파이프라인

```ruby
# logstash/pipeline/main.conf

input {
  beats {
    port => 5044
  }
}

filter {
  # Parse JSON logs
  if [message] =~ /^\{/ {
    json {
      source => "message"
    }
  }

  # Parse non-JSON logs with grok
  if ![level] {
    grok {
      match => {
        "message" => "%{TIMESTAMP_ISO8601:timestamp} %{LOGLEVEL:level} %{GREEDYDATA:log_message}"
      }
    }
  }

  # Parse timestamp
  date {
    match => ["timestamp", "ISO8601", "yyyy-MM-dd HH:mm:ss"]
    target => "@timestamp"
  }

  # GeoIP lookup for IP addresses
  if [source_ip] {
    geoip {
      source => "source_ip"
      target => "geoip"
    }
  }

  # Remove sensitive fields
  mutate {
    remove_field => ["password", "token", "authorization"]
  }

  # Add derived fields
  if [level] == "ERROR" or [level] == "CRITICAL" {
    mutate {
      add_tag => ["error"]
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "app-logs-%{+yyyy.MM.dd}"
    # ILM (Index Lifecycle Management) policy
    ilm_rollover_alias => "app-logs"
    ilm_pattern => "{now/d}-000001"
    ilm_policy => "app-logs-policy"
  }

  # Also send errors to a dedicated index
  if "error" in [tags] {
    elasticsearch {
      hosts => ["elasticsearch:9200"]
      index => "error-logs-%{+yyyy.MM.dd}"
    }
  }
}
```

### 3.5 Elasticsearch 인덱스 수명 주기 관리 (ILM)

```json
PUT _ilm/policy/app-logs-policy
{
  "policy": {
    "phases": {
      "hot": {
        "min_age": "0ms",
        "actions": {
          "rollover": {
            "max_primary_shard_size": "50gb",
            "max_age": "1d"
          },
          "set_priority": { "priority": 100 }
        }
      },
      "warm": {
        "min_age": "7d",
        "actions": {
          "shrink": { "number_of_shards": 1 },
          "forcemerge": { "max_num_segments": 1 },
          "set_priority": { "priority": 50 }
        }
      },
      "cold": {
        "min_age": "30d",
        "actions": {
          "searchable_snapshot": {
            "snapshot_repository": "my-repo"
          },
          "set_priority": { "priority": 0 }
        }
      },
      "delete": {
        "min_age": "90d",
        "actions": {
          "delete": {}
        }
      }
    }
  }
}
```

**ILM 단계 수명 주기:**

```
Hot (0-7일)         Warm (7-30일)       Cold (30-90일)      Delete (90일+)
├─ SSD 스토리지     ├─ HDD 스토리지     ├─ 스냅샷 스토리지   ├─ 삭제됨
├─ 전체 레플리카    ├─ 샤드 축소        ├─ 동결 인덱스       └─ 공간 확보
├─ 쓰기 + 읽기      ├─ 강제 병합        └─ 읽기 전용
└─ 높은 우선순위    └─ 중간 우선순위
```

---

## 4. Loki/Grafana 스택

### 4.1 아키텍처 개요

Loki는 비용 효율적이고 운영하기 쉽도록 설계된 로그 집계 시스템입니다. Elasticsearch와 달리 Loki는 로그 내용을 인덱싱하지 **않습니다** -- 레이블(메타데이터)만 인덱싱합니다. 로그 라인은 오브젝트 스토리지에 압축 저장됩니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                      Loki Stack Architecture                     │
│                                                                  │
│  ┌──────────┐                                                   │
│  │ App Logs │──→ Promtail ──→ ┌──────────┐   ┌──────────┐     │
│  │ (stdout) │    (agent)      │   Loki   │──→│  Grafana │     │
│  └──────────┘                 │ (store)  │   │(visualize│     │
│                               │          │   │& explore)│     │
│  ┌──────────┐                 │ Labels   │   └──────────┘     │
│  │Container │──→ Promtail ──→ │ indexed  │                     │
│  │  Logs    │    (agent)      │ Content  │   Object Storage    │
│  └──────────┘                 │ NOT      │──→ (S3, GCS,       │
│                               │ indexed  │    MinIO)           │
│                               └──────────┘                     │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Loki vs Elasticsearch

| 측면 | Loki | Elasticsearch |
|------|------|---------------|
| **인덱싱** | 레이블만 (Prometheus와 유사) | 전문 역 인덱스 |
| **스토리지 비용** | 낮음 (오브젝트 스토리지에 압축 청크) | 높음 (디스크의 전체 인덱스) |
| **쿼리 속도** | 레이블 필터링 쿼리에 빠름; grep 유사 검색에는 느림 | 모든 전문 검색에 빠름 |
| **운영 복잡도** | 낮음 (무상태 구성 요소, 오브젝트 스토리지) | 높음 (클러스터 관리, JVM 튜닝) |
| **통합** | 네이티브 Grafana | Kibana |
| **적합한 경우** | Prometheus/Grafana를 이미 사용하는 팀, 비용 민감 | 복잡한 검색 요구, 규정 준수, 대규모 분석 |

### 4.3 Promtail 구성

```yaml
# promtail-config.yml
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  # Static file scraping
  - job_name: app-logs
    static_configs:
      - targets:
          - localhost
        labels:
          job: webapp
          environment: production
          __path__: /var/log/app/*.log

  # Kubernetes pod log discovery
  - job_name: kubernetes-pods
    kubernetes_sd_configs:
      - role: pod
    pipeline_stages:
      # Parse JSON logs
      - json:
          expressions:
            level: level
            message: message
            trace_id: trace_id
      # Set log level as a label
      - labels:
          level:
      # Drop debug logs in production
      - match:
          selector: '{level="DEBUG"}'
          action: drop
      # Extract metrics from logs
      - metrics:
          log_lines_total:
            type: Counter
            description: "Total log lines"
            source: level
            config:
              action: inc
```

### 4.4 LogQL (Loki Query Language)

```logql
# 기본 레이블 필터링
{job="webapp", environment="production"}

# 로그 라인 필터 (포함)
{job="webapp"} |= "error"

# 로그 라인 필터 (포함하지 않음)
{job="webapp"} != "health_check"

# 정규식 필터
{job="webapp"} |~ "status=(4|5)\\d{2}"

# JSON 파싱 및 필드 추출
{job="webapp"} | json | level="ERROR"

# JSON 파싱 및 특정 필드 필터
{job="webapp"} | json | duration_ms > 1000

# 로그 라인 포맷팅
{job="webapp"} | json | line_format "{{.timestamp}} [{{.level}}] {{.message}}"

# 집계: 분당 오류 로그 수
count_over_time({job="webapp"} |= "error" [1m])

# 집계: 서비스별 오류 속도
sum by (service) (rate({environment="production"} |= "error" [5m]))

# 집계: 로그 필드에서 p95 지연 시간
quantile_over_time(0.95, {job="webapp"} | json | unwrap duration_ms [5m]) by (endpoint)
```

---

## 5. 로그 집계 패턴

### 5.1 사이드카 패턴 (Kubernetes)

```yaml
# kubernetes deployment with logging sidecar
apiVersion: apps/v1
kind: Deployment
metadata:
  name: webapp
spec:
  template:
    spec:
      containers:
        # Application container
        - name: webapp
          image: webapp:latest
          volumeMounts:
            - name: log-volume
              mountPath: /var/log/app

        # Logging sidecar
        - name: promtail
          image: grafana/promtail:latest
          args:
            - -config.file=/etc/promtail/config.yml
          volumeMounts:
            - name: log-volume
              mountPath: /var/log/app
              readOnly: true
            - name: promtail-config
              mountPath: /etc/promtail

      volumes:
        - name: log-volume
          emptyDir: {}
        - name: promtail-config
          configMap:
            name: promtail-config
```

### 5.2 DaemonSet 패턴 (Kubernetes)

```yaml
# Node-level log collector (preferred for Kubernetes)
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: promtail
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app: promtail
  template:
    spec:
      containers:
        - name: promtail
          image: grafana/promtail:latest
          volumeMounts:
            - name: varlog
              mountPath: /var/log
              readOnly: true
            - name: containers
              mountPath: /var/lib/docker/containers
              readOnly: true
      volumes:
        - name: varlog
          hostPath:
            path: /var/log
        - name: containers
          hostPath:
            path: /var/lib/docker/containers
```

### 5.3 패턴 비교

| 패턴 | 작동 방식 | 장점 | 단점 |
|------|----------|------|------|
| **사이드카** | 파드당 에이전트 | 앱별 세밀한 구성, 격리됨 | 더 높은 리소스 오버헤드, 더 많은 컨테이너 |
| **DaemonSet** | 노드당 에이전트 | 더 낮은 오버헤드, 중앙 집중식 구성 | 노드의 모든 파드에 공유 구성 |
| **직접 푸시** | 앱이 SDK를 통해 로그 전송 | 에이전트 불필요, 즉각적 | 결합도 높음, 앱 내 재시도 로직 |

---

## 6. 로그 보존 정책

### 6.1 보존 정책 설계

| 로그 유형 | Hot (검색 가능) | Warm (아카이브) | 총 보존 기간 | 근거 |
|----------|----------------|----------------|-------------|------|
| **애플리케이션 로그** | 7-14일 | 30-90일 | 90일 | 대부분의 인시던트는 수일 내에 조사됨 |
| **접근 로그** | 7일 | 90-365일 | 365일 | 규정 준수 (PCI-DSS, SOC 2) |
| **감사 로그** | 30일 | 1-7년 | 7년 | 규제 (HIPAA, SOX) |
| **보안 로그** | 30일 | 1-3년 | 3년 | 포렌식, 규정 준수 |
| **디버그 로그** | 1-3일 | 없음 | 3일 | 높은 볼륨, 낮은 장기 가치 |

### 6.2 Loki 보존 구성

```yaml
# loki-config.yml
limits_config:
  retention_period: 744h    # 31 days default

compactor:
  working_directory: /tmp/loki/compactor
  shared_store: s3
  compaction_interval: 10m
  retention_enabled: true
  retention_delete_delay: 2h
  retention_delete_worker_count: 150

# Per-tenant retention (multi-tenant)
overrides:
  tenant-a:
    retention_period: 2160h   # 90 days
  tenant-b:
    retention_period: 720h    # 30 days
```

### 6.3 비용 최적화 전략

1. **불필요한 로그를 조기에 삭제** -- 프로덕션에서 헬스 체크, 디버그 로그를 필터링합니다
2. **대용량 로그에 대해 샘플링 사용** -- 성공 요청의 10%, 오류의 100%를 로깅합니다
3. **적극적으로 압축** -- Loki는 snappy/gzip을 사용; Elasticsearch는 warm 인덱스를 강제 병합합니다
4. **스토리지 계층화** -- Hot (SSD)은 최신, Cold (S3/GCS)는 아카이브용
5. **인덱스 세분화 설정** -- 높은 볼륨에는 일별 인덱스, 낮은 볼륨에는 주별 인덱스

---

## 7. 실용적인 Docker Compose 설정

### 7.1 ELK 스택

```yaml
# docker-compose-elk.yml
version: "3.8"
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.12.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - "ES_JAVA_OPTS=-Xms1g -Xmx1g"
    ports:
      - "9200:9200"
    volumes:
      - es-data:/usr/share/elasticsearch/data

  logstash:
    image: docker.elastic.co/logstash/logstash:8.12.0
    ports:
      - "5044:5044"
    volumes:
      - ./logstash/pipeline:/usr/share/logstash/pipeline
    depends_on:
      - elasticsearch

  kibana:
    image: docker.elastic.co/kibana/kibana:8.12.0
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    depends_on:
      - elasticsearch

  filebeat:
    image: docker.elastic.co/beats/filebeat:8.12.0
    volumes:
      - ./filebeat.yml:/usr/share/filebeat/filebeat.yml:ro
      - /var/log:/var/log:ro
    depends_on:
      - logstash

volumes:
  es-data:
```

### 7.2 Loki 스택

```yaml
# docker-compose-loki.yml
version: "3.8"
services:
  loki:
    image: grafana/loki:2.9.4
    ports:
      - "3100:3100"
    volumes:
      - ./loki-config.yml:/etc/loki/config.yml
      - loki-data:/loki
    command: -config.file=/etc/loki/config.yml

  promtail:
    image: grafana/promtail:2.9.4
    volumes:
      - ./promtail-config.yml:/etc/promtail/config.yml
      - /var/log:/var/log:ro
    command: -config.file=/etc/promtail/config.yml
    depends_on:
      - loki

  grafana:
    image: grafana/grafana:10.3.0
    ports:
      - "3000:3000"
    environment:
      - GF_AUTH_ANONYMOUS_ENABLED=true
      - GF_AUTH_ANONYMOUS_ORG_ROLE=Admin
    volumes:
      - grafana-data:/var/lib/grafana
    depends_on:
      - loki

volumes:
  loki-data:
  grafana-data:
```

---

## 8. 로그 기반 알림

### 8.1 Kibana 알림 규칙

Kibana(Stack Management > Rules)에서 로그 패턴에 따라 트리거되는 규칙을 생성합니다:

```
규칙: 로그의 높은 오류율
조건: level="ERROR"인 문서 수가
      지난 5분 동안 50 초과
동작: Slack 채널 #alerts-backend으로 전송
```

### 8.2 Loki 알림 규칙 (Ruler 사용)

```yaml
# loki-rules.yml
groups:
  - name: log_alerts
    rules:
      # 높은 오류 로그 속도에 대한 알림
      - alert: HighErrorLogRate
        expr: |
          sum(rate({environment="production"} |= "ERROR" [5m])) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "프로덕션에서 높은 오류 로그 속도"
          description: "5분 동안 초당 10개 이상의 오류 로그가 발생하고 있습니다."

      # 특정 심각한 오류 패턴에 대한 알림
      - alert: DatabaseConnectionFailure
        expr: |
          count_over_time({job="webapp"} |= "database connection refused" [5m]) > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "데이터베이스 연결 실패가 감지되었습니다"

      # 패닉/크래시 패턴에 대한 알림
      - alert: ApplicationPanic
        expr: |
          count_over_time({environment="production"} |~ "panic|FATAL|segfault" [1m]) > 0
        for: 0m
        labels:
          severity: critical
        annotations:
          summary: "애플리케이션 패닉/크래시가 감지되었습니다"
```

---

## 9. 다음 단계

- [12_Distributed_Tracing.md](./12_Distributed_Tracing.md) - OpenTelemetry와 Jaeger를 사용한 요청 트레이싱
- [10_Monitoring_and_Alerting.md](./10_Monitoring_and_Alerting.md) - Prometheus를 사용한 메트릭 모니터링

---

## 연습 문제

### 연습 문제 1: 구조화된 로깅 설계

팀이 결제 처리 서비스를 구축하고 있습니다. 다음 이벤트에 대한 구조화된 로그 스키마를 설계하십시오:

1. 결제 요청 수신
2. 결제 게이트웨이에 의한 결제 승인
3. 결제 실패 (은행에서 거부)

각 이벤트에 대해 JSON 필드, 유형, 그리고 Loki에서 레이블로 인덱싱해야 할 필드를 지정하십시오.

<details>
<summary>정답 보기</summary>

**1. 결제 요청 수신:**
```json
{
  "timestamp": "2024-03-15T14:23:45.123Z",
  "level": "INFO",
  "message": "Payment request received",
  "service": "payment-service",
  "request_id": "req-abc123",
  "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
  "order_id": "ord-789",
  "amount": 99.99,
  "currency": "USD",
  "payment_method": "credit_card",
  "card_last_four": "4242",
  "user_id": "user-456"
}
```

**2. 결제 승인:**
```json
{
  "timestamp": "2024-03-15T14:23:45.890Z",
  "level": "INFO",
  "message": "Payment authorized",
  "service": "payment-service",
  "request_id": "req-abc123",
  "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
  "order_id": "ord-789",
  "amount": 99.99,
  "currency": "USD",
  "gateway": "stripe",
  "gateway_transaction_id": "ch_3abc123",
  "authorization_code": "AUTH123",
  "duration_ms": 767
}
```

**3. 결제 실패:**
```json
{
  "timestamp": "2024-03-15T14:23:46.100Z",
  "level": "ERROR",
  "message": "Payment declined by issuing bank",
  "service": "payment-service",
  "request_id": "req-abc123",
  "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
  "order_id": "ord-789",
  "amount": 99.99,
  "currency": "USD",
  "gateway": "stripe",
  "decline_code": "insufficient_funds",
  "decline_message": "Your card has insufficient funds",
  "duration_ms": 523,
  "user_id": "user-456"
}
```

**Loki 레이블** (인덱싱됨, 낮은 카디널리티): `service`, `level`, `environment`, `gateway`

**레이블로 사용하지 않을 것** (높은 카디널리티 -- 인덱스 폭발): `request_id`, `order_id`, `user_id`, `trace_id`

핵심 원칙은 Loki 레이블의 카디널리티가 낮아야 한다는 것입니다 (고유 값이 적어야 함). 높은 카디널리티 필드는 로그 라인에 저장되며 `| json | order_id="ord-789"`로 검색합니다.

</details>

### 연습 문제 2: ELK vs Loki 결정

회사가 하루에 500 GB의 로그를 생성하는 50개의 마이크로서비스를 운영하고 있습니다. 팀은 현재 Elasticsearch를 사용하고 있지만 Loki를 평가하고 있습니다. 다음 각 사용 사례에 대한 트레이드오프를 분석하고 솔루션을 추천하십시오:

1. 인시던트 중 애플리케이션 디버깅
2. 보안 감사 로그 규정 준수 (7년 보존)
3. 실시간 오류 패턴 감지 및 알림

<details>
<summary>정답 보기</summary>

1. **인시던트 중 애플리케이션 디버깅 -- Loki로 충분합니다**
   - 인시던트 중에 엔지니어는 일반적으로 서비스, 시간 범위, 오류 유형을 알고 있습니다. 레이블(`{service="payment-service", level="ERROR"}`)로 필터링한 후 로그 라인을 grep합니다.
   - Loki는 레이블 필터링이 검색 범위를 좁히고, 로그 내용 검색이 작은 서브셋에서 작동하므로 이를 잘 처리합니다.
   - Elasticsearch는 모든 서비스에 걸친 임의의 전문 검색에 더 빠르지만, 인시던트 중에는 거의 필요하지 않습니다.

2. **보안 감사 로그 규정 준수 (7년 보존) -- Loki가 더 좋습니다**
   - 하루 500 GB에 7년 보존이면 Elasticsearch에서 페타바이트급의 인덱싱된 스토리지가 필요합니다 -- 극히 비쌉니다.
   - Loki는 로그 라인을 오브젝트 스토리지(S3/GCS)에 압축 청크로 저장하며, GB당 비용이 Elasticsearch보다 5-10배 저렴합니다.
   - 감사 로그는 거의 쿼리되지 않으므로 (조사 중에만), Loki의 느린 grep 유사 검색이 허용됩니다.
   - 비용 추정: S3 ~$0.023/GB/월 vs Elasticsearch EBS ~$0.10/GB/월.

3. **실시간 오류 패턴 감지 및 알림 -- 둘 다 가능합니다**
   - Loki Ruler는 `count_over_time({level="ERROR"} |= "pattern" [5m]) > threshold`로 알림할 수 있습니다.
   - Elasticsearch에는 실시간 패턴을 감지할 수 있는 Watcher/Kibana Alerts가 있습니다.
   - 단순한 패턴 매칭에는 Loki가 충분하고 더 저렴합니다. 복잡한 상관관계(예: "서비스 A의 오류 후 30초 이내에 서비스 B의 오류가 발생하면 알림")에는 Elasticsearch의 쿼리 기능이 더 강력합니다.

**하루 500 GB에 대한 추천**: **하이브리드 접근 방식**을 사용하십시오. 모든 로그를 비용 효율적인 저장과 기본 쿼리를 위해 Loki에 전송합니다. 보안 및 감사 로그는 Loki(장기)와 작은 Elasticsearch 클러스터(30일 윈도우) 모두에 전송하여 전문 인덱싱이 필요한 규정 준수 검색 요구사항을 충족합니다.

</details>

### 연습 문제 3: LogQL 쿼리 작성

다음 시나리오에 대한 LogQL 쿼리를 작성하십시오:

1. 지난 1시간 동안 `payment-service`의 모든 ERROR 레벨 로그를 찾습니다.
2. 프로덕션에서 각 서비스의 분당 로그 수를 세봅니다.
3. 2초 이상 걸린 요청을 찾습니다 (`duration_ms` 필드가 있는 구조화된 JSON 로그 가정).
4. 엔드포인트별로 그룹화된 로그 데이터에서 95 백분위수 요청 기간을 계산합니다.

<details>
<summary>정답 보기</summary>

1. payment-service의 오류 로그:
```logql
{job="payment-service", level="ERROR"}
```
(시간 범위는 Grafana 시간 선택기에서 "Last 1 hour"로 설정)

2. 서비스별 분당 로그 수:
```logql
sum by (service) (count_over_time({environment="production"}[1m]))
```

3. 느린 요청 (> 2초):
```logql
{job="webapp"} | json | duration_ms > 2000
```

4. 엔드포인트별 P95 요청 기간:
```logql
quantile_over_time(0.95,
  {job="webapp"} | json | unwrap duration_ms [5m]
) by (endpoint)
```

참고: `unwrap`은 집계를 위해 로그 라인에서 숫자 필드를 추출합니다. `unwrap` 없이는 Loki가 모든 로그 내용을 문자열로 처리하며 숫자 연산을 수행할 수 없습니다.

</details>

### 연습 문제 4: 보존 정책 설계

HIPAA 규정을 적용받는 의료 SaaS 애플리케이션에 대한 로그 보존 정책을 설계하십시오. 각 로그 카테고리에 대해 보존 기간, 스토리지 계층, 접근 제어를 지정하십시오.

<details>
<summary>정답 보기</summary>

| 로그 카테고리 | Hot (빠른 검색) | Warm (아카이브) | Cold (규정 준수) | 총 보존 기간 | 접근 제어 |
|-------------|----------------|----------------|-----------------|-------------|----------|
| **애플리케이션 로그** | 14일 (SSD) | 76일 (HDD) | 없음 | 90일 | 개발팀, 당직자 |
| **접근 로그 (PHI)** | 30일 (SSD) | 335일 (HDD) | 5년 (S3 Glacier) | 6년 | 보안팀만 |
| **감사 추적** | 30일 (SSD) | 335일 (HDD) | 6년 (S3 Glacier) | 7년 | 규정 준수팀, 읽기 전용 |
| **인증 로그** | 30일 (SSD) | 335일 (HDD) | 2년 (S3) | 3년 | 보안팀 |
| **디버그 로그** | 3일 (SSD) | 없음 | 없음 | 3일 | 개발팀 |
| **인프라 로그** | 7일 (SSD) | 23일 (HDD) | 없음 | 30일 | 운영팀 |

**HIPAA 특정 요구사항:**
- PHI(보호 대상 건강 정보)를 포함하는 모든 로그는 저장 시(AES-256) 및 전송 시(TLS 1.2+) 암호화되어야 합니다.
- PHI 로그에 대한 접근은 감사 추적에 기록되어야 합니다 (로그 접근 로깅).
- 감사 추적은 변경 불가능해야 합니다 (쓰기 전용 스토리지, 예: S3 Object Lock).
- 로그 삭제는 문서화된 보존 일정을 따라야 하며 감사 가능해야 합니다.

**Elasticsearch ILM에서의 구현:**
- Hot → Warm: `min_age: 30d`, 1 샤드로 축소, 강제 병합
- Warm → Cold: `min_age: 365d`, S3로 검색 가능한 스냅샷
- Cold → Delete: 감사의 경우 `min_age: 2555d` (7년), 앱 로그의 경우 `min_age: 90d`

</details>

---

## 참고 자료

- [Elasticsearch Documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/)
- [Logstash Documentation](https://www.elastic.co/guide/en/logstash/current/)
- [Kibana Documentation](https://www.elastic.co/guide/en/kibana/current/)
- [Grafana Loki Documentation](https://grafana.com/docs/loki/latest/)
- [LogQL Documentation](https://grafana.com/docs/loki/latest/logql/)
- [structlog (Python)](https://www.structlog.org/)
- [Zap Logger (Go)](https://pkg.go.dev/go.uber.org/zap)
