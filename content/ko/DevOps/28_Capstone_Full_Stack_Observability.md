# 28. 캡스톤: 풀스택 관측 가능성(Full-Stack Observability)

**이전**: [AIOps와 이상 탐지](./27_AIOps_Anomaly_Detection.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 프로덕션 마이크로서비스 시스템을 위한 종합(end-to-end) 관측 가능성 플랫폼을 설계할 수 있습니다
2. 메트릭, 로그, 트레이스, 프로파일링 스택 전반에 걸쳐 도구를 선택하고 통합할 수 있습니다
3. 관측 가능성 인프라의 비용 모델을 구축하고 지출을 최적화할 수 있습니다
4. 조직을 위한 관측 가능성 성숙도 로드맵을 작성할 수 있습니다
5. 관측 가능성 도구에 대한 구축(build) vs 구매(buy) 결정을 평가할 수 있습니다
6. 레슨 19-27의 모든 관측 가능성 개념에 대한 숙달을 증명할 수 있습니다

---

이 캡스톤 레슨은 관측 가능성 트랙(레슨 19-27)의 모든 내용을 응집력 있고 프로덕션 준비된 관측 가능성 플랫폼 설계로 종합합니다. 아키텍처를 설계하고, 도구를 선택하고, 프로세스를 정의하고, 현실적인 마이크로서비스 플랫폼의 비용을 계산합니다.

> **비유 -- 병원 건설**: 개별 의료 기술(심장학, 방사선학, 외과)은 필요하지만 환자를 잘 치료하기에 불충분합니다. 병원에는 통합 시스템이 필요합니다: 접수(데이터 수집), 트리아지(알림), 진단(상관 관계 및 조사), 치료(복구), 품질 개선(포스트모템). 이 캡스톤은 의술을 연습하는 것이 아니라 병원을 건설하는 것에 관한 것입니다.

## 1. 참조 아키텍처(Reference Architecture)

### 1.1 대상 시스템

```
전자상거래 플랫폼:
- 30개 마이크로서비스 (Go, Python, Java, Node.js)
- 3개 Kubernetes 클러스터 (us-east, eu-west, ap-southeast)
- PostgreSQL, Redis, Elasticsearch, Kafka
- 피크 10,000 요청/초
- 99.95% 가용성 SLO
- 팀: 8개 팀에 걸쳐 50명의 엔지니어
```

### 1.2 풀스택 관측 가능성 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                        시각화 & 분석                              │
│  ┌───────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │  Grafana   │  │  Grafana     │  │  PagerDuty/  │             │
│  │ 대시보드   │  │  알림        │  │  Opsgenie    │             │
│  │  + SLO    │  │  (통합)      │  │  (페이징)    │             │
│  └─────┬─────┘  └──────┬───────┘  └──────────────┘             │
│        │               │                                         │
│  ┌─────▼───────────────▼─────────────────────────────────────┐  │
│  │              데이터 소스 (Grafana가 모두 연결)                │  │
│  ├───────────┬──────────────┬─────────────┬──────────────────┤  │
│  │Prometheus │   Tempo      │    Loki     │   Pyroscope      │  │
│  │ + Mimir   │  (트레이스)   │   (로그)    │  (프로파일)       │  │
│  │(메트릭)   │              │             │                   │  │
│  └─────┬─────┴──────┬───────┴──────┬──────┴─────────┬────────┘  │
│        │            │              │                │            │
│  ┌─────▼────────────▼──────────────▼────────────────▼────────┐  │
│  │                OTel Collector 게이트웨이                     │  │
│  │  - 테일 샘플링      - 스팬 메트릭 커넥터                     │  │
│  │  - PII 스크러빙     - 서비스 그래프 커넥터                   │  │
│  │  - 라우팅           - 속성 강화                              │  │
│  └──────────────────────────┬────────────────────────────────┘  │
│                             │                                    │
│  ┌──────────────────────────▼────────────────────────────────┐  │
│  │              OTel Collector 에이전트 (DaemonSet)            │  │
│  │  - 배치 + 메모리 제한     - 헬스체크 필터링                  │  │
│  │  - 노드 메타데이터 강화                                     │  │
│  └──────────────────────────┬────────────────────────────────┘  │
└─────────────────────────────┼────────────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────────────┐
│                    애플리케이션 계층                                │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐       │
│  │ Go svc │ │ Py svc │ │Java svc│ │Node svc│ │ ...×30 │       │
│  │+ OTel  │ │+ OTel  │ │+ OTel  │ │+ OTel  │ │        │       │
│  │ SDK    │ │ SDK    │ │ Agent  │ │ SDK    │ │        │       │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘       │
│  ┌─────────────────────────────────────────────────────┐       │
│  │ Cilium + Hubble (eBPF 네트워크 관측 가능성)          │       │
│  └─────────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. 도구 선택(Tool Selection)

### 2.1 선택 기준

| 기준 | 가중치 | 요소 |
|------|-------|------|
| **비용** | 25% | 라이선싱, 인프라, 운영 오버헤드 |
| **상호 운용성** | 20% | 개방 표준 (OTel, PromQL), 데이터 소스 연결 |
| **확장성** | 20% | 3개 클러스터, 30개 서비스, 10K rps 처리 |
| **운영 복잡도** | 15% | 팀 전문성, 유지보수 부담 |
| **기능 완전성** | 10% | 모든 텔레메트리 유형 커버, 신호 간 상관 |
| **벤더 독립성** | 10% | 락인 방지, 개방 프로토콜 사용 |

### 2.2 도구 선택 매트릭스

| 신호 | 도구 | 선택 이유 |
|------|------|---------|
| **메트릭** | Prometheus + Mimir | 업계 표준, PromQL 생태계, 장기 저장 및 멀티 테넌트를 위한 Mimir |
| **트레이스** | Grafana Tempo | 오브젝트 스토리지 백엔드(저렴), 네이티브 Grafana 통합 |
| **로그** | Grafana Loki | 레이블 기반 인덱싱(저비용), LogQL, Grafana 통합 |
| **프로파일** | Grafana Pyroscope | 플레임 그래프, 트레이스-프로파일 연결 |
| **수집** | OTel Collector | 벤더 중립, 모든 신호 지원, 확장 가능 |
| **네트워크** | Cilium Hubble | eBPF 기반, 제로 계측 네트워크 관측 가능성 |
| **시각화** | Grafana | 모든 신호를 위한 통합 UI, 크로스 신호 연결 |
| **알림** | Grafana 통합 알림 | 모든 데이터 소스에 걸친 단일 알림 엔진 |
| **페이징** | PagerDuty | 온콜 스케줄링, 에스컬레이션, 인시던트 관리 |

### 2.3 구축 vs 구매 결정(Build vs Buy)

| 요소 | 자체 호스팅 (OSS) | SaaS (Grafana Cloud, Datadog) |
|------|-----------------|-------------------------------|
| **대규모 비용** | 낮음 (인프라 + 엔지니어링 시간) | 높음 (메트릭/호스트/트레이스당 과금) |
| **운영 부담** | 높음 (업그레이드, 스케일링, HA) | 낮음 (벤더가 관리) |
| **커스터마이징** | 완전 제어 | 벤더 기능으로 제한 |
| **데이터 레지던시** | 완전 제어 | 벤더 리전에 의존 |
| **가치 실현 시간** | 주-월 | 일-주 |
| **팀 전문성** | 관측 가능성 플랫폼 팀 필요 | 최소 (벤더가 지원) |

**권장**: 소규모 팀(< 20명 엔지니어)은 SaaS로 시작합니다. 전용 플랫폼 팀이 있고 비용 절약이 운영 오버헤드를 정당화할 때 자체 호스팅으로 전환합니다.

---

## 3. SLO 프레임워크

### 3.1 서비스 SLO 정의

```yaml
# 전자상거래 플랫폼의 SLO 정의
services:
  - name: api-gateway
    slos:
      - name: availability
        target: 99.99%
        sli: "비5xx 응답 비율"
        window: 30일 롤링
      - name: latency
        target: 99%
        threshold: 200ms
        sli: "200ms 미만 요청 비율"

  - name: payment-service
    slos:
      - name: availability
        target: 99.95%
        sli: "성공적인 결제 시도 비율"
        window: 30일 롤링
      - name: latency
        target: 99%
        threshold: 500ms

  - name: search-service
    slos:
      - name: availability
        target: 99.9%
        sli: "비오류 검색 응답 비율"
      - name: latency
        target: 95%
        threshold: 300ms
      - name: freshness
        target: 99%
        threshold: 60s
        sli: "소스 변경 후 60초 이내 인덱스 업데이트"

  - name: order-service
    slos:
      - name: availability
        target: 99.95%
      - name: latency
        target: 99%
        threshold: 1000ms

# 사용자 여정 SLO
journeys:
  - name: checkout
    target: 99.9%
    sli: "5초 이내 성공적으로 완료된 결제 시도 비율"
    services: [api-gateway, cart-service, payment-service, order-service, inventory-service]

  - name: search-and-browse
    target: 99.5%
    sli: "500ms 이내 결과를 반환하는 검색 비율"
    services: [api-gateway, search-service, recommendation-service]
```

### 3.2 오류 예산 정책(Error Budget Policy)

```
오류 예산 정책 (조직 전체)
────────────────────────────────────────
예산 > 50%:
  → 정상 속도로 기능 출시
  → 표준 배포 관행

예산 25-50%:
  → 필수 카나리 배포
  → 금요일 위험한 배포 금지
  → 주간 SLO 검토

예산 5-25%:
  → 비핵심 변경에 대한 기능 동결
  → 신뢰성 스프린트: 다음 스프린트를 신뢰성에 전념
  → 일일 SLO 검토

예산 < 5%:
  → 전면 배포 동결 (핵심 보안 패치 제외)
  → 예산 소비 이벤트에 대한 인시던트 검토 및 포스트모템
  → 엔지니어링 리더십 통보

예산 소진:
  → SLO 위반 선언
  → 이 SLO에 기여하는 모든 팀 신뢰성 모드 진입
  → 48시간 이내 포스트모템 필수
  → 조치 항목 완료까지 추적
```

---

## 4. 계측 전략(Instrumentation Strategy)

### 4.1 언어별 계측 계획

| 언어 | 자동 계측 | 수동 계측 | SDK |
|------|---------|---------|-----|
| **Go** | OTel contrib 라이브러리 | 비즈니스 로직용 커스텀 스팬 | `go.opentelemetry.io/otel` |
| **Python** | `opentelemetry-instrument` CLI | 비즈니스 로직용 커스텀 스팬 | `opentelemetry-sdk` |
| **Java** | `-javaagent:otel-javaagent.jar` | `@WithSpan` 어노테이션 | OTel Java Agent |
| **Node.js** | `@opentelemetry/auto-instrumentations-node` | 커스텀 스팬 | `@opentelemetry/sdk-node` |

### 4.2 필수 텔레메트리 표준

```yaml
# 조직 전체 텔레메트리 표준
resource_attributes:
  required:
    - service.name              # Kubernetes 서비스 이름과 일치
    - service.version           # 배포의 시맨틱 버전
    - deployment.environment    # production, staging, development
    - k8s.namespace.name        # Kubernetes 네임스페이스
    - k8s.pod.name              # 파드 이름
    - k8s.node.name             # 노드 이름

span_conventions:
  required:
    - http.method               # GET, POST 등
    - http.route                # 인스턴스가 아닌 템플릿 (/users/:id)
    - http.status_code          # 응답 상태 코드
  recommended:
    - db.system                 # postgresql, redis 등
    - db.name                   # 데이터베이스 이름
    - messaging.system          # kafka, rabbitmq 등

log_standards:
  format: JSON
  required_fields:
    - timestamp (ISO 8601)
    - level (INFO, WARN, ERROR)
    - message
    - service_name
    - trace_id
    - span_id
  forbidden:
    - 비밀번호, API 키, 토큰
    - 전체 이메일 주소 (도메인만 사용)
    - 신용카드 번호
    - 주민등록번호

metric_conventions:
  naming: 단위 접미사가 있는 snake_case (_total, _seconds, _bytes)
  labels:
    max_cardinality: 메트릭당 1000
    forbidden: user_id, request_id, email, ip_address
```

---

## 5. 파이프라인 구성(Pipeline Configuration)

### 5.1 OTel Collector 게이트웨이 (프로덕션)

```yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317

processors:
  memory_limiter:
    check_interval: 1s
    limit_mib: 2048
    spike_limit_mib: 512

  # PII 스크러빙
  transform/pii:
    error_mode: ignore
    trace_statements:
      - context: span
        statements:
          - replace_pattern(attributes["db.statement"], "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}", "***@***.***")
          - replace_pattern(attributes["db.statement"], "\\b\\d{4}[- ]?\\d{4}[- ]?\\d{4}[- ]?\\d{4}\\b", "****")
    log_statements:
      - context: log
        statements:
          - replace_pattern(body, "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}", "***@***.***")

  # 테일 샘플링
  tail_sampling:
    decision_wait: 30s
    num_traces: 300000
    policies:
      - name: errors
        type: status_code
        status_code: {status_codes: [ERROR]}
      - name: slow
        type: latency
        latency: {threshold_ms: 2000}
      - name: critical-services
        type: string_attribute
        string_attribute:
          key: service.name
          values: [payment-service, auth-service, order-service]
      - name: sample-rest
        type: probabilistic
        probabilistic: {sampling_percentage: 10}

  # 속성 강화
  attributes/enrich:
    actions:
      - key: platform.team
        from_attribute: service.name
        action: insert
      - key: cost_center
        value: "engineering"
        action: insert

  batch:
    send_batch_size: 2048
    timeout: 10s

connectors:
  spanmetrics:
    histogram:
      explicit:
        buckets: [5ms, 10ms, 50ms, 100ms, 250ms, 500ms, 1s, 2.5s, 5s, 10s]
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
    endpoint: tempo-distributor:4317
  prometheus:
    endpoint: 0.0.0.0:8889
  loki:
    endpoint: http://loki-gateway:3100/loki/api/v1/push
  otlp/pyroscope:
    endpoint: pyroscope:4317

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [memory_limiter, transform/pii, attributes/enrich, tail_sampling, batch]
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
      processors: [memory_limiter, transform/pii, batch]
      exporters: [loki]
```

---

## 6. 비용 관리(Cost Management)

### 6.1 비용 모델

```
구성 요소             월간 비용 추정
─────────────────────────────────────────
메트릭 (Mimir):
  2M 활성 시리즈 × $0.008/1K 시리즈 = $16,000
  스토리지: 500 GB × $0.10/GB             =    $50
                                          --------
                                          $16,050

트레이스 (Tempo):
  10K 스팬/초 × 2,592,000 초/월 = 25.9B 스팬
  샘플링 후 (10%): 2.59B 스팬
  스토리지: ~200 GB × $0.10/GB            =    $20
  (Tempo는 오브젝트 스토리지로 매우 저렴)
                                          --------
                                              $20

로그 (Loki):
  30 서비스 × 5 GB/일 × 30일 = 4.5 TB
  필터링 후: ~1 TB/월
  스토리지: 1 TB × $0.10/GB              =   $100
  수집: 1 TB × $0.50/GB                  =   $500
                                          --------
                                             $600

프로파일 (Pyroscope):
  30 서비스 × 100 Hz × 30일              = 최소 스토리지
                                          --------
                                              $50

인프라 (K8s):
  OTel Collectors: 20 파드 × 0.5 CPU     = 10 CPU
  Mimir: 12 파드 × 2 CPU                 = 24 CPU
  Tempo: 6 파드 × 1 CPU                  = 6 CPU
  Loki: 8 파드 × 1 CPU                   = 8 CPU
  Grafana: 3 파드 × 0.5 CPU              = 1.5 CPU
  합계: ~50 CPU × $0.04/hr × 720 hr      = $1,440
  메모리: 200 GB × $0.005/hr × 720       = $720
                                          --------
                                          $2,160

총 월간 비용:                             ~$18,880
서비스당: $18,880 / 30                    = ~$630/서비스/월
```

### 6.2 비용 최적화 레버

| 레버 | 절약 | 구현 |
|------|------|------|
| 미사용 메트릭 삭제 | 메트릭 비용 20-30% | 분기별 `mimirtool analyze` 실행 |
| 로그 상세도 감소 | 로그 비용 40-60% | Collector에서 DEBUG/TRACE 필터링 |
| 트레이스 샘플링 증가 | 트레이스 비용 50-90% | 10% 대신 5% 테일 샘플링 |
| 보존 기간 단축 | 전체 10-20% | 7일 원본, 30일 집계, 90일 메트릭만 |
| 레코딩 규칙 사용 | 메트릭 비용 10-15% | 높은 카디널리티 쿼리 사전 집계 |
| Collector 리소스 적정화 | 인프라 비용 10-20% | Collector CPU/메모리 사용량 프로파일링 |

---

## 7. 운영 프로세스(Operational Processes)

### 7.1 관측 가능성 팀 책임

```
플랫폼/관측 가능성 팀 (2-3명 엔지니어):
  - Collector 파이프라인 구성 유지보수
  - 메트릭/로그/트레이스 백엔드 운영
  - SLO 대시보드 생성 및 유지보수
  - 텔레메트리 표준 정의
  - 분기별 비용 검토 및 최적화
  - 제품 팀 계측 교육

제품 팀 (각 팀):
  - 서비스 계측 (표준 따르기)
  - 서비스 SLO 정의
  - 서비스별 대시보드 생성
  - 알림에 대한 런북 작성
  - 온콜 순환 참여
  - 인시던트에 대한 포스트모템 수행
```

### 7.2 분기별 관측 가능성 검토

```
분기별 검토 안건 (90분):
  1. (15분) 비용 검토: 실제 vs 예산, 최적화 기회
  2. (15분) SLO 준수: 어떤 서비스가 SLO를 달성/미달했는지
  3. (15분) 인시던트 분석: MTTD/MTTR 추세, 포스트모템 조치 항목 완료
  4. (15분) 도구 평가: 갭, 평가할 새 도구
  5. (15분) 텔레메트리 품질: 카디널리티 추세, 누락된 계측
  6. (15분) 로드맵: 다음 분기 우선순위
```

---

## 8. 성숙도 로드맵(Maturity Roadmap)

### 8.1 12개월 로드맵

```
분기 1: 기반
────────────────────
주 1-2:   OTel Collector (에이전트 + 게이트웨이) 배포
주 3-4:   Mimir, Tempo, Loki, Grafana 배포
주 5-6:   상위 10개 핵심 서비스 자동 계측
주 7-8:   핵심 SLO 대시보드 생성 (가용성, 지연)
주 9-10:  핵심 서비스에 번 레이트 알림 구성
주 11-12: 기본 계측 및 디버깅에 대한 팀 교육
마일스톤: Level 1 (Informed) 달성

분기 2: 상관 관계
──────────────────────
주 1-3:   트레이스-로그 연결 활성화 (모든 로그에 trace_id)
주 4-6:   주요 메트릭에 exemplar 활성화
주 7-8:   spanmetrics, servicegraph 커넥터 배포
주 9-10:  나머지 20개 서비스 계측
주 11-12: 통합 디버깅 대시보드 생성
마일스톤: Level 2 (Investigative) 달성

분기 3: 선제적
────────────────────
주 1-3:   30개 전체 서비스에 SLO 정의
주 4-6:   Pyroscope로 지속적 프로파일링 배포
주 7-8:   동적 알림 구현 (통계적 기준선)
주 9-10:  변경-영향 상관 관계 구축 (배포 → 이상)
주 11-12: 첫 번째 게임 데이 연습 실시
마일스톤: Level 3 (Proactive) 달성

분기 4: 최적화
───────────────────
주 1-3:   비용 최적화 스프린트 (미사용 텔레메트리 제거)
주 4-6:   알려진 패턴에 L2 자동 복구 구현
주 7-8:   Cilium Hubble로 네트워크 관측 가능성 배포
주 9-10:  AIOps 기능 평가 (이상 탐지, RCA)
주 11-12: 관측 가능성 성숙도 보고서 발행
마일스톤: Level 3+ (Proactive, optimized) 달성
```

---

## 9. 관측 가능성 트랙 요약

| 레슨 | 핵심 교훈 |
|------|---------|
| 19. 관측 가능성 엔지니어링 | 관측 가능성은 대시보드 모니터링이 아닌 임의의 질문을 할 수 있는 것 |
| 20. SLO 엔지니어링 | SLO + 오류 예산은 신뢰성 vs 속도의 의사결정 프레임워크 제공 |
| 21. 신호 상관 관계 | 메트릭, 로그, 트레이스 연결로 5배 빠른 디버깅 가능 |
| 22. 고급 메트릭 아키텍처 | Thanos/Mimir로 Prometheus 확장; 카디널리티가 핵심 비용 동인 |
| 23. OpenTelemetry 파이프라인 | Collector 파이프라인 설계가 관측 가능성 품질과 비용 결정 |
| 24. eBPF 관측 가능성 | 코드 변경 없는 커널 수준 관측이 OTel을 보완 |
| 25. 지속적 프로파일링 | 프로파일링은 코드 수준에서 WHY를 알려줌; 플레임 그래프가 주요 도구 |
| 26. 인시던트 대응 | 구조화된 대응 + 비난 없는 포스트모템이 인시던트를 학습으로 전환 |
| 27. AIOps와 이상 탐지 | ML이 알림을 향상시키지만 SLO 기반 알림이 기반 |
| 28. 캡스톤 (이 레슨) | 모든 개념을 프로덕션 준비 플랫폼으로 통합 |

---

## 연습 문제

### 연습 1: 플랫폼 설계

다음 조건의 스타트업을 위한 완전한 관측 가능성 플랫폼을 설계하세요:
- 10개 마이크로서비스 (모두 Python/FastAPI)
- 단일 Kubernetes 클러스터
- 피크 1,000 요청/초
- 15명 엔지니어 팀 (전용 플랫폼 팀 없음)
- 관측 가능성 예산: 월 $3,000

도구 선택, 아키텍처 다이어그램, 3개 핵심 서비스의 SLO 정의, OTel Collector 구성, 비용 분석을 지정하세요.

<details>
<summary>정답 보기</summary>

**도구 선택**: 소규모 팀과 제한된 예산을 감안하여 **Grafana Cloud Free/Pro 티어**를 관리형 관측 가능성으로 사용합니다. 플랫폼 팀의 필요성을 제거합니다.

| 신호 | 도구 | 선택 이유 |
|------|------|---------|
| 메트릭 | Grafana Cloud (관리형 Mimir) | 무료 티어: 10K 시리즈. Pro: $8/1K 시리즈 |
| 트레이스 | Grafana Cloud (관리형 Tempo) | 50 GB 무료. Pro: $0.50/GB |
| 로그 | Grafana Cloud (관리형 Loki) | 50 GB 무료. Pro: $0.50/GB |
| 수집 | OTel Collector | 자체 관리 (단순 DaemonSet) |
| 알림 | Grafana Cloud Alerting | 포함 |
| 페이징 | PagerDuty (5명 미만 무료 티어) | 2-3개 온콜 순환으로 충분 |

**아키텍처:**
```
10 Python/FastAPI 서비스 (opentelemetry-instrument로 자동 계측)
  → OTel Collector DaemonSet (노드당 1개, 3개 노드)
    → Grafana Cloud (OTLP 엔드포인트)
      → Grafana 대시보드 + 알림 + PagerDuty
```

**비용 분석:**
```
Grafana Cloud Pro:
  메트릭: ~50K 시리즈 × $8/1K = $400
  트레이스: ~20 GB/월 (10% 샘플링 후) × $0.50 = $10
  로그: ~100 GB/월 × $0.50 = $50
  프로파일: 10 서비스 × $5 = $50
                                    소계: $510

인프라:
  OTel Collectors: 3 파드 × 0.25 CPU = $25
  (백엔드는 Grafana Cloud가 관리)
                                    소계: $25

PagerDuty (무료 티어):                        $0

합계: ~$535/월 ($3,000 예산 이내)
```

**SLO 정의:**
```yaml
- service: payment-service
  slos:
    - name: availability
      target: 99.9%
      sli: "비5xx 응답 / 총 응답"
    - name: latency
      target: 99%
      threshold: 500ms

- service: auth-service
  slos:
    - name: availability
      target: 99.99%
    - name: latency
      target: 99%
      threshold: 200ms

- service: order-service
  slos:
    - name: availability
      target: 99.9%
    - name: latency
      target: 95%
      threshold: 1000ms
```

</details>

### 연습 2: 인시던트 시뮬레이션

다음 시나리오에 대한 완전한 인시던트 대응을 수행하세요:

월요일 03:00 UTC에 checkout 여정 SLO가 목표 이하로 떨어집니다. `order-service`의 오류율이 15%로 급증합니다. 동시에 `inventory-service` 지연이 50ms에서 5초로 증가합니다. `inventory-events`의 Kafka 컨슈머 랙이 빠르게 증가하고 있습니다.

설명하세요: (a) 감지 및 알림, (b) 트리아지 및 역할 할당, (c) 상관된 관측 스택을 사용한 조사, (d) 완화, (e) 근본 원인 식별, (f) 포스트모템 조치 항목.

<details>
<summary>정답 보기</summary>

**(a) 감지 및 알림:**
- 03:00 -- Checkout 여정 SLO 번 레이트 알림 발동 (14.4배) -- 온콜 페이지
- 03:00 -- OrderServiceHighErrorRate 알림 발동 (15% > 1% 임계값)
- 03:01 -- InventoryServiceHighLatency 알림 발동 (p99 = 5s > 500ms 임계값)
- 03:01 -- KafkaConsumerLagHigh 알림 발동 (inventory-events 컨슈머)
- Alertmanager 그룹핑: 4개 알림 → 1개 인시던트 그룹 (checkout 여정으로 상관)

**(b) 트리아지 및 역할 할당:**
- 03:02 -- 온콜 엔지니어 확인. SEV1 선언 (15% 오류율로 checkout 다운).
- 03:03 -- 인시던트 채널 `#inc-20250317-checkout-failure` 생성
- 03:04 -- 역할 할당: IC = 온콜, 기술 리드 = inventory-service 소유자 (페이지됨), 스크라이브 = 온콜 세컨더리

**(c) 관측 스택을 사용한 조사:**

```
1단계: SLO 대시보드 (메트릭)
  → Checkout 여정 85% (목표 99.9%)
  → Order-service 오류 예산: 거의 소진

2단계: 오류율 분석 (메트릭)
  → order-service POST /orders: 15% 500 오류
  → 모든 오류가 "inventory reservation timeout"

3단계: 오류 급증의 exemplar 클릭 (메트릭 → 트레이스)
  → 트레이스: api-gateway → order-service → inventory-service [TIMEOUT 5s]
  → inventory-service 스팬: "reserve_items"가 5.0초 후 타임아웃

4단계: inventory-service 메트릭 확인
  → Kafka 컨슈머 랙: 50,000 메시지이며 증가 중
  → inventory-service CPU: 10% (CPU 바운드 아님)
  → inventory-service DB 연결: 50/50 (소진!)

5단계: inventory-service 트레이스 확인 (Tempo)
  → 모든 트레이스에서: postgres SELECT가 4-5초 소요
  → 속성: db.statement = "SELECT * FROM inventory WHERE sku IN (...)"

6단계: inventory-service 로그 확인 (Loki)
  → "WARN: Lock wait timeout exceeded for inventory table"
  → "ERROR: cannot acquire lock on row in relation 'inventory'"
  → "INFO: Kafka consumer commit failed: rebalance in progress"

7단계: PostgreSQL 메트릭 확인
  → pg_stat_activity: 48개 활성 쿼리, 40개 "Lock" 대기 상태
  → 2시간 동안 실행 중인 쿼리: ALTER TABLE inventory ADD COLUMN ...

근본 원인 식별: 스키마 마이그레이션(ALTER TABLE ADD COLUMN)이
inventory 테이블에 락을 보유하여 모든 SELECT 쿼리를 차단.
마이그레이션은 03:00에 크론 잡에 의해 트리거됨.
```

**(d) 완화:**
- 03:15 -- ALTER TABLE 쿼리 킬: `SELECT pg_terminate_backend(pid)`
- 03:16 -- Inventory-service 쿼리 재개, 컨슈머 랙 감소 시작
- 03:20 -- 오류율 0%로 감소, checkout SLO 회복 중
- 03:30 -- 모든 메트릭 기준선으로 복귀, 인시던트 해결

**(e) 근본 원인:**
데이터베이스 마이그레이션 크론 잡(03:00 UTC 스케줄)이 대규모 테이블(1천만 행)에서 `ALTER TABLE inventory ADD COLUMN last_restock_date TIMESTAMP`를 실행. 이전 버전의 PostgreSQL에서 기본값이 있는 `ALTER TABLE ADD COLUMN`은 `ACCESS EXCLUSIVE` 락을 획득하여 모든 읽기를 차단. 락이 모든 inventory 쿼리를 차단하여 order-service의 연쇄 타임아웃 발생.

**(f) 포스트모템 조치 항목:**

| # | 조치 | 분류 | 우선순위 |
|---|------|------|---------|
| 1 | `ALTER TABLE ADD COLUMN ... DEFAULT NULL` 사용 (PG 11+에서 락 없음) 또는 동시 마이그레이션 도구 사용 | 예방 | P0 |
| 2 | 스키마 마이그레이션을 크론이 아닌 유지보수 윈도우에 스케줄 | 예방 | P1 |
| 3 | PostgreSQL 락 대기 모니터링 추가: 쿼리 대기 30초 초과 시 알림 | 감지 | P0 |
| 4 | 더 낮은 임계값으로 Kafka 컨슈머 랙 알림 추가 (현재 너무 높음) | 감지 | P1 |
| 5 | 애플리케이션 수준에서 쿼리 타임아웃 구현 (최대 5초, 현재 무한) | 완화 | P1 |
| 6 | order-service와 inventory-service 사이 서킷 브레이커 추가 | 완화 | P2 |
| 7 | 모든 스키마 변경 전 마이그레이션 검토 체크리스트(락 분석) 요구 | 예방 | P1 |

</details>

### 연습 3: 비용 최적화

관측 가능성 플랫폼 비용이 월 $25,000입니다. 리더십이 디버깅 능력에 큰 영향 없이 월 $15,000으로 줄여달라고 요청합니다. 현재 사용량:

- 메트릭: 3M 활성 시리즈 (관리형 플랫폼에서 $24,000)
- 트레이스: 100 GB/월 ($50)
- 로그: 2 TB/월 ($1,000)
- 인프라: $2,000

구체적인 조치와 예상 절약을 포함한 비용 최적화 계획을 제안하세요.

<details>
<summary>정답 보기</summary>

**현재 비용: $27,050/월. 목표: $15,000/월. $12,050 절감 필요.**

비용은 메트릭($24,000 = 전체의 89%)이 압도적으로 지배합니다. 거기에 집중합니다.

**최적화 계획:**

| # | 조치 | 신호 | 예상 절약 | 새 비용 |
|---|------|------|---------|--------|
| 1 | `mimirtool analyze`를 실행하여 미사용 메트릭 찾기. ~30%가 대시보드나 알림에서 쿼리되지 않을 것으로 예상. metric_relabel_configs로 삭제. | 메트릭 | $7,200 ($24,000의 30%) | $16,800 |
| 2 | 히스토그램 버킷 수를 11개에서 7개로 줄이기 (거의 도달하지 않는 버킷 제거). 히스토그램이 가장 큰 카디널리티 배수. | 메트릭 | $2,400 (10%) | $14,400 |
| 3 | 레코딩 규칙으로 파드별 메트릭을 배포별로 사전 집계. 장기 저장에서 파드별 메트릭 삭제. | 메트릭 | $2,400 (10%) | $12,000 |
| 4 | 트레이스 샘플링을 현재 비율에서 5%로 증가. 오류와 느린 트레이스는 100% 유지. | 트레이스 | $25 ($50의 50%) | $11,975 |
| 5 | OTel Collector에서 DEBUG 및 TRACE 수준 로그 필터링. 보통 로그 볼륨의 40-60%. | 로그 | $500 ($1,000의 50%) | $11,475 |
| 6 | Collector 파드 적정화 (실제 CPU/메모리 프로파일링, 요청 감소). | 인프라 | $400 ($2,000의 20%) | $11,075 |

**총 절약: $12,925**
**새 월간 비용: ~$14,125** ($15,000 목표 이내)

**핵심 인사이트**: 메트릭 카디널리티가 #1 비용 동인입니다. 세 가지 메트릭 중심 조치(1, 2, 3)가 $12,925 절약 중 $12,000을 차지합니다. 트레이스와 로그는 이 아키텍처에서 이미 저렴합니다.

**위험 완화**: 메트릭을 삭제하기 전에 대시보드, 알림, 레코딩 규칙에서 사용되지 않는지 확인합니다. 삭제된 메트릭이 수집되지만 저장되지 않는 2주 "섀도우" 기간을 사용하여 필요 시 롤백을 허용합니다.

</details>

---

## 참고 자료

- [Observability Engineering (O'Reilly)](https://www.oreilly.com/library/view/observability-engineering/9781492076438/)
- [Google SRE Book](https://sre.google/sre-book/table-of-contents/)
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [Grafana LGTM Stack](https://grafana.com/oss/lgtm-stack/)
- [CNCF Observability Landscape](https://landscape.cncf.io/card-mode?category=observability-and-analysis)
- [Cloud Native Observability (O'Reilly)](https://www.oreilly.com/library/view/cloud-native-observability/9781098145545/)
