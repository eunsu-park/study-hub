# SRE Practices

**이전**: [Platform Engineering](./17_Platform_Engineering.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Site Reliability Engineering(SRE)을 정의하고 DevOps와의 관계 및 차이점 설명하기
2. 서비스에 대한 SLO, SLI, SLA를 설계하고 이를 활용하여 엔지니어링 의사 결정하기
3. 에러 버짓을 계산하고 관리하여 안정성과 기능 개발 속도 간의 균형 맞추기
4. 자동화와 엔지니어링 솔루션을 통해 토일(toil)을 식별하고 제거하기
5. 명확한 역할, 커뮤니케이션, 에스컬레이션을 갖춘 체계적인 인시던트 관리 구현하기
6. 비난 없이 실행 가능한 개선을 도출하는 효과적인 포스트모텀 수행하기
7. 엔지니어의 안녕을 존중하는 지속 가능한 온콜 프랙티스 설계하기

---

Site Reliability Engineering(SRE)은 소프트웨어 엔지니어링 원칙을 인프라와 운영에 적용하는 분야입니다. 2003년 Google에서 만들어진 SRE는 "소프트웨어 엔지니어에게 운영 팀을 설계하라고 하면 어떻게 될까?"라는 질문에 답합니다. 그 답은 안정성을 기능으로 취급하고, 정밀한 메트릭(SLO)으로 정의하며, 에러 버짓을 사용하여 안정성과 속도 간의 원칙적인 트레이드오프를 만드는 일련의 프랙티스입니다.

> **비유 -- 교량 공학**: 토목 공학자는 어떤 하중에서도 절대 무너지지 않는 다리를 만들지 않습니다 -- 그것은 무한히 비싸고 영원히 걸릴 것입니다. 대신 하중 사양(SLO)을 정의합니다: "이 다리는 안전 계수 2배로 하루 10,000대의 차량을 안전하게 지탱해야 합니다." 실제 하중(SLI)을 측정하고 안전 마진(에러 버짓)을 추적합니다. 다리가 일관되게 용량의 80%라면 버스 노선 추가를 승인할 수 있습니다. 98%라면 보강이 추가될 때까지 새로운 교통을 중단합니다. SRE는 이와 동일한 공학적 규율을 소프트웨어 시스템에 적용합니다.

## 1. SRE vs DevOps

### 1.1 관계

```
DevOps:  A cultural movement and set of practices
         ┌─────────────────────────────────────┐
         │  • Break down silos                 │
         │  • Automate everything              │
         │  • Continuous improvement            │
         │  • Shared responsibility             │
         └─────────────────────────────────────┘
                         │
                         │ "class SRE implements DevOps"
                         ▼
SRE:     A specific implementation with concrete practices
         ┌─────────────────────────────────────┐
         │  • SLOs/SLIs/SLAs (measurable)      │
         │  • Error budgets (decision framework)│
         │  • Toil reduction (< 50% rule)       │
         │  • Incident management (structured)  │
         │  • Postmortems (blameless)           │
         │  • On-call (sustainable)             │
         └─────────────────────────────────────┘
```

### 1.2 주요 차이점

| 측면 | DevOps | SRE |
|--------|--------|-----|
| **기원** | 풀뿌리 문화 운동 (2008) | Google 엔지니어링 프랙티스 (2003) |
| **초점** | 협업, 자동화, CI/CD | 안정성, 측정 가능성, 엔지니어링 |
| **팀 구조** | 모든 팀에 내재 | 전담 SRE 팀 (또는 임베디드 SRE) |
| **의사 결정 프레임워크** | "더 빨리 배포하라" | "에러 버짓이 허용하면 배포하라" |
| **자동화 목표** | 모든 것을 자동화 | 토일 제거를 위한 자동화 (운영 작업 < 50%) |
| **실패 접근 방식** | "빠르게 움직이고, 앞으로 수정하라" | "지속 가능한 속도로 움직이고, 모든 실패에서 배워라" |

### 1.3 SRE 팀 모델

| 모델 | 설명 | 적합 대상 |
|-------|-------------|---------|
| **중앙 집중형 SRE** | 전담 SRE 팀이 여러 제품 팀을 지원 | 대규모 조직, 복잡한 시스템 |
| **임베디드 SRE** | SRE 엔지니어가 각 제품 팀에 배치 | 중규모 조직, 긴밀한 협업 |
| **컨설팅 SRE** | SRE 팀이 제품 팀에 조언하되, 서비스를 직접 운영하지 않음 | 성장 중인 조직, 지식 공유 |
| **모두가 SRE** | 전담 SRE 팀 없음; 개발자가 자체 안정성을 소유 | 소규모 조직, 강한 엔지니어링 문화 |

---

## 2. SLO, SLI, SLA

### 2.1 정의

```
┌─────────────────────────────────────────────────────────────────┐
│                    SLI → SLO → SLA                               │
│                                                                  │
│  SLI (Service Level Indicator)                                   │
│  └── A quantitative measure of service behavior                  │
│      Example: "The proportion of requests that complete in       │
│               less than 300ms"                                   │
│                                                                  │
│  SLO (Service Level Objective)                                   │
│  └── A target value for an SLI                                   │
│      Example: "99.9% of requests complete in < 300ms,            │
│               measured over a 30-day rolling window"             │
│                                                                  │
│  SLA (Service Level Agreement)                                   │
│  └── A contract with consequences for missing the SLO            │
│      Example: "If availability drops below 99.9% in a month,    │
│               the customer receives a 10% service credit"        │
│                                                                  │
│  Relationship: SLI measures → SLO sets target → SLA adds teeth  │
│  Rule: SLO should be stricter than SLA (buffer for internal use)│
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 SLI 선택

| 서비스 유형 | 권장 SLI |
|-------------|-----------------|
| **요청 처리 (API)** | 가용성 (성공 응답 %), 지연 시간 (p50, p95, p99), 에러율 |
| **데이터 처리 (파이프라인)** | 신선도 (마지막 성공 실행 이후 시간), 정확도 (유효한 출력 %), 커버리지 (처리된 데이터 %) |
| **스토리지 (데이터베이스)** | 가용성, 내구성 (손실되지 않은 데이터 %), 읽기/쓰기 지연 시간 |
| **스트리밍 (메시지 큐)** | 엔드투엔드 지연 시간, 메시지 손실률, 컨슈머 랙 |

### 2.3 SLO 정의

```yaml
# SLO definition for an API service
service: order-service
slos:
  - name: availability
    description: "Proportion of successful HTTP responses"
    sli:
      type: availability
      good_events: "http_requests_total{status!~'5..'}"
      total_events: "http_requests_total"
    objective: 99.9          # 99.9% availability
    window: 30d              # 30-day rolling window
    # Budget: 0.1% of requests can fail
    # At 10,000 req/min = 10 failed req/min allowed
    # At 10,000 req/min * 30 days = ~432M total requests
    # Error budget = 432,000 allowed failures

  - name: latency
    description: "Proportion of requests faster than 300ms"
    sli:
      type: latency
      good_events: "http_request_duration_seconds_bucket{le='0.3'}"
      total_events: "http_request_duration_seconds_count"
    objective: 99.0          # 99% of requests < 300ms
    window: 30d

  - name: latency-p99
    description: "99th percentile request duration"
    sli:
      type: latency_percentile
      query: "histogram_quantile(0.99, sum by (le) (rate(http_request_duration_seconds_bucket[5m])))"
    objective_value: 1.0     # p99 < 1 second
    window: 30d
```

### 2.4 Nines 테이블

| 가용성 | 연간 다운타임 | 월간 다운타임 | 주간 다운타임 |
|-------------|--------------|---------------|--------------|
| 99% (two nines) | 3.65일 | 7.3시간 | 1.68시간 |
| 99.9% (three nines) | 8.77시간 | 43.8분 | 10.1분 |
| 99.95% | 4.38시간 | 21.9분 | 5.04분 |
| 99.99% (four nines) | 52.6분 | 4.38분 | 1.01분 |
| 99.999% (five nines) | 5.26분 | 26.3초 | 6.05초 |

**경험 법칙:** 각 추가 nine은 달성하는 데 10배의 비용이 듭니다. 99.9%에서 99.99%로 올리려면 근본적으로 다른 아키텍처(멀티 리전, 자동 페일오버, 단일 장애 지점 없음)가 필요합니다.

### 2.5 Prometheus를 사용한 SLO 구현

```yaml
# Prometheus recording rules for SLO tracking
groups:
  - name: slo_rules
    interval: 30s
    rules:
      # Total requests in the window
      - record: slo:http_requests:total_rate30d
        expr: sum(rate(http_requests_total{job="order-service"}[30d]))

      # Successful requests in the window
      - record: slo:http_requests:good_rate30d
        expr: sum(rate(http_requests_total{job="order-service",status!~"5.."}[30d]))

      # Current availability (SLI)
      - record: slo:http_requests:availability30d
        expr: slo:http_requests:good_rate30d / slo:http_requests:total_rate30d

      # Error budget remaining (1 = 100% remaining, 0 = exhausted)
      - record: slo:http_requests:error_budget_remaining
        expr: |
          1 - (
            (1 - slo:http_requests:availability30d)
            / (1 - 0.999)
          )

  - name: slo_alerts
    rules:
      # Alert when error budget is below 25%
      - alert: ErrorBudgetLow
        expr: slo:http_requests:error_budget_remaining < 0.25
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Error budget below 25% for order-service"
          description: "Only {{ $value | humanizePercentage }} of error budget remaining."

      # Alert when error budget is exhausted
      - alert: ErrorBudgetExhausted
        expr: slo:http_requests:error_budget_remaining < 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Error budget exhausted for order-service"
          description: "SLO is being violated. Halt feature launches."
```

---

## 3. 에러 버짓(Error Budgets)

### 3.1 에러 버짓이란

에러 버짓은 SLO의 역수입니다 -- 허용할 수 있는 비신뢰성의 양입니다.

```
SLO: 99.9% availability

Error Budget = 100% - 99.9% = 0.1%

Over 30 days with 1,000,000 requests/day:
Total requests = 30,000,000
Error budget   = 30,000,000 * 0.001 = 30,000 allowed failures

┌────────────────────────────────────────────────────────────┐
│  Error Budget Over 30 Days                                  │
│                                                             │
│  100% ┤████████████████████████████████████████  Budget     │
│       │████████████████████████████████████████  Remaining  │
│  75%  ┤███████████████████████████████████                  │
│       │████████████████████████████████                     │
│  50%  ┤████████████████████████████                         │
│       │██████████████████████████                           │
│  25%  ┤████████████████████████      ← Warning threshold   │
│       │█████████████████████                                │
│  0%   ┤█████████████████             ← Budget exhausted    │
│       └──────────────────────────────────────────────────── │
│        Day 1    Day 10    Day 20    Day 30                  │
└────────────────────────────────────────────────────────────┘
```

### 3.2 에러 버짓 정책

에러 버짓 정책은 버짓이 양호하거나, 낮거나, 소진되었을 때 어떤 조치를 취해야 하는지 정의합니다:

| 버짓 상태 | 조치 |
|--------------|--------|
| **> 50% 남음** | 정상 운영. 자유롭게 기능 배포. 카오스 실험 실행. |
| **25-50% 남음** | 주의. 안정성 작업 우선. 배포 빈도 감소. 비필수 실험 취소. |
| **< 25% 남음** | 경고. 기능 출시 동결. 모든 엔지니어링 노력을 안정성에 집중. 인시던트 리뷰 필요. |
| **소진 (0%)** | 동결. 안정성 수정을 제외한 배포 없음. 버짓이 회복될 때까지 긴급 안정성 스프린트. |

### 3.3 에러 버짓 계산 예제

```python
# Error budget tracking example
from datetime import datetime, timedelta

class ErrorBudgetTracker:
    def __init__(self, slo_target: float, window_days: int):
        self.slo_target = slo_target
        self.window_days = window_days
        self.error_budget = 1 - slo_target  # e.g., 0.001 for 99.9%

    def budget_remaining(self, total_requests: int, failed_requests: int) -> float:
        """Returns the fraction of error budget remaining (0.0 to 1.0)."""
        allowed_failures = total_requests * self.error_budget
        actual_failures = failed_requests
        if allowed_failures == 0:
            return 0.0
        return max(0, (allowed_failures - actual_failures) / allowed_failures)

    def burn_rate(self, total_requests: int, failed_requests: int) -> float:
        """
        Returns the burn rate.
        1.0 = consuming budget at exactly the expected rate
        2.0 = consuming budget at 2x the expected rate (will exhaust in half the window)
        """
        actual_error_rate = failed_requests / total_requests if total_requests > 0 else 0
        return actual_error_rate / self.error_budget

# Example:
tracker = ErrorBudgetTracker(slo_target=0.999, window_days=30)
budget = tracker.budget_remaining(total_requests=10_000_000, failed_requests=5_000)
burn = tracker.burn_rate(total_requests=10_000_000, failed_requests=5_000)
print(f"Budget remaining: {budget:.1%}")  # 50.0%
print(f"Burn rate: {burn:.1f}x")          # 5.0x (consuming 5x faster than allowed)
```

### 3.4 멀티 윈도우 번 레이트 알림

Google에서 권장하는 SLO 알림 접근 방식은 빠른 번(장애)과 느린 번(성능 저하) 모두를 감지하기 위해 여러 시간 윈도우를 사용합니다:

```yaml
# Fast burn: 14.4x burn rate over 1 hour (exhausts 30-day budget in 2 days)
- alert: SLOFastBurn
  expr: |
    (
      sum(rate(http_requests_total{job="order-service",status=~"5.."}[1h]))
      / sum(rate(http_requests_total{job="order-service"}[1h]))
    ) > (14.4 * 0.001)
  for: 2m
  labels:
    severity: critical
  annotations:
    summary: "Fast error budget burn (14.4x) - order-service"

# Slow burn: 3x burn rate over 6 hours (exhausts budget in 10 days)
- alert: SLOSlowBurn
  expr: |
    (
      sum(rate(http_requests_total{job="order-service",status=~"5.."}[6h]))
      / sum(rate(http_requests_total{job="order-service"}[6h]))
    ) > (3 * 0.001)
  for: 15m
  labels:
    severity: warning
  annotations:
    summary: "Slow error budget burn (3x) - order-service"
```

---

## 4. 토일 제거(Toil Elimination)

### 4.1 토일이란

토일은 프로덕션 서비스 운영과 관련된 작업 중 수동적이고, 반복적이며, 자동화 가능하고, 전술적이며, 지속적인 가치가 없고, 서비스 성장에 비례하여 확대되는 종류의 작업입니다.

| 특성 | 토일 | 토일이 아닌 것 |
|---------------|------|----------|
| **수동** | 사람이 스크립트를 실행하거나 버튼을 클릭 | 자동화된 파이프라인 |
| **반복적** | 매주/매일/매시간 동일한 작업 | 일회성 프로젝트 |
| **자동화 가능** | 소프트웨어가 수행할 수 있음 | 사람의 판단이 필요 |
| **전술적** | 인터럽트 기반, 사후 대응적 | 계획된, 전략적 |
| **지속적 가치 없음** | 시스템이 이전 상태로 복귀 | 영구적 개선 |
| **성장에 비례** | 더 많은 사용자 = 더 많은 토일 | 시스템이 성장을 자동으로 처리 |

### 4.2 예시

| 토일 | 엔지니어링 솔루션 |
|------|---------------------|
| 크래시된 파드를 수동으로 재시작 | liveness probe와 자동 재시작 설정 |
| 트래픽 급증 시 수동으로 스케일링 | Horizontal Pod Autoscaler 구현 |
| TLS 인증서를 수동으로 교체 | 자동 갱신이 포함된 cert-manager 배포 |
| 일상적인 배포를 수동으로 검토하고 승인 | 자동화된 카나리 분석 구현 |
| 주간 데이터베이스 유지보수 쿼리 실행 | 자동화된 검증이 포함된 CronJob 스케줄링 |
| 사용자 계정을 수동으로 프로비저닝 | 셀프서비스 사용자 프로비저닝 구축 |

### 4.3 50% 규칙

Google의 SRE 원칙: **SRE 팀은 토일에 시간의 50% 이상을 사용해서는 안 됩니다.** 나머지 50%는 토일을 영구적으로 줄이거나 안정성을 개선하는 엔지니어링 작업에 사용해야 합니다.

```
Healthy SRE Team:                  Unhealthy SRE Team:
┌─────────────────────┐           ┌─────────────────────┐
│  Engineering  50%   │           │  Engineering  20%   │
│  ████████████████   │           │  ██████             │
│                     │           │                     │
│  Toil         50%   │           │  Toil         80%   │
│  ████████████████   │           │  ████████████████   │
│                     │           │  ████████████████   │
│                     │           │  ████████████████   │
└─────────────────────┘           └─────────────────────┘
 Result: Toil decreases            Result: Team burns out,
 over time as automation            never improves, sinks
 replaces manual work               deeper into toil
```

### 4.4 토일 추적

```yaml
# Toil budget tracking (quarterly review)
team: sre-team
quarter: Q1-2024
team_size: 5
total_hours: 5 * 40 * 13  # 2,600 hours

toil_budget: 1300  # 50% max

toil_items:
  - name: "Manual certificate rotation"
    hours_per_quarter: 40
    frequency: "Monthly per service (20 services)"
    automation_effort: "80 hours one-time"
    priority: high

  - name: "Incident triage and paging"
    hours_per_quarter: 260
    frequency: "On-call rotation"
    automation_effort: "Improve alerting to reduce false positives (120 hours)"
    priority: high

  - name: "Capacity planning reviews"
    hours_per_quarter: 80
    frequency: "Weekly"
    automation_effort: "Auto-scaling + predictive alerts (200 hours)"
    priority: medium

  - name: "Database schema migration assistance"
    hours_per_quarter: 60
    frequency: "Per-request"
    automation_effort: "Self-service migration tool (160 hours)"
    priority: low

total_toil: 440  # 17% of total hours (healthy)
```

---

## 5. 인시던트 관리(Incident Management)

### 5.1 인시던트 수명 주기

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│  Detect  │──→│  Triage  │──→│ Mitigate │──→│ Resolve  │──→│  Review  │
│          │   │          │   │          │   │          │   │(Postmort)│
│ Alert    │   │ Severity │   │ Stop the │   │ Fix root │   │ Learn &  │
│ fires    │   │ assign   │   │ bleeding │   │ cause    │   │ prevent  │
│          │   │ IC named │   │          │   │          │   │          │
└──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
  Minutes        Minutes       Minutes-Hours    Hours-Days      Days
```

### 5.2 심각도 수준

| 심각도 | 영향 | 응답 시간 | 예시 |
|----------|--------|--------------|---------|
| **SEV-1 (치명적)** | 전체 서비스 장애, 데이터 손실, 보안 침해 | 즉시 (< 5분) | 사이트 다운, 데이터베이스 손상, 활발한 침해 |
| **SEV-2 (주요)** | 심각한 기능 저하, 부분 장애 | < 15분 | 결제 처리 중단, 50% 에러율 |
| **SEV-3 (경미)** | 경미한 기능 영향, 성능 저하 | < 1시간 | 느린 응답, 비핵심 기능 장애 |
| **SEV-4 (낮음)** | 사용자 영향 없음, 내부 도구 문제 | 다음 영업일 | 내부 대시보드 다운, 개발 환경 문제 |

### 5.3 인시던트 역할

| 역할 | 책임 |
|------|----------------|
| **Incident Commander (IC)** | 대응을 조율합니다. 결정을 내립니다. 작업을 위임합니다. 직접 디버그하지 않습니다. |
| **Communications Lead** | 이해관계자, 상태 페이지, Slack 채널을 업데이트합니다. 외부 커뮤니케이션을 관리합니다. |
| **Operations Lead** | 기술적 완화(재시작, 롤백, 스케일링)를 실행합니다. |
| **Subject Matter Expert (SME)** | 영향을 받는 시스템에 대한 깊은 지식을 가집니다. 근본 원인을 조사합니다. |
| **Scribe** | 타임라인, 수행한 조치, 결정 사항을 인시던트 문서에 기록합니다. |

### 5.4 인시던트 커뮤니케이션 템플릿

```markdown
# Incident: [Title]
**Severity**: SEV-[1/2/3]
**Status**: [Investigating / Identified / Monitoring / Resolved]
**Incident Commander**: [Name]
**Start Time**: [ISO 8601]

## Impact
[What is broken? How many users are affected?]

## Timeline
| Time (UTC) | Event |
|-----------|-------|
| 14:23 | Alert: HighErrorRate fired for order-service |
| 14:25 | IC assigned: [Name] |
| 14:28 | Identified: Database connection pool exhausted |
| 14:32 | Mitigation: Restarted order-service pods |
| 14:35 | Monitoring: Error rate dropping |
| 14:45 | Resolved: Error rate back to baseline |

## Root Cause
[Brief description]

## Mitigation Applied
[What was done to stop the bleeding]

## Action Items
- [ ] [Action 1] — Owner: [Name] — Due: [Date]
- [ ] [Action 2] — Owner: [Name] — Due: [Date]
```

### 5.5 인시던트 대응 도구

| 도구 | 목적 |
|------|---------|
| **PagerDuty / OpsGenie** | 알림, 온콜 관리, 에스컬레이션 |
| **Slack / Teams** | 실시간 인시던트 커뮤니케이션 채널 |
| **Statuspage** | 외부 커뮤니케이션 (고객 대면 상태) |
| **Jira / Linear** | 인시던트 후 액션 아이템 추적 |
| **Grafana / Datadog** | 인시던트 중 실시간 메트릭 시각화 |
| **Rootly / Incident.io** | 자동화된 인시던트 관리 워크플로우 |

---

## 6. 포스트모텀(Postmortems)

### 6.1 비난 없는 포스트모텀

비난 없는 포스트모텀은 무엇이 일어났고 어떻게 방지할 것인지에 초점을 맞추며, 누가 원인을 제공했는지에 초점을 맞추지 않습니다. 전제는 사람들이 당시 가용한 정보를 기반으로 합리적인 결정을 내린다는 것입니다. 누군가 나쁜 변경을 배포했다면, 시스템이 그것을 잡았어야 합니다 -- 사람이 아니라.

**핵심 원칙:**
- **사람**이 아니라 **시스템**과 **프로세스**에 초점을 맞춥니다
- 관련된 모든 사람의 선의를 가정합니다
- "누가 실패를 일으켰나?"가 아니라 "무엇이 시스템을 실패하게 했나?"를 묻습니다
- 목표는 비난을 할당하는 것이 아니라 시스템을 더 안전하게 만드는 것입니다

### 6.2 포스트모텀 템플릿

```markdown
# Postmortem: [Incident Title]

**Date**: [Date of incident]
**Duration**: [Start time — End time]
**Severity**: SEV-[1/2/3]
**Authors**: [Names]
**Status**: [Draft / Final]

## Summary
[2-3 sentence summary of what happened and the impact]

## Impact
- **Duration**: X hours Y minutes
- **Users affected**: N users / N% of traffic
- **Revenue impact**: $X (if applicable)
- **Data lost**: [None / Description]

## Root Cause
[Technical description of what broke and why]

## Trigger
[What specific event initiated the incident?
 e.g., "A configuration change at 14:15 UTC increased the connection pool timeout from 5s to 60s"]

## Detection
- **How detected**: [Alert / Customer report / Manual observation]
- **Time to detect**: [Minutes from trigger to first alert]
- **Detection gap**: [Was there a monitoring gap? Should an alert have fired earlier?]

## Timeline
| Time (UTC) | Event |
|-----------|-------|
| 14:15 | Config change deployed via CI/CD |
| 14:23 | Error rate exceeds 5% — HighErrorRate alert fires |
| 14:25 | IC assigned. War room opened in Slack #inc-20240315 |
| 14:28 | SME identifies connection pool exhaustion in logs |
| 14:32 | Mitigation: rollback config change |
| 14:35 | Error rate declining |
| 14:45 | Error rate back to baseline. Incident resolved. |

## Root Cause Analysis
[Detailed technical analysis using the "5 Whys" or fishbone diagram]

**5 Whys:**
1. Why did orders fail? → Database connections timed out
2. Why did connections time out? → Connection pool was exhausted
3. Why was the pool exhausted? → Timeout was set to 60s (should be 5s)
4. Why was the timeout changed? → Config change in PR #1234
5. Why was the bad config merged? → No validation or staging test for timeout values

## What Went Well
- Alert fired within 8 minutes of the trigger
- IC was assigned within 2 minutes of the alert
- Rollback was clean and fast (3 minutes)

## What Went Wrong
- Config change was not tested in staging before production
- No automated validation for connection pool timeout values
- Runbook for "connection pool exhaustion" did not exist

## Where We Got Lucky
- The incident happened during business hours with the full team available
- The config change was easily reversible

## Action Items
| # | Action | Type | Owner | Due Date | Status |
|---|--------|------|-------|----------|--------|
| 1 | Add config validation for timeout values (min: 1s, max: 30s) | Prevention | @alice | 2024-03-22 | TODO |
| 2 | Require staging deployment before production for config changes | Prevention | @bob | 2024-03-29 | TODO |
| 3 | Create runbook for connection pool exhaustion | Mitigation | @carol | 2024-03-20 | TODO |
| 4 | Add alert for connection pool utilization > 80% | Detection | @dave | 2024-03-18 | DONE |

## Lessons Learned
[What did we learn that applies beyond this specific incident?]
```

### 6.3 포스트모텀 리뷰 프로세스

1. **작성**: IC와 SME가 48시간 이내에 포스트모텀 초안 작성
2. **리뷰**: 팀이 정확성과 완전성을 검토
3. **공유**: 엔지니어링 조직에 공개 (투명성)
4. **추적**: 액션 아이템을 이슈 트래커에서 추적
5. **후속 조치**: 2주 후 액션 아이템 완료 여부 검토

---

## 7. 온콜 프랙티스(On-Call Practices)

### 7.1 지속 가능한 온콜

| 프랙티스 | 설명 |
|----------|-------------|
| **로테이션 규모** | 로테이션당 최소 8명 (8주에 한 번 이상 온콜하지 않도록 보장) |
| **교대 기간** | 1주 프라이머리, 1주 세컨더리. 연속 주 없음. |
| **보상** | 온콜 수당 또는 보상 휴가 |
| **페이지 볼륨** | 온콜 교대당 < 2 페이지 목표. 5 이상이면 조치 필요. |
| **인수인계** | 로테이션 교체 시 서면 인수인계 문서 (활성 인시던트, 알려진 위험) |
| **에스컬레이션** | 명확한 에스컬레이션 경로: 프라이머리 → 세컨더리 → 팀 리드 → 매니지먼트 |
| **Follow-the-sun** | 글로벌 팀의 경우, 야간 페이지 대신 다음 시간대로 인계 |

### 7.2 온콜 건강 메트릭

| 메트릭 | 건강 | 주의 필요 | 지속 불가능 |
|--------|---------|----------------|--------------|
| **주간 페이지** | 0-2 | 3-5 | > 5 |
| **업무 외 시간 페이지** | 0-1 | 2-3 | > 3 |
| **MTTA (평균 확인 시간)** | < 5분 | 5-15분 | > 15분 |
| **오탐률** | < 10% | 10-30% | > 30% |
| **토일 비율** | < 30% | 30-50% | > 50% |
| **수면 방해** | 주당 0-1회 | 주당 2-3회 | 주당 > 3회 |

### 7.3 온콜 부담 줄이기

```
┌──────────────────────────────────────────────────────────────┐
│              On-Call Improvement Flywheel                      │
│                                                               │
│  1. Measure page volume and false positives                   │
│       │                                                       │
│       ▼                                                       │
│  2. Fix top sources of pages                                  │
│     ├── Tune noisy alerts (reduce false positives)            │
│     ├── Auto-remediate common issues (self-healing)           │
│     └── Fix root causes (not just symptoms)                   │
│       │                                                       │
│       ▼                                                       │
│  3. Fewer pages → better sleep → better engineering           │
│       │                                                       │
│       ▼                                                       │
│  4. Better engineering → more automation → fewer pages        │
│       │                                                       │
│       └──────────────→ (repeat)                               │
└──────────────────────────────────────────────────────────────┘
```

### 7.4 온콜 인수인계 템플릿

```markdown
# On-Call Handoff: [Date]

## Outgoing: [Name]
## Incoming: [Name]

## Active Issues
- [ ] order-service showing elevated p99 latency since Tuesday.
      Likely related to DB index rebuild. Monitor #alert-backend.

## Recent Incidents
- SEV-2 on Monday: Payment timeout. Root cause fixed. Postmortem in progress.

## Upcoming Risks
- Database migration scheduled for Thursday 2 AM UTC.
  Runbook: [link]. Rollback plan: [link].

## Known Alert Noise
- `DiskSpaceLow` on node-7 is a false positive (large log file, rotation scheduled).
  Silence ID: abc123 (expires Friday).

## Tips
- If order-service pages, check the Redis cluster first (it was flaky last week).
- The staging environment is broken; do not deploy there until PR #456 is merged.
```

---

## 8. SRE 의사 결정 프레임워크

### 8.1 에러 버짓을 활용한 의사 결정

```
Decision: "Should we deploy Feature X to production?"

┌──────────────────────────────────────────────────────────┐
│                                                          │
│  Step 1: Check error budget                              │
│  ├── Budget > 50%  →  Yes, deploy freely                │
│  ├── Budget 25-50% →  Deploy with extra monitoring      │
│  ├── Budget < 25%  →  Deploy only if it improves        │
│  │                    reliability                        │
│  └── Budget = 0%   →  No. Only reliability fixes.       │
│                                                          │
│  Step 2: Check deployment risk                           │
│  ├── Low risk (config change, minor UI)  → Rolling      │
│  ├── Medium risk (new feature)           → Canary       │
│  └── High risk (database migration)      → Blue-green   │
│                                                          │
│  Step 3: Monitor after deployment                        │
│  ├── Error budget burn rate < 1x → Healthy              │
│  ├── Error budget burn rate 1-3x → Monitor closely      │
│  └── Error budget burn rate > 3x → Rollback             │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 8.2 안정성 vs 속도 트레이드오프

```
                    Error Budget Remaining
  100% ┤═══════════════════════════════════════════
       │        "Ship features freely"
       │
  75%  ┤─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
       │        "Normal velocity"
       │
  50%  ┤═══════════════════════════════════════════
       │        "Slow down, prioritize reliability"
       │
  25%  ┤─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
       │        "Feature freeze, reliability sprint"
       │
   0%  ┤═══════════════════════════════════════════
       │        "SLO violated, freeze all deploys"
       └────────────────────────────────────────────
```

---

## 9. 다음 단계

- [10_Monitoring_and_Alerting.md](./10_Monitoring_and_Alerting.md) - SRE의 기반이 되는 모니터링 기초
- [16_Chaos_Engineering.md](./16_Chaos_Engineering.md) - 사전 안정성 테스트

---

## 연습 문제(Exercises)

### 연습 문제 1: SLO 설계

다음 서비스에 대한 SLO를 설계하십시오:

1. 하루 50,000건의 요청을 처리하는 이커머스 결제 API
2. 모바일 앱의 클릭스트림 이벤트를 처리하는 데이터 파이프라인
3. 영업 시간 동안 20명의 직원이 사용하는 내부 관리자 대시보드

<details>
<summary>정답 보기</summary>

**1. 이커머스 결제 API:**

| SLO | SLI | 목표 | 윈도우 | 근거 |
|-----|-----|--------|--------|-----------|
| 가용성 | `successful_responses / total_responses` | 99.95% | 30일 | 결제에 중요; 99.9%는 하루 ~50건의 실패를 허용하며, 결제에는 너무 많음. 99.95%는 하루 ~25건을 허용. |
| 지연 시간 (빠른) | `requests < 500ms / total_requests` | 99% | 30일 | 대부분의 결제는 빨라야 함; 느린 결제는 장바구니 이탈을 유발. |
| 지연 시간 (상한) | p99 지연 시간 | < 2초 | 30일 | 가장 느린 요청도 사용자 인내심이 소진되기 전에 완료되어야 함. |

**에러 버짓**: 30일 동안 99.95%, 하루 50,000건 = 총 1,500,000건. 버짓 = 월 750건의 허용 실패 (하루 25건).

**2. 데이터 파이프라인:**

| SLO | SLI | 목표 | 윈도우 | 근거 |
|-----|-----|--------|--------|-----------|
| 신선도 | `time_since_last_successful_run` | < 15분 | 롤링 | 클릭스트림 데이터는 준실시간 개인화에 사용됨. |
| 정확도 | `valid_records_output / total_records_input` | 99.9% | 30일 | 분석용으로 < 0.1%의 이벤트 누락은 허용 가능. |
| 커버리지 | `records_processed / records_received` | 99.99% | 30일 | 데이터가 조용히 누락되는 일은 거의 없어야 함. |

**3. 내부 관리자 대시보드:**

| SLO | SLI | 목표 | 윈도우 | 근거 |
|-----|-----|--------|--------|-----------|
| 가용성 | `successful_responses / total_responses` | 99% (영업 시간 중) | 30일, 영업 시간만 | 20명의 사용자를 가진 내부 도구; 간헐적 다운타임은 허용 가능. |
| 지연 시간 | p95 페이지 로드 시간 | < 3초 | 30일 | 내부 사용자는 느린 페이지에 더 관대. |

**핵심 통찰**: 결제 API (99.95%)는 관리자 대시보드 (99%)보다 훨씬 엄격한 SLO를 가집니다. 실패의 비즈니스 영향이 수십 배 다르기 때문입니다. SLO는 기술적 역량이 아니라 비즈니스 영향을 반영해야 합니다.

</details>

### 연습 문제 2: 에러 버짓 시나리오

order-service는 30일 롤링 윈도우에서 99.9% 가용성의 SLO를 가지고 있습니다. 하루 100,000건의 요청을 처리합니다.

15일째에 잘못된 배포로 인해 2시간 동안 장애가 발생했고, 이 기간 동안 80%의 요청이 실패했습니다. 다음을 계산하십시오:
1. 30일 윈도우의 총 에러 버짓
2. 장애로 인해 소비된 버짓 양
3. 장애 후 남은 버짓
4. 에러 버짓 정책에 따라 취해야 할 조치

<details>
<summary>정답 보기</summary>

**1. 총 에러 버짓:**
```
Total requests in 30 days = 100,000/day * 30 days = 3,000,000
Error budget = 3,000,000 * (1 - 0.999) = 3,000 allowed failures
```

**2. 장애로 인해 소비된 버짓:**
```
Outage duration: 2 hours
Requests during outage: 100,000/day / 24 hours * 2 hours = 8,333 requests
Failed requests: 8,333 * 80% = 6,667 failures
```

**3. 남은 버짓:**
```
Assuming no other failures in the 30-day window:
Budget consumed: 6,667 failures
Budget remaining: 3,000 - 6,667 = -3,667

The error budget is EXHAUSTED (negative).
The team has exceeded their SLO by 3,667 failures.
```

잠깐 -- 소비된 버짓(6,667)이 전체 30일 버짓(3,000)을 초과합니다. 이는 다음을 의미합니다:

```
Budget remaining as percentage: (3,000 - 6,667) / 3,000 = -122%
The SLO is violated.
```

**4. 에러 버짓 정책에 따른 조치:**

버짓이 소진되었고 (상당히 초과되었으므로), 다음 조치가 적용됩니다:

1. **즉각적인 기능 동결**: 에러 버짓이 회복될 때까지 새로운 기능 배포 없음. 안정성 수정만 배포합니다.

2. **필수 포스트모텀**: 48시간 이내에 비난 없는 포스트모텀을 수행하며 다음을 다룹니다:
   - 잘못된 배포가 왜 스테이징에서 발견되지 않았는지
   - 카나리/롤백이 왜 2시간의 영향을 방지하지 못했는지
   - 모니터링이 왜 문제를 더 빨리 감지하지 못했는지

3. **안정성 스프린트**: 다음 스프린트를 전적으로 안정성 개선에 할당합니다:
   - 자동화된 카나리 분석 구현 (아직 없는 경우)
   - 배포 검증 테스트 추가
   - 롤백 속도 개선

4. **회복 타임라인**: 30일 롤링 윈도우는 장애 날짜(15일째)가 45일째에 윈도우에서 벗어나면서 버짓이 "회복"됨을 의미합니다. 그러나 팀은 단순히 기다리면 안 됩니다 -- 정상적인 배포 주기를 재개하기 전에 안정성 개선을 입증해야 합니다.

5. **커뮤니케이션**: SLO 위반을 이해관계자에게 보고합니다. 외부 SLA가 있는 경우, SLA도 위반되었는지 확인하고 어떤 계약 의무가 적용되는지 확인합니다.

</details>

### 연습 문제 3: 토일 분류

다음 각 작업을 **토일** 또는 **토일 아님**으로 분류하고, 근거를 설명하십시오:

1. 매일 밤 2시에 데이터베이스 백업 스크립트 실행
2. 새로운 캐싱 레이어에 대한 설계 문서 작성
3. 메모리 누수로 인해 3일마다 크래시되는 서비스 재시작
4. 팀원들의 풀 리퀘스트 검토 및 머지
5. 회사에 합류한 신규 사용자를 모니터링 시스템에 수동으로 추가
6. 이전에 발생한 적 없는 새로운 프로덕션 인시던트 조사

<details>
<summary>정답 보기</summary>

1. **매일 밤 2시에 데이터베이스 백업 스크립트 실행 -- 토일**
   - 수동: 누군가 스크립트를 실행하거나 트리거해야 합니다.
   - 반복적: 매일 밤.
   - 자동화 가능: CronJob이나 관리형 백업 서비스가 이를 수행할 수 있습니다.
   - 지속적 가치 없음: 각 백업은 소비되고 대체됩니다.
   - **해결**: AWS RDS 자동 백업 사용, 또는 검증이 포함된 Kubernetes CronJob 사용.

2. **새로운 캐싱 레이어에 대한 설계 문서 작성 -- 토일 아님**
   - 사람의 판단과 창의성이 필요합니다.
   - 지속적 가치를 가진 일회성 작업입니다 (설계가 향후 작업에 정보를 제공).
   - 이것은 엔지니어링 작업입니다.

3. **메모리 누수로 인해 3일마다 크래시되는 서비스 재시작 -- 토일**
   - 수동적이고 반복적입니다.
   - 자동화 가능합니다 (하지만 단순히 자동화하면 안 됩니다 -- 메모리 누수를 수정해야 합니다).
   - 지속적 가치 없음 (3일 후 서비스가 다시 크래시됩니다).
   - **해결**: 메모리 누수 수정 (근본 원인). 임시 조치로 메모리 기반 재시작이 포함된 liveness probe를 추가하되, 영구적 수정을 우선시합니다.

4. **풀 리퀘스트 검토 및 머지 -- 토일 아님**
   - 사람의 판단이 필요합니다 (코드 품질, 정확성, 설계).
   - 의미 있는 방식으로 자동화할 수 없습니다 (린팅은 가능하지만, 리뷰는 불가능).
   - 지속적 가치를 제공합니다 (지식 공유, 품질 강화).

5. **신규 사용자를 모니터링 시스템에 수동으로 추가 -- 토일**
   - 수동적이고, 반복적이며, 자동화 가능합니다.
   - 회사 성장에 비례하여 확대됩니다.
   - 지속적 가치 없음 (매번 동일한 프로세스).
   - **해결**: 모니터링을 ID 공급자 (SSO/LDAP)와 통합합니다. 신규 사용자는 자동으로 프로비저닝됩니다.

6. **새로운 프로덕션 인시던트 조사 -- 토일 아님**
   - 반복적이지 않음 (정의상 새로운 인시던트).
   - 사람의 판단과 깊은 기술 분석이 필요합니다.
   - 지속적 가치를 제공합니다 (근본 원인 이해, 포스트모텀, 액션 아이템).
   - 이것이 바로 SRE가 해야 할 종류의 작업입니다.

</details>

### 연습 문제 4: 포스트모텀 작성

다음 사실로 인시던트가 발생했습니다:

- **서비스**: payment-service
- **기간**: 45분 (09:15 - 10:00 UTC)
- **영향**: 신용카드 결제 100% 실패; 직불카드 결제는 영향 없음
- **근본 원인**: 서드파티 결제 프로세서 (Stripe)가 `/v1/charges` 엔드포인트에 호환성을 깨는 API 변경을 배포. 응답 형식이 `{"id": "ch_xxx"}`에서 `{"charge": {"id": "ch_xxx"}}`로 변경.
- **감지**: 알림이 발생하기 전에 고객 지원팀이 15건의 불만을 접수. 첫 실패 후 12분 뒤에 알림 발생.
- **완화**: 두 가지 응답 형식을 모두 파싱하는 핫픽스 배포.

예방, 감지, 완화로 분류된 최소 5개의 액션 아이템으로 포스트모텀의 액션 아이템 섹션을 작성하십시오.

<details>
<summary>정답 보기</summary>

**액션 아이템:**

| # | 조치 | 카테고리 | 담당자 | 기한 | 우선순위 |
|---|--------|----------|-------|----------|----------|
| 1 | **Stripe API 클라이언트에 응답 스키마 검증 추가**: 파싱 전에 Stripe 응답의 구조를 검증합니다. 스키마가 예상과 다르면, 처리 실패 전에 구조화된 에러를 로깅하고 알림을 트리거합니다. | 예방 | @payments-team | 2024-04-01 | P1 |
| 2 | **Stripe API 변경 로그 및 폐기 공지 구독**: Stripe의 API 변경 로그 RSS 피드에 대한 자동화된 모니터링을 설정합니다. 호환성을 깨는 변경이 발표되면 결제 팀에 알림을 보냅니다. | 예방 | @payments-team | 2024-03-25 | P2 |
| 3 | **요청 헤더에 Stripe API 버전 고정**: `Stripe-Version: 2024-02-01` 헤더를 사용하여 통합을 알려진 API 버전으로 잠그고, 예상치 못한 변경이 영향을 미치는 것을 방지합니다. | 예방 | @payments-team | 2024-03-22 | P1 |
| 4 | **알림 갭 수정: 결제 실패율 알림 추가**: 신용카드 결제 실패율이 2분 연속 5%를 초과하면 발생하는 Prometheus 알림을 생성합니다. 현재 알림 임계값이 너무 높았으며 (20%), 이로 인해 감지가 12분 지연되었습니다. | 감지 | @sre-team | 2024-03-20 | P0 |
| 5 | **합성 결제 모니터링 추가**: 5분마다 $0.50 테스트 결제를 실행하는 합성 테스트를 배포합니다. 테스트 결제가 실패하면 즉시 알림을 보냅니다. 이렇게 하면 실제 고객이 영향을 받기 전에 문제를 감지할 수 있었을 것입니다. | 감지 | @sre-team | 2024-04-05 | P1 |
| 6 | **"결제 프로세서 API 장애" 런북 생성**: 외부 결제 API 장애를 진단하고 완화하는 단계를 문서화합니다. 포함 내용: 어떤 결제 수단이 영향을 받는지 식별하는 방법, 백업 프로세서로 트래픽을 라우팅하는 방법, 핫픽스를 배포하는 방법. | 완화 | @payments-team | 2024-03-25 | P2 |
| 7 | **결제 프로세서 폴백 구현**: 두 번째 결제 프로세서 (예: Adyen)를 폴백으로 추가하는 것을 평가합니다. Stripe가 1분 이상 에러를 반환하면 자동으로 백업 프로세서로 트래픽을 라우팅합니다. | 완화 | @payments-team | 2024-05-01 | P2 |

**핵심 교훈:**
- 12분의 감지 갭 (15건의 고객 불만 후 알림 발생)이 가장 중요하게 수정해야 할 항목입니다 (조치 4).
- API 버전 고정 (조치 3)은 이 특정 인시던트를 완전히 예방했을 것입니다.
- 합성 테스트 (조치 5)는 실제 사용자 트래픽과 독립적으로 가장 빠른 감지를 제공합니다.

</details>

---

## 참고 자료(References)

- [Google SRE Book](https://sre.google/sre-book/table-of-contents/)
- [Google SRE Workbook](https://sre.google/workbook/table-of-contents/)
- [The Art of SLOs (Google)](https://sre.google/resources/practices-and-processes/art-of-slos/)
- [Implementing SLOs (O'Reilly)](https://www.oreilly.com/library/view/implementing-service-level/9781492076803/)
- [Incident Management Guide (PagerDuty)](https://response.pagerduty.com/)
- [Blameless Postmortems (Etsy)](https://www.etsy.com/codeascraft/blameless-postmortems/)
