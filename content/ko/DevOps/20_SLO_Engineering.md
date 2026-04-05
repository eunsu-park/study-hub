# 20. SLO 엔지니어링(SLO Engineering)

**이전**: [관측 가능성 엔지니어링](./19_Observability_Engineering.md) | **다음**: [신호 상관 관계](./21_Signal_Correlation.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있다:

1. SLI, SLO, SLA를 정의하고 그 관계를 설명한다
2. 다양한 서비스 유형에 적합한 SLI를 선택하고 SLO 준수를 계산한다
3. 오류 예산(error budget)을 신뢰성과 개발 속도 간의 균형을 위한 의사결정 프레임워크로 구현한다
4. 적절한 긴급도로 SLO 위반을 감지하는 번 레이트(burn rate) 알림을 설계한다
5. 이해관계자에게 신뢰성 상태를 전달하는 SLO 대시보드를 구축한다
6. 기능 출시 및 신뢰성 투자를 관리하는 오류 예산 정책을 운영한다

---

SLO 엔지니어링은 신뢰성을 모호한 열망("신뢰할 수 있기를 원한다")에서 측정 가능하고 실행 가능한 엔지니어링 분야로 전환한다. 서비스가 얼마나 신뢰할 수 있어야 하는지를 정확하게 정의함으로써, 엔지니어링, 제품, 비즈니스 이해관계자 간의 공통 언어를 만들고, 트레이드오프를 명시적으로 만드는 의사결정 프레임워크(오류 예산)를 만든다.

> **비유 -- 제조 공차(Manufacturing Tolerances)**: 기계 기술자는 "좋은 볼트"를 목표로 하지 않는다. 공차를 지정한다: "10mm 직경 +/- 0.05mm." 이 공차가 SLO이다. 측정 도구(캘리퍼)가 SLI이다. 볼트가 사양을 충족한다는 고객과의 계약이 SLA이다. 99.5%의 볼트가 공차 내에 있고 SLO가 99%라면, 더 빠른 가공을 실험할 예산이 있다. 98.5%로 떨어지면 실험을 중단하고 프로세스를 수정한다.

## 1. SLI, SLO, SLA

### 1.1 정의

```
SLA (Service Level Agreement)
  └── 결과가 따르는 비즈니스 계약 (환불, 페널티)
  └── "99.9% 가용성 또는 고객이 10% 크레딧 수령"

SLO (Service Level Objective)
  └── 내부 신뢰성 목표 (SLA보다 엄격)
  └── "30일 롤링 윈도우에서 99.95% 가용성"

SLI (Service Level Indicator)
  └── SLO를 측정하는 메트릭
  └── "300ms 이내에 성공적으로 완료된 요청의 비율"
```

**핵심 관계**: SLA >= SLO (SLO는 항상 SLA보다 엄격하여 버퍼를 제공한다).

### 1.2 SLI 사양(Specification) vs 구현(Implementation)

| 개념 | 정의 | 예시 |
|------|------|------|
| **SLI 사양** | 측정하고자 하는 것 (추상적) | "성공적으로 제공된 유효 요청의 비율" |
| **SLI 구현** | 측정하는 방법 (구체적) | `sum(rate(http_requests_total{status!~"5.."}[5m])) / sum(rate(http_requests_total[5m]))` |

### 1.3 일반적인 SLI 유형

| SLI 유형 | 정의 | 적합한 서비스 |
|---------|------|-------------|
| **가용성(Availability)** | 성공한 유효 요청의 비율 | 요청/응답 서비스 (API) |
| **지연 시간(Latency)** | 임계값보다 빠른 요청의 비율 | 사용자 대면 서비스 |
| **처리량(Throughput)** | 성공적인 처리 비율 | 데이터 처리 파이프라인 |
| **신선도(Freshness)** | 임계값 내에 업데이트된 데이터 비율 | 데이터베이스, 캐시, 검색 인덱스 |
| **정확성(Correctness)** | 올바른 데이터를 반환하는 응답 비율 | 데이터 파이프라인, ML 추론 |
| **내구성(Durability)** | 저장 후 복구 가능한 데이터 비율 | 스토리지 시스템 |

### 1.4 서비스 유형별 SLI 선택

| 서비스 유형 | 주요 SLI | 예시 |
|-----------|---------|------|
| **REST API** | 가용성, 지연 시간 (p50, p99) | 결제 서비스: 99.99% 가용성, p99 < 500ms |
| **스트리밍 파이프라인** | 처리량, 신선도 | Kafka 컨슈머: 99.9% 이벤트가 30초 내 처리 |
| **배치 처리** | 처리량, 정확성, 신선도 | ETL 작업: 99.5% 실행이 2시간 내 완료, < 0.01% 오류율 |
| **스토리지 시스템** | 가용성, 지연 시간, 내구성 | 데이터베이스: 99.99% 가용성, p99 읽기 < 10ms, 99.999999% 내구성 |
| **프론트엔드 (웹)** | 가용성, 지연 시간 (Core Web Vitals) | 페이지 로드의 75%에서 LCP < 2.5초 |

---

## 2. SLO 설계

### 2.1 SLO 문서

모든 서비스에는 작성된 SLO 문서가 있어야 한다:

```yaml
# slo-document.yaml
service: payment-service
owner: payments-team
last_review: 2025-01-15

slos:
  - name: availability
    description: "Proportion of non-5xx responses to valid requests"
    sli:
      type: availability
      specification: "Good events: status < 500. Total events: all HTTP requests excluding health checks."
      implementation:
        numerator: 'sum(rate(http_requests_total{job="payment-service",status!~"5.."}[5m]))'
        denominator: 'sum(rate(http_requests_total{job="payment-service"}[5m]))'
    objective: 99.95%
    window: 30d rolling
    consequences:
      budget_exhausted: "Freeze feature deployments until budget recovers"
      budget_below_25pct: "Cancel next sprint's feature work; focus on reliability"

  - name: latency
    description: "Proportion of requests completed within 300ms"
    sli:
      type: latency
      specification: "Good events: response time < 300ms. Total events: all HTTP requests."
      implementation:
        numerator: 'sum(rate(http_request_duration_seconds_bucket{job="payment-service",le="0.3"}[5m]))'
        denominator: 'sum(rate(http_request_duration_seconds_count{job="payment-service"}[5m]))'
    objective: 99.0%
    window: 30d rolling
    consequences:
      budget_exhausted: "Initiate performance review sprint"
```

### 2.2 적절한 목표 선택

| 목표 | 월간 다운타임 | 월간 오류 예산 | 적합한 서비스 |
|------|-------------|-------------|-------------|
| 99% | 7시간 18분 | 7시간 18분 | 내부 도구, 배치 작업 |
| 99.5% | 3시간 39분 | 3시간 39분 | 비핵심 서비스 |
| 99.9% | 43.8분 | 43.8분 | 표준 프로덕션 서비스 |
| 99.95% | 21.9분 | 21.9분 | 중요한 고객 대면 서비스 |
| 99.99% | 4.38분 | 4.38분 | 핵심 인프라 (인증, 결제) |
| 99.999% | 26.3초 | 26.3초 | DNS, 핵심 라우팅 (달성 비용이 매우 높음) |

**목표 선택 가이드라인:**

1. **사용자 기대에서 시작**: 사용자가 얼마나 많은 불안정성을 허용하는가?
2. **의존성 고려**: SLO는 가장 신뢰성이 낮은 의존성을 초과할 수 없다
3. **비용 감안**: 추가 나인(nine) 하나당 대략 10배의 엔지니어링 비용
4. **SLA 이전에 버퍼 남기기**: SLA가 99.9%이면 SLO를 99.95%로 설정
5. **낮게 시작하고 나중에 강화**: SLO를 강화하는 것이 완화하는 것보다 쉽다

### 2.3 SLO 윈도우

| 윈도우 유형 | 설명 | 장점 | 단점 |
|-----------|------|------|------|
| **롤링(Rolling)** (예: 30일) | 연속 슬라이딩 윈도우 | 부드러움, 절벽 효과 없음 | 인시던트 영향이 전체 윈도우 동안 지속 |
| **캘린더(Calendar)** (예: 월별) | 기간 경계에서 리셋 | 매월 새로 시작 | 월말 절벽 효과; 월초 vs 월말 인시던트가 다르게 취급됨 |

**모범 사례**: 운영 결정에는 롤링 윈도우를, 비즈니스 보고에는 캘린더 윈도우를 사용한다.

---

## 3. 오류 예산(Error Budgets)

### 3.1 오류 예산 개념

```
Error Budget = 1 - SLO

Example:
  SLO = 99.9% availability
  Error Budget = 0.1% of requests can fail

  If you serve 10,000,000 requests/month:
  Error Budget = 10,000 allowed failures/month

  Or in time:
  Error Budget = 30 days × 24h × 60m × 0.001 = 43.2 minutes of downtime/month
```

### 3.2 오류 예산 소비

```python
"""Error budget calculator for request-based SLIs."""
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class ErrorBudgetStatus:
    slo_target: float
    window_days: int
    total_requests: int
    failed_requests: int

    @property
    def current_sli(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return (self.total_requests - self.failed_requests) / self.total_requests

    @property
    def error_budget_total(self) -> int:
        """Total allowed failures in the window."""
        return int(self.total_requests * (1 - self.slo_target))

    @property
    def error_budget_remaining(self) -> int:
        """Remaining failures before SLO violation."""
        return max(0, self.error_budget_total - self.failed_requests)

    @property
    def error_budget_remaining_pct(self) -> float:
        """Percentage of error budget remaining."""
        if self.error_budget_total == 0:
            return 0.0
        return self.error_budget_remaining / self.error_budget_total * 100

    @property
    def is_slo_met(self) -> bool:
        return self.current_sli >= self.slo_target

# Example usage
status = ErrorBudgetStatus(
    slo_target=0.999,       # 99.9%
    window_days=30,
    total_requests=10_000_000,
    failed_requests=3_500,
)

print(f"SLO target:             {status.slo_target:.3%}")
print(f"Current SLI:            {status.current_sli:.4%}")
print(f"Error budget (total):   {status.error_budget_total:,} requests")
print(f"Error budget (used):    {status.failed_requests:,} requests")
print(f"Error budget remaining: {status.error_budget_remaining:,} requests ({status.error_budget_remaining_pct:.1f}%)")
print(f"SLO met:                {status.is_slo_met}")
```

출력:

```
SLO target:             99.900%
Current SLI:            99.965%
Error budget (total):   10,000 requests
Error budget (used):    3,500 requests
Error budget remaining: 6,500 requests (65.0%)
SLO met:                True
```

### 3.3 오류 예산 정책(Error Budget Policy)

오류 예산 정책은 다양한 예산 수준에서 무엇을 할지 정의한다:

| 잔여 예산 | 조치 |
|---------|------|
| **> 50%** | 정상 운영. 기능을 최대 속도로 출시. |
| **25% -- 50%** | 주의. 테스트 강도 강화. 금요일 위험한 배포 금지. |
| **5% -- 25%** | 감속. 신뢰성 작업이 우선. 비핵심 변경에 대한 기능 동결. |
| **0% -- 5%** | 긴급. 전체 기능 동결. 모든 엔지니어링 노력을 신뢰성에 집중. |
| **소진 (< 0%)** | SLO 위반. 모든 배포 중단. 인시던트 리뷰 수행. 포스트모템 발행. |

### 3.4 오류 예산과 배포 의사결정

```
Decision Framework:
────────────────────────────────────────────────
Feature deployment request arrives
    │
    ├── Check error budget remaining
    │   │
    │   ├── Budget > 50%? → APPROVE: Deploy normally
    │   │
    │   ├── Budget 25-50%? → APPROVE with conditions:
    │   │     - Canary deployment required
    │   │     - Automated rollback enabled
    │   │     - Not during peak traffic
    │   │
    │   ├── Budget 5-25%? → REVIEW required:
    │   │     - Risk assessment by SRE
    │   │     - Only reliability-improving changes approved
    │   │
    │   └── Budget < 5%? → DENY:
    │         - Only critical security patches
    │         - Reliability fixes only
    │
    └── Post-deployment:
        - Monitor SLI for 30 minutes
        - Auto-rollback if SLI degrades
```

---

## 4. 번 레이트 알림(Burn Rate Alerts)

### 4.1 임계값 알림이 SLO에 실패하는 이유

전통적인 임계값 알림("오류율 > 1%")은 SLO 윈도우를 고려하지 않는다:

- 1분간 1% 오류율은 거의 예산을 소비하지 않는다
- 1시간 동안 1% 오류율은 우려스럽다
- 1일 동안 1% 오류율은 심각하다

**번 레이트(burn rate)**는 윈도우에 대해 상대적으로 오류 예산을 얼마나 빨리 소비하고 있는지 측정한다.

### 4.2 번 레이트 정의

```
Burn Rate = (Observed error rate) / (SLO-allowed error rate)

Example:
  SLO = 99.9% (allowed error rate = 0.1%)
  Observed error rate = 0.5%

  Burn Rate = 0.5% / 0.1% = 5x

  At 5x burn rate, the 30-day error budget would be exhausted in 6 days.
  Time to exhaustion = Window / Burn Rate = 30 days / 5 = 6 days
```

### 4.3 멀티 윈도우, 멀티 번 레이트 알림

Google의 권장 접근법은 일치하는 룩백 윈도우가 있는 여러 번 레이트를 사용한다:

| 심각도 | 번 레이트 | 긴 윈도우 | 짧은 윈도우 | 소진까지 시간 |
|-------|---------|---------|---------|-----------|
| **페이지 (긴급, critical)** | 14.4x | 1시간 | 5분 | 2일 |
| **페이지 (경고, urgent)** | 6x | 6시간 | 30분 | 5일 |
| **티켓 (주의, warning)** | 3x | 1일 | 2시간 | 10일 |
| **티켓 (정보, info)** | 1x | 3일 | 6시간 | 30일 |

두 윈도우 모두 위반 상태여야 알림이 발생한다. 짧은 윈도우는 오래된 알림을 방지한다 (문제가 이미 해결되었을 수 있음).

### 4.4 Prometheus 구현

```yaml
# Burn rate alerting rules for payment-service availability SLO (99.9%)
groups:
  - name: payment_slo_burn_rate
    rules:
      # --- Recording rules for error ratios ---
      - record: payment_service:error_ratio:rate5m
        expr: |
          sum(rate(http_requests_total{job="payment-service",status=~"5.."}[5m]))
          / sum(rate(http_requests_total{job="payment-service"}[5m]))

      - record: payment_service:error_ratio:rate30m
        expr: |
          sum(rate(http_requests_total{job="payment-service",status=~"5.."}[30m]))
          / sum(rate(http_requests_total{job="payment-service"}[30m]))

      - record: payment_service:error_ratio:rate1h
        expr: |
          sum(rate(http_requests_total{job="payment-service",status=~"5.."}[1h]))
          / sum(rate(http_requests_total{job="payment-service"}[1h]))

      - record: payment_service:error_ratio:rate6h
        expr: |
          sum(rate(http_requests_total{job="payment-service",status=~"5.."}[6h]))
          / sum(rate(http_requests_total{job="payment-service"}[6h]))

      - record: payment_service:error_ratio:rate1d
        expr: |
          sum(rate(http_requests_total{job="payment-service",status=~"5.."}[1d]))
          / sum(rate(http_requests_total{job="payment-service"}[1d]))

      - record: payment_service:error_ratio:rate3d
        expr: |
          sum(rate(http_requests_total{job="payment-service",status=~"5.."}[3d]))
          / sum(rate(http_requests_total{job="payment-service"}[3d]))

      # --- Burn rate alerts ---
      # Critical: 14.4x burn rate (2-day exhaustion)
      - alert: PaymentSLOBurnRateCritical
        expr: |
          payment_service:error_ratio:rate1h > (14.4 * 0.001)
          and
          payment_service:error_ratio:rate5m > (14.4 * 0.001)
        for: 2m
        labels:
          severity: critical
          slo: payment-availability
        annotations:
          summary: "Payment service SLO burn rate critical (14.4x)"
          description: |
            Error budget will be exhausted in {{ printf "%.0f" (div 720.0 14.4) }} hours
            at current burn rate. 1h error ratio: {{ $value }}.

      # Urgent: 6x burn rate (5-day exhaustion)
      - alert: PaymentSLOBurnRateHigh
        expr: |
          payment_service:error_ratio:rate6h > (6 * 0.001)
          and
          payment_service:error_ratio:rate30m > (6 * 0.001)
        for: 5m
        labels:
          severity: warning
          slo: payment-availability
        annotations:
          summary: "Payment service SLO burn rate high (6x)"
          description: |
            Error budget will be exhausted in 5 days at current burn rate.
            6h error ratio: {{ $value }}.

      # Ticket: 3x burn rate (10-day exhaustion)
      - alert: PaymentSLOBurnRateElevated
        expr: |
          payment_service:error_ratio:rate1d > (3 * 0.001)
          and
          payment_service:error_ratio:rate2h > (3 * 0.001)
        for: 15m
        labels:
          severity: info
          slo: payment-availability
        annotations:
          summary: "Payment service SLO burn rate elevated (3x)"
```

### 4.5 지연 시간 SLO 번 레이트

지연 시간 SLO의 경우, "좋은 이벤트(good event)"는 임계값보다 빠른 요청이다:

```yaml
# Latency SLO: 99% of requests < 300ms
groups:
  - name: payment_latency_slo
    rules:
      - record: payment_service:latency_good_ratio:rate1h
        expr: |
          sum(rate(http_request_duration_seconds_bucket{
            job="payment-service", le="0.3"
          }[1h]))
          / sum(rate(http_request_duration_seconds_count{
            job="payment-service"
          }[1h]))

      - alert: PaymentLatencySLOBurnRateCritical
        expr: |
          (1 - payment_service:latency_good_ratio:rate1h) > (14.4 * 0.01)
          and
          (1 - payment_service:latency_good_ratio:rate5m) > (14.4 * 0.01)
        for: 2m
        labels:
          severity: critical
          slo: payment-latency
        annotations:
          summary: "Payment service latency SLO burn rate critical"
```

---

## 5. SLO 대시보드

### 5.1 대시보드 설계 원칙

SLO 대시보드는 10초 이내에 세 가지 질문에 답해야 한다:

1. **SLO를 충족하고 있는가?** (현재 SLI vs 목표)
2. **오류 예산이 얼마나 남았는가?** (예산 게이지)
3. **추세는 어떤가?** (시간에 따른 번 레이트)

### 5.2 대시보드 레이아웃

```
┌─────────────────────────────────────────────────────────┐
│ Payment Service SLO Dashboard                           │
├─────────────────┬───────────────────┬───────────────────┤
│ Availability    │ Latency (p99)     │ Error Budget      │
│ SLO: 99.95%     │ SLO: 99% < 300ms  │ ████████░░ 65%   │
│ Current: 99.97% │ Current: 99.3%    │ 6,500 / 10,000   │
│ Status: OK      │ Status: OK        │ remaining         │
├─────────────────┴───────────────────┴───────────────────┤
│ Error Budget Consumption (30-day rolling)               │
│ ▁▁▂▁▁▁▁▃▁▁▁▅▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁                       │
│ Day 1        Day 10 (incident)    Day 30               │
├─────────────────────────────────────────────────────────┤
│ Burn Rate (current)                                     │
│ 1h: 0.3x  │  6h: 0.5x  │  1d: 0.8x  │  3d: 0.7x     │
│ All within normal range                                 │
├─────────────────────────────────────────────────────────┤
│ Top Error Contributors (last 24h)                       │
│  1. POST /payments/charge: 0.15% error rate (timeout)   │
│  2. GET /payments/:id: 0.02% error rate (404s)          │
│  3. POST /payments/refund: 0.01% error rate             │
│     [View traces →]  [View logs →]                      │
└─────────────────────────────────────────────────────────┘
```

### 5.3 SLO 대시보드를 위한 Grafana PromQL 쿼리

```promql
# Current availability SLI (instant)
1 - (
  sum(rate(http_requests_total{job="payment-service",status=~"5.."}[30d]))
  / sum(rate(http_requests_total{job="payment-service"}[30d]))
)

# Error budget remaining (percentage)
1 - (
  (1 - (
    sum(increase(http_requests_total{job="payment-service",status!~"5.."}[30d]))
    / sum(increase(http_requests_total{job="payment-service"}[30d]))
  ))
  / (1 - 0.9995)  # 1 - SLO target
)

# Burn rate over time (for time-series panel)
(
  sum(rate(http_requests_total{job="payment-service",status=~"5.."}[1h]))
  / sum(rate(http_requests_total{job="payment-service"}[1h]))
) / 0.0005  # SLO budget rate (1 - 0.9995)
```

---

## 6. SLO 운영

### 6.1 SLO 리뷰 주기

| 리뷰 | 빈도 | 참여자 | 안건 |
|------|------|--------|------|
| **주간 SLO 체크** | 주간 | 온콜 엔지니어 | 번 레이트, 예산 상태 검토 |
| **월간 SLO 리뷰** | 월간 | 팀 리드 + SRE | 추세, 인시던트 영향 검토, 임계값 조정 |
| **분기별 SLO 감사** | 분기별 | 엔지니어링 리더십 | 전체 SLO 검토, 새 SLO 제안, 오래된 SLO 폐기 |

### 6.2 SLO 위반 처리

```
SLO Violation Detected
    │
    ├── 1. Immediate (within 1 hour)
    │   └── Page on-call engineer
    │   └── Determine if ongoing or past incident
    │   └── If ongoing: follow incident response (Lesson 26)
    │
    ├── 2. Short-term (within 24 hours)
    │   └── Conduct root cause analysis
    │   └── Apply error budget policy (feature freeze if needed)
    │   └── Communicate to stakeholders
    │
    ├── 3. Medium-term (within 1 week)
    │   └── Postmortem with action items
    │   └── Plan reliability improvements
    │   └── Review if SLO target is appropriate
    │
    └── 4. Long-term (next quarter)
        └── Track action item completion
        └── Review SLO targets during quarterly audit
        └── Consider architectural changes if violations recur
```

### 6.3 의존성 SLO

서비스가 다른 서비스에 의존하는 경우, 의존성 체인을 모델링한다:

```python
"""SLO dependency chain calculator."""

def composite_availability(service_slos: dict[str, float], topology: str) -> float:
    """Calculate composite availability based on dependency topology."""
    slos = list(service_slos.values())

    if topology == "serial":
        # All services must be available: multiply availabilities
        result = 1.0
        for slo in slos:
            result *= slo
        return result

    elif topology == "parallel_redundant":
        # Any one service sufficient: 1 - product of unavailabilities
        result = 1.0
        for slo in slos:
            result *= (1 - slo)
        return 1 - result

    else:
        raise ValueError(f"Unknown topology: {topology}")

# Serial chain: API → Payment → Database
serial = composite_availability({
    "api_gateway": 0.9999,
    "payment_service": 0.9995,
    "database": 0.9999,
}, topology="serial")
print(f"Serial availability: {serial:.4%}")
# 99.93% -- worse than any individual service

# Redundant: Primary DB + Replica DB
redundant = composite_availability({
    "primary_db": 0.999,
    "replica_db": 0.999,
}, topology="parallel_redundant")
print(f"Redundant availability: {redundant:.6%}")
# 99.9999% -- much better than either alone
```

---

## 7. 고급 SLO 패턴

### 7.1 멀티 SLO 서비스

실제 서비스는 다양한 측면을 다루는 여러 SLO가 필요하다:

```yaml
service: search-service
slos:
  - name: availability
    objective: 99.9%
    description: "Search queries return non-error responses"

  - name: latency-p50
    objective: 99%
    threshold: 100ms
    description: "Median search latency under 100ms"

  - name: latency-p99
    objective: 95%
    threshold: 1000ms
    description: "Tail search latency under 1 second"

  - name: freshness
    objective: 99%
    threshold: 60s
    description: "Search index updated within 60s of source change"

  - name: relevance
    objective: 95%
    description: "First result matches user intent (measured by click-through rate)"
```

### 7.2 사용자 여정 SLO(User-Journey SLOs)

서비스별 SLO 대신, 사용자에게 보이는 여정에 대한 SLO를 정의한다:

| 여정 | SLI | SLO |
|------|-----|-----|
| **결제(Checkout)** | 5초 이내에 성공적으로 완료된 결제 시도의 비율 | 99.9% |
| **검색(Search)** | 500ms 이내에 결과를 반환하는 검색의 비율 | 99.5% |
| **로그인(Login)** | 2초 이내에 확정적으로 성공 또는 실패하는 로그인 시도의 비율 | 99.99% |
| **파일 업로드(File upload)** | 30초 이내에 완료되는 100MB 미만 업로드의 비율 | 99.0% |

사용자 여정 SLO는 측정을 위해 합성 모니터링(synthetic monitoring) 또는 실제 사용자 모니터링(RUM)이 필요하다.

### 7.3 SLO as Code

버전 관리되는 설정에 SLO를 정의하고, 알림 및 대시보드 도구에서 사용한다:

```yaml
# sloth.yaml (Sloth -- SLO-to-Prometheus-rules generator)
version: "prometheus/v1"
service: "payment-service"
labels:
  owner: "payments-team"
  tier: "critical"
slos:
  - name: "requests-availability"
    objective: 99.95
    description: "Payment API availability"
    sli:
      events:
        error_query: sum(rate(http_requests_total{job="payment-service",status=~"5.."}[{{.window}}]))
        total_query: sum(rate(http_requests_total{job="payment-service"}[{{.window}}]))
    alerting:
      name: PaymentAvailability
      labels:
        category: availability
      annotations:
        runbook: "https://wiki.example.com/runbooks/payment-availability"
      page_alert:
        labels:
          severity: critical
      ticket_alert:
        labels:
          severity: warning
```

```bash
# Generate Prometheus rules from SLO definition
sloth generate -i sloth.yaml -o /etc/prometheus/rules/payment-slo.yml

# Output: recording rules + burn rate alert rules automatically generated
```

---

## 8. 조직적 도입

### 8.1 동의 얻기(Getting Buy-In)

| 이해관계자 | 메시지 |
|-----------|--------|
| **엔지니어링 리더십** | "SLO는 신뢰성 vs 기능에 대해 데이터 기반 의사결정을 할 수 있게 한다" |
| **프로덕트 매니저** | "오류 예산은 새 기능에 얼마나 많은 리스크를 감수할 수 있는지 정확히 알려준다" |
| **엔지니어** | "SLO는 예산 내의 장애에 대해 비난받는 것으로부터 보호한다" |
| **고객 지원** | "SLO 대시보드는 보고된 문제가 실제인지 격리된 것인지 즉시 알려준다" |

### 8.2 흔한 함정

| 함정 | 문제 | 해결책 |
|------|------|--------|
| **SLO가 너무 많음** | 관심 분산, 상충하는 목표 | 서비스당 1-3개 SLO로 시작 |
| **SLO가 너무 빡빡함** | 지속적인 위반, 팀이 SLO 무시 | 느슨하게 시작, 데이터 기반으로 강화 |
| **SLO가 너무 느슨함** | 위반 없음, 정보 제공 안 됨 | 예산이 가끔 소비될 때까지 강화 |
| **오류 예산 정책 없음** | SLO는 측정되지만 결과가 없음 | SLO 설정 전에 정책 정의 및 시행 |
| **서버 측만 측정** | SLO는 99.9%이지만 사용자 경험은 95% | 에지에서 측정하거나 합성 모니터링 사용 |

---

## 9. 다음 단계

- [21_Signal_Correlation.md](./21_Signal_Correlation.md) -- 빠른 디버깅을 위한 메트릭, 로그, 트레이스 상관 관계
- [22_Advanced_Metrics_Architecture.md](./22_Advanced_Metrics_Architecture.md) -- 페더레이션과 장기 저장소로 메트릭 인프라 확장

---

## 연습 문제

### 연습 문제 1: SLI 선택

아래 각 서비스에 대해 2-3개의 SLI를 선택하고 선택의 이유를 설명하라. SLI 사양(무엇을 측정할 것인지)과 구현(Prometheus/OTel에서 어떻게 측정할 것인지) 모두 지정하라.

1. 이미지를 여러 크기로 리사이즈하는 이미지 업로드 및 처리 서비스
2. 실시간 채팅 메시징 서비스
3. 재무 보고서를 생성하는 야간 배치 작업

<details>
<summary>정답 보기</summary>

**1. 이미지 업로드 및 처리 서비스:**

| SLI | 사양 | 구현 | 근거 |
|-----|------|------|------|
| 가용성 | 오류가 아닌 응답을 반환하는 업로드 요청의 비율 | `sum(rate(http_requests_total{job="image-service",status!~"5.."}[5m])) / sum(rate(http_requests_total{job="image-service"}[5m]))` | 사용자는 업로드가 성공해야 한다 |
| 지연 시간 | 10초 이내에 반환되는 업로드의 비율 (5MB 미만 이미지) | `sum(rate(http_request_duration_seconds_bucket{job="image-service",le="10"}[5m])) / sum(rate(http_request_duration_seconds_count{job="image-service"}[5m]))` | 업로드 지연 시간이 사용자 경험에 직접 영향 |
| 신선도 | 업로드 후 60초 이내에 모든 리사이즈 변형이 사용 가능한 이미지의 비율 | 커스텀 메트릭: `sum(rate(image_processing_completed_within_slo_total[5m])) / sum(rate(image_uploads_total[5m]))` | 사용자는 리사이즈된 이미지를 빠르게 기대한다 |

**2. 실시간 채팅 메시징 서비스:**

| SLI | 사양 | 구현 | 근거 |
|-----|------|------|------|
| 가용성 | 성공하는 메시지 전송 요청의 비율 | `sum(rate(chat_messages_sent_total{status="success"}[5m])) / sum(rate(chat_messages_sent_total[5m]))` | 메시지 전달이 핵심 기능 |
| 지연 시간 | 500ms 이내에 수신자에게 전달되는 메시지의 비율 | 커스텀 메트릭: `sum(rate(chat_message_delivery_duration_seconds_bucket{le="0.5"}[5m])) / sum(rate(chat_message_delivery_duration_seconds_count[5m]))` | 실시간 채팅은 낮은 지연 시간이 필요 |
| 신선도 | 5초 미만의 오래된 데이터를 반환하는 메시지 히스토리 요청의 비율 | `sum(rate(chat_history_freshness_within_slo_total[5m])) / sum(rate(chat_history_requests_total[5m]))` | 사용자는 메시지 히스토리가 최신이기를 기대 |

**3. 야간 배치 작업 (재무 보고서):**

| SLI | 사양 | 구현 | 근거 |
|-----|------|------|------|
| 신선도 | 오전 6시 마감 전에 사용 가능한 보고서의 비율 | `sum(report_generation_completed_before_deadline_total) / sum(report_generation_attempts_total)` | 비즈니스 사용자가 시장 개장 전 보고서 필요 |
| 정확성 | 반올림 허용 범위 내에서 소스 데이터와 일치하는 보고서 행의 비율 | 커스텀 검증: `sum(report_rows_correct_total) / sum(report_rows_total)` | 재무 데이터는 정확해야 한다 |
| 처리량 | 성공적으로 완료되는 예약된 보고서의 비율 | `sum(reports_completed_successfully_total) / sum(reports_scheduled_total)` | 모든 보고서가 생성되어야 한다 |

</details>

### 연습 문제 2: 오류 예산 계산

30일 롤링 윈도우에서 99.9% 가용성 SLO를 가진 서비스가 있다. 지난 30일:
- 총 요청: 50,000,000건
- 5xx 응답: 42,000건
- 15일째 인시던트로 2시간 동안 30,000건의 오류 발생

계산하라: (a) 현재 SLI, (b) 총 오류 예산, (c) 소비된 예산, (d) 잔여 예산 비율, (e) 인시던트 중 번 레이트, (f) 오류 예산 정책이 트리거해야 할 조치.

<details>
<summary>정답 보기</summary>

**(a) 현재 SLI:**
```
SLI = (50,000,000 - 42,000) / 50,000,000 = 49,958,000 / 50,000,000 = 99.916%
```

**(b) 총 오류 예산:**
```
Budget = 50,000,000 × (1 - 0.999) = 50,000,000 × 0.001 = 50,000 errors
```

**(c) 소비된 예산:**
```
Consumed = 42,000 / 50,000 = 84%
```

**(d) 잔여 예산:**
```
Remaining = (50,000 - 42,000) / 50,000 = 8,000 / 50,000 = 16%
```

**(e) 인시던트 중 번 레이트:**
```
Normal error rate = (42,000 - 30,000) / 50,000,000 = 0.024% (background errors)
Incident request rate = 50,000,000 / 30 / 24 = ~69,444 requests/hour
Incident 2-hour requests ≈ 138,889
Incident error rate = 30,000 / 138,889 = 21.6%

Burn rate = 21.6% / 0.1% = 216x

At 216x burn rate, a 14.4x page alert would fire immediately.
The 30-day budget would be exhausted in 30/216 = 3.3 hours.
```

**(f) 오류 예산 정책 조치:**
잔여 예산 16% (5-25% 범위)이므로:
- 비핵심 변경에 대한 기능 동결
- 신뢰성 작업이 우선
- 15일째 인시던트에 대한 포스트모템 수행
- 아키텍처를 감안했을 때 99.9% 목표가 적절한지 검토
- 팀은 기능 개발 속도를 재개하기 전에 신뢰성 개선을 입증해야 한다
- 다음 스프린트는 유사한 인시던트 방지에 초점을 맞춘 "신뢰성 스프린트"여야 한다

</details>

### 연습 문제 3: 번 레이트 알림 설계

30일 윈도우에서 99.5% 지연 시간 SLO(요청의 99.5%가 500ms 이내에 완료되어야 함)를 가진 서비스의 완전한 번 레이트 알림 세트를 설계하라. Prometheus 레코딩 규칙과 알림 규칙을 작성하라. critical (페이지), warning (티켓), informational 심각도 수준을 포함한다.

<details>
<summary>정답 보기</summary>

```yaml
groups:
  - name: service_latency_slo_recording
    rules:
      # Good events: requests completing within 500ms
      - record: service:latency_slo_error_ratio:rate5m
        expr: |
          1 - (
            sum(rate(http_request_duration_seconds_bucket{job="my-service",le="0.5"}[5m]))
            / sum(rate(http_request_duration_seconds_count{job="my-service"}[5m]))
          )

      - record: service:latency_slo_error_ratio:rate30m
        expr: |
          1 - (
            sum(rate(http_request_duration_seconds_bucket{job="my-service",le="0.5"}[30m]))
            / sum(rate(http_request_duration_seconds_count{job="my-service"}[30m]))
          )

      - record: service:latency_slo_error_ratio:rate1h
        expr: |
          1 - (
            sum(rate(http_request_duration_seconds_bucket{job="my-service",le="0.5"}[1h]))
            / sum(rate(http_request_duration_seconds_count{job="my-service"}[1h]))
          )

      - record: service:latency_slo_error_ratio:rate6h
        expr: |
          1 - (
            sum(rate(http_request_duration_seconds_bucket{job="my-service",le="0.5"}[6h]))
            / sum(rate(http_request_duration_seconds_count{job="my-service"}[6h]))
          )

      - record: service:latency_slo_error_ratio:rate1d
        expr: |
          1 - (
            sum(rate(http_request_duration_seconds_bucket{job="my-service",le="0.5"}[1d]))
            / sum(rate(http_request_duration_seconds_count{job="my-service"}[1d]))
          )

      - record: service:latency_slo_error_ratio:rate2h
        expr: |
          1 - (
            sum(rate(http_request_duration_seconds_bucket{job="my-service",le="0.5"}[2h]))
            / sum(rate(http_request_duration_seconds_count{job="my-service"}[2h]))
          )

  - name: service_latency_slo_alerts
    rules:
      # SLO budget rate = 1 - 0.995 = 0.005

      # Critical page: 14.4x burn rate, 1h + 5m windows
      # Threshold: 14.4 * 0.005 = 0.072
      - alert: ServiceLatencySLOCritical
        expr: |
          service:latency_slo_error_ratio:rate1h > 0.072
          and
          service:latency_slo_error_ratio:rate5m > 0.072
        for: 2m
        labels:
          severity: critical
          slo: service-latency
        annotations:
          summary: "Service latency SLO critical burn rate (14.4x)"
          description: |
            Budget will exhaust in ~2 days. Current 1h slow-request ratio: {{ $value }}.
            SLO: 99.5% of requests < 500ms.
          runbook: "https://wiki.example.com/runbooks/latency-slo"

      # Warning ticket: 6x burn rate, 6h + 30m windows
      # Threshold: 6 * 0.005 = 0.030
      - alert: ServiceLatencySLOWarning
        expr: |
          service:latency_slo_error_ratio:rate6h > 0.030
          and
          service:latency_slo_error_ratio:rate30m > 0.030
        for: 5m
        labels:
          severity: warning
          slo: service-latency
        annotations:
          summary: "Service latency SLO elevated burn rate (6x)"
          description: "Budget will exhaust in ~5 days at current rate."

      # Info ticket: 3x burn rate, 1d + 2h windows
      # Threshold: 3 * 0.005 = 0.015
      - alert: ServiceLatencySLOElevated
        expr: |
          service:latency_slo_error_ratio:rate1d > 0.015
          and
          service:latency_slo_error_ratio:rate2h > 0.015
        for: 15m
        labels:
          severity: info
          slo: service-latency
        annotations:
          summary: "Service latency SLO slightly elevated burn rate (3x)"
          description: "Budget will exhaust in ~10 days if trend continues."
```

**핵심 설계 포인트:**
- 각 알림은 두 개의 윈도우를 사용한다: 긴 윈도우는 지속적인 문제를 감지하고, 짧은 윈도우는 오래된 알림을 방지한다.
- 임계값 계산: burn_rate * (1 - SLO target) = burn_rate * 0.005.
- `for` 지속 시간은 낮은 심각도에서 증가하여 노이즈를 줄인다.
- Critical 알림은 즉시 페이지하고, warning과 info는 티켓을 생성한다.

</details>

---

## 참고 자료

- [The Art of SLOs (Google Cloud)](https://sre.google/resources/practices-and-processes/art-of-slos/)
- [Google SRE Workbook -- Implementing SLOs](https://sre.google/workbook/implementing-slos/)
- [Implementing Service Level Objectives (O'Reilly)](https://www.oreilly.com/library/view/implementing-service-level/9781492076803/)
- [Sloth -- SLO as Code](https://github.com/slok/sloth)
- [OpenSLO Specification](https://openslo.com/)
- [Google SRE Book -- Service Level Objectives](https://sre.google/sre-book/service-level-objectives/)
