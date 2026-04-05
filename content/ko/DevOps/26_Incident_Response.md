# 26. 인시던트 대응(Incident Response)

**이전**: [지속적 프로파일링](./25_Continuous_Profiling.md) | **다음**: [AIOps와 이상 탐지](./27_AIOps_Anomaly_Detection.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 명확한 역할과 에스컬레이션(escalation) 경로를 갖춘 구조화된 인시던트 관리 프로세스를 설계하고 운영할 수 있습니다
2. 팀 웰빙과 시스템 신뢰성의 균형을 맞추는 지속 가능한 온콜(on-call) 실무를 구현할 수 있습니다
3. 실행 가능한 개선을 산출하는 비난 없는 포스트모템(blameless postmortem)을 수행할 수 있습니다
4. 인시던트 완화를 가속화하는 효과적인 런북(runbook)을 작성할 수 있습니다
5. MTTR, MTTD, 인시던트 심각도 분류를 사용하여 인시던트 대응 효과를 측정할 수 있습니다
6. 장애를 학습 기회로 취급하는 인시던트 대응 문화를 구축할 수 있습니다

---

복잡한 시스템에서 인시던트는 불가피합니다. 문제는 인시던트가 발생할지 여부가 아니라, 발생했을 때 팀이 얼마나 효과적으로 대응하는가입니다. 잘 구조화된 인시던트 대응 프로세스는 폭발 반경(blast radius)을 줄이고, 해결 시간을 단축하며, 가장 중요하게는 재발을 방지하는 학습을 생성합니다.

> **비유 -- 병원 응급 대응**: 환자가 응급실에 도착하면 즉흥적인 혼란이 아닙니다. 구조화된 프로세스가 있습니다: 트리아지(triage, 심각도 평가), 지정된 역할의 외상 팀(주치의, 간호사, 마취과), 명확한 커뮤니케이션 프로토콜(SBAR), 사후 사례 검토(포스트모템)를 통한 향후 치료 개선. 소프트웨어 인시던트 대응도 동일한 원칙을 따릅니다.

## 1. 인시던트 심각도 분류(Severity Classification)

### 1.1 심각도 수준

| 수준 | 이름 | 기준 | 대응 | 예시 |
|------|------|------|------|------|
| **SEV1** | 긴급(Critical) | 전체 사용자에 영향을 미치는 매출 영향 장애 | 온콜 페이지, 인시던트 팀 소집, 임원 커뮤니케이션 | 결제 처리 완전 중단 |
| **SEV2** | 주요(Major) | 많은 사용자에 영향을 미치는 상당한 성능 저하 | 온콜 페이지, 인시던트 커맨더가 대응 주도 | API 요청의 50%가 500 반환 |
| **SEV3** | 경미(Minor) | 부분적 성능 저하, 우회 방법 있음 | 온콜에 알림, 업무 시간에 조사 | 검색 결과 느리지만 작동 |
| **SEV4** | 낮음(Low) | 외관상 또는 경미한 문제, 사용자 영향 없음 | 티켓 생성, 다음 스프린트에서 해결 | 대시보드 오래된 데이터 표시 |

### 1.2 심각도 결정 트리

```
데이터 손실 또는 보안 침해를 일으키고 있는가?
├── 예 → SEV1 (항상)
└── 아니오
    매출 또는 핵심 기능에 영향이 있는가?
    ├── 예
    │   영향받는 사용자 수는?
    │   ├── > 50% → SEV1
    │   ├── 10-50% → SEV2
    │   └── < 10% → SEV3
    └── 아니오
        우회 방법이 있는가?
        ├── 아니오 → SEV3
        └── 예 → SEV4
```

---

## 2. 인시던트 대응 프로세스

### 2.1 인시던트 생명주기(Lifecycle)

```
감지 → 트리아지 → 대응 → 완화 → 해결 → 포스트모템
   │          │         │           │             │            │
   │          │         │           │             │            └── 학습 및 개선
   │          │         │           │             └── 근본 원인 영구적으로 수정
   │          │         │           └── 출혈 중단 (임시 수정)
   │          │         └── 팀 소집, 수정 작업 중
   │          └── 심각도 평가, 역할 할당
   └── 알림 발동 또는 사용자 문제 보고
```

### 2.2 인시던트 역할(Roles)

| 역할 | 책임 | 담당자 |
|------|------|-------|
| **인시던트 커맨더 (IC)** | 대응 조율, 의사결정, 커뮤니케이션 관리 | 시니어 엔지니어 또는 온콜 리드 |
| **기술 리드(Tech Lead)** | 기술 조사 및 수정 주도 | 해당 분야 전문가 |
| **커뮤니케이션 리드** | 이해관계자, 고객, 상태 페이지 업데이트 | IC 또는 지정된 사람 |
| **스크라이브(Scribe)** | 타임라인, 조치, 발견 사항을 실시간 문서화 | 팀원 누구나 |

### 2.3 인시던트 커뮤니케이션 템플릿

```markdown
## 인시던트: [제목]
**심각도**: SEV2
**상태**: 조사 중 / 식별됨 / 모니터링 중 / 해결됨
**인시던트 커맨더**: @alice
**기술 리드**: @bob

### 타임라인 (UTC)
- 14:00 - 알림 발동: payment-service 오류율 > 5%
- 14:03 - IC 확인, 조사 시작
- 14:08 - 식별: Stripe API 타임아웃으로 연쇄 장애 발생
- 14:12 - 완화: 대체 결제 프로세서 활성화
- 14:15 - 오류율 정상으로 복귀
- 14:30 - 모니터링: 모든 메트릭 SLO 이내
- 15:00 - 해결: Stripe API 복구, 주 프로세서로 복원

### 영향
- 기간: 30분 (14:00 - 14:30)
- 영향받은 사용자: ~5,000건 결제 시도 실패
- 매출 영향: ~$50,000 지연 처리 (최종 모두 처리됨)

### 근본 원인
Stripe API가 내부 인프라 문제로 인해 높은 지연(>30초)을 경험.
30초 타임아웃으로 인해 진행 중인 모든 요청이 실패.

### 조치 항목
- [ ] P1: Stripe 타임아웃을 서킷 브레이커와 함께 5초로 감소 (연쇄 방지)
- [ ] P1: 백업 결제 프로세서로 자동 페일오버 활성화
- [ ] P2: SLO 대시보드에 Stripe API 지연 추가
```

---

## 3. 온콜 실무(On-Call Practices)

### 3.1 지속 가능한 온콜 설계

| 원칙 | 구현 |
|------|------|
| **최소 팀 규모: 8명** | 8주마다 1회 이상 온콜하지 않도록 보장 |
| **순환 기간: 1주** | 컨텍스트 구축에 충분하고, 번아웃 방지에 충분히 짧음 |
| **보상** | 추가 수당, 보상 휴일, 온콜 주간 스프린트 부하 감소 |
| **Primary + Secondary** | Secondary는 백업; Primary가 첫 대응 처리 |
| **Follow-the-sun** | 글로벌 팀: 시간대 간 온콜 인수인계 |
| **영웅 금지** | 한 사람이 주 3회 이상 페이지되면 시스템을 고쳐야 함, 사람이 아님 |

### 3.2 온콜 인수인계 프로세스

```
나가는 온콜 엔지니어:
  1. 인수인계 노트 작성:
     - 활성 인시던트 또는 진행 중인 문제
     - 문제를 일으킬 수 있는 최근 배포
     - 알려진 불안정 알림 (수정 티켓 참조와 함께)
     - 교대 중 관찰된 비정상적인 사항
  2. 모니터링이 녹색인지 확인 (또는 알려진 옐로우/레드 문서화)
  3. 들어오는 엔지니어에게 15분 동기화 브리핑

들어오는 온콜 엔지니어:
  1. 페이저 접근 확인 (자신에게 테스트 페이지)
  2. 열린 인시던트 티켓 검토
  3. 최근 배포 검토 (최근 48시간)
  4. 모든 중요 대시보드와 런북 접근 확인
  5. Slack/채널에서 인수인계 확인
```

### 3.3 알림 위생(Alert Hygiene)

| 문제 | 영향 | 해결 |
|------|------|------|
| **알림 피로(Alert fatigue)** | 엔지니어가 모든 알림 무시 | SLO 기반 알림으로 축소 (레슨 20) |
| **시끄러운 알림** | 실행 불가 이벤트에 페이지 | 모든 알림에 조치 요구; 그렇지 않으면 삭제 또는 조정 |
| **런북 누락** | 엔지니어가 20분간 무엇을 해야 하는지 파악 | 모든 알림은 반드시 런북에 연결 |
| **중복 알림** | 같은 인시던트가 5번 페이지 트리거 | Alertmanager에서 알림 그룹핑 및 억제 구성 |
| **SEV3의 비업무 시간 페이지** | 비긴급 문제로 수면 방해 | SEV3/4를 페이저가 아닌 티켓 큐로 라우팅 |

### 3.4 알림-런북 연결

```yaml
# 런북 링크가 있는 Prometheus 알림
- alert: PaymentServiceHighErrorRate
  expr: payment_service:error_ratio:rate5m > 0.01
  for: 3m
  labels:
    severity: critical
    team: payments
    runbook: "https://wiki.example.com/runbooks/payment-high-error-rate"
  annotations:
    summary: "결제 서비스 오류율 {{ $value | humanizePercentage }}"
    dashboard: "https://grafana.example.com/d/payment-slo"
    description: |
      오류율이 3분 이상 1%를 초과합니다.
      Stripe API 상태와 데이터베이스 연결을 확인하세요.
```

---

## 4. 런북(Runbooks)

### 4.1 런북 구조

```markdown
# 런북: 결제 서비스 높은 오류율

## 개요
이 런북은 결제 서비스의 높은 오류율 알림을 처리합니다.
**알림**: PaymentServiceHighErrorRate
**심각도**: 긴급 (온콜 페이지)
**서비스**: payment-service
**대시보드**: [결제 SLO 대시보드](https://grafana.example.com/d/payment-slo)

## 진단 단계

### 1단계: 오류 유형 식별
```promql
# 오류 분석 확인
sum by (status) (rate(http_requests_total{job="payment-service",status=~"5.."}[5m]))
```
- 대부분 502/504 → 업스트림 의존성 문제 (2단계)
- 대부분 500 → 내부 오류 (3단계)
- 대부분 503 → 서비스 과부하 (4단계)

### 2단계: 업스트림 의존성 확인
1. [Stripe 상태 페이지](https://status.stripe.com) 확인
2. 데이터베이스 연결 확인:
   ```bash
   kubectl exec -it deploy/payment-service -- pg_isready -h postgres
   ```
3. [의존성 대시보드](https://grafana.example.com/d/deps) 확인

### 3단계: 애플리케이션 상태 확인
```bash
kubectl logs -l app=payment-service --tail=100 | grep ERROR
kubectl top pods -l app=payment-service
```

### 4단계: 서비스 과부하 완화
```bash
# 스케일 업
kubectl scale deploy/payment-service --replicas=10

# 스케일 업으로 도움이 되지 않으면 속도 제한 활성화
kubectl set env deploy/payment-service RATE_LIMIT_RPS=100
```

## 완화 조치
- **업스트림 의존성 다운**: 대체 결제 프로세서 활성화
  ```bash
  kubectl set env deploy/payment-service PAYMENT_FALLBACK=true
  ```
- **데이터베이스 문제**: 비쓰기 작업을 읽기 레플리카로 페일오버
- **애플리케이션 크래시 루프**: 마지막 알려진 정상 버전으로 롤백
  ```bash
  kubectl rollout undo deploy/payment-service
  ```

## 에스컬레이션
- 해결 없이 15분 후: 결제 팀 리드 페이지
- 30분 후: 엔지니어링 디렉터 페이지
- 매출 영향 > $100K: VP Engineering 페이지
```

### 4.2 런북 테스트

```bash
# 정기적으로 런북 단계가 작동하는지 확인 (게임 데이 연습)
# 월간 런북 검토 일정:
# 1. 각 진단 단계 실행 -- 명령어가 작동하는가?
# 2. 대시보드 링크가 깨지지 않았는지 확인
# 3. 에스컬레이션 담당자가 최신인지 확인
# 4. 추가해야 할 새로운 장애 모드가 있는지 확인
```

---

## 5. 비난 없는 포스트모템(Blameless Postmortems)

### 5.1 포스트모템 원칙

| 원칙 | 구현 |
|------|------|
| **비난 없음(Blameless)** | 개인이 아닌 시스템과 프로세스에 집중 |
| **적시성(Timely)** | 기억이 생생한 48시간 이내 수행 |
| **철저함(Thorough)** | 타임라인, 근본 원인, 기여 요인, 조치 항목 포함 |
| **조치 지향(Action-oriented)** | 모든 포스트모템은 구체적이고, 할당되고, 추적되는 조치 항목 산출 |
| **공유(Shared)** | 학습을 위해 전체 엔지니어링 조직에 공개 |

### 5.2 포스트모템 템플릿

```markdown
# 포스트모템: 결제 처리 장애 (2025-03-10)

## 요약
Stripe API 깨어진 변경(breaking change)으로 인해 45분간 결제 처리 완전 불가
(09:15-10:00 UTC). 신용카드 결제 100% 실패;
직불카드 결제는 영향 없음.

## 영향
- **기간**: 45분
- **사용자 영향**: ~12,000건 결제 시도 실패
- **매출 영향**: ~$180,000 지연 처리
- **SLO 영향**: 오류 예산이 65%에서 12%로 소진

## 타임라인 (UTC)
| 시간 | 이벤트 |
|------|--------|
| 09:10 | Stripe가 `/v1/charges` 엔드포인트에 API 변경 배포 |
| 09:15 | 첫 번째 결제 실패 로깅 |
| 09:22 | 알림 발동: 결제 오류율 > 1% |
| 09:25 | 온콜 엔지니어 확인, 조사 시작 |
| 09:30 | IC 할당(@alice), SEV1 선언 |
| 09:35 | 근본 원인 식별: Stripe 응답 형식 변경 |
| 09:42 | 두 응답 형식 모두 처리하는 핫픽스 PR 생성 |
| 09:48 | 긴급 파이프라인을 통해 핫픽스 프로덕션 배포 |
| 09:52 | 오류율 감소, 복구 모니터링 |
| 10:00 | 오류율 기준선 복귀, 인시던트 해결 |

## 근본 원인
Stripe가 `/v1/charges` API 엔드포인트에 깨어진 변경을 배포했습니다.
응답 형식이 `{"id": "ch_xxx"}`에서
`{"charge": {"id": "ch_xxx"}}`로 변경되었습니다. Stripe 클라이언트 라이브러리가
이전 형식을 기대하고 응답을 파싱하여 모든 요청에 대해 역직렬화 오류를 발생시켰습니다.

## 기여 요인
1. 요청 헤더에 Stripe API 버전을 고정하지 않았음
2. Stripe 클라이언트가 파싱 전에 응답 스키마를 검증하지 않았음
3. 알림 임계값이 너무 높게(1%) 설정되어 09:15의 초기 0.5% 급증을 놓침
4. 실제 사용자 전에 장애를 감지하는 합성(synthetic) 결제 모니터링 없음

## 잘된 점
- 근본 원인 식별 후 13분 만에 핫픽스 배포
- 커뮤니케이션이 명확하고 적시 (09:32에 상태 페이지 업데이트)
- 인시던트 커맨더가 대응을 집중시킴

## 잘못된 점
- 7분 감지 갭 (09:15 ~ 09:22)
- 백업 결제 프로세서로의 자동 페일오버 없음
- Stripe 변경 로그 깨어진 변경 모니터링 안 함

## 조치 항목
| # | 조치 | 분류 | 담당자 | 우선순위 | 기한 |
|---|------|------|--------|---------|------|
| 1 | 헤더에 Stripe API 버전 고정 | 예방 | @bob | P1 | 2025-03-14 |
| 2 | 응답 스키마 검증 추가 | 예방 | @bob | P1 | 2025-03-17 |
| 3 | 알림 임계값을 0.3%로 낮추기 | 감지 | @alice | P0 | 2025-03-11 |
| 4 | 합성 결제 모니터링 추가 | 감지 | @carol | P1 | 2025-03-21 |
| 5 | 백업 프로세서 자동 페일오버 구현 | 완화 | @dave | P2 | 2025-04-01 |
| 6 | Stripe API 변경 로그 구독 | 예방 | @bob | P2 | 2025-03-14 |

## 교훈
1. 외부 API 의존성은 항상 특정 버전을 고정해야 합니다
2. 응답 검증은 깨어진 변경이 연쇄하기 전에 잡아냅니다
3. 합성 모니터링은 실제 사용자가 영향받기 전에 문제를 감지합니다
```

### 5.3 포스트모템 리뷰 회의

```
안건 (45-60분):
1. (5분)  IC가 요약과 타임라인을 소리내어 읽음
2. (10분) 타임라인 검토 -- 누구나 누락된 세부 사항 추가 가능
3. (15분) 근본 원인과 기여 요인 토론
4. (10분) "잘된 점"과 "잘못된 점" 검토
5. (10분) 조치 항목 검토 및 우선순위 지정
6. (5분)  담당자와 기한 할당

기본 규칙:
- 비난 없음: "시스템이 X를 허용했다" (O) "Y 씨가 X를 일으켰다" (X)
- 관련된 모든 사람 초대; 참석은 권장되지만 의무 아님
- 시스템 개선에 집중, 개인 성과 아님
- 조치 항목은 구체적이고, 할당되고, 이슈 트래커에서 추적되어야 함
```

---

## 6. 인시던트 메트릭(Incident Metrics)

### 6.1 핵심 인시던트 메트릭

| 메트릭 | 정의 | 목표 |
|--------|------|------|
| **MTTD** (평균 감지 시간) | 인시던트 시작부터 알림 발동까지 | < 5분 |
| **MTTA** (평균 확인 시간) | 알림부터 인간 확인까지 | < 5분 |
| **MTTR** (평균 해결 시간) | 감지부터 해결까지 | < 1시간 (SEV1) |
| **MTBF** (평균 장애 간격) | 같은 유형 인시던트 간 시간 | 증가 추세 |
| **인시던트 수** | 기간별 인시던트 수 | 감소 추세 |
| **조치 항목 완료율** | 기한 내 완료된 포스트모템 조치 항목 비율 | > 90% |

### 6.2 추적 및 보고

```python
"""인시던트 메트릭 대시보드 데이터."""
from dataclasses import dataclass
from datetime import timedelta

@dataclass
class IncidentMetrics:
    severity: str
    detected_at: str
    acknowledged_at: str
    resolved_at: str
    ttd_minutes: float
    tta_minutes: float
    ttr_minutes: float
    action_items: int
    action_items_completed: int

# 월간 인시던트 요약
monthly_incidents = [
    IncidentMetrics("SEV1", "2025-03-10 09:15", "2025-03-10 09:25", "2025-03-10 10:00",
                    ttd_minutes=7, tta_minutes=3, ttr_minutes=45, action_items=6, action_items_completed=5),
    IncidentMetrics("SEV2", "2025-03-18 14:30", "2025-03-18 14:33", "2025-03-18 15:00",
                    ttd_minutes=2, tta_minutes=3, ttr_minutes=30, action_items=3, action_items_completed=3),
]

avg_mttd = sum(i.ttd_minutes for i in monthly_incidents) / len(monthly_incidents)
avg_mttr = sum(i.ttr_minutes for i in monthly_incidents) / len(monthly_incidents)
action_completion = sum(i.action_items_completed for i in monthly_incidents) / sum(i.action_items for i in monthly_incidents)

print(f"평균 MTTD: {avg_mttd:.1f}분")
print(f"평균 MTTR: {avg_mttr:.1f}분")
print(f"조치 항목 완료율: {action_completion:.0%}")
```

---

## 7. 인시던트 대응 도구(Tooling)

### 7.1 도구 스택

| 분류 | 도구 | 목적 |
|------|------|------|
| **알림** | PagerDuty, Opsgenie, Grafana OnCall | 온콜 엔지니어에게 알림 라우팅 |
| **커뮤니케이션** | Slack (인시던트 채널), Zoom/Meet | 실시간 협업 |
| **상태 페이지** | Statuspage.io, Cachet, Instatus | 외부 고객 커뮤니케이션 |
| **문서화** | Confluence, Notion, Google Docs | 포스트모템 작성 및 저장 |
| **추적** | Jira, Linear, GitHub Issues | 조치 항목 추적 |
| **자동화** | Rundeck, PagerDuty Automation | 자동 진단 및 완화 조치 |

### 7.2 인시던트 채널 봇

```python
"""Slack 인시던트 봇: 인시던트 채널 생성 및 관리 자동화."""

def create_incident_channel(severity: str, title: str, commander: str) -> str:
    """표준 설정으로 Slack 인시던트 채널 생성."""
    channel_name = f"inc-{datetime.now():%Y%m%d}-{title.lower().replace(' ', '-')[:30]}"

    channel = slack.conversations_create(name=channel_name)

    # 인시던트 템플릿 게시
    slack.chat_postMessage(
        channel=channel["id"],
        text=f"""
:rotating_light: *인시던트 선언: {title}*
*심각도*: {severity}
*IC*: <@{commander}>
*상태*: 조사 중

*빠른 링크:*
- <{grafana_url}|Grafana 대시보드>
- <{runbook_url}|런북>
- <{statuspage_url}|상태 페이지 관리>

*필요한 역할:*
:white_check_mark: IC: <@{commander}>
:question: 기술 리드: (자원 또는 할당)
:question: 커뮤니케이션 리드: (자원 또는 할당)
:question: 스크라이브: (자원 또는 할당)

:eyes: 리액션으로 인시던트 대응에 참여하세요.
        """
    )

    # 채널 토픽 설정
    slack.conversations_setTopic(
        channel=channel["id"],
        topic=f"{severity} | {title} | IC: @{commander} | 상태: 조사 중"
    )

    return channel_name
```

---

## 8. 인시던트 대응 문화 구축

### 8.1 문화적 실천 사항

| 실천 사항 | 중요한 이유 |
|----------|-----------|
| **좋은 인시던트 대응 축하** | 원하는 행동 강화 |
| **포스트모템 폭넓게 공유** | 팀 간 학습이 가치를 배가 |
| **조치 항목 완료 추적** | 후속 조치 없는 포스트모템은 낭비 |
| **게임 데이 실시** | 실제 인시던트 전에 인시던트 대응 연습 |
| **비난이 아닌 학습 보상** | 비난 문화에서 사람들은 문제를 숨김 |
| **MTTR 측정 및 개선** | 측정되는 것이 개선됨 |

### 8.2 게임 데이(Game Days)

```
게임 데이 계획: 결제 서비스 장애 시뮬레이션
─────────────────────────────────────────────────
목표: 결제 서비스 의존성 장애를 감지, 대응, 완화하는
     팀의 능력 테스트.

설정 (게임 마스터가 수행, 온콜에 비공개):
  1. toxiproxy로 Stripe API 호출에 30초 지연 주입
  2. 화요일 오전 10시 시작 (금요일 아님!)

예상 시퀀스:
  10:00 - 주입 시작
  10:03 - 알림 발동 (결제 오류율 > 임계값)
  10:05 - 온콜 확인, 조사 시작
  10:10 - 근본 원인 식별 (Stripe 지연)
  10:15 - 완화 적용 (서킷 브레이커, 대체 프로세서)
  10:20 - 서비스 복구

평가 기준:
  - MTTD < 5분이었는가?
  - 역할이 신속하게 할당되었는가?
  - 런북이 따라졌는가?
  - 커뮤니케이션이 명확했는가?
  - 이해관계자에게 통지했는가?

사후 리뷰:
  - 팀을 놀라게 한 것은?
  - 런북에서 누락되거나 잘못된 단계는?
  - 없거나 사용하기 어려운 도구는?
  - 발견 사항에 따라 런북 및 알림 업데이트
```

---

## 9. 다음 단계

- [27_AIOps_Anomaly_Detection.md](./27_AIOps_Anomaly_Detection.md) -- ML 기반 이상 탐지 및 지능형 알림
- [28_Capstone_Full_Stack_Observability.md](./28_Capstone_Full_Stack_Observability.md) -- 종합 관측 가능성 플랫폼 설계

---

## 연습 문제

### 연습 1: 심각도 분류

각 시나리오의 심각도(SEV1-SEV4)를 분류하고 근거를 설명하세요:

1. 마케팅 홈페이지의 오타 ("Recieve" → "Receive")
2. 로그인 서비스가 요청의 30%에 503 반환
3. 데이터베이스 마이그레이션이 실수로 컬럼을 삭제하여 지난 2시간의 사용자 등록 데이터 손실
4. 내부 위키가 업무 시간 중 다운
5. 정기 로그 검토 중 애플리케이션 로그에 신용카드 번호 발견

<details>
<summary>정답 보기</summary>

**1. 마케팅 홈페이지 오타 → SEV4**
- 기능 영향 없음
- 외관상 문제로 쉬운 수정
- 사용자 워크플로우 영향 없음
- 정상 업무 시간에 수정

**2. 로그인 서비스 30% 503 → SEV2**
- 핵심 기능(인증)이 상당히 저하
- 30% 사용자 영향 (10-50% 사이)
- 우회 방법: 사용자가 재시도 가능 (정상 인스턴스에서 성공할 수 있음)
- 온콜 페이지, IC가 대응 주도

**3. 데이터베이스 마이그레이션 데이터 손실 → SEV1**
- 데이터 손실은 범위에 관계없이 항상 SEV1
- 2시간의 사용자 등록 손실 -- 애플리케이션에서 복구 불가
- 매출 영향 (손실된 등록 = 손실된 고객)
- 즉각 대응 필요: 출혈 중단, 복구 옵션 평가 (백업)

**4. 내부 위키 다운 → SEV3 (또는 SEV4)**
- 외부 사용자 영향 없음
- 업무 시간 중 내부 생산성 영향
- 우회 방법: 캐시된 페이지 사용, 동료에게 문의
- 업무 시간에 조사, 페이지 불필요

**5. 로그에 신용카드 번호 → SEV1**
- 보안 및 규정 준수 위반 (PCI-DSS 위반)
- 보안 인시던트는 현재 사용자 영향에 관계없이 항상 SEV1
- 즉각 조치 필요: PII 로깅 중단, 기존 로그 제거, 노출 평가
- 관할권에 따라 규제 기관 통지 필요 가능
- 정기 검토 중 발견(능동적 악용 아님)이더라도 노출 자체가 중대

</details>

### 연습 2: 포스트모템 작성

다음 사실이 있는 인시던트가 발생했습니다:
- 서비스: search-service
- 기간: 2시간 (06:00 - 08:00 UTC, 일요일)
- 영향: 검색이 오래된 결과(24시간 전) 반환했지만 오류는 없었음
- 근본 원인: Elasticsearch 리인덱싱 크론 잡이 Elasticsearch 클러스터 yellow 상태(미할당 레플리카 샤드)로 인해 묵묵히 실패
- 감지: 고객이 오래된 검색 결과에 대해 트윗; 지원팀이 에스컬레이션
- 인덱스 신선도 모니터링이 존재하지 않았음

최소 5개 조치 항목을 예방, 감지, 완화로 분류하여 포스트모템 조치 항목 섹션을 작성하세요.

<details>
<summary>정답 보기</summary>

| # | 조치 | 분류 | 담당자 | 우선순위 | 기한 |
|---|------|------|--------|---------|------|
| 1 | **검색 인덱스 신선도 모니터링 추가**: Prometheus 메트릭(`search_index_last_update_timestamp`)을 생성하고 인덱스가 1시간 이상 오래되면 알림. | 감지 | @search-team | P0 | 2025-03-18 |
| 2 | **Elasticsearch 클러스터 상태 알림 추가**: 클러스터 상태가 10분 이상 yellow 또는 red이면 알림. Yellow 상태(누락 레플리카)는 복원력을 저하시키므로 조사 필요. | 감지 | @platform-team | P1 | 2025-03-20 |
| 3 | **크론 잡 오류 처리 수정**: 리인덱싱 크론 잡이 현재 오류를 묵묵히 삼킴. 명시적 오류 처리 추가: (a) Elasticsearch 클러스터 상태와 함께 구조화된 오류 로깅, (b) `reindex_job_failure_total` 카운터 메트릭 방출, (c) #search-alerts에 Slack 알림 전송. | 예방 | @search-team | P0 | 2025-03-18 |
| 4 | **합성 검색 신선도 확인 추가**: (a) 소스 데이터베이스에 알려진 문서 작성, (b) 5분 대기, (c) Elasticsearch에서 검색, (d) 찾지 못하면 알림. 크론 실패뿐 아니라 모든 원인의 신선도 문제 포착. | 감지 | @search-team | P1 | 2025-03-25 |
| 5 | **미할당 레플리카 샤드 수정**: 레플리카 샤드가 미할당된 이유 조사 (클러스터를 떠난 노드일 가능성). Elasticsearch 클러스터 크기 조정 또는 노드 수정. Elasticsearch 샤드 할당 문제에 대한 런북 추가. | 예방 | @platform-team | P1 | 2025-03-20 |
| 6 | **사용자 대면 신선도 표시기 추가**: 검색 결과가 신선도 SLO(1시간)보다 오래되면 배너 표시: "검색 결과가 오래되었을 수 있습니다. 갱신 작업 중입니다." 팀이 문제를 수정하는 동안 고객 대면 혼란 감소. | 완화 | @frontend-team | P2 | 2025-04-01 |

**핵심 교훈:**
- 묵묵한 실패는 최악의 실패 유형 -- 크론 잡은 속삭이지 않고 소리쳐야 합니다.
- 신선도 모니터링은 데이터 시스템에서 가용성 모니터링만큼 중요합니다.
- 고객이 보고한 문제(트위터를 통해)는 감지가 완전히 실패했음을 의미 -- MTTD가 사실상 24시간이었습니다.

</details>

### 연습 3: 런북 설계

PostgreSQL 기반 서비스의 "데이터베이스 연결 풀 소진" 알림에 대한 런북을 작성하세요. 개요, 구체적 명령어가 포함된 진단 단계, 완화 조치, 에스컬레이션 경로, 예방 조치를 포함하세요.

<details>
<summary>정답 보기</summary>

```markdown
# 런북: 데이터베이스 연결 풀 소진

## 개요
**알림**: DatabaseConnectionPoolExhausted
**심각도**: 긴급 (온콜 페이지)
**서비스**: order-service
**데이터베이스**: PostgreSQL (orders-db)
**대시보드**: [데이터베이스 상태 대시보드](https://grafana.example.com/d/db-health)

이 알림은 데이터베이스 연결 풀 사용률이 2분 이상 90%를 초과할 때 발동합니다.
풀이 소진되면 새 요청이 큐에 대기하다가 결국 타임아웃되어 연쇄 503 오류를 발생시킵니다.

## 진단 단계

### 1단계: 연결 풀 상태 확인
```bash
# 현재 풀 메트릭 확인
kubectl exec -it deploy/order-service -- curl localhost:8080/metrics | grep db_pool
# db_pool_active_connections 48
# db_pool_max_connections 50
# db_pool_waiting_requests 15
```

### 2단계: 장시간 실행 쿼리 확인
```sql
-- PostgreSQL에 연결
kubectl exec -it statefulset/postgres-0 -- psql -U app -d orders

-- 30초 이상 실행 중인 활성 쿼리 찾기
SELECT pid, now() - pg_stat_activity.query_start AS duration,
       query, state, wait_event_type, wait_event
FROM pg_stat_activity
WHERE state != 'idle'
  AND (now() - pg_stat_activity.query_start) > interval '30 seconds'
ORDER BY duration DESC;

-- 상태별 연결 수 카운트
SELECT state, count(*) FROM pg_stat_activity GROUP BY state;
```

### 3단계: 연결 누수 확인
```bash
# 연결이 시간에 따라 증가하는지 확인 (누수 지표)
kubectl logs -l app=order-service --tail=200 | grep -i "connection\|pool\|leak"

# 애플리케이션 오류 로그 확인
kubectl logs -l app=order-service --tail=200 | grep ERROR
```

### 4단계: 데이터베이스 서버 상태 확인
```sql
-- PostgreSQL 최대 연결 vs 활성 확인
SELECT count(*) AS active, max_conn AS max
FROM pg_stat_activity,
     (SELECT setting::int AS max_conn FROM pg_settings WHERE name='max_connections') mc
GROUP BY max_conn;

-- 락 경합 확인
SELECT blocked.pid AS blocked_pid,
       blocked.query AS blocked_query,
       blocking.pid AS blocking_pid,
       blocking.query AS blocking_query
FROM pg_stat_activity blocked
JOIN pg_locks blocked_locks ON blocked.pid = blocked_locks.pid
JOIN pg_locks blocking_locks ON blocked_locks.locktype = blocking_locks.locktype
     AND blocked_locks.relation = blocking_locks.relation
     AND blocked_locks.pid != blocking_locks.pid
JOIN pg_stat_activity blocking ON blocking_locks.pid = blocking.pid
WHERE NOT blocked_locks.granted;
```

## 완화 조치

### 장시간 실행 쿼리 발견 시:
```sql
-- 장시간 실행 쿼리 종료 (주의하여 사용)
SELECT pg_terminate_backend(<pid>);
```

### 애플리케이션 연결 누수 시:
```bash
# 애플리케이션 파드 재시작 (누수된 연결 해제를 위한 롤링 재시작)
kubectl rollout restart deploy/order-service
```

### 갑작스러운 트래픽 급증 시:
```bash
# 애플리케이션 레플리카 스케일 업 (더 많은 풀에 연결 분산)
kubectl scale deploy/order-service --replicas=10

# 데이터베이스가 처리할 수 있다면 임시로 풀 크기 증가
kubectl set env deploy/order-service DB_POOL_MAX=100
```

### 데이터베이스 과부하 시:
```bash
# 연결 풀링 프록시(PgBouncer) 활성화
kubectl scale deploy/pgbouncer --replicas=3
kubectl set env deploy/order-service DB_HOST=pgbouncer
```

## 에스컬레이션
- 해결 없이 10분 → 데이터베이스 팀 리드 페이지
- 20분 → 엔지니어링 디렉터 페이지
- 데이터 손상 의심 시 → 즉시 VP Engineering 페이지

## 예방
- 애플리케이션과 PostgreSQL 사이에 PgBouncer를 연결 풀러로 배포
- 연결 풀 max-lifetime을 5분으로 설정 (오래된 연결 방지)
- 애플리케이션 헬스체크에 연결 누수 감지 추가
- PostgreSQL에 statement_timeout을 30초로 설정 (폭주 쿼리 종료)
- 쿼리 지속 시간 백분위수 모니터링 및 알림
```

</details>

---

## 참고 자료

- [PagerDuty Incident Response Guide](https://response.pagerduty.com/)
- [Google SRE Book -- Managing Incidents](https://sre.google/sre-book/managing-incidents/)
- [Etsy -- Blameless Postmortems](https://www.etsy.com/codeascraft/blameless-postmortems/)
- [Atlassian -- Incident Management Handbook](https://www.atlassian.com/incident-management/handbook)
- [PagerDuty -- On-Call Best Practices](https://www.pagerduty.com/resources/learn/on-call-best-practices/)
- [Jeli.io -- Incident Analysis](https://www.jeli.io/blog/category/incident-analysis/)
