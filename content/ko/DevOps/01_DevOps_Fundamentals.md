# 레슨 1: DevOps 기초

**다음**: [버전 관리 워크플로우](./02_Version_Control_Workflows.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. DevOps를 정의하고, 개발 팀과 운영 팀 간의 마찰에 대한 해결책으로 왜 등장했는지 설명할 수 있다
2. CALMS 프레임워크(Culture, Automation, Lean, Measurement, Sharing)를 설명하고 조직의 DevOps 성숙도를 평가하는 데 적용할 수 있다
3. DevOps와 전통적인 운영 모델 및 워터폴 방식의 전달 방식을 비교할 수 있다
4. 계획 단계에서 모니터링까지 DevOps 라이프사이클 단계를 매핑할 수 있다
5. 네 가지 DORA 메트릭과 엔지니어링 성과에 왜 중요한지 설명할 수 있다
6. DevOps 도입을 저해하는 일반적인 안티패턴을 식별할 수 있다

---

DevOps는 도구도, 직책도, 팀 이름도 아닙니다. DevOps는 소프트웨어 개발(Dev)과 IT 운영(Ops)을 통합하는 일련의 실천 방법, 문화적 철학, 조직적 패턴입니다. 그 목표는 시스템 개발 라이프사이클을 단축하면서 기능, 수정, 업데이트를 빈번하고 안정적으로 전달하는 것입니다. DevOps 기초를 이해하는 것은 Terraform, Ansible, GitHub Actions와 같은 구체적인 도구를 배우기 전에 필수적입니다. 도구는 그 기반이 되는 문화와 프로세스가 올바를 때만 가치를 발휘하기 때문입니다.

> **비유 -- 레스토랑 주방:** 전통적인 소프트웨어 전달은 셰프(Dev)가 요리를 준비하고 창구를 통해 넘긴 뒤, 서빙하는 웨이터(Ops)와 전혀 대화하지 않는 레스토랑과 같습니다. 고객이 불만을 제기하면 주방과 홀 직원 사이에서 책임 떠넘기기가 발생합니다. DevOps는 그 벽을 허뭅니다. 레시피 설계부터 테이블 피드백까지, 모든 사람이 전체 다이닝 경험에 대한 책임을 공유합니다.

## 1. DevOps란?

DevOps는 소프트웨어 개발과 IT 운영을 결합하여 협업을 개선하고, 반복적인 작업을 자동화하며, 더 빠르고 높은 품질로 소프트웨어를 전달하는 **문화적이고 기술적인 운동**입니다.

### DevOps가 해결하는 문제

```
Traditional Model:
┌──────────────┐          ┌──────────────┐
│  Development │  "Throw  │  Operations  │
│    Team      │──over──▶ │    Team      │
│              │  the     │              │
│ "Build it    │  wall"   │ "Run it      │
│  fast!"      │          │  stable!"    │
└──────────────┘          └──────────────┘
     Speed ◀──── Conflict ────▶ Stability
```

**벽의 결과:**
- 배포가 월간 또는 분기별로 발생하여 리스크가 누적됩니다
- 운영 팀은 프로덕션에서만 버그를 발견합니다
- 비난 문화: "개발이 코드를 잘못 짰다" vs "운영이 제대로 실행을 못한다"
- 팀 간 공유 맥락 부재로 인한 느린 인시던트 대응
- 수동적이고 오류가 발생하기 쉬운 릴리스 프로세스

### DevOps 해결책

```
DevOps Model:
┌────────────────────────────────────────┐
│         Shared Responsibility          │
│                                        │
│  Dev + Ops + QA + Security             │
│                                        │
│  "We build it, we run it,             │
│   we own it together."                │
│                                        │
│  Speed ◀──── Alignment ────▶ Stability │
└────────────────────────────────────────┘
```

---

## 2. DevOps vs 전통적 운영

| 측면 | 전통적 운영 | DevOps |
|------|------------|--------|
| **팀 구조** | 분리된 Dev와 Ops | 교차 기능 팀 |
| **배포 빈도** | 월간/분기별 | 하루에 여러 번 |
| **릴리스 프로세스** | 수동, 변경 위원회 승인 | 자동화된 CI/CD 파이프라인 |
| **인프라** | 수동으로 구성된 서버 | Infrastructure as Code |
| **모니터링** | 운영팀 전용 대시보드 | 공유 관측 가능성 |
| **장애 대응** | 비난과 사후 분석 | 비난 없는 회고 |
| **피드백 루프** | 수 주에서 수 개월 | 수 분에서 수 시간 |
| **테스팅** | 사이클 끝의 QA 게이트 | 지속적인 자동화 테스팅 |

### Waterfall vs Agile vs DevOps

```
Waterfall:    Requirements ──▶ Design ──▶ Code ──▶ Test ──▶ Deploy ──▶ Maintain
              (단계 간 수 개월, 피드백이 너무 늦게 도착)

Agile:        [Sprint 1] ──▶ [Sprint 2] ──▶ [Sprint 3] ──▶ ...
              (반복적 개발, 하지만 Ops는 여전히 분리)

DevOps:       Plan ──▶ Code ──▶ Build ──▶ Test ──▶ Release ──▶ Deploy ──▶ Operate ──▶ Monitor
                 ▲                                                                      │
                 └──────────────────── Continuous Feedback ─────────────────────────────┘
              (완전히 통합된 라이프사이클, 모든 것이 지속적)
```

---

## 3. CALMS 프레임워크

CALMS는 DevOps 성숙도를 평가하기 위해 널리 채택된 프레임워크로, Jez Humble(*Continuous Delivery* 공저자)이 소개했습니다.

### Culture (문화)

문화가 기초입니다. 문화적 변화 없이 도구는 비싼 장식에 불과합니다.

**핵심 원칙:**
- **공유된 책임감**: "직접 만들었으면 직접 운영한다" (Werner Vogels, Amazon CTO)
- **비난 없는 사후 분석**: 개인의 잘못이 아니라 시스템적 원인에 집중
- **심리적 안전감**: 팀원이 오류를 보고하고 실험하는 것에 안전함을 느낌
- **교차 기능적 협업**: 개발 팀에 내장된 운영 엔지니어 또는 개발 팀을 지원하는 플랫폼 팀

```
Blameless Post-Mortem Template:
─────────────────────────────
1. Timeline of events
2. What went wrong (technical root causes)
3. What went right (things that mitigated impact)
4. Action items with owners and deadlines
5. What we learned

NOT included:
✗ Who caused the incident
✗ Punishment or blame assignment
```

### Automation (자동화)

반복적이고 오류가 발생하기 쉽거나 시간이 많이 걸리는 모든 것을 자동화합니다.

**자동화 대상:**
- 빌드 및 컴파일 (CI)
- 테스팅 (단위, 통합, 엔드투엔드)
- 인프라 프로비저닝 (Terraform, CloudFormation)
- 구성 관리 (Ansible, Puppet, Chef)
- 배포 (CD 파이프라인)
- 모니터링 및 알림
- 인시던트 대응 런북

```bash
# Manual deployment (error-prone):
ssh production-server
cd /var/www/app
git pull origin main
pip install -r requirements.txt
systemctl restart app
# Hope nothing breaks...

# Automated deployment (reliable):
# Push to main -> CI runs tests -> CD deploys automatically
git push origin main
# Pipeline handles build, test, deploy, verify, rollback-if-needed
```

### Lean (린)

린 제조업에서 차용하여 소프트웨어 전달에 린 사고를 적용합니다:

- **낭비 제거**: 핸드오프, 대기, 불필요한 승인 제거
- **가치 스트림 매핑**: 코드 커밋에서 프로덕션까지의 모든 단계를 매핑하여 병목 지점을 찾음
- **작은 배치 크기**: 대규모 릴리스를 드물게 하는 대신 작은 변경을 빈번하게 배포
- **진행 중 작업(WIP) 제한**: 동시 작업을 제한하여 흐름을 개선

```
Value Stream Map Example:
─────────────────────────
Code Commit ──[5 min]──▶ Code Review ──[2 hrs wait]──▶ Merge
    ──[10 min]──▶ CI Build ──[3 days wait]──▶ QA Approval
    ──[1 day wait]──▶ Change Board ──[2 hrs]──▶ Deploy

Total lead time: ~4.5 days
Active work time: ~2.5 hours
Wait time: ~4.3 days (96% waste!)
```

### Measurement (측정)

측정하지 않으면 개선할 수 없습니다.

**주요 범주:**
- **전달 성과**: 얼마나 빠르고 안정적으로 전달하는가?
- **운영 건전성**: 프로덕션 환경이 얼마나 안정적인가?
- **품질**: 얼마나 많은 결함이 프로덕션에 도달하는가?
- **비즈니스 영향**: 더 빠른 전달이 비즈니스 가치를 창출하는가?

### Sharing (공유)

지식 사일로를 허뭅니다:

- **내부 기술 발표 및 데모**
- **공유 대시보드 및 런북**
- **코드로서의 문서화** (버전 관리, 코드처럼 리뷰)
- **ChatOps**: 알림, 배포, 쿼리를 공유 채팅 채널에 통합
- **실천 커뮤니티**: 특정 기술을 중심으로 한 교차 팀 그룹

---

## 4. DevOps 라이프사이클

DevOps 라이프사이클은 개발과 운영 활동을 통합하는 무한 루프(종종 8자 또는 무한대 기호로 그려짐)입니다.

```
            ┌─────────────────────────────────────────┐
            │              DevOps Lifecycle             │
            │                                           │
            │    Plan ──▶ Code ──▶ Build ──▶ Test       │
            │      ▲                           │        │
            │      │         ∞ Loop            ▼        │
            │   Monitor ◀── Operate ◀── Deploy ◀── Release│
            │                                           │
            └─────────────────────────────────────────┘
```

### 단계별 상세

| 단계 | 활동 | 도구 (예시) |
|------|------|-------------|
| **Plan** | 요구사항, 사용자 스토리, 스프린트 계획 | Jira, Linear, GitHub Issues |
| **Code** | 코드 작성, 피어 리뷰, 브랜치 관리 | Git, GitHub, VS Code |
| **Build** | 컴파일, 패키징, 아티팩트 생성 | Maven, npm, Docker build |
| **Test** | 단위, 통합, 보안, 성능 테스트 | pytest, Jest, Selenium, OWASP ZAP |
| **Release** | 버전 태깅, 릴리스 노트, 승인 게이트 | GitHub Releases, 시맨틱 버저닝 |
| **Deploy** | 스테이징/프로덕션 환경에 푸시 | Kubernetes, Terraform, Ansible |
| **Operate** | 인프라 관리, 스케일링, 인시던트 대응 | PagerDuty, Kubernetes, AWS |
| **Monitor** | 메트릭, 로그, 트레이스, 알림 | Prometheus, Grafana, ELK, Datadog |

---

## 5. DORA 메트릭

**DevOps Research and Assessment (DORA)** 팀(현재 Google Cloud 소속)은 소프트웨어 전달 성과를 예측하는 네 가지 핵심 메트릭을 식별했습니다. 이 메트릭은 수천 개 조직을 대상으로 한 수년간의 연구에 기반합니다.

### 네 가지 핵심 메트릭

#### 1. 배포 빈도(Deployment Frequency)

조직이 프로덕션에 코드를 얼마나 자주 배포합니까?

```
Elite:   On-demand (multiple deploys per day)
High:    Between once per day and once per week
Medium:  Between once per week and once per month
Low:     Between once per month and once every six months
```

#### 2. 변경 리드 타임(Lead Time for Changes)

코드 커밋부터 프로덕션에서 실행되기까지 얼마나 걸립니까?

```
Elite:   Less than one hour
High:    Between one day and one week
Medium:  Between one week and one month
Low:     Between one month and six months
```

#### 3. 평균 복구 시간(Mean Time to Recovery, MTTR)

서비스 인시던트 발생 시 서비스를 복원하는 데 얼마나 걸립니까?

```
Elite:   Less than one hour
High:    Less than one day
Medium:  Between one day and one week
Low:     More than one week
```

#### 4. 변경 실패율(Change Failure Rate)

배포 중 프로덕션에서 장애를 유발하는 비율은 얼마입니까?

```
Elite:   0-15%
High:    16-30%
Medium:  16-30%
Low:     46-60%
```

### DORA 성과 프로필

```
┌──────────────────────────────────────────────────────────────┐
│                    DORA Performance Levels                     │
│                                                               │
│  Metric              Elite         High          Low          │
│  ─────────────────   ───────────   ──────────   ──────────   │
│  Deploy Frequency    Multi/day     Weekly        Monthly      │
│  Lead Time           < 1 hour      1 day-1 wk   1-6 months   │
│  MTTR                < 1 hour      < 1 day       > 1 week     │
│  Change Fail Rate    0-15%         16-30%        46-60%       │
│                                                               │
│  Key insight: Elite performers are BOTH faster AND more       │
│  stable. Speed and stability are NOT tradeoffs.               │
└──────────────────────────────────────────────────────────────┘
```

### DORA 메트릭 측정

```bash
# Deployment Frequency: count deploys per time period
# From your CI/CD system or deployment logs
git log --oneline --after="2024-01-01" --before="2024-02-01" \
  --grep="deploy" | wc -l

# Lead Time: measure commit-to-deploy duration
# Track the timestamp of each commit and when it reaches production

# MTTR: track incident duration
# Incident start time (alert fired) to resolution time (service restored)

# Change Failure Rate: track failed deployments
# (failed deploys / total deploys) * 100
```

---

## 6. DevOps 원칙의 실제 적용

### 세 가지 방법 (Gene Kim, *The Phoenix Project*)

#### 첫 번째 방법: 흐름 (시스템 사고)

개별 사일로가 아닌 전체 시스템을 최적화합니다.

```
Bad:  Each team optimizes locally
      Dev ships fast ──▶ QA bottleneck ──▶ Ops bottleneck ──▶ Slow delivery

Good: Optimize end-to-end flow
      Dev + QA + Ops aligned ──▶ Smooth, fast delivery
```

#### 두 번째 방법: 피드백

모든 단계에서 빠른 피드백 루프를 만듭니다.

```
Deploy ──▶ Monitor ──▶ Alert ──▶ Fix ──▶ Deploy
  │                                        ▲
  └──── Fast feedback loop (minutes) ──────┘

vs.

Deploy ──▶ Customer complaint (weeks later) ──▶ Investigate ──▶ Fix
```

#### 세 번째 방법: 지속적인 학습과 실험

실험 문화와 실패로부터의 학습을 장려합니다.

- **개선을 위한 시간 할당**: 20% 시간, 해크 데이, 혁신 스프린트
- **게임 데이 실시**: 장애를 시뮬레이션하여 인시던트 대응을 연습
- **카오스 실험 실행**: 의도적으로 시스템을 파괴하여 약점을 발견
- **학습 공유**: 사후 분석 리뷰, 내부 블로그 포스트, 기술 발표

---

## 7. 일반적인 DevOps 안티패턴

### 안티패턴 1: DevOps 팀이 또 다른 사일로

```
Bad:  Dev Team ──▶ "DevOps Team" ──▶ Ops Team
      (사일로를 허무는 대신 새로운 사일로를 만들었음)

Good: Cross-functional teams with shared DevOps practices
      [Team A: Dev + Ops + QA] [Team B: Dev + Ops + QA]
```

### 안티패턴 2: 도구 우선, 문화 나중

```
Bad:  "Kubernetes, Jenkins, Terraform을 구매했으니 이제 우리는 DevOps다!"
      (문화적 변화 없는 도구 = 비싼 장식)

Good: 문화적 변화를 먼저 시작하고, 새로운 실천을 지원하는 도구를 도입
```

### 안티패턴 3: 망가진 프로세스의 자동화

```
Bad:  수동 프로세스에 불필요한 승인 단계가 10개
      → 10단계 모두 자동화 (더 빠르지만 여전히 낭비)

Good: 불필요한 단계를 먼저 제거하고, 남은 것을 자동화
```

### 안티패턴 4: 보안 무시 (Sec 없는 DevOps)

```
Bad:  Code ──▶ Build ──▶ Test ──▶ Deploy ──▶ "잠깐, 보안 리뷰..."
      (보안이 마지막 게이트 = 지연과 반발)

Good: 모든 단계에 보안 통합 (DevSecOps)
      Code [SAST] ──▶ Build [SCA] ──▶ Test [DAST] ──▶ Deploy [runtime security]
```

---

## 8. DevOps 시작하기

### 성숙도 평가 체크리스트

각 차원에 대해 팀을 평가하십시오 (1 = 미시작, 5 = 성숙):

```
Culture:
  [ ] Blameless post-mortems conducted regularly
  [ ] Dev and Ops share on-call responsibilities
  [ ] Teams have autonomy to choose tools and processes

Automation:
  [ ] Automated build and test pipeline exists
  [ ] Infrastructure provisioned via code (not manually)
  [ ] Deployments are one-click or fully automated

Measurement:
  [ ] DORA metrics are tracked and reviewed
  [ ] Application and infrastructure monitoring in place
  [ ] Business metrics tied to delivery performance

Lean:
  [ ] Batch sizes are small (single feature per deploy)
  [ ] WIP limits enforced on boards/sprints
  [ ] Value stream mapped and bottlenecks identified

Sharing:
  [ ] Runbooks and documentation are version-controlled
  [ ] Cross-team knowledge sharing happens regularly
  [ ] Dashboards are visible to all team members
```

### 권장 시작 포인트

1. **모든 것을 버전 관리** -- 코드, 인프라, 구성, 문서
2. **기본 CI 파이프라인 구축** -- 모든 커밋에 대한 자동 빌드 및 테스트
3. **하나의 수동 프로세스 자동화** -- 가장 고통스럽거나 오류가 발생하기 쉬운 작업부터 시작
4. **DORA 메트릭 측정** -- 개선을 시도하기 전에 기준선을 설정
5. **첫 번째 비난 없는 사후 분석 실시** -- 다음 인시던트 이후

---

## 연습 문제

### 연습 문제 1: CALMS 평가

현재 또는 가상의 팀에 대해 CALMS 프레임워크를 사용하여 평가하십시오. 각 차원(Culture, Automation, Lean, Measurement, Sharing)에 대해:
1. 현재 상태를 1-5로 평가하십시오
2. 각 차원에서 가장 큰 격차를 식별하십시오
3. 각 영역을 개선하기 위한 구체적인 실행 항목을 하나씩 제안하십시오
4. 다섯 가지 실행 항목의 우선순위를 정하고 그 이유를 설명하십시오

### 연습 문제 2: 가치 스트림 매핑

여러분이 알고 있는 프로젝트(또는 가상 프로젝트)에서 기능 요청에 대한 가치 스트림을 매핑하십시오:
1. "아이디어"에서 "프로덕션 실행"까지 모든 단계를 나열하십시오
2. 각 단계의 실제 작업 시간과 대기 시간을 추정하십시오
3. 총 리드 타임과 낭비 비율(대기 시간 / 전체 시간)을 계산하십시오
4. 상위 세 가지 병목 지점을 식별하고 각각에 대한 해결책을 제안하십시오

### 연습 문제 3: DORA 메트릭 분석

다음 가상 팀의 데이터를 기반으로 성과 수준을 분류하고 개선 사항을 권장하십시오:
- 배포: 월 4회
- 평균 리드 타임: 12일
- 최근 세 건의 인시던트 복구 시간: 6시간, 48시간, 2시간
- 최근 20회 배포: 5회가 인시던트 유발
1. 각 DORA 메트릭을 계산하십시오
2. 각 메트릭에 대한 팀의 성과 수준을 분류하십시오
3. 가장 약한 메트릭을 식별하고 이를 개선하기 위한 세 가지 구체적인 조치를 제안하십시오

### 연습 문제 4: 안티패턴 식별

다음 시나리오를 읽고 모든 DevOps 안티패턴을 식별하십시오:

*"Acme Corp는 3명으로 구성된 DevOps 팀을 만들었습니다. Jenkins, Terraform, Kubernetes 라이선스를 구매했습니다. DevOps 팀이 모든 CI/CD 파이프라인을 관리합니다. 개발자는 새 서비스 배포가 필요할 때 티켓을 제출합니다. 보안 팀은 프로덕션에 배포되기 전에 모든 릴리스를 검토하며, 이는 3-5일이 소요됩니다. 배포는 변경 자문 위원회 회의 후 목요일에 진행됩니다."*

1. 발견한 모든 안티패턴을 나열하십시오
2. 각 안티패턴이 왜 유해한지 설명하십시오
3. 각각에 대한 구체적인 대안을 제안하십시오

---

[개요](00_Overview.md) | **다음**: [버전 관리 워크플로우](./02_Version_Control_Workflows.md)

**License**: CC BY-NC 4.0
