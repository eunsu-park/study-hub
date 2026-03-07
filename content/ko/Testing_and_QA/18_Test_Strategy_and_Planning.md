# 레슨 18: 테스트 전략과 계획

**이전**: [Testing Legacy Code](./17_Testing_Legacy_Code.md) | **다음**: [Overview](./00_Overview.md)

---

개별 테스트 기술 -- 단위 테스트 작성, 의존성 mocking, 커버리지 측정 -- 은 필요하지만 충분하지 않습니다. 테스트 전략은 개별 테스트가 답할 수 없는 질문들에 답합니다: 무엇을 테스트해야 하는가? 얼마나 철저하게? 어떤 수준에서? 어떤 도구로? 언제? 그리고 아마도 가장 중요한 질문: 무엇을 테스트하지 *않아야* 하는가? 일관된 전략 없이는 팀이 느리고, 불완전하고, 유지보수 비용이 높은 테스트 스위트 -- 방향 없이 유기적으로 성장한 테스트 모음 -- 를 갖게 됩니다.

이 마지막 레슨은 이 토픽의 모든 내용을 프로젝트 생애주기 전반에 걸쳐 테스트를 계획하고, 우선순위를 정하고, 지속하기 위한 전략적 프레임워크로 통합합니다.

**난이도**: ⭐⭐⭐⭐

**사전 요구사항**:
- 이 토픽의 모든 이전 레슨 (레슨 01-17)
- 팀 프로젝트 작업 경험
- 애자일/반복적 개발 프로세스에 대한 익숙함

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 가장 중요한 곳에 노력을 배분하는 위험 기반 테스트 전략을 개발할 수 있다
2. 시간이 제한된 상황을 위한 테스트 우선순위 매트릭스를 구축할 수 있다
3. 컴포넌트와 테스트 수준별로 현실적인 커버리지 목표를 설정할 수 있다
4. 테스트 메트릭을 사용하여 품질을 측정하고 개선할 수 있다
5. 테스트 실천을 애자일 개발 및 CI/CD 워크플로우에 통합할 수 있다
6. 개발 팀 내에서 테스트 문화를 구축하고 유지할 수 있다

---

## 1. 테스트 전략이란?

테스트 전략은 다음 질문에 답하는 상위 수준 문서입니다:

- **범위**: 무엇을 테스트하는가? 무엇을 명시적으로 제외하는가?
- **수준**: 단위, 통합, E2E -- 각각 어떤 비율인가?
- **도구**: 어떤 프레임워크, 라이브러리, 서비스를 사용하는가?
- **환경**: 테스트는 어디서 실행되는가? 로컬, CI, 스테이징, 프로덕션?
- **책임**: 누가 어떤 테스트를 작성하는가?
- **기준**: 릴리스하기에 "충분히 테스트"되었다고 판단하는 기준은 무엇인가?

### 1.1 테스트 전략 vs 테스트 계획

| 테스트 전략 | 테스트 계획 |
|---|---|
| 상위 수준, 장기 유지 | 구체적, 릴리스별 또는 기능별 |
| "어떤 종류의 테스트인가"에 답함 | "어떤 특정 테스트인가"에 답함 |
| 프로젝트 전체에 적용 | 특정 범위에 적용 |
| 분기별 또는 연간 업데이트 | 스프린트별 또는 마일스톤별 업데이트 |

계획 없는 전략은 모호합니다. 전략 없는 계획은 방향이 없습니다. 둘 다 필요합니다.

---

## 2. 위험 기반 테스트

어떤 프로젝트도 테스트에 무한한 시간을 가지고 있지 않습니다. 위험 기반 테스트는 실패할 가능성이 가장 높고 실패 시 가장 큰 피해를 주는 영역이 가장 많은 관심을 받도록 보장합니다.

### 2.1 위험 평가 매트릭스

```
                        영향도
                낮음        중간         높음
           ┌───────────┬────────────┬────────────┐
    높음   │  모니터링  │  철저한    │  핵심      │
가능성     │           │  테스트    │  테스트    │
           ├───────────┼────────────┼────────────┤
    중간   │  기본     │  표준      │  철저한    │
           │  테스트   │  테스트    │  테스트    │
           ├───────────┼────────────┼────────────┤
    낮음   │  최소     │  기본      │  표준      │
           │  테스트   │  테스트    │  테스트    │
           └───────────┴────────────┴────────────┘
```

### 2.2 가능성 평가

결함의 가능성을 높이는 요소:

| 요소 | 낮은 위험 | 높은 위험 |
|---|---|---|
| 코드 복잡도 | 단순 CRUD | 복잡한 알고리즘 |
| 변경 빈도 | 안정적, 거의 수정 안 됨 | 빈번한 변경 |
| 개발자 경험 | 시니어, 도메인 전문가 | 코드베이스에 처음 |
| 의존성 | 적고, 안정적 | 많고, 변동적 |
| 기술 성숙도 | 검증된 스택 | 최신 기술 |

### 2.3 영향도 평가

결함의 영향을 높이는 요소:

| 요소 | 낮은 영향 | 높은 영향 |
|---|---|---|
| 영향받는 사용자 | 내부 도구 | 고객 대면 |
| 데이터 민감도 | 공개 데이터 | 금융, 개인정보 |
| 복구 난이도 | 쉬운 롤백 | 데이터 손상 |
| 규제 | 컴플라이언스 없음 | SOX, HIPAA, PCI |
| 수익 | 무료 기능 | 결제 처리 |

### 2.4 위험 기반 테스트 적용

```python
# Example: Risk-based test allocation for an e-commerce platform

RISK_ASSESSMENT = {
    "payment_processing": {
        "likelihood": "high",     # Complex, many edge cases
        "impact": "critical",     # Revenue, legal, PCI compliance
        "test_strategy": {
            "unit_tests": "comprehensive (>95% coverage)",
            "integration_tests": "all payment gateways, all currencies",
            "e2e_tests": "full checkout flow, including errors",
            "security_tests": "PCI DSS compliance scanning",
            "performance_tests": "load testing under peak traffic",
        }
    },
    "user_profile_page": {
        "likelihood": "low",      # Simple CRUD, stable code
        "impact": "low",          # Cosmetic, no data sensitivity
        "test_strategy": {
            "unit_tests": "basic validation logic",
            "integration_tests": "one happy path",
            "e2e_tests": "none",
        }
    },
    "search_functionality": {
        "likelihood": "medium",   # Moderate complexity
        "impact": "medium",       # Core UX but not data-critical
        "test_strategy": {
            "unit_tests": "query parsing, ranking logic",
            "integration_tests": "search index, relevance",
            "performance_tests": "response time under load",
        }
    },
}
```

---

## 3. 테스트 우선순위 매트릭스

시간이 제한적일 때(항상 그렇듯이), 구조화된 매트릭스를 사용하여 테스트 노력의 우선순위를 정합니다.

### 3.1 우선순위 프레임워크

```python
def calculate_test_priority(component):
    """
    Score each component to determine testing priority.
    Higher score = more testing investment needed.
    """
    score = 0

    # Business criticality (0-10)
    score += component.business_value * 2  # Weight: 2x

    # Change frequency (0-10)
    score += component.change_frequency * 1.5  # Weight: 1.5x

    # Defect history (0-10)
    score += component.past_defects * 1.5  # Weight: 1.5x

    # Complexity (0-10)
    score += component.cyclomatic_complexity_normalized

    # User visibility (0-10)
    score += component.user_facing * 1.0

    return score
```

### 3.2 우선순위 예시

| 컴포넌트 | 비즈니스 | 변경 | 결함 | 복잡도 | 가시성 | 점수 | 우선순위 |
|---|---|---|---|---|---|---|---|
| 결제 | 10 | 6 | 8 | 9 | 10 | 62 | 핵심 |
| 인증 | 9 | 4 | 5 | 7 | 8 | 51 | 높음 |
| 검색 | 7 | 7 | 4 | 6 | 9 | 48 | 높음 |
| 관리자 패널 | 3 | 3 | 2 | 4 | 2 | 19 | 낮음 |
| 로깅 | 2 | 1 | 1 | 2 | 0 | 8 | 최소 |

---

## 4. 컴포넌트별 커버리지 목표

커버리지는 단일 숫자가 아닙니다. 다양한 컴포넌트는 위험 프로필에 따라 다른 커버리지 수준이 필요합니다.

### 4.1 계층별 커버리지 목표

```python
COVERAGE_GOALS = {
    # Tier 1: Critical business logic
    "critical": {
        "line_coverage": 95,
        "branch_coverage": 90,
        "mutation_score": 80,
        "components": ["payment", "auth", "data_integrity"],
    },

    # Tier 2: Important but lower risk
    "important": {
        "line_coverage": 80,
        "branch_coverage": 70,
        "mutation_score": 60,
        "components": ["search", "notifications", "reporting"],
    },

    # Tier 3: Standard coverage
    "standard": {
        "line_coverage": 70,
        "branch_coverage": 50,
        "mutation_score": 0,  # Not required
        "components": ["user_profile", "settings", "help"],
    },

    # Tier 4: Minimal coverage (UI, glue code)
    "minimal": {
        "line_coverage": 50,
        "branch_coverage": 0,  # Not measured
        "mutation_score": 0,
        "components": ["admin", "migrations", "scripts"],
    },
}
```

### 4.2 CI에서 계층별 커버리지 적용

```python
# conftest.py
import json

import pytest


def pytest_sessionfinish(session, exitstatus):
    """Check coverage goals by component after all tests run."""
    coverage_file = "coverage.json"
    try:
        with open(coverage_file) as f:
            coverage_data = json.load(f)
    except FileNotFoundError:
        return

    violations = []
    for tier_name, tier_config in COVERAGE_GOALS.items():
        for component in tier_config["components"]:
            actual = get_component_coverage(coverage_data, component)
            goal = tier_config["line_coverage"]
            if actual < goal:
                violations.append(
                    f"{component}: {actual:.1f}% < {goal}% goal ({tier_name} tier)"
                )

    if violations:
        print("\nCoverage goal violations:")
        for v in violations:
            print(f"  FAIL: {v}")
```

---

## 5. 테스트 문서화

### 5.1 테스트 전략 문서 템플릿

```markdown
# Test Strategy: [Project Name]

## 1. Scope
- **In scope**: [List of components/features to test]
- **Out of scope**: [Explicitly excluded items and why]

## 2. Testing Levels
- **Unit tests**: [Who writes, tools, coverage goals]
- **Integration tests**: [Scope, environment, data]
- **E2E tests**: [Scenarios, tools, frequency]
- **Non-functional tests**: [Performance, security, accessibility]

## 3. Environments
- **Local**: [What runs on developer machines]
- **CI**: [What runs on every commit/PR]
- **Staging**: [What runs before release]
- **Production**: [Smoke tests, monitoring]

## 4. Tools
- **Framework**: pytest
- **Mocking**: unittest.mock
- **Coverage**: pytest-cov
- **CI**: GitHub Actions
- **Performance**: Locust
- **Security**: Bandit, pip-audit

## 5. Entry/Exit Criteria
- **Entry**: Code compiles, passes linting, author self-tested
- **Exit**: All CI checks pass, coverage goals met, no critical bugs

## 6. Risk-Based Priorities
[Risk matrix for each component]

## 7. Responsibilities
- **Developers**: Unit tests, integration tests
- **QA**: E2E tests, exploratory testing
- **Security**: Security scans, penetration testing
- **SRE**: Performance testing, monitoring

## 8. Review Schedule
- Strategy reviewed quarterly
- Updated when architecture changes
```

### 5.2 기능별 테스트 계획

```markdown
# Test Plan: User Registration Feature

## Changes
- New API endpoint: POST /api/v1/register
- New database model: PendingRegistration
- Email verification flow

## Test Cases

### Unit Tests
1. Validate email format (valid, invalid, edge cases)
2. Validate password strength (length, complexity)
3. Check duplicate email detection
4. Token generation and expiration

### Integration Tests
1. Full registration → verification → activation flow
2. Registration with existing email
3. Token expiration handling
4. Database constraint enforcement

### E2E Tests
1. Complete registration from UI
2. Email link opens verification page
3. Verified user can log in

## Not Testing (with justification)
- Email delivery reliability (covered by email service SLA)
- Browser compatibility (covered by existing E2E framework)
```

---

## 6. 애자일과 CI/CD에서의 테스트

### 6.1 스프린트에서의 테스트

```
스프린트 계획:
├── 스토리 추정에 테스트 태스크 포함
├── "완료 정의(Definition of Done)"에 테스트 포함
└── 스프린트 용량의 20-30%를 테스트에 할당

스프린트 진행 중:
├── 코드와 함께 테스트 작성 (이후가 아닌)
├── push 전에 로컬에서 테스트 실행
├── 코드 리뷰에 테스트 리뷰 포함
└── 깨진 테스트 즉시 수정 (라인 중단)

스프린트 리뷰:
├── 테스트 커버리지 변화 표시
├── 발견되고 수정된 결함 논의
└── 테스트 안정성 메트릭 검토
```

### 6.2 CI/CD 파이프라인에서의 테스트

```
                        신뢰도
                        ▲
Commit ──────────────── │ ──────────────── Deploy
  │                     │                    │
  ├─ Lint (초)          │ 낮은 위험,         │
  ├─ 단위 테스트 (분)   │ 빠른 피드백        │
  ├─ 통합 테스트 (분)   │                    │
  ├─ 보안 스캔          │                    │
  │                     │                    │
  ├─ E2E (스테이징)     │ 더 높은 신뢰도     │
  ├─ 성능 테스트        │                    │
  │                     │                    │
  ├─ 카나리 배포        │ 프로덕션           │
  ├─ 스모크 테스트      │ 검증               │
  └─ 모니터링/알림      │                    ▼
```

### 6.3 시프트 레프트 테스트

"시프트 레프트"는 개발 생애주기에서 테스트를 더 일찍 이동시키는 것을 의미합니다:

| 단계 | 전통적 | 시프트 레프트 |
|---|---|---|
| 요구사항 | 테스트 없음 | 테스트 가능성 검토 |
| 설계 | 테스트 없음 | 테스트 전략 계획 |
| 개발 | 코드 먼저, 테스트 나중 | TDD, 코드와 함께 테스트 |
| 코드 리뷰 | 기능 확인 | 테스트도 리뷰 |
| CI | 병합 후 테스트 실행 | 모든 커밋에서 테스트 실행 |
| 릴리스 | 마지막에 QA 게이트 | 지속적 테스트 |

---

## 7. 테스트 메트릭

메트릭은 의사결정에 정보를 제공해야 하며, 의사결정을 주도해서는 안 됩니다. 맥락 없는 메트릭은 위험합니다.

### 7.1 품질 메트릭

| 메트릭 | 측정 대상 | 목표 | 경고 신호 |
|---|---|---|---|
| **결함 밀도** | 코드 1,000줄당 결함 수 | < 5 | > 10은 낮은 품질 |
| **결함 탈출률** | 프로덕션 도달 버그 / 발견된 총 버그 | < 10% | > 25%는 테스트가 너무 많이 놓침 |
| **MTTR** (평균 복구 시간) | 버그 보고부터 수정 배포까지 평균 시간 | 핵심 건 < 4시간 | > 24시간은 프로세스 문제 |
| **MTTD** (평균 감지 시간) | 버그 도입부터 감지까지 시간 | < 1일 | > 1 스프린트는 테스트 누락 |

### 7.2 테스트 스위트 메트릭

| 메트릭 | 측정 대상 | 목표 | 경고 신호 |
|---|---|---|---|
| **테스트 실행 시간** | 전체 스위트 소요 시간 | < 10분 | > 30분은 피드백 루프를 죽임 |
| **불안정 테스트 비율** | 비결정적으로 통과/실패하는 테스트 | < 1% | > 5%는 테스트 신뢰를 침식 |
| **커버리지 추세** | 시간에 따른 커버리지 (증가해야 함) | 양의 기울기 | 감소는 테스트 건너뛰기를 나타냄 |
| **테스트 대 코드 비율** | 프로덕션 코드 대비 테스트 코드 줄 수 | 1:1 ~ 3:1 | < 0.5:1은 테스트 부족 |

### 7.3 시간에 따른 메트릭 추적

```python
# metrics.py — collect and report testing metrics
import json
from dataclasses import dataclass
from datetime import datetime


@dataclass
class TestMetrics:
    date: str
    total_tests: int
    passed: int
    failed: int
    skipped: int
    duration_seconds: float
    line_coverage: float
    flaky_tests: int

    @property
    def pass_rate(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return (self.passed / self.total_tests) * 100

    @property
    def flaky_rate(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return (self.flaky_tests / self.total_tests) * 100


def record_metrics(metrics: TestMetrics, filepath: str = "test_metrics.json"):
    """Append metrics to a JSON file for trend tracking."""
    try:
        with open(filepath) as f:
            history = json.load(f)
    except FileNotFoundError:
        history = []

    history.append({
        "date": metrics.date,
        "total_tests": metrics.total_tests,
        "pass_rate": metrics.pass_rate,
        "duration_seconds": metrics.duration_seconds,
        "line_coverage": metrics.line_coverage,
        "flaky_rate": metrics.flaky_rate,
    })

    with open(filepath, "w") as f:
        json.dump(history, f, indent=2)


def detect_regressions(history: list, window: int = 5) -> list:
    """Detect negative trends in metrics over the last N data points."""
    if len(history) < window:
        return []

    recent = history[-window:]
    warnings = []

    # Coverage declining?
    coverage_trend = recent[-1]["line_coverage"] - recent[0]["line_coverage"]
    if coverage_trend < -2:
        warnings.append(f"Coverage declining: {coverage_trend:.1f}% over {window} runs")

    # Duration increasing?
    duration_trend = recent[-1]["duration_seconds"] - recent[0]["duration_seconds"]
    if duration_trend > 60:
        warnings.append(f"Test duration increasing: +{duration_trend:.0f}s over {window} runs")

    # Flaky rate increasing?
    if recent[-1]["flaky_rate"] > 5:
        warnings.append(f"Flaky test rate: {recent[-1]['flaky_rate']:.1f}% (target: <1%)")

    return warnings
```

---

## 8. 테스트 문화 구축

도구와 프로세스는 필요하지만 충분하지 않습니다. 지속 가능한 테스트 실천은 품질을 중시하는 팀 문화를 필요로 합니다.

### 8.1 테스트 문화의 원칙

1. **테스트는 선택 사항이 아닙니다**: 완료 정의의 일부이며, 있으면 좋은 것이 아닙니다
2. **깨진 테스트는 최우선 과제**: 깨진 테스트는 프로덕션 위험입니다; 새 코드 작성 전에 수정합니다
3. **테스트는 코드입니다**: 프로덕션 코드와 동일한 관심, 리뷰, 리팩토링을 받을 자격이 있습니다
4. **커버리지는 하한선이지 상한선이 아닙니다**: 커버리지 목표 달성이 코드가 잘 테스트되었음을 의미하지 않습니다
5. **불안정한 테스트는 버그입니다**: 신뢰를 침식하며 수정하거나 삭제해야 합니다

### 8.2 실천 단계

| 조치 | 영향 | 노력 |
|---|---|---|
| 완료 정의에 테스트 포함 | 높음 | 낮음 |
| 코드 리뷰에서 테스트 리뷰 | 높음 | 낮음 |
| 테스트 메트릭 추적 및 공개 | 중간 | 낮음 |
| 불안정한 테스트 즉시 수정 | 높음 | 중간 |
| 테스트 설계에 페어 프로그래밍 | 높음 | 중간 |
| 테스트 작성 워크숍 | 중간 | 중간 |
| 테스트 이정표 축하 | 중간 | 낮음 |

### 8.3 피해야 할 안티패턴

- **"나중에 테스트를 추가하겠습니다"**: 나중은 오지 않습니다. 코드와 함께 테스트하십시오.
- **커버리지 극장**: 숫자를 맞추기 위해서만 테스트를 작성합니다. 의미 있는 커버리지에 집중하십시오.
- **테스트를 안 했다고 비난하기**: 개발자에게 시간을 주지 않으면서 테스트를 안 했다고 비난합니다.
- **불안정한 테스트 무시**: "아, 그 테스트는 항상 불안정해, 그냥 다시 돌려." 수정하거나 삭제하십시오.
- **별도의 QA 팀이 모든 테스트 작성**: 개발자가 자신의 테스트를 소유해야 합니다. QA는 대체가 아닌 보완적 계층을 제공합니다.

---

## 9. 모든 것을 종합하기

중간 규모 프로젝트를 위한 완전한 테스트 전략:

```
테스트 전략 요약
═══════════════════════

수준:
  단위 테스트     → 테스트의 70%, 총 5분 미만, 80% 이상 커버리지
  통합 테스트     → 테스트의 20%, 총 10분 미만, 핵심 경로
  E2E             → 테스트의 10%, 총 15분 미만, 사용자 여정

실행:
  모든 커밋마다   → Lint, 단위 테스트, 보안 스캔
  모든 PR마다     → 통합 테스트, 커버리지 검사
  매일 야간       → 전체 E2E 스위트, 성능 테스트
  매주            → 의존성 감사, DAST 스캔

위험 배분:
  핵심 (결제, 인증)       → 95% 커버리지, 속성 테스트, 뮤테이션
  중요 (검색, 리포트)     → 80% 커버리지, 통합 테스트
  표준 (프로필, 설정)     → 70% 커버리지, 단위 테스트
  최소 (관리자, 스크립트)  → 50% 커버리지, 스모크 테스트

추적 메트릭:
  - 커버리지 추세 (컴포넌트별)
  - 결함 탈출률
  - 테스트 스위트 실행 시간
  - 불안정 테스트 비율
  - 핵심 버그에 대한 MTTR

검토 주기:
  스프린트 회고   → 불안정 테스트, 테스트 격차 검토
  분기별          → 테스트 전략 검토 및 업데이트
  연간            → 도구 및 프레임워크 평가
```

---

## 연습 문제

1. **위험 평가**: 실제 애플리케이션(또는 가상의 전자상거래 플랫폼)을 선택하고 상위 10개 컴포넌트에 대한 위험 평가 매트릭스를 작성하십시오. 각 컴포넌트에 테스트 수준과 커버리지 목표를 배정하십시오.

2. **테스트 전략 문서**: 현재 작업 중인 프로젝트를 위한 완전한 테스트 전략 문서를 작성하십시오. 범위, 수준, 도구, 환경, 책임, 기준을 포함하십시오.

3. **메트릭 대시보드**: pytest 출력(JUnit XML + 커버리지 리포트)을 읽고 메트릭 요약을 생성하는 Python 스크립트를 작성하십시오. 최근 10회 실행에 대한 커버리지 추세, 통과율, 실행 시간을 추적하십시오.

4. **우선순위 연습**: 테스트되지 않은 20개의 컴포넌트 백로그와 40시간의 예산이 주어졌을 때, 우선순위 매트릭스를 사용하여 어떤 컴포넌트를 먼저, 어떤 수준으로 테스트할지 결정하십시오. 선택을 정당화하십시오.

5. **문화 평가**: 세 명의 팀원과 테스트 실천에 대해 인터뷰하십시오. 효과적인 테스트에 대한 상위 세 가지 문화적 장벽을 식별하고, 각각을 해결하기 위한 구체적인 조치를 제안하십시오.

---

**License**: CC BY-NC 4.0
