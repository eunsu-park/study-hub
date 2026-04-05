# 공격 보안 개요

**이전**: [00. 개요](./00_Overview.md) | **다음**: [02. 정찰](./02_Reconnaissance.md)

---

공격 보안은 실제 공격을 시뮬레이션하여 시스템, 네트워크 및 애플리케이션을 사전에 테스트하는 분야입니다. 방어 보안이 벽을 쌓고 침해를 모니터링하는 데 중점을 두는 반면, 공격 보안은 허가를 받아 적극적으로 이러한 벽을 뚫으려 시도합니다. 이 레슨은 윤리적, 법적 및 방법론적 기초를 확립합니다.

> **중요**: 이 과정의 모든 기법은 소유하거나 명시적인 서면 허가를 받은 시스템에서만 사용해야 합니다. 무단 접근은 범죄입니다.

**난이도**: ⭐⭐⭐

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 공격 보안 작업을 지배하는 윤리적 프레임워크 설명
2. 주요 법적 프레임워크(CFAA, Computer Misuse Act, GDPR 영향) 이해
3. 침투 테스트, 레드 팀, 취약점 평가 구별
4. PTES(Penetration Testing Execution Standard) 방법론 적용
5. 공인된 보안 평가를 위한 교전 규칙 작성
6. 책임 있는 공개 및 버그 바운티 프로그램 이해
7. 공격 기법 연습을 위한 안전한 랩 환경 설정
8. MITRE ATT&CK 프레임워크를 침투 테스트 단계에 매핑

---

## 목차

1. [공격 보안 마인드셋](#1-공격-보안-마인드셋)
2. [윤리 및 법적 프레임워크](#2-윤리-및-법적-프레임워크)
3. [보안 평가 유형](#3-보안-평가-유형)
4. [침투 테스트 방법론](#4-침투-테스트-방법론)
5. [교전 규칙](#5-교전-규칙)
6. [킬 체인과 MITRE ATT&CK](#6-킬-체인과-mitre-attck)
7. [책임 있는 공개](#7-책임-있는-공개)
8. [랩 환경 설정](#8-랩-환경-설정)
9. [범위 지정 및 계획](#9-범위-지정-및-계획)
10. [문서화 및 보고](#10-문서화-및-보고)
11. [경력 경로 및 자격증](#11-경력-경로-및-자격증)
12. [연습문제](#12-연습문제)
13. [요약](#13-요약)
14. [참고 자료](#14-참고-자료)

---

## 1. 공격 보안 마인드셋

공격 보안 전문가는 적대자처럼 생각하되 성실하게 행동합니다. 모든 가능한 약점을 찾으려는 기술적 호기심과 그 지식을 책임감 있게 사용하는 윤리적 규율의 결합이 이 직업을 정의합니다.

### 1.1 공격자처럼 생각하기

공격자는 규칙을 따르지 않습니다. 기술적 익스플로잇, 소셜 엔지니어링, 설정 오류 및 인적 오류를 결합하여 최소 저항 경로를 찾습니다. 효과적인 침투 테스터는 엄격한 윤리적 경계를 유지하면서도 이러한 창의적이고 제약 없는 사고를 채택해야 합니다.

**공격자 마인드셋의 핵심 원칙:**

- **아무것도 안전하지 않다고 가정**: 모든 시스템에는 취약점이 있으며, 문제는 활동 기간 내에 찾을 수 있느냐입니다
- **약점 체이닝**: 개별 낮은 심각도 발견사항이 결합하여 중요한 공격 경로가 될 수 있습니다
- **횡적 사고**: 가장 영향력 있는 취약점은 종종 예상치 못한 각도에서 옵니다 — 공급망 침해, 간과된 API 엔드포인트, 시스템 간 신뢰 관계 등
- **끈기**: 실제 공격자는 첫 번째 실패 후 포기하지 않으며, 열거하고 피벗하고 다른 방식을 시도합니다

```python
"""
공격 표면 열거 — 공격자 마인드셋 모델링.

공격 보안 전문가가 대상 시스템의 잠재적 진입점을
체계적으로 식별하는 방법을 시연합니다.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class AttackVector(Enum):
    """STRIDE 모델을 따르는 공격 벡터 카테고리."""
    NETWORK = "네트워크 기반"
    WEB_APP = "웹 애플리케이션"
    SOCIAL = "소셜 엔지니어링"
    PHYSICAL = "물리적 접근"
    SUPPLY_CHAIN = "공급망"
    WIRELESS = "무선"
    INSIDER = "내부자 위협"


class Severity(Enum):
    """CVSS 정성적 등급에 맞춘 위험 심각도."""
    CRITICAL = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1
    INFO = 0


@dataclass
class AttackSurface:
    """대상 시스템의 잠재적 진입점."""
    name: str
    vector: AttackVector
    description: str
    severity: Severity
    mitigations: list[str] = field(default_factory=list)
    prerequisites: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)

    def risk_score(self) -> float:
        """단순화된 위험 점수 계산 (0-10)."""
        base = self.severity.value * 2.5
        # 완화 조치가 있으면 점수 감소
        mitigation_factor = max(0.3, 1.0 - (len(self.mitigations) * 0.15))
        # 사전 조건이 적을수록 점수 증가 (익스플로잇 용이)
        prereq_factor = max(0.5, 1.0 + (0.1 * (3 - len(self.prerequisites))))
        return round(min(10.0, base * mitigation_factor * prereq_factor), 1)


def enumerate_web_attack_surface() -> list[AttackSurface]:
    """
    일반적인 웹 애플리케이션 공격 표면을 열거합니다.

    실제 활동에서는 정찰과 자동화된 스캐닝으로 채워집니다.
    """
    surfaces = [
        AttackSurface(
            name="로그인 폼",
            vector=AttackVector.WEB_APP,
            description="크리덴셜을 받는 사용자 인증 엔드포인트",
            severity=Severity.HIGH,
            mitigations=["속도 제한", "계정 잠금", "MFA"],
            prerequisites=["유효한 사용자명 열거"],
            tools=["Burp Suite", "Hydra", "커스텀 스크립트"],
        ),
        AttackSurface(
            name="REST API",
            vector=AttackVector.WEB_APP,
            description="인증 토큰이 있는 JSON API 엔드포인트",
            severity=Severity.HIGH,
            mitigations=["JWT 검증", "RBAC", "입력 유효성 검사"],
            prerequisites=["API 문서 또는 엔드포인트 발견"],
            tools=["Postman", "ffuf", "Burp Suite"],
        ),
        AttackSurface(
            name="파일 업로드",
            vector=AttackVector.WEB_APP,
            description="사용자 대상 파일 업로드 기능",
            severity=Severity.CRITICAL,
            mitigations=["파일 유형 검증", "샌드박스 스토리지"],
            prerequisites=["인증된 세션"],
            tools=["Burp Suite", "커스텀 폴리글롯 파일"],
        ),
        AttackSurface(
            name="DNS 레코드",
            vector=AttackVector.NETWORK,
            description="인프라를 노출하는 공개 DNS 레코드",
            severity=Severity.LOW,
            mitigations=["최소 DNS 노출", "스플릿-호라이즌 DNS"],
            prerequisites=[],
            tools=["dig", "nslookup", "subfinder", "amass"],
        ),
        AttackSurface(
            name="SSL/TLS 설정",
            vector=AttackVector.NETWORK,
            description="TLS 버전 및 암호화 스위트 설정",
            severity=Severity.MEDIUM,
            mitigations=["TLS 1.3만 사용", "강력한 암호화 스위트"],
            prerequisites=[],
            tools=["testssl.sh", "sslyze", "nmap"],
        ),
    ]
    return surfaces


def prioritize_attack_surfaces(
    surfaces: list[AttackSurface],
) -> list[AttackSurface]:
    """공격 표면을 위험 점수 기준으로 정렬합니다 (높은 순)."""
    return sorted(surfaces, key=lambda s: s.risk_score(), reverse=True)


# 실행 예시
if __name__ == "__main__":
    surfaces = enumerate_web_attack_surface()
    prioritized = prioritize_attack_surfaces(surfaces)

    print("=" * 70)
    print("공격 표면 분석 — 위험도별 정렬")
    print("=" * 70)
    for i, surface in enumerate(prioritized, 1):
        print(f"\n[{i}] {surface.name}")
        print(f"    벡터:      {surface.vector.value}")
        print(f"    심각도:    {surface.severity.name}")
        print(f"    위험 점수: {surface.risk_score()}/10")
        print(f"    설명:      {surface.description}")
        print(f"    도구:      {', '.join(surface.tools)}")
        if surface.mitigations:
            print(f"    완화 조치: {', '.join(surface.mitigations)}")
```

### 1.2 해커와 크래커의 차이

보안 커뮤니티는 동기에 따라 다음과 같이 구분합니다:

| 분류 | 동기 | 허가 | 목적 |
|------|------|------|------|
| 화이트햇 | 방어적 | 허가됨 | 보안 향상 |
| 블랙햇 | 악의적 | 무단 | 개인 이익 |
| 그레이햇 | 혼합 | 때때로 | 공개 여부 다양 |
| 버그 바운티 헌터 | 금전적 + 윤리적 | 프로그램 범위 내 | 버그 발견 및 보고 |
| 레드 티머 | 적대자 시뮬레이션 | 계약됨 | 탐지/대응 테스트 |

### 1.3 침투 테스터의 윤리 강령

전문 침투 테스터는 엄격한 윤리 지침을 준수합니다:

1. **서면 허가**: 명시적이고 서명된 허가 없이 절대 테스트 금지
2. **범위 준수**: 정의된 범위 내에서 활동 — 경계를 넘어서지 않음
3. **데이터 보호**: 발견된 민감한 데이터는 극도로 주의하여 처리
4. **피해 최소화**: 프로덕션 시스템에 불필요한 중단 초래 금지
5. **완전한 공개**: 모든 발견사항을 클라이언트에게 보고 — 중요한 것만이 아님
6. **기밀 유지**: 클라이언트 취약점을 제3자에게 절대 공개 금지
7. **지속적 학습**: 진화하는 위협에 맞게 기술을 최신으로 유지

---

## 2. 윤리 및 법적 프레임워크

### 2.1 미국 CFAA (Computer Fraud and Abuse Act)

CFAA(18 U.S.C. § 1030)는 컴퓨터 시스템에 대한 무단 접근을 범죄화합니다:

- **무단 접근**: 허가 없이 컴퓨터에 접근
- **허가 범위 초과**: 일부 접근 권한이 있지만 허가된 범위를 초과
- **처벌**: 초범 최대 10년, 재범 20년 징역
- **민사 책임**: 피해자가 손해배상 소송 가능

> **침투 테스터에게 중요**: 구두 허가가 있더라도 항상 범위, 일정, 허용 방법을 명시적으로 정의한 **서면 허가**를 받아야 합니다.

### 2.2 영국 Computer Misuse Act (CMA)

CMA 1990은 세 가지 주요 범죄를 정의합니다:

1. 컴퓨터 자료에 대한 무단 접근 (제1조)
2. 추가 범죄 의도가 있는 무단 접근 (제2조)
3. 컴퓨터 자료의 무단 수정 (제3조)

### 2.3 GDPR 영향

EU 개인 데이터를 처리하는 시스템을 테스트할 때:

- 침투 테스트는 개인 데이터 처리를 포함할 수 있습니다
- 테스트 범위는 데이터 처리 계약(DPA)에 문서화되어야 합니다
- 테스트 중 접근한 개인 데이터는 GDPR 요건에 따라 처리해야 합니다
- 테스트로 인해 실제 침해가 발생하면 데이터 침해 통지 의무가 적용될 수 있습니다

```python
"""
교전 규칙(RoE) 문서 생성기.

침투 테스트 활동을 위한 표준화된 허가 문서를 생성합니다.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class TestType(Enum):
    """보안 평가 유형."""
    BLACK_BOX = "블랙 박스 — 사전 지식 없음"
    GREY_BOX = "그레이 박스 — 부분적 지식/크리덴셜"
    WHITE_BOX = "화이트 박스 — 전체 소스 코드 및 아키텍처 접근"


class TestScope(Enum):
    """테스트 범위 카테고리."""
    EXTERNAL = "외부 — 인터넷에 노출된 자산만"
    INTERNAL = "내부 — 내부 네트워크 접근 제공"
    WEB_APP = "웹 애플리케이션 — 특정 웹 애플리케이션"
    MOBILE = "모바일 애플리케이션 — iOS/Android 앱"
    WIRELESS = "무선 — WiFi 및 블루투스"
    SOCIAL = "소셜 엔지니어링 — 피싱, 비싱, 물리적"
    CLOUD = "클라우드 인프라 — AWS/GCP/Azure"


@dataclass
class RulesOfEngagement:
    """침투 테스트의 공식 교전 규칙."""
    client_name: str
    tester_name: str
    tester_company: str
    test_type: TestType
    scope: list[TestScope]
    start_date: datetime
    end_date: datetime
    in_scope_targets: list[str]
    out_of_scope_targets: list[str]
    testing_hours: str = "09:00 - 17:00 UTC"
    emergency_contact: str = ""
    allowed_techniques: list[str] = field(default_factory=list)
    prohibited_techniques: list[str] = field(default_factory=list)
    data_handling: str = "모든 데이터 암호화; 보고서 납품 30일 후 폐기"

    def generate_document(self) -> str:
        """형식화된 RoE 문서를 생성합니다."""
        doc = []
        doc.append("=" * 70)
        doc.append("교전 규칙 — 침투 테스트 허가서")
        doc.append("=" * 70)
        doc.append("")
        doc.append(f"클라이언트:      {self.client_name}")
        doc.append(f"테스터:          {self.tester_name} ({self.tester_company})")
        doc.append(f"테스트 유형:     {self.test_type.value}")
        doc.append(f"기간:            {self.start_date:%Y-%m-%d} ~ {self.end_date:%Y-%m-%d}")
        doc.append(f"테스트 시간:     {self.testing_hours}")
        doc.append(f"긴급 연락처:     {self.emergency_contact}")
        doc.append("")
        doc.append("범위:")
        for s in self.scope:
            doc.append(f"  [+] {s.value}")
        doc.append("")
        doc.append("범위 내 대상:")
        for t in self.in_scope_targets:
            doc.append(f"  [+] {t}")
        doc.append("")
        doc.append("범위 외 대상:")
        for t in self.out_of_scope_targets:
            doc.append(f"  [-] {t}")
        doc.append("")
        if self.allowed_techniques:
            doc.append("허용된 기법:")
            for t in self.allowed_techniques:
                doc.append(f"  [+] {t}")
            doc.append("")
        if self.prohibited_techniques:
            doc.append("금지된 기법:")
            for t in self.prohibited_techniques:
                doc.append(f"  [-] {t}")
            doc.append("")
        doc.append(f"데이터 처리: {self.data_handling}")
        doc.append("")
        doc.append("서명:")
        doc.append(f"  클라이언트: _________________ 날짜: _________")
        doc.append(f"  테스터:     _________________ 날짜: _________")
        doc.append("=" * 70)
        return "\n".join(doc)

    def validate(self) -> list[str]:
        """RoE의 일반적인 문제를 검증합니다."""
        issues = []
        if self.end_date <= self.start_date:
            issues.append("종료일은 시작일 이후여야 합니다")
        if not self.in_scope_targets:
            issues.append("범위 내 대상이 최소 하나 이상 필요합니다")
        if not self.emergency_contact:
            issues.append("긴급 연락처가 필요합니다")
        if self.end_date - self.start_date > timedelta(days=90):
            issues.append("활동 기간이 90일을 초과합니다 — 분리를 고려하세요")
        return issues


# 사용 예시
if __name__ == "__main__":
    roe = RulesOfEngagement(
        client_name="예시 기업",
        tester_name="Jane Smith, OSCP",
        tester_company="SecureTest Labs",
        test_type=TestType.GREY_BOX,
        scope=[TestScope.EXTERNAL, TestScope.WEB_APP],
        start_date=datetime(2025, 6, 1),
        end_date=datetime(2025, 6, 14),
        in_scope_targets=[
            "*.example.com (웹 애플리케이션)",
            "203.0.113.0/24 (외부 네트워크 대역)",
            "api.example.com (REST API)",
        ],
        out_of_scope_targets=[
            "프로덕션 데이터베이스 서버",
            "결제 처리 시스템 (PCI 범위)",
            "제3자 SaaS 통합",
            "직원 개인 기기",
        ],
        emergency_contact="security@example.com / +82-10-0000-0000 (24/7 SOC)",
        allowed_techniques=[
            "자동화된 취약점 스캐닝",
            "수동 웹 애플리케이션 테스트",
            "크리덴셜 브루트포싱 (속도 제한)",
            "소셜 엔지니어링 (피싱 시뮬레이션 — 사전 승인 목록)",
        ],
        prohibited_techniques=[
            "서비스 거부 공격 (DoS/DDoS)",
            "물리적 침입",
            "제3자 시스템 테스트",
            "프로덕션 데이터 수정 또는 삭제",
        ],
    )

    issues = roe.validate()
    if issues:
        print("검증 문제:")
        for issue in issues:
            print(f"  [!] {issue}")
    else:
        print(roe.generate_document())
```

---

## 3. 보안 평가 유형

보안 평가 스펙트럼을 이해하면 각 상황에 맞는 접근 방법을 선택하는 데 도움이 됩니다.

### 3.1 취약점 평가 (Vulnerability Assessment)

**취약점 평가**는 취약점을 익스플로잇하지 않고 식별하고 정량화합니다:

- **범위**: 넓음, 모든 자산 포함
- **깊이**: 표면적; 식별하지만 검증하지 않음
- **기간**: 수일~1주
- **결과물**: CVSS 점수와 함께 우선순위가 지정된 취약점 목록
- **도구**: Nessus, OpenVAS, Qualys, Nexpose

### 3.2 침투 테스트 (Penetration Testing)

**침투 테스트**(펜테스트)는 실제 영향을 증명하기 위해 취약점을 능동적으로 익스플로잇합니다:

- **범위**: 대상 지정, 특정 시스템 또는 애플리케이션
- **깊이**: 깊음; 전체 익스플로잇 체인 시도
- **기간**: 1~4주
- **결과물**: 개념 증명(PoC) 익스플로잇 및 수정 지침이 포함된 상세 발견사항
- **단계**: 정찰 → 스캐닝 → 익스플로잇 → 사후 익스플로잇 → 보고

### 3.3 레드 팀 활동 (Red Team Engagement)

**레드 팀 활동**은 장기간에 걸쳐 정교한 적대자를 시뮬레이션합니다:

- **범위**: 조직 전체, 사람과 프로세스 포함
- **깊이**: 최대; APT(Advanced Persistent Threat) 전술 모방
- **기간**: 수주~수개월
- **결과물**: 탐지 및 대응 역량 평가
- **핵심 차이**: 기술적 취약점만이 아니라 블루팀의 탐지 및 대응 능력을 테스트

### 3.4 버그 바운티 프로그램 (Bug Bounty Programs)

**버그 바운티 프로그램**은 취약점 발견을 크라우드소싱합니다:

- **플랫폼**: HackerOne, Bugcrowd, Intigriti
- **범위**: 프로그램 정책으로 정의
- **보상**: 심각도 기반 (일반적으로 $100 ~ $100,000+)
- **규칙**: 각 프로그램에 범위 내 항목에 대한 특정 규칙 있음

```python
"""
활동 계획 도우미 — 노력 및 자원 요구 사항을 계산합니다.
"""

from dataclasses import dataclass
from enum import Enum


class EngagementType(Enum):
    VULN_ASSESSMENT = "취약점 평가"
    PENTEST_EXTERNAL = "외부 침투 테스트"
    PENTEST_INTERNAL = "내부 침투 테스트"
    PENTEST_WEB_APP = "웹 애플리케이션 침투 테스트"
    RED_TEAM = "레드 팀 활동"
    PURPLE_TEAM = "퍼플 팀 훈련"


@dataclass
class EngagementEstimate:
    """보안 활동 추정치."""
    engagement_type: EngagementType
    target_count: int
    complexity: str  # "low", "medium", "high"

    @property
    def estimated_days(self) -> int:
        """유형과 범위를 기반으로 작업일을 추정합니다."""
        base_days = {
            EngagementType.VULN_ASSESSMENT: 3,
            EngagementType.PENTEST_EXTERNAL: 5,
            EngagementType.PENTEST_INTERNAL: 7,
            EngagementType.PENTEST_WEB_APP: 5,
            EngagementType.RED_TEAM: 20,
            EngagementType.PURPLE_TEAM: 10,
        }
        complexity_multiplier = {
            "low": 0.8, "medium": 1.0, "high": 1.5
        }
        base = base_days[self.engagement_type]
        # 대상 수에 따라 로그 스케일로 증가
        import math
        target_factor = 1 + math.log2(max(1, self.target_count)) * 0.3
        cmult = complexity_multiplier.get(self.complexity, 1.0)
        return max(1, round(base * target_factor * cmult))

    @property
    def team_size(self) -> int:
        """권장 팀 규모."""
        if self.engagement_type == EngagementType.RED_TEAM:
            return 3
        if self.estimated_days > 10:
            return 2
        return 1

    @property
    def report_days(self) -> int:
        """보고서 작성에 필요한 날수."""
        return max(2, self.estimated_days // 3)

    def summary(self) -> str:
        lines = [
            f"활동 유형: {self.engagement_type.value}",
            f"대상: {self.target_count}개 | 복잡도: {self.complexity}",
            f"예상 노력: {self.estimated_days}일",
            f"팀 규모: {self.team_size}명",
            f"보고서 작성: 추가 {self.report_days}일",
            f"총 일정: {self.estimated_days + self.report_days}일",
        ]
        return "\n".join(lines)


# 예시
if __name__ == "__main__":
    estimates = [
        EngagementEstimate(EngagementType.PENTEST_WEB_APP, 3, "medium"),
        EngagementEstimate(EngagementType.RED_TEAM, 50, "high"),
        EngagementEstimate(EngagementType.VULN_ASSESSMENT, 200, "low"),
    ]
    for est in estimates:
        print(est.summary())
        print("-" * 50)
```

---

## 4. 침투 테스트 방법론

### 4.1 PTES (Penetration Testing Execution Standard)

PTES는 침투 테스트의 7단계를 정의합니다:

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  1. 사전 활동     │ ──▶ │  2. 인텔리전스    │ ──▶ │  3. 위협         │
│     상호작용      │     │     수집          │     │     모델링       │
└──────────────────┘     └──────────────────┘     └──────────────────┘
         │                                                   │
         ▼                                                   ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  7. 보고          │ ◀── │  6. 사후          │ ◀── │  4. 취약점       │
│                   │     │     익스플로잇    │     │     분석         │
└──────────────────┘     └──────────────────┘     └──────────────────┘
                                  ▲                         │
                                  │                         ▼
                          ┌──────────────────┐
                          │  5. 익스플로잇    │
                          └──────────────────┘
```

**1단계: 사전 활동 상호작용**
- 범위, 교전 규칙, 허가 정의
- 긴급 연락처 및 커뮤니케이션 채널 식별
- 테스트 일정 및 마일스톤 수립

**2단계: 인텔리전스 수집**
- 수동 정찰 (OSINT, DNS, WHOIS)
- 능동 정찰 (포트 스캐닝, 서비스 열거)
- 소셜 미디어 및 직원 정보 수집

**3단계: 위협 모델링**
- 가치 있는 자산 및 데이터 식별
- 가능한 공격 벡터 결정
- 비즈니스 영향을 기반으로 대상 우선순위 지정

**4단계: 취약점 분석**
- 자동화된 스캐닝 (Nessus, OpenVAS)
- 수동 테스트 및 검증
- 오탐 제거

**5단계: 익스플로잇**
- 식별된 취약점 익스플로잇 시도
- 각 발견사항의 개념 증명(PoC) 문서화
- 모든 수행 작업을 꼼꼼히 기록

**6단계: 사후 익스플로잇**
- 침해된 시스템의 가치 결정
- 권한 상승 및 횡적 이동 시도
- 침해된 호스트에서 도달 가능한 추가 대상 식별

**7단계: 보고**
- 경영진을 위한 요약 보고
- 수정 팀을 위한 기술적 상세 사항
- 위험 등급 및 우선순위별 권고사항

### 4.2 OWASP Testing Guide

OWASP Testing Guide는 웹 애플리케이션 테스트를 위한 포괄적 프레임워크를 제공합니다:

- **정보 수집**: 기술 핑거프린팅, 콘텐츠 발견
- **설정 관리**: 기본 크리덴셜, 오류 처리, HTTP 메서드
- **ID 관리**: 사용자 등록, 계정 열거
- **인증**: 크리덴셜 테스트, 세션 관리
- **권한 부여**: 경로 탐색, 권한 상승, IDOR
- **세션 관리**: 쿠키 속성, 세션 고정, CSRF
- **입력 유효성 검사**: SQL 인젝션, XSS, 커맨드 인젝션
- **오류 처리**: 스택 트레이스, 오류 코드
- **암호화**: 약한 알고리즘, 부적절한 구현
- **비즈니스 로직**: 워크플로우 우회, 기능 남용
- **클라이언트 사이드**: DOM XSS, JavaScript 인젝션, 클릭재킹

### 4.3 OSSTMM (Open Source Security Testing Methodology Manual)

OSSTMM은 측정 가능한 결과를 가진 운영 보안 테스트에 중점을 둡니다:

- "공격 표면"을 정량적으로 정의 (RAV — Rave Attack Value)
- 5개 채널에 걸쳐 테스트: 인적, 물리적, 무선, 통신, 데이터 네트워크
- 재현성과 메트릭 강조

---

## 5. 교전 규칙

교전 규칙(RoE) 문서는 모든 보안 평가에서 가장 중요한 산출물입니다. 테스터와 클라이언트 모두에게 법적 보호를 제공합니다.

### 5.1 필수 구성 요소

모든 RoE에는 다음이 포함되어야 합니다:

1. **허가**: 권한 있는 대리인으로부터의 명시적인 서면 허가
2. **범위 정의**: 범위 내의 정확한 IP 대역, 도메인, 애플리케이션, 인원
3. **제외 사항**: 명시적으로 금지된 시스템 및 작업
4. **일정**: 시작일, 종료일, 테스트 시간
5. **허용 기법**: 허가된 방법 (예: 소셜 엔지니어링, DoS)
6. **커뮤니케이션 계획**: 중요한 발견사항을 즉시 보고하는 방법
7. **긴급 연락처**: 테스트로 예기치 않은 문제가 발생할 경우 연락처
8. **데이터 처리**: 테스트 중 발견된 민감한 데이터 보호 방법
9. **제3자 통지**: 클라우드 제공자나 호스팅 업체 통지 필요 여부
10. **법적 조항**: 책임 제한 및 면책

### 5.2 긴급 탈출 서한

허가 서한(때로 "긴급 탈출 카드"라고도 함)은:

- 공식 회사 레터헤드에 작성
- 권한 있는 사람(CISO, CTO 또는 CEO)의 서명
- 침투 테스트가 허가되었음을 명시
- 테스터의 이름과 회사 포함
- 정확한 날짜와 범위 명시
- 긴급 전화번호 포함

> **모범 사례**: 물리적 평가 시 인쇄본을 지참하세요. 원격 테스트 시에는 디지털 사본을 항상 접근 가능한 상태로 유지하세요.

---

## 6. 킬 체인과 MITRE ATT&CK

### 6.1 록히드 마틴 사이버 킬 체인

사이버 킬 체인은 표적 사이버 공격의 단계를 모델링합니다:

```
1. 정찰 ──▶ 2. 무기화 ──▶ 3. 전달
                                │
    7. 목표 달성 ◀── 6. C2 ◀── 5. 설치
                                │
                           4. 익스플로잇
```

1. **정찰**: 대상에 대한 정보 수집
2. **무기화**: 전달 가능한 페이로드 생성 (예: 트로이 문서)
3. **전달**: 페이로드 전송 (이메일, 웹, USB)
4. **익스플로잇**: 취약점 트리거
5. **설치**: 지속적 접근 설치 (백도어, RAT)
6. **명령 및 제어 (C2)**: 통신 채널 수립
7. **목표 달성**: 공격자 목표 달성 (데이터 유출, 파괴)

### 6.2 MITRE ATT&CK 프레임워크

MITRE ATT&CK(Adversarial Tactics, Techniques, and Common Knowledge)는 실제 적대자 행동의 포괄적 매트릭스를 제공합니다:

```python
"""
침투 테스트 단계에 대한 MITRE ATT&CK 전술 매핑.

일관된 보고 및 위협 에뮬레이션을 위해
침투 테스트 활동을 ATT&CK 전술에 매핑합니다.
"""

from dataclasses import dataclass


@dataclass
class ATTCKTactic:
    """MITRE ATT&CK 전술을 나타냅니다."""
    id: str
    name: str
    description: str
    pentest_phase: str
    example_techniques: list[str]


ATTCK_TACTICS = [
    ATTCKTactic(
        id="TA0043",
        name="Reconnaissance",
        description="미래 작전을 계획하기 위한 정보 수집",
        pentest_phase="인텔리전스 수집",
        example_techniques=[
            "T1595 — Active Scanning",
            "T1592 — Gather Victim Host Information",
            "T1589 — Gather Victim Identity Information",
            "T1593 — Search Open Websites/Domains",
        ],
    ),
    ATTCKTactic(
        id="TA0001",
        name="Initial Access",
        description="대상 네트워크에 초기 발판 확보",
        pentest_phase="익스플로잇",
        example_techniques=[
            "T1190 — Exploit Public-Facing Application",
            "T1566 — Phishing",
            "T1078 — Valid Accounts",
            "T1133 — External Remote Services",
        ],
    ),
    ATTCKTactic(
        id="TA0002",
        name="Execution",
        description="공격자 제어 코드 실행",
        pentest_phase="익스플로잇",
        example_techniques=[
            "T1059 — Command and Scripting Interpreter",
            "T1203 — Exploitation for Client Execution",
            "T1047 — Windows Management Instrumentation",
        ],
    ),
    ATTCKTactic(
        id="TA0003",
        name="Persistence",
        description="재시작 후에도 접근 유지",
        pentest_phase="사후 익스플로잇",
        example_techniques=[
            "T1053 — Scheduled Task/Job",
            "T1547 — Boot or Logon Autostart Execution",
            "T1136 — Create Account",
        ],
    ),
    ATTCKTactic(
        id="TA0004",
        name="Privilege Escalation",
        description="상위 권한 획득",
        pentest_phase="사후 익스플로잇",
        example_techniques=[
            "T1068 — Exploitation for Privilege Escalation",
            "T1548 — Abuse Elevation Control Mechanism",
            "T1134 — Access Token Manipulation",
        ],
    ),
    ATTCKTactic(
        id="TA0005",
        name="Defense Evasion",
        description="탐지 회피",
        pentest_phase="사후 익스플로잇",
        example_techniques=[
            "T1070 — Indicator Removal",
            "T1036 — Masquerading",
            "T1027 — Obfuscated Files or Information",
        ],
    ),
    ATTCKTactic(
        id="TA0006",
        name="Credential Access",
        description="크리덴셜 탈취",
        pentest_phase="사후 익스플로잇",
        example_techniques=[
            "T1003 — OS Credential Dumping",
            "T1110 — Brute Force",
            "T1558 — Steal or Forge Kerberos Tickets",
        ],
    ),
    ATTCKTactic(
        id="TA0007",
        name="Discovery",
        description="대상 환경 파악",
        pentest_phase="사후 익스플로잇",
        example_techniques=[
            "T1087 — Account Discovery",
            "T1046 — Network Service Discovery",
            "T1083 — File and Directory Discovery",
        ],
    ),
    ATTCKTactic(
        id="TA0008",
        name="Lateral Movement",
        description="환경 내 이동",
        pentest_phase="사후 익스플로잇",
        example_techniques=[
            "T1021 — Remote Services",
            "T1080 — Taint Shared Content",
            "T1550 — Use Alternate Authentication Material",
        ],
    ),
    ATTCKTactic(
        id="TA0010",
        name="Exfiltration",
        description="대상에서 데이터 탈취",
        pentest_phase="사후 익스플로잇",
        example_techniques=[
            "T1041 — Exfiltration Over C2 Channel",
            "T1048 — Exfiltration Over Alternative Protocol",
            "T1567 — Exfiltration Over Web Service",
        ],
    ),
]


def map_pentest_to_attck(pentest_phase: str) -> list[ATTCKTactic]:
    """침투 테스트 단계에 관련된 ATT&CK 전술을 찾습니다."""
    return [t for t in ATTCK_TACTICS if t.pentest_phase == pentest_phase]


def generate_attck_report() -> str:
    """문서화를 위한 매핑 보고서를 생성합니다."""
    lines = ["침투 테스트에 대한 MITRE ATT&CK 매핑", "=" * 50, ""]
    phases = sorted(set(t.pentest_phase for t in ATTCK_TACTICS))
    for phase in phases:
        lines.append(f"\n--- {phase} ---")
        tactics = map_pentest_to_attck(phase)
        for tactic in tactics:
            lines.append(f"  [{tactic.id}] {tactic.name}")
            for tech in tactic.example_techniques:
                lines.append(f"    - {tech}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(generate_attck_report())
```

---

## 7. 책임 있는 공개

### 7.1 공개 모델

(공식적인 활동 외에서) 취약점이 발견될 때 몇 가지 공개 방법이 있습니다:

**전체 공개**: 모든 취약점 세부 사항을 즉시 공개합니다.
- **장점**: 벤더가 빠르게 패치하도록 압박
- **단점**: 수정이 제공되기 전에 사용자를 위험에 노출

**책임 있는 공개(조정된 공개)**: 벤더에 비공개 통보, 패치를 릴리스할 합리적인 기간(일반적으로 90일) 제공 후 공개합니다.
- **장점**: 벤더 통지와 공공 투명성 균형
- **단점**: 벤더가 응답하지 않거나 무기한 지연할 수 있음

**비공개**: 취약점을 공개하지 않습니다.
- **장점**: 노출 최소화
- **단점**: 다른 연구자가 독립적으로 발견할 수 있음; 사용자 보호 안됨

### 7.2 버그 바운티 모범 사례

버그 바운티 프로그램 참여 시:

1. **정책 읽기**: 테스트 전에 범위 내 항목을 파악하세요
2. **허가 없이 프로덕션 테스트 금지**: 가능한 경우 스테이징 환경 사용
3. **영향 최소화**: 실제 데이터를 유출하지 마세요; 개념 증명 시연 사용
4. **명확한 보고서 작성**: 재현 단계, 영향 분석, 수정 제안 포함
5. **인내심 유지**: 응답 시간은 다양합니다; 공개를 레버리지로 위협하지 마세요
6. **취약점당 하나의 보고서**: 여러 문제를 하나로 묶지 마세요

---

## 8. 랩 환경 설정

### 8.1 격리된 랩 네트워크

적절한 랩 환경은 테스트를 프로덕션 네트워크로부터 격리합니다:

```
┌─────────────────────────────────────────────────┐
│                  호스트 머신                      │
│                                                  │
│  ┌──────────────┐  ┌──────────────┐            │
│  │  Kali Linux  │  │  대상 VM들   │            │
│  │  (공격자)    │  │  (피해자)    │            │
│  │              │  │              │            │
│  │  - Nmap      │  │  - Metasploit│            │
│  │  - Burp      │──│    able 2/3  │            │
│  │  - Metasploit│  │  - DVWA      │            │
│  │  - pwntools  │  │  - WebGoat   │            │
│  └──────────────┘  └──────────────┘            │
│         │                  │                     │
│         └────── NAT/호스트-전용 네트워크 ────────│
│                  (외부 접근 없음)                 │
└─────────────────────────────────────────────────┘
```

### 8.2 필수 가상 머신

| VM | 용도 | 다운로드 |
|----|------|----------|
| Kali Linux | 공격자 워크스테이션 | kali.org |
| Metasploitable 2/3 | Linux 대상 | SourceForge |
| DVWA | 웹 애플리케이션 테스트 | github.com/digininja/DVWA |
| OWASP WebGoat | 웹 보안 교육 | owasp.org |
| Vulnhub VMs | 다양한 난이도 | vulnhub.com |
| HackTheBox | 온라인 랩 | hackthebox.com |

### 8.3 자동화된 랩 설정

```python
"""
랩 환경 검증기 — 필수 도구가 설치되어 있고
네트워크가 적절히 격리되어 있는지 확인합니다.

Kali/공격자 VM에서 실행하여 랩 설정을 검증하세요.
"""

import shutil
import subprocess
import socket
from dataclasses import dataclass


@dataclass
class ToolCheck:
    """필수 도구와 그 상태를 나타냅니다."""
    name: str
    command: str
    required: bool = True
    installed: bool = False
    version: str = ""


def check_tool(tool: ToolCheck) -> ToolCheck:
    """도구가 설치되어 있는지 확인하고 버전을 가져옵니다."""
    path = shutil.which(tool.command)
    if path:
        tool.installed = True
        try:
            result = subprocess.run(
                [tool.command, "--version"],
                capture_output=True, text=True, timeout=5
            )
            version_line = (result.stdout or result.stderr).strip().split("\n")[0]
            tool.version = version_line[:80]  # 긴 버전 문자열 자르기
        except (subprocess.TimeoutExpired, FileNotFoundError):
            tool.version = "설치됨 (버전 알 수 없음)"
    return tool


def check_network_isolation() -> dict:
    """랩 네트워크가 적절히 격리되어 있는지 확인합니다."""
    results = {
        "can_resolve_dns": False,
        "can_reach_internet": False,
        "local_interfaces": [],
    }

    # DNS 해결 확인
    try:
        socket.getaddrinfo("example.com", 80, socket.AF_INET)
        results["can_resolve_dns"] = True
    except socket.gaierror:
        pass

    # 인터넷 연결 확인
    try:
        sock = socket.create_connection(("8.8.8.8", 53), timeout=3)
        sock.close()
        results["can_reach_internet"] = True
    except (socket.timeout, OSError):
        pass

    return results


def validate_lab() -> None:
    """모든 랩 검증 검사를 실행합니다."""
    tools = [
        ToolCheck("Nmap", "nmap"),
        ToolCheck("Metasploit", "msfconsole"),
        ToolCheck("Burp Suite", "burpsuite", required=False),
        ToolCheck("Python 3", "python3"),
        ToolCheck("GDB", "gdb"),
        ToolCheck("Ghidra", "ghidra", required=False),
        ToolCheck("Wireshark", "wireshark", required=False),
        ToolCheck("Gobuster", "gobuster", required=False),
        ToolCheck("ffuf", "ffuf", required=False),
        ToolCheck("SQLMap", "sqlmap"),
        ToolCheck("Hydra", "hydra"),
        ToolCheck("John the Ripper", "john"),
        ToolCheck("Hashcat", "hashcat", required=False),
        ToolCheck("Netcat", "nc"),
        ToolCheck("curl", "curl"),
    ]

    print("=" * 60)
    print("랩 환경 검증")
    print("=" * 60)

    # 도구 확인
    print("\n--- 도구 사용 가능 여부 ---")
    missing_required = []
    for tool in tools:
        check_tool(tool)
        status = "[OK]" if tool.installed else ("[없음!]" if tool.required else "[선택]")
        print(f"  {status:12s} {tool.name:20s} {tool.version}")
        if tool.required and not tool.installed:
            missing_required.append(tool.name)

    # 네트워크 확인
    print("\n--- 네트워크 격리 ---")
    network = check_network_isolation()
    if network["can_reach_internet"]:
        print("  [경고] 인터넷 접근이 감지되었습니다!")
        print("  격리된 랩을 위해 호스트 전용 네트워킹을 사용하세요.")
    else:
        print("  [OK] 인터넷 접근 없음 (적절히 격리됨)")

    # 요약
    print("\n--- 요약 ---")
    if missing_required:
        print(f"  [!] 누락된 필수 도구: {', '.join(missing_required)}")
        print("  설치: sudo apt install <도구명>")
    else:
        print("  [OK] 모든 필수 도구가 설치되어 있습니다")


if __name__ == "__main__":
    validate_lab()
```

---

## 9. 범위 지정 및 계획

### 9.1 활동 범위 정의

적절한 범위 지정은 범위 확대와 법적 문제를 방지합니다:

**네트워크 범위:**
- CIDR 표기법으로 IP 대역 정의: `10.0.0.0/24`
- 특정 호스트 포함/제외: `10.0.0.1` (포함), `10.0.0.50` (제외 — 프로덕션 DB)
- 제한된 경우 포트 범위 명시: `TCP 1-65535, UDP 상위 1000`

**애플리케이션 범위:**
- 특정 URL/도메인 목록: `https://app.example.com/*`
- 인증/비인증 테스트 정의
- 테스트할 사용자 역할 명시: 관리자, 일반 사용자, 게스트

**물리적 범위 (해당하는 경우):**
- 테스트 허가된 건물 및 층
- 소셜 엔지니어링 대상 (또는 명시적 제외)
- 배지 복제, 테일게이팅, 쓰레기통 뒤지기 허가

### 9.2 시간 추정

시간 추정의 경험 규칙:

| 평가 유형 | 소규모 (1-10개) | 중간 (10-50개) | 대규모 (50+개) |
|-----------|----------------|----------------|----------------|
| 취약점 스캔 | 1-2일 | 3-5일 | 1-2주 |
| 외부 침투 테스트 | 3-5일 | 1-2주 | 2-4주 |
| 내부 침투 테스트 | 5-7일 | 2-3주 | 3-6주 |
| 웹 앱 테스트 | 3-5일/앱 | - | - |
| 레드 팀 | 2-4주 | 4-8주 | 8-12주 |

### 9.3 납품물

전문적인 침투 테스트 보고서에는 일반적으로 다음이 포함됩니다:

1. **경영진 요약** (1-2페이지): 경영진을 위한 비즈니스 언어 개요
2. **방법론**: 도구, 기법, 접근 방법 설명
3. **발견사항**: 각 취약점의 심각도, 설명, PoC, 영향, 수정 방법
4. **위험 등급**: CVSS 점수 또는 맞춤 심각도 척도
5. **수정 로드맵**: 우선순위별 수정 권고사항
6. **부록**: 원시 스캔 출력, 스크린샷, 전체 기술적 세부 사항

---

## 10. 문서화 및 보고

### 10.1 활동 중 노트 작성

테스트 중 모든 활동을 문서화하는 것이 필수적입니다. 보고서 작성과 법적 보호를 위해 필요합니다:

```python
"""
활동 로깅 유틸리티.

테스트 중 수행된 모든 작업의 구조화된 로그를 유지합니다.
보고서 작성과 법적 보호에 필수적입니다.
"""

import json
import hashlib
from datetime import datetime
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class LogEntry:
    """활동 중 기록된 단일 작업."""
    timestamp: str
    category: str  # recon, scanning, exploitation, post-exploit
    action: str
    target: str
    result: str
    tool: str = ""
    command: str = ""
    evidence_file: str = ""
    notes: str = ""

    @staticmethod
    def now(category: str, action: str, target: str,
            result: str, **kwargs) -> "LogEntry":
        return LogEntry(
            timestamp=datetime.utcnow().isoformat() + "Z",
            category=category, action=action,
            target=target, result=result, **kwargs
        )


class EngagementLogger:
    """침투 테스트 활동을 위한 구조화된 로깅."""

    def __init__(self, engagement_name: str, log_dir: str = "./logs"):
        self.engagement_name = engagement_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"{engagement_name}.jsonl"
        self.entries: list[LogEntry] = []

    def log(self, entry: LogEntry) -> None:
        """로그 항목을 추가합니다."""
        self.entries.append(entry)
        with open(self.log_file, "a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")

    def log_action(self, category: str, action: str, target: str,
                   result: str, **kwargs) -> None:
        """작업을 기록하는 편의 메서드."""
        entry = LogEntry.now(category, action, target, result, **kwargs)
        self.log(entry)

    def get_entries_by_category(self, category: str) -> list[LogEntry]:
        return [e for e in self.entries if e.category == category]

    def generate_timeline(self) -> str:
        """사람이 읽을 수 있는 타임라인을 생성합니다."""
        lines = [
            f"활동 타임라인: {self.engagement_name}",
            "=" * 60,
        ]
        for entry in self.entries:
            lines.append(
                f"[{entry.timestamp}] [{entry.category.upper():12s}] "
                f"{entry.action} -> {entry.target}: {entry.result}"
            )
        return "\n".join(lines)

    def evidence_hash(self, filepath: str) -> str:
        """무결성을 위한 증거 파일의 SHA-256 해시를 생성합니다."""
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()


# 사용 예시
if __name__ == "__main__":
    logger = EngagementLogger("acme-pentest-2025")

    logger.log_action(
        category="recon",
        action="DNS 열거",
        target="acme.com",
        result="서브도메인 12개 발견",
        tool="subfinder",
        command="subfinder -d acme.com -o subs.txt",
    )
    logger.log_action(
        category="scanning",
        action="포트 스캔",
        target="10.0.0.0/24",
        result="12개 호스트에서 열린 포트 45개 발견",
        tool="nmap",
        command="nmap -sV -sC -oA acme_scan 10.0.0.0/24",
    )
    logger.log_action(
        category="exploitation",
        action="SQL 인젝션",
        target="https://app.acme.com/login",
        result="관리자 크리덴셜 추출 (편집됨)",
        tool="sqlmap",
        notes="username 매개변수에서 시간 기반 블라인드 SQLi",
    )

    print(logger.generate_timeline())
```

### 10.2 보고서 품질

전문적인 보고서는 원시 발견사항을 실행 가능한 인텔리전스로 변환합니다:

- **재현 가능성**: 다른 테스터가 모든 발견사항을 재현할 수 있어야 함
- **스크린샷**: 타임스탬프가 포함된 핵심 증거 스크린샷 포함
- **영향 분석**: 단순한 기술적 심각도가 아닌 비즈니스 영향 설명
- **수정 방법**: "취약점을 패치하세요"가 아닌 구체적이고 실행 가능한 수정 방법 제공
- **개념 증명**: 작동하는 PoC 코드 포함 (민감한 데이터 제거)

---

## 11. 경력 경로 및 자격증

### 11.1 주요 자격증

| 자격증 | 초점 | 난이도 |
|--------|------|--------|
| CompTIA Security+ | 기초 보안 | 입문 |
| CEH (Certified Ethical Hacker) | 윤리적 해킹 범위 | 중급 |
| OSCP (Offensive Security Certified Professional) | 실습 침투 테스트 | 고급 |
| OSWE (Offensive Security Web Expert) | 웹 애플리케이션 보안 | 고급 |
| OSEP (Offensive Security Experienced Penetration Tester) | 고급 익스플로잇 | 전문가 |
| GPEN (GIAC Penetration Tester) | 네트워크 침투 테스트 | 중급 |
| GWAPT (GIAC Web Application Penetration Tester) | 웹 앱 테스트 | 중급 |
| CRTO (Certified Red Team Operator) | 레드 팀 운영 | 고급 |
| PNPT (Practical Network Penetration Tester) | 실용적 침투 테스트 | 중급 |

### 11.2 경력 발전 경로

```
주니어 침투 테스터 ──▶ 침투 테스터 ──▶ 시니어 침투 테스터
                                              │
                                              ▼
                         레드 팀 리드 ◀── 레드 팀 오퍼레이터
                               │
                               ▼
                   보안 컨설턴트 / CISO
```

### 11.3 지속적 학습 자료

- **플랫폼**: Hack The Box, TryHackMe, PortSwigger Web Security Academy
- **컨퍼런스**: DEF CON, Black Hat, BSides, OWASP AppSec
- **커뮤니티**: r/netsec, InfoSec Twitter/Mastodon, 로컬 보안 모임
- **출판물**: Phrack, PoC||GTFO, 보안 연구자 블로그

---

## 12. 연습문제

1. **교전 규칙**: 가상의 전자상거래 회사에 대한 외부 침투 테스트를 위한 완전한 RoE 문서를 작성하세요. 모든 필수 구성 요소를 포함하세요.

2. **ATT&CK 매핑**: 공격자가 피싱 이메일을 통해 초기 접근 → 커널 익스플로잇으로 권한 상승 → DNS로 데이터 유출 시나리오에서 각 단계를 ATT&CK 전술 및 기법에 매핑하세요.

3. **랩 설정**: 격리된 네트워크에 Kali Linux VM과 Metasploitable 2 대상을 설정하세요. 랩 검증 스크립트를 사용하여 격리를 확인하세요.

4. **활동 계획**: 인증 및 비인증 기능이 있는 웹 앱 5개를 보유한 조직에 대한 그레이 박스 웹 애플리케이션 침투 테스트의 필요 노력을 추정하세요.

5. **윤리적 딜레마**: 침투 테스트 중 직원이 횡령하는 증거를 발견했습니다. 범위는 기술적 보안 테스트만 포함합니다. 어떻게 하시겠습니까? 500자 분석을 작성하세요.

6. **방법론 비교**: PTES, OWASP Testing Guide, OSSTMM을 비교하세요. 각각의 장단점 및 최적 사용 사례를 보여주는 표를 작성하세요.

---

## 13. 요약

공격 보안은 기술적 스킬과 윤리적 책임이 동등하게 요구되는 강력한 분야입니다:

- **공격 보안 마인드셋**은 창의적 공격 사고와 엄격한 윤리적 경계를 결합합니다
- **법적 프레임워크** (CFAA, CMA, GDPR)는 허용 범위를 정의합니다
- **교전 규칙**은 모든 평가에 필수적인 법적 보호입니다
- **PTES**는 구조화된 침투 테스트를 위한 7단계 방법론을 제공합니다
- **MITRE ATT&CK**은 실제 적대자 행동을 공통 분류 체계로 매핑합니다
- **책임 있는 공개**는 벤더 통지와 공공 안전을 균형 있게 조율합니다
- **랩 환경**은 프로덕션 네트워크로부터 격리되어야 합니다
- **문서화**는 보고서 작성과 법적 보호를 모두 지원합니다

이 주제의 나머지 레슨은 이 기초 위에 구축되어 정찰에서 익스플로잇, 사후 익스플로잇 및 레드 팀 운영으로 이어집니다.

---

## 14. 참고 자료

- Penetration Testing Execution Standard (PTES): http://www.pentest-standard.org/
- MITRE ATT&CK Framework: https://attack.mitre.org/
- OWASP Testing Guide v4.2: https://owasp.org/www-project-web-security-testing-guide/
- NIST SP 800-115: https://csrc.nist.gov/publications/detail/sp/800-115/final
- Computer Fraud and Abuse Act (18 U.S.C. § 1030): https://www.law.cornell.edu/uscode/text/18/1030
- EC-Council Code of Ethics: https://www.eccouncil.org/code-of-ethics/
- HackerOne Disclosure Guidelines: https://www.hackerone.com/disclosure-guidelines
- Weidman, Georgia. *Penetration Testing: A Hands-On Introduction to Hacking*. No Starch Press, 2014.
