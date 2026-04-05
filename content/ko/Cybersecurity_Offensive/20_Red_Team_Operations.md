# 레드팀 작전

**이전**: [19. CTF 방법론](./19_CTF_Methodology.md)

---

레드팀 작전(Red Team Operations)은 공격 보안의 정점을 나타낸다 — 기술적 통제뿐만 아니라 조직의 사람, 프로세스, 탐지 능력을 테스트하는 전체 범위의 적대적 시뮬레이션(Adversary Simulation)이다. 침투 테스트(Penetration Testing)와 달리 레드팀은 실제 위협 행위자를 모방하여 장기간에 걸쳐 은밀하고 창의적으로 특정 목표를 달성한다.

> **중요**: 레드팀 작전에는 광범위한 계획, 법적 허가, 전문적 실행이 필요하다. 무단 적대적 시뮬레이션을 절대 수행하지 않는다.

**난이도**: ⭐⭐⭐⭐

## 학습 목표

1. 레드팀 작전과 표준 침투 테스트의 차이를 구별한다
2. 명확한 목표와 위협 모델을 갖춘 레드팀 활동을 계획한다
3. MITRE ATT&CK를 사용하여 적대적 에뮬레이션 계획을 수립한다
4. 명령 및 제어(C2) 인프라를 구축하고 관리한다
5. 소셜 엔지니어링(Social Engineering) 캠페인을 설계하고 실행한다
6. 활동 전반에 걸쳐 작전 보안(OPSEC)을 유지한다
7. 탐지 능력 향상을 위한 퍼플 팀(Purple Team) 훈련을 수행한다
8. 실행 가능한 권고사항이 포함된 경영진 수준의 레드팀 보고서를 작성한다

---

## 목차

1. [레드팀 vs 침투 테스트](#1-레드팀-vs-침투-테스트)
2. [활동 계획 및 위협 모델링](#2-활동-계획-및-위협-모델링)
3. [레드팀을 위한 MITRE ATT&CK](#3-레드팀을-위한-mitre-attck)
4. [적대적 에뮬레이션 계획](#4-적대적-에뮬레이션-계획)
5. [명령 및 제어 인프라](#5-명령-및-제어-인프라)
6. [소셜 엔지니어링 캠페인](#6-소셜-엔지니어링-캠페인)
7. [물리적 보안 테스트](#7-물리적-보안-테스트)
8. [우회 및 OPSEC](#8-우회-및-opsec)
9. [퍼플 팀 훈련](#9-퍼플-팀-훈련)
10. [보고 및 교정](#10-보고-및-교정)
11. [연습 문제](#11-연습-문제)
12. [요약](#12-요약)
13. [참고 자료](#13-참고-자료)

---

## 1. 레드팀 vs 침투 테스트

| 측면 | 침투 테스트 | 레드팀 |
|------|-----------|--------|
| **목표** | 취약점 발견 | 탐지 및 대응 테스트 |
| **범위** | 정의된 대상 | 조직 전체 |
| **기간** | 1-4주 | 2-6개월 |
| **은밀성** | 불필요 | 필수 |
| **인지도** | 블루팀 인지 | 제한된 인지 |
| **방법론** | 체계적 스캐닝 | 적대적 에뮬레이션 |
| **소셜 엔지니어링** | 보통 제외 | 핵심 구성 요소 |
| **산출물** | 취약점 목록 | 탐지 격차 분석 |
| **팀 규모** | 1-2명 | 3-5명 이상 |

---

## 2. 활동 계획 및 위협 모델링

### 2.1 레드팀 헌장

```python
"""
Red team engagement planning framework.

Structures the planning process for adversary simulations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ObjectiveType(Enum):
    DATA_THEFT = "Exfiltrate sensitive data"
    DOMAIN_ADMIN = "Achieve domain administration"
    PHYSICAL_ACCESS = "Gain physical access to secure areas"
    BUSINESS_EMAIL = "Compromise executive email"
    FINANCIAL = "Demonstrate financial fraud capability"
    AVAILABILITY = "Demonstrate disruption capability"


@dataclass
class RedTeamEngagement:
    """Red team engagement plan."""
    client: str
    threat_profile: str  # APT group being emulated
    objectives: list[ObjectiveType]
    duration_weeks: int
    team_lead: str
    operators: list[str]
    start_date: datetime
    rules_of_engagement: str
    deconfliction_contact: str

    # Operational details
    c2_infrastructure: list[str] = field(default_factory=list)
    initial_access_vectors: list[str] = field(default_factory=list)
    opsec_requirements: list[str] = field(default_factory=list)
    known_defenses: list[str] = field(default_factory=list)

    def generate_plan(self) -> str:
        lines = [
            "RED TEAM ENGAGEMENT PLAN",
            "=" * 60,
            f"Client: {self.client}",
            f"Threat Profile: {self.threat_profile}",
            f"Duration: {self.duration_weeks} weeks",
            f"Team Lead: {self.team_lead}",
            f"Operators: {', '.join(self.operators)}",
            f"Start: {self.start_date:%Y-%m-%d}",
            f"Deconfliction: {self.deconfliction_contact}",
            "",
            "OBJECTIVES:",
        ]
        for obj in self.objectives:
            lines.append(f"  - {obj.value}")

        lines.append("\nINITIAL ACCESS VECTORS:")
        for vec in self.initial_access_vectors:
            lines.append(f"  - {vec}")

        lines.append("\nOPSEC REQUIREMENTS:")
        for req in self.opsec_requirements:
            lines.append(f"  - {req}")

        lines.append("\nKNOWN DEFENSES:")
        for defense in self.known_defenses:
            lines.append(f"  - {defense}")

        return "\n".join(lines)


if __name__ == "__main__":
    engagement = RedTeamEngagement(
        client="Example Corp",
        threat_profile="APT29 (Cozy Bear)",
        objectives=[ObjectiveType.DATA_THEFT, ObjectiveType.DOMAIN_ADMIN],
        duration_weeks=8,
        team_lead="Red Team Lead",
        operators=["Operator 1", "Operator 2", "Operator 3"],
        start_date=datetime(2025, 7, 1),
        rules_of_engagement="See signed RoE document",
        deconfliction_contact="CISO: security@example.com",
        initial_access_vectors=[
            "Spear-phishing (pre-approved targets)",
            "External service exploitation",
            "Physical access (badge cloning)",
        ],
        opsec_requirements=[
            "No automated scanning tools",
            "All traffic through redirectors",
            "No actions during business hours that could disrupt operations",
            "Immediate stop on detection by blue team (deconfliction)",
        ],
        known_defenses=[
            "CrowdStrike EDR on all endpoints",
            "Palo Alto NGFW",
            "Microsoft Defender for Identity",
            "24/7 SOC with Splunk SIEM",
        ],
    )
    print(engagement.generate_plan())
```

---

## 3. 레드팀을 위한 MITRE ATT&CK

ATT&CK를 사용하여 적대적 에뮬레이션을 구조화한다:

```
Tactic Flow:
Reconnaissance → Initial Access → Execution → Persistence
    → Privilege Escalation → Defense Evasion → Credential Access
        → Discovery → Lateral Movement → Collection
            → Command and Control → Exfiltration → Impact
```

각 전술(Tactic)에는 실제 적대적 행위에 매핑된 기법(Technique)과 하위 기법(Sub-technique)이 있다.

---

## 4. 적대적 에뮬레이션 계획

### 4.1 APT 에뮬레이션 프로세스

1. **위협 행위자 선정**: 고객의 산업 및 위협 환경에 기반하여 선정한다
2. **TTP 조사**: 해당 행위자가 사용하는 ATT&CK 기법을 매핑한다
3. **에뮬레이션 계획 수립**: 행위자를 모방하는 단계별 행동을 수립한다
4. **도구 준비**: 행위자의 능력에 맞는 도구를 선택하거나 개발한다
5. **실행**: 문서화된 편차와 함께 계획을 따라 실행한다
6. **보고**: 실행 가능한 탐지 개선을 위해 발견 사항을 ATT&CK에 매핑한다

### 4.2 MITRE CTID 적대적 에뮬레이션 라이브러리

사전 구축된 에뮬레이션 계획이 다음 그룹에 대해 제공된다:
- APT3 (Gothic Panda)
- APT29 (Cozy Bear)
- FIN6 (금융 범죄)
- Sandworm (러시아 군사 정보기관)

---

## 5. 명령 및 제어 인프라

### 5.1 C2 아키텍처

```
Operator → Team Server → Redirector → Compromised Host
                              │
                    (CDN/Cloud Front)
                              │
                    Categorized Domain
```

### 5.2 인프라 구성 요소

- **팀 서버(Team Server)**: Cobalt Strike, Sliver, Mythic
- **리다이렉터(Redirector)**: 트래픽을 필터링하는 Nginx/Apache 리버스 프록시
- **도메인**: 분류된 도메인 (에이징 완료, HTTPS 적용)
- **CDN 프론팅(CDN Fronting)**: CDN 서비스를 사용하여 C2 트래픽을 은폐한다
- **DNS C2**: 느리지만 탐지가 극히 어렵다

### 5.3 OPSEC 고려 사항

- 활동별로 인프라를 분리한다
- 활동 간에 도메인을 재사용하지 않는다
- 유효한 인증서를 갖춘 HTTPS를 사용한다
- 리다이렉터 필터링을 구현한다 (user-agent, IP 등)
- 활동 종료 후 인프라를 파기한다

---

## 6. 소셜 엔지니어링 캠페인

### 6.1 피싱 캠페인 단계

1. **OSINT**: 대상, 이메일 형식, 관심사를 식별한다
2. **구실 개발(Pretext Development)**: 믿을 만한 시나리오를 작성한다
3. **인프라**: 도메인, 이메일 서버, 랜딩 페이지를 준비한다
4. **페이로드**: 매크로 문서, 자격 증명 수집 페이지 링크를 준비한다
5. **실행**: 이메일을 발송하고 클릭을 모니터링한다
6. **후속 조치**: 획득한 접근 권한을 익스플로잇한다

### 6.2 소셜 엔지니어링 유형

| 유형 | 매체 | 목표 |
|------|------|------|
| 피싱(Phishing) | 이메일 | 자격 증명 탈취 또는 페이로드 전달 |
| 비싱(Vishing) | 전화 | 정보 수집 |
| 스미싱(Smishing) | SMS | 링크 클릭 유도 |
| 프리텍스팅(Pretexting) | 대면 | 물리적 접근 |
| 베이팅(Baiting) | USB/물리적 | 페이로드 전달 |

---

## 7. 물리적 보안 테스트

- **테일게이팅(Tailgating)**: 인가된 사람을 따라 보안 문을 통과한다
- **배지 복제(Badge Cloning)**: RFID/NFC 배지를 복사한다
- **잠금 장치 해제(Lock Picking)**: 물리적 잠금 장치를 우회한다
- **쓰레기통 뒤지기(Dumpster Diving)**: 폐기된 문서를 탐색한다
- **장치 설치(Planted Devices)**: 네트워크 접근을 위한 드롭 박스를 배치한다

---

## 8. 우회 및 OPSEC

### 8.1 작전 보안 규칙

1. **모니터링 가정**: SOC가 모든 것을 볼 수 있다고 가정하고 행동한다
2. **발자국 최소화**: 더 적은 도구, 더 적은 연결을 사용한다
3. **환경에 녹아들기**: 일반 업무 시간, 일반 프로토콜을 사용한다
4. **정리**: 각 작전 후 아티팩트를 제거한다
5. **구획화**: 대상별로 인프라를 분리한다
6. **행동 전 확인**: 행동이 범위 내에 있는지 확인한다

### 8.2 EDR 우회 기법

- 프로세스 인젝션(Process Injection) (의심스러운 부모-자식 관계 회피)
- PowerShell 실행을 위한 AMSI 우회
- 텔레메트리 방지를 위한 ETW 패칭
- 메모리 전용 실행 (파일리스)
- Syscall 스텁 (유저랜드 후킹 우회)
- 타임스탬프 위조(Timestomping) 및 로그 조작

---

## 9. 퍼플 팀 훈련

퍼플 팀(Purple Team)은 레드팀과 블루팀의 노력을 결합하여 상호 개선을 도모한다.

### 9.1 퍼플 팀 워크플로우

```
1. Red team demonstrates technique (ATT&CK mapped)
2. Blue team attempts to detect in real-time
3. Both teams discuss visibility gaps
4. Blue team creates/improves detection rules
5. Red team verifies detection works
6. Document the detection and its coverage
```

### 9.2 장점

- 적대적 레드팀만 수행하는 것보다 더 빠른 탐지 개선이 가능하다
- 공격팀과 방어팀 간의 지식 전수가 이루어진다
- ATT&CK에 대한 탐지 범위를 직접 매핑할 수 있다
- 레드팀과 블루팀을 별도로 운영하는 것보다 비용 효율적이다

---

## 10. 보고 및 교정

### 10.1 레드팀 보고서 구조

1. **경영진 요약** (2-3페이지)
   - 달성된 목표와 전반적인 위험 평가
   - 비즈니스 언어로 작성된 주요 발견 사항
   - 활동 타임라인

2. **공격 내러티브** (10-20페이지)
   - 활동의 시간 순서별 이야기
   - 스크린샷과 증거를 포함한 각 단계
   - MITRE ATT&CK에 매핑

3. **탐지 격차 분석**
   - 탐지된 것 vs 놓친 것
   - 각 단계별 탐지 소요 시간
   - 권장 탐지 개선 사항

4. **교정 로드맵**
   - 우선순위가 부여된 권고 사항
   - 빠른 성과 vs 장기 개선
   - 보안 아키텍처 권고 사항

### 10.2 지표

| 지표 | 설명 |
|------|------|
| 초기 접근 소요 시간 | 팀이 거점을 확보하는 데 걸린 시간 |
| 도메인 관리자 획득 소요 시간 | 초기 접근부터 DA까지의 소요 시간 |
| 탐지 소요 시간 | SOC가 활동을 처음 인지한 시점 |
| 격리 소요 시간 | SOC가 팀을 효과적으로 차단한 시점 |
| 달성된 목표 | 달성된 목표의 수 |
| 사용된 TTP | 사용된 ATT&CK 기법의 수 |

---

## 11. 연습 문제

1. **에뮬레이션 계획**: 금융 기관을 대상으로 하는 APT29 적대적 에뮬레이션 계획을 작성한다.
2. **C2 인프라**: 리다이렉터와 분류된 도메인을 갖춘 Sliver C2 서버를 구축한다.
3. **피싱 캠페인**: 가상의 회사를 대상으로 피싱 캠페인(구실, 이메일, 랜딩 페이지)을 설계한다.
4. **퍼플 팀**: 5개의 ATT&CK 기법을 테스트하는 퍼플 팀 훈련을 수행한다.
5. **보고서 작성**: 가상의 활동에 대한 레드팀 경영진 요약을 작성한다.
6. **OPSEC 검토**: 일련의 레드팀 행동을 검토하고 OPSEC 실패를 식별한다.

---

## 12. 요약

레드팀 작전은 조직의 완전한 보안 태세를 테스트한다:

- **레드팀**은 단순한 취약점이 아닌 탐지 및 대응 능력을 테스트한다
- **위협 모델링**과 적대적 에뮬레이션은 현실적인 테스트를 보장한다
- **MITRE ATT&CK**는 TTP의 공통 언어를 제공한다
- **C2 인프라**는 세심한 OPSEC과 설정이 필요하다
- **소셜 엔지니어링**은 종종 가장 효과적인 초기 접근 벡터이다
- **퍼플 팀**은 보안 개선을 가속한다
- **전문적 보고서**는 발견 사항을 실행 가능한 개선 사항으로 변환한다

이것으로 사이버보안 공격 기법 커리큘럼이 완료된다. 20개 레슨 전반에 걸친 기술은 인가된 보안 테스트, CTF 대회, 방어 보안 이해를 위한 포괄적인 기반을 제공한다.

---

## 13. 참고 자료

- MITRE ATT&CK: https://attack.mitre.org/
- MITRE CTID Adversary Emulation Library: https://github.com/center-for-threat-informed-defense/adversary_emulation_library
- Red Team Field Manual (RTFM)
- Cobalt Strike: https://www.cobaltstrike.com/
- Sliver: https://github.com/BishopFox/sliver
- Atomic Red Team: https://github.com/redcanaryco/atomic-red-team
- The Red Team Guide: https://redteam.guide/
