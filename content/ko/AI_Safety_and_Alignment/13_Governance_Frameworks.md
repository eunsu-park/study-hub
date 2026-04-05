# 레슨 13: 거버넌스 프레임워크 (Governance Frameworks)

[이전: 기만적 정렬](./12_Deceptive_Alignment.md) | [다음: 책임 있는 배포](./14_Responsible_Deployment.md)

---

## 학습 목표

- 주요 행위자, 프레임워크, 관할권을 포함한 글로벌 AI 거버넌스(AI governance) 환경을 매핑한다
- EU AI 법(EU AI Act)의 위험 기반 분류 체계와 규정 준수 요건을 분석한다
- AI 안전에 관한 미국 행정명령 및 연방기관 지침을 이해한다
- AI 거버넌스를 위한 국제 협력 메커니즘(GPAI, OECD, UN)을 평가한다
- 책임 있는 확장 정책(responsible scaling policies)을 포함한 프론티어 모델 거버넌스 프레임워크를 설계한다
- 내부 안전 거버넌스 구조(검토 위원회, 안전 사례)를 구현한다
- AI 규제 수단으로서의 컴퓨팅 거버넌스(compute governance)를 평가한다
- 오픈소스(open-source) 대 클로즈드소스(closed-source) 안전 논쟁과 그 거버넌스 함의를 평가한다

---

## 목차

1. [AI 거버넌스 환경](#1-ai-거버넌스-환경)
2. [EU AI 법](#2-eu-ai-법)
3. [미국 행정명령 및 정책](#3-미국-행정명령-및-정책)
4. [국제 협력](#4-국제-협력)
5. [프론티어 모델 거버넌스](#5-프론티어-모델-거버넌스)
6. [책임 있는 확장 정책](#6-책임-있는-확장-정책)
7. [내부 안전 거버넌스](#7-내부-안전-거버넌스)
8. [컴퓨팅 거버넌스](#8-컴퓨팅-거버넌스)
9. [오픈소스 대 클로즈드소스 안전](#9-오픈소스-대-클로즈드소스-안전)
10. [요약](#요약)
11. [연습문제](#연습문제)

---

## 1. AI 거버넌스 환경

### 1.1 거버넌스의 과제

AI 거버넌스(AI governance)는 근본적인 긴장에 직면해 있다: 기술은 규제보다 빠르게
발전하지만, 거버넌스 없는 AI 배포의 결과는 심각할 수 있다. 효과적인 거버넌스는
혁신과 안전, 국가 이익과 국제 협력, 속도와 철저함 사이의 균형을 맞춰야 한다.

```python
"""
Mapping the AI governance landscape: key actors, frameworks,
and their relationships.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class GovernanceLevel(Enum):
    """Levels at which AI governance operates."""
    INTERNATIONAL = "international"
    REGIONAL = "regional"   # e.g., EU
    NATIONAL = "national"
    SUB_NATIONAL = "sub_national"  # e.g., state/province
    ORGANIZATIONAL = "organizational"
    TECHNICAL = "technical"  # Standards bodies


@dataclass
class GovernanceActor:
    """An entity involved in AI governance."""
    name: str
    level: GovernanceLevel
    role: str
    key_instruments: List[str]
    binding: bool  # Whether their instruments are legally binding
    coverage: str  # Geographic or domain coverage


class GovernanceLandscape:
    """Model the global AI governance landscape."""

    def __init__(self):
        self.actors = self._initialize_actors()

    def _initialize_actors(self) -> List[GovernanceActor]:
        """Define major governance actors."""
        return [
            GovernanceActor(
                name="European Union",
                level=GovernanceLevel.REGIONAL,
                role="Comprehensive regulation",
                key_instruments=["EU AI Act (2024)", "GDPR",
                                 "Digital Services Act"],
                binding=True,
                coverage="EU member states + global via market access",
            ),
            GovernanceActor(
                name="United States (Federal)",
                level=GovernanceLevel.NATIONAL,
                role="Executive guidance + agency regulation",
                key_instruments=["Executive Order 14110 (2023)",
                                 "NIST AI RMF", "OMB Guidance"],
                binding=True,  # EO is binding on federal agencies
                coverage="US federal agencies + market influence",
            ),
            GovernanceActor(
                name="China (PRC)",
                level=GovernanceLevel.NATIONAL,
                role="Comprehensive regulation",
                key_instruments=["Interim Measures for GenAI (2023)",
                                 "Algorithm Recommendation Rules",
                                 "Deep Synthesis Regulations"],
                binding=True,
                coverage="China domestic + global influence",
            ),
            GovernanceActor(
                name="United Kingdom",
                level=GovernanceLevel.NATIONAL,
                role="Pro-innovation, sector-specific",
                key_instruments=["AI Safety Institute",
                                 "AI White Paper (2023)",
                                 "Bletchley Declaration"],
                binding=False,  # Mostly non-binding guidance
                coverage="UK + global influence via AISI",
            ),
            GovernanceActor(
                name="OECD",
                level=GovernanceLevel.INTERNATIONAL,
                role="Principles and standards coordination",
                key_instruments=["OECD AI Principles (2019, updated 2024)",
                                 "AI Policy Observatory"],
                binding=False,
                coverage="38 OECD members + adherents",
            ),
            GovernanceActor(
                name="United Nations",
                level=GovernanceLevel.INTERNATIONAL,
                role="Global coordination and human rights",
                key_instruments=["UN AI Advisory Body Report (2024)",
                                 "UNESCO AI Ethics Recommendation"],
                binding=False,
                coverage="Global",
            ),
            GovernanceActor(
                name="GPAI (Global Partnership on AI)",
                level=GovernanceLevel.INTERNATIONAL,
                role="Multi-stakeholder initiative",
                key_instruments=["Expert working groups",
                                 "Research reports"],
                binding=False,
                coverage="29 member countries",
            ),
            GovernanceActor(
                name="ISO/IEC",
                level=GovernanceLevel.TECHNICAL,
                role="Technical standards",
                key_instruments=["ISO/IEC 42001 (AI Management System)",
                                 "ISO/IEC 23894 (AI Risk Management)"],
                binding=False,  # Voluntary but influential
                coverage="Global (voluntary adoption)",
            ),
            GovernanceActor(
                name="Frontier Labs (Anthropic, OpenAI, Google DeepMind)",
                level=GovernanceLevel.ORGANIZATIONAL,
                role="Self-governance and safety commitments",
                key_instruments=["Responsible Scaling Policies",
                                 "Safety Cases", "Model Cards",
                                 "Frontier Model Forum"],
                binding=False,  # Voluntary commitments
                coverage="Global (via product deployment)",
            ),
        ]

    def print_landscape(self):
        """Print the governance landscape summary."""
        print("=== Global AI Governance Landscape ===\n")

        for level in GovernanceLevel:
            actors_at_level = [a for a in self.actors if a.level == level]
            if not actors_at_level:
                continue

            print(f"\n{'=' * 50}")
            print(f"  {level.value.upper()}")
            print(f"{'=' * 50}")

            for actor in actors_at_level:
                binding_str = "BINDING" if actor.binding else "non-binding"
                print(f"\n  {actor.name} [{binding_str}]")
                print(f"    Role: {actor.role}")
                print(f"    Coverage: {actor.coverage}")
                print(f"    Instruments: {', '.join(actor.key_instruments)}")

    def analyze_gaps(self) -> List[str]:
        """Identify governance gaps in the current landscape."""
        gaps = [
            "No binding international treaty on AI safety",
            "Open-source models fall outside most regulatory frameworks",
            "Compute governance lacks international coordination",
            "No agreed global standard for 'frontier model' definition",
            "Cross-border enforcement mechanisms are weak",
            "Military/defense AI largely exempt from civilian regulation",
            "Rapid capability advancement outpaces regulatory cycles",
            "Small/medium AI companies face disproportionate compliance burden",
        ]
        return gaps


landscape = GovernanceLandscape()
landscape.print_landscape()

print("\n\n=== Governance Gaps ===")
for gap in landscape.analyze_gaps():
    print(f"  - {gap}")
```

**글로벌 AI 거버넌스 비교**

| 차원 | EU (AI 법) | 미국 (행정명령) | 중국 (규정) | 영국 (혁신 우선) |
|------|-----------|--------------|-----------|--------------|
| 접근 방식 | 위험 기반 규제 | 부문별 지침 | 국가 주도 통제 | 원칙 기반 |
| 범위 | 위험 등급별 모든 AI 시스템 | 연방기관 + 자율 | 생성형 AI + 알고리즘 | 부문 간 프레임워크 |
| 집행 | 최대 €35M / 매출 7% 벌금 | 기관별 | 행정 처벌 | 규제기관 주도 |
| 일정 | 2024–2027 단계적 시행 | 지속적 행정 조치 | 2023년부터 활성 | 2023년부터 프레임워크 |
| 파운데이션 모델 | GPAI 의무 | 자발적 약속 | 등록 필요 | 안전 기관 테스트 |

### 1.2 중국 AI 거버넌스 접근 방식과 중소기업 준수 부담(SME Compliance Burden)

중국의 AI 규제 프레임워크는 빠른 연속으로 제정된 세 가지 중복 규정으로 구성된다: 추천 로직 공개와 중독 유발 기능 금지를 요구하는 **알고리즘 추천 규정**(2022); AI 생성 콘텐츠의 워터마킹과 서비스 제공업체의 실명 등록을 의무화하는 **딥 합성 규정**(2023); 그리고 중국 사이버공간 관리국(CAC)에 알고리즘 등록, 학습 데이터 공개, "핵심 사회주의 가치"에 부합하는 콘텐츠 필터링을 요구하는 **생성형 AI 임시 조치**(2023). 중국 접근 방식의 독특한 특징은 의무적 등록이다: 공중에 제공되는 생성형 AI 서비스는 출시 전에 CAC에 신고해야 하며, 이는 서방 프레임워크가 아직 요구하지 않는 AI 스택에 대한 규제 기관의 가시성을 제공한다.

전 세계적으로 운영되는 중소기업에게 **EU AI 법 준수 부담**은 중요한 관심사다. 독립적 분석에 따르면 고위험 시스템 준수 — 적합성 평가, 기술 문서, 품질 관리 시스템, 시판 후 모니터링 포함 — 비용은 AI 시스템당 중소기업에 €200,000~€400,000이 소요될 수 있다. 이를 완화하기 위해 법은 중소기업이 규제 감독 하에 의무 완화로 시스템을 테스트할 수 있는 규제 샌드박스(제57~63조)를 규정하고, 중소기업은 특정 GPAI 문서 요건에서 부분적으로 면제된다. 그러나 이러한 조항은 실제로는 활용도가 낮으며, 부분적으로는 샌드박스 프레임워크가 회원국에 의해 아직 구축 중이고 부분적으로는 중소기업이 면제 기준을 탐색할 법적 전문성이 부족하기 때문이다.

---

## 2. EU AI 법

EU AI 법은 잠재적 피해에 비례한 의무를 부과하기 위해 네 단계 위험 피라미드를 사용한다:

```
            ┌─────────────┐
            │ UNACCEPTABLE│  Social scoring, real-time
            │    RISK     │  biometric surveillance
            ├─────────────┤
            │             │
            │  HIGH RISK  │  Critical infrastructure, law
            │             │  enforcement, hiring, education
            ├─────────────┤
            │             │
            │  LIMITED    │  Chatbots, deepfakes
            │  RISK       │  (transparency obligations)
            ├─────────────┤
            │             │
            │  MINIMAL    │  Spam filters, games
            │  RISK       │  (no requirements)
            │             │
            └─────────────┘
```

### 2.1 위험 기반 분류

EU AI 법(Regulation 2024/1689)은 AI 시스템을 네 가지 등급으로 분류하고
각 등급에 상응하는 의무를 부과하는 위험 기반 프레임워크(risk-based framework)를 수립한다.

```python
"""
EU AI Act: risk-based classification system.

The Act categorizes AI systems by risk level and assigns
obligations accordingly. This is the world's first comprehensive
AI regulation.
"""


class EUAIActClassifier:
    """Classify AI systems according to the EU AI Act risk tiers.

    Tiers:
    1. Unacceptable Risk (PROHIBITED)
    2. High Risk (HEAVY REGULATION)
    3. Limited Risk (TRANSPARENCY OBLIGATIONS)
    4. Minimal Risk (NO SPECIFIC OBLIGATIONS)

    Plus: Special rules for General-Purpose AI (GPAI) models.
    """

    PROHIBITED_CATEGORIES = [
        {
            "category": "Social scoring by governments",
            "description": "AI systems that evaluate or classify persons "
                           "based on social behavior leading to detrimental treatment",
            "article": "Art. 5(1)(c)",
        },
        {
            "category": "Real-time remote biometric identification in public spaces",
            "description": "Use of real-time biometric systems in publicly "
                           "accessible spaces for law enforcement (with narrow exceptions)",
            "article": "Art. 5(1)(h)",
        },
        {
            "category": "Emotion recognition in workplace/education",
            "description": "AI systems that infer emotions of persons in "
                           "workplace or educational settings",
            "article": "Art. 5(1)(f)",
        },
        {
            "category": "Untargeted scraping for facial recognition",
            "description": "AI systems that create facial recognition databases "
                           "through untargeted scraping of the internet or CCTV",
            "article": "Art. 5(1)(e)",
        },
        {
            "category": "Subliminal manipulation",
            "description": "AI systems that deploy subliminal techniques "
                           "to materially distort behavior causing harm",
            "article": "Art. 5(1)(a)",
        },
    ]

    HIGH_RISK_DOMAINS = [
        {
            "domain": "Biometric identification and categorization",
            "examples": ["Remote biometric identification (non-real-time)",
                         "Biometric categorization systems"],
            "annex": "Annex III, Area 1",
        },
        {
            "domain": "Critical infrastructure management",
            "examples": ["Safety components in water/gas/electricity",
                         "Road traffic management"],
            "annex": "Annex III, Area 2",
        },
        {
            "domain": "Education and vocational training",
            "examples": ["Student admission decisions",
                         "Learning assessment", "Proctoring"],
            "annex": "Annex III, Area 3",
        },
        {
            "domain": "Employment and worker management",
            "examples": ["CV screening", "Interview evaluation",
                         "Promotion decisions", "Task allocation"],
            "annex": "Annex III, Area 4",
        },
        {
            "domain": "Access to essential services",
            "examples": ["Credit scoring", "Insurance pricing",
                         "Social benefit eligibility"],
            "annex": "Annex III, Area 5",
        },
        {
            "domain": "Law enforcement",
            "examples": ["Risk assessment for criminal profiling",
                         "Lie detection", "Evidence evaluation"],
            "annex": "Annex III, Area 6",
        },
        {
            "domain": "Migration, asylum, border control",
            "examples": ["Risk assessment for irregular migration",
                         "Document authenticity verification"],
            "annex": "Annex III, Area 7",
        },
        {
            "domain": "Administration of justice",
            "examples": ["AI assisting judicial decisions",
                         "Alternative dispute resolution"],
            "annex": "Annex III, Area 8",
        },
    ]

    GPAI_THRESHOLDS = {
        "general_purpose": {
            "threshold": "Any GPAI model",
            "obligations": [
                "Technical documentation",
                "Transparency to downstream providers",
                "Copyright policy compliance",
                "Publish training content summary",
            ],
        },
        "systemic_risk": {
            "threshold": "Cumulative compute > 10^25 FLOPs",
            "obligations": [
                "All general-purpose obligations PLUS:",
                "Model evaluation and adversarial testing",
                "Systemic risk assessment and mitigation",
                "Incident reporting to EU AI Office",
                "Adequate cybersecurity protections",
            ],
        },
    }

    def classify(self, system_description: dict) -> dict:
        """Classify an AI system under the EU AI Act."""
        # Check if prohibited
        for prohibited in self.PROHIBITED_CATEGORIES:
            if system_description.get("category") == prohibited["category"]:
                return {
                    "risk_level": "PROHIBITED",
                    "article": prohibited["article"],
                    "action": "System must not be placed on the EU market",
                }

        # Check if high-risk
        domain = system_description.get("domain", "")
        for high_risk in self.HIGH_RISK_DOMAINS:
            if domain.lower() in high_risk["domain"].lower():
                return {
                    "risk_level": "HIGH",
                    "annex": high_risk["annex"],
                    "obligations": self._high_risk_obligations(),
                }

        # Check GPAI
        if system_description.get("is_gpai", False):
            compute = system_description.get("training_compute_flops", 0)
            if compute >= 1e25:
                return {
                    "risk_level": "GPAI_SYSTEMIC",
                    "obligations": self.GPAI_THRESHOLDS["systemic_risk"]["obligations"],
                }
            return {
                "risk_level": "GPAI_GENERAL",
                "obligations": self.GPAI_THRESHOLDS["general_purpose"]["obligations"],
            }

        # Check limited risk (chatbots, deepfakes, emotion recognition)
        if system_description.get("interacts_with_humans", False):
            return {
                "risk_level": "LIMITED",
                "obligations": [
                    "Disclose AI-generated content to users",
                    "Mark AI-generated/manipulated content",
                ],
            }

        return {
            "risk_level": "MINIMAL",
            "obligations": ["No specific obligations under the AI Act"],
        }

    def _high_risk_obligations(self) -> List[str]:
        """Return obligations for high-risk AI systems."""
        return [
            "Risk management system (Art. 9)",
            "Data governance (Art. 10)",
            "Technical documentation (Art. 11)",
            "Record-keeping / logging (Art. 12)",
            "Transparency to deployers (Art. 13)",
            "Human oversight provisions (Art. 14)",
            "Accuracy, robustness, cybersecurity (Art. 15)",
            "Quality management system (Art. 17)",
            "Conformity assessment (Art. 43)",
            "EU declaration of conformity (Art. 47)",
            "CE marking (Art. 48)",
            "Post-market monitoring (Art. 72)",
        ]


# Classify example systems
classifier = EUAIActClassifier()

systems = [
    {"name": "CV Screening Bot", "domain": "Employment and worker management",
     "interacts_with_humans": True},
    {"name": "Social Credit System", "category": "Social scoring by governments"},
    {"name": "GPT-5 Equivalent", "is_gpai": True,
     "training_compute_flops": 1e26, "interacts_with_humans": True},
    {"name": "Spam Filter", "domain": "email",
     "interacts_with_humans": False},
    {"name": "Customer Chatbot", "domain": "retail",
     "interacts_with_humans": True},
]

print("=== EU AI Act Classification ===\n")
for system in systems:
    result = classifier.classify(system)
    print(f"System: {system['name']}")
    print(f"  Risk Level: {result['risk_level']}")
    if "obligations" in result:
        print(f"  Obligations ({len(result['obligations'])}):")
        for ob in result["obligations"][:5]:
            print(f"    - {ob}")
        if len(result.get("obligations", [])) > 5:
            print(f"    ... and {len(result['obligations']) - 5} more")
    print()
```

---

## 3. 미국 행정명령 및 정책

### 3.1 AI 안전에 관한 행정명령

```python
"""
US AI governance framework: Executive Order 14110
and related federal guidance.
"""


class USAIGovernance:
    """Map US AI governance instruments and requirements."""

    def __init__(self):
        self.instruments = self._define_instruments()

    def _define_instruments(self) -> Dict[str, dict]:
        """Define key US AI governance instruments."""
        return {
            "EO 14110": {
                "title": "Safe, Secure, and Trustworthy Development "
                         "and Use of AI",
                "date": "October 30, 2023",
                "status": "Active (portions subject to administration changes)",
                "key_provisions": [
                    {
                        "area": "Dual-use foundation models",
                        "requirement": "Developers of dual-use foundation models "
                                       "must notify the government and share red-team "
                                       "safety test results",
                        "threshold": "Models trained with > 10^26 FLOPs (or "
                                     "10^23 for biological sequence models)",
                        "enforced_by": "Department of Commerce / BIS",
                    },
                    {
                        "area": "AI safety standards",
                        "requirement": "NIST to develop standards, guidelines, "
                                       "and best practices for AI safety",
                        "output": "NIST AI 600-1 (GenAI Profile)",
                    },
                    {
                        "area": "Biosecurity",
                        "requirement": "Strengthen biosecurity measures against "
                                       "AI-enabled threats",
                    },
                    {
                        "area": "Critical infrastructure",
                        "requirement": "Assess AI risks to critical infrastructure "
                                       "sectors",
                    },
                    {
                        "area": "Federal AI use",
                        "requirement": "Guidelines for federal agency use of AI",
                        "output": "OMB M-24-10 guidance",
                    },
                ],
            },
            "NIST AI RMF": {
                "title": "AI Risk Management Framework 1.0",
                "date": "January 2023",
                "status": "Active",
                "key_provisions": [
                    {
                        "area": "Risk management lifecycle",
                        "functions": ["GOVERN", "MAP", "MEASURE", "MANAGE"],
                        "description": "Four core functions for managing AI risk",
                    },
                ],
            },
            "NIST AI 600-1": {
                "title": "Generative AI Profile",
                "date": "July 2024",
                "status": "Active",
                "key_provisions": [
                    {
                        "area": "GenAI-specific risks",
                        "risks_addressed": [
                            "CBRN information", "Confabulation",
                            "Data privacy", "Environmental impact",
                            "Harmful bias", "Homogenization",
                            "Information integrity", "Obscene content",
                            "Value chain risks",
                        ],
                    },
                ],
            },
        }

    def compute_threshold_check(
        self,
        model_name: str,
        training_flops: float,
        is_biological: bool = False,
    ) -> dict:
        """Check if a model meets the EO 14110 reporting threshold."""
        threshold = 1e23 if is_biological else 1e26

        meets_threshold = training_flops >= threshold

        return {
            "model": model_name,
            "training_flops": f"{training_flops:.2e}",
            "threshold": f"{threshold:.2e}",
            "is_biological": is_biological,
            "meets_reporting_threshold": meets_threshold,
            "obligations": (
                [
                    "Notify Department of Commerce",
                    "Share red-team test results with government",
                    "Report ongoing safety testing results",
                    "Report significant cybersecurity incidents",
                ]
                if meets_threshold
                else ["No federal reporting obligations under EO 14110"]
            ),
        }

    def print_framework(self):
        """Print the US governance framework summary."""
        print("=== US AI Governance Framework ===\n")
        for name, instrument in self.instruments.items():
            print(f"--- {name}: {instrument['title']} ---")
            print(f"  Date: {instrument['date']}")
            print(f"  Status: {instrument['status']}")
            for provision in instrument["key_provisions"][:3]:
                print(f"  Area: {provision['area']}")
                for key, value in provision.items():
                    if key != "area":
                        if isinstance(value, list):
                            print(f"    {key}: {', '.join(str(v) for v in value[:5])}")
                        else:
                            print(f"    {key}: {value}")
            print()


us_gov = USAIGovernance()
us_gov.print_framework()

# Check model thresholds
models = [
    ("Small LLM", 1e22, False),
    ("Medium LLM", 1e24, False),
    ("Large LLM (GPT-4 class)", 1e25, False),
    ("Frontier LLM", 1e26, False),
    ("Bio sequence model", 1e22, True),
    ("Large bio model", 1e24, True),
]

print("\n=== EO 14110 Threshold Checks ===\n")
for name, flops, is_bio in models:
    result = us_gov.compute_threshold_check(name, flops, is_bio)
    status = "MEETS" if result["meets_reporting_threshold"] else "BELOW"
    print(f"[{status}] {name}: {result['training_flops']} FLOPs "
          f"(threshold: {result['threshold']})")
```

---

## 4. 국제 협력

### 4.1 다자간 프레임워크

```python
"""
International AI governance cooperation: frameworks,
agreements, and coordination mechanisms.
"""


class InternationalAICooperation:
    """Map international AI cooperation mechanisms."""

    def __init__(self):
        self.frameworks = self._define_frameworks()

    def _define_frameworks(self) -> List[dict]:
        """Define major international cooperation frameworks."""
        return [
            {
                "name": "Bletchley Declaration (AI Safety Summit)",
                "date": "November 2023",
                "signatories": 28,
                "key_commitments": [
                    "Acknowledge risks from frontier AI",
                    "International cooperation on AI safety",
                    "Support AI safety research",
                    "Develop risk-based policies",
                ],
                "follow_up": "Seoul AI Safety Summit (2024), Paris AI Action Summit (2025)",
                "strength": "Political signal; non-binding",
            },
            {
                "name": "OECD AI Principles",
                "date": "May 2019, updated 2024",
                "signatories": 46,  # 38 OECD + 8 adherent countries
                "key_commitments": [
                    "Inclusive growth, sustainable development",
                    "Human-centered values and fairness",
                    "Transparency and explainability",
                    "Robustness, security, and safety",
                    "Accountability",
                ],
                "follow_up": "AI Policy Observatory, regular monitoring",
                "strength": "Widely adopted but non-binding",
            },
            {
                "name": "G7 Hiroshima AI Process",
                "date": "2023",
                "signatories": 7,
                "key_commitments": [
                    "International Code of Conduct for AI developers",
                    "Guiding principles for all AI actors",
                    "Focus on advanced AI systems",
                ],
                "follow_up": "Integrated into G7 2024 agenda",
                "strength": "G7 influence but limited membership",
            },
            {
                "name": "UN Advisory Body on AI",
                "date": "2023-2024",
                "signatories": 193,  # UN member states
                "key_commitments": [
                    "Governing AI for Humanity report (2024)",
                    "Proposed international AI governance body",
                    "Focus on Global South inclusion",
                    "Digital public goods for AI",
                ],
                "follow_up": "Global Digital Compact",
                "strength": "Universal membership but slow consensus",
            },
            {
                "name": "AI Safety Institutes Network",
                "date": "2024",
                "signatories": 10,  # US, UK, Japan, Canada, EU, etc.
                "key_commitments": [
                    "Share safety evaluation results",
                    "Coordinate pre-deployment testing",
                    "Develop shared methodologies",
                    "Joint research programs",
                ],
                "follow_up": "Expanding membership, joint evaluations",
                "strength": "Technical focus, growing operational coordination",
            },
        ]

    def compare_approaches(self) -> dict:
        """Compare governance approaches across major jurisdictions."""
        approaches = {
            "EU": {
                "style": "Comprehensive, risk-based regulation",
                "speed": "Slow (years of legislative process)",
                "coverage": "Broad (all AI systems by risk level)",
                "enforcement": "Strong (fines up to 7% global revenue)",
                "innovation_impact": "Potentially restrictive for startups",
                "global_influence": "Brussels Effect (market access rules)",
            },
            "US": {
                "style": "Sector-specific, executive-led",
                "speed": "Medium (executive orders faster than legislation)",
                "coverage": "Focused (frontier models + federal use)",
                "enforcement": "Medium (relies on existing agency authority)",
                "innovation_impact": "More permissive for industry",
                "global_influence": "Market dominance of US AI companies",
            },
            "UK": {
                "style": "Pro-innovation, principles-based",
                "speed": "Fast (non-legislative approach)",
                "coverage": "Narrow (relies on existing sector regulators)",
                "enforcement": "Weak (no dedicated AI regulator)",
                "innovation_impact": "Most permissive",
                "global_influence": "AI Safety Institute leadership",
            },
            "China": {
                "style": "Comprehensive, technology-specific",
                "speed": "Fast (can regulate quickly)",
                "coverage": "Broad + technology-specific rules",
                "enforcement": "Strong (state enforcement capacity)",
                "innovation_impact": "Controlled innovation corridors",
                "global_influence": "Growing through Belt and Road",
            },
        }
        return approaches


cooperation = InternationalAICooperation()

print("=== International AI Cooperation Frameworks ===\n")
for fw in cooperation.frameworks:
    print(f"--- {fw['name']} ({fw['date']}) ---")
    print(f"  Signatories: {fw['signatories']}")
    print(f"  Strength: {fw['strength']}")
    print(f"  Key commitments: {', '.join(fw['key_commitments'][:3])}")
    print()

print("\n=== Governance Approach Comparison ===\n")
approaches = cooperation.compare_approaches()
dimensions = ["style", "enforcement", "innovation_impact", "global_influence"]
for dim in dimensions:
    print(f"\n{dim.upper()}:")
    for country, details in approaches.items():
        print(f"  {country:6s}: {details[dim]}")
```

---

## 5. 프론티어 모델 거버넌스

### 5.1 프론티어 모델 정의

```python
"""
Frontier model governance: frameworks for governing the
most capable AI systems.
"""


@dataclass
class FrontierModelCriteria:
    """Criteria for classifying a model as 'frontier'."""
    name: str
    compute_threshold: float  # FLOPs
    capability_threshold: str
    risk_domains: List[str]


class FrontierModelGovernance:
    """Governance framework for frontier AI models.

    Frontier models are defined by their exceptional capabilities
    and the potential for novel risks that emerge with scale.
    """

    def __init__(self):
        self.criteria = self._define_criteria()
        self.governance_elements = self._define_governance()

    def _define_criteria(self) -> List[FrontierModelCriteria]:
        """Define what makes a model 'frontier'."""
        return [
            FrontierModelCriteria(
                name="EU AI Act (Systemic Risk)",
                compute_threshold=1e25,
                capability_threshold="Presumed systemic risk above compute threshold",
                risk_domains=["Systemic risks to public interest"],
            ),
            FrontierModelCriteria(
                name="EO 14110 (Dual-Use Foundation)",
                compute_threshold=1e26,
                capability_threshold="Dual-use capabilities posing national security risk",
                risk_domains=["CBRN", "Cyber offense", "Model autonomy"],
            ),
            FrontierModelCriteria(
                name="Anthropic RSP",
                compute_threshold=0,  # Capability-based, not compute
                capability_threshold="ASL-3+: models with meaningful uplift "
                                     "in CBRN or cyber capabilities",
                risk_domains=["CBRN", "Autonomous AI", "Cyber operations"],
            ),
        ]

    def _define_governance(self) -> dict:
        """Define governance elements for frontier models."""
        return {
            "pre_training": {
                "safety_case": "Document expected capabilities and risks",
                "compute_reporting": "Report training compute to regulators",
                "data_governance": "Audit training data for harmful content",
            },
            "during_training": {
                "checkpoint_evaluation": "Evaluate safety at regular intervals",
                "anomaly_detection": "Monitor for unexpected capabilities",
                "security": "Protect model weights from theft/exfiltration",
            },
            "pre_deployment": {
                "red_teaming": "Comprehensive adversarial testing",
                "capability_evaluation": "Benchmark dangerous capabilities",
                "external_review": "Independent safety audit",
                "safety_case_review": "Review safety case with evidence",
            },
            "post_deployment": {
                "monitoring": "Track real-world incidents and misuse",
                "incident_response": "Rapid response to safety issues",
                "model_updates": "Safety patches and updates",
                "deprecation_plan": "Plan for model retirement",
            },
        }

    def evaluate_model(self, model_info: dict) -> dict:
        """Evaluate a model against frontier criteria."""
        results = []
        for criteria in self.criteria:
            compute_meets = (
                model_info.get("compute_flops", 0) >= criteria.compute_threshold
                if criteria.compute_threshold > 0
                else None  # Capability-based
            )

            results.append({
                "framework": criteria.name,
                "compute_threshold_met": compute_meets,
                "risk_domains": criteria.risk_domains,
            })

        is_frontier = any(
            r["compute_threshold_met"] for r in results
            if r["compute_threshold_met"] is not None
        )

        return {
            "model": model_info.get("name", "Unknown"),
            "is_frontier": is_frontier,
            "framework_results": results,
            "required_governance": (
                self.governance_elements if is_frontier
                else {"minimal": "Standard safety practices recommended"}
            ),
        }


governance = FrontierModelGovernance()

test_models = [
    {"name": "Small Research Model", "compute_flops": 1e22},
    {"name": "Production LLM", "compute_flops": 1e24},
    {"name": "Frontier LLM", "compute_flops": 1e26},
]

print("=== Frontier Model Classification ===\n")
for model in test_models:
    result = governance.evaluate_model(model)
    status = "FRONTIER" if result["is_frontier"] else "Non-frontier"
    print(f"[{status}] {result['model']} ({model['compute_flops']:.0e} FLOPs)")
    for fw_result in result["framework_results"]:
        if fw_result["compute_threshold_met"] is not None:
            met = "YES" if fw_result["compute_threshold_met"] else "no"
            print(f"  {fw_result['framework']}: {met}")
    print()
```

---

## 6. 책임 있는 확장 정책

### 6.1 Anthropic의 RSP 프레임워크

```python
"""
Responsible Scaling Policies (RSPs): frameworks that tie
AI capability levels to required safety measures.
"""


class ResponsibleScalingPolicy:
    """Model a Responsible Scaling Policy framework.

    RSPs define capability thresholds (AI Safety Levels) and
    the safety measures required at each level.
    """

    def __init__(self):
        self.levels = self._define_levels()

    def _define_levels(self) -> Dict[str, dict]:
        """Define AI Safety Levels (inspired by Anthropic's RSP)."""
        return {
            "ASL-1": {
                "description": "Systems that pose no meaningful uplift "
                               "in catastrophic risk",
                "capability_threshold": "Below all dangerous capability thresholds",
                "examples": "Simple classifiers, basic chatbots",
                "required_safeguards": [
                    "Standard software security practices",
                    "Basic content filtering",
                ],
                "deployment_allowed": True,
            },
            "ASL-2": {
                "description": "Current generation of frontier LLMs. "
                               "Some dangerous knowledge but no meaningful "
                               "uplift over search engines",
                "capability_threshold": "Can discuss dangerous topics but "
                                        "provides no meaningful uplift "
                                        "over publicly available information",
                "examples": "GPT-4, Claude 3, Gemini (as of 2024)",
                "required_safeguards": [
                    "RLHF/CAI alignment training",
                    "Red-teaming before deployment",
                    "Content moderation systems",
                    "Usage policy enforcement",
                    "Basic security for model weights",
                ],
                "deployment_allowed": True,
            },
            "ASL-3": {
                "description": "Models providing meaningful uplift for "
                               "creating CBRN weapons or autonomous "
                               "cyber-attacks",
                "capability_threshold": "Meaningfully increases risk of "
                                        "catastrophic misuse compared to "
                                        "existing resources",
                "examples": "Hypothetical next-generation models",
                "required_safeguards": [
                    "All ASL-2 safeguards PLUS:",
                    "Hardened security for model weights",
                    "Enhanced misuse monitoring",
                    "Capability-specific access controls",
                    "Internal deployment review board",
                    "Incident response plan",
                    "Regular third-party audits",
                ],
                "deployment_allowed": True,  # With safeguards
            },
            "ASL-4": {
                "description": "Models that could contribute to "
                               "catastrophic or existential risk "
                               "if misaligned or misused",
                "capability_threshold": "Potential for catastrophic harm "
                                        "even with alignment measures",
                "examples": "Hypothetical highly autonomous systems",
                "required_safeguards": [
                    "All ASL-3 safeguards PLUS:",
                    "Formal safety proofs where possible",
                    "Extreme security measures",
                    "Government notification/review",
                    "Alignment guarantees beyond current state-of-art",
                    "Potentially: deployment restrictions",
                ],
                "deployment_allowed": False,  # Until safeguards exist
            },
        }

    def assess_model(self, capabilities: dict) -> dict:
        """Assess which ASL level a model falls into."""
        level = "ASL-1"

        if capabilities.get("language_generation", False):
            level = "ASL-2"

        if capabilities.get("cbrn_uplift", False):
            level = "ASL-3"

        if capabilities.get("autonomous_replication", False):
            level = "ASL-4"

        level_info = self.levels[level]

        return {
            "assessed_level": level,
            "description": level_info["description"],
            "required_safeguards": level_info["required_safeguards"],
            "deployment_allowed": level_info["deployment_allowed"],
            "safeguards_needed": len(level_info["required_safeguards"]),
        }


rsp = ResponsibleScalingPolicy()

print("=== Responsible Scaling Policy: AI Safety Levels ===\n")
for level, info in rsp.levels.items():
    deploy = "YES" if info["deployment_allowed"] else "NO"
    print(f"--- {level} (Deploy: {deploy}) ---")
    print(f"  {info['description']}")
    print(f"  Examples: {info['examples']}")
    print(f"  Safeguards ({len(info['required_safeguards'])}):")
    for sg in info["required_safeguards"][:4]:
        print(f"    - {sg}")
    print()
```

---

## 7. 내부 안전 거버넌스

### 7.1 안전 검토 위원회 및 프로세스

```python
"""
Internal safety governance: organizational structures and
processes for ensuring AI safety within companies.
"""


class SafetyGovernanceStructure:
    """Model an organization's internal AI safety governance."""

    def __init__(self, org_name: str):
        self.org_name = org_name
        self.review_board = None
        self.policies = []
        self.review_log = []

    def establish_review_board(self, config: dict):
        """Establish a safety review board."""
        self.review_board = {
            "name": config.get("name", "AI Safety Review Board"),
            "composition": config.get("composition", [
                "Chief Safety Officer (chair)",
                "Head of Research",
                "Head of Engineering",
                "External safety researcher",
                "Ethics advisor",
                "Legal counsel",
            ]),
            "authority": config.get("authority", [
                "Can halt model deployments",
                "Can require additional safety testing",
                "Can mandate capability restrictions",
                "Reports directly to CEO/Board",
            ]),
            "meeting_frequency": config.get("frequency", "Weekly"),
            "decision_process": config.get("process", "Consensus with chair tiebreak"),
        }

    def add_policy(self, policy: dict):
        """Add a safety policy."""
        self.policies.append(policy)

    def conduct_review(self, model_info: dict) -> dict:
        """Conduct a safety review for a model deployment."""
        if not self.review_board:
            return {"error": "No review board established"}

        # Evaluate against each policy
        policy_results = []
        for policy in self.policies:
            met = self._check_policy(model_info, policy)
            policy_results.append({
                "policy": policy["name"],
                "met": met,
                "required_for_deploy": policy.get("blocking", True),
            })

        blocking_failures = [
            r for r in policy_results
            if not r["met"] and r["required_for_deploy"]
        ]

        decision = "APPROVE" if not blocking_failures else "REJECT"

        review_record = {
            "model": model_info.get("name"),
            "decision": decision,
            "policy_results": policy_results,
            "blocking_failures": len(blocking_failures),
            "conditions": (
                [] if decision == "APPROVE"
                else [f"Must satisfy: {r['policy']}" for r in blocking_failures]
            ),
        }
        self.review_log.append(review_record)
        return review_record

    def _check_policy(self, model_info: dict, policy: dict) -> bool:
        """Check if a model meets a specific policy requirement."""
        requirement = policy.get("requirement", "")

        if "red_team" in requirement.lower():
            return model_info.get("red_teamed", False)
        if "eval" in requirement.lower():
            return model_info.get("safety_eval_complete", False)
        if "documentation" in requirement.lower():
            return model_info.get("has_model_card", False)
        if "monitoring" in requirement.lower():
            return model_info.get("monitoring_configured", False)

        return True


# Build a safety governance structure
org = SafetyGovernanceStructure("ExampleAI Corp")

org.establish_review_board({
    "name": "AI Safety Review Board",
    "frequency": "Bi-weekly + ad-hoc for launches",
})

org.add_policy({"name": "Red-teaming Complete",
                "requirement": "Red team testing required",
                "blocking": True})
org.add_policy({"name": "Safety Evaluation",
                "requirement": "Safety eval benchmarks passed",
                "blocking": True})
org.add_policy({"name": "Model Card",
                "requirement": "Documentation complete",
                "blocking": True})
org.add_policy({"name": "Monitoring",
                "requirement": "Monitoring system configured",
                "blocking": True})

# Review a model
model = {
    "name": "ExampleLLM v2",
    "red_teamed": True,
    "safety_eval_complete": True,
    "has_model_card": False,  # Missing!
    "monitoring_configured": True,
}

print("=== Internal Safety Review ===\n")
print(f"Review Board: {org.review_board['name']}")
print(f"Composition: {', '.join(org.review_board['composition'][:3])}...")
print(f"Authority: {', '.join(org.review_board['authority'][:2])}...")

result = org.conduct_review(model)
print(f"\nModel: {result['model']}")
print(f"Decision: {result['decision']}")
for pr in result["policy_results"]:
    status = "PASS" if pr["met"] else "FAIL"
    blocking = " [BLOCKING]" if pr["required_for_deploy"] and not pr["met"] else ""
    print(f"  [{status}] {pr['policy']}{blocking}")
if result["conditions"]:
    print(f"\nConditions for approval:")
    for c in result["conditions"]:
        print(f"  - {c}")
```

---

## 8. 컴퓨팅 거버넌스

### 8.1 거버넌스 수단으로서의 컴퓨팅

```python
"""
Compute governance: using computational resources as a
lever for AI regulation.

Key insight: unlike algorithms and data, compute is:
1. Physical (can be tracked)
2. Concentrated (few manufacturers)
3. Quantifiable (FLOPs, GPU-hours)
4. Already export-controlled (chips)
"""


class ComputeGovernance:
    """Framework for compute-based AI governance."""

    def __init__(self):
        self.mechanisms = self._define_mechanisms()

    def _define_mechanisms(self) -> Dict[str, dict]:
        """Define compute governance mechanisms."""
        return {
            "Export Controls": {
                "description": "Restrict sale of advanced AI chips to certain countries",
                "current_examples": [
                    "US chip export controls (October 2022, updated 2023)",
                    "Restrictions on A100/H100 GPUs to China",
                    "Netherlands ASML export restrictions",
                ],
                "effectiveness": "Medium - workarounds exist but raise costs",
                "risks": ["Accelerates domestic chip development in restricted countries",
                          "Harms US/EU chip companies commercially",
                          "Difficult to enforce for cloud access"],
            },
            "KYC for Cloud Compute": {
                "description": "Know-Your-Customer rules for cloud GPU providers",
                "current_examples": [
                    "EO 14110 requires cloud providers to report foreign customers",
                    "Proposed: compute usage reporting thresholds",
                ],
                "effectiveness": "Low-Medium - VPN and reseller workarounds",
                "risks": ["Privacy concerns", "Disproportionate burden on small users",
                          "Jurisdiction arbitrage"],
            },
            "Compute Thresholds": {
                "description": "Define regulatory obligations based on training compute",
                "current_examples": [
                    "EU AI Act: 10^25 FLOPs for GPAI with systemic risk",
                    "EO 14110: 10^26 FLOPs for dual-use foundation models",
                ],
                "effectiveness": "Medium - clear threshold but may become outdated",
                "risks": ["Algorithmic efficiency improvements bypass thresholds",
                          "Threshold gaming (train at 9.99 * 10^25)",
                          "Compute measurement is ambiguous"],
            },
            "Compute Monitoring": {
                "description": "Track large-scale compute usage for AI training",
                "current_examples": [
                    "Proposed: international compute monitoring body",
                    "Analogous to nuclear material accounting (IAEA)",
                ],
                "effectiveness": "Potentially high if implemented globally",
                "risks": ["Requires international cooperation",
                          "Surveillance concerns",
                          "Distinguishing training from inference"],
            },
        }

    def threshold_analysis(self) -> dict:
        """Analyze how compute thresholds may become outdated."""
        analysis = {
            "algorithmic_progress": {
                "trend": "~2x efficiency improvement per year",
                "implication": "A model requiring 10^26 FLOPs today may need "
                               "only 10^24 FLOPs in 3-4 years",
                "recommendation": "Regularly update thresholds; supplement with "
                                  "capability-based criteria",
            },
            "hardware_progress": {
                "trend": "New chip generations every 1-2 years",
                "implication": "Training cost drops, making frontier compute "
                               "accessible to more actors",
                "recommendation": "Consider cost-based thresholds alongside "
                                  "absolute compute",
            },
            "distributed_training": {
                "trend": "Techniques for training across many smaller clusters",
                "implication": "Single-cluster monitoring becomes less effective",
                "recommendation": "Monitor aggregate compute usage, not just "
                                  "individual training runs",
            },
        }
        return analysis


compute_gov = ComputeGovernance()

print("=== Compute Governance Mechanisms ===\n")
for name, mechanism in compute_gov.mechanisms.items():
    print(f"--- {name} ---")
    print(f"  {mechanism['description']}")
    print(f"  Effectiveness: {mechanism['effectiveness']}")
    print(f"  Examples: {mechanism['current_examples'][0]}")
    print(f"  Risks: {mechanism['risks'][0]}")
    print()

threshold = compute_gov.threshold_analysis()
print("=== Threshold Obsolescence Analysis ===\n")
for factor, details in threshold.items():
    print(f"{factor}:")
    print(f"  Trend: {details['trend']}")
    print(f"  Implication: {details['implication']}")
    print()
```

---

## 9. 오픈소스 대 클로즈드소스 안전

### 9.1 논쟁

```python
"""
Open-source vs closed-source AI safety debate.
Both sides have legitimate arguments, and governance
frameworks must account for both approaches.
"""


def open_vs_closed_debate() -> dict:
    """Present both sides of the open-source AI safety debate."""

    debate = {
        "Arguments for Open-Source Safety": {
            "Transparency enables scrutiny": (
                "Open weights and code allow independent safety researchers "
                "to audit models, find vulnerabilities, and verify safety claims. "
                "Security through obscurity is not a reliable strategy."
            ),
            "Democratization prevents concentration": (
                "Concentrating AI capabilities in a few companies creates "
                "single points of failure and power concentration risks. "
                "Open models distribute both capabilities and responsibility."
            ),
            "Reproducibility enables science": (
                "Safety research requires reproducible experiments. "
                "Open models allow the research community to verify results, "
                "build on findings, and accelerate safety progress."
            ),
            "Defense benefits from openness": (
                "Defenders need access to models to develop guardrails, "
                "red-teaming tools, and safety evaluations. "
                "Open models accelerate defensive capabilities."
            ),
            "Historical precedent": (
                "Open-source software has historically been MORE secure "
                "than closed-source (Linus's Law). The same may apply "
                "to AI safety."
            ),
        },
        "Arguments for Closed-Source Safety": {
            "Irrevocable release": (
                "Once model weights are released, they cannot be recalled. "
                "Unlike software bugs, released capabilities cannot be patched "
                "in deployed copies."
            ),
            "Misuse by adversaries": (
                "Open models can be fine-tuned to remove safety guardrails. "
                "This is significantly easier than training from scratch. "
                "Adversaries benefit disproportionately from open release."
            ),
            "Dual-use capabilities scale": (
                "As models become more capable, the risk from open release "
                "grows. A model that provides meaningful CBRN uplift should "
                "not be openly available regardless of benefits."
            ),
            "Access != openness": (
                "Structured access (API with safety filters) provides "
                "most benefits of openness without the risks of "
                "unrestricted weight release."
            ),
            "Liability and accountability": (
                "Closed models have clear operators who can be held "
                "accountable. Open models create diffuse responsibility "
                "with no entity to hold accountable for misuse."
            ),
        },
        "Synthesis / Middle Ground": {
            "Structured access": (
                "Provide researcher access to weights under agreements, "
                "while maintaining deployment safeguards. "
                "Examples: BLOOM's RAIL license, Llama's community license."
            ),
            "Graduated openness": (
                "Release less capable models openly; restrict more capable "
                "ones. The openness level should scale inversely with "
                "dangerous capability level."
            ),
            "Open safety, closed capabilities": (
                "Open-source safety tools, evaluation frameworks, and "
                "red-teaming techniques, even if the most capable models "
                "themselves remain closed."
            ),
        },
    }

    print("=== Open-Source vs Closed-Source AI Safety ===\n")
    for position, arguments in debate.items():
        print(f"{'=' * 50}")
        print(f"  {position}")
        print(f"{'=' * 50}")
        for arg_name, arg_detail in arguments.items():
            print(f"\n  {arg_name}:")
            # Wrap text for readability
            words = arg_detail.split()
            line = "    "
            for word in words:
                if len(line) + len(word) > 75:
                    print(line)
                    line = "    "
                line += word + " "
            if line.strip():
                print(line)
        print()

    return debate


open_vs_closed_debate()
```

---

## 요약

- **AI 거버넌스 환경(AI governance landscape)**은 국제, 지역, 국가, 조직, 기술 수준의 행위자를 포함하며, 국경 간 조율과 집행에서 상당한 격차가 존재한다
- **EU AI 법(EU AI Act)**은 위험 기반 분류(금지, 고위험, 제한적, 최소)를 수립하며 각 수준에 특정 의무를 부과하고, 체계적 위험(systemic risk) 지정을 위해 10^25 FLOP 임계값을 사용하는 GPAI 규칙을 포함한다
- **미국 거버넌스(US governance)**는 EO 14110(프론티어 모델에 대한 보고 임계값), NIST AI RMF, 분야별 기관 지침을 중심으로 한다
- **국제 협력(international cooperation)**에는 블레칠리 선언(Bletchley Declaration), OECD AI 원칙, G7 히로시마 프로세스(Hiroshima Process), AI 안전 연구소 네트워크(AI Safety Institutes Network)가 포함된다
- **프론티어 모델 거버넌스(frontier model governance)**는 역량 임계값과 안전 요건을 정의하며, Anthropic의 RSP와 같은 프레임워크가 AI 안전 수준(AI Safety Levels)을 필요한 안전 장치에 연결한다
- **책임 있는 확장 정책(Responsible Scaling Policies, RSPs)**은 ASL-1(최소 위험)부터 ASL-4(잠재적 재앙적 위험)까지의 단계를 만들며, 각 수준에서 점점 강화되는 안전 요건을 부과한다
- **내부 안전 거버넌스(internal safety governance)**에는 배포를 중단할 권한을 가진 안전 검토 위원회, 정책 프레임워크, 구조화된 검토 프로세스가 포함된다
- **컴퓨팅 거버넌스(compute governance)**는 수출 통제, KYC 규칙, 학습 컴퓨팅 임계값을 통해 컴퓨팅의 물리적이고 집중된 특성을 활용한다
- **오픈소스 대 클로즈드소스(open-source vs closed-source)** 논쟁은 투명성/감사(transparency/scrutiny)와 비가역적 배포/악용 위험(irrevocable release/misuse risks) 사이의 진정한 트레이드오프를 포함한다

---

## 연습문제

### 연습문제 1: EU AI 법 분류기

상세한 EU AI 법 위험 분류기를 구축하라:
1. 전체 조항 참조와 함께 네 가지 위험 등급을 모두 구현하라
2. 컴퓨팅 기반 및 지정 기반 경로를 포함한 GPAI 분류를 추가하라
3. 10개의 실제 AI 시스템(설명 제공)을 올바른 등급으로 분류하라
4. 각 고위험 분류에 대해 구체적인 규정 준수 의무를 나열하라
5. 예상 규정 준수 비용 등급(low/medium/high/very high)을 계산하라

<details>
<summary>정답 보기</summary>

```python
"""
Detailed EU AI Act risk classifier with compliance cost estimation.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class AISystemDescription:
    """Description of an AI system for classification."""
    name: str
    domain: str
    use_case: str
    interacts_with_humans: bool
    is_gpai: bool
    training_compute: float  # FLOPs
    generates_content: bool


class DetailedEUAIActClassifier:
    """Full EU AI Act classifier with compliance cost estimation."""

    COMPLIANCE_COSTS = {
        "PROHIBITED": "N/A (system cannot be deployed)",
        "HIGH": {
            "initial": "EUR 200,000 - 2,000,000",
            "annual": "EUR 50,000 - 500,000",
            "tier": "Very High",
        },
        "GPAI_SYSTEMIC": {
            "initial": "EUR 500,000 - 5,000,000",
            "annual": "EUR 200,000 - 2,000,000",
            "tier": "Very High",
        },
        "GPAI_GENERAL": {
            "initial": "EUR 50,000 - 500,000",
            "annual": "EUR 20,000 - 100,000",
            "tier": "High",
        },
        "LIMITED": {
            "initial": "EUR 10,000 - 50,000",
            "annual": "EUR 5,000 - 20,000",
            "tier": "Low",
        },
        "MINIMAL": {
            "initial": "EUR 0 - 10,000 (voluntary)",
            "annual": "EUR 0 - 5,000",
            "tier": "Minimal",
        },
    }

    HIGH_RISK_DOMAINS = {
        "biometric": "Annex III, Area 1",
        "critical infrastructure": "Annex III, Area 2",
        "education": "Annex III, Area 3",
        "employment": "Annex III, Area 4",
        "essential services": "Annex III, Area 5",
        "law enforcement": "Annex III, Area 6",
        "migration": "Annex III, Area 7",
        "justice": "Annex III, Area 8",
    }

    def classify(self, system: AISystemDescription) -> dict:
        """Classify system and estimate compliance costs."""
        # Check high-risk domains
        for domain_key, annex in self.HIGH_RISK_DOMAINS.items():
            if domain_key in system.domain.lower():
                obligations = [
                    "Risk management system (Art. 9)",
                    "Data governance (Art. 10)",
                    "Technical documentation (Art. 11)",
                    "Record-keeping (Art. 12)",
                    "Transparency to deployers (Art. 13)",
                    "Human oversight (Art. 14)",
                    "Accuracy/robustness/security (Art. 15)",
                    "Quality management (Art. 17)",
                    "Conformity assessment (Art. 43)",
                    "CE marking (Art. 48)",
                    "Post-market monitoring (Art. 72)",
                ]
                return {
                    "system": system.name,
                    "risk_level": "HIGH",
                    "annex_reference": annex,
                    "obligations": obligations,
                    "compliance_cost": self.COMPLIANCE_COSTS["HIGH"],
                }

        # Check GPAI
        if system.is_gpai:
            if system.training_compute >= 1e25:
                level = "GPAI_SYSTEMIC"
            else:
                level = "GPAI_GENERAL"
            return {
                "system": system.name,
                "risk_level": level,
                "obligations": [
                    "Technical documentation",
                    "Copyright compliance",
                    "Training data summary",
                ] + (["Model evaluation", "Systemic risk assessment",
                      "Incident reporting", "Cybersecurity"]
                     if level == "GPAI_SYSTEMIC" else []),
                "compliance_cost": self.COMPLIANCE_COSTS[level],
            }

        # Limited risk
        if system.interacts_with_humans or system.generates_content:
            return {
                "system": system.name,
                "risk_level": "LIMITED",
                "obligations": ["Disclose AI interaction",
                                "Mark generated content"],
                "compliance_cost": self.COMPLIANCE_COSTS["LIMITED"],
            }

        return {
            "system": system.name,
            "risk_level": "MINIMAL",
            "obligations": ["Voluntary code of conduct"],
            "compliance_cost": self.COMPLIANCE_COSTS["MINIMAL"],
        }


# Classify 10 systems
classifier = DetailedEUAIActClassifier()

systems = [
    AISystemDescription("HR Resume Screener", "Employment", "CV ranking",
                         True, False, 0, False),
    AISystemDescription("Customer Chatbot", "Retail", "Support",
                         True, False, 0, True),
    AISystemDescription("GPT-5 Equivalent", "General", "Foundation model",
                         True, True, 1e26, True),
    AISystemDescription("Spam Filter", "Email", "Classification",
                         False, False, 0, False),
    AISystemDescription("Credit Scorer", "Essential services", "Loan decisions",
                         True, False, 0, False),
    AISystemDescription("Medical Diagnosis", "Critical infrastructure",
                         "Disease detection", True, False, 0, False),
    AISystemDescription("Content Recommender", "Social media", "Feed ranking",
                         True, False, 0, False),
    AISystemDescription("Open LLM 7B", "General", "Foundation model",
                         True, True, 1e22, True),
    AISystemDescription("Proctoring Software", "Education", "Exam monitoring",
                         True, False, 0, False),
    AISystemDescription("Fraud Detector", "Law enforcement", "Transaction analysis",
                         False, False, 0, False),
]

print("=== EU AI Act Classification Results ===\n")
for sys in systems:
    result = classifier.classify(sys)
    cost = result["compliance_cost"]
    cost_tier = cost["tier"] if isinstance(cost, dict) else "N/A"
    print(f"{result['system']:25s} | {result['risk_level']:15s} | "
          f"Cost: {cost_tier}")
    print(f"  Obligations: {', '.join(result['obligations'][:3])}...")
```

</details>

### 연습문제 2: 책임 있는 확장 정책 평가기

책임 있는 확장 정책 평가기를 설계하고 구현하라:
1. 구체적인 역량 임계값을 가진 4개의 AI 안전 수준을 정의하라
2. 각 수준에 대한 역량 평가 체크리스트(10개 항목)를 작성하라
3. 역량 평가 결과를 ASL 수준으로 매핑하는 점수 함수를 구현하라
4. 각 수준에서 필요한 안전 장치를 정의하라(수준당 최소 5개)
5. 5개의 가상 모델에 대해 평가기를 실행하고 안전 보고서를 생성하라

<details>
<summary>정답 보기</summary>

```python
"""
Responsible Scaling Policy evaluator with capability assessment.
"""

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class CapabilityEval:
    """Result of a capability evaluation."""
    name: str
    score: float  # 0.0 to 1.0
    threshold_asl2: float
    threshold_asl3: float
    threshold_asl4: float


class RSPEvaluator:
    """Evaluate models against a Responsible Scaling Policy."""

    def __init__(self):
        self.capability_checks = [
            CapabilityEval("CBRN knowledge uplift", 0, 0.2, 0.5, 0.8),
            CapabilityEval("Cyber offense capability", 0, 0.2, 0.5, 0.8),
            CapabilityEval("Autonomous task completion", 0, 0.3, 0.6, 0.85),
            CapabilityEval("Persuasion/manipulation", 0, 0.3, 0.6, 0.8),
            CapabilityEval("Self-replication potential", 0, 0.1, 0.4, 0.7),
            CapabilityEval("Deception capability", 0, 0.2, 0.5, 0.75),
            CapabilityEval("Long-horizon planning", 0, 0.3, 0.6, 0.85),
            CapabilityEval("Code generation (exploit)", 0, 0.2, 0.5, 0.8),
            CapabilityEval("Scientific reasoning", 0, 0.4, 0.7, 0.9),
            CapabilityEval("Resource acquisition", 0, 0.1, 0.4, 0.7),
        ]

        self.safeguards = {
            "ASL-1": [
                "Basic content filtering",
                "Standard software security",
                "Usage monitoring",
                "Terms of service",
                "Bug bounty program",
            ],
            "ASL-2": [
                "RLHF/CAI alignment",
                "Red-teaming before deployment",
                "Rate limiting and abuse detection",
                "Model weight security (standard)",
                "Incident response plan",
                "User reporting mechanisms",
            ],
            "ASL-3": [
                "Enhanced model weight security (HSM, air-gap)",
                "Capability-specific access controls",
                "Continuous red-teaming program",
                "Third-party safety audits",
                "Government notification protocol",
                "Deployment review board approval",
                "Enhanced monitoring for misuse patterns",
            ],
            "ASL-4": [
                "Formal safety case required",
                "International oversight body review",
                "Extreme security (nation-state threat model)",
                "Mandatory government approval",
                "Kill switch / remote deactivation",
                "Alignment verification (formal methods)",
                "Restricted deployment or pause",
                "Continuous alignment monitoring",
            ],
        }

    def evaluate(self, model_name: str, scores: Dict[str, float]) -> dict:
        """Evaluate a model and assign ASL level."""
        # Apply scores
        for check in self.capability_checks:
            check.score = scores.get(check.name, 0.0)

        # Determine level based on highest triggered threshold
        level = "ASL-1"
        triggered = []

        for check in self.capability_checks:
            if check.score >= check.threshold_asl4:
                level = "ASL-4"
                triggered.append((check.name, "ASL-4", check.score))
            elif check.score >= check.threshold_asl3 and level not in ["ASL-4"]:
                level = "ASL-3"
                triggered.append((check.name, "ASL-3", check.score))
            elif check.score >= check.threshold_asl2 and level in ["ASL-1"]:
                level = "ASL-2"
                triggered.append((check.name, "ASL-2", check.score))

        return {
            "model": model_name,
            "level": level,
            "triggered_capabilities": triggered,
            "required_safeguards": self.safeguards[level],
            "deploy_allowed": level != "ASL-4",
        }


# Evaluate 5 hypothetical models
evaluator = RSPEvaluator()

models = {
    "SmallLM-7B": {"CBRN knowledge uplift": 0.1, "Cyber offense capability": 0.1,
                   "Autonomous task completion": 0.15, "Persuasion/manipulation": 0.1},
    "MediumLM-70B": {"CBRN knowledge uplift": 0.3, "Cyber offense capability": 0.25,
                     "Autonomous task completion": 0.35, "Persuasion/manipulation": 0.3,
                     "Scientific reasoning": 0.5},
    "FrontierLM-v1": {"CBRN knowledge uplift": 0.55, "Cyber offense capability": 0.5,
                      "Autonomous task completion": 0.65, "Persuasion/manipulation": 0.6,
                      "Scientific reasoning": 0.75, "Long-horizon planning": 0.6},
    "HyperLM-Next": {"CBRN knowledge uplift": 0.82, "Cyber offense capability": 0.85,
                     "Autonomous task completion": 0.9, "Self-replication potential": 0.75,
                     "Long-horizon planning": 0.88},
    "OpenAssistant-13B": {"CBRN knowledge uplift": 0.15, "Cyber offense capability": 0.2,
                          "Persuasion/manipulation": 0.25, "Code generation (exploit)": 0.3},
}

print("=== RSP Evaluation Results ===\n")
for name, scores in models.items():
    result = evaluator.evaluate(name, scores)
    deploy = "DEPLOY OK" if result["deploy_allowed"] else "HOLD"
    print(f"[{result['level']}] {result['model']} ({deploy})")
    print(f"  Safeguards required: {len(result['required_safeguards'])}")
    if result["triggered_capabilities"]:
        for cap, lvl, score in result["triggered_capabilities"][:3]:
            print(f"  Triggered: {cap} ({score:.2f} -> {lvl})")
    print()
```

</details>

### 연습문제 3: 안전 검토 위원회 시뮬레이션

안전 검토 위원회 프로세스를 시뮬레이션하라:
1. 서로 다른 역할과 위험 허용도를 가진 5명의 위원회 멤버를 정의하라
2. 다양한 위험 프로필을 가진 3개의 모델 배포 제안서를 작성하라
3. 각 멤버가 안전 기준을 평가하는 투표 프로세스를 구현하라
4. 위원회가 동의하지 않는 경우를 처리하라(동점 결정, 에스컬레이션)
5. 검토를 요약하는 회의록을 생성하라

<details>
<summary>정답 보기</summary>

```python
"""
Safety review board simulation with voting and minutes.
"""

from dataclasses import dataclass
from typing import List, Dict
import numpy as np


@dataclass
class BoardMember:
    """A member of the safety review board."""
    name: str
    role: str
    risk_tolerance: float  # 0.0 (very conservative) to 1.0 (very permissive)
    veto_power: bool = False


@dataclass
class DeploymentProposal:
    """A model deployment proposal for review."""
    model_name: str
    risk_score: float  # 0.0 to 1.0
    capabilities: Dict[str, float]
    safeguards_in_place: List[str]
    business_impact: str


class SafetyReviewBoard:
    """Simulate a safety review board process."""

    def __init__(self):
        self.members = [
            BoardMember("Dr. Chen", "Chief Safety Officer", 0.3, veto_power=True),
            BoardMember("Prof. Martinez", "External Safety Researcher", 0.25),
            BoardMember("Ms. Johnson", "Head of Engineering", 0.6),
            BoardMember("Mr. Williams", "Head of Product", 0.7),
            BoardMember("Dr. Park", "Ethics Advisor", 0.4),
        ]
        self.minutes: List[dict] = []

    def review_proposal(self, proposal: DeploymentProposal) -> dict:
        """Run full board review on a proposal."""
        votes = []
        for member in self.members:
            # Each member evaluates based on their risk tolerance
            threshold = member.risk_tolerance
            vote = proposal.risk_score < threshold
            reasoning = (
                "Risk within acceptable bounds"
                if vote else
                f"Risk ({proposal.risk_score:.2f}) exceeds my threshold ({threshold:.2f})"
            )
            votes.append({
                "member": member.name,
                "role": member.role,
                "vote": "APPROVE" if vote else "REJECT",
                "reasoning": reasoning,
                "has_veto": member.veto_power,
            })

        # Check for veto
        vetoed = any(
            v["vote"] == "REJECT" and v["has_veto"] for v in votes
        )

        approve_count = sum(1 for v in votes if v["vote"] == "APPROVE")
        total = len(votes)

        if vetoed:
            decision = "REJECTED (VETO)"
        elif approve_count > total / 2:
            decision = "APPROVED"
        elif approve_count == total / 2:
            decision = "ESCALATED TO CEO"
        else:
            decision = "REJECTED"

        record = {
            "proposal": proposal.model_name,
            "risk_score": proposal.risk_score,
            "votes": votes,
            "decision": decision,
            "approve_count": approve_count,
            "total_votes": total,
        }
        self.minutes.append(record)
        return record

    def generate_minutes(self) -> str:
        """Generate formatted meeting minutes."""
        lines = [
            "=" * 60,
            "  AI SAFETY REVIEW BOARD - MEETING MINUTES",
            "=" * 60,
        ]
        for i, record in enumerate(self.minutes, 1):
            lines.append(f"\n--- Proposal {i}: {record['proposal']} ---")
            lines.append(f"Risk Score: {record['risk_score']:.2f}")
            lines.append(f"\nVoting Record:")
            for v in record["votes"]:
                veto_str = " [VETO HOLDER]" if v["has_veto"] else ""
                lines.append(f"  {v['member']:20s} ({v['role']}){veto_str}")
                lines.append(f"    Vote: {v['vote']}")
                lines.append(f"    Reason: {v['reasoning']}")
            lines.append(f"\nDecision: {record['decision']} "
                         f"({record['approve_count']}/{record['total_votes']})")
        return "\n".join(lines)


board = SafetyReviewBoard()

proposals = [
    DeploymentProposal("SafeLM-v2", 0.2, {"harmful_output": 0.1},
                       ["RLHF", "red-team", "monitoring"], "Customer support"),
    DeploymentProposal("CodeGen-Pro", 0.5, {"code_exploit": 0.4},
                       ["output filtering"], "Developer tool"),
    DeploymentProposal("AgentLM-Alpha", 0.8, {"autonomous_action": 0.7},
                       ["basic monitoring"], "Autonomous agent"),
]

for proposal in proposals:
    board.review_proposal(proposal)

print(board.generate_minutes())
```

</details>

### 연습문제 4: 컴퓨팅 거버넌스 모니터

컴퓨팅 거버넌스 모니터링 시스템을 구축하라:
1. 학습 실행에 대한 GPU 시간과 추정 FLOP을 추적하라
2. 규제 임계값(EU: 10^25, US: 10^26)을 초과하는 실행을 표시하라
3. 클라우드 컴퓨팅 고객에 대한 KYC 검사를 구현하라
4. 임계값 근접 경고와 함께 규정 준수 보고서를 생성하라
5. 알고리즘 효율성 개선이 임계값 관련성에 미치는 영향을 모델링하라

<details>
<summary>정답 보기</summary>

```python
"""
Compute governance monitoring system.
"""

from dataclasses import dataclass, field
from typing import List, Dict
from datetime import datetime
import numpy as np


@dataclass
class TrainingRun:
    """A model training run to monitor."""
    run_id: str
    organization: str
    model_name: str
    gpu_type: str
    gpu_count: int
    training_hours: float
    estimated_flops: float
    start_date: str
    kyc_verified: bool = True


class ComputeMonitor:
    """Monitor compute usage for regulatory compliance."""

    THRESHOLDS = {
        "EU_AI_ACT": 1e25,
        "US_EO_14110": 1e26,
    }

    GPU_FLOPS_PER_HOUR = {
        "A100": 3.12e17,   # ~312 TFLOPS * 3600s
        "H100": 9.9e17,    # ~990 TFLOPS * 3600s
        "H200": 1.32e18,   # ~1320 TFLOPS * 3600s (estimated)
    }

    def __init__(self):
        self.runs: List[TrainingRun] = []
        self.alerts: List[dict] = []

    def register_run(self, run: TrainingRun) -> dict:
        """Register and check a training run."""
        self.runs.append(run)

        flags = []
        for regime, threshold in self.THRESHOLDS.items():
            ratio = run.estimated_flops / threshold
            if ratio >= 1.0:
                flags.append({
                    "regime": regime,
                    "status": "EXCEEDS_THRESHOLD",
                    "ratio": ratio,
                    "action": "Mandatory reporting required",
                })
            elif ratio >= 0.5:
                flags.append({
                    "regime": regime,
                    "status": "APPROACHING_THRESHOLD",
                    "ratio": ratio,
                    "action": "Warning: within 2x of threshold",
                })

        if not run.kyc_verified:
            flags.append({
                "regime": "KYC",
                "status": "UNVERIFIED_CUSTOMER",
                "action": "Verify customer identity before proceeding",
            })

        if flags:
            self.alerts.extend(flags)

        return {"run_id": run.run_id, "flags": flags}

    def efficiency_projection(self, years_ahead: int = 3) -> dict:
        """Project how efficiency improvements affect thresholds."""
        efficiency_multiplier = 2.0  # 2x per year improvement

        results = {}
        for year in range(years_ahead + 1):
            effective_threshold_eu = self.THRESHOLDS["EU_AI_ACT"] * (
                efficiency_multiplier ** year
            )
            effective_threshold_us = self.THRESHOLDS["US_EO_14110"] * (
                efficiency_multiplier ** year
            )
            results[f"Year +{year}"] = {
                "EU_equivalent_flops": f"{effective_threshold_eu:.1e}",
                "US_equivalent_flops": f"{effective_threshold_us:.1e}",
                "interpretation": (
                    f"With {efficiency_multiplier**year:.0f}x efficiency gain, "
                    f"today's {self.THRESHOLDS['EU_AI_ACT']:.0e} FLOP model "
                    f"can be trained with {self.THRESHOLDS['EU_AI_ACT']/(efficiency_multiplier**year):.1e} FLOPs"
                ),
            }
        return results

    def compliance_report(self) -> str:
        """Generate compliance report."""
        lines = ["=== Compute Governance Compliance Report ===\n"]

        total_flops = sum(r.estimated_flops for r in self.runs)
        lines.append(f"Total runs monitored: {len(self.runs)}")
        lines.append(f"Total compute: {total_flops:.2e} FLOPs")

        for regime, threshold in self.THRESHOLDS.items():
            exceeding = [r for r in self.runs
                         if r.estimated_flops >= threshold]
            lines.append(f"\n{regime} (threshold: {threshold:.0e}):")
            lines.append(f"  Exceeding runs: {len(exceeding)}")
            for run in exceeding:
                lines.append(f"    - {run.run_id}: {run.estimated_flops:.2e}")

        unverified = [r for r in self.runs if not r.kyc_verified]
        lines.append(f"\nUnverified customers: {len(unverified)}")

        lines.append(f"\nTotal alerts: {len(self.alerts)}")
        return "\n".join(lines)


monitor = ComputeMonitor()

runs = [
    TrainingRun("run-001", "SmallCo", "Model-A", "A100", 64, 720,
                64 * 720 * 3.12e17, "2024-01-15"),
    TrainingRun("run-002", "BigCorp", "Frontier-v2", "H100", 4096, 2160,
                4096 * 2160 * 9.9e17, "2024-03-01"),
    TrainingRun("run-003", "Unknown Ltd", "Secret-LM", "H100", 2048, 1440,
                2048 * 1440 * 9.9e17, "2024-06-01", kyc_verified=False),
]

print("=== Registering Training Runs ===\n")
for run in runs:
    result = monitor.register_run(run)
    if result["flags"]:
        for flag in result["flags"]:
            print(f"[{flag['status']}] {run.run_id}: {flag.get('regime', '')} "
                  f"- {flag['action']}")

print("\n" + monitor.compliance_report())

print("\n\n=== Efficiency Projection ===")
projection = monitor.efficiency_projection(years_ahead=3)
for year, data in projection.items():
    print(f"\n{year}: {data['interpretation']}")
```

</details>

### 연습문제 5: 거버넌스 격차 분석 도구

주어진 AI 배포에 대한 거버넌스 격차를 식별하는 도구를 만들어라:
1. 5가지 거버넌스 차원을 모델링하라: 법적, 기술적, 조직적, 윤리적, 국제적
2. 각 차원에 대해 충족되어야 하는 5가지 기준을 정의하라
3. 25개 기준 모두에 대해 배포 시나리오를 평가하라
4. 격차를 식별하고 위험 심각도에 따라 우선순위를 매겨라
5. 일정이 포함된 실행 가능한 개선 계획을 생성하라

<details>
<summary>정답 보기</summary>

```python
"""
AI governance gap analysis tool.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np


@dataclass
class GapCriterion:
    """A governance criterion to evaluate."""
    dimension: str
    name: str
    description: str
    severity_if_missing: str  # "critical", "high", "medium", "low"
    remediation_effort: str   # "easy", "moderate", "hard"


class GovernanceGapAnalyzer:
    """Analyze governance gaps across multiple dimensions."""

    def __init__(self):
        self.criteria = self._define_criteria()

    def _define_criteria(self) -> List[GapCriterion]:
        """Define 25 criteria across 5 dimensions."""
        criteria = []
        # Legal (5)
        legal = [
            ("EU AI Act compliance assessment", "critical", "hard"),
            ("Data protection / GDPR compliance", "critical", "moderate"),
            ("Liability framework defined", "high", "moderate"),
            ("Intellectual property review", "medium", "moderate"),
            ("Terms of service / acceptable use policy", "high", "easy"),
        ]
        for name, sev, effort in legal:
            criteria.append(GapCriterion("Legal", name, "", sev, effort))

        # Technical (5)
        technical = [
            ("Safety evaluation benchmarks completed", "critical", "moderate"),
            ("Red-teaming conducted", "critical", "moderate"),
            ("Monitoring and logging system deployed", "high", "moderate"),
            ("Guardrails / content filtering active", "high", "moderate"),
            ("Model card / documentation published", "medium", "easy"),
        ]
        for name, sev, effort in technical:
            criteria.append(GapCriterion("Technical", name, "", sev, effort))

        # Organizational (5)
        organizational = [
            ("Safety review board established", "high", "moderate"),
            ("Incident response plan documented", "critical", "moderate"),
            ("Clear escalation procedures", "high", "easy"),
            ("Safety training for staff", "medium", "easy"),
            ("Regular safety audit schedule", "high", "moderate"),
        ]
        for name, sev, effort in organizational:
            criteria.append(GapCriterion("Organizational", name, "", sev, effort))

        # Ethical (5)
        ethical = [
            ("Bias/fairness assessment conducted", "high", "moderate"),
            ("Stakeholder impact assessment", "medium", "moderate"),
            ("Transparency about AI use to end users", "high", "easy"),
            ("Dual-use risk assessment", "critical", "hard"),
            ("Environmental impact estimated", "low", "easy"),
        ]
        for name, sev, effort in ethical:
            criteria.append(GapCriterion("Ethical", name, "", sev, effort))

        # International (5)
        international = [
            ("Multi-jurisdiction compliance check", "high", "hard"),
            ("Cross-border data transfer compliance", "critical", "hard"),
            ("International standards alignment (ISO)", "medium", "moderate"),
            ("Export control compliance", "critical", "hard"),
            ("International incident sharing protocol", "low", "moderate"),
        ]
        for name, sev, effort in international:
            criteria.append(GapCriterion("International", name, "", sev, effort))

        return criteria

    def evaluate(self, satisfied: Dict[str, bool]) -> dict:
        """Evaluate criteria and identify gaps."""
        gaps = []
        passed = []
        severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        effort_days = {"easy": 14, "moderate": 60, "hard": 180}

        for criterion in self.criteria:
            is_met = satisfied.get(criterion.name, False)
            if not is_met:
                gaps.append({
                    "dimension": criterion.dimension,
                    "criterion": criterion.name,
                    "severity": criterion.severity_if_missing,
                    "severity_score": severity_order[criterion.severity_if_missing],
                    "effort": criterion.remediation_effort,
                    "estimated_days": effort_days[criterion.remediation_effort],
                })
            else:
                passed.append(criterion.name)

        # Sort by severity
        gaps.sort(key=lambda x: -x["severity_score"])

        return {
            "total_criteria": len(self.criteria),
            "satisfied": len(passed),
            "gaps": len(gaps),
            "gap_list": gaps,
            "score": len(passed) / len(self.criteria),
        }

    def remediation_plan(self, gaps: List[dict]) -> str:
        """Generate prioritized remediation plan."""
        lines = ["\n=== REMEDIATION PLAN ===\n"]
        phases = {
            "Phase 1 (0-30 days) - Critical": [],
            "Phase 2 (30-90 days) - High": [],
            "Phase 3 (90-180 days) - Medium/Low": [],
        }

        for gap in gaps:
            if gap["severity"] == "critical":
                phases["Phase 1 (0-30 days) - Critical"].append(gap)
            elif gap["severity"] == "high":
                phases["Phase 2 (30-90 days) - High"].append(gap)
            else:
                phases["Phase 3 (90-180 days) - Medium/Low"].append(gap)

        for phase, items in phases.items():
            lines.append(f"\n{phase}:")
            if not items:
                lines.append("  No items.")
            for item in items:
                lines.append(f"  [{item['severity'].upper()}] {item['criterion']}")
                lines.append(f"    Dimension: {item['dimension']}")
                lines.append(f"    Effort: {item['effort']} "
                             f"(~{item['estimated_days']} days)")

        return "\n".join(lines)


# Evaluate a deployment scenario
analyzer = GovernanceGapAnalyzer()

# Simulate: some criteria met, some not
np.random.seed(42)
satisfied = {}
for c in analyzer.criteria:
    satisfied[c.name] = np.random.random() > 0.4  # 60% chance of being met

result = analyzer.evaluate(satisfied)
print(f"=== Governance Gap Analysis ===")
print(f"Score: {result['score']:.0%} ({result['satisfied']}/{result['total_criteria']})")
print(f"Gaps found: {result['gaps']}")
print(f"\nTop gaps by severity:")
for gap in result["gap_list"][:5]:
    print(f"  [{gap['severity'].upper()}] {gap['criterion']} ({gap['dimension']})")

print(analyzer.remediation_plan(result["gap_list"]))
```

</details>

---

[이전: 기만적 정렬](./12_Deceptive_Alignment.md) | [개요](./00_Overview.md) | [다음: 책임 있는 배포](./14_Responsible_Deployment.md)

**License**: CC BY-NC 4.0
