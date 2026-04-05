# Lesson 13: Governance Frameworks

[Previous: Deceptive Alignment](./12_Deceptive_Alignment.md) | [Next: Responsible Deployment](./14_Responsible_Deployment.md)

---

## Learning Objectives

- Map the global AI governance landscape including key actors, frameworks, and jurisdictions
- Analyze the EU AI Act's risk-based classification system and compliance requirements
- Understand US executive orders and federal agency guidance on AI safety
- Evaluate international cooperation mechanisms for AI governance (GPAI, OECD, UN)
- Design frontier model governance frameworks including responsible scaling policies
- Implement internal safety governance structures (review boards, safety cases)
- Assess compute governance as a lever for AI regulation
- Evaluate the open-source vs closed-source safety debate and its governance implications

---

## Table of Contents

1. [AI Governance Landscape](#1-ai-governance-landscape)
2. [EU AI Act](#2-eu-ai-act)
3. [US Executive Orders and Policy](#3-us-executive-orders-and-policy)
4. [International Cooperation](#4-international-cooperation)
5. [Frontier Model Governance](#5-frontier-model-governance)
6. [Responsible Scaling Policies](#6-responsible-scaling-policies)
7. [Internal Safety Governance](#7-internal-safety-governance)
8. [Compute Governance](#8-compute-governance)
9. [Open-Source vs Closed-Source Safety](#9-open-source-vs-closed-source-safety)
10. [Summary](#summary)
11. [Exercises](#exercises)

---

## 1. AI Governance Landscape

### 1.1 The Governance Challenge

AI governance faces a fundamental tension: technology moves faster than
regulation, yet the consequences of ungoverned AI deployment can be severe.
Effective governance must balance innovation with safety, national interest
with international coordination, and speed with thoroughness.

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

**Global AI Governance Comparison**

| Dimension | EU (AI Act) | US (Executive Orders) | China (Regulations) | UK (Pro-Innovation) |
|-----------|-------------|----------------------|--------------------|--------------------|
| Approach | Risk-based regulation | Sector-specific guidance | State-directed control | Principles-based |
| Scope | All AI systems by risk tier | Federal agencies + voluntary | Generative AI + algorithms | Cross-sector framework |
| Enforcement | Fines up to €35M / 7% revenue | Agency-specific | Administrative penalties | Regulator-led |
| Timeline | Phased 2024–2027 | Ongoing executive action | Active since 2023 | Framework since 2023 |
| Foundation models | GPAI obligations | Voluntary commitments | Registration required | Safety institute testing |

### 1.2 China's AI Governance Approach and SME Compliance Burden

China's regulatory framework for AI consists of three overlapping regulations enacted in rapid succession: the **Algorithm Recommendation Provisions** (2022), which require disclosure of recommendation logic and prohibit addiction-inducing features; the **Deep Synthesis Regulations** (2023), which mandate watermarking of AI-generated content and real-name registration for service providers; and the **Interim Measures for Generative AI** (2023), which require algorithm registration with the Cyberspace Administration of China (CAC), training data disclosure, and content filtering aligned with "core socialist values." A distinctive feature of China's approach is mandatory registration: any generative AI service available to the public must file with the CAC before launch, giving regulators visibility into the AI stack that Western frameworks do not yet require.

For smaller organizations operating globally, the **EU AI Act compliance burden** is a significant concern. Independent analyses estimate that high-risk system compliance — covering conformity assessments, technical documentation, quality management systems, and post-market monitoring — can cost SMEs between €200,000 and €400,000 per AI system. To mitigate this, the Act provides for regulatory sandboxes (Article 57–63) where SMEs can test systems under regulatory supervision with reduced obligations, and SMEs are partially exempt from certain GPAI documentation requirements. However, these provisions remain underutilized in practice, partly because the sandbox frameworks are still being stood up by member states and partly because SMEs often lack the legal expertise to navigate the exemption criteria.

---

## 2. EU AI Act

The EU AI Act uses a four-tier risk pyramid to assign obligations proportionate to potential harm:

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

### 2.1 Risk-Based Classification

The EU AI Act (Regulation 2024/1689) establishes a risk-based framework
that categorizes AI systems into four tiers with corresponding obligations.

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

## 3. US Executive Orders and Policy

### 3.1 Executive Order on AI Safety

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

## 4. International Cooperation

### 4.1 Multilateral Frameworks

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

## 5. Frontier Model Governance

### 5.1 Defining Frontier Models

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

## 6. Responsible Scaling Policies

### 6.1 Anthropic's RSP Framework

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

## 7. Internal Safety Governance

### 7.1 Safety Review Boards and Processes

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

## 8. Compute Governance

### 8.1 Compute as a Governance Lever

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

## 9. Open-Source vs Closed-Source Safety

### 9.1 The Debate

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

## Summary

- The **AI governance landscape** involves actors at international, regional,
  national, organizational, and technical levels, with significant gaps in
  cross-border coordination and enforcement
- The **EU AI Act** establishes a risk-based classification (prohibited, high,
  limited, minimal) with specific obligations at each level, plus GPAI rules
  using the 10^25 FLOP threshold for systemic risk designation
- **US governance** centers on EO 14110 (reporting thresholds for frontier models),
  NIST AI RMF, and sector-specific agency guidance
- **International cooperation** includes the Bletchley Declaration, OECD AI
  Principles, G7 Hiroshima Process, and the AI Safety Institutes Network
- **Frontier model governance** defines capability thresholds and safety
  requirements, with frameworks like Anthropic's RSP tying AI Safety Levels
  to required safeguards
- **Responsible Scaling Policies** (RSPs) create a ladder from ASL-1 (minimal
  risk) to ASL-4 (potential catastrophic risk), with escalating safety
  requirements at each level
- **Internal safety governance** includes safety review boards with authority
  to halt deployments, policy frameworks, and structured review processes
- **Compute governance** leverages the physical, concentrated nature of compute
  through export controls, KYC rules, and training compute thresholds
- The **open-source vs closed-source** debate involves genuine trade-offs
  between transparency/scrutiny and irrevocable release/misuse risks

---

## Exercises

### Exercise 1: EU AI Act Classifier

Build a detailed EU AI Act risk classifier:
1. Implement all four risk tiers with full Article references
2. Add the GPAI classification with compute-based and designation-based paths
3. Classify 10 real-world AI systems (provide descriptions) into the correct tier
4. For each high-risk classification, list the specific compliance obligations
5. Calculate estimated compliance cost tiers (low/medium/high/very high)

<details>
<summary>Solution</summary>

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

### Exercise 2: Responsible Scaling Policy Evaluator

Design and implement a Responsible Scaling Policy evaluator:
1. Define 4 AI Safety Levels with specific capability thresholds
2. Create a capability evaluation checklist (10 items) for each level
3. Implement a scoring function that maps capability eval results to an ASL level
4. Define the safeguards required at each level (at least 5 per level)
5. Run the evaluator on 5 hypothetical models and generate safety reports

<details>
<summary>Solution</summary>

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

### Exercise 3: Safety Review Board Simulation

Simulate a safety review board process:
1. Define 5 board members with different roles and risk tolerances
2. Create 3 model deployment proposals with varying risk profiles
3. Implement a voting process where each member evaluates safety criteria
4. Handle cases where the board disagrees (tie-breaking, escalation)
5. Generate meeting minutes summarizing the review

<details>
<summary>Solution</summary>

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

### Exercise 4: Compute Governance Monitor

Build a compute governance monitoring system:
1. Track GPU-hours and estimated FLOPs for training runs
2. Flag runs that exceed regulatory thresholds (EU: 10^25, US: 10^26)
3. Implement KYC checks for cloud compute customers
4. Generate compliance reports with threshold proximity warnings
5. Model how algorithmic efficiency improvements affect threshold relevance

<details>
<summary>Solution</summary>

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

### Exercise 5: Governance Gap Analysis Tool

Create a tool that identifies governance gaps for a given AI deployment:
1. Model 5 governance dimensions: legal, technical, organizational, ethical, international
2. For each dimension, define 5 criteria that should be satisfied
3. Evaluate a deployment scenario against all 25 criteria
4. Identify gaps and prioritize them by risk severity
5. Generate an actionable remediation plan with timelines

<details>
<summary>Solution</summary>

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

[Previous: Deceptive Alignment](./12_Deceptive_Alignment.md) | [Overview](./00_Overview.md) | [Next: Responsible Deployment](./14_Responsible_Deployment.md)

**License**: CC BY-NC 4.0
