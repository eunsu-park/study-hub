# 레슨 15: 사회적 영향 (Societal Impact)

[이전: 책임감 있는 배포](./14_Responsible_Deployment.md) | [다음: 미해결 문제](./16_Open_Problems.md)

---

## 학습 목표

- 대체(Displacement), 증강(Augmentation), 새로운 일자리 창출을 포함한 AI의 노동 시장 영향 분석
- 소수의 조직에 집중된 AI 역량으로 인한 경제적 집중(Economic Concentration) 위험 평가
- 허위 정보(Misinformation), 딥페이크(Deepfake), 인식론적 위험(Epistemic Risk)을 포함한 정보 생태계 효과 평가
- 유익한 AI 역량이 유해한 응용을 가능하게 하는 이중 용도(Dual-Use) 우려 이해
- 감시(Surveillance), 조작(Manipulation), 권력 비대칭(Power Asymmetry)을 포함한 AI의 민주주의적 함의 검토
- 대규모 AI 훈련 및 추론의 환경적 영향(Environmental Impact) 정량화
- AI 혜택이 인구 간 불균등하게 축적되는 디지털 격차(Digital Divide) 위험 식별
- 유익한 AI 개발을 위한 장기적 경로와 정책 권고안에 대한 추론

---

## 목차

1. [노동 시장 효과](#1-노동-시장-효과)
2. [경제적 집중](#2-경제적-집중)
3. [정보 생태계 효과](#3-정보-생태계-효과)
4. [이중 용도 우려](#4-이중-용도-우려)
5. [민주주의적 함의](#5-민주주의적-함의)
6. [환경적 영향](#6-환경적-영향)
7. [디지털 격차](#7-디지털-격차)
8. [문화적 영향](#8-문화적-영향)
9. [장기 경로와 정책 권고안](#9-장기-경로와-정책-권고안)
10. [요약](#요약)
11. [연습문제](#연습문제)

---

**AI 사회적 영향 평가 차원**

| 차원 | 긍정적 잠재력 | 부정적 위험 | 측정 접근법 |
|------|------------|-----------|-----------|
| 노동 시장 | 생산성 향상, 새로운 일자리 창출 | 일자리 대체, 임금 압박 | 고용 통계, 업무 자동화율 |
| 형평성 | 전문 지식에 대한 민주화된 접근 | 디지털 격차 심화 | 지니 계수, 접근 지표 |
| 환경 | 에너지 시스템 최적화 | 학습 탄소 발자국 | 학습 실행당 CO₂, PUE 지표 |
| 정보 | 지식 민주화 | 허위 정보 증폭 | 정보 품질 지수 |
| 거버넌스 | 증거 기반 정책 수립 | 감시, 조작 | 자유 지수, 투명성 보고서 |
| 건강 | 신약 발견, 진단 | 의료 AI의 편향 | 인구 통계 간 정확도 균등성 |

---

## 1. 노동 시장 효과

### 1.1 대체, 증강, 그리고 창출

AI는 세 가지 메커니즘을 통해 고용에 영향을 미칩니다: 업무가 자동화되는 근로자의 대체(Displacement), AI를 사용하여 더 생산적이 되는 근로자의 증강(Augmentation), 그리고 이전에 존재하지 않았던 완전히 새로운 역할의 창출(Creation)입니다.

```python
"""
Modeling AI's labor market impact: displacement, augmentation,
and job creation across different sectors and skill levels.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import numpy as np


@dataclass
class Occupation:
    """An occupation with exposure to AI automation."""
    name: str
    sector: str
    skill_level: str  # "low", "medium", "high"
    current_workers: int  # Thousands
    ai_exposure: float  # 0.0 to 1.0 (fraction of tasks automatable)
    augmentation_potential: float  # 0.0 to 1.0
    displacement_risk: float  # 0.0 to 1.0
    wage_level: str  # "low", "medium", "high"


class LaborMarketImpactModel:
    """Model the impact of AI on labor markets.

    Based on frameworks from:
    - Acemoglu & Restrepo (2020): Automation and new tasks
    - Eloundou et al. (2023): GPTs are GPTs (LLM exposure)
    - Autor (2024): Applying AI to rebuild middle-class jobs
    """

    def __init__(self):
        self.occupations = self._define_occupations()

    def _define_occupations(self) -> List[Occupation]:
        """Define occupations with AI exposure metrics.

        Exposure estimates are illustrative, based on published
        research but simplified for educational purposes.
        """
        return [
            Occupation("Software Developer", "Technology", "high",
                       1800, 0.65, 0.80, 0.15, "high"),
            Occupation("Data Entry Clerk", "Administrative", "low",
                       150, 0.90, 0.10, 0.85, "low"),
            Occupation("Radiologist", "Healthcare", "high",
                       35, 0.55, 0.75, 0.10, "high"),
            Occupation("Truck Driver", "Transportation", "medium",
                       3500, 0.40, 0.20, 0.35, "medium"),
            Occupation("Customer Service Rep", "Services", "low",
                       2900, 0.80, 0.30, 0.60, "low"),
            Occupation("Teacher (K-12)", "Education", "high",
                       3700, 0.40, 0.65, 0.05, "medium"),
            Occupation("Paralegal", "Legal", "medium",
                       350, 0.75, 0.60, 0.40, "medium"),
            Occupation("Content Writer", "Media", "medium",
                       150, 0.80, 0.55, 0.50, "medium"),
            Occupation("Registered Nurse", "Healthcare", "high",
                       3100, 0.25, 0.50, 0.05, "medium"),
            Occupation("Financial Analyst", "Finance", "high",
                       300, 0.70, 0.75, 0.20, "high"),
            Occupation("Warehouse Worker", "Logistics", "low",
                       1300, 0.60, 0.15, 0.55, "low"),
            Occupation("Graphic Designer", "Creative", "medium",
                       280, 0.70, 0.50, 0.35, "medium"),
        ]

    def analyze_impact(
        self,
        adoption_rate: float = 0.5,
        augmentation_productivity_gain: float = 0.3,
    ) -> dict:
        """Analyze labor market impact at a given AI adoption rate.

        Parameters
        ----------
        adoption_rate : fraction of exposed tasks actually automated (0-1)
        augmentation_productivity_gain : productivity boost for augmented workers
        """
        results = []
        total_displaced = 0
        total_augmented = 0
        total_current = 0

        for occ in self.occupations:
            # Displacement: workers whose tasks are fully automated
            effective_displacement = (
                occ.displacement_risk * adoption_rate * occ.current_workers
            )
            # Augmentation: workers who become more productive
            effective_augmented = (
                occ.augmentation_potential * adoption_rate * occ.current_workers
            )
            # Net productivity effect
            productivity_effect = (
                effective_augmented * augmentation_productivity_gain
            )

            total_displaced += effective_displacement
            total_augmented += effective_augmented
            total_current += occ.current_workers

            results.append({
                "occupation": occ.name,
                "sector": occ.sector,
                "skill_level": occ.skill_level,
                "current_workers_k": occ.current_workers,
                "displaced_k": effective_displacement,
                "augmented_k": effective_augmented,
                "displacement_pct": effective_displacement / occ.current_workers * 100,
                "ai_exposure": occ.ai_exposure,
            })

        # Sort by displacement percentage
        results.sort(key=lambda x: -x["displacement_pct"])

        # Sector summary
        sector_impact = {}
        for r in results:
            sector = r["sector"]
            if sector not in sector_impact:
                sector_impact[sector] = {"displaced_k": 0, "augmented_k": 0,
                                          "total_k": 0}
            sector_impact[sector]["displaced_k"] += r["displaced_k"]
            sector_impact[sector]["augmented_k"] += r["augmented_k"]
            sector_impact[sector]["total_k"] += r["current_workers_k"]

        # Skill level summary
        skill_impact = {}
        for r in results:
            skill = r["skill_level"]
            if skill not in skill_impact:
                skill_impact[skill] = {"displaced_k": 0, "total_k": 0}
            skill_impact[skill]["displaced_k"] += r["displaced_k"]
            skill_impact[skill]["total_k"] += r["current_workers_k"]

        return {
            "adoption_rate": adoption_rate,
            "total_workers_k": total_current,
            "total_displaced_k": total_displaced,
            "total_augmented_k": total_augmented,
            "displacement_rate": total_displaced / total_current,
            "occupation_details": results,
            "sector_summary": sector_impact,
            "skill_summary": skill_impact,
        }


# Run analysis
model = LaborMarketImpactModel()

# Scenario analysis at different adoption rates
for adoption in [0.2, 0.5, 0.8]:
    result = model.analyze_impact(adoption_rate=adoption)
    print(f"\n=== Adoption Rate: {adoption:.0%} ===")
    print(f"Total workers: {result['total_workers_k']:,.0f}K")
    print(f"Displaced: {result['total_displaced_k']:,.0f}K "
          f"({result['displacement_rate']:.1%})")
    print(f"Augmented: {result['total_augmented_k']:,.0f}K")

    print(f"\nMost affected occupations:")
    for occ in result["occupation_details"][:5]:
        print(f"  {occ['occupation']:25s}: "
              f"{occ['displacement_pct']:5.1f}% displaced "
              f"(exposure: {occ['ai_exposure']:.0%})")

    print(f"\nBy skill level:")
    for skill, data in result["skill_summary"].items():
        rate = data["displaced_k"] / data["total_k"] * 100
        print(f"  {skill:8s}: {rate:.1f}% displacement rate")
```

---

## 2. 경제적 집중

### 2.1 승자독식 역학

```python
"""
Economic concentration risks from AI: analysis of market
power dynamics and winner-take-all effects.
"""


class EconomicConcentrationAnalysis:
    """Analyze economic concentration risks from AI.

    Key concerns:
    1. Compute requirements create barriers to entry
    2. Data advantages compound over time
    3. Talent is concentrated in a few organizations
    4. Network effects in AI platforms
    """

    def __init__(self):
        self.concentration_factors = self._define_factors()

    def _define_factors(self) -> List[dict]:
        """Define factors driving economic concentration."""
        return [
            {
                "factor": "Compute Costs",
                "description": "Training frontier models costs $100M+ and rising",
                "metric": "Training cost of frontier model (USD)",
                "trend": [
                    {"year": 2020, "value": 5e6},
                    {"year": 2022, "value": 50e6},
                    {"year": 2024, "value": 200e6},
                    {"year": 2026, "value": 1e9},  # Projected
                ],
                "concentration_effect": "high",
                "countermeasure": "Open models, efficient architectures, "
                                  "government compute programs",
            },
            {
                "factor": "Data Advantages",
                "description": "Large platforms have proprietary data advantages",
                "metric": "Unique data scale (relative)",
                "concentration_effect": "high",
                "countermeasure": "Data sharing mandates, synthetic data, "
                                  "public domain datasets",
            },
            {
                "factor": "Talent Concentration",
                "description": "Top AI researchers concentrated at few companies",
                "metric": "% of top-cited AI researchers at top 5 companies",
                "concentration_effect": "medium",
                "countermeasure": "Academic funding, talent immigration, "
                                  "distributed research",
            },
            {
                "factor": "Network Effects",
                "description": "AI platforms become more valuable with more users",
                "metric": "Platform user base growth rate",
                "concentration_effect": "high",
                "countermeasure": "Interoperability standards, data portability, "
                                  "antitrust enforcement",
            },
            {
                "factor": "Regulatory Capture",
                "description": "Large companies shape regulation in their favor",
                "metric": "Lobbying spend by AI companies",
                "concentration_effect": "medium",
                "countermeasure": "Transparent regulatory processes, public "
                                  "interest representation, sunset clauses",
            },
        ]

    def compute_herfindahl(self, market_shares: List[float]) -> float:
        """Compute the Herfindahl-Hirschman Index (HHI).

        HHI < 1500: Competitive market
        1500-2500: Moderately concentrated
        > 2500: Highly concentrated
        """
        return sum(s**2 for s in market_shares) * 10000

    def analyze_market(self, market_name: str, shares: Dict[str, float]) -> dict:
        """Analyze market concentration."""
        share_values = list(shares.values())
        hhi = self.compute_herfindahl(share_values)

        top_3_share = sum(sorted(share_values, reverse=True)[:3])

        if hhi > 2500:
            classification = "Highly concentrated"
        elif hhi > 1500:
            classification = "Moderately concentrated"
        else:
            classification = "Competitive"

        return {
            "market": market_name,
            "hhi": hhi,
            "classification": classification,
            "top_3_share": top_3_share,
            "num_players": len(shares),
            "players": shares,
        }


# Analyze AI market concentration
analyzer = EconomicConcentrationAnalysis()

# Hypothetical market shares for AI model providers
markets = {
    "Foundation Model API": {
        "OpenAI": 0.35, "Google": 0.25, "Anthropic": 0.15,
        "Meta (open)": 0.10, "Others": 0.15,
    },
    "AI Cloud Compute": {
        "AWS": 0.32, "Azure": 0.23, "GCP": 0.10,
        "Others": 0.35,
    },
    "AI Chip Manufacturing": {
        "NVIDIA": 0.80, "AMD": 0.10, "Intel": 0.03,
        "Others": 0.07,
    },
}

print("=== AI Market Concentration Analysis ===\n")
for market_name, shares in markets.items():
    result = analyzer.analyze_market(market_name, shares)
    print(f"Market: {result['market']}")
    print(f"  HHI: {result['hhi']:.0f} ({result['classification']})")
    print(f"  Top 3 share: {result['top_3_share']:.0%}")
    print(f"  Players: {result['num_players']}")
    print()

# Show concentration factors
print("\n=== Concentration Drivers ===\n")
for factor in analyzer.concentration_factors:
    print(f"{factor['factor']} [{factor['concentration_effect'].upper()}]:")
    print(f"  {factor['description']}")
    print(f"  Countermeasure: {factor['countermeasure']}")
    print()
```

---

## 3. 정보 생태계 효과

### 3.1 허위 정보와 딥페이크

```python
"""
AI's impact on the information ecosystem: misinformation,
deepfakes, and epistemic risks.
"""


class InformationEcosystemAnalysis:
    """Analyze AI's effects on the information ecosystem."""

    def __init__(self):
        self.threat_vectors = self._define_threats()

    def _define_threats(self) -> List[dict]:
        """Define information ecosystem threats from AI."""
        return [
            {
                "threat": "Synthetic text at scale",
                "description": "LLMs can generate convincing news articles, "
                               "social media posts, and comments at massive scale",
                "current_severity": "high",
                "trend": "increasing",
                "detection_difficulty": "high (improving models are harder to detect)",
                "defenses": [
                    "AI-generated text watermarking (C2PA)",
                    "Statistical detection models",
                    "Provenance tracking",
                    "Media literacy education",
                ],
            },
            {
                "threat": "Deepfake audio and video",
                "description": "AI-generated realistic audio and video of real people "
                               "saying or doing things they never did",
                "current_severity": "high",
                "trend": "increasing rapidly",
                "detection_difficulty": "medium (but gap closing)",
                "defenses": [
                    "Deepfake detection models",
                    "Content authenticity standards (C2PA, IPTC)",
                    "Platform policies and enforcement",
                    "Legal frameworks (deepfake laws)",
                ],
            },
            {
                "threat": "Personalized manipulation",
                "description": "AI enables targeted psychological manipulation at "
                               "individual level using personal data profiles",
                "current_severity": "medium",
                "trend": "increasing",
                "detection_difficulty": "very high (appears as normal content)",
                "defenses": [
                    "Data privacy regulations (GDPR, CCPA)",
                    "Algorithmic auditing",
                    "Transparency requirements for recommendation systems",
                    "Individual AI literacy training",
                ],
            },
            {
                "threat": "Epistemic erosion",
                "description": "As AI-generated content floods the internet, the ability "
                               "to distinguish truth from fiction degrades for everyone",
                "current_severity": "medium",
                "trend": "increasing",
                "detection_difficulty": "systemic (not individual content)",
                "defenses": [
                    "Trusted content provenance systems",
                    "Institutional credibility signals",
                    "Fact-checking infrastructure",
                    "AI-assisted verification tools",
                ],
            },
            {
                "threat": "Search and knowledge pollution",
                "description": "AI-generated SEO spam and low-quality content degrades "
                               "the quality of search results and training data",
                "current_severity": "high",
                "trend": "increasing",
                "detection_difficulty": "medium",
                "defenses": [
                    "Improved search ranking for authoritative sources",
                    "AI-generated content labeling requirements",
                    "Training data curation and filtering",
                    "Model collapse prevention",
                ],
            },
        ]

    def risk_assessment(self) -> dict:
        """Assess overall information ecosystem risk."""
        severity_map = {"low": 1, "medium": 2, "high": 3, "very high": 4}
        trend_map = {"decreasing": -1, "stable": 0, "increasing": 1,
                     "increasing rapidly": 2}

        total_risk = 0
        for threat in self.threat_vectors:
            severity = severity_map.get(threat["current_severity"], 0)
            trend = trend_map.get(threat["trend"], 0)
            total_risk += severity + trend

        max_possible = len(self.threat_vectors) * 6
        risk_score = total_risk / max_possible

        return {
            "overall_risk_score": risk_score,
            "risk_level": (
                "critical" if risk_score > 0.7
                else "high" if risk_score > 0.5
                else "medium" if risk_score > 0.3
                else "low"
            ),
            "num_threats": len(self.threat_vectors),
            "highest_severity": max(
                self.threat_vectors,
                key=lambda t: severity_map.get(t["current_severity"], 0),
            )["threat"],
        }


analysis = InformationEcosystemAnalysis()

print("=== Information Ecosystem Threat Analysis ===\n")
for threat in analysis.threat_vectors:
    print(f"Threat: {threat['threat']}")
    print(f"  Severity: {threat['current_severity']} | Trend: {threat['trend']}")
    print(f"  Detection: {threat['detection_difficulty']}")
    print(f"  Defenses: {threat['defenses'][0]}")
    print()

risk = analysis.risk_assessment()
print(f"Overall Risk: {risk['risk_level']} ({risk['overall_risk_score']:.2f})")
print(f"Highest threat: {risk['highest_severity']}")
```

---

## 4. 이중 용도 우려

### 4.1 유익한 응용 대 유해한 응용

```python
"""
Dual-use analysis: AI capabilities that serve both
beneficial and harmful purposes.
"""


class DualUseAnalysis:
    """Analyze dual-use risks of AI capabilities."""

    def __init__(self):
        self.capabilities = self._define_capabilities()

    def _define_capabilities(self) -> List[dict]:
        """Define AI capabilities with dual-use potential."""
        return [
            {
                "capability": "Protein structure prediction",
                "beneficial": "Drug discovery, disease understanding, agriculture",
                "harmful": "Bioweapon design, novel pathogen creation",
                "risk_level": "high",
                "current_controls": "Limited model access, community norms",
                "recommended_controls": "Biosecurity screening, structured access, "
                                        "international oversight",
            },
            {
                "capability": "Code generation",
                "beneficial": "Software development productivity, accessibility",
                "harmful": "Malware creation, vulnerability exploitation",
                "risk_level": "medium",
                "current_controls": "Output filtering, terms of service",
                "recommended_controls": "Code safety scanning, responsible "
                                        "disclosure integration",
            },
            {
                "capability": "Persuasive text generation",
                "beneficial": "Education, therapy, marketing, accessibility",
                "harmful": "Social engineering, political manipulation, scams",
                "risk_level": "high",
                "current_controls": "Content policies, rate limiting",
                "recommended_controls": "Watermarking, provenance tracking, "
                                        "anti-manipulation guardrails",
            },
            {
                "capability": "Autonomous agents",
                "beneficial": "Scientific research, accessibility assistants",
                "harmful": "Autonomous cyber attacks, financial manipulation",
                "risk_level": "high",
                "current_controls": "Human-in-the-loop requirements, capability limits",
                "recommended_controls": "Sandboxing, audit trails, kill switches, "
                                        "human oversight mandates",
            },
            {
                "capability": "Facial recognition",
                "beneficial": "Security, finding missing persons, accessibility",
                "harmful": "Mass surveillance, authoritarian control, stalking",
                "risk_level": "high",
                "current_controls": "Some jurisdictional bans, company policies",
                "recommended_controls": "Legal restrictions on use cases, "
                                        "consent requirements, bias auditing",
            },
            {
                "capability": "Chemistry synthesis planning",
                "beneficial": "Pharmaceutical development, materials science",
                "harmful": "Chemical weapon synthesis, drug manufacturing",
                "risk_level": "critical",
                "current_controls": "Limited availability, expert access only",
                "recommended_controls": "Strict access controls, chemical safety "
                                        "screening, international monitoring",
            },
        ]

    def risk_matrix(self) -> dict:
        """Generate a risk matrix for dual-use capabilities."""
        risk_levels = {"low": 0, "medium": 1, "high": 2, "critical": 3}

        matrix = {}
        for cap in self.capabilities:
            level = risk_levels.get(cap["risk_level"], 0)
            matrix[cap["capability"]] = {
                "risk_score": level,
                "risk_level": cap["risk_level"],
                "has_adequate_controls": cap["current_controls"] != "Limited",
                "gap": cap["recommended_controls"],
            }

        return matrix


dual_use = DualUseAnalysis()

print("=== Dual-Use Capability Analysis ===\n")
for cap in dual_use.capabilities:
    print(f"Capability: {cap['capability']} [{cap['risk_level'].upper()}]")
    print(f"  Beneficial: {cap['beneficial']}")
    print(f"  Harmful:    {cap['harmful']}")
    print(f"  Controls:   {cap['current_controls']}")
    print()
```

---

## 5. 민주주의적 함의

### 5.1 권력 비대칭과 감시

```python
"""
Democratic implications of AI: surveillance, power asymmetries,
and effects on democratic institutions.
"""


class DemocracyImpactAssessment:
    """Assess AI's impact on democratic institutions and processes."""

    def __init__(self):
        self.risk_areas = self._define_risks()

    def _define_risks(self) -> List[dict]:
        """Define risks to democracy from AI."""
        return [
            {
                "area": "Surveillance expansion",
                "description": "AI enables mass surveillance at low cost, "
                               "shifting the balance of power toward the state",
                "severity": "critical",
                "examples": [
                    "Facial recognition in public spaces",
                    "Social media monitoring for dissent",
                    "Predictive policing targeting minorities",
                ],
                "democratic_values_at_risk": [
                    "Privacy", "Freedom of assembly",
                    "Freedom of expression", "Due process",
                ],
            },
            {
                "area": "Electoral manipulation",
                "description": "AI-powered disinformation campaigns can target "
                               "specific voter segments with personalized messaging",
                "severity": "high",
                "examples": [
                    "Deepfake political ads",
                    "Micro-targeted disinformation",
                    "Bot networks for narrative amplification",
                ],
                "democratic_values_at_risk": [
                    "Free and fair elections", "Informed citizenry",
                    "Political equality",
                ],
            },
            {
                "area": "Concentration of power",
                "description": "AI capabilities concentrated in a few companies "
                               "gives them outsized influence over public discourse",
                "severity": "high",
                "examples": [
                    "Platform content moderation decisions",
                    "Algorithmic recommendation systems shaping opinions",
                    "Corporate lobbying power from AI profits",
                ],
                "democratic_values_at_risk": [
                    "Pluralism", "Democratic accountability",
                    "Equal political influence",
                ],
            },
            {
                "area": "Erosion of human agency",
                "description": "Increasing reliance on AI for decisions reduces "
                               "human autonomy and critical thinking",
                "severity": "medium",
                "examples": [
                    "AI-driven judicial decisions",
                    "Automated welfare eligibility",
                    "Algorithmic hiring and firing",
                ],
                "democratic_values_at_risk": [
                    "Human dignity", "Self-determination",
                    "Right to explanation", "Accountability",
                ],
            },
        ]

    def assess(self) -> dict:
        """Run comprehensive democracy impact assessment."""
        severity_map = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        total_severity = sum(
            severity_map.get(r["severity"], 0) for r in self.risk_areas
        )
        max_severity = len(self.risk_areas) * 4

        all_values_at_risk = set()
        for risk in self.risk_areas:
            all_values_at_risk.update(risk["democratic_values_at_risk"])

        return {
            "overall_risk": total_severity / max_severity,
            "num_risk_areas": len(self.risk_areas),
            "democratic_values_affected": len(all_values_at_risk),
            "values_list": sorted(all_values_at_risk),
            "highest_risk": max(
                self.risk_areas,
                key=lambda r: severity_map.get(r["severity"], 0),
            )["area"],
        }


assessment = DemocracyImpactAssessment()

print("=== Democracy Impact Assessment ===\n")
for risk in assessment.risk_areas:
    print(f"Risk Area: {risk['area']} [{risk['severity'].upper()}]")
    print(f"  {risk['description']}")
    print(f"  Values at risk: {', '.join(risk['democratic_values_at_risk'])}")
    print()

result = assessment.assess()
print(f"Overall democratic risk score: {result['overall_risk']:.2f}")
print(f"Democratic values affected: {result['democratic_values_affected']}")
print(f"Highest risk area: {result['highest_risk']}")
```

---

## 6. 환경적 영향

### 6.1 AI의 탄소 발자국

```python
"""
Environmental impact of AI: energy consumption, carbon emissions,
and water usage from training and inference.
"""


class AIEnvironmentalImpact:
    """Calculate and analyze the environmental impact of AI systems."""

    # Reference values for environmental impact estimation
    GPU_POWER_WATTS = {
        "A100": 400,
        "H100": 700,
        "H200": 700,
        "TPU_v4": 275,
    }

    # Carbon intensity by region (kg CO2e per kWh)
    CARBON_INTENSITY = {
        "us_average": 0.39,
        "us_virginia": 0.28,       # AWS us-east-1
        "us_oregon": 0.08,         # Low-carbon grid
        "eu_average": 0.28,
        "eu_sweden": 0.01,         # Nearly all renewable
        "china_average": 0.55,
    }

    # Water usage for cooling (liters per kWh)
    WATER_PER_KWH = {
        "air_cooled": 1.8,
        "water_cooled": 0.5,
        "hybrid": 1.0,
    }

    def estimate_training_impact(
        self,
        gpu_type: str,
        num_gpus: int,
        training_hours: float,
        region: str = "us_average",
        pue: float = 1.1,  # Power Usage Effectiveness
        cooling: str = "hybrid",
    ) -> dict:
        """Estimate environmental impact of model training.

        PUE (Power Usage Effectiveness): ratio of total facility
        power to IT equipment power. 1.0 = perfect efficiency.
        Typical data centers: 1.1-1.5.
        """
        gpu_power_w = self.GPU_POWER_WATTS.get(gpu_type, 400)
        carbon_intensity = self.CARBON_INTENSITY.get(region, 0.39)
        water_rate = self.WATER_PER_KWH.get(cooling, 1.0)

        # Energy calculation
        it_energy_kwh = num_gpus * gpu_power_w * training_hours / 1000
        total_energy_kwh = it_energy_kwh * pue

        # Carbon emissions
        carbon_kg = total_energy_kwh * carbon_intensity
        carbon_tonnes = carbon_kg / 1000

        # Water usage
        water_liters = total_energy_kwh * water_rate
        water_m3 = water_liters / 1000

        # Comparisons
        households_equivalent = total_energy_kwh / (10700 / 12)  # US monthly avg
        car_miles_equivalent = carbon_kg / 0.404  # kg CO2 per mile
        flights_ny_sf = carbon_kg / 900  # kg CO2 per flight

        return {
            "gpu_type": gpu_type,
            "num_gpus": num_gpus,
            "training_hours": training_hours,
            "region": region,
            "energy": {
                "it_energy_kwh": it_energy_kwh,
                "total_energy_kwh": total_energy_kwh,
                "pue": pue,
            },
            "carbon": {
                "carbon_kg": carbon_kg,
                "carbon_tonnes": carbon_tonnes,
            },
            "water": {
                "water_liters": water_liters,
                "water_m3": water_m3,
            },
            "equivalents": {
                "us_households_months": households_equivalent,
                "car_miles": car_miles_equivalent,
                "ny_sf_flights": flights_ny_sf,
            },
        }

    def estimate_inference_impact(
        self,
        gpu_type: str,
        num_gpus: int,
        queries_per_day: int,
        avg_latency_s: float,
        days: int = 365,
        region: str = "us_average",
    ) -> dict:
        """Estimate annual environmental impact of model inference."""
        gpu_power_w = self.GPU_POWER_WATTS.get(gpu_type, 400)
        carbon_intensity = self.CARBON_INTENSITY.get(region, 0.39)

        # GPU utilization during inference (varies with load)
        total_gpu_seconds_per_day = queries_per_day * avg_latency_s
        gpu_hours_per_day = total_gpu_seconds_per_day / 3600

        # Energy per day (all GPUs running, utilization varies)
        energy_per_day_kwh = num_gpus * gpu_power_w * 24 / 1000 * 1.1  # PUE
        energy_annual_kwh = energy_per_day_kwh * days

        carbon_annual_kg = energy_annual_kwh * carbon_intensity
        carbon_per_query_g = carbon_annual_kg * 1000 / (queries_per_day * days)

        return {
            "annual_energy_kwh": energy_annual_kwh,
            "annual_carbon_kg": carbon_annual_kg,
            "carbon_per_query_g": carbon_per_query_g,
            "queries_per_day": queries_per_day,
            "total_annual_queries": queries_per_day * days,
        }


# Calculate environmental impact
env = AIEnvironmentalImpact()

# Training impact examples
models = [
    ("GPT-4 class", "H100", 25000, 2160, "us_virginia"),    # ~90 days
    ("Llama 2 70B", "A100", 2048, 1440, "us_average"),       # ~60 days
    ("Small model 7B", "A100", 64, 720, "eu_sweden"),         # ~30 days
]

print("=== AI Training Environmental Impact ===\n")
for name, gpu, n_gpus, hours, region in models:
    impact = env.estimate_training_impact(gpu, n_gpus, hours, region)
    print(f"Model: {name}")
    print(f"  Energy: {impact['energy']['total_energy_kwh']:,.0f} kWh")
    print(f"  Carbon: {impact['carbon']['carbon_tonnes']:.1f} tonnes CO2e")
    print(f"  Water:  {impact['water']['water_m3']:.0f} m3")
    print(f"  Equiv:  {impact['equivalents']['ny_sf_flights']:.0f} NY-SF flights")
    print()

# Inference impact
print("=== AI Inference Environmental Impact (Annual) ===\n")
inference = env.estimate_inference_impact(
    "H100", 1000, 10_000_000, 0.5, 365, "us_average"
)
print(f"Service: 10M queries/day on 1000 H100s")
print(f"  Annual energy: {inference['annual_energy_kwh']:,.0f} kWh")
print(f"  Annual carbon: {inference['annual_carbon_kg']:,.0f} kg CO2e")
print(f"  Per query: {inference['carbon_per_query_g']:.2f}g CO2e")
```

---

## 7. 디지털 격차

### 7.1 AI 혜택의 불균등한 분배

```python
"""
Digital divide analysis: how AI benefits and risks
distribute unevenly across populations.
"""


class DigitalDivideAnalysis:
    """Analyze how AI affects different populations unevenly."""

    def __init__(self):
        self.dimensions = self._define_dimensions()

    def _define_dimensions(self) -> List[dict]:
        """Define dimensions of the AI digital divide."""
        return [
            {
                "dimension": "Geographic",
                "description": "AI development concentrated in US, China, EU; "
                               "Global South largely excluded from benefits",
                "indicators": {
                    "AI research papers (Global South share)": "< 10%",
                    "AI companies (Global South share)": "< 5%",
                    "AI training data in non-English languages": "< 15%",
                },
                "risks": [
                    "AI systems not designed for local contexts",
                    "Brain drain from developing countries",
                    "Dependency on foreign AI infrastructure",
                ],
                "recommendations": [
                    "Invest in local AI research capacity",
                    "Support multilingual model development",
                    "Technology transfer programs",
                ],
            },
            {
                "dimension": "Socioeconomic",
                "description": "AI productivity gains primarily benefit capital "
                               "owners and high-skill workers",
                "indicators": {
                    "AI-driven productivity gains captured by top 10%": "~65%",
                    "Low-skill job displacement rate": "2-3x high-skill rate",
                    "Cost of retraining for AI-displaced workers": "$10K-50K",
                },
                "risks": [
                    "Widening income inequality",
                    "Hollowing out of middle-class jobs",
                    "Reduced social mobility",
                ],
                "recommendations": [
                    "Universal basic income or negative income tax",
                    "Massive retraining programs",
                    "AI dividend / compute tax",
                    "Portable benefits for gig workers",
                ],
            },
            {
                "dimension": "Linguistic",
                "description": "AI systems perform significantly better in English "
                               "than in other languages",
                "indicators": {
                    "Languages with good LLM support": "~20 out of 7000+",
                    "English training data fraction": "> 90%",
                    "Performance gap (English vs low-resource)": "20-40% on benchmarks",
                },
                "risks": [
                    "Cultural and linguistic homogenization",
                    "Exclusion of minority language speakers",
                    "Loss of indigenous knowledge systems",
                ],
                "recommendations": [
                    "Fund low-resource language data collection",
                    "Support multilingual model research",
                    "Language preservation through AI tools",
                ],
            },
            {
                "dimension": "Accessibility",
                "description": "AI systems often not designed for people with "
                               "disabilities despite potential for inclusion",
                "indicators": {
                    "AI tools with accessibility features": "< 30%",
                    "Disability representation in AI training data": "minimal",
                    "AI assistive technology adoption": "growing but expensive",
                },
                "risks": [
                    "New barriers created by AI interfaces",
                    "Bias against disabled users in AI systems",
                    "Exclusion from AI-mediated services",
                ],
                "recommendations": [
                    "Accessibility-first AI design standards",
                    "Inclusive training data collection",
                    "Subsidized AI assistive technologies",
                ],
            },
        ]

    def compute_divide_index(self) -> dict:
        """Compute an AI digital divide index across dimensions."""
        scores = {}
        for dim in self.dimensions:
            # Simple severity scoring based on risk count and indicators
            severity = len(dim["risks"]) / 4  # Normalize
            scores[dim["dimension"]] = min(severity, 1.0)

        overall = np.mean(list(scores.values()))
        return {
            "dimension_scores": scores,
            "overall_divide_index": overall,
            "most_severe": max(scores, key=scores.get),
        }


divide = DigitalDivideAnalysis()

print("=== AI Digital Divide Analysis ===\n")
for dim in divide.dimensions:
    print(f"Dimension: {dim['dimension']}")
    print(f"  {dim['description']}")
    print(f"  Key risks:")
    for risk in dim["risks"]:
        print(f"    - {risk}")
    print(f"  Top recommendation: {dim['recommendations'][0]}")
    print()

index = divide.compute_divide_index()
print(f"Overall Digital Divide Index: {index['overall_divide_index']:.2f}")
print(f"Most severe dimension: {index['most_severe']}")
```

### 7.2 글로벌 남반구(Global South) 접근성과 AI 피해의 분배

AI 개발의 지리는 구조적 비대칭을 만든다: AI의 혜택 — 생산성 도구, 의료 진단, 교육 콘텐츠 — 은 안정적인 인터넷 접근과 기술적 리터러시를 갖춘 부유한 국가와 도시 중심에 집중되는 반면, 여러 유형의 피해는 불균형하게 개발도상국에 가중된다.

**데이터 추출**은 가장 잘 문서화된 비대칭 중 하나다. 대형 언어 모델은 인터넷에서 스크래핑된 텍스트로 학습되는데, 이는 디지털 발자국이 작은 언어를 체계적으로 과소 대표한다. 그 결과 아프리카, 동남아시아, 원주민 언어에서 현저히 낮은 성능을 보이는 영어 중심 모델이 만들어져 언어적 위계를 고착화한다. 한편 데이터 자체 — EU나 미국 저작권법에 대한 제한적 수단을 가진 커뮤니티로부터 종종 보상이나 동의 없이 스크래핑된 — 는 학습 조직에게만 귀속되는 가치를 생성한다.

**주석 노동**은 또 다른 차원이다. RLHF 및 콘텐츠 조정 주석 작업의 상당 부분은 종종 중개 플랫폼을 통해 케냐, 필리핀 및 기타 국가의 저임금 노동자들에게 외주화된다. 이 노동자들은 제한적인 심리적 지원으로 대량의 그래픽 및 충격적 콘텐츠에 노출되며, 서방 시장에서 프리미엄 가격에 판매되는 AI 제품의 안전성에 중심적인 그들의 노동을 반영하지 못하는 임금을 받는다.

**연결성 격차**는 AI 도구가 이론적으로 이용 가능하더라도 안정적인 인터넷 접근이 없는 약 26억 명에게는 실제로 접근 불가능하다는 것을 의미한다. 따라서 AI 혜택은 교육, 자본, 인프라에서 이미 유리한 인구로 제한될 위험이 있다.

**환경 비용(Environmental costs)**도 불균등하게 분배된다. 대형 파운데이션 모델 학습은 계산 집약적이다: GPT-4 학습은 약 2,000 tCO₂e로 추정된다(자동차 약 400대의 연간 배출량에 해당). 규모에서의 추론은 상당히 더 많다. 데이터 센터 냉각은 대량의 물을 소비한다 — Microsoft는 GPT-4 학습 중 서버 냉각을 위해 640만 리터의 담수를 소비했다고 보고했다 — 이는 많은 지역에서 압박을 받는 자원이다. 데이터 센터가 출력의 영향을 가장 많이 받는 지역이 아닌 저렴한 에너지와 토지가 있는 곳에 위치하는 경우가 많아, 기후 비용은 경제적 혜택과 지리적으로 분리된다. 효율적인 AI 연구 — 더 작은 모델, 지식 증류, 구조적 가지치기, 추론 최적화 — 는 따라서 접근 장벽을 줄이고 AI 개발의 탄소 발자국(Carbon Footprint)과 물 발자국을 줄이면서 환경적 명령이자 형평성 명령이기도 하다.

---

## 8. 문화적 영향

### 8.1 동질화와 창작

```python
"""
Cultural impacts of AI: effects on creativity, cultural diversity,
and the nature of human expression.
"""


def cultural_impact_analysis() -> dict:
    """Analyze AI's cultural impacts."""

    impacts = {
        "Creative Displacement": {
            "description": "AI generates art, music, writing at scale, "
                           "potentially devaluing human creative work",
            "affected_domains": ["Visual art", "Music composition",
                                 "Creative writing", "Photography",
                                 "Graphic design"],
            "positive_view": "Democratizes creativity, enables new art forms, "
                            "lowers barriers for amateurs",
            "negative_view": "Devalues human artists, homogenizes aesthetic, "
                            "creates copyright conflicts",
            "current_trend": "Rapid displacement of commercial creative work; "
                            "human art may become a premium/luxury category",
        },
        "Cultural Homogenization": {
            "description": "AI trained primarily on Western/English content "
                           "may promote a monoculture globally",
            "affected_domains": ["Language use", "Cultural norms",
                                 "Aesthetic preferences", "Value systems"],
            "positive_view": "Cross-cultural understanding, translation access",
            "negative_view": "Loss of cultural diversity, linguistic erosion, "
                            "Western cultural imperialism via AI",
            "current_trend": "English-centric AI dominance; growing awareness "
                            "but limited multilingual investment",
        },
        "Authenticity and Trust": {
            "description": "When any content could be AI-generated, the concept "
                           "of authenticity and trust fundamentally changes",
            "affected_domains": ["Journalism", "Academic research",
                                 "Personal communication", "Legal evidence"],
            "positive_view": "Forces development of better verification systems",
            "negative_view": "Erodes trust in all media; 'liar's dividend' - "
                            "real evidence dismissed as AI-generated",
            "current_trend": "Growing skepticism of all digital content; "
                            "provenance systems emerging but not widespread",
        },
        "Human-AI Relationships": {
            "description": "AI companions and assistants change the nature "
                           "of human social interaction",
            "affected_domains": ["Social interaction", "Education",
                                 "Mental health", "Companionship"],
            "positive_view": "Support for lonely and isolated individuals, "
                            "personalized education, therapy access",
            "negative_view": "Reduced human-to-human interaction, "
                            "parasocial relationships, manipulation potential",
            "current_trend": "Rapid growth of AI companionship apps; "
                            "limited research on long-term effects",
        },
    }

    print("=== Cultural Impact Analysis ===\n")
    for area, details in impacts.items():
        print(f"--- {area} ---")
        print(f"  {details['description']}")
        print(f"  Positive: {details['positive_view'][:60]}...")
        print(f"  Negative: {details['negative_view'][:60]}...")
        print(f"  Trend: {details['current_trend'][:60]}...")
        print()

    return impacts


cultural_impact_analysis()
```

---

## 9. 장기 경로와 정책 권고안

### 9.1 시나리오 계획

```python
"""
Long-term trajectory analysis and policy recommendations
for beneficial AI development.
"""


class LongTermTrajectoryAnalysis:
    """Analyze possible long-term trajectories for AI's societal impact."""

    def __init__(self):
        self.scenarios = self._define_scenarios()
        self.recommendations = self._define_recommendations()

    def _define_scenarios(self) -> List[dict]:
        """Define possible long-term scenarios."""
        return [
            {
                "scenario": "Shared Prosperity",
                "probability": "Possible but requires deliberate action",
                "description": "AI augments human capabilities broadly, "
                               "productivity gains are widely shared, "
                               "new jobs replace old ones, inequality decreases",
                "key_requirements": [
                    "Strong social safety nets and retraining",
                    "Progressive taxation of AI-driven profits",
                    "Investment in public AI infrastructure",
                    "International cooperation on AI governance",
                ],
                "key_risks": [
                    "Political will for redistribution",
                    "Speed of transition vs adaptation",
                ],
            },
            {
                "scenario": "AI Divide",
                "probability": "Default trajectory without intervention",
                "description": "AI benefits concentrate among those who control "
                               "the technology; inequality widens dramatically; "
                               "two-tier society emerges",
                "key_requirements": [
                    "No additional intervention needed (status quo)",
                ],
                "key_risks": [
                    "Social instability",
                    "Democratic erosion",
                    "Loss of social cohesion",
                ],
            },
            {
                "scenario": "Cautious Development",
                "probability": "Possible with strong governance",
                "description": "AI development slows due to regulation; "
                               "safety prioritized over capabilities; "
                               "benefits come more slowly but more safely",
                "key_requirements": [
                    "International agreement on safety standards",
                    "Binding regulation on frontier AI development",
                    "Strong enforcement mechanisms",
                ],
                "key_risks": [
                    "Competitive disadvantage for regulating countries",
                    "Innovation slowdown",
                    "Regulatory capture",
                ],
            },
            {
                "scenario": "Transformative AI",
                "probability": "Uncertain but non-negligible",
                "description": "AI surpasses human-level capabilities broadly; "
                               "fundamental restructuring of economy, governance, "
                               "and social institutions required",
                "key_requirements": [
                    "Solved alignment problem",
                    "Robust governance frameworks",
                    "Global coordination",
                    "New economic paradigm",
                ],
                "key_risks": [
                    "Misalignment catastrophe",
                    "Power concentration",
                    "Loss of human agency",
                    "Existential risk",
                ],
            },
        ]

    def _define_recommendations(self) -> List[dict]:
        """Define policy recommendations for beneficial AI development."""
        return [
            {
                "area": "Economic Policy",
                "recommendations": [
                    "Invest in massive retraining programs for displaced workers",
                    "Explore AI dividend or compute tax to fund social programs",
                    "Support small business AI adoption to prevent concentration",
                    "Fund public AI research to maintain competitive balance",
                ],
                "urgency": "high",
                "timeline": "1-5 years",
            },
            {
                "area": "Safety and Governance",
                "recommendations": [
                    "Establish international AI safety standards body",
                    "Require pre-deployment safety testing for high-risk AI",
                    "Fund independent AI safety research",
                    "Create mandatory incident reporting for AI systems",
                ],
                "urgency": "critical",
                "timeline": "immediate",
            },
            {
                "area": "Information Integrity",
                "recommendations": [
                    "Mandate provenance tracking for AI-generated content",
                    "Fund fact-checking infrastructure",
                    "Require transparency in algorithmic recommendation",
                    "Invest in media literacy education",
                ],
                "urgency": "high",
                "timeline": "1-3 years",
            },
            {
                "area": "Equity and Access",
                "recommendations": [
                    "Public investment in multilingual AI models",
                    "Technology transfer to developing countries",
                    "Accessibility requirements for AI systems",
                    "Data sovereignty frameworks for communities",
                ],
                "urgency": "medium",
                "timeline": "2-5 years",
            },
            {
                "area": "Democratic Protection",
                "recommendations": [
                    "Ban AI-powered mass surveillance without oversight",
                    "Require human oversight for high-stakes AI decisions",
                    "Transparency requirements for political AI use",
                    "Strengthen antitrust enforcement for AI markets",
                ],
                "urgency": "high",
                "timeline": "1-3 years",
            },
        ]

    def print_analysis(self):
        """Print the complete analysis."""
        print("=== Long-Term AI Trajectories ===\n")
        for scenario in self.scenarios:
            print(f"Scenario: {scenario['scenario']}")
            print(f"  Probability: {scenario['probability']}")
            print(f"  {scenario['description']}")
            print(f"  Key risks: {', '.join(scenario['key_risks'][:2])}")
            print()

        print("\n=== Policy Recommendations ===\n")
        for rec in self.recommendations:
            print(f"{rec['area']} [{rec['urgency'].upper()}] "
                  f"(Timeline: {rec['timeline']})")
            for r in rec["recommendations"]:
                print(f"  - {r}")
            print()


analysis = LongTermTrajectoryAnalysis()
analysis.print_analysis()
```

---

## 요약

- **노동 시장 효과(Labor Market Effects)**에는 대체(주로 저숙련 반복 업무), 증강(AI가 고숙련 근로자를 강화), AI 인접 역할의 창출이 포함됩니다. 순효과는 도입 속도와 정책 대응에 따라 달라집니다
- AI의 **경제적 집중(Economic Concentration)**은 컴퓨팅 비용, 데이터 우위, 인재 부족, 네트워크 효과에 의해 주도됩니다. 시장 집중도 지표(HHI)는 일부 AI 시장이 이미 고도로 집중되어 있음을 보여줍니다
- **정보 생태계 위험(Information Ecosystem Risks)**에는 대규모 합성 텍스트, 딥페이크, 맞춤형 조작, 인식론적 침식, 지식 오염이 포함됩니다
- **이중 용도 우려(Dual-Use Concerns)**는 AI 역량에 본질적입니다: 단백질 예측은 신약 발견과 생물무기 모두에 도움이 되고, 코드 생성은 개발과 악성코드 모두에 도움이 되며, 설득은 교육과 조작 모두에 도움이 됩니다
- **민주주의적 함의(Democratic Implications)**에는 감시 확대, 선거 조작, 권력 집중, 의사결정에서의 인간 주체성 침식이 포함됩니다
- AI 훈련의 **환경적 영향(Environmental Impact)**은 상당합니다(프론티어 모델: 수백 톤 CO2e). 하지만 대규모 추론은 훈련의 영향을 초과할 수 있습니다
- **디지털 격차(Digital Divide)**는 지리적, 사회경제적, 언어적, 접근성 차원에서 작동하며, AI 혜택은 부유한 영어 사용 인구에 집중됩니다
- **문화적 영향(Cultural Impacts)**에는 창작 대체, 동질화 압력, 진정성 침식, 변화하는 인간-AI 관계가 포함됩니다
- **정책 권고안(Policy Recommendations)**은 경제적 지원(재훈련, AI 배당금), 안전 거버넌스, 정보 무결성, 형평성 조치, 민주주의 보호에 걸쳐 있습니다

---

## 연습문제

### 연습문제 1: 노동 시장 영향 시뮬레이터

노동 시장 영향 시뮬레이터를 구축하세요:
1. 5개 부문에 걸쳐 AI 노출도, 증강 잠재력, 대체 위험이 있는 15개 직업을 정의하세요
2. 세 가지 시나리오를 모델링하세요: 느린 도입(20%), 보통(50%), 빠른(80%)
3. 각 시나리오에 대해 부문별, 기술 수준별 순고용 변화를 계산하세요
4. 가장 많이 영향받는 5개 직업과 가장 적게 영향받는 5개 직업을 식별하세요
5. 대체 근로자당 평균 $25K를 가정하여 재훈련 비용을 추정하세요

<details>
<summary>정답 보기</summary>

```python
"""
Labor market impact simulator across adoption scenarios.
"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class Job:
    name: str
    sector: str
    skill: str
    workers_k: int
    exposure: float
    augmentation: float
    displacement_risk: float


class LaborSimulator:
    def __init__(self):
        self.jobs = [
            Job("Software Engineer", "Tech", "high", 1800, 0.65, 0.80, 0.10),
            Job("Data Scientist", "Tech", "high", 200, 0.55, 0.85, 0.08),
            Job("IT Support", "Tech", "medium", 900, 0.70, 0.40, 0.45),
            Job("Accountant", "Finance", "high", 1400, 0.75, 0.70, 0.30),
            Job("Bank Teller", "Finance", "low", 400, 0.85, 0.10, 0.80),
            Job("Financial Advisor", "Finance", "high", 300, 0.50, 0.65, 0.15),
            Job("Doctor", "Healthcare", "high", 1000, 0.35, 0.70, 0.03),
            Job("Medical Coder", "Healthcare", "medium", 200, 0.85, 0.30, 0.65),
            Job("Home Health Aide", "Healthcare", "low", 3500, 0.15, 0.25, 0.05),
            Job("Teacher", "Education", "high", 3700, 0.40, 0.65, 0.05),
            Job("Tutor", "Education", "medium", 200, 0.70, 0.50, 0.35),
            Job("Admin Assistant", "Admin", "low", 3600, 0.80, 0.20, 0.70),
            Job("Receptionist", "Admin", "low", 1000, 0.85, 0.15, 0.75),
            Job("HR Specialist", "Admin", "medium", 700, 0.60, 0.55, 0.25),
            Job("Office Manager", "Admin", "medium", 400, 0.50, 0.45, 0.20),
        ]

    def simulate(self, adoption: float) -> dict:
        results = []
        for job in self.jobs:
            displaced = job.workers_k * job.displacement_risk * adoption
            augmented = job.workers_k * job.augmentation * adoption
            results.append({
                "name": job.name, "sector": job.sector, "skill": job.skill,
                "workers_k": job.workers_k, "displaced_k": displaced,
                "augmented_k": augmented,
                "pct_displaced": displaced / job.workers_k * 100,
            })

        results.sort(key=lambda x: -x["pct_displaced"])

        # Sector summary
        sectors = {}
        for r in results:
            s = r["sector"]
            if s not in sectors:
                sectors[s] = {"displaced": 0, "augmented": 0, "total": 0}
            sectors[s]["displaced"] += r["displaced_k"]
            sectors[s]["augmented"] += r["augmented_k"]
            sectors[s]["total"] += r["workers_k"]

        # Skill summary
        skills = {}
        for r in results:
            sk = r["skill"]
            if sk not in skills:
                skills[sk] = {"displaced": 0, "total": 0}
            skills[sk]["displaced"] += r["displaced_k"]
            skills[sk]["total"] += r["workers_k"]

        total_displaced = sum(r["displaced_k"] for r in results)

        return {
            "adoption": adoption,
            "total_displaced_k": total_displaced,
            "retraining_cost_m": total_displaced * 25,  # $25K per worker
            "occupations": results,
            "sectors": sectors,
            "skills": skills,
        }


sim = LaborSimulator()

for adoption in [0.2, 0.5, 0.8]:
    result = sim.simulate(adoption)
    print(f"\n=== Adoption: {adoption:.0%} ===")
    print(f"Total displaced: {result['total_displaced_k']:,.0f}K workers")
    print(f"Retraining cost: ${result['retraining_cost_m']:,.0f}M")

    print(f"\nMost affected:")
    for occ in result["occupations"][:5]:
        print(f"  {occ['name']:25s}: {occ['pct_displaced']:.1f}%")

    print(f"\nLeast affected:")
    for occ in result["occupations"][-5:]:
        print(f"  {occ['name']:25s}: {occ['pct_displaced']:.1f}%")

    print(f"\nBy skill level:")
    for sk, data in result["skills"].items():
        rate = data["displaced"] / data["total"] * 100
        print(f"  {sk:8s}: {rate:.1f}%")
```

</details>

### 연습문제 2: 정보 생태계 위험 평가

정보 생태계 위험 평가 도구를 생성하세요:
1. 6개의 위협 벡터(합성 텍스트, 딥페이크, 봇 등)를 정의하세요
2. 각각을 심각도, 탐지 난이도, 확장 잠재력, 방어 성숙도로 점수를 매기세요
3. 가중 차원으로 전체 위험 점수를 계산하세요
4. 심각도와 방어 성숙도 간 격차가 가장 큰 위협을 식별하세요
5. 격차 분석에 기반한 우선순위 대응책을 제안하세요

<details>
<summary>정답 보기</summary>

```python
"""
Information ecosystem risk assessment with gap analysis.
"""

import numpy as np
from typing import List, Dict


class InfoEcosystemRiskTool:
    """Information ecosystem risk assessment."""

    def __init__(self):
        self.threats = [
            {"name": "Synthetic news articles", "severity": 0.8,
             "detection": 0.4, "scale": 0.9, "defense": 0.3},
            {"name": "Deepfake video", "severity": 0.9,
             "detection": 0.5, "scale": 0.7, "defense": 0.4},
            {"name": "Social media bot networks", "severity": 0.7,
             "detection": 0.6, "scale": 0.95, "defense": 0.5},
            {"name": "Personalized manipulation", "severity": 0.85,
             "detection": 0.2, "scale": 0.8, "defense": 0.2},
            {"name": "Search/knowledge pollution", "severity": 0.75,
             "detection": 0.5, "scale": 0.9, "defense": 0.35},
            {"name": "Voice cloning / audio deepfakes", "severity": 0.8,
             "detection": 0.45, "scale": 0.6, "defense": 0.25},
        ]

    def assess(self) -> Dict:
        """Run risk assessment with gap analysis."""
        weights = {"severity": 0.35, "detection": 0.2,
                   "scale": 0.25, "defense": 0.2}

        results = []
        for threat in self.threats:
            # Risk = severity * scale * (1 - defense) * (1 - detection)
            risk = (threat["severity"] * threat["scale"]
                    * (1 - threat["defense"]) * (1 - threat["detection"]))
            gap = threat["severity"] - threat["defense"]
            results.append({
                **threat,
                "risk_score": risk,
                "defense_gap": gap,
            })

        results.sort(key=lambda x: -x["risk_score"])

        # Countermeasures prioritized by gap
        gap_sorted = sorted(results, key=lambda x: -x["defense_gap"])
        countermeasures = []
        for r in gap_sorted[:3]:
            countermeasures.append({
                "threat": r["name"],
                "gap": r["defense_gap"],
                "action": f"Invest in defense for {r['name']} "
                          f"(gap: {r['defense_gap']:.2f})",
            })

        return {
            "threat_rankings": results,
            "overall_risk": np.mean([r["risk_score"] for r in results]),
            "largest_gaps": countermeasures,
        }


tool = InfoEcosystemRiskTool()
result = tool.assess()

print("=== Information Ecosystem Risk Assessment ===\n")
print(f"{'Threat':<35} {'Risk':>6} {'Gap':>6} {'Defense':>8}")
print("-" * 60)
for t in result["threat_rankings"]:
    print(f"{t['name']:<35} {t['risk_score']:>6.3f} "
          f"{t['defense_gap']:>6.2f} {t['defense']:>8.2f}")

print(f"\nOverall risk: {result['overall_risk']:.3f}")
print(f"\nPrioritized countermeasures:")
for cm in result["largest_gaps"]:
    print(f"  - {cm['action']}")
```

</details>

### 연습문제 3: 환경 영향 계산기

AI 환경 영향 계산기를 구축하세요:
1. GPU 유형, 수량, 훈련 시간, 데이터센터 지역을 입력으로 받으세요
2. 에너지 소비, 탄소 배출, 물 사용량을 계산하세요
3. 등가 비교를 제공하세요 (자동차 마일, 항공편, 가구 월)
4. 4개 지역에서 동일한 훈련 실행을 비교하세요
5. 최저 영향 구성을 제안하고 절감량을 추정하세요

<details>
<summary>정답 보기</summary>

```python
"""
AI environmental impact calculator with regional comparison.
"""

import numpy as np


class EnvironmentalCalculator:
    """Calculate AI environmental impact across regions."""

    GPU_POWER_W = {"A100": 400, "H100": 700, "H200": 700}
    CARBON_KG_PER_KWH = {
        "US-Virginia": 0.28, "US-Oregon": 0.08,
        "EU-Sweden": 0.01, "China-Average": 0.55,
    }
    WATER_L_PER_KWH = 1.0  # Hybrid cooling average

    def calculate(self, gpu: str, n_gpus: int, hours: float,
                  region: str, pue: float = 1.1) -> dict:
        power = self.GPU_POWER_W.get(gpu, 400)
        carbon_rate = self.CARBON_KG_PER_KWH.get(region, 0.39)

        energy_kwh = n_gpus * power * hours / 1000 * pue
        carbon_kg = energy_kwh * carbon_rate
        water_l = energy_kwh * self.WATER_L_PER_KWH

        return {
            "region": region,
            "energy_kwh": energy_kwh,
            "carbon_kg": carbon_kg,
            "carbon_tonnes": carbon_kg / 1000,
            "water_liters": water_l,
            "equiv_car_miles": carbon_kg / 0.404,
            "equiv_flights_ny_sf": carbon_kg / 900,
            "equiv_household_months": energy_kwh / (10700 / 12),
        }

    def compare_regions(self, gpu: str, n_gpus: int, hours: float) -> list:
        results = []
        for region in self.CARBON_KG_PER_KWH:
            results.append(self.calculate(gpu, n_gpus, hours, region))
        results.sort(key=lambda x: x["carbon_kg"])
        return results


calc = EnvironmentalCalculator()

# Compare a frontier model training run across regions
gpu, n_gpus, hours = "H100", 4096, 2160
print(f"=== Training: {n_gpus}x {gpu} for {hours}h ===\n")

results = calc.compare_regions(gpu, n_gpus, hours)
print(f"{'Region':<20} {'Energy(MWh)':>12} {'Carbon(t)':>10} {'Water(m3)':>10}")
print("-" * 55)
for r in results:
    print(f"{r['region']:<20} {r['energy_kwh']/1000:>12,.0f} "
          f"{r['carbon_tonnes']:>10.1f} {r['water_liters']/1000:>10.0f}")

best = results[0]
worst = results[-1]
savings_t = worst["carbon_tonnes"] - best["carbon_tonnes"]
print(f"\nBest region: {best['region']} ({best['carbon_tonnes']:.1f}t CO2)")
print(f"Worst region: {worst['region']} ({worst['carbon_tonnes']:.1f}t CO2)")
print(f"Savings: {savings_t:.1f} tonnes CO2 "
      f"({savings_t/worst['carbon_tonnes']:.0%} reduction)")
print(f"Equivalent to {best['equiv_flights_ny_sf']:.0f} vs "
      f"{worst['equiv_flights_ny_sf']:.0f} NY-SF flights")
```

</details>

### 연습문제 4: 디지털 격차 지수

국가별 AI 디지털 격차 지수를 생성하세요:
1. 5개 지표를 정의하세요: AI 연구 역량, 인프라, 언어 지원, 교육, 정책
2. 각 지표에 대해 8개 국가/지역을 점수화하세요 (0-1 척도)
3. 종합 AI 준비 지수를 계산하세요
4. 가장 준비된 국가와 가장 덜 준비된 국가 간의 가장 큰 격차를 식별하세요
5. 가장 낮은 점수를 받은 3개 국가에 대한 맞춤형 개입을 제안하세요

<details>
<summary>정답 보기</summary>

```python
"""
AI Digital Divide Index across countries.
"""

import numpy as np
from typing import Dict, List


class DigitalDivideIndex:
    """Compute AI digital divide index for countries."""

    INDICATORS = [
        "AI research capacity",
        "Compute infrastructure",
        "Language support in AI",
        "AI education and talent",
        "AI governance readiness",
    ]

    WEIGHTS = [0.25, 0.20, 0.20, 0.20, 0.15]

    def __init__(self):
        self.countries = {
            "United States": [0.95, 0.95, 0.95, 0.90, 0.70],
            "China": [0.85, 0.85, 0.60, 0.80, 0.65],
            "United Kingdom": [0.80, 0.75, 0.90, 0.80, 0.75],
            "India": [0.50, 0.40, 0.45, 0.55, 0.35],
            "Brazil": [0.30, 0.35, 0.25, 0.35, 0.25],
            "Nigeria": [0.10, 0.10, 0.05, 0.15, 0.10],
            "South Korea": [0.70, 0.80, 0.50, 0.75, 0.60],
            "Kenya": [0.15, 0.15, 0.10, 0.20, 0.15],
        }

    def compute_index(self) -> Dict[str, dict]:
        """Compute composite index for each country."""
        results = {}
        for country, scores in self.countries.items():
            composite = np.average(scores, weights=self.WEIGHTS)
            weakest = self.INDICATORS[np.argmin(scores)]
            results[country] = {
                "composite": composite,
                "scores": dict(zip(self.INDICATORS, scores)),
                "weakest_area": weakest,
            }
        return dict(sorted(results.items(),
                           key=lambda x: -x[1]["composite"]))

    def gap_analysis(self, results: dict) -> dict:
        """Analyze gaps between top and bottom countries."""
        composites = [r["composite"] for r in results.values()]
        countries = list(results.keys())
        top = countries[0]
        bottom = countries[-1]
        gap = results[top]["composite"] - results[bottom]["composite"]

        # Per-indicator gaps
        indicator_gaps = {}
        for i, ind in enumerate(self.INDICATORS):
            top_score = self.countries[top][i]
            bottom_score = self.countries[bottom][i]
            indicator_gaps[ind] = top_score - bottom_score

        return {
            "top_country": top,
            "bottom_country": bottom,
            "overall_gap": gap,
            "indicator_gaps": indicator_gaps,
            "largest_gap_indicator": max(indicator_gaps, key=indicator_gaps.get),
        }

    def interventions(self, results: dict) -> Dict[str, List[str]]:
        """Propose interventions for lowest-scoring countries."""
        bottom_3 = list(results.keys())[-3:]
        interventions = {}
        for country in bottom_3:
            r = results[country]
            actions = []
            for ind, score in r["scores"].items():
                if score < 0.3:
                    if "research" in ind:
                        actions.append(f"Fund AI research centers and partnerships")
                    elif "infrastructure" in ind:
                        actions.append(f"Invest in cloud compute access programs")
                    elif "language" in ind:
                        actions.append(f"Develop local language AI datasets")
                    elif "education" in ind:
                        actions.append(f"AI curriculum in universities and bootcamps")
                    elif "governance" in ind:
                        actions.append(f"Develop national AI strategy and policy")
            interventions[country] = actions
        return interventions


index = DigitalDivideIndex()
results = index.compute_index()

print("=== AI Digital Divide Index ===\n")
print(f"{'Country':<20} {'Index':>6} {'Weakest Area'}")
print("-" * 55)
for country, data in results.items():
    print(f"{country:<20} {data['composite']:>6.2f} {data['weakest_area']}")

gaps = index.gap_analysis(results)
print(f"\nOverall gap: {gaps['overall_gap']:.2f} "
      f"({gaps['top_country']} vs {gaps['bottom_country']})")
print(f"Largest indicator gap: {gaps['largest_gap_indicator']}")

print(f"\n=== Proposed Interventions ===")
intv = index.interventions(results)
for country, actions in intv.items():
    print(f"\n{country}:")
    for a in actions:
        print(f"  - {a}")
```

</details>

### 연습문제 5: 종합 사회적 영향 평가

가상의 AI 배포에 대한 종합 사회적 영향 평가를 구축하세요:
1. 이 레슨의 8개 차원 모두에서 영향을 평가하세요
2. 각 차원을 -5(매우 유해)에서 +5(매우 유익) 척도로 점수를 매기세요
3. 영향받는 인구에 의한 차원 가중치를 적용하세요
4. 전체 사회적 영향 점수를 계산하세요
5. 혜택과 위험을 모두 포함한 균형 잡힌 평가 보고서를 생성하세요

<details>
<summary>정답 보기</summary>

```python
"""
Comprehensive societal impact assessment.
"""

import numpy as np
from typing import Dict


class SocietalImpactAssessment:
    """Assess overall societal impact of an AI deployment."""

    DIMENSIONS = [
        "Labor Market", "Economic Concentration", "Information Ecosystem",
        "Dual-Use Risk", "Democratic Impact", "Environmental",
        "Digital Divide", "Cultural Impact",
    ]

    # Weight by population affected (relative)
    WEIGHTS = [0.20, 0.10, 0.15, 0.10, 0.15, 0.10, 0.10, 0.10]

    def assess(self, deployment: str, scores: Dict[str, float]) -> dict:
        """Run assessment. Scores: -5 (very harmful) to +5 (very beneficial)."""
        results = []
        for dim, weight in zip(self.DIMENSIONS, self.WEIGHTS):
            score = scores.get(dim, 0)
            results.append({
                "dimension": dim,
                "score": score,
                "weight": weight,
                "weighted": score * weight,
                "assessment": (
                    "Very beneficial" if score > 3
                    else "Beneficial" if score > 1
                    else "Neutral" if score > -1
                    else "Harmful" if score > -3
                    else "Very harmful"
                ),
            })

        overall = sum(r["weighted"] for r in results)
        benefits = [r for r in results if r["score"] > 0]
        risks = [r for r in results if r["score"] < 0]

        return {
            "deployment": deployment,
            "overall_score": overall,
            "overall_assessment": (
                "Net positive" if overall > 0.5
                else "Net negative" if overall < -0.5
                else "Neutral/mixed"
            ),
            "dimensions": results,
            "top_benefits": sorted(benefits, key=lambda x: -x["score"])[:3],
            "top_risks": sorted(risks, key=lambda x: x["score"])[:3],
        }

    def report(self, result: dict) -> str:
        lines = [
            "=" * 60,
            f"  SOCIETAL IMPACT ASSESSMENT",
            f"  Deployment: {result['deployment']}",
            f"  Overall: {result['overall_score']:+.2f} ({result['overall_assessment']})",
            "=" * 60,
            "",
            f"{'Dimension':<25} {'Score':>6} {'Assessment'}",
            "-" * 50,
        ]
        for d in result["dimensions"]:
            lines.append(f"{d['dimension']:<25} {d['score']:>+6.1f} {d['assessment']}")

        lines.append("\nKey Benefits:")
        for b in result["top_benefits"]:
            lines.append(f"  + {b['dimension']} ({b['score']:+.1f})")
        lines.append("\nKey Risks:")
        for r in result["top_risks"]:
            lines.append(f"  - {r['dimension']} ({r['score']:+.1f})")

        return "\n".join(lines)


assessor = SocietalImpactAssessment()

# Assess a large-scale AI coding assistant deployment
result = assessor.assess("Global AI Coding Assistant (100M users)", {
    "Labor Market": -1.5,           # Some displacement, strong augmentation
    "Economic Concentration": -2.0,  # Benefits big tech primarily
    "Information Ecosystem": 1.0,    # Better code documentation
    "Dual-Use Risk": -1.0,          # Potential for malware
    "Democratic Impact": 0.5,        # Slight democratization of development
    "Environmental": -1.5,           # Significant compute
    "Digital Divide": -1.0,          # Primarily benefits English speakers
    "Cultural Impact": 1.0,          # New creative possibilities
})

print(assessor.report(result))
```

</details>

---

[이전: 책임감 있는 배포](./14_Responsible_Deployment.md) | [개요](./00_Overview.md) | [다음: 미해결 문제](./16_Open_Problems.md)

**License**: CC BY-NC 4.0
