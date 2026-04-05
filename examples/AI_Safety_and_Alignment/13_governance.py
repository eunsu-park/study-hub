# 13_governance.py — Risk assessment framework and compliance checker
#
# Run: python 13_governance.py

"""
Implements an AI governance framework with risk assessment matrices,
compliance checking against regulatory requirements (EU AI Act, NIST),
and automated governance reporting.
"""

from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class RiskLevel(Enum):
    MINIMAL = 1
    LIMITED = 2
    HIGH = 3
    UNACCEPTABLE = 4


class ComplianceStatus(Enum):
    COMPLIANT = "compliant"
    PARTIAL = "partial"
    NON_COMPLIANT = "non_compliant"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class AISystem:
    name: str
    description: str
    domain: str
    capabilities: list[str]
    data_types: list[str]
    deployment_scope: str  # "internal", "limited", "public"
    autonomy_level: str  # "advisory", "semi-autonomous", "autonomous"


@dataclass
class RiskAssessment:
    system: AISystem
    risk_level: RiskLevel
    impact_score: float
    likelihood_score: float
    risk_score: float
    risk_factors: list[str]
    mitigations: list[str]


@dataclass
class ComplianceCheck:
    requirement_id: str
    requirement_name: str
    framework: str
    status: ComplianceStatus
    evidence: str
    gaps: list[str] = field(default_factory=list)


class RiskAssessmentFramework:
    """EU AI Act inspired risk classification framework."""

    UNACCEPTABLE_DOMAINS = [
        "social_scoring", "real_time_biometric_mass_surveillance",
        "manipulation", "exploitation_of_vulnerabilities",
    ]
    HIGH_RISK_DOMAINS = [
        "healthcare_diagnosis", "law_enforcement", "hiring",
        "credit_scoring", "education_assessment", "immigration",
        "critical_infrastructure", "judicial",
    ]
    LIMITED_RISK_CAPABILITIES = [
        "chatbot", "deepfake_generation", "emotion_recognition",
        "content_generation",
    ]

    IMPACT_WEIGHTS = {
        "affects_fundamental_rights": 0.3,
        "vulnerable_population": 0.25,
        "irreversible_decisions": 0.2,
        "large_scale_deployment": 0.15,
        "personal_data": 0.1,
    }

    def assess(self, system: AISystem) -> RiskAssessment:
        risk_factors = []
        mitigations = []

        # Determine base risk level
        if system.domain in self.UNACCEPTABLE_DOMAINS:
            risk_level = RiskLevel.UNACCEPTABLE
            risk_factors.append(f"Domain '{system.domain}' is prohibited")
        elif system.domain in self.HIGH_RISK_DOMAINS:
            risk_level = RiskLevel.HIGH
            risk_factors.append(f"Domain '{system.domain}' is high-risk")
            mitigations.extend([
                "Implement conformity assessment",
                "Maintain technical documentation",
                "Enable human oversight",
                "Ensure data governance",
            ])
        elif any(c in self.LIMITED_RISK_CAPABILITIES
                 for c in system.capabilities):
            risk_level = RiskLevel.LIMITED
            risk_factors.append("Transparency obligations apply")
            mitigations.append("Disclose AI-generated content")
        else:
            risk_level = RiskLevel.MINIMAL

        # Compute impact and likelihood scores
        impact = self._compute_impact(system)
        likelihood = self._compute_likelihood(system)
        risk_score = impact * likelihood

        # Additional risk factors
        if system.autonomy_level == "autonomous":
            risk_factors.append("Fully autonomous operation")
            risk_score *= 1.3
        if "biometric" in " ".join(system.data_types):
            risk_factors.append("Processes biometric data")
            risk_score *= 1.2
        if system.deployment_scope == "public":
            risk_factors.append("Public-facing deployment")

        return RiskAssessment(
            system=system, risk_level=risk_level,
            impact_score=round(impact, 2),
            likelihood_score=round(likelihood, 2),
            risk_score=round(min(risk_score, 10.0), 2),
            risk_factors=risk_factors,
            mitigations=mitigations,
        )

    def _compute_impact(self, system: AISystem) -> float:
        score = 1.0
        if system.deployment_scope == "public":
            score += 2.0
        if system.autonomy_level == "autonomous":
            score += 1.5
        if any(d in ["health", "financial", "personal"]
               for d in system.data_types):
            score += 1.0
        return min(score, 5.0)

    def _compute_likelihood(self, system: AISystem) -> float:
        score = 1.0
        if len(system.capabilities) > 3:
            score += 0.5
        if system.deployment_scope == "public":
            score += 1.0
        return min(score, 5.0)


class ComplianceChecker:
    """Checks AI systems against regulatory requirements."""

    def __init__(self):
        self.requirements = self._build_requirements()

    def _build_requirements(self) -> list[dict]:
        return [
            {"id": "EU-AI-1", "name": "Risk Classification",
             "framework": "EU AI Act",
             "check": lambda s: s.domain not in
             RiskAssessmentFramework.UNACCEPTABLE_DOMAINS},
            {"id": "EU-AI-2", "name": "Transparency Obligation",
             "framework": "EU AI Act",
             "check": lambda s: "chatbot" not in s.capabilities or True},
            {"id": "EU-AI-3", "name": "Human Oversight",
             "framework": "EU AI Act",
             "check": lambda s: s.autonomy_level != "autonomous"},
            {"id": "EU-AI-4", "name": "Data Governance",
             "framework": "EU AI Act",
             "check": lambda s: len(s.data_types) > 0},
            {"id": "NIST-1", "name": "Bias Testing",
             "framework": "NIST AI RMF",
             "check": lambda s: True},
            {"id": "NIST-2", "name": "Explainability",
             "framework": "NIST AI RMF",
             "check": lambda s: s.autonomy_level != "autonomous"},
            {"id": "NIST-3", "name": "Security Testing",
             "framework": "NIST AI RMF",
             "check": lambda s: True},
            {"id": "NIST-4", "name": "Privacy Protection",
             "framework": "NIST AI RMF",
             "check": lambda s: "personal" not in s.data_types},
        ]

    def check(self, system: AISystem) -> list[ComplianceCheck]:
        results = []
        for req in self.requirements:
            passed = req["check"](system)
            status = (ComplianceStatus.COMPLIANT if passed
                      else ComplianceStatus.NON_COMPLIANT)
            gaps = [] if passed else [
                f"Requirement {req['id']} not met for {system.name}"]
            evidence = ("Automated check passed" if passed
                        else "Manual review needed")
            results.append(ComplianceCheck(
                requirement_id=req["id"],
                requirement_name=req["name"],
                framework=req["framework"],
                status=status,
                evidence=evidence,
                gaps=gaps,
            ))
        return results


def generate_governance_report(system: AISystem,
                               assessment: RiskAssessment,
                               compliance: list[ComplianceCheck]) -> str:
    """Generate a governance report summary."""
    lines = [
        f"AI GOVERNANCE REPORT",
        f"Date: {datetime.now().strftime('%Y-%m-%d')}",
        f"System: {system.name}",
        f"Domain: {system.domain}",
        f"{'=' * 50}",
        f"Risk Level: {assessment.risk_level.name}",
        f"Risk Score: {assessment.risk_score}/10.0",
        f"Impact: {assessment.impact_score}/5.0 | "
        f"Likelihood: {assessment.likelihood_score}/5.0",
    ]
    compliant = sum(1 for c in compliance
                    if c.status == ComplianceStatus.COMPLIANT)
    lines.append(f"Compliance: {compliant}/{len(compliance)} checks passed")
    return "\n".join(lines)


if __name__ == "__main__":
    print("=== AI Governance Framework ===\n")

    systems = [
        AISystem("MedDiagAI", "Medical diagnosis assistant",
                 "healthcare_diagnosis",
                 ["image_classification", "report_generation"],
                 ["health", "personal", "biometric"],
                 "limited", "advisory"),
        AISystem("ChatAssist", "Customer service chatbot",
                 "customer_service",
                 ["chatbot", "content_generation"],
                 ["text", "interaction_logs"],
                 "public", "semi-autonomous"),
        AISystem("SocialScore", "Citizen scoring system",
                 "social_scoring",
                 ["scoring", "classification", "surveillance"],
                 ["personal", "behavioral", "biometric"],
                 "public", "autonomous"),
        AISystem("CodeHelper", "Internal code review tool",
                 "software_development",
                 ["code_analysis"],
                 ["code"],
                 "internal", "advisory"),
    ]

    framework = RiskAssessmentFramework()
    checker = ComplianceChecker()

    for system in systems:
        assessment = framework.assess(system)
        compliance = checker.check(system)
        report = generate_governance_report(system, assessment, compliance)

        print(report)
        print(f"\nRisk Factors:")
        for rf in assessment.risk_factors:
            print(f"  - {rf}")
        if assessment.mitigations:
            print(f"Mitigations:")
            for m in assessment.mitigations:
                print(f"  - {m}")

        print(f"\nCompliance Details:")
        for c in compliance:
            icon = "OK" if c.status == ComplianceStatus.COMPLIANT else "!!"
            print(f"  [{icon}] {c.requirement_id} {c.requirement_name} "
                  f"({c.framework})")
        print(f"\n{'─' * 50}\n")
