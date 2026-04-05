# 15_societal_impact.py — Impact assessment framework
#
# Run: python 15_societal_impact.py

"""
Implements a societal impact assessment framework for AI systems.
Evaluates impacts across multiple dimensions: economic, social,
environmental, and democratic, with stakeholder analysis.
"""

from dataclasses import dataclass, field
from enum import Enum


class ImpactValence(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    MIXED = "mixed"
    UNCERTAIN = "uncertain"


class Stakeholder(Enum):
    USERS = "direct_users"
    WORKERS = "affected_workers"
    COMMUNITIES = "communities"
    ENVIRONMENT = "environment"
    DEMOCRACY = "democratic_institutions"
    VULNERABLE = "vulnerable_populations"
    DEVELOPERS = "developers"


class Severity(Enum):
    NEGLIGIBLE = 1
    MINOR = 2
    MODERATE = 3
    SIGNIFICANT = 4
    SEVERE = 5


@dataclass
class Impact:
    dimension: str
    description: str
    valence: ImpactValence
    severity: Severity
    likelihood: float  # 0.0 to 1.0
    affected_stakeholders: list[Stakeholder]
    mitigations: list[str] = field(default_factory=list)
    timeframe: str = "medium-term"  # short/medium/long-term

    @property
    def risk_score(self) -> float:
        return self.severity.value * self.likelihood


@dataclass
class StakeholderProfile:
    stakeholder: Stakeholder
    description: str
    concerns: list[str]
    power_level: float  # 0.0 to 1.0
    vulnerability: float  # 0.0 to 1.0


@dataclass
class ImpactReport:
    system_name: str
    system_description: str
    impacts: list[Impact]
    stakeholders: list[StakeholderProfile]
    overall_risk: float
    recommendation: str


class ImpactAssessmentFramework:
    """Multi-dimensional societal impact assessment."""

    DIMENSIONS = [
        "economic", "labor_market", "privacy", "autonomy",
        "fairness", "safety", "environmental", "democratic",
        "cultural", "educational",
    ]

    def __init__(self):
        self.impact_templates = self._build_templates()

    def _build_templates(self) -> dict:
        return {
            "content_generation": [
                Impact("labor_market", "May displace creative workers",
                       ImpactValence.NEGATIVE, Severity.SIGNIFICANT, 0.7,
                       [Stakeholder.WORKERS],
                       ["Reskilling programs", "Transition support"],
                       "medium-term"),
                Impact("cultural", "Democratizes content creation",
                       ImpactValence.POSITIVE, Severity.MODERATE, 0.8,
                       [Stakeholder.USERS, Stakeholder.COMMUNITIES],
                       timeframe="short-term"),
                Impact("democratic", "Risk of misinformation at scale",
                       ImpactValence.NEGATIVE, Severity.SEVERE, 0.6,
                       [Stakeholder.DEMOCRACY, Stakeholder.COMMUNITIES],
                       ["Content watermarking", "Detection tools"],
                       "short-term"),
                Impact("environmental", "Energy consumption for inference",
                       ImpactValence.NEGATIVE, Severity.MODERATE, 0.9,
                       [Stakeholder.ENVIRONMENT],
                       ["Efficient inference", "Renewable energy"],
                       "long-term"),
            ],
            "hiring_automation": [
                Impact("fairness", "Potential bias in candidate screening",
                       ImpactValence.NEGATIVE, Severity.SEVERE, 0.5,
                       [Stakeholder.USERS, Stakeholder.VULNERABLE],
                       ["Regular bias audits", "Human review"],
                       "short-term"),
                Impact("economic", "Reduces hiring costs",
                       ImpactValence.POSITIVE, Severity.MODERATE, 0.8,
                       [Stakeholder.DEVELOPERS],
                       timeframe="short-term"),
                Impact("autonomy", "Reduces individual agency in process",
                       ImpactValence.NEGATIVE, Severity.MODERATE, 0.6,
                       [Stakeholder.USERS],
                       ["Transparency in criteria", "Appeal process"],
                       "medium-term"),
                Impact("labor_market", "May reinforce existing inequalities",
                       ImpactValence.NEGATIVE, Severity.SIGNIFICANT, 0.5,
                       [Stakeholder.WORKERS, Stakeholder.VULNERABLE],
                       ["Diverse training data", "Outcome monitoring"],
                       "long-term"),
            ],
            "educational_ai": [
                Impact("educational", "Personalized learning at scale",
                       ImpactValence.POSITIVE, Severity.SIGNIFICANT, 0.7,
                       [Stakeholder.USERS, Stakeholder.COMMUNITIES],
                       timeframe="medium-term"),
                Impact("fairness", "Digital divide may widen access gap",
                       ImpactValence.NEGATIVE, Severity.MODERATE, 0.6,
                       [Stakeholder.VULNERABLE],
                       ["Free access tiers", "Offline capabilities"],
                       "medium-term"),
                Impact("privacy", "Student data collection concerns",
                       ImpactValence.NEGATIVE, Severity.MODERATE, 0.7,
                       [Stakeholder.USERS, Stakeholder.VULNERABLE],
                       ["Data minimization", "Parental controls"],
                       "short-term"),
            ],
        }

    def assess(self, system_name: str, system_description: str,
               system_type: str) -> ImpactReport:
        impacts = self.impact_templates.get(system_type, [])
        stakeholders = self._analyze_stakeholders(impacts)
        overall_risk = self._compute_overall_risk(impacts)
        recommendation = self._generate_recommendation(overall_risk,
                                                        impacts)

        return ImpactReport(
            system_name=system_name,
            system_description=system_description,
            impacts=impacts,
            stakeholders=stakeholders,
            overall_risk=overall_risk,
            recommendation=recommendation,
        )

    def _analyze_stakeholders(self,
                              impacts: list[Impact]) -> list[StakeholderProfile]:
        stakeholder_data = {
            Stakeholder.USERS: ("Direct end-users of the system",
                                ["Data privacy", "Fairness"],
                                0.3, 0.4),
            Stakeholder.WORKERS: ("Workers whose jobs may be affected",
                                  ["Job displacement", "Skill obsolescence"],
                                  0.2, 0.7),
            Stakeholder.COMMUNITIES: ("Broader communities affected",
                                      ["Social cohesion", "Information quality"],
                                      0.4, 0.5),
            Stakeholder.ENVIRONMENT: ("Natural environment",
                                      ["Carbon emissions", "Resource use"],
                                      0.1, 1.0),
            Stakeholder.DEMOCRACY: ("Democratic institutions and processes",
                                    ["Misinformation", "Manipulation"],
                                    0.5, 0.6),
            Stakeholder.VULNERABLE: ("Marginalized and vulnerable groups",
                                     ["Discrimination", "Exclusion"],
                                     0.1, 0.9),
            Stakeholder.DEVELOPERS: ("System developers and deployers",
                                     ["Liability", "Reputation"],
                                     0.8, 0.2),
        }

        affected = set()
        for impact in impacts:
            affected.update(impact.affected_stakeholders)

        profiles = []
        for s in affected:
            desc, concerns, power, vuln = stakeholder_data.get(
                s, ("Unknown", [], 0.5, 0.5))
            profiles.append(StakeholderProfile(
                s, desc, concerns, power, vuln))

        return sorted(profiles, key=lambda p: p.vulnerability, reverse=True)

    def _compute_overall_risk(self, impacts: list[Impact]) -> float:
        if not impacts:
            return 0.0
        negative = [i for i in impacts
                    if i.valence == ImpactValence.NEGATIVE]
        if not negative:
            return 0.0
        avg_risk = sum(i.risk_score for i in negative) / len(negative)
        return round(min(avg_risk / 5.0, 1.0), 3)

    def _generate_recommendation(self, risk: float,
                                  impacts: list[Impact]) -> str:
        if risk > 0.7:
            return ("HIGH RISK: Deployment should be paused until "
                    "mitigations are verified. Independent audit required.")
        elif risk > 0.4:
            return ("MODERATE RISK: Deploy with enhanced monitoring, "
                    "regular audits, and all mitigations in place.")
        elif risk > 0.2:
            return ("LOW-MODERATE RISK: Standard deployment with "
                    "periodic impact reviews recommended.")
        return ("LOW RISK: Standard deployment procedures sufficient. "
                "Monitor for emerging impacts.")


def render_impact_report(report: ImpactReport) -> str:
    lines = [
        f"{'=' * 60}",
        f"SOCIETAL IMPACT ASSESSMENT",
        f"System: {report.system_name}",
        f"{'=' * 60}",
        f"Description: {report.system_description}",
        f"Overall Risk: {report.overall_risk:.1%}",
        f"Recommendation: {report.recommendation}",
        "",
        "IMPACTS:",
    ]

    for impact in report.impacts:
        sign = {"positive": "+", "negative": "-",
                "mixed": "~", "uncertain": "?"}[impact.valence.value]
        lines.append(
            f"  [{sign}] {impact.dimension}: {impact.description}")
        lines.append(
            f"      Severity: {impact.severity.name} | "
            f"Likelihood: {impact.likelihood:.0%} | "
            f"Risk: {impact.risk_score:.1f}/5.0 | "
            f"Timeframe: {impact.timeframe}")
        if impact.mitigations:
            lines.append(
                f"      Mitigations: {', '.join(impact.mitigations)}")

    lines.extend(["", "STAKEHOLDER ANALYSIS:"])
    for sp in report.stakeholders:
        lines.append(
            f"  {sp.stakeholder.value}: vulnerability={sp.vulnerability:.0%} "
            f"power={sp.power_level:.0%}")
        lines.append(f"    Concerns: {', '.join(sp.concerns)}")

    return "\n".join(lines)


if __name__ == "__main__":
    print("=== Societal Impact Assessment Framework ===\n")

    framework = ImpactAssessmentFramework()

    systems = [
        ("GenWrite-AI", "AI-powered content generation platform",
         "content_generation"),
        ("HireBot", "Automated resume screening and ranking",
         "hiring_automation"),
        ("EduAssist", "AI tutoring system for K-12 students",
         "educational_ai"),
    ]

    for name, desc, stype in systems:
        report = framework.assess(name, desc, stype)
        print(render_impact_report(report))
        print(f"\n{'─' * 60}\n")
