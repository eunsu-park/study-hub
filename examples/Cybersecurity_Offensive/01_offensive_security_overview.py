"""
Example: Offensive Security Overview
=====================================
Rules of Engagement generator, attack surface model, engagement planning.

This module demonstrates key concepts from the Offensive Security Overview
lesson in the Cybersecurity Offensive topic.

IMPORTANT: All techniques are for authorized security testing,
CTF competitions, and educational purposes only.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Attack Surface Modeling
# ---------------------------------------------------------------------------

class AttackVector(Enum):
    """Categories of attack vectors."""
    NETWORK = "Network-based"
    WEB_APP = "Web Application"
    SOCIAL = "Social Engineering"
    PHYSICAL = "Physical Access"
    SUPPLY_CHAIN = "Supply Chain"
    WIRELESS = "Wireless"


class Severity(Enum):
    CRITICAL = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1
    INFO = 0


@dataclass
class AttackSurface:
    """Represents a potential entry point into a target system."""
    name: str
    vector: AttackVector
    description: str
    severity: Severity
    mitigations: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)

    def risk_score(self) -> float:
        """Calculate a simplified risk score (0-10)."""
        base = self.severity.value * 2.5
        mitigation_factor = max(0.3, 1.0 - len(self.mitigations) * 0.15)
        return round(min(10.0, base * mitigation_factor), 1)


# ---------------------------------------------------------------------------
# Rules of Engagement
# ---------------------------------------------------------------------------

class TestType(Enum):
    BLACK_BOX = "Black Box"
    GREY_BOX = "Grey Box"
    WHITE_BOX = "White Box"


@dataclass
class RulesOfEngagement:
    """Formal rules of engagement for a penetration test."""
    client_name: str
    tester_name: str
    test_type: TestType
    start_date: datetime
    end_date: datetime
    in_scope_targets: list[str]
    out_of_scope_targets: list[str]
    emergency_contact: str = ""
    allowed_techniques: list[str] = field(default_factory=list)
    prohibited_techniques: list[str] = field(default_factory=list)

    def generate_document(self) -> str:
        """Generate a formatted RoE document."""
        doc = [
            "=" * 60,
            "RULES OF ENGAGEMENT — PENETRATION TESTING",
            "=" * 60,
            f"Client: {self.client_name}",
            f"Tester: {self.tester_name}",
            f"Type: {self.test_type.value}",
            f"Period: {self.start_date:%Y-%m-%d} to {self.end_date:%Y-%m-%d}",
            f"Emergency: {self.emergency_contact}",
            "",
            "IN-SCOPE:", *[f"  [+] {t}" for t in self.in_scope_targets],
            "OUT-OF-SCOPE:", *[f"  [-] {t}" for t in self.out_of_scope_targets],
        ]
        if self.allowed_techniques:
            doc += ["ALLOWED:", *[f"  [+] {t}" for t in self.allowed_techniques]]
        if self.prohibited_techniques:
            doc += ["PROHIBITED:", *[f"  [-] {t}" for t in self.prohibited_techniques]]
        return "\n".join(doc)

    def validate(self) -> list[str]:
        """Validate the RoE for common issues."""
        issues = []
        if self.end_date <= self.start_date:
            issues.append("End date must be after start date")
        if not self.in_scope_targets:
            issues.append("At least one in-scope target required")
        if not self.emergency_contact:
            issues.append("Emergency contact required")
        return issues


# ---------------------------------------------------------------------------
# Engagement Planning
# ---------------------------------------------------------------------------

class EngagementType(Enum):
    VULN_ASSESSMENT = "Vulnerability Assessment"
    PENTEST_EXTERNAL = "External Penetration Test"
    PENTEST_WEB_APP = "Web Application Test"
    RED_TEAM = "Red Team Engagement"


@dataclass
class EngagementEstimate:
    """Estimates for a security engagement."""
    engagement_type: EngagementType
    target_count: int
    complexity: str

    @property
    def estimated_days(self) -> int:
        base = {
            EngagementType.VULN_ASSESSMENT: 3,
            EngagementType.PENTEST_EXTERNAL: 5,
            EngagementType.PENTEST_WEB_APP: 5,
            EngagementType.RED_TEAM: 20,
        }
        import math
        b = base[self.engagement_type]
        factor = 1 + math.log2(max(1, self.target_count)) * 0.3
        mult = {"low": 0.8, "medium": 1.0, "high": 1.5}.get(self.complexity, 1.0)
        return max(1, round(b * factor * mult))


# ---------------------------------------------------------------------------
# MITRE ATT&CK Mapping
# ---------------------------------------------------------------------------

@dataclass
class ATTCKTactic:
    id: str
    name: str
    pentest_phase: str
    techniques: list[str]


ATTCK_TACTICS = [
    ATTCKTactic("TA0043", "Reconnaissance", "Intelligence Gathering",
                ["T1595 — Active Scanning", "T1593 — Search Open Websites"]),
    ATTCKTactic("TA0001", "Initial Access", "Exploitation",
                ["T1190 — Exploit Public-Facing App", "T1566 — Phishing"]),
    ATTCKTactic("TA0004", "Privilege Escalation", "Post-Exploitation",
                ["T1068 — Exploitation for PrivEsc", "T1548 — Abuse Elevation"]),
    ATTCKTactic("TA0008", "Lateral Movement", "Post-Exploitation",
                ["T1021 — Remote Services", "T1550 — Alt Auth Material"]),
    ATTCKTactic("TA0010", "Exfiltration", "Post-Exploitation",
                ["T1041 — Exfil Over C2", "T1048 — Exfil Over Alt Protocol"]),
]


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    """Run demonstration."""
    print("Offensive Security Overview — Example Module")
    print("=" * 60)

    # Attack surface analysis
    surfaces = [
        AttackSurface("Login Form", AttackVector.WEB_APP,
                     "Authentication endpoint", Severity.HIGH,
                     ["Rate limiting", "MFA"], ["Burp Suite", "Hydra"]),
        AttackSurface("REST API", AttackVector.WEB_APP,
                     "JSON API with auth tokens", Severity.HIGH,
                     ["JWT validation", "RBAC"], ["Postman", "ffuf"]),
        AttackSurface("File Upload", AttackVector.WEB_APP,
                     "User file upload", Severity.CRITICAL,
                     ["File type validation"], ["Burp Suite"]),
    ]
    print("\nAttack Surface Analysis:")
    for s in sorted(surfaces, key=lambda x: x.risk_score(), reverse=True):
        print(f"  {s.name}: Risk {s.risk_score()}/10 ({s.severity.name})")

    # RoE generation
    roe = RulesOfEngagement(
        client_name="Acme Corp",
        tester_name="Jane Smith, OSCP",
        test_type=TestType.GREY_BOX,
        start_date=datetime(2025, 6, 1),
        end_date=datetime(2025, 6, 14),
        in_scope_targets=["*.acme.com", "10.0.0.0/24"],
        out_of_scope_targets=["Production DB", "Payment systems"],
        emergency_contact="security@acme.com",
        allowed_techniques=["Vuln scanning", "Manual web testing"],
        prohibited_techniques=["DoS", "Physical intrusion"],
    )
    print("\n" + roe.generate_document())

    # ATT&CK mapping
    print("\n\nMITRE ATT&CK Mapping:")
    for tactic in ATTCK_TACTICS:
        print(f"  [{tactic.id}] {tactic.name} ({tactic.pentest_phase})")
        for tech in tactic.techniques:
            print(f"    - {tech}")


if __name__ == "__main__":
    demo()
