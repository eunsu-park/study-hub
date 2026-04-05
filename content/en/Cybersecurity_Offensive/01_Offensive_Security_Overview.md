# Offensive Security Overview

**Previous**: [00. Overview](./00_Overview.md) | **Next**: [02. Reconnaissance](./02_Reconnaissance.md)

---

Offensive security is the proactive discipline of testing systems, networks, and applications by simulating real-world attacks. Unlike defensive security, which focuses on building walls and monitoring for breaches, offensive security actively seeks to break through those walls — with permission — to identify weaknesses before malicious actors do. This lesson establishes the ethical, legal, and methodological foundations that every subsequent lesson builds upon.

> **IMPORTANT**: All techniques in this course must only be used against systems you own or have explicit written authorization to test. Unauthorized access is a criminal offense.

**Difficulty**: ⭐⭐⭐

## Learning Objectives

After completing this lesson, you will be able to:

1. Articulate the ethical framework governing offensive security work
2. Understand key legal frameworks (CFAA, Computer Misuse Act, GDPR implications)
3. Differentiate between penetration testing, red teaming, and vulnerability assessment
4. Apply the Penetration Testing Execution Standard (PTES) methodology
5. Draft rules of engagement for an authorized security assessment
6. Understand responsible disclosure and bug bounty programs
7. Set up a safe lab environment for practicing offensive techniques
8. Map the MITRE ATT&CK framework to penetration testing phases

---

## Table of Contents

1. [The Offensive Security Mindset](#1-the-offensive-security-mindset)
2. [Ethics and Legal Frameworks](#2-ethics-and-legal-frameworks)
3. [Types of Security Assessments](#3-types-of-security-assessments)
4. [Penetration Testing Methodologies](#4-penetration-testing-methodologies)
5. [Rules of Engagement](#5-rules-of-engagement)
6. [The Kill Chain and MITRE ATT&CK](#6-the-kill-chain-and-mitre-attck)
7. [Responsible Disclosure](#7-responsible-disclosure)
8. [Lab Environment Setup](#8-lab-environment-setup)
9. [Scoping and Planning](#9-scoping-and-planning)
10. [Documentation and Reporting](#10-documentation-and-reporting)
11. [Career Paths and Certifications](#11-career-paths-and-certifications)
12. [Exercises](#12-exercises)
13. [Summary](#13-summary)
14. [References](#14-references)

---

## 1. The Offensive Security Mindset

The offensive security professional thinks like an adversary but acts with integrity. This dual nature — the technical curiosity to find every possible weakness combined with the ethical discipline to use that knowledge responsibly — defines the profession.

### 1.1 Think Like an Attacker

Attackers don't follow the rules. They look for the path of least resistance, combining technical exploits with social engineering, misconfigurations, and human error. An effective penetration tester must adopt this creative, unconstrained thinking while maintaining strict ethical boundaries.

**Key principles of the attacker mindset:**

- **Assume nothing is secure**: Every system has vulnerabilities; the question is whether you can find them within the engagement window
- **Chain weaknesses**: Individual low-severity findings may combine into critical attack paths
- **Lateral thinking**: The most impactful vulnerabilities often come from unexpected angles — supply chain compromises, overlooked API endpoints, or trust relationships between systems
- **Persistence**: Real attackers don't give up after the first failure; they enumerate, pivot, and try different approaches

```python
"""
Modeling the attacker mindset: Attack surface enumeration.

This script demonstrates how offensive security professionals
systematically identify potential entry points into a target system.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class AttackVector(Enum):
    """Categories of attack vectors following STRIDE model."""
    NETWORK = "Network-based"
    WEB_APP = "Web Application"
    SOCIAL = "Social Engineering"
    PHYSICAL = "Physical Access"
    SUPPLY_CHAIN = "Supply Chain"
    WIRELESS = "Wireless"
    INSIDER = "Insider Threat"


class Severity(Enum):
    """Risk severity levels aligned with CVSS qualitative ratings."""
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
    prerequisites: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)

    def risk_score(self) -> float:
        """Calculate a simplified risk score (0-10)."""
        base = self.severity.value * 2.5
        # Reduce score if mitigations exist
        mitigation_factor = max(0.3, 1.0 - (len(self.mitigations) * 0.15))
        # Increase score if fewer prerequisites (easier to exploit)
        prereq_factor = max(0.5, 1.0 + (0.1 * (3 - len(self.prerequisites))))
        return round(min(10.0, base * mitigation_factor * prereq_factor), 1)


def enumerate_web_attack_surface() -> list[AttackSurface]:
    """
    Enumerate common web application attack surfaces.

    In a real engagement, this would be populated through
    reconnaissance and automated scanning.
    """
    surfaces = [
        AttackSurface(
            name="Login Form",
            vector=AttackVector.WEB_APP,
            description="User authentication endpoint accepting credentials",
            severity=Severity.HIGH,
            mitigations=["Rate limiting", "Account lockout", "MFA"],
            prerequisites=["Valid username enumeration"],
            tools=["Burp Suite", "Hydra", "Custom scripts"],
        ),
        AttackSurface(
            name="REST API",
            vector=AttackVector.WEB_APP,
            description="JSON API endpoints with authentication tokens",
            severity=Severity.HIGH,
            mitigations=["JWT validation", "RBAC", "Input validation"],
            prerequisites=["API documentation or endpoint discovery"],
            tools=["Postman", "ffuf", "Burp Suite"],
        ),
        AttackSurface(
            name="File Upload",
            vector=AttackVector.WEB_APP,
            description="User-facing file upload functionality",
            severity=Severity.CRITICAL,
            mitigations=["File type validation", "Sandboxed storage"],
            prerequisites=["Authenticated session"],
            tools=["Burp Suite", "Custom polyglot files"],
        ),
        AttackSurface(
            name="DNS Records",
            vector=AttackVector.NETWORK,
            description="Public DNS records revealing infrastructure",
            severity=Severity.LOW,
            mitigations=["Minimal DNS exposure", "Split-horizon DNS"],
            prerequisites=[],
            tools=["dig", "nslookup", "subfinder", "amass"],
        ),
        AttackSurface(
            name="SSL/TLS Configuration",
            vector=AttackVector.NETWORK,
            description="TLS version and cipher suite configuration",
            severity=Severity.MEDIUM,
            mitigations=["TLS 1.3 only", "Strong cipher suites"],
            prerequisites=[],
            tools=["testssl.sh", "sslyze", "nmap"],
        ),
    ]
    return surfaces


def prioritize_attack_surfaces(
    surfaces: list[AttackSurface],
) -> list[AttackSurface]:
    """Sort attack surfaces by risk score (highest first)."""
    return sorted(surfaces, key=lambda s: s.risk_score(), reverse=True)


# Demonstration
if __name__ == "__main__":
    surfaces = enumerate_web_attack_surface()
    prioritized = prioritize_attack_surfaces(surfaces)

    print("=" * 70)
    print("ATTACK SURFACE ANALYSIS — Prioritized by Risk")
    print("=" * 70)
    for i, surface in enumerate(prioritized, 1):
        print(f"\n[{i}] {surface.name}")
        print(f"    Vector:      {surface.vector.value}")
        print(f"    Severity:    {surface.severity.name}")
        print(f"    Risk Score:  {surface.risk_score()}/10")
        print(f"    Description: {surface.description}")
        print(f"    Tools:       {', '.join(surface.tools)}")
        if surface.mitigations:
            print(f"    Mitigations: {', '.join(surface.mitigations)}")
```

### 1.2 The Difference Between Hackers and Crackers

The security community distinguishes between different motivations:

| Category | Motivation | Authorization | Goal |
|----------|-----------|---------------|------|
| White Hat | Defensive | Authorized | Improve security |
| Black Hat | Malicious | Unauthorized | Personal gain |
| Grey Hat | Mixed | Sometimes | Disclosure varies |
| Bug Bounty Hunter | Financial + Ethical | Program-scoped | Find and report bugs |
| Red Teamer | Adversary simulation | Contracted | Test detection/response |

### 1.3 The Pentester's Code of Ethics

Professional penetration testers adhere to strict ethical guidelines:

1. **Written authorization**: Never test without explicit, signed permission
2. **Scope adherence**: Stay within the defined scope — never exceed boundaries
3. **Data protection**: Handle any sensitive data discovered with extreme care
4. **Do no harm**: Avoid causing unnecessary disruption to production systems
5. **Full disclosure**: Report all findings to the client, not just critical ones
6. **Confidentiality**: Never disclose client vulnerabilities to third parties
7. **Continuous learning**: Keep skills current with evolving threats

---

## 2. Ethics and Legal Frameworks

### 2.1 Computer Fraud and Abuse Act (CFAA) — United States

The CFAA (18 U.S.C. § 1030) criminalizes unauthorized access to computer systems. Key provisions:

- **Unauthorized access**: Accessing a computer without permission
- **Exceeding authorized access**: Having some access but going beyond what's permitted
- **Penalties**: Up to 10 years imprisonment for first offense, 20 years for repeat offenses
- **Civil liability**: Victims can sue for damages

> **Critical for pentesters**: Even with verbal permission, always obtain **written authorization** that explicitly defines the scope, timeline, and methods allowed.

### 2.2 Computer Misuse Act (CMA) — United Kingdom

The CMA 1990 defines three primary offenses:

1. Unauthorized access to computer material (Section 1)
2. Unauthorized access with intent to commit further offenses (Section 2)
3. Unauthorized modification of computer material (Section 3)

### 2.3 GDPR Implications

When testing systems that handle EU personal data:

- Penetration testing may involve processing personal data
- The test scope must be documented in a Data Processing Agreement
- Any personal data accessed during testing must be handled per GDPR requirements
- Data breach notification obligations may apply if testing causes an actual breach

```python
"""
Rules of Engagement (RoE) document generator.

This utility helps create standardized authorization documents
for penetration testing engagements.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class TestType(Enum):
    """Types of security assessments."""
    BLACK_BOX = "Black Box — No prior knowledge"
    GREY_BOX = "Grey Box — Partial knowledge/credentials"
    WHITE_BOX = "White Box — Full source code and architecture access"


class TestScope(Enum):
    """Scope categories for testing."""
    EXTERNAL = "External — Internet-facing assets only"
    INTERNAL = "Internal — Internal network access provided"
    WEB_APP = "Web Application — Specific web applications"
    MOBILE = "Mobile Application — iOS/Android applications"
    WIRELESS = "Wireless — WiFi and Bluetooth"
    SOCIAL = "Social Engineering — Phishing, vishing, physical"
    CLOUD = "Cloud Infrastructure — AWS/GCP/Azure"


@dataclass
class RulesOfEngagement:
    """Formal rules of engagement for a penetration test."""
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
    data_handling: str = "All data encrypted; destroyed 30 days after report delivery"

    def generate_document(self) -> str:
        """Generate a formatted RoE document."""
        doc = []
        doc.append("=" * 70)
        doc.append("RULES OF ENGAGEMENT — PENETRATION TESTING AUTHORIZATION")
        doc.append("=" * 70)
        doc.append("")
        doc.append(f"Client:          {self.client_name}")
        doc.append(f"Tester:          {self.tester_name} ({self.tester_company})")
        doc.append(f"Test Type:       {self.test_type.value}")
        doc.append(f"Period:          {self.start_date:%Y-%m-%d} to {self.end_date:%Y-%m-%d}")
        doc.append(f"Testing Hours:   {self.testing_hours}")
        doc.append(f"Emergency Contact: {self.emergency_contact}")
        doc.append("")
        doc.append("SCOPE:")
        for s in self.scope:
            doc.append(f"  [+] {s.value}")
        doc.append("")
        doc.append("IN-SCOPE TARGETS:")
        for t in self.in_scope_targets:
            doc.append(f"  [+] {t}")
        doc.append("")
        doc.append("OUT-OF-SCOPE TARGETS:")
        for t in self.out_of_scope_targets:
            doc.append(f"  [-] {t}")
        doc.append("")
        if self.allowed_techniques:
            doc.append("ALLOWED TECHNIQUES:")
            for t in self.allowed_techniques:
                doc.append(f"  [+] {t}")
            doc.append("")
        if self.prohibited_techniques:
            doc.append("PROHIBITED TECHNIQUES:")
            for t in self.prohibited_techniques:
                doc.append(f"  [-] {t}")
            doc.append("")
        doc.append(f"DATA HANDLING: {self.data_handling}")
        doc.append("")
        doc.append("SIGNATURES:")
        doc.append(f"  Client:  _________________ Date: _________")
        doc.append(f"  Tester:  _________________ Date: _________")
        doc.append("=" * 70)
        return "\n".join(doc)

    def validate(self) -> list[str]:
        """Validate the RoE for common issues."""
        issues = []
        if self.end_date <= self.start_date:
            issues.append("End date must be after start date")
        if not self.in_scope_targets:
            issues.append("At least one in-scope target is required")
        if not self.emergency_contact:
            issues.append("Emergency contact is required")
        if self.end_date - self.start_date > timedelta(days=90):
            issues.append("Engagement period exceeds 90 days — consider splitting")
        return issues


# Example usage
if __name__ == "__main__":
    roe = RulesOfEngagement(
        client_name="Acme Corporation",
        tester_name="Jane Smith, OSCP",
        tester_company="SecureTest Labs",
        test_type=TestType.GREY_BOX,
        scope=[TestScope.EXTERNAL, TestScope.WEB_APP],
        start_date=datetime(2025, 6, 1),
        end_date=datetime(2025, 6, 14),
        in_scope_targets=[
            "*.acme.com (web applications)",
            "203.0.113.0/24 (external network range)",
            "api.acme.com (REST API)",
        ],
        out_of_scope_targets=[
            "production database servers",
            "payment processing systems (PCI scope)",
            "third-party SaaS integrations",
            "employee personal devices",
        ],
        emergency_contact="security@acme.com / +1-555-0199 (24/7 SOC)",
        allowed_techniques=[
            "Automated vulnerability scanning",
            "Manual web application testing",
            "Credential brute-forcing (rate-limited)",
            "Social engineering (phishing simulation — pre-approved list)",
        ],
        prohibited_techniques=[
            "Denial of Service (DoS/DDoS)",
            "Physical intrusion",
            "Testing of third-party systems",
            "Modification or deletion of production data",
        ],
    )

    issues = roe.validate()
    if issues:
        print("VALIDATION ISSUES:")
        for issue in issues:
            print(f"  [!] {issue}")
    else:
        print(roe.generate_document())
```

---

## 3. Types of Security Assessments

Understanding the spectrum of security assessments helps select the right approach for each situation.

### 3.1 Vulnerability Assessment

A **vulnerability assessment** identifies and quantifies security weaknesses without exploiting them:

- **Scope**: Broad, covers all assets
- **Depth**: Surface-level; identifies but doesn't validate
- **Duration**: Days to a week
- **Output**: Prioritized list of vulnerabilities with CVSS scores
- **Tools**: Nessus, OpenVAS, Qualys, Nexpose

### 3.2 Penetration Testing

A **penetration test** (pentest) actively exploits vulnerabilities to demonstrate real-world impact:

- **Scope**: Targeted, specific systems or applications
- **Depth**: Deep; attempts full exploitation chains
- **Duration**: 1-4 weeks
- **Output**: Detailed findings with proof-of-concept exploits and remediation guidance
- **Phases**: Reconnaissance → Scanning → Exploitation → Post-exploitation → Reporting

### 3.3 Red Team Engagement

A **red team engagement** simulates a sophisticated adversary over an extended period:

- **Scope**: Organization-wide, including people and processes
- **Depth**: Maximum; mimics APT (Advanced Persistent Threat) tactics
- **Duration**: Weeks to months
- **Output**: Assessment of detection and response capabilities
- **Key difference**: Tests the blue team's ability to detect and respond, not just technical vulnerabilities

### 3.4 Bug Bounty Programs

**Bug bounty programs** crowdsource vulnerability discovery:

- **Platforms**: HackerOne, Bugcrowd, Intigriti
- **Scope**: Defined by the program's policy
- **Rewards**: Based on severity (typically $100 — $100,000+)
- **Rules**: Each program has specific rules about what's in-scope

```python
"""
Engagement planning helper — calculates effort and resource requirements.
"""

from dataclasses import dataclass
from enum import Enum


class EngagementType(Enum):
    VULN_ASSESSMENT = "Vulnerability Assessment"
    PENTEST_EXTERNAL = "External Penetration Test"
    PENTEST_INTERNAL = "Internal Penetration Test"
    PENTEST_WEB_APP = "Web Application Penetration Test"
    RED_TEAM = "Red Team Engagement"
    PURPLE_TEAM = "Purple Team Exercise"


@dataclass
class EngagementEstimate:
    """Estimates for a security engagement."""
    engagement_type: EngagementType
    target_count: int
    complexity: str  # "low", "medium", "high"

    @property
    def estimated_days(self) -> int:
        """Estimate working days based on type and scope."""
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
        # Scale by number of targets (logarithmic)
        import math
        target_factor = 1 + math.log2(max(1, self.target_count)) * 0.3
        cmult = complexity_multiplier.get(self.complexity, 1.0)
        return max(1, round(base * target_factor * cmult))

    @property
    def team_size(self) -> int:
        """Recommended team size."""
        if self.engagement_type == EngagementType.RED_TEAM:
            return 3
        if self.estimated_days > 10:
            return 2
        return 1

    @property
    def report_days(self) -> int:
        """Days needed for report writing."""
        return max(2, self.estimated_days // 3)

    def summary(self) -> str:
        lines = [
            f"Engagement: {self.engagement_type.value}",
            f"Targets: {self.target_count} | Complexity: {self.complexity}",
            f"Estimated effort: {self.estimated_days} working days",
            f"Team size: {self.team_size} tester(s)",
            f"Report writing: {self.report_days} additional days",
            f"Total timeline: {self.estimated_days + self.report_days} days",
        ]
        return "\n".join(lines)


# Example
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

## 4. Penetration Testing Methodologies

### 4.1 PTES (Penetration Testing Execution Standard)

The PTES defines seven phases of a penetration test:

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  1. Pre-engagement│ ──▶ │  2. Intelligence  │ ──▶ │  3. Threat       │
│     Interactions  │     │     Gathering     │     │     Modeling     │
└──────────────────┘     └──────────────────┘     └──────────────────┘
         │                                                   │
         ▼                                                   ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  7. Reporting     │ ◀── │  6. Post-         │ ◀── │  4. Vulnerability│
│                   │     │     Exploitation  │     │     Analysis     │
└──────────────────┘     └──────────────────┘     └──────────────────┘
                                  ▲                         │
                                  │                         ▼
                          ┌──────────────────┐
                          │  5. Exploitation  │
                          └──────────────────┘
```

**Phase 1: Pre-engagement Interactions**
- Define scope, rules of engagement, and authorization
- Identify emergency contacts and communication channels
- Establish testing timeline and milestones

**Phase 2: Intelligence Gathering**
- Passive reconnaissance (OSINT, DNS, WHOIS)
- Active reconnaissance (port scanning, service enumeration)
- Social media and employee information gathering

**Phase 3: Threat Modeling**
- Identify valuable assets and data
- Determine likely attack vectors
- Prioritize targets based on business impact

**Phase 4: Vulnerability Analysis**
- Automated scanning (Nessus, OpenVAS)
- Manual testing and validation
- False positive elimination

**Phase 5: Exploitation**
- Attempt to exploit identified vulnerabilities
- Document proof-of-concept for each finding
- Maintain careful notes of all actions taken

**Phase 6: Post-Exploitation**
- Determine the value of compromised systems
- Attempt privilege escalation and lateral movement
- Identify additional targets reachable from compromised hosts

**Phase 7: Reporting**
- Executive summary for management
- Technical details for remediation teams
- Risk ratings and prioritized recommendations

### 4.2 OWASP Testing Guide

The OWASP Testing Guide provides a comprehensive framework for web application testing:

- **Information Gathering**: Technology fingerprinting, content discovery
- **Configuration Management**: Default credentials, error handling, HTTP methods
- **Identity Management**: User registration, account enumeration
- **Authentication**: Credential testing, session management
- **Authorization**: Path traversal, privilege escalation, IDOR
- **Session Management**: Cookie attributes, session fixation, CSRF
- **Input Validation**: SQL injection, XSS, command injection
- **Error Handling**: Stack traces, error codes
- **Cryptography**: Weak algorithms, improper implementation
- **Business Logic**: Workflow bypass, function abuse
- **Client-Side**: DOM XSS, JavaScript injection, clickjacking

### 4.3 OSSTMM (Open Source Security Testing Methodology Manual)

OSSTMM focuses on operational security testing with measurable results:

- Defines "attack surface" quantitatively (RAV — Rave Attack Value)
- Tests across five channels: Human, Physical, Wireless, Telecommunications, Data Networks
- Emphasizes reproducibility and metrics

---

## 5. Rules of Engagement

The Rules of Engagement (RoE) document is the most critical artifact in any security assessment. It provides legal protection for both the tester and the client.

### 5.1 Essential Components

Every RoE must include:

1. **Authorization**: Explicit written permission from an authorized representative
2. **Scope definition**: Exact IP ranges, domains, applications, and personnel in scope
3. **Exclusions**: Systems and actions that are explicitly off-limits
4. **Timeline**: Start date, end date, and testing hours
5. **Allowed techniques**: What methods are permitted (e.g., social engineering, DoS)
6. **Communication plan**: How to report critical findings immediately
7. **Emergency contacts**: Who to call if testing causes unexpected issues
8. **Data handling**: How sensitive data discovered during testing will be protected
9. **Third-party notification**: Whether cloud providers or hosting companies need notification
10. **Legal clauses**: Liability limitations and indemnification

### 5.2 Get-Out-of-Jail-Free Letter

The authorization letter (sometimes called a "get-out-of-jail-free letter") should:

- Be on official company letterhead
- Be signed by someone with authority (CISO, CTO, or CEO)
- Explicitly state that penetration testing is authorized
- Include the tester's name and company
- Specify the exact dates and scope
- Include an emergency phone number

> **Best practice**: Carry a printed copy during physical assessments. Digital copies should be readily accessible during remote testing.

---

## 6. The Kill Chain and MITRE ATT&CK

### 6.1 Lockheed Martin Cyber Kill Chain

The Cyber Kill Chain models the stages of a targeted cyberattack:

```
1. Reconnaissance ──▶ 2. Weaponization ──▶ 3. Delivery
                                                │
    7. Actions on   ◀── 6. Command &    ◀── 5. Installation
       Objectives        Control              │
                                          4. Exploitation
```

1. **Reconnaissance**: Gather information about the target
2. **Weaponization**: Create a deliverable payload (e.g., trojanized document)
3. **Delivery**: Transmit the payload (email, web, USB)
4. **Exploitation**: Trigger the vulnerability
5. **Installation**: Install persistent access (backdoor, RAT)
6. **Command and Control (C2)**: Establish communication channel
7. **Actions on Objectives**: Achieve the attacker's goal (exfiltration, destruction)

### 6.2 MITRE ATT&CK Framework

MITRE ATT&CK (Adversarial Tactics, Techniques, and Common Knowledge) provides a comprehensive matrix of real-world adversary behaviors:

```python
"""
MITRE ATT&CK Tactic mapping for penetration testing phases.

Maps pentest activities to ATT&CK tactics for consistent
reporting and threat emulation.
"""

from dataclasses import dataclass


@dataclass
class ATTCKTactic:
    """Represents a MITRE ATT&CK tactic."""
    id: str
    name: str
    description: str
    pentest_phase: str
    example_techniques: list[str]


ATTCK_TACTICS = [
    ATTCKTactic(
        id="TA0043",
        name="Reconnaissance",
        description="Gather information to plan future operations",
        pentest_phase="Intelligence Gathering",
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
        description="Gain initial foothold in the target network",
        pentest_phase="Exploitation",
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
        description="Run adversary-controlled code",
        pentest_phase="Exploitation",
        example_techniques=[
            "T1059 — Command and Scripting Interpreter",
            "T1203 — Exploitation for Client Execution",
            "T1047 — Windows Management Instrumentation",
        ],
    ),
    ATTCKTactic(
        id="TA0003",
        name="Persistence",
        description="Maintain access across restarts",
        pentest_phase="Post-Exploitation",
        example_techniques=[
            "T1053 — Scheduled Task/Job",
            "T1547 — Boot or Logon Autostart Execution",
            "T1136 — Create Account",
        ],
    ),
    ATTCKTactic(
        id="TA0004",
        name="Privilege Escalation",
        description="Gain higher-level permissions",
        pentest_phase="Post-Exploitation",
        example_techniques=[
            "T1068 — Exploitation for Privilege Escalation",
            "T1548 — Abuse Elevation Control Mechanism",
            "T1134 — Access Token Manipulation",
        ],
    ),
    ATTCKTactic(
        id="TA0005",
        name="Defense Evasion",
        description="Avoid detection",
        pentest_phase="Post-Exploitation",
        example_techniques=[
            "T1070 — Indicator Removal",
            "T1036 — Masquerading",
            "T1027 — Obfuscated Files or Information",
        ],
    ),
    ATTCKTactic(
        id="TA0006",
        name="Credential Access",
        description="Steal credentials",
        pentest_phase="Post-Exploitation",
        example_techniques=[
            "T1003 — OS Credential Dumping",
            "T1110 — Brute Force",
            "T1558 — Steal or Forge Kerberos Tickets",
        ],
    ),
    ATTCKTactic(
        id="TA0007",
        name="Discovery",
        description="Understand the target environment",
        pentest_phase="Post-Exploitation",
        example_techniques=[
            "T1087 — Account Discovery",
            "T1046 — Network Service Discovery",
            "T1083 — File and Directory Discovery",
        ],
    ),
    ATTCKTactic(
        id="TA0008",
        name="Lateral Movement",
        description="Move through the environment",
        pentest_phase="Post-Exploitation",
        example_techniques=[
            "T1021 — Remote Services",
            "T1080 — Taint Shared Content",
            "T1550 — Use Alternate Authentication Material",
        ],
    ),
    ATTCKTactic(
        id="TA0010",
        name="Exfiltration",
        description="Steal data from the target",
        pentest_phase="Post-Exploitation",
        example_techniques=[
            "T1041 — Exfiltration Over C2 Channel",
            "T1048 — Exfiltration Over Alternative Protocol",
            "T1567 — Exfiltration Over Web Service",
        ],
    ),
]


def map_pentest_to_attck(pentest_phase: str) -> list[ATTCKTactic]:
    """Find ATT&CK tactics relevant to a pentest phase."""
    return [t for t in ATTCK_TACTICS if t.pentest_phase == pentest_phase]


def generate_attck_report() -> str:
    """Generate a mapping report for documentation."""
    lines = ["MITRE ATT&CK Mapping for Penetration Test", "=" * 50, ""]
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

## 7. Responsible Disclosure

### 7.1 Disclosure Models

When vulnerabilities are discovered (especially outside formal engagements), there are several disclosure approaches:

**Full Disclosure**: Publicly release all vulnerability details immediately.
- **Pro**: Pressures vendors to patch quickly
- **Con**: Exposes users to risk before a fix is available

**Responsible Disclosure (Coordinated Disclosure)**: Privately notify the vendor, give them a reasonable timeframe (typically 90 days) to release a patch, then disclose publicly.
- **Pro**: Balances vendor notification with public transparency
- **Con**: Vendor may not respond or may delay indefinitely

**Non-Disclosure**: Never reveal the vulnerability publicly.
- **Pro**: Minimizes exposure
- **Con**: Other researchers may independently discover it; users remain unprotected

### 7.2 Bug Bounty Best Practices

When participating in bug bounty programs:

1. **Read the policy**: Understand what's in scope before testing
2. **Don't test production without permission**: Use staging environments when available
3. **Minimize impact**: Don't exfiltrate real data; use proof-of-concept demonstrations
4. **Write clear reports**: Include steps to reproduce, impact analysis, and remediation suggestions
5. **Be patient**: Response times vary; don't threaten public disclosure as leverage
6. **One report per vulnerability**: Don't bundle multiple issues together

---

## 8. Lab Environment Setup

### 8.1 Isolated Lab Network

A proper lab environment isolates your testing from production networks:

```
┌─────────────────────────────────────────────────┐
│                  Host Machine                    │
│                                                  │
│  ┌──────────────┐  ┌──────────────┐            │
│  │  Kali Linux  │  │  Target VMs  │            │
│  │  (Attacker)  │  │  (Victims)   │            │
│  │              │  │              │            │
│  │  - Nmap      │  │  - Metasploit│            │
│  │  - Burp      │──│    able 2/3  │            │
│  │  - Metasploit│  │  - DVWA      │            │
│  │  - pwntools  │  │  - WebGoat   │            │
│  └──────────────┘  └──────────────┘            │
│         │                  │                     │
│         └────── NAT/Host-Only Network ──────────│
│                  (No external access)            │
└─────────────────────────────────────────────────┘
```

### 8.2 Essential Virtual Machines

| VM | Purpose | Download |
|----|---------|----------|
| Kali Linux | Attacker workstation | kali.org |
| Metasploitable 2/3 | Linux targets | SourceForge |
| DVWA | Web application testing | github.com/digininja/DVWA |
| OWASP WebGoat | Web security training | owasp.org |
| Vulnhub VMs | Various difficulty levels | vulnhub.com |
| HackTheBox | Online labs | hackthebox.com |

### 8.3 Automated Lab Setup

```python
"""
Lab environment validator — checks that required tools are installed
and the network is properly isolated.

Run this on your Kali/attacker VM to verify your lab setup.
"""

import shutil
import subprocess
import socket
from dataclasses import dataclass


@dataclass
class ToolCheck:
    """Represents a required tool and its status."""
    name: str
    command: str
    required: bool = True
    installed: bool = False
    version: str = ""


def check_tool(tool: ToolCheck) -> ToolCheck:
    """Check if a tool is installed and get its version."""
    path = shutil.which(tool.command)
    if path:
        tool.installed = True
        try:
            result = subprocess.run(
                [tool.command, "--version"],
                capture_output=True, text=True, timeout=5
            )
            version_line = (result.stdout or result.stderr).strip().split("\n")[0]
            tool.version = version_line[:80]  # Truncate long version strings
        except (subprocess.TimeoutExpired, FileNotFoundError):
            tool.version = "installed (version unknown)"
    return tool


def check_network_isolation() -> dict:
    """Verify the lab network is properly isolated."""
    results = {
        "can_resolve_dns": False,
        "can_reach_internet": False,
        "local_interfaces": [],
    }

    # Check DNS resolution
    try:
        socket.getaddrinfo("example.com", 80, socket.AF_INET)
        results["can_resolve_dns"] = True
    except socket.gaierror:
        pass

    # Check internet connectivity
    try:
        sock = socket.create_connection(("8.8.8.8", 53), timeout=3)
        sock.close()
        results["can_reach_internet"] = True
    except (socket.timeout, OSError):
        pass

    return results


def validate_lab() -> None:
    """Run all lab validation checks."""
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
    print("LAB ENVIRONMENT VALIDATION")
    print("=" * 60)

    # Check tools
    print("\n--- Tool Availability ---")
    missing_required = []
    for tool in tools:
        check_tool(tool)
        status = "[OK]" if tool.installed else ("[MISSING!]" if tool.required else "[optional]")
        print(f"  {status:12s} {tool.name:20s} {tool.version}")
        if tool.required and not tool.installed:
            missing_required.append(tool.name)

    # Check network
    print("\n--- Network Isolation ---")
    network = check_network_isolation()
    if network["can_reach_internet"]:
        print("  [WARNING] Internet access detected!")
        print("  For isolated labs, use host-only networking.")
    else:
        print("  [OK] No internet access (properly isolated)")

    # Summary
    print("\n--- Summary ---")
    if missing_required:
        print(f"  [!] Missing required tools: {', '.join(missing_required)}")
        print("  Install with: sudo apt install <tool>")
    else:
        print("  [OK] All required tools are installed")


if __name__ == "__main__":
    validate_lab()
```

---

## 9. Scoping and Planning

### 9.1 Defining the Engagement Scope

Proper scoping prevents scope creep and legal issues:

**Network scope:**
- Define by IP ranges (CIDR notation): `10.0.0.0/24`
- Include/exclude specific hosts: `10.0.0.1` (include), `10.0.0.50` (exclude — production DB)
- Specify port ranges if limited: `TCP 1-65535, UDP top 1000`

**Application scope:**
- List specific URLs/domains: `https://app.example.com/*`
- Define authenticated vs. unauthenticated testing
- Specify user roles to test: admin, regular user, guest

**Physical scope (if applicable):**
- Buildings and floors authorized for testing
- Social engineering targets (or explicit exclusions)
- Badge cloning, tailgating, dumpster diving permissions

### 9.2 Time Estimation

Rule of thumb for time estimation:

| Assessment Type | Small (1-10 targets) | Medium (10-50) | Large (50+) |
|----------------|---------------------|-----------------|-------------|
| Vulnerability Scan | 1-2 days | 3-5 days | 1-2 weeks |
| External Pentest | 3-5 days | 1-2 weeks | 2-4 weeks |
| Internal Pentest | 5-7 days | 2-3 weeks | 3-6 weeks |
| Web App Test | 3-5 days/app | - | - |
| Red Team | 2-4 weeks | 4-8 weeks | 8-12 weeks |

### 9.3 Deliverables

A professional pentest report typically includes:

1. **Executive Summary** (1-2 pages): Business-language overview for management
2. **Methodology**: Description of tools, techniques, and approach
3. **Findings**: Each vulnerability with severity, description, proof-of-concept, impact, and remediation
4. **Risk Rating**: CVSS scores or custom severity scale
5. **Remediation Roadmap**: Prioritized fix recommendations
6. **Appendices**: Raw scan output, screenshots, full technical details

---

## 10. Documentation and Reporting

### 10.1 During the Engagement

Good note-taking during testing is essential:

```python
"""
Engagement logging utility.

Maintain a structured log of all actions taken during testing.
Essential for report writing and legal protection.
"""

import json
import hashlib
from datetime import datetime
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class LogEntry:
    """A single action logged during the engagement."""
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
    """Structured logging for penetration testing engagements."""

    def __init__(self, engagement_name: str, log_dir: str = "./logs"):
        self.engagement_name = engagement_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"{engagement_name}.jsonl"
        self.entries: list[LogEntry] = []

    def log(self, entry: LogEntry) -> None:
        """Append a log entry."""
        self.entries.append(entry)
        with open(self.log_file, "a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")

    def log_action(self, category: str, action: str, target: str,
                   result: str, **kwargs) -> None:
        """Convenience method to log an action."""
        entry = LogEntry.now(category, action, target, result, **kwargs)
        self.log(entry)

    def get_entries_by_category(self, category: str) -> list[LogEntry]:
        return [e for e in self.entries if e.category == category]

    def generate_timeline(self) -> str:
        """Generate a human-readable timeline."""
        lines = [
            f"Engagement Timeline: {self.engagement_name}",
            "=" * 60,
        ]
        for entry in self.entries:
            lines.append(
                f"[{entry.timestamp}] [{entry.category.upper():12s}] "
                f"{entry.action} -> {entry.target}: {entry.result}"
            )
        return "\n".join(lines)

    def evidence_hash(self, filepath: str) -> str:
        """Generate SHA-256 hash of evidence file for integrity."""
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()


# Example usage
if __name__ == "__main__":
    logger = EngagementLogger("acme-pentest-2025")

    logger.log_action(
        category="recon",
        action="DNS enumeration",
        target="acme.com",
        result="Found 12 subdomains",
        tool="subfinder",
        command="subfinder -d acme.com -o subs.txt",
    )
    logger.log_action(
        category="scanning",
        action="Port scan",
        target="10.0.0.0/24",
        result="Found 45 open ports across 12 hosts",
        tool="nmap",
        command="nmap -sV -sC -oA acme_scan 10.0.0.0/24",
    )
    logger.log_action(
        category="exploitation",
        action="SQL injection",
        target="https://app.acme.com/login",
        result="Extracted admin credentials (redacted)",
        tool="sqlmap",
        notes="Time-based blind SQLi in username parameter",
    )

    print(logger.generate_timeline())
```

### 10.2 Report Quality

A professional report transforms raw findings into actionable intelligence:

- **Reproducibility**: Another tester should be able to reproduce every finding
- **Screenshots**: Include timestamped screenshots of key evidence
- **Impact analysis**: Explain business impact, not just technical severity
- **Remediation**: Provide specific, actionable fixes — not just "patch the vulnerability"
- **Proof of concept**: Include working PoC code (sanitized of sensitive data)

---

## 11. Career Paths and Certifications

### 11.1 Key Certifications

| Certification | Focus | Difficulty |
|--------------|-------|------------|
| CompTIA Security+ | Foundational security | Entry |
| CEH (Certified Ethical Hacker) | Ethical hacking breadth | Intermediate |
| OSCP (Offensive Security Certified Professional) | Hands-on pentesting | Advanced |
| OSWE (Offensive Security Web Expert) | Web application security | Advanced |
| OSEP (Offensive Security Experienced Penetration Tester) | Advanced exploitation | Expert |
| GPEN (GIAC Penetration Tester) | Network pentesting | Intermediate |
| GWAPT (GIAC Web Application Penetration Tester) | Web app testing | Intermediate |
| CRTO (Certified Red Team Operator) | Red team operations | Advanced |
| PNPT (Practical Network Penetration Tester) | Practical pentesting | Intermediate |

### 11.2 Career Progression

```
Junior Pentester ──▶ Penetration Tester ──▶ Senior Pentester
                                                    │
                                                    ▼
                     Red Team Lead ◀── Red Team Operator
                          │
                          ▼
              Security Consultant / CISO
```

### 11.3 Continuous Learning Resources

- **Platforms**: Hack The Box, TryHackMe, PortSwigger Web Security Academy
- **Conferences**: DEF CON, Black Hat, BSides, OWASP AppSec
- **Communities**: r/netsec, InfoSec Twitter/Mastodon, local security meetups
- **Publications**: Phrack, PoC||GTFO, security researcher blogs

---

## 12. Exercises

1. **Rules of Engagement**: Draft a complete RoE document for a hypothetical external penetration test of a fictional e-commerce company. Include all essential components.

2. **MITRE ATT&CK Mapping**: Given a scenario where an attacker gains initial access through a phishing email, escalates privileges via a kernel exploit, and exfiltrates data over DNS, map each step to ATT&CK tactics and techniques.

3. **Lab Setup**: Set up a Kali Linux VM with Metasploitable 2 target in an isolated network. Verify isolation using the lab validator script.

4. **Engagement Planning**: Estimate the effort required for a grey-box web application pentest of an organization with 5 web apps, each with authenticated and unauthenticated functionality.

5. **Ethical Dilemma**: During a pentest you discover evidence that an employee is embezzling money. Your scope only covers technical security testing. What do you do? Write a 500-word analysis.

6. **Methodology Comparison**: Compare PTES, OWASP Testing Guide, and OSSTMM. Create a table showing strengths, weaknesses, and best use cases for each.

---

## 13. Summary

Offensive security is a powerful discipline that requires equal measures of technical skill and ethical responsibility. In this lesson, we established the foundations:

- The **offensive security mindset** combines creative attack thinking with strict ethical boundaries
- **Legal frameworks** (CFAA, CMA, GDPR) define what is and isn't permissible
- **Rules of Engagement** are mandatory legal protections for every assessment
- **PTES** provides a seven-phase methodology for structured penetration testing
- **MITRE ATT&CK** maps real-world adversary behaviors to a common taxonomy
- **Responsible disclosure** balances vendor notification with public safety
- **Lab environments** must be isolated from production networks
- **Documentation** throughout the engagement supports both reporting and legal protection

The remaining lessons in this topic build on these foundations, moving from reconnaissance through exploitation to post-exploitation and red team operations.

---

## 14. References

- Penetration Testing Execution Standard (PTES): http://www.pentest-standard.org/
- MITRE ATT&CK Framework: https://attack.mitre.org/
- OWASP Testing Guide v4.2: https://owasp.org/www-project-web-security-testing-guide/
- NIST SP 800-115: https://csrc.nist.gov/publications/detail/sp/800-115/final
- Computer Fraud and Abuse Act (18 U.S.C. § 1030): https://www.law.cornell.edu/uscode/text/18/1030
- EC-Council Code of Ethics: https://www.eccouncil.org/code-of-ethics/
- HackerOne Disclosure Guidelines: https://www.hackerone.com/disclosure-guidelines
- Weidman, Georgia. *Penetration Testing: A Hands-On Introduction to Hacking*. No Starch Press, 2014.
