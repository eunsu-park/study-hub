# Red Team Operations

**Previous**: [19. CTF Methodology](./19_CTF_Methodology.md)

---

Red team operations represent the pinnacle of offensive security — a full-scope adversary simulation that tests not just technical controls but an organization's people, processes, and detection capabilities. Unlike penetration testing, red teaming emulates real threat actors over extended periods, using stealth and creativity to achieve specific objectives.

> **IMPORTANT**: Red team operations require extensive planning, legal authorization, and professional execution. Never conduct unauthorized adversary simulation.

**Difficulty**: ⭐⭐⭐⭐

## Learning Objectives

1. Distinguish red team operations from standard penetration tests
2. Plan red team engagements with clear objectives and threat models
3. Build adversary emulation plans using MITRE ATT&CK
4. Set up and manage command and control infrastructure
5. Design and execute social engineering campaigns
6. Maintain operational security throughout engagements
7. Conduct purple team exercises to improve detection
8. Write executive-level red team reports with actionable recommendations

---

## Table of Contents

1. [Red Team vs Penetration Testing](#1-red-team-vs-penetration-testing)
2. [Engagement Planning and Threat Modeling](#2-engagement-planning-and-threat-modeling)
3. [MITRE ATT&CK for Red Teams](#3-mitre-attck-for-red-teams)
4. [Adversary Emulation Plans](#4-adversary-emulation-plans)
5. [Command and Control Infrastructure](#5-command-and-control-infrastructure)
6. [Social Engineering Campaigns](#6-social-engineering-campaigns)
7. [Physical Security Testing](#7-physical-security-testing)
8. [Evasion and OPSEC](#8-evasion-and-opsec)
9. [Purple Team Exercises](#9-purple-team-exercises)
10. [Reporting and Remediation](#10-reporting-and-remediation)
11. [Exercises](#11-exercises)
12. [Summary](#12-summary)
13. [References](#13-references)

---

## 1. Red Team vs Penetration Testing

| Aspect | Penetration Test | Red Team |
|--------|-----------------|----------|
| **Goal** | Find vulnerabilities | Test detection and response |
| **Scope** | Defined targets | Organization-wide |
| **Duration** | 1-4 weeks | 2-6 months |
| **Stealth** | Not required | Essential |
| **Knowledge** | Blue team aware | Limited awareness |
| **Methodology** | Systematic scanning | Adversary emulation |
| **Social eng.** | Usually excluded | Core component |
| **Output** | Vulnerability list | Detection gap analysis |
| **Team size** | 1-2 testers | 3-5+ operators |

---

## 2. Engagement Planning and Threat Modeling

### 2.1 Red Team Charter

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

## 3. MITRE ATT&CK for Red Teams

Use ATT&CK to structure adversary emulation:

```
Tactic Flow:
Reconnaissance → Initial Access → Execution → Persistence
    → Privilege Escalation → Defense Evasion → Credential Access
        → Discovery → Lateral Movement → Collection
            → Command and Control → Exfiltration → Impact
```

Each tactic has techniques and sub-techniques mapped to real-world adversary behavior.

---

## 4. Adversary Emulation Plans

### 4.1 APT Emulation Process

1. **Select threat actor**: Based on client's industry and threat landscape
2. **Research TTPs**: Map ATT&CK techniques used by that actor
3. **Build emulation plan**: Step-by-step actions mimicking the actor
4. **Prepare tooling**: Select/develop tools matching actor's capabilities
5. **Execute**: Follow the plan with documented deviations
6. **Report**: Map findings to ATT&CK for actionable detection improvements

### 4.2 MITRE CTID Adversary Emulation Library

Pre-built emulation plans for:
- APT3 (Gothic Panda)
- APT29 (Cozy Bear)
- FIN6 (financial crime)
- Sandworm (Russian military intelligence)

---

## 5. Command and Control Infrastructure

### 5.1 C2 Architecture

```
Operator → Team Server → Redirector → Compromised Host
                              │
                    (CDN/Cloud Front)
                              │
                    Categorized Domain
```

### 5.2 Infrastructure Components

- **Team server**: Cobalt Strike, Sliver, Mythic
- **Redirectors**: Nginx/Apache reverse proxy to filter traffic
- **Domains**: Categorized domains (aged, with HTTPS)
- **CDN fronting**: Use CDN services to hide C2 traffic
- **DNS C2**: Slow but extremely difficult to detect

### 5.3 OPSEC Considerations

- Separate infrastructure per engagement
- Don't reuse domains across engagements
- Use HTTPS with valid certificates
- Implement redirector filtering (user-agent, IP, etc.)
- Destroy infrastructure after engagement

---

## 6. Social Engineering Campaigns

### 6.1 Phishing Campaign Phases

1. **OSINT**: Identify targets, email format, interests
2. **Pretext development**: Create believable scenario
3. **Infrastructure**: Domain, email server, landing page
4. **Payload**: Macro document, link to credential harvester
5. **Execution**: Send emails, monitor clicks
6. **Follow-up**: Exploit access gained

### 6.2 Types of Social Engineering

| Type | Medium | Goal |
|------|--------|------|
| Phishing | Email | Credential theft or payload delivery |
| Vishing | Phone | Information gathering |
| Smishing | SMS | Link click |
| Pretexting | In-person | Physical access |
| Baiting | USB/physical | Payload delivery |

---

## 7. Physical Security Testing

- **Tailgating**: Follow authorized person through secure door
- **Badge cloning**: Copy RFID/NFC badges
- **Lock picking**: Bypass physical locks
- **Dumpster diving**: Search discarded documents
- **Planted devices**: Drop boxes for network access

---

## 8. Evasion and OPSEC

### 8.1 Operational Security Rules

1. **Assume monitoring**: Act as if the SOC can see everything
2. **Minimize footprint**: Fewer tools, fewer connections
3. **Blend in**: Use normal business hours, common protocols
4. **Clean up**: Remove artifacts after each operation
5. **Compartmentalize**: Separate infrastructure per target
6. **Verify before acting**: Ensure actions stay in scope

### 8.2 EDR Evasion Techniques

- Process injection (avoiding suspicious parent-child relationships)
- AMSI bypass for PowerShell execution
- ETW patching to prevent telemetry
- Memory-only execution (fileless)
- Syscall stubs (bypass userland hooks)
- Timestomping and log manipulation

---

## 9. Purple Team Exercises

Purple teaming combines red and blue team efforts for mutual improvement.

### 9.1 Purple Team Workflow

```
1. Red team demonstrates technique (ATT&CK mapped)
2. Blue team attempts to detect in real-time
3. Both teams discuss visibility gaps
4. Blue team creates/improves detection rules
5. Red team verifies detection works
6. Document the detection and its coverage
```

### 9.2 Benefits

- Faster detection improvement than adversarial red teaming alone
- Knowledge transfer between offensive and defensive teams
- Direct mapping of detection coverage to ATT&CK
- Cost-effective compared to separate red and blue engagements

---

## 10. Reporting and Remediation

### 10.1 Red Team Report Structure

1. **Executive Summary** (2-3 pages)
   - Objectives achieved and overall risk assessment
   - Key findings in business language
   - Timeline of the engagement

2. **Attack Narrative** (10-20 pages)
   - Chronological story of the engagement
   - Each phase with screenshots and evidence
   - Mapped to MITRE ATT&CK

3. **Detection Gap Analysis**
   - What was detected vs. what was missed
   - Time-to-detection for each phase
   - Recommended detection improvements

4. **Remediation Roadmap**
   - Prioritized recommendations
   - Quick wins vs. long-term improvements
   - Security architecture recommendations

### 10.2 Metrics

| Metric | Description |
|--------|-------------|
| Time to Initial Access | How quickly the team gained a foothold |
| Time to Domain Admin | Duration from initial access to DA |
| Time to Detection | When the SOC first noticed activity |
| Time to Containment | When the SOC effectively blocked the team |
| Objectives Achieved | Which goals were accomplished |
| TTPs Used | Number of ATT&CK techniques employed |

---

## 11. Exercises

1. **Emulation Plan**: Create an adversary emulation plan for APT29 targeting a financial institution.
2. **C2 Infrastructure**: Set up a Sliver C2 server with a redirector and categorized domain.
3. **Phishing Campaign**: Design a phishing campaign (pretext, email, landing page) for a fictional company.
4. **Purple Team**: Conduct a purple team exercise testing 5 ATT&CK techniques.
5. **Report Writing**: Write a red team executive summary for a fictional engagement.
6. **OPSEC Review**: Review a set of red team actions and identify OPSEC failures.

---

## 12. Summary

Red team operations test an organization's complete security posture:

- **Red teaming** tests detection and response, not just vulnerabilities
- **Threat modeling** and adversary emulation ensure realistic testing
- **MITRE ATT&CK** provides the common language for TTPs
- **C2 infrastructure** requires careful OPSEC and setup
- **Social engineering** is often the most effective initial access vector
- **Purple teaming** accelerates security improvement
- **Professional reporting** translates findings into actionable improvements

This concludes the Cybersecurity Offensive curriculum. The skills across these 20 lessons provide a comprehensive foundation for authorized security testing, CTF competitions, and defensive security understanding.

---

## 13. References

- MITRE ATT&CK: https://attack.mitre.org/
- MITRE CTID Adversary Emulation Library: https://github.com/center-for-threat-informed-defense/adversary_emulation_library
- Red Team Field Manual (RTFM)
- Cobalt Strike: https://www.cobaltstrike.com/
- Sliver: https://github.com/BishopFox/sliver
- Atomic Red Team: https://github.com/redcanaryco/atomic-red-team
- The Red Team Guide: https://redteam.guide/
