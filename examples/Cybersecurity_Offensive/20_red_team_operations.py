"""
Example: Red Team Operations
===============================
C2 infrastructure planner, OPSEC evaluator, threat emulation mapper,
and engagement report generator.

IMPORTANT: For authorized security testing and CTF only.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


# ---------------------------------------------------------------------------
# C2 Infrastructure Model
# ---------------------------------------------------------------------------

class C2Protocol(Enum):
    HTTPS = ("HTTPS", 443, "Blends with web traffic")
    DNS = ("DNS", 53, "Difficult to block, low bandwidth")
    WEBSOCKET = ("WebSocket", 443, "Persistent bidirectional channel")
    DOH = ("DNS-over-HTTPS", 443, "Encrypted DNS tunneling")


@dataclass
class Redirector:
    domain: str
    ip: str
    purpose: str  # "primary", "fallback", "short-haul"
    protocol: C2Protocol

    def __str__(self) -> str:
        return (f"{self.domain} ({self.ip}) "
                f"[{self.protocol.value[0]}/{self.purpose}]")


@dataclass
class C2Infrastructure:
    team_server_ip: str
    redirectors: list[Redirector]
    fallback_channels: list[C2Protocol]
    domain_fronting: dict | None = None
    ssl_cert: str = "Let's Encrypt"

    def display(self) -> str:
        lines = [
            "C2 Infrastructure Layout",
            "=" * 40,
            f"Team Server: {self.team_server_ip} (never exposed directly)",
            f"SSL: {self.ssl_cert}",
            "\nRedirectors:",
        ]
        for r in self.redirectors:
            lines.append(f"  {r}")
        lines.append(f"\nFallback: {[c.value[0] for c in self.fallback_channels]}")
        if self.domain_fronting:
            lines.append(f"Domain Fronting: {self.domain_fronting}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# OPSEC Evaluation
# ---------------------------------------------------------------------------

class OpsecRating(Enum):
    GOOD = "good"
    BAD = "bad"
    CRITICAL = "critical"


@dataclass
class OpsecCheck:
    action: str
    rating: OpsecRating
    issue: str
    remediation: str


OPSEC_RULES = {
    "personal laptop": OpsecCheck(
        "Personal device for ops", OpsecRating.CRITICAL,
        "Personal device links operator identity to engagement",
        "Use dedicated, sanitized hardware for all operations"),
    "real name": OpsecCheck(
        "Real identity in registrations", OpsecRating.CRITICAL,
        "Domain WHOIS exposes operator identity",
        "Use privacy-protected registration and pseudonyms"),
    "default": OpsecCheck(
        "Default tool configurations", OpsecRating.BAD,
        "Default signatures easily detected by EDR/IDS",
        "Customize malleable C2 profiles and tool signatures"),
    "reused infrastructure": OpsecCheck(
        "Infrastructure reuse", OpsecRating.BAD,
        "Previous engagement IOCs may be known to target",
        "Fresh infrastructure per engagement"),
    "encrypted comms": OpsecCheck(
        "Encrypted team communication", OpsecRating.GOOD,
        "N/A", "Continue using encrypted channels"),
    "encrypted storage": OpsecCheck(
        "Encrypted data storage", OpsecRating.GOOD,
        "N/A", "Continue using full-disk encryption"),
}


# ---------------------------------------------------------------------------
# MITRE ATT&CK Technique Emulation
# ---------------------------------------------------------------------------

@dataclass
class EmulationStep:
    tactic: str
    technique_id: str
    technique_name: str
    tool: str
    procedure: str
    detection_opportunity: str


APT29_EMULATION = [
    EmulationStep(
        "Initial Access", "T1566.001", "Spearphishing Attachment",
        "GoPhish + Macro-enabled document",
        "Send targeted phishing email with malicious DOCX attachment",
        "Email gateway scanning, macro execution alerts"),
    EmulationStep(
        "Execution", "T1059.001", "PowerShell",
        "Custom PS1 stager",
        "Execute encoded PowerShell command from macro",
        "PowerShell ScriptBlock logging, AMSI"),
    EmulationStep(
        "Persistence", "T1053.005", "Scheduled Task",
        "schtasks.exe",
        "Create scheduled task for persistent callback",
        "Event ID 4698, scheduled task creation monitoring"),
    EmulationStep(
        "Defense Evasion", "T1027", "Obfuscated Files",
        "Invoke-Obfuscation / custom packer",
        "Obfuscate PowerShell and encode payloads",
        "Entropy analysis, behavioral detection"),
    EmulationStep(
        "Credential Access", "T1003.001", "LSASS Memory",
        "Mimikatz (reflective DLL) or nanodump",
        "Dump LSASS process memory for credential extraction",
        "Sysmon Event ID 10 (process access to lsass)"),
    EmulationStep(
        "Lateral Movement", "T1021.006", "Windows Remote Management",
        "Evil-WinRM or CrackMapExec",
        "Use harvested credentials to WinRM into target hosts",
        "Event ID 4624 Type 3, WinRM connection logs"),
]


# ---------------------------------------------------------------------------
# Engagement Report Finding
# ---------------------------------------------------------------------------

@dataclass
class ReportFinding:
    title: str
    severity: str  # critical, high, medium, low
    description: str
    impact: str
    mitre_techniques: list[str]
    evidence: list[str]
    recommendation: str
    executive_summary: str

    def to_markdown(self) -> str:
        return f"""### {self.title}
**Severity:** {self.severity.upper()}
**MITRE ATT&CK:** {', '.join(self.mitre_techniques)}

**Description:** {self.description}

**Business Impact:** {self.impact}

**Recommendation:** {self.recommendation}

**Executive Summary:** {self.executive_summary}
"""


# ---------------------------------------------------------------------------
# Engagement Timeline
# ---------------------------------------------------------------------------

@dataclass
class TimelineEntry:
    day: int
    phase: str
    action: str
    result: str


def generate_timeline(duration_days: int) -> list[TimelineEntry]:
    """Generate a typical red team engagement timeline."""
    timeline = []
    phases = [
        (1, 3, "Reconnaissance", [
            ("OSINT and passive recon", "Identified key personnel and tech stack"),
            ("Infrastructure setup", "C2 and phishing infra ready"),
        ]),
        (4, 7, "Initial Access", [
            ("Phishing campaign", "Obtained initial foothold via macro"),
            ("Establish C2", "Persistent callback to team server"),
        ]),
        (8, 14, "Discovery & Escalation", [
            ("Internal enumeration", "Mapped AD structure and trusts"),
            ("Privilege escalation", "Obtained Domain Admin credentials"),
        ]),
        (15, 21, "Lateral Movement", [
            ("Move to high-value targets", "Accessed financial systems"),
            ("Data identification", "Located sensitive PII database"),
        ]),
        (22, duration_days, "Objectives & Cleanup", [
            ("Demonstrate impact", "Exfiltrated sample data set"),
            ("Cleanup and reporting", "Removed all implants and artifacts"),
        ]),
    ]
    for start, end, phase, actions in phases:
        if start > duration_days:
            break
        for i, (action, result) in enumerate(actions):
            day = min(start + i, duration_days)
            timeline.append(TimelineEntry(day, phase, action, result))
    return timeline


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    print("Red Team Operations Examples")
    print("=" * 50)

    # C2 Infrastructure
    infra = C2Infrastructure(
        team_server_ip="10.10.10.100 (internal)",
        redirectors=[
            Redirector("update.legit-looking.com", "203.0.113.10",
                        "primary", C2Protocol.HTTPS),
            Redirector("cdn-assets.example.net", "203.0.113.20",
                        "fallback", C2Protocol.DOH),
        ],
        fallback_channels=[C2Protocol.DNS, C2Protocol.DOH],
        domain_fronting={"CDN": "CloudFront",
                         "front": "d1234.cloudfront.net"},
    )
    print(f"\n{infra.display()}")

    # OPSEC evaluation
    print("\nOPSEC Evaluation:")
    for key, check in OPSEC_RULES.items():
        icon = {OpsecRating.GOOD: "OK", OpsecRating.BAD: "WARN",
                OpsecRating.CRITICAL: "FAIL"}[check.rating]
        print(f"  [{icon:4s}] {check.action}")
        if check.rating != OpsecRating.GOOD:
            print(f"         Fix: {check.remediation}")

    # Threat emulation
    print("\nAPT29 Emulation Plan:")
    for step in APT29_EMULATION:
        print(f"  [{step.tactic:20s}] {step.technique_id} {step.technique_name}")
        print(f"    Tool: {step.tool}")
        print(f"    Detect: {step.detection_opportunity}")

    # Report finding
    finding = ReportFinding(
        title="Domain Admin Compromise via Kerberoasting",
        severity="critical",
        description="Obtained Domain Admin credentials by requesting "
                    "service tickets for accounts with weak passwords.",
        impact="Full control of Active Directory environment, "
               "access to all systems and data.",
        mitre_techniques=["T1558.003"],
        evidence=["krb_ticket.kirbi", "da_session.png"],
        recommendation="Enforce 25+ character passwords for service accounts, "
                       "use gMSA, monitor Kerberos TGS requests.",
        executive_summary="The red team gained full control of the corporate "
                          "network by cracking a service account password.",
    )
    print(f"\nSample Report Finding:")
    print(finding.to_markdown())

    # Engagement timeline
    print("Engagement Timeline (30 days):")
    for entry in generate_timeline(30):
        print(f"  Day {entry.day:2d} [{entry.phase:25s}] "
              f"{entry.action} -> {entry.result}")


if __name__ == "__main__":
    demo()
