"""
Incident Response Simulator

Demonstrates:
- NIST incident response lifecycle
- Incident classification and severity
- Response playbook execution
- Timeline tracking and post-mortem

Theory:
- NIST SP 800-61 defines four incident response phases:
  1. Preparation: tools, team, procedures
  2. Detection & Analysis: identify and assess incidents
  3. Containment, Eradication, Recovery: stop the threat,
     remove it, restore systems
  4. Post-Incident Activity: lessons learned, documentation
- Severity levels determine response urgency and escalation.
- Playbooks provide step-by-step procedures for known incident types.
- Mean Time to Detect (MTTD) and Mean Time to Respond (MTTR)
  are key performance metrics.

Adapted from Security Lesson 14.
"""

from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta


class Severity(Enum):
    CRITICAL = 1    # Business-threatening, immediate response
    HIGH = 2        # Significant impact, rapid response
    MEDIUM = 3      # Limited impact, standard response
    LOW = 4         # Minimal impact, routine handling


class Phase(Enum):
    PREPARATION = "Preparation"
    DETECTION = "Detection & Analysis"
    CONTAINMENT = "Containment"
    ERADICATION = "Eradication"
    RECOVERY = "Recovery"
    POST_INCIDENT = "Post-Incident"


class IncidentType(Enum):
    DATA_BREACH = "Data Breach"
    MALWARE = "Malware Infection"
    DDOS = "DDoS Attack"
    PHISHING = "Phishing Campaign"
    UNAUTHORIZED_ACCESS = "Unauthorized Access"
    INSIDER_THREAT = "Insider Threat"


@dataclass
class TimelineEntry:
    timestamp: datetime
    phase: Phase
    action: str
    actor: str
    notes: str = ""


@dataclass
class Incident:
    incident_id: str
    incident_type: IncidentType
    severity: Severity
    description: str
    detected_at: datetime
    status: str = "open"
    timeline: list[TimelineEntry] = field(default_factory=list)
    affected_systems: list[str] = field(default_factory=list)
    iocs: list[str] = field(default_factory=list)  # Indicators of Compromise
    resolved_at: datetime | None = None

    def add_event(self, phase: Phase, action: str, actor: str,
                  minutes_after: int = 0, notes: str = "") -> None:
        ts = self.detected_at + timedelta(minutes=minutes_after)
        self.timeline.append(TimelineEntry(ts, phase, action, actor, notes))

    @property
    def mttd(self) -> int:
        """Minutes from first anomaly to detection (simulated)."""
        return 15  # Simulated

    @property
    def mttr(self) -> int | None:
        """Minutes from detection to resolution."""
        if self.resolved_at:
            delta = self.resolved_at - self.detected_at
            return int(delta.total_seconds() / 60)
        return None


# ── Playbooks ─────────────────────────────────────────────────────────

PLAYBOOKS: dict[IncidentType, list[tuple[Phase, str]]] = {
    IncidentType.DATA_BREACH: [
        (Phase.DETECTION, "Identify affected data and systems"),
        (Phase.DETECTION, "Determine breach scope and timeline"),
        (Phase.DETECTION, "Classify data sensitivity (PII, PHI, financial)"),
        (Phase.CONTAINMENT, "Isolate affected systems from network"),
        (Phase.CONTAINMENT, "Revoke compromised credentials"),
        (Phase.CONTAINMENT, "Enable enhanced logging"),
        (Phase.ERADICATION, "Patch exploited vulnerability"),
        (Phase.ERADICATION, "Remove unauthorized access mechanisms"),
        (Phase.RECOVERY, "Restore from clean backups"),
        (Phase.RECOVERY, "Force password resets for affected users"),
        (Phase.RECOVERY, "Verify system integrity before reconnection"),
        (Phase.POST_INCIDENT, "Notify affected parties (regulatory req.)"),
        (Phase.POST_INCIDENT, "Conduct root cause analysis"),
        (Phase.POST_INCIDENT, "Update security controls"),
    ],
    IncidentType.DDOS: [
        (Phase.DETECTION, "Confirm DDoS (vs. legitimate traffic spike)"),
        (Phase.DETECTION, "Identify attack vector (volumetric, protocol, app)"),
        (Phase.DETECTION, "Determine attack source IPs/patterns"),
        (Phase.CONTAINMENT, "Enable DDoS mitigation service"),
        (Phase.CONTAINMENT, "Apply rate limiting"),
        (Phase.CONTAINMENT, "Block identified malicious IPs"),
        (Phase.CONTAINMENT, "Scale infrastructure if needed"),
        (Phase.ERADICATION, "Update firewall rules"),
        (Phase.ERADICATION, "Implement geo-blocking if applicable"),
        (Phase.RECOVERY, "Gradually restore normal traffic flow"),
        (Phase.RECOVERY, "Monitor for attack resumption"),
        (Phase.POST_INCIDENT, "Analyze attack patterns"),
        (Phase.POST_INCIDENT, "Improve DDoS resilience"),
    ],
    IncidentType.MALWARE: [
        (Phase.DETECTION, "Identify malware type and behavior"),
        (Phase.DETECTION, "Determine infection vector"),
        (Phase.DETECTION, "Identify affected endpoints"),
        (Phase.CONTAINMENT, "Isolate infected systems"),
        (Phase.CONTAINMENT, "Block C2 communication"),
        (Phase.CONTAINMENT, "Quarantine malware samples"),
        (Phase.ERADICATION, "Run full antimalware scans"),
        (Phase.ERADICATION, "Remove malware artifacts"),
        (Phase.ERADICATION, "Patch exploitation vectors"),
        (Phase.RECOVERY, "Reimage severely infected systems"),
        (Phase.RECOVERY, "Restore data from clean backups"),
        (Phase.RECOVERY, "Verify system integrity"),
        (Phase.POST_INCIDENT, "Submit samples to threat intel"),
        (Phase.POST_INCIDENT, "Update detection signatures"),
    ],
}


class IncidentResponseTeam:
    """Simulates incident response execution."""

    def __init__(self):
        self.incidents: list[Incident] = []
        self.log: list[str] = []

    def create_incident(self, incident_type: IncidentType,
                        severity: Severity, description: str,
                        affected: list[str]) -> Incident:
        incident = Incident(
            incident_id=f"INC-{len(self.incidents) + 1:04d}",
            incident_type=incident_type,
            severity=severity,
            description=description,
            detected_at=datetime(2025, 6, 15, 14, 30),
            affected_systems=affected,
        )
        self.incidents.append(incident)
        return incident

    def execute_playbook(self, incident: Incident) -> None:
        """Run the playbook for the incident type."""
        playbook = PLAYBOOKS.get(incident.incident_type, [])
        if not playbook:
            self.log.append(f"No playbook for {incident.incident_type.value}")
            return

        minute = 0
        for phase, action in playbook:
            # Simulate time progression
            if phase == Phase.CONTAINMENT:
                minute = max(minute, 30)
            elif phase == Phase.ERADICATION:
                minute = max(minute, 120)
            elif phase == Phase.RECOVERY:
                minute = max(minute, 240)
            elif phase == Phase.POST_INCIDENT:
                minute = max(minute, 480)

            incident.add_event(phase, action, "IR Team", minute)
            minute += 15

        incident.resolved_at = (
            incident.detected_at + timedelta(minutes=minute))
        incident.status = "resolved"


def calculate_metrics(incidents: list[Incident]) -> dict:
    """Calculate IR performance metrics."""
    resolved = [i for i in incidents if i.resolved_at]
    if not resolved:
        return {}

    mttrs = [i.mttr for i in resolved if i.mttr is not None]
    by_severity = {}
    for i in resolved:
        sev = i.severity.name
        if sev not in by_severity:
            by_severity[sev] = []
        if i.mttr is not None:
            by_severity[sev].append(i.mttr)

    return {
        "total_incidents": len(incidents),
        "resolved": len(resolved),
        "avg_mttr_minutes": sum(mttrs) / len(mttrs) if mttrs else 0,
        "by_severity": {
            s: sum(v) / len(v) for s, v in by_severity.items() if v
        },
    }


# ── Demos ─────────────────────────────────────────────────────────────

def demo_data_breach():
    print("=" * 60)
    print("INCIDENT RESPONSE: DATA BREACH")
    print("=" * 60)

    team = IncidentResponseTeam()
    incident = team.create_incident(
        incident_type=IncidentType.DATA_BREACH,
        severity=Severity.CRITICAL,
        description="Customer PII exposed via SQL injection on /api/users",
        affected=["web-app-01", "db-primary", "db-replica"],
    )
    incident.iocs = [
        "SQL injection payload in access logs",
        "Unusual data export volume from db-primary",
        "New admin account created: 'backdoor_user'",
    ]

    print(f"\n  Incident: {incident.incident_id}")
    print(f"  Type: {incident.incident_type.value}")
    print(f"  Severity: {incident.severity.name}")
    print(f"  Description: {incident.description}")
    print(f"  Affected: {incident.affected_systems}")

    print(f"\n  IoCs (Indicators of Compromise):")
    for ioc in incident.iocs:
        print(f"    - {ioc}")

    # Execute playbook
    team.execute_playbook(incident)

    print(f"\n  Playbook Execution:")
    current_phase = None
    for entry in incident.timeline:
        if entry.phase != current_phase:
            current_phase = entry.phase
            print(f"\n  --- {current_phase.value} ---")
        time_str = entry.timestamp.strftime("%H:%M")
        print(f"    [{time_str}] {entry.action}")

    print(f"\n  Status: {incident.status}")
    print(f"  MTTR: {incident.mttr} minutes "
          f"({incident.mttr // 60}h {incident.mttr % 60}m)")


def demo_ddos():
    print("\n" + "=" * 60)
    print("INCIDENT RESPONSE: DDoS ATTACK")
    print("=" * 60)

    team = IncidentResponseTeam()
    incident = team.create_incident(
        incident_type=IncidentType.DDOS,
        severity=Severity.HIGH,
        description="Volumetric DDoS: 50 Gbps UDP flood on port 443",
        affected=["load-balancer", "web-frontend-01", "web-frontend-02"],
    )

    team.execute_playbook(incident)

    print(f"\n  {incident.incident_id}: {incident.description}")
    print(f"  Severity: {incident.severity.name}")

    # Show phase summary
    phases: dict[str, int] = {}
    for entry in incident.timeline:
        phases[entry.phase.value] = phases.get(entry.phase.value, 0) + 1

    print(f"\n  Phase summary:")
    for phase, count in phases.items():
        print(f"    {phase}: {count} actions")

    print(f"\n  MTTR: {incident.mttr} minutes")


def demo_severity_matrix():
    print("\n" + "=" * 60)
    print("SEVERITY CLASSIFICATION MATRIX")
    print("=" * 60)

    print(f"""
  {'Severity':<12} {'Response':<20} {'Escalation':<25} {'Example':<20}
  {'-'*12} {'-'*20} {'-'*25} {'-'*20}
  {'CRITICAL':<12} {'Immediate (15 min)':<20} {'CISO + Legal + Exec':<25} {'Data breach':<20}
  {'HIGH':<12} {'Rapid (1 hour)':<20} {'Security Lead + Mgmt':<25} {'Active intrusion':<20}
  {'MEDIUM':<12} {'Standard (4 hours)':<20} {'Security Team':<25} {'Phishing success':<20}
  {'LOW':<12} {'Routine (24 hours)':<20} {'On-call engineer':<25} {'Port scan':<20}
""")


def demo_metrics():
    print("=" * 60)
    print("INCIDENT RESPONSE METRICS")
    print("=" * 60)

    team = IncidentResponseTeam()

    scenarios = [
        (IncidentType.DATA_BREACH, Severity.CRITICAL,
         "PII exposure", ["db-01"]),
        (IncidentType.DDOS, Severity.HIGH,
         "UDP flood", ["lb-01"]),
        (IncidentType.MALWARE, Severity.HIGH,
         "Ransomware detected", ["ws-05"]),
    ]

    for itype, severity, desc, affected in scenarios:
        incident = team.create_incident(itype, severity, desc, affected)
        team.execute_playbook(incident)

    metrics = calculate_metrics(team.incidents)

    print(f"\n  Total incidents: {metrics['total_incidents']}")
    print(f"  Resolved: {metrics['resolved']}")
    print(f"  Average MTTR: {metrics['avg_mttr_minutes']:.0f} minutes "
          f"({metrics['avg_mttr_minutes'] / 60:.1f} hours)")

    print(f"\n  MTTR by severity:")
    for sev, mttr in sorted(metrics["by_severity"].items()):
        print(f"    {sev}: {mttr:.0f} minutes ({mttr / 60:.1f} hours)")


def demo_nist_lifecycle():
    print("\n" + "=" * 60)
    print("NIST INCIDENT RESPONSE LIFECYCLE")
    print("=" * 60)

    print(f"""
  ┌──────────────────────────────────────────────────────┐
  │              NIST SP 800-61 Lifecycle                │
  │                                                      │
  │  ┌─────────────┐    ┌──────────────────┐            │
  │  │ Preparation │───▶│ Detection &      │            │
  │  │             │    │ Analysis         │            │
  │  └─────────────┘    └────────┬─────────┘            │
  │        ▲                     │                       │
  │        │                     ▼                       │
  │  ┌─────┴─────────┐  ┌──────────────────┐           │
  │  │ Post-Incident │◀─│ Containment,     │           │
  │  │ Activity      │  │ Eradication &    │           │
  │  └───────────────┘  │ Recovery         │           │
  │                     └──────────────────┘            │
  └──────────────────────────────────────────────────────┘

  Key Documents:
    - Incident Response Plan (IRP)
    - Communication plan (internal + external)
    - Playbooks per incident type
    - Contact lists (on-call, legal, PR)
    - Evidence collection procedures

  Key Metrics:
    - MTTD: Mean Time to Detect
    - MTTR: Mean Time to Respond/Resolve
    - Incidents per month/quarter
    - False positive rate
    - Escalation accuracy""")


if __name__ == "__main__":
    demo_data_breach()
    demo_ddos()
    demo_severity_matrix()
    demo_metrics()
    demo_nist_lifecycle()
