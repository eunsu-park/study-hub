#!/usr/bin/env python3
"""Example: Incident Response — Automated Triage, Runbooks & Communication

Demonstrates incident management: severity classification, automated
runbook execution, stakeholder notification, timeline tracking, and
post-incident analysis with MTTR/MTTD metrics.
Related lesson: 26_Incident_Response.md
"""

# =============================================================================
# WHY FORMALIZED INCIDENT RESPONSE?
# Without a structured process, incidents devolve into chaos. A clear
# framework — severity levels, roles (IC, comms lead), runbooks, and
# automated escalation — reduces Mean Time to Detect (MTTD) and Mean Time
# to Resolve (MTTR) while preserving a blameless learning culture.
# =============================================================================

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Optional


# =============================================================================
# 1. INCIDENT MODEL
# =============================================================================

class Severity(Enum):
    SEV1 = "SEV1"  # Critical: revenue impact, data loss
    SEV2 = "SEV2"  # Major: significant degradation
    SEV3 = "SEV3"  # Minor: partial degradation, workaround exists
    SEV4 = "SEV4"  # Low: cosmetic or minor issues


class IncidentPhase(Enum):
    DETECTED = "detected"
    TRIAGED = "triaged"
    MITIGATING = "mitigating"
    RESOLVED = "resolved"
    POSTMORTEM = "postmortem"
    CLOSED = "closed"


@dataclass
class TimelineEntry:
    """A timestamped event in the incident timeline."""
    timestamp: datetime
    phase: IncidentPhase
    message: str
    actor: str = "system"


@dataclass
class Incident:
    """A tracked incident with full lifecycle metadata."""
    id: str
    title: str
    severity: Severity
    phase: IncidentPhase = IncidentPhase.DETECTED
    commander: str = ""
    comms_lead: str = ""
    affected_services: list[str] = field(default_factory=list)
    timeline: list[TimelineEntry] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: Optional[datetime] = None
    tags: list[str] = field(default_factory=list)

    def add_event(self, message: str, phase: IncidentPhase | None = None,
                  actor: str = "system") -> None:
        if phase:
            self.phase = phase
        self.timeline.append(TimelineEntry(
            timestamp=datetime.now(timezone.utc),
            phase=self.phase, message=message, actor=actor,
        ))

    def resolve(self, message: str = "Incident resolved", actor: str = "system") -> None:
        self.resolved_at = datetime.now(timezone.utc)
        self.add_event(message, IncidentPhase.RESOLVED, actor)

    @property
    def mttd_minutes(self) -> float:
        """Time from creation to first triage event."""
        for entry in self.timeline:
            if entry.phase == IncidentPhase.TRIAGED:
                return (entry.timestamp - self.created_at).total_seconds() / 60
        return 0.0

    @property
    def mttr_minutes(self) -> float:
        """Time from creation to resolution."""
        if self.resolved_at:
            return (self.resolved_at - self.created_at).total_seconds() / 60
        return 0.0


# =============================================================================
# 2. SEVERITY CLASSIFIER
# =============================================================================

@dataclass
class SeverityRule:
    """A rule for automatic severity classification."""
    name: str
    condition: Callable[[dict[str, Any]], bool]
    severity: Severity


SEVERITY_RULES = [
    SeverityRule(
        "Total outage",
        lambda ctx: ctx.get("error_rate", 0) > 0.95 or ctx.get("availability", 1) < 0.01,
        Severity.SEV1,
    ),
    SeverityRule(
        "Major degradation",
        lambda ctx: ctx.get("error_rate", 0) > 0.5 or ctx.get("latency_p99_ms", 0) > 5000,
        Severity.SEV2,
    ),
    SeverityRule(
        "Partial degradation",
        lambda ctx: ctx.get("error_rate", 0) > 0.1 or ctx.get("latency_p99_ms", 0) > 2000,
        Severity.SEV3,
    ),
    SeverityRule(
        "Minor issue",
        lambda ctx: True,  # Default fallback
        Severity.SEV4,
    ),
]


def classify_severity(context: dict[str, Any]) -> tuple[Severity, str]:
    """Classify incident severity from alert context."""
    for rule in SEVERITY_RULES:
        if rule.condition(context):
            return rule.severity, rule.name
    return Severity.SEV4, "Default"


# =============================================================================
# 3. RUNBOOK ENGINE
# =============================================================================

@dataclass
class RunbookStep:
    """A step in an automated runbook."""
    name: str
    action: str  # Description of the action
    automated: bool = False
    timeout_seconds: int = 60


@dataclass
class Runbook:
    """An incident response runbook."""
    name: str
    trigger: str  # What alert triggers this runbook
    steps: list[RunbookStep] = field(default_factory=list)

    def execute(self, incident: Incident) -> list[dict[str, Any]]:
        """Execute the runbook steps (simulated)."""
        results = []
        for i, step in enumerate(self.steps):
            result = {
                "step": i + 1,
                "name": step.name,
                "automated": step.automated,
                "status": "executed" if step.automated else "manual_required",
                "action": step.action,
            }
            results.append(result)
            incident.add_event(
                f"Runbook step {i+1}: {step.name} — {result['status']}",
                IncidentPhase.MITIGATING,
            )
        return results


# Pre-defined runbooks
RUNBOOKS = {
    "high_error_rate": Runbook(
        name="High Error Rate Response",
        trigger="error_rate > 10%",
        steps=[
            RunbookStep("Check recent deployments", "List deploys in last 2h", automated=True),
            RunbookStep("Check dependency health", "Query health endpoints", automated=True),
            RunbookStep("Increase replicas", "Scale deployment to 2x", automated=True),
            RunbookStep("Rollback if deploy-related", "Rollback to previous version", automated=False),
            RunbookStep("Notify stakeholders", "Send Slack/PagerDuty update", automated=True),
        ],
    ),
    "database_overload": Runbook(
        name="Database Overload Response",
        trigger="db_connections > 90%",
        steps=[
            RunbookStep("Identify top queries", "Query pg_stat_statements", automated=True),
            RunbookStep("Kill long-running queries", "Terminate queries > 30s", automated=True),
            RunbookStep("Enable read replicas", "Route reads to replicas", automated=False),
            RunbookStep("Page DBA on-call", "Escalate to database team", automated=True),
        ],
    ),
}


# =============================================================================
# 4. NOTIFICATION MANAGER
# =============================================================================

@dataclass
class NotificationChannel:
    """A notification channel for incident communication."""
    name: str
    channel_type: str  # slack, pagerduty, email
    target: str        # Channel ID, email, etc.


ESCALATION_MATRIX: dict[Severity, list[str]] = {
    Severity.SEV1: ["pagerduty-oncall", "slack-incidents", "email-execs"],
    Severity.SEV2: ["pagerduty-oncall", "slack-incidents"],
    Severity.SEV3: ["slack-incidents"],
    Severity.SEV4: ["slack-alerts"],
}


def generate_notifications(incident: Incident) -> list[dict[str, str]]:
    """Generate notification payloads based on severity escalation matrix."""
    channels = ESCALATION_MATRIX.get(incident.severity, [])
    notifications = []
    for ch in channels:
        notifications.append({
            "channel": ch,
            "title": f"[{incident.severity.value}] {incident.title}",
            "body": (f"Incident {incident.id}: {incident.title}\n"
                     f"Severity: {incident.severity.value}\n"
                     f"Services: {', '.join(incident.affected_services)}\n"
                     f"Commander: {incident.commander or 'Unassigned'}"),
        })
    return notifications


# =============================================================================
# 5. INCIDENT METRICS
# =============================================================================

def compute_incident_metrics(incidents: list[Incident]) -> dict[str, Any]:
    """Compute aggregate incident metrics."""
    resolved = [i for i in incidents if i.resolved_at]
    if not resolved:
        return {"total": len(incidents), "resolved": 0}
    mttrs = [i.mttr_minutes for i in resolved]
    mttds = [i.mttd_minutes for i in resolved if i.mttd_minutes > 0]
    by_severity = {}
    for i in incidents:
        by_severity[i.severity.value] = by_severity.get(i.severity.value, 0) + 1
    return {
        "total": len(incidents),
        "resolved": len(resolved),
        "avg_mttr_min": round(sum(mttrs) / len(mttrs), 1),
        "avg_mttd_min": round(sum(mttds) / len(mttds), 1) if mttds else 0,
        "by_severity": by_severity,
    }


# =============================================================================
# 6. DEMO
# =============================================================================

if __name__ == "__main__":
    # --- Severity Classification ---
    print("=" * 60)
    print("Automated Severity Classification")
    print("=" * 60)
    alerts = [
        {"error_rate": 0.98, "availability": 0.0, "service": "payment-api"},
        {"error_rate": 0.55, "latency_p99_ms": 6000, "service": "order-svc"},
        {"error_rate": 0.15, "latency_p99_ms": 2500, "service": "user-svc"},
        {"error_rate": 0.02, "latency_p99_ms": 200, "service": "healthcheck"},
    ]
    incidents = []
    for idx, alert in enumerate(alerts):
        sev, rule_name = classify_severity(alert)
        inc = Incident(
            id=f"INC-{idx+1:04d}", title=f"{alert['service']} degradation",
            severity=sev, affected_services=[alert["service"]],
            commander="alice",
        )
        inc.add_event(f"Auto-classified as {sev.value} ({rule_name})",
                       IncidentPhase.TRIAGED, "auto-triage")
        incidents.append(inc)
        print(f"  {inc.id}: {sev.value} — {rule_name} ({alert['service']})")

    # --- Runbook Execution ---
    print(f"\n{'=' * 60}")
    print("Runbook Execution")
    print("=" * 60)
    runbook = RUNBOOKS["high_error_rate"]
    results = runbook.execute(incidents[0])
    for r in results:
        status = "AUTO" if r["automated"] else "MANUAL"
        print(f"  Step {r['step']}: [{status}] {r['name']}")

    # --- Notifications ---
    print(f"\n{'=' * 60}")
    print("Incident Notifications")
    print("=" * 60)
    for inc in incidents[:2]:
        notifs = generate_notifications(inc)
        print(f"  {inc.id} ({inc.severity.value}):")
        for n in notifs:
            print(f"    -> {n['channel']}: {n['title']}")

    # --- Resolve and Metrics ---
    for inc in incidents:
        inc.resolve("Service recovered", "on-call")
    print(f"\n{'=' * 60}")
    print("Incident Metrics")
    print("=" * 60)
    metrics = compute_incident_metrics(incidents)
    for k, v in metrics.items():
        print(f"  {k}: {v}")
