#!/usr/bin/env python3
"""Example: Monitoring & Alerting — Rule Engine, Silencing & Escalation

Demonstrates monitoring and alerting infrastructure: alert rule evaluation,
multi-condition alerting, silence/inhibition rules, escalation chains,
and alert fatigue analysis.
Related lesson: 11_Monitoring_and_Alerting.md (extended coverage)
"""

# =============================================================================
# WHY STRUCTURED ALERTING?
# An alert that fires too often gets ignored. An alert that fires too late
# causes outages. A well-designed alerting system has clear severity levels,
# routing rules, silence windows, escalation policies, and fatigue metrics
# to ensure the right person gets the right alert at the right time.
# =============================================================================

import time
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Optional


# =============================================================================
# 1. ALERT RULE ENGINE
# =============================================================================

class AlertState(Enum):
    INACTIVE = "inactive"
    PENDING = "pending"   # Condition met but within `for` duration
    FIRING = "firing"
    RESOLVED = "resolved"


@dataclass
class AlertRule:
    """A Prometheus-style alerting rule."""
    name: str
    expr: str               # Human-readable expression
    evaluate_fn: Callable[[dict[str, float]], bool]  # Actual evaluation
    for_seconds: int = 0    # Must be true for this duration
    severity: str = "warning"
    labels: dict[str, str] = field(default_factory=dict)
    annotations: dict[str, str] = field(default_factory=dict)

    # Internal state
    state: AlertState = AlertState.INACTIVE
    pending_since: Optional[float] = None
    fired_at: Optional[float] = None

    def evaluate(self, metrics: dict[str, float]) -> AlertState:
        """Evaluate the rule against current metrics."""
        condition_met = self.evaluate_fn(metrics)

        if condition_met:
            if self.state == AlertState.INACTIVE or self.state == AlertState.RESOLVED:
                self.state = AlertState.PENDING
                self.pending_since = time.time()
            if self.state == AlertState.PENDING:
                elapsed = time.time() - (self.pending_since or time.time())
                if elapsed >= self.for_seconds:
                    self.state = AlertState.FIRING
                    self.fired_at = time.time()
        else:
            if self.state in (AlertState.FIRING, AlertState.PENDING):
                self.state = AlertState.RESOLVED
            else:
                self.state = AlertState.INACTIVE
            self.pending_since = None

        return self.state


# =============================================================================
# 2. SILENCE AND INHIBITION RULES
# =============================================================================

@dataclass
class SilenceRule:
    """Suppresses alerts matching certain labels during a time window."""
    id: str
    matchers: dict[str, str]  # label -> value
    starts_at: datetime
    ends_at: datetime
    created_by: str
    comment: str = ""

    def matches(self, alert_labels: dict[str, str]) -> bool:
        """Check if an alert matches this silence."""
        now = datetime.now(timezone.utc)
        if not (self.starts_at <= now <= self.ends_at):
            return False
        return all(
            alert_labels.get(k) == v for k, v in self.matchers.items()
        )


@dataclass
class InhibitionRule:
    """Inhibits target alerts when a source alert is firing."""
    source_match: dict[str, str]
    target_match: dict[str, str]
    equal_labels: list[str]  # Labels that must match between source/target

    def should_inhibit(self, source_labels: dict[str, str],
                       target_labels: dict[str, str]) -> bool:
        # Check source matches
        if not all(source_labels.get(k) == v for k, v in self.source_match.items()):
            return False
        # Check target matches
        if not all(target_labels.get(k) == v for k, v in self.target_match.items()):
            return False
        # Check equal labels
        return all(
            source_labels.get(l) == target_labels.get(l) for l in self.equal_labels
        )


# =============================================================================
# 3. ESCALATION POLICY
# =============================================================================

@dataclass
class EscalationStep:
    """A step in an escalation chain."""
    delay_minutes: int
    targets: list[str]
    channel: str  # slack, pagerduty, email


@dataclass
class EscalationPolicy:
    """Multi-step escalation policy for unacknowledged alerts."""
    name: str
    steps: list[EscalationStep] = field(default_factory=list)

    def get_current_step(self, minutes_since_fire: float) -> Optional[EscalationStep]:
        """Determine which escalation step should be active."""
        current_step = None
        for step in self.steps:
            if minutes_since_fire >= step.delay_minutes:
                current_step = step
        return current_step


# Pre-defined escalation policies
ESCALATION_POLICIES = {
    "critical": EscalationPolicy(
        name="critical-escalation",
        steps=[
            EscalationStep(0, ["on-call-primary"], "pagerduty"),
            EscalationStep(15, ["on-call-secondary", "team-lead"], "pagerduty"),
            EscalationStep(30, ["engineering-manager", "vp-engineering"], "phone"),
            EscalationStep(60, ["cto"], "phone"),
        ],
    ),
    "warning": EscalationPolicy(
        name="warning-escalation",
        steps=[
            EscalationStep(0, ["on-call-primary"], "slack"),
            EscalationStep(60, ["on-call-secondary"], "pagerduty"),
        ],
    ),
}


# =============================================================================
# 4. ALERT FATIGUE ANALYZER
# =============================================================================

@dataclass
class AlertEvent:
    """A recorded alert firing/resolution event."""
    alert_name: str
    state: AlertState
    timestamp: float
    acknowledged: bool = False


@dataclass
class FatigueAnalyzer:
    """Analyze alert patterns to detect alert fatigue."""
    events: list[AlertEvent] = field(default_factory=list)

    def record(self, event: AlertEvent) -> None:
        self.events.append(event)

    def analyze(self, window_hours: int = 24) -> dict[str, Any]:
        cutoff = time.time() - window_hours * 3600
        recent = [e for e in self.events if e.timestamp >= cutoff]
        firing = [e for e in recent if e.state == AlertState.FIRING]

        # Group by alert name
        by_name: dict[str, int] = {}
        for e in firing:
            by_name[e.alert_name] = by_name.get(e.alert_name, 0) + 1

        # Identify noisy alerts (>10 firings in window)
        noisy = {k: v for k, v in by_name.items() if v > 10}
        ack_rate = (
            sum(1 for e in firing if e.acknowledged) / len(firing)
            if firing else 0
        )
        return {
            "total_alerts": len(firing),
            "unique_alerts": len(by_name),
            "noisy_alerts": noisy,
            "ack_rate": round(ack_rate, 2),
            "fatigue_risk": "HIGH" if ack_rate < 0.3 else
                            "MEDIUM" if ack_rate < 0.6 else "LOW",
            "recommendation": (
                "Tune or silence noisy alerts" if noisy else
                "Alert volume is healthy"
            ),
        }


# =============================================================================
# 5. DEMO
# =============================================================================

if __name__ == "__main__":
    random.seed(42)

    # --- Alert Rule Evaluation ---
    print("=" * 60)
    print("Alert Rule Engine")
    print("=" * 60)
    rules = [
        AlertRule(
            name="HighErrorRate",
            expr='rate(http_errors[5m]) / rate(http_requests[5m]) > 0.05',
            evaluate_fn=lambda m: m.get("error_rate", 0) > 0.05,
            for_seconds=0, severity="critical",
            labels={"service": "api", "severity": "critical"},
        ),
        AlertRule(
            name="HighLatency",
            expr='histogram_quantile(0.99, http_duration) > 2.0',
            evaluate_fn=lambda m: m.get("latency_p99", 0) > 2.0,
            for_seconds=0, severity="warning",
            labels={"service": "api", "severity": "warning"},
        ),
        AlertRule(
            name="DiskAlmostFull",
            expr='node_filesystem_avail_bytes / node_filesystem_size_bytes < 0.1',
            evaluate_fn=lambda m: m.get("disk_free_pct", 1) < 0.1,
            for_seconds=0, severity="warning",
            labels={"service": "node", "severity": "warning"},
        ),
    ]
    metrics = {"error_rate": 0.08, "latency_p99": 1.5, "disk_free_pct": 0.05}
    print(f"  Metrics: {metrics}")
    for rule in rules:
        state = rule.evaluate(metrics)
        print(f"  {rule.name}: {state.value} (severity={rule.severity})")

    # --- Silence Rules ---
    print(f"\n{'=' * 60}")
    print("Silence Rules")
    print("=" * 60)
    silence = SilenceRule(
        id="sil-001",
        matchers={"service": "api", "severity": "warning"},
        starts_at=datetime.now(timezone.utc) - timedelta(hours=1),
        ends_at=datetime.now(timezone.utc) + timedelta(hours=1),
        created_by="alice",
        comment="Planned maintenance",
    )
    for rule in rules:
        silenced = silence.matches(rule.labels)
        print(f"  {rule.name}: {'SILENCED' if silenced else 'active'}")

    # --- Escalation ---
    print(f"\n{'=' * 60}")
    print("Escalation Policy")
    print("=" * 60)
    policy = ESCALATION_POLICIES["critical"]
    for minutes in [0, 10, 20, 35, 65]:
        step = policy.get_current_step(minutes)
        if step:
            print(f"  At {minutes}min: notify {step.targets} via {step.channel}")

    # --- Alert Fatigue Analysis ---
    print(f"\n{'=' * 60}")
    print("Alert Fatigue Analysis")
    print("=" * 60)
    analyzer = FatigueAnalyzer()
    now = time.time()
    for i in range(50):
        name = random.choice(["NoisyDiskAlert", "HighCPU", "PodRestart", "SSLExpiry"])
        analyzer.record(AlertEvent(
            alert_name=name, state=AlertState.FIRING,
            timestamp=now - random.uniform(0, 86400),
            acknowledged=random.random() < 0.25,
        ))
    report = analyzer.analyze(window_hours=24)
    print(f"  Total: {report['total_alerts']} alerts, {report['unique_alerts']} unique")
    print(f"  Ack rate: {report['ack_rate']:.0%}")
    print(f"  Fatigue risk: {report['fatigue_risk']}")
    if report["noisy_alerts"]:
        print(f"  Noisy alerts: {report['noisy_alerts']}")
    print(f"  Recommendation: {report['recommendation']}")
