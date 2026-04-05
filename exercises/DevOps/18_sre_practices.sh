#!/bin/bash
# Exercises for Lesson 18: SRE Practices
# Topic: DevOps
# Solutions to practice problems from the lesson.

# === Exercise 1: Toil Identification and Reduction ===
# Problem: Identify toil in operational tasks and design automation
# to reduce it. Toil is manual, repetitive, automatable work.
exercise_1() {
    echo "=== Exercise 1: Toil Identification and Reduction ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from dataclasses import dataclass

@dataclass
class ToilTask:
    name: str
    frequency: str             # "daily", "weekly", "per-incident"
    time_per_occurrence_min: int
    monthly_occurrences: int
    is_automatable: bool
    automation_effort_hours: int
    automation_description: str

    @property
    def monthly_toil_hours(self) -> float:
        return self.monthly_occurrences * self.time_per_occurrence_min / 60

    @property
    def roi_months(self) -> float:
        """Months until automation pays for itself."""
        if self.monthly_toil_hours == 0:
            return float('inf')
        return self.automation_effort_hours / self.monthly_toil_hours

toil_inventory = [
    ToilTask("SSL certificate renewal", "monthly", 30, 4, True, 8,
             "cert-manager with Let's Encrypt auto-renewal"),
    ToilTask("Disk space cleanup on servers", "weekly", 15, 8, True, 4,
             "Cron job with log rotation + alerting"),
    ToilTask("Manual database backup verification", "daily", 20, 30, True, 16,
             "Automated restore test pipeline with Slack notification"),
    ToilTask("User access provisioning", "per-incident", 45, 12, True, 24,
             "Self-service portal with manager approval workflow"),
    ToilTask("Deployment rollback", "per-incident", 30, 3, True, 8,
             "Automated rollback on error rate threshold (ArgoCD)"),
    ToilTask("Capacity planning spreadsheet update", "monthly", 120, 1, True, 40,
             "Auto-generated report from Prometheus metrics"),
]

total_toil_hours = sum(t.monthly_toil_hours for t in toil_inventory)
total_automation_hours = sum(t.automation_effort_hours for t in toil_inventory)

print(f"{'Task':<40} {'Toil/mo':>8} {'Auto hrs':>9} {'ROI':>8}")
print("-" * 70)
for t in sorted(toil_inventory, key=lambda x: x.roi_months):
    print(f"{t.name:<40} {t.monthly_toil_hours:>6.1f}h {t.automation_effort_hours:>7}h "
          f"{t.roi_months:>6.1f}mo")
print("-" * 70)
print(f"{'Total':<40} {total_toil_hours:>6.1f}h {total_automation_hours:>7}h")
print(f"\nGoogle SRE guideline: Toil should be <50% of an SRE's time")
print(f"Current toil: {total_toil_hours:.0f}h/month")
print(f"Prioritize by ROI: automate shortest-payback tasks first")
SOLUTION
}

# === Exercise 2: Capacity Planning ===
# Problem: Given traffic growth projections, calculate when current
# infrastructure will hit capacity and plan scaling actions.
exercise_2() {
    echo "=== Exercise 2: Capacity Planning ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import math

def capacity_forecast(
    current_rps: float,
    max_rps: float,
    monthly_growth_pct: float,
    headroom_pct: float = 20.0,
) -> dict:
    """Forecast when capacity will be exhausted.

    Args:
        current_rps: Current requests per second
        max_rps: Maximum capacity in requests per second
        monthly_growth_pct: Expected month-over-month growth
        headroom_pct: Safety margin (alert before hitting max)
    """
    alert_threshold = max_rps * (1 - headroom_pct / 100)
    growth_rate = 1 + monthly_growth_pct / 100

    # Months until alert threshold
    if current_rps >= alert_threshold:
        months_to_alert = 0
    else:
        months_to_alert = math.log(alert_threshold / current_rps) / math.log(growth_rate)

    # Months until hard capacity limit
    if current_rps >= max_rps:
        months_to_max = 0
    else:
        months_to_max = math.log(max_rps / current_rps) / math.log(growth_rate)

    # Project monthly RPS for next 12 months
    projections = []
    for month in range(13):
        projected_rps = current_rps * (growth_rate ** month)
        utilization = projected_rps / max_rps * 100
        projections.append({
            "month": month,
            "rps": round(projected_rps),
            "utilization_pct": round(utilization, 1),
        })

    return {
        "current_rps": current_rps,
        "max_rps": max_rps,
        "current_utilization": f"{current_rps/max_rps*100:.1f}%",
        "months_to_alert": round(months_to_alert, 1),
        "months_to_max": round(months_to_max, 1),
        "projections": projections,
    }

# Example: Order API capacity planning
result = capacity_forecast(
    current_rps=1500,
    max_rps=5000,
    monthly_growth_pct=8,
    headroom_pct=20,
)

print(f"Capacity Forecast — Order API")
print(f"  Current:     {result['current_rps']} RPS ({result['current_utilization']})")
print(f"  Max capacity: {result['max_rps']} RPS")
print(f"  Alert in:    {result['months_to_alert']} months (80% utilization)")
print(f"  Saturated:   {result['months_to_max']} months (100% utilization)")
print()
print(f"  {'Month':>5} {'RPS':>6} {'Utilization':>12}")
for p in result["projections"]:
    bar = "#" * int(p["utilization_pct"] / 5)
    warn = " ***" if p["utilization_pct"] > 80 else ""
    print(f"  {p['month']:>5} {p['rps']:>6} {p['utilization_pct']:>10.1f}% {bar}{warn}")

# Scaling actions:
print("\nScaling Plan:")
print("  Month 0-6:  Monitor, no action needed")
print("  Month 6-8:  Initiate horizontal scaling (add replicas, upgrade HPA)")
print("  Month 8-10: Evaluate vertical scaling (larger instances)")
print("  Month 10+:  Architecture review (caching, read replicas, sharding)")
SOLUTION
}

# === Exercise 3: Release Engineering ===
# Problem: Design a release engineering process with automated validation,
# progressive rollout, and release qualification criteria.
exercise_3() {
    echo "=== Exercise 3: Release Engineering ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
release_process = {
    "Phase 1 — Build & Qualify": {
        "triggers": "Tag pushed (v1.2.0) or manual release dispatch",
        "steps": [
            "CI runs full test suite (unit + integration + E2E)",
            "SAST and dependency scanning pass",
            "Docker image built and signed (cosign)",
            "SBOM generated and published",
            "Release notes auto-generated from conventional commits",
        ],
        "gate": "All checks green, 0 critical vulnerabilities",
    },
    "Phase 2 — Staging Validation": {
        "triggers": "Phase 1 gate passed",
        "steps": [
            "Deploy to staging environment via GitOps",
            "Run smoke tests against staging endpoints",
            "Performance test: load test at 2x production traffic",
            "Verify no p99 latency regression (< 10% increase)",
            "Manual QA sign-off for UI-facing changes",
        ],
        "gate": "Smoke tests pass, no performance regression, QA approved",
    },
    "Phase 3 — Canary Release": {
        "triggers": "Phase 2 gate passed",
        "steps": [
            "Deploy canary to production (5% traffic)",
            "Monitor for 30 minutes: error rate, latency, business metrics",
            "Compare canary vs stable using statistical analysis",
            "Increase to 25% if canary is healthy",
            "Monitor for 1 hour at 25%",
        ],
        "gate": "Canary error rate within 0.1% of stable, no SLO violation",
    },
    "Phase 4 — Full Rollout": {
        "triggers": "Phase 3 gate passed",
        "steps": [
            "Promote canary to 100% traffic",
            "Scale down old version",
            "Update release status in release tracker",
            "Notify stakeholders via Slack/email",
        ],
        "gate": "Stable metrics for 4 hours post-rollout",
    },
}

for phase, details in release_process.items():
    print(f"\n{phase}")
    print(f"  Trigger: {details['triggers']}")
    for step in details["steps"]:
        print(f"    - {step}")
    print(f"  Gate: {details['gate']}")

# Release qualification criteria (must-pass):
print("\nRelease Qualification Criteria:")
criteria = [
    "All unit tests pass (100%)",
    "Integration test coverage > 80%",
    "Zero critical/high vulnerability findings",
    "Performance: p99 latency < baseline + 10%",
    "No memory leak in 24-hour soak test",
    "Rollback tested successfully in staging",
]
for c in criteria:
    print(f"  [x] {c}")
SOLUTION
}

# === Exercise 4: SRE Team Practices ===
# Problem: Design an SRE team charter including responsibilities,
# engagement model, and success metrics.
exercise_4() {
    echo "=== Exercise 4: SRE Team Practices ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
sre_charter = {
    "Mission": "Ensure the reliability, performance, and efficiency of production "
               "systems while enabling development teams to ship quickly and safely.",

    "Responsibilities": [
        "Define and maintain SLOs for all production services",
        "Build and maintain monitoring, alerting, and incident response tooling",
        "Participate in on-call rotation for Tier-1 services",
        "Conduct capacity planning and performance optimization",
        "Automate toil (target: <50% of time on operational work)",
        "Review production readiness for new services (PRR)",
        "Lead blameless postmortems and drive action items",
        "Build self-service platforms for development teams",
    ],

    "Engagement Model": {
        "Embedded SRE": "SRE assigned to a product team for 6-12 months",
        "Consulting SRE": "SRE reviews architecture and provides guidance",
        "Platform SRE": "SRE builds shared tooling (CI/CD, observability, IaC)",
        "Handoff Criteria": "Team graduates from SRE support when they can "
                           "maintain SLOs independently",
    },

    "Production Readiness Review (PRR)": [
        "SLOs defined and dashboards created",
        "Alerting rules configured with runbooks",
        "On-call rotation established (minimum 6 engineers)",
        "Capacity planning documented with growth projections",
        "Disaster recovery tested (backup/restore, failover)",
        "Security review completed (SAST, dependency scan, secrets)",
        "Deployment pipeline supports rollback in < 5 minutes",
    ],

    "Success Metrics": {
        "Availability": "All Tier-1 services meet their SLOs",
        "MTTR": "Mean time to recovery < 30 minutes for SEV1",
        "Toil ratio": "Less than 50% of SRE time on toil",
        "Release velocity": "No decrease in deployment frequency due to reliability concerns",
        "Customer impact": "Error budget not exhausted for any service",
    },
}

print(f"SRE Team Charter")
print(f"Mission: {sre_charter['Mission']}")
print()
print("Responsibilities:")
for r in sre_charter["Responsibilities"]:
    print(f"  - {r}")

print("\nEngagement Model:")
for model, desc in sre_charter["Engagement Model"].items():
    print(f"  {model}: {desc}")

print("\nProduction Readiness Review Checklist:")
for item in sre_charter["Production Readiness Review (PRR)"]:
    print(f"  [ ] {item}")

print("\nSuccess Metrics:")
for metric, target in sre_charter["Success Metrics"].items():
    print(f"  {metric}: {target}")
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 18: SRE Practices"
echo "================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
