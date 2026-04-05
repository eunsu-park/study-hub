#!/usr/bin/env python3
"""Example: SRE Practices — Error Budgets, Toil Tracking & On-Call Management

Demonstrates Site Reliability Engineering practices: SLO/error budget
calculation and burn-rate alerting, toil measurement and reduction tracking,
on-call rotation scheduling, and postmortem templates.
Related lesson: 18_SRE_Practices.md
"""

# =============================================================================
# WHY SRE PRACTICES?
# SRE applies software engineering to operations. Instead of "maximize uptime
# at all costs," SRE balances reliability with velocity using error budgets:
# you can deploy freely while the budget lasts, but must slow down and fix
# reliability when it's exhausted.
# =============================================================================

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any


# =============================================================================
# 1. ERROR BUDGET CALCULATOR
# =============================================================================

@dataclass
class SLODefinition:
    """Service Level Objective definition."""
    name: str
    target: float           # e.g., 0.999 (99.9%)
    window_days: int = 30   # Rolling window
    indicator: str = "availability"  # What the SLI measures

    @property
    def error_budget_fraction(self) -> float:
        """Total allowed failure fraction."""
        return 1.0 - self.target

    @property
    def error_budget_minutes(self) -> float:
        """Total allowed downtime in minutes for the window."""
        return self.window_days * 24 * 60 * self.error_budget_fraction


@dataclass
class ErrorBudgetTracker:
    """Tracks error budget consumption over a rolling window."""
    slo: SLODefinition
    bad_minutes: float = 0.0  # Minutes of SLO violation in window

    @property
    def budget_total(self) -> float:
        return self.slo.error_budget_minutes

    @property
    def budget_remaining(self) -> float:
        return max(0.0, self.budget_total - self.bad_minutes)

    @property
    def budget_consumed_pct(self) -> float:
        if self.budget_total == 0:
            return 100.0
        return (self.bad_minutes / self.budget_total) * 100.0

    @property
    def budget_exhausted(self) -> bool:
        return self.bad_minutes >= self.budget_total

    def record_outage(self, duration_minutes: float) -> dict[str, Any]:
        """Record an outage and return budget status."""
        self.bad_minutes += duration_minutes
        return {
            "outage_minutes": duration_minutes,
            "total_bad_minutes": self.bad_minutes,
            "budget_remaining_min": round(self.budget_remaining, 2),
            "budget_consumed_pct": round(self.budget_consumed_pct, 2),
            "exhausted": self.budget_exhausted,
        }


# =============================================================================
# 2. BURN-RATE ALERTING
# =============================================================================

@dataclass
class BurnRateAlert:
    """Multi-window burn-rate alert (Google SRE book approach)."""
    name: str
    long_window_hours: float   # e.g., 1h
    short_window_hours: float  # e.g., 5min
    burn_rate_threshold: float # e.g., 14.4x for 1h window
    severity: str = "page"

    def evaluate(self, long_error_rate: float, short_error_rate: float,
                 budget_fraction: float) -> dict[str, Any]:
        """Evaluate whether the alert should fire."""
        if budget_fraction == 0:
            return {"firing": False, "reason": "No error budget defined"}
        long_burn = long_error_rate / budget_fraction
        short_burn = short_error_rate / budget_fraction
        firing = (long_burn >= self.burn_rate_threshold and
                  short_burn >= self.burn_rate_threshold)
        return {
            "alert": self.name,
            "firing": firing,
            "long_burn_rate": round(long_burn, 2),
            "short_burn_rate": round(short_burn, 2),
            "threshold": self.burn_rate_threshold,
            "severity": self.severity,
        }


# Standard multi-window burn-rate alerts for 30-day SLO
STANDARD_ALERTS = [
    BurnRateAlert("rapid-burn", 1.0, 5/60, 14.4, "page"),
    BurnRateAlert("sustained-burn", 6.0, 0.5, 6.0, "page"),
    BurnRateAlert("slow-burn", 24.0, 2.0, 3.0, "ticket"),
    BurnRateAlert("gradual-burn", 72.0, 6.0, 1.0, "ticket"),
]


# =============================================================================
# 3. TOIL TRACKING
# =============================================================================

class ToilCategory(Enum):
    MANUAL = "manual"
    REPETITIVE = "repetitive"
    AUTOMATABLE = "automatable"
    REACTIVE = "reactive"
    NO_VALUE = "no-lasting-value"


@dataclass
class ToilEntry:
    """A recorded toil activity."""
    description: str
    category: ToilCategory
    duration_minutes: float
    date: str
    automated: bool = False


@dataclass
class ToilTracker:
    """Track and analyze operational toil."""
    entries: list[ToilEntry] = field(default_factory=list)

    def add(self, entry: ToilEntry) -> None:
        self.entries.append(entry)

    @property
    def total_hours(self) -> float:
        return sum(e.duration_minutes for e in self.entries) / 60.0

    @property
    def automatable_hours(self) -> float:
        return sum(e.duration_minutes for e in self.entries
                   if e.category == ToilCategory.AUTOMATABLE) / 60.0

    def by_category(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for e in self.entries:
            result.setdefault(e.category.value, 0.0)
            result[e.category.value] += e.duration_minutes / 60.0
        return {k: round(v, 1) for k, v in result.items()}

    def toil_ratio(self, total_eng_hours: float) -> float:
        """Toil as percentage of total engineering time (target: <50%)."""
        if total_eng_hours == 0:
            return 0.0
        return (self.total_hours / total_eng_hours) * 100.0


# =============================================================================
# 4. ON-CALL ROTATION
# =============================================================================

@dataclass
class OnCallRotation:
    """Weekly on-call rotation scheduler."""
    team_members: list[str]
    rotation_start: datetime
    shift_days: int = 7
    schedule: list[dict[str, str]] = field(default_factory=list)

    def generate_schedule(self, weeks: int = 8) -> list[dict[str, str]]:
        """Generate a rotation schedule for N weeks."""
        self.schedule.clear()
        for i in range(weeks):
            primary_idx = i % len(self.team_members)
            secondary_idx = (i + 1) % len(self.team_members)
            start = self.rotation_start + timedelta(weeks=i)
            end = start + timedelta(days=self.shift_days)
            self.schedule.append({
                "week": i + 1,
                "start": start.strftime("%Y-%m-%d"),
                "end": end.strftime("%Y-%m-%d"),
                "primary": self.team_members[primary_idx],
                "secondary": self.team_members[secondary_idx],
            })
        return self.schedule


# =============================================================================
# 5. POSTMORTEM TEMPLATE
# =============================================================================

def generate_postmortem(incident: dict[str, Any]) -> str:
    """Generate a blameless postmortem document."""
    return f"""# Postmortem: {incident['title']}

## Summary
- **Severity**: {incident.get('severity', 'P2')}
- **Duration**: {incident.get('duration_min', 0)} minutes
- **Impact**: {incident.get('impact', 'Unknown')}
- **Date**: {incident.get('date', 'Unknown')}

## Timeline
{chr(10).join(f'- {t["time"]}: {t["event"]}' for t in incident.get('timeline', []))}

## Root Cause
{incident.get('root_cause', 'Under investigation')}

## Action Items
{chr(10).join(f'- [ ] {a}' for a in incident.get('action_items', []))}

## Lessons Learned
- What went well: {incident.get('went_well', '')}
- What went poorly: {incident.get('went_poorly', '')}

## Error Budget Impact
- Budget consumed by this incident: {incident.get('budget_consumed_min', 0)} minutes
"""


# =============================================================================
# 6. DEMO
# =============================================================================

if __name__ == "__main__":
    # --- Error Budget ---
    print("=" * 60)
    print("Error Budget Tracking (99.9% SLO, 30-day window)")
    print("=" * 60)
    slo = SLODefinition(name="api-availability", target=0.999)
    tracker = ErrorBudgetTracker(slo=slo)
    print(f"  Total budget: {tracker.budget_total:.1f} minutes")
    for outage in [5.0, 10.0, 15.0, 20.0]:
        status = tracker.record_outage(outage)
        print(f"  +{outage}min outage -> consumed={status['budget_consumed_pct']:.1f}%, "
              f"remaining={status['budget_remaining_min']:.1f}min, "
              f"exhausted={status['exhausted']}")

    # --- Burn Rate Alerts ---
    print(f"\n{'=' * 60}")
    print("Burn Rate Alerts")
    print("=" * 60)
    for alert in STANDARD_ALERTS:
        result = alert.evaluate(
            long_error_rate=0.005, short_error_rate=0.008,
            budget_fraction=slo.error_budget_fraction,
        )
        status = "FIRING" if result["firing"] else "ok"
        print(f"  [{status:7s}] {alert.name}: "
              f"burn={result['long_burn_rate']}x/{result['short_burn_rate']}x "
              f"(threshold={alert.burn_rate_threshold}x) [{alert.severity}]")

    # --- Toil Tracking ---
    print(f"\n{'=' * 60}")
    print("Toil Tracking")
    print("=" * 60)
    toil = ToilTracker()
    toil.add(ToilEntry("Restart crashed pods", ToilCategory.REACTIVE, 30, "2024-01-15"))
    toil.add(ToilEntry("Manual cert renewal", ToilCategory.AUTOMATABLE, 45, "2024-01-16"))
    toil.add(ToilEntry("Capacity spreadsheet update", ToilCategory.MANUAL, 60, "2024-01-17"))
    toil.add(ToilEntry("Deploy hotfix", ToilCategory.REPETITIVE, 20, "2024-01-18"))
    print(f"  Total toil: {toil.total_hours:.1f}h")
    print(f"  Automatable: {toil.automatable_hours:.1f}h")
    print(f"  By category: {toil.by_category()}")
    print(f"  Toil ratio (of 40h eng week): {toil.toil_ratio(40):.1f}%")

    # --- On-Call Rotation ---
    print(f"\n{'=' * 60}")
    print("On-Call Rotation")
    print("=" * 60)
    rotation = OnCallRotation(
        team_members=["Alice", "Bob", "Carol", "Dave"],
        rotation_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    for week in rotation.generate_schedule(6):
        print(f"  Week {week['week']}: {week['start']} — "
              f"primary={week['primary']}, secondary={week['secondary']}")
