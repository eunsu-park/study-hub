#!/usr/bin/env python3
"""Example: SLI/SLO Calculator — Error Budget & Burn Rate Alerts

Computes Service Level Indicators (SLIs), evaluates Service Level
Objectives (SLOs), calculates error budgets, and implements multi-window
burn rate alerting per Google SRE best practices.
Related lesson: 17_SLIs_SLOs_and_Error_Budgets.md
"""

# =============================================================================
# KEY CONCEPTS
#   SLI  — a quantitative measure of service behavior (e.g., latency p99)
#   SLO  — target value for an SLI (e.g., 99.9% availability over 30 days)
#   Error Budget — (1 - SLO) * total requests = allowed failures
#   Burn Rate  — how fast the error budget is being consumed
# =============================================================================

import time
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any


# =============================================================================
# 1. SLI TYPES
# =============================================================================

class SLIType(Enum):
    AVAILABILITY = "availability"    # good_requests / total_requests
    LATENCY = "latency"              # requests_below_threshold / total_requests
    THROUGHPUT = "throughput"         # requests_served / time_window
    ERROR_RATE = "error_rate"        # 1 - (errors / total)
    FRESHNESS = "freshness"          # data_updated_within_threshold / total_checks


@dataclass
class SLIValue:
    """A measured SLI value for a time window."""
    sli_type: SLIType
    value: float           # 0.0 to 1.0 (ratio) or absolute for throughput
    good_events: int
    total_events: int
    window_start: datetime
    window_end: datetime

    @property
    def bad_events(self) -> int:
        return self.total_events - self.good_events


# =============================================================================
# 2. SLO DEFINITION
# =============================================================================

@dataclass
class SLO:
    """Service Level Objective definition."""
    name: str
    sli_type: SLIType
    target: float                    # e.g., 0.999 for 99.9%
    window_days: int = 30            # Rolling window in days
    description: str = ""

    @property
    def error_budget_ratio(self) -> float:
        """Fraction of events allowed to be bad."""
        return 1.0 - self.target

    def error_budget_absolute(self, total_events: int) -> float:
        """Absolute number of bad events allowed."""
        return total_events * self.error_budget_ratio

    def remaining_budget(self, total_events: int, bad_events: int) -> float:
        """Remaining error budget as a fraction of the total budget."""
        budget = self.error_budget_absolute(total_events)
        if budget == 0:
            return 0.0
        return max(0.0, (budget - bad_events) / budget)

    def is_met(self, sli_value: float) -> bool:
        """Check if the SLO target is currently met."""
        return sli_value >= self.target


# =============================================================================
# 3. ERROR BUDGET CALCULATOR
# =============================================================================

@dataclass
class ErrorBudgetReport:
    """Computed error budget status."""
    slo: SLO
    sli_value: float
    total_events: int
    bad_events: int
    budget_total: float
    budget_consumed: float
    budget_remaining: float
    budget_remaining_pct: float
    days_in_window: int
    burn_rate_1h: float = 0.0
    burn_rate_6h: float = 0.0

    @property
    def budget_exhaustion_days(self) -> float | None:
        """Estimated days until error budget is exhausted at current burn rate."""
        if self.burn_rate_1h <= 0:
            return None  # Not burning budget
        remaining_budget = self.budget_remaining
        daily_burn = self.burn_rate_1h * 24
        if daily_burn == 0:
            return None
        return remaining_budget / daily_burn

    def to_dict(self) -> dict[str, Any]:
        return {
            "slo_name": self.slo.name,
            "slo_target": f"{self.slo.target:.3%}",
            "sli_value": f"{self.sli_value:.4%}",
            "slo_met": self.slo.is_met(self.sli_value),
            "total_events": self.total_events,
            "bad_events": self.bad_events,
            "budget_total": round(self.budget_total, 1),
            "budget_consumed": round(self.budget_consumed, 1),
            "budget_remaining": round(self.budget_remaining, 1),
            "budget_remaining_pct": f"{self.budget_remaining_pct:.1%}",
            "burn_rate_1h": round(self.burn_rate_1h, 2),
            "burn_rate_6h": round(self.burn_rate_6h, 2),
            "exhaustion_days": (
                round(self.budget_exhaustion_days, 1)
                if self.budget_exhaustion_days is not None else "N/A"
            ),
        }


def compute_error_budget(
    slo: SLO,
    total_events: int,
    bad_events: int,
    bad_events_1h: int = 0,
    total_events_1h: int = 0,
    bad_events_6h: int = 0,
    total_events_6h: int = 0,
) -> ErrorBudgetReport:
    """Compute error budget status for a given SLO and event counts."""
    sli_value = 1.0 - (bad_events / max(total_events, 1))
    budget_total = slo.error_budget_absolute(total_events)
    budget_consumed = float(bad_events)
    budget_remaining = max(0.0, budget_total - budget_consumed)
    budget_remaining_pct = slo.remaining_budget(total_events, bad_events)

    # Burn rate: how fast are we consuming the budget relative to the window?
    # burn_rate = (bad_events_in_window / total_events_in_window) / error_budget_ratio
    # A burn rate of 1.0 means we'll exactly exhaust the budget by end of window
    # A burn rate of 14.4 means we'll exhaust it in 1/14.4 of the window (2 days for 30d)
    error_budget_ratio = slo.error_budget_ratio

    burn_rate_1h = 0.0
    if total_events_1h > 0 and error_budget_ratio > 0:
        observed_error_rate_1h = bad_events_1h / total_events_1h
        burn_rate_1h = observed_error_rate_1h / error_budget_ratio

    burn_rate_6h = 0.0
    if total_events_6h > 0 and error_budget_ratio > 0:
        observed_error_rate_6h = bad_events_6h / total_events_6h
        burn_rate_6h = observed_error_rate_6h / error_budget_ratio

    return ErrorBudgetReport(
        slo=slo,
        sli_value=sli_value,
        total_events=total_events,
        bad_events=bad_events,
        budget_total=budget_total,
        budget_consumed=budget_consumed,
        budget_remaining=budget_remaining,
        budget_remaining_pct=budget_remaining_pct,
        days_in_window=slo.window_days,
        burn_rate_1h=burn_rate_1h,
        burn_rate_6h=burn_rate_6h,
    )


# =============================================================================
# 4. MULTI-WINDOW BURN RATE ALERTING
# =============================================================================
# Google SRE recommends multi-window, multi-burn-rate alerts:
#   - Short window (e.g., 1h) detects fast burns (outages)
#   - Long window (e.g., 6h) detects slow burns (degradation)
#   - Both must exceed their threshold to fire the alert

class AlertSeverity(Enum):
    CRITICAL = "CRITICAL"   # Page: immediate human response required
    WARNING = "WARNING"     # Ticket: fix within business hours
    INFO = "INFO"           # Log: monitor but no action needed


@dataclass
class BurnRateAlert:
    """Multi-window burn rate alert definition."""
    name: str
    severity: AlertSeverity
    short_window: str            # e.g., "1h"
    long_window: str             # e.g., "6h"
    short_burn_rate_threshold: float
    long_burn_rate_threshold: float

    def evaluate(self, short_burn_rate: float, long_burn_rate: float) -> bool:
        """Returns True if the alert should fire."""
        return (
            short_burn_rate >= self.short_burn_rate_threshold
            and long_burn_rate >= self.long_burn_rate_threshold
        )


# Standard multi-window alert thresholds for a 30-day SLO window:
STANDARD_ALERTS = [
    # 2% of budget consumed in 1h -> 100% in ~2 days
    BurnRateAlert(
        name="fast-burn-critical",
        severity=AlertSeverity.CRITICAL,
        short_window="5m",
        long_window="1h",
        short_burn_rate_threshold=14.4,
        long_burn_rate_threshold=14.4,
    ),
    # 5% of budget consumed in 6h -> 100% in ~5 days
    BurnRateAlert(
        name="slow-burn-critical",
        severity=AlertSeverity.CRITICAL,
        short_window="30m",
        long_window="6h",
        short_burn_rate_threshold=6.0,
        long_burn_rate_threshold=6.0,
    ),
    # 10% of budget consumed in 3 days
    BurnRateAlert(
        name="slow-burn-warning",
        severity=AlertSeverity.WARNING,
        short_window="2h",
        long_window="1d",
        short_burn_rate_threshold=3.0,
        long_burn_rate_threshold=3.0,
    ),
    # Very slow burn — 10% over the entire window
    BurnRateAlert(
        name="slow-burn-info",
        severity=AlertSeverity.INFO,
        short_window="6h",
        long_window="3d",
        short_burn_rate_threshold=1.0,
        long_burn_rate_threshold=1.0,
    ),
]


def evaluate_alerts(
    alerts: list[BurnRateAlert],
    burn_rate_1h: float,
    burn_rate_6h: float,
) -> list[BurnRateAlert]:
    """Evaluate which alerts should fire given current burn rates."""
    fired: list[BurnRateAlert] = []
    for alert in alerts:
        if alert.evaluate(burn_rate_1h, burn_rate_6h):
            fired.append(alert)
    return fired


# =============================================================================
# 5. DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SLI / SLO / Error Budget Calculator")
    print("=" * 70)

    # Define SLOs
    slos = [
        SLO("api-availability", SLIType.AVAILABILITY, 0.999, 30,
            "99.9% of requests return non-5xx"),
        SLO("api-latency-p99", SLIType.LATENCY, 0.99, 30,
            "99% of requests < 300ms"),
        SLO("checkout-availability", SLIType.AVAILABILITY, 0.9999, 30,
            "99.99% checkout success rate"),
    ]

    # Simulate data for each SLO
    scenarios = [
        # (total_events, bad_events, bad_1h, total_1h, bad_6h, total_6h)
        (10_000_000, 5_000, 10, 50_000, 80, 300_000),     # Healthy
        (10_000_000, 50_000, 200, 50_000, 500, 300_000),   # Slow burn
        (1_000_000, 500, 50, 5_000, 100, 30_000),          # Fast burn on checkout
    ]

    for slo, (total, bad, bad_1h, total_1h, bad_6h, total_6h) in zip(slos, scenarios):
        report = compute_error_budget(
            slo, total, bad, bad_1h, total_1h, bad_6h, total_6h,
        )
        d = report.to_dict()

        print(f"\n{'─' * 50}")
        print(f"SLO: {d['slo_name']}")
        print(f"  Target:            {d['slo_target']}")
        print(f"  Current SLI:       {d['sli_value']}")
        print(f"  SLO Met:           {d['slo_met']}")
        print(f"  Total events:      {d['total_events']:,}")
        print(f"  Bad events:        {d['bad_events']:,}")
        print(f"  Budget (total):    {d['budget_total']:,}")
        print(f"  Budget (consumed): {d['budget_consumed']:,}")
        print(f"  Budget remaining:  {d['budget_remaining_pct']}")
        print(f"  Burn rate (1h):    {d['burn_rate_1h']}x")
        print(f"  Burn rate (6h):    {d['burn_rate_6h']}x")
        print(f"  Exhaustion ETA:    {d['exhaustion_days']} days")

        # Evaluate alerts
        fired = evaluate_alerts(
            STANDARD_ALERTS, report.burn_rate_1h, report.burn_rate_6h,
        )
        if fired:
            print(f"  Alerts FIRED:")
            for alert in fired:
                print(f"    [{alert.severity.value}] {alert.name} "
                      f"(short>={alert.short_burn_rate_threshold}x, "
                      f"long>={alert.long_burn_rate_threshold}x)")
        else:
            print(f"  No alerts firing")

    # --- Error budget policy summary ---
    print(f"\n{'=' * 70}")
    print("Error Budget Policy Recommendations")
    print("=" * 70)
    print("""
  Budget > 50% remaining:
    -> Continue feature development at normal velocity
    -> Run chaos experiments and load tests

  Budget 20-50% remaining:
    -> Slow down feature releases
    -> Prioritize reliability improvements
    -> Review recent changes for regressions

  Budget < 20% remaining:
    -> Freeze non-critical feature releases
    -> All engineering effort on reliability
    -> Conduct incident reviews for budget burns

  Budget exhausted (0%):
    -> Hard freeze on all changes except reliability fixes
    -> Escalate to leadership for staffing decisions
    -> Daily SLO review meetings until budget recovers
""")
