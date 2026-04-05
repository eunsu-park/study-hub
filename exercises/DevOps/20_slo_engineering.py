#!/usr/bin/env python3
"""Exercises for Lesson 20: SLO Engineering
Topic: DevOps
"""

from dataclasses import dataclass


# === Exercise 1: SLI Selection ===

def exercise_1():
    """SLI selection for different service types."""
    print("=== Exercise 1: SLI Selection ===\n")
    services = {
        "Image upload service": [
            ("Availability", "Non-error upload responses / total uploads"),
            ("Latency", "Uploads completing within 10s / total uploads"),
            ("Freshness", "Resized variants available within 60s / total uploads"),
        ],
        "Real-time chat": [
            ("Availability", "Successful message sends / total sends"),
            ("Latency", "Messages delivered within 500ms / total messages"),
            ("Freshness", "History requests with < 5s stale data / total requests"),
        ],
        "Nightly batch (financial reports)": [
            ("Freshness", "Reports available by 6 AM / total scheduled"),
            ("Correctness", "Report rows matching source / total rows"),
            ("Throughput", "Completed reports / scheduled reports"),
        ],
    }
    for svc, slis in services.items():
        print(f"  {svc}:")
        for sli_type, spec in slis:
            print(f"    {sli_type}: {spec}")
        print()


# === Exercise 2: Error Budget Calculation ===

@dataclass
class ErrorBudgetStatus:
    slo_target: float
    total_requests: int
    failed_requests: int

    @property
    def current_sli(self) -> float:
        return (self.total_requests - self.failed_requests) / self.total_requests

    @property
    def error_budget_total(self) -> int:
        return int(self.total_requests * (1 - self.slo_target))

    @property
    def budget_consumed_pct(self) -> float:
        return self.failed_requests / self.error_budget_total * 100

    @property
    def budget_remaining_pct(self) -> float:
        return max(0, 100 - self.budget_consumed_pct)


def exercise_2():
    """Error budget calculation."""
    print("=== Exercise 2: Error Budget Calculation ===\n")
    status = ErrorBudgetStatus(slo_target=0.999, total_requests=50_000_000, failed_requests=42_000)
    print(f"SLO target:          {status.slo_target:.3%}")
    print(f"Current SLI:         {status.current_sli:.4%}")
    print(f"Error budget total:  {status.error_budget_total:,} requests")
    print(f"Budget consumed:     {status.budget_consumed_pct:.0f}%")
    print(f"Budget remaining:    {status.budget_remaining_pct:.0f}%")
    print(f"\nWith 16% remaining (5-25% range):")
    print("  → Feature freeze for non-critical changes")
    print("  → Reliability work takes priority")
    print("  → Conduct postmortem for budget-consuming incident")


# === Exercise 3: Burn Rate Alert Design ===

def exercise_3():
    """Burn rate alert design for a latency SLO."""
    print("=== Exercise 3: Burn Rate Alert Design ===\n")
    slo = 0.995  # 99.5% of requests < 500ms
    budget_rate = 1 - slo  # 0.005

    alerts = [
        ("Critical (page)", 14.4, "1h", "5m", 14.4 * budget_rate),
        ("Warning (ticket)", 6.0, "6h", "30m", 6.0 * budget_rate),
        ("Info (ticket)", 3.0, "1d", "2h", 3.0 * budget_rate),
    ]

    print(f"SLO: {slo:.1%} of requests < 500ms")
    print(f"Budget rate: {budget_rate}")
    print(f"\nBurn rate alert thresholds:")
    for name, burn, long_win, short_win, threshold in alerts:
        exhaustion = 30 / burn
        print(f"  {name}: {burn}x burn rate")
        print(f"    Long window: {long_win}, Short window: {short_win}")
        print(f"    Threshold: {threshold:.4f}")
        print(f"    Time to exhaust 30d budget: {exhaustion:.1f} days")


if __name__ == "__main__":
    exercise_1()
    print("\n" + "=" * 70 + "\n")
    exercise_2()
    print("\n" + "=" * 70 + "\n")
    exercise_3()
