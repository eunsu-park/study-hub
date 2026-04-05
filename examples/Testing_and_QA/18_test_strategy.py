#!/usr/bin/env python3
"""Example: Test Strategy

Demonstrates risk-based testing, test planning, coverage goal setting,
test prioritization, and metrics-driven test strategy decisions.
Related lesson: 18_Test_Strategy.md
"""

# =============================================================================
# WHY TEST STRATEGY?
#
# You can't test everything — every project has limited time and budget.
# A test strategy answers:
#   1. WHAT to test (risk-based prioritization)
#   2. HOW MUCH to test (coverage goals per component)
#   3. WHEN to test (shift-left, CI gates, release checks)
#   4. WHERE to test (unit vs integration vs E2E balance)
#
# The goal is maximizing defect detection per hour of testing effort.
# =============================================================================

import pytest
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# =============================================================================
# RISK-BASED TEST PRIORITIZATION
# =============================================================================

class Likelihood(Enum):
    """How likely is a defect in this component?"""
    LOW = 1       # Stable, well-understood code
    MEDIUM = 2    # Moderate complexity or recent changes
    HIGH = 3      # New code, complex logic, or history of bugs


class Impact(Enum):
    """What happens if this component has a defect?"""
    LOW = 1       # Cosmetic issue, workaround available
    MEDIUM = 2    # Feature degraded, some users affected
    HIGH = 3      # Data loss, security breach, or total outage


@dataclass
class Component:
    """A system component to be tested."""
    name: str
    likelihood: Likelihood
    impact: Impact
    lines_of_code: int = 0
    recent_bug_count: int = 0
    last_modified_days_ago: int = 0

    @property
    def risk_score(self) -> int:
        """Risk = Likelihood x Impact. Higher score = more testing needed."""
        return self.likelihood.value * self.impact.value

    @property
    def risk_level(self) -> str:
        score = self.risk_score
        if score >= 6:
            return "CRITICAL"
        elif score >= 3:
            return "MODERATE"
        return "LOW"


class RiskBasedPrioritizer:
    """Prioritize testing effort based on risk assessment."""

    @staticmethod
    def prioritize(components: list[Component]) -> list[Component]:
        """Sort components by risk score (highest first).
        Ties broken by recent bug count (buggier code first)."""
        return sorted(
            components,
            key=lambda c: (c.risk_score, c.recent_bug_count),
            reverse=True,
        )

    @staticmethod
    def recommend_coverage(component: Component) -> dict:
        """Recommend coverage targets based on risk level."""
        recommendations = {
            "CRITICAL": {
                "line_coverage": 90,
                "branch_coverage": 85,
                "mutation_score": 70,
                "test_types": ["unit", "integration", "e2e", "security"],
                "review": "mandatory code review + pair testing",
            },
            "MODERATE": {
                "line_coverage": 75,
                "branch_coverage": 65,
                "mutation_score": 50,
                "test_types": ["unit", "integration"],
                "review": "standard code review",
            },
            "LOW": {
                "line_coverage": 50,
                "branch_coverage": 40,
                "mutation_score": 0,
                "test_types": ["unit"],
                "review": "optional",
            },
        }
        return recommendations[component.risk_level]


# =============================================================================
# TEST PLAN MODEL
# =============================================================================

@dataclass
class TestCase:
    """A planned test case with priority and status."""
    id: str
    description: str
    component: str
    priority: int  # 1 = highest
    automated: bool = False
    status: str = "planned"  # planned, passed, failed, skipped

    def execute(self, result: bool) -> None:
        self.status = "passed" if result else "failed"

    def skip(self, reason: str = "") -> None:
        self.status = "skipped"


@dataclass
class TestPlan:
    """A collection of test cases with execution tracking."""
    name: str
    cases: list[TestCase] = field(default_factory=list)

    def add(self, case: TestCase) -> None:
        self.cases.append(case)

    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.cases if c.status == "passed")

    @property
    def failed(self) -> int:
        return sum(1 for c in self.cases if c.status == "failed")

    @property
    def skipped(self) -> int:
        return sum(1 for c in self.cases if c.status == "skipped")

    @property
    def completion_rate(self) -> float:
        """Percentage of tests that have been executed (passed + failed)."""
        if self.total == 0:
            return 0.0
        executed = self.passed + self.failed
        return round(executed / self.total * 100, 1)

    @property
    def pass_rate(self) -> float:
        """Percentage of executed tests that passed."""
        executed = self.passed + self.failed
        if executed == 0:
            return 0.0
        return round(self.passed / executed * 100, 1)

    def get_by_priority(self, max_priority: int) -> list[TestCase]:
        """Get tests with priority <= max_priority (lower number = higher priority)."""
        return sorted(
            [c for c in self.cases if c.priority <= max_priority],
            key=lambda c: c.priority,
        )


# =============================================================================
# COVERAGE GOAL CALCULATOR
# =============================================================================

@dataclass
class CoverageMetrics:
    """Track coverage metrics for a component."""
    component: str
    line_coverage: float
    branch_coverage: float
    mutation_score: float = 0.0

    def meets_target(self, target: dict) -> dict[str, bool]:
        """Check which coverage targets are met."""
        return {
            "line_coverage": self.line_coverage >= target.get("line_coverage", 0),
            "branch_coverage": self.branch_coverage >= target.get("branch_coverage", 0),
            "mutation_score": self.mutation_score >= target.get("mutation_score", 0),
        }

    def all_targets_met(self, target: dict) -> bool:
        return all(self.meets_target(target).values())


# =============================================================================
# TESTS — RISK PRIORITIZATION
# =============================================================================

class TestRiskPrioritization:
    """Verify risk-based test prioritization logic."""

    @pytest.fixture
    def components(self):
        return [
            Component("Payment Processing", Likelihood.HIGH, Impact.HIGH, recent_bug_count=5),
            Component("User Profile", Likelihood.LOW, Impact.LOW, recent_bug_count=0),
            Component("Search", Likelihood.MEDIUM, Impact.MEDIUM, recent_bug_count=2),
            Component("Auth", Likelihood.MEDIUM, Impact.HIGH, recent_bug_count=3),
            Component("Logging", Likelihood.LOW, Impact.LOW, recent_bug_count=0),
        ]

    def test_risk_score_calculation(self):
        comp = Component("Test", Likelihood.HIGH, Impact.HIGH)
        assert comp.risk_score == 9
        assert comp.risk_level == "CRITICAL"

    def test_risk_levels(self):
        assert Component("A", Likelihood.HIGH, Impact.HIGH).risk_level == "CRITICAL"
        assert Component("B", Likelihood.MEDIUM, Impact.MEDIUM).risk_level == "MODERATE"
        assert Component("C", Likelihood.LOW, Impact.LOW).risk_level == "LOW"

    def test_prioritization_order(self, components):
        """Highest risk components should be tested first."""
        prioritized = RiskBasedPrioritizer.prioritize(components)

        assert prioritized[0].name == "Payment Processing"  # risk=9
        assert prioritized[-1].name in ("User Profile", "Logging")  # risk=1

    def test_coverage_recommendations(self):
        critical = Component("Pay", Likelihood.HIGH, Impact.HIGH)
        rec = RiskBasedPrioritizer.recommend_coverage(critical)

        assert rec["line_coverage"] >= 90
        assert "security" in rec["test_types"]

    def test_low_risk_minimal_coverage(self):
        low = Component("Log", Likelihood.LOW, Impact.LOW)
        rec = RiskBasedPrioritizer.recommend_coverage(low)

        assert rec["line_coverage"] <= 50
        assert "e2e" not in rec["test_types"]


# =============================================================================
# TESTS — TEST PLAN
# =============================================================================

class TestTestPlan:
    """Verify test plan tracking and metrics."""

    @pytest.fixture
    def plan(self):
        p = TestPlan("Sprint 42 Release")
        p.add(TestCase("TC-001", "Login works", "Auth", priority=1, automated=True))
        p.add(TestCase("TC-002", "Payment processes", "Payment", priority=1))
        p.add(TestCase("TC-003", "Profile updates", "Profile", priority=2))
        p.add(TestCase("TC-004", "Search returns results", "Search", priority=2))
        p.add(TestCase("TC-005", "Export to CSV", "Reports", priority=3))
        return p

    def test_initial_state(self, plan):
        assert plan.total == 5
        assert plan.passed == 0
        assert plan.completion_rate == 0.0

    def test_execution_tracking(self, plan):
        plan.cases[0].execute(result=True)
        plan.cases[1].execute(result=False)
        plan.cases[2].skip("deferred")

        assert plan.passed == 1
        assert plan.failed == 1
        assert plan.skipped == 1
        assert plan.completion_rate == 40.0  # 2 of 5 executed
        assert plan.pass_rate == 50.0  # 1 of 2 passed

    def test_priority_filtering(self, plan):
        """When time is limited, run only highest priority tests."""
        critical = plan.get_by_priority(max_priority=1)
        assert len(critical) == 2
        assert all(c.priority == 1 for c in critical)

    def test_empty_plan_metrics(self):
        empty = TestPlan("Empty")
        assert empty.completion_rate == 0.0
        assert empty.pass_rate == 0.0


# =============================================================================
# TESTS — COVERAGE GOALS
# =============================================================================

class TestCoverageGoals:
    """Verify coverage target checking."""

    def test_meets_all_targets(self):
        metrics = CoverageMetrics("Payment", line_coverage=92, branch_coverage=88, mutation_score=75)
        target = {"line_coverage": 90, "branch_coverage": 85, "mutation_score": 70}
        assert metrics.all_targets_met(target)

    def test_misses_target(self):
        metrics = CoverageMetrics("Search", line_coverage=70, branch_coverage=60, mutation_score=40)
        target = {"line_coverage": 75, "branch_coverage": 65, "mutation_score": 50}

        results = metrics.meets_target(target)
        assert results["line_coverage"] is False
        assert results["branch_coverage"] is False
        assert not metrics.all_targets_met(target)

    def test_partial_targets(self):
        """Some targets met, others not — detailed breakdown."""
        metrics = CoverageMetrics("Auth", line_coverage=80, branch_coverage=50, mutation_score=60)
        target = {"line_coverage": 75, "branch_coverage": 65, "mutation_score": 50}

        results = metrics.meets_target(target)
        assert results["line_coverage"] is True
        assert results["branch_coverage"] is False
        assert results["mutation_score"] is True


# =============================================================================
# TEST STRATEGY CHECKLIST (REFERENCE)
# =============================================================================

TEST_STRATEGY_TEMPLATE = """
# Test Strategy Checklist

## 1. Risk Assessment
- [ ] Identify all system components
- [ ] Rate likelihood and impact for each
- [ ] Prioritize by risk score (likelihood x impact)
- [ ] Allocate testing effort proportional to risk

## 2. Coverage Goals (per risk level)
- CRITICAL: 90% line, 85% branch, 70% mutation score
- MODERATE: 75% line, 65% branch, 50% mutation score
- LOW: 50% line, smoke tests only

## 3. Test Types Balance (Pyramid)
- Unit tests: 70% of test count (fast, isolated)
- Integration tests: 20% (component interactions)
- E2E tests: 10% (critical user journeys only)

## 4. CI/CD Gates
- Pre-commit: linting + unit tests (< 30 seconds)
- PR check: full unit + integration (< 5 minutes)
- Pre-deploy: E2E smoke tests (< 15 minutes)
- Post-deploy: health checks + synthetic monitoring

## 5. Maintenance
- Review flaky tests weekly — fix or remove
- Update risk assessment quarterly
- Track defect escape rate (bugs found in production)
- Adjust coverage targets based on defect patterns
"""

# =============================================================================
# RUNNING THIS FILE
# =============================================================================
# pytest 18_test_strategy.py -v

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
