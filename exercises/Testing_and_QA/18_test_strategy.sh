#!/bin/bash
# Exercises for Lesson 18: Test Strategy and Planning
# Topic: Testing_and_QA
# Solutions to practice problems from the lesson.

# === Exercise 1: Risk Assessment ===
# Problem: For a hypothetical e-commerce platform, create a risk
# assessment matrix for its top 10 components. Assign testing levels
# and coverage goals to each.
exercise_1() {
    echo "=== Exercise 1: Risk Assessment ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from dataclasses import dataclass

@dataclass
class ComponentRisk:
    name: str
    likelihood: str       # low, medium, high
    impact: str           # low, medium, high, critical
    testing_level: str    # minimal, basic, standard, thorough, critical
    coverage_goal: int    # percentage

    @property
    def risk_score(self) -> int:
        L = {"low": 1, "medium": 2, "high": 3}
        I = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        return L[self.likelihood] * I[self.impact]

def build_risk_matrix() -> list[ComponentRisk]:
    """Risk assessment for an e-commerce platform."""
    components = [
        ComponentRisk("Payment Processing",    "high",   "critical", "critical",  95),
        ComponentRisk("Authentication",         "medium", "critical", "thorough",  90),
        ComponentRisk("Order Management",       "high",   "high",     "thorough",  85),
        ComponentRisk("Inventory Tracking",     "medium", "high",     "thorough",  80),
        ComponentRisk("Search & Filtering",     "medium", "medium",   "standard",  75),
        ComponentRisk("Shopping Cart",          "medium", "medium",   "standard",  75),
        ComponentRisk("Product Catalog",        "low",    "medium",   "standard",  70),
        ComponentRisk("Email Notifications",    "low",    "low",      "basic",     60),
        ComponentRisk("Admin Dashboard",        "low",    "low",      "basic",     55),
        ComponentRisk("Static Content Pages",   "low",    "low",      "minimal",   40),
    ]
    return sorted(components, key=lambda c: c.risk_score, reverse=True)

# --- Verification ---

def test_risk_matrix_ordering():
    """Highest risk components should be listed first."""
    matrix = build_risk_matrix()
    scores = [c.risk_score for c in matrix]
    assert scores == sorted(scores, reverse=True)

def test_critical_components_have_high_coverage():
    matrix = build_risk_matrix()
    critical = [c for c in matrix if c.testing_level == "critical"]
    for c in critical:
        assert c.coverage_goal >= 90, f"{c.name} coverage too low"

def test_all_components_have_coverage_goal():
    matrix = build_risk_matrix()
    for c in matrix:
        assert 0 < c.coverage_goal <= 100

# Print the matrix
matrix = build_risk_matrix()
print(f"{'Component':<25} {'Likelihood':<10} {'Impact':<10} {'Score':<6} {'Level':<10} {'Coverage'}")
print("-" * 85)
for c in matrix:
    print(f"{c.name:<25} {c.likelihood:<10} {c.impact:<10} {c.risk_score:<6} {c.testing_level:<10} {c.coverage_goal}%")
SOLUTION
}

# === Exercise 2: Test Strategy Document ===
# Problem: Write a complete test strategy document for a project.
# Include scope, levels, tools, environments, responsibilities,
# and criteria.
exercise_2() {
    echo "=== Exercise 2: Test Strategy Document ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
"""
Test Strategy Generator
========================
Generates a structured test strategy document from configuration.
"""

from dataclasses import dataclass, field

@dataclass
class TestStrategy:
    project_name: str
    in_scope: list[str]
    out_of_scope: list[str]

    unit_test_tools: list[str] = field(default_factory=lambda: ["pytest"])
    integration_tools: list[str] = field(default_factory=lambda: ["pytest", "testcontainers"])
    e2e_tools: list[str] = field(default_factory=lambda: ["playwright"])
    ci_platform: str = "GitHub Actions"

    coverage_goal_critical: int = 95
    coverage_goal_standard: int = 80
    coverage_goal_minimal: int = 50

    def generate_document(self) -> str:
        lines = [
            f"# Test Strategy: {self.project_name}",
            "",
            "## 1. Scope",
            "### In Scope",
        ]
        for item in self.in_scope:
            lines.append(f"- {item}")
        lines.append("### Out of Scope")
        for item in self.out_of_scope:
            lines.append(f"- {item}")

        lines.extend([
            "",
            "## 2. Testing Levels",
            f"- **Unit tests**: {', '.join(self.unit_test_tools)} — >80% of all tests",
            f"- **Integration tests**: {', '.join(self.integration_tools)} — critical paths",
            f"- **E2E tests**: {', '.join(self.e2e_tools)} — user journeys",
            "",
            "## 3. Environments",
            "- **Local**: Unit tests, linting (pre-commit)",
            f"- **CI ({self.ci_platform})**: All tests on every PR",
            "- **Staging**: E2E suite, performance tests",
            "- **Production**: Smoke tests, monitoring alerts",
            "",
            "## 4. Coverage Goals",
            f"- Critical components: {self.coverage_goal_critical}%",
            f"- Standard components: {self.coverage_goal_standard}%",
            f"- Minimal components: {self.coverage_goal_minimal}%",
            "",
            "## 5. Entry/Exit Criteria",
            "- **Entry**: Code compiles, linting passes, author self-tested",
            "- **Exit**: CI green, coverage goals met, no open critical bugs",
            "",
            "## 6. Responsibilities",
            "- **Developers**: Unit + integration tests",
            "- **QA**: E2E + exploratory testing",
            "- **SRE**: Performance + monitoring",
        ])

        return "\n".join(lines)

# --- Usage ---

strategy = TestStrategy(
    project_name="E-Commerce Platform",
    in_scope=[
        "Payment processing (Stripe integration)",
        "User authentication and authorization",
        "Order lifecycle (create, process, ship, deliver)",
        "Product search and filtering",
        "Shopping cart operations",
    ],
    out_of_scope=[
        "Third-party email delivery reliability (SLA-covered)",
        "Browser-specific CSS rendering",
        "Legacy admin panel (scheduled for replacement Q3)",
    ],
)

document = strategy.generate_document()
print(document)

# --- Test ---

def test_strategy_document_has_required_sections():
    doc = strategy.generate_document()
    assert "## 1. Scope" in doc
    assert "## 2. Testing Levels" in doc
    assert "## 3. Environments" in doc
    assert "## 4. Coverage Goals" in doc
    assert "## 5. Entry/Exit Criteria" in doc
    assert "## 6. Responsibilities" in doc

def test_strategy_includes_project_name():
    doc = strategy.generate_document()
    assert "E-Commerce Platform" in doc
SOLUTION
}

# === Exercise 3: Metrics Dashboard ===
# Problem: Build a Python script that reads pytest output and produces
# a metrics summary. Track coverage trend, pass rate, and duration
# over the last 10 runs.
exercise_3() {
    echo "=== Exercise 3: Metrics Dashboard ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

@dataclass
class TestRunMetrics:
    date: str
    total_tests: int
    passed: int
    failed: int
    skipped: int
    duration_seconds: float
    line_coverage: float

    @property
    def pass_rate(self) -> float:
        return (self.passed / self.total_tests * 100) if self.total_tests else 0.0

class MetricsDashboard:
    """Collects, stores, and analyzes test metrics over time."""

    def __init__(self, filepath: str = "test_metrics.json"):
        self.filepath = Path(filepath)
        self.history: list[dict] = []
        if self.filepath.exists():
            self.history = json.loads(self.filepath.read_text())

    def record(self, metrics: TestRunMetrics):
        entry = asdict(metrics)
        entry["pass_rate"] = metrics.pass_rate
        self.history.append(entry)
        self.filepath.write_text(json.dumps(self.history, indent=2))

    def summary(self, window: int = 10) -> dict:
        """Summarize the last N runs."""
        recent = self.history[-window:]
        if not recent:
            return {"error": "No data"}

        return {
            "runs_analyzed": len(recent),
            "avg_pass_rate": sum(r["pass_rate"] for r in recent) / len(recent),
            "avg_duration": sum(r["duration_seconds"] for r in recent) / len(recent),
            "coverage_trend": recent[-1]["line_coverage"] - recent[0]["line_coverage"],
            "latest_coverage": recent[-1]["line_coverage"],
            "total_failures": sum(r["failed"] for r in recent),
        }

    def detect_regressions(self, window: int = 5) -> list[str]:
        """Detect negative trends in recent metrics."""
        if len(self.history) < window:
            return []

        recent = self.history[-window:]
        warnings = []

        cov_delta = recent[-1]["line_coverage"] - recent[0]["line_coverage"]
        if cov_delta < -2:
            warnings.append(f"Coverage declining: {cov_delta:.1f}% over {window} runs")

        dur_delta = recent[-1]["duration_seconds"] - recent[0]["duration_seconds"]
        if dur_delta > 60:
            warnings.append(f"Duration increasing: +{dur_delta:.0f}s over {window} runs")

        latest_pass = recent[-1]["pass_rate"]
        if latest_pass < 95:
            warnings.append(f"Pass rate below 95%: {latest_pass:.1f}%")

        return warnings

# --- Tests ---

def test_record_and_summary(tmp_path):
    dashboard = MetricsDashboard(str(tmp_path / "metrics.json"))

    for i in range(5):
        m = TestRunMetrics(
            date=f"2024-01-{i+1:02d}",
            total_tests=100,
            passed=100 - i,
            failed=i,
            skipped=0,
            duration_seconds=60 + i * 5,
            line_coverage=80 + i,
        )
        dashboard.record(m)

    summary = dashboard.summary()
    assert summary["runs_analyzed"] == 5
    assert summary["coverage_trend"] == 4.0  # 80 -> 84
    assert summary["latest_coverage"] == 84.0

def test_detect_regression_coverage(tmp_path):
    dashboard = MetricsDashboard(str(tmp_path / "metrics.json"))

    for i in range(5):
        m = TestRunMetrics(
            date=f"2024-01-{i+1:02d}",
            total_tests=100,
            passed=100,
            failed=0,
            skipped=0,
            duration_seconds=60,
            line_coverage=85 - i * 3,  # Declining: 85 -> 73
        )
        dashboard.record(m)

    warnings = dashboard.detect_regressions()
    assert any("Coverage declining" in w for w in warnings)

def test_no_regressions_when_stable(tmp_path):
    dashboard = MetricsDashboard(str(tmp_path / "metrics.json"))

    for i in range(5):
        m = TestRunMetrics(
            date=f"2024-01-{i+1:02d}",
            total_tests=100,
            passed=100,
            failed=0,
            skipped=0,
            duration_seconds=60,
            line_coverage=85,
        )
        dashboard.record(m)

    warnings = dashboard.detect_regressions()
    assert len(warnings) == 0
SOLUTION
}

# === Exercise 4: Prioritization Exercise ===
# Problem: Given a backlog of 20 untested components and a budget
# of 40 hours, use the prioritization matrix to decide which
# components to test first and at what level.
exercise_4() {
    echo "=== Exercise 4: Prioritization Exercise ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from dataclasses import dataclass

@dataclass
class Component:
    name: str
    business_value: int    # 0-10
    change_frequency: int  # 0-10
    defect_history: int    # 0-10
    complexity: int        # 0-10
    user_facing: int       # 0-10
    estimated_hours: float # Hours to write tests

    @property
    def priority_score(self) -> float:
        return (
            self.business_value * 2.0 +
            self.change_frequency * 1.5 +
            self.defect_history * 1.5 +
            self.complexity * 1.0 +
            self.user_facing * 1.0
        )

def allocate_budget(components: list[Component], budget_hours: float) -> dict:
    """
    Greedily allocate a time budget to the highest-priority components.
    Returns selected components and remaining budget.
    """
    ranked = sorted(components, key=lambda c: c.priority_score, reverse=True)
    selected = []
    remaining = budget_hours

    for comp in ranked:
        if comp.estimated_hours <= remaining:
            selected.append(comp)
            remaining -= comp.estimated_hours

    return {
        "selected": selected,
        "remaining_hours": remaining,
        "coverage_percentage": len(selected) / len(components) * 100,
        "skipped": [c.name for c in ranked if c not in selected],
    }

# --- 20 components ---

backlog = [
    Component("Payment Gateway",      10, 7, 8, 9, 10, 6),
    Component("User Auth",            9,  5, 6, 7, 9,  4),
    Component("Order Pipeline",       9,  8, 7, 8, 8,  5),
    Component("Cart Service",         8,  6, 5, 5, 9,  3),
    Component("Search Engine",        7,  7, 4, 6, 8,  3),
    Component("Product Catalog API",  7,  4, 3, 5, 7,  2),
    Component("Inventory Sync",       6,  5, 5, 6, 3,  3),
    Component("Pricing Engine",       8,  6, 6, 7, 6,  4),
    Component("Recommendation ML",    5,  3, 2, 8, 7,  3),
    Component("Notification Service", 4,  3, 3, 4, 5,  2),
    Component("Shipping Calculator",  6,  4, 4, 5, 6,  2),
    Component("Review System",        4,  3, 2, 3, 6,  2),
    Component("Wishlist",             3,  2, 1, 2, 5,  1),
    Component("Coupon Engine",        6,  5, 5, 5, 5,  2),
    Component("Analytics Pipeline",   3,  2, 2, 5, 1,  2),
    Component("Admin Reports",        3,  2, 1, 3, 2,  1),
    Component("Email Templates",      2,  2, 1, 2, 4,  1),
    Component("Logging Middleware",    2,  1, 1, 2, 0,  1),
    Component("Config Loader",        2,  1, 0, 1, 0,  0.5),
    Component("Health Checks",        3,  1, 1, 1, 2,  0.5),
]

result = allocate_budget(backlog, budget_hours=40)

print(f"Budget: 40 hours")
print(f"Components selected: {len(result['selected'])} / {len(backlog)}")
print(f"Remaining hours: {result['remaining_hours']:.1f}")
print(f"Coverage: {result['coverage_percentage']:.0f}%")
print()
print(f"{'Rank':<5} {'Component':<25} {'Score':<8} {'Hours':<6}")
print("-" * 50)
for i, c in enumerate(result["selected"], 1):
    print(f"{i:<5} {c.name:<25} {c.priority_score:<8.1f} {c.estimated_hours:<6.1f}")
print()
print(f"Skipped: {', '.join(result['skipped'])}")

# --- Tests ---

def test_highest_priority_selected_first():
    result = allocate_budget(backlog, budget_hours=10)
    names = [c.name for c in result["selected"]]
    assert names[0] == "Payment Gateway"  # Highest score

def test_budget_not_exceeded():
    result = allocate_budget(backlog, budget_hours=40)
    total_hours = sum(c.estimated_hours for c in result["selected"])
    assert total_hours <= 40

def test_zero_budget_selects_nothing():
    result = allocate_budget(backlog, budget_hours=0)
    assert len(result["selected"]) == 0
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 18: Test Strategy and Planning"
echo "=============================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
