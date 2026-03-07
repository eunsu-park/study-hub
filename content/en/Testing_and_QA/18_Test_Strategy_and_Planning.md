# Lesson 18: Test Strategy and Planning

**Previous**: [Testing Legacy Code](./17_Testing_Legacy_Code.md) | **Next**: [Overview](./00_Overview.md)

---

Individual testing skills — writing unit tests, mocking dependencies, measuring coverage — are necessary but not sufficient. A test strategy answers the questions that no individual test can: What should we test? How thoroughly? At what level? With what tools? When? And perhaps most importantly: what should we *not* test? Without a coherent strategy, teams end up with test suites that are simultaneously slow, incomplete, and expensive to maintain — a collection of tests that grew organically without direction.

This final lesson brings together everything in this topic into a strategic framework for planning, prioritizing, and sustaining testing across a project's lifecycle.

**Difficulty**: ⭐⭐⭐⭐

**Prerequisites**:
- All previous lessons in this topic (Lessons 01–17)
- Experience working on a team project
- Familiarity with agile/iterative development processes

## Learning Objectives

After completing this lesson, you will be able to:

1. Develop a risk-based test strategy that allocates effort where it matters most
2. Build a test prioritization matrix for time-constrained situations
3. Set realistic coverage goals by component and testing level
4. Use testing metrics to measure and improve quality
5. Integrate testing practices into agile development and CI/CD workflows
6. Build and sustain a testing culture within a development team

---

## 1. What Is a Test Strategy?

A test strategy is a high-level document that answers:

- **Scope**: What is tested? What is explicitly excluded?
- **Levels**: Unit, integration, E2E — what proportion of each?
- **Tools**: Which frameworks, libraries, and services?
- **Environments**: Where do tests run? Local, CI, staging, production?
- **Responsibilities**: Who writes which tests?
- **Criteria**: When is testing "done enough" to release?

### 1.1 Test Strategy vs Test Plan

| Test Strategy | Test Plan |
|---|---|
| High-level, long-lived | Specific, per-release or per-feature |
| Answers "what kind of testing" | Answers "which specific tests" |
| Applies across the project | Applies to a specific scope |
| Updated quarterly or yearly | Updated per sprint or milestone |

A strategy without plans is vague. Plans without a strategy are directionless. You need both.

---

## 2. Risk-Based Testing

No project has infinite time for testing. Risk-based testing ensures that the areas most likely to fail — and most damaging when they do — receive the most attention.

### 2.1 Risk Assessment Matrix

```
                        Impact
                Low         Medium        High
           ┌───────────┬────────────┬────────────┐
    High   │  Monitor  │  Thorough  │  Critical  │
Likelihood │           │  Testing   │  Testing   │
           ├───────────┼────────────┼────────────┤
    Medium │  Basic    │  Standard  │  Thorough  │
           │  Testing  │  Testing   │  Testing   │
           ├───────────┼────────────┼────────────┤
    Low    │  Minimal  │  Basic     │  Standard  │
           │  Testing  │  Testing   │  Testing   │
           └───────────┴────────────┴────────────┘
```

### 2.2 Assessing Likelihood

Factors that increase the likelihood of defects:

| Factor | Low Risk | High Risk |
|---|---|---|
| Code complexity | Simple CRUD | Complex algorithms |
| Change frequency | Stable, rarely modified | Frequent changes |
| Developer experience | Senior, domain expert | New to codebase |
| Dependencies | Few, stable | Many, volatile |
| Technology maturity | Proven stack | Cutting-edge |

### 2.3 Assessing Impact

Factors that increase the impact of defects:

| Factor | Low Impact | High Impact |
|---|---|---|
| Users affected | Internal tool | Customer-facing |
| Data sensitivity | Public data | Financial, PII |
| Recovery difficulty | Easy rollback | Data corruption |
| Regulatory | No compliance | SOX, HIPAA, PCI |
| Revenue | Free feature | Payment processing |

### 2.4 Applying Risk-Based Testing

```python
# Example: Risk-based test allocation for an e-commerce platform

RISK_ASSESSMENT = {
    "payment_processing": {
        "likelihood": "high",     # Complex, many edge cases
        "impact": "critical",     # Revenue, legal, PCI compliance
        "test_strategy": {
            "unit_tests": "comprehensive (>95% coverage)",
            "integration_tests": "all payment gateways, all currencies",
            "e2e_tests": "full checkout flow, including errors",
            "security_tests": "PCI DSS compliance scanning",
            "performance_tests": "load testing under peak traffic",
        }
    },
    "user_profile_page": {
        "likelihood": "low",      # Simple CRUD, stable code
        "impact": "low",          # Cosmetic, no data sensitivity
        "test_strategy": {
            "unit_tests": "basic validation logic",
            "integration_tests": "one happy path",
            "e2e_tests": "none",
        }
    },
    "search_functionality": {
        "likelihood": "medium",   # Moderate complexity
        "impact": "medium",       # Core UX but not data-critical
        "test_strategy": {
            "unit_tests": "query parsing, ranking logic",
            "integration_tests": "search index, relevance",
            "performance_tests": "response time under load",
        }
    },
}
```

---

## 3. Test Prioritization Matrix

When time is limited (it always is), prioritize testing effort using a structured matrix.

### 3.1 The Prioritization Framework

```python
def calculate_test_priority(component):
    """
    Score each component to determine testing priority.
    Higher score = more testing investment needed.
    """
    score = 0

    # Business criticality (0-10)
    score += component.business_value * 2  # Weight: 2x

    # Change frequency (0-10)
    score += component.change_frequency * 1.5  # Weight: 1.5x

    # Defect history (0-10)
    score += component.past_defects * 1.5  # Weight: 1.5x

    # Complexity (0-10)
    score += component.cyclomatic_complexity_normalized

    # User visibility (0-10)
    score += component.user_facing * 1.0

    return score
```

### 3.2 Example Prioritization

| Component | Business | Changes | Defects | Complexity | Visible | Score | Priority |
|---|---|---|---|---|---|---|---|
| Payment | 10 | 6 | 8 | 9 | 10 | 62 | Critical |
| Auth | 9 | 4 | 5 | 7 | 8 | 51 | High |
| Search | 7 | 7 | 4 | 6 | 9 | 48 | High |
| Admin Panel | 3 | 3 | 2 | 4 | 2 | 19 | Low |
| Logging | 2 | 1 | 1 | 2 | 0 | 8 | Minimal |

---

## 4. Coverage Goals by Component

Coverage is not a single number. Different components need different coverage levels based on their risk profile.

### 4.1 Tiered Coverage Goals

```python
COVERAGE_GOALS = {
    # Tier 1: Critical business logic
    "critical": {
        "line_coverage": 95,
        "branch_coverage": 90,
        "mutation_score": 80,
        "components": ["payment", "auth", "data_integrity"],
    },

    # Tier 2: Important but lower risk
    "important": {
        "line_coverage": 80,
        "branch_coverage": 70,
        "mutation_score": 60,
        "components": ["search", "notifications", "reporting"],
    },

    # Tier 3: Standard coverage
    "standard": {
        "line_coverage": 70,
        "branch_coverage": 50,
        "mutation_score": 0,  # Not required
        "components": ["user_profile", "settings", "help"],
    },

    # Tier 4: Minimal coverage (UI, glue code)
    "minimal": {
        "line_coverage": 50,
        "branch_coverage": 0,  # Not measured
        "mutation_score": 0,
        "components": ["admin", "migrations", "scripts"],
    },
}
```

### 4.2 Enforcing Tiered Coverage in CI

```python
# conftest.py
import json

import pytest


def pytest_sessionfinish(session, exitstatus):
    """Check coverage goals by component after all tests run."""
    coverage_file = "coverage.json"
    try:
        with open(coverage_file) as f:
            coverage_data = json.load(f)
    except FileNotFoundError:
        return

    violations = []
    for tier_name, tier_config in COVERAGE_GOALS.items():
        for component in tier_config["components"]:
            actual = get_component_coverage(coverage_data, component)
            goal = tier_config["line_coverage"]
            if actual < goal:
                violations.append(
                    f"{component}: {actual:.1f}% < {goal}% goal ({tier_name} tier)"
                )

    if violations:
        print("\nCoverage goal violations:")
        for v in violations:
            print(f"  FAIL: {v}")
```

---

## 5. Test Documentation

### 5.1 Test Strategy Document Template

```markdown
# Test Strategy: [Project Name]

## 1. Scope
- **In scope**: [List of components/features to test]
- **Out of scope**: [Explicitly excluded items and why]

## 2. Testing Levels
- **Unit tests**: [Who writes, tools, coverage goals]
- **Integration tests**: [Scope, environment, data]
- **E2E tests**: [Scenarios, tools, frequency]
- **Non-functional tests**: [Performance, security, accessibility]

## 3. Environments
- **Local**: [What runs on developer machines]
- **CI**: [What runs on every commit/PR]
- **Staging**: [What runs before release]
- **Production**: [Smoke tests, monitoring]

## 4. Tools
- **Framework**: pytest
- **Mocking**: unittest.mock
- **Coverage**: pytest-cov
- **CI**: GitHub Actions
- **Performance**: Locust
- **Security**: Bandit, pip-audit

## 5. Entry/Exit Criteria
- **Entry**: Code compiles, passes linting, author self-tested
- **Exit**: All CI checks pass, coverage goals met, no critical bugs

## 6. Risk-Based Priorities
[Risk matrix for each component]

## 7. Responsibilities
- **Developers**: Unit tests, integration tests
- **QA**: E2E tests, exploratory testing
- **Security**: Security scans, penetration testing
- **SRE**: Performance testing, monitoring

## 8. Review Schedule
- Strategy reviewed quarterly
- Updated when architecture changes
```

### 5.2 Test Plan for a Feature

```markdown
# Test Plan: User Registration Feature

## Changes
- New API endpoint: POST /api/v1/register
- New database model: PendingRegistration
- Email verification flow

## Test Cases

### Unit Tests
1. Validate email format (valid, invalid, edge cases)
2. Validate password strength (length, complexity)
3. Check duplicate email detection
4. Token generation and expiration

### Integration Tests
1. Full registration → verification → activation flow
2. Registration with existing email
3. Token expiration handling
4. Database constraint enforcement

### E2E Tests
1. Complete registration from UI
2. Email link opens verification page
3. Verified user can log in

## Not Testing (with justification)
- Email delivery reliability (covered by email service SLA)
- Browser compatibility (covered by existing E2E framework)
```

---

## 6. Testing in Agile and CI/CD

### 6.1 Testing in Sprints

```
Sprint Planning:
├── Include test tasks in story estimation
├── "Definition of Done" includes tests
└── Allocate 20-30% of sprint capacity for testing

During Sprint:
├── Write tests alongside code (not after)
├── Run tests locally before pushing
├── Code review includes test review
└── Fix broken tests immediately (stop the line)

Sprint Review:
├── Show test coverage changes
├── Discuss defects found and fixed
└── Review test reliability metrics
```

### 6.2 Testing in CI/CD Pipeline

```
                        Confidence
                        ▲
Commit ──────────────── │ ──────────────── Deploy
  │                     │                    │
  ├─ Lint (seconds)     │ Low risk,          │
  ├─ Unit tests (min)   │ fast feedback      │
  ├─ Integration (min)  │                    │
  ├─ Security scan      │                    │
  │                     │                    │
  ├─ E2E (staging)      │ Higher confidence  │
  ├─ Performance        │                    │
  │                     │                    │
  ├─ Canary deploy      │ Production         │
  ├─ Smoke tests        │ verification       │
  └─ Monitoring/alerts  │                    ▼
```

### 6.3 Shift-Left Testing

"Shift left" means moving testing earlier in the development lifecycle:

| Phase | Traditional | Shift-Left |
|---|---|---|
| Requirements | No testing | Testability review |
| Design | No testing | Test strategy planning |
| Development | Code first, test later | TDD, test alongside code |
| Code review | Check functionality | Review tests too |
| CI | Run tests after merge | Run tests on every commit |
| Release | QA gate at the end | Continuous testing |

---

## 7. Testing Metrics

Metrics should inform decisions, not drive them. A metric without context is dangerous.

### 7.1 Quality Metrics

| Metric | What It Measures | Target | Warning Sign |
|---|---|---|---|
| **Defect density** | Defects per 1,000 lines of code | < 5 | > 10 indicates poor quality |
| **Defect escape rate** | Bugs reaching production / total bugs found | < 10% | > 25% means testing misses too much |
| **MTTR** (Mean Time to Repair) | Average time from bug report to fix deployed | < 4 hours for critical | > 24 hours indicates process issues |
| **MTTD** (Mean Time to Detect) | Time from bug introduction to detection | < 1 day | > 1 sprint means tests are missing |

### 7.2 Test Suite Metrics

| Metric | What It Measures | Target | Warning Sign |
|---|---|---|---|
| **Test execution time** | How long the full suite takes | < 10 minutes | > 30 minutes kills feedback loops |
| **Flaky test rate** | Tests that pass/fail non-deterministically | < 1% | > 5% erodes trust in tests |
| **Coverage trend** | Coverage over time (should increase) | Positive slope | Declining indicates skipping tests |
| **Test-to-code ratio** | Lines of test code per line of production code | 1:1 to 3:1 | < 0.5:1 is undertested |

### 7.3 Tracking Metrics Over Time

```python
# metrics.py — collect and report testing metrics
import json
from dataclasses import dataclass
from datetime import datetime


@dataclass
class TestMetrics:
    date: str
    total_tests: int
    passed: int
    failed: int
    skipped: int
    duration_seconds: float
    line_coverage: float
    flaky_tests: int

    @property
    def pass_rate(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return (self.passed / self.total_tests) * 100

    @property
    def flaky_rate(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return (self.flaky_tests / self.total_tests) * 100


def record_metrics(metrics: TestMetrics, filepath: str = "test_metrics.json"):
    """Append metrics to a JSON file for trend tracking."""
    try:
        with open(filepath) as f:
            history = json.load(f)
    except FileNotFoundError:
        history = []

    history.append({
        "date": metrics.date,
        "total_tests": metrics.total_tests,
        "pass_rate": metrics.pass_rate,
        "duration_seconds": metrics.duration_seconds,
        "line_coverage": metrics.line_coverage,
        "flaky_rate": metrics.flaky_rate,
    })

    with open(filepath, "w") as f:
        json.dump(history, f, indent=2)


def detect_regressions(history: list, window: int = 5) -> list:
    """Detect negative trends in metrics over the last N data points."""
    if len(history) < window:
        return []

    recent = history[-window:]
    warnings = []

    # Coverage declining?
    coverage_trend = recent[-1]["line_coverage"] - recent[0]["line_coverage"]
    if coverage_trend < -2:
        warnings.append(f"Coverage declining: {coverage_trend:.1f}% over {window} runs")

    # Duration increasing?
    duration_trend = recent[-1]["duration_seconds"] - recent[0]["duration_seconds"]
    if duration_trend > 60:
        warnings.append(f"Test duration increasing: +{duration_trend:.0f}s over {window} runs")

    # Flaky rate increasing?
    if recent[-1]["flaky_rate"] > 5:
        warnings.append(f"Flaky test rate: {recent[-1]['flaky_rate']:.1f}% (target: <1%)")

    return warnings
```

---

## 8. Building a Testing Culture

Tools and processes are necessary but not sufficient. A sustainable testing practice requires a team culture that values quality.

### 8.1 Principles of a Testing Culture

1. **Tests are not optional**: They are part of the definition of done, not a nice-to-have
2. **Broken tests are top priority**: A broken test is a production risk; fix it before writing new code
3. **Tests are code**: They deserve the same care, review, and refactoring as production code
4. **Coverage is a floor, not a ceiling**: Meeting the coverage goal does not mean the code is well-tested
5. **Flaky tests are bugs**: They erode trust and must be fixed or deleted

### 8.2 Practical Steps

| Action | Impact | Effort |
|---|---|---|
| Include tests in Definition of Done | High | Low |
| Review tests in code reviews | High | Low |
| Track and publish test metrics | Medium | Low |
| Fix flaky tests immediately | High | Medium |
| Pair programming on test design | High | Medium |
| Test-writing workshops | Medium | Medium |
| Celebrate testing milestones | Medium | Low |

### 8.3 Anti-Patterns to Avoid

- **"We'll add tests later"**: Later never comes. Test alongside code.
- **Coverage theater**: Writing tests just to hit a number. Focus on meaningful coverage.
- **Test-last blame**: Blaming developers for not writing tests without giving them time to do so.
- **Ignoring flaky tests**: "Oh, that test is always flaky, just re-run it." Fix it or delete it.
- **Separate QA team writes all tests**: Developers must own their tests. QA provides a complementary layer, not a substitute.

---

## 9. Putting It All Together

A complete testing strategy for a medium-sized project:

```
Testing Strategy Summary
═══════════════════════

Levels:
  Unit tests      → 70% of tests, <5 min total, >80% coverage
  Integration     → 20% of tests, <10 min total, critical paths
  E2E             → 10% of tests, <15 min total, user journeys

Execution:
  On every commit → Lint, unit tests, security scan
  On every PR     → Integration tests, coverage check
  Nightly         → Full E2E suite, performance tests
  Weekly          → Dependency audit, DAST scan

Risk allocation:
  Critical (payment, auth)     → 95% coverage, property tests, mutation
  Important (search, reports)  → 80% coverage, integration tests
  Standard (profiles, settings)→ 70% coverage, unit tests
  Minimal (admin, scripts)     → 50% coverage, smoke tests

Metrics tracked:
  - Coverage trend (per component)
  - Defect escape rate
  - Test suite duration
  - Flaky test rate
  - MTTR for critical bugs

Review cycle:
  Sprint retro → Review flaky tests, test gaps
  Quarterly    → Review and update test strategy
  Annually     → Evaluate tools and frameworks
```

---

## Exercises

1. **Risk Assessment**: Pick a real application (or a hypothetical e-commerce platform) and create a risk assessment matrix for its top 10 components. Assign testing levels and coverage goals to each.

2. **Test Strategy Document**: Write a complete test strategy document for a project you are working on. Include scope, levels, tools, environments, responsibilities, and criteria.

3. **Metrics Dashboard**: Build a Python script that reads pytest output (JUnit XML + coverage reports) and produces a metrics summary. Track coverage trend, pass rate, and duration over the last 10 runs.

4. **Prioritization Exercise**: Given a backlog of 20 untested components and a budget of 40 hours, use the prioritization matrix to decide which components to test first and at what level. Justify your choices.

5. **Culture Assessment**: Interview three team members about their testing practices. Identify the top three cultural barriers to effective testing and propose concrete actions to address each one.

---

**License**: CC BY-NC 4.0
