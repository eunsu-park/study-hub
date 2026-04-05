"""
Software Engineering Core Principles

Demonstrates:
1. Brooks's Four Properties of Software
2. Seven Core SE Principles (rigor, separation of concerns,
   modularity, abstraction, anticipation of change, generality, incrementality)
3. Programming vs. Software Engineering comparison

Theory:
- Brooks (1987): Software is inherently complex, conformant,
  changeable, and invisible — there is "No Silver Bullet."
- Core principles guide decisions at every level from architecture
  to individual functions.

Adapted from Software Engineering Lesson 01.
"""

from dataclasses import dataclass, field


# ─────────────────────────────────────────────────
# 1. BROOKS'S FOUR ESSENTIAL PROPERTIES
# ─────────────────────────────────────────────────

@dataclass
class SoftwareProperty:
    """One of Brooks's four essential difficulties of software."""
    name: str
    description: str
    implication: str
    mitigation: str


BROOKS_PROPERTIES = [
    SoftwareProperty(
        name="Complexity",
        description="Software entities are more complex for their size "
                    "than perhaps any other human construct.",
        implication="No two parts are alike; scaling adds non-linear complexity.",
        mitigation="Modularization, abstraction, layered architecture.",
    ),
    SoftwareProperty(
        name="Conformity",
        description="Software must conform to human institutions, "
                    "regulations, and other systems — not just physics.",
        implication="Arbitrary complexity from external constraints.",
        mitigation="Adapter patterns, anti-corruption layers, standards.",
    ),
    SoftwareProperty(
        name="Changeability",
        description="Software is constantly pressured to change because "
                    "it is perceived as easy to change.",
        implication="Successful software attracts change requests; "
                    "the more useful, the more pressure to evolve.",
        mitigation="Design for change: SOLID, dependency injection, feature flags.",
    ),
    SoftwareProperty(
        name="Invisibility",
        description="Software has no spatial representation; it cannot "
                    "be visualized in the way buildings or circuits can.",
        implication="Hard to communicate structure to stakeholders.",
        mitigation="UML diagrams, architecture docs, metaphors, prototypes.",
    ),
]


# ─────────────────────────────────────────────────
# 2. SEVEN CORE PRINCIPLES
# ─────────────────────────────────────────────────

@dataclass
class SEPrinciple:
    """A core Software Engineering principle."""
    name: str
    definition: str
    code_example: str
    violation_example: str


PRINCIPLES = [
    SEPrinciple(
        name="Rigor and Formality",
        definition="Apply systematic, disciplined approaches; "
                   "use formal methods where appropriate.",
        code_example="Type hints, contracts, automated tests, code review",
        violation_example="No tests, manual ad-hoc verification, 'it works on my machine'",
    ),
    SEPrinciple(
        name="Separation of Concerns",
        definition="Divide a system into distinct features "
                   "with minimal overlapping functionality.",
        code_example="MVC pattern: Model handles data, View handles display, "
                     "Controller handles logic",
        violation_example="God class that handles DB, UI, and business logic together",
    ),
    SEPrinciple(
        name="Modularity",
        definition="Decompose a system into modules with well-defined interfaces; "
                   "high cohesion, low coupling.",
        code_example="Package per feature: auth/, payments/, notifications/",
        violation_example="Single 5000-line file with all application logic",
    ),
    SEPrinciple(
        name="Abstraction",
        definition="Focus on essential properties while hiding "
                   "implementation details.",
        code_example="def send_notification(user, message) — "
                     "hides email/SMS/push channel selection",
        violation_example="Caller constructs SMTP connection, email headers, "
                          "and sends directly",
    ),
    SEPrinciple(
        name="Anticipation of Change",
        definition="Design so that likely future changes are easy to make.",
        code_example="Strategy pattern for payment processing; "
                     "add new providers without modifying existing code",
        violation_example="Hardcoded if/elif chain for each payment provider",
    ),
    SEPrinciple(
        name="Generality",
        definition="Solve a more general problem when the cost is acceptable; "
                   "reuse over reinvention.",
        code_example="Parameterized query builder instead of hand-written SQL "
                     "for each table",
        violation_example="Copy-pasted SQL queries with minor variations "
                          "across 20 functions",
    ),
    SEPrinciple(
        name="Incrementality",
        definition="Develop in small, validated steps rather than "
                   "big-bang delivery.",
        code_example="Feature flags, CI/CD pipeline, iterative releases "
                     "with user feedback",
        violation_example="12-month development cycle with first user contact at launch",
    ),
]


# ─────────────────────────────────────────────────
# 3. PROGRAMMING vs. SOFTWARE ENGINEERING
# ─────────────────────────────────────────────────

@dataclass
class ComparisonRow:
    dimension: str
    programming: str
    software_engineering: str


COMPARISON = [
    ComparisonRow("Goal", "Working code", "Maintainable, reliable systems"),
    ComparisonRow("Scope", "Individual function/script", "Full lifecycle of a product"),
    ComparisonRow("Team", "Often solo", "Cross-functional teams"),
    ComparisonRow("Time Horizon", "Immediate (get it done)", "Long-term (years of evolution)"),
    ComparisonRow("Testing", "Manual spot-check", "Automated test suites, CI/CD"),
    ComparisonRow("Documentation", "Optional", "Essential (ADRs, specs, runbooks)"),
    ComparisonRow("Change", "Rewrite if needed", "Refactor incrementally"),
    ComparisonRow("Process", "Ad hoc", "Defined methodology (Agile, etc.)"),
]


# ─────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────

def demo_brooks_properties():
    """Display Brooks's four essential difficulties."""
    print("=" * 65)
    print("  Brooks's Four Essential Properties of Software (1987)")
    print("=" * 65)
    for prop in BROOKS_PROPERTIES:
        print(f"\n  {prop.name}")
        print(f"  {'─' * 40}")
        print(f"  Description:  {prop.description}")
        print(f"  Implication:  {prop.implication}")
        print(f"  Mitigation:   {prop.mitigation}")


def demo_principles():
    """Display the seven core principles with examples."""
    print("\n" + "=" * 65)
    print("  Seven Core Software Engineering Principles")
    print("=" * 65)
    for i, p in enumerate(PRINCIPLES, 1):
        print(f"\n  {i}. {p.name}")
        print(f"     {p.definition}")
        print(f"     Good:  {p.code_example}")
        print(f"     Bad:   {p.violation_example}")


def demo_comparison():
    """Display programming vs. software engineering comparison."""
    print("\n" + "=" * 65)
    print("  Programming vs. Software Engineering")
    print("=" * 65)
    header = f"  {'Dimension':<18} {'Programming':<28} {'Software Engineering'}"
    print(f"\n{header}")
    print(f"  {'─' * 18} {'─' * 28} {'─' * 28}")
    for row in COMPARISON:
        print(f"  {row.dimension:<18} {row.programming:<28} {row.software_engineering}")


if __name__ == "__main__":
    demo_brooks_properties()
    demo_principles()
    demo_comparison()

    print("\n" + "=" * 65)
    print("  Key takeaway: Software engineering is programming integrated")
    print("  over time — accounting for maintenance, teams, and change.")
    print("=" * 65)
