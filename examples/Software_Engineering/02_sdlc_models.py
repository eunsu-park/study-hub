"""
Software Development Life Cycle Models

Demonstrates:
1. SDLC Phase Artifacts — what each phase produces
2. Waterfall Model — sequential phases with phase gates
3. V-Model — verification/validation mapping
4. Spiral Model — risk-driven iteration
5. Model Selection Decision Framework

Theory:
- Waterfall: Sequential, document-driven, good for stable requirements.
- V-Model: Each decomposition phase has a corresponding test phase.
- Spiral: Risk analysis drives each iteration; combines prototyping + waterfall.
- Choice depends on requirements stability, risk, team size, and domain.

Adapted from Software Engineering Lesson 02.
"""

from dataclasses import dataclass, field
from enum import Enum


# ─────────────────────────────────────────────────
# 1. SDLC PHASES AND ARTIFACTS
# ─────────────────────────────────────────────────

class Phase(Enum):
    PLANNING = "Planning"
    REQUIREMENTS = "Requirements"
    DESIGN = "Design"
    IMPLEMENTATION = "Implementation"
    TESTING = "Testing"
    DEPLOYMENT = "Deployment"
    MAINTENANCE = "Maintenance"


PHASE_ARTIFACTS = {
    Phase.PLANNING: [
        "Project charter", "Feasibility study", "Resource plan", "Schedule"
    ],
    Phase.REQUIREMENTS: [
        "SRS (Software Requirements Specification)",
        "Use case diagrams", "User stories", "Acceptance criteria"
    ],
    Phase.DESIGN: [
        "Architecture document", "Class diagrams",
        "Database schema", "API specification"
    ],
    Phase.IMPLEMENTATION: [
        "Source code", "Unit tests", "Code review records", "Build scripts"
    ],
    Phase.TESTING: [
        "Test plan", "Test cases", "Bug reports",
        "Coverage report", "Performance benchmarks"
    ],
    Phase.DEPLOYMENT: [
        "Deployment runbook", "Release notes",
        "Infrastructure config", "Rollback plan"
    ],
    Phase.MAINTENANCE: [
        "Change requests", "Incident reports",
        "Patch releases", "Technical debt log"
    ],
}


# ─────────────────────────────────────────────────
# 2. V-MODEL: VERIFICATION ↔ VALIDATION MAPPING
# ─────────────────────────────────────────────────

@dataclass
class VModelPair:
    """A V-Model pair: decomposition phase ↔ testing phase."""
    decomposition: str
    testing: str
    what_is_verified: str


V_MODEL = [
    VModelPair(
        "Business Requirements",
        "Acceptance Testing",
        "Does the system meet business needs?"
    ),
    VModelPair(
        "System Requirements",
        "System Testing",
        "Does the integrated system meet the SRS?"
    ),
    VModelPair(
        "High-Level Design",
        "Integration Testing",
        "Do modules work together correctly?"
    ),
    VModelPair(
        "Detailed Design",
        "Unit Testing",
        "Does each module work in isolation?"
    ),
]


# ─────────────────────────────────────────────────
# 3. SPIRAL MODEL: RISK-DRIVEN ITERATION
# ─────────────────────────────────────────────────

@dataclass
class SpiralIteration:
    """One cycle of the Spiral model."""
    iteration: int
    objective: str
    risks: list[str]
    prototype: str
    decision: str


def simulate_spiral_project() -> list[SpiralIteration]:
    """Simulate a 4-iteration spiral for an e-commerce platform."""
    return [
        SpiralIteration(
            iteration=1,
            objective="Concept of operations",
            risks=["Market viability uncertain", "Technology stack unknown"],
            prototype="Paper prototype + market survey",
            decision="Proceed — market research positive",
        ),
        SpiralIteration(
            iteration=2,
            objective="Software requirements",
            risks=["Payment gateway integration complex", "Performance targets unclear"],
            prototype="Payment gateway proof-of-concept",
            decision="Proceed — Stripe integration validated",
        ),
        SpiralIteration(
            iteration=3,
            objective="Software design",
            risks=["Scalability under load", "Security vulnerabilities"],
            prototype="Load test with 10K concurrent users",
            decision="Proceed — architecture handles target load",
        ),
        SpiralIteration(
            iteration=4,
            objective="Implementation and test",
            risks=["Third-party API rate limits", "Data migration from legacy"],
            prototype="Full system integration test",
            decision="Release — all acceptance criteria met",
        ),
    ]


# ─────────────────────────────────────────────────
# 4. MODEL SELECTION DECISION FRAMEWORK
# ─────────────────────────────────────────────────

@dataclass
class ProjectProfile:
    """Project characteristics for model selection."""
    name: str
    requirements_stability: str   # stable / evolving / unclear
    risk_level: str               # low / medium / high
    team_size: str                # small / medium / large
    domain: str                   # regulated / standard / experimental
    delivery_pressure: str        # low / medium / high


def recommend_model(profile: ProjectProfile) -> str:
    """Recommend an SDLC model based on project characteristics."""
    if profile.domain == "regulated" and profile.requirements_stability == "stable":
        return "Waterfall / V-Model (document-driven, traceable)"
    if profile.risk_level == "high":
        return "Spiral (risk-driven, iterative prototyping)"
    if profile.delivery_pressure == "high" and profile.team_size == "small":
        return "RAD / Prototyping (rapid feedback, throwaway prototypes)"
    if profile.requirements_stability == "evolving":
        return "Agile (iterative, adaptive, customer collaboration)"
    if profile.requirements_stability == "unclear":
        return "Spiral or Prototyping (clarify requirements through iteration)"
    return "Incremental (staged delivery, manageable risk)"


# ─────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────

def demo_phases():
    """Display SDLC phases and their artifacts."""
    print("=" * 65)
    print("  SDLC Phases and Artifacts")
    print("=" * 65)
    for phase in Phase:
        artifacts = PHASE_ARTIFACTS[phase]
        print(f"\n  {phase.value}")
        for a in artifacts:
            print(f"    - {a}")


def demo_v_model():
    """Display V-Model verification/validation pairs."""
    print("\n" + "=" * 65)
    print("  V-Model: Decomposition ↔ Testing")
    print("=" * 65)
    print()
    for pair in V_MODEL:
        print(f"  {pair.decomposition:<24} ←→  {pair.testing}")
        print(f"  {'':24}      {pair.what_is_verified}")
        print()


def demo_spiral():
    """Display spiral model iteration simulation."""
    print("=" * 65)
    print("  Spiral Model: E-Commerce Platform (4 Iterations)")
    print("=" * 65)
    for it in simulate_spiral_project():
        print(f"\n  Iteration {it.iteration}: {it.objective}")
        print(f"  Risks:     {', '.join(it.risks)}")
        print(f"  Prototype: {it.prototype}")
        print(f"  Decision:  {it.decision}")


def demo_model_selection():
    """Demonstrate model selection for different project types."""
    print("\n" + "=" * 65)
    print("  Model Selection Decision Framework")
    print("=" * 65)
    profiles = [
        ProjectProfile("Medical Device Firmware", "stable", "medium",
                        "medium", "regulated", "low"),
        ProjectProfile("Social Media Startup", "evolving", "medium",
                        "small", "standard", "high"),
        ProjectProfile("Autonomous Driving AI", "unclear", "high",
                        "large", "experimental", "medium"),
        ProjectProfile("Internal HR Tool", "evolving", "low",
                        "small", "standard", "medium"),
    ]
    print()
    for p in profiles:
        model = recommend_model(p)
        print(f"  {p.name}")
        print(f"    Requirements: {p.requirements_stability}, "
              f"Risk: {p.risk_level}, Domain: {p.domain}")
        print(f"    → Recommended: {model}")
        print()


if __name__ == "__main__":
    demo_phases()
    demo_v_model()
    demo_spiral()
    demo_model_selection()
