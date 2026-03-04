"""
Team Dynamics and Communication

Demonstrates:
1. Team Structure Models (functional, cross-functional, Team Topologies)
2. Conway's Law Analyzer
3. Code Review Quality Checker
4. Meeting Effectiveness Tracker
5. Psychological Safety Assessment

Theory:
- Conway's Law: Organizations design systems mirroring their
  communication structure. Inverse Conway Maneuver: intentionally
  structure teams to get the architecture you want.
- Team Topologies (Skelton & Pais): 4 team types + 3 interaction modes
  minimize cognitive load and coordination overhead.
- Psychological Safety (Edmondson / Google Project Aristotle): the #1
  predictor of team effectiveness.

Adapted from Software Engineering Lesson 15.
"""

from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


# ─────────────────────────────────────────────────
# 1. TEAM STRUCTURE MODELS
# ─────────────────────────────────────────────────

class TeamType(Enum):
    STREAM_ALIGNED = "Stream-aligned"
    PLATFORM = "Platform"
    ENABLING = "Enabling"
    COMPLICATED_SUBSYSTEM = "Complicated-subsystem"


class InteractionMode(Enum):
    COLLABORATION = "Collaboration"
    X_AS_A_SERVICE = "X-as-a-Service"
    FACILITATING = "Facilitating"


@dataclass
class Team:
    name: str
    team_type: TeamType
    members: int
    domain: str
    dependencies: list[str] = field(default_factory=list)


@dataclass
class TeamInteraction:
    from_team: str
    to_team: str
    mode: InteractionMode
    purpose: str


def analyze_team_structure(teams: list[Team],
                           interactions: list[TeamInteraction]) -> dict:
    """Analyze team structure for potential issues."""
    issues = []

    # Check team sizes (Dunbar: ~7±2 ideal)
    for t in teams:
        if t.members > 9:
            issues.append(f"{t.name}: team too large ({t.members}), "
                          "consider splitting")
        if t.members < 3:
            issues.append(f"{t.name}: team too small ({t.members}), "
                          "bus factor risk")

    # Check dependency count
    dep_counts = defaultdict(int)
    for t in teams:
        dep_counts[t.name] = len(t.dependencies)
    for name, count in dep_counts.items():
        if count > 3:
            issues.append(f"{name}: too many dependencies ({count}), "
                          "coordination overhead")

    # Check for missing platform team if multiple stream-aligned teams
    stream_teams = [t for t in teams if t.team_type == TeamType.STREAM_ALIGNED]
    platform_teams = [t for t in teams if t.team_type == TeamType.PLATFORM]
    if len(stream_teams) > 2 and not platform_teams:
        issues.append("Multiple stream-aligned teams but no platform team — "
                      "risk of duplicated infrastructure work")

    return {
        "team_count": len(teams),
        "total_members": sum(t.members for t in teams),
        "interactions": len(interactions),
        "issues": issues,
    }


# ─────────────────────────────────────────────────
# 2. CONWAY'S LAW ANALYZER
# ─────────────────────────────────────────────────

@dataclass
class SystemModule:
    name: str
    owning_team: str
    depends_on: list[str] = field(default_factory=list)


def detect_conway_misalignment(teams: list[Team],
                                modules: list[SystemModule]) -> list[str]:
    """Detect where org structure conflicts with system architecture."""
    misalignments = []

    for module in modules:
        for dep_name in module.depends_on:
            dep_module = next((m for m in modules if m.name == dep_name), None)
            if dep_module and dep_module.owning_team != module.owning_team:
                # Cross-team dependency
                misalignments.append(
                    f"{module.name} ({module.owning_team}) depends on "
                    f"{dep_name} ({dep_module.owning_team}) — "
                    "cross-team coordination needed"
                )

    return misalignments


# ─────────────────────────────────────────────────
# 3. CODE REVIEW QUALITY CHECKER
# ─────────────────────────────────────────────────

@dataclass
class ReviewComment:
    reviewer: str
    comment: str
    is_blocking: bool
    category: str  # bug / style / design / nit / question


def assess_review_quality(comments: list[ReviewComment]) -> dict:
    """Assess code review quality based on comment distribution."""
    if not comments:
        return {"quality": "insufficient", "issues": ["No comments — rubber stamp?"]}

    by_category = defaultdict(int)
    for c in comments:
        by_category[c.category] += 1

    total = len(comments)
    blocking = sum(1 for c in comments if c.is_blocking)
    issues = []

    # Quality heuristics
    if by_category.get("nit", 0) / total > 0.5:
        issues.append("Too many nits — focus on substance over style")
    if by_category.get("bug", 0) == 0 and by_category.get("design", 0) == 0:
        issues.append("No bug/design feedback — review may be too shallow")
    if blocking > total * 0.5:
        issues.append("Most comments are blocking — consider non-blocking suggestions")
    if len(set(c.reviewer for c in comments)) < 2:
        issues.append("Single reviewer — consider adding a second perspective")

    quality = "good" if not issues else "needs improvement"

    return {
        "quality": quality,
        "total_comments": total,
        "blocking": blocking,
        "by_category": dict(by_category),
        "issues": issues,
    }


# ─────────────────────────────────────────────────
# 4. MEETING EFFECTIVENESS
# ─────────────────────────────────────────────────

@dataclass
class Meeting:
    name: str
    duration_min: int
    attendees: int
    has_agenda: bool
    has_action_items: bool
    decision_count: int
    could_be_async: bool


def calculate_meeting_cost(meetings: list[Meeting],
                           hourly_rate: float = 75.0) -> dict:
    """Calculate meeting costs and identify improvement opportunities."""
    total_person_hours = 0
    wasteful = []

    for m in meetings:
        person_hours = (m.duration_min / 60) * m.attendees
        total_person_hours += person_hours

        if m.could_be_async:
            wasteful.append(f"{m.name}: could be async "
                            f"(saves {person_hours:.1f} person-hours)")
        if not m.has_agenda:
            wasteful.append(f"{m.name}: no agenda — unclear purpose")
        if not m.has_action_items and m.decision_count == 0:
            wasteful.append(f"{m.name}: no outcomes — meeting had no result")

    weekly_cost = total_person_hours * hourly_rate

    return {
        "weekly_person_hours": round(total_person_hours, 1),
        "weekly_cost": round(weekly_cost, 2),
        "meeting_count": len(meetings),
        "improvement_opportunities": wasteful,
    }


# ─────────────────────────────────────────────────
# 5. PSYCHOLOGICAL SAFETY ASSESSMENT
# ─────────────────────────────────────────────────

@dataclass
class SafetyIndicator:
    question: str
    score: int  # 1-5 (Likert scale)
    category: str


def assess_psychological_safety(indicators: list[SafetyIndicator]) -> dict:
    """Assess team psychological safety from survey responses."""
    if not indicators:
        return {"score": 0, "level": "unknown"}

    avg = sum(i.score for i in indicators) / len(indicators)
    by_cat = defaultdict(list)
    for i in indicators:
        by_cat[i.category].append(i.score)

    cat_avgs = {cat: sum(scores) / len(scores) for cat, scores in by_cat.items()}
    weakest = min(cat_avgs, key=cat_avgs.get)

    if avg >= 4.0:
        level = "High — team feels safe to take risks"
    elif avg >= 3.0:
        level = "Moderate — some hesitation to speak up"
    else:
        level = "Low — significant fear of negative consequences"

    return {
        "overall_score": round(avg, 2),
        "level": level,
        "by_category": {k: round(v, 2) for k, v in cat_avgs.items()},
        "weakest_area": weakest,
    }


# ─────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────

def demo_team_structure():
    print("=" * 65)
    print("  Team Structure Analysis (Team Topologies)")
    print("=" * 65)

    teams = [
        Team("Checkout", TeamType.STREAM_ALIGNED, 6, "Payments",
             ["Platform", "Notifications"]),
        Team("Search", TeamType.STREAM_ALIGNED, 7, "Discovery",
             ["Platform"]),
        Team("Platform", TeamType.PLATFORM, 5, "Infrastructure", []),
        Team("ML", TeamType.COMPLICATED_SUBSYSTEM, 4, "Recommendations",
             ["Platform"]),
        Team("DevEx", TeamType.ENABLING, 3, "Developer Productivity", []),
    ]

    interactions = [
        TeamInteraction("Checkout", "Platform", InteractionMode.X_AS_A_SERVICE,
                        "Deploys via platform CI/CD"),
        TeamInteraction("Search", "Platform", InteractionMode.X_AS_A_SERVICE,
                        "Uses platform search infrastructure"),
        TeamInteraction("DevEx", "Checkout", InteractionMode.FACILITATING,
                        "Helping adopt observability stack"),
        TeamInteraction("ML", "Search", InteractionMode.COLLABORATION,
                        "Integrating recommendation engine"),
    ]

    result = analyze_team_structure(teams, interactions)
    print(f"\n  Teams: {result['team_count']}, "
          f"Members: {result['total_members']}, "
          f"Interactions: {result['interactions']}")

    print("\n  Team map:")
    for t in teams:
        deps = f" → [{', '.join(t.dependencies)}]" if t.dependencies else ""
        print(f"    [{t.team_type.value}] {t.name} ({t.members}){deps}")

    if result["issues"]:
        print("\n  Issues:")
        for issue in result["issues"]:
            print(f"    ! {issue}")
    else:
        print("\n  No structural issues detected.")


def demo_conway():
    print("\n" + "=" * 65)
    print("  Conway's Law Analysis")
    print("=" * 65)

    teams = [
        Team("Checkout", TeamType.STREAM_ALIGNED, 6, "Payments"),
        Team("Search", TeamType.STREAM_ALIGNED, 7, "Discovery"),
    ]

    modules = [
        SystemModule("checkout-service", "Checkout", ["payment-gateway", "user-service"]),
        SystemModule("payment-gateway", "Checkout", []),
        SystemModule("search-service", "Search", ["product-catalog"]),
        SystemModule("product-catalog", "Search", []),
        SystemModule("user-service", "Search", []),  # owned by Search but used by Checkout
    ]

    misalignments = detect_conway_misalignment(teams, modules)
    if misalignments:
        print("\n  Cross-team dependencies detected:")
        for m in misalignments:
            print(f"    ! {m}")
        print("\n  Consider: move user-service ownership to a shared "
              "platform team (Inverse Conway Maneuver)")
    else:
        print("\n  Architecture aligns with team structure.")


def demo_review_quality():
    print("\n" + "=" * 65)
    print("  Code Review Quality Assessment")
    print("=" * 65)

    comments = [
        ReviewComment("Alice", "This could cause a null pointer on line 42", True, "bug"),
        ReviewComment("Alice", "Consider extracting this into a separate class", False, "design"),
        ReviewComment("Bob", "Missing semicolon", False, "nit"),
        ReviewComment("Bob", "Why not use a HashMap here?", False, "question"),
        ReviewComment("Alice", "Race condition if called from multiple threads", True, "bug"),
    ]

    result = assess_review_quality(comments)
    print(f"\n  Quality: {result['quality']}")
    print(f"  Comments: {result['total_comments']} "
          f"({result['blocking']} blocking)")
    print(f"  By category: {result['by_category']}")
    if result["issues"]:
        print("  Suggestions:")
        for issue in result["issues"]:
            print(f"    → {issue}")


def demo_meetings():
    print("\n" + "=" * 65)
    print("  Meeting Effectiveness Analysis")
    print("=" * 65)

    meetings = [
        Meeting("Daily Standup", 15, 8, True, False, 0, False),
        Meeting("Sprint Planning", 120, 8, True, True, 5, False),
        Meeting("Status Update", 60, 12, False, False, 0, True),
        Meeting("Design Review", 45, 5, True, True, 3, False),
        Meeting("All-Hands", 60, 40, True, False, 1, True),
    ]

    result = calculate_meeting_cost(meetings)
    print(f"\n  Weekly: {result['meeting_count']} meetings, "
          f"{result['weekly_person_hours']} person-hours, "
          f"${result['weekly_cost']:,.0f}")

    if result["improvement_opportunities"]:
        print("\n  Improvement opportunities:")
        for opp in result["improvement_opportunities"]:
            print(f"    → {opp}")


def demo_safety():
    print("\n" + "=" * 65)
    print("  Psychological Safety Assessment")
    print("=" * 65)

    indicators = [
        SafetyIndicator("I feel safe to take risks on this team", 4, "risk_taking"),
        SafetyIndicator("I can bring up problems without blame", 4, "openness"),
        SafetyIndicator("My mistakes are not held against me", 3, "mistake_tolerance"),
        SafetyIndicator("I can ask for help without judgment", 5, "help_seeking"),
        SafetyIndicator("I feel comfortable challenging ideas", 3, "openness"),
        SafetyIndicator("Different perspectives are valued", 4, "inclusion"),
        SafetyIndicator("Failures lead to learning, not punishment", 3, "mistake_tolerance"),
        SafetyIndicator("I can be myself at work", 4, "inclusion"),
    ]

    result = assess_psychological_safety(indicators)
    print(f"\n  Overall: {result['overall_score']}/5.0 — {result['level']}")
    print(f"  By category:")
    for cat, score in result["by_category"].items():
        bar = "█" * int(score) + "░" * (5 - int(score))
        print(f"    {cat:<20} {score:.1f} {bar}")
    print(f"\n  Focus area: {result['weakest_area']}")


if __name__ == "__main__":
    demo_team_structure()
    demo_conway()
    demo_review_quality()
    demo_meetings()
    demo_safety()
