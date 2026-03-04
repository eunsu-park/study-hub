"""
Software Process Improvement

Demonstrates:
1. CMMI Maturity Level Assessment
2. GQM (Goal-Question-Metric) Framework
3. Root Cause Analysis (5 Whys, Fishbone)
4. DORA Metrics Benchmarking
5. PDCA (Plan-Do-Check-Act) Cycle

Theory:
- CMMI: 5 maturity levels from Initial (chaotic) to Optimizing (continuous improvement).
- GQM: Goal → Questions → Metrics — ties measurement to business objectives.
- 5 Whys: Iterative root cause discovery; stop when you reach a systemic cause.
- DORA: Four key metrics that correlate with high-performing teams.

Adapted from Software Engineering Lesson 12.
"""

from dataclasses import dataclass, field
from enum import IntEnum


# ─────────────────────────────────────────────────
# 1. CMMI MATURITY ASSESSMENT
# ─────────────────────────────────────────────────

class CMMILevel(IntEnum):
    INITIAL = 1
    MANAGED = 2
    DEFINED = 3
    QUANTITATIVELY_MANAGED = 4
    OPTIMIZING = 5


@dataclass
class ProcessArea:
    """A CMMI process area with current capability."""
    name: str
    level: int
    practices: list[str]
    gaps: list[str]


def assess_maturity(areas: list[ProcessArea]) -> dict:
    """Determine overall CMMI maturity level."""
    if not areas:
        return {"level": 1, "name": "Initial"}

    min_level = min(a.level for a in areas)
    total_gaps = sum(len(a.gaps) for a in areas)

    level_names = {
        1: "Initial — Ad hoc, chaotic",
        2: "Managed — Project-level discipline",
        3: "Defined — Organization-wide standards",
        4: "Quantitatively Managed — Data-driven",
        5: "Optimizing — Continuous improvement",
    }

    return {
        "level": min_level,
        "name": level_names.get(min_level, "Unknown"),
        "process_areas": len(areas),
        "total_gaps": total_gaps,
        "areas_at_target": sum(1 for a in areas if not a.gaps),
    }


# ─────────────────────────────────────────────────
# 2. GQM (GOAL-QUESTION-METRIC)
# ─────────────────────────────────────────────────

@dataclass
class GQMGoal:
    """A GQM goal with questions and metrics."""
    purpose: str
    object: str
    viewpoint: str
    questions: list[dict] = field(default_factory=list)

    @property
    def goal_statement(self) -> str:
        return f"Analyze {self.object} for the purpose of {self.purpose} " \
               f"from the viewpoint of {self.viewpoint}."


def build_gqm_model() -> list[GQMGoal]:
    """Build a sample GQM model for a development team."""
    return [
        GQMGoal(
            purpose="improving delivery speed",
            object="the release process",
            viewpoint="the engineering manager",
            questions=[
                {
                    "question": "How long does it take from commit to production?",
                    "metrics": ["Lead time (median, p95)", "Deploy frequency"],
                },
                {
                    "question": "What percentage of deployments cause incidents?",
                    "metrics": ["Change failure rate (%)", "Rollback count"],
                },
            ],
        ),
        GQMGoal(
            purpose="reducing defect rate",
            object="the code review process",
            viewpoint="the QA lead",
            questions=[
                {
                    "question": "Are code reviews catching defects before merge?",
                    "metrics": ["Defects found in review vs. production",
                                "Review turnaround time"],
                },
                {
                    "question": "Is review coverage adequate?",
                    "metrics": ["% of PRs with >=1 reviewer",
                                "Lines changed per review"],
                },
            ],
        ),
    ]


# ─────────────────────────────────────────────────
# 3. ROOT CAUSE ANALYSIS
# ─────────────────────────────────────────────────

@dataclass
class FiveWhysAnalysis:
    """5 Whys root cause analysis."""
    incident: str
    whys: list[str]
    root_cause: str
    corrective_action: str


def run_five_whys() -> FiveWhysAnalysis:
    """Run a 5 Whys analysis on a production incident."""
    return FiveWhysAnalysis(
        incident="Production API returned 500 errors for 30 minutes on Friday",
        whys=[
            "Why? — A config change was deployed with an invalid DB connection string.",
            "Why? — The config was not validated before deployment.",
            "Why? — The CI pipeline does not include config validation.",
            "Why? — Config validation was never added as a pipeline step.",
            "Why? — There is no standard checklist for pipeline requirements.",
        ],
        root_cause="No standard process for ensuring all deployment artifacts "
                   "are validated in CI.",
        corrective_action="Add config validation step to CI pipeline + "
                          "create pipeline requirements checklist.",
    )


@dataclass
class FishboneDiagram:
    """Ishikawa (Fishbone) diagram for cause categorization."""
    problem: str
    categories: dict[str, list[str]]  # category -> causes

    def display(self) -> str:
        lines = [f"  Problem: {self.problem}", ""]
        for cat, causes in self.categories.items():
            lines.append(f"  [{cat}]")
            for c in causes:
                lines.append(f"    └─ {c}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────
# 4. DORA METRICS
# ─────────────────────────────────────────────────

@dataclass
class DORAMetrics:
    """DORA Four Key Metrics for a team."""
    team: str
    deploy_frequency: str      # on-demand / daily / weekly / monthly
    lead_time: str             # <1h / <1d / <1w / <1m
    change_failure_rate: float  # 0-1
    mttr: str                  # <1h / <1d / <1w / <1m


DORA_BENCHMARKS = {
    "Elite": {"deploy_freq": "on-demand", "lead_time": "<1h",
              "cfr": 0.05, "mttr": "<1h"},
    "High": {"deploy_freq": "daily", "lead_time": "<1d",
             "cfr": 0.10, "mttr": "<1d"},
    "Medium": {"deploy_freq": "weekly", "lead_time": "<1w",
               "cfr": 0.15, "mttr": "<1d"},
    "Low": {"deploy_freq": "monthly", "lead_time": "<1m",
            "cfr": 0.45, "mttr": "<1w"},
}


def classify_team(metrics: DORAMetrics) -> str:
    """Classify team performance based on DORA benchmarks."""
    freq_order = ["on-demand", "daily", "weekly", "monthly"]
    time_order = ["<1h", "<1d", "<1w", "<1m"]

    freq_idx = freq_order.index(metrics.deploy_frequency) if metrics.deploy_frequency in freq_order else 3
    lt_idx = time_order.index(metrics.lead_time) if metrics.lead_time in time_order else 3

    if freq_idx <= 0 and lt_idx <= 0 and metrics.change_failure_rate <= 0.05:
        return "Elite"
    if freq_idx <= 1 and lt_idx <= 1 and metrics.change_failure_rate <= 0.10:
        return "High"
    if freq_idx <= 2 and lt_idx <= 2 and metrics.change_failure_rate <= 0.15:
        return "Medium"
    return "Low"


# ─────────────────────────────────────────────────
# 5. PDCA CYCLE
# ─────────────────────────────────────────────────

@dataclass
class PDCACycle:
    """Plan-Do-Check-Act improvement cycle."""
    goal: str
    plan: str
    do: str
    check: str
    act: str
    outcome: str


# ─────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────

def demo_cmmi():
    print("=" * 65)
    print("  CMMI Maturity Assessment")
    print("=" * 65)

    areas = [
        ProcessArea("Requirements Management", 2,
                    ["Req gathering", "Traceability"],
                    []),
        ProcessArea("Project Planning", 2,
                    ["Estimation", "Scheduling", "Risk identification"],
                    ["No resource leveling"]),
        ProcessArea("Configuration Management", 2,
                    ["Version control", "Change management"],
                    []),
        ProcessArea("Process Standardization", 1,
                    ["Ad hoc processes"],
                    ["No org-wide standards", "No process documentation"]),
        ProcessArea("Quantitative Management", 1,
                    [],
                    ["No metrics", "No statistical process control"]),
    ]

    result = assess_maturity(areas)
    print(f"\n  Overall Level: {result['level']} — {result['name']}")
    print(f"  Process Areas: {result['process_areas']}, Gaps: {result['total_gaps']}")
    print(f"  Areas at target: {result['areas_at_target']}/{result['process_areas']}")

    print("\n  Detailed assessment:")
    for a in areas:
        status = "OK" if not a.gaps else f"{len(a.gaps)} gap(s)"
        print(f"    L{a.level} {a.name}: {status}")
        for g in a.gaps:
            print(f"         └─ {g}")


def demo_gqm():
    print("\n" + "=" * 65)
    print("  GQM (Goal-Question-Metric) Model")
    print("=" * 65)

    for goal in build_gqm_model():
        print(f"\n  Goal: {goal.goal_statement}")
        for q in goal.questions:
            print(f"    Q: {q['question']}")
            for m in q["metrics"]:
                print(f"       M: {m}")


def demo_root_cause():
    print("\n" + "=" * 65)
    print("  Root Cause Analysis")
    print("=" * 65)

    analysis = run_five_whys()
    print(f"\n  Incident: {analysis.incident}")
    print()
    for i, why in enumerate(analysis.whys, 1):
        print(f"  {i}. {why}")
    print(f"\n  Root cause:  {analysis.root_cause}")
    print(f"  Action:      {analysis.corrective_action}")

    print("\n  Fishbone Diagram:")
    fishbone = FishboneDiagram(
        problem="Frequent production incidents",
        categories={
            "Process": ["No deployment checklist", "Manual config changes"],
            "People": ["Insufficient on-call training", "Knowledge silos"],
            "Technology": ["No automated rollback", "Missing health checks"],
            "Environment": ["Staging ≠ Production", "No feature flags"],
        },
    )
    print(fishbone.display())


def demo_dora():
    print("\n" + "=" * 65)
    print("  DORA Metrics Benchmarking")
    print("=" * 65)

    teams = [
        DORAMetrics("Platform", "on-demand", "<1h", 0.03, "<1h"),
        DORAMetrics("Mobile", "weekly", "<1w", 0.12, "<1d"),
        DORAMetrics("Legacy", "monthly", "<1m", 0.40, "<1w"),
    ]

    print(f"\n  {'Team':<12} {'Deploy':>10} {'Lead Time':>10} "
          f"{'CFR':>6} {'MTTR':>6} {'Class':>8}")
    print(f"  {'─' * 56}")
    for t in teams:
        cls = classify_team(t)
        print(f"  {t.team:<12} {t.deploy_frequency:>10} {t.lead_time:>10} "
              f"{t.change_failure_rate:>5.0%} {t.mttr:>6} {cls:>8}")


def demo_pdca():
    print("\n" + "=" * 65)
    print("  PDCA Improvement Cycle")
    print("=" * 65)

    cycle = PDCACycle(
        goal="Reduce code review turnaround from 48h to 24h",
        plan="Set SLA of 24h, add dashboard, pair reviewers to PRs",
        do="Run for 2 sprints, collect turnaround data daily",
        check="Median turnaround dropped to 18h; 2 reviewers overwhelmed",
        act="Redistribute review load; adopt standard for next cycle",
        outcome="New baseline: 18h median, ready for next improvement target",
    )
    for phase in ["plan", "do", "check", "act"]:
        print(f"\n  {phase.upper()}: {getattr(cycle, phase)}")
    print(f"\n  Outcome: {cycle.outcome}")


if __name__ == "__main__":
    demo_cmmi()
    demo_gqm()
    demo_root_cause()
    demo_dora()
    demo_pdca()
