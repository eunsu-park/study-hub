# 16_open_problems.py — Research landscape analyzer
#
# Run: python 16_open_problems.py

"""
Analyzes the AI safety research landscape: maps open problems,
tracks research directions, identifies gaps, and visualizes
the dependency graph between research areas.
"""

from dataclasses import dataclass, field
from enum import Enum


class Maturity(Enum):
    NASCENT = "nascent"          # Early exploration
    EMERGING = "emerging"        # Active investigation, few results
    DEVELOPING = "developing"    # Growing body of work
    ESTABLISHED = "established"  # Well-studied, some solutions
    MATURE = "mature"            # Largely solved (for current scale)


class Urgency(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ResearchProblem:
    id: str
    name: str
    description: str
    maturity: Maturity
    urgency: Urgency
    key_papers: list[str]
    dependencies: list[str]  # IDs of prerequisite problems
    approaches: list[str]
    open_questions: list[str]
    estimated_difficulty: float  # 0.0 to 1.0


@dataclass
class ResearchGap:
    problem_id: str
    gap_description: str
    importance: float  # 0.0 to 1.0
    suggested_direction: str


class ResearchLandscape:
    """Maps the AI safety research landscape."""

    def __init__(self):
        self.problems: dict[str, ResearchProblem] = {}
        self._populate()

    def _populate(self):
        problems = [
            ResearchProblem(
                "scalable_oversight", "Scalable Oversight",
                "How to supervise AI systems smarter than the supervisor",
                Maturity.EMERGING, Urgency.CRITICAL,
                ["Bowman2022-Debate", "Burns2023-WeakToStrong"],
                [],
                ["Debate", "Recursive reward modeling",
                 "Weak-to-strong generalization"],
                ["Does debate reliably find truth?",
                 "What is the PGR ceiling?"],
                0.85
            ),
            ResearchProblem(
                "interpretability", "Mechanistic Interpretability",
                "Understanding internal computations of neural networks",
                Maturity.DEVELOPING, Urgency.HIGH,
                ["Elhage2022-ToyModels", "Nanda2023-IOI"],
                [],
                ["Circuit analysis", "Linear probes",
                 "Sparse autoencoders", "Activation patching"],
                ["Can we scale to frontier models?",
                 "Is linear representation hypothesis sufficient?"],
                0.80
            ),
            ResearchProblem(
                "robustness", "Adversarial Robustness",
                "Making models robust to adversarial inputs",
                Maturity.ESTABLISHED, Urgency.HIGH,
                ["Madry2018-PGD", "Carlini2017-CW"],
                [],
                ["Adversarial training", "Certified defenses",
                 "Input sanitization"],
                ["Is there an inherent robustness-accuracy tradeoff?",
                 "How to defend against semantic attacks?"],
                0.65
            ),
            ResearchProblem(
                "reward_modeling", "Reward Modeling",
                "Learning human preferences accurately and robustly",
                Maturity.DEVELOPING, Urgency.CRITICAL,
                ["Christiano2017-RLHF", "Rafailov2023-DPO"],
                ["scalable_oversight"],
                ["RLHF", "DPO", "Constitutional AI", "RLAIF"],
                ["How to handle distributional shift?",
                 "Can we avoid reward hacking at scale?"],
                0.75
            ),
            ResearchProblem(
                "goal_misgeneralization", "Goal Misgeneralization",
                "Models pursuing unintended goals that correlate with "
                "training objective",
                Maturity.EMERGING, Urgency.CRITICAL,
                ["Shah2022-GoalMisgeneralization",
                 "Langosco2022-GoalMisgeneralization"],
                ["interpretability", "reward_modeling"],
                ["Causal confusion analysis", "Distribution shift testing",
                 "Mechanistic analysis of goals"],
                ["How to distinguish correlational from causal goals?",
                 "Can we detect misgeneralization before deployment?"],
                0.90
            ),
            ResearchProblem(
                "deceptive_alignment", "Deceptive Alignment",
                "Models that appear aligned during training but "
                "pursue different goals at deployment",
                Maturity.NASCENT, Urgency.HIGH,
                ["Hubinger2019-Risks", "Ngo2022-AlignmentProblem"],
                ["interpretability", "goal_misgeneralization"],
                ["Consistency testing", "Honeypot evaluations",
                 "Mechanistic detection"],
                ["Is deceptive alignment likely to emerge?",
                 "Can we create reliable detection methods?"],
                0.95
            ),
            ResearchProblem(
                "corrigibility", "Corrigibility",
                "Ensuring AI systems allow themselves to be corrected",
                Maturity.NASCENT, Urgency.MEDIUM,
                ["Soares2015-Corrigibility", "Hadfield-Menell2017-CIRL"],
                ["reward_modeling", "goal_misgeneralization"],
                ["CIRL", "Utility indifference",
                 "Shutdown problem formalization"],
                ["Is corrigibility stable under self-improvement?",
                 "How to formalize 'human values' for correction?"],
                0.92
            ),
            ResearchProblem(
                "multi_agent_safety", "Multi-Agent Safety",
                "Safety in systems with multiple interacting AI agents",
                Maturity.NASCENT, Urgency.MEDIUM,
                ["Dafoe2020-CooperativeAI"],
                ["scalable_oversight", "robustness"],
                ["Game-theoretic frameworks", "Social choice theory",
                 "Mechanism design"],
                ["How do safety properties compose?",
                 "Can competitive dynamics undermine safety?"],
                0.88
            ),
        ]
        for p in problems:
            self.problems[p.id] = p

    def get_problem(self, problem_id: str) -> ResearchProblem:
        return self.problems[problem_id]

    def find_gaps(self) -> list[ResearchGap]:
        """Identify research gaps based on maturity and dependencies."""
        gaps = []
        for pid, problem in self.problems.items():
            # Gap: high urgency but low maturity
            if (problem.urgency.value >= 3 and
                    problem.maturity in (Maturity.NASCENT, Maturity.EMERGING)):
                gaps.append(ResearchGap(
                    pid,
                    f"{problem.name} is {problem.maturity.value} "
                    f"but {problem.urgency.name} urgency",
                    problem.estimated_difficulty,
                    problem.approaches[0] if problem.approaches else "Unknown"
                ))

            # Gap: dependencies not yet established
            for dep_id in problem.dependencies:
                dep = self.problems.get(dep_id)
                if dep and dep.maturity in (Maturity.NASCENT,
                                            Maturity.EMERGING):
                    gaps.append(ResearchGap(
                        pid,
                        f"{problem.name} depends on {dep.name} "
                        f"(currently {dep.maturity.value})",
                        0.8,
                        f"Advance {dep.name} research"
                    ))
        return gaps

    def dependency_order(self) -> list[str]:
        """Topological sort of research problems by dependencies."""
        visited = set()
        order = []

        def visit(pid):
            if pid in visited:
                return
            visited.add(pid)
            problem = self.problems.get(pid)
            if problem:
                for dep in problem.dependencies:
                    visit(dep)
            order.append(pid)

        for pid in self.problems:
            visit(pid)
        return order

    def maturity_summary(self) -> dict[str, list[str]]:
        summary = {}
        for pid, problem in self.problems.items():
            mat = problem.maturity.value
            if mat not in summary:
                summary[mat] = []
            summary[mat].append(problem.name)
        return summary

    def urgency_ranking(self) -> list[tuple[str, ResearchProblem]]:
        ranked = sorted(self.problems.items(),
                        key=lambda x: (-x[1].urgency.value,
                                       -x[1].estimated_difficulty))
        return ranked


def render_landscape(landscape: ResearchLandscape) -> str:
    lines = [
        "=" * 60,
        "AI SAFETY RESEARCH LANDSCAPE",
        "=" * 60, "",
    ]

    # Urgency ranking
    lines.append("PRIORITY RANKING (by urgency and difficulty):\n")
    for pid, problem in landscape.urgency_ranking():
        bar = "#" * int(problem.estimated_difficulty * 20)
        lines.append(
            f"  [{problem.urgency.name:>8}] {problem.name:<30} "
            f"maturity={problem.maturity.value:<12} "
            f"difficulty={bar}")

    # Dependency graph
    lines.extend(["", "DEPENDENCY ORDER (build from foundations):\n"])
    for i, pid in enumerate(landscape.dependency_order()):
        problem = landscape.problems[pid]
        deps = ", ".join(problem.dependencies) or "(none)"
        lines.append(f"  {i+1}. {problem.name} <- [{deps}]")

    # Maturity summary
    lines.extend(["", "MATURITY DISTRIBUTION:\n"])
    for mat, names in sorted(landscape.maturity_summary().items()):
        lines.append(f"  {mat:<12} ({len(names)}): {', '.join(names)}")

    # Research gaps
    lines.extend(["", "IDENTIFIED RESEARCH GAPS:\n"])
    gaps = landscape.find_gaps()
    for gap in gaps:
        lines.append(f"  [GAP] {gap.gap_description}")
        lines.append(f"        Importance: {gap.importance:.0%} | "
                     f"Direction: {gap.suggested_direction}")

    # Open questions
    lines.extend(["", "KEY OPEN QUESTIONS:\n"])
    for pid, problem in landscape.urgency_ranking()[:5]:
        lines.append(f"  {problem.name}:")
        for q in problem.open_questions:
            lines.append(f"    ? {q}")

    return "\n".join(lines)


if __name__ == "__main__":
    landscape = ResearchLandscape()
    print(render_landscape(landscape))
