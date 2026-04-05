# Exercise: Lesson 16 — Open Problems
# Complete the TODO items below.
#
# Run: python 16_open_problems.py


def write_research_proposal(problem_area: str) -> dict:
    """Write a structured research proposal for an AI safety open problem.

    Args:
        problem_area: One of "superalignment", "interpretability",
                      "robustness", "value_learning", "corrigibility",
                      "mesa_optimization", "multi_agent_safety".

    Returns:
        dict with:
            - "title": proposal title
            - "problem_statement": clear description of the problem
            - "motivation": why this problem matters
            - "approach": proposed research approach (2-3 paragraphs)
            - "milestones": list of dicts with "milestone", "timeline",
              "success_criteria"
            - "risks": list of str (what could go wrong)
            - "expected_impact": str
    """
    # TODO: Define the problem statement based on the problem_area.

    # TODO: Articulate the motivation connecting to AI safety.

    # TODO: Propose a concrete research approach with methodology.

    # TODO: Define 3-5 milestones with timelines and success criteria.

    # TODO: Identify research risks and expected impact.
    pass


def analyze_open_problem(problem: dict) -> dict:
    """Analyze an AI safety open problem's current state and difficulty.

    Args:
        problem: dict with:
            - "name": problem name
            - "description": problem description
            - "current_approaches": list of str
            - "known_limitations": list of str
            - "years_studied": int

    Returns:
        dict with:
            - "difficulty_estimate": one of "tractable", "hard",
              "very_hard", "possibly_impossible"
            - "progress_assessment": float (0-1, estimated progress)
            - "bottlenecks": list of str
            - "promising_directions": list of str
            - "required_breakthroughs": list of str
    """
    # TODO: Assess difficulty based on years studied vs progress made,
    # and the nature of known limitations.

    # TODO: Identify key bottlenecks preventing progress.

    # TODO: Suggest promising research directions.

    # TODO: List breakthroughs needed (theoretical or empirical).
    pass


def compare_safety_paradigms(paradigms: list[dict]) -> dict:
    """Compare different AI safety research paradigms.

    Args:
        paradigms: List of dicts with:
            - "name": paradigm name (e.g., "alignment via RLHF",
              "mechanistic interpretability", "formal verification")
            - "assumptions": list of str
            - "strengths": list of str
            - "weaknesses": list of str
            - "scalability": one of "proven", "uncertain", "unlikely"

    Returns:
        dict with:
            - "comparison_matrix": dict mapping paradigm pairs to
              "complementary", "competing", or "orthogonal"
            - "combined_coverage": list of str (problems covered by any paradigm)
            - "uncovered_gaps": list of str (problems no paradigm addresses)
            - "recommended_portfolio": list of paradigm names to pursue
    """
    # TODO: Build a pairwise comparison matrix.

    # TODO: Determine which safety problems each paradigm covers.

    # TODO: Identify gaps not addressed by any paradigm.

    # TODO: Recommend a portfolio of paradigms for maximum coverage.
    pass


def forecast_timeline(factors: list[dict]) -> dict:
    """Forecast when key AI safety milestones might be achieved.

    Args:
        factors: List of dicts with:
            - "milestone": str
            - "current_progress": float (0-1)
            - "annual_progress_rate": float
            - "dependencies": list of other milestone names
            - "uncertainty": float (0-1)

    Returns:
        dict with:
            - "forecasts": list of dicts with "milestone",
              "estimated_years", "confidence_interval" (tuple),
              "blocking_dependencies" (list)
            - "critical_path": list of milestones in dependency order
    """
    # TODO: For each milestone, estimate years to completion based on
    # current progress and annual rate.

    # TODO: Account for dependencies (a milestone cannot complete
    # before its dependencies).

    # TODO: Compute confidence intervals based on uncertainty.

    # TODO: Identify the critical path through the dependency graph.
    pass


if __name__ == "__main__":
    # Test research proposal
    proposal = write_research_proposal("superalignment")
    print(f"Research proposal: {proposal}")

    # Test problem analysis
    problem = {
        "name": "Scalable Oversight",
        "description": "How to supervise AI systems smarter than the supervisor",
        "current_approaches": ["debate", "recursive reward modeling",
                                "market-based approaches"],
        "known_limitations": ["debate may not converge", "decomposition loses context"],
        "years_studied": 6,
    }
    analysis = analyze_open_problem(problem)
    print(f"\nProblem analysis: {analysis}")

    # Test paradigm comparison
    paradigms = [
        {"name": "RLHF", "assumptions": ["human feedback is reliable"],
         "strengths": ["practical", "scalable"],
         "weaknesses": ["reward hacking", "sycophancy"],
         "scalability": "proven"},
        {"name": "Formal Verification", "assumptions": ["specs can be formalized"],
         "strengths": ["mathematical guarantees"],
         "weaknesses": ["doesn't scale to large models"],
         "scalability": "unlikely"},
    ]
    comparison = compare_safety_paradigms(paradigms)
    print(f"\nParadigm comparison: {comparison}")

    # Test timeline forecast
    factors = [
        {"milestone": "interpretable_transformers", "current_progress": 0.2,
         "annual_progress_rate": 0.08, "dependencies": [], "uncertainty": 0.5},
        {"milestone": "scalable_alignment", "current_progress": 0.1,
         "annual_progress_rate": 0.05,
         "dependencies": ["interpretable_transformers"], "uncertainty": 0.7},
    ]
    forecast = forecast_timeline(factors)
    print(f"\nTimeline forecast: {forecast}")
