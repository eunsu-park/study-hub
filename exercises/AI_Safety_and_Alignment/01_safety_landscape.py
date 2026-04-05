# Exercise: Lesson 01 — Safety Landscape
# Complete the TODO items below.
#
# Run: python 01_safety_landscape.py


def classify_ai_risk(scenario: str) -> dict:
    """Classify an AI risk scenario into category and severity.

    Args:
        scenario: A text description of an AI risk scenario.

    Returns:
        dict with keys:
            - "category": one of "misuse", "accident", "structural", "existential"
            - "severity": one of "low", "medium", "high", "critical"
            - "reasoning": brief explanation of the classification
    """
    # TODO: Parse the scenario description and determine its risk category.
    # Consider whether the risk stems from intentional misuse, accidental
    # failure, structural/systemic issues, or existential-level threats.

    # TODO: Assess severity based on scope of impact, reversibility,
    # and likelihood of occurrence.

    # TODO: Return the classification dict with reasoning.
    pass


def map_safety_organizations(orgs: list[dict]) -> dict:
    """Map AI safety organizations by focus area and approach.

    Args:
        orgs: List of dicts with keys "name", "description", "founded_year".

    Returns:
        dict mapping focus areas to lists of organization names.
        Focus areas: "technical_alignment", "governance", "policy",
                     "field_building", "industry_safety"
    """
    # TODO: Define keyword patterns for each focus area.

    # TODO: Classify each organization into one or more focus areas
    # based on its description.

    # TODO: Return the mapping from focus areas to org names.
    pass


def build_risk_taxonomy(risks: list[str]) -> dict:
    """Organize a flat list of AI risks into a hierarchical taxonomy.

    Args:
        risks: List of risk description strings.

    Returns:
        Nested dict representing a taxonomy tree with categories
        and subcategories, each containing relevant risks.
    """
    # TODO: Define top-level categories (e.g., "near_term", "mid_term",
    # "long_term") and subcategories within each.

    # TODO: Assign each risk to the appropriate place in the taxonomy.

    # TODO: Return the nested taxonomy dict.
    pass


def timeline_risk_analysis(capabilities: list[dict]) -> list[dict]:
    """Analyze how risks change as AI capabilities increase over time.

    Args:
        capabilities: List of dicts with "capability", "current_level" (0-10),
                      "projected_level" (0-10), "years_to_projection".

    Returns:
        List of dicts with "capability", "risk_delta", "priority_rank",
        and "mitigation_urgency" (one of "immediate", "near_term", "long_term").
    """
    # TODO: Calculate risk_delta as the difference between projected
    # and current capability levels.

    # TODO: Determine mitigation_urgency based on risk_delta and
    # years_to_projection.

    # TODO: Sort by priority_rank (highest urgency first) and return.
    pass


if __name__ == "__main__":
    # Test classify_ai_risk
    scenario = "An autonomous trading system causes a flash crash by exploiting market patterns in ways its designers did not anticipate."
    result = classify_ai_risk(scenario)
    print(f"Risk classification: {result}")

    # Test map_safety_organizations
    orgs = [
        {"name": "MIRI", "description": "Technical research on alignment theory and agent foundations", "founded_year": 2000},
        {"name": "CAIS", "description": "Policy research and governance frameworks for AI safety", "founded_year": 2019},
    ]
    mapping = map_safety_organizations(orgs)
    print(f"Org mapping: {mapping}")

    # Test build_risk_taxonomy
    risks = ["deepfake generation", "power-seeking behavior", "job displacement", "surveillance overreach"]
    taxonomy = build_risk_taxonomy(risks)
    print(f"Taxonomy: {taxonomy}")

    # Test timeline_risk_analysis
    caps = [
        {"capability": "code_generation", "current_level": 7, "projected_level": 9, "years_to_projection": 2},
        {"capability": "autonomous_research", "current_level": 3, "projected_level": 8, "years_to_projection": 5},
    ]
    analysis = timeline_risk_analysis(caps)
    print(f"Timeline analysis: {analysis}")
