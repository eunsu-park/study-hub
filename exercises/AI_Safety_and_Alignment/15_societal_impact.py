# Exercise: Lesson 15 — Societal Impact
# Complete the TODO items below.
#
# Run: python 15_societal_impact.py


def impact_assessment(system: dict) -> dict:
    """Conduct a societal impact assessment for an AI system.

    Args:
        system: dict with:
            - "name": system name
            - "function": what the system does
            - "deployment_scale": one of "local", "national", "global"
            - "affected_populations": list of str
            - "data_sources": list of str
            - "decision_types": list of str (what decisions it influences)

    Returns:
        dict with:
            - "impact_areas": list of dicts with "area" (e.g., "employment",
              "privacy", "equity", "environment"), "positive_impacts" (list),
              "negative_impacts" (list), "severity" (1-10)
            - "stakeholders": list of dicts with "group", "impact_type",
              "power_level" (low/medium/high)
            - "overall_assessment": str summary
            - "mitigation_priorities": list of str
    """
    # TODO: Identify impact areas based on the system's function
    # and affected populations.

    # TODO: For each area, list positive and negative impacts and
    # assign severity scores.

    # TODO: Map stakeholders and their relative power levels.

    # TODO: Prioritize mitigations starting with highest severity
    # impacts on lowest-power stakeholders.
    pass


def analyze_distributional_effects(outcomes: list[dict]) -> dict:
    """Analyze how an AI system's effects are distributed across groups.

    Args:
        outcomes: List of dicts with:
            - "group": demographic group name
            - "population_share": float (0-1)
            - "benefit_score": float (-10 to 10, negative = harm)
            - "access_level": float (0-1, ability to use the system)

    Returns:
        dict with:
            - "equity_score": float (-1 to 1, 0 = perfectly equitable)
            - "most_benefited": str (group name)
            - "most_harmed": str (group name)
            - "access_gap": float (max - min access_level)
            - "recommendations": list of str
    """
    # TODO: Calculate weighted benefit across groups.

    # TODO: Identify groups with highest and lowest benefit scores.

    # TODO: Compute equity score (e.g., negative Gini-like measure
    # of benefit distribution).

    # TODO: Analyze access gaps and generate equity recommendations.
    pass


def environmental_impact(training_compute: dict,
                         inference_load: dict) -> dict:
    """Estimate the environmental impact of an AI system.

    Args:
        training_compute: dict with "gpu_hours" (float),
                          "gpu_type" (str), "energy_source" (str).
        inference_load: dict with "requests_per_day" (int),
                        "avg_tokens_per_request" (int),
                        "gpu_type" (str).

    Returns:
        dict with:
            - "training_co2_kg": float
            - "daily_inference_co2_kg": float
            - "annual_co2_kg": float
            - "equivalent_flights": float (transatlantic flights equivalent)
            - "reduction_suggestions": list of str
    """
    # TODO: Estimate training energy (GPU hours * power per GPU).

    # TODO: Estimate daily inference energy.

    # TODO: Convert energy to CO2 based on energy source
    # (coal ~ 0.9 kg/kWh, natural gas ~ 0.4, renewable ~ 0.05).

    # TODO: Provide equivalencies and reduction suggestions.
    pass


def labor_displacement_analysis(automation_profile: dict) -> dict:
    """Analyze potential labor displacement from an AI system.

    Args:
        automation_profile: dict with:
            - "tasks_automated": list of str
            - "industry": str
            - "task_complexity": dict mapping task -> "routine"/"non-routine"
            - "current_workforce_size": int

    Returns:
        dict with:
            - "displacement_risk": float (0-1)
            - "jobs_at_risk": int (estimated)
            - "augmentation_potential": float (0-1)
            - "new_roles_created": list of str
            - "transition_recommendations": list of str
    """
    # TODO: Classify tasks by automation risk (routine tasks = higher risk).

    # TODO: Estimate displacement based on proportion of routine tasks.

    # TODO: Identify augmentation potential (tasks where AI + human > AI alone).

    # TODO: Suggest new roles and transition pathways.
    pass


if __name__ == "__main__":
    # Test impact assessment
    system = {
        "name": "ResumeScreener-AI",
        "function": "Automated resume screening for job applications",
        "deployment_scale": "national",
        "affected_populations": ["job_applicants", "hr_professionals",
                                  "minority_groups"],
        "data_sources": ["historical_hiring_data", "resume_database"],
        "decision_types": ["hiring_shortlist", "candidate_ranking"],
    }
    assessment = impact_assessment(system)
    print(f"Impact assessment: {assessment}")

    # Test distributional effects
    outcomes = [
        {"group": "urban_professionals", "population_share": 0.3,
         "benefit_score": 7.0, "access_level": 0.9},
        {"group": "rural_workers", "population_share": 0.4,
         "benefit_score": -2.0, "access_level": 0.3},
        {"group": "elderly", "population_share": 0.3,
         "benefit_score": 1.0, "access_level": 0.2},
    ]
    effects = analyze_distributional_effects(outcomes)
    print(f"\nDistributional effects: {effects}")

    # Test environmental impact
    training = {"gpu_hours": 10000, "gpu_type": "A100",
                "energy_source": "natural_gas"}
    inference = {"requests_per_day": 100000,
                 "avg_tokens_per_request": 500, "gpu_type": "A100"}
    env = environmental_impact(training, inference)
    print(f"\nEnvironmental impact: {env}")

    # Test labor displacement
    profile = {
        "tasks_automated": ["resume_parsing", "keyword_matching",
                            "initial_ranking"],
        "industry": "human_resources",
        "task_complexity": {"resume_parsing": "routine",
                           "keyword_matching": "routine",
                           "initial_ranking": "non-routine"},
        "current_workforce_size": 5000,
    }
    labor = labor_displacement_analysis(profile)
    print(f"\nLabor displacement: {labor}")
