# Exercise: Lesson 14 — Responsible Deployment
# Complete the TODO items below.
#
# Run: python 14_responsible_deployment.py


def generate_model_card(model_info: dict) -> dict:
    """Generate a model card following standard format.

    Args:
        model_info: dict with:
            - "name": model name
            - "version": version string
            - "task": primary task
            - "architecture": model architecture
            - "training_data_description": str
            - "performance_metrics": dict of metric -> score
            - "known_limitations": list of str
            - "intended_use": str
            - "out_of_scope_uses": list of str

    Returns:
        dict with structured model card sections:
            - "overview": dict
            - "intended_use": dict
            - "metrics": dict
            - "limitations": dict
            - "ethical_considerations": dict
            - "recommendations": list of str
    """
    # TODO: Build the overview section with name, version, task, architecture.

    # TODO: Structure intended use and out-of-scope sections.

    # TODO: Format performance metrics with context about evaluation data.

    # TODO: Add ethical considerations based on the task and limitations.

    # TODO: Generate recommendations for responsible use.
    pass


def build_incident_response_plan(system_type: str,
                                 risk_level: str) -> dict:
    """Build an AI incident response plan.

    Args:
        system_type: Type of AI system (e.g., "chatbot", "recommender",
                     "autonomous_vehicle", "medical_diagnostic").
        risk_level: One of "low", "medium", "high", "critical".

    Returns:
        dict with:
            - "severity_levels": list of severity level definitions
            - "response_team": list of role dicts
            - "response_procedures": dict mapping severity -> steps
            - "communication_plan": dict with internal/external templates
            - "post_incident_review": list of review steps
    """
    # TODO: Define severity levels appropriate for the system type
    # (e.g., S1-S4 with descriptions and examples).

    # TODO: Define response team roles and responsibilities.

    # TODO: Create response procedures for each severity level
    # (detection, containment, investigation, resolution, communication).

    # TODO: Add communication templates and post-incident review process.
    pass


def deployment_readiness_checklist(system: dict) -> dict:
    """Evaluate deployment readiness against a comprehensive checklist.

    Args:
        system: dict with:
            - "has_model_card": bool
            - "has_safety_eval": bool
            - "has_bias_audit": bool
            - "has_monitoring": bool
            - "has_rollback_plan": bool
            - "has_user_feedback": bool
            - "has_rate_limiting": bool
            - "has_content_filtering": bool

    Returns:
        dict with:
            - "ready": bool
            - "score": float (0-1)
            - "passed_items": list of str
            - "failed_items": list of str
            - "blockers": list of str (critical missing items)
    """
    # TODO: Define which items are critical (blockers) vs nice-to-have.

    # TODO: Check each item and categorize as passed or failed.

    # TODO: Determine if the system is deployment-ready
    # (all blockers must pass).

    # TODO: Return the readiness assessment.
    pass


def staged_rollout_plan(user_base_size: int,
                        risk_level: str) -> list[dict]:
    """Design a staged rollout plan for a new AI system.

    Args:
        user_base_size: Total target user base.
        risk_level: One of "low", "medium", "high", "critical".

    Returns:
        List of stage dicts with:
            - "stage_name": str
            - "user_percentage": float
            - "user_count": int
            - "duration_days": int
            - "success_criteria": list of str
            - "rollback_triggers": list of str
    """
    # TODO: Define stages (e.g., internal testing, alpha, beta,
    # limited GA, full GA).

    # TODO: Set user percentages and durations based on risk level
    # (higher risk = more gradual rollout).

    # TODO: Define success criteria and rollback triggers for each stage.

    # TODO: Return the rollout plan.
    pass


if __name__ == "__main__":
    # Test model card generation
    info = {
        "name": "SafeChat-v1",
        "version": "1.0.0",
        "task": "conversational AI assistant",
        "architecture": "transformer decoder",
        "training_data_description": "Web text filtered for quality and safety",
        "performance_metrics": {"accuracy": 0.92, "safety_score": 0.97},
        "known_limitations": ["May hallucinate facts", "Limited multilingual support"],
        "intended_use": "General-purpose Q&A for adults",
        "out_of_scope_uses": ["Medical advice", "Legal counsel", "Minors"],
    }
    card = generate_model_card(info)
    print(f"Model card: {card}")

    # Test incident response plan
    plan = build_incident_response_plan("chatbot", "high")
    print(f"\nIncident response plan: {plan}")

    # Test deployment readiness
    system = {
        "has_model_card": True,
        "has_safety_eval": True,
        "has_bias_audit": False,
        "has_monitoring": True,
        "has_rollback_plan": True,
        "has_user_feedback": False,
        "has_rate_limiting": True,
        "has_content_filtering": True,
    }
    readiness = deployment_readiness_checklist(system)
    print(f"\nReadiness: {readiness}")

    # Test staged rollout
    rollout = staged_rollout_plan(1_000_000, "high")
    print(f"\nRollout plan: {rollout}")
