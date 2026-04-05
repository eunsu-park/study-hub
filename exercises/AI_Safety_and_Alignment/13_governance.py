# Exercise: Lesson 13 — Governance
# Complete the TODO items below.
#
# Run: python 13_governance.py


def risk_assessment(system: dict) -> dict:
    """Perform a structured risk assessment for an AI system.

    Args:
        system: dict with:
            - "name": system name
            - "capabilities": list of str
            - "deployment_context": str (e.g., "healthcare", "finance", "general")
            - "user_base": str (e.g., "enterprise", "consumer", "government")
            - "autonomy_level": int (1-5, how autonomous the system is)

    Returns:
        dict with:
            - "risk_level": one of "low", "medium", "high", "critical"
            - "risk_factors": list of dicts with "factor", "score" (1-10),
              "mitigation" (str)
            - "regulatory_category": str (e.g., "high-risk AI" per EU AI Act)
            - "required_assessments": list of str
    """
    # TODO: Evaluate risk factors: autonomy, impact scope, vulnerability
    # of user base, reversibility of decisions, data sensitivity.

    # TODO: Score each factor on 1-10 scale and determine overall risk level.

    # TODO: Map to regulatory categories (e.g., EU AI Act tiers:
    # unacceptable, high-risk, limited risk, minimal risk).

    # TODO: List required assessments based on risk level.
    pass


def compliance_check(system_docs: dict, framework: str) -> dict:
    """Check an AI system's compliance against a governance framework.

    Args:
        system_docs: dict with:
            - "has_impact_assessment": bool
            - "has_model_card": bool
            - "has_data_sheet": bool
            - "has_audit_log": bool
            - "has_human_oversight": bool
            - "has_bias_testing": bool
            - "has_incident_response": bool
            - "transparency_level": str ("none", "partial", "full")
        framework: One of "eu_ai_act", "nist_rmf", "iso_42001",
                   "oecd_principles".

    Returns:
        dict with:
            - "compliant": bool
            - "score": float (0-1)
            - "gaps": list of str (missing requirements)
            - "recommendations": list of str
    """
    # TODO: Define requirements for each framework.

    # TODO: Check system_docs against framework requirements.

    # TODO: Identify gaps and provide recommendations.

    # TODO: Return the compliance assessment.
    pass


def design_governance_board(org_type: str, ai_risk_level: str) -> dict:
    """Design the structure for an AI governance board.

    Args:
        org_type: One of "startup", "enterprise", "government", "research".
        ai_risk_level: One of "low", "medium", "high", "critical".

    Returns:
        dict with:
            - "board_size": int
            - "roles": list of dicts with "title", "responsibilities",
              "expertise_required"
            - "meeting_frequency": str
            - "escalation_process": list of escalation steps
            - "review_triggers": list of events that require board review
    """
    # TODO: Determine board size based on org_type and risk_level.

    # TODO: Define roles (e.g., ethics officer, technical lead,
    # legal counsel, domain expert, public representative).

    # TODO: Set meeting frequency and define escalation processes.

    # TODO: Define review triggers (e.g., new model deployment,
    # incident report, regulatory change).
    pass


def audit_trail_entry(action: str, actor: str, system: str,
                      details: dict) -> dict:
    """Create a structured audit trail entry for governance tracking.

    Args:
        action: What was done (e.g., "model_deployed", "risk_assessed").
        actor: Who performed the action.
        system: Which AI system is involved.
        details: Additional context as key-value pairs.

    Returns:
        dict with "timestamp" (ISO format str), "action", "actor",
        "system", "details", "hash" (integrity check hash).
    """
    # TODO: Generate an ISO format timestamp.

    # TODO: Create the audit entry with all fields.

    # TODO: Compute a simple hash of the entry for integrity verification.

    # TODO: Return the entry.
    pass


if __name__ == "__main__":
    # Test risk assessment
    system = {
        "name": "MedAssist-v2",
        "capabilities": ["diagnosis_suggestion", "drug_interaction_check"],
        "deployment_context": "healthcare",
        "user_base": "consumer",
        "autonomy_level": 3,
    }
    assessment = risk_assessment(system)
    print(f"Risk assessment: {assessment}")

    # Test compliance check
    docs = {
        "has_impact_assessment": True,
        "has_model_card": True,
        "has_data_sheet": False,
        "has_audit_log": True,
        "has_human_oversight": True,
        "has_bias_testing": False,
        "has_incident_response": False,
        "transparency_level": "partial",
    }
    compliance = compliance_check(docs, "eu_ai_act")
    print(f"\nCompliance: {compliance}")

    # Test governance board design
    board = design_governance_board("enterprise", "high")
    print(f"\nGovernance board: {board}")

    # Test audit trail
    entry = audit_trail_entry(
        action="model_deployed",
        actor="ml_ops_team",
        system="MedAssist-v2",
        details={"version": "2.1", "environment": "production"}
    )
    print(f"\nAudit entry: {entry}")
