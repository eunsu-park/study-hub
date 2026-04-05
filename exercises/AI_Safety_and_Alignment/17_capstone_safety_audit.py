# Exercise: Lesson 17 — Capstone Safety Audit
# Complete the TODO items below.
#
# Run: python 17_capstone_safety_audit.py


def build_audit_checklist(system_type: str,
                          risk_level: str) -> list[dict]:
    """Build a comprehensive safety audit checklist for an AI system.

    Args:
        system_type: One of "llm", "recommender", "classifier",
                     "autonomous_agent", "generative_model".
        risk_level: One of "low", "medium", "high", "critical".

    Returns:
        List of checklist item dicts with:
            - "id": item identifier (e.g., "A1.1")
            - "category": audit category (e.g., "alignment", "robustness",
              "fairness", "privacy", "transparency", "governance")
            - "requirement": what must be verified
            - "priority": one of "required", "recommended", "optional"
            - "evidence_needed": what evidence satisfies the check
            - "status": "not_started" (initial state)
    """
    # TODO: Define audit categories relevant to the system type.

    # TODO: For each category, create checklist items with requirements.
    # Higher risk_level should include more stringent requirements.

    # TODO: Assign priorities (required/recommended/optional) based on
    # risk_level and system_type.

    # TODO: Specify what evidence is needed for each item.
    pass


def evaluate_checklist_item(item: dict, evidence: dict) -> dict:
    """Evaluate a single audit checklist item against provided evidence.

    Args:
        item: A checklist item dict (from build_audit_checklist).
        evidence: dict with:
            - "documents": list of str (document names provided)
            - "test_results": dict of test_name -> pass/fail
            - "metrics": dict of metric_name -> value
            - "notes": str

    Returns:
        dict with:
            - "item_id": str
            - "status": one of "pass", "fail", "partial", "not_applicable"
            - "findings": list of str
            - "risk_exposure": str description of risk if item fails
            - "remediation": str or None
    """
    # TODO: Check if the evidence satisfies the item's requirements.

    # TODO: Determine pass/fail/partial status.

    # TODO: Document findings and any risk exposure from failures.

    # TODO: Suggest remediation for failed or partial items.
    pass


def generate_audit_report(checklist: list[dict],
                          evaluations: list[dict],
                          system_name: str) -> dict:
    """Generate a full safety audit report from checklist evaluations.

    Args:
        checklist: The audit checklist (from build_audit_checklist).
        evaluations: List of evaluation results (from evaluate_checklist_item).
        system_name: Name of the audited system.

    Returns:
        dict with:
            - "system_name": str
            - "audit_date": str (ISO format)
            - "overall_verdict": "pass", "conditional_pass", or "fail"
            - "summary_scores": dict of category -> float (0-1)
            - "critical_findings": list of finding dicts
            - "recommendations": list of prioritized recommendations
            - "next_audit_date": str (suggested re-audit date)
    """
    # TODO: Aggregate evaluation results by category.

    # TODO: Compute per-category scores.

    # TODO: Determine overall verdict (fail if any required item fails,
    # conditional_pass if only recommended items fail).

    # TODO: Compile critical findings and prioritized recommendations.

    # TODO: Suggest next audit date based on findings severity.
    pass


def run_full_audit(system: dict) -> dict:
    """Execute a complete safety audit pipeline.

    This is the capstone exercise combining all safety concepts.

    Args:
        system: dict with:
            - "name": system name
            - "type": system type (e.g., "llm")
            - "risk_level": one of "low", "medium", "high", "critical"
            - "documentation": dict of available docs (bool flags)
            - "test_results": dict of test_name -> pass/fail
            - "metrics": dict of safety metrics
            - "deployment_info": dict with deployment context

    Returns:
        dict with the complete audit report including checklist,
        evaluations, and final report.
    """
    # TODO: Build the audit checklist based on system type and risk level.

    # TODO: Evaluate each checklist item using the system's evidence
    # (documentation, test_results, metrics).

    # TODO: Generate the audit report.

    # TODO: Return the complete audit package.
    pass


if __name__ == "__main__":
    # Test building audit checklist
    checklist = build_audit_checklist("llm", "high")
    print(f"Checklist ({len(checklist) if checklist else 0} items):")
    if checklist:
        for item in checklist[:3]:
            print(f"  {item['id']}: {item['requirement']}")

    # Test evaluating a checklist item
    if checklist:
        evidence = {
            "documents": ["model_card", "safety_eval_report"],
            "test_results": {"toxicity_test": True, "bias_test": False},
            "metrics": {"safety_score": 0.92, "bias_score": 0.65},
            "notes": "Bias testing showed disparities in gender categories",
        }
        evaluation = evaluate_checklist_item(checklist[0], evidence)
        print(f"\nEvaluation: {evaluation}")

    # Test full audit
    system = {
        "name": "AssistantLLM-v3",
        "type": "llm",
        "risk_level": "high",
        "documentation": {
            "has_model_card": True,
            "has_safety_eval": True,
            "has_bias_audit": True,
            "has_data_sheet": False,
            "has_incident_plan": True,
        },
        "test_results": {
            "toxicity_test": True,
            "bias_test": False,
            "robustness_test": True,
            "privacy_test": True,
            "jailbreak_test": False,
        },
        "metrics": {
            "safety_score": 0.92,
            "bias_score": 0.65,
            "robustness_score": 0.88,
            "toxicity_rate": 0.02,
        },
        "deployment_info": {
            "scale": "global",
            "user_base": "consumer",
            "has_monitoring": True,
            "has_rollback": True,
        },
    }
    audit = run_full_audit(system)
    print(f"\nFull audit result: {audit}")
