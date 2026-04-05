"""
13. AI Regulation and Governance

Implements a practical AI governance toolkit: EU AI Act risk classification,
Model Card generation, GDPR Article 22 compliance checks, and a NIST AI
RMF-aligned impact assessment -- all as structured Python data pipelines.

Covered topics:
    - EU AI Act risk-level classifier for AI applications
    - GDPR Article 22 automated decision-making compliance checker
    - NIST AI RMF four-function assessment (Govern, Map, Measure, Manage)
    - Model Card generator following Mitchell et al. (2019) template
    - Datasheet for Datasets generator following Gebru et al. (2021)
    - Regulatory compliance dashboard

Related to: L13 - AI Regulation and Governance

Requirements:
    pip install numpy matplotlib
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


# ====== Section 1: EU AI Act Risk Classification ======

class RiskLevel(Enum):
    """EU AI Act risk categories."""
    UNACCEPTABLE = "unacceptable"
    HIGH = "high"
    LIMITED = "limited"
    MINIMAL = "minimal"


@dataclass
class AIApplication:
    """Description of an AI application for risk assessment."""
    name: str
    domain: str
    description: str
    uses_biometric: bool = False
    affects_fundamental_rights: bool = False
    is_safety_critical: bool = False
    interacts_with_humans: bool = True
    makes_autonomous_decisions: bool = False
    processes_personal_data: bool = False
    target_vulnerable_groups: bool = False


def classify_eu_ai_act_risk(app: AIApplication) -> dict:
    """Classify an AI application under the EU AI Act risk framework.

    The EU AI Act (2024) establishes four risk tiers:
      - Unacceptable: Banned practices (social scoring, real-time
        remote biometric identification in public for law enforcement)
      - High: Requires conformity assessment, human oversight, and
        documentation (credit scoring, hiring, medical devices)
      - Limited: Transparency obligations (chatbots, deepfakes)
      - Minimal: No specific requirements (spam filters, games)

    This function applies the classification rules from the Act's
    Annexes I-III to determine the applicable risk level.

    Args:
        app: AIApplication description.

    Returns:
        Dictionary with risk level, obligations, and rationale.
    """
    obligations = []
    rationale = []

    # Check for unacceptable risk (Article 5)
    if app.target_vulnerable_groups and app.makes_autonomous_decisions:
        return {
            "risk_level": RiskLevel.UNACCEPTABLE,
            "rationale": ["Exploitative system targeting vulnerable groups "
                          "with autonomous decisions (Article 5(1)(b))"],
            "obligations": ["PROHIBITED -- system may not be deployed"],
        }

    # Check for high risk (Annex III)
    high_risk_domains = {
        "healthcare", "credit_scoring", "hiring", "education",
        "law_enforcement", "immigration", "critical_infrastructure",
    }

    is_high_risk = False

    if app.domain in high_risk_domains:
        is_high_risk = True
        rationale.append(f"Domain '{app.domain}' listed in Annex III")

    if app.is_safety_critical:
        is_high_risk = True
        rationale.append("Safety-critical application (Annex I)")

    if app.affects_fundamental_rights and app.makes_autonomous_decisions:
        is_high_risk = True
        rationale.append("Affects fundamental rights with autonomous decisions")

    if is_high_risk:
        obligations = [
            "Risk management system (Article 9)",
            "Data governance and quality (Article 10)",
            "Technical documentation (Article 11)",
            "Record-keeping and logging (Article 12)",
            "Transparency to users (Article 13)",
            "Human oversight measures (Article 14)",
            "Accuracy, robustness, cybersecurity (Article 15)",
            "Conformity assessment before deployment",
            "Registration in EU database",
        ]
        return {
            "risk_level": RiskLevel.HIGH,
            "rationale": rationale,
            "obligations": obligations,
        }

    # Check for limited risk (Article 52)
    if app.interacts_with_humans and not app.makes_autonomous_decisions:
        return {
            "risk_level": RiskLevel.LIMITED,
            "rationale": ["Human-facing AI system requires transparency"],
            "obligations": [
                "Disclose AI nature to users (Article 52(1))",
                "Label AI-generated content if applicable (Article 52(3))",
            ],
        }

    # Default: minimal risk
    return {
        "risk_level": RiskLevel.MINIMAL,
        "rationale": ["No specific risk indicators identified"],
        "obligations": ["Voluntary codes of conduct recommended"],
    }


# ====== Section 2: GDPR Article 22 Compliance ======

@dataclass
class DecisionProcess:
    """Description of an automated decision-making process."""
    name: str
    is_solely_automated: bool
    produces_legal_effects: bool
    has_human_review: bool
    explanation_available: bool
    contest_mechanism: bool
    processes_special_categories: bool = False
    has_explicit_consent: bool = False
    has_dpia: bool = False  # Data Protection Impact Assessment


def check_gdpr_article22(process: DecisionProcess) -> dict:
    """Check compliance with GDPR Article 22.

    Article 22 restricts "solely automated decision-making, including
    profiling, which produces legal effects concerning [the data subject]
    or similarly significantly affects him or her."

    Exceptions: explicit consent, contractual necessity, or Union/Member
    State law with safeguards.

    Args:
        process: Description of the automated decision process.

    Returns:
        Dictionary with compliance status, issues, and recommendations.
    """
    issues = []
    recommendations = []
    is_compliant = True

    # Check if Article 22 applies
    if process.is_solely_automated and process.produces_legal_effects:
        # Article 22(1) applies -- generally prohibited
        if not process.has_explicit_consent:
            issues.append(
                "Solely automated decisions with legal effects require "
                "explicit consent or legal basis (Article 22(2))"
            )
            is_compliant = False

        # Article 22(3) safeguards
        if not process.has_human_review:
            issues.append(
                "Right to obtain human intervention not implemented "
                "(Article 22(3))"
            )
            recommendations.append("Implement human review mechanism")
            is_compliant = False

        if not process.explanation_available:
            issues.append(
                "Right to an explanation not fulfilled "
                "(Articles 13(2)(f), 14(2)(g), 15(1)(h))"
            )
            recommendations.append(
                "Implement model explanation capability (e.g., SHAP/LIME)"
            )
            is_compliant = False

        if not process.contest_mechanism:
            issues.append(
                "Right to contest the decision not available "
                "(Article 22(3))"
            )
            recommendations.append("Implement appeals/contest process")
            is_compliant = False

        # Special categories (Article 9)
        if process.processes_special_categories and not process.has_explicit_consent:
            issues.append(
                "Processing special category data in automated decisions "
                "requires explicit consent (Article 22(4))"
            )
            is_compliant = False

        # DPIA requirement
        if not process.has_dpia:
            recommendations.append(
                "Conduct Data Protection Impact Assessment (Article 35)"
            )

    return {
        "article_22_applies": (
            process.is_solely_automated and process.produces_legal_effects
        ),
        "is_compliant": is_compliant,
        "issues": issues,
        "recommendations": recommendations,
    }


# ====== Section 3: NIST AI RMF Assessment ======

def nist_ai_rmf_assessment(
    app_name: str,
    scores: dict[str, dict[str, float]],
) -> dict:
    """Perform a NIST AI Risk Management Framework assessment.

    The NIST AI RMF (2023) defines four core functions:
      1. GOVERN: Organizational policies and accountability
      2. MAP: Context and risk identification
      3. MEASURE: Quantitative risk assessment
      4. MANAGE: Risk treatment and monitoring

    Each function has subcategories scored 0-1 (0 = not addressed,
    1 = fully implemented).

    Args:
        app_name: Name of the AI application.
        scores: Nested dict {function: {subcategory: score}}.

    Returns:
        Assessment report with per-function scores and gaps.
    """
    report = {"application": app_name, "functions": {}}

    for function_name, subcategories in scores.items():
        values = list(subcategories.values())
        avg_score = np.mean(values)
        gaps = {k: v for k, v in subcategories.items() if v < 0.5}

        report["functions"][function_name] = {
            "average_score": float(avg_score),
            "subcategories": subcategories,
            "gaps": gaps,
            "maturity": (
                "Advanced" if avg_score >= 0.8
                else "Intermediate" if avg_score >= 0.5
                else "Initial"
            ),
        }

    # Overall maturity
    all_scores = []
    for func in report["functions"].values():
        all_scores.append(func["average_score"])
    report["overall_score"] = float(np.mean(all_scores))
    report["overall_maturity"] = (
        "Advanced" if report["overall_score"] >= 0.8
        else "Intermediate" if report["overall_score"] >= 0.5
        else "Initial"
    )

    return report


# ====== Section 4: Model Card Generator ======

def generate_model_card(
    model_name: str,
    model_type: str,
    metrics: dict[str, float],
    intended_use: str,
    limitations: list[str],
    fairness_metrics: dict[str, float] | None = None,
) -> str:
    """Generate a Model Card following Mitchell et al. (2019).

    Model Cards standardize ML model documentation, covering intended
    use, performance metrics, fairness analysis, and limitations. They
    are increasingly required by regulations (EU AI Act Article 11).

    Args:
        model_name: Human-readable model name.
        model_type: Architecture type.
        metrics: Performance metrics dict.
        intended_use: Description of intended use.
        limitations: Known limitations.
        fairness_metrics: Optional fairness metrics.

    Returns:
        Formatted Model Card as a string.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d")

    card = f"""
{'=' * 60}
MODEL CARD: {model_name}
{'=' * 60}

Generated: {timestamp}

1. MODEL DETAILS
   - Name: {model_name}
   - Type: {model_type}
   - Version: 1.0

2. INTENDED USE
   - Primary: {intended_use}
   - Out-of-scope: Uses not validated by testing

3. PERFORMANCE METRICS"""

    for metric_name, value in metrics.items():
        card += f"\n   - {metric_name}: {value:.4f}"

    if fairness_metrics:
        card += "\n\n4. FAIRNESS ANALYSIS"
        for metric_name, value in fairness_metrics.items():
            card += f"\n   - {metric_name}: {value:.4f}"

    card += "\n\n5. LIMITATIONS"
    for i, limitation in enumerate(limitations, 1):
        card += f"\n   {i}. {limitation}"

    card += f"\n\n{'=' * 60}\n"
    return card


# ====== Section 5: Visualization ======

def visualize_governance(
    risk_results: list[dict],
    nist_report: dict,
    gdpr_results: list[dict],
    save_path: str = "ai_governance.png",
) -> None:
    """Four-panel governance dashboard visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Panel 1: EU AI Act Risk Classification ---
    ax1 = axes[0, 0]
    risk_colors = {
        RiskLevel.UNACCEPTABLE: "#e74c3c",
        RiskLevel.HIGH: "#f39c12",
        RiskLevel.LIMITED: "#3498db",
        RiskLevel.MINIMAL: "#2ecc71",
    }
    names = [r["name"] for r in risk_results]
    levels = [r["risk_level"].value for r in risk_results]
    colors = [risk_colors[r["risk_level"]] for r in risk_results]
    y_pos = range(len(names))
    ax1.barh(y_pos, [1] * len(names), color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=9)
    ax1.set_xticks([])
    for i, level in enumerate(levels):
        ax1.text(0.5, i, level.upper(), ha="center", va="center",
                 fontsize=10, fontweight="bold", color="white")
    ax1.set_title("EU AI Act Risk Classification")

    # --- Panel 2: NIST AI RMF Scores ---
    ax2 = axes[0, 1]
    functions = list(nist_report["functions"].keys())
    scores = [nist_report["functions"][f]["average_score"] for f in functions]
    bar_colors = ["#2ecc71" if s >= 0.8 else "#f39c12" if s >= 0.5 else "#e74c3c"
                  for s in scores]
    ax2.bar(functions, scores, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax2.set_ylim(0, 1.1)
    ax2.axhline(y=0.8, color="green", linestyle="--", alpha=0.5, label="Advanced")
    ax2.axhline(y=0.5, color="orange", linestyle="--", alpha=0.5, label="Intermediate")
    ax2.set_ylabel("Score")
    ax2.set_title(f"NIST AI RMF Assessment\n(Overall: {nist_report['overall_maturity']})")
    ax2.legend(fontsize=8)

    # --- Panel 3: GDPR Compliance ---
    ax3 = axes[1, 0]
    gdpr_names = [r["name"] for r in gdpr_results]
    compliant = [1 if r["compliant"] else 0 for r in gdpr_results]
    n_issues = [r["n_issues"] for r in gdpr_results]
    x_pos = range(len(gdpr_names))
    ax3.bar(x_pos, n_issues,
            color=["#2ecc71" if c else "#e74c3c" for c in compliant],
            edgecolor="black", linewidth=0.5)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(gdpr_names, rotation=20, fontsize=9)
    ax3.set_ylabel("Number of Issues")
    ax3.set_title("GDPR Article 22 Compliance\n(green = compliant, red = issues)")

    # --- Panel 4: Obligations Count ---
    ax4 = axes[1, 1]
    obligation_counts = [len(r.get("obligations", [])) for r in risk_results]
    ax4.bar(names, obligation_counts,
            color=[risk_colors[r["risk_level"]] for r in risk_results],
            edgecolor="black", linewidth=0.5)
    ax4.set_ylabel("Number of Obligations")
    ax4.set_title("Regulatory Obligations per Application")
    ax4.tick_params(axis="x", rotation=20, labelsize=9)

    plt.suptitle("AI Governance Dashboard", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Figure saved to: {save_path}")
    plt.close()


# ====== Section 6: Main Pipeline ======

def main() -> None:
    """Run AI governance and regulation compliance assessment."""
    print("=" * 65)
    print("  AI Regulation and Governance")
    print("  EU AI Act | GDPR Art.22 | NIST AI RMF | Model Cards")
    print("=" * 65)

    # --- Step 1: EU AI Act Risk Classification ---
    print("\n[1] EU AI Act Risk Classification")
    print("-" * 50)

    applications = [
        AIApplication("Credit Scorer", "credit_scoring",
                       "Automated credit risk assessment",
                       affects_fundamental_rights=True,
                       makes_autonomous_decisions=True,
                       processes_personal_data=True),
        AIApplication("Medical Diagnosis", "healthcare",
                       "AI-assisted radiology screening",
                       is_safety_critical=True,
                       interacts_with_humans=True),
        AIApplication("Customer Chatbot", "customer_service",
                       "Conversational AI for support queries",
                       interacts_with_humans=True),
        AIApplication("Spam Filter", "email",
                       "Email spam classification",
                       interacts_with_humans=False),
    ]

    risk_results = []
    for app in applications:
        result = classify_eu_ai_act_risk(app)
        result["name"] = app.name
        risk_results.append(result)
        print(f"\n  {app.name}:")
        print(f"    Risk Level: {result['risk_level'].value.upper()}")
        print(f"    Rationale: {'; '.join(result['rationale'])}")
        print(f"    Obligations: {len(result['obligations'])}")

    # --- Step 2: GDPR Article 22 ---
    print("\n\n[2] GDPR Article 22 Compliance")
    print("-" * 50)

    processes = [
        DecisionProcess("Loan Approval", is_solely_automated=True,
                         produces_legal_effects=True, has_human_review=True,
                         explanation_available=True, contest_mechanism=True,
                         has_explicit_consent=True, has_dpia=True),
        DecisionProcess("Resume Screener", is_solely_automated=True,
                         produces_legal_effects=True, has_human_review=False,
                         explanation_available=False, contest_mechanism=False),
        DecisionProcess("Product Recommender", is_solely_automated=True,
                         produces_legal_effects=False, has_human_review=False,
                         explanation_available=False, contest_mechanism=False),
    ]

    gdpr_results = []
    for process in processes:
        result = check_gdpr_article22(process)
        gdpr_results.append({
            "name": process.name,
            "compliant": result["is_compliant"],
            "n_issues": len(result["issues"]),
        })
        print(f"\n  {process.name}:")
        print(f"    Article 22 applies: {result['article_22_applies']}")
        print(f"    Compliant: {result['is_compliant']}")
        if result["issues"]:
            for issue in result["issues"]:
                print(f"    Issue: {issue}")
        if result["recommendations"]:
            for rec in result["recommendations"]:
                print(f"    Recommendation: {rec}")

    # --- Step 3: NIST AI RMF ---
    print("\n\n[3] NIST AI RMF Assessment")
    print("-" * 50)

    nist_scores = {
        "GOVERN": {
            "policies": 0.8,
            "accountability": 0.7,
            "culture": 0.6,
            "stakeholder_engagement": 0.5,
        },
        "MAP": {
            "context_identification": 0.9,
            "risk_identification": 0.7,
            "impact_assessment": 0.6,
        },
        "MEASURE": {
            "performance_metrics": 0.9,
            "fairness_metrics": 0.4,
            "robustness_testing": 0.5,
            "explainability_assessment": 0.3,
        },
        "MANAGE": {
            "risk_treatment": 0.6,
            "monitoring": 0.5,
            "incident_response": 0.4,
            "documentation": 0.7,
        },
    }

    nist_report = nist_ai_rmf_assessment("Credit Scoring System", nist_scores)
    print(f"  Overall maturity: {nist_report['overall_maturity']} "
          f"({nist_report['overall_score']:.2f})")
    for func_name, func_data in nist_report["functions"].items():
        gaps = list(func_data["gaps"].keys())
        gap_str = f" -- gaps: {', '.join(gaps)}" if gaps else ""
        print(f"  {func_name}: {func_data['average_score']:.2f} "
              f"({func_data['maturity']}){gap_str}")

    # --- Step 4: Model Card ---
    print("\n\n[4] Model Card Generation")
    print("-" * 50)

    card = generate_model_card(
        model_name="CreditRisk-LR-v1",
        model_type="Logistic Regression",
        metrics={"accuracy": 0.87, "auc_roc": 0.92, "f1_score": 0.85},
        intended_use="Automated credit risk scoring for consumer loans",
        limitations=[
            "Trained on US consumer data only (2019-2023)",
            "Not validated for business/commercial lending",
            "Demographic parity difference exceeds 5% threshold",
            "Requires human review for borderline cases (0.4-0.6 score)",
        ],
        fairness_metrics={
            "demographic_parity_diff": 0.08,
            "equalized_odds_diff": 0.05,
            "calibration_diff": 0.03,
        },
    )
    print(card)

    # --- Step 5: Visualization ---
    print("[5] Generating Governance Dashboard")
    print("-" * 50)

    visualize_governance(risk_results, nist_report, gdpr_results)

    # --- Summary ---
    print("\n" + "=" * 65)
    print("  Summary")
    print("=" * 65)
    print("""
  Key findings:
    1. EU AI Act classifies applications into 4 risk tiers. High-risk
       systems (credit scoring, healthcare) require extensive compliance.
    2. GDPR Article 22 restricts solely automated decisions with legal
       effects -- human review, explanations, and contest mechanisms
       are mandatory safeguards.
    3. NIST AI RMF provides a structured framework for organizational
       AI risk management across Govern, Map, Measure, and Manage.
    4. Model Cards and Datasheets standardize documentation and are
       increasingly required by regulation.
    5. Governance is not just technical -- it requires organizational
       policies, accountability structures, and stakeholder engagement.
    """)


if __name__ == "__main__":
    main()
