"""
Exercises for Lesson 13: AI Regulation and Governance
Topic: Interpretable_AI

Solutions to practice problems from the lesson.
"""

import json
from datetime import datetime


# === Exercise 1: Classify AI Systems by EU AI Act Risk Level ===
# Problem: Given descriptions of AI systems, classify each into the
# appropriate EU AI Act risk tier (Unacceptable, High, Limited, Minimal)
# and explain the interpretability obligations for each.

def exercise_1():
    """Classify AI systems by EU AI Act risk level and state obligations."""

    ai_systems = [
        {
            "name": "Social Credit Scoring System",
            "description": "Government system that scores citizens based on behavior "
                           "and restricts access to services based on social score.",
            "domain": "government",
            "affects_rights": True,
            "biometric": False,
            "subliminal": False,
            "social_scoring": True,
        },
        {
            "name": "Resume Screening Tool",
            "description": "AI that ranks job applicants and filters resumes "
                           "for employment decisions.",
            "domain": "employment",
            "affects_rights": True,
            "biometric": False,
            "subliminal": False,
            "social_scoring": False,
        },
        {
            "name": "Customer Service Chatbot",
            "description": "AI chatbot that answers customer queries on a "
                           "retail website.",
            "domain": "customer_service",
            "affects_rights": False,
            "biometric": False,
            "subliminal": False,
            "social_scoring": False,
        },
        {
            "name": "Medical Diagnosis Assistant",
            "description": "AI system that assists radiologists in detecting "
                           "tumors from medical imaging.",
            "domain": "healthcare",
            "affects_rights": True,
            "biometric": False,
            "subliminal": False,
            "social_scoring": False,
        },
        {
            "name": "Spam Email Filter",
            "description": "ML model that classifies incoming emails as spam "
                           "or legitimate.",
            "domain": "email",
            "affects_rights": False,
            "biometric": False,
            "subliminal": False,
            "social_scoring": False,
        },
    ]

    high_risk_domains = {"employment", "healthcare", "education", "law_enforcement",
                         "critical_infrastructure", "migration", "justice"}

    limited_risk_keywords = {"chatbot", "deepfake", "emotion_recognition",
                             "customer_service"}

    def classify_risk(system):
        # Unacceptable risk
        if system["social_scoring"] or system["subliminal"]:
            return "UNACCEPTABLE"
        # High risk
        if system["domain"] in high_risk_domains or (
                system["affects_rights"] and system["domain"] == "government"):
            return "HIGH"
        # Limited risk (transparency obligations)
        if system["domain"] in limited_risk_keywords:
            return "LIMITED"
        # Minimal risk
        return "MINIMAL"

    obligations = {
        "UNACCEPTABLE": "PROHIBITED - System cannot be deployed in the EU.",
        "HIGH": (
            "- Mandatory risk management system\n"
            "    - Data governance and quality requirements\n"
            "    - Technical documentation and logging\n"
            "    - Transparency and information to users\n"
            "    - Human oversight mechanisms\n"
            "    - Accuracy, robustness, and cybersecurity"
        ),
        "LIMITED": (
            "- Transparency obligation: users must be informed\n"
            "      they are interacting with an AI system\n"
            "    - Disclosure of AI-generated content"
        ),
        "MINIMAL": "- No mandatory requirements (voluntary codes of conduct).",
    }

    print("  EU AI Act Risk Classification:")
    print("  " + "=" * 65)

    for system in ai_systems:
        risk = classify_risk(system)
        print(f"\n  System: {system['name']}")
        print(f"    Description: {system['description']}")
        print(f"    Risk Level: {risk}")
        print(f"    Obligations:")
        print(f"    {obligations[risk]}")

    print(f"\n  Key principle: Higher risk demands greater interpretability.")
    print(f"  High-risk systems must provide explanations of decisions")
    print(f"  and maintain comprehensive technical documentation.")


# === Exercise 2: GDPR Article 22 Applicability ===
# Problem: Given scenarios of automated decision-making, determine whether
# GDPR Article 22 applies and what rights the data subject has.

def exercise_2():
    """Determine GDPR Article 22 applicability for automated decisions."""

    scenarios = [
        {
            "name": "Automated Loan Rejection",
            "description": "Bank uses ML model to automatically reject loan "
                           "applications without human review.",
            "solely_automated": True,
            "legal_or_significant_effect": True,
            "explicit_consent": False,
            "contract_necessary": False,
            "legal_authorization": False,
        },
        {
            "name": "Movie Recommendation",
            "description": "Streaming service recommends movies based on "
                           "viewing history.",
            "solely_automated": True,
            "legal_or_significant_effect": False,
            "explicit_consent": False,
            "contract_necessary": False,
            "legal_authorization": False,
        },
        {
            "name": "Insurance Premium with Human Review",
            "description": "AI calculates insurance premium, but human agent "
                           "reviews and approves each decision.",
            "solely_automated": False,
            "legal_or_significant_effect": True,
            "explicit_consent": False,
            "contract_necessary": False,
            "legal_authorization": False,
        },
        {
            "name": "Automated Fraud Detection",
            "description": "Bank automatically freezes accounts flagged as "
                           "fraudulent by ML model.",
            "solely_automated": True,
            "legal_or_significant_effect": True,
            "explicit_consent": False,
            "contract_necessary": True,
            "legal_authorization": False,
        },
    ]

    print("  GDPR Article 22 Analysis:")
    print("  " + "=" * 65)

    for s in scenarios:
        print(f"\n  Scenario: {s['name']}")
        print(f"    {s['description']}")

        # Article 22(1) applies when: solely automated + legal/significant effect
        art22_applies = s["solely_automated"] and s["legal_or_significant_effect"]

        print(f"    Solely automated: {s['solely_automated']}")
        print(f"    Legal/significant effect: {s['legal_or_significant_effect']}")
        print(f"    Article 22 applies: {art22_applies}")

        if art22_applies:
            # Check exceptions (Article 22(2))
            exception = (s["explicit_consent"] or s["contract_necessary"]
                         or s["legal_authorization"])
            if exception:
                reasons = []
                if s["contract_necessary"]:
                    reasons.append("necessary for contract")
                if s["explicit_consent"]:
                    reasons.append("explicit consent")
                if s["legal_authorization"]:
                    reasons.append("legal authorization")
                print(f"    Exception applies: {', '.join(reasons)}")
                print(f"    -> Automated decision allowed, but must provide:")
                print(f"       - Right to obtain human intervention")
                print(f"       - Right to express point of view")
                print(f"       - Right to contest the decision")
            else:
                print(f"    No exception applies.")
                print(f"    -> Data subject has RIGHT NOT TO BE SUBJECT to")
                print(f"       this automated decision.")

            print(f"    Interpretability requirements:")
            print(f"       - Meaningful information about logic involved")
            print(f"       - Significance and envisaged consequences")
            print(f"       - Right to explanation of the decision")
        else:
            print(f"    Article 22 does not apply to this scenario.")
            if not s["solely_automated"]:
                print(f"    Reason: Decision involves human review.")
            elif not s["legal_or_significant_effect"]:
                print(f"    Reason: No legal or significant effect on individual.")


# === Exercise 3: Writing a Model Card ===
# Problem: Generate a structured model card for a given ML model,
# following the format proposed by Mitchell et al. (2019).

def exercise_3():
    """Generate a model card for a credit scoring model."""

    model_card = {
        "model_details": {
            "name": "CreditScore-LR-v2.1",
            "version": "2.1",
            "type": "Logistic Regression",
            "developer": "FinTech Risk Analytics Team",
            "date": "2025-11-15",
            "framework": "scikit-learn 1.4.0",
            "license": "Proprietary",
            "contact": "risk-analytics@example.com",
        },
        "intended_use": {
            "primary_use": "Credit risk assessment for personal loan applications",
            "primary_users": "Loan officers at partner banks",
            "out_of_scope": [
                "Mortgage lending decisions",
                "Corporate credit assessment",
                "Decisions for applicants under 18",
            ],
        },
        "training_data": {
            "dataset": "Internal loan application dataset (2020-2024)",
            "size": "2.4M applications",
            "features": "42 features (demographics excluded from model input)",
            "label": "Default within 24 months (binary)",
            "preprocessing": "Missing value imputation, standard scaling",
            "splits": "70/15/15 train/val/test (temporal split)",
        },
        "evaluation_metrics": {
            "AUC-ROC": 0.82,
            "Accuracy": 0.76,
            "Precision_positive": 0.71,
            "Recall_positive": 0.68,
            "F1_positive": 0.69,
        },
        "fairness_analysis": {
            "protected_attributes_tested": ["gender", "age_group", "ethnicity"],
            "demographic_parity_gap": {"gender": 0.03, "age_group": 0.07, "ethnicity": 0.05},
            "equal_opportunity_gap": {"gender": 0.04, "age_group": 0.09, "ethnicity": 0.06},
            "four_fifths_rule_pass": {"gender": True, "age_group": False, "ethnicity": True},
        },
        "ethical_considerations": [
            "Model does not use protected attributes as direct inputs",
            "Proxy variable analysis conducted; zip code removed due to correlation with ethnicity",
            "Age group shows >5% equal opportunity gap; mitigation recommended",
            "Regular bias audits scheduled quarterly",
        ],
        "limitations": [
            "Trained on data from 2020-2024; may not generalize to different economic conditions",
            "Performance degrades for applicants with thin credit files (<3 accounts)",
            "Not validated for markets outside the United States",
            "Assumes stable feature distributions; monitoring required for data drift",
        ],
        "interpretability": {
            "global_method": "Logistic regression coefficients (inherently interpretable)",
            "local_method": "Feature contribution breakdown per applicant",
            "adverse_action_reasons": "Top 4 negative factors provided per rejection",
        },
    }

    print("  ============================================")
    print("  MODEL CARD: CreditScore-LR-v2.1")
    print("  ============================================")

    # Model Details
    print("\n  1. MODEL DETAILS")
    for key, val in model_card["model_details"].items():
        print(f"     {key.replace('_', ' ').title()}: {val}")

    # Intended Use
    print("\n  2. INTENDED USE")
    iu = model_card["intended_use"]
    print(f"     Primary use: {iu['primary_use']}")
    print(f"     Primary users: {iu['primary_users']}")
    print(f"     Out of scope:")
    for item in iu["out_of_scope"]:
        print(f"       - {item}")

    # Training Data
    print("\n  3. TRAINING DATA")
    for key, val in model_card["training_data"].items():
        print(f"     {key.replace('_', ' ').title()}: {val}")

    # Evaluation
    print("\n  4. EVALUATION METRICS")
    for metric, val in model_card["evaluation_metrics"].items():
        print(f"     {metric}: {val}")

    # Fairness
    print("\n  5. FAIRNESS ANALYSIS")
    fa = model_card["fairness_analysis"]
    print(f"     Protected attributes: {', '.join(fa['protected_attributes_tested'])}")
    print(f"     {'Attribute':<12} {'DP Gap':>8} {'EO Gap':>8} {'4/5 Rule':>10}")
    for attr in fa["protected_attributes_tested"]:
        dp = fa["demographic_parity_gap"][attr]
        eo = fa["equal_opportunity_gap"][attr]
        rule = "PASS" if fa["four_fifths_rule_pass"][attr] else "FAIL"
        print(f"     {attr:<12} {dp:>8.3f} {eo:>8.3f} {rule:>10}")

    # Ethics
    print("\n  6. ETHICAL CONSIDERATIONS")
    for item in model_card["ethical_considerations"]:
        print(f"     - {item}")

    # Limitations
    print("\n  7. LIMITATIONS")
    for item in model_card["limitations"]:
        print(f"     - {item}")

    # Interpretability
    print("\n  8. INTERPRETABILITY")
    for key, val in model_card["interpretability"].items():
        print(f"     {key.replace('_', ' ').title()}: {val}")

    print("\n  A model card provides stakeholders with essential information")
    print("  for responsible deployment and ongoing governance.")


# === Exercise 4: Simplified AI Impact Assessment ===
# Problem: Create a structured AI impact assessment for a facial recognition
# system deployment scenario.

def exercise_4():
    """Create a simplified AI impact assessment for a facial recognition system."""

    assessment = {
        "system_name": "FaceVerify Access Control System",
        "assessor": "AI Ethics Review Board",
        "date": datetime(2025, 12, 1).strftime("%Y-%m-%d"),
        "system_description": (
            "Facial recognition system for building access control. "
            "Verifies employee identity against enrolled templates. "
            "Uses a CNN-based embedding model with cosine similarity matching."
        ),
        "purpose_and_necessity": {
            "stated_purpose": "Replace badge-based access with contactless biometric verification",
            "necessity": "Medium - badge system works but has security gaps (sharing, theft)",
            "alternatives_considered": [
                "Multi-factor badge + PIN (lower privacy impact)",
                "Fingerprint scanner (similar biometric concerns, less intrusive)",
                "Iris scanning (higher accuracy but more intrusive)",
            ],
            "proportionality": "Questionable - badge + PIN may achieve similar security",
        },
        "stakeholder_analysis": {
            "affected_parties": [
                ("Employees", "Direct - must use system daily", "High"),
                ("Visitors", "Direct - optional enrollment", "Medium"),
                ("Security team", "Operator - manages system", "Medium"),
                ("IT department", "Maintainer - stores biometric data", "Low"),
                ("Management", "Decision maker - bears liability", "Low"),
            ],
        },
        "risk_assessment": [
            {
                "risk": "Demographic bias in face recognition accuracy",
                "likelihood": "HIGH",
                "severity": "HIGH",
                "impact": "Certain demographic groups experience higher false rejection rates",
                "mitigation": "Test across demographic groups; set group-specific thresholds",
            },
            {
                "risk": "Biometric data breach",
                "likelihood": "MEDIUM",
                "severity": "CRITICAL",
                "impact": "Biometric templates are irrevocable; cannot be reissued like passwords",
                "mitigation": "Store only encrypted templates; use cancelable biometrics",
            },
            {
                "risk": "Function creep beyond access control",
                "likelihood": "MEDIUM",
                "severity": "HIGH",
                "impact": "System repurposed for surveillance, time tracking, or profiling",
                "mitigation": "Strict data use policy; technical access controls; audit logging",
            },
            {
                "risk": "Chilling effect on employee behavior",
                "likelihood": "MEDIUM",
                "severity": "MEDIUM",
                "impact": "Employees feel constantly monitored, reducing workplace comfort",
                "mitigation": "Transparency about data retention; clear opt-out policy",
            },
        ],
        "interpretability_requirements": [
            "System must provide clear accept/reject feedback with confidence level",
            "Failed verifications must show reason (no match, poor image, etc.)",
            "Audit logs must be available for security review",
            "Regular accuracy reports disaggregated by demographics",
        ],
        "recommendation": "CONDITIONAL APPROVAL",
        "conditions": [
            "Complete demographic bias testing before deployment",
            "Implement cancelable biometric templates",
            "Provide opt-out alternative (badge + PIN)",
            "Quarterly bias audits and accuracy reporting",
            "Data retention limited to 90 days for logs",
            "Annual re-assessment required",
        ],
    }

    print("  ============================================")
    print("  AI IMPACT ASSESSMENT")
    print("  ============================================")

    print(f"\n  System: {assessment['system_name']}")
    print(f"  Assessor: {assessment['assessor']}")
    print(f"  Date: {assessment['date']}")
    print(f"\n  Description: {assessment['system_description']}")

    print(f"\n  --- Purpose and Necessity ---")
    pn = assessment["purpose_and_necessity"]
    print(f"  Purpose: {pn['stated_purpose']}")
    print(f"  Necessity: {pn['necessity']}")
    print(f"  Proportionality: {pn['proportionality']}")
    print(f"  Alternatives considered:")
    for alt in pn["alternatives_considered"]:
        print(f"    - {alt}")

    print(f"\n  --- Stakeholder Analysis ---")
    print(f"  {'Party':<15} {'Involvement':<35} {'Concern':>8}")
    for party, involvement, concern in assessment["stakeholder_analysis"]["affected_parties"]:
        print(f"  {party:<15} {involvement:<35} {concern:>8}")

    print(f"\n  --- Risk Assessment ---")
    for risk in assessment["risk_assessment"]:
        print(f"\n  Risk: {risk['risk']}")
        print(f"    Likelihood: {risk['likelihood']}, Severity: {risk['severity']}")
        print(f"    Impact: {risk['impact']}")
        print(f"    Mitigation: {risk['mitigation']}")

    print(f"\n  --- Interpretability Requirements ---")
    for req in assessment["interpretability_requirements"]:
        print(f"    - {req}")

    print(f"\n  --- Decision ---")
    print(f"  Recommendation: {assessment['recommendation']}")
    print(f"  Conditions:")
    for i, cond in enumerate(assessment["conditions"], 1):
        print(f"    {i}. {cond}")

    print(f"\n  An AI impact assessment ensures systematic evaluation of risks,")
    print(f"  stakeholder concerns, and interpretability requirements before deployment.")


if __name__ == "__main__":
    print("=== Exercise 1: EU AI Act Risk Classification ===")
    exercise_1()
    print("\n=== Exercise 2: GDPR Article 22 Applicability ===")
    exercise_2()
    print("\n=== Exercise 3: Writing a Model Card ===")
    exercise_3()
    print("\n=== Exercise 4: Simplified AI Impact Assessment ===")
    exercise_4()
    print("\nAll exercises completed!")
