"""
Exercises for Lesson 15: Domain-Specific Interpretability
Topic: Interpretable_AI

Solutions to practice problems from the lesson.
"""

import numpy as np
from collections import defaultdict


# === Exercise 1: Generating Adverse Action Notices from Feature Importances ===
# Problem: Given a credit model's feature importances for a rejected applicant,
# generate a compliant adverse action notice listing the top reasons for denial
# in consumer-friendly language.

def exercise_1():
    """Generate adverse action notices from feature importances."""
    np.random.seed(42)

    # Feature importance mapping with consumer-friendly descriptions
    feature_descriptions = {
        "credit_score": "Your credit score is below our minimum threshold",
        "debt_to_income": "Your debt-to-income ratio is too high",
        "length_of_credit": "Length of credit history is insufficient",
        "recent_inquiries": "Too many recent credit inquiries",
        "payment_history": "Late or missed payments on credit accounts",
        "credit_utilization": "High utilization of available credit",
        "public_records": "Negative public records (bankruptcy, liens)",
        "num_accounts": "Insufficient number of credit accounts",
        "employment_length": "Employment history duration is below threshold",
        "loan_amount_ratio": "Requested loan amount relative to income is too high",
    }

    # Simulated rejected applicants with feature contributions to denial
    applicants = [
        {
            "id": "APP-2025-001",
            "name": "Applicant A",
            "decision": "DENIED",
            "score": 0.32,  # Below threshold of 0.5
            "feature_contributions": {
                "credit_score": -0.25,
                "debt_to_income": -0.18,
                "payment_history": -0.12,
                "credit_utilization": -0.08,
                "recent_inquiries": -0.05,
                "length_of_credit": -0.03,
                "num_accounts": 0.02,
                "employment_length": 0.05,
                "loan_amount_ratio": -0.02,
                "public_records": 0.00,
            },
        },
        {
            "id": "APP-2025-002",
            "name": "Applicant B",
            "decision": "DENIED",
            "score": 0.41,
            "feature_contributions": {
                "credit_score": -0.05,
                "debt_to_income": -0.08,
                "payment_history": -0.02,
                "credit_utilization": -0.20,
                "recent_inquiries": -0.15,
                "length_of_credit": -0.10,
                "num_accounts": 0.01,
                "employment_length": 0.03,
                "loan_amount_ratio": -0.04,
                "public_records": -0.01,
            },
        },
    ]

    def generate_adverse_action_notice(applicant, top_n=4):
        """Generate ECOA/FCRA compliant adverse action notice."""
        notice = []
        notice.append("=" * 60)
        notice.append("ADVERSE ACTION NOTICE")
        notice.append("Equal Credit Opportunity Act / Fair Credit Reporting Act")
        notice.append("=" * 60)
        notice.append(f"Application ID: {applicant['id']}")
        notice.append(f"Date: 2025-12-01")
        notice.append(f"Decision: {applicant['decision']}")
        notice.append("")
        notice.append("Dear Applicant,")
        notice.append("")
        notice.append("After careful review of your application, we regret to")
        notice.append("inform you that your request for credit has been denied.")
        notice.append("")
        notice.append("The principal reason(s) for this decision are:")
        notice.append("")

        # Sort by most negative contribution (most impactful denial reasons)
        negative_factors = {k: v for k, v in applicant["feature_contributions"].items()
                           if v < -0.01}
        sorted_factors = sorted(negative_factors.items(), key=lambda x: x[1])
        top_reasons = sorted_factors[:top_n]

        for i, (feature, contribution) in enumerate(top_reasons, 1):
            description = feature_descriptions.get(feature, feature)
            notice.append(f"  {i}. {description}")

        notice.append("")
        notice.append("You have the right to:")
        notice.append("  - Request a free copy of your credit report within 60 days")
        notice.append("  - Dispute the accuracy of information in your credit report")
        notice.append("  - Request a statement of specific reasons for this decision")
        notice.append("")
        notice.append("Credit Reporting Agency: ExampleBureau")
        notice.append("Phone: 1-800-XXX-XXXX")
        notice.append("=" * 60)

        return "\n".join(notice)

    for applicant in applicants:
        print(f"\n  --- {applicant['name']} ---")
        notice = generate_adverse_action_notice(applicant)
        for line in notice.split("\n"):
            print(f"  {line}")

        # Show feature contribution breakdown
        print(f"\n  Feature Contribution Analysis:")
        sorted_contribs = sorted(applicant["feature_contributions"].items(),
                                 key=lambda x: x[1])
        for feat, contrib in sorted_contribs:
            bar = "+" * int(abs(contrib) * 100) if contrib > 0 else "-" * int(abs(contrib) * 100)
            direction = "+" if contrib > 0 else ""
            print(f"    {feat:<22} {direction}{contrib:>7.3f} {bar}")


# === Exercise 2: Multi-Stakeholder Explanation System ===
# Problem: Design a system that generates different explanation formats
# for different stakeholders (data scientists, business users, regulators,
# affected individuals) from the same underlying model.

def exercise_2():
    """Design a multi-stakeholder explanation system with role-based views."""
    np.random.seed(42)

    # Simulated model prediction with rich explanation data
    prediction = {
        "input": {
            "patient_age": 65,
            "blood_pressure": 155,
            "cholesterol": 240,
            "bmi": 28.5,
            "smoking": 1,
            "diabetes": 0,
            "family_history": 1,
        },
        "output": {
            "risk_score": 0.78,
            "risk_category": "HIGH",
            "recommended_action": "Immediate cardiology referral",
        },
        "explanation": {
            "feature_importances": {
                "blood_pressure": 0.28,
                "cholesterol": 0.22,
                "smoking": 0.18,
                "patient_age": 0.15,
                "family_history": 0.09,
                "bmi": 0.05,
                "diabetes": 0.03,
            },
            "counterfactuals": [
                {"change": "blood_pressure: 155 -> 130", "new_risk": 0.52,
                 "category": "MODERATE"},
                {"change": "smoking: 1 -> 0", "new_risk": 0.58,
                 "category": "MODERATE"},
                {"change": "cholesterol: 240 -> 200", "new_risk": 0.61,
                 "category": "MODERATE"},
            ],
            "model_confidence": 0.92,
            "similar_cases": {"total": 150, "same_outcome_pct": 0.83},
        },
    }

    class StakeholderExplainer:
        """Generates role-appropriate explanations."""

        def __init__(self, prediction_data):
            self.data = prediction_data

        def for_data_scientist(self):
            """Technical explanation with full model details."""
            lines = [
                "TECHNICAL EXPLANATION (Data Scientist View)",
                "-" * 50,
                f"Model output: risk_score={self.data['output']['risk_score']:.4f}",
                f"Model confidence: {self.data['explanation']['model_confidence']:.4f}",
                "",
                "Feature attributions (SHAP values):",
            ]
            for feat, imp in sorted(self.data["explanation"]["feature_importances"].items(),
                                    key=lambda x: -x[1]):
                bar = "#" * int(imp * 50)
                lines.append(f"  {feat:<20} {imp:.4f} {bar}")

            lines.append("")
            lines.append("Counterfactual analysis:")
            for cf in self.data["explanation"]["counterfactuals"]:
                lines.append(f"  If {cf['change']}: risk -> {cf['new_risk']:.4f} "
                             f"({cf['category']})")

            lines.append("")
            sim = self.data["explanation"]["similar_cases"]
            lines.append(f"Similar cases in training data: {sim['total']}")
            lines.append(f"Same-outcome percentage: {sim['same_outcome_pct']:.0%}")
            return "\n".join(lines)

        def for_business_user(self):
            """Simplified explanation for clinical/business staff."""
            d = self.data
            top3 = sorted(d["explanation"]["feature_importances"].items(),
                          key=lambda x: -x[1])[:3]

            lines = [
                "CLINICAL SUMMARY (Business User View)",
                "-" * 50,
                f"Patient Risk Level: {d['output']['risk_category']}",
                f"Risk Score: {d['output']['risk_score']:.0%}",
                f"Recommendation: {d['output']['recommended_action']}",
                "",
                "Top contributing factors:",
            ]
            factor_labels = {
                "blood_pressure": "High blood pressure (155 mmHg)",
                "cholesterol": "Elevated cholesterol (240 mg/dL)",
                "smoking": "Active smoker",
                "patient_age": "Age (65 years)",
                "family_history": "Family history of heart disease",
                "bmi": "BMI slightly elevated (28.5)",
                "diabetes": "No diabetes (protective factor)",
            }
            for i, (feat, imp) in enumerate(top3, 1):
                label = factor_labels.get(feat, feat)
                print_imp = "High" if imp > 0.2 else "Moderate" if imp > 0.1 else "Low"
                lines.append(f"  {i}. {label} (Impact: {print_imp})")

            lines.append("")
            lines.append("What could reduce risk:")
            for cf in d["explanation"]["counterfactuals"][:2]:
                lines.append(f"  - {cf['change'].split(': ')[0].replace('_', ' ').title()}: "
                             f"would reduce risk to {cf['new_risk']:.0%}")
            return "\n".join(lines)

        def for_regulator(self):
            """Audit-focused explanation with compliance details."""
            d = self.data
            lines = [
                "REGULATORY AUDIT VIEW",
                "-" * 50,
                "Model Decision Record:",
                f"  Decision: {d['output']['risk_category']} risk",
                f"  Score: {d['output']['risk_score']:.6f}",
                f"  Confidence: {d['explanation']['model_confidence']:.6f}",
                "",
                "Complete feature attribution (all factors):",
            ]
            total_imp = 0
            for feat, imp in sorted(d["explanation"]["feature_importances"].items(),
                                    key=lambda x: -x[1]):
                input_val = d["input"].get(feat, "N/A")
                lines.append(f"  {feat:<20} input={str(input_val):<8} attribution={imp:.6f}")
                total_imp += imp
            lines.append(f"  {'TOTAL':<20} {'':8} attribution={total_imp:.6f}")

            lines.append("")
            lines.append("Protected attribute check:")
            protected = ["patient_age", "gender", "race", "ethnicity"]
            for attr in protected:
                if attr in d["explanation"]["feature_importances"]:
                    imp = d["explanation"]["feature_importances"][attr]
                    flag = " [REVIEW NEEDED]" if imp > 0.1 else ""
                    lines.append(f"  {attr}: attribution={imp:.6f}{flag}")
                else:
                    lines.append(f"  {attr}: Not used in model")

            lines.append("")
            sim = d["explanation"]["similar_cases"]
            lines.append(f"Statistical support: {sim['total']} similar cases, "
                         f"{sim['same_outcome_pct']:.0%} concordance")
            return "\n".join(lines)

        def for_patient(self):
            """Plain-language explanation for the affected individual."""
            d = self.data
            lines = [
                "YOUR HEALTH RISK ASSESSMENT",
                "-" * 50,
                "",
                f"Your heart disease risk level: {d['output']['risk_category']}",
                "",
                "What this means:",
                "  Based on your health information, your risk of heart disease",
                "  is higher than average. This does not mean you will develop",
                "  heart disease, but it suggests taking preventive steps.",
                "",
                "The main factors in your assessment:",
            ]
            factor_patient_labels = {
                "blood_pressure": "Your blood pressure is elevated",
                "cholesterol": "Your cholesterol level is above the recommended range",
                "smoking": "Smoking significantly increases heart disease risk",
            }
            top3 = sorted(d["explanation"]["feature_importances"].items(),
                          key=lambda x: -x[1])[:3]
            for feat, _ in top3:
                label = factor_patient_labels.get(feat, feat.replace("_", " ").title())
                lines.append(f"  - {label}")

            lines.append("")
            lines.append("Steps that could help reduce your risk:")
            patient_actions = {
                "blood_pressure": "Work with your doctor to manage blood pressure",
                "smoking": "Consider a smoking cessation program",
                "cholesterol": "Discuss cholesterol management with your doctor",
            }
            for cf in d["explanation"]["counterfactuals"][:2]:
                feat = cf["change"].split(":")[0]
                action = patient_actions.get(feat, f"Address {feat}")
                lines.append(f"  - {action}")

            lines.append("")
            lines.append("Please discuss these results with your healthcare provider.")
            return "\n".join(lines)

    explainer = StakeholderExplainer(prediction)
    stakeholders = [
        ("Data Scientist", explainer.for_data_scientist),
        ("Business User (Clinician)", explainer.for_business_user),
        ("Regulator", explainer.for_regulator),
        ("Patient", explainer.for_patient),
    ]

    for name, method in stakeholders:
        print(f"\n  {'=' * 55}")
        print(f"  STAKEHOLDER: {name}")
        print(f"  {'=' * 55}")
        explanation = method()
        for line in explanation.split("\n"):
            print(f"  {line}")

    print(f"\n  Key insight: The same prediction needs different explanations")
    print(f"  tailored to each stakeholder's expertise and needs.")


# === Exercise 3: Token-Level Attribution for NLP ===
# Problem: Implement a simplified token-level attribution method for text
# classification that shows which tokens most influenced the prediction.

def exercise_3():
    """Implement token-level attribution for text classification."""
    np.random.seed(42)

    class SimpleBagOfWordsClassifier:
        """Minimal BoW classifier for sentiment analysis."""

        def __init__(self):
            # Pre-defined sentiment weights (simulating a trained model)
            self.word_weights = {
                "excellent": 0.8, "great": 0.6, "good": 0.4, "nice": 0.3,
                "love": 0.7, "amazing": 0.9, "wonderful": 0.7, "best": 0.6,
                "happy": 0.5, "recommend": 0.4, "enjoy": 0.5, "perfect": 0.8,
                "terrible": -0.8, "bad": -0.6, "awful": -0.9, "worst": -0.7,
                "hate": -0.8, "horrible": -0.7, "disappointing": -0.6,
                "boring": -0.4, "waste": -0.5, "poor": -0.5, "never": -0.2,
                "not": -0.3, "slow": -0.3, "broken": -0.6,
            }
            self.bias = 0.0

        def predict(self, tokens):
            score = self.bias
            for token in tokens:
                score += self.word_weights.get(token.lower(), 0.0)
            prob = 1.0 / (1.0 + np.exp(-score))
            return prob

        def get_token_attributions(self, tokens):
            """Compute per-token attribution via leave-one-out."""
            base_score = self.predict(tokens)
            attributions = []
            for i, token in enumerate(tokens):
                reduced = tokens[:i] + tokens[i+1:]
                reduced_score = self.predict(reduced) if reduced else 0.5
                attribution = base_score - reduced_score
                attributions.append((token, attribution))
            return attributions

    model = SimpleBagOfWordsClassifier()

    texts = [
        "This movie was excellent and the acting was amazing",
        "Terrible film with awful plot and horrible acting",
        "Not bad actually the story was great but pacing was slow",
        "I love this product it is the best I have ever used",
    ]

    print("  Token-Level Attribution for Sentiment Analysis:")
    print("  " + "=" * 60)

    for text in texts:
        tokens = text.lower().split()
        prob = model.predict(tokens)
        sentiment = "POSITIVE" if prob >= 0.5 else "NEGATIVE"
        attributions = model.get_token_attributions(tokens)

        print(f"\n  Text: \"{text}\"")
        print(f"  Prediction: {sentiment} (confidence: {prob:.4f})")
        print(f"  Token attributions:")

        # Display with visual bars
        max_attr = max(abs(a) for _, a in attributions) if attributions else 1.0
        for token, attr in attributions:
            if abs(attr) < 0.001:
                bar = "."
            elif attr > 0:
                bar_len = int(abs(attr) / max(max_attr, 0.01) * 20)
                bar = "+" * max(bar_len, 1)
            else:
                bar_len = int(abs(attr) / max(max_attr, 0.01) * 20)
                bar = "-" * max(bar_len, 1)
            print(f"    {token:<15} {attr:>+.4f}  {bar}")

        # Highlight most influential tokens
        sorted_attr = sorted(attributions, key=lambda x: abs(x[1]), reverse=True)
        top_positive = [(t, a) for t, a in sorted_attr if a > 0.01][:2]
        top_negative = [(t, a) for t, a in sorted_attr if a < -0.01][:2]

        if top_positive:
            tp_str = ", ".join(f"'{t}'" for t, _ in top_positive)
            print(f"  Most positive tokens: {tp_str}")
        if top_negative:
            tn_str = ", ".join(f"'{t}'" for t, _ in top_negative)
            print(f"  Most negative tokens: {tn_str}")


# === Exercise 4: Selecting Explanation Methods for a Given Domain ===
# Problem: Given domain constraints and requirements, systematically select
# the most appropriate interpretability methods.

def exercise_4():
    """Select explanation methods based on domain requirements."""

    # Define method capabilities
    methods = {
        "Linear Coefficients": {
            "model_types": ["linear", "logistic_regression"],
            "scope": "global",
            "faithfulness": "exact",
            "human_readable": True,
            "regulatory_accepted": True,
            "real_time": True,
            "handles_interactions": False,
            "handles_nonlinear": False,
        },
        "Decision Tree / Rules": {
            "model_types": ["tree", "rule_list"],
            "scope": "global",
            "faithfulness": "exact",
            "human_readable": True,
            "regulatory_accepted": True,
            "real_time": True,
            "handles_interactions": True,
            "handles_nonlinear": True,
        },
        "SHAP (TreeSHAP)": {
            "model_types": ["tree", "random_forest", "gradient_boosting"],
            "scope": "local",
            "faithfulness": "exact",
            "human_readable": True,
            "regulatory_accepted": True,
            "real_time": True,
            "handles_interactions": True,
            "handles_nonlinear": True,
        },
        "LIME": {
            "model_types": ["any"],
            "scope": "local",
            "faithfulness": "approximate",
            "human_readable": True,
            "regulatory_accepted": True,
            "real_time": False,
            "handles_interactions": False,
            "handles_nonlinear": True,
        },
        "Integrated Gradients": {
            "model_types": ["neural_network"],
            "scope": "local",
            "faithfulness": "high",
            "human_readable": False,
            "regulatory_accepted": False,
            "real_time": True,
            "handles_interactions": True,
            "handles_nonlinear": True,
        },
        "Attention Visualization": {
            "model_types": ["transformer"],
            "scope": "local",
            "faithfulness": "low",
            "human_readable": True,
            "regulatory_accepted": False,
            "real_time": True,
            "handles_interactions": True,
            "handles_nonlinear": True,
        },
        "Concept-based (TCAV)": {
            "model_types": ["neural_network"],
            "scope": "global",
            "faithfulness": "high",
            "human_readable": True,
            "regulatory_accepted": False,
            "real_time": False,
            "handles_interactions": True,
            "handles_nonlinear": True,
        },
    }

    # Domain scenarios
    domains = [
        {
            "name": "Healthcare - Clinical Decision Support",
            "model_type": "gradient_boosting",
            "requirements": {
                "regulatory_accepted": True,
                "human_readable": True,
                "real_time": False,
                "scope_needed": "local",
            },
            "priority": "Clinician trust and regulatory compliance",
        },
        {
            "name": "Finance - Automated Lending",
            "model_type": "logistic_regression",
            "requirements": {
                "regulatory_accepted": True,
                "human_readable": True,
                "real_time": True,
                "scope_needed": "local",
            },
            "priority": "ECOA adverse action notices, speed",
        },
        {
            "name": "NLP - Content Moderation",
            "model_type": "transformer",
            "requirements": {
                "regulatory_accepted": False,
                "human_readable": True,
                "real_time": True,
                "scope_needed": "local",
            },
            "priority": "Token-level explanations, speed at scale",
        },
        {
            "name": "Manufacturing - Defect Detection",
            "model_type": "neural_network",
            "requirements": {
                "regulatory_accepted": False,
                "human_readable": True,
                "real_time": True,
                "scope_needed": "local",
            },
            "priority": "Visual saliency maps, operator trust",
        },
    ]

    print("  Domain-Specific Method Selection:")
    print("  " + "=" * 65)

    for domain in domains:
        print(f"\n  Domain: {domain['name']}")
        print(f"    Model type: {domain['model_type']}")
        print(f"    Priority: {domain['priority']}")
        print(f"    Requirements: {domain['requirements']}")

        # Score each method
        scored = []
        for method_name, method in methods.items():
            # Check model compatibility
            compatible = (domain["model_type"] in method["model_types"]
                          or "any" in method["model_types"])
            if not compatible:
                continue

            score = 0
            reasons = []

            # Check requirements
            reqs = domain["requirements"]
            if reqs["regulatory_accepted"] and not method["regulatory_accepted"]:
                score -= 10
                reasons.append("not regulatory accepted")
            elif method["regulatory_accepted"]:
                score += 5
                reasons.append("regulatory OK")

            if reqs["human_readable"] and method["human_readable"]:
                score += 3
            elif reqs["human_readable"] and not method["human_readable"]:
                score -= 3
                reasons.append("not human-readable")

            if reqs["real_time"] and method["real_time"]:
                score += 3
            elif reqs["real_time"] and not method["real_time"]:
                score -= 5
                reasons.append("too slow for real-time")

            # Scope match
            if reqs["scope_needed"] == method["scope"] or method["scope"] == "both":
                score += 2

            # Faithfulness bonus
            faith_scores = {"exact": 4, "high": 3, "approximate": 1, "low": 0}
            score += faith_scores.get(method["faithfulness"], 0)

            scored.append((method_name, score, reasons))

        scored.sort(key=lambda x: -x[1])

        print(f"\n    Ranked methods:")
        for i, (name, score, reasons) in enumerate(scored):
            marker = " <-- RECOMMENDED" if i == 0 else ""
            reason_str = f" ({', '.join(reasons)})" if reasons else ""
            print(f"      {i+1}. {name:<30} score={score:>3}{reason_str}{marker}")

    print(f"\n  Method selection must consider domain regulations, user expertise,")
    print(f"  latency requirements, and the specific model architecture.")


if __name__ == "__main__":
    print("=== Exercise 1: Adverse Action Notices ===")
    exercise_1()
    print("\n=== Exercise 2: Multi-Stakeholder Explanation System ===")
    exercise_2()
    print("\n=== Exercise 3: Token-Level Attribution for NLP ===")
    exercise_3()
    print("\n=== Exercise 4: Selecting Methods for a Domain ===")
    exercise_4()
    print("\nAll exercises completed!")
