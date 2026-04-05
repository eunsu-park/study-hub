"""
15. Domain-Specific Interpretability

Demonstrates interpretability techniques tailored to three domains:
healthcare (risk factor attribution with clinical thresholds), finance
(adverse action notice generation for credit decisions), and NLP
(token-level importance for text classification).

Covered topics:
    - Healthcare: clinical risk factor attribution with threshold alerts
    - Finance: ECOA-compliant adverse action reason code generation
    - NLP: token-level importance via leave-one-out for text classification
    - Domain-specific explanation formatting and stakeholder adaptation
    - Decision matrix for choosing interpretability methods by domain

Related to: L15 - Domain-Specific Interpretability

Requirements:
    pip install numpy matplotlib scikit-learn
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# ====== Section 1: Healthcare Interpretability ======

@dataclass
class ClinicalThreshold:
    """Clinical reference range for a health metric."""
    name: str
    unit: str
    low_normal: float
    high_normal: float
    low_critical: float
    high_critical: float


CLINICAL_THRESHOLDS = {
    "systolic_bp": ClinicalThreshold(
        "Systolic Blood Pressure", "mmHg", 90, 120, 70, 180,
    ),
    "glucose": ClinicalThreshold(
        "Fasting Glucose", "mg/dL", 70, 100, 40, 200,
    ),
    "cholesterol": ClinicalThreshold(
        "Total Cholesterol", "mg/dL", 125, 200, 100, 300,
    ),
    "bmi": ClinicalThreshold(
        "BMI", "kg/m2", 18.5, 24.9, 15, 40,
    ),
    "age": ClinicalThreshold(
        "Age", "years", 0, 120, 0, 120,
    ),
    "creatinine": ClinicalThreshold(
        "Creatinine", "mg/dL", 0.6, 1.2, 0.3, 4.0,
    ),
}


def generate_clinical_data(n: int = 1000, seed: int = 42) -> dict:
    """Generate synthetic patient data for cardiovascular risk prediction.

    Feature distributions approximate realistic clinical ranges.
    The outcome (cardiovascular event) depends on risk factors with
    known clinical relationships.

    Args:
        n: Number of patients.
        seed: Random seed.

    Returns:
        Dictionary with features, labels, and feature names.
    """
    rng = np.random.default_rng(seed)

    systolic_bp = rng.normal(130, 20, n)
    glucose = rng.normal(100, 25, n)
    cholesterol = rng.normal(200, 40, n)
    bmi = rng.normal(26, 5, n)
    age = rng.normal(55, 12, n).clip(20, 90)
    creatinine = rng.normal(1.0, 0.3, n).clip(0.3, 4.0)

    features = np.column_stack([
        systolic_bp, glucose, cholesterol, bmi, age, creatinine,
    ])
    feature_names = list(CLINICAL_THRESHOLDS.keys())

    # Normalize for model training
    means = features.mean(axis=0)
    stds = features.std(axis=0) + 1e-8
    features_norm = (features - means) / stds

    # Risk score based on clinical knowledge
    risk = (0.3 * (systolic_bp - 120) / 20 +
            0.2 * (glucose - 100) / 25 +
            0.2 * (cholesterol - 200) / 40 +
            0.1 * (bmi - 25) / 5 +
            0.15 * (age - 50) / 12 +
            0.05 * (creatinine - 1.0) / 0.3)
    risk += rng.normal(0, 0.3, n)
    labels = (risk > np.percentile(risk, 70)).astype(int)

    return {
        "features": features,
        "features_norm": features_norm,
        "labels": labels,
        "feature_names": feature_names,
        "means": means,
        "stds": stds,
    }


def clinical_explanation(
    model: GradientBoostingClassifier,
    patient: np.ndarray,
    patient_raw: np.ndarray,
    feature_names: list[str],
    background: np.ndarray,
) -> dict:
    """Generate a clinical risk factor attribution for one patient.

    Combines feature importance with clinical threshold checking to
    produce an explanation suitable for clinician review.

    Args:
        model: Trained risk prediction model.
        patient: Normalized patient features (d,).
        patient_raw: Raw (unnormalized) patient features (d,).
        feature_names: Clinical feature names.
        background: Background dataset for marginal expectations.

    Returns:
        Clinical explanation dictionary.
    """
    base_prob = model.predict_proba(patient.reshape(1, -1))[0, 1]

    # Compute local feature importance via marginal contribution
    importances = {}
    for j, name in enumerate(feature_names):
        diffs = []
        for _ in range(20):
            x_perm = patient.copy()
            bg_idx = np.random.randint(0, len(background))
            x_perm[j] = background[bg_idx, j]
            perm_prob = model.predict_proba(x_perm.reshape(1, -1))[0, 1]
            diffs.append(base_prob - perm_prob)
        importances[name] = float(np.mean(diffs))

    # Clinical threshold assessment
    threshold_alerts = []
    for j, name in enumerate(feature_names):
        if name not in CLINICAL_THRESHOLDS:
            continue
        threshold = CLINICAL_THRESHOLDS[name]
        value = patient_raw[j]

        if value >= threshold.high_critical or value <= threshold.low_critical:
            level = "CRITICAL"
        elif value >= threshold.high_normal or value <= threshold.low_normal:
            level = "ABNORMAL"
        else:
            level = "NORMAL"

        if level != "NORMAL":
            threshold_alerts.append({
                "feature": threshold.name,
                "value": f"{value:.1f} {threshold.unit}",
                "level": level,
                "reference": f"{threshold.low_normal}-{threshold.high_normal} {threshold.unit}",
            })

    # Sort risk factors by absolute importance
    sorted_factors = sorted(
        importances.items(), key=lambda x: abs(x[1]), reverse=True,
    )

    return {
        "risk_probability": float(base_prob),
        "risk_category": "HIGH" if base_prob > 0.5 else "MODERATE" if base_prob > 0.3 else "LOW",
        "top_risk_factors": sorted_factors[:4],
        "threshold_alerts": threshold_alerts,
        "all_importances": importances,
    }


# ====== Section 2: Financial Interpretability ======

def generate_credit_data(n: int = 1000, seed: int = 42) -> dict:
    """Generate synthetic credit application data."""
    rng = np.random.default_rng(seed)

    feature_names = [
        "payment_history", "credit_utilization", "credit_age",
        "recent_inquiries", "debt_to_income", "annual_income",
    ]

    payment_history = rng.normal(0.8, 0.15, n).clip(0, 1)
    credit_utilization = rng.normal(0.35, 0.2, n).clip(0, 1)
    credit_age = rng.normal(8, 4, n).clip(0, 30)
    recent_inquiries = rng.poisson(2, n).clip(0, 10)
    debt_to_income = rng.normal(0.35, 0.15, n).clip(0, 1)
    annual_income = rng.normal(60000, 25000, n).clip(15000, 300000)

    features = np.column_stack([
        payment_history, credit_utilization, credit_age,
        recent_inquiries, debt_to_income, annual_income / 100000,
    ])

    score = (2.0 * payment_history - 1.5 * credit_utilization +
             0.5 * credit_age / 10 - 0.3 * recent_inquiries / 5 -
             1.0 * debt_to_income + 0.3 * annual_income / 100000)
    score += rng.normal(0, 0.3, n)
    labels = (score > np.median(score)).astype(int)

    return {
        "features": features,
        "labels": labels,
        "feature_names": feature_names,
    }


# ECOA adverse action reason code mapping
ADVERSE_ACTION_CODES = {
    "payment_history": {
        "code": "A01",
        "reason": "Payment history does not meet minimum requirements",
    },
    "credit_utilization": {
        "code": "A02",
        "reason": "Credit utilization ratio is too high",
    },
    "credit_age": {
        "code": "A03",
        "reason": "Insufficient length of credit history",
    },
    "recent_inquiries": {
        "code": "A04",
        "reason": "Too many recent credit inquiries",
    },
    "debt_to_income": {
        "code": "A05",
        "reason": "Debt-to-income ratio exceeds acceptable threshold",
    },
    "annual_income": {
        "code": "A06",
        "reason": "Annual income is below minimum requirement",
    },
}


def generate_adverse_action_notice(
    model: LogisticRegression,
    applicant: np.ndarray,
    feature_names: list[str],
    background: np.ndarray,
    max_reasons: int = 4,
) -> dict:
    """Generate an ECOA-compliant adverse action notice.

    Under the Equal Credit Opportunity Act (ECOA) and Regulation B,
    when a credit application is denied, the lender must provide
    specific reasons for the denial.  The reasons must be drawn from
    a standardized list and ordered by importance.

    This function computes feature-level importance and maps the top
    negative factors to ECOA reason codes.

    Args:
        model: Trained credit scoring model.
        applicant: Single applicant's features (d,).
        feature_names: Feature names.
        background: Background dataset.
        max_reasons: Maximum number of adverse action reasons (ECOA: 4).

    Returns:
        Adverse action notice dictionary.
    """
    prob = model.predict_proba(applicant.reshape(1, -1))[0, 1]
    decision = "APPROVED" if prob >= 0.5 else "DENIED"

    # Compute feature contributions
    contributions = {}
    for j, name in enumerate(feature_names):
        diffs = []
        for _ in range(30):
            x_perm = applicant.copy()
            bg_idx = np.random.randint(0, len(background))
            x_perm[j] = background[bg_idx, j]
            perm_prob = model.predict_proba(x_perm.reshape(1, -1))[0, 1]
            diffs.append(prob - perm_prob)
        contributions[name] = float(np.mean(diffs))

    # For denied applications, find features that hurt the score most
    # (negative contribution = removing this feature would increase score)
    adverse_factors = sorted(
        [(name, val) for name, val in contributions.items() if val < 0],
        key=lambda x: x[1],
    )[:max_reasons]

    reasons = []
    for name, contribution in adverse_factors:
        if name in ADVERSE_ACTION_CODES:
            reasons.append({
                "code": ADVERSE_ACTION_CODES[name]["code"],
                "reason": ADVERSE_ACTION_CODES[name]["reason"],
                "feature": name,
                "contribution": contribution,
            })

    return {
        "decision": decision,
        "score": float(prob),
        "adverse_action_reasons": reasons,
        "all_contributions": contributions,
    }


# ====== Section 3: NLP Token-Level Interpretability ======

def generate_text_data(n: int = 500, seed: int = 42) -> dict:
    """Generate synthetic text classification data as bag-of-words.

    Simulates a sentiment analysis task where certain tokens are
    predictive of positive/negative sentiment.

    Args:
        n: Number of documents.
        seed: Random seed.

    Returns:
        Dictionary with feature matrix, labels, and token names.
    """
    rng = np.random.default_rng(seed)

    tokens = [
        "excellent", "terrible", "good", "bad", "amazing",
        "awful", "great", "poor", "love", "hate",
        "the", "is", "it", "a", "very",
    ]

    positive_tokens = {"excellent", "good", "amazing", "great", "love"}
    negative_tokens = {"terrible", "bad", "awful", "poor", "hate"}

    features = np.zeros((n, len(tokens)))
    labels = np.zeros(n, dtype=int)

    for i in range(n):
        # Random token counts
        for j, token in enumerate(tokens):
            features[i, j] = rng.poisson(2)

        # Label based on sentiment balance
        pos_count = sum(features[i, j] for j, t in enumerate(tokens) if t in positive_tokens)
        neg_count = sum(features[i, j] for j, t in enumerate(tokens) if t in negative_tokens)
        labels[i] = 1 if pos_count > neg_count + rng.normal(0, 1) else 0

    return {
        "features": features,
        "labels": labels,
        "token_names": tokens,
    }


def token_importance_leave_one_out(
    model: RandomForestClassifier,
    document: np.ndarray,
    token_names: list[str],
) -> dict:
    """Compute token-level importance via leave-one-out.

    For each token, we set its count to zero and measure the change
    in prediction probability. This is a simple but effective method
    for understanding which tokens drive a text classification.

    In real NLP systems, this would be replaced by token-level SHAP,
    attention weights, or rationale extraction (ERASER benchmark).

    Args:
        model: Trained text classifier.
        document: Token count vector (d,).
        token_names: Token names.

    Returns:
        Dictionary with per-token importance and prediction details.
    """
    base_prob = model.predict_proba(document.reshape(1, -1))[0, 1]
    base_pred = "POSITIVE" if base_prob >= 0.5 else "NEGATIVE"

    token_effects = {}
    for j, name in enumerate(token_names):
        if document[j] == 0:
            token_effects[name] = {"importance": 0.0, "present": False}
            continue

        x_removed = document.copy()
        x_removed[j] = 0
        new_prob = model.predict_proba(x_removed.reshape(1, -1))[0, 1]

        token_effects[name] = {
            "importance": float(base_prob - new_prob),
            "present": True,
            "count": int(document[j]),
        }

    # Sort by absolute importance
    sorted_tokens = sorted(
        [(k, v) for k, v in token_effects.items() if v["present"]],
        key=lambda x: abs(x[1]["importance"]),
        reverse=True,
    )

    return {
        "prediction": base_pred,
        "confidence": float(base_prob),
        "token_effects": token_effects,
        "top_tokens": sorted_tokens[:5],
    }


# ====== Section 4: Visualization ======

def visualize_domain_explanations(
    clinical_exp: dict,
    financial_exp: dict,
    nlp_exp: dict,
    feature_names_clinical: list[str],
    save_path: str = "domain_specific_interp.png",
) -> None:
    """Four-panel domain-specific interpretability visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Panel 1: Clinical Risk Factors ---
    ax1 = axes[0, 0]
    factors = clinical_exp["top_risk_factors"]
    names = [f[0] for f in factors]
    vals = [f[1] for f in factors]
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in vals]
    ax1.barh(names, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_xlabel("Risk Contribution")
    ax1.set_title(f"Clinical Risk Factors\n"
                   f"Risk: {clinical_exp['risk_category']} "
                   f"({clinical_exp['risk_probability']:.2%})")
    ax1.axvline(x=0, color="gray", linewidth=0.5)

    # --- Panel 2: Adverse Action Reasons ---
    ax2 = axes[0, 1]
    if financial_exp["adverse_action_reasons"]:
        reasons = financial_exp["adverse_action_reasons"]
        codes = [r["code"] for r in reasons]
        contribs = [abs(r["contribution"]) for r in reasons]
        ax2.barh(codes, contribs, color="#f39c12",
                 edgecolor="black", linewidth=0.5)
        for i, r in enumerate(reasons):
            ax2.text(contribs[i] + 0.001, i, r["reason"][:35] + "...",
                     va="center", fontsize=8)
    ax2.set_xlabel("|Contribution|")
    ax2.set_title(f"Adverse Action Notice\n"
                   f"Decision: {financial_exp['decision']} "
                   f"(score={financial_exp['score']:.2f})")

    # --- Panel 3: Token Importance (NLP) ---
    ax3 = axes[1, 0]
    top_tokens = nlp_exp["top_tokens"]
    token_names_plot = [t[0] for t in top_tokens]
    token_vals = [t[1]["importance"] for t in top_tokens]
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in token_vals]
    ax3.barh(token_names_plot, token_vals, color=colors,
             edgecolor="black", linewidth=0.5)
    ax3.set_xlabel("Importance (leave-one-out)")
    ax3.set_title(f"Token-Level Importance\n"
                   f"Prediction: {nlp_exp['prediction']} "
                   f"({nlp_exp['confidence']:.2%})")
    ax3.axvline(x=0, color="gray", linewidth=0.5)

    # --- Panel 4: Method Selection Matrix ---
    ax4 = axes[1, 1]
    ax4.axis("off")
    matrix_data = [
        ["Domain", "Method", "Stakeholder", "Regulation"],
        ["Healthcare", "Risk factors", "Clinician", "FDA SaMD"],
        ["Finance", "Adverse action", "Applicant", "ECOA/FCRA"],
        ["NLP", "Token highlight", "End user", "EU AI Act"],
        ["CV", "Saliency map", "Engineer", "Domain-dep."],
    ]
    table = ax4.table(
        cellText=matrix_data[1:],
        colLabels=matrix_data[0],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)
    ax4.set_title("Method Selection Matrix by Domain", pad=20)

    plt.suptitle("Domain-Specific Interpretability", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Figure saved to: {save_path}")
    plt.close()


# ====== Section 5: Main Pipeline ======

def main() -> None:
    """Demonstrate domain-specific interpretability across three domains."""
    print("=" * 65)
    print("  Domain-Specific Interpretability")
    print("  Healthcare | Finance | NLP")
    print("=" * 65)

    # --- Step 1: Healthcare ---
    print("\n[1] Healthcare: Clinical Risk Prediction")
    print("-" * 50)

    clinical_data = generate_clinical_data(n=1000)
    clinical_model = GradientBoostingClassifier(
        n_estimators=50, max_depth=3, random_state=42,
    )
    clinical_model.fit(clinical_data["features_norm"], clinical_data["labels"])
    acc = clinical_model.score(clinical_data["features_norm"], clinical_data["labels"])
    print(f"  Model accuracy: {acc:.4f}")

    # Explain a high-risk patient
    high_risk_idx = np.where(clinical_data["labels"] == 1)[0][0]
    clinical_exp = clinical_explanation(
        clinical_model,
        clinical_data["features_norm"][high_risk_idx],
        clinical_data["features"][high_risk_idx],
        clinical_data["feature_names"],
        clinical_data["features_norm"][:100],
    )

    print(f"\n  Patient Risk Assessment:")
    print(f"    Risk Category: {clinical_exp['risk_category']}")
    print(f"    Risk Probability: {clinical_exp['risk_probability']:.2%}")
    print(f"    Top Risk Factors:")
    for name, imp in clinical_exp["top_risk_factors"]:
        print(f"      {name}: {imp:+.4f}")
    if clinical_exp["threshold_alerts"]:
        print(f"    Clinical Alerts:")
        for alert in clinical_exp["threshold_alerts"]:
            print(f"      [{alert['level']}] {alert['feature']}: "
                  f"{alert['value']} (ref: {alert['reference']})")

    # --- Step 2: Finance ---
    print("\n\n[2] Finance: Credit Decision with Adverse Action")
    print("-" * 50)

    credit_data = generate_credit_data(n=1000)
    credit_model = LogisticRegression(max_iter=1000, random_state=42)
    credit_model.fit(credit_data["features"], credit_data["labels"])
    acc = credit_model.score(credit_data["features"], credit_data["labels"])
    print(f"  Model accuracy: {acc:.4f}")

    # Find a denied applicant
    probs = credit_model.predict_proba(credit_data["features"])[:, 1]
    denied_idx = np.where(probs < 0.4)[0][0]

    financial_exp = generate_adverse_action_notice(
        credit_model,
        credit_data["features"][denied_idx],
        credit_data["feature_names"],
        credit_data["features"][:100],
    )

    print(f"\n  Credit Decision:")
    print(f"    Decision: {financial_exp['decision']}")
    print(f"    Score: {financial_exp['score']:.4f}")
    if financial_exp["adverse_action_reasons"]:
        print(f"    Adverse Action Reasons:")
        for reason in financial_exp["adverse_action_reasons"]:
            print(f"      [{reason['code']}] {reason['reason']}")

    # --- Step 3: NLP ---
    print("\n\n[3] NLP: Token-Level Sentiment Explanation")
    print("-" * 50)

    text_data = generate_text_data(n=500)
    text_model = RandomForestClassifier(
        n_estimators=50, max_depth=5, random_state=42,
    )
    text_model.fit(text_data["features"], text_data["labels"])
    acc = text_model.score(text_data["features"], text_data["labels"])
    print(f"  Model accuracy: {acc:.4f}")

    # Explain a positive prediction
    pos_idx = np.where(text_data["labels"] == 1)[0][0]
    nlp_exp = token_importance_leave_one_out(
        text_model,
        text_data["features"][pos_idx],
        text_data["token_names"],
    )

    print(f"\n  Sentiment Analysis:")
    print(f"    Prediction: {nlp_exp['prediction']}")
    print(f"    Confidence: {nlp_exp['confidence']:.2%}")
    print(f"    Top Contributing Tokens:")
    for name, details in nlp_exp["top_tokens"]:
        direction = "+" if details["importance"] > 0 else "-"
        print(f"      {name} (x{details.get('count', 0)}): "
              f"{direction}{abs(details['importance']):.4f}")

    # --- Step 4: Visualization ---
    print("\n\n[4] Generating Domain Comparison Visualization")
    print("-" * 50)

    visualize_domain_explanations(
        clinical_exp, financial_exp, nlp_exp,
        clinical_data["feature_names"],
    )

    # --- Summary ---
    print("\n" + "=" * 65)
    print("  Summary")
    print("=" * 65)
    print("""
  Key findings:
    1. Healthcare: explanations must map to clinical concepts and
       reference ranges. Clinicians need risk factors ranked by
       contribution with threshold alerts.
    2. Finance: ECOA mandates specific adverse action reason codes
       for denied applications. Feature importance maps directly
       to standardized reason codes.
    3. NLP: token-level importance (leave-one-out, SHAP, attention)
       highlights which words drive the prediction. Positive tokens
       push toward positive sentiment, negative tokens push away.
    4. Each domain has unique stakeholders, regulatory requirements,
       and explanation formats. One-size-fits-all does not work.
    """)


if __name__ == "__main__":
    main()
