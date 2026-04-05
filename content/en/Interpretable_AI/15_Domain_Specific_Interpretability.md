# Lesson 15: Domain-Specific Interpretability

[Previous: Production Interpretability](./14_Production_Interpretability.md) | [Next: Mechanistic Interpretability](./16_Mechanistic_Interpretability.md)

---

## Learning Objectives

- Apply FDA guidance on clinical decision support to design interpretable healthcare AI systems with appropriate explanations for clinicians, patients, and regulators
- Implement ECOA-compliant adverse action notices and SR 11-7 model risk management for financial AI, including feature importance for credit scoring and stress testing
- Use token-level SHAP, attention-based highlighting, and rationale extraction (ERASER benchmark) to explain NLP model predictions with faithful text explanations
- Apply pixel attribution methods and concept-based explanations to computer vision tasks including medical imaging and autonomous driving
- Select the appropriate interpretability method for a given domain using a structured decision matrix that considers stakeholder needs, regulatory requirements, and technical constraints

---

## 1. Healthcare AI Interpretability

### 1.1 Regulatory Context: FDA and Clinical Decision Support

Healthcare AI operates under some of the strictest regulatory requirements.
The FDA's framework for Software as a Medical Device (SaMD) determines what
level of oversight applies based on the system's intended use and risk.

```python
"""
FDA Framework for Clinical Decision Support (CDS)

The FDA distinguishes between CDS that IS and IS NOT a medical device:

NOT a device (exempt from FDA oversight) if ALL four criteria are met:
  1. Not intended to acquire, process, or analyze a medical image/signal
  2. Intended for displaying, analyzing, or printing medical information
  3. Intended for use by healthcare professionals
  4. Intended to enable the HCP to independently review the basis for
     the recommendation (i.e., the system is TRANSPARENT)

IS a device (requires FDA clearance/approval) if any criterion fails.

KEY INSIGHT: Criterion 4 creates a DIRECT LINK between interpretability
and regulatory status. A black-box AI system that provides recommendations
without explanation IS a medical device. The same system with transparent
explanations MAY be exempt.

This means interpretability can literally change a product's
regulatory classification — saving millions in compliance costs
and years in time-to-market.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class FDAClassification(Enum):
    """FDA classification outcomes for clinical decision support."""
    NOT_DEVICE = "not_a_device"           # CDS criteria met — exempt
    CLASS_I = "class_i_device"            # Low risk, general controls
    CLASS_II = "class_ii_device"          # Moderate risk, 510(k) clearance
    CLASS_III = "class_iii_device"        # High risk, PMA approval
    DE_NOVO = "de_novo"                   # Novel, no predicate device


@dataclass
class CDSAssessment:
    """Assess whether a clinical AI system meets the CDS exemption criteria.

    Each of the four criteria is evaluated independently.
    ALL must be True for the system to be exempt from device regulation.
    """
    system_name: str
    description: str

    # Criterion 1: Does NOT acquire/process/analyze medical image or signal
    does_not_process_images: bool

    # Criterion 2: Displays/analyzes/prints medical information
    displays_information: bool

    # Criterion 3: Intended for healthcare professionals
    for_hcp_use: bool

    # Criterion 4: HCP can independently review the basis
    # THIS IS THE INTERPRETABILITY CRITERION
    provides_transparent_basis: bool

    explanation_method: Optional[str] = None

    def is_exempt(self) -> bool:
        """Check if all four CDS criteria are met."""
        return (
            self.does_not_process_images
            and self.displays_information
            and self.for_hcp_use
            and self.provides_transparent_basis
        )

    def classify(self) -> FDAClassification:
        """Determine FDA classification.

        Simplified classification logic:
        - If CDS criteria met: not a device
        - If processes images: likely Class II+ (depends on indication)
        - Otherwise: Class II (typical for CDS that fails criterion 4)
        """
        if self.is_exempt():
            return FDAClassification.NOT_DEVICE
        if not self.does_not_process_images:
            return FDAClassification.CLASS_II  # Simplified; could be III
        return FDAClassification.CLASS_II

    def report(self) -> str:
        """Generate an assessment report."""
        lines = [
            f"FDA CDS Assessment: {self.system_name}",
            f"  Description: {self.description}",
            f"",
            f"  Criterion 1 (no image/signal processing): {self.does_not_process_images}",
            f"  Criterion 2 (displays information):       {self.displays_information}",
            f"  Criterion 3 (for HCP use):                {self.for_hcp_use}",
            f"  Criterion 4 (transparent basis):          {self.provides_transparent_basis}",
            f"",
            f"  Exempt from device regulation: {self.is_exempt()}",
            f"  Classification: {self.classify().value}",
        ]
        if self.explanation_method:
            lines.append(f"  Explanation method: {self.explanation_method}")
        return "\n".join(lines)


# Example assessments
examples = [
    CDSAssessment(
        system_name="Drug Interaction Checker",
        description="Checks patient medications for known interactions using a rules database",
        does_not_process_images=True,
        displays_information=True,
        for_hcp_use=True,
        provides_transparent_basis=True,  # Shows which drugs interact and why
        explanation_method="Rule-based: shows specific interaction rules triggered",
    ),
    CDSAssessment(
        system_name="Chest X-ray Pneumonia Detector",
        description="CNN that classifies chest X-rays as pneumonia-positive/negative",
        does_not_process_images=False,  # FAILS criterion 1 — processes images
        displays_information=True,
        for_hcp_use=True,
        provides_transparent_basis=True,  # GradCAM highlights
        explanation_method="GradCAM activation maps overlaid on radiograph",
    ),
    CDSAssessment(
        system_name="Sepsis Risk Score (Black-box)",
        description="Deep learning model predicting 48-hour sepsis onset from vitals + labs",
        does_not_process_images=True,
        displays_information=True,
        for_hcp_use=True,
        provides_transparent_basis=False,  # FAILS criterion 4 — no explanation
        explanation_method=None,
    ),
    CDSAssessment(
        system_name="Sepsis Risk Score (Explainable)",
        description="Same model, but with SHAP explanations showing which vitals drive the risk",
        does_not_process_images=True,
        displays_information=True,
        for_hcp_use=True,
        provides_transparent_basis=True,  # PASSES with SHAP explanations
        explanation_method="SHAP feature importance for each patient prediction",
    ),
]

print("FDA CLINICAL DECISION SUPPORT ASSESSMENTS")
print("=" * 65)
for ex in examples:
    print(f"\n{ex.report()}")
    print(f"{'─' * 65}")
```

### 1.2 GradCAM for Medical Imaging

GradCAM is the most widely used interpretability method for medical imaging AI.
It produces visual explanations that clinicians can evaluate against their domain
expertise.

```python
"""
GradCAM for Medical Imaging: Chest X-ray Interpretation

GradCAM produces a heatmap showing which regions of an image
the model focused on for its prediction. In medical imaging:

1. The heatmap should highlight CLINICALLY RELEVANT regions
   (e.g., lung fields for pneumonia, not image artifacts)

2. Misaligned attention (model looking at wrong region) is a
   RED FLAG for relying on spurious correlations

3. Clinicians use GradCAM to BUILD TRUST by verifying the model
   is "looking at the right thing"

IMPORTANT CAVEAT:
  GradCAM shows where the model looks, NOT why it decides.
  A model can look at the right region for the wrong reason.
  GradCAM is necessary but not sufficient for clinical trust.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleChestXrayModel(nn.Module):
    """Simplified CNN for chest X-ray classification.

    Architecture mimics a lightweight model suitable for
    GradCAM demonstration. In production, you'd use a
    pretrained model (DenseNet-121, ResNet-50) fine-tuned
    on medical imaging data.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1: 1 -> 32 channels
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 64 -> 128 channels (target layer for GradCAM)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Global average pooling + classifier
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        features = self.features(x)
        pooled = self.gap(features).flatten(1)
        logits = self.classifier(pooled)
        return logits


class MedicalGradCAM:
    """GradCAM implementation tailored for medical imaging.

    Key differences from generic GradCAM:
    1. Supports single-channel (grayscale) images
    2. Includes clinical confidence thresholding
    3. Returns both raw heatmap and clinical overlay
    4. Computes spatial statistics (centroid, spread) for reporting
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._gradients = None
        self._activations = None

        # Register hooks to capture gradients and activations
        # WHY hooks: GradCAM needs the gradient of the output w.r.t.
        # the feature maps. Hooks let us capture these during backprop
        # without modifying the model's forward method.
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """Hook to save feature map activations during forward pass."""
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save gradients during backward pass."""
        self._gradients = grad_output[0].detach()

    def generate(self, image: torch.Tensor, target_class: int = None) -> dict:
        """Generate GradCAM heatmap for a medical image.

        Args:
            image: Input image tensor, shape (1, 1, H, W) for grayscale
            target_class: Class to explain (default: predicted class)

        Returns:
            Dictionary with heatmap, prediction, confidence, and spatial stats

        WHY we return spatial statistics:
        In medical imaging, LOCATION matters. A heatmap centered on
        the right lung lower lobe is clinically meaningful for
        pneumonia diagnosis. Spatial stats quantify this.
        """
        self.model.eval()

        # Forward pass
        logits = self.model(image)
        probs = F.softmax(logits, dim=1)

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        confidence = probs[0, target_class].item()

        # Backward pass for target class
        self.model.zero_grad()
        logits[0, target_class].backward()

        # Compute GradCAM weights
        # Global average pool the gradients to get importance per channel
        # WHY global average: each channel detects a different feature
        # (e.g., edges, textures). The gradient magnitude tells us
        # which features (channels) matter most for this prediction.
        weights = self._gradients.mean(dim=[2, 3], keepdim=True)

        # Weighted combination of feature maps
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)  # Only positive contributions

        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Upsample to input resolution
        cam_upsampled = F.interpolate(
            cam,
            size=image.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        heatmap = cam_upsampled.squeeze().numpy()

        # Compute spatial statistics for clinical reporting
        # These help quantify WHERE the model is focusing
        y_coords, x_coords = np.mgrid[:heatmap.shape[0], :heatmap.shape[1]]
        total_activation = heatmap.sum()

        if total_activation > 0:
            centroid_y = (y_coords * heatmap).sum() / total_activation
            centroid_x = (x_coords * heatmap).sum() / total_activation
            # Spread: weighted standard deviation of activated region
            spread_y = np.sqrt(((y_coords - centroid_y)**2 * heatmap).sum() / total_activation)
            spread_x = np.sqrt(((x_coords - centroid_x)**2 * heatmap).sum() / total_activation)
        else:
            centroid_y = centroid_x = spread_y = spread_x = 0.0

        # Fraction of image that is "active" (above threshold)
        active_fraction = (heatmap > 0.3).mean()

        return {
            "heatmap": heatmap,
            "predicted_class": target_class,
            "confidence": confidence,
            "spatial_stats": {
                "centroid": (float(centroid_y), float(centroid_x)),
                "spread": (float(spread_y), float(spread_x)),
                "active_fraction": float(active_fraction),
            },
        }


# Demonstrate GradCAM for medical imaging
torch.manual_seed(42)

# Create model and synthetic "chest X-ray"
model = SimpleChestXrayModel(num_classes=2)
# Synthetic grayscale image (224x224, single channel)
image = torch.randn(1, 1, 224, 224)

# Target last conv layer for GradCAM
target_layer = model.features[-1]  # Last ReLU
gradcam = MedicalGradCAM(model, target_layer)

# Generate explanation
result = gradcam.generate(image)

print("MEDICAL IMAGING GradCAM RESULT")
print("=" * 50)
print(f"Predicted class: {result['predicted_class']} "
      f"({'pneumonia' if result['predicted_class'] == 1 else 'normal'})")
print(f"Confidence: {result['confidence']:.3f}")
print(f"Heatmap shape: {result['heatmap'].shape}")
print(f"\nSpatial Statistics:")
stats = result["spatial_stats"]
print(f"  Centroid (y, x): ({stats['centroid'][0]:.1f}, {stats['centroid'][1]:.1f})")
print(f"  Spread (y, x): ({stats['spread'][0]:.1f}, {stats['spread'][1]:.1f})")
print(f"  Active fraction: {stats['active_fraction']:.3f}")
```

### 1.3 Concept-Based Explanations for Diagnosis

```python
"""
Concept-Based Explanations for Clinical Diagnosis

Instead of explaining in terms of pixels (GradCAM) or raw features,
concept-based methods explain in terms of CLINICAL CONCEPTS that
doctors already understand.

Example for skin lesion classification:
  "This lesion is classified as melanoma because:
   - Asymmetry score: HIGH (sensitivity 0.89)
   - Border irregularity: PRESENT (confidence 0.92)
   - Color variation: HIGH (multiple colors detected)
   - Diameter: >6mm
   These correspond to the ABCD clinical criteria."

WHY concept-based > pixel-based for healthcare:
  1. Clinicians think in concepts, not pixels
  2. Concepts are ACTIONABLE (doctor can verify asymmetry)
  3. Concepts map to established clinical frameworks (ABCD, BIRADS)
  4. Concept errors are detectable by domain experts
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ClinicalConcept:
    """A clinical concept used in medical diagnosis.

    Each concept is:
    - Defined by clinical experts (not learned from data)
    - Measurable or observable in patient data
    - Part of an established clinical framework
    """
    name: str
    description: str
    measurement_method: str
    value: float                  # Current observation (0-1 scale)
    threshold: float              # Clinical significance threshold
    clinical_framework: str       # Which clinical framework uses this

    @property
    def is_significant(self) -> bool:
        """Whether this concept exceeds the clinical threshold."""
        return self.value >= self.threshold

    @property
    def significance_label(self) -> str:
        """Human-readable significance."""
        if self.value >= 0.8:
            return "HIGH"
        elif self.value >= self.threshold:
            return "MODERATE"
        else:
            return "LOW"


@dataclass
class ConceptBasedExplanation:
    """Explanation for a clinical prediction based on clinical concepts.

    This explanation format is designed for CLINICIAN consumption:
    it maps model predictions to clinical concepts that doctors
    already use in their diagnostic reasoning.
    """
    prediction: str
    confidence: float
    concepts: list[ClinicalConcept]
    clinical_summary: str
    recommended_action: str

    def generate_clinician_report(self) -> str:
        """Generate a report suitable for clinician review.

        Format follows clinical reporting conventions:
        finding, interpretation, confidence, recommendation.
        """
        lines = [
            f"AI-ASSISTED CLINICAL ASSESSMENT",
            f"{'=' * 50}",
            f"Assessment: {self.prediction}",
            f"AI Confidence: {self.confidence:.0%}",
            f"",
            f"CONTRIBUTING FACTORS:",
        ]

        significant = [c for c in self.concepts if c.is_significant]
        non_significant = [c for c in self.concepts if not c.is_significant]

        if significant:
            lines.append("\n  Significant findings:")
            for c in sorted(significant, key=lambda x: -x.value):
                lines.append(
                    f"    - {c.name}: {c.significance_label} "
                    f"(score: {c.value:.2f}, threshold: {c.threshold:.2f})"
                )
                lines.append(f"      Method: {c.measurement_method}")

        if non_significant:
            lines.append("\n  Non-significant findings:")
            for c in non_significant:
                lines.append(
                    f"    - {c.name}: {c.significance_label} "
                    f"(score: {c.value:.2f})"
                )

        lines.extend([
            f"",
            f"CLINICAL SUMMARY:",
            f"  {self.clinical_summary}",
            f"",
            f"RECOMMENDED ACTION:",
            f"  {self.recommended_action}",
            f"",
            f"NOTE: This is an AI-generated assessment. Final clinical",
            f"decision must be made by the treating physician.",
        ])

        return "\n".join(lines)


# Example: Skin lesion classification with ABCD criteria
skin_explanation = ConceptBasedExplanation(
    prediction="Suspicious for melanoma",
    confidence=0.87,
    concepts=[
        ClinicalConcept(
            name="Asymmetry",
            description="Degree of asymmetry in lesion shape",
            measurement_method="Computed from segmented lesion boundary using PCA",
            value=0.82,
            threshold=0.5,
            clinical_framework="ABCD Dermatoscopy Rule",
        ),
        ClinicalConcept(
            name="Border Irregularity",
            description="Irregularity of lesion border",
            measurement_method="Fractal dimension of segmented boundary",
            value=0.75,
            threshold=0.6,
            clinical_framework="ABCD Dermatoscopy Rule",
        ),
        ClinicalConcept(
            name="Color Variation",
            description="Number and distribution of colors within lesion",
            measurement_method="Color histogram analysis (6-color model)",
            value=0.91,
            threshold=0.5,
            clinical_framework="ABCD Dermatoscopy Rule",
        ),
        ClinicalConcept(
            name="Diameter",
            description="Maximum diameter of lesion",
            measurement_method="Calibrated measurement from dermoscopic image",
            value=0.65,
            threshold=0.6,  # Corresponds to >6mm
            clinical_framework="ABCD Dermatoscopy Rule",
        ),
        ClinicalConcept(
            name="Blue-White Veil",
            description="Presence of blue-white structures",
            measurement_method="Color space analysis in blue-white region",
            value=0.35,
            threshold=0.5,
            clinical_framework="7-Point Checklist",
        ),
    ],
    clinical_summary=(
        "The lesion demonstrates high asymmetry, border irregularity, "
        "color variation, and diameter exceeding 6mm — meeting 4 of 4 "
        "ABCD criteria. These findings are consistent with melanoma risk."
    ),
    recommended_action=(
        "URGENT: Refer for dermatologist evaluation and consider excisional "
        "biopsy. Do not delay based on AI assessment alone."
    ),
)

print(skin_explanation.generate_clinician_report())
```

---

## 2. Financial AI Interpretability

### 2.1 ECOA Adverse Action Notices

The Equal Credit Opportunity Act (ECOA) requires lenders to provide specific
reasons when denying credit. This creates a legal mandate for interpretability.

```python
"""
ECOA Adverse Action Notices

Regulation B (12 CFR 1002.9) requires creditors to notify applicants
of adverse action and provide SPECIFIC REASONS for the denial.

KEY REQUIREMENTS:
1. Notice must be provided within 30 days of action
2. Must include the SPECIFIC principal reasons (up to 4)
3. Reasons must be drawn from an approved list (model-derived OK)
4. Must be understandable to a layperson

HOW INTERPRETABILITY FITS:
  The "principal reasons" must accurately reflect what the model
  actually used to make the decision. This means:
  - Feature importance methods (SHAP, LIME) map to adverse action reasons
  - The top-K important features become the adverse action reasons
  - Reasons must be phrased in consumer-understandable language

IMPORTANT: Generic reasons like "model score too low" do NOT comply.
The reasons must be SPECIFIC to the individual applicant.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# Standard adverse action reason codes (subset of industry codes)
# These are the human-readable versions of feature importance outputs
REASON_CODE_MAP = {
    "debt_ratio": {
        "code": "AR001",
        "consumer_text": "Amount owed on accounts is too high relative to income",
        "detail": "Your current debt obligations represent a high proportion of your reported income.",
    },
    "credit_history_length": {
        "code": "AR002",
        "consumer_text": "Length of credit history is too short",
        "detail": "The length of time your credit accounts have been established is shorter than required.",
    },
    "payment_history": {
        "code": "AR003",
        "consumer_text": "Recent payment history shows delinquencies",
        "detail": "Your payment records show one or more late or missed payments in the past 12 months.",
    },
    "num_recent_inquiries": {
        "code": "AR004",
        "consumer_text": "Too many recent credit inquiries",
        "detail": "The number of times your credit has been checked recently exceeds our criteria.",
    },
    "income": {
        "code": "AR005",
        "consumer_text": "Income is insufficient for the requested loan amount",
        "detail": "Your reported income does not meet the minimum threshold for the loan amount requested.",
    },
    "employment_duration": {
        "code": "AR006",
        "consumer_text": "Length of employment is insufficient",
        "detail": "The duration of your current employment is shorter than our minimum requirement.",
    },
    "loan_amount": {
        "code": "AR007",
        "consumer_text": "Requested loan amount exceeds lending guidelines",
        "detail": "The amount requested exceeds the maximum for your credit profile.",
    },
    "existing_credit_utilization": {
        "code": "AR008",
        "consumer_text": "Proportion of credit limits used is too high",
        "detail": "You are currently using a high percentage of your available credit across all accounts.",
    },
}


@dataclass
class AdverseActionNotice:
    """ECOA-compliant adverse action notice generated from model explanation.

    This notice is the legal artifact that the lender must provide.
    It translates model feature importance into consumer-understandable
    reasons for the credit denial.
    """
    applicant_id: str
    application_date: str
    decision: str
    principal_reasons: list[dict]  # Up to 4 reasons
    credit_score_used: Optional[float] = None
    notice_date: str = ""
    creditor_name: str = ""

    def generate_notice(self) -> str:
        """Generate the formal adverse action notice.

        This is the text that would be mailed or displayed to the applicant.
        Format follows Regulation B Appendix C sample forms.
        """
        lines = [
            f"NOTICE OF ADVERSE ACTION",
            f"{'=' * 50}",
            f"",
            f"Date: {self.notice_date}",
            f"From: {self.creditor_name}",
            f"Application Reference: {self.applicant_id}",
            f"Application Date: {self.application_date}",
            f"",
            f"Dear Applicant,",
            f"",
            f"We regret to inform you that your recent application for credit",
            f"has been {self.decision}.",
            f"",
        ]

        if self.credit_score_used is not None:
            lines.extend([
                f"Credit Score Information:",
                f"  Your credit score: {self.credit_score_used:.0f}",
                f"",
            ])

        lines.extend([
            f"PRINCIPAL REASONS FOR THIS DECISION:",
            f"",
        ])

        for i, reason in enumerate(self.principal_reasons, 1):
            lines.extend([
                f"  {i}. {reason['consumer_text']}",
                f"     {reason['detail']}",
                f"",
            ])

        lines.extend([
            f"YOUR RIGHTS:",
            f"  You have the right to:",
            f"  - Request the specific reasons for this decision (provided above)",
            f"  - Obtain a free copy of your credit report within 60 days",
            f"  - Dispute any inaccurate information on your credit report",
            f"  - Reapply for credit at any time",
            f"",
            f"EQUAL CREDIT OPPORTUNITY ACT NOTICE:",
            f"  The federal Equal Credit Opportunity Act prohibits creditors",
            f"  from discriminating against credit applicants on the basis of",
            f"  race, color, religion, national origin, sex, marital status,",
            f"  age, or because you receive public assistance.",
        ])

        return "\n".join(lines)


def generate_adverse_action_from_shap(
    applicant_id: str,
    shap_values: dict[str, float],
    max_reasons: int = 4,
) -> AdverseActionNotice:
    """Generate an adverse action notice from SHAP feature importance.

    This is the key integration point between interpretability and
    regulatory compliance. SHAP values are translated into legally
    required adverse action reasons.

    HOW THE TRANSLATION WORKS:
    1. Sort features by SHAP value (most negative = most harmful)
    2. Take the top-K most negative features
    3. Map each feature to its consumer-readable reason code
    4. Package as an AdverseActionNotice

    WHY SHAP (not other methods):
    SHAP provides SIGNED importance — we need to know which features
    HURT the application (negative SHAP). Unsigned importance (like
    tree feature_importances_) cannot distinguish helpful from harmful.
    """
    # Sort by SHAP value — most negative (most harmful) first
    sorted_features = sorted(shap_values.items(), key=lambda x: x[1])

    # Take up to max_reasons features with NEGATIVE contribution
    harmful_features = [
        (name, value) for name, value in sorted_features if value < 0
    ][:max_reasons]

    # Map to reason codes
    principal_reasons = []
    for feature_name, shap_value in harmful_features:
        if feature_name in REASON_CODE_MAP:
            reason = REASON_CODE_MAP[feature_name].copy()
            reason["shap_value"] = round(shap_value, 4)
            reason["feature_name"] = feature_name
            principal_reasons.append(reason)

    return AdverseActionNotice(
        applicant_id=applicant_id,
        application_date="2024-06-01",
        decision="denied",
        principal_reasons=principal_reasons,
        credit_score_used=620,
        notice_date="2024-06-15",
        creditor_name="Example Bank, N.A.",
    )


# Example: Generate adverse action notice
# Simulated SHAP values (negative = hurts application)
shap_values = {
    "debt_ratio": -0.25,              # Strongest negative factor
    "payment_history": -0.18,         # Second strongest
    "income": -0.12,                  # Third
    "num_recent_inquiries": -0.08,    # Fourth
    "credit_history_length": 0.05,    # Slightly positive
    "employment_duration": 0.15,      # Positive
    "existing_credit_utilization": -0.03,  # Slightly negative
}

notice = generate_adverse_action_from_shap("APP-2024-001234", shap_values)
print(notice.generate_notice())
```

### 2.2 SR 11-7 Model Risk Management

```python
"""
Federal Reserve SR 11-7: Guidance on Model Risk Management

SR 11-7 (April 2011) is the foundational US regulatory guidance
for model risk management in banking. It applies to ALL models
used in banking decisions, including ML models.

THREE PILLARS:
1. MODEL DEVELOPMENT: Sound design, theory, and testing
2. MODEL VALIDATION: Independent challenge and verification
3. MODEL GOVERNANCE: Board oversight, policies, and controls

KEY REQUIREMENTS FOR ML/AI:
- Models must have a "conceptual soundness" that can be explained
- Outcomes analysis must be ongoing
- Documentation must be comprehensive
- Validation must be independent from development

WHY SR 11-7 matters for interpretability:
  "Conceptual soundness" requires that the model's approach can be
  EXPLAINED and JUSTIFIED. A pure black-box model with no
  interpretability may fail this requirement.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class ModelTier(Enum):
    """SR 11-7 model tiering based on risk.

    Tier determines the depth of validation and documentation required.
    """
    TIER_1 = "tier_1"  # Critical: directly impacts significant financial decisions
    TIER_2 = "tier_2"  # Significant: material impact on business decisions
    TIER_3 = "tier_3"  # Minor: limited impact, used for analysis/reporting


@dataclass
class SR117ModelInventoryEntry:
    """Entry in the model inventory per SR 11-7.

    Every model used in a banking organization must be registered
    in a model inventory. This entry captures the minimum information
    required for risk management oversight.
    """
    model_id: str
    model_name: str
    model_type: str              # "ML/AI", "Statistical", "Expert System", etc.
    business_use: str
    model_tier: ModelTier
    model_owner: str
    last_validation_date: str
    next_validation_date: str
    interpretability_method: str  # Required for ML models
    documentation_status: str

    # SR 11-7 specific assessments
    conceptual_soundness: str     # Can the model approach be explained?
    outcomes_analysis: str        # Are predictions monitored over time?
    stability_analysis: str       # Is the model stable over time?

    def validation_requirements(self) -> list[str]:
        """Return validation requirements based on model tier.

        Higher-tier models require more frequent and thorough validation.
        """
        base = [
            "Conceptual soundness review",
            "Outcomes analysis (back-testing)",
            "Sensitivity analysis",
            "Documentation review",
        ]

        if self.model_tier == ModelTier.TIER_1:
            base.extend([
                "Independent replicate model (challenger model)",
                "Stress testing under adverse scenarios",
                "Board-level reporting",
                "Quarterly performance monitoring",
                "Annual comprehensive validation",
            ])
        elif self.model_tier == ModelTier.TIER_2:
            base.extend([
                "Semi-annual performance monitoring",
                "Biennial comprehensive validation",
                "Senior management reporting",
            ])
        else:  # TIER_3
            base.extend([
                "Annual performance monitoring",
                "Triennial comprehensive validation",
            ])

        return base

    def report(self) -> str:
        """Generate SR 11-7 model inventory report."""
        reqs = self.validation_requirements()
        req_str = "\n    ".join(f"- {r}" for r in reqs)
        return (
            f"SR 11-7 MODEL INVENTORY ENTRY\n"
            f"{'=' * 50}\n"
            f"Model ID: {self.model_id}\n"
            f"Name: {self.model_name}\n"
            f"Type: {self.model_type}\n"
            f"Tier: {self.model_tier.value}\n"
            f"Owner: {self.model_owner}\n"
            f"\nBusiness Use: {self.business_use}\n"
            f"\nInterpretability: {self.interpretability_method}\n"
            f"Conceptual Soundness: {self.conceptual_soundness}\n"
            f"Outcomes Analysis: {self.outcomes_analysis}\n"
            f"Stability: {self.stability_analysis}\n"
            f"\nValidation Requirements:\n    {req_str}\n"
        )


# Example inventory entry
credit_model_entry = SR117ModelInventoryEntry(
    model_id="MOD-2024-0042",
    model_name="Consumer Credit Score v3.2",
    model_type="ML/AI — Gradient Boosted Trees",
    business_use="Consumer loan approval decisions (EUR 1K-50K)",
    model_tier=ModelTier.TIER_1,
    model_owner="Credit Risk Analytics Team",
    last_validation_date="2024-03-15",
    next_validation_date="2025-03-15",
    interpretability_method=(
        "SHAP (TreeExplainer) for individual explanations; "
        "global feature importance for model-level understanding"
    ),
    documentation_status="Complete — last updated 2024-03-15",
    conceptual_soundness=(
        "Model uses established credit risk features (DTI, payment history, "
        "credit utilization). Gradient boosting is well-understood for "
        "tabular prediction. SHAP explanations verified against domain "
        "knowledge by credit analysts."
    ),
    outcomes_analysis=(
        "Monthly: predicted vs. actual default rates by decile. "
        "Model discrimination (Gini) monitored continuously. "
        "Current Gini: 0.65 (within acceptable range 0.55-0.75)."
    ),
    stability_analysis=(
        "PSI (Population Stability Index) computed monthly on input "
        "feature distributions. All features PSI < 0.1 (stable). "
        "Model retrained quarterly to prevent drift."
    ),
)

print(credit_model_entry.report())
```

---

## 3. NLP Interpretability

### 3.1 Token-Level SHAP for Text Classification

```python
"""
Token-Level SHAP for NLP Models

In NLP, explanations operate at the TOKEN level: which words or
subwords contributed most to the model's prediction?

KEY CHALLENGES:
1. Vocabulary size: SHAP must consider 30K-100K features (tokens)
2. Interactions: word meaning depends heavily on context
3. Subword tokens: "unhappy" -> "un" + "happy" (attribution must be combined)
4. Faithfulness: do token highlights actually reflect model reasoning?

APPROACHES:
1. Partition SHAP: group tokens into segments, attribute by segment
2. Gradient-based: use input gradients (fast, but may be unfaithful)
3. Attention-based: use attention weights (controversial — see L04)
4. Occlusion/leave-one-out: mask tokens individually (slow, faithful)
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class TokenAttribution:
    """Attribution score for a single token."""
    token: str
    position: int
    attribution: float       # Signed: positive = supports prediction
    is_special: bool = False  # [CLS], [SEP], [PAD]

    @property
    def direction(self) -> str:
        """Whether this token supports or opposes the prediction."""
        if self.attribution > 0.01:
            return "supports"
        elif self.attribution < -0.01:
            return "opposes"
        else:
            return "neutral"


@dataclass
class TextExplanation:
    """Complete explanation for a text prediction."""
    text: str
    tokens: list[str]
    attributions: list[TokenAttribution]
    prediction: str
    confidence: float
    method: str

    def highlight_text(self, top_k: int = 5) -> str:
        """Generate highlighted text showing top-k contributing tokens.

        Uses simple text formatting to show which tokens matter most.
        In a UI, these would be highlighted with color gradients.
        """
        # Sort by absolute attribution
        sorted_attrs = sorted(
            self.attributions,
            key=lambda a: abs(a.attribution),
            reverse=True,
        )

        top_tokens = set()
        for attr in sorted_attrs[:top_k]:
            if not attr.is_special:
                top_tokens.add(attr.position)

        # Build highlighted text
        parts = []
        for attr in self.attributions:
            if attr.is_special:
                continue
            if attr.position in top_tokens:
                if attr.attribution > 0:
                    parts.append(f"[+{attr.token}+]")
                else:
                    parts.append(f"[-{attr.token}-]")
            else:
                parts.append(attr.token)

        return " ".join(parts)

    def summary(self) -> str:
        """Generate a text summary of the explanation."""
        # Get top positive and negative tokens
        non_special = [a for a in self.attributions if not a.is_special]
        positive = sorted(
            [a for a in non_special if a.attribution > 0],
            key=lambda a: -a.attribution,
        )[:3]
        negative = sorted(
            [a for a in non_special if a.attribution < 0],
            key=lambda a: a.attribution,
        )[:3]

        lines = [
            f"Prediction: {self.prediction} (confidence: {self.confidence:.2f})",
            f"Method: {self.method}",
            f"",
            f"Words SUPPORTING the prediction:",
        ]
        for a in positive:
            lines.append(f'  "{a.token}" (attribution: {a.attribution:+.3f})')

        lines.append(f"\nWords OPPOSING the prediction:")
        for a in negative:
            lines.append(f'  "{a.token}" (attribution: {a.attribution:+.3f})')

        lines.append(f"\nHighlighted text:")
        lines.append(f"  {self.highlight_text()}")

        return "\n".join(lines)


class OcclusionExplainer:
    """Leave-one-out token attribution for NLP models.

    Occlusion is the simplest and most FAITHFUL token attribution method:
    - Remove each token one at a time
    - Measure the change in prediction
    - Attribution = prediction_with_token - prediction_without_token

    WHY occlusion:
    Unlike gradient or attention methods, occlusion directly measures
    causal impact: "what happens if this token is removed?" This is
    the closest to a true causal explanation.

    LIMITATIONS:
    - Slow: O(n) forward passes where n = number of tokens
    - Ignores interactions: removing "not" from "not good" loses the negation
    - Mask token choice matters: replacing with [MASK], [PAD], or random token
      gives different results
    """

    def __init__(self, predict_fn, tokenizer_fn):
        """
        predict_fn: function(text) -> dict with 'label' and 'score'
        tokenizer_fn: function(text) -> list of tokens
        """
        self.predict_fn = predict_fn
        self.tokenizer_fn = tokenizer_fn

    def explain(self, text: str) -> TextExplanation:
        """Generate token-level explanations using occlusion.

        For each token position:
        1. Create a version of the text with that token removed
        2. Get the model's prediction on the modified text
        3. Attribution = original_score - modified_score
           (positive = removing the token hurts the prediction)
        """
        tokens = self.tokenizer_fn(text)
        original_pred = self.predict_fn(text)
        original_score = original_pred["score"]

        attributions = []
        for i, token in enumerate(tokens):
            # Create text with token i removed
            modified_tokens = tokens[:i] + tokens[i+1:]
            modified_text = " ".join(modified_tokens)

            if modified_text.strip():
                modified_pred = self.predict_fn(modified_text)
                modified_score = modified_pred["score"]
            else:
                modified_score = 0.5  # No text -> neutral

            # Attribution: how much does removing this token change the score?
            attribution = original_score - modified_score

            attributions.append(TokenAttribution(
                token=token,
                position=i,
                attribution=round(attribution, 4),
                is_special=token in ["[CLS]", "[SEP]", "[PAD]"],
            ))

        return TextExplanation(
            text=text,
            tokens=tokens,
            attributions=attributions,
            prediction=original_pred["label"],
            confidence=original_pred["score"],
            method="occlusion (leave-one-out)",
        )


# Example with a simple sentiment model
def simple_sentiment_model(text: str) -> dict:
    """Simple keyword-based sentiment model for demonstration.

    In production, this would be a transformer model.
    We use a simple model here so the explanations are interpretable.
    """
    positive_words = {"great", "excellent", "amazing", "love", "wonderful", "good", "best", "happy"}
    negative_words = {"terrible", "awful", "horrible", "hate", "bad", "worst", "poor", "disappointing"}

    words = text.lower().split()
    pos_count = sum(1 for w in words if w in positive_words)
    neg_count = sum(1 for w in words if w in negative_words)
    total = len(words)

    if total == 0:
        return {"label": "neutral", "score": 0.5}

    score = 0.5 + 0.2 * (pos_count - neg_count)
    score = max(0.0, min(1.0, score))

    label = "positive" if score > 0.5 else "negative" if score < 0.5 else "neutral"
    return {"label": label, "score": score}


def simple_tokenizer(text: str) -> list[str]:
    """Simple whitespace tokenizer."""
    return text.split()


# Generate explanations
explainer = OcclusionExplainer(simple_sentiment_model, simple_tokenizer)

texts = [
    "The product was great but the service was terrible",
    "I love this amazing and wonderful experience",
    "The quality is poor and the design is disappointing",
]

print("TOKEN-LEVEL NLP EXPLANATIONS")
print("=" * 60)
for text in texts:
    explanation = explainer.explain(text)
    print(f"\n{explanation.summary()}")
    print(f"{'─' * 60}")
```

### 3.2 Rationale Extraction (ERASER Benchmark)

```python
"""
Rationale Extraction for NLP

Rather than attributing importance to EVERY token, rationale extraction
identifies a MINIMAL SUBSET of tokens that is sufficient to justify
the prediction. This is inspired by how humans explain text decisions:
they highlight the key passage, not every word.

ERASER Benchmark (DeYoung et al., 2020):
  Evaluating Rationale And Simple English Reasoning
  Provides datasets with human-annotated rationales for evaluation.

Key papers:
  - Lei et al. (2016): "Rationalizing Neural Predictions"
    Uses a generator-encoder architecture where the generator selects
    tokens and the encoder predicts from only those tokens.

  - DeYoung et al. (2020): ERASER benchmark
    Standardized evaluation of rationale quality: faithfulness
    (does the rationale actually drive the prediction?) and
    plausibility (does it match human reasoning?).

TWO TYPES OF RATIONALES:
1. EXTRACTIVE: Select a subset of input tokens as the rationale
   (like highlighting with a marker)
2. ABSTRACTIVE: Generate new text that explains the prediction
   (like writing a summary)

Extractive rationales are preferred for faithfulness because they
directly reference the input text.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Rationale:
    """An extracted rationale — a subset of tokens justifying a prediction."""
    tokens: list[str]            # The full input tokens
    selected_indices: list[int]  # Which positions are in the rationale
    prediction_from_full: str    # Prediction using all tokens
    prediction_from_rationale: str  # Prediction using ONLY rationale tokens
    confidence_full: float
    confidence_rationale: float

    @property
    def rationale_tokens(self) -> list[str]:
        """The tokens that form the rationale."""
        return [self.tokens[i] for i in self.selected_indices]

    @property
    def rationale_text(self) -> str:
        """The rationale as a text span."""
        return " ".join(self.rationale_tokens)

    @property
    def compression_ratio(self) -> float:
        """Fraction of tokens selected as rationale."""
        return len(self.selected_indices) / len(self.tokens)

    @property
    def sufficiency(self) -> float:
        """How well the rationale alone supports the prediction.

        Sufficiency = 1 if rationale gives same prediction as full text.
        Measures whether the rationale CONTAINS enough information.
        """
        return self.confidence_rationale / max(self.confidence_full, 1e-10)

    def faithfulness_score(self, predict_fn) -> float:
        """Compute faithfulness: does removing the rationale change prediction?

        Faithfulness = P(full text) - P(text without rationale)
        High faithfulness means the rationale IS what drives the prediction.

        WHY faithfulness matters:
        A rationale that is plausible but not faithful is dangerous —
        it gives users a false sense of understanding.
        """
        # Prediction without rationale (complement)
        complement_indices = [
            i for i in range(len(self.tokens))
            if i not in self.selected_indices
        ]
        complement_text = " ".join(self.tokens[i] for i in complement_indices)

        if complement_text.strip():
            complement_pred = predict_fn(complement_text)
            return self.confidence_full - complement_pred["score"]
        else:
            return self.confidence_full  # Removing rationale = no text


def extract_rationale_greedy(
    text: str,
    predict_fn,
    tokenizer_fn,
    target_compression: float = 0.3,
) -> Rationale:
    """Greedy rationale extraction.

    Algorithm:
    1. Start with all tokens selected
    2. Iteratively remove the token whose removal causes the LEAST
       change in prediction confidence
    3. Stop when we've removed enough tokens to reach target compression

    This is a simple baseline. More sophisticated methods (Lei et al.)
    use learned generators. But greedy extraction is interpretable
    itself — you can explain WHY each token was kept.

    WHY greedy instead of optimal:
    Finding the optimal rationale is NP-hard (subset selection).
    Greedy gives a good approximation in O(n^2) time.
    """
    tokens = tokenizer_fn(text)
    full_pred = predict_fn(text)

    # Start with all tokens selected
    selected = list(range(len(tokens)))
    n_to_remove = int(len(tokens) * (1 - target_compression))

    for _ in range(n_to_remove):
        if len(selected) <= 1:
            break

        # Try removing each selected token
        min_impact = float("inf")
        best_removal = None

        for idx in selected:
            # Create text without this token
            remaining = [i for i in selected if i != idx]
            remaining_text = " ".join(tokens[i] for i in remaining)

            if remaining_text.strip():
                pred = predict_fn(remaining_text)
                impact = abs(full_pred["score"] - pred["score"])
            else:
                impact = float("inf")  # Don't remove last token

            if impact < min_impact:
                min_impact = impact
                best_removal = idx

        if best_removal is not None:
            selected.remove(best_removal)

    # Get prediction from rationale only
    rationale_text = " ".join(tokens[i] for i in selected)
    rationale_pred = predict_fn(rationale_text)

    return Rationale(
        tokens=tokens,
        selected_indices=sorted(selected),
        prediction_from_full=full_pred["label"],
        prediction_from_rationale=rationale_pred["label"],
        confidence_full=full_pred["score"],
        confidence_rationale=rationale_pred["score"],
    )


# Extract rationales for example texts
print("RATIONALE EXTRACTION")
print("=" * 60)

texts = [
    "The movie had amazing cinematography but the plot was terrible and confusing",
    "I absolutely love this restaurant the food is excellent and the service is wonderful",
]

for text in texts:
    rationale = extract_rationale_greedy(
        text, simple_sentiment_model, simple_tokenizer,
        target_compression=0.4,
    )

    faith = rationale.faithfulness_score(simple_sentiment_model)

    print(f"\nFull text: {text}")
    print(f"Prediction: {rationale.prediction_from_full} ({rationale.confidence_full:.2f})")
    print(f"\nExtracted rationale: {rationale.rationale_text}")
    print(f"Rationale prediction: {rationale.prediction_from_rationale} ({rationale.confidence_rationale:.2f})")
    print(f"Compression: {rationale.compression_ratio:.0%} of tokens selected")
    print(f"Sufficiency: {rationale.sufficiency:.2f}")
    print(f"Faithfulness: {faith:.3f}")
    print(f"{'─' * 60}")
```

---

## 4. Computer Vision Interpretability

### 4.1 Pixel Attribution Comparison

```python
"""
Pixel Attribution Methods Comparison for Computer Vision

Different attribution methods highlight different aspects of an image.
Comparing them reveals what the model is actually learning.

METHODS COMPARED:
1. Vanilla Gradient: gradient of output w.r.t. input pixels
2. Integrated Gradients: path integral from baseline to input
3. GradCAM: class-discriminative, low-resolution
4. SmoothGrad: gradient with noise averaging

WHEN TO USE WHICH:
- GradCAM: Quick visual check, clinician-facing, low resolution OK
- Integrated Gradients: When mathematical rigor is needed (axiom-satisfying)
- SmoothGrad: When vanilla gradients are too noisy
- Vanilla Gradient: Baseline comparison only (usually too noisy)

AGREEMENT ANALYSIS:
When methods agree: HIGH confidence in the explanation
When methods disagree: investigate further — model may be using shortcuts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field


@dataclass
class AttributionResult:
    """Result of a single attribution method."""
    method_name: str
    attribution_map: np.ndarray  # Shape: (H, W)
    computation_time_ms: float

    @property
    def top_k_fraction(self) -> float:
        """Fraction of total attribution in top 10% of pixels.

        High concentration = model focuses on specific region.
        Low concentration = model uses diffuse information.
        """
        flat = np.abs(self.attribution_map.flatten())
        threshold = np.percentile(flat, 90)
        top_sum = flat[flat >= threshold].sum()
        total_sum = flat.sum()
        if total_sum == 0:
            return 0.0
        return float(top_sum / total_sum)


def compute_vanilla_gradient(model, image, target_class):
    """Compute vanilla gradient attribution.

    The simplest attribution method: take the gradient of the output
    with respect to the input pixels.

    WHY this is often not enough:
    Vanilla gradients are locally linear approximations. They can be
    noisy and may not capture the model's global behavior.
    But they serve as a useful baseline for comparison.
    """
    import time
    start = time.time()

    image_var = image.clone().detach().requires_grad_(True)
    model.zero_grad()

    output = model(image_var)
    output[0, target_class].backward()

    gradient = image_var.grad.data.squeeze().numpy()

    # Aggregate across channels if multi-channel
    if gradient.ndim == 3:
        gradient = np.abs(gradient).max(axis=0)

    elapsed = (time.time() - start) * 1000

    return AttributionResult(
        method_name="Vanilla Gradient",
        attribution_map=gradient,
        computation_time_ms=elapsed,
    )


def compute_integrated_gradients(model, image, target_class, steps=50):
    """Compute Integrated Gradients (Sundararajan et al., 2017).

    IG satisfies two desirable axioms:
    1. SENSITIVITY: if changing a feature changes the output, it gets non-zero attribution
    2. IMPLEMENTATION INVARIANCE: functionally identical models give same attributions

    Algorithm:
    1. Define a baseline (black image)
    2. Interpolate between baseline and input in 'steps' steps
    3. Compute gradient at each step
    4. Average the gradients
    5. Multiply by (input - baseline)

    WHY baseline matters:
    The baseline represents "absence of information." For images,
    a black image is standard. For medical images, the background
    intensity may be more appropriate.
    """
    import time
    start = time.time()

    baseline = torch.zeros_like(image)
    scaled_inputs = [
        baseline + (float(i) / steps) * (image - baseline)
        for i in range(steps + 1)
    ]

    # Compute gradients at each interpolation step
    gradients = []
    for scaled_input in scaled_inputs:
        scaled_input = scaled_input.clone().detach().requires_grad_(True)
        model.zero_grad()
        output = model(scaled_input)
        output[0, target_class].backward()
        gradients.append(scaled_input.grad.data.clone())

    # Average gradients
    avg_gradient = torch.stack(gradients).mean(dim=0)

    # Multiply by input - baseline
    ig = (image - baseline) * avg_gradient
    ig = ig.squeeze().numpy()

    if ig.ndim == 3:
        ig = np.abs(ig).max(axis=0)

    elapsed = (time.time() - start) * 1000

    return AttributionResult(
        method_name="Integrated Gradients",
        attribution_map=ig,
        computation_time_ms=elapsed,
    )


def compute_smoothgrad(model, image, target_class, n_samples=20, noise_level=0.1):
    """Compute SmoothGrad (Smilkov et al., 2017).

    Averages gradients over noisy versions of the input.
    This reduces the noise in vanilla gradients while preserving
    the signal.

    WHY SmoothGrad:
    Vanilla gradients are noisy because of the non-smooth
    activation functions (ReLU). Adding noise and averaging
    smooths out these artifacts.
    """
    import time
    start = time.time()

    all_gradients = []
    for _ in range(n_samples):
        noisy_image = image + noise_level * torch.randn_like(image)
        noisy_image = noisy_image.clone().detach().requires_grad_(True)

        model.zero_grad()
        output = model(noisy_image)
        output[0, target_class].backward()

        all_gradients.append(noisy_image.grad.data.clone())

    avg_gradient = torch.stack(all_gradients).mean(dim=0)
    attribution = avg_gradient.squeeze().numpy()

    if attribution.ndim == 3:
        attribution = np.abs(attribution).max(axis=0)

    elapsed = (time.time() - start) * 1000

    return AttributionResult(
        method_name="SmoothGrad",
        attribution_map=attribution,
        computation_time_ms=elapsed,
    )


def compare_attributions(results: list[AttributionResult]) -> dict:
    """Compare multiple attribution methods.

    Metrics:
    1. Rank correlation between methods (Spearman)
    2. Top-K overlap: do methods agree on most important pixels?
    3. Concentration: how focused is each method's attribution?
    """
    from scipy.stats import spearmanr

    comparison = {
        "methods": [r.method_name for r in results],
        "computation_times": {r.method_name: r.computation_time_ms for r in results},
        "concentrations": {r.method_name: r.top_k_fraction for r in results},
        "rank_correlations": {},
    }

    # Pairwise rank correlations
    for i, r1 in enumerate(results):
        for j, r2 in enumerate(results):
            if i >= j:
                continue
            flat1 = r1.attribution_map.flatten()
            flat2 = r2.attribution_map.flatten()
            # Handle case where one array has zero variance
            if np.std(flat1) == 0 or np.std(flat2) == 0:
                corr = 0.0
            else:
                corr, _ = spearmanr(flat1, flat2)
            pair_name = f"{r1.method_name} vs {r2.method_name}"
            comparison["rank_correlations"][pair_name] = round(float(corr), 3)

    return comparison


# Demonstrate attribution comparison
torch.manual_seed(42)

# Use the simple model from Section 1.2
model = SimpleChestXrayModel(num_classes=2)
model.eval()

image = torch.randn(1, 1, 64, 64)  # Smaller for speed

# Get prediction
with torch.no_grad():
    logits = model(image)
    target_class = logits.argmax(dim=1).item()

# Compute attributions
results = [
    compute_vanilla_gradient(model, image, target_class),
    compute_integrated_gradients(model, image, target_class, steps=20),
    compute_smoothgrad(model, image, target_class, n_samples=10),
]

# Compare
comparison = compare_attributions(results)

print("PIXEL ATTRIBUTION COMPARISON")
print("=" * 60)
print(f"\nPredicted class: {target_class}")

print(f"\nComputation Times:")
for method, time_ms in comparison["computation_times"].items():
    print(f"  {method:25s}: {time_ms:>8.1f}ms")

print(f"\nAttribution Concentration (fraction in top 10% pixels):")
for method, conc in comparison["concentrations"].items():
    print(f"  {method:25s}: {conc:.3f}")

print(f"\nRank Correlations Between Methods:")
for pair, corr in comparison["rank_correlations"].items():
    agreement = "AGREE" if corr > 0.5 else "DISAGREE" if corr < 0.2 else "PARTIAL"
    print(f"  {pair:45s}: {corr:+.3f} ({agreement})")
```

---

## 5. Method Selection Decision Matrix

### 5.1 Choosing the Right Method for the Domain

```python
"""
Interpretability Method Selection Decision Matrix

Given a domain, stakeholder, and model type, which interpretability
method should you use? This decision matrix codifies best practices
from published guidelines and industry experience.

FACTORS IN THE DECISION:
1. Domain requirements (regulatory, clinical, etc.)
2. Stakeholder (data scientist, clinician, end user, auditor)
3. Model type (tree, neural network, linear, ensemble)
4. Data type (tabular, image, text, time series)
5. Latency requirements (real-time, interactive, batch)
6. Faithfulness requirements (advisory vs. high-stakes)
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MethodRecommendation:
    """A recommended interpretability method for a specific context."""
    method: str
    primary_reason: str
    alternatives: list[str]
    caveats: list[str]
    latency: str  # "fast" (<100ms), "moderate" (100ms-1s), "slow" (>1s)
    faithfulness: str  # "high", "moderate", "low"


DECISION_MATRIX = {
    # (domain, data_type, stakeholder) -> recommendation
    ("healthcare", "imaging", "clinician"): MethodRecommendation(
        method="GradCAM",
        primary_reason="Visual, low-resolution, maps to clinical inspection workflow",
        alternatives=["Integrated Gradients (if more precise localization needed)",
                      "Concept-based (if clinical concepts available)"],
        caveats=["GradCAM is coarse — may miss small features",
                "Always present alongside the original image",
                "Include confidence score and spatial statistics"],
        latency="fast",
        faithfulness="moderate",
    ),
    ("healthcare", "tabular", "clinician"): MethodRecommendation(
        method="SHAP (TreeExplainer or Exact)",
        primary_reason="Individual patient-level explanations with signed importance",
        alternatives=["Concept-based (if clinical ontology available)",
                      "Counterfactual (for 'what-if' patient communication)"],
        caveats=["Map feature names to clinical terminology",
                "Group related features (e.g., all vitals together)",
                "Include normal ranges for context"],
        latency="fast",
        faithfulness="high",
    ),
    ("healthcare", "tabular", "patient"): MethodRecommendation(
        method="Counterfactual explanation",
        primary_reason="Patients understand 'what would need to change' better than feature importance",
        alternatives=["Plain-language SHAP summary"],
        caveats=["Must be actionable (patient can actually change the factors)",
                "Avoid medical jargon",
                "Always include 'consult your doctor' disclaimer"],
        latency="slow",
        faithfulness="moderate",
    ),
    ("finance", "tabular", "regulator"): MethodRecommendation(
        method="SHAP (KernelExplainer or TreeExplainer)",
        primary_reason="Satisfies ECOA adverse action requirements and SR 11-7 conceptual soundness",
        alternatives=["LIME (if SHAP too slow)",
                      "Partial dependence plots (for model-level understanding)"],
        caveats=["Must map to approved reason codes",
                "Store explanations for audit trail",
                "Run fairness checks on explanation distributions"],
        latency="moderate",
        faithfulness="high",
    ),
    ("finance", "tabular", "customer"): MethodRecommendation(
        method="Adverse action notice (SHAP-derived)",
        primary_reason="Legal requirement (ECOA) for consumer-facing explanations",
        alternatives=["Counterfactual (what would change the decision)"],
        caveats=["Maximum 4 reasons per notice",
                "Must use consumer-understandable language",
                "Include rights information"],
        latency="fast",
        faithfulness="high",
    ),
    ("nlp", "text", "data_scientist"): MethodRecommendation(
        method="Token-level SHAP (Partition SHAP)",
        primary_reason="Faithful, handles token interactions via coalitional values",
        alternatives=["Integrated Gradients (faster, but less faithful)",
                      "Attention visualization (fast, but unfaithful — see L04)"],
        caveats=["Combine subword tokens for display",
                "Verify with occlusion for critical decisions",
                "Check ERASER benchmark faithfulness metrics"],
        latency="slow",
        faithfulness="high",
    ),
    ("nlp", "text", "end_user"): MethodRecommendation(
        method="Extractive rationale",
        primary_reason="Users understand highlighted passages better than per-word scores",
        alternatives=["Natural language explanation (abstractive)"],
        caveats=["Keep rationale short (20-40% of text)",
                "Verify rationale sufficiency",
                "Show rationale in context of full text"],
        latency="moderate",
        faithfulness="moderate",
    ),
    ("cv", "image", "data_scientist"): MethodRecommendation(
        method="Integrated Gradients + GradCAM (compare both)",
        primary_reason="IG gives pixel-level precision; GradCAM gives class-discriminative regions",
        alternatives=["SmoothGrad (for denoising)", "SHAP (ImageSHAP for superpixels)"],
        caveats=["Always compare at least 2 methods",
                "Check for adversarial vulnerability",
                "Run sanity checks (Adebayo et al.)"],
        latency="moderate",
        faithfulness="high",
    ),
    ("cv", "image", "autonomous_driving"): MethodRecommendation(
        method="Concept-based + attention maps",
        primary_reason="Need to verify model detects correct concepts (pedestrian, sign, lane)",
        alternatives=["GradCAM for quick visual check",
                      "Feature visualization for understanding learned features"],
        caveats=["Verify concept alignment with safety-critical objects",
                "Run failure mode analysis on edge cases",
                "Real-time constraints limit method choice"],
        latency="fast",
        faithfulness="moderate",
    ),
}


def recommend_method(domain: str, data_type: str, stakeholder: str) -> Optional[MethodRecommendation]:
    """Look up the recommended interpretability method.

    Falls back to general recommendations if specific
    combination not found.
    """
    key = (domain.lower(), data_type.lower(), stakeholder.lower())
    return DECISION_MATRIX.get(key)


# Display the decision matrix
print("INTERPRETABILITY METHOD SELECTION MATRIX")
print("=" * 70)

for (domain, data_type, stakeholder), rec in DECISION_MATRIX.items():
    print(f"\nDomain: {domain} | Data: {data_type} | Stakeholder: {stakeholder}")
    print(f"  Recommended: {rec.method}")
    print(f"  Reason: {rec.primary_reason}")
    print(f"  Latency: {rec.latency} | Faithfulness: {rec.faithfulness}")
    if rec.caveats:
        print(f"  Key caveat: {rec.caveats[0]}")
    print(f"  {'─' * 60}")
```

---

## 6. Practical: Clinical Prediction Model with Multi-Stakeholder Explanations

### 6.1 Building the Complete Pipeline

```python
"""
Practical: Multi-Stakeholder Explanation Pipeline

Build a clinical prediction model (30-day hospital readmission risk)
with explanations tailored for THREE different stakeholders:
1. CLINICIAN: Feature importance + clinical concept mapping
2. PATIENT: Plain-language counterfactual explanation
3. AUDITOR: Model card + fairness analysis + full SHAP

This demonstrates that the SAME model needs DIFFERENT explanations
for DIFFERENT audiences — a core principle of production interpretability.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from dataclasses import dataclass, field


def create_readmission_data(n_samples=3000, random_state=42):
    """Create synthetic hospital readmission data.

    Features simulate clinical variables:
    - age, comorbidity_count, length_of_stay, num_medications,
    - prior_admissions, hemoglobin, creatinine, heart_rate
    """
    np.random.seed(random_state)
    X, y = make_classification(
        n_samples=n_samples, n_features=8, n_informative=5,
        n_redundant=2, random_state=random_state, flip_y=0.15,
    )
    feature_names = [
        "age", "comorbidity_count", "length_of_stay", "num_medications",
        "prior_admissions", "hemoglobin", "creatinine", "heart_rate",
    ]
    return X, y, feature_names


# Create data and train model
X, y, feature_names = create_readmission_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = GradientBoostingClassifier(n_estimators=80, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Clinical terminology mapping
CLINICAL_NAMES = {
    "age": "Patient age",
    "comorbidity_count": "Number of comorbid conditions",
    "length_of_stay": "Length of hospital stay (days)",
    "num_medications": "Number of discharge medications",
    "prior_admissions": "Prior hospital admissions (12 months)",
    "hemoglobin": "Hemoglobin level (g/dL)",
    "creatinine": "Serum creatinine (mg/dL)",
    "heart_rate": "Resting heart rate (bpm)",
}

# Normal ranges for clinical context
NORMAL_RANGES = {
    "hemoglobin": (12.0, 17.5),
    "creatinine": (0.7, 1.3),
    "heart_rate": (60, 100),
}


def clinician_explanation(model, features, feature_names):
    """Generate explanation for a CLINICIAN.

    Clinicians need:
    1. Risk score with confidence interval
    2. Top contributing factors in clinical terminology
    3. Abnormal values flagged
    4. Comparison to population risk
    """
    proba = model.predict_proba(features.reshape(1, -1))[0, 1]
    importances = model.feature_importances_

    # Get population risk for comparison
    all_probas = model.predict_proba(X_test)[:, 1]
    risk_percentile = (all_probas < proba).mean() * 100

    print("CLINICIAN EXPLANATION")
    print("=" * 50)
    print(f"30-Day Readmission Risk: {proba:.1%}")
    print(f"Risk Percentile: {risk_percentile:.0f}th (compared to patient population)")
    print(f"Risk Category: {'HIGH' if proba > 0.3 else 'MODERATE' if proba > 0.15 else 'LOW'}")

    print("\nContributing Factors (ranked by importance):")
    sorted_idx = np.argsort(importances)[::-1]
    for rank, idx in enumerate(sorted_idx[:5], 1):
        name = feature_names[idx]
        clinical_name = CLINICAL_NAMES.get(name, name)
        value = features[idx]
        importance = importances[idx]

        # Flag abnormal values
        flag = ""
        if name in NORMAL_RANGES:
            low, high = NORMAL_RANGES[name]
            if value < low:
                flag = " [BELOW NORMAL]"
            elif value > high:
                flag = " [ABOVE NORMAL]"

        print(f"  {rank}. {clinical_name}: {value:.2f}{flag}")
        print(f"     Contribution: {'#' * int(importance * 50)} ({importance:.3f})")


def patient_explanation(model, features, feature_names):
    """Generate explanation for a PATIENT.

    Patients need:
    1. Simple risk assessment (low/moderate/high)
    2. Plain language — no medical jargon
    3. What they CAN DO (actionable recommendations)
    4. Reassurance and context
    """
    proba = model.predict_proba(features.reshape(1, -1))[0, 1]

    risk_level = "HIGH" if proba > 0.3 else "MODERATE" if proba > 0.15 else "LOW"

    print("PATIENT EXPLANATION")
    print("=" * 50)
    print(f"Your Risk of Returning to Hospital: {risk_level}")
    print()

    if risk_level == "HIGH":
        print("What this means:")
        print("  Based on your health information, our assessment tool")
        print("  suggests you have a higher-than-average chance of needing")
        print("  to return to the hospital within the next 30 days.")
    elif risk_level == "MODERATE":
        print("What this means:")
        print("  Based on your health information, you have a moderate")
        print("  chance of needing to return to the hospital.")
    else:
        print("What this means:")
        print("  Based on your health information, you have a lower")
        print("  chance of needing to return to the hospital.")

    print("\nThings that may help:")
    print("  - Take all prescribed medications as directed")
    print("  - Attend your follow-up appointments")
    print("  - Contact your doctor if you notice new symptoms")
    print("  - Keep a record of your vital signs at home")

    print("\nIMPORTANT:")
    print("  This is a tool to help your care team plan your care.")
    print("  It does NOT predict what will definitely happen.")
    print("  Always follow your doctor's advice.")


def auditor_explanation(model, X_test, y_test, feature_names):
    """Generate explanation for an AUDITOR / regulator.

    Auditors need:
    1. Model performance metrics (disaggregated)
    2. Feature importance (global)
    3. Fairness analysis
    4. Model documentation (Model Card summary)
    """
    from sklearn.metrics import accuracy_score, roc_auc_score

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("AUDITOR EXPLANATION")
    print("=" * 50)

    print("\n1. MODEL PERFORMANCE")
    print(f"   Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"   AUC-ROC:  {roc_auc_score(y_test, y_proba):.3f}")

    print("\n2. GLOBAL FEATURE IMPORTANCE")
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for idx in sorted_idx:
        bar = "#" * int(importances[idx] * 80)
        print(f"   {feature_names[idx]:25s}: {importances[idx]:.4f} {bar}")

    print("\n3. PREDICTION DISTRIBUTION")
    print(f"   Mean risk score: {y_proba.mean():.3f}")
    print(f"   Median risk score: {np.median(y_proba):.3f}")
    print(f"   Std risk score: {y_proba.std():.3f}")
    print(f"   Predicted positive rate: {y_pred.mean():.3f}")
    print(f"   Actual positive rate: {y_test.mean():.3f}")

    print("\n4. MODEL DOCUMENTATION STATUS")
    print("   Model Card: Present")
    print("   Data Lineage: Documented")
    print("   Validation Report: Current (2024-Q2)")
    print("   Bias Audit: Scheduled (2024-Q3)")


# Generate all three explanations for the same patient
patient_features = X_test[0]

print("MULTI-STAKEHOLDER EXPLANATION DEMO")
print("=" * 60)
print("Same patient, three different explanations\n")

clinician_explanation(model, patient_features, feature_names)
print()
patient_explanation(model, patient_features, feature_names)
print()
auditor_explanation(model, X_test, y_test, feature_names)
```

---

## Summary

- **Healthcare AI** operates under FDA CDS guidance where interpretability directly affects regulatory classification — a transparent model may be exempt from device regulation while a black-box version requires full FDA clearance
- **GradCAM** is the dominant method for medical imaging, but it shows WHERE the model looks, not WHY — concept-based explanations using clinical ontologies (ABCD criteria, BIRADS) bridge the gap between pixel attribution and clinical reasoning
- **Financial AI** must comply with ECOA adverse action notice requirements (specific reasons for credit denial) and SR 11-7 model risk management (conceptual soundness, outcomes analysis) — SHAP values map directly to adverse action reason codes
- **NLP interpretability** uses token-level attribution (SHAP, occlusion) and rationale extraction (Lei et al., ERASER benchmark) to identify which words drive predictions, with faithfulness and sufficiency as key quality metrics
- **Computer vision** benefits from comparing multiple attribution methods (vanilla gradient, Integrated Gradients, GradCAM, SmoothGrad) — when methods agree, confidence in the explanation is high; disagreement warrants investigation
- A **decision matrix** maps (domain, data type, stakeholder) to the recommended interpretability method, accounting for latency, faithfulness, and regulatory requirements
- The **same model needs different explanations** for different stakeholders: clinicians need clinical terminology and abnormal-value flagging, patients need plain language and actionable advice, auditors need disaggregated metrics and documentation

---

## Exercises

### Exercise 1: FDA CDS Classification

For each of the following AI systems, determine whether it meets the CDS exemption criteria:

1. An AI that analyzes ECG signals to detect atrial fibrillation and alerts the physician
2. A decision support tool that recommends medication dosages based on patient weight, kidney function, and drug interactions using a rules database
3. A deep learning model that predicts 5-year mortality risk from chest X-rays, displayed to patients on a wellness app
4. A natural language processing system that extracts structured data from clinical notes for billing coding

For each, identify which CDS criterion (1-4) it fails (if any) and what changes would be needed to achieve exemption.

### Exercise 2: ECOA Adverse Action Pipeline

Build a complete adverse action notice pipeline:

1. Train a credit scoring model on synthetic data with 10 features
2. Compute SHAP values for each denied application
3. Map the top-4 SHAP features to consumer-readable reason codes
4. Generate formatted adverse action notices for 5 sample denials
5. Verify that the reasons accurately reflect the SHAP values (no misleading reasons)
6. Check that reason distributions across demographic groups do not reveal prohibited discrimination

### Exercise 3: NLP Faithfulness Evaluation

Using the ERASER benchmark methodology, evaluate the faithfulness of three token attribution methods on a text classification task:

1. Implement occlusion-based, gradient-based, and attention-based attribution
2. For each method, extract rationales at 20%, 30%, and 40% compression
3. Compute sufficiency: P(y|rationale) vs P(y|full text)
4. Compute comprehensiveness: P(y|full text) - P(y|complement)
5. Rank methods by faithfulness and explain which is most reliable

### Exercise 4: Multi-Stakeholder Explanation System

Design and implement a complete multi-stakeholder explanation system for a loan approval model with four stakeholders:

1. **Applicant**: Plain-language explanation with counterfactual ("what would change the decision")
2. **Loan officer**: Feature importance dashboard with risk factors highlighted
3. **Compliance officer**: Bias analysis and adverse action notice generation
4. **Model validator**: Stability analysis, feature importance drift, and out-of-distribution detection

For each stakeholder, implement the explanation generation, format the output appropriately, and demonstrate with sample inputs.

### Exercise 5: Domain Transfer Challenge

You have a well-tested SHAP explanation pipeline for credit scoring (tabular data). Adapt it for three new domains:

1. **Medical diagnosis** (tabular patient data): What changes in feature naming, normalization, and presentation?
2. **Document classification** (text): How do you handle variable-length inputs and token interactions?
3. **Quality inspection** (image): How do you move from pixel-level to region-level explanations?

For each domain, identify: (a) what can be reused from the credit scoring pipeline, (b) what must be rebuilt, and (c) what new domain-specific requirements apply.

---

[Previous: Production Interpretability](./14_Production_Interpretability.md) | [Overview](./00_Overview.md) | [Next: Mechanistic Interpretability](./16_Mechanistic_Interpretability.md)

**License**: CC BY-NC 4.0
