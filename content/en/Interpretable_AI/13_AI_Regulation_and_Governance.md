# Lesson 13: AI Regulation and Governance

[Previous: Fairness Mitigation](./12_Fairness_Mitigation.md) | [Next: Production Interpretability](./14_Production_Interpretability.md)

---

## Learning Objectives

- Understand the EU AI Act's risk-based classification system and map AI applications to appropriate risk categories with corresponding obligations
- Apply GDPR Article 22 requirements for automated decision-making, including meaningful explanations and the right to contest
- Navigate the NIST AI Risk Management Framework's four core functions (Govern, Map, Measure, Manage) for organizational AI governance
- Create Model Cards and Datasheets for Datasets following established templates to document model behavior and data provenance
- Conduct a practical regulatory compliance assessment for an AI system using structured checklists and impact assessment templates

---

## 1. The Regulatory Landscape

### 1.1 Why AI Regulation Matters

The deployment of AI systems in high-stakes domains has created an urgent need for
regulatory frameworks. Unlike traditional software, AI systems can exhibit emergent
biases, opaque decision-making, and unpredictable failure modes that existing
regulations were not designed to address.

```python
"""
The AI Regulation Landscape: Key Motivations

1. ACCOUNTABILITY GAP
   Traditional software: deterministic, auditable, traceable.
   ML systems: probabilistic, opaque, data-dependent.
   -> Existing product liability laws struggle with ML failure modes.

2. RIGHTS IMPACT
   AI now makes or influences decisions about:
   - Credit and lending (ECOA, Fair Credit Reporting Act)
   - Employment (Title VII, ADA, ADEA)
   - Healthcare (FDA, HIPAA)
   - Criminal justice (14th Amendment Due Process)
   -> These require explanations, contestability, and non-discrimination.

3. TRUST DEFICIT
   Public trust in AI is fragile:
   - 2023 Pew Research: 52% of Americans more concerned than excited about AI
   - High-profile failures (healthcare algorithms, facial recognition)
   -> Regulation provides a trust framework.

4. COMPETITIVE DYNAMICS
   Companies and nations want:
   - Clear rules to reduce regulatory uncertainty
   - Standards that prevent a "race to the bottom"
   - Interoperable frameworks for global AI trade
"""

# Timeline of major AI regulatory milestones
REGULATORY_TIMELINE = {
    2016: "GDPR adopted (effective 2018) — Article 22 on automated decisions",
    2019: "OECD AI Principles — first intergovernmental AI framework",
    2020: "EU White Paper on AI — lays groundwork for AI Act",
    2021: [
        "EU AI Act proposal (April 2021)",
        "NYC Local Law 144 (December 2021) — automated employment tools",
        "NIST AI RMF draft released",
    ],
    2022: [
        "NIST AI RMF 1.0 published (January 2023 final)",
        "Blueprint for an AI Bill of Rights (White House OSTP)",
    ],
    2023: [
        "EU AI Act political agreement (December 2023)",
        "Executive Order 14110 on Safe AI (October 2023)",
        "China Interim Measures for Generative AI (August 2023)",
    ],
    2024: [
        "EU AI Act formally adopted (March 2024)",
        "EU AI Act enters into force (August 2024)",
        "Colorado AI Act signed (May 2024)",
    ],
    2025: [
        "EU AI Act: prohibited practices apply (February 2025)",
        "EU AI Act: GPAI obligations apply (August 2025)",
    ],
    2026: [
        "EU AI Act: high-risk obligations fully apply (August 2026)",
    ],
}

def print_timeline():
    """Display the regulatory timeline in a readable format."""
    for year, events in sorted(REGULATORY_TIMELINE.items()):
        if isinstance(events, str):
            events = [events]
        print(f"\n{year}:")
        for event in events:
            print(f"  - {event}")

print_timeline()
```

### 1.2 Regulatory Approaches Compared

Different jurisdictions have taken fundamentally different approaches to AI
regulation. Understanding these differences is essential for organizations
operating globally.

```python
"""
Three Regulatory Philosophies

1. RISK-BASED (EU AI Act)
   - Categorizes AI systems by risk level
   - Higher risk = stricter requirements
   - Proactive: must comply BEFORE deployment
   - Philosophy: precautionary principle

2. SECTOR-SPECIFIC (United States)
   - Existing regulators extend their mandate to AI
   - FDA for health AI, SEC for financial AI, FTC for consumer AI
   - Reactive: enforcement after harm
   - Philosophy: innovation-friendly, case-by-case

3. GOVERNMENT-LED (China)
   - Specific regulations for specific AI types
   - Algorithm recommendation, deepfakes, generative AI
   - Centralized: government approval required for some systems
   - Philosophy: state control + innovation
"""

# Comparison matrix of regulatory approaches
REGULATORY_COMPARISON = {
    "EU AI Act": {
        "approach": "Horizontal, risk-based",
        "scope": "All AI systems in EU market",
        "key_requirement": "Conformity assessment for high-risk",
        "explanation_mandate": "Yes — transparency obligations",
        "penalties": "Up to EUR 35M or 7% global turnover",
        "effective": "Phased 2025-2027",
    },
    "US Federal (EO 14110)": {
        "approach": "Sector-specific + executive action",
        "scope": "Federal agencies, dual-use foundation models",
        "key_requirement": "Safety testing for powerful models",
        "explanation_mandate": "Varies by sector",
        "penalties": "Varies by enforcing agency",
        "effective": "2023-ongoing",
    },
    "China (multiple)": {
        "approach": "Technology-specific regulations",
        "scope": "Algorithm recommendation, GenAI, deepfakes",
        "key_requirement": "Algorithm registration, content moderation",
        "explanation_mandate": "Yes — algorithm transparency",
        "penalties": "Varies; can include service suspension",
        "effective": "2021-ongoing",
    },
    "OECD AI Principles": {
        "approach": "Voluntary, principle-based",
        "scope": "46 member/partner countries",
        "key_requirement": "Transparency, accountability, fairness",
        "explanation_mandate": "Recommended, not mandated",
        "penalties": "None (soft law)",
        "effective": "2019",
    },
}

def compare_regulations():
    """Print a formatted comparison of regulatory approaches."""
    for regulation, details in REGULATORY_COMPARISON.items():
        print(f"\n{'=' * 60}")
        print(f"  {regulation}")
        print(f"{'=' * 60}")
        for key, value in details.items():
            print(f"  {key:25s}: {value}")

compare_regulations()
```

---

## 2. The EU AI Act (2024)

### 2.1 Risk Categories

The EU AI Act establishes four risk tiers. Each tier carries different obligations,
ranging from outright prohibition to minimal transparency requirements.

```python
"""
EU AI Act Risk Classification (Article 5, 6, 50, 52)

The Act classifies AI systems into four risk categories.
Understanding the classification is the FIRST step in compliance —
misclassification can lead to either non-compliance (too low)
or unnecessary burden (too high).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class RiskLevel(Enum):
    """The four risk tiers defined by the EU AI Act."""
    UNACCEPTABLE = "unacceptable"  # Article 5: Prohibited
    HIGH = "high"                  # Article 6, Annex III: Strict obligations
    LIMITED = "limited"            # Article 50: Transparency obligations
    MINIMAL = "minimal"            # No specific obligations (voluntary codes)


@dataclass
class AISystemClassification:
    """Classifies an AI system according to EU AI Act risk categories."""

    system_name: str
    description: str
    risk_level: RiskLevel
    rationale: str
    obligations: list[str] = field(default_factory=list)
    annex_reference: Optional[str] = None

    def __str__(self):
        obligations_str = "\n    ".join(f"- {o}" for o in self.obligations)
        return (
            f"System: {self.system_name}\n"
            f"  Risk Level: {self.risk_level.value.upper()}\n"
            f"  Rationale: {self.rationale}\n"
            f"  Obligations:\n    {obligations_str}"
        )


# ----- UNACCEPTABLE RISK (Article 5) -----
# These are BANNED outright. No compliance pathway exists.
PROHIBITED_SYSTEMS = [
    AISystemClassification(
        system_name="Social Scoring (general purpose)",
        description="Government system that scores citizens based on social behavior",
        risk_level=RiskLevel.UNACCEPTABLE,
        rationale="Article 5(1)(c): Social scoring by public authorities that leads "
                  "to detrimental treatment disproportionate to the context",
        obligations=["PROHIBITED — must not be placed on EU market"],
    ),
    AISystemClassification(
        system_name="Subliminal Manipulation",
        description="AI that deploys subliminal techniques to materially distort behavior",
        risk_level=RiskLevel.UNACCEPTABLE,
        rationale="Article 5(1)(a): Techniques beyond a person's consciousness "
                  "causing physical or psychological harm",
        obligations=["PROHIBITED — must not be placed on EU market"],
    ),
    AISystemClassification(
        system_name="Real-time Biometric ID (public, law enforcement)",
        description="Live facial recognition in public spaces for law enforcement",
        risk_level=RiskLevel.UNACCEPTABLE,
        rationale="Article 5(1)(h): Real-time remote biometric identification in "
                  "publicly accessible spaces for law enforcement (with narrow exceptions)",
        obligations=[
            "PROHIBITED in general",
            "Narrow exceptions: missing children, imminent terrorist threat, "
            "serious criminal offenses (requires judicial authorization)"
        ],
    ),
    AISystemClassification(
        system_name="Emotion Recognition (workplace/education)",
        description="AI inferring emotions of employees or students",
        risk_level=RiskLevel.UNACCEPTABLE,
        rationale="Article 5(1)(f): Emotion recognition in workplace and "
                  "educational institutions (with medical/safety exceptions)",
        obligations=["PROHIBITED — must not be deployed in these contexts"],
    ),
]

# ----- HIGH RISK (Article 6, Annex III) -----
# Must undergo conformity assessment, maintain technical documentation,
# implement risk management, data governance, human oversight, etc.
HIGH_RISK_SYSTEMS = [
    AISystemClassification(
        system_name="CV Screening Tool",
        description="AI that filters job applications based on resume analysis",
        risk_level=RiskLevel.HIGH,
        rationale="Annex III, Area 4: Employment — recruitment and selection "
                  "of natural persons, especially for screening applications",
        annex_reference="Annex III, 4(a)",
        obligations=[
            "Risk management system (Article 9)",
            "Data governance — training data quality (Article 10)",
            "Technical documentation (Article 11, Annex IV)",
            "Record-keeping / logging (Article 12)",
            "Transparency — inform users it's AI (Article 13)",
            "Human oversight measures (Article 14)",
            "Accuracy, robustness, cybersecurity (Article 15)",
            "Conformity assessment BEFORE deployment (Article 43)",
            "Registration in EU database (Article 49)",
            "Post-market monitoring (Article 72)",
        ],
    ),
    AISystemClassification(
        system_name="Credit Scoring Model",
        description="AI that evaluates creditworthiness for loan decisions",
        risk_level=RiskLevel.HIGH,
        rationale="Annex III, Area 5(b): Access to essential private services — "
                  "creditworthiness evaluation",
        annex_reference="Annex III, 5(b)",
        obligations=[
            "Risk management system (Article 9)",
            "Data governance (Article 10)",
            "Technical documentation (Article 11)",
            "Logging for traceability (Article 12)",
            "Transparency to users (Article 13)",
            "Human oversight (Article 14)",
            "Conformity assessment (Article 43)",
        ],
    ),
    AISystemClassification(
        system_name="Medical Diagnostic AI",
        description="AI for radiological image analysis in cancer screening",
        risk_level=RiskLevel.HIGH,
        rationale="Product safety legislation — AI as a medical device "
                  "(EU MDR 2017/745) + Annex III, Area 5(a)",
        annex_reference="Annex III, 5(a) + EU MDR",
        obligations=[
            "All high-risk obligations",
            "PLUS: medical device conformity (CE marking)",
            "Clinical evaluation per MDR Article 61",
            "Notified body assessment (Class IIa+ devices)",
        ],
    ),
]

# ----- LIMITED RISK (Article 50) -----
# Transparency obligations only — must inform users they're interacting with AI.
LIMITED_RISK_SYSTEMS = [
    AISystemClassification(
        system_name="Customer Service Chatbot",
        description="AI chatbot for handling customer inquiries",
        risk_level=RiskLevel.LIMITED,
        rationale="Article 50(1): AI systems interacting with natural persons — "
                  "must disclose AI nature unless obvious from context",
        obligations=[
            "Inform users they are interacting with an AI system",
            "Disclosure must be clear, timely, and understandable",
        ],
    ),
    AISystemClassification(
        system_name="Deepfake Generator",
        description="AI that generates synthetic images/video of real people",
        risk_level=RiskLevel.LIMITED,
        rationale="Article 50(4): AI-generated content depicting real persons "
                  "or events must be labeled as artificially generated",
        obligations=[
            "Label output as AI-generated (machine-readable)",
            "Mark content in a way detectable by downstream systems",
        ],
    ),
]

# ----- MINIMAL RISK -----
# No specific obligations. Voluntary codes of conduct encouraged.
MINIMAL_RISK_SYSTEMS = [
    AISystemClassification(
        system_name="Email Spam Filter",
        description="AI that classifies emails as spam or not",
        risk_level=RiskLevel.MINIMAL,
        rationale="Does not fall under Annex III categories, not a prohibited "
                  "practice, not interacting with persons in a way requiring "
                  "transparency disclosure",
        obligations=[
            "No mandatory obligations",
            "Voluntary: may adopt codes of conduct (Article 95)",
        ],
    ),
    AISystemClassification(
        system_name="Video Game NPC AI",
        description="AI controlling non-player characters in a video game",
        risk_level=RiskLevel.MINIMAL,
        rationale="Entertainment use, no impact on fundamental rights or safety",
        obligations=[
            "No mandatory obligations",
        ],
    ),
]

# Print all classifications
print("=" * 70)
print("EU AI ACT RISK CLASSIFICATION EXAMPLES")
print("=" * 70)
for category_name, systems in [
    ("UNACCEPTABLE (PROHIBITED)", PROHIBITED_SYSTEMS),
    ("HIGH RISK", HIGH_RISK_SYSTEMS),
    ("LIMITED RISK", LIMITED_RISK_SYSTEMS),
    ("MINIMAL RISK", MINIMAL_RISK_SYSTEMS),
]:
    print(f"\n{'─' * 70}")
    print(f" {category_name}")
    print(f"{'─' * 70}")
    for system in systems:
        print(f"\n{system}")
```

### 2.2 Conformity Assessment for High-Risk Systems

High-risk AI systems must undergo a conformity assessment before being placed on
the EU market. This is the most demanding compliance requirement.

```python
"""
EU AI Act Conformity Assessment Process (Articles 40-49)

There are TWO pathways for conformity assessment:
1. INTERNAL CONTROL (self-assessment) — most high-risk systems
2. THIRD-PARTY ASSESSMENT (notified body) — biometric categorization,
   critical infrastructure, and some other categories

Understanding which pathway applies is crucial for planning
compliance timelines and budgets.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ConformityStep:
    """A single step in the conformity assessment process."""
    step_number: int
    title: str
    description: str
    article_reference: str
    deliverable: str
    estimated_effort: str  # rough guide


# The internal control procedure (Annex VI)
INTERNAL_CONTROL_STEPS = [
    ConformityStep(
        step_number=1,
        title="Quality Management System",
        description=(
            "Establish a QMS covering: design specifications, development "
            "process, testing/validation, risk management integration, "
            "post-market monitoring procedures, data management."
        ),
        article_reference="Article 17",
        deliverable="QMS documentation",
        estimated_effort="2-6 months to establish",
    ),
    ConformityStep(
        step_number=2,
        title="Technical Documentation",
        description=(
            "Prepare comprehensive technical documentation per Annex IV: "
            "system description, design choices, architecture, algorithms used, "
            "training data description, validation results, risk assessment."
        ),
        article_reference="Article 11, Annex IV",
        deliverable="Technical documentation package",
        estimated_effort="1-3 months",
    ),
    ConformityStep(
        step_number=3,
        title="Risk Management System",
        description=(
            "Implement a continuous risk management system: identify known and "
            "foreseeable risks, estimate and evaluate risks, adopt risk "
            "management measures, test effectiveness of measures."
        ),
        article_reference="Article 9",
        deliverable="Risk management plan + residual risk analysis",
        estimated_effort="2-4 months",
    ),
    ConformityStep(
        step_number=4,
        title="Data Governance",
        description=(
            "Ensure training, validation, and testing datasets meet quality "
            "criteria: relevance, representativeness, completeness, absence of "
            "errors. Document data provenance and preprocessing."
        ),
        article_reference="Article 10",
        deliverable="Data governance documentation",
        estimated_effort="1-3 months",
    ),
    ConformityStep(
        step_number=5,
        title="Logging and Traceability",
        description=(
            "Implement automatic logging of events during system operation, "
            "including input data, outputs, and system behavior. Logs must "
            "enable traceability of system functioning."
        ),
        article_reference="Article 12",
        deliverable="Logging architecture + retention policy",
        estimated_effort="1-2 months",
    ),
    ConformityStep(
        step_number=6,
        title="Human Oversight Measures",
        description=(
            "Design system so humans can effectively oversee it: understand "
            "capabilities/limitations, monitor operation, intervene or halt, "
            "interpret outputs correctly."
        ),
        article_reference="Article 14",
        deliverable="Human oversight protocol",
        estimated_effort="1-2 months",
    ),
    ConformityStep(
        step_number=7,
        title="Testing and Validation",
        description=(
            "Verify the system meets accuracy, robustness, and cybersecurity "
            "requirements. Test against predefined metrics BEFORE deployment."
        ),
        article_reference="Article 15",
        deliverable="Test reports + validation results",
        estimated_effort="1-3 months",
    ),
    ConformityStep(
        step_number=8,
        title="EU Declaration of Conformity",
        description=(
            "The provider draws up a written EU declaration of conformity "
            "for each AI system, stating compliance with the Act."
        ),
        article_reference="Article 47",
        deliverable="Signed Declaration of Conformity",
        estimated_effort="1 week (after all other steps)",
    ),
    ConformityStep(
        step_number=9,
        title="CE Marking",
        description=(
            "Affix the CE marking to the AI system or its documentation, "
            "indicating conformity with the Act."
        ),
        article_reference="Article 48",
        deliverable="CE marking affixed",
        estimated_effort="1 day (after declaration)",
    ),
    ConformityStep(
        step_number=10,
        title="Registration in EU Database",
        description=(
            "Register the high-risk AI system in the EU public database "
            "BEFORE placing it on the market."
        ),
        article_reference="Article 49",
        deliverable="Database registration confirmation",
        estimated_effort="1-2 weeks",
    ),
]


def print_conformity_roadmap():
    """Display the conformity assessment roadmap."""
    print("EU AI ACT — CONFORMITY ASSESSMENT ROADMAP (Internal Control)")
    print("=" * 70)
    for step in INTERNAL_CONTROL_STEPS:
        print(f"\nStep {step.step_number}: {step.title}")
        print(f"  Article: {step.article_reference}")
        print(f"  Description: {step.description}")
        print(f"  Deliverable: {step.deliverable}")
        print(f"  Estimated Effort: {step.estimated_effort}")
        print(f"  {'─' * 50}")

print_conformity_roadmap()
```

### 2.3 Technical Documentation Requirements (Annex IV)

Annex IV specifies what technical documentation must contain. This is the single
most detailed compliance requirement for high-risk systems.

```python
"""
Annex IV Technical Documentation Template

This template captures the minimum information required by Annex IV
of the EU AI Act. In practice, organizations should maintain this
as a living document that evolves with the AI system.

WHY this matters: Technical documentation is the PRIMARY artifact
reviewed during conformity assessment. Incomplete documentation
is the most common reason for compliance failure.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Optional


@dataclass
class AnnexIVDocumentation:
    """Template for EU AI Act Annex IV technical documentation.

    Each field corresponds to a specific requirement in Annex IV.
    Fields marked 'required' must be filled for conformity.
    """

    # Section 1: General description
    system_name: str = ""
    intended_purpose: str = ""
    provider_name: str = ""
    provider_address: str = ""
    version: str = ""
    date_of_documentation: str = ""

    # Section 2: Detailed description of the system
    # WHY: Regulators need to understand HOW the system works
    # to assess its risk profile
    system_elements: str = ""       # hardware, software, infrastructure
    development_process: str = ""    # methodology, tools, frameworks
    design_specifications: str = ""  # architecture, algorithms, data flow
    system_architecture: str = ""    # diagram + textual description

    # Section 3: Algorithms and key design choices
    # WHY: This is the core of interpretability compliance.
    # The regulator must be able to understand the model's logic.
    algorithm_description: str = ""       # algorithm type, why chosen
    key_design_choices: str = ""          # tradeoffs made and rationale
    optimization_objective: str = ""      # what the model optimizes for
    expected_output: str = ""             # output format and interpretation
    computational_resources: str = ""     # training and inference requirements

    # Section 4: Data requirements
    # WHY: Data is where bias enters. This section enables auditors
    # to evaluate whether training data is representative and fair.
    training_data_description: str = ""   # source, size, composition
    data_collection_process: str = ""     # how data was gathered
    data_preparation: str = ""            # preprocessing, cleaning, labeling
    data_biases_identified: str = ""      # known limitations and gaps
    data_provenance: str = ""             # chain of custody

    # Section 5: Validation and testing
    # WHY: Without evidence of testing, there's no basis for trust.
    validation_methodology: str = ""      # how validation was performed
    metrics_used: str = ""                # accuracy, fairness, robustness
    test_results: str = ""                # quantitative results
    known_limitations: str = ""           # failure modes, edge cases

    # Section 6: Risk management
    risk_assessment: str = ""             # identified risks and severity
    risk_mitigation_measures: str = ""    # how risks are addressed
    residual_risks: str = ""              # risks that remain after mitigation

    # Section 7: Monitoring and updates
    monitoring_plan: str = ""             # post-deployment monitoring
    change_management: str = ""           # how updates are handled

    def completeness_check(self) -> dict[str, bool]:
        """Check which required sections are filled.

        Returns a dict mapping section names to completion status.
        This is a practical compliance readiness check.
        """
        required_fields = {
            "system_name": self.system_name,
            "intended_purpose": self.intended_purpose,
            "provider_name": self.provider_name,
            "algorithm_description": self.algorithm_description,
            "training_data_description": self.training_data_description,
            "validation_methodology": self.validation_methodology,
            "test_results": self.test_results,
            "risk_assessment": self.risk_assessment,
            "monitoring_plan": self.monitoring_plan,
        }
        return {k: bool(v.strip()) for k, v in required_fields.items()}

    def compliance_score(self) -> float:
        """Return the fraction of required fields that are complete."""
        checks = self.completeness_check()
        filled = sum(checks.values())
        return filled / len(checks)


# Example: filling out documentation for a credit scoring model
credit_model_docs = AnnexIVDocumentation(
    system_name="CreditScore-AI v2.3",
    intended_purpose=(
        "Evaluate creditworthiness of natural persons applying for "
        "consumer loans between EUR 1,000 and EUR 50,000."
    ),
    provider_name="FinTech Corp GmbH",
    provider_address="Berlin, Germany",
    version="2.3.1",
    date_of_documentation=str(date.today()),
    algorithm_description=(
        "Gradient Boosted Decision Trees (XGBoost). Chosen for high "
        "predictive accuracy on tabular data and native support for "
        "feature importance, which aids transparency requirements."
    ),
    training_data_description=(
        "500,000 historical loan applications from 2019-2023. "
        "Features: income, employment duration, debt-to-income ratio, "
        "payment history, loan amount requested. "
        "Demographics: 52% male, 48% female; age range 18-75; "
        "geographic distribution across all 16 German Bundeslaender."
    ),
    validation_methodology=(
        "5-fold stratified cross-validation on held-out 20% test set. "
        "Fairness evaluation using demographic parity, equalized odds, "
        "and calibration across protected groups."
    ),
    test_results=(
        "AUC: 0.847 (95% CI: 0.841-0.853). "
        "Demographic parity ratio: 0.89 (male/female). "
        "Equalized odds difference: 0.04."
    ),
    risk_assessment=(
        "Primary risks: (1) Proxy discrimination via correlated features "
        "(postal code -> ethnicity). Mitigation: removed postal code, "
        "monitor disparate impact quarterly. (2) Concept drift as economic "
        "conditions change. Mitigation: monthly performance monitoring "
        "with automatic retraining trigger at AUC < 0.82."
    ),
    monitoring_plan=(
        "Monthly: AUC on recent decisions, fairness metrics by group. "
        "Quarterly: comprehensive bias audit. "
        "Annual: full re-validation with updated test set."
    ),
)

# Check compliance readiness
print("Compliance Readiness Check")
print("=" * 50)
checks = credit_model_docs.completeness_check()
for field_name, is_complete in checks.items():
    status = "COMPLETE" if is_complete else "MISSING"
    print(f"  {field_name:35s} [{status}]")
print(f"\nOverall Score: {credit_model_docs.compliance_score():.0%}")
```

---

## 3. GDPR Article 22: Automated Decision-Making

### 3.1 The Right to Explanation

GDPR Article 22 is the most widely cited legal basis for AI explainability
in Europe. It grants individuals specific rights when subjected to automated
decision-making.

```python
"""
GDPR Article 22 — Automated Individual Decision-Making

Article 22(1):
  "The data subject shall have the right NOT to be subject to a decision
   based solely on automated processing, including profiling, which
   produces legal effects concerning him or her or similarly significantly
   affects him or her."

KEY CONDITIONS for Article 22 to apply:
  1. Decision is SOLELY automated (no meaningful human involvement)
  2. Decision produces LEGAL EFFECTS or SIMILARLY SIGNIFICANT effects
  3. No valid exception applies (consent, contract, or law)

If Article 22 applies, Recital 71 states the controller must provide:
  - "meaningful information about the logic involved"
  - "the significance and envisaged consequences"

WHY this matters for interpretability:
  The phrase "meaningful information about the logic involved" is the
  legal basis for requiring AI explanations in the EU. However, there
  is ongoing legal debate about HOW detailed explanations must be.
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


class Article22Applicability(Enum):
    """Possible outcomes of Article 22 applicability analysis."""
    APPLIES = "Article 22 applies — full obligations"
    EXCEPTION = "Article 22 applies but exception available"
    DOES_NOT_APPLY = "Article 22 does not apply"


@dataclass
class Article22Assessment:
    """Assess whether GDPR Article 22 applies to an AI system.

    This assessment determines the explanation obligations
    the controller must fulfill.
    """
    system_name: str
    is_solely_automated: bool
    has_legal_effects: bool
    has_significant_effects: bool
    has_consent: bool
    is_contractual_necessity: bool
    has_legal_basis: bool

    def assess(self) -> Article22Applicability:
        """Determine if Article 22 applies.

        The logic follows the Article 29 Working Party guidelines
        on automated decision-making (WP251).
        """
        # Step 1: Is it solely automated?
        # "Solely" means no meaningful human involvement.
        # A human rubber-stamping does NOT count as involvement.
        if not self.is_solely_automated:
            return Article22Applicability.DOES_NOT_APPLY

        # Step 2: Does it produce legal or significant effects?
        # Legal effects: denial of citizenship, contract termination
        # Significant effects: credit denial, job rejection, insurance pricing
        if not (self.has_legal_effects or self.has_significant_effects):
            return Article22Applicability.DOES_NOT_APPLY

        # Step 3: Is there a valid exception?
        # Three exceptions exist (Article 22(2)):
        # (a) necessary for contract, (b) authorized by law, (c) explicit consent
        if self.has_consent or self.is_contractual_necessity or self.has_legal_basis:
            return Article22Applicability.EXCEPTION

        # No exception: full Article 22 prohibition applies
        return Article22Applicability.APPLIES

    def required_safeguards(self) -> list[str]:
        """Return the safeguards required based on the assessment.

        Even when an exception applies, Article 22(3) requires
        "suitable measures to safeguard the data subject's rights."
        """
        result = self.assess()

        if result == Article22Applicability.DOES_NOT_APPLY:
            return ["No specific Article 22 obligations (general GDPR still applies)"]

        # Both APPLIES and EXCEPTION cases require safeguards
        safeguards = [
            "Right to obtain meaningful information about the logic involved (Art. 13(2)(f), 14(2)(g))",
            "Right to obtain human intervention (Art. 22(3))",
            "Right to express their point of view (Art. 22(3))",
            "Right to contest the decision (Art. 22(3))",
        ]

        if result == Article22Applicability.APPLIES:
            safeguards.insert(0, "PROHIBITION: Decision cannot be made — must add human review or obtain exception")

        return safeguards


# Example assessments
examples = [
    Article22Assessment(
        system_name="Auto-rejected loan application (no human review)",
        is_solely_automated=True,
        has_legal_effects=False,
        has_significant_effects=True,  # Credit denial is "significant"
        has_consent=False,
        is_contractual_necessity=True,  # Part of loan application contract
        has_legal_basis=False,
    ),
    Article22Assessment(
        system_name="Content recommendation algorithm",
        is_solely_automated=True,
        has_legal_effects=False,
        has_significant_effects=False,  # Showing different content is not "significant"
        has_consent=False,
        is_contractual_necessity=False,
        has_legal_basis=False,
    ),
    Article22Assessment(
        system_name="Automated insurance pricing with no appeal process",
        is_solely_automated=True,
        has_legal_effects=False,
        has_significant_effects=True,  # Insurance pricing is "significant"
        has_consent=False,
        is_contractual_necessity=False,
        has_legal_basis=False,
    ),
]

for ex in examples:
    result = ex.assess()
    print(f"\nSystem: {ex.system_name}")
    print(f"  Assessment: {result.value}")
    print(f"  Required safeguards:")
    for s in ex.required_safeguards():
        print(f"    - {s}")
```

### 3.2 What Counts as "Meaningful Information About the Logic"

```python
"""
Levels of Explanation Under GDPR

The GDPR does not specify exactly what "meaningful information about
the logic involved" requires. Legal scholars and Data Protection
Authorities have proposed different interpretations:

LEVEL 1: System-level explanation (generally accepted as minimum)
  - What type of model is used
  - What data categories are processed
  - What the system aims to predict
  - Example: "We use a machine learning model that evaluates your
    income, employment history, and payment record to predict
    likelihood of loan repayment."

LEVEL 2: Factor-level explanation (emerging consensus)
  - Which factors were most important for THIS specific decision
  - How changing key factors would change the outcome
  - Example: "Your application was declined primarily because your
    debt-to-income ratio (0.45) exceeds our threshold (0.40).
    Reducing your outstanding debt by EUR 2,000 would likely
    change the outcome."

LEVEL 3: Full algorithmic transparency (generally NOT required)
  - Complete model weights, training data, source code
  - Most DPAs agree this is NOT what the GDPR requires
  - Would also conflict with trade secret protections

The Article 29 Working Party (WP251) recommends Level 2 as the
practical target: meaningful, individual-specific explanations
that help the data subject understand and contest the decision.
"""

from dataclasses import dataclass


@dataclass
class ExplanationTemplate:
    """Template for GDPR-compliant explanations at different levels."""

    system_description: str       # Level 1: what the system does
    data_categories: list[str]    # Level 1: what data is used
    decision_factors: dict[str, float]  # Level 2: factor importance
    counterfactual: str           # Level 2: what would change outcome
    decision: str

    def level_1_explanation(self) -> str:
        """Generate a system-level explanation (minimum GDPR compliance)."""
        data_list = ", ".join(self.data_categories)
        return (
            f"Decision: {self.decision}\n\n"
            f"System Description: {self.system_description}\n\n"
            f"Data Used: The following categories of your personal data "
            f"were processed: {data_list}.\n\n"
            f"You have the right to request human review of this decision, "
            f"express your point of view, and contest the decision."
        )

    def level_2_explanation(self) -> str:
        """Generate a factor-level explanation (recommended best practice)."""
        # Sort factors by importance (absolute value)
        sorted_factors = sorted(
            self.decision_factors.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        factors_text = ""
        for factor, importance in sorted_factors[:5]:
            direction = "increased" if importance > 0 else "decreased"
            factors_text += (
                f"  - {factor}: {direction} likelihood "
                f"(importance: {abs(importance):.2f})\n"
            )

        return (
            f"Decision: {self.decision}\n\n"
            f"Key Factors in Your Decision:\n{factors_text}\n"
            f"What Would Change This Decision:\n  {self.counterfactual}\n\n"
            f"Your Rights:\n"
            f"  - Request human review of this decision\n"
            f"  - Provide additional information for reconsideration\n"
            f"  - Contest this decision through our appeals process"
        )


# Example: loan decision explanation
loan_explanation = ExplanationTemplate(
    system_description=(
        "Our automated credit assessment system uses a machine learning "
        "model to evaluate loan applications. The model was trained on "
        "historical loan data to predict the likelihood of timely repayment."
    ),
    data_categories=[
        "annual income",
        "employment duration",
        "existing debt obligations",
        "payment history",
        "requested loan amount",
    ],
    decision_factors={
        "Debt-to-income ratio": -0.35,      # Negative = hurt the application
        "Payment history (12 months)": 0.28,  # Positive = helped
        "Employment duration": 0.22,
        "Requested loan amount": -0.18,
        "Annual income": 0.15,
    },
    counterfactual=(
        "If your debt-to-income ratio were below 0.40 (currently 0.47), "
        "the model would likely approve this application. This could be "
        "achieved by reducing outstanding debt by approximately EUR 3,500 "
        "or increasing monthly income by approximately EUR 800."
    ),
    decision="Application DECLINED",
)

print("=" * 60)
print("LEVEL 1 EXPLANATION (Minimum GDPR Compliance)")
print("=" * 60)
print(loan_explanation.level_1_explanation())

print("\n" + "=" * 60)
print("LEVEL 2 EXPLANATION (Recommended Best Practice)")
print("=" * 60)
print(loan_explanation.level_2_explanation())
```

---

## 4. NIST AI Risk Management Framework

### 4.1 The Four Core Functions

The NIST AI RMF provides a voluntary framework for managing AI risks. Unlike the
EU AI Act, it is not legally binding, but it has become the de facto standard for
AI governance in the United States and is increasingly referenced internationally.

```python
"""
NIST AI Risk Management Framework (AI RMF 1.0, January 2023)

The framework is organized around four core functions:
  GOVERN → MAP → MEASURE → MANAGE

Unlike the EU AI Act (which prescribes specific obligations),
NIST AI RMF is OUTCOME-BASED: it describes what organizations
should achieve, not how to achieve it.

WHY use NIST AI RMF even if you're not US-based:
  1. It maps well to EU AI Act requirements (good starting point)
  2. It's technology-neutral and flexible
  3. It's widely recognized by auditors and regulators
  4. It provides a common vocabulary for AI risk discussions
"""

from dataclasses import dataclass, field


@dataclass
class NISTFunction:
    """Represents one of the four NIST AI RMF core functions."""
    name: str
    purpose: str
    categories: list[dict[str, str]]  # Each has 'id', 'name', 'description'


GOVERN = NISTFunction(
    name="GOVERN",
    purpose=(
        "Cultivate and implement a culture of risk management. "
        "Establish organizational policies, processes, procedures, "
        "and practices for AI risk management."
    ),
    categories=[
        {
            "id": "GOVERN 1",
            "name": "Policies and Procedures",
            "description": (
                "Legal and regulatory requirements are understood, "
                "managed, and documented. Policies reflect risk tolerance."
            ),
        },
        {
            "id": "GOVERN 2",
            "name": "Accountability Structures",
            "description": (
                "Clear roles and responsibilities for AI risk management. "
                "Teams have the skills and resources needed."
            ),
        },
        {
            "id": "GOVERN 3",
            "name": "Workforce Diversity",
            "description": (
                "Workforce is diverse, equitable, and inclusive. "
                "AI actors come from a range of backgrounds."
            ),
        },
        {
            "id": "GOVERN 4",
            "name": "Organizational Culture",
            "description": (
                "Cultures of safety, responsible AI, and continuous "
                "improvement are fostered at all levels."
            ),
        },
        {
            "id": "GOVERN 5",
            "name": "Stakeholder Engagement",
            "description": (
                "Ongoing engagement with relevant stakeholders, including "
                "affected communities, to inform risk management."
            ),
        },
        {
            "id": "GOVERN 6",
            "name": "Policies for Third-Party AI",
            "description": (
                "Risk management of third-party AI systems, including "
                "supply chain, vendor assessments, and procurement."
            ),
        },
    ],
)

MAP = NISTFunction(
    name="MAP",
    purpose=(
        "Identify the context in which the AI system will operate. "
        "Characterize the risks, benefits, and potential impacts."
    ),
    categories=[
        {
            "id": "MAP 1",
            "name": "Intended Purpose and Context",
            "description": (
                "The intended purpose, context of use, potential benefits, "
                "and costs are understood and documented."
            ),
        },
        {
            "id": "MAP 2",
            "name": "Interdisciplinary Analysis",
            "description": (
                "Risks and benefits are analyzed by an interdisciplinary "
                "team with domain expertise and diverse perspectives."
            ),
        },
        {
            "id": "MAP 3",
            "name": "AI-Specific Risks",
            "description": (
                "AI-specific risks (bias, drift, adversarial attacks, "
                "hallucination) are identified and documented."
            ),
        },
        {
            "id": "MAP 4",
            "name": "Risk Prioritization",
            "description": (
                "Identified risks are prioritized based on potential "
                "impact and likelihood."
            ),
        },
        {
            "id": "MAP 5",
            "name": "Impact on Affected Communities",
            "description": (
                "Impacts on individuals, groups, communities, and the "
                "environment are characterized."
            ),
        },
    ],
)

MEASURE = NISTFunction(
    name="MEASURE",
    purpose=(
        "Quantify, assess, benchmark, and monitor AI risks. "
        "Use appropriate methods and metrics."
    ),
    categories=[
        {
            "id": "MEASURE 1",
            "name": "Risk Metrics",
            "description": (
                "Appropriate methods and metrics are identified and "
                "applied to evaluate AI risks."
            ),
        },
        {
            "id": "MEASURE 2",
            "name": "Trustworthiness Assessment",
            "description": (
                "AI systems are evaluated for accuracy, fairness, "
                "privacy, security, resilience, transparency, and "
                "explainability."
            ),
        },
        {
            "id": "MEASURE 3",
            "name": "Tracking Over Time",
            "description": (
                "Risks are tracked over the AI system's lifecycle, "
                "including during deployment and post-deployment."
            ),
        },
        {
            "id": "MEASURE 4",
            "name": "Feedback Mechanisms",
            "description": (
                "Mechanisms for receiving and incorporating feedback "
                "from affected individuals and communities."
            ),
        },
    ],
)

MANAGE = NISTFunction(
    name="MANAGE",
    purpose=(
        "Allocate resources to mitigate and manage identified AI risks. "
        "Maximize benefits while minimizing harms."
    ),
    categories=[
        {
            "id": "MANAGE 1",
            "name": "Risk Treatment",
            "description": (
                "Strategies to respond to assessed risks are developed "
                "and implemented. Options: accept, mitigate, transfer."
            ),
        },
        {
            "id": "MANAGE 2",
            "name": "Risk Prioritization and Response",
            "description": (
                "Risks are managed based on priority, with the highest "
                "risks receiving the most resources and attention."
            ),
        },
        {
            "id": "MANAGE 3",
            "name": "Risk Response Monitoring",
            "description": (
                "Risk responses are monitored for effectiveness and "
                "adjusted as needed based on new information."
            ),
        },
        {
            "id": "MANAGE 4",
            "name": "Incident Response",
            "description": (
                "Plans for addressing incidents and errors, including "
                "communication with affected stakeholders."
            ),
        },
    ],
)


def print_nist_framework():
    """Display the complete NIST AI RMF framework."""
    for function in [GOVERN, MAP, MEASURE, MANAGE]:
        print(f"\n{'=' * 60}")
        print(f"  {function.name}")
        print(f"  {function.purpose}")
        print(f"{'=' * 60}")
        for cat in function.categories:
            print(f"\n  [{cat['id']}] {cat['name']}")
            print(f"    {cat['description']}")

print_nist_framework()
```

---

## 5. NYC Local Law 144: Automated Employment Decision Tools

### 5.1 Overview and Requirements

NYC Local Law 144 (effective July 5, 2023) is the first US local law specifically
targeting automated decision tools in employment.

```python
"""
NYC Local Law 144 — Automated Employment Decision Tools (AEDTs)

KEY REQUIREMENTS:
1. BIAS AUDIT: An independent bias audit must be conducted no more than
   one year before use of the AEDT. The audit must assess disparate
   impact on sex/gender and race/ethnicity categories.

2. PUBLIC NOTICE: Employers must provide notice to candidates that an
   AEDT will be used, at least 10 business days before use.

3. IMPACT RATIOS: The audit must calculate selection rates and impact
   ratios for each demographic category.

4. PUBLICATION: Summary audit results must be publicly available on
   the employer's website.

WHY this matters:
  LL144 creates a concrete, testable standard for algorithmic fairness
  in employment. Unlike the EU AI Act (which is principles-based for
  many requirements), LL144 specifies EXACTLY what to measure and report.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class BiasAuditResult:
    """Results of a bias audit per NYC LL144 requirements.

    The audit must calculate:
    1. Selection rate for each category
    2. Impact ratio (selection rate of group / selection rate of most-selected group)

    The 4/5ths (80%) rule from EEOC Uniform Guidelines is commonly
    used as a benchmark, though LL144 does not mandate a specific threshold.
    """
    category_name: str
    group_name: str
    total_applicants: int
    selected: int

    @property
    def selection_rate(self) -> float:
        """Selection rate = selected / total_applicants."""
        if self.total_applicants == 0:
            return 0.0
        return self.selected / self.total_applicants


def conduct_ll144_audit(results: list[BiasAuditResult]) -> dict:
    """Conduct a bias audit following NYC LL144 methodology.

    Steps:
    1. Calculate selection rate for each demographic group
    2. Find the highest selection rate (reference group)
    3. Calculate impact ratio = group_rate / reference_rate
    4. Flag any impact ratio below 0.80 (4/5ths rule)

    Returns a structured audit report.
    """
    # Step 1: Calculate selection rates
    rates = {}
    for r in results:
        rates[r.group_name] = {
            "selection_rate": r.selection_rate,
            "total": r.total_applicants,
            "selected": r.selected,
        }

    # Step 2: Find the reference (highest) selection rate
    max_rate = max(r["selection_rate"] for r in rates.values())
    reference_group = [
        name for name, r in rates.items()
        if r["selection_rate"] == max_rate
    ][0]

    # Step 3: Calculate impact ratios
    impact_ratios = {}
    for group, data in rates.items():
        if max_rate > 0:
            ratio = data["selection_rate"] / max_rate
        else:
            ratio = 0.0
        impact_ratios[group] = {
            "selection_rate": data["selection_rate"],
            "impact_ratio": ratio,
            "passes_4_5ths": ratio >= 0.80,
            "total": data["total"],
            "selected": data["selected"],
        }

    return {
        "reference_group": reference_group,
        "reference_rate": max_rate,
        "impact_ratios": impact_ratios,
        "audit_passes": all(
            v["passes_4_5ths"] for v in impact_ratios.values()
        ),
    }


# Example: Bias audit of an automated resume screening tool
# Simulated data for race/ethnicity categories
np.random.seed(42)

race_results = [
    BiasAuditResult("Race/Ethnicity", "White", 1000, 250),
    BiasAuditResult("Race/Ethnicity", "Black", 800, 160),
    BiasAuditResult("Race/Ethnicity", "Hispanic", 600, 132),
    BiasAuditResult("Race/Ethnicity", "Asian", 500, 140),
]

gender_results = [
    BiasAuditResult("Sex/Gender", "Male", 1500, 420),
    BiasAuditResult("Sex/Gender", "Female", 1400, 262),
]

print("NYC LOCAL LAW 144 — BIAS AUDIT REPORT")
print("=" * 60)
print(f"System: Automated Resume Screening Tool v3.1")
print(f"Audit Date: 2024-06-15")
print(f"Auditor: Independent AI Audit LLC")

for category_name, results in [("Race/Ethnicity", race_results), ("Sex/Gender", gender_results)]:
    audit = conduct_ll144_audit(results)
    print(f"\n{'─' * 60}")
    print(f"Category: {category_name}")
    print(f"Reference Group: {audit['reference_group']} "
          f"(rate: {audit['reference_rate']:.1%})")
    print(f"{'─' * 60}")
    print(f"{'Group':12s} {'Total':>6s} {'Selected':>8s} {'Rate':>8s} {'Impact':>8s} {'4/5ths':>8s}")

    for group, data in audit["impact_ratios"].items():
        flag = "PASS" if data["passes_4_5ths"] else "FAIL"
        print(
            f"{group:12s} {data['total']:>6d} {data['selected']:>8d} "
            f"{data['selection_rate']:>7.1%} {data['impact_ratio']:>7.2f} "
            f"{'  ' + flag:>8s}"
        )

    overall = "PASS" if audit["audit_passes"] else "FAIL"
    print(f"\nOverall Audit Result: {overall}")
```

---

## 6. Model Cards and Datasheets

### 6.1 Model Cards (Mitchell et al., 2019)

Model Cards are standardized documentation for trained machine learning models.
They were proposed by Margaret Mitchell et al. at Google to promote transparent
reporting of model performance across different conditions.

```python
"""
Model Cards for Model Reporting (Mitchell et al., 2019)

A Model Card answers: "What should a user know about this model
BEFORE using it?"

The template includes:
1. Model Details — basic identification
2. Intended Use — what the model is for (and what it's NOT for)
3. Factors — relevant demographic/environmental factors
4. Metrics — how performance is measured
5. Evaluation Data — what data was used for evaluation
6. Training Data — what data was used for training
7. Quantitative Analyses — disaggregated performance results
8. Ethical Considerations — known ethical issues
9. Caveats and Recommendations — usage warnings

WHY Model Cards matter:
  They create a SHARED UNDERSTANDING between model developers and
  model users. Without them, users often make incorrect assumptions
  about model capabilities and limitations.
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import date


@dataclass
class ModelCard:
    """Model Card template following Mitchell et al. (2019).

    This implementation can be serialized to Markdown for documentation
    or to JSON for programmatic consumption.
    """

    # Section 1: Model Details
    model_name: str = ""
    model_version: str = ""
    model_type: str = ""
    paper_or_resource: str = ""
    developer: str = ""
    model_date: str = ""
    license: str = ""
    contact: str = ""

    # Section 2: Intended Use
    primary_intended_uses: list[str] = field(default_factory=list)
    primary_intended_users: list[str] = field(default_factory=list)
    out_of_scope_uses: list[str] = field(default_factory=list)

    # Section 3: Factors
    relevant_factors: list[str] = field(default_factory=list)
    evaluation_factors: list[str] = field(default_factory=list)

    # Section 4: Metrics
    performance_metrics: list[str] = field(default_factory=list)
    decision_thresholds: str = ""
    variation_approaches: str = ""

    # Section 5 & 6: Data
    training_data: str = ""
    evaluation_data: str = ""

    # Section 7: Quantitative Analyses
    unitary_results: dict = field(default_factory=dict)
    intersectional_results: dict = field(default_factory=dict)

    # Section 8 & 9: Ethics and Caveats
    ethical_considerations: list[str] = field(default_factory=list)
    caveats_and_recommendations: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Generate a Markdown-formatted Model Card.

        This is the standard output format for model cards.
        Many organizations include this in their model registry.
        """
        sections = []

        # Model Details
        sections.append(f"# Model Card: {self.model_name}\n")
        sections.append("## Model Details\n")
        sections.append(f"- **Name**: {self.model_name}")
        sections.append(f"- **Version**: {self.model_version}")
        sections.append(f"- **Type**: {self.model_type}")
        sections.append(f"- **Developer**: {self.developer}")
        sections.append(f"- **Date**: {self.model_date}")
        sections.append(f"- **License**: {self.license}")
        sections.append(f"- **Reference**: {self.paper_or_resource}")
        sections.append(f"- **Contact**: {self.contact}\n")

        # Intended Use
        sections.append("## Intended Use\n")
        sections.append("### Primary Intended Uses")
        for use in self.primary_intended_uses:
            sections.append(f"- {use}")
        sections.append("\n### Primary Intended Users")
        for user in self.primary_intended_users:
            sections.append(f"- {user}")
        sections.append("\n### Out-of-Scope Uses")
        for use in self.out_of_scope_uses:
            sections.append(f"- {use}")

        # Metrics
        sections.append("\n## Metrics\n")
        for metric in self.performance_metrics:
            sections.append(f"- {metric}")

        # Data
        sections.append(f"\n## Training Data\n{self.training_data}")
        sections.append(f"\n## Evaluation Data\n{self.evaluation_data}")

        # Quantitative Analysis
        sections.append("\n## Quantitative Analysis\n")
        if self.unitary_results:
            sections.append("### Overall Performance")
            for metric, value in self.unitary_results.items():
                sections.append(f"- {metric}: {value}")
        if self.intersectional_results:
            sections.append("\n### Disaggregated Performance")
            for group, metrics in self.intersectional_results.items():
                sections.append(f"\n**{group}**")
                for metric, value in metrics.items():
                    sections.append(f"- {metric}: {value}")

        # Ethics
        sections.append("\n## Ethical Considerations\n")
        for consideration in self.ethical_considerations:
            sections.append(f"- {consideration}")

        # Caveats
        sections.append("\n## Caveats and Recommendations\n")
        for caveat in self.caveats_and_recommendations:
            sections.append(f"- {caveat}")

        return "\n".join(sections)


# Example: Creating a model card for a sentiment analysis model
sentiment_card = ModelCard(
    model_name="SentimentBERT-v2",
    model_version="2.1.0",
    model_type="Fine-tuned BERT-base for binary sentiment classification",
    paper_or_resource="Internal technical report TR-2024-042",
    developer="NLP Team, Example Corp",
    model_date="2024-03-15",
    license="Proprietary",
    contact="nlp-team@example.com",
    primary_intended_uses=[
        "Classify customer review sentiment (positive/negative)",
        "Aggregate sentiment trends for product feedback dashboards",
    ],
    primary_intended_users=[
        "Product managers reviewing customer feedback",
        "Customer support teams prioritizing negative reviews",
    ],
    out_of_scope_uses=[
        "Detecting sarcasm or nuanced emotions (not trained for this)",
        "Processing languages other than English",
        "Making automated decisions about individuals based on sentiment",
        "Political sentiment analysis or opinion polling",
    ],
    performance_metrics=[
        "Accuracy (primary): proportion of correct predictions",
        "F1 Score: harmonic mean of precision and recall",
        "AUC-ROC: discrimination ability across thresholds",
    ],
    training_data=(
        "250,000 product reviews from the company's e-commerce platform "
        "(2020-2023). Manually labeled by 12 trained annotators with "
        "inter-annotator agreement (Cohen's kappa) of 0.87. "
        "Class balance: 58% positive, 42% negative."
    ),
    evaluation_data=(
        "Held-out test set of 25,000 reviews (stratified by product "
        "category and review date). Not seen during training or tuning."
    ),
    unitary_results={
        "Accuracy": "91.3%",
        "F1 (positive)": "0.927",
        "F1 (negative)": "0.894",
        "AUC-ROC": "0.963",
    },
    intersectional_results={
        "Electronics reviews": {"Accuracy": "93.1%", "F1": "0.941"},
        "Clothing reviews": {"Accuracy": "90.2%", "F1": "0.912"},
        "Books reviews": {"Accuracy": "89.5%", "F1": "0.901"},
        "Short reviews (<20 words)": {"Accuracy": "84.7%", "F1": "0.856"},
        "Long reviews (>200 words)": {"Accuracy": "92.8%", "F1": "0.935"},
    },
    ethical_considerations=[
        "Model may reflect biases present in customer review data",
        "Product categories with fewer reviews have lower accuracy",
        "Short/terse reviews are harder to classify — may disadvantage "
        "users who write brief feedback",
    ],
    caveats_and_recommendations=[
        "Do NOT use for automated moderation without human review",
        "Re-evaluate quarterly as product lines and customer language evolve",
        "Monitor for concept drift — customer sentiment vocabulary changes",
        "Accuracy on non-English text is UNDEFINED — do not apply",
    ],
)

print(sentiment_card.to_markdown())
```

### 6.2 Datasheets for Datasets (Gebru et al., 2021)

```python
"""
Datasheets for Datasets (Gebru et al., 2021)

Companion to Model Cards — focuses on the DATA rather than the model.
Just as electronics come with datasheets specifying operating conditions,
datasets should come with datasheets specifying their characteristics.

The template asks seven categories of questions:
1. Motivation — Why was the dataset created?
2. Composition — What does the dataset contain?
3. Collection — How was the data collected?
4. Preprocessing — What preprocessing was done?
5. Uses — What is the dataset intended for?
6. Distribution — How is the dataset shared?
7. Maintenance — Who maintains the dataset?

WHY Datasheets matter for interpretability:
  You cannot interpret a model without understanding its data.
  Biased data leads to biased models, and datasheets make
  data characteristics explicit and auditable.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DatasetDatasheet:
    """Datasheet template following Gebru et al. (2021)."""

    # Section 1: Motivation
    dataset_name: str = ""
    creators: str = ""
    funding: str = ""
    purpose: str = ""

    # Section 2: Composition
    instance_description: str = ""
    total_instances: int = 0
    is_sample: bool = False
    sampling_strategy: str = ""
    missing_data: str = ""
    confidentiality: str = ""
    content_warnings: str = ""

    # Section 3: Collection
    collection_mechanism: str = ""
    who_collected: str = ""
    time_period: str = ""
    consent: str = ""

    # Section 4: Preprocessing
    preprocessing_steps: list[str] = field(default_factory=list)
    raw_data_available: bool = False

    # Section 5: Uses
    intended_uses: list[str] = field(default_factory=list)
    not_recommended_uses: list[str] = field(default_factory=list)

    # Section 6: Distribution
    distribution_method: str = ""
    license: str = ""

    # Section 7: Maintenance
    maintainer: str = ""
    update_frequency: str = ""

    def to_markdown(self) -> str:
        """Generate Markdown-formatted datasheet."""
        lines = [
            f"# Datasheet: {self.dataset_name}\n",
            "## Motivation\n",
            f"- **Created by**: {self.creators}",
            f"- **Funded by**: {self.funding}",
            f"- **Purpose**: {self.purpose}\n",
            "## Composition\n",
            f"- **Instance description**: {self.instance_description}",
            f"- **Total instances**: {self.total_instances:,}",
            f"- **Is sample**: {'Yes — ' + self.sampling_strategy if self.is_sample else 'No (full population)'}",
            f"- **Missing data**: {self.missing_data}",
            f"- **Confidentiality**: {self.confidentiality}\n",
            "## Collection\n",
            f"- **Mechanism**: {self.collection_mechanism}",
            f"- **Collected by**: {self.who_collected}",
            f"- **Time period**: {self.time_period}",
            f"- **Consent**: {self.consent}\n",
        ]

        lines.append("## Preprocessing\n")
        for step in self.preprocessing_steps:
            lines.append(f"- {step}")
        lines.append(f"- **Raw data available**: {'Yes' if self.raw_data_available else 'No'}\n")

        lines.append("## Recommended Uses\n")
        for use in self.intended_uses:
            lines.append(f"- {use}")
        lines.append("\n## Not Recommended Uses\n")
        for use in self.not_recommended_uses:
            lines.append(f"- {use}")

        lines.extend([
            f"\n## Distribution\n",
            f"- **Method**: {self.distribution_method}",
            f"- **License**: {self.license}\n",
            f"## Maintenance\n",
            f"- **Maintainer**: {self.maintainer}",
            f"- **Update frequency**: {self.update_frequency}",
        ])

        return "\n".join(lines)


# Example datasheet
credit_datasheet = DatasetDatasheet(
    dataset_name="Consumer Loan Applications 2019-2023",
    creators="FinTech Corp Data Engineering Team",
    funding="Internal — FinTech Corp operational budget",
    purpose=(
        "Training and evaluating credit scoring models. Created to "
        "replace reliance on external bureau scores with an internal "
        "model that incorporates alternative data sources."
    ),
    instance_description=(
        "Each instance is a loan application. Features include: "
        "applicant demographics (age, gender, location), financial "
        "(income, debts, assets), employment (duration, type), "
        "and loan terms (amount, purpose, term)."
    ),
    total_instances=500_000,
    is_sample=False,
    missing_data=(
        "Employment type missing for 3.2% of instances. Income "
        "verification status missing for 8.1%. Gender missing for "
        "1.5% (not required in all application channels)."
    ),
    confidentiality=(
        "Contains PII. Access restricted to authorized personnel. "
        "All analysis must be conducted in the secure data environment."
    ),
    collection_mechanism="Collected through online and branch loan applications",
    who_collected="FinTech Corp, through its standard application process",
    time_period="January 2019 to December 2023",
    consent="Applicants consent to data use for credit assessment per Terms of Service",
    preprocessing_steps=[
        "Removed duplicate applications (same applicant, same day)",
        "Standardized income to annual EUR",
        "Geocoded addresses to NUTS-3 regions",
        "Imputed missing employment type using mode within age bracket",
    ],
    raw_data_available=True,
    intended_uses=[
        "Training credit scoring models for consumer loans",
        "Fairness evaluation and bias testing",
        "Model monitoring and concept drift detection",
    ],
    not_recommended_uses=[
        "Marketing or targeted advertising",
        "Employment decisions",
        "Training models for loan amounts > EUR 50,000 (distribution is sparse)",
        "Geographic analysis outside Germany (all applicants are German residents)",
    ],
    distribution_method="Internal data lake, access via approved SQL queries only",
    license="Proprietary — FinTech Corp internal use only",
    maintainer="Data Engineering Team (data-eng@fintechcorp.example.com)",
    update_frequency="Monthly append of new applications; annual full refresh",
)

print(credit_datasheet.to_markdown())
```

---

## 7. AI Impact Assessment

### 7.1 Impact Assessment Template

An AI Impact Assessment (AIA) is a structured process for evaluating the potential
impacts of an AI system before deployment. It combines elements of Data Protection
Impact Assessment (DPIA, required by GDPR Article 35), the NIST AI RMF, and
emerging best practices.

```python
"""
AI Impact Assessment Template

This template synthesizes requirements from:
  - GDPR Article 35 (DPIA) — mandatory for high-risk processing
  - EU AI Act Article 9 (risk management) — mandatory for high-risk AI
  - NIST AI RMF MAP function — voluntary but widely adopted
  - Canadian Algorithmic Impact Assessment (Treasury Board)

WHY conduct an impact assessment:
  1. Legal requirement (DPIA for GDPR, risk management for EU AI Act)
  2. Risk reduction (identify problems before deployment)
  3. Stakeholder trust (demonstrate due diligence)
  4. Organizational learning (systematic risk thinking)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ImpactLevel(Enum):
    NEGLIGIBLE = 1
    LOW = 2
    MODERATE = 3
    HIGH = 4
    VERY_HIGH = 5


class LikelihoodLevel(Enum):
    VERY_UNLIKELY = 1
    UNLIKELY = 2
    POSSIBLE = 3
    LIKELY = 4
    VERY_LIKELY = 5


@dataclass
class IdentifiedRisk:
    """A single identified risk in the impact assessment."""
    risk_id: str
    description: str
    affected_group: str
    impact: ImpactLevel
    likelihood: LikelihoodLevel
    mitigation: str
    residual_impact: ImpactLevel
    residual_likelihood: LikelihoodLevel
    owner: str

    @property
    def risk_score(self) -> int:
        """Risk score = impact * likelihood (standard risk matrix)."""
        return self.impact.value * self.likelihood.value

    @property
    def residual_risk_score(self) -> int:
        """Residual risk after mitigation."""
        return self.residual_impact.value * self.residual_likelihood.value


@dataclass
class AImpactAssessment:
    """Complete AI Impact Assessment document."""

    # System identification
    system_name: str = ""
    system_owner: str = ""
    assessment_date: str = ""
    assessor: str = ""

    # System description
    purpose: str = ""
    scope: str = ""
    affected_populations: list[str] = field(default_factory=list)
    decision_type: str = ""  # automated, semi-automated, advisory

    # Risks
    identified_risks: list[IdentifiedRisk] = field(default_factory=list)

    # Oversight
    human_oversight_plan: str = ""
    appeal_mechanism: str = ""
    monitoring_plan: str = ""

    # Decision
    recommendation: str = ""  # proceed, proceed with conditions, do not proceed

    def risk_summary(self) -> str:
        """Generate a risk summary table."""
        lines = [
            f"\n{'ID':6s} {'Risk Score':>10s} {'Residual':>10s} {'Status':>10s}",
            "-" * 40,
        ]
        for risk in sorted(self.identified_risks,
                          key=lambda r: r.risk_score, reverse=True):
            status = "ACCEPTABLE" if risk.residual_risk_score <= 6 else "MONITOR"
            if risk.residual_risk_score > 12:
                status = "UNACCEPTABLE"
            lines.append(
                f"{risk.risk_id:6s} {risk.risk_score:>10d} "
                f"{risk.residual_risk_score:>10d} {status:>10s}"
            )
        return "\n".join(lines)


# Example: Impact assessment for an automated hiring tool
hiring_aia = AImpactAssessment(
    system_name="TalentMatch AI v1.0",
    system_owner="HR Technology Team",
    assessment_date="2024-05-01",
    assessor="AI Governance Board",
    purpose="Screen job applications and rank candidates by predicted fit",
    scope="All entry-level positions in EU offices",
    affected_populations=["Job applicants (18-65)", "Hiring managers", "HR staff"],
    decision_type="Semi-automated (AI ranks, human makes final decision)",
    identified_risks=[
        IdentifiedRisk(
            risk_id="R-001",
            description="Gender bias in ranking due to historical hiring patterns",
            affected_group="Female applicants",
            impact=ImpactLevel.HIGH,
            likelihood=LikelihoodLevel.LIKELY,
            mitigation="Pre-deployment bias audit + quarterly monitoring",
            residual_impact=ImpactLevel.MODERATE,
            residual_likelihood=LikelihoodLevel.UNLIKELY,
            owner="Fairness Lead",
        ),
        IdentifiedRisk(
            risk_id="R-002",
            description="Age discrimination through proxy features (graduation year)",
            affected_group="Applicants over 40",
            impact=ImpactLevel.HIGH,
            likelihood=LikelihoodLevel.POSSIBLE,
            mitigation="Remove graduation year; test for age correlation",
            residual_impact=ImpactLevel.LOW,
            residual_likelihood=LikelihoodLevel.UNLIKELY,
            owner="ML Engineer",
        ),
        IdentifiedRisk(
            risk_id="R-003",
            description="Lack of transparency — candidates cannot understand ranking",
            affected_group="All applicants",
            impact=ImpactLevel.MODERATE,
            likelihood=LikelihoodLevel.VERY_LIKELY,
            mitigation="Provide factor-level explanation to each candidate",
            residual_impact=ImpactLevel.LOW,
            residual_likelihood=LikelihoodLevel.POSSIBLE,
            owner="Product Manager",
        ),
    ],
    human_oversight_plan=(
        "Hiring managers review AI-generated rankings but make all "
        "final decisions. AI ranking is advisory only — managers can "
        "override any ranking with documented justification."
    ),
    appeal_mechanism=(
        "Candidates may request explanation of their ranking and "
        "appeal through HR within 30 days of notification."
    ),
    recommendation="PROCEED WITH CONDITIONS",
)

print("AI IMPACT ASSESSMENT SUMMARY")
print("=" * 50)
print(f"System: {hiring_aia.system_name}")
print(f"Recommendation: {hiring_aia.recommendation}")
print(f"\nRisk Summary:{hiring_aia.risk_summary()}")
```

---

## 8. Industry Standards

### 8.1 IEEE 7000 and ISO/IEC 42001

```python
"""
Industry Standards for AI Governance

Two key standards provide frameworks for organizational AI governance:

1. IEEE 7000-2021: Model Process for Addressing Ethical Concerns
   During System Design
   - Defines a process for embedding ethical values into AI systems
   - Uses "value-based system design" methodology
   - Complements (but does not replace) regulatory requirements

2. ISO/IEC 42001:2023: AI Management System (AIMS)
   - First international standard for AI management systems
   - Based on the ISO management system structure (like ISO 27001)
   - Can be CERTIFIED — providing third-party assurance
   - Covers the entire AI lifecycle

WHY these standards matter:
  Standards provide AUDITABLE frameworks. When a regulator asks
  "how do you manage AI risk?", pointing to ISO 42001 certification
  is a concrete, defensible answer.
"""

# ISO/IEC 42001 structure mapped to AI lifecycle
ISO_42001_CLAUSES = {
    "Clause 4: Context of the Organization": {
        "4.1": "Understanding the organization and its context",
        "4.2": "Understanding stakeholder needs and expectations",
        "4.3": "Scope of the AI management system",
        "4.4": "AI management system",
        "ai_relevance": (
            "Identifies WHERE AI is used, WHO is affected, "
            "and WHAT regulations apply."
        ),
    },
    "Clause 5: Leadership": {
        "5.1": "Leadership and commitment",
        "5.2": "AI policy",
        "5.3": "Roles, responsibilities, and authorities",
        "ai_relevance": (
            "Top management must demonstrate commitment to responsible AI. "
            "An AI policy must be established and communicated."
        ),
    },
    "Clause 6: Planning": {
        "6.1": "Actions to address risks and opportunities",
        "6.2": "AI objectives and planning to achieve them",
        "ai_relevance": (
            "Risk assessment for AI systems. Maps to NIST AI RMF "
            "MAP function and EU AI Act Article 9."
        ),
    },
    "Clause 7: Support": {
        "7.1": "Resources",
        "7.2": "Competence",
        "7.3": "Awareness",
        "7.4": "Communication",
        "7.5": "Documented information",
        "ai_relevance": (
            "Ensures staff have AI-specific competencies. "
            "Documentation requirements align with EU AI Act Annex IV."
        ),
    },
    "Clause 8: Operation": {
        "8.1": "Operational planning and control",
        "8.2": "AI risk assessment",
        "8.3": "AI risk treatment",
        "8.4": "AI system impact assessment",
        "ai_relevance": (
            "Core operational requirements. This is where the actual "
            "AI development and deployment controls are specified."
        ),
    },
    "Clause 9: Performance Evaluation": {
        "9.1": "Monitoring, measurement, analysis, and evaluation",
        "9.2": "Internal audit",
        "9.3": "Management review",
        "ai_relevance": (
            "Ongoing monitoring of AI systems. Maps to NIST MEASURE "
            "function and EU AI Act post-market monitoring."
        ),
    },
    "Clause 10: Improvement": {
        "10.1": "Continual improvement",
        "10.2": "Nonconformity and corrective action",
        "ai_relevance": (
            "Learning from incidents and improving the AI management "
            "system. Maps to NIST MANAGE function."
        ),
    },
}


def print_iso_42001_overview():
    """Print the ISO/IEC 42001 structure with AI relevance notes."""
    print("ISO/IEC 42001:2023 — AI Management System")
    print("=" * 60)
    for clause_name, details in ISO_42001_CLAUSES.items():
        print(f"\n{clause_name}")
        for key, value in details.items():
            if key == "ai_relevance":
                print(f"  AI Relevance: {value}")
            else:
                print(f"  {key}: {value}")

print_iso_42001_overview()
```

---

## 9. Practical: Writing a Model Card and Compliance Check

### 9.1 End-to-End Compliance Workflow

This practical exercise demonstrates how to create regulatory documentation
for a real AI system, combining Model Cards, bias audits, and impact assessments.

```python
"""
Practical Exercise: Regulatory Compliance for a Loan Approval Model

This code demonstrates a complete compliance workflow:
1. Train a simple model on the German Credit dataset
2. Generate a Model Card
3. Conduct a bias audit (NYC LL144 style)
4. Perform an EU AI Act risk classification
5. Check GDPR Article 22 applicability

WHY this is important:
  In production, these steps are not optional — they are legal
  requirements (EU AI Act, GDPR) or industry best practice
  (Model Cards, NIST AI RMF). Automating as much as possible
  reduces compliance burden and improves consistency.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from collections import defaultdict


def create_synthetic_credit_data(n_samples=5000, random_state=42):
    """Create synthetic credit data with demographic attributes.

    We use synthetic data because:
    1. Real credit data is sensitive and cannot be shared
    2. We can control the bias characteristics for educational purposes
    3. The workflow is identical regardless of data source
    """
    np.random.seed(random_state)

    # Generate base features
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        random_state=random_state,
        flip_y=0.1,  # 10% label noise — realistic for credit data
    )

    # Add demographic attributes (not used in model, used for audit)
    # These represent protected characteristics
    gender = np.random.choice(["male", "female"], size=n_samples, p=[0.55, 0.45])
    age_group = np.random.choice(
        ["18-30", "31-50", "51+"],
        size=n_samples,
        p=[0.3, 0.45, 0.25],
    )

    # Introduce realistic bias: slight correlation between gender and approval
    # This simulates the kind of bias compliance testing should detect
    bias_mask = (gender == "female") & (np.random.random(n_samples) < 0.08)
    y[bias_mask] = 0  # Slightly higher rejection for one group

    feature_names = [
        "income", "employment_years", "debt_ratio", "credit_history",
        "loan_amount", "savings", "housing_status", "num_dependents",
        "existing_credits", "installment_rate",
    ]

    return X, y, gender, age_group, feature_names


def train_credit_model(X_train, y_train):
    """Train a gradient boosting model for credit scoring.

    GBM is chosen because:
    1. Strong performance on tabular data
    2. Native feature importance (aids transparency)
    3. Widely used in industry credit scoring
    """
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def conduct_bias_audit(model, X_test, y_test, demographics, group_name):
    """Conduct a bias audit following NYC LL144 methodology.

    This function:
    1. Calculates selection rates per demographic group
    2. Computes impact ratios
    3. Flags groups failing the 4/5ths rule

    Returns a structured audit report.
    """
    y_pred = model.predict(X_test)

    # Group predictions by demographic category
    group_results = defaultdict(lambda: {"total": 0, "selected": 0})

    for pred, group in zip(y_pred, demographics):
        group_results[group]["total"] += 1
        if pred == 1:  # "Selected" (approved)
            group_results[group]["selected"] += 1

    # Calculate selection rates
    rates = {}
    for group, counts in group_results.items():
        rate = counts["selected"] / counts["total"] if counts["total"] > 0 else 0
        rates[group] = {
            "selection_rate": rate,
            "total": counts["total"],
            "selected": counts["selected"],
        }

    # Calculate impact ratios
    max_rate = max(r["selection_rate"] for r in rates.values())

    audit_report = {
        "group_name": group_name,
        "reference_rate": max_rate,
        "groups": {},
    }

    for group, data in rates.items():
        impact_ratio = data["selection_rate"] / max_rate if max_rate > 0 else 0
        audit_report["groups"][group] = {
            **data,
            "impact_ratio": impact_ratio,
            "passes_4_5ths": impact_ratio >= 0.80,
        }

    return audit_report


def generate_compliance_report(model, X_test, y_test, gender_test,
                                age_test, feature_names):
    """Generate a complete compliance report.

    This combines:
    - Model performance metrics
    - Feature importance (for transparency)
    - Bias audit results
    - EU AI Act classification
    - GDPR Article 22 assessment
    """
    # Model performance
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print("COMPLIANCE REPORT — Credit Scoring Model")
    print("=" * 60)

    # Section 1: Performance
    print("\n1. MODEL PERFORMANCE")
    print(f"   Accuracy: {accuracy:.3f}")
    print(f"   AUC-ROC:  {auc:.3f}")

    # Section 2: Feature Importance (transparency)
    print("\n2. FEATURE IMPORTANCE (Top 5)")
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for i in range(min(5, len(feature_names))):
        idx = sorted_idx[i]
        print(f"   {feature_names[idx]:25s}: {importances[idx]:.4f}")

    # Section 3: Bias Audit
    print("\n3. BIAS AUDIT (NYC LL144 Methodology)")

    gender_audit = conduct_bias_audit(model, X_test, y_test, gender_test, "Gender")
    age_audit = conduct_bias_audit(model, X_test, y_test, age_test, "Age Group")

    for audit in [gender_audit, age_audit]:
        print(f"\n   Category: {audit['group_name']}")
        print(f"   {'Group':12s} {'Rate':>8s} {'Impact':>8s} {'4/5ths':>8s}")
        for group, data in audit["groups"].items():
            flag = "PASS" if data["passes_4_5ths"] else "FAIL"
            print(f"   {group:12s} {data['selection_rate']:>7.1%} "
                  f"{data['impact_ratio']:>7.2f}  {flag:>6s}")

    # Section 4: EU AI Act Classification
    print("\n4. EU AI ACT CLASSIFICATION")
    print("   Risk Level: HIGH")
    print("   Rationale: Annex III, Area 5(b) — creditworthiness evaluation")
    print("   Required: Conformity assessment, technical documentation,")
    print("             risk management, data governance, human oversight")

    # Section 5: GDPR Article 22
    print("\n5. GDPR ARTICLE 22 ASSESSMENT")
    print("   Solely automated: YES (if no human review in loop)")
    print("   Significant effects: YES (credit decisions)")
    print("   Applicable: YES — must provide meaningful explanation")
    print("   Required safeguards:")
    print("     - Meaningful information about logic involved")
    print("     - Right to human intervention")
    print("     - Right to contest the decision")

    return accuracy, auc


# Run the complete workflow
print("Creating synthetic credit data...")
X, y, gender, age_group, feature_names = create_synthetic_credit_data()

print("Splitting data...")
X_train, X_test, y_train, y_test, gender_train, gender_test, age_train, age_test = (
    train_test_split(X, y, gender, age_group, test_size=0.2, random_state=42)
)

print("Training model...")
model = train_credit_model(X_train, y_train)

print("Generating compliance report...\n")
accuracy, auc = generate_compliance_report(
    model, X_test, y_test, gender_test, age_test, feature_names
)
```

### 9.2 Automated Compliance Checklist

```python
"""
Automated Compliance Checklist

This checklist aggregates requirements from multiple regulatory
frameworks into a single actionable list. In production, this
would be integrated into CI/CD pipelines to catch compliance
gaps before deployment.
"""

from dataclasses import dataclass
from enum import Enum


class ComplianceStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_APPLICABLE = "N/A"
    NEEDS_REVIEW = "REVIEW"


@dataclass
class ComplianceCheck:
    """A single compliance check item."""
    framework: str       # Which regulation/standard
    requirement: str     # What is required
    status: ComplianceStatus
    evidence: str        # How compliance is demonstrated
    notes: str = ""


def run_compliance_checklist() -> list[ComplianceCheck]:
    """Run a comprehensive compliance checklist.

    In production, many of these checks would be automated:
    - Model Card existence: check file system
    - Bias audit: run statistical tests
    - Logging: verify logging configuration
    - Performance metrics: compute from test set
    """
    checks = [
        # EU AI Act
        ComplianceCheck(
            framework="EU AI Act",
            requirement="Risk classification documented",
            status=ComplianceStatus.PASS,
            evidence="Classification: HIGH (Annex III, 5b)",
        ),
        ComplianceCheck(
            framework="EU AI Act",
            requirement="Technical documentation (Annex IV)",
            status=ComplianceStatus.PASS,
            evidence="AnnexIVDocumentation template completed",
        ),
        ComplianceCheck(
            framework="EU AI Act",
            requirement="Risk management system (Article 9)",
            status=ComplianceStatus.PASS,
            evidence="AI Impact Assessment completed with 3 identified risks",
        ),
        ComplianceCheck(
            framework="EU AI Act",
            requirement="Data governance (Article 10)",
            status=ComplianceStatus.NEEDS_REVIEW,
            evidence="Dataset Datasheet created; representativeness TBD",
            notes="Need demographic representativeness analysis",
        ),
        ComplianceCheck(
            framework="EU AI Act",
            requirement="Human oversight (Article 14)",
            status=ComplianceStatus.PASS,
            evidence="Human-in-the-loop for all decisions > EUR 10,000",
        ),
        ComplianceCheck(
            framework="EU AI Act",
            requirement="Logging (Article 12)",
            status=ComplianceStatus.FAIL,
            evidence="Logging not yet implemented",
            notes="BLOCKER: Must implement before deployment",
        ),

        # GDPR
        ComplianceCheck(
            framework="GDPR Art. 22",
            requirement="Meaningful explanation capability",
            status=ComplianceStatus.PASS,
            evidence="SHAP-based explanations for each decision",
        ),
        ComplianceCheck(
            framework="GDPR Art. 22",
            requirement="Right to contest mechanism",
            status=ComplianceStatus.PASS,
            evidence="Appeal form available; human review within 5 days",
        ),
        ComplianceCheck(
            framework="GDPR Art. 35",
            requirement="Data Protection Impact Assessment",
            status=ComplianceStatus.NEEDS_REVIEW,
            evidence="DPIA drafted; awaiting DPO sign-off",
        ),

        # Documentation
        ComplianceCheck(
            framework="Best Practice",
            requirement="Model Card",
            status=ComplianceStatus.PASS,
            evidence="Model Card v2.1 in model registry",
        ),
        ComplianceCheck(
            framework="Best Practice",
            requirement="Dataset Datasheet",
            status=ComplianceStatus.PASS,
            evidence="Datasheet in data catalog",
        ),
        ComplianceCheck(
            framework="NYC LL144",
            requirement="Bias audit (if used for employment)",
            status=ComplianceStatus.NOT_APPLICABLE,
            evidence="System used for credit, not employment",
        ),
    ]
    return checks


def print_compliance_dashboard(checks: list[ComplianceCheck]):
    """Print a compliance dashboard summary."""
    print("\nCOMPLIANCE DASHBOARD")
    print("=" * 70)

    # Summary counts
    counts = {}
    for check in checks:
        counts[check.status] = counts.get(check.status, 0) + 1

    total = len(checks)
    passed = counts.get(ComplianceStatus.PASS, 0)
    failed = counts.get(ComplianceStatus.FAIL, 0)
    review = counts.get(ComplianceStatus.NEEDS_REVIEW, 0)
    na = counts.get(ComplianceStatus.NOT_APPLICABLE, 0)

    print(f"\nTotal Checks: {total}")
    print(f"  PASS:   {passed:3d} ({passed/total:.0%})")
    print(f"  FAIL:   {failed:3d} ({failed/total:.0%})")
    print(f"  REVIEW: {review:3d} ({review/total:.0%})")
    print(f"  N/A:    {na:3d}")

    deployment_ready = failed == 0
    print(f"\nDeployment Ready: {'YES' if deployment_ready else 'NO — resolve FAILs first'}")

    # Detailed results
    print(f"\n{'Framework':15s} {'Requirement':40s} {'Status':>8s}")
    print("-" * 70)
    for check in checks:
        print(f"{check.framework:15s} {check.requirement:40s} {check.status.value:>8s}")
        if check.notes:
            print(f"{'':15s}   NOTE: {check.notes}")


checks = run_compliance_checklist()
print_compliance_dashboard(checks)
```

---

## Summary

- The **EU AI Act** establishes a risk-based framework with four categories (unacceptable, high, limited, minimal), where high-risk systems face extensive obligations including conformity assessment, technical documentation, and human oversight
- **GDPR Article 22** grants individuals the right not to be subject to solely automated decisions with significant effects, requiring "meaningful information about the logic involved" — the legal basis for AI explanations in the EU
- The **NIST AI RMF** provides a voluntary four-function framework (Govern, Map, Measure, Manage) that organizations can adopt regardless of jurisdiction to systematically manage AI risks
- **NYC Local Law 144** pioneered concrete bias audit requirements for automated employment decision tools, requiring calculation of selection rates and impact ratios across demographic groups
- **Model Cards** (Mitchell et al., 2019) and **Datasheets for Datasets** (Gebru et al., 2021) provide standardized templates for documenting model behavior and data characteristics
- **AI Impact Assessments** combine risk identification, stakeholder analysis, and mitigation planning into a structured pre-deployment review process
- **Industry standards** like ISO/IEC 42001 provide certifiable AI management system frameworks that complement regulatory requirements
- Compliance is not a one-time event — it requires continuous monitoring, documentation updates, and organizational governance structures

---

## Exercises

### Exercise 1: Risk Classification Challenge

Given the following AI systems, classify each under the EU AI Act:

1. An AI system that monitors employee productivity by tracking keystrokes and screen time
2. A recommender system for a music streaming service
3. An AI tool that helps judges set bail amounts based on defendant risk profiles
4. A chatbot that helps customers choose the right shoe size
5. An AI system used in autonomous vehicle emergency braking

For each, determine: (a) risk level, (b) applicable Annex III category (if high-risk), (c) key obligations.

### Exercise 2: GDPR Article 22 Analysis

A health insurance company uses an AI system to automatically set premium rates based on customer-provided health questionnaire data, wearable device data, and publicly available lifestyle information. There is no human review of individual pricing decisions.

1. Does GDPR Article 22 apply? Walk through each condition
2. What exceptions might the insurer rely on?
3. Draft a Level 2 explanation for a customer whose premium was increased
4. Design an appeal mechanism that satisfies Article 22(3) requirements

### Exercise 3: Complete Model Card

Train a model of your choice on a public dataset (e.g., UCI Adult, COMPAS, German Credit). Create a complete Model Card following the Mitchell et al. template, including:

1. Disaggregated performance metrics for at least 3 demographic groups
2. At least 3 out-of-scope uses with justification
3. A concrete "what would change the outcome" counterfactual explanation
4. An honest assessment of the model's limitations and ethical concerns

### Exercise 4: Bias Audit Implementation

Implement a complete NYC LL144-compliant bias audit for a hiring model:

1. Generate synthetic hiring data with gender and race/ethnicity attributes
2. Train a classifier to predict "hire" decisions
3. Calculate selection rates and impact ratios for ALL demographic groups
4. Identify any groups failing the 4/5ths rule
5. Propose specific mitigation strategies for each failing group
6. Re-run the audit after applying one mitigation strategy

### Exercise 5: Regulatory Mapping

You are deploying a credit scoring AI system that will be used by a German bank for EU customers. Create a compliance matrix that maps requirements across:

1. EU AI Act (high-risk obligations)
2. GDPR (Articles 13, 14, 22, 35)
3. German Federal Data Protection Act (BDSG Section 37)
4. NIST AI RMF (all four functions)
5. ISO/IEC 42001 (relevant clauses)

For each requirement, identify: (a) what is needed, (b) who is responsible, (c) what evidence demonstrates compliance, and (d) the monitoring frequency.

---

[Previous: Fairness Mitigation](./12_Fairness_Mitigation.md) | [Overview](./00_Overview.md) | [Next: Production Interpretability](./14_Production_Interpretability.md)

**License**: CC BY-NC 4.0
