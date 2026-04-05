"""
Exercises for Lesson 01: Interpretability Foundations
Topic: Interpretable_AI

Solutions to practice problems from the lesson.
"""

import numpy as np


# === Exercise 1: Classify Explanation Methods by Type ===
# Problem: Given a list of explanation methods, classify each as
# intrinsic vs post-hoc and local vs global. Score each on a
# faithfulness-comprehensibility spectrum (1-10 scale).

def exercise_1():
    """Classify explanation methods by type and scope."""
    print("=" * 60)
    print("Exercise 1: Classify Explanation Methods by Type")
    print("=" * 60)

    methods = {
        "Decision Tree": {
            "intrinsic_or_posthoc": "intrinsic",
            "local_or_global": "global",
            "faithfulness": 10,
            "comprehensibility": 9,
            "description": (
                "The tree structure IS the model; the explanation "
                "perfectly mirrors reasoning (high faithfulness). "
                "Easy to visualize and follow (high comprehensibility)."
            ),
        },
        "LIME": {
            "intrinsic_or_posthoc": "post-hoc",
            "local_or_global": "local",
            "faithfulness": 5,
            "comprehensibility": 8,
            "description": (
                "Approximates local behavior with a simple surrogate "
                "(moderate faithfulness since it's an approximation). "
                "Output is a short list of weighted features (comprehensible)."
            ),
        },
        "SHAP (KernelSHAP)": {
            "intrinsic_or_posthoc": "post-hoc",
            "local_or_global": "local",
            "faithfulness": 7,
            "comprehensibility": 7,
            "description": (
                "Grounded in Shapley axioms giving theoretical guarantees "
                "(good faithfulness). Feature importance values are "
                "understandable but require statistical literacy."
            ),
        },
        "Attention Weights": {
            "intrinsic_or_posthoc": "intrinsic",
            "local_or_global": "local",
            "faithfulness": 4,
            "comprehensibility": 8,
            "description": (
                "Directly from the model but debated whether attention "
                "truly explains decisions (low faithfulness). "
                "Heatmaps are visually intuitive (high comprehensibility)."
            ),
        },
        "Linear Regression Coefficients": {
            "intrinsic_or_posthoc": "intrinsic",
            "local_or_global": "global",
            "faithfulness": 10,
            "comprehensibility": 9,
            "description": (
                "Coefficients ARE the model's parameters; they define "
                "the decision boundary exactly. Easy to interpret for "
                "domain experts."
            ),
        },
    }

    print(f"\n{'Method':<30} {'Type':<12} {'Scope':<8} "
          f"{'Faith.':<8} {'Compr.':<8}")
    print("-" * 70)
    for name, info in methods.items():
        print(f"{name:<30} {info['intrinsic_or_posthoc']:<12} "
              f"{info['local_or_global']:<8} "
              f"{info['faithfulness']:<8} {info['comprehensibility']:<8}")

    print("\nDetailed reasoning:")
    for name, info in methods.items():
        print(f"\n  {name}:")
        print(f"    {info['description']}")


# === Exercise 2: Faithfulness-Comprehensibility Tradeoff ===
# Problem: Simulate the tradeoff curve. Given model complexity,
# compute expected faithfulness and comprehensibility, then find
# the Pareto-optimal explanation complexity.

def exercise_2():
    """Compute faithfulness-comprehensibility tradeoff curve."""
    print("\n" + "=" * 60)
    print("Exercise 2: Faithfulness-Comprehensibility Tradeoff")
    print("=" * 60)

    # Model: a deep neural network with given complexity score
    model_complexity = 100  # arbitrary units

    # Explanation complexity ranges from 1 (very simple) to 100 (full model)
    explanation_complexities = np.arange(1, 101)

    # Faithfulness: how well explanation reflects model behavior
    # Increases with explanation complexity (sigmoid shape)
    faithfulness = 1.0 / (1.0 + np.exp(-0.1 * (explanation_complexities - 50)))

    # Comprehensibility: how easy for humans to understand
    # Decreases with explanation complexity (inverse sigmoid)
    comprehensibility = 1.0 / (1.0 + np.exp(0.08 * (explanation_complexities - 40)))

    # Combined score (harmonic mean to penalize imbalance)
    combined = (2 * faithfulness * comprehensibility /
                (faithfulness + comprehensibility + 1e-10))

    best_idx = np.argmax(combined)
    best_complexity = explanation_complexities[best_idx]

    print(f"\n  Model complexity: {model_complexity}")
    print(f"\n  Tradeoff curve (sampled points):")
    print(f"  {'Expl. Complexity':<20} {'Faithfulness':<15} "
          f"{'Comprehensibility':<20} {'Combined':<10}")
    print("  " + "-" * 65)

    sample_points = [1, 10, 25, 40, 50, 60, 75, 90, 100]
    for c in sample_points:
        idx = c - 1
        print(f"  {c:<20} {faithfulness[idx]:<15.4f} "
              f"{comprehensibility[idx]:<20.4f} {combined[idx]:<10.4f}")

    print(f"\n  Optimal explanation complexity: {best_complexity}")
    print(f"  At optimum: faithfulness={faithfulness[best_idx]:.4f}, "
          f"comprehensibility={comprehensibility[best_idx]:.4f}, "
          f"combined={combined[best_idx]:.4f}")
    print(f"\n  Insight: The sweet spot balances enough detail to be "
          f"faithful without overwhelming the user.")


# === Exercise 3: Map Methods to Lipton's Taxonomy ===
# Problem: Lipton (2018) distinguishes transparency (simulatability,
# decomposability, algorithmic transparency) from post-hoc explanations
# (text, visualization, local, example-based). Map given methods.

def exercise_3():
    """Map explanation methods to Lipton's taxonomy categories."""
    print("\n" + "=" * 60)
    print("Exercise 3: Map Methods to Lipton's Taxonomy")
    print("=" * 60)

    taxonomy = {
        "Transparency": {
            "Simulatability": {
                "definition": (
                    "A human can mentally step through the model's "
                    "computation in reasonable time."
                ),
                "examples": [
                    "Small decision tree (depth <= 5)",
                    "Short rule list (< 10 rules)",
                    "Sparse linear model (< 10 features)",
                ],
            },
            "Decomposability": {
                "definition": (
                    "Each part of the model (input, parameter, computation) "
                    "has an intuitive explanation."
                ),
                "examples": [
                    "Generalized additive models (GAMs)",
                    "Naive Bayes (independent feature contributions)",
                    "Linear regression with meaningful features",
                ],
            },
            "Algorithmic Transparency": {
                "definition": (
                    "The learning algorithm itself has properties that "
                    "guarantee understandable behavior."
                ),
                "examples": [
                    "Linear regression (unique global optimum, convex loss)",
                    "k-NN (decision based on similarity, no training phase)",
                    "Finite convergence guarantees of decision trees",
                ],
            },
        },
        "Post-hoc Explanations": {
            "Text Explanations": {
                "definition": "Natural language rationales for predictions.",
                "examples": [
                    "Chain-of-thought prompting in LLMs",
                    "Generated explanations from explanation models",
                ],
            },
            "Visualization": {
                "definition": "Visual rendering of learned representations.",
                "examples": [
                    "Saliency maps / GradCAM",
                    "t-SNE / UMAP of hidden representations",
                    "Feature visualization (activation maximization)",
                ],
            },
            "Local Explanations": {
                "definition": "Explain individual predictions.",
                "examples": [
                    "LIME (local linear approximation)",
                    "SHAP values (per-instance feature attribution)",
                    "Counterfactual explanations",
                ],
            },
            "Example-based": {
                "definition": (
                    "Explain by pointing to similar or influential "
                    "training examples."
                ),
                "examples": [
                    "Influence functions",
                    "Prototype-based explanations (ProtoPNet)",
                    "Nearest-neighbor retrieval",
                ],
            },
        },
    }

    for category, subcats in taxonomy.items():
        print(f"\n  [{category}]")
        for subcat_name, details in subcats.items():
            print(f"\n    {subcat_name}:")
            print(f"      Definition: {details['definition']}")
            print(f"      Examples:")
            for ex in details["examples"]:
                print(f"        - {ex}")


# === Exercise 4: Determine Regulatory Requirements ===
# Problem: Given descriptions of AI systems, determine what level of
# interpretability is required by regulations (GDPR, EU AI Act, ECOA)
# and recommend appropriate explanation methods.

def exercise_4():
    """Determine regulatory interpretability requirements for AI systems."""
    print("\n" + "=" * 60)
    print("Exercise 4: Regulatory Requirements for AI Systems")
    print("=" * 60)

    systems = [
        {
            "name": "Credit Scoring Model",
            "description": (
                "A gradient-boosted tree model used by a bank to decide "
                "loan approvals for individual consumers."
            ),
            "regulations": ["GDPR Art. 22", "ECOA / Reg B", "EU AI Act (High-risk)"],
            "required_level": "High",
            "reasoning": (
                "Automated individual decision-making with legal effects. "
                "GDPR requires meaningful information about the logic involved. "
                "ECOA requires specific adverse action reasons. "
                "EU AI Act classifies credit scoring as high-risk AI."
            ),
            "recommended_methods": [
                "SHAP values for per-applicant feature importance",
                "Counterfactual explanations (what to change for approval)",
                "Global feature importance for model documentation",
            ],
        },
        {
            "name": "Content Recommendation Engine",
            "description": (
                "A collaborative filtering system recommending movies "
                "on a streaming platform."
            ),
            "regulations": ["GDPR Art. 13/14 (transparency)", "DSA (EU)"],
            "required_level": "Medium",
            "reasoning": (
                "Profiling-based recommendations require transparency about "
                "the logic and significance. DSA requires labeling of "
                "recommender system parameters. No adverse legal effect, "
                "so less stringent than credit scoring."
            ),
            "recommended_methods": [
                "Content-based explanations (recommended because of X)",
                "Similar-user explanations (users like you also watched)",
                "System-level transparency documentation",
            ],
        },
        {
            "name": "Medical Image Diagnosis",
            "description": (
                "A CNN-based system that detects tumors in radiology images, "
                "used as a clinical decision support tool."
            ),
            "regulations": ["EU AI Act (High-risk)", "FDA SaMD guidance", "MDR"],
            "required_level": "High",
            "reasoning": (
                "Health domain is high-risk under EU AI Act. FDA requires "
                "clinical validation and interpretability for Software as "
                "Medical Device. Clinicians need to understand and validate."
            ),
            "recommended_methods": [
                "GradCAM for spatial localization of detected features",
                "Concept-based explanations (TCAV) for clinical concepts",
                "Prototype explanations showing similar known cases",
            ],
        },
        {
            "name": "Spam Email Filter",
            "description": (
                "A logistic regression model classifying incoming emails "
                "as spam or not-spam for a personal inbox."
            ),
            "regulations": ["Minimal regulatory requirements"],
            "required_level": "Low",
            "reasoning": (
                "No significant legal or safety impact. Users can review "
                "spam folder. Intrinsic model transparency (logistic "
                "regression coefficients) may suffice."
            ),
            "recommended_methods": [
                "Model coefficients (intrinsically interpretable)",
                "Highlighted spam trigger words in the email",
            ],
        },
    ]

    for system in systems:
        print(f"\n  System: {system['name']}")
        print(f"  Description: {system['description']}")
        print(f"  Applicable regulations: {', '.join(system['regulations'])}")
        print(f"  Required interpretability level: {system['required_level']}")
        print(f"  Reasoning: {system['reasoning']}")
        print(f"  Recommended methods:")
        for method in system["recommended_methods"]:
            print(f"    - {method}")
        print()


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
