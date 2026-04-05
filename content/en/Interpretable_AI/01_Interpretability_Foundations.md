# Lesson 1: Interpretability Foundations

[Overview](./00_Overview.md) | [Next: Gradient Attribution](./02_Gradient_Attribution.md)

---

## Learning Objectives

- Distinguish Lipton's three notions of interpretability: transparency, simulatability, and decomposability
- Apply the Doshi-Velez & Kim evaluation framework to select appropriate explanation methods
- Classify explanation outputs into five types: feature attribution, rules, examples, counterfactuals, and concepts
- Analyze the faithfulness-comprehensibility tension in practical explanation design
- Map the current interpretability research landscape and identify when interpretability is legally required

---

Interpretable AI has moved from a niche research curiosity to a central pillar of
responsible machine learning. Regulators, clinicians, loan officers, and judges
increasingly demand that AI systems justify their decisions. Yet the word
"interpretability" itself is notoriously vague. This lesson establishes the
conceptual foundations you need before diving into specific methods like gradient
attribution (Lesson 02), class activation mapping (Lesson 03), or attention
analysis (Lesson 04).

We begin with Zachary Lipton's influential taxonomy, move to formal evaluation
criteria from Doshi-Velez and Kim, catalog the types of explanation outputs that
methods can produce, confront the core tension between faithfulness and
comprehensibility, and close with a map of the research landscape and a
discussion of when interpretability is legally or ethically mandatory.

---

## 1. Bridge from Machine Learning Lesson 16

If you have completed **[Machine Learning Lesson 16: Model Explainability](../Machine_Learning/16_Model_Explainability.md)**,
you already understand the practical toolkit of post-hoc explainability: SHAP
values for feature attribution, LIME for local linear approximations, and
PDP/ICE/ALE plots for visualizing marginal feature effects. Those methods answer
the question *"which features drove this prediction?"* and remain indispensable
in day-to-day ML work.

This lesson does **not** re-teach those methods. Instead, we step back and ask a
deeper set of questions: *What does it even mean for a model to be
"interpretable"? How do we evaluate whether an explanation is any good? What
types of explanations exist beyond feature attribution?* The answers form the
theoretical scaffolding on which every subsequent lesson in this topic builds.
Where ML L16 gave you tools, this lesson gives you the conceptual framework to
choose, evaluate, and critique those tools.

---

## 2. What Does "Interpretability" Mean?

The term "interpretability" is used so loosely in the ML literature that it risks
becoming meaningless. Zachary Lipton's 2018 paper *"The Mythos of Model
Interpretability"* remains the most cited attempt to bring precision to the
discussion.

### 2.1 Lipton's Three Notions of Interpretability

Lipton identifies three distinct properties that people conflate under the
umbrella of "interpretability":

```python
"""
Lipton's Three Notions of Interpretability (2018)

1. TRANSPARENCY — Can we understand the model mechanism?
   Three sub-types:
   a) Simulatability:  Can a human step through the entire model in
                       reasonable time? (e.g., a 3-rule decision list)
   b) Decomposability: Can each component (feature, parameter, layer)
                       be given an intuitive explanation?
   c) Algorithmic transparency: Do we understand the training algorithm's
                       convergence properties? (e.g., linear regression
                       has a unique global optimum; NNs do not)

2. POST-HOC EXPLANATIONS — Can we explain predictions after the fact?
   Methods that do NOT require understanding internals:
   - Feature attribution (SHAP, LIME)
   - Saliency maps (gradient-based)
   - Example-based explanations (prototypes, criticisms)
   - Rule extraction from neural networks

3. TRUST — Does the user believe the model is reliable?
   Trust can come from:
   - Transparency (I understand it, so I trust it)
   - Post-hoc explanations (the explanation seems reasonable)
   - Performance on held-out data (it gets the right answers)
   - Domain knowledge alignment (its reasoning matches mine)

Key insight: These three notions are INDEPENDENT. A model can be trusted
without being transparent (e.g., well-validated deep learning in radiology).
A model can be transparent but not trusted (e.g., linear regression with
garbage features).
"""
```

### 2.2 Transparency: The Three Sub-Types in Detail

Let us examine each transparency sub-type with concrete examples:

```python
"""
=== Sub-Type A: SIMULATABILITY ===

Definition: A model is simulatable if a human can take the input, step through
            the entire computation, and arrive at the output in reasonable time.

Examples of simulatable models:
  - Decision tree with 5 nodes
  - Linear model with 3 features: y = 0.5*age + 0.3*income - 0.2*debt
  - Rule list: IF income > 50k AND debt < 10k THEN approve

Examples of NON-simulatable models:
  - Random forest with 500 trees, each having 100+ leaves
  - Any neural network (even a small MLP with 2 hidden layers)
  - Gradient boosting with 1000 estimators

Critical question: Is simulatability binary or a spectrum?
  → It is a spectrum. A decision tree with 10 nodes is marginally simulatable;
    one with 1000 nodes is not. The threshold depends on the human's expertise
    and patience.


=== Sub-Type B: DECOMPOSABILITY ===

Definition: Each part of the model (input features, parameters, intermediate
            computations) admits an intuitive explanation.

Requires:
  - Features must be human-understandable (not raw pixels or embeddings)
  - Parameters must be meaningful (regression coefficients often are;
    neural network weights in hidden layers are not)
  - Intermediate representations must be interpretable

Example: Linear regression on tabular data with named features
  - coefficient for "age" = 0.05 → "each additional year of age increases
    the predicted value by 0.05 units"
  - This works because both the feature (age) and the parameter (0.05) are
    meaningful to a human

Counter-example: CNN on raw pixels
  - Input: pixel (127, 234) has value 0.45 → not meaningful to humans
  - Weight in conv layer 3, filter 17, position (2,3) = -0.012 → opaque
  - Neither the feature nor the parameter is decomposable


=== Sub-Type C: ALGORITHMIC TRANSPARENCY ===

Definition: We understand how the learning algorithm works: its convergence
            properties, what objective it optimizes, and what solution it finds.

Examples:
  - Linear regression: closed-form solution via normal equations, unique
    global optimum, well-understood bias-variance trade-off
  - k-NN: no training at all; prediction is a simple majority vote
  - SVM: convex optimization, unique global optimum (for given kernel)

Counter-examples:
  - Deep learning: non-convex loss landscape, stochastic optimization,
    sensitive to initialization, no guarantee of finding global optimum
  - We understand gradient descent, but we do NOT fully understand why
    specific minima generalize well (the "lottery ticket" hypothesis,
    "grokking" phenomenon, etc.)
"""
```

### 2.3 Why These Distinctions Matter

```python
"""
Practical impact of Lipton's taxonomy:

Scenario 1: Loan approval in the EU (GDPR Article 22)
  → Regulators require that individuals receive "meaningful information about
    the logic involved" in automated decisions.
  → This demands at minimum POST-HOC EXPLANATIONS, and arguably DECOMPOSABILITY.
  → Simulatability is nice but not strictly required by law.

Scenario 2: Medical diagnosis support
  → Clinicians need to verify whether the model's reasoning aligns with
    medical knowledge.
  → DECOMPOSABILITY matters: the explanation should reference medically
    meaningful features (tumor size, margin type) not pixel indices.
  → TRUST must be calibrated: neither blind trust nor blanket rejection.

Scenario 3: Autonomous driving safety certification
  → ALGORITHMIC TRANSPARENCY matters: regulators want to understand what
    objective the model optimizes and whether it can fail catastrophically.
  → POST-HOC EXPLANATIONS help after incidents (why did the car brake?).
  → Full SIMULATABILITY is impossible for real-time perception models.

Takeaway: Different stakeholders need different types of interpretability.
No single notion covers all use cases.
"""
```

---

## 3. The Doshi-Velez & Kim Evaluation Framework

While Lipton clarifies what interpretability *means*, Doshi-Velez and Kim (2017)
address how to *evaluate* whether an explanation method actually works. Their
paper *"Towards a Rigorous Science of Interpretable Machine Learning"* proposes
a three-level evaluation taxonomy.

### 3.1 Three Levels of Evaluation

```python
"""
Doshi-Velez & Kim Evaluation Taxonomy (2017)

Level 1: APPLICATION-GROUNDED EVALUATION
  - Test with real humans on real tasks
  - Example: Show radiologists two explanation methods and measure which
    one leads to more accurate diagnoses
  - Gold standard but expensive and domain-specific
  - Requires IRB approval for human subjects research

Level 2: HUMAN-GROUNDED EVALUATION
  - Test with lay humans on simplified tasks
  - Example: Show Amazon Mechanical Turk workers two heatmaps and ask
    which one better highlights the object in the image
  - Cheaper than Level 1; still captures human perception
  - Risk: lay evaluators may prefer "pretty" explanations over faithful ones

Level 3: FUNCTIONALLY-GROUNDED EVALUATION
  - Use formal metrics without any humans in the loop
  - Examples:
    a) Faithfulness: Does masking high-attribution features degrade
       predictions more than masking low-attribution features?
    b) Stability: Do similar inputs get similar explanations?
    c) Sparsity: Are explanations concise?
    d) Consistency: Do different methods agree on the same input?
  - Cheapest and most reproducible
  - Risk: metric gaming — a method can score well on a metric without
    producing useful explanations for humans
"""
```

### 3.2 Implementing Functionally-Grounded Metrics

```python
import numpy as np
from typing import Callable, List, Tuple


def faithfulness_correlation(
    model_predict: Callable,
    input_data: np.ndarray,
    attributions: np.ndarray,
    num_perturbations: int = 100,
    seed: int = 42
) -> float:
    """
    Measure faithfulness of an attribution method via correlation.

    The idea: if an attribution method correctly identifies important features,
    then perturbing high-attribution features should change the prediction more
    than perturbing low-attribution features.

    We randomly mask subsets of features, measure the prediction change, and
    compute the Pearson correlation between the sum of attributions of masked
    features and the magnitude of prediction change.

    A high positive correlation indicates a faithful attribution method.

    Parameters
    ----------
    model_predict : callable
        Function that takes input array and returns prediction (scalar).
    input_data : np.ndarray
        Single input instance, shape (num_features,).
    attributions : np.ndarray
        Attribution values for each feature, shape (num_features,).
    num_perturbations : int
        Number of random feature subsets to test.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    float
        Pearson correlation between attribution sums and prediction changes.
        Range: [-1, 1]. Higher is more faithful.
    """
    rng = np.random.RandomState(seed)
    num_features = len(input_data)

    # Get the original prediction as our reference point
    original_pred = model_predict(input_data.reshape(1, -1))[0]

    attribution_sums = []
    prediction_changes = []

    for _ in range(num_perturbations):
        # Randomly select a subset of features to mask (set to zero)
        # We vary the subset size to get diverse perturbations
        subset_size = rng.randint(1, max(2, num_features // 2))
        mask_indices = rng.choice(num_features, size=subset_size, replace=False)

        # Create the perturbed input by zeroing out selected features
        perturbed = input_data.copy()
        perturbed[mask_indices] = 0.0

        # Measure how much the prediction changed
        perturbed_pred = model_predict(perturbed.reshape(1, -1))[0]
        pred_change = abs(original_pred - perturbed_pred)

        # Sum the attributions of the masked features
        # If the method is faithful, masking high-attribution features
        # should cause large prediction changes
        attr_sum = np.sum(np.abs(attributions[mask_indices]))

        attribution_sums.append(attr_sum)
        prediction_changes.append(pred_change)

    # Pearson correlation: how well do attribution sums predict
    # the magnitude of prediction changes?
    correlation = np.corrcoef(attribution_sums, prediction_changes)[0, 1]

    return correlation


def explanation_stability(
    explain_fn: Callable,
    input_data: np.ndarray,
    noise_scale: float = 0.01,
    num_neighbors: int = 20,
    seed: int = 42
) -> float:
    """
    Measure stability (Lipschitz continuity) of an explanation method.

    A stable explanation method should produce similar explanations for
    similar inputs. If adding tiny noise to the input dramatically changes
    the explanation, the method is unstable and potentially unreliable.

    We add small Gaussian noise to the input, compute explanations for
    each noisy version, and measure the maximum relative change.

    Parameters
    ----------
    explain_fn : callable
        Function that takes an input and returns attributions.
    input_data : np.ndarray
        Single input instance, shape (num_features,).
    noise_scale : float
        Standard deviation of Gaussian noise (relative to input norm).
    num_neighbors : int
        Number of noisy neighbors to test.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    float
        Maximum Lipschitz estimate. Lower means more stable.
    """
    rng = np.random.RandomState(seed)

    # Get the explanation for the original input
    original_explanation = explain_fn(input_data)
    original_norm = np.linalg.norm(original_explanation)

    if original_norm < 1e-10:
        # If the original explanation is essentially zero,
        # stability is not meaningful
        return 0.0

    max_lipschitz = 0.0

    for _ in range(num_neighbors):
        # Add small Gaussian noise to the input
        noise = rng.normal(0, noise_scale, size=input_data.shape)
        noisy_input = input_data + noise

        # Get the explanation for the noisy input
        noisy_explanation = explain_fn(noisy_input)

        # Compute the Lipschitz ratio:
        # ||explanation(x) - explanation(x')|| / ||x - x'||
        explanation_diff = np.linalg.norm(original_explanation - noisy_explanation)
        input_diff = np.linalg.norm(noise)

        if input_diff > 1e-10:
            lipschitz = explanation_diff / input_diff
            max_lipschitz = max(max_lipschitz, lipschitz)

    return max_lipschitz


def explanation_sparsity(attributions: np.ndarray, threshold: float = 0.01) -> float:
    """
    Measure sparsity of an explanation.

    Humans can typically process 5-9 pieces of information at once (Miller's
    Law). Sparser explanations are easier to understand. This metric computes
    the fraction of features with negligible attribution.

    Parameters
    ----------
    attributions : np.ndarray
        Attribution values for each feature.
    threshold : float
        Features with |attribution| / max(|attribution|) below this
        threshold are considered negligible.

    Returns
    -------
    float
        Fraction of features that are negligible. Range [0, 1].
        Higher means sparser (more concise) explanation.
    """
    abs_attr = np.abs(attributions)
    max_attr = np.max(abs_attr)

    if max_attr < 1e-10:
        # All attributions are zero — maximally sparse (or meaningless)
        return 1.0

    # Normalize attributions relative to the maximum
    normalized = abs_attr / max_attr

    # Count features below the threshold
    negligible = np.sum(normalized < threshold)

    return negligible / len(attributions)


# --- Demonstration ---

def demo_evaluation_metrics():
    """
    Demonstrate the evaluation metrics on a simple linear model.

    A linear model is the ideal test case because we KNOW the ground truth
    attributions: they are exactly the feature coefficients times the input
    values.
    """
    # Create a simple linear model where we know the true importances
    np.random.seed(42)
    true_weights = np.array([0.5, -0.3, 0.8, 0.0, 0.0, 0.1, 0.0, -0.6])

    def linear_model(X):
        return X @ true_weights

    # Generate a test input
    test_input = np.random.randn(8)

    # Ground-truth attribution: weight * input value
    true_attributions = true_weights * test_input

    # A good attribution method should match the ground truth
    good_attributions = true_attributions + np.random.normal(0, 0.01, size=8)

    # A bad attribution method assigns random importances
    bad_attributions = np.random.randn(8)

    # 1. Faithfulness
    faith_good = faithfulness_correlation(
        linear_model, test_input, good_attributions
    )
    faith_bad = faithfulness_correlation(
        linear_model, test_input, bad_attributions
    )
    print(f"Faithfulness (good method): {faith_good:.3f}")
    print(f"Faithfulness (bad method):  {faith_bad:.3f}")

    # 2. Sparsity
    # The true model has 5 non-zero weights out of 8 features
    sparsity = explanation_sparsity(true_attributions)
    print(f"Sparsity of true attributions: {sparsity:.3f}")

    print("\nInterpretation:")
    print("  Faithfulness close to 1.0 → method correctly identifies")
    print("    which features matter most")
    print("  High sparsity → explanation is concise and human-readable")


if __name__ == "__main__":
    demo_evaluation_metrics()
```

### 3.3 Choosing the Right Evaluation Level

```python
"""
Decision Guide: Which Evaluation Level to Use?

┌─────────────────────────────────────────────────────────┐
│                  START HERE                              │
│         What is the evaluation purpose?                  │
└───────────────┬─────────────────────┬───────────────────┘
                │                     │
        Deployment in a          Research /
        specific domain          Benchmarking
                │                     │
                ▼                     ▼
    ┌───────────────────┐   ┌─────────────────────┐
    │ Level 1            │   │ Can you recruit     │
    │ Application-       │   │ lay participants?   │
    │ Grounded          │   └──────┬──────────────┘
    │                    │          │         │
    │ Use domain experts │       Yes         No
    │ on real tasks      │          │         │
    └───────────────────┘          ▼         ▼
                            ┌──────────┐ ┌──────────┐
                            │ Level 2  │ │ Level 3  │
                            │ Human-   │ │ Function │
                            │ Grounded │ │ Grounded │
                            └──────────┘ └──────────┘

Practical advice:
  - Start with Level 3 (cheapest, fastest iteration)
  - Graduate to Level 2 for user-facing explanations
  - Use Level 1 only when deploying in high-stakes domains
  - Always report MULTIPLE metrics — no single metric captures
    explanation quality
"""
```

---

## 4. Explanation Output Types

Different interpretability methods produce fundamentally different *types* of
explanations. Understanding these types helps you choose the right method for
your stakeholder and domain.

### 4.1 Five Types of Explanation Outputs

```python
"""
Type 1: FEATURE ATTRIBUTION
  What: A numerical score for each input feature indicating its contribution
        to the prediction.
  Methods: SHAP, LIME, Integrated Gradients, gradient saliency
  Output: Vector of real numbers, one per feature
  Best for: Tabular data, debugging models, regulatory compliance
  Limitation: Does not explain HOW features interact

Type 2: RULES
  What: If-then-else logic that approximates the model's behavior.
  Methods: Decision tree extraction, Anchors (Ribeiro et al. 2018), LORE
  Output: A set of logical conditions
  Example: "IF income > 50k AND employment_years > 3 THEN approve loan"
  Best for: Non-technical stakeholders, policy compliance
  Limitation: May oversimplify complex decision boundaries

Type 3: EXAMPLES
  What: Training instances that are similar to the query or representative
        of the model's learned concepts.
  Methods: Prototypes, criticisms, k-nearest neighbors in embedding space,
           influence functions (Koh & Liang 2017)
  Output: A set of training examples
  Example: "This skin lesion was classified as melanoma because it is
            similar to these 5 confirmed melanoma cases in the training set"
  Best for: Image and medical domains where visual similarity is intuitive
  Limitation: Requires meaningful distance metrics; computationally expensive

Type 4: COUNTERFACTUALS
  What: Minimal changes to the input that would flip the prediction.
  Methods: Wachter et al. 2017, DiCE (Mothilal et al. 2020)
  Output: A modified input instance
  Example: "Your loan was denied. If your income were $5k higher, it would
            have been approved."
  Best for: Actionable recourse, GDPR "right to explanation"
  Limitation: May suggest infeasible changes (e.g., "change your age")

Type 5: CONCEPTS
  What: High-level human-understandable concepts that the model has learned.
  Methods: TCAV (Kim et al. 2018), Concept Bottleneck Models, ACE
  Output: Concept importance scores or concept activation vectors
  Example: "The model classified this bird as a 'cardinal' because it
            detected the concepts 'red feathers' and 'pointed crest'"
  Best for: Understanding what abstract features a model uses
  Limitation: Requires predefined concept datasets or discovery algorithms
"""
```

### 4.2 Comparing Explanation Types

```python
import matplotlib.pyplot as plt
import numpy as np


def visualize_explanation_types():
    """
    Create a comparison chart showing the trade-offs between
    different explanation output types across key dimensions.
    """
    explanation_types = [
        "Feature\nAttribution",
        "Rules",
        "Examples",
        "Counter-\nfactuals",
        "Concepts"
    ]

    # Scores (1-5) for each dimension
    # These are qualitative assessments based on the literature
    dimensions = {
        "Comprehensibility": [3, 5, 4, 5, 4],
        "Faithfulness":      [4, 2, 3, 3, 3],
        "Actionability":     [2, 3, 2, 5, 2],
        "Scalability":       [4, 3, 2, 3, 2],
        "Generality":        [5, 3, 4, 4, 2],
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(explanation_types))
    width = 0.15
    multiplier = 0

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']

    for i, (attribute, scores) in enumerate(dimensions.items()):
        offset = width * multiplier
        bars = ax.bar(x + offset, scores, width, label=attribute,
                      color=colors[i], alpha=0.85)
        multiplier += 1

    ax.set_xlabel('Explanation Type', fontsize=12)
    ax.set_ylabel('Score (1=Low, 5=High)', fontsize=12)
    ax.set_title('Comparison of Explanation Output Types', fontsize=14)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(explanation_types)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 6)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig("explanation_types_comparison.png", dpi=150)
    plt.show()

    print("\nKey observations:")
    print("  - Rules are most comprehensible but least faithful")
    print("  - Feature attribution is most general but least actionable")
    print("  - Counterfactuals are most actionable (recourse-oriented)")
    print("  - Concepts have limited generality (require concept datasets)")
    print("  - No single type dominates all dimensions")


if __name__ == "__main__":
    visualize_explanation_types()
```

### 4.3 Matching Explanation Types to Stakeholders

```python
"""
Stakeholder → Explanation Type Mapping

┌────────────────────┬──────────────────┬──────────────────────────┐
│ Stakeholder        │ Best Type(s)     │ Why                      │
├────────────────────┼──────────────────┼──────────────────────────┤
│ Data Scientist     │ Feature          │ Needs to debug models,   │
│                    │ Attribution      │ understand feature        │
│                    │                  │ interactions              │
├────────────────────┼──────────────────┼──────────────────────────┤
│ Business Manager   │ Rules,           │ Needs simple, actionable │
│                    │ Counterfactuals  │ summaries for decisions   │
├────────────────────┼──────────────────┼──────────────────────────┤
│ Affected           │ Counterfactuals  │ Needs to know what to    │
│ Individual         │                  │ change to get a          │
│ (loan applicant)   │                  │ different outcome        │
├────────────────────┼──────────────────┼──────────────────────────┤
│ Regulator          │ Rules,           │ Needs to verify legal    │
│                    │ Feature Attr.    │ compliance and identify  │
│                    │                  │ protected attribute use   │
├────────────────────┼──────────────────┼──────────────────────────┤
│ Clinician          │ Examples,        │ Reasons by analogy to    │
│                    │ Concepts         │ known cases and          │
│                    │                  │ medical concepts         │
├────────────────────┼──────────────────┼──────────────────────────┤
│ ML Researcher      │ All types,       │ Needs faithful           │
│                    │ esp. Feature     │ understanding of model   │
│                    │ Attribution +    │ internals for            │
│                    │ Concepts         │ improvement              │
└────────────────────┴──────────────────┴──────────────────────────┘
"""
```

---

## 5. The Faithfulness-Comprehensibility Tension

This is arguably the most important conceptual insight in interpretable AI:
there is a fundamental tension between explanations that are *faithful* to the
model's actual reasoning and explanations that are *comprehensible* to humans.

### 5.1 Defining the Tension

```python
"""
FAITHFULNESS
  An explanation is faithful if it accurately reflects the model's actual
  computation. The explanation describes what the model ACTUALLY does,
  not what we wish it did.

  Example of a faithful explanation:
    "The model's prediction depends on a complex interaction between
     features 3, 7, 12, 15, 23, and 42, mediated by three nonlinear
     transformations in the hidden layers."

  Problem: This is accurate but incomprehensible to most humans.


COMPREHENSIBILITY
  An explanation is comprehensible if a human can understand it and use
  it to reason about the model's behavior.

  Example of a comprehensible explanation:
    "The loan was denied because of low income."

  Problem: This may be a gross simplification. The actual model might
  consider 50 features in complex interactions. Reducing to one feature
  is comprehensible but potentially misleading.


THE TENSION
  As explanations become more faithful, they tend to become less
  comprehensible (they must capture the model's full complexity).
  As explanations become more comprehensible, they tend to become
  less faithful (they must simplify away real model behavior).

  This is NOT a strict mathematical trade-off — some methods navigate
  it better than others — but it is a persistent design challenge.
"""
```

### 5.2 Quantifying the Trade-off

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score


def demonstrate_faithfulness_comprehensibility():
    """
    Demonstrate the faithfulness-comprehensibility trade-off using
    surrogate models of varying complexity.

    A surrogate model approximates a black-box model's predictions.
    Simple surrogates (small decision trees) are comprehensible but
    may be unfaithful. Complex surrogates (large trees) are more
    faithful but less comprehensible.
    """
    # Create a moderately complex classification problem
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )

    # Train a complex black-box model (the one we want to explain)
    black_box = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        max_iter=500,
        random_state=42
    )
    black_box.fit(X, y)
    bb_predictions = black_box.predict(X)

    print("Black-box model accuracy: "
          f"{accuracy_score(y, bb_predictions):.3f}")
    print()

    # Now train surrogate models of increasing complexity
    # Faithfulness = how well the surrogate matches the black box
    # Comprehensibility = how simple the surrogate is (fewer nodes)
    max_depths = [2, 3, 5, 8, 12, 20, None]

    print(f"{'Depth':<10} {'Nodes':<10} {'Faithfulness':<15} "
          f"{'Comprehensible?':<15}")
    print("-" * 50)

    for depth in max_depths:
        # Train a decision tree to mimic the BLACK BOX (not the true labels)
        # This is the key idea: the surrogate learns to predict what the
        # black box predicts, not the ground truth
        surrogate = DecisionTreeClassifier(
            max_depth=depth,
            random_state=42
        )
        surrogate.fit(X, bb_predictions)  # Learn from black box predictions
        surrogate_predictions = surrogate.predict(X)

        # Faithfulness: agreement between surrogate and black box
        faithfulness = accuracy_score(bb_predictions, surrogate_predictions)

        # Comprehensibility proxy: number of nodes in the tree
        num_nodes = surrogate.tree_.node_count

        # Heuristic: trees with < 15 nodes are easily comprehensible
        comprehensible = "Yes" if num_nodes < 15 else (
            "Marginal" if num_nodes < 50 else "No"
        )

        depth_str = str(depth) if depth is not None else "None"
        print(f"{depth_str:<10} {num_nodes:<10} {faithfulness:<15.3f} "
              f"{comprehensible:<15}")

    print()
    print("Observation: As depth increases, faithfulness improves but")
    print("comprehensibility decreases. There is no free lunch.")
    print()
    print("Practical advice: Choose the simplest surrogate that achieves")
    print("'acceptable' faithfulness for your domain. What counts as")
    print("'acceptable' depends on the stakes of the decision.")


if __name__ == "__main__":
    demonstrate_faithfulness_comprehensibility()
```

### 5.3 Strategies for Navigating the Tension

```python
"""
Five strategies researchers use to navigate the
faithfulness-comprehensibility tension:

1. HIERARCHICAL EXPLANATIONS
   Provide explanations at multiple levels of detail.
   Top level: "Income was the most important factor" (comprehensible)
   Detail level: Full SHAP waterfall with feature interactions (faithful)
   Let the user drill down as needed.

2. INTERACTIVE EXPLANATIONS
   Let the user explore counterfactuals interactively.
   "What if my income were $10k higher?" → model re-predicts
   Each individual query is comprehensible; the full exploration
   can be arbitrarily faithful.

3. DOMAIN-SPECIFIC VOCABULARY
   Map model internals to domain concepts.
   Instead of: "feature 42 has attribution 0.23"
   Use: "the ST-segment elevation contributed positively to the
         diagnosis" (meaningful to a cardiologist)

4. HONEST UNCERTAINTY
   When the faithful explanation is too complex for a simple summary,
   say so: "The decision depends on a complex interaction between
   your credit history and employment sector. Here is a simplified
   summary, but the full picture is more nuanced."

5. SELECTIVE FAITHFULNESS
   Be faithful about the MOST IMPORTANT aspects and approximate
   the rest. SHAP does this implicitly: it reports exact Shapley
   values for the top features and groups the rest into "other."
"""
```

---

## 6. Research Landscape Map

### 6.1 Five Major Research Directions

```python
"""
The interpretable AI research landscape (as of 2024-2025):

┌─────────────────────────────────────────────────────────────────┐
│                   INTERPRETABLE AI METHODS                      │
├─────────────┬────────────┬───────────┬───────────┬─────────────┤
│ Attribution │ Concept-   │ Example-  │ Mechanis- │ Causal      │
│ Methods     │ Based      │ Based     │ tic       │ Explanations│
├─────────────┼────────────┼───────────┼───────────┼─────────────┤
│ SHAP        │ TCAV       │ Influence │ Circuits  │ CausalSHAP  │
│ LIME        │ Concept    │ Functions │ Sparse    │ DoWhy       │
│ Integrated  │ Bottleneck │ Prototypes│ Auto-     │ Counterfact.│
│ Gradients   │ Models     │ Criticisms│ encoders  │ Fairness    │
│ GradCAM     │ ACE        │ Representer│ Activation│ Structural  │
│ Attention   │ Network    │ Points    │ Patching  │ Causal      │
│ Rollout     │ Dissection │           │ Logit Lens│ Models      │
├─────────────┼────────────┼───────────┼───────────┼─────────────┤
│ Lessons:    │ Lesson 07  │ Lesson 08 │ Lesson 16 │ Lesson 09   │
│ 02,03,04,06 │            │           │           │             │
└─────────────┴────────────┴───────────┴───────────┴─────────────┘

Direction 1: ATTRIBUTION METHODS (most mature)
  - Assign numerical importance to each input feature
  - Well-developed theory (Shapley values) and tooling (SHAP, Captum)
  - Covered in ML L16 (SHAP, LIME basics) and this topic L02-L06
  - Key challenge: faithfulness guarantees are method-dependent

Direction 2: CONCEPT-BASED EXPLANATIONS (growing rapidly)
  - Explain in terms of high-level concepts, not raw features
  - TCAV, Concept Bottleneck Models, Automatic Concept Extraction
  - Covered in Lesson 07
  - Key challenge: defining and validating concept sets

Direction 3: EXAMPLE-BASED EXPLANATIONS (well-established)
  - Explain by pointing to similar or influential training examples
  - Influence functions, prototype networks, representer points
  - Covered in Lesson 08
  - Key challenge: computational cost of influence functions

Direction 4: MECHANISTIC INTERPRETABILITY (frontier research)
  - Reverse-engineer the actual algorithms learned by neural networks
  - Superposition hypothesis, sparse autoencoders, circuit discovery
  - Covered in Lesson 16
  - Led primarily by Anthropic, Google DeepMind, MATS researchers
  - Key challenge: scaling beyond small models

Direction 5: CAUSAL EXPLANATIONS (theoretically grounded)
  - Use causal reasoning to distinguish genuine effects from
    spurious correlations
  - Structural causal models, do-calculus, counterfactual fairness
  - Covered in Lesson 09
  - Key challenge: requires causal graph specification (often unknown)
"""
```

### 6.2 Method Selection Decision Tree

```python
"""
Practical Decision Tree for Choosing an Interpretability Method

Q1: What type of model are you explaining?
  ├── Tabular model (RF, XGBoost, etc.)
  │     → Start with SHAP (covered in ML L16)
  │     → Add counterfactuals if recourse is needed (L08)
  │     → Add Anchors if rules are needed for non-technical audience
  │
  ├── CNN (image classification)
  │     → Use GradCAM for localization (L03)
  │     → Use Integrated Gradients for pixel-level attribution (L02)
  │     → Use TCAV for concept-level understanding (L07)
  │
  ├── Transformer (NLP)
  │     → Start with attention visualization (L04)
  │     → Apply probing classifiers to understand representations (L05)
  │     → Use mechanistic interpretability for deep analysis (L16)
  │
  └── Any black box (API-only access)
        → LIME (model-agnostic, covered in ML L16)
        → Counterfactuals (only needs predict function, L08)

Q2: Who is the audience?
  ├── ML Engineer → feature attribution + stability metrics
  ├── Domain Expert → examples + concepts
  ├── Affected Individual → counterfactuals
  ├── Regulator → rules + feature attribution
  └── Researcher → all methods, emphasize faithfulness

Q3: What are the stakes?
  ├── Low stakes (recommendation) → any method, favor speed
  ├── Medium stakes (content moderation) → validated method + human review
  └── High stakes (medical, legal, financial) → multiple methods +
        human-grounded evaluation + regulatory compliance
"""
```

---

## 7. When Interpretability Is Required

### 7.1 Legal Requirements

```python
"""
LEGAL MANDATES FOR INTERPRETABILITY (2024-2025)

1. GDPR Article 22 (EU, 2018)
   - Applies to: Automated individual decision-making
   - Requires: "meaningful information about the logic involved"
   - Scope: Decisions with "legal effects" or "similarly significant effects"
   - Examples: Credit scoring, insurance pricing, hiring
   - Penalty: Up to 4% of global annual revenue

2. EU AI Act (2024, phased enforcement)
   - High-risk AI systems must be "sufficiently transparent to enable
     users to interpret the system's output and use it appropriately"
   - Requires technical documentation including:
     a) Description of the logic involved
     b) Key design choices and their rationale
     c) Known limitations and foreseeable risks
   - Prohibited practices (Article 5):
     a) Social scoring by governments
     b) Real-time biometric identification (with exceptions)
     c) Manipulation and exploitation

3. US: Equal Credit Opportunity Act (ECOA) + Regulation B
   - Creditors must provide "specific reasons" for adverse actions
   - Requires: Top features that drove the denial
   - SHAP/LIME outputs can satisfy this requirement (with care)

4. US: Fair Housing Act
   - Cannot use protected attributes (race, religion, sex) in
     housing decisions
   - Need interpretability to PROVE the model does not use proxies
     for protected attributes

5. Industry-Specific:
   - FDA (medical devices): Explanation of AI-assisted diagnoses
   - SEC (finance): Model risk management requirements
   - NIST AI RMF: Voluntary framework but increasingly referenced
"""
```

### 7.2 Ethical Requirements

```python
"""
ETHICAL MANDATES (beyond legal compliance)

Even when not legally required, interpretability is ethically necessary when:

1. IRREVERSIBLE CONSEQUENCES
   - Criminal sentencing, parole decisions (COMPAS controversy)
   - Medical treatment decisions
   - Autonomous vehicle safety-critical decisions

2. VULNERABILITY OF AFFECTED INDIVIDUALS
   - Children (recommendation algorithms, content moderation)
   - Marginalized communities (facial recognition accuracy disparities)
   - People without technical literacy to challenge AI decisions

3. POWER ASYMMETRY
   - Employer using AI to screen employees
   - Government using AI for benefit allocation
   - Insurance company using AI for pricing

4. SCIENTIFIC INTEGRITY
   - Drug discovery: need to verify the model's "reasoning" aligns
     with chemistry, not dataset artifacts
   - Climate modeling: need to verify learned patterns are physical

5. DEBUGGING AND SAFETY
   - Any production model benefits from interpretability during
     development, even if end-user explanations are not required
   - Detecting dataset leakage, spurious correlations, distribution
     shift, and adversarial vulnerability
"""
```

### 7.3 Organizational Decision Framework

```python
def interpretability_requirement_assessment(
    domain: str,
    decision_impact: str,
    audience: str,
    regulatory_jurisdiction: str,
    model_complexity: str
) -> dict:
    """
    Assess the level of interpretability required for a given AI system.

    This is a simplified decision framework. Real-world assessments
    should involve legal counsel and domain experts.

    Parameters
    ----------
    domain : str
        Application domain ("healthcare", "finance", "marketing", etc.)
    decision_impact : str
        Impact level ("low", "medium", "high", "critical")
    audience : str
        Primary explanation consumer ("developer", "domain_expert",
        "affected_individual", "regulator")
    regulatory_jurisdiction : str
        Applicable regulation ("EU", "US", "none")
    model_complexity : str
        Model type ("linear", "tree_ensemble", "deep_learning")

    Returns
    -------
    dict
        Assessment with recommended interpretability level and methods.
    """
    # Start with baseline requirement
    requirement_level = "low"
    recommended_methods = []
    legal_obligations = []

    # --- Domain escalation ---
    high_stakes_domains = {"healthcare", "finance", "criminal_justice",
                           "hiring", "insurance", "education"}
    if domain in high_stakes_domains:
        requirement_level = "high"

    # --- Impact escalation ---
    if decision_impact == "critical":
        requirement_level = "critical"
    elif decision_impact == "high" and requirement_level != "critical":
        requirement_level = "high"
    elif decision_impact == "medium" and requirement_level == "low":
        requirement_level = "medium"

    # --- Regulatory obligations ---
    if regulatory_jurisdiction == "EU":
        legal_obligations.append("GDPR Article 22: meaningful information "
                                 "about logic involved")
        legal_obligations.append("EU AI Act: transparency requirements "
                                 "for high-risk systems")
        if requirement_level in ("low", "medium"):
            requirement_level = "high"

    elif regulatory_jurisdiction == "US":
        if domain == "finance":
            legal_obligations.append("ECOA/Reg B: specific reasons for "
                                     "adverse actions")
        if domain in ("hiring", "housing"):
            legal_obligations.append("Anti-discrimination laws: prove no "
                                     "proxy discrimination")

    # --- Method recommendations ---
    if requirement_level == "low":
        recommended_methods = ["Feature importance (basic)"]
    elif requirement_level == "medium":
        recommended_methods = ["SHAP or LIME", "PDP/ICE plots"]
    elif requirement_level == "high":
        recommended_methods = [
            "SHAP (local + global)",
            "Counterfactual explanations",
            "Stability analysis",
            "Fairness audit"
        ]
    elif requirement_level == "critical":
        recommended_methods = [
            "SHAP (local + global)",
            "Counterfactual explanations",
            "Multiple attribution methods (cross-validation)",
            "Human-grounded evaluation (Level 2)",
            "Application-grounded evaluation (Level 1)",
            "Fairness audit with multiple definitions",
            "Model cards and datasheets",
            "Third-party audit"
        ]

    # --- Audience-specific additions ---
    if audience == "affected_individual":
        if "Counterfactual explanations" not in recommended_methods:
            recommended_methods.append("Counterfactual explanations")
    elif audience == "regulator":
        recommended_methods.append("Model cards / documentation")
        recommended_methods.append("Rule extraction for audit trail")

    return {
        "requirement_level": requirement_level,
        "recommended_methods": recommended_methods,
        "legal_obligations": legal_obligations,
        "note": (
            "This is a simplified assessment. Consult legal counsel "
            "for binding regulatory advice."
        )
    }


# --- Example assessments ---

if __name__ == "__main__":
    # Case 1: EU healthcare AI
    result = interpretability_requirement_assessment(
        domain="healthcare",
        decision_impact="critical",
        audience="domain_expert",
        regulatory_jurisdiction="EU",
        model_complexity="deep_learning"
    )
    print("=== EU Healthcare AI ===")
    print(f"  Level: {result['requirement_level']}")
    print(f"  Methods: {result['recommended_methods']}")
    print(f"  Legal: {result['legal_obligations']}")
    print()

    # Case 2: US marketing recommendation
    result = interpretability_requirement_assessment(
        domain="marketing",
        decision_impact="low",
        audience="developer",
        regulatory_jurisdiction="US",
        model_complexity="tree_ensemble"
    )
    print("=== US Marketing Recommendation ===")
    print(f"  Level: {result['requirement_level']}")
    print(f"  Methods: {result['recommended_methods']}")
    print(f"  Legal: {result['legal_obligations']}")
```

---

## 8. Interpretability vs. Explainability: Terminology Clarification

### 8.1 The Terminological Landscape

```python
"""
The literature uses many overlapping terms. Here is a practical glossary:

INTERPRETABILITY (Lipton, Rudin)
  The degree to which a human can understand the cause of a decision.
  Emphasizes the MODEL being understandable.
  Rudin argues we should build inherently interpretable models rather
  than explaining black boxes.

EXPLAINABILITY (XAI community, DARPA)
  The ability to explain predictions after the fact.
  Emphasizes producing EXPLANATIONS (artifacts) for any model.
  DARPA's XAI program (2017-2021) popularized this term.

TRANSPARENCY (EU AI Act)
  Legal term: the system's design and logic can be understood by
  appropriate stakeholders.
  Broader than interpretability — includes documentation, auditing,
  and governance.

UNDERSTANDABILITY
  User-centric: whether a specific human can understand a specific
  explanation in a specific context.
  Depends on the user's background, not just the explanation.

FAITHFULNESS
  Technical: whether an explanation accurately reflects the model.
  A post-hoc explanation can be unfaithful (plausible but wrong).

In this topic, we use:
  - "Interpretability" as the umbrella term (following the topic title)
  - "Explanation" for the output of any method
  - "Faithful" vs "comprehensible" for the key tension
  - Specific method names when precision is needed
"""
```

---

## 9. Historical Context and Key Papers

```python
"""
Timeline of Interpretable AI milestones:

1990s  - Rule extraction from neural networks (early attempts)
         Craven & Shavlik (1996): TREPAN extracts decision trees from NNs

2002   - Breiman's "Statistical Modeling: The Two Cultures"
         Argues for predictive models over interpretable ones
         (controversial; sparked decades of debate)

2014   - Attention mechanism (Bahdanau et al.)
         Initially presented AS an interpretation mechanism
         (later contested; see Lesson 04)

2016   - LIME (Ribeiro, Singh, Guestrin)
         Local Interpretable Model-agnostic Explanations
         → Covered in ML L16

2016   - CAM (Zhou et al.)
         Class Activation Mapping for CNNs
         → Covered in Lesson 03

2017   - SHAP (Lundberg & Lee)
         Unified framework via Shapley values
         → Covered in ML L16

2017   - Integrated Gradients (Sundararajan et al.)
         Axiomatic approach to gradient attribution
         → Covered in Lesson 02

2017   - GradCAM (Selvaraju et al.)
         Generalized CAM to any CNN architecture
         → Covered in Lesson 03

2018   - Lipton "The Mythos of Model Interpretability"
         Taxonomized what interpretability means
         → Covered in this lesson (Section 2)

2018   - Kim et al. TCAV
         Testing with Concept Activation Vectors
         → Covered in Lesson 07

2018   - GDPR enforcement begins
         Article 22: right to explanation for automated decisions

2018   - Adebayo et al. "Sanity Checks for Saliency Maps"
         Showed many gradient methods fail basic reliability tests
         → Covered in Lesson 02

2019   - "Attention is not Explanation" (Jain & Wallace)
         Challenged attention as explanation mechanism
         → Covered in Lesson 04

2019   - Rudin "Stop Explaining Black Box ML Models for High-Stakes
         Decisions and Use Interpretable Models Instead"
         Influential argument for intrinsic interpretability

2020   - DiCE (Mothilal et al.)
         Diverse Counterfactual Explanations
         → Covered in Lesson 08

2022   - Anthropic: Toy Models of Superposition
         Foundation for mechanistic interpretability
         → Covered in Lesson 16

2023   - Anthropic: Scaling Monosemanticity
         Sparse autoencoders on Claude
         → Covered in Lesson 16

2024   - EU AI Act enters into force (phased enforcement)
         Legal requirements for transparency and explainability
         → Covered in Lesson 13
"""
```

---

## Summary

- **Lipton's taxonomy** distinguishes three independent notions of interpretability:
  transparency (simulatability, decomposability, algorithmic transparency),
  post-hoc explanations, and trust. These are not interchangeable.

- **Doshi-Velez & Kim** provide three evaluation levels: application-grounded
  (real experts, real tasks), human-grounded (lay users, simplified tasks), and
  functionally-grounded (automated metrics, no humans). Start with Level 3 for
  rapid iteration; graduate to Levels 2 and 1 for deployment.

- **Five explanation output types** serve different needs: feature attribution
  (debugging, compliance), rules (non-technical audiences), examples (medical
  domains), counterfactuals (recourse), and concepts (high-level understanding).

- **The faithfulness-comprehensibility tension** is the central design challenge:
  more faithful explanations tend to be less comprehensible, and vice versa.
  Navigate this via hierarchical explanations, interactivity, domain vocabulary,
  honest uncertainty, and selective faithfulness.

- **The research landscape** spans five major directions: attribution methods
  (most mature), concept-based (growing), example-based (established),
  mechanistic (frontier), and causal (theoretically grounded).

- **Interpretability is legally required** in the EU (GDPR, AI Act), in US
  financial services (ECOA), and in many high-stakes domains. Even without legal
  mandates, ethical considerations make it necessary when consequences are
  irreversible, individuals are vulnerable, or power is asymmetric.

---

## Exercises

### Exercise 1: Taxonomy Application (Conceptual)

Consider a hospital deploying a deep learning model to predict patient
deterioration from vital signs. For each of Lipton's three transparency
sub-types (simulatability, decomposability, algorithmic transparency):
(a) Is the sub-type satisfied? Why or why not?
(b) Which sub-type matters most in this context?
(c) What post-hoc explanation method(s) would you recommend?

### Exercise 2: Faithfulness Measurement (Coding)

Using the `faithfulness_correlation` function from Section 3.2:
1. Train a Random Forest on the California Housing dataset.
2. Compute SHAP values for a test instance.
3. Compute the faithfulness correlation for the SHAP attributions.
4. Create random attributions and compute their faithfulness correlation.
5. Compare and interpret the results.

### Exercise 3: Explanation Type Selection (Design)

You are building a credit scoring system for a European bank. The system must
comply with GDPR Article 22 and will serve three audiences: (a) the credit
officer reviewing applications, (b) the applicant receiving the decision, and
(c) the banking regulator conducting an annual audit.

For each audience:
1. Which explanation output type(s) would you use?
2. What evaluation level (Doshi-Velez & Kim) is appropriate?
3. Write a mock explanation in the appropriate format.

### Exercise 4: Stability Analysis (Coding)

Using the `explanation_stability` function from Section 3.2:
1. Train both a Linear Regression and a Neural Network on a dataset of your choice.
2. Use SHAP to explain a single prediction from each model.
3. Measure the stability of the SHAP explanations for both models.
4. Which model produces more stable explanations? Why might this be?

### Exercise 5: Comprehensive Assessment (Integration)

Use the `interpretability_requirement_assessment` function from Section 7.3 to
evaluate the following three scenarios:
1. A US fintech startup using XGBoost for credit scoring
2. An EU hospital using a CNN for X-ray diagnosis
3. A social media company using Transformers for content moderation

For each scenario, extend the function output with: (a) specific Python
libraries you would use, (b) estimated implementation effort, and (c) one risk
if interpretability is neglected.

---

[Overview](./00_Overview.md) | [Next: Gradient Attribution](./02_Gradient_Attribution.md)

---

**License**: CC BY-NC 4.0
