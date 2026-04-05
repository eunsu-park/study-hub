"""
Exercises for Lesson 07: Concept-Based Explanations
Topic: Interpretable_AI

Solutions to practice problems from the lesson.
"""

import numpy as np


# === Exercise 1: Train a CAV from Activation Vectors ===
# Problem: Given activation vectors labeled as "concept present" or
# "concept absent", train a Concept Activation Vector (linear classifier)
# and analyze its direction in activation space.

def exercise_1():
    """Train a Concept Activation Vector (CAV) from activations."""
    print("=" * 60)
    print("Exercise 1: Train a CAV from Activation Vectors")
    print("=" * 60)

    np.random.seed(42)

    # Simulate activation vectors from a CNN's intermediate layer
    # Concept: "stripes" in an image classification model
    d_activation = 6
    n_pos = 30   # striped images
    n_neg = 30   # non-striped images

    # Positive examples: activations where "stripes" concept shifts dims 1,3
    pos_activations = np.random.randn(n_pos, d_activation) * 0.5
    pos_activations[:, 1] += 1.5   # stripes signal in dim 1
    pos_activations[:, 3] += 0.8   # stripes signal in dim 3

    # Negative examples: random activations (no stripe concept)
    neg_activations = np.random.randn(n_neg, d_activation) * 0.5

    # Labels: 1 = concept present, 0 = concept absent
    X = np.vstack([pos_activations, neg_activations])
    y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])

    # Shuffle
    idx = np.random.permutation(len(y))
    X, y = X[idx], y[idx]

    # Train linear classifier (SVM-like via logistic regression)
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    # Gradient descent
    w = np.zeros(d_activation)
    b = 0.0
    lr = 0.1
    for epoch in range(500):
        z = X @ w + b
        pred = sigmoid(z)
        dw = X.T @ (pred - y) / len(y) + 0.01 * w  # L2 regularization
        db = np.mean(pred - y)
        w -= lr * dw
        b -= lr * db

    # CAV = the weight vector (normal to the decision boundary)
    cav = w / (np.linalg.norm(w) + 1e-10)  # normalize to unit vector

    # Evaluate accuracy
    train_pred = (sigmoid(X @ w + b) > 0.5).astype(int)
    accuracy = np.mean(train_pred == y)

    print(f"\n  Concept: 'stripes'")
    print(f"  Activation dimension: {d_activation}")
    print(f"  Training samples: {n_pos} positive, {n_neg} negative")
    print(f"  Classifier accuracy: {accuracy:.2%}")

    print(f"\n  Learned CAV (unit normal vector):")
    for dim in range(d_activation):
        bar_len = int(abs(cav[dim]) * 30)
        sign = "+" if cav[dim] > 0 else "-"
        bar = "#" * bar_len
        print(f"    dim {dim}: {cav[dim]:+.4f}  {sign}{bar}")

    # Statistical significance via permutation test
    n_permutations = 100
    perm_accuracies = []
    for _ in range(n_permutations):
        y_perm = y[np.random.permutation(len(y))]
        w_perm = np.zeros(d_activation)
        b_perm = 0.0
        for _ in range(200):
            z = X @ w_perm + b_perm
            pred = sigmoid(z)
            dw = X.T @ (pred - y_perm) / len(y_perm) + 0.01 * w_perm
            db = np.mean(pred - y_perm)
            w_perm -= lr * dw
            b_perm -= lr * db
        perm_pred = (sigmoid(X @ w_perm + b_perm) > 0.5).astype(int)
        perm_accuracies.append(np.mean(perm_pred == y_perm))

    p_value = np.mean(np.array(perm_accuracies) >= accuracy)

    print(f"\n  Statistical significance (permutation test):")
    print(f"    Real accuracy: {accuracy:.2%}")
    print(f"    Mean permutation accuracy: {np.mean(perm_accuracies):.2%}")
    print(f"    p-value: {p_value:.4f}")
    print(f"    Significant (p < 0.05): {p_value < 0.05}")


# === Exercise 2: Compute TCAV Scores ===
# Problem: Given a trained CAV and a set of test activations, compute
# the TCAV score (fraction of inputs whose class prediction increases
# when moving in the CAV direction).

def exercise_2():
    """Compute TCAV scores for concept sensitivity testing."""
    print("\n" + "=" * 60)
    print("Exercise 2: Compute TCAV Scores")
    print("=" * 60)

    np.random.seed(42)

    d_activation = 6
    n_test = 100

    # Simulated CAV for "stripes" concept (from Exercise 1)
    cav_stripes = np.array([0.1, 0.7, 0.05, 0.5, -0.1, 0.02])
    cav_stripes = cav_stripes / np.linalg.norm(cav_stripes)

    # Simulated classifier: f(h) = W @ h + bias
    # Two classes: "zebra" (class 0) and "horse" (class 1)
    W_classifier = np.random.randn(2, d_activation) * 0.5
    # Make zebra class sensitive to stripes direction
    W_classifier[0] += 0.8 * cav_stripes  # zebra aligns with stripes
    bias = np.array([0.0, 0.0])

    def class_score(h, class_idx):
        return (W_classifier[class_idx] @ h + bias[class_idx])

    # Generate test activations (representing different images)
    test_activations = np.random.randn(n_test, d_activation) * 0.5

    # TCAV score: fraction of test inputs where the directional derivative
    # of class_score along CAV direction is positive
    # S_C,k,l = |{x: grad_h(f_k) . v_C > 0}| / |test set|

    classes = {"zebra": 0, "horse": 1}
    print(f"\n  Concept: 'stripes'")
    print(f"  CAV direction: {cav_stripes}")
    print(f"  Test samples: {n_test}")

    print(f"\n  {'Class':<12} {'TCAV Score':<14} {'Interpretation':<40}")
    print("  " + "-" * 66)

    for class_name, class_idx in classes.items():
        positive_count = 0
        directional_derivatives = []

        for i in range(n_test):
            h = test_activations[i]
            # Gradient of class score w.r.t. activation = W_classifier[class_idx]
            grad = W_classifier[class_idx]
            # Directional derivative along CAV
            dir_deriv = np.dot(grad, cav_stripes)
            directional_derivatives.append(dir_deriv)
            if dir_deriv > 0:
                positive_count += 1

        tcav_score = positive_count / n_test

        if tcav_score > 0.6:
            interp = "Concept POSITIVELY influences class"
        elif tcav_score < 0.4:
            interp = "Concept NEGATIVELY influences class"
        else:
            interp = "Concept has MINIMAL influence on class"

        print(f"  {class_name:<12} {tcav_score:<14.3f} {interp:<40}")

        if class_name == "zebra":
            mean_dd = np.mean(directional_derivatives)
            print(f"  {'':12} Mean dir. derivative: {mean_dd:.4f}")

    print(f"\n  Interpretation:")
    print(f"  TCAV score > 0.5 means the concept 'stripes' is positively")
    print(f"  associated with the class prediction. A score near 1.0 for 'zebra'")
    print(f"  confirms that the model uses stripe-like features for zebra detection.")
    print(f"  The statistical test compares against random CAVs to ensure")
    print(f"  the result is not due to chance.")


# === Exercise 3: Design Concept Sets for a Domain ===
# Problem: For a given model and domain, design appropriate concept sets
# and analyze potential pitfalls (concept overlap, completeness, etc.)

def exercise_3():
    """Design and analyze concept sets for a medical imaging domain."""
    print("\n" + "=" * 60)
    print("Exercise 3: Designing Concept Sets for a Domain")
    print("=" * 60)

    # Domain: Skin lesion classification (melanoma detection)
    # Classes: melanoma, benign nevus, seborrheic keratosis

    concept_sets = {
        "Asymmetry": {
            "positive_examples": "Images of asymmetric lesions",
            "negative_examples": "Images of symmetric lesions",
            "relevance": "ABCDE rule: Asymmetry is a key melanoma indicator",
            "n_positive": 50,
            "n_negative": 50,
        },
        "Irregular Border": {
            "positive_examples": "Lesions with jagged/uneven borders",
            "negative_examples": "Lesions with smooth, well-defined borders",
            "relevance": "ABCDE rule: Border irregularity suggests malignancy",
            "n_positive": 45,
            "n_negative": 55,
        },
        "Color Variation": {
            "positive_examples": "Lesions with multiple colors (brown, black, red)",
            "negative_examples": "Uniformly colored lesions",
            "relevance": "ABCDE rule: Color heterogeneity indicates melanoma",
            "n_positive": 40,
            "n_negative": 60,
        },
        "Large Diameter": {
            "positive_examples": "Lesions > 6mm diameter",
            "negative_examples": "Lesions < 6mm diameter",
            "relevance": "ABCDE rule: Larger lesions are more suspicious",
            "n_positive": 35,
            "n_negative": 65,
        },
        "Blue-White Structure": {
            "positive_examples": "Lesions with blue-white veil pattern",
            "negative_examples": "Lesions without blue-white patterns",
            "relevance": "Dermoscopic feature specific to melanoma",
            "n_positive": 25,
            "n_negative": 75,
        },
    }

    print(f"\n  Domain: Skin Lesion Classification (Melanoma Detection)")
    print(f"\n  Designed Concept Sets:")

    for concept, details in concept_sets.items():
        print(f"\n    Concept: '{concept}'")
        print(f"      Positive: {details['positive_examples']}")
        print(f"      Negative: {details['negative_examples']}")
        print(f"      Relevance: {details['relevance']}")
        print(f"      Samples: {details['n_positive']}+ / {details['n_negative']}-")

    # Analyze potential issues
    print(f"\n  Potential Pitfalls Analysis:")

    issues = [
        {
            "issue": "Concept Overlap",
            "description": (
                "'Asymmetry' and 'Irregular Border' are correlated -- "
                "asymmetric lesions often have irregular borders. This can "
                "make CAVs non-orthogonal and TCAV scores interdependent."
            ),
            "mitigation": (
                "Ensure concept example sets are diverse. Use concept "
                "decorrelation or report pairwise CAV cosine similarities."
            ),
        },
        {
            "issue": "Concept Purity",
            "description": (
                "Positive examples for 'Color Variation' may inadvertently "
                "include images that also have irregular borders, confounding "
                "the CAV with border features."
            ),
            "mitigation": (
                "Curate examples carefully. Use adversarial concept examples "
                "that vary only in the target concept."
            ),
        },
        {
            "issue": "Concept Completeness",
            "description": (
                "These 5 concepts may not fully explain the model's decisions. "
                "The model might use texture, vascular patterns, or other "
                "features not captured by our concept set."
            ),
            "mitigation": (
                "Compute concept completeness score. Add residual concept "
                "to capture unexplained variance."
            ),
        },
        {
            "issue": "Granularity Mismatch",
            "description": (
                "'Large Diameter' is a global property but CAVs operate on "
                "intermediate CNN activations which are local/spatial. The "
                "concept may not be well-captured at the probed layer."
            ),
            "mitigation": (
                "Probe multiple layers. Use global features from later "
                "layers for global concepts like size."
            ),
        },
    ]

    for item in issues:
        print(f"\n    [{item['issue']}]")
        print(f"      {item['description']}")
        print(f"      Mitigation: {item['mitigation']}")


# === Exercise 4: Analyze Concept Completeness ===
# Problem: Compute the concept completeness score — how well a set of
# concept activations can predict the model's output class.

def exercise_4():
    """Analyze concept completeness for a set of concepts."""
    print("\n" + "=" * 60)
    print("Exercise 4: Concept Completeness Analysis")
    print("=" * 60)

    np.random.seed(42)

    n_samples = 200
    n_concepts = 4
    n_classes = 3

    concept_names = ["Asymmetry", "Irregular_Border", "Color_Variation", "Texture"]
    class_names = ["melanoma", "benign_nevus", "seb_keratosis"]

    # Simulate concept scores (projection of activations onto CAVs)
    # Each sample has a concept score for each concept
    concept_scores = np.random.randn(n_samples, n_concepts)

    # True class labels (model predictions)
    # Classes depend on concepts plus some unexplained factors
    # Melanoma: high asymmetry + high color variation
    # Benign: low asymmetry + low color variation
    # Seb keratosis: high texture + moderate color

    logits = np.zeros((n_samples, n_classes))
    logits[:, 0] = 1.5 * concept_scores[:, 0] + 1.0 * concept_scores[:, 2]
    logits[:, 1] = -1.0 * concept_scores[:, 0] - 0.5 * concept_scores[:, 2]
    logits[:, 2] = 0.3 * concept_scores[:, 1] + 1.2 * concept_scores[:, 3]

    # Add unexplained variance (concepts are NOT complete)
    unexplained = np.random.randn(n_samples, n_classes) * 0.8
    logits_full = logits + unexplained

    # Model predictions (from full logits including unexplained)
    model_predictions = np.argmax(logits_full, axis=1)

    # Concept-only predictions (from concept-based logits only)
    concept_predictions = np.argmax(logits, axis=1)

    # Completeness score: how well concept scores predict model output
    # Method 1: Train a linear model from concept scores to predict class
    def softmax(z):
        z_shifted = z - z.max(axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    # Train linear concept-to-class mapping
    W_concept = np.zeros((n_classes, n_concepts))
    lr = 0.05
    for _ in range(500):
        logits_pred = concept_scores @ W_concept.T
        probs = softmax(logits_pred)
        # One-hot encode targets
        y_onehot = np.zeros((n_samples, n_classes))
        y_onehot[np.arange(n_samples), model_predictions] = 1.0
        # Gradient
        grad = concept_scores.T @ (probs - y_onehot) / n_samples
        W_concept -= lr * grad.T

    # Evaluate
    final_logits = concept_scores @ W_concept.T
    concept_based_pred = np.argmax(final_logits, axis=1)
    concept_accuracy = np.mean(concept_based_pred == model_predictions)

    # Random baseline (chance accuracy)
    random_accuracy = 1.0 / n_classes

    # Completeness score normalized: (concept_acc - random) / (1 - random)
    completeness = (concept_accuracy - random_accuracy) / (1 - random_accuracy)

    print(f"\n  Concepts: {concept_names}")
    print(f"  Classes: {class_names}")
    print(f"  Samples: {n_samples}")

    print(f"\n  Concept-to-class prediction accuracy: {concept_accuracy:.2%}")
    print(f"  Random baseline: {random_accuracy:.2%}")
    print(f"  Completeness score: {completeness:.3f}")

    if completeness > 0.8:
        verdict = "HIGH: concepts nearly fully explain model behavior"
    elif completeness > 0.5:
        verdict = "MODERATE: concepts explain significant model behavior"
    elif completeness > 0.2:
        verdict = "LOW: concepts only partially explain model behavior"
    else:
        verdict = "VERY LOW: concepts fail to capture model's decision process"
    print(f"  Verdict: {verdict}")

    # Per-concept importance (from learned weights)
    print(f"\n  Per-concept importance (weight magnitudes):")
    for c in range(n_concepts):
        importance = np.linalg.norm(W_concept[:, c])
        bar = "#" * int(importance * 10)
        print(f"    {concept_names[c]:<20} {importance:.4f}  {bar}")

    print(f"\n  Per-class concept reliance:")
    print(f"  {'Class':<20}", end="")
    for cn in concept_names:
        print(f"  {cn:<14}", end="")
    print()
    print("  " + "-" * (20 + 16 * n_concepts))
    for cls_idx, cls_name in enumerate(class_names):
        print(f"  {cls_name:<20}", end="")
        for c in range(n_concepts):
            print(f"  {W_concept[cls_idx, c]:<14.3f}", end="")
        print()

    print(f"\n  To improve completeness, consider:")
    print(f"  1. Adding more concepts (e.g., vascular patterns, shape regularity)")
    print(f"  2. Using nonlinear concept-to-class mappings")
    print(f"  3. Probing at a different network layer")
    print(f"  4. The gap (1 - completeness) represents model behavior")
    print(f"     not captured by any defined concept.")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
