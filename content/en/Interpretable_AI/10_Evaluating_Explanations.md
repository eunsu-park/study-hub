# Lesson 10: Evaluating Explanations

[Previous: Causal Inference for Interpretability](./09_Causal_Inference_for_Interpretability.md) | [Next: Advanced Algorithmic Fairness](./11_Advanced_Algorithmic_Fairness.md)

---

## Learning Objectives

- Define and compute faithfulness metrics (comprehensiveness, sufficiency, monotonicity deletion curves) to measure how accurately explanations reflect model behavior
- Evaluate explanation stability and robustness using Lipschitz continuity and sensitivity analysis
- Implement the ROAR benchmark (Hooker et al., 2019) to compare attribution methods on a level playing field
- Understand the Doshi-Velez and Kim taxonomy of human evaluation: application-grounded, human-grounded, and functionally-grounded
- Benchmark multiple explanation methods (SHAP, Integrated Gradients, LIME, GradCAM) on the same model and dataset

---

## 1. Why We Need to Evaluate Explanations

### 1.1 The Explanation Quality Problem

Explanation methods produce plausible-looking outputs, but how do we know they are
*correct*? A saliency map that highlights a dog's face looks reasonable for a "dog"
classification, but the model might actually rely on the grass in the background.
Without rigorous evaluation, we risk trusting explanations that misrepresent the
model's reasoning.

The core challenge is that for most models, we do not have a ground-truth explanation
to compare against. Unlike prediction accuracy (where we have labels), explanation
quality has no simple metric. This lesson covers the best available approaches for
filling that gap.

### 1.2 Desirable Properties of Explanations

Before we can evaluate, we need to define what a "good" explanation looks like:

```python
"""
A taxonomy of explanation quality properties.

These properties are often in tension with each other -- improving
one may degrade another. Evaluation requires measuring multiple
properties and making explicit tradeoffs.
"""

explanation_properties = {
    "Faithfulness (Fidelity)": {
        "definition": "The explanation accurately reflects the model's actual "
                      "reasoning process, not just plausible-sounding rationale.",
        "measurement": "Remove important features -> prediction should change. "
                       "Remove unimportant features -> prediction should not change.",
        "why_it_matters": "An unfaithful explanation is actively misleading -- "
                          "it tells you the model uses features it does not.",
    },
    "Stability (Robustness)": {
        "definition": "Similar inputs receive similar explanations. Small "
                      "perturbations to the input should not drastically "
                      "change the explanation.",
        "measurement": "Lipschitz continuity of the explanation function. "
                       "Correlation between explanations for nearby inputs.",
        "why_it_matters": "Unstable explanations cannot be trusted -- they "
                          "suggest the explanation captures noise, not signal.",
    },
    "Completeness": {
        "definition": "The explanation accounts for the full prediction, "
                      "not just a subset of contributing features.",
        "measurement": "Sum of attributions equals prediction minus baseline "
                       "(as in SHAP's efficiency axiom).",
        "why_it_matters": "Incomplete explanations may hide important factors.",
    },
    "Compactness (Sparsity)": {
        "definition": "The explanation is concise, highlighting only the most "
                      "relevant features rather than spreading attribution "
                      "across all features.",
        "measurement": "Number of features with non-negligible attribution. "
                       "Gini coefficient of attribution distribution.",
        "why_it_matters": "Humans can only process a limited amount of information. "
                          "Sparse explanations are more actionable.",
    },
    "Consistency": {
        "definition": "Different models that make the same predictions on the "
                      "same inputs receive the same explanations.",
        "measurement": "Compare explanations across models with similar accuracy.",
        "why_it_matters": "Explanations should describe model behavior, not be "
                          "artifacts of the explanation method.",
    },
}

for prop_name, details in explanation_properties.items():
    print(f"\n{prop_name}")
    print(f"  Definition:  {details['definition']}")
    print(f"  Measurement: {details['measurement']}")
    print(f"  Importance:  {details['why_it_matters']}")
```

---

## 2. Faithfulness (Fidelity) Metrics

### 2.1 Comprehensiveness

Comprehensiveness measures how much the prediction drops when we remove the
features that the explanation says are most important. A faithful explanation
should identify features whose removal causes a large prediction change.

```python
"""
Comprehensiveness metric for evaluating explanation faithfulness.

Comprehensiveness = f(x) - f(x \ top_k)

where x \ top_k means x with the top-k attributed features removed
(set to a baseline value like the mean or zero).

Higher comprehensiveness = more faithful explanation (removing important
features causes a bigger prediction drop).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import shap


def compute_comprehensiveness(
    model,
    X_instance: np.ndarray,
    attributions: np.ndarray,
    baseline: np.ndarray,
    k_values: list = None,
) -> dict:
    """Compute comprehensiveness at multiple top-k thresholds.

    For each k, we remove the top-k features (by attribution magnitude)
    and measure how much the prediction drops. If the explanation is
    faithful, removing high-attribution features should cause a large drop.

    Parameters
    ----------
    model : trained classifier with predict_proba
    X_instance : single instance to explain (1D array)
    attributions : feature attributions from an explanation method (1D array)
    baseline : baseline values to replace removed features with
    k_values : list of k values (number of features to remove)

    Returns
    -------
    dict mapping k -> comprehensiveness score
    """
    n_features = len(X_instance)
    if k_values is None:
        k_values = list(range(1, n_features + 1))

    # Original prediction probability for the predicted class
    original_pred = model.predict_proba(X_instance.reshape(1, -1))[0]
    predicted_class = np.argmax(original_pred)
    original_prob = original_pred[predicted_class]

    # Rank features by attribution magnitude (descending)
    ranked_features = np.argsort(-np.abs(attributions))

    results = {}
    for k in k_values:
        # Remove top-k features by replacing with baseline
        x_modified = X_instance.copy()
        features_to_remove = ranked_features[:k]
        x_modified[features_to_remove] = baseline[features_to_remove]

        # Prediction after removal
        modified_prob = model.predict_proba(
            x_modified.reshape(1, -1)
        )[0][predicted_class]

        # Comprehensiveness: how much did the prediction drop?
        comprehensiveness = original_prob - modified_prob
        results[k] = comprehensiveness

    return results


# --- Setup: train a model and generate explanations ---
np.random.seed(42)
X, y = make_classification(
    n_samples=2000, n_features=20, n_informative=8,
    n_redundant=4, n_classes=2, random_state=42,
)
feature_names = [f"feature_{i}" for i in range(20)]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42,
)

model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Compute SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test[:100])

# Evaluate comprehensiveness for the first test instance
x_instance = X_test[0]
instance_shap = shap_values[0]  # SHAP values for this instance
baseline = X_train.mean(axis=0)  # Mean training values as baseline

comp_results = compute_comprehensiveness(
    model, x_instance, instance_shap, baseline,
    k_values=[1, 3, 5, 10, 15, 20],
)

print("=== Comprehensiveness (SHAP) ===")
for k, score in comp_results.items():
    print(f"  Top-{k:2d} removed: comprehensiveness = {score:+.4f}")
```

### 2.2 Sufficiency

Sufficiency is the complement of comprehensiveness: instead of removing important
features, we keep *only* the important features and measure whether the prediction
is preserved.

```python
"""
Sufficiency metric: prediction using ONLY the top-k features.

Sufficiency = f(x_top_k) - f(baseline)

where x_top_k means keeping only the top-k attributed features and
replacing everything else with the baseline.

Higher sufficiency = more faithful explanation (the important features
alone are sufficient to reproduce the prediction).
"""


def compute_sufficiency(
    model,
    X_instance: np.ndarray,
    attributions: np.ndarray,
    baseline: np.ndarray,
    k_values: list = None,
) -> dict:
    """Compute sufficiency at multiple top-k thresholds.

    For each k, we keep ONLY the top-k features (by attribution magnitude)
    and set everything else to the baseline. If the explanation is faithful,
    keeping high-attribution features should be sufficient to maintain the
    prediction close to the original.

    Returns
    -------
    dict mapping k -> sufficiency score (prediction using only top-k minus
    prediction using baseline)
    """
    n_features = len(X_instance)
    if k_values is None:
        k_values = list(range(1, n_features + 1))

    original_pred = model.predict_proba(X_instance.reshape(1, -1))[0]
    predicted_class = np.argmax(original_pred)
    original_prob = original_pred[predicted_class]

    baseline_prob = model.predict_proba(
        baseline.reshape(1, -1)
    )[0][predicted_class]

    ranked_features = np.argsort(-np.abs(attributions))

    results = {}
    for k in k_values:
        # Keep only top-k features; everything else is baseline
        x_modified = baseline.copy()
        features_to_keep = ranked_features[:k]
        x_modified[features_to_keep] = X_instance[features_to_keep]

        modified_prob = model.predict_proba(
            x_modified.reshape(1, -1)
        )[0][predicted_class]

        # Sufficiency: how much of the prediction is recovered?
        sufficiency = modified_prob - baseline_prob
        results[k] = sufficiency

    return results


suff_results = compute_sufficiency(
    model, x_instance, instance_shap, baseline,
    k_values=[1, 3, 5, 10, 15, 20],
)

print("=== Sufficiency (SHAP) ===")
original_prob = model.predict_proba(x_instance.reshape(1, -1))[0].max()
for k, score in suff_results.items():
    recovery_pct = score / (original_prob - model.predict_proba(
        baseline.reshape(1, -1)
    )[0].max()) * 100
    print(f"  Top-{k:2d} kept: sufficiency = {score:+.4f} "
          f"({recovery_pct:.1f}% of prediction recovered)")
```

### 2.3 Monotonicity Deletion Curve

The monotonicity deletion curve extends comprehensiveness to a full curve,
removing features one at a time in order of decreasing attribution.

```python
"""
Monotonicity deletion curve: the gold standard for faithfulness evaluation.

Process:
1. Rank features by attribution (highest to lowest)
2. Remove one feature at a time (replace with baseline)
3. Record the prediction after each removal
4. The resulting curve should be monotonically decreasing if the
   explanation is perfectly faithful

A steeper initial drop indicates the explanation correctly identifies
the most important features first.

We also compute the Area Under the Deletion Curve (AUDC), which
summarizes the entire curve in a single number. Lower AUDC = better
explanation (prediction drops quickly when important features are removed).
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt


def deletion_curve(
    model,
    X_instance: np.ndarray,
    attributions: np.ndarray,
    baseline: np.ndarray,
    order: str = "most_important_first",
) -> dict:
    """Compute the full deletion curve.

    Parameters
    ----------
    order : "most_important_first" for deletion curve (MoRF),
            "least_important_first" for insertion curve (LeRF)
    """
    n_features = len(X_instance)
    predicted_class = np.argmax(
        model.predict_proba(X_instance.reshape(1, -1))[0]
    )

    # Sort features by attribution magnitude
    if order == "most_important_first":
        ranked = np.argsort(-np.abs(attributions))
    else:
        ranked = np.argsort(np.abs(attributions))

    # Compute predictions as features are progressively removed
    predictions = []
    x_current = X_instance.copy()

    # Start with all features present
    pred = model.predict_proba(x_current.reshape(1, -1))[0][predicted_class]
    predictions.append(pred)

    for i in range(n_features):
        x_current[ranked[i]] = baseline[ranked[i]]
        pred = model.predict_proba(x_current.reshape(1, -1))[0][predicted_class]
        predictions.append(pred)

    predictions = np.array(predictions)

    # Area Under the Deletion Curve (normalized to [0, 1])
    audc = np.trapz(predictions, dx=1.0 / n_features)

    # Check monotonicity: are predictions non-increasing?
    diffs = np.diff(predictions)
    monotonicity_violations = (diffs > 0.001).sum()  # Small threshold for noise

    return {
        "predictions": predictions,
        "audc": audc,
        "monotonicity_violations": monotonicity_violations,
        "is_monotonic": monotonicity_violations == 0,
    }


# Compute deletion curves for SHAP
del_curve = deletion_curve(model, x_instance, instance_shap, baseline)

print("=== Monotonicity Deletion Curve (SHAP) ===")
print(f"AUDC: {del_curve['audc']:.4f} (lower = better)")
print(f"Monotonicity violations: {del_curve['monotonicity_violations']}")
print(f"Monotonic: {del_curve['is_monotonic']}")
print(f"\nPrediction trajectory (first 10 steps):")
for i, pred in enumerate(del_curve["predictions"][:11]):
    bar = "#" * int(pred * 40)
    print(f"  Features removed: {i:2d}  P(class) = {pred:.4f}  {bar}")
```

---

## 3. Stability and Robustness

### 3.1 Lipschitz Continuity of Explanations

A stable explanation method should produce similar explanations for similar
inputs. Lipschitz continuity formalizes this: the change in explanation should
be bounded by a constant times the change in input.

```python
"""
Measuring explanation stability via local Lipschitz estimates.

For an explanation function g(x):
  ||g(x) - g(x')|| / ||x - x'|| <= L

where L is the Lipschitz constant. A large L means the explanation
is highly sensitive to small input changes -- unstable.

We estimate L empirically by computing the ratio for many nearby
input pairs and taking the maximum.
"""


def estimate_local_lipschitz(
    explanation_fn,
    x_instance: np.ndarray,
    n_neighbors: int = 50,
    epsilon: float = 0.01,
) -> dict:
    """Estimate the local Lipschitz constant of an explanation method.

    We perturb x slightly in random directions and measure how much
    the explanation changes relative to the input change.

    Parameters
    ----------
    explanation_fn : function that takes an instance and returns attributions
    x_instance : the instance to evaluate stability around
    n_neighbors : number of nearby points to sample
    epsilon : perturbation magnitude (fraction of feature std)

    Returns
    -------
    dict with Lipschitz estimates and stability statistics
    """
    base_explanation = explanation_fn(x_instance)
    ratios = []

    for _ in range(n_neighbors):
        # Random perturbation
        perturbation = np.random.normal(0, epsilon, size=x_instance.shape)
        x_perturbed = x_instance + perturbation

        # Explanation for perturbed input
        perturbed_explanation = explanation_fn(x_perturbed)

        # Compute ratio ||g(x) - g(x')|| / ||x - x'||
        explanation_change = np.linalg.norm(
            base_explanation - perturbed_explanation
        )
        input_change = np.linalg.norm(perturbation)

        if input_change > 1e-10:  # Avoid division by zero
            ratios.append(explanation_change / input_change)

    ratios = np.array(ratios)

    return {
        "max_lipschitz": ratios.max(),
        "mean_lipschitz": ratios.mean(),
        "median_lipschitz": np.median(ratios),
        "std_lipschitz": ratios.std(),
        "stability_score": 1.0 / (1.0 + ratios.mean()),  # 0 to 1, higher = more stable
    }


# Define explanation function wrappers
def shap_explain(x):
    """Wrapper: compute SHAP values for a single instance."""
    sv = explainer.shap_values(x.reshape(1, -1))
    return sv[0]


# Evaluate stability
stability = estimate_local_lipschitz(
    shap_explain, x_instance,
    n_neighbors=30, epsilon=0.05,
)

print("=== SHAP Stability (Local Lipschitz) ===")
print(f"Max Lipschitz constant:    {stability['max_lipschitz']:.4f}")
print(f"Mean Lipschitz constant:   {stability['mean_lipschitz']:.4f}")
print(f"Stability score (0-1):     {stability['stability_score']:.4f}")
print(f"  (Higher = more stable)")
```

### 3.2 Sensitivity to Hyperparameters

Many explanation methods have hyperparameters (e.g., number of samples in LIME,
number of steps in Integrated Gradients). A robust method should produce consistent
explanations across reasonable hyperparameter ranges.

```python
"""
Testing explanation sensitivity to method hyperparameters.

If explanations change dramatically with different hyperparameter settings,
users cannot trust the results -- the explanation depends more on the
method's configuration than on the model's behavior.
"""

from lime.lime_tabular import LimeTabularExplainer


def lime_sensitivity_test(
    model,
    X_train: np.ndarray,
    x_instance: np.ndarray,
    feature_names: list,
    n_samples_list: list = None,
) -> pd.DataFrame:
    """Test LIME's sensitivity to the num_samples hyperparameter.

    LIME fits a local linear model using random perturbations. The
    num_samples parameter controls how many perturbations are used.
    More samples generally give more stable results, but at higher
    computational cost.

    We run LIME multiple times with different num_samples values and
    measure how much the explanations vary.
    """
    if n_samples_list is None:
        n_samples_list = [100, 500, 1000, 2000, 5000]

    all_attributions = []

    for n_samples in n_samples_list:
        lime_explainer = LimeTabularExplainer(
            X_train,
            feature_names=feature_names,
            class_names=["0", "1"],
            mode="classification",
            random_state=42,
        )

        exp = lime_explainer.explain_instance(
            x_instance,
            model.predict_proba,
            num_features=len(feature_names),
            num_samples=n_samples,
        )

        # Extract attributions in feature order
        attr_dict = dict(exp.as_list())
        attributions = np.array([
            attr_dict.get(name, 0.0) for name in feature_names
        ])
        all_attributions.append(attributions)

    all_attributions = np.array(all_attributions)

    # Compute pairwise rank correlations
    from scipy.stats import spearmanr
    correlations = []
    for i in range(len(n_samples_list)):
        for j in range(i + 1, len(n_samples_list)):
            corr, _ = spearmanr(
                np.abs(all_attributions[i]),
                np.abs(all_attributions[j]),
            )
            correlations.append({
                "n_samples_1": n_samples_list[i],
                "n_samples_2": n_samples_list[j],
                "rank_correlation": corr,
            })

    return pd.DataFrame(correlations)


# Run sensitivity test
sensitivity_results = lime_sensitivity_test(
    model, X_train, x_instance, feature_names,
    n_samples_list=[100, 500, 1000, 5000],
)

print("=== LIME Sensitivity to num_samples ===")
print(sensitivity_results.to_string(index=False))
print(f"\nMean rank correlation: {sensitivity_results['rank_correlation'].mean():.4f}")
print("(1.0 = perfectly stable, <0.8 = concerning instability)")
```

### 3.3 Random Seed Sensitivity

```python
"""
Testing whether explanation results change across random seeds.

Non-deterministic explanation methods (LIME, KernelSHAP with sampling)
may produce different results each time they run. This measures
the magnitude of that variance.
"""


def seed_sensitivity_test(
    model,
    X_train: np.ndarray,
    x_instance: np.ndarray,
    feature_names: list,
    n_seeds: int = 10,
    n_samples: int = 1000,
) -> dict:
    """Run LIME with different random seeds and measure variance.

    Returns statistics about how much the explanation varies across seeds.
    """
    all_attributions = []

    for seed in range(n_seeds):
        lime_explainer = LimeTabularExplainer(
            X_train,
            feature_names=feature_names,
            class_names=["0", "1"],
            mode="classification",
            random_state=seed,
        )

        exp = lime_explainer.explain_instance(
            x_instance,
            model.predict_proba,
            num_features=len(feature_names),
            num_samples=n_samples,
        )

        attr_dict = dict(exp.as_list())
        attributions = np.array([
            attr_dict.get(name, 0.0) for name in feature_names
        ])
        all_attributions.append(attributions)

    all_attributions = np.array(all_attributions)

    # Per-feature coefficient of variation
    means = np.abs(all_attributions).mean(axis=0)
    stds = all_attributions.std(axis=0)
    cv = np.where(means > 1e-10, stds / means, 0)

    # Top-k feature agreement: do the same features appear as "top 5"?
    top_k = 5
    top_feature_sets = [
        set(np.argsort(-np.abs(attr))[:top_k])
        for attr in all_attributions
    ]

    # Jaccard similarity between all pairs of top-k sets
    jaccard_scores = []
    for i in range(len(top_feature_sets)):
        for j in range(i + 1, len(top_feature_sets)):
            intersection = len(top_feature_sets[i] & top_feature_sets[j])
            union = len(top_feature_sets[i] | top_feature_sets[j])
            jaccard_scores.append(intersection / union if union > 0 else 1.0)

    return {
        "mean_cv": cv.mean(),
        "max_cv": cv.max(),
        "top_k_jaccard": np.mean(jaccard_scores),
        "per_feature_cv": dict(zip(feature_names, cv)),
    }


seed_results = seed_sensitivity_test(
    model, X_train, x_instance, feature_names,
    n_seeds=10, n_samples=1000,
)

print("=== LIME Seed Sensitivity ===")
print(f"Mean coefficient of variation: {seed_results['mean_cv']:.4f}")
print(f"Top-5 feature agreement (Jaccard): {seed_results['top_k_jaccard']:.4f}")
print("(Jaccard = 1.0 means top features are identical across seeds)")
```

---

## 4. The ROAR Benchmark

### 4.1 RemOve And Retrain (ROAR)

ROAR (Hooker et al., 2019) addresses a critical flaw in standard deletion-based
evaluation: when you remove features (replace with zeros or means) and evaluate
the *same* model, the modified input is out-of-distribution. The model was never
trained on inputs with zeroed-out features, so its predictions may be unreliable.

ROAR solves this by *retraining* the model after feature removal. This ensures
the model is always evaluated on in-distribution data.

```python
"""
ROAR Benchmark (Hooker et al., 2019).

Protocol:
1. Train a model on the full dataset
2. Compute attributions for all training instances using method M
3. For each removal fraction k% (e.g., 10%, 20%, ..., 90%):
   a. For each training instance, replace the top k% attributed features
      with the per-feature mean (uninformative baseline)
   b. RETRAIN the model on this modified training set
   c. Evaluate the retrained model on a similarly modified test set
4. Plot accuracy vs. removal fraction

A better attribution method identifies truly important features, so
removing them causes a steeper accuracy drop even after retraining.
"""

from sklearn.metrics import accuracy_score
from sklearn.base import clone


def roar_benchmark(
    model_class,
    model_params: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    attribution_methods: dict,
    removal_fractions: list = None,
) -> pd.DataFrame:
    """Run the ROAR benchmark for multiple attribution methods.

    Parameters
    ----------
    model_class : sklearn model class (not instance)
    model_params : parameters for model initialization
    X_train, y_train : training data
    X_test, y_test : test data
    attribution_methods : dict of {method_name: attribution_fn}
        Each function takes (model, X) and returns attributions array
        with shape (n_samples, n_features)
    removal_fractions : list of fractions to remove (0.0 to 1.0)

    Returns
    -------
    DataFrame with columns: method, fraction_removed, accuracy
    """
    if removal_fractions is None:
        removal_fractions = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

    n_features = X_train.shape[1]
    feature_means = X_train.mean(axis=0)

    # Train baseline model (no removal)
    base_model = model_class(**model_params)
    base_model.fit(X_train, y_train)
    base_accuracy = accuracy_score(y_test, base_model.predict(X_test))

    results = []

    for method_name, attribution_fn in attribution_methods.items():
        print(f"\nComputing attributions for {method_name}...")

        # Compute attributions for training and test data
        train_attributions = attribution_fn(base_model, X_train)
        test_attributions = attribution_fn(base_model, X_test)

        for fraction in removal_fractions:
            k = int(fraction * n_features)

            if k == 0:
                # No removal -- use original data
                accuracy = base_accuracy
            else:
                # Remove top-k features for each instance
                X_train_modified = X_train.copy()
                X_test_modified = X_test.copy()

                for i in range(len(X_train)):
                    top_k_features = np.argsort(
                        -np.abs(train_attributions[i])
                    )[:k]
                    X_train_modified[i, top_k_features] = feature_means[top_k_features]

                for i in range(len(X_test)):
                    top_k_features = np.argsort(
                        -np.abs(test_attributions[i])
                    )[:k]
                    X_test_modified[i, top_k_features] = feature_means[top_k_features]

                # RETRAIN on modified data (the key difference from naive deletion)
                retrained_model = model_class(**model_params)
                retrained_model.fit(X_train_modified, y_train)
                accuracy = accuracy_score(
                    y_test, retrained_model.predict(X_test_modified)
                )

            results.append({
                "method": method_name,
                "fraction_removed": fraction,
                "accuracy": accuracy,
            })
            print(f"  {method_name} | {fraction:.0%} removed | "
                  f"accuracy = {accuracy:.4f}")

    # Add random baseline
    for fraction in removal_fractions:
        k = int(fraction * n_features)
        if k == 0:
            accuracy = base_accuracy
        else:
            X_train_rand = X_train.copy()
            X_test_rand = X_test.copy()
            for i in range(len(X_train)):
                random_features = np.random.choice(
                    n_features, k, replace=False
                )
                X_train_rand[i, random_features] = feature_means[random_features]
            for i in range(len(X_test)):
                random_features = np.random.choice(
                    n_features, k, replace=False
                )
                X_test_rand[i, random_features] = feature_means[random_features]

            retrained_model = model_class(**model_params)
            retrained_model.fit(X_train_rand, y_train)
            accuracy = accuracy_score(
                y_test, retrained_model.predict(X_test_rand)
            )

        results.append({
            "method": "Random",
            "fraction_removed": fraction,
            "accuracy": accuracy,
        })

    return pd.DataFrame(results)


# --- Define attribution methods ---
def shap_attributions(model, X):
    """Compute TreeSHAP attributions for all instances."""
    exp = shap.TreeExplainer(model)
    return exp.shap_values(X)


def random_attributions(model, X):
    """Random attributions (baseline for comparison)."""
    return np.random.randn(*X.shape)


def gradient_proxy_attributions(model, X):
    """Feature importance as attribution proxy (same for all instances).

    This is a simple baseline: use the model's global feature importance
    as the attribution for every instance. It ignores instance-specific
    information but captures overall feature relevance.
    """
    importances = model.feature_importances_
    return np.tile(importances, (X.shape[0], 1))


# --- Run ROAR benchmark ---
# (This is computationally expensive; reduce dataset for demonstration)
X_train_small = X_train[:500]
y_train_small = y_train[:500]
X_test_small = X_test[:200]
y_test_small = y_test[:200]

roar_results = roar_benchmark(
    model_class=GradientBoostingClassifier,
    model_params={"n_estimators": 50, "random_state": 42},
    X_train=X_train_small,
    y_train=y_train_small,
    X_test=X_test_small,
    y_test=y_test_small,
    attribution_methods={
        "TreeSHAP": shap_attributions,
        "GlobalImportance": gradient_proxy_attributions,
    },
    removal_fractions=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7],
)

print("\n=== ROAR Benchmark Results ===")
pivot = roar_results.pivot(
    index="fraction_removed", columns="method", values="accuracy"
)
print(pivot.round(4).to_string())
```

### 4.2 ROAR Limitations and Alternatives

```python
"""
Limitations of ROAR and alternative approaches.

ROAR has known issues:
1. Computational cost: retraining for every fraction x every method is expensive
2. Feature replacement: mean imputation may not be information-neutral
3. Model capacity: retrained model may compensate by using remaining features
   differently, masking the effect of removing important features

Alternatives:
- KAR (Keep And Retrain): opposite of ROAR -- keep top-k% and retrain
- FRESH (Faithful and Robust Evaluation of SHapley values): uses
  held-out ground truth from linear models
- Recursive ROAR: progressively remove features and re-explain
"""

# KAR (Keep And Retrain) -- the complement of ROAR
# Instead of removing important features, we KEEP only the important ones.
# A good explanation method's top-k features should be sufficient
# for a retrained model to maintain accuracy.

def kar_benchmark(
    model_class,
    model_params: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    attribution_fn,
    keep_fractions: list = None,
) -> pd.DataFrame:
    """KAR: Keep And Retrain benchmark.

    For each keep fraction k%:
    1. Keep only the top-k% features (replace rest with mean)
    2. Retrain the model
    3. Evaluate accuracy

    A better attribution method should maintain accuracy with fewer features.
    """
    if keep_fractions is None:
        keep_fractions = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

    n_features = X_train.shape[1]
    feature_means = X_train.mean(axis=0)

    base_model = model_class(**model_params)
    base_model.fit(X_train, y_train)
    attributions = attribution_fn(base_model, X_train)
    test_attributions = attribution_fn(base_model, X_test)

    results = []
    for fraction in keep_fractions:
        k = max(1, int(fraction * n_features))

        X_train_modified = np.tile(feature_means, (len(X_train), 1))
        X_test_modified = np.tile(feature_means, (len(X_test), 1))

        for i in range(len(X_train)):
            top_k = np.argsort(-np.abs(attributions[i]))[:k]
            X_train_modified[i, top_k] = X_train[i, top_k]

        for i in range(len(X_test)):
            top_k = np.argsort(-np.abs(test_attributions[i]))[:k]
            X_test_modified[i, top_k] = X_test[i, top_k]

        retrained = model_class(**model_params)
        retrained.fit(X_train_modified, y_train)
        acc = accuracy_score(y_test, retrained.predict(X_test_modified))

        results.append({
            "fraction_kept": fraction,
            "accuracy": acc,
            "n_features_kept": k,
        })

    return pd.DataFrame(results)
```

---

## 5. Human Evaluation Taxonomy

### 5.1 The Doshi-Velez and Kim Framework (2017)

Doshi-Velez and Kim proposed three levels of evaluation rigor for interpretability
methods, forming a hierarchy from most expensive/realistic to cheapest/abstract.

```python
"""
Three levels of human evaluation for interpretability methods.

Level 1: Application-Grounded
  - Real humans, real tasks
  - Example: radiologists use explanations to diagnose patients
  - Measures: task performance improvement, decision time, trust calibration
  - Cost: Very high (requires domain experts and real deployment)

Level 2: Human-Grounded
  - Real humans, simplified tasks
  - Example: crowdworkers compare two explanations and choose the "better" one
  - Measures: preference rates, perceived quality, understandability
  - Cost: Moderate (requires human subjects but not domain experts)

Level 3: Functionally-Grounded
  - No humans, proxy metrics
  - Example: comprehensiveness, sufficiency, ROAR (Sections 2-4 of this lesson)
  - Measures: automatic metrics computed from model behavior
  - Cost: Low (fully automated)
"""

evaluation_framework = {
    "Application-Grounded": {
        "participants": "Domain experts (doctors, judges, engineers)",
        "task": "Real-world decision-making task with explanations",
        "metrics": [
            "Task accuracy with vs. without explanations",
            "Decision speed improvement",
            "Trust calibration (do users trust model more when it is correct?)",
            "Appropriate reliance (do users override model when it is wrong?)",
        ],
        "example_study": (
            "Caruana et al.: doctors use GAM explanations for pneumonia "
            "risk prediction. Measured whether explanations helped doctors "
            "catch the model's spurious asthma-reduces-risk pattern."
        ),
        "strengths": "Most realistic, directly measures real-world impact",
        "weaknesses": "Expensive, time-consuming, hard to control confounds",
    },
    "Human-Grounded": {
        "participants": "Lay people or crowdworkers (no domain expertise needed)",
        "task": "Simplified tasks (binary comparison, Likert ratings)",
        "metrics": [
            "Forward simulation: given explanation, predict model output",
            "Counterfactual simulation: predict how output changes if input changes",
            "Preference: which of two explanations is 'better'?",
            "Surprise: rate how surprising the explanation is",
        ],
        "example_study": (
            "Ribeiro et al. (LIME paper): crowdworkers chose between two "
            "classifiers based on their LIME explanations. Workers identified "
            "the classifier using spurious features."
        ),
        "strengths": "Cheaper, captures human cognitive aspects",
        "weaknesses": "Simplified tasks may not reflect real-world usage",
    },
    "Functionally-Grounded": {
        "participants": "None (fully automated)",
        "task": "Compute proxy metrics from model behavior",
        "metrics": [
            "Faithfulness (comprehensiveness, sufficiency)",
            "Stability (Lipschitz continuity)",
            "ROAR benchmark",
            "Monotonicity of deletion curves",
            "Agreement between methods",
        ],
        "example_study": (
            "Hooker et al. (ROAR): compared 7 attribution methods by "
            "removing top-k% features and retraining. Found random "
            "baseline was competitive with several popular methods."
        ),
        "strengths": "Scalable, reproducible, no human subjects needed",
        "weaknesses": "Proxy metrics may not correlate with real-world utility",
    },
}

for level, details in evaluation_framework.items():
    print(f"\n{'='*60}")
    print(f"  {level}")
    print(f"{'='*60}")
    print(f"  Participants: {details['participants']}")
    print(f"  Task: {details['task']}")
    print(f"  Example: {details['example_study']}")
    print(f"  Strengths: {details['strengths']}")
    print(f"  Weaknesses: {details['weaknesses']}")
    print(f"  Metrics:")
    for metric in details['metrics']:
        print(f"    - {metric}")
```

### 5.2 Implementing Forward Simulation Tests

```python
"""
Forward simulation: a human-grounded evaluation task.

The test: given an explanation of a model's prediction, can the human
predict what the model would output for a NEW input?

If explanations are helpful, humans with explanations should predict
the model's behavior more accurately than humans without explanations.

We can automate a simplified version of this test using synthetic
"simulated humans" that follow simple rules.
"""


def automated_forward_simulation(
    model,
    X_test: np.ndarray,
    attributions: np.ndarray,
    feature_names: list,
    top_k: int = 3,
) -> dict:
    """Automated forward simulation test.

    Simulates a "human" who uses the top-k features from the explanation
    to predict the model's output. The simulated human uses a simple rule:
    if the sum of (top-k feature values * top-k attributions) is positive,
    predict class 1; otherwise predict class 0.

    This is a proxy for how well the explanation helps a human understand
    the model's decision process.
    """
    n_test = len(X_test)
    model_predictions = model.predict(X_test)

    # Simulated human predictions using only explanation information
    human_predictions = []
    for i in range(n_test):
        # Human looks at top-k features identified by the explanation
        top_features = np.argsort(-np.abs(attributions[i]))[:top_k]

        # Simple decision rule: weighted sum of important features
        weighted_sum = sum(
            X_test[i, j] * attributions[i, j] for j in top_features
        )

        human_pred = 1 if weighted_sum > 0 else 0
        human_predictions.append(human_pred)

    human_predictions = np.array(human_predictions)

    # How well does the simulated human predict the MODEL's output?
    agreement = (human_predictions == model_predictions).mean()

    # Random baseline
    random_agreement = max(
        (model_predictions == 1).mean(),
        (model_predictions == 0).mean(),
    )

    return {
        "human_model_agreement": agreement,
        "random_baseline": random_agreement,
        "improvement_over_random": agreement - random_agreement,
        "top_k_used": top_k,
    }


# Test with SHAP explanations
sim_result = automated_forward_simulation(
    model, X_test[:100], shap_values[:100], feature_names, top_k=5,
)

print("=== Forward Simulation Test ===")
print(f"Human-model agreement: {sim_result['human_model_agreement']:.3f}")
print(f"Random baseline:       {sim_result['random_baseline']:.3f}")
print(f"Improvement:           {sim_result['improvement_over_random']:+.3f}")
print(f"(Using top-{sim_result['top_k_used']} features from SHAP)")
```

---

## 6. Agreement Between Methods

### 6.1 When SHAP and Integrated Gradients Disagree

Different explanation methods often produce different explanations for the same
prediction. Understanding when and why they disagree is crucial for practitioners.

```python
"""
Comparing explanations from multiple methods on the same model and instance.

When methods agree, we have higher confidence in the explanation.
When they disagree, we need to understand why and determine which
method is more appropriate for our use case.
"""

from scipy.stats import spearmanr, kendalltau


def compare_attribution_methods(
    attributions_dict: dict,
    feature_names: list,
    top_k: int = 5,
) -> pd.DataFrame:
    """Compare multiple attribution methods using rank correlation.

    We use Spearman rank correlation on the absolute attribution values
    because the RANKING of feature importance is typically more meaningful
    than the exact values (which are on different scales).

    Parameters
    ----------
    attributions_dict : dict of {method_name: attributions_array}
    feature_names : list of feature names
    top_k : number of top features to compare

    Returns
    -------
    DataFrame with pairwise comparison metrics
    """
    methods = list(attributions_dict.keys())
    comparisons = []

    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i >= j:
                continue

            attr1 = np.abs(attributions_dict[method1])
            attr2 = np.abs(attributions_dict[method2])

            # Spearman rank correlation
            spearman_corr, spearman_p = spearmanr(attr1, attr2)

            # Kendall tau rank correlation
            kendall_corr, kendall_p = kendalltau(attr1, attr2)

            # Top-k agreement (Jaccard similarity)
            top_k_1 = set(np.argsort(-attr1)[:top_k])
            top_k_2 = set(np.argsort(-attr2)[:top_k])
            jaccard = len(top_k_1 & top_k_2) / len(top_k_1 | top_k_2)

            # Sign agreement: do the methods agree on the direction
            # (positive vs negative attribution) for all features?
            attr1_signed = attributions_dict[method1]
            attr2_signed = attributions_dict[method2]
            sign_agreement = (np.sign(attr1_signed) == np.sign(attr2_signed)).mean()

            comparisons.append({
                "Method 1": method1,
                "Method 2": method2,
                "Spearman rho": spearman_corr,
                "Kendall tau": kendall_corr,
                f"Top-{top_k} Jaccard": jaccard,
                "Sign Agreement": sign_agreement,
            })

    return pd.DataFrame(comparisons)


# --- Generate attributions from multiple methods ---

# Method 1: TreeSHAP
shap_attr = shap_values[0]  # First test instance

# Method 2: Permutation importance (instance-level approximation)
from sklearn.inspection import permutation_importance

def instance_permutation_importance(model, x_instance, X_bg, n_repeats=50):
    """Approximate instance-level permutation importance.

    For each feature, permute its value with random background values
    and measure how much the prediction changes.
    """
    original_pred = model.predict_proba(x_instance.reshape(1, -1))[0, 1]
    importances = np.zeros(len(x_instance))

    for feat_idx in range(len(x_instance)):
        pred_changes = []
        for _ in range(n_repeats):
            x_permuted = x_instance.copy()
            random_bg_idx = np.random.randint(len(X_bg))
            x_permuted[feat_idx] = X_bg[random_bg_idx, feat_idx]
            new_pred = model.predict_proba(x_permuted.reshape(1, -1))[0, 1]
            pred_changes.append(abs(original_pred - new_pred))

        importances[feat_idx] = np.mean(pred_changes)

    # Add sign: positive if feature value pushes prediction up
    for feat_idx in range(len(x_instance)):
        if x_instance[feat_idx] > X_bg[:, feat_idx].mean():
            importances[feat_idx] *= np.sign(shap_attr[feat_idx])
        else:
            importances[feat_idx] *= -np.sign(shap_attr[feat_idx])

    return importances


perm_attr = instance_permutation_importance(model, x_instance, X_train[:100])

# Method 3: Random baseline
random_attr = np.random.randn(len(x_instance))

# Compare all methods
comparison_df = compare_attribution_methods(
    {
        "TreeSHAP": shap_attr,
        "PermutationImportance": perm_attr,
        "Random": random_attr,
    },
    feature_names,
    top_k=5,
)

print("=== Method Agreement ===")
print(comparison_df.to_string(index=False))
```

---

## 7. Practical: Benchmarking Four Methods

### 7.1 Full Benchmark Pipeline

```python
"""
Complete benchmark comparing SHAP, Integrated Gradients (approximation),
LIME, and a simple baseline on the same model and dataset.

This is the kind of systematic comparison every practitioner should do
before choosing an explanation method for their use case.
"""


class ExplanationBenchmark:
    """Benchmark multiple explanation methods on the same model/dataset.

    Evaluates each method on:
    1. Faithfulness (comprehensiveness, sufficiency)
    2. Stability (Lipschitz estimate)
    3. Speed (wall-clock time per explanation)
    4. Inter-method agreement
    """

    def __init__(self, model, X_train, X_test, y_test, feature_names):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
        self.baseline = X_train.mean(axis=0)
        self.results = {}

    def evaluate_method(
        self,
        method_name: str,
        explanation_fn,
        n_instances: int = 50,
    ) -> dict:
        """Evaluate a single explanation method across multiple metrics.

        Parameters
        ----------
        method_name : identifier for this method
        explanation_fn : function(x_instance) -> attributions array
        n_instances : number of test instances to evaluate
        """
        import time

        metrics = {
            "comprehensiveness": [],
            "sufficiency": [],
            "deletion_audc": [],
            "monotonicity_violations": [],
            "times": [],
        }

        for i in range(min(n_instances, len(self.X_test))):
            x = self.X_test[i]

            # Time the explanation
            start = time.time()
            attr = explanation_fn(x)
            elapsed = time.time() - start
            metrics["times"].append(elapsed)

            # Comprehensiveness (top-5)
            comp = compute_comprehensiveness(
                self.model, x, attr, self.baseline, k_values=[5]
            )
            metrics["comprehensiveness"].append(comp[5])

            # Sufficiency (top-5)
            suff = compute_sufficiency(
                self.model, x, attr, self.baseline, k_values=[5]
            )
            metrics["sufficiency"].append(suff[5])

            # Deletion curve
            del_c = deletion_curve(self.model, x, attr, self.baseline)
            metrics["deletion_audc"].append(del_c["audc"])
            metrics["monotonicity_violations"].append(
                del_c["monotonicity_violations"]
            )

        result = {
            "method": method_name,
            "mean_comprehensiveness": np.mean(metrics["comprehensiveness"]),
            "mean_sufficiency": np.mean(metrics["sufficiency"]),
            "mean_deletion_audc": np.mean(metrics["deletion_audc"]),
            "mean_monotonicity_violations": np.mean(
                metrics["monotonicity_violations"]
            ),
            "mean_time_per_explanation": np.mean(metrics["times"]),
            "total_time": sum(metrics["times"]),
        }

        self.results[method_name] = result
        return result

    def summary(self) -> pd.DataFrame:
        """Generate a summary comparison table."""
        return pd.DataFrame(list(self.results.values())).set_index("method")


# --- Run the benchmark ---
benchmark = ExplanationBenchmark(
    model, X_train, X_test[:50], y_test[:50], feature_names
)

# Method 1: TreeSHAP
tree_explainer = shap.TreeExplainer(model)
benchmark.evaluate_method(
    "TreeSHAP",
    lambda x: tree_explainer.shap_values(x.reshape(1, -1))[0],
    n_instances=50,
)

# Method 2: LIME
lime_explainer = LimeTabularExplainer(
    X_train, feature_names=feature_names,
    class_names=["0", "1"], mode="classification", random_state=42,
)

def lime_explain(x):
    exp = lime_explainer.explain_instance(
        x, model.predict_proba, num_features=20, num_samples=500,
    )
    attr_dict = dict(exp.as_list())
    return np.array([attr_dict.get(name, 0.0) for name in feature_names])

benchmark.evaluate_method("LIME", lime_explain, n_instances=50)

# Method 3: Global feature importance (baseline)
global_importance = model.feature_importances_
benchmark.evaluate_method(
    "GlobalImportance",
    lambda x: global_importance,
    n_instances=50,
)

# Method 4: Random attributions (sanity check baseline)
benchmark.evaluate_method(
    "Random",
    lambda x: np.random.randn(len(x)),
    n_instances=50,
)

# --- Display results ---
print("=" * 70)
print("  EXPLANATION METHOD BENCHMARK RESULTS")
print("=" * 70)
summary = benchmark.summary()
print(summary.round(4).to_string())

print("\n--- Interpretation Guide ---")
print("Comprehensiveness: Higher = better (removing top features causes bigger drop)")
print("Sufficiency:       Higher = better (top features alone recover prediction)")
print("Deletion AUDC:     Lower  = better (prediction drops quickly)")
print("Monotonicity:      Lower  = better (fewer violations = more faithful ranking)")
print("Time:              Lower  = better (faster is more practical)")
```

### 7.2 Visualization and Reporting

```python
"""
Generate a visual comparison report for the benchmark results.
"""


def create_benchmark_report(summary_df: pd.DataFrame) -> str:
    """Generate a text-based benchmark report.

    In practice, you would create plots (bar charts, radar charts),
    but this function generates a structured text report suitable
    for documentation or stakeholder communication.
    """
    report = []
    report.append("=" * 60)
    report.append("  EXPLANATION METHOD EVALUATION REPORT")
    report.append("=" * 60)

    # Rank methods on each metric
    metrics_higher_better = ["mean_comprehensiveness", "mean_sufficiency"]
    metrics_lower_better = [
        "mean_deletion_audc",
        "mean_monotonicity_violations",
        "mean_time_per_explanation",
    ]

    rankings = {}
    for metric in metrics_higher_better:
        if metric in summary_df.columns:
            ranked = summary_df[metric].rank(ascending=False)
            rankings[metric] = ranked

    for metric in metrics_lower_better:
        if metric in summary_df.columns:
            ranked = summary_df[metric].rank(ascending=True)
            rankings[metric] = ranked

    rankings_df = pd.DataFrame(rankings)
    rankings_df["avg_rank"] = rankings_df.mean(axis=1)

    report.append("\nOverall Rankings (1=best):")
    for method in rankings_df.sort_values("avg_rank").index:
        avg_rank = rankings_df.loc[method, "avg_rank"]
        report.append(f"  {method:20s} Average Rank: {avg_rank:.2f}")

    report.append("\nRecommendations:")
    best_method = rankings_df["avg_rank"].idxmin()
    report.append(f"  Best overall: {best_method}")

    fastest = summary_df["mean_time_per_explanation"].idxmin()
    report.append(f"  Fastest:      {fastest}")

    most_faithful = summary_df["mean_comprehensiveness"].idxmax()
    report.append(f"  Most faithful: {most_faithful}")

    return "\n".join(report)


report = create_benchmark_report(summary)
print(report)
```

---

## Summary

- **Faithfulness** is the most important property of an explanation: does it accurately
  reflect what the model actually computes? Comprehensiveness (prediction drops when
  important features are removed) and sufficiency (important features alone recover
  the prediction) are the primary metrics.
- **Monotonicity deletion curves** provide a fine-grained view of faithfulness by
  removing features one at a time in attribution order. The Area Under the Deletion
  Curve (AUDC) summarizes this in a single number.
- **Stability** ensures explanations are not sensitive to small input perturbations
  or random seeds. The local Lipschitz constant quantifies this.
- **ROAR** (Hooker et al., 2019) is the gold standard for comparing attribution
  methods because it retrains the model after feature removal, avoiding
  out-of-distribution evaluation artifacts.
- **Human evaluation** ranges from expensive application-grounded studies (real
  experts, real tasks) to cheap functionally-grounded metrics (automated, no
  humans). The Doshi-Velez and Kim taxonomy organizes these into three levels.
- **Method agreement** (Spearman rank correlation, top-k Jaccard) reveals when
  different methods give consistent vs. contradictory explanations. Disagreement
  signals that at least one method is unfaithful.
- **Always benchmark** multiple methods on your specific model and dataset before
  choosing one. No single method dominates across all metrics and use cases.

---

## Exercises

### Exercise 1: Deletion Curves for Neural Networks (Beginner)

Train a simple neural network (2-layer MLP) on a tabular classification dataset.
Compute deletion curves using:
1. Random attributions (baseline)
2. Gradient-based attributions (input gradient * input)
3. SHAP (KernelSHAP or DeepSHAP)

Plot all three deletion curves on the same graph. Which method produces the
steepest initial drop?

### Exercise 2: ROAR at Scale (Intermediate)

Implement the full ROAR benchmark on CIFAR-10 using a pretrained CNN:
1. Compute GradCAM and Integrated Gradients attributions for 1000 training images
2. Run ROAR with removal fractions [10%, 30%, 50%, 70%, 90%]
3. Include random attribution as a baseline
4. Plot accuracy vs. removal fraction for all methods

Hint: for images, "removing" a pixel means replacing it with the dataset mean pixel value.

### Exercise 3: Human-Grounded Evaluation Design (Intermediate)

Design a human-grounded evaluation study for LIME explanations of a sentiment
analysis model:
1. Define the forward simulation task (what question do you ask participants?)
2. Define the counterfactual simulation task
3. Specify how you would compute inter-annotator agreement
4. Describe your control condition (what do participants without explanations see?)
5. Estimate the required sample size for statistical significance

Write the study protocol as a structured document (no need to run the study).

### Exercise 4: Multi-Method Benchmark Report (Advanced)

Extend the ExplanationBenchmark class to include:
1. At least 5 explanation methods (add Integrated Gradients, GradCAM for CNNs)
2. Stability measurement (Lipschitz estimates) for each method
3. Seed sensitivity measurement
4. A "radar chart" visualization comparing methods across all dimensions
5. A written recommendation for which method to use in (a) a healthcare setting
   requiring high faithfulness, and (b) a real-time serving setting requiring speed

---

[Previous: Causal Inference for Interpretability](./09_Causal_Inference_for_Interpretability.md) | [Overview](./00_Overview.md) | [Next: Advanced Algorithmic Fairness](./11_Advanced_Algorithmic_Fairness.md)

**License**: CC BY-NC 4.0
