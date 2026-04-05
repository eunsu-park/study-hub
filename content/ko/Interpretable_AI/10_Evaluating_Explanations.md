# 레슨 10: 설명 평가(Evaluating Explanations)

[이전: 해석 가능성을 위한 인과 추론](./09_Causal_Inference_for_Interpretability.md) | [다음: 고급 알고리즘 공정성](./11_Advanced_Algorithmic_Fairness.md)

---

## 학습 목표

- 설명이 모델 동작을 얼마나 정확히 반영하는지 측정하기 위해 충실도 지표(faithfulness metrics)(포괄성, 충분성, 단조성 삭제 곡선)를 정의하고 계산한다
- 리프시츠 연속성(Lipschitz continuity)과 민감도 분석을 사용하여 설명의 안정성과 견고성을 평가한다
- ROAR 벤치마크(Hooker et al., 2019)를 구현하여 기여도 방법들을 공정한 조건에서 비교한다
- Doshi-Velez와 Kim의 인간 평가 분류법을 이해한다: 응용 기반(application-grounded), 인간 기반(human-grounded), 기능 기반(functionally-grounded)
- 동일한 모델과 데이터셋에서 여러 설명 방법(SHAP, Integrated Gradients, LIME, GradCAM)을 벤치마킹한다

---

## 1. 설명을 평가해야 하는 이유

### 1.1 설명 품질 문제

설명 방법은 그럴듯해 보이는 출력을 생성하지만, 그것이 *정확*한지 어떻게 알 수 있는가? 개의 얼굴을 강조하는 현저성 맵(saliency map)은 "개" 분류에 합리적으로 보이지만, 모델은 실제로 배경의 잔디에 의존할 수 있다. 엄격한 평가 없이는 모델의 추론을 잘못 표현하는 설명을 신뢰하는 위험이 있다.

핵심 과제는 대부분의 모델에 대해 비교할 수 있는 정답 설명(ground-truth explanation)이 없다는 것이다. 예측 정확도(레이블이 있는)와 달리, 설명 품질에는 간단한 지표가 없다. 이 레슨에서는 그 격차를 메우기 위한 최선의 접근법을 다룬다.

### 1.2 바람직한 설명의 속성

평가하기 전에, "좋은" 설명이 어떤 모습인지 정의해야 한다:

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

## 2. 충실도(Faithfulness, Fidelity) 지표

### 2.1 포괄성(Comprehensiveness)

포괄성은 설명이 가장 중요하다고 말하는 특성을 제거했을 때 예측이 얼마나 떨어지는지를 측정한다. 충실한 설명은 제거 시 큰 예측 변화를 일으키는 특성을 식별해야 한다.

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

### 2.2 충분성(Sufficiency)

충분성은 포괄성의 보완이다: 중요한 특성을 제거하는 대신, 중요한 특성*만* 유지하고 예측이 보존되는지 측정한다.

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

### 2.3 단조성 삭제 곡선(Monotonicity Deletion Curve)

단조성 삭제 곡선은 포괄성을 완전한 곡선으로 확장하여, 기여도가 높은 순서대로 특성을 하나씩 제거한다.

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

## 3. 안정성과 견고성(Stability and Robustness)

### 3.1 설명의 리프시츠 연속성(Lipschitz Continuity of Explanations)

안정적인 설명 방법은 유사한 입력에 대해 유사한 설명을 생성해야 한다. 리프시츠 연속성(Lipschitz continuity)은 이를 형식화한다: 설명의 변화는 입력의 변화에 상수를 곱한 것으로 상한이 제한되어야 한다.

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

### 3.2 하이퍼파라미터에 대한 민감도(Sensitivity to Hyperparameters)

많은 설명 방법에는 하이퍼파라미터(예: LIME의 샘플 수, Integrated Gradients의 스텝 수)가 있다. 견고한 방법은 합리적인 하이퍼파라미터 범위에서 일관된 설명을 생성해야 한다.

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

### 3.3 랜덤 시드 민감도(Random Seed Sensitivity)

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

## 4. ROAR 벤치마크

### 4.1 제거 후 재훈련(RemOve And Retrain, ROAR)

ROAR (Hooker et al., 2019)는 표준 삭제 기반 평가의 중요한 결함을 해결한다: 특성을 제거(0이나 평균으로 대체)하고 *동일한* 모델로 평가하면, 수정된 입력은 분포 밖(out-of-distribution)이다. 모델은 특성이 0으로 된 입력으로 훈련된 적이 없으므로, 예측이 신뢰할 수 없을 수 있다.

ROAR은 특성 제거 후 모델을 *재훈련*하여 이를 해결한다. 이로써 모델이 항상 분포 내 데이터로 평가됨을 보장한다.

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

### 4.2 ROAR의 한계와 대안(ROAR Limitations and Alternatives)

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

## 5. 인간 평가 분류법(Human Evaluation Taxonomy)

### 5.1 Doshi-Velez와 Kim 프레임워크 (2017)

Doshi-Velez와 Kim은 해석 가능성 방법에 대한 세 가지 수준의 평가 엄격성을 제안했으며, 가장 비용이 높고 현실적인 것부터 가장 저렴하고 추상적인 것까지의 계층 구조를 형성한다.

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

### 5.2 순방향 시뮬레이션 테스트 구현(Implementing Forward Simulation Tests)

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

## 6. 방법 간 일치도(Agreement Between Methods)

### 6.1 SHAP과 Integrated Gradients가 불일치할 때

서로 다른 설명 방법은 동일한 예측에 대해 종종 다른 설명을 생성한다. 언제 그리고 왜 불일치하는지 이해하는 것은 실무자에게 중요하다.

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

## 7. 실습: 네 가지 방법 벤치마킹(Practical: Benchmarking Four Methods)

### 7.1 전체 벤치마크 파이프라인

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

### 7.2 시각화 및 보고서(Visualization and Reporting)

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

## 요약

- **충실도(Faithfulness)**는 설명의 가장 중요한 속성이다: 모델이 실제로 계산하는 것을 정확히 반영하는가? 포괄성(중요한 특성을 제거하면 예측이 떨어지는 것)과 충분성(중요한 특성만으로 예측을 복원하는 것)이 주요 지표이다.
- **단조성 삭제 곡선(Monotonicity deletion curves)**은 기여도 순서대로 특성을 하나씩 제거하여 충실도의 세밀한 뷰를 제공한다. 삭제 곡선 아래 면적(AUDC)은 이를 단일 수치로 요약한다.
- **안정성(Stability)**은 설명이 작은 입력 섭동이나 랜덤 시드에 민감하지 않음을 보장한다. 국소 리프시츠 상수(local Lipschitz constant)가 이를 정량화한다.
- **ROAR** (Hooker et al., 2019)는 특성 제거 후 모델을 재훈련하여 분포 밖 평가 인공물을 피하기 때문에, 기여도 방법을 비교하기 위한 표준 기법이다.
- **인간 평가(Human evaluation)**는 비용이 높은 응용 기반 연구(실제 전문가, 실제 작업)부터 저렴한 기능 기반 지표(자동화, 인간 불필요)까지 다양하다. Doshi-Velez와 Kim의 분류법은 이를 세 가지 수준으로 조직한다.
- **방법 일치도(Method agreement)** (스피어만 순위 상관관계, top-k 자카드)는 서로 다른 방법이 일관된 설명을 제공하는지 아니면 모순되는 설명을 제공하는지를 드러낸다. 불일치는 적어도 하나의 방법이 충실하지 않다는 신호이다.
- **항상 벤치마킹하라**: 하나를 선택하기 전에 특정 모델과 데이터셋에서 여러 방법을 벤치마킹하라. 모든 지표와 사용 사례에서 단일 방법이 지배적이지는 않다.

---

## 연습 문제

### 연습 1: 신경망을 위한 삭제 곡선 (초급)

간단한 신경망(2층 MLP)을 테이블 형태 분류 데이터셋에서 훈련시키라. 다음을 사용하여 삭제 곡선을 계산하라:
1. 랜덤 기여도 (기준선)
2. 그래디언트 기반 기여도 (입력 그래디언트 * 입력)
3. SHAP (KernelSHAP 또는 DeepSHAP)

세 가지 삭제 곡선을 동일한 그래프에 그리라. 어떤 방법이 가장 가파른 초기 하락을 보이는가?

### 연습 2: 대규모 ROAR (중급)

사전 훈련된 CNN에서 CIFAR-10에 대한 전체 ROAR 벤치마크를 구현하라:
1. 1000개 훈련 이미지에 대해 GradCAM과 Integrated Gradients 기여도를 계산하라
2. 제거 비율 [10%, 30%, 50%, 70%, 90%]로 ROAR을 실행하라
3. 기준선으로 랜덤 기여도를 포함하라
4. 모든 방법에 대해 정확도 대 제거 비율 그래프를 그리라

힌트: 이미지의 경우, 픽셀을 "제거"한다는 것은 데이터셋 평균 픽셀 값으로 대체하는 것을 의미한다.

### 연습 3: 인간 기반 평가 설계 (중급)

감성 분석 모델의 LIME 설명을 위한 인간 기반 평가 연구를 설계하라:
1. 순방향 시뮬레이션 과제를 정의하라 (참가자에게 어떤 질문을 하는가?)
2. 반사실적 시뮬레이션 과제를 정의하라
3. 주석자 간 일치도를 어떻게 계산할지 명시하라
4. 통제 조건을 설명하라 (설명이 없는 참가자들은 무엇을 보는가?)
5. 통계적 유의성을 위한 필요 표본 크기를 추정하라

연구 프로토콜을 구조화된 문서로 작성하라 (연구를 실행할 필요는 없다).

### 연습 4: 다중 방법 벤치마크 보고서 (고급)

ExplanationBenchmark 클래스를 확장하여 다음을 포함하라:
1. 최소 5개 설명 방법 (CNN용 Integrated Gradients, GradCAM 추가)
2. 각 방법에 대한 안정성 측정 (리프시츠 추정치)
3. 시드 민감도 측정
4. 모든 차원에서 방법을 비교하는 "레이더 차트" 시각화
5. (a) 높은 충실도가 필요한 헬스케어 환경과 (b) 속도가 필요한 실시간 서빙 환경에서 어떤 방법을 사용할지에 대한 서면 권장 사항

---

[이전: 해석 가능성을 위한 인과 추론](./09_Causal_Inference_for_Interpretability.md) | [개요](./00_Overview.md) | [다음: 고급 알고리즘 공정성](./11_Advanced_Algorithmic_Fairness.md)

**License**: CC BY-NC 4.0
