# Lesson 11: Advanced Algorithmic Fairness

[Previous: Evaluating Explanations](./10_Evaluating_Explanations.md) | [Next: Fairness Mitigation](./12_Fairness_Mitigation.md)

---

## Learning Objectives

- Understand individual fairness (Dwork et al., 2012) and how metric learning defines "similar individuals"
- Formalize counterfactual fairness (Kusner et al., 2017) using Structural Causal Models and reason about path-specific effects
- Prove and interpret the impossibility theorem: why calibration, false positive parity, and false negative parity cannot simultaneously hold except in degenerate cases
- Analyze intersectional fairness across multiple protected attributes
- Use Fairlearn and AIF360 toolkits to detect and quantify bias in ML models

---

## 1. Review of Group Fairness Basics

### 1.1 Foundation from Machine Learning Lesson 16

This lesson assumes familiarity with the basic group fairness definitions
covered in Machine Learning Lesson 16 (Model Explainability). We briefly
recap the three core metrics before moving to advanced topics.

**Demographic Parity** (Statistical Parity): P(Y_hat = 1 | A = 0) = P(Y_hat = 1 | A = 1).
The prediction rate should be the same across groups, regardless of the true label.
**Equalized Odds** (Hardt et al., 2016): P(Y_hat = 1 | A = a, Y = y) is the same for
all groups a, for both y = 0 and y = 1. Equal TPR and FPR across groups.
**Predictive Parity** (Calibration): P(Y = 1 | Y_hat = 1, A = a) is the same for all
groups. Among those predicted positive, the fraction of true positives is equal.

For a complete introduction, see [Machine Learning L16](../Machine_Learning/16_Model_Explainability.md).
This lesson moves beyond group-level metrics to individual, counterfactual, and
causal definitions of fairness.

```python
"""
Quick reference: computing the three basic group fairness metrics.
This code serves as a foundation for the advanced metrics in this lesson.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def compute_group_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
) -> pd.DataFrame:
    """Compute basic group fairness metrics for binary classification.

    Parameters
    ----------
    y_true : true labels (0 or 1)
    y_pred : predicted labels (0 or 1)
    sensitive_attr : binary sensitive attribute (0 or 1)

    Returns
    -------
    DataFrame with metrics per group and their differences
    """
    groups = sorted(np.unique(sensitive_attr))
    results = {}

    for group in groups:
        mask = sensitive_attr == group
        y_t = y_true[mask]
        y_p = y_pred[mask]

        tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()

        results[f"Group {group}"] = {
            "Selection Rate": y_p.mean(),
            "TPR (Recall)": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "FPR": fp / (fp + tn) if (fp + tn) > 0 else 0,
            "PPV (Precision)": tp / (tp + fp) if (tp + fp) > 0 else 0,
            "Accuracy": (tp + tn) / len(y_t),
        }

    df = pd.DataFrame(results).T

    # Fairness metrics (differences between groups)
    g0, g1 = f"Group {groups[0]}", f"Group {groups[1]}"
    fairness = {
        "Demographic Parity Diff": abs(
            df.loc[g0, "Selection Rate"] - df.loc[g1, "Selection Rate"]
        ),
        "Equalized Odds (TPR Diff)": abs(
            df.loc[g0, "TPR (Recall)"] - df.loc[g1, "TPR (Recall)"]
        ),
        "Equalized Odds (FPR Diff)": abs(
            df.loc[g0, "FPR"] - df.loc[g1, "FPR"]
        ),
        "Predictive Parity Diff": abs(
            df.loc[g0, "PPV (Precision)"] - df.loc[g1, "PPV (Precision)"]
        ),
    }

    return df, fairness
```

---

## 2. Individual Fairness

### 2.1 The Dwork et al. (2012) Framework

Group fairness treats all members of a demographic group identically. But within
any group, individuals differ. Individual fairness (Dwork et al., 2012) requires
that **similar individuals receive similar outcomes**, regardless of which group
they belong to.

Formally, for a classifier h and a task-specific similarity metric d:

    d_outcome(h(x), h(x')) <= L * d_input(x, x')

This is a Lipschitz condition: the difference in outcomes should be bounded by
a constant L times the difference in inputs (under the appropriate metric).

```python
"""
Individual fairness: similar individuals should receive similar predictions.

The critical challenge is defining "similarity" -- it must be a task-specific
metric that captures legitimate differences (qualifications) while ignoring
protected attributes (race, gender). This is where domain expertise is
essential.
"""

from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform


def individual_fairness_score(
    model,
    X: np.ndarray,
    similarity_metric: str = "euclidean",
    k_neighbors: int = 10,
    sensitive_features: list = None,
) -> dict:
    """Measure individual fairness by checking consistency among similar individuals.

    For each individual, we find their k nearest neighbors (using a metric
    that EXCLUDES sensitive features) and check whether the model gives
    them similar predictions.

    A high consistency score means the model treats similar people similarly.
    Low consistency indicates potential individual fairness violations.

    Parameters
    ----------
    model : trained classifier with predict_proba
    X : feature matrix
    similarity_metric : distance metric for finding neighbors
    k_neighbors : number of neighbors to consider
    sensitive_features : indices of features to EXCLUDE from similarity
                        computation (the metric should not depend on protected
                        attributes)
    """
    n_samples = len(X)

    # Build similarity space excluding sensitive features
    if sensitive_features:
        non_sensitive_idx = [
            i for i in range(X.shape[1]) if i not in sensitive_features
        ]
        X_similarity = X[:, non_sensitive_idx]
    else:
        X_similarity = X

    # Find k nearest neighbors in the non-sensitive feature space
    nn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric=similarity_metric)
    nn.fit(X_similarity)
    distances, indices = nn.kneighbors(X_similarity)

    # Get model predictions
    predictions = model.predict_proba(X)[:, 1]

    # Measure consistency: for each individual, how much do their
    # neighbors' predictions differ from theirs?
    consistency_scores = []
    max_violations = []

    for i in range(n_samples):
        # Exclude self (first neighbor is always self)
        neighbor_idx = indices[i, 1:]
        neighbor_preds = predictions[neighbor_idx]
        own_pred = predictions[i]

        # Mean absolute difference with neighbors
        mean_diff = np.abs(neighbor_preds - own_pred).mean()
        max_diff = np.abs(neighbor_preds - own_pred).max()

        consistency_scores.append(1.0 - mean_diff)
        max_violations.append(max_diff)

    return {
        "mean_consistency": np.mean(consistency_scores),
        "min_consistency": np.min(consistency_scores),
        "mean_max_violation": np.mean(max_violations),
        "worst_violation": np.max(max_violations),
        "pct_highly_consistent": (np.array(consistency_scores) > 0.9).mean() * 100,
    }


# --- Example ---
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

np.random.seed(42)
n = 3000

# Generate data where gender (sensitive) is correlated with outcome
# but should not affect the prediction
gender = np.random.binomial(1, 0.5, n)
education = np.random.normal(14, 2, n)
experience = np.random.normal(10, 3, n)
skill_score = np.random.normal(50, 10, n)

# True outcome depends on education, experience, skill -- NOT gender
logit = -5 + 0.3 * education + 0.2 * experience + 0.05 * skill_score
prob = 1 / (1 + np.exp(-logit))
hired = np.random.binomial(1, prob, n)

X = np.column_stack([gender, education, experience, skill_score])
feature_names = ["gender", "education", "experience", "skill_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, hired, test_size=0.3, random_state=42
)

# Train model (may inadvertently use gender)
model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Measure individual fairness
fairness_result = individual_fairness_score(
    model, X_test,
    k_neighbors=10,
    sensitive_features=[0],  # Exclude gender from similarity metric
)

print("=== Individual Fairness Score ===")
for key, value in fairness_result.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")
```

### 2.2 Metric Learning for Similarity

The most challenging aspect of individual fairness is defining the right
similarity metric. Ilvento (2020) and others have proposed learning the
metric from human judgments or domain constraints.

```python
"""
Learning a fair similarity metric from domain constraints.

The idea: instead of using Euclidean distance (which treats all features
equally), learn a Mahalanobis distance that:
1. Captures legitimate differences between individuals
2. Ignores protected attributes
3. Reflects domain-specific notions of "similar qualifications"
"""

from sklearn.preprocessing import StandardScaler


def learn_fair_metric(
    X: np.ndarray,
    sensitive_idx: list,
    feature_names: list,
    method: str = "project_out",
) -> np.ndarray:
    """Learn a fair similarity metric by projecting out sensitive features.

    This produces a transformation matrix W such that distances
    computed as ||W(x - x')||_2 are invariant to the sensitive features.

    Method "project_out": Remove the sensitive feature dimensions and
    re-weight remaining features by their variance. This is the simplest
    approach but may miss correlations between sensitive and non-sensitive
    features.

    Parameters
    ----------
    X : feature matrix
    sensitive_idx : indices of sensitive features
    feature_names : feature names
    method : "project_out" for simple projection

    Returns
    -------
    W : transformation matrix (n_non_sensitive x n_features)
    """
    n_features = X.shape[1]
    non_sensitive_idx = [i for i in range(n_features) if i not in sensitive_idx]

    if method == "project_out":
        # Simple: project onto non-sensitive feature space
        W = np.zeros((len(non_sensitive_idx), n_features))
        for i, idx in enumerate(non_sensitive_idx):
            W[i, idx] = 1.0

        # Scale by inverse standard deviation for normalized distances
        scaler = StandardScaler()
        X_ns = X[:, non_sensitive_idx]
        scaler.fit(X_ns)
        scale_diag = np.diag(1.0 / (scaler.scale_ + 1e-10))
        W = scale_diag @ W

        print("Fair metric learned (projection method):")
        print(f"  Original features: {n_features}")
        print(f"  Fair metric dimensions: {len(non_sensitive_idx)}")
        print(f"  Removed features: {[feature_names[i] for i in sensitive_idx]}")

    return W


def compute_fair_distances(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Compute pairwise distances using the fair metric.

    d_fair(x, x') = ||W(x - x')||_2

    This distance is blind to sensitive features, so individuals who
    differ only in their protected attribute will have distance = 0.
    """
    X_transformed = X @ W.T
    distances = squareform(pdist(X_transformed, metric="euclidean"))
    return distances


# Learn and apply the fair metric
W = learn_fair_metric(X_test, [0], feature_names)
fair_distances = compute_fair_distances(X_test[:100], W)
regular_distances = squareform(pdist(X_test[:100], metric="euclidean"))

print(f"\nFair distance range: [{fair_distances.min():.3f}, {fair_distances[fair_distances > 0].max():.3f}]")
print(f"Regular distance range: [{regular_distances.min():.3f}, {regular_distances[regular_distances > 0].max():.3f}]")
```

---

## 3. Counterfactual Fairness

### 3.1 The Kusner et al. (2017) Definition

Counterfactual fairness asks: "Would this individual have received the same
prediction in a counterfactual world where their sensitive attribute was
different?" This is a causal notion of fairness that requires a Structural
Causal Model.

Formally, a predictor Y_hat is counterfactually fair if:

    P(Y_hat_A<-a(U) = y | X = x, A = a) = P(Y_hat_A<-a'(U) = y | X = x, A = a)

In words: the distribution of predictions would be the same whether the
individual's sensitive attribute A were a or a' (in the counterfactual world),
given the same exogenous variables U.

```python
"""
Counterfactual fairness: would the prediction change if the individual's
sensitive attribute were different, keeping everything else at its
natural value?

This requires a causal model because changing A may causally affect other
features (e.g., changing gender might affect salary through discrimination,
which affects creditworthiness). Counterfactual fairness says: trace through
ALL causal consequences and check if the prediction changes.
"""

from dataclasses import dataclass
from typing import Dict, List, Callable, Tuple


@dataclass
class FairnessSCM:
    """Structural Causal Model for fairness analysis.

    This SCM explicitly models how the sensitive attribute A
    causally affects other features and the outcome.
    """
    variables: List[str]
    equations: Dict[str, Tuple[List[str], Callable]]
    noise_distributions: Dict[str, Callable]

    def compute_counterfactual(
        self,
        factual_data: Dict[str, float],
        intervention: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute counterfactual values under an intervention.

        Given the factual observation (what actually happened), compute
        what WOULD have happened if we intervened on some variables.

        Steps (Pearl's three-step counterfactual procedure):
        1. ABDUCTION: Infer the exogenous noise U from the factual data
        2. ACTION: Apply the intervention (modify structural equations)
        3. PREDICTION: Propagate forward with the inferred noise

        For linear models, this simplifies to direct computation.
        """
        # Step 1: Abduction -- infer noise from factual data
        inferred_noise = {}
        for var in self.variables:
            parents, func = self.equations[var]
            parent_values = {p: factual_data[p] for p in parents}
            # For linear models: noise = observed - f(parents)
            # We approximate by computing the residual
            predicted = func(parent_values, 0)  # Zero noise
            inferred_noise[var] = factual_data[var] - predicted

        # Step 2 & 3: Action + Prediction
        counterfactual = {}
        for var in self.variables:
            if var in intervention:
                # Intervened variable takes the intervention value
                counterfactual[var] = intervention[var]
            else:
                parents, func = self.equations[var]
                parent_values = {
                    p: counterfactual[p] if p in counterfactual
                    else factual_data[p]
                    for p in parents
                }
                # Use the SAME noise as in the factual world
                counterfactual[var] = func(
                    parent_values, inferred_noise[var]
                )

        return counterfactual


def evaluate_counterfactual_fairness(
    model,
    scm: FairnessSCM,
    data: pd.DataFrame,
    sensitive_attr: str,
    feature_names: List[str],
    counterfactual_value: float = None,
) -> dict:
    """Evaluate counterfactual fairness of a model.

    For each individual:
    1. Compute the counterfactual: what would their features be if
       their sensitive attribute were different?
    2. Predict using the counterfactual features
    3. Compare with the factual prediction

    If predictions differ, the model is counterfactually unfair for
    that individual.
    """
    results = []
    unique_values = data[sensitive_attr].unique()

    for idx, row in data.iterrows():
        factual = row.to_dict()
        factual_pred = model.predict_proba(
            row[feature_names].values.reshape(1, -1)
        )[0, 1]

        # Counterfactual: flip the sensitive attribute
        if counterfactual_value is not None:
            cf_value = counterfactual_value
        else:
            cf_value = 1.0 - factual[sensitive_attr]

        # Compute counterfactual features using the SCM
        cf_data = scm.compute_counterfactual(
            factual, {sensitive_attr: cf_value}
        )

        cf_features = np.array([cf_data[f] for f in feature_names])
        cf_pred = model.predict_proba(cf_features.reshape(1, -1))[0, 1]

        results.append({
            "factual_pred": factual_pred,
            "counterfactual_pred": cf_pred,
            "pred_difference": abs(factual_pred - cf_pred),
            "factual_sensitive": factual[sensitive_attr],
        })

    results_df = pd.DataFrame(results)

    return {
        "mean_pred_difference": results_df["pred_difference"].mean(),
        "max_pred_difference": results_df["pred_difference"].max(),
        "pct_unfair": (results_df["pred_difference"] > 0.05).mean() * 100,
        "counterfactual_flip_rate": (
            (results_df["factual_pred"] > 0.5) !=
            (results_df["counterfactual_pred"] > 0.5)
        ).mean() * 100,
    }


# --- Build SCM for the hiring example ---
# Causal structure:
#   Gender -> Education (discrimination in access to education)
#   Gender -> Experience (discrimination in hiring -> less experience)
#   Education -> Skill Score (better education -> better skills)
#   Education, Experience, Skill Score -> Hired (legitimate factors)
#   Gender -> Hired (should NOT exist in a fair world)

hiring_scm = FairnessSCM(
    variables=["gender", "education", "experience", "skill_score"],
    equations={
        "gender": ([], lambda p, n: n),  # Exogenous
        "education": (
            ["gender"],
            # Gender affects education through discrimination
            lambda p, n: 14 + 0.5 * p.get("gender", 0) + n,
        ),
        "experience": (
            ["gender", "education"],
            lambda p, n: 10 + 0.3 * p.get("gender", 0) - 0.1 * p.get("education", 14) + n,
        ),
        "skill_score": (
            ["education"],
            lambda p, n: 30 + 1.5 * p.get("education", 14) + n,
        ),
    },
    noise_distributions={
        "gender": lambda n: np.random.binomial(1, 0.5, n),
        "education": lambda n: np.random.normal(0, 2, n),
        "experience": lambda n: np.random.normal(0, 3, n),
        "skill_score": lambda n: np.random.normal(0, 10, n),
    },
)

# Evaluate counterfactual fairness
test_df = pd.DataFrame(X_test, columns=feature_names)
cf_result = evaluate_counterfactual_fairness(
    model, hiring_scm, test_df.head(200),
    sensitive_attr="gender",
    feature_names=feature_names,
)

print("=== Counterfactual Fairness Evaluation ===")
for key, value in cf_result.items():
    print(f"  {key}: {value:.4f}")
print("\n  A counterfactual flip rate > 0% means the model is")
print("  counterfactually unfair for some individuals.")
```

### 3.2 Path-Specific Fairness

Counterfactual fairness is an all-or-nothing criterion: any causal path from
the sensitive attribute to the prediction is considered unfair. But some paths
might be legitimate.

For example, in college admissions:
- Gender -> Test Score -> Admission: potentially unfair (if test is biased)
- Gender -> Major Choice -> Admission: possibly legitimate (free choice)

Path-specific fairness (Nabi & Shpitser, 2018) allows specifying which
causal pathways are fair and which are unfair.

```python
"""
Path-specific fairness: not all causal paths from sensitive attribute
to prediction are equally problematic.

Fair paths: A -> X -> Y where X is a legitimate mediator (free choice)
Unfair paths: A -> X -> Y where X is an illegitimate mediator (discrimination)

Example:
  Gender -> Major Choice -> Salary: fair (free choice)
  Gender -> Discrimination -> Lower Salary: unfair (bias)

Path-specific fairness requires blocking ONLY the unfair paths while
allowing the fair paths to operate.
"""


def path_specific_fairness(
    model,
    scm: FairnessSCM,
    data: pd.DataFrame,
    sensitive_attr: str,
    feature_names: List[str],
    unfair_mediators: List[str],
    fair_mediators: List[str],
) -> dict:
    """Evaluate path-specific fairness.

    We compute a "nested counterfactual" that:
    - Keeps fair mediators at their NATURAL values (as if A had not changed)
      Wait -- actually the opposite:
    - Sets unfair mediators to counterfactual values (as if A were different)
    - Keeps fair mediators at their factual values (as they actually are)

    This isolates the effect of the unfair pathway.

    The unfair path-specific effect is:
    E[Y_hat(a, M_unfair(a'), M_fair(a))] - E[Y_hat(a, M_unfair(a), M_fair(a))]
    """
    results = []

    for idx, row in data.iterrows():
        factual = row.to_dict()
        a_factual = factual[sensitive_attr]
        a_counter = 1.0 - a_factual

        # Factual prediction (everything as observed)
        factual_pred = model.predict_proba(
            row[feature_names].values.reshape(1, -1)
        )[0, 1]

        # Path-specific counterfactual:
        # Change A, update ONLY unfair mediators, keep fair mediators fixed
        cf_data = scm.compute_counterfactual(
            factual, {sensitive_attr: a_counter}
        )

        # Build hybrid features: unfair mediators from counterfactual,
        # fair mediators from factual, sensitive attribute stays factual
        hybrid = factual.copy()
        for mediator in unfair_mediators:
            hybrid[mediator] = cf_data[mediator]
        # Fair mediators stay at factual values (already in hybrid)

        hybrid_features = np.array([hybrid[f] for f in feature_names])
        hybrid_pred = model.predict_proba(hybrid_features.reshape(1, -1))[0, 1]

        # The unfair path-specific effect
        unfair_effect = abs(factual_pred - hybrid_pred)
        results.append(unfair_effect)

    return {
        "mean_unfair_path_effect": np.mean(results),
        "max_unfair_path_effect": np.max(results),
        "pct_affected": (np.array(results) > 0.05).mean() * 100,
    }


# Example: education is unfairly mediated (discrimination in access),
# but skill_score is fair (legitimate outcome of education)
ps_result = path_specific_fairness(
    model, hiring_scm, test_df.head(200),
    sensitive_attr="gender",
    feature_names=feature_names,
    unfair_mediators=["education"],  # Gender -> Education is unfair
    fair_mediators=["skill_score"],  # Education -> Skill is fair
)

print("=== Path-Specific Fairness ===")
for key, value in ps_result.items():
    print(f"  {key}: {value:.4f}")
```

---

## 4. The Impossibility Theorem

### 4.1 Chouldechova (2017) and Kleinberg et al. (2016)

One of the most important results in algorithmic fairness is the **impossibility
theorem**: except in degenerate cases, a classifier cannot simultaneously satisfy
calibration (predictive parity), balance for the positive class (equal FNR), and
balance for the negative class (equal FPR) across groups.

```python
"""
The Impossibility Theorem: formal proof sketch.

Notation:
  A in {0, 1}: sensitive attribute (group membership)
  Y in {0, 1}: true label
  S in [0,1]: risk score (model output)
  d: decision threshold (predict positive if S >= d)

Three fairness criteria:
  1. Calibration: P(Y=1|S=s, A=0) = P(Y=1|S=s, A=1) for all s
     (The risk score means the same thing for both groups)

  2. Balance for positives (equal FNR):
     E[S|Y=1, A=0] = E[S|Y=1, A=1]
     (True positives receive similar scores regardless of group)

  3. Balance for negatives (equal FPR):
     E[S|Y=0, A=0] = E[S|Y=0, A=1]
     (True negatives receive similar scores regardless of group)

Theorem (Chouldechova, 2017; Kleinberg et al., 2016):
  If the base rates differ (P(Y=1|A=0) != P(Y=1|A=1)),
  then criteria 1, 2, and 3 CANNOT all hold simultaneously
  (except when the classifier is perfect or trivial).
"""


def impossibility_demonstration(
    base_rate_0: float = 0.3,
    base_rate_1: float = 0.5,
    n_samples: int = 50000,
) -> None:
    """Demonstrate the impossibility theorem numerically.

    We train a well-calibrated classifier and show that it necessarily
    violates either balance for positives or balance for negatives
    when base rates differ between groups.
    """
    np.random.seed(42)

    # Generate data with different base rates
    n_per_group = n_samples // 2

    # Group 0: lower base rate
    y0 = np.random.binomial(1, base_rate_0, n_per_group)
    # Group 1: higher base rate
    y1 = np.random.binomial(1, base_rate_1, n_per_group)

    # Generate a CALIBRATED risk score
    # S | Y=1 ~ Beta(alpha_pos, beta_pos), S | Y=0 ~ Beta(alpha_neg, beta_neg)
    # Choose parameters so that the score is well-calibrated
    from scipy.stats import beta as beta_dist

    def generate_calibrated_scores(y, base_rate):
        """Generate approximately calibrated risk scores."""
        scores = np.zeros(len(y))
        # True positives get higher scores
        pos_mask = y == 1
        neg_mask = y == 0
        scores[pos_mask] = np.random.beta(5, 2, pos_mask.sum())
        scores[neg_mask] = np.random.beta(2, 5, neg_mask.sum())
        return scores

    s0 = generate_calibrated_scores(y0, base_rate_0)
    s1 = generate_calibrated_scores(y1, base_rate_1)

    print(f"=== Impossibility Theorem Demonstration ===")
    print(f"Base rate Group 0: {base_rate_0:.2f}")
    print(f"Base rate Group 1: {base_rate_1:.2f}")

    # Check calibration (approximately)
    for group_name, scores, labels in [("Group 0", s0, y0), ("Group 1", s1, y1)]:
        # Bin scores and check calibration
        bins = np.linspace(0, 1, 11)
        bin_indices = np.digitize(scores, bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(bins) - 2)

        print(f"\n{group_name} Calibration (binned):")
        for b in range(len(bins) - 1):
            mask = bin_indices == b
            if mask.sum() > 10:
                actual_rate = labels[mask].mean()
                bin_center = (bins[b] + bins[b + 1]) / 2
                print(f"  Score [{bins[b]:.1f}, {bins[b+1]:.1f}): "
                      f"predicted={bin_center:.2f}, actual={actual_rate:.2f}")

    # Check balance for positives: E[S|Y=1, A=a]
    mean_score_pos_0 = s0[y0 == 1].mean()
    mean_score_pos_1 = s1[y1 == 1].mean()
    print(f"\nBalance for Positives:")
    print(f"  E[S|Y=1, A=0] = {mean_score_pos_0:.4f}")
    print(f"  E[S|Y=1, A=1] = {mean_score_pos_1:.4f}")
    print(f"  Difference:     {abs(mean_score_pos_0 - mean_score_pos_1):.4f}")

    # Check balance for negatives: E[S|Y=0, A=a]
    mean_score_neg_0 = s0[y0 == 0].mean()
    mean_score_neg_1 = s1[y1 == 0].mean()
    print(f"\nBalance for Negatives:")
    print(f"  E[S|Y=0, A=0] = {mean_score_neg_0:.4f}")
    print(f"  E[S|Y=0, A=1] = {mean_score_neg_1:.4f}")
    print(f"  Difference:     {abs(mean_score_neg_0 - mean_score_neg_1):.4f}")

    # The key insight: with different base rates and calibration,
    # the balance conditions MUST differ
    print("\n--- Impossibility Result ---")
    print("With different base rates AND calibration, the model")
    print("CANNOT achieve equal balance for both positives and negatives.")
    print("This is a mathematical impossibility, not a failure of the algorithm.")

impossibility_demonstration(base_rate_0=0.3, base_rate_1=0.5)
```

### 4.2 The Formal Proof Sketch

```python
"""
Formal proof sketch of the impossibility theorem.

Setup:
  Let p_a = P(Y=1|A=a) be the base rate for group a.
  Let S be a calibrated score: P(Y=1|S=s, A=a) = s for all s, a.
  Let mu_a+ = E[S|Y=1, A=a] and mu_a- = E[S|Y=0, A=a].

By the law of total probability:
  p_a = E[S|A=a] = P(Y=1|A=a) * E[S|Y=1, A=a] + P(Y=0|A=a) * E[S|Y=0, A=a]
  p_a = p_a * mu_a+ + (1 - p_a) * mu_a-

Rearranging:
  mu_a- = (p_a - p_a * mu_a+) / (1 - p_a)
  mu_a- = p_a * (1 - mu_a+) / (1 - p_a)

Now, suppose balance for positives holds: mu_0+ = mu_1+ = mu+
Then:
  mu_0- = p_0 * (1 - mu+) / (1 - p_0)
  mu_1- = p_1 * (1 - mu+) / (1 - p_1)

For balance for negatives: mu_0- = mu_1- requires:
  p_0 * (1 - mu+) / (1 - p_0) = p_1 * (1 - mu+) / (1 - p_1)

If mu+ != 1 (non-trivial classifier), we can divide both sides by (1 - mu+):
  p_0 / (1 - p_0) = p_1 / (1 - p_1)

This implies p_0 = p_1, contradicting our assumption of unequal base rates.

QED: Calibration + Balance+ + Balance- implies p_0 = p_1. ///
"""


def verify_impossibility_algebra(p0: float, p1: float, mu_plus: float) -> None:
    """Verify the impossibility theorem algebraically.

    Given base rates p0, p1 and a shared positive balance mu_plus,
    compute the implied negative balance and show they differ.
    """
    # If balance for positives holds with shared value mu_plus:
    mu_0_minus = p0 * (1 - mu_plus) / (1 - p0)
    mu_1_minus = p1 * (1 - mu_plus) / (1 - p1)

    print(f"Given: p0={p0}, p1={p1}, mu+={mu_plus}")
    print(f"  mu_0- = {p0} * (1-{mu_plus}) / (1-{p0}) = {mu_0_minus:.4f}")
    print(f"  mu_1- = {p1} * (1-{mu_plus}) / (1-{p1}) = {mu_1_minus:.4f}")
    print(f"  Difference: {abs(mu_0_minus - mu_1_minus):.4f}")

    if abs(p0 - p1) > 1e-10:
        print("  => Balance for negatives is VIOLATED (impossibility confirmed)")
    else:
        print("  => Base rates are equal, so all three can hold simultaneously")

verify_impossibility_algebra(0.3, 0.5, 0.7)
verify_impossibility_algebra(0.4, 0.4, 0.7)  # Equal base rates: possible
```

### 4.3 Practical Implications

```python
"""
Practical implications of the impossibility theorem.

Since we cannot have everything, practitioners must CHOOSE which
fairness criterion to prioritize based on the specific context.
"""

fairness_tradeoffs = {
    "Criminal Justice (e.g., recidivism)": {
        "priority": "Calibration",
        "reasoning": "A risk score of 0.7 should mean 70% recidivism risk "
                     "regardless of race. Different base rates in the data "
                     "may lead to different FPR/FNR across groups.",
        "tradeoff": "Accept unequal error rates across groups to maintain "
                    "calibrated risk assessments.",
        "example": "COMPAS score calibration: both groups with score 7 "
                   "should have similar actual recidivism rates.",
    },
    "Lending (e.g., loan approval)": {
        "priority": "Equal FPR (balance for negatives)",
        "reasoning": "Equal FPR means qualified applicants from all groups "
                     "have the same chance of being correctly approved. "
                     "Focuses on equal opportunity for the deserving.",
        "tradeoff": "May sacrifice calibration: the same approval score "
                    "may mean different things for different groups.",
        "example": "Equal Credit Opportunity Act: similar denial rates for "
                   "qualified applicants across racial groups.",
    },
    "Healthcare (e.g., disease screening)": {
        "priority": "Equal FNR (balance for positives)",
        "reasoning": "Missing a disease case (false negative) is more costly "
                     "than a false alarm. We want equal sensitivity across "
                     "all demographic groups.",
        "tradeoff": "May have higher FPR for some groups (more unnecessary "
                    "follow-up tests) to ensure equal disease detection.",
        "example": "Mammography screening should detect cancer at equal "
                   "rates regardless of race/ethnicity.",
    },
}

print("=== Impossibility Theorem: Choosing a Fairness Criterion ===\n")
for domain, details in fairness_tradeoffs.items():
    print(f"Domain: {domain}")
    print(f"  Priority:   {details['priority']}")
    print(f"  Reasoning:  {details['reasoning']}")
    print(f"  Tradeoff:   {details['tradeoff']}")
    print(f"  Example:    {details['example']}")
    print()
```

---

## 5. Intersectional Fairness

### 5.1 Why Single-Axis Analysis Is Insufficient

Standard fairness analysis considers one protected attribute at a time (gender OR
race). But discrimination often operates at the intersection of multiple identities
(e.g., Black women face challenges distinct from those of Black men or white women).
Crenshaw (1989) coined the term "intersectionality" for this phenomenon.

```python
"""
Intersectional fairness analysis: examining fairness across
combinations of multiple protected attributes.

Single-axis analysis can miss violations that only appear at
intersections. For example, a model might be fair with respect
to gender and fair with respect to race, but unfair for a
specific race-gender combination.
"""


def intersectional_fairness_audit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected_attrs: pd.DataFrame,
) -> pd.DataFrame:
    """Audit fairness across all intersections of protected attributes.

    For each combination of protected attribute values, compute
    fairness metrics and flag potential violations.

    Parameters
    ----------
    y_true : true labels
    y_pred : predicted labels
    protected_attrs : DataFrame with columns for each protected attribute

    Returns
    -------
    DataFrame with metrics per intersectional group
    """
    # Create intersection groups
    intersection_col = protected_attrs.apply(
        lambda row: "_".join(str(v) for v in row), axis=1
    )

    results = []
    overall_selection_rate = y_pred.mean()
    overall_tpr = y_true[y_pred == 1].mean() if y_pred.sum() > 0 else 0

    for group_name in intersection_col.unique():
        mask = intersection_col == group_name
        n_group = mask.sum()

        if n_group < 20:  # Skip very small groups
            continue

        y_t = y_true[mask]
        y_p = y_pred[mask]

        selection_rate = y_p.mean()
        base_rate = y_t.mean()

        # Compute metrics
        tp = ((y_t == 1) & (y_p == 1)).sum()
        fp = ((y_t == 0) & (y_p == 1)).sum()
        fn = ((y_t == 1) & (y_p == 0)).sum()
        tn = ((y_t == 0) & (y_p == 0)).sum()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan

        # Disparate impact ratio: selection_rate / overall_selection_rate
        di_ratio = (
            selection_rate / overall_selection_rate
            if overall_selection_rate > 0 else np.nan
        )

        results.append({
            "group": group_name,
            "n": n_group,
            "base_rate": base_rate,
            "selection_rate": selection_rate,
            "TPR": tpr,
            "FPR": fpr,
            "PPV": ppv,
            "disparate_impact_ratio": di_ratio,
            # Four-fifths rule: DI ratio < 0.8 suggests disparate impact
            "four_fifths_violation": di_ratio < 0.8 if not np.isnan(di_ratio) else None,
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("disparate_impact_ratio")

    return results_df


# --- Generate intersectional data ---
np.random.seed(42)
n = 5000

race = np.random.choice([0, 1, 2], n, p=[0.6, 0.25, 0.15])  # 3 groups
gender = np.random.binomial(1, 0.5, n)
age_group = (np.random.normal(40, 10, n) > 45).astype(int)

# Features
education = np.random.normal(14, 2, n) + 0.5 * gender - 0.3 * (race == 2)
experience = np.random.normal(10, 3, n) + 2 * age_group
score = np.random.normal(50, 10, n)

# True outcome
logit = -8 + 0.3 * education + 0.15 * experience + 0.04 * score
y_true = np.random.binomial(1, 1 / (1 + np.exp(-logit)), n)

# Biased model predictions (introduce intersectional bias)
# The model is slightly biased against race=2 AND gender=0
bias = -0.3 * (race == 2) * (1 - gender)  # Bias against race-2 women
logit_biased = logit + bias
y_pred = (1 / (1 + np.exp(-logit_biased)) > 0.5).astype(int)

# Audit
protected = pd.DataFrame({
    "race": race,
    "gender": gender,
})

audit_df = intersectional_fairness_audit(y_true, y_pred, protected)

print("=== Intersectional Fairness Audit ===")
print(audit_df.to_string(index=False))

# Check single-axis vs intersectional results
print("\n=== Single-Axis Analysis ===")
for attr in ["race", "gender"]:
    single_audit = intersectional_fairness_audit(
        y_true, y_pred,
        protected[[attr]],
    )
    print(f"\nBy {attr}:")
    print(single_audit[["group", "n", "selection_rate", "disparate_impact_ratio"]].to_string(index=False))

print("\nNotice: single-axis analysis may show acceptable fairness,")
print("but intersectional analysis reveals specific subgroup violations.")
```

---

## 6. Fairlearn Toolkit

### 6.1 MetricFrame: Disaggregated Metrics

```python
"""
Fairlearn: Microsoft's toolkit for assessing and improving fairness.

MetricFrame computes any sklearn metric disaggregated by sensitive features,
making it easy to identify fairness violations.
"""

# pip install fairlearn

from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score


def fairlearn_audit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
    feature_name: str = "sensitive",
) -> None:
    """Comprehensive fairness audit using Fairlearn MetricFrame.

    MetricFrame disaggregates any metric by sensitive features,
    computing per-group values and overall statistics.
    """
    # Define metrics to compute
    metrics = {
        "accuracy": accuracy_score,
        "precision": precision_score,
        "recall": recall_score,
        "selection_rate": selection_rate,
    }

    # Create MetricFrame
    mf = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )

    print(f"=== Fairlearn Audit (by {feature_name}) ===\n")
    print("Per-group metrics:")
    print(mf.by_group.round(4).to_string())

    print(f"\nOverall metrics:")
    print(mf.overall.round(4).to_string())

    print(f"\nDifferences (max - min across groups):")
    print(mf.difference(method="between_groups").round(4).to_string())

    print(f"\nRatios (min / max across groups):")
    print(mf.ratio(method="between_groups").round(4).to_string())

    # Summary fairness metrics
    dp_diff = demographic_parity_difference(
        y_true, y_pred, sensitive_features=sensitive_features
    )
    dp_ratio = demographic_parity_ratio(
        y_true, y_pred, sensitive_features=sensitive_features
    )

    print(f"\nDemographic Parity Difference: {dp_diff:.4f}")
    print(f"Demographic Parity Ratio:      {dp_ratio:.4f}")
    print(f"  (Four-fifths rule threshold:  0.80)")
    print(f"  {'VIOLATION' if dp_ratio < 0.8 else 'PASS'}")


# Run Fairlearn audit
fairlearn_audit(y_true, y_pred, race, "race")
fairlearn_audit(y_true, y_pred, gender, "gender")
```

### 6.2 Fairlearn Dashboard Concepts

```python
"""
Fairlearn's assessment dashboard (programmatic equivalent).

The Fairlearn dashboard provides interactive exploration of fairness
metrics. Here we create a programmatic equivalent that generates
the same insights.
"""


def fairness_dashboard_report(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sensitive_features: np.ndarray,
    feature_names: list,
    sensitive_name: str,
) -> dict:
    """Generate a comprehensive fairness dashboard report.

    Includes:
    1. Model performance by group
    2. Fairness metrics summary
    3. Distribution analysis
    4. Recommendations
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    groups = sorted(np.unique(sensitive_features))
    report = {"groups": {}}

    for group in groups:
        mask = sensitive_features == group
        n_group = mask.sum()

        report["groups"][f"{sensitive_name}={group}"] = {
            "n": int(n_group),
            "base_rate": float(y_test[mask].mean()),
            "selection_rate": float(y_pred[mask].mean()),
            "accuracy": float(accuracy_score(y_test[mask], y_pred[mask])),
            "mean_score": float(y_prob[mask].mean()),
            "score_std": float(y_prob[mask].std()),
        }

    # Overall fairness summary
    mf = MetricFrame(
        metrics={"selection_rate": selection_rate, "accuracy": accuracy_score},
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )

    report["fairness_summary"] = {
        "demographic_parity_diff": float(
            mf.difference(method="between_groups")["selection_rate"]
        ),
        "accuracy_diff": float(
            mf.difference(method="between_groups")["accuracy"]
        ),
    }

    return report


# Generate report
X_all = np.column_stack([gender, education, experience, score])
report = fairness_dashboard_report(
    model, X_test, y_test,
    sensitive_features=X_test[:, 0].astype(int),  # gender
    feature_names=feature_names,
    sensitive_name="gender",
)

print("=== Fairness Dashboard Report ===")
import json
print(json.dumps(report, indent=2, default=str))
```

---

## 7. AIF360 Toolkit

### 7.1 Bias Detection with AIF360

```python
"""
AIF360 (AI Fairness 360): IBM's comprehensive toolkit for fairness.

AIF360 provides a broader set of fairness metrics than Fairlearn,
including individual fairness metrics, and a rich set of bias
mitigation algorithms.
"""

# pip install aif360

# AIF360 uses its own data format: BinaryLabelDataset
# Here we show the core concepts programmatically

from collections import defaultdict


class FairnessMetricsSuite:
    """Comprehensive fairness metrics implementation.

    Implements the core metrics from AIF360 without requiring
    the full AIF360 installation. Useful for understanding what
    each metric measures.
    """

    def __init__(self, y_true, y_pred, y_prob, sensitive):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_prob = np.array(y_prob)
        self.sensitive = np.array(sensitive)
        self.groups = sorted(np.unique(sensitive))

    def disparate_impact(self) -> float:
        """Ratio of selection rates: min(SR_a) / max(SR_a).

        Values < 0.8 indicate disparate impact (four-fifths rule).
        Perfect fairness: 1.0.
        """
        rates = [self.y_pred[self.sensitive == g].mean() for g in self.groups]
        return min(rates) / max(rates) if max(rates) > 0 else 0

    def statistical_parity_difference(self) -> float:
        """Difference in selection rates between groups.

        Perfect fairness: 0.0.
        """
        rates = [self.y_pred[self.sensitive == g].mean() for g in self.groups]
        return max(rates) - min(rates)

    def equal_opportunity_difference(self) -> float:
        """Difference in True Positive Rates between groups.

        Measures whether qualified individuals from all groups
        have equal chances of being correctly identified.
        """
        tprs = []
        for g in self.groups:
            mask = (self.sensitive == g) & (self.y_true == 1)
            if mask.sum() > 0:
                tprs.append(self.y_pred[mask].mean())
        return max(tprs) - min(tprs) if tprs else 0

    def average_odds_difference(self) -> float:
        """Average of TPR difference and FPR difference.

        This is the metric used in equalized odds.
        """
        tpr_diff = self.equal_opportunity_difference()

        fprs = []
        for g in self.groups:
            mask = (self.sensitive == g) & (self.y_true == 0)
            if mask.sum() > 0:
                fprs.append(self.y_pred[mask].mean())
        fpr_diff = max(fprs) - min(fprs) if fprs else 0

        return (tpr_diff + fpr_diff) / 2

    def theil_index(self) -> float:
        """Theil index: an individual fairness metric.

        Measures inequality in the benefit (or harm) received by
        individuals. Based on information theory.

        Lower = more individually fair. 0 = perfect equality.
        """
        # Benefit: y_pred == y_true (correct prediction is a benefit)
        benefits = (self.y_pred == self.y_true).astype(float)
        mean_benefit = benefits.mean()

        if mean_benefit == 0:
            return float("inf")

        normalized = benefits / mean_benefit
        # Avoid log(0)
        normalized = np.clip(normalized, 1e-10, None)
        theil = (normalized * np.log(normalized)).mean()

        return theil

    def summary(self) -> pd.DataFrame:
        """Generate a summary of all fairness metrics."""
        metrics = {
            "Disparate Impact Ratio": self.disparate_impact(),
            "Statistical Parity Diff": self.statistical_parity_difference(),
            "Equal Opportunity Diff": self.equal_opportunity_difference(),
            "Average Odds Diff": self.average_odds_difference(),
            "Theil Index": self.theil_index(),
        }

        thresholds = {
            "Disparate Impact Ratio": (">=", 0.8),
            "Statistical Parity Diff": ("<=", 0.1),
            "Equal Opportunity Diff": ("<=", 0.1),
            "Average Odds Diff": ("<=", 0.1),
            "Theil Index": ("<=", 0.1),
        }

        rows = []
        for name, value in metrics.items():
            op, threshold = thresholds[name]
            if op == ">=":
                passed = value >= threshold
            else:
                passed = value <= threshold

            rows.append({
                "Metric": name,
                "Value": value,
                "Threshold": f"{op} {threshold}",
                "Status": "PASS" if passed else "FAIL",
            })

        return pd.DataFrame(rows)


# Run comprehensive fairness audit
suite = FairnessMetricsSuite(
    y_true=y_true,
    y_pred=y_pred,
    y_prob=1 / (1 + np.exp(-logit_biased)),
    sensitive=race,
)

print("=== Comprehensive Fairness Metrics ===")
print(suite.summary().to_string(index=False))
```

---

## 8. Practical: Auditing a COMPAS-Style Recidivism Model

### 8.1 Data Generation and Model Training

```python
"""
Practical: Auditing a COMPAS-style recidivism prediction model.

COMPAS (Correctional Offender Management Profiling for Alternative
Sanctions) is a proprietary algorithm used in US courts to predict
recidivism risk. ProPublica's 2016 analysis revealed that COMPAS
exhibited racial bias: similar error rates overall but very different
false positive rates for Black vs white defendants.

We create a synthetic dataset that exhibits similar patterns.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def generate_compas_like_data(n: int = 10000) -> pd.DataFrame:
    """Generate synthetic recidivism data with realistic biases.

    The data is designed to exhibit the same type of bias found
    in the real COMPAS analysis:
    - Different base rates across racial groups (reflecting systemic
      factors, not inherent differences)
    - A well-meaning model that achieves similar overall accuracy
      but very different false positive rates
    """
    np.random.seed(42)

    # Protected attribute: race (0 = white, 1 = Black)
    # In real data, base rates differ due to systemic factors
    race = np.random.binomial(1, 0.4, n)

    # Features (some correlated with race due to systemic inequality)
    age = np.random.normal(33, 8, n) - 2 * race  # Age at assessment
    age = np.clip(age, 18, 70)

    prior_convictions = np.random.poisson(1.5 + 0.5 * race, n)
    prior_convictions = np.clip(prior_convictions, 0, 15)

    # Socioeconomic features correlated with race
    employment_score = np.random.normal(5, 2, n) - 1.0 * race
    employment_score = np.clip(employment_score, 0, 10)

    education_years = np.random.normal(12, 2, n) - 0.5 * race
    education_years = np.clip(education_years, 6, 20)

    # Charge severity (1-10)
    charge_severity = np.random.poisson(3, n) + 1
    charge_severity = np.clip(charge_severity, 1, 10)

    # True recidivism (2-year window)
    # Depends on legitimate factors + systemic factors
    logit = (
        -2.0
        + 0.15 * prior_convictions
        - 0.03 * age
        - 0.1 * employment_score
        - 0.05 * education_years
        + 0.1 * charge_severity
        + 0.3 * race  # Systemic factor (not individual tendency)
    )
    prob_recidivism = 1 / (1 + np.exp(-logit))
    recidivism = np.random.binomial(1, prob_recidivism, n)

    return pd.DataFrame({
        "race": race,
        "age": age,
        "prior_convictions": prior_convictions,
        "employment_score": employment_score,
        "education_years": education_years,
        "charge_severity": charge_severity,
        "recidivism": recidivism,
    })


# Generate data
compas_data = generate_compas_like_data(10000)

print("=== COMPAS-Like Dataset ===")
print(f"Total samples: {len(compas_data)}")
print(f"\nBase rates by race:")
print(compas_data.groupby("race")["recidivism"].mean())
print(f"\nFeature means by race:")
print(compas_data.groupby("race")[
    ["age", "prior_convictions", "employment_score", "education_years"]
].mean().round(2))
```

### 8.2 Training and Comprehensive Audit

```python
"""
Train a model and perform a comprehensive fairness audit.
"""

# Prepare data
features = ["age", "prior_convictions", "employment_score",
            "education_years", "charge_severity"]
X = compas_data[features].values
y = compas_data["recidivism"].values
race_attr = compas_data["race"].values

X_train, X_test, y_train, y_test, race_train, race_test = train_test_split(
    X, y, race_attr, test_size=0.3, random_state=42
)

# Train model (intentionally does NOT use race as a feature)
recid_model = RandomForestClassifier(
    n_estimators=200, max_depth=6, random_state=42
)
recid_model.fit(X_train, y_train)

y_pred = recid_model.predict(X_test)
y_prob = recid_model.predict_proba(X_test)[:, 1]

print("=== Model Performance ===")
print(classification_report(y_test, y_pred, target_names=["No Recid", "Recid"]))

# --- Full fairness audit ---
print("\n=== Group Fairness Metrics ===")
metrics_df, fairness_dict = compute_group_fairness_metrics(y_test, y_pred, race_test)
print(metrics_df.round(4).to_string())
print("\nFairness gaps:")
for metric, value in fairness_dict.items():
    status = "PASS" if value < 0.1 else "FAIL"
    print(f"  {metric}: {value:.4f} [{status}]")

# Individual fairness
print("\n=== Individual Fairness ===")
if_result = individual_fairness_score(
    recid_model, X_test,
    k_neighbors=10,
    sensitive_features=None,  # All features used for similarity
)
print(f"  Mean consistency: {if_result['mean_consistency']:.4f}")
print(f"  Worst violation:  {if_result['worst_violation']:.4f}")

# Intersectional analysis (race x age group)
print("\n=== Intersectional Analysis (race x age_group) ===")
age_group = (X_test[:, 0] > 35).astype(int)
intersectional_audit = intersectional_fairness_audit(
    y_test, y_pred,
    pd.DataFrame({"race": race_test, "age_group": age_group}),
)
print(intersectional_audit[
    ["group", "n", "selection_rate", "TPR", "FPR", "disparate_impact_ratio"]
].to_string(index=False))

# Comprehensive metrics suite
print("\n=== Comprehensive Metrics Suite ===")
suite = FairnessMetricsSuite(y_test, y_pred, y_prob, race_test)
print(suite.summary().to_string(index=False))
```

### 8.3 Calibration Analysis

```python
"""
Calibration analysis: does the model's risk score mean the same thing
for both racial groups?

This directly relates to the impossibility theorem: if we achieve
calibration, we may not achieve equal error rates.
"""


def calibration_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sensitive: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Analyze calibration by sensitive group.

    A well-calibrated model satisfies: P(Y=1 | S=s, A=a) = s for all a.
    That is, a risk score of 0.7 means 70% chance of recidivism
    regardless of the defendant's race.
    """
    groups = sorted(np.unique(sensitive))
    bins = np.linspace(0, 1, n_bins + 1)

    results = []
    for group in groups:
        mask = sensitive == group
        probs = y_prob[mask]
        labels = y_true[mask]

        bin_indices = np.digitize(probs, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        for b in range(n_bins):
            bin_mask = bin_indices == b
            if bin_mask.sum() >= 5:
                predicted = probs[bin_mask].mean()
                actual = labels[bin_mask].mean()
                results.append({
                    "group": f"Race={group}",
                    "bin": f"[{bins[b]:.1f}, {bins[b+1]:.1f})",
                    "predicted_risk": predicted,
                    "actual_rate": actual,
                    "calibration_error": abs(predicted - actual),
                    "n": bin_mask.sum(),
                })

    return pd.DataFrame(results)


cal_df = calibration_analysis(y_test, y_prob, race_test)
print("=== Calibration by Race ===")
print(cal_df.to_string(index=False))

# Summary: is the model calibrated?
for group in cal_df["group"].unique():
    group_cal = cal_df[cal_df["group"] == group]
    mean_cal_error = group_cal["calibration_error"].mean()
    print(f"\n{group}: Mean calibration error = {mean_cal_error:.4f}")
```

### 8.4 Audit Report Generation

```python
"""
Generate a structured fairness audit report.
"""


def generate_audit_report(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    sensitive: np.ndarray,
    sensitive_name: str,
) -> str:
    """Generate a human-readable fairness audit report."""

    suite = FairnessMetricsSuite(y_true, y_pred, y_prob, sensitive)
    summary = suite.summary()

    report_lines = [
        "=" * 60,
        f"  FAIRNESS AUDIT REPORT",
        f"  Model: {model_name}",
        f"  Protected Attribute: {sensitive_name}",
        f"  Test Set Size: {len(y_true)}",
        "=" * 60,
        "",
        "1. OVERALL PERFORMANCE",
        f"   Accuracy: {accuracy_score(y_true, y_pred):.4f}",
        f"   Selection Rate: {y_pred.mean():.4f}",
        "",
        "2. FAIRNESS METRICS",
    ]

    for _, row in summary.iterrows():
        report_lines.append(
            f"   {row['Metric']:30s} {row['Value']:.4f}  "
            f"({row['Threshold']})  [{row['Status']}]"
        )

    # Count violations
    n_violations = (summary["Status"] == "FAIL").sum()
    report_lines.extend([
        "",
        "3. SUMMARY",
        f"   Violations: {n_violations} / {len(summary)}",
    ])

    if n_violations > 0:
        report_lines.append("   STATUS: FAIRNESS CONCERNS DETECTED")
        report_lines.append("")
        report_lines.append("4. RECOMMENDATIONS")
        if suite.disparate_impact() < 0.8:
            report_lines.append(
                "   - Consider pre-processing (reweighing) or in-processing "
                "(constrained optimization) to reduce disparate impact"
            )
        if suite.equal_opportunity_difference() > 0.1:
            report_lines.append(
                "   - Apply threshold optimization per group to equalize "
                "true positive rates"
            )
    else:
        report_lines.append("   STATUS: NO SIGNIFICANT FAIRNESS VIOLATIONS")

    return "\n".join(report_lines)


report = generate_audit_report(
    "RandomForest Recidivism Predictor",
    y_test, y_pred, y_prob, race_test, "race",
)
print(report)
```

---

## Summary

- **Individual fairness** (Dwork et al., 2012) requires that similar individuals
  receive similar predictions. The central challenge is defining "similarity"
  through a task-specific metric that ignores protected attributes while
  capturing legitimate differences.
- **Counterfactual fairness** (Kusner et al., 2017) asks whether a prediction would
  change if the individual's sensitive attribute were different in a counterfactual
  world. It requires a Structural Causal Model to trace causal effects of the
  sensitive attribute through all downstream features.
- **Path-specific fairness** refines counterfactual fairness by distinguishing
  fair causal pathways (legitimate mediators) from unfair ones (discrimination).
- The **impossibility theorem** (Chouldechova, 2017; Kleinberg et al., 2016) proves
  that calibration, equal FPR, and equal FNR cannot simultaneously hold when base
  rates differ between groups. Practitioners must choose which criterion to
  prioritize based on the application context.
- **Intersectional fairness** examines bias at the intersection of multiple protected
  attributes (e.g., race and gender). Single-axis analysis can miss violations
  that only appear at these intersections.
- **Fairlearn** provides MetricFrame for disaggregated metric computation and
  constrained optimization algorithms for bias mitigation.
- **AIF360** offers a broader suite of metrics (including individual fairness
  measures like the Theil index) and multiple mitigation algorithms at the
  pre-processing, in-processing, and post-processing stages.

---

## Exercises

### Exercise 1: Individual Fairness Audit (Beginner)

Using the hiring dataset from Section 2:
1. Implement two different similarity metrics: (a) Euclidean distance excluding
   gender, and (b) Mahalanobis distance trained on a subset without gender
2. Compute individual fairness scores using both metrics
3. Find the 5 most individually unfair predictions (largest prediction differences
   among similar individuals)
4. Explain why the two metrics give different results

### Exercise 2: Counterfactual Fairness Analysis (Intermediate)

Build a complete counterfactual fairness analysis for a loan approval model:
1. Define a causal DAG with at least 5 variables (including gender/race, income,
   education, credit score, loan approval)
2. Implement the three-step counterfactual procedure (abduction, action, prediction)
3. Compute the counterfactual fairness gap for 1000 test instances
4. Identify which individuals are most counterfactually unfair
5. Compare results with standard demographic parity metrics

### Exercise 3: Impossibility Theorem Exploration (Intermediate)

Extend the impossibility theorem demonstration:
1. Create a visualization showing the Pareto frontier between calibration and
   equalized odds for different base rate ratios
2. For base rates p0 = 0.2 and p1 = 0.6, compute exactly how much equalized
   odds must be violated to maintain perfect calibration
3. Implement a "fairness dial" that smoothly interpolates between calibration
   and equalized odds optimization
4. Discuss which setting you would choose for (a) criminal justice, (b) healthcare,
   (c) hiring, with justification

### Exercise 4: Intersectional Audit with Fairlearn (Advanced)

Using the Adult Income dataset (or a synthetic equivalent):
1. Audit a gradient boosting classifier for fairness with respect to race, gender,
   and their intersection
2. Use Fairlearn's MetricFrame with custom metrics to compute all six core metrics
   (selection rate, TPR, FPR, PPV, FNR, accuracy) per intersectional group
3. Identify which intersectional group is most disadvantaged and by which metric
4. Compare the "worst-off group" across different fairness definitions

### Exercise 5: Full COMPAS Audit Report (Advanced)

Extend the COMPAS-style practical to produce a publication-quality audit:
1. Train three different models (logistic regression, random forest, gradient boosting)
2. For each model, compute all fairness metrics (group + individual + calibration)
3. Apply the impossibility theorem analysis: show calibration vs equalized odds tradeoff
4. Perform intersectional analysis with race x age x prior_convictions
5. Write a 2-page "algorithmic impact assessment" summarizing findings, identifying
   the most concerning disparities, and recommending concrete mitigation steps

---

[Previous: Evaluating Explanations](./10_Evaluating_Explanations.md) | [Overview](./00_Overview.md) | [Next: Fairness Mitigation](./12_Fairness_Mitigation.md)

**License**: CC BY-NC 4.0
