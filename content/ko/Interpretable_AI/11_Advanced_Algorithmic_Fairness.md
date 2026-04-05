# 레슨 11: 고급 알고리즘 공정성(Advanced Algorithmic Fairness)

[이전: 설명 평가](./10_Evaluating_Explanations.md) | [다음: 공정성 완화](./12_Fairness_Mitigation.md)

---

## 학습 목표

- 개별 공정성(Individual Fairness, Dwork et al., 2012)을 이해하고 메트릭 학습(Metric Learning)이 "유사한 개인"을 어떻게 정의하는지 파악한다
- 구조적 인과 모형(Structural Causal Models)을 사용하여 반사실적 공정성(Counterfactual Fairness, Kusner et al., 2017)을 형식화하고 경로별 효과(Path-Specific Effects)를 추론한다
- 불가능성 정리(Impossibility Theorem)를 증명하고 해석한다: 퇴화된 경우를 제외하면 보정(Calibration), 위양성률 동등(False Positive Parity), 위음성률 동등(False Negative Parity)을 동시에 만족할 수 없는 이유를 이해한다
- 다중 보호 속성에 걸친 교차적 공정성(Intersectional Fairness)을 분석한다
- Fairlearn과 AIF360 도구킷을 사용하여 ML 모델의 편향을 탐지하고 정량화한다

---

## 1. 그룹 공정성 기초 복습

### 1.1 머신러닝 레슨 16의 기초

이 레슨은 머신러닝 레슨 16(모델 설명가능성)에서 다룬 기본 그룹 공정성 정의에
대한 사전 지식을 전제로 한다. 고급 주제로 넘어가기 전에 세 가지 핵심 지표를
간략히 복습한다.

**인구통계적 동등(Demographic Parity)** (통계적 동등): P(Y_hat = 1 | A = 0) = P(Y_hat = 1 | A = 1).
실제 레이블과 무관하게 예측 비율이 그룹 간에 동일해야 한다.
**균등화된 오즈(Equalized Odds)** (Hardt et al., 2016): P(Y_hat = 1 | A = a, Y = y)가
y = 0과 y = 1 모두에서 모든 그룹 a에 대해 동일하다. 그룹 간 동일한 TPR과 FPR을 의미한다.
**예측 동등(Predictive Parity)** (보정): P(Y = 1 | Y_hat = 1, A = a)가 모든 그룹에서
동일하다. 양성으로 예측된 사람들 중 실제 양성의 비율이 동일하다.

완전한 소개는 [머신러닝 L16](../Machine_Learning/16_Model_Explainability.md)을 참조하라.
이 레슨은 그룹 수준 지표를 넘어 개별, 반사실적, 인과적 공정성 정의로 나아간다.

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

## 2. 개별 공정성(Individual Fairness)

### 2.1 Dwork et al. (2012) 프레임워크

그룹 공정성(Group Fairness)은 인구통계 그룹의 모든 구성원을 동일하게 취급한다.
그러나 어떤 그룹 내에서도 개인들은 서로 다르다. 개별 공정성(Individual Fairness,
Dwork et al., 2012)은 어떤 그룹에 속하는지와 무관하게 **유사한 개인이 유사한
결과를 받아야** 한다고 요구한다.

형식적으로, 분류기 h와 과제 특화 유사도 메트릭(Similarity Metric) d에 대해:

    d_outcome(h(x), h(x')) <= L * d_input(x, x')

이것은 립시츠 조건(Lipschitz Condition)이다: 결과의 차이는 입력의 차이(적절한
메트릭 하에서)에 상수 L을 곱한 값 이하로 제한되어야 한다.

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

### 2.2 유사도를 위한 메트릭 학습(Metric Learning)

개별 공정성에서 가장 어려운 측면은 올바른 유사도 메트릭을 정의하는 것이다.
Ilvento (2020) 등은 인간의 판단이나 도메인 제약 조건으로부터 메트릭을
학습하는 방법을 제안했다.

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

## 3. 반사실적 공정성(Counterfactual Fairness)

### 3.1 Kusner et al. (2017) 정의

반사실적 공정성(Counterfactual Fairness)은 다음과 같은 질문을 한다: "이 개인의
민감한 속성이 달랐던 반사실적 세계에서도 동일한 예측을 받았을 것인가?" 이것은
구조적 인과 모형(Structural Causal Model)을 필요로 하는 인과적 공정성 개념이다.

형식적으로, 예측기 Y_hat이 반사실적으로 공정하려면:

    P(Y_hat_A<-a(U) = y | X = x, A = a) = P(Y_hat_A<-a'(U) = y | X = x, A = a)

즉, 동일한 외생 변수(Exogenous Variables) U가 주어졌을 때, 개인의 민감한 속성
A가 a이든 a'이든(반사실적 세계에서) 예측의 분포가 동일해야 한다.

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

### 3.2 경로 특화 공정성(Path-Specific Fairness)

반사실적 공정성은 전부 아니면 전무의 기준이다: 민감한 속성에서 예측으로의 어떤
인과 경로든 불공정한 것으로 간주된다. 그러나 일부 경로는 정당할 수 있다.

예를 들어, 대학 입학에서:
- 성별 -> 시험 점수 -> 입학: 잠재적으로 불공정(시험이 편향된 경우)
- 성별 -> 전공 선택 -> 입학: 가능하게 정당(자유로운 선택)

경로 특화 공정성(Path-Specific Fairness, Nabi & Shpitser, 2018)은 어떤 인과
경로가 공정하고 어떤 것이 불공정한지 지정할 수 있게 한다.

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

## 4. 불가능성 정리(Impossibility Theorem)

### 4.1 Chouldechova (2017)와 Kleinberg et al. (2016)

알고리즘 공정성에서 가장 중요한 결과 중 하나는 **불가능성 정리(Impossibility
Theorem)**이다: 퇴화된 경우를 제외하면, 분류기가 보정(Calibration, 예측
동등), 양성 클래스에 대한 균형(동일 FNR), 음성 클래스에 대한 균형(동일 FPR)을
그룹 간에 동시에 만족할 수 없다.

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

### 4.2 형식적 증명 스케치(Formal Proof Sketch)

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

### 4.3 실무적 함의

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

## 5. 교차적 공정성(Intersectional Fairness)

### 5.1 단일 축 분석의 한계

표준 공정성 분석은 한 번에 하나의 보호 속성만 고려한다(성별 또는 인종). 그러나
차별은 종종 다중 정체성의 교차점에서 작동한다(예: 흑인 여성은 흑인 남성이나
백인 여성과는 다른 도전에 직면한다). Crenshaw (1989)는 이 현상에 대해
"교차성(Intersectionality)"이라는 용어를 만들었다.

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

## 6. Fairlearn 도구킷

### 6.1 MetricFrame: 세분화된 지표

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

### 6.2 Fairlearn 대시보드 개념

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

## 7. AIF360 도구킷

### 7.1 AIF360을 이용한 편향 탐지

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

## 8. 실습: COMPAS 스타일 재범 모델 감사

### 8.1 데이터 생성 및 모델 학습

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

### 8.2 학습 및 종합 감사

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

### 8.3 보정 분석(Calibration Analysis)

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

### 8.4 감사 보고서 생성

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

## 요약

- **개별 공정성(Individual Fairness)** (Dwork et al., 2012)은 유사한 개인이
  유사한 예측을 받을 것을 요구한다. 핵심 과제는 보호 속성을 무시하면서 정당한
  차이를 포착하는 과제 특화 메트릭을 통해 "유사성"을 정의하는 것이다.
- **반사실적 공정성(Counterfactual Fairness)** (Kusner et al., 2017)은 반사실적
  세계에서 개인의 민감한 속성이 달랐을 경우 예측이 변경되었을 것인지 묻는다.
  민감한 속성의 인과적 효과를 모든 하위 특성을 통해 추적하기 위해 구조적 인과
  모형(Structural Causal Model)이 필요하다.
- **경로 특화 공정성(Path-Specific Fairness)**은 공정한 인과 경로(정당한
  매개자)와 불공정한 경로(차별)를 구분하여 반사실적 공정성을 정교화한다.
- **불가능성 정리(Impossibility Theorem)** (Chouldechova, 2017; Kleinberg et al.,
  2016)는 그룹 간 기저율(Base Rates)이 다를 때 보정(Calibration), 동일 FPR,
  동일 FNR이 동시에 성립할 수 없음을 증명한다. 실무자는 적용 맥락에 따라 어떤
  기준을 우선시할지 선택해야 한다.
- **교차적 공정성(Intersectional Fairness)**은 다중 보호 속성의 교차점(예: 인종과
  성별)에서의 편향을 검토한다. 단일 축 분석으로는 이러한 교차점에서만 나타나는
  위반을 놓칠 수 있다.
- **Fairlearn**은 세분화된 지표 계산을 위한 MetricFrame과 편향 완화를 위한
  제약 최적화 알고리즘을 제공한다.
- **AIF360**은 개별 공정성 지표(Theil 지수 등)를 포함한 더 넓은 범위의 지표
  모음과 전처리(Pre-processing), 인처리(In-processing), 후처리(Post-processing)
  단계의 다양한 완화 알고리즘을 제공한다.

---

## 연습문제

### 연습문제 1: 개별 공정성 감사 (초급)

섹션 2의 채용 데이터셋을 사용하여:
1. 두 가지 다른 유사도 메트릭을 구현하라: (a) 성별을 제외한 유클리드 거리,
   (b) 성별 없이 훈련된 마할라노비스 거리
2. 두 메트릭을 사용하여 개별 공정성 점수를 계산하라
3. 가장 개별적으로 불공정한 예측 5개(유사한 개인 간 가장 큰 예측 차이)를
   찾아라
4. 두 메트릭이 다른 결과를 주는 이유를 설명하라

### 연습문제 2: 반사실적 공정성 분석 (중급)

대출 승인 모델에 대한 완전한 반사실적 공정성 분석을 구축하라:
1. 최소 5개의 변수(성별/인종, 소득, 교육, 신용 점수, 대출 승인 포함)를 포함하는
   인과 DAG를 정의하라
2. 3단계 반사실적 절차(귀추, 행동, 예측)를 구현하라
3. 1000개의 테스트 인스턴스에 대해 반사실적 공정성 격차를 계산하라
4. 가장 반사실적으로 불공정한 개인을 식별하라
5. 결과를 표준 인구통계적 동등 지표와 비교하라

### 연습문제 3: 불가능성 정리 탐구 (중급)

불가능성 정리 시연을 확장하라:
1. 다양한 기저율 비율에 대해 보정과 균등화된 오즈 간의 파레토 프론티어를
   보여주는 시각화를 만들어라
2. 기저율 p0 = 0.2와 p1 = 0.6에서 완벽한 보정을 유지하기 위해 균등화된
   오즈가 정확히 얼마나 위반되어야 하는지 계산하라
3. 보정과 균등화된 오즈 최적화 사이를 부드럽게 보간하는 "공정성 다이얼"을
   구현하라
4. (a) 형사 사법, (b) 의료, (c) 채용에 대해 어떤 설정을 선택할지 정당화와
   함께 논의하라

### 연습문제 4: Fairlearn을 이용한 교차적 감사 (고급)

Adult Income 데이터셋(또는 합성 대체물)을 사용하여:
1. 인종, 성별, 그리고 그 교차점에 대해 그래디언트 부스팅 분류기의 공정성을
   감사하라
2. Fairlearn의 MetricFrame과 사용자 정의 지표를 사용하여 교차적 그룹별
   모든 6개 핵심 지표(선택률, TPR, FPR, PPV, FNR, 정확도)를 계산하라
3. 어떤 교차적 그룹이 가장 불리한지, 어떤 지표로 그런지 식별하라
4. 다양한 공정성 정의에 걸쳐 "가장 불리한 그룹"을 비교하라

### 연습문제 5: 완전한 COMPAS 감사 보고서 (고급)

COMPAS 스타일 실습을 확장하여 출판 수준의 감사를 작성하라:
1. 세 가지 다른 모델(로지스틱 회귀, 랜덤 포레스트, 그래디언트 부스팅)을 훈련하라
2. 각 모델에 대해 모든 공정성 지표(그룹 + 개별 + 보정)를 계산하라
3. 불가능성 정리 분석을 적용하라: 보정 대 균등화된 오즈 트레이드오프를 보여주라
4. 인종 x 연령 x 전과 횟수에 대한 교차적 분석을 수행하라
5. 발견 사항을 요약하고 가장 우려되는 격차를 식별하며 구체적인 완화 조치를
   권장하는 2페이지 "알고리즘 영향 평가"를 작성하라

---

[이전: 설명 평가](./10_Evaluating_Explanations.md) | [개요](./00_Overview.md) | [다음: 공정성 완화](./12_Fairness_Mitigation.md)

**License**: CC BY-NC 4.0
