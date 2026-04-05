# Lesson 12: Fairness Mitigation

[Previous: Advanced Algorithmic Fairness](./11_Advanced_Algorithmic_Fairness.md) | [Next: AI Regulation and Governance](./13_AI_Regulation_and_Governance.md)

---

## Learning Objectives

- Implement pre-processing mitigation strategies (reweighing, disparate impact remover, learning fair representations) that modify training data before model training
- Apply in-processing methods (adversarial debiasing, exponentiated gradient, constrained optimization) that incorporate fairness constraints during training
- Use post-processing techniques (threshold optimization, equalized odds calibration, reject option) that adjust model outputs after training
- Analyze the accuracy-fairness Pareto frontier and make principled tradeoffs using multi-objective optimization
- Choose the right mitigation strategy based on context using a decision framework, and detect proxy discrimination

---

## 1. Overview of Mitigation Strategies

### 1.1 The Three Intervention Points

Fairness mitigation can intervene at three stages of the ML pipeline. Each stage
has different advantages, constraints, and assumptions.

```python
"""
Overview of the three intervention points for fairness mitigation.

Pre-processing:  Modify the TRAINING DATA to remove bias before training.
                 Advantage: model-agnostic, any downstream model benefits.
                 Limitation: may lose information; cannot address model bias.

In-processing:   Modify the TRAINING ALGORITHM to incorporate fairness
                 constraints directly during optimization.
                 Advantage: directly optimizes fairness-accuracy tradeoff.
                 Limitation: model-specific; may require custom training loops.

Post-processing: Modify the MODEL OUTPUT to satisfy fairness constraints.
                 Advantage: does not require retraining; works with any model.
                 Limitation: does not fix root cause; may reduce overall accuracy.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from typing import Dict, List, Tuple, Optional


def generate_credit_scoring_data(n: int = 10000) -> pd.DataFrame:
    """Generate synthetic credit scoring data with known biases.

    This dataset models a credit scoring scenario where:
    - Gender affects income through labor market discrimination
    - Race affects neighborhood (and thus collateral value) through
      historical segregation
    - Both legitimate factors (income, credit history) and biased factors
      (gender, race proxies) predict default

    We use this dataset throughout this lesson to compare all three
    mitigation approaches.
    """
    np.random.seed(42)

    # Protected attributes
    gender = np.random.binomial(1, 0.5, n)  # 0: female, 1: male
    race = np.random.binomial(1, 0.4, n)    # 0: majority, 1: minority

    # Legitimate features (some affected by protected attributes)
    # Income: gender pay gap and racial wage gap
    income = (
        40000
        + 10000 * gender        # Gender pay gap (unfair)
        - 5000 * race           # Racial wage gap (unfair)
        + np.random.normal(0, 15000, n)
    )
    income = np.clip(income, 10000, 200000)

    # Credit history: slightly correlated with income
    credit_history = (
        5.0
        + 0.00003 * income
        + np.random.normal(0, 1.5, n)
    )
    credit_history = np.clip(credit_history, 0, 10)

    # Employment years
    employment_years = np.random.exponential(5, n) + 1
    employment_years = np.clip(employment_years, 0, 30)

    # Debt-to-income ratio
    dti = np.random.beta(2, 5, n) * 0.8
    dti = np.clip(dti, 0.01, 0.8)

    # Collateral value (affected by race through neighborhood)
    collateral = (
        50000
        - 15000 * race          # Neighborhood value gap (unfair)
        + 0.5 * income
        + np.random.normal(0, 20000, n)
    )
    collateral = np.clip(collateral, 0, 500000)

    # True default: depends on legitimate factors
    # But the training data may also encode historical bias
    logit = (
        1.5
        - 0.00003 * income
        - 0.15 * credit_history
        - 0.05 * employment_years
        + 2.0 * dti
        - 0.000005 * collateral
        + 0.2 * race     # Historical bias in default labels
    )
    prob_default = 1 / (1 + np.exp(-logit))
    default = np.random.binomial(1, prob_default, n)

    return pd.DataFrame({
        "gender": gender,
        "race": race,
        "income": income,
        "credit_history": credit_history,
        "employment_years": employment_years,
        "dti": dti,
        "collateral": collateral,
        "default": default,
    })


# Generate and split data
credit_data = generate_credit_scoring_data(10000)
feature_cols = ["income", "credit_history", "employment_years", "dti", "collateral"]

X = credit_data[feature_cols].values
y = credit_data["default"].values
gender = credit_data["gender"].values
race = credit_data["race"].values

X_train, X_test, y_train, y_test, race_train, race_test, gender_train, gender_test = \
    train_test_split(X, y, race, gender, test_size=0.3, random_state=42)

# Baseline model (no fairness intervention)
baseline_model = GradientBoostingClassifier(
    n_estimators=100, max_depth=4, random_state=42
)
baseline_model.fit(X_train, y_train)
baseline_pred = baseline_model.predict(X_test)

print("=== Baseline Model (No Mitigation) ===")
print(f"Accuracy: {accuracy_score(y_test, baseline_pred):.4f}")
print(f"\nSelection rates by race:")
for r in [0, 1]:
    mask = race_test == r
    sr = 1 - baseline_pred[mask].mean()  # Approval rate (non-default prediction)
    print(f"  Race={r}: approval rate = {sr:.4f}")

dp_diff = abs(
    (1 - baseline_pred[race_test == 0]).mean() -
    (1 - baseline_pred[race_test == 1]).mean()
)
print(f"\nDemographic Parity Difference: {dp_diff:.4f}")
```

---

## 2. Pre-Processing Methods

### 2.1 Reweighing (Kamiran & Calders, 2012)

Reweighing assigns different weights to training instances to remove the
association between the sensitive attribute and the label in the weighted data.

```python
"""
Reweighing: assign instance weights to achieve statistical parity in
the training data.

The idea: if group A=0 with Y=1 is underrepresented, give those
instances higher weight. If group A=1 with Y=0 is underrepresented,
give those higher weight too. The result is a weighted dataset where
the sensitive attribute and label are statistically independent.

Weight formula for instance (a, y):
  w(a, y) = P(A=a) * P(Y=y) / P(A=a, Y=y)

This makes the weighted joint distribution P_w(A, Y) = P(A) * P(Y),
which means A and Y are independent in the weighted data.
"""


def compute_reweighing_weights(
    sensitive: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Compute instance weights for reweighing.

    For each combination of (sensitive_value, label_value), compute:
      weight = P(A=a) * P(Y=y) / P(A=a, Y=y)

    This ensures that in the weighted data, the sensitive attribute
    and label are statistically independent, removing any existing
    association (whether causal or spurious).

    Parameters
    ----------
    sensitive : binary sensitive attribute array
    labels : binary label array

    Returns
    -------
    weights : array of instance weights (same length as inputs)
    """
    n = len(sensitive)
    weights = np.ones(n)

    # Marginal probabilities
    p_a = {}
    p_y = {}
    p_ay = {}

    for a in np.unique(sensitive):
        p_a[a] = (sensitive == a).mean()

    for y_val in np.unique(labels):
        p_y[y_val] = (labels == y_val).mean()

    for a in np.unique(sensitive):
        for y_val in np.unique(labels):
            mask = (sensitive == a) & (labels == y_val)
            p_ay[(a, y_val)] = mask.mean()

    # Compute weights
    for i in range(n):
        a = sensitive[i]
        y_val = labels[i]
        # Weight = P(A) * P(Y) / P(A, Y)
        # This makes the weighted distribution satisfy A _|_ Y
        if p_ay[(a, y_val)] > 0:
            weights[i] = (p_a[a] * p_y[y_val]) / p_ay[(a, y_val)]
        else:
            weights[i] = 1.0

    # Normalize so weights sum to n (preserves effective sample size)
    weights = weights * n / weights.sum()

    return weights


# Apply reweighing
rw_weights = compute_reweighing_weights(race_train, y_train)

print("=== Reweighing Weights ===")
for a in [0, 1]:
    for y_val in [0, 1]:
        mask = (race_train == a) & (y_train == y_val)
        mean_weight = rw_weights[mask].mean()
        count = mask.sum()
        print(f"  Race={a}, Default={y_val}: "
              f"n={count:4d}, mean_weight={mean_weight:.4f}")

# Train model with reweighed data
rw_model = GradientBoostingClassifier(
    n_estimators=100, max_depth=4, random_state=42
)
rw_model.fit(X_train, y_train, sample_weight=rw_weights)
rw_pred = rw_model.predict(X_test)

print(f"\n=== Reweighed Model ===")
print(f"Accuracy: {accuracy_score(y_test, rw_pred):.4f}")
for r in [0, 1]:
    mask = race_test == r
    sr = 1 - rw_pred[mask].mean()
    print(f"  Race={r}: approval rate = {sr:.4f}")

rw_dp = abs(
    (1 - rw_pred[race_test == 0]).mean() -
    (1 - rw_pred[race_test == 1]).mean()
)
print(f"Demographic Parity Difference: {rw_dp:.4f}")
```

### 2.2 Disparate Impact Remover (Feldman et al., 2015)

The Disparate Impact Remover modifies feature distributions so that each
feature has the same distribution across all groups, while preserving the
within-group ranking.

```python
"""
Disparate Impact Remover: modify features to remove group-specific information.

For each feature, the distribution is adjusted so that it looks the same
for all groups. The key insight is to use quantile matching: preserve each
individual's rank within their group, but map it to the median distribution.

This is a form of data preprocessing that removes the statistical
dependency between features and the sensitive attribute.
"""

from scipy.stats import rankdata


def disparate_impact_remover(
    X: np.ndarray,
    sensitive: np.ndarray,
    repair_level: float = 1.0,
) -> np.ndarray:
    """Remove disparate impact from features using quantile matching.

    For each feature:
    1. Compute the CDF within each group
    2. Compute the overall (median) CDF
    3. Map each value to where it would fall in the median distribution

    The repair_level parameter (0 to 1) controls how much repair to do:
    - 0.0: no repair (original data)
    - 1.0: full repair (features are group-invariant)
    - 0.5: partial repair (interpolation between original and repaired)

    Parameters
    ----------
    X : feature matrix (n_samples, n_features)
    sensitive : binary sensitive attribute
    repair_level : interpolation between original (0) and repaired (1) data

    Returns
    -------
    X_repaired : repaired feature matrix
    """
    X_repaired = X.copy().astype(float)
    groups = np.unique(sensitive)
    n = len(X)

    for feat_idx in range(X.shape[1]):
        # Step 1: For each group, compute within-group quantiles
        quantiles_by_group = {}
        for g in groups:
            mask = sensitive == g
            values = X[mask, feat_idx]
            # Rank within group (percentile rank)
            ranks = rankdata(values, method="average") / len(values)
            quantiles_by_group[g] = (mask, values, ranks)

        # Step 2: Compute the "median" distribution
        # Use all values sorted to create the reference distribution
        all_values_sorted = np.sort(X[:, feat_idx])

        # Step 3: For each individual, find their within-group quantile
        # and map it to the corresponding value in the median distribution
        for g in groups:
            mask, original_values, ranks = quantiles_by_group[g]
            indices = mask.nonzero()[0]

            for i, idx in enumerate(indices):
                quantile = ranks[i]
                # Map quantile to median distribution value
                median_idx = int(np.clip(quantile * n - 1, 0, n - 1))
                repaired_value = all_values_sorted[median_idx]

                # Interpolate between original and repaired
                X_repaired[idx, feat_idx] = (
                    (1 - repair_level) * original_values[i]
                    + repair_level * repaired_value
                )

    return X_repaired


# Apply disparate impact removal
X_train_repaired = disparate_impact_remover(X_train, race_train, repair_level=0.8)
X_test_repaired = disparate_impact_remover(X_test, race_test, repair_level=0.8)

# Train model on repaired data
dir_model = GradientBoostingClassifier(
    n_estimators=100, max_depth=4, random_state=42
)
dir_model.fit(X_train_repaired, y_train)
dir_pred = dir_model.predict(X_test_repaired)

print("=== Disparate Impact Remover (repair=0.8) ===")
print(f"Accuracy: {accuracy_score(y_test, dir_pred):.4f}")
for r in [0, 1]:
    mask = race_test == r
    sr = 1 - dir_pred[mask].mean()
    print(f"  Race={r}: approval rate = {sr:.4f}")

dir_dp = abs(
    (1 - dir_pred[race_test == 0]).mean() -
    (1 - dir_pred[race_test == 1]).mean()
)
print(f"Demographic Parity Difference: {dir_dp:.4f}")
```

### 2.3 Learning Fair Representations (Zemel et al., 2013)

A more sophisticated approach: learn a new representation of the data that
is explicitly designed to be uninformative about the sensitive attribute
while preserving information about the target.

```python
"""
Learning Fair Representations (Zemel et al., 2013).

The idea: learn a mapping Z = f(X) such that:
1. Z is informative about Y (preserves prediction accuracy)
2. Z is uninformative about A (removes sensitive information)
3. Z preserves individual-level information (similar individuals
   map to similar representations)

This is implemented as an optimization problem with three terms:
  minimize  L_prediction + alpha * L_fairness + beta * L_distortion

where:
  L_prediction: classification loss using Z
  L_fairness: statistical distance between P(Z|A=0) and P(Z|A=1)
  L_distortion: information loss from the mapping X -> Z
"""

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def learn_fair_representation_simple(
    X: np.ndarray,
    sensitive: np.ndarray,
    n_components: int = 5,
    fairness_weight: float = 1.0,
) -> Tuple[np.ndarray, object]:
    """Learn fair representations using an adversarial-style approach.

    This simplified version:
    1. Standardizes the data
    2. Projects out the direction most correlated with the sensitive attribute
    3. Applies PCA on the residual

    The fairness_weight controls how aggressively we remove sensitive
    information (0 = no removal, higher = more removal).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Find the direction most correlated with sensitive attribute
    # This is essentially what a linear model predicting A from X would learn
    from sklearn.linear_model import Ridge
    sensitive_predictor = Ridge(alpha=1.0)
    sensitive_predictor.fit(X_scaled, sensitive)

    # The model's coefficients define the "sensitive direction"
    sensitive_direction = sensitive_predictor.coef_
    sensitive_direction = sensitive_direction / np.linalg.norm(sensitive_direction)

    # Project out the sensitive direction (scaled by fairness_weight)
    projections = X_scaled @ sensitive_direction.reshape(-1, 1)
    X_fair = X_scaled - fairness_weight * projections @ sensitive_direction.reshape(1, -1)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=min(n_components, X_fair.shape[1]))
    X_transformed = pca.fit_transform(X_fair)

    print(f"Fair representation: {X.shape[1]} features -> {X_transformed.shape[1]} components")
    print(f"Variance retained: {pca.explained_variance_ratio_.sum():.4f}")

    # Verify: can we still predict A from the fair representation?
    from sklearn.linear_model import LogisticRegression
    a_predictor = LogisticRegression(random_state=42)
    a_predictor.fit(X_transformed, sensitive)
    a_pred_accuracy = a_predictor.score(X_transformed, sensitive)
    print(f"Sensitive attr predictability: {a_pred_accuracy:.4f} "
          f"(baseline: {max(sensitive.mean(), 1-sensitive.mean()):.4f})")

    return X_transformed, (scaler, sensitive_direction, fairness_weight, pca)


def transform_fair(X_new, transform_params):
    """Apply learned fair transformation to new data."""
    scaler, sensitive_direction, fairness_weight, pca = transform_params
    X_scaled = scaler.transform(X_new)
    projections = X_scaled @ sensitive_direction.reshape(-1, 1)
    X_fair = X_scaled - fairness_weight * projections @ sensitive_direction.reshape(1, -1)
    return pca.transform(X_fair)


# Learn fair representation
X_train_fair, transform_params = learn_fair_representation_simple(
    X_train, race_train, n_components=5, fairness_weight=1.5
)
X_test_fair = transform_fair(X_test, transform_params)

# Train on fair representation
fr_model = GradientBoostingClassifier(
    n_estimators=100, max_depth=4, random_state=42
)
fr_model.fit(X_train_fair, y_train)
fr_pred = fr_model.predict(X_test_fair)

print(f"\n=== Fair Representation Model ===")
print(f"Accuracy: {accuracy_score(y_test, fr_pred):.4f}")
fr_dp = abs(
    (1 - fr_pred[race_test == 0]).mean() -
    (1 - fr_pred[race_test == 1]).mean()
)
print(f"Demographic Parity Difference: {fr_dp:.4f}")
```

---

## 3. In-Processing Methods

### 3.1 Adversarial Debiasing (Zhang et al., 2018)

Adversarial debiasing trains a classifier and an adversary simultaneously.
The classifier tries to predict the label, while the adversary tries to predict
the sensitive attribute from the classifier's output. The classifier is
penalized when the adversary succeeds.

```python
"""
Adversarial Debiasing: a minimax approach to fair classification.

Architecture:
  Predictor (P): X -> Y_hat (predict the label)
  Adversary (A): Y_hat -> A_hat (predict sensitive attribute from prediction)

Training:
  1. Update Predictor to minimize: L_pred(Y, Y_hat) - lambda * L_adv(A, A_hat)
     (The negative sign means P tries to FOOL the adversary)
  2. Update Adversary to minimize: L_adv(A, A_hat)
     (A tries to correctly predict the sensitive attribute)

At equilibrium, the predictor makes accurate predictions that reveal
no information about the sensitive attribute to the adversary.

Note: this is conceptually similar to GANs, but for fairness rather
than data generation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class Predictor(nn.Module):
    """The main classifier that predicts the target label."""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)


class Adversary(nn.Module):
    """The adversary that tries to predict the sensitive attribute
    from the predictor's output.

    If the adversary can predict A from Y_hat, then Y_hat contains
    information about A, which means the predictor is unfair.
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, pred):
        return self.network(pred)


def train_adversarial_debiasing(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sensitive_train: np.ndarray,
    adversary_weight: float = 1.0,
    n_epochs: int = 100,
    batch_size: int = 256,
    lr_predictor: float = 0.001,
    lr_adversary: float = 0.001,
) -> Tuple[Predictor, list]:
    """Train a fair classifier using adversarial debiasing.

    Parameters
    ----------
    adversary_weight : lambda parameter controlling fairness-accuracy tradeoff.
        Higher values prioritize fairness (make it harder for the adversary
        to predict the sensitive attribute from the predictions).
        Lower values prioritize accuracy.

    Returns
    -------
    predictor : trained fair predictor
    history : training history (loss values per epoch)
    """
    # Prepare PyTorch data
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    a_tensor = torch.FloatTensor(sensitive_train).unsqueeze(1)

    dataset = TensorDataset(X_tensor, y_tensor, a_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize models
    predictor = Predictor(X_train.shape[1])
    adversary = Adversary()

    # Separate optimizers for predictor and adversary
    # The predictor wants to minimize prediction loss and MAXIMIZE adversary loss
    # The adversary wants to MINIMIZE its own loss
    opt_pred = optim.Adam(predictor.parameters(), lr=lr_predictor)
    opt_adv = optim.Adam(adversary.parameters(), lr=lr_adversary)

    criterion = nn.BCELoss()
    history = []

    for epoch in range(n_epochs):
        epoch_pred_loss = 0
        epoch_adv_loss = 0
        n_batches = 0

        for X_batch, y_batch, a_batch in loader:
            # --- Step 1: Update Adversary ---
            # The adversary learns to predict A from Y_hat
            predictor.eval()
            adversary.train()

            with torch.no_grad():
                y_pred = predictor(X_batch)

            a_pred = adversary(y_pred.detach())
            adv_loss = criterion(a_pred, a_batch)

            opt_adv.zero_grad()
            adv_loss.backward()
            opt_adv.step()

            # --- Step 2: Update Predictor ---
            # The predictor minimizes prediction loss MINUS adversary success
            predictor.train()
            adversary.eval()

            y_pred = predictor(X_batch)
            pred_loss = criterion(y_pred, y_batch)

            # Adversary's prediction using CURRENT predictor output
            a_pred = adversary(y_pred)
            adv_loss_for_pred = criterion(a_pred, a_batch)

            # Total predictor loss: prediction loss - lambda * adversary loss
            # By SUBTRACTING adversary loss, the predictor learns to make
            # predictions that are HARD for the adversary to use
            total_pred_loss = pred_loss - adversary_weight * adv_loss_for_pred

            opt_pred.zero_grad()
            total_pred_loss.backward()
            opt_pred.step()

            epoch_pred_loss += pred_loss.item()
            epoch_adv_loss += adv_loss.item()
            n_batches += 1

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}: pred_loss={epoch_pred_loss/n_batches:.4f}, "
                  f"adv_loss={epoch_adv_loss/n_batches:.4f}")

        history.append({
            "epoch": epoch,
            "pred_loss": epoch_pred_loss / n_batches,
            "adv_loss": epoch_adv_loss / n_batches,
        })

    return predictor, history


# --- Train adversarial debiasing model ---
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

adv_predictor, adv_history = train_adversarial_debiasing(
    X_train_scaled, y_train, race_train,
    adversary_weight=2.0,
    n_epochs=80,
    batch_size=256,
)

# Evaluate
adv_predictor.eval()
with torch.no_grad():
    adv_prob = adv_predictor(torch.FloatTensor(X_test_scaled)).numpy().ravel()
adv_pred = (adv_prob > 0.5).astype(int)

print(f"\n=== Adversarial Debiasing ===")
print(f"Accuracy: {accuracy_score(y_test, adv_pred):.4f}")
adv_dp = abs(
    (1 - adv_pred[race_test == 0]).mean() -
    (1 - adv_pred[race_test == 1]).mean()
)
print(f"Demographic Parity Difference: {adv_dp:.4f}")
```

### 3.2 Exponentiated Gradient (Agarwal et al., 2018)

The exponentiated gradient method solves the fair classification problem as
a constrained optimization: minimize loss subject to fairness constraints.

```python
"""
Exponentiated Gradient Reduction (Agarwal et al., 2018).

This method reduces the fair classification problem to a sequence of
standard classification problems. It uses the exponentiated gradient
algorithm to find the optimal Lagrange multipliers for the fairness
constraints.

Available in Fairlearn as ExponentiatedGradient.
"""

from fairlearn.reductions import (
    ExponentiatedGradient,
    DemographicParity,
    EqualizedOdds,
    TruePositiveRateParity,
    ErrorRateParity,
)


def exponentiated_gradient_fair_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sensitive_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sensitive_test: np.ndarray,
    constraint: str = "demographic_parity",
    base_estimator=None,
) -> dict:
    """Train a fair model using Fairlearn's ExponentiatedGradient.

    The ExponentiatedGradient solver iteratively:
    1. Fits a classifier on a weighted version of the training data
    2. Checks if fairness constraints are satisfied
    3. Updates the weights to move toward constraint satisfaction
    4. Repeats until convergence

    The result is a randomized classifier: a distribution over
    base classifiers that together satisfy the fairness constraint.

    Parameters
    ----------
    constraint : one of "demographic_parity", "equalized_odds",
                 "true_positive_rate_parity", "error_rate_parity"
    base_estimator : sklearn classifier (default: LogisticRegression)
    """
    if base_estimator is None:
        base_estimator = LogisticRegression(
            solver="lbfgs", max_iter=1000, random_state=42
        )

    # Map constraint name to Fairlearn constraint object
    constraint_map = {
        "demographic_parity": DemographicParity(),
        "equalized_odds": EqualizedOdds(),
        "true_positive_rate_parity": TruePositiveRateParity(),
        "error_rate_parity": ErrorRateParity(),
    }

    fairness_constraint = constraint_map[constraint]

    # Train with ExponentiatedGradient
    mitigator = ExponentiatedGradient(
        estimator=base_estimator,
        constraints=fairness_constraint,
    )
    mitigator.fit(X_train, y_train, sensitive_features=sensitive_train)

    # Predict
    y_pred = mitigator.predict(X_test)

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    dp_diff = abs(
        y_pred[sensitive_test == 0].mean() - y_pred[sensitive_test == 1].mean()
    )

    # TPR per group
    tprs = {}
    fprs = {}
    for g in [0, 1]:
        mask = sensitive_test == g
        tp = ((y_test[mask] == 1) & (y_pred[mask] == 1)).sum()
        fn = ((y_test[mask] == 1) & (y_pred[mask] == 0)).sum()
        fp = ((y_test[mask] == 0) & (y_pred[mask] == 1)).sum()
        tn = ((y_test[mask] == 0) & (y_pred[mask] == 0)).sum()
        tprs[g] = tp / (tp + fn) if (tp + fn) > 0 else 0
        fprs[g] = fp / (fp + tn) if (fp + tn) > 0 else 0

    return {
        "constraint": constraint,
        "accuracy": accuracy,
        "dp_diff": dp_diff,
        "tpr_diff": abs(tprs[0] - tprs[1]),
        "fpr_diff": abs(fprs[0] - fprs[1]),
        "model": mitigator,
    }


# Train models with different fairness constraints
print("=== Exponentiated Gradient Results ===\n")
eg_results = {}

for constraint in ["demographic_parity", "equalized_odds"]:
    result = exponentiated_gradient_fair_model(
        X_train, y_train, race_train,
        X_test, y_test, race_test,
        constraint=constraint,
    )
    eg_results[constraint] = result
    print(f"Constraint: {constraint}")
    print(f"  Accuracy:          {result['accuracy']:.4f}")
    print(f"  DP Difference:     {result['dp_diff']:.4f}")
    print(f"  TPR Difference:    {result['tpr_diff']:.4f}")
    print(f"  FPR Difference:    {result['fpr_diff']:.4f}")
    print()
```

### 3.3 Constrained Optimization Framework

```python
"""
General constrained optimization for fair classification.

The mathematical formulation:

  minimize    L(theta)                    (prediction loss)
  subject to  |g_a(theta)| <= epsilon     (fairness constraint)

where g_a measures the fairness violation (e.g., difference in
selection rates between groups).

This can be solved with Lagrangian relaxation:
  minimize  L(theta) + lambda * |g_a(theta)|

where lambda is the Lagrange multiplier that trades off accuracy
and fairness. Higher lambda = more emphasis on fairness.
"""


def constrained_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sensitive_train: np.ndarray,
    fairness_penalty: float = 1.0,
    n_iterations: int = 1000,
    learning_rate: float = 0.01,
) -> np.ndarray:
    """Train a logistic regression with a demographic parity penalty.

    Loss = Cross-Entropy + lambda * |DP_difference|

    where DP_difference = mean(sigmoid(Xw))_A=0 - mean(sigmoid(Xw))_A=1

    This is a simplified implementation for educational purposes.
    Production code should use Fairlearn's ExponentiatedGradient.
    """
    n_features = X_train.shape[1]
    # Initialize weights
    weights = np.zeros(n_features + 1)  # +1 for bias

    # Add bias column
    X_aug = np.column_stack([X_train, np.ones(len(X_train))])

    for iteration in range(n_iterations):
        # Forward pass
        logits = X_aug @ weights
        probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))

        # Prediction loss (cross-entropy gradient)
        pred_grad = X_aug.T @ (probs - y_train) / len(y_train)

        # Fairness penalty gradient
        # DP = mean(probs | A=0) - mean(probs | A=1)
        mask_0 = sensitive_train == 0
        mask_1 = sensitive_train == 1
        dp = probs[mask_0].mean() - probs[mask_1].mean()

        # Gradient of DP w.r.t. weights (through sigmoid)
        dp_grad_0 = (X_aug[mask_0].T @ (probs[mask_0] * (1 - probs[mask_0]))) / mask_0.sum()
        dp_grad_1 = (X_aug[mask_1].T @ (probs[mask_1] * (1 - probs[mask_1]))) / mask_1.sum()
        fairness_grad = np.sign(dp) * (dp_grad_0 - dp_grad_1)

        # Combined gradient
        total_grad = pred_grad + fairness_penalty * fairness_grad

        # Update
        weights -= learning_rate * total_grad

        if (iteration + 1) % 200 == 0:
            loss = -np.mean(
                y_train * np.log(probs + 1e-10) +
                (1 - y_train) * np.log(1 - probs + 1e-10)
            )
            print(f"  Iter {iteration+1:4d}: loss={loss:.4f}, DP={abs(dp):.4f}")

    return weights


# Train constrained model
print("=== Constrained Logistic Regression ===")
from sklearn.preprocessing import StandardScaler

scaler_constrained = StandardScaler()
X_train_sc = scaler_constrained.fit_transform(X_train)
X_test_sc = scaler_constrained.transform(X_test)

clr_weights = constrained_logistic_regression(
    X_train_sc, y_train, race_train,
    fairness_penalty=5.0,
    n_iterations=1000,
    learning_rate=0.01,
)

# Predict
X_test_aug = np.column_stack([X_test_sc, np.ones(len(X_test))])
clr_prob = 1 / (1 + np.exp(-X_test_aug @ clr_weights))
clr_pred = (clr_prob > 0.5).astype(int)

print(f"\nAccuracy: {accuracy_score(y_test, clr_pred):.4f}")
clr_dp = abs(clr_pred[race_test == 0].mean() - clr_pred[race_test == 1].mean())
print(f"DP Difference: {clr_dp:.4f}")
```

---

## 4. Post-Processing Methods

### 4.1 Threshold Optimization Per Group

The simplest post-processing approach: use different decision thresholds for
each group to equalize the desired metric (selection rate, TPR, FPR).

```python
"""
Threshold optimization: choose different classification thresholds
for each group to satisfy fairness constraints.

This is the most common post-processing approach and works with ANY
model that outputs probabilities.

The key insight: instead of using threshold = 0.5 for everyone,
find thresholds t_0 and t_1 such that:
  P(Y_hat=1 | A=0, threshold=t_0) = P(Y_hat=1 | A=1, threshold=t_1)
"""


def optimize_thresholds_dp(
    y_prob: np.ndarray,
    sensitive: np.ndarray,
    target_rate: float = None,
    n_thresholds: int = 1000,
) -> Dict[int, float]:
    """Find per-group thresholds that achieve demographic parity.

    For each group, find the threshold that gives a selection rate
    closest to the target_rate. If no target is specified, use the
    overall selection rate as the target.

    Parameters
    ----------
    y_prob : predicted probabilities
    sensitive : sensitive attribute
    target_rate : desired selection rate (if None, use overall rate)
    n_thresholds : number of candidate thresholds to try

    Returns
    -------
    dict mapping group -> optimal threshold
    """
    if target_rate is None:
        target_rate = (y_prob > 0.5).mean()

    thresholds = np.linspace(0, 1, n_thresholds)
    optimal = {}

    for group in np.unique(sensitive):
        mask = sensitive == group
        group_probs = y_prob[mask]

        best_threshold = 0.5
        best_diff = float("inf")

        for t in thresholds:
            selection_rate = (group_probs > t).mean()
            diff = abs(selection_rate - target_rate)

            if diff < best_diff:
                best_diff = diff
                best_threshold = t

        optimal[group] = best_threshold
        sr = (group_probs > best_threshold).mean()
        print(f"  Group {group}: threshold={best_threshold:.3f}, "
              f"selection_rate={sr:.4f} (target={target_rate:.4f})")

    return optimal


def optimize_thresholds_equalized_odds(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    sensitive: np.ndarray,
    n_thresholds: int = 1000,
) -> Dict[int, float]:
    """Find per-group thresholds that minimize equalized odds violation.

    Equalized odds requires equal TPR AND equal FPR across groups.
    We optimize thresholds to minimize the combined TPR + FPR difference.
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    groups = sorted(np.unique(sensitive))

    # For each combination of group thresholds, compute equalized odds gap
    best_thresholds = {g: 0.5 for g in groups}
    best_gap = float("inf")

    # Grid search over group thresholds
    for t0 in np.linspace(0.1, 0.9, 50):
        for t1 in np.linspace(0.1, 0.9, 50):
            tprs = {}
            fprs = {}

            for g, t in zip(groups, [t0, t1]):
                mask = sensitive == g
                pred = (y_prob[mask] > t).astype(int)
                tp = ((y_true[mask] == 1) & (pred == 1)).sum()
                fn = ((y_true[mask] == 1) & (pred == 0)).sum()
                fp = ((y_true[mask] == 0) & (pred == 1)).sum()
                tn = ((y_true[mask] == 0) & (pred == 0)).sum()

                tprs[g] = tp / (tp + fn) if (tp + fn) > 0 else 0
                fprs[g] = fp / (fp + tn) if (fp + tn) > 0 else 0

            # Equalized odds gap
            gap = abs(tprs[groups[0]] - tprs[groups[1]]) + \
                  abs(fprs[groups[0]] - fprs[groups[1]])

            if gap < best_gap:
                best_gap = gap
                best_thresholds = {groups[0]: t0, groups[1]: t1}

    for g, t in best_thresholds.items():
        mask = sensitive == g
        pred = (y_prob[mask] > t).astype(int)
        tp = ((y_true[mask] == 1) & (pred == 1)).sum()
        fn = ((y_true[mask] == 1) & (pred == 0)).sum()
        fp = ((y_true[mask] == 0) & (pred == 1)).sum()
        tn = ((y_true[mask] == 0) & (pred == 0)).sum()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        print(f"  Group {g}: threshold={t:.3f}, TPR={tpr:.4f}, FPR={fpr:.4f}")

    return best_thresholds


# Apply threshold optimization
baseline_prob = baseline_model.predict_proba(X_test)[:, 1]

print("=== Threshold Optimization: Demographic Parity ===")
dp_thresholds = optimize_thresholds_dp(baseline_prob, race_test)

# Apply optimized thresholds
tp_pred = np.zeros(len(X_test), dtype=int)
for group, threshold in dp_thresholds.items():
    mask = race_test == group
    tp_pred[mask] = (baseline_prob[mask] > threshold).astype(int)

print(f"\nAccuracy: {accuracy_score(y_test, tp_pred):.4f}")
tp_dp = abs(tp_pred[race_test == 0].mean() - tp_pred[race_test == 1].mean())
print(f"DP Difference: {tp_dp:.4f}")

print("\n=== Threshold Optimization: Equalized Odds ===")
eo_thresholds = optimize_thresholds_equalized_odds(
    baseline_prob, y_test, race_test
)

eo_pred = np.zeros(len(X_test), dtype=int)
for group, threshold in eo_thresholds.items():
    mask = race_test == group
    eo_pred[mask] = (baseline_prob[mask] > threshold).astype(int)

print(f"\nAccuracy: {accuracy_score(y_test, eo_pred):.4f}")
```

### 4.2 Equalized Odds Post-Processing (Hardt et al., 2016)

```python
"""
Equalized Odds post-processing (Hardt et al., 2016).

Instead of finding a single threshold per group, this method finds the
optimal randomized classifier that satisfies equalized odds. The
result is a stochastic decision rule: for each group and for each
outcome of the original classifier, there's a probability of flipping
the decision.

The optimization is a linear program:
  maximize  accuracy
  subject to  TPR_0 = TPR_1  and  FPR_0 = FPR_1
"""

from fairlearn.postprocessing import ThresholdOptimizer


def equalized_odds_postprocessing(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    sensitive_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sensitive_test: np.ndarray,
    constraint: str = "equalized_odds",
) -> dict:
    """Apply Fairlearn's ThresholdOptimizer for post-processing.

    The ThresholdOptimizer finds group-specific thresholds that satisfy
    the specified fairness constraint while maximizing accuracy.

    This is equivalent to finding the optimal point on the ROC curve
    for each group such that the fairness constraint is satisfied.
    """
    postprocessor = ThresholdOptimizer(
        estimator=model,
        constraints=constraint,
        predict_method="predict_proba",
    )

    postprocessor.fit(X_train, y_train, sensitive_features=sensitive_train)
    y_pred = postprocessor.predict(X_test, sensitive_features=sensitive_test)

    accuracy = accuracy_score(y_test, y_pred)
    dp_diff = abs(
        y_pred[sensitive_test == 0].mean() - y_pred[sensitive_test == 1].mean()
    )

    # Per-group metrics
    tprs = {}
    fprs = {}
    for g in [0, 1]:
        mask = sensitive_test == g
        tp = ((y_test[mask] == 1) & (y_pred[mask] == 1)).sum()
        fn = ((y_test[mask] == 1) & (y_pred[mask] == 0)).sum()
        fp = ((y_test[mask] == 0) & (y_pred[mask] == 1)).sum()
        tn = ((y_test[mask] == 0) & (y_pred[mask] == 0)).sum()
        tprs[g] = tp / (tp + fn) if (tp + fn) > 0 else 0
        fprs[g] = fp / (fp + tn) if (fp + tn) > 0 else 0

    return {
        "accuracy": accuracy,
        "dp_diff": dp_diff,
        "tpr_diff": abs(tprs[0] - tprs[1]),
        "fpr_diff": abs(fprs[0] - fprs[1]),
        "tprs": tprs,
        "fprs": fprs,
    }


# Apply post-processing
eo_result = equalized_odds_postprocessing(
    baseline_model, X_train, y_train, race_train,
    X_test, y_test, race_test,
    constraint="equalized_odds",
)

print("=== Equalized Odds Post-Processing (Fairlearn) ===")
print(f"Accuracy:       {eo_result['accuracy']:.4f}")
print(f"DP Difference:  {eo_result['dp_diff']:.4f}")
print(f"TPR Difference: {eo_result['tpr_diff']:.4f}")
print(f"FPR Difference: {eo_result['fpr_diff']:.4f}")
for g in [0, 1]:
    print(f"  Group {g}: TPR={eo_result['tprs'][g]:.4f}, FPR={eo_result['fprs'][g]:.4f}")
```

### 4.3 Reject Option Classification (Kamiran et al., 2012)

```python
"""
Reject Option Classification: flip predictions near the decision
boundary in a way that benefits the disadvantaged group.

The idea: predictions near the decision boundary (probability close to 0.5)
are uncertain anyway. For those instances, we can change the prediction
to favor the disadvantaged group without significantly harming accuracy.

Specifically:
- If P(Y=1) in [theta_low, theta_high] (the "reject region"):
  - For disadvantaged group: predict favorable outcome (Y_hat = 0 for default)
  - For advantaged group: predict unfavorable outcome (Y_hat = 1 for default)
"""


def reject_option_classification(
    y_prob: np.ndarray,
    sensitive: np.ndarray,
    disadvantaged_group: int = 1,
    theta_low: float = 0.4,
    theta_high: float = 0.6,
    favorable_label: int = 0,
) -> np.ndarray:
    """Apply reject option classification.

    In the "reject region" (uncertainty band around the decision boundary),
    flip predictions to favor the disadvantaged group.

    Parameters
    ----------
    y_prob : predicted probability of positive class (default=1)
    sensitive : sensitive attribute
    disadvantaged_group : which group to favor in the reject region
    theta_low, theta_high : boundaries of the reject region
    favorable_label : which label is the favorable outcome
                     (0 = "no default" is favorable in credit scoring)
    """
    # Start with standard predictions
    y_pred = (y_prob > 0.5).astype(int)

    # Identify instances in the reject region
    in_reject_region = (y_prob >= theta_low) & (y_prob <= theta_high)

    # For disadvantaged group in reject region: give favorable outcome
    disadvantaged_reject = in_reject_region & (sensitive == disadvantaged_group)
    y_pred[disadvantaged_reject] = favorable_label

    # For advantaged group in reject region: give unfavorable outcome
    advantaged_group = 1 - disadvantaged_group
    advantaged_reject = in_reject_region & (sensitive == advantaged_group)
    y_pred[advantaged_reject] = 1 - favorable_label

    n_flipped = disadvantaged_reject.sum() + advantaged_reject.sum()
    print(f"Reject region: [{theta_low:.2f}, {theta_high:.2f}]")
    print(f"Instances in reject region: {in_reject_region.sum()} "
          f"({in_reject_region.mean()*100:.1f}%)")
    print(f"Predictions flipped: {n_flipped}")

    return y_pred


# Apply reject option
print("=== Reject Option Classification ===")
roc_pred = reject_option_classification(
    baseline_prob, race_test,
    disadvantaged_group=1,
    theta_low=0.35,
    theta_high=0.65,
    favorable_label=0,
)

print(f"\nAccuracy: {accuracy_score(y_test, roc_pred):.4f}")
roc_dp = abs(
    roc_pred[race_test == 0].mean() - roc_pred[race_test == 1].mean()
)
print(f"DP Difference: {roc_dp:.4f}")
```

---

## 5. The Pareto Frontier: Accuracy vs Fairness

### 5.1 Plotting the Accuracy-Fairness Tradeoff

```python
"""
The Pareto frontier of accuracy vs fairness.

There is generally a tradeoff between accuracy and fairness: achieving
perfect fairness requires some sacrifice in accuracy (unless the model
was already fair). The Pareto frontier shows the best achievable
combinations: points where you cannot improve one metric without
degrading the other.
"""


def compute_pareto_frontier(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sensitive_test: np.ndarray,
    metric: str = "demographic_parity",
) -> pd.DataFrame:
    """Compute the accuracy-fairness Pareto frontier by varying thresholds.

    We sweep through threshold combinations for the two groups and compute
    the accuracy and fairness at each point. The Pareto frontier is the
    set of points that are not dominated by any other point.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    points = []

    thresholds = np.linspace(0.1, 0.9, 30)
    groups = sorted(np.unique(sensitive_test))

    for t0 in thresholds:
        for t1 in thresholds:
            pred = np.zeros(len(X_test), dtype=int)
            pred[sensitive_test == groups[0]] = (
                y_prob[sensitive_test == groups[0]] > t0
            ).astype(int)
            pred[sensitive_test == groups[1]] = (
                y_prob[sensitive_test == groups[1]] > t1
            ).astype(int)

            acc = accuracy_score(y_test, pred)

            if metric == "demographic_parity":
                fairness_violation = abs(
                    pred[sensitive_test == groups[0]].mean() -
                    pred[sensitive_test == groups[1]].mean()
                )
            elif metric == "equalized_odds":
                tpr0 = pred[(sensitive_test == groups[0]) & (y_test == 1)].mean()
                tpr1 = pred[(sensitive_test == groups[1]) & (y_test == 1)].mean()
                fpr0 = pred[(sensitive_test == groups[0]) & (y_test == 0)].mean()
                fpr1 = pred[(sensitive_test == groups[1]) & (y_test == 0)].mean()
                fairness_violation = abs(tpr0 - tpr1) + abs(fpr0 - fpr1)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            points.append({
                "threshold_0": t0,
                "threshold_1": t1,
                "accuracy": acc,
                "fairness_violation": fairness_violation,
            })

    df = pd.DataFrame(points)

    # Find Pareto-optimal points
    # A point is Pareto-optimal if no other point has BOTH higher accuracy
    # AND lower fairness violation
    pareto_mask = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        for j in range(len(df)):
            if i == j:
                continue
            # Check if point j dominates point i
            if (df.iloc[j]["accuracy"] >= df.iloc[i]["accuracy"] and
                df.iloc[j]["fairness_violation"] <= df.iloc[i]["fairness_violation"] and
                (df.iloc[j]["accuracy"] > df.iloc[i]["accuracy"] or
                 df.iloc[j]["fairness_violation"] < df.iloc[i]["fairness_violation"])):
                pareto_mask[i] = False
                break

    df["pareto_optimal"] = pareto_mask

    return df


# Compute Pareto frontier
pareto_df = compute_pareto_frontier(
    baseline_model, X_test, y_test, race_test,
    metric="demographic_parity",
)

pareto_points = pareto_df[pareto_df["pareto_optimal"]].sort_values("accuracy")

print("=== Pareto Frontier (Accuracy vs DP Violation) ===")
print(f"{'Accuracy':>10s} {'DP Violation':>14s} {'Threshold_0':>13s} {'Threshold_1':>13s}")
for _, row in pareto_points.head(10).iterrows():
    print(f"{row['accuracy']:10.4f} {row['fairness_violation']:14.4f} "
          f"{row['threshold_0']:13.2f} {row['threshold_1']:13.2f}")

print(f"\nTotal points evaluated: {len(pareto_df)}")
print(f"Pareto-optimal points: {pareto_points.shape[0]}")
```

### 5.2 Multi-Objective Optimization

```python
"""
Multi-objective optimization: finding the best accuracy-fairness tradeoff.

Instead of picking an arbitrary point on the Pareto frontier, we can
use a scalarization approach with a preference parameter:

  minimize  (1 - alpha) * (1 - accuracy) + alpha * fairness_violation

where alpha in [0, 1] controls the preference:
  alpha = 0: optimize only accuracy (ignore fairness)
  alpha = 1: optimize only fairness (ignore accuracy)
  alpha = 0.5: balanced tradeoff
"""


def multi_objective_selection(
    pareto_df: pd.DataFrame,
    alpha: float = 0.5,
) -> dict:
    """Select the best operating point from the Pareto frontier.

    Uses linear scalarization to combine accuracy and fairness into
    a single objective.
    """
    pareto_points = pareto_df[pareto_df["pareto_optimal"]].copy()

    # Normalize metrics to [0, 1]
    acc_range = pareto_points["accuracy"].max() - pareto_points["accuracy"].min()
    fair_range = pareto_points["fairness_violation"].max() - pareto_points["fairness_violation"].min()

    if acc_range > 0:
        acc_normalized = (pareto_points["accuracy"] - pareto_points["accuracy"].min()) / acc_range
    else:
        acc_normalized = 1.0

    if fair_range > 0:
        fair_normalized = (pareto_points["fairness_violation"] - pareto_points["fairness_violation"].min()) / fair_range
    else:
        fair_normalized = 0.0

    # Scalarized objective: maximize accuracy, minimize fairness violation
    pareto_points = pareto_points.copy()
    pareto_points["score"] = (1 - alpha) * acc_normalized - alpha * fair_normalized

    best_idx = pareto_points["score"].idxmax()
    best = pareto_points.loc[best_idx]

    return {
        "alpha": alpha,
        "accuracy": best["accuracy"],
        "fairness_violation": best["fairness_violation"],
        "threshold_0": best["threshold_0"],
        "threshold_1": best["threshold_1"],
    }


# Find best operating points for different preferences
print("=== Multi-Objective Operating Points ===")
for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
    result = multi_objective_selection(pareto_df, alpha)
    print(f"alpha={alpha:.2f}: accuracy={result['accuracy']:.4f}, "
          f"DP_violation={result['fairness_violation']:.4f}, "
          f"thresholds=({result['threshold_0']:.2f}, {result['threshold_1']:.2f})")
```

---

## 6. Choosing a Strategy

### 6.1 Decision Framework

```python
"""
Decision framework for choosing a fairness mitigation strategy.

The right approach depends on:
1. When can you intervene? (data collection, training, deployment)
2. What is the model type? (custom NN, sklearn model, black box API)
3. What is the fairness requirement? (demographic parity, equalized odds, etc.)
4. What is the cost of accuracy loss?
5. Are there regulatory constraints?
"""

decision_tree = {
    "Can you modify the training data?": {
        True: {
            "Is the bias in the labels?": {
                True: "Use REWEIGHING to correct label bias",
                False: {
                    "Is the bias in the features?": {
                        True: "Use DISPARATE IMPACT REMOVER or FAIR REPRESENTATIONS",
                        False: "Bias may not be in the data; consider in-processing",
                    }
                }
            }
        },
        False: {
            "Can you modify the training algorithm?": {
                True: {
                    "Do you need a specific fairness constraint?": {
                        True: "Use EXPONENTIATED GRADIENT with the constraint",
                        False: "Use ADVERSARIAL DEBIASING for general fairness",
                    }
                },
                False: {
                    "Can you access model probabilities?": {
                        True: {
                            "What fairness metric?": {
                                "Demographic Parity": "Use THRESHOLD OPTIMIZATION",
                                "Equalized Odds": "Use EQUALIZED ODDS POST-PROCESSING",
                                "General": "Use REJECT OPTION CLASSIFICATION",
                            }
                        },
                        False: "Limited options: consider retraining with fairness constraints",
                    }
                }
            }
        }
    }
}


def print_strategy_comparison() -> pd.DataFrame:
    """Print a comparison table of all mitigation strategies."""
    strategies = [
        {
            "Strategy": "Reweighing",
            "Stage": "Pre-processing",
            "Model Agnostic": "Yes",
            "Accuracy Impact": "Low",
            "Fairness Guarantee": "Weak (depends on model)",
            "Best For": "Label bias, quick fix",
        },
        {
            "Strategy": "Disparate Impact Remover",
            "Stage": "Pre-processing",
            "Model Agnostic": "Yes",
            "Accuracy Impact": "Medium",
            "Fairness Guarantee": "Moderate",
            "Best For": "Feature-level bias",
        },
        {
            "Strategy": "Fair Representations",
            "Stage": "Pre-processing",
            "Model Agnostic": "Yes",
            "Accuracy Impact": "Medium-High",
            "Fairness Guarantee": "Strong",
            "Best For": "Multiple downstream models",
        },
        {
            "Strategy": "Adversarial Debiasing",
            "Stage": "In-processing",
            "Model Agnostic": "No (neural nets)",
            "Accuracy Impact": "Medium",
            "Fairness Guarantee": "Strong",
            "Best For": "Deep learning models",
        },
        {
            "Strategy": "Exponentiated Gradient",
            "Stage": "In-processing",
            "Model Agnostic": "Yes",
            "Accuracy Impact": "Low-Medium",
            "Fairness Guarantee": "Strong (provable)",
            "Best For": "Specific fairness constraints",
        },
        {
            "Strategy": "Threshold Optimization",
            "Stage": "Post-processing",
            "Model Agnostic": "Yes",
            "Accuracy Impact": "Low",
            "Fairness Guarantee": "Exact (for chosen metric)",
            "Best For": "Quick deployment fix",
        },
        {
            "Strategy": "Equalized Odds PP",
            "Stage": "Post-processing",
            "Model Agnostic": "Yes",
            "Accuracy Impact": "Low-Medium",
            "Fairness Guarantee": "Exact",
            "Best For": "Equalized odds requirement",
        },
        {
            "Strategy": "Reject Option",
            "Stage": "Post-processing",
            "Model Agnostic": "Yes",
            "Accuracy Impact": "Low",
            "Fairness Guarantee": "Moderate",
            "Best For": "Uncertain predictions",
        },
    ]

    return pd.DataFrame(strategies)


comparison_df = print_strategy_comparison()
print("=== Strategy Comparison ===")
print(comparison_df.to_string(index=False))
```

---

## 7. Proxy Discrimination

### 7.1 Detecting Proxy Features

Even when the sensitive attribute is removed from the feature set, the model
may learn to use *proxy features* -- features that are strongly correlated
with the protected attribute and thus indirectly encode it.

```python
"""
Proxy discrimination: features that act as stand-ins for protected attributes.

Common proxies:
- Zip code -> race (residential segregation)
- First name -> gender (gendered names)
- School name -> socioeconomic status
- Language spoken -> national origin

Detecting proxy features is essential because simply removing the
protected attribute from the training data ("fairness through unawareness")
does NOT guarantee fairness.
"""


def detect_proxy_features(
    X: np.ndarray,
    sensitive: np.ndarray,
    feature_names: List[str],
    threshold: float = 0.1,
) -> pd.DataFrame:
    """Detect features that act as proxies for the sensitive attribute.

    Method: For each feature, train a simple model to predict the
    sensitive attribute from that single feature. Features with high
    predictive power are potential proxies.

    Additionally, compute the correlation between each feature and
    the sensitive attribute.
    """
    results = []

    for i, name in enumerate(feature_names):
        # Correlation
        corr = np.corrcoef(X[:, i], sensitive)[0, 1]

        # Predictive power (single-feature logistic regression)
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X[:, i:i+1], sensitive)
        pred_accuracy = lr.score(X[:, i:i+1], sensitive)
        baseline_accuracy = max(sensitive.mean(), 1 - sensitive.mean())

        # Mutual information (discretized)
        from sklearn.metrics import mutual_info_score
        X_binned = pd.qcut(X[:, i], q=10, labels=False, duplicates="drop")
        mi = mutual_info_score(X_binned, sensitive)

        is_proxy = (
            abs(corr) > threshold or
            (pred_accuracy - baseline_accuracy) > 0.05
        )

        results.append({
            "feature": name,
            "correlation": corr,
            "pred_accuracy": pred_accuracy,
            "baseline_accuracy": baseline_accuracy,
            "accuracy_lift": pred_accuracy - baseline_accuracy,
            "mutual_information": mi,
            "is_proxy": is_proxy,
        })

    df = pd.DataFrame(results)
    df = df.sort_values("accuracy_lift", ascending=False)
    return df


# Detect proxies for race
proxy_df = detect_proxy_features(X_train, race_train, feature_cols)

print("=== Proxy Feature Detection (for race) ===")
print(proxy_df.to_string(index=False))
print(f"\nDetected proxies: {proxy_df[proxy_df['is_proxy']]['feature'].tolist()}")
```

### 7.2 Mitigating Proxy Discrimination

```python
"""
Strategies for mitigating proxy discrimination:

1. Remove proxy features (but may lose legitimate information)
2. Decorrelate features from the sensitive attribute
3. Use causal analysis to separate legitimate from illegitimate effects
4. Apply fairness constraints that account for proxies
"""


def decorrelate_features(
    X: np.ndarray,
    sensitive: np.ndarray,
    feature_names: List[str],
    proxy_features: List[str],
) -> np.ndarray:
    """Remove the component of proxy features that correlates with
    the sensitive attribute, while preserving the residual information.

    For each proxy feature X_i:
      X_i_fair = X_i - E[X_i | A] + E[X_i]

    This removes the group mean difference while preserving within-group
    variation. It is a form of conditional mean correction.
    """
    X_fair = X.copy().astype(float)
    overall_mean = X.mean(axis=0)

    for feat_name in proxy_features:
        feat_idx = feature_names.index(feat_name)

        # Compute group means
        for g in np.unique(sensitive):
            mask = sensitive == g
            group_mean = X[mask, feat_idx].mean()
            # Shift group values to have the overall mean
            # This removes the between-group difference
            X_fair[mask, feat_idx] = (
                X[mask, feat_idx] - group_mean + overall_mean[feat_idx]
            )

    return X_fair


# Decorrelate proxy features
proxy_features = proxy_df[proxy_df["is_proxy"]]["feature"].tolist()
X_train_decorr = decorrelate_features(X_train, race_train, feature_cols, proxy_features)
X_test_decorr = decorrelate_features(X_test, race_test, feature_cols, proxy_features)

# Train on decorrelated features
decorr_model = GradientBoostingClassifier(
    n_estimators=100, max_depth=4, random_state=42
)
decorr_model.fit(X_train_decorr, y_train)
decorr_pred = decorr_model.predict(X_test_decorr)

print("=== Decorrelated Features Model ===")
print(f"Accuracy: {accuracy_score(y_test, decorr_pred):.4f}")
decorr_dp = abs(
    decorr_pred[race_test == 0].mean() - decorr_pred[race_test == 1].mean()
)
print(f"DP Difference: {decorr_dp:.4f}")
```

---

## 8. Practical: Credit Scoring Comparison

### 8.1 Comprehensive Comparison of All Methods

```python
"""
Final comparison: apply all mitigation strategies to the credit scoring
problem and compare on the accuracy-fairness tradeoff.
"""


def comprehensive_comparison(
    X_train, y_train, race_train,
    X_test, y_test, race_test,
) -> pd.DataFrame:
    """Compare all mitigation strategies on the same dataset."""
    results = []

    # 1. Baseline (no mitigation)
    baseline = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    baseline.fit(X_train, y_train)
    pred = baseline.predict(X_test)
    results.append({
        "Method": "Baseline (no mitigation)",
        "Stage": "None",
        "Accuracy": accuracy_score(y_test, pred),
        "DP Difference": abs(pred[race_test==0].mean() - pred[race_test==1].mean()),
    })

    # 2. Reweighing
    weights = compute_reweighing_weights(race_train, y_train)
    rw = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    rw.fit(X_train, y_train, sample_weight=weights)
    pred = rw.predict(X_test)
    results.append({
        "Method": "Reweighing",
        "Stage": "Pre-processing",
        "Accuracy": accuracy_score(y_test, pred),
        "DP Difference": abs(pred[race_test==0].mean() - pred[race_test==1].mean()),
    })

    # 3. Disparate Impact Remover
    X_tr_rep = disparate_impact_remover(X_train, race_train, repair_level=0.8)
    X_te_rep = disparate_impact_remover(X_test, race_test, repair_level=0.8)
    di = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    di.fit(X_tr_rep, y_train)
    pred = di.predict(X_te_rep)
    results.append({
        "Method": "Disparate Impact Remover",
        "Stage": "Pre-processing",
        "Accuracy": accuracy_score(y_test, pred),
        "DP Difference": abs(pred[race_test==0].mean() - pred[race_test==1].mean()),
    })

    # 4. Exponentiated Gradient (Demographic Parity)
    eg = ExponentiatedGradient(
        estimator=LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42),
        constraints=DemographicParity(),
    )
    eg.fit(X_train, y_train, sensitive_features=race_train)
    pred = eg.predict(X_test)
    results.append({
        "Method": "Exponentiated Gradient (DP)",
        "Stage": "In-processing",
        "Accuracy": accuracy_score(y_test, pred),
        "DP Difference": abs(pred[race_test==0].mean() - pred[race_test==1].mean()),
    })

    # 5. Exponentiated Gradient (Equalized Odds)
    eg_eo = ExponentiatedGradient(
        estimator=LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42),
        constraints=EqualizedOdds(),
    )
    eg_eo.fit(X_train, y_train, sensitive_features=race_train)
    pred = eg_eo.predict(X_test)
    results.append({
        "Method": "Exponentiated Gradient (EO)",
        "Stage": "In-processing",
        "Accuracy": accuracy_score(y_test, pred),
        "DP Difference": abs(pred[race_test==0].mean() - pred[race_test==1].mean()),
    })

    # 6. Threshold Optimization (DP)
    prob = baseline.predict_proba(X_test)[:, 1]
    dp_thresh = optimize_thresholds_dp(prob, race_test)
    pred = np.zeros(len(X_test), dtype=int)
    for g, t in dp_thresh.items():
        mask = race_test == g
        pred[mask] = (prob[mask] > t).astype(int)
    results.append({
        "Method": "Threshold Optimization (DP)",
        "Stage": "Post-processing",
        "Accuracy": accuracy_score(y_test, pred),
        "DP Difference": abs(pred[race_test==0].mean() - pred[race_test==1].mean()),
    })

    # 7. Equalized Odds Post-Processing
    try:
        to = ThresholdOptimizer(
            estimator=baseline,
            constraints="equalized_odds",
            predict_method="predict_proba",
        )
        to.fit(X_train, y_train, sensitive_features=race_train)
        pred = to.predict(X_test, sensitive_features=race_test)
        results.append({
            "Method": "Equalized Odds Post-Proc",
            "Stage": "Post-processing",
            "Accuracy": accuracy_score(y_test, pred),
            "DP Difference": abs(pred[race_test==0].mean() - pred[race_test==1].mean()),
        })
    except Exception as e:
        print(f"Equalized Odds Post-Proc failed: {e}")

    # 8. Reject Option
    pred = reject_option_classification(
        prob, race_test,
        disadvantaged_group=1,
        theta_low=0.35, theta_high=0.65,
        favorable_label=0,
    )
    results.append({
        "Method": "Reject Option",
        "Stage": "Post-processing",
        "Accuracy": accuracy_score(y_test, pred),
        "DP Difference": abs(pred[race_test==0].mean() - pred[race_test==1].mean()),
    })

    return pd.DataFrame(results)


# Run comprehensive comparison
print("=" * 70)
print("  COMPREHENSIVE MITIGATION COMPARISON")
print("=" * 70)

comparison = comprehensive_comparison(
    X_train, y_train, race_train,
    X_test, y_test, race_test,
)

print("\n" + comparison.sort_values("DP Difference").to_string(index=False))

# Identify the best strategy
best_balanced = comparison.loc[
    (comparison["DP Difference"] < 0.05) &
    (comparison["Accuracy"] == comparison[comparison["DP Difference"] < 0.05]["Accuracy"].max())
]

if not best_balanced.empty:
    print(f"\nBest balanced strategy (DP < 0.05):")
    print(f"  {best_balanced.iloc[0]['Method']}")
    print(f"  Accuracy: {best_balanced.iloc[0]['Accuracy']:.4f}")
    print(f"  DP Diff:  {best_balanced.iloc[0]['DP Difference']:.4f}")
```

### 8.2 Summary Report

```python
"""
Generate a final summary report comparing all approaches.
"""


def generate_mitigation_report(comparison_df: pd.DataFrame) -> str:
    """Generate a structured mitigation comparison report."""
    report = [
        "=" * 60,
        "  FAIRNESS MITIGATION REPORT",
        "  Credit Scoring Model",
        "=" * 60,
        "",
        "1. BASELINE PERFORMANCE",
    ]

    baseline = comparison_df[comparison_df["Method"].str.contains("Baseline")]
    if not baseline.empty:
        report.append(f"   Accuracy:       {baseline.iloc[0]['Accuracy']:.4f}")
        report.append(f"   DP Difference:  {baseline.iloc[0]['DP Difference']:.4f}")

    report.extend(["", "2. MITIGATION RESULTS"])

    for stage in ["Pre-processing", "In-processing", "Post-processing"]:
        stage_methods = comparison_df[comparison_df["Stage"] == stage]
        if not stage_methods.empty:
            report.append(f"\n   {stage}:")
            for _, row in stage_methods.iterrows():
                report.append(
                    f"     {row['Method']:40s} "
                    f"Acc={row['Accuracy']:.4f}  DP={row['DP Difference']:.4f}"
                )

    # Best overall
    fair_methods = comparison_df[comparison_df["DP Difference"] < 0.05]
    if not fair_methods.empty:
        best = fair_methods.loc[fair_methods["Accuracy"].idxmax()]
        report.extend([
            "",
            "3. RECOMMENDATION",
            f"   Best method (DP < 0.05): {best['Method']}",
            f"   Accuracy: {best['Accuracy']:.4f}",
            f"   DP Difference: {best['DP Difference']:.4f}",
        ])

    report.extend([
        "",
        "4. CONSIDERATIONS",
        "   - Pre-processing methods are model-agnostic but may lose information",
        "   - In-processing methods offer the best fairness-accuracy tradeoff",
        "   - Post-processing is the quickest fix but does not address root causes",
        "   - Always audit for proxy discrimination after mitigation",
        "   - The impossibility theorem means some tradeoffs are unavoidable",
    ])

    return "\n".join(report)


report = generate_mitigation_report(comparison)
print(report)
```

---

## Summary

- **Pre-processing** methods modify training data to remove bias:
  - *Reweighing* (Kamiran & Calders): assigns weights to equalize P(Y|A) across groups
  - *Disparate Impact Remover* (Feldman et al.): adjusts feature distributions via quantile matching
  - *Learning Fair Representations* (Zemel et al.): projects data into a space uninformative about A
- **In-processing** methods incorporate fairness during training:
  - *Adversarial Debiasing* (Zhang et al.): trains a predictor to fool a sensitive-attribute adversary
  - *Exponentiated Gradient* (Agarwal et al.): solves constrained optimization via reduction to weighted classification
  - *Constrained Optimization*: adds a Lagrangian penalty for fairness violations to the training loss
- **Post-processing** methods adjust model outputs:
  - *Threshold Optimization*: different decision thresholds per group to equalize rates
  - *Equalized Odds Post-Processing* (Hardt et al.): finds the optimal randomized classifier satisfying equalized odds
  - *Reject Option Classification* (Kamiran et al.): flips uncertain predictions to favor the disadvantaged group
- The **Pareto frontier** maps the accuracy-fairness tradeoff. Multi-objective optimization with a preference parameter alpha selects the best operating point.
- **Proxy discrimination** occurs when protected attributes are encoded in seemingly neutral features (e.g., zip code as a proxy for race). Detection via correlation analysis and mitigation via decorrelation are essential complements to any fairness intervention.
- **Strategy selection** depends on intervention constraints (when can you act?), model type (can you modify training?), fairness definition (which metric?), and regulatory context.

---

## Exercises

### Exercise 1: Reweighing Deep Dive (Beginner)

1. Implement reweighing on a binary classification dataset with two protected attributes (race and gender)
2. Compute instance weights for all four intersectional groups
3. Train three models: (a) unweighted, (b) reweighed for race, (c) reweighed for gender
4. Compare fairness metrics across all three models and discuss which group disparities remain

### Exercise 2: Adversarial Debiasing Tuning (Intermediate)

Using the adversarial debiasing implementation from Section 3.1:
1. Sweep the adversary_weight parameter from 0.1 to 10.0
2. For each value, record accuracy and demographic parity difference
3. Plot the accuracy-fairness Pareto frontier
4. Find the "elbow point" where further fairness improvement requires disproportionate accuracy loss
5. Discuss the practical implications of choosing a specific operating point

### Exercise 3: Post-Processing Comparison (Intermediate)

For a pre-trained gradient boosting model on a dataset of your choice:
1. Implement threshold optimization for (a) demographic parity, (b) equalized odds, (c) equal opportunity
2. Implement reject option classification with three different reject region widths
3. Compare all six approaches on accuracy, DP difference, TPR difference, and FPR difference
4. Create a summary table recommending the best approach for different fairness priorities

### Exercise 4: Full Pipeline with Proxy Detection (Advanced)

Build an end-to-end fair ML pipeline for credit scoring:
1. Generate or load a credit scoring dataset with at least 6 features and 2 protected attributes
2. Detect proxy features using the method from Section 7
3. Apply one pre-processing, one in-processing, and one post-processing method
4. For each method, check if proxy discrimination remains after mitigation
5. Compute the Pareto frontier and identify the best operating point
6. Write a 1-page deployment recommendation covering which method to use, why, and what residual risks remain

### Exercise 5: Intersectional Mitigation (Advanced)

Extend the mitigation methods to handle intersectional fairness:
1. Define fairness constraints for 4 intersectional groups (race x gender)
2. Modify the reweighing algorithm to compute weights for all 4 groups
3. Apply Fairlearn's ExponentiatedGradient with intersectional sensitive features
4. Apply threshold optimization with separate thresholds for each intersectional group
5. Compare: which method best reduces the MAXIMUM disparity across all 4 groups?
6. Discuss the tradeoff between intersectional fairness and overall accuracy

---

[Previous: Advanced Algorithmic Fairness](./11_Advanced_Algorithmic_Fairness.md) | [Overview](./00_Overview.md) | [Next: AI Regulation and Governance](./13_AI_Regulation_and_Governance.md)

**License**: CC BY-NC 4.0
