# Lesson 8: Counterfactual Explanations

[Previous: Concept-Based Explanations](./07_Concept_Based_Explanations.md) | [Next: Causal Inference for Interpretability](./09_Causal_Inference_for_Interpretability.md)

---

## Learning Objectives

- Formalize counterfactual explanations as an optimization problem and implement the Wachter et al. (2017) formulation from scratch
- Generate diverse counterfactual explanations using DiCE with determinantal point processes for set diversity
- Evaluate counterfactual quality using proximity, sparsity, plausibility, and actionability metrics
- Incorporate causal constraints so that counterfactuals respect real-world dependencies between features
- Build contrastive explanations that answer "why A instead of B?" for multi-class settings

---

When a loan application is denied, the applicant does not want to know "feature 7 contributed -0.3 to the score." They want to know: **"What would I need to change to get approved?"** This is the core promise of counterfactual explanations: instead of attributing responsibility for a decision, they describe the minimal change to the input that would change the decision.

Counterfactual explanations are inherently actionable. They naturally translate into recommendations ("increase your income by $5,000 and reduce your debt by $2,000") and satisfy legal requirements for algorithmic recourse — the right to know what you can do to change an unfavorable automated decision.

---

## 1. Counterfactual Foundations

### 1.1 What Is a Counterfactual Explanation?

```python
"""
A counterfactual explanation for instance x with prediction y is an
alternative instance x' such that:
    1. The model predicts a DIFFERENT outcome for x': f(x') ≠ y
    2. x' is as CLOSE to x as possible (minimal change)
    3. x' is PLAUSIBLE (a realistic data point)

Formally:
    x' = argmin_{x'} distance(x, x')
         subject to: f(x') = y_target

Example:
    x = {income: 45000, debt: 15000, age: 28, employed: True}
    f(x) = "Denied"

    x' = {income: 52000, debt: 15000, age: 28, employed: True}
    f(x') = "Approved"

    Explanation: "If your income were $52,000 instead of $45,000,
                 your loan would be approved."

Key properties of good counterfactuals:
1. PROXIMITY: close to the original (small changes)
2. SPARSITY: few features changed (ideally 1-3)
3. PLAUSIBILITY: the counterfactual is a realistic data point
4. ACTIONABILITY: the changes are feasible (can't change age, race)
5. CAUSALITY: respects causal relationships between features
"""
```

### 1.2 The Wachter et al. (2017) Formulation

```python
"""
Wachter et al. (2017): The foundational counterfactual formulation.

Optimization:
    x' = argmin_{x'} lambda * (f(x') - y_target)^2 + distance(x, x')

Where:
    - lambda * (f(x') - y_target)^2: prediction loss (forces x' to get
      the desired prediction). Lambda controls how strongly we push
      toward the target prediction.
    - distance(x, x'): proximity loss (keeps x' close to x).
      Typically L1 (encourages sparsity) or L2 (smooth changes).

This is solved via gradient descent: start from x, iteratively move
toward a point that gets the desired prediction while staying close.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class CounterfactualResult:
    """Container for a counterfactual explanation."""
    original: np.ndarray
    counterfactual: np.ndarray
    original_pred: float
    cf_pred: float
    target: float
    distance_l1: float
    distance_l2: float
    n_features_changed: int
    features_changed: list[tuple[str, float, float]]  # (name, old, new)
    success: bool


class WachterCounterfactual:
    """
    Implementation of Wachter et al. (2017) counterfactual generation.

    This is the foundational method. More advanced methods (DiCE, Alibi)
    build on this with additional constraints.
    """

    def __init__(
        self,
        model: nn.Module,
        feature_names: list[str],
        feature_ranges: Optional[dict[str, tuple[float, float]]] = None,
        categorical_features: Optional[list[str]] = None
    ):
        """
        Parameters:
            model: Trained PyTorch model (binary classifier)
            feature_names: Names of input features
            feature_ranges: Optional min/max bounds for each feature.
                          Prevents generating impossible values
                          (e.g., negative age).
            categorical_features: Features that should not be
                                continuously optimized (handled separately)
        """
        self.model = model
        self.model.eval()
        self.feature_names = feature_names
        self.feature_ranges = feature_ranges or {}
        self.categorical_features = set(categorical_features or [])

    def generate(
        self,
        x: np.ndarray,
        target_class: float = 1.0,
        lambda_param: float = 0.1,
        lr: float = 0.01,
        max_iterations: int = 1000,
        distance_metric: str = "l1",
        convergence_threshold: float = 1e-4
    ) -> CounterfactualResult:
        """
        Generate a counterfactual explanation for instance x.

        Parameters:
            x: Original instance, shape (n_features,)
            target_class: Desired prediction (e.g., 1.0 for "approved")
            lambda_param: Weight on the prediction loss.
                         Higher lambda → stronger push toward target
                         but potentially larger changes.
                         Lower lambda → minimal changes but might not
                         reach the target prediction.
            lr: Learning rate for gradient descent
            max_iterations: Maximum optimization steps
            distance_metric: "l1" (sparse changes) or "l2" (smooth changes)
            convergence_threshold: Stop when loss change < this

        Returns:
            CounterfactualResult with the explanation
        """
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

        # Initialize counterfactual at the original point
        # We optimize this variable to find the closest point with
        # the target prediction
        x_cf = x_tensor.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([x_cf], lr=lr)
        prev_loss = float('inf')

        for iteration in range(max_iterations):
            optimizer.zero_grad()

            # Prediction loss: push toward target class
            pred = self.model(x_cf).squeeze()
            if pred.dim() == 0:
                pred_loss = (pred - target_class) ** 2
            else:
                # Multi-class: use target class probability
                pred_loss = -pred[int(target_class)]

            # Distance loss: stay close to original
            diff = x_cf - x_tensor

            if distance_metric == "l1":
                # L1 norm encourages sparsity (few features change)
                # We use MAD (Median Absolute Deviation) weighting
                # to normalize features to comparable scales
                dist_loss = torch.abs(diff).sum()
            elif distance_metric == "l2":
                # L2 norm encourages smooth, distributed changes
                dist_loss = (diff ** 2).sum()
            elif distance_metric == "elastic":
                # Elastic net: combination of L1 and L2
                dist_loss = 0.5 * torch.abs(diff).sum() + \
                           0.5 * (diff ** 2).sum()

            # Total loss
            total_loss = lambda_param * pred_loss + dist_loss

            total_loss.backward()
            optimizer.step()

            # Enforce feature ranges (project back into valid ranges)
            with torch.no_grad():
                for feat_idx, feat_name in enumerate(self.feature_names):
                    if feat_name in self.feature_ranges:
                        low, high = self.feature_ranges[feat_name]
                        x_cf[0, feat_idx].clamp_(low, high)

                    # Categorical features: round to nearest valid value
                    if feat_name in self.categorical_features:
                        x_cf[0, feat_idx] = torch.round(x_cf[0, feat_idx])

            # Check convergence
            current_loss = total_loss.item()
            if abs(prev_loss - current_loss) < convergence_threshold:
                break
            prev_loss = current_loss

            # Early success: if we've reached the target prediction
            with torch.no_grad():
                current_pred = self.model(x_cf).squeeze()
                if current_pred.dim() == 0:
                    reached_target = abs(current_pred.item() - target_class) < 0.1
                else:
                    reached_target = current_pred.argmax().item() == int(target_class)

                if reached_target and iteration > 50:
                    # Continue optimizing for a bit to minimize distance
                    pass

        # Construct result
        x_cf_np = x_cf.detach().squeeze().numpy()
        x_np = x_tensor.squeeze().numpy()

        with torch.no_grad():
            original_pred = self.model(x_tensor).squeeze().item()
            cf_pred = self.model(x_cf).squeeze().item()

        # Identify changed features
        diff_np = x_cf_np - x_np
        change_threshold = 0.01  # Ignore tiny changes
        features_changed = []
        for i, name in enumerate(self.feature_names):
            if abs(diff_np[i]) > change_threshold:
                features_changed.append((name, x_np[i], x_cf_np[i]))

        return CounterfactualResult(
            original=x_np,
            counterfactual=x_cf_np,
            original_pred=original_pred,
            cf_pred=cf_pred,
            target=target_class,
            distance_l1=np.abs(diff_np).sum(),
            distance_l2=np.sqrt((diff_np ** 2).sum()),
            n_features_changed=len(features_changed),
            features_changed=features_changed,
            success=abs(cf_pred - target_class) < 0.3
        )

    def explain(self, x: np.ndarray, target_class: float = 1.0) -> str:
        """
        Generate a human-readable counterfactual explanation.
        """
        result = self.generate(x, target_class)

        if not result.success:
            return ("Could not find a counterfactual that changes the "
                    "prediction. Try increasing lambda_param or max_iterations.")

        lines = []
        lines.append(f"Original prediction: {result.original_pred:.3f}")
        lines.append(f"Target prediction:   {result.target:.3f}")
        lines.append(f"Counterfactual prediction: {result.cf_pred:.3f}")
        lines.append(f"")
        lines.append(f"To change the prediction, make these changes:")

        for name, old_val, new_val in result.features_changed:
            direction = "increase" if new_val > old_val else "decrease"
            change = abs(new_val - old_val)
            lines.append(f"  - {direction} {name} from {old_val:.2f} "
                        f"to {new_val:.2f} (change: {change:+.2f})")

        lines.append(f"")
        lines.append(f"Distance (L1): {result.distance_l1:.4f}")
        lines.append(f"Features changed: {result.n_features_changed}")

        return "\n".join(lines)
```

---

## 2. DiCE: Diverse Counterfactual Explanations

### 2.1 Why Diversity Matters

```python
"""
DiCE (Mothilal et al. 2020): Diverse Counterfactual Explanations

Problem with Wachter: it generates a SINGLE counterfactual.
But there might be many ways to change the outcome, and showing
only one is:
1. Potentially misleading (user thinks it's the ONLY way)
2. Less useful (user might not be able to make that specific change)
3. Lacking robustness (the single CF might be on a decision boundary)

DiCE generates a SET of diverse counterfactuals using:
1. Diversity regularization: penalize counterfactuals that are too
   similar to each other (using determinantal point processes)
2. Multiple starting points and optimization paths
3. Feature constraints: immutable features, range constraints

Example output for a loan denial:
    CF1: "Increase income by $7,000"
    CF2: "Reduce debt by $5,000 and get employed"
    CF3: "Increase income by $3,000 and improve credit score by 50 points"

The user can choose the path most feasible for their situation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class DiCEExplainer:
    """
    Diverse Counterfactual Explanations using DiCE methodology.

    Generates a set of diverse, valid, and proximate counterfactuals.
    """

    def __init__(
        self,
        model: nn.Module,
        feature_names: list[str],
        feature_ranges: dict[str, tuple[float, float]],
        immutable_features: Optional[list[str]] = None,
        categorical_features: Optional[dict[str, list]] = None
    ):
        """
        Parameters:
            model: Trained classifier
            feature_names: Feature names
            feature_ranges: Min/max for each feature
            immutable_features: Features that CANNOT be changed
                              (e.g., race, sex if protected;
                               or age if decreasing is impossible)
            categorical_features: {feature_name: [valid_values]}
        """
        self.model = model
        self.model.eval()
        self.feature_names = feature_names
        self.feature_ranges = feature_ranges
        self.immutable_features = set(immutable_features or [])
        self.categorical_features = categorical_features or {}
        self.n_features = len(feature_names)

    def _compute_diversity_loss(
        self, cfs: torch.Tensor, method: str = "dpp"
    ) -> torch.Tensor:
        """
        Compute diversity loss to encourage counterfactuals to be different.

        Methods:
        1. DPP (Determinantal Point Process): uses the determinant of the
           similarity kernel as a diversity measure.
           det(K) is maximized when points are dissimilar.

        2. pairwise: simple pairwise distance maximization

        Parameters:
            cfs: Counterfactual batch, shape (n_cfs, n_features)
            method: "dpp" or "pairwise"
        """
        n_cfs = cfs.shape[0]

        if method == "dpp":
            # DPP diversity: maximize det(K) where K_ij = k(cf_i, cf_j)
            # Using RBF kernel: k(x, y) = exp(-||x-y||^2 / (2*sigma^2))
            diffs = cfs.unsqueeze(0) - cfs.unsqueeze(1)  # (n, n, d)
            sq_dists = (diffs ** 2).sum(dim=2)  # (n, n)
            sigma = sq_dists.median().sqrt().clamp(min=1e-3)
            K = torch.exp(-sq_dists / (2 * sigma ** 2))

            # We want to MAXIMIZE det(K), so we MINIMIZE -log(det(K))
            # Adding small identity for numerical stability
            K = K + 1e-4 * torch.eye(n_cfs)
            diversity_loss = -torch.logdet(K)

        elif method == "pairwise":
            # Pairwise diversity: maximize sum of pairwise distances
            total_dist = 0.0
            count = 0
            for i in range(n_cfs):
                for j in range(i + 1, n_cfs):
                    total_dist += torch.abs(cfs[i] - cfs[j]).sum()
                    count += 1
            # Minimize negative distance (= maximize distance)
            diversity_loss = -total_dist / max(count, 1)

        return diversity_loss

    def generate_diverse(
        self,
        x: np.ndarray,
        target_class: float = 1.0,
        n_counterfactuals: int = 5,
        lambda_pred: float = 1.0,
        lambda_dist: float = 0.5,
        lambda_diversity: float = 1.0,
        lr: float = 0.05,
        max_iterations: int = 500,
        diversity_method: str = "dpp"
    ) -> list[CounterfactualResult]:
        """
        Generate a diverse set of counterfactual explanations.

        The optimization minimizes:
            Loss = lambda_pred * prediction_loss
                 + lambda_dist * proximity_loss
                 + lambda_diversity * diversity_loss

        Where diversity_loss encourages the counterfactuals to
        differ from each other (using DPP or pairwise distance).
        """
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

        # Initialize counterfactuals with random perturbations
        # Starting from different points helps find diverse solutions
        cfs = []
        for i in range(n_counterfactuals):
            # Random perturbation in feature-range-normalized space
            noise = torch.randn(1, self.n_features) * 0.1
            cf_init = x_tensor.clone() + noise
            cf_init = cf_init.detach().requires_grad_(True)
            cfs.append(cf_init)

        optimizer = torch.optim.Adam(cfs, lr=lr)

        for iteration in range(max_iterations):
            optimizer.zero_grad()

            # Stack all counterfactuals
            cf_batch = torch.cat(cfs, dim=0)  # (n_cfs, n_features)

            # 1. Prediction loss: each CF should get target prediction
            preds = self.model(cf_batch).squeeze()
            if preds.dim() == 0:
                preds = preds.unsqueeze(0)
            pred_loss = ((preds - target_class) ** 2).mean()

            # 2. Proximity loss: CFs should be close to original
            diffs = cf_batch - x_tensor
            prox_loss = torch.abs(diffs).sum(dim=1).mean()

            # 3. Diversity loss: CFs should be different from each other
            div_loss = self._compute_diversity_loss(cf_batch, diversity_method)

            # Total loss
            total_loss = (lambda_pred * pred_loss +
                         lambda_dist * prox_loss +
                         lambda_diversity * div_loss)

            total_loss.backward()
            optimizer.step()

            # Project: enforce immutable features and ranges
            with torch.no_grad():
                for cf in cfs:
                    for feat_idx, feat_name in enumerate(self.feature_names):
                        # Immutable: reset to original value
                        if feat_name in self.immutable_features:
                            cf[0, feat_idx] = x_tensor[0, feat_idx]

                        # Range constraint
                        if feat_name in self.feature_ranges:
                            low, high = self.feature_ranges[feat_name]
                            cf[0, feat_idx].clamp_(low, high)

                        # Categorical: round to nearest valid value
                        if feat_name in self.categorical_features:
                            valid = self.categorical_features[feat_name]
                            current = cf[0, feat_idx].item()
                            nearest = min(valid, key=lambda v: abs(v - current))
                            cf[0, feat_idx] = nearest

            if (iteration + 1) % 100 == 0:
                with torch.no_grad():
                    all_preds = self.model(torch.cat(cfs, dim=0)).squeeze()
                    success_rate = (torch.abs(all_preds - target_class) < 0.3).float().mean()
                print(f"  Iter {iteration+1}: loss={total_loss.item():.4f}, "
                      f"pred_loss={pred_loss.item():.4f}, "
                      f"success_rate={success_rate.item():.2%}")

        # Construct results
        results = []
        for cf in cfs:
            cf_np = cf.detach().squeeze().numpy()
            x_np = x_tensor.squeeze().numpy()

            with torch.no_grad():
                orig_pred = self.model(x_tensor).squeeze().item()
                cf_pred = self.model(cf).squeeze().item()

            diff = cf_np - x_np
            features_changed = [
                (self.feature_names[i], x_np[i], cf_np[i])
                for i in range(self.n_features)
                if abs(diff[i]) > 0.01
            ]

            results.append(CounterfactualResult(
                original=x_np,
                counterfactual=cf_np,
                original_pred=orig_pred,
                cf_pred=cf_pred,
                target=target_class,
                distance_l1=np.abs(diff).sum(),
                distance_l2=np.sqrt((diff ** 2).sum()),
                n_features_changed=len(features_changed),
                features_changed=features_changed,
                success=abs(cf_pred - target_class) < 0.3
            ))

        # Sort by number of successful, then by L1 distance
        results.sort(key=lambda r: (not r.success, r.distance_l1))

        return results

    def print_diverse_explanations(
        self, results: list[CounterfactualResult]
    ):
        """Pretty-print a set of diverse counterfactual explanations."""
        print(f"\n{'='*60}")
        print(f"DIVERSE COUNTERFACTUAL EXPLANATIONS")
        print(f"{'='*60}")
        print(f"Original prediction: {results[0].original_pred:.3f}")
        print(f"Target prediction: {results[0].target:.3f}")

        for i, cf in enumerate(results):
            status = "SUCCESS" if cf.success else "FAILED"
            print(f"\n--- Counterfactual {i+1} [{status}] ---")
            print(f"  Prediction: {cf.cf_pred:.3f}")
            print(f"  L1 distance: {cf.distance_l1:.4f}")
            print(f"  Features changed ({cf.n_features_changed}):")

            for name, old_val, new_val in cf.features_changed:
                direction = "+" if new_val > old_val else ""
                change = new_val - old_val
                print(f"    {name:20s}: {old_val:8.2f} → {new_val:8.2f} "
                      f"({direction}{change:.2f})")
```

### 2.2 Using the DiCE Library

```python
def dice_library_example():
    """
    Using the dice-ml library for counterfactual generation.

    The library handles many practical concerns:
    - Feature type detection (continuous/categorical)
    - Feature range enforcement
    - Multiple backend support (sklearn, PyTorch, TensorFlow)
    - Built-in diversity methods
    """
    import dice_ml
    from sklearn.ensemble import GradientBoostingClassifier
    import pandas as pd

    # Create loan approval dataset
    np.random.seed(42)
    n = 2000

    data = pd.DataFrame({
        "income": np.random.normal(50000, 15000, n).clip(15000, 150000),
        "debt": np.random.normal(10000, 8000, n).clip(0, 80000),
        "credit_score": np.random.normal(650, 80, n).clip(300, 850).astype(int),
        "employed": np.random.choice([0, 1], n, p=[0.15, 0.85]),
        "years_employed": np.random.exponential(5, n).clip(0, 40).astype(int),
        "age": np.random.normal(38, 12, n).clip(18, 80).astype(int),
    })

    # Create target: loan approved/denied
    score = (
        0.3 * (data["income"] - 40000) / 10000 +
        0.2 * (data["credit_score"] - 600) / 50 +
        -0.2 * data["debt"] / 10000 +
        0.15 * data["employed"] * 2 +
        0.1 * data["years_employed"] / 5 +
        np.random.normal(0, 0.5, n)
    )
    data["approved"] = (score > 0.5).astype(int)

    print(f"Approval rate: {data['approved'].mean():.2%}")

    # Train model
    feature_cols = ["income", "debt", "credit_score", "employed",
                    "years_employed", "age"]
    X = data[feature_cols]
    y = data["approved"]

    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=4, random_state=42
    )
    model.fit(X, y)

    # Set up DiCE
    # Specify the data interface (tells DiCE about feature types)
    dice_data = dice_ml.Data(
        dataframe=data,
        continuous_features=["income", "debt", "credit_score",
                            "years_employed", "age"],
        outcome_name="approved"
    )

    # Specify the model interface
    dice_model = dice_ml.Model(model=model, backend="sklearn")

    # Create the explainer
    explainer = dice_ml.Dice(dice_data, dice_model, method="random")

    # Find a denied applicant to explain
    denied = data[data["approved"] == 0].iloc[0:1]
    denied_features = denied[feature_cols]

    print(f"\nDenied applicant:")
    for col in feature_cols:
        print(f"  {col:20s}: {denied_features[col].values[0]}")

    # Generate diverse counterfactuals
    # features_to_vary: which features can change
    # permitted_range: min/max values
    dice_exp = explainer.generate_counterfactuals(
        denied_features,
        total_CFs=5,
        desired_class="opposite",
        features_to_vary=["income", "debt", "credit_score",
                         "years_employed", "employed"],
        # age is NOT in features_to_vary → immutable
        permitted_range={
            "income": [15000, 150000],
            "debt": [0, 80000],
            "credit_score": [300, 850],
            "years_employed": [0, 40]
        }
    )

    # Display results
    print("\nDiverse Counterfactual Explanations:")
    dice_exp.visualize_as_dataframe(show_only_changes=True)

    return dice_exp
```

---

## 3. Quality Metrics for Counterfactual Explanations

### 3.1 Comprehensive Evaluation Framework

```python
"""
A good counterfactual must satisfy multiple criteria simultaneously.
Here we implement a comprehensive evaluation framework.
"""


class CounterfactualEvaluator:
    """
    Evaluate the quality of counterfactual explanations using
    multiple metrics.
    """

    def __init__(
        self,
        X_train: np.ndarray,
        feature_names: list[str],
        categorical_indices: Optional[list[int]] = None,
        feature_stds: Optional[np.ndarray] = None
    ):
        """
        Parameters:
            X_train: Training data for computing plausibility metrics
            feature_names: Feature names
            categorical_indices: Indices of categorical features
            feature_stds: Standard deviations per feature (for normalization)
        """
        self.X_train = X_train
        self.feature_names = feature_names
        self.categorical_indices = set(categorical_indices or [])
        self.feature_stds = feature_stds if feature_stds is not None \
            else X_train.std(axis=0)
        # Prevent division by zero
        self.feature_stds = np.clip(self.feature_stds, 1e-10, None)

    def proximity_l1(self, x: np.ndarray, cf: np.ndarray) -> float:
        """
        L1 proximity (Manhattan distance), normalized by feature scales.

        Lower is better. Measures the total magnitude of changes.
        Normalized by MAD (Median Absolute Deviation) or std so that
        features on different scales are comparable.
        """
        return np.sum(np.abs(x - cf) / self.feature_stds)

    def proximity_l2(self, x: np.ndarray, cf: np.ndarray) -> float:
        """
        L2 proximity (Euclidean distance), normalized.

        Lower is better. More sensitive to large changes in any single feature.
        """
        return np.sqrt(np.sum(((x - cf) / self.feature_stds) ** 2))

    def sparsity(self, x: np.ndarray, cf: np.ndarray, threshold: float = 0.01) -> int:
        """
        Count the number of features changed.

        Lower is better. Humans prefer explanations that change few features.
        Threshold ignores negligible changes from floating-point optimization.
        """
        return np.sum(np.abs(x - cf) > threshold)

    def plausibility_lof(self, cf: np.ndarray, n_neighbors: int = 20) -> float:
        """
        Plausibility via Local Outlier Factor.

        Measures whether the counterfactual is a realistic data point
        (i.e., lies within the training data distribution).

        LOF score:
        - ≈ 1.0: cf is as dense as its neighbors (plausible)
        - >> 1.0: cf is less dense than neighbors (outlier, implausible)
        - < 1.0: cf is in a denser region than neighbors (very plausible)

        Lower is better (closer to 1.0).
        """
        from sklearn.neighbors import LocalOutlierFactor

        lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
        lof.fit(self.X_train)

        # Score: negative LOF (sklearn convention). More negative = more outlier.
        score = -lof.score_samples(cf.reshape(1, -1))[0]

        return score

    def plausibility_density(
        self, cf: np.ndarray, bandwidth: str = "scott"
    ) -> float:
        """
        Plausibility via kernel density estimation.

        Higher density = more plausible counterfactual.
        """
        from sklearn.neighbors import KernelDensity

        # Normalize data for KDE
        X_normalized = self.X_train / self.feature_stds
        cf_normalized = cf / self.feature_stds

        kde = KernelDensity(bandwidth=0.5, kernel="gaussian")
        kde.fit(X_normalized)

        # Log-density of the counterfactual
        log_density = kde.score_samples(cf_normalized.reshape(1, -1))[0]

        return np.exp(log_density)

    def actionability(
        self,
        x: np.ndarray,
        cf: np.ndarray,
        immutable_features: list[str] = None,
        increasing_only: list[str] = None,
        decreasing_only: list[str] = None
    ) -> dict:
        """
        Check if the counterfactual respects actionability constraints.

        Actionability means: can the person actually make these changes?

        Types of constraints:
        1. Immutable: cannot change (age can't decrease, race can't change)
        2. Increasing only: can only go up (education level)
        3. Decreasing only: can only go down (some debts)
        4. Range constraints: checked during generation, not here

        Returns:
            Dict with violation details
        """
        immutable = immutable_features or []
        increasing = increasing_only or []
        decreasing = decreasing_only or []

        violations = []

        for i, name in enumerate(self.feature_names):
            change = cf[i] - x[i]

            if abs(change) < 0.01:
                continue  # No change, no possible violation

            if name in immutable:
                violations.append(
                    f"VIOLATION: {name} changed by {change:.4f} "
                    f"(immutable feature)"
                )

            if name in increasing and change < -0.01:
                violations.append(
                    f"VIOLATION: {name} decreased by {abs(change):.4f} "
                    f"(can only increase)"
                )

            if name in decreasing and change > 0.01:
                violations.append(
                    f"VIOLATION: {name} increased by {change:.4f} "
                    f"(can only decrease)"
                )

        is_actionable = len(violations) == 0

        return {
            "is_actionable": is_actionable,
            "violations": violations,
            "n_violations": len(violations)
        }

    def evaluate_full(
        self,
        x: np.ndarray,
        cf: np.ndarray,
        model_fn=None,
        target_class: float = None,
        immutable_features: list[str] = None,
        increasing_only: list[str] = None
    ) -> dict:
        """
        Run the full evaluation suite on a counterfactual.
        """
        results = {
            "proximity_l1": self.proximity_l1(x, cf),
            "proximity_l2": self.proximity_l2(x, cf),
            "sparsity": self.sparsity(x, cf),
            "plausibility_lof": self.plausibility_lof(cf),
        }

        # Validity: does the CF actually achieve the target prediction?
        if model_fn is not None and target_class is not None:
            cf_pred = model_fn(cf.reshape(1, -1))[0]
            results["validity"] = abs(cf_pred - target_class) < 0.3
            results["cf_prediction"] = cf_pred

        # Actionability
        action_result = self.actionability(
            x, cf, immutable_features, increasing_only
        )
        results["is_actionable"] = action_result["is_actionable"]
        results["actionability_violations"] = action_result["violations"]

        # Print report
        print("Counterfactual Quality Report")
        print("=" * 50)
        print(f"  Proximity (L1, normalized): {results['proximity_l1']:.4f}")
        print(f"  Proximity (L2, normalized): {results['proximity_l2']:.4f}")
        print(f"  Sparsity (features changed): {results['sparsity']}")
        print(f"  Plausibility (LOF score):   {results['plausibility_lof']:.4f}")
        if "validity" in results:
            status = "VALID" if results["validity"] else "INVALID"
            print(f"  Validity:                   {status} "
                  f"(pred={results['cf_prediction']:.3f})")
        print(f"  Actionable:                 {results['is_actionable']}")
        if not results["is_actionable"]:
            for v in action_result["violations"]:
                print(f"    {v}")

        return results
```

---

## 4. Causal Constraints

### 4.1 Why Causal Constraints Matter

```python
"""
Causal Constraints in Counterfactual Explanations

Problem: Standard counterfactuals can suggest IMPOSSIBLE changes
because they ignore causal relationships between features.

Example without causal constraints:
    "Increase your income by $20,000 while keeping savings the same."
    But in reality, higher income usually means higher savings!

Example with causal constraints:
    "Increase your income by $20,000" → savings also increases by ~$4,000
    (because we model the causal effect of income on savings)

Causal counterfactual = standard counterfactual + propagate changes
through the causal graph using structural equations.
"""


class CausalCounterfactual:
    """
    Generate counterfactuals that respect causal structure.

    Uses a Structural Causal Model (SCM) to propagate changes
    from intervened features to their causal descendants.
    """

    def __init__(
        self,
        model: nn.Module,
        feature_names: list[str],
        causal_graph: dict[str, list[str]],
        structural_equations: dict[str, callable]
    ):
        """
        Parameters:
            model: Trained classifier
            feature_names: Feature names
            causal_graph: {feature: [list of parents]}
                         Defines which features causally influence which.
            structural_equations: {feature: function(parent_values) -> value}
                                Functions that compute a feature's value
                                given its parents' values.

        Example:
            causal_graph = {
                "age": [],
                "education": [],
                "income": ["age", "education"],
                "savings": ["income", "age"],
                "credit_score": ["income", "debt"]
            }
            structural_equations = {
                "income": lambda age, edu: 20000 + 500*age + 15000*edu,
                "savings": lambda income, age: 0.2*income + 100*age,
                "credit_score": lambda income, debt: 500 + 0.003*income - 0.01*debt
            }
        """
        self.model = model
        self.model.eval()
        self.feature_names = feature_names
        self.name_to_idx = {n: i for i, n in enumerate(feature_names)}
        self.causal_graph = causal_graph
        self.structural_equations = structural_equations

    def _topological_sort(self) -> list[str]:
        """Sort features in causal (topological) order."""
        visited = set()
        order = []

        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for parent in self.causal_graph.get(node, []):
                dfs(parent)
            order.append(node)

        for feat in self.feature_names:
            dfs(feat)

        return order

    def propagate_intervention(
        self,
        x: np.ndarray,
        interventions: dict[str, float],
        noise_scale: float = 0.0
    ) -> np.ndarray:
        """
        Propagate interventions through the causal graph.

        When we intervene on feature A (set it to a new value),
        we must update all of A's descendants according to the
        structural equations.

        Parameters:
            x: Original instance
            interventions: {feature_name: new_value} for directly changed features
            noise_scale: Add noise to propagated values (for diversity)

        Returns:
            Causally consistent instance with interventions propagated
        """
        x_new = x.copy()

        # Apply direct interventions
        for feat_name, new_value in interventions.items():
            idx = self.name_to_idx[feat_name]
            x_new[idx] = new_value

        # Propagate through the causal graph in topological order
        topo_order = self._topological_sort()

        for feat_name in topo_order:
            if feat_name in interventions:
                # This feature was directly intervened on; skip propagation
                continue

            if feat_name not in self.structural_equations:
                # No structural equation; keep the original value
                continue

            parents = self.causal_graph.get(feat_name, [])
            if not parents:
                continue

            # Check if any ancestor was intervened on
            has_intervened_ancestor = False
            stack = list(parents)
            visited = set()
            while stack:
                p = stack.pop()
                if p in visited:
                    continue
                visited.add(p)
                if p in interventions:
                    has_intervened_ancestor = True
                    break
                stack.extend(self.causal_graph.get(p, []))

            if has_intervened_ancestor:
                # Recompute this feature using the structural equation
                parent_values = {p: x_new[self.name_to_idx[p]] for p in parents}
                new_value = self.structural_equations[feat_name](**parent_values)

                # Add noise for diversity
                if noise_scale > 0:
                    new_value += np.random.normal(0, noise_scale)

                x_new[self.name_to_idx[feat_name]] = new_value

        return x_new

    def generate(
        self,
        x: np.ndarray,
        target_class: float = 1.0,
        actionable_features: list[str] = None,
        n_candidates: int = 100,
        perturbation_scale: float = 0.3
    ) -> list[CounterfactualResult]:
        """
        Generate causal counterfactuals by:
        1. Sampling interventions on actionable features
        2. Propagating through the causal graph
        3. Checking which interventions achieve the target prediction
        4. Returning the best (smallest change) valid counterfactuals
        """
        if actionable_features is None:
            # Default: features with no parents are actionable
            actionable_features = [
                f for f in self.feature_names
                if not self.causal_graph.get(f, [])
            ]

        valid_cfs = []

        for _ in range(n_candidates):
            # Sample random interventions on actionable features
            interventions = {}
            n_features_to_change = np.random.randint(1, len(actionable_features) + 1)
            features_to_change = np.random.choice(
                actionable_features, n_features_to_change, replace=False
            )

            for feat_name in features_to_change:
                idx = self.name_to_idx[feat_name]
                current_value = x[idx]
                # Sample perturbation proportional to feature scale
                perturbation = np.random.normal(0, perturbation_scale) * \
                              abs(current_value) if current_value != 0 else \
                              np.random.normal(0, 1)
                interventions[feat_name] = current_value + perturbation

            # Propagate through causal graph
            x_cf = self.propagate_intervention(x, interventions)

            # Check prediction
            with torch.no_grad():
                x_cf_tensor = torch.tensor(x_cf, dtype=torch.float32).unsqueeze(0)
                pred = self.model(x_cf_tensor).squeeze().item()

            if abs(pred - target_class) < 0.3:
                diff = x_cf - x
                features_changed = [
                    (self.feature_names[i], x[i], x_cf[i])
                    for i in range(len(self.feature_names))
                    if abs(diff[i]) > 0.01
                ]

                valid_cfs.append(CounterfactualResult(
                    original=x,
                    counterfactual=x_cf,
                    original_pred=self.model(
                        torch.tensor(x, dtype=torch.float32).unsqueeze(0)
                    ).squeeze().item(),
                    cf_pred=pred,
                    target=target_class,
                    distance_l1=np.abs(diff).sum(),
                    distance_l2=np.sqrt((diff ** 2).sum()),
                    n_features_changed=len(features_changed),
                    features_changed=features_changed,
                    success=True
                ))

        # Sort by L1 distance (closest first)
        valid_cfs.sort(key=lambda r: r.distance_l1)

        print(f"Found {len(valid_cfs)} valid causal counterfactuals "
              f"out of {n_candidates} candidates")

        return valid_cfs[:10]  # Return top 10
```

---

## 5. Contrastive Explanations

### 5.1 "Why A Instead of B?"

```python
"""
Contrastive Explanations: "Why did the model predict A instead of B?"

Standard counterfactuals answer: "What minimal change gives prediction B?"
Contrastive explanations answer: "What is the minimal DIFFERENCE between
instances that get prediction A vs prediction B?"

This is psychologically grounded: humans naturally think in contrasts.
"Why was I denied instead of approved?" is more natural than
"Why was I denied?"

Two approaches:
1. Pertinent Positives (PP): features whose PRESENCE is necessary for A
   "You were approved BECAUSE of your high credit score."
2. Pertinent Negatives (PN): features whose ABSENCE distinguishes A from B
   "You were approved INSTEAD OF denied because your debt is low."
"""


class ContrastiveExplainer:
    """
    Generate contrastive explanations of the form
    "Why class A instead of class B?"
    """

    def __init__(self, model: nn.Module, feature_names: list[str]):
        self.model = model
        self.model.eval()
        self.feature_names = feature_names

    def explain_contrast(
        self,
        x: np.ndarray,
        predicted_class: int,
        contrast_class: int,
        lr: float = 0.01,
        max_iterations: int = 500,
        lambda_sparse: float = 0.1
    ) -> dict:
        """
        Find the minimal change that would flip the prediction from
        predicted_class to contrast_class.

        This directly answers "Why A instead of B?" by showing what
        would need to change to get B.

        Returns:
            Dict with pertinent negatives (what keeps x in class A)
            and the contrastive counterfactual (smallest change to get B)
        """
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        x_cf = x_tensor.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([x_cf], lr=lr)

        for iteration in range(max_iterations):
            optimizer.zero_grad()

            logits = self.model(x_cf)

            # Push toward contrast class
            if logits.dim() == 2 and logits.shape[1] > 1:
                # Multi-class: maximize contrast class logit
                pred_loss = -logits[0, contrast_class] + logits[0, predicted_class]
            else:
                # Binary: push toward contrast
                pred_loss = (logits.squeeze() - contrast_class) ** 2

            # Sparsity: change as few features as possible
            diff = x_cf - x_tensor
            sparsity_loss = lambda_sparse * torch.abs(diff).sum()

            total_loss = pred_loss + sparsity_loss
            total_loss.backward()
            optimizer.step()

        # Analyze the contrastive explanation
        cf_np = x_cf.detach().squeeze().numpy()
        diff = cf_np - x

        # Pertinent negatives: features where the change flips the class
        # (features that distinguish class A from class B)
        pertinent_negatives = []
        for i in range(len(self.feature_names)):
            if abs(diff[i]) > 0.01:
                pertinent_negatives.append({
                    "feature": self.feature_names[i],
                    "current_value": x[i],
                    "contrast_value": cf_np[i],
                    "change": diff[i],
                    "importance": abs(diff[i])
                })

        pertinent_negatives.sort(key=lambda p: p["importance"], reverse=True)

        # Generate natural language explanation
        explanation_parts = []
        for pn in pertinent_negatives[:3]:  # Top 3 distinguishing features
            name = pn["feature"]
            current = pn["current_value"]
            contrast = pn["contrast_value"]

            if contrast > current:
                explanation_parts.append(
                    f"your {name} ({current:.1f}) would need to be "
                    f"{contrast:.1f} or higher"
                )
            else:
                explanation_parts.append(
                    f"your {name} ({current:.1f}) would need to be "
                    f"{contrast:.1f} or lower"
                )

        nl_explanation = (
            f"The model predicted class {predicted_class} instead of "
            f"class {contrast_class} because: to get class {contrast_class}, "
            + ", and ".join(explanation_parts) + "."
        )

        return {
            "pertinent_negatives": pertinent_negatives,
            "counterfactual": cf_np,
            "natural_language": nl_explanation
        }
```

---

## 6. Practical: Loan Approval Counterfactuals

### 6.1 Complete End-to-End Pipeline

```python
def loan_approval_counterfactual_demo():
    """
    Complete practical example: generating actionable counterfactual
    explanations for a loan approval model.

    This demonstrates:
    1. Training a loan approval model
    2. Generating standard counterfactuals (Wachter)
    3. Generating diverse counterfactuals (DiCE-style)
    4. Generating causal counterfactuals
    5. Evaluating all explanations for quality
    6. Producing customer-facing recommendations
    """
    import torch
    import torch.nn as nn
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    np.random.seed(42)
    torch.manual_seed(42)

    # --- Step 1: Create realistic loan data ---
    print("Step 1: Generating loan application data")
    print("=" * 60)

    n = 5000
    age = np.random.normal(38, 12, n).clip(18, 75)
    education = np.random.choice([0, 1, 2, 3], n, p=[0.2, 0.4, 0.3, 0.1])
    # Education: 0=HS, 1=Bachelor, 2=Master, 3=PhD

    # Causal structure: education & age → income → savings
    income = (20000 + 500 * age + 12000 * education +
              np.random.normal(0, 8000, n)).clip(15000)
    savings = (0.15 * income + 100 * age +
               np.random.normal(0, 5000, n)).clip(0)
    debt = np.random.exponential(8000, n).clip(0, 60000)
    credit_score = (500 + 0.002 * income - 0.003 * debt +
                    2 * education + np.random.normal(0, 30, n)).clip(300, 850)
    employed = (np.random.random(n) < (0.7 + 0.05 * education)).astype(float)

    X = np.column_stack([income, savings, debt, credit_score, employed, age, education])
    feature_names = ["income", "savings", "debt", "credit_score",
                     "employed", "age", "education"]

    # Approval decision
    score = (0.25 * (income - 40000) / 15000 +
             0.15 * (savings - 10000) / 10000 +
             -0.20 * (debt - 10000) / 10000 +
             0.20 * (credit_score - 600) / 100 +
             0.10 * employed +
             0.05 * (age - 25) / 20 +
             np.random.normal(0, 0.3, n))
    y = (score > 0.3).astype(float)

    print(f"  Samples: {n}, Approval rate: {y.mean():.2%}")

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Step 2: Train model ---
    print("\nStep 2: Training loan approval model")
    print("=" * 60)

    class LoanModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(7, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.net(x)

    model = LoanModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        preds = model(X_tensor).squeeze()
        loss = criterion(preds, y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        accuracy = ((model(X_tensor).squeeze() > 0.5) == y_tensor).float().mean()
    print(f"  Model accuracy: {accuracy:.4f}")

    # --- Step 3: Find a denied applicant ---
    denied_indices = np.where(y == 0)[0]
    idx = denied_indices[42]  # Pick a specific denied applicant
    x_original = X_scaled[idx]
    x_raw = X[idx]

    print(f"\nStep 3: Denied applicant profile")
    print("=" * 60)
    for i, name in enumerate(feature_names):
        print(f"  {name:15s}: {x_raw[i]:>10.1f}")

    with torch.no_grad():
        orig_pred = model(torch.tensor(x_original, dtype=torch.float32).unsqueeze(0))
    print(f"  Prediction: {orig_pred.item():.3f} (Denied)")

    # --- Step 4: Standard Wachter counterfactual ---
    print(f"\nStep 4: Standard Counterfactual (Wachter)")
    print("=" * 60)

    wachter = WachterCounterfactual(
        model=model,
        feature_names=feature_names,
        feature_ranges={
            "income": (-3, 5),     # Allow increase but not too extreme
            "savings": (-3, 5),
            "debt": (-5, 3),
            "credit_score": (-3, 3),
            "employed": (-2, 2),
            "age": (-2, 2),        # Limited age change
            "education": (-2, 2),
        }
    )

    wachter_result = wachter.generate(
        x_original,
        target_class=1.0,
        lambda_param=0.5,
        lr=0.01,
        max_iterations=1000,
        distance_metric="elastic"
    )

    # Convert changes back to original scale for readability
    print(f"  Success: {wachter_result.success}")
    if wachter_result.success:
        cf_raw = scaler.inverse_transform(
            wachter_result.counterfactual.reshape(1, -1)
        )[0]
        print(f"  Counterfactual prediction: {wachter_result.cf_pred:.3f}")
        print(f"  Changes needed:")
        for i, name in enumerate(feature_names):
            change = cf_raw[i] - x_raw[i]
            if abs(change) > 0.5:
                direction = "+" if change > 0 else ""
                print(f"    {name:15s}: {x_raw[i]:>10.1f} → "
                      f"{cf_raw[i]:>10.1f} ({direction}{change:.1f})")

    # --- Step 5: Diverse counterfactuals ---
    print(f"\nStep 5: Diverse Counterfactuals (DiCE-style)")
    print("=" * 60)

    dice = DiCEExplainer(
        model=model,
        feature_names=feature_names,
        feature_ranges={name: (-4, 4) for name in feature_names},
        immutable_features=["age"],  # Can't change age
        categorical_features={"employed": [0, 1], "education": [0, 1, 2, 3]}
    )

    diverse_results = dice.generate_diverse(
        x_original,
        target_class=1.0,
        n_counterfactuals=5,
        lambda_pred=2.0,
        lambda_dist=0.3,
        lambda_diversity=1.0,
        max_iterations=500
    )

    for i, cf in enumerate(diverse_results[:3]):  # Show top 3
        if cf.success:
            cf_raw = scaler.inverse_transform(
                cf.counterfactual.reshape(1, -1)
            )[0]
            print(f"\n  Option {i+1} (pred={cf.cf_pred:.3f}):")
            for j, name in enumerate(feature_names):
                change = cf_raw[j] - x_raw[j]
                if abs(change) > 0.5:
                    print(f"    {name:15s}: {x_raw[j]:>10.1f} → {cf_raw[j]:>10.1f}")

    # --- Step 6: Causal counterfactuals ---
    print(f"\nStep 6: Causal Counterfactuals")
    print("=" * 60)

    # Define causal structure in the SCALED feature space
    causal_graph = {
        "income": ["age", "education"],
        "savings": ["income", "age"],
        "debt": [],
        "credit_score": ["income", "debt"],
        "employed": [],
        "age": [],
        "education": [],
    }

    # Simplified structural equations (in scaled space)
    structural_equations = {
        "income": lambda age, education: 0.3 * age + 0.5 * education,
        "savings": lambda income, age: 0.4 * income + 0.2 * age,
        "credit_score": lambda income, debt: 0.3 * income - 0.2 * debt,
    }

    causal_cf = CausalCounterfactual(
        model=model,
        feature_names=feature_names,
        causal_graph=causal_graph,
        structural_equations=structural_equations
    )

    causal_results = causal_cf.generate(
        x_original,
        target_class=1.0,
        actionable_features=["education", "employed", "debt"],
        n_candidates=500,
        perturbation_scale=0.5
    )

    if causal_results:
        print(f"  Found {len(causal_results)} causal counterfactuals")
        best = causal_results[0]
        best_raw = scaler.inverse_transform(
            best.counterfactual.reshape(1, -1)
        )[0]
        print(f"\n  Best causal counterfactual (pred={best.cf_pred:.3f}):")
        for i, name in enumerate(feature_names):
            change = best_raw[i] - x_raw[i]
            if abs(change) > 0.5:
                note = " (propagated)" if name in ["income", "savings",
                                                     "credit_score"] else ""
                print(f"    {name:15s}: {x_raw[i]:>10.1f} → "
                      f"{best_raw[i]:>10.1f}{note}")

    # --- Step 7: Quality evaluation ---
    print(f"\nStep 7: Counterfactual Quality Evaluation")
    print("=" * 60)

    evaluator = CounterfactualEvaluator(
        X_train=X_scaled,
        feature_names=feature_names
    )

    if wachter_result.success:
        print("\nWachter Counterfactual:")
        evaluator.evaluate_full(
            x=x_original,
            cf=wachter_result.counterfactual,
            model_fn=lambda z: model(
                torch.tensor(z, dtype=torch.float32)
            ).detach().numpy().squeeze(),
            target_class=1.0,
            immutable_features=["age"],
            increasing_only=["education"]
        )

    # --- Step 8: Customer-facing recommendation ---
    print(f"\nStep 8: Customer-Facing Recommendation")
    print("=" * 60)

    if wachter_result.success:
        cf_raw = scaler.inverse_transform(
            wachter_result.counterfactual.reshape(1, -1)
        )[0]

        print("\n  Dear Applicant,")
        print()
        print("  Your loan application was not approved at this time.")
        print("  Based on our analysis, here are steps that could")
        print("  improve your chances of approval:")
        print()

        recommendations = []
        for i, name in enumerate(feature_names):
            change = cf_raw[i] - x_raw[i]
            if name in ["age"]:
                continue  # Don't recommend changing age
            if abs(change) > 0.5:
                if name == "income":
                    recommendations.append(
                        f"  - Increase your annual income to at least "
                        f"${cf_raw[i]:,.0f}"
                    )
                elif name == "debt":
                    recommendations.append(
                        f"  - Reduce your outstanding debt to "
                        f"${cf_raw[i]:,.0f} or below"
                    )
                elif name == "credit_score":
                    recommendations.append(
                        f"  - Improve your credit score to at least "
                        f"{cf_raw[i]:.0f}"
                    )
                elif name == "savings":
                    recommendations.append(
                        f"  - Build your savings to at least "
                        f"${cf_raw[i]:,.0f}"
                    )

        for rec in recommendations:
            print(rec)

        print()
        print("  These recommendations are based on your specific profile")
        print("  and represent the minimal changes that would likely result")
        print("  in approval. Meeting any combination of these targets")
        print("  would strengthen your application.")

    return {
        "wachter": wachter_result,
        "diverse": diverse_results,
        "causal": causal_results
    }


loan_approval_counterfactual_demo()
```

---

## Summary

- **Counterfactual explanations** describe the minimal change to an input that changes the model's prediction. They are inherently actionable and satisfy algorithmic recourse requirements.
- **Wachter et al. (2017)** formulates counterfactual generation as an optimization problem balancing prediction loss (reach the target class) and distance loss (minimize changes). L1 distance encourages sparse changes; L2 encourages smooth ones.
- **DiCE** (Mothilal et al. 2020) generates diverse counterfactual sets using determinantal point processes, giving users multiple paths to change the outcome. Diversity prevents over-reliance on a single explanation.
- **Quality metrics** for counterfactuals include proximity (how close), sparsity (how few features), plausibility (is it realistic via LOF/KDE), validity (does it achieve the target), and actionability (are the changes feasible).
- **Causal constraints** ensure that counterfactuals respect real-world dependencies: when income increases, savings should also increase. This requires a structural causal model to propagate interventions through the causal graph.
- **Contrastive explanations** answer "why A instead of B?" by identifying the minimal features that distinguish the two classes, matching how humans naturally reason about decisions.

---

## Exercises

### Exercise 1: Basic Counterfactual Generation (Beginner)

Train a logistic regression model on the UCI Adult Income dataset (predict income >$50k). For 10 individuals earning <$50k, generate counterfactual explanations using the Wachter method. Compare L1 and L2 distance metrics: which produces sparser explanations? Which produces smaller total change? Visualize the counterfactuals in feature space alongside the decision boundary.

### Exercise 2: Diverse Counterfactuals with DiCE (Intermediate)

Using the dice-ml library on a credit default dataset, generate 5 diverse counterfactuals for each of 20 denied applicants. Measure the pairwise diversity of the counterfactual sets (average pairwise L1 distance). Compare the DPP diversity method against random initialization. How many distinct "strategies" (clusters of similar counterfactuals) emerge across all applicants?

### Exercise 3: Counterfactual Quality Audit (Intermediate)

Implement the full CounterfactualEvaluator on counterfactuals generated by three methods: Wachter, DiCE, and a random baseline (random perturbations that happen to cross the decision boundary). Compare all three on: proximity, sparsity, plausibility (LOF), and actionability. Create a radar chart showing the quality tradeoffs of each method.

### Exercise 4: Causal Counterfactuals (Advanced)

Define a causal graph for the German Credit dataset with at least 5 causal edges (e.g., job type causes income, age causes employment duration). Implement the CausalCounterfactual class and generate explanations for 10 denied applicants. Compare causal counterfactuals against standard (acausal) ones: how often do standard counterfactuals violate causal constraints? Measure the plausibility improvement from incorporating causality.

### Exercise 5: Contrastive Explanation System (Advanced)

Build a contrastive explanation system for a multi-class disease diagnosis model (at least 5 diseases). For each prediction, generate contrastive explanations against the top 2 alternative diagnoses: "Why disease A instead of B?" and "Why disease A instead of C?" Evaluate whether the contrastive features are medically meaningful by consulting a medical textbook or domain expert. Implement a user interface that allows doctors to ask "Why not [specific disease]?" interactively.

---

[Previous: Concept-Based Explanations](./07_Concept_Based_Explanations.md) | [Overview](./00_Overview.md) | [Next: Causal Inference for Interpretability](./09_Causal_Inference_for_Interpretability.md)

---

**License**: CC BY-NC 4.0
