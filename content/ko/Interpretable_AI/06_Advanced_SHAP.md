# 레슨 6: 고급 SHAP(Advanced SHAP)

[이전: 프로빙과 표현 분석](./05_Probing_and_Representation_Analysis.md) | [다음: 개념 기반 설명](./07_Concept_Based_Explanations.md)

---

## 학습 목표

- 딥 신경망을 위한 섀플리 값(Shapley Values)의 효율적 근사로서 DeepSHAP과 GradientSHAP을 이해한다
- SHAP 상호작용 값(Interaction Values)을 계산하고 해석하여 쌍별 특성 의존성을 식별한다
- 비대칭 섀플리 값(Asymmetric Shapley Values)과 인과적 SHAP(Causal SHAP)을 적용하여 인과 구조를 설명에 통합한다
- 프로덕션 규모의 SHAP을 위한 계산 최적화(배경 샘플링, 배치 처리, GPU 가속)를 구현한다
- TreeSHAP의 병리 현상과 특성 상관관계 편향에 대한 Sundararajan & Najmi의 비판을 비판적으로 평가한다

---

> **사전 요구 사항 참고**: SHAP 기초(TreeExplainer, KernelExplainer, 포스 플롯, 요약 플롯, 의존성 플롯)에 대해서는 [머신러닝 레슨 16: 모델 설명 가능성](../Machine_Learning/16_Model_Explainability.md)을 참조하라. 이 레슨은 섀플리 값 이론, KernelSHAP 알고리즘, 기본 SHAP 시각화에 익숙하다고 가정한다. 이러한 기초 위에 딥러닝 특화 방법, 상호작용 효과, 인과적 확장, 계산 최적화를 다룬다.

---

## 1. DeepSHAP: DeepLIFT 기반 섀플리 값

### 1.1 DeepLIFT에서 DeepSHAP으로

```python
"""
DeepSHAP connects two ideas:
1. DeepLIFT (Shrikumar et al. 2017): Attributes predictions by comparing
   activations to a reference (baseline) activation, propagating
   "contribution scores" backward through the network.
2. Shapley values: The unique fair allocation of a prediction among features.

DeepSHAP = DeepLIFT + Shapley value averaging over multiple baselines.

Key insight: DeepLIFT with a single reference computes an approximation
to Shapley values under specific assumptions. By averaging DeepLIFT
contributions over a set of background (reference) samples, we get
a better Shapley value approximation.

Why DeepSHAP over KernelSHAP for deep models?
- KernelSHAP: model-agnostic, accurate, but O(2^n) worst case
- DeepSHAP: leverages network structure, runs in a single backward pass,
  but makes approximations (composition assumption)
"""

import torch
import torch.nn as nn
import numpy as np
import shap


# First, let's understand DeepLIFT's contribution rules
class DeepLIFTExplainer:
    """
    Simplified DeepLIFT implementation to illustrate the core ideas.

    DeepLIFT defines 'contribution' relative to a reference:
        delta_y = y(x) - y(x_ref)  (output difference)
        delta_x_i = x_i - x_ref_i  (input difference)

    It decomposes: delta_y = sum_i C(delta_x_i)
    where C(delta_x_i) is the contribution of feature i.

    The Rescale Rule (simplest):
        For a neuron z = f(a1*x1 + a2*x2 + b):
        C(delta_x_i) = (a_i * delta_x_i / delta_z) * delta_f(z)

    The RevealCancel Rule (handles positive/negative contributions):
        Separates positive and negative contributions for non-linear
        activations to avoid cancellation artifacts.
    """

    def __init__(self, model: nn.Module, reference: torch.Tensor):
        """
        Parameters:
            model: PyTorch neural network
            reference: Baseline input (e.g., zero vector, mean of training data)
                      Shape: (1, n_features) or (n_background, n_features)
        """
        self.model = model
        self.model.eval()
        self.reference = reference

    def explain_rescale(self, x: torch.Tensor) -> np.ndarray:
        """
        Compute DeepLIFT attributions using the Rescale rule.

        This is the simplest DeepLIFT variant. For each neuron:
        - Compute activation difference: delta = activation(x) - activation(ref)
        - Distribute the difference proportionally to input differences

        Limitation: The rescale rule can give zero attribution to features
        that contribute but whose contributions cancel out.
        """
        x.requires_grad_(True)

        # Forward pass for input
        output_x = self.model(x)
        # Forward pass for reference
        with torch.no_grad():
            output_ref = self.model(self.reference)

        # Output difference
        delta_output = output_x - output_ref.mean(dim=0)

        # Backward pass to get gradients
        # DeepLIFT's rescale rule at ReLU nodes is equivalent to:
        # gradient * (delta_x) / (delta_activation + epsilon)
        # For a linear network, this simplifies to regular gradients
        delta_output.sum().backward()

        # Attribution = gradient * (input - reference)
        # This is the "gradient x input difference" formulation
        attribution = x.grad * (x - self.reference.mean(dim=0))

        return attribution.detach().numpy()
```

### 1.2 다중 배경 샘플을 사용한 DeepSHAP

```python
def demonstrate_deepshap():
    """
    Show how DeepSHAP uses multiple background samples to compute
    Shapley-compatible attributions.

    DeepSHAP averages DeepLIFT attributions over a set of background
    (reference) samples drawn from the training distribution.

    E[DeepLIFT(x, x_ref)] over x_ref ~ background ≈ SHAP values

    The background set is CRITICAL for DeepSHAP quality:
    - Too few backgrounds: noisy, unstable attributions
    - Too many: slow computation
    - Non-representative backgrounds: biased attributions

    Recommended: 100-1000 background samples from training data.
    """

    # Define a simple neural network for demonstration
    class SimpleNN(nn.Module):
        def __init__(self, input_dim: int = 10, hidden_dim: int = 64):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )

        def forward(self, x):
            return self.network(x)

    # Create and train model (simplified)
    torch.manual_seed(42)
    model = SimpleNN(input_dim=10)

    # Training data
    X_train = torch.randn(1000, 10)
    y_train = (X_train[:, 0] * 2 + X_train[:, 1] ** 2 +
               X_train[:, 2] * X_train[:, 3] +
               torch.randn(1000) * 0.1)

    # Train the model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(100):
        optimizer.zero_grad()
        pred = model(X_train).squeeze()
        loss = nn.MSELoss()(pred, y_train)
        loss.backward()
        optimizer.step()

    model.eval()

    # --- DeepSHAP with SHAP library ---
    # Background: random subset of training data
    # The library handles the DeepLIFT + Shapley averaging internally
    background = X_train[:200]  # 200 background samples

    explainer = shap.DeepExplainer(model, background)

    # Explain a batch of test instances
    X_test = torch.randn(50, 10)
    shap_values = explainer.shap_values(X_test)

    print(f"SHAP values shape: {np.array(shap_values).shape}")
    # Shape: (50, 10) — one attribution per feature per instance

    # Verify the completeness property:
    # sum(shap_values[i]) + E[f(x)] ≈ f(x_i)
    with torch.no_grad():
        predictions = model(X_test).numpy().squeeze()
        base_value = model(background).mean().item()

    for i in range(5):
        shap_sum = np.sum(shap_values[i]) + base_value
        actual = predictions[i]
        print(f"Instance {i}: SHAP sum = {shap_sum:.4f}, "
              f"Prediction = {actual:.4f}, "
              f"Diff = {abs(shap_sum - actual):.6f}")

    return shap_values, explainer


shap_values, explainer = demonstrate_deepshap()
```

### 1.3 배경 선택 전략

```python
"""
Background selection is the most important hyperparameter for DeepSHAP.

Different strategies and their tradeoffs:
"""


def compare_background_strategies(
    model: nn.Module,
    X_train: torch.Tensor,
    X_test: torch.Tensor
) -> dict:
    """
    Compare different background selection strategies for DeepSHAP.

    Strategies:
    1. Random sample: Simple, common, but may miss distribution edges
    2. K-means centroids: Covers the space more evenly
    3. Zero reference: Single baseline (original DeepLIFT approach)
    4. Mean reference: Single baseline at training mean
    5. Stratified sample: Preserves class balance
    """
    import shap
    from sklearn.cluster import KMeans

    results = {}

    # Strategy 1: Random sample (most common)
    random_bg = X_train[np.random.choice(len(X_train), 200, replace=False)]
    exp_random = shap.DeepExplainer(model, random_bg)
    sv_random = exp_random.shap_values(X_test[:10])
    results["random_200"] = np.array(sv_random)

    # Strategy 2: K-means centroids (better coverage, fewer samples)
    # Advantage: covers the data manifold with fewer samples
    # Disadvantage: centroids may not be realistic data points
    kmeans = KMeans(n_clusters=50, random_state=42, n_init=10)
    kmeans.fit(X_train.numpy())
    kmeans_bg = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    exp_kmeans = shap.DeepExplainer(model, kmeans_bg)
    sv_kmeans = exp_kmeans.shap_values(X_test[:10])
    results["kmeans_50"] = np.array(sv_kmeans)

    # Strategy 3: Single zero reference (fastest, least accurate)
    # Warning: using zeros as reference assumes "absence" is zero,
    # which is often not meaningful for real features
    zero_bg = torch.zeros(1, X_train.shape[1])
    exp_zero = shap.DeepExplainer(model, zero_bg)
    sv_zero = exp_zero.shap_values(X_test[:10])
    results["zero"] = np.array(sv_zero)

    # Strategy 4: Training mean reference
    mean_bg = X_train.mean(dim=0, keepdim=True)
    exp_mean = shap.DeepExplainer(model, mean_bg)
    sv_mean = exp_mean.shap_values(X_test[:10])
    results["mean"] = np.array(sv_mean)

    # Compare stability across strategies
    print("SHAP value agreement across background strategies:")
    print("=" * 60)

    strategies = list(results.keys())
    for i in range(len(strategies)):
        for j in range(i + 1, len(strategies)):
            s1, s2 = strategies[i], strategies[j]
            # Pearson correlation between SHAP values
            corr = np.corrcoef(
                results[s1].flatten(),
                results[s2].flatten()
            )[0, 1]
            print(f"  {s1:12s} vs {s2:12s}: r = {corr:.4f}")

    # Key takeaway: if results vary dramatically across strategies,
    # the explanations are not reliable. This is a red flag.

    return results
```

---

## 2. GradientSHAP: 기대 그래디언트(Expected Gradients)

### 2.1 기대 그래디언트 정식화

```python
"""
GradientSHAP (Erion et al. 2021) is based on the Expected Gradients
formulation, which connects Integrated Gradients to Shapley values.

Integrated Gradients (from Lesson 2):
    IG_i(x) = (x_i - x'_i) * integral_0^1 (dF/dx_i)(x' + t*(x - x')) dt

Expected Gradients:
    EG_i(x) = E_{x' ~ D, t ~ U[0,1]} [ (x_i - x'_i) * (dF/dx_i)(x' + t*(x - x')) ]

The key insight: if we replace the single baseline x' with an expectation
over training distribution D, Integrated Gradients becomes equivalent
to Shapley values (under certain assumptions about feature independence).

GradientSHAP = practical implementation of Expected Gradients:
1. Sample background points x' from training data
2. For each x', sample random interpolation point t ~ U[0,1]
3. Compute gradient at the interpolated point
4. Multiply gradient by (x - x')
5. Average across all (x', t) samples

Why GradientSHAP over DeepSHAP?
- GradientSHAP uses actual gradients (no approximation at non-linearities)
- DeepSHAP uses DeepLIFT rules (faster but makes composition assumptions)
- GradientSHAP is more theoretically grounded but noisier
"""

import torch
import torch.nn as nn
import numpy as np


class GradientSHAPExplainer:
    """
    GradientSHAP implementation from first principles.

    This implements the expected gradients formula by:
    1. Sampling reference points from a background dataset
    2. Computing gradients along random interpolation paths
    3. Averaging to get Shapley value approximations
    """

    def __init__(
        self,
        model: nn.Module,
        background: torch.Tensor,
        n_samples: int = 200
    ):
        """
        Parameters:
            model: PyTorch model
            background: Background dataset, shape (N, d)
            n_samples: Number of (background, alpha) pairs to sample
                       More samples = less noise but slower
        """
        self.model = model
        self.model.eval()
        self.background = background
        self.n_samples = n_samples

    def explain(self, x: torch.Tensor) -> np.ndarray:
        """
        Compute GradientSHAP values for input x.

        Returns:
            SHAP values, shape (d,) where d = number of features
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        d = x.shape[1]
        attributions = np.zeros(d)

        for _ in range(self.n_samples):
            # Step 1: Sample a random background point
            bg_idx = np.random.randint(len(self.background))
            x_ref = self.background[bg_idx].unsqueeze(0)

            # Step 2: Sample a random interpolation coefficient
            alpha = np.random.uniform(0, 1)

            # Step 3: Create interpolated point
            # x_interp = x_ref + alpha * (x - x_ref)
            x_interp = x_ref + alpha * (x - x_ref)
            x_interp = x_interp.detach().requires_grad_(True)

            # Step 4: Compute gradient at interpolated point
            output = self.model(x_interp)
            if output.dim() > 1:
                output = output.squeeze()
            output.backward()

            grad = x_interp.grad.detach().squeeze()

            # Step 5: Multiply gradient by input difference
            # This is the expected gradients formula:
            # attribution_i = grad_i * (x_i - x_ref_i)
            diff = (x - x_ref).squeeze()
            attribution = (grad * diff).numpy()

            attributions += attribution

        # Average over all samples
        attributions /= self.n_samples

        return attributions

    def explain_batch(
        self, X: torch.Tensor, show_progress: bool = True
    ) -> np.ndarray:
        """
        Compute GradientSHAP for multiple inputs efficiently.
        """
        all_attributions = []

        for i in range(len(X)):
            if show_progress and (i + 1) % 10 == 0:
                print(f"  Explaining instance {i+1}/{len(X)}")
            attr = self.explain(X[i])
            all_attributions.append(attr)

        return np.array(all_attributions)


def gradient_shap_with_library():
    """Use the SHAP library's GradientExplainer (equivalent to GradientSHAP)."""

    # Define model
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(10, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )

        def forward(self, x):
            return self.net(x)

    model = MLP()

    # Training data (simulated)
    X_train = torch.randn(500, 10)
    background = X_train[:200]

    # GradientExplainer is SHAP's implementation of GradientSHAP
    explainer = shap.GradientExplainer(model, background)

    X_test = torch.randn(20, 10)
    shap_values = explainer.shap_values(X_test)

    # Visualize
    shap.summary_plot(
        shap_values,
        X_test.numpy(),
        feature_names=[f"x{i}" for i in range(10)],
        show=False
    )

    return shap_values
```

### 2.2 심층 비교: DeepSHAP vs GradientSHAP vs KernelSHAP

```python
def deep_comparison_experiment():
    """
    Compare DeepSHAP, GradientSHAP, and KernelSHAP on the same model.

    This is the essential sanity check: if all three methods agree,
    we can be confident in the explanations. If they disagree, we need
    to understand WHY.

    Expected findings:
    - KernelSHAP: most accurate (model-agnostic, samples coalitions),
      but slowest (O(2^n) worst case, practical ~O(n*k) with sampling)
    - DeepSHAP: fastest (single backward pass), but composition assumption
      can cause errors at non-linearities
    - GradientSHAP: moderate speed, moderate accuracy, noisier than DeepSHAP
      but fewer structural assumptions
    """
    import time

    # Shared model and data
    class DeepNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(20, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )

        def forward(self, x):
            return self.net(x)

    torch.manual_seed(42)
    model = DeepNet()

    # Generate data with known interactions
    X_train = torch.randn(1000, 20)
    y_train = (
        3 * X_train[:, 0] +          # Strong linear effect
        2 * X_train[:, 1] ** 2 +      # Non-linear effect
        X_train[:, 2] * X_train[:, 3] + # Interaction effect
        0.5 * torch.randn(1000)        # Noise
    )

    # Train model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(200):
        optimizer.zero_grad()
        loss = nn.MSELoss()(model(X_train).squeeze(), y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    background = X_train[:100]
    X_test = X_train[900:910]  # 10 test instances

    # --- Method 1: DeepSHAP ---
    print("Method 1: DeepSHAP")
    t0 = time.time()
    deep_explainer = shap.DeepExplainer(model, background)
    sv_deep = deep_explainer.shap_values(X_test)
    t_deep = time.time() - t0
    print(f"  Time: {t_deep:.3f}s")

    # --- Method 2: GradientSHAP ---
    print("\nMethod 2: GradientSHAP")
    t0 = time.time()
    grad_explainer = shap.GradientExplainer(model, background)
    sv_grad = grad_explainer.shap_values(X_test)
    t_grad = time.time() - t0
    print(f"  Time: {t_grad:.3f}s")

    # --- Method 3: KernelSHAP ---
    print("\nMethod 3: KernelSHAP")

    def model_predict(x):
        """Wrapper for KernelSHAP (needs numpy in, numpy out)."""
        with torch.no_grad():
            return model(torch.tensor(x, dtype=torch.float32)).numpy()

    t0 = time.time()
    kernel_explainer = shap.KernelExplainer(
        model_predict, background.numpy()
    )
    sv_kernel = kernel_explainer.shap_values(X_test.numpy(), nsamples=500)
    t_kernel = time.time() - t0
    print(f"  Time: {t_kernel:.3f}s")

    # --- Compare results ---
    print("\n" + "=" * 60)
    print("Pairwise Correlation of SHAP Values")
    print("=" * 60)

    methods = {
        "DeepSHAP": np.array(sv_deep).flatten(),
        "GradientSHAP": np.array(sv_grad).flatten(),
        "KernelSHAP": np.array(sv_kernel).flatten()
    }

    for name1, vals1 in methods.items():
        for name2, vals2 in methods.items():
            if name1 < name2:
                corr = np.corrcoef(vals1, vals2)[0, 1]
                mae = np.mean(np.abs(vals1 - vals2))
                print(f"  {name1:15s} vs {name2:15s}: "
                      f"r={corr:.4f}, MAE={mae:.4f}")

    # --- Timing comparison ---
    print(f"\nTiming Summary:")
    print(f"  DeepSHAP:     {t_deep:.3f}s (fastest)")
    print(f"  GradientSHAP: {t_grad:.3f}s")
    print(f"  KernelSHAP:   {t_kernel:.3f}s (slowest but most accurate)")

    return {
        "deep": sv_deep,
        "gradient": sv_grad,
        "kernel": sv_kernel
    }


comparison = deep_comparison_experiment()
```

---

## 3. SHAP 상호작용 값(SHAP Interaction Values)

### 3.1 섀플리 상호작용 지수(The Shapley Interaction Index)

```python
"""
SHAP Interaction Values extend SHAP to capture pairwise feature interactions.

Standard SHAP: phi_i = contribution of feature i alone
SHAP Interaction: phi_ij = additional contribution from the PAIR (i, j)

The Shapley Interaction Index (Grabisch & Roubens 1999):
    phi_ij = sum over S not containing i,j of
             w(S) * [f(S ∪ {i,j}) - f(S ∪ {i}) - f(S ∪ {j}) + f(S)]

This is the "surplus" from including both i and j together,
beyond their individual effects. It directly measures synergy.

Properties:
- phi_ij = phi_ji (symmetric)
- sum_j phi_ij = phi_i (interactions decompose the main effect)
- If features i and j are independent, phi_ij ≈ 0

The interaction matrix has shape (n_features, n_features):
- Diagonal entries phi_ii = main effects
- Off-diagonal entries phi_ij = interaction effects
"""

import shap
import numpy as np
import matplotlib.pyplot as plt


def compute_shap_interactions():
    """
    Compute and visualize SHAP interaction values using TreeExplainer.

    TreeExplainer supports exact interaction values computation
    in polynomial time (O(T * L * D^2) where T=trees, L=leaves, D=depth).
    This is tractable for gradient boosted models.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.datasets import make_friedman1

    # Generate data with known interactions
    # Friedman #1: y = 10*sin(pi*x0*x1) + 20*(x2-0.5)^2 + 10*x3 + 5*x4
    # This has a strong x0*x1 interaction (inside the sin)
    X, y = make_friedman1(n_samples=2000, n_features=10, random_state=42)
    feature_names = [f"x{i}" for i in range(10)]

    # Train gradient boosted model
    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, random_state=42
    )
    model.fit(X, y)

    # Compute SHAP interaction values
    # TreeExplainer computes EXACT interaction values (not approximations)
    explainer = shap.TreeExplainer(model)

    # For a subset of test instances (interaction computation is expensive)
    X_test = X[:100]
    interaction_values = explainer.shap_interaction_values(X_test)

    print(f"Interaction values shape: {interaction_values.shape}")
    # Shape: (100, 10, 10) — for each instance, a 10x10 interaction matrix

    # --- Analyze average interactions ---
    # Average the absolute interaction values across all instances
    avg_interactions = np.abs(interaction_values).mean(axis=0)

    # The diagonal contains main effects, off-diagonal are interactions
    print("\nAverage Interaction Matrix (top 5x5):")
    print("  " + "".join(f"{name:>8s}" for name in feature_names[:5]))
    for i in range(5):
        row = "".join(f"{avg_interactions[i, j]:8.3f}" for j in range(5))
        print(f"  {feature_names[i]:3s}{row}")

    # --- Identify strongest interactions ---
    n_features = len(feature_names)
    interactions = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            interactions.append((
                feature_names[i], feature_names[j],
                avg_interactions[i, j]
            ))

    interactions.sort(key=lambda x: x[2], reverse=True)

    print("\nTop 5 Feature Interactions:")
    for f1, f2, strength in interactions[:5]:
        print(f"  {f1} x {f2}: {strength:.4f}")
    # Expected: x0 x x1 should be the strongest (from sin(pi*x0*x1))

    # --- Visualize interaction matrix ---
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(avg_interactions[:5, :5], cmap="YlOrRd")
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(feature_names[:5])
    ax.set_yticklabels(feature_names[:5])
    ax.set_title("Average |SHAP Interaction Values|")

    # Annotate cells
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{avg_interactions[i, j]:.3f}",
                    ha="center", va="center", fontsize=10)

    plt.colorbar(im, label="Mean |Interaction|")
    plt.tight_layout()
    plt.savefig("shap_interactions.png", dpi=150)
    plt.show()

    return interaction_values


interaction_values = compute_shap_interactions()
```

### 3.2 상호작용 효과 해석

```python
def interpret_interaction_pair(
    interaction_values: np.ndarray,
    X: np.ndarray,
    feature_i: int,
    feature_j: int,
    feature_names: list[str]
):
    """
    Deep dive into a specific feature interaction.

    For the pair (feature_i, feature_j), we want to understand:
    1. How does the interaction change across the feature space?
    2. Is the interaction positive (synergy) or negative (redundancy)?
    3. In which regions of the data is the interaction strongest?
    """
    import matplotlib.pyplot as plt

    fi = feature_i
    fj = feature_j

    # Extract the interaction for this pair across all instances
    pair_interaction = interaction_values[:, fi, fj]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Interaction value vs feature_i, colored by feature_j
    scatter = axes[0].scatter(
        X[:, fi], pair_interaction,
        c=X[:, fj], cmap="coolwarm", alpha=0.6, s=20
    )
    axes[0].set_xlabel(feature_names[fi])
    axes[0].set_ylabel(f"SHAP interaction ({feature_names[fi]} x {feature_names[fj]})")
    axes[0].set_title(f"Interaction vs {feature_names[fi]}")
    plt.colorbar(scatter, ax=axes[0], label=feature_names[fj])

    # Plot 2: Interaction value vs feature_j, colored by feature_i
    scatter = axes[1].scatter(
        X[:, fj], pair_interaction,
        c=X[:, fi], cmap="coolwarm", alpha=0.6, s=20
    )
    axes[1].set_xlabel(feature_names[fj])
    axes[1].set_ylabel(f"SHAP interaction ({feature_names[fi]} x {feature_names[fj]})")
    axes[1].set_title(f"Interaction vs {feature_names[fj]}")
    plt.colorbar(scatter, ax=axes[1], label=feature_names[fi])

    # Plot 3: 2D interaction landscape
    scatter = axes[2].scatter(
        X[:, fi], X[:, fj],
        c=pair_interaction, cmap="RdBu_r", alpha=0.6, s=20
    )
    axes[2].set_xlabel(feature_names[fi])
    axes[2].set_ylabel(feature_names[fj])
    axes[2].set_title("Interaction Landscape")
    plt.colorbar(scatter, ax=axes[2], label="Interaction Value")

    plt.suptitle(
        f"SHAP Interaction: {feature_names[fi]} x {feature_names[fj]}",
        fontsize=14, y=1.02
    )
    plt.tight_layout()
    plt.savefig(f"interaction_{feature_names[fi]}_{feature_names[fj]}.png",
                dpi=150, bbox_inches="tight")
    plt.show()

    # Summary statistics
    print(f"\nInteraction Summary: {feature_names[fi]} x {feature_names[fj]}")
    print(f"  Mean interaction: {pair_interaction.mean():.4f}")
    print(f"  Mean |interaction|: {np.abs(pair_interaction).mean():.4f}")
    print(f"  Max interaction: {pair_interaction.max():.4f}")
    print(f"  Min interaction: {pair_interaction.min():.4f}")
    print(f"  Std: {pair_interaction.std():.4f}")

    # Positive interaction = synergy (together they contribute MORE)
    # Negative interaction = redundancy (together they contribute LESS)
    pos_frac = (pair_interaction > 0).mean()
    print(f"  Fraction positive (synergistic): {pos_frac:.2%}")
    print(f"  Fraction negative (redundant): {1-pos_frac:.2%}")
```

---

## 4. 비대칭 섀플리 값과 인과적 SHAP(Asymmetric Shapley Values and Causal SHAP)

### 4.1 표준 SHAP의 문제

```python
"""
Standard Shapley values treat all feature orderings as equally likely.
When features are causally related, this is WRONG.

Example: Income → Savings (income causally affects savings)

Standard SHAP considers the coalition {savings} without income.
But this is causally impossible: in reality, if you remove income,
savings would also change. Standard SHAP ignores this dependency.

Two solutions:
1. Asymmetric Shapley Values (Frye et al. 2020)
   - Respect a causal ordering: only consider coalitions consistent
     with the causal DAG
2. Causal SHAP (Heskes et al. 2020)
   - Use interventional (do-calculus) conditionals instead of
     observational conditionals
"""
```

### 4.2 비대칭 섀플리 값

```python
import numpy as np
from itertools import permutations


def asymmetric_shapley_values(
    model_fn,
    x: np.ndarray,
    background: np.ndarray,
    causal_ordering: list[list[int]],
    n_samples: int = 1000
) -> np.ndarray:
    """
    Compute Asymmetric Shapley Values (Frye et al. 2020).

    In standard Shapley values, we average over ALL permutations.
    In asymmetric Shapley, we only average over permutations that
    RESPECT the causal ordering.

    Parameters:
        model_fn: f(x) -> prediction
        x: Instance to explain, shape (d,)
        background: Background dataset, shape (N, d)
        causal_ordering: List of lists defining the causal order.
                        E.g., [[0, 1], [2], [3, 4]] means
                        features 0,1 come before 2, which comes before 3,4.
                        Within a group, order is flexible.
        n_samples: Number of permutation samples

    Returns:
        Asymmetric Shapley values, shape (d,)

    Example causal graph:
        Age → Income → Savings
        Education → Income

    causal_ordering = [[age, education], [income], [savings]]
    This means: we never consider savings without income,
    or income without age/education.
    """
    d = len(x)
    attributions = np.zeros(d)
    counts = np.zeros(d)

    def generate_causal_permutation():
        """Generate a random permutation respecting the causal order."""
        perm = []
        for group in causal_ordering:
            # Shuffle features within each causal group
            shuffled = list(group)
            np.random.shuffle(shuffled)
            perm.extend(shuffled)
        return perm

    def marginal_contribution(perm, feature_idx, x, background):
        """
        Compute marginal contribution of feature at position in permutation.

        v(S ∪ {i}) - v(S) where S = features before i in the permutation
        """
        pos = perm.index(feature_idx)

        # Coalition S: features before feature_idx in the permutation
        S = set(perm[:pos])
        S_with_i = S | {feature_idx}

        # For features NOT in coalition: use background distribution
        # For features IN coalition: use actual values
        def evaluate_coalition(coalition):
            """E[f(x)] where features in coalition use x, others use background."""
            # Sample from background for non-coalition features
            n_bg = min(50, len(background))
            bg_sample = background[np.random.choice(len(background), n_bg)]

            preds = []
            for bg in bg_sample:
                x_modified = bg.copy()
                for feat in coalition:
                    x_modified[feat] = x[feat]
                preds.append(model_fn(x_modified.reshape(1, -1))[0])

            return np.mean(preds)

        v_with = evaluate_coalition(S_with_i)
        v_without = evaluate_coalition(S)

        return v_with - v_without

    # Monte Carlo estimation
    for _ in range(n_samples):
        perm = generate_causal_permutation()

        for feature_idx in range(d):
            mc = marginal_contribution(perm, feature_idx, x, background)
            attributions[feature_idx] += mc
            counts[feature_idx] += 1

    attributions /= counts

    return attributions


def demonstrate_asymmetric_shapley():
    """
    Show the difference between standard and asymmetric Shapley values
    on a causal scenario.
    """
    from sklearn.ensemble import GradientBoostingRegressor

    np.random.seed(42)

    # Generate causal data:
    # Age → Income → Savings
    # Education → Income
    n = 2000
    age = np.random.uniform(20, 65, n)
    education = np.random.randint(0, 4, n)  # 0=HS, 1=BS, 2=MS, 3=PhD
    income = 20000 + 500 * age + 15000 * education + np.random.normal(0, 5000, n)
    savings = 0.2 * income + 100 * age + np.random.normal(0, 3000, n)

    # Target: loan approval score
    X = np.column_stack([age, education, income, savings])
    y = 0.3 * income / 1000 + 0.4 * savings / 1000 + 0.2 * age + np.random.normal(0, 5, n)

    feature_names = ["age", "education", "income", "savings"]

    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Standard SHAP
    explainer = shap.TreeExplainer(model)
    standard_shap = explainer.shap_values(X[:1])

    # Asymmetric SHAP with causal ordering
    causal_ordering = [[0, 1], [2], [3]]  # [age, edu] → income → savings

    asym_shap = asymmetric_shapley_values(
        model_fn=model.predict,
        x=X[0],
        background=X[:200],
        causal_ordering=causal_ordering,
        n_samples=500
    )

    print("Standard vs Asymmetric Shapley Values")
    print("=" * 50)
    print(f"{'Feature':12s} {'Standard':>10s} {'Asymmetric':>12s} {'Diff':>8s}")
    print("-" * 50)
    for i, name in enumerate(feature_names):
        std_val = standard_shap[0][i] if isinstance(standard_shap, list) else standard_shap[i]
        diff = asym_shap[i] - std_val
        print(f"{name:12s} {std_val:10.4f} {asym_shap[i]:12.4f} {diff:8.4f}")

    # Expected: Asymmetric gives MORE credit to upstream causes (age, education)
    # and LESS to downstream effects (savings), because standard SHAP
    # "double-counts" the causal pathway.

    return standard_shap, asym_shap
```

### 4.3 인과적 SHAP(Causal SHAP)

```python
"""
Causal SHAP (Heskes et al. 2020) addresses a different problem than
Asymmetric Shapley values.

The issue: When computing SHAP values, we need to evaluate
    E[f(x) | x_S = x_S_observed]
for various feature subsets S. This conditional expectation can be
computed in two ways:

1. OBSERVATIONAL: E[f(x) | X_S = x_S]
   "Given that we OBSERVE these feature values..."
   This uses the joint distribution: p(x_{-S} | x_S)
   Problem: leaks information through correlations

2. INTERVENTIONAL: E[f(x) | do(X_S = x_S)]
   "Given that we SET (intervene on) these feature values..."
   This uses the causal mechanism: p(x_{-S} | do(X_S = x_S))
   Advantage: respects causal structure, no information leakage

Causal SHAP uses interventional conditionals, computed via
the causal graph (DAG) and do-calculus.
"""


def causal_shap_values(
    model_fn,
    x: np.ndarray,
    X_train: np.ndarray,
    causal_graph: dict[int, list[int]],
    n_samples: int = 200,
    n_permutations: int = 500
) -> np.ndarray:
    """
    Compute Causal SHAP values using interventional conditionals.

    Parameters:
        model_fn: f(x) -> prediction
        x: Instance to explain, shape (d,)
        X_train: Training data for sampling, shape (N, d)
        causal_graph: Dict mapping feature index to its parents.
                     E.g., {0: [], 1: [], 2: [0, 1], 3: [2]}
                     means feature 2 depends on 0 and 1,
                     feature 3 depends on 2.
        n_samples: Number of background samples per evaluation
        n_permutations: Number of permutation samples for Shapley

    Returns:
        Causal SHAP values, shape (d,)
    """
    d = len(x)

    def interventional_expectation(coalition: set, x_values: np.ndarray):
        """
        Compute E[f(x) | do(X_S = x_S)] via interventional sampling.

        For features in the coalition: set them to x_values.
        For features NOT in the coalition: sample from their
        CAUSAL mechanism (conditioned on parents), NOT from the
        marginal or observational conditional.

        This is the key difference from standard SHAP.
        """
        samples = []

        for _ in range(n_samples):
            # Start with a random background sample
            x_sample = X_train[np.random.randint(len(X_train))].copy()

            # Process features in topological order
            # (parents before children)
            processed = set()
            to_process = list(range(d))

            while to_process:
                for feat in list(to_process):
                    parents = causal_graph.get(feat, [])
                    if all(p in processed for p in parents):
                        if feat in coalition:
                            # INTERVENE: set to observed value
                            x_sample[feat] = x_values[feat]
                        else:
                            # SAMPLE from causal mechanism
                            # In practice, this uses a structural equation model.
                            # Here we use a simplified version: sample from
                            # the conditional distribution given parents.
                            if parents:
                                parent_vals = x_sample[parents]
                                # Find training samples with similar parent values
                                dists = np.abs(
                                    X_train[:, parents] - parent_vals
                                ).sum(axis=1)
                                nearest = np.argsort(dists)[:20]
                                x_sample[feat] = X_train[
                                    np.random.choice(nearest), feat
                                ]
                            # If no parents, keep the random background value

                        processed.add(feat)
                        to_process.remove(feat)

            samples.append(model_fn(x_sample.reshape(1, -1))[0])

        return np.mean(samples)

    # Compute Shapley values using interventional expectations
    attributions = np.zeros(d)

    for _ in range(n_permutations):
        perm = np.random.permutation(d)
        coalition = set()

        for feat in perm:
            # Marginal contribution with interventional conditional
            v_without = interventional_expectation(coalition, x)

            coalition.add(feat)
            v_with = interventional_expectation(coalition, x)

            attributions[feat] += (v_with - v_without)

    attributions /= n_permutations

    return attributions
```

---

## 5. 계산 최적화(Computational Optimization)

### 5.1 배경 샘플링과 미니 배치 처리

```python
"""
SHAP computation can be expensive, especially for large models.
Here are practical optimizations for production use.
"""


class OptimizedSHAPExplainer:
    """
    Production-ready SHAP computation with performance optimizations.

    Key optimizations:
    1. Background subsampling: Use k-means centroids instead of full dataset
    2. Mini-batching: Process multiple instances in parallel
    3. GPU acceleration: Move computations to GPU
    4. Caching: Cache explanations for repeated queries
    5. Early stopping: Stop sampling when values converge
    """

    def __init__(
        self,
        model: nn.Module,
        background: torch.Tensor,
        n_background: int = 100,
        use_kmeans: bool = True,
        device: str = "auto"
    ):
        from sklearn.cluster import KMeans

        # Auto-detect GPU
        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.model.eval()

        # Background compression via k-means
        if use_kmeans and len(background) > n_background:
            print(f"Compressing {len(background)} background samples "
                  f"to {n_background} k-means centroids...")
            kmeans = KMeans(
                n_clusters=n_background, random_state=42, n_init=10
            )
            kmeans.fit(background.numpy())
            self.background = torch.tensor(
                kmeans.cluster_centers_, dtype=torch.float32
            ).to(self.device)

            # Store cluster weights (larger clusters = higher weight)
            cluster_sizes = np.bincount(kmeans.labels_, minlength=n_background)
            self.bg_weights = torch.tensor(
                cluster_sizes / cluster_sizes.sum(), dtype=torch.float32
            ).to(self.device)
        else:
            self.background = background[:n_background].to(self.device)
            self.bg_weights = torch.ones(
                len(self.background), dtype=torch.float32
            ).to(self.device) / len(self.background)

        # Explanation cache
        self._cache = {}

    @torch.no_grad()
    def explain_batch_gpu(
        self,
        X: torch.Tensor,
        batch_size: int = 32,
        n_gradient_samples: int = 50
    ) -> np.ndarray:
        """
        GPU-accelerated GradientSHAP for a batch of inputs.

        This is significantly faster than per-instance computation,
        especially on GPU. The key is to vectorize across both
        the batch dimension and the background sample dimension.
        """
        X = X.to(self.device)
        n_inputs = len(X)
        n_features = X.shape[1]
        all_attributions = torch.zeros(n_inputs, n_features, device=self.device)

        for start in range(0, n_inputs, batch_size):
            end = min(start + batch_size, n_inputs)
            batch = X[start:end]  # (B, d)
            batch_size_actual = len(batch)

            batch_attr = torch.zeros(
                batch_size_actual, n_features, device=self.device
            )

            for _ in range(n_gradient_samples):
                # Sample random backgrounds and interpolation coefficients
                bg_indices = torch.randint(
                    0, len(self.background), (batch_size_actual,)
                )
                bg = self.background[bg_indices]  # (B, d)
                alpha = torch.rand(
                    batch_size_actual, 1, device=self.device
                )

                # Interpolated points
                x_interp = bg + alpha * (batch - bg)
                x_interp.requires_grad_(True)

                # Forward pass
                output = self.model(x_interp)
                if output.dim() > 1:
                    output = output.sum(dim=1)

                # Backward pass for all instances in batch simultaneously
                output.sum().backward()

                # Attribution = gradient * (input - reference)
                grad = x_interp.grad  # (B, d)
                diff = batch - bg    # (B, d)
                batch_attr += grad * diff

                x_interp.grad = None  # Clear gradients

            batch_attr /= n_gradient_samples
            all_attributions[start:end] = batch_attr

        return all_attributions.cpu().numpy()

    def explain_with_convergence(
        self,
        x: torch.Tensor,
        max_samples: int = 1000,
        convergence_threshold: float = 0.01,
        check_interval: int = 50
    ) -> tuple[np.ndarray, int]:
        """
        Compute SHAP values with early stopping when values converge.

        Instead of using a fixed number of samples, we monitor the
        running average of SHAP values and stop when changes fall
        below a threshold. This saves computation for "easy" instances
        where fewer samples suffice.

        Returns:
            (shap_values, n_samples_used)
        """
        x = x.to(self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        n_features = x.shape[1]
        running_sum = torch.zeros(n_features, device=self.device)
        prev_values = None
        n_used = 0

        for sample_idx in range(max_samples):
            # One GradientSHAP sample
            bg_idx = torch.randint(0, len(self.background), (1,))
            bg = self.background[bg_idx]
            alpha = torch.rand(1, 1, device=self.device)

            x_interp = (bg + alpha * (x - bg)).detach().requires_grad_(True)

            output = self.model(x_interp)
            if output.dim() > 1:
                output = output.squeeze()
            output.backward()

            grad = x_interp.grad.squeeze()
            diff = (x - bg).squeeze()
            running_sum += grad * diff
            n_used += 1

            # Check convergence periodically
            if n_used % check_interval == 0 and n_used > check_interval:
                current_values = (running_sum / n_used).cpu().numpy()

                if prev_values is not None:
                    # Relative change in SHAP values
                    max_change = np.max(np.abs(current_values - prev_values))
                    max_value = np.max(np.abs(current_values)) + 1e-10
                    relative_change = max_change / max_value

                    if relative_change < convergence_threshold:
                        print(f"  Converged after {n_used} samples "
                              f"(relative change: {relative_change:.6f})")
                        return current_values, n_used

                prev_values = current_values.copy()

        final_values = (running_sum / n_used).cpu().numpy()
        print(f"  Used all {max_samples} samples (may not have converged)")
        return final_values, n_used
```

---

## 6. TreeSHAP 병리 현상(TreeSHAP Pathologies)

### 6.1 상관된 특성 문제(The Correlated Feature Problem)

```python
"""
TreeSHAP Pathology: Bias with Correlated Features

Sundararajan & Najmi (2020) identified a fundamental issue with
TreeSHAP (the fast exact algorithm for tree models):

TreeSHAP uses the "interventional" conditional:
    E[f(x) | x_S = x_S] ≈ (1/N) * sum of leaf values reached
    when features in S are fixed and others follow tree paths

This IGNORES feature correlations. When features are correlated,
TreeSHAP can:
1. Assign attribution to an uninformative feature
2. Split attribution between correlated features unpredictably
3. Produce explanations that contradict causal reasoning

The fix options:
- Use observational conditional (shap.TreeExplainer with
  feature_perturbation="tree_path_dependent" — the default)
- Use interventional conditional (feature_perturbation="interventional")
- Neither is perfect: interventional ignores correlations,
  observational can leak information
"""


def demonstrate_treeshap_pathology():
    """
    Show that TreeSHAP gives counterintuitive results with correlated features.

    Setup: x1 and x2 are perfectly correlated (x2 = x1 + noise).
    The model uses x1 but not x2. Standard SHAP should give all
    attribution to x1. TreeSHAP might split attribution between them.
    """
    from sklearn.ensemble import GradientBoostingRegressor

    np.random.seed(42)

    # Generate correlated data
    n = 5000
    x1 = np.random.normal(0, 1, n)
    x2 = x1 + np.random.normal(0, 0.01, n)  # Nearly identical to x1
    x3 = np.random.normal(0, 1, n)           # Independent, irrelevant
    X = np.column_stack([x1, x2, x3])

    # Target depends ONLY on x1
    y = 3 * x1 + np.random.normal(0, 0.1, n)

    feature_names = ["x1 (causal)", "x2 (correlated copy)", "x3 (irrelevant)"]

    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, random_state=42
    )
    model.fit(X, y)

    # --- TreeSHAP with default (tree_path_dependent) ---
    print("TreeSHAP (tree_path_dependent / observational):")
    explainer_obs = shap.TreeExplainer(
        model, feature_perturbation="tree_path_dependent"
    )
    sv_obs = explainer_obs.shap_values(X[:100])
    mean_abs_obs = np.abs(sv_obs).mean(axis=0)
    for i, name in enumerate(feature_names):
        print(f"  {name:30s}: mean |SHAP| = {mean_abs_obs[i]:.4f}")

    # --- TreeSHAP with interventional ---
    print("\nTreeSHAP (interventional):")
    explainer_int = shap.TreeExplainer(
        model, X[:200], feature_perturbation="interventional"
    )
    sv_int = explainer_int.shap_values(X[:100])
    mean_abs_int = np.abs(sv_int).mean(axis=0)
    for i, name in enumerate(feature_names):
        print(f"  {name:30s}: mean |SHAP| = {mean_abs_int[i]:.4f}")

    # --- KernelSHAP (model-agnostic, uses observational conditional) ---
    print("\nKernelSHAP (model-agnostic reference):")
    kernel_explainer = shap.KernelExplainer(model.predict, X[:200])
    sv_kernel = kernel_explainer.shap_values(X[:10], nsamples=500)
    mean_abs_kernel = np.abs(sv_kernel).mean(axis=0)
    for i, name in enumerate(feature_names):
        print(f"  {name:30s}: mean |SHAP| = {mean_abs_kernel[i]:.4f}")

    print("\n" + "=" * 60)
    print("ANALYSIS:")
    print("  x1 is the ONLY causal feature. Ideally all attribution goes to x1.")
    print("  x2 is a correlated copy — it should get ZERO attribution.")
    print("  x3 is independent and irrelevant — it should get ZERO.")
    print("")
    print("  Tree_path_dependent: May 'leak' attribution to x2 via correlations.")
    print("  Interventional: May STILL split between x1 and x2 because the")
    print("    tree might split on x2 (since x2 ≈ x1, the tree can use either).")
    print("  KernelSHAP: Uses observational conditional, may also leak to x2.")
    print("")
    print("  This is a FUNDAMENTAL limitation: without causal knowledge,")
    print("  no method can perfectly distinguish x1 from x2.")

    return {
        "observational": sv_obs,
        "interventional": sv_int,
        "kernel": sv_kernel
    }


demonstrate_treeshap_pathology()
```

### 6.2 실용적 권장 사항

```python
"""
Practical recommendations for dealing with SHAP pathologies:

1. CHECK FEATURE CORRELATIONS FIRST
   Before computing SHAP, compute the correlation matrix.
   If features are highly correlated (|r| > 0.8), be cautious
   about interpreting individual feature attributions.
"""


def shap_with_correlation_audit(
    model,
    X: np.ndarray,
    feature_names: list[str],
    correlation_threshold: float = 0.8
):
    """
    Compute SHAP values with a correlation audit that warns about
    potentially unreliable attributions.
    """
    # Step 1: Correlation audit
    corr_matrix = np.corrcoef(X.T)
    n_features = len(feature_names)

    print("Step 1: Feature Correlation Audit")
    print("-" * 50)

    correlated_pairs = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            r = abs(corr_matrix[i, j])
            if r > correlation_threshold:
                correlated_pairs.append((
                    feature_names[i], feature_names[j], r
                ))
                print(f"  WARNING: |corr({feature_names[i]}, "
                      f"{feature_names[j]})| = {r:.3f}")

    if not correlated_pairs:
        print("  No highly correlated feature pairs found.")

    # Step 2: Compute SHAP
    print(f"\nStep 2: Computing SHAP values...")
    explainer = shap.TreeExplainer(model) if hasattr(model, 'estimators_') \
        else shap.KernelExplainer(model.predict, X[:100])

    shap_values = explainer.shap_values(X[:50])

    # Step 3: Cross-reference correlations with attributions
    if correlated_pairs:
        print(f"\nStep 3: Attribution Analysis for Correlated Features")
        print("-" * 50)

        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        for f1_name, f2_name, r in correlated_pairs:
            f1_idx = feature_names.index(f1_name)
            f2_idx = feature_names.index(f2_name)

            attr1 = mean_abs_shap[f1_idx]
            attr2 = mean_abs_shap[f2_idx]
            combined = attr1 + attr2

            print(f"\n  Correlated pair: {f1_name} & {f2_name} (r={r:.3f})")
            print(f"    {f1_name} attribution: {attr1:.4f}")
            print(f"    {f2_name} attribution: {attr2:.4f}")
            print(f"    Combined attribution:  {combined:.4f}")
            print(f"    CAUTION: Individual attributions may be unreliable.")
            print(f"    Consider reporting the COMBINED attribution instead.")
            print(f"    Or use domain knowledge to determine causality.")

    return shap_values
```

---

## 7. 종합 파이프라인: 전체 분석(Putting It All Together: Complete Pipeline)

```python
def advanced_shap_analysis_pipeline(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    causal_graph: dict = None
):
    """
    Complete advanced SHAP analysis pipeline.

    Steps:
    1. Correlation audit
    2. Standard SHAP values
    3. SHAP interaction values
    4. Causal SHAP (if causal graph provided)
    5. Comparison and recommendations
    """
    import shap

    print("=" * 70)
    print("ADVANCED SHAP ANALYSIS PIPELINE")
    print("=" * 70)

    # Step 1: Correlation audit
    print("\n--- Step 1: Feature Correlation Audit ---")
    corr = np.corrcoef(X_train.T)
    high_corr = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            if abs(corr[i, j]) > 0.7:
                high_corr.append((feature_names[i], feature_names[j], corr[i, j]))
                print(f"  WARN: {feature_names[i]} x {feature_names[j]}: "
                      f"r = {corr[i, j]:.3f}")
    if not high_corr:
        print("  No highly correlated features detected.")

    # Step 2: Standard SHAP
    print("\n--- Step 2: Standard SHAP Values ---")
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_test[:50])

    mean_abs = np.abs(sv).mean(axis=0)
    ranked = np.argsort(-mean_abs)
    print("  Feature importance ranking:")
    for rank, idx in enumerate(ranked):
        print(f"    {rank+1}. {feature_names[idx]:20s}: "
              f"mean |SHAP| = {mean_abs[idx]:.4f}")

    # Step 3: Interaction values
    print("\n--- Step 3: SHAP Interaction Values ---")
    interaction_vals = explainer.shap_interaction_values(X_test[:50])
    avg_inter = np.abs(interaction_vals).mean(axis=0)

    # Top interactions (off-diagonal)
    interactions = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            interactions.append((
                feature_names[i], feature_names[j], avg_inter[i, j]
            ))
    interactions.sort(key=lambda x: x[2], reverse=True)

    print("  Top 5 feature interactions:")
    for f1, f2, strength in interactions[:5]:
        print(f"    {f1} x {f2}: {strength:.4f}")

    # Step 4: Causal SHAP (if graph provided)
    if causal_graph:
        print("\n--- Step 4: Causal SHAP ---")
        causal_sv = causal_shap_values(
            model_fn=model.predict,
            x=X_test[0],
            X_train=X_train,
            causal_graph=causal_graph,
            n_samples=100,
            n_permutations=200
        )
        print("  Causal vs Standard SHAP for first instance:")
        for i, name in enumerate(feature_names):
            std = sv[0][i] if isinstance(sv, list) else sv[0, i]
            print(f"    {name:20s}: standard={std:8.4f}, "
                  f"causal={causal_sv[i]:8.4f}")
    else:
        print("\n--- Step 4: Causal SHAP ---")
        print("  Skipped (no causal graph provided)")

    # Step 5: Summary and recommendations
    print("\n--- Step 5: Recommendations ---")
    if high_corr:
        print("  [!] Correlated features detected. Consider:")
        print("      - Reporting combined attributions for correlated groups")
        print("      - Using causal SHAP if causal structure is known")
        print("      - Running sensitivity analysis across background sets")
    else:
        print("  [OK] No major concerns. Standard SHAP values are reliable.")

    return {
        "shap_values": sv,
        "interaction_values": interaction_vals,
    }
```

---

## 요약

- **DeepSHAP**은 DeepLIFT의 역방향 전파와 다중 배경 샘플에 대한 섀플리 값 평균화를 결합한다. 배경 선택(무작위, k-평균, 계층적)이 결과에 결정적인 영향을 미친다.
- **GradientSHAP**은 기대 그래디언트(Expected Gradients)를 구현한다 — 무작위 기준선과 보간 지점에 대해 적분 그래디언트(Integrated Gradients)를 평균화한다. DeepSHAP보다 이론적으로 더 견고하지만 노이즈가 더 많다.
- 방법을 비교할 때, **KernelSHAP이 가장 정확**하고(모델 비의존적), **DeepSHAP이 가장 빠르며**(단일 역방향 패스), **GradientSHAP은 둘 사이의 균형**을 맞춘다.
- **SHAP 상호작용 값(Interaction Values)**은 기여도를 주 효과(대각선)와 쌍별 상호작용(비대각선)으로 분해하여 특성 시너지를 직접 측정한다.
- **비대칭 섀플리 값(Asymmetric Shapley Values)**은 순열을 인과적 순서에 맞게 제한하고, **인과적 SHAP(Causal SHAP)**은 관측적 조건부 대신 개입적 조건부를 사용한다. 둘 다 표준 SHAP이 인과 구조를 무시한다는 근본적인 한계를 해결한다.
- **TreeSHAP 병리 현상**은 상관된 특성에서 발생한다: 어떤 조건부(관측적 또는 개입적)를 사용하든 상관된 특성 사이로 기여도가 누출될 수 있다. 개별 SHAP 값을 해석하기 전에 항상 상관관계를 감사하라.
- **계산 최적화** — k-평균 배경 압축, GPU 배치 처리, 조기 중단 수렴 검사 — 는 프로덕션 배포에 필수적이다.

---

## 연습문제

### 연습문제 1: 배경 민감도 분석 (초급)

캘리포니아 주택 데이터셋에서 그래디언트 부스팅 모델을 훈련하라. 세 가지 다른 배경 전략을 사용하여 DeepSHAP 값을 계산하라: (a) 50개 무작위 샘플, (b) 200개 무작위 샘플, (c) 50개 k-평균 중심점. 피어슨 상관관계를 사용하여 결과 SHAP 값을 비교하라. 어떤 전략이 가장 안정적인 결과를 제공하는가? 계산 시간은 어떻게 확장되는가?

### 연습문제 2: 딥러닝 방법 비교 (중급)

선택한 테이블형 데이터셋에서 4레이어 MLP를 훈련하라. 50개 테스트 인스턴스에 대해 DeepSHAP, GradientSHAP, KernelSHAP을 비교하라. 각 방법 쌍에 대해 특성 중요도의 순위 상관관계를 계산하라. 방법들이 가장 많이 불일치하는 인스턴스를 식별하고 그 이유를 조사하라(힌트: 해당 영역의 비선형성을 살펴보라).

### 연습문제 3: 상호작용 발견 (중급)

알려진 x0*x1 상호작용이 있는 Friedman #1 합성 데이터셋을 사용하여 그래디언트 부스팅 모델로 SHAP 상호작용 값을 계산하라. x0-x1 상호작용이 가장 강한지 확인하라. 그런 다음 상관된 특성(x10 = x0 + 노이즈)을 추가하고 다시 실행하라. 허위 상관관계가 상호작용 값에 어떤 영향을 미치는가?

### 연습문제 4: 대출 결정을 위한 인과적 SHAP (고급)

특성이 나이, 교육, 소득, 저축, 신용 점수인 대출 승인 모델을 구축하라. 인과 그래프를 정의하라(교육과 나이가 소득을 유발하고, 소득이 저축을 유발하는 등). 표준 SHAP, 비대칭 섀플리, 인과적 SHAP 값을 비교하라. (a) 규제 감사, (b) 고객 대면 설명, (c) 모델 디버깅 세션에 어떤 설명이 가장 적절한지 설명하는 보고서를 작성하라.

### 연습문제 5: 프로덕션 SHAP 파이프라인 (고급)

다음을 수행하는 프로덕션 준비 SHAP 서비스를 구현하라: (a) k-평균을 사용한 배경 압축, (b) GPU 가속 GradientSHAP 사용, (c) 조기 중단 수렴 구현, (d) 반복 쿼리에 대한 결과 캐싱, (e) 상관관계 감사 경고 시스템 포함. 50개 이상의 특성과 10,000개 이상의 테스트 인스턴스를 가진 모델에서 지연 시간을 벤치마크하라. 목표: 설명당 100ms 미만.

---

[이전: 프로빙과 표현 분석](./05_Probing_and_Representation_Analysis.md) | [개요](./00_Overview.md) | [다음: 개념 기반 설명](./07_Concept_Based_Explanations.md)

---

**License**: CC BY-NC 4.0
