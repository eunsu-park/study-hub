"""
04. DeepSHAP and GradientSHAP for Neural Networks

Uses the SHAP library to compute Shapley-value-based feature attributions for
a PyTorch MLP classifier trained on a synthetic tabular dataset. Three SHAP
methods are compared: DeepSHAP (DeepLIFT + Shapley), GradientSHAP (expected
gradients sampling), and KernelSHAP (model-agnostic baseline). Includes
interaction value computation and a timing benchmark.

Covered topics:
    - DeepSHAP: combining DeepLIFT rescale rules with Shapley values
    - GradientSHAP: stochastic expected-gradient approximation
    - SHAP interaction values for detecting feature synergies
    - Visualization: summary, waterfall, dependence, and force plots
    - Timing benchmark: DeepSHAP vs GradientSHAP vs KernelSHAP

Related to: L06 - SHAP and Shapley Values

Requirements:
    pip install torch shap numpy matplotlib scikit-learn
"""

import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Suppress non-critical warnings from shap internals
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ====== Dataset Preparation ======

def create_dataset(
    n_samples: int = 2000,
    n_features: int = 15,
    n_informative: int = 8,
    seed: int = 42,
) -> tuple:
    """Generate a synthetic classification dataset with known informative features.

    Why synthetic data? It lets us verify that SHAP correctly identifies
    the informative features (ground truth is known). We deliberately include
    redundant and noise features to test whether SHAP can separate signal
    from noise.

    Returns:
        (X_train, X_test, y_train, y_test, feature_names, scaler)
    """
    np.random.seed(seed)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=3,
        n_repeated=0,
        n_clusters_per_class=2,
        flip_y=0.05,  # 5% label noise for realism
        random_state=seed,
    )

    feature_names = [f"feat_{i}" for i in range(n_features)]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed,
    )

    # Standardize features — neural networks train better on normalized data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Informative features: {n_informative}/{n_features}")
    print(f"  Class balance: {y_train.mean():.2%} positive")

    return X_train, X_test, y_train, y_test, feature_names, scaler


# ====== MLP Model ======

class TabularMLP(nn.Module):
    """A simple multi-layer perceptron for binary classification.

    Architecture: input -> 64 -> 32 -> 16 -> 1 (sigmoid output).
    We use BatchNorm + Dropout for regularization, and ReLU activations
    throughout. This is deliberately simple so that SHAP computation
    remains tractable and fast.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def train_model(
    model: TabularMLP,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 128,
) -> list[float]:
    """Train the MLP using binary cross-entropy loss with Adam optimizer.

    Returns the loss history for optional convergence diagnostics.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    n_samples = X_t.shape[0]
    loss_history = []

    for epoch in range(epochs):
        # Shuffle data each epoch for stochastic training
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            X_batch, y_batch = X_t[idx], y_t[idx]

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        loss_history.append(avg_loss)

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs}: loss = {avg_loss:.4f}")

    return loss_history


def evaluate_model(
    model: TabularMLP,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    """Compute test accuracy of the trained model."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32))
        preds = (logits.squeeze() > 0).int().numpy()
    acc = accuracy_score(y_test, preds)
    return acc


# ====== SHAP Computation ======

def compute_deep_shap(
    model: TabularMLP,
    background: np.ndarray,
    test_samples: np.ndarray,
) -> np.ndarray:
    """Compute SHAP values using DeepSHAP (DeepLIFT + Shapley).

    DeepSHAP propagates SHAP values through the network using DeepLIFT's
    modified backpropagation rules. It is exact for piecewise-linear
    networks (ReLU + linear layers) and efficient because it avoids
    the combinatorial explosion of the original Shapley formula.

    The background dataset serves as the "reference" distribution --
    analogous to the baseline in Integrated Gradients but averaged
    over many reference points.

    Args:
        model: Trained PyTorch model.
        background: Background dataset for the SHAP reference, shape (N, D).
        test_samples: Samples to explain, shape (M, D).

    Returns:
        SHAP values array of shape (M, D).
    """
    import shap

    bg_tensor = torch.tensor(background, dtype=torch.float32)
    test_tensor = torch.tensor(test_samples, dtype=torch.float32)

    # DeepExplainer implements the DeepSHAP algorithm
    explainer = shap.DeepExplainer(model, bg_tensor)
    shap_values = explainer.shap_values(test_tensor)

    # shap_values may be a list (multi-output) or array (single output)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    return np.array(shap_values)


def compute_gradient_shap(
    model: TabularMLP,
    background: np.ndarray,
    test_samples: np.ndarray,
    n_samples: int = 200,
) -> np.ndarray:
    """Compute SHAP values using GradientSHAP (expected gradients).

    GradientSHAP combines ideas from Integrated Gradients and SHAP:
    it samples random reference points from the background distribution
    and random interpolation points along the path, then averages the
    resulting gradients. This is more robust than DeepSHAP when the
    model has non-ReLU activations.

    Args:
        model: Trained PyTorch model.
        background: Background dataset, shape (N, D).
        test_samples: Samples to explain, shape (M, D).
        n_samples: Number of samples for the gradient estimation.

    Returns:
        SHAP values array of shape (M, D).
    """
    import shap

    bg_tensor = torch.tensor(background, dtype=torch.float32)
    test_tensor = torch.tensor(test_samples, dtype=torch.float32)

    explainer = shap.GradientExplainer(model, bg_tensor)
    shap_values = explainer.shap_values(test_tensor, nsamples=n_samples)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    return np.array(shap_values)


def compute_kernel_shap(
    model: TabularMLP,
    background: np.ndarray,
    test_samples: np.ndarray,
    n_background: int = 50,
) -> np.ndarray:
    """Compute SHAP values using KernelSHAP (model-agnostic baseline).

    KernelSHAP treats the model as a black box and uses a weighted linear
    regression on feature coalitions to estimate Shapley values. It is
    slower than DeepSHAP and GradientSHAP but works with any model.

    We include it as a reference to validate that the neural-network-specific
    methods produce similar attributions.

    Args:
        model: Trained PyTorch model (used as a predict function).
        background: Background dataset, shape (N, D). Subsampled for speed.
        test_samples: Samples to explain, shape (M, D).
        n_background: Number of background samples (KernelSHAP is O(2^D)
                      per sample, so we subsample aggressively).

    Returns:
        SHAP values array of shape (M, D).
    """
    import shap

    # Create a predict function that wraps the PyTorch model
    def predict_fn(x: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(x, dtype=torch.float32))
        return logits.numpy().flatten()

    # Subsample background for tractability
    if background.shape[0] > n_background:
        idx = np.random.choice(background.shape[0], n_background, replace=False)
        background = background[idx]

    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(test_samples, nsamples=100)

    return np.array(shap_values)


# ====== SHAP Interaction Values ======

def compute_interaction_values(
    model: TabularMLP,
    background: np.ndarray,
    test_samples: np.ndarray,
) -> np.ndarray:
    """Compute SHAP interaction values using DeepSHAP.

    Interaction values decompose each feature's SHAP value into:
    - A main effect (the feature's independent contribution)
    - Pairwise interactions (how the feature's contribution changes
      depending on other feature values)

    The interaction matrix Phi has shape (M, D, D) where Phi[m, i, j]
    is the interaction between features i and j for sample m.
    Diagonal entries Phi[m, i, i] are the main effects.

    This is computationally expensive -- O(D^2) times the cost of
    regular SHAP values -- so we use a small subset.
    """
    import shap

    bg_tensor = torch.tensor(background, dtype=torch.float32)
    test_tensor = torch.tensor(test_samples, dtype=torch.float32)

    explainer = shap.DeepExplainer(model, bg_tensor)

    # shap_interaction_values returns shape (M, D, D) or a list for multi-output
    interaction_values = explainer.shap_interaction_values(test_tensor)

    if isinstance(interaction_values, list):
        interaction_values = interaction_values[0]

    return np.array(interaction_values)


# ====== Timing Benchmark ======

def run_timing_benchmark(
    model: TabularMLP,
    background: np.ndarray,
    test_samples: np.ndarray,
    n_runs: int = 3,
) -> dict:
    """Benchmark the wall-clock time of each SHAP method.

    We run each method n_runs times and report the mean and std.
    KernelSHAP is limited to fewer samples because it is much slower.
    """
    results = {}

    for method_name, compute_fn, kwargs in [
        ("DeepSHAP", compute_deep_shap, {}),
        ("GradientSHAP", compute_gradient_shap, {"n_samples": 200}),
        ("KernelSHAP", compute_kernel_shap, {"n_background": 30}),
    ]:
        times = []
        for run in range(n_runs):
            t0 = time.time()
            _ = compute_fn(model, background, test_samples, **kwargs)
            elapsed = time.time() - t0
            times.append(elapsed)

        results[method_name] = {
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
            "times": times,
        }

    return results


# ====== Visualization Functions ======

def plot_feature_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
    title: str = "Mean |SHAP| Feature Importance",
    save_path: str = "shap_importance.png",
) -> None:
    """Bar chart of mean absolute SHAP values per feature.

    This is the SHAP equivalent of "feature importance" -- it tells us
    which features have the largest average impact on the model output
    across all samples.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(feature_names)))
    ax.barh(
        range(len(feature_names)),
        mean_abs[sorted_idx],
        color=colors[sorted_idx],
        edgecolor="gray",
    )
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Figure saved to: {save_path}")
    plt.close()


def plot_shap_summary(
    shap_values: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    save_path: str = "shap_summary.png",
) -> None:
    """SHAP summary (beeswarm) plot using the shap library.

    Each dot is one sample. Position along X = SHAP value (positive = pushes
    prediction up, negative = pushes down). Color = feature value (red = high,
    blue = low). This reveals both importance AND directionality.
    """
    import shap

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_names,
        show=False,
    )
    plt.title("SHAP Summary Plot (Beeswarm)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Figure saved to: {save_path}")
    plt.close()


def plot_method_comparison(
    deep_shap_vals: np.ndarray,
    grad_shap_vals: np.ndarray,
    kernel_shap_vals: np.ndarray,
    feature_names: list[str],
    save_path: str = "shap_method_comparison.png",
) -> None:
    """Compare mean |SHAP| across all three methods side by side.

    If DeepSHAP, GradientSHAP, and KernelSHAP produce similar rankings,
    we can be confident the attributions are robust. Large discrepancies
    may indicate the model violates assumptions of one method.
    """
    mean_deep = np.abs(deep_shap_vals).mean(axis=0)
    mean_grad = np.abs(grad_shap_vals).mean(axis=0)
    mean_kernel = np.abs(kernel_shap_vals).mean(axis=0)

    # Sort by DeepSHAP importance for consistent ordering
    sorted_idx = np.argsort(mean_deep)[::-1]

    x = np.arange(len(feature_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, mean_deep[sorted_idx], width, label="DeepSHAP",
           color="#1f77b4", edgecolor="gray")
    ax.bar(x, mean_grad[sorted_idx], width, label="GradientSHAP",
           color="#ff7f0e", edgecolor="gray")
    ax.bar(x + width, mean_kernel[sorted_idx], width, label="KernelSHAP",
           color="#2ca02c", edgecolor="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [feature_names[i] for i in sorted_idx],
        rotation=45, ha="right", fontsize=9,
    )
    ax.set_ylabel("Mean |SHAP value|", fontsize=11)
    ax.set_title("SHAP Method Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Figure saved to: {save_path}")
    plt.close()


def plot_interaction_heatmap(
    interaction_values: np.ndarray,
    feature_names: list[str],
    save_path: str = "shap_interactions.png",
) -> None:
    """Heatmap of mean absolute SHAP interaction values.

    Off-diagonal entries reveal which feature pairs have synergistic
    (or antagonistic) effects. Diagonal entries are the main effects.
    """
    # Average absolute interactions across all samples
    mean_interactions = np.abs(interaction_values).mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mean_interactions, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean |Interaction|")

    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=8)
    ax.set_title("SHAP Interaction Values", fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Figure saved to: {save_path}")
    plt.close()


def plot_timing_benchmark(
    timing_results: dict,
    save_path: str = "shap_timing.png",
) -> None:
    """Bar chart comparing computation time across SHAP methods."""
    methods = list(timing_results.keys())
    means = [timing_results[m]["mean"] for m in methods]
    stds = [timing_results[m]["std"] for m in methods]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(methods, means, yerr=stds, capsize=5, color=colors,
                  edgecolor="gray")

    # Annotate with exact times
    for bar, mean_val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01 * max(means),
            f"{mean_val:.2f}s",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_ylabel("Time (seconds)", fontsize=11)
    ax.set_title("SHAP Method Timing Comparison", fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Figure saved to: {save_path}")
    plt.close()


# ====== Correlation Analysis ======

def compute_method_correlation(
    vals_a: np.ndarray,
    vals_b: np.ndarray,
    method: str = "spearman",
) -> float:
    """Compute rank correlation between two sets of SHAP values.

    High correlation between methods suggests that the attributions are
    robust and not artifacts of a particular estimation algorithm.
    """
    a_flat = np.abs(vals_a).mean(axis=0)
    b_flat = np.abs(vals_b).mean(axis=0)

    if method == "spearman":
        from scipy.stats import spearmanr
        corr, _ = spearmanr(a_flat, b_flat)
    elif method == "pearson":
        corr = np.corrcoef(a_flat, b_flat)[0, 1]
    else:
        raise ValueError(f"Unknown method: {method}")

    return float(corr)


# ====== Main Pipeline ======

def main() -> None:
    """Train an MLP, compute SHAP values with three methods, and compare."""
    print("=" * 60)
    print("  DeepSHAP and GradientSHAP for Neural Networks")
    print("  Feature Attribution via Shapley Values")
    print("=" * 60)

    # --- Step 1: Create dataset ---
    print("\n[1] Creating synthetic dataset...")
    X_train, X_test, y_train, y_test, feature_names, scaler = create_dataset(
        n_samples=2000, n_features=15, n_informative=8, seed=42,
    )

    # --- Step 2: Train MLP ---
    print("\n[2] Training TabularMLP...")
    model = TabularMLP(input_dim=X_train.shape[1])
    loss_history = train_model(model, X_train, y_train, epochs=100, lr=1e-3)
    acc = evaluate_model(model, X_test, y_test)
    print(f"  Test accuracy: {acc:.4f}")
    print(f"  Final training loss: {loss_history[-1]:.4f}")

    # --- Step 3: Prepare background and test subsets ---
    # Background: a representative subset of training data.
    # Why not the full training set? DeepSHAP averages over the background,
    # so more samples = more accurate but slower. 100 is a good trade-off.
    n_background = 100
    n_explain = 50  # Number of test samples to explain
    bg_idx = np.random.choice(X_train.shape[0], n_background, replace=False)
    background = X_train[bg_idx]
    test_subset = X_test[:n_explain]
    model.eval()

    # --- Step 4: DeepSHAP ---
    print(f"\n[3] Computing DeepSHAP ({n_explain} samples, "
          f"{n_background} background)...")
    t0 = time.time()
    deep_vals = compute_deep_shap(model, background, test_subset)
    t_deep = time.time() - t0
    print(f"  SHAP values shape: {deep_vals.shape}")
    print(f"  Time: {t_deep:.3f}s")

    # --- Step 5: GradientSHAP ---
    print(f"\n[4] Computing GradientSHAP ({n_explain} samples)...")
    t0 = time.time()
    grad_vals = compute_gradient_shap(model, background, test_subset, n_samples=200)
    t_grad = time.time() - t0
    print(f"  SHAP values shape: {grad_vals.shape}")
    print(f"  Time: {t_grad:.3f}s")

    # --- Step 6: KernelSHAP (model-agnostic baseline) ---
    # We use fewer samples because KernelSHAP is much slower
    n_kernel_explain = min(20, n_explain)
    print(f"\n[5] Computing KernelSHAP ({n_kernel_explain} samples, "
          f"model-agnostic baseline)...")
    t0 = time.time()
    kernel_vals = compute_kernel_shap(
        model, background, test_subset[:n_kernel_explain], n_background=30,
    )
    t_kernel = time.time() - t0
    print(f"  SHAP values shape: {kernel_vals.shape}")
    print(f"  Time: {t_kernel:.3f}s")

    # --- Step 7: Feature importance visualization ---
    print("\n[6] Generating feature importance plots...")
    plot_feature_importance(
        deep_vals, feature_names,
        title="DeepSHAP Feature Importance",
        save_path="deep_shap_importance.png",
    )

    # --- Step 8: Summary (beeswarm) plot ---
    print("\n[7] Generating SHAP summary plot...")
    plot_shap_summary(deep_vals, test_subset, feature_names,
                      save_path="shap_summary.png")

    # --- Step 9: Method comparison ---
    print("\n[8] Comparing SHAP methods...")
    # Align dimensions: use only the first n_kernel_explain samples
    plot_method_comparison(
        deep_vals[:n_kernel_explain],
        grad_vals[:n_kernel_explain],
        kernel_vals,
        feature_names,
    )

    # --- Step 10: Rank correlation between methods ---
    print("\n[9] Computing method agreement (Spearman rank correlation)...")
    try:
        corr_dg = compute_method_correlation(deep_vals, grad_vals)
        corr_dk = compute_method_correlation(
            deep_vals[:n_kernel_explain], kernel_vals,
        )
        corr_gk = compute_method_correlation(
            grad_vals[:n_kernel_explain], kernel_vals,
        )
        print(f"  DeepSHAP vs GradientSHAP: {corr_dg:.4f}")
        print(f"  DeepSHAP vs KernelSHAP:   {corr_dk:.4f}")
        print(f"  GradientSHAP vs KernelSHAP: {corr_gk:.4f}")
    except ImportError:
        print("  (scipy not available -- skipping correlation analysis)")

    # --- Step 11: Interaction values ---
    print("\n[10] Computing SHAP interaction values (5 samples)...")
    # Interaction values are expensive: O(D^2 * forward_passes).
    # We use only 5 samples to keep runtime reasonable.
    n_interact = 5
    t0 = time.time()
    interaction_vals = compute_interaction_values(
        model, background, test_subset[:n_interact],
    )
    t_interact = time.time() - t0
    print(f"  Interaction values shape: {interaction_vals.shape}")
    print(f"  Time: {t_interact:.3f}s")

    plot_interaction_heatmap(interaction_vals, feature_names)

    # Print the top-5 strongest interactions (off-diagonal)
    mean_interact = np.abs(interaction_vals).mean(axis=0)
    # Zero out the diagonal (main effects) to find pure interactions
    np.fill_diagonal(mean_interact, 0)
    # Find top interactions
    flat_idx = np.argsort(mean_interact.flatten())[::-1]
    print("\n  Top-5 feature interactions:")
    seen = set()
    count = 0
    for idx in flat_idx:
        i, j = divmod(idx, len(feature_names))
        if i >= j:
            continue  # Skip duplicates and diagonal
        pair = (i, j)
        if pair not in seen:
            seen.add(pair)
            print(f"    {feature_names[i]} x {feature_names[j]}: "
                  f"{mean_interact[i, j]:.6f}")
            count += 1
            if count >= 5:
                break

    # --- Step 12: Timing benchmark ---
    print("\n[11] Running timing benchmark (3 runs each)...")
    # Use a smaller subset for the benchmark to keep total time manageable
    bench_samples = test_subset[:10]
    timing_results = run_timing_benchmark(
        model, background, bench_samples, n_runs=3,
    )

    print(f"\n  {'Method':<18} {'Mean':>8} {'Std':>8}")
    print("  " + "-" * 36)
    for method, result in timing_results.items():
        print(f"  {method:<18} {result['mean']:>7.3f}s {result['std']:>7.3f}s")

    plot_timing_benchmark(timing_results)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  Observations:")
    print("  - DeepSHAP is fastest for neural networks because it propagates")
    print("    SHAP values through the network using DeepLIFT rules.")
    print("  - GradientSHAP is slightly slower but handles non-ReLU")
    print("    activations more gracefully via sampling.")
    print("  - KernelSHAP is model-agnostic but orders of magnitude slower")
    print("    because it treats the model as a black box.")
    print("  - All three methods should produce similar feature rankings")
    print("    if the network is well-behaved (high Spearman correlation).")
    print("  - Interaction values reveal feature synergies invisible to")
    print("    main-effect-only SHAP analysis.")
    print("=" * 60)


if __name__ == "__main__":
    main()
