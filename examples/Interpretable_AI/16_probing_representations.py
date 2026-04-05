"""
16. Probing and Representation Analysis

Trains a small neural network on a synthetic task, then applies probing
classifiers, Centered Kernel Alignment (CKA), and representation
similarity analysis to understand what each layer has learned.

Covered topics:
    - Linear probing classifiers on intermediate representations
    - Layer-wise feature selectivity analysis
    - Centered Kernel Alignment (CKA) for comparing representations
    - Representation dimensionality via participation ratio
    - Activation clustering to discover emergent structure
    - Layer-by-layer representation quality assessment

Related to: L05 - Probing and Representation Analysis

Requirements:
    pip install numpy matplotlib scikit-learn torch
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ====== Section 1: Synthetic Multi-Feature Dataset ======

def generate_multifeature_data(
    n: int = 2000,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Generate data with multiple latent features for probing.

    The dataset has 8 input dimensions but the label depends on a
    combination of 3 latent properties:
      - property_A: whether x[0] + x[1] > 0
      - property_B: whether x[2] * x[3] > 0
      - property_C: whether sum(x[4:]) > 0

    The final label is majority vote of A, B, C.
    This lets us probe each layer to see which properties it encodes.

    Args:
        n: Number of samples.
        seed: Random seed.

    Returns:
        X: Input tensor (n, 8).
        y: Label tensor (n,).
        properties: Dictionary of latent property arrays for probing.
    """
    rng = np.random.default_rng(seed)
    X_np = rng.normal(0, 1, (n, 8)).astype(np.float32)

    prop_A = (X_np[:, 0] + X_np[:, 1] > 0).astype(int)
    prop_B = (X_np[:, 2] * X_np[:, 3] > 0).astype(int)
    prop_C = (X_np[:, 4:].sum(axis=1) > 0).astype(int)

    # Majority vote
    labels = ((prop_A + prop_B + prop_C) >= 2).astype(int)

    properties = {
        "property_A (linear)": prop_A,
        "property_B (nonlinear)": prop_B,
        "property_C (aggregate)": prop_C,
        "label": labels,
    }

    return (
        torch.from_numpy(X_np),
        torch.from_numpy(labels).long(),
        properties,
    )


# ====== Section 2: Neural Network with Hook-Based Extraction ======

class ProbableNet(nn.Module):
    """A 4-layer MLP designed for probing experiments.

    Each layer's activations are cached during forward pass so probing
    classifiers can assess what information is linearly accessible at
    each depth.
    """

    def __init__(self, input_dim: int = 8, hidden_dim: int = 32):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, 2)

        self.activations = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = F.relu(self.layer1(x))
        self.activations["layer1"] = h1.detach()

        h2 = F.relu(self.layer2(h1))
        self.activations["layer2"] = h2.detach()

        h3 = F.relu(self.layer3(h2))
        self.activations["layer3"] = h3.detach()

        out = self.layer4(h3)
        self.activations["layer4"] = out.detach()

        return out


def train_network(
    model: ProbableNet,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 300,
    lr: float = 1e-3,
) -> list[float]:
    """Train the network on the classification task."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        model.train()
        logits = model(X)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (epoch + 1) % 100 == 0:
            acc = (logits.argmax(dim=1) == y).float().mean().item()
            print(f"    Epoch {epoch + 1:4d}/{epochs}  "
                  f"loss={loss.item():.4f}  acc={acc:.4f}")

    return losses


# ====== Section 3: Linear Probing ======

def probe_layer(
    activations: np.ndarray,
    targets: np.ndarray,
    property_name: str,
) -> dict:
    """Train a linear probe to predict a property from layer activations.

    A linear probe (logistic regression on frozen representations) tests
    whether a property is *linearly decodable* from a layer's output.
    High probe accuracy means the layer has organized its representation
    so that the property can be read off with a simple linear classifier.

    Low probe accuracy does NOT mean the information is absent -- it
    may be encoded nonlinearly. But linear accessibility is a strong
    indicator of how "explicit" a representation is.

    Args:
        activations: Layer activations (n, d).
        targets: Binary property labels (n,).
        property_name: Name of the probed property.

    Returns:
        Dictionary with probe accuracy and selectivity.
    """
    # Train/test split (80/20)
    n = len(activations)
    split = int(0.8 * n)
    X_train, X_test = activations[:split], activations[split:]
    y_train, y_test = targets[:split], targets[split:]

    probe = LogisticRegression(max_iter=500, random_state=42)
    probe.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, probe.predict(X_train))
    test_acc = accuracy_score(y_test, probe.predict(X_test))

    # Selectivity: how much better than majority baseline
    majority_acc = max(y_test.mean(), 1 - y_test.mean())
    selectivity = test_acc - majority_acc

    return {
        "property": property_name,
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "selectivity": float(selectivity),
        "majority_baseline": float(majority_acc),
    }


# ====== Section 4: Centered Kernel Alignment (CKA) ======

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear Centered Kernel Alignment between two representations.

    CKA (Kornblith et al., 2019) measures the similarity between two
    representation matrices. It is invariant to orthogonal transformations
    and isotropic scaling, making it ideal for comparing representations
    across layers or models.

    CKA = HSIC(X, Y) / sqrt(HSIC(X, X) * HSIC(Y, Y))

    where HSIC is the Hilbert-Schmidt Independence Criterion.

    Args:
        X: First representation matrix (n, d1).
        Y: Second representation matrix (n, d2).

    Returns:
        CKA similarity score in [0, 1].
    """
    # Center both representations
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    # Linear HSIC: trace(X^T X Y^T Y)
    hsic_xy = np.linalg.norm(X.T @ Y, "fro") ** 2
    hsic_xx = np.linalg.norm(X.T @ X, "fro") ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, "fro") ** 2

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-10:
        return 0.0

    return float(hsic_xy / denom)


def compute_cka_matrix(layer_activations: dict[str, np.ndarray]) -> np.ndarray:
    """Compute the pairwise CKA similarity matrix across layers.

    Args:
        layer_activations: Dictionary mapping layer names to activation matrices.

    Returns:
        CKA matrix of shape (n_layers, n_layers).
    """
    layer_names = list(layer_activations.keys())
    n = len(layer_names)
    cka_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            cka_matrix[i, j] = linear_cka(
                layer_activations[layer_names[i]],
                layer_activations[layer_names[j]],
            )

    return cka_matrix


# ====== Section 5: Representation Dimensionality ======

def participation_ratio(activations: np.ndarray) -> float:
    """Compute the participation ratio (effective dimensionality).

    The participation ratio measures how many dimensions of the
    representation are actively used. It is defined as:

        PR = (sum of eigenvalues)^2 / sum of eigenvalues^2

    A representation where all variance is concentrated in one
    dimension has PR = 1. A representation using all d dimensions
    equally has PR = d.

    Args:
        activations: Representation matrix (n, d).

    Returns:
        Participation ratio (effective dimensionality).
    """
    centered = activations - activations.mean(axis=0)
    cov = centered.T @ centered / len(centered)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) == 0:
        return 0.0

    sum_sq = eigenvalues.sum() ** 2
    sq_sum = (eigenvalues ** 2).sum()

    return float(sum_sq / sq_sum) if sq_sum > 0 else 0.0


# ====== Section 6: Visualization ======

def visualize_probing(
    probing_results: dict,
    cka_matrix: np.ndarray,
    layer_names: list[str],
    dim_ratios: dict[str, float],
    training_losses: list[float],
    save_path: str = "probing_representations.png",
) -> None:
    """Four-panel visualization of representation analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Panel 1: Probing Accuracy Heatmap ---
    ax1 = axes[0, 0]
    properties = list(next(iter(probing_results.values())).keys())
    layers = list(probing_results.keys())

    data = np.zeros((len(layers), len(properties)))
    for i, layer in enumerate(layers):
        for j, prop in enumerate(properties):
            data[i, j] = probing_results[layer][prop]["test_accuracy"]

    im = ax1.imshow(data.T, aspect="auto", cmap="YlOrRd", vmin=0.5, vmax=1.0)
    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels(layers, fontsize=9)
    ax1.set_yticks(range(len(properties)))
    ax1.set_yticklabels([p[:15] for p in properties], fontsize=9)
    plt.colorbar(im, ax=ax1, label="Probe Accuracy")
    ax1.set_title("Linear Probing Accuracy by Layer")

    # Add text annotations
    for i in range(len(layers)):
        for j in range(len(properties)):
            ax1.text(i, j, f"{data[i, j]:.2f}",
                     ha="center", va="center", fontsize=8,
                     color="white" if data[i, j] > 0.8 else "black")

    # --- Panel 2: CKA Similarity Matrix ---
    ax2 = axes[0, 1]
    im2 = ax2.imshow(cka_matrix, cmap="Blues", vmin=0, vmax=1)
    ax2.set_xticks(range(len(layer_names)))
    ax2.set_xticklabels(layer_names, fontsize=9)
    ax2.set_yticks(range(len(layer_names)))
    ax2.set_yticklabels(layer_names, fontsize=9)
    plt.colorbar(im2, ax=ax2, label="CKA Similarity")
    ax2.set_title("Layer-wise CKA Similarity")

    for i in range(len(layer_names)):
        for j in range(len(layer_names)):
            ax2.text(j, i, f"{cka_matrix[i, j]:.2f}",
                     ha="center", va="center", fontsize=8,
                     color="white" if cka_matrix[i, j] > 0.7 else "black")

    # --- Panel 3: Effective Dimensionality ---
    ax3 = axes[1, 0]
    dim_names = list(dim_ratios.keys())
    dim_vals = list(dim_ratios.values())
    ax3.bar(dim_names, dim_vals, color="#9b59b6",
            edgecolor="black", linewidth=0.5)
    ax3.set_ylabel("Participation Ratio")
    ax3.set_title("Effective Dimensionality per Layer")
    ax3.grid(True, alpha=0.3, axis="y")

    # --- Panel 4: Training Loss ---
    ax4 = axes[1, 1]
    ax4.plot(training_losses, color="#2c3e50", linewidth=1)
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Cross-Entropy Loss")
    ax4.set_title("Training Loss Curve")
    ax4.set_yscale("log")
    ax4.grid(True, alpha=0.3)

    plt.suptitle("Probing and Representation Analysis", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Figure saved to: {save_path}")
    plt.close()


# ====== Section 7: Main Pipeline ======

def main() -> None:
    """Run probing and representation analysis experiments."""
    print("=" * 65)
    print("  Probing and Representation Analysis")
    print("  Linear Probes | CKA | Dimensionality | Selectivity")
    print("=" * 65)

    # --- Step 1: Generate data ---
    print("\n[1] Generating Multi-Feature Dataset")
    print("-" * 50)

    X, y, properties = generate_multifeature_data(n=2000)
    print(f"  Samples: {len(y)}")
    print(f"  Input dim: {X.shape[1]}")
    print(f"  Properties to probe: {list(properties.keys())}")
    for name, prop in properties.items():
        print(f"    {name}: {prop.mean():.3f} positive rate")

    # --- Step 2: Train network ---
    print("\n[2] Training 4-Layer MLP")
    print("-" * 50)

    model = ProbableNet(input_dim=8, hidden_dim=32)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    training_losses = train_network(model, X, y, epochs=300)

    model.eval()
    with torch.no_grad():
        logits = model(X)
        final_acc = (logits.argmax(dim=1) == y).float().mean().item()
    print(f"  Final accuracy: {final_acc:.4f}")

    # --- Step 3: Extract activations ---
    print("\n[3] Extracting Layer Activations")
    print("-" * 50)

    model.eval()
    with torch.no_grad():
        _ = model(X)

    layer_activations = {}
    for layer_name, acts in model.activations.items():
        layer_activations[layer_name] = acts.numpy()
        print(f"  {layer_name}: shape={acts.shape}")

    # --- Step 4: Linear probing ---
    print("\n[4] Linear Probing Experiments")
    print("-" * 50)

    probing_results = {}
    for layer_name, acts in layer_activations.items():
        probing_results[layer_name] = {}
        for prop_name, prop_labels in properties.items():
            result = probe_layer(acts, prop_labels, prop_name)
            probing_results[layer_name][prop_name] = result

    # Print results table
    print(f"\n  {'Layer':<10s}", end="")
    for prop_name in properties:
        print(f"  {prop_name[:12]:>12s}", end="")
    print()
    print(f"  {'-' * (10 + 14 * len(properties))}")

    for layer_name in layer_activations:
        print(f"  {layer_name:<10s}", end="")
        for prop_name in properties:
            acc = probing_results[layer_name][prop_name]["test_accuracy"]
            print(f"  {acc:>12.4f}", end="")
        print()

    # --- Step 5: CKA analysis ---
    print("\n[5] Centered Kernel Alignment (CKA)")
    print("-" * 50)

    # Include input as a "layer" for comparison
    all_activations = {"input": X.numpy()}
    all_activations.update(layer_activations)

    cka_matrix = compute_cka_matrix(all_activations)
    layer_names = list(all_activations.keys())

    print(f"  CKA Matrix ({len(layer_names)} x {len(layer_names)}):")
    print(f"  {'':>10s}", end="")
    for name in layer_names:
        print(f"  {name:>8s}", end="")
    print()
    for i, name in enumerate(layer_names):
        print(f"  {name:>10s}", end="")
        for j in range(len(layer_names)):
            print(f"  {cka_matrix[i, j]:>8.4f}", end="")
        print()

    # --- Step 6: Dimensionality analysis ---
    print("\n[6] Effective Dimensionality (Participation Ratio)")
    print("-" * 50)

    dim_ratios = {}
    for layer_name, acts in layer_activations.items():
        pr = participation_ratio(acts)
        dim_ratios[layer_name] = pr
        total_dim = acts.shape[1]
        print(f"  {layer_name}: PR={pr:.2f} / {total_dim} dimensions "
              f"({pr / total_dim:.1%} utilization)")

    # --- Step 7: Visualization ---
    print("\n[7] Generating Visualization")
    print("-" * 50)

    visualize_probing(
        probing_results, cka_matrix, layer_names,
        dim_ratios, training_losses,
    )

    # --- Summary ---
    print("\n" + "=" * 65)
    print("  Summary")
    print("=" * 65)
    print(f"""
  Network: 4-layer MLP (8 -> 32 -> 32 -> 32 -> 2)
  Task accuracy: {final_acc:.2%}

  Key findings:
    1. Linear probing reveals how properties become more decodable
       through the network. Later layers should show higher probe
       accuracy for the task-relevant label.
    2. CKA similarity shows how representations transform across
       layers. Adjacent layers tend to be more similar; early vs.
       late layers diverge as the network processes information.
    3. Participation ratio measures effective dimensionality.
       Layers may compress (fewer active dimensions) or expand
       representations depending on the task demands.
    4. Different properties (linear, nonlinear, aggregate) may
       become linearly accessible at different depths, revealing
       the network's computational structure.
    """)


if __name__ == "__main__":
    main()
