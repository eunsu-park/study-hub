"""
06. Testing with Concept Activation Vectors (TCAV)

Implements a simplified version of TCAV (Kim et al. 2018) to test whether
high-level human concepts (e.g. "striped", "dotted") influence a CNN's
predictions.  A Concept Activation Vector (CAV) is the normal to a hyperplane
that separates concept-positive from concept-negative activations in a
hidden layer.  TCAV scores measure how sensitive the model's predictions are
to movement in the concept direction.

Covered topics:
    - Extracting intermediate activations from a CNN layer via hooks
    - Training a linear CAV (SVM) from concept vs. random examples
    - Computing directional derivatives and TCAV scores
    - Statistical significance testing with random concept baselines
    - Visualization of concept sensitivity across classes

Related to: L07 - Concept-Based Explanations

Requirements:
    pip install torch torchvision scikit-learn numpy matplotlib
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# ====== Section 1: Synthetic Concept Image Generator ======

def generate_concept_images(
    concept: str,
    n_images: int = 100,
    image_size: int = 64,
    seed: int = None,
) -> torch.Tensor:
    """Generate simple synthetic images representing visual concepts.

    Instead of requiring real image datasets, we create procedural images
    with clearly identifiable visual patterns.  This keeps the example
    self-contained while still exercising the full TCAV pipeline.

    Concepts:
      - "striped"  : horizontal stripes with varying frequency
      - "dotted"   : random circular dots on a background
      - "edges"    : strong vertical/horizontal edges
      - "random"   : uniform random noise (baseline / negative concept)
    """
    if seed is not None:
        np.random.seed(seed)

    images = np.zeros((n_images, 3, image_size, image_size), dtype=np.float32)

    for i in range(n_images):
        if concept == "striped":
            # Horizontal stripes with random frequency and phase
            freq = np.random.uniform(3, 8)
            phase = np.random.uniform(0, 2 * np.pi)
            y = np.linspace(0, 2 * np.pi * freq, image_size)
            pattern = (np.sin(y + phase) > 0).astype(np.float32)
            # Apply to all channels with slight color variation
            for c in range(3):
                color = np.random.uniform(0.3, 1.0)
                images[i, c] = pattern[:, None] * color

        elif concept == "dotted":
            # Random circular dots on a dark background
            bg = np.random.uniform(0, 0.2)
            images[i, :] = bg
            n_dots = np.random.randint(5, 15)
            for _ in range(n_dots):
                cx = np.random.randint(4, image_size - 4)
                cy = np.random.randint(4, image_size - 4)
                r = np.random.randint(2, 5)
                yy, xx = np.ogrid[-cx:image_size - cx, -cy:image_size - cy]
                mask = (xx ** 2 + yy ** 2) <= r ** 2
                color = np.random.uniform(0.5, 1.0, 3)
                for c in range(3):
                    images[i, c, mask] = color[c]

        elif concept == "edges":
            # Strong vertical or horizontal edge in a random position
            images[i, :] = np.random.uniform(0.1, 0.3)
            if np.random.random() > 0.5:
                # Vertical edge
                pos = np.random.randint(image_size // 4, 3 * image_size // 4)
                images[i, :, :, pos:] = np.random.uniform(0.7, 1.0)
            else:
                # Horizontal edge
                pos = np.random.randint(image_size // 4, 3 * image_size // 4)
                images[i, :, pos:, :] = np.random.uniform(0.7, 1.0)

        elif concept == "random":
            # Pure random noise -- serves as the negative concept
            images[i] = np.random.uniform(0, 1, (3, image_size, image_size))

        else:
            raise ValueError(f"Unknown concept: {concept}")

    return torch.tensor(images)


# ====== Section 2: Simple CNN Classifier ======

class SimpleCNN(nn.Module):
    """Small CNN for multi-class image classification.

    Architecture is kept minimal because we need to inspect intermediate
    activations -- a deep network would obscure the TCAV demonstration.
    Three conv blocks feed into a two-layer classifier head.
    """

    def __init__(self, n_classes: int = 4):
        super().__init__()
        # Three convolutional blocks with increasing channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        # Global average pooling + classifier head
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # Global average pooling: (B, 64, H, W) -> (B, 64)
        x = x.mean(dim=[2, 3])
        return self.classifier(x)


def train_classifier(
    model: SimpleCNN,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int = 30,
    lr: float = 1e-3,
) -> list[float]:
    """Train the CNN classifier and return per-epoch losses."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        model.train()
        logits = model(X_train)
        loss = F.cross_entropy(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses


# ====== Section 3: Activation Extraction via Hooks ======

class ActivationExtractor:
    """Extract activations from a named layer using forward hooks.

    PyTorch hooks let us capture intermediate tensors without modifying
    the model's forward() method.  We register a hook on the target
    layer, run a forward pass, and collect the output.  The hook is
    automatically removed after extraction to prevent memory leaks.
    """

    def __init__(self, model: nn.Module, layer_name: str):
        self.model = model
        self.layer_name = layer_name
        self.activations = None
        self._hook_handle = None

    def _hook_fn(self, module, input, output):
        """Store the layer's output tensor during forward pass."""
        # Global-average-pool spatial dimensions for a compact vector
        if output.dim() == 4:
            self.activations = output.mean(dim=[2, 3]).detach()
        else:
            self.activations = output.detach()

    def extract(self, images: torch.Tensor) -> np.ndarray:
        """Run forward pass and return activations as a numpy array.

        We process in eval mode with no_grad to avoid unnecessary
        computation and gradient tracking.
        """
        # Locate the target layer by name
        target_layer = dict(self.model.named_modules())[self.layer_name]
        self._hook_handle = target_layer.register_forward_hook(self._hook_fn)

        self.model.eval()
        with torch.no_grad():
            _ = self.model(images)

        # Clean up the hook immediately
        self._hook_handle.remove()
        self._hook_handle = None

        return self.activations.numpy()


# ====== Section 4: CAV Training ======

def train_cav(
    concept_activations: np.ndarray,
    random_activations: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Train a Concept Activation Vector using a linear SVM.

    The CAV is the weight vector (normal to the decision hyperplane) of
    a linear classifier that separates concept-positive activations from
    random activations.  A high SVM accuracy indicates that the concept
    is linearly separable in the activation space -- a prerequisite for
    meaningful TCAV scores.

    Returns:
        cav: Unit-norm weight vector pointing in the concept direction.
        accuracy: SVM classification accuracy on a held-out validation set.
    """
    # Combine concept (label=1) and random (label=0) activations
    X = np.vstack([concept_activations, random_activations])
    y = np.array([1] * len(concept_activations) + [0] * len(random_activations))

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    # Linear SVM -- the weight vector IS the CAV
    svm = LinearSVC(max_iter=5000, random_state=42)
    svm.fit(X_train, y_train)

    accuracy = accuracy_score(y_val, svm.predict(X_val))

    # Extract and normalise the weight vector
    cav = svm.coef_[0].copy()
    cav = cav / (np.linalg.norm(cav) + 1e-10)

    return cav, accuracy


# ====== Section 5: TCAV Score Computation ======

def compute_tcav_score(
    model: nn.Module,
    images: torch.Tensor,
    target_class: int,
    cav: np.ndarray,
    layer_name: str,
) -> float:
    """Compute the TCAV score for a target class with respect to a concept.

    TCAV score = fraction of inputs for which the directional derivative
    of the target class logit along the CAV direction is positive.

    A score > 0.5 means the concept positively influences the model's
    prediction for this class; < 0.5 means negative influence.  Values
    near 0.5 indicate no meaningful relationship.

    We compute directional derivatives by:
      1. Enabling gradient tracking on the target layer's output.
      2. Forward-passing to get class logits.
      3. Back-propagating to get gradients w.r.t. the layer output.
      4. Projecting those gradients onto the CAV direction.
    """
    model.eval()
    cav_tensor = torch.tensor(cav, dtype=torch.float32)

    positive_count = 0
    total_count = 0

    # Process images one at a time to get per-sample directional derivatives
    for i in range(len(images)):
        img = images[i:i + 1]

        # Hook to capture and enable grad on intermediate activations
        activation_holder = {}

        def hook_fn(module, input, output):
            # Keep the activation in the graph so we can differentiate through it
            if output.dim() == 4:
                activation_holder["act"] = output.mean(dim=[2, 3])
            else:
                activation_holder["act"] = output

        target_layer = dict(model.named_modules())[layer_name]
        handle = target_layer.register_forward_hook(hook_fn)

        # We need grad w.r.t. intermediate activations, so enable grad
        img_input = img.clone().requires_grad_(False)
        logits = model(img_input)

        handle.remove()

        # Get the activation and compute gradient of target logit w.r.t. it
        act = activation_holder["act"]
        act.retain_grad()

        target_logit = logits[0, target_class]
        model.zero_grad()
        target_logit.backward(retain_graph=False)

        if act.grad is not None:
            grad = act.grad[0].detach()
            # Directional derivative = dot product of gradient and CAV
            directional_deriv = torch.dot(grad, cav_tensor).item()
            if directional_deriv > 0:
                positive_count += 1
        total_count += 1

    return positive_count / max(total_count, 1)


# ====== Section 6: Statistical Significance Testing ======

def tcav_significance_test(
    model: nn.Module,
    test_images: torch.Tensor,
    target_class: int,
    concept_cav: np.ndarray,
    layer_name: str,
    extractor: ActivationExtractor,
    n_random_runs: int = 10,
    n_random_images: int = 80,
    significance_level: float = 0.05,
) -> dict:
    """Test whether a TCAV score is statistically significant.

    We compare the real concept's TCAV score against a distribution of
    TCAV scores computed with *random* CAVs (trained on random-vs-random
    data).  If the real score lies outside the random distribution, the
    concept has a meaningful relationship with the target class.

    Uses a two-sided test: the real score must be more extreme than
    (1 - significance_level) of random scores.
    """
    # Compute the real TCAV score
    real_score = compute_tcav_score(
        model, test_images, target_class, concept_cav, layer_name,
    )

    # Generate random CAVs by training on random-vs-random splits
    random_scores = []
    for run in range(n_random_runs):
        # Two independent sets of random images
        rand_a = generate_concept_images("random", n_random_images, seed=run * 100)
        rand_b = generate_concept_images("random", n_random_images, seed=run * 100 + 50)

        act_a = extractor.extract(rand_a)
        act_b = extractor.extract(rand_b)

        random_cav, _ = train_cav(act_a, act_b)
        rand_score = compute_tcav_score(
            model, test_images, target_class, random_cav, layer_name,
        )
        random_scores.append(rand_score)

    random_scores = np.array(random_scores)
    random_mean = random_scores.mean()
    random_std = random_scores.std()

    # Two-sided p-value: fraction of random scores at least as extreme
    if random_std > 1e-8:
        z_score = abs(real_score - random_mean) / random_std
    else:
        z_score = 0.0

    # Empirical p-value
    more_extreme = np.sum(np.abs(random_scores - random_mean)
                          >= abs(real_score - random_mean))
    p_value = (more_extreme + 1) / (n_random_runs + 1)

    return {
        "real_score": real_score,
        "random_mean": random_mean,
        "random_std": random_std,
        "z_score": z_score,
        "p_value": p_value,
        "significant": p_value < significance_level,
        "random_scores": random_scores,
    }


# ====== Section 7: Visualization ======

def visualize_tcav_results(
    results: dict[str, dict],
    class_names: list[str],
    save_path: str = "tcav_results.png",
) -> None:
    """Bar chart of TCAV scores across concepts and classes.

    Each group of bars represents a target class, with one bar per
    concept.  A dashed line at 0.5 marks the "no influence" baseline.
    Significant results are marked with an asterisk.
    """
    concepts = list(results.keys())
    n_concepts = len(concepts)
    n_classes = len(class_names)

    fig, ax = plt.subplots(figsize=(10, 5))

    bar_width = 0.8 / n_concepts
    x = np.arange(n_classes)

    colors = plt.cm.Set2(np.linspace(0, 1, n_concepts))

    for i, concept in enumerate(concepts):
        scores = []
        significances = []
        for cls_name in class_names:
            key = f"{concept}_{cls_name}"
            if key in results[concept]:
                scores.append(results[concept][key]["real_score"])
                significances.append(results[concept][key]["significant"])
            else:
                scores.append(0.5)
                significances.append(False)

        offset = (i - n_concepts / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, scores, bar_width, label=concept,
                      color=colors[i], edgecolor="black", linewidth=0.5)

        # Mark significant results with an asterisk
        for j, (bar, sig) in enumerate(zip(bars, significances)):
            if sig:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        "*", ha="center", va="bottom", fontsize=14, fontweight="bold")

    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7,
               label="No influence (0.5)")
    ax.set_xlabel("Target Class", fontsize=12)
    ax.set_ylabel("TCAV Score", fontsize=12)
    ax.set_title("TCAV Scores: Concept Influence on Model Predictions", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Figure saved to: {save_path}")
    plt.close()


# ====== Section 8: Main Pipeline ======

def main() -> None:
    """Run the full TCAV pipeline on a synthetic concept classification task."""
    print("=" * 65)
    print("  TCAV -- Testing with Concept Activation Vectors")
    print("  Concept Sensitivity | CAV Training | Statistical Testing")
    print("=" * 65)

    # --- Step 1: Generate training data (4-class problem) ---
    print("\n[1] Generating Synthetic Image Dataset (4 classes)")
    print("-" * 50)

    CLASS_NAMES = ["striped", "dotted", "edges", "random"]
    N_PER_CLASS = 150
    IMAGE_SIZE = 64

    all_images = []
    all_labels = []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        imgs = generate_concept_images(cls_name, N_PER_CLASS, IMAGE_SIZE,
                                       seed=cls_idx * 42)
        all_images.append(imgs)
        all_labels.extend([cls_idx] * N_PER_CLASS)

    X_all = torch.cat(all_images, dim=0)
    y_all = torch.tensor(all_labels, dtype=torch.long)

    # Shuffle and split
    perm = torch.randperm(len(X_all))
    X_all = X_all[perm]
    y_all = y_all[perm]

    split = int(0.8 * len(X_all))
    X_train, X_test = X_all[:split], X_all[split:]
    y_train, y_test = y_all[:split], y_all[split:]

    print(f"  Total images: {len(X_all)} ({N_PER_CLASS} per class)")
    print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")
    print(f"  Image shape: {tuple(X_train[0].shape)}")

    # --- Step 2: Train CNN classifier ---
    print("\n[2] Training CNN Classifier")
    print("-" * 50)

    model = SimpleCNN(n_classes=len(CLASS_NAMES))
    losses = train_classifier(model, X_train, y_train, epochs=40)
    print(f"  Final training loss: {losses[-1]:.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        preds = logits.argmax(dim=1)
        test_acc = (preds == y_test).float().mean().item()
    print(f"  Test accuracy: {test_acc:.2%}")

    # Per-class accuracy
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        mask = y_test == cls_idx
        if mask.sum() > 0:
            cls_acc = (preds[mask] == y_test[mask]).float().mean().item()
            print(f"    {cls_name:10s}: {cls_acc:.2%} ({mask.sum().item()} samples)")

    # --- Step 3: Extract activations and train CAVs ---
    print("\n[3] Training Concept Activation Vectors (CAVs)")
    print("-" * 50)

    # We probe the third convolutional block -- deep enough to have
    # learned meaningful features, shallow enough for linear separability
    LAYER_NAME = "conv3"
    extractor = ActivationExtractor(model, LAYER_NAME)

    # Generate dedicated concept and random images for CAV training
    # (separate from the classifier training data to avoid data leakage)
    N_CAV = 100
    CONCEPTS = ["striped", "dotted", "edges"]

    cavs = {}
    for concept in CONCEPTS:
        concept_imgs = generate_concept_images(concept, N_CAV, IMAGE_SIZE,
                                               seed=hash(concept) % 10000)
        random_imgs = generate_concept_images("random", N_CAV, IMAGE_SIZE,
                                              seed=hash(concept) % 10000 + 1)

        concept_acts = extractor.extract(concept_imgs)
        random_acts = extractor.extract(random_imgs)

        cav, svm_acc = train_cav(concept_acts, random_acts)
        cavs[concept] = cav

        print(f"  Concept '{concept}':  SVM accuracy = {svm_acc:.2%}  "
              f"|  CAV shape = {cav.shape}")

    # --- Step 4: Compute TCAV scores ---
    print("\n[4] Computing TCAV Scores")
    print("-" * 50)

    # Use a subset of test images per class for TCAV computation
    # (smaller subset keeps runtime manageable)
    N_TCAV_SAMPLES = 20

    tcav_results = {concept: {} for concept in CONCEPTS}

    for concept in CONCEPTS:
        print(f"\n  Concept: '{concept}'")
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            # Get test images for this class
            mask = y_test == cls_idx
            cls_images = X_test[mask][:N_TCAV_SAMPLES]

            if len(cls_images) == 0:
                continue

            score = compute_tcav_score(
                model, cls_images, cls_idx, cavs[concept], LAYER_NAME,
            )
            tcav_results[concept][f"{concept}_{cls_name}"] = {"real_score": score}
            print(f"    -> class '{cls_name}': TCAV = {score:.3f}  "
                  f"({'positive' if score > 0.5 else 'negative' if score < 0.5 else 'neutral'})")

    # --- Step 5: Statistical significance testing ---
    print("\n[5] Statistical Significance Testing")
    print("-" * 50)
    print("  Running random-CAV baseline comparisons ...")

    for concept in CONCEPTS:
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            mask = y_test == cls_idx
            cls_images = X_test[mask][:N_TCAV_SAMPLES]
            if len(cls_images) == 0:
                continue

            sig_result = tcav_significance_test(
                model, cls_images, cls_idx, cavs[concept], LAYER_NAME,
                extractor, n_random_runs=8, n_random_images=N_CAV,
            )

            # Store the full result for visualization
            tcav_results[concept][f"{concept}_{cls_name}"] = sig_result

            sig_marker = "*" if sig_result["significant"] else " "
            print(f"  [{sig_marker}] concept='{concept}', class='{cls_name}': "
                  f"TCAV={sig_result['real_score']:.3f}  "
                  f"random_mean={sig_result['random_mean']:.3f} "
                  f"+/- {sig_result['random_std']:.3f}  "
                  f"p={sig_result['p_value']:.3f}")

    # --- Step 6: Visualization ---
    print("\n[6] Generating TCAV Visualization")
    print("-" * 50)

    visualize_tcav_results(tcav_results, CLASS_NAMES)

    # --- Summary ---
    print("\n" + "=" * 65)
    print("  Summary")
    print("=" * 65)
    print("""
  TCAV answers: "Is concept C important for the model's prediction
  of class K?"

  Key takeaways:
    1. A CAV is a linear probe that finds the direction in activation
       space corresponding to a human concept.
    2. TCAV score > 0.5 means the concept positively influences
       predictions for that class; < 0.5 means negative influence.
    3. Statistical testing against random CAVs is essential to avoid
       false conclusions -- many directions in high-dimensional space
       will appear meaningful by chance.
    4. High SVM accuracy when training the CAV indicates the concept
       is linearly separable in the layer -- a necessary condition
       for reliable TCAV scores.
    5. Layer choice matters: earlier layers capture textures, later
       layers capture higher-level features.
    """)


if __name__ == "__main__":
    main()
