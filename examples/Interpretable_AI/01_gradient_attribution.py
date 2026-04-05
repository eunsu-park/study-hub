"""
01. Gradient-Based Attribution Methods

Implements three fundamental gradient attribution techniques from scratch
using PyTorch on a pretrained ResNet18. These methods answer the question
"which input pixels mattered most for this prediction?" by computing how
the output changes with respect to each pixel.

Covered topics:
    - Vanilla saliency maps (gradient magnitude)
    - Integrated Gradients with configurable steps and baseline
    - SmoothGrad with Gaussian noise sampling
    - Side-by-side visualization of all methods
    - ImageNet class prediction with pretrained ResNet18

Related to: L02 - Gradient-Based Attribution

Requirements:
    pip install torch torchvision matplotlib numpy Pillow
"""

import io
import time
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


# ====== Helper Utilities ======

def load_sample_image(url: str = None) -> Image.Image:
    """Download a sample image or create a synthetic one if download fails.

    We use a publicly available ImageNet sample (a golden retriever) so
    that the pretrained ResNet18 produces meaningful class predictions.
    If the download fails, we generate a synthetic gradient image that
    still exercises all code paths.
    """
    if url is None:
        url = (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/"
            "2/26/YellowLabradorLooking_new.jpg/1200px-YellowLabradorLooking_new.jpg"
        )
    try:
        print(f"  Downloading sample image from:\n    {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            img = Image.open(io.BytesIO(resp.read())).convert("RGB")
        print("  Download successful.")
        return img
    except Exception as e:
        print(f"  Download failed ({e}). Using synthetic gradient image.")
        # Create a 224x224 RGB gradient as fallback
        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        arr[:, :, 0] = np.linspace(0, 255, 224, dtype=np.uint8)[None, :]
        arr[:, :, 1] = np.linspace(0, 255, 224, dtype=np.uint8)[:, None]
        arr[:, :, 2] = 128
        return Image.fromarray(arr)


def get_imagenet_transform() -> transforms.Compose:
    """Standard ImageNet preprocessing pipeline.

    ResNet expects 224x224 images normalized with the ImageNet channel
    means and standard deviations. This transform is identical to what
    was used during training.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # ImageNet channel-wise normalization
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalization for display purposes.

    Converts a CHW tensor back to a HWC numpy array with pixel values
    clipped to [0, 1] so matplotlib can render it correctly.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


# ====== Vanilla Saliency Map ======

def compute_vanilla_saliency(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
) -> np.ndarray:
    """Compute a vanilla saliency map via input gradients.

    The simplest attribution method: take the gradient of the target
    class score with respect to the input pixels, then collapse across
    color channels by taking the maximum absolute gradient.

    Reference: Simonyan et al., "Deep Inside Convolutional Networks:
    Visualising Image Classification Models and Saliency Maps" (2014).

    Args:
        model: A pretrained classifier in eval mode.
        input_tensor: Preprocessed image tensor of shape (1, C, H, W).
        target_class: ImageNet class index to attribute to.

    Returns:
        Saliency map as a 2D numpy array of shape (H, W).
    """
    # We need gradients w.r.t. the input, so enable requires_grad
    input_tensor = input_tensor.clone().detach().requires_grad_(True)

    # Forward pass
    output = model(input_tensor)
    score = output[0, target_class]

    # Backward pass — compute d(score)/d(input)
    model.zero_grad()
    score.backward()

    # The saliency is the max absolute gradient across color channels.
    # This captures "the channel that changed the score the most" at
    # each spatial location.
    saliency = input_tensor.grad.data.abs()  # shape: (1, 3, H, W)
    saliency, _ = saliency.max(dim=1)        # shape: (1, H, W)
    saliency = saliency.squeeze().cpu().numpy()

    return saliency


# ====== Integrated Gradients ======

def compute_integrated_gradients(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    baseline: torch.Tensor = None,
    steps: int = 50,
) -> np.ndarray:
    """Compute Integrated Gradients attribution.

    Unlike vanilla saliency, Integrated Gradients (IG) satisfies the
    axioms of Sensitivity and Implementation Invariance. It integrates
    the gradients along a straight-line path from a baseline (typically
    a black image) to the actual input.

    IG(x)_i = (x_i - x'_i) * integral_0^1 (dF/dx_i)(x' + a*(x - x')) da

    We approximate the integral with a Riemann sum over `steps` points.

    Reference: Sundararajan et al., "Axiomatic Attribution for Deep
    Networks" (ICML 2017).

    Args:
        model: A pretrained classifier in eval mode.
        input_tensor: Preprocessed image tensor of shape (1, C, H, W).
        target_class: ImageNet class index to attribute to.
        baseline: Reference input (default: zero tensor = black image).
        steps: Number of interpolation steps for the Riemann sum.
                Higher values yield more accurate attributions but cost
                more compute. 50-300 is typical.

    Returns:
        Attribution map as a 2D numpy array of shape (H, W).
    """
    if baseline is None:
        # Black image baseline — the most common choice because it
        # represents "absence of information" for each pixel
        baseline = torch.zeros_like(input_tensor)

    # Generate interpolation coefficients: alpha in [0, 1]
    # We use steps+1 points so the Riemann sum has `steps` intervals
    alphas = torch.linspace(0, 1, steps + 1).view(-1, 1, 1, 1)

    # Build all interpolated images at once: x' + alpha * (x - x')
    delta = input_tensor - baseline
    interpolated = baseline + alphas * delta  # shape: (steps+1, C, H, W)

    # Accumulate gradients across all interpolation points
    total_gradients = torch.zeros_like(input_tensor)

    for i in range(steps + 1):
        x_step = interpolated[i].unsqueeze(0).clone().detach().requires_grad_(True)
        output = model(x_step)
        score = output[0, target_class]

        model.zero_grad()
        score.backward()

        total_gradients += x_step.grad.data

    # Approximate the integral via the trapezoidal rule:
    # integral ~ (sum of gradients) * (1/steps) * (x - baseline)
    avg_gradients = total_gradients / (steps + 1)
    integrated_grads = delta * avg_gradients

    # Collapse to a single spatial map by summing across channels
    # (sum, not max, because IG gives signed attributions per channel)
    attribution = integrated_grads.sum(dim=1).squeeze().cpu().numpy()

    return attribution


# ====== SmoothGrad ======

def compute_smoothgrad(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    num_samples: int = 50,
    noise_level: float = 0.15,
) -> np.ndarray:
    """Compute SmoothGrad — noise-averaged saliency.

    Vanilla saliency maps are notoriously noisy because gradients
    fluctuate sharply in high-dimensional input spaces. SmoothGrad
    reduces this noise by averaging gradients over many copies of the
    input, each perturbed with Gaussian noise.

    SmoothGrad(x) = (1/N) * sum_i grad(f(x + epsilon_i))
    where epsilon_i ~ N(0, sigma^2)

    Reference: Smilkov et al., "SmoothGrad: removing noise by adding
    noise" (2017).

    Args:
        model: A pretrained classifier in eval mode.
        input_tensor: Preprocessed image tensor of shape (1, C, H, W).
        target_class: ImageNet class index to attribute to.
        num_samples: How many noisy copies to average over.
                     More samples = smoother result but slower.
        noise_level: Standard deviation of Gaussian noise as a fraction
                     of the input range. 0.10-0.20 is typical.

    Returns:
        Smoothed saliency map as a 2D numpy array of shape (H, W).
    """
    # Estimate the input's dynamic range to scale noise appropriately
    stdev = noise_level * (input_tensor.max() - input_tensor.min()).item()

    total_gradients = torch.zeros_like(input_tensor)

    for i in range(num_samples):
        # Add Gaussian noise to the input
        noise = torch.randn_like(input_tensor) * stdev
        noisy_input = (input_tensor + noise).clone().detach().requires_grad_(True)

        output = model(noisy_input)
        score = output[0, target_class]

        model.zero_grad()
        score.backward()

        total_gradients += noisy_input.grad.data

    # Average over all noisy samples
    avg_gradients = total_gradients / num_samples

    # Take absolute value and max across channels, like vanilla saliency
    saliency = avg_gradients.abs().max(dim=1)[0].squeeze().cpu().numpy()

    return saliency


# ====== Visualization ======

def normalize_attribution(attr: np.ndarray) -> np.ndarray:
    """Scale attribution values to [0, 1] for visualization.

    Handles edge cases where the attribution is constant (e.g., all
    zeros) by returning a zero map instead of dividing by zero.
    """
    vmin, vmax = attr.min(), attr.max()
    if vmax - vmin < 1e-10:
        return np.zeros_like(attr)
    return (attr - vmin) / (vmax - vmin)


def visualize_attributions(
    original_image: np.ndarray,
    vanilla: np.ndarray,
    integrated: np.ndarray,
    smoothgrad: np.ndarray,
    predicted_class: str,
    save_path: str = "gradient_attribution_comparison.png",
) -> None:
    """Create a side-by-side comparison of all three attribution methods.

    The layout is a 1x4 grid: original image, vanilla saliency,
    integrated gradients, and SmoothGrad. Each attribution map is
    overlaid with a 'hot' colormap so bright regions = high attribution.

    Args:
        original_image: Denormalized image as HWC numpy array in [0, 1].
        vanilla: Vanilla saliency map (H, W).
        integrated: Integrated Gradients map (H, W).
        smoothgrad: SmoothGrad map (H, W).
        predicted_class: Human-readable class label for the title.
        save_path: Where to save the comparison figure.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Panel 1: Original image
    axes[0].imshow(original_image)
    axes[0].set_title(f"Original\nPredicted: {predicted_class}", fontsize=11)
    axes[0].axis("off")

    # Panel 2: Vanilla Saliency
    axes[1].imshow(original_image, alpha=0.4)
    axes[1].imshow(normalize_attribution(vanilla), cmap="hot", alpha=0.6)
    axes[1].set_title("Vanilla Saliency\n(input gradient magnitude)", fontsize=11)
    axes[1].axis("off")

    # Panel 3: Integrated Gradients
    # IG can be negative, so we visualize the absolute value
    axes[2].imshow(original_image, alpha=0.4)
    axes[2].imshow(normalize_attribution(np.abs(integrated)), cmap="hot", alpha=0.6)
    axes[2].set_title("Integrated Gradients\n(path integral from baseline)", fontsize=11)
    axes[2].axis("off")

    # Panel 4: SmoothGrad
    axes[3].imshow(original_image, alpha=0.4)
    axes[3].imshow(normalize_attribution(smoothgrad), cmap="hot", alpha=0.6)
    axes[3].set_title("SmoothGrad\n(noise-averaged saliency)", fontsize=11)
    axes[3].axis("off")

    plt.suptitle("Gradient-Based Attribution Methods Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Figure saved to: {save_path}")
    plt.close()


# ====== ImageNet Label Loader ======

def load_imagenet_labels() -> list[str]:
    """Load the 1000 ImageNet class labels.

    Falls back to numeric indices if the label file cannot be fetched.
    """
    url = (
        "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/"
        "master/imagenet-simple-labels.json"
    )
    try:
        import json
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            labels = json.loads(resp.read().decode("utf-8"))
        return labels
    except Exception:
        # Return generic numeric labels
        return [f"class_{i}" for i in range(1000)]


# ====== Quantitative Comparison ======

def compute_sparsity(attr: np.ndarray, threshold_frac: float = 0.01) -> float:
    """Fraction of pixels whose attribution falls below a threshold.

    Higher sparsity means the method concentrates importance in fewer
    pixels. Threshold is relative to the attribution's own maximum.
    """
    norm = normalize_attribution(attr)
    return float((norm < threshold_frac).sum() / norm.size)


def compute_top_k_mass(attr: np.ndarray, k_frac: float = 0.10) -> float:
    """Fraction of total attribution held by the top-k% of pixels.

    Higher values indicate a more concentrated (less diffuse) attribution.
    """
    norm = normalize_attribution(attr)
    total = norm.sum()
    if total < 1e-10:
        return 0.0
    sorted_vals = np.sort(norm.flatten())[::-1]
    top_k = max(1, int(k_frac * len(sorted_vals)))
    return float(sorted_vals[:top_k].sum() / total)


def print_stats_table(methods: dict[str, np.ndarray]) -> None:
    """Print a comparison table of attribution statistics.

    Columns: mean, std, sparsity, top-10% mass. These metrics
    give complementary views on how "focused" each method's
    attribution is.
    """
    print(f"\n  {'Method':<22} {'Mean':>8} {'Std':>8} {'Sparsity':>10} {'Top-10%':>10}")
    print("  " + "-" * 60)
    for name, attr in methods.items():
        norm = normalize_attribution(np.abs(attr))
        row = (
            f"  {name:<22} "
            f"{norm.mean():>8.4f} "
            f"{norm.std():>8.4f} "
            f"{compute_sparsity(attr):>9.2%} "
            f"{compute_top_k_mass(attr):>9.2%}"
        )
        print(row)


# ====== Main Pipeline ======

def main() -> None:
    """Run all three gradient attribution methods and compare them."""
    print("=" * 60)
    print("  Gradient-Based Attribution Methods")
    print("  Vanilla Saliency | Integrated Gradients | SmoothGrad")
    print("=" * 60)

    # --- Step 1: Load model ---
    print("\n[1] Loading pretrained ResNet18...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model = model.to(device).eval()
    print("  ResNet18 loaded (pretrained on ImageNet).")

    # --- Step 2: Load and preprocess image ---
    print("\n[2] Loading sample image...")
    pil_image = load_sample_image()
    transform = get_imagenet_transform()
    input_tensor = transform(pil_image).unsqueeze(0).to(device)
    print(f"  Input tensor shape: {input_tensor.shape}")

    # Keep a displayable copy for visualization
    display_image = denormalize(input_tensor.squeeze(0))

    # --- Step 3: Predict class ---
    print("\n[3] Running forward pass...")
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)
        confidence, predicted_idx = probs.max(dim=1)

    target_class = predicted_idx.item()
    labels = load_imagenet_labels()
    class_name = labels[target_class]
    print(f"  Predicted class: {target_class} ({class_name})")
    print(f"  Confidence: {confidence.item():.4f}")

    # --- Step 4: Vanilla Saliency ---
    print("\n[4] Computing Vanilla Saliency Map...")
    t0 = time.time()
    vanilla = compute_vanilla_saliency(model, input_tensor, target_class)
    t_vanilla = time.time() - t0
    print(f"  Saliency shape: {vanilla.shape}")
    print(f"  Value range: [{vanilla.min():.6f}, {vanilla.max():.6f}]")
    print(f"  Time: {t_vanilla:.3f}s")

    # --- Step 5: Integrated Gradients ---
    print("\n[5] Computing Integrated Gradients (50 steps)...")
    t0 = time.time()
    integrated = compute_integrated_gradients(
        model, input_tensor, target_class, steps=50,
    )
    t_ig = time.time() - t0
    print(f"  Attribution shape: {integrated.shape}")
    print(f"  Value range: [{integrated.min():.6f}, {integrated.max():.6f}]")
    print(f"  Time: {t_ig:.3f}s")

    # --- Step 6: SmoothGrad ---
    print("\n[6] Computing SmoothGrad (50 samples, noise=0.15)...")
    t0 = time.time()
    smoothgrad = compute_smoothgrad(
        model, input_tensor, target_class, num_samples=50, noise_level=0.15,
    )
    t_sg = time.time() - t0
    print(f"  Saliency shape: {smoothgrad.shape}")
    print(f"  Value range: [{smoothgrad.min():.6f}, {smoothgrad.max():.6f}]")
    print(f"  Time: {t_sg:.3f}s")

    # --- Step 7: Visualize ---
    print("\n[7] Generating comparison visualization...")
    visualize_attributions(
        display_image, vanilla, integrated, smoothgrad,
        predicted_class=class_name,
    )

    # --- Step 8: Quantitative comparison ---
    print("\n[8] Attribution statistics:")
    methods = {
        "Vanilla Saliency": vanilla,
        "Integrated Gradients": integrated,
        "SmoothGrad": smoothgrad,
    }
    print_stats_table(methods)

    # --- Step 9: Rank correlation analysis ---
    print("\n[9] Spearman rank correlation between methods:")
    try:
        from scipy import stats as scipy_stats

        def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
            """Spearman correlation between flattened attribution maps."""
            corr, _ = scipy_stats.spearmanr(a.flatten(), b.flatten())
            return corr

        r_vi = rank_corr(vanilla, np.abs(integrated))
        r_vs = rank_corr(vanilla, smoothgrad)
        r_is = rank_corr(np.abs(integrated), smoothgrad)
        print(f"  Vanilla vs IntegratedGrad: {r_vi:.4f}")
        print(f"  Vanilla vs SmoothGrad:     {r_vs:.4f}")
        print(f"  IntegratedGrad vs Smooth:  {r_is:.4f}")
    except ImportError:
        print("  (scipy not available -- skipping correlation analysis)")

    # --- Step 10: Timing summary ---
    print(f"\n[10] Timing summary:")
    print(f"  Vanilla Saliency:     {t_vanilla:.3f}s  (1 forward + 1 backward)")
    print(f"  Integrated Gradients: {t_ig:.3f}s  (51 forward + 51 backward)")
    print(f"  SmoothGrad:           {t_sg:.3f}s  (50 forward + 50 backward)")

    print("\n" + "=" * 60)
    print("  Observations:")
    print("  - Vanilla saliency is fast but noisy (single gradient).")
    print("  - Integrated Gradients satisfies axioms (sensitivity +")
    print("    implementation invariance) but requires many forward passes.")
    print("  - SmoothGrad reduces noise by averaging, trading compute")
    print("    for visual clarity.")
    print("=" * 60)


if __name__ == "__main__":
    main()
