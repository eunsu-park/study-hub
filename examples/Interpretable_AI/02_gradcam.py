"""
02. Grad-CAM and Grad-CAM++ for CNN Visualization

Implements class-discriminative localization maps that highlight which spatial
regions of an image a CNN attends to when making a prediction. Both the original
Grad-CAM (Selvaraju et al., 2017) and the improved Grad-CAM++ (Chattopadhyay
et al., 2018) are built from scratch using PyTorch forward/backward hooks.

Covered topics:
    - Grad-CAM: global-average-pooled gradients as channel weights
    - Grad-CAM++: higher-order gradient weighting for better localization
    - Hook-based feature/gradient extraction (works with any CNN layer)
    - Heatmap overlay visualization on original images
    - Shallow vs. deep layer comparison (receptive field effects)

Related to: L03 - Class Activation Maps

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

    A real-world image lets us see whether the CAM highlights the actual
    object versus spurious background. The fallback synthetic image still
    exercises the full pipeline.
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
        print(f"  Download failed ({e}). Using synthetic image.")
        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        arr[:, :, 0] = np.linspace(0, 255, 224, dtype=np.uint8)[None, :]
        arr[:, :, 1] = np.linspace(0, 255, 224, dtype=np.uint8)[:, None]
        arr[:, :, 2] = 128
        return Image.fromarray(arr)


def get_imagenet_transform() -> transforms.Compose:
    """Standard ImageNet preprocessing pipeline."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalization for display."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def load_imagenet_labels() -> list[str]:
    """Load ImageNet class labels, falling back to numeric indices."""
    url = (
        "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/"
        "master/imagenet-simple-labels.json"
    )
    try:
        import json
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return [f"class_{i}" for i in range(1000)]


# ====== Grad-CAM Implementation ======

class GradCAM:
    """Grad-CAM: Visual Explanations from Deep Networks (Selvaraju et al., 2017).

    Core idea:
    1. Forward-pass the image through the CNN.
    2. Capture the feature maps A^k at a target convolutional layer.
    3. Backpropagate the target class score to that layer.
    4. Weight each feature map by the global-average-pooled gradient:
         w_k = (1/Z) * sum_{i,j} dY^c / dA^k_{i,j}
    5. Take a ReLU of the weighted combination:
         L_GradCAM = ReLU( sum_k w_k * A^k )

    The result is a coarse heatmap (same spatial size as the feature maps)
    that can be upsampled to the input resolution for overlay.

    Usage:
        cam = GradCAM(model, target_layer_name)
        heatmap = cam(input_tensor, target_class)
    """

    def __init__(self, model: torch.nn.Module, target_layer: str):
        """Register hooks on the specified layer.

        Args:
            model: A CNN in eval mode (e.g., ResNet50).
            target_layer: Dot-separated path to a Conv2d or Sequential block
                          (e.g., 'layer4' or 'layer4.2.conv2').
        """
        self.model = model
        self.target_layer = target_layer

        # Storage for captured activations and gradients
        self._features = None
        self._gradients = None

        # Navigate the model hierarchy to find the target module
        module = self._find_module(model, target_layer)

        # Forward hook: captures the output of the target layer
        module.register_forward_hook(self._save_features)

        # Backward hook: captures the gradient flowing into the target layer
        module.register_full_backward_hook(self._save_gradients)

    @staticmethod
    def _find_module(model: torch.nn.Module, layer_name: str) -> torch.nn.Module:
        """Resolve a dot-separated layer name to the actual module.

        For example, 'layer4.2.conv2' traverses model.layer4[2].conv2.
        This approach is model-agnostic -- it works with any PyTorch CNN.
        """
        module = model
        for part in layer_name.split("."):
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module

    def _save_features(self, module, input, output):
        """Forward hook callback: store feature maps for later use."""
        self._features = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        """Backward hook callback: store gradients for weighting."""
        self._gradients = grad_output[0].detach()

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: int = None,
    ) -> np.ndarray:
        """Generate the Grad-CAM heatmap.

        Args:
            input_tensor: Preprocessed image of shape (1, C, H, W).
            target_class: Class index to explain. If None, uses argmax.

        Returns:
            Heatmap as a 2D numpy array of shape (H_input, W_input),
            values in [0, 1].
        """
        # Forward pass — this triggers the forward hook
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero all gradients, then backprop the target class score
        self.model.zero_grad()
        score = output[0, target_class]
        score.backward(retain_graph=True)

        # Compute channel weights via global average pooling of gradients
        # gradients shape: (1, C, H_feat, W_feat)
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of feature maps
        cam = (weights * self._features).sum(dim=1, keepdim=True)  # (1, 1, H_feat, W_feat)

        # ReLU: we only care about features that positively influence the class
        cam = F.relu(cam)

        # Upsample to input resolution using bilinear interpolation
        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-10:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam


# ====== Grad-CAM++ Implementation ======

class GradCAMPlusPlus:
    """Grad-CAM++: Improved Visual Explanations (Chattopadhyay et al., 2018).

    Grad-CAM uses uniform spatial averaging to compute channel weights.
    This fails when multiple instances of the target class appear, because
    different spatial locations contribute unequally. Grad-CAM++ fixes this
    by using pixel-wise weighting of the gradients:

        alpha^k_{ij} = grad2^k_{ij} / (2*grad2^k_{ij} + sum_{a,b} A^k_{ab} * grad3^k_{ij})

    where grad2 and grad3 are the second and third partial derivatives of
    the class score w.r.t. the feature map activations. In practice these
    higher-order derivatives are computed via the chain rule from the
    first-order gradients by exploiting the ReLU activation structure.
    """

    def __init__(self, model: torch.nn.Module, target_layer: str):
        self.model = model
        self._features = None
        self._gradients = None

        module = GradCAM._find_module(model, target_layer)
        module.register_forward_hook(self._save_features)
        module.register_full_backward_hook(self._save_gradients)

    def _save_features(self, module, input, output):
        self._features = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: int = None,
    ) -> np.ndarray:
        """Generate Grad-CAM++ heatmap with higher-order gradient weighting."""
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, target_class]
        score.backward(retain_graph=True)

        grads = self._gradients   # (1, C, H, W)
        feats = self._features    # (1, C, H, W)

        # Compute the second and third powers of gradients
        # These approximate the higher-order derivatives for ReLU networks
        grad2 = grads.pow(2)
        grad3 = grads.pow(3)

        # Spatial sum of (feature_maps * grad3) for normalization
        spatial_sum = (feats * grad3).sum(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Alpha: pixel-wise importance weighting
        # The denominator prevents division by zero where grad2 is small
        alpha = grad2 / (2.0 * grad2 + spatial_sum + 1e-10)

        # Only keep positive-gradient pixels (same ReLU reasoning as Grad-CAM)
        alpha = alpha * F.relu(grads)

        # Channel weights are the spatial sum of alpha-weighted gradients
        weights = alpha.sum(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of feature maps
        cam = (weights * feats).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Upsample and normalize
        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-10:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam


# ====== Visualization Functions ======

def create_heatmap_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> np.ndarray:
    """Blend a heatmap with an original image for spatial interpretation.

    Why overlay rather than standalone heatmap? A standalone heatmap loses
    all spatial context -- you cannot tell whether the highlighted region
    corresponds to the object's head, body, or a background artifact.

    Args:
        image: Original RGB image in [0, 1], shape (H, W, 3).
        heatmap: Normalized CAM in [0, 1], shape (H, W).
        alpha: Blending weight for the heatmap (0=image only, 1=heatmap only).
        colormap: Matplotlib colormap name.

    Returns:
        Blended RGB image in [0, 1], shape (H, W, 3).
    """
    cmap = plt.cm.get_cmap(colormap)
    colored_heatmap = cmap(heatmap)[:, :, :3]  # Drop alpha channel
    overlay = (1 - alpha) * image + alpha * colored_heatmap
    return np.clip(overlay, 0, 1)


def visualize_gradcam_comparison(
    image: np.ndarray,
    cam_map: np.ndarray,
    campp_map: np.ndarray,
    class_name: str,
    save_path: str = "gradcam_comparison.png",
) -> None:
    """Side-by-side comparison: original, Grad-CAM, Grad-CAM++.

    This lets us see whether Grad-CAM++ provides tighter localization
    than vanilla Grad-CAM, especially when multiple object instances
    are present.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(image)
    axes[0].set_title(f"Original\nPredicted: {class_name}", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(cam_map, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap", fontsize=11)
    axes[1].axis("off")

    overlay_cam = create_heatmap_overlay(image, cam_map)
    axes[2].imshow(overlay_cam)
    axes[2].set_title("Grad-CAM Overlay", fontsize=11)
    axes[2].axis("off")

    overlay_pp = create_heatmap_overlay(image, campp_map)
    axes[3].imshow(overlay_pp)
    axes[3].set_title("Grad-CAM++ Overlay", fontsize=11)
    axes[3].axis("off")

    plt.suptitle("Grad-CAM vs Grad-CAM++ Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Figure saved to: {save_path}")
    plt.close()


def visualize_layer_comparison(
    image: np.ndarray,
    heatmaps: dict[str, np.ndarray],
    class_name: str,
    save_path: str = "gradcam_layer_comparison.png",
) -> None:
    """Compare Grad-CAM heatmaps across different network depths.

    Shallow layers have small receptive fields and produce detailed but
    semantically weak maps. Deep layers have large receptive fields and
    produce coarse but semantically meaningful maps. This visualization
    makes that trade-off concrete.
    """
    n_maps = len(heatmaps)
    fig, axes = plt.subplots(1, n_maps + 1, figsize=(5 * (n_maps + 1), 5))

    axes[0].imshow(image)
    axes[0].set_title(f"Original\n({class_name})", fontsize=11)
    axes[0].axis("off")

    for i, (layer_name, heatmap) in enumerate(heatmaps.items()):
        overlay = create_heatmap_overlay(image, heatmap)
        axes[i + 1].imshow(overlay)
        axes[i + 1].set_title(f"Layer: {layer_name}", fontsize=11)
        axes[i + 1].axis("off")

    plt.suptitle("Grad-CAM: Shallow vs Deep Layer Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Figure saved to: {save_path}")
    plt.close()


# ====== Quantitative Metrics ======

def compute_cam_metrics(cam: np.ndarray, name: str) -> dict:
    """Compute localization quality metrics for a CAM heatmap.

    Metrics:
    - Energy concentration: what fraction of total "heat" is in the
      top 20% of pixels? Higher = more focused.
    - Effective area: fraction of pixels above 50% of max activation.
      Lower = tighter localization.
    """
    total_energy = cam.sum()
    sorted_vals = np.sort(cam.flatten())[::-1]

    # Top-20% energy concentration
    top20_count = max(1, int(0.2 * len(sorted_vals)))
    top20_energy = sorted_vals[:top20_count].sum() / (total_energy + 1e-10)

    # Effective area: fraction of pixels above half-max
    threshold = 0.5 * cam.max()
    effective_area = (cam > threshold).sum() / cam.size

    return {
        "name": name,
        "mean": float(cam.mean()),
        "top20_energy": float(top20_energy),
        "effective_area": float(effective_area),
    }


# ====== Main Pipeline ======

def main() -> None:
    """Run Grad-CAM and Grad-CAM++ on ResNet50, comparing layers."""
    print("=" * 60)
    print("  Grad-CAM and Grad-CAM++ Visualization")
    print("  Class-Discriminative Localization Maps")
    print("=" * 60)

    # --- Step 1: Load model ---
    print("\n[1] Loading pretrained ResNet50...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Why ResNet50 instead of ResNet18? It has more layers to compare
    # shallow vs. deep behavior, and its residual structure is the
    # standard backbone in most CAM research.
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model = model.to(device).eval()
    print("  ResNet50 loaded (pretrained on ImageNet).")

    # --- Step 2: Load and preprocess image ---
    print("\n[2] Loading sample image...")
    pil_image = load_sample_image()
    transform = get_imagenet_transform()
    input_tensor = transform(pil_image).unsqueeze(0).to(device)
    display_image = denormalize(input_tensor.squeeze(0))
    print(f"  Input tensor shape: {input_tensor.shape}")

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

    # --- Step 4: Grad-CAM on final layer ---
    # 'layer4' is the last residual block in ResNet50 — it has the
    # largest receptive field and produces the most semantically
    # meaningful (though spatially coarse) activation maps.
    print("\n[4] Computing Grad-CAM (layer4)...")
    t0 = time.time()
    gradcam = GradCAM(model, target_layer="layer4")
    cam_map = gradcam(input_tensor, target_class)
    t_cam = time.time() - t0
    print(f"  Heatmap shape: {cam_map.shape}")
    print(f"  Value range: [{cam_map.min():.4f}, {cam_map.max():.4f}]")
    print(f"  Time: {t_cam:.3f}s")

    # --- Step 5: Grad-CAM++ on final layer ---
    print("\n[5] Computing Grad-CAM++ (layer4)...")
    t0 = time.time()
    # We need a fresh model to avoid hook conflicts from the GradCAM instance
    model_pp = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model_pp = model_pp.to(device).eval()
    gradcampp = GradCAMPlusPlus(model_pp, target_layer="layer4")
    campp_map = gradcampp(input_tensor, target_class)
    t_campp = time.time() - t0
    print(f"  Heatmap shape: {campp_map.shape}")
    print(f"  Value range: [{campp_map.min():.4f}, {campp_map.max():.4f}]")
    print(f"  Time: {t_campp:.3f}s")

    # --- Step 6: Visualize Grad-CAM vs Grad-CAM++ ---
    print("\n[6] Generating Grad-CAM vs Grad-CAM++ comparison...")
    visualize_gradcam_comparison(display_image, cam_map, campp_map, class_name)

    # --- Step 7: Layer comparison (shallow vs deep) ---
    print("\n[7] Comparing Grad-CAM across layers...")
    # ResNet50 layers from shallow to deep:
    #   layer1 -> 64 channels, 56x56 spatial (fine-grained edges)
    #   layer2 -> 128 channels, 28x28 spatial (textures, parts)
    #   layer3 -> 256 channels, 14x14 spatial (object parts)
    #   layer4 -> 512 channels, 7x7 spatial (whole-object semantics)
    layer_names = ["layer1", "layer2", "layer3", "layer4"]
    layer_heatmaps = {}

    for layer_name in layer_names:
        print(f"  Processing {layer_name}...")
        # Fresh model for each layer to avoid hook accumulation
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        m = m.to(device).eval()
        cam_obj = GradCAM(m, target_layer=layer_name)
        hmap = cam_obj(input_tensor, target_class)
        layer_heatmaps[layer_name] = hmap
        print(f"    Mean activation: {hmap.mean():.4f}, "
              f"Max activation: {hmap.max():.4f}")

    visualize_layer_comparison(display_image, layer_heatmaps, class_name)

    # --- Step 8: Quantitative analysis ---
    print("\n[8] Localization quality metrics:")
    print(f"  {'Method':<20} {'Mean':>8} {'Top-20% E':>12} {'Eff. Area':>12}")
    print("  " + "-" * 55)

    for name, hmap in [("Grad-CAM", cam_map), ("Grad-CAM++", campp_map)]:
        metrics = compute_cam_metrics(hmap, name)
        print(f"  {metrics['name']:<20} "
              f"{metrics['mean']:>8.4f} "
              f"{metrics['top20_energy']:>11.2%} "
              f"{metrics['effective_area']:>11.2%}")

    # --- Step 9: Summary ---
    print("\n" + "=" * 60)
    print("  Observations:")
    print("  - Grad-CAM produces a coarse class-discriminative heatmap.")
    print("  - Grad-CAM++ better handles multiple instances and gives")
    print("    more complete coverage of the target object.")
    print("  - Shallow layers (layer1/2) capture edges/textures but lack")
    print("    semantic meaning. Deep layers (layer3/4) capture object-level")
    print("    concepts but lose fine spatial detail.")
    print("  - The trade-off: spatial precision vs semantic relevance.")
    print("=" * 60)


if __name__ == "__main__":
    main()
