# Lesson 3: Class Activation Mapping

[Previous: Gradient Attribution](./02_Gradient_Attribution.md) | [Next: Attention Interpretation](./04_Attention_Interpretation.md)

---

## Learning Objectives

- Explain how the original CAM method exploits the Global Average Pooling layer to produce class-discriminative localization maps
- Implement GradCAM from scratch using PyTorch hooks, understanding why it generalizes CAM to any CNN architecture
- Compare GradCAM++, Score-CAM, and Eigen-CAM in terms of their theoretical motivations and practical trade-offs
- Apply class activation mapping to a medical imaging use case and interpret the results critically
- Identify the fundamental limitations of CAM-family methods including resolution, class-discriminativeness, and architectural assumptions

---

Class Activation Mapping (CAM) methods answer a spatial question that gradient
attribution alone cannot: *"Where in the image did the model look to make its
decision?"* While gradient saliency maps (Lesson 02) highlight individual pixels,
CAM methods produce coarse localization maps that highlight image *regions*,
making them far more intuitive for human interpretation.

This lesson traces the evolution from the original CAM (which requires a specific
architecture) through GradCAM (which works with any CNN) to modern variants like
Score-CAM and Eigen-CAM. We implement each method, apply them to real-world use
cases, and critically examine their limitations.

---

## 1. Original CAM (Zhou et al. 2016)

### 1.1 The Key Insight

```python
"""
CLASS ACTIVATION MAPPING (Zhou et al. 2016)

Observation: Many modern CNNs end with:
  Convolutional layers → Global Average Pooling (GAP) → Fully Connected → Softmax

The GAP layer computes the spatial average of each feature map:
  g_k = (1/HW) Σ_{i,j} A_k(i, j)

where A_k(i, j) is the activation of feature map k at spatial position (i, j).

Then the class score for class c is:
  S_c = Σ_k w_k^c * g_k

where w_k^c is the weight connecting feature map k to class c.

Substituting the GAP formula:
  S_c = Σ_k w_k^c * (1/HW) Σ_{i,j} A_k(i, j)
      = (1/HW) Σ_{i,j} Σ_k w_k^c * A_k(i, j)

Define the Class Activation Map:
  CAM_c(i, j) = Σ_k w_k^c * A_k(i, j)

This gives us a spatial map where each position (i, j) indicates how
much that region contributes to class c.

The CAM is simply a weighted sum of the last convolutional layer's
feature maps, where the weights are the classification layer's weights.

CRITICAL LIMITATION: This only works for architectures that have a
GAP layer immediately before the final classification layer. If the
network has fully connected layers between convolutions and output
(like the original VGG), CAM cannot be applied directly.
"""
```

### 1.2 Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Optional, Tuple


class OriginalCAM:
    """
    Implementation of the original Class Activation Mapping (Zhou et al. 2016).

    This method requires that the model has:
    1. A convolutional backbone that produces feature maps
    2. A Global Average Pooling layer
    3. A single fully connected (linear) classification layer

    Models that satisfy this: ResNet, DenseNet, EfficientNet, MobileNet
    Models that do NOT satisfy this: VGG (has multiple FC layers), AlexNet
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module, fc_layer: nn.Linear):
        """
        Parameters
        ----------
        model : nn.Module
            The full model.
        target_layer : nn.Module
            The last convolutional layer before GAP.
        fc_layer : nn.Linear
            The final classification linear layer after GAP.
        """
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.fc_layer = fc_layer

        # Storage for the feature maps captured by the hook
        self.feature_maps = None

        # Register a forward hook to capture the feature maps
        # The hook fires every time data passes through target_layer
        self.hook = target_layer.register_forward_hook(self._save_feature_maps)

    def _save_feature_maps(self, module, input, output):
        """Forward hook callback: store the feature maps."""
        # output shape: (batch, num_channels, H, W)
        self.feature_maps = output.detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Generate the Class Activation Map for a given input and class.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Preprocessed input image, shape (1, 3, H_in, W_in).
        target_class : int, optional
            Class index. If None, uses the predicted class.

        Returns
        -------
        Tuple[np.ndarray, int]
            - cam: Class activation map, shape (H_feat, W_feat),
              normalized to [0, 1].
            - target_class: The class used for the CAM.
        """
        # Forward pass: this triggers the hook, storing feature maps
        with torch.no_grad():
            output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Get the weights from the FC layer for the target class
        # fc_layer.weight has shape (num_classes, num_features)
        # We want the row corresponding to target_class
        weights = self.fc_layer.weight[target_class]  # shape: (num_features,)

        # Feature maps have shape (1, num_features, H, W)
        feature_maps = self.feature_maps[0]  # shape: (num_features, H, W)

        # CAM = weighted sum of feature maps
        # weights: (num_features,) → (num_features, 1, 1) for broadcasting
        cam = (weights.unsqueeze(-1).unsqueeze(-1) * feature_maps).sum(dim=0)

        # Apply ReLU: we only care about features that have a positive
        # influence on the target class score
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.cpu().numpy(), target_class

    def cleanup(self):
        """Remove the forward hook to prevent memory leaks."""
        self.hook.remove()


def original_cam_demo():
    """
    Demonstrate original CAM on ResNet-50.

    ResNet-50 architecture (relevant parts):
      ... → layer4 → AdaptiveAvgPool2d → fc (Linear)

    This satisfies the GAP + single FC requirement.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.eval()

    # The last convolutional layer in ResNet-50
    target_layer = model.layer4[-1]  # Last bottleneck block
    fc_layer = model.fc  # The final classification layer

    cam_extractor = OriginalCAM(model, target_layer, fc_layer)

    # Create a dummy input (replace with real image)
    dummy_input = torch.randn(1, 3, 224, 224)

    cam_map, pred_class = cam_extractor.generate(dummy_input)
    print(f"Predicted class: {pred_class}")
    print(f"CAM shape: {cam_map.shape}")  # Will be (7, 7) for ResNet-50

    cam_extractor.cleanup()
    return cam_map
```

---

## 2. GradCAM (Selvaraju et al. 2017)

### 2.1 Generalizing CAM to Any Architecture

```python
"""
GRADIENT-WEIGHTED CLASS ACTIVATION MAPPING (GradCAM)

The original CAM only works with GAP + single FC architectures.
GradCAM removes this restriction by using GRADIENTS to compute
the importance weights instead of the FC layer's weights.

Key insight: The gradient of the class score with respect to a
feature map tells us how important that feature map is for the class.

For target class c and feature map k of the last conv layer:

  Importance weight:
    α_k^c = (1/HW) Σ_{i,j} (∂S_c / ∂A_k(i, j))

  This is the Global Average Pooling of the gradient — it measures
  the average importance of feature map k for class c.

  GradCAM:
    L_c(i, j) = ReLU( Σ_k α_k^c * A_k(i, j) )

  ReLU is applied because we only care about features with POSITIVE
  influence on the target class. Negative influences are features
  that belong to OTHER classes.

WHY THIS WORKS FOR ANY ARCHITECTURE:
  - We do NOT need a GAP layer in the model
  - We do NOT need a single FC layer
  - We only need the gradient of ANY scalar output w.r.t. ANY
    convolutional layer's feature maps
  - This works for ResNet, VGG, Inception, any CNN

MATHEMATICAL CONNECTION TO ORIGINAL CAM:
  For a GAP + single FC architecture, GradCAM's weights α_k^c are
  EXACTLY equal to the FC weights w_k^c. GradCAM is a strict
  generalization of CAM.
"""
```

### 2.2 Complete GradCAM Implementation

```python
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Selvaraju et al. 2017).

    Works with ANY CNN architecture — does not require GAP or specific
    layer structure. This is the most commonly used CAM variant in
    practice.

    Usage:
        grad_cam = GradCAM(model, target_layer)
        cam_map = grad_cam.generate(input_tensor, target_class)
        grad_cam.cleanup()  # Important: prevent memory leaks
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Parameters
        ----------
        model : nn.Module
            Any CNN model.
        target_layer : nn.Module
            The convolutional layer to compute GradCAM for.
            Typically the last convolutional layer before the classifier.
        """
        self.model = model
        self.model.eval()

        # Storage for hook data
        self.activations = None
        self.gradients = None

        # Register both forward and backward hooks
        # Forward hook: captures the feature maps (activations)
        self.forward_hook = target_layer.register_forward_hook(
            self._save_activations
        )
        # Backward hook: captures the gradients flowing through this layer
        self.backward_hook = target_layer.register_full_backward_hook(
            self._save_gradients
        )

    def _save_activations(self, module, input, output):
        """Forward hook: store activations (feature maps)."""
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        """Backward hook: store gradients of the output."""
        # grad_output is a tuple; first element has shape (batch, C, H, W)
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        upsample_to_input: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Generate GradCAM heatmap.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Preprocessed input, shape (1, 3, H, W).
        target_class : int, optional
            Target class. If None, uses predicted class.
        upsample_to_input : bool
            If True, upsample the CAM to match input spatial dimensions.

        Returns
        -------
        Tuple[np.ndarray, int]
            - cam: Heatmap, shape (H, W), values in [0, 1].
            - target_class: Class used.
        """
        # Forward pass: triggers the forward hook
        # We need gradients, so no torch.no_grad()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Create a one-hot target for backpropagation
        # We want the gradient of the target class score ONLY
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward(retain_graph=False)

        # At this point:
        # self.activations has shape (1, num_channels, H_feat, W_feat)
        # self.gradients has shape (1, num_channels, H_feat, W_feat)

        # Step 1: Compute importance weights by global average pooling the gradients
        # α_k = mean over spatial dimensions of the gradient for channel k
        weights = self.gradients[0].mean(dim=(1, 2))  # shape: (num_channels,)

        # Step 2: Weighted combination of feature maps
        # Expand weights for broadcasting: (num_channels,) → (num_channels, 1, 1)
        activations = self.activations[0]  # shape: (num_channels, H_feat, W_feat)
        cam = (weights.unsqueeze(-1).unsqueeze(-1) * activations).sum(dim=0)

        # Step 3: Apply ReLU
        # We only want the positive contributions to the target class
        # Negative values represent features for OTHER classes
        cam = F.relu(cam)

        # Step 4: Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Step 5: Optionally upsample to input resolution
        if upsample_to_input:
            # input_tensor shape: (1, 3, H_input, W_input)
            h_input, w_input = input_tensor.shape[2], input_tensor.shape[3]
            # Bilinear interpolation from (H_feat, W_feat) to (H_input, W_input)
            cam = F.interpolate(
                cam.unsqueeze(0).unsqueeze(0),  # (1, 1, H_feat, W_feat)
                size=(h_input, w_input),
                mode='bilinear',
                align_corners=False
            )[0, 0]

        return cam.cpu().numpy(), target_class

    def cleanup(self):
        """Remove hooks to prevent memory leaks."""
        self.forward_hook.remove()
        self.backward_hook.remove()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False


# --- Usage ---

def gradcam_demo():
    """
    Complete GradCAM demonstration with ResNet-50.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.eval()

    # Standard ImageNet preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # Create dummy input (replace with real image)
    dummy_input = torch.randn(1, 3, 224, 224)

    # The target layer is the last conv layer of the last block
    # For ResNet-50, that is layer4[-1].conv3
    # Why the LAST conv layer? Because it has the richest semantic
    # information — earlier layers detect edges and textures, while
    # later layers detect object parts and whole objects
    target_layer = model.layer4[-1].conv3

    with GradCAM(model, target_layer) as grad_cam:
        cam_map, pred_class = grad_cam.generate(dummy_input)

    print(f"Predicted class: {pred_class}")
    print(f"CAM shape: {cam_map.shape}")

    return cam_map
```

### 2.3 Visualization Utilities

```python
def visualize_gradcam(
    original_image: np.ndarray,
    cam_map: np.ndarray,
    target_class: int,
    class_name: str = "",
    alpha: float = 0.5
) -> None:
    """
    Visualize GradCAM heatmap overlaid on the original image.

    Parameters
    ----------
    original_image : np.ndarray
        Original image, shape (H, W, 3), values [0, 255] uint8.
    cam_map : np.ndarray
        GradCAM heatmap, shape (H, W), values [0, 1].
    target_class : int
        Target class index.
    class_name : str
        Human-readable class name.
    alpha : float
        Transparency for the overlay. 0 = original image only,
        1 = heatmap only.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 1. Original image
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # 2. Raw GradCAM heatmap
    axes[1].imshow(cam_map, cmap="jet")
    axes[1].set_title("GradCAM Heatmap")
    axes[1].axis("off")

    # 3. Overlay: heatmap on top of original
    axes[2].imshow(original_image)
    axes[2].imshow(cam_map, cmap="jet", alpha=alpha)
    axes[2].set_title("GradCAM Overlay")
    axes[2].axis("off")

    # 4. Masked original: show only the high-activation regions
    # This helps verify that the highlighted region actually contains
    # the object of interest
    mask = cam_map > 0.3  # Threshold: keep regions with CAM > 0.3
    masked_image = original_image.copy()
    # Darken regions with low activation
    for c in range(3):
        masked_image[:, :, c] = np.where(
            mask,
            original_image[:, :, c],
            (original_image[:, :, c] * 0.3).astype(np.uint8)
        )
    axes[3].imshow(masked_image)
    axes[3].set_title(f"Focus Region (>{0.3:.0%})")
    axes[3].axis("off")

    title = f"GradCAM — Class: {target_class}"
    if class_name:
        title += f" ({class_name})"
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig("gradcam_visualization.png", dpi=150)
    plt.show()


def compare_layers(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    original_image: np.ndarray
) -> None:
    """
    Compare GradCAM from different layers of the network.

    Earlier layers produce higher-resolution but less semantic maps.
    Later layers produce lower-resolution but more class-discriminative maps.

    This visualization helps you choose the right layer for your use case.
    """
    # For ResNet-50, define layers from shallow to deep
    layers = {
        "layer1[-1]": model.layer1[-1].conv3,  # 56x56 feature maps
        "layer2[-1]": model.layer2[-1].conv3,  # 28x28 feature maps
        "layer3[-1]": model.layer3[-1].conv3,  # 14x14 feature maps
        "layer4[-1]": model.layer4[-1].conv3,  # 7x7 feature maps
    }

    fig, axes = plt.subplots(1, len(layers) + 1, figsize=(24, 5))

    axes[0].imshow(original_image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    for idx, (layer_name, layer) in enumerate(layers.items()):
        with GradCAM(model, layer) as grad_cam:
            cam_map, _ = grad_cam.generate(input_tensor.clone(), target_class)

        axes[idx + 1].imshow(original_image)
        axes[idx + 1].imshow(cam_map, cmap="jet", alpha=0.5)
        axes[idx + 1].set_title(f"{layer_name}\n{cam_map.shape}")
        axes[idx + 1].axis("off")

    plt.suptitle("GradCAM at Different Network Depths", fontsize=14)
    plt.tight_layout()
    plt.savefig("gradcam_layer_comparison.png", dpi=150)
    plt.show()

    print("Observation:")
    print("  Shallow layers → high resolution, detect edges/textures")
    print("  Deep layers → low resolution, detect semantic regions")
    print("  Best practice: use the LAST conv layer for class-discriminative maps")
```

---

## 3. GradCAM++ (Chattopadhay et al. 2018)

### 3.1 Motivation: Multi-Object Scenes

```python
"""
GRADCAM++ MOTIVATION

GradCAM uses simple global average pooling of the gradients to compute
importance weights:

    α_k = (1/HW) Σ_{i,j} ∂S_c / ∂A_k(i,j)

This treats all spatial positions equally. In images with MULTIPLE
instances of the same class (e.g., three cats in one image), GradCAM
tends to focus on only ONE instance — usually the largest or most
prominent one.

GradCAM++ addresses this by using a WEIGHTED average of the gradients,
where the weights depend on the second-order gradient (the gradient of
the gradient). This gives different spatial positions different
importance, allowing the method to highlight multiple instances.

GRADCAM++ FORMULA

    α_k^c = Σ_{i,j} a_k^{c,ij} * ReLU(∂S_c / ∂A_k(i,j))

where a_k^{c,ij} are the pixel-wise weights computed from second-order
gradients:

    a_k^{c,ij} = (∂²S_c / ∂A_k(i,j)²) /
                  (2 * ∂²S_c / ∂A_k(i,j)² + Σ_{a,b} A_k(a,b) * ∂³S_c / ∂A_k(i,j)³)

In practice, the third derivative is expensive to compute, so
implementations use an efficient approximation.
"""
```

### 3.2 Implementation

```python
class GradCAMPlusPlus:
    """
    GradCAM++ (Chattopadhay et al. 2018).

    Extends GradCAM with weighted gradient pooling for better
    localization of multiple instances of the same class.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.model.eval()
        self.activations = None
        self.gradients = None

        self.forward_hook = target_layer.register_forward_hook(
            self._save_activations
        )
        self.backward_hook = target_layer.register_full_backward_hook(
            self._save_gradients
        )

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Generate GradCAM++ heatmap.

        The key difference from GradCAM is in how the weights are computed:
        instead of simple averaging, we use weighted averaging based on
        second-order gradient information.
        """
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward(retain_graph=False)

        activations = self.activations[0]  # (C, H, W)
        gradients = self.gradients[0]       # (C, H, W)

        # Compute second and third order approximations
        # grad^2 and grad^3 approximate the higher-order derivatives
        grad_2 = gradients ** 2
        grad_3 = gradients ** 3

        # Pixel-wise weights (the a_k^{c,ij} terms)
        # The denominator prevents division by zero and implements
        # the second-order weighting scheme
        sum_activations = activations.sum(dim=(1, 2), keepdim=True)
        denominator = 2 * grad_2 + sum_activations * grad_3 + 1e-8

        # Compute pixel-wise importance weights
        pixel_weights = grad_2 / denominator  # (C, H, W)

        # Apply ReLU to gradients (only positive contributions)
        positive_gradients = F.relu(gradients)

        # Weighted combination: pixel_weights * positive_gradients
        # Then sum over spatial dimensions to get per-channel weights
        weights = (pixel_weights * positive_gradients).sum(dim=(1, 2))  # (C,)

        # Weighted sum of feature maps (same as GradCAM from here)
        cam = (weights.unsqueeze(-1).unsqueeze(-1) * activations).sum(dim=0)
        cam = F.relu(cam)

        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Upsample to input size
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode='bilinear',
            align_corners=False
        )[0, 0]

        return cam.cpu().numpy(), target_class

    def cleanup(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False
```

---

## 4. Score-CAM (Wang et al. 2020)

### 4.1 Gradient-Free Approach

```python
"""
SCORE-CAM: GRADIENT-FREE CLASS ACTIVATION MAPPING

Motivation: All previous CAM methods rely on gradients, which can be:
  1. Noisy (as we saw in Lesson 02)
  2. Vulnerable to gradient saturation
  3. Computationally tied to backpropagation

Score-CAM takes a fundamentally different approach: instead of using
gradients, it uses the model's output SCORES as importance weights.

Algorithm:
  1. Extract feature maps A_k from the target layer (forward hook)
  2. For each feature map k:
     a) Upsample A_k to input resolution
     b) Normalize to [0, 1]
     c) Use A_k as a mask on the original input: masked_input = input * A_k
     d) Forward pass the masked input through the model
     e) The score for the target class is the importance weight: w_k = S_c(masked_input)
  3. CAM = Σ_k w_k * A_k  (weighted sum of feature maps)

Intuition: If masking the input with feature map k preserves (or
increases) the class score, then feature map k captures information
important for that class.

Advantages:
  ✓ No gradients needed — avoids all gradient pathologies
  ✓ More principled: importance is measured by actual model output
  ✓ Works with non-differentiable models (in principle)

Disadvantages:
  ✗ SLOW: requires N forward passes (one per feature map)
    For ResNet-50 layer4: N = 2048 forward passes!
  ✗ Masking assumption: using feature maps as soft masks is ad-hoc
"""
```

### 4.2 Implementation

```python
class ScoreCAM:
    """
    Score-CAM (Wang et al. 2020): Gradient-free class activation mapping.

    This method is significantly slower than GradCAM because it requires
    one forward pass per feature map. Use it when gradient reliability
    is a concern or for validation purposes.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.model.eval()
        self.activations = None

        # Only need a forward hook — no gradients involved
        self.forward_hook = target_layer.register_forward_hook(
            self._save_activations
        )

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        batch_size: int = 32
    ) -> Tuple[np.ndarray, int]:
        """
        Generate Score-CAM heatmap.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input image, shape (1, 3, H, W).
        target_class : int, optional
            Target class. If None, uses predicted class.
        batch_size : int
            Number of masked images to process at once.
            Higher = faster but more memory.

        Returns
        -------
        Tuple[np.ndarray, int]
            - cam: Heatmap, shape (H, W), values [0, 1].
            - target_class: Class used.
        """
        h_input, w_input = input_tensor.shape[2], input_tensor.shape[3]

        # Step 1: Forward pass to get feature maps and prediction
        with torch.no_grad():
            output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Feature maps: (1, C, H_feat, W_feat)
        feature_maps = self.activations[0]  # (C, H_feat, W_feat)
        num_channels = feature_maps.shape[0]

        # Step 2: Upsample each feature map to input resolution
        # and normalize to [0, 1]
        upsampled = F.interpolate(
            feature_maps.unsqueeze(0),  # (1, C, H_feat, W_feat)
            size=(h_input, w_input),
            mode='bilinear',
            align_corners=False
        )[0]  # (C, H_input, W_input)

        # Normalize each feature map independently to [0, 1]
        # This ensures each mask has a consistent scale
        for k in range(num_channels):
            fmap = upsampled[k]
            fmin, fmax = fmap.min(), fmap.max()
            if fmax - fmin > 1e-8:
                upsampled[k] = (fmap - fmin) / (fmax - fmin)
            else:
                upsampled[k] = torch.zeros_like(fmap)

        # Step 3: For each feature map, create masked input and get score
        scores = torch.zeros(num_channels, device=input_tensor.device)

        # Process in batches for efficiency
        for start in range(0, num_channels, batch_size):
            end = min(start + batch_size, num_channels)
            batch_masks = upsampled[start:end]  # (B, H, W)

            # Create masked inputs: input * mask
            # input_tensor: (1, 3, H, W), batch_masks: (B, H, W)
            # We need to broadcast the mask across the 3 channels
            masked_inputs = input_tensor * batch_masks.unsqueeze(1)  # (B, 3, H, W)

            # Forward pass on all masked inputs
            with torch.no_grad():
                masked_outputs = self.model(masked_inputs)

            # Extract scores for the target class
            scores[start:end] = masked_outputs[:, target_class]

        # Normalize scores using softmax to get proper weights
        # This prevents any single feature map from dominating
        weights = F.softmax(scores, dim=0)

        # Step 4: Weighted sum of feature maps
        cam = (weights.unsqueeze(-1).unsqueeze(-1) * feature_maps).sum(dim=0)
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Upsample to input resolution
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(h_input, w_input),
            mode='bilinear',
            align_corners=False
        )[0, 0]

        return cam.cpu().numpy(), target_class

    def cleanup(self):
        self.forward_hook.remove()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False
```

---

## 5. Eigen-CAM (Muhammad & Yeasin 2020)

### 5.1 Principal Component Approach

```python
"""
EIGEN-CAM: PCA-BASED CLASS ACTIVATION MAPPING

Eigen-CAM takes yet another approach: instead of using gradients or
model scores, it uses the PRINCIPAL COMPONENT of the activation maps.

Algorithm:
  1. Extract feature maps A from the target layer: shape (C, H, W)
  2. Reshape to a 2D matrix: (C, H*W)
  3. Compute SVD: U, S, V^T = SVD(A_reshaped)
  4. The first principal component V[:, 0] captures the dominant
     spatial pattern in the activations
  5. Reshape back to (H, W) → this is the Eigen-CAM

Intuition: The first principal component captures the spatial pattern
that is most common across all feature channels. For a well-trained
classifier, this dominant pattern tends to correspond to the main
object in the image.

Advantages:
  ✓ No gradients needed
  ✓ Very fast (just one SVD, no extra forward passes)
  ✓ Class-agnostic by default (captures the dominant pattern)

Disadvantages:
  ✗ NOT class-discriminative: the same map is produced regardless
    of which class you ask about
  ✗ May highlight background textures if they dominate the activations
  ✗ Less theoretically motivated than GradCAM or Score-CAM
"""


class EigenCAM:
    """
    Eigen-CAM (Muhammad & Yeasin 2020): PCA-based activation mapping.

    The fastest CAM variant — requires only a forward pass and an SVD.
    However, it is NOT class-discriminative.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.model.eval()
        self.activations = None

        self.forward_hook = target_layer.register_forward_hook(
            self._save_activations
        )

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Generate Eigen-CAM heatmap.

        Note: target_class is accepted for API consistency but is NOT
        used — Eigen-CAM is class-agnostic.
        """
        h_input, w_input = input_tensor.shape[2], input_tensor.shape[3]

        with torch.no_grad():
            output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Feature maps: (C, H_feat, W_feat)
        feature_maps = self.activations[0]
        C, H, W = feature_maps.shape

        # Reshape to 2D: (C, H*W)
        reshaped = feature_maps.reshape(C, H * W).cpu().numpy()

        # SVD: find the principal component
        # U: (C, C), S: (min(C, H*W),), Vt: (H*W, H*W)
        # The first column of V (first row of Vt) is the dominant
        # spatial pattern
        U, S, Vt = np.linalg.svd(reshaped, full_matrices=False)

        # First principal component, reshaped to spatial dimensions
        cam = Vt[0].reshape(H, W)

        # Ensure non-negative (the sign of PCs is arbitrary)
        cam = np.abs(cam)

        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Upsample to input resolution
        cam_tensor = torch.tensor(cam, dtype=torch.float32)
        cam_tensor = F.interpolate(
            cam_tensor.unsqueeze(0).unsqueeze(0),
            size=(h_input, w_input),
            mode='bilinear',
            align_corners=False
        )[0, 0]

        return cam_tensor.numpy(), target_class

    def cleanup(self):
        self.forward_hook.remove()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False
```

---

## 6. Comprehensive Method Comparison

```python
def compare_all_cam_methods(
    model: nn.Module,
    target_layer: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    original_image: np.ndarray
) -> None:
    """
    Side-by-side comparison of all CAM variants on the same image.

    This is the standard comparison figure used in CAM papers.
    It reveals the strengths and weaknesses of each method.
    """
    methods = {}

    # 1. GradCAM
    with GradCAM(model, target_layer) as gc:
        methods["GradCAM"], _ = gc.generate(input_tensor.clone(), target_class)

    # 2. GradCAM++
    with GradCAMPlusPlus(model, target_layer) as gcpp:
        methods["GradCAM++"], _ = gcpp.generate(input_tensor.clone(), target_class)

    # 3. Score-CAM (slow — reduce batch_size if GPU memory is limited)
    with ScoreCAM(model, target_layer) as sc:
        methods["Score-CAM"], _ = sc.generate(
            input_tensor.clone(), target_class, batch_size=64
        )

    # 4. Eigen-CAM
    with EigenCAM(model, target_layer) as ec:
        methods["Eigen-CAM"], _ = ec.generate(input_tensor.clone(), target_class)

    # Visualization
    num_methods = len(methods)
    fig, axes = plt.subplots(2, num_methods + 1, figsize=(4 * (num_methods + 1), 8))

    # Top row: heatmaps
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title("Original", fontsize=11)
    axes[0, 0].axis("off")

    for idx, (name, cam_map) in enumerate(methods.items()):
        axes[0, idx + 1].imshow(cam_map, cmap="jet")
        axes[0, idx + 1].set_title(name, fontsize=11)
        axes[0, idx + 1].axis("off")

    # Bottom row: overlays
    axes[1, 0].imshow(original_image)
    axes[1, 0].set_title("Original", fontsize=11)
    axes[1, 0].axis("off")

    for idx, (name, cam_map) in enumerate(methods.items()):
        axes[1, idx + 1].imshow(original_image)
        axes[1, idx + 1].imshow(cam_map, cmap="jet", alpha=0.5)
        axes[1, idx + 1].set_title(f"{name}\n(overlay)", fontsize=11)
        axes[1, idx + 1].axis("off")

    plt.suptitle(f"CAM Methods Comparison — Class {target_class}", fontsize=14)
    plt.tight_layout()
    plt.savefig("cam_methods_comparison.png", dpi=150)
    plt.show()

    # Quantitative comparison: IoU between methods
    print("\nPairwise agreement (Pearson correlation):")
    method_names = list(methods.keys())
    for i in range(len(method_names)):
        for j in range(i + 1, len(method_names)):
            corr = np.corrcoef(
                methods[method_names[i]].flatten(),
                methods[method_names[j]].flatten()
            )[0, 1]
            print(f"  {method_names[i]} vs {method_names[j]}: r = {corr:.3f}")
```

---

## 7. Practical Application: Medical Imaging

### 7.1 Chest X-Ray Interpretation

```python
"""
MEDICAL IMAGING USE CASE: CHEST X-RAY CLASSIFICATION

GradCAM is one of the most popular interpretability tools in medical
imaging because:

1. Clinicians think spatially — "show me WHERE in the image"
2. The output is intuitive — a heatmap overlaid on the scan
3. It can reveal whether the model is looking at the pathology
   or at an artifact (e.g., the hospital logo on the image border)

CRITICAL WARNING FOR MEDICAL AI:
GradCAM showing the "right" region does NOT prove the model is correct.
The model might be:
- Looking at the right region for the WRONG reason
  (e.g., using image artifacts correlated with disease)
- Correct on this example but unreliable in general
- Overfitting to confounders in the training data

GradCAM is a DEBUGGING tool, not a VALIDATION tool.
Always combine with proper statistical evaluation.
"""


def medical_imaging_gradcam(
    model: nn.Module,
    target_layer: nn.Module,
    xray_tensor: torch.Tensor,
    xray_image: np.ndarray,
    class_names: list,
    threshold: float = 0.5
) -> dict:
    """
    Apply GradCAM to a chest X-ray classifier with clinical reporting.

    Parameters
    ----------
    model : nn.Module
        Pretrained chest X-ray classifier.
    target_layer : nn.Module
        Target convolutional layer for GradCAM.
    xray_tensor : torch.Tensor
        Preprocessed X-ray, shape (1, 1, H, W) or (1, 3, H, W).
    xray_image : np.ndarray
        Original X-ray image for visualization.
    class_names : list
        List of class names (e.g., ["Normal", "Pneumonia", "Effusion"]).
    threshold : float
        Confidence threshold for positive classification.

    Returns
    -------
    dict
        Clinical report with predictions, confidence, and CAM maps.
    """
    model.eval()

    # Get predictions
    with torch.no_grad():
        logits = model(xray_tensor)
        # For multi-label classification (common in radiology),
        # use sigmoid instead of softmax
        probabilities = torch.sigmoid(logits)[0].cpu().numpy()

    # Generate GradCAM for each positive class
    report = {
        "predictions": {},
        "cam_maps": {},
        "warnings": []
    }

    for class_idx, class_name in enumerate(class_names):
        prob = probabilities[class_idx]
        report["predictions"][class_name] = float(prob)

        if prob >= threshold:
            # Generate GradCAM for this class
            with GradCAM(model, target_layer) as gc:
                cam_map, _ = gc.generate(xray_tensor.clone(), class_idx)

            report["cam_maps"][class_name] = cam_map

            # Check for potential artifacts
            # If the highest activation is at the image border, the model
            # might be using non-medical features (labels, equipment, etc.)
            border_fraction = _border_activation_fraction(cam_map, border_width=20)
            if border_fraction > 0.3:
                report["warnings"].append(
                    f"WARNING: {class_name} — {border_fraction:.0%} of high "
                    f"activation is at the image border. The model may be "
                    f"using non-medical features (artifacts, labels)."
                )

    return report


def _border_activation_fraction(cam_map: np.ndarray, border_width: int = 20) -> float:
    """
    Compute the fraction of high activation near the image border.

    A high fraction suggests the model may be using artifacts
    (hospital logos, text annotations) rather than medical content.
    """
    threshold = 0.5  # Consider pixels with activation > 0.5 as "high"
    high_activation = cam_map > threshold
    total_high = high_activation.sum()

    if total_high == 0:
        return 0.0

    # Create a border mask
    h, w = cam_map.shape
    border_mask = np.zeros_like(cam_map, dtype=bool)
    border_mask[:border_width, :] = True  # Top
    border_mask[-border_width:, :] = True  # Bottom
    border_mask[:, :border_width] = True  # Left
    border_mask[:, -border_width:] = True  # Right

    border_high = (high_activation & border_mask).sum()

    return border_high / total_high


def visualize_medical_report(
    xray_image: np.ndarray,
    report: dict
) -> None:
    """
    Visualize the clinical GradCAM report.
    """
    positive_classes = [name for name, prob in report["predictions"].items()
                        if prob >= 0.5]

    if not positive_classes:
        print("No positive findings above threshold.")
        return

    num_findings = len(positive_classes)
    fig, axes = plt.subplots(1, num_findings + 1, figsize=(6 * (num_findings + 1), 6))

    if num_findings == 0:
        axes = [axes]

    # Original X-ray
    ax0 = axes[0] if num_findings > 0 else axes
    ax0.imshow(xray_image, cmap="gray")
    ax0.set_title("Original X-Ray")
    ax0.axis("off")

    # GradCAM for each positive class
    for idx, class_name in enumerate(positive_classes):
        prob = report["predictions"][class_name]
        cam_map = report["cam_maps"][class_name]

        axes[idx + 1].imshow(xray_image, cmap="gray")
        axes[idx + 1].imshow(cam_map, cmap="jet", alpha=0.4)
        axes[idx + 1].set_title(f"{class_name}\n(conf: {prob:.2%})")
        axes[idx + 1].axis("off")

    plt.suptitle("Chest X-Ray GradCAM Analysis", fontsize=14)
    plt.tight_layout()
    plt.savefig("medical_gradcam_report.png", dpi=150)
    plt.show()

    # Print warnings
    for warning in report["warnings"]:
        print(f"\n{warning}")
```

---

## 8. Limitations of CAM-Family Methods

```python
"""
FUNDAMENTAL LIMITATIONS OF CAM METHODS

1. RESOLUTION
   CAM methods operate on the feature map grid of the target layer.
   For ResNet-50 layer4, this is 7×7. Even after upsampling to
   224×224, the localization is inherently coarse.

   Impact: Cannot pinpoint which pixel or fine-grained feature
   matters. For tasks requiring pixel-level attribution (e.g.,
   "which edge of the tumor?"), use gradient methods instead.

2. CLASS-DISCRIMINATIVENESS (partial)
   GradCAM and GradCAM++ are class-discriminative: different classes
   produce different heatmaps. But Eigen-CAM is NOT — it shows the
   same dominant pattern regardless of the class.

   Impact: For multi-class problems where you want to understand
   why a specific class was chosen over alternatives, use GradCAM.

3. ARCHITECTURAL ASSUMPTIONS
   All CAM methods assume the model has spatial feature maps
   (i.e., convolutional layers). They cannot be directly applied to:
   - Fully connected networks
   - Tabular data models (use SHAP instead)
   - Graph neural networks (use GNNExplainer)
   - Standard Transformers (use attention analysis — Lesson 04)

4. SINGLE-IMAGE LIMITATION
   CAM methods explain one image at a time. They do not provide
   GLOBAL explanations (overall model behavior). For global
   understanding, aggregate many CAM maps or use TCAV (Lesson 07).

5. FAITHFULNESS QUESTIONS
   GradCAM passes Adebayo et al. sanity checks (good), but it still
   has faithfulness limitations:
   - It only captures the LAST convolutional layer's perspective
   - Information from earlier layers (edges, textures) is not shown
   - The importance weights (global average of gradients) are a
     coarse summary of spatial importance

6. MULTI-OBJECT AMBIGUITY
   Even GradCAM++ sometimes fails to highlight all instances.
   The problem is fundamental: the feature maps may not separate
   individual objects in crowded scenes.

7. NEGATION PROBLEM
   CAM methods show what IS important for a class, not what is
   important AGAINST a class. Understanding why a model rejected
   an alternative class requires generating CAMs for multiple
   classes and comparing them.

RECOMMENDATIONS:
  - Use GradCAM as a STARTING POINT for spatial understanding
  - Combine with gradient attribution for fine-grained analysis
  - Always check multiple target classes, not just the predicted one
  - For critical applications, validate with Score-CAM (gradient-free)
  - Never treat CAM as ground truth — it is a visualization aid
"""
```

---

## Summary

- **Original CAM** (Zhou et al. 2016) produces class-discriminative localization maps
  by linearly combining the last convolutional layer's feature maps using the
  classification layer's weights. It requires a GAP + single FC architecture.

- **GradCAM** (Selvaraju et al. 2017) generalizes CAM to any CNN by replacing the
  FC weights with gradient-based importance weights (global average pooling of the
  gradient of the class score). It is the most widely used CAM variant.

- **GradCAM++** (Chattopadhay et al. 2018) improves multi-object localization by
  using second-order gradient information for weighted (non-uniform) gradient
  pooling across spatial positions.

- **Score-CAM** (Wang et al. 2020) eliminates gradients entirely, using each
  feature map as a mask and measuring the resulting class score change. More
  principled but significantly slower (one forward pass per feature map).

- **Eigen-CAM** (Muhammad & Yeasin 2020) uses SVD to find the principal component
  of the activation maps. It is fast and gradient-free but NOT class-discriminative.

- CAM methods are especially valuable in **medical imaging** for verifying that
  models focus on clinically relevant regions, but border activation checks should
  be used to detect potential artifact-based predictions.

- **Fundamental limitations** include coarse resolution (bounded by feature map grid
  size), architectural assumptions (convolutional layers required), and the inability
  to provide global explanations or fine-grained pixel attribution.

---

## Exercises

### Exercise 1: GradCAM from Scratch (Coding)

1. Load a pretrained VGG-16 model (which has NO GAP layer).
2. Implement GradCAM targeting the last convolutional layer (`features[-1]`).
3. Verify that your implementation produces a reasonable heatmap on an ImageNet image.
4. Try targeting `features[10]` (an earlier layer). How does the CAM change?

### Exercise 2: Method Comparison (Coding + Analysis)

1. Using a pretrained ResNet-50:
   a. Generate GradCAM, GradCAM++, Score-CAM, and Eigen-CAM for the same image.
   b. Compute the pairwise Pearson correlation between all methods.
   c. Find an image where GradCAM and Eigen-CAM disagree significantly.
   d. Which method do you trust more, and why?

### Exercise 3: Multi-Class Discrimination (Coding)

1. Find an ImageNet image containing two different classes (e.g., a dog and a cat).
2. Generate GradCAM for each class separately.
3. Verify that the heatmaps focus on different regions.
4. What happens when you generate Eigen-CAM for both classes?

### Exercise 4: Medical Imaging Pipeline (Integration)

1. Download a chest X-ray dataset (e.g., CheXpert or NIH Chest X-ray).
2. Use a pretrained DenseNet-121 (common in medical imaging).
3. Implement the `medical_imaging_gradcam` pipeline from Section 7.
4. Run the border activation check on 20 images. How many trigger the warning?
5. Manually inspect the flagged images and determine if the warnings are legitimate.

### Exercise 5: Sanity Check for CAM Methods (Research)

1. Apply Adebayo et al.'s model randomization test (Lesson 02, Section 5) to GradCAM.
2. Progressively randomize ResNet-50's layers from top to bottom.
3. Plot the Pearson correlation between the trained CAM and each randomized CAM.
4. Does GradCAM pass the sanity check? At what layer does the CAM begin to degrade?
5. Compare with Score-CAM. Does gradient-free produce different sanity check behavior?

---

[Previous: Gradient Attribution](./02_Gradient_Attribution.md) | [Overview](./00_Overview.md) | [Next: Attention Interpretation](./04_Attention_Interpretation.md)

---

**License**: CC BY-NC 4.0
