# Lesson 2: 그래디언트 어트리뷰션 방법(Gradient Attribution Methods)

[이전: 해석 가능성 기초](./01_Interpretability_Foundations.md) | [다음: 클래스 활성화 매핑](./03_Class_Activation_Mapping.md)

---

## 학습 목표

- 기본 원리부터 바닐라 그래디언트 현저성 맵(Vanilla Gradient Saliency Maps)을 유도하고 PyTorch로 구현
- 적분 그래디언트(Integrated Gradients) 공리(민감도와 구현 불변성)를 설명하고 바닐라 그래디언트가 이를 위반하는 이유를 설명
- 적절한 기준선(Baseline) 선택과 수렴 확인을 통한 적분 그래디언트 구현
- SmoothGrad를 적용하여 그래디언트 노이즈를 줄이고 시각적으로 더 깨끗한 어트리뷰션 맵 생성
- Adebayo et al.의 정상성 검사(Sanity Checks)(모델 및 데이터 무작위화 테스트)를 사용하여 그래디언트 방법을 비판적으로 평가

---

그래디언트 기반 어트리뷰션 방법은 신경망에 *"어떤 입력 특성이 이 예측에 중요했는가?"*라고 물어보는 가장 직접적인 방법입니다. 핵심 아이디어는 우아합니다: 입력에 대한 출력의 그래디언트를 계산하고, 각 그래디언트 성분의 크기(또는 부호)를 해당 입력 특성의 중요도 측정치로 사용합니다.

이 레슨은 단순한 그래디언트 현저성 맵에서 이론적으로 원리에 기반한 적분 그래디언트 방법까지의 진행 과정을 다루며, 노이즈 감소 기법(SmoothGrad)과 모든 실무자가 반드시 이해해야 하는 비판적 평가(Adebayo et al. 2018)를 포함합니다. 또한 역사적 방법(DeconvNet, Guided Backpropagation)을 다루어 그 계보를 이해하고 일부 인기 있는 방법이 근본적으로 결함이 있는 이유를 설명합니다.

---

## 1. 바닐라 그래디언트 현저성 맵(Vanilla Gradient Saliency Maps)

### 1.1 수학적 기초

가장 단순한 그래디언트 어트리뷰션 방법은 Simonyan et al. (2014)이 *"Deep Inside Convolutional Networks"*에서 도입했습니다. 아이디어는 직관적입니다:

```python
"""
VANILLA GRADIENT SALIENCY

Given:
  - A trained model F: R^n → R^C  (maps n-dim input to C class scores)
  - An input x ∈ R^n
  - A target class c

The saliency map is simply the gradient of the class score with respect
to the input:

    S(x)_i = ∂F_c(x) / ∂x_i

Intuition: If changing pixel x_i by a tiny amount ε causes a large change
in the class score F_c, then x_i is "important" for class c.

For images with 3 channels (RGB), we typically take the maximum absolute
gradient across channels for each spatial position:

    S(x)_{h,w} = max_{c ∈ {R,G,B}} |∂F_c(x) / ∂x_{h,w,c}|

This produces a single-channel heatmap highlighting important regions.

Properties:
  ✓ Fast: single backward pass
  ✓ Model-specific: uses actual gradients, not approximations
  ✗ Noisy: gradients of deep networks are notoriously noisy
  ✗ Saturated: if a feature is in a flat region of F, its gradient
    is near zero even if the feature is important (gradient saturation)
  ✗ Not faithful: fails Adebayo et al. sanity checks (see Section 5)
"""
```

### 1.2 PyTorch 구현

```python
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def compute_vanilla_gradient(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int
) -> np.ndarray:
    """
    Compute the vanilla gradient saliency map for a given input and target class.

    Parameters
    ----------
    model : nn.Module
        A pretrained classification model.
    input_tensor : torch.Tensor
        Preprocessed input image, shape (1, C, H, W).
        Must have requires_grad=True.
    target_class : int
        Index of the target class.

    Returns
    -------
    np.ndarray
        Saliency map, shape (H, W). Values represent the importance of
        each spatial position for the target class.
    """
    # Ensure the model is in evaluation mode
    # This matters because BatchNorm and Dropout behave differently
    # during training vs evaluation
    model.eval()

    # Enable gradient computation for the input
    # By default, leaf tensors created by the user have requires_grad=False
    # We need to explicitly enable it to compute ∂F/∂x
    input_tensor.requires_grad_(True)

    # Forward pass: compute the class scores
    output = model(input_tensor)

    # We want the gradient of a scalar, so we extract the target class score
    # output shape is (1, num_classes), so output[0, target_class] is a scalar
    target_score = output[0, target_class]

    # Backward pass: compute ∂F_c/∂x
    # This populates input_tensor.grad with the gradient
    model.zero_grad()
    target_score.backward()

    # Extract the gradient
    # input_tensor.grad has the same shape as input_tensor: (1, C, H, W)
    gradient = input_tensor.grad.data[0]  # shape: (C, H, W)

    # Convert to saliency map by taking the max absolute gradient across
    # the color channels. This follows the original Simonyan et al. approach.
    # Why max instead of mean? Because a feature that is important in ANY
    # channel is important for the prediction.
    saliency = gradient.abs().max(dim=0)[0]  # shape: (H, W)

    return saliency.cpu().numpy()


def visualize_saliency(
    original_image: np.ndarray,
    saliency_map: np.ndarray,
    title: str = "Vanilla Gradient Saliency"
) -> None:
    """
    Visualize the saliency map overlaid on the original image.

    Parameters
    ----------
    original_image : np.ndarray
        Original image in RGB format, shape (H, W, 3), values [0, 255].
    saliency_map : np.ndarray
        Saliency values, shape (H, W).
    title : str
        Plot title.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Raw saliency map
    axes[1].imshow(saliency_map, cmap="hot")
    axes[1].set_title(title)
    axes[1].axis("off")

    # Overlay: saliency on top of original image
    axes[2].imshow(original_image)
    # We normalize the saliency to [0, 1] for alpha blending
    normalized_saliency = (saliency_map - saliency_map.min()) / (
        saliency_map.max() - saliency_map.min() + 1e-8
    )
    axes[2].imshow(normalized_saliency, cmap="jet", alpha=0.5)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("vanilla_gradient_saliency.png", dpi=150)
    plt.show()


# --- Full pipeline ---

def vanilla_gradient_pipeline():
    """
    Complete pipeline: load a pretrained model, preprocess an image,
    compute saliency, and visualize.
    """
    # Load a pretrained ResNet-50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.eval()

    # Standard ImageNet preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # ImageNet normalization: these are the dataset mean and std
        # Without this normalization, the model's predictions will be garbage
        # because it was trained on normalized inputs
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # Load and preprocess an image
    # Replace with your own image path
    image = Image.open("example_image.jpg").convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

    # Get the predicted class
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = output.argmax(dim=1).item()

    print(f"Predicted class index: {predicted_class}")

    # Compute saliency
    saliency = compute_vanilla_gradient(model, input_tensor, predicted_class)

    # Visualize
    # Convert the original image to a numpy array for display
    original_np = np.array(image.resize((224, 224)))
    visualize_saliency(original_np, saliency)


if __name__ == "__main__":
    vanilla_gradient_pipeline()
```

### 1.3 그래디언트 x 입력(Gradient x Input)

```python
"""
GRADIENT × INPUT

A simple improvement over vanilla gradients: multiply each gradient by
the corresponding input value.

    Attribution_i = x_i * (∂F_c / ∂x_i)

Motivation: The gradient tells us the SENSITIVITY of the output to
changes in x_i. But sensitivity alone does not capture CONTRIBUTION.
A feature might have a large gradient but a small value, contributing
little to the actual output.

Gradient × Input approximates the first-order Taylor expansion:
    F(x) ≈ F(0) + Σ_i x_i * (∂F/∂x_i)

So x_i * (∂F/∂x_i) represents the contribution of feature i to the
difference between F(x) and F(0).

Advantages over vanilla gradient:
  ✓ Better captures actual contribution, not just sensitivity
  ✓ Same computational cost (single backward pass)

Disadvantages:
  ✗ Choice of baseline (zero) is arbitrary
  ✗ Still noisy
  ✗ Still suffers from gradient saturation
"""


def compute_gradient_times_input(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int
) -> np.ndarray:
    """
    Compute Gradient × Input attribution.

    This is a drop-in replacement for vanilla gradient. The only change
    is that we multiply the gradient by the input value.
    """
    model.eval()
    input_tensor.requires_grad_(True)

    output = model(input_tensor)
    target_score = output[0, target_class]

    model.zero_grad()
    target_score.backward()

    gradient = input_tensor.grad.data[0]  # (C, H, W)
    input_values = input_tensor.data[0]    # (C, H, W)

    # Element-wise multiplication: gradient * input
    # This gives us the first-order contribution of each feature
    attribution = (gradient * input_values).abs().max(dim=0)[0]

    return attribution.cpu().numpy()
```

---

## 2. 적분 그래디언트(Integrated Gradients)

바닐라 그래디언트는 근본적인 이론적 문제를 가지고 있습니다. 적분 그래디언트(Sundararajan, Taly, and Yan, 2017)는 공리적 접근법으로 이를 해결합니다.

### 2.1 바닐라 그래디언트의 문제점

```python
"""
WHY VANILLA GRADIENTS FAIL: THE SATURATION PROBLEM

Consider a ReLU network with input x and output F(x):

    F(x) = ReLU(w * x + b)

If w * x + b > 0 (active region):
    F(x) = w * x + b
    ∂F/∂x = w           ← gradient is non-zero, good

If w * x + b ≤ 0 (saturated region):
    F(x) = 0
    ∂F/∂x = 0           ← gradient is ZERO even though x matters!

Real example: A model has learned that "high income → approve loan."
For a person with income = $200k (well above the threshold), the
model is in a flat (saturated) region:
    ∂F/∂income ≈ 0

Vanilla gradient says: "Income is not important for this prediction."
But that is clearly wrong! Income is THE REASON for the approval.

This is the SENSITIVITY axiom violation:
  If changing a feature from its baseline value changes the output,
  the feature should receive non-zero attribution.

Vanilla gradients violate this because they only look at LOCAL sensitivity
(the gradient at the specific input point), ignoring the GLOBAL effect of
the feature on the prediction.
"""
```

### 2.2 공리와 경로 적분 공식화

```python
"""
INTEGRATED GRADIENTS: AXIOMATIC FOUNDATION

Sundararajan et al. (2017) define two axioms that any attribution method
SHOULD satisfy:

AXIOM 1: SENSITIVITY
  If the input x and a baseline x' differ in exactly one feature i,
  and F(x) ≠ F(x'), then feature i should receive non-zero attribution.

  Why it matters: If a feature provably affects the output, the method
  must acknowledge it. Vanilla gradients violate this (see above).

AXIOM 2: IMPLEMENTATION INVARIANCE
  If two networks compute exactly the same function (same input-output
  mapping), they should produce exactly the same attributions,
  regardless of their internal architecture.

  Why it matters: Explanations should describe what the model COMPUTES,
  not how it is IMPLEMENTED. Two mathematically equivalent networks
  should get the same explanation.

  Methods that violate this: DeconvNet, Guided Backpropagation
  (they depend on the network architecture, not just the function).


THE INTEGRATED GRADIENTS FORMULA

Given:
  - Input x ∈ R^n
  - Baseline x' ∈ R^n (typically the zero vector for images)
  - Model F: R^n → R

The Integrated Gradients attribution for feature i is:

    IG_i(x, x') = (x_i - x'_i) × ∫₀¹ (∂F(x' + α(x - x')) / ∂x_i) dα

In words: walk along the straight line from the baseline x' to the
input x, computing the gradient at each step. The attribution is the
integral (sum) of these gradients, scaled by the input-baseline
difference.

Intuition: Instead of looking at the gradient at a single point
(which may be in a saturated region), we look at gradients along
the ENTIRE PATH from baseline to input. This captures the cumulative
effect of the feature.

Key theorem: Integrated Gradients is the UNIQUE method that satisfies
both Sensitivity and Implementation Invariance (plus completeness and
linearity).
"""
```

### 2.3 완전한 구현

```python
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


def integrated_gradients(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    baseline: Optional[torch.Tensor] = None,
    num_steps: int = 300,
    internal_batch_size: int = 32
) -> Tuple[np.ndarray, float]:
    """
    Compute Integrated Gradients attribution for an image classifier.

    This implementation follows Sundararajan et al. (2017) exactly,
    with practical optimizations for GPU efficiency.

    Parameters
    ----------
    model : nn.Module
        Pretrained classification model.
    input_tensor : torch.Tensor
        Input image, shape (1, C, H, W).
    target_class : int
        Target class index.
    baseline : torch.Tensor, optional
        Baseline input. If None, uses a black image (all zeros).
        The choice of baseline is critical (see Section 2.4).
    num_steps : int
        Number of interpolation steps for the Riemann approximation
        of the integral. More steps = more accurate but slower.
        Rule of thumb: 50 for quick checks, 300 for publication.
    internal_batch_size : int
        Process this many interpolated images at once for GPU efficiency.
        Higher = faster but more memory.

    Returns
    -------
    Tuple[np.ndarray, float]
        - attributions: shape (C, H, W), the integrated gradients
        - convergence_delta: should be close to 0 if the approximation
          is accurate (see Section 2.5)
    """
    model.eval()
    device = input_tensor.device

    # Default baseline: black image (all zeros AFTER normalization)
    # This represents the "absence of information" for the model
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)

    # Validate shapes
    assert input_tensor.shape == baseline.shape, (
        f"Input shape {input_tensor.shape} != baseline shape {baseline.shape}"
    )

    # The difference between input and baseline
    # This is the "direction" we will integrate along
    delta = input_tensor - baseline  # shape: (1, C, H, W)

    # Create interpolation alphas: 0, 1/N, 2/N, ..., 1
    # These define the points along the straight-line path
    # from baseline to input where we will evaluate the gradient
    alphas = torch.linspace(0, 1, num_steps + 1, device=device)

    # Accumulate gradients in a running sum
    total_gradients = torch.zeros_like(input_tensor)

    # Process interpolated images in batches for efficiency
    # Without batching, we would need num_steps forward+backward passes
    for start_idx in range(0, num_steps + 1, internal_batch_size):
        end_idx = min(start_idx + internal_batch_size, num_steps + 1)
        batch_alphas = alphas[start_idx:end_idx]

        # Create interpolated inputs: x' + alpha * (x - x')
        # batch_alphas has shape (batch_size,)
        # We need to broadcast to (batch_size, C, H, W)
        alpha_expanded = batch_alphas.view(-1, 1, 1, 1)
        interpolated = baseline + alpha_expanded * delta

        # The interpolated inputs need gradients for backprop
        interpolated = interpolated.detach().requires_grad_(True)

        # Forward pass on the batch
        output = model(interpolated)

        # Sum the target class scores across the batch
        # We sum (not mean) because we want the total gradient
        target_scores = output[:, target_class].sum()

        # Backward pass: compute gradients for all interpolated inputs at once
        model.zero_grad()
        target_scores.backward()

        # Accumulate the gradients
        # interpolated.grad has shape (batch_size, C, H, W)
        total_gradients += interpolated.grad.sum(dim=0, keepdim=True)

    # Approximate the integral using the trapezoidal rule
    # The simple Riemann sum divides by (num_steps + 1)
    avg_gradients = total_gradients / (num_steps + 1)

    # Scale by the input-baseline difference
    # This is the (x_i - x'_i) factor in the IG formula
    attributions = (delta * avg_gradients).squeeze(0)  # shape: (C, H, W)

    # --- Completeness check (convergence delta) ---
    # The completeness axiom states:
    #   Σ_i IG_i(x, x') = F(x) - F(x')
    # If our approximation is accurate, the sum of attributions should
    # equal the difference in model outputs.
    with torch.no_grad():
        output_at_input = model(input_tensor)[0, target_class].item()
        output_at_baseline = model(baseline)[0, target_class].item()

    expected_diff = output_at_input - output_at_baseline
    actual_sum = attributions.sum().item()
    convergence_delta = abs(expected_diff - actual_sum)

    return attributions.detach().cpu().numpy(), convergence_delta


def visualize_integrated_gradients(
    original_image: np.ndarray,
    attributions: np.ndarray,
    convergence_delta: float,
    percentile: float = 99
) -> None:
    """
    Visualize Integrated Gradients attributions.

    Parameters
    ----------
    original_image : np.ndarray
        Original image, shape (H, W, 3), values [0, 255].
    attributions : np.ndarray
        IG attributions, shape (C, H, W).
    convergence_delta : float
        Completeness check value (should be close to 0).
    percentile : float
        Clip attribution values at this percentile for visualization.
        Prevents outliers from washing out the heatmap.
    """
    # Aggregate across channels: sum of absolute values
    # We use absolute value because both positive and negative
    # attributions indicate importance (positive = supports class,
    # negative = opposes class)
    attr_map = np.abs(attributions).sum(axis=0)  # shape: (H, W)

    # Clip at percentile to handle outliers
    # Without this, a few very high attribution pixels can make
    # everything else invisible
    vmax = np.percentile(attr_map, percentile)
    attr_map = np.clip(attr_map, 0, vmax)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Positive attributions only (features supporting the prediction)
    pos_attr = np.maximum(attributions.sum(axis=0), 0)
    axes[1].imshow(pos_attr, cmap="Reds")
    axes[1].set_title("Positive Attributions")
    axes[1].axis("off")

    # Negative attributions only (features opposing the prediction)
    neg_attr = np.abs(np.minimum(attributions.sum(axis=0), 0))
    axes[2].imshow(neg_attr, cmap="Blues")
    axes[2].set_title("Negative Attributions")
    axes[2].axis("off")

    # Overlay
    axes[3].imshow(original_image)
    norm_attr = attr_map / (attr_map.max() + 1e-8)
    axes[3].imshow(norm_attr, cmap="jet", alpha=0.5)
    axes[3].set_title(f"Overlay (Δ={convergence_delta:.4f})")
    axes[3].axis("off")

    plt.suptitle("Integrated Gradients Attribution", fontsize=14)
    plt.tight_layout()
    plt.savefig("integrated_gradients.png", dpi=150)
    plt.show()

    print(f"Convergence delta: {convergence_delta:.6f}")
    if convergence_delta < 0.05:
        print("  ✓ Good convergence (delta < 0.05)")
    else:
        print("  ✗ Poor convergence — increase num_steps")
```

### 2.4 기준선 선택(Baseline Selection)

```python
"""
THE CRITICAL ROLE OF BASELINE SELECTION

The baseline x' represents the "absence of information." Integrated
Gradients measures each feature's contribution relative to this
baseline. Different baselines lead to different attributions.

Common choices:

1. BLACK IMAGE (all zeros after normalization)
   - Most common for image models
   - Represents "no visual information"
   - Problem: black pixels are not "absent" — they are a specific
     color. A model might have learned to use black as a feature.

2. WHITE IMAGE (all max values)
   - Alternative for images
   - Same problem as black but in reverse

3. GAUSSIAN NOISE (random baseline)
   - Represents "uninformative noise"
   - More principled than black for some models
   - Nondeterministic: different runs give different attributions
   - Solution: average over multiple random baselines

4. BLURRED VERSION OF THE INPUT
   - Represents "same low-frequency content, no details"
   - Good for object detection: highlights fine-grained features

5. TRAINING SET MEAN
   - Represents the "average example"
   - Attributions measure deviation from average behavior
   - Good for tabular data; debatable for images

6. UNIFORM DISTRIBUTION (for text embeddings)
   - Represents "no word is more likely than any other"
   - Used for NLP models where zero embedding is meaningless

Best practice: Try multiple baselines and check that the top-attributed
features are consistent. If attributions change drastically with
different baselines, the results are unreliable.
"""


def compare_baselines(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    original_image: np.ndarray,
    num_steps: int = 200
) -> None:
    """
    Compare Integrated Gradients with different baseline choices.

    This helps identify whether the attributions are robust to
    baseline selection or an artifact of a specific choice.
    """
    device = input_tensor.device

    baselines = {
        "Black (zeros)": torch.zeros_like(input_tensor),
        "White (ones)": torch.ones_like(input_tensor),
        "Gaussian noise": torch.randn_like(input_tensor),
        "Uniform gray": torch.full_like(input_tensor, 0.5),
    }

    fig, axes = plt.subplots(1, len(baselines) + 1, figsize=(20, 4))

    axes[0].imshow(original_image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    for idx, (name, baseline) in enumerate(baselines.items()):
        attrs, delta = integrated_gradients(
            model, input_tensor, target_class,
            baseline=baseline, num_steps=num_steps
        )

        # Visualize absolute sum across channels
        attr_map = np.abs(attrs).sum(axis=0)
        vmax = np.percentile(attr_map, 99)
        attr_map = np.clip(attr_map, 0, vmax)

        axes[idx + 1].imshow(attr_map, cmap="hot")
        axes[idx + 1].set_title(f"{name}\nΔ={delta:.4f}")
        axes[idx + 1].axis("off")

    plt.suptitle("Integrated Gradients: Baseline Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig("ig_baseline_comparison.png", dpi=150)
    plt.show()
```

### 2.5 수렴 검증

```python
def check_convergence(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    baseline: Optional[torch.Tensor] = None,
    step_counts: list = None
) -> None:
    """
    Verify that Integrated Gradients converges as num_steps increases.

    The completeness axiom guarantees that:
        Σ_i IG_i(x, x') = F(x) - F(x')

    If the Riemann approximation is accurate, the convergence delta
    (difference between left and right sides) should approach zero.

    This function computes IG at different step counts and plots the
    convergence behavior. If the delta does not decrease, there is
    a numerical issue.
    """
    if step_counts is None:
        step_counts = [10, 25, 50, 100, 200, 300, 500, 1000]

    deltas = []

    for n_steps in step_counts:
        _, delta = integrated_gradients(
            model, input_tensor, target_class,
            baseline=baseline, num_steps=n_steps
        )
        deltas.append(delta)
        print(f"  Steps: {n_steps:>5d}  |  Convergence delta: {delta:.6f}")

    # Plot convergence
    plt.figure(figsize=(8, 5))
    plt.plot(step_counts, deltas, 'bo-', linewidth=2, markersize=8)
    plt.xlabel("Number of Integration Steps", fontsize=12)
    plt.ylabel("Convergence Delta (|Σ IG - ΔF|)", fontsize=12)
    plt.title("Integrated Gradients Convergence Check", fontsize=14)
    plt.yscale("log")
    plt.grid(True, alpha=0.3)

    # Add a reference line at delta = 0.01
    plt.axhline(y=0.01, color='r', linestyle='--', label='Target: Δ < 0.01')
    plt.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig("ig_convergence.png", dpi=150)
    plt.show()

    print(f"\nFinal delta at {step_counts[-1]} steps: {deltas[-1]:.6f}")
    if deltas[-1] < 0.01:
        print("✓ Convergence achieved")
    else:
        print("✗ Consider increasing num_steps further")
```

---

## 3. SmoothGrad: 노이즈 평균화

### 3.1 노이즈 문제

```python
"""
WHY GRADIENTS ARE NOISY

Neural network loss landscapes are highly irregular. Even small changes
in the input can cause large changes in the gradient direction. This
means that vanilla gradient saliency maps are visually noisy — they
highlight individual pixels seemingly at random, rather than coherent
regions.

The root cause: ReLU activations create piecewise-linear functions.
The gradient is constant within each linear region but can change
discontinuously at the boundary. A tiny perturbation can move the
input across a boundary, completely changing the gradient.

SMOOTHGRAD (Smilkov et al. 2017)

Key idea: Average the gradient over many slightly perturbed versions
of the input. The noise in individual gradients cancels out, revealing
the stable signal.

    SmoothGrad_i(x) = (1/N) Σ_{k=1}^{N} (∂F / ∂x_i)(x + ε_k)

where ε_k ~ N(0, σ²I) is Gaussian noise.

The hyperparameters:
  - N: number of samples (typically 50-200)
  - σ: noise standard deviation (typically 10-20% of input range)

Higher N → smoother but slower
Higher σ → smoother but risk of moving too far from the original input

SmoothGrad can be applied to ANY gradient-based method:
  - SmoothGrad + Vanilla Gradient
  - SmoothGrad + Integrated Gradients
  - SmoothGrad + Gradient × Input
"""
```

### 3.2 구현

```python
def smooth_grad(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    num_samples: int = 50,
    noise_level: float = 0.15,
    base_method: str = "vanilla"
) -> np.ndarray:
    """
    Compute SmoothGrad: noise-averaged gradient attribution.

    Parameters
    ----------
    model : nn.Module
        Pretrained classification model.
    input_tensor : torch.Tensor
        Input image, shape (1, C, H, W).
    target_class : int
        Target class index.
    num_samples : int
        Number of noisy samples to average over.
        More samples → smoother result but slower.
        50 is a good default; 200 for publication-quality.
    noise_level : float
        Standard deviation of Gaussian noise as a fraction of the
        input range. 0.15 means 15% of (max - min).
    base_method : str
        Which gradient method to smooth: "vanilla" or "gradient_x_input".

    Returns
    -------
    np.ndarray
        Smoothed saliency map, shape (H, W).
    """
    model.eval()

    # Compute the noise standard deviation based on the input range
    # This makes the noise level scale-invariant
    input_range = input_tensor.max() - input_tensor.min()
    sigma = noise_level * input_range.item()

    # Accumulator for the averaged gradients
    smooth_saliency = np.zeros((input_tensor.shape[2], input_tensor.shape[3]))

    for i in range(num_samples):
        # Add Gaussian noise to the input
        noise = torch.randn_like(input_tensor) * sigma
        noisy_input = (input_tensor + noise).detach().requires_grad_(True)

        # Forward pass
        output = model(noisy_input)
        target_score = output[0, target_class]

        # Backward pass
        model.zero_grad()
        target_score.backward()

        gradient = noisy_input.grad.data[0]  # (C, H, W)

        if base_method == "gradient_x_input":
            # Multiply gradient by the ORIGINAL input (not the noisy one)
            # This is a design choice: we want the noise to affect only
            # the gradient estimation, not the input values
            gradient = gradient * input_tensor.data[0]

        # Convert to single-channel saliency map
        saliency = gradient.abs().max(dim=0)[0].cpu().numpy()
        smooth_saliency += saliency

    # Average over all samples
    smooth_saliency /= num_samples

    return smooth_saliency


def compare_smooth_vs_vanilla(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    original_image: np.ndarray
) -> None:
    """
    Side-by-side comparison of vanilla gradient and SmoothGrad.

    This visually demonstrates the noise reduction effect of SmoothGrad.
    """
    # Vanilla gradient
    vanilla = compute_vanilla_gradient(model, input_tensor.clone(), target_class)

    # SmoothGrad with different sample counts
    sg_50 = smooth_grad(model, input_tensor, target_class, num_samples=50)
    sg_200 = smooth_grad(model, input_tensor, target_class, num_samples=200)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Normalize each saliency map independently for fair comparison
    for ax, saliency, title in [
        (axes[1], vanilla, "Vanilla Gradient"),
        (axes[2], sg_50, "SmoothGrad (N=50)"),
        (axes[3], sg_200, "SmoothGrad (N=200)"),
    ]:
        vmax = np.percentile(saliency, 99)
        ax.imshow(np.clip(saliency, 0, vmax), cmap="hot")
        ax.set_title(title)
        ax.axis("off")

    plt.suptitle("Effect of SmoothGrad on Gradient Attribution", fontsize=14)
    plt.tight_layout()
    plt.savefig("smoothgrad_comparison.png", dpi=150)
    plt.show()
```

---

## 4. DeconvNet과 Guided Backpropagation (역사적 방법)

### 4.1 더 이상 사용되지 않는 방법을 공부하는 이유

```python
"""
DeconvNet (Zeiler & Fergus, 2014) and Guided Backpropagation
(Springenberg et al., 2015) are historically important but have been
shown to produce explanations that are NOT faithful to the model.

We study them for three reasons:
  1. They are still cited and sometimes used in practice (by people
     who do not know they are flawed)
  2. Understanding WHY they fail deepens your understanding of what
     makes a good attribution method
  3. Adebayo et al.'s sanity checks (Section 5) were designed
     specifically to expose their failures


DECONVNET
  Modifies backpropagation at ReLU layers:
  Instead of: mask = (input > 0)    [standard backprop]
  Uses:       mask = (gradient > 0)  [only propagate positive gradients]

  Effect: Produces cleaner visualizations by filtering out negative
  gradient signal. But this filtering means the visualization does NOT
  reflect the model's actual computation.

GUIDED BACKPROPAGATION
  Combines both masks:
  mask = (input > 0) AND (gradient > 0)

  Effect: Even cleaner visualizations. Highlights edges and textures
  that look "meaningful" to humans.

  The fatal flaw: Guided Backprop produces nearly IDENTICAL
  visualizations for a trained model and a RANDOM model.
  It is essentially performing edge detection on the input image,
  not explaining the model.
"""
```

### 4.2 비교를 위한 구현

```python
class GuidedBackpropReLU(torch.autograd.Function):
    """
    Custom ReLU that implements Guided Backpropagation.

    During forward pass: standard ReLU (mask negative inputs)
    During backward pass: mask gradients where EITHER the input
    was negative OR the gradient is negative.

    WARNING: This method fails Adebayo et al. sanity checks.
    Implemented here for educational comparison only.
    """

    @staticmethod
    def forward(ctx, input):
        # Standard ReLU forward
        positive_mask = (input > 0).float()
        ctx.save_for_backward(input, positive_mask)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, positive_mask = ctx.saved_tensors
        # Guided Backprop: only propagate gradients where BOTH
        # the input was positive AND the incoming gradient is positive
        guided_grad = grad_output * positive_mask * (grad_output > 0).float()
        return guided_grad


def apply_guided_backprop(model: nn.Module) -> nn.Module:
    """
    Replace all ReLU activations in a model with GuidedBackpropReLU.

    This modifies the model in-place. Use on a copy of the model
    to avoid corrupting the original.

    Parameters
    ----------
    model : nn.Module
        Model to modify.

    Returns
    -------
    nn.Module
        Modified model with Guided Backprop ReLUs.
    """
    import copy
    guided_model = copy.deepcopy(model)

    # Replace all ReLU modules
    for name, module in guided_model.named_modules():
        if isinstance(module, nn.ReLU):
            # Navigate to the parent module and replace the ReLU
            parts = name.split('.')
            parent = guided_model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], GuidedBackpropReLU.apply)

    return guided_model
```

---

## 5. 현저성 맵의 정상성 검사(Sanity Checks for Saliency Maps) (Adebayo et al. 2018)

이것은 해석 가능한 AI에서 가장 중요한 논문 중 하나입니다. 여러 널리 사용되는 어트리뷰션 방법이 근본적으로 신뢰할 수 없음을 보여주었습니다.

### 5.1 두 가지 정상성 검사 테스트

```python
"""
ADEBAYO ET AL. (2018): "SANITY CHECKS FOR SALIENCY MAPS"

Key finding: Some popular saliency methods produce visually plausible
explanations that have NOTHING TO DO with what the model learned.

Two tests:

TEST 1: MODEL PARAMETER RANDOMIZATION
  Procedure:
    1. Take a trained model
    2. Progressively randomize its weights from the top layer down
       (cascade randomization) or randomly re-initialize all layers
    3. Compute the saliency map after each randomization
  Expected behavior:
    A faithful method's saliency map should CHANGE dramatically
    when the model is randomized, because the model's "reasoning"
    has been destroyed.
  Failure:
    Guided Backprop and DeconvNet produce nearly IDENTICAL saliency
    maps for the trained model and the fully randomized model.
    → They are explaining the INPUT, not the MODEL.

TEST 2: DATA RANDOMIZATION
  Procedure:
    1. Train the model on data with RANDOMLY SHUFFLED labels
    2. Compute saliency maps
  Expected behavior:
    A faithful method should produce DIFFERENT saliency maps,
    because the model learned different features (random features).
  Failure:
    Again, Guided Backprop and DeconvNet are invariant.


METHODS THAT PASS:
  ✓ Vanilla gradients (partially — changes with model randomization)
  ✓ Gradient × Input
  ✓ Integrated Gradients
  ✓ GradCAM (passes model randomization)
  ✓ SHAP (passes both)

METHODS THAT FAIL:
  ✗ Guided Backpropagation
  ✗ DeconvNet
  ✗ Guided GradCAM (the guided part fails)

LESSON: Visual plausibility is NOT the same as faithfulness.
A method can produce beautiful, edge-highlighting heatmaps that
look "correct" to humans but tell you nothing about the model.
"""
```

### 5.2 정상성 검사 구현

```python
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List
import copy


def cascade_randomize_model(
    model: nn.Module,
    num_layers_to_randomize: int
) -> nn.Module:
    """
    Progressively randomize model parameters from the top layer down.

    This simulates destroying what the model has learned, starting
    from the most abstract features (top layers) down to the most
    basic features (bottom layers).

    Parameters
    ----------
    model : nn.Module
        Trained model to randomize.
    num_layers_to_randomize : int
        Number of layers (from the top) to randomize.

    Returns
    -------
    nn.Module
        Model with some layers randomized.
    """
    randomized_model = copy.deepcopy(model)

    # Get all layers with learnable parameters
    # We reverse the list so that index 0 is the TOP (output) layer
    layers_with_params = []
    for name, module in randomized_model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            layers_with_params.append((name, module))

    layers_with_params.reverse()  # Top layer first

    # Randomize the specified number of layers
    for i in range(min(num_layers_to_randomize, len(layers_with_params))):
        name, module = layers_with_params[i]
        # Re-initialize the layer's weights with random values
        # using the same initialization scheme (Kaiming for conv, Xavier for linear)
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    return randomized_model


def model_randomization_test(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    attribution_fn: Callable,
    original_image: np.ndarray,
    method_name: str = "Method"
) -> None:
    """
    Perform Adebayo et al.'s model parameter randomization test.

    Computes attributions for the trained model and several progressively
    randomized versions. A faithful method should show visible degradation
    as more layers are randomized.

    Parameters
    ----------
    model : nn.Module
        Trained model.
    input_tensor : torch.Tensor
        Input image, shape (1, C, H, W).
    target_class : int
        Target class index.
    attribution_fn : callable
        Function(model, input_tensor, target_class) -> np.ndarray.
    original_image : np.ndarray
        Original image for visualization, shape (H, W, 3).
    method_name : str
        Name of the attribution method being tested.
    """
    # Count layers with parameters
    num_param_layers = sum(
        1 for m in model.modules()
        if hasattr(m, 'weight') and m.weight is not None
    )

    # Test points: 0 (trained), 25%, 50%, 75%, 100% randomized
    randomization_levels = [0, num_param_layers // 4, num_param_layers // 2,
                            3 * num_param_layers // 4, num_param_layers]

    fig, axes = plt.subplots(1, len(randomization_levels) + 1, figsize=(24, 4))

    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Store attributions for SSIM-like comparison
    all_attributions = []

    for idx, num_random in enumerate(randomization_levels):
        # Create a model with the specified number of randomized layers
        if num_random == 0:
            test_model = model
            label = "Trained\n(0% random)"
        else:
            test_model = cascade_randomize_model(model, num_random)
            pct = int(100 * num_random / num_param_layers)
            label = f"{pct}% random\n({num_random} layers)"

        # Compute attribution
        attribution = attribution_fn(test_model, input_tensor.clone(), target_class)
        all_attributions.append(attribution)

        # Visualize
        vmax = np.percentile(np.abs(attribution), 99)
        axes[idx + 1].imshow(np.clip(np.abs(attribution), 0, vmax), cmap="hot")
        axes[idx + 1].set_title(label)
        axes[idx + 1].axis("off")

    plt.suptitle(f"Sanity Check: {method_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"sanity_check_{method_name.lower().replace(' ', '_')}.png", dpi=150)
    plt.show()

    # Quantitative comparison: correlation between trained and randomized
    trained_flat = all_attributions[0].flatten()
    print(f"\n{method_name} — Correlation with trained model's attribution:")
    for idx, num_random in enumerate(randomization_levels):
        if num_random == 0:
            continue
        random_flat = all_attributions[idx].flatten()
        corr = np.corrcoef(trained_flat, random_flat)[0, 1]
        pct = int(100 * num_random / num_param_layers)
        status = "✓ PASS" if corr < 0.5 else "✗ FAIL"
        print(f"  {pct:>3d}% randomized: r = {corr:.3f}  {status}")

    print()
    print("Interpretation:")
    print("  A faithful method should show DECREASING correlation")
    print("  as more layers are randomized.")
    print("  If correlation stays high → method is NOT model-dependent.")
```

### 5.3 완전한 정상성 검사 스위트 실행

```python
def run_sanity_check_suite(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    original_image: np.ndarray
) -> None:
    """
    Run Adebayo et al. sanity checks on multiple attribution methods.

    This provides a comprehensive comparison showing which methods
    are model-dependent (faithful) and which are not.
    """
    print("=" * 60)
    print("ADEBAYO ET AL. SANITY CHECKS FOR SALIENCY MAPS")
    print("=" * 60)

    # Define attribution methods to test
    methods = {
        "Vanilla Gradient": lambda m, x, c: compute_vanilla_gradient(m, x, c),
        "Gradient x Input": lambda m, x, c: compute_gradient_times_input(m, x, c),
        "SmoothGrad": lambda m, x, c: smooth_grad(m, x, c, num_samples=50),
    }

    # We would also test Integrated Gradients, but it is slow for
    # many randomized models. In practice, IG passes the sanity checks.

    for method_name, method_fn in methods.items():
        print(f"\n--- Testing: {method_name} ---")
        model_randomization_test(
            model, input_tensor, target_class,
            method_fn, original_image, method_name
        )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Methods that should PASS (correlation drops with randomization):")
    print("  ✓ Vanilla Gradient")
    print("  ✓ Gradient × Input")
    print("  ✓ Integrated Gradients")
    print("  ✓ SmoothGrad (when applied to passing methods)")
    print("  ✓ GradCAM")
    print()
    print("Methods that typically FAIL (correlation stays high):")
    print("  ✗ Guided Backpropagation")
    print("  ✗ DeconvNet")
    print()
    print("Recommendation: ALWAYS run sanity checks before trusting")
    print("any gradient-based attribution method on a new model.")
```

---

## 6. PyTorch 훅(Hook)을 활용한 어트리뷰션

### 6.1 훅 기반 구현

```python
class GradientAttributor:
    """
    A reusable class for computing gradient-based attributions using
    PyTorch backward hooks.

    Hooks are the correct way to intercept gradients in PyTorch:
    - register_backward_hook: captures gradients flowing through a module
    - register_forward_hook: captures activations during forward pass

    This class encapsulates the hook lifecycle (register, use, remove)
    to prevent memory leaks.
    """

    def __init__(self, model: nn.Module):
        """
        Parameters
        ----------
        model : nn.Module
            The model to compute attributions for.
        """
        self.model = model
        self.model.eval()
        self.hooks = []
        self.gradients = {}
        self.activations = {}

    def register_hooks(self, target_layer: nn.Module, layer_name: str) -> None:
        """
        Register forward and backward hooks on a specific layer.

        Forward hook captures the layer's output (activations).
        Backward hook captures the gradient flowing through the layer.

        These are stored in dictionaries for later retrieval.

        Parameters
        ----------
        target_layer : nn.Module
            The layer to hook into.
        layer_name : str
            Name for storing the captured data.
        """

        def forward_hook(module, input, output):
            # Store the activations (detached from the computation graph
            # to prevent memory leaks)
            self.activations[layer_name] = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # grad_output is a tuple; we want the first element
            # This is the gradient of the loss w.r.t. the layer's output
            self.gradients[layer_name] = grad_output[0].detach()

        # Register both hooks and store handles for cleanup
        fh = target_layer.register_forward_hook(forward_hook)
        bh = target_layer.register_full_backward_hook(backward_hook)
        self.hooks.extend([fh, bh])

    def compute_attribution(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        method: str = "vanilla"
    ) -> np.ndarray:
        """
        Compute gradient-based attribution.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input, shape (1, C, H, W).
        target_class : int
            Target class index.
        method : str
            "vanilla", "gradient_x_input", or "smooth" (smooth_grad).

        Returns
        -------
        np.ndarray
            Attribution map.
        """
        input_tensor = input_tensor.detach().requires_grad_(True)

        # Forward
        output = self.model(input_tensor)
        score = output[0, target_class]

        # Backward
        self.model.zero_grad()
        score.backward()

        # Extract input gradient
        grad = input_tensor.grad.data[0]  # (C, H, W)

        if method == "vanilla":
            attribution = grad.abs().max(dim=0)[0]
        elif method == "gradient_x_input":
            attribution = (grad * input_tensor.data[0]).abs().max(dim=0)[0]
        else:
            raise ValueError(f"Unknown method: {method}")

        return attribution.cpu().numpy()

    def cleanup(self) -> None:
        """
        Remove all registered hooks.

        IMPORTANT: Always call this when done to prevent memory leaks.
        Hooks keep references to tensors and can cause GPU memory to
        accumulate over time.
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.gradients.clear()
        self.activations.clear()

    def __enter__(self):
        """Support context manager usage for automatic cleanup."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Automatically clean up hooks when exiting the context."""
        self.cleanup()
        return False


# --- Usage example ---

def hook_based_attribution_demo():
    """
    Demonstrate how to use the GradientAttributor class with hooks.

    The context manager pattern ensures hooks are always cleaned up,
    even if an exception occurs.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Create a dummy input (replace with a real image in practice)
    dummy_input = torch.randn(1, 3, 224, 224)

    # Use context manager for automatic cleanup
    with GradientAttributor(model) as attributor:
        # Register hooks on the last convolutional layer
        # For ResNet-50, this is layer4[-1].conv3
        target_layer = model.layer4[-1].conv3
        attributor.register_hooks(target_layer, "layer4_conv3")

        # Compute attribution
        # The forward and backward passes populate the hook data
        with torch.no_grad():
            pred_class = model(dummy_input).argmax(dim=1).item()

        attribution = attributor.compute_attribution(
            dummy_input, pred_class, method="vanilla"
        )

        # Access the captured activations and gradients
        print(f"Activation shape: {attributor.activations['layer4_conv3'].shape}")
        print(f"Gradient shape: {attributor.gradients['layer4_conv3'].shape}")
        print(f"Attribution shape: {attribution.shape}")

    # Hooks are automatically removed here
    print("Hooks cleaned up successfully.")


if __name__ == "__main__":
    hook_based_attribution_demo()
```

---

## 7. 실용적 비교: 방법들의 나란히 비교

```python
def comprehensive_gradient_comparison(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    original_image: np.ndarray
) -> None:
    """
    Side-by-side comparison of all gradient attribution methods
    covered in this lesson.

    This is the visualization you should create for every new model
    and dataset to understand which method works best in your context.
    """
    print("Computing attributions (this may take a few minutes)...")

    # 1. Vanilla Gradient
    vanilla = compute_vanilla_gradient(
        model, input_tensor.clone(), target_class
    )

    # 2. Gradient x Input
    grad_input = compute_gradient_times_input(
        model, input_tensor.clone(), target_class
    )

    # 3. SmoothGrad (vanilla)
    sg = smooth_grad(
        model, input_tensor, target_class,
        num_samples=100, noise_level=0.15
    )

    # 4. Integrated Gradients
    ig_attrs, ig_delta = integrated_gradients(
        model, input_tensor, target_class,
        num_steps=300
    )
    ig_map = np.abs(ig_attrs).max(axis=0)  # Max across channels

    # 5. SmoothGrad (gradient x input)
    sg_gi = smooth_grad(
        model, input_tensor, target_class,
        num_samples=100, noise_level=0.15,
        base_method="gradient_x_input"
    )

    # Visualization
    methods = [
        ("Original", original_image, False),
        ("Vanilla\nGradient", vanilla, True),
        ("Gradient\n× Input", grad_input, True),
        ("SmoothGrad\n(N=100)", sg, True),
        ("Integrated\nGradients", ig_map, True),
        ("SmoothGrad\n× Input", sg_gi, True),
    ]

    fig, axes = plt.subplots(1, len(methods), figsize=(24, 4))

    for idx, (title, data, is_heatmap) in enumerate(methods):
        if is_heatmap:
            vmax = np.percentile(data, 99)
            axes[idx].imshow(np.clip(data, 0, vmax), cmap="hot")
        else:
            axes[idx].imshow(data)
        axes[idx].set_title(title, fontsize=11)
        axes[idx].axis("off")

    plt.suptitle(
        f"Gradient Attribution Methods Comparison (class={target_class})",
        fontsize=14
    )
    plt.tight_layout()
    plt.savefig("gradient_methods_comparison.png", dpi=150)
    plt.show()

    print(f"\nIntegrated Gradients convergence delta: {ig_delta:.6f}")
    print("\nRecommendation:")
    print("  - For quick exploration: Vanilla Gradient or Gradient × Input")
    print("  - For publication: Integrated Gradients (check convergence delta)")
    print("  - For visual clarity: SmoothGrad on any base method")
    print("  - ALWAYS: Run Adebayo sanity checks before trusting results")
```

---

## 8. 흔한 함정과 모범 사례

```python
"""
COMMON PITFALLS IN GRADIENT ATTRIBUTION

1. FORGETTING TO SET model.eval()
   BatchNorm and Dropout behave differently in train vs eval mode.
   In train mode, BatchNorm uses batch statistics (noisy) and Dropout
   randomly zeros neurons. This adds noise to gradients that has
   nothing to do with feature importance.
   → ALWAYS call model.eval() before computing attributions.

2. NOT CHECKING CONVERGENCE FOR INTEGRATED GRADIENTS
   The Riemann sum approximation requires enough steps to converge.
   Using too few steps (e.g., 20) produces inaccurate attributions
   that do not satisfy the completeness axiom.
   → Always compute and report the convergence delta.
   → Use at least 300 steps for reliable results.

3. TREATING SALIENCY MAPS AS GROUND TRUTH
   Gradient attributions are one perspective on feature importance.
   They can disagree with other methods (SHAP, LIME) and with each
   other. No single method has a monopoly on truth.
   → Compare multiple methods. Trust regions where they agree.

4. CONFUSING SENSITIVITY WITH IMPORTANCE
   Vanilla gradients measure local sensitivity (∂F/∂x), not
   importance. A feature can be important (removing it changes the
   prediction) without having a large gradient (saturation).
   → Use Integrated Gradients for importance; use vanilla gradient
     for sensitivity analysis.

5. OVER-INTERPRETING NOISY SALIENCY MAPS
   Individual pixel attributions in vanilla gradient are noisy and
   unreliable. Do not read significance into individual bright or
   dark pixels.
   → Apply SmoothGrad or use Integrated Gradients.
   → Focus on REGIONS, not individual pixels.

6. IGNORING THE BASELINE FOR INTEGRATED GRADIENTS
   The baseline choice affects the attributions. Using only one
   baseline and treating the result as definitive is risky.
   → Test multiple baselines and report which features are
     consistently important.

7. MEMORY LEAKS FROM HOOKS
   Forgetting to remove PyTorch hooks causes memory to accumulate,
   eventually crashing the process.
   → Use context managers or try/finally blocks.
   → Call hook.remove() when done.

8. USING METHODS THAT FAIL SANITY CHECKS
   Guided Backpropagation and DeconvNet produce pretty but
   meaningless visualizations.
   → Never use them for model explanation.
   → Run Adebayo et al. sanity checks on any new method.
"""
```

---

## 요약

- **바닐라 그래디언트 현저성(Vanilla Gradient Saliency)** (Simonyan et al. 2014)은 단일 역전파 패스로 dF/dx를 계산합니다. 빠르지만 노이즈가 많고 그래디언트 포화(Gradient Saturation) 문제를 겪습니다: 평탄한 영역의 특성은 중요하더라도 0의 어트리뷰션을 받습니다.

- **그래디언트 x 입력(Gradient x Input)**은 그래디언트에 입력 값을 곱하여 바닐라 방법을 개선하며, 1차 테일러 기여도를 근사합니다. 같은 속도에 약간 더 나은 어트리뷰션을 제공하지만, 여전히 포화 및 기준선 모호성 문제가 있습니다.

- **적분 그래디언트(Integrated Gradients)** (Sundararajan et al. 2017)는 기준선에서 입력까지의 경로를 따라 그래디언트를 적분하여 포화 문제를 해결합니다. 민감도(Sensitivity)와 구현 불변성(Implementation Invariance) 공리를 모두 만족하는 유일한 방법입니다. 항상 완전성 검사(Completeness Check)를 통해 수렴을 검증하세요.

- **SmoothGrad** (Smilkov et al. 2017)는 입력의 노이즈가 추가된 여러 버전에 대한 어트리뷰션을 평균화하여 그래디언트 노이즈를 줄입니다. 모든 그래디언트 기반 방법과 결합할 수 있습니다.

- **Adebayo et al. (2018) 정상성 검사(Sanity Checks)**는 필수적입니다: 방법의 어트리뷰션이 실제로 모델(모델 무작위화 테스트)과 데이터(데이터 무작위화 테스트)에 의존하는지 테스트합니다. Guided Backpropagation과 DeconvNet은 이러한 테스트에 실패하며 모델 설명에 사용해서는 안 됩니다.

- **PyTorch 훅(Hook)** (register_forward_hook, register_full_backward_hook)은 그래디언트와 활성화를 가로채는 올바른 메커니즘입니다. 메모리 누수를 방지하기 위해 항상 훅을 정리하세요.

---

## 연습문제

### 연습문제 1: 구현 및 비교 (코딩)

1. 사전 학습된 VGG-16 모델과 원하는 이미지를 로드하세요.
2. 바닐라 그래디언트, 그래디언트 x 입력, SmoothGrad를 구현하세요.
3. 세 가지를 나란히 시각화하세요.
4. 어떤 방법이 가장 집중된 어트리뷰션을 생성하는가? 왜 그런가?

### 연습문제 2: 적분 그래디언트 심화 (코딩)

1. MNIST에서 학습된 간단한 2층 MLP에 대해 적분 그래디언트를 구현하세요.
2. 숫자 "3"과 숫자 "8"에 대한 어트리뷰션을 계산하세요.
3. 완전성 공리를 검증하세요 (수렴 델타 < 0.01).
4. 기준선을 비교하세요: 검은 이미지 vs. 가우시안 노이즈 vs. 평균 이미지.
5. 어떤 숫자가 기준선 간에 가장 일관된 어트리뷰션을 가지는가?

### 연습문제 3: 정상성 검사 구현 (코딩)

1. CIFAR-10에서 작은 CNN을 학습시키세요 (또는 사전 학습된 것을 사용).
2. 바닐라 그래디언트와 SmoothGrad에 대한 모델 무작위화 테스트를 구현하세요.
3. 학습된 어트리뷰션과 무작위화된 어트리뷰션 간의 피어슨 상관관계를 계산하세요.
4. 무작위화된 레이어 비율 대비 상관관계를 플롯하세요.
5. 두 방법 모두에서 상관관계가 거의 0으로 떨어지는가?

### 연습문제 4: 기준선 조사 (연구)

1. 텍스트 분류 모델(예: 감성 분석)에 적분 그래디언트를 사용하여:
   - 단어 임베딩의 기준선은 무엇이어야 하는가?
   - 시도: 영벡터, 패딩 토큰 임베딩, 평균 임베딩.
   - 기준선에 따라 상위 어트리뷰션 단어가 어떻게 변하는가?
2. NLP에서의 기준선 선택에 대한 1문단 권장 사항을 작성하세요.

### 연습문제 5: 비판적 분석 (개념)

Adebayo et al. (2018) 논문 "Sanity Checks for Saliency Maps"을 읽고 답하세요:
1. Guided Backpropagation이 데이터 무작위화 테스트는 통과하지만 모델 무작위화 테스트는 실패하는 이유는 무엇인가?
2. 저자들은 두 가지 유형의 무작위화를 제안합니다: 연쇄(cascade, 하향식)와 독립(independent, 각 레이어 별도). 차이점은 무엇이며, 언제 다른 결과를 줄 수 있는가?
3. 모델 의존성이나 데이터 의존성과 다른 속성을 테스트하는 새로운 정상성 검사를 제안하세요.

---

[이전: 해석 가능성 기초](./01_Interpretability_Foundations.md) | [개요](./00_Overview.md) | [다음: 클래스 활성화 매핑](./03_Class_Activation_Mapping.md)

---

**라이선스**: CC BY-NC 4.0
