"""
Exercises for Lesson 03: Class Activation Mapping
Topic: Interpretable_AI

Solutions to practice problems from the lesson.
"""

import numpy as np


# === Exercise 1: Compute CAM Weights from GAP Layer ===
# Problem: Given feature maps from the last convolutional layer and
# the fully connected layer weights, compute Class Activation Maps.

def exercise_1():
    """Compute CAM weights from a Global Average Pooling layer."""
    print("=" * 60)
    print("Exercise 1: Compute CAM Weights from GAP Layer")
    print("=" * 60)

    # Simulate a small CNN's last conv layer output:
    # 3 feature maps (channels), each 4x4 spatial resolution
    np.random.seed(42)

    n_channels = 3
    spatial_h, spatial_w = 4, 4
    n_classes = 2

    # Feature maps from the last conv layer (shape: channels x H x W)
    feature_maps = np.array([
        [[0.5, 0.8, 0.1, 0.0],
         [0.9, 1.2, 0.3, 0.1],
         [0.2, 0.4, 0.0, 0.0],
         [0.0, 0.1, 0.0, 0.0]],

        [[0.0, 0.1, 0.6, 0.7],
         [0.0, 0.2, 0.8, 1.0],
         [0.0, 0.0, 0.5, 0.6],
         [0.0, 0.0, 0.1, 0.2]],

        [[0.3, 0.3, 0.3, 0.3],
         [0.4, 0.4, 0.4, 0.4],
         [0.3, 0.3, 0.3, 0.3],
         [0.2, 0.2, 0.2, 0.2]],
    ])

    # Fully connected weights after GAP: shape (n_classes, n_channels)
    # Each row = weights for one class
    fc_weights = np.array([
        [0.9, -0.2,  0.1],   # class 0: strongly uses channel 0
        [-0.3, 0.8,  0.3],   # class 1: strongly uses channel 1
    ])

    # Step 1: Global Average Pooling
    gap = np.mean(feature_maps, axis=(1, 2))
    print(f"\n  Feature map shapes: {feature_maps.shape}")
    print(f"  GAP output (per-channel mean): {gap}")

    # Step 2: Classification logits
    logits = fc_weights @ gap
    print(f"\n  FC weights:\n{fc_weights}")
    print(f"  Logits = FC_weights @ GAP = {logits}")
    print(f"  Predicted class: {np.argmax(logits)}")

    # Step 3: CAM = weighted sum of feature maps
    print(f"\n  Computing CAM for each class:")
    for c in range(n_classes):
        weights_c = fc_weights[c]
        cam = np.zeros((spatial_h, spatial_w))
        for k in range(n_channels):
            cam += weights_c[k] * feature_maps[k]
        cam = np.maximum(cam, 0)  # ReLU to keep only positive contributions

        print(f"\n  Class {c} CAM (weights = {weights_c}):")
        for row in cam:
            print(f"    [{', '.join(f'{v:6.3f}' for v in row)}]")
        print(f"    Peak location: {np.unravel_index(np.argmax(cam), cam.shape)}")


# === Exercise 2: Simplified Grad-CAM for a Small CNN ===
# Problem: Implement Grad-CAM by computing gradients of the class score
# with respect to feature maps, then using gradient-weighted combination.

def exercise_2():
    """Implement simplified Grad-CAM for a small CNN."""
    print("\n" + "=" * 60)
    print("Exercise 2: Simplified Grad-CAM Implementation")
    print("=" * 60)

    # Same feature maps as Exercise 1
    feature_maps = np.array([
        [[0.5, 0.8, 0.1, 0.0],
         [0.9, 1.2, 0.3, 0.1],
         [0.2, 0.4, 0.0, 0.0],
         [0.0, 0.1, 0.0, 0.0]],

        [[0.0, 0.1, 0.6, 0.7],
         [0.0, 0.2, 0.8, 1.0],
         [0.0, 0.0, 0.5, 0.6],
         [0.0, 0.0, 0.1, 0.2]],

        [[0.3, 0.3, 0.3, 0.3],
         [0.4, 0.4, 0.4, 0.4],
         [0.3, 0.3, 0.3, 0.3],
         [0.2, 0.2, 0.2, 0.2]],
    ])

    fc_weights = np.array([
        [0.9, -0.2,  0.1],
        [-0.3, 0.8,  0.3],
    ])

    target_class = 0

    # Grad-CAM key insight: for a linear classifier after GAP,
    # the gradient of y_c w.r.t. feature map A^k is:
    #   dy_c / dA^k_{ij} = w_c^k / (H * W)
    # where w_c^k is the FC weight for class c, channel k.
    #
    # The Grad-CAM weight alpha_k^c = GAP of the gradient = w_c^k / (H*W) * H * W = w_c^k
    # So for linear-after-GAP, Grad-CAM reduces to CAM!

    n_channels, H, W = feature_maps.shape

    print(f"\n  Target class: {target_class}")
    print(f"\n  Step 1: Compute gradients dy_c / dA^k")

    grad_weights = np.zeros(n_channels)
    for k in range(n_channels):
        # Gradient of class score w.r.t. each spatial location in feature map k
        gradient_map = np.full((H, W), fc_weights[target_class, k] / (H * W))

        # Grad-CAM weight: global average of the gradient
        alpha_k = np.mean(gradient_map)
        grad_weights[k] = alpha_k
        print(f"    Channel {k}: gradient = {fc_weights[target_class, k]/(H*W):.4f} "
              f"(uniform), alpha_{k} = {alpha_k:.4f}")

    print(f"\n  Step 2: Weighted combination of feature maps")
    grad_cam = np.zeros((H, W))
    for k in range(n_channels):
        grad_cam += grad_weights[k] * feature_maps[k]

    print(f"  Pre-ReLU Grad-CAM:")
    for row in grad_cam:
        print(f"    [{', '.join(f'{v:7.4f}' for v in row)}]")

    # ReLU
    grad_cam = np.maximum(grad_cam, 0)
    print(f"\n  Step 3: After ReLU:")
    for row in grad_cam:
        print(f"    [{', '.join(f'{v:7.4f}' for v in row)}]")

    # Normalize to [0, 1]
    if grad_cam.max() > 0:
        grad_cam_norm = grad_cam / grad_cam.max()
    else:
        grad_cam_norm = grad_cam

    print(f"\n  Normalized Grad-CAM [0,1]:")
    for row in grad_cam_norm:
        print(f"    [{', '.join(f'{v:5.3f}' for v in row)}]")
    print(f"    Peak at: {np.unravel_index(np.argmax(grad_cam_norm), grad_cam_norm.shape)}")


# === Exercise 3: Localization Quality Comparison ===
# Problem: Compare CAM, Grad-CAM, and a hypothetical Grad-CAM++
# in terms of localization quality using IoU with ground truth.

def exercise_3():
    """Compare localization quality across CAM variants."""
    print("\n" + "=" * 60)
    print("Exercise 3: Localization Quality Comparison")
    print("=" * 60)

    # Ground truth bounding box on an 8x8 grid (object at rows 1-4, cols 2-5)
    grid_size = 8
    ground_truth = np.zeros((grid_size, grid_size))
    ground_truth[1:5, 2:6] = 1.0  # object region

    def compute_iou(heatmap, ground_truth, threshold=0.5):
        """Compute IoU between thresholded heatmap and ground truth."""
        if heatmap.max() > 0:
            heatmap_norm = heatmap / heatmap.max()
        else:
            heatmap_norm = heatmap
        prediction = (heatmap_norm >= threshold).astype(float)
        intersection = np.sum(prediction * ground_truth)
        union = np.sum(np.clip(prediction + ground_truth, 0, 1))
        return intersection / union if union > 0 else 0.0

    # Simulate CAM-like heatmaps (low resolution, upsampled)
    # CAM: coarse localization, captures general area
    cam_heatmap = np.zeros((grid_size, grid_size))
    cam_heatmap[0:6, 1:7] = 0.5   # broad activation
    cam_heatmap[1:5, 2:6] = 0.9   # stronger in correct area
    cam_heatmap[2:4, 3:5] = 1.0   # peak

    # Grad-CAM: slightly tighter localization
    gradcam_heatmap = np.zeros((grid_size, grid_size))
    gradcam_heatmap[0:6, 1:7] = 0.3
    gradcam_heatmap[1:5, 2:6] = 0.85
    gradcam_heatmap[2:4, 3:5] = 1.0

    # Grad-CAM++: better multi-object, slightly better localization
    gradcampp_heatmap = np.zeros((grid_size, grid_size))
    gradcampp_heatmap[1:5, 2:6] = 0.8
    gradcampp_heatmap[2:4, 3:5] = 1.0

    methods = {
        "CAM": cam_heatmap,
        "Grad-CAM": gradcam_heatmap,
        "Grad-CAM++": gradcampp_heatmap,
    }

    print(f"\n  Ground truth region: rows [1:5], cols [2:6]")
    print(f"  Grid size: {grid_size}x{grid_size}")

    thresholds = [0.3, 0.5, 0.7]
    print(f"\n  {'Method':<15}", end="")
    for t in thresholds:
        print(f"  {'IoU@' + str(t):<10}", end="")
    print(f"  {'Peak loc.':<15}")
    print("  " + "-" * 60)

    for name, heatmap in methods.items():
        peak = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        print(f"  {name:<15}", end="")
        for t in thresholds:
            iou = compute_iou(heatmap, ground_truth, threshold=t)
            print(f"  {iou:<10.4f}", end="")
        print(f"  {peak}")

    print(f"\n  Analysis:")
    print(f"  - CAM: Broadest activation, lowest IoU at strict thresholds.")
    print(f"    Limited by GAP requirement (needs specific architecture).")
    print(f"  - Grad-CAM: Architecture-agnostic, tighter localization.")
    print(f"    Uses gradients instead of FC weights.")
    print(f"  - Grad-CAM++: Pixel-wise weighting of gradients gives")
    print(f"    more precise localization, especially for multiple objects.")


# === Exercise 4: Resolution vs Specificity Tradeoff ===
# Problem: Analyze how choosing different convolutional layers for
# CAM/Grad-CAM affects the resolution-specificity tradeoff.

def exercise_4():
    """Analyze resolution-vs-specificity tradeoff across layers."""
    print("\n" + "=" * 60)
    print("Exercise 4: Resolution vs Specificity Tradeoff")
    print("=" * 60)

    # Simulate feature maps at different layers of a CNN
    # Earlier layers: higher resolution, lower semantic specificity
    # Later layers: lower resolution, higher semantic specificity

    layers = [
        {
            "name": "Layer 1 (early conv)",
            "spatial_size": 16,
            "receptive_field": 3,
            "n_channels": 8,
            "semantic_level": "edges, textures",
            "specificity": 0.3,
        },
        {
            "name": "Layer 3 (mid conv)",
            "spatial_size": 8,
            "receptive_field": 11,
            "n_channels": 16,
            "semantic_level": "parts, patterns",
            "specificity": 0.6,
        },
        {
            "name": "Layer 5 (late conv)",
            "spatial_size": 4,
            "receptive_field": 27,
            "n_channels": 32,
            "semantic_level": "objects, scenes",
            "specificity": 0.9,
        },
    ]

    print(f"\n  Tradeoff: extracting Grad-CAM at different network depths\n")
    print(f"  {'Layer':<25} {'Spatial':<10} {'RF':<6} {'Channels':<10} "
          f"{'Semantics':<20} {'Specificity':<12}")
    print("  " + "-" * 85)

    for layer in layers:
        print(f"  {layer['name']:<25} {layer['spatial_size']:<10} "
              f"{layer['receptive_field']:<6} {layer['n_channels']:<10} "
              f"{layer['semantic_level']:<20} {layer['specificity']:<12.1f}")

    # Simulate heatmap quality metrics at each layer
    print(f"\n  Simulated Grad-CAM quality at each layer:")
    print(f"  {'Layer':<25} {'Resolution':<12} {'Class-Discr.':<15} "
          f"{'Localization':<14} {'Interpretable':<14}")
    print("  " + "-" * 80)

    metrics = [
        ("Layer 1 (early conv)", "High (16x16)", 0.35, 0.80, 0.25),
        ("Layer 3 (mid conv)",   "Med (8x8)",    0.65, 0.70, 0.60),
        ("Layer 5 (late conv)",  "Low (4x4)",    0.90, 0.55, 0.85),
    ]

    for name, resolution, class_disc, localization, interpretable in metrics:
        print(f"  {name:<25} {resolution:<12} {class_disc:<15.2f} "
              f"{localization:<14.2f} {interpretable:<14.2f}")

    print(f"\n  Key insights:")
    print(f"  - Early layers: Fine spatial detail but activations respond to")
    print(f"    low-level features (edges) not specific to any class.")
    print(f"  - Late layers: Highly class-discriminative but coarse spatially.")
    print(f"    Grad-CAM on the last conv layer is standard because it")
    print(f"    maximizes semantic meaning while providing enough spatial info.")
    print(f"  - Middle layers: A compromise, sometimes useful for detecting")
    print(f"    object parts rather than entire objects.")
    print(f"  - Upsampling (bilinear interpolation) of late-layer CAMs partially")
    print(f"    mitigates the resolution limitation but cannot recover lost detail.")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
