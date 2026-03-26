"""
Exercise Solutions for Lesson 25: Semantic Segmentation
Computer Vision - FCN, U-Net, Atrous Convolution, Segmentation Metrics

Topics covered:
- Pixel-wise classification and label maps
- Fully Convolutional Network (FCN) forward pass simulation
- U-Net encoder-decoder with skip connections
- Atrous (dilated) convolution
- Segmentation loss functions (cross-entropy, Dice)
- Evaluation metrics (IoU, mIoU, pixel accuracy)
"""

import numpy as np


# =============================================================================
# Helper: Synthetic segmentation data
# =============================================================================

def generate_segmentation_scene(h=64, w=64, n_classes=4):
    """
    Generate a synthetic image and its ground-truth segmentation map.

    Classes: 0=background, 1=rectangle, 2=circle, 3=triangle.

    Returns:
        (image, label_map)  image shape (h,w), label_map shape (h,w)
    """
    np.random.seed(42)
    image = np.random.randint(20, 50, (h, w), dtype=np.uint8)
    label_map = np.zeros((h, w), dtype=np.int32)

    # Class 1: rectangle
    r_y, r_x = 10, 8
    r_h, r_w = 20, 25
    image[r_y:r_y+r_h, r_x:r_x+r_w] = np.random.randint(140, 180)
    label_map[r_y:r_y+r_h, r_x:r_x+r_w] = 1

    # Class 2: circle
    cy, cx, radius = 40, 45, 10
    yy, xx = np.ogrid[:h, :w]
    circle_mask = ((xx - cx)**2 + (yy - cy)**2) <= radius**2
    image[circle_mask] = np.random.randint(80, 120)
    label_map[circle_mask] = 2

    # Class 3: triangle (upper-right)
    for row in range(5, 25):
        col_start = 45
        col_end = min(45 + (row - 5), w)
        if col_start < w:
            image[row, col_start:col_end] = np.random.randint(180, 220)
            label_map[row, col_start:col_end] = 3

    return image, label_map


def softmax_2d(logits):
    """Apply softmax over the class axis (axis=0) of a (C, H, W) array."""
    exp = np.exp(logits - logits.max(axis=0, keepdims=True))
    return exp / exp.sum(axis=0, keepdims=True)


# =============================================================================
# Exercise 1: Pixel-Wise Classification Basics
# =============================================================================

def exercise_1_pixel_classification():
    """
    Demonstrate pixel-wise classification fundamentals.

    Steps:
    1. Generate a multi-class label map
    2. Simulate per-pixel class logits
    3. Apply argmax to produce predicted labels
    4. Compute confusion matrix

    Returns:
        (pred_labels, confusion_matrix)
    """
    np.random.seed(42)
    h, w = 64, 64
    n_classes = 4
    image, gt_labels = generate_segmentation_scene(h, w, n_classes)

    print("Pixel-Wise Classification Basics")
    print(f"  Image size: {w}x{h}")
    print(f"  Classes: {n_classes}")
    print("=" * 60)

    # Simulate noisy logits that roughly follow ground truth
    logits = np.random.randn(n_classes, h, w).astype(np.float64) * 0.5
    for c in range(n_classes):
        logits[c][gt_labels == c] += 2.5  # Boost correct class

    # Predict via argmax
    probs = softmax_2d(logits)
    pred_labels = np.argmax(probs, axis=0)

    # Confusion matrix
    confusion = np.zeros((n_classes, n_classes), dtype=np.int32)
    for c_true in range(n_classes):
        for c_pred in range(n_classes):
            confusion[c_true, c_pred] = np.sum(
                (gt_labels == c_true) & (pred_labels == c_pred)
            )

    print("\n  Confusion Matrix (rows=GT, cols=Pred):")
    header = "       " + "".join(f"  C{c}" for c in range(n_classes))
    print(f"  {header}")
    for c_true in range(n_classes):
        row_str = "  ".join(f"{confusion[c_true, c_pred]:4d}"
                            for c_pred in range(n_classes))
        print(f"    C{c_true}  {row_str}")

    # Per-class accuracy
    print("\n  Per-class accuracy:")
    for c in range(n_classes):
        total = confusion[c].sum()
        correct = confusion[c, c]
        acc = correct / total if total > 0 else 0.0
        print(f"    Class {c}: {correct}/{total} = {acc:.3f}")

    overall = np.trace(confusion) / confusion.sum()
    print(f"\n  Overall pixel accuracy: {overall:.4f}")

    return pred_labels, confusion


# =============================================================================
# Exercise 2: FCN Forward Pass Simulation
# =============================================================================

def exercise_2_fcn_forward():
    """
    Simulate an FCN forward pass using numpy convolutions and upsampling.

    Steps:
    1. Encoder: successive convolution + downsampling
    2. 1x1 convolution for class scores at coarse resolution
    3. Bilinear upsampling back to original resolution
    4. Skip connections from encoder stages

    Returns:
        (coarse_pred, fine_pred)
    """
    np.random.seed(42)
    h, w = 64, 64
    n_classes = 4
    image, gt_labels = generate_segmentation_scene(h, w, n_classes)
    image_f = image.astype(np.float64) / 255.0

    print("FCN Forward Pass Simulation")
    print(f"  Input: {w}x{h}")
    print("=" * 60)

    def conv3x3(feat, kernel):
        """Simple 3x3 convolution with zero padding."""
        fh, fw = feat.shape
        out = np.zeros_like(feat)
        padded = np.pad(feat, 1, mode='constant')
        for i in range(fh):
            for j in range(fw):
                out[i, j] = np.sum(padded[i:i+3, j:j+3] * kernel)
        return out

    def downsample_2x(feat):
        """2x downsampling via averaging."""
        fh, fw = feat.shape
        oh, ow = fh // 2, fw // 2
        out = np.zeros((oh, ow), dtype=np.float64)
        for i in range(oh):
            for j in range(ow):
                out[i, j] = np.mean(feat[2*i:2*i+2, 2*j:2*j+2])
        return out

    def upsample_2x(feat):
        """2x upsampling via nearest-neighbor."""
        fh, fw = feat.shape
        out = np.zeros((fh * 2, fw * 2), dtype=np.float64)
        for i in range(fh):
            for j in range(fw):
                out[2*i:2*i+2, 2*j:2*j+2] = feat[i, j]
        return out

    # Encoder stage 1: conv + pool -> 32x32
    k1 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float64) / 8
    feat1 = np.maximum(conv3x3(image_f, k1), 0)  # ReLU
    pool1 = downsample_2x(feat1)
    print(f"  Stage 1: {feat1.shape} -> pool {pool1.shape}")

    # Encoder stage 2: conv + pool -> 16x16
    k2 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float64) / 16
    feat2 = np.maximum(conv3x3(pool1, k2), 0)
    pool2 = downsample_2x(feat2)
    print(f"  Stage 2: {feat2.shape} -> pool {pool2.shape}")

    # Encoder stage 3: conv + pool -> 8x8
    k3 = np.ones((3, 3), dtype=np.float64) / 9
    feat3 = np.maximum(conv3x3(pool2, k3), 0)
    pool3 = downsample_2x(feat3)
    print(f"  Stage 3: {feat3.shape} -> pool {pool3.shape}")

    # 1x1 "convolution" for class scores at 8x8
    coarse_scores = np.zeros((n_classes, 8, 8), dtype=np.float64)
    for c in range(n_classes):
        weight = np.random.randn() * 0.5
        bias = -0.5 if c > 0 else 0.5
        coarse_scores[c] = pool3 * weight + bias

    # Boost scores using downsampled GT (simulating a trained network)
    gt_small = gt_labels[::8, ::8][:8, :8]
    for c in range(n_classes):
        coarse_scores[c][gt_small == c] += 2.0

    coarse_pred = np.argmax(coarse_scores, axis=0)
    print(f"\n  Coarse prediction shape: {coarse_pred.shape}")

    # FCN-8s style: upsample + skip connections
    # Upsample coarse 8x8 -> 16x16
    up1 = np.stack([upsample_2x(coarse_scores[c]) for c in range(n_classes)])
    # Add skip from stage 2 (16x16)
    skip2_scores = np.zeros((n_classes, 16, 16), dtype=np.float64)
    gt_16 = gt_labels[::4, ::4][:16, :16]
    for c in range(n_classes):
        skip2_scores[c] = pool2[:16, :16] * np.random.randn() * 0.3
        skip2_scores[c][gt_16 == c] += 1.0
    fused1 = up1[:, :16, :16] + skip2_scores

    # Upsample 16x16 -> 32x32
    up2 = np.stack([upsample_2x(fused1[c]) for c in range(n_classes)])
    # Add skip from stage 1 (32x32)
    skip1_scores = np.zeros((n_classes, 32, 32), dtype=np.float64)
    gt_32 = gt_labels[::2, ::2][:32, :32]
    for c in range(n_classes):
        skip1_scores[c] = pool1[:32, :32] * np.random.randn() * 0.2
        skip1_scores[c][gt_32 == c] += 0.5
    fused2 = up2[:, :32, :32] + skip1_scores

    # Final upsample 32x32 -> 64x64
    fine_scores = np.stack([upsample_2x(fused2[c]) for c in range(n_classes)])
    fine_pred = np.argmax(fine_scores, axis=0)
    print(f"  Fine prediction shape: {fine_pred.shape}")

    # Compare coarse vs fine
    coarse_up = np.zeros((h, w), dtype=np.int32)
    for i in range(h):
        for j in range(w):
            coarse_up[i, j] = coarse_pred[min(i // 8, 7), min(j // 8, 7)]

    acc_coarse = np.mean(coarse_up == gt_labels)
    acc_fine = np.mean(fine_pred == gt_labels)
    print(f"\n  Coarse (upsampled) accuracy: {acc_coarse:.4f}")
    print(f"  Fine (skip connections) accuracy: {acc_fine:.4f}")
    print(f"  Improvement from skips: {acc_fine - acc_coarse:+.4f}")

    return coarse_pred, fine_pred


# =============================================================================
# Exercise 3: U-Net Encoder-Decoder with Skip Connections
# =============================================================================

def exercise_3_unet_encoder_decoder():
    """
    Simulate U-Net encoder-decoder processing with skip connections.

    Steps:
    1. Encoder path: successive downsampling with feature extraction
    2. Bottleneck processing
    3. Decoder path: upsampling + concatenation with encoder features
    4. Compare with and without skip connections

    Returns:
        (pred_with_skip, pred_without_skip)
    """
    np.random.seed(42)
    h, w = 64, 64
    n_classes = 4
    image, gt_labels = generate_segmentation_scene(h, w, n_classes)
    image_f = image.astype(np.float64) / 255.0

    print("U-Net Encoder-Decoder Simulation")
    print(f"  Input: {w}x{h}")
    print("=" * 60)

    def pool_2x(feat):
        oh, ow = feat.shape[0] // 2, feat.shape[1] // 2
        out = np.zeros((oh, ow), dtype=np.float64)
        for i in range(oh):
            for j in range(ow):
                out[i, j] = np.max(feat[2*i:2*i+2, 2*j:2*j+2])
        return out

    def unpool_2x(feat, target_h, target_w):
        fh, fw = feat.shape
        out = np.zeros((target_h, target_w), dtype=np.float64)
        for i in range(fh):
            for j in range(fw):
                i2, j2 = 2 * i, 2 * j
                if i2 < target_h and j2 < target_w:
                    out[i2, j2] = feat[i, j]
                    if i2 + 1 < target_h:
                        out[i2+1, j2] = feat[i, j]
                    if j2 + 1 < target_w:
                        out[i2, j2+1] = feat[i, j]
                    if i2 + 1 < target_h and j2 + 1 < target_w:
                        out[i2+1, j2+1] = feat[i, j]
        return out

    # Encoder
    enc1 = np.maximum(image_f + np.random.randn(h, w) * 0.1, 0)  # 64x64
    enc1_pooled = pool_2x(enc1)                                     # 32x32

    enc2 = np.maximum(enc1_pooled + np.random.randn(32, 32) * 0.1, 0)
    enc2_pooled = pool_2x(enc2)                                     # 16x16

    enc3 = np.maximum(enc2_pooled + np.random.randn(16, 16) * 0.1, 0)
    enc3_pooled = pool_2x(enc3)                                     # 8x8

    bottleneck = enc3_pooled + np.random.randn(8, 8) * 0.05

    print("  Encoder path:")
    print(f"    Level 1: {enc1.shape} -> pool {enc1_pooled.shape}")
    print(f"    Level 2: {enc2.shape} -> pool {enc2_pooled.shape}")
    print(f"    Level 3: {enc3.shape} -> pool {enc3_pooled.shape}")
    print(f"    Bottleneck: {bottleneck.shape}")

    # Decoder WITH skip connections
    dec3 = unpool_2x(bottleneck, 16, 16)
    dec3_skip = (dec3 + enc3) / 2  # Simulate concat + conv as average

    dec2 = unpool_2x(dec3_skip, 32, 32)
    dec2_skip = (dec2 + enc2) / 2

    dec1 = unpool_2x(dec2_skip, 64, 64)
    dec1_skip = (dec1 + enc1) / 2

    # Final classification with skip
    scores_skip = np.zeros((n_classes, h, w), dtype=np.float64)
    for c in range(n_classes):
        weight = np.random.randn() * 0.3
        scores_skip[c] = dec1_skip * weight + np.random.randn() * 0.1
        scores_skip[c][gt_labels == c] += 2.0
    pred_with_skip = np.argmax(scores_skip, axis=0)

    # Decoder WITHOUT skip connections
    dec3_no = unpool_2x(bottleneck, 16, 16)
    dec2_no = unpool_2x(dec3_no, 32, 32)
    dec1_no = unpool_2x(dec2_no, 64, 64)

    scores_no = np.zeros((n_classes, h, w), dtype=np.float64)
    for c in range(n_classes):
        weight = np.random.randn() * 0.3
        scores_no[c] = dec1_no * weight + np.random.randn() * 0.1
        scores_no[c][gt_labels == c] += 1.5  # Weaker signal without skips
    pred_without_skip = np.argmax(scores_no, axis=0)

    acc_skip = np.mean(pred_with_skip == gt_labels)
    acc_no_skip = np.mean(pred_without_skip == gt_labels)

    print(f"\n  Accuracy with skip connections:    {acc_skip:.4f}")
    print(f"  Accuracy without skip connections: {acc_no_skip:.4f}")
    print(f"  Improvement from skips: {acc_skip - acc_no_skip:+.4f}")

    # Boundary analysis
    boundary = np.zeros((h, w), dtype=bool)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            neighbors = gt_labels[i-1:i+2, j-1:j+2]
            if neighbors.min() != neighbors.max():
                boundary[i, j] = True
    n_boundary = boundary.sum()
    if n_boundary > 0:
        b_acc_skip = np.mean(pred_with_skip[boundary] == gt_labels[boundary])
        b_acc_no = np.mean(pred_without_skip[boundary] == gt_labels[boundary])
        print(f"\n  Boundary pixels: {n_boundary}")
        print(f"  Boundary accuracy (skip):    {b_acc_skip:.4f}")
        print(f"  Boundary accuracy (no skip): {b_acc_no:.4f}")

    return pred_with_skip, pred_without_skip


# =============================================================================
# Exercise 4: Atrous (Dilated) Convolution
# =============================================================================

def exercise_4_atrous_convolution():
    """
    Implement and compare standard vs atrous (dilated) convolutions.

    Demonstrates:
    1. Standard 3x3 convolution (rate=1)
    2. Dilated convolution at rates 2, 4, 6
    3. ASPP-style multi-scale feature extraction
    4. Receptive field analysis

    Returns:
        dict of rate -> feature_map
    """
    np.random.seed(42)
    h, w = 64, 64
    image, _ = generate_segmentation_scene(h, w)
    image_f = image.astype(np.float64) / 255.0

    print("Atrous (Dilated) Convolution")
    print(f"  Input: {w}x{h}")
    print("=" * 60)

    # Base 3x3 kernel (edge detector)
    kernel = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ], dtype=np.float64) / 8

    def dilated_conv(feat, kernel, rate=1):
        """
        Apply dilated convolution with given rate.
        rate=1 is standard convolution.
        Effective kernel size = 3 + (3-1)*(rate-1) = 2*rate + 1
        """
        fh, fw = feat.shape
        kh, kw = kernel.shape
        eff_kh = kh + (kh - 1) * (rate - 1)
        eff_kw = kw + (kw - 1) * (rate - 1)
        pad_h = eff_kh // 2
        pad_w = eff_kw // 2

        padded = np.pad(feat, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        out = np.zeros_like(feat)

        for i in range(fh):
            for j in range(fw):
                val = 0.0
                for ki in range(kh):
                    for kj in range(kw):
                        pi = i + pad_h + (ki - kh // 2) * rate
                        pj = j + pad_w + (kj - kw // 2) * rate
                        if 0 <= pi < padded.shape[0] and 0 <= pj < padded.shape[1]:
                            val += padded[pi, pj] * kernel[ki, kj]
                out[i, j] = val
        return out

    results = {}
    rates = [1, 2, 4, 6]

    for rate in rates:
        feat = dilated_conv(image_f, kernel, rate=rate)
        feat = np.maximum(feat, 0)  # ReLU
        results[rate] = feat

        eff_size = 2 * rate + 1
        energy = np.sum(feat ** 2)
        nonzero = np.sum(feat > 0.01)
        print(f"\n  Rate={rate}:")
        print(f"    Effective receptive field: {eff_size}x{eff_size}")
        print(f"    Feature energy: {energy:.4f}")
        print(f"    Active pixels: {nonzero} / {h*w} "
              f"({100*nonzero/(h*w):.1f}%)")

    # ASPP-style: concatenate features from all rates
    print("\n  ASPP Multi-Scale Fusion:")
    aspp_features = np.stack([results[r] for r in rates], axis=0)  # (4, h, w)
    aspp_mean = aspp_features.mean(axis=0)
    aspp_max = aspp_features.max(axis=0)

    print(f"    Combined shape: {aspp_features.shape}")
    print(f"    Mean-pooled range: [{aspp_mean.min():.4f}, {aspp_mean.max():.4f}]")
    print(f"    Max-pooled range: [{aspp_max.min():.4f}, {aspp_max.max():.4f}]")

    # Show which rate dominates at each pixel
    dominant_rate = np.argmax(aspp_features, axis=0)
    for idx, rate in enumerate(rates):
        count = np.sum(dominant_rate == idx)
        print(f"    Dominant rate={rate}: {count} pixels "
              f"({100*count/(h*w):.1f}%)")

    return results


# =============================================================================
# Exercise 5: Segmentation Loss Functions
# =============================================================================

def exercise_5_loss_functions():
    """
    Implement and compare segmentation loss functions.

    Loss functions:
    1. Cross-entropy loss (per-pixel)
    2. Dice loss (region overlap)
    3. Focal loss (hard example mining)
    4. Combined CE + Dice

    Returns:
        dict of loss_name -> loss_value
    """
    np.random.seed(42)
    h, w = 64, 64
    n_classes = 4
    _, gt_labels = generate_segmentation_scene(h, w, n_classes)

    print("Segmentation Loss Functions")
    print(f"  Image size: {w}x{h}, Classes: {n_classes}")
    print("=" * 60)

    # Create predictions with varying quality
    def make_logits(quality):
        """Generate logits with given quality level (0 to 1)."""
        logits = np.random.randn(n_classes, h, w) * 0.5
        for c in range(n_classes):
            logits[c][gt_labels == c] += quality * 3.0
        return logits

    def cross_entropy_loss(logits, labels):
        """Per-pixel cross-entropy loss."""
        probs = softmax_2d(logits)
        loss = 0.0
        n_pixels = h * w
        for i in range(h):
            for j in range(w):
                c = labels[i, j]
                prob = max(probs[c, i, j], 1e-10)
                loss -= np.log(prob)
        return loss / n_pixels

    def dice_loss(logits, labels, smooth=1.0):
        """Dice loss averaged over classes."""
        probs = softmax_2d(logits)
        total_dice = 0.0
        valid_classes = 0
        for c in range(n_classes):
            pred_c = probs[c]
            gt_c = (labels == c).astype(np.float64)
            intersection = np.sum(pred_c * gt_c)
            union = np.sum(pred_c) + np.sum(gt_c)
            dice = (2 * intersection + smooth) / (union + smooth)
            total_dice += (1 - dice)
            valid_classes += 1
        return total_dice / valid_classes

    def focal_loss(logits, labels, alpha=0.25, gamma=2.0):
        """Focal loss for handling class imbalance."""
        probs = softmax_2d(logits)
        loss = 0.0
        n_pixels = h * w
        for i in range(h):
            for j in range(w):
                c = labels[i, j]
                pt = max(probs[c, i, j], 1e-10)
                loss -= alpha * (1 - pt) ** gamma * np.log(pt)
        return loss / n_pixels

    # Test at different quality levels
    qualities = [0.2, 0.5, 0.8, 1.0]
    losses_by_quality = {}

    print(f"\n  {'Quality':>8} | {'CE':>8} | {'Dice':>8} | {'Focal':>8} | {'CE+Dice':>8}")
    print(f"  {'-'*50}")

    for q in qualities:
        logits = make_logits(q)
        ce = cross_entropy_loss(logits, gt_labels)
        dl = dice_loss(logits, gt_labels)
        fl = focal_loss(logits, gt_labels)
        combined = ce + dl

        losses_by_quality[q] = {'CE': ce, 'Dice': dl, 'Focal': fl, 'Combined': combined}
        print(f"  {q:>8.1f} | {ce:>8.4f} | {dl:>8.4f} | {fl:>8.4f} | {combined:>8.4f}")

    # Class imbalance analysis
    print("\n  Class distribution:")
    for c in range(n_classes):
        count = np.sum(gt_labels == c)
        print(f"    Class {c}: {count} pixels ({100*count/(h*w):.1f}%)")

    # Compare losses on imbalanced prediction (class 0 over-predicted)
    print("\n  Imbalanced prediction test:")
    biased_logits = np.zeros((n_classes, h, w), dtype=np.float64)
    biased_logits[0] = 3.0  # Always predict class 0
    ce_biased = cross_entropy_loss(biased_logits, gt_labels)
    dice_biased = dice_loss(biased_logits, gt_labels)
    focal_biased = focal_loss(biased_logits, gt_labels)
    print(f"    CE (all class 0):    {ce_biased:.4f}")
    print(f"    Dice (all class 0):  {dice_biased:.4f}")
    print(f"    Focal (all class 0): {focal_biased:.4f}")

    return losses_by_quality


# =============================================================================
# Exercise 6: Segmentation Evaluation Metrics
# =============================================================================

def exercise_6_evaluation_metrics():
    """
    Compute standard segmentation evaluation metrics.

    Metrics:
    1. Pixel accuracy
    2. Mean pixel accuracy (per-class)
    3. IoU per class
    4. Mean IoU (mIoU)
    5. Frequency-weighted IoU

    Returns:
        dict of metric_name -> value
    """
    np.random.seed(42)
    h, w = 64, 64
    n_classes = 4
    _, gt_labels = generate_segmentation_scene(h, w, n_classes)

    # Generate predictions with some errors
    logits = np.random.randn(n_classes, h, w) * 0.5
    for c in range(n_classes):
        logits[c][gt_labels == c] += 2.5
    pred_labels = np.argmax(logits, axis=0)

    print("Segmentation Evaluation Metrics")
    print(f"  Image size: {w}x{h}, Classes: {n_classes}")
    print("=" * 60)

    # 1. Pixel accuracy
    pixel_acc = np.sum(pred_labels == gt_labels) / (h * w)

    # 2. Mean pixel accuracy
    class_accs = []
    for c in range(n_classes):
        mask = gt_labels == c
        if mask.sum() > 0:
            class_accs.append(np.sum(pred_labels[mask] == c) / mask.sum())
    mean_pixel_acc = np.mean(class_accs)

    # 3. IoU per class
    ious = []
    for c in range(n_classes):
        pred_c = pred_labels == c
        gt_c = gt_labels == c
        intersection = np.sum(pred_c & gt_c)
        union = np.sum(pred_c | gt_c)
        iou = intersection / union if union > 0 else 0.0
        ious.append(iou)

    # 4. Mean IoU
    valid_ious = [iou for iou in ious if iou > 0]
    miou = np.mean(valid_ious) if valid_ious else 0.0

    # 5. Frequency-weighted IoU
    freq = np.array([np.sum(gt_labels == c) for c in range(n_classes)],
                    dtype=np.float64)
    freq /= freq.sum()
    fwiou = np.sum(freq * np.array(ious))

    # 6. Dice score per class
    dice_scores = []
    for c in range(n_classes):
        pred_c = pred_labels == c
        gt_c = gt_labels == c
        intersection = np.sum(pred_c & gt_c)
        dice = 2 * intersection / (pred_c.sum() + gt_c.sum()) if (pred_c.sum() + gt_c.sum()) > 0 else 0.0
        dice_scores.append(dice)

    print(f"\n  Overall Pixel Accuracy:     {pixel_acc:.4f}")
    print(f"  Mean Pixel Accuracy:        {mean_pixel_acc:.4f}")
    print(f"  Mean IoU (mIoU):            {miou:.4f}")
    print(f"  Frequency-Weighted IoU:     {fwiou:.4f}")

    print(f"\n  Per-Class Metrics:")
    print(f"    {'Class':>6} | {'Pixels':>6} | {'IoU':>6} | {'Dice':>6} | {'Accuracy':>8}")
    print(f"    {'-'*45}")
    for c in range(n_classes):
        n_pixels = np.sum(gt_labels == c)
        print(f"    {c:>6} | {n_pixels:>6} | {ious[c]:>6.4f} | "
              f"{dice_scores[c]:>6.4f} | {class_accs[c]:>8.4f}")

    metrics = {
        'pixel_accuracy': pixel_acc,
        'mean_pixel_accuracy': mean_pixel_acc,
        'miou': miou,
        'fwiou': fwiou,
        'per_class_iou': ious,
        'per_class_dice': dice_scores,
    }

    return metrics


# =============================================================================
# Exercise 7: Real-Time Segmentation Simulation
# =============================================================================

def exercise_7_realtime_segmentation():
    """
    Simulate a real-time segmentation pipeline comparing model sizes.

    Compares:
    1. Large model (more convolutions, higher accuracy)
    2. Medium model (balanced)
    3. Lightweight model (fewer ops, faster)

    Returns:
        performance metrics dict
    """
    import time

    np.random.seed(42)

    print("Real-Time Segmentation Simulation")
    print("=" * 60)

    def simulate_model(image, n_classes, model_size='medium'):
        """Simulate segmentation with different model complexities."""
        h, w = image.shape
        image_f = image.astype(np.float64) / 255.0

        if model_size == 'large':
            n_convs = 6
            n_channels = 4
        elif model_size == 'medium':
            n_convs = 3
            n_channels = 2
        else:  # light
            n_convs = 1
            n_channels = 1

        # Simulate convolutions
        feat = image_f.copy()
        kernel = np.ones((3, 3), dtype=np.float64) / 9
        for _ in range(n_convs):
            padded = np.pad(feat, 1, mode='constant')
            out = np.zeros_like(feat)
            for ci in range(h):
                for cj in range(w):
                    out[ci, cj] = np.sum(padded[ci:ci+3, cj:cj+3] * kernel)
            feat = np.maximum(out, 0)

        # Generate class scores
        scores = np.random.randn(n_classes, h, w) * 0.3
        for c in range(n_classes):
            scores[c] += feat * np.random.randn() * 0.5

        return np.argmax(scores, axis=0)

    resolutions = [
        (32, 32, "32x32"),
        (48, 48, "48x48"),
        (64, 64, "64x64"),
    ]
    model_sizes = ['light', 'medium', 'large']
    n_classes = 4
    results = {}

    for res_h, res_w, label in resolutions:
        image = np.random.randint(30, 200, (res_h, res_w), dtype=np.uint8)
        # Add some structure
        image[res_h//4:res_h//2, res_w//4:res_w//2] = 180
        image[res_h//2:3*res_h//4, res_w//2:3*res_w//4] = 100

        for size in model_sizes:
            start = time.perf_counter()
            n_runs = 3
            for _ in range(n_runs):
                pred = simulate_model(image, n_classes, size)
            elapsed = (time.perf_counter() - start) / n_runs

            fps = 1.0 / elapsed if elapsed > 0 else float('inf')
            key = f"{label}_{size}"
            results[key] = {
                'resolution': label,
                'model': size,
                'time_ms': elapsed * 1000,
                'fps': fps,
            }

    # Print results
    print(f"\n  {'Resolution':>10} | {'Model':>8} | {'Time (ms)':>10} | {'FPS':>8}")
    print(f"  {'-'*45}")
    for key in sorted(results.keys()):
        r = results[key]
        print(f"  {r['resolution']:>10} | {r['model']:>8} | "
              f"{r['time_ms']:>10.1f} | {r['fps']:>8.1f}")

    # Speed-accuracy tradeoff summary
    print("\n  Tradeoff Summary:")
    print("    Larger models -> higher accuracy but slower")
    print("    Depthwise separable convolutions reduce computation")
    print("    Knowledge distillation: train small model from large")

    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n>>> Exercise 1: Pixel-Wise Classification Basics")
    exercise_1_pixel_classification()

    print("\n>>> Exercise 2: FCN Forward Pass Simulation")
    exercise_2_fcn_forward()

    print("\n>>> Exercise 3: U-Net Encoder-Decoder with Skip Connections")
    exercise_3_unet_encoder_decoder()

    print("\n>>> Exercise 4: Atrous (Dilated) Convolution")
    exercise_4_atrous_convolution()

    print("\n>>> Exercise 5: Segmentation Loss Functions")
    exercise_5_loss_functions()

    print("\n>>> Exercise 6: Segmentation Evaluation Metrics")
    exercise_6_evaluation_metrics()

    print("\n>>> Exercise 7: Real-Time Segmentation Simulation")
    exercise_7_realtime_segmentation()

    print("\nAll exercises completed successfully.")
