[Previous: SLAM Introduction](./23_SLAM_Introduction.md) | [Next: Instance Segmentation](./26_Instance_Segmentation.md)

---

# 25. Semantic Segmentation

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain semantic segmentation and its difference from classification and detection
2. Implement Fully Convolutional Networks (FCN) for pixel-wise prediction
3. Build the U-Net architecture with skip connections for precise segmentation
4. Describe DeepLab v3+ with atrous convolution and ASPP modules
5. Evaluate segmentation models using IoU, mIoU, and pixel accuracy

---

## Table of Contents

Before the reference, read [**Theory & Principles**](#theory--principles) — segmentation as per-pixel classification, the FCN → U-Net → DeepLab architectural progression, the receptive-field problem, and the IoU / Dice loss formulations.

1. [Segmentation Overview](#1-segmentation-overview)
2. [Fully Convolutional Networks (FCN)](#2-fully-convolutional-networks-fcn)
3. [U-Net Architecture](#3-u-net-architecture)
4. [DeepLab v3+](#4-deeplab-v3)
5. [Loss Functions for Segmentation](#5-loss-functions-for-segmentation)
6. [Evaluation Metrics](#6-evaluation-metrics)
7. [Practical Implementation](#7-practical-implementation)
8. [Exercises](#8-exercises)

---

## Theory & Principles

Semantic segmentation assigns **every pixel** in an image to one of a fixed set of classes. No bounding boxes, no object instances — just "every pixel of the road is the road class, every pixel of the sky is the sky class". It is the pixel-level extreme of classification: instead of one label per image, one label per pixel.

This section covers:

- **(A) The per-pixel classification framing** — what makes segmentation fundamentally different from detection.
- **(B) Why FCN was the breakthrough** — using a CNN for dense prediction instead of classification.
- **(C) The receptive-field problem and skip connections** — U-Net's architectural answer.
- **(D) Dilated / atrous convolution** — DeepLab's answer to the same problem.
- **(E) Loss functions** — cross-entropy, dice loss, focal loss, and when each is right.
- **(F) Evaluation metrics** — IoU and mIoU, pixel accuracy, and why accuracy is misleading.

### A. Per-Pixel Classification

Semantic segmentation output shape: `(H, W, K)` logits, where `K` is the number of classes. Apply softmax along the `K` axis to get per-pixel class probabilities. The network structurally **looks like a classifier applied at every pixel** — hence "per-pixel classification".

This framing immediately suggests a problem: classification networks are designed to output one prediction per image, not one per pixel. They aggressively downsample (max-pool, strided conv) throughout the network, reducing a `224×224` image to a `1×1` feature vector at the end. For segmentation we need the output to be the same resolution as the input.

The entire architectural history of segmentation is about how to keep spatial resolution high enough to localize pixel labels while still aggregating enough global context to classify them correctly.

### B. FCN: the Breakthrough

Long, Shelhamer & Darrell (2015) introduced the Fully Convolutional Network. Two ideas:

1. **Remove the fully-connected classifier head** and replace it with a 1×1 convolution that produces `K` channels. The network now outputs a low-resolution class-probability map instead of a single vector.
2. **Upsample** the output back to input resolution via transposed convolution (or bilinear interpolation + convolution).

The architecture: take a pretrained classification network (VGG), strip the FC layers, replace with conv layers, train on segmentation data. The network now does what the human eye expects: dense pixel-wise class predictions.

The problem with vanilla FCN: by the time features reach the final classifier, spatial resolution is 32× lower than the input. Upsampling that 32× back to original resolution produces blurry segmentations without fine detail — exactly where you need it (boundaries).

### C. U-Net: Skip Connections for Resolution

U-Net (Ronneberger et al., 2015, originally for biomedical images) solves FCN's resolution loss with **symmetric encoder-decoder structure and skip connections**:

```
Input (572×572)
  │
  │  encoder (downsampling path): 4 levels of conv + pool
  │
  ▼    ─────────► skip connection ─────────┐
Level 1 (64 ch)                             │
  │                                         │
  │                                         ▼  decoder output (64 ch)
  ▼    ─────────► skip connection ──────┐    ▲
Level 2 (128 ch)                        │    │ upsample
  │                                     │    │
  │                                     ▼    │
  ▼    ─────────► skip connection ──┐   ... (same pattern at each level)
Level 3 (256 ch)                    │
  │                                 │
  │                                 │
  ▼                                 │
Bottleneck (1024 ch, 32×32)         │
```

The key insight: the encoder throws away spatial information as it goes deeper; the skip connections route that spatial information **directly** to the decoder, where it is concatenated with the upsampled features. The decoder thus has both **semantic information** (from the bottleneck, which saw the whole image) and **spatial information** (from the skip connections, which preserve fine detail).

U-Net became the template for medical imaging, satellite imagery, and many other segmentation tasks. Variants (nnU-Net, TransUNet) all share the skip-connection idea.

### D. DeepLab: Atrous Convolution Without Downsampling

DeepLab (Chen et al., 2015-2018) takes a different approach. Instead of downsampling and upsampling (U-Net style), it keeps the feature map at higher resolution and uses **atrous (dilated) convolution** to grow the receptive field:

- A regular 3×3 conv with dilation rate 1 sees a 3×3 region.
- A 3×3 conv with dilation rate 2 sees a 5×5 region but only samples 9 points (every other one).
- A 3×3 conv with dilation rate 4 sees a 9×9 region with the same 9 parameters.

By stacking atrous convs with increasing rates, you get a large effective receptive field without downsampling. The **Atrous Spatial Pyramid Pooling (ASPP)** module in DeepLab v3+ applies multiple parallel atrous convs with different dilation rates, then concatenates, capturing multi-scale context.

DeepLab v3+ adds a small decoder with skip connections on top of the atrous backbone, combining both approaches.

### E. Loss Functions

#### E.1 Cross-entropy

The default: per-pixel categorical cross-entropy. Same as classification, just applied at every pixel. Problem: **class imbalance**. In a driving scene, 60% of pixels might be road and 0.5% might be traffic signs. Cross-entropy treats every pixel equally, so the network becomes very good at road and barely learns signs.

#### E.2 Dice / IoU loss

Dice loss directly optimizes the overlap between predicted and ground-truth masks:

```
Dice(A, B) = 2 · |A ∩ B| / (|A| + |B|)
Loss = 1 - Dice
```

For binary masks: `Dice = 2 · Σ(p · g) / (Σp + Σg)` where `p`, `g` are predicted and ground-truth probabilities. Insensitive to class imbalance because it only cares about the foreground overlap, not the background. Popular in medical imaging where the class of interest (tumor) is small relative to the background.

#### E.3 Focal loss

Cross-entropy with an extra `(1 - p)^γ` factor that **downweights well-classified easy pixels**, focusing training on hard pixels. Another way to combat class imbalance, introduced by RetinaNet and popular in segmentation too.

Typical practice: combine cross-entropy with dice loss (sum or weighted average). Cross-entropy provides stable gradient; dice provides direct optimization of the metric.

### F. Evaluation Metrics

#### F.1 Pixel accuracy

Simplest: fraction of pixels correctly classified. Misleading because of class imbalance — on a driving-scene dataset you might get 95% pixel accuracy by just predicting "road everywhere" for half the classes.

#### F.2 Intersection-over-Union (IoU)

Per-class IoU:

```
IoU_c = |pred_c ∩ true_c| / |pred_c ∪ true_c|
```

For class `c`: the fraction of the union of predicted and true `c`-pixels that are in both. IoU of 1 means perfect, 0 means no overlap.

#### F.3 Mean IoU (mIoU)

Average IoU across all `K` classes, with each class weighted equally regardless of its pixel count. **This is the standard segmentation metric** on Cityscapes, ADE20K, Pascal VOC, and every benchmark. Robust to class imbalance because rare classes are weighted the same as common classes.

Fringe benefit: if mIoU improves but pixel accuracy drops, you are getting better at segmenting rare classes at the slight expense of the dominant ones — usually the right trade-off.

### From Theory to the Functions Below

- Modern libraries (PyTorch: `torchvision.models.segmentation`, `segmentation_models_pytorch`) provide pretrained FCN, U-Net, DeepLab models with one-line loading.
- OpenCV's DNN module can run exported ONNX segmentation models; the inference pipeline follows §19.
- Key hyperparameters: input size (larger = more context but slower), backbone (ResNet, EfficientNet), loss function combination (CE + Dice).
- Post-processing: argmax along class axis for final label map, optional CRF (conditional random field) for edge refinement.

---

## 1. Segmentation Overview

### 1.1 Types of Segmentation

```
Image Classification:
  Input: Image → Output: Single label
  "This is a cat"

Object Detection:
  Input: Image → Output: Bounding boxes + labels
  "Cat at (x1,y1,x2,y2)"

Semantic Segmentation:
  Input: Image → Output: Class label for EVERY pixel
  "Pixel (i,j) is cat, pixel (i,j+1) is background"

Instance Segmentation:
  Input: Image → Output: Class + instance ID for every pixel
  "Pixel (i,j) is cat #1, pixel (i,j+5) is cat #2"

Panoptic Segmentation:
  Semantic + Instance for ALL classes (stuff + things)
```

### 1.2 Applications

```
Semantic segmentation applications:

Autonomous Driving:
  Road, sidewalk, vehicles, pedestrians, sky, buildings
  Input: 1920×1080 camera → Output: per-pixel labels

Medical Imaging:
  Tumor, organ, tissue segmentation from CT/MRI
  Critical for diagnosis and surgical planning

Satellite/Aerial Imaging:
  Land use classification: forest, water, urban, agriculture

Robotics:
  Scene understanding for navigation and manipulation

Augmented Reality:
  Real-time person/background segmentation (video calls)
```

---

## 2. Fully Convolutional Networks (FCN)

### 2.1 From Classification to Segmentation

```
Key insight: Replace fully-connected layers with convolutional layers.

Classification CNN:
  Image → Conv layers → FC layers → [cat, dog, ...]
                         ↑ Destroys spatial information!

FCN:
  Image → Conv layers → 1×1 Conv → Upsample → Pixel-wise labels
                                    ↑ Preserve spatial information!
```

### 2.2 FCN Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class FCN8s(nn.Module):
    """FCN-8s: Fully Convolutional Network with 8x upsampling."""

    def __init__(self, n_classes=21):
        super().__init__()
        # Encoder (VGG-16 backbone)
        self.conv1 = self._make_block(3, 64, 2)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = self._make_block(64, 128, 2)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = self._make_block(128, 256, 3)
        self.pool3 = nn.MaxPool2d(2, 2)    # 1/8 resolution

        self.conv4 = self._make_block(256, 512, 3)
        self.pool4 = nn.MaxPool2d(2, 2)    # 1/16 resolution

        self.conv5 = self._make_block(512, 512, 3)
        self.pool5 = nn.MaxPool2d(2, 2)    # 1/32 resolution

        # FCN head (replace FC with 1x1 conv)
        self.fc6 = nn.Conv2d(512, 4096, 1)
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.score = nn.Conv2d(4096, n_classes, 1)

        # Skip connections
        self.score_pool4 = nn.Conv2d(512, n_classes, 1)
        self.score_pool3 = nn.Conv2d(256, n_classes, 1)

        # Upsampling layers
        self.upscore2 = nn.ConvTranspose2d(n_classes, n_classes, 4, stride=2, padding=1)
        self.upscore4 = nn.ConvTranspose2d(n_classes, n_classes, 4, stride=2, padding=1)
        self.upscore8 = nn.ConvTranspose2d(n_classes, n_classes, 16, stride=8, padding=4)

    def _make_block(self, in_ch, out_ch, n_convs):
        layers = []
        for i in range(n_convs):
            layers.append(nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        pool3_out = x                          # 1/8

        x = self.pool4(self.conv4(x))
        pool4_out = x                          # 1/16

        x = self.pool5(self.conv5(x))          # 1/32

        # FCN head
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.score(x)                      # 1/32, n_classes channels

        # FCN-8s: fuse pool3, pool4, and fc7
        x = self.upscore2(x)                   # 1/16
        x = x + self.score_pool4(pool4_out)    # Skip connection

        x = self.upscore4(x)                   # 1/8
        x = x + self.score_pool3(pool3_out)    # Skip connection

        x = self.upscore8(x)                   # 1/1 (original resolution)

        return x
```

---

## 3. U-Net Architecture

### 3.1 U-Net Design

```
U-Net: Encoder-Decoder with skip connections at every level.

  Encoder (contracting path)    Decoder (expanding path)
  ┌────────────────────┐       ┌────────────────────┐
  │  64 ch, 256×256    │━━━━━━▶│  64 ch, 256×256    │ → Output
  │  ↓ MaxPool         │       │  ↑ UpConv           │
  │ 128 ch, 128×128    │━━━━━━▶│ 128 ch, 128×128    │
  │  ↓ MaxPool         │       │  ↑ UpConv           │
  │ 256 ch, 64×64      │━━━━━━▶│ 256 ch, 64×64      │
  │  ↓ MaxPool         │       │  ↑ UpConv           │
  │ 512 ch, 32×32      │━━━━━━▶│ 512 ch, 32×32      │
  │  ↓ MaxPool         │       │  ↑ UpConv           │
  │ 1024 ch, 16×16     │───────┘                     │
  └────────────────────┘  Bottleneck                  │
                          ━━━━━▶ = skip connection (concatenate)
```

### 3.2 U-Net Implementation

```python
class DoubleConv(nn.Module):
    """Two consecutive conv-bn-relu blocks."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """U-Net for semantic segmentation."""

    def __init__(self, in_channels=3, n_classes=21, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder (downsampling path)
        for feat in features:
            self.encoder.append(DoubleConv(in_channels, feat))
            in_channels = feat

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder (upsampling path)
        for feat in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(feat * 2, feat, 2, stride=2)
            )
            self.decoder.append(DoubleConv(feat * 2, feat))

        # Output
        self.final = nn.Conv2d(features[0], n_classes, 1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for enc in self.encoder:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder
        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)      # Upsample
            skip = skip_connections[i // 2]

            # Handle size mismatch
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])

            x = torch.cat([skip, x], dim=1)  # Concatenate
            x = self.decoder[i + 1](x)       # Double conv

        return self.final(x)
```

---

## 4. DeepLab v3+

### 4.1 Atrous (Dilated) Convolution

```
Standard 3×3 conv: receptive field = 3×3
Atrous conv (rate=2): receptive field = 5×5 (with gaps)
Atrous conv (rate=4): receptive field = 9×9 (with gaps)

  Standard (rate=1):     Atrous (rate=2):
  ■ ■ ■                  ■ ○ ■ ○ ■
  ■ ■ ■                  ○ ○ ○ ○ ○
  ■ ■ ■                  ■ ○ ■ ○ ■
                          ○ ○ ○ ○ ○
  3×3 RF                  ■ ○ ■ ○ ■
                          5×5 RF (same parameters!)

Advantage: Larger receptive field without more parameters or downsampling.
```

### 4.2 ASPP Module

```python
class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling."""

    def __init__(self, in_channels, out_channels=256, rates=[6, 12, 18]):
        super().__init__()
        modules = []

        # 1×1 convolution
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ))

        # Atrous convolutions at different rates
        for rate in rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3,
                          padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ))

        # Global average pooling branch
        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ))

        self.branches = nn.ModuleList(modules)

        # Project concatenated features
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(rates) + 2), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        outputs = []
        for branch in self.branches[:-1]:
            outputs.append(branch(x))

        # Global pooling branch: upsample to input size
        gap = self.branches[-1](x)
        gap = F.interpolate(gap, size=x.shape[2:], mode='bilinear', align_corners=False)
        outputs.append(gap)

        x = torch.cat(outputs, dim=1)
        return self.project(x)
```

---

## 5. Loss Functions for Segmentation

### 5.1 Common Loss Functions

```python
def cross_entropy_loss(pred, target, ignore_index=255):
    """Standard cross-entropy for segmentation."""
    return F.cross_entropy(pred, target, ignore_index=ignore_index)


def dice_loss(pred, target, smooth=1.0):
    """Dice loss: good for imbalanced classes."""
    pred = F.softmax(pred, dim=1)
    n_classes = pred.shape[1]
    total_loss = 0

    for c in range(n_classes):
        pred_c = pred[:, c]
        target_c = (target == c).float()

        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()

        dice = (2 * intersection + smooth) / (union + smooth)
        total_loss += (1 - dice)

    return total_loss / n_classes


def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """Focal loss: down-weight easy examples."""
    ce = F.cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-ce)
    loss = alpha * (1 - pt) ** gamma * ce
    return loss.mean()


class CombinedLoss(nn.Module):
    """Combine CE + Dice for best results."""

    def __init__(self, ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        ce = cross_entropy_loss(pred, target)
        dl = dice_loss(pred, target)
        return self.ce_weight * ce + self.dice_weight * dl
```

---

## 6. Evaluation Metrics

### 6.1 Segmentation Metrics

```python
import numpy as np


def compute_iou(pred, target, n_classes):
    """
    Intersection over Union (IoU) per class.
    Also known as Jaccard Index.

    IoU = TP / (TP + FP + FN)
    """
    ious = []
    for c in range(n_classes):
        pred_c = (pred == c)
        target_c = (target == c)

        intersection = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()

        if union == 0:
            ious.append(float('nan'))  # Class not present
        else:
            ious.append(intersection / union)

    return ious


def mean_iou(pred, target, n_classes):
    """Mean IoU (mIoU): Average IoU across all classes."""
    ious = compute_iou(pred, target, n_classes)
    valid = [iou for iou in ious if not np.isnan(iou)]
    return np.mean(valid) if valid else 0.0


def pixel_accuracy(pred, target):
    """Overall pixel accuracy."""
    correct = (pred == target).sum().item()
    total = target.numel()
    return correct / total


def evaluate_segmentation(model, dataloader, n_classes, device='cpu'):
    """Full evaluation on a dataset."""
    total_iou = np.zeros(n_classes)
    total_count = np.zeros(n_classes)
    total_correct = 0
    total_pixels = 0

    model.eval()
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            for c in range(n_classes):
                pred_c = (preds == c)
                target_c = (targets == c)

                intersection = (pred_c & target_c).sum().item()
                union = (pred_c | target_c).sum().item()

                if union > 0:
                    total_iou[c] += intersection / union
                    total_count[c] += 1

            total_correct += (preds == targets).sum().item()
            total_pixels += targets.numel()

    # Compute mean IoU
    class_ious = []
    for c in range(n_classes):
        if total_count[c] > 0:
            class_ious.append(total_iou[c] / total_count[c])

    miou = np.mean(class_ious) if class_ious else 0.0
    pixel_acc = total_correct / total_pixels

    print(f"mIoU: {miou:.4f}")
    print(f"Pixel Accuracy: {pixel_acc:.4f}")
    return miou, pixel_acc
```

---

## 7. Practical Implementation

### 7.1 Training Pipeline

```python
def train_segmentation(model, train_loader, val_loader, n_classes,
                        epochs=50, lr=1e-3, device='cuda'):
    """Complete training pipeline for segmentation."""
    criterion = CombinedLoss(ce_weight=1.0, dice_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model = model.to(device)
    best_miou = 0

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device).long()

            outputs = model(images)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # Validation
        miou, pixel_acc = evaluate_segmentation(
            model, val_loader, n_classes, device
        )

        print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, "
              f"mIoU={miou:.4f}, PixAcc={pixel_acc:.4f}")

        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), 'best_segmentation.pth')

    return best_miou
```

---

## 8. Exercises

### Exercise 1: FCN Implementation

Build FCN from scratch:
1. Implement FCN-32s (single 32x upsampling)
2. Add skip connections for FCN-16s and FCN-8s
3. Train on Pascal VOC 2012 segmentation dataset
4. Compare FCN-32s, FCN-16s, FCN-8s: mIoU improvement from skip connections
5. Visualize segmentation predictions overlaid on images

### Exercise 2: U-Net for Medical Imaging

Build U-Net for a medical segmentation task:
1. Download a medical image dataset (e.g., lung CT, cell segmentation)
2. Implement U-Net with configurable depth
3. Train with combined CE + Dice loss
4. Evaluate with IoU and Dice score per organ/structure
5. Apply data augmentation: rotation, flipping, elastic deformation

### Exercise 3: DeepLab v3+ with ASPP

Implement DeepLab v3+ architecture:
1. Build ASPP module with rates [6, 12, 18]
2. Use ResNet-50 backbone (pretrained on ImageNet)
3. Implement encoder-decoder structure
4. Train on Cityscapes dataset (urban scene segmentation)
5. Compare with U-Net on the same dataset

### Exercise 4: Loss Function Comparison

Compare segmentation loss functions:
1. Implement: cross-entropy, dice, focal, Lovasz-softmax
2. Train the same model with each loss on imbalanced dataset
3. Measure: mIoU, per-class IoU, convergence speed
4. Create intentionally imbalanced dataset (1 rare class)
5. Show that Dice/focal loss handle imbalance better

### Exercise 5: Real-Time Segmentation

Build a real-time segmentation system:
1. Implement a lightweight model (e.g., BiSeNet or ENet)
2. Optimize for speed: reduce channels, use depthwise separable conv
3. Measure FPS on CPU and GPU
4. Apply to webcam feed for real-time segmentation
5. Compare speed-accuracy tradeoff: U-Net vs lightweight model

---

*End of Lesson 25*
