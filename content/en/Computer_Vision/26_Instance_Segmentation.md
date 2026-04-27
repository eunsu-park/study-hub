[Previous: Semantic Segmentation](./25_Semantic_Segmentation.md) | [Next: Panoptic Segmentation](./27_Panoptic_Segmentation.md)

---

# 26. Instance Segmentation

## Learning Objectives

After completing this lesson, you will be able to:

1. Distinguish instance segmentation from semantic segmentation
2. Implement Mask R-CNN for simultaneous detection and segmentation
3. Build anchor-free instance segmentation methods (YOLACT, SOLOv2)
4. Evaluate with COCO-style AP metrics for mask quality
5. Apply instance segmentation to real-world counting and analysis tasks

---

## Table of Contents

1. [Instance vs Semantic Segmentation](#1-instance-vs-semantic-segmentation)
2. [Mask R-CNN Architecture](#2-mask-r-cnn-architecture)
3. [YOLACT: Real-Time Instance Segmentation](#3-yolact-real-time-instance-segmentation)
4. [Anchor-Free Methods (SOLO, SOLOv2)](#4-anchor-free-methods-solo-solov2)
5. [Training and Data Preparation](#5-training-and-data-preparation)
6. [Evaluation Metrics (COCO AP)](#6-evaluation-metrics-coco-ap)
7. [Practical Applications](#7-practical-applications)
8. [Exercises](#8-exercises)

---

## 1. Instance vs Semantic Segmentation

### Theory: The Instance Segmentation Problem

Output: for every object in the image, a **bounding box**, **class label**, and **binary mask**. This is a set-valued output (variable-length) where each element has mixed types (box, label, mask).

Why it is harder than alternatives:

- **Harder than detection**: detection just needs to find the object; instance segmentation needs the object's exact shape.
- **Harder than semantic segmentation**: semantic segmentation merges all cars into one "car" region; instance segmentation must separate each car from neighboring cars, even when they touch or overlap.

The two architectural approaches:

- **Two-stage**: first detect, then segment within each detection (Mask R-CNN).
- **One-stage**: directly predict per-pixel instance labels (SOLO) or per-pixel prototype masks + per-instance coefficients (YOLACT).

### 1.1 Key Differences

```
Semantic Segmentation:
  Pixel labels: "car", "person", "road"
  Cannot distinguish between individual instances
  Two adjacent cars → both labeled "car" (same class)

Instance Segmentation:
  Pixel labels: "car #1", "car #2", "person #1"
  Each object instance gets a unique mask
  Two adjacent cars → "car #1" and "car #2" (separate!)

  Semantic          Instance
  ┌──────────┐     ┌──────────┐
  │ ██  ██   │     │ ██  ▓▓   │
  │ ██  ██   │     │ ██  ▓▓   │  ██ = car #1
  │          │     │          │  ▓▓ = car #2
  │  ████    │     │  ████    │  ██ = person #1
  └──────────┘     └──────────┘
  All cars = same   Each car = unique
```

---

## 2. Mask R-CNN Architecture

### Theory: Mask R-CNN: Detection-Then-Segmentation

Mask R-CNN (He et al., 2017) extends Faster R-CNN (detection) by adding a **mask prediction branch**:

1. **Region Proposal Network (RPN)**: generates candidate object regions from feature maps.
2. **RoI pooling/align**: extracts a fixed-size feature map for each candidate region.
3. **Parallel heads**: classification, box regression, and **mask prediction** run in parallel on each RoI feature.
4. Mask head outputs a `K × 28 × 28` binary mask (one per class); at test time, select the mask for the predicted class.

#### B.1 RoIAlign: the key technical contribution

The original RoI pooling in Faster R-CNN **discretizes** spatial coordinates — it rounds RoI boundaries and quantizes the pooled output. For classification this is fine (small misalignments don't hurt class prediction). For mask prediction it is catastrophic — pixel-level misalignment propagates into the mask shape.

**RoIAlign** replaces discretization with bilinear interpolation: for each cell of the pooled output, sample the feature map at exact fractional positions and interpolate. Preserves precise spatial alignment between features and original pixel coordinates.

This single change (RoIAlign vs RoIPool) was worth ~5 points of mask AP — a huge improvement from what looks like a minor detail, showing how important sub-pixel alignment is for mask quality.

#### B.2 Loss function

Mask R-CNN trains with a multi-task loss:

```
L = L_classification + L_bbox_regression + L_mask
```

where `L_mask` is average binary cross-entropy over the `K × 28 × 28` mask output, computed **only for the ground-truth class** (other class channels contribute zero). Decoupling the mask loss from the classification loss lets the network focus on shape without competing class predictions.

### 2.1 Mask R-CNN Pipeline

```
Mask R-CNN = Faster R-CNN + Mask Branch

  Image → Backbone (ResNet+FPN) → Feature Maps
                                       │
                              ┌────────┼────────┐
                              ▼        ▼        ▼
                            RPN     Box Head  Mask Head
                          (proposals) (class+box) (per-instance mask)
                              │        │        │
                              ▼        ▼        ▼
                           Regions   Classes  Binary masks
                                    + BBoxes  (28×28 per instance)
```

### 2.2 Implementation with Torchvision

```python
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_mask_rcnn(n_classes, pretrained=True):
    """Build Mask R-CNN with custom number of classes."""
    model = maskrcnn_resnet50_fpn_v2(pretrained=pretrained)

    # Replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes)

    # Replace mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, n_classes
    )

    return model


def predict_masks(model, image, threshold=0.5, device='cuda'):
    """Run instance segmentation inference."""
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        prediction = model([image.to(device)])[0]

    # Filter by confidence
    keep = prediction['scores'] > threshold
    masks = prediction['masks'][keep]      # (N, 1, H, W) binary masks
    boxes = prediction['boxes'][keep]       # (N, 4) bounding boxes
    labels = prediction['labels'][keep]     # (N,) class labels
    scores = prediction['scores'][keep]     # (N,) confidence scores

    return masks, boxes, labels, scores


def visualize_instances(image, masks, boxes, labels, scores, class_names):
    """Overlay instance masks on image."""
    import numpy as np
    import matplotlib.pyplot as plt

    img = image.permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)

    colors = plt.cm.Set3(np.linspace(0, 1, len(masks)))

    for i, (mask, box, label, score) in enumerate(
        zip(masks, boxes, labels, scores)
    ):
        # Overlay mask with transparency
        m = mask[0].cpu().numpy() > 0.5
        color_mask = np.zeros_like(img)
        color_mask[m] = colors[i][:3]
        ax.imshow(color_mask, alpha=0.4)

        # Draw bounding box
        x1, y1, x2, y2 = box.cpu().numpy()
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                            fill=False, color=colors[i], linewidth=2)
        ax.add_patch(rect)

        # Label
        name = class_names[label.item()] if class_names else str(label.item())
        ax.text(x1, y1-5, f'{name}: {score:.2f}',
                color='white', fontsize=10,
                bbox=dict(boxstyle='round', facecolor=colors[i][:3]))

    plt.axis('off')
    plt.tight_layout()
    plt.savefig('instance_segmentation.png', dpi=150)
    plt.show()
```

### 2.3 RoIAlign

```
RoIAlign: Key improvement in Mask R-CNN over Faster R-CNN.

Problem with RoIPool:
  Quantization of RoI boundaries loses spatial precision.
  Fine for bounding boxes, BAD for pixel-level masks.

RoIAlign solution:
  Use bilinear interpolation instead of quantization.
  Sample exact sub-pixel locations for each bin.

  RoIPool:  round(x) → snap to grid → misaligned features
  RoIAlign: bilinear(x) → smooth interpolation → pixel-perfect
```

---

## 3. YOLACT: Real-Time Instance Segmentation

### Theory: One-Stage Detection-Free Methods

Two-stage methods like Mask R-CNN are accurate but slow. One-stage methods aim for real-time inference.

#### C.1 YOLACT: Prototypes + Coefficients

YOLACT (Bolya et al., 2019) factorizes the instance segmentation problem:

- **Prototype branch**: produces a set of `k` (e.g. 32) prototype masks at the full image resolution. These are not instance-specific; they are a shared basis.
- **Prediction heads**: per anchor, predict class, bounding box, and `k` **coefficients**.
- **Final mask**: `mask_i = sigmoid( Σ_j  coef_{i,j} · prototype_j )`, cropped by the bounding box.

Because prototypes are image-wide and shared, the heavy computation is done once per image; per-instance work is just a linear combination. Real-time speed (~30 fps) with reasonable accuracy.

#### C.2 SOLO: Per-Pixel Instance Prediction

SOLO (Wang et al., 2020) treats each spatial location on the feature map as a potential instance. At each grid cell, the network predicts:

- **Instance category**: class of the object whose **center** falls in this cell.
- **Instance mask**: binary mask of that object in full image resolution.

So a `40×40` feature map can have up to `1600` potential instances (most predicted as "no instance"). This decouples instance identity from bounding boxes entirely — SOLO produces masks directly, no box regression needed. SOLOv2 adds dynamic convolution to efficiently produce per-instance masks.

### 3.1 YOLACT Concept

```
YOLACT: You Only Look At CoefficienTs

Key idea: Separate mask generation into two parallel tasks:
  1. Generate K prototype masks (full-image, class-agnostic)
  2. Predict K coefficients per instance

  Final mask = linear combination of prototypes weighted by coefficients

  Instance mask_i = σ(Σ_k c_ik × prototype_k)

  This is MUCH faster than Mask R-CNN because:
  - Prototypes are computed once for the whole image
  - No per-instance RoI operation for masks

  Speed: ~30 FPS on a single GPU (vs ~5 FPS for Mask R-CNN)
```

### 3.2 YOLACT Architecture Overview

```python
class YOLACTHead(torch.nn.Module):
    """Simplified YOLACT prediction head."""

    def __init__(self, in_channels, n_classes, n_prototypes=32):
        super().__init__()
        self.n_prototypes = n_prototypes

        # Classification head
        self.cls_head = torch.nn.Conv2d(in_channels, n_classes, 3, padding=1)

        # Box regression head
        self.box_head = torch.nn.Conv2d(in_channels, 4, 3, padding=1)

        # Mask coefficient head
        self.coeff_head = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, n_prototypes, 3, padding=1),
            torch.nn.Tanh(),  # Coefficients in [-1, 1]
        )

    def forward(self, features):
        cls = self.cls_head(features)
        box = self.box_head(features)
        coeffs = self.coeff_head(features)
        return cls, box, coeffs
```

---

## 4. Anchor-Free Methods (SOLO, SOLOv2)

### Theory: Transformer-Based Instance Segmentation

DETR (Carion et al., 2020) reformulated detection as a **set prediction** problem: output a fixed-size set of `N` predictions (each with class and box), match them to ground-truth via Hungarian algorithm. MaskFormer and Mask2Former extend this to segmentation:

- Each of the `N` queries produces a class + mask.
- Queries attend to image features via cross-attention.
- Training uses bipartite matching to assign predictions to ground-truth instances.

Advantages: unified framework for detection, semantic, instance, and panoptic segmentation; no anchors, no NMS, no ROI align. Disadvantages: slower convergence during training, more compute per forward pass.

Mask2Former is currently the state-of-the-art for instance segmentation on COCO and ADE20K.

### 4.1 SOLO: Segmenting Objects by Locations

```
SOLO eliminates anchors and bounding boxes entirely:

Divide image into S×S grid cells.
Each cell predicts:
  - Category (which class?)
  - Instance mask (binary mask for that instance)

If object center falls in cell (i,j):
  Cell (i,j) is responsible for that object's mask.

SOLOv2 improvement:
  Dynamically generate mask kernel weights per instance.
  Kernel × Feature = Instance mask
  Much more efficient than predicting full masks.
```

---

## 5. Training and Data Preparation

### 5.1 COCO Format Dataset

```python
# COCO annotation format for instance segmentation:
# {
#   "images": [{"id": 1, "file_name": "img.jpg", "width": 640, "height": 480}],
#   "annotations": [{
#     "id": 1,
#     "image_id": 1,
#     "category_id": 1,
#     "segmentation": [[x1,y1,x2,y2,...,xn,yn]],  # Polygon points
#     "bbox": [x, y, w, h],
#     "area": 1234.5,
#     "iscrowd": 0
#   }],
#   "categories": [{"id": 1, "name": "car"}]
# }

import torch
from torch.utils.data import Dataset
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import cv2


class COCOInstanceDataset(Dataset):
    """COCO-format instance segmentation dataset."""

    def __init__(self, root, annotation_file, transforms=None):
        self.root = root
        self.coco = COCO(annotation_file)
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]

        # Load image
        img_path = f"{self.root}/{img_info['file_name']}"
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        masks = []
        boxes = []
        labels = []

        for ann in anns:
            # Binary mask
            mask = self.coco.annToMask(ann)
            masks.append(mask)

            # Bounding box [x, y, w, h] → [x1, y1, x2, y2]
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])

            labels.append(ann['category_id'])

        target = {
            'masks': torch.as_tensor(np.array(masks), dtype=torch.uint8),
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([img_id]),
        }

        img = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)
```

---

## 6. Evaluation Metrics (COCO AP)

### Theory: Evaluation: COCO-Style Average Precision

The standard metric for instance segmentation is **mask AP** (Average Precision), computed similarly to detection AP but using mask IoU instead of box IoU.

Procedure:

1. For each predicted instance with confidence `c` and mask `m`:
   - Find the ground-truth instance of the same class with the highest mask IoU.
   - Mark as true positive if IoU ≥ threshold, else false positive.
2. Sweep through predictions by decreasing confidence, accumulate TP and FP counts, compute precision-recall curve.
3. AP = area under precision-recall curve.

COCO reports AP averaged over 10 IoU thresholds from 0.5 to 0.95 in 0.05 steps:

- **AP@0.5** (PASCAL-style): lenient, rewards approximate masks.
- **AP@0.75**: strict, requires precise masks.
- **AP** (averaged over all thresholds): standard leaderboard metric.

Also reported per size: `AP_small` (mask area < 32²), `AP_medium` (32²-96²), `AP_large` (> 96²). Small objects are consistently the hardest — a fixed-pixel mask error affects small masks more than large ones.

### 6.1 Mask AP Calculation

```python
def compute_mask_iou(pred_mask, gt_mask):
    """Compute IoU between two binary masks."""
    intersection = (pred_mask & gt_mask).sum().float()
    union = (pred_mask | gt_mask).sum().float()
    return (intersection / (union + 1e-6)).item()


def compute_ap(precisions, recalls):
    """Compute Average Precision using 101-point interpolation (COCO style)."""
    recall_points = np.linspace(0, 1, 101)
    ap = 0

    for r in recall_points:
        prec_at_r = max([p for p, rec in zip(precisions, recalls) if rec >= r],
                        default=0)
        ap += prec_at_r

    return ap / 101


# COCO evaluation uses:
# AP @ IoU=0.50:0.95 (average over 10 IoU thresholds)
# AP@50 (IoU threshold = 0.50, like VOC)
# AP@75 (IoU threshold = 0.75, stricter)
# AP_small, AP_medium, AP_large (by object size)
```

---

## 7. Practical Applications

### 7.1 Object Counting and Analysis

```python
def count_and_measure_instances(model, image, class_names, device='cuda'):
    """Count instances and measure their properties."""
    masks, boxes, labels, scores = predict_masks(model, image, device=device)

    results = {}
    for label_id in labels.unique():
        class_mask = labels == label_id
        class_name = class_names[label_id.item()]
        n_instances = class_mask.sum().item()

        # Measure area of each instance
        areas = []
        for m in masks[class_mask]:
            area = (m > 0.5).sum().item()
            areas.append(area)

        results[class_name] = {
            'count': n_instances,
            'total_area': sum(areas),
            'avg_area': np.mean(areas) if areas else 0,
            'min_area': min(areas) if areas else 0,
            'max_area': max(areas) if areas else 0,
        }

    return results
```

---

## 8. Exercises

### Exercise 1: Mask R-CNN Fine-Tuning

Fine-tune Mask R-CNN on a custom dataset:
1. Prepare a dataset with polygon annotations (use LabelMe or CVAT)
2. Fine-tune pretrained Mask R-CNN on your dataset
3. Evaluate with COCO-style mask AP
4. Visualize predictions: compare with ground truth
5. Analyze failure cases: what does the model get wrong?

### Exercise 2: Real-Time Instance Segmentation

Build a real-time instance segmentation system:
1. Compare Mask R-CNN vs YOLACT on the same dataset
2. Measure FPS for each model on GPU
3. Apply to webcam feed
4. Implement instance tracking across frames (simple IoU-based)
5. Display: count of each class, colored masks in real-time

### Exercise 3: Instance Counting Application

Build an object counting application:
1. Train instance segmentation model on a counting dataset (e.g., cells, fruits)
2. Post-process masks: handle overlapping instances
3. Implement counting with confidence thresholds
4. Evaluate: counting accuracy vs detection threshold
5. Handle edge cases: occluded objects, clustered objects

### Exercise 4: Mask Quality Analysis

Analyze mask quality across methods:
1. Train Mask R-CNN and YOLACT on the same dataset
2. Compare mask quality: AP@50, AP@75, AP@50:95
3. Analyze per-class: which classes have better/worse masks?
4. Visualize boundary quality: zoom into mask edges
5. Measure: computation time vs mask quality tradeoff

### Exercise 5: Custom Loss for Better Masks

Experiment with loss functions for mask prediction:
1. Train with binary cross-entropy (default Mask R-CNN)
2. Train with Dice loss
3. Train with boundary-aware loss (emphasize mask edges)
4. Compare mask quality, especially at boundaries
5. Combine losses and find optimal weighting

---

*End of Lesson 26*
