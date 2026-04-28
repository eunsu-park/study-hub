[Previous: Instance Segmentation](./26_Instance_Segmentation.md) | [Next: Neural Radiance Fields](./28_Neural_Radiance_Fields.md)

---

# 27. Panoptic Segmentation

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain panoptic segmentation as the unification of semantic and instance segmentation
2. Distinguish "stuff" (amorphous regions) from "things" (countable objects)
3. Implement Panoptic FPN combining FPN with semantic and instance heads
4. Build Mask2Former as a unified framework for all segmentation tasks
5. Evaluate panoptic quality (PQ) and its decomposition into SQ and RQ

---

## Table of Contents

1. [Panoptic Segmentation Overview](#1-panoptic-segmentation-overview)
2. [Stuff vs Things](#2-stuff-vs-things)
3. [Panoptic FPN](#3-panoptic-fpn)
4. [Mask2Former](#4-mask2former)
5. [Panoptic Quality (PQ)](#5-panoptic-quality-pq)
6. [Post-Processing and Fusion](#6-post-processing-and-fusion)
7. [Practical Pipeline](#7-practical-pipeline)
8. [Exercises](#8-exercises)

---

## 1. Panoptic Segmentation Overview

### Theory: The Unified Output

Panoptic output: every pixel `(u, v)` gets a pair `(class, instance_id)`:

- For **stuff pixels** (sky, road, building as a category): `class = <stuff class>, instance_id = 0` (no instance separation within stuff).
- For **thing pixels** (person, car as countable objects): `class = <thing class>, instance_id = <unique ID per object>`.

This is the most complete scene parse: every pixel is assigned, each object is separated, and background stuff is properly categorized. Downstream applications (autonomous driving, AR, image editing) need this complete parse — knowing both "all these pixels are road" and "those pixels are car 1, those are car 2" at the same time.

### 1.1 The Unified View

```
Panoptic = Semantic + Instance for ALL pixels

Semantic Segmentation:     Labels every pixel (stuff + things)
Instance Segmentation:     Segments individual objects (things only)
Panoptic Segmentation:     Both! Every pixel gets a class AND instance ID

Output format: For each pixel (i, j):
  - class_id: what class does this pixel belong to?
  - instance_id: which specific instance (for "things" classes)?

Example scene:
  sky (stuff, no instance)
  road (stuff, no instance)
  car #1 (thing, instance 1)
  car #2 (thing, instance 2)
  person #1 (thing, instance 3)
  tree (stuff, no instance)
```

---

## 2. Stuff vs Things

### Theory: Things vs Stuff: the Ontology

The things/stuff distinction is not arbitrary; it reflects how humans conceptualize scene elements:

- **Things** are **countable** objects with boundaries: cars, people, chairs, dogs. "Three chairs" makes sense. Each has a distinct shape and can be individually counted or tracked.
- **Stuff** is **amorphous**: sky, road, grass, water. "Three skies" doesn't make sense. Stuff doesn't have countable instances; it has extent and coverage.

This distinction is useful because the two kinds of content require **different evaluation**: for things, we care about instance separation (was each object found?); for stuff, we just care about spatial coverage (what fraction of the road was labeled road?). The PQ metric (covered in §5) handles both cases uniformly.

The ontology is fixed in the dataset — COCO Panoptic has 80 thing classes and 53 stuff classes, Cityscapes has 8 things and 11 stuff, ADE20K has 100 things and 50 stuff.

### 2.1 Classification

```
"Things": Countable objects with well-defined shape
  - People, cars, animals, furniture
  - Have instances: car #1, car #2
  - Need instance segmentation

"Stuff": Amorphous regions without clear boundaries
  - Sky, road, grass, water, wall
  - No individual instances
  - Need semantic segmentation only

COCO Panoptic: 80 thing classes + 53 stuff classes = 133 total
Cityscapes: 8 thing classes + 11 stuff classes = 19 total
```

---

## 3. Panoptic FPN

### Theory: Panoptic FPN: Two Heads, One Backbone

Panoptic FPN (Kirillov et al., 2019) was the first effective panoptic architecture. The idea: run **both** a semantic segmentation head and an instance segmentation head on top of the same Feature Pyramid Network backbone, then **fuse** their outputs into the final panoptic map.

Architecture:

1. **Shared backbone + FPN**: extract multi-scale features.
2. **Semantic head**: dense prediction of stuff classes + thing-class occupancy (say which pixels are "person" even if you don't know which person).
3. **Instance head** (Mask R-CNN style): detect things, predict per-instance masks.
4. **Fusion**: combine the two outputs. For each pixel:
   - If it is inside a confident instance mask, assign that instance's class and ID.
   - Otherwise, use the semantic head's prediction (stuff class or "unassigned thing").

The fusion step is surprisingly non-trivial because the two heads can disagree (instance mask says "person" at a pixel, semantic head says "wall"). The standard resolution: instance predictions override semantic predictions for things, but only above a confidence threshold.

### 3.1 Architecture

```
Panoptic FPN: Add semantic segmentation head to Mask R-CNN's FPN.

                    ┌─────────────────────────┐
  Image → Backbone → FPN (multi-scale features) │
                    └──────────┬──────────────┘
                               │
                    ┌──────────┼──────────────┐
                    ▼          ▼              ▼
              Semantic Head  RPN + Box Head  Mask Head
              (stuff labels) (thing detect)  (thing masks)
                    │          │              │
                    ▼          ▼              ▼
              Stuff segments  Thing boxes   Thing masks
                    │          └──────┬──────┘
                    │                 ▼
                    │          Instance segments
                    └────────┬────────┘
                             ▼
                    Panoptic Fusion
                    (merge stuff + things)
                             │
                             ▼
                    Panoptic Output
```

### 3.2 Semantic Head

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class PanopticFPNSemanticHead(nn.Module):
    """Semantic segmentation head for Panoptic FPN."""

    def __init__(self, fpn_channels=256, n_stuff_classes=54, n_thing_classes=80):
        super().__init__()
        n_classes = n_stuff_classes + n_thing_classes

        # Upsample each FPN level to 1/4 resolution and merge
        self.scale_heads = nn.ModuleList()
        for _ in range(4):  # P2, P3, P4, P5
            self.scale_heads.append(nn.Sequential(
                nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1),
                nn.GroupNorm(32, fpn_channels),
                nn.ReLU(inplace=True),
            ))

        self.predictor = nn.Conv2d(fpn_channels, n_classes, 1)

    def forward(self, fpn_features):
        """
        Args:
            fpn_features: dict of {level: tensor} from FPN
        Returns:
            semantic_pred: (B, n_classes, H/4, W/4)
        """
        target_size = fpn_features['p2'].shape[2:]
        merged = torch.zeros_like(fpn_features['p2'])

        for i, (level, head) in enumerate(
            zip(['p2', 'p3', 'p4', 'p5'], self.scale_heads)
        ):
            feat = head(fpn_features[level])
            feat = F.interpolate(feat, size=target_size,
                               mode='bilinear', align_corners=False)
            merged += feat

        return self.predictor(merged)
```

---

## 4. Mask2Former

### Theory: Mask2Former: One Architecture for All

Mask2Former (Cheng et al., 2022) showed that **the same transformer architecture can do semantic, instance, and panoptic segmentation with only minor changes**. Key observations:

- Any segmentation task can be framed as **predicting a set of binary masks + their class labels**.
- Semantic segmentation: one mask per class (no instance separation).
- Instance segmentation: one mask per thing instance.
- Panoptic segmentation: one mask per stuff class + one mask per thing instance.

The architecture:

1. **Pixel decoder**: extracts multi-scale features from a backbone (Swin Transformer is common).
2. **Transformer decoder**: produces `N` queries (e.g. 100), each attending to image features via masked attention.
3. **Per-query prediction**: each query outputs a class probability and a binary mask (via dot product with a per-pixel embedding).
4. **Training**: bipartite matching between queries and ground-truth instances (Hungarian algorithm), per-mask classification loss + per-mask binary cross-entropy + dice loss.

For panoptic, simply include "stuff" classes as valid predictions alongside "thing" instances — nothing else changes. This unification is the current state-of-the-art direction.

### 4.1 Mask2Former Architecture

```
Mask2Former: One architecture for ALL segmentation tasks.

Key innovation: Masked attention in transformer decoder.
  Instead of attending to all pixels, attend only to predicted mask region.
  This focuses computation where it matters.

Architecture:
  Image → Backbone → Pixel Decoder → Multi-scale Features
                                          │
  Learnable Queries → Transformer ← ──────┘
  (N queries)         Decoder (masked cross-attention)
                          │
                   ┌──────┼──────┐
                   ▼      ▼      ▼
               Class    Mask    Stuff/Thing
               Pred     Pred    Assignment
```

### 4.2 Simplified Mask2Former

```python
class Mask2FormerDecoder(nn.Module):
    """Simplified Mask2Former decoder."""

    def __init__(self, d_model=256, n_queries=100, n_classes=133,
                 n_heads=8, n_layers=6):
        super().__init__()
        self.n_queries = n_queries

        # Learnable object queries
        self.query_embed = nn.Embedding(n_queries, d_model)

        # Transformer decoder layers with masked attention
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=2048, dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, n_layers)

        # Prediction heads
        self.class_head = nn.Linear(d_model, n_classes + 1)  # +1 for "no object"
        self.mask_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, pixel_features):
        """
        Args:
            pixel_features: (B, HW, d_model) from pixel decoder
        Returns:
            class_logits: (B, N, n_classes+1)
            mask_logits: (B, N, H, W)
        """
        B = pixel_features.shape[0]

        # Initialize queries
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        # Transformer decoder
        output = self.decoder(queries, pixel_features)

        # Class predictions
        class_logits = self.class_head(output)  # (B, N, C+1)

        # Mask predictions via dot product with pixel features
        mask_embed = self.mask_head(output)  # (B, N, d_model)
        mask_logits = torch.bmm(mask_embed, pixel_features.transpose(1, 2))

        return class_logits, mask_logits
```

---

## 5. Panoptic Quality (PQ)

### Theory: Panoptic Quality (PQ)

The canonical panoptic metric. For each class `c`:

```
TP_c = set of matched (predicted, ground-truth) pairs with IoU > 0.5 and same class
FP_c = unmatched predictions
FN_c = unmatched ground-truth instances

PQ_c = (Σ IoU over TP_c) / (|TP_c| + 0.5·|FP_c| + 0.5·|FN_c|)
```

Then `PQ = mean(PQ_c over all classes)`.

#### The SQ × RQ decomposition

PQ can be factored:

```
PQ = SQ × RQ

SQ = (Σ IoU over TP) / |TP|          Segmentation Quality: how well-aligned are matched masks?
RQ = |TP| / (|TP| + 0.5·FP + 0.5·FN)  Recognition Quality: what fraction of objects were correctly found?
```

This decomposition is useful for diagnosis:

- Low **RQ** with high **SQ**: the system finds some objects perfectly but misses many. Improve recall.
- High **RQ** with low **SQ**: the system finds the right objects but their masks are loose. Improve precision.

For things, RQ is an F1-like score over instance matching; for stuff, each class contributes a single TP/FN (the whole-class mask), so RQ_for_stuff is more about whether the stuff class exists in the prediction than about how many instances there are.

#### Why PQ is better than just "semantic mIoU + instance AP"

A naïve panoptic metric would be `0.5·mIoU + 0.5·AP`. But that is unsatisfying: mIoU rewards a model that predicts "road everywhere", while AP rewards finding many detections even if masks are loose. PQ instead imposes the same threshold (IoU > 0.5) on everything, so a prediction only counts if it is both correctly localized and correctly classified. This produces a single coherent number that behaves intuitively.

### 5.1 PQ Metric

```python
import numpy as np


def panoptic_quality(pred_segments, gt_segments, pred_labels, gt_labels,
                     iou_threshold=0.5):
    """
    Compute Panoptic Quality (PQ).

    PQ = (Σ IoU(p,g)) / (|TP| + 0.5|FP| + 0.5|FN|)
       = SQ × RQ

    SQ (Segmentation Quality) = average IoU of matched segments
    RQ (Recognition Quality) = F1 score of segment matching

    Higher is better. Range: [0, 1].
    """
    matched_iou = []
    tp = 0
    fp = 0
    fn = 0

    gt_matched = set()

    for pred_id in np.unique(pred_segments):
        if pred_id == 0:
            continue  # Skip void

        pred_mask = pred_segments == pred_id
        pred_label = pred_labels.get(pred_id, -1)

        best_iou = 0
        best_gt = None

        for gt_id in np.unique(gt_segments):
            if gt_id == 0 or gt_id in gt_matched:
                continue

            gt_mask = gt_segments == gt_id
            gt_label = gt_labels.get(gt_id, -1)

            if pred_label != gt_label:
                continue

            intersection = (pred_mask & gt_mask).sum()
            union = (pred_mask | gt_mask).sum()
            iou = intersection / (union + 1e-6)

            if iou > best_iou:
                best_iou = iou
                best_gt = gt_id

        if best_iou > iou_threshold:
            tp += 1
            matched_iou.append(best_iou)
            gt_matched.add(best_gt)
        else:
            fp += 1

    # Count unmatched ground truth as FN
    for gt_id in np.unique(gt_segments):
        if gt_id != 0 and gt_id not in gt_matched:
            fn += 1

    # Compute PQ
    sq = np.mean(matched_iou) if matched_iou else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-6)
    pq = sq * rq

    return {
        'PQ': pq,
        'SQ': sq,
        'RQ': rq,
        'TP': tp,
        'FP': fp,
        'FN': fn,
    }
```

---

## 6. Post-Processing and Fusion

### 6.1 Panoptic Fusion

```python
def panoptic_fusion(semantic_pred, instance_masks, instance_labels,
                    instance_scores, stuff_classes, thing_classes,
                    overlap_threshold=0.5, score_threshold=0.5):
    """
    Merge semantic (stuff) and instance (things) predictions
    into a single panoptic segmentation map.
    """
    H, W = semantic_pred.shape
    panoptic = np.zeros((H, W), dtype=np.int32)
    segment_info = []
    next_id = 1

    # 1. Place instance masks (things) - highest priority
    # Sort by confidence score (highest first)
    sorted_idx = np.argsort(-instance_scores.numpy())

    occupied = np.zeros((H, W), dtype=bool)

    for idx in sorted_idx:
        score = instance_scores[idx].item()
        if score < score_threshold:
            continue

        mask = instance_masks[idx].numpy() > 0.5
        label = instance_labels[idx].item()

        if label not in thing_classes:
            continue

        # Check overlap with already placed instances
        overlap = (mask & occupied).sum() / (mask.sum() + 1e-6)
        if overlap > overlap_threshold:
            continue

        panoptic[mask] = next_id
        occupied[mask] = True
        segment_info.append({
            'id': next_id,
            'category_id': label,
            'isthing': True,
            'score': score,
        })
        next_id += 1

    # 2. Fill remaining pixels with stuff classes
    for stuff_class in stuff_classes:
        stuff_mask = (semantic_pred == stuff_class) & (~occupied)
        if stuff_mask.sum() > 0:
            panoptic[stuff_mask] = next_id
            segment_info.append({
                'id': next_id,
                'category_id': stuff_class,
                'isthing': False,
            })
            next_id += 1

    return panoptic, segment_info
```

---

## 7. Practical Pipeline

### 7.1 Complete Training Example

```python
def train_panoptic_model(model, train_loader, val_loader, n_classes,
                          epochs=50, lr=1e-4, device='cuda'):
    """Train a panoptic segmentation model."""
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import StepLR

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            total_loss += losses.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")
```

---

## 8. Exercises

### Exercise 1: Panoptic Fusion Implementation

Build the panoptic fusion pipeline:
1. Use pretrained Mask R-CNN for things and DeepLab for stuff
2. Implement the fusion algorithm (merge masks with priority rules)
3. Evaluate with Panoptic Quality (PQ) on COCO val
4. Visualize: color-coded panoptic output with legend
5. Analyze: PQ_things vs PQ_stuff separately

### Exercise 2: Mask2Former Training

Train Mask2Former for panoptic segmentation:
1. Use Detectron2's Mask2Former implementation
2. Train on Cityscapes panoptic (19 classes)
3. Report PQ, SQ, RQ, and per-class breakdown
4. Compare with Panoptic FPN on the same dataset
5. Analyze: which classes benefit most from Mask2Former?

### Exercise 3: Stuff-Things Analysis

Analyze the stuff vs things distinction:
1. Take a panoptic dataset and compute statistics per category
2. How much of the image is stuff vs things (pixel area)?
3. Which stuff classes are most often confused?
4. Which thing classes have worst instance separation?
5. Propose: how could boundary quality be improved?

### Exercise 4: Video Panoptic Segmentation

Extend panoptic segmentation to video:
1. Run panoptic segmentation on consecutive frames
2. Track thing instances across frames using IoU matching
3. Maintain consistent IDs for the same object
4. Handle: object appearance, disappearance, occlusion
5. Measure temporal consistency of stuff predictions

### Exercise 5: Custom Panoptic Dataset

Create and annotate a panoptic dataset:
1. Collect 100 images of a specific domain (e.g., kitchen, office)
2. Define stuff classes (wall, floor, ceiling) and thing classes (chair, cup)
3. Annotate using COCO panoptic format
4. Train a model on your custom dataset
5. Evaluate and analyze domain-specific challenges

---

*End of Lesson 27*
