"""
14. Edge AI for Computer Vision

Demonstrates computer vision models and techniques optimized for
edge deployment: lightweight detection, segmentation, and tracking.

Covers:
- Lightweight image classification (MobileNet-style)
- Anchor-free object detection head
- Lightweight semantic segmentation
- Simple object tracking with IoU matching
- Multi-task edge vision model

Requirements:
    pip install torch numpy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from typing import List, Tuple, Dict

print("=" * 60)
print("Edge AI — Computer Vision on Edge")
print("=" * 60)


# ============================================
# 1. Lightweight Feature Backbone
# ============================================
print("\n[1] Lightweight Feature Backbone")
print("-" * 40)


class DepthwiseSeparable(nn.Module):
    """Depthwise separable convolution: depthwise + pointwise."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1,
                            groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = F.relu(self.bn1(self.dw(x)), inplace=True)
        x = F.relu(self.bn2(self.pw(x)), inplace=True)
        return x


class EdgeBackbone(nn.Module):
    """Lightweight backbone producing multi-scale feature maps."""

    def __init__(self, width_mult=0.5):
        super().__init__()
        def ch(c):
            return max(8, int(c * width_mult))

        self.stem = nn.Sequential(
            nn.Conv2d(3, ch(32), 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch(32)), nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            DepthwiseSeparable(ch(32), ch(64), stride=2),
            DepthwiseSeparable(ch(64), ch(64)),
        )
        self.stage2 = nn.Sequential(
            DepthwiseSeparable(ch(64), ch(128), stride=2),
            DepthwiseSeparable(ch(128), ch(128)),
        )
        self.stage3 = nn.Sequential(
            DepthwiseSeparable(ch(128), ch(256), stride=2),
            DepthwiseSeparable(ch(256), ch(256)),
        )
        self._channels = [ch(64), ch(128), ch(256)]

    @property
    def out_channels(self):
        return self._channels

    def forward(self, x):
        x = self.stem(x)
        f1 = self.stage1(x)   # 1/4 resolution
        f2 = self.stage2(f1)  # 1/8 resolution
        f3 = self.stage3(f2)  # 1/16 resolution
        return f1, f2, f3


backbone = EdgeBackbone(width_mult=0.5)
backbone.eval()
params = sum(p.numel() for p in backbone.parameters())
print(f"EdgeBackbone (width_mult=0.5): {params:,} parameters")
print(f"Feature channels: {backbone.out_channels}")

dummy = torch.randn(1, 3, 128, 128)
with torch.no_grad():
    f1, f2, f3 = backbone(dummy)
print(f"Input:  {dummy.shape}")
print(f"Stage1: {f1.shape}  (1/4)")
print(f"Stage2: {f2.shape}  (1/8)")
print(f"Stage3: {f3.shape} (1/16)")


# ============================================
# 2. Edge Image Classifier
# ============================================
print("\n[2] Edge Image Classifier")
print("-" * 40)


class EdgeClassifier(nn.Module):
    """Lightweight classifier using the edge backbone."""

    def __init__(self, num_classes=10, width_mult=0.5):
        super().__init__()
        self.backbone = EdgeBackbone(width_mult)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.backbone.out_channels[-1], num_classes)

    def forward(self, x):
        _, _, f3 = self.backbone(x)
        x = self.pool(f3).flatten(1)
        return self.fc(x)


classifier = EdgeClassifier(num_classes=10, width_mult=0.5)
classifier.eval()
params = sum(p.numel() for p in classifier.parameters())
print(f"EdgeClassifier: {params:,} parameters")

with torch.no_grad():
    out = classifier(dummy)
    probs = F.softmax(out, dim=1)
print(f"Output shape: {out.shape}")
print(f"Top class: {probs.argmax().item()}, confidence: {probs.max().item():.3f}")

# Benchmark
start = time.perf_counter()
with torch.no_grad():
    for _ in range(200):
        classifier(dummy)
ms_per_frame = (time.perf_counter() - start) / 200 * 1000
print(f"Inference: {ms_per_frame:.2f} ms/frame ({1000 / ms_per_frame:.0f} FPS)")


# ============================================
# 3. Anchor-Free Detection Head
# ============================================
print("\n[3] Anchor-Free Object Detection Head")
print("-" * 40)
print("Predicts center, width/height, and class per feature map cell.\n")


class DetectionHead(nn.Module):
    """Simple anchor-free detection head (FCOS-style)."""

    def __init__(self, in_channels, num_classes=5):
        super().__init__()
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, 1),
        )
        self.reg_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 4, 1),  # (l, t, r, b) distances
        )
        self.centerness = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, 1),
        )

    def forward(self, feature_map):
        cls_logits = self.cls_head(feature_map)
        bbox_reg = F.relu(self.reg_head(feature_map))  # Positive distances
        center = torch.sigmoid(self.centerness(feature_map))
        return cls_logits, bbox_reg, center


class EdgeDetector(nn.Module):
    """Lightweight object detector with backbone + detection head."""

    def __init__(self, num_classes=5, width_mult=0.5):
        super().__init__()
        self.backbone = EdgeBackbone(width_mult)
        # Use stage2 features for detection (balance resolution/semantics)
        self.det_head = DetectionHead(self.backbone.out_channels[1], num_classes)

    def forward(self, x):
        _, f2, _ = self.backbone(x)
        return self.det_head(f2)


detector = EdgeDetector(num_classes=5, width_mult=0.5)
detector.eval()
params = sum(p.numel() for p in detector.parameters())
print(f"EdgeDetector: {params:,} parameters")

with torch.no_grad():
    cls_out, reg_out, ctr_out = detector(dummy)
print(f"Classification map: {cls_out.shape}  (B, C, H, W)")
print(f"Regression map:     {reg_out.shape}  (B, 4, H, W)")
print(f"Centerness map:     {ctr_out.shape}  (B, 1, H, W)")


# ============================================
# 4. Simple NMS Post-processing
# ============================================
print("\n[4] Non-Maximum Suppression (NMS)")
print("-" * 40)


def simple_nms(boxes, scores, iou_threshold=0.5):
    """Basic greedy NMS implementation for edge deployment."""
    if len(boxes) == 0:
        return []

    # Sort by score descending
    order = scores.argsort(descending=True)
    keep = []

    while len(order) > 0:
        idx = order[0].item()
        keep.append(idx)

        if len(order) == 1:
            break

        # Compute IoU with remaining boxes
        remaining = order[1:]
        xx1 = torch.maximum(boxes[idx, 0], boxes[remaining, 0])
        yy1 = torch.maximum(boxes[idx, 1], boxes[remaining, 1])
        xx2 = torch.minimum(boxes[idx, 2], boxes[remaining, 2])
        yy2 = torch.minimum(boxes[idx, 3], boxes[remaining, 3])

        inter = torch.clamp(xx2 - xx1, min=0) * torch.clamp(yy2 - yy1, min=0)
        area_a = (boxes[idx, 2] - boxes[idx, 0]) * (boxes[idx, 3] - boxes[idx, 1])
        area_b = (boxes[remaining, 2] - boxes[remaining, 0]) * \
                 (boxes[remaining, 3] - boxes[remaining, 1])
        iou = inter / (area_a + area_b - inter + 1e-6)

        mask = iou <= iou_threshold
        order = remaining[mask]

    return keep


# Test NMS
boxes = torch.tensor([
    [10.0, 10.0, 50.0, 50.0],
    [12.0, 12.0, 52.0, 52.0],  # Overlaps with box 0
    [100.0, 100.0, 150.0, 150.0],
    [105.0, 105.0, 155.0, 155.0],  # Overlaps with box 2
])
scores = torch.tensor([0.9, 0.8, 0.95, 0.7])

kept = simple_nms(boxes, scores, iou_threshold=0.5)
print(f"Input boxes: {len(boxes)}, After NMS: {len(kept)}")
print(f"Kept indices: {kept}")


# ============================================
# 5. Lightweight Semantic Segmentation
# ============================================
print("\n[5] Lightweight Semantic Segmentation")
print("-" * 40)


class EdgeSegmentationHead(nn.Module):
    """Simple upsampling segmentation head."""

    def __init__(self, in_channels_list, num_classes=5):
        super().__init__()
        # Merge multi-scale features with lateral connections
        self.lateral3 = nn.Conv2d(in_channels_list[2], 64, 1)
        self.lateral2 = nn.Conv2d(in_channels_list[1], 64, 1)
        self.lateral1 = nn.Conv2d(in_channels_list[0], 64, 1)
        self.smooth = nn.Conv2d(64, 64, 3, padding=1)
        self.predict = nn.Conv2d(64, num_classes, 1)

    def forward(self, f1, f2, f3):
        # Top-down pathway
        p3 = self.lateral3(f3)
        p2 = self.lateral2(f2) + F.interpolate(p3, size=f2.shape[2:], mode="nearest")
        p1 = self.lateral1(f1) + F.interpolate(p2, size=f1.shape[2:], mode="nearest")
        out = self.smooth(p1)
        return self.predict(out)


class EdgeSegmenter(nn.Module):
    """Lightweight semantic segmentation model."""

    def __init__(self, num_classes=5, width_mult=0.5):
        super().__init__()
        self.backbone = EdgeBackbone(width_mult)
        self.seg_head = EdgeSegmentationHead(self.backbone.out_channels, num_classes)

    def forward(self, x):
        f1, f2, f3 = self.backbone(x)
        logits = self.seg_head(f1, f2, f3)
        # Upsample to original resolution
        return F.interpolate(logits, size=x.shape[2:], mode="bilinear",
                             align_corners=False)


segmenter = EdgeSegmenter(num_classes=5, width_mult=0.5)
segmenter.eval()
params = sum(p.numel() for p in segmenter.parameters())
print(f"EdgeSegmenter: {params:,} parameters")

with torch.no_grad():
    seg_out = segmenter(dummy)
print(f"Input:  {dummy.shape}")
print(f"Output: {seg_out.shape} (per-pixel class logits)")
print(f"Pred mask: {seg_out.argmax(1).shape}")


# ============================================
# 6. Simple IoU Tracker
# ============================================
print("\n[6] Simple Object Tracker (IoU-based)")
print("-" * 40)
print("Lightweight tracking by matching detections across frames via IoU.\n")


class SimpleTracker:
    """IoU-based multi-object tracker for edge deployment."""

    def __init__(self, iou_threshold=0.3, max_age=5):
        self.tracks = {}
        self.next_id = 0
        self.iou_threshold = iou_threshold
        self.max_age = max_age

    def _compute_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / (area1 + area2 - inter + 1e-6)

    def update(self, detections: List[Tuple[float, float, float, float]]):
        """Match detections to existing tracks, create new tracks."""
        matched_tracks = set()
        matched_dets = set()

        # Greedy IoU matching
        for det_idx, det_box in enumerate(detections):
            best_iou = 0
            best_track_id = None
            for track_id, track_info in self.tracks.items():
                if track_id in matched_tracks:
                    continue
                iou = self._compute_iou(det_box, track_info["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id

            if best_iou >= self.iou_threshold and best_track_id is not None:
                self.tracks[best_track_id]["box"] = det_box
                self.tracks[best_track_id]["age"] = 0
                matched_tracks.add(best_track_id)
                matched_dets.add(det_idx)

        # Create new tracks for unmatched detections
        for det_idx, det_box in enumerate(detections):
            if det_idx not in matched_dets:
                self.tracks[self.next_id] = {"box": det_box, "age": 0}
                self.next_id += 1

        # Age out unmatched tracks
        to_remove = []
        for track_id in self.tracks:
            if track_id not in matched_tracks:
                self.tracks[track_id]["age"] += 1
                if self.tracks[track_id]["age"] > self.max_age:
                    to_remove.append(track_id)
        for tid in to_remove:
            del self.tracks[tid]

        return {tid: info["box"] for tid, info in self.tracks.items()}


# Simulate 5 frames of detections
tracker = SimpleTracker(iou_threshold=0.3)

frames_detections = [
    [(10, 10, 50, 50), (100, 100, 150, 150)],
    [(12, 12, 52, 52), (102, 98, 152, 148)],
    [(15, 14, 55, 54), (105, 95, 155, 145), (200, 200, 240, 240)],
    [(18, 16, 58, 56), (200, 202, 240, 242)],
    [(20, 18, 60, 58), (200, 204, 240, 244)],
]

for frame_idx, dets in enumerate(frames_detections):
    active = tracker.update(dets)
    track_ids = list(active.keys())
    print(f"Frame {frame_idx}: {len(dets)} detections -> {len(active)} tracks "
          f"(IDs: {track_ids})")


# ============================================
# 7. Multi-Task Edge Vision Model
# ============================================
print("\n[7] Multi-Task Edge Vision Model")
print("-" * 40)
print("Share backbone across classification + segmentation.\n")


class MultiTaskEdgeModel(nn.Module):
    """Shared backbone with classification and segmentation heads."""

    def __init__(self, num_classes_cls=10, num_classes_seg=5, width_mult=0.5):
        super().__init__()
        self.backbone = EdgeBackbone(width_mult)
        ch = self.backbone.out_channels

        # Classification head
        self.cls_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_fc = nn.Linear(ch[-1], num_classes_cls)

        # Segmentation head (simplified)
        self.seg_conv = nn.Conv2d(ch[0], num_classes_seg, 1)

    def forward(self, x):
        f1, f2, f3 = self.backbone(x)
        # Classification from deepest features
        cls_out = self.cls_fc(self.cls_pool(f3).flatten(1))
        # Segmentation from shallowest features
        seg_out = F.interpolate(self.seg_conv(f1), size=x.shape[2:],
                                mode="bilinear", align_corners=False)
        return cls_out, seg_out


mt_model = MultiTaskEdgeModel(num_classes_cls=10, num_classes_seg=5)
mt_model.eval()

# Compare parameter count: multi-task vs separate models
mt_params = sum(p.numel() for p in mt_model.parameters())
sep_params = sum(p.numel() for p in classifier.parameters()) + \
             sum(p.numel() for p in segmenter.parameters())

print(f"Multi-task model:    {mt_params:,} parameters")
print(f"Separate models:     {sep_params:,} parameters")
print(f"Parameter savings:   {(1 - mt_params / sep_params) * 100:.1f}%")

with torch.no_grad():
    cls_out, seg_out = mt_model(dummy)
print(f"\nClassification output: {cls_out.shape}")
print(f"Segmentation output:   {seg_out.shape}")


# ============================================
# 8. Summary
# ============================================
print("\n[8] Summary")
print("-" * 40)
print("Key takeaways:")
print("- Depthwise separable convolutions cut FLOPs by 8-9x vs standard conv")
print("- Width multiplier scales model size linearly with accuracy trade-off")
print("- Anchor-free detection avoids anchor tuning overhead on edge")
print("- Top-down FPN-style segmentation works well at low resolution")
print("- IoU-based tracking is simple and CPU-efficient for edge deployment")
print("- Multi-task models share backbone to save memory and compute")
