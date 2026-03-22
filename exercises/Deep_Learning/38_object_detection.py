"""
Exercises for Lesson 38: Object Detection
Topic: Deep_Learning

Solutions to practice problems from the lesson.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# === Exercise 1: IoU and GIoU Implementation ===
# Problem: Implement IoU and GIoU for bounding boxes.

def exercise_1():
    """Implement IoU and GIoU for bounding box pairs."""

    def calculate_iou(box1, box2):
        """Calculate IoU for [x1, y1, x2, y2] format boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def calculate_giou(box1, box2):
        """Calculate Generalized IoU."""
        iou = calculate_iou(box1, box2)

        # Enclosing box
        enc_x1 = min(box1[0], box2[0])
        enc_y1 = min(box1[1], box2[1])
        enc_x2 = max(box1[2], box2[2])
        enc_y2 = max(box1[3], box2[3])
        enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)

        # Union
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        union = area1 + area2 - intersection

        giou = iou - (enc_area - union) / enc_area if enc_area > 0 else iou
        return giou

    # Test cases
    cases = [
        ("Overlapping", [0, 0, 100, 100], [50, 50, 150, 150]),
        ("Non-overlapping", [0, 0, 100, 100], [200, 200, 300, 300]),
        ("Identical", [0, 0, 100, 100], [0, 0, 100, 100]),
    ]

    print(f"  {'Case':<18} {'IoU':>8} {'GIoU':>8}")
    print(f"  {'-'*18} {'-'*8} {'-'*8}")
    for name, b1, b2 in cases:
        iou = calculate_iou(b1, b2)
        giou = calculate_giou(b1, b2)
        print(f"  {name:<18} {iou:8.4f} {giou:8.4f}")

    print("\n  Non-overlapping: IoU=0 (no gradient signal), GIoU=-0.5278 (still useful).")
    print("  GIoU penalizes predicted boxes that are far from ground truth,")
    print("  providing gradient signal even when boxes don't overlap.")


# === Exercise 2: YOLOv8-style Inference (Simulated) ===
# Problem: Simulate object detection inference.

def exercise_2():
    """Simulated YOLO-style detection inference."""
    torch.manual_seed(42)

    # Simulate detection output
    class SimpleDetector(nn.Module):
        def __init__(self, n_classes=80, n_anchors=3):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            )
            # Detection head: (x, y, w, h, obj_conf, class_scores)
            self.head = nn.Conv2d(32, n_anchors * (5 + n_classes), 1)

        def forward(self, x):
            features = self.backbone(x)
            return self.head(features)

    model = SimpleDetector(n_classes=10)
    image = torch.randn(1, 3, 128, 128)

    with torch.no_grad():
        output = model(image)

    # Parse detections
    n_anchors = 3
    n_classes = 10
    B, C, H, W = output.shape
    output = output.view(B, n_anchors, 5 + n_classes, H, W)
    output = output.permute(0, 1, 3, 4, 2)  # (B, anchors, H, W, 5+classes)

    # Extract predictions
    obj_conf = torch.sigmoid(output[..., 4])
    class_scores = torch.softmax(output[..., 5:], dim=-1)
    class_conf, class_pred = class_scores.max(dim=-1)

    # Filter by confidence
    conf = obj_conf * class_conf
    mask = conf > 0.1  # Low threshold for demo

    detections = conf[mask].tolist()[:10]  # Top 10
    classes = class_pred[mask].tolist()[:10]

    print(f"  Simulated detection on 128x128 image:")
    print(f"  Feature map size: {H}x{W}, anchors_per_cell: {n_anchors}")
    print(f"  Total candidate boxes: {n_anchors * H * W}")
    print(f"  Detections above 0.1 confidence: {mask.sum().item()}")
    print(f"\n  Top detections:")
    for i, (conf_val, cls) in enumerate(zip(detections[:5], classes[:5])):
        print(f"    Detection {i}: class={cls}, confidence={conf_val:.4f}")


# === Exercise 3: Faster R-CNN Custom Dataset (Simulated) ===
# Problem: Demonstrate Faster R-CNN training on synthetic data.

def exercise_3():
    """Simplified Faster R-CNN training with synthetic data."""
    torch.manual_seed(42)

    class SimpleFRCNN(nn.Module):
        """Highly simplified Faster R-CNN for demonstration."""
        def __init__(self, num_classes=3):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d(7),
            )
            self.classifier = nn.Linear(64 * 7 * 7, num_classes)
            self.box_regressor = nn.Linear(64 * 7 * 7, 4)
            self.objectness = nn.Linear(64 * 7 * 7, 2)
            self.rpn_box = nn.Linear(64 * 7 * 7, 4)

        def forward(self, images, targets=None):
            features = self.backbone(images)
            features_flat = features.view(features.size(0), -1)

            # Losses (simplified)
            cls_logits = self.classifier(features_flat)
            box_pred = self.box_regressor(features_flat)
            obj_logits = self.objectness(features_flat)
            rpn_box = self.rpn_box(features_flat)

            if targets is not None:
                loss_cls = F.cross_entropy(cls_logits, targets['labels'])
                loss_box = F.smooth_l1_loss(box_pred, targets['boxes'])
                loss_obj = F.cross_entropy(obj_logits, torch.ones(images.size(0), dtype=torch.long))
                loss_rpn = F.smooth_l1_loss(rpn_box, targets['boxes'])
                return {
                    'loss_classifier': loss_cls,
                    'loss_box_reg': loss_box,
                    'loss_objectness': loss_obj,
                    'loss_rpn_box_reg': loss_rpn,
                }
            return cls_logits, box_pred

    model = SimpleFRCNN(num_classes=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Synthetic dataset
    n = 32
    images = torch.randn(n, 3, 64, 64)
    targets = {
        'labels': torch.randint(0, 3, (n,)),
        'boxes': torch.rand(n, 4) * 64,
    }

    # Training iterations
    for iteration in range(5):
        model.train()
        loss_dict = model(images, targets)
        total_loss = sum(loss_dict.values())
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    print(f"  Faster R-CNN loss components after 5 iterations:")
    for name, val in loss_dict.items():
        print(f"    {name}: {val.item():.4f}")
    print(f"    total: {total_loss.item():.4f}")

    print("\n  Loss components:")
    print("    loss_classifier: Classifies the object in each proposed region")
    print("    loss_box_reg: Refines bounding box coordinates")
    print("    loss_objectness: RPN determines if anchor contains an object")
    print("    loss_rpn_box_reg: RPN refines anchor to better fit the object")


# === Exercise 4: Anchor Generation and IoU Matching ===
# Problem: Generate anchors and compute IoU with ground truth.

def exercise_4():
    """Generate anchors and analyze IoU with a ground truth box."""

    def generate_anchors(feature_h, feature_w, stride=16,
                         scales=(32, 64, 128), ratios=(0.5, 1.0, 2.0)):
        """Generate anchor boxes for a feature map."""
        anchors = []
        for h in range(feature_h):
            for w in range(feature_w):
                cx = (w + 0.5) * stride
                cy = (h + 0.5) * stride
                for scale in scales:
                    for ratio in ratios:
                        # Width and height of anchor
                        aw = scale * np.sqrt(ratio)
                        ah = scale / np.sqrt(ratio)
                        # Convert to (x1, y1, x2, y2)
                        anchors.append([cx - aw / 2, cy - ah / 2,
                                        cx + aw / 2, cy + ah / 2])
        return np.array(anchors)

    def compute_iou(box, anchors):
        """Compute IoU between one box and array of anchors."""
        x1 = np.maximum(box[0], anchors[:, 0])
        y1 = np.maximum(box[1], anchors[:, 1])
        x2 = np.minimum(box[2], anchors[:, 2])
        y2 = np.minimum(box[3], anchors[:, 3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_box = (box[2] - box[0]) * (box[3] - box[1])
        area_anchors = (anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1])
        union = area_box + area_anchors - intersection

        return intersection / (union + 1e-6)

    feature_h, feature_w = 38, 50
    stride = 16
    scales = (32, 64, 128)
    ratios = (0.5, 1.0, 2.0)

    anchors = generate_anchors(feature_h, feature_w, stride, scales, ratios)
    n_shapes = len(scales) * len(ratios)

    print(f"  Feature map: {feature_h}x{feature_w}, stride={stride}")
    print(f"  Total anchors: {len(anchors)} ({feature_h}*{feature_w}*{n_shapes})")
    print(f"  Anchor shapes (9 types):")

    # Show 9 anchor shapes at origin
    for i, (s, r) in enumerate([(s, r) for s in scales for r in ratios]):
        w = s * np.sqrt(r)
        h = s / np.sqrt(r)
        print(f"    Scale {s}, Ratio {r:.1f}: {w:.0f} x {h:.0f}")

    # Ground truth box
    gt_box = [200, 150, 400, 350]  # (x1, y1, x2, y2)
    gt_w, gt_h = gt_box[2] - gt_box[0], gt_box[3] - gt_box[1]

    # Get anchors at position (h=9, w=12)
    pos_h, pos_w = 9, 12
    start_idx = (pos_h * feature_w + pos_w) * n_shapes
    local_anchors = anchors[start_idx:start_idx + n_shapes]

    ious = compute_iou(gt_box, local_anchors)
    best_idx = ious.argmax()

    print(f"\n  GT box: {gt_box} ({gt_w}x{gt_h})")
    print(f"  IoUs at position ({pos_h},{pos_w}):")
    for i, iou in enumerate(ious):
        marker = " <-- best" if i == best_idx else ""
        print(f"    Anchor {i}: IoU={iou:.4f}{marker}")

    print(f"\n  Best anchor matches GT aspect ratio and scale most closely.")
    print(f"  This is why multi-scale, multi-ratio anchors are important.")


if __name__ == "__main__":
    print("=== Exercise 1: IoU and GIoU ===")
    exercise_1()
    print("\n=== Exercise 2: YOLO-style Inference ===")
    exercise_2()
    print("\n=== Exercise 3: Faster R-CNN Training ===")
    exercise_3()
    print("\n=== Exercise 4: Anchor Generation ===")
    exercise_4()
    print("\nAll exercises completed!")
