[Previous: Modern Deep Learning Architectures](./37_Modern_Architectures.md) | [Next: Practical Image Classification Project](./39_Practical_Image_Classification.md)

---

# 38. Object Detection

## Learning Objectives
- Understand the difference between Two-stage vs One-stage detectors
- Learn YOLO architecture and its evolution
- Understand Faster R-CNN structure and RPN
- Grasp DETR (Detection Transformer) concepts
- Practice with PyTorch/Ultralytics

---

## 1. Object Detection Overview

### 1.1 Problem Definition

```
┌─────────────────────────────────────────────────────────────────┐
│                  Computer Vision Task Comparison                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Image Classification                                        │
│     └─ Entire image → Single class                              │
│     └─ Output: "dog"                                            │
│                                                                 │
│  2. Object Detection                                            │
│     └─ Image → Multiple object locations + classes             │
│     └─ Output: [(x1,y1,x2,y2, "dog", 0.95), ...]              │
│                                                                 │
│  3. Semantic Segmentation                                       │
│     └─ Assign class to each pixel                              │
│     └─ Objects of the same class are not distinguished         │
│                                                                 │
│  4. Instance Segmentation                                       │
│     └─ Object detection + pixel mask for each object           │
│     └─ Distinguish individual objects even of the same class   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Detector Classification

```
┌─────────────────────────────────────────────────────────────────┐
│                    Detector Classification                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           Two-Stage Detectors                            │   │
│  │  Stage 1: Region Proposal (generate candidates)         │   │
│  │  Stage 2: Classification + Regression                    │   │
│  │                                                           │   │
│  │  Examples: R-CNN → Fast R-CNN → Faster R-CNN            │   │
│  │      Pros: High accuracy                                 │   │
│  │      Cons: Slow speed                                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           One-Stage Detectors                            │   │
│  │  Single network predicting location + class together    │   │
│  │                                                           │   │
│  │  Examples: YOLO, SSD, RetinaNet, CenterNet              │   │
│  │      Pros: Fast speed, real-time processing possible    │   │
│  │      Cons: Difficulty detecting small objects (improved)│   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           Transformer-based Detectors                    │   │
│  │  DETR, Deformable DETR, RT-DETR                          │   │
│  │  End-to-end training, no NMS required                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 Evaluation Metrics

```python
"""
Object detection evaluation metrics
"""

def calculate_iou(box1, box2):
    """
    IoU (Intersection over Union) calculation

    Args:
        box1, box2: [x1, y1, x2, y2] format

    Returns:
        IoU value (0~1)
    """
    # Intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Intersection area
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

# Example
pred_box = [100, 100, 200, 200]
gt_box = [120, 110, 210, 210]
print(f"IoU: {calculate_iou(pred_box, gt_box):.3f}")  # approximately 0.68


"""
mAP (mean Average Precision) calculation process:

1. For each class:
   - Sort predictions by confidence
   - TP if IoU > threshold, otherwise FP
   - Calculate Precision-Recall curve
   - AP = area under the curve

2. mAP = average of all class APs

COCO dataset metrics:
- mAP@0.5: IoU=0.5 threshold
- mAP@0.75: IoU=0.75 threshold (strict)
- mAP@[.5:.95]: average of IoU from 0.5 to 0.95
"""
```

---

## 2. R-CNN Family

### 2.1 R-CNN Evolution

```
┌─────────────────────────────────────────────────────────────────┐
│                    R-CNN Family Evolution                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  R-CNN (2014):                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌─────────┐   │
│  │ Image    │ → │ Selective│ → │ CNN      │ → │ SVM     │   │
│  │          │    │ Search   │    │ (AlexNet)│    │ Classifier│ │
│  └──────────┘    │ (~2000)  │    └──────────┘    └─────────┘   │
│                  └──────────┘                                   │
│  Problem: ~2000 CNN passes → very slow                          │
│                                                                 │
│  Fast R-CNN (2015):                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌─────────┐   │
│  │ Image    │ → │ CNN      │ → │ RoI      │ → │ FC +    │   │
│  │          │    │ Feature  │    │ Pooling  │    │ Softmax │   │
│  └──────────┘    │ Map      │    └──────────┘    └─────────┘   │
│  Improvement: Single CNN pass, RoI Pooling for region extraction│
│  Problem: Selective Search still slow                           │
│                                                                 │
│  Faster R-CNN (2015):                                           │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌─────────┐   │
│  │ Image    │ → │ Backbone │ → │ RPN      │ → │ Head    │   │
│  │          │    │ (ResNet) │    │          │    │         │   │
│  └──────────┘    └──────────┘    └──────────┘    └─────────┘   │
│  Innovation: RPN makes Region Proposal also learnable           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Faster R-CNN Structure

```python
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class CustomFasterRCNN:
    """Custom Faster R-CNN model"""

    def __init__(self, num_classes: int, pretrained: bool = True):
        """
        Args:
            num_classes: Number of classes including background (e.g., 10 objects → 11)
            pretrained: Use COCO pretrained weights
        """
        # Load pretrained model
        self.model = fasterrcnn_resnet50_fpn_v2(
            weights="DEFAULT" if pretrained else None
        )

        # Replace box predictor (adjust for number of classes)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes
        )

    def get_model(self):
        return self.model


def train_faster_rcnn():
    """Faster R-CNN training example"""

    # Create model (background + 10 classes)
    model = CustomFasterRCNN(num_classes=11).get_model()
    model.train()

    # Dummy data
    images = [torch.rand(3, 600, 800) for _ in range(2)]

    targets = [
        {
            "boxes": torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]]),
            "labels": torch.tensor([1, 2]),  # class ID
        },
        {
            "boxes": torch.tensor([[50, 50, 150, 150]]),
            "labels": torch.tensor([3]),
        }
    ]

    # Forward pass (returns loss in training mode)
    loss_dict = model(images, targets)

    # Loss types:
    # - loss_classifier: class classification loss
    # - loss_box_reg: box regression loss
    # - loss_objectness: RPN object/non-object classification
    # - loss_rpn_box_reg: RPN box regression

    total_loss = sum(loss for loss in loss_dict.values())
    print(f"Total loss: {total_loss.item():.4f}")

    return loss_dict


def inference_faster_rcnn(model, image, threshold=0.5):
    """Faster R-CNN inference"""

    model.eval()

    with torch.no_grad():
        predictions = model([image])

    pred = predictions[0]

    # Filter predictions above threshold
    keep = pred["scores"] > threshold

    result = {
        "boxes": pred["boxes"][keep],
        "labels": pred["labels"][keep],
        "scores": pred["scores"][keep],
    }

    return result
```

### 2.3 RPN (Region Proposal Network)

```python
"""
RPN key concepts:

1. Anchor Boxes:
   - Pre-defined boxes of multiple sizes/ratios at each location
   - Example: 3 scales × 3 ratios = 9 anchors

2. Outputs:
   - objectness score: object presence probability (2-class)
   - box regression: anchor → actual box transformation

3. Training:
   - Positive: anchors with IoU > 0.7
   - Negative: anchors with IoU < 0.3
   - Ignored: between 0.3~0.7
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleRPN(nn.Module):
    """Simplified RPN implementation"""

    def __init__(
        self,
        in_channels: int = 256,
        num_anchors: int = 9,  # 3 scales × 3 ratios
    ):
        super().__init__()

        # 3×3 conv for feature processing
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)

        # objectness prediction (object/background)
        self.objectness = nn.Conv2d(in_channels, num_anchors * 2, 1)

        # bbox regression (dx, dy, dw, dh)
        self.bbox_reg = nn.Conv2d(in_channels, num_anchors * 4, 1)

    def forward(self, feature_map):
        """
        Args:
            feature_map: (B, C, H, W)

        Returns:
            objectness: (B, num_anchors*2, H, W)
            bbox_deltas: (B, num_anchors*4, H, W)
        """
        x = F.relu(self.conv(feature_map))

        objectness = self.objectness(x)
        bbox_deltas = self.bbox_reg(x)

        return objectness, bbox_deltas


def generate_anchors(
    feature_size: tuple,
    anchor_scales: list = [128, 256, 512],
    anchor_ratios: list = [0.5, 1.0, 2.0],
    stride: int = 16
):
    """
    Generate anchor boxes

    Args:
        feature_size: (H, W) feature map size
        anchor_scales: square root of anchor area
        anchor_ratios: width/height ratios
        stride: downsampling ratio from original image

    Returns:
        anchors: (H*W*num_anchors, 4) anchor coordinates
    """
    H, W = feature_size
    anchors = []

    for h in range(H):
        for w in range(W):
            # feature map position → original image coordinates
            cx = (w + 0.5) * stride
            cy = (h + 0.5) * stride

            for scale in anchor_scales:
                for ratio in anchor_ratios:
                    # width/height based on ratio
                    anchor_w = scale * (ratio ** 0.5)
                    anchor_h = scale / (ratio ** 0.5)

                    # (x1, y1, x2, y2) format
                    anchors.append([
                        cx - anchor_w / 2,
                        cy - anchor_h / 2,
                        cx + anchor_w / 2,
                        cy + anchor_h / 2
                    ])

    return torch.tensor(anchors)

# Example
anchors = generate_anchors((38, 50))  # 600×800 image, stride=16
print(f"Generated {len(anchors)} anchors")  # 38*50*9 = 17,100
```

---

## 3. YOLO (You Only Look Once)

### 3.1 YOLO Evolution

```
┌─────────────────────────────────────────────────────────────────┐
│                    YOLO Version Comparison                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  YOLOv1 (2016): Grid-based detection with single CNN            │
│  YOLOv2 (2017): Batch Norm, Anchor Boxes introduction           │
│  YOLOv3 (2018): Darknet-53, FPN, 3-scale predictions            │
│  YOLOv4 (2020): CSPDarknet, SPP, PANet                          │
│  YOLOv5 (2020): PyTorch implementation, Ultralytics             │
│  YOLOv6 (2022): Speed optimization, EfficientRep                │
│  YOLOv7 (2022): E-ELAN, Auxiliary Head                          │
│  YOLOv8 (2023): Unified Framework, Anchor-free                  │
│  YOLOv9 (2024): GELAN, PGI                                      │
│  YOLOv10 (2024): NMS-free, Dual Assignments                     │
│  YOLO11 (2024): Faster and more accurate version                │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Performance (COCO val2017)    mAP50-95   Speed (ms)   │   │
│  │  ─────────────────────────────────────────────────────  │   │
│  │  YOLOv8n                         37.3        1.2       │   │
│  │  YOLOv8s                         44.9        1.9       │   │
│  │  YOLOv8m                         50.2        4.3       │   │
│  │  YOLOv8l                         52.9        6.7       │   │
│  │  YOLOv8x                         53.9        9.8       │   │
│  │  YOLO11x                         54.7        11.3      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Ultralytics YOLOv8 Practice

```python
from ultralytics import YOLO
import torch

# ===============================
# 1. Model Loading
# ===============================

# Load pretrained model
model = YOLO("yolov8n.pt")  # nano version (fastest)
# model = YOLO("yolov8s.pt")  # small
# model = YOLO("yolov8m.pt")  # medium
# model = YOLO("yolov8l.pt")  # large
# model = YOLO("yolov8x.pt")  # extra-large

# Create empty model for training
# model = YOLO("yolov8n.yaml")


# ===============================
# 2. Inference
# ===============================

def detect_objects(image_path: str, conf_threshold: float = 0.25):
    """Object detection on image"""

    results = model(image_path, conf=conf_threshold)

    for result in results:
        boxes = result.boxes

        print(f"Number of detected objects: {len(boxes)}")

        for box in boxes:
            # Coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Class and confidence
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls]

            print(f"  {class_name}: {conf:.2f} at ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")

    return results

# Usage example
# results = detect_objects("image.jpg")

# Visualize results
# results[0].show()  # Display image
# results[0].save("result.jpg")  # Save


# ===============================
# 3. Video/Webcam Detection
# ===============================

def detect_video(source: str = 0):
    """
    Real-time detection on video or webcam

    Args:
        source: 0=webcam, or video file path
    """
    results = model(source, stream=True)  # return as generator

    for result in results:
        # Per-frame processing
        annotated_frame = result.plot()  # frame with boxes drawn

        # Can display with cv2.imshow() etc.
        yield annotated_frame


# ===============================
# 4. Custom Dataset Training
# ===============================

def train_custom_model():
    """Train YOLO on custom dataset"""

    # Dataset yaml file example (data.yaml):
    """
    path: /path/to/dataset
    train: images/train
    val: images/val

    names:
      0: cat
      1: dog
      2: bird
    """

    # Model training
    model = YOLO("yolov8n.pt")

    results = model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,  # GPU 0, or "cpu"
        patience=50,  # Early stopping
        save=True,
        project="runs/detect",
        name="custom_model",
    )

    return results


# ===============================
# 5. Model Export
# ===============================

def export_model():
    """Export model to various formats"""

    model = YOLO("yolov8n.pt")

    # Export to ONNX
    model.export(format="onnx")

    # Export to TensorRT (GPU inference optimization)
    # model.export(format="engine")

    # Export to CoreML (Apple)
    # model.export(format="coreml")

    # Export to TFLite (mobile)
    # model.export(format="tflite")
```

### 3.3 YOLOv8 Loss Function

```python
"""
YOLOv8 loss function components:

1. Box Loss (CIoU Loss):
   - Accuracy of box location and size
   - CIoU = IoU - (distance penalty + aspect ratio penalty)

2. Classification Loss (BCE):
   - Binary cross entropy for each class
   - Focal Loss variant can be used

3. DFL Loss (Distribution Focal Loss):
   - Distribution prediction for box boundaries
   - New regression method in YOLOv8

Total Loss = λ_box * L_box + λ_cls * L_cls + λ_dfl * L_dfl
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def ciou_loss(pred_boxes, target_boxes, eps=1e-7):
    """
    Complete IoU Loss

    Args:
        pred_boxes: (N, 4) predicted boxes [x1, y1, x2, y2]
        target_boxes: (N, 4) ground truth boxes

    Returns:
        CIoU loss
    """
    # IoU calculation
    inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                 torch.clamp(inter_y2 - inter_y1, min=0)

    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * \
                (pred_boxes[:, 3] - pred_boxes[:, 1])
    target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * \
                  (target_boxes[:, 3] - target_boxes[:, 1])

    union_area = pred_area + target_area - inter_area
    iou = inter_area / (union_area + eps)

    # Center point distance
    pred_cx = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
    pred_cy = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
    target_cx = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
    target_cy = (target_boxes[:, 1] + target_boxes[:, 3]) / 2

    center_dist_sq = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2

    # Diagonal distance (enclosing box)
    enclose_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    enclose_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    enclose_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    enclose_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])

    enclose_diag_sq = (enclose_x2 - enclose_x1) ** 2 + \
                      (enclose_y2 - enclose_y1) ** 2

    # Aspect ratio consistency
    pred_w = pred_boxes[:, 2] - pred_boxes[:, 0]
    pred_h = pred_boxes[:, 3] - pred_boxes[:, 1]
    target_w = target_boxes[:, 2] - target_boxes[:, 0]
    target_h = target_boxes[:, 3] - target_boxes[:, 1]

    v = (4 / (torch.pi ** 2)) * \
        (torch.atan(target_w / (target_h + eps)) -
         torch.atan(pred_w / (pred_h + eps))) ** 2

    alpha = v / (1 - iou + v + eps)

    # CIoU
    ciou = iou - (center_dist_sq / (enclose_diag_sq + eps)) - alpha * v

    return 1 - ciou
```

---

## 4. DETR (Detection Transformer)

### 4.1 DETR Concept

```
┌─────────────────────────────────────────────────────────────────┐
│                    DETR Architecture                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Traditional detector issues:                                   │
│  - Anchor design required                                       │
│  - NMS (Non-Maximum Suppression) post-processing required       │
│  - Complex pipeline                                             │
│                                                                 │
│  DETR innovations:                                              │
│  - End-to-end training                                          │
│  - Direct object prediction with Object Queries                │
│  - Training with Hungarian Matching                             │
│  - No NMS required                                              │
│                                                                 │
│  ┌────────────┐    ┌────────────┐    ┌────────────────────────┐ │
│  │ Backbone   │ → │ Transformer│ → │ FFN Heads             │ │
│  │ (ResNet)   │    │ Encoder/   │    │ (class + box)         │ │
│  │            │    │ Decoder    │    │                        │ │
│  └────────────┘    └────────────┘    └────────────────────────┘ │
│       ↓                 ↓                      ↓                │
│  Feature Map    Object Queries (100)      100 prediction outputs│
│  + Positional   ↓                                               │
│    Encoding     Self-attention + Cross-attention                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 DETR Implementation

```python
import torch
import torch.nn as nn
from torchvision.models import resnet50
import torch.nn.functional as F

class DETR(nn.Module):
    """
    Simplified DETR implementation
    """

    def __init__(
        self,
        num_classes: int,
        num_queries: int = 100,
        hidden_dim: int = 256,
        nheads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
    ):
        super().__init__()

        # Backbone
        backbone = resnet50(weights="DEFAULT")
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        # Feature map → hidden_dim
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            batch_first=True,
        )

        # Object Queries (learned embeddings)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Output heads
        self.class_head = nn.Linear(hidden_dim, num_classes + 1)  # +1 for no-object
        self.bbox_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),  # cx, cy, w, h
            nn.Sigmoid(),
        )

        # Positional Encoding
        self.row_embed = nn.Embedding(50, hidden_dim // 2)
        self.col_embed = nn.Embedding(50, hidden_dim // 2)

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) input image

        Returns:
            class_logits: (B, num_queries, num_classes+1)
            bbox_pred: (B, num_queries, 4)
        """
        B = x.shape[0]

        # Backbone feature extraction
        features = self.backbone(x)  # (B, 2048, H/32, W/32)
        features = self.conv(features)  # (B, 256, H/32, W/32)

        _, _, H, W = features.shape

        # Generate Positional Encoding
        pos_embed = self._get_positional_encoding(H, W, features.device)

        # Flatten for Transformer
        src = features.flatten(2).permute(0, 2, 1)  # (B, H*W, 256)
        src = src + pos_embed.flatten(0, 1).unsqueeze(0).expand(B, -1, -1)

        # Object Queries
        query_embed = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        # Transformer
        tgt = torch.zeros_like(query_embed)
        hs = self.transformer(src, tgt + query_embed)  # (B, num_queries, 256)

        # Predictions
        class_logits = self.class_head(hs)
        bbox_pred = self.bbox_head(hs)

        return class_logits, bbox_pred

    def _get_positional_encoding(self, H, W, device):
        """Generate 2D Positional Encoding"""
        i = torch.arange(W, device=device)
        j = torch.arange(H, device=device)

        x_embed = self.col_embed(i)  # (W, 128)
        y_embed = self.row_embed(j)  # (H, 128)

        pos = torch.cat([
            x_embed.unsqueeze(0).expand(H, -1, -1),
            y_embed.unsqueeze(1).expand(-1, W, -1),
        ], dim=-1)  # (H, W, 256)

        return pos


# Hungarian Matching Loss (simplified)
class HungarianMatcher:
    """
    Optimal matching between predictions and GT

    Cost = λ_cls * L_cls + λ_box * L_box + λ_giou * L_giou
    """

    def __init__(self, cost_class=1, cost_bbox=5, cost_giou=2):
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    def __call__(self, outputs, targets):
        """
        Perform bipartite matching using
        scipy.optimize.linear_sum_assignment
        """
        # Implementation omitted (uses scipy)
        pass
```

### 4.3 RT-DETR (Real-Time DETR)

```python
from ultralytics import RTDETR

# RT-DETR usage (Ultralytics)
model = RTDETR("rtdetr-l.pt")

# Inference
results = model("image.jpg")

# Training
model.train(data="coco.yaml", epochs=100)

"""
RT-DETR features:
- Maintains DETR's end-to-end advantages
- Real-time inference possible (YOLO-level speed)
- Efficient Hybrid Encoder
- IoU-aware Query Selection
"""
```

---

## 5. Instance Segmentation

### 5.1 Mask R-CNN

```python
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def create_mask_rcnn(num_classes: int):
    """
    Create custom Mask R-CNN

    Mask R-CNN = Faster R-CNN + Mask Head
    """
    model = maskrcnn_resnet50_fpn_v2(weights="DEFAULT")

    # Replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model


def inference_mask_rcnn(model, image, threshold=0.5):
    """Mask R-CNN inference"""

    model.eval()

    with torch.no_grad():
        predictions = model([image])

    pred = predictions[0]
    keep = pred["scores"] > threshold

    result = {
        "boxes": pred["boxes"][keep],
        "labels": pred["labels"][keep],
        "scores": pred["scores"][keep],
        "masks": pred["masks"][keep],  # (N, 1, H, W) soft masks
    }

    # Convert to hard masks
    result["masks"] = (result["masks"] > 0.5).squeeze(1)  # (N, H, W)

    return result


# YOLOv8-seg usage
from ultralytics import YOLO

seg_model = YOLO("yolov8n-seg.pt")
results = seg_model("image.jpg")

# Extract masks from results
for result in results:
    if result.masks is not None:
        masks = result.masks.data  # (N, H, W)
```

### 5.2 SAM (Segment Anything Model)

```python
from segment_anything import sam_model_registry, SamPredictor
import numpy as np

# Load SAM model
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
predictor = SamPredictor(sam)

# Set image
predictor.set_image(image)  # (H, W, 3) numpy array

# Segmentation with point prompt
input_point = np.array([[500, 375]])  # click coordinates
input_label = np.array([1])  # 1 = foreground

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,  # return 3 mask candidates
)

# Segmentation with box prompt
input_box = np.array([100, 100, 400, 400])  # x1, y1, x2, y2

masks, scores, logits = predictor.predict(
    box=input_box,
    multimask_output=False,
)

"""
SAM features:
- Promptable segmentation (point, box, text)
- Zero-shot generalization
- Very high segmentation quality
- Large model with slow speed → lightweight versions like MobileSAM, FastSAM exist
"""
```

---

## 6. Practical Tips

### 6.1 Dataset Formats

```python
"""
Major dataset formats:

1. COCO Format:
   - All annotations stored in annotations.json
   - Images and annotations separated

2. YOLO Format:
   - .txt file for each image
   - class x_center y_center width height (normalized)

3. Pascal VOC Format:
   - XML file for each image annotation
"""

# YOLO format example (labels/train/image001.txt)
"""
0 0.5 0.5 0.2 0.3
1 0.3 0.7 0.1 0.15
"""

# COCO to YOLO conversion
def coco_to_yolo(coco_box, img_width, img_height):
    """
    COCO: [x_min, y_min, width, height]
    YOLO: [x_center, y_center, width, height] (normalized)
    """
    x, y, w, h = coco_box

    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height

    return [x_center, y_center, w_norm, h_norm]
```

### 6.2 Training Tips

```python
"""
Object detection training checklist:

1. Data Quality
   - Verify label accuracy
   - Handle class imbalance (Focal Loss, oversampling)
   - Use appropriate augmentation

2. Hyperparameters
   - Learning rate: 1e-4 ~ 1e-3 (starting from pretrained)
   - Batch size: as large as possible within GPU memory
   - Image size: use model default (YOLO: 640)

3. Augmentation Strategy
   - Mosaic: compose 4 images (YOLO)
   - MixUp: image blending
   - Basic: Flip, Scale, Color Jitter

4. Model Selection
   - Real-time: YOLO (YOLOv8n, YOLOv8s)
   - Accuracy: Faster R-CNN, DETR
   - Segmentation: YOLOv8-seg, Mask R-CNN
"""

# Ultralytics training example
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,

    # Augmentation
    mosaic=1.0,      # Mosaic probability
    mixup=0.0,       # MixUp probability
    hsv_h=0.015,     # Hue augmentation
    hsv_s=0.7,       # Saturation augmentation
    hsv_v=0.4,       # Value augmentation
    degrees=0.0,     # Rotation
    translate=0.1,   # Translation
    scale=0.5,       # Scale
    fliplr=0.5,      # Horizontal flip

    # Regularization
    weight_decay=0.0005,

    # Training schedule
    warmup_epochs=3,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    lr0=0.01,        # Initial learning rate
    lrf=0.01,        # Final learning rate ratio
)
```

---

## Summary

### Detector Selection Guide

| Requirements | Recommended Model |
|--------------|-------------------|
| Real-time (30+ FPS) | YOLOv8n/s |
| High accuracy | YOLOv8x, Faster R-CNN |
| Small objects | YOLO + SAHI, RetinaNet |
| Instance segmentation | YOLOv8-seg, Mask R-CNN |
| End-to-end | DETR, RT-DETR |
| Zero-shot | Grounding DINO, SAM |

### Key Concepts Summary

| Concept | Description |
|---------|-------------|
| IoU | Box overlap degree (0~1) |
| mAP | mean Average Precision (accuracy metric) |
| NMS | Non-Maximum Suppression (duplicate box removal) |
| Anchor | Pre-defined reference boxes |
| FPN | Multi-scale feature extraction |
| GIoU/CIoU | Improved IoU loss functions |

### Next Steps
- [24_Semantic_Segmentation.md](24_Semantic_Segmentation.md): Semantic Segmentation
- [Computer_Vision/19_DNN_Module.md](../../Computer_Vision/19_DNN_Module.md): OpenCV DNN

---

## Exercises

### Exercise 1: Understand IoU and Implement GIoU

1. Using the provided `calculate_iou` function, compute the IoU for the following pairs of boxes: `(a) [0,0,100,100] vs [50,50,150,150]`, `(b) [0,0,100,100] vs [200,200,300,300]`, `(c) [0,0,100,100] vs [0,0,100,100]`. Verify the results match your intuition.
2. Implement GIoU (Generalized IoU): `GIoU = IoU - (area_enclosing - area_union) / area_enclosing`, where the enclosing box is the smallest box containing both predicted and ground-truth boxes.
3. For the non-overlapping pair `(b)`, show that GIoU gives a more informative gradient signal than standard IoU. Explain why this matters for training.

### Exercise 2: Run YOLOv8 Inference on Your Own Image

Using the Ultralytics library:
1. Load `yolov8n.pt` (nano model) and run inference on a photograph of your choice containing multiple objects.
2. Print the detected object names, confidence scores, and bounding box coordinates.
3. Save the annotated result image using `results[0].save("my_result.jpg")`.
4. Now run the same image through `yolov8x.pt` (extra-large model). Compare the number of detections and confidence scores between nano and extra-large. What trade-offs does this illustrate?

### Exercise 3: Fine-tune Faster R-CNN on a Custom Dataset

Using `torchvision`'s Faster R-CNN:
1. Create a small custom dataset of at least 2 classes (you can use a subset of COCO or create a synthetic dataset with random rectangles).
2. Implement a `torch.utils.data.Dataset` class that returns images and targets in the Faster R-CNN format: `{"boxes": tensor, "labels": tensor}`.
3. Instantiate `CustomFasterRCNN(num_classes=3)` (background + 2 classes) and run 5 training iterations.
4. Verify that `loss_dict` contains four keys (`loss_classifier`, `loss_box_reg`, `loss_objectness`, `loss_rpn_box_reg`) and describe what each loss component is responsible for.

### Exercise 4: Analyze Anchor Generation and IoU Matching

1. Using `generate_anchors`, generate all anchor boxes for a `(38, 50)` feature map with `stride=16`.
2. Plot all 9 anchor shapes (3 scales × 3 ratios) centered at the feature map origin on a blank 600×800 canvas.
3. For a ground-truth box `[200, 150, 400, 350]`, compute the IoU between this box and all 9 anchor shapes at position `(h=9, w=12)` (a nearby feature map cell).
4. Which anchor has the highest IoU? What does this tell you about the importance of choosing anchor sizes that match the expected object scales in your dataset?

---

## References

### Papers
- "Faster R-CNN" (Ren et al., 2015)
- "YOLO: You Only Look Once" (Redmon et al., 2016)
- "DETR: End-to-End Object Detection with Transformers" (Carion et al., 2020)
- "Segment Anything" (Kirillov et al., 2023)

### Code & Resources
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [torchvision detection](https://pytorch.org/vision/stable/models.html#object-detection)
- [COCO Dataset](https://cocodataset.org/)
- [Roboflow](https://roboflow.com/) - Dataset management
