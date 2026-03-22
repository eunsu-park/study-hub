[이전: 현대 딥러닝 아키텍처](./37_Modern_Architectures.md) | [다음: 실전 이미지 분류 프로젝트](./39_Practical_Image_Classification.md)

---

# 38. 객체 탐지 (Object Detection)

## 학습 목표
- Two-stage vs One-stage 탐지기 차이 이해
- YOLO 아키텍처와 발전 과정 학습
- Faster R-CNN의 구조와 RPN 이해
- DETR (Detection Transformer) 개념 파악
- PyTorch/Ultralytics로 실습

---

## 1. 객체 탐지 개요

### 1.1 문제 정의

```
┌─────────────────────────────────────────────────────────────────┐
│                    Computer Vision 태스크 비교                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 이미지 분류 (Image Classification)                           │
│     └─ 이미지 전체 → 하나의 클래스                                │
│     └─ 출력: "강아지"                                            │
│                                                                 │
│  2. 객체 탐지 (Object Detection)                                 │
│     └─ 이미지 → 여러 객체의 위치 + 클래스                         │
│     └─ 출력: [(x1,y1,x2,y2, "강아지", 0.95), ...]              │
│                                                                 │
│  3. 시맨틱 분할 (Semantic Segmentation)                          │
│     └─ 픽셀마다 클래스 할당                                       │
│     └─ 같은 클래스의 객체들은 구분 안 됨                           │
│                                                                 │
│  4. 인스턴스 분할 (Instance Segmentation)                        │
│     └─ 객체 탐지 + 각 객체의 픽셀 마스크                          │
│     └─ 같은 클래스라도 개별 객체 구분                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 탐지기 분류

```
┌─────────────────────────────────────────────────────────────────┐
│                    탐지기 분류 체계                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           Two-Stage Detectors                            │   │
│  │  1단계: Region Proposal (후보 영역 생성)                   │   │
│  │  2단계: Classification + Regression                       │   │
│  │                                                           │   │
│  │  예: R-CNN → Fast R-CNN → Faster R-CNN                   │   │
│  │      장점: 높은 정확도                                     │   │
│  │      단점: 느린 속도                                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           One-Stage Detectors                            │   │
│  │  단일 네트워크로 위치 + 클래스 동시 예측                    │   │
│  │                                                           │   │
│  │  예: YOLO, SSD, RetinaNet, CenterNet                     │   │
│  │      장점: 빠른 속도, 실시간 처리 가능                     │   │
│  │      단점: 작은 객체 탐지 어려움 (개선됨)                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           Transformer-based Detectors                    │   │
│  │  DETR, Deformable DETR, RT-DETR                          │   │
│  │  End-to-end 학습, NMS 불필요                              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 평가 지표

```python
"""
객체 탐지 평가 지표
"""

def calculate_iou(box1, box2):
    """
    IoU (Intersection over Union) 계산

    Args:
        box1, box2: [x1, y1, x2, y2] 형식

    Returns:
        IoU 값 (0~1)
    """
    # 교집합 좌표
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 교집합 면적
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    # 합집합 면적
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

# 예시
pred_box = [100, 100, 200, 200]
gt_box = [120, 110, 210, 210]
print(f"IoU: {calculate_iou(pred_box, gt_box):.3f}")  # 약 0.68


"""
mAP (mean Average Precision) 계산 과정:

1. 각 클래스별로:
   - 예측을 confidence 순으로 정렬
   - IoU > threshold인 경우 TP, 아니면 FP
   - Precision-Recall 곡선 계산
   - AP = 곡선 아래 면적

2. mAP = 모든 클래스 AP의 평균

COCO 데이터셋 기준:
- mAP@0.5: IoU=0.5 기준
- mAP@0.75: IoU=0.75 기준 (엄격)
- mAP@[.5:.95]: 0.5~0.95 IoU의 평균
"""
```

---

## 2. R-CNN 계열

### 2.1 R-CNN의 발전

```
┌─────────────────────────────────────────────────────────────────┐
│                    R-CNN Family 발전                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  R-CNN (2014):                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌─────────┐   │
│  │ 이미지    │ → │ Selective│ → │ CNN      │ → │ SVM     │   │
│  │          │    │ Search   │    │ (AlexNet)│    │ 분류기   │   │
│  └──────────┘    │ (~2000개)│    └──────────┘    └─────────┘   │
│                  └──────────┘                                   │
│  문제점: ~2000번 CNN 통과 → 매우 느림                            │
│                                                                 │
│  Fast R-CNN (2015):                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌─────────┐   │
│  │ 이미지    │ → │ CNN      │ → │ RoI      │ → │ FC +    │   │
│  │          │    │ Feature  │    │ Pooling  │    │ Softmax │   │
│  └──────────┘    │ Map      │    └──────────┘    └─────────┘   │
│  개선: CNN 1번만 통과, RoI Pooling으로 후보 영역 추출              │
│  문제점: Selective Search 여전히 느림                            │
│                                                                 │
│  Faster R-CNN (2015):                                           │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌─────────┐   │
│  │ 이미지    │ → │ Backbone │ → │ RPN      │ → │ Head    │   │
│  │          │    │ (ResNet) │    │          │    │         │   │
│  └──────────┘    └──────────┘    └──────────┘    └─────────┘   │
│  혁신: RPN으로 Region Proposal도 학습                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Faster R-CNN 구조

```python
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class CustomFasterRCNN:
    """Faster R-CNN 커스텀 모델"""

    def __init__(self, num_classes: int, pretrained: bool = True):
        """
        Args:
            num_classes: 배경 포함 클래스 수 (예: 10개 객체 → 11)
            pretrained: COCO 사전학습 가중치 사용
        """
        # 사전학습된 모델 로드
        self.model = fasterrcnn_resnet50_fpn_v2(
            weights="DEFAULT" if pretrained else None
        )

        # Box predictor 교체 (클래스 수 맞춤)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes
        )

    def get_model(self):
        return self.model


def train_faster_rcnn():
    """Faster R-CNN 학습 예제"""

    # 모델 생성 (배경 + 10개 클래스)
    model = CustomFasterRCNN(num_classes=11).get_model()
    model.train()

    # 가상 데이터
    images = [torch.rand(3, 600, 800) for _ in range(2)]

    targets = [
        {
            "boxes": torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]]),
            "labels": torch.tensor([1, 2]),  # 클래스 ID
        },
        {
            "boxes": torch.tensor([[50, 50, 150, 150]]),
            "labels": torch.tensor([3]),
        }
    ]

    # 순전파 (학습 모드에서는 loss 반환)
    loss_dict = model(images, targets)

    # 손실 종류:
    # - loss_classifier: 클래스 분류 손실
    # - loss_box_reg: 박스 회귀 손실
    # - loss_objectness: RPN의 객체/비객체 분류
    # - loss_rpn_box_reg: RPN 박스 회귀

    total_loss = sum(loss for loss in loss_dict.values())
    print(f"Total loss: {total_loss.item():.4f}")

    return loss_dict


def inference_faster_rcnn(model, image, threshold=0.5):
    """Faster R-CNN 추론"""

    model.eval()

    with torch.no_grad():
        predictions = model([image])

    pred = predictions[0]

    # threshold 이상인 예측만 필터링
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
RPN 핵심 개념:

1. Anchor Boxes:
   - 각 위치에서 여러 크기/비율의 박스 미리 정의
   - 예: 3개 크기 × 3개 비율 = 9개 anchor

2. 출력:
   - objectness score: 객체 존재 확률 (2-class)
   - box regression: anchor → 실제 박스 변환

3. 학습:
   - Positive: IoU > 0.7인 anchor
   - Negative: IoU < 0.3인 anchor
   - 무시: 0.3~0.7 사이
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleRPN(nn.Module):
    """간략화된 RPN 구현"""

    def __init__(
        self,
        in_channels: int = 256,
        num_anchors: int = 9,  # 3 scales × 3 ratios
    ):
        super().__init__()

        # 3×3 conv로 feature 처리
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)

        # objectness 예측 (객체/배경)
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
    Anchor 박스 생성

    Args:
        feature_size: (H, W) feature map 크기
        anchor_scales: anchor 면적의 제곱근
        anchor_ratios: 가로/세로 비율
        stride: 원본 이미지 대비 축소 비율

    Returns:
        anchors: (H*W*num_anchors, 4) 형태의 anchor 좌표
    """
    H, W = feature_size
    anchors = []

    for h in range(H):
        for w in range(W):
            # feature map 위치 → 원본 이미지 좌표
            cx = (w + 0.5) * stride
            cy = (h + 0.5) * stride

            for scale in anchor_scales:
                for ratio in anchor_ratios:
                    # 비율에 따른 너비/높이
                    anchor_w = scale * (ratio ** 0.5)
                    anchor_h = scale / (ratio ** 0.5)

                    # (x1, y1, x2, y2) 형식
                    anchors.append([
                        cx - anchor_w / 2,
                        cy - anchor_h / 2,
                        cx + anchor_w / 2,
                        cy + anchor_h / 2
                    ])

    return torch.tensor(anchors)

# 예시
anchors = generate_anchors((38, 50))  # 600×800 이미지, stride=16
print(f"Generated {len(anchors)} anchors")  # 38*50*9 = 17,100개
```

---

## 3. YOLO (You Only Look Once)

### 3.1 YOLO 발전사

```
┌─────────────────────────────────────────────────────────────────┐
│                    YOLO 버전 비교                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  YOLOv1 (2016): 단일 CNN으로 그리드 기반 탐지                     │
│  YOLOv2 (2017): Batch Norm, Anchor Boxes 도입                   │
│  YOLOv3 (2018): Darknet-53, FPN, 3개 스케일 예측                 │
│  YOLOv4 (2020): CSPDarknet, SPP, PANet                          │
│  YOLOv5 (2020): PyTorch 구현, Ultralytics                       │
│  YOLOv6 (2022): 속도 최적화, EfficientRep                       │
│  YOLOv7 (2022): E-ELAN, Auxiliary Head                          │
│  YOLOv8 (2023): Unified Framework, Anchor-free                  │
│  YOLOv9 (2024): GELAN, PGI                                      │
│  YOLOv10 (2024): NMS-free, Dual Assignments                     │
│  YOLO11 (2024): 더 빠르고 정확한 버전                             │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  성능 (COCO val2017)           mAP50-95   Speed (ms)   │   │
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

### 3.2 Ultralytics YOLOv8 실습

```python
from ultralytics import YOLO
import torch

# ===============================
# 1. 모델 로드
# ===============================

# 사전학습 모델 로드
model = YOLO("yolov8n.pt")  # nano 버전 (가장 빠름)
# model = YOLO("yolov8s.pt")  # small
# model = YOLO("yolov8m.pt")  # medium
# model = YOLO("yolov8l.pt")  # large
# model = YOLO("yolov8x.pt")  # extra-large

# 빈 모델 생성 후 학습
# model = YOLO("yolov8n.yaml")


# ===============================
# 2. 추론 (Inference)
# ===============================

def detect_objects(image_path: str, conf_threshold: float = 0.25):
    """이미지에서 객체 탐지"""

    results = model(image_path, conf=conf_threshold)

    for result in results:
        boxes = result.boxes

        print(f"탐지된 객체 수: {len(boxes)}")

        for box in boxes:
            # 좌표
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # 클래스와 confidence
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls]

            print(f"  {class_name}: {conf:.2f} at ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")

    return results

# 사용 예시
# results = detect_objects("image.jpg")

# 결과 시각화
# results[0].show()  # 이미지 표시
# results[0].save("result.jpg")  # 저장


# ===============================
# 3. 비디오/웹캠 탐지
# ===============================

def detect_video(source: str = 0):
    """
    비디오 또는 웹캠에서 실시간 탐지

    Args:
        source: 0=웹캠, 또는 비디오 파일 경로
    """
    results = model(source, stream=True)  # generator로 반환

    for result in results:
        # 프레임별 처리
        annotated_frame = result.plot()  # 박스 그려진 프레임

        # 여기서 cv2.imshow() 등으로 표시 가능
        yield annotated_frame


# ===============================
# 4. 커스텀 데이터셋 학습
# ===============================

def train_custom_model():
    """커스텀 데이터셋으로 YOLO 학습"""

    # 데이터셋 yaml 파일 예시 (data.yaml):
    """
    path: /path/to/dataset
    train: images/train
    val: images/val

    names:
      0: cat
      1: dog
      2: bird
    """

    # 모델 학습
    model = YOLO("yolov8n.pt")

    results = model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,  # GPU 0, 또는 "cpu"
        patience=50,  # Early stopping
        save=True,
        project="runs/detect",
        name="custom_model",
    )

    return results


# ===============================
# 5. 모델 내보내기
# ===============================

def export_model():
    """다양한 형식으로 모델 내보내기"""

    model = YOLO("yolov8n.pt")

    # ONNX로 내보내기
    model.export(format="onnx")

    # TensorRT로 내보내기 (GPU 추론 최적화)
    # model.export(format="engine")

    # CoreML로 내보내기 (Apple)
    # model.export(format="coreml")

    # TFLite로 내보내기 (모바일)
    # model.export(format="tflite")
```

### 3.3 YOLOv8 손실 함수

```python
"""
YOLOv8 손실 함수 구성:

1. Box Loss (CIoU Loss):
   - 박스 위치와 크기의 정확도
   - CIoU = IoU - (거리 페널티 + 종횡비 페널티)

2. Classification Loss (BCE):
   - 각 클래스에 대한 이진 교차 엔트로피
   - Focal Loss 변형 사용 가능

3. DFL Loss (Distribution Focal Loss):
   - 박스 경계의 분포 예측
   - YOLOv8의 새로운 회귀 방식

Total Loss = λ_box * L_box + λ_cls * L_cls + λ_dfl * L_dfl
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def ciou_loss(pred_boxes, target_boxes, eps=1e-7):
    """
    Complete IoU Loss

    Args:
        pred_boxes: (N, 4) 예측 박스 [x1, y1, x2, y2]
        target_boxes: (N, 4) 정답 박스

    Returns:
        CIoU loss
    """
    # IoU 계산
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

    # 중심점 거리
    pred_cx = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
    pred_cy = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
    target_cx = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
    target_cy = (target_boxes[:, 1] + target_boxes[:, 3]) / 2

    center_dist_sq = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2

    # 대각선 거리 (enclosing box)
    enclose_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    enclose_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    enclose_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    enclose_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])

    enclose_diag_sq = (enclose_x2 - enclose_x1) ** 2 + \
                      (enclose_y2 - enclose_y1) ** 2

    # 종횡비 일관성
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

### 4.1 DETR 개념

```
┌─────────────────────────────────────────────────────────────────┐
│                    DETR 아키텍처                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  기존 탐지기 문제점:                                              │
│  - Anchor 설계 필요                                              │
│  - NMS (Non-Maximum Suppression) 후처리 필요                     │
│  - 복잡한 파이프라인                                             │
│                                                                 │
│  DETR 혁신:                                                      │
│  - End-to-end 학습                                               │
│  - Object Query로 직접 객체 예측                                  │
│  - Hungarian Matching으로 학습                                   │
│  - NMS 불필요                                                    │
│                                                                 │
│  ┌────────────┐    ┌────────────┐    ┌────────────────────────┐ │
│  │ Backbone   │ → │ Transformer│ → │ FFN Heads             │ │
│  │ (ResNet)   │    │ Encoder/   │    │ (class + box)         │ │
│  │            │    │ Decoder    │    │                        │ │
│  └────────────┘    └────────────┘    └────────────────────────┘ │
│       ↓                 ↓                      ↓                │
│  Feature Map    Object Queries (100개)    100개 예측 출력       │
│  + Positional   ↓                                               │
│    Encoding     Self-attention + Cross-attention                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 DETR 구현

```python
import torch
import torch.nn as nn
from torchvision.models import resnet50
import torch.nn.functional as F

class DETR(nn.Module):
    """
    간략화된 DETR 구현
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

        # Object Queries (학습되는 임베딩)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # 출력 헤드
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
            x: (B, 3, H, W) 입력 이미지

        Returns:
            class_logits: (B, num_queries, num_classes+1)
            bbox_pred: (B, num_queries, 4)
        """
        B = x.shape[0]

        # Backbone feature 추출
        features = self.backbone(x)  # (B, 2048, H/32, W/32)
        features = self.conv(features)  # (B, 256, H/32, W/32)

        _, _, H, W = features.shape

        # Positional Encoding 생성
        pos_embed = self._get_positional_encoding(H, W, features.device)

        # Flatten for Transformer
        src = features.flatten(2).permute(0, 2, 1)  # (B, H*W, 256)
        src = src + pos_embed.flatten(0, 1).unsqueeze(0).expand(B, -1, -1)

        # Object Queries
        query_embed = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        # Transformer
        tgt = torch.zeros_like(query_embed)
        hs = self.transformer(src, tgt + query_embed)  # (B, num_queries, 256)

        # 예측
        class_logits = self.class_head(hs)
        bbox_pred = self.bbox_head(hs)

        return class_logits, bbox_pred

    def _get_positional_encoding(self, H, W, device):
        """2D Positional Encoding 생성"""
        i = torch.arange(W, device=device)
        j = torch.arange(H, device=device)

        x_embed = self.col_embed(i)  # (W, 128)
        y_embed = self.row_embed(j)  # (H, 128)

        pos = torch.cat([
            x_embed.unsqueeze(0).expand(H, -1, -1),
            y_embed.unsqueeze(1).expand(-1, W, -1),
        ], dim=-1)  # (H, W, 256)

        return pos


# Hungarian Matching Loss (간략화)
class HungarianMatcher:
    """
    예측과 GT를 최적으로 매칭

    Cost = λ_cls * L_cls + λ_box * L_box + λ_giou * L_giou
    """

    def __init__(self, cost_class=1, cost_bbox=5, cost_giou=2):
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    def __call__(self, outputs, targets):
        """
        scipy.optimize.linear_sum_assignment 사용하여
        이분 매칭 수행
        """
        # 구현 생략 (scipy 사용)
        pass
```

### 4.3 RT-DETR (Real-Time DETR)

```python
from ultralytics import RTDETR

# RT-DETR 사용 (Ultralytics)
model = RTDETR("rtdetr-l.pt")

# 추론
results = model("image.jpg")

# 학습
model.train(data="coco.yaml", epochs=100)

"""
RT-DETR 특징:
- DETR의 end-to-end 장점 유지
- 실시간 추론 가능 (YOLO 수준 속도)
- Efficient Hybrid Encoder
- IoU-aware Query Selection
"""
```

---

## 5. 인스턴스 분할 (Instance Segmentation)

### 5.1 Mask R-CNN

```python
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def create_mask_rcnn(num_classes: int):
    """
    커스텀 Mask R-CNN 생성

    Mask R-CNN = Faster R-CNN + Mask Head
    """
    model = maskrcnn_resnet50_fpn_v2(weights="DEFAULT")

    # Box predictor 교체
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Mask predictor 교체
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model


def inference_mask_rcnn(model, image, threshold=0.5):
    """Mask R-CNN 추론"""

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

    # Hard mask로 변환
    result["masks"] = (result["masks"] > 0.5).squeeze(1)  # (N, H, W)

    return result


# YOLOv8-seg 사용
from ultralytics import YOLO

seg_model = YOLO("yolov8n-seg.pt")
results = seg_model("image.jpg")

# 결과에서 마스크 추출
for result in results:
    if result.masks is not None:
        masks = result.masks.data  # (N, H, W)
```

### 5.2 SAM (Segment Anything Model)

```python
from segment_anything import sam_model_registry, SamPredictor
import numpy as np

# SAM 모델 로드
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
predictor = SamPredictor(sam)

# 이미지 설정
predictor.set_image(image)  # (H, W, 3) numpy array

# Point prompt로 분할
input_point = np.array([[500, 375]])  # 클릭 좌표
input_label = np.array([1])  # 1 = foreground

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,  # 3개의 마스크 후보 반환
)

# Box prompt로 분할
input_box = np.array([100, 100, 400, 400])  # x1, y1, x2, y2

masks, scores, logits = predictor.predict(
    box=input_box,
    multimask_output=False,
)

"""
SAM 특징:
- Promptable segmentation (point, box, text)
- Zero-shot generalization
- 매우 높은 분할 품질
- 대형 모델로 느린 속도 → MobileSAM, FastSAM 등 경량화 버전 존재
"""
```

---

## 6. 실전 팁

### 6.1 데이터셋 형식

```python
"""
주요 데이터셋 형식:

1. COCO Format:
   - annotations.json에 모든 어노테이션 저장
   - 이미지와 어노테이션 분리

2. YOLO Format:
   - 각 이미지마다 .txt 파일
   - class x_center y_center width height (정규화)

3. Pascal VOC Format:
   - XML 파일로 각 이미지 어노테이션
"""

# YOLO format 예시 (labels/train/image001.txt)
"""
0 0.5 0.5 0.2 0.3
1 0.3 0.7 0.1 0.15
"""

# COCO to YOLO 변환
def coco_to_yolo(coco_box, img_width, img_height):
    """
    COCO: [x_min, y_min, width, height]
    YOLO: [x_center, y_center, width, height] (정규화)
    """
    x, y, w, h = coco_box

    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height

    return [x_center, y_center, w_norm, h_norm]
```

### 6.2 학습 팁

```python
"""
객체 탐지 학습 체크리스트:

1. 데이터 품질
   - 라벨 정확도 확인
   - 클래스 불균형 처리 (Focal Loss, 오버샘플링)
   - 적절한 증강 사용

2. 하이퍼파라미터
   - 학습률: 1e-4 ~ 1e-3 (사전학습 시작)
   - 배치 크기: GPU 메모리에 맞춰 최대한 크게
   - 이미지 크기: 모델 기본값 사용 (YOLO: 640)

3. 증강 전략
   - Mosaic: 4개 이미지 합성 (YOLO)
   - MixUp: 이미지 블렌딩
   - 기본: Flip, Scale, Color Jitter

4. 모델 선택
   - 실시간: YOLO (YOLOv8n, YOLOv8s)
   - 정확도: Faster R-CNN, DETR
   - 분할: YOLOv8-seg, Mask R-CNN
"""

# Ultralytics 학습 예시
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,

    # 증강
    mosaic=1.0,      # Mosaic 확률
    mixup=0.0,       # MixUp 확률
    hsv_h=0.015,     # Hue 증강
    hsv_s=0.7,       # Saturation 증강
    hsv_v=0.4,       # Value 증강
    degrees=0.0,     # 회전
    translate=0.1,   # 이동
    scale=0.5,       # 스케일
    fliplr=0.5,      # 좌우 반전

    # 정규화
    weight_decay=0.0005,

    # 학습 스케줄
    warmup_epochs=3,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    lr0=0.01,        # 초기 학습률
    lrf=0.01,        # 최종 학습률 비율
)
```

---

## 정리

### 탐지기 선택 가이드

| 요구사항 | 추천 모델 |
|----------|----------|
| 실시간 (30+ FPS) | YOLOv8n/s |
| 높은 정확도 | YOLOv8x, Faster R-CNN |
| 작은 객체 | YOLO + SAHI, RetinaNet |
| 인스턴스 분할 | YOLOv8-seg, Mask R-CNN |
| End-to-end | DETR, RT-DETR |
| Zero-shot | Grounding DINO, SAM |

### 핵심 개념 요약

| 개념 | 설명 |
|------|------|
| IoU | 박스 겹침 정도 (0~1) |
| mAP | 평균 정밀도 (정확도 지표) |
| NMS | 중복 박스 제거 후처리 |
| Anchor | 사전 정의된 기준 박스 |
| FPN | 다중 스케일 특징 추출 |
| GIoU/CIoU | 개선된 IoU 손실 함수 |

### 다음 단계
- Computer Vision 토픽에서 시맨틱 분할(Semantic Segmentation)을 학습합니다.
- [Computer_Vision/19_DNN_Module.md](../../Computer_Vision/19_DNN_Module.md): OpenCV DNN

---

## 연습 문제

### 연습 1: IoU 이해 및 GIoU 구현

1. 제공된 `calculate_iou` 함수를 사용하여 다음 박스 쌍에 대한 IoU를 계산하세요: `(a) [0,0,100,100] vs [50,50,150,150]`, `(b) [0,0,100,100] vs [200,200,300,300]`, `(c) [0,0,100,100] vs [0,0,100,100]`. 결과가 직관과 일치하는지 확인하세요.
2. GIoU(Generalized IoU)를 구현하세요: `GIoU = IoU - (area_enclosing - area_union) / area_enclosing`. 여기서 enclosing box는 예측 박스와 정답(ground-truth) 박스를 모두 포함하는 가장 작은 박스입니다.
3. 겹치지 않는 쌍 `(b)`에 대해, GIoU가 표준 IoU보다 더 유익한 기울기(gradient) 신호를 제공함을 보여주세요. 이것이 학습에 왜 중요한지 설명하세요.

### 연습 2: 직접 찍은 이미지에 YOLOv8 추론 실행

Ultralytics 라이브러리를 사용하여:
1. `yolov8n.pt` (nano 모델)를 불러와 여러 물체가 포함된 사진에서 추론을 실행하세요.
2. 탐지된 물체명, 신뢰도 점수, 바운딩 박스 좌표를 출력하세요.
3. `results[0].save("my_result.jpg")`를 사용하여 주석이 달린 결과 이미지를 저장하세요.
4. 이제 같은 이미지를 `yolov8x.pt` (extra-large 모델)로 실행하세요. nano와 extra-large 간의 탐지 수와 신뢰도 점수를 비교하세요. 이것이 어떤 트레이드오프를 보여주나요?

### 연습 3: 사용자 정의 데이터셋에서 Faster R-CNN 미세조정

`torchvision`의 Faster R-CNN을 사용하여:
1. 최소 2개의 클래스를 갖는 소규모 사용자 정의 데이터셋을 만드세요 (COCO의 일부를 사용하거나 랜덤 사각형으로 합성 데이터셋을 만들 수 있습니다).
2. Faster R-CNN 형식으로 이미지와 타겟을 반환하는 `torch.utils.data.Dataset` 클래스를 구현하세요: `{"boxes": tensor, "labels": tensor}`.
3. `CustomFasterRCNN(num_classes=3)` (배경 + 2개 클래스)을 인스턴스화하고 5번의 학습 반복을 실행하세요.
4. `loss_dict`에 4개의 키(`loss_classifier`, `loss_box_reg`, `loss_objectness`, `loss_rpn_box_reg`)가 포함되어 있는지 확인하고 각 손실 구성 요소가 담당하는 것을 설명하세요.

### 연습 4: 앵커 생성(Anchor Generation) 및 IoU 매칭 분석

1. `generate_anchors`를 사용하여 `stride=16`으로 `(38, 50)` 특징 맵의 모든 앵커 박스를 생성하세요.
2. 빈 600×800 캔버스에서 특징 맵 원점 중심의 9개의 앵커 형태(3 스케일 × 3 비율)를 모두 플롯하세요.
3. 정답 박스 `[200, 150, 400, 350]`에 대해, 위치 `(h=9, w=12)` (근처 특징 맵 셀)에서 이 박스와 모든 9개의 앵커 형태 사이의 IoU를 계산하세요.
4. 어느 앵커가 가장 높은 IoU를 가지나요? 이것이 데이터셋에서 예상 물체 크기와 일치하는 앵커 크기를 선택하는 중요성에 대해 무엇을 알려주나요?

---

## 참고 자료

### 논문
- "Faster R-CNN" (Ren et al., 2015)
- "YOLO: You Only Look Once" (Redmon et al., 2016)
- "DETR: End-to-End Object Detection with Transformers" (Carion et al., 2020)
- "Segment Anything" (Kirillov et al., 2023)

### 코드 & 자료
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [torchvision detection](https://pytorch.org/vision/stable/models.html#object-detection)
- [COCO Dataset](https://cocodataset.org/)
- [Roboflow](https://roboflow.com/) - 데이터셋 관리
