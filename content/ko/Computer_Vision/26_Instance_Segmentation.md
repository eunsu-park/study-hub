[이전: 시맨틱 세그멘테이션](./25_Semantic_Segmentation.md)

---

# 26. 인스턴스 세그멘테이션

## 학습 목표

이 수업을 완료하면 다음을 수행할 수 있습니다:

1. 인스턴스 세그멘테이션과 시맨틱 세그멘테이션의 차이 구분
2. 동시 탐지 및 세그멘테이션을 위한 Mask R-CNN 구현
3. 앵커 프리 인스턴스 세그멘테이션 방법 (YOLACT, SOLOv2) 구축
4. COCO 스타일 AP 지표로 마스크 품질 평가
5. 실제 카운팅 및 분석 작업에 인스턴스 세그멘테이션 적용

---

## 목차

1. [인스턴스 vs 시맨틱 세그멘테이션](#1-인스턴스-vs-시맨틱-세그멘테이션)
2. [Mask R-CNN 아키텍처](#2-mask-r-cnn-아키텍처)
3. [YOLACT: 실시간 인스턴스 세그멘테이션](#3-yolact-실시간-인스턴스-세그멘테이션)
4. [앵커 프리 방법 (SOLO, SOLOv2)](#4-앵커-프리-방법-solo-solov2)
5. [학습 및 데이터 준비](#5-학습-및-데이터-준비)
6. [평가 지표 (COCO AP)](#6-평가-지표-coco-ap)
7. [실용적 응용](#7-실용적-응용)
8. [연습문제](#8-연습문제)

---

## 1. 인스턴스 vs 시맨틱 세그멘테이션

### 1.1 주요 차이점

```
Semantic Segmentation:
  픽셀 레이블: "자동차", "사람", "도로"
  개별 인스턴스를 구분할 수 없음
  인접한 두 자동차 → 둘 다 "자동차"로 레이블링 (동일 클래스)

Instance Segmentation:
  픽셀 레이블: "자동차 #1", "자동차 #2", "사람 #1"
  각 객체 인스턴스가 고유한 마스크를 가짐
  인접한 두 자동차 → "자동차 #1"과 "자동차 #2" (분리!)

  Semantic          Instance
  ┌──────────┐     ┌──────────┐
  │ ██  ██   │     │ ██  ▓▓   │
  │ ██  ██   │     │ ██  ▓▓   │  ██ = 자동차 #1
  │          │     │          │  ▓▓ = 자동차 #2
  │  ████    │     │  ████    │  ██ = 사람 #1
  └──────────┘     └──────────┘
  모든 자동차 = 동일   각 자동차 = 고유
```

---

## 2. Mask R-CNN 아키텍처

### 2.1 Mask R-CNN 파이프라인

```
Mask R-CNN = Faster R-CNN + Mask 분기

  이미지 → 백본 (ResNet+FPN) → 특징 맵
                                       │
                              ┌────────┼────────┐
                              ▼        ▼        ▼
                            RPN     Box Head  Mask Head
                          (후보 영역) (클래스+박스) (인스턴스별 마스크)
                              │        │        │
                              ▼        ▼        ▼
                           영역들     클래스    이진 마스크
                                    + 바운딩박스 (인스턴스당 28×28)
```

### 2.2 Torchvision을 사용한 구현

```python
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_mask_rcnn(n_classes, pretrained=True):
    """사용자 정의 클래스 수를 가진 Mask R-CNN 구축."""
    model = maskrcnn_resnet50_fpn_v2(pretrained=pretrained)

    # Box predictor 교체
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes)

    # Mask predictor 교체
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, n_classes
    )

    return model


def predict_masks(model, image, threshold=0.5, device='cuda'):
    """인스턴스 세그멘테이션 추론 실행."""
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        prediction = model([image.to(device)])[0]

    # 신뢰도로 필터링
    keep = prediction['scores'] > threshold
    masks = prediction['masks'][keep]      # (N, 1, H, W) 이진 마스크
    boxes = prediction['boxes'][keep]       # (N, 4) 바운딩 박스
    labels = prediction['labels'][keep]     # (N,) 클래스 레이블
    scores = prediction['scores'][keep]     # (N,) 신뢰도 점수

    return masks, boxes, labels, scores


def visualize_instances(image, masks, boxes, labels, scores, class_names):
    """이미지에 인스턴스 마스크 오버레이."""
    import numpy as np
    import matplotlib.pyplot as plt

    img = image.permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)

    colors = plt.cm.Set3(np.linspace(0, 1, len(masks)))

    for i, (mask, box, label, score) in enumerate(
        zip(masks, boxes, labels, scores)
    ):
        # 투명도를 적용한 마스크 오버레이
        m = mask[0].cpu().numpy() > 0.5
        color_mask = np.zeros_like(img)
        color_mask[m] = colors[i][:3]
        ax.imshow(color_mask, alpha=0.4)

        # 바운딩 박스 그리기
        x1, y1, x2, y2 = box.cpu().numpy()
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                            fill=False, color=colors[i], linewidth=2)
        ax.add_patch(rect)

        # 레이블
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
RoIAlign: Mask R-CNN에서 Faster R-CNN 대비 핵심 개선점.

RoIPool의 문제점:
  RoI 경계의 양자화로 공간 정밀도 손실.
  바운딩 박스에는 괜찮지만, 픽셀 수준 마스크에는 좋지 않음.

RoIAlign 해결책:
  양자화 대신 이중 선형 보간 사용.
  각 빈에 대해 정확한 서브픽셀 위치 샘플링.

  RoIPool:  round(x) → 그리드에 맞춤 → 정렬되지 않은 특징
  RoIAlign: bilinear(x) → 부드러운 보간 → 픽셀 완벽 정렬
```

---

## 3. YOLACT: 실시간 인스턴스 세그멘테이션

### 3.1 YOLACT 개념

```
YOLACT: You Only Look At CoefficienTs

핵심 아이디어: 마스크 생성을 두 개의 병렬 작업으로 분리:
  1. K개의 프로토타입 마스크 생성 (전체 이미지, 클래스 무관)
  2. 인스턴스별 K개의 계수 예측

  최종 마스크 = 계수로 가중된 프로토타입의 선형 결합

  인스턴스 mask_i = σ(Σ_k c_ik × prototype_k)

  Mask R-CNN보다 훨씬 빠른 이유:
  - 프로토타입은 전체 이미지에 대해 한 번만 계산
  - 마스크에 대한 인스턴스별 RoI 연산 불필요

  속도: 단일 GPU에서 ~30 FPS (Mask R-CNN의 ~5 FPS 대비)
```

### 3.2 YOLACT 아키텍처 개요

```python
class YOLACTHead(torch.nn.Module):
    """간소화된 YOLACT 예측 헤드."""

    def __init__(self, in_channels, n_classes, n_prototypes=32):
        super().__init__()
        self.n_prototypes = n_prototypes

        # 분류 헤드
        self.cls_head = torch.nn.Conv2d(in_channels, n_classes, 3, padding=1)

        # 박스 회귀 헤드
        self.box_head = torch.nn.Conv2d(in_channels, 4, 3, padding=1)

        # 마스크 계수 헤드
        self.coeff_head = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, n_prototypes, 3, padding=1),
            torch.nn.Tanh(),  # 계수 범위 [-1, 1]
        )

    def forward(self, features):
        cls = self.cls_head(features)
        box = self.box_head(features)
        coeffs = self.coeff_head(features)
        return cls, box, coeffs
```

---

## 4. 앵커 프리 방법 (SOLO, SOLOv2)

### 4.1 SOLO: 위치 기반 객체 세그멘테이션

```
SOLO는 앵커와 바운딩 박스를 완전히 제거:

이미지를 S×S 그리드 셀로 분할.
각 셀이 예측:
  - 카테고리 (어떤 클래스?)
  - 인스턴스 마스크 (해당 인스턴스의 이진 마스크)

객체 중심이 셀 (i,j)에 있으면:
  셀 (i,j)가 해당 객체의 마스크를 담당.

SOLOv2 개선:
  인스턴스별 마스크 커널 가중치를 동적으로 생성.
  커널 × 특징 = 인스턴스 마스크
  전체 마스크를 예측하는 것보다 훨씬 효율적.
```

---

## 5. 학습 및 데이터 준비

### 5.1 COCO 형식 데이터셋

```python
# 인스턴스 세그멘테이션을 위한 COCO 어노테이션 형식:
# {
#   "images": [{"id": 1, "file_name": "img.jpg", "width": 640, "height": 480}],
#   "annotations": [{
#     "id": 1,
#     "image_id": 1,
#     "category_id": 1,
#     "segmentation": [[x1,y1,x2,y2,...,xn,yn]],  # 폴리곤 포인트
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
    """COCO 형식 인스턴스 세그멘테이션 데이터셋."""

    def __init__(self, root, annotation_file, transforms=None):
        self.root = root
        self.coco = COCO(annotation_file)
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]

        # 이미지 로드
        img_path = f"{self.root}/{img_info['file_name']}"
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 어노테이션 로드
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        masks = []
        boxes = []
        labels = []

        for ann in anns:
            # 이진 마스크
            mask = self.coco.annToMask(ann)
            masks.append(mask)

            # 바운딩 박스 [x, y, w, h] → [x1, y1, x2, y2]
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

## 6. 평가 지표 (COCO AP)

### 6.1 마스크 AP 계산

```python
def compute_mask_iou(pred_mask, gt_mask):
    """두 이진 마스크 간의 IoU 계산."""
    intersection = (pred_mask & gt_mask).sum().float()
    union = (pred_mask | gt_mask).sum().float()
    return (intersection / (union + 1e-6)).item()


def compute_ap(precisions, recalls):
    """101-포인트 보간을 사용한 Average Precision 계산 (COCO 스타일)."""
    recall_points = np.linspace(0, 1, 101)
    ap = 0

    for r in recall_points:
        prec_at_r = max([p for p, rec in zip(precisions, recalls) if rec >= r],
                        default=0)
        ap += prec_at_r

    return ap / 101


# COCO 평가 사용:
# AP @ IoU=0.50:0.95 (10개 IoU 임계값에 대한 평균)
# AP@50 (IoU 임계값 = 0.50, VOC처럼)
# AP@75 (IoU 임계값 = 0.75, 더 엄격)
# AP_small, AP_medium, AP_large (객체 크기별)
```

---

## 7. 실용적 응용

### 7.1 객체 카운팅 및 분석

```python
def count_and_measure_instances(model, image, class_names, device='cuda'):
    """인스턴스를 카운트하고 속성을 측정."""
    masks, boxes, labels, scores = predict_masks(model, image, device=device)

    results = {}
    for label_id in labels.unique():
        class_mask = labels == label_id
        class_name = class_names[label_id.item()]
        n_instances = class_mask.sum().item()

        # 각 인스턴스의 면적 측정
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

## 8. 연습문제

### 연습문제 1: Mask R-CNN 파인튜닝

커스텀 데이터셋에서 Mask R-CNN 파인튜닝:
1. 폴리곤 어노테이션이 있는 데이터셋 준비 (LabelMe 또는 CVAT 사용)
2. 사전 학습된 Mask R-CNN을 데이터셋에 파인튜닝
3. COCO 스타일 마스크 AP로 평가
4. 예측 시각화: 정답과 비교
5. 실패 사례 분석: 모델이 어디에서 잘못되는가?

### 연습문제 2: 실시간 인스턴스 세그멘테이션

실시간 인스턴스 세그멘테이션 시스템 구축:
1. 동일 데이터셋에서 Mask R-CNN vs YOLACT 비교
2. 각 모델의 GPU에서 FPS 측정
3. 웹캠 피드에 적용
4. 프레임 간 인스턴스 추적 구현 (간단한 IoU 기반)
5. 표시: 각 클래스의 개수, 실시간 색상 마스크

### 연습문제 3: 인스턴스 카운팅 애플리케이션

객체 카운팅 애플리케이션 구축:
1. 카운팅 데이터셋 (예: 세포, 과일)에서 인스턴스 세그멘테이션 모델 학습
2. 마스크 후처리: 겹치는 인스턴스 처리
3. 신뢰도 임계값을 사용한 카운팅 구현
4. 평가: 카운팅 정확도 vs 탐지 임계값
5. 엣지 케이스 처리: 가려진 객체, 밀집된 객체

### 연습문제 4: 마스크 품질 분석

방법 간 마스크 품질 분석:
1. 동일 데이터셋에서 Mask R-CNN과 YOLACT 학습
2. 마스크 품질 비교: AP@50, AP@75, AP@50:95
3. 클래스별 분석: 어떤 클래스의 마스크가 더 좋거나 나쁜가?
4. 경계 품질 시각화: 마스크 가장자리 확대
5. 측정: 계산 시간 vs 마스크 품질 트레이드오프

### 연습문제 5: 더 나은 마스크를 위한 커스텀 손실

마스크 예측을 위한 손실 함수 실험:
1. 이진 교차 엔트로피로 학습 (기본 Mask R-CNN)
2. Dice 손실로 학습
3. 경계 인식 손실로 학습 (마스크 가장자리 강조)
4. 마스크 품질 비교, 특히 경계에서
5. 손실 결합 및 최적 가중치 찾기

---

*26강 끝*
