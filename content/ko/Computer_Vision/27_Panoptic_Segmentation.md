[이전: 인스턴스 세그멘테이션](./26_Instance_Segmentation.md)

---

# 27. 파놉틱 세그멘테이션

## 학습 목표

이 수업을 완료하면 다음을 수행할 수 있습니다:

1. 시맨틱과 인스턴스 세그멘테이션의 통합으로서의 파놉틱 세그멘테이션 설명
2. "stuff" (비정형 영역)와 "things" (셀 수 있는 객체) 구분
3. 시맨틱 및 인스턴스 헤드와 FPN을 결합한 Panoptic FPN 구현
4. 모든 세그멘테이션 작업을 위한 통합 프레임워크인 Mask2Former 구축
5. Panoptic Quality (PQ)와 SQ, RQ로의 분해 평가

---

## 목차

1. [파놉틱 세그멘테이션 개요](#1-파놉틱-세그멘테이션-개요)
2. [Stuff vs Things](#2-stuff-vs-things)
3. [Panoptic FPN](#3-panoptic-fpn)
4. [Mask2Former](#4-mask2former)
5. [Panoptic Quality (PQ)](#5-panoptic-quality-pq)
6. [후처리 및 융합](#6-후처리-및-융합)
7. [실습 파이프라인](#7-실습-파이프라인)
8. [연습문제](#8-연습문제)

---

## 1. 파놉틱 세그멘테이션 개요

### 이론: 통합 출력

파놉틱 출력: 모든 픽셀 `(u, v)`가 `(class, instance_id)` 쌍을 얻음:

- **Stuff 픽셀**(범주로서의 하늘, 도로, 건물): `class = <stuff 클래스>, instance_id = 0` (stuff 내 인스턴스 분리 없음).
- **Thing 픽셀**(셀 수 있는 객체로서의 사람, 자동차): `class = <thing 클래스>, instance_id = <객체당 고유 ID>`.

이것이 가장 완전한 장면 파싱: 모든 픽셀이 할당됨, 각 객체가 분리됨, 배경 stuff가 적절히 분류됨. 후단 응용(자율 주행, AR, 이미지 편집)은 이 완전한 파싱이 필요 — "이 모든 픽셀은 도로"와 "저 픽셀은 자동차 1, 저 픽셀은 자동차 2"를 동시에 아는 것.

### 1.1 통합 관점

```
Panoptic = 모든 픽셀에 대한 Semantic + Instance

Semantic Segmentation:     모든 픽셀에 레이블 부여 (stuff + things)
Instance Segmentation:     개별 객체 세그멘테이션 (things만)
Panoptic Segmentation:     둘 다! 모든 픽셀이 클래스와 인스턴스 ID를 가짐

출력 형식: 각 픽셀 (i, j)에 대해:
  - class_id: 이 픽셀이 어떤 클래스에 속하는가?
  - instance_id: 어떤 특정 인스턴스인가 ("things" 클래스의 경우)?

예시 장면:
  하늘 (stuff, 인스턴스 없음)
  도로 (stuff, 인스턴스 없음)
  자동차 #1 (thing, 인스턴스 1)
  자동차 #2 (thing, 인스턴스 2)
  사람 #1 (thing, 인스턴스 3)
  나무 (stuff, 인스턴스 없음)
```

---

## 2. Stuff vs Things

### 이론: Things vs Stuff: 존재론

Things/stuff 구분은 임의가 아님; 인간이 장면 요소를 개념화하는 방식을 반영:

- **Things**는 경계가 있는 **셀 수 있는** 객체: 자동차, 사람, 의자, 개. "의자 3개"가 말이 됨. 각각 뚜렷한 모양을 가지고 개별적으로 세거나 추적할 수 있음.
- **Stuff**는 **비정형**: 하늘, 도로, 잔디, 물. "하늘 3개"는 말이 안 됨. Stuff는 셀 수 있는 인스턴스를 가지지 않음; 범위와 커버리지를 가짐.

이 구분이 유용한 이유는 두 종류 내용이 **다른 평가**를 요구하기 때문: things에 대해서는 인스턴스 분리에 신경 씀(각 객체가 찾아졌나?); stuff에 대해서는 단지 공간 커버리지에 신경 씀(도로의 얼마나 많은 부분이 도로로 레이블됐나?). PQ 메트릭(§5에서 다룸)이 두 경우 모두 균일하게 처리.

존재론은 데이터셋에 고정됨 — COCO Panoptic은 80 thing 클래스와 53 stuff 클래스, Cityscapes는 8 things와 11 stuff, ADE20K는 100 things와 50 stuff.

### 2.1 분류

```
"Things": 잘 정의된 형태를 가진 셀 수 있는 객체
  - 사람, 자동차, 동물, 가구
  - 인스턴스가 있음: 자동차 #1, 자동차 #2
  - 인스턴스 세그멘테이션 필요

"Stuff": 명확한 경계가 없는 비정형 영역
  - 하늘, 도로, 잔디, 물, 벽
  - 개별 인스턴스 없음
  - 시맨틱 세그멘테이션만 필요

COCO Panoptic: 80개 thing 클래스 + 53개 stuff 클래스 = 총 133개
Cityscapes: 8개 thing 클래스 + 11개 stuff 클래스 = 총 19개
```

---

## 3. Panoptic FPN

### 이론: Panoptic FPN: 두 헤드, 하나의 백본

Panoptic FPN(Kirillov 등, 2019)이 첫 효과적 파놉틱 아키텍처. 아이디어: 같은 Feature Pyramid Network 백본 위에 **시맨틱 세그멘테이션 헤드와 인스턴스 세그멘테이션 헤드 둘 다** 실행, 그 다음 출력을 최종 파놉틱 맵으로 **융합**.

아키텍처:

1. **공유 백본 + FPN**: 다중 스케일 특징 추출.
2. **시맨틱 헤드**: stuff 클래스 + thing 클래스 점유(어느 픽셀이 "사람"인지 말함, 어느 사람인지는 몰라도)의 밀집 예측.
3. **인스턴스 헤드**(Mask R-CNN 스타일): things 검출, 인스턴스별 마스크 예측.
4. **융합**: 두 출력을 결합. 각 픽셀에 대해:
   - 신뢰 있는 인스턴스 마스크 내부면, 그 인스턴스의 클래스와 ID 할당.
   - 그렇지 않으면, 시맨틱 헤드의 예측 사용(stuff 클래스 또는 "할당 안 됨 thing").

융합 단계는 놀랍게도 사소하지 않음 — 두 헤드가 불일치할 수 있음(인스턴스 마스크가 픽셀에서 "사람"이라고 말하는데, 시맨틱 헤드가 "벽"이라고 말함). 표준 해결: 인스턴스 예측이 things에 대해 시맨틱 예측을 무시, 하지만 신뢰 임계값 이상일 때만.

### 3.1 아키텍처

```
Panoptic FPN: Mask R-CNN의 FPN에 시맨틱 세그멘테이션 헤드 추가.

                    ┌─────────────────────────┐
  이미지 → 백본  → FPN (다중 스케일 특징)       │
                    └──────────┬──────────────┘
                               │
                    ┌──────────┼──────────────┐
                    ▼          ▼              ▼
              Semantic Head  RPN + Box Head  Mask Head
              (stuff 레이블) (thing 탐지)    (thing 마스크)
                    │          │              │
                    ▼          ▼              ▼
              Stuff 세그먼트  Thing 박스     Thing 마스크
                    │          └──────┬──────┘
                    │                 ▼
                    │          인스턴스 세그먼트
                    └────────┬────────┘
                             ▼
                    Panoptic 융합
                    (stuff + things 병합)
                             │
                             ▼
                    Panoptic 출력
```

### 3.2 Semantic Head

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class PanopticFPNSemanticHead(nn.Module):
    """Panoptic FPN을 위한 시맨틱 세그멘테이션 헤드."""

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
            fpn_features: FPN의 {level: tensor} 딕셔너리
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

### 이론: Mask2Former: 모든 것을 위한 하나의 아키텍처

Mask2Former(Cheng 등, 2022)는 **같은 transformer 아키텍처가 최소한의 변경만으로 시맨틱, 인스턴스, 파놉틱 세그멘테이션을 할 수 있음**을 보였습니다. 주요 관찰:

- 어떠한 세그멘테이션 작업도 **이진 마스크 집합 + 그 클래스 레이블 예측**으로 프레이밍 가능.
- 시맨틱 세그멘테이션: 클래스당 마스크 하나(인스턴스 분리 없음).
- 인스턴스 세그멘테이션: thing 인스턴스당 마스크 하나.
- 파놉틱 세그멘테이션: stuff 클래스당 마스크 하나 + thing 인스턴스당 마스크 하나.

아키텍처:

1. **Pixel 디코더**: 백본에서 다중 스케일 특징 추출(Swin Transformer가 흔함).
2. **Transformer 디코더**: `N` 쿼리(예: 100) 생성, 각각이 masked attention을 통해 이미지 특징에 주의.
3. **쿼리별 예측**: 각 쿼리가 클래스 확률과 이진 마스크 출력(픽셀별 임베딩과의 내적을 통해).
4. **훈련**: 쿼리와 ground-truth 인스턴스 간 이분 매칭(Hungarian 알고리즘), 마스크별 분류 손실 + 마스크별 이진 cross-entropy + dice loss.

파놉틱의 경우, 단지 "stuff" 클래스를 "thing" 인스턴스와 함께 유효 예측으로 포함 — 다른 것은 변하지 않음. 이 통합이 현재 최고 수준 방향.

### 4.1 Mask2Former 아키텍처

```
Mask2Former: 모든 세그멘테이션 작업을 위한 하나의 아키텍처.

핵심 혁신: Transformer 디코더에서의 마스크 어텐션.
  모든 픽셀에 어텐션하는 대신, 예측된 마스크 영역에만 어텐션.
  이로써 중요한 곳에 계산을 집중.

아키텍처:
  이미지 → 백본 → Pixel Decoder → 다중 스케일 특징
                                          │
  학습 가능한 쿼리 → Transformer ← ──────┘
  (N개의 쿼리)       Decoder (마스크 교차 어텐션)
                          │
                   ┌──────┼──────┐
                   ▼      ▼      ▼
               클래스    마스크   Stuff/Thing
               예측     예측     할당
```

### 4.2 간소화된 Mask2Former

```python
class Mask2FormerDecoder(nn.Module):
    """간소화된 Mask2Former 디코더."""

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
            pixel_features: (B, HW, d_model) pixel decoder에서의 출력
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

### 이론: Panoptic Quality (PQ)

정식 파놉틱 메트릭. 각 클래스 `c`에 대해:

```
TP_c = IoU > 0.5이고 같은 클래스인 매칭된 (예측, ground-truth) 쌍 집합
FP_c = 매칭 안 된 예측
FN_c = 매칭 안 된 ground-truth 인스턴스

PQ_c = (Σ IoU over TP_c) / (|TP_c| + 0.5·|FP_c| + 0.5·|FN_c|)
```

그 다음 `PQ = mean(PQ_c over all classes)`.

#### SQ × RQ 분해

PQ는 분해 가능:

```
PQ = SQ × RQ

SQ = (Σ IoU over TP) / |TP|          Segmentation Quality: 매칭된 마스크가 얼마나 잘 정렬됐나?
RQ = |TP| / (|TP| + 0.5·FP + 0.5·FN)  Recognition Quality: 객체의 어느 비율이 올바로 찾아졌나?
```

이 분해는 진단에 유용:

- 낮은 **RQ**에 높은 **SQ**: 시스템이 일부 객체를 완벽히 찾지만 많이 놓침. Recall 개선.
- 높은 **RQ**에 낮은 **SQ**: 시스템이 올바른 객체를 찾지만 마스크가 헐거움. Precision 개선.

Things의 경우, RQ는 인스턴스 매칭에 대한 F1 같은 점수; stuff의 경우, 각 클래스가 단일 TP/FN(전체 클래스 마스크)을 기여하므로, RQ_for_stuff는 인스턴스가 얼마나 많은지보다 stuff 클래스가 예측에 존재하는지에 관한 것.

#### PQ가 단지 "시맨틱 mIoU + 인스턴스 AP"보다 나은 이유

순진한 파놉틱 메트릭은 `0.5·mIoU + 0.5·AP`. 하지만 불만족: mIoU는 "모든 곳 도로"를 예측하는 모델을 보상하는 반면, AP는 마스크가 헐거워도 많은 검출을 찾는 것을 보상. PQ는 대신 모든 것에 같은 임계값(IoU > 0.5)을 부과, 따라서 예측이 올바로 국소화되고 올바로 분류되어야 카운트. 직관적으로 동작하는 단일 일관된 숫자 생성.

### 5.1 PQ 지표

```python
import numpy as np


def panoptic_quality(pred_segments, gt_segments, pred_labels, gt_labels,
                     iou_threshold=0.5):
    """
    Panoptic Quality (PQ) 계산.

    PQ = (Σ IoU(p,g)) / (|TP| + 0.5|FP| + 0.5|FN|)
       = SQ × RQ

    SQ (Segmentation Quality) = 매칭된 세그먼트의 평균 IoU
    RQ (Recognition Quality) = 세그먼트 매칭의 F1 점수

    높을수록 좋음. 범위: [0, 1].
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

## 6. 후처리 및 융합

### 6.1 Panoptic 융합

```python
def panoptic_fusion(semantic_pred, instance_masks, instance_labels,
                    instance_scores, stuff_classes, thing_classes,
                    overlap_threshold=0.5, score_threshold=0.5):
    """
    시맨틱 (stuff)과 인스턴스 (things) 예측을
    단일 파놉틱 세그멘테이션 맵으로 병합.
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

## 7. 실습 파이프라인

### 7.1 완전한 학습 예제

```python
def train_panoptic_model(model, train_loader, val_loader, n_classes,
                          epochs=50, lr=1e-4, device='cuda'):
    """파놉틱 세그멘테이션 모델 학습."""
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

## 8. 연습문제

### 연습문제 1: Panoptic 융합 구현

파놉틱 융합 파이프라인 구축:
1. 사전 학습된 Mask R-CNN (things)과 DeepLab (stuff) 사용
2. 융합 알고리즘 구현 (우선순위 규칙으로 마스크 병합)
3. COCO val에서 Panoptic Quality (PQ)로 평가
4. 시각화: 범례가 있는 색상 코딩된 파놉틱 출력
5. 분석: PQ_things vs PQ_stuff 별도 분석

### 연습문제 2: Mask2Former 학습

파놉틱 세그멘테이션을 위한 Mask2Former 학습:
1. Detectron2의 Mask2Former 구현 사용
2. Cityscapes panoptic (19개 클래스)으로 학습
3. PQ, SQ, RQ 및 클래스별 분석 보고
4. 동일 데이터셋에서 Panoptic FPN과 비교
5. 분석: 어떤 클래스가 Mask2Former에서 가장 많은 이점을 얻는가?

### 연습문제 3: Stuff-Things 분석

Stuff vs Things 구분 분석:
1. 파놉틱 데이터셋에서 카테고리별 통계 계산
2. 이미지에서 stuff vs things의 비율은 (픽셀 면적)?
3. 가장 자주 혼동되는 stuff 클래스는?
4. 인스턴스 분리가 가장 나쁜 thing 클래스는?
5. 제안: 경계 품질을 어떻게 개선할 수 있는가?

### 연습문제 4: 비디오 파놉틱 세그멘테이션

파놉틱 세그멘테이션을 비디오로 확장:
1. 연속 프레임에서 파놉틱 세그멘테이션 실행
2. IoU 매칭을 사용하여 프레임 간 thing 인스턴스 추적
3. 동일 객체에 대한 일관된 ID 유지
4. 처리: 객체 출현, 사라짐, 가림
5. Stuff 예측의 시간적 일관성 측정

### 연습문제 5: 커스텀 Panoptic 데이터셋

파놉틱 데이터셋 생성 및 어노테이션:
1. 특정 도메인의 이미지 100장 수집 (예: 주방, 사무실)
2. Stuff 클래스 (벽, 바닥, 천장)와 Thing 클래스 (의자, 컵) 정의
3. COCO panoptic 형식으로 어노테이션
4. 커스텀 데이터셋에서 모델 학습
5. 도메인 특화 문제점 평가 및 분석

---

*27강 끝*
