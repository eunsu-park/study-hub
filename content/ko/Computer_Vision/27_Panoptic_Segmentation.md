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

        # 각 FPN 레벨을 1/4 해상도로 업샘플링하고 병합
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

        # 학습 가능한 객체 쿼리
        self.query_embed = nn.Embedding(n_queries, d_model)

        # 마스크 어텐션을 사용하는 Transformer 디코더 레이어
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=2048, dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, n_layers)

        # 예측 헤드
        self.class_head = nn.Linear(d_model, n_classes + 1)  # +1은 "객체 없음"
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

        # 쿼리 초기화
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        # Transformer 디코더
        output = self.decoder(queries, pixel_features)

        # 클래스 예측
        class_logits = self.class_head(output)  # (B, N, C+1)

        # 픽셀 특징과의 내적을 통한 마스크 예측
        mask_embed = self.mask_head(output)  # (B, N, d_model)
        mask_logits = torch.bmm(mask_embed, pixel_features.transpose(1, 2))

        return class_logits, mask_logits
```

---

## 5. Panoptic Quality (PQ)

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
            continue  # void 건너뛰기

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

    # 매칭되지 않은 정답을 FN으로 카운트
    for gt_id in np.unique(gt_segments):
        if gt_id != 0 and gt_id not in gt_matched:
            fn += 1

    # PQ 계산
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

    # 1. 인스턴스 마스크 (things) 배치 - 최우선 순위
    # 신뢰도 점수 기준 정렬 (높은 것 먼저)
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

        # 이미 배치된 인스턴스와의 겹침 확인
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

    # 2. 나머지 픽셀을 stuff 클래스로 채우기
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
